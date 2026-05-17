"""
This fetcher queries the DEX pool state every five minutes. 
Thus, aligned to the UTC clock, we get contemporaneous data on the underlying pool, liquidity, and liquidity distribution

Because the Graph does not allow for historical data collection, 
this code is meant to be run 24/7 during the data collection timeframe alongside the binance orderbook.
"""

import time, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import UNISWAP_GRAPH_URL, UNISWAP_POOL_ID, UNISWAP_TICK_NUMBER, INTERVAL, DEX_POOL, DEX_TICKS, DEX_MINTS_BURNS, UNISWAP_TOKEN0_DECIMAL_PLACES, UNISWAP_TOKEN1_DECIMAL_PLACES
from utilities import gql, append_csv, sqrt_price_to_price
from datetime import datetime, timezone

# ── GraphQL queries ───────────────────────────────────────────────────────────

POOL_Q = """
query Pool($id: ID!) {
  pool(id: $id) {
    liquidity
    tick
    sqrtPrice
    token0Price
    token1Price
    totalValueLockedToken0
    totalValueLockedToken1
    totalValueLockedUSD
  }
}"""

MINTS_BURNS_Q = """
query MintsBurns($pool: String!, $since: Int!) {
  mints(
    first: 1000
    where: { pool: $pool, timestamp_gte: $since }
    orderBy: timestamp
    orderDirection: asc
  ) {
    timestamp amount amount0 amount1 amountUSD tickLower tickUpper
  }
  burns(
    first: 1000
    where: { pool: $pool, timestamp_gte: $since }
    orderBy: timestamp
    orderDirection: asc
  ) {
    timestamp amount amount0 amount1 amountUSD tickLower tickUpper
  }
}"""

TICKS_Q = """
query Ticks($pool: Bytes!, $lo: BigInt!, $hi: BigInt!) {
  ticks(
    first: 1000
    where: { poolAddress: $pool, tickIdx_gte: $lo, tickIdx_lte: $hi }
    orderBy: tickIdx
    orderDirection: asc
  ) {
    tickIdx liquidityGross liquidityNet price0 price1
  }
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(v, default=0.0):                             # need that later for summing mints and burns
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

# ── Per-cycle queries ─────────────────────────────────────────────────────────

def query_pool(window_end_str):
    data  = gql(UNISWAP_GRAPH_URL, POOL_Q, {"id": UNISWAP_POOL_ID})
    pool     = data["pool"]
    price = sqrt_price_to_price(UNISWAP_TOKEN0_DECIMAL_PLACES, UNISWAP_TOKEN1_DECIMAL_PLACES, pool["sqrtPrice"])
    tick  = int(pool["tick"]) if pool["tick"] is not None else None
    row   = {
        "window_end":    window_end_str,
        "liquidity":     pool["liquidity"],
        "tick":          tick,
        "sqrt_price":    pool["sqrtPrice"],
        "price":         round(price, 8),
        "token0_price":  pool["token0Price"],
        "token1_price":  pool["token1Price"],
        "tvl_token0":    pool["totalValueLockedToken0"],
        "tvl_token1":    pool["totalValueLockedToken1"],
        "tvl_usd":       pool["totalValueLockedUSD"],
    }
    return row, tick


def query_mints_burns(window_end_str, since_ts):
    data  = gql(UNISWAP_GRAPH_URL, MINTS_BURNS_Q, {"pool": UNISWAP_POOL_ID, "since": since_ts})
    mints = data["mints"]
    burns = data["burns"]

    def ssum(list_to_be_summed, key):                                       # list is either mints or burns
        return sum(safe_float(x[key]) for x in list_to_be_summed)

    row = {
        "window_end":     window_end_str,
        "n_mints":        len(mints),
        "n_burns":        len(burns),
        "mint_liq":       ssum(mints, "amount"),
        "burn_liq":       ssum(burns, "amount"),
        "mint_amount0":   round(ssum(mints, "amount0"), 4),
        "mint_amount1":   round(ssum(mints, "amount1"), 4),
        "mint_usd":       round(ssum(mints, "amountUSD"), 2),
        "burn_amount0":   round(ssum(burns, "amount0"), 4),
        "burn_amount1":   round(ssum(burns, "amount1"), 4),
        "burn_usd":       round(ssum(burns, "amountUSD"), 2),
        "net_liq_change": ssum(mints, "amount") - ssum(burns, "amount"),
    }
    return row


def query_ticks(window_end_str, current_tick):
    lo   = current_tick - UNISWAP_TICK_NUMBER   # cannot write low instead of lo, because of query params
    hi   = current_tick + UNISWAP_TICK_NUMBER
    data = gql(UNISWAP_GRAPH_URL, TICKS_Q, {"pool": UNISWAP_POOL_ID, "lo": str(lo), "hi": str(hi)})
    ticks = data["ticks"]

    total_gross  = sum(int(t["liquidityGross"]) for t in ticks)
    # liquidityNet > 0 for ticks above current: entering that tick going up adds liquidity
    # liquidityNet < 0 for ticks above current: exiting that tick going up removes liquidity
    net_above = sum(int(t["liquidityNet"]) for t in ticks if int(t["tickIdx"]) >  current_tick)
    net_below = sum(int(t["liquidityNet"]) for t in ticks if int(t["tickIdx"]) <= current_tick)

    row = {
        "window_end":      window_end_str,
        "current_tick":    current_tick,
        "tick_lo":         lo,
        "tick_hi":         hi,
        "n_ticks":         len(ticks),
        "total_liq_gross": total_gross,         # "liquidity near current price"
        "net_liq_above":   net_above,           # net liq change if price moves up through range
        "net_liq_below":   net_below,           # net liq change if price moves down through range
    }
    return row


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"Uniswap v3 poller | pool={UNISWAP_POOL_ID} | tick_N={UNISWAP_TICK_NUMBER} | interval={INTERVAL}s")

    # Align to next 5-min clock boundary
    now  = time.time()
    wait = INTERVAL - (now % INTERVAL)
    print(f"Waiting {wait:.0f}s for next {INTERVAL//60}-min boundary...")  # 5min boundary, but in case I want to change
    time.sleep(wait)

    # since it runs after we align to the next 5-min it selects the since timestamp of the previous window end
    collection_start = int(now) - INTERVAL 

    while True:
        # window_end will be the collection time -> for mints/burns collects xx:x5 - xx:10, rest "screenshots" at xx:10
        now             = time.time()
        window_end      = int(now) - (int(now) % INTERVAL)                                                  # floor to boundary
        window_end_str  = datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") # for CSV
        date_str        = datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%d-%m")             # for filename

        print(f"\n[{window_end_str}] querying...")

        # 1. Pool state
        current_tick = None
        try:
            pool_row, current_tick = query_pool(window_end_str)
            append_csv(DEX_POOL, f"{date_str}-pool.csv", pool_row)
            print(f"  pool : tick={current_tick}  price={pool_row['price']}  tvl_usd={pool_row['tvl_usd']}")
        except Exception as e:
            print(f"  pool query error: {e}")

        # 2. Mints/Burns since last window
        try:
            mintburn_row = query_mints_burns(window_end_str, collection_start)
            append_csv(DEX_MINTS_BURNS, f"{date_str}-mints-burns.csv", mintburn_row)
            print(f"  mints={mintburn_row['n_mints']}  burns={mintburn_row['n_burns']}  net_liq={mintburn_row['net_liq_change']:.0f}")
        except Exception as e:
            print(f"  mints/burns query error: {e}")

        # 3. Ticks near current tick
        if current_tick is not None:
            try:
                tick_row = query_ticks(window_end_str, current_tick)
                append_csv(DEX_TICKS, f"{date_str}-ticks.csv", tick_row)
                print(f"  ticks: n={tick_row['n_ticks']}  total_gross={tick_row['total_liq_gross']}")
            except Exception as e:
                print(f"  ticks query error: {e}")

        collection_start = window_end            # such that the next mint/burn collection starts at xx:10

        # Sleep until next boundary
        now  = time.time()
        wait = INTERVAL - (now % INTERVAL)
        time.sleep(max(wait, 1.0))


if __name__ == "__main__":
    main()
