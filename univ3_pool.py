#!/usr/bin/env python3
"""
Uniswap v3 Pool Poller  –  every 5 minutes (clock-aligned)

Queries per cycle (3 requests / 5 min):
  1. Pool state   → DD-MM-pool.csv
  2. Mints/Burns  → DD-MM-mints-burns.csv
  3. Ticks ±N     → DD-MM-ticks.csv

Run alongside orderbook.py as a separate process.
"""

import time, csv, os, requests
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "xxx"
SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
GRAPH_URL = (
    f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
)
POOL_ID   = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"  # USDC/USDT 0.01% fee
TICK_N    = 10    # query [current_tick - TICK_N, current_tick + TICK_N]
T0_DEC    = 6     # USDC decimals
T1_DEC    = 6     # USDT decimals
INTERVAL          = 300   # 5 minutes in seconds
DIR_POOL          = "D:/data/uniswap_DEX/liquidity_state"
DIR_MINTS_BURNS   = "D:/data/uniswap_DEX/mints_burns"
DIR_TICKS         = "D:/data/uniswap_DEX/liquidity_flow"
# ─────────────────────────────────────────────────────────────────────────────


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

def gql(query, variables=None):
    resp = requests.post(
        GRAPH_URL,
        json={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def sqrt_price_to_price(sqrt_price_x96_str):
    # price = (sqrtPriceX96 / 2^96)^2, adjusted for token decimals
    price = (int(sqrt_price_x96_str) / 2**96) ** 2
    price *= 10 ** (T0_DEC - T1_DEC)   # = 1 for USDC/USDT (same decimals)
    return price


def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def append_csv(directory, filename, row):
    os.makedirs(directory, exist_ok=True)
    path   = os.path.join(directory, filename)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ── Per-cycle queries ─────────────────────────────────────────────────────────

def query_pool(window_end_str):
    data  = gql(POOL_Q, {"id": POOL_ID})
    p     = data["pool"]
    price = sqrt_price_to_price(p["sqrtPrice"])
    tick  = int(p["tick"]) if p["tick"] is not None else None
    row   = {
        "window_end":    window_end_str,
        "liquidity":     p["liquidity"],
        "tick":          tick,
        "sqrt_price":    p["sqrtPrice"],
        "price":         round(price, 8),
        "token0_price":  p["token0Price"],
        "token1_price":  p["token1Price"],
        "tvl_token0":    p["totalValueLockedToken0"],
        "tvl_token1":    p["totalValueLockedToken1"],
        "tvl_usd":       p["totalValueLockedUSD"],
    }
    return row, tick


def query_mints_burns(window_end_str, since_ts):
    data  = gql(MINTS_BURNS_Q, {"pool": POOL_ID, "since": since_ts})
    mints = data["mints"]
    burns = data["burns"]

    def ssum(lst, key):
        return sum(safe_float(x[key]) for x in lst)

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
    lo   = current_tick - TICK_N
    hi   = current_tick + TICK_N
    data = gql(TICKS_Q, {"pool": POOL_ID, "lo": str(lo), "hi": str(hi)})
    ticks = data["ticks"]

    total_gross  = sum(int(t["liquidityGross"]) for t in ticks)
    # liquidityNet > 0 for ticks above current: entering that tick going up adds liquidity
    # liquidityNet < 0 for ticks above current: exiting that tick going up removes liquidity
    net_above = sum(int(t["liquidityNet"]) for t in ticks if int(t["tickIdx"]) >  current_tick)
    net_below = sum(int(t["liquidityNet"]) for t in ticks if int(t["tickIdx"]) <= current_tick)

    row = {
        "window_end":     window_end_str,
        "current_tick":    current_tick,
        "tick_lo":         lo,
        "tick_hi":         hi,
        "n_ticks":         len(ticks),
        "total_liq_gross": total_gross,   # "liquidity near current price"
        "net_liq_above":   net_above,     # net liq change if price moves up through range
        "net_liq_below":   net_below,     # net liq change if price moves down through range
    }
    return row


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"Uniswap v3 poller | pool={POOL_ID} | tick_N={TICK_N} | interval={INTERVAL}s")

    # Align to next 5-min clock boundary
    now  = time.time()
    wait = INTERVAL - (now % INTERVAL)
    print(f"Waiting {wait:.0f}s for next {INTERVAL//60}-min boundary...")
    time.sleep(wait)

    # last_ts tracks the end of the previous window (for mints/burns "since")
    last_ts = int(time.time()) - INTERVAL

    while True:
        now          = time.time()
        window_end = int(now) - (int(now) % INTERVAL)   # floor to boundary
        window_end_str = datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        date_str      = datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%d-%m")

        print(f"\n[{window_end_str}] querying...")

        # 1. Pool state
        current_tick = None
        try:
            pool_row, current_tick = query_pool(window_end_str)
            append_csv(DIR_POOL, f"{date_str}-pool.csv", pool_row)
            print(f"  pool : tick={current_tick}  price={pool_row['price']}  tvl_usd={pool_row['tvl_usd']}")
        except Exception as e:
            print(f"  pool query error: {e}")

        # 2. Mints/Burns since last window
        try:
            mb_row = query_mints_burns(window_end_str, last_ts)
            append_csv(DIR_MINTS_BURNS, f"{date_str}-mints-burns.csv", mb_row)
            print(f"  mints={mb_row['n_mints']}  burns={mb_row['n_burns']}  net_liq={mb_row['net_liq_change']:.0f}")
        except Exception as e:
            print(f"  mints/burns query error: {e}")

        # 3. Ticks near current tick
        if current_tick is not None:
            try:
                tick_row = query_ticks(window_end_str, current_tick)
                append_csv(DIR_TICKS, f"{date_str}-ticks.csv", tick_row)
                print(f"  ticks: n={tick_row['n_ticks']}  total_gross={tick_row['total_liq_gross']}")
            except Exception as e:
                print(f"  ticks query error: {e}")

        last_ts = window_end

        # Sleep until next boundary
        now  = time.time()
        wait = INTERVAL - (now % INTERVAL)
        time.sleep(max(wait, 1.0))


if __name__ == "__main__":
    main()
