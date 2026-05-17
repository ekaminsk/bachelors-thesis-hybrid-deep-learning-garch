"""
This query fetches swaps from the Uniswap pool between a given start and end date. First, these swaps are saved as raw swaps into a CSV. Secondly, 5-min Klines are calculated from the swaps.
"""

import os, time, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import UNISWAP_GRAPH_URL, UNISWAP_POOL_ID, UNISWAP_TOKEN0_DECIMAL_PLACES, UNISWAP_TOKEN1_DECIMAL_PLACES, INTERVAL, UNISWAP_LARGE_TRADE_THRESHOLD, DEX_SWAPS, UNISWAP_START_DATE, UNISWAP_END_DATE
from utilities import gql, save_csv, sqrt_price_to_price
from datetime import datetime, timezone

# ── GraphQL query (cursor-paginated by id) ────────────────────────────────────

SWAPS_Q = """
query Swaps($pool: String!, $startTs: Int!, $endTs: Int!, $cursor: ID!) {
  swaps(
    first: 1000
    where: { pool: $pool, timestamp_gte: $startTs, timestamp_lt: $endTs, id_gt: $cursor }
    orderBy: id
    orderDirection: asc
  ) {
    id
    timestamp
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
    sender
    recipient
  }
}"""

# ── Fetch all swaps with cursor pagination ────────────────────────────────────

def fetch_all_swaps(start_ts, end_ts):
    all_swaps = []
    cursor    = ""
    page      = 0

    while True:
        data  = gql(UNISWAP_GRAPH_URL, SWAPS_Q, {
            "pool":    UNISWAP_POOL_ID,
            "startTs": start_ts,
            "endTs":   end_ts,
            "cursor":  cursor,
        })
        batch = data["swaps"]                                                               # extend all_swaps list with paginated 1000 entries
        all_swaps.extend(batch)
        print(f"  page {page:>3}: {len(batch):>4} swaps  (total: {len(all_swaps)})")

        if len(batch) < 1000:                                                               # if there is less than 1000 entries -> no more entries left -> all has been taken
            break

        cursor = batch[-1]["id"]
        page  += 1
        time.sleep(0.25)                                                                    # gentle rate-limiting between pages

    return all_swaps


# ── Build 5-min klines from raw swaps ────────────────────────────────────────

def build_klines(swaps):
    if not swaps:
        return []                                                                           # if no swaps fetched, empty the list

    swaps = sorted(swaps, key=lambda s: int(s["timestamp"]))                                # Sort chronologically (fetch order was by id, not time)

    buckets = {}                                                                            # how the candle is constructed
    for s in swaps:
        ts      = int(s["timestamp"])                                                       # each timestamp
        bucket  = ts - (ts % INTERVAL)                                                      # floor to the nearest 5min boundary 
        price   = sqrt_price_to_price(UNISWAP_TOKEN0_DECIMAL_PLACES, UNISWAP_TOKEN1_DECIMAL_PLACES, s["sqrtPriceX96"])          
        amt_usd = abs(float(s["amountUSD"]))                                                # because uniswap has negative numbers for one side of the trade (depending on sell / buy)
        amt0    = float(s["amount0"])

        if bucket not in buckets:                                                           # first swap initializes the candle through setting all prices to that candle
            buckets[bucket] = {
                "open": price, "high": price, "low": price, "close": price,
                "volume_usd":         0.0,                                            
                "n_swaps":            0,
                "net_amount0":        0.0,                                                  # positive = USDC sold into pool
                "abs_amount0":        0.0,
                "large_trades_count": 0,
                "large_trades_usd":   0.0,
            }

        b = buckets[bucket]                                                                 # short-hand; from here, iteratively adjust high, low, close based on incoming swaps
        b["high"]          = max(b["high"], price)
        b["low"]           = min(b["low"],  price)
        b["close"]         = price                                                          # as the last swap overwrites and next swap in next period does not count into this bucket
        b["volume_usd"]   += amt_usd
        b["n_swaps"]      += 1
        b["net_amount0"]  += amt0
        b["abs_amount0"]  += abs(amt0)
        if amt_usd >= UNISWAP_LARGE_TRADE_THRESHOLD:
            b["large_trades_count"] += 1
            b["large_trades_usd"]   += amt_usd

    rows = []
    for bucket_ts in sorted(buckets):                                                       # bucket is the timeframe xx:x0 - xx:x5 (i.e. five min.), add buckets to rows (to build CSV)
        b   = buckets[bucket_ts]
        tot = b["abs_amount0"]
        imbalance = b["net_amount0"] / tot if tot > 0 else 0.0

        rows.append({
            "window_start":       datetime.fromtimestamp(bucket_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "open":               round(b["open"],  8),
            "high":               round(b["high"],  8),
            "low":                round(b["low"],   8),
            "close":              round(b["close"], 8),
            "volume_usd":         round(b["volume_usd"], 2),
            "n_swaps":            b["n_swaps"],
            "imbalance":          round(imbalance, 6),
            "large_trades_count": b["large_trades_count"],
            "large_trades_usd":   round(b["large_trades_usd"], 2),
        })

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    start_dt = datetime.strptime(UNISWAP_START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    start_ts = int(start_dt.timestamp())
    end_dt   = datetime.strptime(UNISWAP_END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)  
    end_ts   = int(end_dt.timestamp())
    date_str = start_dt.strftime("%d-%m")

    print(f"Fetching swaps between {UNISWAP_START_DATE} UTC and {UNISWAP_END_DATE}")
    print(f"  window: {start_ts} → {end_ts}")          

    swaps = fetch_all_swaps(start_ts, end_ts)
    print(f"Total swaps: {len(swaps)}")

    if not swaps:
        print("Nothing to save.")
        return

    # ── Save raw swaps ──────────────────────────────────────────────────────
    filepath_raw = os.path.join(DEX_SWAPS, f"{date_str}-swaps_raw.csv")
    fields_raw = ["id", "timestamp", "amount0", "amount1", "amountUSD",
                  "sqrtPriceX96", "tick", "sender", "recipient"]
    rows_raw = [{key: s[key] for key in fields_raw} for s in swaps]
    save_csv(rows_raw, filepath_raw)

    # ── Build and save 5-min klines ─────────────────────────────────────────
    klines      = build_klines(swaps)
    filepath_kline = os.path.join(DEX_SWAPS, f"{date_str}-klines.csv")
    save_csv(klines, filepath_kline)


if __name__ == "__main__":
    main()
