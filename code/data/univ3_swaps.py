#!/usr/bin/env python3
"""
Uniswap v3 Daily Swap Fetcher  +  5-min Kline Builder

Fetches ALL swaps for the previous UTC day (00:00:00 → 23:59:59) using
cursor-based pagination (avoids the 5000-row skip limit).  Saves:
  - SAVE_DIR/DD-MM-swaps-raw.csv   (one row per swap, raw)
  - SAVE_DIR/DD-MM-klines.csv      (5-min OHLCV + imbalance + large trades)

Run once per day (e.g. shortly after UTC midnight) to collect yesterday's data.

Requires: pip install requests
"""

import os, time, csv, requests
from datetime import datetime, timezone, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "xxx"
SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
GRAPH_URL = (
    f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
)
# GRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

POOL_ID         = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"  # USDC/USDT 0.01%
T0_DEC          = 6        # USDC decimals
T1_DEC          = 6        # USDT decimals
INTERVAL        = 300      # 5-min kline bucket (seconds)
LARGE_TRADE_USD = 100_000  # flag swaps above this as large trades
SAVE_DIR        = r"D:/data/uniswap_DEX/klines_dex"  # directory to save CSVs
# ─────────────────────────────────────────────────────────────────────────────


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
    price = (int(sqrt_price_x96_str) / 2**96) ** 2
    price *= 10 ** (T0_DEC - T1_DEC)
    return price


def save_path(filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    return os.path.join(SAVE_DIR, filename)


# ── Fetch all swaps with cursor pagination ────────────────────────────────────

def fetch_all_swaps(start_ts, end_ts):
    all_swaps = []
    cursor    = ""
    page      = 0

    while True:
        data  = gql(SWAPS_Q, {
            "pool":    POOL_ID,
            "startTs": start_ts,
            "endTs":   end_ts,
            "cursor":  cursor,
        })
        batch = data["swaps"]
        all_swaps.extend(batch)
        print(f"  page {page:>3}: {len(batch):>4} swaps  (total: {len(all_swaps)})")

        if len(batch) < 1000:
            break

        cursor = batch[-1]["id"]
        page  += 1
        time.sleep(0.25)   # gentle rate-limiting between pages

    return all_swaps


# ── Build 5-min klines from raw swaps ────────────────────────────────────────

def build_klines(swaps):
    if not swaps:
        return []

    # Sort chronologically (fetch order was by id, not time)
    swaps = sorted(swaps, key=lambda s: int(s["timestamp"]))

    buckets = {}
    for s in swaps:
        ts      = int(s["timestamp"])
        bucket  = ts - (ts % INTERVAL)        # floor to 5-min boundary
        price   = sqrt_price_to_price(s["sqrtPriceX96"])
        amt_usd = abs(float(s["amountUSD"]))
        amt0    = float(s["amount0"])

        if bucket not in buckets:
            buckets[bucket] = {
                "open": price, "high": price, "low": price, "close": price,
                "volume_usd":         0.0,
                "n_swaps":            0,
                "net_amount0":        0.0,   # positive = USDC sold into pool
                "abs_amount0":        0.0,
                "large_trades_count": 0,
                "large_trades_usd":   0.0,
            }

        b = buckets[bucket]
        b["high"]          = max(b["high"], price)
        b["low"]           = min(b["low"],  price)
        b["close"]         = price
        b["volume_usd"]   += amt_usd
        b["n_swaps"]      += 1
        b["net_amount0"]  += amt0
        b["abs_amount0"]  += abs(amt0)
        if amt_usd >= LARGE_TRADE_USD:
            b["large_trades_count"] += 1
            b["large_trades_usd"]   += amt_usd

    rows = []
    for bucket_ts in sorted(buckets):
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
    # Previous UTC day: 00:00:00 → 23:59:59
    today     = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    start_ts  = int(yesterday.timestamp())
    end_ts    = int(today.timestamp()) - 1      # 23:59:59 of yesterday
    date_str  = yesterday.strftime("%d-%m")

    print(f"Fetching swaps for {yesterday.strftime('%Y-%m-%d')} UTC")
    print(f"  window: {start_ts} → {end_ts}  (00:00:00 → 23:59:59)")

    swaps = fetch_all_swaps(start_ts, end_ts)
    print(f"Total swaps: {len(swaps)}")

    if not swaps:
        print("Nothing to save.")
        return

    # ── Save raw swaps ──────────────────────────────────────────────────────
    raw_file   = save_path(f"{date_str}-swaps-raw.csv")
    raw_fields = ["id", "timestamp", "amount0", "amount1", "amountUSD",
                  "sqrtPriceX96", "tick", "sender", "recipient"]
    with open(raw_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for s in swaps:
            w.writerow({k: s[k] for k in raw_fields})
    print(f"Raw swaps  → {raw_file}")

    # ── Build and save 5-min klines ─────────────────────────────────────────
    klines      = build_klines(swaps)
    klines_file = save_path(f"{date_str}-klines.csv")
    with open(klines_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=klines[0].keys())
        w.writeheader()
        w.writerows(klines)
    print(f"Klines     → {klines_file}  ({len(klines)} rows)")


if __name__ == "__main__":
    main()
