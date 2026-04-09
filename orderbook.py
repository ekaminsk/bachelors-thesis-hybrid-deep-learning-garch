#!/usr/bin/env python3
"""
Binance Local Order Book Recorder
==================================
- Maintains a live local order book via Binance diff-depth stream
  following the official "how to manage a local order book correctly" procedure
- Every 1 second: computes best_bid, best_ask, spread, bid_depth (top N),
  ask_depth (top N), imbalance, midprice  →  stored in memory
- Every 5 minutes (clock-aligned): aggregates the 1-second rows and appends
  one record to  DD-MM-orderbook.csv

Requires:
    pip install websockets requests
"""

import asyncio
import json
import csv
import os
import time
import math
import requests
import websockets
from datetime import datetime, timezone

# ── Settings ──────────────────────────────────────────────────────────────────
SYMBOL     = "usdcusdt"      # lowercase, as Binance expects in the WS URL
DEPTH_N    = 20             # top N bid/ask levels for depth and imbalance
SNAP_LIMIT = 1000           # REST snapshot depth (1000 or 5000)
AGG_SECS   = 300            # aggregation window in seconds (300 = 5 minutes)
SAVE_DIR   = r"D:/data/binance_CEX/orderbook"  # directory to save CSVs (will be created if it doesn't exist)
# ──────────────────────────────────────────────────────────────────────────────

REST_URL = (
    f"https://api.binance.com/api/v3/depth"
    f"?symbol={SYMBOL.upper()}&limit={SNAP_LIMIT}"
)
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth@100ms"

# ── Shared state (module-level, no classes) ────────────────────────────────────
bids = {}           # price_str -> float qty  (maintained live)
asks = {}           # price_str -> float qty  (maintained live)
last_update_id = 0
ob_ready = False    # True once book is synced and safe to read

second_rows = []    # list of 1-second metric dicts, cleared every 5 min
# ──────────────────────────────────────────────────────────────────────────────


# ── Order book helpers ─────────────────────────────────────────────────────────

def apply_side(side, updates):
    """
    Apply a list of [price_str, qty_str] updates to one side of the book.
    qty == 0 → remove the level.
    """
    for price, qty in updates:
        q = float(qty)
        if q == 0.0:
            side.pop(price, None)
        else:
            side[price] = q


def fetch_snapshot_sync():
    """Blocking REST call – run in executor so it doesn't block the event loop."""
    resp = requests.get(REST_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── 1-second metric computation ────────────────────────────────────────────────

def compute_metrics():
    """
    Compute order book metrics from the current bids/asks dicts.
    Returns a dict, or None if the book is empty.
    """
    if not bids or not asks:
        return None

    sorted_bids = sorted(bids.items(), key=lambda x: float(x[0]), reverse=True)
    sorted_asks = sorted(asks.items(), key=lambda x: float(x[0]))

    best_bid = float(sorted_bids[0][0])
    best_ask = float(sorted_asks[0][0])
    spread   = best_ask - best_bid
    midprice = (best_bid + best_ask) / 2.0

    top_bids = sorted_bids[:DEPTH_N]
    top_asks = sorted_asks[:DEPTH_N]

    # Depth = notional value (USD) of the top-N levels: Σ price × qty
    bid_depth = sum(float(p) * q for p, q in top_bids)
    ask_depth = sum(float(p) * q for p, q in top_asks)

    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

    return {
        "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_bid":  best_bid,
        "best_ask":  best_ask,
        "spread":    spread,
        "midprice":  midprice,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
    }


# ── 5-minute aggregation and CSV save ─────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs)

def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def aggregate_and_save(rows, window_end):
    """
    Aggregate a list of 1-second dicts and append one row to the daily CSV.
    window_end is a datetime (UTC) marking the end of the 5-min window.
    """
    if not rows:
        print(f"[{window_end.strftime('%H:%M')}] No data collected – skipping save.")
        return

    spreads    = [r["spread"]    for r in rows]
    imbalances = [r["imbalance"] for r in rows]
    bid_depths = [r["bid_depth"] for r in rows]
    ask_depths = [r["ask_depth"] for r in rows]

    record = {
        "window_end":     window_end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_seconds":      len(rows),
        "spread_mean":    round(_mean(spreads),    6),
        "spread_std":     round(_std(spreads),     6),
        "imbalance_mean": round(_mean(imbalances), 6),
        "imbalance_std":  round(_std(imbalances),  6),
        "bid_depth_mean": round(_mean(bid_depths), 2),
        "bid_depth_max":  round(max(bid_depths),   2),
        "bid_depth_min":  round(min(bid_depths),   2),
        "ask_depth_mean": round(_mean(ask_depths), 2),
        "ask_depth_max":  round(max(ask_depths),   2),
        "ask_depth_min":  round(min(ask_depths),   2),
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = os.path.join(
        SAVE_DIR,
        window_end.strftime("%d-%m") + "-orderbook.csv"
    )
    exists   = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=record.keys())
        if not exists:
            w.writeheader()
        w.writerow(record)

    print(
        f"[{window_end.strftime('%H:%M')}] Saved {len(rows)} s → {filename} | "
        f"spread={record['spread_mean']:.4f}  imbal={record['imbalance_mean']:+.4f}"
    )


# ── WebSocket loop (order book maintenance) ────────────────────────────────────

async def ws_loop():
    """
    Connects to Binance, maintains the local order book following the official
    procedure, and sets ob_ready=True once the book is synced.
    Reconnects automatically on any error or gap.
    """
    global bids, asks, last_update_id, ob_ready

    while True:
        ob_ready = False
        buffer   = []
        prev_u   = None

        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,   # Binance server manages pings; avoid conflicting with its keepalive
            ) as ws:
                print(f"WebSocket connected → {WS_URL}")

                # ── Phase 1: buffer messages while fetching the REST snapshot ──
                loop      = asyncio.get_running_loop()
                snap_task = loop.run_in_executor(None, fetch_snapshot_sync)

                while not snap_task.done():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        buffer.append(json.loads(raw))
                    except asyncio.TimeoutError:
                        pass

                snap           = await snap_task
                last_update_id = snap["lastUpdateId"]
                bids           = {p: float(q) for p, q in snap["bids"]}
                asks           = {p: float(q) for p, q in snap["asks"]}
                print(
                    f"Snapshot: lastUpdateId={last_update_id}, "
                    f"{len(bids)} bids, {len(asks)} asks, {len(buffer)} buffered msgs"
                )

                # ── Phase 2: apply buffered messages ──────────────────────────
                # Step 4: drop events where u <= lastUpdateId
                # Step 5: first valid event must have U <= lastUpdateId+1 AND u >= lastUpdateId+1
                # Step 6: each subsequent event's U must equal previous u + 1
                found_start = False
                for msg in buffer:
                    if msg["u"] <= last_update_id:
                        continue                           # step 4: discard
                    if not found_start:
                        valid_start = (
                            msg["U"] <= last_update_id + 1
                            and msg["u"] >= last_update_id + 1
                        )
                        if not valid_start:
                            continue                       # step 5: not the right entry point yet
                        found_start = True
                    else:
                        if msg["U"] != prev_u + 1:        # step 6: gap in buffer
                            print("Gap in buffered messages – will re-sync via live stream.")
                            found_start = False
                            break

                    apply_side(bids, msg["b"])
                    apply_side(asks, msg["a"])
                    last_update_id = msg["u"]
                    prev_u         = msg["u"]

                ob_ready = True
                print(f"Order book ready (last_update_id={last_update_id})")

                # ── Phase 3: live stream ───────────────────────────────────────
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        print("No message received in 30 s – reconnecting...")
                        ob_ready = False
                        break

                    msg = json.loads(raw)

                    if msg["u"] <= last_update_id:
                        continue                           # already applied

                    if prev_u is not None and msg["U"] != prev_u + 1:
                        print(
                            f"Stream gap: expected U={prev_u + 1}, got U={msg['U']}. "
                            "Re-syncing..."
                        )
                        ob_ready = False
                        break

                    apply_side(bids, msg["b"])
                    apply_side(asks, msg["a"])
                    last_update_id = msg["u"]
                    prev_u         = msg["u"]

        except Exception as e:
            print(f"WebSocket error: {e}. Reconnecting in 5 s...")
            ob_ready = False
            await asyncio.sleep(5)


# ── 1-second tick loop (metrics + 5-min aggregation) ──────────────────────────

async def tick_loop():
    global second_rows

    # Align the first tick to the next whole second
    await asyncio.sleep(1.0 - (time.time() % 1.0))

    # Compute the next clock-aligned 5-minute boundary
    now      = time.time()
    next_5min = now + (AGG_SECS - now % AGG_SECS)

    while True:
        t0 = time.time()

        # Compute and store 1-second metrics
        if ob_ready:
            m = compute_metrics()
            if m:
                second_rows.append(m)

        # Check if we just crossed a 5-minute boundary
        if time.time() >= next_5min:
            window_end = datetime.fromtimestamp(
                next_5min, tz=timezone.utc
            )
            rows_to_save = second_rows[:]   # copy before clearing
            second_rows  = []
            aggregate_and_save(rows_to_save, window_end)
            next_5min += AGG_SECS

        # Sleep for the remainder of this second
        elapsed = time.time() - t0
        await asyncio.sleep(max(0.0, 1.0 - elapsed))


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    print(
        f"Binance OB recorder | symbol={SYMBOL.upper()} | "
        f"depth_N={DEPTH_N} | agg_window={AGG_SECS}s"
    )
    await asyncio.gather(ws_loop(), tick_loop())


if __name__ == "__main__":
    asyncio.run(main())
