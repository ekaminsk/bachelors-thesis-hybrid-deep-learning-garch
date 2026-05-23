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

import asyncio, json, os, sys, time, math, requests, websockets, csv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_SYMBOL, BINANCE_ORDERBOOK_DEPTH, INTERVAL, BINANCE_REST_URL, BINANCE_WEBSOCKET, CEX_ORDERBOOK, CEX_KLINES
from utilities import append_csv
from datetime import datetime, timezone

# ── Shared state ────────────────────────────────────

bids = {}               # (key) price_str -> (value) qty  (maintained live)
asks = {}               # (key) price_str -> (value) qty  (maintained live)
last_update_id = 0      # Binance applies an ID to each orderbook update; used to detect gaps
ob_ready = False        # True once book is synced and safe to read

second_rows = []        # list of 1-second metric dicts, cleared every 5 min

# ── Order book helpers ─────────────────────────────────────────────────────────

def apply_side(side, updates):
    # Apply a list of [price_str, qty_str] updates to one side of the book. qty == 0 -> remove the level.
    for price, qty in updates:
        q = float(qty)
        if q == 0.0:
            side.pop(price, None)
        else:
            side[price] = q


def fetch_snapshot_sync():
    # get initial snapshot of orderbook to have a basis to apply updates to. Sync because of ASYNCIO
    resp = requests.get(BINANCE_REST_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── 1-second metric computation ────────────────────────────────────────────────

def compute_metrics():
    # Compute order book metrics from the current bids/asks dicts.
    if not bids or not asks:
        return None

    sorted_bids = sorted(bids.items(), key=lambda x: float(x[0]), reverse=True) # Making float, so that ordering is correct, high first
    sorted_asks = sorted(asks.items(), key=lambda x: float(x[0]))

    best_bid = float(sorted_bids[0][0])
    best_ask = float(sorted_asks[0][0])
    spread   = best_ask - best_bid
    midprice = (best_bid + best_ask) / 2.0

    top_bids = sorted_bids[:BINANCE_ORDERBOOK_DEPTH]            # 20 best bids (depth = 20)
    top_asks = sorted_asks[:BINANCE_ORDERBOOK_DEPTH]            # 20 best asks (depth = 20)

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

def _mean(list):
    return sum(list) / len(list)

def _std(list):
    m = _mean(list)
    return math.sqrt(sum((x - m) ** 2 for x in list) / len(list))


def aggregate_and_save(rows, window_end):
    # Aggregate a list of 1-second dicts and append one row to the daily CSV.
    if not rows:
        print(f"[{window_end.strftime('%H:%M')}] No data collected - skipping save.")
        return

    spreads    = [observation["spread"]    for observation in rows]
    imbalances = [observation["imbalance"] for observation in rows]
    bid_depths = [observation["bid_depth"] for observation in rows]
    ask_depths = [observation["ask_depth"] for observation in rows]

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

    filename = f"{window_end.strftime('%d-%m')}-orderbook.csv"
    append_csv(CEX_ORDERBOOK, filename, record)

    print(
        f"[{window_end.strftime('%H:%M')}] Saved {len(rows)} s → {filename} | "
        f"spread={record['spread_mean']:.4f}  imbal={record['imbalance_mean']:+.4f}"
    )


# ── WebSocket loop (order book maintenance) ────────────────────────────────────
    # The loop works like this: We use asyncio to have two loops running concurrently.
    # ws_loop() 
        # fetches a snapshot and meanwhile buffers incoming updates (as long as it takes to get the snapshot).
        # After that, ws_loop() finds the bridge from snapshot to livestream and continues receiving updates from the stream and adjusting the local orderbook.
    # tick_loop()
        # takes local orderbook every second (global list) and computes metrics.
        # if while running this loop, time crosses a 5min boundary it saves all metric-rows into CSV and clears the computed metrics locally.
    # Normally, the loops switch when ws_loop() is wating for an update (every 100ms one update comes in) to tick_loop().
    # tick_loop() takes a few milliseconds to run and then goes to sleep for the remainder of the second, switching back to ws_loop().
    # since tick_loop() sleeps for most of the time (remainder of the second after it runs to compute metrics) ws_loop() receives update, switches to tick_loop() and immediately switches back to ws_loop() 9/10 times per second.
    # If a gap is found, it tries to reconnect the orderbook first, if that does not work, the orderbook snapshot is fetched again. If all goes well, ws_loop() stays in phase 3.

async def ws_loop():
    global bids, asks, last_update_id, ob_ready

    while True:                             # to have a clean slate if I need to restart the code
        ob_ready = False
        buffer   = []
        prev_u   = None

        try:
            async with websockets.connect(
                BINANCE_WEBSOCKET,
                ping_interval=None,             # Binance server manages pings; avoid conflicting with its keepalive
            ) as ws:
                print(f"WebSocket connected → {BINANCE_WEBSOCKET}")

        # ── Phase 1: buffer messages while fetching the REST snapshot ──────────
                loop      = asyncio.get_running_loop()
                snap_task = loop.run_in_executor(None, fetch_snapshot_sync)
                    # snap task is being run, in the meanwhile the program (already connected to websockets) collects updates
                while not snap_task.done():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.3)    # execute: waiting for an update for 0.3 sec
                        buffer.append(json.loads(raw))                          # append update to buffer list
                    except asyncio.TimeoutError:
                        pass

                snap           = await snap_task                                # get value from the future that has been completed (because loop exited)
                last_update_id = snap["lastUpdateId"]
                bids           = {p: float(q) for p, q in snap["bids"]}
                asks           = {p: float(q) for p, q in snap["asks"]}
                print(
                    f"Snapshot: lastUpdateId={last_update_id}, "
                    f"{len(bids)} bids, {len(asks)} asks, {len(buffer)} buffered msgs"
                )
        
        # ── Phase 2: apply buffered messages ───────────────────────────────────
            # The goal is to take the snapshot and connect the livestream (pre buffered) to the snapshot
            # For that, we need to find the first update (batch) (from livestream) that has one update with ID = lastUpdateID from snapshot
            # 'u' is the ID of the last update in the batch, 'U' is the ID of the first update in the batch 
                found_start = False
                for msg in buffer:
                    if msg["u"] <= last_update_id:
                        continue                                # discard update batch, if all updates are already in the snapshot
                    if not found_start:                         
                        valid_start = (
                            msg["U"] <= last_update_id + 1
                            and msg["u"] >= last_update_id + 1  # logic for what the valid bridge is -> first update can come before the snapshot but last update is after the snapshot -> somewhere in that batch there is last_update_id
                        )
                        if not valid_start:
                            continue                            # happens if somehow the order of updates is scrambled (e.g. 100 = last_update_id, msg1 = [95,98], msg2 = [110,111], msg3 = [99,102])
                        found_start = True
                    else:
                        if msg["U"] != prev_u + 1:              # if a gap in the buffer gets detected (e.g. u_prev = 110, U_next = 112) then the livestreams tries to fix the gap, if it cannot it restarts
                            print("Gap in buffered messages. Will re-sync via live stream.")
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
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)             # if nothing arrives for 30sec set ob_ready = False -> tick_loop exits
                    except asyncio.TimeoutError:
                        print("No message received in 30 s - reconnecting...")
                        ob_ready = False
                        break

                    msg = json.loads(raw)                                               # each incoming update pasted into a python dictionary

                    if msg["u"] <= last_update_id:
                        continue                                                        # skips update if already applied

                    if prev_u is not None and msg["U"] != prev_u + 1:                   # gap detection -> here reconnect into phase 1
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

    await asyncio.sleep(1.0 - (time.time() % 1.0))                  # Align the first tick to the next whole second

    now      = time.time()
    next_5min = now + (INTERVAL - now % INTERVAL)                   # Compute the first clock-aligned 5-minute boundary after starting the loop

    while True:
        t0 = time.time()                                            # one iteration per second, later for sleeping

        if ob_ready:
            m = compute_metrics()                                   # compute metrics if there are no gaps
            if m:
                second_rows.append(m)                               # if there is anything computed, append

        if time.time() >= next_5min:                                # since this loop keeps running, it checks if the 5min boundary was crossed, if so it saves and resets the local metric list
            window_end = datetime.fromtimestamp(                    
                next_5min, tz=timezone.utc
            )
            rows_to_save = second_rows[:]                           # copy all computed metric enteries before clearing the local list
            second_rows  = []
            aggregate_and_save(rows_to_save, window_end)
            next_5min += INTERVAL                                   # next 5min interval

        elapsed = time.time() - t0                                  # Sleep for the remainder of this second
        await asyncio.sleep(max(0.0, 1.0 - elapsed))


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    os.makedirs(CEX_KLINES, exist_ok=True)
    print(f"Please place your downloaded kline files into: {CEX_KLINES}")
    print(
        f"Binance OB recorder | symbol={BINANCE_SYMBOL.upper()} | "
        f"depth_N={BINANCE_ORDERBOOK_DEPTH} | agg_window={INTERVAL}s"
    )
    await asyncio.gather(ws_loop(), tick_loop())


if __name__ == "__main__":
    asyncio.run(main())
