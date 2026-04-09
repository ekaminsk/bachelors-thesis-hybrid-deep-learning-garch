#!/usr/bin/env python3
"""
Dune Analytics Query Fetcher
=============================
Executes all 6 thesis queries via the Dune API and saves results as CSV
in D:/data/dune_ONCHAIN/ with the filenames expected by aggregate_5min.py.

Flow per query:
  1. POST  /v1/query/{id}/execute     -- trigger execution with parameters
  2. GET   /v1/execution/{id}/status  -- poll until COMPLETED or FAILED
  3. GET   /v1/execution/{id}/results -- paginate and collect all rows
  4. Write CSV to SAVE_DIR

"""

import os, time, csv, requests
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
DUNE_API_KEY     = "xxx"

START_DATE       = "2026-03-06 00:00:00"
END_DATE         = "2026-03-17 00:00:00"
WHALE_THRESHOLD  = "1000000"              # USD, for Q1 only

SAVE_DIR         = "D:/data/dune_ONCHAIN"
BASE_URL         = "https://api.dune.com/api/v1"

POLL_INTERVAL    = 5     # seconds between status checks
RESULTS_PER_PAGE = 10000 # rows per results page (max Dune allows)
# ───────────────────────────────────────────────────────────────────────────────


# ── Query definitions ──────────────────────────────────────────────────────────

QUERIES = [
    {
        "id":       6763552,
        "name":     "Q1 – Whale Transfers",
        "filename": "q1_whale_transfers.csv",
        "params": {
            "start_date":          START_DATE,
            "end_date":            END_DATE,
            "whale_threshold_usd": WHALE_THRESHOLD,
        },
    },
    {
        "id":       6763555,
        "name":     "Q2 – CEX Inflows & Outflows",
        "filename": "q2_cex_flows.csv",
        "params": {
            "start_date": START_DATE,
            "end_date":   END_DATE,
        },
    },
    {
        "id":       6763557,
        "name":     "Q3 – Gas Price Time Series",
        "filename": "q3_gas_price.csv",
        "params": {
            "start_date": START_DATE,
            "end_date":   END_DATE,
        },
    },
    {
        "id":       6763559,
        "name":     "Q4 – Gas Used & Block Utilization",
        "filename": "q4_gas_used.csv",
        "params": {
            "start_date": START_DATE,
            "end_date":   END_DATE,
        },
    },
    {
        "id":       6763560,
        "name":     "Q5 – Mempool Congestion Proxies",
        "filename": "q5_mempool_congestion.csv",
        "params": {
            "start_date": START_DATE,
            "end_date":   END_DATE,
        },
    },
    {
        "id":       6763561,
        "name":     "Q6 – USDC/USDT Mints & Burns",
        "filename": "q6_supply_changes.csv",
        "params": {
            "start_date": START_DATE,
            "end_date":   END_DATE,
        },
    },
]


# ── API helpers ────────────────────────────────────────────────────────────────

def headers():
    return {"X-Dune-API-Key": DUNE_API_KEY}


def execute_query(query_id, params):
    """Trigger query execution. Returns execution_id."""
    url  = f"{BASE_URL}/query/{query_id}/execute"
    body = {"query_parameters": params, "performance": "medium"}
    resp = requests.post(url, json=body, headers=headers(), timeout=30)
    resp.raise_for_status()
    execution_id = resp.json()["execution_id"]
    return execution_id


def poll_until_done(execution_id, query_name):
    """Poll status until COMPLETED, FAILED, or CANCELLED. Returns final state."""
    url = f"{BASE_URL}/execution/{execution_id}/status"
    while True:
        resp  = requests.get(url, headers=headers(), timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        state = data.get("state", "UNKNOWN")
        print(f"    [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {query_name}: {state}")

        if state == "QUERY_STATE_COMPLETED":
            rows = data.get("result_metadata", {}).get("total_row_count", "?")
            print(f"    Done -- {rows} rows")
            return state
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"):
            print(f"    ERROR: {data}")
            return state

        time.sleep(POLL_INTERVAL)


def fetch_all_rows(execution_id):
    """Paginate through results JSON and return all rows as a list of dicts."""
    all_rows = []
    offset   = 0

    while True:
        url    = f"{BASE_URL}/execution/{execution_id}/results"
        params = {"limit": RESULTS_PER_PAGE, "offset": offset}
        resp   = requests.get(url, headers=headers(), params=params, timeout=60)
        resp.raise_for_status()
        data   = resp.json()

        rows = data.get("result", {}).get("rows", [])
        all_rows.extend(rows)

        total = data.get("result", {}).get("metadata", {}).get("total_row_count", 0)
        offset += len(rows)

        print(f"    Fetched {offset} / {total} rows")

        if offset >= total or not rows:
            break

    return all_rows


def save_csv(rows, filepath):
    """Write list of dicts to CSV. Preserves column order from first row."""
    if not rows:
        print(f"    WARNING: no rows to save -- skipping {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Saved {len(rows)} rows -> {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if DUNE_API_KEY == "YOUR_DUNE_API_KEY":
        print("ERROR: set your DUNE_API_KEY at the top of this file.")
        print("       Find it at: dune.com -> top-right avatar -> Settings -> API")
        return

    print(f"Dune fetcher  |  {START_DATE}  ->  {END_DATE}")
    print(f"Saving to: {SAVE_DIR}\n")

    for q in QUERIES:
        print(f"{'='*60}")
        print(f"{q['name']}  (ID: {q['id']})")

        # 1. Trigger execution
        try:
            exec_id = execute_query(q["id"], q["params"])
            print(f"  execution_id: {exec_id}")
        except Exception as e:
            print(f"  FAILED to execute: {e}")
            continue

        # 2. Poll until done
        state = poll_until_done(exec_id, q["name"])
        if state != "QUERY_STATE_COMPLETED":
            print(f"  Skipping -- query did not complete (state={state})")
            continue

        # 3. Fetch all result rows
        try:
            rows = fetch_all_rows(exec_id)
        except Exception as e:
            print(f"  FAILED to fetch results: {e}")
            continue

        # 4. Save CSV
        filepath = os.path.join(SAVE_DIR, q["filename"])
        save_csv(rows, filepath)

    print(f"\n{'='*60}")
    print("All queries done.")
    print(f"\nNext steps:")
    print(f"  1. Set DUNE_ENABLED = True  in  D:/data/aggregate_5min.py")
    print(f"  2. Run: python D:/data/aggregate_5min.py")


if __name__ == "__main__":
    main()
