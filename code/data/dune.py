
import os, time, requests
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DUNE_API_KEY, DUNE_START_DATE, DUNE_END_DATE, DUNE_WHALE_THRESHOLD, DUNE_DIR, DUNE_BASE_URL, DUNE_POLL_INTERVAL, DUNE_RESULTS_PER_PAGE
from utilities import save_csv
from datetime import datetime, timezone

# ── Query definitions ──────────────────────────────────────────────────────────
QUERIES = [
    {
        "id":       6763552,
        "name":     "Q1 : Whale Transfers",
        "filename": "q1_whale_transfers.csv",
        "params": {
            "start_date":          DUNE_START_DATE,
            "end_date":            DUNE_END_DATE,
            "whale_threshold_usd": DUNE_WHALE_THRESHOLD,
        },
    },
    {
        "id":       6763555,
        "name":     "Q2 : CEX Inflows & Outflows",
        "filename": "q2_cex_flows.csv",
        "params": {
            "start_date": DUNE_START_DATE,
            "end_date":   DUNE_END_DATE,
        },
    },
    {
        "id":       6763557,
        "name":     "Q3 : Gas Price Time Series",
        "filename": "q3_gas_price.csv",
        "params": {
            "start_date": DUNE_START_DATE,
            "end_date":   DUNE_END_DATE,
        },
    },
    {
        "id":       6763559,
        "name":     "Q4 : Gas Used & Block Utilization",
        "filename": "q4_gas_used.csv",
        "params": {
            "start_date": DUNE_START_DATE,
            "end_date":   DUNE_END_DATE,
        },
    },
    {
        "id":       6763560,
        "name":     "Q5 : Mempool Congestion Proxies",
        "filename": "q5_mempool_congestion.csv",
        "params": {
            "start_date": DUNE_START_DATE,
            "end_date":   DUNE_END_DATE,
        },
    },
    {
        "id":       6763561,
        "name":     "Q6 : USDC/USDT Mints & Burns",
        "filename": "q6_supply_changes.csv",
        "params": {
            "start_date": DUNE_START_DATE,
            "end_date":   DUNE_END_DATE,
        },
    },
]


# ── API helpers ────────────────────────────────────────────────────────────────

HEADERS = {"X-Dune-Api-Key": DUNE_API_KEY}


def execute_query(query_id, params):
    url  = f"{DUNE_BASE_URL}/query/{query_id}/execute"                                   # Where the query is found, because dune runs the queries in their own environment and API allows to access results
    body = {"query_parameters": params, "performance": "medium"}                    # performance is a trade off of tokens vs. speed (mainly)
    resp = requests.post(url, json=body, headers=HEADERS, timeout=30)             
    resp.raise_for_status()
    execution_id = resp.json()["execution_id"]                                      # dune returns execution_id, which after running through can be called to collect data
    return execution_id


def poll_until_done(execution_id, query_name):
    url = f"{DUNE_BASE_URL}/execution/{execution_id}/status"                                             # check status of query
    while True:                                                                                     # loop until return
        resp  = requests.get(url, headers=HEADERS, timeout=30)                                    # get request to find status
        resp.raise_for_status()
        data  = resp.json()
        state = data.get("state", "UNKNOWN")
        print(f"    [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {query_name}: {state}")     # "what is the state", if none not error but unknown

        if state == "QUERY_STATE_COMPLETED":
            rows = data.get("result_metadata", {}).get("total_row_count", "?")                      # find how many rows for return statement, to check if its as expected (usually 12 per hour 60/5)
            print(f"    Done -- {rows} rows")
            return state
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"):
            print(f"    ERROR: {data}")                                                             # if error, then data is the HTTP error
            return state

        time.sleep(DUNE_POLL_INTERVAL)


def fetch_all_rows(execution_id):
    """Paginate through results JSON and return all rows as a list of dicts."""
    all_rows = []
    offset   = 0

    while True:
        url    = f"{DUNE_BASE_URL}/execution/{execution_id}/results"                             # after we know the results are there, put into list
        params = {"limit": DUNE_RESULTS_PER_PAGE, "offset": offset}                              # max. 10000 per page
        resp   = requests.get(url, headers=HEADERS, params=params, timeout=60)
        resp.raise_for_status()
        data   = resp.json()

        rows = data.get("result", {}).get("rows", [])                                       # take 10000 rows from results add to all_rows list 
        all_rows.extend(rows)

        total = data.get("result", {}).get("metadata", {}).get("total_row_count", 0)        # find total number of rows
        offset += len(rows)                                                                 # 0 -> 10.000 -> 20.000

        print(f"Fetched {offset} / {total} rows")

        if offset >= total or not rows:                                                     # only if collected >= total or there are no rows it stops
            break

    return all_rows

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Dune fetcher  |  {DUNE_START_DATE}  ->  {DUNE_END_DATE}")
    print(f"Saving to: {DUNE_DIR}\n")

    for q in QUERIES:
        print(f"{'─'*60}")
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
        filepath = os.path.join(DUNE_DIR, q["filename"])
        save_csv(rows, filepath)

    print(f"\n{'='*60}")
    print("All queries done.")

if __name__ == "__main__":
    main()
