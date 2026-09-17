"""
All this does is to find a timeframe with ARCH effects. 
Therefore, I won't connect it to a config file. 
"""

import pandas as pd 
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ── Constants & Filepaths ─────────────────────────────────────────────────────

BASE_PATH       = os.path.dirname(os.path.abspath(__file__)) 
INPUT_PATH      = os.path.join(BASE_PATH, "input_blocks.csv")
OUTPUT_PATH     = os.path.join(BASE_PATH, "output_return_series.csv")
START_DATE      = "2023-01-01 00:00:00"             #YYYY-MM-DD HH:MM:SS
END_DATE        = "2023-01-03 00:00:00"

load_dotenv()
UNISWAP_API_KEY = os.getenv("UNISWAP_API_KEY")

SUBGRAPH_ID     = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
GRAPH_URL       = f"https://gateway.thegraph.com/api/{UNISWAP_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
POOL_ID         = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"


# ── Import block numbers ──────────────────────────────────────────────────────

df                  = pd.read_csv(INPUT_PATH)
df["window_end"]    = df["window_end"].str.split(pat=".",n=1).str[0]
df                  = df.set_index("window_end")
df_cut              = df.truncate(before=START_DATE, after=END_DATE)

# ── GraphQL Query ─────────────────────────────────────────────────────────────

QUERY = """
query ReturnSeriesThroughSwaps($pool: String!, $startTs: Int!, $endTs: Int!, $cursor: ID!){
    swaps(
    first: 1000,
    where: {pool: $pool, timestamp_gt: $startTs, timestamp_lt: $endTs, id_gt: $cursor},
    orderBy: id,
    orderDirection: asc){
    id,
    timestamp,
    sqrtPriceX96
    }
}
"""

# ── Data collection helpers ───────────────────────────────────────────────────

retry_strategy = Retry(
    total               = 10,
    backoff_factor      = 1,
    status_forcelist    = [429, 500, 502, 503, 504],
    allowed_methods     = frozenset(["POST"]),
)
graph_adapter = HTTPAdapter(max_retries=retry_strategy)

def gql_over_blocks(query, variables=None, max_attempts=5, backoff =2):          # Exponential backoff due to some index being pruned around 2024
    for attempt in range(1, max_attempts + 1):
        request = s.post(
            url     = GRAPH_URL,
            json    = {"query": query, "variables": variables},
            timeout = 30
            )
        data = request.json()
        if "errors" not in data:
            return data
        else:
            if attempt < max_attempts:
                time.sleep(backoff**attempt)
    raise RuntimeError(data["errors"])
        
# ── Fetching price data ───────────────────────────────────────────────────────

with requests.Session() as s:                                                   # This should keep a connection open, so I don't have to open it every time
    s.mount("https://gateway.thegraph.com", graph_adapter)

def pull_raw_data():
    cursor      = ""
    all_swaps   = []
    page        = 0
    start       = int((datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).timestamp())
    end         = int((datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).timestamp())
    while True:
        data = gql_over_blocks(query=QUERY, variables={
            "pool":     POOL_ID,
            "startTs":  start,
            "endTs":    end,
            "cursor":   cursor 
        })
        batch = data["data"]["swaps"]
        if not batch:
            break
        df_batch = pd.DataFrame(batch)
        df_batch["timestamp"] = df_batch["timestamp"].astype(int)                           # For sorting step
        df_batch["price"] = (df_batch["sqrtPriceX96"].astype(float)/ 2**96) ** 2
        all_swaps.append(df_batch[["id", "timestamp", "price"]])

        if len(batch) < 1000:
            break

        cursor = batch[-1]["id"]
        page += 1
        time.sleep(0.25)

    print(f"Collected {len(all_swaps)} observations through {page+1} queries")
    df_result = pd.concat(all_swaps, ignore_index=True)
    df_result = df_result.sort_values(by="timestamp")
    df_result["timestamp"] = pd.to_datetime(df_result["timestamp"],unit="s",utc=True)
    return df_result

# ── Creating the 5-min grid ───────────────────────────────────────────────────

grid    = pd.date_range(start=START_DATE, end=END_DATE, freq="5min", tz="UTC", unit="s")   # Build a 5min grid
grid_df = pd.DataFrame({"timestamp": grid})

data_df = pull_raw_data()

df_final = pd.merge_asof(                                   # "Left join" on the 5min grid, where I take the price value that is closest less-equal to the 5min step
    left        = grid_df,
    right       = data_df,
    on          = "timestamp",
    direction   = "backward" 
)[["timestamp","price"]].set_index("timestamp")

df_final.to_csv(OUTPUT_PATH, index=True)
print(f"File was saved to {OUTPUT_PATH}")
