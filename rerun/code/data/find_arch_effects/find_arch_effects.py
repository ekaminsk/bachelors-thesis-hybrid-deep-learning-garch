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
from requests.exceptions import RetryError
from urllib3.util.retry import Retry


# ── Constants & Filepaths ─────────────────────────────────────────────────────

BASE_PATH       = os.path.dirname(os.path.abspath(__file__)) 
INPUT_PATH      = os.path.join(BASE_PATH, "input_blocks.csv")
OUTPUT_PATH     = os.path.join(BASE_PATH, "output_return_series.csv")
START_DATE      = "2023-01-01 00:00:00"             #YYYY-MM-DD HH:MM:SS
END_DATE        = "2023-01-01 00:10:00"

load_dotenv()
UNISWAP_API_KEY         = os.getenv("UNISWAP_API_KEY")

SUBGRAPH_ID     = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
GRAPH_URL       = f"https://gateway.thegraph.com/api/{UNISWAP_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
POOL_ID         = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"


# ── Import block numbers ──────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)
df["window_end"] = df["window_end"].str.split(pat=".",n=1).str[0]
df = df.set_index("window_end")
df_cut = df.truncate(before=START_DATE, after=END_DATE)

# ── GraphQL Query ─────────────────────────────────────────────────────────────
"""
The query takes id and block as parameters. 
Then it enters the pool data, and filters for id and block, which we use the block's number for, and returns the sqrtPrice. 
"""

QUERY = """
query ReturnSeries($id: ID!, $block: Int!){
    pool(
    id: $id,
    block: {number: $block}
    ){
    sqrtPrice
    }
}
"""

# ── Data collection setup ─────────────────────────────────────────────────────

retry_strategy = Retry(
    total               =10,
    backoff_factor      =1,
    status_forcelist    =[429, 500, 502, 503, 504],
    allowed_methods     =frozenset(["POST"]),
)
graph_adapter = HTTPAdapter(max_retries=retry_strategy)

def gql_over_blocks(block_number, max_attempts=5, backoff =2):          # Exponential backoff due to some index being pruned around 2024
    for attempt in range(1, max_attempts + 1):
        request = s.post(
            url=GRAPH_URL,
            json={"query": QUERY, "variables":{"id": POOL_ID, "block": block_number}},
            timeout=30
            )
        data = request.json()
        if "errors" not in data:
            return data
        else:
            if attempt < max_attempts:
                time.sleep(backoff**attempt)
    raise RuntimeError(data["errors"])
        
# ── Outputting price data ─────────────────────────────────────────────────────

results = []                        # Apperently its more computationally intensive to append each row to df

with requests.Session() as s:       # This should keep a connection open, so I don't have to open it every time
    max_attempts = 10
    s.mount("https://gateway.thegraph.com", graph_adapter)
    for timestamp, row in df_cut.iterrows():
        result = gql_over_blocks(block_number = int(row.block_number))
        raw_price = result["data"]["pool"]["sqrtPrice"]
        clean_price = (int(raw_price) / 2**96) ** 2
        row = {
            "window_end": timestamp,
            "price" : clean_price
            }
        results.append(row)

df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"File was saved to {OUTPUT_PATH}")
