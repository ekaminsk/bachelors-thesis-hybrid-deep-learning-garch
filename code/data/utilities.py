"""
This file will contain a few functions, which are use across multiple data fetchers.
"""
import csv
import os
import requests

# ── Save to CSV ───────────────────────────────────────────────────────────────────
def save_csv(rows, filepath): 
    # Write lists of dictionaries into CSV. Check if there is anything, if so create a directory, add a writing file, add keys as headers and add all rows
    if not rows:
        print(f"WARNING: no rows to save -- skipping {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {filepath}")

def append_csv(directory, filename, row):
    # Append new information onto a CSV for streaming. If first time, create directory and file, after that just append new stuff to the file
    os.makedirs(directory, exist_ok=True)
    path   = os.path.join(directory, filename)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

# ── GraphQL queries ───────────────────────────────────────────────────────────────
# Here, because this function is used in both Uniswap data fetchers.

# resp is a GraphQL query, which takes the URL, retuns a json, parses it into python dictionary, and if it does not hear anything from the API for 30sec it stops.
def gql(url, query, variables=None):
    resp = requests.post(
        url,
        json={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]

# ── Square root price ─────────────────────────────────────────────────────────────
# Uniswap saves prices as sqrt(p)*2^96. Thus the function recomputes the price from the square-root price

def sqrt_price_to_price(token0, token1, sqrt_price_x96_str):
    price = (int(sqrt_price_x96_str) / 2**96) ** 2
    price *= 10 ** (token0 - token1)    # adjust decimal places | not relevant in the thesis, but in case I change decimal places
    return price