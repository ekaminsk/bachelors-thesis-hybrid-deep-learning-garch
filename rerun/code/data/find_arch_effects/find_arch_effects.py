"""
All this does is to find a timeframe with ARCH effects. 
Therefore, I won't connect it to a config file. 
"""

import pandas, requests, sys, os

# ── Constants & Filepaths ─────────────────────────────────────────────────────

BASE_PATH       = os.path.dirname(os.path.abspath(__file__)) 
INPUT_PATH      = os.path.join(BASE_PATH, "input_blocks.csv")
OUTPUT_PATH     = os.path.join(BASE_PATH, "output_return_series.csv")



# ── Import block numbers ──────────────────────────────────────────────────────

