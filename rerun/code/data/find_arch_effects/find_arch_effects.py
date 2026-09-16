"""
All this does is to find a timeframe with ARCH effects. 
Therefore, I won't connect it to a config file. 
"""

import pandas as pd 
import requests, os

# ── Constants & Filepaths ─────────────────────────────────────────────────────

BASE_PATH       = os.path.dirname(os.path.abspath(__file__)) 
INPUT_PATH      = os.path.join(BASE_PATH, "input_blocks.csv")
OUTPUT_PATH     = os.path.join(BASE_PATH, "output_return_series.csv")
START_DATE      = "2023-02-01 00:00:00.000 UTC"             #YYYY-MM-DD HH:MM:SS.MSS
END_DATE        = "2023-05-01 00:00:00.000 UTC"


# ── Import block numbers ──────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)
df = df.set_index("window_end")
df_cut = df.truncate(before=START_DATE, after=END_DATE)
print(df_cut.head(10))


