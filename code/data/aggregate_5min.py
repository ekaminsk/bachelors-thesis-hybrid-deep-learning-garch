"""
Merges ALL data sources into a single time-indexed file at 5-min resolution.
Time index: window_end (UTC) -- the candle/snapshot closes at this timestamp.
  e.g. window_end = 2026-03-06 00:05:00  covers  00:00:00 -> 00:05:00
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

from config import DEX_MINTS_BURNS, DEX_POOL, DEX_SWAPS, DEX_TICKS
from config import CEX_KLINES, CEX_ORDERBOOK
from config import DUNE_DIR
from config import AGGREGATE_OUTPUT, AGGREGATED_DATA
from config import INTERVAL

# ── Extra Config ─────────────────────────────────────────────────────────────

WINDOW = timedelta(seconds=INTERVAL)


# ── DEX Loaders ──────────────────────────────────────────────────────────────

def load_dex_klines():
    # DEX Klines calculated in univ3_swap.py
    # Sort files in DEX_SWAPS with ending -klines.csv. Change window_start to window_end for later. Rename columns
    frames = []
    for path in sorted(Path(DEX_SWAPS).glob("*-klines.csv")):
        df = pd.read_csv(path, parse_dates=["window_start"])
        df["window_end"] = df["window_start"] + WINDOW
        frames.append(df)
    if not frames:
        print(f"  [DEX] WARNING: no Kline files found in {DEX_SWAPS}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end")
    out = out.set_index("window_end").drop(columns=["window_start"])
    out.columns = [
        "dex_klines_price_open",
        "dex_klines_price_high",
        "dex_klines_price_low",
        "dex_klines_price_close",
        "dex_klines_volume_usd",
        "dex_klines_n_swaps",
        "dex_klines_imbalance",
        "dex_klines_large_trades_count",
        "dex_klines_large_trades_usd",
    ]
    return out


def load_dex_swaps_raw():
    # Swaps from univ3_swaps.py
    # Here first need to aggregate the swaps into 5min windows. Also clean up, because timestamps are not clean, USD amount might be negative.
    frames = []
    for path in sorted(Path(DEX_SWAPS).glob("*-swaps-raw.csv")):
        df = pd.read_csv(path)
        df["dt"] = pd.to_datetime(
            df["timestamp"].astype(int),unit="s", utc=True                          # here convert Unix into readable timestamp also stuff with timezone info so that it merges correctly with other datetimes later 
        ).dt.tz_localize(None)
        df["amountUSD"] = pd.to_numeric(df["amountUSD"], errors="coerce").abs()
        df["window_end"] = df["dt"].dt.floor("5min") + WINDOW
        frames.append(df)
    if not frames:
        print(f"  [DEX] WARNING: no Swap files found in {DEX_SWAPS}")
        return pd.DataFrame()
    all_swaps = pd.concat(frames, ignore_index=True)
    out = all_swaps.groupby("window_end").agg(
        dex_swaps_n_total               =("amountUSD", "count"),
        dex_swaps_unique_senders        =("sender",    "nunique"),
        dex_swaps_unique_recipients     =("recipient", "nunique"),
        dex_swaps_tick_min              =("tick",      "min"),
        dex_swaps_tick_max              =("tick",      "max"),
        dex_swaps_usd_mean_per_swap     =("amountUSD", "mean"),
        dex_swaps_usd_median_per_swap   =("amountUSD", "median"),
        dex_swaps_usd_max_single_swap   =("amountUSD", "max"),
    )
    return out


def load_dex_pool():
    # Pool data from univ3_pool.py
    # Just concat based on window_end, and rename columns.
    frames = []
    for path in sorted(Path(DEX_POOL).glob("*-pool.csv")):
        df = pd.read_csv(path, parse_dates=["window_end"])
        frames.append(df)
    if not frames:
        print(f"  [DEX] WARNING: no files found in {DEX_POOL}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end").set_index("window_end")
    out = out.rename(columns={
        "liquidity":    "dex_pool_liquidity",
        "tick":         "dex_pool_tick",
        "sqrt_price":   "dex_pool_sqrt_price",
        "price":        "dex_pool_price",
        "token0_price": "dex_pool_token0_price",
        "token1_price": "dex_pool_token1_price",
        "tvl_token0":   "dex_pool_tvl_usdc",
        "tvl_token1":   "dex_pool_tvl_usdt",
        "tvl_usd":      "dex_pool_tvl_usd",
    })
    return out


def load_dex_ticks():
    # Tick data from univ3_pool.py
    # Same thing as load_dex_pool()
    frames = []
    for path in sorted(Path(DEX_TICKS).glob("*-ticks.csv")):
        df = pd.read_csv(path, parse_dates=["window_end"])
        frames.append(df)
    if not frames:
        print(f"  [DEX] WARNING: no files found in {DEX_TICKS}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end").set_index("window_end")
    out = out.rename(columns={
        "current_tick":    "dex_ticks_current_tick",
        "tick_lo":         "dex_ticks_active_lo",
        "tick_hi":         "dex_ticks_active_hi",
        "n_ticks":         "dex_ticks_n_active",
        "total_liq_gross": "dex_ticks_total_liq_gross",
        "net_liq_above":   "dex_ticks_net_liq_above",
        "net_liq_below":   "dex_ticks_net_liq_below",
    })
    return out


def load_dex_mints_burns():
    frames = []
    for path in sorted(Path(DEX_MINTS_BURNS).glob("*-mints-burns.csv")):
        df = pd.read_csv(path, parse_dates=["window_end"])
        frames.append(df)
    if not frames:
        print(f"  [DEX] WARNING: no files found in {DEX_MINTS_BURNS}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end").set_index("window_end")
    out = out.rename(columns={
        "n_mints":        "dex_liquidityproviders_n_mints",
        "n_burns":        "dex_liquidityproviders_n_burns",
        "mint_liq":       "dex_liquidityproviders_mint_liq_raw",
        "burn_liq":       "dex_liquidityproviders_burn_liq_raw",
        "mint_amount0":   "dex_liquidityproviders_mint_amount0_usdc",
        "mint_amount1":   "dex_liquidityproviders_mint_amount1_usdt",
        "mint_usd":       "dex_liquidityproviders_mint_usd",
        "burn_amount0":   "dex_liquidityproviders_burn_amount0_usdc",
        "burn_amount1":   "dex_liquidityproviders_burn_amount1_usdt",
        "burn_usd":       "dex_liquidityproviders_burn_usd",
        "net_liq_change": "dex_liquidityproviders_net_liq_change",
    })
    return out


# ── CEX Loaders ──────────────────────────────────────────────────────────────

def load_cex_klines():
    # Data downloaded from https://data.binance.vision/?prefix=data/spot/daily/klines/
    # Since this data is uploaded by Binance, I need to adjust data types. Mostly already explained above, only new thing is changing columns to numeric, because Binance has them in string. 
    # Admittingly, "taker_buy_..." is most likely irrelevant
    frames = []
    for path in sorted(Path(CEX_KLINES).glob("USDCUSDT-5m-*.csv")):
        df = pd.read_csv(path)
        df["window_end"] = pd.to_datetime(
            df["window_start"], unit="us", utc=True             # Binance window_end is the last microsecond of the candle (e.g. 00:04:59.999999). Use window_start + 5min to get a clean 00:05:00 boundary instead.
        ).dt.tz_localize(None) + WINDOW
        for col in ["open", "high", "low", "close", "volume",
                    "quote_asset_volume",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        print(f"  [CEX] WARNING: no files found in {CEX_KLINES}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end").set_index("window_end")
    out = out.drop(columns=["window_start", "ignore"], errors="ignore")

    taker_buy  = out["taker_buy_base_asset_volume"]
    total_base = out["volume"]

    out = out.rename(columns={
        "open":                        "cex_klines_price_open",
        "high":                        "cex_klines_price_high",
        "low":                         "cex_klines_price_low",
        "close":                       "cex_klines_price_close",
        "volume":                      "cex_klines_volume_usdc",
        "quote_asset_volume":          "cex_klines_volume_usdt",
        "trades":                      "cex_klines_n_trades",
        "taker_buy_base_asset_volume": "cex_klines_taker_buy_usdc",
        "taker_buy_quote_asset_volume":"cex_klines_taker_buy_usdt",
    })
    out["cex_klines_taker_sell_usdc"]       = total_base - taker_buy                                           # counterpart to taker_buy...
    out["cex_klines_taker_buy_sell_ratio"]  = (taker_buy / total_base).where(total_base > 0)                   # directional pressure (> 0.5 more buying, < 0.5 more selling)
    return out


def load_cex_orderbook():
    # Orderbook data from local orderbook copy snapshots
    frames = []
    for path in sorted(Path(CEX_ORDERBOOK).glob("*-orderbook.csv")):
        df = pd.read_csv(path, parse_dates=["window_end"])
        frames.append(df)
    if not frames:
        print(f"  [CEX] WARNING: no files found in {CEX_ORDERBOOK}")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("window_end").set_index("window_end")
    out = out.rename(columns={
        "n_seconds":      "cex_orderbook_n_seconds",
        "spread_mean":    "cex_orderbook_spread_mean",
        "spread_std":     "cex_orderbook_spread_std",
        "imbalance_mean": "cex_orderbook_imbalance_mean",
        "imbalance_std":  "cex_orderbook_imbalance_std",
        "bid_depth_mean": "cex_orderbook_bid_depth_mean",
        "bid_depth_max":  "cex_orderbook_bid_depth_max",
        "bid_depth_min":  "cex_orderbook_bid_depth_min",
        "ask_depth_mean": "cex_orderbook_ask_depth_mean",
        "ask_depth_max":  "cex_orderbook_ask_depth_max",
        "ask_depth_min":  "cex_orderbook_ask_depth_min",
    })
    return out


# ── Dune Loaders ─────────────────────────────────────────────────────────────

def _dune_ts(series):
    # Cleaning the timestamp ('2026-03-06 00:05:00.000 UTC' -> datetime('2026-03-06 00:05:00'))
    return pd.to_datetime(series.str.replace(r"\.000 UTC$", "", regex=True))

def load_dune_whale_transfers():
    # Query 1: Whale Transfers
    path = Path(DUNE_DIR) / "q1_whale_transfers.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping")
        return pd.DataFrame()
    # Dune returns some data in "long format" (e.g. one line for Mints, one for Burns) thus use Pivot to convert long -> wide
    # CSV is already aggregated per (window_end, token, flow_direction) -> just fix long format
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    pivot = df.pivot_table(
        index="window_end",
        columns=["token", "flow_direction"],
        values=["total_usd", "transfer_count"],
        aggfunc="sum",
    )
    pivot.columns = [
        f"dune_whale_{tok.lower()}_{direction}_{metric}"
        for metric, tok, direction in pivot.columns
    ]
    return pivot


def load_dune_cex_flows():
    # Query 2: CEX Inflows and Outflows
    # This query has long output with multiple rows per timestamp (e.g. Binance, Kraken, OKX,...), another aggregation step is needed
    path = Path(DUNE_DIR) / "q2_cex_flows.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    # Lowercase flow_type so pivot column names are consistent (Inflow -> inflow)
    df["flow_type"] = df["flow_type"].str.lower()
    agg = df.groupby(["window_end", "token_symbol", "flow_type"], as_index=False).agg(
        total_usd      =("total_usd",       "sum"),
        transfer_count =("transfer_count",  "sum"),
        net_usd_flow   =("net_usd_flow",    "sum"),
    )
    pivot = agg.pivot_table(
        index="window_end",
        columns=["token_symbol", "flow_type"],
        values=["total_usd", "transfer_count"],
        aggfunc="sum",
    )
    pivot.columns = [
        f"dune_flows_{tok.lower()}_{ftype}_{metric}"
        for metric, tok, ftype in pivot.columns
    ]
    return pivot


def load_dune_gas_price():
    # Query 3: Gas Price Time Series (Base + Priority Fee) 
    path = Path(DUNE_DIR) / "q3_gas_price.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    df = df.set_index("window_end")
    rename = {
        "avg_base_fee_gwei":         "dune_gas_base_fee_gwei",
        "median_base_fee_gwei":      "dune_gas_base_fee_median_gwei",
        "min_base_fee_gwei":         "dune_gas_base_fee_min_gwei",
        "max_base_fee_gwei":         "dune_gas_base_fee_max_gwei",
        "priority_fee_p10_gwei":     "dune_gas_tip_p10_gwei",
        "priority_fee_p50_gwei":     "dune_gas_tip_p50_gwei",
        "priority_fee_p80_gwei":     "dune_gas_tip_p80_gwei",
        "gas_price_p10_gwei":        "dune_gas_price_p10_gwei",
        "gas_price_p50_gwei":        "dune_gas_price_p50_gwei",
        "gas_price_p80_gwei":        "dune_gas_price_p80_gwei",
        "approx_effective_gas_gwei": "dune_gas_effective_gwei",
        "tx_count":                  "dune_gas_tx_count",
        "block_count":               "dune_gas_block_count",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def load_dune_gas_used():
    # Query 4: Gas Used Per Block & Block Utilization 
    path = Path(DUNE_DIR) / "q4_gas_used.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    df = df.set_index("window_end")
    rename = {
        "block_count":             "dune_block_count",
        "avg_gas_used":            "dune_block_gas_used_mean",
        "min_gas_used":            "dune_block_gas_used_min",
        "max_gas_used":            "dune_block_gas_used_max",
        "median_gas_used":         "dune_block_gas_used_median",
        "avg_gas_limit":           "dune_block_gas_limit_mean",
        "avg_utilization":         "dune_block_utilization",
        "median_utilization":      "dune_block_utilization_median",
        "max_utilization":         "dune_block_utilization_max",
        "pct_blocks_above_target": "dune_block_pct_above_target",
        "pct_blocks_near_full":    "dune_block_pct_near_full",
        "avg_base_fee_gwei":       "dune_block_base_fee_gwei",
        "avg_blob_gas_used":       "dune_block_blob_gas_mean",
        "avg_block_size_bytes":    "dune_block_size_bytes_mean",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def load_dune_mempool():
    # Query 5: Mempool Congestion Proxies
    path = Path(DUNE_DIR) / "q5_mempool_congestion.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    df = df.set_index("window_end")
    rename = {
        "block_count":              "dune_mempool_block_count",
        "tx_count":                 "dune_mempool_tx_count",
        "avg_fill_ratio":           "dune_mempool_fill_ratio",
        "max_fill_ratio":           "dune_mempool_fill_ratio_max",
        "pct_blocks_near_full":     "dune_mempool_pct_near_full",
        "pct_blocks_above_target":  "dune_mempool_pct_above_target",
        "avg_base_fee_gwei":        "dune_mempool_base_fee_gwei",
        "base_fee_pct_change":      "dune_mempool_base_fee_change",
        "priority_fee_median_gwei": "dune_mempool_tip_median_gwei",
        "priority_fee_p80_gwei":    "dune_mempool_tip_p80_gwei",
        "priority_fee_spread_gwei": "dune_mempool_tip_spread_gwei",
        "avg_tx_gas_limit":         "dune_mempool_tx_gas_limit_mean",
        "avg_tx_gas_used":          "dune_mempool_tx_gas_used_mean",
        "congestion_score":         "dune_mempool_congestion_score",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def load_dune_supply_changes():
    # Query 6: On-Chain Mints and Burns
    path = Path(DUNE_DIR) / "q6_supply_changes.csv"
    if not path.exists():
        print(f"  [DUNE] WARNING: {path} not found -- skipping supply changes")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["window_end"] = _dune_ts(df["window_end"])
    pivot = df.pivot_table(
        index="window_end",
        columns=["token", "event_type"],
        values=["total_token_amount", "event_count", "cumulative_net_supply_delta"],
        aggfunc="sum",
    )
    pivot.columns = [
        f"dune_supply_{tok.lower()}_{etype}_{metric}"
        for metric, tok, etype in pivot.columns
    ]
    return pivot


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_data(data):
    df = data.copy()
    df.index = pd.to_datetime(df.index)                                                         # fixing a bug with the timestamps being in incorrect data type
    df = df.sort_index()   
    df['return'] = 10000 * np.log(df['dex_pool_price'] / df['dex_pool_price'].shift(1))         # Compute log return in basis points based on dex pool price
    df['gap'] = (df.index.to_series() - df.index.to_series().shift(1)) > pd.Timedelta('5min')   # Identify gaps — where consecutive timestamps are more than 5 minutes apart
    df.loc[df['gap'], 'return'] = np.nan                                                        # Blank out returns immediately following a gap
    df.iloc[0, df.columns.get_loc('return')] = np.nan                                           # Also blank out the first return (initialization)                                                     
    return df


# ── Merge ────────────────────────────────────────────────────────────────────

def build_5min():
    sources = [
        ("DEX klines",         load_dex_klines),
        ("DEX swaps raw",      load_dex_swaps_raw),
        ("DEX pool state",     load_dex_pool),
        ("DEX ticks",          load_dex_ticks),
        ("DEX mints/burns",    load_dex_mints_burns),
        ("CEX klines",         load_cex_klines),
        ("CEX orderbook",      load_cex_orderbook),
        ("DUNE gas price",     load_dune_gas_price),
        ("DUNE gas used",      load_dune_gas_used),
        ("DUNE mempool",       load_dune_mempool),
        ("DUNE supply chg",    load_dune_supply_changes),
        ("DUNE CEX flows",     load_dune_cex_flows),
        ("DUNE whale xfers",   load_dune_whale_transfers),
    ]
    loaded = []
    for name, loader in sources:                    # loader is function
        print(f"Loading {name}...")
        df = loader()
        if not df.empty:
            print(f"{len(df):>5} rows  x  {len(df.columns):>3} cols")
            loaded.append(df)
        else:
            print(f"(empty / skipped)")

    if not loaded:
        raise RuntimeError("No data loaded at all -- check paths.")

    merged = loaded[0]
    for df in loaded[1:]:
        merged = merged.join(df, how="outer")
    return clean_data(merged)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(AGGREGATE_OUTPUT, exist_ok=True)
    print("Building 5-min aggregated dataset...")
    df = build_5min()

    print(f"\n  Total 5-min rows : {len(df)}")
    print(f"  Time range       : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Total columns    : {len(df.columns)}")

    df.to_csv(AGGREGATED_DATA)
    print(f"\nSaved: {AGGREGATED_DATA}.")
