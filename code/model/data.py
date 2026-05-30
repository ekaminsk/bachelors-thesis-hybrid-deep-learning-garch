"""
Loading and preprocess the data set. Returns all data as a PyTorch tensor with chronological splits (train/val/test).
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import AGGREGATE_OUTPUT
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


# ── Feature columns (41 total) ───────────────────────────────────────────────

ZERO_FILL_COLS = [
    # DEX LP events (no swaps = 0 mints/burns)
    "dex_liquidityproviders_n_mints",
    "dex_liquidityproviders_n_burns",
    "dex_liquidityproviders_net_liq_change",
    # CEX flows (null = no transfer occurred)
    "dune_flows_usdc_inflow_total_usd",
    "dune_flows_usdc_outflow_total_usd",
    "dune_flows_usdt_inflow_total_usd",
    "dune_flows_usdt_outflow_total_usd",
    # Whale transfers
    "dune_whale_usdc_cex_inflow_total_usd",
    "dune_whale_usdc_cex_outflow_total_usd",
    "dune_whale_usdt_cex_inflow_total_usd",
    "dune_whale_usdt_cex_outflow_total_usd",
    # Protocol supply events
    "dune_supply_usdc_burn_total_token_amount",
    "dune_supply_usdc_mint_total_token_amount",
    "dune_supply_usdt_burn_blacklist_total_token_amount",
]

FEATURE_COLS = [
    # --- CEX (7) ---
    "cex_klines_volume_usdc",
    "cex_klines_n_trades",
    "cex_klines_taker_buy_sell_ratio",
    "cex_orderbook_imbalance_mean",
    "cex_orderbook_bid_depth_mean",
    "cex_orderbook_ask_depth_mean",
    "cex_orderbook_spread_mean",
    # --- DEX klines (5) ---
    "dex_klines_volume_usd",
    "dex_klines_n_swaps",
    "dex_klines_imbalance",
    "dex_klines_large_trades_count",
    "dex_klines_large_trades_usd",
    # --- DEX pool (2) ---
    "dex_pool_liquidity",
    "dex_pool_tvl_usd",
    # --- DEX ticks (4) ---
    "dex_ticks_total_liq_gross",
    "dex_ticks_net_liq_above",
    "dex_ticks_net_liq_below",
    "dex_ticks_n_active",
    # --- DEX LP (3) ---
    "dex_liquidityproviders_net_liq_change",
    "dex_liquidityproviders_n_mints",
    "dex_liquidityproviders_n_burns",
    # --- DUNE Gas (4) ---
    "dune_gas_base_fee_gwei",
    "dune_gas_tip_p50_gwei",
    "dune_gas_tip_p80_gwei",
    "dune_gas_effective_gwei",
    # --- DUNE Mempool / blocks (4) ---
    "dune_mempool_congestion_score",
    "dune_block_utilization",
    "dune_block_pct_near_full",
    "dune_mempool_base_fee_change",
    # --- DUNE Protocol supply (3) --- 
    "dune_supply_usdc_burn_total_token_amount",
    "dune_supply_usdc_mint_total_token_amount",
    "dune_supply_usdt_burn_blacklist_total_token_amount",
    # --- DUNE CEX flows (4) ---
    "dune_flows_usdc_inflow_total_usd",
    "dune_flows_usdc_outflow_total_usd",
    "dune_flows_usdt_inflow_total_usd",
    "dune_flows_usdt_outflow_total_usd",
    # --- DUNE Whale transfers (4) ---
    "dune_whale_usdc_cex_inflow_total_usd",
    "dune_whale_usdc_cex_outflow_total_usd",
    "dune_whale_usdt_cex_inflow_total_usd",
    "dune_whale_usdt_cex_outflow_total_usd",
    # --- Derived (1) --- (see further down)
    "dex_cex_price_spread",
]

DATA_PATH = Path(AGGREGATE_OUTPUT) / "aggregated_5min_data.csv"


# ── Loading data ─────────────────────────────────────────────────────────────
"""
- Load and clean(again) the data: load CSV into df, fill sparse colums with 0 (to remove NaN) compute dex_cex_price_spread, clip negative gas values
- Extract return to be the dependent: pull return column, identify gap boundaries, flag them
- Extract and fill features: pull 41 feature columns (FEATURE_COLS) after they are cleaned (if in ZERO_FILL_COLS)
- Split chronolgically: Split 70:10:20 -- train / val / test  
- Scale and tensorify: fit StandardScaler on training data, extend to all data, convert to PyTorch compatible tensor 
"""
def load_data(path: str = DATA_PATH, train_frac: float = 0.8, val_frac: float = 0.1):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")


    # ── Importing data ─────────────────────────────────────────────────────────
    
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)


    # ── derived feature: DEX-CEX price spread ────────────────────────────────

    df["dex_cex_price_spread"] = df["dex_pool_price"] - df["cex_klines_price_close"]


    # ── zero-fill DEX klines swap count (spec: no swaps = 0) ─────────────────

    if "dex_klines_n_swaps" in df.columns:
        df["dex_klines_n_swaps"] = df["dex_klines_n_swaps"].fillna(0.0)


    # ── clip priority fee tips to >= 0 (guard against negative values) ───────

    for col in ("dune_gas_tip_p50_gwei", "dune_gas_tip_p80_gwei"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)


    # ── returns (dependent variable) ─────────────────────────────────────────

    returns = df["return"].values.astype(np.float32)
    gap_mask = np.isnan(returns)                                    # True at gap boundaries
    returns_filled = np.where(gap_mask, 0.0, returns)               # NaN -> 0 for tensor


    # ── features ─────────────────────────────────────────────────────────────

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)

    # Remaining NaNs in features: forward-fill then fill remaining with 0 -> defensive, should never occur in practice because of ZERO_FILL_COLS
    X_df = pd.DataFrame(X, columns=FEATURE_COLS).ffill().fillna(0.0)
    X = X_df.values.astype(np.float32)

    length = len(df)
    train_end = int(train_frac * length)                            # e.g., 80% -> test boundary
    val_start = train_end - int(val_frac * length)                  # last val_frac of training window

    print(f"Total timesteps : {length}")
    print(f"Train window    : [0, {train_end})  ({train_end} rows)")
    print(f"Val window      : [{val_start}, {train_end})  ({train_end - val_start} rows)")
    print(f"Test window     : [{train_end}, {length})  ({length - train_end} rows)")
    print(f"Gap boundaries  : {gap_mask.sum()} rows with NaN return")


    # ── StandardScaler fitted on training window only ────────────────────────

    scaler = StandardScaler()                                       # required to make inputs comparable by fitting to mean 0 std 1
    X[:train_end] = scaler.fit_transform(X[:train_end])
    X[train_end:] = scaler.transform(X[train_end:])

    
    # ── convert to tensors ───────────────────────────────────────────────────

    X_t = torch.tensor(X, dtype=torch.float32)
    r_t = torch.tensor(returns_filled, dtype=torch.float32)
    gap_t = torch.tensor(gap_mask, dtype=torch.bool)

    timestamps = (
        df["window_end"].tolist() if "window_end" in df.columns else None           # for outputing training data later (so you can see with which value what happened)
    )

    return {
        "X_all": X_t,
        "r_all": r_t,
        "gap_mask": gap_t,
        "train_end": train_end,
        "val_start": val_start,
        "length": length,
        "scaler": scaler,
        "timestamps": timestamps,
        "feature_cols": FEATURE_COLS,
    }


if __name__ == "__main__":
    d = load_data()
    print("\nFeature tensor shape:", d["X_all"].shape)
    print("Return tensor shape :", d["r_all"].shape)
    print("NaN in X_all        :", torch.isnan(d["X_all"]).sum().item())
