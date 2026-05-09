"""
plots.py — thesis figures for model results 8.

Run:  py plots.py
All figures saved to D:\\data\\model\\results 8\\
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
RESULTS_DIR = r"D:\data\model\results 8"
DATA_PATH   = r"D:\data\model\thesis_5min_edited.csv"
GARCH_OUT   = r"D:\data\model\results 8\garch_output.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")
df["ts"] = pd.to_datetime(df["window_end (UTC)"])
returns   = df["return_basis_points"].values          # NaN at gaps
timestamps = df["ts"].values                           # numpy datetime64


# ===========================================================================
# Plot 1 — Return series with gap markers
# ===========================================================================
def plot_return_series():
    print("\n[1] Return series")

    gap_mask = np.isnan(returns)
    gap_ts   = timestamps[gap_mask]

    fig, ax = plt.subplots(figsize=(14, 4))

    # Main return line — NaN values create natural breaks
    ax.plot(timestamps, returns,
            color="#1f3d6e", linewidth=0.55, alpha=0.85, zorder=3)

    # Subtle vertical markers at gap locations
    for gt in gap_ts:
        ax.axvline(gt, color="#b0b0b0", linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=2)

    # One invisible line just for the legend entry
    ax.axvline(gap_ts[0], color="#b0b0b0", linewidth=0.8,
               linestyle=":", alpha=0.7, label=f"Missing observation (n={gap_mask.sum()})",
               zorder=2)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4, zorder=1)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Return (basis points)", fontsize=10)
    ax.set_title("5-Minute Log-Returns — USDC/USDT DEX Pool\n"
                 "2026-03-06 to 2026-03-17", fontsize=11)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(timestamps[0], timestamps[-1])

    savefig("plot1_return_series.png")


# ===========================================================================
# Plot 2 — Conditional variance vs realised volatility (single plot)
# ===========================================================================
def plot_conditional_variance():
    print("\n[2] Conditional variance vs 5-min realized variance (r_t^2)")

    # ---- 5-min realized variance = r_t^2 --------------------------------
    r2_raw = returns ** 2   # NaN at gaps, 0 where no price movement

    # ---- hybrid sigma² (full series) ------------------------------------
    go = pd.read_csv(GARCH_OUT, parse_dates=["window_end"])
    go = go.rename(columns={"window_end": "ts"})
    hybrid_ts   = go["ts"].values
    hybrid_s2   = go["sigma2"].values
    train_end_ts = go.loc[go["split"] == "test", "ts"].iloc[0]

    # ---- standard GARCH sigma² (training window, NaN beyond) -----------
    bl = pd.read_csv(r"D:\data\model\results 8\garch_baseline_sigma2.csv",
                     parse_dates=["window_end"])
    bl = bl.rename(columns={"window_end": "ts"})
    bl_full = pd.DataFrame({"ts": go["ts"]}).merge(
        bl[["ts", "sigma2_garch"]], on="ts", how="left"
    )
    bl_s2 = bl_full["sigma2_garch"].values

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # r_t^2 as scatter dots — only plot nonzero / non-NaN values
    valid_rv = np.isfinite(r2_raw) & (r2_raw > 0)
    ax.scatter(timestamps[valid_rv], r2_raw[valid_rv],
               color="#e05c00", s=3, alpha=0.55, linewidths=0,
               zorder=2, label="$r_t^2$  (5-min realized variance)")

    # Standard GARCH (only where not NaN — training window)
    bl_valid = np.isfinite(bl_s2)
    ax.plot(hybrid_ts[bl_valid], bl_s2[bl_valid],
            color="#2c6e49", linewidth=1.1, alpha=0.9,
            label="Standard GARCH(1,1)  $\\hat{\\sigma}_t^2$", zorder=3)

    # Hybrid GARCH-NN (full series)
    ax.plot(hybrid_ts, hybrid_s2,
            color="#1f3d6e", linewidth=1.1, alpha=0.9,
            label="Hybrid GARCH-NN  $\\hat{\\sigma}_t^2$", zorder=4)

    # Train / test boundary
    ax.axvline(pd.Timestamp(train_end_ts), color="black", linewidth=1.0,
               linestyle="--", alpha=0.55, zorder=5, label="Train / test split")

    ax.set_yscale("log")
    ax.set_ylabel("Variance, log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        "Conditional Variance vs 5-min Realized Variance ($r_t^2$)\n"
        "USDC/USDT DEX Pool · 2026-03-06 to 2026-03-17",
        fontsize=11,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)

    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_xlim(timestamps[0], timestamps[-1])

    plt.tight_layout()
    savefig("plot2_conditional_variance.png")


# ===========================================================================
# Plot 3 — One-step-ahead test-window forecasts + error metrics
# ===========================================================================
def plot_forecast_evaluation():
    print("\n[3] One-step-ahead forecast evaluation — test window")

    # ---- full data -------------------------------------------------------
    df_full = pd.read_csv(DATA_PATH, sep=";")
    df_full["ts"] = pd.to_datetime(df_full["window_end (UTC)"])
    r_full   = df_full["return_basis_points"].values.astype(float)
    ts_full  = df_full["ts"].values

    # ---- load hybrid output (has test split already) --------------------
    go = pd.read_csv(GARCH_OUT, parse_dates=["window_end"])
    go = go.rename(columns={"window_end": "ts"})

    train_end = go[go["split"] == "test"].index[0]   # integer index into go

    # Hybrid test-window sigma2 (these ARE one-step-ahead: sigma2_t uses r_{t-1})
    hybrid_test  = go[go["split"] == "test"].copy().reset_index(drop=True)
    hybrid_s2    = hybrid_test["sigma2"].values
    test_ts      = hybrid_test["ts"].values

    # ---- build standard GARCH one-step-ahead forecasts for test window --
    # Parameters (fallback): alpha=0.05, beta=0.90
    # omega derived from empirical median of training sigma2 (avoids rescaling ambiguity)
    bl = pd.read_csv(r"D:\data\model\results 8\garch_baseline_sigma2.csv",
                     parse_dates=["window_end"])
    alpha_g = 0.05
    beta_g  = 0.90
    omega_g = float(bl["sigma2_garch"].median()) * (1.0 - alpha_g - beta_g)

    # Initial state: last valid sigma2 and last return from training window
    last_s2  = float(bl["sigma2_garch"].dropna().iloc[-1])
    last_r2  = float(r_full[train_end - 1] ** 2) if not np.isnan(r_full[train_end - 1]) else last_s2

    gap_mask_full = np.isnan(r_full)
    garch_s2_test = []
    s2_prev = last_s2
    r2_prev = last_r2

    for i in range(train_end, len(r_full)):
        if gap_mask_full[i]:
            # Gap: reset to unconditional variance, don't update state with NaN
            s2_t   = omega_g / (1.0 - alpha_g - beta_g)
            s2_prev = s2_t
            r2_prev = s2_t          # E[r²] = E[σ²] at restart
        else:
            s2_t    = omega_g + alpha_g * r2_prev + beta_g * s2_prev
            s2_prev = s2_t
            r2_prev = float(r_full[i] ** 2)

        garch_s2_test.append(s2_t)

    garch_s2_test = np.array(garch_s2_test)

    # ---- realized variance proxy: r_t^2 ---------------------------------
    r2_test = r_full[train_end:] ** 2   # NaN at gaps

    # ---- metrics (exclude gap rows where r²=NaN) -------------------------
    valid = np.isfinite(r2_test)

    def metrics(pred, actual, mask):
        e  = pred[mask] - actual[mask]
        mse  = float(np.mean(e ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(e)))
        return mse, rmse, mae

    mse_g,  rmse_g,  mae_g  = metrics(garch_s2_test, r2_test, valid)
    mse_h,  rmse_h,  mae_h  = metrics(hybrid_s2,     r2_test, valid)

    print(f"  Standard GARCH  — MSE={mse_g:.6f}  RMSE={rmse_g:.6f}  MAE={mae_g:.6f}")
    print(f"  Hybrid GARCH-NN — MSE={mse_h:.6f}  RMSE={rmse_h:.6f}  MAE={mae_h:.6f}")

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {"model": "Standard GARCH(1,1)", "MSE": mse_g, "RMSE": rmse_g, "MAE": mae_g},
        {"model": "Hybrid GARCH-NN",     "MSE": mse_h, "RMSE": rmse_h, "MAE": mae_h},
    ])
    metrics_path = os.path.join(RESULTS_DIR, "forecast_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  saved: {metrics_path}")

    # Save forecast series CSV
    forecast_df = pd.DataFrame({
        "window_end":       test_ts,
        "r2_realized":      r2_test,
        "sigma2_garch":     garch_s2_test,
        "sigma2_hybrid":    hybrid_s2,
    })
    fcast_path = os.path.join(RESULTS_DIR, "forecast_series.csv")
    forecast_df.to_csv(fcast_path, index=False)
    print(f"  saved: {fcast_path}")

    # ---- plot: test window only -----------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))

    # Realized variance (r_t^2)
    valid_rv = np.isfinite(r2_test) & (r2_test > 0)
    ax.scatter(test_ts[valid_rv], r2_test[valid_rv],
               color="#e05c00", s=4, alpha=0.6, linewidths=0,
               zorder=2, label="$r_t^2$  (realized variance)")

    # Standard GARCH forecast
    ax.plot(test_ts, garch_s2_test,
            color="#2c6e49", linewidth=1.2, alpha=0.9, zorder=3,
            label=f"Standard GARCH(1,1)  $\\hat{{\\sigma}}_t^2$"
                  f"  [RMSE={rmse_g:.4f}, MAE={mae_g:.4f}]")

    # Hybrid GARCH-NN forecast
    ax.plot(test_ts, hybrid_s2,
            color="#1f3d6e", linewidth=1.2, alpha=0.9, zorder=4,
            label=f"Hybrid GARCH-NN  $\\hat{{\\sigma}}_t^2$"
                  f"  [RMSE={rmse_h:.4f}, MAE={mae_h:.4f}]")

    ax.set_yscale("log")
    ax.set_ylabel("Variance, log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        "One-Step-Ahead Variance Forecasts vs Realized Variance — Test Window\n"
        "USDC/USDT DEX Pool · 2026-03-14 to 2026-03-17",
        fontsize=11,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)

    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_xlim(test_ts[0], test_ts[-1])

    plt.tight_layout()
    savefig("plot3_forecast_evaluation.png")


# ===========================================================================
# Plot 4 — Same evaluation but using Run 3 hybrid model
# ===========================================================================
def plot_forecast_run3():
    print("\n[4] One-step-ahead forecast evaluation — Run 3 hybrid vs GARCH benchmark")

    df_full = pd.read_csv(DATA_PATH, sep=";")
    df_full["ts"] = pd.to_datetime(df_full["window_end (UTC)"])
    r_full      = df_full["return_basis_points"].values.astype(float)
    gap_mask    = np.isnan(r_full)

    # ---- Run 3 hybrid test sigma2 ----------------------------------------
    go3      = pd.read_csv(r"D:\data\model\results 3\garch_output.csv",
                           parse_dates=["window_end"])
    test3    = go3[go3["split"] == "test"].reset_index(drop=True)
    hybrid_s2 = test3["sigma2"].values
    test_ts   = test3["window_end"].values
    train_end = go3[go3["split"] == "test"].index[0]

    # ---- Same GARCH baseline as plot 3 -----------------------------------
    bl       = pd.read_csv(r"D:\data\model\results 8\garch_baseline_sigma2.csv",
                           parse_dates=["window_end"])
    alpha_g  = 0.05
    beta_g   = 0.90
    omega_g  = float(bl["sigma2_garch"].median()) * (1.0 - alpha_g - beta_g)

    last_s2  = float(bl["sigma2_garch"].dropna().iloc[-1])
    last_r2  = float(r_full[train_end - 1] ** 2) if not np.isnan(r_full[train_end - 1]) else last_s2

    garch_s2_test = []
    s2_prev, r2_prev = last_s2, last_r2
    for i in range(train_end, len(r_full)):
        if gap_mask[i]:
            s2_t    = omega_g / (1.0 - alpha_g - beta_g)
            s2_prev = s2_t
            r2_prev = s2_t
        else:
            s2_t    = omega_g + alpha_g * r2_prev + beta_g * s2_prev
            s2_prev = s2_t
            r2_prev = float(r_full[i] ** 2)
        garch_s2_test.append(s2_t)
    garch_s2_test = np.array(garch_s2_test)

    # ---- realized variance -----------------------------------------------
    r2_test = r_full[train_end:] ** 2
    valid   = np.isfinite(r2_test)

    def metrics(pred, actual, mask):
        e = pred[mask] - actual[mask]
        mse  = float(np.mean(e ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(e)))
        return mse, rmse, mae

    mse_g,  rmse_g,  mae_g  = metrics(garch_s2_test, r2_test, valid)
    mse_h,  rmse_h,  mae_h  = metrics(hybrid_s2,     r2_test, valid)

    print(f"  Standard GARCH  — MSE={mse_g:.6f}  RMSE={rmse_g:.6f}  MAE={mae_g:.6f}")
    print(f"  Run 3 Hybrid    — MSE={mse_h:.6f}  RMSE={rmse_h:.6f}  MAE={mae_h:.6f}")

    # Save metrics
    metrics_df = pd.DataFrame([
        {"model": "Standard GARCH(1,1)", "MSE": mse_g, "RMSE": rmse_g, "MAE": mae_g},
        {"model": "Hybrid GARCH-NN (Run 3)", "MSE": mse_h, "RMSE": rmse_h, "MAE": mae_h},
    ])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "forecast_metrics_run3.csv"), index=False)

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))

    valid_rv = np.isfinite(r2_test) & (r2_test > 0)
    ax.scatter(test_ts[valid_rv], r2_test[valid_rv],
               color="#e05c00", s=4, alpha=0.6, linewidths=0,
               zorder=2, label="$r_t^2$  (realized variance)")

    ax.plot(test_ts, garch_s2_test,
            color="#2c6e49", linewidth=1.2, alpha=0.9, zorder=3,
            label=f"Standard GARCH(1,1)  [RMSE={rmse_g:.4f}, MAE={mae_g:.4f}]")

    ax.plot(test_ts, hybrid_s2,
            color="#1f3d6e", linewidth=1.2, alpha=0.9, zorder=4,
            label=f"Hybrid GARCH-NN Run 3  [RMSE={rmse_h:.4f}, MAE={mae_h:.4f}]")

    ax.set_yscale("log")
    ax.set_ylabel("Variance, log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        "One-Step-Ahead Variance Forecasts vs Realized Variance — Run 3 Hybrid\n"
        "USDC/USDT DEX Pool · Test Window (2026-03-14 to 2026-03-17)",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_xlim(test_ts[0], test_ts[-1])

    plt.tight_layout()
    savefig("plot4_forecast_run3.png")


# ===========================================================================
# Plot 5 — GARCH benchmark conditional variance only
# ===========================================================================
def plot_garch_benchmark():
    print("\n[5] GARCH benchmark conditional variance")

    bl = pd.read_csv(r"D:\data\model\results 8\garch_baseline_sigma2.csv",
                     parse_dates=["window_end"])
    bl = bl.rename(columns={"window_end": "ts"})

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(bl["ts"], bl["sigma2_garch"],
            color="#2c6e49", linewidth=0.8, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("$\\hat{\\sigma}_t^2$ — log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        "Standard GARCH(1,1) — Conditional Variance Estimates\n"
        "USDC/USDT DEX Pool · Training Window",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(bl["ts"].iloc[0], bl["ts"].iloc[-1])

    plt.tight_layout()
    savefig("plot5_garch_benchmark.png")


# ===========================================================================
# Run all plots
# ===========================================================================
if __name__ == "__main__":
    plot_return_series()
    plot_conditional_variance()
    plot_forecast_evaluation()
    plot_forecast_run3()
    plot_garch_benchmark()
    print("\nDone.")
