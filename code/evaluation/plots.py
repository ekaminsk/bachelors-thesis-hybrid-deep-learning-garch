"""
These plots are the plots used in the thesis document itself.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import PLOTS_DIR, AGGREGATED_DATA, GARCH_OUTPUT, GARCH_BASELINE_SIGMA2, GARCH_BASELINE_PARAMS, RESULTS_DIR

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(AGGREGATED_DATA)
df["ts"] = pd.to_datetime(df["window_end"])
returns   = df["return"].values                                 # NaN at gaps
timestamps = df["ts"].values                                    # numpy datetime64
date_from = pd.Timestamp(timestamps[0]).strftime("%Y-%m-%d")    # first timestamp
date_to   = pd.Timestamp(timestamps[-1]).strftime("%Y-%m-%d")   # last timestamp



# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Return series with gap markers
# ─────────────────────────────────────────────────────────────────────────────
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

    # One invisible line just for the legend entry (guard in case there are no gaps)
    if len(gap_ts) > 0:
        ax.axvline(gap_ts[0], color="#b0b0b0", linewidth=0.8,
                linestyle=":", alpha=0.7, label=f"Missing observation (n={gap_mask.sum()})",
                zorder=2)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4, zorder=1)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Return (basis points)", fontsize=10)
    ax.set_title("5-Minute Log-Returns — USDC/USDT DEX Pool\n"
                 f"{date_from} to {date_to}", fontsize=11)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(timestamps[0], timestamps[-1])

    savefig("plot1_return_series.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Conditional variance vs realised volatility (single plot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_conditional_variance():
    print("\n[2] Conditional variance vs 5-min realized variance (r_t^2)")

    # ── 5-min realized variance ──────────────────────────────────────────────
    r2_raw = returns ** 2   # NaN at gaps, 0 where no price movement

    # ── hybrid sigma² ────────────────────────────────────────────────────────
    garchoutput = pd.read_csv(GARCH_OUTPUT, parse_dates=["window_end"])                      # go = garch output
    garchoutput = garchoutput.rename(columns={"window_end": "ts"})
    hybrid_ts   = garchoutput["ts"].values
    hybrid_sigma2   = garchoutput["sigma2"].values
    train_end_ts = garchoutput.loc[garchoutput["split"] == "test", "ts"].iloc[0]

    # ── standard GARCH sigma² (training window, NaN beyond) ──────────────────
    baseline = pd.read_csv(GARCH_BASELINE_SIGMA2,
                     parse_dates=["window_end"])
    baseline = baseline.rename(columns={"window_end": "ts"})

    baselinel_sigma2 = np.full(len(garchoutput), np.nan)
    baselinel_sigma2[:len(baseline)] = baseline["sigma2_garch"].values
    baseline_valid = np.isfinite(baselinel_sigma2)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # r_t^2 as scatter dots — only plot nonzero / non-NaN values
    valid_rv = np.isfinite(r2_raw) & (r2_raw > 0)
    ax.scatter(timestamps[valid_rv], r2_raw[valid_rv],
               color="#e05c00", s=3, alpha=0.55, linewidths=0,
               zorder=2, label="$r_t^2$  (5-min realized variance)")

    # Standard GARCH (only where not NaN — training window)
    ax.plot(hybrid_ts[baseline_valid], baselinel_sigma2[baseline_valid],
            color="#2c6e49", linewidth=1.1, alpha=0.9,
            label="Standard GARCH(1,1)  $\\hat{\\sigma}_t^2$", zorder=3)

    # Hybrid GARCH-NN (full series)
    ax.plot(hybrid_ts, hybrid_sigma2,
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
        f"USDC/USDT DEX Pool: {date_from} to {date_to}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — One-step-ahead test-window forecasts + error metrics
# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast_evaluation():
    print("\n[3] One-step-ahead forecast evaluation — test window")

    # ── load hybrid output ───────────────────────────────────────────────────
    garchoutput = pd.read_csv(GARCH_OUTPUT, parse_dates=["window_end"])
    garchoutput = garchoutput.rename(columns={"window_end": "ts"})                    # just for consistency

    train_end = garchoutput[garchoutput["split"] == "test"].index[0]                  # find where train_end is (-> train.py)

    # Hybrid test-window sigma2 (these ARE one-step-ahead: sigma2_t uses r_{t-1})
    hybrid_test  = garchoutput[garchoutput["split"] == "test"].copy().reset_index(drop=True)
    hybrid_sigma2    = hybrid_test["sigma2"].values
    test_ts      = hybrid_test["ts"].values

    # ── build standard GARCH one-step-ahead forecasts for test window ────────
    baseline = pd.read_csv(GARCH_BASELINE_SIGMA2, parse_dates=["window_end"])
    baseline_params = pd.read_csv(GARCH_BASELINE_PARAMS)
    alpha_g = float(baseline_params.loc[baseline_params["param"] == "alpha[1]", "estimate"].iloc[0])
    beta_g  = float(baseline_params.loc[baseline_params["param"] == "beta[1]",  "estimate"].iloc[0])
    omega_g = float(baseline_params.loc[baseline_params["param"] == "omega",    "estimate"].iloc[0])

    # Initial state: last valid sigma2 and last return from training window
    # last_sigma2 is the starting sigma^2 -> final sigma^2 the GARCH developed (dropna in case its NaN)
    last_sigma2  = float(baseline["sigma2_garch"].dropna().iloc[-1])
    # last_return2 is either end-1 return^2 or if NaN then the last_sigma2 (because E(r^2) = E(sigma^2) (zero mean assumption)) 
    last_return2  = float(returns[train_end - 1] ** 2) if not np.isnan(returns[train_end - 1]) else last_sigma2

    gap_mask_full = np.isnan(returns)
    garch_sigma2_test = []
    sigma2_prev = last_sigma2               # to get overriden each iteration
    return2_prev = last_return2             # to get overriden each iteration

    for i in range(train_end, len(returns)):       # loop only test window
        if gap_mask_full[i]:
            # Gap: reset to unconditional variance, don't update state with NaN
            sigma2_t   = omega_g / (1.0 - alpha_g - beta_g)
            sigma2_prev = sigma2_t
            return2_prev = sigma2_t         # E[r^2] = E[sigma^2] at restart
        else:                               # normal step of GARCH
            sigma2_t    = omega_g + alpha_g * return2_prev + beta_g * sigma2_prev
            sigma2_prev = sigma2_t
            return2_prev = float(returns[i] ** 2)

        garch_sigma2_test.append(sigma2_t)              # result appended to list

    garch_sigma2_test = np.array(garch_sigma2_test)     # make numpy array

    # ── realized variance proxy: r_t^2 ───────────────────────────────────────
    return2_test = returns[train_end:] ** 2   # NaN at gaps
    
    # ── metrics (exclude gap rows where r²=NaN) ────────────────────────────── 
    valid = np.isfinite(return2_test)

    def metrics(pred, actual, mask):
        e  = pred[mask] - actual[mask]
        mse  = float(np.mean(e ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(e)))
        return mse, rmse, mae

    mse_g,  rmse_g,  mae_g  = metrics(garch_sigma2_test, return2_test, valid)
    mse_h,  rmse_h,  mae_h  = metrics(hybrid_sigma2,     return2_test, valid)

    print(f"  Standard GARCH  — MSE={mse_g:.6f}  RMSE={rmse_g:.6f}  MAE={mae_g:.6f}")
    print(f"  Hybrid GARCH-NN — MSE={mse_h:.6f}  RMSE={rmse_h:.6f}  MAE={mae_h:.6f}")

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {"model": "Standard GARCH(1,1)", "MSE": mse_g, "RMSE": rmse_g, "MAE": mae_g},
        {"model": "Hybrid GARCH-NN",     "MSE": mse_h, "RMSE": rmse_h, "MAE": mae_h},
    ])
    metrics_path = os.path.join(RESULTS_DIR, "plot3_forecast_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  saved: {metrics_path}")

    # Save forecast series CSV
    forecast_df = pd.DataFrame({
        "window_end":       test_ts,
        "r2_realized":      return2_test,
        "sigma2_garch":     garch_sigma2_test,
        "sigma2_hybrid":    hybrid_sigma2,
    })
    fcast_path = os.path.join(RESULTS_DIR, "plot3_forecast_series.csv")
    forecast_df.to_csv(fcast_path, index=False)
    print(f"  saved: {fcast_path}")

    # ── plot: test window only ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))

    # Realized variance (r_t^2)
    valid_rv = np.isfinite(return2_test) & (return2_test > 0)
    ax.scatter(test_ts[valid_rv], return2_test[valid_rv],
               color="#e05c00", s=4, alpha=0.6, linewidths=0,
               zorder=2, label="$r_t^2$  (realized variance)")

    # Standard GARCH forecast
    ax.plot(test_ts, garch_sigma2_test,
            color="#2c6e49", linewidth=1.2, alpha=0.9, zorder=3,
            label=f"Standard GARCH(1,1)  $\\hat{{\\sigma}}_t^2$"
                  f"  [RMSE={rmse_g:.4f}, MAE={mae_g:.4f}]")

    # Hybrid GARCH-NN forecast
    ax.plot(test_ts, hybrid_sigma2,
            color="#1f3d6e", linewidth=1.2, alpha=0.9, zorder=4,
            label=f"Hybrid GARCH-NN  $\\hat{{\\sigma}}_t^2$"
                  f"  [RMSE={rmse_h:.4f}, MAE={mae_h:.4f}]")

    ax.set_yscale("log")
    ax.set_ylabel("Variance, log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    
    test_date_from = pd.Timestamp(test_ts[0]).strftime("%Y-%m-%d")
    test_date_to   = pd.Timestamp(test_ts[-1]).strftime("%Y-%m-%d")
    
    ax.set_title(
        "One-Step-Ahead Variance Forecasts vs Realized Variance — Test Window\n"
        f"USDC/USDT DEX Pool: {test_date_from} to {test_date_to}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — GARCH benchmark conditional variance only
# ─────────────────────────────────────────────────────────────────────────────
def plot_garch_benchmark():
    print("\n[4] GARCH benchmark conditional variance")

    baseline = pd.read_csv(GARCH_BASELINE_SIGMA2, parse_dates=["window_end"])
    baseline = baseline.rename(columns={"window_end": "ts"})

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(baseline["ts"], baseline["sigma2_garch"],
            color="#2c6e49", linewidth=0.8, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("$\\hat{\\sigma}_t^2$ — log scale (bp²)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        "Standard GARCH(1,1) — Conditional Variance Estimates\n"
        "USDC/USDT DEX Pool: Training Window",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(baseline["ts"].iloc[0], baseline["ts"].iloc[-1])

    plt.tight_layout()
    savefig("plot4_garch_benchmark.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all plots
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_return_series()
    plot_conditional_variance()
    plot_forecast_evaluation()
    plot_garch_benchmark()
    print("\nDone.")
