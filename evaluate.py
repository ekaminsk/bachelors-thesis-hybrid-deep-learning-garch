"""
evaluate.py — post-training diagnostics for the GARCH-NN model.

Reads:
  D:\data\model\training_results.txt   — epoch loss history + summary
  D:\data\model\garch_output.csv       — per-timestep sigma2, alpha_t, beta_t, split
  D:\data\model\best_model.pt          — trained model weights

Generates (all saved to D:\data\model\results\):
  loss_curve.png / loss_curve.csv      — train/val NLL per epoch
  acf_residuals.png                    — ACF of std. residuals (train window)
  acf_sq_residuals.png                 — ACF of squared std. residuals (train window)
  residuals.csv                        — eps_t, eps_t^2, sigma2_t for train window
  alpha_beta_stats.csv                 — per-split summary stats for alpha_t, beta_t
  permutation_importance.png           — permutation-based feature importance
  permutation_importance.csv           — numerical importance scores

Note on standardised residuals:
  eps_t = r_t / sqrt(sigma2_t).  r_t and sigma2_t are both available without
  re-running the model:  r_t comes from the raw CSV, sigma2_t from garch_output.csv.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")        # headless — no display needed
import matplotlib.pyplot as plt

# Make sure model/ is on the path when called from elsewhere
sys.path.insert(0, os.path.dirname(__file__))
from data  import load_data, FEATURE_COLS
from model import GARCHNet

# ---------------------------------------------------------------------------
RESULTS_DIR      = r"D:\data\model\results"
TRAIN_RESULTS    = r"D:\data\model\results\training_results.txt"
GARCH_OUTPUT     = r"D:\data\model\results\garch_output.csv"
BEST_MODEL       = r"D:\data\model\results\best_model.pt"
MODEL_CFG        = r"D:\data\model\results\model_config.json"
N_PERM_REPS      = 10    # permutation repetitions per feature (more = less noisy)
ACF_NLAGS        = 40    # lags to show in ACF plots
DEVICE           = torch.device("cpu")


os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(name: str):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def acf_series(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample ACF from lag 0 to nlags (inclusive)."""
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    if c0 == 0:
        return np.zeros(nlags + 1)
    vals = [1.0]
    for k in range(1, nlags + 1):
        ck = np.dot(x[:n - k], x[k:]) / n
        vals.append(ck / c0)
    return np.array(vals)


def plot_acf(acf_vals: np.ndarray, title: str, filename: str, n_obs: int):
    lags = np.arange(len(acf_vals))
    ci = 1.96 / np.sqrt(n_obs)           # 95% Bartlett confidence band

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags[1:], acf_vals[1:], color="#3a7ebf", width=0.6, label="ACF")
    ax.axhline(ci,  color="red", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0,   color="black", linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(acf_vals))
    savefig(filename)


# ===========================================================================
# 1. Loss curve
# ===========================================================================
def plot_loss_curve():
    print("\n[1] Loss curve")

    # Parse training_results.txt — CSV section starts after the header line
    rows = []
    in_csv = False
    with open(TRAIN_RESULTS) as f:
        for line in f:
            line = line.strip()
            if line == "Epoch,TrainNLL,ValNLL":
                in_csv = True
                continue
            if in_csv and line:
                parts = line.split(",")
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))

    if not rows:
        print("  WARNING: no epoch data found in training_results.txt — skipping")
        return

    df = pd.DataFrame(rows, columns=["epoch", "train_nll", "val_nll"])

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "loss_curve.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["epoch"], df["train_nll"], label="Train NLL", linewidth=1.2)
    ax.plot(df["epoch"], df["val_nll"],   label="Val NLL",   linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean NLL (basis points²)")
    ax.set_title("GARCH-NN Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("loss_curve.png")


# ===========================================================================
# 2. Standardised residuals + ACF
# ===========================================================================
def compute_residuals(data: dict, garch_df: pd.DataFrame):
    """
    Returns residuals DataFrame for the train window only.
    eps_t = r_t / sqrt(sigma2_t), using gap_mask to exclude NaN returns.
    """
    train_end  = data["train_end"]
    gap_mask   = data["gap_mask"].numpy().astype(bool)
    r_all      = data["r_all"].numpy()

    sigma2_all = garch_df["sigma2"].values

    # Work on train window
    r_tr     = r_all[:train_end]
    sigma2_tr = sigma2_all[:train_end]
    gap_tr   = gap_mask[:train_end]

    valid = ~gap_tr
    r_v   = r_tr[valid]
    s2_v  = sigma2_tr[valid]

    eps   = r_v / np.sqrt(s2_v)
    eps2  = eps ** 2

    df_res = pd.DataFrame({
        "r_t":     r_v,
        "sigma2_t": s2_v,
        "eps_t":   eps,
        "eps2_t":  eps2,
    })
    return df_res


def plot_residuals(data: dict, garch_df: pd.DataFrame):
    print("\n[2] Standardised residuals & ACF")

    df_res = compute_residuals(data, garch_df)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "residuals.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    eps  = df_res["eps_t"].values
    eps2 = df_res["eps2_t"].values
    n    = len(eps)

    # ACF of standardised residuals
    acf_eps  = acf_series(eps,  ACF_NLAGS)
    acf_eps2 = acf_series(eps2, ACF_NLAGS)

    plot_acf(
        acf_eps,
        title="ACF of Standardised Residuals  (train window)",
        filename="acf_residuals.png",
        n_obs=n,
    )
    plot_acf(
        acf_eps2,
        title="ACF of Squared Standardised Residuals  (train window)",
        filename="acf_sq_residuals.png",
        n_obs=n,
    )

    # Save ACF values as CSV
    acf_df = pd.DataFrame({
        "lag":      np.arange(ACF_NLAGS + 1),
        "acf_eps":  acf_eps,
        "acf_eps2": acf_eps2,
    })
    acf_csv = os.path.join(RESULTS_DIR, "acf_values.csv")
    acf_df.to_csv(acf_csv, index=False)
    print(f"  saved: {acf_csv}")

    # Quick descriptive stats
    print(f"  eps_t  — mean={eps.mean():.4f}  std={eps.std():.4f}  "
          f"skew={_skew(eps):.4f}  kurt={_kurt(eps):.4f}")
    print(f"  eps2_t — mean={eps2.mean():.4f}  std={eps2.std():.4f}")


def _skew(x):
    m = x.mean(); s = x.std()
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else 0.0

def _kurt(x):
    m = x.mean(); s = x.std()
    return float(np.mean(((x - m) / s) ** 4)) - 3.0 if s > 0 else 0.0


# ===========================================================================
# 3. alpha_t / beta_t summary statistics
# ===========================================================================
def alpha_beta_stats(garch_df: pd.DataFrame):
    print("\n[3] alpha_t / beta_t summary stats")

    rows = []
    for split in ("train", "val", "test"):
        sub = garch_df[garch_df["split"] == split]
        if sub.empty:
            continue
        for col in ("alpha_t", "beta_t"):
            v = sub[col].values
            rows.append({
                "split": split,
                "param": col,
                "mean":  v.mean(),
                "std":   v.std(),
                "min":   v.min(),
                "p25":   np.percentile(v, 25),
                "p50":   np.percentile(v, 50),
                "p75":   np.percentile(v, 75),
                "max":   v.max(),
            })
        # persistence
        a = sub["alpha_t"].values
        b = sub["beta_t"].values
        rows.append({
            "split": split, "param": "alpha+beta",
            "mean":  (a + b).mean(), "std": (a + b).std(),
            "min":   (a + b).min(),  "p25": np.percentile(a + b, 25),
            "p50":   np.percentile(a + b, 50), "p75": np.percentile(a + b, 75),
            "max":   (a + b).max(),
        })

    df_stats = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "alpha_beta_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")
    print(df_stats.to_string(index=False))


# ===========================================================================
# 4. Permutation feature importance
# ===========================================================================
def permutation_importance(model: GARCHNet, data: dict):
    print(f"\n[4] Permutation importance ({N_PERM_REPS} reps per feature)")

    X   = data["X_all"].to(DEVICE)
    r   = data["r_all"].to(DEVICE)
    gap = data["gap_mask"].to(DEVICE)
    train_end = data["train_end"]

    X_tr  = X[:train_end]
    r_tr  = r[:train_end]
    gap_tr = gap[:train_end]

    model.eval()

    # Baseline NLL on training window
    with torch.no_grad():
        sigma2_base, _, _ = model(X_tr, r_tr, gap_tr)
        base_nll = GARCHNet.nll_loss(sigma2_base, r_tr, gap_tr).item()

    print(f"  Baseline train NLL: {base_nll:.4f}")

    n_features = X_tr.shape[1]
    scores = np.zeros(n_features)      # mean NLL increase over reps

    for j in range(n_features):
        delta_list = []
        for _ in range(N_PERM_REPS):
            X_perm = X_tr.clone()
            perm_idx = torch.randperm(train_end)
            X_perm[:, j] = X_tr[perm_idx, j]   # permute column j only

            with torch.no_grad():
                sigma2_p, _, _ = model(X_perm, r_tr, gap_tr)
                nll_p = GARCHNet.nll_loss(sigma2_p, r_tr, gap_tr).item()

            delta_list.append(nll_p - base_nll)

        scores[j] = np.mean(delta_list)
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{n_features} features done")

    # Sort descending
    order = np.argsort(scores)[::-1]
    feat_names = [FEATURE_COLS[i] for i in order]
    feat_scores = scores[order]

    # Save CSV
    df_imp = pd.DataFrame({"feature": feat_names, "nll_increase": feat_scores})
    csv_path = os.path.join(RESULTS_DIR, "permutation_importance.csv")
    df_imp.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    # Plot — horizontal bar chart, top-20 for readability
    top_n = min(41, len(feat_names))
    fig, ax = plt.subplots(figsize=(9, top_n * 0.35 + 1.5))

    colors = ["#d62728" if s > 0 else "#aec7e8" for s in feat_scores[:top_n]]
    ax.barh(
        range(top_n),
        feat_scores[:top_n],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_names[:top_n], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Mean NLL increase after permutation\n(higher = more important)")
    ax.set_title(
        f"Permutation Feature Importance — train window\n"
        f"({N_PERM_REPS} reps, baseline NLL = {base_nll:.4f})",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("permutation_importance.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("GARCH-NN Evaluation")
    print("=" * 60)

    # ---- load data -------------------------------------------------------
    data = load_data()
    X    = data["X_all"]
    r    = data["r_all"]
    gap  = data["gap_mask"]

    # ---- load garch output CSV ------------------------------------------
    garch_df = pd.read_csv(GARCH_OUTPUT)
    print(f"\ngarch_output.csv: {len(garch_df):,} rows, splits: {garch_df['split'].value_counts().to_dict()}")

    # ---- load model config (a, b derived from standard GARCH fit) --------
    with open(MODEL_CFG) as f:
        cfg = json.load(f)
    a = cfg["a"]
    b = cfg["b"]
    print(f"Scaling bounds: a={a:.6f}  b={b:.6f}")

    # ---- load model -------------------------------------------------------
    model = GARCHNet(n_features=cfg["n_features"], hidden=cfg["hidden"], a=a, b=b).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()

    omega = torch.nn.functional.softplus(model.alpha0_raw).item()
    print(f"alpha_0 (omega): {omega:.8f}")

    # ---- run all diagnostics ---------------------------------------------
    plot_loss_curve()
    plot_residuals(data, garch_df)
    alpha_beta_stats(garch_df)
    permutation_importance(model, data)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
