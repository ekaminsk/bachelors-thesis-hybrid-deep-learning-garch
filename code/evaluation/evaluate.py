"""
This module serves for diagnostics, i.e. does that model work. 
Even though it has plots, those are not the thesis figures. Thesis figures are in plots.py.
Since the figures here serve a different purpose (and are mainly there for personal checking) I will keep them seperate here.
"""

import json
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from config import EVALUATE_DIR, TRAIN_RESULTS, GARCH_OUTPUT, BEST_MODEL, MODEL_CONFIG, PERMUTATION_REPETITIONS, ACF_LAGS, PLOTS_DIR
from data  import load_data, FEATURE_COLS
from model import GARCHNet

# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
os.makedirs(EVALUATE_DIR, exist_ok=True)                 # in case, not necessary

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)                # still decide to put it in plot folder, but under different naming
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def acf_series(x: np.ndarray, nlags: int) -> np.ndarray:
    # Sample ACF from lag 0 to nlags (inclusive)
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


def _skew(x):
    m = x.mean(); s = x.std()
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else 0.0

def _kurt(x):
    m = x.mean(); s = x.std()
    return float(np.mean(((x - m) / s) ** 4)) - 3.0 if s > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss Curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curve():
    print("\n[1] Loss curve")

    # Parse training_results.txt — CSV section starts after the header line
    rows = []
    in_csv = False
    with open(TRAIN_RESULTS) as f:
        for line in f:
            line = line.strip()
            if line == "Epoch,TrainNLL,ValNLL":             # find where the CSV section starts
                in_csv = True
                continue
            if in_csv and line:
                parts = line.split(",")
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))      # split data row into tuple

    if not rows:
        print("  WARNING: no epoch data found in training_results.txt — skipping")  # defensive
        return

    df = pd.DataFrame(rows, columns=["epoch", "train_nll", "val_nll"])              # convert tuples into dataframes

    # Save CSV
    csv_path = os.path.join(EVALUATE_DIR, "evaluate_loss_curve.csv")
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. Standardised residuals + ACF
# ─────────────────────────────────────────────────────────────────────────────
def compute_residuals(data: dict, garch_df: pd.DataFrame):
    # Returns residuals DataFrame for the train window only.
    train_end       = data["train_end"]
    gap_mask        = data["gap_mask"].numpy().astype(bool)
    returns_all     = data["r_all"].numpy()

    sigma2_all      = garch_df["sigma2"].values

    # Work on train window
    returns_train   = returns_all[:train_end]
    sigma2_train    = sigma2_all[:train_end]
    gap_train       = gap_mask[:train_end]

    valid           = ~gap_train
    returns_valid   = returns_train[valid]
    sigma2_valid    = sigma2_train[valid]

    epsilon         = returns_valid / np.sqrt(sigma2_valid)
    epsilon2        = epsilon ** 2

    df_res = pd.DataFrame({
        "return_t"       : returns_valid,
        "sigma2_t"  : sigma2_valid,
        "epsilon_t"     : epsilon,
        "epsilon2_t"    : epsilon2,
    })
    return df_res


def plot_residuals(data: dict, garch_df: pd.DataFrame):
    print("\n[2] Standardised residuals & ACF")

    df_res = compute_residuals(data, garch_df)

    # Save CSV
    csv_path = os.path.join(EVALUATE_DIR, "evaluate_residuals.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    epsilon  = df_res["epsilon_t"].values
    epsilon2 = df_res["epsilon2_t"].values
    n    = len(epsilon)

    # ACF of standardised residuals
    acf_epsilon  = acf_series(epsilon,  ACF_LAGS)
    acf_epsilon2 = acf_series(epsilon2, ACF_LAGS)

    # Figures
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(epsilon, lags=ACF_LAGS, ax=ax, zero=False)
    ax.set_title("ACF of Standardised Residuals  (train window)", fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    savefig("evaluate_acf_residuals.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(epsilon2, lags=ACF_LAGS, ax=ax, zero=False)
    ax.set_title("ACF of Squared Standardised Residuals  (train window)", fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    savefig("evaluate_acf_sq_residuals.png")

    # Save ACF values as CSV
    acf_df = pd.DataFrame({
        "lag":      np.arange(ACF_LAGS + 1),
        "acf_epsilon":  acf_epsilon,
        "acf_epsilon2": acf_epsilon2,
    })
    acf_csv = os.path.join(EVALUATE_DIR, "acf_values.csv")
    acf_df.to_csv(acf_csv, index=False)
    print(f"  saved: {acf_csv}")

    # Quick descriptive stats
    print(f"  eps_t  — mean={epsilon.mean():.4f}  std={epsilon.std():.4f}  "
          f"skew={_skew(epsilon):.4f}  kurt={_kurt(epsilon):.4f}")
    print(f"  eps2_t — mean={epsilon2.mean():.4f}  std={epsilon2.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. alpha_t / beta_t summary statistics
# ─────────────────────────────────────────────────────────────────────────────
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
    csv_path = os.path.join(EVALUATE_DIR, "evaluate_alpha_beta_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")
    print(df_stats.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Permutation feature importance
# ─────────────────────────────────────────────────────────────────────────────
def permutation_importance(model: GARCHNet, data: dict):
    print(f"\n[4] Permutation importance ({PERMUTATION_REPETITIONS} reps per feature)")

    X           = data["X_all"].to(DEVICE)              # feature matrix
    returns     = data["r_all"].to(DEVICE)
    gap         = data["gap_mask"].to(DEVICE)
    train_end   = data["train_end"]

    X_train     = X[:train_end]
    r_train     = returns[:train_end]
    gap_train   = gap[:train_end]

    model.eval()

    # Baseline NLL on training window (how well does it do with all features intact)
    with torch.no_grad():
        sigma2_base, _, _ = model(X_train, r_train, gap_train)
        base_nll = GARCHNet.nll_loss(sigma2_base, r_train, gap_train).item()

    print(f"  Baseline train NLL: {base_nll:.4f}")

    n_features = X_train.shape[1]
    scores = np.zeros(n_features)                           # mean NLL increase over reps

    # start removing features and see how much worse it gets
    for j in range(n_features):
        delta_list = []
        for _ in range(PERMUTATION_REPETITIONS):            # repeat it PERM_REP times
            X_perm = X_train.clone()                        # clone full matrixm shuffle only column j
            perm_idx = torch.randperm(train_end)
            X_perm[:, j] = X_train[perm_idx, j]             # permute column j only

            with torch.no_grad():                           # run with broken feature
                sigma2_p, _, _ = model(X_perm, r_train, gap_train)
                nll_p = GARCHNet.nll_loss(sigma2_p, r_train, gap_train).item()

            delta_list.append(nll_p - base_nll)             # positive means NLL got worse

        scores[j] = np.mean(delta_list)                     # mean to reduce noise
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{n_features} features done")

    # Sort descending
    order = np.argsort(scores)[::-1]
    feat_names = [FEATURE_COLS[i] for i in order]
    feat_scores = scores[order]

    # Save CSV
    df_imp = pd.DataFrame({"feature": feat_names, "nll_increase": feat_scores})
    csv_path = os.path.join(EVALUATE_DIR, "evaluation_permutation_importance.csv")
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
        f"({PERMUTATION_REPETITIONS} reps, baseline NLL = {base_nll:.4f})",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("evaluate_permutation_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GARCH-NN Evaluation")
    print("=" * 60)

    # ── load data ────────────────────────────────────────────────────────────
    data = load_data()

    # ── load garch output CSV ────────────────────────────────────────────────
    garch_df = pd.read_csv(GARCH_OUTPUT)
    print(f"\ngarch_output.csv: {len(garch_df):,} rows, splits: {garch_df['split'].value_counts().to_dict()}")

    # ── load model config (a, b derived from standard GARCH fit) ─────────────
    with open(MODEL_CONFIG) as f:
        cfg = json.load(f)
    a = cfg["a"]
    b = cfg["b"]
    print(f"Scaling bounds: a={a:.6f}  b={b:.6f}")

    # ── load model ───────────────────────────────────────────────────────────
    model = GARCHNet(n_features=cfg["n_features"], hidden=cfg["hidden"], a=a, b=b).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()

    omega = torch.nn.functional.softplus(model.omega_raw).item()
    print(f"omega: {omega:.8f}")

    # ── run all diagnostics ──────────────────────────────────────────────────
    plot_loss_curve()
    plot_residuals(data, garch_df)
    alpha_beta_stats(garch_df)
    permutation_importance(model, data)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {EVALUATE_DIR}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
