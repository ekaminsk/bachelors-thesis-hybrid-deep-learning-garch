"""
garch_baseline.py — Fit a standard GARCH(1,1) on the training return series.

Used by train.py to derive the scaling bounds a and b for the NN-GARCH model.

Outputs saved to D:/data/model/results/:
  garch_baseline_summary.txt   — full arch model summary
  garch_baseline_params.csv    — scalar estimates (omega, alpha, beta, SE, AIC, BIC, ...)
  garch_baseline_sigma2.csv    — per-timestep sigma2, eps_t for the train window
  garch_baseline_acf.png       — ACF of eps_t and eps_t^2 (train window, 40 lags)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\data\model\results"
ACF_NLAGS   = 40


# ---------------------------------------------------------------------------
# Minimal ACF helpers (self-contained, no dependency on evaluate.py)
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    if c0 == 0:
        return np.zeros(nlags + 1)
    vals = [1.0]
    for k in range(1, nlags + 1):
        vals.append(np.dot(x[: n - k], x[k:]) / n / c0)
    return np.array(vals)


def _plot_acf(acf_vals, title, filename, n_obs):
    lags = np.arange(len(acf_vals))
    ci = 1.96 / np.sqrt(n_obs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags[1:], acf_vals[1:], color="#3a7ebf", width=0.6)
    ax.axhline( ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0,   color="black", linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(acf_vals))
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_standard_garch(
    train_returns: np.ndarray,   # returns in basis points, shape (train_end,)
    gap_mask: np.ndarray,        # bool mask, shape (train_end,) — True = NaN return
    timestamps=None,             # optional list of strings for the output CSV
) -> dict:
    """
    Fit GARCH(1,1) with zero mean on the training returns.
    NaN returns at gap boundaries are kept as NaN — arch handles them natively.

    Returns
    -------
    dict with keys: omega, alpha, beta, a, b, loglik, aic, bic
      where a = 2*alpha, b = min(2*beta, 0.999 - 2*alpha)
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError(
            "The 'arch' package is required for standard GARCH fitting.\n"
            "Install it with:  py -m pip install arch"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use only valid (non-gap) returns for fitting.
    # 23 gaps out of ~2,353 training rows — negligible for parameter estimation.
    r_valid = train_returns[~gap_mask].astype(float)

    print("\n" + "=" * 60)
    print("Fitting standard GARCH(1,1) on training window...")
    print(f"  Using {len(r_valid)} valid observations (excluded {gap_mask.sum()} gap rows)")
    # rescale=True lets arch normalise internally for numerical stability;
    # alpha and beta are dimensionless so they are unaffected by rescaling.
    am  = arch_model(r_valid, vol="Garch", p=1, q=1, mean="Zero", dist="Normal",
                     rescale=True)
    res = am.fit(disp="off", show_warning=False)

    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta  = float(res.params["beta[1]"]  )

    omega_se = float(res.std_err["omega"])
    alpha_se = float(res.std_err["alpha[1]"])
    beta_se  = float(res.std_err["beta[1]"] )

    loglik = float(res.loglikelihood)
    aic    = float(res.aic)
    bic    = float(res.bic)

    # Guard against degenerate IGARCH fit (alpha+beta >= 0.999) or negative b.
    # This can happen with stablecoin pairs that have near-zero variance.
    # Fall back to conservative defaults typical for high-frequency FX data.
    _FALLBACK_ALPHA = 0.05
    _FALLBACK_BETA  = 0.90
    if alpha + beta >= 0.999 or alpha <= 0 or beta <= 0:
        print(f"  WARNING: degenerate GARCH fit (alpha+beta={alpha+beta:.4f}). "
              f"Using fallback alpha={_FALLBACK_ALPHA}, beta={_FALLBACK_BETA}.")
        alpha = _FALLBACK_ALPHA
        beta  = _FALLBACK_BETA

    a = 2.0 * alpha
    b = min(2.0 * beta, 0.999 - 2.0 * alpha)
    b = max(b, 0.01)   # safety floor — b must be positive

    print(f"  omega : {omega:.6f}  (SE={omega_se:.6f})")
    print(f"  alpha : {alpha:.6f}  (SE={alpha_se:.6f})")
    print(f"  beta  : {beta:.6f}  (SE={beta_se:.6f})")
    print(f"  alpha+beta : {alpha+beta:.6f}  (persistence)")
    print(f"  Log-lik: {loglik:.2f}  AIC: {aic:.2f}  BIC: {bic:.2f}")
    print(f"  => a = 2*alpha = {a:.6f}")
    print(f"  => b = min(2*beta, 0.999-2*alpha) = {b:.6f}")

    # ---- save summary text -----------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "garch_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(res.summary()))
        f.write(f"\n\nDerived scaling bounds:\n  a = {a:.8f}\n  b = {b:.8f}\n")
    print(f"  saved: {summary_path}")

    # ---- save scalar params CSV ------------------------------------------
    params_df = pd.DataFrame([{
        "param":       "omega",
        "estimate":    omega,
        "std_err":     omega_se,
        "t_stat":      omega / omega_se if omega_se > 0 else np.nan,
    }, {
        "param":       "alpha[1]",
        "estimate":    alpha,
        "std_err":     alpha_se,
        "t_stat":      alpha / alpha_se if alpha_se > 0 else np.nan,
    }, {
        "param":       "beta[1]",
        "estimate":    beta,
        "std_err":     beta_se,
        "t_stat":      beta / beta_se if beta_se > 0 else np.nan,
    }, {
        "param":       "persistence (alpha+beta)",
        "estimate":    alpha + beta,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }, {
        "param":       "log_likelihood",
        "estimate":    loglik,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }, {
        "param":       "AIC",
        "estimate":    aic,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }, {
        "param":       "BIC",
        "estimate":    bic,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }, {
        "param":       "scale_a (2*alpha)",
        "estimate":    a,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }, {
        "param":       "scale_b",
        "estimate":    b,
        "std_err":     np.nan,
        "t_stat":      np.nan,
    }])
    params_path = os.path.join(RESULTS_DIR, "garch_baseline_params.csv")
    params_df.to_csv(params_path, index=False)
    print(f"  saved: {params_path}")

    # ---- conditional variance series + residuals -------------------------
    # arch was fitted on r_valid (no gaps), so sigma2_fit has len = sum(~gap_mask)
    sigma2_valid = np.asarray(res.conditional_volatility) ** 2   # (n_valid,)
    eps          = r_valid / np.sqrt(np.maximum(sigma2_valid, 1e-12))

    # Reconstruct full-length series aligned to train window (NaN at gaps)
    sigma2_full = np.full(len(train_returns), np.nan)
    sigma2_full[~gap_mask] = sigma2_valid

    sigma2_df = pd.DataFrame({
        "sigma2_garch": sigma2_full,
        "gap":          gap_mask.astype(int),
    })
    if timestamps is not None:
        sigma2_df.insert(0, "window_end", timestamps)

    sigma2_path = os.path.join(RESULTS_DIR, "garch_baseline_sigma2.csv")
    sigma2_df.to_csv(sigma2_path, index=False)
    print(f"  saved: {sigma2_path}")

    # ---- ACF plots -------------------------------------------------------
    n_valid = len(eps)
    acf_eps  = _acf(eps,    ACF_NLAGS)
    acf_eps2 = _acf(eps**2, ACF_NLAGS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ci = 1.96 / np.sqrt(n_valid)
    lags = np.arange(ACF_NLAGS + 1)

    for ax, acf_vals, title in [
        (axes[0], acf_eps,  "ACF of Standardised Residuals\n(Standard GARCH(1,1), train)"),
        (axes[1], acf_eps2, "ACF of Squared Standardised Residuals\n(Standard GARCH(1,1), train)"),
    ]:
        ax.bar(lags[1:], acf_vals[1:], color="#3a7ebf", width=0.6)
        ax.axhline( ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0,   color="black", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend(fontsize=7)

    plt.tight_layout()
    acf_path = os.path.join(RESULTS_DIR, "garch_baseline_acf.png")
    plt.savefig(acf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {acf_path}")

    print(f"  eps_t — mean={eps.mean():.4f}  std={eps.std():.4f}  "
          f"skew={float(np.mean(((eps-eps.mean())/eps.std())**3)):.4f}  "
          f"excess kurt={float(np.mean(((eps-eps.mean())/eps.std())**4))-3:.4f}")
    print("=" * 60)

    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "a": a, "b": b,
        "loglik": loglik, "aic": aic, "bic": bic,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_data
    d = load_data()
    r_np  = d["r_all"].numpy()
    gap_np = d["gap_mask"].numpy().astype(bool)
    result = fit_standard_garch(
        r_np[:d["train_end"]],
        gap_np[:d["train_end"]],
        timestamps=d["timestamps"][:d["train_end"]] if d["timestamps"] else None,
    )
    print(result)
