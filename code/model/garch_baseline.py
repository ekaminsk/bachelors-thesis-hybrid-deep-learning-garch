"""
Create standard, econometric GARCH(1,1). Used by train.py to derive the scaling bounds a and b for the NN-GARCH model.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import RESULTS_DIR, ACF_LAGS, GARCH_BASELINE_SUMMARY, GARCH_BASELINE_PARAMS, GARCH_BASELINE_SIGMA2, GARCH_BASELINE_ACF
from data import load_data
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model


# ── ACF Helpers ──────────────────────────────────────────────────────────────

def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()                                        # de-means the series for autocorrelation
    n = len(x)                                      
    c0 = np.dot(x, x) / n                                   # variance
    if c0 == 0:
        return np.zeros(nlags + 1)
    vals = [1.0]                                            # lag 0 autocorrelation is always 1
    for k in range(1, nlags + 1):
        vals.append(np.dot(x[: n - k], x[k:]) / n / c0)     # autocorrelation at lag k
    return np.array(vals)


# ── Main fitting function ────────────────────────────────────────────────────

def fit_standard_garch(
    train_returns: np.ndarray,          # returns in basis points, shape: 1D (train_end)
    gap_mask: np.ndarray,               # bool mask, shape 1D (train_end) — True = NaN return
    timestamps=None,                    # optional list of strings for the output CSV
) -> dict:
    """
    Fit GARCH(1,1) with zero mean on the training returns.
    NaN returns at gap boundaries are kept as NaN
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    r_valid = train_returns[~gap_mask].astype(float)        # filtering gaps, float needed for ARCH

    print("\n" + "=" * 60)
    print("Fitting standard GARCH(1,1) on training window...")
    print(f"  Using {len(r_valid)} valid observations (excluded {gap_mask.sum()} gap rows)")

    # GARCH(1,1) model, zero cond. mean assumption, Gaussian innovation assumption, rescale something for numerical stability
    am  = arch_model(r_valid, vol="Garch", p=1, q=1, mean="Zero", dist="Normal",
                     rescale=True)
    
    # fitting model with max. likelihood
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
    # (spoiler, didnt work)
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

    # ── save summary text ────────────────────────────────────────────────────
    with open(GARCH_BASELINE_SUMMARY, "w") as f:
        f.write(str(res.summary()))                 # ARCH library with estimates, stderrors, t-stats, AIC, BIC, etc.
        f.write(f"\n\nDerived scaling bounds:\n  a = {a:.8f}\n  b = {b:.8f}\n")
    print(f"  saved: {GARCH_BASELINE_SUMMARY}")

    # ── save scalar params CSV ───────────────────────────────────────────────
    params_df = pd.DataFrame([{                     # building a table programmatically
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
    params_df.to_csv(GARCH_BASELINE_PARAMS, index=False)
    print(f"  saved: {GARCH_BASELINE_PARAMS}")

    # ── conditional variance series + residuals ──────────────────────────────
    
    # arch was fitted on r_valid (no gaps), so sigma2_fit has len = sum(~gap_mask)
    sigma2_valid = np.asarray(res.conditional_volatility) ** 2          # need to square to get variance
    eps          = r_valid / np.sqrt(np.maximum(sigma2_valid, 1e-12))   # returns / cond. standard dev. -> white noise if no GARCH effects

    # Reconstruct full-length series aligned to train window (NaN at gaps)
    sigma2_full = np.full(len(train_returns), np.nan)
    sigma2_full[~gap_mask] = sigma2_valid

    sigma2_df = pd.DataFrame({
        "sigma2_garch": sigma2_full,
        "gap":          gap_mask.astype(int),
    })
    if timestamps is not None:
        sigma2_df.insert(0, "window_end", timestamps)

    sigma2_df.to_csv(GARCH_BASELINE_SIGMA2, index=False)
    print(f"  saved: {GARCH_BASELINE_SIGMA2}")

    # ── ACF plots ────────────────────────────────────────────────────────────
    n_valid = len(eps)
    acf_eps  = _acf(eps,    ACF_LAGS)
    acf_eps2 = _acf(eps**2, ACF_LAGS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ci = 1.96 / np.sqrt(n_valid)
    lags = np.arange(ACF_LAGS + 1)

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
    plt.savefig(GARCH_BASELINE_ACF, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {GARCH_BASELINE_ACF}")

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
    d = load_data()
    r_np  = d["r_all"].numpy()
    gap_np = d["gap_mask"].numpy().astype(bool)
    result = fit_standard_garch(
        r_np[:d["train_end"]],
        gap_np[:d["train_end"]],
        timestamps=d["timestamps"][:d["train_end"]] if d["timestamps"] else None,
    )
    print(result)               # direct result for standalone testing
