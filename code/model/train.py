"""
train.py — Training loop for the hybrid GARCH-Neural Network.

Training strategy
-----------------
The GARCH recursion is sequential, so the entire training window is fed as
one batch per epoch (no minibatches).  Backprop traces through the recurrent
sigma2 computation.

Split (chronological):
  train : [0, val_start)        — gradient updates
  val   : [val_start, train_end) — early stopping (carved from train window)
  test  : [train_end, T)         — held-out final evaluation

For the GARCH state to be continuous, the forward pass always runs from t=0
through train_end.  The val loss is read off the sigma2 values for [val_start,
train_end) without re-running the recursion — both train and val NLLs come
from the same forward pass.

Hyperparameters (all per model_specification.txt and user-confirmed choices)
  hidden       = 32
  lr           = 1e-3
  weight_decay = 1e-4   (L2 regularisation on Adam, no dropout)
  max_epochs   = 500
  patience     = 50     (early stopping on val NLL)
  clip_norm    = 1.0    (gradient clipping)
"""

import json
import os
import time
import numpy as np
import torch
import torch.optim as optim

from data           import load_data
from model          import GARCHNet
from garch_baseline import fit_standard_garch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HIDDEN       = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 6000
PATIENCE     = 50
CLIP_NORM    = 1.0
SAVE_PATH    = r"D:\data\model\results\best_model.pt"
RESULTS_PATH = r"D:\data\model\results\training_results.txt"
MODEL_CFG    = r"D:\data\model\results\model_config.json"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Scaling bounds for alpha_t and beta_t.
# Set to None to auto-derive from a standard GARCH(1,1) fit (recommended).
# Override manually here if needed without breaking anything else, e.g.:
#   ALPHA_SCALE_A = 0.20
#   BETA_SCALE_B  = 0.60
# ---------------------------------------------------------------------------
ALPHA_SCALE_A = None   # a = 2 * hat_alpha  (auto when None)
BETA_SCALE_B  = None   # b = min(2*hat_beta, 0.999 - 2*hat_alpha)  (auto when None)


# ---------------------------------------------------------------------------
def compute_train_val_loss(
    model:    GARCHNet,
    X:        torch.Tensor,   # (T, 41) full sequence
    r:        torch.Tensor,   # (T,)
    gap:      torch.Tensor,   # (T,) bool
    val_start: int,
    train_end: int,
):
    """
    Single forward pass through the training window [0, train_end).
    Returns (train_loss, val_loss, sigma2_train, alpha_t, beta_t).
    """
    X_tr  = X[:train_end]
    r_tr  = r[:train_end]
    gap_tr = gap[:train_end]

    sigma2, alpha_t, beta_t = model(X_tr, r_tr, gap_tr)

    # Train loss: [0, val_start)
    train_gap_mask = gap_tr[:val_start]
    train_loss = GARCHNet.nll_loss(sigma2[:val_start], r_tr[:val_start], train_gap_mask)

    # Val loss: [val_start, train_end) — same forward, no extra computation
    val_gap_mask = gap_tr[val_start:]
    val_loss = GARCHNet.nll_loss(sigma2[val_start:], r_tr[val_start:], val_gap_mask)

    return train_loss, val_loss, sigma2, alpha_t, beta_t


def compute_test_loss(
    model:     GARCHNet,
    X:         torch.Tensor,
    r:         torch.Tensor,
    gap:       torch.Tensor,
    train_end: int,
):
    """
    Run the full sequence [0, T) and evaluate NLL on the test window [train_end, T).
    """
    with torch.no_grad():
        sigma2, alpha_t, beta_t = model(X, r, gap)

    test_loss = GARCHNet.nll_loss(sigma2[train_end:], r[train_end:], gap[train_end:])
    return test_loss, sigma2, alpha_t, beta_t


# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ---- load data -------------------------------------------------------
    data = load_data()
    X   = data["X_all"].to(DEVICE)
    r   = data["r_all"].to(DEVICE)
    gap = data["gap_mask"].to(DEVICE)

    train_end  = data["train_end"]
    val_start  = data["val_start"]
    T          = data["T"]
    n_features = X.shape[1]

    print(f"\nFeatures : {n_features}")
    print(f"Total T  : {T}")
    print(f"Train    : {val_start} rows  [0, {val_start})")
    print(f"Val      : {train_end - val_start} rows  [{val_start}, {train_end})")
    print(f"Test     : {T - train_end} rows  [{train_end}, {T})")
    print("=" * 60)

    # ---- standard GARCH(1,1) fit to derive scaling bounds ---------------
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    garch_fit = fit_standard_garch(
        train_returns=data["r_all"].numpy()[:train_end],
        gap_mask=data["gap_mask"].numpy().astype(bool)[:train_end],
        timestamps=data["timestamps"][:train_end] if data["timestamps"] else None,
    )

    a = ALPHA_SCALE_A if ALPHA_SCALE_A is not None else garch_fit["a"]
    b = BETA_SCALE_B  if BETA_SCALE_B  is not None else garch_fit["b"]
    print(f"\nUsing scaling bounds: a={a:.6f}  b={b:.6f}")
    if ALPHA_SCALE_A is not None or BETA_SCALE_B is not None:
        print("  (manually overridden)")

    # Save config so evaluate.py can reload the model with the same a, b
    cfg = {"n_features": n_features, "hidden": HIDDEN, "a": a, "b": b,
           "garch_alpha_hat": garch_fit["alpha"], "garch_beta_hat": garch_fit["beta"]}
    with open(MODEL_CFG, "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- model -----------------------------------------------------------
    model = GARCHNet(n_features=n_features, hidden=HIDDEN, a=a, b=b).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print(f"alpha0 (init)   : {torch.nn.functional.softplus(model.alpha0_raw).item():.6f}")

    # ---- training loop ---------------------------------------------------
    best_val_loss   = float("inf")
    patience_count  = 0
    best_epoch      = 0
    history         = []

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_loss, val_loss, _, _, _ = compute_train_val_loss(
            model, X, r, gap, val_start, train_end
        )

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        train_nll = train_loss.item()
        val_nll   = val_loss.item()
        history.append((epoch, train_nll, val_nll))

        # ---- early stopping ----------------------------------------------
        if val_nll < best_val_loss:
            best_val_loss  = val_nll
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            patience_count += 1

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{MAX_EPOCHS}  "
                f"train_NLL={train_nll:8.4f}  "
                f"val_NLL={val_nll:8.4f}  "
                f"best_val={best_val_loss:8.4f} (ep {best_epoch})  "
                f"patience={patience_count}/{PATIENCE}  "
                f"elapsed={elapsed:.0f}s"
            )

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    elapsed_total = time.time() - t0
    print(f"\nTraining done in {elapsed_total:.1f}s")

    # ---- reload best model and evaluate on test --------------------------
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        _, final_val, sigma2_tr, alpha_tr, beta_tr = compute_train_val_loss(
            model, X, r, gap, val_start, train_end
        )
        test_nll, sigma2_all, alpha_all, beta_all = compute_test_loss(
            model, X, r, gap, train_end
        )

    alpha0_val = torch.nn.functional.softplus(model.alpha0_raw).item()

    print("\n" + "=" * 60)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best val NLL    : {best_val_loss:.4f}")
    print(f"Test NLL        : {test_nll.item():.4f}")
    print(f"alpha_0 (omega) : {alpha0_val:.6f}")

    # ---- summary statistics for alpha_t and beta_t on test window --------
    a_test = alpha_all[train_end:].cpu().numpy()
    b_test = beta_all[train_end:].cpu().numpy()
    s2_test = sigma2_all[train_end:].cpu().numpy()

    print(f"\nTest window — time-varying parameters:")
    print(f"  alpha_t : mean={a_test.mean():.4f}  std={a_test.std():.4f}  "
          f"min={a_test.min():.4f}  max={a_test.max():.4f}")
    print(f"  beta_t  : mean={b_test.mean():.4f}  std={b_test.std():.4f}  "
          f"min={b_test.min():.4f}  max={b_test.max():.4f}")
    print(f"  alpha+beta : mean={( a_test + b_test ).mean():.4f}")
    print(f"  sigma2 (bp^2): mean={s2_test.mean():.2f}  std={s2_test.std():.2f}")
    print(f"  ann. vol (%)  : {np.sqrt( s2_test.mean() * 105120 ) / 100:.2f}%"
          " (5-min periods, 365-day year)")

    # ---- save training results to text file ------------------------------
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("GARCH-NN Training Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best epoch      : {best_epoch}\n")
        f.write(f"Best val NLL    : {best_val_loss:.6f}\n")
        f.write(f"Test NLL        : {test_nll.item():.6f}\n")
        f.write(f"alpha_0 (omega) : {alpha0_val:.8f}\n\n")
        f.write(f"Standard GARCH(1,1) reference:\n")
        f.write(f"  hat_alpha : {garch_fit['alpha']:.6f}\n")
        f.write(f"  hat_beta  : {garch_fit['beta']:.6f}\n")
        f.write(f"  persistence: {garch_fit['alpha']+garch_fit['beta']:.6f}\n")
        f.write(f"Scaling bounds used:\n")
        f.write(f"  a = {a:.6f}\n")
        f.write(f"  b = {b:.6f}\n\n")
        f.write(f"Test alpha_t  mean={a_test.mean():.4f}  std={a_test.std():.4f}\n")
        f.write(f"Test beta_t   mean={b_test.mean():.4f}  std={b_test.std():.4f}\n")
        f.write(f"Test alpha+beta  mean={(a_test+b_test).mean():.4f}\n\n")
        f.write("Epoch,TrainNLL,ValNLL\n")
        for ep, tr, vl in history:
            f.write(f"{ep},{tr:.6f},{vl:.6f}\n")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Best model saved to: {SAVE_PATH}")

    # ---- save sigma2 series to CSV for further analysis ------------------
    import pandas as pd
    sigma2_np = sigma2_all.cpu().numpy()
    alpha_np  = alpha_all.cpu().numpy()
    beta_np   = beta_all.cpu().numpy()

    out = pd.DataFrame({
        "sigma2":   sigma2_np,
        "alpha_t":  alpha_np,
        "beta_t":   beta_np,
        "split":    (
            ["train"] * val_start
            + ["val"]  * (train_end - val_start)
            + ["test"] * (T - train_end)
        ),
    })
    if data["timestamps"] is not None:
        out.insert(0, "window_end", data["timestamps"])

    csv_out = r"D:\data\model\results\garch_output.csv"
    out.to_csv(csv_out, index=False)
    print(f"sigma2 series saved to: {csv_out}")

    # ---- run diagnostics -------------------------------------------------
    print("\n" + "=" * 60)
    print("Running post-training diagnostics (evaluate.py)...")
    import evaluate
    evaluate.main()


if __name__ == "__main__":
    main()
