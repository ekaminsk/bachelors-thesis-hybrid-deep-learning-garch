"""
The GARCH recursion is sequential, so the entire training window is fed as
one batch per epoch (no minibatches).  Backprop traces through the recurrent
sigma2 computation.
"""

import os, time, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from config         import RESULTS_DIR, HIDDEN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, CLIP_NORM, ALPHA_SCALE_A, BETA_SCALE_B, TRAIN_RESULTS, BEST_MODEL, MODEL_CONFIG, GARCH_OUTPUT
from data           import load_data
from model          import GARCHNet
from garch_baseline import fit_standard_garch
import evaluate     # type: ignore

# ── Configuration ────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────

def compute_train_val_loss(
    model:    GARCHNet,         # initializing with nn.Module
    X:        torch.Tensor,     # (T, 41) full sequence
    r:        torch.Tensor,     # (T,)
    gap:      torch.Tensor,     # (T,) bool
    val_start: int,
    train_end: int,
):
    """
    Single forward pass through the training window [0, train_end).
    Returns (train_loss, val_loss, sigma2_train, alpha_t, beta_t).
    """
    X_tr  = X[:train_end]                                       # [:train_end] includes validation window (by definition)
    r_tr  = r[:train_end]
    gap_tr = gap[:train_end]

    sigma2, alpha_t, beta_t = model(X_tr, r_tr, gap_tr)         # calls forward() function

    train_gap_mask = gap_tr[:val_start]                         # take the pure training window
    train_loss = GARCHNet.nll_loss(
        sigma2[:val_start], r_tr[:val_start], train_gap_mask    # calculate training loss
        )

    val_gap_mask = gap_tr[val_start:]
    val_loss = GARCHNet.nll_loss(
        sigma2[val_start:], r_tr[val_start:], val_gap_mask      # calculate validation loss
        )

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


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── load data ────────────────────────────────────────────────────────────
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

    # ── standard GARCH(1,1) fit to derive scaling bounds ─────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    garch_fit = fit_standard_garch(
        train_returns=data["r_all"].numpy()[:train_end],
        gap_mask=data["gap_mask"].numpy().astype(bool)[:train_end],
        timestamps=data["timestamps"][:train_end] if data["timestamps"] else None,
    )

    # Specification what scale to use, if it is manually set in the config use it, else calculate it
    a = ALPHA_SCALE_A if ALPHA_SCALE_A is not None else garch_fit["a"]      
    b = BETA_SCALE_B  if BETA_SCALE_B  is not None else garch_fit["b"]
    print(f"\nUsing scaling bounds: a={a:.6f}  b={b:.6f}")
    if ALPHA_SCALE_A is not None or BETA_SCALE_B is not None:
        print("  (manually overridden)")

    # Save config so evaluate.py can reload the model with the same a, b & for checking afterwards
    cfg = {"n_features": n_features, "hidden": HIDDEN_LAYERS, "a": a, "b": b,
           "garch_alpha_hat": garch_fit["alpha"], "garch_beta_hat": garch_fit["beta"]}
    with open(MODEL_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)

    
    # ── model ────────────────────────────────────────────────────────────────
    
    model = GARCHNet(n_features=n_features, hidden=HIDDEN_LAYERS, a=a, b=b).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print(f"omega (init)   : {torch.nn.functional.softplus(model.omega_raw).item():.6f}")

    # ── training loop ────────────────────────────────────────────────────────

    best_val_loss   = float("inf")                  # initialize to infinity such that first run always improves it
    patience_count  = 0
    best_epoch      = 0
    history         = []

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):                          # convention (Epoch 1, 2, 3 not Epoch 0,...)
        model.train()
        optimizer.zero_grad()

        train_loss, val_loss, _, _, _ = compute_train_val_loss(     # don't need the rest here
            model, X, r, gap, val_start, train_end
        )

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        train_nll = train_loss.item()
        val_nll   = val_loss.item()
        history.append((epoch, train_nll, val_nll))

        # ── early stopping ───────────────────────────────────────────────────
        if val_nll < best_val_loss:
            best_val_loss  = val_nll
            best_epoch     = epoch
            patience_count = 0                              # when it hits the PATIENCE then stops
            torch.save(model.state_dict(), BEST_MODEL)
        else:
            patience_count += 1

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{MAX_EPOCHS}"
                f"train_NLL={train_nll:8.4f}"
                f"val_NLL={val_nll:8.4f}"
                f"best_val={best_val_loss:8.4f} (ep {best_epoch})"
                f"patience={patience_count}/{PATIENCE}"
                f"elapsed={elapsed:.0f}s"
            )

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    elapsed_total = time.time() - t0
    print(f"\nTraining done in {elapsed_total:.1f}s")


    # ── reload best model and evaluate on test ───────────────────────────────
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()                # nn.Module native function

    with torch.no_grad():
        test_nll, sigma2_all, alpha_all, beta_all = compute_test_loss(
            model, X, r, gap, train_end
        )

    omega_val = torch.nn.functional.softplus(model.omega_raw).item()

    print("\n" + "=" * 60)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best val NLL    : {best_val_loss:.4f}")
    print(f"Test NLL        : {test_nll.item():.4f}")
    print(f"omega           : {omega_val:.6f}")

    # ── summary statistics for alpha_t and beta_t on test window ─────────────
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

    # ── save training results to text file ───────────────────────────────────
    with open(TRAIN_RESULTS, "w") as f:
        f.write("GARCH-NN Training Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best epoch      : {best_epoch}\n")
        f.write(f"Best val NLL    : {best_val_loss:.6f}\n")
        f.write(f"Test NLL        : {test_nll.item():.6f}\n")
        f.write(f"omega : {omega_val:.8f}\n\n")
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

    print(f"\nResults saved to: {TRAIN_RESULTS}")
    print(f"Best model saved to: {BEST_MODEL}")

    # ── save sigma2 series to CSV (in case) ──────────────────────────────────
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
    out.to_csv(GARCH_OUTPUT, index=False)
    print(f"sigma2 series saved to: {GARCH_OUTPUT}")

    # ── run diagnostics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running post-training diagnostics (evaluate.py)...")
    evaluate.main()


if __name__ == "__main__":
    main()
