"""
model.py — Hybrid GARCH-Neural Network.

Architecture
------------
GARCHNet:
  Input  : (T, 41) standardised microstructure features
  Hidden : Linear(41->32) -> ReLU -> Linear(32->32) -> ReLU -> Linear(32->2)
  Output : two independent raw logits (z1, z2) per timestep

Parameter mapping (per timestep t):
  omega   = softplus(alpha0_raw)              -- constant scalar alpha_0 > 0
  alpha_t = a * sigmoid(z1_t)                -- scales range into (0, a)
  beta_t  = b * sigmoid(z2_t)                -- scales range into (0, b)

  a and b are fixed scalars derived from a standard GARCH(1,1) fit:
    a = 2 * hat_alpha
    b = min(2 * hat_beta, 0.999 - 2 * hat_alpha)
  They can also be set manually in train.py (ALPHA_SCALE_A / BETA_SCALE_B).

  At sigmoid(z) = 0.5 (zero logit / random init) the model recovers
  approximately the standard GARCH parameters as a prior.

Post-hoc stationarity constraint (differentiable):
  If alpha_t + beta_t >= 1 at any t, both are rescaled so their sum equals 0.999:
    scale  = 0.999 / (alpha_t + beta_t)   [only applied where sum >= 1]
    alpha_t = alpha_t * scale
    beta_t  = beta_t  * scale

GARCH recursion (sequential, differentiable):
  sigma2_t = omega + alpha_t * a2_{t-1} + beta_t * sigma2_{t-1}
  where a_t = r_t  (zero conditional mean assumption)

Gap boundary handling:
  When return_t is NaN (gap_mask[t] = True):
    - sigma2_t is reset to the unconditional variance omega / (1 - alpha_t - beta_t)
    - a2_{t-1} for the next step is also set to this value  (E[r^2] = E[sigma^2])
    - gradient does NOT flow across the gap (detach)
    - the timestep is excluded from the loss

Loss: mean negative Gaussian log-likelihood over valid (non-gap) timesteps
  L = mean_t [ ln(sigma2_t) + r_t^2 / sigma2_t ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GARCHNet(nn.Module):
    """
    Hybrid GARCH-NN that produces time-varying alpha_t and beta_t from a
    feedforward network conditioned on contemporaneous microstructure features.

    Parameters
    ----------
    n_features : int    — number of input features (default 41)
    hidden     : int    — neurons per hidden layer (default 32)
    a          : float  — upper bound on alpha_t = a * sigmoid(z1); set to 2*hat_alpha
    b          : float  — upper bound on beta_t  = b * sigmoid(z2); set per formula
    """

    def __init__(
        self,
        n_features: int = 41,
        hidden:     int = 32,
        a:          float = 0.2,   # overridden by train.py from GARCH fit
        b:          float = 0.7,   # overridden by train.py from GARCH fit
    ):
        super().__init__()

        self.a = a
        self.b = b

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

        # alpha_0 (omega): long-run variance intercept, kept constant across time.
        # Parameterised as softplus(alpha0_raw) to enforce positivity.
        self.alpha0_raw = nn.Parameter(torch.tensor(-1.0))

    # ------------------------------------------------------------------
    def forward(
        self,
        X:        torch.Tensor,   # (T, n_features)  scaled features
        returns:  torch.Tensor,   # (T,)              returns in basis points
        gap_mask: torch.Tensor,   # (T,)  bool        True = NaN return / gap boundary
    ):
        """
        Run the GARCH recursion through all T timesteps.

        Returns
        -------
        sigma2 : FloatTensor (T,)  — conditional variance at each step
        alpha_t: FloatTensor (T,)  — time-varying ARCH coefficient  (post-constraint)
        beta_t : FloatTensor (T,)  — time-varying GARCH coefficient (post-constraint)
        """
        T      = X.shape[0]
        device = X.device

        omega = F.softplus(self.alpha0_raw)   # scalar, alpha_0 > 0

        # NN forward: batched over all T timesteps
        z = self.net(X)                              # (T, 2)

        alpha_t = self.a * torch.sigmoid(z[:, 0])   # (T,)  in (0, a)
        beta_t  = self.b * torch.sigmoid(z[:, 1])   # (T,)  in (0, b)

        # Post-hoc stationarity constraint (differentiable via torch.where)
        sum_ab = alpha_t + beta_t
        scale  = torch.where(
            sum_ab >= 1.0,
            torch.full_like(sum_ab, 0.999) / sum_ab,
            torch.ones_like(sum_ab),
        )
        alpha_t = alpha_t * scale
        beta_t  = beta_t  * scale

        # Pre-compute unconditional variance for each t (used at gap resets)
        uncond = omega / (1.0 - alpha_t - beta_t + 1e-8)   # (T,)

        # Convert gap_mask to a plain Python bool list for fast loop access
        gap_list = gap_mask.tolist()

        # ---- Initialise state -------------------------------------------
        sigma2_prev = uncond[0].detach()
        a2_prev     = uncond[0].detach()   # E[r^2] = E[sigma^2] = uncond at t=0

        sigma2_list: list[torch.Tensor] = []

        # ---- Sequential GARCH recursion ---------------------------------
        for t in range(T):
            if gap_list[t]:
                # Gap boundary: reset to unconditional variance; stop gradient
                sigma2_t    = uncond[t]
                sigma2_prev = sigma2_t.detach()
                a2_prev     = sigma2_t.detach()   # neutral restart for next step
            else:
                sigma2_t    = omega + alpha_t[t] * a2_prev + beta_t[t] * sigma2_prev
                sigma2_prev = sigma2_t
                a2_prev     = returns[t] * returns[t]   # r^2_{t}, data — no grad

            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list)   # (T,)
        return sigma2, alpha_t, beta_t

    # ------------------------------------------------------------------
    @staticmethod
    def nll_loss(
        sigma2:   torch.Tensor,   # (T,)
        returns:  torch.Tensor,   # (T,)
        gap_mask: torch.Tensor,   # (T,) bool
    ) -> torch.Tensor:
        """
        Mean negative Gaussian log-likelihood over valid (non-gap) timesteps.

        L = mean_t [ ln(sigma2_t) + r_t^2 / sigma2_t ]
        """
        valid = ~gap_mask
        s2 = sigma2[valid]
        r  = returns[valid]
        return (torch.log(s2) + r * r / s2).mean()
