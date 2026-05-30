"""
This file contains the code for the GARCHNet (GARCH Neural-Network) model

Architecture
───────────────────────────────────────────────────────────────────────────────
GARCHNet:
    Input  : (Timesteps, 41) standardised microstructure features
    Hidden : Linear(41->32) -> ReLU -> Linear(32->32) -> ReLU -> Linear(32->2)
    Output : two independent raw logits (unconstrained raw scores) (z1, z2) per timestep

Parameter mapping (per timestep t):
    omega   = softplus(alpha0_raw)              -- constant scalar alpha_0 > 0 (using softplus as it is more stable for small values)
    alpha_t = a * sigmoid(z1_t)                 -- scales range into (0, a)
    beta_t  = b * sigmoid(z2_t)                 -- scales range into (0, b)

    a and b are fixed scalars derived from a standard GARCH(1,1):
        a = 2 * hat_alpha
        b = min(2 * hat_beta, 0.999 - 2 * hat_alpha)
    They can also be set manually in train.py (ALPHA_SCALE_A / BETA_SCALE_B).
    I use this to try to enforce some structure onto my parameters. Such that the model gets starts off on standard GARCH parameters and learns from there.
    At sigmoid(z) = 0.5 (zero logit / random init) the model recovers approximately the standard GARCH parameters as a prior.

Post-hoc stationarity constraint (differentiable):
    If alpha_t + beta_t >= 1 at any t, both are rescaled so their sum equals 0.999:
        scale  = 0.999 / (alpha_t + beta_t)   [only applied where sum >= 1]
        alpha_t = alpha_t * scale
        beta_t  = beta_t  * scale

GARCH recursion (sequential, differentiable):
    sigma2_t = omega + alpha_t * a^2_{t-1} + beta_t * sigma^2_{t-1}                 -- where a_t = r_t  (zero conditional mean assumption)

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



class GARCHNet(nn.Module):                      # inherits PyTorch's nn.Module
    def __init__(
        self,
        n_features: int = 41,
        hidden:     int = 32,
        a:          float = 0.2,                # overridden by train.py from GARCH fit
        b:          float = 0.7,                # overridden by train.py from GARCH fit
    ):
        super().__init__()                      # Calls constructor (super refers to nn.Module)

        self.a = a
        self.b = b

        self.net = nn.Sequential(               # Logic for the layers (nn.Sequential defines network as sequence of layers), rest as explained above
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

        self.alpha0_raw = nn.Parameter(torch.tensor(-1.0))  # alpha_0 (omega): long-run variance intercept, with nn.Parameter it is optimized as well.
                                                            # Later becomes parameterized with softplus

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        X:        torch.Tensor,                 # (T, n_features)  scaled features
        returns:  torch.Tensor,                 # (T,)              returns in basis points
        gap_mask: torch.Tensor,                 # (T,)  bool        True = NaN return / gap boundary
    ):
    
        T     = X.shape[0]                      # Used in sequential GARCH recursion
        omega = F.softplus(self.alpha0_raw)

        z = self.net(X)                             # (T, 2) runs all timesteps at once

        alpha_t = self.a * torch.sigmoid(z[:, 0])   # (T,)  in (0, a)
        beta_t  = self.b * torch.sigmoid(z[:, 1])   # (T,)  in (0, b)

        # Post-hoc stationarity constraint (differentiable via torch.where -> if else over all tensors)
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


        # ── Initialise state ─────────────────────────────────────────────────
        sigma2_prev = uncond[0].detach()
        a2_prev     = uncond[0].detach()   # E[r^2] = E[sigma^2] = uncond at t=0

        sigma2_list: list[torch.Tensor] = []

        # ── Sequential GARCH recursion ───────────────────────────────────────
        for t in range(T):
            if gap_list[t]:                     # Boolean list
                # Gap boundary: reset to unconditional variance; stop gradient
                sigma2_t    = uncond[t]
                sigma2_prev = sigma2_t.detach()
                a2_prev     = sigma2_t.detach()   # neutral restart for next step
            else:
                sigma2_t    = omega + alpha_t[t] * a2_prev + beta_t[t] * sigma2_prev
                sigma2_prev = sigma2_t
                a2_prev     = returns[t] * returns[t]   # r^2_{t}, data — has no grad anyways

            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list)   # (T,) converts into one tensor of shape (T,)
        return sigma2, alpha_t, beta_t

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod                           # Method does not need self. Essentially standalone function, still added here because it fits into module. Called as GARCHNet.nll_loss()
    def nll_loss(
        sigma2:   torch.Tensor,                     # (T,)
        returns:  torch.Tensor,                     # (T,)
        gap_mask: torch.Tensor,                     # (T,) bool
    ) -> torch.Tensor:                              # Type hint, just documentation
        valid = ~gap_mask                           # ~ is Boolean not on tensors -> valid = True if gap_mask == False
        s2 = sigma2[valid]                           
        r  = returns[valid]
        return (torch.log(s2) + r * r / s2).mean()  # NLL formula elementwise over valid timesteps. Then averages
