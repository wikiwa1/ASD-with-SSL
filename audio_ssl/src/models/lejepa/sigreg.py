from __future__ import annotations

import torch
from torch import nn

_TRAPEZOID = getattr(torch, "trapezoid", torch.trapz)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

    Pushes a batch of embeddings toward an isotropic standard Gaussian by slicing them onto
    many random 1-D directions and applying an Epps-Pulley characteristic-function test of
    N(0,1) on each slice, then averaging. Directions are resampled every call.

    Per slice, with projected samples x_j (j=1..N):
        EP = integral over t of  |phi_hat(t) - exp(-t^2/2)|^2 * exp(-t^2/sigma^2) dt
        phi_hat(t) = mean_j exp(i t x_j)  (empirical characteristic function)
    The projections are NOT standardized, so the test also penalizes non-zero mean /
    non-unit variance -> the whole embedding distribution is driven to N(0, I).
    """

    def __init__(self, num_slices: int = 512, num_points: int = 17,
                 t_max: float = 5.0, sigma: float = 1.0):
        super().__init__()
        self.num_slices = int(num_slices)
        t = torch.linspace(-t_max, t_max, int(num_points))
        self.register_buffer("t", t)
        self.register_buffer("target_cf", torch.exp(-0.5 * t ** 2))   # CF of N(0,1) (real)
        self.register_buffer("weight", torch.exp(-(t ** 2) / (sigma ** 2)))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (N, D). Random unit directions, resampled each step.
        directions = torch.randn(self.num_slices, embeddings.shape[1], device=embeddings.device)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
        proj = embeddings @ directions.t()                          # (N, S)

        arg = proj.unsqueeze(-1) * self.t.view(1, 1, -1)            # (N, S, T)
        cos_mean = arg.cos().mean(0)                                # (S, T) Re[phi_hat]
        sin_mean = arg.sin().mean(0)                                # (S, T) Im[phi_hat]
        diff_sq = (cos_mean - self.target_cf.view(1, -1)) ** 2 + sin_mean ** 2
        ep = _TRAPEZOID(diff_sq * self.weight.view(1, -1), self.t, dim=-1)  # (S,)
        return ep.mean()
