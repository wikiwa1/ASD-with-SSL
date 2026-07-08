from __future__ import annotations

import torch
from torch import nn


class DenseAutoEncoder(nn.Module):
    """Dense autoencoder matching the MIMII baseline architecture by default."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int, ...] = (64, 64, 8, 64, 64)):
        super().__init__()
        dims = [input_dim, *hidden_dims, input_dim]
        layers: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

