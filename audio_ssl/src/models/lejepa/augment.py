from __future__ import annotations

import torch

# LeJEPA needs augmentation-invariant embeddings. The original repo uses image augmentations
# (crop / color / blur); for log-mel spectrograms we use mild SpecAugment-style nuisance
# transforms that preserve the (stationary/periodic) machine-sound identity: time roll, a
# constant log-gain offset, additive noise, and time/frequency band masking.


def _band_mask(x: torch.Tensor, dim: int, max_width: int, fill: torch.Tensor) -> torch.Tensor:
    size = x.shape[dim]
    width = int(torch.randint(0, max_width + 1, ()).item()) if max_width > 0 else 0
    if width == 0:
        return x
    start = int(torch.randint(0, max(1, size - width + 1), ()).item())
    x = x.clone()
    index = [slice(None)] * x.dim()
    index[dim] = slice(start, start + width)
    x[tuple(index)] = fill
    return x


def augment_view(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """One augmented view of a (B, 1, n_mels, T) log-mel batch (batch-level randomness)."""
    fill = x.mean(dim=(2, 3), keepdim=True)  # neutral value for masked bands

    roll = int(cfg.get("time_roll_frames", 20))
    if roll > 0:
        x = torch.roll(x, shifts=int(torch.randint(-roll, roll + 1, ()).item()), dims=-1)

    offset = float(cfg.get("offset", 1.0))
    if offset > 0:
        x = x + (torch.rand((), device=x.device) * 2 - 1) * offset  # constant log-gain

    noise = float(cfg.get("noise_std", 0.5))
    if noise > 0:
        x = x + noise * torch.randn_like(x)

    x = _band_mask(x, dim=3, max_width=int(cfg.get("time_mask_frames", 30)), fill=fill)
    x = _band_mask(x, dim=2, max_width=int(cfg.get("freq_mask_bins", 8)), fill=fill)
    return x


def make_views(x: torch.Tensor, num_views: int, cfg: dict) -> list[torch.Tensor]:
    return [augment_view(x, cfg) for _ in range(int(num_views))]
