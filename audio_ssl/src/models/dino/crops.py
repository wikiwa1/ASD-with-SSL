from __future__ import annotations

"""Frequency-band multi-crop for cached log-mel spectrograms (dino_asd_v2.ipynb).

Crops are contiguous MEL-FREQUENCY bands over the full clip duration (machine-sound
anomalies live in specific bands; time structure is near-stationary), resized to a fixed
(n_mels x fixed_frames) input. Augmentations are plain-torch ports of the notebook's
torchaudio/torchvision ops, applied batch-level (same randomness for the whole batch
within a crop, fresh randomness across crops).
"""

import torch
import torch.nn.functional as F


def _time_mask(x: torch.Tensor, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return x
    width = int(torch.randint(0, max_width + 1, ()).item())
    if width == 0:
        return x
    start = int(torch.randint(0, max(1, x.shape[-1] - width + 1), ()).item())
    x = x.clone()
    x[..., start : start + width] = 0.0
    return x


def _random_erase(x: torch.Tensor, p: float = 0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)) -> torch.Tensor:
    if float(torch.rand(())) >= p:
        return x
    h, w = x.shape[-2], x.shape[-1]
    area = float(torch.empty(()).uniform_(*scale)) * h * w
    aspect = float(torch.empty(()).uniform_(*ratio))
    eh = max(1, min(h, int(round((area * aspect) ** 0.5))))
    ew = max(1, min(w, int(round((area / aspect) ** 0.5))))
    top = int(torch.randint(0, h - eh + 1, ()).item())
    left = int(torch.randint(0, w - ew + 1, ()).item())
    x = x.clone()
    x[..., top : top + eh, left : left + ew] = 0.0
    return x


def augment_crop(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """(B, 1, mels, frames) -> augmented. Frequency masking is deliberately absent:
    crops ARE frequency bands, masking bins would blank the signal being compared."""
    x = _time_mask(x, int(cfg.get("time_mask_frames", 24)))
    noise = float(cfg.get("noise_std", 0.05))
    if noise > 0 and float(torch.rand(())) < 0.5:
        x = x + noise * torch.randn_like(x)
    if cfg.get("extra_augs", False):
        x = _random_erase(x)
        if float(torch.rand(())) < 0.5:  # +-5 dB gain shift in the (normalized) log domain
            shift = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 10.0
            x = x + shift * float(cfg.get("gain_scale", 1.0))
    return x


def _rand_crop(spec: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    full = spec.shape[dim]
    if size >= full:
        return spec
    start = int(torch.randint(0, full - size + 1, ()).item())
    index = [slice(None)] * spec.dim()
    index[dim] = slice(start, start + size)
    return spec[tuple(index)]


def multi_crop(spec: torch.Tensor, n_global: int = 2, n_local: int = 6,
               global_frac: float = 0.6, local_frac: float = 0.25,
               out_size: tuple[int, int] = (128, 128), augment_cfg: dict | None = None,
               train: bool = True, crop_axis: str = "freq") -> list[torch.Tensor]:
    """(B, 1, bins, T) spectrograms -> list of crops, global first.

    crop_axis "freq" (ResNet variant): mel-band crops resized to the fixed `out_size`
    backbone input. crop_axis "time" (BEATs variant, notebook's multi_crop_wave
    equivalent on cached fbanks): time-segment crops at native resolution, NO resize —
    BEATs handles variable length and its 128 fbank bins must stay intact.
    """
    dim = -2 if crop_axis == "freq" else -1
    full = spec.shape[dim]
    g = max(1, int(full * global_frac))
    l = max(1, int(full * local_frac))
    crops = [_rand_crop(spec, g, dim) for _ in range(n_global)]
    crops += [_rand_crop(spec, l, dim) for _ in range(n_local)]
    if crop_axis == "freq":
        crops = [F.interpolate(c, size=out_size, mode="bilinear", align_corners=False) for c in crops]
    if train:
        crops = [augment_crop(c, augment_cfg or {}) for c in crops]
    return crops
