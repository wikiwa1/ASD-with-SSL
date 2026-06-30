from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from audio_ssl.src.models.jepa.masking import sample_block_mask


@torch.inference_mode()
def score_spectrograms(
    module,
    specs: torch.Tensor,
    num_masks: int = 16,
    batch_size: int = 256,
    device: torch.device | None = None,
    base_seed: int = 0,
) -> np.ndarray:
    """Per-clip JEPA anomaly score = mean latent prediction error over `num_masks` fixed
    masks. `specs` is (N, 1, n_mels, T) raw log-mel; normalization uses the model buffers.

    The mask set is fixed (seeds base_seed..base_seed+num_masks-1) so the score is
    deterministic and identical across clips/targets — only the ranking matters for AUC.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(device)
    grid_h, grid_w = module.grid

    masks = []
    for k in range(num_masks):
        generator = torch.Generator().manual_seed(base_seed + k)
        target_idx, context_idx = sample_block_mask(
            grid_h, grid_w, module.hparams.num_blocks,
            module.hparams.mask_scale, module.hparams.mask_aspect, generator=generator,
        )
        masks.append((target_idx.to(device), context_idx.to(device)))

    scores = []
    for start in range(0, len(specs), batch_size):
        x = specs[start : start + batch_size].to(device)
        x = module.normalize(x)
        acc = torch.zeros(x.size(0), device=device)
        for target_idx, context_idx in masks:
            targets = module.target_encoder(x)[:, target_idx, :]
            context_enc = module.context_encoder(x, keep_idx=context_idx)
            preds = module.predictor(context_enc, context_idx, target_idx)
            acc += F.smooth_l1_loss(preds, targets, reduction="none").mean(dim=(1, 2))
        scores.append((acc / len(masks)).float().cpu())
    return torch.cat(scores).numpy().astype(np.float64)
