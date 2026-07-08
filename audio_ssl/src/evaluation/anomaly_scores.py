from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from audio_ssl.src.features.logmel import file_to_vector_array


@torch.inference_mode()
def score_feature_array(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> float:
    if len(features) == 0:
        return float("nan")
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    recon = model(x)
    errors = torch.mean((x - recon) ** 2, dim=1)
    return float(errors.mean().cpu().item())


def score_files(
    model: torch.nn.Module,
    files: list[str | Path],
    feature_kwargs: dict,
    device: torch.device,
) -> np.ndarray:
    model.eval().to(device)
    scores: list[float] = []
    for file_name in tqdm(files, desc="score eval files"):
        features = file_to_vector_array(file_name, **feature_kwargs)
        scores.append(score_feature_array(model, features, device=device))
    return np.asarray(scores, dtype=np.float64)

