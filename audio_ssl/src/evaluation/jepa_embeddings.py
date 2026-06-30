from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audio_ssl.src.data.splits import find_target_dirs, make_baseline_split, parse_target_info
from audio_ssl.src.features.spectrogram import stack_logmels


@torch.inference_mode()
def embed_spectrograms(
    module,
    specs: torch.Tensor,
    batch_size: int = 256,
    device: torch.device | None = None,
    encoder: str = "target",
) -> np.ndarray:
    """Mean-pooled clip embeddings from a frozen JEPA encoder: (N, embed_dim).

    Defaults to the EMA `target` encoder because it processed full (unmasked)
    spectrograms during training — the context encoder only ever saw masked subsets,
    so a full clip is out-of-distribution for it.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(device)
    enc = module.target_encoder if encoder == "target" else module.context_encoder

    embeddings = []
    for start in range(0, len(specs), batch_size):
        x = module.normalize(specs[start : start + batch_size].to(device))
        tokens = enc(x)  # (B, num_patches, D), full spectrogram, no masking
        embeddings.append(tokens.mean(dim=1).float().cpu())
    return torch.cat(embeddings).numpy().astype(np.float32)


def normal_train_embeddings_by_target(
    module,
    config: dict,
    cache_path: str | Path,
    batch_size: int,
    device: torch.device,
    encoder: str = "target",
) -> dict[str, np.ndarray]:
    """Embed every target's normal-train clips, reusing the global pretraining spectrogram
    cache. The cache is the per-target train spectrograms concatenated in
    `find_target_dirs` order, so contiguous slices map to each target (validated below).
    """
    data_cfg = config["data"]
    specs = np.load(Path(cache_path), mmap_mode="r")  # (N_total, n_mels, T), gather order

    out: dict[str, np.ndarray] = {}
    offset = 0
    for target_dir in find_target_dirs(data_cfg["base_directory"]):
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        count = len(split.train_files)
        chunk = np.asarray(specs[offset : offset + count])
        out[info.key] = embed_spectrograms(
            module, torch.from_numpy(chunk).unsqueeze(1), batch_size, device, encoder
        )
        offset += count

    if offset != len(specs):  # cache no longer matches the gather order -> fail loudly
        raise RuntimeError(
            f"Spectrogram cache ({len(specs)}) does not match pooled train clips ({offset}); "
            "re-generate the cache or evaluate without it."
        )
    return out


def fit_set_embeddings(
    module,
    config: dict,
    target_dirs: list,
    cache_path: str | Path,
    feature_cfg: dict,
    batch_size: int,
    device: torch.device,
    encoder: str = "target",
) -> dict[str, np.ndarray]:
    """Per-target normal-train embeddings (the one-class fit set). Reuses the global
    spectrogram cache for a full-dataset run; otherwise extracts the given targets'
    train clips on the fly (e.g. when evaluating a single --target-dir)."""
    data_cfg = config["data"]
    all_dirs = find_target_dirs(data_cfg["base_directory"])
    full_run = Path(cache_path).exists() and {str(p) for p in target_dirs} == {str(p) for p in all_dirs}
    if full_run:
        return normal_train_embeddings_by_target(module, config, cache_path, batch_size, device, encoder)

    out: dict[str, np.ndarray] = {}
    for target_dir in target_dirs:
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        specs = stack_logmels(split.train_files, msg=f"train spectrograms {info.key}", **feature_cfg)
        out[info.key] = embed_spectrograms(
            module, torch.from_numpy(specs).unsqueeze(1), batch_size, device, encoder
        )
    return out
