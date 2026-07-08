from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

from audio_ssl.src.features.logmel import load_audio_channel


def file_to_logmel(
    file_name: str | Path,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    channel: int = 0,
    target_frames: int = 313,
) -> np.ndarray:
    """Full-clip log-mel spectrogram, fixed to `target_frames` (pad/crop), shape (n_mels, T).

    Same mel + log10 scaling as the MIMII baseline (`file_to_vector_array`), but kept as
    a 2D time-frequency image for the JEPA ViT instead of stacked frame vectors.
    """
    sr, audio = load_audio_channel(file_name, channel=channel)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    )
    log_mel = (20.0 / power) * np.log10(mel + sys.float_info.epsilon)
    log_mel = log_mel.astype(np.float32, copy=False)

    t = log_mel.shape[1]
    if t < target_frames:  # pad on the right with the spectrogram floor
        pad = np.full((n_mels, target_frames - t), log_mel.min(), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    elif t > target_frames:  # center crop
        start = (t - target_frames) // 2
        log_mel = log_mel[:, start : start + target_frames]
    return log_mel


def stack_logmels(
    file_list: list[str | Path],
    msg: str = "extract spectrograms",
    n_jobs: int | None = None,
    **feature_kwargs,
) -> np.ndarray:
    """Stack per-clip log-mel spectrograms into (N, n_mels, T).

    Extraction is parallelized across processes (librosa per file) — extracting ~50k
    clips single-threaded would take hours; this is a one-time cost before caching.
    """
    import os
    from functools import partial

    from tqdm import tqdm

    extract = partial(file_to_logmel, **feature_kwargs)
    if n_jobs is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit():
            n_jobs = int(slurm_cpus)  # match the SLURM allocation, no over-subscription
        else:
            n_jobs = min(32, os.cpu_count() or 8)

    if n_jobs <= 1 or len(file_list) < 64:
        mats = [extract(f) for f in tqdm(file_list, desc=msg)]
    else:
        import multiprocessing as mp

        with mp.Pool(n_jobs) as pool:
            mats = list(tqdm(pool.imap(extract, file_list, chunksize=16),
                             total=len(file_list), desc=msg))
    return np.stack(mats, axis=0).astype(np.float32, copy=False)
