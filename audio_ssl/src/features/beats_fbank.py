from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_ssl.src.features.logmel import load_audio_channel

# BEATs preprocessing constants (BEATs.preprocess): kaldi fbank of wav*2^15, then
# (fbank - MEAN) / (2 * STD). Cached fbanks are stored ALREADY normalized, so the
# Lightning module's normalize() is the identity for this frontend.
FBANK_MEAN = 15.41663
FBANK_STD = 6.55582
NUM_MEL_BINS = 128


def file_to_beats_fbank(
    file_name: str | Path,
    channel: int = 0,
    target_frames: int = 998,
    **_ignored,  # tolerate logmel-style keys (n_fft, ...) from shared config plumbing
) -> np.ndarray:
    """BEATs-normalized kaldi fbank for one clip, shape (128, target_frames) — stored in
    the same (bins, frames) orientation as our log-mels so all downstream shape
    conventions (caches, monitors, eval) are unchanged."""
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi

    sr, audio = load_audio_channel(file_name, channel=channel)
    if sr != 16000:
        raise ValueError(f"BEATs expects 16 kHz audio, got {sr} Hz for {file_name}")
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0) * 2**15
    fbank = ta_kaldi.fbank(wav, num_mel_bins=NUM_MEL_BINS, sample_frequency=16000,
                           frame_length=25, frame_shift=10)  # (frames, 128)
    fbank = ((fbank - FBANK_MEAN) / (2 * FBANK_STD)).numpy().astype(np.float32).T  # (128, frames)

    t = fbank.shape[1]
    if t < target_frames:
        fbank = np.concatenate(
            [fbank, np.zeros((NUM_MEL_BINS, target_frames - t), dtype=np.float32)], axis=1)
    elif t > target_frames:
        start = (t - target_frames) // 2
        fbank = fbank[:, start : start + target_frames]
    return fbank


def _fbank_worker_init() -> None:
    import torch
    torch.set_num_threads(1)  # 16 workers x default OpenMP threads would thrash the node


def stack_beats_fbanks(
    file_list: list[str | Path],
    msg: str = "extract BEATs fbanks",
    n_jobs: int | None = None,
    **feature_kwargs,
) -> np.ndarray:
    """Stack per-clip BEATs fbanks into (N, 128, T) — same parallel pattern as
    stack_logmels, but with a SPAWN pool: these workers run torch ops (kaldi fbank), and
    forking a CUDA-initialized parent (the eval script loads the model onto the GPU
    before extracting) deadlocks the children. Spawned workers start clean."""
    import os
    from functools import partial

    from tqdm import tqdm

    extract = partial(file_to_beats_fbank, **feature_kwargs)
    if n_jobs is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else min(32, os.cpu_count() or 8)

    if n_jobs <= 1 or len(file_list) < 64:
        mats = [extract(f) for f in tqdm(file_list, desc=msg)]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(n_jobs, initializer=_fbank_worker_init) as pool:
            mats = list(tqdm(pool.imap(extract, file_list, chunksize=16),
                             total=len(file_list), desc=msg))
    return np.stack(mats, axis=0).astype(np.float32, copy=False)
