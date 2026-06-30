from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def load_audio_channel(file_name: str | Path, channel: int = 0) -> tuple[int, np.ndarray]:
    """Load a wav at native sample rate and select one channel if needed."""
    audio, sr = librosa.load(str(file_name), sr=None, mono=False)
    audio = np.asarray(audio)
    if audio.ndim <= 1:
        return sr, audio.astype(np.float32, copy=False)
    if channel >= audio.shape[0]:
        raise ValueError(f"Requested channel {channel}, but audio has {audio.shape[0]} channels")
    return sr, audio[channel].astype(np.float32, copy=False)


def file_to_vector_array(
    file_name: str | Path,
    n_mels: int = 64,
    frames: int = 5,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    channel: int = 0,
) -> np.ndarray:
    """Convert a wav file into concatenated log-mel frame vectors.

    This mirrors the MIMII Keras baseline: mel spectrogram, log10 scaling, then
    concatenation of adjacent mel frames into vectors of length n_mels * frames.
    """
    dims = n_mels * frames
    sr, audio = load_audio_channel(file_name, channel=channel)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    log_mel = 20.0 / power * np.log10(mel + sys.float_info.epsilon)
    vector_count = log_mel.shape[1] - frames + 1
    if vector_count < 1:
        return np.empty((0, dims), dtype=np.float32)

    vectors = np.zeros((vector_count, dims), dtype=np.float32)
    for frame_idx in range(frames):
        start = n_mels * frame_idx
        end = n_mels * (frame_idx + 1)
        vectors[:, start:end] = log_mel[:, frame_idx : frame_idx + vector_count].T
    return vectors


def list_to_vector_array(
    file_list: list[str | Path],
    msg: str = "extract features",
    **feature_kwargs,
) -> np.ndarray:
    chunks = [
        file_to_vector_array(file_name, **feature_kwargs)
        for file_name in tqdm(file_list, desc=msg)
    ]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]
    if not chunks:
        dims = feature_kwargs.get("n_mels", 64) * feature_kwargs.get("frames", 5)
        return np.empty((0, dims), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

