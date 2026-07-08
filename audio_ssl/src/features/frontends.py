from __future__ import annotations

"""Feature-frontend dispatch: every training/eval/monitor path extracts clip features via
`stack_features`, routed by the config's `feature.frontend` key. `logmel` (default) is
the 64-mel spectrogram the scratch JEPA/LeJEPA/AE use; `beats_fbank` is the
kaldi-fbank-128 input BEATs was pretrained on. Both return (N, bins, T) float32 so cache
files, monitor npz layouts, and (B, 1, bins, T) model inputs are shape-compatible."""

import numpy as np

from audio_ssl.src.features.beats_fbank import stack_beats_fbanks
from audio_ssl.src.features.spectrogram import stack_logmels


def stack_features(file_list, msg: str = "extract features", **feature_kwargs) -> np.ndarray:
    frontend = str(feature_kwargs.pop("frontend", "logmel")).lower()
    if frontend == "logmel":
        return stack_logmels(file_list, msg=msg, **feature_kwargs)
    if frontend == "beats_fbank":
        return stack_beats_fbanks(file_list, msg=msg, **feature_kwargs)
    raise ValueError(f"unknown feature frontend '{frontend}' (logmel|beats_fbank)")
