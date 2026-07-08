"""Log-mel feature extraction, SpecAugment, DINO multi-crop, and the ResNet backbone."""
import random
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torchvision.models import resnet18 as _resnet18

from .config import DEVICE, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, FIXED_FRAMES

# SpecAugment-style masks applied per crop (operate on the (B, 1, mel, frame)
# tensor). Frequency masking is deliberately not applied below: since each crop
# is a band of mel bins, masking frequencies would erase the content the crops
# compare. These modules are device-agnostic, so they need no `.to(device)`.
freq_mask = AT.FrequencyMasking(freq_mask_param=24)
time_mask = AT.TimeMasking(time_mask_param=24)
random_erasing = VT.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)


@lru_cache(maxsize=None)
def _mel_extractor(device):
    """MelSpectrogram module for a given device (built once, then cached)."""
    return AT.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, n_mels=N_MELS).to(device)


@lru_cache(maxsize=None)
def _amp_to_db(device):
    return AT.AmplitudeToDB(stype='power', top_db=80).to(device)


def make_resnet18():
    """ResNet-18 adapted to single-channel spectrograms; outputs 512-d features."""
    m = _resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Identity()
    return m


def to_logmel(waveform, device=DEVICE):
    """(B, T) raw audio -> per-sample-normalized log-mel (B, 1, N_MELS, frames)."""
    spec = _mel_extractor(device)(waveform)        # (B, N_MELS, frames)
    spec = _amp_to_db(device)(spec).unsqueeze(1)   # (B, 1, N_MELS, frames)
    mean = spec.mean(dim=(2, 3), keepdim=True)
    std = spec.std(dim=(2, 3), keepdim=True) + 1e-5
    return (spec - mean) / std


def apply_transform(spec, train=True, use_extra_augs=False):
    """Per-crop augmentation on a log-mel spectrogram (B, 1, mel, frame).
    Returns the spectrogram unchanged when `train` is False."""
    if not train:
        return spec

    # Time masking + a little additive noise (frequency masking is off; see above).
    spec = time_mask(spec)
    if random.random() < 0.5:
        spec = spec + 0.05 * torch.randn_like(spec)

    if use_extra_augs:
        # Erase a random rectangular patch.
        spec = random_erasing(spec)

        # Random gain shift, i.e. a constant offset in the log (dB) domain.
        if random.random() < 0.5:
            shift = (torch.rand(spec.shape[0], 1, 1, 1, device=spec.device) - 0.5) * 10.0
            spec = spec + shift

    return spec


def _rand_freq_crop(spec, height):
    """Crop a contiguous band of `height` mel bins (the frequency axis, dim -2)."""
    Mf = spec.shape[-2]
    if height >= Mf:
        return spec
    start = random.randint(0, Mf - height)
    return spec[..., start:start + height, :]


def multi_crop(waveform, n_global=2, n_local=6, global_frac=0.6, local_frac=0.25,
               train=True, use_extra_augs=False, device=DEVICE):
    """(B, T) raw audio -> list of augmented, equal-sized spectrogram crops.
    The first `n_global` entries are the global crops (fed to the teacher); the
    rest are smaller local crops. Crops are taken along the mel-frequency axis,
    so each sees a different frequency band over the full clip duration, then all
    are resized to (N_MELS, FIXED_FRAMES)."""
    spec = to_logmel(waveform.to(device), device=device)   # (B, 1, N_MELS, Tt)
    Mf = spec.shape[-2]
    g, l = max(1, int(Mf * global_frac)), max(1, int(Mf * local_frac))
    crops = [_rand_freq_crop(spec, g) for _ in range(n_global)]
    crops += [_rand_freq_crop(spec, l) for _ in range(n_local)]
    crops = [F.interpolate(c, size=(N_MELS, FIXED_FRAMES),
                           mode='bilinear', align_corners=False) for c in crops]
    return [apply_transform(c, train=train, use_extra_augs=use_extra_augs) for c in crops]


def multi_crop_wave(waveform, n_global=2, n_local=4, global_frac=0.6,
                    local_frac=0.3, train=True, device=DEVICE):
    """Waveform-domain DINO multi-crop for the BEATs backbone. Global crops are
    long time-segments, local crops short ones. Light additive noise is the only
    augmentation (SpecAugment doesn't apply: BEATs builds the spectrogram itself)."""
    wav = waveform.to(device)                    # (B, T)
    T = wav.shape[-1]
    g, l = max(1, int(T * global_frac)), max(1, int(T * local_frac))

    def crop(width):
        if width >= T:
            c = wav
        else:
            start = random.randint(0, T - width)
            c = wav[..., start:start + width]
        if train and random.random() < 0.5:
            c = c + 0.005 * torch.randn_like(c)
        return c

    return [crop(g) for _ in range(n_global)] + [crop(l) for _ in range(n_local)]
