"""Project-wide configuration: constants, paths, seeding, and device."""
import random

import numpy as np
import torch
from pathlib import Path

SEED = 42

# Dataset / experiment selection
DATA_ROOT = Path('data/fan')
# Change this to wherever you save/load model weights.
SAVE_DIR = Path('/content/drive/MyDrive/ASD-with-SSL')
MACHINE = 'fan'
MACHINE_IDS = [0, 2, 4, 6]

# Audio / feature-extraction settings
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FIXED_FRAMES = 128   # width every crop is resized to before the backbone
BACKBONE_DIM = 512   # ResNet-18 feature dimension

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
