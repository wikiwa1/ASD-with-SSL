"""MIMII dataset and train/test DataLoader construction."""
import os

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class MIMII(Dataset):
    """Dataset of clips for one MIMII machine split.

    `data_dir` is a list of .wav paths and `labels` their matching
    0 (normal) / 1 (abnormal) integer labels. Each clip is decoded to a mono
    waveform once at construction and kept in memory, so training epochs never
    re-read from disk.
    """
    def __init__(self, data_dir, labels):
        self.labels = labels
        self.data_dir = data_dir
        self.cache = []
        for path in data_dir:
            # soundfile is used instead of torchaudio.load (which needs the
            # torchcodec backend); it returns a (T, C) array.
            x, sr = sf.read(path, dtype='float32', always_2d=True)
            if x.size == 0:
                raise ValueError(f"Empty audio file: {path}")
            self.cache.append(torch.from_numpy(x).mean(dim=1))  # mix to mono -> (T,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cache[idx], self.labels[idx]


def audiodir(machine, id=0, Data='normal'):
    """Collect the .wav paths and labels for one machine/id/condition.

    Inputs:
        machine: Name of the machine (valve/slider/fan/pump)
        id: ID of the machine (0, 2, 4, 6)
        Data: 'normal' or 'abnormal'

    Outputs:
        dirs: List of .wav file paths
        label: List of labels (0 -> normal, 1 -> abnormal)
    """
    # Support both layouts: data/<machine>/id_XX/normal and data/<machine>/normal
    id_dir = os.path.join('data', machine, 'id_' + str(format(id, '02d')))
    flat_dir = os.path.join(machine)

    if os.path.isdir(id_dir):
        normaldir = os.path.join(id_dir, 'normal')
        abnormaldir = os.path.join(id_dir, 'abnormal')
    else:
        normaldir = os.path.join('data', flat_dir, 'normal')
        abnormaldir = os.path.join('data', flat_dir, 'abnormal')

    if not os.path.isdir(normaldir):
        raise FileNotFoundError(f"Normal data directory not found: {normaldir}")
    if not os.path.isdir(abnormaldir):
        raise FileNotFoundError(f"Abnormal data directory not found: {abnormaldir}")

    dirs = []
    label = []
    if Data == 'normal':
        file_list = sorted(os.listdir(normaldir))
        for i in file_list:
            if i.endswith('.wav'):
                dirs.append(os.path.join(normaldir, i))
                label.append(0)
    else:
        file_list = sorted(os.listdir(abnormaldir))
        for i in file_list:
            if i.endswith('.wav'):
                dirs.append(os.path.join(abnormaldir, i))
                label.append(1)

    return dirs, label


def train_test(machine='fan', id=0, train_size=0.75, batch_size=128,
               num_workers=0, verbosity=1):
    """Build train/test DataLoaders for one machine id.

    Training data is normal clips only (unsupervised setup). The test set is the
    held-out normal clips plus all abnormal clips, so ROC-AUC can be measured
    against the true normal/abnormal labels.
    """
    dir_normal, label_normal = audiodir(machine, id)
    dir_abnormal, label_abnormal = audiodir(machine, id, Data='abnormal')

    dataset_normal = MIMII(dir_normal, label_normal)
    dataset_abnormal = MIMII(dir_abnormal, label_abnormal)

    n_train = int(len(dataset_normal) * train_size)
    n_test = len(dataset_normal) - n_train
    train_dataset, test_normal_dataset = torch.utils.data.random_split(
        dataset_normal, [n_train, n_test]
    )

    test_dataset = ConcatDataset([test_normal_dataset, dataset_abnormal])

    # Audio is already cached in RAM, so num_workers=0 is fine; pin_memory speeds
    # up the host -> GPU copy.
    Train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True, pin_memory=True)
    Test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

    if verbosity > 0:
        print(f"Machine: {machine}, ID: {id}")
        print(f"  Normal samples: {len(dataset_normal)} (train: {n_train}, test: {n_test})")
        print(f"  Abnormal samples: {len(dataset_abnormal)}")
        print(f"  Total test: {len(test_dataset)}")

    return Train, Test
