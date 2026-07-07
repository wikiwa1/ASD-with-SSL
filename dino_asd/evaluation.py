"""Feature extraction and anomaly scoring (Mahalanobis distance, kNN)."""
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .config import DEVICE, N_MELS, FIXED_FRAMES
from .features import to_logmel


@torch.no_grad()
def extract_features(backbone, loader, device=DEVICE):
    """Deterministic global-view features (full spectrogram, no crops/aug)."""
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        spec = to_logmel(x.to(device), device=device)
        spec = F.interpolate(spec, size=(N_MELS, FIXED_FRAMES),
                             mode='bilinear', align_corners=False)
        feats.append(backbone(spec).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def _global_features(backbone, loader, device=DEVICE):
    """Like extract_features but restores the backbone's train/eval mode, so it
    is safe to call from inside a training loop (for in-training AUC)."""
    was_training = backbone.training
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        spec = to_logmel(x.to(device), device=device)
        spec = F.interpolate(spec, size=(N_MELS, FIXED_FRAMES),
                             mode='bilinear', align_corners=False)
        feats.append(backbone(spec).cpu())
        labels.append(y)
    if was_training:
        backbone.train()
    return torch.cat(feats), torch.cat(labels)


def fit_gaussian(train_feats, eps=1e-3):
    """Fit mean + (regularized) inverse covariance of the normal feature distribution."""
    mu = train_feats.mean(dim=0)
    centered = train_feats - mu
    cov = centered.t() @ centered / (train_feats.shape[0] - 1)
    cov += eps * torch.eye(cov.shape[0])          # shrinkage for invertibility
    cov_inv = torch.linalg.inv(cov)
    return mu, cov_inv


def mahalanobis_score(test_feats, mu, cov_inv):
    """Mahalanobis distance of each test feature from the normal distribution."""
    centered = test_feats - mu                      # (n_test, d)
    m = (centered @ cov_inv * centered).sum(dim=1)  # quadratic form, (n_test,)
    return torch.sqrt(m.clamp_min(0.0))             # higher = more anomalous


def _mahalanobis_auc(backbone, Train, Test, eps=1e-3, device=DEVICE):
    """Fit a Gaussian on normal train features, score Test by Mahalanobis
    distance, and return the ROC-AUC against the true normal/abnormal labels."""
    tr, _ = _global_features(backbone, Train, device=device)
    te, y = _global_features(backbone, Test, device=device)
    mu, cov_inv = fit_gaussian(tr, eps=eps)
    scores = mahalanobis_score(te, mu, cov_inv)
    return roc_auc_score(y.numpy(), scores.numpy())


@torch.no_grad()
def extract_beats_features(model, loader, device=DEVICE):
    """Mean-pooled BEATs patch embeddings (B, D) per clip. BEATs ingests raw
    16 kHz waveform and computes its own kaldi-fbank internally."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        rep = model.extract_features(x.to(device), padding_mask=None)[0]  # (B, T', D)
        feats.append(rep.mean(dim=1).cpu())                               # time mean-pool
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def knn_score(test_feats, train_feats, k=5):
    """Anomaly score = 1 - mean cosine similarity to k nearest *normal* neighbours.
    More robust than a full-covariance Gaussian when the feature dim is large."""
    tr = F.normalize(train_feats, dim=1)
    te = F.normalize(test_feats, dim=1)
    sim = te @ tr.t()                                  # (n_test, n_train) cosine sim
    topk = sim.topk(min(k, tr.shape[0]), dim=1).values
    return 1.0 - topk.mean(dim=1)                      # higher = more anomalous
