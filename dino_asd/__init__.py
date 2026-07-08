"""Acoustic anomaly detection (MIMII) with DINO self-distillation.

Modules:
    config      constants, paths, seeding, device
    data        MIMII dataset and DataLoader construction
    features    log-mel features, SpecAugment, DINO multi-crop, ResNet backbone
    models      DINO head, multi-crop wrapper, loss, EMA update, BEATs backbone
    training    ResNet-DINO and BEATs-DINO training loops
    evaluation  feature extraction and anomaly scoring (Mahalanobis, kNN)
"""
