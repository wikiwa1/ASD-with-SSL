from __future__ import annotations

"""DINO backbone registry. A backbone maps a (B, 1, bins, T) spectrogram crop to (B, D)
features. Still planned: our JEPA ViT (mean-pooled tokens).
"""

from torch import nn


class _BEATsDinoBackbone(nn.Module):
    """Pretrained BEATs as a DINO backbone (notebook's BEATsBackbone): (B, 1, 128, T)
    cached fbank crop -> mean-pooled tokens (B, 768). Variable T is fine."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x).mean(dim=1)


def build_backbone(name: str, cfg: dict | None = None) -> tuple[nn.Module, int]:
    """Returns (backbone, feature_dim). `cfg` carries backbone-specific keys
    (beats: beats_cfg / beats_checkpoint / finetune_last_n)."""
    name = str(name).lower()
    cfg = cfg or {}
    if name == "resnet18":
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Identity()
        return m, 512
    if name == "beats":
        from audio_ssl.src.models.beats_jepa.encoder import BEATsEncoder

        beats_cfg = cfg.get("beats_cfg")
        if beats_cfg is None:
            beats_cfg = BEATsEncoder.load_pretrained_cfg(cfg["beats_checkpoint"])
        enc = BEATsEncoder(beats_cfg, target_frames=int(cfg.get("target_frames", 998)),
                           checkpoint_path=cfg.get("beats_checkpoint"))
        # Notebook's freeze_beats_except_last + the shared-relative-position-bias fix
        # from LitBEATsJEPA (the bias module is aliased into every layer; unfreezing a
        # late layer would otherwise silently train all layers' attention bias).
        for p in enc.parameters():
            p.requires_grad_(False)
        n = int(cfg.get("finetune_last_n", 2))
        if n > 0:
            for layer in enc.model.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
                if getattr(layer.self_attn, "has_relative_attention_bias", False):
                    for p in layer.self_attn.relative_attention_bias.parameters():
                        p.requires_grad_(False)
        return _BEATsDinoBackbone(enc), enc.embed_dim
    raise ValueError(f"unknown DINO backbone '{name}' (available: resnet18, beats)")
