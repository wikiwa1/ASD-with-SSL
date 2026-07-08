from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from audio_ssl.src.models.beats_jepa.BEATs import BEATs, BEATsConfig


class BEATsEncoder(nn.Module):
    """Pretrained BEATs wrapped with the JEPA-encoder surface our eval stack expects:
    `forward(x)` takes (B, 1, 128, T) cached fbanks (log-mel orientation) and returns
    full-sequence tokens (B, N, 768); `forward_masked` substitutes a mask token at target
    positions AFTER patch-embed + projection but BEFORE the transformer, so BEATs' conv
    positional embedding and relative position bias see an intact sequence (token
    dropping, as in our scratch I-JEPA, would corrupt both).

    Architecture is built from `beats_cfg` (a plain dict, checkpointable in hparams);
    pretrained weights are loaded only when `checkpoint_path` is given — Lightning
    restores finetuned weights on top at eval time, so the .pt is only needed at
    training time.
    """

    def __init__(self, beats_cfg: dict, target_frames: int = 998,
                 checkpoint_path: str | Path | None = None):
        super().__init__()
        self.cfg = BEATsConfig(dict(beats_cfg))
        self.cfg.finetuned_model = False  # never want the AudioSet classifier head
        self.model = BEATs(self.cfg)
        if checkpoint_path is not None:
            state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
            unexpected = [k for k in unexpected if not k.startswith("predictor")]
            if missing or unexpected:
                raise RuntimeError(f"BEATs load mismatch: missing={missing} unexpected={unexpected}")

        patch = int(self.cfg.input_patch_size)
        self.grid = (target_frames // patch, 128 // patch)  # (time_patches, freq_patches)
        self.num_patches = self.grid[0] * self.grid[1]
        self.embed_dim = int(self.cfg.encoder_embed_dim)

    @staticmethod
    def load_pretrained_cfg(checkpoint_path: str | Path) -> dict:
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        return dict(state["cfg"])

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, T) cached fbank -> projected patch tokens (B, N, D), pre-transformer.
        BEATs patch-embeds (B, 1, frames, 128), so transpose the fbank image first; token
        order is then time-major (index = t * freq_patches + f), matching self.grid."""
        fbank = x.squeeze(1).transpose(1, 2).unsqueeze(1)  # (B, 1, T, 128)
        feats = self.model.patch_embedding(fbank)          # (B, 512, T', F')
        feats = feats.flatten(2).transpose(1, 2)           # (B, N, 512)
        feats = self.model.layer_norm(feats)
        if self.model.post_extract_proj is not None:
            feats = self.model.post_extract_proj(feats)
        return feats                                       # (B, N, 768)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.model.dropout_input(tokens)
        out, _ = self.model.encoder(tokens, padding_mask=None)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(self._patch_tokens(x))

    def forward_masked(self, x: torch.Tensor, target_idx: torch.Tensor,
                       mask_token: torch.Tensor) -> torch.Tensor:
        tokens = self._patch_tokens(x)
        tokens[:, target_idx, :] = mask_token.to(tokens.dtype)
        return self._encode(tokens)
