from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    """Pre-norm transformer block (MHSA + MLP)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch: tuple[int, int]):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, Hp, Wp)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class JEPAEncoder(nn.Module):
    """Patch-embed + learned positional embedding + transformer. Can encode all patches
    (target encoder) or a kept subset (context encoder)."""

    def __init__(
        self,
        n_mels: int,
        target_frames: int,
        patch: tuple[int, int],
        in_chans: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.grid = (n_mels // patch[0], target_frames // patch[1])
        self.num_patches = self.grid[0] * self.grid[1]
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = Transformer(embed_dim, depth, heads, mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        if keep_idx is not None:
            tokens = tokens[:, keep_idx, :]
        return self.norm(self.transformer(tokens))


class JEPAPredictor(nn.Module):
    """Predicts target-position embeddings from encoded context + learned mask tokens."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        pred_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = Transformer(pred_dim, depth, heads, mlp_ratio)
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, embed_dim)

    def forward(
        self, context_enc: torch.Tensor, context_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        batch = context_enc.size(0)
        ctx = self.input_proj(context_enc) + self.pos_embed[:, context_idx, :]
        mask = (self.mask_token + self.pos_embed[:, target_idx, :]).expand(batch, -1, -1)
        x = torch.cat([ctx, mask], dim=1)
        x = self.norm(self.transformer(x))
        pred = x[:, ctx.size(1) :, :]  # outputs at the target positions
        return self.output_proj(pred)  # (B, n_target, embed_dim)
