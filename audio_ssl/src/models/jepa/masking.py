from __future__ import annotations

import torch


def sample_block_mask(
    grid_h: int,
    grid_w: int,
    num_blocks: int = 4,
    scale_range: tuple[float, float] = (0.15, 0.3),
    aspect_range: tuple[float, float] = (0.5, 2.0),
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample I-JEPA-style time-frequency block masks on the patch grid.

    Returns (target_idx, context_idx) as 1-D LongTensors of flat patch indices. The
    union of `num_blocks` rectangular blocks is the prediction target; the complement is
    the context. The same mask is used for the whole batch so tensors stay rectangular.
    """
    total = grid_h * grid_w

    def rnd(lo: float, hi: float) -> float:
        return float(torch.rand((), generator=generator) * (hi - lo) + lo)

    is_target = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    for _ in range(num_blocks):
        area = rnd(*scale_range) * total
        aspect = rnd(*aspect_range)
        h = max(1, min(grid_h, int(round((area * aspect) ** 0.5))))
        w = max(1, min(grid_w, int(round((area / aspect) ** 0.5))))
        top = int(torch.randint(0, grid_h - h + 1, (), generator=generator))
        left = int(torch.randint(0, grid_w - w + 1, (), generator=generator))
        is_target[top : top + h, left : left + w] = True

    flat = is_target.flatten()
    if flat.all():  # never leave the context empty
        flat[0] = False
    if not flat.any():  # never leave the target empty
        flat[total // 2] = True

    all_idx = torch.arange(total)
    return all_idx[flat], all_idx[~flat]


def tile_blocks(grid_h: int, grid_w: int, block_h: int, block_w: int) -> list[torch.Tensor]:
    """Deterministic non-overlapping tiling of the grid into blocks (for leave-block-out
    scoring). Returns a list of 1-D LongTensors of flat patch indices, one per tile."""
    grid = torch.arange(grid_h * grid_w).reshape(grid_h, grid_w)
    blocks = []
    for top in range(0, grid_h, block_h):
        for left in range(0, grid_w, block_w):
            tile = grid[top : top + block_h, left : left + block_w].reshape(-1)
            if tile.numel() > 0:
                blocks.append(tile)
    return blocks
