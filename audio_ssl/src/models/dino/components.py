from __future__ import annotations

"""DINO head, multi-crop wrapper, and loss — ported from dino_asd_v2.ipynb, with one
DDP fix: the loss center is all-reduced across ranks (per-rank centering drifts)."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> weight-normed prototype layer (norm frozen at 1)."""

    def __init__(self, in_dim: int, out_dim: int = 1024, hidden_dim: int = 512,
                 bottleneck_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.utils.parametrizations.weight_norm(self.last_layer, name="weight")
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    """Backbone + head over a list of crops, batching equal-shaped crops into single
    forward passes (as in the original DINO)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops) -> torch.Tensor:
        if not isinstance(crops, (list, tuple)):
            crops = [crops]
        idx_splits = [0]
        for i in range(1, len(crops)):
            if crops[i].shape[-1] != crops[i - 1].shape[-1]:
                idx_splits.append(i)
        idx_splits.append(len(crops))
        out = []
        for s, e in zip(idx_splits[:-1], idx_splits[1:]):
            out.append(self.backbone(torch.cat(crops[s:e])))
        return self.head(torch.cat(out))


class DINOLoss(nn.Module):
    """Cross-entropy between centered/sharpened teacher and student distributions over
    (teacher global crop, student crop) pairs, skipping identical-view pairs."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor,
                n_student_crops: int, n_teacher_crops: int) -> torch.Tensor:
        student = (student_out / self.student_temp).chunk(n_student_crops)
        teacher = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        teacher = teacher.detach().chunk(n_teacher_crops)

        total, n_terms = 0.0, 0
        for iq, q in enumerate(teacher):
            for iv, v in enumerate(student):
                if iv == iq:
                    continue
                total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                n_terms += 1
        self.update_center(teacher_out)
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))
