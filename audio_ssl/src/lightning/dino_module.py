from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from audio_ssl.src.models.dino.backbones import build_backbone
from audio_ssl.src.models.dino.components import DINOHead, DINOLoss, MultiCropWrapper
from audio_ssl.src.models.dino.crops import multi_crop


class _BackboneAdapter(nn.Module):
    """Presents a DINO backbone as a token encoder for the shared eval stack:
    embed_spectrograms does `enc(x).mean(dim=1)`, so return (B, 1, D). For the fixed-
    input ResNet (freq crops) the full spectrogram is resized to the training crop size;
    variable-length backbones (BEATs, time crops) get the native full input."""

    def __init__(self, backbone: nn.Module, out_size: tuple[int, int], resize: bool):
        super().__init__()
        self.backbone = backbone
        self.out_size = out_size
        self.resize = resize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.backbone(x).unsqueeze(1)  # (B, 1, D)


class LitDINO(pl.LightningModule):
    """DINO on log-mel spectrograms (ported from dino_asd_v2.ipynb): student/teacher
    with frequency-band multi-crop self-distillation; anomaly scoring later uses the
    backbone features (embedding-distance), the head/prototypes are pretext-only.

    Backbone is config-swappable (resnet18 now; ViT / pretrained BEATs planned — see
    models/dino/backbones.py). Exposes target_encoder (EMA teacher backbone),
    context_encoder (student backbone), and per-sample instance `normalize`, so
    eval_jepa_embedding / PeriodicAUCMonitor / plotting work unchanged.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        backbone_cfg: dict | None = None,
        out_dim: int = 1024,
        head_hidden: int = 512,
        head_bottleneck: int = 64,
        n_global: int = 2,
        n_local: int = 6,
        global_frac: float = 0.6,
        local_frac: float = 0.25,
        crop_axis: str = "freq",     # freq (ResNet, resized band crops) | time (BEATs)
        input_norm: str = "instance",  # instance (raw log-mels) | none (pre-normed fbanks)
        crop_mels: int = 128,
        crop_frames: int = 128,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        augment: dict | None = None,
        lr: float = 5e-4,
        weight_decay: float = 0.04,
        warmup_frac: float = 0.0,
        lr_schedule: str = "constant",
    ):
        super().__init__()
        self.save_hyperparameters()
        s_backbone, feat_dim = build_backbone(backbone, backbone_cfg)
        t_backbone, _ = build_backbone(backbone, backbone_cfg)
        self.feat_dim = feat_dim
        self.student = MultiCropWrapper(s_backbone, DINOHead(feat_dim, out_dim, head_hidden, head_bottleneck))
        self.teacher = MultiCropWrapper(t_backbone, DINOHead(feat_dim, out_dim, head_hidden, head_bottleneck))
        self.teacher.load_state_dict(self.student.state_dict())  # no deepcopy: weight_norm
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.dino_loss = DINOLoss(out_dim, teacher_temp, student_temp, center_momentum)
        self.augment_cfg = dict(augment or {})
        self._crop_size = (int(crop_mels), int(crop_frames))
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        print(f"[dino] backbone={backbone}: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M "
              f"student params", flush=True)

    # --- eval/monitor surface ---
    @property
    def target_encoder(self):
        return _BackboneAdapter(self.teacher.backbone, self._crop_size,
                                resize=self.hparams.crop_axis == "freq")

    @property
    def context_encoder(self):
        return _BackboneAdapter(self.student.backbone, self._crop_size,
                                resize=self.hparams.crop_axis == "freq")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """instance: per-sample normalization (raw log-mels, notebook convention);
        none: identity (BEATs fbank cache is already BEATs-normalized)."""
        if self.hparams.input_norm == "none":
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        return (x - mean) / std

    def _loss(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        crops = multi_crop(
            self.normalize(x), self.hparams.n_global, self.hparams.n_local,
            self.hparams.global_frac, self.hparams.local_frac,
            out_size=self._crop_size, augment_cfg=self.augment_cfg, train=train,
            crop_axis=self.hparams.crop_axis)
        with torch.no_grad():  # loss detaches the teacher anyway; skip its graph
            teacher_out = self.teacher(crops[: self.hparams.n_global])
        student_out = self.student(crops)
        return self.dino_loss(student_out, teacher_out,
                              self.hparams.n_global + self.hparams.n_local, self.hparams.n_global)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch[0])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        if batch[0].numel() == 0:
            return None
        # NOTE: crop randomness is not seeded here (unlike JEPA's val masks); val_loss is
        # a noisy diagnostic — monitor_AUC is the model-selection signal for DINO.
        loss = self._loss(batch[0], train=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def _ema_update(self, momentum: float) -> None:
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.mul_(momentum).add_(sp.detach(), alpha=1.0 - momentum)

    def on_train_batch_end(self, *args) -> None:
        self._ema_update(float(self.hparams.ema_momentum))

    # NOTE deliberately NO teacher.eval() forcing here (unlike the LayerNorm-based BEATs
    # module): ResNet teachers must run in TRAIN mode during training so BatchNorm uses
    # batch statistics — in eval mode BN normalizes with its init-frozen running stats
    # (EMA copies parameters, not buffers), the teacher outputs near-constant garbage,
    # centering flattens it to uniform, and the loss pins at ln(out_dim) = collapse.
    # This exact failure burned run dino_fan_20260704_plucky_raven_8654 (loss 6.931 =
    # ln 1024). In train mode the teacher also accumulates its own BN running stats,
    # which is what makes it usable in eval mode for embedding extraction later.

    def configure_optimizers(self):
        params = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(self.hparams.warmup_frac * total)) if self.hparams.warmup_frac > 0 else 0
        constant = self.hparams.lr_schedule == "constant"

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return step / warmup
            if constant:
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
