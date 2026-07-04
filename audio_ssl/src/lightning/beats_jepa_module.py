from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from audio_ssl.src.models.beats_jepa.encoder import BEATsEncoder
from audio_ssl.src.models.jepa.masking import sample_block_mask


class LitBEATsJEPA(pl.LightningModule):
    """Continued SSL on MIMII with a pretrained BEATs encoder (data2vec-style JEPA):
    the EMA teacher encodes the full fbank; the student sees the same sequence with
    target-block patch embeddings replaced by a learnable mask token; a small head
    predicts the (layer-normalized) teacher tokens at the masked positions.

    This is the same latent-block-prediction principle as our scratch I-JEPA, but
    sequence-preserving (no token dropping, no separate predictor) because BEATs' conv
    positional embedding + relative position bias require intact sequences — and it is
    how BEATs itself was pretrained, so the backbone is on-distribution.

    Only the last `finetune_last_n` transformer layers (+ mask token + head) train; the
    rest stays frozen at the AudioSet weights. finetune_last_n=0 = frozen probe.
    Exposes target_encoder/context_encoder/normalize so the embedding eval, AUC monitor,
    and plotting stack work unchanged (cached fbanks are pre-normalized -> normalize is
    the identity).
    """

    def __init__(
        self,
        beats_cfg: dict,
        beats_checkpoint: str | None = None,
        target_frames: int = 998,
        finetune_last_n: int = 2,
        head_hidden: int = 768,
        num_blocks: int = 4,
        mask_scale: tuple[float, float] = (0.15, 0.3),
        mask_aspect: tuple[float, float] = (0.5, 2.0),
        ema_start: float = 0.999,
        ema_end: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.1,
        lr_schedule: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.student = BEATsEncoder(beats_cfg, target_frames, checkpoint_path=beats_checkpoint)
        # No deepcopy: BEATs' weight-normed pos_conv holds non-leaf tensors deepcopy rejects.
        self.teacher = BEATsEncoder(beats_cfg, target_frames, checkpoint_path=None)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        dim = self.student.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.head = nn.Sequential(nn.Linear(dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, dim))
        self.grid = self.student.grid

        # Freeze the student except the last N transformer layers.
        for p in self.student.parameters():
            p.requires_grad_(False)
        n = int(finetune_last_n)
        if n > 0:
            for layer in self.student.model.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
                # The relative attention bias is one module SHARED by reference across all
                # layers (backbone.py builds it in layer 0 and aliases it everywhere), so
                # unfreezing a late layer would silently train the bias of every frozen
                # layer too. Keep it frozen.
                if getattr(layer.self_attn, "has_relative_attention_bias", False):
                    for p in layer.self_attn.relative_attention_bias.parameters():
                        p.requires_grad_(False)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[beats-jepa] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params "
              f"(last {n} layers + mask token + head)", flush=True)

    # --- eval/monitor surface (embed_spectrograms, PeriodicAUCMonitor) ---
    @property
    def target_encoder(self):
        return self.teacher

    @property
    def context_encoder(self):
        return self.student

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x  # cached BEATs fbanks are already BEATs-normalized

    def _loss(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        target_idx, _ = sample_block_mask(
            self.grid[0], self.grid[1], self.hparams.num_blocks,
            self.hparams.mask_scale, self.hparams.mask_aspect, generator=generator)
        target_idx = target_idx.to(x.device)
        with torch.no_grad():
            targets = self.teacher(x)[:, target_idx, :]
            targets = F.layer_norm(targets, (targets.shape[-1],))  # data2vec target norm
        student_tokens = self.student.forward_masked(x, target_idx, self.mask_token)
        preds = self.head(student_tokens[:, target_idx, :])
        return F.smooth_l1_loss(preds, targets)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch[0])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        if batch[0].numel() == 0:
            return None
        generator = torch.Generator().manual_seed(1234 + batch_idx)  # stable val masks
        loss = self._loss(batch[0], generator=generator)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def _ema_update(self, momentum: float) -> None:
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            if sp.requires_grad:  # frozen params are identical by construction
                tp.mul_(momentum).add_(sp.detach(), alpha=1.0 - momentum)

    def on_train_batch_end(self, *args) -> None:
        total = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, self.global_step / total)
        momentum = self.hparams.ema_start + (self.hparams.ema_end - self.hparams.ema_start) * progress
        self._ema_update(momentum)

    def on_train_epoch_start(self) -> None:
        self.teacher.eval()  # never let Lightning flip teacher dropout back on

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(self.hparams.warmup_frac * total))
        constant = self.hparams.lr_schedule == "constant"

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            if constant:
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
