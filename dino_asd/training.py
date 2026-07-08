"""Training loops: ResNet-DINO from scratch and BEATs-initialised DINO fine-tune."""
import os
import copy

import torch

from .config import DEVICE, SAVE_DIR, BACKBONE_DIM
from .features import make_resnet18, multi_crop, multi_crop_wave
from .models import (DINOHead, MultiCropWrapper, DINOLoss, ema_update,
                     BEATsBackbone, freeze_beats_except_last)
from .evaluation import _mahalanobis_auc


def dino_trainer(Train, Test=None, num_epochs=400, n_global=2, n_local=6, out_dim=1024,
                 momentum_teacher=0.996, lr=5e-4, weight_decay=0.04,
                 verbosity=1, pretrain=False, save_path=os.path.join(SAVE_DIR, 'dino_student.pt'),
                 use_extra_augs=False, eval_every=10):
    """Train the ResNet DINO student/teacher on one machine's normal clips.

    With `pretrain=True`, the saved backbone is loaded from `save_path` and
    returned without training. If `Test` is given, Mahalanobis ROC-AUC is
    measured every `eval_every` epochs and stored on the returned student as
    `student.dino_history` (a dict with 'epoch', 'loss', 'auc'), which lets the
    training loss and AUC be compared. Returns (student, teacher).
    """
    student = MultiCropWrapper(make_resnet18(), DINOHead(BACKBONE_DIM, out_dim)).to(DEVICE)
    teacher = MultiCropWrapper(make_resnet18(), DINOHead(BACKBONE_DIM, out_dim)).to(DEVICE)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    if pretrain:
        student.backbone.load_state_dict(
            torch.load(save_path, map_location=DEVICE, weights_only=True))
        if verbosity > 0:
            print(f"Backbone loaded from {save_path}")
        return student, teacher

    dino_loss = DINOLoss(out_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "loss": [], "auc": []}
    for epoch in range(num_epochs):
        student.train()
        epoch_loss, num_batch = 0, 0
        for x, _ in Train:                                  # labels ignored (self-supervised)
            crops = [c.to(DEVICE) for c in multi_crop(x, n_global, n_local, train=True, use_extra_augs=use_extra_augs)]
            # Mixed-precision (bf16) forward: faster and lower memory. bfloat16
            # needs no GradScaler, and softmax/log_softmax stay in fp32.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                teacher_out = teacher(crops[:n_global])     # global crops only
                student_out = student(crops)                # all crops
                loss = dino_loss(student_out, teacher_out, n_global + n_local, n_global)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(student, teacher, momentum_teacher)  # EMA teacher update

            epoch_loss += loss.item()
            num_batch += 1

        avg_loss = epoch_loss / num_batch
        if Test is not None and epoch % eval_every == eval_every - 1:
            auc = _mahalanobis_auc(student.backbone, Train, Test)
            history["epoch"].append(epoch + 1)
            history["loss"].append(avg_loss)
            history["auc"].append(auc)
            if verbosity > 0:
                print(f"epoch : {epoch + 1}/{num_epochs}, loss = {avg_loss:.6f}, AUC = {auc:.4f}")
        elif verbosity > 0 and epoch % 10 == 9:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, avg_loss))

    student.dino_history = history
    torch.save(student.backbone.state_dict(), save_path)
    if verbosity > 0:
        print(f"Backbone saved to {save_path}")
    return student, teacher


def beats_dino_trainer(Train, beats, num_epochs=30, n_global=2, n_local=4,
                       out_dim=1024, momentum_teacher=0.996, lr=1e-5,
                       weight_decay=0.04, n_unfrozen_layers=2, verbosity=1,
                       save_path=None):
    """DINO self-distillation with a pretrained BEATs backbone (partial fine-tune).

    Only the top `n_unfrozen_layers` BEATs blocks and the DINO head are trained.
    If `save_path` is given, the fine-tuned BEATs backbone state_dict is saved
    there (reload with `m = BEATs(beats_cfg); m.load_state_dict(torch.load(path))`).
    Returns (student, teacher).
    """
    in_dim = beats.cfg.encoder_embed_dim

    student_beats = freeze_beats_except_last(copy.deepcopy(beats), n_unfrozen_layers)
    teacher_beats = copy.deepcopy(student_beats)

    student = MultiCropWrapper(BEATsBackbone(student_beats), DINOHead(in_dim, out_dim)).to(DEVICE)
    teacher = MultiCropWrapper(BEATsBackbone(teacher_beats), DINOHead(in_dim, out_dim)).to(DEVICE)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(out_dim).to(DEVICE)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    if verbosity > 0:
        print(f"BEATs+DINO: fine-tuning {sum(p.numel() for p in trainable)/1e6:.1f}M "
              f"trainable params ({n_unfrozen_layers} top layers + head)")

    for epoch in range(num_epochs):
        student.train()
        epoch_loss, num_batch = 0, 0
        for x, _ in Train:
            crops = multi_crop_wave(x, n_global, n_local, train=True)
            # Mixed-precision (bf16) forward: faster and lower memory for the
            # BEATs transformer. bfloat16 needs no GradScaler; softmax stays fp32.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                teacher_out = teacher(crops[:n_global])     # global crops only
                student_out = student(crops)                # all crops
                loss = dino_loss(student_out, teacher_out, n_global + n_local, n_global)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(student, teacher, momentum_teacher)  # frozen layers are a no-op

            epoch_loss += loss.item()
            num_batch += 1
        if verbosity > 0 and epoch % 5 == 4:
            print(f"  epoch {epoch + 1}/{num_epochs}, loss = {epoch_loss / num_batch:.4f}")

    if save_path is not None:
        # Save only the fine-tuned BEATs backbone (what downstream scoring uses).
        torch.save(student.backbone.beats.state_dict(), save_path)
        if verbosity > 0:
            print(f"Fine-tuned BEATs backbone saved to {save_path}")

    return student, teacher
