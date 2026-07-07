"""DINO components (head, multi-crop wrapper, loss, EMA) and the BEATs backbone adapter."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """Projection head mapping backbone features to the DINO output space:
    an MLP, then an L2-normalised bottleneck, then a weight-normalised linear
    layer. The weight-norm magnitude is frozen to 1 for training stability."""
    def __init__(self, in_dim, out_dim=1024, hidden_dim=512, bottleneck_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.utils.parametrizations.weight_norm(self.last_layer, name='weight')

        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False  # freeze norm for stability

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    """Runs backbone+head over a list of crops, batching equal-shaped crops
    into a single forward pass (as in the original DINO)."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops):
        if not isinstance(crops, (list, tuple)):
            crops = [crops]
        # Group consecutive crops of equal size so each group is one backbone call.
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
    """DINO self-distillation loss.

    Cross-entropy between the sharpened, centred teacher distribution and the
    student distribution, averaged over all (teacher-global crop, student crop)
    pairs except identical ones. Centering (an EMA of the teacher outputs) plus
    temperature sharpening prevent representational collapse.
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out, n_student_crops, n_teacher_crops):
        student = (student_out / self.student_temp).chunk(n_student_crops)
        teacher = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        teacher = teacher.detach().chunk(n_teacher_crops)

        total, n_terms = 0.0, 0
        for iq, q in enumerate(teacher):           # teacher (global) crops
            for iv, v in enumerate(student):       # all student crops
                if iv == iq:
                    continue                       # skip identical crop pair
                total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                n_terms += 1
        self.update_center(teacher_out)
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_out):
        """EMA update of the centering vector subtracted from teacher outputs."""
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center * (1 - self.center_momentum))


@torch.no_grad()
def ema_update(student, teacher, m):
    """Update the teacher as an exponential moving average of the student:
    teacher = m * teacher + (1 - m) * student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.detach().data, alpha=1 - m)


class BEATsBackbone(nn.Module):
    """Adapts BEATs to the DINO backbone interface: a raw-waveform crop (B, L)
    -> mean-pooled clip embedding (B, D). BEATs computes its own fbank inside
    extract_features, so crops here live in the time (waveform) domain."""
    def __init__(self, beats):
        super().__init__()
        self.beats = beats

    def forward(self, wav):
        rep = self.beats.extract_features(wav, padding_mask=None)[0]  # (B, T', D)
        return rep.mean(dim=1)                                        # (B, D)


def freeze_beats_except_last(beats, n_unfrozen_layers=2):
    """Freeze all of BEATs, then re-enable grads on the top-n encoder blocks.
    Keeps the fine-tune cheap and stable on ~760 clips per machine id."""
    for p in beats.parameters():
        p.requires_grad = False
    for blk in beats.encoder.layers[-n_unfrozen_layers:]:
        for p in blk.parameters():
            p.requires_grad = True
    return beats
