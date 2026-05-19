"""
architectures.py
================
CNN architectures for EMG gesture classification.

  SlowFusionEMG      — Option A:   replication of Côté-Allard et al. [1]
  SlowFusionSE_EMG   — Option A+:  SE channel attention after each block
  MobileNetV2EMG     — Option B:   honest ImageNet transfer learning baseline

Classifier heads:
  - Linear (default): standard logit head + CrossEntropyLoss
  - ArcFace:          additive angular margin head for inter-class separation

Input to all: (N, 4, 8, 14)  — (Batch, Time, Channel, Freq)
Output:       (N, n_classes)  — logits (or ArcFace cosines during training)

References
----------
[1]  Côté-Allard et al. (2019) "Deep Learning for EMG-Based ..."
[2]  Hu et al. (2018) "Squeeze-and-Excitation Networks." CVPR.
[3]  Altuwaijri et al. (2022) "Multi-Branch CNN with SE Attention
     Blocks for EEG-Based Motor Imagery." Diagnostics 12(4).
[4]  Deng et al. (2019) "ArcFace: Additive Angular Margin Loss
     for Deep Face Recognition." CVPR.
[5]  Song et al. (2024) "L3AM: Linear Adaptive Angular Margin
     Loss for Video-Based Hand Gesture Authentication." IJCV 132(9).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ════════════════════════════════════════════════════════════════
# Building blocks
# ════════════════════════════════════════════════════════════════

class SE3d(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps  [2][3].

    GAP over (T,H,W) → FC(C→C//r) → ReLU → FC(C//r→C) → Sigmoid
    Re-weights channels to emphasise discriminative frequency bands.
    Overhead: 32-ch block = +40 params, 128-ch block = +544 params.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, H, W)
        w = x.mean(dim=(2, 3, 4))                                   # squeeze → (N,C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # excite  → (N,C,1,1,1)
        return x * w


class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin classifier  [4][5].

    Replaces the final Linear with a cosine-similarity head that inserts
    angular margin m between a sample and its true class centre, widening
    the decision boundary between geometrically close gestures.

    Train:  forward(x, labels) → margin logits × s
    Eval:   forward(x)         → plain cosine logits × s  (no margin)

    Parameters
    ----------
    s : scale factor (default 30.0)
    m : angular margin in radians (default 0.50 ≈ 28.6°)
    """
    def __init__(self, in_features: int, n_classes: int,
                 s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s     = s
        self.m     = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)          # numerical stability threshold
        self.mm    = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(x_norm, w_norm)           # (N, n_classes)

        if labels is None or not self.training:
            return cosine * self.s

        sine    = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m      # cos(θ+m)
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


# ════════════════════════════════════════════════════════════════
# Option A — Slow-Fusion ConvNet  (Côté-Allard et al. 2019 [1])
# ════════════════════════════════════════════════════════════════

class SlowFusionEMG(nn.Module):
    """Baseline replication of [1]. Unchanged."""

    def __init__(self, n_classes: int = 17, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=0),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(2, 3, 3), padding=0),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )
        self.gap        = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# Option A+ / A++ — SlowFusion + SE  ±  ArcFace
# ════════════════════════════════════════════════════════════════

class SlowFusionSE_EMG(nn.Module):
    """
    SlowFusion with SE3d channel attention after every conv block  [2][3].

    use_arcface=False → "slow_fusion_se"      (SE only, Linear head)
    use_arcface=True  → "slow_fusion_se_arc"  (SE + ArcFace head)

    ArcFace forward contract:
      model.train(); model(x, labels=y)  → margin logits
      model.eval();  model(x)            → plain cosine logits
    """

    def __init__(self, n_classes: int = 17, dropout: float = 0.5,
                 se_reduction: int = 4, use_arcface: bool = False,
                 arc_s: float = 30.0, arc_m: float = 0.50):
        super().__init__()
        self.use_arcface = use_arcface

        # ── Conv blocks (identical to SlowFusionEMG) ────────
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=0),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
        )
        self.se1 = SE3d(32, se_reduction)

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.se2 = SE3d(64, se_reduction)

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(2, 3, 3), padding=0),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )
        self.se3 = SE3d(128, se_reduction)

        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        # ── Classifier head ──────────────────────────────────
        if use_arcface:
            self.classifier = ArcFaceHead(128, n_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.se1(self.block1(x))
        x = self.se2(self.block2(x))
        x = self.se3(self.block3(x))
        x = self.dropout(self.flatten(self.gap(x)))
        if self.use_arcface:
            return self.classifier(x, labels)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# Option B — MobileNetV2  (honest transfer learning baseline)
# ════════════════════════════════════════════════════════════════

class MobileNetV2EMG(nn.Module):
    """
    MobileNetV2 for 4-channel EMG spectrograms.
    freeze_until=7, differential LR groups, 64×64 upsample.
    """

    def __init__(self, n_classes: int = 17, dropout: float = 0.5,
                 freeze_until: int = 7):
        super().__init__()
        self._upsample    = (64, 64)
        self._freeze_until = freeze_until

        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        old = base.features[0][0]
        new_conv = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=False)
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        base.features[0][0] = new_conv

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, n_classes),
        )

        self.features   = base.features
        self.classifier = base.classifier

        for i, layer in enumerate(self.features):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """mid × 0.01  |  top × 0.1  |  classifier × 1.0"""
        mid_params, top_params = [], []
        for i, layer in enumerate(self.features):
            params = [p for p in layer.parameters() if p.requires_grad]
            if not params:
                continue
            (mid_params if i < 14 else top_params).extend(params)
        cls_params = [p for p in self.classifier.parameters() if p.requires_grad]
        return [
            {"params": mid_params, "lr": base_lr * 0.01},
            {"params": top_params, "lr": base_lr * 0.1},
            {"params": cls_params, "lr": base_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self._upsample, mode="bilinear", align_corners=False)
        x = self.features(x)
        return self.classifier(x.mean([2, 3]))


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

def build_model(name: str, n_classes: int = 17, dropout: float = 0.5) -> nn.Module:
    """
    name : "slow_fusion" | "slow_fusion_se" | "slow_fusion_se_arc" | "mobilenet"
    """
    if name == "slow_fusion":
        return SlowFusionEMG(n_classes=n_classes, dropout=dropout)
    elif name == "slow_fusion_se":
        return SlowFusionSE_EMG(n_classes=n_classes, dropout=dropout, use_arcface=False)
    elif name == "slow_fusion_se_arc":
        return SlowFusionSE_EMG(n_classes=n_classes, dropout=dropout, use_arcface=True)
    elif name == "mobilenet":
        return MobileNetV2EMG(n_classes=n_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {name!r}. "
                         "Choose 'slow_fusion', 'slow_fusion_se', "
                         "'slow_fusion_se_arc', or 'mobilenet'.")


# ════════════════════════════════════════════════════════════════
# Sanity check
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(8, 4, 8, 14).to(device)
    labels = torch.randint(0, 17, (8,)).to(device)

    for name in ("slow_fusion", "slow_fusion_se", "slow_fusion_se_arc", "mobilenet"):
        m = build_model(name).to(device)
        m.train()
        out = m(dummy, labels=labels) if (hasattr(m, "use_arcface") and m.use_arcface) else m(dummy)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name:20s}]  out={tuple(out.shape)}  params={total:,}  trainable={trainable:,}")