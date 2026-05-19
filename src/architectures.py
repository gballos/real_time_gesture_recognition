"""
architectures.py
================
Two CNN architectures for EMG gesture classification.

  SlowFusionEMG   — Option A: replication of Côté-Allard et al. [1]
                    Trained from scratch. Conv3d slow temporal fusion.

  MobileNetV2EMG  — Option B: honest ImageNet transfer learning baseline.
                    4-ch input, upsampled to 64×64, early layers frozen.

Input to both: (N, 4, 8, 14)  — (Batch, Time, Channel, Freq)
Output:        (N, n_classes)  — logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ════════════════════════════════════════════════════════════════
# Option A — Slow-Fusion ConvNet  (Côté-Allard et al. 2019 [1])
# ════════════════════════════════════════════════════════════════

class SlowFusionEMG(nn.Module):
    """
    Slow-fusion ConvNet.

    Time is progressively collapsed across three Conv3d blocks:
      Block 1 — kernel_t=1  (each frame processed independently)
      Block 2 — kernel_t=2  (adjacent frames begin to merge)
      Block 3 — kernel_t=2  (remaining time fully collapsed)

    No pretrained weights. Dropout(p) for MC-Dropout regularisation.

    Input : (N,  4,  8, 14)
    Output: (N, n_classes)
    """

    def __init__(self, n_classes: int = 17, dropout: float = 0.5):
        super().__init__()

        # Block 1 — spatial only, time untouched
        # (N,1,4,8,14) → (N,32,4,6,12)
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Block 2 — begin temporal fusion
        # (N,32,4,6,12) → (N,64,2,4,10)
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Block 3 — full temporal fusion
        # (N,64,2,4,10) → (N,128,1,2,8)
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.gap        = nn.AdaptiveAvgPool3d(1)  # (N,128,1,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)   # (N,4,8,14) → (N,1,4,8,14)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# Option B — MobileNetV2  (honest transfer learning baseline)
# ════════════════════════════════════════════════════════════════

class MobileNetV2EMG(nn.Module):
    """
    MobileNetV2 adapted for 4-channel EMG spectrograms.

    Transfer learning strategy
    --------------------------
    - features[0..freeze_until-1] : FROZEN  (generic low-level features)
    - features[freeze_until..]    : fine-tuned
    - classifier                  : fine-tuned, rebuilt for n_classes

    First conv: 3 → 4 ch, Kaiming init (no RGB weight copying).
    Input upsampled to 64×64 to prevent spatial collapse.
    """

    def __init__(self, n_classes: int = 17, dropout: float = 0.5,
                 freeze_until: int = 7):
        super().__init__()
        self._upsample = (64, 64)

        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Replace first conv: 3 → 4 channels (fresh Kaiming init)
        old = base.features[0][0]
        new_conv = nn.Conv2d(
            4, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        base.features[0][0] = new_conv

        # Rebuild classifier
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )

        self.features    = base.features
        self.classifier  = base.classifier
        self._freeze_until = freeze_until

        # Freeze early layers
        for i, layer in enumerate(self.features):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Three parameter groups with differential learning rates:
          - mid layers  features[_freeze_until..13]  : base_lr * 0.01
          - top layer   features[14..]               : base_lr * 0.1
          - classifier                               : base_lr

        Only includes parameters with requires_grad=True.
        """
        mid_params, top_params = [], []
        for i, layer in enumerate(self.features):
            params = [p for p in layer.parameters() if p.requires_grad]
            if not params:
                continue
            if i < 14:
                mid_params.extend(params)
            else:
                top_params.extend(params)
        cls_params = [p for p in self.classifier.parameters() if p.requires_grad]
        return [
            {"params": mid_params, "lr": base_lr * 0.01},
            {"params": top_params, "lr": base_lr * 0.1},
            {"params": cls_params, "lr": base_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self._upsample,
                          mode="bilinear", align_corners=False)
        x = self.features(x)
        x = x.mean([2, 3])          # global average pool
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

def build_model(name: str, n_classes: int = 17,
                dropout: float = 0.5) -> nn.Module:
    """
    Parameters
    ----------
    name : "slow_fusion" | "mobilenet"
    """
    if name == "slow_fusion":
        return SlowFusionEMG(n_classes=n_classes, dropout=dropout)
    elif name == "mobilenet":
        return MobileNetV2EMG(n_classes=n_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {name!r}. "
                         "Choose 'slow_fusion' or 'mobilenet'.")


# ════════════════════════════════════════════════════════════════
# Sanity check
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(8, 4, 8, 14).to(device)

    for name in ("slow_fusion", "mobilenet"):
        m = build_model(name).to(device)
        out = m(dummy)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name:12s}]  out={tuple(out.shape)}  "
              f"params={total:,}  trainable={trainable:,}")