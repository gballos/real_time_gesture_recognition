"""
architectures.py
================
CNN/TCN architectures for EMG gesture classification.

  SlowFusionEMG      — Option A:  Côté-Allard et al. [1] (Conv3d, STFT input)
  TCN_EMG            — Option C:  Tsinganos et al. [9] (causal dilated 1D conv,
                                  raw EMG input, AoT or Attention classifier)
  MobileNetV2EMG     — Option B:  ImageNet transfer learning baseline (STFT input)

Input conventions:
  SlowFusion/MobileNet: (N, 4, 8, 14) — STFT spectrogram
  TCN:                  (N, 8, T)      — raw EMG (8 channels, T time steps)

References
----------
[1]  Côté-Allard et al. (2019) "Deep Learning for EMG-Based ..."
[9]  Tsinganos et al. (2019) "Improved Gesture Recognition Based on
     sEMG Signals and TCN." ICASSP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ════════════════════════════════════════════════════════════════
# TCN building blocks  (Tsinganos et al. [9], Bai et al. [6])
# ════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """
    Causal (left-padded) dilated 1D convolution.
    Output length == input length for any dilation.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad so output only sees past + current
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    One residual block: two causal dilated convs + residual connection.

    CausalConv → BN → ReLU → Dropout → CausalConv → BN → ReLU → Dropout
                                    + residual (1x1 conv if channels differ)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.downsample(x))


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially growing dilation.

    Parameters
    ----------
    in_channels  : int — number of input channels (8 for EMG)
    n_hidden     : int — hidden channel width for all blocks
    n_layers     : int — number of TemporalBlocks (dilation = 2^i)
    kernel_size  : int — conv kernel width (default 3)
    dropout      : float
    """
    def __init__(self, in_channels: int = 8, n_hidden: int = 32,
                 n_layers: int = 4, kernel_size: int = 3,
                 dropout: float = 0.05):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch  = in_channels if i == 0 else n_hidden
            layers.append(TemporalBlock(in_ch, n_hidden, kernel_size,
                                        dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)       # (N, n_hidden, T)


class AttentionPool(nn.Module):
    """
    Attention-based temporal pooling  [9, Yang et al. 2016].

    v_n   = tanh(W_a · h_n + b_a)          — hidden representation
    a_n   = softmax(v_n^T · u_a)            — scalar importance per timestep
    s     = sum_n(a_n · h_n)               — weighted summary vector

    u_a is initialised to zeros so initial scores are uniform across
    all timesteps, preventing the attention from collapsing onto a
    single timestep before the network has learned anything useful.
    """
    def __init__(self, n_hidden: int):
        super().__init__()
        self.W_a = nn.Linear(n_hidden, n_hidden)
        self.u_a = nn.Parameter(torch.zeros(n_hidden))   # zeros → uniform init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T) → (N, T, C)
        x = x.transpose(1, 2)
        v      = torch.tanh(self.W_a(x))               # (N, T, C)
        scores = (v * self.u_a).sum(dim=2)              # (N, T)
        alpha  = torch.softmax(scores, dim=1).unsqueeze(2)  # (N, T, 1)
        return (x * alpha).sum(dim=1)                   # (N, C)


class TCN_EMG(nn.Module):
    """
    TCN for raw EMG gesture classification  [9].

    Input:  (N, 8, T) — raw 8-channel EMG, variable length T
    Output: (N, n_classes)

    Parameters
    ----------
    classifier : "aot" | "attention"
        - "aot":       Average-over-Time (mean pool across T)
        - "attention":  Learned attention pooling
    n_layers   : int — 4 for short RF (~300ms), 7 for long RF (~2500ms)
    n_hidden   : int — channel width (default 32)
    """
    def __init__(self, n_classes: int = 17, n_channels: int = 8,
                 n_hidden: int = 32, n_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.05,
                 classifier: str = "aot"):
        super().__init__()
        self.classifier_type = classifier

        self.tcn = TemporalConvNet(
            in_channels=n_channels, n_hidden=n_hidden,
            n_layers=n_layers, kernel_size=kernel_size,
            dropout=dropout,
        )

        if classifier == "attention":
            self.pool = AttentionPool(n_hidden)
        else:
            self.pool = None   # AoT: simple mean

        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 8, T)
        h = self.tcn(x)                              # (N, n_hidden, T)
        if self.pool is not None:
            s = self.pool(h)                          # (N, n_hidden)
        else:
            s = h.mean(dim=2)                         # AoT: (N, n_hidden)
        return self.fc(s)                             # (N, n_classes)


# ════════════════════════════════════════════════════════════════
# Option A — Slow-Fusion ConvNet  (Côté-Allard et al. 2019 [1])
# ════════════════════════════════════════════════════════════════

class SlowFusionEMG(nn.Module):
    """Baseline replication of [1]. Input: (N, 4, 8, 14) STFT."""

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
# Option B — MobileNetV2  (honest transfer learning baseline)
# ════════════════════════════════════════════════════════════════

class MobileNetV2EMG(nn.Module):
    """
    MobileNetV2 adapted for EMG spectrograms.

    Input layout: (N, 4, 8, 14) — (Batch, Time, Electrodes, Freq)

    Reshaping strategy (Option 3 — EMG image layout):
      (N, 4, 8, 14) → permute → (N, 8, 4, 14)
                    → reshape → (N, 1, 8, 56)   [electrode rows × time-freq cols]
                    → upsample → (N, 1, 64, 64)
                    → expand → (N, 3, 64, 64)   [replicate for pretrained conv]

    References
    ----------
    [10] Geng et al. (2016) "Gesture recognition by instantaneous
         surface EMG images." Sci. Rep. 6, 36571.
    """

    def __init__(self, n_classes: int = 17, dropout: float = 0.5,
                 freeze_until: int = 7):
        super().__init__()
        self._upsample     = (64, 64)
        self._freeze_until = freeze_until

        # Load pretrained weights — keep first conv as 3-channel
        # (we replicate the single-channel input 3× at runtime)
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, n_classes),
        )

        self.features   = base.features   # first conv stays 3-channel
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
        # x: (N, 4, 8, 14)
        N = x.shape[0]

        # Reshape to EMG image: electrode rows × (time-freq) cols  [10]
        x = x.permute(0, 2, 1, 3)           # (N, 8, 4, 14)
        x = x.reshape(N, 1, 8, 56)          # (N, 1, 8, 56)

        # Upsample to 64×64
        x = F.interpolate(x, size=self._upsample,
                          mode="bilinear", align_corners=False)  # (N, 1, 64, 64)

        # Replicate single channel 3× to match pretrained first conv weights
        # Preserves all ImageNet initialisation — no Kaiming reinit needed
        x = x.expand(-1, 3, -1, -1)         # (N, 3, 64, 64)

        x = self.features(x)
        return self.classifier(x.mean([2, 3]))

# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

# Which architectures need STFT features vs raw EMG windows
STFT_ARCHS = {"slow_fusion", "mobilenet"}
RAW_ARCHS  = {"tcn_aot", "tcn_att"}


def build_model(name: str, n_classes: int = 17,
                dropout: float = 0.5) -> nn.Module:
    """
    name : "slow_fusion" | "mobilenet" | "tcn_aot" | "tcn_att"
    """
    if name == "slow_fusion":
        return SlowFusionEMG(n_classes=n_classes, dropout=dropout)
    elif name == "mobilenet":
        return MobileNetV2EMG(n_classes=n_classes, dropout=dropout)
    elif name == "tcn_aot":
        return TCN_EMG(n_classes=n_classes, n_hidden=32, n_layers=4,
                       dropout=0.05, classifier="aot")
    elif name == "tcn_att":
        return TCN_EMG(n_classes=n_classes, n_hidden=32, n_layers=4,
                       dropout=0.05, classifier="attention")
    else:
        raise ValueError(f"Unknown architecture: {name!r}. "
                         "Choose 'slow_fusion', 'mobilenet', 'tcn_aot', or 'tcn_att'.")


# ════════════════════════════════════════════════════════════════
# Sanity check
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # STFT architectures
    dummy_stft = torch.randn(8, 4, 8, 14).to(device)
    for name in ("slow_fusion",):
        m = build_model(name).to(device)
        out = m(dummy_stft)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name:15s}]  in={tuple(dummy_stft.shape)}  out={tuple(out.shape)}  "
              f"params={total:,}  trainable={trainable:,}")

    # Raw EMG architectures (T=52 samples = 260ms window)
    dummy_raw = torch.randn(8, 8, 52).to(device)
    for name in ("tcn_aot", "tcn_att"):
        m = build_model(name).to(device)
        out = m(dummy_raw)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name:15s}]  in={tuple(dummy_raw.shape)}  out={tuple(out.shape)}  "
              f"params={total:,}  trainable={trainable:,}")