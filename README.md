# EMG Gesture Classification — NinaPro DB5

Replication and comparison of two CNN architectures for sEMG-based hand gesture recognition,
following Côté-Allard et al. (2019) [1].

## Repo structure

```
repo/
├── data/ninaprodb5/     # place *.zip files here (downloaded from ninapro.net)
├── models/              # saved checkpoints  (auto-created)
├── plots/               # output figures     (auto-created)
├── src/
│   ├── preprocess.py    # unzip → load .mat → notch filter → train/test split
│   ├── feat_extract.py  # sliding window → STFT spectrogram → (N, 4, 8, 14)
│   ├── architectures.py # SlowFusionEMG [A] and MobileNetV2EMG [B]
│   ├── train.py         # shared training loop with early stopping
│   ├── eval.py          # accuracy, F1, confusion matrix, comparison plots
│   └── real_time.py     # latency benchmark, FLOP count, RT feasibility report
├── run_all.py           # configurable orchestrator
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place NinaPro DB5 zip files in `data/ninaprodb5/`.

## Usage

```bash
# Full pipeline (both architectures, all subjects)
python run_all.py

# Stop after feature extraction
python run_all.py --until features

# Start from evaluation (checkpoints already saved)
python run_all.py --from eval

# Single architecture only
python run_all.py --arch slow_fusion

# Override training config
python run_all.py --epochs 20 --batch-size 64

# Disable notch filter
python run_all.py --no-notch

# List all stages
python run_all.py --list-stages
```

## Architectures

| | [A] SlowFusionEMG | [B] MobileNetV2EMG |
|---|---|---|
| Based on | Côté-Allard et al. [1] | ImageNet transfer learning baseline |
| Input | (N, 4, 8, 14) via Conv3d | (N, 4, 8, 14) → upsampled 64×64 |
| Pretrained | No — trained from scratch | Yes — ImageNet (early layers frozen) |
| Time fusion | Gradual (3 Conv3d blocks) | Implicit (2D convolutions) |

## References

[1] Côté-Allard et al. (2019). Deep Learning for EMG Hand Gesture Classification
    Using Transfer Learning. IEEE TNSRE 27(4).
[2] Atzori et al. (2014). Electromyography data for non-invasive       naturally-controlledrobotic hand prostheses. Scientific Data.