"""
eval.py
=======
Evaluation utilities: test-set metrics + visualisation.

Functions
---------
  evaluate_model()     — accuracy, macro F1, confusion matrix for one model/subject
  evaluate_all()       — runs evaluate_model over all subjects, returns summary dict
  plot_comparison()    — bar charts: accuracy, F1 per subject for both architectures
  plot_confusion()     — side-by-side confusion matrices for a given subject
  print_summary()      — formatted table to stdout
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

log = logging.getLogger(__name__)

N_CLASSES = 17
PAPER_ACC = 0.6898   # Côté-Allard [1] reference for spectrogram model


# ─────────────────────────────────────────────
# Single model / subject
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   data: dict,
                   device: torch.device,
                   batch_size: int = 128) -> dict:
    """
    Returns
    -------
    dict with keys: acc, f1, cm  (confusion matrix ndarray)
    """
    X_test = torch.tensor(data["X_test"]).to(device)
    y_true = data["y_test"]

    model.eval()
    preds = []
    for i in range(0, len(X_test), batch_size):
        out = model(X_test[i:i + batch_size])
        preds.extend(out.argmax(1).cpu().numpy())

    y_pred = np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    return dict(acc=acc, f1=f1, cm=cm)


# ─────────────────────────────────────────────
# All subjects × all architectures
# ─────────────────────────────────────────────

def evaluate_all(trained_models: dict,
                 stft_data: dict,
                 device: torch.device,
                 batch_size: int = 128) -> dict:
    """
    Parameters
    ----------
    trained_models : {arch_name -> {sid -> nn.Module}}
    stft_data      : {sid -> {"X_test", "y_test", ...}}

    Returns
    -------
    results : {arch_name -> {sid -> {acc, f1, cm}}}
    """
    results = {}
    for arch, models_by_sid in trained_models.items():
        results[arch] = {}
        for sid in tqdm(models_by_sid, desc=f"Evaluating [{arch}]", unit="subject"):
            results[arch][sid] = evaluate_model(
                models_by_sid[sid], stft_data[sid], device, batch_size
            )
            log.info("[%s][%s] acc=%.3f  f1=%.3f",
                     arch, sid,
                     results[arch][sid]["acc"],
                     results[arch][sid]["f1"])
    return results


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

COLORS = {
    "slow_fusion":        "steelblue",
    "slow_fusion_se":     "royalblue",
    "slow_fusion_se_arc": "darkblue",
    "mobilenet":          "coral",
}
LABELS = {
    "slow_fusion":        "[A] SlowFusion",
    "slow_fusion_se":     "[A+] SlowFusion+SE",
    "slow_fusion_se_arc": "[A++] SF+SE+ArcFace",
    "mobilenet":          "[B] MobileNetV2",
}


def plot_comparison(results: dict, save_path: str | None = None) -> None:
    """Bar charts: accuracy and macro F1 per subject for each architecture."""
    archs = list(results.keys())
    sids  = list(next(iter(results.values())).keys())
    x     = np.arange(len(sids))
    w     = 0.8 / len(archs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Architecture Comparison", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, ["acc", "f1"],
                                  ["Test Accuracy", "Macro F1"]):
        for i, arch in enumerate(archs):
            vals = [results[arch][s][metric] for s in sids]
            offset = (i - len(archs) / 2 + 0.5) * w
            ax.bar(x + offset, vals, w,
                   label=LABELS.get(arch, arch),
                   color=COLORS.get(arch, f"C{i}"))

        if metric == "acc":
            ax.axhline(PAPER_ACC, color="green", linestyle="--",
                       linewidth=1.2, label="[1] Paper ref (~68.98%)")

        ax.set_xticks(x)
        ax.set_xticklabels(sids)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved comparison plot → %s", save_path)
    plt.show()


def plot_confusion(results: dict, sid: str,
                   save_path: str | None = None) -> None:
    """Side-by-side confusion matrices for one subject."""
    archs = list(results.keys())
    fig, axes = plt.subplots(1, len(archs),
                             figsize=(7 * len(archs), 6))
    if len(archs) == 1:
        axes = [axes]

    fig.suptitle(f"Confusion Matrices — {sid}", fontsize=13, fontweight="bold")
    cmaps = ["Blues", "Oranges", "Greens"]

    for ax, arch, cmap in zip(axes, archs, cmaps):
        cm = results[arch][sid]["cm"]
        sns.heatmap(cm, ax=ax, annot=False, fmt="d", cmap=cmap, cbar=True)
        acc = results[arch][sid]["acc"]
        ax.set_title(f"{LABELS.get(arch, arch)}  (acc={acc:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved confusion matrix → %s", save_path)
    plt.show()


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(results: dict) -> None:
    archs = list(results.keys())
    sids  = list(next(iter(results.values())).keys())

    header_parts = ["Subject".ljust(10)]
    for arch in archs:
        lbl = LABELS.get(arch, arch)
        header_parts += [f"{lbl} Acc".rjust(14), f"{lbl} F1".rjust(10)]
    print("=" * (10 + 24 * len(archs)))
    print("".join(header_parts))
    print("=" * (10 + 24 * len(archs)))

    for sid in sids:
        row = [sid.ljust(10)]
        for arch in archs:
            r = results[arch][sid]
            row += [f"{r['acc']:.3f}".rjust(14), f"{r['f1']:.3f}".rjust(10)]
        print("".join(row))

    print("=" * (10 + 24 * len(archs)))
    row = ["Mean".ljust(10)]
    for arch in archs:
        mean_acc = np.mean([results[arch][s]["acc"] for s in sids])
        mean_f1  = np.mean([results[arch][s]["f1"]  for s in sids])
        row += [f"{mean_acc:.3f}".rjust(14), f"{mean_f1:.3f}".rjust(10)]
    print("".join(row))
    print("=" * (10 + 24 * len(archs)))
    print(f"\n[1] Reference: ~{PAPER_ACC:.2%} (Côté-Allard 2019, spectrogram)")