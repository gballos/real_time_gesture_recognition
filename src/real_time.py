"""
real_time.py
============
Real-time feasibility analysis for each architecture.

Metrics
-------
  1. Parameter count              (total + trainable)
  2. Model size on disk           (MB)
  3. FLOPs for one inference      (GFLOPs) via thop if available
  4. CPU latency per window       (ms, mean ± std over N runs)
  5. Real-time budget check       (acquisition 260 ms + inference <= 300 ms)
                                   -> inference must be <= 40 ms on CPU

The 300 ms perceptual limit is from Farrell & Weir (2007) as cited in [1].

Usage:
  from src.real_time import benchmark_all, print_rt_report, plot_rt_comparison
  rt = benchmark_all(trained_models, device)
  print_rt_report(rt)
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

log = logging.getLogger(__name__)

RT_BUDGET_MS = 40.0       # ms available for inference  (300 - 260)
N_WARMUP     = 50
N_RUNS       = 500
DUMMY_INPUT  = (1, 4, 8, 14)   # single window
LABELS = {"slow_fusion": "[A] SlowFusion", "mobilenet": "[B] MobileNetV2"}


# ─────────────────────────────────────────────
# Parameter count + disk size
# ─────────────────────────────────────────────

def model_footprint(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = buf.tell() / 1e6

    return dict(total_params=total, trainable_params=trainable, size_mb=size_mb)


# ─────────────────────────────────────────────
# FLOPs (uses thop if available, else skips)
# ─────────────────────────────────────────────

def count_flops(model: nn.Module, device: torch.device):
    """
    Returns GFLOPs for one forward pass measured on CPU.

    Always uses CPU regardless of `device` so the FLOP count is on the same
    hardware as benchmark_latency — keeping required_gflops_per_sec meaningful.
    Returns None if thop is not installed.
    """
    try:
        from thop import profile
        cpu_model = model.cpu()
        dummy = torch.randn(*DUMMY_INPUT)   # CPU tensor
        with torch.no_grad():
            flops, _ = profile(cpu_model, inputs=(dummy,), verbose=False)
        model.to(device)   # restore original device
        return flops / 1e9
    except ImportError:
        log.warning("thop not installed — skipping FLOP count. "
                    "Install with: pip install thop")
        return None


def required_gflops_per_sec(flops_g: float | None,
                             budget_ms: float = RT_BUDGET_MS) -> float | None:
    """
    Minimum sustained GFLOP/s a processor must deliver to meet the RT budget.

      required = GFLOPs_per_inference / budget_in_seconds

    This is a theoretical lower bound assuming 100% utilisation with no
    overhead — real hardware needs headroom above this figure.

    Parameters
    ----------
    flops_g   : GFLOPs per inference (from count_flops)
    budget_ms : inference time budget in ms (default RT_BUDGET_MS = 40 ms)

    Returns
    -------
    float  GFLOP/s required, or None if flops_g is None
    """
    if flops_g is None:
        return None
    return flops_g / (budget_ms / 1000.0)


# ─────────────────────────────────────────────
# CPU latency
# ─────────────────────────────────────────────

def benchmark_latency(model: nn.Module,
                      n_warmup: int = N_WARMUP,
                      n_runs: int = N_RUNS) -> dict:
    """
    Measures single-window CPU inference time.
    Returns dict: mean_ms, std_ms, min_ms, max_ms, p95_ms, rt_ok
    """
    cpu_model = model.cpu().eval()
    dummy     = torch.randn(*DUMMY_INPUT)

    with torch.no_grad():
        for _ in range(n_warmup):
            cpu_model(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            cpu_model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return dict(
        mean_ms = float(np.mean(times)),
        std_ms  = float(np.std(times)),
        min_ms  = float(np.min(times)),
        max_ms  = float(np.max(times)),
        p95_ms  = float(np.percentile(times, 95)),
        rt_ok   = float(np.percentile(times, 95)) <= RT_BUDGET_MS,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def benchmark_all(trained_models: dict, device: torch.device) -> dict:
    """
    Parameters
    ----------
    trained_models : {arch_name -> {sid -> nn.Module}}
                     Uses first subject's model — arch is identical across subjects.

    Returns
    -------
    rt : {arch_name -> {footprint, flops, latency}}
    """
    rt = {}
    for arch, models_by_sid in tqdm(trained_models.items(),
                                    desc="RT benchmark", unit="arch"):
        model = next(iter(models_by_sid.values()))
        flops_g = count_flops(model, device)
        rt[arch] = dict(
            footprint      = model_footprint(model),
            flops          = flops_g,
            required_gflops= required_gflops_per_sec(flops_g),
            latency        = benchmark_latency(model),
        )
        log.info("[%s] params=%s  size=%.1f MB  p95=%.1f ms  rt_ok=%s",
                 arch,
                 f"{rt[arch]['footprint']['total_params']:,}",
                 rt[arch]['footprint']['size_mb'],
                 rt[arch]['latency']['p95_ms'],
                 rt[arch]['latency']['rt_ok'])

        model.to(device)   # move back after CPU benchmark

    return rt


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def print_rt_report(rt: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Real-Time Feasibility Report':^70}")
    print(f"{'Budget: acquisition 260ms + inference <= 300ms  =>  <=40ms':^70}")
    print("=" * 70)

    for arch, data in rt.items():
        lbl   = LABELS.get(arch, arch)
        fp    = data["footprint"]
        lat   = data["latency"]
        flops = data["flops"]
        req   = data["required_gflops"]

        print(f"\n  {lbl}")
        print(f"    Parameters : {fp['total_params']:>12,}  "
              f"(trainable: {fp['trainable_params']:,})")
        print(f"    Model size : {fp['size_mb']:>9.1f} MB")
        if flops is not None:
            print(f"    GFLOPs     : {flops:>9.4f}  (per inference)")
            print(f"    Min GFLOP/s: {req:>9.4f}  (to meet {RT_BUDGET_MS:.0f} ms budget, "
                  f"theoretical lower bound)")
        print(f"    Latency    :  {lat['mean_ms']:.2f} ± {lat['std_ms']:.2f} ms  "
              f"(p95={lat['p95_ms']:.2f} ms)")
        status = "PASS" if lat["rt_ok"] else "FAIL"
        sign   = "<=" if lat["rt_ok"] else ">"
        print(f"    RT check   :  {status}  (p95 {sign} {RT_BUDGET_MS:.0f} ms)")

    print("=" * 70)


def plot_rt_comparison(rt: dict, save_path: str = None) -> None:
    """Three-panel: latency bars, trainable parameter count, min GFLOP/s required."""
    archs  = list(rt.keys())
    colors = ["steelblue", "coral", "seagreen"]
    x      = np.arange(len(archs))

    has_flops = any(rt[a]["required_gflops"] is not None for a in archs)
    ncols = 3 if has_flops else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.suptitle("Real-Time Feasibility", fontsize=13, fontweight="bold")

    # Panel 1 — latency
    ax    = axes[0]
    means = [rt[a]["latency"]["mean_ms"] for a in archs]
    p95s  = [rt[a]["latency"]["p95_ms"]  for a in archs]
    ax.bar(x, means, color=[colors[i % len(colors)] for i in range(len(archs))],
           alpha=0.8, label="Mean")
    ax.scatter(x, p95s, marker="D", color="black", zorder=5, label="p95")
    ax.axhline(RT_BUDGET_MS, color="red", linestyle="--",
               linewidth=1.5, label=f"Budget ({RT_BUDGET_MS:.0f} ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(a, a) for a in archs])
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("CPU Latency per Window")
    ax.legend()

    # Panel 2 — trainable params
    ax = axes[1]
    trainable = [rt[a]["footprint"]["trainable_params"] / 1e6 for a in archs]
    ax.bar(x, trainable,
           color=[colors[i % len(colors)] for i in range(len(archs))], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(a, a) for a in archs])
    ax.set_ylabel("Trainable parameters (M)")
    ax.set_title("Model Complexity")

    # Panel 3 — minimum GFLOP/s required to meet RT budget
    if has_flops:
        ax = axes[2]
        req_vals = [rt[a]["required_gflops"] or 0.0 for a in archs]
        bars = ax.bar(x, req_vals,
                      color=[colors[i % len(colors)] for i in range(len(archs))],
                      alpha=0.8)
        # Annotate each bar with the actual value
        for bar, val in zip(bars, req_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(req_vals) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(a, a) for a in archs])
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"Min GFLOP/s to Meet {RT_BUDGET_MS:.0f} ms Budget\n"
                     "(theoretical lower bound)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved RT plot -> %s", save_path)
    plt.show()