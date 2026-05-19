"""
real_time.py
============
Real-time feasibility analysis for each architecture.

Measures the FULL inference pipeline per window, not just the network:
  1. STFT spectrogram extraction  (scipy.signal.stft, 8 channels)
  2. log1p compression + reshape
  3. Tensor conversion
  4. Network forward pass

Metrics
-------
  1. Parameter count              (total + trainable)
  2. Model size on disk           (MB)
  3. FLOPs for one inference      (GFLOPs)
  4. Full pipeline CPU latency    (ms, mean +/- std over N runs)
     — broken down: preprocessing vs network
  5. Real-time budget check       (acquisition 260 ms + pipeline <= 300 ms)
                                   -> pipeline must be <= 40 ms on CPU

The 300 ms perceptual limit is from Farrell & Weir (2007) as cited in [1].
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import stft
from tqdm import tqdm

log = logging.getLogger(__name__)

RT_BUDGET_MS = 40.0       # ms available for full pipeline  (300 - 260)
N_WARMUP     = 50
N_RUNS       = 500
DUMMY_INPUT  = (1, 4, 8, 14)   # single spectrogram tensor
DUMMY_WINDOW = (52, 8)         # single raw EMG window

# STFT params matching feat_extract.py
STFT_NPERSEG  = 28
STFT_NOVERLAP = 20

LABELS = {
    "slow_fusion":       "[A] SlowFusion",
    "slow_fusion_se":    "[A+] SlowFusion+SE",
    "slow_fusion_se_arc":"[A++] SF+SE+ArcFace",
    "mobilenet":         "[B] MobileNetV2",
}


# ─────────────────────────────────────────────
# Preprocessing (mirrors feat_extract.window_to_spectrogram)
# ─────────────────────────────────────────────

def _preprocess_window(window: np.ndarray,
                       nperseg: int = STFT_NPERSEG,
                       noverlap: int = STFT_NOVERLAP) -> np.ndarray:
    """
    Convert one raw EMG window (52, 8) → spectrogram (1, 4, 8, 14) float32.
    Identical to feat_extract.window_to_spectrogram but kept local to avoid
    circular imports and to make the benchmark self-contained.
    """
    specs = []
    for ch in range(window.shape[1]):
        _, _, Zxx = stft(window[:, ch], nperseg=nperseg,
                         noverlap=noverlap, window="hann")
        mag = np.abs(Zxx)        # (15, 4)
        mag = mag[1:, :4]        # drop DC → (14, 4)
        mag = np.log1p(mag)
        specs.append(mag)        # (14, 4)
    specs = np.stack(specs, axis=0)          # (8, 14, 4)
    specs = specs.transpose(2, 0, 1)         # (4, 8, 14)
    return specs.astype(np.float32)[np.newaxis]  # (1, 4, 8, 14)


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
# FLOPs (network only — STFT is not a tensor op)
# ─────────────────────────────────────────────

def count_flops(model: nn.Module, device: torch.device) -> float | None:
    """
    Returns GFLOPs for one forward pass, always measured on CPU.

    Strategy (in priority order):
      1. torch.utils.flop_counter.FlopCounterMode  (PyTorch >= 2.1)
      2. thop.profile                              (pip install thop)
      3. Returns None with warning
    """
    cpu_model = model.cpu().eval()
    dummy = torch.randn(*DUMMY_INPUT)

    flops = None

    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(cpu_model, display=False) as fcm:
                cpu_model(dummy)
        flops = fcm.get_total_flops()
    except (ImportError, AttributeError):
        pass

    if flops is None:
        try:
            from thop import profile as thop_profile
            with torch.no_grad():
                flops, _ = thop_profile(cpu_model, inputs=(dummy,), verbose=False)
        except ImportError:
            pass

    model.to(device)

    if flops is None:
        log.warning(
            "FLOPs could not be counted. "
            "PyTorch >= 2.1 required for native counting, or: pip install thop"
        )
        return None

    return flops / 1e9


def required_gflops_per_sec(flops_g: float | None,
                             budget_ms: float = RT_BUDGET_MS) -> float | None:
    """
    Min sustained GFLOP/s to meet the RT budget (theoretical lower bound).
    Note: this covers only the network forward pass, not preprocessing.
    """
    if flops_g is None:
        return None
    return flops_g / (budget_ms / 1000.0)


# ─────────────────────────────────────────────
# CPU latency — FULL PIPELINE
# ─────────────────────────────────────────────

def benchmark_latency(model: nn.Module,
                      n_warmup: int = N_WARMUP,
                      n_runs: int = N_RUNS) -> dict:
    """
    Measures the full single-window pipeline on CPU:
      raw EMG (52,8) → STFT → log1p → tensor → forward pass

    Returns dict with:
      preprocess_mean_ms, preprocess_p95_ms   — STFT + log1p + tensor conversion
      network_mean_ms, network_p95_ms         — model forward pass only
      total_mean_ms, total_p95_ms             — end-to-end
      rt_ok                                   — total_p95_ms <= RT_BUDGET_MS
    """
    cpu_model = model.cpu().eval()
    dummy_emg = np.random.randn(*DUMMY_WINDOW).astype(np.float32)

    # ── Warmup (full pipeline) ───────────────────────────────
    with torch.no_grad():
        for _ in range(n_warmup):
            spec = _preprocess_window(dummy_emg)
            cpu_model(torch.from_numpy(spec))

    # ── Timed runs ───────────────────────────────────────────
    preprocess_times = []
    network_times    = []
    total_times      = []

    with torch.no_grad():
        for _ in range(n_runs):
            t_start = time.perf_counter()

            # Step 1: preprocessing (STFT + log1p + reshape)
            spec = _preprocess_window(dummy_emg)
            t_preprocess = time.perf_counter()

            # Step 2: tensor conversion + forward pass
            tensor = torch.from_numpy(spec)
            cpu_model(tensor)
            t_end = time.perf_counter()

            preprocess_times.append((t_preprocess - t_start) * 1000)
            network_times.append((t_end - t_preprocess) * 1000)
            total_times.append((t_end - t_start) * 1000)

    pre  = np.array(preprocess_times)
    net  = np.array(network_times)
    tot  = np.array(total_times)

    return dict(
        # Preprocessing breakdown
        preprocess_mean_ms = float(np.mean(pre)),
        preprocess_std_ms  = float(np.std(pre)),
        preprocess_p95_ms  = float(np.percentile(pre, 95)),
        # Network breakdown
        network_mean_ms    = float(np.mean(net)),
        network_std_ms     = float(np.std(net)),
        network_p95_ms     = float(np.percentile(net, 95)),
        # Total end-to-end
        total_mean_ms      = float(np.mean(tot)),
        total_std_ms       = float(np.std(tot)),
        total_min_ms       = float(np.min(tot)),
        total_max_ms       = float(np.max(tot)),
        total_p95_ms       = float(np.percentile(tot, 95)),
        rt_ok              = float(np.percentile(tot, 95)) <= RT_BUDGET_MS,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def benchmark_all(trained_models: dict, device: torch.device) -> dict:
    """
    Parameters
    ----------
    trained_models : {arch_name -> {sid -> nn.Module}}

    Returns
    -------
    rt : {arch_name -> {footprint, flops, required_gflops, latency}}
    """
    rt = {}
    for arch, models_by_sid in tqdm(trained_models.items(),
                                    desc="RT benchmark", unit="arch"):
        model = next(iter(models_by_sid.values()))
        flops_g = count_flops(model, device)
        rt[arch] = dict(
            footprint       = model_footprint(model),
            flops           = flops_g,
            required_gflops = required_gflops_per_sec(flops_g),
            latency         = benchmark_latency(model),
        )
        lat = rt[arch]["latency"]
        log.info("[%s] params=%s  size=%.1f MB  "
                 "preprocess_p95=%.1f ms  network_p95=%.1f ms  "
                 "total_p95=%.1f ms  rt_ok=%s",
                 arch,
                 f"{rt[arch]['footprint']['total_params']:,}",
                 rt[arch]['footprint']['size_mb'],
                 lat['preprocess_p95_ms'],
                 lat['network_p95_ms'],
                 lat['total_p95_ms'],
                 lat['rt_ok'])

        model.to(device)

    return rt


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def print_rt_report(rt: dict) -> None:
    print("\n" + "=" * 75)
    print(f"{'Real-Time Feasibility Report':^75}")
    print(f"{'Budget: acquisition 260ms + full pipeline <= 300ms  =>  <=40ms':^75}")
    print(f"{'Pipeline: STFT(8ch) + log1p + tensor conv + network forward':^75}")
    print("=" * 75)

    for arch, data in rt.items():
        lbl   = LABELS.get(arch, arch)
        fp    = data["footprint"]
        lat   = data["latency"]
        flops = data["flops"]
        req   = data["required_gflops"]

        print(f"\n  {lbl}")
        print(f"    Parameters   : {fp['total_params']:>12,}  "
              f"(trainable: {fp['trainable_params']:,})")
        print(f"    Model size   : {fp['size_mb']:>9.1f} MB")
        if flops is not None:
            print(f"    GFLOPs       : {flops:>9.4f}  (network only, per inference)")
            print(f"    Min GFLOP/s  : {req:>9.4f}  (to meet {RT_BUDGET_MS:.0f} ms budget, "
                  f"theoretical lower bound)")
        print(f"    Preprocessing:  {lat['preprocess_mean_ms']:.2f} "
              f"\u00b1 {lat['preprocess_std_ms']:.2f} ms  "
              f"(p95={lat['preprocess_p95_ms']:.2f} ms)")
        print(f"    Network      :  {lat['network_mean_ms']:.2f} "
              f"\u00b1 {lat['network_std_ms']:.2f} ms  "
              f"(p95={lat['network_p95_ms']:.2f} ms)")
        print(f"    Total        :  {lat['total_mean_ms']:.2f} "
              f"\u00b1 {lat['total_std_ms']:.2f} ms  "
              f"(p95={lat['total_p95_ms']:.2f} ms)")
        status = "PASS" if lat["rt_ok"] else "FAIL"
        sign   = "<=" if lat["rt_ok"] else ">"
        print(f"    RT check     :  {status}  (total p95 {sign} {RT_BUDGET_MS:.0f} ms)")

    print("=" * 75)


def plot_rt_comparison(rt: dict, save_path: str = None) -> None:
    """Four-panel: stacked latency, trainable params, min GFLOP/s, model size."""
    archs  = list(rt.keys())
    labels = [LABELS.get(a, a) for a in archs]
    colors_pre = "lightgray"
    colors_net = ["steelblue", "coral", "royalblue", "darkblue"]
    x = np.arange(len(archs))

    has_flops = any(rt[a]["required_gflops"] is not None for a in archs)
    ncols = 3 if has_flops else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.suptitle("Real-Time Feasibility (Full Pipeline)", fontsize=13, fontweight="bold")

    # ── Panel 1: stacked latency (preprocessing + network) ──
    ax = axes[0]
    pre_vals = [rt[a]["latency"]["preprocess_mean_ms"] for a in archs]
    net_vals = [rt[a]["latency"]["network_mean_ms"]    for a in archs]
    tot_p95  = [rt[a]["latency"]["total_p95_ms"]       for a in archs]

    ax.bar(x, pre_vals, color=colors_pre, alpha=0.9, label="Preprocessing (STFT)")
    ax.bar(x, net_vals, bottom=pre_vals,
           color=[colors_net[i % len(colors_net)] for i in range(len(archs))],
           alpha=0.8, label="Network")
    ax.scatter(x, tot_p95, marker="D", color="black", zorder=5, s=40, label="Total p95")
    ax.axhline(RT_BUDGET_MS, color="red", linestyle="--",
               linewidth=1.5, label=f"Budget ({RT_BUDGET_MS:.0f} ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Full Pipeline Latency")
    ax.legend(fontsize=7)

    # ── Panel 2: trainable params ────────────────────────────
    ax = axes[1]
    trainable = [rt[a]["footprint"]["trainable_params"] / 1e6 for a in archs]
    ax.bar(x, trainable,
           color=[colors_net[i % len(colors_net)] for i in range(len(archs))], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Trainable parameters (M)")
    ax.set_title("Model Complexity")

    # ── Panel 3: min GFLOP/s ─────────────────────────────────
    if has_flops:
        ax = axes[2]
        req_vals = [rt[a]["required_gflops"] or 0.0 for a in archs]
        bars = ax.bar(x, req_vals,
                      color=[colors_net[i % len(colors_net)] for i in range(len(archs))],
                      alpha=0.8)
        for bar, val in zip(bars, req_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(req_vals) * 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"Min GFLOP/s to Meet {RT_BUDGET_MS:.0f} ms Budget\n"
                     "(network only, theoretical lower bound)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved RT plot -> %s", save_path)
    plt.show()