"""
run_all.py
==========
Configurable end-to-end pipeline runner.

Stages (run in order, stop at --until <stage>):
  1  preprocess   — unzip, load .mat files, apply notch filter, train/test split
  2  features     — sliding window + STFT spectrogram extraction
  3  train        — train both architectures per subject, save checkpoints
  4  eval         — accuracy, F1, confusion matrices, comparison plots
  5  realtime     — latency benchmark, FLOP count, real-time report

Usage examples
--------------
  # Full pipeline:
  python run_all.py

  # Stop after feature extraction (useful for inspecting data before training):
  python run_all.py --until features

  # Only run evaluation (assumes checkpoints already exist):
  python run_all.py --from eval

  # Single architecture:
  python run_all.py --arch slow_fusion

  # Override config inline:
  python run_all.py --epochs 20 --batch-size 64 --no-notch

  # List all stages:
  python run_all.py --list-stages
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ── Project imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess    import load_all_subjects
from src.feat_extract  import extract_all
from src.architectures import build_model
from src.train         import train_one_subject
from src.eval          import evaluate_all, plot_comparison, plot_confusion, print_summary
from src.real_time     import benchmark_all, print_rt_report, plot_rt_comparison

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Stage order ──────────────────────────────────────────────
STAGES = ["preprocess", "features", "train", "eval", "realtime"]


# ════════════════════════════════════════════════════════════
# Config defaults — edit here or override via CLI flags
# ════════════════════════════════════════════════════════════

torch.manual_seed(42)
np.random.seed(42)

DEFAULT_CFG = dict(
    # Paths
    data_dir    = "data/ninaprodb5",
    models_dir  = "models",
    plots_dir   = "plots",
    cache_dir   = "cache",          # stage output persistence

    # Data
    unzip       = True,
    notch       = True,

    # Windowing / STFT
    window_size   = 52,
    step          = 5,
    stft_nperseg  = 28,
    stft_noverlap = 20,

    # Training
    architectures     = ["slow_fusion", "mobilenet"],
    epochs            = 30,
    batch_size        = 128,
    lr                = 0.00681,
    lr_factor         = 0.2,
    lr_patience       = 3,
    max_lr_reductions = 2,
    dropout           = 0.5,
    n_classes         = 17,
)


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _ckpt_path(models_dir: str, arch: str, sid: str) -> str:
    return os.path.join(models_dir, arch, f"{sid}.pt")


def _load_checkpoint(path: str, arch: str, cfg: dict,
                     device: torch.device):
    """Load model from checkpoint if it exists, else return None."""
    if not os.path.exists(path):
        return None, None
    ckpt  = torch.load(path, map_location=device)
    model = build_model(arch, n_classes=cfg["n_classes"],
                        dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Loaded checkpoint: %s", path)
    return model, ckpt.get("history", [])


def _reconstruct_trained_models(cfg: dict, device: torch.device) -> dict | None:
    """
    Rebuild trained_models dict from saved .pt checkpoint files.

    Called when --from skips the train stage so eval/realtime can still run
    without a cache/train.pkl.  Returns None if any expected checkpoint is
    missing (caller should abort with a clear error).
    """
    archs      = cfg["architectures"]
    models_dir = cfg["models_dir"]

    # Discover subjects from the checkpoint directory of the first arch
    first_arch_dir = Path(models_dir) / archs[0]
    if not first_arch_dir.exists():
        return None

    sids = sorted(p.stem for p in first_arch_dir.glob("*.pt"))
    if not sids:
        return None

    trained_models = {arch: {} for arch in archs}
    for arch in archs:
        for sid in sids:
            ckpt_path = _ckpt_path(models_dir, arch, sid)
            model, _ = _load_checkpoint(ckpt_path, arch, cfg, device)
            if model is None:
                log.error("Missing checkpoint: %s — run the train stage first.",
                          ckpt_path)
                return None
            trained_models[arch][sid] = model

    log.info("Reconstructed trained_models from .pt files: %d arch × %d subjects",
             len(archs), len(sids))
    return trained_models


def _stage_index(name: str) -> int:
    try:
        return STAGES.index(name)
    except ValueError:
        log.error("Unknown stage %r. Valid stages: %s", name, STAGES)
        sys.exit(1)


# ════════════════════════════════════════════════════════════
# Stage cache — persist / restore each stage's output
# ════════════════════════════════════════════════════════════

def _cache_path(cache_dir: str, stage: str) -> Path:
    return Path(cache_dir) / f"{stage}.pkl"


def _save_cache(cache_dir: str, stage: str, obj) -> None:
    """Pickle stage output to cache_dir/<stage>.pkl."""
    path = _cache_path(cache_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Cache saved  → %s", path)


def _load_cache(cache_dir: str, stage: str):
    """Load pickled stage output, or return None if it doesn't exist."""
    path = _cache_path(cache_dir, stage)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.info("Cache loaded ← %s", path)
    return obj


# ════════════════════════════════════════════════════════════
# Stage functions
# ════════════════════════════════════════════════════════════

def stage_preprocess(cfg: dict) -> dict:
    log.info("── Stage 1: PREPROCESS ──────────────────────────────")
    splits = load_all_subjects(cfg)
    _save_cache(cfg["cache_dir"], "preprocess", splits)
    return splits


def stage_features(splits: dict, cfg: dict) -> dict:
    log.info("── Stage 2: FEATURE EXTRACTION ──────────────────────")
    stft_data = extract_all(splits, cfg)
    _save_cache(cfg["cache_dir"], "features", stft_data)
    return stft_data


def stage_train(stft_data: dict, cfg: dict,
                device: torch.device) -> dict:
    """
    Returns
    -------
    trained_models : {arch -> {sid -> nn.Module}}
    """
    log.info("── Stage 3: TRAINING ────────────────────────────────")
    trained_models = {arch: {} for arch in cfg["architectures"]}
    sids  = list(stft_data.keys())
    archs = cfg["architectures"]

    outer_bar = tqdm(archs, desc="Architectures", unit="arch", position=0)
    for arch in outer_bar:
        outer_bar.set_description(f"Architecture: {arch}")

        inner_bar = tqdm(sids, desc="  Subjects", unit="subject",
                         position=1, leave=False)
        for sid in inner_bar:
            inner_bar.set_description(f"  {sid}")

            ckpt = _ckpt_path(cfg["models_dir"], arch, sid)
            model, history = _load_checkpoint(ckpt, arch, cfg, device)

            if model is None:
                model = build_model(arch, n_classes=cfg["n_classes"],
                                    dropout=cfg["dropout"]).to(device)
                model, history = train_one_subject(
                    model, stft_data[sid], cfg, device,
                    checkpoint_path=ckpt,
                    arch_label=arch,
                    sid=sid,
                )

            trained_models[arch][sid] = model

    return trained_models


def stage_eval(trained_models: dict, stft_data: dict,
               cfg: dict, device: torch.device) -> dict:
    log.info("── Stage 4: EVALUATION ──────────────────────────────")
    results = evaluate_all(trained_models, stft_data, device,
                           batch_size=cfg["batch_size"])
    print_summary(results)

    plots_dir = cfg.get("plots_dir", "plots")
    plot_comparison(results,
                    save_path=os.path.join(plots_dir, "accuracy_f1_comparison.png"))

    sids = list(stft_data.keys())
    plot_confusion(results, sid=sids[0],
                   save_path=os.path.join(plots_dir, f"confusion_{sids[0]}.png"))

    _save_cache(cfg["cache_dir"], "eval", results)
    return results


def stage_realtime(trained_models: dict, cfg: dict,
                   device: torch.device) -> dict:
    log.info("── Stage 5: REAL-TIME BENCHMARK ─────────────────────")
    rt = benchmark_all(trained_models, device)
    print_rt_report(rt)

    plots_dir = cfg.get("plots_dir", "plots")
    plot_rt_comparison(rt,
                       save_path=os.path.join(plots_dir, "realtime_comparison.png"))
    return rt


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EMG gesture classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--until",      default="realtime",
                   choices=STAGES, metavar="STAGE",
                   help="Run pipeline up to and including this stage "
                        f"(default: realtime). Choices: {STAGES}")
    p.add_argument("--from",       default="preprocess", dest="from_stage",
                   choices=STAGES, metavar="STAGE",
                   help="Start pipeline from this stage "
                        "(assumes previous stages already ran).")
    ALL_ARCHS = ["slow_fusion", "slow_fusion_se", 
                 "slow_fusion_se_arc", "mobilenet"]
    p.add_argument("--arch", nargs="+", default=None,
                   choices=ALL_ARCHS)  
    p.add_argument("--list-stages", action="store_true",
                   help="Print available stages and exit.")
    p.add_argument("--data-dir",   default=DEFAULT_CFG["data_dir"])
    p.add_argument("--models-dir", default=DEFAULT_CFG["models_dir"])
    p.add_argument("--plots-dir",  default=DEFAULT_CFG["plots_dir"])
    p.add_argument("--cache-dir",  default=DEFAULT_CFG["cache_dir"],
                   help="Directory for stage cache files (default: cache/)")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_CFG["epochs"])
    p.add_argument("--batch-size", type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--no-notch",   action="store_true",
                   help="Disable 50 Hz notch filter.")
    p.add_argument("--no-unzip",   action="store_true",
                   help="Skip unzipping archives.")
    return p.parse_args()


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.list_stages:
        print("Available stages (in order):")
        for i, s in enumerate(STAGES, 1):
            print(f"  {i}. {s}")
        sys.exit(0)

    # Build config from defaults + CLI overrides
    cfg = dict(DEFAULT_CFG)
    cfg.update(dict(
        data_dir      = args.data_dir,
        models_dir    = args.models_dir,
        plots_dir     = args.plots_dir,
        cache_dir     = args.cache_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        notch         = not args.no_notch,
        unzip         = not args.no_unzip,
        architectures = args.arch or DEFAULT_CFG["architectures"],
    ))

    os.makedirs(cfg["models_dir"], exist_ok=True)
    os.makedirs(cfg["plots_dir"],  exist_ok=True)
    os.makedirs(cfg["cache_dir"],  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Architectures: %s", cfg["architectures"])

    from_idx  = _stage_index(args.from_stage)
    until_idx = _stage_index(args.until)

    if from_idx > until_idx:
        log.error("--from stage (%s) must come before --until stage (%s).",
                  args.from_stage, args.until)
        sys.exit(1)

    def should_run(stage: str) -> bool:
        idx = _stage_index(stage)
        return from_idx <= idx <= until_idx

    # ── State shared across stages ───────────────────────────
    # Each variable is populated either by running the stage or by loading
    # the cache written when that stage last ran.
    splits         = None
    stft_data      = None
    trained_models = None
    results        = None

    if should_run("preprocess"):
        splits = stage_preprocess(cfg)
    else:
        splits = _load_cache(cfg["cache_dir"], "preprocess")

    if should_run("features"):
        if splits is None:
            log.error("Stage 'features' needs 'preprocess' output. "
                      "Run from 'preprocess' first, or ensure cache/preprocess.pkl exists.")
            sys.exit(1)
        stft_data = stage_features(splits, cfg)
    else:
        stft_data = _load_cache(cfg["cache_dir"], "features")

    if should_run("train"):
        if stft_data is None:
            log.error("Stage 'train' needs 'features' output. "
                      "Run from 'features' first, or ensure cache/features.pkl exists.")
            sys.exit(1)
        trained_models = stage_train(stft_data, cfg, device)
    else:
        trained_models = _reconstruct_trained_models(cfg, device)
        if trained_models is None:
            log.error("Could not reconstruct trained_models from .pt files. "
                      "Run the train stage first: python run_all.py --until train")
            sys.exit(1)

    if should_run("eval"):
        if trained_models is None or stft_data is None:
            log.error("Stage 'eval' needs 'train' and 'features' output. "
                      "Ensure cache/train.pkl and cache/features.pkl exist.")
            sys.exit(1)
        results = stage_eval(trained_models, stft_data, cfg, device)
    else:
        results = _load_cache(cfg["cache_dir"], "eval")

    if should_run("realtime"):
        if trained_models is None:
            log.error("Stage 'realtime' needs 'train' output. "
                      "Ensure cache/train.pkl exists.")
            sys.exit(1)
        stage_realtime(trained_models, cfg, device)

    log.info("Done.")


if __name__ == "__main__":
    main()