"""
preprocess.py
=============
Handles everything from raw .mat files to clean per-subject splits.

Steps:
  1. Unzip NinaPro DB5 archives
  2. Load Exercise B (.mat) files for each subject
  3. Extract EMG (lower Myo, 8 ch), restimulus, rerepetition
  4. Apply optional 50 Hz notch filter
  5. Split by repetition: train={1,3,4,6}, test={2,5}

Usage:
  from src.preprocess import load_all_subjects
  subjects = load_all_subjects(cfg)          # returns dict of per-subject splits
"""

import os
import glob
import zipfile
import logging
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
from tqdm import tqdm

log = logging.getLogger(__name__)

TRAIN_REPS = {1, 3, 4, 6}
TEST_REPS  = {2, 5}


# ─────────────────────────────────────────────
# 1. Unzip
# ─────────────────────────────────────────────

def unzip_archives(data_dir: str) -> None:
    """Unzip all *.zip files found under data_dir (in-place)."""
    zips = sorted(glob.glob(os.path.join(data_dir, "*.zip")))
    if not zips:
        log.info("No zip files found in %s — skipping unzip.", data_dir)
        return

    for zf in tqdm(zips, desc="Unzipping", unit="file"):
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(data_dir)
            log.debug("✓ %s", os.path.basename(zf))
        except zipfile.BadZipFile:
            log.warning("✗ Bad zip: %s", zf)


# ─────────────────────────────────────────────
# 2. Notch filter
# ─────────────────────────────────────────────

def apply_notch(emg: np.ndarray, fs: int = 200, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Zero-phase 50 Hz notch filter applied along axis=0."""
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, emg, axis=0)


# ─────────────────────────────────────────────
# 3. Load one .mat file
# ─────────────────────────────────────────────

def _load_mat(path: str, notch: bool) -> dict:
    mat         = sio.loadmat(path)
    emg         = mat["emg"][:, :8].astype(np.float32)   # lower Myo only
    stimulus    = mat["restimulus"].flatten().astype(np.int32)   # corrected labels
    rerepetition = mat["rerepetition"].flatten().astype(np.int32)

    if notch:
        emg = apply_notch(emg).astype(np.float32)

    return {"emg": emg, "stimulus": stimulus, "rerepetition": rerepetition}


# ─────────────────────────────────────────────
# 4. Train / test split by repetition
# ─────────────────────────────────────────────

def _split(data: dict) -> dict:
    emg, stim, rep = data["emg"], data["stimulus"], data["rerepetition"]

    tr = np.isin(rep, list(TRAIN_REPS))
    te = np.isin(rep, list(TEST_REPS))

    return {
        "train": {"emg": emg[tr], "stimulus": stim[tr]},
        "test":  {"emg": emg[te], "stimulus": stim[te]},
    }


# ─────────────────────────────────────────────
# 5. Public API
# ─────────────────────────────────────────────

def load_all_subjects(cfg: dict) -> dict:
    """
    Parameters
    ----------
    cfg : dict  (from config in run_all.py)
        data_dir   : str   path to the folder with .zip / .mat files
        notch      : bool  apply 50 Hz notch filter
        unzip      : bool  unzip archives first

    Returns
    -------
    splits : dict  {subject_id -> {"train": {...}, "test": {...}}}
    """
    data_dir = cfg["data_dir"]

    if cfg.get("unzip", True):
        unzip_archives(data_dir)

    pattern = os.path.join(data_dir, "**", "*E2*.mat")
    files   = sorted(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(f"No E2 .mat files found under {data_dir!r}. "
                                "Check that the zips were extracted correctly.")

    splits = {}
    for f in tqdm(files, desc="Loading subjects", unit="subject"):
        sid = os.path.basename(os.path.dirname(f))   # e.g. "s1"
        try:
            data     = _load_mat(f, notch=cfg.get("notch", True))
            splits[sid] = _split(data)
            tr_shape = splits[sid]["train"]["emg"].shape
            te_shape = splits[sid]["test"]["emg"].shape
            log.info("✓ %s | train EMG %s | test EMG %s", sid, tr_shape, te_shape)
        except Exception as e:
            log.error("✗ %s: %s", f, e)

    log.info("Loaded %d subjects.", len(splits))
    return splits