"""
feat_extract.py
===============
Sliding-window segmentation → STFT spectrogram features.

Pipeline per subject:
  raw EMG split
    → make_windows()          (N, 52, 8) + labels
    → window_to_spectrogram() (N, 4, 8, 14)
    → labels remapped 1..17 → 0..16

All parameters from Côté-Allard et al. [1] unless noted.

Usage:
  from src.feat_extract import extract_all
  stft_data = extract_all(splits, cfg)
"""

import logging
import numpy as np
from scipy.signal import stft
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Defaults (overridable via cfg) ───────────────────────────
WINDOW_SIZE  = 52    # samples — 260 ms @ 200 Hz  [1]
STEP         = 5     # samples — 25 ms step        [1]
STFT_NPERSEG = 28    # Hann window length           [1]
STFT_NOVERLAP = 20   # overlap → step=8             [1]
N_CLASSES    = 17    # gestures 1..17 → 0..16


# ─────────────────────────────────────────────
# Sliding window
# ─────────────────────────────────────────────

def make_windows(emg: np.ndarray, stimulus: np.ndarray,
                 window_size: int, step: int) -> tuple:
    """
    Segment continuous EMG into overlapping windows.

    Parameters
    ----------
    emg      : (N, 8)  float32
    stimulus : (N,)    int32   — restimulus labels (0=rest, 1..17=gesture)

    Returns
    -------
    X : (n_windows, window_size, 8)  float32
    y : (n_windows,)                 int32    — majority label, rest filtered, remapped 0..16
    """
    X_list, y_list = [], []
    n = len(emg)

    for start in range(0, n - window_size + 1, step):
        end    = start + window_size
        window = emg[start:end]                        # (52, 8)
        labels = stimulus[start:end]                   # (52,)

        vals, counts = np.unique(labels, return_counts=True)
        majority     = vals[np.argmax(counts)]

        X_list.append(window)
        y_list.append(majority)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Filter rest windows (label == 0)
    mask = y != 0
    X, y = X[mask], y[mask]

    # Remap gesture labels 1..17 → 0..16
    y = (y - 1).astype(np.int32)

    return X, y


# ─────────────────────────────────────────────
# STFT spectrogram
# ─────────────────────────────────────────────

def window_to_spectrogram(window: np.ndarray,
                           nperseg: int, noverlap: int) -> np.ndarray:
    """
    Convert one EMG window to a spectrogram.

    Parameters
    ----------
    window : (52, 8)  float32

    Returns
    -------
    spec : (4, 8, 14)  float32   — (Time, Channel, Freq)
    """
    specs = []
    for ch in range(window.shape[1]):
        _, _, Zxx = stft(window[:, ch], nperseg=nperseg,
                         noverlap=noverlap, window="hann")
        mag = np.abs(Zxx)       # (15, 4)
        mag = mag[1:, :4]       # drop DC bin → (14, 4)

        # Log-compress bc EMG spectrograms are very sparse and skewed.
        mag = np.log1p(mag)

        specs.append(mag)       # (14, 4)

    specs = np.stack(specs, axis=0)          # (8, 14, 4)
    specs = specs.transpose(2, 0, 1)         # (4, 8, 14) = (Time, Ch, Freq)
    return specs.astype(np.float32)


def _compute_spectrograms(X_windows: np.ndarray,
                           nperseg: int, noverlap: int,
                           desc: str = "") -> np.ndarray:
    """Vectorised wrapper with progress bar."""
    out = np.stack(
        [window_to_spectrogram(w, nperseg, noverlap)
         for w in tqdm(X_windows, desc=desc, leave=False, unit="win")],
        axis=0,
    )
    return out   # (N, 4, 8, 14)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def extract_all(splits: dict, cfg: dict) -> dict:
    """
    Parameters
    ----------
    splits : output of preprocess.load_all_subjects()
    cfg    : dict with optional keys:
               window_size, step, stft_nperseg, stft_noverlap

    Returns
    -------
    stft_data : {sid -> {"X_train", "y_train", "X_test", "y_test"}}
                 X shape: (N, 4, 8, 14)   y shape: (N,)  values 0..16
    """
    ws  = cfg.get("window_size",   WINDOW_SIZE)
    st  = cfg.get("step",          STEP)
    nps = cfg.get("stft_nperseg",  STFT_NPERSEG)
    nov = cfg.get("stft_noverlap", STFT_NOVERLAP)

    stft_data = {}

    for sid, sp in tqdm(splits.items(), desc="Feature extraction", unit="subject"):
        X_tr, y_tr = make_windows(sp["train"]["emg"], sp["train"]["stimulus"], ws, st)
        X_te, y_te = make_windows(sp["test"]["emg"],  sp["test"]["stimulus"],  ws, st)

        X_tr = _compute_spectrograms(X_tr, nps, nov, desc=f"{sid} train")
        X_te = _compute_spectrograms(X_te, nps, nov, desc=f"{sid} test")

        stft_data[sid] = {
            "X_train": X_tr, "y_train": y_tr,
            "X_test":  X_te, "y_test":  y_te,
        }

        log.info("%s | X_train %s  X_test %s | classes %s",
                 sid, X_tr.shape, X_te.shape, np.unique(y_tr).tolist())

    return stft_data