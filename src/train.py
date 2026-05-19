"""
train.py
========
Training loop shared by both architectures.

Features:
  - tqdm progress bars (epoch + batch level)
  - ReduceLROnPlateau + early stopping matching [1]
  - Per-epoch history (train_loss, val_loss, val_acc)
  - Checkpoint saving to models/
  - Resumes from checkpoint if one exists (optional)
"""

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Defaults matching [1] ────────────────────────────────────
DEFAULTS = dict(
    batch_size        = 128,
    lr                = 0.00681,
    epochs            = 30,
    val_split         = 0.1,
    lr_factor         = 0.2,    # annealing factor 1/5 from [1]
    lr_patience       = 3,
    max_lr_reductions = 2,      # stop after 2 reductions [1]
    dropout           = 0.5,
)

# MobileNet needs more epochs and a more patient scheduler because
# three differential LR groups converge more slowly than a single LR.
MOBILENET_OVERRIDES = dict(
    epochs            = 60,
    lr_patience       = 5,
    max_lr_reductions = 3,
)


def _resolve_cfg(cfg: dict, arch_label: str) -> dict:
    """
    Return a cfg dict with architecture-specific overrides applied.
    Keys already set explicitly in cfg take precedence over overrides.
    """
    if arch_label == "mobilenet":
        merged = {**MOBILENET_OVERRIDES, **cfg}
        return merged
    return cfg


def _make_loaders(data: dict, cfg: dict, device: torch.device):
    bs  = cfg.get("batch_size", DEFAULTS["batch_size"])
    val = cfg.get("val_split",  DEFAULTS["val_split"])

    X = torch.tensor(data["X_train"]).to(device)
    y = torch.tensor(data["y_train"], dtype=torch.long).to(device)

    dataset    = TensorDataset(X, y)
    val_size   = max(1, int(len(dataset) * val))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    return train_loader, val_loader, train_size, val_size


def train_one_subject(
    model: nn.Module,
    data: dict,
    cfg: dict,
    device: torch.device,
    checkpoint_path: str | None = None,
    arch_label: str = "model",
    sid: str = "",
) -> tuple[nn.Module, list[dict]]:
    """
    Train model on one subject's data.

    Returns
    -------
    model   : trained nn.Module
    history : list of dicts  {epoch, train_loss, val_loss, val_acc, lr}
    """
    cfg = _resolve_cfg(cfg, arch_label)
    epochs        = cfg.get("epochs",            DEFAULTS["epochs"])
    lr            = cfg.get("lr",                DEFAULTS["lr"])
    lr_factor     = cfg.get("lr_factor",         DEFAULTS["lr_factor"])
    lr_patience   = cfg.get("lr_patience",       DEFAULTS["lr_patience"])
    max_reductions= cfg.get("max_lr_reductions", DEFAULTS["max_lr_reductions"])

    train_loader, val_loader, train_size, val_size = _make_loaders(data, cfg, device)

    # Use differential LR param groups when the model supports it (MobileNet),
    # otherwise fall back to a flat single-LR over all trainable params.
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr)
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    optimizer  = optim.Adam(param_groups, lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )
    criterion  = nn.CrossEntropyLoss()

    lr_reductions = 0
    history       = []

    epoch_bar = tqdm(range(1, epochs + 1),
                     desc=f"  {sid} [{arch_label}]",
                     unit="epoch", leave=False)

    for epoch in epoch_bar:
        # ── Train ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                correct  += (out.argmax(1) == yb).sum().item()
        val_loss /= val_size
        val_acc   = correct / val_size

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_loss=val_loss, val_acc=val_acc, lr=current_lr))

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.3f}",
            lr=f"{current_lr:.2e}",
        )

        # ── Early stopping ───────────────────────────────────
        prev_lr = current_lr
        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] < prev_lr:
            lr_reductions += 1
            log.debug("[%s][%s] LR reduced → %d/%d",
                      sid, arch_label, lr_reductions, max_reductions)
            if lr_reductions >= max_reductions:
                log.info("[%s][%s] Early stopping at epoch %d.",
                         sid, arch_label, epoch)
                break

    # ── Save checkpoint ──────────────────────────────────────
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "history":     history,
                    "cfg":         cfg}, checkpoint_path)
        log.info("Checkpoint saved → %s", checkpoint_path)

    return model, history