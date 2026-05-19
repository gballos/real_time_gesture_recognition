"""
train.py
========
Training loop shared by all architectures.

Features:
  - ReduceLROnPlateau + early stopping for SlowFusion variants  [1]
  - CosineAnnealingLR for MobileNet  [7]
  - Label smoothing (ε=0.1) for MobileNet  [6][8]
  - ArcFace-aware forward pass (passes labels in training)
  - Per-epoch history, checkpoint saving

References
----------
[6]  Müller, Kornblith & Hinton (2019) "When Does Label Smoothing Help?" NeurIPS.
[7]  Loshchilov & Hutter (2017) "SGDR: SGD with Warm Restarts." ICLR.
[8]  Szegedy et al. (2016) "Rethinking the Inception Architecture." CVPR.
"""

import logging
import os

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
    lr_factor         = 0.2,
    lr_patience       = 3,
    max_lr_reductions = 2,
    dropout           = 0.5,
    label_smoothing   = 0.0,   # hard labels for [1]
    scheduler         = "plateau",
)

# MobileNet: cosine LR + label smoothing  [6][7][8]
MOBILENET_OVERRIDES = dict(
    epochs            = 60,
    lr_patience       = 5,
    max_lr_reductions = 3,
    label_smoothing   = 0.1,
    scheduler         = "cosine",
)

# SE variant: slightly more epochs for SE params to settle
SE_OVERRIDES = dict(
    epochs            = 40,
)

# ArcFace variant: more epochs + patient scheduler (angular margin is harder)
SE_ARC_OVERRIDES = dict(
    epochs            = 50,
    lr_patience       = 5,
    max_lr_reductions = 3,
)


def _resolve_cfg(cfg: dict, arch_label: str) -> dict:
    """Apply arch-specific overrides. Explicit cfg keys always win."""
    overrides = {
        "mobilenet":          MOBILENET_OVERRIDES,
        "slow_fusion_se":     SE_OVERRIDES,
        "slow_fusion_se_arc": SE_ARC_OVERRIDES,
    }.get(arch_label, {})
    return {**overrides, **cfg} if overrides else cfg


def _make_loaders(data: dict, cfg: dict, device: torch.device):
    bs  = cfg.get("batch_size", DEFAULTS["batch_size"])
    val = cfg.get("val_split",  DEFAULTS["val_split"])

    X = torch.tensor(data["X_train"]).to(device)
    y = torch.tensor(data["y_train"], dtype=torch.long).to(device)

    dataset    = TensorDataset(X, y)
    val_size   = max(1, int(len(dataset) * val))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds,   batch_size=bs, shuffle=False),
            train_size, val_size)


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
    cfg           = _resolve_cfg(cfg, arch_label)
    epochs        = cfg.get("epochs",            DEFAULTS["epochs"])
    lr            = cfg.get("lr",                DEFAULTS["lr"])
    lr_factor     = cfg.get("lr_factor",         DEFAULTS["lr_factor"])
    lr_patience   = cfg.get("lr_patience",       DEFAULTS["lr_patience"])
    max_reductions= cfg.get("max_lr_reductions", DEFAULTS["max_lr_reductions"])
    smoothing     = cfg.get("label_smoothing",   DEFAULTS["label_smoothing"])
    sched_type    = cfg.get("scheduler",         DEFAULTS["scheduler"])

    train_loader, val_loader, train_size, val_size = _make_loaders(data, cfg, device)

    arcface = getattr(model, "use_arcface", False)

    # ── Optimizer ────────────────────────────────────────────
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr)
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(param_groups, lr=lr)

    # ── Scheduler ────────────────────────────────────────────
    if sched_type == "cosine":
        # Smooth decay to lr*0.01 over all epochs  [7]
        scheduler   = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)
        use_plateau = False
    else:
        scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience)
        use_plateau = True

    # ── Loss ─────────────────────────────────────────────────
    # ArcFace already regularises the angular space, so label smoothing
    # is disabled for it — combining both would fight the margin objective.
    effective_smoothing = 0.0 if arcface else smoothing
    criterion     = nn.CrossEntropyLoss(label_smoothing=effective_smoothing)
    val_criterion = nn.CrossEntropyLoss()   # always hard labels for val metric

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
            logits = model(xb, labels=yb) if arcface else model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)          # ArcFace: no labels → no margin
                val_loss += val_criterion(out, yb).item() * len(xb)
                correct  += (out.argmax(1) == yb).sum().item()
        val_loss /= val_size
        val_acc   = correct / val_size

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_loss=val_loss, val_acc=val_acc, lr=current_lr))
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.3f}", lr=f"{current_lr:.2e}",
        )

        # ── LR scheduling + early stopping ───────────────────
        if use_plateau:
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
        else:
            scheduler.step()

    # ── Save checkpoint ──────────────────────────────────────
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "history": history, "cfg": cfg}, checkpoint_path)
        log.info("Checkpoint saved → %s", checkpoint_path)

    return model, history