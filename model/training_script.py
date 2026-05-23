"""
train.py - Two-Phase DINOv2 Fine-Tuning for PSF Regression
===========================================================
Trains PSFRegressor (DINOv2 ViT-B/14 + MLP Head) to predict three
Perceived Spatial Functionality (PSF) scores from architectural renderings:
  - Creativity  (novel / innovative geometry)
  - Belonging   (harmony and contextual fit)
  - EaseOfUse   (intuitive, human-centric design)

Training Strategy
-----------------
Phase 1 – Head Warm-up (backbone FROZEN)
    Only the regression MLP is trained. Higher learning rate (lr_head).
    Stabilises the head before touching the pretrained backbone.

Phase 2 – Deep Fine-tuning (last N blocks UNFROZEN)
    The last `unfreeze_blocks` transformer blocks + final LayerNorm are
    gradually tuned with a conservative learning rate (lr_finetune).

EMA (Exponential Moving Average)
    Shadow weights are maintained via timm's ModelEmaV2. Validation is
    performed on EMA weights, which typically generalise better.

Usage
-----
    # Use the best validated configuration:
    python train.py --config configs/training.yaml

    # Override the data root:
    python train.py --config configs/training.yaml --data_root /path/to/dataset

    # Set via environment variable (useful for Colab):
    PSF_DATA_ROOT=/path/to/dataset python train.py --config configs/training.yaml

Output
------
Checkpoints and a training_history.csv are saved to:
    checkpoints/<timestamp>/best.pt
TensorBoard logs are saved to:
    runs/<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model  import build_model
from dataset import StudioPSFDataset, get_transforms
from timm.utils import ModelEmaV2

try:
    import wandb; _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
    _HAS_TM = True
except ImportError:
    _HAS_TM = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = 2026, det: bool = True) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
#  Metric Tracker
# ──────────────────────────────────────────────────────────────────────────────


class MetricTracker:
    """Accumulates predictions and ground-truth labels per epoch."""

    def __init__(self, dev: torch.device):
        self.dev = dev
        self.reset()
        if _HAS_TM:
            self.mae_fn = MeanAbsoluteError().to(dev)
            self.mse_fn = MeanSquaredError(squared=True).to(dev)
            self.r2_fn  = R2Score(multioutput="uniform_average").to(dev)

    def reset(self):
        self.preds, self.targets, self.losses = [], [], []

    def update(self, p: torch.Tensor, t: torch.Tensor, l: float):
        self.preds.append(p.detach())
        self.targets.append(t.detach())
        self.losses.append(l)

    def compute(self) -> dict:
        if not self.preds:
            return {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}
        p = torch.cat(self.preds)
        t = torch.cat(self.targets)
        if _HAS_TM:
            mae  = self.mae_fn(p, t).item()
            mse  = self.mse_fn(p, t).item()
            rmse = mse ** 0.5
            r2   = self.r2_fn(p, t).item()
            self.mae_fn.reset(); self.mse_fn.reset(); self.r2_fn.reset()
        else:
            d    = p - t
            mae  = d.abs().mean().item()
            mse  = (d ** 2).mean().item()
            rmse = mse ** 0.5
            ss_r = ((t - p) ** 2).sum().item()
            ss_t = ((t - t.mean(0, keepdim=True)) ** 2).sum().item()
            r2   = 1 - ss_r / (ss_t + 1e-8)
        return {"loss": float(np.mean(self.losses)), "mae": mae, "rmse": rmse, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────────
#  Train / Validate
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, crit, opt, scaler, dev, amp, trk, ema=None):
    model.train(); trk.reset()
    pbar = tqdm(loader, desc="  Training", leave=False)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs); loss = crit(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            preds = model(imgs); loss = crit(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        trk.update(preds, labels, loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if ema is not None:
            ema.update(model)
    return trk.compute()


@torch.no_grad()
def validate(model, loader, crit, dev, amp, trk):
    model.eval(); trk.reset()
    pbar = tqdm(loader, desc="  Validating", leave=False)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        if amp:
            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs); loss = crit(preds, labels)
        else:
            preds = model(imgs); loss = crit(preds, labels)
        trk.update(preds, labels, loss.item())
    return trk.compute()


# ──────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg: dict):
    tc, dc, cc, lc = (
        cfg["train"], cfg["data"],
        cfg.get("checkpoint", {}), cfg.get("logging", {})
    )
    set_seed(tc.get("seed", 2026), tc.get("deterministic", True))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}")
    if dev.type == "cpu":
        print("\n" + "!" * 60)
        print("WARNING: CUDA (GPU) not available. Training will be very slow.")
        print("Install the CUDA build: pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")
        print("!" * 60 + "\n")

    root     = Path(dc["root"])
    img_size = dc.get("img_size", 518)

    # Build full dataset to determine split sizes
    full_ds = StudioPSFDataset(
        root_dir=root,
        image_dir=dc.get("image_dir", "images"),
        labels_csv=dc.get("labels_csv", "labels.csv"),
        image_col=dc.get("image_col", "Name"),
        label_cols=dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"]),
        image_ext=dc.get("image_ext", ".png"),
        transform=get_transforms(img_size, False),
    )
    n       = len(full_ds)
    n_test  = int(n * dc.get("test_split", 0.0))
    n_val   = int(n * dc.get("val_split", 0.20))
    n_train = n - n_val - n_test

    idx = list(range(n))
    rng = np.random.RandomState(tc.get("seed", 2026))
    rng.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]

    aug_cfg = cfg.get("augmentation", {})

    def mk(idxs, tr):
        return StudioPSFDataset(
            root_dir=root,
            image_dir=dc.get("image_dir", "images"),
            labels_csv=dc.get("labels_csv", "labels.csv"),
            image_col=dc.get("image_col", "Name"),
            label_cols=dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"]),
            image_ext=dc.get("image_ext", ".png"),
            transform=get_transforms(img_size, is_train=tr, aug_cfg=aug_cfg if tr else None),
            indices=idxs,
        )

    train_ds, val_ds = mk(train_idx, True), mk(val_idx, False)
    bs  = tc.get("batch_size", 4)
    nw  = tc.get("num_workers", 4)
    pin = (dev.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    print(f"[data] Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{n_test}")

    model = build_model(cfg).to(dev)
    crit  = nn.MSELoss()

    # Logging setup
    use_wandb = lc.get("wandb", False) or (_HAS_WANDB and os.environ.get("WANDB_API_KEY"))
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb  = SummaryWriter(str(root / f"{lc.get('log_dir', 'runs')}_{ts}")) if lc.get("tensorboard", True) else None
    if use_wandb and _HAS_WANDB:
        wandb.init(project=lc.get("wandb_project", "psf-dinov2"), config=cfg)

    ckpt_dir = root / f"{cc.get('dir', 'checkpoints')}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    freeze_ep = tc.get("freeze_epochs",  10)
    ft_ep     = tc.get("finetune_epochs", 18)
    total_ep  = freeze_ep + ft_ep
    patience  = tc.get("patience", 5)
    use_amp   = tc.get("amp", True) and torch.cuda.is_available()
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    trkT, trkV = MetricTracker(dev), MetricTracker(dev)

    # EMA
    use_ema   = tc.get("use_ema", True)
    model_ema = ModelEmaV2(model, decay=tc.get("ema_decay", 0.999)) if use_ema else None

    best_mae, pat_cnt, hist = float("inf"), 0, []
    optimizer = sched = None
    phase = 1

    for epoch in range(1, total_ep + 1):
        t0 = time.time()

        if epoch == 1:
            phase = 1
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=tc.get("lr_head", 2e-4), weight_decay=tc.get("weight_decay", 0.01)
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_ep, eta_min=tc.get("min_lr", 1e-6))

        elif epoch == freeze_ep + 1:
            phase = 2
            model.unfreeze_last_blocks(tc.get("unfreeze_blocks", 4))
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=tc.get("lr_finetune", 5e-5), weight_decay=tc.get("weight_decay", 0.01)
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_ep, eta_min=tc.get("min_lr", 1e-6))

        tm = train_one_epoch(model, train_loader, crit, optimizer, scaler, dev, use_amp, trkT, ema=model_ema)

        eval_model = model_ema.module if model_ema is not None else model
        vm = validate(eval_model, val_loader, crit, dev, use_amp, trkV)
        sched.step()
        lr_now = optimizer.param_groups[0]["lr"]
        el = time.time() - t0

        print(
            f"Ep {epoch:3d}/{total_ep} [P{phase}] lr={lr_now:.2e} "
            f"t_loss={tm['loss']:.4f} v_loss={vm['loss']:.4f} "
            f"v_MAE={vm['mae']:.4f} v_RMSE={vm['rmse']:.4f} v_R2={vm['r2']:.4f} ({el:.1f}s)"
        )

        row = {
            "epoch": epoch, "phase": phase, "lr": lr_now,
            **{f"train_{k}": v for k, v in tm.items()},
            **{f"val_{k}":   v for k, v in vm.items()},
        }
        hist.append(row)

        if tb:
            for k, v in tm.items(): tb.add_scalar(f"train/{k}", v, epoch)
            for k, v in vm.items(): tb.add_scalar(f"val/{k}",   v, epoch)
        if use_wandb and _HAS_WANDB:
            wandb.log(row)

        if vm["mae"] < best_mae:
            best_mae = vm["mae"]; pat_cnt = 0
            state = {
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "val_mae": best_mae,
                "config":  cfg,
            }
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  [SAVED] Best {'EMA' if use_ema else 'Base'} model (val_MAE={best_mae:.4f})")
        else:
            pat_cnt += 1

        pd.DataFrame(hist).to_csv(ckpt_dir / "training_history.csv", index=False)

        if pat_cnt >= patience:
            print(f"[early stop] No improvement for {patience} epochs.")
            break

    print(f"\n[done] Best val_MAE={best_mae:.4f}  →  {ckpt_dir / 'best.pt'}")
    if tb:
        tb.close()
    if use_wandb and _HAS_WANDB:
        wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train PSFRegressor (DINOv2 fine-tuning)")
    pa.add_argument("--config",    default="configs/training.yaml",
                    help="Path to YAML configuration file")
    pa.add_argument("--data_root", default=None,
                    help="Override data.root in the config (e.g. path to your dataset folder)")
    a   = pa.parse_args()
    cfg = load_config(a.config)

    if a.data_root:
        cfg["data"]["root"] = a.data_root
    env = os.getenv("PSF_DATA_ROOT")
    if env and cfg["data"]["root"] == ".":
        cfg["data"]["root"] = env

    main(cfg)
