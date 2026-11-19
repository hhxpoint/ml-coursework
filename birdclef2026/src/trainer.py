# ============================================================
# BirdCLEF 2026 — src/trainer.py
# Training Loop: Full pipeline with Mixup, Augmentation, Pseudo-labeling
# 
# Proven best formula (v28):
#   LR: 5e-4 → cosine, warmup=3ep
#   Batch: 32, Epochs: 35
#   Mixup: α=0.5, circ_shift: p=0.5
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, Tuple
from pathlib import Path

from .model import build_model, count_parameters
from .dataset import create_dataloaders
from .losses import build_criterion, AsymmetricLoss, DualLoss


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def batch_mixup(
    mel: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp at batch level.
    
    Args:
        mel: (batch, 3, n_mels, time)
        labels: (batch, num_classes)
        alpha: Beta distribution parameter
        
    Returns:
        mixed_mel, mixed_labels, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = mel.size(0)
    index = torch.randperm(batch_size, device=mel.device)

    mixed_mel = lam * mel + (1 - lam) * mel[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_mel, mixed_labels, lam


def compute_val_auc(
    model: nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int = 234,
) -> float:
    """
    Compute validation ROC-AUC on soundscape split.
    
    Returns:
        Macro-averaged ROC-AUC
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)

            outputs = model(mel)
            logits = outputs["clip_logits"]
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute per-class AUC, skip classes with no positives
    auc_scores = []
    for c in range(num_classes):
        if all_targets[:, c].sum() > 0:
            try:
                auc = roc_auc_score(all_targets[:, c], all_preds[:, c])
                auc_scores.append(auc)
            except:
                pass

    return np.mean(auc_scores) if auc_scores else 0.0


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train for one epoch with MixUp and dual loss.
    
    Returns:
        dict of training metrics
    """
    model.train()

    loss_meter = AverageMeter()
    clip_loss_meter = AverageMeter()
    frame_loss_meter = AverageMeter()

    use_amp = config.get("use_amp", True)
    mixup_alpha = config.get("mixup_alpha", 0.5)

    for step, batch in enumerate(train_loader):
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)

        # Apply batch-level MixUp
        if mixup_alpha > 0 and np.random.random() < 0.5:
            mel, labels, lam = batch_mixup(mel, labels, alpha=mixup_alpha)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(mel)
                if isinstance(criterion, DualLoss) and "frame_logits" in outputs:
                    result = criterion(
                        outputs["clip_logits"],
                        outputs["frame_logits"],
                        labels,
                    )
                    loss = result["loss"]
                else:
                    if isinstance(criterion, DualLoss):
                        logits = outputs["clip_logits"]
                    else:
                        logits = outputs.get("clip_logits", outputs.get("logits", list(outputs.values())[0]))
                    loss = criterion(logits, labels)
                    if isinstance(loss, dict):
                        loss = loss["loss"] if "loss" in loss else list(loss.values())[0]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mel)
            if isinstance(criterion, DualLoss) and "frame_logits" in outputs:
                result = criterion(
                    outputs["clip_logits"],
                    outputs["frame_logits"],
                    labels,
                )
                loss = result["loss"]
            else:
                if isinstance(criterion, DualLoss):
                    logits = outputs["clip_logits"]
                else:
                    logits = outputs.get("clip_logits", outputs.get("logits", list(outputs.values())[0]))
                loss = criterion(logits, labels)
                if isinstance(loss, dict):
                    loss = loss["loss"] if "loss" in loss else list(loss.values())[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = mel.size(0)
        loss_meter.update(loss.item(), batch_size)

        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch} Step {step}/{len(train_loader)} | "
                  f"Loss: {loss_meter.avg:.4f} | LR: {lr:.2e}")

    return {
        "train_loss": loss_meter.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


class BirdCLEFTrainer:
    """
    Full training pipeline for BirdCLEF 2026.
    
    Features:
    - ASL / Dual loss
    - Cosine warmup schedule
    - MixUp augmentation
    - Mixed precision training
    - Validation AUC monitoring
    - Best model checkpointing
    - Model soup support
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seed
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Config: {config['backbone']}, epochs={config['epochs']}, "
              f"batch={config['batch_size']}, lr={config['lr']}")

    def train(
        self,
        fold: Optional[int] = None,
        fold_csv: Optional[str] = None,
    ) -> str:
        """
        Run full training loop.
        
        Args:
            fold: Cross-validation fold number
            fold_csv: Path to fold CSV
            
        Returns:
            Path to best checkpoint
        """
        config = self.config

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config, fold=fold, fold_csv=fold_csv)

        # Build model
        model = build_model(config).to(self.device)
        print(f"[Trainer] Model parameters: {count_parameters(model):,}")

        # Build criterion
        criterion = build_criterion(config).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Scheduler: warmup + cosine
        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = len(train_loader) * config.get("warmup_epochs", 3)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision
        use_amp = config.get("use_amp", True)
        scaler = GradScaler() if use_amp else None

        # Training loop
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = config.get("patience", 7)
        history = []

        print(f"[Trainer] Starting training for {config['epochs']} epochs...")

        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.time()

            # Train one epoch
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer,
                self.device, epoch, config, scaler,
            )

            scheduler.step()

            # Validation
            val_auc = compute_val_auc(
                model, val_loader, self.device,
                num_classes=config["num_classes"],
            )

            epoch_time = time.time() - epoch_start

            # Logging
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_auc": val_auc,
                "lr": train_metrics["lr"],
                "time": epoch_time,
            }
            history.append(metrics)

            print(f"Epoch {epoch}/{config['epochs']} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {train_metrics['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0

                ckpt_path = self.output_dir / f"best_model"
                if fold is not None:
                    ckpt_path = Path(str(ckpt_path) + f"_fold{fold}")
                ckpt_path = Path(str(ckpt_path) + ".pt")

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config": config,
                }, ckpt_path)

                print(f"  ✓ New best AUC: {val_auc:.4f} (epoch {epoch}) → {ckpt_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_auc:.4f} @ epoch {best_epoch})")
                break

            # Save last checkpoint
            last_path = self.output_dir / f"last_model"
            if fold is not None:
                last_path = Path(str(last_path) + f"_fold{fold}")
            last_path = Path(str(last_path) + ".pt")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "config": config,
            }, last_path)

        # Save training history
        import json
        history_path = self.output_dir / f"history"
        if fold is not None:
            history_path = Path(str(history_path) + f"_fold{fold}")
        history_path = Path(str(history_path) + ".json")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n[Trainer] Training complete! Best AUC: {best_auc:.4f} @ epoch {best_epoch}")
        return str(self.output_dir / f"best_model" + (f"_fold{fold}" if fold is not None else "") + ".pt")


def model_soup(
    checkpoints: list,
    config: dict,
    weights: Optional[list] = None,
) -> nn.Module:
    """
    Model Soup: Average weights of multiple trained models.
    
    Proven effective in BirdCLEF: weight=1.5 for soups.
    
    Args:
        checkpoints: List of checkpoint paths
        config: Model config
        weights: Optional per-model weights
        
    Returns:
        Averaged model
    """
    if weights is None:
        weights = [1.0] * len(checkpoints)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    model = build_model(config)
    avg_state = {}

    for ckpt_path, w in zip(checkpoints, weights):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state_dict"]

        for key in state:
            if key not in avg_state:
                avg_state[key] = state[key] * w
            else:
                avg_state[key] = avg_state[key] + state[key] * w

    model.load_state_dict(avg_state)
    return model
