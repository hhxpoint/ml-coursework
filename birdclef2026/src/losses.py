# ============================================================
# BirdCLEF 2026 — src/losses.py
# Loss Functions: ASL + Dual Loss + Soft Label Distillation
# 
# Proven best formula (v28):
#   ASL (γ−=4, γ+=0, clip=0.05)
#   Dual loss: clip=0.5, frame=0.5
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    
    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification" (2021)
    
    Key insight: Different focusing parameters for positive/negative labels.
    For BirdCLEF: γ−=4 (strong negative suppression), γ+=0 (no positive downweighting),
    clip=0.05 (prevent over-suppression of easy negatives).
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw logits
            targets: (batch, num_classes) multi-label targets (can be soft)
            
        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)

        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0)
        else:
            probs_neg = probs

        # Basic BCE components
        log_p = torch.log(probs.clamp(min=self.eps))
        log_1mp = torch.log((1 - probs_neg).clamp(min=self.eps))

        # Positive loss
        loss_pos = targets * log_p

        # Negative loss
        loss_neg = (1 - targets) * log_1mp

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs
            pt_neg = 1 - probs_neg

            modulator_pos = (1 - pt_pos).pow(self.gamma_pos)
            modulator_neg = (1 - pt_neg).pow(self.gamma_neg)

            loss_pos = modulator_pos * loss_pos
            loss_neg = modulator_neg * loss_neg

        loss = -(loss_pos + loss_neg)

        return loss.mean()


class SoftAsymmetricLoss(nn.Module):
    """
    Asymmetric Loss with soft label support for knowledge distillation.
    
    Used for training with Perch teacher soft pseudo-labels.
    Labels are probability vectors rather than hard 0/1.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) student logits
            soft_targets: (batch, num_classes) teacher soft labels [0, 1]
            
        Returns:
            Soft ASL loss
        """
        # Temperature-scaled logits
        scaled_logits = logits / self.temperature
        probs = torch.sigmoid(scaled_logits)

        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs_pos = probs
            probs_neg = (probs - self.clip).clamp(min=0)
        else:
            probs_pos = probs
            probs_neg = probs

        log_p = torch.log(probs_pos.clamp(min=self.eps))
        log_1mp = torch.log((1 - probs_neg).clamp(min=self.eps))

        # Soft targets work naturally with BCE
        loss_pos = soft_targets * log_p
        loss_neg = (1 - soft_targets) * log_1mp

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs_pos
            pt_neg = 1 - probs_neg

            modulator_pos = (1 - pt_pos).pow(self.gamma_pos)
            modulator_neg = (1 - pt_neg).pow(self.gamma_neg)

            loss_pos = modulator_pos * loss_pos
            loss_neg = modulator_neg * loss_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean() * (self.temperature ** 2)


class DualLoss(nn.Module):
    """
    Dual Loss: Combines clip-level and frame-level losses.
    
    Used in v28 best formula: clip_weight=0.5, frame_weight=0.5
    Provides both whole-clip classification and temporal localization supervision.
    """
    def __init__(
        self,
        clip_criterion: nn.Module,
        frame_criterion: nn.Module,
        clip_weight: float = 0.5,
        frame_weight: float = 0.5,
    ):
        super().__init__()
        self.clip_criterion = clip_criterion
        self.frame_criterion = frame_criterion
        self.clip_weight = clip_weight
        self.frame_weight = frame_weight

    def forward(
        self,
        clip_logits: torch.Tensor,
        frame_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            clip_logits: (batch, num_classes) clip-level predictions
            frame_logits: (batch, num_classes) frame-level predictions
            targets: (batch, num_classes) multi-label targets
            
        Returns:
            dict with 'loss', 'clip_loss', 'frame_loss'
        """
        clip_loss = self.clip_criterion(clip_logits, targets)
        frame_loss = self.frame_criterion(frame_logits, targets)

        total_loss = self.clip_weight * clip_loss + self.frame_weight * frame_loss

        return {
            "loss": total_loss,
            "clip_loss": clip_loss.item(),
            "frame_loss": frame_loss.item(),
        }


class SoftLabelDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for soft pseudo-labels.
    
    Combines hard label supervision (ASL) with soft label distillation
    from Perch teacher model.
    
    Used for r5 soft pseudo-labels (1176 clips × 10 oversample).
    """
    def __init__(
        self,
        hard_criterion: nn.Module,
        soft_criterion: nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.hard_criterion = hard_criterion
        self.soft_criterion = soft_criterion
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        source: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) model predictions
            hard_targets: (batch, num_classes) hard labels
            soft_targets: (batch, num_classes) soft teacher labels (or None)
            source: Data source identifier
            
        Returns:
            Combined loss
        """
        # Hard label loss (always computed)
        hard_loss = self.hard_criterion(logits, hard_targets)

        if soft_targets is not None and source in ["soft_pseudo", "combined_pseudo"]:
            # Soft distillation loss
            soft_loss = self.soft_criterion(logits, soft_targets)
            return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return hard_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = -alpha * (1 - pt).pow(self.gamma) * torch.log(pt.clamp(min=self.eps))
        return loss.mean()


class AUCPMLoss(nn.Module):
    """
    Pairwise AUC Maximization Loss.
    
    Directly optimizes AUC by comparing positive/negative pairs.
    Robust to label noise and class imbalance.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch, num_classes) binary
        """
        loss = 0.0
        num_classes = 0

        for c in range(logits.shape[1]):
            pos_mask = targets[:, c] == 1
            neg_mask = targets[:, c] == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_logits = logits[pos_mask, c]
            neg_logits = logits[neg_mask, c]

            # Pairwise ranking loss
            pairs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
            loss += F.relu(self.margin - pairs).mean()
            num_classes += 1

        if num_classes > 0:
            loss = loss / num_classes

        return loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    
    Smooths hard labels (0→ε/2, 1→1-ε/2) to reduce overconfidence.
    """
    def __init__(self, smoothing: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.smoothing = smoothing
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - self.smoothing) + self.smoothing / 2
        loss = F.binary_cross_entropy_with_logits(logits, smoothed, reduction="mean")
        return loss


def build_criterion(config: dict) -> nn.Module:
    """
    Build loss criterion from config.
    
    Returns:
        Loss function
    """
    # Primary: ASL
    asl = AsymmetricLoss(
        gamma_neg=config.get("asl_gamma_neg", 4.0),
        gamma_pos=config.get("asl_gamma_pos", 0.0),
        clip=config.get("asl_clip", 0.05),
    )

    if config.get("dual_loss", True):
        # Dual loss with ASL on both heads
        frame_asl = AsymmetricLoss(
            gamma_neg=config.get("asl_gamma_neg", 4.0),
            gamma_pos=config.get("asl_gamma_pos", 0.0),
            clip=config.get("asl_clip", 0.05),
        )
        return DualLoss(
            clip_criterion=asl,
            frame_criterion=frame_asl,
            clip_weight=config.get("dual_clip_weight", 0.5),
            frame_weight=config.get("dual_frame_weight", 0.5),
        )

    return asl
