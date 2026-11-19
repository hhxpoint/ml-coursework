# ============================================================
# BirdCLEF 2026 — src/model.py
# Model Architecture: EfficientNet-B0 + GEM Pooling
# 
# Proven best formula (v28):
#   Backbone: tf_efficientnet_b0.ns_jft_in1k (6.2M params)
#   GEM Pool: p_init=3.0
#   Dropout: 0.1
#   In channels: 3
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling.
    Adaptive pooling with learnable exponent p.
    
    Reference: Radenović et al., "Fine-Tuning CNN Image Retrieval" (2018)
    
    When p=1 → average pooling, p→∞ → max pooling
    Proven effective in BirdCLEF: p_init=3.0
    """
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width) feature maps
        Returns:
            (batch, channels) pooled features
        """
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling as alternative to GEM.
    Learns to weight spatial locations for aggregation.
    """
    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) or (batch, channels, h, w)
        Returns:
            (batch, features) attended features
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)

        weights = self.attention(x)  # (b, seq_len, 1)
        return (x * weights).sum(dim=1)  # (b, features)


class BirdCLEFModel(nn.Module):
    """
    BirdCLEF 2026 Main Model.
    
    Architecture:
        Input: (batch, 3, n_mels, time) 3-channel mel-spectrogram
        → EfficientNet-B0 backbone (pretrained on JFT)
        → GeM Pooling (p=3.0, learnable)
        → Dropout (0.1)
        → Linear classifier (num_classes)
    
    Supports dual loss (clip + frame-level predictions).
    """
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
        num_classes: int = 234,
        in_channels: int = 3,
        dropout: float = 0.1,
        gem_pool: bool = True,
        gem_p_init: float = 3.0,
        pretrained: bool = True,
        dual_head: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dual_head = dual_head
        self.gem_pool = gem_pool

        # Build backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # Remove classifier head
            global_pool="",  # Remove global pooling
        )

        # Get feature dimensions
        dummy = torch.randn(1, in_channels, 128, 313)  # Typical mel shape
        with torch.no_grad():
            features = self.backbone(dummy)
            if features.dim() == 4:
                self.feature_channels = features.shape[1]
                self.feature_h = features.shape[2]
                self.feature_w = features.shape[3]
            else:
                self.feature_channels = features.shape[-1]

        # Pooling
        if gem_pool:
            self.pool = GeMPooling(p_init=gem_p_init)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Dropout
        self.drop = nn.Dropout(dropout)

        # Classifier head
        self.classifier = nn.Linear(self.feature_channels, num_classes)

        # Frame-level classifier for dual loss
        if dual_head:
            self.frame_classifier = nn.Sequential(
                nn.Linear(self.feature_channels, num_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """
        Args:
            x: (batch, 3, n_mels, time) mel-spectrogram
            return_features: If True, return intermediate features
            
        Returns:
            dict with:
                'clip_logits': (batch, num_classes) clip-level predictions
                'frame_logits': (batch, num_classes) frame-level predictions (if dual_head)
                'features': (batch, channels) features (if return_features)
        """
        features = self.backbone(x)

        if features.dim() == 4:
            # Spatial features
            if self.gem_pool:
                pooled = self.pool(features)
            else:
                pooled = self.pool(features).squeeze(-1).squeeze(-1)

            # Frame-level: average over width (time axis) for per-frame predictions
            if self.dual_head:
                # (batch, channels, h, w) → (batch, w, channels)
                frame_features = features.mean(dim=2)  # avg over freq axis
                frame_features = frame_features.permute(0, 2, 1)  # (batch, w, channels)
                frame_logits = self.frame_classifier(frame_features)  # (batch, w, classes)
                frame_logits = frame_logits.mean(dim=1)  # (batch, classes)
        else:
            pooled = features

        pooled = self.drop(pooled)
        clip_logits = self.classifier(pooled)

        result = {"clip_logits": clip_logits}

        if self.dual_head and features.dim() == 4:
            result["frame_logits"] = frame_logits

        if return_features:
            result["features"] = pooled

        return result


class BirdCLEFSEDModel(nn.Module):
    """
    BirdCLEF 2026 Sound Event Detection (SED) Model.
    
    Frame-level predictions for temporal localization.
    Outputs per-frame probabilities instead of clip-level.
    """
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
        num_classes: int = 234,
        in_channels: int = 3,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

        # Get feature dimensions
        dummy = torch.randn(1, in_channels, 128, 313)
        with torch.no_grad():
            features = self.backbone(dummy)
            self.feature_channels = features.shape[1]

        self.drop = nn.Dropout(dropout)

        # Frame-level classifier: predicts per-time-step
        self.frame_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, num_classes, kernel_size=1),
        )

        # Clip-level aggregation for training
        self.clip_head = nn.Linear(self.feature_channels, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, 3, n_mels, time)
            
        Returns:
            dict:
                'frame_probs': (batch, num_classes, time_frames)
                'clip_logits': (batch, num_classes)
        """
        features = self.backbone(x)  # (batch, C, H, W)

        # Frame-level predictions
        frame_logits = self.frame_head(features)  # (batch, classes, H, W)
        frame_logits = frame_logits.mean(dim=2)  # avg over freq → (batch, classes, W)
        frame_probs = torch.sigmoid(frame_logits)

        # Clip-level: global average
        clip_features = features.mean(dim=(-2, -1))  # (batch, C)
        clip_logits = self.clip_head(self.drop(clip_features))

        return {
            "frame_probs": frame_probs,
            "clip_logits": clip_logits,
        }


class ModelEnsemble(nn.Module):
    """
    Weighted ensemble of multiple models for inference.
    """
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with weighted averaging."""
        outputs = []
        for model in self.models:
            with torch.no_grad():
                out = model(x)
                logits = out["clip_logits"]
                probs = torch.sigmoid(logits)
                outputs.append(probs)

        # Weighted average
        weights = self.weights / self.weights.sum()
        ensemble_probs = sum(w * p for w, p in zip(weights, outputs))
        return ensemble_probs


def build_model(config: dict) -> BirdCLEFModel:
    """
    Build model from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BirdCLEFModel instance
    """
    return BirdCLEFModel(
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        in_channels=config["in_channels"],
        dropout=config["dropout"],
        gem_pool=config.get("gem_pool", True),
        gem_p_init=config.get("gem_p_init", 3.0),
        pretrained=True,
        dual_head=config.get("dual_loss", True),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    return model
