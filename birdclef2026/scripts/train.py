# ============================================================
# BirdCLEF 2026 — scripts/train.py
# Main Training Script
#
# Usage:
#   python -m scripts.train --config configs/default.yaml
#   python -m scripts.train --config configs/default.yaml --fold 0
#   python -m scripts.train --config configs/default.yaml --4fold
# ============================================================

import sys
import os
import yaml
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trainer import BirdCLEFTrainer, model_soup
from src.model import build_model, count_parameters
from src.inference import export_to_onnx


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_fold_csv(
    train_csv: str,
    taxonomy_csv: str,
    soundscape_labels_csv: str,
    output_path: str,
    n_folds: int = 4,
    seed: int = 42,
) -> str:
    """
    Create K-fold cross-validation split based on species.
    
    Ensures each fold has representative species distribution.
    Soundscapes are always in training (valuable domain data).
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)

    # Create fold assignment based on primary_label (stratified)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = train_df["primary_label"].values

    folds = np.zeros(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, labels)):
        folds[val_idx] = fold_idx

    train_df["fold"] = folds

    # Save fold CSV
    train_df[["filename", "primary_label", "fold"]].to_csv(output_path, index=False)
    print(f"[Folds] Created {n_folds}-fold split → {output_path}")
    print(f"[Folds] Distribution:")
    for f in range(n_folds):
        count = (train_df["fold"] == f).sum()
        print(f"  Fold {f}: {count} samples")

    return output_path


def train_single_model(config: dict, fold: int = None, fold_csv: str = None):
    """Train a single model (optionally with fold)."""
    trainer = BirdCLEFTrainer(config)
    best_path = trainer.train(fold=fold, fold_csv=fold_csv)
    return best_path


def train_4fold(config: dict, start_fold: int = 0):
    """Train 4-fold cross-validation models."""
    fold_csv = Path(config["output_dir"]) / "folds_4fold.csv"
    fold_csv = str(fold_csv)

    # Create folds if not exist
    if not os.path.exists(fold_csv):
        create_fold_csv(
            config["train_csv"],
            config["taxonomy_csv"],
            config.get("soundscape_labels_csv", ""),
            fold_csv,
            n_folds=4,
        )

    # Train each fold
    checkpoints = []
    for fold in range(start_fold, 4):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold}/4")
        print(f"{'='*60}")

        ckpt_path = train_single_model(config, fold=fold, fold_csv=fold_csv)
        checkpoints.append(ckpt_path)

    # Model soup across folds
    print(f"\n{'='*60}")
    print(f"Creating Model Soup from {len(checkpoints)} fold models")
    print(f"{'='*60}")

    soup_model = model_soup(checkpoints, config, weights=[1.5] * len(checkpoints))

    soup_path = Path(config["output_dir"]) / "model_soup_4fold.pt"
    torch.save({
        "model_state_dict": soup_model.state_dict(),
        "config": config,
        "source_checkpoints": checkpoints,
    }, soup_path)
    print(f"[Soup] Saved → {soup_path}")

    return checkpoints, str(soup_path)


def export_models(config: dict, checkpoint_paths: list):
    """Export trained models to ONNX."""
    onnx_dir = Path(config["output_dir"]) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoint_paths:
        model = build_model(config)

        # Load checkpoint
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        # Export
        onnx_name = Path(ckpt_path).stem + ".onnx"
        onnx_path = str(onnx_dir / onnx_name)

        export_to_onnx(
            model,
            onnx_path,
            input_shape=(1, 3, 128, 313),
            opset_version=13,
        )

    print(f"[Export] All models exported to {onnx_dir}")


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to train (0-3)")
    parser.add_argument("--4fold", action="store_true",
                        help="Train all 4 folds + model soup")
    parser.add_argument("--export", action="store_true",
                        help="Export trained models to ONNX")
    parser.add_argument("--export-only", type=str, default=None,
                        help="Export specific checkpoint to ONNX")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
        config["train_audio_dir"] = os.path.join(args.data_dir, "train_audio")
        config["train_soundscape_dir"] = os.path.join(args.data_dir, "train_soundscapes")
        config["test_soundscape_dir"] = os.path.join(args.data_dir, "test_soundscapes")
        config["train_csv"] = os.path.join(args.data_dir, "train.csv")
        config["taxonomy_csv"] = os.path.join(args.data_dir, "taxonomy.csv")
        config["soundscape_labels_csv"] = os.path.join(args.data_dir, "train_soundscapes_labels.csv")

    if args.output_dir:
        config["output_dir"] = args.output_dir
        config["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")
    if args.epochs:
        config["epochs"] = args.epochs
    if args.lr:
        config["lr"] = args.lr
    if args.batch_size:
        config["batch_size"] = args.batch_size

    # Ensure output directories exist
    os.makedirs(config.get("output_dir", "./output"), exist_ok=True)
    os.makedirs(config.get("checkpoint_dir", "./checkpoints"), exist_ok=True)

    # Save config
    config_save = Path(config.get("output_dir", "./output")) / "config_used.yaml"
    with open(config_save, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration saved to: {config_save}")

    # Export only mode
    if args.export_only:
        export_models(config, [args.export_only])
        return

    # Training
    if args._4fold:
        checkpoints, soup_path = train_4fold(config)
        if args.export:
            checkpoints.append(soup_path)
            export_models(config, checkpoints)
    elif args.fold is not None:
        fold_csv = Path(config["output_dir"]) / "folds_4fold.csv"
        if not os.path.exists(fold_csv):
            create_fold_csv(
                config["train_csv"],
                config["taxonomy_csv"],
                config.get("soundscape_labels_csv", ""),
                str(fold_csv),
                n_folds=4,
            )
        ckpt = train_single_model(config, fold=args.fold, fold_csv=str(fold_csv))
        if args.export:
            export_models(config, [ckpt])
    else:
        # Single model training (no fold)
        ckpt = train_single_model(config)
        if args.export:
            export_models(config, [ckpt])

    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
