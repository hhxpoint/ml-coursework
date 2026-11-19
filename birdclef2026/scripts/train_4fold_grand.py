# ============================================================
# BirdCLEF 2026 — scripts/train_4fold_grand.py
# 4-Fold Grand Combo Training Pipeline
#
# This is the complete multi-stage training pipeline:
#   1. Create stratified 4-fold splits
#   2. Train each fold with full kitchen-sink config
#   3. Model soup per fold (if multiple runs)
#   4. Final ensemble of all fold models
#
# Usage:
#   python -m scripts.train_4fold_grand --config configs/v28_grand.yaml
#   python -m scripts.train_4fold_grand --config configs/v28_grand.yaml --start-fold 2
#   python -m scripts.train_4fold_grand --config configs/v28_grand.yaml --export-only
# ============================================================

import sys
import os
import yaml
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import build_model, count_parameters, load_checkpoint
from src.trainer import BirdCLEFTrainer, model_soup
from src.inference import export_to_onnx
from src.dataset import create_dataloaders


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_stratified_folds(
    train_csv: str,
    taxonomy_csv: str,
    output_path: str,
    n_folds: int = 4,
    seed: int = 42,
) -> str:
    """
    Create species-stratified K-fold splits.
    
    Ensures each fold has proportional representation of all species.
    Particularly important for rare non-bird classes.
    """
    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)
    taxonomy = pd.read_csv(taxonomy_csv)

    # Stratify by primary_label
    labels = train_df["primary_label"].astype(str).values

    # Handle rare classes (< n_folds samples) by grouping
    label_counts = pd.Series(labels).value_counts()
    rare_labels = label_counts[label_counts < n_folds].index.tolist()

    stratify_labels = labels.copy()
    for rare in rare_labels:
        stratify_labels[stratify_labels == rare] = "RARE_GROUP"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = np.zeros(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, stratify_labels)):
        folds[val_idx] = fold_idx

    train_df["fold"] = folds

    # Save
    fold_df = train_df[["filename", "primary_label", "fold"]].copy()
    fold_df.to_csv(output_path, index=False)

    # Print distribution
    print(f"[Folds] Created {n_folds}-fold split → {output_path}")
    print(f"[Folds] Total samples: {len(train_df)}")
    for f in range(n_folds):
        fold_data = train_df[train_df["fold"] == f]
        n_species = fold_data["primary_label"].nunique()
        print(f"  Fold {f}: {len(fold_data)} samples, {n_species} species")

    return output_path


class GrandComboPipeline:
    """
    Complete 4-fold training pipeline for BirdCLEF 2026.
    
    Implements the proven v28 kitchen-sink formula:
    - ASL + dual loss + soft KD ×10 + hard pseudo ×5 + circ_shift + SpecAugment
    - Combined r4r5 pseudo labels (6601 clips × 10)
    - Model soup across folds (weight=1.5)
    """
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_fold(self, fold: int, fold_csv: str) -> str:
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} TRAINING")
        print(f"{'='*60}")

        trainer = BirdCLEFTrainer(self.config)
        best_path = trainer.train(fold=fold, fold_csv=fold_csv)

        return best_path

    def run_all_folds(
        self,
        start_fold: int = 0,
        n_folds: int = 4,
    ) -> list:
        """Train all folds and create model soup."""
        config = self.config

        # Step 1: Create folds
        fold_csv = str(self.output_dir / "folds_grand.csv")
        if not os.path.exists(fold_csv):
            create_stratified_folds(
                config["train_csv"],
                config["taxonomy_csv"],
                fold_csv,
                n_folds=n_folds,
            )

        # Step 2: Train each fold
        checkpoints = []
        for fold in range(start_fold, n_folds):
            ckpt_path = self.run_fold(fold, fold_csv)
            checkpoints.append(ckpt_path)

        # Step 3: Model soup
        print(f"\n{'='*60}")
        print(f"  MODEL SOUP ({len(checkpoints)} folds)")
        print(f"{'='*60}")

        soup_model = model_soup(
            checkpoints, config,
            weights=[1.5] * len(checkpoints),
        )

        soup_path = self.output_dir / "model_soup_grand.pt"
        torch.save({
            "model_state_dict": soup_model.state_dict(),
            "config": config,
            "fold_checkpoints": checkpoints,
            "soup_weights": [1.5] * len(checkpoints),
        }, soup_path)

        print(f"[Soup] Saved → {soup_path}")

        # Step 4: Export all to ONNX
        if config.get("export_onnx", True):
            self.export_all(checkpoints + [str(soup_path)])

        return checkpoints, str(soup_path)

    def export_all(self, checkpoint_paths: list):
        """Export all models to ONNX."""
        onnx_dir = self.output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_path in checkpoint_paths:
            model = build_model(self.config)

            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)

            model.eval()

            onnx_name = Path(ckpt_path).stem + ".onnx"
            onnx_path = str(onnx_dir / onnx_name)

            export_to_onnx(
                model, onnx_path,
                input_shape=(1, 3, 128, 313),
                opset_version=13,
            )

        print(f"[Export] All models exported to {onnx_dir}")


def main():
    parser = argparse.ArgumentParser(description="4-Fold Grand Combo Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Start from specific fold")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--export-only", action="store_true",
                        help="Only export existing checkpoints to ONNX")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    # Overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
        config["train_audio_dir"] = os.path.join(args.data_dir, "train_audio")
        config["train_soundscape_dir"] = os.path.join(args.data_dir, "train_soundscapes")
        config["train_csv"] = os.path.join(args.data_dir, "train.csv")
        config["taxonomy_csv"] = os.path.join(args.data_dir, "taxonomy.csv")
        config["soundscape_labels_csv"] = os.path.join(args.data_dir, "train_soundscape_labels.csv")

    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Run pipeline
    pipeline = GrandComboPipeline(config)

    if args.export_only:
        # Find existing checkpoints
        ckpt_dir = Path(config.get("output_dir", "./output"))
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        pipeline.export_all([str(c) for c in ckpts])
    else:
        checkpoints, soup_path = pipeline.run_all_folds(
            start_fold=args.start_fold,
            n_folds=args.n_folds,
        )

    print("\n[DONE] 4-Fold Grand Combo complete!")


if __name__ == "__main__":
    main()
