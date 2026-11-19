# ============================================================
# BirdCLEF 2026 — scripts/generate_pseudo_labels.py
# Pseudo-Label Generation Pipeline
#
# Generates soft and hard pseudo-labels for unlabeled soundscapes
# using trained model predictions.
# ============================================================

import sys
import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import ONNXInferenceEngine
from src.features import (
    load_audio,
    create_three_channel_mel,
    crop_or_pad_mel,
)


def generate_pseudo_labels(
    onnx_path: str,
    soundscape_dir: str,
    taxonomy_csv: str,
    config: dict,
    soft_threshold: float = 0.3,
    hard_threshold: float = 0.7,
    output_dir: str = "./data/pseudo",
):
    """
    Generate pseudo-labels from unlabeled soundscapes.
    
    Args:
        onnx_path: Trained ONNX model path
        soundscape_dir: Directory with unlabeled soundscape .ogg files
        taxonomy_csv: Species taxonomy CSV
        config: Configuration dict
        soft_threshold: Minimum confidence for soft pseudo-labels
        hard_threshold: Minimum confidence for hard pseudo-labels
        output_dir: Output directory for pseudo-label CSVs
    """
    taxonomy = pd.read_csv(taxonomy_csv)
    species_list = sorted(taxonomy["primary_label"].unique().tolist())
    num_classes = len(species_list)

    sample_rate = config["sample_rate"]
    duration = config.get("infer_duration", 5.0)
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    use_pcen = config["use_pcen"]

    # Load ONNX engine
    engine = ONNXInferenceEngine(
        onnx_path=onnx_path,
        num_threads=config.get("onnx_threads", 8),
        batch_size=config.get("onnx_batch_size", 64),
    )

    # Find soundscape files
    soundscape_dir = Path(soundscape_dir)
    files = sorted(soundscape_dir.glob("*.ogg"))
    if len(files) == 0:
        files = sorted(soundscape_dir.glob("*.wav"))

    print(f"[Pseudo] Processing {len(files)} soundscape files")

    # Process each file
    all_soft_rows = []
    all_hard_rows = []

    for file_idx, fpath in enumerate(files):
        fname = fpath.name.replace(fpath.suffix, "")

        # Predict on all windows
        preds = engine.predict_file(
            str(fpath),
            sample_rate=sample_rate,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            use_pcen=use_pcen,
        )

        # For each window, generate pseudo-labels
        for w in range(preds.shape[0]):
            window_probs = preds[w]
            row_id = f"{fname}_{w + 1}"

            # Soft pseudo-label: include all predictions above threshold
            soft_max = window_probs.max()
            if soft_max >= soft_threshold:
                soft_row = {"row_id": row_id, "filename": fpath.name, "window": w + 1}
                for i, sp in enumerate(species_list):
                    if window_probs[i] >= soft_threshold:
                        soft_row[sp] = float(window_probs[i])
                all_soft_rows.append(soft_row)

            # Hard pseudo-label: only high-confidence predictions
            hard_max = window_probs.max()
            if hard_max >= hard_threshold:
                hard_row = {"row_id": row_id, "filename": fpath.name, "window": w + 1}
                for i, sp in enumerate(species_list):
                    if window_probs[i] >= hard_threshold:
                        hard_row[sp] = 1.0
                all_hard_rows.append(hard_row)

        if (file_idx + 1) % 100 == 0:
            print(f"  Processed {file_idx + 1}/{len(files)} files")

    # Save pseudo-labels
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_soft_rows:
        soft_df = pd.DataFrame(all_soft_rows)
        soft_path = output_dir / "soft_pseudo_r5.csv"
        soft_df.to_csv(soft_path, index=False)
        print(f"[Pseudo] Soft labels: {len(soft_df)} clips → {soft_path}")

    if all_hard_rows:
        hard_df = pd.DataFrame(all_hard_rows)
        hard_path = output_dir / "hard_pseudo_r5.csv"
        hard_df.to_csv(hard_path, index=False)
        print(f"[Pseudo] Hard labels: {len(hard_df)} clips → {hard_path}")

    return str(output_dir / "soft_pseudo_r5.csv"), str(output_dir / "hard_pseudo_r5.csv")


def combine_pseudo_labels(
    soft_csvs: List[str],
    output_path: str,
    species_list: List[str],
):
    """
    Combine pseudo-labels from multiple rounds (r4 + r5).
    
    For combined_pseudo_r4r5.csv (6601 unique clips × 10 oversample).
    """
    combined = {}

    for csv_path in soft_csvs:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = row.get("row_id", row.get("filename", ""))
            if key not in combined:
                combined[key] = {"row_id": key, "filename": row.get("filename", "")}
                for sp in species_list:
                    combined[key][sp] = 0.0

            for sp in species_list:
                if sp in row and pd.notna(row[sp]):
                    val = float(row[sp])
                    combined[key][sp] = max(combined[key].get(sp, 0.0), val)

    combined_df = pd.DataFrame(list(combined.values()))
    combined_df.to_csv(output_path, index=False)
    print(f"[Combine] {len(combined_df)} unique clips → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--onnx", type=str, required=True,
                        help="ONNX model path")
    parser.add_argument("--soundscape-dir", type=str, default=None,
                        help="Unlabeled soundscape directory")
    parser.add_argument("--soft-threshold", type=float, default=0.3)
    parser.add_argument("--hard-threshold", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="./data/pseudo")
    parser.add_argument("--combine", type=str, nargs="+", default=None,
                        help="Combine multiple pseudo CSVs")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.soundscape_dir is None:
        args.soundscape_dir = config.get("train_soundscape_dir", "./data/train_soundscapes")

    if args.combine:
        taxonomy = pd.read_csv(config["taxonomy_csv"])
        species_list = sorted(taxonomy["primary_label"].unique().tolist())
        combine_pseudo_labels(args.combine, args.output_dir + "/combined_pseudo_r4r5.csv", species_list)
    else:
        generate_pseudo_labels(
            onnx_path=args.onnx,
            soundscape_dir=args.soundscape_dir,
            taxonomy_csv=config["taxonomy_csv"],
            config=config,
            soft_threshold=args.soft_threshold,
            hard_threshold=args.hard_threshold,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
