# ============================================================
# BirdCLEF 2026 — scripts/evaluate.py
# Validation & Evaluation Script (OOF AUC)
#
# Computes out-of-fold ROC-AUC on soundscape validation split.
# This is the primary offline metric that correlates with LB.
#
# Usage:
#   python -m scripts.evaluate --config configs/default.yaml --checkpoint output/best_model_fold0.pt
#   python -m scripts.evaluate --config configs/default.yaml --onnx output/onnx/best_model.onnx
#   python -m scripts.evaluate --config configs/default.yaml --checkpoint output/best_model.pt --soundscape-dir data/train_soundscapes
# ============================================================

import sys
import os
import yaml
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import build_model, load_checkpoint
from src.inference import ONNXInferenceEngine, PyTorchInferenceEngine
from src.features import load_audio, create_three_channel_mel, crop_or_pad_mel
from src.postprocessing import apply_postprocessing


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_soundscape_split(
    checkpoint_path: str,
    config: dict,
    soundscape_dir: str,
    soundscape_labels_csv: str,
    taxonomy_csv: str,
    use_onnx: bool = False,
    apply_postproc: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on soundscape validation split.
    
    Computes:
    - Macro ROC-AUC (main metric)
    - Per-class AUC
    - Per-taxon AUC (Aves, Amphibia, Insecta, Mammalia, Reptilia)
    - Average Precision
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dict
        soundscape_dir: Directory with soundscape audio files
        soundscape_labels_csv: Ground truth labels CSV
        taxonomy_csv: Species taxonomy CSV
        use_onnx: Use ONNX engine instead of PyTorch
        apply_postproc: Apply post-processing
        
    Returns:
        dict of evaluation metrics
    """
    taxonomy = pd.read_csv(taxonomy_csv)
    species_list = sorted(taxonomy["primary_label"].unique().tolist())
    num_classes = len(species_list)
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    # Build species→taxon mapping
    species_to_taxon = {}
    for _, row in taxonomy.iterrows():
        sp = row["primary_label"]
        taxon = row.get("class_name", "Unknown")
        species_to_taxon[sp] = taxon

    # Load ground truth labels
    labels_df = pd.read_csv(soundscape_labels_csv)

    # Build ground truth matrix
    gt_labels = {}
    for _, row in labels_df.iterrows():
        row_id = str(row.get("row_id", ""))
        if not row_id:
            fname = row.get("filename", "")
            # Try to extract window from filename like "file_1.ogg"
            row_id = fname.replace(".ogg", "").replace(".wav", "")

        label_vec = np.zeros(num_classes, dtype=np.float32)
        for sp in species_list:
            if sp in row and pd.notna(row[sp]):
                if float(row[sp]) > 0:
                    label_vec[species_to_idx[sp]] = 1.0
        gt_labels[row_id] = label_vec

    print(f"[Eval] Ground truth: {len(gt_labels)} windows")

    # Load model
    sample_rate = config["sample_rate"]
    duration = config.get("infer_duration", 5.0)
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    use_pcen = config["use_pcen"]
    window_samples = int(sample_rate * duration)
    target_time_steps = int(sample_rate * duration / hop_length) + 1

    if use_onnx:
        engine = ONNXInferenceEngine(
            checkpoint_path,
            num_threads=config.get("onnx_threads", 8),
        )
    else:
        model = build_model(config)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        engine = PyTorchInferenceEngine(model, device="cpu")

    # Process soundscape files
    soundscape_dir = Path(soundscape_dir)
    files = sorted(soundscape_dir.glob("*.ogg"))
    if len(files) == 0:
        files = sorted(soundscape_dir.glob("*.wav"))

    all_preds = []
    all_gts = []
    all_row_ids = []

    for fpath in files:
        fname = fpath.stem

        # Load audio
        full_audio, _ = load_audio(str(fpath), sr=sample_rate, duration=None)
        num_windows = len(full_audio) // window_samples

        for w in range(num_windows):
            start_sample = w * window_samples
            end_sample = start_sample + window_samples
            window_audio = full_audio[start_sample:end_sample]

            if len(window_audio) < window_samples:
                window_audio = np.pad(window_audio, (0, window_samples - len(window_audio)))

            mel = create_three_channel_mel(
                window_audio, sr=sample_rate,
                n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=fmin, fmax=fmax,
                use_pcen=use_pcen,
            )
            mel = crop_or_pad_mel(mel, target_time_steps)

            mel_batch = mel[np.newaxis, ...]
            probs = engine.predict_batch(mel_batch)
            pred = probs[0]

            row_id = f"{fname}_{w + 1}"

            if row_id in gt_labels:
                all_preds.append(pred)
                all_gts.append(gt_labels[row_id])
                all_row_ids.append(row_id)

    if not all_preds:
        print("[Eval] No matching predictions found!")
        return {}

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)

    print(f"[Eval] Matched {len(all_preds)} windows")

    # Apply post-processing
    if apply_postproc:
        all_preds = apply_postprocessing(all_preds, config)

    # Compute metrics
    metrics = {}

    # Per-class AUC
    class_aucs = []
    class_aps = []
    taxon_aucs = {}

    for c in range(num_classes):
        if all_gts[:, c].sum() > 0:
            try:
                auc = roc_auc_score(all_gts[:, c], all_preds[:, c])
                class_aucs.append(auc)

                ap = average_precision_score(all_gts[:, c], all_preds[:, c])
                class_aps.append(ap)

                # Per-taxon
                sp = species_list[c]
                taxon = species_to_taxon.get(sp, "Unknown")
                if taxon not in taxon_aucs:
                    taxon_aucs[taxon] = []
                taxon_aucs[taxon].append(auc)

            except Exception as e:
                pass

    # Macro AUC (main metric)
    metrics["macro_auc"] = float(np.mean(class_aucs)) if class_aucs else 0.0
    metrics["macro_ap"] = float(np.mean(class_aps)) if class_aps else 0.0
    metrics["num_classes_evaluated"] = len(class_aucs)

    # Per-taxon metrics
    for taxon, aucs in taxon_aucs.items():
        metrics[f"auc_{taxon.lower()}"] = float(np.mean(aucs))

    # Std of AUC
    metrics["auc_std"] = float(np.std(class_aucs)) if class_aucs else 0.0

    # Worst classes
    if class_aucs:
        sorted_indices = np.argsort(class_aucs)
        metrics["worst_5_auc"] = float(np.mean([class_aucs[i] for i in sorted_indices[:5]]))
        metrics["best_5_auc"] = float(np.mean([class_aucs[i] for i in sorted_indices[-5:]]))

    return metrics


def evaluate_ensemble(
    checkpoint_paths: list,
    weights: list,
    config: dict,
    soundscape_dir: str,
    soundscape_labels_csv: str,
    taxonomy_csv: str,
) -> Dict[str, float]:
    """Evaluate ensemble of models."""
    taxonomy = pd.read_csv(taxonomy_csv)
    species_list = sorted(taxonomy["primary_label"].unique().tolist())
    num_classes = len(species_list)
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    # Load labels
    labels_df = pd.read_csv(soundscape_labels_csv)
    gt_labels = {}
    for _, row in labels_df.iterrows():
        row_id = str(row.get("row_id", ""))
        if not row_id:
            fname = row.get("filename", "")
            row_id = fname.replace(".ogg", "").replace(".wav", "")

        label_vec = np.zeros(num_classes, dtype=np.float32)
        for sp in species_list:
            if sp in row and pd.notna(row[sp]):
                if float(row[sp]) > 0:
                    label_vec[species_to_idx[sp]] = 1.0
        gt_labels[row_id] = label_vec

    # Load all models
    models = []
    for ckpt_path in checkpoint_paths:
        model = build_model(config)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        models.append(model)

    weights = np.array(weights) / sum(weights)

    # Process files
    sample_rate = config["sample_rate"]
    duration = config.get("infer_duration", 5.0)
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    use_pcen = config["use_pcen"]
    window_samples = int(sample_rate * duration)
    target_time_steps = int(sample_rate * duration / hop_length) + 1

    soundscape_dir = Path(soundscape_dir)
    files = sorted(soundscape_dir.glob("*.ogg"))
    if not files:
        files = sorted(soundscape_dir.glob("*.wav"))

    all_preds = []
    all_gts = []

    for fpath in files:
        fname = fpath.stem
        full_audio, _ = load_audio(str(fpath), sr=sample_rate, duration=None)
        num_windows = len(full_audio) // window_samples

        for w in range(num_windows):
            start_sample = w * window_samples
            end_sample = start_sample + window_samples
            window_audio = full_audio[start_sample:end_sample]

            if len(window_audio) < window_samples:
                window_audio = np.pad(window_audio, (0, window_samples - len(window_audio)))

            mel = create_three_channel_mel(
                window_audio, sr=sample_rate,
                n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=fmin, fmax=fmax,
                use_pcen=use_pcen,
            )
            for c in range(3):
                mel[c] = crop_or_pad_mel(mel[c], target_time_steps)

            mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)

            # Ensemble prediction
            ensemble_probs = np.zeros(num_classes, dtype=np.float32)
            with torch.no_grad():
                for model, w in zip(models, weights):
                    out = model(mel_tensor)
                    logits = out["clip_logits"]
                    probs = torch.sigmoid(logits).numpy()[0]
                    ensemble_probs += w * probs

            row_id = f"{fname}_{w + 1}"
            if row_id in gt_labels:
                all_preds.append(ensemble_probs)
                all_gts.append(gt_labels[row_id])

    if not all_preds:
        return {"macro_auc": 0.0}

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)

    # Apply post-processing
    all_preds = apply_postprocessing(all_preds, config)

    # Macro AUC
    class_aucs = []
    for c in range(num_classes):
        if all_gts[:, c].sum() > 0:
            try:
                auc = roc_auc_score(all_gts[:, c], all_preds[:, c])
                class_aucs.append(auc)
            except:
                pass

    return {
        "macro_auc": float(np.mean(class_aucs)) if class_aucs else 0.0,
        "num_classes": len(class_aucs),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BirdCLEF 2026 Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--ensemble", type=str, nargs="+", default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--soundscape-dir", type=str, default=None)
    parser.add_argument("--soundscape-labels", type=str, default=None)
    parser.add_argument("--taxonomy", type=str, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.data_dir:
        config["data_dir"] = args.data_dir
        config["train_soundscape_dir"] = os.path.join(args.data_dir, "train_soundscapes")
        config["taxonomy_csv"] = os.path.join(args.data_dir, "taxonomy.csv")
        config["soundscape_labels_csv"] = os.path.join(args.data_dir, "train_soundscape_labels.csv")

    soundscape_dir = args.soundscape_dir or config.get("train_soundscape_dir", "./data/train_soundscapes")
    soundscape_labels = args.soundscape_labels or config.get("soundscape_labels_csv", "")
    taxonomy_csv = args.taxonomy or config.get("taxonomy_csv", "")

    if args.ensemble:
        metrics = evaluate_ensemble(
            args.ensemble,
            args.weights or [1.0] * len(args.ensemble),
            config, soundscape_dir, soundscape_labels, taxonomy_csv,
        )
    else:
        ckpt = args.onnx or args.checkpoint
        if not ckpt:
            print("Error: provide --checkpoint or --onnx")
            return

        metrics = evaluate_soundscape_split(
            ckpt, config, soundscape_dir, soundscape_labels, taxonomy_csv,
            use_onnx=(args.onnx is not None),
            apply_postproc=(not args.no_postprocess),
        )

    # Print results
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print("=" * 50)

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
