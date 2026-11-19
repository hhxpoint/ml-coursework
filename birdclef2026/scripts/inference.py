# ============================================================
# BirdCLEF 2026 — scripts/inference.py
# Inference Script for Kaggle Submission
#
# Usage:
#   python -m scripts.inference --config configs/default.yaml --onnx output/onnx/best_model.onnx
#   python -m scripts.inference --config configs/default.yaml --soundscape-dir data/test_soundscapes
# ============================================================

import sys
import os
import yaml
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import (
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    run_inference_on_soundscapes,
    profile_inference_speed,
    export_to_onnx,
)
from src.model import build_model, load_checkpoint
from src.postprocessing import (
    apply_postprocessing,
    ensemble_predictions,
)
from src.features import (
    load_audio,
    create_three_channel_mel,
    crop_or_pad_mel,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_submission_single_model(
    config: dict,
    onnx_path: str,
    soundscape_dir: str,
    output_csv: str = "submission.csv",
):
    """Generate submission using a single ONNX model."""
    taxonomy = pd.read_csv(config["taxonomy_csv"])
    species_list = sorted(taxonomy["primary_label"].unique().tolist())

    output_path = run_inference_on_soundscapes(
        soundscape_dir=soundscape_dir,
        onnx_path=onnx_path,
        config=config,
        output_csv=output_csv,
        species_list=species_list,
    )

    return output_path


def generate_submission_ensemble(
    config: dict,
    onnx_paths: list,
    weights: list,
    soundscape_dir: str,
    output_csv: str = "submission_ensemble.csv",
    ensemble_method: str = "quantile_mix",
):
    """Generate submission using ensemble of ONNX models."""
    taxonomy = pd.read_csv(config["taxonomy_csv"])
    species_list = sorted(taxonomy["primary_label"].unique().tolist())

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

    # Find soundscape files
    soundscape_dir = Path(soundscape_dir)
    files = sorted(soundscape_dir.glob("*.ogg"))
    if len(files) == 0:
        files = sorted(soundscape_dir.glob("*.wav"))

    print(f"[Ensemble] {len(files)} files, {len(onnx_paths)} models")

    # Load ONNX engines
    engines = []
    for onnx_path in onnx_paths:
        engine = ONNXInferenceEngine(
            onnx_path=onnx_path,
            num_threads=config.get("onnx_threads", 8),
            batch_size=config.get("onnx_batch_size", 64),
        )
        engines.append(engine)

    start_time = time.time()
    all_row_ids = []
    all_predictions = []

    for file_idx, fpath in enumerate(files):
        fname = fpath.name.replace(fpath.suffix, "")

        # Load audio
        audio, _ = load_audio(str(fpath), sr=sample_rate, duration=None)
        num_windows = len(audio) // window_samples

        # Get predictions from each engine
        file_preds_all = []
        for engine_idx, engine in enumerate(engines):
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
            file_preds_all.append(preds)

        # Ensemble predictions per file
        min_windows = min(p.shape[0] for p in file_preds_all)
        file_preds_trimmed = [p[:min_windows] for p in file_preds_all]

        ensembled = ensemble_predictions(
            file_preds_trimmed,
            weights=weights,
            method=ensemble_method,
        )

        for w in range(min_windows):
            all_row_ids.append(f"{fname}_{w + 1}")
            all_predictions.append(ensembled[w])

        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[Ensemble] {file_idx + 1}/{len(files)} | {elapsed:.1f}s")

    all_predictions = np.array(all_predictions)

    # Apply post-processing
    if config.get("apply_postprocessing", True):
        all_predictions = apply_postprocessing(all_predictions, config)

    # Write submission
    submission = pd.DataFrame({"row_id": all_row_ids})
    for i, sp in enumerate(species_list):
        submission[sp] = all_predictions[:, i]

    submission.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    print(f"[Ensemble] Complete! {total_time:.1f}s → {output_csv}")

    return output_csv


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--onnx", type=str, nargs="+", default=None,
                        help="ONNX model path(s)")
    parser.add_argument("--soundscape-dir", type=str, default=None,
                        help="Test soundscape directory")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Ensemble weights")
    parser.add_argument("--method", type=str, default="quantile_mix",
                        choices=["mean", "rank", "quantile_mix"])
    parser.add_argument("--profile", action="store_true",
                        help="Profile inference speed")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable post-processing")
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    # Apply overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
        config["train_audio_dir"] = os.path.join(args.data_dir, "train_audio")
        config["test_soundscape_dir"] = os.path.join(args.data_dir, "test_soundscapes")
        config["train_csv"] = os.path.join(args.data_dir, "train.csv")
        config["taxonomy_csv"] = os.path.join(args.data_dir, "taxonomy.csv")

    if args.no_postprocess:
        config["apply_postprocessing"] = False

    # Determine soundscape directory
    soundscape_dir = args.soundscape_dir or config.get("test_soundscape_dir", "./data/test_soundscapes")

    # Determine ONNX paths
    onnx_paths = args.onnx
    if onnx_paths is None:
        # Auto-discover
        onnx_dir = Path(config.get("output_dir", "./output")) / "onnx"
        if onnx_dir.exists():
            onnx_paths = sorted(str(p) for p in onnx_dir.glob("*.onnx"))

    if not onnx_paths:
        print("Error: No ONNX models specified or found. Use --onnx flag.")
        return

    print(f"ONNX models: {onnx_paths}")
    print(f"Soundscape dir: {soundscape_dir}")

    # Profile mode
    if args.profile:
        for onnx_path in onnx_paths:
            print(f"\nProfiling: {onnx_path}")
            profile_inference_speed(onnx_path, config, num_files=5)
        return

    # Single model
    if len(onnx_paths) == 1:
        generate_submission_single_model(
            config, onnx_paths[0], soundscape_dir, args.output,
        )
    else:
        # Ensemble
        weights = args.weights or [1.0] * len(onnx_paths)
        generate_submission_ensemble(
            config, onnx_paths, weights, soundscape_dir,
            args.output, args.method,
        )


if __name__ == "__main__":
    main()
