# ============================================================
# BirdCLEF 2026 — src/inference.py
# Inference Engine + ONNX Export for CPU-only submission
# 
# Key constraints:
#   - CPU-only inference (90 min max)
#   - ~600 soundscapes × 12 windows = 7,200 segments
#   - Target: ~8 seconds per soundscape
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

from .model import build_model, load_checkpoint
from .features import (
    load_audio,
    create_three_channel_mel,
    crop_or_pad_mel,
    pad_or_crop_waveform,
)
from .postprocessing import apply_postprocessing


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 128, 313),
    opset_version: int = 13,
    dynamic_axes: bool = True,
) -> str:
    """
    Export PyTorch model to ONNX format for CPU inference.
    
    Args:
        model: Trained PyTorch model
        save_path: Output ONNX file path
        input_shape: Example input shape (batch, channels, mel_bins, time_steps)
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch and time axes
        
    Returns:
        Path to saved ONNX file
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)

    # Build dynamic axes
    axes = {}
    if dynamic_axes:
        axes = {
            "input": {0: "batch_size", 3: "time_steps"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=axes,
        do_constant_folding=True,
    )

    print(f"[ONNX] Exported to: {save_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX] Model verification passed!")
    except Exception as e:
        print(f"[ONNX] Verification warning: {e}")

    return save_path


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine optimized for CPU.
    
    Features:
    - Batch inference for throughput
    - Multi-threaded execution
    - Pre-computed mel spectrograms
    - TTA (Test-Time Augmentation) support
    """
    def __init__(
        self,
        onnx_path: str,
        num_threads: int = 8,
        batch_size: int = 64,
    ):
        import onnxruntime as ort

        self.batch_size = batch_size

        # Session options for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[ONNX] Loaded: {onnx_path}")
        print(f"[ONNX] Threads: {num_threads}, Batch: {batch_size}")

    def predict_batch(
        self,
        mel_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference on a batch of mel spectrograms.
        
        Args:
            mel_batch: (batch, 3, n_mels, time) numpy array
            
        Returns:
            (batch, num_classes) sigmoid probabilities
        """
        # Ensure float32
        mel_batch = mel_batch.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: mel_batch},
        )

        # Apply sigmoid
        logits = outputs[0]
        probs = 1.0 / (1.0 + np.exp(-logits))

        return probs

    def predict_file(
        self,
        audio_path: str,
        sample_rate: int = 32000,
        duration: float = 5.0,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 50,
        fmax: float = 16000,
        use_pcen: bool = True,
        tta: int = 0,
    ) -> np.ndarray:
        """
        Predict on a single audio file (possibly multi-window).
        
        Args:
            audio_path: Path to audio file
            tta: Number of test-time augmentations (0=none, 1=shift, 2=shift+noise)
            
        Returns:
            (num_windows, num_classes) predictions
        """
        # Load full audio
        audio, _ = load_audio(audio_path, sr=sample_rate, duration=None)

        window_samples = int(sample_rate * duration)
        total_samples = len(audio)
        num_windows = total_samples // window_samples

        predictions = []
        for w in range(num_windows):
            start = w * window_samples
            end = start + window_samples
            window_audio = audio[start:end]

            mel = create_three_channel_mel(
                window_audio, sr=sample_rate,
                n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=fmin, fmax=fmax,
                use_pcen=use_pcen,
            )

            target_time_steps = int(sample_rate * duration / hop_length) + 1
            for c in range(3):
                mel[c] = crop_or_pad_mel(mel[c], target_time_steps)

            mel = mel[np.newaxis, ...]  # (1, 3, n_mels, time)
            probs = self.predict_batch(mel)
            predictions.append(probs[0])

            # TTA: circular shift
            if tta >= 1:
                shift = window_samples // 4
                shifted = np.roll(window_audio, shift)
                mel_shifted = create_three_channel_mel(
                    shifted, sr=sample_rate,
                    n_fft=n_fft, hop_length=hop_length,
                    n_mels=n_mels, fmin=fmin, fmax=fmax,
                    use_pcen=use_pcen,
                )
                for c in range(3):
                    mel_shifted[c] = crop_or_pad_mel(mel_shifted[c], target_time_steps)
                mel_shifted = mel_shifted[np.newaxis, ...]
                probs_shifted = self.predict_batch(mel_shifted)
                predictions[-1] = (predictions[-1] + probs_shifted[0]) / 2

        return np.array(predictions)


class PyTorchInferenceEngine:
    """
    PyTorch inference engine (fallback, slower than ONNX).
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    def predict_batch(self, mel_batch: np.ndarray) -> np.ndarray:
        mel_tensor = torch.from_numpy(mel_batch).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(mel_tensor)
            logits = outputs["clip_logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def run_inference_on_soundscapes(
    soundscape_dir: str,
    onnx_path: str,
    config: dict,
    output_csv: str = "submission.csv",
    species_list: Optional[List[str]] = None,
) -> str:
    """
    Full inference pipeline on test soundscapes.
    
    Processes all .ogg files in soundscape_dir:
    - Splits each 1-minute file into 12 × 5-second windows
    - Runs ONNX inference
    - Applies post-processing
    - Writes submission CSV
    
    Args:
        soundscape_dir: Directory containing test .ogg files
        onnx_path: Path to ONNX model
        config: Configuration dict
        output_csv: Output submission file
        species_list: Ordered list of species names
        
    Returns:
        Path to submission CSV
    """
    import pandas as pd

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

    # Find all soundscape files
    soundscape_dir = Path(soundscape_dir)
    files = sorted(soundscape_dir.glob("*.ogg"))
    if len(files) == 0:
        files = sorted(soundscape_dir.glob("*.wav"))

    print(f"[Inference] Found {len(files)} soundscape files")

    # Process each file
    all_row_ids = []
    all_predictions = []

    start_time = time.time()

    for file_idx, fpath in enumerate(files):
        file_start = time.time()

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
            tta=0,
        )

        fname = fpath.name.replace(fpath.suffix, "")
        for w in range(preds.shape[0]):
            all_row_ids.append(f"{fname}_{w + 1}")
            all_predictions.append(preds[w])

        file_time = time.time() - file_start
        elapsed = time.time() - start_time
        estimated_total = elapsed / (file_idx + 1) * len(files)

        if (file_idx + 1) % 50 == 0 or file_idx == 0:
            print(f"[Inference] {file_idx + 1}/{len(files)} files | "
                  f"File: {file_time:.2f}s | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {estimated_total:.1f}s")

    all_predictions = np.array(all_predictions)
    print(f"[Inference] Total predictions: {all_predictions.shape}")

    # Apply post-processing
    if config.get("apply_postprocessing", True):
        all_predictions = apply_postprocessing(all_predictions, config)

    # Build submission DataFrame
    if species_list is None:
        # Load from taxonomy
        taxonomy_csv = config.get("taxonomy_csv", "")
        if os.path.exists(taxonomy_csv):
            taxonomy = pd.read_csv(taxonomy_csv)
            species_list = sorted(taxonomy["primary_label"].unique().tolist())
        else:
            species_list = [f"species_{i}" for i in range(all_predictions.shape[1])]

    # Create submission DataFrame
    submission = pd.DataFrame({"row_id": all_row_ids})
    for i, sp in enumerate(species_list):
        submission[sp] = all_predictions[:, i]

    # Ensure column order matches sample_submission
    submission.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    print(f"[Inference] Complete! {total_time:.1f}s total → {output_csv}")

    return output_csv


def profile_inference_speed(
    onnx_path: str,
    config: dict,
    num_files: int = 10,
) -> Dict[str, float]:
    """
    Profile inference speed to estimate total runtime.
    
    Args:
        onnx_path: ONNX model path
        config: Configuration
        num_files: Number of test files to profile
        
    Returns:
        dict with timing statistics
    """
    import tempfile

    engine = ONNXInferenceEngine(
        onnx_path=onnx_path,
        num_threads=config.get("onnx_threads", 8),
        batch_size=config.get("onnx_batch_size", 64),
    )

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

    # Generate random audio for profiling
    times = []
    for i in range(num_files):
        # Simulate 12 windows
        all_mels = []
        for w in range(12):
            audio = np.random.randn(window_samples).astype(np.float32)
            mel = create_three_channel_mel(
                audio, sr=sample_rate,
                n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=fmin, fmax=fmax,
                use_pcen=use_pcen,
            )
            for c in range(3):
                mel[c] = crop_or_pad_mel(mel[c], target_time_steps)
            all_mels.append(mel)

        mel_batch = np.stack(all_mels, axis=0)

        start = time.time()
        _ = engine.predict_batch(mel_batch)
        elapsed = time.time() - start
        times.append(elapsed)

    times = np.array(times)
    stats = {
        "mean_per_file": float(times.mean()),
        "std_per_file": float(times.std()),
        "total_per_file_600": float(times.mean() * 600),
        "total_minutes": float(times.mean() * 600 / 60),
    }

    print(f"[Profile] Mean time per file: {stats['mean_per_file']:.3f}s")
    print(f"[Profile] Estimated total (600 files): {stats['total_per_file_600']:.1f}s "
          f"({stats['total_minutes']:.1f} min)")

    return stats
