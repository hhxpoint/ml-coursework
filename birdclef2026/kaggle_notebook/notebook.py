# ============================================================
# BirdCLEF 2026 — Kaggle Submission Notebook
# 
# This notebook is designed for Kaggle's CPU-only inference:
# - 90 minutes max runtime
# - No GPU, no internet
# - All models pre-packaged
# ============================================================

# %% [markdown]
# # BirdCLEF 2026 — Submission Notebook
# 
# **Model:** EfficientNet-B0 + GEM Pooling (6.2M params)
# **Inference:** ONNX Runtime (CPU-optimized)
# **Post-processing:** lmax_pre_aves → lscale → SoftmaxRichness

# %% [code]
# Install dependencies (pre-installed on Kaggle)
# !pip install onnxruntime librosa soundfile

import os
import time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

print("Libraries loaded!")

# %% [code]
# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 32000
DURATION = 5.0  # seconds per window
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 50
FMAX = 16000
NUM_CLASSES = 234
ONNX_THREADS = 8
BATCH_SIZE = 64

# Paths (Kaggle)
MODEL_DIR = "/kaggle/input/birdclef2026-model"  # Your uploaded dataset
TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes"
TAXONOMY_CSV = "/kaggle/input/birdclef-2026/taxonomy.csv"

# Post-processing params (from discovery report R52)
LMAX_ALPHA = 0.1
LMAX_RADIUS = 1
LMAX_AVES_START = 72  # Aves classes start at index 72
LSCALE = 0.9
RICH_TEMP = 0.15
RICH_CP_BLEND = 0.60

# %% [code]
# ============================================================
# Audio Feature Extraction
# ============================================================

def load_audio(path, sr=SAMPLE_RATE, duration=DURATION, offset=0.0):
    """Load audio file."""
    audio, _ = librosa.load(path, sr=sr, duration=duration, offset=offset, mono=True)
    return audio


def audio_to_mel(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                  n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Convert audio to mel-spectrogram with PCEN normalization."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    # PCEN normalization (critical!)
    mel = librosa.pcen(mel * (2**31), sr=sr, hop_length=hop_length)
    return mel


def create_three_channel_mel(audio, sr=SAMPLE_RATE):
    """Create 3-channel mel-spectrogram at different frequency resolutions."""
    mel_0 = audio_to_mel(audio, sr, n_fft=N_FFT)
    mel_1 = audio_to_mel(audio, sr, n_fft=N_FFT * 2)
    mel_2 = audio_to_mel(audio, sr, n_fft=N_FFT // 2)

    # Align time dimensions
    min_time = min(mel_0.shape[1], mel_1.shape[1], mel_2.shape[1])
    mel_0, mel_1, mel_2 = mel_0[:, :min_time], mel_1[:, :min_time], mel_2[:, :min_time]

    return np.stack([mel_0, mel_1, mel_2], axis=0)  # (3, n_mels, time)


def crop_or_pad_mel(mel, target_steps):
    """Crop or pad mel-spectrogram to fixed time steps."""
    _, current = mel.shape
    if current > target_steps:
        start = (current - target_steps) // 2
        mel = mel[:, start:start + target_steps]
    elif current < target_steps:
        pad_left = (target_steps - current) // 2
        pad_right = target_steps - current - pad_left
        mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant")
    return mel


# %% [code]
# ============================================================
# ONNX Inference Engine
# ============================================================

class ONNXEngine:
    """ONNX Runtime inference engine optimized for CPU."""
    
    def __init__(self, model_path, num_threads=ONNX_THREADS):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"ONNX model loaded: {model_path}")
    
    def predict(self, mel_batch):
        """Run inference on batch of mel spectrograms."""
        mel_batch = mel_batch.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: mel_batch})
        logits = outputs[0]
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid


# %% [code]
# ============================================================
# Post-Processing
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def lmax_pre_aves(probs, alpha=LMAX_ALPHA, radius=LMAX_RADIUS, aves_start=LMAX_AVES_START):
    """Local-max propagation in logit space (Aves classes only)."""
    logits = logit(probs)
    result = logits.copy()
    num_windows, num_classes = logits.shape
    
    for t in range(num_windows):
        for r in range(1, radius + 1):
            for dt in [-r, r]:
                nt = t + r if dt > 0 else t - r
                if 0 <= nt < num_windows:
                    local_max = np.maximum(logits[t, aves_start:], logits[nt, aves_start:])
                    result[t, aves_start:] += alpha * (local_max - logits[t, aves_start:])
    
    return sigmoid(result)

def logit_scale(probs, scale=LSCALE):
    """Logit space scale shrink."""
    return sigmoid(logit(probs) * scale)

def softmax_richness(probs, temperature=RICH_TEMP):
    """SoftmaxRichness normalization."""
    exp_p = np.exp(probs / temperature)
    softmax_p = exp_p / (exp_p.sum(axis=-1, keepdims=True) + 1e-10)
    richness = softmax_p.sum(axis=-1, keepdims=True)
    avg_richness = richness.mean()
    if avg_richness > 0:
        factor = avg_richness / (richness + 1e-6)
        return np.clip(probs * factor, 0, 1)
    return probs

def postprocess_file(predictions):
    """Apply full post-processing pipeline to one file's predictions."""
    # Step 1: lmax_pre_aves
    preds = lmax_pre_aves(predictions)
    # Step 2: logit scale
    preds = logit_scale(preds)
    # Step 3: SoftmaxRichness
    preds = softmax_richness(preds)
    return preds


# %% [code]
# ============================================================
# Load Models
# ============================================================

# Find ONNX model
onnx_files = list(Path(MODEL_DIR).glob("*.onnx"))
if not onnx_files:
    # Try to find in nested directories
    onnx_files = list(Path(MODEL_DIR).rglob("*.onnx"))

assert len(onnx_files) > 0, f"No ONNX models found in {MODEL_DIR}"
print(f"Found {len(onnx_files)} ONNX model(s)")

# Load engines
engines = []
for onnx_path in onnx_files:
    engine = ONNXEngine(str(onnx_path))
    engines.append(engine)

# Ensemble weights (equal by default)
ensemble_weights = np.ones(len(engines)) / len(engines)
print(f"Ensemble: {len(engines)} models, weights={ensemble_weights}")

# %% [code]
# ============================================================
# Load Taxonomy & Prepare Output
# ============================================================

taxonomy = pd.read_csv(TAXONOMY_CSV)
species_list = sorted(taxonomy["primary_label"].unique().tolist())
assert len(species_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} species, got {len(species_list)}"

print(f"Species: {NUM_CLASSES}")
print(f"Test dir: {TEST_DIR}")

# Find test soundscapes
test_files = sorted(Path(TEST_DIR).glob("*.ogg"))
if len(test_files) == 0:
    test_files = sorted(Path(TEST_DIR).glob("*.wav"))

print(f"Test files: {len(test_files)}")

# %% [code]
# ============================================================
# Main Inference Loop
# ============================================================

WINDOW_SAMPLES = int(SAMPLE_RATE * DURATION)
TARGET_TIME_STEPS = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1

all_row_ids = []
all_predictions = []

start_time = time.time()

for file_idx, fpath in enumerate(test_files):
    file_start = time.time()
    fname = fpath.stem
    
    # Load full audio
    full_audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
    num_windows = len(full_audio) // WINDOW_SAMPLES
    
    # Prepare all windows as mel spectrograms
    all_mels = []
    for w in range(num_windows):
        start_sample = w * WINDOW_SAMPLES
        end_sample = start_sample + WINDOW_SAMPLES
        window_audio = full_audio[start_sample:end_sample]
        
        # Ensure exact length
        if len(window_audio) < WINDOW_SAMPLES:
            window_audio = np.pad(window_audio, (0, WINDOW_SAMPLES - len(window_audio)))
        
        mel = create_three_channel_mel(window_audio, SAMPLE_RATE)
        mel = crop_or_pad_mel(mel, TARGET_TIME_STEPS)
        all_mels.append(mel)
    
    # Batch inference
    if all_mels:
        mel_batch = np.stack(all_mels, axis=0)  # (num_windows, 3, n_mels, time)
        
        # Get predictions from all engines
        file_preds = []
        for engine_idx, engine in enumerate(engines):
            preds = engine.predict(mel_batch)
            file_preds.append(preds)
        
        # Ensemble
        ensemble_preds = sum(w * p for w, p in zip(ensemble_weights, file_preds))
        
        # Post-process
        ensemble_preds = postprocess_file(ensemble_preds)
        
        # Collect results
        for w in range(num_windows):
            all_row_ids.append(f"{fname}_{w + 1}")
            all_predictions.append(ensemble_preds[w])
    
    # Progress
    file_time = time.time() - file_start
    elapsed = time.time() - start_time
    if (file_idx + 1) % 50 == 0 or file_idx == 0:
        print(f"  [{file_idx+1}/{len(test_files)}] {fname} | "
              f"file: {file_time:.2f}s | elapsed: {elapsed:.1f}s")

total_time = time.time() - start_time
print(f"\nInference complete! {total_time:.1f}s total")

# %% [code]
# ============================================================
# Generate Submission CSV
# ============================================================

all_predictions = np.array(all_predictions)
print(f"Predictions shape: {all_predictions.shape}")

submission = pd.DataFrame({"row_id": all_row_ids})
for i, sp in enumerate(species_list):
    submission[sp] = all_predictions[:, i]

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")
print(submission.head())

# Verify
assert "row_id" in submission.columns
assert len(species_list) == 234
assert submission.shape[0] > 0
print("\nSubmission verification passed!")
