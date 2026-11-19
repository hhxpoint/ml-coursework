# %% [markdown]
# # BirdCLEF 2026 — Training Notebook (Kaggle GPU)
#
# **Model:** EfficientNet-B0 + GEM Pooling (6.2M params)
# **Loss:** ASL (γ−=4, γ+=0) + Dual Loss
# **Data:** train_audio + soundscapes + pseudo labels
#
# **Output:** best_model.pt → upload as Dataset → inference notebook

# %% [code]
# ============================================================
# Install Dependencies (fix P100 compatibility)
# ============================================================
!pip install -q timm librosa soundfile scikit-learn

# %% [code]
# ============================================================
# Imports + Auto-detect Device & Paths
# ============================================================
import os
import ast
import json
import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# AMP: handle different PyTorch versions
def get_amp_context(device):
    """Get autocast context compatible with different PyTorch versions."""
    try:
        from torch.amp import autocast
        device_type = "cuda" if device.type == "cuda" else "cpu"
        return autocast(device_type=device_type, enabled=(device.type == "cuda"))
    except (ImportError, TypeError):
        from torch.cuda.amp import autocast
        return autocast(enabled=(device.type == "cuda"))

def get_grad_scaler(enabled=True):
    """Get GradScaler compatible with different PyTorch versions."""
    try:
        from torch.amp import GradScaler
        return GradScaler(enabled=enabled)
    except ImportError:
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import librosa
import timm

# ============================================================
# USER CONFIG: Set your local data path here
# ============================================================
LOCAL_DATA_DIR = r"D:\C++\birdclef2026\data"  # BirdCLEF 2026 data directory

# ============================================================
# Auto-detect GPU / CPU (fix P100 compatibility)
# ============================================================
USE_AMP = False
DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU detected: {gpu_name} (compute capability {capability})")
    try:
        # Test if GPU actually works
        x = torch.randn(2, 2).cuda()
        y = x @ x
        DEVICE = torch.device("cuda")
        USE_AMP = True
        print(f"✓ GPU working: {gpu_name}")
    except Exception as e:
        print(f"✗ GPU not compatible ({e}), falling back to CPU")
        DEVICE = torch.device("cpu")
else:
    print("No GPU available, using CPU")

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ============================================================
# Auto-detect Kaggle data paths
# ============================================================
def find_data_paths():
    """Auto-detect correct data paths on Kaggle or local."""
    
    # Check LOCAL_DATA_DIR first
    if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
        data_dir = LOCAL_DATA_DIR
        print(f"Using LOCAL_DATA_DIR: {data_dir}")
    else:
        # Kaggle detection
        base = "/kaggle/input"
        if not os.path.exists(base):
            # Local mode: check current directory structure
            print("Not on Kaggle. Set LOCAL_DATA_DIR to your data path.")
            return None
            
        # Find birdclef dataset directory
        candidates = [
            "birdclef-2026",
            "birdclef2026", 
            "birdclef_2026",
        ]
        
        data_dir = None
        for c in candidates:
            path = os.path.join(base, c)
            if os.path.exists(path):
                data_dir = path
                break
        
        # If not found, search
        if data_dir is None:
            try:
                for d in os.listdir(base):
                    if "birdclef" in d.lower() and "2026" in d:
                        data_dir = os.path.join(base, d)
                        break
            except:
                pass
        
        if data_dir is None:
            print(f"WARNING: No birdclef dataset found")
            if os.path.exists(base):
                print(f"Available in {base}: {os.listdir(base)}")
            return None
    
    print(f"Found data: {data_dir}")
    
    # Find specific files inside
    def find_file(name):
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f == name or f.replace(".csv", "") == name.replace(".csv", ""):
                    return os.path.join(root, f)
        return None
    
    def find_dir(name):
        for d in os.listdir(data_dir):
            if d.lower() == name.lower():
                return os.path.join(data_dir, d)
        return None
    
    paths = {
        "data_dir": data_dir,
        "train_csv": find_file("train.csv") or os.path.join(data_dir, "train.csv"),
        "taxonomy_csv": find_file("taxonomy.csv") or os.path.join(data_dir, "taxonomy.csv"),
        "soundscape_labels_csv": find_file("train_soundscape_labels.csv") or os.path.join(data_dir, "train_soundscape_labels.csv"),
        "train_audio_dir": find_dir("train_audio") or os.path.join(data_dir, "train_audio"),
        "train_soundscape_dir": find_dir("train_soundscapes") or os.path.join(data_dir, "train_soundscapes"),
        "test_soundscape_dir": find_dir("test_soundscapes") or os.path.join(data_dir, "test_soundscapes"),
    }
    
    # Verify and print
    for key, path in paths.items():
        exists = os.path.exists(path)
        print(f"  {key}: {path} {'✓' if exists else '✗'}")
    
    return paths

DATA_PATHS = find_data_paths()

# %% [code]
# ============================================================
# Configuration (v28 Kitchen-Sink Best Formula)
# ============================================================

CONFIG = {
    # --- Paths (auto-detected) ---
    "data_dir": DATA_PATHS["data_dir"] if DATA_PATHS else "/kaggle/input/birdclef-2026",
    "train_audio_dir": DATA_PATHS["train_audio_dir"] if DATA_PATHS else "/kaggle/input/birdclef-2026/train_audio",
    "train_soundscape_dir": DATA_PATHS["train_soundscape_dir"] if DATA_PATHS else "/kaggle/input/birdclef-2026/train_soundscapes",
    "train_csv": DATA_PATHS["train_csv"] if DATA_PATHS else "/kaggle/input/birdclef-2026/train.csv",
    "taxonomy_csv": DATA_PATHS["taxonomy_csv"] if DATA_PATHS else "/kaggle/input/birdclef-2026/taxonomy.csv",
    "soundscape_labels_csv": DATA_PATHS["soundscape_labels_csv"] if DATA_PATHS else "/kaggle/input/birdclef-2026/train_soundscape_labels.csv",
    # Pseudo labels (upload as separate dataset if available)
    "soft_pseudo_csv": None,
    "hard_pseudo_csv": None,
    "combined_pseudo_csv": None,
    "output_dir": "/kaggle/working/output",
    # Auto-detected device
    "device": str(DEVICE),
    "use_amp": USE_AMP,

    # --- Audio ---
    "sample_rate": 32000,
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 128,
    "fmin": 50,
    "fmax": 16000,
    "train_duration": 10.0,
    "infer_duration": 5.0,
    "use_pcen": True,

    # --- Model ---
    "backbone": "tf_efficientnet_b0.ns_jft_in1k",
    "in_channels": 3,
    "num_classes": 234,
    "dropout": 0.1,
    "gem_pool": True,
    "gem_p_init": 3.0,
    "dual_loss": True,
    "dual_clip_weight": 0.5,
    "dual_frame_weight": 0.5,

    # --- Training ---
    "seed": 42,
    "num_workers": 2,
    "epochs": 35,
    "batch_size": 32,
    "lr": 5.0e-4,
    "warmup_epochs": 3,
    "patience": 7,
    "weight_decay": 1e-5,

    # --- ASL Loss ---
    "asl_gamma_neg": 4.0,
    "asl_gamma_pos": 0.0,
    "asl_clip": 0.05,

    # --- Augmentation ---
    "mixup_alpha": 0.5,
    "circ_shift_prob": 0.5,
    "specaugment_time_masks": 2,
    "specaugment_time_mask_param": 0.15,
    "specaugment_freq_masks": 2,
    "specaugment_freq_mask_param": 40,
    "noise_level": 0.005,
    "gain_min": 0.7,
    "gain_max": 1.3,

    # --- Data ---
    "clips_per_file": 3,
    "soundscape_oversample": 5,
    "soft_pseudo_oversample": 10,
    "hard_pseudo_oversample": 5,
    "combined_pseudo_oversample": 10,
}

# Ensure output dir exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("Config loaded!")

# %% [code]
# ============================================================
# Audio Feature Extraction
# ============================================================

def load_audio(path, sr=32000, duration=None, offset=0.0, mono=True):
    audio, _ = librosa.load(path, sr=sr, duration=duration, offset=offset, mono=mono)
    return audio

def audio_to_mel(audio, sr=32000, n_fft=1024, hop_length=320, n_mels=128, fmin=50, fmax=16000, use_pcen=True):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    if use_pcen:
        mel = librosa.pcen(mel * (2**31), sr=sr, hop_length=hop_length)
    else:
        mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return mel

def crop_or_pad_mel(mel, target_time_steps):
    n_mels, current_steps = mel.shape
    if current_steps > target_time_steps:
        start = np.random.randint(0, current_steps - target_time_steps + 1)
        mel = mel[:, start:start + target_time_steps]
    elif current_steps < target_time_steps:
        pad_left = (target_time_steps - current_steps) // 2
        pad_right = target_time_steps - current_steps - pad_left
        mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
    return mel

def create_three_channel_mel(audio, sr=32000, n_fft=1024, hop_length=320, n_mels=128, fmin=50, fmax=16000, use_pcen=True):
    mel_0 = audio_to_mel(audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen)
    mel_1 = audio_to_mel(audio, sr, n_fft=n_fft*2, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen)
    mel_2 = audio_to_mel(audio, sr, n_fft=n_fft//2, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen)
    min_time = min(mel_0.shape[1], mel_1.shape[1], mel_2.shape[1])
    mel_0, mel_1, mel_2 = mel_0[:, :min_time], mel_1[:, :min_time], mel_2[:, :min_time]
    return np.stack([mel_0, mel_1, mel_2], axis=0)

def pad_or_crop_waveform(waveform, target_samples):
    current = len(waveform)
    if current > target_samples:
        start = np.random.randint(0, current - target_samples + 1)
        waveform = waveform[start:start + target_samples]
    elif current < target_samples:
        pad_left = (target_samples - current) // 2
        pad_right = target_samples - current - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")
    return waveform

class AudioAugmentation:
    def __init__(self, spec_time_masks=2, spec_time_mask_param=0.15, spec_freq_masks=2,
                 spec_freq_mask_param=40, noise_level=0.005, gain_min=0.7, gain_max=1.3, circ_shift_prob=0.5):
        self.spec_time_masks = spec_time_masks
        self.spec_time_mask_param = spec_time_mask_param
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_mask_param = spec_freq_mask_param
        self.noise_level = noise_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.circ_shift_prob = circ_shift_prob

    def spec_augment(self, mel):
        mel = mel.copy()
        n_mels, n_time = mel.shape
        for _ in range(self.spec_time_masks):
            max_mask = int(n_time * self.spec_time_mask_param)
            if max_mask > 0:
                mask_len = np.random.randint(0, max_mask + 1)
                mask_start = np.random.randint(0, n_time - mask_len + 1)
                mel[:, mask_start:mask_start + mask_len] = 0
        for _ in range(self.spec_freq_masks):
            max_mask = min(self.spec_freq_mask_param, n_mels)
            if max_mask > 0:
                mask_len = np.random.randint(0, max_mask + 1)
                mask_start = np.random.randint(0, n_mels - mask_len + 1)
                mel[mask_start:mask_start + mask_len, :] = 0
        return mel

    def circular_shift(self, audio):
        if np.random.random() < self.circ_shift_prob:
            audio = np.roll(audio, np.random.randint(0, len(audio)))
        return audio

    def add_noise(self, audio):
        if self.noise_level > 0:
            audio = audio + np.random.randn(*audio.shape).astype(np.float32) * self.noise_level
        return audio

    def apply_gain(self, audio):
        return audio * np.random.uniform(self.gain_min, self.gain_max)

    def __call__(self, mel):
        return self.spec_augment(mel)

print("Feature extraction ready!")

# %% [code]
# ============================================================
# Dataset
# ============================================================

class BirdCLEFDataset(Dataset):
    def __init__(self, train_csv, audio_dir, taxonomy_csv,
                 soundscape_labels_csv=None, soft_pseudo_csv=None,
                 hard_pseudo_csv=None, combined_pseudo_csv=None,
                 soundscape_dir=None, sample_rate=32000, duration=10.0,
                 n_fft=1024, hop_length=320, n_mels=128, fmin=50, fmax=16000,
                 use_pcen=True, clips_per_file=3, soundscape_oversample=5,
                 soft_pseudo_oversample=10, hard_pseudo_oversample=5,
                 combined_pseudo_oversample=10, augment=True, mode="train",
                 fold=None, fold_csv=None):

        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.use_pcen = use_pcen
        self.clips_per_file = clips_per_file
        self.augment = augment
        self.mode = mode
        self.augmenter = AudioAugmentation()

        taxonomy = pd.read_csv(taxonomy_csv)
        self.species_list = sorted(taxonomy["primary_label"].unique().tolist())
        self.num_classes = len(self.species_list)
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}

        self.samples, self.labels, self.sources = [], [], []

        # Load training audio
        train_df = pd.read_csv(train_csv)
        if fold is not None and fold_csv is not None:
            fold_df = pd.read_csv(fold_csv)
            if "fold" in fold_df.columns:
                val_files = fold_df[fold_df["fold"] == fold]["filename"].tolist()
                if mode == "train":
                    train_df = train_df[~train_df["filename"].isin(val_files)]
                else:
                    train_df = train_df[train_df["filename"].isin(val_files)]

        for _, row in train_df.iterrows():
            filename = row["filename"]
            primary_label = str(row["primary_label"])
            secondary_labels = str(row.get("secondary_labels", "[]"))
            if primary_label not in self.species_to_idx:
                continue
            audio_path = self.audio_dir / filename
            if not audio_path.exists():
                continue

            label = np.zeros(self.num_classes, dtype=np.float32)
            label[self.species_to_idx[primary_label]] = 1.0
            try:
                secondary = ast.literal_eval(secondary_labels)
                if isinstance(secondary, list):
                    for s in secondary:
                        s = str(s)
                        if s in self.species_to_idx:
                            label[self.species_to_idx[s]] = 1.0
            except:
                pass

            for _ in range(clips_per_file):
                self.samples.append({"path": str(audio_path), "source": "labeled", "offset": 0.0, "filename": filename})
                self.labels.append(label)
                self.sources.append("labeled")

        # Load soundscape labels
        if soundscape_labels_csv and os.path.exists(soundscape_labels_csv):
            sc_df = pd.read_csv(soundscape_labels_csv)
            sc_dir = Path(soundscape_dir) if soundscape_dir else self.audio_dir
            for _, row in sc_df.iterrows():
                filename = row.get("filename", "")
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        if float(row[sp]) > 0:
                            label[self.species_to_idx[sp]] = 1.0
                audio_path = sc_dir / filename
                if not audio_path.exists():
                    continue
                for _ in range(soundscape_oversample):
                    self.samples.append({"path": str(audio_path), "source": "soundscape", "offset": 0.0, "filename": filename})
                    self.labels.append(label)
                    self.sources.append("soundscape")

        # Load soft pseudo labels
        if soft_pseudo_csv and os.path.exists(soft_pseudo_csv):
            soft_df = pd.read_csv(soft_pseudo_csv)
            for _, row in soft_df.iterrows():
                filename = row.get("filename", "")
                audio_path = self.audio_dir / filename
                if not audio_path.exists():
                    continue
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        val = float(row[sp])
                        if val > 0:
                            label[self.species_to_idx[sp]] = val
                for _ in range(soft_pseudo_oversample):
                    self.samples.append({"path": str(audio_path), "source": "soft_pseudo", "offset": 0.0, "filename": filename})
                    self.labels.append(label)
                    self.sources.append("soft_pseudo")

        # Load hard pseudo labels
        if hard_pseudo_csv and os.path.exists(hard_pseudo_csv):
            hard_df = pd.read_csv(hard_pseudo_csv)
            for _, row in hard_df.iterrows():
                filename = row.get("filename", "")
                audio_path = self.audio_dir / filename
                if not audio_path.exists():
                    continue
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        if float(row[sp]) > 0:
                            label[self.species_to_idx[sp]] = 1.0
                for _ in range(hard_pseudo_oversample):
                    self.samples.append({"path": str(audio_path), "source": "hard_pseudo", "offset": 0.0, "filename": filename})
                    self.labels.append(label)
                    self.sources.append("hard_pseudo")

        # Load combined pseudo labels
        if combined_pseudo_csv and os.path.exists(combined_pseudo_csv):
            comb_df = pd.read_csv(combined_pseudo_csv)
            for _, row in comb_df.iterrows():
                filename = row.get("filename", "")
                audio_path = self.audio_dir / filename
                if not audio_path.exists():
                    continue
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        val = float(row[sp])
                        if val > 0:
                            label[self.species_to_idx[sp]] = val
                for _ in range(combined_pseudo_oversample):
                    self.samples.append({"path": str(audio_path), "source": "combined_pseudo", "offset": 0.0, "filename": filename})
                    self.labels.append(label)
                    self.sources.append("combined_pseudo")

        print(f"[Dataset] Mode={mode}, Samples={len(self.samples)}, Classes={self.num_classes}")
        src_cnt = {}
        for s in self.sources:
            src_cnt[s] = src_cnt.get(s, 0) + 1
        print(f"[Dataset] Sources: {src_cnt}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx].copy()
        audio, _ = load_audio(sample["path"], sr=self.sample_rate, duration=None, offset=sample.get("offset", 0.0))

        if self.augment and self.mode == "train":
            audio = self.augmenter.circular_shift(audio)
            audio = self.augmenter.add_noise(audio)
            audio = self.augmenter.apply_gain(audio)

        audio = pad_or_crop_waveform(audio, self.target_samples)
        mel = create_three_channel_mel(audio, sr=self.sample_rate, n_fft=self.n_fft,
                                        hop_length=self.hop_length, n_mels=self.n_mels,
                                        fmin=self.fmin, fmax=self.fmax, use_pcen=self.use_pcen)

        target_time_steps = int(self.sample_rate * self.duration / self.hop_length) + 1
        for c in range(3):
            mel[c] = crop_or_pad_mel(mel[c], target_time_steps)

        if self.augment and self.mode == "train":
            for c in range(3):
                mel[c] = self.augmenter(mel[c])

        return {
            "mel": torch.from_numpy(mel).float(),
            "label": torch.from_numpy(label).float(),
            "filename": sample.get("filename", ""),
            "source": sample.get("source", "unknown"),
        }

    def get_sample_weights(self):
        weights = np.ones(len(self.samples), dtype=np.float64)
        label_counts = np.zeros(self.num_classes, dtype=np.float64)
        for label in self.labels:
            label_counts += label
        class_weights = len(self.labels) / (self.num_classes * label_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        for i, label in enumerate(self.labels):
            active = label > 0
            if active.any():
                weights[i] = class_weights[active].mean()
        return weights

print("Dataset ready!")

# %% [code]
# ============================================================
# Model: EfficientNet-B0 + GEM Pooling
# ============================================================

class GeMPooling(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class BirdCLEFModel(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b0.ns_jft_in1k", num_classes=234,
                 in_channels=3, dropout=0.1, gem_pool=True, gem_p_init=3.0,
                 pretrained=True, dual_head=True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dual_head = dual_head
        self.gem_pool = gem_pool

        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_channels,
                                           num_classes=0, global_pool="")

        dummy = torch.randn(1, in_channels, 128, 313)
        with torch.no_grad():
            features = self.backbone(dummy)
            self.feature_channels = features.shape[1] if features.dim() == 4 else features.shape[-1]

        self.pool = GeMPooling(p_init=gem_p_init) if gem_pool else nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_channels, num_classes)
        if dual_head:
            self.frame_classifier = nn.Linear(self.feature_channels, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if features.dim() == 4:
            pooled = self.pool(features)
            if self.dual_head:
                frame_features = features.mean(dim=2).permute(0, 2, 1)
                frame_logits = self.frame_classifier(frame_features).mean(dim=1)
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


def build_model(config):
    return BirdCLEFModel(
        backbone=config["backbone"], num_classes=config["num_classes"],
        in_channels=config["in_channels"], dropout=config["dropout"],
        gem_pool=config.get("gem_pool", True), gem_p_init=config.get("gem_p_init", 3.0),
        pretrained=True, dual_head=config.get("dual_loss", True),
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model ready!")

# %% [code]
# ============================================================
# Loss Functions: ASL + Dual Loss
# ============================================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_neg = (probs - self.clip).clamp(min=0) if self.clip > 0 else probs
        log_p = torch.log(probs.clamp(min=self.eps))
        log_1mp = torch.log((1 - probs_neg).clamp(min=self.eps))
        loss_pos = targets * log_p
        loss_neg = (1 - targets) * log_1mp
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            modulator_pos = (1 - probs).pow(self.gamma_pos)
            modulator_neg = (1 - (1 - probs_neg)).pow(self.gamma_neg)
            loss_pos = modulator_pos * loss_pos
            loss_neg = modulator_neg * loss_neg
        return -(loss_pos + loss_neg).mean()


class DualLoss(nn.Module):
    def __init__(self, clip_criterion, frame_criterion, clip_weight=0.5, frame_weight=0.5):
        super().__init__()
        self.clip_criterion = clip_criterion
        self.frame_criterion = frame_criterion
        self.clip_weight = clip_weight
        self.frame_weight = frame_weight

    def forward(self, clip_logits, frame_logits, targets):
        clip_loss = self.clip_criterion(clip_logits, targets)
        frame_loss = self.frame_criterion(frame_logits, targets)
        total_loss = self.clip_weight * clip_loss + self.frame_weight * frame_loss
        return {"loss": total_loss, "clip_loss": clip_loss.item(), "frame_loss": frame_loss.item()}


def build_criterion(config):
    asl = AsymmetricLoss(gamma_neg=config.get("asl_gamma_neg", 4.0),
                          gamma_pos=config.get("asl_gamma_pos", 0.0),
                          clip=config.get("asl_clip", 0.05))
    if config.get("dual_loss", True):
        frame_asl = AsymmetricLoss(gamma_neg=config.get("asl_gamma_neg", 4.0),
                                    gamma_pos=config.get("asl_gamma_pos", 0.0),
                                    clip=config.get("asl_clip", 0.05))
        return DualLoss(asl, frame_asl, config.get("dual_clip_weight", 0.5), config.get("dual_frame_weight", 0.5))
    return asl

print("Loss functions ready!")

# %% [code]
# ============================================================
# Training Loop
# ============================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val, self.sum, self.count = val, self.sum + val * n, self.count + n
        self.avg = self.sum / self.count

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def batch_mixup(mel, labels, alpha=0.5):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(mel.size(0), device=mel.device)
    return lam * mel + (1 - lam) * mel[index], lam * labels + (1 - lam) * labels[index], lam

def compute_val_auc(model, val_loader, device, num_classes=234):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            outputs = model(mel)
            probs = torch.sigmoid(outputs["clip_logits"])
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc_scores = []
    for c in range(num_classes):
        if all_targets[:, c].sum() > 0:
            try:
                auc_scores.append(roc_auc_score(all_targets[:, c], all_preds[:, c]))
            except:
                pass
    return np.mean(auc_scores) if auc_scores else 0.0

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    use_amp = config.get("use_amp", True)
    mixup_alpha = config.get("mixup_alpha", 0.5)

    for step, batch in enumerate(train_loader):
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)

        if mixup_alpha > 0 and np.random.random() < 0.5:
            mel, labels, _ = batch_mixup(mel, labels, alpha=mixup_alpha)

        optimizer.zero_grad()
        if use_amp and scaler:
            with get_amp_context(device):
                outputs = model(mel)
                if isinstance(criterion, DualLoss) and "frame_logits" in outputs:
                    result = criterion(outputs["clip_logits"], outputs["frame_logits"], labels)
                    loss = result["loss"]
                else:
                    loss = criterion(outputs["clip_logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mel)
            if isinstance(criterion, DualLoss) and "frame_logits" in outputs:
                result = criterion(outputs["clip_logits"], outputs["frame_logits"], labels)
                loss = result["loss"]
            else:
                loss = criterion(outputs["clip_logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_meter.update(loss.item(), mel.size(0))
        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch} Step {step}/{len(train_loader)} | Loss: {loss_meter.avg:.4f} | LR: {lr:.2e}")

    return {"train_loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]}

print("Training loop ready!")

# %% [code]
# ============================================================
# Main Training Function
# ============================================================

def train_model(config, fold=None, fold_csv=None):
    device = DEVICE  # Use auto-detected device
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Config: AMP + Device
    config["device"] = str(DEVICE)
    config["use_amp"] = USE_AMP
    print(f"[Trainer] Device: {DEVICE}, AMP: {USE_AMP}")

    # Datasets
    train_dataset = BirdCLEFDataset(
        train_csv=config["train_csv"], audio_dir=config["train_audio_dir"],
        taxonomy_csv=config["taxonomy_csv"],
        soundscape_labels_csv=config.get("soundscape_labels_csv"),
        soft_pseudo_csv=config.get("soft_pseudo_csv"),
        hard_pseudo_csv=config.get("hard_pseudo_csv"),
        combined_pseudo_csv=config.get("combined_pseudo_csv"),
        soundscape_dir=config.get("train_soundscape_dir"),
        sample_rate=config["sample_rate"], duration=config["train_duration"],
        n_fft=config["n_fft"], hop_length=config["hop_length"],
        n_mels=config["n_mels"], fmin=config["fmin"], fmax=config["fmax"],
        use_pcen=config["use_pcen"], clips_per_file=config["clips_per_file"],
        soundscape_oversample=config["soundscape_oversample"],
        soft_pseudo_oversample=config["soft_pseudo_oversample"],
        hard_pseudo_oversample=config["hard_pseudo_oversample"],
        combined_pseudo_oversample=config.get("combined_pseudo_oversample", 10),
        augment=True, mode="train", fold=fold, fold_csv=fold_csv,
    )

    val_dataset = BirdCLEFDataset(
        train_csv=config["train_csv"], audio_dir=config["train_audio_dir"],
        taxonomy_csv=config["taxonomy_csv"],
        soundscape_labels_csv=config.get("soundscape_labels_csv"),
        sample_rate=config["sample_rate"], duration=config["train_duration"],
        n_fft=config["n_fft"], hop_length=config["hop_length"],
        n_mels=config["n_mels"], fmin=config["fmin"], fmax=config["fmax"],
        use_pcen=config["use_pcen"], clips_per_file=1, soundscape_oversample=1,
        augment=False, mode="val", fold=fold, fold_csv=fold_csv,
    )

    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler,
                               num_workers=config.get("num_workers", 2), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config.get("num_workers", 2), pin_memory=True)

    # Model
    model = build_model(config).to(device)
    print(f"[Trainer] Parameters: {count_parameters(model):,}")

    criterion = build_criterion(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-5))

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config.get("warmup_epochs", 3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = get_grad_scaler(enabled=config.get("use_amp", True))

    best_auc, best_epoch, patience_counter = 0.0, 0, 0
    patience = config.get("patience", 7)
    history = []

    print(f"[Trainer] Starting {config['epochs']} epochs...")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler)
        scheduler.step()
        val_auc = compute_val_auc(model, val_loader, device, config["num_classes"])
        epoch_time = time.time() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_metrics["train_loss"], "val_auc": val_auc,
                        "lr": train_metrics["lr"], "time": epoch_time})

        print(f"Epoch {epoch}/{config['epochs']} | Loss: {train_metrics['train_loss']:.4f} | "
              f"Val AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s")

        if val_auc > best_auc:
            best_auc, best_epoch, patience_counter = val_auc, epoch, 0
            ckpt_name = "best_model" + (f"_fold{fold}" if fold is not None else "") + ".pt"
            ckpt_path = output_dir / ckpt_name
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(), "val_auc": val_auc, "config": config}, ckpt_path)
            print(f"  ✓ New best AUC: {val_auc:.4f} → {ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_auc:.4f} @ epoch {best_epoch})")
            break

    # Save history
    history_path = output_dir / ("history" + (f"_fold{fold}" if fold is not None else "") + ".json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Trainer] Done! Best AUC: {best_auc:.4f} @ epoch {best_epoch}")
    return str(output_dir / ("best_model" + (f"_fold{fold}" if fold is not None else "") + ".pt"))

# %% [code]
# ============================================================
# Model Soup
# ============================================================

def model_soup(checkpoints, config, weights=None):
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
            avg_state[key] = state[key] * w if key not in avg_state else avg_state[key] + state[key] * w
    model.load_state_dict(avg_state)
    return model

# %% [code]
# ============================================================
# ONNX Export
# ============================================================

def export_to_onnx(model, save_path, input_shape=(1, 3, 128, 313), opset_version=13):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    torch.onnx.export(model, dummy_input, save_path, opset_version=opset_version,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size", 3: "time_steps"}, "output": {0: "batch_size"}},
                      do_constant_folding=True)
    print(f"[ONNX] Exported: {save_path}")

# %% [code]
# ============================================================
# Create K-Fold Splits
# ============================================================

def create_fold_csv(train_csv, taxonomy_csv, output_path, n_folds=4, seed=42):
    np.random.seed(seed)
    train_df = pd.read_csv(train_csv)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = train_df["primary_label"].values
    folds = np.zeros(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, labels)):
        folds[val_idx] = fold_idx
    train_df["fold"] = folds
    train_df[["filename", "primary_label", "fold"]].to_csv(output_path, index=False)
    print(f"[Folds] Created {n_folds}-fold split → {output_path}")
    for f in range(n_folds):
        print(f"  Fold {f}: {(train_df['fold'] == f).sum()} samples")
    return output_path

# %% [code]
# ============================================================
# RUN TRAINING
# ============================================================

# === Option 1: Single model (no fold) ===
print("=" * 60)
print("  BirdCLEF 2026 Training — Single Model")
print("=" * 60)
best_path = train_model(CONFIG)
print(f"\nBest model saved: {best_path}")

# === Option 2: 4-Fold + Model Soup (uncomment to use) ===
# print("=" * 60)
# print("  BirdCLEF 2026 Training — 4-Fold")
# print("=" * 60)
# fold_csv = os.path.join(CONFIG["output_dir"], "folds_4fold.csv")
# create_fold_csv(CONFIG["train_csv"], CONFIG["taxonomy_csv"], fold_csv, n_folds=4)
#
# checkpoints = []
# for fold in range(4):
#     print(f"\n{'='*40} Fold {fold} {'='*40}")
#     ckpt = train_model(CONFIG, fold=fold, fold_csv=fold_csv)
#     checkpoints.append(ckpt)
#
# # Model soup
# soup = model_soup(checkpoints, CONFIG, weights=[1.5]*4)
# soup_path = os.path.join(CONFIG["output_dir"], "model_soup_4fold.pt")
# torch.save({"model_state_dict": soup.state_dict(), "config": CONFIG}, soup_path)
# print(f"Model soup saved: {soup_path}")

# %% [code]
# ============================================================
# Export to ONNX
# ============================================================

print("\nExporting to ONNX...")
model = build_model(CONFIG)
ckpt = torch.load(best_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

onnx_path = os.path.join(CONFIG["output_dir"], "best_model.onnx")
export_to_onnx(model, onnx_path)

print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"  Model: {best_path}")
print(f"  ONNX:  {onnx_path}")
print(f"\n  Next steps:")
print(f"  1. Download model from Kaggle output")
print(f"  2. Upload as Kaggle Dataset")
print(f"  3. Use inference notebook for submission")
print("=" * 60)
