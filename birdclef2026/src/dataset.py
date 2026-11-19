# ============================================================
# BirdCLEF 2026 — src/dataset.py
# Dataset & DataLoader for training, validation, and inference
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from .features import (
    load_audio,
    audio_to_mel,
    crop_or_pad_mel,
    create_three_channel_mel,
    pad_or_crop_waveform,
    AudioAugmentation,
    mixup,
)


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF 2026 Training Dataset.
    
    Combines:
    - Labeled training audio (train.csv + train_audio/)
    - Labeled soundscapes (train_soundscapes/)
    - Soft pseudo-labels (Perch teacher, r5 × 10 oversample)
    - Hard pseudo-labels (r5, extra_soundscape_csv)
    - Combined pseudo-labels (r4r5, 6601 clips × 10)
    
    Training on 10s windows, 3-channel mel-spectrogram.
    """
    def __init__(
        self,
        train_csv: str,
        audio_dir: str,
        taxonomy_csv: str,
        soundscape_labels_csv: Optional[str] = None,
        soft_pseudo_csv: Optional[str] = None,
        hard_pseudo_csv: Optional[str] = None,
        combined_pseudo_csv: Optional[str] = None,
        soundscape_dir: Optional[str] = None,
        sample_rate: int = 32000,
        duration: float = 10.0,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 50,
        fmax: float = 16000,
        use_pcen: bool = True,
        clips_per_file: int = 3,
        soundscape_oversample: int = 5,
        soft_pseudo_oversample: int = 10,
        hard_pseudo_oversample: int = 5,
        combined_pseudo_oversample: int = 10,
        augment: bool = True,
        mode: str = "train",
        fold: Optional[int] = None,
        fold_csv: Optional[str] = None,
    ):
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

        # Audio augmentation
        self.augmenter = AudioAugmentation()

        # Load taxonomy to build species mapping
        self.taxonomy = pd.read_csv(taxonomy_csv)
        self.species_list = sorted(self.taxonomy["primary_label"].unique().tolist())
        self.num_classes = len(self.species_list)
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}

        # Load training data
        self.samples = []
        self.labels = []
        self.sources = []

        # --- Load labeled training audio ---
        train_df = pd.read_csv(train_csv)

        # Filter by fold if provided
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
            rating = float(row.get("rating", 0))

            if primary_label not in self.species_to_idx:
                continue

            audio_path = self.audio_dir / filename
            if not audio_path.exists():
                continue

            # Build multi-label vector
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[self.species_to_idx[primary_label]] = 1.0

            # Parse secondary labels
            try:
                import ast
                secondary = ast.literal_eval(secondary_labels)
                if isinstance(secondary, list):
                    for s in secondary:
                        s = str(s)
                        if s in self.species_to_idx:
                            label[self.species_to_idx[s]] = 1.0
            except:
                pass

            # Add multiple clips per file (random crop positions)
            for _ in range(clips_per_file):
                self.samples.append({
                    "path": str(audio_path),
                    "source": "labeled",
                    "offset": 0.0,
                    "filename": filename,
                })
                self.labels.append(label)
                self.sources.append("labeled")

        # --- Load labeled soundscape labels ---
        if soundscape_labels_csv and os.path.exists(soundscape_labels_csv):
            soundscape_df = pd.read_csv(soundscape_labels_csv)
            sc_dir = Path(soundscape_dir) if soundscape_dir else self.audio_dir

            for _, row in soundscape_df.iterrows():
                filename = row.get("filename", "")
                row_id = row.get("row_id", "")

                # Build label from columns
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        if float(row[sp]) > 0:
                            label[self.species_to_idx[sp]] = 1.0

                audio_path = sc_dir / filename
                if not audio_path.exists():
                    continue

                # Oversample soundscapes
                for _ in range(soundscape_oversample):
                    self.samples.append({
                        "path": str(audio_path),
                        "source": "soundscape",
                        "offset": 0.0,
                        "filename": filename,
                    })
                    self.labels.append(label)
                    self.sources.append("soundscape")

        # --- Load soft pseudo-labels (Perch teacher) ---
        if soft_pseudo_csv and os.path.exists(soft_pseudo_csv):
            soft_df = pd.read_csv(soft_pseudo_csv)
            for _, row in soft_df.iterrows():
                filename = row.get("filename", "")
                audio_path = self.audio_dir / filename
                if not audio_path.exists():
                    continue

                # Soft labels are probability vectors
                label = np.zeros(self.num_classes, dtype=np.float32)
                for sp in self.species_list:
                    if sp in row and pd.notna(row[sp]):
                        val = float(row[sp])
                        if val > 0:
                            label[self.species_to_idx[sp]] = val

                # Oversample soft pseudo (×10)
                for _ in range(soft_pseudo_oversample):
                    self.samples.append({
                        "path": str(audio_path),
                        "source": "soft_pseudo",
                        "offset": 0.0,
                        "filename": filename,
                    })
                    self.labels.append(label)
                    self.sources.append("soft_pseudo")

        # --- Load hard pseudo-labels ---
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
                    self.samples.append({
                        "path": str(audio_path),
                        "source": "hard_pseudo",
                        "offset": 0.0,
                        "filename": filename,
                    })
                    self.labels.append(label)
                    self.sources.append("hard_pseudo")

        # --- Load combined pseudo-labels (r4r5, 6601 clips) ---
        if combined_pseudo_csv and os.path.exists(combined_pseudo_csv):
            combined_df = pd.read_csv(combined_pseudo_csv)
            for _, row in combined_df.iterrows():
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
                    self.samples.append({
                        "path": str(audio_path),
                        "source": "combined_pseudo",
                        "offset": 0.0,
                        "filename": filename,
                    })
                    self.labels.append(label)
                    self.sources.append("combined_pseudo")

        print(f"[Dataset] Mode={mode}, Samples={len(self.samples)}, Classes={self.num_classes}")
        source_counts = {}
        for s in self.sources:
            source_counts[s] = source_counts.get(s, 0) + 1
        print(f"[Dataset] Source distribution: {source_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        label = self.labels[idx].copy()
        audio_path = sample["path"]

        # Load audio
        audio, _ = load_audio(
            audio_path, sr=self.sample_rate,
            duration=None, offset=sample.get("offset", 0.0),
        )

        # Circular shift augmentation (before pad/crop)
        if self.augment and self.mode == "train":
            audio = self.augmenter.circular_shift(audio)
            audio = self.augmenter.add_noise(audio)
            audio = self.augmenter.apply_gain(audio)

        # Pad or crop to target duration
        audio = pad_or_crop_waveform(audio, self.target_samples)

        # Create 3-channel mel-spectrogram
        mel = create_three_channel_mel(
            audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            use_pcen=self.use_pcen,
        )

        # Crop/pad mel to fixed time steps (for 10s: 10*32000/320 = 1000 steps)
        target_time_steps = int(self.sample_rate * self.duration / self.hop_length) + 1
        for c in range(3):
            mel[c] = crop_or_pad_mel(mel[c], target_time_steps)

        # SpecAugment
        if self.augment and self.mode == "train":
            for c in range(3):
                mel[c] = self.augmenter(mel[c])

        mel = torch.from_numpy(mel).float()
        label = torch.from_numpy(label).float()

        return {
            "mel": mel,
            "label": label,
            "filename": sample.get("filename", ""),
            "source": sample.get("source", "unknown"),
        }

    def get_class_weights(self) -> np.ndarray:
        """Compute inverse class frequency weights for sampling."""
        label_counts = np.zeros(self.num_classes, dtype=np.float64)
        for label in self.labels:
            label_counts += label

        # Inverse frequency weighting
        total = len(self.labels)
        weights = total / (self.num_classes * label_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return weights

    def get_sample_weights(self) -> np.ndarray:
        """Compute per-sample weights for WeightedRandomSampler."""
        weights = np.ones(len(self.samples), dtype=np.float64)
        class_weights = self.get_class_weights()

        for i, label in enumerate(self.labels):
            # Weight by average of active class weights
            active = label > 0
            if active.any():
                weights[i] = class_weights[active].mean()

        return weights


class BirdCLEFInferenceDataset(Dataset):
    """
    BirdCLEF 2026 Inference Dataset.
    
    Processes test soundscapes into 5-second windows for prediction.
    Each 1-minute soundscape → 12 five-second windows.
    """
    def __init__(
        self,
        soundscape_dir: str,
        sample_rate: int = 32000,
        duration: float = 5.0,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 50,
        fmax: float = 16000,
        use_pcen: bool = True,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.window_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.use_pcen = use_pcen

        # Find all soundscape files
        self.files = sorted(self.soundscape_dir.glob("*.ogg"))
        if len(self.files) == 0:
            self.files = sorted(self.soundscape_dir.glob("*.wav"))

        # Build window index: (file_idx, window_start_sec)
        self.windows = []
        for file_idx, fpath in enumerate(self.files):
            # Each file is ~60 seconds → 12 windows of 5 seconds
            audio_duration = 60.0  # Assume 60s
            num_windows = int(audio_duration / self.duration)
            for w in range(num_windows):
                self.windows.append({
                    "file_idx": file_idx,
                    "file_path": str(fpath),
                    "file_name": fpath.name,
                    "offset": w * self.duration,
                    "window_id": w + 1,
                })

        print(f"[InferenceDataset] {len(self.files)} files, {len(self.windows)} windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]

        # Load audio window
        audio, _ = load_audio(
            window["file_path"],
            sr=self.sample_rate,
            duration=self.duration,
            offset=window["offset"],
        )

        # Ensure exact length
        audio = pad_or_crop_waveform(audio, self.window_samples)

        # Create 3-channel mel-spectrogram
        mel = create_three_channel_mel(
            audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            use_pcen=self.use_pcen,
        )

        target_time_steps = int(self.sample_rate * self.duration / self.hop_length) + 1
        for c in range(3):
            mel[c] = crop_or_pad_mel(mel[c], target_time_steps)

        mel = torch.from_numpy(mel).float()

        return {
            "mel": mel,
            "file_name": window["file_name"],
            "window_id": window["window_id"],
            "row_id": f"{window['file_name'].replace('.ogg', '')}_{window['window_id']}",
        }


def create_dataloaders(
    config: dict,
    fold: Optional[int] = None,
    fold_csv: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders with weighted sampling.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = BirdCLEFDataset(
        train_csv=config["train_csv"],
        audio_dir=config["train_audio_dir"],
        taxonomy_csv=config["taxonomy_csv"],
        soundscape_labels_csv=config.get("soundscape_labels_csv"),
        soft_pseudo_csv=config.get("soft_pseudo_csv"),
        hard_pseudo_csv=config.get("hard_pseudo_csv"),
        combined_pseudo_csv=config.get("combined_pseudo_csv"),
        soundscape_dir=config.get("train_soundscape_dir"),
        sample_rate=config["sample_rate"],
        duration=config["train_duration"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        use_pcen=config["use_pcen"],
        clips_per_file=config["clips_per_file"],
        soundscape_oversample=config["soundscape_oversample"],
        soft_pseudo_oversample=config["soft_pseudo_oversample"],
        hard_pseudo_oversample=config["hard_pseudo_oversample"],
        combined_pseudo_oversample=config.get("combined_pseudo_oversample", 10),
        augment=True,
        mode="train",
        fold=fold,
        fold_csv=fold_csv,
    )

    val_dataset = BirdCLEFDataset(
        train_csv=config["train_csv"],
        audio_dir=config["train_audio_dir"],
        taxonomy_csv=config["taxonomy_csv"],
        soundscape_labels_csv=config.get("soundscape_labels_csv"),
        sample_rate=config["sample_rate"],
        duration=config["train_duration"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        use_pcen=config["use_pcen"],
        clips_per_file=1,
        soundscape_oversample=1,
        augment=False,
        mode="val",
        fold=fold,
        fold_csv=fold_csv,
    )

    # Weighted sampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader
