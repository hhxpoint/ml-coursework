# ============================================================
# BirdCLEF 2026 — src/augmentation.py
# Advanced Data Augmentation Pipeline
#
# Includes: FilterAugment, Background Noise Injection,
# Speed/Pitch Perturbation, Time Stretch, Random EQ
# ============================================================

import numpy as np
import random
from typing import Optional, Tuple, List
from pathlib import Path
import librosa


class FilterAugment:
    """
    FilterAugment: Random frequency-domain filtering augmentation.
    
    Applies random bandpass, lowpass, highpass, or notch filters
    to simulate different recording conditions and microphone responses.
    
    Reference: Widely used in DCASE and BirdCLEF solutions.
    """
    def __init__(
        self,
        p: float = 0.5,
        sr: int = 32000,
        n_fft: int = 1024,
    ):
        self.p = p
        self.sr = sr
        self.n_fft = n_fft

    def bandpass(self, audio: np.ndarray, low_freq: int, high_freq: int) -> np.ndarray:
        """Apply bandpass filter."""
        S = librosa.stft(audio, n_fft=self.n_fft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        mask = np.ones_like(S, dtype=np.float64)
        mask[freqs < low_freq] = 0
        mask[freqs > high_freq] = 0

        # Smooth transition
        transition = int(self.n_fft * 0.05)
        for i in range(transition):
            if low_freq > 0:
                idx = np.argmin(np.abs(freqs - low_freq))
                if idx - i >= 0:
                    mask[idx - i] = i / transition
                if idx + i < mask.shape[0]:
                    mask[idx + i] = i / transition

        S_filtered = S * mask
        return librosa.istft(S_filtered, n_fft=self.n_fft)

    def lowpass(self, audio: np.ndarray, cutoff: int) -> np.ndarray:
        """Apply lowpass filter."""
        S = librosa.stft(audio, n_fft=self.n_fft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        mask = np.ones_like(S, dtype=np.float64)
        mask[freqs > cutoff] = 0

        S_filtered = S * mask
        return librosa.istft(S_filtered, n_fft=self.n_fft)

    def highpass(self, audio: np.ndarray, cutoff: int) -> np.ndarray:
        """Apply highpass filter."""
        S = librosa.stft(audio, n_fft=self.n_fft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        mask = np.ones_like(S, dtype=np.float64)
        mask[freqs < cutoff] = 0

        S_filtered = S * mask
        return librosa.istft(S_filtered, n_fft=self.n_fft)

    def notch(self, audio: np.ndarray, center_freq: int, bandwidth: int = 200) -> np.ndarray:
        """Apply notch filter (remove narrow frequency band)."""
        S = librosa.stft(audio, n_fft=self.n_fft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        mask = np.ones_like(S, dtype=np.float64)
        notch_mask = (freqs > center_freq - bandwidth / 2) & (freqs < center_freq + bandwidth / 2)
        mask[notch_mask] = 0

        S_filtered = S * mask
        return librosa.istft(S_filtered, n_fft=self.n_fft)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return audio

        audio = audio.copy()
        filter_type = np.random.choice(["bandpass", "lowpass", "highpass", "notch"])

        if filter_type == "bandpass":
            low = np.random.randint(100, 2000)
            high = np.random.randint(low + 1000, 16000)
            audio = self.bandpass(audio, low, high)
        elif filter_type == "lowpass":
            cutoff = np.random.randint(4000, 16000)
            audio = self.lowpass(audio, cutoff)
        elif filter_type == "highpass":
            cutoff = np.random.randint(100, 2000)
            audio = self.highpass(audio, cutoff)
        elif filter_type == "notch":
            center = np.random.randint(500, 8000)
            audio = self.notch(audio, center, bandwidth=np.random.randint(100, 500))

        return audio


class BackgroundNoiseInjector:
    """
    Inject background noise from soundscape recordings.
    
    Mixes random soundscape segments as background noise
    to improve model robustness to environmental noise.
    """
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        snr_range: Tuple[float, float] = (10.0, 30.0),
        p: float = 0.3,
    ):
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.snr_range = snr_range
        self.p = p
        self.noise_files = []

        if self.noise_dir and self.noise_dir.exists():
            self.noise_files = (
                list(self.noise_dir.glob("*.ogg")) +
                list(self.noise_dir.glob("*.wav"))
            )

    def _get_noise_segment(self, length: int, sr: int = 32000) -> np.ndarray:
        """Get a random noise segment."""
        if not self.noise_files:
            return np.random.randn(length).astype(np.float32) * 0.005

        noise_file = random.choice(self.noise_files)
        try:
            noise, _ = librosa.load(str(noise_file), sr=sr, mono=True)

            if len(noise) < length:
                noise = np.tile(noise, (length // len(noise)) + 1)[:length]
            else:
                start = np.random.randint(0, len(noise) - length)
                noise = noise[start:start + length]

            return noise
        except:
            return np.random.randn(length).astype(np.float32) * 0.005

    def __call__(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        if np.random.random() > self.p:
            return audio

        noise = self._get_noise_segment(len(audio), sr)

        # Compute SNR
        snr_db = np.random.uniform(*self.snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            noise = noise * scale

        return audio + noise


class SpeedPerturbation:
    """
    Speed perturbation augmentation.
    
    Changes playback speed by ±10-20% without changing pitch.
    Effective for bird calls of varying duration.
    """
    def __init__(self, speed_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        self.speed_range = speed_range
        self.p = p

    def __call__(self, audio: np.ndarray, sr: int = 32000) -> Tuple[np.ndarray, int]:
        if np.random.random() > self.p:
            return audio, sr

        speed = np.random.uniform(*self.speed_range)
        audio_stretched = librosa.effects.time_stretch(audio, rate=speed)

        # Pad or crop to original length
        if len(audio_stretched) > len(audio):
            audio_stretched = audio_stretched[:len(audio)]
        else:
            pad_len = len(audio) - len(audio_stretched)
            audio_stretched = np.pad(audio_stretched, (0, pad_len), mode="constant")

        return audio_stretched, sr


class PitchShift:
    """
    Pitch shift augmentation.
    
    Shifts pitch by ±N semitones without changing speed.
    Simulates different species or individual variations.
    """
    def __init__(self, shift_range: Tuple[float, float] = (-2.0, 2.0), p: float = 0.3):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        if np.random.random() > self.p:
            return audio

        n_steps = np.random.uniform(*self.shift_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


class RandomEQ:
    """
    Random equalization augmentation.
    
    Applies random gain to different frequency bands
    to simulate different recording equipment responses.
    """
    def __init__(self, n_bands: int = 5, max_gain_db: float = 6.0, p: float = 0.3):
        self.n_bands = n_bands
        self.max_gain_db = max_gain_db
        self.p = p

    def __call__(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        if np.random.random() > self.p:
            return audio

        n_fft = 1024
        S = librosa.stft(audio, n_fft=n_fft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Create random gain curve
        band_edges = np.linspace(0, len(freqs), self.n_bands + 1, dtype=int)
        gains_db = np.random.uniform(-self.max_gain_db, self.max_gain_db, self.n_bands)
        gains = 10 ** (gains_db / 20)

        # Apply gains
        gain_curve = np.ones(len(freqs))
        for i in range(self.n_bands):
            start, end = band_edges[i], band_edges[i + 1]
            gain_curve[start:end] = gains[i]

        S_eq = S * gain_curve[:, np.newaxis]
        return librosa.istft(S_eq, n_fft=n_fft)


class RandomCrop:
    """
    Random crop augmentation for variable-length training.
    
    Instead of always using the start of the file, randomly
    selects a crop window. Combined with multiple clips_per_file
    to create diverse training samples from one recording.
    """
    def __init__(
        self,
        target_duration: float = 10.0,
        sr: int = 32000,
        min_duration_ratio: float = 0.5,
    ):
        self.target_samples = int(sr * target_duration)
        self.sr = sr
        self.min_samples = int(sr * target_duration * min_duration_ratio)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        total = len(audio)

        if total <= self.target_samples:
            # Pad if too short
            pad_len = self.target_samples - total
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            return np.pad(audio, (pad_left, pad_right), mode="constant")

        # Random crop
        start = np.random.randint(0, total - self.target_samples)
        return audio[start:start + self.target_samples]


class Compose:
    """Compose multiple augmentations."""
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        for aug in self.augmentations:
            if callable(aug):
                result = aug(audio, sr) if "sr" in aug.__call__.__code__.co_varnames else aug(audio)
                if isinstance(result, tuple):
                    audio, sr = result
                else:
                    audio = result
        return audio


def build_augmentation_pipeline(
    config: dict,
) -> Compose:
    """
    Build augmentation pipeline from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed augmentation pipeline
    """
    augmentations = []

    # Circular shift (always on during training)
    # Handled in dataset

    # Speed perturbation
    if config.get("aug_speed", False):
        augmentations.append(SpeedPerturbation(
            speed_range=tuple(config.get("aug_speed_range", (0.8, 1.2))),
            p=config.get("aug_speed_p", 0.3),
        ))

    # Pitch shift
    if config.get("aug_pitch", False):
        augmentations.append(PitchShift(
            shift_range=tuple(config.get("aug_pitch_range", (-2.0, 2.0))),
            p=config.get("aug_pitch_p", 0.3),
        ))

    # FilterAugment
    if config.get("aug_filter", True):
        augmentations.append(FilterAugment(
            p=config.get("aug_filter_p", 0.3),
            sr=config.get("sample_rate", 32000),
        ))

    # Background noise
    if config.get("aug_background_noise", True):
        augmentations.append(BackgroundNoiseInjector(
            noise_dir=config.get("train_soundscape_dir"),
            snr_range=tuple(config.get("aug_noise_snr", (10.0, 30.0))),
            p=config.get("aug_noise_p", 0.3),
        ))

    # Random EQ
    if config.get("aug_eq", False):
        augmentations.append(RandomEQ(
            n_bands=config.get("aug_eq_bands", 5),
            max_gain_db=config.get("aug_eq_max_gain", 6.0),
            p=config.get("aug_eq_p", 0.3),
        ))

    return Compose(augmentations)
