# ============================================================
# BirdCLEF 2026 — src/features.py
# Audio Feature Extraction: PCEN + Mel-Spectrogram
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
from typing import Optional, Tuple


class PCENTransform(nn.Module):
    """
    Per-Channel Energy Normalization (PCEN).
    Critical for BirdCLEF — nPCEN ablation showed -0.049 AUC drop.
    Reference: Wang et al., "Trainable Front-Ends for Robust Sound Event Detection" (2017)
    """
    def __init__(
        self,
        alpha: float = 0.96,
        delta: float = 2.0,
        root: float = 2.0,
        smooth_coef: float = 0.04,
        floor: float = 1e-6,
        trainable: bool = False,
        per_channel: bool = True,
        num_channels: int = 128,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=trainable)
        self.delta = nn.Parameter(torch.tensor(delta), requires_grad=trainable)
        self.root = nn.Parameter(torch.tensor(root), requires_grad=trainable)
        self.smooth_coef = smooth_coef
        self.floor = floor
        self.per_channel = per_channel

        if per_channel and trainable:
            self.alpha = nn.Parameter(torch.full((num_channels,), alpha))
            self.delta = nn.Parameter(torch.full((num_channels,), delta))
            self.root = nn.Parameter(torch.full((num_channels,), root))

        self.smooth_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time) mel-spectrogram
        Returns:
            PCEN-normalized spectrogram
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        alpha = self.alpha.to(x.device)
        delta = self.delta.to(x.device)
        root = self.root.to(x.device)

        if not self.per_channel:
            alpha = alpha.unsqueeze(0).unsqueeze(-1)
            delta = delta.unsqueeze(0).unsqueeze(-1)
            root = root.unsqueeze(0).unsqueeze(-1)
        else:
            alpha = alpha.unsqueeze(0).unsqueeze(-1)
            delta = delta.unsqueeze(0).unsqueeze(-1)
            root = root.unsqueeze(0).unsqueeze(-1)

        if self.smooth_state is None or self.smooth_state.shape != x.shape:
            self.smooth_state = x.clone()
        else:
            self.smooth_state = self.smooth_coef * self.smooth_state + (1 - self.smooth_coef) * x

        smooth = self.smooth_state
        smooth = smooth.clamp(min=self.floor)

        pcen = (x / (self.floor + smooth.pow(alpha))) * (1 + delta).pow(root) - delta.pow(root)
        return pcen.clamp(min=0)


class MelSpectrogramExtractor(nn.Module):
    """
    Mel-Spectrogram feature extractor with optional PCEN normalization.
    Converts raw audio waveforms to log-scale mel-spectrograms.
    """
    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 50,
        fmax: float = 16000,
        use_pcen: bool = True,
        pcen_smooth: float = 0.04,
        pcen_floor: float = 1e-6,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_pcen = use_pcen
        self.top_db = top_db

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
            normalized=False,
        )

        if use_pcen:
            self.pcen = PCENTransform(
                smooth_coef=pcen_smooth,
                floor=pcen_floor,
                num_channels=n_mels,
            )
        else:
            self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) or (batch, channels, samples)
        Returns:
            mel: (batch, n_mels, time) mel-spectrogram
        """
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)  # mono

        mel = self.mel_spec(waveform)

        if self.use_pcen:
            mel = self.pcen(mel)
        else:
            mel = self.amp_to_db(mel)

        return mel


def load_audio(
    path: str,
    sr: int = 32000,
    duration: Optional[float] = None,
    offset: float = 0.0,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        duration: Duration in seconds (None for full file)
        offset: Start offset in seconds
        mono: Convert to mono
        
    Returns:
        audio: (samples,) numpy array
        sr: Sample rate
    """
    audio, file_sr = librosa.load(path, sr=sr, duration=duration, offset=offset, mono=mono)
    return audio, file_sr


def audio_to_mel(
    audio: np.ndarray,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 50,
    fmax: float = 16000,
    use_pcen: bool = True,
) -> np.ndarray:
    """
    Convert audio waveform to mel-spectrogram using librosa.
    
    Args:
        audio: (samples,) waveform
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        use_pcen: Apply PCEN normalization
        
    Returns:
        mel: (n_mels, time) mel-spectrogram
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    if use_pcen:
        mel = librosa.pcen(mel * (2**31), sr=sr, hop_length=hop_length)
    else:
        mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    return mel


def crop_or_pad_mel(
    mel: np.ndarray,
    target_time_steps: int,
) -> np.ndarray:
    """
    Crop or pad mel-spectrogram to fixed time steps.
    
    Args:
        mel: (n_mels, time) mel-spectrogram
        target_time_steps: Target number of time steps
        
    Returns:
        mel: (n_mels, target_time_steps) mel-spectrogram
    """
    n_mels, current_steps = mel.shape

    if current_steps > target_time_steps:
        # Random crop
        start = np.random.randint(0, current_steps - target_time_steps + 1)
        mel = mel[:, start:start + target_time_steps]
    elif current_steps < target_time_steps:
        # Pad with zeros
        pad_left = (target_time_steps - current_steps) // 2
        pad_right = target_time_steps - current_steps - pad_left
        mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

    return mel


def create_three_channel_mel(
    audio: np.ndarray,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 50,
    fmax: float = 16000,
    use_pcen: bool = True,
) -> np.ndarray:
    """
    Create 3-channel mel-spectrogram by computing at different frequency resolutions.
    
    Channel 0: Standard mel-spectrogram
    Channel 1: Higher frequency resolution (n_fft=2048)
    Channel 2: Lower frequency resolution (n_fft=512)
    
    Returns:
        mel_3ch: (3, n_mels, time) 3-channel mel-spectrogram
    """
    # Channel 0: Standard
    mel_0 = audio_to_mel(
        audio, sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen,
    )

    # Channel 1: Higher freq resolution
    mel_1 = audio_to_mel(
        audio, sr, n_fft=n_fft * 2, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen,
    )

    # Channel 2: Lower freq resolution
    mel_2 = audio_to_mel(
        audio, sr, n_fft=n_fft // 2, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, use_pcen=use_pcen,
    )

    # Align time dimensions
    min_time = min(mel_0.shape[1], mel_1.shape[1], mel_2.shape[1])
    mel_0 = mel_0[:, :min_time]
    mel_1 = mel_1[:, :min_time]
    mel_2 = mel_2[:, :min_time]

    mel_3ch = np.stack([mel_0, mel_1, mel_2], axis=0)  # (3, n_mels, time)
    return mel_3ch


class AudioAugmentation:
    """
    Audio augmentation pipeline for BirdCLEF 2026.
    Implements: SpecAugment, circular shift, noise, gain, mixup
    """
    def __init__(
        self,
        spec_time_masks: int = 2,
        spec_time_mask_param: float = 0.15,
        spec_freq_masks: int = 2,
        spec_freq_mask_param: int = 40,
        noise_level: float = 0.005,
        gain_min: float = 0.7,
        gain_max: float = 1.3,
        circ_shift_prob: float = 0.5,
    ):
        self.spec_time_masks = spec_time_masks
        self.spec_time_mask_param = spec_time_mask_param
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_mask_param = spec_freq_mask_param
        self.noise_level = noise_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.circ_shift_prob = circ_shift_prob

    def spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """SpecAugment: time and frequency masking."""
        mel = mel.copy()
        n_mels, n_time = mel.shape

        # Time masking
        for _ in range(self.spec_time_masks):
            max_mask = int(n_time * self.spec_time_mask_param)
            if max_mask > 0:
                mask_len = np.random.randint(0, max_mask + 1)
                mask_start = np.random.randint(0, n_time - mask_len + 1)
                mel[:, mask_start:mask_start + mask_len] = 0

        # Frequency masking
        for _ in range(self.spec_freq_masks):
            max_mask = min(self.spec_freq_mask_param, n_mels)
            if max_mask > 0:
                mask_len = np.random.randint(0, max_mask + 1)
                mask_start = np.random.randint(0, n_mels - mask_len + 1)
                mel[mask_start:mask_start + mask_len, :] = 0

        return mel

    def circular_shift(self, audio: np.ndarray) -> np.ndarray:
        """Circular time shift augmentation."""
        if np.random.random() < self.circ_shift_prob:
            shift = np.random.randint(0, len(audio))
            audio = np.roll(audio, shift)
        return audio

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        if self.noise_level > 0:
            noise = np.random.randn(*audio.shape).astype(np.float32) * self.noise_level
            audio = audio + noise
        return audio

    def apply_gain(self, audio: np.ndarray) -> np.ndarray:
        """Random gain augmentation."""
        gain = np.random.uniform(self.gain_min, self.gain_max)
        audio = audio * gain
        return audio

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to mel-spectrogram."""
        return self.spec_augment(mel)


def mixup(
    mel: np.ndarray,
    label: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    MixUp augmentation for multi-label classification.
    
    Args:
        mel: (3, n_mels, time) mel-spectrogram
        label: (num_classes,) one-hot label
        alpha: Beta distribution parameter
        
    Returns:
        mixed_mel, mixed_label, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = mel.shape[0] if mel.ndim == 4 else 1
    if batch_size == 1:
        # Single sample mixup — needs a partner from same batch
        # This is handled at batch level in DataLoader
        return mel, label, lam

    index = np.random.permutation(batch_size)
    mixed_mel = lam * mel + (1 - lam) * mel[index]
    mixed_label = lam * label + (1 - lam) * label[index]

    return mixed_mel, mixed_label, lam


def pad_or_crop_waveform(
    waveform: np.ndarray,
    target_samples: int,
) -> np.ndarray:
    """
    Pad or crop waveform to target length.
    
    Args:
        waveform: (samples,) waveform
        target_samples: Target number of samples
        
    Returns:
        waveform: (target_samples,) waveform
    """
    current = len(waveform)
    if current > target_samples:
        start = np.random.randint(0, current - target_samples + 1)
        waveform = waveform[start:start + target_samples]
    elif current < target_samples:
        pad_left = (target_samples - current) // 2
        pad_right = target_samples - current - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")
    return waveform
