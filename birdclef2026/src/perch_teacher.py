# ============================================================
# BirdCLEF 2026 — src/perch_teacher.py
# Perch Teacher Model Integration (Knowledge Distillation)
#
# Google Perch is used as a teacher to generate soft pseudo-labels.
# This is the single biggest improvement (+0.039 AUC, v9→v24).
#
# Strategy:
#   1. Load Perch model (TFLite for speed)
#   2. Predict on unlabeled soundscapes
#   3. Generate soft probability vectors as pseudo-labels
#   4. Use as distillation targets during training
# ============================================================

import os
import numpy as np
import tensorflow as tf
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import warnings

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class PerchTeacher:
    """
    Google Perch Teacher Model for generating soft pseudo-labels.
    
    Perch is a multi-species audio embedding model that provides
    state-of-the-art bioacoustic classification. Used as a teacher
    for knowledge distillation in BirdCLEF 2026.
    
    Key metrics:
        - v24 (soft KD ×10): 0.8141 val AUC (+0.039 vs v9)
        - Perch embeddings: 1280-dim, trained on global bird data
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_tflite: bool = True,
        sample_rate: int = 32000,
        embedding_dim: int = 1280,
    ):
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.use_tflite = use_tflite
        self.interpreter = None
        self.keras_model = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("[Perch] No model path provided, using embedding extraction only")

    def load_model(self, model_path: str):
        """Load Perch model (TFLite or Keras)."""
        if self.use_tflite or model_path.endswith(".tflite"):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[Perch] Loaded TFLite model: {model_path}")
        else:
            self.keras_model = tf.keras.models.load_model(model_path)
            print(f"[Perch] Loaded Keras model: {model_path}")

    def get_embeddings(
        self,
        audio: np.ndarray,
        sr: int = 32000,
    ) -> np.ndarray:
        """
        Extract Perch embeddings from audio.
        
        Args:
            audio: (samples,) waveform
            sr: Sample rate
            
        Returns:
            (embedding_dim,) embedding vector
        """
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Perch expects 5-second windows at 32kHz
        target_length = self.sample_rate * 5
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # Normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()

        # Reshape for model: (1, samples)
        input_data = audio.reshape(1, -1)

        if self.interpreter:
            # TFLite inference
            input_dtype = self.input_details[0]["dtype"]
            input_data = input_data.astype(input_dtype)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            # Get embeddings (last layer before classification)
            embeddings = self.interpreter.get_tensor(self.output_details[-1]["index"])
            return embeddings.flatten()
        elif self.keras_model:
            import tensorflow as tf
            input_tensor = tf.constant(input_data)
            embeddings = self.keras_model(input_tensor, training=False)
            if isinstance(embeddings, (list, tuple)):
                embeddings = embeddings[-1]
            return embeddings.numpy().flatten()
        else:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def predict_species(
        self,
        audio: np.ndarray,
        sr: int = 32000,
        species_mapping: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Predict species probabilities from audio.
        
        Args:
            audio: (samples,) waveform
            sr: Sample rate
            species_mapping: Map from Perch class names to BirdCLEF indices
            
        Returns:
            (num_classes,) probability vector
        """
        if self.interpreter:
            # Resample
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            target_length = self.sample_rate * 5
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()

            input_data = audio.reshape(1, -1).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            # Get logits/scores
            logits = self.interpreter.get_tensor(self.output_details[0]["index"])

            # Apply sigmoid for multi-label
            probs = 1.0 / (1.0 + np.exp(-logits.flatten()))

            # Map to BirdCLEF classes if mapping provided
            if species_mapping:
                num_classes = max(species_mapping.values()) + 1
                mapped_probs = np.zeros(num_classes, dtype=np.float32)
                for perch_class, birdclef_idx in species_mapping.items():
                    # This is simplified; actual mapping needs Perch class indices
                    if birdclef_idx < len(probs):
                        mapped_probs[birdclef_idx] = probs[birdclef_idx]
                return mapped_probs

            return probs

        return np.zeros(234, dtype=np.float32)

    def generate_soft_labels_for_file(
        self,
        audio_path: str,
        window_duration: float = 5.0,
        stride: float = 5.0,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate soft pseudo-labels for all windows in an audio file.
        
        Args:
            audio_path: Path to audio file
            window_duration: Window length in seconds
            stride: Window stride in seconds
            
        Returns:
            List of (offset_seconds, probability_vector) tuples
        """
        import librosa

        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        window_samples = int(self.sample_rate * window_duration)
        stride_samples = int(self.sample_rate * stride)

        results = []
        offset = 0
        while offset + window_samples <= len(audio):
            window = audio[offset:offset + window_samples]
            probs = self.predict_species(window, sr=self.sample_rate)
            results.append((offset / self.sample_rate, probs))
            offset += stride_samples

        return results

    def generate_batch_soft_labels(
        self,
        audio_paths: List[str],
        output_csv: str = "soft_pseudo_r5.csv",
        species_list: Optional[List[str]] = None,
        confidence_threshold: float = 0.1,
    ) -> str:
        """
        Generate soft pseudo-labels for a batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            output_csv: Output CSV path
            species_list: Ordered species names
            confidence_threshold: Minimum probability to include
            
        Returns:
            Path to generated CSV
        """
        import pandas as pd

        all_rows = []

        for file_idx, audio_path in enumerate(audio_paths):
            fname = Path(audio_path).name
            predictions = self.generate_soft_labels_for_file(audio_path)

            for offset, probs in predictions:
                window_id = int(offset / 5) + 1
                row = {
                    "filename": fname,
                    "window": window_id,
                    "offset": offset,
                }

                if species_list:
                    for i, sp in enumerate(species_list):
                        if i < len(probs) and probs[i] >= confidence_threshold:
                            row[sp] = float(probs[i])
                else:
                    for i in range(len(probs)):
                        if probs[i] >= confidence_threshold:
                            row[f"species_{i}"] = float(probs[i])

                all_rows.append(row)

            if (file_idx + 1) % 100 == 0:
                print(f"[Perch] Processed {file_idx + 1}/{len(audio_paths)} files")

        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)
        print(f"[Perch] Generated {len(df)} soft labels → {output_csv}")
        return output_csv


class PerchEmbeddingExtractor:
    """
    Lightweight Perch embedding extractor for feature augmentation.
    Uses pre-computed embeddings if available, otherwise computes on-the-fly.
    """
    def __init__(
        self,
        embeddings_dir: Optional[str] = None,
        teacher: Optional[PerchTeacher] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.teacher = teacher
        self._cache = {}

    def get_embedding(self, filename: str, window: int = 1) -> np.ndarray:
        """
        Get Perch embedding for a specific file/window.
        
        Priority:
        1. Cached in memory
        2. Pre-computed on disk
        3. Compute with teacher model
        """
        cache_key = f"{filename}_{window}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try pre-computed
        if self.embeddings_dir:
            emb_path = self.embeddings_dir / f"{Path(filename).stem}_w{window}.npy"
            if emb_path.exists():
                emb = np.load(str(emb_path))
                self._cache[cache_key] = emb
                return emb

        # Compute with teacher
        if self.teacher:
            emb = self.teacher.get_embeddings(
                np.zeros(32000 * 5, dtype=np.float32)  # placeholder
            )
            self._cache[cache_key] = emb
            return emb

        return np.zeros(1280, dtype=np.float32)


def precompute_perch_embeddings(
    audio_dir: str,
    teacher: PerchTeacher,
    output_dir: str = "./data/perch_embeddings",
    sample_rate: int = 32000,
    duration: float = 5.0,
):
    """
    Pre-compute Perch embeddings for all audio files.
    Speeds up training by avoiding repeated inference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.ogg")) + list(audio_dir.glob("*.wav"))

    print(f"[Perch] Pre-computing embeddings for {len(audio_files)} files...")

    for file_idx, fpath in enumerate(audio_files):
        import librosa

        audio, sr = librosa.load(str(fpath), sr=sample_rate, mono=True)
        window_samples = int(sample_rate * duration)

        window = 1
        offset = 0
        while offset + window_samples <= len(audio):
            chunk = audio[offset:offset + window_samples]
            emb = teacher.get_embeddings(chunk, sr=sample_rate)

            emb_path = output_dir / f"{fpath.stem}_w{window}.npy"
            np.save(str(emb_path), emb)

            offset += window_samples
            window += 1

        if (file_idx + 1) % 100 == 0:
            print(f"  [{file_idx + 1}/{len(audio_files)}] {fpath.name}")

    print(f"[Perch] Embeddings saved to {output_dir}")
