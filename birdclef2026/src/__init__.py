# ============================================================
# BirdCLEF 2026 — src/__init__.py
# Package initialization
# ============================================================

from .features import (
    load_audio,
    audio_to_mel,
    create_three_channel_mel,
    PCENTransform,
    MelSpectrogramExtractor,
    AudioAugmentation,
    crop_or_pad_mel,
    pad_or_crop_waveform,
    mixup,
)
from .dataset import BirdCLEFDataset, BirdCLEFInferenceDataset, create_dataloaders
from .model import (
    BirdCLEFModel,
    GeMPooling,
    ModelEnsemble,
    build_model,
    count_parameters,
    load_checkpoint,
)
from .losses import (
    AsymmetricLoss,
    SoftAsymmetricLoss,
    DualLoss,
    SoftLabelDistillationLoss,
    FocalLoss,
    AUCPMLoss,
    LabelSmoothingBCE,
    build_criterion,
)
from .trainer import BirdCLEFTrainer
from .inference import (
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    export_to_onnx,
    run_inference_on_soundscapes,
    profile_inference_speed,
)
from .postprocessing import (
    apply_postprocessing,
    ensemble_predictions,
    lmax_pre_aves,
    logit_scale,
    softmax_richness,
    temporal_smoothing,
    noisy_or_aggregate,
    soft_entropy_weighting,
    power_scaling,
    species_calibration,
)
from .augmentation import (
    FilterAugment,
    BackgroundNoiseInjector,
    SpeedPerturbation,
    PitchShift,
    RandomEQ,
    RandomCrop,
    Compose,
    build_augmentation_pipeline,
)
from .perch_teacher import (
    PerchTeacher,
    PerchEmbeddingExtractor,
    precompute_perch_embeddings,
)
