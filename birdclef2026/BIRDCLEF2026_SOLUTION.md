# BirdCLEF 2026 Competition Solution Design

Based on the BirdCLEF+ 2026 Analysis Report, this document provides a comprehensive solution design for achieving a high score in the competition while respecting the CPU-only inference constraint.

## Table of Contents
1. [Competition Overview](#competition-overview)
2. [Key Challenges](#key-challenges)
3. [Solution Architecture](#solution-architecture)
4. [Implementation Plan](#implementation-plan)
5. [Timeline](#timeline)
6. [Key Success Factors](#key-success-factors)
7. [References](#references)

## Competition Overview
- **Task**: Predict probability of 234 species presence in 5-second audio windows from test soundscapes
- **Evaluation Metric**: Macro-averaged ROC-AUC
- **Key Constraints**:
  - CPU-only inference (90 minutes maximum runtime)
  - No internet access during submission
  - Must pre-package all models in notebook
- **Data**:
  - 35,549 training recordings (audio)
  - ~600 test soundscapes (1 minute each)
  - 66 labeled soundscape files (gold standard domain data)
  - Additional unlabeled soundscape files (for semi-supervised learning)

## Key Challenges
1. **Extreme Class Imbalance**: 97.9% training recordings are birds; Reptilia has only 1 species with 1 recording
2. **Few-shot Species**: 25 species have <10 recordings; 18 species have ≤5 recordings
3. **Domain Shift**: Training data = short focal recordings; Test data = continuous passive soundscapes
4. **Multi-label Detection**: 89.4% soundscape segments contain multiple simultaneous species
5. **CPU Inference Bottleneck**: ~7,200 five-second segments must be processed in 90 minutes
6. **Label Noise**: Variable quality in Xeno-canto/iNaturalist data

## Solution Architecture

### High-Level Pipeline
```
Data Pipeline → Feature Extraction → Model Ensemble → Post-processing → Submission
```

### 1. Data Pipeline Strategy
- **Three-stage Data Utilization**:
  1. Pre-training: BirdCLEF 2021-2025 datasets (+1-2% AUC)
  2. Fine-tuning: 2026 train.csv + labeled soundscape labels (domain priority)
  3. Semi-supervised: Pseudo-labeling unlabeled soundscapes (domain adaptation)
- **Sampling Strategy**: Inverse class frequency weighting (oversample rare species)
- **Audio Processing**: 
  - Training: 5-10 second windows (increased diversity)
  - Inference: Fixed 5 second windows (test format match)
  - Sample rate: 32kHz (standard for bioacoustics)

### 2. Feature Extraction
**Primary Path** (High Priority):
- Mel-spectrogram: n_mels=128, hop_length=320 (10ms), window 5-10s
- Augmentation: Time masking, frequency masking, MixUp, FilterAugment, noise injection, gain jitter
- Implementation: librosa/torchaudio + albumentations

**Secondary Path** (Medium Priority):
- Pre-trained bioacoustic embeddings (Perch/BirdNET) via TFLite for CPU efficiency
- Provides biological priors for robust detection of atypical calls

### 3. Model Architecture
**Ensemble Strategy** (Quantile-Mix α=0.5):
```
Member 1: EfficientNetV2-S (tf_efficientnetv2_s_in21k) - Primary backbone
Member 2: EfficientNet-B3 - Lightweight diversity member
Member 3: ECA-NfNet-L0 - Alternative architecture for robustness
Optional Member 4: Perch-based lightweight head (TFLite optimized)
```

**Non-Bird Specialist Handling**:
- Two-stage prediction: General bird/non-bird classifier → Fine-grained species classification
- Dedicated non-bird sub-model fine-tuned on soundscape labels
- Extreme few-shot handling: Reptilia (1 sample) using Perch embedding similarity thresholds

### 4. Training Strategy
**Three-stage Training**:
```
Stage 1 (Pre-training):
  - Data: BirdCLEF 2021-2025 (all available)
  - Task: Multi-label classification (234 classes)
  - Loss: SoftAUCLoss + BCE (for stability)
  - Output: Pre-trained weights

Stage 2 (Fine-tuning):
  - Data: train.csv + soundscape labels (gold standard domain data)
  - Sampling: Inverse class frequency weighting
  - Loss: SoftAUCLoss (directly optimizes evaluation metric)
  - Output: Baseline fine-tuned model

Stage 3 (Semi-supervised):
  - Data: Unlabeled soundscapes + high-confidence pseudo-labels (threshold ≥0.75)
  - Technique: MixUp + StochasticDepth regularization
  - Iterations: 2-3 rounds with gradually increasing confidence thresholds
  - Output: Domain-adapted final models
```

### 5. Inference Optimization (Critical for 90-min CPU limit)
- **Model Format**: Convert all PyTorch models to ONNX
- **Runtime**: ONNX Runtime with multi-threading enabled
- **Batching**: Soundscape-level batching (process all windows of one soundscape together)
- **Precision**: FP16 mixed precision (optional INT8 for further acceleration)
- **Pre-computation**: Pre-calculate and store Mel-spectrograms for all test soundscapes
- **Time Budget**: ~8 seconds/soundscape (90 min / 600 soundscapes = 9 sec/soundscape with margin)

### 6. Post-processing
- **Temperature Scaling**: Per-species temperature parameters calibrated on validation soundscapes
- **Temporal Smoothing**: Gaussian smoothing across adjacent 5-second windows (species persistence assumption)
- **Species-specific Thresholds**: Optimized F1 thresholds per species from validation soundscapes
- **Ensemble**: Quantile-Mix (α=0.5): 50% mean average + 50% rank average

## Implementation Plan

### Phase 1: Environment Setup & Data Exploration (Day 1)
```bash
# Setup
mkdir -p birdclef2026 && cd birdclef2026
pip install --upgrade torch torchaudio tensorflow librosa numpy pandas scikit-learn tqdm
pip install timm onnx onnxruntime albumentations soundfile

# Data exploration
python data_explore.py  # Analyze train.csv, species distribution, etc.
```

### Phase 2: Baseline Establishment (Days 2-3)
```bash
# Feature extraction
python feature_extraction.py  # Implement Mel-spectrogram pipeline

# Baseline model
python baseline_model.py      # Simple EfficientNet-B0 model
python train_baseline.py      # Initial training
python generate_baseline_submission.py  # First submission
```

### Phase 3: Strong Baseline (Days 4-6)
```bash
# Pre-training on historical data
python pretrain.py            # BirdCLEF 2021-2025 pre-training

# Soundscape label utilization
python soundscape_train.py    # Special handling of labeled soundscapes

# Improved training with weighted sampling
python improved_train.py      # Class-weighted sampling + combined data
```

### Phase 4: Domain Adaptation & Semi-supervised Learning (Days 7-9)
```bash
# Pseudo-labeling pipeline
python pseudo_labeling.py     # Generate and iterate pseudo-labels

# Non-bird specialist
python non_bird_specialist.py # Dedicated handling for rare taxa
```

### Phase 5: Ensemble & Post-processing (Days 10-11)
```bash
# Train ensemble members
python train_ensemble.py      # Train multiple model architectures

# Ensemble inference
python ensemble_inference.py  # Quantile-Mix combination

# Post-processing
python post_processing.py     # Temperature scaling, temporal smoothing
```

### Phase 6: Inference Optimization & Validation (Day 12)
```bash
# ONNX conversion
python convert_to_onnx.py     # Convert PyTorch → ONNX for CPU speed

# Speed verification
python verify_onnx_inference.py  # Benchmark and estimate total time
```

### Phase 7: Final Submission (Days 13-14)
```bash
# Generate final submission
python generate_final_submission.py  # ONNX-based ensemble with all optimizations

# Validation checks
# (Run validation checklist from Phase 6.2)
```

## Timeline (Adapted from Report)
- **Week 1-2 (Mar 11-25)**: Foundation
  - Complete EDA, establish baseline pipeline, verify <90min CPU inference
  - Download BirdCLEF 2021-2025 datasets for pre-training
- **Week 3-4 (Mar 25 - Apr 8)**: Strong Baseline
  - Pre-train on historical data, fine-tune on 2026 data + soundscapes
  - Implement ONNX export, validate inference speed
  - Submit first competitive notebook
- **Week 5-7 (Apr 8 - 29)**: Domain Adaptation
  - Implement pseudo-labeling pipeline for unlabeled soundscapes
  - Experiment with SoftAUCLoss, weighted sampling for non-birds
  - Add augmentation (FilterAugment, MixUp, noise)
  - Run ablation studies
- **Week 8-10 (Apr 29 - May 20)**: Ensemble & Polish
  - Train ensemble members (EfficientNet variants, ECA-NfNet-L0)
  - Implement Quantile-Mix ensemble
  - Post-processing: temporal smoothing, species calibration
  - Investigate Perch embeddings as ensemble member
- **Week 11-12 (May 20 - Jun 3)**: Final Submissions
  - Focus on submission reliability (no timeouts!)
  - Final ensemble selection based on CV vs public LB
  - Prepare 2 best submissions for final selection
  - Consider working note submission ($2,500 prize, deadline Jun 17)

## Key Success Factors
1. **Non-Bird Species Focus**: This is the primary leaderboard differentiator
2. **Soundscape Label Utilization**: Treat as domain gold standard (some species ONLY here)
3. **CPU-first Optimization**: Profile inference speed from day one, don't leave it to the end
4. **Historical Pre-training**: Non-negotiable +1-2% AUC boost from BirdCLEF 2021-2025
5. **Progressive Pseudo-labeling**: Key for domain adaptation between focal recordings and soundscapes
6. **Quantile-Mix Ensemble**: Proven strategy to balance accuracy and diversity
7. **10s Train → 5s Infer**: Consistent winner strategy across BirdCLEF years
8. **Quality over Quantity for External Data**: Prioritize geographically relevant (South America) data

## Critical Implementation Notes
- **Validation Strategy**: Use labeled soundscapes as validation set (same domain as test)
- **Threshold Optimization**: Calculate species-specific thresholds from validation soundscapes
- **Temporal Consistency**: Apply smoothing assuming species persist across consecutive windows
- **Confidence Calibration**: Use temperature scaling to improve probability estimates
- **Submission Format**: Match sample_submission.csv EXACTLY (row_id format, column order)
- **Error Handling**: Robust audio loading (handle corrupted files, variable lengths)
- **Memory Management**: Batch processing to avoid OOM during feature extraction

## Expected Performance
With proper implementation of this solution:
- Baseline (EfficientNet-B0): ~0.80-0.85 AUC
- + Pre-training: +1-2% AUC
- + Soundscape labels: +1-2% AUC  
- + Pseudo-labeling: +1-3% AUC
- + Non-bird specialist: Critical differentiator for leaderboard position
- + Ensemble + Post-processing: +0.5-1.5% AUC
- **Target Range**: 0.88-0.93+ AUC (top 10-20% competitive range)

## Files to Create
Based on this plan, you should create these implementation files:
1. `data_explore.py` - Initial data analysis
2. `feature_extraction.py` - Mel-spectrogram pipeline
3. `baseline_model.py` & `train_baseline.py` - Initial model & training
4. `generate_baseline_submission.py` - First submission generator
5. `pretrain.py` - Historical data pre-training
6. `soundscape_train.py` - Specialized soundscape label usage
7. `improved_train.py` - Weighted sampling & combined data training
8. `pseudo_labeling.py` - Iterative pseudo-labeling pipeline
9. `non_bird_specialist.py` - Dedicated non-bird handling
10. `train_ensemble.py` - Multi-model ensemble training
11. `ensemble_inference.py` - Quantile-Mix ensemble inference
12. `post_processing.py` - Temperature scaling & temporal smoothing
13. `convert_to_onnx.py` - PyTorch to ONNX conversion
14. `verify_onnx_inference.py` - Speed benchmarking
15. `generate_final_submission.py` - Final submission pipeline

Good luck with BirdCLEF 2026! Remember: A model that finishes in time beats a model that times out, no matter how accurate it is.