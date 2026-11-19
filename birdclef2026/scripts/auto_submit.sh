# ============================================================
# BirdCLEF 2026 — scripts/auto_submit.sh
# Automatic Kaggle Code Competition Submission Pipeline
#
# Based on: automatic_submission_workflow.txt
#
# Pipeline:
#   1. Train model locally (or use existing checkpoint)
#   2. Upload model as Kaggle Dataset
#   3. Push Kaggle notebook
#   4. Kaggle reruns notebook → generates submission.csv → submits
#
# Prerequisites:
#   pip install kaggle
#   ~/.kaggle/kaggle.json configured
#
# Usage:
#   bash scripts/auto_submit.sh
#   bash scripts/auto_submit.sh --skip-train
#   bash scripts/auto_submit.sh --config configs/v28_kitchen_sink.yaml
# ============================================================

#!/bin/bash
set -e

# ============================================================
# Configuration
# ============================================================
CONFIG="configs/default.yaml"
SKIP_TRAIN=false
SKIP_UPLOAD=false
MODEL_VERSION="v$(date +%Y%m%d)"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-YOUR_KAGGLE_USERNAME}"
DATASET_ID="${KAGGLE_USERNAME}/birdclef2026-model"
KERNEL_ID="${KAGGLE_USERNAME}/birdclef2026-inference"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-upload) SKIP_UPLOAD=true; shift ;;
        --version) MODEL_VERSION="$2"; shift 2 ;;
        --username) KAGGLE_USERNAME="$2"; DATASET_ID="${KAGGLE_USERNAME}/birdclef2026-model"; KERNEL_ID="${KAGGLE_USERNAME}/birdclef2026-inference"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  BirdCLEF 2026 Auto-Submit Pipeline"
echo "============================================================"
echo "  Config: ${CONFIG}"
echo "  Version: ${MODEL_VERSION}"
echo "  Kaggle User: ${KAGGLE_USERNAME}"
echo "  Dataset: ${DATASET_ID}"
echo "  Kernel: ${KERNEL_ID}"
echo "============================================================"

# ============================================================
# Step 1: Check Kaggle CLI
# ============================================================
echo ""
echo "[Step 1] Checking Kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
    echo "  Installing Kaggle CLI..."
    pip install kaggle
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "  ERROR: ~/.kaggle/kaggle.json not found!"
    echo "  Download from: Kaggle → Account → Create New API Token"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json
echo "  ✓ Kaggle CLI ready"

# ============================================================
# Step 2: Train Model (or skip)
# ============================================================
echo ""
echo "[Step 2] Training..."

if [ "$SKIP_TRAIN" = false ]; then
    echo "  Running 4-fold grand combo training..."
    python -m scripts.train_4fold_grand --config "${CONFIG}"

    if [ $? -ne 0 ]; then
        echo "  ERROR: Training failed!"
        exit 1
    fi
    echo "  ✓ Training complete"
else
    echo "  Skipping training (--skip-train)"
fi

# Find latest checkpoint
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['output_dir'])")
ONNX_DIR="${OUTPUT_DIR}/onnx"

if [ ! -d "${ONNX_DIR}" ]; then
    echo "  ERROR: ONNX directory not found: ${ONNX_DIR}"
    echo "  Run training first or use --skip-train with existing checkpoints"
    exit 1
fi

ONNX_FILES=($(ls ${ONNX_DIR}/*.onnx 2>/dev/null))
if [ ${#ONNX_FILES[@]} -eq 0 ]; then
    echo "  ERROR: No ONNX models found in ${ONNX_DIR}"
    exit 1
fi

echo "  Found ${#ONNX_FILES[@]} ONNX model(s)"

# ============================================================
# Step 3: Prepare Kaggle Dataset
# ============================================================
echo ""
echo "[Step 3] Preparing Kaggle Dataset..."

DATASET_DIR="./model_dataset"
mkdir -p "${DATASET_DIR}"

# Copy ONNX models
cp ${ONNX_DIR}/*.onnx "${DATASET_DIR}/"

# Copy config
cp "${CONFIG}" "${DATASET_DIR}/config.yaml"

# Copy taxonomy (needed by notebook)
if [ -f "data/taxonomy.csv" ]; then
    cp data/taxonomy.csv "${DATASET_DIR}/"
fi

# Create dataset metadata
cat <<EOF > "${DATASET_DIR}/dataset-metadata.json"
{
  "title": "BirdCLEF2026 Model ${MODEL_VERSION}",
  "id": "${DATASET_ID}",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

echo "  ✓ Dataset prepared: ${DATASET_DIR}"

# ============================================================
# Step 4: Upload/Update Kaggle Dataset
# ============================================================
echo ""
echo "[Step 4] Uploading to Kaggle..."

if [ "$SKIP_UPLOAD" = false ]; then
    # Check if dataset exists
    if kaggle datasets list -s "${DATASET_ID}" 2>/dev/null | grep -q "${DATASET_ID}"; then
        echo "  Updating existing dataset..."
        kaggle datasets version -p "${DATASET_DIR}" -m "Update ${MODEL_VERSION}"
    else
        echo "  Creating new dataset..."
        kaggle datasets create -p "${DATASET_DIR}"
    fi

    if [ $? -ne 0 ]; then
        echo "  ERROR: Dataset upload failed!"
        exit 1
    fi
    echo "  ✓ Dataset uploaded: ${DATASET_ID}"
else
    echo "  Skipping upload (--skip-upload)"
fi

# ============================================================
# Step 5: Prepare Inference Notebook
# ============================================================
echo ""
echo "[Step 5] Preparing inference notebook..."

NOTEBOOK_DIR="./kaggle_notebook_submission"
mkdir -p "${NOTEBOOK_DIR}"

# Copy inference notebook
cp kaggle_notebook/notebook.py "${NOTEBOOK_DIR}/notebook.py"

# Create kernel metadata
cat <<EOF > "${NOTEBOOK_DIR}/kernel-metadata.json"
{
  "id": "${KERNEL_ID}",
  "title": "BirdCLEF2026 Inference ${MODEL_VERSION}",
  "code_file": "notebook.py",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "competition_sources": ["birdclef-2026"],
  "dataset_sources": ["${DATASET_ID}"]
}
EOF

echo "  ✓ Notebook prepared: ${NOTEBOOK_DIR}"

# ============================================================
# Step 6: Push Notebook (Triggers Submission)
# ============================================================
echo ""
echo "[Step 6] Pushing notebook to Kaggle..."

kaggle kernels push -p "${NOTEBOOK_DIR}"

if [ $? -ne 0 ]; then
    echo "  ERROR: Notebook push failed!"
    exit 1
fi

echo "  ✓ Notebook pushed: ${KERNEL_ID}"
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo "  Kaggle will now:"
echo "    1. Rerun your notebook on CPU"
echo "    2. Generate submission.csv"
echo "    3. Submit to leaderboard"
echo ""
echo "  Monitor: https://www.kaggle.com/code/${KERNEL_ID}"
echo "============================================================"
