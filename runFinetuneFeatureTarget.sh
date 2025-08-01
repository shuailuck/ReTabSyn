#!/bin/bash

# Exit on any error
set -e

echo "Starting ReTabSyn DPO Fine-tuning..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rluf

# Check if CUDA is available
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "CUDA is available for training"
else
    echo "Warning: CUDA is not available. Training will use CPU (this may be very slow)"
fi

# Define configuration variables
LLM="gpt2"
DATA_PATH="csv/example/wilt.csv"
SYNTH_DATA_DIR="synth_data"
CHECKPOINT_PATH="checkpoints/greatgpt2_checkpoint_wilt_0"
BASE_MODEL_PATH="checkpoints/trained_base_gpt2_wilt_0"
OUTPUT_DIR="checkpoints/dpo_model"
MAX_SEQ_LEN=512
EPOCHS=3
DPO_DATA=0
TEST_IDX=0
SHUFFLE=1
N_AUG=1000
PERTURB_TARGET_PROB=0.5

# Check if required files exist
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file $DATA_PATH not found"
    echo "Please ensure the data file exists in the specified path"
    exit 1
fi

if [ ! -d "$SYNTH_DATA_DIR" ]; then
    echo "Error: Synthetic data directory $SYNTH_DATA_DIR not found"
    echo "Please run pre-training first to generate synthetic data"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory $CHECKPOINT_PATH not found"
    echo "Please run pre-training first to generate checkpoints"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "Error: Base model directory $BASE_MODEL_PATH not found"
    echo "Please run pre-training first to generate base model"
    exit 1
fi

echo "Configuration:"
echo "  LLM Model: $LLM"
echo "  Data Path: $DATA_PATH"
echo "  Synthetic Data Dir: $SYNTH_DATA_DIR"
echo "  Checkpoint Path: $CHECKPOINT_PATH"
echo "  Base Model Path: $BASE_MODEL_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Max Seq Len: $MAX_SEQ_LEN"
echo "  Epochs: $EPOCHS"
echo "  Test Index: $TEST_IDX"
echo "  N Augmentation: $N_AUG"
echo "  Perturb Target Prob: $PERTURB_TARGET_PROB"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting DPO fine-tuning..."

python DPOFeatureAndTarget.py \
  --llm "$LLM" \
  --data_path "$DATA_PATH" \
  --synth_data_dir "$SYNTH_DATA_DIR" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --base_model_path "$BASE_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --epochs "$EPOCHS" \
  --dpo_data "$DPO_DATA" \
  --test_idx "$TEST_IDX" \
  --shuffle "$SHUFFLE" \
  --N_aug "$N_AUG" \
  --perturb_target_prob "$PERTURB_TARGET_PROB"

echo "DPO fine-tuning completed successfully!"