#!/bin/bash

# Exit on any error
set -e

echo "Starting ReTabSyn Pre-training..."

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

# Define variables for command-line arguments
LLM_NAME="gpt2"  # Specify the desired language model
EFFICIENT_TRAINING=""  # Specify efficient training method if any
DATA_PATH="csv/example/wilt.csv"  # Path to your input CSV file
TEST_IDX=0  # Test index for multiple runs
DURATION=50  # Training duration/epochs
BATCH_SIZE=64  # Batch size for training
CONTINUE_TRAINING=False  # Set to true to continue training from checkpoint
CHECKPOINT_DIR="checkpoints"  # Directory for saving checkpoints
SYNTH_DATA_DIR="synth_data"  # Directory for saving synthetic data
SAVE_STEPS=30000  # Steps between saving checkpoints
N_AUG=1000  # Number of synthetic samples to augment to training data

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file $DATA_PATH not found"
    echo "Please ensure the data file exists in the specified path"
    exit 1
fi

echo "Configuration:"
echo "  LLM Model: $LLM_NAME"
echo "  Data Path: $DATA_PATH"
echo "  Test Index: $TEST_IDX"
echo "  Duration: $DURATION"
echo "  Batch Size: $BATCH_SIZE"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  Synthetic Data Dir: $SYNTH_DATA_DIR"
echo "  N Augmentation: $N_AUG"

# Create directories if they don't exist
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$SYNTH_DATA_DIR"

echo "Starting pre-training..."

# Construct the command to run the Python script
python3 pretrain.py \
  --llm_name "$LLM_NAME" \
  --efficient_finetuning "$EFFICIENT_TRAINING" \
  --data_path "$DATA_PATH" \
  --test_idx "$TEST_IDX" \
  --duration "$DURATION" \
  --batch_size "$BATCH_SIZE" \
  $( [ "$CONTINUE_TRAINING" = true ] && echo "--continue_training" ) \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --synth_data_dir "$SYNTH_DATA_DIR" \
  --save_steps "$SAVE_STEPS" \
  --split_label \
  --N_aug "$N_AUG" \
  --shuffle_data

echo "Pre-training completed successfully!" 
