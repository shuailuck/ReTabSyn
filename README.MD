# ReTabSyn: Reinforcement Learning for Realistic Tabular Synthesis

This repository contains code for training and fine-tuning language models for tabular data generation using GReaT (Generative Representation Learning for Tabular Data) and DPO (Direct Preference Optimization).

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Pre-training with GReaT](#pre-training-with-great)
3. [DPO Fine-tuning](#dpo-fine-tuning)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)

## Environment Setup

### 1. Install Conda Environment

The project uses a conda environment with all necessary dependencies. To set up the environment:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate rluf
```

### 2. Verify Installation

After activating the environment, verify that all dependencies are installed correctly:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import trl; print(f'TRL version: {trl.__version__}')"
```

### 3. Data Preparation

Ensure your CSV data files are placed in the appropriate directories:
- For pre-training: Place data files in `csv/` directory
- The scripts expect specific data formats with optional `SOURCE_LABEL` column for train/test splits

## Pre-training with GReaT

### Overview

The pre-training step uses GReaT (Generative Representation Learning for Tabular Data) to train a language model on tabular data. This creates a foundation model that can generate synthetic tabular data.

### Running Pre-training

Use the provided script to run pre-training:

```bash
# Make the script executable
chmod +x runPretrain.sh

# Run pre-training
./runPretrain.sh
```

### Configuration Options

The `runPretrain.sh` script includes the following key parameters:

- **LLM_NAME**: Base language model (default: "gpt2")
- **DATA_PATH**: Path to input CSV file (default: "csv/imbalance/creditcard.csv")
- **TEST_IDX**: Test index for multiple runs (default: 20005)
- **DURATION**: Training duration/epochs (default: 50)
- **BATCH_SIZE**: Batch size for training (default: 64)
- **CHECKPOINT_DIR**: Directory for saving checkpoints (default: "checkpoints_imbalance")
- **SYNTH_DATA_DIR**: Directory for saving synthetic data (default: "synth_data_imbalance")
- **N_AUG**: Number of synthetic samples for augmentation (default: 1000)

### Output

The pre-training process will generate:
- Checkpoint files in the specified `CHECKPOINT_DIR`
- Synthetic data files in the specified `SYNTH_DATA_DIR`
- Training logs and metrics

## DPO Fine-tuning

### Overview

DPO (Direct Preference Optimization) fine-tuning improves the model's ability to generate high-quality synthetic data by learning from preference comparisons between generated samples.

### Running DPO Fine-tuning

Use the provided script to run DPO fine-tuning:

```bash
# Make the script executable
chmod +x runFinetuneFeatureTarget.sh

# Run DPO fine-tuning
./runFinetuneFeatureTarget.sh
```

### Configuration Options

The `runFinetuneFeatureTarget.sh` script includes the following key parameters:

- **llm**: Base language model (default: "gpt2")
- **data_path**: Path to input CSV file
- **synth_data_dir**: Directory containing synthetic data
- **checkpoint_path**: Path to pre-trained model checkpoint
- **base_model_path**: Path to base model
- **output_dir**: Directory for saving DPO fine-tuned model
- **max_seq_len**: Maximum sequence length (default: 512)
- **epochs**: Number of training epochs (default: 3)
- **N_aug**: Number of synthetic samples (default: 1000)
- **perturb_target_prob**: Probability of perturbing target column (default: 0.5)

### Prerequisites

Before running DPO fine-tuning, ensure you have:
1. Completed the pre-training step
2. Generated synthetic data
3. Located the checkpoint files from pre-training

### Output

The DPO fine-tuning process will generate:
- Fine-tuned model in the specified `output_dir`
- Training logs and metrics
- Evaluation results