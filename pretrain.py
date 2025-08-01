import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from be_great import GReaT

from transformers import AutoTokenizer, AutoModelForCausalLM
from custom_begreat import CustomGReaT
from smote import generate_smote_synthetic

from DPOUtils import calculate_max_seq_len, suggest_synthetic_count

def parse_args():
    parser = argparse.ArgumentParser(description='Pre-train GReaT model for tabular data generation')
    parser.add_argument('--llm_name', type=str, default='gpt2',
                        help='LLM model backbone')
    parser.add_argument('--efficient_finetuning', type=str, default="",
                        help='LLM efficient training method.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--test_idx', type=int, default=0,
                        help='Test index for multiple runs')
    parser.add_argument('--num_sample', type=int, default=None,
                        help='Number of samples to use (None for all)')
    parser.add_argument('--duration', type=int, default=1000,
                        help='Training duration/epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from checkpoint')
    parser.add_argument('--split_label', action='store_true',
                        help='Whether the input data contains split labels.')
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help='Whether to resume from checkpoint.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--synth_data_dir', type=str, default='synth_data',
                        help='Directory for saving synthetic data')
    parser.add_argument('--save_steps', type=int, default=30000,
                        help='Steps between saving checkpoints')
    parser.add_argument('--N_aug', type=int, default=-1,
                        help='Number of SMOTE synthetic samples to generate for data augmentation. If negative, use len(train)')
    parser.add_argument('--guided_sampling', action='store_true',
                        help='Whether to use guided sampling.')
    parser.add_argument('--shuffle_data', action='store_true',
                        help='Whether to shuffle data.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.synth_data_dir, exist_ok=True)

    # Setup paths and names
    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    llm_name = args.llm_name
    llm_name_for_saving = llm_name.replace("/", "-")
    output_path = f"{args.synth_data_dir}/{data_name}_GReaT{llm_name_for_saving}_default_{args.test_idx}.csv"

    # Load and prepare data
    data = pd.read_csv(args.data_path)
    if args.num_sample is not None:
        print("subsampling!:", args.num_sample)
        data = data.sample(args.num_sample, random_state=42)
    if args.split_label and 'SOURCE_LABEL' in data.columns:
        print("Splitting based on source label!")
        train = data[data['SOURCE_LABEL'] == 'train'].drop(columns=['SOURCE_LABEL'])
        test = data[data['SOURCE_LABEL'] == 'test'].drop(columns=['SOURCE_LABEL'])
    else:
        train, test = train_test_split(data, test_size=0.2, random_state=42)

    num_sample = len(train) # Use train set size without augmentation
    
    # SMOTE data augmentation
    N_aug = args.N_aug
    if N_aug == -114514:
        N_aug = suggest_synthetic_count(train)
        print(f"Suggesting {N_aug} synthetic samples for data augmentation using dynamic strategy")    
    elif N_aug < 0:
        ratio = abs(N_aug)
        N_aug = len(train) * ratio
        print(f"Suggesting {N_aug} synthetic samples for data augmentation which is {ratio} times of the training set")
        
    if N_aug > 0:
            # Identify integer columns for proper type handling
        integer_columns = []
        for col in train.columns:
            if train[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                integer_columns.append(col)
        
        # Generate synthetic samples
        print("train missing rows:", train.isnull().sum())
        
        synthetic_train = generate_smote_synthetic(
                train, 
                n_samples=N_aug, 
                integer_columns=integer_columns,
                random_state=42 + args.test_idx
            )
            
        # Concatenate original and synthetic data
        print(f"Training set augmented from {len(train) - N_aug} to {len(train)} samples")
        print("synthetic_train missing rows:", synthetic_train.isnull().sum())
        train = pd.concat([train, synthetic_train], ignore_index=True)
        print(f"Training set augmented from {len(train) - N_aug} to {len(train)} samples")
    
    print("data dimension:")
    print(data)
    print(data.columns)
    print(data.shape)
    print(train.shape)
    # Use first column as condition column, instead of target column
    # This aligns with the DPO training where target column correctness is used for reward
    cond_col = train.columns[0]
    print("columns in train:", train.columns)

    # Calculate max sequence length
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    max_seq_len = calculate_max_seq_len(train, tokenizer)
    print("max_seq_len:", max_seq_len)

    # Initialize GReaT model
    custom_modules = ["q_proj", "v_proj"]

    model_great = CustomGReaT(
        llm=args.llm_name,
        efficient_finetuning=args.efficient_finetuning,
        batch_size=args.batch_size,
        epochs=args.duration,
        fp16=True,
        custom_target_modules=custom_modules,
        save_steps=args.save_steps
    )

    # Load checkpoint if exists and continuing training
    checkpoint_path = f"{args.checkpoint_dir}/great{llm_name_for_saving}_checkpoint_{data_name}_{args.test_idx}"
    base_model_path = f"{args.checkpoint_dir}/trained_base_{llm_name_for_saving}_{data_name}_{args.test_idx}"
    
    if os.path.exists(checkpoint_path) and args.continue_training:
        print(f"Loading checkpoint from {checkpoint_path}")
        model_great.load_from_dir(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model_great._update_column_information(train)
        model_great.model = base_model

    # Train model
    if not os.path.exists(checkpoint_path) or args.continue_training:
        print("Fitting!")
        model_great.fit(train,resume_from_checkpoint=args.resume_from_checkpoint,conditional_col=cond_col, shuffle_data=args.shuffle_data)

    # Save model and checkpoints
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
        model_great.save(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        base_model = model_great.model
        base_model.save_pretrained(base_model_path)
        print(f"Saved base model to {base_model_path}")

    # Generate synthetic data
    synthetic_data = model_great.sample(n_samples=num_sample, max_length=max_seq_len, guided_sampling=args.guided_sampling)
    
    # Save synthetic data
    synthetic_data.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")

if __name__ == "__main__":
    main() 