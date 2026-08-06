import argparse
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import  DPOConfig
from scipy.special import softmax

from be_great import GReaT

from smote import generate_smote_synthetic

from DPOUtils import suggest_synthetic_count, calculate_max_seq_len, DPOPositiveTrainer


def create_perturbed_dataset(df, target, p=0.1, shuffle=False, tau=1.0, min_corr_threshold=0.1, Q=10, perturb_target_prob=0.5):
    """
    Create perturbed dataset based on correlation between column pairs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        p (float): Probability of selecting additional pair
        shuffle (bool): Whether to shuffle the dataset
        tau (float): Temperature parameter for softmax
        min_corr_threshold (float): Minimum correlation threshold
        Q (int): Number of quantiles for numerical columns
        perturb_target_prob (float): Probability of perturbing only target column vs correlation-based perturbation
    """
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Calculate correlation matrix
    from feature_correlation import create_combined_correlation_matrix
    # Create a copy to avoid modifying the original dataframe
    df = df.copy().dropna().reset_index(drop=True)
    corr_matrix = create_combined_correlation_matrix(df)
    abs_corr = np.abs(corr_matrix.values)
    print("Correlation matrix shape:", corr_matrix.shape)
    print("Correlation matrix:", corr_matrix)
    
    # Step 2: Convert correlations to sampling probabilities
    # Fill NaN values with 0
    abs_corr = np.nan_to_num(abs_corr, nan=0.0)
    # Zero out diagonal first (self-correlations = 1.0 should not be sampled)
    np.fill_diagonal(abs_corr, 0)
    # Zero out correlations below threshold
    abs_corr[abs_corr < min_corr_threshold] = 0
    print("abs_corr:", abs_corr)
    
    # Check if we have any valid correlations left
    if abs_corr.sum() == 0:
        print("Warning: No correlations above threshold found. Using uniform sampling.")
        abs_corr = np.ones_like(abs_corr)
        np.fill_diagonal(abs_corr, 0)
    
    # Apply softmax with temperature to get probabilities
    probs = softmax(abs_corr.flatten() / tau)
    probs = probs.reshape(abs_corr.shape)
    
    print("Min prob:", probs.min(), "Max prob:", probs.max(), "Sum:", probs.sum(), "pairs.shape:", probs.shape)
    
    # Convert to list of (i,j) pairs with their probabilities
    n_cols = len(df.columns)
    pairs = []
    pair_probs = []
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j and probs[i,j] > 0:
                pairs.append((i,j))
                pair_probs.append(probs[i,j])
    pair_probs = np.array(pair_probs) / sum(pair_probs)
    
    # Pre-compute quantile boundaries for numerical columns
    quantile_bounds = {}
    for col in df.columns:
        if df[col].dtype != 'object' and len(df[col].unique()) > 15:
            # Create quantile boundaries for numerical columns
            quantile_bounds[col] = pd.qcut(df[col], Q, retbins=True, duplicates='drop')[1]
    
    data = []
    for idx, row in df.iterrows():
        # Decide whether to perturb target only or use correlation-based perturbation
        perturb_target_only = random.random() < perturb_target_prob
        
        if perturb_target_only:
            # Perturb only the target column
            perturbed_row = row.copy()
            
            # Apply same perturbation logic to target column
            if df[target].dtype == 'object' or len(df[target].unique()) <= 15:
                # Categorical perturbation
                current_value = row[target]
                other_values = df[target][df[target] != current_value].unique()
                if len(other_values) > 0:
                    perturbed_row[target] = np.random.choice(other_values)
            else:
                # Numerical perturbation - randomly select different quantile
                if target in quantile_bounds:
                    bounds = quantile_bounds[target]
                    actual_quantiles = len(bounds) - 1
                    
                    # Find current quantile
                    current_value = row[target]
                    current_quantile = None
                    for q in range(actual_quantiles):
                        if bounds[q] <= current_value <= bounds[q + 1]:
                            current_quantile = q
                            break
                    
                    # If we couldn't find the quantile (edge case), assign randomly
                    if current_quantile is None:
                        current_quantile = np.random.randint(0, actual_quantiles)
                    
                    # Randomly select a different quantile
                    other_quantiles = list(range(actual_quantiles))
                    if current_quantile in other_quantiles:
                        other_quantiles.remove(current_quantile)
                    
                    if len(other_quantiles) > 0:
                        new_quantile = np.random.choice(other_quantiles)
                        
                        # Get value range for new quantile
                        lower_bound = bounds[new_quantile]
                        upper_bound = bounds[new_quantile + 1]
                        
                        # Sample uniform value from new quantile range
                        new_value = np.random.uniform(lower_bound, upper_bound)
                        # Round to reasonable precision to avoid tokenization issues
                        if isinstance(new_value, (int, np.integer)):
                            perturbed_row[target] = int(new_value)
                        else:
                            perturbed_row[target] = round(float(new_value), 6)
            
            # For target perturbation, split before target column
            target_idx = df.columns.get_loc(target)
            prompt_cols = df.columns[:target_idx]
            prompt_str = ", ".join([f"{col} is {row[col]}" for col in prompt_cols])
            
            # Create chosen string (target with real value)
            chosen_str = f"{target} is {row[target]}"
            
            # Create rejected string (target with perturbed value)
            val = perturbed_row[target]
            # Handle potential NaN/inf values (only for numeric types)
            if pd.isna(val) or (isinstance(val, (int, float, np.number)) and np.isinf(val)):
                val = row[target]  # Fall back to original value
            
            # Format numeric values to prevent very long decimals
            if isinstance(val, (float, np.floating)):
                val = round(float(val), 4)
                if val == int(val):
                    val = int(val)
            elif isinstance(val, (int, np.integer)):
                val = int(val)
            
            rejected_str = f"{target} is {val}"
            
        else:
            # Use existing correlation-based perturbation logic
            # Step 3: Sample column pairs
            chosen_pairs = []
            # Always pick one pair
            pair_idx = np.random.choice(len(pairs), p=pair_probs)
            chosen_pairs.append(pairs[pair_idx])
            # Additional pair with probability p
            if random.random() < p:
                pair_idx = np.random.choice(len(pairs), p=pair_probs)
                chosen_pairs.append(pairs[pair_idx])
            
            # If shuffle is True, create shuffled column order
            if shuffle:
                # Get unique column names involved in chosen pairs
                involved_cols = set()
                for i, j in chosen_pairs:
                    involved_cols.add(df.columns[i])
                    involved_cols.add(df.columns[j])
                involved_cols = list(involved_cols)
                
                # Create shuffled column order for all columns
                shuffled_columns = df.columns.tolist()
                np.random.shuffle(shuffled_columns)
                
                # Create mapping from original to shuffled indices
                original_to_shuffled = {col: shuffled_columns.index(col) for col in df.columns}
                
                # Update chosen pairs to use shuffled indices
                shuffled_pairs = []
                for i, j in chosen_pairs:
                    orig_col_i = df.columns[i]
                    orig_col_j = df.columns[j]
                    shuffled_i = original_to_shuffled[orig_col_i]
                    shuffled_j = original_to_shuffled[orig_col_j]
                    shuffled_pairs.append((shuffled_i, shuffled_j))
                chosen_pairs = shuffled_pairs
                
                # Use shuffled column order for this row
                current_columns = shuffled_columns
            else:
                current_columns = df.columns.tolist()
            
            # Step 4: Perturb chosen pairs
            perturbed_row = row.copy()
            for i, j in chosen_pairs:
                col_i = current_columns[i]
                col_j = current_columns[j]
                
                # Determine which column to perturb (right one in pair)
                to_perturb_col = col_j
                
                if df[to_perturb_col].dtype == 'object' or len(df[to_perturb_col].unique()) <= 15:
                    # Categorical perturbation
                    current_value = row[to_perturb_col]
                    other_values = df[to_perturb_col][df[to_perturb_col] != current_value].unique()
                    if len(other_values) == 0:
                        continue  # 该列只有单一值，无法扰动
                    perturbed_row[to_perturb_col] = np.random.choice(other_values)
                else:
                    # Numerical perturbation - randomly select different quantile
                    bounds = quantile_bounds[to_perturb_col]
                    actual_quantiles = len(bounds) - 1
                    
                    # Find current quantile
                    current_value = row[to_perturb_col]
                    current_quantile = None
                    for q in range(actual_quantiles):
                        if bounds[q] <= current_value <= bounds[q + 1]:
                            current_quantile = q
                            break
                    
                    # If we couldn't find the quantile (edge case), assign randomly
                    if current_quantile is None:
                        current_quantile = np.random.randint(0, actual_quantiles)
                    
                    # Randomly select a different quantile
                    other_quantiles = list(range(actual_quantiles))
                    if current_quantile in other_quantiles:
                        other_quantiles.remove(current_quantile)
                    new_quantile = np.random.choice(other_quantiles)
                    
                    # Get value range for new quantile
                    lower_bound = bounds[new_quantile]
                    upper_bound = bounds[new_quantile + 1]
                    
                    # Sample uniform value from new quantile range
                    new_value = np.random.uniform(lower_bound, upper_bound)
                    # Round to reasonable precision to avoid tokenization issues
                    if isinstance(new_value, (int, np.integer)):
                        perturbed_row[to_perturb_col] = int(new_value)
                    else:
                        perturbed_row[to_perturb_col] = round(float(new_value), 6)

            # Create dataset entry once per row (outside the perturbation loop)
            # Use the minimum perturbation index to determine split point
            # + 1 becuase we want to include the non-perturbed column in prompt.
            # Use original column order for determining split point
            if shuffle:
                # Map back to original column indices for split point calculation
                original_pairs = []
                for i, j in chosen_pairs:
                    shuffled_col_i = current_columns[i]
                    shuffled_col_j = current_columns[j]
                    orig_i = df.columns.get_loc(shuffled_col_i)
                    orig_j = df.columns.get_loc(shuffled_col_j)
                    original_pairs.append((orig_i, orig_j))
                min_perturbed_idx = min([min(i,j) for i,j in original_pairs]) + 1
            else:
                min_perturbed_idx = min([min(i,j) for i,j in chosen_pairs]) + 1
            
            prompt_cols = df.columns[:min_perturbed_idx]
            prompt_str = ", ".join([f"{col} is {row[col]}" for col in prompt_cols])

            # Create chosen string (remaining columns with real values)
            chosen_cols = df.columns[min_perturbed_idx:]
            chosen_parts = []
            for col in chosen_cols:
                val = row[col]
                # Format numeric values to prevent very long decimals
                if isinstance(val, (float, np.floating)):
                    val = round(float(val), 4)
                    if val == int(val):
                        val = int(val)
                elif isinstance(val, (int, np.integer)):
                    val = int(val)
                chosen_parts.append(f"{col} is {val}")
            chosen_str = ", ".join(chosen_parts)
            
            # Create rejected string (remaining columns with perturbed values)
            # Format values properly to avoid tokenization issues
            rejected_parts = []
            for col in chosen_cols:
                val = perturbed_row[col]
                # Handle potential NaN/inf values (only for numeric types)
                if pd.isna(val) or (isinstance(val, (int, float, np.number)) and np.isinf(val)):
                    val = row[col]  # Fall back to original value
                
                # Format numeric values to prevent very long decimals
                if isinstance(val, (float, np.floating)):
                    # Limit to 4 decimal places for floating point numbers
                    val = round(float(val), 4)
                    # Convert to int if it's effectively a whole number
                    if val == int(val):
                        val = int(val)
                elif isinstance(val, (int, np.integer)):
                    val = int(val)
                
                # Additional validation for problematic values
                val_str = str(val)
                if len(val_str) > 50:  # Very long string representation
                    print(f"WARNING: Very long value in {col}: {val_str[:50]}...")
                    val = row[col]  # Fall back to original
                    if isinstance(val, (float, np.floating)):
                        val = round(float(val), 4)
                
                rejected_parts.append(f"{col} is {val}")
            rejected_str = ", ".join(rejected_parts)

        data.append({
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str
        })
        
        if idx % 1000 == 0:
            print(f"Creating perturbed dataset: {idx} / {len(df)}")
        
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    return dataset

def main(args):
    # Set environment variable
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Load and prepare data
    data_name = args.data_path.split('/')[-1].split('.')[0]
    data = pd.read_csv(args.data_path)
    
    if 'SOURCE_LABEL' in data.columns:
        print("Splitting based on source label!")
        train = data[data['SOURCE_LABEL'] == 'train'].drop(columns=['SOURCE_LABEL']).reset_index(drop=True)
        test = data[data['SOURCE_LABEL'] == 'test'].drop(columns=['SOURCE_LABEL']).reset_index(drop=True)
    else:
        train, test = train_test_split(data, test_size=0.2, random_state=42+args.test_idx)
    target_column = train.columns[-1]
    cond_col = train.columns[0]
    original_N = len(train)
    print("Data dimensions:", train.shape, test.shape)

    # SMOTE data augmentation
    N_aug = args.N_aug
    if N_aug == -114514:
        N_aug = suggest_synthetic_count(train)
        print(f"Suggesting {N_aug} synthetic samples for data augmentation using dynamic strategy")
    elif N_aug < 0:
        ratio = abs(N_aug)
        N_aug = len(train) * ratio
        print(f"Suggesting {N_aug} synthetic samples for data augmentation which is {ratio} times of the training set")
        
    assert N_aug >= 0, "N_aug must be non-negative"
        
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

    # Calculate max sequence length
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    max_seq_len = calculate_max_seq_len(train, tokenizer)
    print("max_seq_len:", max_seq_len)

    # load a trained GReaT model
    great_model_before_rl = GReaT(
        llm=args.llm,
        max_length=max_seq_len
    )
    print(f"Loading checkpoint from {args.checkpoint_path}")
    great_model_before_rl.load_from_dir(args.checkpoint_path)
    great_model_before_rl.model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    great_model_before_rl._update_column_information(train)
    great_model_before_rl._update_conditional_information(train, cond_col)
    
    # Create initial perturbed dataset
    if args.dpo_data == 0:
        print(f"Creating perturbed dataset with split ratio {args.split_ratio}")
        dataset = create_perturbed_dataset(train, target_column, args.split_ratio, args.shuffle==1, perturb_target_prob=args.perturb_target_prob)
        
        # Apply class balancing if minor_to_major_ratio is specified
        if args.minor_to_major_ratio != -1:
            print(f"Applying class balancing with minor_to_major_ratio: {args.minor_to_major_ratio}")
            
            # Determine majority class from training data
            target_counts = train[target_column].value_counts()
            majority_class = target_counts.index[0]  # Most frequent class
            minority_classes = target_counts.index[1:].tolist()  # All other classes
            
            print(f"Majority class: {majority_class} (count: {target_counts[majority_class]})")
            print(f"Minority classes: {minority_classes} (counts: {[target_counts[cls] for cls in minority_classes]})")
            
            # Create mapping from dataset index to original target class
            # Since dataset is created row by row from train, indices match
            target_mapping = train[target_column].tolist()
            
            # Separate majority and minority samples from dataset
            majority_indices = []
            minority_indices = []
            
            for idx, target_val in enumerate(target_mapping):
                if target_val == majority_class:
                    majority_indices.append(idx)
                else:
                    minority_indices.append(idx)
            
            print(f"Dataset distribution - Majority: {len(majority_indices)}, Minority: {len(minority_indices)}")
            
            # Calculate how many majority samples to keep
            total_minority = len(minority_indices)
            desired_majority = int(total_minority / args.minor_to_major_ratio)
            
            if desired_majority < len(majority_indices):
                print(f"Downsampling majority class from {len(majority_indices)} to {desired_majority} samples")
                
                # Randomly sample majority indices
                import random
                random.seed(42 + args.test_idx)
                sampled_majority_indices = random.sample(majority_indices, desired_majority)
                
                # Combine minority and downsampled majority indices
                balanced_indices = minority_indices + sampled_majority_indices
                
                # Create balanced dataset
                balanced_data = []
                for idx in balanced_indices:
                    balanced_data.append(dataset[idx])
                
                # Shuffle the balanced dataset
                random.shuffle(balanced_data)
                
                # Convert back to Dataset
                dataset = Dataset.from_list(balanced_data)
                
                print(f"Balanced dataset created with {len(dataset)} samples")
                print(f"Final ratio - Minority: {len(minority_indices)}, Majority: {len(sampled_majority_indices)}")
            else:
                print(f"No downsampling needed. Majority class already has {len(majority_indices)} <= {desired_majority} samples")
    
    # Combine datasets
    if args.dpo_data == 0:
        dpo_dataset = dataset
    else:
        raise ValueError(f"Invalid value for dpo_data: {args.dpo_data}")
    
    # Split into train and test
    print(f"Splitting dataset into train and test with ratio {args.test_train_ratio}")
    train_dataset, test_dataset = dpo_dataset.train_test_split(
        test_size=args.test_train_ratio
    ).values()
    
    # Initialize tokenizer and models
    print(f"Initializing tokenizer and models for DPO training")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    model_ref = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    
    # Configure DPO training with disabled shuffling for debugging
    training_args = DPOConfig(
        beta=args.beta,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        logging_steps=10,  # More frequent logging
        save_steps=100,    # Save more frequently in case of crash
        # learning_rate=5e-5,  # Add proper DPO learning rate (10-100x smaller than SFT)
        # lr_scheduler_type="cosine",  # Add learning rate scheduler
        # warmup_ratio=0.1,  # Add warmup for stable training
        # max_grad_norm=0.3,  # Add gradient clipping for stability
    )
    
    # Initialize and train DPO trainer
    dpo_trainer = DPOPositiveTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        lambda_reg=args.lambda_reg
    )
    
    # Train and save model
    dpo_trainer.train()
    dpo_trainer.save_model()
    
    # Load fine tuned model in GReaT
    # Generate new synthetic data
    #model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    great_model_before_rl.model = model
    new_synthetic_data = great_model_before_rl.sample(
        n_samples=original_N,
        max_length=max_seq_len
    )
    
    # Save final synthetic data
    os.makedirs(args.synth_data_dir, exist_ok=True)
    new_synthetic_data.to_csv(
        os.path.join(args.synth_data_dir, f"{data_name}_GReaTDPOFeatureTarget_default_{args.test_idx}.csv"),
        index=False
    )
    print(f"Synthetic data after DPO training saved to: {args.synth_data_dir}/{data_name}_GReaTDPOFeatureTarget_default_{args.test_idx}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPO Training Script')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--test_idx', type=int, default=0, help='Test index for file naming')
    parser.add_argument('--synth_data_dir', type=str, required=True, help='Directory to save synthetic data')
    
    # Model parameters
    parser.add_argument('--llm', type=str, default='gpt2', help='Base language model to use') # checkpoint_path
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained GReaT model checkpoint')
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to base model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
    
    # Training parameters
    parser.add_argument('--split_ratio', type=float, default=1, help='Split ratio for perturbed dataset')
    parser.add_argument('--shuffle', type=int, default=0, help='Shuffle the dataset rows or not.')
    parser.add_argument('--syn_to_train_ratio', type=int, default=20, help='Synthetic to training data ratio')
    parser.add_argument('--test_train_ratio', type=float, default=0.2, help='Train/test split ratio')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter for DPO training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Lambda regularization parameter')
    parser.add_argument('--dpo_data', type=int, default=2, help='Mode for using real / synthetic data in DPO training: 0 is real, 1 is synthetic corrected, 2 is both.')
    parser.add_argument('--N_aug', type=int, default=-1,
                        help='Number of SMOTE synthetic samples to generate for data augmentation. If negative, use len(train)')
    parser.add_argument('--perturb_target_prob', type=float, default=0.5, 
                        help='Probability of perturbing only target column vs correlation-based perturbation')
    parser.add_argument('--minor_to_major_ratio', type=float, default=-1, 
                        help='Ratio of minority to majority class in dataset. If -1, no balancing is applied')
    
    # Other parameters
    parser.add_argument('--max_seq_len', type=int, required=True, help='Maximum sequence length')
    
    # Add new model type argument
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'decision_tree'],
                      help='Type of model to use for training (xgboost or decision_tree)')
    
    args = parser.parse_args()
    
    main(args)