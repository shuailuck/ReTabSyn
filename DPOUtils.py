import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from trl import DPOTrainer, DPOConfig
from sklearn.preprocessing import LabelEncoder



import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from typing import Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from trl import DPOTrainer, DPOConfig

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not available. Install with: pip install tabpfn")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

def calculate_max_seq_len(train, tokenizer):
    row1 = train.iloc[0,:]
    text = ', '.join([f"{c} is {row1[c]}" for c in train.columns])
    tokens = tokenizer.encode(text)
    return int(len(tokens) * 1.5)


def suggest_synthetic_count(df: pd.DataFrame) -> int:
    """
    Suggest the number of synthetic rows to add based on dataset size:
    
    - N <= 128: suggest N * 30
    - N <= 256: suggest N * 10  
    - N <= 1000: suggest N * 5
    - N > 1000: suggest N * 1
    
    Where N is the number of existing rows in the dataset.
    
    Returns:
      - int: suggested synthetic rows to generate
    """
    N = len(df)
    
    if N <= 128:
        return N * 30
    elif N <= 256:
        return N * 10
    elif N <= 1000:
        return N * 5
    else:
        return N * 1

class DPOPositiveTrainer(DPOTrainer):
    def __init__(self, *args, lambda_reg=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        dpo_loss, chosen_rewards, rejected_rewards = super().dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        reg_term = self.lambda_reg * torch.relu(reference_chosen_logps - policy_chosen_logps)
        dpo_positive_loss = dpo_loss + reg_term
        return dpo_positive_loss, chosen_rewards, rejected_rewards

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to catch and debug tokenization errors."""
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "Expected tensor for argument #1 'indices'" in str(e):
                print(f"\n=== TOKENIZATION ERROR DETECTED ===")
                print(f"Error: {str(e)}")
                print(f"Batch keys: {inputs.keys()}")
                
                # Print input_ids info
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    print(f"input_ids shape: {input_ids.shape}")
                    print(f"input_ids dtype: {input_ids.dtype}")
                    print(f"input_ids min: {input_ids.min()}, max: {input_ids.max()}")
                    
                    # Try to decode problematic sequences
                    for i, seq in enumerate(input_ids[:3]):  # Check first 3 sequences
                        try:
                            decoded = self.tokenizer.decode(seq)
                            print(f"Sequence {i}: {decoded[:200]}...")
                        except Exception as decode_error:
                            print(f"Sequence {i}: Failed to decode - {decode_error}")
                
                print(f"=== END ERROR DEBUG ===\n")
            raise e
        
class CustomLabelEncoder(LabelEncoder):
    def __init__(self, unknown_label='unknown'):
        super().__init__()
        self.unknown_label = unknown_label
        self.classes_ = None

    def fit(self, y):
        super().fit(np.append(y, self.unknown_label))
        return self

    def transform(self, y):
        y = np.array(y)
        unseen_mask = ~np.isin(y, self.classes_)
        y[unseen_mask] = self.unknown_label
        return super().transform(y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        y = super().inverse_transform(y)
        y[y == self.classes_.size - 1] = None
        return y
    
def update_incorrect_predictions(new_df, differences, predictions, target_column):
    incorrect_rows = new_df.loc[differences].copy()
    incorrect_rows[target_column] = predictions[differences]
    return incorrect_rows
    
def find_optimal_threshold(model, X_train, y_train, task_type, cv=5):
    """Find optimal threshold using cross-validation and precision-recall curve"""
    if task_type == 'binary':
        # Get cross-validated probabilities
        y_proba_cv = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_train, y_proba_cv)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        
        # Find threshold that maximizes F1
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]
        
        print(f"Optimal threshold found: {best_threshold:.4f} (CV F1: {best_f1:.4f})")
        return best_threshold
    else:
        # For multiclass, use default threshold (0.5 equivalent)
        print("Multiclass classification: using default threshold")
        return 0.5


def train_and_predict(input_df, target_column, categorical_columns=None, model_type='xgboost', custom_param_grid=None):
    df = input_df.copy()
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for column in categorical_columns:
        le = CustomLabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Determine target type
    is_classification = len(y.value_counts()) <= 2 or df[target_column].dtype == 'object'
    
    if is_classification:
        task_type = "multi_class" if len(y.value_counts()) > 2 else "binary"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(eval_metric='mlogloss')
            param_grid = custom_param_grid if custom_param_grid else {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'tabpfn':
            if not TABPFN_AVAILABLE:
                raise ImportError("TabPFN is not available. Please install it with: pip install tabpfn")
            model = TabPFNClassifier()
            param_grid = {}  # TabPFN doesn't require hyperparameter tuning
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is not available. Please install it with: pip install catboost")
            model = CatBoostClassifier(random_state=42, verbose=False, eval_metric='AUC')
            param_grid = custom_param_grid if custom_param_grid else {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
        elif model_type == 'regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = custom_param_grid if custom_param_grid else {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            raise ValueError(f"Unsupported model type for classification: {model_type}")

        scoring = "f1" if task_type == 'binary' else "f1_macro"
        
        if model_type == 'tabpfn':
            # TabPFN doesn't need hyperparameter tuning
            model.fit(X_train, y_train)
            best_model = model
        else:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scoring)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        # Find optimal threshold using cross-validation
        optimal_threshold = find_optimal_threshold(best_model, X_train, y_train, task_type)

        # Predict probabilities and apply optimal threshold
        y_proba = best_model.predict_proba(X_test)
        
        if task_type == 'binary':
            y_pred = (y_proba[:, 1] >= optimal_threshold).astype(int)
        else:
            y_pred = best_model.predict(X_test)  # Use default prediction for multiclass
        
        average = 'binary' if task_type == 'binary' else "macro"
        f1 = f1_score(y_test, y_pred, average=average)
        
        # Calculate AUROC
        if task_type == 'binary':
            auroc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auroc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        
        print(f"Test F1 (optimal threshold): {f1:.4f}")
        print(f"Test AUROC: {auroc:.4f}")
        
        # Log best parameters, F1 score, AUROC, and optimal threshold
        with open('model_performance_log.csv', 'a') as f:
            if model_type == 'tabpfn':
                best_params = 'no_hyperparams'
            else:
                best_params = ','.join([f"{k}={v}" for k, v in grid_search.best_params_.items()])
            f.write(f"{model_type},{best_params},f1={f1:.4f},auroc={auroc:.4f},threshold={optimal_threshold:.4f}\n")

    else:
        # Regression logic
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'xgboost':
            model = xgb.XGBRegressor(eval_metric='rmse')
            param_grid = custom_param_grid if custom_param_grid else {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'tabpfn':
            # Note: TabPFN primarily supports classification, not regression
            raise ValueError("TabPFN does not support regression tasks. Use classification models instead.")
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is not available. Please install it with: pip install catboost")
            model = CatBoostRegressor(random_state=42, verbose=False, eval_metric='RMSE')
            param_grid = custom_param_grid if custom_param_grid else {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
        elif model_type == 'regression':
            model = LinearRegression()
            param_grid = custom_param_grid if custom_param_grid else {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        else:
            raise ValueError(f"Unsupported model type for regression: {model_type}")

        scoring = "neg_mean_squared_error"
        
        if model_type == 'regression' and isinstance(model, LinearRegression):
            # Linear regression doesn't need extensive hyperparameter tuning
            model.fit(X_train, y_train)
            best_model = model
        else:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scoring)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Test RMSE: {rmse}")
        
        # Set default threshold for regression (not used but needed for predict function)
        optimal_threshold = 0.5
        
        # Log best parameters and RMSE
        with open('model_performance_log.csv', 'a') as f:
            if model_type == 'regression' and isinstance(model, LinearRegression):
                best_params = 'basic_linear_regression'
            else:
                best_params = ','.join([f"{k}={v}" for k, v in grid_search.best_params_.items()])
            f.write(f"{model_type},{best_params},rmse={rmse:.4f}\n")

    def predict(new_df, target_column=None):
        new_df_copy = new_df.copy()
        for column in categorical_columns:
            le = label_encoders[column]
            new_df_copy[column] = le.transform(new_df_copy[column])
        
        if is_classification and task_type == 'binary':
            # Use optimal threshold for binary classification
            proba = best_model.predict_proba(new_df_copy.drop(columns=target_column) if target_column else new_df_copy)
            predictions = (proba[:, 1] >= optimal_threshold).astype(int)
        else:
            # Use default prediction for regression or multiclass
            predictions = best_model.predict(new_df_copy.drop(columns=target_column) if target_column else new_df_copy)
        
        if target_column:
            # CRITICAL FIX: Encode the target column for proper comparison
            if target_column in label_encoders:
                # Target is categorical - encode it for comparison with predictions
                encoded_target = label_encoders[target_column].transform(new_df[target_column])
                differences = new_df.index[encoded_target != predictions].tolist()
            else:
                # Target is numerical - compare directly
                differences = new_df.index[new_df[target_column] != predictions].tolist()
            return predictions, differences
        return predictions

    return best_model, label_encoders, predict