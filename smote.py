import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def generate_smote_synthetic(
    df: pd.DataFrame,
    n_samples: int,
    integer_columns: list = None,
    random_state: int = 42,
    k_neighbors: int = 5
) -> pd.DataFrame:
    """
    Generate synthetic samples using a SMOTE-like linear interpolation
    for numeric columns within homogeneous categorical buckets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing mixed numeric and categorical columns.
        Columns with dtype 'object' or 'category' are treated as categorical;
        all others as numeric.
    n_samples : int
        Number of synthetic samples to generate.
    integer_columns : list, optional (default=None)
        List of column names that should be rounded to nearest integer in output.
        If None, uses dtype-based inference for integer columns.
    random_state : int, optional (default=42)
        Seed for reproducibility.
    k_neighbors : int, optional (default=5)
        Number of neighbors to consider for interpolation in each bucket.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_samples, df.shape[1]) containing synthetic rows.
    """
    np.random.seed(random_state)

    # Handle default value for integer_columns
    if integer_columns is None:
        integer_columns = []
    if any(df.isnull().sum() > 0):
        print("Dropping missing rows:", df.isnull().sum())
        df = df.dropna()

    # Identify numeric vs. categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    # Create buckets by categorical signature
    if categorical_cols:
        buckets = [group for _, group in df.groupby(categorical_cols)]
        sizes = np.array([len(b) for b in buckets], dtype=float)
        probs = sizes / sizes.sum()
        bucket_choices = np.random.choice(len(buckets), size=n_samples, p=probs)
    else:
        buckets = [df]
        bucket_choices = np.zeros(n_samples, dtype=int)

    synthetic_rows = []
    for choice in bucket_choices:
        bucket = buckets[choice].reset_index(drop=True)
        X_num = bucket[numeric_cols].values

        if len(bucket) > 1:
            # Fit k-NN on numeric columns
            nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(bucket)))
            nn.fit(X_num)

            # Pick a random seed index
            seed_idx = np.random.randint(len(bucket))
            # Find neighbors (including itself)
            _, neigh_idxs = nn.kneighbors([X_num[seed_idx]], return_distance=True)
            # Exclude self-index
            possible = [i for i in neigh_idxs[0] if i != seed_idx]
            if not possible:
                neigh_idx = seed_idx
            else:
                neigh_idx = np.random.choice(possible)

            # Linear interpolation
            lam = np.random.rand()
            new_num = X_num[seed_idx] + lam * (X_num[neigh_idx] - X_num[seed_idx])
        else:
            # If only one row, replicate it
            new_num = X_num[0]

        # Snap numeric values to valid types/ranges
        new_row = {}
        for col, val in zip(numeric_cols, new_num):
            if col in integer_columns:
                new_row[col] = int(round(val))
            else:
                new_row[col] = float(val)

        # Copy categorical columns unchanged
        for col in categorical_cols:
            new_row[col] = bucket[col].iloc[0]

        synthetic_rows.append(new_row)

    return pd.DataFrame(synthetic_rows)

# Example usage (uncomment to run):
df = pd.DataFrame({
    'Age': [25, 40, 60],
    'Charge': [3000.0, 7000.5, 15000.0],
    'Gender': ['M', 'F', 'M'],
    'Smoker': ['Yes', 'No', 'Yes']
})
integer_columns = ['Age']
synth = generate_smote_synthetic(df, n_samples=5, integer_columns=integer_columns, random_state=42)
print(synth)
