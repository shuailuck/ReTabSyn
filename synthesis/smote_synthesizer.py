import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from synthesis.synthesizer import BaseTabularSynthesizer
# ===========================================================================
# SMOTE 算法的具体实现
# ===========================================================================

class SmoteSynthesizer(BaseTabularSynthesizer):
    """基于类别 Bucket 内 KNN 线性插值的 SMOTE 合成器

    支持 target_classes 定点合成：仅从指定目标列的类别值所在 bucket 合成样本。
    """

    def __init__(self, integer_columns: list = None, random_state: int = 42,
                 k_neighbors: int = 5, target_col: str | None = None,
                 target_classes: list | None = None):
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.k_neighbors = k_neighbors
        self.target_col = target_col
        self.target_classes = target_classes
        self.buckets = []
        self.bucket_probs = None

    def _fit(self, df: pd.DataFrame):
        cat_cols = self.transformer.categorical_cols

        if cat_cols:
            self.buckets = [group.reset_index(drop=True) for _, group in df.groupby(cat_cols)]
        else:
            self.buckets = [df.reset_index(drop=True)]

        # 定点合成：仅保留目标类别的 bucket
        if self.target_col and self.target_classes and self.target_col in cat_cols:
            filtered = []
            for b in self.buckets:
                val = b[self.target_col].iloc[0]
                if val in self.target_classes:
                    filtered.append(b)
            if filtered:
                self.buckets = filtered
                print(f"[SMOTE] 定点合成 target={self.target_col} classes={self.target_classes}, "
                      f"buckets={len(filtered)}/{len(self.buckets)}")

        sizes = np.array([len(b) for b in self.buckets], dtype=float)
        self.bucket_probs = sizes / sizes.sum()

    def _sample(self, n_samples: int) -> pd.DataFrame:
        num_cols = self.transformer.numeric_cols
        cat_cols = self.transformer.categorical_cols

        bucket_choices = np.random.choice(len(self.buckets), size=n_samples, p=self.bucket_probs)

        synthetic_rows = []
        for choice in bucket_choices:
            bucket = self.buckets[choice]
            X_num = bucket[num_cols].values

            if len(bucket) > 1:
                k = min(self.k_neighbors + 1, len(bucket))
                nn = NearestNeighbors(n_neighbors=k).fit(X_num)

                seed_idx = np.random.randint(len(bucket))
                _, neigh_idxs = nn.kneighbors([X_num[seed_idx]], return_distance=True)

                possible = [i for i in neigh_idxs[0] if i != seed_idx]
                neigh_idx = np.random.choice(possible) if possible else seed_idx

                lam = np.random.rand()
                new_num = X_num[seed_idx] + lam * (X_num[neigh_idx] - X_num[seed_idx])
            else:
                new_num = X_num[0]

            new_row = dict(zip(num_cols, new_num))
            for c in cat_cols:
                new_row[c] = bucket[c].iloc[0]

            synthetic_rows.append(new_row)

        return pd.DataFrame(synthetic_rows)


if __name__ == "__main__":

    df = pd.read_csv('./csv/example/wilt_labeled.csv')
    synthesizer = SmoteSynthesizer(random_state=42, k_neighbors=5)
    # Fit & Sample 机制
    synthesizer.fit(df)
    synth_df = synthesizer.sample(n_samples=15)

    print(synth_df)

    synth_df.to_csv()