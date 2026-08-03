"""
scenarios/imbalanced.py: Imbalanced Data 场景构建
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scenario import BaseScenario


class ImbalancedDataScenario(BaseScenario):
    """Imbalanced Data 场景：将少数类下采样至指定流行率/患病率 (如 1%)"""

    def __init__(self, target_col: str, minority_prev: float = 0.01, seed: int = 42):
        super().__init__(seed=seed)
        self.target_col = target_col
        self.minority_prev = minority_prev

    def build_imbalanced_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """辅助方法：对单个 DataFrame 施加少数类下采样"""
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不存在于数据中。")

        counts = df[self.target_col].value_counts()
        if len(counts) < 2:
            return df.copy()

        minority_class = counts.index[-1]
        majority_class = counts.index[0]

        df_min = df[df[self.target_col] == minority_class]
        df_maj = df[df[self.target_col] == majority_class]

        n_maj = len(df_maj)
        target_min = max(int(np.round(n_maj * self.minority_prev / (1.0 - self.minority_prev))), 1)

        if len(df_min) > target_min:
            df_min = self._subsample(df_min, target_min, seed=self.seed)

        return (
            pd.concat([df_maj, df_min], axis=0)
            .sample(frac=1.0, random_state=self.seed)
            .reset_index(drop=True)
        )

    def build(self, df: pd.DataFrame, full_test: pd.DataFrame | None = None, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """在 Train 和 Test 中同步将少数类下采样至指定比例"""
        if full_test is None:
            full_train, full_test = train_test_split(
                df, test_size=0.2, random_state=self.seed, stratify=df[self.target_col]
            )
        else:
            full_train = df

        train_df = self.build_imbalanced_df(full_train)
        test_df = self.build_imbalanced_df(full_test)

        return train_df, test_df