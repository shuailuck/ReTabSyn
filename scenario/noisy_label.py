"""
scenario/noisy_label.py: Noisy Label 场景。

训练集标签按 noise_ratio 概率翻转，测试集保持干净。
场景额外维护 clean_train（未加噪训练集），用于评估噪声带来的性能损失上界。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scenario.scenario import BaseScenario


class NoisyLabelScenario(BaseScenario):
    """Noisy Label 场景：训练集标签按 noise_ratio 概率翻转。"""

    def __init__(self, target_col: str, noise_ratio: float = 0.1, seed: int = 42, save_config=None):
        super().__init__(seed=seed, save_config=save_config)
        self.target_col = target_col
        self.noise_ratio = noise_ratio
        self.clean_train: pd.DataFrame | None = None

    def _build(self, df: pd.DataFrame, full_test: pd.DataFrame | None = None,
               **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不存在于数据中。")

        if full_test is None:
            stratify = df[self.target_col]
            full_train, full_test = train_test_split(
                df, test_size=0.2, random_state=self.seed, stratify=stratify
            )
        else:
            full_train = df

        train_df = full_train.copy().reset_index(drop=True)
        test_df = full_test.copy().reset_index(drop=True)

        # 保存未加噪的干净训练集
        self.clean_train = train_df.copy()

        # 对训练集标签加噪
        train_df = self._flip_labels(train_df)

        return train_df, test_df

    def evaluate(self, target_col: str) -> dict:
        """在标准评估基础上，追加 clean baseline。"""
        results = super().evaluate(target_col)

        if self.clean_train is not None:
            results["clean"] = self.evaluator.evaluate(
                train_df=self.clean_train, test_df=self.test_df,
                target_col=target_col, seed=self.seed,
            )
        return results

    def _flip_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 noise_ratio 概率翻转训练集标签。"""
        y = df[self.target_col]
        classes = np.unique(y)
        if len(classes) < 2:
            return df  # 单类别无法翻转

        rng = np.random.RandomState(self.seed)
        n_flip = int(len(df) * self.noise_ratio)
        flip_indices = rng.choice(len(df), size=n_flip, replace=False)

        for idx in flip_indices:
            current = y.iloc[idx]
            other_classes = [c for c in classes if c != current]
            new_label = rng.choice(other_classes)
            df.loc[idx, self.target_col] = new_label

        return df
