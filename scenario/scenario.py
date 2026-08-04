"""
scenarios/base.py: 所有场景的公共抽象基类及通用采样工具
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseScenario(ABC):
    """场景构建器抽象基类 (公共接口)"""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def build(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建并返回切分后的 (train_df, test_df)

        :param df: 输入的完整原始 DataFrame
        :return: (train_df, test_df) 组装好的训练集与测试集
        """
        pass

    @staticmethod
    def _subsample(df: pd.DataFrame, n: int, seed: int, target_col: str | None = None) -> pd.DataFrame:
        """公共辅助工具：安全无放回随机/分层子采样"""
        if len(df) <= n:
            return df.copy().reset_index(drop=True)

        # 针对分类任务，优先使用分层采样 (Stratified Subsampling) 防止抽样丢掉类别
        if target_col and target_col in df.columns and df[target_col].nunique() < 20:
            frac = n / len(df)
            sub_df, _ = train_test_split(
                df,
                train_size=frac,
                random_state=seed,
                stratify=df[target_col]
            )
            return sub_df.reset_index(drop=True)

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(df), size=n, replace=False)
        return df.iloc[indices].reset_index(drop=True)