"""
evaluator/base.py: 评估器抽象接口。

评估器与数据语义无关：只负责「用 train_df 训练下游模型，在 test_df 上评测」。
哪些数据作为 train / test，由 scenario 掌控。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseEvaluator(ABC):
    """评估器抽象基类（无状态、数据无关）。"""

    @abstractmethod
    def evaluate(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        seed: int,
    ) -> dict:
        """在 train_df 上训练下游模型，在 test_df 上评测。

        Returns
        -------
        dict : {模型名: 分数}
        """
        ...
