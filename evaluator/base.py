"""
evaluator/base.py: 评估器抽象接口。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseEvaluator(ABC):
    """评估器抽象基类。"""

    @abstractmethod
    def evaluate(
        self,
        *,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synth_df: pd.DataFrame | None,
        target_col: str,
        seed: int,
    ) -> dict:
        """执行评估，返回结果字典。"""
        ...
