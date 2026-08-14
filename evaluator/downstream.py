"""
evaluator/downstream.py: 下游机器学习评估器。

真实数据、合成数据、真实+合成数据在下游多个分类模型上的性能评估。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from evaluator.base import BaseEvaluator
from evaluator.utility import evaluate_all_modes


class DownstreamEvaluator(BaseEvaluator):
    """标准下游评估器：real / synthetic / augment 三种训练模式。"""

    def __init__(self, metric: str = "auroc"):
        self.metric = metric

    def evaluate(
        self,
        *,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synth_df: pd.DataFrame | None,
        target_col: str,
        seed: int,
    ) -> dict:
        return evaluate_all_modes(
            real_train=real_train,
            real_test=real_test,
            synth_df=synth_df,
            target_col=target_col,
            seed=seed,
            metric=self.metric,
        )
