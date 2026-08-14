"""
scenario/scenario.py: 场景抽象基类。

场景负责完整流程：数据生成(build) → 合成(synthesize) → 评估(evaluate)。
场景维护生成的数据(train_df/test_df)与合成数据(synth_df)。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseScenario(ABC):
    """场景构建器抽象基类。"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        # 场景状态：生成的数据与合成数据
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.synth_df: pd.DataFrame | None = None

    # -------------------------------------------------------------------
    # 数据生成
    # -------------------------------------------------------------------

    def build(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """生成 train/test 并保存到场景状态。"""
        train_df, test_df = self._build(df, **kwargs)
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    @abstractmethod
    def _build(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """子类实现具体的数据切分逻辑。"""
        ...

    def set_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """直接注入预生成的数据（跳过 build）。"""
        self.train_df = train_df
        self.test_df = test_df

    # -------------------------------------------------------------------
    # 合成
    # -------------------------------------------------------------------

    def synthesize(self, synthesizer_cls, synthesizer_kwargs: dict | None = None,
                   n_samples: int | None = None) -> pd.DataFrame:
        """用合成算法在 train_df 上合成数据，保存到场景状态。"""
        if self.train_df is None:
            raise RuntimeError("请先调用 build() 生成训练数据")

        kwargs = synthesizer_kwargs or {}
        synth = synthesizer_cls(random_state=self.seed, **kwargs)
        synth.fit(self.train_df)
        n = n_samples if n_samples is not None else len(self.train_df)
        self.synth_df = synth.sample(n_samples=n)
        return self.synth_df

    # -------------------------------------------------------------------
    # 评估
    # -------------------------------------------------------------------

    @property
    def evaluator(self):
        """该场景使用的评估器。"""
        from evaluator.downstream import DownstreamEvaluator
        return DownstreamEvaluator()

    def evaluate(self, target_col: str) -> dict:
        """标准评估：real / synthetic / augment 三种训练数据。

        Returns
        -------
        {mode: {模型名: 分数}}
        """
        results = {}

        # real: 真实数据训练
        results["real"] = self.evaluator.evaluate(
            train_df=self.train_df, test_df=self.test_df,
            target_col=target_col, seed=self.seed,
        )

        # synthetic: 仅合成数据训练
        if self.synth_df is not None and len(self.synth_df) >= 10:
            results["synthetic"] = self.evaluator.evaluate(
                train_df=self.synth_df, test_df=self.test_df,
                target_col=target_col, seed=self.seed,
            )

            # augment: 真实+合成混合训练
            n_real = len(self.train_df) // 2
            n_synth = min(len(self.train_df) - n_real, len(self.synth_df))
            real_sub = self.train_df.sample(n=n_real, random_state=self.seed) if n_real < len(self.train_df) else self.train_df
            synth_sub = self.synth_df.sample(n=n_synth, random_state=self.seed + 1) if n_synth < len(self.synth_df) else self.synth_df
            augment_df = pd.concat([real_sub, synth_sub], axis=0)
            results["augment"] = self.evaluator.evaluate(
                train_df=augment_df, test_df=self.test_df,
                target_col=target_col, seed=self.seed,
            )

        return results

    # -------------------------------------------------------------------
    # 工具
    # -------------------------------------------------------------------

    @staticmethod
    def _subsample(df: pd.DataFrame, n: int, seed: int, target_col: str | None = None) -> pd.DataFrame:
        """公共辅助工具：安全无放回随机/分层子采样"""
        if len(df) <= n:
            return df.copy().reset_index(drop=True)

        if target_col and target_col in df.columns and df[target_col].nunique() < 20:
            frac = n / len(df)
            sub_df, _ = train_test_split(
                df, train_size=frac, random_state=seed, stratify=df[target_col]
            )
            return sub_df.reset_index(drop=True)

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(df), size=n, replace=False)
        return df.iloc[indices].reset_index(drop=True)
