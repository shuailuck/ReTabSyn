"""
scenarios/small_data.py: Small Data 场景构建
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from scenario import BaseScenario
from imbalanced import ImbalancedDataScenario


class SmallDataScenario(BaseScenario):
    """Small Data 场景：限制 Train 和 Test 在小样本预算 N"""


    def __init__(
        self,
        target_col: str,
        n_samples: int = 64,
        balance_mode: str = "raw",
        seed: int = 42
    ):
        super().__init__(seed=seed)
        self.target_col = target_col
        self.n_samples = n_samples
        self.balance_mode = balance_mode

    def build(self, df: pd.DataFrame, full_test: pd.DataFrame | None = None, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 1. 基础 80/20 切分 (若未传入外部独立 test 集)
        if full_test is None:
            stratify = df[self.target_col] if self.target_col in df.columns else None
            full_train, full_test = train_test_split(
                df, test_size=0.2, random_state=self.seed, stratify=stratify
            )
        else:
            full_train = df

        # 2. 根据 balance_mode 调整样本分布
        if self.balance_mode == "balanced":
            balancer = ImbalancedDataScenario(target_col=self.target_col, minority_prev=0.5, seed=self.seed)
            base_tr = balancer.build_imbalanced_df(full_train)
            base_te = balancer.build_imbalanced_df(full_test)
        elif self.balance_mode == "imbalanced":
            imbalancer = ImbalancedDataScenario(target_col=self.target_col, minority_prev=0.01, seed=self.seed)
            base_tr = imbalancer.build_imbalanced_df(full_train)
            base_te = imbalancer.build_imbalanced_df(full_test)
        elif self.balance_mode == "raw":
            base_tr, base_te = full_train, full_test
        else:
            raise ValueError(f"未知的 balance_mode: {self.balance_mode}")

        # 3. 采样至 N 个样本
        sub_train = self._subsample(base_tr, self.n_samples, seed=self.seed, target_col=self.target_col)
        sub_test = self._subsample(base_te, self.n_samples, seed=self.seed + 100, target_col=self.target_col)

        return sub_train, sub_test