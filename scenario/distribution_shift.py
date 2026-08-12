"""
scenarios/distribution_shift.py: Distribution Shift 场景构建
"""

import pandas as pd
from scenario.scenario import BaseScenario


class DistributionShiftScenario(BaseScenario):
    """Distribution Shift 场景：按指定敏感/人口列拆分 Train/Test 并移除切分列"""

    def __init__(self, split_col: str, seed: int = 42):
        super().__init__(seed=seed)
        self.split_col = split_col

    def build(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.split_col not in df.columns:
            raise ValueError(f"划分列 '{self.split_col}' 不存在于数据中。")

        counts = df[self.split_col].value_counts()
        if len(counts) < 2:
            raise ValueError(f"列 '{self.split_col}' 类别数少于 2，无法切分。")

        # 将样本最多的组设为 Train，其余所有组为 Test
        train_group = counts.index[0]

        train_df = df[df[self.split_col] == train_group].drop(columns=[self.split_col]).reset_index(drop=True)
        test_df = df[df[self.split_col] != train_group].drop(columns=[self.split_col]).reset_index(drop=True)

        return train_df, test_df