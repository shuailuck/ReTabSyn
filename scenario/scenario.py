"""
scenario/scenario.py: 场景抽象基类。

场景负责完整流程：数据生成(build) → 合成(synthesize) → 评估(evaluate)。
场景在生成/合成完成后自动保存数据，并在已存在时跳过重新生成。
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SaveConfig:
    """场景数据与合成数据的保存路径配置。"""

    scenario_name: str = ""
    scenario_output_dir: str = ""
    scenario_label: str = ""
    synth_output_dir: str = ""
    synthesizer_name: str = ""


class BaseScenario(ABC):
    """场景构建器抽象基类。"""

    def __init__(self, seed: int = 42, save_config: SaveConfig | None = None):
        self.seed = seed
        self.save_config = save_config or SaveConfig()

        # 场景状态：生成的数据与合成数据
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.synth_df: pd.DataFrame | None = None

    # -------------------------------------------------------------------
    # 数据生成
    # -------------------------------------------------------------------

    def build(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """生成 train/test。若已存在则加载，否则生成并保存。"""
        if self._scenario_data_exists():
            self._load_scenario_data()
            print(f"[skip] 场景数据已存在: {self.save_config.scenario_label}/seed{self.seed}")
            return self.train_df, self.test_df

        train_df, test_df = self._build(df, **kwargs)
        self.train_df = train_df
        self.test_df = test_df
        self._save_scenario_data()
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
        """合成数据。若已存在则加载，否则合成并保存。"""
        if self.train_df is None:
            raise RuntimeError("请先调用 build() 生成训练数据")

        if self._synth_data_exists():
            self._load_synth_data()
            print(f"[skip] 合成数据已存在: {self.save_config.synthesizer_name}/{self.save_config.scenario_label}/seed{self.seed}")
            return self.synth_df

        synth = synthesizer_cls(random_state=self.seed, **(synthesizer_kwargs or {}))
        synth.fit(self.train_df)
        n = n_samples if n_samples is not None else len(self.train_df)
        self.synth_df = synth.sample(n_samples=n)
        self._save_synth_data()
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
        """标准评估：real / synthetic / augment 三种训练数据。"""
        results = {}

        results["real"] = self.evaluator.evaluate(
            train_df=self.train_df, test_df=self.test_df,
            target_col=target_col, seed=self.seed,
        )

        if self.synth_df is not None and len(self.synth_df) >= 10:
            results["synthetic"] = self.evaluator.evaluate(
                train_df=self.synth_df, test_df=self.test_df,
                target_col=target_col, seed=self.seed,
            )

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
    # 保存 / 加载
    # -------------------------------------------------------------------

    def _train_path(self) -> str:
        return os.path.join(self.save_config.scenario_output_dir, f"train_{self.save_config.scenario_label}_seed{self.seed}.csv")

    def _test_path(self) -> str:
        return os.path.join(self.save_config.scenario_output_dir, f"test_{self.save_config.scenario_label}_seed{self.seed}.csv")

    def _clean_path(self) -> str:
        return os.path.join(self.save_config.scenario_output_dir, f"clean_train_{self.save_config.scenario_label}_seed{self.seed}.csv")

    def _synth_path(self) -> str:
        return os.path.join(self.save_config.synth_output_dir, f"{self.save_config.scenario_name}_{self.save_config.synthesizer_name}_{self.save_config.scenario_label}_seed{self.seed}.csv")

    def _scenario_data_exists(self) -> bool:
        return bool(self.save_config.scenario_output_dir and self.save_config.scenario_label and
                    os.path.exists(self._train_path()) and os.path.exists(self._test_path()))

    def _synth_data_exists(self) -> bool:
        return bool(self.save_config.synth_output_dir and self.save_config.synthesizer_name and
                    os.path.exists(self._synth_path()))

    def _save_scenario_data(self) -> None:
        if self.save_config.scenario_output_dir and self.save_config.scenario_label:
            os.makedirs(self.save_config.scenario_output_dir, exist_ok=True)
            self.train_df.to_csv(self._train_path(), index=False)
            self.test_df.to_csv(self._test_path(), index=False)
            clean_train = getattr(self, "clean_train", None)
            if clean_train is not None:
                clean_train.to_csv(self._clean_path(), index=False)

    def _load_scenario_data(self) -> None:
        self.train_df = pd.read_csv(self._train_path())
        self.test_df = pd.read_csv(self._test_path())
        if os.path.exists(self._clean_path()):
            self.clean_train = pd.read_csv(self._clean_path())

    def _save_synth_data(self) -> None:
        if self.save_config.synth_output_dir and self.save_config.synthesizer_name:
            os.makedirs(self.save_config.synth_output_dir, exist_ok=True)
            self.synth_df.to_csv(self._synth_path(), index=False)

    def _load_synth_data(self) -> None:
        self.synth_df = pd.read_csv(self._synth_path())

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
