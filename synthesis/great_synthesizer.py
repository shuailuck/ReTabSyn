"""
GReaT 合成器：基于 LLM 的表格数据合成（可选 SMOTE 数据增强）。

Paper: Language Models are Realistic Tabular Data Generators (ICLR 2023)
"""

import os
import numpy as np
import pandas as pd

from smote import generate_smote_synthetic
from DPOUtils import suggest_synthetic_count
from synthesis.synthesizer import BaseTabularSynthesizer


class GreatSynthesizer(BaseTabularSynthesizer):
    """GReaT: 将表格行序列化为文本，微调 LLM 学习数据分布并生成合成样本。

    可选 SMOTE 数据增强，支持固定数量 / 比例 / 动态建议三种模式。
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        llm: str = "distilgpt2",
        batch_size: int = 32,
        epochs: int = 50,
        fp16: bool = True,
        dataloader_num_workers: int = 0,
        float_precision: int | None = None,
        n_aug: int = 0,
        conditional_col: str | None = None,
        model_save_dir: str = "./great_model",
    ):
        """
        Parameters
        ----------
        llm : 基座语言模型，如 "distilgpt2", "tabularisai/Qwen3-0.3B-distil"
        batch_size : 训练 batch size
        epochs : 训练轮数
        fp16 : 是否使用混合精度训练
        dataloader_num_workers : DataLoader 工作进程数
        float_precision : 浮点数小数位数限制，None 表示不限制
        n_aug : SMOTE 增强样本数。>0=固定数量, 0=不增强,
                -1~-100=训练集的 abs(n_aug) 倍, -114514=动态建议策略
        conditional_col : 条件列名，None 则使用第一列
        model_save_dir : 训练完成后模型保存目录 (默认 ./great_model)
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.llm = llm
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.float_precision = float_precision
        self.n_aug = n_aug
        self.conditional_col = conditional_col
        self.model_save_dir = model_save_dir

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        import torch
        from be_great import GReaT

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train = df.copy().reset_index(drop=True)

        # SMOTE 数据增强
        n_aug = self._resolve_n_aug(train)
        if n_aug > 0:
            train = self._augment_with_smote(train, n_aug)

        extra = {"save_strategy": "no"}
        if self.float_precision is not None:
            extra["float_precision"] = self.float_precision

        self._model = GReaT(
            llm=self.llm,
            batch_size=self.batch_size,
            epochs=self.epochs,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            **extra,
        )

        cond_col = self.conditional_col or train.columns[0]
        self._model.fit(train, conditional_col=cond_col)

        # 保存训练好的最终模型
        if self.model_save_dir:
            os.makedirs(self.model_save_dir, exist_ok=True)
            self._model.save(self.model_save_dir)
            print(f"[GReaT] 模型已保存至: {self.model_save_dir}")

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        import torch

        torch.manual_seed(self.random_state + 1)

        return self._model.sample(n_samples=n_samples)

    # -------------------------------------------------------------------
    # SMOTE 增强
    # -------------------------------------------------------------------

    def _resolve_n_aug(self, df: pd.DataFrame) -> int:
        if self.n_aug == -114514:
            n = suggest_synthetic_count(df)
            print(f"[GReaT] 动态建议 SMOTE 增强数: {n}")
            return n
        elif self.n_aug < 0:
            ratio = abs(self.n_aug)
            n = int(len(df) * ratio)
            print(f"[GReaT] SMOTE 增强: {n} ({ratio}x 训练集)")
            return n
        return self.n_aug

    def _augment_with_smote(self, df: pd.DataFrame, n_aug: int) -> pd.DataFrame:
        integer_cols = [
            c for c in df.columns
            if df[c].dtype in ("int64", "int32", "int16", "int8")
        ]
        synth = generate_smote_synthetic(
            df, n_samples=n_aug,
            integer_columns=integer_cols,
            random_state=self.random_state,
        )
        return pd.concat([df, synth], ignore_index=True)
