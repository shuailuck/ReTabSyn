import numpy as np
import pandas as pd

from synthesis.synthesizer import BaseTabularSynthesizer


class GreatSynthesizer(BaseTabularSynthesizer):
    """基于 GReaT (be_great) 的表格数据合成器。

    GReaT 使用预训练语言模型 (如 distilgpt2) 将表格行编码为文本序列，
    通过微调 LLM 学习数据分布并生成合成样本。
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
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.llm = llm
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.float_precision = float_precision

    def _fit(self, df: pd.DataFrame):
        import torch
        from be_great import GReaT

        torch.manual_seed(self.random_state)

        extra = {}
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
        self._model.fit(df)

    def _sample(self, n_samples: int) -> pd.DataFrame:
        import torch

        torch.manual_seed(self.random_state + 1)

        return self._model.sample(n_samples=n_samples)
