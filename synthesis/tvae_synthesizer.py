"""
TVAE 合成器：基于 Conditional VAE 的表格数据合成。

Paper: Modeling Tabular Data using Conditional GAN (NeurIPS 2019)
       TVAE 是 CTGAN 论文中提出的 VAE 变体，训练更稳定、速度更快。

实现基于 sdv-dev/CTGAN 官方库。
"""

import numpy as np
import pandas as pd

from synthesis.synthesizer import BaseTabularSynthesizer


class TVAESynthesizer(BaseTabularSynthesizer):
    """TVAE (Tabular VAE) 合成器。

    使用变分自编码器学习表格数据的分布，相比 GAN 类方法训练更稳定。
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        embedding_dim: int = 128,
        compress_dims: tuple = (128, 128),
        decompress_dims: tuple = (128, 128),
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: float = 2,
        enable_gpu: bool = True,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        embedding_dim : 随机噪声向量的维度
        compress_dims : 编码器各隐藏层维度
        decompress_dims : 解码器各隐藏层维度
        l2scale : L2 正则化系数
        batch_size : 训练批量大小
        epochs : 训练轮数
        loss_factor : 重建误差的权重系数
        enable_gpu : 是否启用 GPU 加速
        verbose : 是否打印训练进度
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.enable_gpu = enable_gpu
        self.verbose = verbose

    def _fit(self, df: pd.DataFrame):
        import torch
        from ctgan import TVAE

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        discrete_cols = []
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == "category":
                discrete_cols.append(col)

        self._model = TVAE(
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs,
            loss_factor=self.loss_factor,
            enable_gpu=self.enable_gpu,
            verbose=self.verbose,
        )
        self._model.fit(df, discrete_columns=discrete_cols if discrete_cols else None)

    def _sample(self, n_samples: int) -> pd.DataFrame:
        import torch

        torch.manual_seed(self.random_state + 1)
        np.random.seed(self.random_state + 1)

        return self._model.sample(samples=n_samples)
