from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd


# ===========================================================================
# 1. 公共数据转换与处理工具类
# ===========================================================================

class TabularDataTransformer:
    """负责表格数据的类型识别、数值边界保护与格式后处理。"""

    def __init__(self, integer_columns: list = None, drop_na: bool = True):
        self.integer_columns = integer_columns
        self.drop_na = drop_na
        self.numeric_cols = []
        self.categorical_cols = []
        self.num_ranges = {}

    def fit_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析列类型，记录数值边界，处理缺失值。"""
        df_clean = df.copy()

        if self.drop_na and df_clean.isnull().any().any():
            warnings.warn("Input DataFrame contains NaNs. Dropping missing rows.")
            df_clean = df_clean.dropna().reset_index(drop=True)

        self.numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in df_clean.columns if c not in self.numeric_cols]

        if self.integer_columns is None:
            self.integer_columns = df_clean[self.numeric_cols].select_dtypes(include=["int64", "int32"]).columns.tolist()

        for col in self.numeric_cols:
            self.num_ranges[col] = (df_clean[col].min(), df_clean[col].max())

        return df_clean

    def postprocess(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        """对合成数据进行数值裁剪与类型还原。"""
        synth_df = synth_df.copy()

        for col in self.numeric_cols:
            if col in synth_df.columns:
                min_v, max_v = self.num_ranges[col]
                synth_df[col] = synth_df[col].clip(lower=min_v, upper=max_v)

                if col in self.integer_columns:
                    synth_df[col] = synth_df[col].round().astype(int)
                else:
                    synth_df[col] = synth_df[col].astype(float)

        return synth_df


# ===========================================================================
# 2. 统一算法基类抽象
# ===========================================================================

class BaseTabularSynthesizer(ABC):
    """所有表格合成算法的基类抽象。"""

    def __init__(self, integer_columns: list = None, random_state: int = 42):
        self.random_state = random_state
        self.transformer = TabularDataTransformer(integer_columns=integer_columns)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """训练入口：仅做预处理与列元信息记录，不对输入数据做划分。"""
        np.random.seed(self.random_state)
        df_clean = self.transformer.fit_preprocess(df)
        self._fit(df_clean)
        self.is_fitted = True
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """采样并自动做数值后处理。"""
        if not self.is_fitted:
            raise RuntimeError("Synthesizer must be fitted before calling sample().")

        np.random.seed(self.random_state)
        synth_raw = self._sample(n_samples)
        return self.transformer.postprocess(synth_raw)

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _sample(self, n_samples: int) -> pd.DataFrame:
        pass
