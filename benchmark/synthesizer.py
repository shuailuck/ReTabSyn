from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. 公共数据转换与处理工具类（已添加 SOURCE_LABEL 划分）
# ===========================================================================

class TabularDataTransformer:
    """负责表格数据的类型识别、数值边界保护、格式后处理以及 SOURCE_LABEL 自动划分的工具类"""

    def __init__(self, integer_columns: list = None, drop_na: bool = True, test_size: float = 0.2):
        self.integer_columns = integer_columns
        self.drop_na = drop_na
        self.test_size = test_size
        self.numeric_cols = []
        self.categorical_cols = []
        self.num_ranges = {}  # 保存原始数值列的最大极值

    def fit_transform_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析列类型，记录数值边界，处理缺失值，并为真实数据添加 SOURCE_LABEL 列"""
        df_clean = df.copy()

        # 1. 缺失值处理
        if self.drop_na and df_clean.isnull().any().any():
            warnings.warn("Input DataFrame contains NaNs. Dropping missing rows.")
            df_clean = df_clean.dropna().reset_index(drop=True)

        # 2. SOURCE_LABEL 列处理：如果原始数据不存在 SOURCE_LABEL，则使用 train_test_split 进行划分
        if "SOURCE_LABEL" not in df_clean.columns:
            train_idx, test_idx = train_test_split(
                df_clean.index, 
                test_size=self.test_size, 
                random_state=42, 
                shuffle=True
            )
            df_clean["SOURCE_LABEL"] = "train"
            df_clean.loc[test_idx, "SOURCE_LABEL"] = "test"

        # 3. 列类型自动识别（排除 SOURCE_LABEL 列）
        feature_cols = [c for c in df_clean.columns if c != "SOURCE_LABEL"]
        self.numeric_cols = df_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in feature_cols if c not in self.numeric_cols]

        # 4. 推断整数列（若未显式指定）
        if self.integer_columns is None:
            self.integer_columns = df_clean[self.numeric_cols].select_dtypes(include=["int64", "int32"]).columns.tolist()

        # 5. 记录数值列的区间边界 (Min-Max Range)
        for col in self.numeric_cols:
            self.num_ranges[col] = (df_clean[col].min(), df_clean[col].max())

        return df_clean

    def postprocess(self, synth_df: pd.DataFrame, default_source_label: str = "train") -> pd.DataFrame:
        """对生成的合成数据进行数值修剪、类型还原，并添加 SOURCE_LABEL 列"""
        synth_df = synth_df.copy()

        # 1. 自动添加/补充 SOURCE_LABEL 列（合成数据默认全部作为训练集）
        if "SOURCE_LABEL" not in synth_df.columns:
            synth_df["SOURCE_LABEL"] = default_source_label

        # 2. 数值裁剪与类型还原
        for col in self.numeric_cols:
            if col in synth_df.columns:
                # 截断越界值，确保在真实物理范围内
                min_v, max_v = self.num_ranges[col]
                synth_df[col] = synth_df[col].clip(lower=min_v, upper=max_v)

                # 整数列四舍五入并转为 int
                if col in self.integer_columns:
                    synth_df[col] = synth_df[col].round().astype(int)
                else:
                    synth_df[col] = synth_df[col].astype(float)

        return synth_df


# ===========================================================================
# 2. 统一算法基类抽象
# ===========================================================================

class BaseTabularSynthesizer(ABC):
    """所有表格合成算法的基类抽象"""

    def __init__(self, integer_columns: list = None, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.transformer = TabularDataTransformer(integer_columns=integer_columns, test_size=test_size)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """训练入口"""
        np.random.seed(self.random_state)
        df_clean = self.transformer.fit_transform_preprocess(df)
        self._fit(df_clean)
        self.is_fitted = True
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """采样入口：采样后自动经过 postprocess 添加 SOURCE_LABEL='train'"""
        if not self.is_fitted:
            raise RuntimeError("Synthesizer must be fitted before calling sample().")

        np.random.seed(self.random_state)
        synth_raw = self._sample(n_samples)
        # 后处理添加 SOURCE_LABEL="train"
        return self.transformer.postprocess(synth_raw, default_source_label="train")

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _sample(self, n_samples: int) -> pd.DataFrame:
        pass