"""
tabpfn/context.py: Context 构建策略。

定义 TabPFN 推理时如何从真实/合成数据中选择 context 样本。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ContextBuilder(ABC):
    """Context 构建器抽象基类。

    子类实现 _select() 决定 context 的组成方式。
    """

    def __init__(self, max_context: int = 2000, seed: int = 42):
        self.max_context = max_context
        self.seed = seed

    def build(
        self,
        X_train: np.ndarray,
        y_train_enc: np.ndarray,
        X_synth: np.ndarray | None,
        synth_df: pd.DataFrame | None,
        target_col: str,
        label_encoder: LabelEncoder,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Returns
        -------
        (X_context, y_context, info_dict)
        """
        np.random.seed(self.seed)
        return self._select(X_train, y_train_enc, X_synth, synth_df,
                            target_col, label_encoder)

    @abstractmethod
    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder) -> tuple[np.ndarray, np.ndarray, dict]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def _sample(self, X: np.ndarray, n: int) -> np.ndarray:
        """无放回采样 n 条（不够则重复）。"""
        if n >= len(X):
            return np.arange(len(X))
        return np.random.choice(len(X), size=n, replace=False)

    def _get_synth_y(self, synth_df, X_synth, idx, target_col,
                     label_encoder: LabelEncoder) -> np.ndarray:
        """从合成 DataFrame 中获取标签并编码。"""
        y_raw = synth_df[target_col][idx].astype(str)
        return label_encoder.transform(y_raw)


# ===========================================================================
# 具体策略
# ===========================================================================

class RealOnlyBuilder(ContextBuilder):
    """仅使用真实数据。"""

    name = "real_only"

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        n = min(len(X_train), self.max_context)
        idx = self._sample(X_train, n)
        return X_train[idx], y_train_enc[idx], {"real": n, "synth": 0}


class SynthOnlyBuilder(ContextBuilder):
    """仅使用合成数据。"""

    name = "synth_only"

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        if X_synth is None or len(X_synth) == 0:
            raise ValueError("synth_only 策略需要合成数据")
        n = min(len(X_synth), self.max_context)
        idx = self._sample(X_synth, n)
        y = self._get_synth_y(synth_df, X_synth, idx, target_col, label_encoder)
        return X_synth[idx], y, {"real": 0, "synth": n}


class MixedBuilder(ContextBuilder):
    """真实 + 合成混合，配比控制 synth:real ≤ max_synth_ratio。"""

    name = "real_plus_synth"

    def __init__(self, max_context: int = 2000, seed: int = 42,
                 max_synth_ratio: float = 3.0):
        super().__init__(max_context, seed)
        self.max_synth_ratio = max_synth_ratio

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        n_real = min(len(X_train), self.max_context // 2)
        n_synth_max = min(int(n_real * self.max_synth_ratio),
                          self.max_context - n_real)
        n_synth = min(len(X_synth), n_synth_max) if X_synth is not None else 0

        parts_X, parts_y = [], []
        idx_r = self._sample(X_train, n_real)
        parts_X.append(X_train[idx_r])
        parts_y.append(y_train_enc[idx_r])

        if n_synth > 0 and X_synth is not None:
            idx_s = self._sample(X_synth, n_synth)
            parts_X.append(X_synth[idx_s])
            parts_y.append(self._get_synth_y(synth_df, X_synth, idx_s,
                                              target_col, label_encoder))

        X_ctx = np.vstack(parts_X) if parts_X else X_train
        y_ctx = np.concatenate(parts_y) if parts_y else y_train_enc
        return X_ctx, y_ctx, {"real": n_real, "synth": n_synth}


class FilteredMixedBuilder(MixedBuilder):
    """混合 + Isolation Forest 密度筛选。"""

    name = "real_plus_synth_filtered"

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        if X_synth is not None and len(X_synth) > 0:
            X_synth, y_synth_raw = _apply_if_filter(
                X_train, X_synth, synth_df, target_col, self.seed
            )
            synth_df = synth_df.copy()
            synth_df[target_col] = y_synth_raw
        return super()._select(X_train, y_train_enc, X_synth, synth_df,
                               target_col, label_encoder)


class ImbalancedAllSynthBuilder(ContextBuilder):
    """不平衡场景：全部合成数据直接作 context（不混合真实数据）。"""

    name = "imbalanced_all_synth"

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        if X_synth is None or len(X_synth) == 0:
            raise ValueError("imbalanced_all_synth 策略需要合成数据")
        n = min(len(X_synth), self.max_context)
        idx = np.arange(n) if n == len(X_synth) else self._sample(X_synth, n)
        y = self._get_synth_y(synth_df, X_synth, idx, target_col, label_encoder)
        return X_synth[idx], y, {"real": 0, "synth": n}


class AllInBuilder(ContextBuilder):
    """全部真实 + 全部合成数据，不加筛选，不控比例。"""

    name = "all_in"

    def _select(self, X_train, y_train_enc, X_synth, synth_df,
                target_col, label_encoder):
        parts_X, parts_y = [X_train], [y_train_enc]
        n_real, n_synth = len(X_train), 0

        if X_synth is not None and len(X_synth) > 0:
            parts_X.append(X_synth)
            parts_y.append(self._get_synth_y(synth_df, X_synth,
                           np.arange(len(X_synth)), target_col, label_encoder))
            n_synth = len(X_synth)

        total = n_real + n_synth
        if total > self.max_context:
            print(f"  [AllIn] 总样本 {total} 超过 max_context，截断")
            # 保留全部真实 + 尽量多的合成
            n_synth = min(n_synth, self.max_context - n_real)
            parts_X = [X_train, X_synth[:n_synth]]
            parts_y = [y_train_enc,
                       self._get_synth_y(synth_df, X_synth,
                       np.arange(n_synth), target_col, label_encoder)]

        return np.vstack(parts_X), np.concatenate(parts_y), \
            {"real": n_real, "synth": min(n_synth, self.max_context - n_real)}


# ===========================================================================
# Isolation Forest 密度筛选 (内部函数)
# ===========================================================================

def _apply_if_filter(
    X_train: np.ndarray,
    X_synth: np.ndarray,
    synth_df: pd.DataFrame,
    target_col: str,
    seed: int,
) -> tuple[np.ndarray, pd.Series]:
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(random_state=seed, contamination=0.1)
    iso.fit(X_train)
    preds = iso.predict(X_synth)
    mask = preds == 1
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  [IF] 过滤掉 {n_removed}/{len(X_synth)} 条合成样本")
    y_synth = synth_df[target_col]
    return X_synth[mask], y_synth[mask].reset_index(drop=True)
