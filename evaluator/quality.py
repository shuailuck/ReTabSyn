"""
evaluator/quality.py: 合成数据质量评估。

两个维度:
  1. 统计相似性 (Statistical similarity): column-wise 指标
     - Chi-squared test
     - Total Variation distance
     - KL Divergence
  2. 标签正确性 (Label accuracy): 原始数据训练模型，在合成数据上测试
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ===========================================================================
# 1. 统计相似性
# ===========================================================================

def _aligned_frequencies(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, col: str, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """对齐 real 和 synth 的列分布，返回归一化概率数组。

    类别列直接用值对齐；数值列按 real 的分位数分箱，synth 用相同边界。
    """
    is_categorical = (real_df[col].dtype == object or
                      real_df[col].dtype.name == "category" or
                      real_df[col].nunique() <= n_bins)

    if is_categorical:
        vals = sorted(set(real_df[col].astype(str)) | set(synth_df[col].astype(str)))
        r = np.array([(real_df[col].astype(str) == v).mean() for v in vals])
        s = np.array([(synth_df[col].astype(str) == v).mean() for v in vals])
    else:
        # 数值列：用 real 的分位数边界
        _, edges = pd.qcut(real_df[col], q=n_bins, retbins=True, duplicates="drop")
        r = np.histogram(real_df[col], bins=edges, density=True)[0]
        s = np.histogram(synth_df[col], bins=edges, density=True)[0]
        # 归一化
        r = r / r.sum() if r.sum() > 0 else r
        s = s / s.sum() if s.sum() > 0 else s

    return r, s


def chi_squared_test(real_df: pd.DataFrame, synth_df: pd.DataFrame, col: str) -> float:
    """Chi-squared 检验的 p-value（越小说明差异越显著）。"""
    r, s = _aligned_frequencies(real_df, synth_df, col)
    n = len(synth_df)
    observed = s * n
    expected = r * n
    expected[expected == 0] = 1e-9
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    from scipy.stats import chi2
    p_value = 1.0 - chi2.cdf(chi2_stat, max(len(r) - 1, 1))
    return float(p_value)


def total_variation(real_df: pd.DataFrame, synth_df: pd.DataFrame, col: str) -> float:
    """Total Variation distance: 0.5 * sum |p_i - q_i| ∈ [0, 1]。"""
    r, s = _aligned_frequencies(real_df, synth_df, col)
    return float(0.5 * np.sum(np.abs(r - s)))


def kl_divergence(real_df: pd.DataFrame, synth_df: pd.DataFrame, col: str) -> float:
    """KL divergence: D_KL(real || synth)，衡量 synth 对 real 的覆盖。"""
    r, s = _aligned_frequencies(real_df, synth_df, col)
    r = np.clip(r, 1e-9, None)
    s = np.clip(s, 1e-9, None)
    return float(np.sum(r * np.log(r / s)))


def statistical_similarity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """逐列计算统计相似性，返回 DataFrame。

    Returns
    -------
    列: column, chi2_pvalue, total_variation, kl_divergence
    """
    if columns is None:
        columns = [c for c in real_df.columns if c in synth_df.columns]

    rows = []
    for col in columns:
        rows.append({
            "column": col,
            "chi2_pvalue": chi_squared_test(real_df, synth_df, col),
            "total_variation": total_variation(real_df, synth_df, col),
            "kl_divergence": kl_divergence(real_df, synth_df, col),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 2. 标签正确性
# ===========================================================================

def _preprocess_for_classifier(
    real_train: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LabelEncode 特征与标签，返回 (X_real, y_real, X_synth, y_synth)。"""
    feat_cols = [c for c in real_train.columns if c != target_col and c in synth_df.columns]
    X_real, X_synth = real_train[feat_cols].copy(), synth_df[feat_cols].copy()

    for col in feat_cols:
        if X_real[col].dtype == object or X_real[col].dtype.name == "category":
            le = LabelEncoder()
            all_vals = pd.concat([X_real[col], X_synth[col]], axis=0).astype(str)
            le.fit(all_vals)
            X_real[col] = le.transform(X_real[col].astype(str))
            X_synth[col] = le.transform(X_synth[col].astype(str))
        else:
            m = X_real[col].median()
            X_real[col] = X_real[col].fillna(m)
            X_synth[col] = X_synth[col].fillna(m)

    le_y = LabelEncoder()
    le_y.fit(pd.concat([real_train[target_col], synth_df[target_col]], axis=0).astype(str))
    y_real = le_y.transform(real_train[target_col].astype(str))
    y_synth = le_y.transform(synth_df[target_col].astype(str))

    return X_real.values, y_real, X_synth.values, y_synth


def label_accuracy(
    real_train: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
) -> dict[str, float]:
    """原始数据训练模型，在合成数据上测试标签正确性。

    Returns
    -------
    {model_name: accuracy}
    """
    X_real, y_real, X_synth, y_synth = _preprocess_for_classifier(
        real_train, synth_df, target_col
    )

    models = {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed),
    }

    results = {}
    for name, clf in models.items():
        try:
            clf.fit(X_real, y_real)
            y_pred = clf.predict(X_synth)
            results[name] = float(accuracy_score(y_synth, y_pred))
        except Exception:
            results[name] = np.nan
    return results
