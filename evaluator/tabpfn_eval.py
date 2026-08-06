"""
evaluator/tabpfn_eval.py: TabPFN In-Context Learning 评估。

利用合成数据作为 TabPFN 的额外 context examples，增强对 test 数据的推理能力。
TabPFN 无需训练 —— 前向传播时直接使用 context 样本进行 in-context 推理。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_tabpfn_icl(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int = 42,
    metric: str = "auroc",
    max_context: int = 2000,
    weight_synth: float = 0.5,
) -> float:
    """
    TabPFN In-Context Learning 评估。

    合成数据不用于训练，而是作为额外的 context examples 提供给 TabPFN。
    TabPFN 在推理时将 context 样本作为参考，通过 in-context learning 直接预测。

    Parameters
    ----------
    real_train : 真实训练集
    real_test : 真实测试集（用于评估）
    synth_df : 合成数据，作为 context 增强
    target_col : 目标列名
    seed : 随机种子
    metric : "auroc" | "prauc"
    max_context : context 样本上限（TabPFN 最大 ~2000）
    weight_synth : 合成样本在 context 中的权重比例 [0, 1]

    Returns
    -------
    float : 评估分数
    """
    from tabpfn import TabPFNClassifier

    np.random.seed(seed)

    # 1. 特征对齐
    feat_cols = [c for c in real_test.columns if c != target_col]
    if not feat_cols:
        raise ValueError("没有可用的特征列")

    # 2. 预处理：LabelEncode 类别特征 + 填充缺失值
    X_train, X_test, X_synth = _preprocess(real_train, real_test, synth_df, feat_cols)

    # 3. 标签编码
    y_train = real_train[target_col]
    y_test = real_test[target_col]
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test], axis=0).astype(str))
    y_train_enc = le.transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    # 4. 构建 context: 合成样本占 weight_synth 比例
    n_context = min(max_context, len(X_train) + (len(X_synth) if X_synth is not None else 0))
    n_synth = int(n_context * weight_synth) if X_synth is not None else 0
    n_synth = min(n_synth, len(X_synth)) if X_synth is not None else 0
    n_real = n_context - n_synth

    # 从真实和合成数据中采样构建 context
    context_parts_X, context_parts_y = [], []

    if n_real > 0:
        indices = np.random.choice(len(X_train), size=min(n_real, len(X_train)), replace=False)
        context_parts_X.append(X_train[indices])
        context_parts_y.append(y_train_enc[indices])

    if n_synth > 0 and X_synth is not None and synth_df is not None and target_col in synth_df.columns:
        y_synth = synth_df[target_col]
        y_synth_enc = le.transform(y_synth.astype(str))
        indices = np.random.choice(len(X_synth), size=n_synth, replace=False)
        context_parts_X.append(X_synth[indices])
        context_parts_y.append(y_synth_enc[indices])

    X_context = np.vstack(context_parts_X) if context_parts_X else X_train
    y_context = np.concatenate(context_parts_y) if context_parts_y else y_train_enc
    print(f"  TabPFN context: {len(y_context)} samples (real={n_real}, synth={n_synth})")

    # 5. TabPFN In-Context 推理
    clf = TabPFNClassifier(random_state=seed, n_estimators=4, show_progress_bar=False)
    clf.fit(X_context, y_context)
    y_prob = clf.predict_proba(X_test)

    # 6. 计算指标
    n_classes = len(np.unique(y_test_enc))
    if n_classes == 2:
        score = (
            average_precision_score(y_test_enc, y_prob[:, 1])
            if metric == "prauc"
            else roc_auc_score(y_test_enc, y_prob[:, 1])
        )
    else:
        score = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="macro")
    return float(score)


def evaluate_tabpfn_icl_all(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int = 42,
    metric: str = "auroc",
    max_context: int = 2000,
) -> dict[str, float]:
    """
    对比不同 context 策略下的 TabPFN 性能。

    Returns
    -------
    {"real_only": score, "synth_only": score, "real_plus_synth": score}
    """
    results = {}

    # Real-only context
    results["real_only"] = evaluate_tabpfn_icl(
        real_train=real_train, real_test=real_test, synth_df=None,
        target_col=target_col, seed=seed, metric=metric, max_context=max_context,
    )

    # Synth-only context（合成样本作为唯一 context）
    if synth_df is not None and len(synth_df) >= 10:
        results["synth_only"] = evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            weight_synth=1.0,
        )

        # Real + Synth 混合 context
        results["real_plus_synth"] = evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            weight_synth=0.5,
        )

    return results


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _preprocess(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """预处理特征：LabelEncode 类别列 + 中位数填充缺失值。"""
    X_train = real_train[feat_cols].copy()
    X_test = real_test[feat_cols].copy()
    X_synth = synth_df[feat_cols].copy() if synth_df is not None else None

    for col in feat_cols:
        if X_train[col].dtype == object or X_train[col].dtype.name == "category":
            le = LabelEncoder()
            all_vals = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            if X_synth is not None:
                all_vals = pd.concat([all_vals, X_synth[col].astype(str)], axis=0)
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            if X_synth is not None:
                X_synth[col] = le.transform(X_synth[col].astype(str))
        else:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            if X_synth is not None:
                X_synth[col] = X_synth[col].fillna(median_val)

    return X_train.values, X_test.values, X_synth.values if X_synth is not None else None
