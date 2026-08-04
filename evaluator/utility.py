"""
evaluator/utility.py: 下游机器学习评估模块

支持真实数据、合成数据、真实+合成数据在下游多个分类模型上的性能评估。
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _build_classifiers(seed: int) -> dict:
    """构建分类器集合。"""
    models = {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "NB": GaussianNB(),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=seed, verbosity=0),
    }
    if _CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100, random_seed=seed, verbose=0, allow_writing_files=False
        )
    return models


def _preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """特征编码与归一化：类别特征 LabelEncoding + 数值特征 StandardScaler + 缺失值填充。"""
    X_tr, X_te = X_train.copy(), X_test.copy()

    for col in X_tr.columns:
        if X_tr[col].dtype == object or X_tr[col].dtype.name == "category":
            le = LabelEncoder()
            combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype(str)
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))
        else:
            median_val = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(median_val)
            X_te[col] = X_te[col].fillna(median_val)

    scaler = StandardScaler()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])

    return X_tr, X_te


def _encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """标签编码。"""
    le = LabelEncoder()
    combined = pd.concat([y_train, y_test], axis=0).astype(str)
    le.fit(combined)
    return le.transform(y_train.astype(str)), le.transform(y_test.astype(str))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def evaluate_downstream(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int,
    metric: str = "auroc",
    mode: str = "real",
) -> dict[str, float]:
    """
    单次下游评估。

    Parameters
    ----------
    real_train : 真实训练集
    real_test : 真实测试集（始终用于评估）
    synth_df : 合成数据（mode 为 synthetic/augment 时使用）
    target_col : 目标列名
    seed : 随机种子
    metric : "auroc" 或 "prauc"
    mode : "real" | "synthetic" | "augment"

    Returns
    -------
    {classifier_name: score}
    """
    # 1. 按模式组装训练集
    if mode == "real":
        train_df = real_train.copy()

    elif mode == "synthetic":
        if synth_df is None or len(synth_df) < 10:
            return {}
        n_use = min(len(real_train), len(synth_df))
        train_df = synth_df.sample(n=n_use, random_state=seed) if n_use < len(synth_df) else synth_df.copy()

    elif mode == "augment":
        if synth_df is None or len(synth_df) < 10:
            train_df = real_train.copy()
        else:
            n_total = len(real_train)
            n_real = n_total // 2
            n_synth = min(n_total - n_real, len(synth_df))
            real_sub = real_train.sample(n=n_real, random_state=seed) if n_real < len(real_train) else real_train.copy()
            synth_sub = synth_df.sample(n=n_synth, random_state=seed + 1) if n_synth < len(synth_df) else synth_df.copy()
            train_df = pd.concat([real_sub, synth_sub], axis=0)
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 2. 特征与标签对齐
    feat_cols = [c for c in real_test.columns if c != target_col and c in train_df.columns]
    if not feat_cols:
        return {}

    X_train = train_df[feat_cols]
    y_train = train_df[target_col]
    X_test = real_test[feat_cols]
    y_test = real_test[target_col]

    # 3. 预处理
    X_train, X_test = _preprocess_features(X_train, X_test)
    y_train_enc, y_test_enc = _encode_target(y_train, y_test)

    n_classes = len(np.unique(y_test_enc))
    is_binary = (n_classes == 2)

    # 4. 训练并评估
    results = {}
    for name, clf in _build_classifiers(seed).items():
        try:
            clf.fit(X_train, y_train_enc)
            if is_binary:
                y_prob = clf.predict_proba(X_test)[:, 1]
                score = (
                    average_precision_score(y_test_enc, y_prob)
                    if metric == "prauc"
                    else roc_auc_score(y_test_enc, y_prob)
                )
            else:
                y_prob = clf.predict_proba(X_test)
                score = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="macro")
            results[name] = score
        except Exception:
            results[name] = np.nan

    return results


def evaluate_all_modes(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int,
    metric: str = "auroc",
) -> dict[str, dict[str, float]]:
    """
    一次性评估 real / synthetic / augment 三种训练模式。

    Returns
    -------
    {"real": {clf: score}, "synthetic": {clf: score}, "augment": {clf: score}}
    """
    results = {}
    for mode in ("real", "synthetic", "augment"):
        results[mode] = evaluate_downstream(
            real_train=real_train,
            real_test=real_test,
            synth_df=synth_df,
            target_col=target_col,
            seed=seed,
            metric=metric,
            mode=mode,
        )
    return results
