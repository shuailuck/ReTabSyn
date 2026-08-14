"""
evaluator/downstream.py: 下游机器学习评估器。

通用的「训练 + 评测」：在 train_df 上训练多个分类器，在 test_df 上评测。
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

from evaluator.base import BaseEvaluator

warnings.filterwarnings("ignore")


class DownstreamEvaluator(BaseEvaluator):
    """标准下游评估器：多分类器在 train_df 上训练、test_df 上评测。"""

    def __init__(self, metric: str = "auroc"):
        self.metric = metric

    def evaluate(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        seed: int,
    ) -> dict:
        feat_cols = [c for c in test_df.columns if c != target_col and c in train_df.columns]
        if not feat_cols:
            return {}

        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test = test_df[feat_cols]
        y_test = test_df[target_col]

        X_train, X_test = self._preprocess_features(X_train, X_test)
        y_train_enc, y_test_enc = self._encode_target(y_train, y_test)

        n_classes = len(np.unique(y_test_enc))
        is_binary = (n_classes == 2)

        results = {}
        for name, clf in self._build_classifiers(seed).items():
            try:
                clf.fit(X_train, y_train_enc)
                if is_binary:
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    score = (average_precision_score(y_test_enc, y_prob)
                             if self.metric == "prauc"
                             else roc_auc_score(y_test_enc, y_prob))
                else:
                    y_prob = clf.predict_proba(X_test)
                    score = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="macro")
                results[name] = score
            except Exception:
                results[name] = np.nan
        return results

    # -------------------------------------------------------------------
    # 内部工具
    # -------------------------------------------------------------------

    def _build_classifiers(self, seed):
        models = {
            "LR": LogisticRegression(max_iter=2000, random_state=seed),
            "NB": GaussianNB(),
            "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            "XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=seed, verbosity=0),
        }
        if _CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(
                iterations=100, random_seed=seed, verbose=0, allow_writing_files=False)
        return models

    def _preprocess_features(self, X_train, X_test):
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

    def _encode_target(self, y_train, y_test):
        le = LabelEncoder()
        combined = pd.concat([y_train, y_test], axis=0).astype(str)
        le.fit(combined)
        return le.transform(y_train.astype(str)), le.transform(y_test.astype(str))
