"""
tabpfn/core.py: TabPFN In-Context Learning 推理引擎。

负责数据预处理、TabPFN 推理和指标计算。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from tabpfn_icl.context import ContextBuilder


class TabPFNEngine:
    """TabPFN In-Context Learning 推理引擎。

    给定真实训练/测试数据和合成数据，用 ContextBuilder 构建 context，
    通过 TabPFN 进行 in-context 推理并计算指标。
    """

    def __init__(self, seed: int = 42, metric: str = "auroc"):
        self.seed = seed
        self.metric = metric

    # -------------------------------------------------------------------
    # 预处理
    # -------------------------------------------------------------------

    def preprocess(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synth_df: pd.DataFrame | None,
        target_col: str,
    ) -> dict:
        """LabelEncode + 中位数填充，返回预处理后的 numpy 数组。"""
        feat_cols = [c for c in real_test.columns if c != target_col]
        if not feat_cols:
            raise ValueError("没有可用的特征列")

        # 过滤非法合成行
        if synth_df is not None and len(synth_df) > 0:
            synth_df = self._filter_invalid(real_train, synth_df, target_col)

        X_train, X_test, X_synth = self._encode_features(
            real_train, real_test, synth_df, feat_cols
        )
        y_train = real_train[target_col]
        y_test = real_test[target_col]
        y_train_enc, y_test_enc, le = self._encode_targets(
            y_train, y_test, synth_df, target_col
        )

        return {
            "X_train": X_train, "X_test": X_test, "X_synth": X_synth,
            "y_train_enc": y_train_enc, "y_test_enc": y_test_enc,
            "label_encoder": le, "synth_df": synth_df,
            "feat_cols": feat_cols,
        }

    # -------------------------------------------------------------------
    # 推理
    # -------------------------------------------------------------------

    def evaluate(
        self,
        data: dict,
        builder: ContextBuilder,
        target_col: str,
    ) -> float:
        """用 builder 构建 context → TabPFN 推理 → 返回 score。"""
        X_ctx, y_ctx, info = builder.build(
            X_train=data["X_train"],
            y_train_enc=data["y_train_enc"],
            X_synth=data["X_synth"],
            synth_df=data["synth_df"],
            target_col=target_col,
            label_encoder=data["label_encoder"],
        )
        print(f"  TabPFN context: {len(y_ctx)} samples "
              f"(real={info['real']}, synth={info['synth']}, "
              f"strategy={builder.name})")

        return self._infer(
            X_ctx, y_ctx, data["X_test"], data["y_test_enc"]
        )

    # -------------------------------------------------------------------
    # 内部
    # -------------------------------------------------------------------

    def _infer(self, X_ctx, y_ctx, X_test, y_test_enc) -> float:
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier(random_state=self.seed, n_estimators=4)
        clf.fit(X_ctx, y_ctx)
        y_prob = clf.predict_proba(X_test)

        n_classes = len(np.unique(y_test_enc))
        if n_classes == 2:
            return float(
                average_precision_score(y_test_enc, y_prob[:, 1])
                if self.metric == "prauc"
                else roc_auc_score(y_test_enc, y_prob[:, 1])
            )
        return float(roc_auc_score(y_test_enc, y_prob,
                                   multi_class="ovr", average="macro"))

    def _filter_invalid(
        self, real_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str,
    ) -> pd.DataFrame:
        mask = pd.Series(True, index=synth_df.index)
        for col in synth_df.columns:
            if col == target_col or synth_df[col].dtype == object or \
               synth_df[col].dtype.name == "category":
                valid = set(real_df[col].dropna().unique())
                col_mask = synth_df[col].astype(str).isin(valid)
            else:
                col_mask = pd.to_numeric(synth_df[col], errors="coerce").notna()
            mask = mask & col_mask
        n = (~mask).sum()
        if n > 0:
            print(f"  [Filter] 剔除 {n}/{len(synth_df)} 条非法合成样本")
        return synth_df[mask].reset_index(drop=True) if mask.any() else synth_df

    def _encode_features(self, train, test, synth, cols):
        X_tr, X_te = train[cols].copy(), test[cols].copy()
        X_sy = synth[cols].copy() if synth is not None else None
        for col in cols:
            if X_tr[col].dtype in (object, "category"):
                le = LabelEncoder()
                all_vals = pd.concat([X_tr[col], X_te[col]], axis=0).astype(str)
                if X_sy is not None:
                    all_vals = pd.concat([all_vals, X_sy[col].astype(str)], axis=0)
                le.fit(all_vals)
                X_tr[col] = le.transform(X_tr[col].astype(str))
                X_te[col] = le.transform(X_te[col].astype(str))
                if X_sy is not None:
                    X_sy[col] = le.transform(X_sy[col].astype(str))
            else:
                m = X_tr[col].median()
                X_tr[col], X_te[col] = X_tr[col].fillna(m), X_te[col].fillna(m)
                if X_sy is not None:
                    X_sy[col] = X_sy[col].fillna(m)
        return (X_tr.values, X_te.values,
                X_sy.values if X_sy is not None else None)

    def _encode_targets(self, y_train, y_test, synth_df, target_col):
        all_labels = [y_train.astype(str), y_test.astype(str)]
        if synth_df is not None and target_col in synth_df.columns:
            all_labels.append(synth_df[target_col].astype(str))
        le = LabelEncoder()
        le.fit(pd.concat(all_labels, axis=0))
        return (le.transform(y_train.astype(str)),
                le.transform(y_test.astype(str)), le)
