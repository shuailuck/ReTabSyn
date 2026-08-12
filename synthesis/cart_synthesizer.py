"""
CART 合成器：基于 CART 决策树的表格数据合成。

Paper: Reiter (2005) — Using CART to Generate Partially Synthetic,
       Public Use Microdata (Journal of Official Statistics, 21(3), 441-462)

核心流程:
  1. 为每列训练一棵 CART 树，以其他列为特征
  2. 均匀随机选择一个已存在的行作为 parent
  3. 对 parent 逐列用 CART 树重新生成值: 遍历到叶节点后从叶节点分布中采样
  4. 无序不平衡处理，纯 CART 结构合成
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from synthesis.synthesizer import BaseTabularSynthesizer


class CARTSynthesizer(BaseTabularSynthesizer):
    """CART 合成器 (Reiter 2005): 基于 CART 树的表格数据合成。

    每列单独训练一棵 CART 树，合成时逐列用叶节点分布采样新值。
    与 CARTGen-IR 的区别: 无稀有度加权，无噪声注入，纯分布合成。
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        tree_max_depth: int | None = 5,
        tree_min_samples_leaf: int = 5,
    ):
        """
        Parameters
        ----------
        tree_max_depth : CART 树最大深度 (None = 不限制)
        tree_min_samples_leaf : 叶节点最小样本数
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        np.random.seed(self.random_state)

        self._columns = list(df.columns)
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = [c for c in df.columns if c not in self._numeric_cols]
        self._df = df.copy().reset_index(drop=True)

        # Label-encode 类别列作为树特征
        self._label_encoders = {}
        df_encoded = df.copy()
        for c in self._cat_cols:
            le = LabelEncoder()
            df_encoded[c] = le.fit_transform(df[c].astype(str))
            self._label_encoders[c] = le

        # 为每列训练一棵 CART 树
        self._trees = {}
        self._leaf_data = {}

        for col in self._columns:
            feature_cols = [c for c in self._columns if c != col]
            X = df_encoded[feature_cols]
            y = df[col]

            if col in self._cat_cols:
                tree = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    min_samples_leaf=self.tree_min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
                leaf_ids = tree.apply(X)
                leaf_dist = {}
                for lid in np.unique(leaf_ids):
                    mask = leaf_ids == lid
                    values, counts = np.unique(y[mask], return_counts=True)
                    leaf_dist[lid] = (values, counts / counts.sum())
                self._leaf_data[col] = leaf_dist
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.tree_max_depth,
                    min_samples_leaf=self.tree_min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
                leaf_ids = tree.apply(X)
                leaf_range = {}
                for lid in np.unique(leaf_ids):
                    mask = leaf_ids == lid
                    leaf_range[lid] = (y[mask].min(), y[mask].max(), y[mask].std())
                self._leaf_data[col] = leaf_range

            self._trees[col] = tree

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        np.random.seed(self.random_state + 1)

        # 均匀随机选择 parent 行
        parent_indices = np.random.choice(len(self._df), size=n_samples, replace=True)

        synthetic_rows = []
        for idx in parent_indices:
            parent_row = self._df.iloc[idx].to_dict()
            synth_row = self._generate_one(parent_row)
            synthetic_rows.append(synth_row)

        return pd.DataFrame(synthetic_rows, columns=self._columns)

    def _generate_one(self, parent_row: dict) -> dict:
        """逐列用 CART 树重新生成值。"""
        current = parent_row.copy()
        column_order = list(self._columns)
        np.random.shuffle(column_order)

        for col in column_order:
            feature_cols = [c for c in self._columns if c != col]

            query = {}
            for fc in feature_cols:
                val = current[fc]
                if fc in self._cat_cols and fc in self._label_encoders:
                    try:
                        val = self._label_encoders[fc].transform([str(val)])[0]
                    except ValueError:
                        val = 0
                query[fc] = val

            X_query = pd.DataFrame([query], columns=feature_cols)
            tree = self._trees[col]
            leaf_id = tree.apply(X_query)[0]

            if col in self._cat_cols:
                leaf_dist = self._leaf_data[col]
                if leaf_id in leaf_dist:
                    values, probs = leaf_dist[leaf_id]
                    current[col] = np.random.choice(values, p=probs)
            else:
                leaf_range = self._leaf_data[col]
                if leaf_id in leaf_range:
                    lo, hi, std = leaf_range[leaf_id]
                    if hi > lo:
                        current[col] = np.random.uniform(lo, hi)
                    else:
                        current[col] = lo + np.random.normal(0, max(std, 1e-8))

        return current
