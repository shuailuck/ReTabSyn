"""
CARTGen-IR 合成器：基于 CART 决策树的不平衡回归表格数据合成。

Paper: CARTGen-IR: Synthetic Tabular Data Generation for
       Imbalanced Regression (IDA 2025)

核心流程:
  1. 对目标列计算稀有度分数 (KDE / DenseWeight / relevance)
  2. 根据稀有度分数加权采样 parent 实例
  3. 对每个 parent 实例，逐列用 CART 树生成合成值:
     - 每列拟合一棵 CART 决策树，以其他列为特征
     - 遍历到叶节点后，从叶节点值分布中采样
  4. 可选高斯噪声 δ 防止过拟合
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KernelDensity

from synthesis.synthesizer import BaseTabularSynthesizer


class CARTGenIRSynthesizer(BaseTabularSynthesizer):
    """CARTGen-IR: 基于 CART 树的不平衡表格数据合成器。

    专为不平衡回归设计，无需对连续目标变量做离散化。
    也适用于分类任务（将目标列视为普通特征参与 CART 树拟合）。
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        density: str = "denseweight",
        alpha: float = 1.5,
        eta: float = 0.75,
        delta: float = 0.001,
        tree_max_depth: int | None = 5,
        tree_min_samples_leaf: int = 5,
        synthetic_per_parent: int = 5,
    ):
        """
        Parameters
        ----------
        density : 稀有度权重方法 "kde" | "denseweight" | "relevance"
        alpha : 稀有度指数, 越大越偏向稀有样本 {1.0, 1.5, 2.0}
        eta : 合成样本相对于原始数据的比例 {0.5, 0.75}
        delta : 高斯噪声标准差 {0, 0.001}
        tree_max_depth : CART 树最大深度 (None = 不限制)
        tree_min_samples_leaf : CART 树叶节点最小样本数
        synthetic_per_parent : 每个 parent 实例生成的合成样本数 (默认 5)
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.density = density
        self.alpha = alpha
        self.eta = eta
        self.delta = delta
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.synthetic_per_parent = synthetic_per_parent

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        np.random.seed(self.random_state)

        self._columns = list(df.columns)
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = [c for c in df.columns if c not in self._numeric_cols]
        self._df = df.copy().reset_index(drop=True)

        # Label-encode 类别列，使它们可以作为 CART 树的特征
        from sklearn.preprocessing import LabelEncoder
        self._label_encoders = {}
        df_encoded = df.copy()
        for c in self._cat_cols:
            le = LabelEncoder()
            df_encoded[c] = le.fit_transform(df[c].astype(str))
            self._label_encoders[c] = le

        # 1. 计算稀有度分数（基于目标列，即最后一列）
        target_col = self._columns[-1]
        self._rarity_scores = self._compute_rarity_scores(df[target_col])

        # 2. 为每列训练一棵 CART 树（以其他列为特征预测当前列）
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
                for leaf_id in np.unique(leaf_ids):
                    mask = leaf_ids == leaf_id
                    values, counts = np.unique(y[mask], return_counts=True)
                    leaf_dist[leaf_id] = (values, counts / counts.sum())
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
                for leaf_id in np.unique(leaf_ids):
                    mask = leaf_ids == leaf_id
                    leaf_range[leaf_id] = (y[mask].min(), y[mask].max(), y[mask].std())
                self._leaf_data[col] = leaf_range

            self._trees[col] = tree

    # -------------------------------------------------------------------
    # 稀有度分数
    # -------------------------------------------------------------------

    def _compute_rarity_scores(self, target: pd.Series) -> np.ndarray:
        """计算每个样本的稀有度分数（归一化后作为采样权重）。"""
        if target.dtype == object or target.dtype.name == "category":
            value_counts = target.value_counts(normalize=True)
            densities = target.map(value_counts).values
            rarity = 1.0 / (densities + 1e-8)
        elif self.density == "kde":
            rarity = self._rarity_kde(target)
        elif self.density == "denseweight":
            rarity = self._rarity_denseweight(target)
        elif self.density == "relevance":
            rarity = self._rarity_relevance(target)
        else:
            raise ValueError(f"未知 density 方法: {self.density}")

        rarity = np.clip(rarity, 0, None)
        rarity = rarity ** self.alpha
        total = rarity.sum()
        if total > 0:
            rarity = rarity / total
        return rarity

    def _rarity_kde(self, target: pd.Series) -> np.ndarray:
        """KDE 密度估计 → 稀有度 = 1/密度。"""
        y = target.values.astype(float).reshape(-1, 1)
        bandwidth = max(np.std(y) * 1.06 * len(y) ** (-0.2), 1e-4)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y)
        log_density = kde.score_samples(y)
        density = np.exp(log_density)
        density = np.clip(density, 1e-10, None)
        return 1.0 / density

    def _rarity_denseweight(self, target: pd.Series) -> np.ndarray:
        """DenseWeight: KDE 密度加权。"""
        y = target.values.astype(float).reshape(-1, 1)
        bandwidth = max(np.std(y) * 1.06 * len(y) ** (-0.2), 1e-4)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y)
        log_density = kde.score_samples(y)
        density = np.exp(log_density)
        density = np.clip(density, 1e-10, None)
        rarity = 1.0 / density
        norm = np.sqrt((rarity ** 2).sum())
        return rarity / norm if norm > 0 else rarity

    def _rarity_relevance(self, target: pd.Series) -> np.ndarray:
        """Relevance 函数：极端值获得更高权重。"""
        y = target.values.astype(float)
        median = np.median(y)
        mad = np.median(np.abs(y - median))
        if mad < 1e-8:
            mad = np.std(y) or 1.0
        relevance = np.abs(y - median) / mad
        relevance = relevance / (1.0 + relevance)
        return relevance

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        np.random.seed(self.random_state + 1)

        n_parents = max(int(np.ceil(n_samples / self.synthetic_per_parent)), 1)

        parent_indices = np.random.choice(
            len(self._df),
            size=n_parents,
            replace=True,
            p=self._rarity_scores,
        )

        synthetic_rows = []
        for idx in parent_indices:
            parent_row = self._df.iloc[idx].to_dict()

            if self.delta > 0:
                parent_row = self._add_noise(parent_row)

            for _ in range(self.synthetic_per_parent):
                synth_row = self._generate_one(parent_row)
                synthetic_rows.append(synth_row)

        result = pd.DataFrame(synthetic_rows, columns=self._columns)
        return result.iloc[:n_samples]

    def _add_noise(self, row: dict) -> dict:
        """对数值列添加高斯噪声。"""
        row = row.copy()
        for col in self._numeric_cols:
            if col in row and row[col] is not None:
                scale = self.delta * abs(row[col]) if row[col] != 0 else self.delta
                row[col] = row[col] + np.random.normal(0, scale)
        return row

    def _generate_one(self, parent_row: dict) -> dict:
        """对单个 parent 实例，逐列用 CART 树生成合成值。"""
        current = parent_row.copy()
        column_order = list(self._columns)
        np.random.shuffle(column_order)

        for col in column_order:
            feature_cols = [c for c in self._columns if c != col]

            # 构造查询特征：类别列需编码为数值
            query = {}
            for fc in feature_cols:
                val = current[fc]
                if fc in self._cat_cols and fc in self._label_encoders:
                    try:
                        val = self._label_encoders[fc].transform([str(val)])[0]
                    except ValueError:
                        val = 0  # unseen category → fallback
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
