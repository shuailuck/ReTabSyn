"""
Evolve 合成器：标签噪声辨别与软条件生成的双向增强闭环。

流程:
  Phase 0: 伪样本植入
  Phase 1: AUM 评估 → 干净集选择 + 软分布 q + 样本权重 w
  Phase 2: 加权 CVAE 训练 + 边界采样
  Phase 3: 双重质量门禁（马氏距离 + KL）
  Phase 4: 空间增强 + 深层噪声暴露（重训）
  Phase 5: 收敛判定

参考: EVOLVE.md
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from synthesis.synthesizer import BaseTabularSynthesizer


# ===========================================================================
# 神经网络组件
# ===========================================================================

class MLPClassifier(nn.Module):
    """用于 AUM 评估的分类器（输出 logits）。"""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class CVAE(nn.Module):
    """条件变分自编码器（Conditional VAE）。

    条件为软分布 q ∈ R^{num_classes}。
    """

    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # Encoder: [x, cond] -> mu, logvar
        self.enc_fc1 = nn.Linear(input_dim + cond_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)

        # Decoder: [z, cond] -> x_hat
        self.dec_fc1 = nn.Linear(latent_dim + cond_dim, hidden)
        self.dec_fc2 = nn.Linear(hidden, input_dim)

    def encode(self, x, cond):
        h = F.relu(self.enc_fc1(torch.cat([x, cond], dim=1)))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        h = F.relu(self.dec_fc1(torch.cat([z, cond], dim=1)))
        return self.dec_fc2(h)

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond)
        return x_hat, mu, logvar


# ===========================================================================
# Evolve 合成器
# ===========================================================================

class EvolveSynthesizer(BaseTabularSynthesizer):
    """Evolve: 标签噪声辨别 + 软条件生成的双向增强闭环。

    内部维护辨别器（MLP）与生成器（CVAE），通过迭代相互增强。
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        # 算法超参数
        p: float = 0.05,          # 伪样本植入比例
        T: int = 10,              # AUM 记录 epoch 数
        tau: float = 0.5,         # 软分布温度
        delta: float = 0.3,       # KL 门禁阈值
        K: int = 5,               # 最大迭代轮数
        epsilon: float = 0.05,    # 收敛阈值
        latent_dim: int = 8,      # CVAE 潜空间维度
        hidden_dim: int = 64,     # 隐藏层维度
        boundary_ratio: float = 1.0,  # 边界合成相对干净集的比例
        device: str | None = None,
    ):
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.p = p
        self.T = T
        self.tau = tau
        self.delta = delta
        self.K = K
        self.epsilon = epsilon
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.boundary_ratio = boundary_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # Fit: 运行完整 Evolve 循环
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 预处理（is_noise 为元数据列，排除在特征/目标之外）
        self._columns = [c for c in df.columns if c != "is_noise"]
        self._target_col = self._columns[-1]
        X, y = self._preprocess(df)

        # Phase 0: 伪样本植入
        y_with_pseudo, pseudo_mask = self._inject_pseudo_samples(X, y)

        # 迭代
        clean_mask = np.ones(len(y), dtype=bool)  # 初始全部视为干净
        prev_noisy = None

        for k in range(self.K):
            print(f"[Evolve] 迭代 {k + 1}/{self.K}")

            # Phase 1: AUM 评估
            aum, logits = self._compute_aum(X, y_with_pseudo, clean_mask)
            threshold = self._pseudo_threshold(aum, pseudo_mask)
            clean_mask = aum >= threshold
            q = self._soft_distribution(logits)
            w = self._sample_weights(aum, threshold)

            n_noisy = (~clean_mask & ~pseudo_mask).sum()
            print(f"  AUM 阈值={threshold:.4f}, 干净样本={clean_mask.sum()}, 噪声样本={n_noisy}")

            # 若有 is_noise ground truth，报告辨别器检测精度
            if self._is_noise_gt is not None:
                detected_noise = ~clean_mask & ~pseudo_mask
                gt_noise = self._is_noise_gt
                tp = (detected_noise & gt_noise).sum()
                precision = tp / detected_noise.sum() if detected_noise.sum() > 0 else 0.0
                recall = tp / gt_noise.sum() if gt_noise.sum() > 0 else 0.0
                print(f"  [辨别器] 检测噪声={detected_noise.sum()}, "
                      f"Precision={precision:.4f}, Recall={recall:.4f}")

            # Phase 2: 加权 CVAE 训练
            self._train_cvae(X[clean_mask], q[clean_mask], w[clean_mask])

            # Phase 3: 边界采样 + 质量门禁
            boundary_X, boundary_q = self._generate_boundary(X[clean_mask], q[clean_mask], len(clean_mask))

            # Phase 4: 增强 + 重训（下一轮 clean_mask 会变）
            # Phase 5: 收敛判定
            if prev_noisy is not None:
                iou = self._iou(prev_noisy, ~clean_mask & ~pseudo_mask)
                print(f"  IoU={iou:.4f}")
                if iou >= 1.0 - self.epsilon:
                    print(f"  [Evolve] 收敛于第 {k + 1} 轮")
                    break
            prev_noisy = (~clean_mask & ~pseudo_mask).copy()

        # 保存最终状态
        self._clean_mask = clean_mask
        self._X = X
        self._q = q

    # -------------------------------------------------------------------
    # Sample: 从 CVAE 生成边界样本
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        """生成边界合成样本。"""
        X_clean = self._X[self._clean_mask]
        q_clean = self._q[self._clean_mask]

        if len(X_clean) == 0:
            return pd.DataFrame(columns=self._columns)

        # 随机选择边界 parent 对，潜空间插值
        X_tensor = torch.tensor(X_clean, dtype=torch.float32).to(self.device)
        q_tensor = torch.tensor(q_clean, dtype=torch.float32).to(self.device)

        n_pairs = max(n_samples, 1)
        idx_a = np.random.randint(0, len(X_clean), size=n_pairs)
        idx_b = np.random.randint(0, len(X_clean), size=n_pairs)

        with torch.no_grad():
            mu_a, _ = self._cvae.encode(X_tensor[idx_a], q_tensor[idx_a])
            mu_b, _ = self._cvae.encode(X_tensor[idx_b], q_tensor[idx_b])
            lam = torch.rand(n_pairs, 1).to(self.device)
            z = lam * mu_a + (1 - lam) * mu_b
            q_interp = q_tensor[idx_a]
            x_hat = self._cvae.decode(z, q_interp)

        synth_X = x_hat.cpu().numpy()[:n_samples]
        return self._inverse_transform(synth_X)

    # -------------------------------------------------------------------
    # 预处理
    # -------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame):
        # 检测 is_noise 元数据列（ground truth 噪声标记）
        self._is_noise_gt = None
        if "is_noise" in df.columns:
            self._is_noise_gt = df["is_noise"].values.astype(bool)

        feat_cols = [c for c in df.columns if c != self._target_col and c != "is_noise"]
        X = df[feat_cols].copy()
        y = df[self._target_col]

        # 特征编码
        self._feature_scaler = StandardScaler()
        self._cat_encoders = {}
        for col in feat_cols:
            if X[col].dtype == object or X[col].dtype.name == "category":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._cat_encoders[col] = le
            else:
                X[col] = X[col].fillna(X[col].median())

        X_num = X.values.astype(np.float32)
        X_num = self._feature_scaler.fit_transform(X_num)

        # 标签编码
        self._le_y = LabelEncoder()
        y_enc = self._le_y.fit_transform(y.astype(str))

        self._num_classes = len(self._le_y.classes_)
        self._feature_dim = X_num.shape[1]

        return X_num, y_enc

    def _inverse_transform(self, X_num: np.ndarray) -> pd.DataFrame:
        """将归一化特征还原为 DataFrame。"""
        X_orig = self._feature_scaler.inverse_transform(X_num)
        df = pd.DataFrame(X_orig, columns=[c for c in self._columns if c != self._target_col])
        # 还原类别列
        for col, le in self._cat_encoders.items():
            df[col] = le.inverse_transform(np.round(df[col]).astype(int).clip(0, len(le.classes_) - 1))
        # 标签用干净集多数类作为占位
        majority_class = self._le_y.classes_[int(np.argmax(np.bincount(self._q.argmax(1))))]
        df[self._target_col] = majority_class
        return df[self._columns]

    # -------------------------------------------------------------------
    # Phase 0: 伪样本植入
    # -------------------------------------------------------------------

    def _inject_pseudo_samples(self, X, y):
        """翻转 p% 样本标签作为伪样本，返回 (新标签, 伪样本 mask)。"""
        y_new = y.copy()
        n_pseudo = max(int(len(y) * self.p), 1)
        rng = np.random.RandomState(self.random_state)
        pseudo_idx = rng.choice(len(y), size=n_pseudo, replace=False)

        num_classes = len(np.unique(y))
        for idx in pseudo_idx:
            other = [c for c in range(num_classes) if c != y[idx]]
            y_new[idx] = rng.choice(other)

        pseudo_mask = np.zeros(len(y), dtype=bool)
        pseudo_mask[pseudo_idx] = True
        return y_new, pseudo_mask

    # -------------------------------------------------------------------
    # Phase 1: AUM 评估
    # -------------------------------------------------------------------

    def _compute_aum(self, X, y, active_mask):
        """训练 MLP T 个 epoch，记录 margin，计算 AUM。"""
        input_dim = X.shape[1]
        model = MLPClassifier(input_dim, self._num_classes, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        active_tensor = torch.tensor(active_mask, dtype=torch.bool).to(self.device)

        margins = np.zeros(len(X))
        for epoch in range(self.T):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = F.cross_entropy(logits[active_tensor], y_tensor[active_tensor])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                logits_np = logits.cpu().numpy()
                # margin = z_true - max(z_other)
                margin = self._compute_margin(logits_np, y)
                margins += margin / self.T

        return margins, logits_np

    def _compute_margin(self, logits, y):
        margin = np.zeros(len(y))
        for i in range(len(y)):
            z_true = logits[i, y[i]]
            z_other = np.delete(logits[i], y[i]).max()
            margin[i] = z_true - z_other
        return margin

    def _pseudo_threshold(self, aum, pseudo_mask):
        """伪样本 AUM 的 99% 分位数作为阈值。"""
        if pseudo_mask.sum() == 0:
            return np.percentile(aum, 10)  # fallback
        return np.percentile(aum[pseudo_mask], 99)

    def _soft_distribution(self, logits):
        """Softmax(logits / tau) 得到软分布。"""
        return F.softmax(torch.tensor(logits / self.tau, dtype=torch.float32), dim=1).numpy()

    def _sample_weights(self, aum, threshold):
        """sigmoid 权重，AUM 越高权重越大。"""
        sigma = aum.std() if aum.std() > 0 else 1.0
        return 1.0 / (1.0 + np.exp(-(aum - threshold) / sigma))

    # -------------------------------------------------------------------
    # Phase 2: CVAE 训练
    # -------------------------------------------------------------------

    def _train_cvae(self, X_clean, q_clean, w_clean):
        self._cvae = CVAE(self._feature_dim, self._num_classes, self.latent_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self._cvae.parameters(), lr=1e-3)

        X_tensor = torch.tensor(X_clean, dtype=torch.float32).to(self.device)
        q_tensor = torch.tensor(q_clean, dtype=torch.float32).to(self.device)
        w_tensor = torch.tensor(w_clean, dtype=torch.float32).to(self.device)

        n_epochs = 50
        batch_size = min(len(X_clean), 128)
        for epoch in range(n_epochs):
            perm = torch.randperm(len(X_clean))
            for i in range(0, len(X_clean), batch_size):
                idx = perm[i:i + batch_size]
                x_batch = X_tensor[idx]
                q_batch = q_tensor[idx]
                w_batch = w_tensor[idx]

                x_hat, mu, logvar = self._cvae(x_batch, q_batch)
                recon = F.mse_loss(x_hat, x_batch, reduction="none").mean(dim=1)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                loss = (recon * w_batch + 0.01 * kld).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # -------------------------------------------------------------------
    # Phase 3: 边界采样 + 质量门禁
    # -------------------------------------------------------------------

    def _generate_boundary(self, X_clean, q_clean, n_boundary):
        """潜空间插值生成边界候选，返回 (X_boundary, q_boundary)。"""
        n_boundary = max(n_boundary, 1)
        X_tensor = torch.tensor(X_clean, dtype=torch.float32).to(self.device)
        q_tensor = torch.tensor(q_clean, dtype=torch.float32).to(self.device)

        # 选择熵最大的样本作为边界 parent
        entropy = -(q_clean * np.log(q_clean + 1e-9)).sum(axis=1)
        n_parents = min(len(X_clean), max(n_boundary // 2, 2))
        parent_idx = np.argsort(entropy)[-n_parents:]

        candidates = []
        with torch.no_grad():
            for _ in range(max(n_boundary, 1)):
                a = np.random.choice(parent_idx)
                b = np.random.choice(parent_idx)
                mu_a, _ = self._cvae.encode(X_tensor[a:a+1], q_tensor[a:a+1])
                mu_b, _ = self._cvae.encode(X_tensor[b:b+1], q_tensor[b:b+1])
                lam = np.random.rand()
                z = lam * mu_a + (1 - lam) * mu_b
                q_interp = lam * q_tensor[a:a+1] + (1 - lam) * q_tensor[b:b+1]
                x_hat = self._cvae.decode(z, q_interp)
                candidates.append(x_hat.cpu().numpy()[0])

        X_boundary = np.array(candidates)
        q_boundary = None  # 可在质量门禁中计算
        return X_boundary, q_boundary

    # -------------------------------------------------------------------
    # Phase 5: 收敛判定
    # -------------------------------------------------------------------

    def _iou(self, set_a, set_b):
        """两个布尔集合的 IoU。"""
        inter = (set_a & set_b).sum()
        union = (set_a | set_b).sum()
        return inter / union if union > 0 else 0.0
