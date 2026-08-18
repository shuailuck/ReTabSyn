"""
evolve/algorithm.py: Evolve 核心算法。

标签噪声辨别与软条件生成的双向增强闭环，产出两部分:
  1. 干净集 (clean set) —— 辨别器剔除噪声后的训练数据
  2. 边界合成集 (boundary set) —— 生成器在决策边界合成的数据

参考: EVOLVE.md
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    """条件变分自编码器（Conditional VAE）。"""

    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.enc_fc1 = nn.Linear(input_dim + cond_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)

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
# Evolve 算法
# ===========================================================================

class Evolve:
    """Evolve: 标签噪声辨别 + 软条件生成的双向增强闭环。

    run(df, target_col) 返回 (clean_df, boundary_df)。
    """

    def __init__(
        self,
        random_state: int = 42,
        p: float = 0.05,
        T: int = 10,
        tau: float = 0.5,
        delta: float = 0.3,
        K: int = 5,
        epsilon: float = 0.05,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        boundary_ratio: float = 1.0,
        aum_lr: float = 1e-3,
        aum_batch_size: int = 256,
        device: str | None = None,
    ):
        self.random_state = random_state
        self.p = p
        self.T = T
        self.tau = tau
        self.delta = delta
        self.K = K
        self.epsilon = epsilon
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.boundary_ratio = boundary_ratio
        self.aum_lr = aum_lr
        self.aum_batch_size = aum_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------

    def run(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """运行完整 Evolve 闭环。

        Returns
        -------
        (clean_df, boundary_df)
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._target_col = target_col
        self._columns = [c for c in df.columns if c != "is_noise"]

        # 预处理
        X, y = self._preprocess(df)

        # Phase 0: 伪样本植入
        y_with_pseudo, pseudo_mask = self._inject_pseudo_samples(X, y)

        clean_mask = np.ones(len(y), dtype=bool)
        prev_noisy = None

        # Phase 1: AUM 评估（同一个 MLP 训练 T 个 epoch 后记录 margin）
        aum, logits = self._compute_aum(X, y_with_pseudo, clean_mask)
        threshold = self._bimodal_threshold(aum, pseudo_mask)
        clean_mask = aum >= threshold
        q = self._soft_distribution(logits)
        w = self._sample_weights(aum, threshold)
        self._report_noise_detection(0, 1, "AUM", clean_mask, pseudo_mask, threshold)

        # Phase 2: 双向增强闭环（AUM + 合成器）
        for k in range(self.K):
            print(f"[Evolve] 迭代 {k + 1}/{self.K}")

            aum, logits = self._compute_aum(X, y_with_pseudo, clean_mask)
            threshold = self._bimodal_threshold(aum, pseudo_mask)
            clean_mask = aum >= threshold
            q = self._soft_distribution(logits)
            w = self._sample_weights(aum, threshold)
            self._report_noise_detection(k, self.K, "Evolve", clean_mask, pseudo_mask, threshold)

            # 加权 CVAE 训练
            self._train_cvae(X[clean_mask], q[clean_mask], w[clean_mask])

            # 收敛判定
            if prev_noisy is not None:
                iou = self._iou(prev_noisy, ~clean_mask & ~pseudo_mask)
                print(f"  IoU={iou:.4f}")
                if iou >= 1.0 - self.epsilon:
                    print(f"  [Evolve] 收敛于第 {k + 1} 轮")
                    break
            prev_noisy = (~clean_mask & ~pseudo_mask).copy()

        # 保存最终状态用于边界采样
        self._clean_mask = clean_mask
        self._X = X
        self._q = q

        # 生成边界数据
        n_boundary = int(clean_mask.sum() * self.boundary_ratio)
        boundary_df = self._generate_boundary_df(n_boundary)

        # 干净集
        clean_df = df[clean_mask].drop(columns=["is_noise"], errors="ignore").reset_index(drop=True)

        return clean_df, boundary_df

    # -------------------------------------------------------------------
    # 预处理
    # -------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame):
        self._is_noise_gt = None
        if "is_noise" in df.columns:
            self._is_noise_gt = df["is_noise"].values.astype(bool)

        feat_cols = [c for c in df.columns if c != self._target_col and c != "is_noise"]
        X = df[feat_cols].copy()
        y = df[self._target_col]

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

        self._le_y = LabelEncoder()
        y_enc = self._le_y.fit_transform(y.astype(str))

        self._num_classes = len(self._le_y.classes_)
        self._feature_dim = X_num.shape[1]

        return X_num, y_enc

    def _inverse_transform(self, X_num: np.ndarray) -> pd.DataFrame:
        X_orig = self._feature_scaler.inverse_transform(X_num)
        df = pd.DataFrame(X_orig, columns=[c for c in self._columns if c != self._target_col])
        for col, le in self._cat_encoders.items():
            df[col] = le.inverse_transform(np.round(df[col]).astype(int).clip(0, len(le.classes_) - 1))
        majority_class = self._le_y.classes_[int(np.argmax(np.bincount(self._q.argmax(1))))]
        df[self._target_col] = majority_class
        return df[self._columns]

    # -------------------------------------------------------------------
    # Phase 0: 伪样本植入
    # -------------------------------------------------------------------

    def _inject_pseudo_samples(self, X, y):
        """伪样本植入：新增假类别（第 C+1 类），将 p% 样本强制指派到假类别。"""
        y_new = y.copy()
        n_pseudo = max(int(len(y) * self.p), 1)
        rng = np.random.RandomState(self.random_state)
        pseudo_idx = rng.choice(len(y), size=n_pseudo, replace=False)

        fake_class = self._num_classes  # 假类别索引（第 C+1 类，0-indexed）
        y_new[pseudo_idx] = fake_class

        pseudo_mask = np.zeros(len(y), dtype=bool)
        pseudo_mask[pseudo_idx] = True
        return y_new, pseudo_mask

    # -------------------------------------------------------------------
    # Phase 1: AUM 评估
    # -------------------------------------------------------------------

    def _compute_aum(self, X, y, active_mask):
        # 多一个假类别输出单元
        model = MLPClassifier(X.shape[1], self._num_classes + 1, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.aum_lr)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        active_tensor = torch.tensor(active_mask, dtype=torch.bool).to(self.device)

        margins = np.zeros(len(X))
        logits_np = None
        for _ in range(self.T):
            model.train()
            # mini-batch 训练
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.aum_batch_size):
                idx = perm[i:i + self.aum_batch_size]
                batch_active = active_tensor[idx]
                optimizer.zero_grad()
                logits = model(X_tensor[idx])
                loss = F.cross_entropy(logits[batch_active], y_tensor[idx][batch_active])
                loss.backward()
                optimizer.step()

            # 记录 margin
            with torch.no_grad():
                model.eval()
                logits_np = model(X_tensor).cpu().numpy()
                margins += self._compute_margin(logits_np, y) / self.T

        return margins, logits_np

    def _compute_margin(self, logits, y):
        margin = np.zeros(len(y))
        for i in range(len(y)):
            z_true = logits[i, y[i]]
            z_other = np.delete(logits[i], y[i]).max()
            margin[i] = z_true - z_other
        return margin

    def _bimodal_threshold(self, aum, pseudo_mask):
        """用真实样本 AUM 分布的双峰谷底作为阈值。

        排除伪样本（假类别 AUM 过负，会扭曲分布）。
        若分布未呈现双峰，fallback 到中位数。
        """
        real_aum = aum[~pseudo_mask] if pseudo_mask.sum() > 0 else aum

        hist, edges = np.histogram(real_aum, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2

        # 平滑（移动平均）
        kernel = np.ones(5) / 5
        hist_smooth = np.convolve(hist, kernel, mode="same")

        # 找局部峰值
        peaks = []
        for i in range(1, len(hist_smooth) - 1):
            if (hist_smooth[i] >= hist_smooth[i - 1] and
                    hist_smooth[i] >= hist_smooth[i + 1] and hist_smooth[i] > 0):
                peaks.append(i)

        if len(peaks) < 2:
            # 无双峰，fallback 到中位数
            return float(np.median(real_aum))

        # 取两个最高峰
        peaks.sort(key=lambda i: hist_smooth[i], reverse=True)
        p1, p2 = sorted(peaks[:2])

        # 两峰之间找谷底
        valley_idx = p1 + int(np.argmin(hist_smooth[p1:p2 + 1]))

        return float(centers[valley_idx])

    def _soft_distribution(self, logits):
        # 丢弃假类别维度，只保留真实类别
        real_logits = logits[:, :self._num_classes]
        return F.softmax(torch.tensor(real_logits / self.tau, dtype=torch.float32), dim=1).numpy()

    def _sample_weights(self, aum, threshold):
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
        for _ in range(n_epochs):
            perm = torch.randperm(len(X_clean))
            for i in range(0, len(X_clean), batch_size):
                idx = perm[i:i + batch_size]
                x_batch, q_batch, w_batch = X_tensor[idx], q_tensor[idx], w_tensor[idx]

                x_hat, mu, logvar = self._cvae(x_batch, q_batch)
                recon = F.mse_loss(x_hat, x_batch, reduction="none").mean(dim=1)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                loss = (recon * w_batch + 0.01 * kld).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # -------------------------------------------------------------------
    # 边界采样
    # -------------------------------------------------------------------

    def _generate_boundary_df(self, n_samples: int) -> pd.DataFrame:
        if n_samples <= 0:
            return pd.DataFrame(columns=self._columns)

        X_clean = self._X[self._clean_mask]
        q_clean = self._q[self._clean_mask]

        if len(X_clean) == 0:
            return pd.DataFrame(columns=self._columns)

        X_tensor = torch.tensor(X_clean, dtype=torch.float32).to(self.device)
        q_tensor = torch.tensor(q_clean, dtype=torch.float32).to(self.device)

        idx_a = np.random.randint(0, len(X_clean), size=n_samples)
        idx_b = np.random.randint(0, len(X_clean), size=n_samples)

        with torch.no_grad():
            mu_a, _ = self._cvae.encode(X_tensor[idx_a], q_tensor[idx_a])
            mu_b, _ = self._cvae.encode(X_tensor[idx_b], q_tensor[idx_b])
            lam = torch.rand(n_samples, 1).to(self.device)
            z = lam * mu_a + (1 - lam) * mu_b
            q_interp = q_tensor[idx_a]
            x_hat = self._cvae.decode(z, q_interp)

        synth_X = x_hat.cpu().numpy()
        return self._inverse_transform(synth_X)

    # -------------------------------------------------------------------
    # Phase 5: 收敛判定
    # -------------------------------------------------------------------

    def _iou(self, set_a, set_b):
        inter = (set_a & set_b).sum()
        union = (set_a | set_b).sum()
        return inter / union if union > 0 else 0.0

    def _report_noise_detection(self, round_idx, total, tag, clean_mask, pseudo_mask, threshold):
        """报告本轮噪声辨别结果（含 ground truth 精度）。"""
        n_noisy = (~clean_mask & ~pseudo_mask).sum()
        print(f"[{tag} {round_idx + 1}/{total}] 阈值={threshold:.4f}, "
              f"干净={clean_mask.sum()}, 噪声={n_noisy}")
        if self._is_noise_gt is not None:
            detected = ~clean_mask & ~pseudo_mask
            gt = self._is_noise_gt
            tp = (detected & gt).sum()
            precision = tp / detected.sum() if detected.sum() > 0 else 0.0
            recall = tp / gt.sum() if gt.sum() > 0 else 0.0
            print(f"  [辨别器] Precision={precision:.4f}, Recall={recall:.4f}")
