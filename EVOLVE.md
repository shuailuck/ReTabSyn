# Evolve: 标签噪声辨别与软条件生成的双向增强闭环

> Evolve = **E**rro**v** label c**o**rrection **v**ia **l**atent **e**diting  
> 面向标签噪声数据的合成增强算法，构建噪声辨别器（Cleaner）与软条件生成器（Synthesizer）之间的双向增强闭环。

---

## 1. 问题定义

给定含噪数据集 $\mathcal{D}_{\text{raw}} = \{(x_i, \tilde{y}_i)\}_{i=1}^{N}$，其中 $\tilde{y}_i$ 可能不等于真实标签 $y_i$。目标：

1. **辨别噪声**：识别并剔除标签错误的样本；
2. **合成高质量数据**：在干净数据分布上生成合成样本，尤其补全类别决策边界处的数据空白；
3. **提升下游效用**：用干净集 + 合成集训练下游模型，获得比直接在含噪数据上训练更高的性能。

---

## 2. 核心思想：双向增强闭环

### 2.1 辨别器 $\mathcal{M}$ 如何提升生成器 $\mathcal{G}$

| 机制 | 痛点 | 促进方式 |
|------|------|----------|
| **噪声阻断** | 生成器直接拟合噪声分布，产生错标偏见 | $\mathcal{M}$ 基于 AUM 剔除错标样本，为 $\mathcal{G}$ 提供高纯度干净集 $\mathcal{D}_{\text{clean}}$ |
| **软条件消错** | 离散硬标签遇错标产生对抗性梯度 | $\mathcal{M}$ 输出连续概率 $q=\text{Softmax}(\mathcal{M}(x)/\tau)$，作为 $\mathcal{G}$ 的条件输入 |
| **梯度引导** | 盲目全空间采样效率低 | 用 $\mathcal{M}$ 的分类熵梯度 $\nabla_x \mathcal{H}(\mathcal{M}(x))$ 将采样拉向决策边界 |
| **逻辑门禁** | 生成样本可能 OOD | $\mathcal{M}$ 用 KL 散度 + 马氏距离过滤非法样本 |

### 2.2 生成器 $\mathcal{G}$ 如何反哺辨别器 $\mathcal{M}$

| 机制 | 痛点 | 促进方式 |
|------|------|----------|
| **流形脚手架** | 表格数据稀疏，边界处数据空白导致决策面扭曲 | $\mathcal{G}$ 在边界区合成密集数据 $\mathcal{D}_{\text{boundary}}$，撑起平滑决策面 |
| **AUM 坍塌效应** | 深层噪声被硬记忆，初始 Margin 不低 | 边界合成数据包围深层噪声，使其 Margin 暴跌跌破阈值，下一轮被精准剔除 |
| **概率校准** | 稀疏数据导致过置信 | 生成数据注入起到数据正则化，使 $q$ 贴近真实后验 |

---

## 3. 算法流程

### Phase 0: 伪样本植入

植入 $p\%$ 伪类别参照样本 $\mathcal{D}_{\text{thresh}}$（标签被强制翻转的样本），构建初始数据集：

$$\mathcal{D}_0 = \mathcal{D}_{\text{raw}} \cup \mathcal{D}_{\text{thresh}}$$

伪样本用于确定 AUM 切割阈值，无需人工指定。

### Phase 1: AUM 评估与软分布计算

训练分类器 $\mathcal{M}_k$，记录前 $T$ 个 epoch 的 Logit，计算 AUM（Area Under the Margin）：

$$\text{AUM}_i = \frac{1}{T}\sum_{t=1}^{T}\left(z_{i,y_i}^{(t)} - \max_{j \neq y_i} z_{i,j}^{(t)}\right)$$

依据伪样本 AUM 的 99% 分位数确定切割阈值 $\alpha^{(k)}$：

$$\mathcal{D}_{\text{clean}}^{(k)} = \{ (x_i, \tilde{y}_i) \mid \text{AUM}_i^{(k)} \ge \alpha^{(k)} \}$$

导出软分布 $q_i^{(k)}$ 与样本权重 $w_i^{(k)}$：

$$q_i^{(k)} = \text{Softmax}\left(\frac{\mathcal{M}_k(x_i)}{\tau}\right), \quad w_i^{(k)} = \sigma\left(\frac{\text{AUM}_i^{(k)} - \alpha^{(k)}}{\sigma_{\text{aum}}}\right)$$

### Phase 2: 加权 CVAE 训练与边界采样

以 $w_i^{(k)}$ 极小化 $\mathcal{L}_{\text{CVAE}}$，训练编码器 $E_\phi$ 与解码器 $D_\psi$。条件为软分布 $q_i^{(k)}$。

抽取边界样本对 $(x_A, x_B)$（分类最犹豫的样本），在 Latent 空间插值并解码，批量生成边界候选集 $\mathcal{D}_{\text{cand}}^{(k)}$：

$$\hat{x} = D_\psi\big(\lambda\, E_\phi(x_A, q_A) + (1-\lambda)\, E_\phi(x_B, q_B),\ \bar{q}\big)$$

### Phase 3: 双重质量门禁

- **密度检查**：剔除离 $\mathcal{D}_{\text{clean}}^{(k)}$ 马氏距离过远的 OOD 样本；
- **逻辑一致性**：要求 $D_{\text{KL}}\big(q_{\text{bound}} \parallel \text{Softmax}(\mathcal{M}_k(\hat{x}))\big) < \delta$。

保留合格样本得到 $\mathcal{D}_{\text{boundary}}^{(k)}$。

### Phase 4: 空间增强与深层噪声暴露

$$\mathcal{D}_{k+1} = \mathcal{D}_{\text{clean}}^{(k)} \cup \mathcal{D}_{\text{boundary}}^{(k)} \cup \mathcal{D}_{\text{thresh}}$$

重训 $\mathcal{M}_{k+1}$，边界脚手架数据拉平决策面，促使深层噪声 Margin 骤降。

### Phase 5: 评估收敛性

若 $\text{IoU}\big(\mathcal{D}_{\text{noisy}}^{(k)}, \mathcal{D}_{\text{noisy}}^{(k-1)}\big) \ge 1 - \epsilon$ 则终止，输出干净集与最终下游模型。

---

## 4. 伪代码

```text
输入: 含噪数据集 D_raw, 迭代轮数 K, 伪样本比例 p
输出: 干净集 D_clean, 下游模型 M

D_0 = D_raw ∪ 注入伪样本(p%)
for k = 0 to K-1:
    # Phase 1: AUM 评估
    M_k = 训练分类器(D_k, T epochs, 记录 logits)
    AUM = 计算每个样本的 Area Under Margin
    α = 伪样本 AUM 的 99% 分位数
    D_clean = {样本 | AUM >= α}
    q = Softmax(M_k(x) / τ)          # 软分布
    w = σ((AUM - α) / σ_aum)         # 样本权重

    # Phase 2: 加权 CVAE + 边界采样
    E_φ, D_ψ = 训练 CVAE(D_clean, 条件=q, 权重=w)
    (x_A, x_B) = 抽取边界样本对
    D_cand = 潜空间插值解码(x_A, x_B)

    # Phase 3: 质量门禁
    D_boundary = {候选 | 马氏距离 < 阈值 且 KL < δ}

    # Phase 4: 增强 + 重训
    D_{k+1} = D_clean ∪ D_boundary ∪ D_thresh

    # Phase 5: 收敛判定
    if IoU(D_noisy^{(k)}, D_noisy^{(k-1)}) >= 1 - ε:
        break

返回 D_clean, M_K
```

---

## 5. 超参数

| 超参数 | 含义 | 默认值 |
|--------|------|--------|
| `p` | 伪样本植入比例 | 0.05 |
| `T` | AUM 记录 epoch 数 | 10 |
| `τ` | 软分布温度 | 0.5 |
| `σ_aum` | AUM 归一化尺度 | AUM 标准差 |
| `λ` | 潜空间插值系数 | U(0,1) |
| `δ` | KL 门禁阈值 | 0.3 |
| `K` | 最大迭代轮数 | 5 |
| `ε` | 收敛阈值 | 0.05 |

---

## 6. 与现有框架的关系

- 作为 `synthesis/` 下的合成器实现，遵循 `BaseTabularSynthesizer` 接口；
- 内部包含辨别器（分类器）与生成器（CVAE）两个子模块；
- 输出为「干净集 + 边界合成集」的合并数据，可直接用于下游评估。
