---
title: "Variational Lossy Autoencoder (VLAE)"
authors: "Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, Pieter Abbeel"
institution: "UC Berkeley, OpenAI"
venue: "ICLR 2017"
arxiv_id: "1611.02731"
tags: ["paper", "VAE", "自回归模型", "PixelCNN", "表示学习", "有损压缩"]
---
# Variational Lossy Autoencoder (VLAE)

**作者:** Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, Pieter Abbeel  
**机构:** UC Berkeley, OpenAI  
**发表:** ICLR 2017  
**arXiv:** 1611.02731  
**标签:** VAE, 自回归模型, PixelCNN, 表示学习, 有损压缩, Bits-Back 编码, Normalizing Flows

---

## 一句话总结

本文从 **Bits-Back 编码**的信息论视角解释了为什么 VAE 搭配强力自回归解码器时隐变量会被忽略（"posterior collapse"），并将这一"缺陷"**反转为设计工具**：通过限制自回归解码器的感受野，可以精确控制哪些信息进入隐变量（全局结构）、哪些由解码器局部建模（纹理），从而构建一个有原则的**有损压缩**表示学习框架。同时提出用**自回归流（AF）作为先验**来提升密度估计性能，在 MNIST、OMNIGLOT、Caltech-101 上达到 SOTA。

---

## 核心问题：为什么 VAE 并不总是"自编码"？

### 背景

VAE 的 ELBO 可以分解为：

$$\mathcal{L}(\mathbf{x}; \theta) = \mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$$

第一项是重构项，第二项是正则项。当解码器 $p(\mathbf{x}\mid\mathbf{z})$ 使用简单的因子化分布 $\prod_i p(x_i \mid \mathbf{z})$ 时，重构需要 $\mathbf{z}$ 传递信息，VAE 表现为自编码器。

但当解码器使用**强力自回归模型**（如 RNN）时，$p(\mathbf{x}\mid\mathbf{z}) = \prod_i p(x_i \mid \mathbf{z}, \mathbf{x}_{<i})$ 已经能独立建模任意复杂分布而不依赖 $\mathbf{z}$，隐变量被完全忽略。这在文献中被广泛报告（Bowman et al., 2015 等），通常被归因于"优化困难"。

### 本文的关键洞察：这不仅仅是优化问题

作者从 Bits-Back 编码视角给出了更深层的解释。VAE 的期望编码长度为：

$$\mathcal{C}_{\text{BitsBack}}(\mathbf{x}) = \mathbb{E}_{\mathbf{x} \sim \text{data}} [-\mathcal{L}(\mathbf{x})]$$

将其展开：

$$\mathcal{C}_{\text{BitsBack}}(\mathbf{x}) = \mathbb{E}_{\mathbf{x} \sim \text{data}} [-\log p(\mathbf{x}) + D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}\mid\mathbf{x}))]$$

$$\geq \mathcal{H}(\text{data}) + \mathbb{E}_{\mathbf{x} \sim \text{data}} [D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}\mid\mathbf{x}))]$$

关键观察：

- Bits-Back 编码相比 Shannon 熵有一个**不可避免的额外代价** $D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}\mid\mathbf{x}))$，源于近似后验与真实后验的不匹配
- 如果 $p(\mathbf{x}\mid\mathbf{z})$ 足够强大，能在不使用 $\mathbf{z}$ 的情况下建模 $p_{\text{data}}(\mathbf{x})$，则真实后验 $p(\mathbf{z}\mid\mathbf{x}) = p(\mathbf{z})$，模型可以令 $q(\mathbf{z}\mid\mathbf{x}) = p(\mathbf{z})$，从而**避免支付这笔额外代价**
- 因此，即使优化完美，隐变量也应该被忽略——这是**信息论层面的最优选择**

### 信息偏好性质（Information Preference Property）

> **能被解码分布 $p(\mathbf{x}\mid\mathbf{z})$ 在不依赖 $\mathbf{z}$ 的情况下局部建模的信息，将被局部编码；只有剩余信息才会编入 $\mathbf{z}$。**

这个性质是 VLAE 整个方法论的基础。

---

## 方法：Variational Lossy Autoencoder

### 3.1 通过显式信息放置实现有损编码

核心思想：**不要对抗信息偏好性质，而是利用它**。

如果我们想学习只捕获**全局结构**的表示（丢弃局部纹理），只需设计一个解码器，使其：
- **能够**建模我们不想让隐变量捕获的信息（局部纹理）
- **无法**建模我们想让隐变量捕获的信息（全局结构）

具体实现：使用感受野受限的自回归解码器：

$$p_{\text{local}}(\mathbf{x}\mid\mathbf{z}) = \prod_i p(x_i \mid \mathbf{z}, \mathbf{x}_{\text{WindowAround}(i)})$$

其中 $\mathbf{x}_{\text{WindowAround}(i)}$ 是 $x_i$ 周围的一个**小局部窗口**，严格小于完整的 $\mathbf{x}_{<i}$。

**为什么有效：**

- 小窗口可以完全捕获局部统计量（如纹理、笔画宽度）→ 这些信息由解码器局部建模
- 小窗口**无法**捕获长程依赖（如物体形状、全局布局）→ 这些信息被迫编入隐变量 $\mathbf{z}$
- 通过调节窗口大小，可以**精细控制**什么信息进入 $\mathbf{z}$

**不同感受野 = 不同的有损压缩方案：**

| 解码器设计 | $\mathbf{z}$ 编码的内容 |
|-----------|---------------------|
| 因子化 $\prod_i p(x_i \mid \mathbf{z})$ | 几乎所有信息（标准 VAE） |
| 小感受野 PixelCNN | 全局结构（丢弃局部纹理） |
| 下采样感受野 | 局部高频信息（丢弃长程模式） |
| 灰度感受野 | 全局结构 + 颜色信息 |
| 完整自回归 $\prod_i p(x_i \mid \mathbf{z}, \mathbf{x}_{<i})$ | 几乎无信息（隐变量被忽略） |

### 3.2 用自回归流（AF）作为可学习先验

第二个贡献是提升密度估计性能。思路是用**自回归流**参数化先验 $p(\mathbf{z})$。

**逆自回归流（IAF）** 是一种常用的改善近似后验的方法：

$$z_i = \frac{y_i - \mu_i(y_{1:i-1})}{\sigma_i(y_{1:i-1})}$$

其中 $y$ 是简单高斯，$z$ 是更灵活的分布。

**本文提出用自回归流（AF）作为先验**，即从简单噪声 $\epsilon$ 通过 AF 变换为隐变量：$\mathbf{z} = f(\epsilon)$。

关键理论结果：AF 先验在编码器路径上等价于 IAF 后验，但在解码器路径上更深：

$$\mathcal{L}(\mathbf{x}; \theta) = \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}\mid\mathbf{x}), \epsilon = f^{-1}(\mathbf{z})} \left[\log p(\mathbf{x}\mid f(\epsilon)) + \log u(\epsilon) - \underbrace{(\log q(\mathbf{z}\mid\mathbf{x}) - \log\det\frac{d\epsilon}{d\mathbf{z}})}_{\text{IAF 后验}}\right]$$

**AF 先验 vs IAF 后验：**

| | AF 先验 | IAF 后验 |
|---|--------|---------|
| 编码器路径 | 相同 | 相同 |
| 解码器路径 | $p(\mathbf{x}\mid f(\epsilon))$ —— 更深 | $p(\mathbf{x}\mid\mathbf{z})$ —— 更浅 |
| 训练代价 | 相同 | 相同 |

结论：AF 先验在训练代价相同的情况下，获得了更有表达力的生成模型——**"免费的午餐"**。

---

## 与前序论文的联系

### 与 Hinton & van Camp (1993) 的直接联系

本文明确引用了 Hinton & van Camp (1993) 的 Bits-Back 编码论证，并将其作为分析 VAE 行为的核心工具。回顾：

- Hinton & van Camp 提出：权重的编码代价 = $D_{\mathrm{KL}}(Q \| P)$（后验与先验的 KL 散度），使用 bits-back 论证
- 本文将同样的 bits-back 推理应用于 VAE 的隐变量：编码代价 = $\mathcal{C}_{\text{BitsBack}} = -\mathcal{L}(\mathbf{x})$
- 本文进一步揭示了 bits-back 编码的**低效性**（$D_{\mathrm{KL}}(q \| p_{\text{true}})$ 项）如何导致隐变量被忽略

这是 Hinton 1993 年的 bits-back 思想在 VAE 时代最优雅的应用之一。

### 与 AlexNet 的间接联系

AlexNet 使用 Dropout 来防止过拟合，本文中 Bowman et al. (2015) 提出用 dropout 削弱自回归解码器来迫使隐变量被使用。VLAE 的方法比 dropout 更有原则：不是随机削弱解码器，而是**有目的地限制其感受野**。

---

## 实验结果

### 有损压缩效果

在 Statically Binarized MNIST 上：

| 模型 | 隐变量平均编码量 |
|------|-------------|
| 标准 VAE（因子化解码器） | 37.3 bits |
| **VLAE（PixelCNN 解码器）** | **19.2 bits** |

VLAE 学到了更"有损"的压缩。可视化显示：从 $\mathbf{z}$ 解码的图像保留了原始图像的**全局结构**（数字身份、大致形状），但**局部统计量**（笔画宽度、二值化模式）被重新生成。

### 密度估计（二值图像）

| 数据集 | 此前 SOTA | VLAE | 
|--------|---------|------|
| Static MNIST | 79.20 (PixelRNN) | **79.03** |
| Dynamic MNIST | 79.10 (IAF VAE) | **78.53** |
| OMNIGLOT | < 91.00 (Conv DRAW) | **89.83** |
| Caltech-101 | 88.48 (SpARN) | **77.36** |

在 Caltech-101 Silhouettes 上的提升尤为惊人（88.48 → 77.36）。

**AF 先验 vs IAF 后验的消融：**

| 模型 | Static MNIST NLL |
|------|-----------------|
| IAF VAE (Kingma et al., 2016) | 79.88 |
| AF VAE（本文） | 79.30 |
| VLAE（AF 先验 + PixelCNN 解码器） | **79.03** |

AF 先验确实优于等价的 IAF 后验，验证了理论预测。

### CIFAR-10（自然图像）

| 方法 | bits/dim |
|------|----------|
| Gated PixelCNN | 3.03 |
| PixelRNN | 3.00 |
| PixelCNN++ | **2.92** |
| ResNet VAE + IAF | 3.11 |
| ResNet VLAE | 3.04 |
| **DenseNet VLAE** | **2.95** |

DenseNet VLAE 在变分隐变量模型中达到 SOTA，且接近纯自回归模型 PixelCNN++ 的性能。

### 感受野大小对有损编码的影响（CIFAR-10）

| 感受野 | $\mathbf{z}$ 保留的信息 |
|--------|---------------------|
| 4×2（小） | 较详细的形状信息 |
| 5×3（中） | 中等结构信息 |
| 7×4（大） | 仅粗略形状 |
| 7×4 灰度 | 粗略形状 + 颜色信息 |

有趣发现：在 RGB 感受野下，颜色信息通常被从隐变量中省略（因为颜色在局部高度可预测）。但如果将感受野限制为灰度版本，颜色信息就被迫编入 $\mathbf{z}$——展示了解码器设计的灵活性。

---

## 实现细节

### 二值图像

- VAE 编码器/解码器：ResNet VAE（与 Kingma et al., 2016 相同）
- PixelCNN 解码器：6 层 masked convolution，$3 \times 3$ 滤波器，感受野为小局部窗口
- AF 先验：4 步自回归流，每步用 3 层 MADE（640 隐藏单元）
- 隐变量维度：64
- 优化器：Adamax，学习率 0.002
- 稳定训练：free bits（0.01 nats/data-dim）
- 参数平均：Polyak averaging（$\alpha = 0.998$）
- NLL 估计：4096 个重要性采样

### CIFAR-10

- 隐变量：16 个 $8 \times 8$ 的特征图
- AF 先验：6 步自回归流，每步用 PixelCNN（2 隐藏层，128 特征图）
- 编码器：ResNet 或 DenseNet
- 解码器：PixelCNN++（channel-autoregressive 变体）
- 训练稳定性：**Soft Free Bits**（本文提出的新技术，融合 KL annealing 和 free bits 的优点）

### Soft Free Bits（附录 C）

标准 free bits 在边界处有尖锐过渡。本文提出平滑版本：

$$\mathcal{L}_{\text{SoftFreeBits}}(\mathbf{x}; \theta) = \mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}[\log p(\mathbf{x}\mid\mathbf{z})] - \gamma \cdot D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$$

其中 $0 < \gamma \leq 1$，在线调节：KL 远高于 $\lambda$ 时增大 $\gamma$，KL 低于 $\lambda$ 时减小 $\gamma$。

---

## 局限性

- **生成速度慢**：自回归解码器需要逐像素顺序生成
- **有损编码不总是捕获"有意义"的全局信息**：在 OMNIGLOT 上，由于字符在小 patch 中已有丰富变化，语义信息有时未被保留
- **需要根据任务/数据集设计解码器**：没有通用最优的感受野大小

---

## 核心思想图示

```
标准 VAE（因子化解码器）：
  z 编码 ──→ [几乎所有信息] ──→ 因子化 p(x_i|z)
  
VLAE（受限感受野解码器）：
  z 编码 ──→ [仅全局结构]   ──→ 小窗口 PixelCNN p(x_i|z, x_window)
                                    ↑
                              [局部纹理由此建模]

完整自回归解码器（隐变量崩塌）：
  z 编码 ──→ [无信息]        ──→ 完整 RNN p(x_i|z, x_{<i})
                                    ↑
                              [所有信息由此建模]
```

---

## 关键公式速查

| 概念 | 公式 |
|------|------|
| VAE ELBO | $\mathcal{L} = \mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$ |
| Bits-Back 编码长度 | $\mathcal{C} = -\mathcal{L}(\mathbf{x}) = \mathbb{E}[-\log p(\mathbf{x})] + \mathbb{E}[D_{\mathrm{KL}}(q \| p_{\text{true}})]$ |
| 有损解码器 | $p_{\text{local}}(\mathbf{x}\mid\mathbf{z}) = \prod_i p(x_i \mid \mathbf{z}, \mathbf{x}_{\text{Window}(i)})$ |
| AF 先验 | $\mathbf{z} = f(\epsilon)$，$\log p(\mathbf{z}) = \log u(\epsilon) + \log\det\frac{d\epsilon}{d\mathbf{z}}$ |
| IAF 变换 | $z_i = \frac{y_i - \mu_i(y_{1:i-1})}{\sigma_i(y_{1:i-1})}$ |
| 信息偏好性质 | 可被解码器局部建模的信息不会进入 $\mathbf{z}$ |

---

## 论文谱系

```
Hinton & van Camp (1993)     Kingma & Welling (2013)     van den Oord et al. (2016)
  "Bits-Back 编码"              "VAE"                      "PixelCNN/RNN"
        ↓                         ↓                            ↓
        └──────────────────→ VLAE (2017) ←─────────────────────┘
                                  ↓
                    Kingma et al. (2016) "IAF"
                    → 本文提出 AF 先验作为改进
```
