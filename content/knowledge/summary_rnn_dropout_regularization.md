---
title: "Recurrent Neural Network Regularization"
authors: "Wojciech Zaremba (NYU), Ilya Sutskever, Oriol Vinyals (Google Brain)"
institution: "Unknown"
venue: "ICLR 2015"
arxiv_id: "1409.2329"
tags: ["paper"]
---
# Recurrent Neural Network Regularization

**论文信息**

- 标题: Recurrent Neural Network Regularization
- 作者: Wojciech Zaremba (NYU), Ilya Sutskever, Oriol Vinyals (Google Brain)
- 发表: ICLR 2015
- arXiv: [1409.2329](https://arxiv.org/abs/1409.2329)
- 代码: [github.com/wojzaremba/lstm](https://github.com/wojzaremba/lstm)

---

## 核心贡献

这篇论文提出了**如何正确地将 Dropout 应用于 LSTM**，解决了之前 Dropout 在 RNN 上效果不佳的问题。核心思想非常简单：**只在非循环连接上应用 Dropout，而不在循环连接上应用**。

---

## 问题背景

### Dropout 在 RNN 上的困境

Dropout 是前馈神经网络最成功的正则化技术，但直接应用于 RNN 效果不好：

| 网络类型 | 直接应用 Dropout |
| --- | --- |
| 前馈神经网络 | 效果很好 |
| RNN/LSTM | 效果差，甚至有害 |

**原因**：Bayer et al. (2013) 认为循环连接会放大噪声，从而损害学习。

**结果**：实践中只能使用较小的 RNN，因为大模型容易过拟合。

---

## LSTM 回顾

### 符号定义

- 下标表示时间步，上标表示层
- $h^l_t \in \mathbb{R}^n$：第 $l$ 层在时间步 $t$ 的隐藏状态
- $c^l_t \in \mathbb{R}^n$：第 $l$ 层在时间步 $t$ 的细胞状态
- $T_{n,m}: \mathbb{R}^n \rightarrow \mathbb{R}^m$：仿射变换（$Wx + b$）
- $\odot$：逐元素乘法

### 经典 RNN

$$h^l_t = f(T_{n,n} h^{l-1}_t + T_{n,n} h^l_{t-1})$$

其中 $f \in \{\mathrm{sigm}, \tanh\}$。

### LSTM 公式

$$\text{LSTM}: h^{l-1}_t, h^l_{t-1}, c^l_{t-1} \rightarrow h^l_t, c^l_t$$

$$\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \mathrm{sigm} \\ \mathrm{sigm} \\ \mathrm{sigm} \\ \tanh \end{pmatrix} T_{2n, 4n} \begin{pmatrix} h^{l-1}_t \\ h^l_{t-1} \end{pmatrix}$$

$$c^l_t = f \odot c^l_{t-1} + i \odot g$$

$$h^l_t = o \odot \tanh(c^l_t)$$

其中：
- $i$：输入门 (input gate)
- $f$：遗忘门 (forget gate)
- $o$：输出门 (output gate)
- $g$：输入调制门 (input modulation gate)

---

## 核心方法：正确的 Dropout 应用方式

### 关键洞察

**只在非循环连接上应用 Dropout，保持循环连接不变。**

### 数学表示

设 $\mathbf{D}$ 为 Dropout 算子（随机将部分元素置零）：

$$\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \mathrm{sigm} \\ \mathrm{sigm} \\ \mathrm{sigm} \\ \tanh \end{pmatrix} T_{2n, 4n} \begin{pmatrix} \mathbf{D}(h^{l-1}_t) \\ h^l_{t-1} \end{pmatrix}$$

$$c^l_t = f \odot c^l_{t-1} + i \odot g$$

$$h^l_t = o \odot \tanh(c^l_t)$$

注意：**Dropout 只作用于 $h^{l-1}_t$（来自下层的输入），不作用于 $h^l_{t-1}$（来自上一时间步的循环连接）**。

### 图示理解

```
时间步:    t-2      t-1       t       t+1      t+2

输出 y:    y_{t-2}  y_{t-1}   y_t     y_{t+1}  y_{t+2}
            ↑        ↑        ↑        ↑        ↑
           [D]      [D]      [D]      [D]      [D]    ← Dropout (垂直连接)
            ↑        ↑        ↑        ↑        ↑
第2层:    [LSTM] → [LSTM] → [LSTM] → [LSTM] → [LSTM]
            ↑        ↑        ↑        ↑        ↑     ← 水平连接无 Dropout
           [D]      [D]      [D]      [D]      [D]    ← Dropout (垂直连接)
            ↑        ↑        ↑        ↑        ↑
第1层:    [LSTM] → [LSTM] → [LSTM] → [LSTM] → [LSTM]
            ↑        ↑        ↑        ↑        ↑     ← 水平连接无 Dropout
           [D]      [D]      [D]      [D]      [D]    ← Dropout (垂直连接)
            ↑        ↑        ↑        ↑        ↑
输入 x:    x_{t-2}  x_{t-1}   x_t     x_{t+1}  x_{t+2}
```

- **虚线（垂直方向）**：应用 Dropout
- **实线（水平方向）**：不应用 Dropout

---

## 为什么这样有效？

### 信息流分析

考虑信息从时间步 $t-2$ 流向 $t+2$ 的预测：

1. 信息被 Dropout 扰动的次数恰好是 $L + 1$（$L$ 是网络深度）
2. 这个次数**与时间步数无关**

### 直觉解释

- **Dropout 的作用**：破坏信息，迫使网络更鲁棒地进行计算
- **循环连接的作用**：记住长期信息

如果在循环连接上也应用 Dropout：
- 每多一个时间步，信息就多被破坏一次
- 长期记忆能力被严重损害

通过只在非循环连接上应用 Dropout：
- 保留了 LSTM 的长期记忆能力
- 同时获得了 Dropout 的正则化效果

---

## 实验结果

### 1. 语言建模（Penn Treebank）

| 模型 | 验证集困惑度 | 测试集困惑度 |
| --- | --- | --- |
| Pascanu et al. (2013) | - | 107.5 |
| Cheng et al. | - | 100.0 |
| 无正则化 LSTM | 120.7 | 114.5 |
| **中等正则化 LSTM** | 86.2 | 82.7 |
| **大型正则化 LSTM** | 82.2 | **78.4** |

**模型配置**：

| 配置 | 中等 LSTM | 大型 LSTM |
| --- | --- | --- |
| 层数 | 2 | 2 |
| 每层单元数 | 650 | 1500 |
| Dropout 率 | 50% | 65% |
| 展开步数 | 35 | 35 |
| 训练轮数 | 39 | 55 |
| 梯度裁剪 | 5 | 10 |

### 2. 语音识别（冰岛语数据集）

| 模型 | 训练集准确率 | 验证集准确率 |
| --- | --- | --- |
| 无正则化 LSTM | 71.6% | 68.9% |
| **正则化 LSTM** | 69.4% | **70.5%** |

注意：训练准确率下降但验证准确率上升，这正是 Dropout 防止过拟合的典型表现。

### 3. 机器翻译（英法翻译）

| 模型 | 测试困惑度 | 测试 BLEU |
| --- | --- | --- |
| 无正则化 LSTM | 5.8 | 25.9 |
| **正则化 LSTM** | 5.0 | 29.03 |
| LIUM 短语系统 | - | 33.30 |

**模型配置**：
- 4 层 LSTM
- 每层 1000 单元
- 英语词汇 160k，法语词汇 80k
- Dropout 率 20%

### 4. 图像描述生成（MSCOCO）

| 模型 | 测试困惑度 | 测试 BLEU |
| --- | --- | --- |
| 无正则化模型 | 8.47 | 23.5 |
| **正则化模型** | 7.99 | 24.3 |
| 10 个无正则化模型集成 | 7.5 | 24.4 |

**有趣发现**：单个正则化模型的效果接近 10 个无正则化模型的集成。

---

## 实验细节

### Penn Treebank 数据集

- 训练集：929k 词
- 验证集：73k 词
- 测试集：82k 词
- 词汇量：10k

### 训练技巧

1. **隐藏状态初始化**：用上一个 minibatch 的最终隐藏状态初始化下一个 minibatch
2. **梯度裁剪**：按 minibatch 大小归一化后裁剪
3. **学习率调度**：初始学习率 1，后期逐步衰减

### 生成样本

论文展示了用大型正则化 LSTM 生成的文本样本：

> *the meaning of life is* that only if an end would be of the whole supplier. widespread rules are regarded as the companies of refuses to deliver...

虽然语法正确但语义无意义，这反映了 2014 年语言模型的水平。

---

## 与其他工作的关系

### 同期独立发现

Pham et al. (2013) 独立发现了相同的方法，并应用于手写识别。

### 后续发展

这篇论文的 Dropout 方法成为 RNN 正则化的标准做法，后来又有更多变体：

| 方法 | 年份 | 特点 |
| --- | --- | --- |
| 本文方法 | 2014 | 只在非循环连接应用 |
| Variational Dropout | 2015 | 在循环连接上使用相同的 mask |
| Zoneout | 2016 | 随机保持隐藏状态不变 |
| DropConnect | - | 在权重上应用 Dropout |

---

## 核心要点总结

1. **问题**：标准 Dropout 在 RNN 上效果差，因为循环连接会放大噪声

2. **解决方案**：只在非循环连接（层间/输入输出）应用 Dropout

3. **原理**：信息被 Dropout 扰动的次数只与网络深度相关，与时间步数无关

4. **效果**：在语言建模、语音识别、机器翻译、图像描述等任务上都有显著提升

5. **实践意义**：使得可以训练更大的 RNN 模型而不过拟合

---

## 与相关论文的联系

这篇论文与本仓库中其他 LSTM 相关论文形成知识链：

1. **Understanding LSTM Networks (Olah, 2015)**：解释 LSTM 的工作原理
2. **本文 (Zaremba et al., 2014)**：如何正则化 LSTM
3. **The Unreasonable Effectiveness of RNNs (Karpathy, 2015)**：展示 LSTM 的强大能力

这三篇文章共同构成了 2014-2015 年 LSTM 研究的核心文献。
