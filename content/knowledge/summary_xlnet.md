---
title: "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
authors: "Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell (Carnegie Mellon University), Ruslan Salakhutdinov (Carnegie Mellon University), Quoc V. Le (Google AI Brain)"
institution: "Unknown"
venue: "arXiv 2019"
arxiv_id: "1906.08237"
tags: ["paper"]
---
# XLNet: Generalized Autoregressive Pretraining for Language Understanding

**论文**: XLNet: Generalized Autoregressive Pretraining for Language Understanding
**作者**: Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell (Carnegie Mellon University), Ruslan Salakhutdinov (Carnegie Mellon University), Quoc V. Le (Google AI Brain)
**arXiv**: 1906.08237 (2019, NeurIPS 2019)

## 核心问题

语言模型预训练主要有两种范式：

### 自回归 (AR) 语言建模

$$\max_{\theta}\quad \log p_\theta(\mathbf{x}) = \sum_{t=1}^{T} \log p_\theta(x_t \mid \mathbf{x}_{<t})$$

**优点**: 无独立性假设，无预训练-微调不一致
**缺点**: 只能利用单向上下文（左到右或右到左）

### 自编码 (AE) 预训练 (BERT)

$$\max_{\theta}\quad \log p_\theta(\bar{\mathbf{x}} \mid \hat{\mathbf{x}}) \approx \sum_{t=1}^{T} m_t \log p_\theta(x_t \mid \hat{\mathbf{x}})$$

其中 $\hat{\mathbf{x}}$ 是用 [MASK] 替换部分 token 后的损坏序列，$\bar{\mathbf{x}}$ 是被遮盖的 token。

**优点**: 可以利用双向上下文
**缺点**:
1. **独立性假设**: 被遮盖的 token 被假设为相互独立重建
2. **预训练-微调不一致**: [MASK] 符号在下游任务中从不出现

### 核心问题

**是否存在一种预训练方法能同时获得 AR 和 AE 的优点？**

## 排列语言建模 (Permutation Language Modeling)

### 核心思想

对于长度为 $T$ 的序列，存在 $T!$ 种不同的自回归分解顺序。如果模型参数在所有分解顺序间共享，模型将学会从两个方向收集信息。

### 目标函数

$$\max_{\theta}\quad \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[ \sum_{t=1}^{T} \log p_\theta(x_{z_t} \mid \mathbf{x}_{\mathbf{z}_{<t}}) \right]$$

其中：
- $\mathcal{Z}_T$ 是索引序列 $[1, 2, \ldots, T]$ 所有排列的集合
- $z_t$ 是排列 $\mathbf{z}$ 的第 $t$ 个元素
- $\mathbf{z}_{<t}$ 是排列的前 $t-1$ 个元素

### 关键点

**排列的是分解顺序，不是序列顺序**。模型：
- 保持原始序列顺序
- 使用原始序列的位置编码
- 通过注意力掩码实现分解顺序的排列

这确保了微调时模型只会遇到自然顺序的文本。

## 双流自注意力 (Two-Stream Self-Attention)

### 问题：标准参数化失效

使用标准 Transformer 参数化时：

$$p_\theta(X_{z_t} = x \mid \mathbf{x}_{\mathbf{z}_{<t}}) = \frac{\exp(e(x)^\top h_\theta(\mathbf{x}_{\mathbf{z}_{<t}}))}{\sum_{x'} \exp(e(x')^\top h_\theta(\mathbf{x}_{\mathbf{z}_{<t}}))}$$

**问题**: 表示 $h_\theta(\mathbf{x}_{\mathbf{z}_{<t}})$ 不依赖于要预测的位置 $z_t$。不同目标位置会产生相同的预测分布。

### 解决方案：目标位置感知表示

重新参数化为：

$$p_\theta(X_{z_t} = x \mid \mathbf{x}_{\mathbf{z}_{<t}}) = \frac{\exp(e(x)^\top g_\theta(\mathbf{x}_{\mathbf{z}_{<t}}, z_t))}{\sum_{x'} \exp(e(x')^\top g_\theta(\mathbf{x}_{\mathbf{z}_{<t}}, z_t))}$$

其中 $g_\theta(\mathbf{x}_{\mathbf{z}_{<t}}, z_t)$ 额外接收目标位置 $z_t$ 作为输入。

### 两种表示流

存在矛盾的需求：
1. 预测 $x_{z_t}$ 时，$g_\theta$ 只能使用位置 $z_t$，不能使用内容 $x_{z_t}$
2. 预测其他 token $x_{z_j}$ ($j > t$) 时，需要编码 $x_{z_t}$ 的内容

**解决方案**：使用两套隐藏表示：

| 表示类型 | 符号 | 访问内容 | 用途 |
| --- | --- | --- | --- |
| 内容流 (Content Stream) | $h_{z_t}$ | 上下文 + $x_{z_t}$ 自身 | 提供完整上下文信息 |
| 查询流 (Query Stream) | $g_{z_t}$ | 上下文 + 位置 $z_t$（不含 $x_{z_t}$） | 用于预测 |

### 更新规则

对于每一层 $m = 1, \ldots, M$：

$$g_{z_t}^{(m)} \leftarrow \text{Attention}(\text{Q} = g_{z_t}^{(m-1)}, \text{KV} = \mathbf{h}_{\mathbf{z}_{<t}}^{(m-1)}; \theta)$$

$$h_{z_t}^{(m)} \leftarrow \text{Attention}(\text{Q} = h_{z_t}^{(m-1)}, \text{KV} = \mathbf{h}_{\mathbf{z}_{\leq t}}^{(m-1)}; \theta)$$

**关键区别**: 查询流不能访问 $x_{z_t}$（使用 $\mathbf{z}_{<t}$），内容流可以（使用 $\mathbf{z}_{\leq t}$）。

### 初始化

- 内容流：$h_i^{(0)} = e(x_i)$（词嵌入）
- 查询流：$g_i^{(0)} = w$（可学习向量）

### 微调时

直接丢弃查询流，使用内容流作为普通 Transformer。

## 部分预测 (Partial Prediction)

完整的排列语言建模目标优化困难。解决方案：只预测分解顺序中靠后的 token。

将 $\mathbf{z}$ 分为：
- 非目标子序列 $\mathbf{z}_{\leq c}$
- 目标子序列 $\mathbf{z}_{> c}$

修改后的目标：

$$\max_{\theta}\quad \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[ \sum_{t=c+1}^{|\mathbf{z}|} \log p_\theta(x_{z_t} \mid \mathbf{x}_{\mathbf{z}_{<t}}) \right]$$

使用超参数 $K$ 控制预测比例：约 $1/K$ 的 token 被选为预测目标。

## 集成 Transformer-XL

XLNet 集成了 Transformer-XL 的两个关键技术：

### 1. 相对位置编码

基于原始序列位置计算相对距离，而非排列后的位置。

### 2. 段级循环机制 (Segment Recurrence)

给定两个连续段 $\tilde{\mathbf{x}} = \mathbf{s}_{1:T}$ 和 $\mathbf{x} = \mathbf{s}_{T+1:2T}$：

1. 处理第一个段，缓存每层的内容表示 $\tilde{\mathbf{h}}^{(m)}$
2. 处理第二个段时，注意力可以访问缓存的表示：

$$h_{z_t}^{(m)} \leftarrow \text{Attention}(\text{Q} = h_{z_t}^{(m-1)}, \text{KV} = [\tilde{\mathbf{h}}^{(m-1)}, \mathbf{h}_{\mathbf{z}_{\leq t}}^{(m-1)}]; \theta)$$

**优势**: 位置编码只依赖原始位置，因此记忆的复用与前一段的分解顺序无关。

## 多段建模

### 输入格式

与 BERT 相同：`[CLS, A, SEP, B, SEP]`

### 相对段编码

不同于 BERT 的绝对段嵌入，XLNet 使用**相对段编码**：

给定位置对 $(i, j)$：
- 同一段：$\mathbf{s}_{ij} = \mathbf{s}_+$
- 不同段：$\mathbf{s}_{ij} = \mathbf{s}_-$

计算注意力权重：$a_{ij} = (\mathbf{q}_i + \mathbf{b})^\top \mathbf{s}_{ij}$

**优势**:
1. 相对编码的归纳偏置提升泛化
2. 可以处理超过两段的输入

### 下一句预测

XLNet-Large **不使用**下一句预测目标（消融研究未显示一致改进）。

## 与 BERT 的对比

### 具体例子

句子：[New, York, is, a, city]

假设两者都选择 [New, York] 作为预测目标，XLNet 采样分解顺序 [is, a, city, New, York]：

$$\mathcal{J}_{\text{BERT}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \text{is a city})$$

$$\mathcal{J}_{\text{XLNet}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \textcolor{red}{\text{New}}, \text{is a city})$$

**XLNet 能捕获 (New, York) 之间的依赖，BERT 不能**。

### 形式化分析

定义目标-上下文对集合 $\mathcal{I} = \{(x, \mathcal{U})\}$，其中 $\mathcal{U}$ 是 $x$ 的上下文。

- BERT：如果 $\mathcal{U} \subseteq \mathcal{N}$（非目标 token），依赖被覆盖
- XLNet：如果 $\mathcal{U} \subseteq \mathcal{N} \cup \mathcal{T}_{<x}$，依赖被覆盖

**结论**: XLNet 总是能覆盖更多的依赖关系，包含更密集的有效训练信号。

## 实验结果

### 预训练配置

| 参数 | 值 |
| --- | --- |
| 层数 | 24 |
| 隐藏维度 | 1024 |
| 注意力头数 | 16 |
| FFN 隐藏维度 | 4096 |
| 部分预测 $K$ | 6 |
| 批大小 | 8192 |
| 训练步数 | 500K |
| 训练数据 | 32.89B subword pieces |

训练数据：Wikipedia + BooksCorpus + Giga5 + ClueWeb + Common Crawl

### 公平对比 (相同数据和超参数)

| 任务 | BERT-Large | XLNet-Large |
| --- | --- | --- |
| SQuAD1.1 (EM/F1) | 86.7/92.8 | **88.2/94.0** |
| SQuAD2.0 (EM/F1) | 82.8/85.5 | **85.1/87.8** |
| RACE | 75.1 | **77.4** |
| MNLI | 87.3 | **88.4** |
| RTE | 74.0 | **81.2** |
| SST-2 | 94.0 | **94.4** |

### 与 RoBERTa 对比 (扩大规模)

| 任务 | BERT | RoBERTa | XLNet |
| --- | --- | --- | --- |
| RACE | 72.0 | 83.2 | **85.4** |
| SQuAD2.0 (EM/F1) | 78.98/81.77 | 86.5/89.4 | **87.9/90.6** |
| MNLI (dev) | 86.6 | 90.2 | **90.8** |
| SST-2 (dev) | 93.2 | 96.4 | **97.0** |

### 文本分类 (错误率 ↓)

| 数据集 | BERT | XLNet |
| --- | --- | --- |
| IMDB | 4.51 | **3.20** |
| Yelp-2 | 1.89 | **1.37** |
| Yelp-5 | 29.32 | **27.05** |
| Amazon-2 | 2.63 | **2.11** |
| Amazon-5 | 34.17 | **31.67** |

### GLUE 测试集 (多任务集成)

| 任务 | RoBERTa | XLNet |
| --- | --- | --- |
| MNLI | 90.8/90.2 | **90.9/90.9** |
| QNLI | 98.9 | **99.0** |
| SST-2 | 96.7 | **97.1** |
| CoLA | 67.8 | **70.2** |
| STS-B | 92.2 | **93.0** |
| WNLI | 89.0 | **92.5** |

## 消融研究

| 模型 | RACE | SQuAD2.0 (F1) | MNLI |
| --- | --- | --- | --- |
| BERT-Base | 64.3 | 76.30 | 84.34 |
| DAE + Transformer-XL | 65.03 | 79.56 | 84.88 |
| XLNet-Base ($K=7$) | 66.05 | **81.33** | **85.84** |
| XLNet-Base ($K=6$) | **66.66** | 80.98 | 85.63 |
| - memory | 65.55 | 80.15 | 85.32 |
| - span-based pred | 65.95 | 80.61 | 85.49 |
| - bidirectional data | 66.34 | 80.65 | 85.31 |
| + next-sent pred | 66.76 | 79.83 | 85.32 |

**关键发现**:
1. Transformer-XL 和排列 LM 都对性能有贡献
2. Memory 机制对长文本任务（如 RACE）特别重要
3. Span-based prediction 和双向数据管道都有帮助
4. 下一句预测不能带来一致改进

## 注意力模式分析

### XLNet 独有的模式

1. **自排除模式 (Self-exclusion)**: 关注所有其他 token 但不关注自己，快速收集全局信息
2. **相对步进模式 (Relative stride)**: 相对于查询位置，每隔几个位置关注一次
3. **单侧遮罩模式 (One-side masked)**: 学习不关注相对右半部分

这些模式都涉及**相对位置**而非绝对位置，由 XLNet 的相对注意力机制启用。

## 核心贡献

1. **排列语言建模**: 通过考虑所有分解顺序，结合 AR 和 AE 的优点
2. **双流自注意力**: 解决排列 LM 中的目标位置感知问题
3. **Transformer-XL 集成**: 利用相对位置编码和循环机制处理长序列
4. **相对段编码**: 比绝对段嵌入更灵活
5. **全面的实验验证**: 在 20 个任务上超越 BERT

## 局限性与讨论

### 计算开销

- 双流注意力在预训练时增加计算
- 但微调时只用内容流，与 BERT 相同

### 与语言建模的关系

XLNet 弥合了语言建模和预训练之间的差距，使语言建模的进展可以直接用于预训练。

## 历史意义

1. **统一 AR 和 AE**: 首次提出能同时获得两种范式优点的预训练方法
2. **影响后续工作**: 启发了 ALBERT、ELECTRA 等模型的设计
3. **验证 Transformer-XL 的价值**: 展示了长程依赖建模对预训练的重要性

---

*本摘要基于 arXiv:1906.08237，这是 CMU 和 Google AI Brain 的重要工作，提出了排列语言建模和双流自注意力机制，通过广义自回归预训练同时获得双向上下文建模能力和 AR 模型的优点，在多项 NLU 任务上取得了当时的最佳结果。*
