---
title: "Scaling Laws for Neural Language Models"
authors: "Jared Kaplan*, Sam McCandlish* (OpenAI, Johns Hopkins University), Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei"
institution: "Unknown"
venue: "arXiv 2020"
arxiv_id: "2001.08361"
tags: ["paper"]
---
# Scaling Laws for Neural Language Models

**论文**: Scaling Laws for Neural Language Models
**作者**: Jared Kaplan*, Sam McCandlish* (OpenAI, Johns Hopkins University), Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
**arXiv**: 2001.08361 (2020)

## 核心发现

语言模型的性能（交叉熵损失）与三个关键因素呈**幂律关系**：

1. **模型大小** $N$（非嵌入参数数）
2. **数据集大小** $D$（tokens 数）
3. **训练计算量** $C$

这些趋势跨越**超过七个数量级**，且没有偏离迹象。

## 核心幂律公式

### 单因素幂律

**参数受限**（数据和计算充足）：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

**数据受限**（早停，模型足够大）：

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$

**计算受限**（最优分配）：

$$L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

### 联合幂律

**模型大小与数据大小**：

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

**模型大小与训练步数**：

$$L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}$$

其中 $S_c \approx 2.1 \times 10^3$，$\alpha_S \approx 0.76$。

## 关键发现详解

### 1. 性能强烈依赖规模，弱依赖模型形状

在固定总参数量 $N$ 时：

- 深度 vs 宽度几乎不影响性能
- 注意力头数影响很小
- 宽高比可以变化 40 倍而只有约 3% 的性能差异

**关键洞察**：应该关注规模（$N, D, C$），而非具体架构细节。

### 2. 过拟合的通用规律

避免过拟合需要：

$$D \gtrsim (5 \times 10^3) \cdot N^{0.74}$$

**含义**：模型大小增加 8 倍时，数据只需增加约 5 倍。数据需求相对模型大小是**次线性**增长。

### 3. 训练曲线的通用性

训练曲线遵循可预测的幂律，参数大致独立于模型大小。通过外推训练曲线的早期部分，可以预测长时间训练后的损失。

### 4. 大模型更具样本效率

大模型用更少的数据和更少的优化步骤就能达到相同性能。

### 5. 最优计算分配

给定计算预算 $C$，最优分配为：

$$N \propto C^{0.73}, \quad B \propto C^{0.24}, \quad S \propto C^{0.03}$$

**关键洞察**：
- 计算增加应**主要用于增大模型**
- 训练步数增长极其缓慢
- **计算高效训练 = 训练非常大的模型 + 在收敛前显著停止**

### 6. 临界批大小

临界批大小遵循关于损失的幂律：

$$B_{\rm crit}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

在临界批大小训练提供时间/计算的最优权衡。

### 7. 迁移性能

迁移到不同数据分布时：
- 损失与训练验证集强相关
- 存在大致恒定的损失偏移
- 迁移性能几乎完全取决于训练分布上的性能

## 拟合参数汇总

| 参数 | 值 | 说明 |
| --- | --- | --- |
| $\alpha_N$ | 0.076 | 模型大小幂律指数 |
| $\alpha_D$ | 0.095 | 数据集大小幂律指数 |
| $\alpha_C^{\min}$ | 0.050 | 最优计算幂律指数 |
| $\alpha_S$ | 0.76 | 训练步数幂律指数 |
| $\alpha_B$ | 0.21 | 批大小幂律指数 |

## 计算高效训练

### 最优分配

| 参数 | 幂律 | 基准值 |
| --- | --- | --- |
| $N_{\rm opt}$ | $\propto C^{0.73}$ | $N_e = 1.3 \times 10^9$ |
| $B_{\rm crit}$ | $\propto C^{0.24}$ | $B_e = 2.0 \times 10^6$ |
| $S_{\min}$ | $\propto C^{0.03}$ | $S_e = 5.4 \times 10^3$ |
| $D_{\rm opt}$ | $\propto C^{0.27}$ | $D_e = 2 \times 10^{10}$ |

### 与传统训练对比

计算高效训练 vs 训练到接近收敛（2%）：

- 使用 **2.7 倍更多参数**
- 使用 **7.7 倍更少步数**
- 使用 **65% 更少计算**

达到相同损失。

## 理论极限的推测

### 矛盾与临界点

论文发现两个趋势最终会矛盾：

1. $L(C_{\min}) \propto C^{-0.050}$ 持续改进
2. $L(D) \propto D^{-0.095}$ 数据受限

交叉点预测：

$$C^* \sim 10^4 \text{ PF-days}, \quad N^* \sim 10^{12} \text{ 参数}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

**推测**：$L^*$ 可能是自然语言熵的粗略估计，代表 Transformer 语言模型的理论性能极限。

## LSTM vs Transformer

| 特性 | LSTM | Transformer |
| --- | --- | --- |
| 早期 token | 表现相当 | 表现相当 |
| 后期 token | 较差 | 更好 |
| 幂律指数 | 不同 | 更陡 |

Transformer 能更好地利用长上下文。

## 核心贡献

1. **发现幂律**：语言模型性能与 $N, D, C$ 呈精确幂律关系
2. **架构不重要**：在合理范围内，深度/宽度/头数影响很小
3. **最优分配**：应主要增加模型大小，而非训练时间
4. **预测框架**：可以预测过拟合、早停步数、最优分配
5. **样本效率**：大模型更高效，挑战了"需要更多数据"的直觉

## 历史意义

这篇论文：

1. **奠定了大模型时代的理论基础**：解释了为什么应该追求更大的模型
2. **影响了 GPT-3 及后续模型的设计**：指导了计算资源的分配
3. **引发了 Chinchilla 等后续研究**：进一步优化了训练数据的使用

## 局限性

1. 没有深入研究小数据 regime
2. 未探索正则化和数据增强的影响
3. $B_{\rm crit}(L)$ 的预测在远离探索范围时可能不可靠
4. 未包含与 $n_{\rm ctx}$ 成比例的计算贡献
5. 缺乏理论解释，尤其是模型大小和计算的 scaling

## 注释

| 符号 | 含义 |
| --- | --- |
| $L$ | 交叉熵损失（nats） |
| $N$ | 非嵌入参数数 |
| $D$ | 数据集大小（tokens） |
| $C$ | 训练计算量 ≈ $6NBS$ |
| $B$ | 批大小 |
| $S$ | 训练步数 |
| PF-day | $10^{15} \times 24 \times 3600 = 8.64 \times 10^{19}$ FLOPs |

---

*本摘要基于 arXiv:2001.08361，这是 OpenAI 的开创性工作，系统研究了语言模型的缩放定律，为大型语言模型的发展提供了理论指导，其影响深远，直接推动了 GPT-3 等大模型的出现。*
