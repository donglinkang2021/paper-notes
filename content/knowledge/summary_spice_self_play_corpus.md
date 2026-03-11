---
title: "SPICE: Self-Play In Corpus Environments Improves Reasoning"
authors: "Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, Jason Weston"
institution: "FAIR at Meta, National University of Singapore"
venue: "arXiv 2025"
arxiv_id: "2510.24684"
tags: ["self-play", "corpus-grounded", "reasoning", "reinforcement-learning", "curriculum-learning", "document-mining"]
---

# SPICE: 基于语料库环境的自博弈推理改进

## 核心贡献

SPICE (Self-Play In Corpus Environments) 提出了一种突破性的强化学习框架，通过将大规模文档语料库作为外部环境，解决了现有无根据自博弈方法的幻觉和信息对称性问题。单个模型同时扮演两个角色：Challenger（出题者）从语料库中挖掘文档生成多样化推理任务，Reasoner（推理者）解决这些任务。通过对抗性动态，Challenger 在 Reasoner 能力边界自动创建课程，而语料库根据提供了持续改进所需的丰富、近乎无穷的外部信号。在数学推理（+8.9%）和通用推理（+9.8%）基准上实现了一致性提升。

## 方法详解

### 核心架构

SPICE 采用双角色自博弈框架：

1. **Challenger（出题者）**：从文档语料库中挖掘内容，生成基于文档的推理问题
   - 输入：从 NaturalReasoning 或 Nemotron-CC-Math 语料库采样的文档
   - 输出：多选题（MCQ）或自由形式问题
   - 目标：生成处于 Reasoner 能力边界的挑战性任务

2. **Reasoner（推理者）**：解决 Challenger 提出的问题
   - 输入：Challenger 生成的问题
   - 输出：结构化推理过程和答案
   - 目标：最大化正确率

### 语料库根据的关键作用

与纯自博弈方法（如 R-Zero）不同，SPICE 的核心创新在于将外部文档作为环境：

- **信息不对称**：Challenger 看到文档，Reasoner 只看到问题，打破了纯自博弈的信息对称性
- **外部真实信号**：文档提供客观知识基础，防止模型陷入自我强化的幻觉
- **近乎无穷的多样性**：大规模语料库（20,000 文档）提供持续的新鲜训练信号

### 奖励函数设计

**Challenger 奖励（高斯方差）**：
$$R_{\text{Ch}} = \exp\left(-\frac{(p - 0.5)^2}{2\sigma^2}\right)$$

其中 $p$ 是 Reasoner 多次采样中最常见答案的比例。该奖励在 $p=0.5$ 时达到峰值，鼓励 Challenger 生成不太简单也不太难的问题——正好处于 Reasoner 能力边界。

**Reasoner 奖励**：
- 二元正确性奖励（0 或 1）
- 格式错误惩罚：-0.1

### 训练算法

使用 DrGRPO（Distributed Regularized Group Relative Policy Optimization）：

1. 从语料库采样文档
2. Challenger 对每个文档生成 8 个候选问题
3. Reasoner 对每个问题生成 8 个候选答案
4. 计算奖励和优势函数
5. 同时更新 Challenger 和 Reasoner 策略

训练配置：
- 640 次迭代
- 批大小 128
- 温度 1.0
- KL 散度正则化防止偏离参考模型

### 问题生成策略

Challenger 被训练生成两种类型的问题：

1. **多选题（MCQ）**：提供结构化答案选项，便于可靠验证
2. **自由形式问题**：鼓励灵活推理

问题设计原则：
- 100% 自包含，不引用原文档
- 需要多步推理，不能通过查找单个事实回答
- 综合文档中的多个概念
- 目标难度：HARD 或 EXTRA HARD

## 实验结果

### 基础模型

在多个模型家族上验证：
- Qwen3-4B-Base
- Llama-3.1-8B-Base
- Llama-3.3-70B-Instruct

### 主要结果（Qwen3-4B-Base）

| 方法 | 数学推理 | 通用推理 | 总体平均 |
|---|---|---|---|
| Base model | 41.7 | 27.2 | 36.5 |
| R-Zero（无语料库） | 42.3 | 27.8 | 37.0 |
| Absolute Zero | 43.8 | 28.5 | 38.0 |
| Strong Challenger | 48.1 | 32.1 | 42.1 |
| **SPICE** | **50.6** | **35.0** | **44.9** |

**关键发现**：
1. SPICE 相比基础模型在数学推理上提升 +8.9%，通用推理上提升 +9.8%
2. 相比无语料库的 R-Zero，SPICE 提升显著（+7.9% 总体）
3. 语料库根据是持续改进的关键

### 具体基准表现

**数学推理**：
- MATH-500: 81.2%（基线 71.6%）
- GSM8K: 91.8%（基线 86.3%）
- OlympiadBench: 46.3%（基线 37.2%）
- AIME'24: 14.8%（基线 9.8%）
- AIME'25: 22.1%（基线 11.9%）

**通用推理**：
- MMLU-Pro: 56.5%（基线 48.2%）
- GPQA-Diamond: 36.6%（基线 30.3%）
- SuperGPQA: 28.5%（基线 24.1%）
- BBEH: 11.5%（基线 9.6%）

### 推理模式演变

训练过程中 Reasoner 的推理能力显著进化：

**早期训练**：直觉猜测
- "月球在 374,000 km，星星更远，可能 1000 倍？所以 374,000,000 km"

**后期训练**：结构化多步推理
1. 识别给定信息
2. 理解完美日食条件（角大小相等）
3. 建立方程：$\frac{3,475}{374,000} = \frac{1,391,000}{d}$
4. 求解：$d = 149,708,489$ km
5. 匹配最接近选项
6. 验证计算

## 与其他方法的对比

| 维度 | R-Zero | Absolute Zero | Strong Challenger | SPICE |
|---|---|---|---|---|
| 语料库根据 | 无 | Python 执行器 | 有（固定 Challenger） | 有（可训练 Challenger） |
| Challenger 训练 | 是 | 是 | 否 | 是 |
| 信息对称性 | 对称 | 对称 | 不对称 | 不对称 |
| 训练稳定性 | 5 轮后退化 | 稳定但性能受限 | 稳定 | 稳定 |
| 最终性能 | 37.0% | 38.0% | 42.1% | **44.9%** |

**关键差异**：
1. **R-Zero**：纯零和博弈，无外部根据，快速退化
2. **Absolute Zero**：使用 Python 执行器验证，但 Challenger 只生成代码题，多样性受限
3. **Strong Challenger**：使用预训练的强 Challenger（Llama-3.3-70B），但不可训练，无法适应 Reasoner 能力
4. **SPICE**：语料库根据 + 可训练 Challenger，实现最佳性能

## 消融实验

### 语料库组成

| 语料库类型 | 数学推理 | 通用推理 | 总体 |
|---|---|---|---|
| NaturalReasoning | 44.4 | **37.0** | 41.7 |
| Nemotron-CC-Math | **53.4** | 29.8 | 43.2 |
| 两者结合 | 50.6 | 35.0 | **44.9** |

**洞察**：不同语料库针对不同能力，组合使用达到最佳平衡。

### 任务类型

| 任务类型 | 数学推理 | 通用推理 | 总体 |
|---|---|---|---|
| 仅 MCQ | 46.9 | **35.7** | 42.0 |
| 仅自由形式 | **52.5** | 31.8 | 43.7 |
| 两者混合 | 50.6 | 35.0 | **44.9** |

**洞察**：MCQ 提供可靠验证，自由形式鼓励灵活推理，混合使用最优。

### Challenger 奖励策略

| 奖励类型 | 数学推理 | 通用推理 | 总体 |
|---|---|---|---|
| Absolute Zero（$1-p$） | 43.8 | 28.5 | 38.0 |
| Threshold | 46.9 | 31.4 | 40.8 |
| R-Zero（$1-2|p-0.5|$） | 48.1 | 32.8 | 42.1 |
| **高斯方差** | **50.6** | **35.0** | **44.9** |

**洞察**：高斯方差奖励最有效地将任务难度校准到 Reasoner 能力边界。

## 局限性

1. **语料库质量依赖**：性能上限受语料库质量和覆盖范围约束
2. **计算成本**：需要大规模采样（Challenger 8 次 × Reasoner 8 次）
3. **任务-测试分布不匹配**：Challenger 生成的问题风格可能与标准基准不同
4. **验证机制依赖**：数学问题依赖 GPT-4o 进行等价性检查
5. **模型规模验证有限**：主要在 4B-70B 规模验证，更大模型效果未知

## 关键洞察

1. **语料库作为环境的范式转变**：将文档语料库视为 RL 环境，提供近乎无穷的外部信号，这是对纯自博弈方法的根本性改进

2. **信息不对称打破对称性困境**：Challenger 看到文档但 Reasoner 看不到，创造了真正的信息梯度，防止了纯自博弈的退化

3. **自动课程学习**：通过高斯方差奖励，Challenger 自动学会生成处于 Reasoner 能力边界的问题，形成自适应课程

4. **推理模式的涌现**：从直觉猜测到结构化多步推理的演变，证明了自博弈可以涌现出复杂的认知能力

5. **可持续自我改进**：与 R-Zero（5 轮后退化）不同，SPICE 可以稳定训练 640 轮，展示了语料库根据对长期训练的关键作用

6. **与 LSP 的互补性**：SPICE 使用外部语料库，LSP 完全无数据；SPICE 专注推理任务，LSP 更通用。两者可能结合使用——SPICE 用于有语料库的领域，LSP 用于完全开放域

7. **问题合成的价值**：Challenger 学会从文档中提炼和合成问题，这与"problem synthesis"研究方向高度相关——模型不仅解决问题，还学会发现和提出有价值的问题
