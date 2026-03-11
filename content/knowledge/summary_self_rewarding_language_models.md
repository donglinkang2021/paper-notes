---
title: "Self-Rewarding Language Models"
authors: "Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu"
institution: "Meta AI, NYU"
venue: "ICML 2024"
arxiv_id: "2401.10020"
tags: ["self-rewarding", "LLM-as-a-Judge", "iterative-training", "DPO", "alignment", "reward-modeling"]
---

# 自奖励语言模型

## 核心贡献

Self-Rewarding Language Models 提出让语言模型同时充当生成器和奖励模型，通过迭代训练同时提升指令遵循能力和奖励建模能力。核心突破在于打破了传统 RLHF 中冻结奖励模型的瓶颈，使模型能够在训练过程中持续改进自身的评判能力，从而为自己提供越来越高质量的训练信号。

关键创新点：
- **双重能力统一**：模型同时具备指令遵循和自我评判能力，通过 LLM-as-a-Judge 提供自身奖励
- **动态奖励模型**：奖励模型不再冻结，而是随训练迭代持续改进
- **自我改进循环**：每次迭代中，模型用改进后的评判能力为下一轮训练生成更高质量的偏好数据

## 方法详解

### 训练框架

Self-Rewarding 训练分为两个交替阶段：

**阶段 1：自我指令创建（Self-Instruction Creation）**
- 模型使用 LLM-as-a-Judge prompting 为给定 prompt 生成多个候选响应
- 模型自己评估这些候选响应的质量，给出 1-5 分的评分
- 根据评分构建偏好对 $(x, y_w, y_l)$，其中 $y_w$ 得分高于 $y_l$

**阶段 2：指令遵循训练（Instruction Following Training）**
- 使用 DPO（Direct Preference Optimization）在构建的偏好数据上训练模型
- DPO 目标函数：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### 迭代训练流程

**初始化（Iteration 0）**：
1. 使用 Llama 2 70B 作为基础模型
2. 在 3,200 个 Open Assistant 高质量样本上进行 SFT
3. 在 1,630 个 EFT（Evaluation Fine-Tuning）样本上训练 LLM-as-a-Judge 能力

**迭代训练（Iteration 1, 2, 3...）**：
1. 使用当前模型 $M_t$ 为训练 prompts 生成多个候选响应
2. $M_t$ 自己评分并构建偏好数据集
3. 将新偏好数据与 IFT/EFT 种子数据混合
4. 使用 DPO 训练得到 $M_{t+1}$

### LLM-as-a-Judge Prompting

使用 5 分制评分标准评估响应质量：
- **5 分**：完美响应，全面、准确、有帮助
- **4 分**：高质量但有小瑕疵
- **3 分**：可接受但有明显不足
- **2 分**：质量较差
- **1 分**：完全不合格

模型通过 few-shot prompting 学习评分标准，并为每个响应生成评分和理由。

## 实验结果

### 基础设置
- 基础模型：Llama 2 70B
- 种子数据：3,200 IFT 样本 + 1,630 EFT 样本（来自 Open Assistant）
- 评估基准：AlpacaEval 2.0（与 GPT-4 Turbo 对比的胜率）

### 主要结果

**AlpacaEval 2.0 胜率**：

| 模型 | 胜率 |
|---|---|
| SFT Baseline | - |
| Self-Rewarding Iteration 1 ($M_1$) | 9.94% |
| Self-Rewarding Iteration 2 ($M_2$) | 15.38% |
| **Self-Rewarding Iteration 3 ($M_3$)** | **20.44%** |
| Claude 2 | 17.19% |
| Gemini Pro | 16.85% |
| GPT-4 0613 | 15.76% |

**关键发现**：
1. **指令遵循能力持续提升**：$M_3$ 超越 Claude 2、Gemini Pro 和 GPT-4 0613
2. **奖励建模能力同步改进**：在 Reward Bench 上，$M_3$ 的评判准确率比 $M_1$ 提升显著
3. **迭代训练必要性**：每次迭代都带来实质性提升，$M_3$ 大幅领先 $M_2$ 和 $M_1$

### Head-to-Head 对比

使用 GPT-4 评估模型间的胜率：
- $M_3$ vs SFT Baseline：$M_3$ 胜率约 75%
- $M_3$ vs $M_2$：$M_3$ 胜率约 60%
- $M_2$ vs $M_1$：$M_2$ 胜率约 55%

### 奖励建模能力提升

在 EFT 评估集上的准确率：
- $M_1$：基准水平
- $M_2$：相对 $M_1$ 提升
- $M_3$：进一步提升，证明评判能力随迭代改进

## 与其他方法的对比

### vs 传统 RLHF
| 维度 | Self-Rewarding | 传统 RLHF |
|---|---|---|
| 奖励模型 | 动态更新，持续改进 | 冻结不变 |
| 数据需求 | 模型自我生成偏好数据 | 需要大量人类偏好标注 |
| 能力上限 | 可超越初始人类数据质量 | 受限于冻结奖励模型 |
| 训练方式 | 迭代自我改进 | 两阶段训练 |

### vs DPO
- Self-Rewarding 使用 DPO 作为优化算法，但偏好数据来自模型自我评判
- 传统 DPO 依赖固定的人类偏好数据集
- Self-Rewarding 通过迭代训练突破单轮 DPO 的性能上限

### vs SPIN
- SPIN 通过区分自身生成和人类数据进行自博弈
- Self-Rewarding 通过显式的自我评判和奖励建模
- Self-Rewarding 的奖励模型能力可迁移到其他任务

## 局限性

1. **计算成本高**：每次迭代需要生成多个候选响应并评分，计算开销大
2. **评判能力天花板**：模型的自我评判能力可能存在固有偏差和局限
3. **过度优化风险**：模型可能学会"欺骗"自己的评判系统，产生看似高质量但实际有问题的响应
4. **初始种子数据依赖**：仍需要高质量的人类标注数据进行初始化
5. **长期收敛性未知**：论文只展示了 3 次迭代，更多迭代是否持续改进尚不清楚
6. **评估偏差**：使用 GPT-4 作为评估器可能引入特定偏好

## 关键洞察

1. **统一架构的力量**：将生成和评判能力整合到同一模型中，实现了任务间的正向迁移，打破了传统分离架构的瓶颈

2. **自我改进的可能性**：模型可以通过迭代训练超越初始训练数据的质量上限，这为 AGI 发展提供了新思路

3. **LLM-as-a-Judge 的有效性**：大语言模型具备足够的评判能力，可以替代人类标注者提供训练信号

4. **迭代训练的复合收益**：每次迭代同时改进生成和评判两个维度，形成正反馈循环

5. **超越人类标注的潜力**：通过自我改进，模型有可能发现人类标注者未能捕捉的质量维度

6. **可扩展性优势**：相比依赖人类标注的方法，Self-Rewarding 可以更容易地扩展到更多数据和更多迭代
