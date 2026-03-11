---
title: "Multiagent Finetuning of Language Models"
authors: "Anonymous (ICLR 2025 submission)"
institution: "Under review"
venue: "ICLR 2025 (under review)"
arxiv_id: "2501.05707"
tags: ["multiagent", "debate", "self-improvement", "specialization", "iterative-training"]
---

# Multiagent Finetuning: 通过角色专业化实现持续自我提升

## 核心贡献

Multiagent Finetuning 提出了一种多智能体微调框架，通过将单一模型分化为多个专业化角色（Generation Agents 和 Critic Agents），解决了单模型自我提升中的多样性衰减和性能饱和问题。关键创新是让每个 agent 在自己生成的数据子集上独立微调，从而实现持续的性能提升。

## 方法详解

### 问题背景

传统单模型自我提升方法（如 STaR）存在的问题：
1. **多样性衰减**：重复微调导致生成响应风格趋同
2. **性能饱和**：通常 2-3 轮后性能停滞
3. **过拟合风险**：模型固定在狭窄的响应模式上

### 多智能体辩论（Multiagent Debate）

基础数据生成方法：
1. $N$ 个 agent 对问题 $x$ 各自生成初始响应
2. 进行 $M$ 轮辩论：每个 agent 看到其他 agent 的响应摘要，生成更新的响应
3. 最后一轮通过多数投票确定最终答案 $\hat{y}$

### 角色专业化微调

**Generation Agents（生成者）**：
- 角色：生成初始响应，依赖多样化的推理链
- 数据构建：对于 agent $A_n^G$，收集其在辩论第一轮中生成的、最终与 $\hat{y}$ 一致的响应 $y_n$
- 数据集：$\mathcal{D}_n^G = \{(x, y_n) \mid y_n \text{ matches } \hat{y}\}$
- 目标：保持多样性，各自专注于不同的推理路径

**Critic Agents（评论者）**：
- 角色：评估其他 agent 的响应，生成改进的答案
- 数据构建：收集 agent 在辩论后续轮次中的响应（包含对其他响应的评论）
- 数据集：$\mathcal{D}_n^C = \{(x, [y_1, \ldots, y_N], y_n') \mid y_n' \text{ matches } \hat{y}\}$
- 目标：学习批判性思维和响应整合能力

### 多轮迭代训练

**算法流程**：

**输入**：基础模型 $\pi_\theta$，任务数据集 $\mathcal{D}_\text{task}$

**For** iteration $t = 1$ to $T$:
1. 用当前 $N$ 个 agent 进行多智能体辩论，生成数据
2. 构建 $N$ 个 Generation 数据集 $\{\mathcal{D}_1^G, \ldots, \mathcal{D}_N^G\}$
3. 构建 $N$ 个 Critic 数据集 $\{\mathcal{D}_1^C, \ldots, \mathcal{D}_N^C\}$
4. 从基础模型 $\pi_\theta$ 出发，独立微调每个 agent：
   - Generation agent $n$: 在 $\mathcal{D}_n^G$ 上微调
   - Critic agent $n$: 在 $\mathcal{D}_n^C$ 上微调

**关键设计**：
- 每轮从原始基础模型重新开始微调（类似 STaR）
- 每个 agent 只在自己生成的数据上训练，促进专业化
- Generation 和 Critic 角色分离，形成互补的能力

### 推理阶段

使用微调后的 $N$ 个 agent 进行多智能体辩论，通过多数投票得到最终答案。

## 实验结果

### 基础设置

- 模型：Phi-3-mini-4k-instruct (3.8B)、Mistral-7B-Instruct-v0.2
- 任务：Arithmetic、GSM8K、MATH（前 3 个难度级别）
- 配置：3 个 agent，2 轮辩论
- 训练数据：每个任务 500 个样本
- 测试数据：500 个 held-out 样本

### 主要结果（MATH 数据集，Phi-3）

| 方法 | 准确率 | 提升 |
|---|---|---|
| Base (单模型) | 42.4% | - |
| Majority (多模型投票) | 48.0% | +5.6% |
| Debate (多智能体辩论) | 53.4% | +11.0% |
| Debate + Single-agent FT | 58.8% | +16.4% |
| **Multiagent FT (1 轮)** | **62.0%** | **+19.6%** |
| **Multiagent FT (5 轮)** | **66.0%** | **+23.6%** |

### 跨任务结果

**Phi-3 模型**：
- Arithmetic: 42.4% → 66.0% (+23.6%)
- GSM8K: 53.4% → 62.0% (+8.6%)
- MATH: 42.4% → 66.0% (+23.6%)

**Mistral 模型**：
- MATH: 22.5% → 28.2% (+5.7%)

### 关键发现

1. **持续提升**：Multiagent FT 可以持续 5 轮迭代而不饱和，而 Single-agent FT 在 1 轮后就开始下降
2. **优于单模型**：5 轮后比最佳 baseline（Debate + Single-agent FT）高 12.6%
3. **泛化能力**：在 MATH 上微调的模型，在 GSM8K 上 zero-shot 准确率从 53.4% 提升到 62.0%
4. **规模效应**：5 个 agent 比 3 个 agent 效果更好（见附录）

## 与其他方法的对比

| 维度 | STaR/SPIN | ReST | LSP | Multiagent FT |
|---|---|---|---|---|
| 数据生成 | 单模型 | 单模型 | Challenger-Solver | **多智能体辩论** |
| 训练策略 | 单模型微调 | 单模型微调 | 单模型自博弈 | **多模型专业化** |
| 多样性维持 | 依赖 rationalization | 依赖高阈值过滤 | 依赖 Challenger | **角色分化** |
| 迭代能力 | 2-3 轮饱和 | 受奖励模型限制 | 可持续但需正则化 | **5+ 轮持续提升** |
| 推理成本 | 低（单模型） | 低（单模型） | 低（单模型） | **高（N 个模型）** |

## 局限性

1. **计算成本高**：
   - 训练：需要 4x H100 或 4x A100，120-240GB GPU 内存
   - 推理：需要运行 $N$ 个模型并进行辩论，耗时 12-24 小时

2. **扩展性问题**：
   - 随着 agent 数量增加，成本线性增长
   - 可能的解决方案：权重共享、蒸馏到单模型、量化

3. **数据效率**：
   - 需要足够的训练数据来支持多个 agent 的专业化
   - 小数据集上可能无法充分发挥优势

4. **理论理解不足**：
   - 为什么角色专业化能防止饱和？
   - 最优 agent 数量和辩论轮次如何确定？

## 关键洞察

1. **专业化 > 泛化**：让多个模型各自专注于不同推理路径，比让单个模型学习所有路径更有效
2. **角色分离的价值**：Generation 和 Critic 的分离形成了类似"提出-批判-改进"的认知循环
3. **辩论作为数据增强**：多智能体辩论不仅提升推理质量，还自然产生了多样化的训练数据
4. **从基础模型重启**：每轮从原始模型重新微调，避免了累积的分布偏移
5. **与用户研究兴趣的联系**：这种多 agent 协作模式与用户在 Notion 中提到的"collective intelligence"方向高度相关——不同专业化的 agent 通过辩论形成集体智能
