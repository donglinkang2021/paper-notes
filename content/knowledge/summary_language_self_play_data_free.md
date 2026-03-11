---
title: "Language Self-Play For Data-Free Training"
authors: "Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan, Jason Chen"
institution: "Meta Superintelligence Labs, UC Berkeley"
venue: "arXiv 2025"
arxiv_id: "2509.07414"
tags: ["self-play", "data-free", "reinforcement-learning", "challenger-solver", "GRPO"]
---

# LSP: 无数据的语言自博弈训练

## 核心贡献

Language Self-Play (LSP) 提出了一种完全不依赖训练数据的 RL 方法，通过博弈论框架让模型同时扮演"出题者"（Challenger）和"解题者"（Solver），在自博弈中持续提升。数是对"数据瓶颈"问题的根本性解决方案。

## 方法详解

### 博弈框架

定义一个极小极大博弈：
$$\min_{\pi_\text{Ch}} \max_{\pi_\text{Sol}} \mathbb{E}_{\mathbf{q} \sim \pi_\text{Ch}, \mathbf{a} \sim \pi_\text{Sol}}[R(\mathbf{q}, \mathbf{a})]$$

- **Challenger** $\pi_\text{Ch}$：生成越来越难的指令/问题
- **Solver** $\pi_\text{Sol}$：学习更好地回答这些问题

关键洞察：两个玩家共享 token 空间，因此可以用同一个模型 $\pi^\theta$ 实现自博弈：
- $\pi_\text{Ch}^\theta(\mathbf{q}) = \pi^\theta(\mathbf{q} \mid \text{<cp>})$（通过 Challenger Prompt 触发）
- $\pi_\text{Sol}^\theta(\mathbf{a} \mid \mathbf{q}) = \pi^\theta(\mathbf{a} \mid \mathbf{q})$

### 优势函数设计

基于 GRPO 的 group-relative 技巧：

1. Challenger 生成 $N$ 个查询 $\mathbf{q}_1, \ldots, \mathbf{q}_N$
2. Solver 对每个查询生成 $G$ 个回答，获得奖励 $R(\mathbf{q}_i, \mathbf{a}_i^j)$
3. 计算查询价值：$V(\mathbf{q}_i) = \frac{1}{G}\sum_{j=1}^G R(\mathbf{q}_i, \mathbf{a}_i^j)$

**Solver 优势**（鼓励好回答）：
$$A_\text{Sol}(\mathbf{q}_i, \mathbf{a}_i^j) = R(\mathbf{q}_i, \mathbf{a}_i^j) - V(\mathbf{q}_i)$$

**Challenger 优势**（鼓励难问题）：
$$A_\text{Ch}(\mathbf{q}_i) = V - V(\mathbf{q}_i)$$

其中 $V = \frac{1}{N}\sum_{i=1}^N V(\mathbf{q}_i)$ 是全局基线。

### 损失函数

$$\mathcal{L}_\text{Self-Play} = \mathcal{L}_\text{Sol} + \alpha_\text{Ch} \cdot \mathcal{L}_\text{Ch}$$

两个损失都包含 KL 散度正则化项，防止偏离参考模型。

### LSP-Zero vs LSP

- **LSP-Zero**：纯零和博弈，容易退化（Solver 用 Python 代码回答所有问题来 hack 奖励）
- **LSP**：加入自奖励（self-reward）$R_Q$ 作为质量正则化，让模型自己评估交互质量。加入后训练可以无限持续而不退化

### 完整算法

**输入**：预训练模型 $\pi^\theta$，奖励函数 $R$

**For** $t = 1$ to $T$:
1. Challenger 生成 $N$ 个查询
2. Solver 对每个查询生成 $G$ 个回答
3. 计算奖励 $R$、自奖励 $R_Q$、优势函数 $A_\text{Sol}$、$A_\text{Ch}$
4. 计算总损失并更新参数

## 实验结果

### 基础模型

Llama-3.2-3B-Instruct

### AlpacaEval 基准

| 方法 | 总体胜率 | 数据需求 |
|---|---|---|
| Base model | 26.5% | - |
| LSP-Zero（无数据） | 32.0% | 无 |
| **LSP（无数据）** | **36.4%** | **无** |
| GRPO（有数据） | 38.8% | Alpaca 数据 |
| LSP + RL | 39.5% | Alpaca 数据 |

**关键发现**：
1. LSP 在完全不使用训练数据的情况下，接近有数据 GRPO 的性能
2. LSP 作为 RL 的预训练阶段，可以进一步提升最终性能
3. 在对话类任务（如 Koala）上，LSP 显著优于 GRPO

### 标准基准

在 MATH、GSM8K、HumanEval、Alpaca 上，LSP 恢复了 GRPO 大部分性能增益。

### Challenger 生成的查询演变

随着训练进行，Challenger 生成的查询越来越复杂：
- 500 轮：简单创意任务（"在荒岛上画藏宝图"）
- 1000 轮：约束性任务（"5岁小孩用10块积木搭桥"）
- 1500 轮：技术性任务（"写2048行汇编代码"）

## 与其他方法的对比

| 维度 | STaR | SPIN | ReST | LSP |
|---|---|---|---|---|
| 数据需求 | 需要问题+答案 | 需要 SFT 数据 | 需要数据+奖励模型 | **完全无数据** |
| 问题来源 | 固定数据集 | 固定数据集 | 固定数据集 | **模型自生成** |
| 训练信号 | 正确/错误过滤 | 分布匹配 | 奖励过滤 | 博弈论奖励 |
| 可持续性 | 会饱和 | 收敛到数据分布 | 会过拟合奖励模型 | **可无限训练** |

## 局限性

1. **奖励模型依赖**：性能上限受奖励模型质量约束
2. **查询-测试分布不匹配**：Challenger 生成的查询风格可能与实际测试不同
3. **LSP-Zero 不稳定**：纯零和博弈容易退化，需要自奖励正则化
4. **仅在 3B 模型上验证**：更大模型的效果未知

## 关键洞察

1. **数据生成即 RL 动作**：将"提供训练数据"建模为 agent 的动作，从根本上消除数据依赖
2. **自博弈的自然课程学习**：Challenger 自动生成由易到难的训练课程
3. **自奖励作为正则化**：防止零和博弈退化为对抗性无意义序列
4. **与用户研究兴趣的联系**：LSP 正是"让模型独立发现问题"的一种实现——Challenger 学会提出有价值的问题，这与用户在 Notion 中提到的"problem synthesis"方向高度相关
