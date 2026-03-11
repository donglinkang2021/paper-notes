---
title: "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
authors: "Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng"
institution: "University of Virginia, Princeton University (PLI)"
venue: "NeurIPS 2025"
arxiv_id: "2506.01347"
tags: ["RLVR", "negative-reinforcement", "reasoning", "Pass@k", "inference-scaling", "gradient-analysis", "REINFORCE", "entropy"]
---

# 负样本强化在 LLM 推理中的惊人有效性

## 核心贡献

本文将 RLVR（基于可验证奖励的强化学习）的学习信号分解为两个独立范式：正样本强化（PSR，强化正确回答）和负样本强化（NSR，惩罚错误回答），并揭示了一个令人惊讶的发现：仅使用负样本训练（不强化任何正确回答）就能在整个 Pass@$k$ 谱上持续提升模型性能，在许多情况下匹配甚至超越 PPO 和 GRPO。

1. **分解 RLVR 目标函数**：将 $\mathcal{L}_{\text{RLVR}} = \mathcal{L}_{\text{PSR}} + \mathcal{L}_{\text{NSR}}$ 拆分，独立研究正负信号的作用
2. **发现 NSR 的惊人有效性**：仅惩罚错误回答就能有效提升推理能力，同时保持生成多样性
3. **Token 级梯度分析**：从理论上解释了 NSR 为何有效——它按模型先验概率重新分配概率质量
4. **提出 Weighted-REINFORCE**：一个简单的 REINFORCE 变体，通过降低正奖励权重（$\lambda=0.1$），在 MATH、AIME 2025、AMC23 上全面超越 PPO 和 GRPO

## 方法详解

### RLVR 分解

RLVR 使用二元奖励（正确 $+1$，错误 $-1$），其目标函数可以自然分解为：

$$\mathcal{L}_{\text{RLVR}}(\theta) = \underbrace{-\mathbb{E}\left[\sum_{\boldsymbol{y}: r=1} \pi_\theta(\boldsymbol{y}|\boldsymbol{x})\right]}_{\text{PSR: 增大正确回答概率}} + \underbrace{-\mathbb{E}\left[\sum_{\boldsymbol{y}: r=-1} -\pi_\theta(\boldsymbol{y}|\boldsymbol{x})\right]}_{\text{NSR: 减小错误回答概率}}$$

PSR 本质上等价于在线 SFT（增大正确回答的似然），NSR 则是似然最小化（降低错误回答的概率）。

### Token 级梯度分析

对 PSR 和 NSR 分别计算 logit 梯度，揭示了关键差异：

**PSR 的梯度**（对正确回答中的 token $y_t$）：
- 被采样 token：梯度 $\propto \pi_v(1-\pi_v)$，增大其 logit
- 未被采样 token：梯度 $\propto -\pi_{y_t} \cdot \pi_v$，压制所有其他 token

效果：PSR 将概率集中到已采样的正确路径上，压制所有替代方案（包括其他正确路径），导致分布坍塌。

**NSR 的梯度**（对错误回答中的 token $y_t$）：
- 被采样 token：梯度 $\propto -\pi_v \cdot (1-\pi_v)$，降低其 logit
- 未被采样 token：梯度 $\propto \pi_{y_t} \cdot \pi_v$，按当前概率比例提升其他 token

效果：NSR 压制错误 token，并按模型先验概率比例将概率质量重新分配给其他候选 token。这意味着：

1. **保护高置信先验**：当模型对某个 token 已有高概率（$\pi_{y_t} \to 1$），即使它出现在错误回答中，$(1-\pi_{y_t})$ 因子使梯度很小，避免破坏预训练知识
2. **先验引导的概率重分配**：概率质量按 $\pi_v$ 比例分配给未采样 token，高概率的正确候选获得更多提升
3. **隐式正则化**：一旦模型不再生成某个错误回答，NSR 自然停止更新，防止过拟合

### Weighted-REINFORCE

基于上述分析，提出对正奖励加权的简单变体：

$$\mathcal{L}_{\text{W-REINFORCE}}(\theta) = \lambda \cdot \mathcal{L}_{\text{PSR}}(\theta) + \mathcal{L}_{\text{NSR}}(\theta)$$

当 $\lambda=1$ 时退化为标准 REINFORCE，实验中使用 $\lambda=0.1$，即大幅降低正样本强化的权重，让 NSR 主导学习过程。

## 实验结果

### 主实验：Qwen2.5-Math-7B 上的 Pass@$k$

在 MATH 上的关键结果：

| 方法 | Pass@1 | Pass@16 | Pass@64 | Pass@256 |
|---|---|---|---|---|
| Base Model | 63.2 | 91.6 | 95.2 | 96.9 |
| PPO | **76.6** | 91.7 | 94.7 | 96.3 |
| GRPO | 76.3 | 90.6 | 93.6 | 95.5 |
| PSR | 74.1 | 86.2 | 89.3 | 91.2 |
| NSR | 75.7 | **92.4** | **95.3** | **96.9** |
| W-REINFORCE | **76.6** | **92.4** | **95.3** | 96.7 |

在 AIME 2025 上，W-REINFORCE 在除 Pass@1 外的所有 $k$ 值上均取得最佳结果（Pass@256 达到 56.7，超过 GRPO 的 50.0 和 PPO 的 43.3）。

### 训练动态

- **熵变化**：NSR 在训练过程中保持接近基础模型的高熵，而 PSR 导致熵急剧下降；PPO 和 GRPO 介于两者之间
- **正确率提升**：NSR 稳步提升批次中的正确样本比例，但不像 PSR 那样激进，避免了过度自信
- **完全解决率**：PSR 快速提高完全解决率（所有 rollout 都正确），但这恰恰反映了分布坍塌

### Qwen3-4B 实验

在 Qwen3-4B（非思考模式）上，PSR 完全无法激活模型的潜在推理能力，甚至在 MATH 和 AMC23 上显著退化。而 NSR 和 GRPO 成功将非思考模式的性能提升到接近思考模式水平（NSR 在 MATH 上 Pass@1 达到 94.0，思考模式为 94.5）。

### Llama-3.1-8B-Instruct 实验

所有 RL 方法在 Llama 上都导致推理缩放性能下降，但 NSR 造成的退化最小。这表明骨干模型本身的先验质量决定了 RL 能否带来收益。

### $\lambda$ 消融实验

| $\lambda$ | MATH Pass@1 | MATH Pass@256 | AIME Pass@256 |
|---|---|---|---|
| 0 (纯 NSR) | 75.7 | **96.9** | 53.3 |
| 0.1 | **76.6** | 96.7 | **56.7** |
| 0.2 | 75.8 | 95.9 | 43.3 |
| 1 (标准 REINFORCE) | 74.8 | 92.0 | 50.0 |

$\lambda=0.1$ 在准确性和多样性之间取得最佳平衡。

## 与其他方法的对比

| 维度 | PSR | NSR | PPO | GRPO | W-REINFORCE |
|---|---|---|---|---|---|
| Pass@1 | 高 | 中高 | 高 | 高 | 高 |
| Pass@256 | 低于基线 | 匹配/超越基线 | 低于基线 | 低于基线 | 接近基线 |
| 熵保持 | 差（急剧下降） | 好（接近基线） | 中等 | 中等 | 较好 |
| 多样性 | 严重损失 | 良好保持 | 中等损失 | 中等损失 | 较好保持 |
| 实现复杂度 | 简单 | 简单 | 复杂（需 critic） | 中等 | 简单 |

与 Yue et al. 的发现一致：RL 训练后的模型在大 $k$ 时往往不如基础模型。本文进一步揭示这主要是 PSR 造成的，而 NSR 可以避免这个问题。

## 局限性

1. **长期训练不稳定**：NSR 在数百步梯度更新后性能会下降（但 W-REINFORCE 不存在此问题）。这并非 NSR 独有——GRPO 等标准 RL 算法也有类似的训练坍塌现象
2. **仅适用于稀疏二元奖励**：本文聚焦于 RLVR 的二元奖励设定（$\pm 1$），对于需要密集奖励信号的任务（如评估中间推理步骤或主观任务），PSR/NSR/W-REINFORCE 的表现尚不清楚
3. **依赖基础模型先验质量**：NSR 的有效性建立在模型已有较强先验知识的前提上（Qwen 系列效果好，Llama 效果差），对于先验较弱的模型，NSR 可能不足以引导学习

## 关键洞察

### 1. 高概率采样 token 本身就大概率正确

PSR 强化的是模型已经倾向于生成的正确回答，这些回答对应的 token 本身就有较高概率。反复强化它们只会让分布更尖锐，压制其他同样正确但概率较低的"尾部"回答。这就是 PSR 损害 Pass@$k$（大 $k$）的根本原因。

### 2. 保护低概率正确尾部样本是防止熵坍塌的关键

NSR 的概率重分配机制按 $\pi_v$ 比例提升未采样 token，这意味着所有正确候选都能获得概率提升，而不仅仅是当前最高概率的那个。这自然保护了低概率但正确的"尾部"回答，防止了熵坍塌。

### 3. NSR 是一种"精炼"而非"教学"

NSR 不引入新行为，而是通过排除错误选项来精炼模型已有的知识。这类似于雕塑——通过去除不需要的部分来揭示已经存在的形状。当基础模型的先验足够强时，这种精炼比直接教学更有效。

### 4. 熵是推理缩放性能的关键指标

本文建立了熵与 Pass@$k$ 性能之间的清晰联系：高熵 → 高多样性 → 强 Pass@$k$（大 $k$）。这为评估 RL 训练效果提供了一个简单而有效的代理指标。

### 5. 简单方法的力量

W-REINFORCE 仅仅是将正奖励权重从 1.0 降到 0.1，就在多个基准上全面超越了复杂得多的 PPO 和 GRPO。这再次证明，理解问题的本质比堆叠复杂技术更重要。
