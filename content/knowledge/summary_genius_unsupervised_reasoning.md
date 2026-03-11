---
title: "Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning"
authors: "Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu"
institution: "Shanghai AI Lab, Xi'an Jiaotong University, The University of Hong Kong, Peking University, HKUST"
venue: "ACL 2025"
arxiv_id: "2504.08672"
tags: ["unsupervised-learning", "self-training", "reasoning", "reinforcement-learning", "foresight-sampling", "DPO", "zero-supervision"]
---

# Genius: 完全无监督的通用推理自训练框架

## 核心贡献

本文提出 **Genius** 框架，实现了完全不依赖任何外部监督信号（无标注答案、无奖励模型）的大语言模型推理能力提升。这是首个仅用通用查询（general queries）就能自我改进推理能力的系统。

关键创新：
1. **零监督范式**：不需要ground-truth答案、不需要外部奖励模型、不需要标注推理链，仅需通用问题
2. **逐步前瞻重采样（Stepwise Foresight Re-sampling）**：通过模拟未来步骤来评估当前步骤价值，克服自回归生成的短视性
3. **优势校准优化（ACO Loss）**：引入优势值校准自奖励函数，提升无监督训练的鲁棒性
4. **显著性能提升**：仅用25K通用查询，7个推理基准平均提升7.43%，超越所有需要监督的基线方法

## 方法详解

### 问题设定与动机

**现有方法的局限**：
- **SFT**：需要 $(x, a, y)$ 三元组（问题、推理链、答案），依赖昂贵的人工标注
- **结果监督（Outcome Supervision）**：需要明确答案验证，仅适用于数学/编程等特定领域
- **奖励模型（Reward Model）**：训练成本高，存在reward hacking问题

**Genius的目标**：仅用通用查询 $x$（无任何标注），让模型自主生成响应 $a$ 并自我改进

### 核心技术：逐步前瞻重采样

**1. 步骤展开与前瞻（Step Rollouts with Foresight）**

在第 $k$ 步，模型维护 $M$ 条前序路径 $\mathbf{a}_{<k}$（beam search），每条路径生成 $N$ 个候选步骤 $a_k$，共 $M \times N$ 个候选。

对每个候选步骤，执行**前瞻**：模拟未来步骤 $\mathbf{a}'_{>k}$，计算前瞻分数：

$$f_k = \text{avg\_log\_prob}(\mathbf{a}'_{>k} | \mathbf{a}_{<k}, a_k)$$

构造完整响应 $T = (\mathbf{a}_{<k}, a_k, \mathbf{a}'_{>k})$，将 $M \times N$ 个前瞻分数归一化为分布：

$$\mathbf{F}_k(i) = \frac{\exp(f_k^{(i)} / \tau)}{\sum_j \exp(f_k^{(j)} / \tau)}$$

**2. 重采样策略（Re-sampling for Exploration & Exploitation）**

**探索（Exploration）**：从分布 $\mathbf{F}_k$ 中采样 $M$ 个步骤作为下一轮的beam：

$$\{a_k^{(m)}\}_{m=1}^M \sim \text{Categorical}(\mathbf{F}_k)$$

更新步骤价值：$Q_k^{(m)} := f_k^{(m)}$

**利用（Exploitation）**：构造训练数据
- 正样本：选择前瞻分数最高的响应 $T_k^w$（分数 $f_k^w$）
- 负样本：从分布中重采样（排除最高分）：$T_k^l \sim \text{Categorical}(\mathbf{F}_k / f_k^w)$

**3. 优势值计算（Advantage Calculation）**

由于不同beam的步骤价值不可直接比较，引入优势值：

$$A_k^w = f_k^w - Q_{k-1}^w, \quad A_k^l = f_k^l - Q_{k-1}^l$$

最终训练数据为五元组：$(x, T_k^w, A_k^w, T_k^l, A_k^l)$

### 优势校准优化（ACO Loss）

**挑战**：无监督设置下，基于前瞻分数的采样不可避免引入噪声，可能出现负样本实际优势高于正样本的情况。

**解决方案**：用优势值校准DPO的自奖励函数。

标准DPO自奖励：

$$\phi(x, T) = \beta \log \frac{\pi_\theta(T|x)}{\pi_{\text{ref}}(T|x)}$$

**ACO修改**：为负样本添加松弛项 $w(x, A)$：

$$\phi_l(x, T^l) = \beta \cdot w(x, A) \cdot \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}$$

$$w(x, A) = \text{clip}\left(\exp\frac{-(A^l - A^w)}{\alpha}, 1\right)$$

**直觉理解**：
- **正常区域**（$A^l - A^w \leq 0$）：负样本确实更差，$w \approx 1$，正常惩罚
- **校准区域**（$A^l - A^w > 0$）：负样本实际优势更高（估计不一致），$w < 1$，减少惩罚

最终ACO损失：

$$\mathcal{L}_{\text{ACO}} = -\mathbb{E} \log \sigma\left[\beta \log \frac{\pi_\theta(T^w|x)}{\pi_{\text{ref}}(T^w|x)} - \beta \cdot w(x,A) \cdot \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}\right]$$

### 训练流程

**配置**：$M=2$（beam size），$N=4$（候选数），$K=4$（最大步数）
- 每个查询生成 $K \times M = 8$ 个训练对
- Magpie 25K查询 → 100K训练对
- OpenHermes 32K查询 → 128K训练对

## 实验结果

### 主要结果（LLaMA3.1-8B-Instruct）

**7个推理基准平均性能**：

| 方法 | 数据 | 监督 | GSM8K | MATH | ReClor | LogiQA | StrategyQA | GPQA | ARC-c | 平均 |
|------|------|------|-------|------|--------|--------|------------|------|-------|------|
| Base (CoT) | - | - | 70.28 | 30.52 | 49.40 | 33.33 | 58.91 | 26.56 | 78.33 | 49.65 |
| **Genius** | 25K | ✗ | **78.32** | **34.64** | **58.80** | **40.86** | **72.53** | **30.35** | **84.04** | **57.08** |
| SPIN | 25K | ✓ | 74.91 | 31.49 | 57.40 | 40.09 | 71.35 | 29.91 | 83.96 | 55.59 |
| Self-Rewarding | 25K | ✗ | 76.04 | 30.19 | 55.80 | 37.94 | 70.48 | 28.35 | 82.17 | 54.42 |
| CoH | 25K | ✗ | 74.37 | 32.29 | 56.20 | 38.56 | 69.08 | 28.13 | 82.51 | 54.45 |

**关键发现**：
- 相比基座模型提升 **+7.43%**，超越所有基线 >2%
- 在困难任务MATH上超越Self-Rewarding **+4.45%**
- 所有7个基准均达到SOTA，一致性最强

### 泛化能力

**Qwen2.5系列模型**：

| 模型 | 平均提升 |
|------|----------|
| Qwen2.5-3B-Instruct | +4.2% |
| Qwen2.5-7B-Instruct | +3.8% |

**竞赛级任务AIME 2024**：
- LLaMA3.1-8B：+6.67%
- Qwen2.5-7B：+6.67%

### 通用能力保持

在6个通用基准上保持稳定，部分提升：
- **Arena-Hard**：30.31 → 50.00（+19.69，显著提升人类偏好对齐）
- **MMLU**：71.14 → 72.21（+1.07）
- **AlpacaEval**：24.60 → 26.96（+2.36）

无灾难性遗忘现象。

## 与其他方法的对比

| 方法 | 需要答案 | 需要奖励模型 | 需要推理链 | 适用范围 |
|------|----------|--------------|------------|----------|
| SFT | ✓ | ✗ | ✓ | 有标注数据的领域 |
| Outcome Supervision | ✓ | ✗ | ✗ | 数学/编程等可验证领域 |
| RLHF/RLAIF | ✗ | ✓ | ✗ | 需训练奖励模型 |
| STaR | ✓ | ✗ | 部分自生成 | 需答案验证 |
| Self-Rewarding | ✗ | ✗ | ✗ | 响应级自奖励 |
| **Genius** | ✗ | ✗ | ✗ | **通用查询，零监督** |

**与STaR的关键区别**：
- STaR：需要答案验证，仅在正确响应上SFT
- Genius：无需答案，用前瞻分数构造偏好对，用RL优化

**与Self-Rewarding的区别**：
- Self-Rewarding：响应级自评分，缺乏细粒度监督
- Genius：步骤级前瞻评估，全局意识更强

**与MCTS的区别**：
- MCTS：需要复杂回溯，计算成本高
- Genius：简单前瞻rollout，效率与质量平衡

## 消融实验

### 采样策略消融

| 变体 | Magpie平均 | OpenHermes平均 | 下降 |
|------|------------|----------------|------|
| Genius | 57.08 | 56.90 | - |
| w/o foresight | 53.91 | 53.65 | -3.2% |
| w/o sampling | 52.98 | 53.80 | -3.3% |

**结论**：
- 前瞻机制缓解短视性，提升3.2%
- 重采样策略平衡探索与利用，提升3.3%

### 优化方法消融

| 优化方法 | Magpie平均 | OpenHermes平均 |
|----------|------------|----------------|
| **ACO** | **57.08** | **56.90** |
| DPO | 55.51 | 55.73 |
| ROPO | 55.30 | 55.25 |
| IPO | 52.31 | 52.20 |
| SimPO | 50.42 | 50.87 |
| SFT | 44.63 | 49.70 |

**结论**：ACO在无监督自训练场景下最优，比DPO高1.5%，比ROPO高1.8%

### 扩展定律（Scaling Law）

训练步数从0到10K，Genius持续平滑提升，未见饱和迹象，而其他基线方法增长停滞。

**潜力**：考虑到通用查询的海量可用性，Genius可能通过持续扩展大幅提升推理能力。

## 局限性

1. **计算成本**：前瞻rollout增加推理开销（$M \times N$ 倍候选生成）
2. **前瞻估计粗糙**：仅用平均对数概率评估未来，可能不够精确
3. **确定性假设**：假设推理路径相对确定，对开放式创造性任务可能不适用
4. **模型规模限制**：实验仅在3B-8B模型上验证，更大模型效果未知
5. **领域泛化**：虽在多领域推理任务上有效，但未测试代码生成、多模态推理等

## 关键洞察

1. **通用查询的力量**：无需特定领域数据，通用问题足以驱动推理能力提升，革新推理扩展定律
2. **步骤级自监督的可行性**：通过前瞻模拟，模型可自主评估步骤质量，无需外部真值
3. **探索-利用的统一**：同一前瞻分布既用于选择下一步（探索），又用于构造偏好对（利用）
4. **鲁棒优化的必要性**：无监督设置下噪声不可避免，优势校准显著提升训练稳定性
5. **RL优于SFT**：在通用数据上，强化学习比监督微调更能激发已预训练模型的推理潜力
6. **一致性胜过峰值**：Genius在所有基准上均达到最优，而其他方法存在性能波动

**未来方向**：
- 扩展到更大规模模型（70B+）和更多训练数据（100K+）
- 探索更精细的前瞻估计（如蒙特卡洛树搜索的轻量化版本）
- 结合过程奖励模型（PRM）进一步提升步骤评估质量
- 应用于多模态推理、代码生成等其他领域
- 研究自适应beam size和rollout深度
