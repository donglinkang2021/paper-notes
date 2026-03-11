---
title: "Reinforced Self-Training (ReST) for Language Modeling"
authors: "Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, Nando de Freitas"
institution: "Google DeepMind"
venue: "arXiv 2023"
arxiv_id: "2308.08998"
tags: ["RLHF", "self-training", "offline-RL", "machine-translation", "reward-filtering"]
---

# ReST: 基于奖励过滤的强化自训练

## 核心贡献

ReST 提出了一种简洁的 growing-batch RL 算法，将 RLHF 的数据生成和策略改进解耦为两个离线阶段：**Grow**（生成数据）和 **Improve**（过滤+微调）。相比在线 RL（如 PPO），ReST 更高效且更稳定。

## 方法详解

### 基本框架

给定条件语言模型 $\pi_\theta(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^T \pi_\theta(y_t \mid \mathbf{y}_{1:t-1}, \mathbf{x})$ 和数据集 $\mathcal{D}$，ReST 交替执行两个步骤：

### Grow 步骤（数据生成）

从当前策略采样生成新数据，并与原始数据合并：
$$\mathcal{D}_g = \{(\mathbf{x}^i, \mathbf{y}^i) \mid \mathbf{x}^i \sim \mathcal{D}, \; \mathbf{y}^i \sim \pi_\theta(\mathbf{y} \mid \mathbf{x}^i)\} \cup \mathcal{D}$$

用奖励模型 $R(\mathbf{x}, \mathbf{y})$ 对所有样本打分。

### Improve 步骤（策略改进）

定义过滤函数：
$$F(\mathbf{x}, \mathbf{y}; \tau) = \mathbb{1}_{R(\mathbf{x}, \mathbf{y}) > \tau}$$

在过滤后的数据上优化策略：
$$J(\theta) = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}_g} \left[F(\mathbf{x}, \mathbf{y}; \tau) \cdot \mathcal{L}(\mathbf{x}, \mathbf{y}; \theta)\right]$$

**关键设计**：多轮 Improve 步骤中，逐步提高过滤阈值 $\tau_1 < \tau_2 < \cdots < \tau_N$，得到质量递增但规模递减的数据子集。每轮从上一轮策略出发，用更低的学习率微调。

### 梯度的概率解释

当 $\mathcal{L} = \mathcal{L}_\text{NLL}$ 时，梯度为：
$$\nabla J(\theta) = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[\lambda \mathbb{E}_{\mathbf{y} \sim \pi_{\theta'}(\mathbf{y} \mid \mathbf{x})} \left[F(\mathbf{x}, \mathbf{y}; \tau) \nabla \log \pi_\theta(\mathbf{y} \mid \mathbf{x})\right] + (1-\lambda) \mathbb{E}_{\mathbf{y} \sim p(\mathbf{y} \mid \mathbf{x})} \left[F(\mathbf{x}, \mathbf{y}; \tau) \nabla \log \pi_\theta(\mathbf{y} \mid \mathbf{x})\right]\right]$$

- 第一项类似 on-policy 策略梯度（$\theta \approx \theta'$ 时）
- 第二项是 offline 策略梯度，防止策略偏离原始数据分布（避免 model collapse）

### 完整算法

**输入**：数据集 $\mathcal{D}$，奖励模型 $R$，Grow 步数 $G$，Improve 步数 $I$

1. 在 $\mathcal{D}$ 上用 NLL 训练初始策略 $\pi_\theta$
2. **For** $g = 1$ to $G$:
   - **Grow**：从 $\pi_\theta$ 采样生成 $\mathcal{D}_g$，用 $R$ 打分
   - **For** $i = 1$ to $I$:
     - **Improve**：选择阈值 $\tau_i$（递增），在过滤数据上优化 $\theta$

## 实验结果

### 机器翻译基准

在 IWSLT 2014 (De-En)、WMT 2020 (Zh-En)、Web Domain (En-Zh) 上测试：

**核心发现**：

1. **多轮 Improve 有效**：每一轮 Improve 步骤都显著提升翻译质量
2. **额外 Grow 步骤有效**：第二轮 Grow 在 IWSLT 上额外提升 5.3 分
3. **BC loss 最优**：简单的 NLL loss 优于 OAC、B-VMPO、GOLD 等 offline RL loss
4. **优于在线 RL**：

| 算法 | 平均奖励 | 样本数 |
|---|---|---|
| BC (G=0, I=0) | 70.9 | 16M |
| ReST (G=1, I=0) | 71.9 | 16M |
| ReST (G=1, I=4) | 77.8 | 16M |
| **ReST (G=2, I=3)** | **83.1** | 32M |
| Online RL (PPO) | 71.6 | 24M |

5. **Best-of-N 兼容**：ReST + Best-of-N ($N<10$) 匹配 BC + Best-of-N ($N=200$) 的性能
6. **人类评估**：所有 ReST 变体在人类评估中显著优于 BC baseline

### 奖励模型过拟合问题

- 随着 Grow 步数增加，奖励分数持续上升，但人类评估分数不一定同步
- 原因：策略偏离训练分布后，奖励模型泛化能力下降
- 建议：在连续 Grow 步骤之间对奖励模型进行微调

## 局限性

1. **依赖奖励模型质量**：需要鲁棒的奖励模型，否则会过拟合
2. **Off-policy 特性**：Grow 步骤生成的数据是 off-policy 的，可能限制探索
3. **确定性环境假设**：在随机环境中，阈值过滤可能导致学习次优行为
4. **BC loss 的局限**：虽然 BC 在实验中最优，但可能因为 RL 中的稀疏奖励和信用分配问题

## 与 STaR 的对比

| 维度 | STaR | ReST |
|---|---|---|
| 过滤方式 | 正确/错误二值过滤 | 奖励模型 + 阈值过滤 |
| 数据来源 | 模型自生成 + rationalization | 模型自生成（off-policy） |
| 训练策略 | 每轮从预训练模型重启 | 从上一轮策略继续微调 |
| 应用场景 | 推理任务（有标准答案） | 开放生成任务（需奖励模型） |
| 阈值策略 | 无（二值） | 递增阈值序列 |

## 关键洞察

1. **Grow-Improve 解耦**：将数据生成和策略改进分离，允许数据复用，比在线 RL 更高效
2. **递增阈值**：逐步提高质量要求，在数据质量和数量之间取得平衡
3. **简单即有效**：BC loss + 过滤 > 复杂的 offline RL loss
4. **Alignment tax 更低**：ReST 提升奖励的同时不损害 BLEU 等其他指标，而 PPO 会导致 BLEU 下降 8 分
