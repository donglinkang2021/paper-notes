---
title: "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"
authors: "Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu"
institution: "UCLA"
venue: "ICML 2024"
arxiv_id: "2401.01335"
tags: ["self-play", "fine-tuning", "DPO", "iterative-training", "alignment"]
---

# SPIN: 通过自博弈将弱语言模型转化为强语言模型

## 核心贡献

SPIN 提出了一种基于自博弈（self-play）的微调方法，让 LLM 通过与自身前一版本对抗来持续提升，无需额外的人类标注数据或偏好数据。核心思想借鉴 AlphaGo Zero：当模型生成的响应与人类数据不可区分时，训练自然收敛。

## 方法详解

### 自博弈框架

SPIN 将微调建模为两人博弈：
- **主玩家**（Main Player）：新模型 $p_{\boldsymbol{\theta}_{t+1}}$，目标是区分对手生成的响应和人类响应
- **对手玩家**（Opponent Player）：旧模型 $p_{\boldsymbol{\theta}_t}$，目标是生成与人类数据不可区分的响应

### 训练目标

基于积分概率度量（IPM），主玩家最大化人类数据和对手数据之间的价值差：
$$f_{t+1} = \arg\min_{f \in \mathcal{F}_t} \mathbb{E}\left[\ell\left(f(\mathbf{x}, \mathbf{y}) - f(\mathbf{x}, \mathbf{y}')\right)\right]$$

其中 $\mathbf{y} \sim p_\text{data}(\cdot \mid \mathbf{x})$，$\mathbf{y}' \sim p_{\boldsymbol{\theta}_t}(\cdot \mid \mathbf{x})$，$\ell$ 为单调递减凸函数。

### 函数类选择

通过 KL 正则化的对手更新闭式解，推导出函数类：
$$\mathcal{F}_t = \left\{\lambda \cdot \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_t}(\mathbf{y} \mid \mathbf{x})} \;\middle|\; \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\}$$

### 端到端训练目标

选择 logistic loss $\ell(t) = \log(1 + \exp(-t))$ 后，得到最终目标：
$$L_\text{SPIN}(\boldsymbol{\theta}, \boldsymbol{\theta}_t) = \mathbb{E}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_t}(\mathbf{y} \mid \mathbf{x})} - \lambda \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y}' \mid \mathbf{x})}{p_{\boldsymbol{\theta}_t}(\mathbf{y}' \mid \mathbf{x})}\right)\right]$$

这与 DPO 的形式相似，但有本质区别。

### 算法流程

**输入**：SFT 数据集 $\{(\mathbf{x}_i, \mathbf{y}_i)\}$，初始模型 $p_{\boldsymbol{\theta}_0}$

**For** $t = 0, \ldots, T-1$:
1. 用 $p_{\boldsymbol{\theta}_t}$ 为每个 prompt 生成合成响应 $\mathbf{y}_i' \sim p_{\boldsymbol{\theta}_t}(\cdot \mid \mathbf{x}_i)$
2. 优化 $\boldsymbol{\theta}_{t+1} = \arg\min_{\boldsymbol{\theta}} L_\text{SPIN}(\boldsymbol{\theta}, \boldsymbol{\theta}_t)$

### 与 DPO 的关键区别

| 维度 | SPIN | DPO |
|---|---|---|
| 数据需求 | 仅需 SFT 数据 $(\mathbf{x}, \mathbf{y})$ | 需要偏好数据 $(\mathbf{x}, \mathbf{y}_w, \mathbf{y}_l)$ |
| 训练方式 | 天然迭代（自博弈） | 单轮训练 |
| 参考模型 | 每轮更新（$p_{\boldsymbol{\theta}_t}$ 随迭代变化） | 固定不变 |
| 理论基础 | IPM + 自博弈收敛 | Bradley-Terry 偏好模型 |
| 反馈来源 | 隐式自评估 | 需要外部偏好标注 |

## 理论保证

**定理（收敛条件）**：在 $\ell$ 单调递减且凸的条件下：
- **充分性**：若 $p_{\boldsymbol{\theta}_t}(\cdot \mid \mathbf{x}) = p_\text{data}(\cdot \mid \mathbf{x})$，则 $\boldsymbol{\theta}_t$ 是全局最优
- **必要性**：若 $p_{\boldsymbol{\theta}_t}(\cdot \mid \mathbf{x}) \neq p_\text{data}(\cdot \mid \mathbf{x})$，则存在 $\lambda$ 使得 $\boldsymbol{\theta}_t$ 不是全局最优

**定理（更新方向）**：使用 logistic loss 时，更新满足：
$$p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_t}(\mathbf{y} \mid \mathbf{x}) \left(\frac{p_\text{data}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_t}(\mathbf{y} \mid \mathbf{x})}\right)^{1/\lambda}$$

即模型会增加被低估的响应概率，减少被高估的响应概率，自然收敛到目标分布。

## 实验结果

### 基础设置

- 基础模型：`zephyr-7b-sft-full`（Mistral-7B + Ultrachat200k SFT）
- 训练数据：50k prompts 子集，每轮 2 epochs
- 评估：HuggingFace Open LLM Leaderboard, MT-Bench

### 主要结果

| 方法 | Open LLM 平均分 | MT-Bench |
|---|---|---|
| SFT baseline | 58.14 | 5.94 |
| SPIN iter 0 | 60.80 | - |
| SPIN iter 1 | 62.12 | - |
| **SPIN iter 3** | **63.16** | **6.78** |
| DPO (+ 62k 偏好数据) | ~62.0 | ~6.6 |

**关键发现**：
1. SPIN iter 0 就已经与使用额外 62k GPT-4 偏好数据的 DPO 性能相当
2. SPIN iter 1 开始超越 DPO
3. GSM8k 提升 >10%，TruthfulQA 提升 >5%
4. 迭代训练是必要的——单轮内增加 epoch 数无法达到下一轮迭代的效果

### 消融实验

- **训练规模**：14k → 26k → 50k 数据量增加带来持续提升
- **迭代 vs 多 epoch**：在 iter 0 内训练更多 epoch 会饱和，无法超越 iter 1 的性能
- **稳定性**：延长训练不会导致性能下降

## 局限性

1. **性能天花板**：收敛到 $p_\text{data}$，无法超越人类数据质量
2. **计算成本**：每轮迭代需要为所有 prompt 生成合成数据
3. **固定目标分布**：无法处理动态变化数据目标分布
4. **理论与实践的差距**：理论假设全局最优，实际训练可能陷入局部最优

## 关键洞察

1. **自博弈的隐式偏好学习**：无需显式偏好标注，通过区分自身生成和人类数据隐式学习偏好
2. **迭代训练的必要性**：单轮训练有固有上限，迭代是突破的关键
3. **DPO 的自博弈替代**：SPIN 证明了不需要额外偏好数据也能达到甚至超越 DPO 的效果
4. **收敛保证**：理论上保证只在达到目标分布时停止，提供了明确的优化方向
