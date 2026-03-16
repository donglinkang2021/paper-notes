---
title: "Reinforcement Learning via Self-Distillation (SDPO)"
authors: "Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause"
institution: "ETH Zurich, Max Planck Institute, MIT, Stanford"
venue: "arXiv 2026"
arxiv_id: "2601.20802"
tags: ["paper"]
---
# Reinforcement Learning via Self-Distillation (SDPO)

**Paper ID:** arXiv 2601.20802
**Authors:** Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause
**Institutions:** ETH Zurich, Max Planck Institute, MIT, Stanford
**Code:** [github.com/lasgroup/SDPO](https://github.com/lasgroup/SDPO)

---

## 核心思想

本文提出了 **Self-Distillation Policy Optimization (SDPO)**，一种利用环境丰富反馈进行自蒸馏的强化学习算法。核心洞察：

> **同一个模型可以扮演两个角色：作为"学生"进行初始尝试，作为"教师"在事后基于反馈评估行动价值。**

### 从 RLVR 到 RLRF

论文首先定义了新的问题设定：

- **RLVR (Reinforcement Learning with Verifiable Rewards)**：传统设定，环境只返回标量奖励（如代码通过/失败）
- **RLRF (Reinforcement Learning with Rich Feedback)**：环境返回丰富的文本反馈（如运行时错误、失败的测试用例、LLM 评判）

RLVR 的关键限制是**信用分配瓶颈**——稀疏的标量奖励无法告诉模型"哪里出错了"。

---

## 方法详解

### Self-Teacher 概念

- **学生策略** $\pi_\theta(\cdot \mid x)$：只看问题 $x$，生成回答
- **自我教师** $\pi_\theta(\cdot \mid x, f)$：看问题 $x$ 和反馈 $f$，评估学生生成的回答

关键：两者共享相同参数 $\theta$，只是输入条件不同。自我教师由于获得了额外的反馈信息，应该比学生更准确。

### SDPO 损失函数

$$\mathcal{L}_{\mathrm{SDPO}}(\theta) = \sum_{t} \mathrm{KL}\left(\pi_\theta(\cdot \mid x, y_{<t}) \,\|\, \mathrm{stopgrad}\left(\pi_\theta(\cdot \mid x, f, y_{<t})\right)\right)$$

其中 $\mathrm{stopgrad}$ 阻止梯度通过教师，防止教师退化向学生靠拢。

### SDPO 梯度

$$\nabla_\theta \mathcal{L}_{\mathrm{SDPO}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^{|y|} \sum_{\hat{y}_t \in \mathcal{V}} \nabla_\theta \log \pi_\theta(\hat{y}_t \mid x, y_{<t}) \cdot \log \frac{\pi_\theta(\hat{y}_t \mid x, y_{<t})}{\pi_\theta(\hat{y}_t \mid x, f, y_{<t})} \right]$$

### SDPO vs GRPO 优势函数对比

**GRPO 优势：**

$$A_{i,t}^{\mathrm{GRPO}}(\hat{y}_{i,t}) = \mathbb{1}[y_{i,t} = \hat{y}_{i,t}] \cdot \left( r_i - \mathrm{mean}(\{r_j\}_{j=1}^G) \right)$$

- 所有 token 获得相同的优势（序列级别）

**SDPO 优势：**

$$A_{i,t}^{\mathrm{SDPO}}(\hat{y}_{i,t}) = \log \frac{\pi_\theta(\hat{y}_{i,t} \mid x, f_i, y_{i,<t})}{\pi_\theta(\hat{y}_{i,t} \mid x, y_{i,<t})}$$

- 每个 token 获得独立的优势（logit 级别）

关键区别：SDPO 实现密集信用分配，可以精确识别"哪些 token 出错了"。

### 算法流程

**输入:** 语言模型 $\pi_\theta$，问题数据集，每问题采样数 $G$，环境

**重复:**
1. 采样问题 $x$
2. 采样响应：$\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot \mid x)$
3. 获取环境反馈 $f_i$（运行时错误、失败用例等）
4. **[自蒸馏]** 计算自我教师的 log-probs：$\log \pi_\theta(y_{i,t} \mid x, f_i, y_{i,<t})$
5. **[自蒸馏]** 用梯度下降更新 $\theta$，最小化 $\mathcal{L}_{\mathrm{SDPO}}(\theta)$

**直到收敛**

### 反馈类型

1. **环境输出**：运行时错误、失败的测试用例等
2. **样本解答**：同一 rollout 组中成功的尝试（如果有）
3. **原始尝试**：学生的原始回答（实验发现不包含更好）

---

## 与其他方法对比

| 方法 | 采样 | 信号 | 反馈来源 |
|------|------|------|----------|
| SFT/蒸馏 | 离策略 | 丰富 | 强教师 |
| 在策略蒸馏 | 在策略 | 丰富 | 强教师 |
| RLVR (GRPO) | 在策略 | 稀疏 | 环境 |
| **SDPO (本文)** | 在策略 | 丰富 | 环境 |

---

## 实验结果

### 1. 无丰富反馈环境（标准 RLVR）

在科学推理（化学、物理、生物、材料）和工具使用任务上：

| 任务 | Qwen3-8B + GRPO | Qwen3-8B + SDPO |
|------|-----------------|-----------------|
| Chemistry | 60.0% | **70.1%** |
| Physics | 72.7% | **75.6%** |
| Biology | 51.8% | **52.9%** |
| Materials | 77.1% | **78.4%** |
| Tool use | 67.7% | **68.5%** |

**关键发现：** 即使环境只返回标量奖励，SDPO 也能利用同一 batch 中成功的尝试作为"反馈"来指导失败的尝试。

### 2. 丰富反馈环境（LiveCodeBench v6）

- **SDPO：48.8%** vs GRPO：41.2%
- SDPO 用 **4× 更少的生成** 达到 GRPO 的最终准确率
- 超越 Claude Sonnet 4 (40.5%) 和 Claude Opus 4 (39.7%)

### 3. 模型规模效应

| 模型 | GRPO | SDPO |
|------|------|------|
| Qwen3-0.6B | 19% | 21% |
| Qwen3-1.7B | 26% | 32% |
| Qwen3-4B | 34% | 40% |
| Qwen3-8B | 41% | **49%** |

**结论：** SDPO 的改进随模型规模增长，因为更大的模型有更强的上下文学习能力，能更好地进行"自我回顾"。

### 4. 推理简洁性

SDPO 生成的响应比 GRPO **短 3-7 倍**，同时准确率更高。

GRPO 响应特征：

- 频繁出现 "Hmm"、"Wait"、"No" 等填充词
- 循环推理（"Wait I'm going in circles"）
- 重复计算

SDPO 响应特征：

- 简洁直接
- 避免循环推理
- 高效推理

---

## Test-Time Self-Distillation

论文还提出在测试时对单个困难问题进行 SDPO 训练，加速发现解决方案。

### Discovery@k 指标

$$\mathrm{discovery@}k := \mathbb{P}(\text{discovery time} \leq k)$$

### 结果（LiveCodeBench 极难题）

在 $\mathrm{pass@}64 < 0.03$ 的极难问题上：

| 方法 | discovery@2750 |
|------|----------------|
| Best-of-k | 41.5% |
| Multi-turn | 35.6% |
| **SDPO** | **53.2%** |

- 达到 22% 发现概率，SDPO 需要的生成次数比其他方法 **少 3×**
- SDPO 能解决 best-of-k 和 multi-turn 都无法解决的问题

### 为什么有效？

- RLVR 在二元奖励任务上，找到第一个解之前梯度为零
- SDPO 可以利用丰富反馈（错误信息）持续学习，即使还没找到正确解

---

## 实现细节

### 稳定性改进

1. **正则化教师**：使用 EMA 或与初始教师插值
2. **Jensen-Shannon 散度**：替代 KL 散度，更稳定

### 计算效率

- 唯一额外开销：计算自我教师的 log-probs（可并行，比生成快得多）
- **Top-K 蒸馏**：只计算 top-$K$ (如 $K=100$) 的 logits，避免显存问题

### 混合方法

可以结合 GRPO 和 SDPO：

$$A_{i,t}^{\mathrm{SDPO+GRPO}}(\hat{y}_{i,t}) = \lambda A_{i,t}^{\mathrm{GRPO}}(\hat{y}_{i,t}) + (1-\lambda) A_{i,t}^{\mathrm{SDPO}}(\hat{y}_{i,t}), \quad \lambda \in [0,1]$$

这在弱模型上更稳定（SDPO 优势不可靠时，GRPO 提供稳定信号）。

---

## 局限性

1. **依赖模型能力**：SDPO 需要足够强的上下文学习能力，在弱模型上可能不如 GRPO
2. **依赖反馈质量**：如果环境反馈无信息或误导性，SDPO 无法有效学习
3. **小额外开销**：对于生成时间很短的小模型，计算 log-probs 的开销相对较大

---

## 未来方向

1. **长程和智能体设定**：RLRF 在长轨迹或暴露中间状态信息时特别有吸引力
2. **大规模训练**：在前沿模型上进行大规模多任务 RL 训练
3. **非可验证奖励**：将 SDPO 扩展到开放式文本生成或连续奖励任务
4. **推理行为研究**：系统研究 SDPO 如何产生不同的推理模式

---

## 核心贡献总结

1. 形式化 **RLRF** 范式，利用丰富文本反馈进行 RL
2. 提出 **SDPO**，通过自蒸馏实现密集信用分配，无需外部教师
3. 证明 SDPO 是策略梯度的扩展，可作为 RLVR 的即插即用替代
4. 在科学推理、工具使用、代码生成任务上超越 GRPO
5. 提出 **Test-Time Self-Distillation**，加速困难问题的解决方案发现

---

## 与 OPSD 论文的对比

本文 (SDPO) 与 arXiv 2601.18734 (OPSD) 都提出了自蒸馏方法，但有关键区别：

| 方面 | OPSD | SDPO |
|------|------|------|
| 教师信息 | 真实答案/推理轨迹 | 环境反馈（错误信息、成功尝试） |
| 应用场景 | 有监督数据集 | 可验证环境（代码、数学） |
| 核心优势 | 利用数据集中的参考答案 | 利用环境的丰富反馈 |
| 测试时应用 | 无 | Test-Time Self-Distillation |

两篇论文的共同洞察：**通过给模型提供额外信息（OPSD: 答案; SDPO: 反馈），同一个模型可以作为自己的教师**。
