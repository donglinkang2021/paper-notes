---
title: "Self-Distillation Enables Continual Learning (SDFT)"
authors: "Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal"
institution: "MIT, Improbable AI Lab, ETH Zurich"
venue: "ICLR"
arxiv_id: "2601.19897"
tags: ["paper"]
---
# Self-Distillation Enables Continual Learning (SDFT)

**Paper ID:** arXiv 2601.19897
**Authors:** Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal
**Institutions:** MIT, Improbable AI Lab, ETH Zurich
**Website:** [idanshenfeld.com/SDFT](http://idanshenfeld.com/SDFT)

---

## 核心思想

本文提出了 **Self-Distillation Fine-Tuning (SDFT)**，一种从专家示范中进行在策略学习的方法，实现**持续学习**而不产生灾难性遗忘。

核心问题：

- **SFT (监督微调)**：从示范中学习的标准方法，但它是离策略的，导致严重的灾难性遗忘
- **在策略 RL**：可以减少遗忘，但需要显式的奖励函数，通常不可用

**SDFT 的解决方案**：利用模型的上下文学习能力，将示范条件化的模型作为自己的教师，生成在策略训练信号。

---

## 方法详解

### 教师-学生框架

给定基础模型 $\pi$，对于每个查询 $x$ 和专家示范 $c$：

- **学生策略** $\pi_\theta(\cdot \mid x)$：只看问题
- **教师策略** $\pi(\cdot \mid x, c)$：看问题 + 示范

关键：两者是**同一个模型**，只是输入条件不同。

### 教师 Prompt 模板

```text
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

这个 prompt 确保模型不会简单复制示范，而是利用上下文学习能力生成自己的理解。

### SDFT 损失函数

最小化学生和教师之间的反向 KL 散度：

$$\mathcal{L}(\theta) = D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid x) \,\|\, \pi(\cdot \mid x, c)\right) = \mathbb{E}_{y \sim \pi_\theta(y \mid x)}\left[\log \frac{\pi_\theta(y \mid x)}{\pi(y \mid x, c)}\right]$$

分解为 token 级别的梯度估计：

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_t \sum_{y_t \in \mathcal{V}} \log \frac{\pi_\theta(y_t \mid y_{<t}, x)}{\pi(y_t \mid y_{<t}, x, c)} \cdot \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}, x) \right]$$

### SDFT = 隐式逆强化学习

论文证明 SDFT 等价于最大化一个隐式奖励函数。

**信任域正则化 RL 目标：**

$$\pi_{k+1} = \max_\pi \mathbb{E}_{y \sim \pi}[r(y, x)] - \beta D_{\mathrm{KL}}\left(\pi(\cdot \mid x) \,\|\, \pi_k(\cdot \mid x)\right)$$

**关键假设（In-Context Assumption）：**

$$\pi_{k+1}^*(y \mid x) \approx \pi(y \mid x, c)$$

即：条件化在示范上的模型近似于最优策略。

**推导出的隐式奖励：**

$$r(y, x, c) = \log \pi(y \mid x, c) - \log \pi_k(y \mid x)$$

**Token 级别：**

$$r_t(y_t \mid y_{<t}, x, c) = \log \frac{\pi(y_t \mid y_{<t}, x, c)}{\pi_k(y_t \mid y_{<t}, x)}$$

---

## ICL 假设的验证

SDFT 依赖于两个条件：

### 1. 最优性

示范条件化的教师应该能达到接近最优的表现：

$$\mathbb{E}_{y \sim \pi(y \mid x, c)}[r(y, x)] \approx \mathbb{E}_{y \sim \pi_{k+1}^*}[r(y, x)]$$

**实验验证：** 在 ToolAlpaca 数据集上：

- 基础模型准确率：42%
- 教师策略（条件化在示范上）：100%

### 2. 最小偏离

教师应该在 KL 意义上接近当前策略（而不是简单复制示范）：

$$D_{\mathrm{KL}}\left(\pi(\cdot \mid x, c) \,\|\, \pi_k(\cdot \mid x)\right) \approx D_{\mathrm{KL}}\left(\pi_{k+1}^*(\cdot \mid x) \,\|\, \pi_k(\cdot \mid x)\right)$$

**实验验证：**

- SFT 模型与基础模型的 KL 散度：1.26 nats
- 教师与基础模型的 KL 散度：0.68 nats（几乎一半）

这验证了教师产生高质量输出的同时保持接近基础策略。

---

## 算法流程

**输入:** 示范数据集 $\mathcal{D} = \{(x_i, c_i)\}_{i=1}^N$，自回归模型 $\pi_\theta$，教师 EMA 率 $\alpha$

**初始化:** 教师权重 $\phi = \theta$

**For each training step:**
1. 采样 minibatch $\mathcal{B} = \{(x_i, c_i)\}$
2. 对每个样本并行执行：
   - 学生 rollout：$y_i \sim \pi_\theta(\cdot \mid x_i)$ [在策略]
   - 计算学生 logprobs：$\ell^S_{i,t} = \log \pi_\theta(y_{i,t} \mid y_{i,<t}, x_i)$
   - 计算教师 logprobs：$\ell^T_{i,t} = \log \pi_\phi(y_{i,t} \mid y_{i,<t}, x_i, c_i)$
3. 使用解析梯度估计器计算梯度
4. 更新学生：$\theta \leftarrow \theta - \eta \cdot g$
5. 更新教师 (EMA)：$\phi \leftarrow \alpha \cdot \theta + (1-\alpha) \cdot \phi$

---

## 实验设置

### 技能学习 (Skill Learning)

- **Science Q&A**：SciKnowEval 化学子集
- **Tool Use**：ToolAlpaca
- **Medical**：HuatuoGPT-o1 临床推理

### 知识获取 (Knowledge Acquisition)

- 2025年自然灾害的 Wikipedia 文章（模型知识截止后）
- 从文章生成问答对

---

## 实验结果

### 1. 单任务学习 - 准确率 vs 遗忘

| 任务 | 方法 | 新任务准确率 | 先前能力平均 |
|------|------|------------|-------------|
| Science Q&A | Base | 32.1% | 65.5% |
| | SFT | 66.2% | 53.4% |
| | SDFT | **70.2%** | **64.5%** |
| Tool Use | Base | 42.9% | 65.5% |
| | SFT | 63.2% | 56.0% |
| | SDFT | **70.6%** | **65.4%** |
| Medical | Base | 30.1% | 65.5% |
| | SFT | 35.5% | 60.2% |
| | SDFT | **40.2%** | **65.4%** |

**关键发现：** SDFT 是唯一能够在提高新任务性能的同时不显著降低先前能力的方法。

### 2. 多任务持续学习

在三个技能上顺序训练同一个模型：

- **SDFT**：能够学习每个新技能同时保持之前学到的技能
- **SFT**：学习新技能时，之前的技能表现急剧下降（振荡行为）

### 3. 知识获取

| 方法 | 严格准确率 | 宽松准确率 | OOD 准确率 |
|------|-----------|-----------|-----------|
| Base | 0% | 0% | 0% |
| Oracle RAG | 91% | 100% | 100% |
| CPT | 9% | 37% | 7% |
| SFT | 80% | 95% | 80% |
| **SDFT** | **89%** | **100%** | **98%** |

**关键发现：** SDFT 在分布外问题上接近完美准确率，而 SFT 表现较差，说明 SFT 只是记忆特定答案而非真正整合知识。

### 4. 模型规模效应

| 模型大小 | SFT | SDFT |
|----------|-----|------|
| 3B | 58% | 54% |
| 7B | 66% | 70% |
| 14B | 64% | 71% |

**结论：** SDFT 的优势随模型规模增长，因为更大的模型有更强的上下文学习能力。

### 5. 训练推理模型（无推理数据）

使用 Olmo-3-7B-Think（一个长思维链模型）在只有最终答案的数据上训练：

| 方法 | 准确率 | 平均 token 数 |
|------|--------|--------------|
| Base | 31.2% | 4612 |
| SFT | 23.5% | 3273 |
| **SDFT** | **43.7%** | 4180 |

SFT 导致推理能力退化（响应变短），而 SDFT 保持推理深度同时提高准确率。

---

## 消融研究

### 梯度估计器选择

比较三种 KL 梯度估计器：

1. **Token-level（部分）**：有偏，高方差
2. **解析 per-token**：有偏但低方差，实践中最稳定
3. **Rao-Blackwellized**：无偏但计算昂贵，无显著收益

最终选择：解析 per-token 估计器 + 每个 prompt 单个轨迹

### 教师选择

| 教师类型 | 效果 |
|----------|------|
| 冻结基础模型 | 稳定但表现差 |
| 学生自身 | 训练不稳定 |
| **EMA** | 稳定且表现最佳 |

### 离策略 vs 在策略

即使使用同样的示范条件化教师：

- 在策略 SDFT > 离策略蒸馏 > SFT

验证了在策略学习对性能的重要性。

---

## 计算成本

相比 SFT：

- FLOPs：约 $2.5\times$
- 墙钟时间：约 $4\times$

但考虑到 Re-invoke 等方法需要多阶段训练，SDFT 可能实际上**减少**总训练时间。

---

## 局限性

1. **学习到的伪影**：学生可能继承教师的语言模式（如"Based on the text..."）
   - 解决方案：训练时 mask 前几个 token 的损失

2. **模型能力要求**：小模型的 ICL 能力不足，无法提供有意义的教师信号

3. **行为改变限制**：SDFT 擅长学习新技能/知识同时保持现有能力，但难以进行根本性的行为改变（如将非推理模型变为推理模型）

---

## 与其他方法的关系

### vs 在策略 RL

- SDFT：适用于只有示范、无奖励函数的场景
- 在策略 RL：需要显式奖励函数

两者可以**组合使用**：SDFT 作为 RL 微调的初始化。

### vs OPSD/SDPO

| 方面 | SDFT | OPSD | SDPO |
|------|------|------|------|
| 教师信息 | 专家示范 | 真实答案 | 环境反馈 |
| 主要目标 | 持续学习 | 提高推理 | 信用分配 |
| 应用场景 | 学习新技能/知识 | 数学推理 | 代码/数学 |
| 遗忘关注 | 核心关注点 | 次要 | 次要 |

三篇论文的共同核心：**利用上下文学习能力，通过不同的条件化方式让模型成为自己的教师**。

---

## 核心贡献总结

1. 提出 **SDFT**：一种从示范进行在策略学习的方法
2. 证明 SDFT 等价于**隐式逆强化学习**
3. 在技能学习和知识获取任务上超越 SFT，同时显著减少灾难性遗忘
4. 展示**真正的持续学习**：单个模型可以顺序学习多个技能而不退化
5. 证明可以**无需推理数据训练推理模型**

---

## 核心洞察

> **在策略学习是持续学习的关键**。离策略方法（如 SFT）在学习新任务时不可避免地会遗忘旧能力，而在策略方法通过在当前策略分布上训练来保持稳定性。

> **上下文学习是自蒸馏的桥梁**。通过将模型条件化在示范上，我们可以获得一个"更明智"的教师版本，而无需额外的模型或奖励函数。
