---
title: "STaR: Self-Taught Reasoner - Bootstrapping Reasoning With Reasoning"
authors: "Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman"
institution: "Stanford University, Google Research"
venue: "NeurIPS 2022"
arxiv_id: "2203.14465"
tags: ["self-play", "reasoning", "chain-of-thought", "bootstrapping", "iterative-training"]
---

# STaR: 自举推理能力的自学推理器

## 核心贡献

STaR 提出了一种迭代自举（bootstrapping）方法，让语言模型从少量带推理链的示例出发，逐步学会为大量问题生成高质量的推理过程（rationale）。核心思想是：**用模型自己生成的正确推理来训练自己**。

关键创新：
1. **Rationale Generation**：用 few-shot prompting 让模型生成推理链，只保留得到正确答案的推理
2. **Rationalization**：对于答错的问题，给模型提供正确答案作为提示，让它"逆向推理"生成合理化解释
3. **迭代训练**：在生成的推理数据上微调，然后用新模型重复上述过程

## 方法详解

### 算法流程

给定预训练模型 $M$ 和数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^D$（问题 + 答案），以及少量带推理的示例 $\mathcal{P} = \{(x_i^p, r_i^p, y_i^p)\}_{i=1}^P$（$P \ll D$，如 $P=10$）。

**外循环迭代**（$n = 1, 2, \ldots, N$）：

1. **Rationale Generation**：用当前模型 $M_{n-1}$ 对所有问题生成推理链和答案
   $$(\hat{r}_i, \hat{y}_i) \leftarrow M_{n-1}(x_i) \quad \forall i \in [1, D]$$

2. **Rationalization**：对于答错的问题，给出正确答案 $y_i$ 作为提示，生成新的推理链
   $$(\hat{r}_i^\text{rat}, \hat{y}_i^\text{rat}) \leftarrow M_{n-1}(\text{add\_hint}(x_i, y_i)) \quad \forall i \in [1, D]$$

3. **过滤**：只保留最终答案正确的推理链
   $$\mathcal{D}_n = \{(x_i, \hat{r}_i, y_i) \mid \hat{y}_i = y_i\}$$
   $$\mathcal{D}_n^\text{rat} = \{(x_i, \hat{r}_i^\text{rat}, y_i) \mid \hat{y}_i \neq y_i \land \hat{y}_i^\text{rat} = y_i\}$$

4. **微调**：从原始预训练模型 $M$ 出发，在合并数据集上训练
   $$M_n \leftarrow \text{train}(M, \mathcal{D}_n \cup \mathcal{D}_n^\text{rat})$$

### 与强化学习的联系

STaR 可以看作策略梯度的近似。将模型视为离散隐变量模型：
$$p_M(y \mid x) = \sum_r p(r \mid x) p(y \mid x, r)$$

给定指示奖励函数 $\mathbb{1}(\hat{y} = y)$，期望奖励为：
$$J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot \mid x_i)} \mathbb{1}(\hat{y}_i = y_i)$$

梯度为：
$$\nabla J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot \mid x_i)} \left[\mathbb{1}(\hat{y}_i = y_i) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i \mid x_i)\right]$$

STaR 通过过滤（只保留 $\hat{y}_i = y_i$ 的样本）来近似这个目标，并用贪婪解码降低方差。

### Rationalization 的作用

**问题**：纯 rationale generation 会在模型无法解决新问题时停滞（因为没有训练信号）。

**解决**：Rationalization 让模型"逆向推理"——给定正确答案，生成合理的推理过程。这相当于从条件分布 $p(r \mid x, y)$ 采样，而非 $p(r \mid x)$，扩展了搜索空间。

**效果**：
- 让模型接触到原本答不出的难题
- 增加数据集规模
- 加速学习过程（尤其在算术任务上）

## 实验结果

### 1. 算术任务（$n$ 位数加法）

- **数据**：50,000 个随机生成的加法问题（1-5 位数）
- **Few-shot baseline**：2 位数加法准确率 < 1%
- **STaR（无 rationalization）**：16 轮迭代后达到 89.5%
- **STaR（有 rationalization）**：
  - 第 1 轮后 2 位数加法从 <1% 提升到 32%
  - 可以同时学习多个位数长度
  - 泛化到 9-10 位数（训练时未见过）

### 2. CommonsenseQA

| 方法 | Dev 准确率 | 训练数据使用比例 |
|---|---|---|
| Few-shot Direct GPT-J | 20.9% | ~0% |
| Few-shot CoT GPT-J | 36.6% | ~0% |
| Few-shot CoT LaMDA 137B | 55.6% | ~0% |
| GPT-J Direct Finetuned | 60.0% | 100% |
| STaR (无 rationalization) | 68.8% | 69.7% |
| **STaR (有 rationalization)** | **72.5%** | **86.7%** |
| GPT-3 Direct Finetuned (30x 更大) | 73.0% | 100% |

**关键发现**：
- STaR 让 6B 参数的 GPT-J 达到与 180B 参数 GPT-3 相当的性能
- 相比直接微调提升 12.5%
- 相比 few-shot CoT 提升 35.9%

**人类评估**：众包工作者认为 STaR 生成的推理比 few-shot 生成的推理质量高 30%（$p=0.039$）。

### 3. GSM8K（小学数学）

| 方法 | Test 准确率 | 训练数据使用比例 |
|---|---|---|
| Few-shot Direct GPT-J | 3.0% | ~0% |
| Few-shot CoT GPT-J | 3.1% | ~0% |
| GPT-J Direct Finetuned | 5.8% | 100% |
| STaR (无 rationalization) | 10.1% | 25.0% |
| **STaR (有 rationalization)** | **10.7%** | **28.7%** |

模型生成的计算步骤数与人类标注的步骤数匹配度为 53-57%，有时模型会找到更简洁的解法。

## 局限性

1. **模型规模要求**：需要足够大的模型才能有 few-shot 推理能力（GPT-2 无法成功）
2. **高随机性任务**：在二分类等高随机性任务上，会产生大量错误推理，难以过滤
3. **Hint 设计**：Rationalization 的提示方式不是自然而然的，需要针对任务设计
4. **温度采样无效**：高温度采样会产生错误推理，反而降低性能
5. **Few-shot prompt 依赖**：初始推理质量受 few-shot 示例质量限制

## 关键洞察

1. **自举的协同效应**：推理能力提升 → 训练数据质量提升 → 推理能力进一步提升
2. **逆向推理的价值**：给定答案后生成推理，比直接生成推理更容易，且能覆盖难题
3. **过滤 > 采样**：低温贪婪解码 + 过滤，优于高温多样性采样
4. **从预训练重启**：每轮迭代从原始预训练模型重新训练，避免过拟合

## 与后续工作的联系

STaR 是 self-play 和 iterative training 的早期代表作，后续工作如 ReST、SPIN、R-Zero 等都在此基础上发展：
- **ReST**：引入 reward model 进行 off-policy 过滤
- **SPIN**：用对抗训练替代简单过滤
- **R-Zero**：完全去除人类标注数据，从零开始自举
