---
title: "Guided Self-Evolving LLMs with Minimal Human Supervision"
authors: "Tencent AI Seattle Lab"
institution: "Tencent AI Seattle Lab"
venue: "arXiv 2025"
arxiv_id: "2512.02472"
tags: ["self-play", "self-evolution", "minimal-supervision", "challenger-solver", "curriculum-learning", "GRPO"]
---

# R-Few: 最小人类监督下的引导式自进化大模型

## 核心贡献

R-Few 提出了一种引导式自博弈框架，通过引入**极少量人类监督**（仅 1-5% 数据）来稳定和加速 LLM 的自进化过程。它在 R-Zero 的基础上解决了两个核心问题：

1. **概念漂移（Concept Drift）**：无引导的自博弈会强化模型自身的偏见，偏离事实正确性
2. **多样性崩溃（Diversity Collapse）**：自生成任务趋向低熵、熟悉的区域，探索停滞

R-Few 通过两个关键创新实现稳定的自进化：
- **Few-Shot Grounded Challenger**：用少量人类"锚点"数据引导合成问题生成
- **Online Curriculum Solver**：基于难度的在线课程学习，动态选择中等难度样本

## 方法详解

### 整体框架

R-Few 延续 Challenger-Solver 自博弈架构，但加入轻量级人类监督：

**Challenger（出题者）**：
- 从人类锚点数据 $\mathcal{D}_H$ 中随机采样 $k \in \{0,1,\ldots,5\}$ 个示例作为 in-context examples
- 当 $k=0$ 时退化为 R-Zero 的无数据模式，保留开放式探索能力
- 当 $k>0$ 时，通过少样本学习软引导问题生成，保持语义一致性

生成过程：
$$q_t \sim Q_\theta(\cdot \mid \mathcal{C}_t, \mathcal{H}_t)$$

其中 $\mathcal{C}_t = \text{Sample}_k(\mathcal{D}_H)$ 是采样的上下文示例。

**Challenger 奖励函数**：
$$\mathcal{R}_{\text{chal}}^{\text{R-Few}}(q_t) = \underbrace{1 - 2|\widehat{p}_{\text{succ}}(q_t) - \tfrac{1}{2}|}_{\text{难度塑形}} - \lambda_{\text{rep}}\,\text{RepPenalty}(q_t)$$

- 鼓励生成成功率接近 50% 的"中等难度"问题
- 重复惩罚项防止生成重复内容

**技巧**：Challenger 训练前先用 SFT 热身，帮助基础模型在长 prompt 下遵循指令格式。

### Online Curriculum Solver

Solver 不是无差别训练所有数据，而是采用**在线课程学习机制**：

1. **难度估计**：对每个问题 $q$，Solver 进行 $M$ 次推理，计算成功率：
   $$\widehat{p}_{\text{succ}}(q) = \frac{1}{M} \sum_{m=1}^{M} J(q, a^{(m)})$$

2. **课程选择**：根据成功率排序，选择中等难度区间 $[\tau_{\text{low}}, \tau_{\text{high}}]$（实验中设为 $[0.3, 0.7]$）：
   $$\mathcal{D}_{\text{cur}} = \{(q,a) : 0.3 \le \widehat{p}_{\text{succ}}(q) \le 0.7\}$$

3. **混合训练**：将合成数据和人类数据统一纳入课程，确保 Solver 专注于"最近发展区"

**Solver 奖励函数**：
$$\mathcal{R}_{\text{sol}}^{\text{R-Few}}(q,a) = w_{\text{cur}}(q) \cdot \mathbf{1}\{a = \tilde{y}(q)\} + \lambda_{\text{hum}} \, w_{\text{hum}}(q) \cdot \mathbf{1}\{(q,a)\in\mathcal{D}_H\}$$

- $w_{\text{cur}}(q)$：课程权重
- $\lambda_{\text{hum}}$：人类数据上采样权重（实验中设为 2.0），防止遗忘

### 迭代共同进化

每轮迭代：
1. Challenger 用 few-shot 示例生成合成问题
2. Solver 尝试解答并估计成功率
3. 在线课程过滤器从合成+人类数据池中选择中等难度样本
4. 用 GRPO 分别更新 Challenger 和 Solver

## 实验结果

### 基础模型

- Qwen3-4B-Base
- Qwen3-8B-Base

### 数学推理基准

| 模型 | Math 平均分 | 数据量 |
|---|---|---|
| Qwen3-8B-Base | 68.0 | - |
| R-Zero | 68.0 | 无 |
| **R-Few (1%)** | **69.1** | **2.3k** |
| **R-Few (5%)** | **71.0** | **11.6k** |
| General-Reasoner | 71.0 | 232k |

**关键发现**：
- R-Few (5%) 用 **5% 数据**达到 General-Reasoner 的性能（后者用 20 倍数据）
- 相比 R-Zero，R-Few 在数学任务上提升 **+3.0 分**

### 通用推理基准

在 MMLU-Pro、SuperGPQA、GPQA-Diamond、BBEH 上：
- Qwen3-8B + R-Few (5%) 达到 38.9 分
- 显著优于 R-Zero (36.4) 和基础模型 (35.4)

### 模型规模效应

8B 模型比 4B 模型展现更强的自进化能力：
- Qwen3-4B + R-Few (5%)：50.7
- Qwen3-8B + R-Few (5%)：56.7（超越 General-Reasoner 的 56.0）

说明更大模型能更好地解释人类引导信号，生成更高质量的合成数据。

## 与其他方法的对比

| 维度 | R-Zero | Absolute Zero | SPICE | R-Few |
|---|---|---|---|---|
| 数据需求 | 完全无数据 | 无数据（限代码） | 需要文档语料库 | **1-5% 锚点数据** |
| 引导方式 | 无引导 | Python 执行验证 | 文档检索 | **Few-shot 示例** |
| 课程学习 | 无 | 无 | 无 | **在线难度排序** |
| 稳定性 | 易崩溃 | 限定领域 | 依赖语料质量 | **高稳定性** |
| 适用范围 | 通用但不稳定 | 仅代码生成 | 需文档支持 | **通用推理** |

## 消融实验

在 Qwen3-8B-Base 上移除各组件的影响（R-Few 5%）：

| 配置 | Math | General |
|---|---|---|
| 完整 R-Few | 71.0 | 38.9 |
| 去除 Challenger 训练 | 68.0 (-3.0) | 37.4 (-1.5) |
| 去除 Challenger 热身 | 69.1 (-1.9) | 37.9 (-1.0) |
| 去除课程学习 | 69.1 (-1.9) | 38.1 (-0.8) |

**Challenger 训练**影响最大，说明引导式问题生成是核心。

## 关键分析

### 1. 领域相关性

实验发现人类数据的领域与性能提升高度相关：
- **数学数据**对所有领域都有广泛贡献（最有用的类别）
- 领域内数据效果最好（如数学数据→数学任务）
- 跨领域强关联：数学-物理、商业-经济

**可控性启示**：通过选择特定领域的锚点数据，可以引导模型向期望方向进化。

### 2. 稳定性与奖励 Hacking

对比 R-Zero 和 R-Few 的训练曲线：

**R-Zero 的问题**：
- **多样性崩溃**：2-gram 多样性从 35 暴跌至 20（前 50 步）
- **长度膨胀**：问题长度爆炸式增长，通过冗长来伪造多样性
- **奖励 Hacking**：用啰嗦而非深度推理来提升感知难度

**R-Few 的优势**：
- 多样性全程稳定
- 问题长度保持一致
- 难度提升来自真实推理复杂度，而非表面特征

### 3. 真实难度 vs 表面难度

用 Gemini-2.5-Pro 重新标注不同训练阶段生成的问题：
- R-Zero：难度提升主要来自长度增加（奖励 Hacking）
- R-Few：在保持长度稳定的情况下，生成真正更难的推理问题

## 局限性

1. **人类数据依赖**：虽然只需 1-5%，但仍需高质量锚点数据
2. **验证信号限制**：当前主要在可验证任务（数学、推理）上有效
3. **开放式领域挑战**：缺乏客观正确性信号的领域（如创意写作）难以应用
4. **计算成本**：在线课程学习需要多次推理来估计难度

## 关键洞察

1. **最小监督的威力**：仅 1-5% 人类数据就能防止概念漂移和多样性崩溃，实现稳定自进化
2. **Few-shot 作为软引导**：通过随机采样 0-5 个示例，在开放探索和语义锚定之间取得平衡
3. **在线课程的必要性**：动态选择中等难度样本是高效学习的关键，避免过简单或过难的样本
4. **规模与引导的协同**：更大模型能更好地利用人类引导信号，展现更强的自进化潜力
5. **可控的自进化**：通过选择特定领域的锚点数据，可以引导模型向期望方向进化，而非无目的漂移
6. **真实难度 vs 表面特征**：轻量级人类监督防止模型通过表面特征（如冗长）来 hack 奖励，确保真实的推理能力提升
