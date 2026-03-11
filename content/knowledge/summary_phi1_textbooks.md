---
title: "Textbooks Are All You Need"
authors: "Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li"
institution: "Microsoft Research"
venue: "arXiv 2023"
arxiv_id: "2306.11644"
tags: [code-generation, data-quality, synthetic-data, scaling-laws, small-models, phi-1]
---

# "教科书就是你所需要的一切"——用高质量数据打破缩放定律

## 核心贡献

本文的核心论点是：**数据质量可以从根本上改变缩放定律的形态**。作者训练了一个仅 1.3B 参数的代码生成模型 phi-1，使用不到 7B tokens 的"教科书质量"数据（包括 GPT-3.5 合成数据和从网络筛选的高质量代码），在 HumanEval 上达到 50.6% pass@1，在 MBPP 上达到 55.5% pass@1。这一成绩超越了几乎所有开源模型，而这些模型的参数量和训练数据量通常是 phi-1 的 10 倍到 100 倍以上。

具体而言：
- phi-1（1.3B 参数）在 HumanEval 上超越了 StarCoder（15.5B，33.6%）、CodeGen-Mono-16.1B（29.3%）、PaLM-Coder（540B，35.9%）等模型
- 训练仅需 8 块 A100 GPU 运行不到 4 天
- 在 CodeExercises 上微调后，模型展现出训练数据中不存在的涌现能力

## 方法详解

### 数据构建：三个数据集

phi-1 的训练依赖三个精心构建的数据集：

**1. 筛选后的代码-语言数据集（约 6B tokens）**

从 The Stack（Python 子集）和 StackOverflow 中筛选，总计超过 3500 万个文件/样本。筛选流程：
- 使用 GPT-4 对约 10 万个样本标注"教育价值"（这是 GPT-4 的唯一用途，仅用于替代人工标注）
- 基于标注数据训练随机森林分类器，使用预训练 CodeGen 模型的输出嵌入作为特征
- 用分类器对全部数据进行质量筛选

作者指出标准代码数据集的四个问题：不自包含（依赖外部模块）、缺乏有意义的计算（多为样板代码）、算法逻辑埋在复杂函数中、主题分布不均衡。

**2. 合成教科书数据集（< 1B tokens）**

由 GPT-3.5 生成的 Python 教科书，特点是自然语言与代码片段交织。通过约束主题和目标受众来实现多样性。内容侧重于推理和基础算法技能。

**3. 合成练习数据集 CodeExercises（约 180M tokens）**

由 GPT-3.5 生成的 Python 函数补全练习。每个练习是一个带 docstring 的函数，需要模型补全实现。通过约束函数名来实现多样性。

前两个数据集合称 "CodeTextbook"，用于预训练得到 phi-1-base；第三个数据集用于微调得到 phi-1。

### 模型架构与训练

架构方面采用常规设计：
- Decoder-only Transformer，使用 FlashAttention
- phi-1（1.3B）：24 层，隐藏维度 2048，MLP 内部维度 8192，32 个注意力头
- phi-1-small（350M）：20 层，隐藏维度 1024，16 个注意力头
- 使用旋转位置编码（RoPE），rotary dimension 32
- 未使用 Fill-In-the-Middle（FIM）或 Multi-Query-Attention（MQA）

训练细节：
- 预训练：有效批大小 1024，最大学习率 1e-3，训练 36,000 步（取 24,000 步的检查点，约 8 个 epoch，总计约 50B tokens）
- 微调：有效批大小 256，最大学习率 1e-4，训练 6,000 步
- 使用 fp16 训练，AdamW 优化器，线性预热-线性衰减学习率调度
- 预训练不到 4 天，微调额外 7 小时

## 实验结果

### 主要基准测试

| 模型 | 参数量 | 训练 Tokens | HumanEval | MBPP |
| --- | --- | --- | --- | --- |
| Codex-12B | 12B | 100B | 28.8% | - |
| CodeGen-Mono-16.1B | 16.1B | 577B | 29.3% | 35.3% |
| PaLM-Coder | 540B | 780B | 35.9% | 47.0% |
| StarCoder | 15.5B | 1T | 33.6% | 52.7% |
| StarCoder-Prompted | 15.5B | 1T | 40.8% | 49.5% |
| GPT-3.5 | 175B | N.A. | 47% | - |
| WizardCoder | 16B | 1T | 57.3% | 51.8% |
| **phi-1** | **1.3B** | **7B** | **50.6%** | **55.5%** |

### 数据质量的消融实验

- 350M 模型在未筛选 Stack 上训练：HumanEval 12.19%（200B tokens 后饱和）
- 350M 模型在筛选后数据上训练：17.68%（36k 步）
- 加入合成教科书后：20.12%
- phi-1-base（1.3B，仅预训练）：29%
- phi-1（1.3B，微调后）：50.6%

### 非常规问题评估

为排除数据污染，独立团队设计了 50 个非常规编程问题，使用 GPT-4 评分（0-10 分）：

| 模型 | 参数量 | 得分 |
| --- | --- | --- |
| CodeGen-Mono-350M | 350M | 19% |
| CodeGen-Mono-16.1B | 16.1B | 38% |
| StarCoder | 15.5B | 51% |
| phi-1-base | 1.3B | 37% |
| phi-1-small | 350M | 45% |
| **phi-1** | **1.3B** | **52%** |

### 数据去污染实验

通过嵌入距离和 AST 编辑距离对 CodeExercises 进行激进剪枝（最多移除 40% 数据），重新训练后的 phi-1 在 HumanEval 上仍达到 45.1%，依然超越 StarCoder-Prompted（41.5%）。这证明性能提升并非来自数据污染。

### 涌现能力

微调后的 phi-1 展现出训练数据中不存在的能力：
- **外部库使用**：能正确使用 PyGame、Tkinter 等库（CodeExercises 中不包含这些库）
- **复杂推理**：能理解多步骤逻辑指令并生成正确代码
- **对话能力**：展现出一定的问答交互能力（聊天数据仅存在于预训练中，不在微调数据中）

这表明微调过程帮助模型"重组和巩固"了预训练阶段获得的知识。

## 局限性

1. **仅限 Python**：phi-1 专注于 Python 编程，不支持多语言代码生成
2. **领域知识有限**：缺乏特定 API 和冷门包的使用知识
3. **对 prompt 风格敏感**：由于训练数据结构化程度高、语言风格多样性不足，模型对 prompt 中的语法错误或风格变化非常敏感，性能会显著下降
4. **合成数据错误率**：GPT-3.5 生成的数据存在较高错误率（但模型仍能从中学习正确模式）
5. **多样性度量缺失**：缺乏有效方法衡量数据集的多样性和冗余程度

## 关键洞察

**数据质量 > 数据数量 > 模型大小**。这篇论文最深刻的启示是：精心策划的小数据集可以胜过大规模低质量数据。phi-1 用不到 7B tokens 达到了需要 100B-1T tokens 的模型的性能水平，计算量差距达到两个数量级。

**"教科书质量"的定义**：好的训练数据应该像好的教科书一样——清晰、自包含、有教学意义、主题均衡。标准代码数据集中大量的样板代码、不完整片段和缺乏文档的复杂函数，对模型学习来说是低效的信号。

**合成数据的多样性是关键挑战**。简单地让 LLM 生成教科书会产生高度同质化的内容。受 TinyStories 启发，作者通过在 prompt 中注入随机约束（主题、受众、函数名）来诱导多样性。

**微调的"知识重组"效应**。仅 180M tokens 的 CodeExercises 微调不仅提升了目标任务性能（29% → 50.6%），还解锁了预训练阶段潜在的能力（如外部库使用）。这暗示微调可能起到了"知识蒸馏"或"知识对齐"的作用，帮助模型更好地调用已有知识。

**参数量对涌现的作用**。phi-1-small（350M）使用相同流程训练，在 HumanEval 上达到 45%，但在复杂任务上明显弱于 phi-1（1.3B），说明参数量仍然是涌现能力的关键因素。

---

*本摘要基于 arXiv:2306.11644，这是 Microsoft Research 的重要工作。phi-1 证明了在代码生成领域，高质量数据可以从根本上改变模型训练的效率，用 1.3B 参数和 7B tokens 达到了此前需要数百亿参数和万亿 tokens 才能实现的性能。这一发现对后续的 Phi 系列模型（Phi-1.5、Phi-2、Phi-3）以及整个 LLM 社区的数据策略产生了深远影响。*
