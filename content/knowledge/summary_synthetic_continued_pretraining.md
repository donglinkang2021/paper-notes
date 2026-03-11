---
title: "Synthetic Continued Pretraining"
authors: "Zitong Yang, Neil Band, Shuangping Li, Emmanuel Candès, Tatsunori Hashimoto"
institution: "Stanford University"
venue: "ICLR 2025"
arxiv_id: "2409.07431"
tags: ["synthetic-data", "continued-pretraining", "entity-graph", "knowledge-acquisition", "data-efficiency", "RAG", "parametric-knowledge", "scaling-analysis"]
---

# Synthetic Continued Pretraining：用实体图合成解决小语料知识获取

## 核心贡献

语言模型在大规模预训练中可以获取大量世界知识，但这一知识获取过程是**极其数据低效的**——要学会一个事实，模型需要在数百到数千种不同表述中见过它。这在适配小规模领域语料时造成严重问题：每个事实可能仅出现一次或少数几次，直接做 continued pretraining (CPT) 效果很差。

论文提出 **Synthetic Continued Pretraining**：先用小领域语料合成一个更大的、更适合学习的合成语料，再在合成语料上做 continued pretraining。具体实例化为 **EntiGraph** 算法：

1. **实体抽取**：从源文档中提取显著实体
2. **关系分析**：用 LLM 对实体子集（对和三元组）生成关于它们在文档上下文中关系的描述文本
3. **合成语料构建**：收集所有关系分析文本作为合成语料

将 EntiGraph 应用于 265 篇 QuALITY 文章和书籍（1.3M tokens），合成出 455M tokens 的语料，在 Llama 3 8B 上做 CPT 后，闭卷问答准确率从 39.49% 提升到 56.22%，且这种知识与 RAG 互补，组合后进一步提升到 62.60%。

## 方法详解

### 一、问题设定

论文定义了一个清晰的问题框架：

- **输入**：小规模源文档集 $\mathcal{D}_s$（如一组教科书、文章）
- **目标**：让预训练模型学会 $\mathcal{D}_s$ 中的知识，存储在参数中（parametric knowledge）
- **评估**：在不允许访问 $\mathcal{D}_s$ 的条件下，用知识密集���查询 $\mathcal{Q}_t$ 测试模型

合成 CPT 的核心操作：

$$\mathcal{A}_{\mathrm{synth}}: \mathcal{D}_s \longmapsto \mathcal{D}_{\mathrm{synth}}$$

然后在 $\mathcal{D}_{\mathrm{synth}}$ 上而非 $\mathcal{D}_s$ 上做 continued pretraining。

### 二、EntiGraph 算法

EntiGraph 通过两步提示策略将多样性外化到组合结构上：

#### Step 1: 实体抽取

从每篇文档中提取显著实体 $\{E_1, E_2, \ldots, E_n\}$：

$$\{E_1, E_2, \ldots, E_n\} \sim \mathrm{LM}_{\mathrm{gen}}(\texttt{entity\_extraction}(\mathcal{D}_s))$$

实体包括人名、地名、概念、事件等。例如，线性代数教科书中可能提取出"线性空间"、"向量"、"SVD"等。

#### Step 2: 关系分析

对实体子集进行关系分析，生成描述它们在源文档上下文中关系的合成文本：

$$\widetilde{D}_{E_{i_1} \ldots E_{i_k}} \sim \mathrm{LM}_{\mathrm{gen}}\big(\texttt{relation\_analysis}(D, E_{i_1}, \ldots, E_{i_k})\big)$$

实际中，EntiGraph 穷举所有实体对和三元组的关系分析，将多样性问题**外化**为图上的组合问题——$n$ 个实体可以产生 $\binom{n}{2} + \binom{n}{3}$ 种组合。

### 三、为什么 EntiGraph 有效？

论文提供了理论分析来解释 EntiGraph 的成功机制。

#### 数学模型

将知识表示为实体集合 $\mathcal{V}$ 上的关系图，源文档 $\mathcal{D}_s$ 对应图中已知的边集。训练被建模为记忆过程——模型记住训练中显式看到的关系。

EntiGraph 的合成过程被建模为图上的随机游走：
1. 随机选择实体对 $(x, y)$
2. 在已知关系图中做 BFS 搜索 $x$ 到 $y$ 的路径
3. 如果路径存在，所有路径上的关系都被加入模型的知识集

#### 核心定理

论文证明了准确率 $\mathrm{Acc}(\mathbf{M}_t)$ 随合成 token 数 $t$ 的增长遵循混合指数形式：

$$\mathrm{Acc}(\mathbf{M}_t) \approx 1 - \sum_{k=2}^{V} \alpha_k e^{-\beta_k t}$$

这预测了三阶段行为：(i) 线性增长 → (ii) 对数线性增长 → (iii) 渐近饱和。实验数据的拟合非常好，验证了该理论模型。

关键洞察：**EntiGraph 不创造新知识，而是"重排"知识布局**，使其更适合通过下一 token 预测来学习。

## 实验结果

### 一、实验设置

- **源语料**：265 篇 QuALITY 文章和书籍，总计 1.3M tokens（比现代 CPT 工作小 10,000 倍）
- **合成语料**：用 GPT-4 (turbo) 运行 EntiGraph，生成 455M tokens
- **模型**：Llama 3 8B Base
- **评估**：4,609 个闭卷多选 QA 题（QuALITY 数据集）

### 二、主要结果

| 方法 | 闭卷 QA 准确率 |
|---|---:|
| Llama 3 8B Base | 39.49% |
| Raw CPT（直接在 1.3M tokens 上训练） | < 39.49%（退化） |
| Rephrase CPT（释义增强，38M tokens） | ~45%（低效扩展） |
| **EntiGraph CPT（455M tokens）** | **56.22%** |
| GPT-3.5（闭卷） | 44.81% |
| GPT-4（闭卷） | 51.30% |

关键发现：

- **Raw CPT 甚至不如基线**：直接在 1.3M tokens 上训练导致性能退化，因为 (i) 分布过窄损害通用能力，(ii) 知识表述多样性不足导致反转诅咒等问题
- **释义增强扩展缓慢**：Rephrase CPT 增加了表达多样性但未增加知识结构的多样性
- **EntiGraph 展现对数线性扩展**：闭卷 QA 准确率随合成 token 数呈对数线性增长，最高达 455M tokens
- **EntiGraph CPT 超越 GPT-4 闭卷表现**：在 QuALITY 知识上，8B 参数模型通过合成 CPT 超越了 GPT-4 的闭卷表现

### 三、开卷实验：合成 CPT 与 RAG 互补

| 方法 | 准确率 | Recall@8 |
|---|---:|---:|
| Llama 3 8B + RAG | 60.35% | 99.63% |
| **EntiGraph CPT + RAG** | **62.60%** | **99.63%** |
| GPT-3.5 + Oracle RAG | 72.60% | 100% |
| GPT-4 + Oracle RAG | 86.09% | 100% |

关键发现：
- RAG 带来 20.86% 的提升（39.49%→60.35%），EntiGraph CPT 带来 16.73%（39.49%→56.22%），即 **EntiGraph CPT 提供了 RAG 改进的 80% 以上**
- 两者组合进一步提升到 62.60%，证明**参数化知识与非参数化检索互补**

### 四、指令微调兼容性

EntiGraph CPT 模型经过��令微调后（EntiGraph Instruct），可以：
- 根据文章标题生成摘要（无需文章本身）
- 回答关于文章的隐式引用问题
- 跨文章进行比较推理

闭卷摘要评估表明，EntiGraph Instruct 的虚假声明率与 GPT-3.5/GPT-4 相当，远低于 Raw Instruct。

## 与相关工作的关系

### 与 SBP 的对比

同一研究组的后续工作 SBP（2509.15248）将类似思想从"小语料 CPT"推广到"通用预训练"：
- **EntiGraph**：显式构建实体图，通过提示 LLM 生成关系分析文本
- **SBP**：隐式学习文档间条件分布 $p(d_2 \mid d_1)$，无需显式实体图
- 两者共享核心洞察：标准训练忽略了跨文档的知识连接

### 与释义/重写方法的对比

EntiGraph 与 WRAP、Kimi K2 的释义增强有本质区别：
- 释义仅增加表达多样性，不改变知识结构
- EntiGraph 通过实体组合改变���识的组织方式，使之前隐式的跨实体关系变为显式训练信号
- 实验证实 EntiGraph 的扩展效率远高于释义

### 与知识编辑的关系

论文区分了知识编辑（修改单个事实三元组）和知识获取（从整个文档语料中学习）。最相关的是 deductive closure training——先推导事实编辑的蕴含，再在蕴含上微调。EntiGraph 可视为这种思想的大规模推广。

## 局限性

1. **合成成本高**：使用 GPT-4 生成 455M tokens 的成本不低，尽管比 QA SFT 更经济
2. **实体图的组合爆炸**：穷举所有实体对和三元组在实体数很多时不可行
3. **依赖外部 LLM 生成**：合成质量上限受限于生成模型（GPT-4）的能力
4. **仅在 QuALITY 数据集上验证**：能否推广到其他领域需要更多实验
5. **理论模型的简化假设**：BFS 模型假设知识获取为纯记忆过程，忽略了泛化能力

## 关键洞察

1. **知识获取的数据低效性是根本障碍**：要学会 $(A, B)$ 和 $(B, C)$ 的关系，模型不一定能推断 $(A, C)$。这不是模型能力问题，而是下一 token 预测目标的固有局限。
2. **多样性是合成 CPT 成功的关键**：释义增加了表达多样性但效果有限，EntiGraph 通过实体图的组合结构实现了更深层的知识结构多样性。
3. **合成数据不创造新知识，而是重排知识**：EntiGraph 的理论分析表明，合成数据通过"知识布局重排"使原本隐式的关系变为显式训练信号，从而提升学习效率。
4. **参数化知识与 RAG 互补**：这是一个重要的实践发现——合成 CPT 获得的参数化知识能进一步提升 RAG 系统的性能，两种知识获取方式并不冲突。
5. **对数线性扩展暗示持续改进空间**：455M tokens 的扩展曲线仍处于对数线性阶段，尚未达到理论预测的饱和平台，表明进一步扩展合成数据可能带来更多收益。

## 一句话总结

**Synthetic Continued Pretraining 通过 EntiGraph 实体图增强将 1.3M tokens 的小语料合成为 455M tokens，使 Llama 3 8B 的闭卷问答准确率从 39.49% 提升到 56.22%，并证明了合成获得的参数化知识与 RAG 互补，揭示了"知识重排"作为提升数据学习效率的核心机制。**
