---
title: "LIMO: Less is More for Reasoning"
authors: "Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu"
institution: "Shanghai Jiao Tong University, SII-GAIR, Fudan University, The Hong Kong Polytechnic University"
venue: "COLM 2025"
arxiv_id: "2502.03387"
tags: ["reasoning", "data-efficiency", "supervised-finetuning", "data-curation", "test-time-scaling", "math-reasoning", "less-is-more", "cognitive-templates"]
---

# LIMO：少即是多的推理

## 核心贡献

LIMA 证明了对齐只需少量数据，但推理是否也如此？主流观点认为数学推理需要大规模训练数据（数万到数十万条），因为推理任务的计算复杂度远高于格式对齐。LIMO 直接挑战了这一假设。

论文提出 **Less-Is-More Reasoning Hypothesis（LIMO 假设）**：在预训练阶段已经全面编码了领域知识的基础模型中，复杂推理能力可以通过极少量但精心设计的认知过程示范来激发。

实验验证：仅用 **800 条**精选数学推理样本对 Qwen2.5-32B-Instruct 做 SFT，LIMO 在 AIME24 上达到 63.3%，在 MATH500 上达到 95.6%，超越了使用 100 倍以上数据训练的模型（如 NuminaMath-100k 仅 6.5% AIME24、59.2% MATH500）。更重要的是，LIMO 展现了强大的分布外泛化能力，在未见过的 benchmark 上平均提升 45.8%。

## 方法详解

### 一、LIMO 假设的两个前提

LIMO 假设建立在两个关键前提上：

1. **预训练知识的完备性**：现代基础模型在预训练中已包含大量数学内容（Llama 3 使用 3.7T tokens 的数学推理数据），模型参数中已编码了丰富的数学知识
2. **推理链的认知模板作用**：高质量的推理链不是在"教"模型新知识，而是作为"认知模板"（cognitive templates），引导模型将已有知识组织成有效的推理过程

因此，激发复杂推理的阈值不取决于任务复杂度，而取决于：(1) 模型预训练知识的完备程度，(2) 后训练示例作为认知模板的有效性。

### 二、数据策划流程

LIMO 的数据策划是一个多层过滤系统：

#### 问题选择

从数百万数学题的候选池出发：

1. **粗粒度难度过滤**：用 Qwen2.5-Math-7B-Instruct 做 4 次尝试，能做对的题目被排除
2. **细粒度难度评估**：用 DeepSeek-R1-Distill-Qwen-32B 对每题采样 32 次，仅保留成功率在 1-3/32 的题目
3. **去重**：与所有评测 benchmark 做 n-gram 匹配去重

经过这一流程，从数百万题中筛选出 2,125 题（LIMO-Pool）。

#### 推理链构建

对 LIMO-Pool 中的每个问题：

1. 使用三个 SOTA 推理模型（DeepSeek R1、DeepSeek-R1-Distill-Qwen-32B、QwQ-32B）生成多个解答
2. 用规则化评分系统对推理链质量打分，四个维度：
   - **Elaborated Reasoning**（30%）：解答长度，衡量推理展开的充分程度
   - **Self-Verification**（20%）：验证相关词频（"check"、"verify"），衡量自我检查程度
   - **Exploratory Approach**（25%）：试探性表达频率（"perhaps"、"might"），衡量探索性
   - **Adaptive Granularity**（25%）：连接词频率（"therefore"、"since"），衡量推理粒度适应性
3. 所有关键词频率按文本长度归一化

最终选择每题最高分的解答，按总分排序取 top-800，构成 LIMO 数据集。

### 三、训练配方

- 基座模型：Qwen2.5-32B-Instruct
- 全参数 SFT，DeepSpeed ZeRO-3 + FlashAttention-2
- 学习率 $5 \times 10^{-6}$，余弦衰减，无 warmup
- 15 个 epoch，batch size 64
- 最大序列长度 16,384 tokens

## 实验结果

### 一、主要结果

| 模型 | 训练数据量 | AIME24 | MATH500 | AMC23 |
|---|---:|---:|---:|---:|
| Qwen2.5-32B-Instruct（基座） | — | 16.5 | 79.4 | — |
| NuminaMath-100k | 100K | 6.5 | 59.2 | — |
| OpenThoughts-114k | 114K | 56.7 | 94.2 | — |
| QwQ-32B-Preview | — | 50.0 | 89.8 | 83.6 |
| OpenAI o1-preview | — | 44.6 | 85.5 | — |
| **LIMO（800 条）** | **800** | **63.3** | **95.6** | **96.3** |

关键发现：
- LIMO 用 0.8K 数据超越了 100K+ 数据训练的模型
- 在 AIME24 上超越 o1-preview（63.3% vs 44.6%）和 QwQ-32B-Preview（63.3% vs 50.0%）
- NuminaMath-100k 反而导致性能退化（AIME24 从 16.5% 降到 6.5%），说明大量低质量数据可能有害

### 二、分布外泛化

| Benchmark | 基座模型 | LIMO | 提升 |
|---|---:|---:|---:|
| OlympiadBench | 45.3 | 67.6 | +22.3 |
| CHMath | 68.5 | 84.2 | +15.7 |
| GradeSchool | 63.8 | 76.2 | +12.4 |
| GPQA | — | 70.7 | — |
| MinervaMath | — | 55.6 | — |

LIMO 在完全未见过的 benchmark 上平均提升 45.8%，证明它学到的不是特定题目的解法，而是通用的推理策略。

### 三、消融实验

#### RQ1: 数据量 vs 数据质量

对比 LIMO（800 条高质量）与 NuminaMath-100k（100K 条普通质量）和 OpenThoughts-114k（114K 条中等质量）：
- LIMO 用 1% 的数据量超越了 NuminaMath-100k
- LIMO 用 0.7% 的数据量与 OpenThoughts-114k 持平甚至更优

#### RQ2: 推理链质量

对比不同质量的推理链（短 CoT vs 长 CoT）：
- 使用 DeepSeek-R1 风格的长推理链（包含自我验证、探索、回溯）显著优于传统短 CoT
- 推理链的质量维度中，Self-Verification 和 Exploratory Approach 对性能贡献最大

#### RQ3: 预训练知识的关键性

对比 Qwen1.5-32B-Chat vs Qwen2.5-32B-Instruct（相同架构，不同预训练数据）：
- Qwen2.5 + LIMO：AIME24 63.3%，MATH500 95.6%
- Qwen1.5 + LIMO：AIME24 9.2%，MATH500 65.2%
- **54.1% 的绝对差距**直接证明了 LIMO 假设的第一个前提：预训练知识的完备性是少样本推理激发的前提条件

#### RQ4: 模型规模效应

在 Qwen2.5-Instruct 系列（3B→7B→14B→32B→72B）上用相同 LIMO 数据集训练：
- AIME24：3B 仅 2.5%，72B 达 68.3%
- 32B→72B 的提升边际递减（63.3%→68.3%），暗示存在饱和点
- MATH500 上小模型也能达到较高水平（14B 已达 93.0%）

#### RQ5: 样本效率

从 LIMO-Pool 中按质量排序取不同大小子集（400→800→1200→1600→2000）：
- **400 条就能产生巨大提升**：AIME24 从 16.5% 到 57.5%，MATH500 从 79.4% 到 94.8%
- 800 条之后边际收益递减
- 2000 条达到最高（AIME24 69.6%，MATH500 95.8%），但相比 800 条提升有限

## 与相关工作的关系

### 与 LIMA 的传承

LIMO 明确引用 LIMA 作为灵感来源，将"少即是多"从对齐推广到推理。两者的核心假设高度一致：

| 维度 | LIMA | LIMO |
|---|---|---|
| 假设名称 | Superficial Alignment Hypothesis | Less-Is-More Reasoning Hypothesis |
| 核心论点 | 能力在预训练中习得，对齐只是格式 | 知识在预训练中编码，推理只需认知模板 |
| 数据量 | 1,000 条 | 800 条 |
| 基座模型 | LLaMa 65B | Qwen2.5-32B-Instruct |
| 任务 | 通用指令遵循 | 数学推理 |

### 与 s1 的对比

s1 和 LIMO 几乎同时期发表，都用极少量数据做推理微调，但方法论不同：
- **s1**：1K 样本 + Budget Forcing 解码控制，强调测试时计算扩展
- **LIMO**：800 样本，纯 SFT 无解码干预，强调推理链质量和认知模板
- 两者共同验证了"少量高质量推理样本可以激活预训练模型的推理能力"

### 与大规模推理训练的对比

NuminaMath-100k 的失败案例（AIME24 从 16.5% 退化到 6.5%）是论文最有力的反面证据——大量低质量推理数据不仅无益，反而有害。这与 LIMA 发现的"增加数量不提升性能"一致，但更极端：在推理场景下，低质量数据会主动损害模型能力。

## 局限性

1. **仅在数学推理上验证**：LIMO 假设是否适用于代码推理、科学推理等其他领域尚未验证
2. **依赖强基座模型**：Qwen1.5 vs Qwen2.5 的对比表明，LIMO 的成功高度依赖预训练质量，弱基座模型无法受益
3. **推理链来源依赖 SOTA 模型**：推理链由 DeepSeek R1 和 QwQ 生成，本质上仍是蒸馏
4. **评分系统的启发式性质**：四维度加权评分是手工设计的，最优权重和维度选择可能因任务而异
5. **小模型效果有限**：3B 模型在 AIME24 上仅 2.5%，说明 LIMO 方法对模型规模有较高要求

## 关键洞察

1. **推理能力的激发阈值远低于预期**：400 条数据就能将 AIME24 从 16.5% 提升到 57.5%，这说明推理能力的"激活能量"很低，关键在于找到正确的"认知模板"。
2. **预训练知识是少样本推理的前提**：Qwen1.5 vs Qwen2.5 的 54.1% 差距是论文最重要的实验发现。它说明 LIMO 假设有一个硬性前提——如果预训练知识不够丰富，再好的认知模板也无法激发推理能力。
3. **大量低质量数据可能有害**：NuminaMath-100k 导致性能退化，这比 LIMA 的"增加数量无益"更极端。在推理场景下，低质量数据可能教会模型错误的推理模式，比不训练更糟。
4. **长推理链是关键的认知模板**：包含自我验证、探索、回溯的长推理链（DeepSeek R1 风格）远优于传统短 CoT。这与 s1 的 Budget Forcing 发现一致——模型需要足够的"认知工作空间"来展开推理。
5. **从 LIMA 到 LIMO 的范式推广**：LIMA 证明了"对齐是表面的"，LIMO 进一步证明了"推理激发也可以是表面的"——只要预训练足够好，极少量高质量示范就能激活复杂推理。这暗示了一个更一般的原则：**后训练的核心价值不在于灌输新能力，而在于提供正确的"认知脚手架"来组织已有知识。**

## 一句话总结

**LIMO 用 800 条精选数学推理样本将 Qwen2.5-32B-Instruct 的 AIME24 准确率从 16.5% 提升到 63.3%，超越使用 100 倍数据的方法，证明了在知识丰富的基础模型中，复杂推理可以通过极少量"认知模板"式的高质量示范来激发。**
