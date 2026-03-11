---
title: "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing"
authors: "Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin"
institution: "University of Washington, Allen Institute for AI"
venue: "ICLR 2025"
arxiv_id: "2406.08464"
tags: [alignment, data-synthesis, instruction-tuning, LLM, self-synthesis, SFT, DPO]
---

# Magpie：通过"无提示"提示对齐LLM从零合成对齐数据

## 核心贡献

1. 提出了一种全新的对齐数据合成方法 Magpie，核心思想极其简洁：仅向已对齐的LLM输入预查询模板（pre-query template），不提供任何用户查询，利用自回归特性让模型自动生成高质量、多样化的指令。
2. 该方法完全自动化，无需人工干预、无需种子问题、无需提示工程、无需调用GPT-4等商业API，成本极低且可大规模扩展。
3. 仅使用Magpie生成的数据进行SFT，即可超越使用其他公开数据集进行SFT+DPO的模型表现；使用不超过40万条数据对齐的模型，在AlpacaEval 2上甚至超过了使用超过1000万条数据对齐的官方Llama-3-8B-Instruct。
4. 展示了Magpie的可扩展性：支持多轮对话、偏好优化数据、领域特定数据和多语言数据的生成。

## 方法详解

Magpie的核心流程分为两步：

**第一步：指令生成。** 对于开源对齐模型（如Llama-3-8B-Instruct），其聊天模板定义了预查询模板 $T_{pre-query}$（如 `<|start_header_id|>user<|end_header_id|>`）。Magpie仅将这个预查询模板作为输入送入模型，由于模型在对齐训练中已学习了指令的分布，它会自回归地生成一条完整的用户指令，直到产生结束符。重复此过程即可批量生成多样化的指令集。

**第二步：响应生成。** 将第一步生成的指令按标准聊天模板格式送入同一LLM，生成对应的响应。指令与响应配对即构成指令数据集。

**扩展能力：**
- **数据过滤：** 提供8种可配置的过滤指标（质量、难度、相似度、奖励分数等），用户可自定义筛选策略。
- **多轮对话（Magpie-MT）：** 在首轮指令-响应后追加预查询模板，模型继续生成后续轮次的用户指令。
- **偏好优化数据（Magpie-DPO）：** 对高质量指令采样k次响应，用奖励模型标注最优和最差响应，构建偏好对。
- **领域特定与多语言：** 通过系统提示控制生成指令的领域（数学、代码等）和语言。

**关键发现：** 即使对齐训练中指令损失被mask掉，LLM仍能生成高质量指令，说明模型对指令分布存在隐式记忆。

## 实验结果

**数据集规模与成本：**
- Magpie-Air（Llama-3-8B-Instruct生成）：300万条，206 GPU小时，每千条约$0.12
- Magpie-Pro（Llama-3-70B-Instruct生成）：100万条，614 GPU小时，每千条约$1.1

**主要对比实验（基于Llama-3-8B-Base微调）：**
- 在AlpacaEval 2和Arena-Hard上，Magpie数据集的SFT模型全面超越ShareGPT、WildChat、Evol Instruct、UltraChat、OpenHermes、Tulu V2 Mix、GenQA等8个公开数据集的SFT模型。
- 仅用Magpie做SFT，即超过使用UltraChat+UltraFeedback做SFT+DPO的模型。
- 在AlpacaEval 2上，Magpie-SFT模型对比官方Llama-3-8B-Instruct的LC胜率超过50%，说明评估者更偏好Magpie对齐的模型。
- 结合Magpie-DPO后，模型在AlpacaEval 2上甚至超过GPT-4-Turbo(1106)。

**跨模型泛化：**
- 在Qwen2-1.5B、Qwen1.5-4B、Qwen1.5-7B上微调，Magpie数据同样优于各模型的官方对齐版本。
- MagpieLM（基于Llama-3.1-Minitron-4B）在10B以下开源指令模型中排名第一。

**数据质量分析：**
- t-SNE可视化显示Magpie-Pro的覆盖范围完全包含Alpaca、Evol Instruct和UltraChat。
- 大部分指令质量评级为"average"及以上，安全性分析显示有害内容不足1%。
- 任务类别分布与真实用户请求分布一致，以信息检索为主，其次是创意写作、建议寻求、规划和数学。

## 局限性

1. **推理能力不足：** Magpie对齐的模型在数学和推理基准测试上表现下降。虽然通过生成领域特定的"booster"数据集（15万条数学/代码/推理指令）可以缓解，但与官方模型仍有差距。
2. **依赖源模型能力：** 生成数据的质量和多样性受限于所使用的对齐LLM本身的能力。Magpie-Pro（70B生成）质量明显优于Magpie-Air（8B生成）。
3. **安全风险：** 虽然有害数据不足1%，但直接使用未过滤的原始数据进行微调可能导致不安全行为，需要过滤流程。

## 关键洞察

1. **对齐LLM本身就是最好的指令数据源。** 经过大规模对齐训练的模型已经内化了高质量指令的分布，Magpie的核心洞察是直接"提取"这些隐式知识，而非通过复杂的提示工程间接获取。
2. **数据质量 > 数据数量。** Magpie用不到40万条数据就超越了使用超过1000万条数据的官方模型，再次验证了高质量数据的重要性。
3. **自回归特性的创造性利用。** 聊天模板的预查询部分本质上是一个"不完整的提示"，模型的自回归补全机制自然地将其补全为一条合理的用户指令——这是一个极其简洁优雅的设计。
4. **motivation.tex中的反直觉发现：** 强模型的监督不一定能提升小模型（SLM）的性能。实验表明，用同族模型（如Gemma2系列监督Gemma2-2B）的效果往往优于跨族强模型（如Llama-3-405B），这为Magpie使用同一模型自我合成数据提供了理论支撑。
