---
title: "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model"
authors: "Loubna Ben Allal*, Anton Lozhkov*, Elie Bakouch*, Gabriel Martín Blázquez*, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf"
institution: "HuggingFace"
venue: "arXiv 2025"
arxiv_id: "2502.02737"
tags: ["小语言模型", "数据策展", "多阶段预训练", "指令微调", "数学推理", "代码生成", "开源"]
---

# SmolLM2：当"小"变得强大——以数据为中心的小型语言模型训练

## 核心贡献

1. **构建了当前最优的小型语言模型 SmolLM2（17亿参数）**，在多项基准测试中超越 Qwen2.5-1.5B 和 Llama3.2-1B。
2. **提出三个全新高质量数据集**：
   - **FineMath**：高达540亿token的数学数据集，聚焦逐步推理内容，通过分类器过滤 Common Crawl 获得。
   - **Stack-Edu**：约1250亿token的教育性代码数据集，覆盖15种编程语言，从 StarCoder2Data 中筛选。
   - **SmolTalk**：指令微调数据集，融合合成对话数据（MagPie-Ultra）与任务专用数据（约束遵循、摘要、改写等）。
3. **提出基于性能驱动的多阶段训练策略**，在约11万亿token上进行训练，通过在线调整数据混合比例来最大化模型性能，避免了多次从头训练的高昂成本（约25万美元GPU算力）。

## 方法详解

### 预训练数据策展

- **英文网页数据**：对比 FineWeb-Edu（教育性强）和 DCLM（多样性强），最终采用 60% FineWeb-Edu + 40% DCLM 的混合比例，兼顾教育内容与常识推理，共5.1万亿token。
- **数学数据（FineMath）**：现有数据集（OpenWebMath 12B token、InfiMM-WebMath 40B token）规模不足且缺乏逐步推理内容。FineMath 从 Common Crawl 中提取数学页面，使用 Llama-3.1-70B-Instruct 进行质量评分和分类器过滤，最终得到 FineMath4+（10B token，仅保留4-5分内容）和 FineMath3+（34B token）。FineMath4+ 在 GSM8K 上实现2倍提升，在 MATH 上实现6倍提升。
- **代码数据（Stack-Edu）**：对 StarCoder2Data 的15种主要编程语言训练教育质量分类器（基于 Llama3-70B-Instruct 标注），过滤后保留约1250亿token。过滤后 Python HumanEval 从20.7提升至25.6。

### 四阶段预训练

采用 Warmup-Stable-Decay（WSD）学习率调度，在256块H100上使用 nanotron 框架训练：

| 阶段 | Token范围 | 数据混合 | 关键发现 |
|------|-----------|----------|----------|
| 阶段1（稳定期） | 0-6T | 90%网页（FW-Edu/DCLM=60/40）+ 10%代码（StarCoderData） | 知识和推理表现符合预期，但数学和代码能力较弱 |
| 阶段2（稳定期） | 6-8T | 75%网页 + 20%代码 + 5%数学（OWM） | 代码能力提升；MMLU MCF准确率超过随机水平（>25%） |
| 阶段3（稳定期） | 8-10T | 70%网页（FW-Edu/DCLM调整为40/60）+ 20%代码（Stack-Edu）+ 10%数学 | 引入 Stack-Edu 和 InfiMM-WebMath；出现loss spike但指标最终恢复 |
| 阶段4（衰减期） | 10-11T | 58%网页 + 24%代码 + 14%数学（含FineMath4+）+ 4% Cosmopedia | 数学和代码能力大幅提升 |

训练后还进行了上下文长度扩展（2K→8K），使用 RoPE 值130K，混合40%长文档数据。

### 后训练

- **监督微调（SFT）**：在 SmolTalk 上训练2个epoch，包含 MagPie-Ultra（100万样本三轮对话）、Smol-Constraint（3.6万条约束指令）、Smol-Summarization（100万条摘要）、Smol-Rewrite（60万条改写）、NuminaMath-CoT、MetaMathQA、Self-OSS-Starcoder2-Instruct 等。
- **偏好对齐（DPO）**：使用 UltraFeedback 数据集进行 Direct Preference Optimization，beta=0.5，训练2个epoch。

## 实验结果

### 基座模型对比

| 基准测试 | SmolLM2-1.7B | Llama3.2-1B | Qwen2.5-1.5B |
|----------|:---:|:---:|:---:|
| HellaSwag | **68.7** | 61.2 | 66.4 |
| ARC | **60.5** | 49.2 | 58.5 |
| PIQA | **77.6** | 74.8 | 76.1 |
| CommonsenseQA | **43.6** | 41.2 | 34.1 |
| MMLU-Pro（未监控） | **19.4** | 11.7 | 13.7 |
| TriviaQA（未监控） | **36.7** | 28.1 | 20.9 |
| GSM8K (5-shot) | 31.1 | 7.6 | **61.7** |
| MATH (4-shot) | 11.6 | 3.3 | **34.3** |
| HumanEval | 22.6 | 18.9 | **37.2** |

### 指令微调模型对比

| 基准测试 | SmolLM2-1.7B | Llama3.2-1B | Qwen2.5-1.5B |
|----------|:---:|:---:|:---:|
| IFEval | **56.7** | 53.5 | 47.4 |
| MT-Bench | 6.13 | 5.48 | **6.52** |
| HellaSwag | **66.1** | 56.1 | 60.9 |
| GSM8K (5-shot) | 48.8 | 37.4 | **63.3** |
| MATH (4-shot) | **21.0** | 19.5 | 19.6 |

SmolLM2 在知识、推理和指令遵循方面全面领先，在 MMLU-Pro 上超出 Qwen2.5-1.5B 近6个百分点。数学方面虽然基座模型落后于 Qwen2.5，但指令微调后 MATH 得分反超。

## 局限性

1. **数学和代码能力仍有差距**：基座模型在 GSM8K 和 MATH 上显著落后于 Qwen2.5-1.5B（31.1 vs 61.7，11.6 vs 34.3），说明17亿参数模型在数学推理上的容量限制。
2. **训练过程中出现loss spike**：阶段3出现不明原因的损失尖峰，即使回退训练并跳过相关数据也无法消除。
3. **长上下文能力有限**：在 HELMET 基准的多项指标上（Recall、ICL、Re-rank）落后于 Qwen2.5-1.5B 和 Llama3.2-1B。
4. **多阶段训练策略依赖人工判断**：数据混合比例的调整基于经验观察而非系统化搜索，可能未达到全局最优。
5. **仅支持英文**：数据集和评估均聚焦英文，多语言能力未被探索。

## 关键洞察

1. **数据质量对小模型至关重要**：小模型容量有限，必须精心优化训练数据以学习核心知识而非记忆无关事实。FineMath 通过聚焦逐步推理内容实现了数学能力的数倍提升。
2. **分类器过滤是通用且有效的数据策展方法**：无论是网页数据（FineWeb-Edu）、数学数据（FineMath）还是代码数据（Stack-Edu），基于 LLM 标注训练分类器进行质量过滤都显著提升了下游性能。
3. **多阶段训练优于固定混合**：将高质量专业数据（FineMath、Stack-Edu）保留到训练后期引入，可以最大化其影响力，避免被大规模网页数据稀释。
4. **过度训练小模型是值得的**：SmolLM2 在11万亿token上训练（远超 Chinchilla 最优），虽然偏离理论最优，但推理成本的降低使其成为合理的工程权衡。
5. **小模型经过充分训练可以获得"涌现"能力**：在6万亿token训练后，SmolLM2 在 MMLU 多选格式上超过随机水平，这一能力此前被认为仅存在于更大模型中。
6. **互补数据源的混合优于单一来源**：FineWeb-Edu 擅长教育内容，DCLM 擅长常识推理，两者混合实现了最佳平衡。
