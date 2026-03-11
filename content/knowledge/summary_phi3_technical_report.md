---
title: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"
authors: "Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, ..., Xiren Zhou (Microsoft, 按字母排序共约100位作者)"
institution: "Microsoft Research"
venue: "arXiv 2024"
arxiv_id: "2404.14219"
tags: [small language model, data quality, phi-3, on-device inference, MoE, multimodal, long context, synthetic data]
---

# Phi-3 技术报告：可在手机上本地运行的高性能语言模型

## 核心贡献

1. 提出 phi-3-mini（3.8B 参数），在 MMLU 上达到 69%、MT-bench 上达到 8.38，性能媲美 Mixtral 8x7B（45B 参数）和 GPT-3.5，同时小到可以在手机上本地运行（4-bit 量化后仅占约 1.8GB 内存，在 iPhone 14 上实现每秒 12+ token 的推理速度）。
2. 提出"数据最优范式"（Data Optimal Regime）：不同于传统的"计算最优范式"，针对给定模型规模精心筛选训练数据质量，使小模型在推理能力上大幅超越同规模模型。
3. 扩展至 phi-3-small（7B）和 phi-3-medium（14B），分别在 MMLU 上达到 75.7% 和 78.0%。
4. 推出 phi-3.5 系列：phi-3.5-mini（增强多语言和 128K 长上下文）、phi-3.5-MoE（16x3.8B MoE，6.6B 活跃参数，性能媲美 Gemini-1.5-Flash）、phi-3.5-Vision（4.2B 多模态模型）。

## 方法详解

### 模型架构

- **phi-3-mini**：Transformer decoder 架构，与 Llama-2 相同的块结构和 tokenizer（词表大小 32064）。隐藏维度 3072，32 头，32 层，默认上下文长度 4K，通过 LongRope 扩展至 128K。
- **phi-3-small（7B）**：使用 tiktoken tokenizer（词表 100352），GEGLU 激活函数，采用 muP（Maximal Update Parametrization）从小代理模型迁移超参数。创新性地引入块稀疏注意力（blocksparse attention），交替使用密集注意力层和块稀疏注意力层，在保持长上下文检索性能的同时优化 KV cache。分组查询注意力（4 个 query 共享 1 个 key）。
- **phi-3-medium（14B）**：与 phi-3-mini 相同的 tokenizer 和架构，40 头 40 层，嵌入维度 5120。
- **phi-3.5-MoE**：16 个专家网络的 MoE 架构，每个 token 激活 top-2 专家（每个专家为独立的 GLU 网络），总参数 42B，活跃参数 6.6B。使用 SparseMixer 方法训练稀疏路由器。
- **phi-3.5-Vision（4.2B）**：CLIP ViT-L/14 图像编码器 + phi-3.5-mini 文本解码器，采用动态裁剪策略处理高分辨率和不同宽高比的图像。

### 训练数据策略

训练数据是本文的核心创新，延续"Textbooks Are All You Need"系列工作：

- **数据来源**：经过严格过滤的公开网络数据 + LLM 生成的合成数据。
- **两阶段预训练**：
  - 阶段一：以网络数据为主，教模型通用知识和语言理解。
  - 阶段二：更严格过滤的网络数据子集 + 合成数据，教模型逻辑推理和各种专项技能。
- **数据最优范式**：针对小模型的容量限制，过滤掉纯事实性信息（如某天的英超比赛结果），保留更多能提升推理能力的网页内容，将有限的模型容量留给"推理"而非"记忆"。
- phi-3-mini 训练 3.3T token，phi-3-small 和 phi-3-medium 训练 4.8T token。

### 后训练

- **SFT（监督微调）**：使用跨数学、编程、推理、对话、模型身份和安全等多领域的高质量数据。
- **DPO（直接偏好优化）**：覆盖聊天格式、推理和负责任 AI（RAI），将不良输出作为"拒绝"响应来引导模型行为。

### 多语言与长上下文

- 在中间训练阶段加入更多多语言和长文本数据。
- 使用 LongRope + 混合上下文窗口方法将上下文从 4K 扩展到 128K，且不损害 4K 任务性能。
- phi-3.5-mini 多语言 MMLU 平均分从 phi-3-mini 的 47.3 提升至 55.4，phi-3.5-MoE 达到 69.9。

## 实验结果

### 语言基准测试（phi-3 系列 vs 同规模模型）

| 模型 | 参数量 | MMLU | GSM-8K | HumanEval | MT-bench |
|------|--------|------|--------|-----------|----------|
| phi-3-mini | 3.8B | 68.8 | 82.5 | 58.5 | 8.38 |
| phi-3-small | 7B | 75.7 | 89.6 | 61.0 | 8.70 |
| phi-3-medium | 14B | 78.0 | 91.0 | 62.2 | 8.91 |
| Llama-3-Instruct-8B | 8B | 66.5 | 77.4 | 60.4 | - |
| Mixtral 8x7B | 45B | 70.5 | 64.7 | 37.8 | - |
| GPT-3.5 | - | 71.4 | 78.1 | 62.2 | 8.35 |

### phi-3.5 系列对比

- phi-3.5-MoE 在 18 项基准测试中平均得分 69.2，超越 Llama-3.1-8B（61.0）和 Mistral-Nemo-12B（61.3），接近 Gemini-1.5-Flash（68.5），达到 GPT-4o-mini（74.9）约 92% 的性能。
- 长上下文：phi-3.5-MoE 在 RepoQA 上得分 85（超越 Llama-3.1-8B 的 71），RULER 上得分 87.1。
- phi-3.5-Vision 在单图理解基准上全面超越同规模开源模型（如 LLaVA-1.6、Qwen-VL-Chat），在 ChartQA 上甚至超过 GPT-4O（81.8 vs 64.0）。

### 安全性

- 经过安全对齐后，有害响应率显著下降。
- 在内部多轮对话 RAI 基准上，phi-3 系列在越狱抵抗、有害内容延续等指标上优于 Mistral-7B 和 Gemma-7B。

## 局限性

1. **事实知识存储有限**：模型容量限制导致在 TriviaQA 等知识密集型任务上表现较弱（phi-3-mini 仅 64.0，远低于 GPT-3.5 的 85.8），但可通过搜索引擎增强来弥补。
2. **语言覆盖不足**：phi-3-mini 主要限于英语，多语言能力在 phi-3.5 系列中有所改善但仍有差距。
3. **长上下文性能衰减**：在 RULER 基准的 128K 上下文测试中性能显著下降（phi-3.5-MoE 从 4K 的 94.8 降至 128K 的 64.2），疑因中间训练阶段缺乏高质量长上下文数据。
4. **幻觉与偏见**：与所有 LLM 一样，仍存在事实不准确、偏见放大和不当内容生成的问题。
5. **7B 到 14B 的收益递减**：部分基准从 7B 到 14B 的提升远小于 3.8B 到 7B，表明数据配方在更大规模上可能需要进一步优化。
6. **Vision 模型的高级推理不足**：phi-3.5-Vision 在需要高级推理的问题上表现有限，偶尔生成无依据的输出。

## 关键洞察

1. **数据质量 > 模型规模**：Phi-3 系列最核心的洞察是，通过精心策划的训练数据（过滤的网络数据 + 合成数据），3.8B 参数的小模型可以达到 45B 参数模型的性能水平，打破了传统 scaling law 的假设。
2. **"数据最优范式"的实践意义**：对于小模型，应该主动放弃存储事实知识，将有限容量集中在推理能力上——这是一种与大模型截然不同的训练哲学。
3. **端侧部署的可行性**：4-bit 量化后 1.8GB 的内存占用和 12+ token/s 的速度，证明了高质量小模型在移动设备上的实用价值。
4. **MoE 架构的效率优势**：phi-3.5-MoE 以 6.6B 活跃参数实现了接近 GPT-4o-mini 的性能，展示了稀疏激活在效率-性能权衡上的巨大潜力。
5. **搜索增强是小模型的天然互补**：论文明确指出小模型的知识存储瓶颈可通过外部搜索引擎解决，这为 RAG + 小模型的端侧应用范式提供了理论支撑。
