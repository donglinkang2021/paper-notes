---
title: "Nemotron-4 340B Technical Report"
authors: "Jupinder Parmar, Shrimai Prabhumoye, Joseph Jennings, Deepak Narayanan, Mostofa Patwary, Dan Su, Shengyang Sun, Jiaqi Zeng, et al."
institution: "Nvidia"
venue: "arXiv 2024"
arxiv_id: "2406.11704"
tags: [LLM, alignment, synthetic-data, reward-model, RLHF, DPO, weak-to-strong, 340B]
---

# Nemotron-4 340B: 基于合成数据与迭代弱到强对齐的大规模语言模型

## 核心贡献

1. 发布了 Nemotron-4 340B 模型家族，包含三个模型：Base（基座模型）、Instruct（指令模型）和 Reward（奖励模型），均以宽松的 NVIDIA Open Model License 开源，支持商业应用。
2. 提出了"迭代弱到强对齐"（Iterative Weak-to-Strong Alignment）方法，通过合成数据生成与模型对齐的交替迭代，实现模型能力的持续提升。整个对齐过程中超过 98% 的数据为合成生成，仅使用约 20K 人工标注数据。
3. 提出了 Reward-aware Preference Optimization（RPO）算法，利用奖励模型提供的细粒度奖励差值信息来优化偏好学习，缓解 DPO 的过拟合问题。
4. 训练的奖励模型 Nemotron-4-340B-Reward 在发布时取得了 RewardBench 排行榜最高准确率（92.0%），超越 GPT-4o、Gemini 1.5 Pro 等闭源模型。

## 方法详解

### 预训练

- 架构：标准 decoder-only Transformer，96 层，隐藏维度 18432，96 个注意力头，8 个 KV 头（GQA），序列长度 4096，词表大小 256K。总参数量约 340B（9.4B 嵌入参数 + 331.6B 非嵌入参数）。
- 使用 RoPE 位置编码、SentencePiece 分词器、Squared ReLU 激活函数，无 bias，无 dropout。
- 训练数据：9T tokens，包含英文自然语言（70%）、多语言（15%，53 种语言）和代码（15%，43 种编程语言）。前 8T 为正式预训练，后 1T 为继续训练（调整数据分布，提高高质量数据权重）。
- 训练基础设施：768 个 DGX H100 节点（6144 GPU），使用 8 路张量并行 + 12 路流水线并行 + 数据并行，MFU 约 41-42%。

### 奖励模型

- 基于 Base 模型，将最后的 softmax 层替换为线性投影头，映射到 HelpSteer 的 5 个属性维度（Helpfulness、Correctness、Coherence、Complexity、Verbosity）。
- 使用多属性回归方式训练（而非成对排序），能更好地区分真正的有用性与长度偏好等无关因素。
- 仅使用 10K 人工标注的 HelpSteer2 数据训练，即在 RewardBench 上达到 SOTA（整体 92.0%，Chat-Hard 类别 87.1%）。

### 合成数据生成流水线

- Prompt 生成：使用 Mixtral-8x7B-Instruct 生成多样化的合成 prompt，覆盖 open Q&A、writing、closed Q&A、math & coding 四大类任务，以及 instruction-following 和 two-turn 对话 prompt。通过层级式主题生成（宏观主题 -> 子主题）确保多样性。
- 对话生成：使用指令模型生成 3 轮对话，交替扮演 Assistant 和 User 角色，并用奖励模型进行质量过滤。
- 偏好数据生成：对每个 prompt 使用多个中间模型生成多个回复，通过三种方式判断偏好排序：
  - Ground-Truth-as-Judge：对有标准答案的任务（如 GSM8K、MATH）直接验证正确性。
  - LLM-as-Judge：让 LLM 比较两个回复（交换顺序避免位置偏差）。
  - Reward-Model-as-Judge：使用奖励模型预测每个回复的奖励值进行排序。实验表明 Reward-Model-as-Judge 在 Chat-Hard 类别上远优于 LLM-as-Judge（87% vs 54%）。

### 迭代弱到强对齐

核心思想：弱模型生成的合成数据可以训练出更强的模型，更强的模型又能生成更高质量的数据，形成自增强的飞轮效应。

- 第 1 轮：用 Mixtral-8x7B-Instruct（弱模型）生成数据 -> 对齐 340B 中间检查点 -> 得到 340B-Interm-1-Instruct，已超越 Mixtral-8x7B-Instruct。
- 第 2 轮：用 340B-Interm-1-Instruct 生成更高质量数据 -> 对齐更强的基座模型 -> 得到 340B-Interm-2-Instruct。
- 持续迭代多轮，模型能力不断提升。

改进来源于两个方面：(1) 更强的基座模型产生更强的指令模型；(2) 更高质量的数据产生更强的指令模型。

### 对齐算法

对齐分为多个阶段：

1. Code SFT：先在约 800K 合成代码样本上微调（使用 Genetic Instruct 方法生成），提升编码能力。
2. General SFT：在 200K 混合任务样本上微调（包含 2% 代码数据防止遗忘），提升通用能力。
3. DPO：使用 160K 偏好数据训练，额外加入 chosen response 的 SFT loss 防止策略偏移过大。
4. RPO（3 轮迭代）：使用 300K 偏好数据，每轮以上一轮检查点为初始化和参考策略。RPO 的核心改进是利用奖励模型给出的奖励差值作为学习目标，而非 DPO 的二元偏好信号，避免对高质量 rejected response 的过度"遗忘"。

RPO 损失函数：

$$\mathcal{L}_{rpo} = \mathbb{D}\left[\beta \log\frac{\pi(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \| \eta(r^*(x,y_c) - r^*(x,y_l))\right]$$

其中 $\mathbb{D}$ 使用 KL 散度的 sigmoid 形式。

## 实验结果

### 基座模型

| 模型 | ARC-c | Winogrande | Hellaswag | MMLU | BBH | HumanEval |
|------|-------|------------|-----------|------|-----|-----------|
| Nemotron-4-340B-Base | **94.28** | **89.50** | **90.53** | 81.10 | **85.44** | 57.32 |
| Llama-3 70B | 93.00 | 85.30 | 88.00 | 79.50 | 81.30 | 48.20 |
| Qwen-2 72B | 68.90 | 85.10 | 87.60 | **84.20** | 82.40 | **64.60** |

在常识推理和 BBH 上表现最优，MMLU 和代码上具有竞争力。

### 指令模型

- Arena Hard: 54.2（开源最优，超越 Llama-3-70B-Instruct 的 41.1）
- MT-Bench (GPT-4-Turbo): 8.22（开源中与 Qwen-2-72B-Instruct 的 8.26 接近）
- IFEval Prompt-Strict-Acc: 79.9（开源最优）
- GSM8K (0-shot): 92.3（开源最优）
- 人类评估：与 GPT-4-1106-preview 对比，胜率 28.19%，平局 46.57%，负率 25.24%，整体表现可比。

### 奖励模型

在 RewardBench 上整体准确率 92.0%，Chat-Hard 类别 87.1%，均为发布时最高，超越所有闭源模型。

### 对齐阶段消融

各阶段的贡献清晰可见：Code SFT 大幅提升 HumanEval（57.3 -> 70.7），General SFT 提升 MT-Bench 和 MMLU，DPO 进一步提升多数指标，RPO 三轮迭代均匀提升所有指标（MT-Bench 7.90 -> 8.22，IFEval 61.7 -> 79.9）。

## 局限性

1. 模型规模巨大（340B），虽然可在单个 DGX H100（8 GPU）上以 FP8 部署，但推理成本仍然较高。
2. 安全性方面存在已知问题：对恶意软件生成请求的拦截不完全，面对对抗性幻觉（如错误否认质数）表现不佳，对越狱攻击的防御通过率低于 30%。
3. 合成数据生成流水线依赖于初始种子模型的质量和多样性，可能存在分布偏差。
4. 奖励模型在 RewardBench 的 Prior Sets 上表现相对较低（67.4%），可能因为未使用这些数据集的训练数据。
5. 安全评估中的 AEGIS 安全模型本身存在假阳性和假阴性错误，缺乏人工标注的 ground truth 来量化误判率。

## 关键洞察

1. 合成数据的有效性已被充分验证：仅用约 20K 人工标注数据 + 98% 合成数据即可训练出与 GPT-4 可比的指令模型，这大幅降低了对齐的人工标注成本。
2. 弱模型可以训练强模型：Mixtral-8x7B（远小于 340B）生成的数据足以让 340B 模型超越 Mixtral 本身，说明数据生成器不构成学生模型的能力上限。
3. 多属性回归奖励模型优于成对排序模型：能更好地捕捉细粒度的质量差异，避免长度偏好等伪相关。
4. 分阶段 SFT 优于混合训练：先 Code SFT 再 General SFT 的策略避免了多任务学习中的冲突，尤其在代码任务上效果显著。
5. RPO 通过引入奖励差值信息解决了 DPO 的过拟合问题：DPO 中 chosen 和 rejected 的似然度同时下降，RPO 通过学习近似奖励差值而非最大化隐式奖励差距来缓解这一问题。
6. Reward-Model-as-Judge 在困难偏好判断上远优于 LLM-as-Judge（Chat-Hard: 87% vs 54%），是合成偏好数据生成的更优选择。
