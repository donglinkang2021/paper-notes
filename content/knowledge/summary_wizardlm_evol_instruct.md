---
title: "WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions"
authors: "Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang"
institution: "Microsoft, Peking University"
venue: "ICLR 2024"
arxiv_id: "2304.12244"
tags: ["instruction-tuning", "synthetic-data", "evol-instruct", "LLM", "data-augmentation", "instruction-following"]
---

# WizardLM：通过进化指令赋能大型预训练语言模型遵循复杂指令

## 核心贡献

1. 提出 Evol-Instruct 方法，利用 LLM 自动将简单指令逐步进化为更复杂的指令，替代昂贵的人工标注过程
2. 训练得到的 WizardLM 模型在多个基准测试上显著超越 Alpaca 和 Vicuna 等同规模开源模型
3. 初步验证了指令复杂度对监督微调性能的重要性——随着训练指令复杂度的提升，模型性能同步提高

## 方法详解

Evol-Instruct 的核心流程包含两个组件：指令进化器（Instruction Evolver）和指令淘汰器（Instruction Eliminator）。

**指令进化器**包含两类进化策略：

- **深度进化（In-Depth Evolving）**：通过 5 种操作增加指令复杂度
  - 添加约束条件（Add Constraints）
  - 深化（Deepening）
  - 具体化（Concretizing）
  - 增加推理步骤（Increase Reasoning Steps）
  - 复杂化输入（Complicate Input）
- **广度进化（In-Breadth Evolving）**：基于给定指令生成一个全新的、更长尾的指令，以增加主题和技能的多样性

关键设计细节：
- 每次进化仅增加"一点点"难度（限制每次仅增加 10-20 个词），避免指令集被极端复杂的指令填满
- 要求进化后的指令必须合理且人类可理解，防止 AI 随意生成不切实际的指令

**指令淘汰器**过滤四类失败进化：(1) 无信息增益；(2) LLM 无法生成有效回复（回复含"sorry"且少于 80 词）；(3) 回复仅含标点和停用词；(4) 进化指令明显复制了提示词中的模板文本。

**数据构建流程**：以 Alpaca 的 52k 指令为种子，执行 4 轮进化（每轮对每条指令随机选择 6 种进化操作之一），使用 ChatGPT API 共调用约 624k 次，最终获得 250k 条指令。为公平比较，从中随机采样 70k 条用于微调 LLaMA 13B。

## 实验结果

**自动评测**（9 个基准测试）：WizardLM-13b 在大多数基准上领先同规模开源模型：
- 平均分 58.96，超过 Vicuna-13b（54.60）、Alpaca-13b（43.44）
- 在代码（HumanEval: 24.0 vs 12.5）、数学（GSM8k: 37.15 vs 24.34）方面优势尤为明显
- AlpacaEval: 75.31, MT-Bench: 6.35, WizardEval: 89.1

**人工评测**（WizardEval 测试集，218 条真实指令，覆盖 29 种技能）：WizardLM 在与 Alpaca 和 Vicuna 的盲评对比中显著胜出，标注者间一致性 Kappa > 0.6。

**消融实验关键发现**：
- 使用 ShareGPT 作为种子数据效果更好（平均分 61.87 vs 58.96）
- 更大的进化数据量（250k）可进一步提升性能（60.30）
- Evol-Instruct 不依赖 ChatGPT，LLaMA-2-70B-Chat 也可作为进化执行器
- 方法可泛化到不同基座模型：WizardLM-70b 达到 71.33 平均分，WizardLM-7b (Mistral) 达到 65.81
- 深度进化分析：随着进化轮次增加，指令难度和模型性能同步提升
- 广度进化分析：t-SNE 聚类显示进化后的指令比 ShareGPT 和 Alpaca 分布更均匀，主题多样性更高

## 局限性

- 在代码、数学和复杂推理等高难度场景上，WizardLM 与 ChatGPT 仍有明显差距
- GPT-4 自动评测与人工评测在高难度指令上存在不一致性（人工偏好格式整洁和可编译的代码）
- 进化过程依赖 LLM 的能力上限，进化质量受限于执行进化的模型本身
- 人类创建的指令难度分布偏向简单和中等，而 Evol-Instruct 虽然缓解了这一问题，但极高难度指令的质量仍难以保证
- 训练成本：8 张 V100 GPU 训练 140 小时，加上大量 API 调用费用

## 关键洞察

- **指令复杂度是关键驱动力**：实验清晰表明，训练数据的复杂度比数量更重要。从 C0 到 C4 每轮进化数据约 52k，但随着复杂度递增，模型性能持续提升。
- **AI 生成的指令可以超越人工指令**：Evol-Instruct 从几乎无人工参与的 Alpaca 种子数据出发，生成的指令在难度和多样性上均超过人类创建的 ShareGPT 数据。
- **渐进式进化优于一步到位**：每次仅增加"一点点"难度的设计至关重要，避免了极端复杂指令对模型泛化性能的损害。
- **方法的通用性强**：Evol-Instruct 不绑定特定种子数据、进化模型或基座模型，具有良好的可迁移性。这为合成数据生成提供了一个可扩展的范式。
