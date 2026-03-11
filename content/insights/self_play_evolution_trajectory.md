---
tags:
  - insight
time: 2026-03-05T18:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_star_bootstrapping_reasoning.md
  - ../knowledge/summary_rest_reinforced_self_training.md
  - ../knowledge/summary_spin_self_play_finetuning.md
  - ../knowledge/summary_language_self_play_data_free.md
  - ../knowledge/summary_multiagent_finetuning.md
  - ../knowledge/summary_r_zero_self_evolving.md
  - ../knowledge/summary_absolute_zero_reasoning.md
  - ../knowledge/summary_r_few_guided_self_evolving.md
  - ../knowledge/summary_spice_self_play_corpus.md
  - ../knowledge/summary_can_reasoning_models_self_train.md
  - ../knowledge/summary_constitutional_ai.md
  - ../knowledge/summary_self_rewarding_language_models.md
---

# Self-Play 演化轨迹：从自举到环境驱动的自主学习

回顾这十篇工作，可以清晰地看到 self-play 在 LLM 微调中的演化脉络。STaR (NeurIPS 2022) 开创了"用模型自己生成的正确推理来训练自己"的范式，核心机制极其简洁——生成推理链、过滤正确答案、微调模型、重复。ReST (2023) 引入奖励模型替代二值过滤，将适用范围从"有标准答案的推理任务"扩展到"开放生成任务"。SPIN (ICML 2024) 将博弈论正式引入，让新旧模型形成对抗，有漂亮的理论保证但天花板明确。LSP (2025) 彻底打破数据依赖，通过 Challenger-Solver 极小极大博弈实现完全无数据训练。Multiagent FT 通过多智能体角色专业化解决单模型自我提升中的多样性衰减问题。

**零数据自进化的三次尝试**：Group 2 论文代表了"完全零数据自进化"的三次尝试，每次都揭示了新的失败模式和解决方案。R-Zero (ICLR 2026) 首次在通用推理领域实现零数据训练，通过独立共同进化的 Challenger-Solver 架构，但伪标签质量系统性下降导致第 3 轮后崩溃。SRT (ICLR 2026) 研究了大型推理模型的自训练能力，用多数投票作为自奖励信号，短期内接近有真实标签的 RLVR 性能，但延长训练后所有模型都出现突然且完全的性能崩溃——这是奖励黑客的终极形态。Absolute Zero (2025) 在代码推理领域实现了真正的零数据成功，关键在于用 Python 执行器作为可验证环境，避免了神经奖励模型的不可靠性。

**最小监督的两个方向**：面对零数据方法的失败，研究者探索了"最小监督"的两个方向。R-Few 用 1-5% 人类数据作为锚点，通过 few-shot 示例软引导，成功防止了概念漂移和多样性崩溃，用 5% 数据达到 20 倍数据量方法的性能。SPICE 用大规模文档语料库作为环境，Challenger 从文档中挖掘内容生成问题，打破了纯自博弈的信息对称性，实现了 640 轮稳定训练。这两个方向揭示了"最小监督"的两种形式：语义锚点（人类数据）vs 知识环境（文档语料库）。

这条演化路径揭示了两个清晰的趋势：

1. **训练信号来源的演进**：从外部标注数据逐步转向环境交互。STaR 依赖标准答案，ReST 依赖奖励模型，SPIN 依赖 SFT 数据分布，LSP 依赖博弈论均衡，R-Zero 依赖伪标签，Absolute Zero 和 SPICE 依赖外部环境（代码执行器、文档语料库）。最后一步是质的飞跃——从"模型内部的自组织"转向"模型与环境的交互"。

2. **评估能力的内化**：从外部人类评判（RLHF）到自我批评（Constitutional AI）再到自我奖励（Self-Rewarding LM）。Constitutional AI 让模型学会批评和修订自己的输出，但奖励模型仍然冻结。Self-Rewarding LM 打破了这个瓶颈，让奖励模型随训练迭代持续改进，实现了"评判能力"与"生成能力"的协同进化。

## Evidence

**基础自举方法 (2022-2024)**：
- [STaR](../knowledge/summary_star_bootstrapping_reasoning.md): 开创 rationale generation + rationalization + 迭代训练，6B 模型达到 180B 模型水平
- [ReST](../knowledge/summary_rest_reinforced_self_training.md): 引入奖励模型和递增阈值过滤，Grow-Improve 解耦使得 BC loss + 过滤优于复杂的 offline RL loss
- [SPIN](../knowledge/summary_spin_self_play_finetuning.md): 将自博弈形式化为 IPM 框架，仅用 SFT 数据即超越使用 62k GPT-4 偏好数据的 DPO
- [LSP](../knowledge/summary_language_self_play_data_free.md): 完全无数据训练，Challenger 自动生成由易到难的课程，接近有数据 GRPO 的性能
- [Multiagent FT](../knowledge/summary_multiagent_finetuning.md): 通过角色专业化实现 5+ 轮持续提升，MATH 准确率从 42.4% 提升到 66.0%

**零数据自进化的三次尝试 (2025-2026)**：
- [R-Zero](../knowledge/summary_r_zero_self_evolving.md): 独立共同进化架构，数学 +6.49，通用 +7.54，但伪标签准确率从 79% 降至 47%，第 3 轮后崩溃
- [SRT](../knowledge/summary_can_reasoning_models_self_train.md): 多数投票自奖励，短期接近 RLVR，但延长训练后所有 4 个模型突然完全崩溃
- [Absolute Zero](../knowledge/summary_absolute_zero_reasoning.md): Python 执行器验证，代码 +5.0，数学 +15.2，超越使用 57k-484k 数据的方法，14B 模型 500 步后仍在提升

**最小监督的两个方向 (2025)**：
- [R-Few](../knowledge/summary_r_few_guided_self_evolving.md): 1-5% 人类锚点，用 5% 数据达到 20 倍数据量方法的性能，多样性全程稳定
- [SPICE](../knowledge/summary_spice_self_play_corpus.md): 20,000 文档语料库环境，相比无语料库方法 +7.9%，稳定训练 640 轮

**自我评估与反馈的演进 (2022-2024)**：
- [Constitutional AI](../knowledge/summary_constitutional_ai.md): 模型自我批评和修订有害回复，用 AI 反馈替代人类无害性标注，但奖励模型仍然冻结
- [Self-Rewarding LM](../knowledge/summary_self_rewarding_language_models.md): 奖励模型不再冻结，模型同时提升指令遵循和评判能力，3 次迭代后超越 Claude 2 和 GPT-4 0613

## Implications

这条演化轨迹指向一个明确的未来方向：**环境驱动的自主学习**。Group 2 论文的核心教训是：纯粹的自我封闭循环（R-Zero、SRT）最终会崩溃，而引入外部环境（Absolute Zero 的代码执行器、SPICE 的文档语料库、R-Few 的人类锚点）可以实现稳定的长期提升。

这与用户笔记中的核心愿景完美契合："让环境介入地进行数据合成"、"独立发现问题、独立解决问题、独立使用工具验证答案"。Absolute Zero 已经实现了这个完整循环——模型提出代码任务（发现问题）、编写代码解决（解决问题）、Python 执行器验证（工具验证）。SPICE 展示了如何将这个范式扩展到更广泛的领域——用文档语料库替代代码执行器。

下一步的关键突破点：
1. **多环境融合**：结合 Absolute Zero 的可验证环境 + SPICE 的语料库环境 + R-Few 的人类锚点
2. **环境发现**：不仅在给定环境中学习，还要学会发现和构建新的验证环境
3. **跨域迁移**：在代码环境中学到的"提出-解决-验证"循环能否迁移到其他领域（形式化证明、物理模拟、真实世界交互）

LSP 的 Challenger 本质上就是用户所说的"problem synthesizer"——学会了"什么问题才是好问题"。将这个能力与 Absolute Zero 的环境验证结合，就能实现真正的自主学习。
