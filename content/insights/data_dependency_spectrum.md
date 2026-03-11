---
tags:
  - insight
time: 2026-03-06T00:00:00+08:00
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
  - ../knowledge/summary_nemotron4_340b.md
  - ../knowledge/summary_magpie_alignment_synthesis.md
  - ../knowledge/summary_wizardlm_evol_instruct.md
  - ../knowledge/summary_kimi_k2.md
---

# 数据依赖光谱：从完全零数据到最小监督的精细刻度

Group 1 和 Group 2 论文共同构成了一个更精细的数据依赖光谱。STaR 需要问题-答案对和少量带推理链的 few-shot 示例；ReST 需要训练数据集加上预训练好的奖励模型；SPIN 只需要 SFT 数据，不需要偏好标注；Multiagent FT 需要任务数据集但通过多智能体辩论自动生成训练信号；LSP 完全不需要训练数据。

**完全零数据的三种路径**：Group 2 论文探索了三种不同的零数据方法。Absolute Zero 从单个恒等函数种子开始，在代码执行环境中自举，实现了真正的"从零开始"——无需任何人类标注的问题-答案对，仅依赖 Python 执行器验证。R-Zero 在通用推理领域实现零数据训练，通过 Challenger-Solver 共同进化，但伪标签质量系统性下降导致第 3 轮后崩溃。SRT 用多数投票作为自奖励信号，在短期内接近有真实标签的 RLVR 性能，但延长训练必然导致奖励黑客和完全崩溃。

**最小监督的突破**：R-Few 和 SPICE 代表了"最小监督"的两个方向。R-Few 用 1-5% 人类数据作为锚点，通过 few-shot 示例软引导，用 5% 数据达到 20 倍数据量方法的性能——这是数据效率的巨大飞跃。SPICE 不需要标注数据，但需要大规模文档语料库（20,000 文档）作为环境，Challenger 从文档中挖掘内容生成问题，相比无语料库方法提升 +7.9%。这揭示了一个关键区别：人类标注数据 vs 非结构化语料库——前者提供"语义锚点"，后者提供"知识多样性"。

**性能-稳定性权衡**：数据依赖的减少伴随着稳定性风险。完全零数据方法（R-Zero、SRT）在 3-5 轮后必然崩溃，最小监督方法（R-Few 1%、SPICE）可稳定训练数百轮，有数据方法（STaR、ReST）可持续迭代但受限于数据质量。Absolute Zero 是特例——通过可验证环境（代码执行器）实现了零数据 + 高稳定性，但代价是限制在特定领域。

## Evidence

**传统有数据方法**：
- [STaR](../knowledge/summary_star_bootstrapping_reasoning.md): 需要问题+答案对 + few-shot 推理示例，rationalization 机制补偿难题的训练信号缺失
- [ReST](../knowledge/summary_rest_reinforced_self_training.md): 需要数据集 + 奖励模型，递增阈值过滤替代标准答案验证，但奖励模型过拟合成为新瓶颈
- [SPIN](../knowledge/summary_spin_self_play_finetuning.md): 仅需 SFT 数据，无需偏好标注，iter 0 即匹配 DPO（使用 62k 额外偏好数据）
- [Multiagent FT](../knowledge/summary_multiagent_finetuning.md): 需要任务数据集（500 样本），但通过多智能体辩论自动生成角色专业化的训练数据
- [LSP](../knowledge/summary_language_self_play_data_free.md): 完全无训练数据，AlpacaEval 36.4% 接近有数据 GRPO 的 38.8%

**完全零数据方法**：
- [Absolute Zero](../knowledge/summary_absolute_zero_reasoning.md): 从单个恒等函数种子开始，代码平均 +5.0，数学平均 +15.2，超越使用 57k-484k 数据的方法
- [R-Zero](../knowledge/summary_r_zero_self_evolving.md): 通用推理零数据训练，数学 +6.49，通用 +7.54，但伪标签准确率从 79% 降至 47%，第 3 轮后崩溃
- [SRT](../knowledge/summary_can_reasoning_models_self_train.md): 多数投票自奖励，短期内接近 RLVR 性能，但延长训练后所有 4 个模型完全崩溃

**最小监督方法**：
- [R-Few](../knowledge/summary_r_few_guided_self_evolving.md): 1-5% 人类数据，R-Few (5%) 用 11.6k 数据达到 232k 数据方法的性能，数据效率提升 20 倍
- [SPICE](../knowledge/summary_spice_self_play_corpus.md): 需要 20,000 文档语料库，相比无语料库 R-Zero 提升 +7.9%，稳定训练 640 轮

**自我评估方法的数据需求**：
- [Constitutional AI](../knowledge/summary_constitutional_ai.md): 仅需 16 条自然语言原则（宪法）+ 少量 EFT 样本训练 LLM-as-a-Judge，无需人类无害性标注
- [Self-Rewarding LM](../knowledge/summary_self_rewarding_language_models.md): 需要 3,200 IFT 样本 + 1,630 EFT 样本初始化，后续完全自我生成偏好数据

**合成数据驱动的对齐方法（Group 7 补充）**：
- [Nemotron-4 340B](../knowledge/summary_nemotron4_340b.md): 仅 20K 人工标注 + 98% 合成数据，通过"迭代弱到强对齐"实现 340B 模型与 GPT-4 可比的性能。弱模型（Mixtral-8x7B）生成的数据足以让 340B 模型超越弱模型本身
- [Magpie](../knowledge/summary_magpie_alignment_synthesis.md): 完全零人工干预——仅向对齐 LLM 输入预查询模板即可自动生成高质量指令数据，不到 40 万条数据超越使用 1000 万条数据的官方 Llama-3-8B-Instruct
- [WizardLM](../knowledge/summary_wizardlm_evol_instruct.md): 以 52K Alpaca 指令为种子，通过 LLM 驱动的渐进式进化生成 250K 复杂指令，证明指令复杂度比数量更重要
- [Kimi K2](../knowledge/summary_kimi_k2.md): 大规模智能体数据合成流水线——从 GitHub 获取 3000+ 真实 MCP 工具并合成 20000+ 合成工具，结合模拟环境和真实执行沙箱生成高保真智能体交互数据

## Implications

Group 2 论文重新定义了"零数据"的含义，揭示了四个关键洞察：

1. **可验证环境 > 神经奖励模型**：Absolute Zero 用代码执行器实现了零数据 + 高稳定性，而 R-Zero 和 SRT 依赖神经信号（多数投票）最终都崩溃。这表明"零数据"不等于"零外部信号"——关键是信号来源是否可验证、是否独立于模型自身。

2. **1-5% 的魔法数字**：R-Few 证明极少量人类数据（1-5%）即可防止崩溃并实现 20 倍数据效率。这不是简单的数据量问题，而是"锚点密度"问题——足够稀疏以保持探索性，足够密集以防止漂移。

3. **环境复杂性的价值**：SPICE 的 20,000 文档语料库提供了"近乎无穷的多样性"，这是纯自博弈无法获得的。这回应了用户笔记中的核心愿景："让环境介入地进行数据合成"——环境不是被动的验证器，而是主动的知识来源。

4. **原则 > 标注**：Constitutional AI 证明少量明确的自然语言原则（16 条）比大量隐式的人类标注更透明、可控、易迭代。Self-Rewarding LM 进一步展示了如何用少量种子数据（3,200 IFT + 1,630 EFT）启动自我改进循环，后续完全自我生成训练数据。

未来方向应探索"混合锚定"：Absolute Zero 的可验证环境 + R-Few 的最小人类监督 + SPICE 的语料库多样性。这也暗示了一个更深层的问题：奖励函数本身能否由模型自主学习？SRT 的失败表明纯自奖励不可持续，但 Absolute Zero 的成功表明环境奖励可以。关键在于找到更多可验证的环境——形式化证明、物理模拟器、真实世界交互。

5. **Group 7 揭示了数据依赖光谱的"合成数据层"**：Group 1-2 关注的是"需要多少人类数据"，而 Group 7 论文揭示了一个正交维度——"合成数据的生成方式"。Nemotron 的迭代弱到强对齐证明弱模型可以训练强模型，Magpie 证明对齐模型本身就是最好的指令数据源，WizardLM 证明渐进式进化可以系统性地提升指令复杂度。这些方法与 Group 1-2 的自博弈方法形成互补：自博弈关注"如何从自身学习"，而 Group 7 关注"如何从教师/环境合成高质量数据"。两者的结合——用合成数据预训练/微调，再用自博弈持续改进——可能是最优路径。
