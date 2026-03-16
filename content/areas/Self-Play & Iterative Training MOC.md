---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-15T03:40:13+00:00
author: Linkdom
---

# Self-Play & Iterative Training MOC

自博弈与迭代训练方向，研究如何让 LLM 通过自身生成的数据进行迭代式自我提升，涵盖自举推理、奖励过滤、自博弈对齐、无数据训练、多智能体微调等方法。

## Papers

- [STaR: Self-Taught Reasoner](../knowledge/summary_star_bootstrapping_reasoning.md) — 通过迭代自举让模型用自己生成的正确推理链训练自己
- [ReST: Reinforced Self-Training](../knowledge/summary_rest_reinforced_self_training.md) — 将 RLHF 解耦为 Grow-Improve 两阶段，用递增阈值过滤实现离线强化自训练
- [SPIN: Self-Play Fine-Tuning](../knowledge/summary_spin_self_play_finetuning.md) — 基于自博弈的微调方法，无需偏好数据即可达到甚至超越 DPO 效果
- [LSP: Language Self-Play](../knowledge/summary_language_self_play_data_free.md) — 完全无数据的语言自博弈训练，模型同时扮演出题者和解题者
- [Multiagent Finetuning](../knowledge/summary_multiagent_finetuning.md) — 通过多智能体辩论和角色专业化实现持续自我提升
- [Absolute Zero Reasoning](../knowledge/summary_absolute_zero_reasoning.md) — 完全零数据的强化自博弈推理，利用代码执行器作为可验证环境实现自主进化
- [R-Zero: Self-Evolving LLM](../knowledge/summary_r_zero_self_evolving.md) — 从零数据自进化的推理模型，通过 Challenger-Solver 独立共同进化实现自适应课程学习
- [SPICE: Self-Play In Corpus](../knowledge/summary_spice_self_play_corpus.md) — 基于语料库环境的自博弈推理，通过文档根据解决纯自博弈的幻觉和信息对称性问题
- [R-Few: Guided Self-Evolving](../knowledge/summary_r_few_guided_self_evolving.md) — 最小人类监督下的引导式自进化，用 1-5% 锚点数据防止概念漂移和多样性崩溃
- [Can Reasoning Models Self-Train?](../knowledge/summary_can_reasoning_models_self_train.md) — 研究大型推理模型通过多数投票自我训练的可行性与奖励黑客导致的模型崩溃问题
- [Constitutional AI: Harmlessness from AI Feedback](../knowledge/summary_constitutional_ai.md) — 通过自然语言宪法原则实现 AI 自我批评和修订，用 AI 反馈替代人类标注训练无害助手
- [Self-Rewarding Language Models](../knowledge/summary_self_rewarding_language_models.md) — 模型同时充当生成器和奖励模型，通过迭代训练同时提升指令遵循和自我评判能力
- [Self-Questioning Language Models](../knowledge/summary_self_questioning_lm.md) — 完全零数据的非对称自博弈框架，模型自己生成问题并解答，通过多数投票实现无监督验证
- [Self-Challenging Language Model Agents](../knowledge/summary_self_challenging_agents.md) — 智能体通过 Code-as-Task 自己生成训练任务，在多轮工具使用场景中实现自我挑战和改进
- [Genius: Unsupervised Reasoning](../knowledge/summary_genius_unsupervised_reasoning.md) — 完全无监督的推理自训练框架，通过逐步前瞻重采样和优势校准优化实现零监督推理提升
- [STaR-GATE: Clarifying Questions](../knowledge/summary_star_gate_clarifying_questions.md) — 将 STaR 扩展到对话场景，通过自博弈教会模型主动提出澄清性问题以激发用户偏好
- [Beyond Pass@1: Variational Problem Synthesis](../knowledge/summary_variational_problem_synthesis.md) — 通过变分问题合成的在线自博弈维持 RLVR 训练熵，解决熵坍塌问题并显著提升 Pass@k 性能
- [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](../knowledge/summary_retaining_by_doing_on_policy_data.md) — 从“数据是否 on-policy”解释 RL 更不易遗忘，并提示 Iterative-SFT/每轮刷新数据可作为闭环自训练的稳定化手段
- [RL's Razor: Why Online Reinforcement Learning Forgets Less](../knowledge/summary_rls_razor_forgets_less.md) — 把 on-policy RL 的“保守性”形式化为 KL-minimal 偏置，并用新任务分布上的 forward KL 作为遗忘风险的领先指标

## Related MOCs

- [Reinforcement Learning MOC](<./Reinforcement Learning MOC.md>)
