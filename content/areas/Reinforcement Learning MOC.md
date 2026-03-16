---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-15T03:40:13+00:00
author: Linkdom
---

# Reinforcement Learning MOC

强化学习方向的论文索引，涵盖 policy optimization、self-distillation for RL、丰富反馈等方向。

## Papers

- [Reinforcement Learning via Self-Distillation (SDPO)](../knowledge/summary_sdpo_rl_self_distillation.md) — 利用环境丰富文本反馈进行自蒸馏的 RL 算法
- [Self-Distillation Enables Continual Learning (SDFT)](../knowledge/summary_sdft_continual_learning.md) — 通过自蒸馏实现从专家示范的在策略持续学习
- [Self-Distilled Reasoner (OPSD)](../knowledge/summary_on_policy_self_distillation.md) — 单个 LLM 通过访问正确答案作为教师蒸馏自身
- [On-Policy Distillation](../knowledge/summary_on_policy_distillation_tml.md) — 结合 RL 的在策略采样与 SFT 的密集监督信号
- [Beyond Scalar Rewards: Learning from Text Feedback](../knowledge/summary_rl_text_feedback.md) — 用自然语言文本反馈替代稀疏标量奖励
- [Learning Adaptive Parallel Reasoning](../knowledge/summary_adaptive_parallel_reasoning.md) — 借鉴多线程机制让模型自适应编排串行与并行推理，通过 GRPO 端到端优化并行策略
- [The Surprising Effectiveness of Negative Reinforcement](../knowledge/summary_negative_reinforcement_reasoning.md) — 将 RLVR 分解为正负样本强化，发现仅惩罚错误回答即可有效提升推理并保持多样性
- [CUDA-L1: Improving CUDA Optimization via Contrastive RL](../knowledge/summary_cuda_l1.md) — 通过对比强化学习训练 LLM 自动优化 CUDA 内核，在 KernelBench 上实现平均 3.12 倍加速
- [RL's Razor: Why Online Reinforcement Learning Forgets Less](../knowledge/summary_rls_razor_forgets_less.md) — 提出用新任务分布上的 forward KL 预测遗忘，并解释 on-policy RL 为何更“保守”从而更不易遗忘
- [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](../knowledge/summary_retaining_by_doing_on_policy_data.md) — 通过消融论证“on-policy 数据”是 RL 抗遗忘主因，并提出 Iterative-SFT 等近似 on-policy 的省算力替代

## Related MOCs

- [Knowledge Distillation MOC](<./Knowledge Distillation MOC.md>)
- [Self-Play & Iterative Training MOC](<./Self-Play & Iterative Training MOC.md>)
