---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-05T14:30
author: Linkdom
---

# Scaling Laws MOC

缩放定律与训练效率方向，研究模型规模、数据量与性能的关系。

## Papers

- [Scaling Laws for Neural Language Models](../knowledge/summary_scaling_laws_lm.md) — 语言模型性能与模型大小、数据量、计算量呈幂律关系
- [Training Compute-Optimal Large Language Models (Chinchilla)](../knowledge/summary_chinchilla_scaling.md) — 模型大小和训练数据量应等比例增长
- [Modular Manifolds](../knowledge/summary_modular_manifolds.md) — 将权重矩阵约束在数学流形上改善训练稳定性和优化效率
- [Will we run out of data?](../knowledge/summary_will_run_out_data.md) — 预测公开人类文本数据将在2026-2032年耗尽，分析突破数据瓶颈的三大路径
- [Scaling Laws of Synthetic Data](../knowledge/summary_scaling_laws_synthetic_data.md) — 合成数据遵循修正缩放定律，300B tokens后性能增益递减
- [Breaking the Curse of Recursion](../knowledge/summary_breaking_curse_recursion.md) — 累积真实和合成数据可避免模型坍塌，测试误差有有限上界
- [How to Synthesize without Model Collapse?](../knowledge/summary_synthesize_without_collapse.md) — Token级编辑方法通过保持分布覆盖避免模型坍塌

## 合成数据与知识获取效率

- [Synthetic Bootstrapped Pretraining (SBP)](../knowledge/summary_synthetic_bootstrapped_pretraining.md) — 通过文档间关联自举合成，在计算匹配实验中持续扩展，最优合成比例随模型规模增长
- [Synthetic Continued Pretraining (EntiGraph)](../knowledge/summary_synthetic_continued_pretraining.md) — 合成 CPT 展现对数线性扩展，理论证明混合指数形式的准确率增长曲线
- [s1: Simple Test-Time Scaling](../knowledge/summary_s1_simple_test_time_scaling.md) — 1K 精选样本 + Budget Forcing 实现顺序式 test-time scaling，挑战"数据越多越好"的假设
- [LIMA: Less Is More for Alignment](../knowledge/summary_lima_less_is_more_alignment.md) — 消融实验证明对齐数据量从 2K 到 32K 性能不变，对齐的缩放定律由多样性和质量而非数量决定
- [LIMO: Less is More for Reasoning](../knowledge/summary_limo_less_is_more_reasoning.md) — 800 条数据超越 100K+ 数据，100K 低质量数据反而导致退化，揭示推理微调中数据量-性能的非单调关系

## Related MOCs

- [Data Synthesis & Curation MOC](<./Data Synthesis & Curation MOC.md>)
