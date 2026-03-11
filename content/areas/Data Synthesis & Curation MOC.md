---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-06T10:00
author: Linkdom
---

# Data Synthesis & Curation MOC

合成数据生成与数据策展方向，研究如何通过合成数据、数据过滤和数据质量优化来提升语言模型训练效率，涵盖指令进化、对齐数据合成、教科书质量数据等方法。

## Phi 系列：合成数据驱动的小模型

- [Textbooks Are All You Need (Phi-1)](../knowledge/summary_phi1_textbooks.md) — 用"教科书质量"合成数据训练 1.3B 代码模型，以不到 7B tokens 超越百倍规模模型
- [Textbooks Are All You Need II (Phi-1.5)](../knowledge/summary_phi15_textbooks2.md) — 将合成教科书方法从代码扩展到常识推理，1.3B 模型达到 5-10 倍参数量模型水平
- [Phi-3 Technical Report](../knowledge/summary_phi3_technical_report.md) — 提出"数据最优范式"，3.8B 模型通过精心策划数据达到 Mixtral 8x7B 性能，可在手机端运行
- [Phi-4 Technical Report](../knowledge/summary_phi4_technical_report.md) — 40% 合成数据预训练 + 关键令牌搜索（PTS）DPO，14B 模型在 STEM 上超越教师模型 GPT-4o

## 对齐数据合成

- [Magpie: Alignment Data Synthesis from Scratch](../knowledge/summary_magpie_alignment_synthesis.md) — 仅用预查询模板触发对齐 LLM 自回归生成指令，无需种子数据即可大规模合成高质量对齐数据
- [WizardLM: Evol-Instruct](../knowledge/summary_wizardlm_evol_instruct.md) — 通过深度和广度进化策略将简单指令逐步进化为复杂指令，替代昂贵的人工标注
- [Nemotron-4 340B](../knowledge/summary_nemotron4_340b.md) — 迭代弱到强对齐，98% 合成数据 + 仅 20K 人工标注训练出与 GPT-4 可比的指令模型

## 合成预训练数据增强

- [Synthetic Bootstrapped Pretraining (SBP)](../knowledge/summary_synthetic_bootstrapped_pretraining.md) — 从预训练数据自身学习文档间条件关联，自举合成新语料，计算匹配下恢复 oracle 改进的 42%–58%
- [Synthetic Continued Pretraining (EntiGraph)](../knowledge/summary_synthetic_continued_pretraining.md) — 通过实体图增强将 1.3M tokens 小语料合成为 455M tokens，闭卷 QA 从 39.49% 提升到 56.22%，且与 RAG 互补

## 数据策展与小模型训练

- [SmolLM2](../knowledge/summary_smollm2.md) — 以数据为中心的多阶段训练策略，构建 FineMath/Stack-Edu 等高质量数据集训练 1.7B SOTA 小模型

## 少量数据激活能力：从对齐到推理

- [LIMA: Less Is More for Alignment](../knowledge/summary_lima_less_is_more_alignment.md) — 1,000 条精选示例做 SFT 即可对齐 65B 模型，提出"表面对齐假设"：能力在预训练中习得，对齐只是格式
- [LIMO: Less is More for Reasoning](../knowledge/summary_limo_less_is_more_reasoning.md) — 800 条精选推理样本将 AIME24 从 16.5% 提升到 63.3%，将 LIMA 的"少即是多"从对齐推广到数学推理
- [s1: Simple Test-Time Scaling](../knowledge/summary_s1_simple_test_time_scaling.md) — 用 1,000 条精选蒸馏推理样本 + Budget Forcing 解码技巧实现 test-time scaling，AIME24 从 26.7% 提升到 56.7%

## Related MOCs

- [Scaling Laws MOC](<./Scaling Laws MOC.md>)
- [Self-Play & Iterative Training MOC](<./Self-Play & Iterative Training MOC.md>)
- [Knowledge Distillation MOC](<./Knowledge Distillation MOC.md>)
