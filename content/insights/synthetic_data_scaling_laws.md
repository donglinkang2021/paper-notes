---
tags:
  - insight
time: 2026-03-06T00:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_scaling_laws_synthetic_data.md
  - ../knowledge/summary_will_run_out_data.md
  - ../knowledge/summary_phi1_textbooks.md
  - ../knowledge/summary_phi15_textbooks2.md
  - ../knowledge/summary_phi3_technical_report.md
  - ../knowledge/summary_phi4_technical_report.md
  - ../knowledge/summary_smollm2.md
  - ../knowledge/summary_nemotron4_340b.md
  - ../knowledge/summary_synthetic_bootstrapped_pretraining.md
  - ../knowledge/summary_synthetic_continued_pretraining.md
  - ../knowledge/summary_s1_simple_test_time_scaling.md
  - ../knowledge/summary_lima_less_is_more_alignment.md
  - ../knowledge/summary_limo_less_is_more_reasoning.md
---

# 合成数据缩放定律与数据耗尽时间线

合成数据能否像真实预训练数据一样展现可预测的缩放规律？Group 4 论文给出了肯定答案，同时揭示了人类数据耗尽的紧迫时间线（2026-2032）和合成数据的性能边界（300B tokens）。

## 合成数据遵循修正缩放定律

GLAN v1.5 在数学推理领域的实验证明：合成数据在不同模型规模（3B、8B）下都遵循修正缩放定律，性能增益具有可预测性——数据集规模翻倍，MATH 基准错误率持续降低约 4%。

**关键发现**：
- **性能饱和点**：约 300B tokens 后增益递减，存在"数据墙"
- **模型规模与数据效率**：8B 模型仅需 1T tokens 达到最佳性能，3B 模型需要 4T tokens
- **更大模型展现更高样本效率**

这与真实数据的 Chinchilla 缩放定律（每参数 20 tokens）形成对比——合成数据的最优 tokens/参数比更高，表明合成数据信息密度低于真实数据。

## 人类数据耗尽时间线：2026-2032

Will We Run Out of Data 论文系统估算了公开人类文本数据存量和 LLM 训练需求的交汇点：

**数据存量估算**（经质量过滤和多轮训练调整）：
- Common Crawl：130T tokens
- 索引网页：510T tokens（有效存量约 400T tokens）
- 整个网络：3100T tokens

**数据需求预测**：
- 历史增长率：每年 0.38 OOM（2.4 倍）
- 当前最大数据集：Llama 3 使用 15T tokens（2024）
- **耗尽时间**：2028 年（中位数），95% 置信区间 2026-2032 年
- **耗尽时的数据集规模**：约 400T tokens

**紧迫性**：如果模型过度训练 5 倍（tokens/参数比是计算最优的 5 倍），数据瓶颈将提前约 1 年出现。

## 300B Tokens 性能边界的含义

GLAN v1.5 发现的 300B tokens 性能饱和点与 Will We Run Out of Data 的 400T tokens 耗尽预测形成有趣对比：

- **300B = 0.3T**：单个领域（数学）的合成数据饱和点
- **400T**：所有公开人类文本数据的有效存量
- **差距 1000 倍**：表明跨领域合成数据仍有巨大空间

但这也揭示了合成数据的根本限制：**信息密度低于真实数据**。如果每个领域都需要 300B tokens 合成数据才能饱和，而真实数据可能用更少的量达到相同效果，那么合成数据的总需求可能远超真实数据存量。

## Evidence

- [Scaling Laws of Synthetic Data](../knowledge/summary_scaling_laws_synthetic_data.md)：合成数据遵循修正缩放定律，300B tokens 后收益递减，8B 模型需 1T tokens 达最优
- [Will We Run Out of Data](../knowledge/summary_will_run_out_data.md)：公开人类文本数据将在 2026-2032 年（中位数 2028）耗尽，有效存量约 400T tokens
- [Phi-1](../knowledge/summary_phi1_textbooks.md)：1.3B 模型用不到 7B tokens 的"教科书质量"数据达到 15B+ 模型的代码生成水平，证明高质量合成数据可将缩放定律的效率提升两个数量级
- [Phi-1.5](../knowledge/summary_phi15_textbooks2.md)：1.3B 模型用 30B tokens 合成数据在推理任务上超越 1T tokens 训练的 7B 模型，训练成本仅为 Llama-7B 的 1/50
- [Phi-3](../knowledge/summary_phi3_technical_report.md)：3.8B 模型在 MMLU 上达到 69%，媲美 45B 参数的 Mixtral 8x7B，提出"数据最优范式"——针对小模型容量限制优化数据质量而非数量
- [Phi-4](../knowledge/summary_phi4_technical_report.md)：合成数据占预训练 40%，在合成数据上训练 13.8 轮仍不过拟合，且多轮合成数据训练优于引入更多新鲜网络数据——挑战了"数据重复导致过拟合"的传统认知
- [SmolLM2](../knowledge/summary_smollm2.md)：1.7B 模型在 11T tokens 上训练（远超 Chinchilla 最优），FineMath 通过聚焦逐步推理内容实现数学能力数倍提升，验证了分类器过滤作为通用数据策展方法的有效性
- [Nemotron-4 340B](../knowledge/summary_nemotron4_340b.md)：98% 合成数据 + 仅 20K 人工标注即可训练出与 GPT-4 可比的指令模型，验证了合成数据在大规模对齐中的可行性

## Implications

这两篇论文共同描绘了 LLM 数据生态的未来图景：

**时间窗口**：2024-2028 年是关键过渡期。当前（2024）最大数据集 15T tokens，距离耗尽点 400T tokens 还有约 4 年。这个窗口内必须解决：
1. 合成数据生成方法的成熟化
2. 多模态迁移学习的缩放定律
3. 数据效率提升算法

**合成数据的双重角色**：
- **短期**：补充真实数据，延缓耗尽时间
- **长期**：成为主要数据源，但需突破 300B tokens 性能墙

**突破路径的优先级**：
1. **合成数据 + 可验证环境**：在数学、编程等可验证领域，合成数据已展现出色效果（AlphaZero、AlphaGeometry）
2. **多模态迁移学习**：图像（300T tokens）、视频（1350T tokens）数据存量更大，但迁移效率未知
3. **数据效率提升**：算法进步每年 0.4 OOM，可能部分抵消数据耗尽影响

**与模型坍塌的联系**：数据耗尽迫使我们依赖合成数据，但 Breaking Curse 和 ToEdit 论文表明只要方法得当（数据累积或 token 级编辑），合成数据可以避免模型坍塌。这为合成数据主导的未来提供了理论保障。

**Group 7 的关键补充——Phi 系列重新定义了缩放定律的形态**：

GLAN v1.5 发现的 300B tokens 性能墙是在"固定质量、增加数量"的范式下观察到的。但 Phi 系列论文揭示了另一条路径：**提升数据质量可以在远低于 300B tokens 的数据量下达到甚至超越性能墙**。Phi-1 用 7B tokens、Phi-1.5 用 30B tokens 就达到了需要 100B-1T tokens 的模型性能。Phi-4 更进一步证明合成数据可以多轮训练（13.8 轮）而不过拟合，这意味着 300B tokens 的性能墙可能并非合成数据的固有限制，而是数据质量不足的表现。

SmolLM2 的经验则提供了互补视角：即使在"过度训练"（11T tokens，远超 Chinchilla 最优）的情况下，通过多阶段引入高质量专业数据（FineMath、Stack-Edu），小模型仍能持续获益。这表明缩放定律的"最优点"高度依赖于数据质量分布，而非仅由模型规模和数据量决定。

**开放问题**：
- 跨领域合成数据的总需求是否会超过真实数据存量？
- 300B tokens 性能墙能否通过更好的生成方法突破？
- 多模态数据的有效 tokens 如何换算（1 张图像 = 多少文本 tokens）？

## 新论文的补充证据

**SBP 的缩放行为**：SBP 在 200B、1T (3B)、1T (6B) 三个规模点上一致优于重复基线，且改进幅度随模型规模增长（42%→48%→58% 的 oracle 恢复率）。这暗示合成数据的缩放定律可能存在"模型规模"这一额外维度——更大模型能更有效地利用合成数据中的额外信号。SBP 的最优合成/真实数据比例也随模型规模增长（3B: ~12.5%, 6B: ~25%），这与 GLAN v1.5 发现的固定性能墙形成对比：**性能墙的位置可能取决于模型规模和合成方法的信号类型**。

**EntiGraph 的对数线性扩展**：EntiGraph 合成 CPT 在 455M tokens 内展现清晰的对数线性扩展曲线，且理论分析预测了混合指数形式 $\mathrm{Acc}(t) \approx 1 - \sum_k \alpha_k e^{-\beta_k t}$ 的三阶段增长。这提供了比 GLAN v1.5 更精细的扩展定律形式。值得注意的是，EntiGraph 在 455M tokens 时尚未达到饱和平台，暗示在小语料设定下性能墙可能远高于 300B tokens。

**s1 的数据效率极限**：s1 用 1K 精选样本就达到了需要 800K+ 样本的系统的相当水平。将其放入缩放定律框架中，这意味着在推理微调任务上，"数据量→性能"的曲线斜率高度依赖于数据选择方法。s1 的数据消融（1K-random 36.7% vs s1K 50.0% vs 59K-full 53.3%）更直接说明：**在数据稀缺场景下，选择策略（质量×难度×多样性）可以实现与 59 倍数据量几乎相当的性能**。

**LIMA 和 LIMO 揭示的"反缩放"现象**：LIMA 的消融实验表明，对齐数据从 2K 增加到 32K 性能不变——这是一条完全平坦的缩放曲线。LIMO 更进一步发现了负缩放：NuminaMath-100k（100K 条）导致 AIME24 从 16.5% 退化到 6.5%，而 LIMO（800 条）提升到 63.3%。这意味着在后训练阶段，传统的"更多数据→更好性能"缩放定律可能根本不成立。取而代之的是一种"质量阈值"模型：**存在一个数据质量阈值，低于该阈值的数据无论多少都无法提升性能甚至有害，高于该阈值的数据只需极少量即可激发能力**。这与预训练阶段的幂律缩放形成鲜明对比，暗示预训练和后训练遵循根本不同的缩放规律。
