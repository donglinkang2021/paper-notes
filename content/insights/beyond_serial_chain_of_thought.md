---
tags:
  - insight
time: 2026-03-05T22:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_adaptive_parallel_reasoning.md
  - ../knowledge/summary_negative_reinforcement_reasoning.md
  - ../knowledge/summary_can_reasoning_models_self_train.md
  - ../knowledge/summary_absolute_zero_reasoning.md
---

# 超越串行思维链：推理计算的多维扩展

传统推理扩展（test-time scaling）几乎完全依赖串行思维链（CoT）——通过加长序列实现"更深的思考"。Group 6 的两篇论文从不同角度揭示了这一范式的根本局限，并指向推理计算的多维扩展方向：APR 提出"更宽的搜索"（并行线程），NSR 揭示"更聪明的学习"（保护推理多样性以支撑 inference scaling）。

## Evidence

**串行 CoT 的瓶颈**：
- [APR](../knowledge/summary_adaptive_parallel_reasoning.md): 上下文窗口是串行推理的根本瓶颈。在 Countdown 任务上，串行方法（SoS+）在 4k 窗口下准确率仅 60.0%，而 APR 通过将搜索分布到多个并行线程达到 83.4%（+23.4%）。串行方法必须将整个搜索轨迹塞入单一上下文窗口，而并行方法绕过了这一限制
- [NSR](../knowledge/summary_negative_reinforcement_reasoning.md): RL 训练（PSR 主导）在提升 Pass@1 的同时严重损害 Pass@$k$（大 $k$），即模型变得"更准但更窄"。PPO 和 GRPO 训练后 Pass@256 均低于基线，说明传统 RL 优化串行 CoT 质量的同时牺牲了推理路径的多样性

**并行化作为新维度**：
- APR 借鉴操作系统多线程机制，模型通过 `spawn()` 派生子线程并行探索，`join()` 返回结果。RL 后模型自主学会扩展搜索宽度：子线程数从 6.1 增至 8.2（+34.4%），远超序列长度增长（+22.1%）
- APR 用 57.4% 更少的总计算量即可匹配串行方法 pass@8 的性能
- 在相同延迟约束下（~5000ms），APR 准确率 75.2% vs 串行 57.3%

**推理多样性作为 inference scaling 的基础**：
- NSR 建立了熵与 Pass@$k$ 性能的清晰联系：高熵 → 高多样性 → 强 Pass@$k$（大 $k$）
- NSR 训练后 Pass@256 达到 96.9（匹配基线），而 PSR 仅 91.2——说明保护推理路径多样性对 inference scaling 至关重要
- SRT 论文的崩溃现象（所有输出退化为模板答案）是推理多样性完全丧失的极端案例

## Implications

这两篇论文共同指向一个核心洞察：**推理能力的扩展不应仅依赖"更长的思考"，而应同时追求"更宽的探索"和"更丰富的路径"**。

1. **宽度 vs 深度的权衡**：APR 的 RL 实验发现，模型自主选择了"更宽"而非"更深"的搜索策略（子线程增长 34.4% vs 序列长度增长 22.1%）。这暗示在许多推理任务上，探索更多可能性比沿单一路径深入更有效。

2. **多样性是 inference scaling 的前提**：NSR 揭示了一个被忽视的问题——传统 RL 训练在优化 Pass@1 的同时破坏了 inference scaling 的基础（推理路径多样性）。如果模型只会一种解题方式，那么无论采样多少次（best-of-N）或搜索多宽（beam search），都无法获得额外收益。

3. **从"教模型思考"到"教模型编排计算"**：APR 展示了一种元层次的能力——模型不仅学会解题，还学会了如何分配计算资源（何时 spawn、spawn 多少、传递什么上下文）。这与传统 CoT 仅优化"思考内容"形成对比，指向一种更高层次的推理能力。

4. **两种互补的扩展路径**：APR 的并行化在推理时（inference time）扩展计算宽度，NSR 的多样性保护在训练时（training time）保护模型的探索能力。两者可以组合——用 NSR/W-REINFORCE 训练保持模型的推理多样性，用 APR 的并行机制在推理时高效利用这种多样性。

未来方向：将 APR 的并行推理框架扩展到预训练语言模型和通用推理任务，同时用 NSR 主导的 RL 训练保护模型在并行探索中的路径多样性。
