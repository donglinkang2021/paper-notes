---
tags:
  - insight
time: 2026-03-27T00:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_parallelizing_deltanet.md
  - ../knowledge/summary_deep_delta_learning.md
  - ../knowledge/summary_attention_residuals.md
  - ../knowledge/summary_tracerl_diffusion_llms.md
---

# 从时间到深度：状态更新规则正在成为新的模型设计主轴

这四篇论文虽然分属不同子问题——线性 Transformer、残差架构、深度路由、扩散模型 RL——但它们共享一个更深层的转向：**研究重点正在从“层里算什么”转向“状态如何沿某个轴被读取、擦除、重写与路由”**。这个轴可以是时间（DeltaNet）、深度（DDL / AttnRes），也可以是扩散推理轨迹（TraceRL）。

传统 Transformer 的很多设计默认这些轴上的信息流是固定的：时间上是简单累加或 softmax 聚合，深度上是 unit residual，扩散推理上是用最终答案奖励整个 rollout。新一代工作则在问：这些默认规则是否本身就是瓶颈？如果把“状态更新机制”显式建模，是否能带来更强的记忆、更稳的优化、更好的多步推理？这四篇论文的答案基本一致：**是的，而且收益往往来自让模型学会有选择地保留什么、删除什么、以及从哪里重新取回信息。**

## Evidence

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](../knowledge/summary_parallelizing_deltanet.md): 将线性注意力的纯加法 memory update 改成 delta rule，使模型能基于当前 key 先读旧值、再按误差擦除和写入，从而显著提升 associative recall；论文的核心贡献不是新记忆规则本身，而是把这种 state-to-state update 重新写成可 chunkwise 并行、适合 GPU 的训练算法
- [Deep Delta Learning](../knowledge/summary_deep_delta_learning.md): 将标准 identity shortcut 改成 $\mathbf{A}(\mathbf{X}) = \mathbf{I} - \beta(\mathbf{X})\mathbf{k}(\mathbf{X})\mathbf{k}(\mathbf{X})^\top$，把层间更新变成沿特定方向的 read-erase-write；其关键 insight 是 shortcut 本身也是应被学习的状态转移算子，而不应默认固定为 identity
- [Attention Residuals](../knowledge/summary_attention_residuals.md): 把所有历史层输出视为可检索 source memory，在深度维上做 softmax attention，让每层可以选择性读取更早表示；这相当于把 Transformer 在时间维做的事情复制到深度维，并直接缓解了 PreNorm dilution 与梯度分布失衡
- [TraceRL：面向扩散语言模型的轨迹感知强化学习框架](../knowledge/summary_tracerl_diffusion_llms.md): 指出 diffusion LM 的后训练目标与真实推理轨迹错位，进而把 RL 优化对象从最终结果改成 trajectory 本身；这里的“状态更新”不再是 hidden state recurrence，而是 rollout trace 上的 credit assignment 规则

## Implications

1. **“状态更新规则”正在从实现细节上升为一级架构变量。**
   过去很多模型把 recurrence / residual / rollout credit assignment 当作默认背景；这几篇论文说明，它们其实决定了模型是否能保留记忆、是否会发生表示稀释、以及训练目标是否真正贴合部署时的信息流。

2. **时间维、深度维、推理轨迹维正在出现统一抽象。**
   DeltaNet 在时间上做 read-erase-write，DDL 在深度上做 read-erase-write，AttnRes 在深度上做 selective retrieval，TraceRL 在 rollout 维度上重定义 credit assignment。它们都在处理同一个问题：**面对沿某条轴积累的状态，当前步骤到底应该如何访问历史、如何覆盖旧信息、如何避免“只会盲目累加”。**

3. **低秩或选择性更新是一个重要折中点。**
   这些方法都没有走向“完全自由的全状态重写”——那样通常太贵也太不稳定。相反，它们都选择了结构化、低秩、可解释的更新：DeltaNet 的 rank-1 delta rule、DDL 的 rank-1 Householder-style operator、AttnRes 的 block-wise source selection、TraceRL 的 shrinkage-aware trajectory aggregation。这说明真正可扩展的路线可能不是“更通用”，而是“足够结构化，能贴合硬件与优化约束”。

4. **这条路线尤其适合多步推理与长程依赖。**
   这四篇工作的收益都在需要跨步检索、持续累积、或防止旧信息被淹没的场景里最明显：DeltaNet 强于 associative recall，AttnRes 强于 GPQA/Math/HumanEval，TraceRL 强于复杂推理与 diffusion decoding。这提示一个方向：未来的 reasoning architecture 可能不会只靠更大的 MLP / 更多头，而会更多依赖“状态流动规则”的升级。

5. **你当前 repo 里的多条线索正在会合。**
   你之前读的很多论文在讨论 self-play、data synthesis、RLVR、memory、linear attention。这组论文把它们串起来了：无论是数据闭环、训练闭环还是模型内部闭环，真正的核心能力都越来越像“在历史状态上做受控、可学习的更新”。这可能是一个值得单独继续追踪的主题。