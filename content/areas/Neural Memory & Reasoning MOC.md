---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-05T14:30
author: Linkdom
---

# Neural Memory & Reasoning MOC

神经记忆与推理方向，涵盖记忆增强网络、关系推理、指针网络等。

## Papers

- [ReAct: Synergizing Reasoning and Acting](../knowledge/summary_react_reasoning_acting.md) — 让 LLM 交替生成推理轨迹和任务动作
- [Pointer Networks](../knowledge/summary_pointer_networks.md) — 注意力作为指针指向输入序列元素
- [Neural Turing Machines](../knowledge/summary_neural_turing_machines.md) — 可微分外部存储器的神经网络计算机
- [Order Matters: Sequence to Sequence for Sets](../knowledge/summary_order_matters_seq2seq_sets.md) — 置换不变的 Read-Process-Write 架构
- [A Simple Neural Network Module for Relational Reasoning](../knowledge/summary_relation_networks.md) — 对所有对象对计算关系函数实现端到端关系推理
- [Relational Recurrent Neural Networks](../knowledge/summary_relational_memory_core.md) — 多头注意力实现记忆槽间显式交互

## 状态更新与层间记忆

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](../knowledge/summary_parallelizing_deltanet.md) — 将 delta rule 记忆更新并行化到序列维，显著提升线性 Transformer 的 associative recall 与语言建模能力
- [Deep Delta Learning](../knowledge/summary_deep_delta_learning.md) — 把残差连接重写成沿特定方向的 read-erase-write 过程，让 hidden state 在深度上具备可控记忆更新
- [Attention Residuals](../knowledge/summary_attention_residuals.md) — 将历史层输出视为可检索 memory source，在深度维做选择性聚合以缓解 PreNorm dilution

## Related MOCs

- [Sequence Models MOC](<./Sequence Models MOC.md>)
- [Relational Reasoning MOC](<./Relational Reasoning MOC.md>)
