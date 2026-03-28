---
tags:
  - MOC
  - knowledge-moc
time: 2026-03-05T14:30
author: Linkdom
---

# Sequence Models MOC

涵盖 RNN、LSTM、Transformer、注意力机制等序列建模的经典与前沿工作。

## RNN & LSTM 基础

- [The Unreasonable Effectiveness of Recurrent Neural Networks](../knowledge/summary_rnn_effectiveness.md) — 通过生动实验展示 RNN 在字符级序列建模上的强大能力
- [Understanding LSTM Networks](../knowledge/summary_understanding_lstm.md) — 以清晰图解直观解释 LSTM 门控机制的经典教程
- [Recurrent Neural Network Regularization](../knowledge/summary_rnn_dropout_regularization.md) — 只在非循环连接上应用 Dropout 来正则化 LSTM

## 注意力机制

- [Neural Machine Translation by Jointly Learning to Align and Translate](../knowledge/summary_attention_nmt.md) — 提出注意力机制让解码器自适应关注源句子不同部分
- [Pointer Networks](../knowledge/summary_pointer_networks.md) — 将注意力输出直接作为指针指向输入序列元素

## 记忆增强与推理

- [Neural Turing Machines](../knowledge/summary_neural_turing_machines.md) — 将神经网络与外部存储器耦合，通过可微注意力实现读写
- [Order Matters: Sequence to Sequence for Sets](../knowledge/summary_order_matters_seq2seq_sets.md) — 揭示顺序对 seq2seq 性能的影响，提出 Read-Process-Write 架构
- [Relational Recurrent Neural Networks](../knowledge/summary_relational_memory_core.md) — 用多头注意力实现记忆槽间的显式交互

## 预训练语言模型

- [XLNet: Generalized Autoregressive Pretraining](../knowledge/summary_xlnet.md) — 通过排列语言建模同时获得自回归和自编码预训练的优点

## 现代状态更新与残差机制

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](../knowledge/summary_parallelizing_deltanet.md) — 用 chunkwise 并行算法把 DeltaNet 扩展到现代语言建模规模，证明 delta rule 记忆更新比普通线性注意力更擅长 associative recall
- [Deep Delta Learning](../knowledge/summary_deep_delta_learning.md) — 将恒等残差推广为可学习的 rank-1 几何算子，在层间实现沿特定方向的擦除与写入
- [Attention Residuals](../knowledge/summary_attention_residuals.md) — 把固定残差累加改成沿深度做 softmax attention，使每层可选择性读取更早层表示
- [TraceRL：面向扩散语言模型的轨迹感知强化学习框架](../knowledge/summary_tracerl_diffusion_llms.md) — 让 diffusion LM 的后训练显式对齐采样轨迹，用 trajectory-aware RL 优化推理过程而非只看最终输出

## Related MOCs

- [Neural Memory & Reasoning MOC](<./Neural Memory & Reasoning MOC.md>)
- [Reinforcement Learning MOC](<./Reinforcement Learning MOC.md>)
- [Information Theory & Complexity MOC](<./Information Theory & Complexity MOC.md>)
