---
title: "Deep Delta Learning"
authors: "Yifan Zhang, Yifeng Liu, Mengdi Wang, Quanquan Gu"
institution: "Princeton University, University of California, Los Angeles"
venue: "arXiv 2026"
arxiv_id: "2601.00417"
tags: ["architecture", "residual learning", "transformer", "geometry", "householder", "delta rule", "language modeling"]
---

# Deep Delta Learning：把残差连接从恒等映射推广为可学习几何算子

## 核心贡献

1. **重新审视残差连接的 inductive bias**：论文认为标准 ResNet / Transformer 的 shortcut 始终是 identity map，这虽然缓解了梯度消失，但也把层间状态转移限制成“在恒等映射上加一个残差”的纯加法模式，难以表达更复杂、特别是带负特征值的 hidden state transition。
2. **提出 Deep Delta Learning (DDL)**：将 shortcut 从固定的恒等映射推广为状态相关的 rank-1 线性算子
   $$
   \mathbf{A}(\mathbf{X}) = \mathbf{I} - \beta(\mathbf{X})\,\mathbf{k}(\mathbf{X})\mathbf{k}(\mathbf{X})^\top
   $$
   其中 $\mathbf{k}(\mathbf{X})$ 是单位方向，$\beta(\mathbf{X})$ 是 gate。
3. **把 identity / projection / reflection 统一进同一个连续模块**：当 $\beta=0$ 时是 identity；$\beta=1$ 时退化为正交投影；$\beta=2$ 时是 Householder reflection。也就是说，网络可以沿着某个数据相关方向，连续地控制 shortcut 的谱性质。
4. **把残差写成 delta-rule 形式**：DDL 的层更新不仅是几何 shortcut，也可以重写成同步的 rank-1 erase-and-write 更新，和 DeltaNet / delta rule 在代数结构上对应。
5. **在语言模型 Transformer 中做 drop-in replacement**：论文把标准残差加法换成 DDL 更新，在 124M / 353M 规模、FineWeb-Edu 100B 上得到更低 validation loss / perplexity，并在多个下游 1-shot / 0-shot benchmark 上取得稳定提升。

## 方法详解

### 1. 从标准残差到 Delta Operator

标准残差写作：

$$
\mathbf{X}_{l+1} = \mathbf{X}_l + \mathcal{F}(\mathbf{X}_l)
$$

作者认为这种更新有很强的 translational bias：shortcut 的 Jacobian 总是 identity，因此网络天然偏向“保留原方向 + 加少量修正”。

DDL 把 shortcut 改为一个状态相关的 rank-1 变换：

$$
\mathbf{A}(\mathbf{X}) = \mathbf{I} - \beta(\mathbf{X})\,\mathbf{k}(\mathbf{X})\mathbf{k}(\mathbf{X})^\top
$$

这里：

- $\mathbf{k}(\mathbf{X}) \in \mathbb{R}^d$ 是归一化方向；
- $\beta(\mathbf{X}) \in (0,2)$ 由 sigmoid parameterization 给出；
- hidden state 被写成矩阵 $\mathbf{X} \in \mathbb{R}^{d \times d_v}$，其中 $d_v$ 是 value channels。

完整 block 输出是：

$$
\mathbf{X}_{l+1} = \mathbf{A}(\mathbf{X}_l)\mathbf{X}_l + \beta(\mathbf{X}_l)\mathbf{k}(\mathbf{X}_l)\mathbf{v}(\mathbf{X}_l)^\top
$$

也就是：先对当前状态沿 $\mathbf{k}$ 方向做受控几何变换，再沿同一方向写入一个新的 value 向量 $\mathbf{v}(\mathbf{X})$。

### 2. 等价的 delta update 形式

把上式展开后可得：

$$
\mathbf{X}_{l+1} = \mathbf{X}_l + \beta(\mathbf{X}_l)\,\mathbf{k}(\mathbf{X}_l)\Big(\mathbf{v}(\mathbf{X}_l)^\top - \mathbf{k}(\mathbf{X}_l)^\top\mathbf{X}_l\Big)
$$

这个式子非常关键，因为它把 DDL 显式写成了“先读取当前沿 $\mathbf{k}$ 的投影，再减掉旧分量、写入新分量”的 delta rule：

- $\mathbf{k}^\top \mathbf{X}_l$：当前沿该方向的 readout；
- $\mathbf{v}^\top$：希望写入的新内容；
- 两者差值：correction signal；
- $\beta$：同步调节 erase 和 write 的 step size。

因此 DDL 并不是“又一个 fancy residual trick”，而是把残差学习解释成一种深度方向上的 memory rewrite 过程。

### 3. 谱分析：为什么 $\beta$ 有几何意义

对固定 $\beta$ 和单位向量 $\mathbf{k}$，Delta Operator

$$
\mathbf{A} = \mathbf{I} - \beta \mathbf{k}\mathbf{k}^\top
$$

的特征值非常简单：

- 在 $\mathbf{k}^\perp$ 上，特征值都是 $1$；
- 在 $\mathbf{k}$ 方向上，特征值是 $1 - \beta$。

所以：

- **$\beta \to 0$**：$1-\beta \to 1$，整个算子接近 identity；
- **$\beta \to 1$**：$1-\beta \to 0$，在 $\mathbf{k}$ 方向上做投影消除；
- **$\beta \to 2$**：$1-\beta \to -1$，变成标准 Householder reflection。

这意味着网络可以沿着某个选择出来的方向，决定是：

1. 基本保留原状态；
2. 删掉该方向上的旧分量；
3. 翻转该方向上的分量；
4. 再写入新的内容。

论文把这种能力解释为：相比固定 identity shortcut，DDL 提供了对 shortcut spectrum 的显式控制。

### 4. 和 DeltaNet / ODE 的关系

论文给出两个很有意思的解释：

- **ODE 视角**：标准残差是 $\dot{\mathbf{X}} = \mathcal{F}(\mathbf{X})$ 的 Euler step，而 DDL 的更新可看成
  $$
  \dot{\mathbf{X}} = \mathbf{k}(\mathbf{X})\big(\mathbf{v}(\mathbf{X})^\top - \mathbf{k}(\mathbf{X})^\top\mathbf{X}\big)
  $$
  的状态相关步长离散化。
- **DeltaNet 视角**：DDL 与 DeltaNet 的记忆更新在代数形式上同构，只是 DeltaNet 在时间维度上更新 memory，而 DDL 在网络深度维度上更新 hidden state。

这使 DDL 不只是一个 heuristic，而是和已有 memory model / geometric operator / ODE 视角有明确结构联系。

### 5. DDL Transformer 的两种设置

论文主要在 Transformer LM 中验证 DDL：

#### 标量值设置：$d_v = 1$

当 value dimension 缩成 1 时，矩阵状态退化为向量状态 $\mathbf{x} \in \mathbb{R}^d$，更新变成：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \beta_l\,(v_l - \mathbf{k}_l^\top \mathbf{x}_l)\,\mathbf{k}_l
$$

这是最直接的“沿一个方向修改当前表示”的版本。

#### 扩展状态设置：$d_v > 1$

作者更强调这个 regime：把 residual state 扩展成矩阵 $\mathbf{X}_l \in \mathbb{R}^{d \times d_v}$，其中 backbone 宽度仍是 $d$，但 memory capacity 随 $d_v$ 增长。为兼容标准 Transformer 子层，使用 **Compress-Process-Expand** 接口，把矩阵状态压回向量交给 attention / MLP，再展开回矩阵。

这个设置的直觉很像：backbone 负责计算，expanded residual state 负责存储 / 重写更多信息。

## 实验结果

### 1. 语言建模 loss / perplexity 优于 baseline

实验在 FineWeb-Edu 100B 上进行，比较对象是基于 nanoGPT 的 baseline 与 DDL Transformer。

- 模型规模：124M（small）与 353M（medium）
- 训练步数：100k steps
- 序列长度：1024
- 4 张 NVIDIA H200

最终 validation loss / perplexity：

#### Small (124M)
- Baseline: loss $2.85426$, perplexity $17.3616$
- DDL ($d_v=1$): loss $2.84817$, perplexity $17.2562$
- **DDL ($d_v=4$): loss $2.83545$, perplexity $17.0381$**

#### Medium (353M)
- Baseline: loss $2.60532$, perplexity $13.5356$
- DDL ($d_v=1$): loss $2.60388$, perplexity $13.5161$
- **DDL ($d_v=4$): loss $2.59267$, perplexity $13.3654$**

可以看出，DDL 在两个规模上都优于 baseline，而且 expanded-state 的 $d_v=4$ 效果最好。

### 2. 下游 1-shot benchmark 也有稳定提升

在 small 模型上，baseline 平均分是 **48.56**；

- DDL ($d_v=1$): **48.73**
- DDL ($d_v=4$): **48.91**
- DDL-CC: **49.13**
- **DDL-EC: 49.47**
- DDL-CC-EC: **49.29**

在 medium 模型上：

- Baseline: **53.96**
- DDL ($d_v=1$): **54.69**
- **DDL ($d_v=4$): 54.83**

也就是说，DDL 不只是把训练 loss 压低了一点，而是能迁移到 ARC、HellaSwag、OpenBookQA、PIQA、SciQ、Social IQA、WinoGrande 等泛化评测上。

### 3. 变体结果：expanded state + 卷积还能继续增益

论文还测了若干 DDL 变体：

- **DDL-CC**：沿 $d_v$ 维做 channel convolution；
- **DDL-EC**：在 embedding 后沿 sequence length 做 convolution；
- **DDL-CC-EC**：两者叠加。

结论是：这些变体在某些设置上进一步提升表现，说明 DDL 的收益并不只来自一个单点技巧，而可能和更丰富的 state parameterization 搭配。

## 分析与讨论

### 1. DDL 的核心不是“更复杂的残差”，而是“shortcut 终于可以不是 identity”

标准残差层的 shortcut 太强了：它保证了稳定训练，但也固定了层间信息流的几何结构。DDL 的价值在于，它只用一个非常低秩的 rank-1 operator，就把 shortcut 从“永远不动”变成“可选择地保留、删除、翻转、再写入”。

这比直接大改 backbone 更有意思，因为它针对的是 Transformer 里一个非常基础、但通常被视为不可动的结构。

### 2. 这个工作和 memory / state-space 思路是相通的

如果从 repo 常见的“模型如何维护、擦除、重写内部状态”视角来看，DDL 其实在做一件很像 memory model 的事：

- 先沿某个方向 read 当前状态；
- 再决定 erase 多少；
- 最后沿同方向 write 新内容。

只不过它把这件事放在深度维度，而不是时间维度或外部 memory 里。这也是论文反复强调它和 DeltaNet 同构的原因。

### 3. 论文最强的点在“结构可解释性”

很多新架构 paper 只是经验上有效，但 DDL 的一个优点是：其核心 operator 的谱完全可分析。你能明确知道 gate $\beta$ 在控制什么，reflection / projection / identity 三种模式如何统一，以及何时会出现 sign flip、何时会近似 skip。这个可解释性比单纯“训练更稳、分数更高”更难得。

### 4. 经验收益目前还是中等幅度，不是颠覆式大跳跃

从结果上看，DDL 的提升是稳定但温和的，而不是像某些 scaling / data recipe 改动那样带来大幅跃迁。这意味着它更像一种值得叠加的 architectural prior，而不是单独就足以重写 SOTA 的方法。

## 局限性

1. **实验规模还不大**：只验证到 353M 参数，没有证明在更大语言模型上仍能保持同样收益。
2. **收益幅度偏稳健而非巨大**：虽然 loss / perplexity / benchmark 都提升，但提升量多数是 incremental 的，因此还需要看它是否能和更大模型、更多训练 token 共同放大。
3. **引入了额外状态结构与分支设计**：DDL 虽是 drop-in replacement，但实际上需要额外的 $\mathbf{k}$、$\mathbf{v}$、$\beta$ 分支，以及可能的 expanded-state 读写逻辑，实现复杂度高于普通 residual add。
4. **目前主要验证在语言建模上**：虽然论文动机来自一般 deep residual network，但实验重心集中在 LM Transformer，没有展示在 vision / multimodal / long-context memory 等更广场景的效果。

## 关键洞察

1. **残差网络的 shortcut 本身也应该是可学习对象，而不只是默认 identity。**
2. **用低秩、可分析的几何 operator 改造 shortcut，可能是一条比“继续堆更大 MLP/attention”更干净的架构路线。**
3. **DDL 的本质是一个 depth-wise delta rule：它把“读-擦-写”机制引入了层间状态演化。**
4. **把残差状态扩展成矩阵 $\mathbf{X} \in \mathbb{R}^{d \times d_v}$，等于把 residual stream 从单通道提升成轻量 memory substrate，这个想法本身很值得继续追。**
5. **对当前 repo 的阅读脉络来说，这篇论文补的是“架构层面的状态重写”视角：不是只靠更好的数据、训练或 RL，而是直接改变 hidden state 在深度上的更新几何。**
