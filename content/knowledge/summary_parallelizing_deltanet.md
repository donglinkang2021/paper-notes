---
title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
authors: "Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim"
institution: "Massachusetts Institute of Technology, Soochow University, MIT-IBM Watson AI Lab"
venue: "NeurIPS 2024"
arxiv_id: "2406.06484"
tags: ["linear transformer", "deltanet", "delta rule", "sequence parallelism", "associative recall", "hardware efficiency", "hybrid attention"]
---

# 并行化 DeltaNet：让带 delta rule 的线性 Transformer 能按序列维高效训练

## 核心贡献

1. **解决 DeltaNet 的核心工程瓶颈**：DeltaNet 早已被证明比普通 linear transformer 更擅长 associative recall / in-context retrieval，但其训练实现本质上是严格按时间步递归的，无法沿 sequence length 并行，因此在现代 GPU 上训练效率很差。本文的首要贡献，就是给出一个真正能在序列维并行训练 DeltaNet 的硬件友好算法。
2. **把 DeltaNet 重写成 memory-efficient 的 pseudo-value 形式**：作者观察到 DeltaNet 虽然递推依赖前一时刻状态，但仍可重写成和线性注意力类似的 additive form，只不过 value 从原始 $\vv_t$ 变成“伪值” $\vu_t$。这是后续 chunkwise parallelization 的关键。
3. **利用 Householder matrix product 的 WY / UT 表示推导 chunkwise DeltaNet**：论文借助 generalized Householder transforms 的结构，把本来需要显式 materialize $d \times d$ state 的递推，改写成只需 $O(d)$ memory 就能构造 chunk 内更新量，从而得到 chunkwise parallel form。
4. **将 DeltaNet 扩展到现代语言建模规模**：借助新算法，作者训练了 1.3B / 100B-token 的 DeltaNet，并显示其在 perplexity 与多个零样本任务上优于 Mamba、GLA 等线性时间 baselines。
5. **验证 hybrid architecture 很有效**：在纯 DeltaNet 之外，作者还测试了混合模型——加入 sliding-window attention 或只保留两个 global attention 层——发现这类 hybrid 甚至能超过强 Transformer baseline。

## 方法详解

### 1. 背景：为什么 DeltaNet 比普通 linear attention 更有记忆力

普通 linear transformer 的核心状态更新是：

$$
\rmS_t = \rmS_{t-1} + \vv_t \vk_t^\top,
\qquad
\vo_t = \rmS_t \vq_t
$$

这其实就是不断把新的 key-value association 加到 memory $\rmS_t$ 中。但纯加法会让 memory 随时间堆积，难以删除旧关联；当 context 很长时，容易出现 key collision 和记忆容量不足。

DeltaNet 用 delta rule 替代纯加法：

$$
\rmS_t = \rmS_{t-1} - \beta_t (\rmS_{t-1}\vk_t - \vv_t) \vk_t^\top
$$

也可写成：

$$
\rmS_t = \rmS_{t-1}(\rmI - \beta_t \vk_t \vk_t^\top) + \beta_t \vv_t \vk_t^\top
$$

其中：

- $\rmS_{t-1}\vk_t$ 是当前 key 对旧 memory 的读出；
- $\vv_t$ 是希望写入的新目标值；
- 二者之差就是 delta / prediction error；
- $\beta_t \in (0,1)$ 控制写入强度。

这相当于在线地对 loss

$$
\mathcal{L}_t(\rmS) = \tfrac{1}{2}\|\rmS\vk_t - \vv_t\|^2
$$

做一步 SGD 更新。直观上，DeltaNet 不只是“加一条新记忆”，而是会基于当前 key 主动擦除旧值、再写入新值，因此 recall capacity 更强。

### 2. 难点：DeltaNet 为什么不好并行

普通 linear attention 可以改写成 fully parallel 或 chunkwise parallel 形式，因为 value vectors $\vv_t$ 是已知的，序列上的很多计算都可重排成 matmul。

但 DeltaNet 的麻烦在于：更新里需要先算旧值

$$
\vv_t^{\text{old}} = \rmS_{t-1}\vk_t
$$

而这依赖前一时刻的 state，因此不能像 vanilla linear attention 那样直接把所有 token 的 value 一次性并行出来。

### 3. 核心技巧一：Pseudo-value 重参数化

作者先证明 DeltaNet 的 state 其实可以写成：

$$
\rmS_t = \sum_{i=1}^{t} \vu_i \vk_i^\top
$$

其中

$$
\vu_i = \beta_i (\vv_i - \vv_i^{\text{old}})
$$

也就是把原始 value $\vv_i$ 替换为 pseudo-value $\vu_i$。这样一来，一旦 $\vu_i$ 被构造出来，后面的输出计算就和普通 linear attention 一样：

$$
\rmO = (\rmQ \rmK^\top \odot \rmM)\rmU
$$

其中 $\rmU$ 是所有 $\vu_i$ 组成的矩阵。

问题仍在于：如何高效算出全部 $\vu_i$，而不显式 materialize 每步的 $\rmS_t \in \mathbb{R}^{d \times d}$？

### 4. 核心技巧二：WY / UT 表示与 chunkwise parallel form

论文借助 generalized Householder matrix products 的 WY representation，推导出 chunk 内的状态转移可以用两个 rank-1 累加对象表示：

$$
\rmP_{[t]}^{r} = \rmI - \sum_{i=1}^{r}\vw_{[t]}^i\vk_{[t]}^{i\top},
\qquad
\rmH_{[t]}^{r} = \sum_{i=1}^{r}\vu_{[t]}^i\vk_{[t]}^{i\top}
$$

并给出递推：

$$
\vw_{[t]}^r = \beta_{[t]}^r \left(\vk_{[t]}^r - \sum_{i=1}^{r-1} \vw_{[t]}^i (\vk_{[t]}^{i\top}\vk_{[t]}^r)\right)
$$

$$
\vu_{[t]}^r = \beta_{[t]}^r \left(\vv_{[t]}^r - \sum_{i=1}^{r-1} \vu_{[t]}^i (\vk_{[t]}^{i\top}\vk_{[t]}^r)\right)
$$

这样 chunk 内虽然仍有递归结构，但只需维护 $O(d)$ 向量，不需要维护完整 $d \times d$ state。

接着，作者进一步用 UT transform 把很多递归步骤重写成矩阵运算：

$$
\rmT_{[t]} = \left(\rmI + \operatorname{tril}(\operatorname{diag}(\beta_{[t]})\rmK_{[t]}\rmK_{[t]}^\top,-1)\right)^{-1} \operatorname{diag}(\beta_{[t]})
$$

$$
\rmW_{[t]} = \rmT_{[t]}\rmK_{[t]},
\qquad
\rmU_{[t]} = \rmT_{[t]}\rmV_{[t]}
$$

最终得到 chunk-level recurrence：

$$
\rmS_{[t+1]} = \rmS_{[t]} + (\rmU_{[t]} - \rmW_{[t]}\rmS_{[t]}^\top)^\top \rmK_{[t]}
$$

$$
\rmO_{[t]} = \rmQ_{[t]}\rmS_{[t]}^\top + (\rmQ_{[t]}\rmK_{[t]}^\top \odot \rmM)(\rmU_{[t]} - \rmW_{[t]}\rmS_{[t]}^\top)
$$

于是 DeltaNet 就被改写成了类似 chunkwise linear attention 的形式：

- chunk 间顺序传状态；
- chunk 内用大量 matmul 并行；
- backward 时再重算 hidden state 以省显存。

这正是论文所谓“hardware-efficient”的关键：不是只看 FLOPs，更要让运算富含 matmul、能吃到 tensor cores 和高 GPU occupancy。

### 5. DeltaNet Transformer 与 hybrid 模型

在模型层面，作者基本沿用 LLaMA-style Transformer++，只把自注意力层替换成 DeltaNet layer，并在 q/k 上使用：

$$
\vk_t = \frac{\operatorname{SiLU}(\rmW_K\vx_t)}{\|\operatorname{SiLU}(\rmW_K\vx_t)\|_2},
\qquad
\vq_t = \frac{\operatorname{SiLU}(\rmW_Q\vx_t)}{\|\operatorname{SiLU}(\rmW_Q\vx_t)\|_2}
$$

作者强调：

- 用 **SiLU** 比 ELU+1 更好；
- 用 **$L_2$ normalization** 比原始 DeltaNet 的 $L_1$ normalization 更好；
- $L_2$ 归一化让 $\rmI - \beta_t \vk_t \vk_t^\top$ 的谱解释更自然：当 $\beta_t = 1$ 时，正好是对某一子空间做 projection，利于 targeted forgetting。

此外还测试了两类 hybrid：

1. **Sliding window attention**：和 DeltaNet 层交替；
2. **Global attention**：仅把第 2 层和中间层替换成全局 softmax attention。

这是因为纯线性 / 递归模型在局部精确 token comparison 和显式位置信息上仍有弱点，而少量 softmax attention 可以弥补这个短板。

## 实验结果

### 1. 合成任务：DeltaNet 的 recall 能力确实更强

在 MQAR、MAD、RegBench 等 synthetic tasks 上，DeltaNet 明显优于普通 linear transformer / Mamba 等线性时间基线，特别是在 associative recall 类任务上优势最明显。作者将此视为 delta rule 提升 memory capacity 的直接证据。

### 2. 340M / 15B-token 语言建模

在 340M 设置上：

- **WikiText perplexity**：DeltaNet (w. conv) 为 **28.24**，优于 GLA w. conv 的 29.47，也优于自身无卷积版本 29.08；
- **LAMBADA ppl / acc**：37.37 / 32.1，优于 baseline 与 GLA；
- 零样本平均分（PiQA/Hella/Wino/ARC 等）达到 **42.1**，略高于其他线性时间基线；
- recall-intensive tasks（SWDE / SQuAD / FDA）上，纯 DeltaNet 也优于 GLA，说明 delta rule 的确提高了在固定 state size 下的检索能力。

加入 hybrid 后效果更强：

- **DeltaNet + Sliding Attn**：Wiki ppl 降到 **27.06**；
- **DeltaNet + Global Attn (2 layers)**：LAMBADA ppl 降到 **35.04**，SWDE/FDA 也进一步显著提升。

### 3. 1.3B / 100B-token 主结果

论文最重要的 real-world 结果是在 1.3B 模型上：

- **Wiki ppl**：Transformer++ 16.85，DeltaNet 16.87，几乎持平；
- **LAMBADA ppl**：DeltaNet **12.21**，优于 Transformer++ 的 13.44、GLA 的 14.47、Mamba 的 13.89；
- 零样本平均分：DeltaNet **51.6**，高于 GLA 的 51.0 和 Mamba 的 50.0；
- ARC-c 上 DeltaNet 达到 **28.3**，略优于其他线性时间模型。

更重要的是 hybrid：

- **DeltaNet + Sliding Attn** 平均分 **52.1**，已经超过 Transformer++ 的 50.9；
- **DeltaNet + Global Attn (2 layers)** 在 SWDE / SQuAD / FDA 等 retrieval-heavy benchmarks 上尤其强，例如 SWDE 达到 **71.0**，FDA 达到 **29.8**，都超过纯 Transformer++。

### 4. 3B / 1T-token 扩展

作者还把 DeltaNet 扩展到 3B / 1T-token，并与 Llama-3.2-3B、PowerLM-3B 及若干 recurrent baselines 比较：

- DeltaNet-3B 平均分 **59.8**；
- 虽然低于 Transformer-based 的 Llama-3.2-3B（62.8）和 PowerLM-3B（62.3），但仍明显优于 RecurrentGemma-2B（57.9）、RWKV-6-3B（54.9）和 Mamba-2.7B（53.3）。

也就是说，纯 DeltaNet 在更大规模上还没完全追平强 Transformer，但已经站稳了“最强 recurrent / linear-time baselines”一档。

### 5. 训练吞吐

论文还强调：新的 chunkwise DeltaNet kernel 训练速度显著快于原始 recurrent 实现；在 1.3B 设置上，训练速度接近 GLA，并明显快于 Mamba。对长序列训练而言，所有 linear-time 模型都优于 Transformer，而新的 DeltaNet 并行算法让它终于具备了这种实际可训练性。

## 分析与讨论

### 1. 这篇论文真正重要的是“把更强的记忆更新规则变得可训练”

DeltaNet 的 idea 其实并不新：用 delta rule 替代加法更新来提升 associative recall，早有理论和小规模实验支持。真正卡住它的是工程：序列维无法并行，现代 GPU 上不划算。

这篇论文的关键价值在于：**它不是提出一种更强的 recurrent rule，而是把一个已知更强的 rule，变成了一个现代硬件上真的能规模化训练的 layer。**

### 2. 核心 insight：structured matrix recurrence 仍然可以 chunkwise parallelize

Mamba / GLA 这类模型之所以好并行，部分原因是 recurrence 足够 elementwise；DeltaNet 的 recurrence 涉及 state-to-state interactions，看似更难并行。论文展示：只要 transition matrix 有足够结构（这里是 generalized Householder / rank-1 update），就仍然可以构造 chunkwise algorithm。

这意味着“可并行”和“高表达力”之间并不一定只有 elementwise recurrence 这一条路。

### 3. Hybrid 结果说明：DeltaNet 很强，但 softmax attention 仍然有补位价值

纯 DeltaNet 在 recall 和语言建模上已经很强，但真正超过 Transformer baseline 的是 hybrid 版本。这很说明问题：

- DeltaNet 提供强 memory update / associative retrieval；
- softmax attention 提供精确局部比较、全局路由和位置信息处理；
- 二者并不是非此即彼，而是可以互补。

这和后续很多“少量 global attention + 大量线性时间层”的设计方向是一致的。

### 4. 对当前 repo 的意义：它把 delta rule 从“记忆理论”推进到“现代 LLM 基础设施”

如果把这篇放进你 repo 当前的阅读脉络里，它的独特价值是：

- 它不是再讲“如何让模型用 RL 自我改进”；
- 也不是纯 architecture intuition paper；
- 而是一个非常典型的 **algorithmic systems paper**：把一个在记忆能力上更好的 primitive，重新包装成 matmul-rich、GPU-friendly 的实现。

这对于理解后来的 Kimi Delta Attention、Gated DeltaNet、乃至更广泛的 sequence model systems design 都很关键。

## 局限性

1. **训练速度仍不如最简单的 elementwise recurrence 模型**：尽管新算法比旧 DeltaNet 快很多，但论文明确承认其速度仍落后于 GLA，因为 DeltaNet 需要处理 intra-state dependencies，kernel 更复杂。
2. **state size scalability 仍有问题**：在 recall-intensive tasks 上，state size 很关键，而 DeltaNet 当前 kernel 受 head dimension / state size 扩展限制，这会拖累大规模设置下的 recall 表现。
3. **长度泛化不如带显式 decay 的模型**：作者发现 DeltaNet 的 extrapolation beyond training length 不如 GLA / RetNet / Mamba，推测是因为它没有显式 decay factor。
4. **纯 DeltaNet 在更大规模上仍未完全追平 Transformer**：3B 结果显示它虽优于其他 recurrent baselines，但仍弱于强 Transformer 基线，说明它还不是全面替代 softmax Transformer 的终点方案。

## 关键洞察

1. **delta rule 确实比简单 additive memory update 更适合 retrieval-heavy / associative recall 任务。**
2. **难点从来不是“有没有更好的 recurrence”，而是“能否把它变成 matmul-rich、sequence-parallel 的训练算法”。**
3. **Householder / low-rank structured transitions 是一类很值得重视的中间地带：比 elementwise recurrence 更有表达力，又不像全矩阵 recurrence 那样贵得不可用。**
4. **Hybrid architecture 很可能是现实中的最优路线：用 DeltaNet 提供线性时间记忆更新能力，再用少量 softmax attention 补足局部精确比较和全局路由。**
5. **对当前 repo 来说，这篇论文是理解“delta-rule memory → Kimi Delta / modern linear attention”演化链条中的关键一站。**
