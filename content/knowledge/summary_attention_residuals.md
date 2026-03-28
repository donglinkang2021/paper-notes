---
title: "Attention Residuals"
authors: "Kimi Team"
institution: "Moonshot AI"
venue: "Technical Report 2026"
arxiv_id: "2603.15031"
tags: ["architecture", "residual connection", "attention over depth", "transformer", "scaling law", "reasoning", "kimi"]
---

# Attention Residuals：把残差连接改成沿深度做注意力聚合

## 核心贡献

1. **从“时间上的 attention”推广到“深度上的 attention”**：论文认为标准残差连接把所有历史层输出都以固定权重 1 累加到当前 hidden state 中，这相当于把跨层信息压缩进一个单一状态里，既不能 selective access，也会造成 hidden-state magnitude 随深度无控制增长。作者由此提出：既然 Transformer 用 attention 解决了时间维的 RNN bottleneck，那么也可以用 attention 解决深度维的 residual bottleneck。
2. **提出 Attention Residuals (AttnRes)**：当前层不再只接收 $\bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$，而是对所有先前层输出做 $\operatorname{softmax}$ attention，根据 learned weights 选择性聚合前面各层的表示。
3. **提出 Block AttnRes**：为解决 full attention over depth 的 memory / communication 开销，把层划分成 block，在 block 内累加、在 block 间做 attention，把开销从 $O(Ld)$ / $O(L^2)$ 压到 $O(Nd)$ / $O(N^2)$，同时保留大部分收益。
4. **补齐大规模训练与推理基础设施**：除了模型本身，论文还给出 cross-stage caching、two-phase computation、memory-efficient prefilling 等系统优化，使 Block AttnRes 成为大模型训练中的可落地 drop-in replacement。
5. **在 Kimi Linear 上做大规模验证**：把 AttnRes 集成进 Kimi Linear 架构，在 1.4T token 预训练后，相比 baseline 在 general reasoning、math、code、中文评测上全部持平或更优，尤其对多步推理类任务增益明显。

## 方法详解

### 1. 动机：标准 residual 的三个问题

论文把标准 residual 写成：

$$
\bm{h}_{l} = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})
$$

展开后有：

$$
\bm{h}_l = \bm{h}_1 + \sum_{i=1}^{l-1} f_i(\bm{h}_i)
$$

这带来几个问题：

1. **No selective access**：attention 层和 MLP 层都只能接收到同一个压缩后的聚合状态，不能按需查看更早的特定层输出；
2. **Irreversible loss**：一旦信息在累加中被“淹没”，后续层无法再直接取回某个历史层的独立表示；
3. **Output growth / PreNorm dilution**：因为所有历史输出都在累积，越深层越需要输出更大幅度的激活才能“发声”，这会造成 hidden-state magnitude 增长和梯度分布失衡。

作者把这个问题和 RNN over time 的瓶颈做类比：RNN 只有一个递归 state，而 Transformer 用 attention 让每个位置都能访问所有历史 token；同理，深层网络也可以让每一层访问所有历史层表示。

### 2. Full Attention Residuals

AttnRes 的核心公式是：

$$
\bm{h}_{l} = \alpha_{0 \to l} \cdot \bm{h}_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} \cdot f_i(\bm{h}_i)
$$

其中 $\alpha_{i \to l}$ 是跨深度的 attention weights，满足：

$$
\sum_{i=0}^{l-1} \alpha_{i \to l} = 1
$$

作者采用基于 query / key 的 softmax attention：

$$
\alpha_{i \to l} =
\frac{\phi(\bm{q}_l, \bm{k}_i)}{\sum_{j=0}^{l-1} \phi(\bm{q}_l, \bm{k}_j)}
$$

其中：

$$
\phi(\bm{q}, \bm{k}) = \exp\big(\bm{q}^\top \operatorname{RMSNorm}(\bm{k})\big)
$$

并定义：

$$
\bm{q}_l = \bm{w}_l,
\qquad
\bm{k}_i = \bm{v}_i =
\begin{cases}
\bm{h}_1 & i = 0 \\
f_i(\bm{h}_i) & 1 \le i \le l-1
\end{cases}
$$

这里的 query $\bm{w}_l$ 是每层一个可学习向量，而不是从当前 token 动态生成。这个设计很重要，因为它让 attention weights 可以在 block 内并行预计算，不必等待该层真正 forward 完成。

最终每层输入是：

$$
\bm{h}_{l} = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \bm{v}_i
$$

直观上，这相当于把 residual connection 从“永远只看上一层 + 固定权重 1”改成“对所有历史层做按内容选择的深度注意力”。

### 3. Block AttnRes：工程上可用的折中

Full AttnRes 的问题在于，虽然深度 $L$ 比序列长度短很多，但在大规模分布式训练里仍会引入额外 memory / communication overhead。于是论文提出 Block AttnRes：

- 把 $L$ 层划分成 $N$ 个 block；
- block 内把层输出求和成一个 block representation：
  $$
  \bm{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\bm{h}_j)
  $$
- 每层只对前面 block representations 和当前 block 的 partial sum 做 attention。

这样：

- memory 从 $O(Ld)$ 降到 $O(Nd)$；
- computation 从 $O(L^2)$ 降到 $O(N^2)$；
- 当 $N=L$ 时恢复 Full AttnRes；
- 当 $N=1$ 时退化回标准 residual（外加 embedding source）。

论文经验上发现：**固定大约 8 个 blocks 已能恢复大部分收益**。

### 4. 系统优化：让它真的能训大模型

这篇论文不仅给模型，还给了完整 infra 设计，这一点很关键。

#### 训练侧：Cross-stage caching

在 pipeline parallelism 下，naive 实现需要把所有已累计的 block representations 反复跨 stage 传递，通信代价很高。作者的做法是在 physical stage 本地缓存 earlier virtual stages 已收到的 block，只传增量块，从而把 peak per-transition cost 从 $O(C)$ 降到 $O(P)$，wall-clock overhead 控制在 **4\% 以下**。

#### 推理侧：Two-phase computation

对每个 block：

1. **Phase 1**：把该 block 内所有层的 pseudo-query 一次性批量拿去和前面 blocks 做 inter-block attention；
2. **Phase 2**：顺序处理当前 block 内的 intra-block partial sum，并通过 online softmax merge 把两部分结果合并。

这样把 per-layer I/O 大幅摊薄，最终实际 inference latency overhead 控制在 **2\% 以下**。

#### 长上下文 prefilling

对 128K context，block representations cache 可能很大。作者通过按 sequence dimension 在 tensor-parallel devices 上分片，把一个示例中的 per-device overhead 从 15GB 降到约 1.9GB，再配合 chunked prefill 可降到 0.3GB 以下。

## 实验结果

### 1. Scaling law：全尺度持续优于 baseline

论文在五个模型规模上做 scaling law sweep，比较 Baseline、Block AttnRes（$N \approx 8$）、Full AttnRes 和 mHC(-lite)。

例如在 largest compact setting：

- Baseline val loss: **1.719**
- Block AttnRes: **1.693**
- Full AttnRes: **1.692**
- mHC(-lite): **1.694**

拟合得到的 scaling curves 也显示，AttnRes 与 baseline 斜率接近，但在整个 compute 区间上都系统性更低。作者估计在 5.6 PFLOP/s-days 时，Block AttnRes 相比 baseline 相当于 **1.25× compute advantage**。

### 2. 1.4T-token 预训练后的主结果

作者把 Block AttnRes 集成到 Kimi Linear 48B（3B activated）架构中，按 Kimi Linear 同样 recipe 训练：

- 1T tokens WSD pretraining
- 再加约 400B high-quality tokens mid-training
- 后续扩展到 32K context

在 downstream benchmarks 上，AttnRes **全部持平或更优**：

#### General
- MMLU: **73.5 → 74.6**
- MMLU-Pro: **52.2 → 52.2**
- GPQA-Diamond: **36.9 → 44.4**
- BBH: **76.3 → 78.0**
- ARC-Challenge: **64.6 → 65.7**
- HellaSwag: **83.2 → 83.4**
- TriviaQA: **69.9 → 71.8**

#### Math & Code
- GSM8K: **81.7 → 82.4**
- MGSM: **64.9 → 66.1**
- Math: **53.5 → 57.1**
- CMath: **84.7 → 85.1**
- HumanEval: **59.1 → 62.2**
- MBPP: **72.0 → 73.9**

#### Chinese
- CMMLU: **82.0 → 82.9**
- C-Eval: **79.6 → 82.5**

其中最明显的提升集中在：

- **GPQA-Diamond +7.5**
- **Math +3.6**
- **HumanEval +3.1**

论文据此认为，AttnRes 特别有利于需要多步组合、需要后层回看前层表示的任务。

### 3. 训练动态分析：缓解 PreNorm dilution

训练曲线分析给出三个很关键的现象：

1. **Validation loss 全程更低**；
2. **Output magnitude 更平稳**：baseline 的 hidden-state magnitude 随深度单调增大，而 Block AttnRes 在 block 边界会“重置”累积，形成更有界的周期模式；
3. **Gradient magnitude 更均匀**：baseline 最早层梯度过大，而 AttnRes 的 softmax competition 让梯度在不同深度分布更平衡。

这说明 AttnRes 不是只在最终指标上稍微占优，而是在优化动力学上确实改变了深层网络的信息流。

### 4. 消融：内容相关选择比固定 mixing 更重要

若把深度聚合改成 input-independent scalar mixing，性能会下降；把 softmax 改成 sigmoid，也会变差；multihead depth aggregation 反而不如单一 depth mixture。这些结果共同支持一个结论：**depth-wise source selection 需要竞争式、内容相关、层整体级别的选择，而不是简单固定混合或 per-head 分散选择。**

## 分析与讨论

### 1. 这篇论文本质上在做“depth Transformer”

最核心的抽象是：

- RNN 的瓶颈在时间维；Transformer 用 attention 解决；
- residual network 的瓶颈在深度维；AttnRes 用 attention 解决。

这个 time-depth duality 非常漂亮，而且比很多 residual generalization 方法更清楚：它不是加一些 gating 或 dense skip，而是明确说“每层就应该像 token 一样，成为可以被选择性读取的 source”。

### 2. 它修复的是 LLM 中一个很真实的结构问题

论文抓住了一个在 PreNorm LLM 中越来越明显的问题：随着深度增加，后层要影响最终表示就必须输出更大幅度的激活，导致早层贡献被稀释。AttnRes 的改进点不是增加模型容量，而是让深层可以重新分配来自 earlier layers 的权重，从而避免盲目累加。

### 3. Block 设计是这篇论文落地的关键

Full AttnRes 的思想很直接，但真正让它可训练、可部署的是 Block AttnRes + infra optimization。没有这一层，论文可能只是一个很美的概念；有了 block、caching、two-phase inference，它才从“理论上可行”变成“Moonshot 真能在 48B 模型上训出来”。

### 4. 对当前 repo 的意义：它提供了一条“非 token 维度 attention 化”的路线

如果把这篇和 repo 里常见的 reasoning / RL / architecture 论文放一起看，它特别有价值的一点是：**attention 不一定只用在 token 维，也可以用在深度维、memory 维、trajectory 维。**

这和很多“让模型自己选择哪些历史信息重要”的工作属于同一大脉络，只不过这里选择对象从 past tokens 换成了 past layers。

## 局限性

1. **Full AttnRes 仍有较重开销**：虽然 depth 比 sequence 短，但 full version 在大规模 pipeline parallel 训练里仍会带来显著 memory / communication 压力，因此必须依赖 Block variant。
2. **增益依赖于强工程实现**：cross-stage caching、two-phase computation、prefill sharding 这些都是必要条件，意味着它并不是“改几行模型代码就能白拿收益”的方法。
3. **当前实验主要基于 Kimi / MoE / hybrid attention recipe**：虽然 small-scale scaling law 结果较全面，但最大规模主结果绑定在 Kimi Linear 上，跨更多 backbone 的普适性还需要进一步验证。
4. **更深模型未必是部署最优**：论文发现 AttnRes 更偏好 deeper / narrower 架构，但更深通常也意味着更高推理延迟，因此训练最优与部署最优之间仍需折中。

## 关键洞察

1. **残差连接的问题不只是“权重固定”，而是“每层只能看到一个被压缩的深度状态”。**
2. **把层输出当成可被 attention 访问的 memory source，是对 residual learning 的自然推广。**
3. **在深度维做 selective aggregation，能同时改善训练稳定性、梯度分布和多步推理表现。**
4. **Block 压缩说明：要取得这些收益，并不一定需要精确保留每一层；保留少量 block-level summaries 就足够恢复大部分价值。**
5. **对当前 repo 的阅读脉络而言，这篇论文很重要，因为它把“attention beyond sequence”具体化为一个已在大模型上验证的深度路由机制。**
