---
title: "Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models"
authors: "Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, Mengdi Wang"
institution: "Princeton University, University of Chicago"
venue: "arXiv 2025"
arxiv_id: "2509.06949"
tags: ["diffusion language model", "reinforcement learning", "trajectory-aware training", "reasoning", "math", "coding", "block diffusion", "value model"]
---

# TraceRL：面向扩散语言模型的轨迹感知强化学习框架

## 核心贡献

1. **指出 DLM 后训练目标与真实推理轨迹之间的错配**：现有 diffusion language model (DLM) 的后训练通常沿用 fully random masking 目标，但实际推理却是依赖静态/动态 unmask 轨迹、KV-cache 和 block-wise generation 的。论文认为，这种“训练目标不看轨迹、推理过程高度依赖轨迹”的错位，是当前 DLM 强化学习效果受限的根源。
2. **提出 TraceRL**：一个直接对推理轨迹进行优化的 RL 框架。它不只对最终完成的 response 给奖励，而是把 rollout 过程中产生的 trajectory 作为训练对象，使策略更新更贴近 DLM 的真实采样过程。
3. **提出 diffusion-based value model**：把 value estimation 扩展到 diffusion 轨迹上，为每个 trace step / token 提供 advantage baseline，降低方差并稳定训练。
4. **同时适配 full-attention 与 block-attention DLM**：TraceRL 不只是某一种 diffusion 架构的 patch，而是试图成为统一 RL 后训练框架。论文同时展示了它对 full-attention DLM、block diffusion model、数学任务和代码任务的适用性。
5. **给出一条“扩散模型也能做长推理”的路线**：在 TraceRL 基础上叠加 long-CoT SFT，作者训练出 TraDo-8B-Thinking，并将其定位为首个 long-CoT diffusion language model。
6. **开源完整训练与部署框架**：不仅给方法，还同步发布 dLLM-RL 框架，覆盖训练、推理加速、KV-cache、不同 diffusion 架构与多种 post-training 算法实现。

## 方法详解

### 1. 背景：为什么 random masking 目标不够

论文先回顾 masked DLM 的基本训练目标。给定原始序列 $x_0$ 和被 mask 后的 $x_t$，其 forward corruption 写作：

$$
q(x_t \mid x_0) = \prod_{i=1}^{n} \mathrm{Cat}\left(x_t^i; (1 - t) \delta_{x_0^i} + t \delta_{\text{[MASK]}}\right)
$$

对应训练目标是：

$$
\mathcal{J}_{\mathrm{full}}(x_0, Q, \theta) = \int_{0}^{1} \frac{1}{t |x_0|} \,
\mathbb{E}_{q(x_t \mid x_0)} \left[
\sum_{i: x_t^i = \text{[MASK]}}
\log p_{\theta}(x_0^i \mid x_t, Q)
\right] dt
$$

这个目标适合 pretraining，但后训练阶段存在两个问题：

- 语言生成本质上依赖前文，不是完全随机恢复；
- 实际推理会采用 **static sampling** 或 **dynamic sampling**，并借助 KV-cache / block-wise generation 加速。

因此，训练时的 fully random masking 与推理时的 preferred inference trace 并不一致。

### 2. 从 semi-AR SFT 到 preferred trace

作者先做一个中间实验：把后训练目标从 fully random masking 改为更贴近 left-to-right 轨迹的 semi-autoregressive 目标：

$$
\mathcal{J}_{\text{semi}}(x, Q, \theta) = \sum_{i = 1}^{\lceil L / B \rceil} \mathcal{J}_{\text{full}}\big(x^{(i - 1)B:\min(iB, L)}, [Q, x^{0:(i - 1)B}], \theta\big)
$$

结果显示：即便在相同计算量下，沿着更符合真实生成顺序的目标训练，性能也明显优于 fully random masking。进一步地，作者直接收集模型自己的 preferred inference traces 做 finetuning，效果又优于普通 semi-AR 和 fully random baselines。

这里的关键结论是：**对 DLM 来说，后训练应该对齐模型真实采样轨迹，而不是只对齐最终 token reconstruction。**

### 3. TraceRL：直接优化 rollout trajectory

在 RL 阶段，给定任务 $Q$，策略 $\pi_\theta$ 生成 response 的 trajectory：

$$
\tau_i \triangleq (\tau_i(1), \dots, \tau_i(|\tau_i|))
$$

其中 $\tau_i(t)$ 是第 $t$ 个解码 step 被恢复的 token 集合。TraceRL 不只对最终答案给 reward，而是让策略更新显式依赖整个 trajectory。

作者引入 shrinkage parameter $s$，把相邻 $s$ 个 step 聚合成更粗粒度的 trace，以降低 full-attention 模型训练成本：

$$
\tau_i^s(k) \triangleq \bigcup_{j = s(k - 1) + 1}^{\min(sk, |\tau_i|)} \tau_i(j),
\qquad
|\tau_i^s| = \lceil |\tau_i|/s \rceil
$$

对应策略目标为：

$$
\mathcal{J}_{\mathrm{policy}}(\theta_p) =
\mathbb{E}
\Big[
\sum_{i = 1}^{G} \sum_{t = 1}^{|\tau_i^s|}
\sum_{o_k \in \tau_{i,t}^s}
C_{\epsilon}\!\left(
\frac{\pi_{\theta_p}(o_k \mid \tau_i^s(1:(t-1)))}{\pi_{\mathrm{old}}(o_k \mid \tau_i^s(1:(t-1)))},
A_i
\right)
/ |\tau_i^s(t)|
\Big]
- \beta \, \mathbb{KL}[\pi_\theta \Vert \pi_{\mathrm{old}}]
$$

其中 $C_\epsilon$ 是 PPO-style clipped objective。直观上，它把 RL 优化从“奖励整个完成结果”改成“奖励生成轨迹上的决策序列”。

### 4. Diffusion value model

为了降低训练波动，论文又为 diffusion trajectory 定义了 value model。对每个 trace step 上的 token，先估计 token-wise value，再聚合成 step-wise value 与 GAE advantage。value network 的训练目标是 clipped regression：

$$
\mathcal{J}_{\mathrm{value}}(\theta_v) = \tfrac{1}{2}\,\mathbb{E}_{\tau}\Big[\tfrac{1}{|\tau|}\sum_{j \in \tau} \max\big((V_{\theta_v}(\tau)_j - R_j)^2,\; (V_j^{\mathrm{clip}} - R_j)^2\big)\Big]
$$

这部分的意义不是“把 PPO 原样搬过来”，而是针对 diffusion rollout 的结构，把 sequence-level reward 分解成更细粒度、prefix-conditioned 的估计，从而减少 variance。

### 5. Block diffusion 的 sliced training

对于 block diffusion，论文进一步把 trace 重写成 block slice，使每个 slice 只需一次 block-attention forward。这样 block-attention 的结构优势就不仅体现在推理时，也能体现在 RL 训练时。这使 TraceRL 在 block model 上特别自然。

## 实验结果

### 1. TraDo 指令模型显著提升 diffusion reasoning

作者基于 SDAR block diffusion 模型，使用 TraceRL 训练出 TraDo-4B-Instruct 和 TraDo-8B-Instruct。

在主表中，TraDo-8B-Instruct 相比 SDAR-8B-Chat 的提升为：

- **MATH500**：$74.3 \rightarrow 78.5$（static）
- **AIME2024**：$11.8 \rightarrow 13.3$
- **GSM8K**：$91.1 \rightarrow 92.3$
- **LiveCodeBench-v2**：$18.5 \rightarrow 25.9$
- **LiveBench**：$11.5 \rightarrow 22.7$

同时，TraDo-4B-Instruct 在数学任务上也能压过更大的 AR 基线。作者强调：4B 规模的 diffusion instruction model 已经能稳定超过 Qwen2.5-7B-Instruct 和 Llama3.1-8B-Instruct 的若干数学推理表现。

### 2. 首个 long-CoT diffusion model

TraDo-8B-Thinking 在 TraDo-8B-Instruct 基础上通过 long-CoT SFT + TraceRL 得到。论文报告其成绩为：

- **MATH500**：$87.4$
- **AIME2024**：$35.5$
- **GSM8K**：$94.2$
- **LiveCodeBench-v2**：$34.6$
- **LiveBench**：$36.0$

这说明 diffusion LM 并不只能做“快速但浅”的生成；在合适的后训练框架下，也能承载长链式推理。

### 3. TraceRL 优于已有 diffusion RL 变体

作者把 TraceRL 与 random masking RL、coupled RL 等现有 diffusion RL 方法对比。无论在 block diffusion 还是 full-attention diffusion 上，TraceRL 都取得更快收敛和更优最终性能。论文将核心原因归结为：**其他方法仍在优化随机 mask 的局部恢复，而 TraceRL 直接优化模型真实采样轨迹。**

### 4. Value model 确实降低波动

在 4B 数学任务训练曲线中，使用 value model 的版本波动明显更小。作者因此认为 diffusion value model 可以像 AR-RL 中的 critic 一样提供 variance-reducing baseline，只不过这里它需要显式适配 diffusion trajectory。

### 5. 额外收益：更大 block 与更快采样

论文还展示了两个“不是 benchmark accuracy 本身、但很工程化”的结果：

1. **扩展 block size**：把原本 block size $B=4$ 的模型适配到 $B=8$ 后，MATH500 从直接切换时的 $60.2$ 恢复到 $67.7$，表明 TraceRL 可以帮助模型适配更大的并行生成粒度。
2. **动态采样加速**：在 MATH500 上，TraDo-4B-Instruct 的 acceleration ratio 从 $2.28$ 提升到 $2.63$，同时平均响应长度也增加，说明训练不仅提升了正确率，也改变了动态采样过程中 token unmask 的节奏与效率。

## 分析与讨论

### 1. 这篇论文真正优化的是“推理过程”，不是“结果分数”

TraceRL 最重要的思想不是又造了一个 PPO 变种，而是把 DLM rollout 中的 **inference trace** 视作第一类训练对象。对于 diffusion LM，这一点尤其重要，因为它的生成不是标准 AR token-by-token，而是由一系列 mask/unmask 决策构成。若训练忽略这些中间步骤，就会出现优化目标和部署行为割裂。

### 2. 对 repo 当前主题的关联：它把“采样路径”正式纳入 RL 目标

如果把你 repo 里常见的 self-improvement / reasoning 论文放在一起看，很多工作都在优化最终 answer 或 trajectory-level reward；而这篇论文的独特点在于：**trajectory 本身就是结构化对象**。这和 diffusion inference 的内部 mechanics 紧密相关，也启发一个更一般的问题：当生成过程不是单一路径时，RL 应该如何定义“credit assignment 的粒度”。

### 3. 它兼顾了 accuracy、speed、architecture generality

很多 reasoning 论文只汇报 benchmark score，但这篇论文同时关心：

- 后训练目标是否与推理机制一致；
- 是否支持 full-attention 与 block-attention 两类 DLM；
- 是否能改善 sampling flexibility 与 acceleration；
- 是否能落到开源 framework。

因此它不像单纯的“刷分 paper”，而更像一篇面向 DLM post-training infrastructure 的系统论文。

## 局限性

1. **主要验证集中在可验证奖励任务**：数学、代码等 RLVR 场景最适合该框架。对于开放式对话、创作等主观任务，文中没有给出充分验证。
2. **很多收益与特定采样机制绑定较深**：TraceRL 的优势来自对 diffusion inference trace 的精细对齐，这使它对 DLM 很自然，但也意味着它不一定直接迁移到普通 AR 模型。
3. **long-CoT 模型仍依赖 SFT 课程学习**：TraDo-8B-Thinking 并非纯 TraceRL 产物，而是 long-CoT SFT 与 TraceRL 的组合，因此“RL 单独能否学出 long-CoT”并未被完全回答。
4. **论文更强调系统有效性，理论部分相对克制**：它清晰展示了 mismatch 与经验收益，但没有给出像一些 RL 理论论文那样严格的最优性解释，更像强工程归纳而非完整理论闭环。

## 关键洞察

1. **扩散语言模型的后训练，关键不是照搬 AR-RL，而是显式对齐 diffusion 的采样轨迹。**
2. **如果生成过程是多步并行 unmask，而不是单步 next-token，那么 reward assignment 的自然粒度也应该改变。**
3. **训练目标与部署推理机制的一致性，本身就是后训练性能的重要来源。** 这在 DLM 上被论文非常直接地展示出来。
4. **TraceRL 的价值不只在 benchmark 分数，而在于它把 DLM 的 RL training、inference acceleration、block-size adaptation、value estimation 放进了一个统一框架。**
5. **对当前 repo 的阅读脉络而言，这篇论文补上了一个很关键的方向：不仅要问“模型能否通过 RL 学会更强推理”，还要问“模型内部的生成轨迹结构，是否决定了 RL 应该怎样训练它”。**
