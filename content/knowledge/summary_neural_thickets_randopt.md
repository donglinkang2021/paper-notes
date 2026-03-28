---
title: "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights"
authors: "Yulu Gan, Phillip Isola"
institution: "MIT CSAIL"
venue: "arXiv 2026"
arxiv_id: "2603.12228"
tags: ["neural_thickets", "weight_space_sampling", "post_training", "ensemble", "scaling_laws"]
---

# Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

## TL;DR

这篇论文提出一个与“预训练权重是单个起点”不同的视角：**把预训练结果看成权重空间中一个分布的中心**。随着模型规模与预训练质量提升，预训练权重附近的局部邻域会从“needle-in-a-haystack（附近几乎没有更好的解）”转变为“thicket（灌木丛：附近密集存在大量任务改进的专家解）”。基于该现象，作者提出一个极简、完全并行的 post-training 方法 **RandOpt**：

- 采样 $N$ 个高斯权重扰动 $\theta_i = \theta + \sigma_i \epsilon(s_i)$
- 用少量训练/验证数据 $\mathcal{D}_{\mathrm{train}}$ 对每个扰动模型打分
- 选出 top-$K$ 模型
- 推理时对 top-$K$ 进行多数投票（majority vote）集成

在相同训练 FLOPs 下，RandOpt 在多种 LLM/VLM post-training 设定上能与 PPO/GRPO/ES 等方法竞争；并且在 wall-clock 上由于完全并行，训练步数为 $\mathcal{O}(1)$（对比需要 $\mathcal{O}(T)$ 序列迭代的基线）。

## 1. 关键概念与定义

### 1.1 Solution Density（解密度）

设性能度量为 $s: \mathbb{R}^d \to \mathbb{R}$，模型参数为 $\boldsymbol{\theta}\in\mathbb{R}^d$。定义 margin 为 $m$ 时的解密度：

$$
\delta(m) = \mathbb{P}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \left[ s(\boldsymbol{\theta} + \boldsymbol{\epsilon}) \ge s(\boldsymbol{\theta}) + m \right].
$$

直观上：$\delta(m)$ 就是**随机猜权重**在局部邻域里“命中更好解”的概率（hit rate）。作者报告：$\delta(m)$ 随模型规模单调上升，形成类似 scaling law。

### 1.2 Spectral Discordance（谱不一致性 / 多样性指标）

作者用一个“跨任务排序相关性”的指标衡量不同扰动模型是否是“专家”（specialists）还是“通才”（generalists）。

设 $\mathbf{P}\in[0,1]^{N\times M}$ 是 $N$ 个扰动在 $M$ 个任务上的 percentile-rank 矩阵，$\mathbf{C}\in\mathbb{R}^{M\times M}$ 为其列的 Pearson 相关矩阵，则定义：

$$
\mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \ne k} \mathbf{C}_{jk}.
$$

$\mathcal{D}\to 1$ 意味着任务之间排序几乎正交（专家高度分化），$\mathcal{D}\to 0$ 意味着各任务排序接近平行（更像通才整体提升）。作者观察到 $\mathcal{D}$ 也随规模增大而增大：**不仅更容易采到好解，而且采到的好解彼此互补**。

## 2. 主要发现（论文的“现象学”贡献）

1) **密度随规模上升**：在大模型附近，随机高斯扰动更可能带来任务性能提升（解密度 $\delta(m)$ 增大）。

2) **多样性随规模上升**：扰动模型往往是不同任务的“专家”，在某些任务上显著变好、另一些任务变差；并且这种专家分化随规模增大更明显（谱不一致性 $\mathcal{D}$ 增大）。

3) **从“针”到“灌木丛”的相变式解释**：
- 小模型：邻域里更好解极少，需要结构化搜索（梯度下降等）
- 大模型：邻域里更好解大量存在，随机采样 + 选择就够用

4) **改进来源可能混合了“浅层格式修复”和“深层推理提升”**：作者在 GSM8K 的分析中将收益分解为 format thicket vs reasoning thicket 等成分，指出不少提升来自输出格式/风格被修复，但也有一部分来自真正答对原本答错的问题。

## 3. RandOpt：算法与数学形式

论文给出的核心生成/选择/集成流程可以概括为：

**训练（选择）阶段：**

- 采样 $N$ 个随机种子 $\{s_i\}_{i=1}^N$ 与噪声尺度 $\{\sigma_i\}$（来自一组尺度集合 $\Sigma$）
- 构造扰动参数：

$$
\boldsymbol{\theta}_i = \boldsymbol{\theta} + \sigma_i \boldsymbol{\epsilon}(s_i), \quad \boldsymbol{\epsilon}(s_i) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)
$$

- 在小规模训练/验证集上打分，取 top-$K$：

$$
\mathcal{I}_{\mathrm{top}} = \mathop{\mathrm{arg\,topK}}_{i\in[N]} v_i.
$$

**推理（集成）阶段：**

对输入 $x$，用 $\mathcal{I}_{\mathrm{top}}$ 的模型生成答案并投票：

$$
\hat{y} = \mathop{\mathrm{mode}}\left( \left\{ \mathop{\mathrm{arg\,max}}_y f_{\boldsymbol{\theta}_i}(y\mid x) \mid i \in \mathcal{I}_{\mathrm{top}} \right\} \right).
$$

特点：
- 完全并行（无梯度、无序列更新）
- 训练步数 $\mathcal{O}(1)$（但需要大量并行算力）
- 推理成本是 $K$ 倍 forward pass

## 4. 实验设置概览（从 TeX 可见信息）

- LLM：Qwen / Llama / OLMo3（0.5B–8B），base 与 instruct variants
- 任务覆盖：math（Countdown, GSM8K, MATH-500, OlympiadBench），code（MBPP），writing（ROCStories），chemistry（USPTO）
- 对比：TT-MV（test-time majority vote）、PPO、GRPO、ES
- VLM：Qwen2.5-VL-3B-Instruct 在 GQA 上（冻结视觉 encoder，只扰动语言模型）

## 5. 论文想表达的“机制”直觉

- 预训练把权重带到一个区域：这个区域不仅在点 $\theta$ 处好，而且在其邻域也存在大量任务专用的高性能局部解。
- 越大的模型/越强的预训练，越像进入一个“高密度 + 高多样性”的局部结构：
  - 你不必精确地走梯度下降路径去找某一个最优点
  - 只要并行撒点（随机扰动）并筛选，就能快速抓到多个互补专家

作者并不强调 RandOpt 必然是最强的 post-training 方法，而是把它作为“探针”（probe）：它能成功说明邻域结构已经变得“容易”。

## 6. 与本仓库主题的关联 / 可复用启发

这篇论文对“post-training / test-time scaling / selection”类工作有一个统一视角：

- 许多方法可看作在某个空间（输出空间或权重空间）做 **Best-of-N + selection**。
- RandOpt 是“权重空间的 Best-of-N”。如果把 reward/verifier 接入，甚至可以把它当成一种可泛化的“在权重空间做 search”。

对于你这个仓库（summary→MOC→insight）而言，最直接的 insight 候选是：

- 当某个领域的论文不断出现“选择/集成比单点优化更重要”的证据时，可能形成一个跨论文的 insight：
  - **“强预训练表示使得局部搜索/选择变容易：优化方法差异变小，selection+aggregation 变关键。”**

## 7. 局限与风险点（从论文文字脉络推断）

- 需要大量并行算力：训练“步数”少不等于总体计算量低。
- 推理成本随 $K$ 增大。
- 基准收益中可能包含不少格式/风格类“浅层”改进（作者自己也分析了 format thicket）。
- 随机扰动与筛选在不同任务上的稳定性、可解释性与安全性仍需额外研究。

## 8. 我记录下来的关键引用点（便于回到原文定位）

- 标题、作者、机构、abstract：`main.tex` L120–163
- Solution density 定义：L227–233
- Spectral discordance 定义：L269–275
- RandOpt 伪代码：L362–383（Algorithm 2）
- RandOpt 推理投票公式：L398–402
- LLM 实验任务/模型概述：L412–419

