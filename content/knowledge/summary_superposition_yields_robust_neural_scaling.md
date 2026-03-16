---
title: "Superposition Yields Robust Neural Scaling"
authors: "Yizhou Liu, Ziming Liu, Jeff Gore"
institution: "Massachusetts Institute of Technology"
venue: "arXiv 2025"
arxiv_id: "2505.10465"
tags: ["paper", "scaling-laws", "superposition", "representations", "mechanistic-interpretability", "llm", "width", "power-law", "neurips"]
---

# Superposition Yields Robust Neural Scaling

## TL;DR
本文提出一个机制性解释：**表示的 superposition（特征数 $n$ 远大于表示维度 $m$）本身就能驱动神经 scaling law**。在 Anthropic 风格 toy autoencoder 中，作者用可控的 weight decay/growth 调节 superposition 强弱，发现：

- **弱 superposition**：loss 主要来自“没被表示的低频特征”，因此只有当特征频率 $p_i$ 本身满足幂律（如 Zipf）时，loss 才会随宽度呈幂律。
- **强 superposition**：loss 主要来自表示向量之间的**几何重叠（overlap/interference）**，在很广泛的频率分布下都能得到稳定的
  $$L \propto \frac{1}{m}$$
  （“one-over-width”）缩放。
- 对真实 LLM（OPT/GPT2/Qwen/Pythia）的语言模型 head 行向量做统计，作者观察到：**归一化行向量的均方重叠近似满足 $1/m$**，且 loss-vs-$m$ 的拟合指数 $\alpha_m \approx 0.91$，与 toy 模型强 superposition 预测一致，并与 Chinchilla scaling 在换元后相容。

## 1. 研究问题
经典 scaling law（loss 随模型规模/宽度/数据量等呈幂律下降）很稳健，但其起源解释众多且往往依赖对数据谱/技能重要性幂律的假设。本文聚焦 LLM 的一个现实约束：

- 词表/概念/特征数量巨大（$n$ 大），隐藏维度有限（$m$ 小）。
- 因此表示层（embedding / LM head）天然处于“**需要在低维里承载超多特征**”的 superposition 状态。

核心问题：**superposition 会如何影响 loss 随宽度 $m$ 的缩放形态与指数？**

## 2. Toy model（来自 Anthropic superposition toy model）
### 2.1 数据生成
输入 $x\in\mathbb{R}^n$，第 $i$ 个特征：
$$x_i = u_i v_i,\quad u_i\sim\mathrm{Bernoulli}(p_i),\; v_i\sim U(0,2).$$
- $p_i$：特征出现频率（按重要性降序，随 $i$ 递减）
- activation density：$E=\sum_{i=1}^n p_i$

### 2.2 模型与损失
参数 $W\in\mathbb{R}^{n\times m}$、$b\in\mathbb{R}^n$：
- 编码：$h = W^\top x$（$m\ll n$）
- 解码：$y = \mathrm{ReLU}(Wh + b)$
- MSE loss：
$$L = \langle \|y-x\|_2^2 \rangle_x.$$

行向量 $W_i$ 表示“特征 $i$ 在 $m$ 维隐空间中的表示”。

### 2.3 用 decoupled weight decay/growth 调节 superposition
作者在 AdamW 中加入按“行向量”定义的 decoupled 权重衰减/增长：
$$
W_{i,t+1} =
\begin{cases}
W_{i,t} - \eta_t \gamma W_{i,t}, & \gamma \ge 0 \\
W_{i,t} - \eta_t \gamma W_{i,t}(1/\|W_{i,t}\|_2 - 1), & \gamma < 0
\end{cases}
$$
其中 $\gamma<0$ 可视作在优化 $(\|W_i\|_2-1)^2$，倾向把行向量推向 unit norm，从而更容易进入强 superposition。

并用
$$\phi_{1/2} = |\{i: \|W_i\|_2>1/2\}|/n$$
衡量“被表示的特征比例”。实证上：小 $\gamma$（或 $\gamma<0$）给强 superposition（$\phi_{1/2}\approx 1$），大正 $\gamma$ 给弱/无 superposition（$\phi_{1/2}\sim m/n$）。

## 3. 结果一：弱 superposition = “Power law in, power law out”
在理想弱 superposition 中，模型几乎只表示最重要的 $\phi_{1/2}n$ 个特征，剩余特征被忽略。此时 loss 近似为“忽略特征频率之和”：
$$
L = \sum_{i>\phi_{1/2} n} \langle (x_i-\langle x_i\rangle)^2\rangle
\approx \langle v^2\rangle \sum_{i>\phi_{1/2} n} p_i.
$$
当 $p_i\propto 1/i^\alpha$ 且 $\alpha>1$、并取接近无 superposition 的情况 $\phi_{1/2}n\approx m$ 时：
$$\sum_{i>m} p_i \propto m^{-(\alpha-1)}\quad\Rightarrow\quad L\propto m^{-(\alpha-1)}.$$
因此弱 superposition 的幂律指数由**数据频率幂律**决定（不稳健、依赖数据分布尾部）。

## 4. 结果二：强 superposition 的几何重叠导致稳健 $1/m$
强 superposition 下，许多（甚至大量）特征都被表示，但行向量之间不可避免存在 overlap，干扰项主导 loss。

作者给出核心几何直觉：对“各向同性”的单位向量，内积平方 $ (w_i\cdot w_j)^2 $ 的期望尺度是 $1/m$，因此干扰项天然给出 $1/m$ 量级。

进一步，作者认为训练会倾向于最小化“最大重叠”（有利于误差纠正），并引入 Welch bound：对 $\nu\ge m$ 个单位向量
$$\max_{i\ne j} |w_i\cdot w_j| \ge \sqrt{\frac{\nu-m}{m(\nu-1)}}\equiv \kappa\approx \sqrt{\frac{1}{m}}\quad(\nu\gg m).$$
当达到下界时形成等角紧框架（ETF）。实证观察到重要特征的 $W_i/\|W_i\|_2$ 更“ETF-like”，均方重叠随 $m$ 呈 $1/m$ 缩放。

**结论（even frequencies / 各向同性近似成立时）：**
$$L \propto \frac{1}{m},\quad \alpha_m\approx 1.$$

当频率分布更偏斜（大 $\alpha$）时，向量分布非各向同性，$\alpha_m$ 可增大；作者用一个极端近似（$\sim m^2/2$ 个最重要特征近乎不贡献 loss）推导
$$L\sim\sum_{i\gtrsim m^2/2} p_i \sim m^{-2(\alpha-1)}\quad\Rightarrow\quad \alpha_m\approx 2(\alpha-1),$$
与 toy 实验趋势接近。

## 5. 连接到真实 LLM：LM head 的 overlap 与 loss-vs-width
作者把 token 视作“原子特征”，取 $n\approx |\mathcal{V}|$，把语言模型 head 矩阵作为 $W$，统计其行向量：

- 归一化行向量 $W_i/\|W_i\|_2$ 的**均方重叠**随宽度近似按 $1/m$ 缩放；据此作者认为 LLM 处于强 superposition。
- 对 OPT/GPT2/Qwen/Pythia，多数据集评估的 loss vs $m$ 可拟合为
  $$L = \frac{C_m}{m^{\alpha_m}} + L_{\backslash m},$$
  得到 $\alpha_m\approx 0.91\pm 0.04$。
- 结合 Chinchilla 中参数量与宽度的经验关系 $N\propto m^{2.52\pm0.03}$，换元后也与 $\alpha_m\approx 1$ 一致量级。

## 6. 启示与局限
### 启示
- scaling law 的“稳健幂律”不一定来自数据频率幂律；**强 superposition 的几何干扰本身就能给出稳健 $1/m$**。
- 若表示瓶颈是主导项，则提升/改变表示（例如约束到球面、改变正则/优化器以促进 superposition）可能改变 scaling law 的系数，甚至在某些任务中改变指数。

### 局限
- toy model 只刻画表示层（无 transformer 计算层）；实际 LLM 总 loss 还叠加 parsing/计算层贡献。
- 强 superposition 下对 $W$ 的结构分析多为现象 + 几何下界/类比，缺少严格求解；当频率偏斜、各向异性显著时，指数不再完全稳健。

## 7. 和你 repo 现有主题的连接
你此前两篇总结（2509/2510）讨论的是 post-training 中“分布漂移/遗忘”与 KL；本文讨论的是 pretraining 中“表示几何/重叠”与 scaling。可以形成一个更大的视角：

- **几何重叠（superposition）**决定了模型在表示层的“干扰结构”，可能影响后续对齐/持续学习时的可塑性与稳定性。
- 如果把“遗忘”也理解为某种干扰/重叠在新分布上被放大，那么这篇的几何量（例如 overlaps 的 $1/m$ 缩放）或许能启发：在更宽的表示空间里，保持能力（或保持 KL 小）可能更容易。
