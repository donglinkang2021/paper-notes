---
title: "RL's Razor: Why Online Reinforcement Learning Forgets Less"
authors: "Idan Shenfeld, Jyothish Pari, Pulkit Agrawal"
institution: "Improbable AI Lab, MIT"
venue: "arXiv 2025"
arxiv_id: "2509.04259"
tags: ["paper", "continual-learning", "post-training", "rl", "sft", "forgetting", "kl-divergence", "on-policy", "iclr"]
---

# RL's Razor: Why Online Reinforcement Learning Forgets Less

## TL;DR
本文系统比较了在同等新任务性能下，**在线/On-policy RL 微调比 SFT（监督微调）更不容易遗忘**。核心发现是一个可操作的经验律：

> 遗忘程度主要由“新任务分布上”的分布漂移决定，且可用 **forward KL** 预测。

作者进一步提出 **RL’s Razor**：当存在多个都能完成新任务的解（多种输出分布同样正确）时，**on-policy RL 会偏向 KL 上最接近基座模型的解**，因此更“保守”，遗忘更少；而 SFT 会被外部标注分布牵引到可能非常远的分布，从而更易遗忘。

## 1. 问题与动机
面向“长期在线”的 foundation model/agent，我们希望模型在持续学习新能力时不抹掉旧能力（catastrophic forgetting）。本文关注一个实践常见但解释不足的现象：

- RL fine-tuning（尤其是 on-policy 的 policy gradient 类）在很多情况下看起来比 SFT 更“稳”，旧能力掉得更少。

作者的目标不是提出新算法，而是找到能解释“何时/为何遗忘”的简单变量，并据此解释 RL vs SFT 的差异。

## 2. 主要贡献
1. 经验上：在多个任务/模型上展示 **RL 在同等新任务性能下遗忘显著更少**（Pareto frontier 更好）。
2. 提出并验证 **经验遗忘律（empirical forgetting law）**：

   令基座策略为 $\pi_0$，微调后策略为 $\pi$，新任务输入分布为 $\tau$，则遗忘程度可由
   $$\mathbb{E}_{x\sim\tau}\big[ D_{\mathrm{KL}}(\pi_0(\cdot\mid x)\,\Vert\,\pi(\cdot\mid x)) \big]$$
   进行高可信预测（forward KL 在新任务上测）。
3. 提出 **RL’s Razor**：on-policy RL 在多解情形下隐式偏向 **KL-minimal** 的解；SFT 则可能收敛到离基座任意远的解。
4. 实证 + 理论：通过对比四类目标（是否 on-policy、是否有负例梯度）表明**关键不是负例，而是 on-policy 数据**；并给出在简化设定下 policy gradient 收敛到 KL-最近最优解的理论解释。

## 3. 实验设置与现象
### 3.1 主要任务（LLM + 机器人）
- LLM：Qwen 2.5 3B-Instruct，在
  - 数学推理（Open-Reasoner-Zero 数据）
  - Science QA（SciKnowEval 化学子集）
  - 工具使用（ToolAlpaca）
- 机器人：OpenVLA 7B，在 SimplerEnv 上的 pick-and-place（拾取罐子）

对比方法：
- **SFT**：交叉熵拟合外部标注分布
- **RL（GRPO）**：只用二元成功指标 reward，且**不显式加 KL 正则**

衡量：
- 新任务表现（accuracy/success rate）
- 旧任务/通用能力保持（多 benchmark：HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval；机器人用其他 drawer 任务）

结论：在同等新任务性能下，RL 的 prior-task retention 明显更高；SFT 往往用“牺牲旧能力”换新任务提升。

### 3.2 控制玩具环境：ParityMNIST
作者指出大型 RL 训练昂贵，难以做系统扫参和机制验证，于是构造 ParityMNIST：
- 任务：预测奇偶（even/odd），但输出空间仍是 10 个 digit label。
- 正确性：只要预测任意正确奇偶集合内的数字即可。

这个设定的关键：**存在大量“都正确”的输出分布**（多解），与许多生成式任务类似。

在该设定里，作者可以：
- 完整收敛
- 系统扫参
- 分析 KL 与遗忘的关系

结果：遗忘-新任务 trade-off 在 RL 与 SFT 中都存在，但如果用 forward KL 作为横轴，遗忘曲线高度对齐（说明 KL 是更“因果/决定性”的变量）。

## 4. 经验遗忘律：KL 是强预测因子
作者测试了多类候选变量（权重变化、表征漂移、更新稀疏/秩、其他分布距离 TV / reverse KL / $L_2$ 等），发现：

- **forward KL（在新任务分布上测）**对遗忘的解释力最强。
- 在 ParityMNIST 上一个二次拟合可达 $R^2\approx 0.96$；在 LLM 上也有较强相关（文中报告 $R^2\approx 0.71$，受估计噪声影响更大）。

直觉：新任务上策略分布走得越远（特别是把基座高概率质量转移走），越可能破坏原有通用能力。

## 5. 为什么 RL 更“保守”：RL’s Razor
### 5.1 关键对比：SFT vs RL 目标
对离散输出：

- SFT：给定外部监督分布 $\pi_\beta$（可任意），最小化
  $$\mathcal{L}_{\mathrm{SFT}}(\pi) = -\mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_\beta}[\log \pi(y\mid x)]$$

- RL（policy gradient/GRPO 形式）：从当前策略采样 $y\sim\pi$，按 advantage $A(x,y)$ 重新加权：
  $$\mathcal{L}_{\mathrm{RL}}(\pi) = -\mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi}\big[A(x,y)\log \pi(y\mid x)\big]$$

两个差异：
1. **采样分布**：RL 用当前策略（on-policy），SFT 用外部标注（offline）
2. **负例梯度**：RL 会对失败样本产生负系数（或 0），SFT 通常没有

### 5.2 消融：负例不是关键，on-policy 才是关键
作者构造四象限实验（on-policy/offline × 有无负例梯度）：
- GRPO（on-policy + 负例）
- 1–0 REINFORCE（on-policy + 无负例；只对正确样本做正向）
- SFT（offline + 无负例）
- SimPO（offline + 有负例）

结果：
- 两个 **on-policy**（GRPO、1–0 REINFORCE）都更低 KL、更少遗忘
- 两个 **offline**（SFT、SimPO）都更高 KL、更易遗忘

因此：**on-policy 机制**导致 RL 倾向于“只在已有概率质量附近微调”，从而 KL 漂移更小。

### 5.3 “剃刀”表述
在“多解”任务上，存在许多 $\pi$ 都能达到高 reward / 高 accuracy。RL’s Razor 说：

- policy gradient 的 on-policy 迭代隐式实现一种 KL-最近偏置，倾向收敛到
  $$\pi^\dagger = \arg\min_{\pi \in P^*\cap\Pi} D_{\mathrm{KL}}\big(\pi\,\Vert\,\pi_0\big)$$
  其中 $P^*$ 是最优策略集合（满足期望 reward 最优），$\Pi$ 是可表示策略族。

这相当于：在所有同样“会做新任务”的策略里，RL 更偏向最接近原模型的那个。

## 6. 一个很强的验证：Oracle SFT 可以超过 RL
若遗忘只由 KL 决定，那么只要给 SFT 一个“最小 KL 的正确标注分布”，SFT 也能不遗忘。

作者在 ParityMNIST 中可以解析构造 oracle 标注分布 $q^*$（在所有 100% 正确分布中最接近 $\pi_0$），并验证：
- 用 $q^*$ 做 SFT 得到的 trade-off **比 RL 还好**。

含义：RL 的优势并非“RL 本质更强”，而是 **on-policy 让它更容易找到 KL-minimal 解**；如果你能让 SFT 也朝 KL-minimal 目标走，它同样可以很稳。

## 7. 对你现有阅读主题的连接（repo 语境）
你这个仓库里已有不少“自我改进/自举推理/自我博弈”相关总结（例如 R0/Absolute Zero、self-play data-free 等）。这篇 RL’s Razor 可以作为一个 **post-training 稳定性/持续学习** 的解释框架：

- 很多自举方法本质会“让策略分布大幅迁移”（尤其在 synthetic data 或自生成标签偏置下）。本文提供了一个非常具体的可观测量：**新任务分布上的 forward KL**，可用作“分布漂移/遗忘风险”的在线监控指标。
- RL’s Razor 暗示：如果你的自举/自我改进流程能够保持 on-policy 或显式约束 KL（尤其是 forward KL 视角），可能更利于长期能力保持。

## 8. 局限与开放问题
- 经验律在更大规模 frontier 模型、更多生成式领域上的稳定性仍待验证。
- 机制层面仍缺少更细粒度解释：为什么新任务上的大 KL 会导致旧能力损伤？（表征干扰、容量限制、优化路径等）
- 文中也提到未系统研究 online 但 off-policy 的 RL 算法。

## 9. 可复用的“实践结论”（如果你要用在系统里）
1. 监控指标：在训练/微调过程中，持续估计
   $$\mathbb{E}_{x\sim\tau} D_{\mathrm{KL}}(\pi_0(\cdot\mid x)\Vert\pi(\cdot\mid x))$$
   可作为遗忘风险的领先指标。
2. 算法偏置：当任务存在多解时，on-policy 更新天然偏向 KL-minimal，可能比纯 SFT 更稳。
3. 训练设计：SFT 若能通过数据构造/目标重加权逼近 KL-minimal 标注分布，有机会达到甚至超过 RL 的稳定性。

