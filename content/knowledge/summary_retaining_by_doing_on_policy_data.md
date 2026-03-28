---
title: "Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting"
authors: "Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen"
institution: "Princeton Language and Intelligence, Princeton University"
venue: "arXiv 2025"
arxiv_id: "2510.18874v1"
tags: ["paper", "post-training", "continual-learning", "catastrophic-forgetting", "on-policy", "rl", "sft", "kl-divergence", "reverse-kl", "forward-kl", "iclr"]
---

# Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

## TL;DR
本文研究 LM post-training 中一个反复出现的现象：**RL 微调往往比 SFT 更不易遗忘（catastrophic forgetting 更轻）**。作者提出一个核心解释：

- RL 的优势主要来自 **on-policy 数据**（训练数据由当前策略生成），而不是 advantage 估计、也不是 KL 正则本身。
- 从 KL 视角看：SFT 可视为 **forward KL** 最小化（mode-covering），RL 可视为 **reverse KL** 最小化（mode-seeking）。
- 反直觉点：mode-seeking（reverse KL）通常被认为更容易丢覆盖、从而更易遗忘；作者用一个“混合分布/多模态”的简化分析说明：当初始策略是多模态（更贴近现实 LM）时，**reverse KL 的 mode-seeking 反而更能在学新任务时保住旧模式**。
- 实用结论：不必完全 on-policy 才有效。**“近似 on-policy”数据（例如每个 epoch 开始生成一次数据的 Iterative-SFT / RAFT 风格）**就能显著降低遗忘，计算更省。

## 1. 研究问题
对齐/微调 LM 到新任务（instruction following、知识、算术推理等）时，常见风险是旧能力下降。本文系统比较：

- 监督微调 SFT（含 teacher SFT、Self-SFT 等）
- RL（主要使用 GRPO，verifiable reward setting）

目标：找出 **RL 比 SFT 更稳的根因**，并给出工程上可落地的“抗遗忘”指导。

## 2. 主要贡献
1. **大规模实证对比**：跨模型家族（Llama 3、Qwen 2.5）、多任务（IFEval、MMLU、Countdown 等），RL 在达到相当或更高 target performance 时，non-target drop 明显更小。
2. **KL 视角的解释框架**：
   - SFT $\leftrightarrow$ forward KL 最小化（mode-covering）
   - KL-regularized RL $\leftrightarrow$ reverse KL 最小化（mode-seeking）
3. **简化混合分布分析（Gaussian mixture）**：解释为何在多模态初始策略下，reverse KL 的 mode-seeking 反而可能更“保留旧模式”。
4. **消融验证根因是 on-policy 数据**：排除 KL 正则与 advantage 估计的必要性。
5. **提出可用的近似 on-policy 方案**：Iterative-SFT（每个 epoch 开始生成数据）等即可显著降低遗忘。

## 3. 任务与度量（实证部分）
### 3.1 语言模型表示
将 LM 看作策略 $\pi_\theta(y\mid x)$。

### 3.2 RL 目标与 SFT 目标
- SFT 交叉熵：
  $$\mathcal{L}_{\mathrm{SFT}}(\theta;x)=\sum_y -\pi^*(y\mid x)\log\pi_\theta(y\mid x)$$
- RL（文中用 RLVR：二元可验证 reward），带 KL 正则的形式：
  $$J_{\mathrm{RL}}(\theta;x)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]-\beta\,D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\theta_0}(\cdot\mid x)\big)$$

### 3.3 “Gain/Drop” 指标
- Target task gain：
  $$\Delta_g=\mathcal{A}(\pi_{\theta_T},\mathcal{T})-\mathcal{A}(\pi_{\theta_0},\mathcal{T})$$
- Non-target tasks drop（衡量遗忘）：
  $$\Delta_d=\frac{1}{M}\sum_{j=1}^M \Big(\mathcal{A}(\pi_{\theta_0},\mathcal{T}'_j)-\mathcal{A}(\pi_{\theta_T},\mathcal{T}'_j)\Big)$$

### 3.4 具体任务
- Target tasks：IFEval（指令遵循）、MMLU（通识知识）、Countdown（算术推理）
- Non-target tasks 还包含：MATH，以及安全相关 WildJailbreak、WildGuardTest（安全能力易被微调侵蚀）
- 模型：Llama-3.x instruct、Qwen-2.5 instruct（最高到 8B）
- RL：GRPO；reward 设为正确=1、错误=0
- SFT 变体：
  1) SFT：teacher 生成（Llama-3.3-70B-Instruct）
  2) Self-SFT：初始模型自生成 + reward 过滤正确样本

**实证结论**：RL 的 gain-drop 曲线显著更优：在相近 gain 下 drop 更小；SFT 想要高 gain 往往需要更大 lr，从而遗忘更严重。

## 4. KL 视角与“反直觉”解释
### 4.1 SFT 与 forward KL
SFT 可写为：
$$\mathcal{L}_{\mathrm{SFT}}=D_{\mathrm{KL}}\big(\pi^*(\cdot\mid x)\,\Vert\,\pi_\theta(\cdot\mid x)\big)+\mathcal{H}(\pi^*(\cdot\mid x))$$
因此等价于 forward KL 最小化（mode-covering）。

### 4.2 RL 与 reverse KL
对于 KL-regularized RL，其最优策略可写为（指数倾斜）：
$$\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\theta_0}(y\mid x)\exp\big(r(x,y)/\beta\big)$$
从而
$$J_{\mathrm{RL}}(\theta;x)= -\beta\,D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi^*(\cdot\mid x)\big)+\beta\log Z(x)$$
可视为 reverse KL 最小化（mode-seeking）。

### 4.3 为什么多模态时 reverse KL 反而更“保留旧能力”？
作者用一个简化的 mixture-of-Gaussians 来建模“旧能力模式 + 新任务模式”。结论是：

- **单峰初始策略（uni-modal）**：forward KL 更不易遗忘（符合传统直觉）
- **多峰初始策略（multi-modal，更像现实 LM）**：reverse KL 的 mode-seeking 可以把“新模式”部分移动到新任务上，同时保留旧模式，而 forward KL 可能需要“拉扯/重分配”概率质量，导致旧模式覆盖下降。

这个理论部分给了一个把“RL 更稳”与“reverse KL mode-seeking”统一起来的解释：关键在于 **初始策略的多模态性**。

## 5. 关键实证：on-policy 数据是主要贡献者
作者指出 GRPO(RL) 与 SFT 的三类差异：
1. on-policy 数据 vs off-policy 数据
2. RL 常用 KL 正则，SFT 不一定
3. RL 有 advantage 估计，SFT 没有

消融结论：
- **KL 正则不是主要原因**：non-regularized GRPO 与 KL-regularized GRPO 的 gain-drop 类似（少数例外）。
- **advantage 不是主要原因**：REINFORCE（无 advantage）同样能保持低遗忘（但 target gain 略弱）。

因此：**on-policy 数据是 RL 抗遗忘的主因**。

## 6. 近似 on-policy：更省算力也有效
作者进一步问：需要多 on-policy 才能有效？

发现：
- 只用初始策略一次性自生成（Self-SFT）不够，仍会严重遗忘。
- **Iterative-SFT**（每个 epoch 开始，用当前策略生成数据再 SFT；RAFT/自举系思路）可以：
  - 达到接近甚至超过 SFT 的 target accuracy
  - 同时遗忘显著降低，接近 RL
- 另外，把 RL 过程中生成的轨迹拿来做 SFT，也能降低遗忘。

工程启示：**无需每步都 rollout 才能享受 on-policy 的稳定性**；epoch 级别的 refresh 可能是一个很好的性价比折中。

## 7. 和你 repo 内已有主题的连接
你已有的阅读主题里（如 self-play / 自举推理 / 零数据自改进等）经常涉及“模型自己生成训练数据”的闭环。本文给出一个更明确的设计要点：

- “数据来自谁”比“优化目标/算法细节”更关键：**越接近 on-policy 采样，越能减少遗忘**。
- 如果你在做自生成数据管线：可以考虑采用 **iterative refresh**（每轮/每 epoch 更新采样分布），避免数据分布长期滞后导致 off-policy 训练，从而更易遗忘。

## 8. 局限与开放问题
- 对更大规模模型/更大数据的规律仍待验证（算力限制）。
- 理论上还需要更严谨地刻画 on-policy 数据为何能系统性抑制遗忘（本文主要给直觉 + 简化模拟 + 实证）。

