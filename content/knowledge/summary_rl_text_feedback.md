---
title: "RL from Text Feedback (RLTF)"
authors: "{Yuda Song"
institution: "Unknown"
venue: "ICML"
arxiv_id: "2602.02482"
tags: ["paper"]
---
# RL from Text Feedback (RLTF)

**论文标题:** Beyond Scalar Rewards: Learning from Text Feedback in LLM Post-Training

**作者:** Yuda Song, Lili Chen, Fahim Tajwar, Rémi Munos, Deepak Pathak, J. Andrew Bagnell, Aarti Singh, Andrea Zanette

**机构:** Carnegie Mellon University, Inria, Aurora Innovation

**链接:** https://arxiv.org/abs/2602.02482 | https://rl-textfeedback.github.io

---

## 1. 核心问题

传统 LLM 后训练中的 RL 依赖于**稀疏的标量奖励**（每个 rollout 只有一个 bit 的信息），学习效率极低。另一方面，**蒸馏**提供密集监督但需要专家示范，成本高且难以扩展。

**核心洞察:** 自然语言文本反馈是一个理想的中间信号——比标量奖励更丰富，比完整示范更便宜。文本反馈可以定位错误、指出违反的约束或建议修复方案。

**关键挑战:** 训练时可以获得反馈，但推理时通常没有反馈。模型必须**内化**反馈来提升单轮测试性能，而不是仅仅学会在给定反馈时做得更好。

---

## 2. 问题形式化

### 交互协议

- 初始提示 $x_0 \sim \mu(\mathcal{X}_0)$
- 策略生成输出 $y_0 \sim \pi(\cdot \mid x_0)$
- 获得奖励 $r_0 = R(x_0, y_0)$ 和反馈 $c_0 \sim \mathcal{M}(\cdot \mid x_0, y_0)$
- 第二轮提示 $x_1 = f(x_0, y_0, c_0)$（通常为拼接）
- 生成修订输出 $y_1 \sim \pi(\cdot \mid x_1)$

### 学习目标

**多轮目标:**
$$J_{\mathsf{MultiTurn}}(\pi) = \mathbb{E}^\pi \left[\sum_{h=0}^{H-1} r_h\right]$$

**单轮目标（真正关心的）:**
$$J_{\mathsf{SingleTurn}}(\pi) = \mathbb{E}_{x_0 \sim \mu} \left[\mathbb{E}_{y \sim \pi(\cdot \mid x_0)} [R(x_0, y)]\right]$$

**核心研究问题:** 给定训练时的反馈增强轨迹，如何设计学习目标来提升 $J_{\mathsf{SingleTurn}}(\pi)$？

---

## 3. 方法一：Self Distillation (RLTF-SD)

### 核心思想

将反馈条件下的第二轮输出作为隐式教师，蒸馏到单轮策略中。本质上是"编译掉"对反馈的依赖——将测试时的改进能力转化为训练信号。

### 蒸馏目标

$$\ell_{\mathsf{distill}}(\pi) = \mathbb{E}_{x_1 \sim \mathbb{P}^\pi, y_1 \sim \pi(\cdot \mid x_1)} \left[ \frac{\mathrm{sg}[\pi(y_1 \mid x_0)]}{\pi_{\mathrm{ref}}(y_1 \mid x_1)} A(x_0, y_1) \right]$$

当 $\pi_{\mathrm{ref}}(\cdot \mid x_1) = \pi(\cdot \mid x_1)$ 且 $A(y_1) = R(x_0, y_1)$ 时，恢复单轮目标的无偏梯度估计：

$$\mathbb{E}_{y_1 \sim \pi(\cdot \mid x_1)} \left[\frac{\mathrm{sg}[\pi(y_1 \mid x_0)]}{\mathrm{sg}[\pi(y_1 \mid x_1)]} R(x_0, y_1)\right] = J_{\mathsf{SingleTurn}}(\pi)$$

### 关键设计选择

#### (1) Baseline 选择：一阶基线 vs 二阶基线

**二阶基线（GRPO 风格）存在梯度信号坍塌问题：**
$$A_i^{(1)} := R(x_0, y_1^i) - \frac{1}{N}\sum_{j=1}^N R(x_0, y_1^j)$$

当反馈使第二轮成功率 $p_1 \to 1$ 时，非零更新概率 $\approx N(1-p_1)$，即使教师总是正确，学生也得不到学习信号。

**一阶基线（推荐）:**
$$b^{(0)} := \frac{1}{N}\sum_{j=1}^N R(x_0, y_0^j), \quad A_i^{(0)} := R(x_0, y_1^i) - b^{(0)}$$

只有当学生自己已经正确时更新才为零，避免了梯度坍塌。

#### (2) 重要性加权的偏差-方差权衡

完整重要性采样是无偏的，但在 LLM 长序列上方差极大（跨 token 累积）。

**实践选择:** 设置 $\pi_{\mathrm{ref}}(\cdot \mid x_1) = \pi(\cdot \mid x_0)$，移除重要性加权，得到类似 AWR 的低方差目标：
$$\ell^{\mathsf{awr}}_{\mathsf{distill}}(\pi) = \mathbb{E}_{y_1 \sim \pi(\cdot \mid x_1)} \left[A(y_1) \nabla \log \pi(y_1 \mid x_0)\right]$$

实验表明方差主导偏差，轻微偏差是可接受的。

---

## 4. 方法二：Feedback Modeling (RLTF-FM)

### 核心思想

将反馈本身作为监督信号，训练策略预测反馈。反馈在每轮都可观测，比标量奖励丰富得多，提供了 token 级的密集梯度。

### 反馈预测损失

定义反馈预测分布：
$$p_\pi(c \mid x, y) := \pi(c \mid f_{\mathsf{feedback}}(x, y))$$

其中 $f_{\mathsf{feedback}}$ 是引出批评式反馈的 prompt 模板。

优化交叉熵目标：
$$\ell_{\mathsf{feedback}}(\pi) := \mathbb{E}_\pi \left[\sum_{h=0}^{H-1} -\log p_\pi(c_h \mid x_h, y_h)\right]$$

### 联合目标

$$\max_\pi J_{\mathsf{MultiTurn}}(\pi) - \lambda_{\mathsf{feedback}} \cdot \ell_{\mathsf{feedback}}(\pi)$$

### 理论分析：表示学习视角

**纯奖励学习的瓶颈：**

1. **稀有事件估计:** 稀疏奖励下，成功概率 $\varepsilon_0$ 很低，SNR $\propto \sqrt{\varepsilon_0}$，需要 $O(1/\varepsilon_0)$ 个 rollout 才能可靠估计单个梯度分量。

2. **表示方向的弱可辨识性:** 即使条件于成功，奖励加权梯度信号也集中在少数表示方向，存在大量"低信号子空间"。

**反馈建模的优势：**

在覆盖假设下，RLTF-FM 提供了额外的监督信号，能够填补那些纯奖励 RL 难以学习的表示方向。本质上，RLTF-FM 作为"表示预条件器"，改善了表示自由度的可辨识性和条件数。

### 测试时扩展：自反馈

由于 $p_\pi(c \mid x, y)$ 由同一策略产生，模型可以在推理时进入"反馈模式"：
1. 采样 $y_0 \sim \pi(\cdot \mid x_0)$
2. 生成自我批评 $\tilde{c}_0 \sim p_\pi(\cdot \mid x_0, y_0)$
3. 更新 $x_1 = f(x_0, y_0, \tilde{c}_0)$
4. 重新采样 $y_1 \sim \pi(\cdot \mid x_1)$

无需单独的 judge 模型即可实现测试时扩展。

---

## 5. 实验结果

### 实验设置

- **反馈提供者:** Qwen3-235B-A22B-Instruct-2507
- **学习者:** Llama-3.1-8B-Instruct
- **任务:** 推理谜题、竞赛数学、创意写作

### 主要发现

| 任务类型 | RLTF-SD | RLTF-FM | 多轮 GRPO | 单轮 GRPO |
|---------|---------|---------|----------|----------|
| Knights and Knaves | **0.802** | 0.765 | 0.277 | 0.052 |
| Binary Matrix | **0.976** | 0.933 | 0.904 | 0.001 |
| Shortest Path | **0.830** | 0.757 | 0.521 | 0.035 |
| MATH500 (DAPO) | 0.699 | **0.718** | 0.696 | 0.678 |
| AIME24 (DAPO) | 0.233 | **0.267** | 0.133 | 0.167 |

**关键观察：**

1. **多轮 GRPO 的局限:** 朴素地将反馈作为上下文无法内化学习信号，单轮性能提升有限。

2. **RLTF-SD vs RLTF-FM:**
   - RLTF-SD 在创意写作等教师-学生分布匹配较好的任务上表现更优
   - RLTF-FM 在数学和推理任务上更好，因为反馈更客观、预测损失更易优化

3. **设计选择验证:**
   - 一阶基线显著优于二阶基线
   - AWR 风格（无重要性加权）优于带裁剪的重要性采样

4. **文本反馈 vs 正确性反馈:** 语义丰富的文本反馈显著优于简单的"正确/错误"信号。

5. **测试时扩展:** RLTF-FM + 自我批评训练可实现有效的多轮测试时改进。

---

## 6. 算法伪代码

### RLTF-SD 算法

**输入:** 策略 $\pi_\theta$，反馈提供者 $\mathcal{M}$，初始提示分布 $\mu$

**While not converged:**
1. 采样 $x_0 \sim \mu$，生成 $\{y_0^i\}_{i=1}^N \sim \pi_\theta(\cdot \mid x_0)$
2. 获取反馈 $c_0^i \sim \mathcal{M}(\cdot \mid x_0, y_0^i)$
3. 构造 $x_1^i = f(x_0, y_0^i, c_0^i)$
4. 生成 $y_1^i \sim \pi_\theta(\cdot \mid x_1^i)$
5. 计算一阶基线 $b^{(0)} = \frac{1}{N}\sum_j R(x_0, y_0^j)$
6. 计算优势 $A_i = R(x_0, y_1^i) - b^{(0)}$
7. 更新：$\theta \leftarrow \theta + \eta \sum_i A_i \nabla_\theta \log \pi_\theta(y_1^i \mid x_0)$

### RLTF-FM 算法

**训练:**
1. 对每个 $(x_h, y_h, c_h)$，计算 $\ell_{\mathsf{feedback}} = -\log p_\pi(c_h \mid x_h, y_h)$
2. 联合优化 RL 目标和反馈预测损失

**测试时扩展（可选）:**
1. $y_0 \sim \pi(\cdot \mid x_0)$
2. **For** $t = 0, 1, \ldots, T-1$:
   - $\tilde{c}_t \sim p_\pi(\cdot \mid x_t, y_t)$
   - $x_{t+1} = f(x_t, y_t, \tilde{c}_t)$
   - $y_{t+1} \sim \pi(\cdot \mid x_{t+1})$

---

## 7. 与相关工作的联系

| 方法 | 信息密度 | 可扩展性 | 单轮测试性能 |
|-----|---------|---------|------------|
| 标量奖励 RL | 低（1 bit/rollout） | 高 | 直接优化 |
| 专家蒸馏 | 高（完整示范） | 低 | 需要专家 |
| 多轮 RL | 中 | 中 | 无提升 |
| **RLTF（本文）** | **中-高** | **高** | **显著提升** |

---

## 8. 局限性与未来方向

1. **反馈质量:** 实际应用中反馈可能有噪声或主观偏见，需要数据筛选。
2. **长程交互:** 超长反馈序列可能需要摘要等技术处理分布漂移和上下文限制。
3. **理论扩展:** 当前理论分析聚焦于基策略生成分布附近的表示学习，端到端分析是未来方向。
4. **与过程奖励模型结合:** 探索与 PRM 等细粒度监督方法的协同效应。

---

## 9. 核心贡献总结

1. **问题形式化:** 明确了 RLTF 的目标——利用训练时反馈提升单轮测试性能
2. **两种互补方法:**
   - RLTF-SD：将反馈条件下的改进蒸馏为单轮能力
   - RLTF-FM：通过预测反馈学习更好的表示
3. **理论洞察:** 从表示学习角度解释了反馈建模的收益
4. **实验验证:** 在推理、数学、写作任务上显著超越基线
