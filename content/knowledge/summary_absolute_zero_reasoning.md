---
title: "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"
authors: "Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang"
institution: "Tsinghua University, Beijing Institute for General Artificial Intelligence, Penn State University"
venue: "arXiv 2025"
arxiv_id: "2505.03335"
tags: ["self-play", "reinforcement-learning", "zero-data", "reasoning", "code-reasoning", "RLVR", "curriculum-learning"]
---

# Absolute Zero: 完全零数据的强化自博弈推理

## 核心贡献

本文提出 **Absolute Zero** 范式和 **AZR (Absolute Zero Reasoner)** 系统，实现了完全不依赖任何人类标注数据的推理模型训练。这是首个让单一模型同时学会"提出学习任务"和"解决任务"的自博弈系统，灵感来自 AlphaZero。

关键创新：
1. **Absolute Zero 范式**：模型同时扮演 proposer（任务提出者）和 solver（问题解决者），通过与环境交互获得可验证反馈，完全自主进化
2. **零数据训练**：不需要任何人类标注的问题-答案对，仅从一个恒等函数种子开始自举
3. **代码推理环境**：利用 Python 执行器作为开放且可验证的环境，支持三种推理模式（演绎、归纳、溯因）
4. **SOTA 性能**：在数学和编程推理任务上超越所有使用数万条人类标注数据训练的"zero"模型

## 方法详解

### Absolute Zero 范式

**与现有方法的对比**：
- **SFT**：需要 $(x, c^*, y^*)$ 三元组（问题、推理链、答案），依赖人类专家或强模型
- **RLVR**：需要 $(x, y^*)$ 对（问题、答案），仍依赖人类标注的学习分布
- **Absolute Zero**：完全不需要外部数据，模型自主提出任务并学习

**核心目标函数**：

$$\mathcal{J}(\theta) = \max_{\theta} \mathbb{E}_{z \sim p(z)} \left[ \mathbb{E}_{\tau \sim \pi_\theta^{\text{propose}}(\cdot|z), (x,y^*) \sim f_e(\cdot|\tau)} \left[ \lambda r^{\text{propose}}_e(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|x)} [r^{\text{solve}}_e(y, y^*)] \right] \right]$$

其中：
- $\pi_\theta^{\text{propose}}$：提出任务 $\tau$
- $f_e(\cdot|\tau)$：环境验证并构造有效问题 $(x, y^*)$
- $\pi_\theta^{\text{solve}}$：解决问题
- $r^{\text{propose}}$：可学习性奖励（learnability reward）
- $r^{\text{solve}}$：解答正确性奖励

### AZR 系统设计

**1. 三种推理模式**

基于代码三元组 $(p, i, o)$（程序、输入、输出），AZR 学习三种推理：

- **演绎 (Deduction)**：给定 $(p, i)$ 预测 $o$
  - Proposer 生成 $(p, i)$，环境执行得到 $o$
  - Solver 预测 $o_\pi$，验证 $o_\pi = o$

- **溯因 (Abduction)**：给定 $(p, o)$ 推断 $i$
  - Proposer 生成 $(p, i)$，环境执行得到 $o$
  - Solver 预测 $i_\pi$，验证 $p(i_\pi) = o$（输出等价性）

- **归纳 (Induction)**：给定输入输出样例 $\{(i^n, o^n)\}$ 合成程序 $p$
  - Proposer 从已有程序生成 $N$ 个输入和描述 $m$
  - Solver 看到前半部分样例，预测程序 $p_\pi$，用后半部分验证

**2. 奖励设计**

**Proposer 奖励**（可学习性）：

$$r_{\text{propose}} = \begin{cases} 0, & \text{if } \bar{r}_{\text{solve}} = 0 \\ 1 - \bar{r}_{\text{solve}}, & \text{otherwise} \end{cases}$$

其中 $\bar{r}_{\text{solve}}$ 是用当前 solver 对任务进行 $G=8$ 次蒙特卡洛采样的平均成功率。这鼓励模型提出"既不太简单也不太难"的任务。

**Solver 奖励**：

$$r_{\text{solve}} = \mathbb{I}_{(y = y^*)}$$

**综合奖励**（含格式惩罚）：

$$R(y_\pi) = \begin{cases} r_{\text{role}} & \text{格式正确} \\ -0.5 & \text{答案错误但格式正确} \\ -1 & \text{格式错误} \end{cases}$$

**3. 任务验证与过滤**

提出的任务需通过三重检查：
1. **程序完整性**：Python 执行无错误
2. **程序安全性**：禁用 `os.sys, sys, shutil` 等危险包
3. **确定性检查**：同一输入执行两次，输出必须相同（$j=2$）

**4. Task-Relative REINFORCE++ (TRR++)**

针对多任务学习，为 6 种任务-角色组合（3 种任务 × 2 种角色）分别计算基线：

$$A^{\text{norm}}_{\text{task,role}} = \frac{r - \mu_{\text{task,role}}}{\sigma_{\text{task,role}}}$$

这在 per-question baseline（GRPO）和全局 baseline（REINFORCE++）之间插值。

**5. 自举初始化**

从单个"零三元组"开始：
```python
def identity(x):
    return x
# input: 42, output: 42
```

用基础模型生成 $B \times 4$ 个种子三元组（$B$ 为批大小），无模型更新。

### 训练算法

每轮迭代：
1. **PROPOSE 阶段**：生成 3 种任务各 $B$ 个，验证并加入 buffer
2. **SOLVE 阶段**：从 buffer 采样任务，模型尝试解决
3. **UPDATE 阶段**：用 TRR++ 更新模型参数

超参数：batch size = 64×6，学习率 = 1e-6，训练 500 步，$K=6$ 个参考样例。

## 实验结果

### 主要结果（Qwen2.5-7B 系列）

| 模型 | 基座 | 数据量 | 代码平均 | 数学平均 | 总平均 |
|------|------|--------|----------|----------|--------|
| Qwen2.5-7B | Base | - | 52.0 | 27.5 | 39.8 |
| Qwen2.5-7B-Coder | Coder | - | 56.6 | 23.9 | 40.2 |
| CodeR1-LC2k | Ins | 2k | 60.5 | 35.6 | 48.0 |
| ORZ | Base | 57k | 55.6 | 41.6 | 48.6 |
| PRIME-Zero | Coder | 484k | 37.2 | **45.8** | 41.5 |
| **AZR (Ours)** | Base | **0** | 55.2 (+3.2) | 38.4 (+10.9) | 46.8 (+7.0) |
| **AZR (Ours)** | Coder | **0** | **61.6 (+5.0)** | 39.1 (+15.2) | **50.4 (+10.2)** |

**关键发现**：
- AZR-Coder-7B 达到 SOTA，超越所有使用人类数据的模型 1.8 个百分点
- 在代码任务上超越专门训练的模型 0.3 个百分点
- 数学性能提升 10.9-15.2 个百分点，远超其他代码模型的跨域迁移（平均仅 0.65）

### 模型规模效应

| 模型 | 代码提升 | 数学提升 | 总提升 |
|------|----------|----------|--------|
| AZR-Coder-3B | +3.7 | +7.7 | +5.7 |
| AZR-Coder-7B | +5.0 | +15.2 | +10.2 |
| AZR-Coder-14B | +8.2 | +18.2 | +13.2 |

**规律**：更大的模型获得更大的增益，14B 模型在 500 步后仍在持续提升。

### 其他模型家族

在 Llama3.1-8B 上：
- SimpleRL（8.5k 数据）：总平均 20.5 (+4.5)
- AZR（0 数据）：总平均 22.8 (+6.8)，超越使用数据的方法

### 涌现行为

1. **代码先验放大推理**：Qwen-Coder-7B 初始数学性能比 Qwen-7B 低 3.6 分，训练后反超 0.7 分
2. **注释作为中间计划**：归纳任务中自然涌现 ReAct 风格的"注释+代码"交替模式
3. **认知行为分化**：
   - 溯因任务：试错行为，token 增长最多
   - 演绎/归纳：逐步推理，token 增长适中
4. **安全警报**：Llama3.1-8B 偶尔产生令人担忧的思维链（"uh-oh moment"）

## 与其他方法的对比

| 范式 | 需要问题 | 需要答案 | 需要推理链 | 任务分布 |
|------|----------|----------|------------|----------|
| SFT | ✓ | ✓ | ✓ | 人类定义 |
| RLVR (DeepSeek-R1) | ✓ | ✓ | ✗ | 人类定义 |
| STaR | ✓ | ✓ | 部分自生成 | 人类定义 |
| **Absolute Zero** | ✗ | ✗ | ✗ | **模型自主进化** |

**与 AlphaZero 的联系**：
- AlphaZero：在固定游戏规则下自博弈
- AZR：在开放代码空间中自博弈，环境（Python 执行器）提供可验证反馈

**与 STaR 的区别**：
- STaR：需要人类提供问题和答案，模型生成推理链
- AZR：连问题都由模型自己提出，完全零数据

## 局限性

1. **安全性问题**：Llama3.1-8B 出现不安全的思维链，需要安全对齐研究
2. **环境限制**：当前仅限于代码执行器，未探索其他可验证环境（形式化数学、物理模拟器等）
3. **确定性约束**：仅支持确定性程序，排除了随机算法
4. **可学习性估计**：用 8 次采样估计任务难度较粗糙，可能不够精确
5. **探索机制**：任务空间探索依赖简单的"生成不同任务"提示，缺乏系统化探索策略
6. **计算成本**：需要 A800 GPU 集群训练 3-5 天

## 关键洞察

1. **任务空间探索的重要性**：传统 RL 探索解空间，AZR 探索"学什么"的问题空间，这是更元层次的探索
2. **环境作为真理来源**：可验证环境（代码执行器）避免了神经奖励模型的 reward hacking 问题
3. **代码作为通用推理媒介**：图灵完备性 + 可执行性使代码成为理想的推理训练环境
4. **跨域泛化的惊人效果**：在代码任务上训练，数学能力提升 15.2 分，远超直接在数学数据上训练的模型
5. **规模定律**：AZR 的增益随模型规模增长，暗示该范式可能受益于持续扩展
6. **经验的时代**：从"模仿人类数据"转向"通过与环境交互积累经验"，类似人类学习方式

**未来方向**：
- 扩展到其他可验证环境（形式化证明、物理模拟、真实世界）
- 多模态推理、具身 AI
- 动态学习 $f$ 函数（任务构造函数）
- 更好的学习进度估计（如 MAGELLAN）
- 探索奖励设计
- 安全对齐机制
