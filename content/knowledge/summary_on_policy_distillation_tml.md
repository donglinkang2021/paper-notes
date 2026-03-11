# On-Policy Distillation

**博客信息**

- 来源: Thinking Machines Lab
- 作者: Kevin Lu et al.
- 发布时间: October 2025
- DOI: 10.64434/tml.20251026
- 链接: [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

## 核心思想

On-Policy Distillation 是一种后训练技术，结合了强化学习(RL)的优势和监督微调(SFT)的效率：

- **从学生模型采样轨迹**（on-policy）
- **用教师模型的 log probabilities 对每个 token 评分**（dense supervision）

这样既保证了训练数据与学生分布一致，又获得了密集的监督信号。

---

## 背景：三个训练阶段

| 阶段 | 目标 |
| ---- | ---- |
| Pre-training | 通用语言能力 |
| Mid-training | 领域特定知识 |
| Post-training | 目标行为（指令遵循、推理等） |

## 两种训练范式的问题

### Off-Policy 训练 (SFT)

- 使用外部目标输出
- ✓ 提供密集奖励
- ✗ 训练在教师访问的状态上，而非学生生成的状态
- ✗ **分布不匹配 (distribution mismatch)**

### On-Policy 训练 (RL)

- 从学生 rollouts 采样
- ✓ 训练在学生生成的状态上
- ✗ 稀疏反馈：无论使用多少 token，每个 episode 只有一个 bit 的信息

---

## On-Policy Distillation 方案

### 核心流程

1. **初始化教师客户端**
2. **从学生模型采样轨迹**
3. **查询教师在采样序列上的 logprobs**
4. **设置优势函数为负 reverse KL，通过重要性采样训练**

### 损失函数：Reverse KL 散度

最小化学生和教师分布之间的逐 token reverse KL 散度：

$$D_{KL}(\pi_\theta \| \pi_{\text{teacher}}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \pi_\theta(x_{t+1} \mid x_{1..t}) - \log \pi_{\text{teacher}}(x_{t+1} \mid x_{1..t}) \right]$$

### 方法优势

| 特性 | SFT | RL | On-Policy Distillation |
| ---- | --- | -- | ---------------------- |
| 密集监督 | ✓ | ✗ | ✓ |
| On-policy | ✗ | ✓ | ✓ |
| 计算效率 | ✓ | ✗ | ✓ |

---

## 实验结果

### 数学推理 (AIME'24)

| 方法 | 准确率 | GPU 小时 |
| ---- | ------ | -------- |
| Baseline SFT (400k prompts) | 60% | - |
| RL | 68% | 17,920 |
| **On-policy distillation** | **74.4%** | **1,800** |

**关键发现**：On-policy distillation 达到最高性能，同时计算成本仅为 RL 的 1/10。

### 计算效率对比

- 相比 2M prompt SFT baseline：**9-30× 成本降低**
- 相比 RL：**50-100× 更快**学习等效策略
- 支持在单个 prompt 上多 epoch 训练而不过拟合

---

## 个性化应用：知识注入后恢复指令遵循能力

### 场景

在内部文档上进行 mid-training 会导致指令遵循能力下降：

| 阶段 | IF-eval 性能 | 知识保留 |
| ---- | ------------ | -------- |
| Mid-training 后 | 下降 | +100% 知识 |
| On-policy distillation 后 | **恢复 83%** 原始聊天能力 | 保留 **41%** 知识增益 |

### 持续学习的洞察

> SFT 在模型自己的 zero-KL 样本上训练，反而会因为有限 batch 的不匹配导致 off-policy 漂移，从而降低性能。On-policy distillation 通过固定教师、让学生向其收敛来保持稳定性。

---

## 技术洞察

### RL 的本质

作者认为 RL 主要是在**语义策略空间**中进行搜索，而非参数空间的探索。一旦策略被发现，蒸馏可以高效地传递它们，而无需建模中间学习课程。

### 类比

> "发现科学成果需要探索；而教授它们只需要表达。"

### 为什么 On-Policy 重要

- **Dense supervision**：教会模型哪些具体 token 偏离了教师
- **On-policy relevance**：在学生实际出错的状态上训练
- **数据复用**：每个 prompt 可以训练多个样本
- **持续学习**：通过行为恢复防止灾难性遗忘

---

## 核心优势总结

| 优势 | 描述 |
| ---- | ---- |
| 密集监督 | 每个 token 都有 ~N bits 的信息 |
| On-policy 相关性 | 训练在学生生成的状态上 |
| 计算效率 | 比 RL 快 50-100× |
| 数据效率 | 可在单个 prompt 上多次训练 |
| 持续学习友好 | 防止灾难性遗忘 |

---

## 与相关工作的对比

### vs Self-Distillation (OPSD)

| 方面 | On-Policy Distillation | OPSD |
| ---- | ---------------------- | ---- |
| 教师模型 | 外部（更强的）模型 | 同一模型（条件不同） |
| 特权信息 | 教师的能力 | 参考答案 |
| 适用场景 | 有强教师可用 | 无外部教师 |

### vs 传统 SFT

On-policy distillation 通过在学生生成的轨迹上训练，避免了 SFT 的分布不匹配问题。

### vs RL

On-policy distillation 提供密集的 token 级反馈，而 RL 只有稀疏的 episode 级奖励。

---

## 实现

完整实现可在 Tinker cookbook 中获取，支持 Qwen3 等模型。

---

## 与本仓库其他笔记的关联

- **summary_on_policy_self_distillation.md**: OPSD 使用同一模型的不同条件版本作为教师，而本文使用外部教师模型
- **summary_sdpo_rl_self_distillation.md**: SDPO 结合了 DPO 和自蒸馏，是另一种后训练方法
- **summary_sdft_continual_learning.md**: 讨论持续学习中的知识保留问题，与本文的个性化应用场景相关

---

## 关键洞察

> "RL 发现策略，蒸馏传递策略。" —— 一旦最优行为被发现，高效的监督学习就足以将其传授给学生模型。

> "On-policy + Dense = 最佳组合" —— 既保证训练分布一致性，又获得丰富的监督信号。
