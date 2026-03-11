# 论文索引 Paper Index

> 本目录收录的论文总结，按主题和标签分类索引。

---

## 📊 统计概览

| 类别 | 数量 |
|------|------|
| 总论文数 | 3 |
| 自蒸馏 (Self-Distillation) | 3 |
| 强化学习 (RL) | 2 |
| 持续学习 (Continual Learning) | 1 |

---

## 🏷️ 标签索引

### 按方法分类

| 标签 | 论文 |
|------|------|
| `self-distillation` | OPSD, SDPO, SDFT |
| `on-policy` | OPSD, SDPO, SDFT |
| `knowledge-distillation` | OPSD, SDPO, SDFT |
| `reinforcement-learning` | SDPO, SDFT |
| `policy-gradient` | SDPO |
| `inverse-rl` | SDFT |

### 按应用场景分类

| 标签 | 论文 |
|------|------|
| `math-reasoning` | OPSD |
| `code-generation` | SDPO |
| `continual-learning` | SDFT |
| `test-time-training` | SDPO |
| `credit-assignment` | SDPO |

### 按模型/数据分类

| 标签 | 论文 |
|------|------|
| `qwen` | OPSD, SDPO, SDFT |
| `llm-post-training` | OPSD, SDPO, SDFT |
| `no-external-teacher` | OPSD, SDPO, SDFT |

---

## 📑 论文列表

### 1. OPSD - On-Policy Self-Distillation

**Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models**

| 属性 | 内容 |
|------|------|
| 📄 文件 | [summary_on_policy_self_distillation.md](./summary_on_policy_self_distillation.md) |
| 🔗 arXiv | [2601.18734](https://arxiv.org/abs/2601.18734) |
| 🏛️ 机构 | UCLA, Meta Superintelligence Labs, HKU |
| 📅 会议 | ICML 2026 Submission |

**标签:** `self-distillation` `on-policy` `math-reasoning` `qwen` `no-external-teacher` `token-efficiency`

**核心思想:** 利用参考答案作为特权信息，让模型"合理化"正确解法并教授没有答案的自己。

**关键结果:**
- 在数学推理任务上超越 GRPO
- 实现 **4-8× token 效率提升**
- 每问题仅需 1 个 rollout（vs GRPO 的 8 个）

---

### 2. SDPO - Self-Distillation Policy Optimization

**Reinforcement Learning via Self-Distillation**

| 属性 | 内容 |
|------|------|
| 📄 文件 | [summary_sdpo_rl_self_distillation.md](./summary_sdpo_rl_self_distillation.md) |
| 🔗 arXiv | [2601.20802](https://arxiv.org/abs/2601.20802) |
| 🏛️ 机构 | ETH Zurich, MPI, MIT, Stanford |
| 💻 代码 | [github.com/lasgroup/SDPO](https://github.com/lasgroup/SDPO) |

**标签:** `self-distillation` `on-policy` `reinforcement-learning` `code-generation` `credit-assignment` `test-time-training` `rich-feedback`

**核心思想:** 利用环境的丰富反馈（错误信息、测试结果）让模型"事后回顾"并识别错误，实现密集信用分配。

**关键结果:**
- LiveCodeBench v6: **48.8%** vs GRPO 41.2%
- 超越 Claude Sonnet 4 和 Opus 4
- 响应长度减少 **3-7×** 同时准确率更高
- Test-Time Self-Distillation 加速困难问题求解 **3×**

---

### 3. SDFT - Self-Distillation Fine-Tuning

**Self-Distillation Enables Continual Learning**

| 属性 | 内容 |
|------|------|
| 📄 文件 | [summary_sdft_continual_learning.md](./summary_sdft_continual_learning.md) |
| 🔗 arXiv | [2601.19897](https://arxiv.org/abs/2601.19897) |
| 🏛️ 机构 | MIT, Improbable AI Lab, ETH Zurich |
| 🌐 网站 | [idanshenfeld.com/SDFT](http://idanshenfeld.com/SDFT) |

**标签:** `self-distillation` `on-policy` `continual-learning` `inverse-rl` `catastrophic-forgetting` `skill-learning` `knowledge-acquisition`

**核心思想:** 利用专家示范作为条件，让模型通过上下文学习成为自己的教师，实现持续学习而不遗忘。

**关键结果:**
- 新任务准确率更高 + 先前能力保持
- 顺序学习 3 个技能无退化
- 知识获取 OOD 准确率 **98%**（vs SFT 80%）
- 等价于隐式逆强化学习

---

## 🔄 三篇论文对比

| 方面 | OPSD | SDPO | SDFT |
|------|------|------|------|
| **教师信息** | 参考答案 $y^\star$ | 环境反馈 $f$ | 专家示范 $c$ |
| **主要目标** | 提高推理能力 | 密集信用分配 | 持续学习 |
| **应用场景** | 数学推理 | 代码/科学推理 | 技能/知识获取 |
| **遗忘关注** | 次要 | 次要 | **核心** |
| **测试时应用** | ✗ | ✓ (TTT) | ✗ |
| **理论解释** | - | 策略梯度扩展 | 隐式 IRL |

### 共同核心

$$\text{教师} = \pi_\theta(\cdot \mid x, \text{额外信息})$$
$$\text{学生} = \pi_\theta(\cdot \mid x)$$

> **通过给模型提供额外信息（答案/反馈/示范），同一个模型可以作为自己的教师。**

---

## 🔍 快速查找

### 我想了解...

| 需求 | 推荐论文 |
|------|----------|
| 如何提高数学推理能力？ | OPSD |
| 如何利用错误反馈改进模型？ | SDPO |
| 如何避免灾难性遗忘？ | SDFT |
| 如何减少训练采样成本？ | OPSD, SDPO |
| 如何在测试时继续优化？ | SDPO (TTT) |
| 如何从示范学习而不需要奖励函数？ | SDFT |
| 如何让模型推理更简洁？ | SDPO |

---

## 📚 相关概念

| 概念 | 定义 | 相关论文 |
|------|------|----------|
| **On-Policy Learning** | 在当前策略分布上训练 | 全部 |
| **Self-Distillation** | 模型自己作为教师 | 全部 |
| **Credit Assignment** | 识别哪些 token 贡献了奖励 | SDPO |
| **Catastrophic Forgetting** | 学习新任务时遗忘旧能力 | SDFT |
| **In-Context Learning** | 通过上下文示例改变行为 | 全部 |
| **RLVR** | 可验证奖励的强化学习 | SDPO |
| **RLRF** | 丰富反馈的强化学习 | SDPO |

---

*最后更新: 2026-02-04*
