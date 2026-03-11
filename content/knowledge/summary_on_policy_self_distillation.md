# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

**Paper ID:** arXiv 2601.18734
**Authors:** Siyan Zhao (UCLA), Zhihui Xie (HKU), Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen (Meta Superintelligence Labs), Aditya Grover (UCLA)
**Venue:** ICML 2026 Submission

---

## 核心思想

本文提出了 **On-Policy Self-Distillation (OPSD)**，一种让单个模型同时扮演"教师"和"学生"角色的自蒸馏框架。核心洞察是：一个足够强大的 LLM 可以通过访问正确答案来"合理化"解题步骤，并教授没有访问答案权限的"弱版本自己"。

### 人类学习的类比

当学生解题错误后，可以查看正确答案，理解推理步骤，并找出自己推理失败的地方。类似地，LLM 的"评估"通常比"生成"更容易，"合理化"（解释一个给定的正确答案）同样比从头生成更简单。

---

## 方法详解

### 教师与学生策略

从同一个 LLM $p_\theta$ 实例化两个策略：

- **学生策略** $p_S(\cdot | x)$：只看到问题 $x$（与推理时条件相同）
- **教师策略** $p_T(\cdot | x, y^\star)$：同时看到问题 $x$ 和参考答案 $y^\star$（特权信息）

关键：两个策略共享相同参数 $\theta$，仅在输入条件上不同。

### 训练流程

1. 学生策略生成 on-policy 响应：$\hat{y} \sim p_S(\cdot | x)$
2. 两个策略在学生生成的每个位置评估下一个 token 的分布
3. 最小化教师和学生分布之间的逐 token 散度

### 损失函数

$$\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{(x, y^\star) \sim \mathcal{S}} \mathbb{E}_{\hat{y} \sim p_S(\cdot|x)} \sum_{n=1}^{|\hat{y}|} D\left(p_T(\cdot | x, y^\star, \hat{y}_{<n}) \| p_S(\cdot | x, \hat{y}_{<n})\right)$$

其中 $D$ 是散度度量，论文使用 **广义 Jensen-Shannon 散度** ($\text{JSD}_{\beta=0.5}$)：

$$\text{JSD}_\beta(p_T \| p_S) = \beta D_{KL}(p_T \| m) + (1-\beta) D_{KL}(p_S \| m)$$

其中 $m = \beta p_T + (1-\beta) p_S$。

### 算法伪代码

**输入:** 数据集 $\mathcal{S} = \{(x_i, y_i^\star)\}$，模型 $p_\theta$，散度 $D$

**定义:**
- 学生策略：$p_S(\cdot \mid x) := p_\theta(\cdot \mid x)$
- 教师策略：$p_T(\cdot \mid x, y^\star) := p_\theta(\cdot \mid x, y^\star)$ （相同参数，不同条件）

**While not converged:**
1. 采样 minibatch $\mathcal{B} \subset \mathcal{S}$
2. For each $(x, y^\star) \in \mathcal{B}$：
   - 采样 on-policy 响应：$\hat{y} \sim p_S(\cdot \mid x)$
   - 计算逐 token 散度：$\ell(x, y^\star) = \frac{1}{|\hat{y}|} \sum_{n=1}^{|\hat{y}|} D\left(p_T(\cdot \mid \hat{y}_{<n}, x, y^\star) \,\|\, p_S(\cdot \mid \hat{y}_{<n}, x)\right)$
3. 批量损失：$\mathcal{L}_{\mathrm{OPSD}}(\theta) = \mathrm{mean}(\ell)$
4. 更新：$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\mathrm{OPSD}}(\theta)$

**Return** $\theta$ （用于推理时的 $p_S(\cdot \mid x)$）

---

## 与其他方法的对比

| 特性 | SFT/离策略蒸馏 | GRPO | 在策略蒸馏 | **OPSD (本文)** |
|------|--------------|------|----------|----------------|
| On-Policy 数据 | ✗ | ✓ | ✓ | ✓ |
| 密集学习信号 | ✓ | ✗ | ✓ | ✓ |
| 低采样成本 | ✓ | ✗ | ✓ | ✓ |
| 无需外部教师 | ✓ | ✓ | ✗ | ✓ |

### 各方法的局限性

- **SFT**：存在曝光偏差（exposure bias），泛化能力较弱
- **GRPO**：
  - 每个问题需要采样多个响应（8个），计算成本高
  - 奖励信号稀疏，只在序列级别提供反馈
  - 当所有样本全对或全错时，梯度信号消失
- **传统在策略蒸馏**：需要独立的（通常更大的）教师模型

---

## 实验结果

### 主要结果（Qwen3 模型系列）

在 AIME24、AIME25、HMMT25、AMO-Bench 等竞赛级数学推理基准上：

| 模型 | 方法 | 平均准确率 |
|------|------|-----------|
| Qwen3-8B | Base | 50.0% |
| Qwen3-8B | + SFT | 50.0% |
| Qwen3-8B | + GRPO | 51.3% |
| Qwen3-8B | **+ OPSD** | **52.2%** |
| Qwen3-4B | Base | 48.3% |
| Qwen3-4B | + GRPO | 49.6% |
| Qwen3-4B | **+ OPSD** | **50.6%** |

### Token 效率

- OPSD 实现 **4-8 倍** 的 token 效率提升
- GRPO：每个问题 8 个 rollout，每个 16k tokens
- OPSD：每个问题 1 个 rollout，每个 2k tokens

### 消融研究

1. **模型规模效应**：
   - 1.7B 模型：OPSD 与 GRPO 相当
   - 4B/8B 模型：OPSD 显著优于 GRPO
   - 结论：自蒸馏需要足够的模型能力来"合理化"参考答案

2. **生成长度效应**：
   - 更长的生成提供更多教师监督信号
   - 2048 和 4096 tokens 显著优于 1024 tokens

3. **散度目标对比**：
   - 全词表 logit 蒸馏优于采样 token 蒸馏
   - AIME25: 84.1% vs 82.1%
   - 但全词表计算需要更多显存

---

## 实现细节

### 训练配置

- 优化器：AdamW
- 学习率：2e-5，余弦衰减
- LoRA：rank=64, alpha=128
- 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 训练数据：OpenThoughts 数学推理子集，30K 样本
- 硬件：8×A100 GPU

### OPSD 特定配置

- 每个问题生成数：1（vs GRPO 的 8）
- 最大生成长度：2048（vs GRPO 的 16000）
- 温度：1.2
- 散度：JSD_0.5
- **关键**：教师策略固定为初始策略，而非当前更新中的策略（稳定训练，隐式正则化）

### Prompt 设计

**学生 Prompt:**
```
Problem: [问题描述]
Answer:
```

**教师 Prompt:**
```
Problem: [问题描述]
Here is a reference solution:
[参考答案 y*]
After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
```

---

## 局限性与未来方向

1. **规模限制**：实验仅限于 8B 参数，更大模型（70B+）的效果待验证
2. **未利用正确性验证**：当前框架未显式利用生成答案的正确性验证
3. **问题难度**：如果问题超出模型理解能力，教师策略即使有答案也无法提供有意义的监督
4. **未来方向**：课程学习策略——随着模型改进逐渐增加问题难度

---

## 核心贡献总结

1. 提出 OPSD 框架：单模型同时作为教师和学生，利用真实答案提供密集 token 级监督
2. 在数学推理任务上超越 SFT 和 GRPO
3. 实现 4-8 倍 token 效率提升，降低计算成本
4. 分析了模型规模、生成长度、散度目标的影响

---

## 关键洞察

> "评估比生成更容易" —— 模型在看到正确答案后进行"合理化"比从头推理更简单，这使得自蒸馏成为可能。

> "早期 token 更重要" —— 作者假设早期 token 代表更重要的分支点，因此短生成（2k tokens）也能获得有效监督。
