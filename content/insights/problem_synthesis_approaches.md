---
tags:
  - insight
time: 2026-03-05T21:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_self_questioning_lm.md
  - ../knowledge/summary_self_challenging_agents.md
  - ../knowledge/summary_variational_problem_synthesis.md
  - ../knowledge/summary_star_gate_clarifying_questions.md
---

# 问题合成的四种范式：从非对称博弈到变分生成

Group 5 论文揭示了"问题合成"（problem synthesis）这一被长期忽视的能力维度。正如用户笔记所指出："什么是一个好答案大家总是能很快看出来，但什么问题才是一个好问题没有定论也很难评判"。这四篇工作提供了四种截然不同的问题合成范式，每种都对应不同的难度控制机制和验证策略。

## 四种问题合成范式

**范式一：非对称自博弈（SQLM）**

SQLM 通过 Proposer-Solver 的非对称博弈实现自适应难度控制。核心机制是奖励 Proposer 生成"部分 Solver 能解决"的问题：

$$\mathcal{R}_P(x) = \begin{cases}
1 & \text{if } 0 < |\{y_i : y_i = y_{\text{maj}}\}| < N \\
0 & \text{otherwise}
\end{cases}$$

这个设计自动形成课程学习：
- 初期：简单的三位数加减法
- 中期：包含乘除的混合运算
- 后期：包含括号和指数的复杂表达式

**验证策略**：根据生成-验证差距选择方法
- 小差距（算术、代数）：多数投票作为伪标签
- 大差距（编程）：单元测试验证

**实验结果**：Qwen2.5-3B 在算术任务上提升 14%，代数任务上提升 16%，编程任务上提升 7%

**范式二：Code-as-Task（Self-Challenging Agents）**

SCA 用代码表达任务的所有组件，确保可行性和可验证性。每个任务包含四个组件：

1. **指令**：发送给执行者的任务描述
2. **验证函数**：可执行的 Python 代码
3. **示例解决方案**：必须通过验证函数
4. **失败案例**：三个不应通过验证的案例

**自动过滤机制**：
- 验证函数可运行：47.7% 通过率
- 示例解决方案通过验证：9.5% 通过率
- 失败案例全部不通过：5.2% 最终通过率

**关键洞察**：严格的自动过滤（5.2% 通过率）确保训练数据质量，即使挑战者经常失败也能收集足够的高质量任务。

**实验结果**：Llama-3.1-8B 在自我改进设置下成功率翻倍（12.0% → 23.5%），在蒸馏设置下平均提升 20.2%

**范式三：变分问题合成（SvS）**

SvS 基于正确解答合成语义一致但结构不同的变分问题。核心洞察：正确解答 $y_i$ 包含原问题 $x$ 的全部信息，可用于逆向映射生成变分问题。

**难度控制**：只增强表现不佳的问题（准确率在 $[\mathrm{acc}_{\mathrm{l}}, \mathrm{acc}_{\mathrm{h}}]$ 范围内），聚焦于策略能力边界。

**奖励塑形**：防止问题退化
$$\mathbf{R}_{\mathrm{v}}(\hat{x}_i^j) = \mathbb{I}\left(\hat{\mathrm{acc}}_{\mathrm{l}} \le \mathrm{Acc}(\hat{x}_i^j, a) \le \hat{\mathrm{acc}}_{\mathrm{h}}\right)$$

过于简单（全对）或无法验证（全错）的变分问题获得负奖励。

**实验结果**：在竞赛级 AIME 24 和 AIME 25 基准上，Pass@32 分别获得 18.3% 和 22.8% 的绝对提升，同时维持稳定的策略熵，避免训练坍塌。

**范式四：澄清性问题生成（STaR-GATE）**

STaR-GATE 将问题合成应用于交互式场景——当任务模糊时主动提问以获取用户偏好信息。

**目标函数**：最大化预训练模型对黄金回复的对数概率
$$J(Q, R, T, U) = \sum_{i,j} \mathbb{E}_{s_{ij}} \log p_{Q_{BASE}}(g_{ij} | t_i, s_{ij})$$

其中 $s_{ij} = [q_{ij1}, h_{ij1}, \ldots, q_{ijk}, h_{ijk}]$ 是模拟对话。

**响应正则化**：必须同时训练提问和回答，否则模型会退化为"只会提问的机器"。

**实验结果**：迭代 2 后胜率达到 72%（峰值），自我 Oracle 实验显示足够强的模型可能不需要更强的 Oracle。

## 跨范式的共同模式

**模式一：能力边界对齐**

所有四种方法都强调在模型能力边界生成问题：
- SQLM：奖励部分正确的问题（既不太简单也不太困难）
- SCA：CaT 过滤保留中等难度任务
- SvS：只增强准确率在特定范围内的问题
- STaR-GATE：通过对数概率选择最有效的问题

这与用户笔记中的"out-of-distribution 探索"完美契合——好问题应该在分布边界，而非分布中心。

**模式二：验证机制的分层**

四种方法形成了验证可靠性的谱系：
- **最可靠**：SCA 的代码执行 + 失败案例（完全消除假阳性）
- **较可靠**：SQLM 的单元测试（大生成-验证差距场景）
- **中等可靠**：SQLM 的多数投票（小生成-验证差距场景）
- **较弱**：STaR-GATE 的对数概率（依赖 Oracle 模型）

**模式三：在线生成优于预生成**

SQLM 和 SvS 都证明了在线迭代生成优于批量预生成：
- SQLM：在线生成的问题多样性显著高于预生成（PCA 可视化）
- SvS：在线问题增强维持稳定的策略熵，避免训练坍塌

原因：模型难以从抽象指令（"生成多样化问题"）中操作化多样性，自博弈提供了具体的、量化的难度信号。

## Evidence

- [SQLM](../knowledge/summary_self_questioning_lm.md): 非对称自博弈，自适应难度控制，算术 +14%，代数 +16%，编程 +7%
- [Self-Challenging Agents](../knowledge/summary_self_challenging_agents.md): Code-as-Task 四组件，5.2% 严格过滤，自我改进成功率翻倍（12.0% → 23.5%）
- [SvS](../knowledge/summary_variational_problem_synthesis.md): 变分问题合成，AIME 24 Pass@32 +18.3%，AIME 25 Pass@32 +22.8%，维持策略熵
- [STaR-GATE](../knowledge/summary_star_gate_clarifying_questions.md): 澄清性问题生成，迭代 2 后胜率 72%，响应正则化防止退化

## Implications

这四种范式为用户笔记中的核心问题"什么问题才是一个好问题"提供了可操作化的答案：

**好问题的三个标准**：
1. **难度适中**：在模型能力边界，既不太简单也不太困难
2. **可验证**：有可靠的正确性验证机制（代码执行 > 单元测试 > 多数投票 > 神经评估）
3. **多样性**：在线生成，随模型能力动态演化

**问题合成的层次结构**：
- **Level 1**：固定难度问题生成（传统数据增强）
- **Level 2**：自适应难度问题生成（SQLM、SvS）
- **Level 3**：交互式问题生成（STaR-GATE）
- **Level 4**：环境驱动问题发现（结合 Absolute Zero 的代码执行器、SPICE 的文档语料库）

**与环境驱动学习的结合**：

将 Group 5 的问题合成能力与 Group 2-4 的环境验证结合，可以实现用户愿景的完整循环：
1. **独立发现问题**：SQLM 的 Proposer + SCA 的 Code-as-Task
2. **独立解决问题**：SQLM 的 Solver + SCA 的 Executor
3. **独立使用工具验证**：Absolute Zero 的代码执行器 + SCA 的验证函数

关键突破点：将 SCA 的 Code-as-Task 范式扩展到非代码领域——用形式化语言（如逻辑公式、数学证明）表达任务的所有组件，实现跨域的可验证问题合成。
