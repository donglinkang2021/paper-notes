---
tags:
  - insight
time: 2026-03-15T03:40:13+00:00
author: Linkdom
sources:
  - ../knowledge/summary_rls_razor_forgets_less.md
  - ../knowledge/summary_retaining_by_doing_on_policy_data.md
---

# On-policy 数据、forward KL 与“少遗忘”的机制统一

两篇工作从互补视角解释了一个经验现象：**在线/在策略（on-policy）的强化学习式训练，比离线/异策略（off-policy）的监督式训练更不容易遗忘旧任务**。

- [RL's Razor: Why Online Reinforcement Learning Forgets Less](../knowledge/summary_rls_razor_forgets_less.md) 给出一个“领先指标”：在新任务分布 $\tau$ 上的 **forward KL**
  $$\mathbb{E}_{x\sim\tau}\big[D_{\mathrm{KL}}(\pi_0(\cdot\mid x)\,\Vert\,\pi(\cdot\mid x))\big]$$
  对遗忘程度有强预测力。并提出 **RL’s Razor**：在存在多解的任务上，on-policy policy gradient 具有一种偏置，倾向于在最优解集合 $P^*$ 内找到与初始策略 $\pi_0$ **“KL 距离最小”** 的解（形式化为 reverse KL 最小化）。

- [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](../knowledge/summary_retaining_by_doing_on_policy_data.md) 则通过系统消融指出“抗遗忘”的主要来源并不是“RL 更新规则本身”，而是 **训练数据是否来自当前策略（on-policy）**。它进一步提出了可操作的工程近似：例如 **Iterative-SFT**（每轮用当前模型采样、再用这些样本做 SFT）能以较低代价逼近 on-policy 的分布对齐效果。

一个有用的统一理解是：

1. **遗忘是“分布错配”累积的外显**：如果训练只在某个固定数据分布（off-policy）上进行，当模型参数被推动去适应新任务时，旧任务上策略分布可能发生大幅漂移；此时在新任务分布上测得的 forward KL 会显著增大，并与旧任务性能下降同步。
2. **on-policy 训练把“漂移”限制在更保守的方向**：由于采样来自当前策略，优化会更强地惩罚那些会让策略与自身近期行为不一致的大幅更新（可被 KL 正则/信赖域视角理解），从而在多解空间中倾向于选择更接近 $\pi_0$ 的解。
3. **Iterative-SFT 是把“on-policy 性质”拆出来单独复用**：它不需要完整的 RL 交互与高方差梯度，只要能确保每一轮监督数据持续刷新、紧贴当前策略，就能获得部分“保守更新/少遗忘”的收益。

## Evidence

- [RL's Razor](../knowledge/summary_rls_razor_forgets_less.md)：
  - 经验遗忘律：forward KL 在新任务分布上是遗忘的强预测指标。
  - 理论偏置：on-policy RL 在多解任务中倾向于 KL-minimal 解（对 $\pi_0$ 更保守）。
- [Retaining by Doing](../knowledge/summary_retaining_by_doing_on_policy_data.md)：
  - 消融结论：是否 on-policy 数据是 RL 抗遗忘的主因。
  - 方法建议：Iterative-SFT/每轮刷新数据可作为近似 on-policy 的省算力替代。

## Implications

- **设计训练流程时，把“数据是否 on-policy”当作一等公民**：与其争论 RL vs SFT，不如先问“训练数据是否紧贴当前策略分布”。
- **监控指标**：在新任务分布 $\tau$ 上的 forward KL 可作为早期告警信号，用于判断是否正在走向严重遗忘。
- **工程折中**：当 full on-policy RL 成本太高时，可以用 Iterative-SFT / 生成-过滤-再训练的闭环来获得部分保守性与留存能力；关键是“每轮刷新、不过度复用旧样本”。
