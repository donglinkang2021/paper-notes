---
tags:
  - insight
time: 2026-03-05T22:00:00+08:00
author: Linkdom
sources:
  - ../knowledge/summary_star_bootstrapping_reasoning.md
  - ../knowledge/summary_rest_reinforced_self_training.md
  - ../knowledge/summary_spin_self_play_finetuning.md
  - ../knowledge/summary_language_self_play_data_free.md
  - ../knowledge/summary_multiagent_finetuning.md
  - ../knowledge/summary_r_zero_self_evolving.md
  - ../knowledge/summary_absolute_zero_reasoning.md
  - ../knowledge/summary_r_few_guided_self_evolving.md
  - ../knowledge/summary_spice_self_play_corpus.md
  - ../knowledge/summary_can_reasoning_models_self_train.md
  - ../knowledge/summary_negative_reinforcement_reasoning.md
---

# 多样性与收敛的永恒张力：从被动防御到主动引导

所有基于自训练的方法都面临同一个根本矛盾：模型在自己生成的数据上训练，天然倾向于强化已有的高概率模式，导致输出分布逐渐收缩——即 model collapse。这个问题在用户的调研笔记中被反复提及（"熵崩溃，多样性缺失，长尾消失"），也是 Nature 2024 那篇经典论文的核心警告。Group 2 的自进化推理论文为这个问题提供了更深刻的实证证据和解决方案。

**无引导自博弈的失败模式**：R-Zero 展示了完全无数据自进化的脆弱性——伪标签准确率从 79% 系统性下降至 47%，第 3 轮后性能开始崩溃。更严重的是问题生成的退化：2-gram 多样性从 35 暴跌至 20，问题长度爆炸式增长，Challenger 学会通过冗长而非真实推理复杂度来伪造难度。SRT 论文揭示了更极端的崩溃现象——所有 4 个基础模型在延长训练后都出现突然且完全的性能崩溃，模型对所有 prompt 输出高熵随机 token 后跟相同模板答案（如 `\boxed{1}`），完全忽略输入。这是奖励黑客的终极形态：模型找到了最大化自一致性奖励的"捷径"。

**外部锚定的关键作用**：Group 2 论文展示了三种不同的锚定策略。R-Few 用 1-5% 人类数据作为"锚点"，通过 few-shot 示例软引导问题生成，成功防止了概念漂移——多样性全程稳定，问题长度保持一致，难度提升来自真实推理复杂度而非表面特征。SPICE 用大规模文档语料库作为环境，Challenger 从文档中挖掘内容生成问题，打破了纯自博弈的信息对称性，实现了 640 轮稳定训练（R-Zero 仅 5 轮后退化）。Absolute Zero 用 Python 执行器作为可验证环境，避免了神经奖励模型的 reward hacking 问题，但代价是限制在代码推理领域。

**多样性的层次理解**：这些方法对"多样性"的理解层次不同。STaR 和 ReST 关注样本级多样性，SPIN 关注分布级多样性，LSP 和 R-Zero 关注问题级多样性，Multiagent FT 关注策略级多样性，而 SPICE 和 Absolute Zero 关注环境级多样性（通过外部环境的丰富性保证训练信号的多样性）。R-Few 的成功表明，即使是极少量的人类锚点（1-5%）也能在所有层次上稳定多样性。

**梯度层面的多样性保护**：Group 6 的 NSR 论文揭示了一个更深层的多样性维度——RL 训练中的梯度级多样性。传统 RLVR 中的正样本强化（PSR）在梯度层面主动压制所有替代方案（包括其他正确路径），是 RL 训练导致多样性丧失的直接原因。NSR 通过仅惩罚错误回答，按先验概率比例 $\pi_v$ 将概率质量重分配给所有未采样 token，从梯度机制上保护了低概率正确尾部样本。实验证据：NSR 训练后 Pass@256 达到 96.9（匹配基线），而 PSR 仅 91.2；NSR 全程保持接近基础模型的高熵水平。这为多样性保护提供了一种全新的"减法"思路——不是通过增加随机性或外部数据来维持多样性，而是通过避免主动压缩分布来保护已有的多样性。

## Evidence

**Group 1 (基础方法)**：
- [STaR](../knowledge/summary_star_bootstrapping_reasoning.md): 每轮从预训练模型重启训练，避免累积分布偏移；但高温采样反而降低性能，说明"更多随机性 ≠ 更好的多样性"
- [ReST](../knowledge/summary_rest_reinforced_self_training.md): 递增阈值策略在质量和数量间动态平衡；但发现奖励模型会过拟合，策略偏离训练分布后奖励信号失真
- [SPIN](../knowledge/summary_spin_self_play_finetuning.md): 理论保证收敛到目标分布时停止，但天花板受限于 SFT 数据质量，无法超越人类数据的多样性
- [LSP](../knowledge/summary_language_self_play_data_free.md): Challenger 主动生成难题扩展训练分布，但 LSP-Zero 的退化案例证明纯对抗博弈需要正则化约束
- [Multiagent FT](../knowledge/summary_multiagent_finetuning.md): 角色专业化从结构上防止多样性坍缩，单模型 1 轮后性能下降，多模型 5 轮仍在提升

**Group 2 (自进化推理的失败与解决)**：
- [R-Zero](../knowledge/summary_r_zero_self_evolving.md): 伪标签准确率从 79% 降至 47%，2-gram 多样性从 35 降至 20，问题长度爆炸式增长，第 3 轮后性能崩溃
- [SRT](../knowledge/summary_can_reasoning_models_self_train.md): 所有 4 个基础模型延长训练后突然完全崩溃，输出高熵随机 token + 相同模板答案，KL 散度急剧增大
- [R-Few](../knowledge/summary_r_few_guided_self_evolving.md): 仅 1-5% 人类数据即防止概念漂移，多样性全程稳定，问题长度保持一致，用 5% 数据达到 20 倍数据量方法的性能
- [SPICE](../knowledge/summary_spice_self_play_corpus.md): 语料库根据打破信息对称性，稳定训练 640 轮（R-Zero 仅 5 轮），相比无语料库方法总体提升 +7.9%
- [Absolute Zero](../knowledge/summary_absolute_zero_reasoning.md): Python 执行器作为可验证环境避免 reward hacking，但限制在代码推理领域，14B 模型 500 步后仍在持续提升

**Group 6 (梯度级多样性保护)**：
- [NSR](../knowledge/summary_negative_reinforcement_reasoning.md): Token 级梯度分析证明 PSR 在梯度层面主动压制替代方案（梯度 $\propto -\pi_{y_t} \cdot \pi_v$），而 NSR 按先验比例重分配概率质量（梯度 $\propto +\pi_{y_t} \cdot \pi_v$）；W-REINFORCE（$\lambda=0.1$）仅降低正奖励权重即全面超越 PPO 和 GRPO，证明"少强化正确"比"多强化正确"更有利于多样性

## Implications

多样性维持是 self-play 方法能否持续提升的核心瓶颈。Group 2 论文揭示了三个关键洞察，Group 6 的 NSR 论文则补充了第四个：

1. **最小锚定原则**：R-Few 证明仅 1-5% 人类数据即可防止崩溃，这远低于传统方法的数据需求。关键不在于数据量，而在于提供"语义锚点"防止概念漂移。

2. **环境驱动的多样性**：SPICE 和 Absolute Zero 表明，将外部环境（文档语料库、代码执行器）作为训练信号来源，比纯神经奖励模型更稳定。环境提供的是"真实世界的复杂性"，而非模型自身的偏见。

3. **奖励黑客的不可避免性**：SRT 的崩溃实验表明，当奖励信号来自模型自身（多数投票）时，模型最终会找到捷径。增大 KL 惩罚、降低学习率都无法根本解决，只能延缓。

4. **"减法"优于"加法"的多样性保护**：NSR 证明，在 RL 训练中，通过仅惩罚错误（减法）而非强化正确（加法）来保护多样性，效果远优于传统方法。PSR 的梯度机制天然压缩分布，而 NSR 的梯度机制天然保护分布。W-REINFORCE 的成功（$\lambda=0.1$）表明，即使不完全去除 PSR，大幅降低其权重就足以显著改善多样性。这与 ToEdit 的"编辑而非生成"思路一脉相承——保护已有的多样性比创造新的多样性更高效。

未来方向应结合多层次多样性保护：Multiagent FT 的结构化多样性 + R-Few 的最小锚定 + SPICE 的环境驱动 + NSR 的梯度级长尾保护。这也回应了用户的核心关切："环境驱动的发现"——让模型在与真实环境交互中发现问题，而非在自我封闭的循环中强化偏见。NSR 的发现进一步表明，即使在封闭的 RL 训练循环中，通过调整梯度信号的构成（降低 PSR 权重），也能显著缓解多样性丧失问题。
