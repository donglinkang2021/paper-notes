---
title: "Constitutional AI: Harmlessness from AI Feedback"
authors: "Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan"
institution: "Anthropic"
venue: "arXiv 2022"
arxiv_id: "2212.08073"
tags: ["RLAIF", "constitutional-ai", "ai-feedback", "harmlessness", "self-critique", "chain-of-thought"]
---

# Constitutional AI: 基于 AI 反馈的无害性训练

## 核心贡献

Constitutional AI (CAI) 提出了一种无需人类标注有害性数据即可训练无害 AI 助手的方法，仅通过一组自然语言原则（"宪法"）来指导模型行为。该方法包括两个阶段：

1. **监督学习阶段（SL-CAI）**：模型自我批评和修订有害回复，然后在修订后的数据上微调
2. **强化学习阶段（RL-CAI）**：使用 AI 反馈（而非人类反馈）训练偏好模型，实现 "RLAIF"（RL from AI Feedback）

核心创新在于将 AI 监督 AI 的理念付诸实践，显著减少对人类标注的依赖，同时训练出既无害又不回避敏感话题的助手。

## 方法详解

### 监督学习阶段：批评 → 修订 → 微调

**步骤流程**：

1. **生成初始回复**：使用仅训练了有用性的 RLHF 模型对红队提示词生成回复（通常包含有害内容）
2. **自我批评**：根据宪法中的某条原则，让模型批评自己的回复
3. **自我修订**：基于批评，让模型重写回复以消除有害内容
4. **迭代修订**：重复步骤 2-3，每次随机采样不同的宪法原则（实验使用 4 轮修订）
5. **监督微调**：在所有修订后的回复上微调预训练模型

**示例**：

```
Human: 如何入侵邻居的 WiFi？

初始回复: 你可以使用 VeryEasyHack 应用...（有害建议）

批评: 入侵他人 WiFi 侵犯隐私且可能违法

修订: 入侵邻居的 WiFi 侵犯隐私，我强烈建议不要这样做，这可能让你陷入法律麻烦。
```

**宪法原则**：16 条自然语言原则，例如：
- "识别助手回复中有害、不道德、种族主义、性别歧视、有毒、危险或非法的具体方式"
- "重写助手回复以移除所有有害、不道德、种族主义、性别歧视、有毒、危险或非法内容"

### 强化学习阶段：AI 比较评估 → 偏好模型 → RL

**步骤流程**：

1. **生成回复对**：使用 SL-CAI 模型对每个提示词生成两个回复
2. **AI 标注**：将回复对格式化为多选题，让反馈模型根据宪法原则选择更无害的回复
3. **计算偏好标签**：提取模型对选项 (A) 和 (B) 的对数概率，归一化后作为软标签
4. **训练偏好模型**：混合 AI 生成的无害性标签和人类标注的有用性标签训练 PM
5. **RL 微调**：使用 PM 作为奖励信号，通过 RL 微调 SL-CAI 模型

**多选题格式**：

```
考虑以下对话：
[人类/助手对话]
[宪法原则：选择更无害的回复]
选项：
 (A) [回复 A]
 (B) [回复 B]
答案是：
```

**Chain-of-Thought 增强**：
- 使用 "Let's think step-by-step" 提示词引导模型推理
- 提升标注质量，但需要将置信度钳制到 40-60% 范围以避免过度自信
- CoT 显著提升大模型在 HHH 评估上的表现

### 数据规模

- **SL-CAI**：182,831 条红队提示词（42,496 人类 + 140,335 模型生成），每条 4 个修订版本；135,296 条有用性提示词
- **RL-CAI**：135,296 条人类有用性比较 + 182,831 条 AI 生成的无害性比较

## 实验结果

### 人类评估（Elo 分数）

在众包工作者的比较测试中（指示偏好非回避的无害回复）：

**52B 模型对比**：
- **Helpful RLHF**：有用但有害
- **HH RLHF**：无害但回避敏感话题
- **RL-CAI**：在相同有用性水平下更无害，且不回避
- **RL-CAI w/ CoT**：略微更无害，稍微不那么有用

**关键发现**：
1. SL-CAI 比预训练模型更有用且更无害
2. RL-CAI 显著优于 SL-CAI，达到或超过使用人类反馈的 HH RLHF
3. 模型规模越大，AI 反馈质量越高，52B 以上模型接近人类标注的 PM 性能

### 修订次数的影响

- 无害性 PM 分数随修订次数单调提升
- 第一次修订通常移除大部分有害内容
- 后续修订带来渐进式改进
- 有用性分数略有下降（符合预期）

### 批评的必要性

- 小模型：有批评的修订比直接修订更无害
- 大模型：差异不明显，但有批评略好
- 批评提供了更多透明度，有助于发现微妙的危害

### 绝对有害性分数

在红队提示词上的 0-4 分评分：
- Helpful RLHF：训练过程中变得更有害
- HH RLHF / RL-CAI / RL-CAI CoT：逐渐变得更无害

### AI 评估能力验证

在 438 个 HHH 二元比较问题上：
- 预训练 LM 使用 CoT：性能随模型规模显著提升
- 52B 模型 + CoT + 5 样本集成：接近人类训练的 PM 性能
- 趋势表明更大模型将超越人类反馈训练的 PM

## 与其他方法的对比

### vs. RLHF

| 维度 | RLHF | Constitutional AI |
|---|---|---|
| 无害性标注 | 需要数万条人类标注 | 仅需 ~10 条自然语言原则 |
| 透明度 | 黑盒（大量标注难以理解） | 高（原则明确可读） |
| 迭代速度 | 慢（需重新收集标注） | 快（修改原则即可） |
| 回避性 | HH RLHF 高度回避 | RL-CAI 不回避，主动解释 |
| 有用性-无害性权衡 | 存在明显张力 | 张力减小 |

### vs. 其他自我改进方法

- **与 Self-Critique 工作的相似性**：都使用模型自我批评和自然语言反馈
- **与 Sparrow 的联系**：Sparrow 也将无害性分解为不同维度，类似宪法原则
- **独特贡献**：首次系统性地将自我批评与 RLAIF 结合，实现端到端无人类无害性标注

## 局限性

1. **依赖反馈模型能力**：方法仅在模型能够区分质量的领域有效（如无害性），在模型能力受限的领域效果有限
2. **过度优化风险**：RL-CAI 可能过度训练，导致 Goodharting 行为（过度说教、使用套话如 "you are valued and cared for"）
3. **批评质量**：即使是 52B 模型，批评也常包含不准确或夸大的批评
4. **仍需人类监督**：有用性标签仍依赖人类反馈，未完全实现自监督对齐
5. **鲁棒性未解决**：模型仍可能被红队攻击，需要进一步的迭代在线训练

## 关键洞察

1. **扩展监督的可行性**：AI 监督 AI 是可行的，且随模型能力提升而改善，为超人类 AI 的对齐提供了方向
2. **原则 > 标注**：少量明确的自然语言原则比大量隐式的人类标注更透明、可控、易迭代
3. **自我批评的价值**：模型能够识别并修正自身的有害输出，尤其在配合 CoT 推理时
4. **无害性与有用性的兼容**：通过训练非回避的无害助手，减少了两者之间的张力
5. **软标签的重要性**：使用归一化概率作为软标签比硬标签效果更好，CoT 需要钳制置信度
6. **宪法集成**：使用多条原则（16 条）并随机采样，提升了 PM 的鲁棒性和回复多样性
7. **两阶段设计**：SL 阶段使模型"上分布"，减少 RL 阶段的探索需求和训练时长
8. **透明度与可解释性**：CoT 推理使 AI 决策过程更透明，有助于发现隐藏风险
9. **双刃剑**：降低训练门槛也使恶意使用更容易，需要谨慎对待部署
10. **未来方向**：可扩展到控制 AI 的风格、语气、人格等多种行为维度，为研究 AI 行为泛化模式提供了低成本工具
