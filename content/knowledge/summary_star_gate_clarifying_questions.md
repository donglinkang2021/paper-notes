---
title: "STaR-GATE: Teaching Language Models to Ask Clarifying Questions"
authors: "Chinmaya Andukuri, Jan-Philipp Fränken, Tobias Gerstenberg, Noah D. Goodman"
institution: "Stanford University"
venue: "COLM 2024"
arxiv_id: "2403.19154"
tags: ["self-play", "preference-elicitation", "clarifying-questions", "bootstrapping", "iterative-training", "personalization"]
---

# STaR-GATE: 教语言模型提出澄清性问题

## 核心贡献

STaR-GATE 将 STaR（自举推理）与 GATE（主动偏好激发）结合，通过自我博弈的方式教会语言模型在任务模糊时主动提问以获取用户偏好信息。核心思想是：**用模型自己生成的有效问题来训练自己**。

关键创新：
1. **任务模糊性场景**：当用户请求（如"给我一个意大利面食谱"）存在多种解释时，模型需要通过提问澄清用户偏好（素食？喜欢什么酱料？）
2. **合成数据集**：生成 25,500 个独特的 persona-task-response 三元组用于训练
3. **基于对数概率的奖励**：用 Oracle 模型（知道完整 persona）生成的黄金回复的对数概率作为问题质量的评估标准
4. **响应正则化**：在训练时同时学习提问和回答，避免模型忘记如何生成回复

## 方法详解

### 问题设定

- **Questioner（提问者）**：需要训练的策略模型，不知道用户的 persona
- **Roleplayer（角色扮演者）**：模拟用户，拥有隐藏的 persona 信息
- **Oracle（预言者）**：知道完整 persona，生成黄金回复 $g_{ij}$

### 目标函数

最大化预训练模型 $Q_{BASE}$ 对黄金回复的对数概率：

$$J(Q, R, T, U) = \sum_{i,j} \mathbb{E}_{s_{ij}} \log p_{Q_{BASE}}(g_{ij} | t_i, s_{ij})$$

其中 $s_{ij} = [q_{ij1}, h_{ij1}, \ldots, q_{ijk}, h_{ijk}]$ 是模拟对话，$q$ 是问题，$h$ 是回答。

### 算法流程

**外循环迭代**（$\eta = 1, 2, \ldots, N$）：

1. **对话生成**：用当前模型 $Q_{\eta-1}$ 为每个 (task, persona) 对生成 10 个模拟对话（最多 3 轮）

2. **对话过滤**：选择使黄金回复对数概率最高的对话
   $$s_{ij}^* = \argmax_{s_{ij}^c} \log p_{Q_{BASE}}(g_{ij} | t_i, s_{ij}^c)$$

3. **响应采样**：用 $Q_{\eta-1}$ 基于最佳对话生成响应 $r_{ij}$（贪婪解码）

4. **微调**：从原始预训练模型 $Q_{BASE}$ 出发，在过滤的对话和采样的响应上训练
   - 只训练问题 $q$ 和响应 $r$，不训练 Roleplayer 的回答 $h$（mask 掉）
   - 响应正则化防止模型忘记如何回答

### 与 STaR 的区别

- **STaR**：学习生成推理链（rationale），目标是得到正确答案
- **STaR-GATE**：学习生成有效问题，目标是激发有用信息以生成个性化回复
- **共同点**：都使用专家迭代（Expert Iteration）+ 从预训练模型重启训练

## 实验结果

### 数据集构建

- **任务**：从 instruct-human-assistant-prompt 数据集选取 550 个日常任务（美食、职业、教育等）
- **Persona**：用 GPT-4 生成 110 个用户画像（如"小餐馆老板，喜欢艺术和音乐"）
- **黄金回复**：用 GPT-4 作为 Oracle，基于完整 persona 生成个性化回复
- **模型**：Questioner 使用 mistral-7b-instruct，Roleplayer 使用 mixtral-8x7b-instruct

### 主要结果

**对数概率提升**：
- 负对照（无信息）：基线
- 正对照（完整 persona）：上界
- Q-Experimental（STaR-GATE）：对数概率随迭代持续上升
- Q-Random（随机 persona）：略有提升但远低于 Q-Experimental

**胜率评估**（GPT-4 评判）：
- 迭代 0（初始模型）：50%
- 迭代 1：约 65%
- **迭代 2：72%**（峰值）
- 迭代 3：略有下降

### 消融实验

**1. 无响应正则化**：
- 只训练问题，不训练响应
- 结果：胜率下降，模型忘记如何回答，总是继续提问

**2. 使用黄金回复训练**：
- 训练时用 Oracle 的黄金回复替代自采样响应
- 结果：对数概率更高，但胜率收敛到 50%
- 原因：产生幻觉，生成的回复包含对话中未激发的信息

**3. Roleplayer 鲁棒性**：
- 用不同模型作为 Roleplayer 测试泛化能力
- mistral-7b-instruct：胜率 65%
- gemma-7b-instruct：胜率 62%
- 结论：可以泛化，但性能略有下降

**4. 自我 Oracle**：
- 用 llama3-8b-instruct 同时作为 Oracle、Questioner 和 Roleplayer
- 结果：1 轮迭代后达到 65% 胜率
- 与使用 GPT-4 作为 Oracle 的 66% 胜率相当
- 说明：足够强的模型可能不需要更强的 Oracle

## 与其他方法的对比

**与 GATE 的关系**：
- GATE：用 prompting 让模型提问，但问题质量不稳定
- STaR-GATE：通过迭代训练系统性提升提问能力

**与 STaR 的关系**：
- STaR：自举推理能力（生成 rationale）
- STaR-GATE：自举对话能力（生成有效问题）

**与其他偏好激发方法的对比**：
- Bayesian optimal experimental design：限制为成对比较，问题空间受限
- Offline RL 方法：需要专家模型生成高质量对话数据
- STaR-GATE：通过自我博弈自动生成训练数据

## 局限性

1. **依赖黄金标签**：当前方法需要 Oracle 生成黄金回复，虽然自我 Oracle 实验显示有希望消除这一依赖
2. **Roleplayer 泛化**：更换 Roleplayer 会导致性能下降，未来应在训练时使用多个 Roleplayer
3. **计算成本**：多轮迭代计算密集，需要探索减少迭代次数的方法
4. **数据集规模未消融**：未评估数据集大小对性能的影响
5. **领域泛化未测试**：未评估在其他领域的性能

## 关键洞察

1. **提问能力可自举**：通过奖励有效问题（提高黄金回复概率），模型可以自我改进提问策略
2. **响应正则化至关重要**：必须同时训练提问和回答，否则模型会退化为"只会提问的机器"
3. **自采样优于黄金回复**：用模型自己生成的响应训练比用 Oracle 的黄金回复更好，避免幻觉
4. **对话质量 > 对话数量**：过滤出最佳对话比生成大量对话更重要
5. **RLHF 可能抑制对话能力**：当前对齐策略可能无意中限制了模型的有效对话能力
6. **自我博弈的潜力**：足够强的模型（如 llama3-8b）可以作为自己的 Oracle，实现真正的自我改进

## 与后续工作的联系

STaR-GATE 展示了将 STaR 的自举思想扩展到对话场景的可能性，为以下方向提供启发：
- **个性化对话系统**：通过主动提问适应不同用户
- **交互式任务学习**：在任务模糊时通过对话澄清需求
- **自我改进对话能力**：无需大量人工标注的对话数据
