---
title: "A Simple Neural Network Module for Relational Reasoning"
authors: "Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap (DeepMind)"
institution: "Unknown"
venue: "arXiv 2017"
arxiv_id: "1706.01427"
tags: ["paper"]
---
# A Simple Neural Network Module for Relational Reasoning

**论文**: A simple neural network module for relational reasoning
**作者**: Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap (DeepMind)
**arXiv**: 1706.01427 (2017, NeurIPS 2017)

## 核心问题

**关系推理 (Relational Reasoning)** 是通用智能的核心能力，但对神经网络来说却很困难。例如：

- 找出公园里距离最远的两棵树（需要比较所有树对之间的距离）
- 在推理小说中根据线索推断凶手（需要在更广泛的上下文中考虑每条线索）

传统深度学习架构（CNN、MLP）在需要稀疏但复杂的关系推理任务上表现不佳。

## 关系网络 (Relation Networks)

### 核心公式

RN 是一个简洁的复合函数：

$$\text{RN}(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right)$$

其中：

- $O = \{o_1, o_2, \ldots, o_n\}$ 是"对象"集合，$o_i \in \mathbb{R}^m$
- $g_\theta$ 计算对象对 $(o_i, o_j)$ 之间的"关系"
- $f_\phi$ 聚合所有关系并产生最终输出
- $f_\phi$ 和 $g_\theta$ 都是 MLP，参数可端到端学习

### 三大优势

**1. 学习推断关系**

RN 考虑**所有**对象对之间的潜在关系。模型不需要预先知道哪些关系存在，而是学习推断关系的存在和意义。

**2. 数据高效**

RN 使用**单一函数** $g_\theta$ 计算所有关系。相比之下，MLP 需要在权重中隐式嵌入 $n^2$ 个相同的函数来处理所有可能的对象对。

**3. 集合置换不变性**

求和操作确保 RN 对输入对象的顺序不变，这符合集合的数学性质。

## 处理不同类型的输入

### 处理像素图像

使用 CNN 将图像转换为对象集合：

1. CNN 将 $128 \times 128$ 图像卷积为 $k$ 个 $d \times d$ 的特征图
2. 每个 $d^2$ 个 $k$ 维向量被视为一个"对象"
3. 为每个对象添加相对空间位置坐标

**关键**: 模型不预设什么是"对象"——可以是背景、物体、纹理或物体组合。

### 条件化问题嵌入

关系的存在和意义应该依赖于问题。修改后的公式：

$$a = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j, q)\right)$$

其中 $q$ 是 LSTM 处理问题后的最终状态。

### 处理状态描述

当输入是预分解的对象表示（如物体属性矩阵）时，可以直接输入 RN。

### 处理自然语言

对于文本问答任务（如 bAbI）：

1. 识别问题前最多 20 个句子
2. 用 LSTM 独立处理每个句子
3. 每个句子的 LSTM 最终状态作为一个"对象"

## 实验任务

### 1. CLEVR 数据集

视觉问答任务，包含 3D 渲染物体的图像和相关问题。

**问题类型**：

- `query attribute`: "球体是什么颜色？"
- `compare attribute`: "立方体和圆柱体材质相同吗？"
- `count`: "有多少个红色物体？"

**挑战**: CLEVR 明确要求关系推理，之前最好的模型只有 68.5%（人类 92.6%）。

### 2. Sort-of-CLEVR

专门设计用于分离关系和非关系问题：

**非关系问题**:

- "红色物体是什么形状？"
- "蓝色物体在图像的左边还是右边？"

**关系问题**:

- "离灰色物体最远的物体是什么形状？"
- "有多少物体与绿色物体形状相同？"

### 3. bAbI

纯文本问答数据集，包含 20 个推理子任务（演绎、归纳、计数等）。

### 4. 动态物理系统

使用 MuJoCo 物理引擎模拟的质量-弹簧系统：

- 10 个彩色球在桌面上运动
- 部分球对之间有不可见的弹簧或刚性连接

**任务 1**: 推断球之间是否存在连接
**任务 2**: 计算连通系统的数量

## 实验结果

### CLEVR 结果（从像素）

| 模型 | 总体 | Count | Exist | Compare Numbers | Query Attribute | Compare Attribute |
| --- | --- | --- | --- | --- | --- | --- |
| Human | 92.6 | 86.7 | 96.6 | 86.5 | 95.0 | 96.0 |
| CNN+LSTM+SA | 68.5 | 52.2 | 71.1 | 73.5 | 85.3 | 52.3 |
| **CNN+LSTM+RN** | **95.5** | **90.1** | **97.8** | **93.6** | **97.9** | **97.1** |

**关键发现**:

- RN 达到 95.5%，超越人类表现
- 比之前最佳提升 27%
- 在最困难的 `compare attribute` 类别上从 52.3% 提升到 97.1%

### CLEVR 结果（从状态描述）

准确率 96.4%，证明 RN 对输入形式具有鲁棒性。

### Sort-of-CLEVR 结果

| 模型 | 非关系问题 | 关系问题 |
| --- | --- | --- |
| CNN+MLP | ~94% | 63% |
| CNN+RN | ~94% | **94%** |

**关键发现**: CNN+MLP 完全无法解决关系问题，但增加 RN 模块后立即获得这种能力。

### bAbI 结果

- RN 通过 **18/20** 个任务（95% 阈值）
- 成功解决基本归纳任务（其他模型失败）
- 未通过的 2 个任务也没有灾难性失败

### 动态物理系统结果

| 任务 | RN | MLP |
| --- | --- | --- |
| 连接推断 | 93% | chance |
| 系统计数 | 95% | chance |

模型还能迁移到真实动作捕捉数据，预测行走人体的关节连接。

## 模型配置

CLEVR（从像素）任务的配置：

- **CNN**: 4 层，每层 24 个 $3 \times 3$ 卷积核，ReLU，批归一化
- **LSTM**: 128 单元（问题处理）
- **词嵌入**: 32 维
- **$g_\theta$**: 4 层 MLP，每层 256 单元，ReLU
- **$f_\phi$**: 3 层 MLP（256, 256+50% dropout, 29 单元），ReLU
- **优化器**: Adam，学习率 $2.5 \times 10^{-4}$

**重要发现**: 较小的模型表现更好。

## 失败案例分析

RN 主要在以下情况失败：

1. 物体严重遮挡
2. 需要高精度位置表示
3. 对人类来说也很困难的问题

## 核心贡献

1. **架构创新**: 提出简单的关系网络模块，专门用于关系推理
2. **即插即用**: RN 可以轻松集成到 CNN、LSTM 等架构中
3. **隐式对象发现**: 通过联合训练，RN 可以诱导上游处理产生有用的"类对象"表示
4. **跨领域泛化**: 在视觉、文本、物理推理任务上都取得出色结果

## 与相关工作的联系

### 与 Graph Neural Networks 的关系

- GNN、Gated GNN、Interaction Networks 也支持关系计算
- RN 更简单，对输入形式要求更低
- RN 可以从非结构化输入（CNN/LSTM 嵌入）中工作

### 与 Pointer Networks 的关系

- Pointer Networks 用注意力指向输入元素
- RN 用注意力式的机制考虑对象对之间的关系
- 两者都是处理可变大小输入的方法

### 与 Memory Networks 的关系

- Memory Networks 使用外部记忆进行推理
- RN 在 bAbI 上达到类似性能，但架构更简单

## 局限性与未来方向

**局限性**:

- 复杂度为 $O(n^2)$（考虑所有对象对）
- 遮挡和精确空间定位仍是挑战

**未来方向**:

- 使用注意力机制过滤不重要的关系，降低复杂度
- 应用于强化学习、社交网络建模、抽象问题求解

## 核心洞察

> **处理 (Processing) 与推理 (Reasoning) 是不同的能力**。
>
> 强大的视觉处理器（如 ResNet）不一定适合推理任意关系。RN 提供了一种专门的关系推理机制，使 CNN 可以专注于处理局部空间结构。

---

*本摘要基于 arXiv:1706.01427，这是 DeepMind 的重要工作，提出了关系网络这一简洁而强大的模块，在需要关系推理的任务上取得了突破性进展，展示了专门的归纳偏置对于特定推理能力的重要性。*
