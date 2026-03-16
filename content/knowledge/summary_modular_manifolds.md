---
title: "Modular Manifolds"
authors: "Jeremy Bernstein"
institution: "Unknown"
venue: "Blog 2025"
arxiv_id: "N/A"
tags: ["blog"]
---
# Modular Manifolds

**博客信息**

- 来源: Thinking Machines Lab
- 作者: Jeremy Bernstein
- 发布时间: September 2025
- DOI: 10.64434/tml.20250926
- 链接: [thinkingmachines.ai/blog/modular-manifolds](https://thinkingmachines.ai/blog/modular-manifolds/)

---

## 一句话概括

这篇博客提出将神经网络的权重矩阵约束在特定的数学流形(manifold)上，使权重在训练过程中始终保持"健康"的数值范围，从而改善训练稳定性和优化效率。

---

## 为什么需要关心这个问题？

### 训练大模型的数值问题

训练大型神经网络时，张量(tensor)的数值范围很重要：

- **激活值(activations)**：通常通过 LayerNorm 等归一化技术控制
- **梯度(gradients)**：可以通过梯度裁剪/归一化控制
- **权重(weights)**：缺乏系统性的控制手段

如果权重矩阵的奇异值过大或过小，会导致梯度爆炸/消失、训练不稳定等问题。本文的思路是：**直接将权重约束在一个"形状良好"的流形上**。

---

## 第一部分：流形优化器的基本思想

### 什么是流形(Manifold)？

流形是一个弯曲的表面，在局部看起来是平的（就像地球表面在小范围内看起来是平的）。

**最简单的例子**：单位超球面 $S^{d-1}$

- 约束：向量 $w \in \mathbb{R}^d$，满足 $\|w\|_2 = 1$
- 直觉：参数不能随意走，必须待在这个球面上

### 切空间(Tangent Space)

在流形上某点 $w$ 的切空间是该点附近的"局部平面近似"。在超球面上：

- 切空间 = 所有与 $w$ 正交的方向
- 切空间条件：$a^T w = 0$

### 优化问题

在流形上做优化，每一步需要解决：

$$\min_a \quad a^T g \quad \text{(沿梯度方向最小化损失)}$$

$$\text{s.t.} \quad \|a\|_2 = \eta \quad \text{(步长约束)}$$

$$a^T w = 0 \quad \text{(保持在切空间上)}$$

其中 $g$ 是梯度，$w$ 是当前点，$\eta$ 是学习率。

### 解的形式

使用拉格朗日乘子法求解：

$$a_{\text{opt}} = -\eta \cdot \frac{g - w w^T g}{\|g - w w^T g\|_2}$$

直觉：先将梯度投影到切空间（减去沿 $w$ 方向的分量），然后归一化，最后乘以学习率。

### 回缩(Retraction)

更新后的点可能偏离流形，需要"拉回去"：

$$w \leftarrow \frac{w - \eta \cdot \frac{g - w w^T g}{\|g - w w^T g\|_2}}{\sqrt{1 + \eta^2}}$$

### 流形优化器的三步框架

1. **找到最优切向量**：在切空间中找到最大程度降低损失的单位方向
2. **缩放并更新**：乘以学习率，从当前权重中减去
3. **回缩到流形**：将更新后的权重拉回到流形上

---

## 第二部分：Manifold Muon

### 背景：SVD 与矩阵的"健康度"

权重矩阵 $W$ 通过 $y = Wx$ 对输入进行线性变换。通过奇异值分解(SVD)：

$$M = U \Sigma V^T$$

奇异值 $\Sigma$ 反映了矩阵对输入的"拉伸"程度：

- 奇异值过大 → 输出爆炸
- 奇异值过小 → 信息丢失
- 奇异值全为 1 → 等距变换，最"健康"

### Stiefel 流形

Stiefel 流形正是所有奇异值恰好等于 1 的矩阵集合：

$$\mathrm{Stiefel}(m, n) := \{ W \in \mathbb{R}^{m \times n} \mid W^T W = I_n \} \quad (m \geq n)$$

直觉：这些矩阵是"正交矩阵的推广"，它们不会拉伸或压缩输入向量。

切空间条件变为矩阵形式：

$$A^T W + W^T A = 0$$

### Manifold Muon 的优化问题

$$\min_A \quad \mathrm{tr}(G^T A) \quad \text{(线性近似损失变化)}$$

$$\text{s.t.} \quad \|A\|_{\text{spectral}} \leq \eta \quad \text{(谱范数约束步长)}$$

$$A^T W + W^T A = 0 \quad \text{(Stiefel 切空间约束)}$$

其中 $G$ 是损失对 $W$ 的梯度。

### 什么是 Muon？

Muon 是一种使用谱范数(spectral norm)来约束更新步长的优化器。普通 Muon 不要求权重在流形上；**Manifold Muon** 额外增加了 Stiefel 流形约束。

### 优化器家族对照表

| 空间 | 范数 | 优化器 |
| ---- | ---- | ------ |
| 欧几里得空间 | 欧几里得范数 | 普通梯度下降 (SGD) |
| 欧几里得空间 | 无穷范数 | Sign 梯度下降 (SignSGD) |
| 超球面 | 欧几里得范数 | 超球面下降 |
| 矩阵空间 | 谱范数 | Muon |
| **Stiefel 流形** | **谱范数** | **Manifold Muon** |

### 求解方法：对偶上升

Manifold Muon 的约束优化问题通过拉格朗日对偶转化为无约束问题：

1. 引入拉格朗日乘子 $\Lambda \in \mathbb{R}^{n \times n}$
2. 转化为对偶最大化问题：

$$\max_\Lambda \; -\eta \cdot \|G + 2W(\Lambda + \Lambda^T)\|_{\text{nuclear}}$$

3. 对偶函数的梯度涉及矩阵符号函数 $\mathrm{msign}$：

$$H(\Lambda) = -\eta \left[ W^T \mathrm{msign}(G + 2W(\Lambda + \Lambda^T)) + \mathrm{msign}(G + 2W(\Lambda + \Lambda^T))^T W \right]$$

其中 $\mathrm{msign}(M)$ 是将矩阵 $M$ 的所有奇异值置为 1 的操作（保留旋转结构，消除拉伸）。

### Manifold Muon 算法步骤

**输入:** 权重 $W \in \mathrm{Stiefel}(m,n)$，梯度 $G$，学习率 $\eta$

1. **对偶上升求解 $\Lambda_{\text{opt}}$**：

   迭代 $\Lambda \leftarrow \Lambda + \alpha \cdot H(\Lambda)$ 直到收敛

2. **计算更新方向**：

   $A_{\text{opt}} = -\eta \cdot \mathrm{msign}(G + 2W(\Lambda_{\text{opt}} + \Lambda_{\text{opt}}^T))$

3. **应用更新**：

   $W \leftarrow W + A_{\text{opt}}$

4. **回缩到 Stiefel 流形**：

   $W \leftarrow \mathrm{msign}(W)$

### 实验验证

在 CIFAR-10 上训练小型 MLP（3 个 epoch）：

- Manifold Muon 的训练和测试准确率均高于 AdamW
- 训练后权重矩阵的奇异值紧密聚集在 1 附近
- 当前实现有计算开销，但可优化

---

## 第三部分：Modular Manifolds —— 可组合的流形

### 核心抽象

将整个网络看作模块的组合，每个模块有三个属性：

1. **前向函数**：$f: \mathcal{W} \times \mathcal{X} \to \mathcal{Y}$
2. **权重流形**：$\mathcal{M} \subset \mathcal{W}$
3. **权重范数**：$\|\cdot\|: \mathcal{W} \to \mathbb{R}$

### 示例：StiefelLinear 模块

$$\text{StiefelLinear} = \left\{ \begin{array}{ll} (W, x) \mapsto Wx & \text{(前向函数)} \\ \mathrm{Stiefel}(m, n) & \text{(流形)} \\ \|\cdot\|_{\text{spectral}} & \text{(范数)} \end{array} \right.$$

### 组合规则

当两个模块 $(\mathcal{M}_1, f_1, \|\cdot\|_1)$ 和 $(\mathcal{M}_2, f_2, \|\cdot\|_2)$ 串联时，新模块为：

1. **前向函数**（组合）：

   $$f_3((w_1, w_2), x) := f_2(w_2, f_1(w_1, x))$$

2. **流形**（笛卡尔积）：

   $$\mathcal{M}_3 = \mathcal{M}_1 \times \mathcal{M}_2$$

3. **范数**（加权最大值）：

   $$\|(w_1, w_2)\|_3 := \max(s_1 \cdot \|w_1\|_1, \; s_2 \cdot \|w_2\|_2)$$

其中标量系数 $s_1, s_2$ 用于在层之间分配学习率预算。

### 关键洞察

这种构造将**学习率分配**问题与**网络输出对权重的 Lipschitz 敏感性**联系起来：

- 流形约束提供更紧的敏感性上界
- 每层的优化器可以独立运行
- 学习率自动按层缩放

---

## 第四部分：未来研究方向

| 方向 | 问题 |
| ---- | ---- |
| 模块性 | 哪些流形适合 attention head、embedding、unembedding？ |
| 数值精度 | 流形约束如何影响低精度训练？ |
| 凸优化 | 更高效的对偶上升求解方法？ |
| 收敛分析 | 收敛速率如何？权重矩阵的条件数是否改善收敛？ |
| 正则化 | 流形约束能否提升泛化能力？ |
| 架构-优化器协同设计 | 超越硬约束的协同设计机会？ |
| 非黎曼几何 | 神经网络使用算子范数，缺乏内积结构，这意味着什么？ |
| 高效实现 | GPU 高效的流形操作（如 Polar Express 算法） |

---

## 从直觉到理解

### 为什么要用流形？

想象你在训练一个深度网络，每一层做 $y = Wx$：

- 如果 $W$ 的奇异值是 $[0.1, 10, 0.5]$，有的方向被压缩，有的被拉伸
- 信号经过多层后，会指数级放大或缩小
- **LayerNorm 是事后补救；流形约束是从根源解决问题**

### 为什么选 Stiefel 流形？

$W^T W = I$ 意味着 $W$ 是"等距映射"（isometry）：
- 输入向量的长度不变
- 不同方向不会被差异性地拉伸
- 多层叠加后信号仍然稳定

### Muon vs Manifold Muon

- **Muon**：用谱范数约束更新步长，但不约束权重本身
- **Manifold Muon**：同时约束权重在 Stiefel 流形上 + 用谱范数约束步长

### 模块化的意义

不需要为整个网络设计一个全局优化器，而是：

1. 每个层/模块声明自己的流形和范数
2. 组合规则自动推导出整个网络的优化策略
3. 学习率按层自动缩放

---

## 参考文献

**流形优化:**
- Absil, Mahony & Sepulchre 教材（标准参考）
- Edelman et al., 1998（Stiefel 流形）

**Muon 相关:**
- Jianlin Su 的博客（Stiefel Muon）
- Franz Louis Cesista 的系列文章（启发式和广义解法）

**Modula 项目:**
- 项目主页: modula.systems
- Modular norm paper (2024)
- Modular duality paper (2024)

---

## 与本仓库其他笔记的关联

本文与之前总结的后训练方法（蒸馏、RL、SFT）处于不同层面：

- **蒸馏/RL/SFT** 关注"训练什么"（数据、目标、策略）
- **Modular Manifolds** 关注"如何训练"（优化器、权重约束、数值稳定性）

两者是互补的：一个好的优化器可以让蒸馏/RL 训练更高效、更稳定。

---

## 关键洞察

> 归一化技术（LayerNorm、梯度裁剪）是对症状的治疗；流形约束是对病因的治疗 —— 直接确保权重矩阵在数值上是"健康"的。

> 网络的模块化结构天然适合模块化的流形约束 —— 每个模块声明自己的几何结构，组合规则自动推导全局优化策略。
