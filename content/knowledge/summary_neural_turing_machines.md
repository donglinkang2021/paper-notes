# Neural Turing Machines

**论文**: Neural Turing Machines
**作者**: Alex Graves, Greg Wayne, Ivo Danihelka (Google DeepMind)
**arXiv**: 1410.5401 (2014)

## 核心问题

传统计算机程序依赖三个基本机制：基本运算、逻辑流控制和外部存储器。然而，现代机器学习主要关注前者，很少利用逻辑控制和外部存储。虽然 RNN 理论上是图灵完备的，但在实践中很难学习需要长期记忆和算法操作的任务。

**核心思想**: 将神经网络与外部存储器耦合，通过注意力机制实现可微的读写操作，从而创建一种"可微分计算机"，能够从数据中学习算法。

## 架构概述

NTM 由两个核心组件组成：

1. **控制器 (Controller)**: 一个神经网络（前馈或 LSTM），负责接收输入、产生输出，并通过"读写头"与外部存储交互
2. **外部存储器 (Memory Bank)**: 一个 $N \times M$ 的矩阵，$N$ 是存储位置数，$M$ 是每个位置的向量维度

### 与传统计算机的类比

| NTM 组件 | 传统计算机 |
| --- | --- |
| 控制器 | CPU |
| 外部存储矩阵 | RAM |
| LSTM 控制器的隐藏状态 | 寄存器 |
| 读写头 | 内存总线 |

## 读写机制

### 读操作

给定在时间 $t$ 的权重向量 $\mathbf{w}_t$（满足 $\sum_i w_t(i) = 1$ 且 $0 \leq w_t(i) \leq 1$），读向量通过加权求和得到：

$$\mathbf{r}_t \leftarrow \sum_i w_t(i) \mathbf{M}_t(i)$$

### 写操作

写操作分为**擦除**和**添加**两步（灵感来自 LSTM 的输入门和遗忘门）：

**擦除步骤**：给定擦除向量 $\mathbf{e}_t \in (0,1)^M$：

$$\tilde{\mathbf{M}}_t(i) \leftarrow \mathbf{M}_{t-1}(i) \left[\mathbf{1} - w_t(i) \mathbf{e}_t\right]$$

**添加步骤**：给定添加向量 $\mathbf{a}_t$：

$$\mathbf{M}_t(i) \leftarrow \tilde{\mathbf{M}}_t(i) + w_t(i) \mathbf{a}_t$$

## 寻址机制（核心创新）

NTM 结合了两种互补的寻址方式：

### 1. 基于内容的寻址 (Content-Based Addressing)

通过相似度匹配找到相关内容：

$$w^c_t(i) \leftarrow \frac{\exp\left(\beta_t K[\mathbf{k}_t, \mathbf{M}_t(i)]\right)}{\sum_j \exp\left(\beta_t K[\mathbf{k}_t, \mathbf{M}_t(j)]\right)}$$

其中 $\mathbf{k}_t$ 是查询键向量，$\beta_t > 0$ 是键强度（控制聚焦程度），$K[\cdot, \cdot]$ 是余弦相似度：

$$K[\mathbf{u}, \mathbf{v}] = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

### 2. 基于位置的寻址 (Location-Based Addressing)

支持迭代访问和相对跳转：

**插值门** $g_t \in (0,1)$ 混合当前内容权重与上一时刻权重：

$$\mathbf{w}^g_t \leftarrow g_t \mathbf{w}^c_t + (1 - g_t) \mathbf{w}_{t-1}$$

**循环卷积**实现位置偏移（使用偏移权重 $\mathbf{s}_t$）：

$$\tilde{w}_t(i) \leftarrow \sum_{j=0}^{N-1} w^g_t(j) s_t(i - j)$$

**锐化**防止权重模糊（$\gamma_t \geq 1$）：

$$w_t(i) \leftarrow \frac{\tilde{w}_t(i)^{\gamma_t}}{\sum_j \tilde{w}_t(j)^{\gamma_t}}$$

### 三种寻址模式

1. **纯内容寻址**: 根据内容相似性直接定位
2. **内容+偏移**: 先按内容找到位置，再偏移到相邻位置
3. **纯位置迭代**: 忽略内容，仅基于上一时刻位置进行偏移（支持遍历）

## 实验任务与结果

### 1. Copy 任务

输入随机二进制向量序列，要求网络完整复制。

**关键发现**：

- NTM 学习速度远快于 LSTM
- NTM 能泛化到 **120 长度**的序列（训练只到 20）
- LSTM 在超出训练长度后迅速失效

**学到的算法**（伪代码）：
```
初始化: 移动头到起始位置
While 未见到分隔符:
    接收输入向量
    写入当前头位置
    头位置 +1
返回头到起始位置
While True:
    从头位置读取
    输出
    头位置 +1
```

### 2. Repeat Copy 任务

复制序列指定次数后输出结束标记。

**结果**: NTM 能泛化到更长序列，也能执行超过训练范围的重复次数（但无法正确预测结束时机）。

### 3. Associative Recall 任务

给定多个项目序列，查询某项目时返回其后继项目。

**结果**：

- NTM 比 LSTM 快得多达到近零损失
- 前馈控制器的 NTM 学习比 LSTM 控制器更快
- NTM 泛化到 2 倍训练长度（12 项）时近乎完美

**学到的算法**: 网络学会为每个项目创建压缩表示，存储在单独位置。查询时重新计算表示，用内容寻址找到位置，再偏移一位读取后继项目。

### 4. Dynamic N-Grams 任务

在线学习并预测动态生成的 N-gram 序列。

**结果**: NTM 接近贝叶斯最优估计器，使用存储器作为可重写的计数表。

### 5. Priority Sort 任务

按优先级对向量排序。

**发现**: NTM 学会使用优先级的线性函数确定写入位置，然后按顺序读取实现排序。

## 参数效率

| 任务 | NTM (FF) 参数 | NTM (LSTM) 参数 | LSTM 参数 |
| --- | --- | --- | --- |
| Copy | 17,162 | 67,561 | 1,352,969 |
| Repeat Copy | 16,712 | 66,111 | 5,312,007 |
| Associative | 146,845 | 70,330 | 1,344,518 |

NTM 用**远少的参数**实现了更好的性能和泛化能力。

## 核心贡献

1. **可微的外部存储**: 首次实现端到端可微的神经网络与外部存储器耦合
2. **混合寻址机制**: 结合内容寻址和位置寻址，支持多种访问模式
3. **算法学习**: 展示神经网络可以从示例中学习简单算法（复制、排序、关联检索）
4. **泛化能力**: 学到的算法能泛化到远超训练范围的输入

## 与其他工作的关系

### 与 Pointer Networks 的联系

Pointer Networks (1506.03134) 将注意力机制用作指向输入的指针，而 NTM 将注意力用于读写外部存储器。两者都使用 softmax 注意力，但目的不同：

- **Pointer Net**: 输出指向输入位置的指针
- **NTM**: 通过注意力访问可读写的外部存储

### 与 LSTM 的关系

NTM 可以看作 LSTM 的扩展：

- LSTM 的 cell state 是固定大小的内部存储
- NTM 提供大规模、可寻址的外部存储
- NTM 的存储参数不随存储大小增长

## 局限性与后续发展

**局限性**：

- 训练不稳定，需要仔细调参
- 存储容量固定
- 读写头数量限制计算能力

**后续工作**：

- **Differentiable Neural Computer (DNC)**: 改进的动态存储分配
- **Memory Networks**: 面向问答的外部存储
- **Transformer**: 自注意力机制的广泛应用

---

*本摘要基于 arXiv:1410.5401，这是 DeepMind 提出的开创性工作，首次展示了神经网络可以通过可微的外部存储学习简单算法，为后续的神经符号系统和大规模 Transformer 模型奠定了基础。*
