# Neural Machine Translation by Jointly Learning to Align and Translate

**论文**: Neural Machine Translation by Jointly Learning to Align and Translate
**作者**: Dzmitry Bahdanau (Jacobs University), Kyunghyun Cho, Yoshua Bengio (Université de Montréal)
**arXiv**: 1409.0473 (2014, ICLR 2015)

## 核心问题

传统的编码器-解码器 (Encoder-Decoder) 架构将整个源句子压缩成一个**固定长度的向量**，然后从这个向量解码出翻译。

**问题**: 这种固定长度向量是一个瓶颈，尤其对于长句子。实验表明，基本 encoder-decoder 的性能随着输入句子长度增加而急剧下降。

## 核心创新：注意力机制 (Attention Mechanism)

**核心思想**: 不再将整个源句子压缩成单一向量，而是：

1. 将源句子编码为**向量序列**（annotations）
2. 在解码每个目标词时，**自适应地选择**这些向量的子集
3. 使用"软搜索"机制让模型学会在源句子中找到与当前预测相关的部分

## 模型架构

### 编码器：双向 RNN

使用双向 RNN 为每个源词生成 annotation：

$$h_j = \left[\overrightarrow{h}_j^\top; \overleftarrow{h}_j^\top\right]^\top$$

其中：

- $\overrightarrow{h}_j$ 是前向 RNN 在位置 $j$ 的隐藏状态
- $\overleftarrow{h}_j$ 是后向 RNN 在位置 $j$ 的隐藏状态

**优势**: annotation $h_j$ 包含了位置 $j$ 周围的双向上下文信息。

### 解码器：带注意力的 RNN

**条件概率**:

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)$$

其中 $s_i$ 是解码器隐藏状态，$c_i$ 是**上下文向量**。

**关键区别**: 与基本 encoder-decoder 不同，每个目标词 $y_i$ 使用**不同的**上下文向量 $c_i$。

### 注意力计算

**上下文向量**是 annotations 的加权和：

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

**注意力权重**通过 softmax 归一化：

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

**对齐分数**由对齐模型计算：

$$e_{ij} = a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

其中 $a(\cdot)$ 是一个前馈神经网络，与系统其他部分**联合训练**。

### 直观理解

- $\alpha_{ij}$ 可以理解为目标词 $y_i$ 与源词 $x_j$ 对齐的概率
- $c_i$ 是所有 annotations 的**期望**，期望是基于对齐概率计算的
- 这实现了一种**注意力机制**：解码器决定关注源句子的哪些部分

### 软对齐 vs 硬对齐

**软对齐的优势**:

1. **可微分**: 梯度可以通过对齐模型反向传播，实现端到端训练
2. **灵活处理词组长度不匹配**: 不需要将某些词映射到"空"(NULL)
3. **同时考虑多个源词**: 例如将 "the man" 翻译为 "l'homme" 时，模型可以同时看到 "the" 和 "man"

## GRU 详细公式

论文使用 Gated Recurrent Unit (GRU) 作为 RNN 单元：

**隐藏状态更新**:

$$s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$$

**候选隐藏状态**:

$$\tilde{s}_i = \tanh(W e(y_{i-1}) + U[r_i \circ s_{i-1}] + C c_i)$$

**更新门**:

$$z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i)$$

**重置门**:

$$r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)$$

## 实验设置

### 数据集

- **任务**: 英语到法语翻译
- **数据**: WMT '14 平行语料库，约 348M 词
- **词表**: 源语言和目标语言各 30,000 个最常见词

### 模型配置

| 参数 | 值 |
| --- | --- |
| 隐藏层大小 $n$ | 1000 |
| 词嵌入维度 $m$ | 620 |
| 对齐模型隐藏层 $n'$ | 1000 |
| Maxout 隐藏层 $l$ | 500 |

### 训练

- **优化器**: SGD + Adadelta ($\epsilon = 10^{-6}$, $\rho = 0.95$)
- **批大小**: 80 句子
- **梯度裁剪**: L2 范数阈值为 1
- **训练时间**: 约 5 天

## 实验结果

### BLEU 分数

| 模型 | 全部句子 | 无 UNK 句子 |
| --- | --- | --- |
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses | 33.30 | 35.63 |

**关键发现**:

1. RNNsearch 显著优于 RNNencdec（+8-9 BLEU）
2. 在无 UNK 词的句子上，RNNsearch-50* 达到了与 Moses（短语统计翻译系统）相当的性能
3. RNNsearch-30 甚至优于 RNNencdec-50

### 长句子性能

RNNencdec 的性能随句子长度增加**急剧下降**，而 RNNsearch 保持稳定：

- RNNsearch-50 在 50+ 词的句子上**没有性能下降**
- 这验证了论文的核心假设：固定长度向量是处理长句子的瓶颈

## 对齐可视化

论文展示了模型学到的软对齐（通过可视化 $\alpha_{ij}$）：

1. **大致单调**: 英法翻译中对角线附近权重较高
2. **非平凡对齐**: 正确处理形容词-名词顺序差异
   - 例如：[European Economic Area] → [zone économique européenne]
   - 模型正确地将 [zone] 与 [Area] 对齐，然后反向处理形容词

3. **软对齐优势**: 处理 [the man] → [l'homme] 时，模型同时关注 "the" 和 "man" 来决定使用 "l'" 而非 "le"

## 长句子翻译示例

**源句** (36词):
> An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital.

**RNNencdec-50** (错误翻译，约30词后开始偏离):
> Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical **d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé**.

**RNNsearch-50** (正确翻译):
> Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical **pour effectuer un diagnostic ou une procédure, selon son statut de travailleur des soins de santé à l'hôpital**.

## 核心贡献

1. **注意力机制**: 首次在 NMT 中引入（软）注意力，允许模型在解码时动态关注源句子不同部分
2. **解决长句子问题**: 消除了固定长度向量瓶颈
3. **可解释性**: 对齐权重提供了模型决策的可解释性
4. **端到端训练**: 对齐模型与翻译模型联合训练，无需外部对齐工具

## 历史意义

这篇论文是深度学习历史上最具影响力的工作之一：

1. **奠定 Transformer 基础**: Transformer 的自注意力机制直接继承自这里的注意力思想
2. **启发众多后续工作**:
   - Pointer Networks（将注意力用作指针）
   - Memory Networks（注意力访问外部记忆）
   - 各种视觉注意力模型

3. **"Attention is All You Need"**: 2017 年的 Transformer 论文标题本身就是对这一工作的致敬

## 局限性

1. **计算复杂度**: 需要对每个目标词计算所有源词的注意力，复杂度 $O(T_x \times T_y)$
2. **未知词问题**: 仍使用固定词表，罕见词处理困难
3. **单向解码**: 解码器仍是自回归的，无法并行化

---

*本摘要基于 arXiv:1409.0473，这是 Bahdanau、Cho 和 Bengio 的开创性工作，首次将注意力机制引入神经机器翻译，彻底改变了序列到序列学习的范式，为后来的 Transformer 架构奠定了基础。*
