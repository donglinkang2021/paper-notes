---
title: "AI can Autonomously Evolve Pretraining Data Curation"
authors: "Tiantian Mi, Dongming Shan, Zhen Huang, Yiwei Qin, Muhang Xie, Yuxuan Qiao, Yixiu Liu, Chenyang Zhou, Pengfei Liu"
institution: "SII, FDU, SJTU, GAIR, KPS"
venue: "arXiv 2026"
arxiv_id: "2603.14420v1"
tags: ["data quality", "数据策展", "pretraining", "data cleaning", "data curation", "evolutionary optimization"]
---

# DataEvolve：AI 可以自主进化预训练数据策展策略

## 核心贡献

1. **把“数据策展策略设计”本身变成可优化对象**：论文不再假设人工先写好 prompt / cleaning rules，而是把每个类别的数据策展策略视作一个需要持续迭代优化的对象，让系统自己观察数据、提出策略、执行、评估、再改进。
2. **提出闭环式自动化框架 DataEvolve**：系统由 `data observer`、`strategy designer`、`data cleaner`、`quality judge` 四个角色构成，并通过 `experience pool` 与 `strategy pool` 跨代积累问题模式与高分策略，实现真正的 evolutionary optimization。
3. **把“难以承受的全量预训练验证”替换为 sample-based 质量代理**：作者明确指出，若每个候选策略都要清洗全量数据并训练模型收敛，代价是“每个候选策略数千 GPU 小时”，几乎不可行；DataEvolve 用样本级质量评估近似策略优劣，从而让策略搜索可迭代。
4. **在预训练规模上验证自动策略设计有效**：针对 Nemotron-CC 中 8 个 academic category（总计 672B tokens），DataEvolve 演化出按类别定制的清洗策略，最终构建出 504B-token 的 Darwin-CC，并在相同 3B / 500B-token 训练预算下超过 DCLM、Ultra-FineWeb、FineWeb-Edu。
5. **揭示“cleaning > transformation”的关键规律**：最优策略并没有收敛到大规模重写或风格转写，而是收敛到更保守的“有针对性的噪声删除 + 格式归一化 + 领域保留规则”，说明预训练数据质量提升未必依赖昂贵的内容重写。

## 方法详解

### 问题定义

作者把预训练数据策展问题形式化为：给定多个数据类别 $\mathcal{C}=\{c_1,\dots,c_m\}$，每个类别 $c_i$ 都有自己的原始语料 $\mathcal{D}_i$，需要为其找到最优策略 $s_i$。一个策略通过 LLM 执行器 $f(\cdot,\cdot)$ 作用于文档，得到清洗后的语料：

$$
\mathcal{D}'_i = \{f(d, s_i) \mid d \in \mathcal{D}_i\}
$$

理想情况下，最优策略应当最大化基于清洗后语料训练得到的语言模型性能：

$$
(s^*_1, \ldots, s^*_m) = \argmax_{s_1, \ldots, s_m} Q(M_{\mathcal{D}'})
$$

但这个目标极其昂贵，因为每评估一个候选策略都要经历“清洗数据 → 从头预训练模型 → 跑 benchmark”这一整条链路。于是论文用样本级质量评分函数 $S(\mathcal{D}'_i)$ 近似：

$$
s^*_i = \argmax_{s_i} S(\mathcal{D}'_i)
$$

这使得策略优化从“几乎不可搜索”变成“可在迭代闭环中不断逼近”。

### DataEvolve 闭环

DataEvolve 包含四个核心角色：

- **data observer**：读取目标类别的数据样本，识别该类文档中常见的质量问题。
- **strategy designer**：根据已有观察结果、历史反馈和高分父策略，生成或修正新的 cleaning strategy。
- **data cleaner**：按策略对采样文档执行清洗，产出原文—清洗后配对结果。
- **quality judge**：对清洗结果做 1–10 分打分，并给出诊断性反馈，指出哪些问题被解决、哪些指令不清晰、还发现了哪些新问题。

两个跨迭代知识容器是整个方法的关键：

- **experience pool**：保存已发现的数据质量问题；
- **strategy pool**：保存每轮出现过的策略、分数与诊断分析。

其核心机制不是“每轮重新想一个 prompt”，而是：

1. 先观察该类别数据，建立初始问题画像；
2. 生成初始策略；
3. 在新的样本上执行策略；
4. 评分并分析覆盖度、可执行性和残留问题；
5. 把新问题写回 experience pool，把当前策略与得分写回 strategy pool；
6. 下一轮以当前最优策略为 parent，在诊断反馈指导下继续变异和优化。

因此，这篇论文真正优化的不是单次 cleaning 结果，而是**“能够持续进化的数据策展策略”**。

### 数据与实验设置

- **数据源**：Nemotron-CC 的真实（non-synthetic）部分。
- **细分类方式**：结合质量标注与 `EAI-Distill-0.5B` 文档分类器，对 subject domain 和 document type 做再映射。
- **最终实验子集**：仅保留 academic content，并覆盖四个技术学科：mathematics、computer science、medicine、other stem；每个学科再按 high / not-high 两种质量层级划分，得到 **8 个类别**。
- **规模**：原始数据总量 **672B tokens**，清洗后得到 **504B tokens**。
- **迭代配置**：每个类别独立演化 **30 iterations**。
- **组件实现**：
  - data observer：`GPT-4o-mini`
  - strategy designer：`o4-mini`
  - data cleaner：`gpt-oss-120b`
  - quality judge：`gpt-5-mini`
- **每轮采样规模**：observer 看 100 个样本；cleaner 清洗 500 个文档；judge 评估 50 对原文/清洗文档。
- **最终训练设置**：用相同配置从头训练 **3B Qwen2.5 架构**模型，每个模型训练 **500B tokens**，确保不同数据集比较公平。

## 实验结果

### 1. 自动化策略设计确实能提升预训练数据质量

在 `Raw`、`Sub-opt`（低分策略）和 `Best`（DataEvolve 最优策略，即 Darwin-CC）三种设置下：

- **平均分**：40.17 → 41.20 → **44.13**
- 相比 Raw，Darwin-CC **平均提升 +3.96**
- 相比 Sub-opt，Darwin-CC 仍有 **+2.93** 的差距

这说明：

- 不是“只要让 LLM 清洗一下数据”就够了；
- 真正关键的是**策略本身是否经过持续演化优化**。

### 2. 增益主要集中在 knowledge-intensive tasks

Darwin-CC 在事实知识和专业知识密集型 benchmark 上收益非常大：

- **MMLU**：27.49 → **46.13**（+18.64）
- **CSQA**：20.31 → **39.12**（+18.80）
- **MedQA**：26.77 → **40.25**（+13.48）
- **MedMCQA**：28.86 → **40.97**（+12.10）
- **ARC-C**：43.52 → **49.32**（+5.80）

作者据此认为，清洗后的数据更好地保留了知识承载内容，同时去除了 HTML 噪声、格式损坏、重复片段等干扰物，使模型更容易在预训练中吸收事实与专业知识。

### 3. 对非正式语言理解任务未必有利

论文也明确呈现了 trade-off：

- **HellaSwag**：65.32 → 62.21（-3.11）
- **DROP**：19.57 → 18.49（-1.08）
- **SIQA**：44.36 → 43.57（-0.79）
- **PIQA**：76.79 → 76.15（-0.64）

作者推测，系统性清洗会让语料分布更正式、更规整，从而损失一部分口语化、情境化表达，这些表达对某些 everyday-language benchmark 反而有帮助。

### 4. 与现有预训练语料比较也有竞争力

在相同 3B / 500B-token 设置下，与其他公开数据集相比：

- **Darwin-CC**：**44.13**
- DCLM：42.42
- FineWeb-Edu：36.52
- Ultra-FineWeb：36.29

并且在多个知识型任务上显著领先：

- **MMLU**：46.13（对比 DCLM 的 28.54）
- **MedQA**：40.25（对比 DCLM 的 24.88）
- **MedMCQA**：40.97（对比 DCLM 的 28.15）

这表明按类别自动演化的 cleaning strategy，至少在 academic pretraining 子空间里，已经可以和强基线数据集正面竞争。

## 分析与讨论

### 清洗而不是重写

论文最重要的分析结论是：**DataEvolve learns to clean, not to transform.**

最优策略主要做三类事：

1. **删除 web artifacts**：HTML 标签、导航栏、广告、PII、重复句子等；
2. **归一化格式**：空白、标点、数字表达、乱码；
3. **应用领域保留规则**：
   - STEM 保留公式和科学术语；
   - 数学保留 theorem / proof；
   - 医学保留临床单位和药名；
   - 计算机科学保留代码块与技术语法。

这和很多“把原文改写成教科书风格 / Wikipedia 风格 / QA 风格”的数据生成路线不同。作者认为，预训练数据质量的提升不一定来自重写成统一风格，而可以来自**更保守但更准确的去噪与保真**。

### 有效策略的四个共同特征

高分策略通常都具备四个性质：

1. **concrete criteria**：用可操作、可阈值化的标准替代模糊描述；
2. **targeted deletion**：删具体问题片段，而不是整篇粗暴过滤；
3. **explicit preservation rules**：明确告诉系统什么必须保留，防止过清洗；
4. **conservative operations**：优先做低风险清理，而不是可能引入错误的激进改写。

### 清洗并没有显著损害语料多样性

作者给出的多样性指标显示：

- Self-ROUGE-2：**-21.7%**
- Shannon Entropy：**+0.3%**
- L2 Distance：几乎不变

这意味着 DataEvolve 的收益并不是来自把所有文本改成更同质的模板化风格，而是在保留语义多样性的前提下，主要消除了技术性噪声与污染。

## 局限性

1. **覆盖范围有限**：目前只处理了 Nemotron-CC 中 8 个 academic category，尚未验证对更广泛网页内容类型的泛化能力。
2. **固定 30 轮并不一定最优**：论文没有设计自适应 stopping criterion，因此部分类别可能尚未完全收敛。
3. **sample-based 代理目标仍有噪声**：虽然样本级质量评估让搜索变得可行，但它仍不是最终下游预训练表现本身。作者也指出，策略差异有时只会在训练了数百亿甚至数千亿 tokens 后才稳定显现，早期小规模 proxy 可能误判排序。
4. **与通用语料的横向比较不完全公平**：Darwin-CC 聚焦 academic 内容，而 DCLM / FineWeb-Edu 等覆盖更广主题，因此跨语料比较更应理解为“展示资源价值”，而不是严格意义上的全域 superiority claim。

## 关键洞察

1. **预训练数据策展的瓶颈，可能已经从“执行清洗”转移到“设计策略”**。当 LLM 足以执行复杂清洗时，真正昂贵的是 per-category strategy design 与验证。
2. **sample-based fitness approximation 是这篇论文的关键工程转折点**。它没有直接解决“如何最准确评估策略”，而是先把一个本来完全不可迭代的问题变成了可以持续优化的问题。
3. **category-specific cleaning 可能比全局统一 rewriting 更有扩展性**。不同领域对“什么是噪声、什么必须保留”的定义差异很大，因此按类别演化更符合预训练语料的异质结构。
4. **数据清洗会改变能力分布，而不是对所有 benchmark 等比例增益**。它更偏向提升知识记忆和专业理解，而可能牺牲一部分口语化、情景化任务表现。
5. **对数据工程的启发是：高质量预训练语料未必来自更强的“生成”，也可能来自更稳健的“保守清洗 + 精细保留规则”**。这为数据质量研究提供了和“合成数据 / 重写数据”不同的一条路线。
