---
title: "Which-Language-Should-a-Skeleton-Speak-Language-Choices-in-M"
source: https://arxiv.org/pdf/2610.09607v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:45:18"
field: "多语言大语言模型推理"
keywords: ["多语言推理", "骨架提示", "语言选择", "低资源语言", "Chain-of-Thought", "数学推理"]
innovations: ["形式化骨架语言配置空间(查询/骨架/答案语言三元组)", "识别三种骨架语言效应模式(方向一致/评估依赖/不对称负向)", "建立MGSM-LowResource 34语言数据集及双模型翻译质量控制管道"]
benchmarks: ["MGSM", "MATH-500", "PolyMath", "MSVAMP", "MGSM-LowResource"]
---

# 论文速读：Which-Language-Should-a-Skeleton-Speak-Language-Choices-in-M

## 一句话总结
本文提出语言感知骨架探索框架（LASEF），系统研究多语言数学推理中骨架语言的选择问题，发现英语骨架平均带来小幅正向增益（对更小模型和低资源语言更明显），但英语并非 universally optimal；真正的大提升来自查询翻译而非骨架本身，且骨架语言效果呈现方向一致、评估依赖、不对称负向三种模式，无法仅用生成质量解释。

## 研究问题与动机
1. **英语中心假设的局限**：现有骨架推理工作（如 SOT、Plan-and-Solve）几乎都固定使用英语骨架，缺乏对 multilingual deployment 中语言组合的系统分析。
2. **三重语言选择的复杂性**：骨架推理隐式涉及三个语言维度——查询语言 $\ell_q$、骨架语言 $\ell_s$、推理/答案语言 $\ell_a$，不同组合会激活不同的语言先验和推理轨迹，但此前无人系统探索此设计空间。
3. **低资源语言推理不稳定性**：非英语语言（尤其低资源语言）的数学推理性能 gap 显著，骨架可能作为认知脚手架 stabilizes 推理过程，但是否有益取决于语言配对。
4. **翻译 vs 结构的贡献分离**：已有工作表明翻译成英语再推理有效，但骨架的语言选择是独立于翻译的结构性因素，需要 ablation 分离二者贡献。

## 核心贡献（创新点）
1. **形式化语言组合作为核心设计变量**：将 $(\ell_q, \ell_s, \ell_a)$ 三元组明确定义为多语言骨架推理的设计空间，与 Qi et al. (2025c) 固定英语骨架的做法本质不同。
2. **提出 LASEF 训练免费探索框架**：保持骨架结构和生成方法固定，仅系统变化语言配置进行实证分析，与 Ning et al. (2023) 的 Skeleton-of-Thought 相比，本文聚焦语言维度而非结构维度。
3. **识别三种骨架语言效应模式**：方向一致（如 eu-ko、lo-th 在多解码协议下符号保持）、评估/基准依赖（如 mt 在 greedy 下显著但在 multi-rollout 衰减）、不对称负向（如 mn-ru、ug-zh 共享表层特征却负迁移），这些无法用生成质量单独解释。
4. **建立 MGSM-LowResource 34语言数据集**：通过双模型翻译管道（Gemini+GPT）、数值保留检查和回溯翻译质量选择（BTQS）构建高质量低资源语料，填补了此前多语言推理基准缺乏低资源语言的空白。
5. **跨三模型家族验证尺度效应**：在 Qwen2.5、Llama 3.1、Ministral 三个家族均观察到"小模型受益、大模型增益缩小甚至变负"的定性趋势，Spearman $\rho = -0.568$ 验证基线准确率与骨架效应负相关。

## 方法详解
**语言配置空间**：骨架推理过程形式化为 $\hat{a} = f(q^{(\ell_q)}, s^{(\ell_s)}, \ell_a)$，其中 $f$ 为预训练 LM 推理函数。LASEF 固定骨架结构和生成 prompt，仅变化语言配置 $(\ell_q, \ell_s, \ell_a)$ 进行 benchmark 分析。

**骨架生成约束**：骨架平均约120 token，遵循原则"描述问题结构是什么，从不描述如何求解"。输出分两组件：(1) Problem Structure——识别对象、变量、关系、约束、问题性质；(2) Key Concepts/Tools——用标准学术术语描述所需数学/科学原理。禁止出现最终答案、具体数值或计算步骤，数字/公式替换为概念角色（如"系数"、"边界值"）。

**实验设置**：
- 模型：Qwen2.5-Instruct（7B/14B/72B）、Llama 3.1（8B/70B）、Ministral（3B/14B）
- 基准：MGSM（初等算术，5语言）、MATH-500（中级，需人工翻译）、PolyMath（高级，4难度级别）
- 语言：高资源（zh/es/ko/th/sw）+ 低资源（19种，含 kk/ky/mn/ug/hy/lo/eu/mt 等）
- 评估：Exact Match（EM），仅统计 $\ell_a = \ell_t$ 样本（FastText 语言识别过滤），报告贪婪解码和 multi-rollout（T=0.7, N=5）

**基线配置**：
- CoT-$\ell_t$：$(\ell_t, -, \ell_t)$，无骨架
- +SKELETON：$(\ell_t, en, \ell_t)$，英语骨架
- CoT-en$^\dagger$：$(en, -, \ell_t)$，Google Translate 译成英语查询
- +SKELETON：$(en, en, \ell_t)$，英语查询+英语骨架

**非英语骨架探索**：用 zh/es/ru/ko/th 五种非英语骨架替换英语骨架，比较 $\Delta = \text{Acc}(\text{non-En}) - \text{Acc}(\text{En})$。

**翻译消融**：用 GPT-5-mini 将英语骨架翻译成五种非英语，固定语义内容仅变化表层语言，分离生成质量与跨语言对齐的贡献。

**统计检验**：McNemar's test（配对二元结果）+ FDR 校正，跨基准泛化用 MSVAMP 验证。

## 实验与结果
**Table 1 主要结果（Qwen2.5，贪婪解码）**：

| 模型 | 基准 | CoT-$\ell_t$ AVG. | +SKELETON AVG. | CoT-en$^\dagger$ AVG. | +SKELETON AVG. |
|------|------|-------------------|----------------|------------------------|----------------|
| 7B | MGSM | 65.3 | 67.5 (+2.2) | 76.5 | 78.4 (+1.9) |
| 7B | MATH-500 | 45.1 | 46.9 (+1.8) | 56.0 | 57.4 (+1.4) |
| 7B | PolyMath | 23.9 | 25.1 (+1.2) | 27.9 | 28.9 (+1.0) |
| 72B | MGSM | 84.5 | 84.3 (-0.2) | 86.3 | 86.8 (+0.5) |

- **规模效应**：7B/14B 全基准正向（+0.9~2.5pp），72B 平均下降0.2pp，大模型外部英语骨架可能成为 unnecessary constraint。
- **难度效应**：MGSM（+2.2）> MATH-500（+1.8）> PolyMath（+1.2），简单任务受益更大。
- **资源 gap**：Swahili（sw）在 7B+MGSM 上从 63.1→71.7（+8.6pp），Chinese（zh）仅 +0.0pp。

**Table 2 低资源语言（19种，Qwen2.5-7B，MGSM）**：
- CoT-$\ell_t$：平均 33.99%，+SKELETON 平均 36.37%（+2.38pp），13/19 语言正向
- CoT-en$^\dagger$：平均 58.06%，+SKELETON 平均 60.08%（+2.02pp）
- FDR 校正后仅 Kazakh（kk，+11.34pp）和 Uyghur（ug，+10.52pp）显著
- _item-level_  pooling 后整体效应显著（$\Delta = +2.27$pp, $p = 0.0014$）

**关键结论**：翻译贡献 +24.07pp（33.99→58.06），骨架仅贡献 +2.02pp，翻译是 main driver。

**Figure 2 非英语 vs 英语骨架差异（Qwen2.5-7B，CoT-$\ell_t$）**：
- 平均性能下降 0.63~2.60pp，英语是 stable anchor
- **方向一致正例**：Lao-Thai（+2.81→+3.18）、Basque-Korean（+6.32→+2.78）、Armenian-Korean（+5.17→+4.61）
- **评估依赖例**：Maltese-Russian（greedy +8.62，multi-rollout -0.23），MSVAMP 上 mt-ru 甚至 reverses 到 -1.27
- **不对称负向例**：Mongolian-Russian（-5.00/-2.65）、Uyghur-Chinese（-6.45/-1.12）——共享表层特征（西里尔字母/地理邻近）但结构不对齐

**Table 9 Llama 3.1 验证**：8B 平均 +0.81pp，70B 平均 -0.43/-0.81pp，趋势与 Qwen2.5 一致。

**Table 10 Ministral 验证**：3B +1.29pp，14B +0.11pp，Spearman $\rho = -0.568$（$p = 0.0011$）确认基线准确率与骨架效应负相关。

**最强结果**：Kazakh（kk）在 CoT-$\ell_t$ + 英语骨架下从 40.08→51.42（+11.34pp，FDR 显著）；Uyghur（ug）从 27.13→37.65（+10.52pp）。

## 相关工作脉络
1. **Chain-of-Thought (Wei et al., 2022)**：零样本推理提示，本文以此为基础 baseline（CoT-$\ell_t$），但 CoT 不涉及显式骨架结构。
2. **Skeleton-of-Thought (Ning et al., 2023)**：并行解码骨架方法，本文沿用其推理结构但聚焦语言维度而非结构优化。
3. **English-Pivot Reasoning (Shi et al., 2022; Qin et al., 2023; Huang et al., 2023)**：将查询翻译为英语再推理，本文 CoT-en$^\dagger$ 直接对比此类方法，发现翻译是主要增益来源。
4. **Multilingual Structured Reasoning (Qi et al., 2025c)**：将 SOT 扩展到多语言但固定英语骨架，本文与之本质区别在于系统探索 $\ell_s$ 变化。
5. **Cross-lingual Instruction Tuning (Chai et al., 2025; Lai & Nissim, 2024)**：通过微调改进多语言推理一致性，本文强调 training-free 场景下的设计空间探索。
6. **Language Mixing in Reasoning (Wang et al., 2025a)**：研究推理过程中的语言混合模式，本文从骨架语言选择角度补充了结构设计维度。

## 局限性与未来方向
1. **领域局限**：仅评估数学推理，commonsense reasoning 或 code generation 的结论需额外验证。
2. **训练免费限制**：training-free 确保普适性但排除了 fine-tuning 可能带来的更大增益。
3. **骨架结构固定**：为隔离语言效应固定了骨架结构，骨架设计与语言选择的交互未 explored。
4. **计算成本**：全 LASEF 配置搜索计算昂贵，仅适合作为 offline diagnostic，非 per-deployment 步骤。
5. **跨基准泛化不完全**：MGSM 上的显著效应在 MSVAMP 上仅 66% 保持符号（23/35 cells），最优骨架语言无法先验指定。
6. **未来方向**：开发高效预测方法来搜索有效语言配置；扩展到更难基准验证泛化性；探索骨架结构与语言的联合优化。

## 研究启发与可借鉴点
1. **语言组合的形式化框架**：将 $(\ell_q, \ell_s, \ell_a)$ 三元组作为设计变量而非隐含假设，此框架可直接迁移到多语言摘要、对话生成、指令遵循等任务。
2. **双模型翻译管道 + BTQS 质量控制**：Gemini 和 GPT 并行翻译、数值保留检查（Eq.1）、回溯翻译质量选择（Eq.2-3）、round-trip reasoning validation（Table 5）的组合是一套可扩展的低资源语料构建 pipeline，值得复用到其他低资源语言任务。
3. **三种效应模式的分类方法论**：方向一致/评估依赖/不对称负向的分类结合 greedy decoding、multi-rollout、翻译 ablation、跨基准验证的四重验证 protocol，可作为多语言prompt engineering 的标准评估范式。
4. **尺度效应的普适性洞察**：三模型家族（Qwen/Llama/Ministral）均观察到"基线性能越高、骨架增益越小甚至为负"的定性趋势（Spearman $\rho = -0.568$），提示在大模型应用中需谨慎评估结构化提示的边际收益。
5. **结构对齐优于表层相似**：eu-ko（结构对齐但无表层相似）正向迁移，mn-ru 和 ug-zh（表层相似但结构不对齐）负向迁移，表明语言类型学特征（形态类型+基本语序）比文字系统/地理邻近更能预测骨架迁移效果，可为多语言 NLP 中的语言选择提供理论指导。

## 关键术语表
**Skeleton（骨架）**：浓缩问题求解所需核心推理结构的抽象表示，约120 token，不含具体计算步骤或答案，仅提供问题结构和关键概念的工具性指引。

**LASEF（Language-Aware Skeleton Exploration Framework）**：本文提出的训练免费探索框架，固定骨架结构和生成方法，系统变化语言配置 $(\ell_q, \ell_s, \ell_a)$ 进行 benchmark 分析以识别有效配置。

**CoT-$\ell_t$ / CoT-en$^\dagger$**：两种 baseline——前者查询和推理均在目标语言（native input）；后者查询经 Google Translate 译为英语后再用目标语言推理（English-pivot）。

**方向一致效应（Directionally Consistent Effects）**：在非英语骨架与英语骨架比较中，性能增益的符号在 greedy decoding 和 multi-rollout 评估下均保持一致的模式（如 eu-ko、lo-th）。

**不对称负向效应（Asymmetric Negative Effects）**：共享表层相似性（文字系统/地理邻近）但结构不对齐的语言 pair（如 mn-ru、ug-zh），非英语骨架导致性能显著下降的模式。

**结构对齐（Structural Alignment）**：目标语言与骨架语言共享相同主导形态类型（analytic/agglutinative/fusional）和基本语序（如 SOV/SVO）的语言学特征，本文发现其比表层相似更能预测正向迁移。

**MGSM-LowResource**：本文新构建的包含34种低资源语言的 MGSM 扩展数据集，共8,500个高质量样本，通过双模型翻译管道和 BTQS 质量控制构建。

**翻译消融（Translation Ablation）**：用 GPT-5-mini 将英语骨架翻译成非英语以固定语义内容、仅变化表层语言，用于分离骨架生成质量与跨语言对齐贡献的 ablation 实验。

## 可复现要素
- **数据集**：MGSM（公开）、MATH-500（公开，需人工翻译）、PolyMath（公开）、MSVAMP（公开）、MGSM-LowResource 34语言（**本文开源**）；所有资源 release 于论文脚注¹
- **代码/权重**：论文声明"All resources are released"，使用 Qwen2.5-Instruct 和 Llama 3.1-Instruct 开源模型（Alibaba Cloud / Meta）
- **关键超参**：骨架平均120 token；multi-rollout T=0.7, N=5；语言识别用 FastText classifier；数值保留检查阈值 Recall$_{num} < 0.8$ 触发重生成；BTQS 相似度阈值 0.8；McNemar's test + FDR 校正
- **翻译管线**：Dual-model（Gemini flash + GPT-5-mini）并行生成 → 数值保留检查 → BTQS（BAAI/bge-m3 编码器余弦相似度）→ Round-trip reasoning validation
