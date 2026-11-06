---
title: "Zero-Shot-Multi-Class-Wafer-Map-Classification-with-Rule-Ord"
source: https://arxiv.org/pdf/2610.35080v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:48:46"
field: "晶圆图缺陷识别与零样本学习"
keywords: ["wafer map classification", "zero-shot learning", "large multimodal model", "prompt engineering", "defect pattern recognition", "semiconductor manufacturing"]
innovations: ["构建training-free六类晶圆图分类框架，结合冻结LMM与连通分量高亮", "设计形态-位置-排除边界三要素的结构化语义规则，明确类别决策边界", "定义规则呈现顺序以编码类别优先级，缓解零样本下证据重叠歧义"]
benchmarks: ["Mixed-WM38", "WM811K"]
---

# 论文速读：Zero-Shot-Multi-Class-Wafer-Map-Classification-with-Rule-Ordered Prompts

## 一句话总结
本文提出一种无需训练（training-free）的六类晶圆图（wafer map）缺陷分类框架，将冻结的大多数模态模型（LMM）与连通分量高亮（connected-component highlighting）及形态/位置/排除边界感知（morphology-, location-, exclusion-aware）的规则化提示词相结合，实现零样本细粒度缺陷识别。

## 研究问题与动机
- **监督方法依赖标注与重训**：实际制造环境中，新产品线或新工艺节点往往缺乏高质量标注数据，且产线/工艺节点/晶圆尺寸变化会导致缺陷分布漂移，现有监督方法需重新训练或校准，成本高昂。
- **人工标签存在噪声**：专家标注可能不一致，影响模型可靠性。
- **已有零样本研究局限于二分类**：先前工作仅支持正常/异常二分类，难以直接扩展到细粒度六类缺陷识别（Center, Donut, Edge, Local, Random, Scratch）。
- **类别间空间证据高度重叠**：如Center与Donut均位于晶圆中心附近，Edge与Local均靠近边界，仅靠类别名称无法提供足够的决策边界区分。

## 核心贡献（创新点）
- **构建training-free六类分类框架**：与需微调或重训的监督方法不同，本文在冻结LMM上通过图像-文本联合输入直接输出类别，无需目标域训练样本。
- **设计边界感知语义规则（boundary-aware semantic rules）**：将每类缺陷分解为正形形态描述（positive morphology）、空间锚点（spatial anchor）与排除边界（exclusion boundary），明确告诉模型“观察什么、在哪里观察、为何排除相似类”，区别于仅列出类别名的粗粒度提示。
- **定义固定规则呈现顺序以编码优先级**：针对证据部分重叠的图样，通过排序（Donut > Scratch > Edge > Random > Center > Local）为LLM提供指令级决策偏向，弥补零样本设置下无参数更新可依赖的不足。
- **系统分析视觉呈现范围的影响**：对比单区域（K=1）与多区域（K=5）连通分量高亮，揭示不同缺陷模式对视觉证据范围的敏感性差异。

## 方法详解
- **问题形式化**：给定原始晶圆图 $\mathbf{X}$ 和六类集合 $\mathcal{Y}$，冻结LMM $F_\theta$ 预测 $\hat{y} = F_\theta(Z, P(Z; \mathcal{R}, \pi))$，其中 $Z = V(\mathbf{X})$ 为视觉提示表示，$\mathcal{R}$ 为语义规则集，$\pi$ 为规则呈现顺序。
- **视觉提示接口**：使用8-连通性分割缺陷掩码，选取尺寸最大的 $K$ 个连通分量并赋予不同颜色（红色为最大分量，后续分量用其他颜色），未选中失效点保持白色可见。$K=1$ 时仅高亮最大分量，$K=5$ 时高亮前五个最大分量。
- **文本提示构建**：$P = L(Z) \oplus [\bigoplus_{j=1}^{|\mathcal{Y}|} T(r_{\pi_j})] \oplus C$，其中 $L(Z)$ 为视觉图例，$T(\cdot)$ 将结构化规则翻译为自然语言，$C$ 为输出约束指令。
- **规则三要素**：每个类别规则 $r_c = (M_c, A_c, E_c)$：
  - **M_c（正形形态）**：描述主导缺陷结构的几何特征（如实心团块、空心环、长条形、边缘弧段等）。
  - **A_c（空间锚点）**：指定该结构相对于晶圆关键区域（中心、内部、圆形边界）的位置。
  - **E_c（排除边界）**：明确排除与其视觉相似的邻近类别的证据（如"排除向内凸出的紧凑团块"以区分Edge与Local）。
- **规则顺序设计 rationale**：Donut排在Center前因为环形结构比单纯中心位置更具判别性；Scratch排在Edge前因为非圆周方向的拉长缺陷应先于边界接触判断；Edge排在Random前因为沿边界弧段先于广域覆盖判断；Random排在Center/Local前因为晶圆级覆盖应优先于局部聚集判断；Center排在Local前因为中心锚定比一般非中心紧凑区域更特异。
- **推理流程**：对每张晶圆图执行单次前向推理，模型返回唯一标签。

## 实验与结果
- **数据集**：
  - **Mixed-WM38**：6,866张单模式缺陷图（Center 1000, Donut 1000, Edge 2000, Local 1000, Random 866, Scratch 1000）。
  - **WM811K**：从90,000张随机采样，含 1,747 Center, 502 Donut, 3,430 Edge, 1,830 Local, 555 Random, 936 Scratch。
- **基线对比条件**：Raw（原始图像）、Single（单区域高亮 K=1）、Multi（多区域高亮 K=5）、Names-only（仅类别名无规则描述）。
- **模型与设置**：冻结 GPT-4o-2024-08-06，temperature=0.1，单次推理。
- **Mixed-WM38 结果**：
  - Multi（默认顺序）达到 **92.9% macro / 93.8% overall** 最优。
  - Single 达 92.4%/93.3%，优于 Raw（44.6%/37.5%）和 Names-only（58.2%/63.6%）。
  - Single在Center (99.6%) 和 Local (98.0%) 上表现更好，Multi在Donut (98.7%)、Edge (96.8%) 和 Random (78.5%) 上更好。
- **规则顺序消融**：默认顺序 D>S>E>R>C>L 达到最高93.8%；八种替代顺序整体准确率为 90.1%–91.9%，证明顺序敏感性显著。
- **WM811K 结果**：
  - Multi 达 **79.1% macro / 81.3% overall**，Single 达 76.7%/80.1%，Raw 仅 58.7%/43.7%。
  - 较Mixed-WM38下降约12-14个百分点，主要因WM811K含不同尺寸晶圆及更多模糊/碎片化缺陷模式。
- **代表性错误分析**：Donut样本可能仅为不完整弧段而非完整环状；Edge与Scratch证据重叠（边界接触+拉长结构）；Local样本呈长条状而非紧凑团块。

## 相关工作脉络
- **CLIP与零样本分类**：Radford等提出通过自然语言监督实现跨模态零样本分类，本文继承其零样本思想但面向半导体制造特定领域。
- **视觉提示（Visual Prompting）**：Yang等提出用可见标记引导冻结模型关注特定图像区域，本文采用连通分量高亮作为视觉提示的具体实现。
- **二分类先行工作**：作者先前工作（Cao et al., 2026）已验证连通分量高亮+冻结LMM可实现正常/异常二分类，本文将其扩展至六类细粒度分类。
- **监督/半监督晶圆图分类**：传统CNN方法（Nakazawa & Kulkarni, 2018; Kyeong & Kim, 2018）及少数/自监督学习（Geng et al., 2021）需大量标注或预训练，本文避免此类依赖。
- **LLM提示顺序敏感性**：Pezemshkpour & Hruschka（2024）及Guan等（2025）研究发现LLM对输入顺序敏感，本文将其显式用于规则优先级编码。
- **状态空间模型**：Nie等（2025）探索高效状态空间模型处理混合缺陷，属监督学习范式，与本文training-free定位不同。

## 局限性与未来方向
- **跨数据集性能下降**：WM811K上准确率较Mixed-WM38下降明显，主因是真实数据含尺寸不一、形态模糊的晶圆图。
- **规则顺序次优性**：当前顺序基于领域直觉设计，未进行全局优化搜索。
- **仅评估单模式缺陷**：未测试Mixed-WM38中的混合类型缺陷（mixed-type patterns）。
- **推理成本较高**：API单次推理约0.5–1秒/图，速度慢于本地部署的训练分类器。
- **未验证其他LMM**：仅在GPT-4o上实验，泛化到其他多模态模型未知。

## 研究启发与可借鉴点
- **规则化提示词工程**：将领域知识（形态-位置-排除边界三元组）结构化为自然语言规则，可迁移至其他需要细粒度视觉分类但标注稀缺的工业场景。
- **顺序即先验**：利用LLM对输入顺序的敏感性，通过规则呈现顺序显式编码类别优先级，为zero-shot决策提供轻量级"软约束"机制。
- **视觉呈现范围可调节**：通过调节高亮连通分量数量（K值）控制视觉证据scope，可根据不同缺陷模式的证据分布特性动态选择。
- **可检查/可修订决策过程**：规则与视觉标记完全可解释，便于工艺工程师审计与修正，适合对可靠性要求高的制造环境。
- **与团队方向结合点**：若团队关注低资源或缺乏标注场景下的缺陷检测，此training-free范式可作为baseline或冷启动方案；亦可探索将规则顺序学习自动化。

## 关键术语表
- **Wafer map classification**：基于二维网格记录晶圆的合格/不合格die空间分布，识别缺陷模式以支持良率分析。
- **Zero-shot learning**：无需目标域训练样本即可进行分类的范式，本文依靠冻结LMM的预训练知识。
- **Large multimodal model (LMM)**：如GPT-4o，可同时处理图像与文本输入并生成文本输出的大模型。
- **Connected-component highlighting**：将缺陷掩码按8-连通性分割后，对最大K个连通区域用不同颜色高亮，作为视觉提示。
- **Class-semantic rules**：每类缺陷的结构化描述，包含正形形态、空间锚点和排除边界三要素。
- **Rule-presentation order**：规则在文本提示中的排列顺序，编码类别间的决策优先级。
- **Macro accuracy**：各类别准确率的未加权平均值，反映类别平衡性能。
- **Overall accuracy**：正确分类样本数占总样本数的比例。

## 可复现要素
- **数据集**：Mixed-WM38（公开），WM811K（公开）
- **代码**：论文未提及开源代码
- **权重**：使用冻结的 GPT-4o-2024-08-06，via OpenAI API
- **关键超参**：highlighted components数量 K=1（Single）或 K=5（Multi）；temperature=0.1
- **规则集与顺序**：Table 1中的六类规则，默认顺序 Donut>Scratch>Edge>Random>Center>Local
