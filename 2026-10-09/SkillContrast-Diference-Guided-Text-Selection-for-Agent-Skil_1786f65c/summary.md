---
title: "SkillContrast-Diference-Guided-Text-Selection-for-Agent-Skil"
source: https://arxiv.org/pdf/2610.11650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:26:47"
field: "Agent技能检索与重排序"
keywords: ["agent skill retrieval", "reranking", "text selection", "candidate difference", "zero-shot selector", "SameCapRisk-Bench"]
innovations: ["提出Difference-Guided免训练文本选择，通过候选间行级对齐提取差异块供reranker使用", "在SameCapRisk-Bench上以等长对照验证差异文本是洁净命中增益的主要来源（+54~+72 clean hits）", "相比完整正文节省51~59% tokens，4B模型下追平或超越full-body性能"]
benchmarks: ["SameCapRisk-Bench"]
---

# 论文速读：SkillContrast-Diference-Guided-Text-Selection-for-Agent-Skil

## 一句话总结
提出SkillContrast，一种免训练的候选文本选择方法，通过对比相似Agent技能并保留其差异性文本片段（配合局部上下文）供预训练reranker使用，在相同输入长度下相比TF-IDF查询选择显著提升了有效命中数，同时减少了51–59%的模型输入tokens。

## 研究问题与动机
1. **相似技能的区分难题**：Agent技能库中多个技能可能服务于相同目的，共享核心指令文本，但在前置条件、目标版本或输入格式上存在关键差异，导致基于查询的相关性排序难以区分适用性。
2. **现有文本选择方法的盲区**：KeyB、Provence等基于查询相关性的段落选择方法倾向于保留所有候选都高度相关的共享指令，却遗漏了决定技能适用边界的差异条件。
3. **长文本reranking的效率瓶颈**：直接传入完整技能正文虽能保留关键信息，但导致reranker输入过长、计算开销大；如何在压缩输入的同时保留区分性信息是一个开放问题。
4. **免训练选择的必要性**：R3-Skill等方法需要训练兼容性监督模型，SkillSight需校准评分偏差；本文希望在不微调reranker的前提下，仅通过改进输入文本构成来提升top-k召回质量。

## 核心贡献（创新点）
1. **提出Difference-Guided文本选择范式**：不同于传统查询-文档匹配思路，改为"候选间两两比较→定位差异块→拼接局部上下文"的选择逻辑，使reranker直接看到区分条件。
2. **零训练、即插即用**：完全基于TF-IDF向量空间进行候选连通图构建与行级对齐，无需任何标注数据或模型微调，可与任意冻结reranker（如Qwen3-Reranker）对接。
3. **系统的组件消融实验设计**：通过"替换差异块/首段/前导上下文为等长查询相关文本"的对照，定量分离出差异文本是主要增益来源（ΔClean=-78），而非上下文窗口本身。
4. **在SameCapRisk-Bench上验证洁净命中增益**：与同等长度的TF-IDF查询选择相比，在4组设置下稳定增加54–72个clean hits；相比完整正文，4B模型下甚至追平或超越。

## 方法详解
### 整体流程
给定请求 $q$ 和Top-20候选集合 $C=\{c_1,\ldots,c_{20}\}$，对每个候选 $c_i$ 的可见正文 $d_i$ 进行片段选择，得到reranker输入 $d_i'$，再以冻结reranker独立打分后，经MMR(λ=0.75)融合多样性选出Top-3。

### 差异块提取（有邻居分支）
1. **TF-IDF表示与连通图构建**：对所有技能正文拟合TF-IDF（小写unigram/bigram，最多160,000特征），计算余弦相似度，以阈值0.65构建连通分量。
2. **行级对齐**：在连通分量内逐行对齐两个技能正文，映射差异到段落/标题块，保持围栏代码块完整，跨技能跨行复制被禁止。
3. **块索引定义**：对候选 $c_i$，记 $\Gamma_i$ 为其连通分量内所有其他候选（含间接链接），$\Delta_{ij}$ 为与 $c_j$ 比对时 $d_i$ 中被选中的差异块集合。前导块定义为 $P(D)=\{b-1\mid b\in D,\;b>0\}$。
4. **保留块公式**：
   $$A_i = \{0\} \cup D_i \cup P(D_i), \quad D_i = \bigcup_{j\in\Gamma_i}\Delta_{ij}$$
   其中块0为 openings context；去重后按原文顺序保留。
5. **截断策略**：设定768-token体上限；若差异为空、超过上限或无邻居，则回退到全文。

### 无邻居分支（纯查询选择）
按 heading/line/sentence 单元做 query-unit TF-IDF 打分，保护围栏和行内代码；先保留第一单元作开头上下文，再依次累加正分单元及其前导单元，直至达到256-token上限。

### Token级保留
对含有效差异的候选，仅对完整输入 tokenize 一次，按选中块及其分隔符提取对应token，保持原始顺序和token ID，避免 decode+re-tokenize 引入的偏差。query/instruction/suffix 固定不变。

### 打分与MMR重排序
$$\arg\max_{i\notin S}\;\lambda r_i - (1-\lambda)\max_{j\in S}G_{ij},\quad \lambda=0.75$$
其中 $r_i=(N-\rho_i)/(N-1)$ 为归一化相关度，$G$ 为全文TF-IDF余弦相似度。

## 实验与结果
### 数据集
SameCapRisk-Bench [3]，823个主测试单元产生1,235条请求（411条operation-contract、774条source-role、50条caller-interface）。评估指标：R@3、Harmful Sibling Rate HSR@3、CleanHit CH@3。

### 基线
- 通用检索：BGE-M3、BGE-Reranker、Qwen-Listwise、RRF
- 公开技能检索方法：SkillRouter、SkillRet、R3-Skill、SkillSight
- 文本选择对照：Full body、Prefix（前缀截断）、Query-selected（TF-IDF查询选择）、Query-context variant、BM25-selected

### 主要结果（Table 1）
SkillContrast以SkillSight候选+Qwen3-Reranker-0.6B跑通，CH@3=**0.837**，超过R3-Skill（0.813）与SkillSight（0.767），HSR@3=0.113为最低。

### 主要结果（Table 2，等长对照）
| 设置 | SkillContrast CH@3 | 最佳对照 CH@3 | Clean Hit提升 |
|---|---|---|---|
| SkillSight/0.6B | 0.837 | Query-selected 0.794 | +54 |
| SkillSight/4B | 0.906 | Query-selected 0.860 | +57 |
| SkillRet/0.6B | 0.829 | Query-selected 0.773 | +69 |
| SkillRet/4B | 0.886 | Query-selected 0.828 | +72 |

相对完整正文：0.6B减少10–18个clean hits；4B反而多10个（SkillSight）或与SkillRet持平（0.886 vs 0.886）。

### Token压缩
SkillSight：786→323 tokens（**-58.8%**）；SkillRet：620→303 tokens（**-51.1%**）。在线 reranking 耗时减少10.4%，单请求节省约76.8 ms。

### 组件消融（Table 3，SkillSight/0.6B）
- 替换差异块：Clean Hit由1034降至956（Δ=-78），风险暴露由139升至203。
- 替换首段：CH变化为0。
- 替换前导上下文：CH+4。
- BM25-selected：CH=944（Δ=-90）。
结论：**差异文本是主要贡献者**，贡献6.32个百分点的CH@3优势（95%区间[4.47%, 8.16%]）。

## 相关工作脉络
1. **KeyB [5]**：基于Lexical/Learned score选取query相关段落做预排名。本文指出其对相似技能的选择不鲁棒——共享描述会被优先保留；SkillContrast改以候选间比较替代query单一视角。
2. **Provence [2]**：联合reranker做句子级剪枝，仍依赖query依赖打分。本文与之的关键差异在于：Provence是端到端可训练策略，SkillContrast完全免训练且显式构造"差异文本"信号。
3. **R3-Skill [11]**：引入query-conditioned compatibility supervision，需训练兼容模型。SkillContrast放弃训练，用文本层面的对比选择达成类似"暴露适用条件"的目标。
4. **SkillSight [12]**：解决generic-content bias导致的评分偏移。本文与其定位不同：SkillSight校正分数，SkillContrast改善输入内容构成，两者正交可叠加。
5. **MMR [1] / SMMR [6]**：列表去冗余重排。本文仅在MMR外层做最终候选挑选，核心贡献聚焦于MMR之前的文本选择阶段。
6. **SkillRet [4]**：大规模Agent技能检索benchmark。SameCapRisk-Bench是其补充，更关注"helpful与risky sibling共存"的风险场景，本文在此dataset验证。

## 局限性与未来方向
1. **未执行验证**：评估仅基于文本级命中，未经过真实agent执行链路测试，CleanHit定义排除了"marked risky sibling"但未覆盖所有不安全候选。
2. **TF-IDF表示有限**：连通图构建依赖TF-IDF余弦相似度，对于语义相近但词形差异大的技能可能漏建边；低资源语言或高度缩写场景下阈值0.65可能不稳定。
3. **对齐策略朴素**：当前采用行级对齐+段落映射，未利用LSTMs/LLM-level语义对齐，复杂嵌套结构的差异捕获能力受限。
4. **单reranker架构**：只在Qwen3-Reranker上验证，对跨架构迁移性缺乏证据。
5. **未来方向**：作者明确提出，需在具备applicability标签的独立技能库上验证泛化性；可探索将差异选择与reranker微调联合训练。

## 研究启发与可借鉴点
1. **差异优先于相关的选择直觉**：在候选间对比提取discriminative text的思路可迁移至文档去重、API选型、多版本软件包检索等场景。
2. **Token级精确保留避免重分词偏差**：一次tokenize后按块切片，保持原始token ID，该工程技巧对保证reranker行为稳定具有普适价值。
3. **替换消融的干净设计**：以"等长查询相关文本"替换差异块/上下文，直接分离内容贡献，这种对照范式可推广至其他text selection工作。
4. **免训练+固定reranker的评估协议**：固定候选池、reranker和MMR超参，仅改变选中文本，使对比结果纯粹反映选择策略本身，值得在复现其他方法时参考。
5. **紧凑输入与4B大模型的意外收益**：4B模型在SkillContrast下反超full-body，提示更大模型对稀疏但判别性信息的利用率更高，可能隐含"信息密度>绝对长度"的设计原则。

## 关键术语表
**Clean Hit (CH)**：在Top-3结果中召回到标记为helpful的技能，且同时未召回其对应的marked risky sibling，衡量"准确+避坑"双重能力。
**Harmful Sibling Rate (HSR)**：Top-3中包含marked risky sibling的期望比例，越低越好。
**SameCapRisk-Bench**：本文评测基准，构造成对"功能等价但条件不同"的技能，检验reranker能否区分适用性。
**Connected Component (TF-IDF)**：以0.65余弦阈值为边构建的技能相似图，组件内两两对比提取差异。
**MMR (Maximal Marginal Relevance)**：平衡相关度与多样性的重排目标函数，本文取λ=0.75。
**Different Block**：通过行级对齐定位的、在两个候选间文本内容不一致的段落/标题块。
**Query-context Variant**：保留差异块，但将前导上下文替换为query相关文本的对照策略。

## 可复现要素
- **数据集**：SameCapRisk-Bench（引用[3]），reproduction bundle已随代码发布，包含exact data identity。
- **代码**：已开源，https://anonymous.4open.science/r/SkillContrast-92C8/。
- **权重**：使用冻结Qwen3-Reranker-0.6B/4B，bf16精度，SDPA推理。
- **关键超参**：TF-IDF阈值0.65；差异分支768-token上限；无邻居分支256-token上限；MMR λ=0.75；Top-N=20，Top-K=3。
- **Bootstrap**：10,000次分层抽样，Bonferroni校正后置信水平98.75%。
