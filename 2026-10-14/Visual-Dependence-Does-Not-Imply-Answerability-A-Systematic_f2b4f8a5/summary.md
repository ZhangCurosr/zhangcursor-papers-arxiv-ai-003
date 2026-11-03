---
title: "Visual-Dependence-Does-Not-Imply-Answerability-A-Systematic"
source: https://arxiv.org/pdf/2610.17902v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:37:34"
field: "多模态可靠性与可解释性"
keywords: ["Visual Question Answering", "Answerability", "Selective Prediction", "Vision-Language Model", "Frozen VLM", "Visual Dependence", "Intervention Analysis"]
innovations: ["系统审计冻结VLM中多种可答性信号（视觉依赖、不确定性、对齐等）的区分能力", "证明视觉依赖不等于证据充分性：目标掩码引起显著置信度下降但对可答性区分无效（AUROC 0.5004）", "通过多维度失败模式分析与统计鲁棒性验证，揭示信号在异质不可答情形下的不稳定性"]
benchmarks: ["UNK-VQA"]
---

# 论文速读：Visual-Dependence-Does-Not-Imply-Answerability-A-Systematic

## 一句话总结
本文在冻结的 InstructBLIP 上系统评估了多种可答性信号（视觉依赖性、预测不确定性、跨模态对齐、视觉定位、内部表示等），发现尽管模型对问题相关的视觉证据确实存在依赖，但这种依赖并不能可靠地区分可答与不可答样本（AUROC 仅 0.50–0.54）。

## 研究问题与动机
- VQA 模型即使在视觉证据不足时仍可能生成看似合理但无法被图像支持的回答，需要可靠的可答性估计以支持选择性回答机制。
- 置信度（confidence）与可答性（answerability）是不同的概念：模型可能基于有限证据高度自信地回答，但这并不意味着证据足够。
- 视觉依赖（visual dependence）是一个直观候选信号——如果问题可从图像回答，移除相关问题相关视觉证据应导致模型行为变化；但依赖与证据充分性未必等价。
- 现有工作多集中于构建基准或 abstention 机制，缺乏对冻结 VLM 内部信号是否足以作为可答性代理的系统性审计。

## 核心贡献（创新点）
- **系统性信号审计**：在统一的冻结 VLM 框架下，系统评估了预测不确定性、跨模态对齐、视觉定位、视觉依赖、内部表示及其组合等多个候选信号的可答性区分能力。
- **视觉依赖不等于证据充分**：通过针对性视觉证据移除实验证明，尽管目标掩码（0.1426）比随机掩码（0.0258）引起更大的置信度下降，但干预敏感度对可答性区分的 AUROC 仅为 0.5004（近乎随机）。
- **增强策略无效性验证**：证明增加探针容量（MLP）、特征融合、非线性探针、原因感知监督均无法带来稳健提升。
- **全面鲁棒性与失败模式分析**：通过配对 bootstrap、多种子实验、冻结后交叉验证及扰动/原因维度的细分评估，证明各信号在异质不可答情形下均不稳定。

## 方法详解
- **评估框架**：使用 UNK-VQA 的 10K 分层子集（7K/1K/2K 划分），平衡可答/不可答（5K/5K）及 5 种扰动类型（T-1 词替换、T-2 语义否定、I-1 图像替换、I-2 图像掩码、I-3 复制移动）。
- **基础模型**：InstructBLIP（含 Flan-T5-XL）完全冻结，不微调。
- **预测不确定性**：Generation confidence（贪心选择 token 概率的几何均值）、负预测熵；通过 L2 正则化 1-D LR 校准。
- **跨模态对齐**：使用 CLIP ViT-B/32 计算图像-问题余弦相似度。
- **视觉定位**：Grounding DINO + SAM 定位问题相关区域，提取 max 分数及 3-D 特征（实体覆盖率、掩码面积比）。
- **视觉依赖（干预）**：固定原图生成的答案 $A_{\mathrm{orig}}$，移除目标视觉证据后得到 $I_{\mathrm{rem}}$，计算干预敏感度：
  $$s_{\mathrm{int}} = c(A_{\mathrm{orig}} \mid I, q) - c(A_{\mathrm{orig}} \mid I_{\mathrm{rem}}, q)$$
  其中 $c$ 为条件置信度；同时使用等量随机掩码作为控制。
- **内部表示**：探针 Q-Former 和 question 表示，PCA 降维，以及特征融合（Q-Former + Question + Confidence + Grounding features）。
- **非线性探针**：Binary MLP（输入为多特征拼接）；原因感知 MTL 探针（联合预测可答性及不可答原因 R1–R4）。
- **评估协议**：AUROC 为主指标（0.5 为随机）；阈值依赖评估使用单一全局阈值；选择性回答使用风险-覆盖率曲线（AURC）；统计检验使用配对 bootstrap 95% CI。

## 实验与结果
- **总体 AUROC 结果**（Table 3）：所有信号表现均接近随机，范围为 0.5004–0.5430。
  - CLIP Alignment 最高（0.5430），Confidence（0.5388）、Entropy（0.5384）次之。
  - Grounding max（0.5061）、Intervention（0.5004）几乎无区分能力。
  - 多信号融合（7-feature LR）仅 0.5296，特征融合提升至 0.5300。
  - Binary MLP 训练 AUROC 0.7718 但测试仅 0.5147（过拟合）。
- **干预实验**（Fig. 2）：目标掩码平均置信度下降 0.1426 vs. 随机掩码 0.0258，说明模型确有关键区域依赖，但该信号对可答性区分无效（AUROC 0.5004）。
- **统计鲁棒性**（Sec. 4.4）：CLIP Alignment 相对 Confidence 的优势 (+0.0042) 的 95% CI 为 [−0.0298, 0.0377]，不显著；Intervention 显著低于 Confidence（−0.0384, 95% CI [−0.0698, −0.0061]）。
- **选择性回答**（Sec. 4.5）：在风险 ≤ 0.3 条件下，Confidence 仅保留 2.85% 样本，Entropy 仅 0.20%，CLIP Alignment 无可行操作点。
- **失败模式分析**（Sec. 5）：无方法在所有扰动类型或不可答原因（R1–R4）上 consistently 可靠；Intervention 在各原因上均接近随机（Table 5）。

## 相关工作脉络
- **Selective prediction / abstention in VQA**：Whitehead et al. [36]、Dancette et al. [8] 等侧重于训练选择性回答机制；本文聚焦于从冻结信号中提取可答性信息，不提出新 abstention 架构。
- **UNanswerable VQA & UNK-VQA**：Guo et al. [12] 构建可控不可答基准；Madhusudhan et al. [25] 区分视觉模态依赖与证据充分性构建基准；本文直接测试干预依赖信号是否实际具有可答性区分力。
- **CLIP-UP**：Vardi et al. [35] 结合 CLIP 对齐与轻量学习层检测不可答问题；本文独立评估 CLIP 对齐作为 standalone proxy 的鲁棒性。
- **EVICT**：Kotte [19] 使用 counterfactual evidence dropout 进行选择性 VQA；本文指出其依赖信号在 hard-negative 上表现较弱，且未系统评估其对可答性的泛化能力。
- **Visual grounding & hallucination**：Shrestha et al. [32]、Chen et al. [20] 等探讨视觉定位与幻觉缓解；本文指出定位质量改善不一定意味着正确使用正确视觉证据。
- **Probing frozen representations**：Alain & Bengio [1]、Hewitt & Liang [17] 奠基性工作；本文将其扩展至多模态可答性场景，发现内部表示亦不可靠。

## 局限性与未来方向
- **仅评估单一冻结 backbone（InstructBLIP）**，结论未必适用于具有不同视觉路径的架构（如 Qwen2.5-VL [29]、InternVL3 [38]）。
- **人工平衡子集**：破坏了原始 UNK-VQA 中不可答原因的天然不平衡分布，实际部署场景可能不同。
- **测试子集在开发过程中被 inspect**，虽通过冻结后交叉验证缓解，但完全未见基准将提供更强证据。
- **仅覆盖所测试的冻结信号与探针**，联合训练的 abstention 模型或更好的证据验证方法可能表现不同。
- **未来方向**：需显式建模证据充分性，而非依赖单一全局分数；当必要证据缺失时应主动 abstain 或请求澄清。

## 研究启发与可借鉴点
- **干预实验设计**：固定原答案、针对性移除证据、对比随机掩码的控制设计，可有效分离"视觉依赖"与"证据充分性"，方法可迁移至其他多模态可靠性研究。
- **分层平衡子集构建**：通过 question id 保持跨 split 的源问题不相交，同时平衡可答/不可答及扰动类型，是控制混杂变量的有效策略。
- **多维度失败模式分析**：按扰动类型（T-1/T-2/I-1/I-2/I-3）和不可答原因（R1–R4）分别报告 AUROC 和 U-Recall，揭示信号的不稳定性，值得在多模态评估中推广。
- **选择性回答的实际操作点分析**：不仅报告 AUROC，还报告风险-覆盖率曲线及低风险条件下的最大覆盖率，更贴近实际部署需求。
- **统计验证套件**：配对 bootstrap CI、多种子重复、冻结后交叉验证三重验证，可作为后续信号评估研究的标准化流程参考。

## 关键术语表
**Answerability（可答性）**：图像-问题对是否包含足够证据以支持有效回答的性质，区别于模型预测的置信度。
**Visual Dependence（视觉依赖）**：模型预测对特定视觉证据的敏感程度，通过干预（移除证据）后的行为变化衡量。
**Intervention Sensitivity（干预敏感度）**：固定原答案，比较原图与移除证据后图像的置信度下降幅度。
**Selective Answering（选择性回答）**：模型在低于风险阈值时选择不回答（abstain）的机制。
**AUROC（Area Under ROC Curve）**：衡量二分类排序质量的指标，0.5 表示随机排序。
**UNK-VQA**：包含可控扰动（词替换、语义否定、图像替换/掩码/移动）的可答/不可答 VQA 基准。
**R1–R4（不可答原因）**：R1 多合理答案、R2 难以理解、R3 图像缺少关键信息、R4 需要高层知识。
**Frozen VLM Evaluation**：不微调预训练视觉语言模型，仅从其输出和内部表示中提取信号的评估范式。

## 可复现要素
- **数据集**：UNK-VQA（https://github.com/yangyangguo0917/UNK-VQA），本文使用其 10K 分层子集；论文未声明公开代码，但提供了一个可复现的子集构建描述（Table 1 统计）。
- **代码**：论文未明确声明开源代码仓库；需联系作者获取。
- **模型**：InstructBLIP（冻结）+ Flan-T5-XL；CLIP ViT-B/32；Grounding DINO；SAM——均为开源模型。
- **关键超参**：L2 正则化 Logistic Regression；Binary MLP（未提及具体层数/hidden size）；5-fold cross-validation；bootstrap 95% CI。
