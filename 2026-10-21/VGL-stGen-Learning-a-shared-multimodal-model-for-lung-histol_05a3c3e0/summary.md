---
title: "VGL-stGen-Learning-a-shared-multimodal-model-for-lung-histol"
source: https://arxiv.org/pdf/2610.28266v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:57:12"
field: "计算病理学与空间转录组学"
keywords: ["spatial transcriptomics", "histology", "multimodal LLM", "gene expression prediction", "cell typing", "lung cancer"]
innovations: ["基因条件伪标签蒸馏实现跨模态监督迁移", "固定查询名自回归表达式补全提升结构化预测准确性", "单一checkpoint支持13种组织学-转录组联合任务"]
benchmarks: ["TCGA-LUAD binary tumor", "Visium/Xenium cell typing", "masked expression completion"]
---

# 论文速读：VGL-stGen: Learning a shared multimodal model for lung histology and spatial transcriptomics

## 一句话总结
VGL-stGen是将预训练医学视觉语言模型MedGemma-1.5-4b-it适配为支持13种肺组织学与空间转录组联合任务的多模态生成模型，通过基因条件伪标签和条件表达补全实现跨模态知识迁移。

## 研究问题与动机
- 现有影像组学方法多针对特定模态或预测任务设计，缺乏统一框架连接组织形态、基因表达与细胞身份
- 组织病理图像与空间转录组数据分属不同建模范式，难以在共享表征空间中进行跨模态推理
- 已有方法如ST-Net、BLEEP、GenAR、SpaCKLE等各自专注单一任务组合，无法灵活应对不同观测条件下的预测需求
- 医学视觉语言模型在肺组织学与空间转录组联合学习任务上尚未充分探索

## 核心贡献（创新点）
1. **提出首个支持13种任务的多模态生成框架**：同一checkpoint支持H&E图像分类、基因列表生成、表达水平补全等多样化任务，区别于现有单一模态或单一任务模型
2. **基因条件伪标签蒸馏机制**：利用转录组教师模型预测细胞类型，通过log-likelihood gap阈值筛选1.32M伪标签样本监督配对图像，使图像→细胞类型准确率从0.161提升至0.226
3. **固定查询名自回归表达式补全**：从75个观测表达水平预测25个命名基因的表达等级，Pair-F1达0.6596（图像条件）和0.6858（细胞类型条件）
4. **Level-only监督策略优化**：仅对请求的表达等级token施加损失而非完整答案，图像条件Pair-F1提升+0.0085
5. **多阶段训练与权重平均**：整合基因适应、细胞类型适应、表达分箱、组织学蒸馏、癌症训练五个阶段，对6个晚期adapter进行concatenation averaging提升泛化

## 方法详解
- **基础架构**：基于MedGemma-1.5-4b-it，冻结视觉tower，语言侧使用QLoRA适配（rank=16, alpha=32, 4-bit NF4量化, bfloat16计算）
- **训练数据**：PortraiATLAS中69个ST-H&E配对样本（47 Xenium + 22 Visium），30,965个癌症patch，1,745,240个细胞/spot转录组数据
- **表达分箱**：将表达量分为Absent/Low/Med/High/Very High五类，覆盖2,000基因（图像任务）和1,478基因（细胞类型任务）
- **基因条件伪标签**：教师模型根据测量基因预测细胞类型，取log-likelihood gap≥1.0的1.32M样本（占1.75M候选的75%）作为学生模型的图像监督信号
- **多任务解码**：
  - 分类任务：prefix-tree解码限制输出类别
  - 基因列表任务：生成后解析基因符号
  - 表达补全任务：固定25个查询基因名，自回归预测各等级token
- **训练策略**：Focal loss（γ=2.0）+ label smoothing（0.1）+ inverse-frequency加权；采样比例按softmax验证损失动态调整；最终对6个late adapter在rank-96处concatenation averaging

## 实验与结果
- **数据集**：Visium（9,964 spots）、Xenium（214,246 cells）、TCGA-LUAD（452 slides/357,692 patches）
- **基线对比**：10个现有多模态模型（MedGemma-1.5-4B、Gemma3-4B、Gemma4-12B/31B、Qwen3-VL-8B/32B、Qwen3.8-27B、DeepSeek-VL2-small/2、GLM-4.6V-Flash）
- **主要结果**：
  - 基因→细胞类型：BA=0.6293，F1m=0.06023（Gemma4-31B: 0.5035/0.4870）
  - 图像→二元肿瘤：BA=0.7524（Gemma4-12B: 0.6986）
  - 图像+观测水平→查询水平：Pair-F1=0.6596（图像）/0.6858（细胞类型），超越Qwen3.8-27B的0.6130/0.5621
  - 表达等级→细胞类型：Acc=0.7420（多数基线0.3620）
  - 九类组织分类（校正后）：F1m=0.2801
- **应用验证**：GSE127465数据BA=0.815/F1m=0.819；TCGA-LUAD患者级bulk RNA相关分析显示152个基因FDR<0.05显著，淋巴细胞程序r=0.17-0.41

## 相关工作脉络
- **ST-Net**（2020）：从H&E预测表达，单向模态映射，无跨任务灵活性
- **BLEEP**（2023）：对比学习检索参考表达，不直接生成结构化输出
- **GenAR**（2025）：自回归生成表达计数，专注单一生成任务
- **OmiCLIP**（2025）：组织学与排序基因文本对齐，非自回归生成
- **SPADE**（2026）： mixture-of-experts学习对齐表示，无多任务联合训练
- **STPath**（2025）：生成式基础模型支持spot imputation，但未扩展到13种任务统一框架
- **GHIST**（2025）：从H&E预测单细胞表达，使用辅助细胞类型与邻域任务，非自回归生成架构

## 局限性与未来方向
- 评估参考注释为计算派生（ARA-CNN、TACCO转移标签），存在共享标注谱系误差
- 部分H&E数据在癌症训练中暴露，外部分类任务未完全独立于训练分布
- TCGA开发子集（39 slides）包含在452 slide评估中，患者级泛化性受限
-  masked expression benchmark中69个测试样本全部出现在gene-list训练，未建立跨任务患者独立泛化
- 多模体比较区间以item为单位重采样，未充分考虑slide/patient聚类效应
- 固定token budget可能惩罚需要中间推理的模型
- 需要独立注释、验证的患者分离和前瞻性终点才能向临床应用推进

## 研究启发与可借鉴点
1. **基因条件伪标签蒸馏**：利用一个模态的教师模型为另一模态生成软标签，实现跨模态监督迁移，可应用于其他多组学联合学习
2. **固定查询名自回归解码**：在表达式补全任务中分离查询名与预测值，避免模型生成错误基因名，提升结构化预测准确性
3. **Level-only监督策略**：仅对目标token施加损失而非完整答案，可减少无关token干扰，适合部分可观测序列预测任务
4. **动态任务采样**：基于softmax验证损失的动态采样比例，平衡多任务训练中的难例挖掘
5. **多阶段adapter averaging**：通过concatenation averaging融合多阶段adapter，在保留各阶段能力的同时提升泛化

## 关键术语表
- **VGL-stGen**：Vision-Gene Language model–spatial transcriptomics generative，多模态大语言模型，适配肺组织学与空间转录组联合任务
- **QLoRA**：Quantized LoRA，基于4-bit NF4量化的低秩适配技术，rank=16, alpha=32
- **Pair-F1**：Gene-level pair F1 score，衡量查询基因-表达等级对预测准确性的指标
- **TACCO**：Annotation transfer and decomposition method for cell identities，用于跨数据源细胞类型注释转移
- **LuCA atlas**：Lung cancer atlas，肺癌单细胞参考图谱，提供细胞类型标签
- **RA-CNN**：Deep learning-based tumor segmentation model，用于TCGA-slide的肿瘤区域分割标注
- **Focal Loss**：处理类别不平衡的损失函数，γ=2.0，配合label smoothing=0.1使用
- **Prefix-tree decoding**：前缀树解码，限制分类任务输出为预定义类别集合

## 可复现要素
- **数据集**：PortraiATLAS（69个ST-H&E样本）、TCGA-LUAD、GSE127465、10x公共肺癌空间数据集（Visium/Xenium）；论文未明确是否公开全部数据
- **代码/权重**：https://huggingface.co/Portraier/VGL-stGen-0.5-Lung，计划preprint发布时开放gated access
- **关键超参**：QLoRA rank=16, alpha=32, dropout=0.05, LR=1e-4（蒸馏阶段）/2e-4（癌症阶段）, batch=512（effective）, epoch=1, focal loss γ=2.0, label smoothing=0.1
