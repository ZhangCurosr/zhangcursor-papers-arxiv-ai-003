---
title: "RT-J-Large-Scale-Pretraining-of-Relational-Transformers-for"
source: https://arxiv.org/pdf/2610.17867v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:00:24"
field: "关系深度学习"
keywords: ["关系基础模型", "In-context Learning", "预训练", "随机游走检索", "表格预测"]
innovations: ["两阶段预训练配方（PluRel 合成数据 + THE JOIN 真实数据）", "轻量级随机游走检索器用于关系上下文构建", "测试时上下文调整与集成扩展"]
benchmarks: ["RelBench", "RelArena-α"]
---

# 论文速读：RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions

## 一句话总结
论文提出了 RT-J，一个 86M 参数的关系基础模型，通过在大规模合成数据（PluRel）和真实数据（THE JOIN，639 个数据库、6023 个 LLM 生成预测任务）上长上下文预训练，并结合随机游走检索器和测试时上下文调整/集成，实现了在仅有少量 in-context 示例时的高精度关系预测，比强特征工程流水线少用约 10× 示例即可匹配。

## 研究问题与动机
- **关系数据的 few-shot/in-context 预测能力缺失**：现有关系预测系统要么依赖任务特定训练，要么需要大量标注数据，限制了在标注稀缺场景（如小样本企业预测、冷启动实体预测）中的应用。
- **现有关系基础模型（RFM）的不足**：RelGT、RelGNN 等架构虽原生处理多表数据，但需为每个 schema 和任务训练；Griffin、KumoRFM、RT 等 RFM 虽支持零样本/少样本，但未针对少量 in-context 示例的强预测进行优化。
- **预训练配方与推理需求不匹配**：现有 RFM 的预训练配方主要针对 zero-shot 或 fine-tuning 模式，无法有效利用 in-context 示例；上下文构建要么忽略关系结构，要么通过朴素 BFS 扩展导致相关行检索质量低。
- **关系数据上下文构造的特殊性**：目标行周围的关系邻域存在重叠，独立采样 support rows 会浪费上下文预算；需要能够优先检索与目标行关系邻近且携带预测信号的行的检索机制。

## 核心贡献（创新点）
- **THE JOIN 大规模预训练语料库**：收集 639 个真实世界关系数据库（涵盖电商、体育、金融、医疗等），通过 LLM 生成 6023 个预测任务，覆盖罕见事件、冷启动、重尾分布等硬场景；这是目前已公开最大的真实关系预训练数据集。
- **两阶段预训练配方（PluRel → THE JOIN）**：先在合成数据 PluRel 上预训练获得 RT-PluRel，再在 THE JOIN 上继续预训练得到 RT-J；合成数据提供结构化学习信号，真实数据提升泛化能力，两者结合在 in-context 学习下带来显著增益。
- **轻量级随机游走检索器**：提出两阶段上下文构建（随机游走全局检索 + 有界 BFS 局部扩展），通过并行随机游走对目标表中的候选行进行访 visited 计数排序，再作为 BFS 种子检索高质量 in-context 示例；相比原 RT 的朴素 BFS 或向量相似度检索，更有效地选择关系邻近且携带预测信号的行。
- **测试时计算扩展技术**：通过上下文超参数调优（对每个任务在验证集上搜索最优 $(L, \ell, b, \rho)$）和上下文集成（多 seed 采样上下文后平均预测），在不增加训练成本的情况下提升推理性能；在分类任务上集成 16 个上下文可带来 +1.1–1.3 AUROC 增益。

## 方法详解
- **模型架构**：基于 Relational Transformer (RT)，86M 参数，12 层 Transformer 块，隐藏维度 512，8 个注意力头；每个块依次应用列注意力（同列 token 间）、特征注意力（同行及其父行间）、邻居注意力（主键-外键链接行间），结合 gated FFN 和 QK-Norm。
- **Cell Token Embedding**：每个 cell 视为 token，embedding 由值编码（数值/类别/文本类型特定投影）和 schema 语义编码（表名+列名的 frozen text embedding，来自 all-MiniLM-L12-v2）拼接而成。
- **预训练任务**：掩码单元格预测（Masked Token Prediction），仅掩码目标 cell（$p_{\max}=0$），使用 Huber 损失（回归）和类别输出头；支持 autocompletion（掩码已有列）和 forecasting（预测未来时间窗标签）任务。
- **随机游走检索器设计**：从目标行 $r$ 出发执行 $W=10{,}000$ 条长度 $K=20$ 的随机游走，沿主键-外键边移动并约束时间戳不超过目标时间；统计目标表中各行被访问次数，按访 visited 计数（可选时间衰减）排序后，作为 BFS 种子逐行扩展局部上下文，直到总 cell 数达到 $L$。
- **上下文超参数采样**：训练时从集合中随机采样上下文尺寸 $L \in \{512, \ldots, 8192\}$、局部预算 $\ell$、BFS 宽度 $b \in \{8, \ldots, 256\}$（2 的幂）、时间衰减参数 $\rho \in \{0,1\}$，使模型适应多样化上下文分布。
- **优化设置**：使用 Muon 优化器（2D 权重）和 AdamW，恒定学习率 $5 \times 10^{-4}$，线性 warmup 2000 步，SWA（动量 0.9995）选择最佳 checkpoint；每阶段训练约 1 天（8×A100 GPU）。

## 实验与结果
- **评测基准**：RelBench（21 个预测任务，7 个数据库，9 个回归 + 12 个分类），所有评测数据库未在预训练中出现。
- **最强基线**：特征工程流水线（RDBLearn 或 LLM Agent 提取特征 + TabICLv2 或 LightGBM 预测），每个方法看到与 RT-J 相同的 in-context 示例。
- **回归结果**：RT-J 在所有上下文尺寸下均领先，2.3–5.0 nMAE 点优势；仅需 15 个 in-context 示例即可击败最强基线在 225 示例时的最佳误差，匹配同等性能需约 10× 更少示例。
- **分类结果**：示例稀缺时领先 3.1 AUROC 点；在最大上下文尺寸（8192 cells，约 225 示例）下最强的 TabICLv2 流水线接近 RT-J，但 RT-J 在少样本场景优势显著。
- **消融结论**：
  - THE JOIN 预训练阶段在 in-context 下带来 +0.6 nMAE / +1.5 AUROC 增益；
  - 随机游走检索器 vs 均匀随机检索：+4.1 nMAE / +6.1 AUROC 增益；
  - Schema 语义（表/列名文本嵌入）去除导致退化，且随上下文增大差距扩大；
  - 上下文集成（16 个 seed）带来 +1.1 AUROC 分类增益、+0.2 nMAE 回归增益。
- **RelArena-α  leaderboard**：RT-PluRel 以 Elo 1861 排名第一，领先 TabPFN-Rel（1826）及所有关系深度学习基线。

## 相关工作脉络
- **Tabular Foundation Models (TFMs)**：TabPFN、TabICL/TabICLv2 提供单表 in-context 学习；本文扩展到多表关系场景，保留 cell-level 原生处理而非 flatten 为特征表。
- **Relational Deep Learning**：RelGT、RelGNN 等原生处理多表数据但需任务训练；本文的 RT-J 支持零训练 few-shot 预测。
- **Existing RFMs**：Griffin（图中心架构）、KumoRFM（闭源/不支持精确 in-context 控制）、RDB-PFN（仅支持分类）无法公平比较；本文聚焦 RT 架构的 recipe 改进。
- **Relational Feature Engineering**：RDBLearn 将关系邻域聚合为平坦特征表再输入 tabular predictor；本文保留关系结构原生推理，避免信息损失。
- **Graph Foundation Models**：Mole-BERT、PRODIGY、GraphAny 等针对分子/通用图预训练；本文针对关系数据库的 schema 感知和 cell-level 建模。

## 局限性与未来方向
- **上下文长度限制**：RT-J 仅训练到 8192 cells，更长上下文（支持更多 in-context 示例）需高效长序列训练技术。
- **任务类型受限**：无法解决 link prediction 和推荐任务（与原始 RT 相同限制）。
- **评测规模有限**：仅在 RelBench（21 任务/7 数据库）评测，建议扩展到更大真实世界 benchmark。
- **实时推理延迟**：随机游走检索在测试时有 ~28ms CPU 开销，虽可接受但可进一步优化。
- **未来方向**：扩展至 link prediction、开发更长的上下文训练方法、构建更大规模的关系预测 benchmark。

## 研究启发与可借鉴点
- **随机游走检索用于关系上下文构建**：无需预计算 embedding，可在线并行执行，高效捕捉关系邻近性；可迁移到图结构的 few-shot 学习场景。
- **两阶段预训练（合成→真实）**：合成数据提供结构可控的学习信号，真实数据注入领域语义；该配方适用于其他需要结构化知识迁移的 foundation model。
- **测试时 compute 扩展策略**：上下文超参数调优 + 多 seed 集成，以少量额外计算换取显著性能提升；可应用于其他 in-context learning 系统的部署优化。
- **Schema 语义嵌入的重要性**：表名/列名的 text embedding 提供先验知识，且其作用随上下文增大而增强；值得在 relation-aware NLP 任务中进一步探索。
- **任务模板化生成方法**：使用 LLM 生成预测任务 proposal + 确定性 checker 验证 + XGBoost 难度过滤的流程，可复用于其他领域的预训练数据构建。

## 关键术语表
- **Relational Foundation Model (RFM)**：针对关系数据库预训练的基础模型，支持跨 schema 迁移和零样本/少样本预测。
- **In-context Learning (ICL)**：在推理时通过 prompt/context 中的示例学习，无需梯度更新；本文将其应用于关系预测。
- **THE JOIN**：本文构建的大规模预训练语料库，包含 639 个真实关系数据库和 6023 个 LLM 生成的预测任务。
- **Random-walk Retriever**：两阶段检索器，先用并行随机游走评分目标表中的候选行，再用 BFS 扩展局部上下文，用于构建高质量 in-context 示例集。
- **Context Tuning / Ensembling**：测试时计算扩展技术；前者在每个任务上搜索最优上下文超参数，后者对多个随机 seed 采样的上下文预测取平均。
- **Relational Transformer (RT)**：原生处理多表关系的 Transformer 架构，将 cell 视为 token，应用列/特征/邻居三层注意力。
- **PluRel**：合成关系数据预训练语料，用于第一阶段训练，提供结构化的多任务学习信号。
- **nMAE / AUROC**：归一化平均绝对误差（回归，越低越好）和 ROC 曲线下面积（分类，越高越好），本文使用的主要评估指标。

## 可复现要素
- **数据集**：THE JOIN 已公开（CC-BY-SA 4.0），含 639 个数据库；PluRel 已在先前工作中公开。
- **代码/模型**：RT-J、RT-PluRel 权重及预训练/微调/推理代码已开源，访问 https://star-project.stanford.edu/rt-j。
- **关键超参**：模型 86M 参数，12 层，隐藏 512，8 head；预训练 batch size 1024，lr $5 \times 10^{-4}$，warmup 2000 步；随机游走 $W=10{,}000$、$K=20$；上下文尺寸 $L \in \{512, \ldots, 8192\}$。
