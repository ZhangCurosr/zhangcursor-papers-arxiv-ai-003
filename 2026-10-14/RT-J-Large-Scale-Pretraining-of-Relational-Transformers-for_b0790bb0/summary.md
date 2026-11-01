---
title: "RT-J-Large-Scale-Pretraining-of-Relational-Transformers-for"
source: https://arxiv.org/pdf/2610.17867v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:00:10"
field: "关系深度学习"
keywords: ["Relational Foundation Model", "In-context Learning", "Relational Transformer", "Tabular Prediction", "Masked Token Prediction", "Random Walk Retrieval"]
innovations: ["提出THE JOIN：639库/6023任务的真实关系预训练语料库", "改进RT预训练食谱：随机上下文大小+随机游走检索器支持上下文学习", "测试时计算扩展：上下文调优+集成实现少样本高效预测"]
benchmarks: ["RelBench", "RelArena-α"]
---

# 论文速读：RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions

## 一句话总结
本文提出RT-J，一个85M参数的关系基础模型（RFM），通过在639个真实Web爬取数据库（THE JOIN语料库）上进行大规模预训练，仅用数百个甚至更少的上下文示例即可完成高精度预测，在RelBench基准上比最强特征工程流水线少用约10倍示例。

## 研究问题与动机
- **关系预测在少样本场景下仍缺乏有效方法**：现有关系预测器通常依赖任务特定训练或大量标注数据，在标注稀缺但需要快速ad-hoc预测的场景（如临床罕见事件、冷启动用户预测）中受限。
- **现有RFM未针对上下文学习场景优化**：KumoRFM、Griffin等模型虽支持跨模式迁移，但未针对"极少上下文示例"这一推理 regime 进行架构或训练食谱设计。
- **预训练数据与食谱存在缺口**：公开真实关系预训练数据稀缺；原RT食谱面向零样本/微调而非上下文学习；上下文构建忽略关系结构或朴素展开局部邻域。
- **单表模型无法直接迁移到关系数据**：TabPFN/TabICL等单表基础模型需先将关系数据库flatten为特征表，丢失了跨表关系、schema语义和实体级关联结构。

## 核心贡献（创新点）
- **提出THE JOIN大规模真实关系预训练语料库**：采集639个Web数据库，由LLM自动生成6,023个预测任务（涵盖25种模板家族），是迄今最大的开源关系预训练语料。*区别*：不同于以往主要依赖合成数据（如PluRel），THE JOIN覆盖电商、体育、金融、医疗等真实领域，具有丰富schema语义。
- **改进RT预训练食谱以支持上下文学习**：引入随机上下文大小采样（最长8192单元格）、随机游走检索器、两阶段上下文构建；*区别*：原RT食谱面向零样本掩码预训练，未考虑推理时少样本上下文示例的检索与利用。
- **提出轻量级随机游走检索器**：通过W=10,000条长度K=20的并行随机游走对目标表候选行排名，再执行有界宽度BFS构建上下文；*区别*：不同于向量相似度检索（需预计算）或朴素BFS，该检索器无需离线索引、可在线运行且优先考虑关系近邻中的高质量支持行。
- **实验验证上下文效率显著提升**：在RelBench回归任务上，RT-J在每上下文大小下比最强基线低2.3–5.0 nMAE点，且仅需约10倍更少示例即可匹敌；*区别*：相比RDBLearn+TabICLv2等流水线，RT-J直接处理原始关系上下文而非flatten后特征。
- **开放全部模型、数据与代码**：释放RT-J/RT-PluRel权重、THE JOIN语料、LLM任务生成pipeline及预训练/微调/推理代码。

## 方法详解
- **模型骨架**：基于Relational Transformer（RT）架构，86M可训练参数，12层，hidden size=512，8个注意力头（d_h=64），含QK-Norm、SwiGLU FFN等改进。
- **Cell级输入表示**：每个单元格视为token，嵌入由cell值嵌入和schema语义嵌入（冻结all-MiniLM-L12-v2编码表名+列名）相加构成；布尔值标准化为{0,1}，数值列按列z-score归一化，时间列使用全局归一化。
- **三元注意力机制**：每层依次应用column attention（同列内）、feature attention（同行及父行内）、neighbor attention（跨主外键链接行），掩码由schema图决定。
- **预训练两阶段**：第一阶段在PluRel合成数据（86,211任务/1,900库）上预训练得RT-PluRel；第二阶段在THE JOIN的523个数据库子集（13,243任务）上继续预训练得RT-J，共计约70×预训练FLOPs于原RT。
- **上下文采样超参随机化**：全局上下文大小L∈{512,…,8192}、局部大小ℓ∈{128,…,8192}、BFS宽度b∈{8,…,256}（2的幂）、时效偏置ρ∈{0,1}均随机采样，使模型适应多尺度上下文。
- **两阶段检索构建上下文**：①全局检索：从目标行出发执行W=10,000条、长度K=20的随机游走，统计目标表行的访问频次排名；②局部扩展：按排名依次以候选行为种子做有界BFS填充至总单元格预算L。
- **掩码与损失**：仅掩码目标单元格（p_max=0），非目标单元格不掩码；所有类型目标使用Huber损失（δ=1），优化器为Muon（2D权重）+ AdamW，学习率5×10⁻⁴，线性warmup 2000步，SWA动量0.9995选最优checkpoint。
- **测试时计算扩展**：上下文调优（在验证集上网格搜索120组(L,ℓ,b,ρ)选最优）和上下文集成（n_ens∈{1,2,4,8,16}种子取平均预测）。

## 实验与结果
- **评估基准**：RelBench（21个任务：9回归+12二分类，涵盖罕见事件、冷启动、重尾、长视界等困难 regime），所有评测数据库在预训练阶段完全未见。
- **最强结果（in-context，默认上下文L=8192）**：回归平均nMAE=32.35%，分类平均AUROC=74.45%；RT-PluRel对应32.95%/72.98%。
- **相对基线提升**：在匹配示例数量下，RT-J在每种上下文大小上均领先最强pipeline 2.3–5.0 nMAE点（回归）和最高3.1 AUROC点（分类）；仅15个示例时已优于任意pipeline用225示例的最佳误差。
- **基线构成**：RDBLearn featurizer + TabICLv2/LightGBM predictor；LLM Agent (Claude Opus 4.6写SQL特征工程) + TabICLv2/LightGBM。其他RFM（KumoRFM、Griffin、RDB-PFN）因闭源或不支持本regime无法比较。
- **合成→真实数据增益**：PluRel预训练带来微调nMAE↓1.7、AUROC↑0.8；THE JOIN继续预训练额外带来微调nMAE↓0.5、AUROC↑0.5；在上下文设定下带来nMAE↓0.6、AUROC↑1.5。
- **RelArena-α排名**：RT-PluRel作为系统提交排名第一（Elo=1861），超过TabPFN-Rel（1826）及所有关系深度学习基线。
- **消融**：随机游走检索vs均匀随机：回归+4.1 nMAE、分类+6.1 AUROC；schema语义扰动：分类退化最高2.1 AUROC点；多单元格掩码劣于单单元格掩码。

## 相关工作脉络
- **RT [23]**：本文基线架构，cell-level关系Transformer，但原食谱面向零样本/微调，未针对少样本上下文学习优化；本文在此基础上改进预训练与推理食谱。
- **PluRel [26]**：合成关系数据生成器，用于RT-PluRel预训练；本文沿用其作为第一阶段数据，但强调真实数据在schema语义吸收上的关键作用。
- **RDBLearn [21]**：将关系邻域聚合为扁平特征表后接TabICLv2；本文将其作为最强baseline之一，指出其flatten操作丢失跨表关系推理能力。
- **KumoRFM/KumoRFM-2 [10,11]**：关系上下文学习模型，但权重闭源且API不支持控制上下文示例，无法公平比较。
- **Griffin [22]**：图中心RFM，不支持上下文学习，架构亦非纯transformer，不适合本实验设置。
- **TabICLv2 [19,20]**：单表上下文学习foundation model，作为pipeline预测器组件使用；本文通过原生关系建模在其之上取得显著优势。

## 局限性与未来方向
- **上下文长度上限8192单元格**：限制可同时纳入的示例数和邻居规模；未来需探索高效长上下文训练（如线性注意力、稀疏Attention）以突破此瓶颈。
- **不支持链接预测与推荐任务**：与原始RT相同，当前设计仅覆盖autocompletion和forecasting两类cell预测任务。
- **评测范围有限**：仅在RelBench 21个任务/7个数据库上验证，多样性不及THE JOIN所覆盖的639库；需更大规模真实基准。
- **随机游走检索的延迟开销**：默认设置W=10,000/K=20带来约28ms/cell-sampler延迟，虽低于forward pass的10ms但仍有优化空间；可调参数敏感性表明降W/K可显著降延迟而精度损失有限。
- **上下文调优在分类任务上略降均值性能**：因val→test transfer不稳定，仅5/12分类任务受益，提示未来需更鲁棒的自动超参选择策略。

## 研究启发与可借鉴点
- **随机游走检索器可迁移至其他关系/图学习系统**：无需预计算嵌入、CPU并行、在线运行的特性适合大规模关系数据库的实时上下文构建，可作为通用"相似行检索"模块集成到RDBLearn、RelGT等框架中。
- **上下文调优+集成作为测试时计算扩展范式**：对任何固定容量 transformer，通过多配置采样与平均可稳定提升鲁棒性，尤其对分类任务收益显著（+1.1 AUROC）；可推广至单表TFLs或图 foundation models。
- **真实数据预训练对schema语义吸收至关重要**：shuffle表/列名实验显示分类退化最高2.1 AUROC，且差距随示例数增加扩大；提示任何关系模型预训练必须包含具语义的真实schema名称分布，而非仅结构数据。
- **混合预训练任务（forecasting+autocomplete）的稳健性**：单一任务类型（仅forecasting或仅autocomplete）在另一指标上均出现明显退化；未来多任务预训练设计可参考此混合策略。
- **长上下文随机采样训练提升推理灵活性**：批量内随机化L、ℓ、b、ρ使单一checkpoint适应多种上下文尺度；该方法可迁移至长上下文LLM训练，避免过拟合固定窗口。

## 关键术语表
- **RT-J**：本文提出的85M参数关系基础模型，基于RT架构经PluRel+THE JOIN两阶段预训练，专精少样本上下文预测。
- **THE JOIN**：639个真实Web爬取数据库组成的预训练语料库，含6,023个LLM生成预测任务+11,000个autocomplete任务，以CC-BY-SA 4.0发布。
- **随机游走检索器**：两阶段上下文构建的全局检索模块，通过并行随机游走统计目标表行访问频次，优先选取关系近邻中高价值支持行。
- **上下文调优（Context Tuning）**：在验证集上网格搜索(L,ℓ,b,ρ)最优组合，应用于测试集以提升单任务预测质量。
- **上下文集成（Context Ensembling）**：对n_ens个独立随机种子采样的上下文分别预测后取平均，降低检索随机性带来的方差。
- **RelBench**：斯坦福提出的关系深度学习基准，含21个实体级时序预测任务（9回归+12分类），覆盖冷启动、罕见事件、重尾等困难场景。
- **nMAE**：Z-score归一化的平均绝对误差，回归任务评估指标，越低越好。
- **RFM（Relational Foundation Model）**：直接在关系数据库上预训练、支持跨schema迁移的关系基础模型统称。

## 可复现要素
- **数据集**：THE JOIN已开源（CC-BY-SA 4.0）；PluRel为已有合成数据；RelBench评测集为 held-out 标准划分。
- **代码/权重**：RT-J、RT-PluRel权重，THE JOIN数据，LLM agent任务生成pipeline，预训练/微调/推理代码均已在 https://star-project.stanford.edu/rt-j 开源。
- **关键超参**：L∈{512,…,8192}、ℓ∈{128,…,8192}、b∈{8,16,32,64,128,256}、ρ∈{0,1}、W=10,000、K=20、学习率5×10⁻⁴、batch size=1024、Muon+AdamW、SWA动量0.9995；未提及的超参详见原文附录G/H。
