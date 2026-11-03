---
title: "zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature"
source: https://arxiv.org/pdf/2610.17696v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:56:17"
field: "蛋白质‑配体结合位点预测"
keywords: ["口袋预测", "共折叠模型", "冻结特征", "基准数据集", "药物发现", "隐蔽口袋"]
innovations: ["提出冻结共折叠 trunk + 轻量级残基级头部框架", "构建时间划分的 PocketWatch 基准并分层评估泛化能力", "设计重排序与混合解码器提升现有检测器性能"]
benchmarks: ["PocketWatch", "Fragment‑screen simulation", "Human proteome scan"]
---

# 论文速读：zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature

## 一句话总结
本文提出 **zPocket**，一种在被冻结的共折叠模型（如 ESMFold2、Protenix）trunk 特征上训练轻量级残基级预测头的口袋预测方法，并在自建的 **PocketWatch** 时间划分基准上显著超越几何检测器与已有深度学习基线。

## 研究问题与动机
- **现有方法依赖输入结构**：传统口袋检测器需显式几何输入，而共折叠模型已从序列学习中蕴含丰富的结合位点信息，却未被用于口袋预测。
- **缺乏合适的泛化基准**：已有基准的口袋已出现在 AF3 时代共折叠模型的训练 cutoff 之前，无法区分“记忆”与真正泛化。
- **隐蔽口袋评估不足**：静态结构上的口袋预测难以覆盖仅在配体结合后开放的隐蔽口袋，需结合结构相似性与隐蔽性分层评估。

## 核心贡献（创新点）
1. **构建 PocketWatch 时间划分基准**：测试集包含模型未见过的新口袋，按序列相似性、结构相似性（max_qcov）和隐蔽性分层，首次公平评估共折叠模型的泛化能力。
2. **提出 zPocket 冻结 trunk + 轻量级头部框架**：在不微调共折叠主体的情况下，仅训练参数量极小的残基级预测头，即可从 trunk 内部表示（pair representation z）中提取口袋信号。
3. **系统验证多共折叠 trunk 的口袋信号通用性**：证明口袋信号广泛存在于 Protenix、ESMFold2、Chai-1、RF3 等多种开放权重的共折叠模型中，且头部架构比 trunk 选择更重要。
4. **发明重排序解码器提升现有检测器性能**：zPocket 预测头可重新排序 P2Rank、fpocket、Lacuna 等方法的候选质心，显著提高 top‑N+2 召回率。
5. **展示两个下游应用**：晶体学片段筛选模拟与全人类蛋白质组（~2 万蛋白）的 novel pocket 假设挖掘，体现方法在药物发现中的实用价值。

## 方法详解
- **特征缓存**：对每条蛋白质序列，用冻结的共折叠模型（ESMFold2 或 Protenix‑v2）折叠一次，缓存内部表示：pair representation **z** ∈ ℝ^(L×L×C) 与 per‑residue 表示 **s**。
- **头部架构**：四个变体（参数均 < 33 万）：
  - **pooled**：直接对 z 的行/列均值与对角线拼接后过 LayerNorm + MLP。
  - **axial**：先做 row‑then‑column 自注意力，再 pooling。
  - **pairformer1/2**：堆叠 K 个 AlphaFold‑style 三角对块后进行相同 readout。
- **训练损失**：每残基加权 focal BCE：  
  `BCE_{w,γ}(ŷ, y) = −[w(1−ŷ)^γ y log ŷ + ŷ^γ (1−y) log(1−ŷ)]`，默认 w=1, γ=0。
- **解码策略**：
  - **聚类解码**：阈值 τ=0.3 以上的残基在 Cα 坐标上做加权 mean‑shift，密度峰作为口袋质心。
  - **重排序解码**：对任意检测器提出的候选质心，聚合其 lining 残基的头部概率（mean/max/sum/k‑highest），按得分重排。
  - **混合解码**：合并 zPocket 聚类候选与其他检测器候选后再重排序。
- **实验设置**：trunk 完全冻结，每个蛋白 forward 一次；Adam 优化，lr=2e−3，8 蛋白累积梯度一步，训练 20 epoch，取验证集 best per‑target AUPRC checkpoint。

## 实验与结果
- **数据集**：PocketWatch 有效测试集含 1,275 蛋白质、1,562 个口袋，正残基比例 4.86%。
- **基线**：P2Rank、fpocket、Lacuna、UniSite‑1D（pub & PW）、VN‑EGNN（pub & PW）。
- **主要结果（per‑target AUPRC）**：
  - zPocket‑ESMFold2 pairformer2：**0.745**（AUROC 0.951）
  - VN‑EGNN (PW)：0.598；UniSite‑1D (PW)：0.595；P2Rank：0.477
  - 重训练基线较原始权重有提升，但仍远低于所有 zPocket 配置。
- **分层表现**：在序列相似性、结构相似性（max_qcov）、隐蔽性三个分层上，zPocket 均领先所有基线；在 max_qcov < 20% 的 truly novel 口袋上 AUPRC 降至 0.327，但仍为各分层最优。
- **多 trunk 比较**：七种共折叠 trunk 的 pairformer2 头 AUPRC 均在 0.733–0.757；Protenix‑v2 略优（0.757），Chai‑1 紧随（0.746）。集成全部七个 trunk 的 logits 可达 **0.777**。
- **口袋质心预测（DCC_N+2 @ 4Å）**：
  - zPocket 聚类解码器精度最高（60% 命中真实位点），但平均仅输出 1.1 个口袋/蛋白，召回受限。
  - 重排序显著提升所有基线：P2Rank 从 0.324 升至 0.426，fpocket 从 0.197 升至 0.426。
  - 混合解码（zPocket 候选 + P2Rank/Lacuna/VN‑EGNN 候选）进一步将 DCC_N+2 推高至 0.675（VN‑EGNN PW + zPocket）。
- **应用验证**：
  - **片段筛选模拟**（12 蛋白，35 口袋）：zPocket AUPRC 0.494，重排序 fpocket 后 DCC_N+2 达 0.391。
  - **人类蛋白质组扫描**：产出 ~22,000 个预测口袋，经几何与 novelty 过滤后保留 155 个新口袋假设（如 DET1、COP1）。

## 相关工作脉络
1. **传统几何检测器**（fpocket、P2Rank）：基于溶剂可及表面与物理化学描述符打分，zPocket 完全绕过显式几何计算。
2. **深度学习口袋预测器**（VN‑EGNN、UniSite‑1D）：分别从图结构或序列预测，zPocket 使用冻结 trunk 特征且无需结构输入，泛化性更强。
3. **AF2Bind**：首次利用 AlphaFold2 trunk 预测口袋，但需添加 20 个假配体残基；zPocket 直接读取 pair representation，更简洁。
4. **IsoDDE / CryptoBench**：侧重隐蔽口袋评估，但未做时间划分，可能泄露训练数据；PocketWatch 明确分离训练/测试 cutoff，防止过拟合。
5. **Runs N’ Poses**：指出共折叠模型在分布外口袋上性能崩塌；本文沿用其 max_qcov 分层思想，进一步量化泛化差距。

## 局限性与未来方向
- 测试集中真正新颖（max_qcov < 20%）的口袋仅占 28.4%，整体性能仍受熟悉口袋主导。
- 仅评估单链口袋，界面口袋（interface pockets）因被 mask 而未测试。
- 预测口袋不一定是功能位点，但作者认为非正构口袋在降解剂/分子胶时代同样具有治疗价值。
- 聚类解码器平均每个蛋白仅输出 1.1 个口袋，难以应对多口袋蛋白。
- 计算开销较高：每条序列需完整运行一次共折叠推理（ESMFold2 10 loop、Protenix 10 recycle）。

## 研究启发与可借鉴点
- **“冻结 trunk + 轻量头部”范式**可复用于其他生物分子属性预测（如结合亲和力、突变效应、复合物界面）。
- **时间划分 + 多维度分层（序列/结构/隐蔽性）** 是评估大模型泛化能力的可靠策略，可迁移至其他结构生物学任务。
- **pair representation z 中蕴含强口袋信号**，即使最简 pooled 头也能超越大部分基线，提示后续工作可优先挖掘二阶交互特征。
- **重排序解码器**作为即插即用模块，可低成本提升任意候选生成器的排名质量。
- **人类蛋白质组扫描流程**（ novelty filters + 几何阈值 +  motif 传播）为长尾靶点挖掘提供了可复现的 pipeline。

## 关键术语表
- **PocketWatch**：本文构建的时间划分口袋预测基准数据集，测试集包含 cutoff 之后首次观测到的口袋。
- **zPocket**：基于冻结共折叠 trunk 特征训练轻量级残基级预测头的方法。
- **Co‑folding model**：能同时预测蛋白质与配体/复合物结构的深度学习模型（如 ESMFold2、Protenix）。
- **Pair representation (z)**：共折叠模型 trunk 中编码残基间相互作用的二维特征矩阵。
- **Per‑target AUPRC**：以蛋白质为单位计算的平均Precision‑Recall 曲线下面积，平均时等权对待每个蛋白。
- **DCC_N+2**：在预测列表中允许 N+2 个候选（N 为蛋白真实口袋数）时，4Å 半径内命中的真实口袋比例。
- **Cryptic pocket**：仅在配体结合后出现的隐蔽口袋，在游离态 apo 结构中无法观测到。
- **max_qcov**：测试口袋残基被 pre‑cutoff 口袋最大覆盖的比例，用于衡量结构相似性。

## 可复现要素
- **数据集**：PocketWatch 已在 Hugging Face 公开（https://huggingface.co/datasets/genesisml/PocketWatch）。
- **代码**：GitHub 开源（https://github.com/genesistherapeutics/zpocket）。
- **模型权重**：Hugging Face 提供（https://huggingface.co/genesisml/zPocket）。
- **关键超参**：Adam，lr=2e−3，20 epoch，batch=8（累积梯度），τ=0.3，带宽=14Å，m=3；pair width C=256（ESMFold2/Protenix‑v2/Chai‑1）或 128（其余 trunk）。
- **特征存储**：半精度缓存 per‑sequence 的 z（ℝ^(L×L×C)）与 s；每条序列仅推理一次 trunk。
