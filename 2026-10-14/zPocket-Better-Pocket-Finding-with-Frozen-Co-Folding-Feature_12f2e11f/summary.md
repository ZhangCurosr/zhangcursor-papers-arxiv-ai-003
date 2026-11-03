---
title: "zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature"
source: https://arxiv.org/pdf/2610.17696v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:56:03"
field: "蛋白质-配体结合位点预测"
keywords: ["protein-ligand binding site prediction", "co-folding model", "pocket detection", "frozen backbone fine-tuning", "temporal benchmark"]
innovations: ["在冻结共折叠trunk的pair表征上训练轻量pairformer head用于per-residue口袋预测", "构建PocketWatch时序拆分基准并按序列/结构相似性与crypticity分层评估泛化", "提出重排序与聚类混合解码策略，将zPocket作为通用rescorer提升多种基线方法的centroid召回"]
benchmarks: ["PocketWatch", "XChem fragment screening", "human proteome scan"]
---

# 论文速读：zPocket: Better Pocket Finding with Frozen Co-Folding Features

## 一句话总结
论文提出了 **zPocket**，一种在冻结的共折叠模型（co-folding model）特征上训练口袋预测头的方法，并结合新构建的时序拆分基准 **PocketWatch**，显著优于现有蛋白-配体结合位点预测方法。

## 研究问题与动机
- 蛋白-配体结合口袋（pocket）的识别对理解蛋白功能和药物开发至关重要，但现有方法严重依赖输入结构，难以利用共折叠模型学到的丰富表示。
- 已有基准（如 UniSite、LIGYSIS）的时间切分早于 AF3 时代共折叠模型的训练截止期，导致无法有效评估模型对全新口袋的泛化能力。
-  cryptic pockets（隐性口袋）在 apo 结构中闭合、仅在 holo 结构中出现，对基于静态结构的预测方法构成严峻挑战。
- 缺乏公开可用的、针对 AF3 时代共折叠模型设计的盲口袋预测基准，阻碍了该方向的科学比较。

## 核心贡献（创新点）
1. **构建 PocketWatch 时序拆分基准**：以 2021-09-30 和 2024-01-01 为截断点建立 train/val/test，按序列相似性、结构相似性（max_qcov）和 crypticity 分层，首次支持评估共折叠模型对真·新口袋的泛化。
2. **提出 zPocket 冻结头架构**：在冻结的共折叠 trunk（ESMFold2 / Protenix-v2）pair 表征 z 上训练轻量 per-residue 预测头，核心设计是利用 pairformer 三角注意力块在 pooling 前充分混合 pair 信号。
3. **证明 pocket 信号广泛存在于多种共折叠 trunk 中**：在 7 种开源共折叠模型（Protenix-v1/v2、ESMFold2、Chai-1、OpenFold3-preview、Boltz-1、RF3）上统一训练 head，pairformer2 在所有 trunk 上均远超最强基线。
4. **提出可复用的 centroid 解码策略**：包括聚聚类解码（mean-shift）和重排序解码（re-ranking），后者可将 zPocket 作为后处理器大幅提升 fpocket、P2Rank、VN-EGNN 等方法在 DCC_{N+2} 上的表现。
5. **开放数据集、代码与权重**：PocketWatch 数据集、zPocket 代码及模型权重全部开源。

## 方法详解
- **数据构建（PocketWatch）**：
  - 从 PDB（截至 2026-01-02）中筛选 BioLiP2 定义的"生物学相关"配体。
  - 配体约束：10–50 个重原子，链长 40–1200，接触残基数 ≥10 且 SIFTS 覆盖率 >80%。
  - 排除结晶添加剂、金属簇、脂质、代谢物等；保留有机辅因子、糖类、核苷酸。
  - 将同一 UniProt 序列下的所有口袋实例按 ligand centroid 做 single-linkage 聚类（5 Å），得到离散 pocket site。
  - 按 earliest pocket instance 的释放日期做时间切分：≥2024-01-01 进 test，>2021-09-30 进 val，其余 train。
  - 屏蔽 interface-only 口袋（所有 member 均为多链接触），保证与单链方法的公平比较。

- **特征提取**：
  - 对每条蛋白序列用冻结的 co-folding trunk（带 MSA）折叠一次，缓存内部特征：
    - Protenix-v2：保留 pair 表征 $z \in \mathbb{R}^{L \times L \times 256}$ 和 single 表征 $s$。
    - ESMFold2：保留 $z \in \mathbb{R}^{L \times L \times 256}$ 和最终层 ESMC-6B embedding。
  - 所有 trunk 的实验训练结构均在 2021-09-30 之前，无验证/测试泄漏。

- **Head 架构**（四种，参数量 29K–330K）：
  - **pooled**：直接对 z 做 row-mean + col-mean + diag 拼接，经 LayerNorm + MLP 输出 logits。
  - **axial**：投影到窄宽度（32）后做 row-then-column self-attention。
  - **pairformer1**：1 个 AlphaFold-style 三角 pair block（width=64）。
  - **pairformer2**：2 个三角 pair block（width=64），效果最佳。
  - 所有 head 共享同一 readout：row mean / col mean / self-pair diagonal 拼接 → LayerNorm → MLP → scalar logit。

- **损失函数**：
  - 掩码 per-residue binary cross-entropy：
    $$\mathcal{L} = \frac{\sum_i m_i \cdot \text{BCE}_{w,\gamma}(\hat{y}_i, y_i)}{\sum_i m_i}$$
    其中 $\text{BCE}_{w,\gamma}(\hat{y}, y) = -[w(1-\hat{y})^\gamma y \log \hat{y} + \hat{y}^\gamma (1-y) \log(1-\hat{y})]$。
  - 主实验采用 $w=1, \gamma=0$（标准 BCE），val 上选择最佳 checkpoint（per-target AUPRC）。

- **解码策略**：
  - **聚类解码**：阈值 $\tau=0.3$ 以上残基 seed weighted mean-shift（bandwidth=14 Å，最小簇大小 $m=3$），输出 pocket centroid。
  - **重排序解码**：聚合候选 cavity 内 lining residues 的 head 概率（mean/max/sum/k-highest），重新排序。
  - **混合解码**：将 zPocket 聚类 centroid 并入其他方法的 proposal pool，再用 zPocket 统一重排。

- **基线**：fpocket、P2Rank、Lacuna（默认与 surface-fusion）、VN-EGNN（pub 与 PW retrained）、UniSite-1D（pub 与 PW retrained），均在 ESMFold2 folds 上运行。

## 实验与结果
- **数据集规模**：train 11,016 chains / 14,331 pockets；val 1,553 / 2,165；test 1,566 / 2,301；有效测试集 1,275 chains / 1,562 pockets，正残基率 4.86%。
- **Per-residue 主结果（ESMFold2 trunk）**：
  - 最优 zPocket pairformer2：**per-target AUPRC = 0.745**，per-pocket AUPRC = 0.702。
  - 最强基线 VN-EGNN (PW)：0.598；UniSite-1D (PW)：0.595；P2Rank：0.477。
  - 即便只用 final-layer ESMC-6B embedding（无 pair 信息），single head 也能达到 0.611，超过所有 published 基线。
- **Trunk 泛化**：7 种 trunk 上 pairformer2 得分介于 0.733–0.757，Head 架构差异（pairformer2 vs pooled）带来的增益（+0.08–0.12）远大于 trunk 间差异（0.024）。
- **Ensemble**：Protenix-v2 + ESMFold2 等权平均 logit 达到 **0.773**；7 trunk 全 ensemble 达到 **0.777**。
- **Retraining 增益**：VN-EGNN +0.039，UniSite-1D +0.64，说明 PocketWatch 训练集本身有价值。
- **Centroid 结果（DCC_{N+2}@4Å）**：
  - zPocket 聚类解码精度最高（60% proposals 命中真实 site），但 recall 受限于低发射率（平均 1.1 pockets/protein）。
  - Re-ranking 平均提升各方法 DCC_{N+2} 达 +0.090；Hybrid（加 zPocket cluster + re-rank）平均提升 +0.108。
  - 最佳组合 VN-EGNN(PW) + zPocket proposals：DCC_{N+2} = **0.675**。
- **分层结果**：在所有序列相似性 bin、max_qcov bin 和 crypticity bin 中，zPocket 均领先所有基线；但不熟悉口袋（max_qcov < 20%）AUPRC 降至 0.327。
- **应用 1 — 片段筛选（12 蛋白 35 sites）**：zPocket per-residue AUPRC = 0.494（vs P2Rank 0.459、VN-EGNN 0.438、UniSite-1D 0.385）；fpocket + zPocket re-rank 达到 DCC_{N+2} = 0.391。
- **应用 2 — 人类蛋白质组扫描**：对 ~20,000 reviewed human proteins 扫描得到 ~22,000 predicted pockets，经几何与 novelty filter 后得到 155 个候选新口袋；案例 DET1（degrader-recruiter site）和 COP1（peptide-binding hotspot subsite）展示实用性。

## 相关工作脉络
1. **AF2Bind [12]**：首次从 AlphaFold2 trunk 特征预测结合位点，但需附加 20 个 bait 氨基酸间接诱导信号；zPocket 直接在 frozen trunk pair 表征上训练轻量 head，无需 bait。
2. **P2Rank [19]**：经典 ML 口袋打分器，基于溶剂可及表面积点 + 局部物理化学描述符 + random forest 排序；zPocket 完全不需要几何输入。
3. **VN-EGNN [27]**：E(3)-等变图神经网络，在蛋白图结构上直接预测口袋；zPocket 在其折叠结果（ESMFold2 fold）上运行并大幅超越。
4. **UniSite [11]**：纯序列-based set predictor（Mask2Former 风格）；zPocket 在相同任务上提供更强的 per-residue 信号。
5. **Lacuna [23]**：通过构象 ensemble 检测 cryptic 口袋；zPocket 不依赖 ensemble 且在所有 crypticity stratum 均超越。
6. **IsoDDE [15] / AnewDDE [2]**：AF3 时代商业 co-folding 模型声称在盲口袋预测上表现优异，但缺乏公开基准与代码；zPocket 首次在开源、时序严格拆分的基准上量化比较。

## 局限性与未来方向
- **新口袋数量不足**：71.6% 的测试口袋处于 max_qcov 高位（即结构上有近 precedent），真正 novel 的口袋样本稀少，聚合指标可能被"熟悉口袋"主导。
- **仅评估 monomer 口袋**：interface-only 口袋被屏蔽；未来需扩展到界面口袋预测。
- **Cryptic 定义局限**：crypticity 基于实验 apo-holo RMSD，与模型实际预测 fold 的准确性解耦；研究发现 fold accuracy 比 cryptic label 更能预测性能。
- **预测位点未必是功能位点**：作者认为这在新药发现范式（degraders、molecular glues、radioligands 靶向非正构位点）下影响减弱，但仍需功能验证。
- **Protenix-v2 已转为专有许可**：主实验改用 ESMFold2，限制了对更强 trunk 的持续评估。

## 研究启发与可借鉴点
1. **"Frozen trunk + 轻量 head"范式可迁移**：对任何预训练结构/序列模型，缓存 frozen trunk 的 pair 表征并用简单 pooling + pair-mixing head 下游任务，是一种高效且通用的特征提取策略。
2. **pair 表征中的 pocket 信号强于 single 表征**：Protenix-v2 的 s-feature head（0.474）远逊于 pooled z head（0.669），说明 pair 关系蕴含更丰富的结合位点信息，可启发其他基于 pair 特征的下游任务设计。
3. **时序拆分（temporal split）作为防泄漏基准**：以结构沉积日期而非序列相似性划分 train/test，能有效区分泛化与记忆；这一思路可推广至其他结构生物学 ML 任务。
4. **Re-ranking 作为即插即用后处理模块**：zPocket head 可作为通用 rescorer 显著提升 fpocket、P2Rank、VN-EGNN 等传统/学习方法的 centroid 召回，证明"检测+重排"分工范式的实用价值。
5. **多 trunk ensemble 无需调参即可稳定提升**：对不同 co-folding model 的 head logits 等权平均可带来 +0.02~+0.03 AUPRC，且每增加一个 trunk 边际收益递减，提示可探索更智能的 ensemble 策略。

## 关键术语表
- **Pocket（口袋）**：蛋白表面可与小分子配体结合的凹陷区域，通常由接触残基定义。
- **Co-folding model（共折叠模型）**：联合折叠蛋白与配体/核酸的深度学习模型（如 AlphaFold3、ESMFold2、Protenix），其 trunk 内部分层表征包含丰富的相互作用信号。
- **Trunk / Head 架构**：Trunk 指预训练的骨干网络（此处为冻结的 co-folding 模型），Head 指附加在 trunk 之上的轻量下游预测层。
- **Pair 表征 z**：co-folding trunk 输出的残基间成对关系张量 $z \in \mathbb{R}^{L \times L \times c_z}$，编码残基对间的几何与进化关联。
- **Cryptic pocket（隐性口袋）**：在 apo（无配体）实验结构中闭合、仅在 holo 结构中开放的结合位点。
- **max_qcov**：PLINDER 风格的结构相似性度量，衡量 Held-out 口袋残基被 pre-cutoff 已知口袋覆盖的最大比例。
- **DCC_{N+2}**：Top-(N+2) 召回率，预测的 N+2 个 centroid 中有多少落在真实口袋 4 Å 以内。
- **Temporal split（时序拆分）**：按结构沉积日期而非序列相似度划分 train/val/test，用于防止训练-测试泄漏。

## 可复现要素
- **数据集**：PocketWatch，公开于 HuggingFace（https://huggingface.co/datasets/genesisml/PocketWatch）。
- **代码**：开源于 GitHub（https://github.com/genesistherapeutics/zpocket）。
- **模型权重**：开源于 HuggingFace（https://huggingface.co/genesisml/zPocket）。
- **关键超参**：head 训练 20 epochs（single head 40 epochs），Adam LR=2×10⁻³，batch 等效 8 protein/step；聚类解码 τ=0.3、bandwidth=14 Å、m=3；Loss w=1, γ=0。
- **Trunk 推理设置**：Protenix-v2 10 recycling cycles；ESMFold2 num_loops=11；MSAs 从共享缓存读取，无 templates，固定 seed。
