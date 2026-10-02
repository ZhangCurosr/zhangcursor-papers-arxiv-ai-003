---
title: "SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE"
source: https://arxiv.org/pdf/2609.37842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:56:44"
field: "大语言模型训练数据归因与可解释性"
keywords: ["influence functions", "gradient compression", "LLM attribution", "EK-FAC", "one-bit quantization", "data valuation"]
innovations: ["在最坏情况影响误差下证明半白化梯度的 top-k PCA 为最优线性表示", "提出两阶段 EK-FAC+子空间 PCA 修正逼近理论最优", "用 1-bit 符号量化配合尺度因子在极低存储预算下保留高保真影响排序"]
benchmarks: ["GPT-2 on WikiText-2", "OLMo 2 1B/7B/13B/32B on Tulu 3 SFT mixture"]
---

# 论文速读：SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE

## 一句话总结
本文提出 EOGP（Eigenbasis-corrected One-bit Gradient Projection），通过"EK-FAC 降维 + 子空间内 PCA 修正 + 1-bit 符号量化"三阶段压缩训练梯度，在极小存储预算下（GPT-2 仅用基线 1/16、OLMo 2 1B~32B 不到 16 KB/example）仍能高保真地重放影响函数排序，显著提升可复用梯度表示的存储效率与推断精度。

## 研究问题与动机
- **核心问题**：影响函数（Influence Functions）通过分析训练样本对模型行为的影响来支撑数据归因与选择，但同一模型需要多次不同查询，重复计算训练梯度代价高昂；若能一次性存储训练梯度并复用，则可大幅降本。
- **全量存储不现实**：8B 参数模型单条训练梯度的半精度占用约 16 GB，大规模场景下存储开销不可承受。
- **现有压缩方法的瓶颈**：已有工作（LoGra、GraSS、LoRIF 等）通过随机投影或曲率感知投影压缩梯度，但未充分考虑"未知未来查询"下的最坏情况影响误差，且在极低预算下保真度显著下降。
- **优化目标缺失**：缺少一套在不知道具体查询时的理论可证明的最优线性压缩准则，导致实际系统在压缩方向选择和精度分配上缺乏统一指导。

## 核心贡献（创新点）
- **提出 EOGP 全链路压缩框架**：结合 EK-FAC 降维、子空间 PCA 修正与 1-bit 量化，使每条训练样本的存储远低于现有方法；与 LoGra/GraSS 等基线相比，在 GPT-2 上仅需其 1/16 存储即可超越其性能，在 OLMo 2 1B~32B 上在超 100 倍更大预算下仍具竞争力。
- **给出最优线性压缩的理论与构造**：在归一化最坏情况影响误差准则下，证明存储"半白化（half-whitened）训练梯度的 top-k PCA 坐标"是固定维度线性表示的最优解；这一结论为压缩方向选择提供了明确理论基准。
- **提出两阶段投影（Two-stage projection）以逼近理论最优**：直接在全空间做 PCA 计算代价过高，因此先利用已有的 EK-FAC 特征基构造候选子空间，再在该子空间内对第一阶段的半白化梯度做 PCA 修正，得到更低的重构误差，并给出严格的误差上界（Proposition 2）。
- **引入 1-bit 符号量化与尺度因子以在固定预算下容纳更多坐标**：将投影后坐标存储为每条坐标 1 bit 符号 + 每模块 1 个 FP16 均值尺度，显著扩展可用坐标数；同时提出 EOGP-R 变体以 SRHT 替代密集 PCA 矩阵，提升在大预算下的可扩展性。
- **在 LLM 尺度上进行系统化评估**：在 GPT-2 上与重训练（LDS、反事实重训）真值对齐验证，并在 OLMo 2 1B~32B SFT 模型上与 EK-FAC 参考的一致性（NDCG@20、Spearman）验证，证明压缩梯度仍能有效保留可复用归因信息。

## 方法详解
- **影响函数与半白化梯度定义**：对目标查询 $f$，影响分数为 $\hat{\mathcal{T}}(i)=\nabla_\theta f^\top H_\lambda^{-1} \nabla_\theta \ell_i(\theta^*)$，其中 $H_\lambda=G+\lambda I$ 为阻尼广义 GGN/Fisher；引入半白化梯度 $\tilde{t}_i=H_\lambda^{-1/2}\nabla_\theta \ell_i$ 与 $\tilde{q}=H_\lambda^{-1/2}\nabla_\theta f$，则影响等价于内积 $\langle \tilde{q},\tilde{t}_i\rangle$。
- **最坏情况影响误差准则**：在由 $H_\lambda^{-1}$ 加权范数定义的单位球 $\mathcal{B}$ 上最小化 $\mathcal{E}(V,M)=\mathbb{E}_i[\sup_{\tilde{q}:\|\tilde{q}\|_2\le 1}|\tilde{q}^\top(I-RR^\top)\tilde{t}_i|^2]$，可将其转化为低秩重构问题，证明最优解为存储 $\Sigma=\mathbb{E}[\tilde{t}_i\tilde{t}_i^\top]$ 的 top-k 特征向量方向的坐标（Proposition 1）。
- **第一阶段：EK-FAC 子空间选取**：利用 K-FAC/EK-FAC 的块对角特征基 $Q$ 与校正特征值 $\Lambda$，近似半白化算子与训练梯度二阶矩，得到代理海森 $\Sigma_\mathrm{proxy}=Q\mathrm{diag}(\lambda_j/(\lambda_j+\lambda))Q^\top$；按其单调性，取校正特征值最大的 $m$ 个 EK-FAC 特征向量构成 $Q_m$，并用 $(\Lambda_m+\lambda I)^{-1/2}Q_m^\top$ 对训练梯度做第一阶段坐标化。
- **第二阶段：子空间内 PCA 修正**：对第一阶段坐标 $z_i=L_u g_{i,u}$ 做无中心化 PCA，取其 top-$k$ 右奇异向量构成的矩阵 $P$，最终坐标为 $\tilde{t}_i^{(2)}=P^\top z_i$；该步骤将候选维数 $m>k$ 的信息融合为 k 维最优子空间，且均方重构误差小于等于仅取前 $k$ 个 EK-FAC 轴的误差（Proposition 2）。
- **EOGP-R 变体（大预算场景）**：为规避存储密集 $P\in\mathbb{R}^{m\times k}$ 的开销，使用隐式应用的次采样随机 Hadamard 变换（SRHT）将 $m$ 维第一阶坐标映射至 $k$ 维，保留范数与内积的近似保持性，适合更大 $k$ 的情形。
- **1-bit 量化与查询估计**：对每模块存储符号向量 $b_{i,u}=\mathrm{sign}(\tilde{t}_{i,u}^{(2)})$ 与尺度 $s_{i,u}=\frac{1}{k_u}\|\tilde{t}_{i,u}^{(2)}\|_1$（由最小化 $\|a-sb\|_2^2$ 导出）；查询梯度不经量化，直接投影为 $y_u$，最终影响估计为 $\widehat{\mathcal{T}}(i)=\sum_u s_{i,u}\langle y_u,b_{i,u}\rangle$。
- **实现细节**：使用 Kronfluence 拟合 EK-FAC 统计与 damping（$\lambda_u=0.1\,\mathrm{tr}(\Lambda_u)/d_u$），PCA 拟合集为 10,000 条 SFT 对话；共享的 Kronecker 特征基、半白化权重与 PCA 修正矩阵跨样本与查询复用，修正矩阵在 32B 模型上每模块约 1 GB，随存储规模可扩展摊销。

## 实验与结果
- **GPT-2 上的重训练验证**：在 WikiText-2 上比较 12/24/96/384/1536 KB 多种预算；EOGP 在 96 KB 即超越所有基线在 1,536 KB 上的 LDS，且反事实移除所选样本造成的验证 perplexity 上升最大；在 384 KB 时 LDS 接近未压缩 K-FAC，而存储仅为未压缩 162 MB 的约 1/400。
- **OLMo 2 SFT 上与大模型一致性评估**：在 1B/7B/13B/32B 模型上使用 Tulu 3 SFT 数据与留持查询，以 NDCG@20 和 Spearman 相关衡量与 EK-FAC 参考的一致性；EOGP 在各规模上尤其在小预算下显著优于 LoGra、GraSS、LoRIF，压缩比达 $10^4$–$10^6$ 级别。
- **保真与重训练质量的关联**：GPT-2 上 EOGP 各配置的 NDCG@20 与 LDS 的 Spearman 相关约 0.95、Spearman 相关约 0.84，说明基于 EK-FAC 排名的保真度能有效预测实际重训练效果。
- **消融与对比**：固定存储预算下 EOGP 始终优于直接 EK-FAC 轴选择；相同坐标数下 EOGP 在 FP16 与 1-bit 均胜过所有基线；1-bit 对 EOGP 的影响较小（FP16 与 1-bit Pearson 中位相关在 GPT-2 达 0.89–0.92），但对部分基线反而显著劣化。
- **存储与构建成本**：EOGP 在单个 NVIDIA B200 上每条示例构建时间与 LoGra 相当（32B 模型约 0.247s vs 0.278s）；PCA 修正矩阵的一次性拟合开销在 32B 上为 18.4 分钟；共享矩阵与批量查询处理可在大规模场景下摊薄。

## 相关工作脉络
- **Influence estimation via curvature approximations**：K-FAC/EK-FAC（Martens & Grosse, 2015; George et al., 2018; Grosse et al., 2023）用结构化曲率近似支持 LLM 规模的影响分析；本文在此基础上进一步将曲率估计与梯度压缩联合优化。
- **TRAK/TracIn/TrackStar 等可伸缩归因方法**：TRAK（Park et al., 2023）与 TrackStar（Chang et al., 2025）分别通过随机投影与曲率校正梯度提升可扩展性；本文聚焦"存储可复用训练梯度"而非单次快速估计，目标更偏长期复用。
- **LoGra / GraSS / LoRIF 等压缩基线**：LoGra 用 Kronecker 投影（随机/PCA 初始化），GraSS 用稀疏投影与 CountSketch，LoRIF 用低秩因子与截断 SVD 近似逆曲率；本文在同等或更小预算下提供更优保真，并给出理论支撑。
- **LESS/QLESS 等数据选择方法**：基于梯度相似度而非逆曲率影响分数进行指令数据选择；本文方法与这些方法属于不同设计范式，可作为互补的底层归因能力被集成。
- **Grosse et al. 2023（Kronfluence）在 LLM 泛化中的应用**：展示影响函数在大模型通用性研究中的潜力；本文回应其提出的存储复用挑战，使跨查询分析在经济上更可行。

## 局限性与未来方向
- **理论最优建立在固定维度线性表示假设上**：实际中非线性近似或自适应查询依赖表示可能进一步降低误差，但不在本文框架内。
- **1-bit 量化在高维度/高方差场景存在失真风险**：虽在 EOGP 半白化表示下表现稳定，但对其他压缩方向或不同损失函数的泛化仍需验证。
- **PCA 修正矩阵的共享存储成本随 $m_u,k_u$ 线性增长**：在小样本早期建库阶段相对占比高，需结合摊销策略或低秩近似进一步压缩。
- **仅针对注意力与 MLP 线性层进行归因**：未包含 embedding、LM head 与归一化层参数，可能遗漏部分重要归因信号。
- **未来可扩展至更多下游任务与动态增量存储**：当前实验集中于 SFT 场景；如何在持续训练、增量数据与多任务微调中高效更新表示是开放问题。

## 研究启发与可借鉴点
- **理论驱动的压缩目标设定**：将"未知未来查询下的最坏情况影响误差"作为统一目标，为后续梯度压缩方法提供可复用的评测与设计基准，值得在更多归因任务中推广。
- **两阶段投影思想可迁移**：先用结构化曲率/特征基构造候选子空间，再在子空间内做精确降维（PCA/正交化），这一"粗筛+精修"范式可移植到其它需要长期复用的表征压缩任务。
- **1-bit 符号量化配合尺度因子在受控表示上可保留高相关性**：对半白化后的梯度坐标，符号+均值的极简量化几乎不损失排序质量；这种"先预处理再硬量化"的策略可在其它模型规模与任务中尝试。
- **SRHT 变体为大预算场景提供可扩展替代**：当 $k$ 较大时，用随机结构化变换替代密集 PCA 矩阵能在保持低失真前提下显著降低存储与计算压力，是可复用的工程技巧。
- **共享修正矩阵的摊销分析有助于评估长期部署成本**：论文对 1 亿/10 亿样本情形下修正矩阵占比的分析方法，可为团队在建库策略、预算规划时提供参考框架。

## 关键术语表
- **影响函数（Influence Functions）**：估计移除或修改某训练样本后模型预测行为的近似变化，用于训练数据归因。
- **半白化梯度（Half-whitened gradient）**：用阻尼逆曲率矩阵平方根对训练梯度做线性变换，使影响分数等价于简单内积。
- **EK-FAC**：在 K-FAC 的 Kronecker 特征基上校正特征值的 Fisher/GGN 近似，兼顾结构效率与谱精度。
- **两阶段投影（Two-stage projection）**：先在 EK-FAC 特征空间中选取候选子空间，再在该子空间内做 PCA 得到最终压缩方向。
- **EOGP-R**：以隐式 SRHT 替代密集 PCA 修正矩阵的变体，适合较大输出维度以节省存储与计算。
- **1-bit 量化与尺度因子**：将投影坐标仅存符号位，并用每模块一个均值尺度恢复幅度，实现在固定字节预算下容纳更多坐标。
- **LDS（Linear Data Modeling Score）**：通过影响估计预测随机子集重训练效果的 Spearman 相关，作为影响估计质量的重训练基准。
- **NDCG@20 / Spearman 相关**：在 LLM 尺度实验中用于衡量压缩影响排序与 EK-FAC 参考排序一致性的主要指标。

## 可复现要素
- **数据集**：GPT-2 使用 WikiText-2（wikitext-2-raw-v1），OLMo 2 使用 Tulu 3 SFT mixture（allenai/tulu-3-sft-olmo-2-mixture 系列），均为公开数据。
- **代码/权重**：论文使用开源模型 checkpoint（gpt2、allenai/OLMo-2-*），引用 Kronfluence/LogIX 等实现；论文未声明独立开源仓库，但提供了算法伪代码与详细超参（damping、维度配置、拟合集合大小等）。
- **关键超参**：相对阻尼 $\lambda_u=0.1\,\mathrm{tr}(\Lambda_u)/d_u$；第一阶候选维度 $m_u=262{,}144$；最终维度 $k_u$ 在 256–8,192 之间（按预算匹配）；PCA 拟合集 10,000 条对话、截断至 2,048 token；训练使用 AdamW、lr $3\times10^{-5}$、weight decay 0.01、3 epochs、batch size 8。
