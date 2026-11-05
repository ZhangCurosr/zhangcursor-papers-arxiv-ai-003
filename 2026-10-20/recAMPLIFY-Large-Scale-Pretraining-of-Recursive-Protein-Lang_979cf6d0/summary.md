---
title: "recAMPLIFY-Large-Scale-Pretraining-of-Recursive-Protein-Lang"
source: https://arxiv.org/pdf/2610.26747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:23:15"
field: "蛋白质语言模型与计算扩展"
keywords: ["protein language models", "test-time scaling", "recursive transformers", "adaptive computation", "GMR", "protein structure prediction"]
innovations: ["提出recAMPLIFY递归预训练框架，以56%更少参数实现与AMPLIFY-350M竞争的性能", "设计GMR门控混合递归统一暂停与残差门控，实现自适应测试时计算", "揭示递归暂停深度编码序列熵和疏水性等可解释生物信号"]
benchmarks: ["CASP16 contact recovery", "ProteinGym DMS", "KL calibration", "Fold prediction", "Localization prediction", "Stability prediction", "Secondary structure prediction"]
---

# 论文速读：recAMPLIFY-Large-Scale-Pretraining-of-Recursive-Protein-Lang

## 一句话总结
论文提出 recAMPLIFY，一个递归蛋白质语言模型，通过权重绑定的 Transformer 核心迭代精炼残基对齐的隐状态，实现了测试时计算扩展，以 155M 唯一参数（比 AMPLIFY-350M 少 56%）在 2T tokens 上预训练后，在 CASP 接触恢复和折叠准确率上显著提升（+22.5%），并引入门控混合递归（GMR）框架实现自适应计算深度。

## 研究问题与动机
1. **参数扩展的瓶颈**：蛋白质语言模型（pLMs）的性能提升依赖参数和数据规模，但会导致训练昂贵和记忆化问题，AMPLIFY-350M 仅用 350M 参数即可匹敌 ESM-2 15B。
2. **现有测试时扩展方法不适用**：自回归 LLM 的测试时计算扩展（如 CoT、反复采样）无法直接迁移到屏蔽编码器，因为氨基酸序列无法言语化，且插入推理 token 会破坏残基到状态的对齐。
3. **固定深度递归的局限**：不同任务/实验的最佳递归深度不同，无单一最优深度，固定深度在超出训练深度后会退化（如稳定性预测从 0.644 降至 0.465）。
4. **需要输入依赖的自适应计算**：蛋白质多样性高（从短信号肽到多结构域酶），需要模型能自适应分配计算预算，决定何时停止更新以及如何更新隐状态。

## 核心贡献（创新点）
1. **递归预训练框架用于测试时计算扩展**：首次将递归预训练适配到屏蔽 pLMs，分离有效模型深度与唯一参数计数，使计算预算在训练后可调整。
2. **recAMPLIFY 高效预训练**：155M 参数模型在相同 2T tokens 上预训练，比 AMPLIFY-350M 少 56% 参数，折叠准确率提升 22.5%（56.8%→69.6%），定位准确率提升 3.5%。
3. **门控混合递归（GMR）框架**：统一了暂停机制和门控机制，从递归轨迹的概率加权隐状态混合解码，并通过学习残差门控控制每个循环更新的接受量，三种门控分辨率（标量、低秩、逐维）。
4. **机制分析揭示生物信号**：残差门控控制训练深度之外的递归稳定性，暂停深度跟踪序列熵和疏水性等生物属性，为自适应计算提供可解释性。

## 方法详解
**recAMPLIFY 架构**：
- **前奏层**（prelude）：前 p 层 Transformer 编码器将 token embeddings 映射到残基对齐的隐状态空间 $\mathbf{x} \in \mathbb{R}^{T \times D}$。
- **递归核心**：r 层权重绑定的 Transformer 核心迭代精炼单个状态 $\mathbf{z} \in \mathbb{R}^{T \times D}$，经过 L 个循环。
- **后奏层**（coda）：c 层映射精炼后的状态到 MLM 输出空间。
- 状态更新公式：$\mathbf{z}_k = f_r(\mathbf{z}_{k-1}, \mathbf{x})$，每循环重新注入输入 x 以保证稳定性。
- 初始状态：$\mathbf{z}_{init} = \mathbf{1}_T \cdot \mathbf{u}^\top$，其中 $\mathbf{u} \sim \mathcal{N}(0, \sigma^2 I)$，$\sigma = 0.02$ 且冻结。
- 有效深度：$\mathcal{D}_{eff} = p + (r \cdot L) + c$，L 作为测试时可变计算值暴露。

**Gated Mixture Recurrence (GMR)**：
- **暂停机制**：halt head 读取状态发出条件暂停概率 $\lambda_k = \text{sigmoid}(\text{HaltHead}(\text{sg}(\tilde{\mathbf{z}}_k)))$，停止梯度流以防止直接操控 LM 特征。
- 无条件暂停质量：$p_k = \lambda_k \prod_{j<k}(1-\lambda_j)$，解码从概率加权混合 $\bar{\mathbf{z}} = \sum_k p_k \mathbf{z}_k$。
- 训练损失：$\mathcal{L} = \mathcal{L}_{MLM}(\text{LMHead}(f_c(\bar{\mathbf{z}}))) - H(\{p_k\})$，熵正则鼓励更多循环。
- **残差门控**：$g_k^{ch} = \text{sigmoid}(\text{GateHead}(\text{sg}(\tilde{\mathbf{z}}_k)))$ 产生通道级存活值。
- 门控状态更新：$\mathbf{z}_k = \mathbf{z}_{k-1} + S_k^D \odot (\tilde{\mathbf{z}}_k - \mathbf{z}_{k-1})$，其中 $S_k^D = \text{project}(S_k^{ch})$。
- 三种门控分辨率：(a) 标量门（ch=1，广播到所有通道），(b) 低秩门（ch=16，投影到 D=960），(c) 逐维门（ch=D，每通道独立生存轨迹）。

## 实验与结果
**数据集**：UniRef100（一般蛋白质组）、SCOP（结构域多样性）、OAS（抗体序列），混合预训练 1M 步（2T tokens）或 100K 步用于消融实验。

**评估基准**：
- 零样本：KL 校准（T/NT）、ProteinGym DMS、CASP16 接触恢复（P@T）
- 监督任务：折叠准确率、定位准确率、稳定性预测、二级结构预测（SSP）

**主要结果（100K 步，Table 1）**：
| 模型 | CASP P@T | Fold acc | Loc acc | Stability ρ | SSP | 参数 | 有效深度 |
|------|----------|----------|---------|-------------|-----|------|----------|
| AMPLIFY-350M | 0.104 | 0.570 | 0.672 | 0.479 | 0.729 | 354M | 32 |
| fixed-L recAMPLIFY | **0.128** | **0.601** | **0.700** | **0.644** | **0.720** | 133M | 52 |
| + halt-only GMR | 0.117 | 0.637 | 0.714 | 0.650 | 0.729 | 133M | 33 |
| + scalar gated GMR | **0.135** | 0.628 | 0.708 | **0.737** | 0.722 | 133M | 37 |
| + per-dim gated GMR | 0.143 | 0.621 | **0.718** | 0.627 | 0.722 | 133M | 37 |

- recAMPLIFY 固定深度（L=6）在 CASP、折叠、定位、稳定性上均超越 AMPLIFY-350M，仅 SSP 略低。
- GMR 变体将期望测试时间深度从 52 降至 33-37 层应用（减少 32-40%），同时提升 CASP 接触恢复和多数监督任务。
- 变量深度预训练提高对未见深度的鲁棒性，但仍无法找到跨任务单一最优深度。

**机制分析发现**：
- 标量和逐维残差门控驱动状态收敛到近似不动点，步长范数趋近于零。
- 仅逐维门控在测试外深度恢复前馈 Transformer 的混合异质性profile。
- 受控暂停深度编码序列熵（+1.47 Cohen's d）和疏水性分数（+1.45），同一基因的剪接异构体暂停深度更紧密聚集（比随机对紧致约 46%）。

## 相关工作脉络
1. **ESM-2 与 AMPLIFY**：ESM-2 通过大规模参数扩展建立 pLM 基准；AMPLIFY 证明数据策展和架构比参数更重要，350M 参数可匹敌 15B ESM-2，本文以此为直接基线。
2. **递归/循环 Transformer**：Universal Transformers (Dehghani et al., 2019)、Looped Transformers (Giannou et al., 2023) 探索权重绑定递归，本文首次将其适配到屏蔽 pLM 测试时计算扩展。
3. **PonderNet 与自适应计算**：Banino et al. (2021) 提出自监督暂停学习，本文将其扩展为统一门控+暂停的 GMR 框架，解决 pLM 无验证器的难题。
4. **AlphaFold/ESMFold 迭代精炼**：这些模型在结构预测 trunk 中递归精炼表示，但使用全局固定计数且操作于结构表示，本文在序列编码器内预训练递归并研究输入依赖深度。
5. **深度递归机制分析**：Bansal et al. (2022)、Geiping et al. (2025) 研究递归动力学、深度外推和"过度思考"失败模式，本文在此基础上分析残差门控如何稳定递归并编码生物信号。

## 局限性与未来方向
1. **暂停深度变异小**：per-protein expected halt depth E[k] 在 10K UniProt 序列中仅均值附近约 1% 变化，自适应是微调而非大幅调整。
2. **缺乏理论稳定性证明**：状态收敛通过经验观察（步长范数消失、残差衰减）表征，未通过 Jacobian 分析建立递归算子的局部稳定性。
3. **门控为序列级非 token 级**：计算适应跨蛋白质而非序列内，向 per-token 生存扩展是自然方向。
4. **仅验证于 AMPLIFY 数据配方**：未测试到其他规模、架构或模态的泛化性。
5. **标量门控的表征稳定性代价**：低自由度下门控分布坍缩为输入不变轨迹，以存储默认值替代真正输入驱动的递归。

## 研究启发与可借鉴点
1. **权重绑定递归用于测试时计算扩展**：可在其他屏蔽模型（如文本 BERT 类）中复用此框架，将深度作为计算预算变量，探索"递归 thinking"在序列模型中的潜力。
2. **GMR 统一暂停+门控设计**：将 halting 和 gating 统一到单一学习存活信号，适用于任何需要自适应计算的深度网络，可扩展到其他连续 latent space reasoning。
3. **机制分析结合生物可解释性**：将暂停深度与序列属性（熵、疏水性）关联，为递归计算提供生物学解释，可推广到分析模型计算分配的内在逻辑。
4. **变量深度预训练策略**：每 mini-batch 均匀采样 L ∈ {1,...,6} 训练可提升深度外推鲁棒性，适用于递归/循环模型的训练技巧。
5. **输入注入保持递归有界**：每循环重新注入输入 x 防止状态发散，是递归 pLM 稳定训练的关键设计，可借鉴到其他递归架构。

## 关键术语表
**recAMPLIFY**：递归 AMPLIFY 模型，通过权重绑定 Transformer 核心迭代精炼残基对齐隐状态的蛋白质语言模型。
**Gated Mixture Recurrence (GMR)**：门控混合递归框架，从递归轨迹的概率加权隐状态混合解码，并通过残差门控控制各循环更新的接受量。
**测试时计算扩展（Test-time scaling）**：在固定参数下增加推理计算以提升性能，区别于传统的参数扩展。
**有效深度（Effective depth）**：$\mathcal{D}_{eff} = p + (r \cdot L) + c$，计算层应用总数，区分于唯一参数计数。
**残差门控（Residual gate）**：控制每个循环提议更新被接受到状态中的比例的 learned survival signal。
**CASP P@T**：CASP16 接触预测的 precision at T，评估长程接触预测性能。
**ProteinGym DMS**：大规模深度突变扫描基准，包含 217 个 assays 和 2.4M 突变变体。
**KL 校准**：模型预测分布与人类 MSA 共识分布之间的 KL 散度校准评分。

## 可复现要素
- **数据集**：UniRef100、SCOP、OAS，使用 AMPLIFY 公开数据和代码 [Fournier et al., 2024]。
- **代码开源**：github.com/chandar-lab/AMPLIFY/tree/recamplify
- **模型权重**：huggingface.co/chandar-lab/recamplify
- **关键超参**：
  - 隐藏维度 D=960，注意力头数 15，FFN 宽度 2560
  - 架构 (p,r,c)=(2,10,2) 用于 1M 步训练，(2,8,2) 用于 100K 步消融
  - 训练步数：1M（主实验）/ 100K（消融）
  - 优化器：AdamW (β₁=0.9, β₂=0.95)，weight decay 10⁻²，梯度裁剪 1.0
  - 学习率：峰值 10⁻³，cosine schedule，1000 步 warmup，10% 最小 LR
  - 掩码率：0.15，最大序列长度：512
  - ε=0.001（暂停阈值）
  - 初始化 σ=0.02
