---
title: "Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM"
source: https://arxiv.org/pdf/2609.37169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:55:55"
field: "大语言模型训练与缩放"
keywords: ["mid-training", "trajectory merging", "model averaging", "compute scaling", "checkpoint merging", "LLM training"]
innovations: ["提出Trajectory Soup将mid-training compute分配到多条独立轨迹并在两层维度进行checkpoint选择与平均", "局部bias-variance分析证明两层uniform average分别消除轨迹内波动和轨迹间持久差异", "通过控制变量实验隔离trajectory diversity相对于sampling density的贡献"]
benchmarks: ["ARC-Easy/Challenge", "MMLU-Pro", "GSM8K", "HumanEval", "LiveCodeBench", "C-Eval", "IFEval", "BFCL v4"]
---

# 论文速读：Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM

## 一句话总结
论文针对大语言模型mid-training阶段串行训练compute收益饱和甚至退化的问题，提出Trajectory Soup方法，通过将训练预算分配到多条独立优化轨迹并在两层维度（轨迹内+轨迹间）进行checkpoint选择与平均，突破串行训练的scaling天花板，在匹配compute预算下持续提升下游性能。

## 研究问题与动机
1. **Mid-training的scaling wall**：与pretraining不同，增加mid-training串行compute不会持续提升下游性能，如图1a所示，性能先快速提升后饱和，继续训练甚至会退化某些能力，形成实际scaling瓶颈。
2. **单轨迹内的平均存在边际收益递减**：Intra-trajectory averaging可以衰减训练路径上的随机波动，但一旦局部波动被充分压制，进一步增加checkpoint采样密度带来的噪声降低越来越少。
3. **轨迹分配轴未被充分探索**：现有方法要么关注单轨迹内的intra-trajectory merging，要么关注轨迹间的inter-trajectory merging，两者结合以优化compute分配的问题尚未系统研究。
4. **compute分配视角的启发**：受scaling laws启发，作者将mid-training的compute分配从"单条轨迹跑多长"重新框架化为"跑多少条轨迹"，探索轨迹数量作为新的计算分配维度。

## 核心贡献（创新点）
1. **重新定义mid-training compute分配轴**：提出将轨迹数量作为与轨迹长度平行的compute分配变量，在单条轨迹长度饱和后，增加轨迹数量仍能带来性能增益，且不同recipe微扰产生的轨迹在参数空间中几何分离。
2. **提出两层merge的Trajectory Soup框架**：先在每个分支内按validation loss筛选Top-K checkpoints并uniform averaging得到trajectory anchor，再对所有分支anchor进行inter-trajectory uniform averaging；局部bias-variance分析证明两层average分别消除不同类型的残差误差。
3. **理论证明uniform weight的两级方差最优性**：在共享曲率加权协方差条件下，两级的uniform平均（而非rank-based或其他权重方案）是最优的方差最小化选择，且checkpoint选择会带来有界bias从而决定最优K值。
4. **控制变量实验隔离多样性来源**：通过匹配候选池大小和merge checkpoint数量的对照实验，证明性能提升源于独立演化轨迹的互补信息，而非更大的checkpoint池或更高采样密度。

## 方法详解
**问题设定**：从共享预训练checkpoint $\theta_0$ 出发，在相同训练分布$\mathcal{P}$上并行运行$N$条轨迹，每条轨迹使用不同recipe $\psi_n$（通过扰动data shuffle seed、peak learning rate、batch size、lr schedule、optimizer等产生），每条轨迹消耗$t$ tokens，总预算$T = Nt$。每条轨迹保存checkpoint的兼容性筛选条件：相对training loss gap不超过$\varepsilon = 0.01$。

**Step 1 - Intra-Trajectory Merging**：对每条分支$n$，按validation loss排名保留最优的$K$个checkpoint，uniform averaging得到trajectory anchor：
$$\bar{\theta}_n(t, K) = \frac{1}{K} \sum_{i \in \mathcal{I}_n(t,K)} \theta_n(i)$$

**Step 2 - Inter-Trajectory Merging**：对所有$N$条轨迹的anchor进行uniform averaging：
$$\theta_{\text{Traj-Soup}}(N,t,K) = \frac{1}{N} \sum_{n=1}^N \bar{\theta}_n(t,K)$$

**理论分析（局部二次模型）**：假设validation loss在局部近似为$\mathcal{L}_Q(\theta) \doteq \mathcal{L}^\star + \frac{1}{2}\|\theta - \theta^\star\|_H^2$，通过bias-variance分解得到：
$$\mathbb{E}[\mathcal{L}_Q(\theta_{\text{Traj-Soup}})] - \mathcal{L}^\star = B_{\text{soup}}(K) + \frac{V_{\text{intra}}}{NK_{\text{eff}}(K)} + \frac{V_{\text{inter}}}{N}$$
其中$B_{\text{soup}}(K)$为selection-dependent bias，$V_{\text{intra}}/K_{\text{eff}}(K)$为轨迹内波动经平均后的残差，$V_{\text{inter}}/N$为轨迹间持久差异经平均后的残差。该分解说明：内层average仅能减少$V_{\text{intra}}$，而$V_{\text{inter}}$必须通过跨轨迹average消除；同时checkpoint selection带来的bias $B_{\text{soup}}(K)$随$K$增大而增大，因此存在有限的最优$K^*$。

## 实验与结果
**模型与设置**：主实验使用Ling-3.0-Tiny（7.9B总参数，1.3B激活参数的稀疏MoE模型），默认分支数$N=3$（EXP1 Baseline、EXP2 Data Shuffle、EXP3 Learning Rate & Batch Size变化），每条轨迹$t=600B$ tokens，每25B tokens保存checkpoint。

**Mid-training结果（Table 2）**：
- Single-Trajectory Merge最佳整体平均：**68.55**
- Model Soup (Limited)：**68.43**；Model Soup (Extended)：**68.67**
- **Trajectory Soup (Limited)：68.72**；**Trajectory Soup (Extended)：68.96**（+0.41 vs Single-Trajectory）
- 在各子能力类别（General Knowledge & Reasoning、Language Modeling、Professional Knowledge、Math、Code）均有提升或持平。

**Post-training结果（Table 3）**：应用相同SFT流程后，Trajectory Soup优势保持：
- Single-Trajectory Merge整体平均：**61.11**
- Model Soup (Extended)：**61.23**
- **Trajectory Soup (Extended)：61.52**（+0.41 vs Single-Trajectory）

**小模型验证（Table 4）**：在2B参数MoE模型上使用WSD schedule复现，Limited设置下Trajectory Soup达**49.64**，Extended达**50.07**，趋势一致。

**Ablation（Table 5）**：
- Coefficient：Uniform（EQUAL）最优，1SQRT/RANK/RSQRT均不及baseline（68.85~68.89 vs 68.96）
- Selection：Top-K each最优，Global top-NK（68.76）、Tail-K（68.35）、All checkpoints（68.60）均不及
- Balanced allocation：图6显示对称分配（8:8）最优，呈倒U型

**Scaling analysis（Fig. 8）**：$N$从2增至5，最佳分数从68.79升至69.08；每分支最优$K$约在10~16之间，超出后性能震荡或下降。

**Directional diversity预测gain（Fig. 9）**：pairwise cosine similarity与interpolation gain呈强负相关（Pearson $r=-0.80$, Spearman $\rho=-0.94$），几乎正交的分支对增益最大。

## 相关工作脉络
1. **Stochastic Weight Averaging (SWA)** (Izmailov et al., 2018)：在constant/cyclical lr训练中对iterates做平均以找到更平坦的optima；本文在此基础上扩展到多轨迹场景并引入checkpoint selection。
2. **Model Soups** (Wortsman et al., 2022)：对不同hyperparameter的fine-tuned模型endpoint做平均；本文强调inter-trajectory merging需配合intra-trajectory selection，而非直接使用endpoint。
3. **Extra-Merge** (Zhou et al., 2026)：发现late-stage trajectory近似rank-1结构并用于extrapolation；本文利用该方向多样性分析但采用两层平均而非外推。
4. **Branch-Train-Merge** (Li et al., 2022) 与 **Branch-Train-MiX** (Sukhbaatar et al., 2024)：并行训练domain-specialized expert并ensemble或mix入MoE；本文聚焦于单模型weight-space consolidation而非expert架构。
5. **WSM** (Tian et al., 2026)：通过decay-free lr schedule结合checkpoint merging；本文不改变lr schedule设计，而是探索compute如何分配到多条轨迹。
6. **Scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022)：将compute分配问题扩展至轨迹数量维度，类比pretraining中parameters vs tokens的权衡。

## 局限性与未来方向
1. **Compute accounting不完整**：当前只固定了processed tokens，未计入checkpoint存储、validation ranking开销、搜索最优$(N,K)$的成本，以及不同加速器利用率差异。
2. **理论假设局限**：局部二次模型仅在selected checkpoints所在区域有效；几何诊断（方向角、插值profile）是post-hoc测量，无法在训练前预测哪些recipe perturbation会产生可被average消除的error。
3. **强相关轨迹存在性能floor**：若分支间高度相关则merged模型无法超越成员；perturbation过激会将分支推入合并后更差的参数区域。
4. **超参需validation search确定**：当前$N$、$K$通过validation sweep选定，未来希望实现training过程中自适应allocation。
5. **扩展性待验证**：data mixture变更、更长post-training pipeline中的适用性尚未检验。

## 研究启发与可借鉴点
1. **Compute分配维度的范式转换**：将"轨迹数量"作为与"模型规模""token数"同等的scaling变量，为mid-training/continued pretraining的资源规划提供新视角，可迁移至domain adaptation等场景。
2. **两层merge的结构化设计**：Intra + Inter两层average分别针对不同误差源（时间波动vs持久分支差异），配合简洁的Top-K选择，避免过度调参；该框架可直接复用到pretraining的checkpoint merging pipeline。
3. **Directional diversity作为cheap predictor**：Pairwise cosine similarity与merge gain强负相关（$\rho=-0.94$），可在merge前快速筛选互补分支，降低试错成本。
4. **平衡性优于复杂度**：Ablation表明uniform weight和balanced allocation已接近最优，复杂的rank-based weighting反而不如简单平均，提示后续工作不必过度设计合并系数。
5. **控制变量隔离贡献来源**：通过匹配candidate pool size和merge count的对照实验（Fig. 7）严格证明trajectory diversity的价值，而非归结为sampling density，实验设计严谨可借鉴。

## 关键术语表
**Mid-training**：Pretraining之后、post-training（SFT/RLHF）之前的训练阶段，用于赋予LLM领域专精能力和推理能力。
**Trajectory Soup**：本文提出的方法，将mid-training compute分配到多条独立轨迹，在轨迹内和轨迹间两层进行checkpoint选择与平均，融合为单一模型。
**Intra-trajectory merging**：在单条训练轨迹内，按validation performance筛选并平均多个checkpoint以减少训练路径上的随机波动。
**Inter-trajectory merging**：跨多条独立训练的轨迹，对其anchor（通常是endpoint或 averaged checkpoint）进行平均以融合互补优化方向。
**Compatibility screen**：通过relative training-loss gap（$\varepsilon=0.01$）筛选recipe perturbation后的分支，确保各分支优化质量相当但参数空间方向不同。
**Effective checkpoint count ($K_{\text{eff}}$)**：轨迹内checkpoint平均的有效数量，衡量时间波动衰减效率，受checkpoint间残差相关性制约（$1 \leq K_{\text{eff}} \leq K$）。
**Bias-Variance decomposition**：将expected excess validation loss分解为bias项（mean displacement from optimum）和variance项（fluctuation around mean），用于分析两层average各自消除的误差分量。
**Compute-matched vs Compute-expanded**：Limited setting（固定总budget分配给N条轨迹）用于公平比较；Extended setting（每轨迹完整训练后增加N）用于验证scaling潜力。

## 可复现要素
- **数据集**：论文使用内部mid-training corpus（具体配比未公开），evaluation使用41个benchmark配置（Appendix D.4详列）
- **代码/权重**：基座模型Ling-3.0-Tiny从HuggingFace公开（https://huggingface.co/inclusionAI/Ling-3.0-tiny）；论文附录D标注部分experimental completion fields尚未填写，代码未声明开源
- **关键超参**：$t=600B$ tokens/branch，checkpoint interval=25B tokens，$\varepsilon=0.01$（compatibility screen），默认$N=3$，$K=4$（Limited）或$K$在10~16区间（Extended）；recipe perturbation维度见Table 8
