---
title: "Readout-Scaling-Shapes-Sparse-Feature-Recovery"
source: https://arxiv.org/pdf/2610.18462v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:32:36"
field: "语言模型可解释性与稀疏表征学习"
keywords: ["sparse autoencoder", "readout scaling", "RC-DS", "feature recovery", "model interpretability", "representation learning", "reconstruction vs usefulness"]
innovations: ["提出RC-DS坐标去除读出奇异值增益同时保留决策信息，隔离训练几何效应", "证明生成几何决定最优训练坐标，相同条件数下可翻转RC-DS与logits的优劣", "通过精确wrapper与双空间评估揭示重建增益不等同于标签检测或steering有用性"]
benchmarks: ["OpenSpiel Bargaining/Goofspiel policy reconstruction", "Qwen2.5/Qwen3/SmolLM2 Emotion AG News IMDb label detection", "Synthetic factor recovery with condition number sweep"]
---

# 论文速读：Readout Scaling Shapes Sparse Feature Recovery

## 一句话总结
该论文研究了稀疏自编码器（SAE）在神经网络线性读出附近的特征恢复特性，提出**RC-DS（readout-canonicalized decision-space）坐标**来去除读出矩阵的奇异值增益，同时保留决策信息；证明读出缩放与数据分布共同塑造稀疏学习，并揭示更好的重建精度不等于更好的语义特征实用性。

## 研究问题与动机
1. **核心问题**：神经网络的线性读出层对隐藏方向的不等比例缩放如何影响稀疏特征恢复？读出矩阵的奇异值条件数 alone 是否能预测移除增益的收益？
2. **现有方法不足**：传统 centered-logit SAE 直接在高维词汇表输出空间训练，强方向误差被放大数万倍（如方向100 vs 方向1 → 误差权重比 10000:1），弱放大方向的重建被严重忽视；条件数 alone 无法说明数据在奇异方向上的实际分布与稀疏结构。
3. **信息等价但训练几何不同**：RC-DS 与保留的 centered logits 在同一掩码（mask）下可相互转换，携带完全相同的决策信息；二者比较可隔离"训练几何变化"而非"信息变化"的影响。
4. **重建 ≠ 语义有用性**：更强的 gain-free 重建不一定转化为更好的标签检测或 steering 效果；需要独立的应用测试区分两种能力。

## 核心贡献（创新点）
1. **RC-DS 坐标框架**：通过 SVD 分解中心化的合法动作输出矩阵 $W_c$，定义 $r_0 = UV^\top(h-\mu)$ 去除奇异值增益 $\Sigma$，在保留决策信息的前提下将训练坐标归一化——与 covariance whitening 本质不同，后者基于观测协方差，前者基于读出矩阵本身。
2. **生成几何决定最优坐标**：在相同条件数 $\kappa=100$ 下，改变因子的生成空间会翻转最优训练坐标（$g=1$ 时 RC-DS 最佳 0.8364 vs logit 0.5031；$g=0$ 时反转），证明"移除增益是否有利"取决于生成空间与训练坐标的匹配程度，而非条件数 alone。
3. **精确 wrapper 隔离训练几何效应**：构造 $\hat{y}=TF_\theta(T^{-1}y)$ 可复现 RC-DS 训练至数值精度（最大差异 $1.33\times10^{-15}$），证明 RC-DS 的收益来自编码器输入/输出/损失的联合变换，而非仅损失重加权（损失重加权仅恢复约 0.1443，RC-DS 恢复约 0.3335）。
4. **重建增益与训练预算的关系刻画**：在小标签头实验中，延长训练从 60 到 1000  epochs 使 RC-DS vs logits 的 logit EV 优势从 0.0198 降至 0.0016；ZCA 在充分训练后超越两者，说明固定步数比较不能判定收敛排序。
5. **重构优势 ≠ 标签/steering 优势**：在所有 18 个 full-vocabulary 设置中，原始奇异坐标（raw probes）的 AUROC 均高于训练后的 RC-DS 特征；仅 Qwen Emotion 的受控 steering 试验显示学成方向超过最强控制 +7.73%，但不自动迁移到其他任务或分布。

## 方法详解
**RC-DS 坐标构造**：
- 设合法动作集合 $L(x)$，选择矩阵 $S_L$，中心化矩阵 $C_L = I - |L|^{-1}\mathbf{1}\mathbf{1}^\top$。
- 中心化输出矩阵 $W_c = C_L S_L W$，其 SVD 为 $W_c^{(\rho)} = U\Sigma V^\top$，保留 $\rho$ 个大于数值容差的奇异值。
- RC-DS 坐标：$r_0(x) = UV^\top(h(x)-\mu)$，其中 $\mu$ 为训练集上的隐藏状态均值。
- 一般化：$r_\alpha(x) = U\Sigma^\alpha V^\top(h(x)-\mu)$，$\alpha=0$ 为 RC-DS，$\alpha=1$ 为 centered logits，$\alpha=1/2$ 为半移除。

**训练控制与评估**：
- 使用 Top-K 编码器与单位长度 decoder atoms：$f(s) = \text{TopK}_K(\text{ReLU}(W_e(s-b_d)+b_e))$，$\hat{s} = Df(s)+b_d$。
- 精确 wrapper：$\hat{y} = T F_\theta(T^{-1}y)$，损失 $\mathcal{L}=\|T^{-1}(y-\hat{y})\|_2^2=\|r-F_\theta(r)\|_2^2$，保持参数与优化器状态一致。
- 两种评估空间：(1) **logit EV**：还原为中心化 logits 后计算 $1-\frac{\sum\|\ell_i-\hat{\ell}_i\|^2}{\sum\|\ell_i-\bar{\ell}\|^2}$；(2) **gain-free EV**：在去除缩放后的坐标中同样公式，衡量对未缩放表示的保真度。
- 因子恢复（合成数据）：将 learned directions 映射回生成坐标，按 sign-preserving 一对一匹配后平均 cosine similarity。

**条件数影响**：
- 当因子在 scaling 前生成（$g=1$），随 $\kappa$ 从 1 增至 100，logit training 恢复从 0.9134 降至 0.5063，而 RC-DS 始终保持 0.9134（因变换可逆，输入恒定）。
- 添加 pre-gain noise 时，$\kappa=100$ 下 RC-DS 比 logit 恢复高 0.4034（paired 95% CI [0.3627, 0.4445]）。

## 实验与结果
**合成数据（因子恢复）**：
| 实验 | 条件 | RC-DS | Centered logit | 提升 |
|------|------|--------|----------------|------|
| 因子在 scaling 前生成, $\kappa=100$ | 60 epochs, 20 seeds | 0.9134 | 0.5063 | +0.4070 |
| 坐标匹配（$t=g$） | 40 epochs, 12 seeds | 0.8364 | 0.8364（对角等价） | 构造验证 |
| 仅损失重加权 vs RC-DS | $\kappa=100$, 40 epochs | 0.8176 | 0.6284（loss only）/ 0.4841（MSE） | +0.3335 |

**Policy 重建（OpenSpiel）**：
- **Bargaining**：mean logit EV 从 0.8802 升至 0.9144（+0.0342），所有 3 个 policy 均为正。
- **Goofspiel**：mean logit EV 从 0.7654 升至 0.8637（+0.0983），3 个 policy 中 2 个为正（seed 1120 为 -0.0045）。

**Full-vocabulary 语言模型（60 epochs, 6000 updates）**：
| 模型 | 条件数 | Gain-free EV Δ (RC-DS vs logits) | Logit EV Δ |
|------|--------|-----------------------------------|------------|
| Qwen2.5-0.5B | 22.6 | +0.0216 ± 0.0008 | -0.0058 ± 0.0006 |
| Qwen3-0.6B-Base | 14.4 | +0.0090 ± 0.0012 | -0.0033 ± 0.0010 |
| SmolLM2-360M | 79.4 | +0.0566 ± 0.0025 | -0.0272 ± 0.0011 |

- 在 18 个 task×sparsity 设置中，RC-DS 的 gain-free EV 在全部 18 个为正；logit EV 仅在 1 个为负（其余 17 个 logits 胜出）。
- 最佳案例：SmolLM2 Emotion K=8，gain-free EV 从 0.5428 升至 0.6397（+0.0969）。

**更长训练（Qwen label head）**：
- 60 → 1000 epochs：RC-DS vs logits 的 logit EV 优势从 0.0198 降至 0.0016；ZCA 达到 0.9690，超越两者的 0.9655 / 0.9639。
- 新 prompt 上 K=4：Qwen2.5 diff +0.00058 [−0.00011, +0.00128]；Qwen3 diff +0.00008 [−0.00106, +0.00141]（均跨越零）。

**标签检测与 Steering**：
- Raw singular coordinates 在所有 18 个设置中的 AUROC 均高于训练 RC-DS（差距 0.0002–0.1838）。
- Qwen Emotion controlled steering：学成方向超过最强控制 +7.73% [4.37%, 11.18%]（4/5 seeds >5%）；但更长训练的 RC-DS vs logits 对比为 −4.08% [−10.73%, +0.82%]。
- Bargaining/Goofspiel steering 未通过确认阈值；IMDb/Generated-state steering 也未通过。

## 相关工作脉络
1. **SAE 基础与架构改进**：Cunningham et al. (2023), Bricken et al. (2023), Gao et al. (2024) 研究特征发现；Rajamanoharan et al. (2024a,b) 的 Gated/JumpReLU SAE 分离特征选择与幅度估计以减少 shrinkage——本文固定 SAE 架构，关注输入坐标而非结构创新。
2. **SAE 基准与批判**：Karvonen et al. (2025) SAEBench、Kantamneni et al. (2025) sparse probing、Leask et al. (2025) 对 canonical features 的批判、Korznikov et al. (2026) random-baseline sanity check——本文通过 Raw probe 与 Exact wrapper 控制响应这些质疑。
3. **Logit lens / Tuned lens**：nostalgebraist (2020), Belrose et al. (2023) 用输出读出来解释中间层——本文直接从模型真实线性读出出发，改变坐标而非学习逐层翻译器。
4. **Whitening 与输入缩放**：Saraswatula & Klindt (2025) PCA/ZCA whitening 报告可解释性增益但存在重建 trade-off——本文证明 RC-DS 基于读出矩阵，不同于基于观测协方差（含数据分布）的 whitening，且两者在不同目标下各有优势。
5. **优化坐标敏感性**：Adam 的坐标 wise 更新依赖参数空间旋转（Kingma & Ba 2015; Zhang et al. 2025）——本文承认 Adam 在正交变换下的行为差异是 hidden-state vs RC-DS 小差异的一个来源，但通过 Exact wrapper 将几何效应与信息等价性分开。
6. **几何预测与 steering side effects**：Zaher et al. (2026) 用几何预测 layerwise SAE scaling laws；Duan (2026) 用统计预测 steering side effects——本文预测 RC-DS vs logits 的 paired recovery 差方向，而非单个特征的 intervention 效应。

## 局限性与未来方向
1. **模型覆盖有限**：仅测试 3 个固定 checkpoint（Qwen2.5, Qwen3, SmolLM2）与 2 个游戏环境（Bargaining, Goofspiel），不是随机模型调查；结果不直接外推至其他架构或更大模型。
2. **训练预算未收敛**：full-vocabulary 比较使用固定 6000 updates；small-label-head 延长至 1000 epochs 后仍有 31/180 fits 未达 loss-stability 阈值，无法断言收敛排序。
3. **因子恢复仅适用于合成数据**：真实模型无已知生成因子，语义有用性（label detection / steering）与重建增益之间无可靠对应。
4. **SVD 截断影响**：保留 $\rho$ 个奇异值会丢弃 nullspace 中的隐藏方向；Appendix A.4 讨论截断误差界，但不同 mask 的保留秩可能不同。
5. **预测器表现有限**：fitting-data geometry 预测器在所有 36 对 forecast 中方向正确，但 MAE=0.0526 未达 0.05 目标；且 ZCA 的 support F1 在 34/36 对中优于 RC-DS，说明 RC-DS 并非最优预处理。
6. **Steering 泛化不足**：Qwen Emotion 的受控 steering 通过确认，但游戏与生成任务的 steering 均未通过，说明学成方向的有用性高度依赖目标与状态分布。

## 研究启发与可借鉴点
1. **信息等价对照设计**：RC-DS vs centered logits 的比较是一个"控制变量"典范——二者携带完全相同的决策信息，唯一差异是训练几何，这让因果归因更干净。团队可在自己的 SAE 研究中设计类似的信息等价对照组。
2. **双空间评估必须**：同时报告 logit EV 与 gain-free EV 能揭示权衡（preserving strongly amplified outputs vs preserving unscaled representation）。建议后续工作也采用这种双指标，避免单一 metric 误导结论。
3. **Exact wrapper 作为数值验证**：通过 $\hat{y}=TF_\theta(T^{-1}y)$ 复现训练至 $10^{-15}$ 精度，可将"坐标变换"与"额外信息"彻底分离。这是可复用的验证模板，适用于任何涉及坐标变换的表示学习工作。
4. **重建 ≠ 应用有用的警示**：Raw probe 在所有 18 个设置中胜出，说明 SAE 的 sparse training 不必然产生更好的下游特征。团队在做 interpretability/steering 前应加入 raw-coordinate 与 direct-control 基线。
5. **训练预算敏感性分析**：从 60 到 1000 epochs 的比较显示 ZCA 最终超越 RC-DS，提醒固定步数结论不可靠。建议报告 learning curves 与 late-change diagnostic，并在讨论中明确预算边界。

## 关键术语表
- **RC-DS（readout-canonicalized decision-space）**：一种基于中心化输出矩阵 SVD 的坐标变换，去除读出奇异值增益 $\Sigma$ 而保留旋转 $UV^\top$，使训练对强/弱放大方向平等对待。
- **Condition number $\kappa$**：读出矩阵最大与最小保留奇异值之比（$\sigma_1/\sigma_\rho$）；$\kappa=1$ 表示均匀缩放，$\kappa$ 大表示方向间权重悬殊。
- **Gain-free EV vs Logit EV**：前者在去缩放坐标中计算 explained variance（平等加权），后者在原始 logits 中计算（按 $\sigma_j^2$ 加权），二者揭示不同的重建目标。
- **Exact wrapper**：对 scaled 输入 $y$ 执行 $T^{-1}y \to F_\theta \to T\hat{r}$ 并评估 $\|T^{-1}(y-T\hat{r})\|^2$，在相同参数与优化器下复现 RC-DS 训练轨迹。
- **Support F1**：将 SAE 的 active features 与 known generating factors 做一对一匹配后，基于激活支持计算的 F1 score，衡量"是否激活了正确的因子"。
- **Raw singular coordinate probe**：直接使用 RC-DS 或 logit 空间的单个坐标轴作为二分类器，不经 SAE 稀疏学习；在本实验中持续优于训练特征。
- **Oracle gain（steering）**：在 per-example KL 预算下，利用已知目标行 label 所能达到的最大概率提升；作为受控 steering 效果的 upper bound。
- **Fixed-mask readout equivalence**：在固定 checkpoint 与合法动作掩码下，RC-DS 与 retained centered logits 可通过仿射变换互转，信息完全等价。

## 可复现要素
- **数据集**：合成数据（16 因子/8 维，公开构造细节）；OpenSpiel  bargaining/Goofspiel 策略（需自行训练或从作者内部记录获取）；Qwen2.5-0.5B / Qwen3-0.6B-Base / SmolLM2-360M 公开 checkpoint；Emotion (CARER), AG News, IMDb 公开文本。
- **代码/权重**：**论文未公开**。Appendix I.4 与 L.6 声明源码与 checkpoint 锁定在内部研究记录中，SHA-256 前缀 `7fcfe2e0d79652b1`，但未提供公共仓库链接。
- **关键超参**：SAE dictionary size 1024–1792；Top-K ∈ {8, 16, 32}；Adam lr ∈ {0.001, 0.003}；update 6000 或 epoch 15–1000；batch 128–512；coherence weight 0.05（bargaining）或 0.001；gradient clip norm 1；relative rank tolerance $10^{-6}$；whitening covariance floor $10^{-6}$。
