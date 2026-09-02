---
title: "When-Features-Become-Instances-Inverted-Contrastive-Learning"
source: https://arxiv.org/pdf/2609.00782v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:21:24"
field: "无监督特征选择"
keywords: ["Unsupervised Feature Selection", "Contrastive Learning", "Representation Learning", "Self-Supervised Learning", "Feature Ranking"]
innovations: ["将UFS重构为特征级对比学习问题，转置数据矩阵使特征成为实例", "利用InfoNCE诱导的projector嵌入范数作为特征显著性信号进行排序", "结构化多视图扰动+decorrelation正则+LGRC后处理的三级冗余控制框架"]
benchmarks: ["ARCENE", "BASEHOCK", "COIL20", "GISETTE", "LUNG", "NCI9", "PCMAC", "PROSTATE", "RELATHE", "TOX171", "WARPPIE10P", "ALLAML"]
---

# 论文速读：When-Features-Become-Instances-Inverted-Contrastive-Learning

## 一句话总结
论文提出ICLFS（Inverted Contrastive Learning for Unsupervised Feature Selection），将无监督特征选择重构为**特征级别的对比表示学习问题**：通过转置数据矩阵使每个特征成为学习实例，构建多视图masked正样本与shuffled负样本，利用InfoNCE损失学习特征级表征一致性，并以projector空间嵌入范数作为特征显著性信号完成排名。

## 研究问题与动机
- **无监督特征选择缺乏标签信号**：传统UFS依赖间接结构准则（如相似度保持、局部邻域、聚类几何或重建质量），难以直接衡量特征效用。
- **现有方法局限于代理目标**：经典方法（LS、SPEC、MCFS、NDFS）依赖预定义相似度图或谱结构；深度方法（CAE、LS-CAE）依赖重建目标，可能保留对输入恢复有用但非判别性的特征。
- **对比学习视角缺失**：现有对比学习方法均以样本为实例，极少将数据矩阵转置后以特征为学习对象，导致对比优化无法直接用于特征排序。
- **嵌入范数作为显著性信号的潜力**：近期研究表明InfoNCE训练会系统性增大嵌入范数，且范数与模型置信度/典型性相关，可作为特征重要性替代信号。

## 核心贡献（创新点）
- **特征级对比学习范式转换**：首次将UFS重构为"特征即实例"的对比表示学习问题，通过转置数据矩阵使每个特征获得独立对比优化。
- **项目空间嵌入范数作为特征显著性信号**：利用InfoNCE训练诱导的嵌入范数增长特性，以projector输出未归一化向量的ℓ₂范数直接排名特征。
- **结构化多视图扰动设计**：构造4个masked正视图（light/heavy/complementary pair）和1个shuffled负视图，使特征在部分遮挡下保持身份一致性。
- **两级冗余控制机制**：训练时引入inter-feature decorrelation regularizer鼓励特征占据不同嵌入区域；推理时通过Laplacian-Gated Ranking Correction（LGRC）后处理剔除局部冗余候选。
- **广泛的基准验证**：在12个benchmark数据集上对比经典与神经网络基线，在10个数据集取得最优聚类准确率，最强提升达27.89%（PROSTATE）。

## 方法详解
**整体架构分为四个组件：**

1. **预处理与特征级转置**：输入X∈ℝⁿˣᵈ标准化后转置为Xᵀ∈ℝᵈˣⁿ，每行xⱼ∈ℝⁿ为特征j的样本profile向量，成为后续对比学习的"实例"。

2. **对比任务视图构造**：
   - 4个masked正视图：light-mask（保留率0.90）、heavy-mask（保留率0.60）、complementary mask pair（保留率0.50，重叠率0.10）
   - 1个shuffled负视图：随机打乱样本顺序，破坏样本对齐结构
   - 堆叠为M=[m₁,m₂,m₃,m₄,m⁻]∈ℝ⁵ˣᵈˣⁿ

3. **神经网络模型**：
   - **Encoder**：单层self-attention（捕捉特征间交互）→ 两层MLP（线性层+BatchNorm+LeakyReLU+Dropout）→ 输出H⁽ᵛ⁾∈ℝᵈˣᵈʰ
   - **Projector**：残差MLP（两层投影块+线性输出层）→ 输出Z⁽ᵛ⁾∈ℝᵈˣᵈᶻ
   - 根据样本量采用两档架构：小样本（dₑ=16,dₕ=512,dₚ=128,d_z=16）/ 大样本（dₑ=64,dₕ=1440,dₚ=2048,d_z=32）

4. **优化目标**：
   - **InfoNCE对比损失**：对每个mask视图m∈{1,...,4}：
     $$\mathcal{L}_{\text{con}}^{(m)} = -\frac{1}{d}\sum_{j=1}^{d}\log\frac{\exp(\langle\hat{z}_j^{(a)},\hat{z}_j^{(m)}\rangle/\tau)}{\sum_{\ell=1}^{d}\exp(\langle\hat{z}_j^{(a)},\hat{z}_\ell^{(m)}\rangle/\tau) + \exp(\langle\hat{z}_j^{(a)},\hat{z}_j^{(-)}\rangle/\tau)}$$
     其中$\hat{z}$为行-wise ℓ₂归一化，τ=0.05为温度参数。
   - ** decorrelation正则项**：
     $$\mathcal{L}_{\text{decorr}} = \frac{1}{d^2}\|\mathbf{C}^{(a)}-\mathbf{I}_d\|_F^2, \quad \mathbf{C}^{(a)}=\hat{\mathbf{Z}}^{(a)}\hat{\mathbf{Z}}^{(a)\top}$$
     抑制不同特征在projector空间的余弦相似性。
   - **总损失**：$\mathcal{L}=\sum_{m=1}^{4}\left(\frac{1}{4}\mathcal{L}_{\text{con}}^{(m)}+\lambda_{\text{decorr}}\mathcal{L}_{\text{decorr}}\right)$，小样本λ=0.20，大样本λ=0.40。

5. **特征选择与排序修正**：
   - **Projector-norm ranking**：计算sⱼ=‖zⱼ⁽ᵃ⁾‖₂，降序排列得初始排序R̂。
   - **LGRC后处理**：对目标基数k，从R̂顶部取候选池Pₖ（大小p=round(αk)，α=1.5），计算各特征Laplacian score ℓⱼ，以q=0.75分位数为阈值γ_q，迭代剔除池中Laplacian score超过阈值的特征并替换，最终返回Top-k子集。

## 实验与结果
**数据集**：12个benchmark数据集（scikit-feature仓库），涵盖图像、生物医学、文本、质谱领域，样本数60-7000，特征数1024-10000。

**评估协议**：标准聚类型UFS评估——选定不同特征基数a∈{50,100,150,200,250,300}后运行k-means聚类20次，报告最高平均准确率（匈牙利匹配最优对齐）。

**基线方法**：LS、MCFS、NDFS、SPEC、CAE、LS-CAE。

**主要结果**：
| 数据集 | ICLFS最佳准确率 | 提升幅度（vs最优基线） | 备注 |
|--------|----------------|----------------------|------|
| PROSTATE | 83.97% (50) | +23.19%~+27.89% | 最大增益 |
| NCI9 | 52.33% (100) | +11.25%~+15.16% | |
| LUNG | 67.66% (300) | +8.40%~+14.24% | |
| WARPPIE10P | 44.14% (50) | +4.57%~+7.69% | |
| GISETTE | 76.42% (250) | +0.44%~+19.91% | |
| TOX171 | 54.77% (100) | +4.57%~+7.69% | |
| COIL20 | 68.68% (150) | +1.85%~+12.37% | |
| PCMAC | 52.25% (50) | +0.94%~+1.64% | |
| BASEHOCK | 51.88% (100) | +0.44%~+1.20% | |
| ARCENE | 65.17% (250) | +0.47%~+2.74% | |
| RELATHE | 55.46% (200) | 并列第二，-0.41%~+1.23% | |
| ALLAML | 66.60% (100) | -4.51%~+12.09% | LS-CAE最优71.11% |

**消融实验关键发现**：
- 移除LGRC：微小下降（0.34%~1.14%），说明核心对比排序已足够鲁棒。
- 移除decorrelation：显著下降（PROSTATE -14.51%，WARPPIE10P -8.54%）。
- 移除attention：ARCENE -8.87%，COIL20 -9.90%，PROSTATE -12.21%。
- 替换为单随机mask：PROSTATE暴跌26.91%，验证结构化多视图必要性。

## 相关工作脉络
- **经典图/谱方法（LS、SPEC、MCFS）**：依赖预定义相似度图或谱准则，本质是局部/全局结构保持，无法显式建模特征间对比关系。
- **判别性/聚类联合学习（NDFS、UDFS）**：联合优化聚类分配与稀疏特征权重，但仍绑定于谱聚类目标，高维下对图质量敏感。
- **重建驱动深度方法（CAE、LS-CAE、DUFS）**：通过可微子集选择+重建目标学习特征子集，保留对输入恢复有用的特征，但可能忽视判别性互补特征。
- **谱自监督特征选择（SSFS）**：结合图谱结构与自监督代理学习，属过渡性方法，仍非真正的特征级对比范式。
- **ICLFS定位**：与上述方法本质区别在于——不以重建或谱图为代理，而是直接将特征作为对比实例，通过多视图一致性学习特征显著性。

## 局限性与未来方向
- **小样本场景可能欠拟合**：虽然采用双档架构适配不同样本量，但特征数d远大于样本数n时，self-attention的n维序列长度可能限制表征容量。
- **LGRC依赖Laplacian score质量**：后处理阶段使用经典Laplacian Score进行冗余过滤，若局部邻域结构本身噪声较大，可能引入偏差。
- **单一ranking机制**：当前仅支持排序后截断选择，无法直接生成任意形状的特征子集（如树状/图状依赖结构）。
- **未来方向**：扩展至task-adaptive选择策略、混合选择机制、以及更一般的特征表征学习基础框架。

## 研究启发与可借鉴点
- **"反转数据矩阵"范式**：将样本级对比学习迁移到特征级是一个简洁而有力的思路转换，可复用于其他特征分析任务（如特征聚类、特征分组）。
- **嵌入范数作为显著性信号**：利用InfoNCE训练诱导的norm增长特性替代传统打分函数，为无需额外监督信号的特征排序提供新思路。
- **结构化多视图设计**：light/heavy/complementary组合比单一随机mask更能维持特征身份，对需要多视图增强的表征学习任务有借鉴价值。
- **两级冗余控制分离**：训练时decorrelation正则负责全局分离，推理时LGRC负责局部精炼，这种"学习+后处理"分离设计兼顾效率与效果。
- **样本感知架构调节**：按样本量切换两档超参的配置策略简单有效，可推广至其他数据规模差异大的跨域实验中。

## 关键术语表
- **Unsupervised Feature Selection (UFS)**：在无类别标签条件下，从原始特征中选择最具信息量的紧凑子集。
- **Inverted Contrastive Learning**：将数据矩阵转置后，以特征（而非样本）为对比学习实例的表示学习方法。
- **Projector-space Embedding Norm**：特征经encoder-projector网络后，未归一化输出向量的ℓ₂范数，用作特征显著性排序依据。
- **InfoNCE Loss**：对比学习标准目标函数，通过最大化正样本对相似度、最小化负样本对相似度学习不变表示。
- **Decorrelation Regularizer**：抑制不同特征在projector空间余弦相似性的正则项，鼓励特征占据分离的嵌入区域。
- **Laplacian-Gated Ranking Correction (LGRC)**：基于Laplacian Score的后处理修正模块，剔除候选集中局部邻域保持差的冗余特征。
- **Masked Positive View**：对特征样本profile向量施加部分遮挡（保留率0.50-0.90）生成的正样本视图。
- **Shuffled Negative View**：随机打乱特征样本profile向量中样本顺序生成的负样本视图，破坏样本对齐结构。

## 可复现要素
- **数据集**：12个benchmark数据集来自scikit-feature仓库（公开）。
- **代码**：已开源，地址https://github.com/neilghos/ICLFS。
- **关键超参**：
  - 优化器：Adam，lr=10⁻³，weight decay=10⁻⁴
  - 训练轮数：100 epochs
  - 对比温度：τ=0.05
  - Mask保留率：light=0.90, heavy=0.60, complementary=0.50, overlap=0.10
  - LGRC参数：α=1.5, q=0.75
  - 小样本 regime：dₑ=16, dₕ=512, dₚ=128, d_z=16, λ_decorr=0.20
  - 大样本 regime：dₑ=64, dₕ=1440, dₚ=2048, d_z=32, λ_decorr=0.40
- **硬件**：NVIDIA RTX 5060 Ti 16GB GPU，AMD Ryzen 5 3600X CPU，8GB RAM。
