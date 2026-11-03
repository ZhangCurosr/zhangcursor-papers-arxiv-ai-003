---
title: "To-Align-or-Not-To-Align-Check-Your-COMPASS-Before-You-Train"
source: https://arxiv.org/pdf/2610.18638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:53:02"
---

# 论文速读：To Align or Not To Align: Check Your COMPASS Before You Train

## 一句话总结
提出 COMPASS 双轴诊断工具，通过在昂贵对齐或合并操作前仅运行少量 probe 前向传播，利用局部 k-NN 邻域重叠准确预测模型对的兼容性；其预测分数与下游成功率高度相关，显著优于全局 CKA 等传统度量。

## 研究问题与动机
- **核心痛点**：跨模态对齐与任务算术合并是现代 AI 流水线的常见操作，但判断两个预训练模型是否“兼容”通常需运行完整的下游训练/合并实验，计算成本随候选对数量二次增长。
- **现有方法不足**：传统可迁移性度量（如 Task2vec、LEEP、LogME）聚焦单一模型微调潜力；全局表征相似度指标（如 CKA）在跨架构/跨模态场景下与下游性能相关性弱，且无法区分“初始接近度”与“训练范式上限”。
- **理论契机**：Aristotelian Representation Hypothesis 表明独立训练模型的全局几何虽不同，但局部邻域结构高度一致；由此启发可通过轻量级局部重叠反推对齐/合并可行性。
- **缺失环节**：缺乏一个可在单模型级别预计算、跨候选对复用、且能同时刻画起始条件与范式瓶颈的通用诊断指标。

## 核心贡献（创新点）
- **提出 COMPASS 双轴诊断框架**：将模型兼容性解耦为成对特异性 Ease 与范式层级 Ceiling，仅凭少量 probe 样本即可在 Training/Merging 前完成高效排序。
- **给出坚实的理论支撑**：Theorem 4.1 证明起始邻域重叠控制有限训练预算内的收敛概率；Theorem 4.2 证明两范式间的 target discrepancy 为可达重叠设定了不可突破的硬上限。
- **构建四象限决策规则**：将连续评分映射为 Act now / Budget-limited / Ceiling-limited / Incompatible 四类工程策略，使排名结果直接转化为可执行建议。
- **系统性实证验证**：在 72 对跨模态对齐与 392 次任务算术合并上验证，COMPASS 与下游成功率的 Pearson 相关系数达 0.90/0.91 与 0.70，全面超越 Global CKA 及其他轻量基线。

## 方法详解
- **基础度量 k-NN Overlap**：对 probe 集每个样本 $i$，分别取两编码器 $X, Y$ 的 $k$ 个最近邻集合，重叠率 $o_i = |\text{kNN}_X(i) \cap \text{kNN}_Y(i)| / k$，数据集均值 $O$ 作为基础局部一致性感知。全文固定 $k=10$，在 $k\in\{20,50,100\}$ 下结论稳定。
- **Ease 轴（起始易用性）**：在相关训练动态起点测量 per-sample 重叠 $o_i^{(0)}$。跨模态对齐中为预训练冻结编码器的重叠；任务合并中为两个微调 checkpoint 的预合并重叠。Theorem 4.1 表明 $o_i^{(0)}$ 越高，有限步数内两模型邻域分布 TV 距离收缩至阈值 $\eta$ 的概率越大。
- **Ceiling 轴（范式上限）**：衡量训练目标本身允许的最大邻域一致性。跨模态中，vision encoder 的 ceiling 为其与参考文本库所有编码器的平均重叠；任务合并中定义为 $C(b) = 1 - \text{mean}_{v\in\mathcal{L}} O(b,v)$ 以表征范式特异性。两者均反映“无论初始多接近，该范式组合理论上能达到的最高重叠”。
- **综合评分与象限划分**：连续评分 $S = \text{Ease} \times \text{Ceiling}$ 用于严格排序；将两轴分别按中位数切分得到四象限：High-High（Act now，优先使用）、Low-High（Budget-limited，需增数据/优化/更强伴侣）、High-Low（Ceiling-limited，应更换下游目标相近的范式）、Low-Low（Incompatible，降低优先级）。
- **计算复杂度**：特征提取仅随模型数量线性扩展，对 19×10 对齐网格仅需 29 次 probe 前向传播，相比 190 次完整 CSA 训练节省约两个数量级 wall-clock 时间。

## 实验与结果
- **跨模态对齐**：19 个视觉模型（ImageNet-21K/MAE/DINOv2/CLIP/CLIP-FT/SigLIP）与 10 个文本编码器构成 72 对，使用 CSA adapter 在 MS-COCO 训练，Flickr30k 与 COCO 测试 I2T R@1。COMPASS 在 Flickr30k 与 COCO 上 Pearson r 分别达 **0.90** 与 **0.91**，Spearman ρ 达 0.91/0.93；Top-1 正确率 100%，NDCG@5 为 0.957，Budget-to-best 为 1/72。Global CKA 仅 0.63/0.68。
- **任务算术合并**：14 个 ViT backbone 在 8 个图像分类任务（MNIST/SVHN/GTSRB/DT D/EuroSAT/RESISC45/Stanford Cars/SUN397）上微调，生成 392 对合并组合，评估均值保留率（mean retention）。COMPASS Pearson r=**0.70**，优于 Ease alone (0.67)、Ceiling alone (0.64) 与 Global CKA (0.25)。Kruskal-Wallis 检验证实四象限性能分布极显著（$H=198.2, p<10^{-42}$）。
- **鲁棒性与泛化**：更换 k 值、probe 集（MNIST 单任务 vs 8 任务混合）、adapter 类型（Linear/MLP/LoRA）后预测一致性保持；逐家族/范式剔除后相关性仅微降（对齐 0.85–0.93，合并 0.60–0.72）；ceiling 注入 50% 噪声仍维持 0.883/0.695 的相关性。

## 相关工作脉络
- **表征收敛假设**：Platonic Hypothesis [15] 主张全局表征收敛，Aristotelian Hypothesis [16] 证明实际可观测的是局部邻域一致；本文继承后者并将观测结论工程化为配对筛选工具。
- **训练目标几何统一**：I-Con 框架 [23] 用 KL 散度统一对比/重建/聚类目标；本文借此将目标差异形式化为 $\gamma_i$，奠定 Ceiling 的理论边界。
- **多模态对齐方法**：LiT [1]、BLIP-2 [2]、FLamingo [3] 等聚焦改进对齐架构或减少配对数据需求；本文不竞争对齐算法本身，而是前置解决“哪些冻结编码器对值得投入对齐成本”的选型问题。
- **权重空间合并**：Model soups [7]、Task arithmetic [11]、TIES-merging [9] 依赖平均或向量编辑；本文指出这些方法的成败在合并前已由局部邻域一致性与范式匹配度决定。
- **单模型可迁移性度量**：Task2vec [12]、LEEP [13]、LogME [14] 评估单一预训练模型对目标任务的特征可用性；本文填补“双模型配对兼容性”这一更贴近现代 pipeline 操作的诊断空白。

## 局限性与未来方向
- **Ceiling 的领域依赖性**：Ceiling 的具体计算需结合任务场景与参考库设计，目前尚无完全通用的单公式，依赖一定领域先验。
- **动力学假设为充分非必要条件**：The
