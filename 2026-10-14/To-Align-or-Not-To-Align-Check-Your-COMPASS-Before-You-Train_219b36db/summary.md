---
title: "To-Align-or-Not-To-Align-Check-Your-COMPASS-Before-You-Train"
source: https://arxiv.org/pdf/2610.18638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:52:27"
field: "表示学习与模型组合"
keywords: ["model compatibility", "cross-modal alignment", "task arithmetic merging", "representational convergence", "k-NN overlap", "model merging", "representation learning"]
innovations: ["将模型对兼容性分解为成对易用性（ease）与范式级上限（ceiling）两个廉价计算轴", "从局部k-NN重叠出发给出有限时间收敛与可达重叠的理论界", "在跨模态对齐与任务算术合并两大场景验证诊断有效性，超越全局CKA"]
benchmarks: ["Flickr30k", "COCO", "task-arithmetic merging benchmark"]
---

# 论文速读：To Align or Not To Align: Check Your COMPASS Before You Train

## 一句话总结
提出 COMPASS（COMPatibility ASsessment Score），一个在昂贵对齐/合并操作之前即可廉价估算模型对兼容性的诊断工具；通过将兼容性分解为"成对易用性"（ease）和"范式级上限"（ceiling）两个轴，在跨模态对齐（Pearson r = 0.91）和任务算术模型合并（Pearson r = 0.70）两个任务上均显著优于全局 CKA 等传统度量。

## 研究问题与动机
- **核心问题**：现代 AI 管线中，组合两个预训练模型（跨模态对齐、权重合并）是最关键且昂贵的决策之一；然而模型对的兼容性（compatibility）通常只有在完成昂贵对齐/合并后才可知。
- **现有方法不足**：现有可迁移性度量（Task2vec、LEEP、LogME）仅针对单模型选择；全局表示相似度（如 CKA）无法捕捉局部邻域结构，实证显示其与下游成功相关性远低于局部 k-NN overlap。
- **理论缺口**：Platonic/Aristotelian Representation Hypothesis 等仅描述了局部收敛现象，缺乏可操作的诊断框架；缺少将"起始接近度"与"范式兼容上限"分离的定量机制。
- **动机**：若能仅用少量 probe 样本的前向传播估算兼容性并据此排序候选对，可将评估成本从与候选对数量成正比降至与模型数量成正比。

## 核心贡献（创新点）
1. **提出 COMPASS 诊断框架**：将模型对兼容性分解为 pair-specific ease 和 paradigm-level ceiling 两个廉价计算轴，仅需少量 probe 样本前向传播即可完成打分。
   - 与已有工作本质区别：不同于仅依赖全局表示相似度（CKA）或元数据标签的方法，COMPASS 明确区分"成对起始重叠"与"范式可达上限"，并据此给出可操作的四象限策略（act now / budget-limited / ceiling-limited / incompatible）。

2. **给出局部 k-NN overlap 预测兼容性的理论支撑**：Theorem 4.1 证明起始重叠越高，有限训练预算内收敛概率越大；Theorem 4.2 证明目标差异（target discrepancy）为可达重叠设定了不可逾越的上界。
   - 与已有工作本质区别：以往工作仅经验观察局部收敛现象，本文首次将其形式化为可计算的 finite-time convergence bound 与 achievable overlap bound。

3. **在两大赛道验证诊断有效性**：72 对跨模态对齐（Flickr30k/COCO R@1，Pearson r = 0.90/0.91）与 392 对任务算术合并（mean retention，Pearson r = 0.70），全面超越全局 CKA（0.63/0.68 与 0.25）及其他基线。
   - 与已有工作本质区别：不仅给出排序质量（Spearman ρ、NDCG），还证明低成本筛选（budgeted evaluation）可立即定位最优对——Flickr30k 最优对齐对在第 1 位、最优合并对在 392 候选中排第 5。

## 方法详解
- **局部 k-NN overlap 定义**：对 probe 集中每样本 i，分别取 encoder X 与 Y 的 k 近邻集合，计算交集比例 $o_i(k) = |\text{kNN}_X(i) \cap \text{kNN}_Y(i)| / k$，整体平均 $O(k) = \frac{1}{N}\sum_i o_i(k)$，论文默认 $k = 10$。
- **Ease score（成对易用性）**：衡量候选模型对在任何训练开始前的初始局部重叠，即 $\text{ease}(X,Y) = O(k)$。对于跨模态对齐为预训练 vision-text 对的 k-NN overlap；对于任务合并为两个 fine-tuned checkpoint 之间的 k-NN overlap。
- **Ceiling score（范式级上限）**：衡量训练范式本身所能支持的平均局部重叠上界。对于对齐，$\text{ceiling}(b) = \frac{1}{|\mathcal{L}_{\text{text}}|} \sum_{v' \in \mathcal{L}_{\text{text}}} O(b, v')$；对于合并，$\text{ceiling}(b) = 1 - \frac{1}{|\mathcal{L}_{\text{vision}}|-1} \sum_{v} O(b,v)$（即相对其他范式的"特殊化"程度）。
- **COMPASS 综合得分**：$\text{COMPASS} = \text{ease} \times \text{ceiling}$，对候选对排序；同时将两轴各自以中位数为阈值划分为四个象限：high-ease/high-ceiling（act now）、low-ease/high-ceiling（budget-limited）、high-ease/low-ceiling（ceiling-limited）、low-ease/low-ceiling（incompatible）。
- **理论依据**：Theorem 4.1（起始重叠控制有限时间收敛概率）、Theorem 4.2（目标差异 TV 给出可达重叠上界 $1 - \gamma_i/(1-\varepsilon)$）。

## 实验与结果
- **跨模态对齐**：72 对 vision-language（19 vision / 10 text），基于 CSA adapter（闭式 canonical correlation），评估 Flickr30k 与 COCO 的 I2T R@1。COMPASS 得分为 $r = 0.90$（Flickr30k）与 $r = 0.91$（COCO）；ease 单轴 0.84/0.83，ceiling 单轴 0.85/0.83；Global CKA 仅 0.63/0.68。四象限均值 R@1：act now 0.46 ± 0.09 > ceiling-limited 0.34 > budget-limited 0.30 > incompatible 0.21（Kruskal–Wallis $p < 10^{-9}$）。
- **任务算术合并**：14 个 ViT backbone × 8 个分类任务 = 112 checkpoint，共 392 对合并，mean retention 为指标。COMPASS $r = 0.70$；ease 0.67、ceiling 0.64；Global CKA 仅 0.25。四象限均值：act now 0.84 ± 0.16 > budget-limited 0.57 ≈ ceiling-limited 0.56 > incompatible 0.35（$p < 10^{-42}$）。
- **排序质量**：Flickr30k 上 Spearman ρ = 0.91/0.93（COCO），NDCG@5 = 0.957，Top-1 regret = 0%；合并上 ρ = 0.70，Budget-to-best = 5/392。
- **鲁棒性**：对齐方法（CSA/Linear/MLP/LoRA）下排名一致（Spearman 0.82–0.99）；保留任意单一 family/paradigm 后相关性仍维持 0.85–0.93（对齐）与 0.60–0.72（合并）；ceiling 粗粒度化（同范式共享值、加高斯噪声）几乎不影响排名；去除 ceiling 仅剩 ease 时对齐从 0.91 降至 0.87。

## 相关工作脉络
- **Platonic/Aristotelian Representation Hypothesis**（Huh et al. 2024; Gröger et al. 2026）：前者主张全局表示收敛，后者指出仅局部邻域一致可被观测；本文继承局部视角并向前推进为可操作诊断。
- **I-Con 统一目标框架**（Alshammari et al. 2025）：用条件分布 KL 散度统一对比/重建/聚类目标；本文借其连接目标结构差异与邻域结构差异，为 ceiling 提供理论基础。
- **Multi-modal alignment 与 modality gap**（Li et al. 2022; Liang et al. 2022）：CLIP 等存在 persistent 模态鸿沟；本文不改进对齐方法，而是提供对齐前的模型选择诊断。
- **Model soups / Task arithmetic / TIES-merging / Evolutionary merging**（Wortsman et al. 2022; Ilharco et al. 2023; Yadav et al. 2023; Akiba et al. 2025）：这些方法多在合并后才知道效果；本文首次实现合并前的兼容性预判。
- **Transferability metrics**（Task2vec, LEEP, LogME）：仅评估单模型到单任务的迁移性；本文处理"两模型对的组合兼容性"这一更高层问题。
- **Convergence analysis under SGD noise**（Appendix F.5）：将 A2 放松为带 additive noise floor 的形式，使理论适用于更实际的随机优化场景。

## 局限性与未来方向
- Ceiling proxy 依赖具体应用（对齐 vs. 合并）与领域知识，无法写成通用公式。
- Theorem 4.1 假设（A1–A2）为充分非必要条件；对优化器/数据/目标差异极大的模型，衰减可能较慢导致 bound 宽松。
- Phase diagram 反映的是有限预算下的实际可行性，并非不可逾越的 impossibility result；投入更多数据/更大 adapter/更长训练仍可能挽救低分对。
- 论文未检验除跨模态对齐与任务合并外的其他双模型交互场景（如蒸馏、参数高效微调）；作者明确表示此为未来方向。
- 实验仅覆盖 ViT / CLIP / SigLIP / DINOv2 / MAE / 若干 LLM 语言塔等有限范式和架构。

## 研究启发与可借鉴点
- **两轴分解思路**：将复杂兼容性拆为"起始距离"与"范式上限"是极具迁移性的分析范式，可推广至 PEFT 适配、模型蒸馏、多专家融合等场景。
- **局部 vs. 全局表示相似度的选择**：实证表明局部 k-NN overlap 在缩放性和预测力上均优于 CKA；后续研究在需要快速诊断时应优先考虑局部指标。
- **闭式诊断 + 四象限决策**：COMPASS 不只给出分数还给出行动建议（act now / 加预算 / 换范式 / 放弃），这种"诊断即干预"的设计可直接嵌入模型库管理流程。
- **参考库重用的效率增益**：特征仅按模型而非候选对抽取，使得 19×10 对齐网格只需 29 次 probe pass 而非 190 次下游训练，wall-clock 降低两个数量级——对大规模模型库筛选具有直接工程价值。
- **与团队方向结合机会**：若团队涉及多模态预训练、模型合并/融合、或大模型微调流水线，COMPASS 可作为上线前的"兼容性过滤器"，显著降低试错成本。

## 关键术语表
- **COMPASS (COMPatibility ASsessment Score)**：本文提出的模型对兼容性诊断分数，由 ease × ceiling 构成。
- **Ease score**：成对特异性指标，衡量候选模型对在任意训练/合并开始前于 probe 集上的局部 k-NN overlap。
- **Ceiling score**：范式级指标，衡量训练目标本身允许达到的平均局部重叠上界；不同模型共享同一 base 范式时 ceiling 相同。
- **Local k-NN overlap**：两编码器对同一 probe 样本的 k 近邻集合交集比例，衡量局部邻域结构一致性。
- **Target discrepancy ($\gamma_i$)**：两范式目标条件分布之间的总变差距离，决定 achievable overlap 的硬上界。
- **Act now / Budget-limited / Ceiling-limited / Incompatible**：以 ease 与 ceiling 中位数为阈划分的四象限，分别对应立即使用、追加预算、更换范式、放弃四个决策。
- **Task arithmetic merging**：将同一 base 模型在不同任务上 fine-tune 的权重偏差向量相加以实现多任务合并。
- **CSA (Canonical Space Alignment)**：基于闭式 canonical correlation 的跨模态对齐 adapter，无需迭代训练。

## 可复现要素
- **数据集**：WIT-1024（probe set，CC BY-SA 3.0）、MS-COCO 2014（对齐训练/评估，CC BY 4.0）、Flickr30k（检索评估）、Task-arithmetic benchmark 的 8 个图像分类任务（MNIST/SVHN/GTSRB/DTD/EuroSAT/RESISC45/Stanford Cars/SUN397）。
- **代码**：开源，发布于 https://github.com/mlbio-epfl/COMPASS（论文声明接受后发布，附录给出完整配置与脚本路径）。
- **模型权重**：使用公开 timm checkpoint（ViT 系列、CLIP、SigLIP、DINOv2、MAE）与公开 LLM 语言塔；未发布新权重。
- **关键超参**：k = 10（neighborhood size）、ε 平滑参数（小常数，如 1/N）、canonical dimension = 512、probe 大小 1024（对齐）/ 2048（合并）、fine-tuning：AdamW、LR backbone 1e-4 / head 1e-3、20 epochs、batch 32、cosine schedule、seed = 42。
