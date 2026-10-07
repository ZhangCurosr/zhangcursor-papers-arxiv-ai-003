---
title: "TICDA-TABULAR-IN-CONTEXT-DATA-ATTRIBUTION"
source: https://arxiv.org/pdf/2610.07996v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:26:14"
---

# 论文速读：TICDA-TABULAR-IN-CONTEXT-DATA-ATTRIBUTION

## 一句话总结
TICDA 提出了一种针对表格基础模型（TFMs）的上下文数据归因方法，通过在 TFM 冻结隐层嵌入上拟合岭回归代理模型并固定表征，以单次前向传播和极低成本估计每个演示示例对查询预测的影响，可广泛应用于标注错误检测、上下文筛选、跨模型迁移与主动学习。

## 研究问题与动机
- **核心问题**：TFM 通过上下文学习进行预测（推理时参数完全冻结），但现有方法无法高效归因上下文中每个演示示例对特定查询预测的影响。
- **重采样方法不可行**：如 DemoShapley 需要遍历组合数量的上下文子集，每个子集需一次完整前向传播，对于数千演示的长上下文计算成本极高。
- **梯度方法不适用**：影响函数等基于参数的梯度方法依赖模型参数更新路径，而 TFM 推理时不更新任何参数，梯度无从计算。
- **实际痛点**：TFM 上下文通常由可用标签数据随意组装，可能混入错误标签、冗余或低质量示例， silently 降低预测精度并增加不必要的计算开销（如 TabPFN 推理成本随演示数二次增长）。

## 核心贡献（创新点）
1. **提出 TICDA 归因框架**：通过将影响函数思想从参数空间迁移到固定表征空间，在 TFM 冻结行表示上拟合岭回归代理，以单次前向传播输出所有演示的归因分数。与已有工作的本质区别：首次将影响函数适配到无参数更新的 ICL 场景，避免了重采样的组合爆炸。
2. **同时实现高质量归因与极低开销**：在 38 个 TabArena 数据集上，TICDA 在忠实度（faithfulness）和标注错误检测 AUC 上达到最优或次优，运行时 1.88–2.74 秒，比 LOO/DemoShapley/IG 快 2–3 个数量级。与已有工作的本质区别：唯一同时兼顾顶级归因质量和亚秒级速度的方法。
3. **上下文筛选实现精度保持甚至提升**：基于查询影响分数移除低贡献演示，在 40% 噪声下最多可移除 50% 演示且平衡准确率超过完整上下文；归因分数可在不同 TFM（TabICLv2/TabDPT → TabPFN-3/3.5）间迁移复用。与已有工作的本质区别：DemoShapley 跟踪随机基线，IG/DETAIL 严重退化，TICDA 是唯一实现"删减即增强"的方法。
4. **推导主动学习 acquisition 策略**：利用 TICDA 影响分数指导池化主动学习中候选样本的选择，在低数据 regime（≤512 标签）下 AULC 达 69.02%，优于熵采样和随机基线。与已有工作的本质区别：将归因思想从分析任务扩展至数据获取策略，填补了 TFM 主动学习的空白。

## 方法详解
TICDA 的核心是绕过 TFM 参数冻结的限制，构造一个可微代理来完成影响函数计算，关键设计如下：

- **Step 1 — 提取 TFM 表征**：对完整上下文 $\mathcal{D}$ 执行一次前向传播，从选定隐藏层提取每个演示的行表示 $m_j = \phi(x_j | \mathcal{D})$ 和查询表示 $m_q = \phi(x_q | \mathcal{D})$，拼成矩阵 $M \in \mathbb{R}^{n \times d}$ 和标签矩阵 $Y \in \mathbb{R}^{n \times K}$。

- **Step 2 — 拟合岭回归代理**：在固定表征上求解
  $$\widehat{B} = \arg\min_B \left[\frac{1}{2}\|MB - Y\|_F^2 + \frac{\lambda}{2}\|B\|_F^2\right] = H^{-1}M^\top Y, \quad H = M^\top M + \lambda I_d$$
  残差 $r_j = \widehat{B}^\top m_j - y_j$，代理预测为 $\widehat{B}^\top m_j$。

- **关键简化——固定表征近似**：假设移除演示 $i$ 后其余行和查询的表征不变，即 $\phi(\cdot|\mathcal{D}_{\setminus i}) \approx \phi(\cdot|\mathcal{D})$。Appendix C.1 实验表明，4096 演示时相对变化约 0.5%，且随上下文增大按 $1/n$ 衰减。该近似使得只需一次前向传播，后续所有归因仅涉及代理的矩阵运算。

- **Step 3 — 闭式影响分数**：对演示 $i$ 施加权重微扰 $\epsilon$，推导得
  $$s_{iq} = (m_q^\top H^{-1}m_i)(r_q^\top r_i)$$
  无需求 TFM 的 Hessian，只需代理的 $d \times d$ 矩阵 $H^{-1}$。

- **两种派生分数**：
  - **自影响（self-influence）**：$T_i = s_{ii} = h_i\|r_i\|_2^2$，其中 $h_i = m_i^\top H^{-1}m_i$ 为杠杆值。$T_i$ 大意味着该演示的标签不被上下文支持，用于标注错误检测。
  - **查询影响（query-influence）**：$u_i = \frac{1}{|\mathcal{V}|}\sum_{q \in \mathcal{V}} s_{iq}$。$u_i$ 负值越大表示该演示越损害验证集预测，用于上下文筛选。

- **迭代版本 TICDA-IT**：每步移除 5% 最低 $u_i$ 演示，重新拟合代理并重新排名，直至达到目标移除比例。

## 实验与结果
- **数据集与模型**：38 个 TabArena 分类数据集，每上下文最多 4096 演示；backbone 为 TabICLv2 和 TabDPT 1.2；跨模型迁移目标为 TabPFN-3 和 TabPFN-3.5。
- **LOO 忠实度与标注错误检测**（Table 1）：TabDPT 上 TICDA 忠实度 0.79（次优，DemoShapley 0.91）、AUC 0.88（与 DemoShapley 持平）；TabICL 上 TICDA 忠实度 0.67（最优）、AUC 0.89（最优，显著高于 LOO 的 0.80）。IG 和 DETAIL 两项指标均很差。
- **计算效率**（Figure 2）：TICDA 运行时间 1.88–2.74 秒，比 LOO/DemoShapley/IG 快 2–3 个数量级；DETAIL 也快但准确性低。
- **消融**（Table 2）：自影响显著优于查询影响，忠实度从 0.33 提升至 0.67，AUC 从 0.86 提升至 0.89；降维至 20% 对结果几乎无影响。
- **上下文筛选**（Figure 3）：40% 噪声下，TICDA/TICDA-IT 在移除 40–45% 演示时平衡准确率超过完整上下文；IG/DETAIL 严重退化，DemoShapley 跟踪随机基线。TICDA-IT 在所有设置下均优于单次 TICDA。
- **跨模型迁移**（Table 3）：用 TabICLv2 或 TabDPT 计算的分数直接用于 TabPFN-3/3.5 的上下文筛选，移除 50% 演示后平衡准确率达 ~64%，比完整上下文基线（~59%）高出 4+ 个百分点。
- **主动学习**（Table 4）：全预算下 TICDA AULC 为 68.35%，低数据（≤512 标签）下为 69.02%，均显著优于熵采样（67.56%/67.51%）和随机（66.23%/66.38%）。

## 相关工作脉络
1. **Data Shapley / DemoShapley（Ghorbani & Zou, 2019; Xie et al., 2025）**：重采样类归因，通过组合排列评估边际贡献；DemoShapley 适用于 LLM few-shot 但计算开销随上下文规模指数增长，无法扩展到 TFM 长上下文。
2. **Influence Functions（Koh & Liang, 2017）**：基于参数的梯度一阶近似，需计算 Hessian 逆；天然依赖参数更新，无法直接应用于推理时冻结参数的 TFM。
3. **DETAIL（Zhou et al., 2024）**：概念最接近 TICDA 的 LLM 归因方法，同样在隐状态上拟合岭回归代理并应用影响函数；但针对 LLM 短提示（高维空间少样本）设计，岭惩罚全部分配给单个演示，导致自影响语义与 TICDA 不同（Appendix C 详细推导）。
4. **Integrated Gradients（Sundararajan et al., 2017）**：沿路径积分预测梯度估算贡献；在 TFM 归因任务中忠实度和 AUC 均表现差（忠实度仅 0.20–0.30）。
5. **TabPFN 上下文研究（Rundel et al., 2024; Helli et al., 2024; Marszałek et al., 2026）**：观察到 TabPFN 上下文选择和噪声对性能的影响，但缺乏高效归因工具，本文填补了这一空白。

## 局限性与未来方向
- **固定表征近似未经验证更长上下文**：仅测试到 4096 演示，超过此规模时近似误差需进一步验证；Appendix C.1 的理论分析表明误差按 $1/n$ 衰减，大上下文下应更可靠。
- **仅适用于分类任务**：方法基于 one-hot 标签和平方损失，未扩展到回归和多模态 TFM。
- **消融实验单一 backbone**：维度消融仅在 TabICLv2 上进行，TabDPT 上的一致性未单独验证。
- **未来方向**：拓展至回归/多模态场景；改进固定表征近似以捕获表示的动态变化；与 kNN prompting、DPP 等 ICL 框架集成。

## 研究启发与可借鉴点
1. **"固定表征+线性代理"范式可迁移**：将影响函数从参数空间迁移到表征空间的思路，适用于任何推理时不更新参数的模型（如 ICL、retrieval-augmented generation），为黑盒模型的归因提供了通用框架。
2. **自影响与查询影响的任务分工值得借鉴**：自影响（leverage × residual²）天然捕捉"自身不一致性"，适合异常/错误检测；查询影响面向全局预测质量，适合筛选；两者分离设计具有普适价值。
3. **跨模型归因迁移的实现路径清晰**：在一个 TFM 上计算归因分数后直接应用于另一个 TFM 的上下文筛选，一次计算多次复用，大幅降低部署成本，这一策略可推广至其他 foundation model 家族。
4. **迭代粗筛策略（TICDA-IT）的工程价值**：每步移除 5% 后重新拟合+重排，比单次筛选更稳健，该思路可推广至其他需要逐步剔除的上下文优化场景。

## 关键术语表
- **Tabular Foundation Model (TFM)**：针对表格数据训练的基于 Transformer 的基础模型，支持上下文学习（ICL），推理时不更新
