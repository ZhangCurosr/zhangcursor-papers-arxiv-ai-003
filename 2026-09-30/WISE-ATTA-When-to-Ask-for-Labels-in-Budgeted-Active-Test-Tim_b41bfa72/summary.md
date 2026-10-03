---
title: "WISE-ATTA-When-to-Ask-for-Labels-in-Budgeted-Active-Test-Tim"
source: https://arxiv.org/pdf/2609.37687v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:14:09"
field: "测试时自适应与主动学习"
keywords: ["测试时自适应", "主动测试时自适应", "预算化标注", "在线决策", "预测漂移", "分布偏移鲁棒性"]
innovations: ["提出预算化ATTA新设定，将监督决策从批量内选样扩展到时间轴上选批次", "设计无回溯的预算债务节奏控制器，实现无需预知流长度的在线标注率均衡", "基于EMA锚点预测漂移的单样本选择准则，避免静态熵标准对模糊样本的无效选择"]
benchmarks: ["ImageNet-C", "ImageNet-R", "ImageNet-K", "ImageNet-A"]
---

# 论文速读：WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation

## 一句话总结
本文提出了**预算约束下的主动测试时自适应（Budgeted ATTA）**新设定，将监督从"批量内选样本"转向"时间轴上选批次"，设计了无需回放缓冲区或教师模型的 **WISE-ATTA** 方法，在 ImageNet-C/R/K/A 上以最多少 50% 的标签量达到与最新 ATTA 基线持平或更优的性能。

## 研究问题与动机
1. **现有 ATTA 方法的标注假设不现实**：主流方法默认每个测试批次都可请求标注，但实际部署中人工标注成本高、周期长，长期流式部署累积开销巨大。
2. **批中心化视角忽略时间维度**：已有工作聚焦"批次内选择哪些样本标注"（如 SimATTA、EATTA），将所有批次均匀分配监督，未考虑不同时间步的标注效用差异。
3. **测试时标注决策包含两个独立层次**：何时（when）申请监督 vs. 标哪个样本（what），两者需不同信号——前者评估批次级适配可靠性，后者捕捉样本级持续漂移状态。
4. **流式部署中流长度未知**：无法预知总测试时长，需要无全局规划能力的在线自适应预算分配策略。

## 核心贡献（创新点）
1. **提出预算化 ATTA 新设定**：将问题从批量内选择转向时间轴上分配，形式化为固定标注比例 $r$ 下的在线决策过程，更贴近真实部署约束。
2. **基于预算节奏的批次选择机制**：设计无需预知流长度的滑动窗口分位数 + 预算债务校正控制器（budget debt pacing），动态调整阈值以维持目标标注率，相比均匀/随机分配显著降低误差。
3. **基于预测漂移的单样本选择准则**：利用当前模型与 EMA 锚点之间的预测距离 $d_t^i = \|p_t^i - \bar{p}_t^i\|_2$ 选取最具学习潜力的样本，避免了纯熵标准在模糊样本上的失效。
4. **无回放缓冲区、无教师模型**：WISE-ATTA 不依赖外部标注源或历史样本缓存，仅用在线计算量即可实现比 SimATTA/HILTTA/CEMA 等更强的零样本主动适应。

## 方法详解
WISE-ATTA 由两个决策模块串联构成：

**1. 批次选择（Batch Selection）**
- **批次效用分数**：用低熵预测占比作为适配稳定性的代理指标
$$s_t = \sum_{i=1}^{n} \mathbb{I}[H(p_t^i) < \tau_{\text{ent}}]$$
其中 $\tau_{\text{ent}} = 0.4 \ln(C)$。
- **自适应分位数阈值**：维护最近 $W$ 个历史分数 $\mathcal{H}$，当 $s_t \geq \text{Quantile}_{1-\tilde{r}_t}(\mathcal{H})$ 时选中该批次。
- **预算节奏控制器**：定义预算债务 $d_t = rt - u_t$（目标累计用量减去实际用量），有效标注率 $\tilde{r}_t = \text{clip}(r + d_t/H_c, 0, 1)$，其中 $H_c$ 为局部校正窗口。预算用慢时 $\tilde{r}_t$ 上升放宽阈值，用快时收紧。
- **实用优化**： warmup 阶段（$|\mathcal{H}| < M$）用 Bernoulli$(r)$ 采样；设置 slack $\delta$ 防止短期波动导致预算闲置。

**2. 样本选择（Sample Selection）**
- 在选定批次中，计算每个样本当前预测与 EMA 锚点预测的漂移距离：$d_t^i = \|p_t^i - \bar{p}_t^i\|_2$
- 选择漂移最大的单一样本获取标注：$i_t^* = \arg\max_i d_t^i$
- 锚点参数每步以动量 $\mu$ 更新：$\bar{\theta} \leftarrow \mu\bar{\theta} + (1-\mu)\theta$
- 漂移的几何含义：一阶展开 $(d_t^i)^2 \approx \Delta_t^\top (J_t^i)^\top J_t^i \Delta_t$，衡量样本在当前适应方向上的响应度。

**3. 整体损失**
$$L_{\text{total}}^t = \lambda_{\text{sup}} L_{\text{sup}}^t + \lambda_{\text{ent}} L_{\text{ent}}^t$$
其中 $\lambda_{\text{sup}}=0.9, \lambda_{\text{ent}}=0.1$；无标签时仅更新归一化层，有标签时额外更新分类器头。

## 实验与结果
- **数据集**：合成失真（ImageNet-C，严重级别5，两种协议 CTTA/FTTA）；自然域偏移（ImageNet-R/K/A）
- **模型**：ResNet-50-BN（RN50-BN）、ViT-B-16
- **关键结果（FTTA，ImageNet-C）**：WISE-ATTA（1 label/batch）平均错误率 **51.6%**，优于 EATTA（52.7%，+1.1）和 HILTTA（52.8%，3 labels/batch）；CTTA 下 53.3%，优于 EATTA（54.6%，+1.3）和 HILTTA（53.7%，+0.4）。
- **自然偏移（FTTA，ImageNet-R/K/A）**：RN50-BN 平均错误率 **70.9%**（优于 EATTA 72.0%）；ViT-B-16 平均 **56.6%**（优于 EATTA 58.0%）。
- **低预算优势**：在 $r=0.2$ 时，ImageNet-K(ViT-B-16) 从 66.78 降至 59.12（+7.66 点提升），$r=0.6$ 时接近 $r=1.0$ 性能。
- **鲁棒性**：超参（$W, M, \delta$）变化时平均误差波动不超过 0.5 点；不同 batch size（16~128）和小延迟（≤100步）下仍保持最优。

## 相关工作脉络
1. **TENT [41] / CoTTA [45]**：纯无监督 TTA 基线，存在误差累积问题，WISE-ATTA 在此基础上引入稀疏监督校正。
2. **SimATTA [9]**：批内代表性样本选择，需3个标签/批次+回放缓冲区；WISE-ATTA 用1个标签/批次即超越，且无需任何存储。
3. **HILTTA [22]**：人工介入的在线模型选择，需3个标签/批次；WISE-ATTA 以更少标注实现同等或更强性能。
4. **CEMA [3]**：需教师模型蒸馏，标注成本高；WISE-ATTA 无需教师模型，仅用自监督信号完成主动选择。
5. **EATTA [42]**：最近单标签 ATTA，基于预测敏感度选样；WISE-ATTA 在相同预算下更优，核心区别是引入了时间维度的批次选择。
6. **Active Learning 经典（ uncertainty/disagreement/core-set）**：WISE-ATTA 将主动学习思想迁移至测试时场景，但将查询决策从"样本选择"扩展为"批次时间分配+样本选择"双层结构。

## 局限性与未来方向
1. **标注延迟敏感**：当标签到达延迟超过100步时，ViT-B-16/ImageNet-K 错误率从 58.0% 飙升至 93.0%，揭示 stale supervision 的破坏性。
2. **单分布偏移假设**：每个批次仅含单一分布偏移，未处理批次内多模态/多来源共存的异构场景。
3. **未做计算预算联合优化**：仅在监督层面做预算，未同时优化更新频率（Appendix F 显示跳过未选中批次的更新几乎无损，提示进一步裁剪空间）。
4. **未来方向**：延迟感知更新（recency-weighted updates）、异构批次中的多模态效用估计、联合预算适应与计算节约。

## 研究启发与可借鉴点
1. **双层决策框架的解耦思路**：将"何时监督"和"选什么样本"分离并用不同信号（批次熵 vs. 样本漂移）分别驱动，避免单一指标同时承担两项任务的信息瓶颈，可迁移到其他主动学习/在线决策场景。
2. **预算债务控制器（Budget Debt Pacing）**：用 $d_t = rt - u_t$ 的累积偏差调节有效阈值的思路，是一种无需预知总长度的在线均衡策略，可推广到任何有全局预算约束的序列决策任务。
3. **EMA 锚点+漂移选择**：用指数移动平均作为时间平滑参考、以预测变化幅度替代静态熵值来评估样本价值，为"活跃但未收敛"样本的识别提供了简洁有效的度量，适用于任何在线适应的样本筛选。
4. **附录实验设计**：延迟敏感性分析、不同 batch size 鲁棒性测试、更新跳过 ablation 等系统性部署研究，为后续工作提供了完整的基准验证范式。

## 关键术语表
**Test-Time Adaptation (TTA)**：模型部署后利用无标签测试流进行在线轻量更新的自适应技术。
**Active Test-Time Adaptation (ATTA)**：在 TTA 中引入稀疏监督（少量标注），用于稳定长期适应过程并纠正误差累积。
**Budgeted ATTA**：本文提出的新设定，标注资源仅占测试批次的一个固定比例，需决定"何时"而非仅"选什么"。
**Prediction Drift**：当前模型预测与 EMA 锚点预测之间的 L2 距离，衡量样本在当前适应方向上的响应程度。
**Budget Debt Pacing**：用累积预算偏差动态调整有效标注率 $\tilde{r}_t$，实现无全局规划下的在线预算均衡。
**EMA Anchor**：以动量 $\mu$ 平滑当前模型参数的指数移动平均模型，作为时间一致性参考。
**FTTA / CTTA**：Fully/Continual Test-Time Adaptation 两种协议——FTTA 在分布切换时重置模型，CTTA 跨多个分布连续适应。
**Supervisory Update Scope**：本文发现仅用监督更新分类器头（classifier head）最稳定，更新深层参数易破坏预训练表示。

## 可复现要素
- **数据集**：ImageNet-C（严重级别5）、ImageNet-R/K/A，全部公开
- **代码**：已开源，https://github.com/Muhammad-Huzaifaa/WISE-ATTA
- **模型权重**：使用 ImageNet-1K 预训练的 RN50-BN 和 ViT-B-16，公开可获取
- **关键超参**：学习率（ImageNet-C: 2.5e-4，R/K: 1e-3，A: 5e-3）；EMA 动量 $\mu=0.9$；历史窗口 $W=250$；warmup $M=1$；校正窗口 $H_c=50$；slack $\delta=1$；熵阈值 $\tau_{\text{ent}}=0.4\ln(C)$；损失权重 $\lambda_{\text{sup}}=0.9, \lambda_{\text{ent}}=0.1$；batch size=64；SGD 优化；3次随机种子平均（0, 41, 58）
