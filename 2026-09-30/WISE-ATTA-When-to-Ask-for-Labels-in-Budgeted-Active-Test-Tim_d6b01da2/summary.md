---
title: "WISE-ATTA-When-to-Ask-for-Labels-in-Budgeted-Active-Test-Tim"
source: https://arxiv.org/pdf/2609.37687v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:14:09"
field: "测试时自适应与主动学习"
keywords: ["active test-time adaptation", "budgeted supervision", "distribution shift", "test-time adaptation", "online learning", "label efficiency"]
innovations: ["提出预算化主动测试时适配设定，将监督决策从批内扩展到跨时间预算分配", "设计基于预算债务反馈的自适应批次选择策略", "提出基于预测漂移（EMA锚点对比）的单样本选择准则以捕捉未收敛适应动力学"]
benchmarks: ["ImageNet-C", "ImageNet-R", "ImageNet-K", "ImageNet-A"]
---

# 论文速读：WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation

## 一句话总结
本文提出了**WISE-ATTA**，一种面向预算约束的主动测试时适配方法，通过在线判断"何时应请求标签"而非仅关注"批内选哪个样本"，在显著减少标注量的同时保持甚至提升模型在分布漂移下的鲁棒性。

## 研究问题与动机
1. **标注成本高企**：现有 ATTA 方法隐含假设每个测试批次都可获得标注，但在长期部署中每批次请求标签会累积高昂成本（人工或教师模型）。
2. **时间维度被忽视**：已有工作聚焦于"批内样本选择"（what to label），忽略了"跨时间步的预算分配"（when to label）这一更宏观问题。
3. **均匀分配效率低**：在固定全局预算下，若对所有批次一视同仁地施加监督，会因为不同时间段的适应收益差异而浪费标注机会。
4. **现实约束更贴近场景**：实际部署常受人力、算力或延迟限制，监督只能间歇性可用，因此需要一种能自适应调节标注节奏的方法。

## 核心贡献（创新点）
1. **提出预算化主动测试时适配（Budgeted ATTA）设定**：将监督约束形式化为跨整个测试流的全局标注比例 $r$，把核心决策从"批内选样"拓展至"跨时间选择批次"。
2. **设计预算 paced 的批次选择策略（Budget-Paced Batch Selection）**：基于滑动历史缓冲区中的置信度分数（低熵样本占比）与自适应分位数阈值动态决策是否申请标签，并引入"预算债务"反馈机制控制长期标注速率。
3. **提出基于预测漂移的单样本选择准则（Drift-Based Sample Selection）**：通过当前模型与 EMA 锚点的预测差异衡量样本的适应动力学，选取仍在适应但未收敛的高价值样本进行单标签更新。
4. **无需回放缓冲区与教师模型**：与 CEMA 等既有工作相比，WISE-ATTA 不依赖外部监督信号，仅用轻量在线统计即可完成高效标注调度。

## 方法详解
**总体框架**：WISE-ATTA 分两阶段决策——批次选择（when）与样本选择（what）。

### 1. 批次选择（Batch Selection）
- **批次效用得分**：对每个批次 $\boldsymbol{B}_t$，统计低熵预测样本数：
  $$s_t = \sum_{i=1}^{n} \mathbb{I}[H(p_t^i) < \tau_{\mathrm{ent}}]$$
  其中 $\tau_{\mathrm{ent}} = 0.4 \ln(C)$，$s_t$ 越大表示批次内越"可信"，适合施加监督。
- **自适应分位数阈值**：维护最近 $W$ 个 $s_t$ 的滑动窗口 $\mathcal{H}$，阈值设为 $\tau_t = \mathrm{Quantile}_{1-\tilde{r}_t}(\mathcal{H})$。
- **预算债务控制器**：定义 $d_t = rt - u_t$（目标累计用量与实际用量之差），有效比率 $\tilde{r}_t = \mathrm{clip}(r + d_t/H_c, 0, 1)$。当标注过慢时 $\tilde{r}_t$ 增大使阈值放宽，反之收紧。
- **启动与兜底机制**：历史不足时采用 $\mathrm{Bernoulli}(r)$ 随机采样；当使用量显著低于目标时强制补偿标注。

### 2. 样本选择（Sample Selection）
- **EMA 锚点模型**：维护当前参数 $\theta$ 的指数移动平均 $\bar{\theta}$，动量 $\mu = 0.9$。
- **预测漂移**：$d_t^i = \|p_t^i - \bar{p}_t^i\|_2$，反映样本预测随适应进程的变化幅度。
- **选择策略**：在选定批次中取漂移最大者 $\displaystyle i_t^* = \arg\max_i d_t^i$ 获取唯一标签。
- 理论解释：一阶展开显示 $(d_t^i)^2 \approx \Delta_t^\top (J_t^i)^\top J_t^i \Delta_t$，即漂移衡量的是样本沿最近适应方向的敏感性。

### 3. 总体损失
$$L_{\mathrm{total}}^t = \lambda_{\mathrm{sup}} L_{\mathrm{sup}}^t + \lambda_{\mathrm{ent}} L_{\mathrm{ent}}^t$$
其中 $\lambda_{\mathrm{sup}}=0.9,\ \lambda_{\mathrm{ent}}=0.1$；无监督批次仅更新归一化参数，有监督批次额外更新分类头。

## 实验与结果
**数据集**：ImageNet-C（15类合成畸变，严重度5）、ImageNet-R/K/A（自然分布偏移）。

**基线**：TENT、CoTTA、SAR、ETA、SimATTA、HILTTA、EATTA、CEMA。

**主要结果**：
- **ImageNet-C（CTTA）**：WISE-ATTA 平均错误率 **53.3%**（1 label/batch），优于 EATTA（54.6%）与 HILTTA（3 label/batch, 53.7%）。
- **ImageNet-C（FTTA）**：平均错误率 **51.6%**，领先 EATTA 1.1 个百分点。
- **ImageNet-R/K/A（FTTA）**：RN50-BN 平均 **70.9%**、ViT-B-16 平均 **56.6%**，均优于所有基线（EATTA 分别为 72.0% / 58.0%）。
- **变化预算率实验**：在 $r \in [0, 0.6]$ 范围内持续领先 UNIFORM 与 RANDOM；$r=0.6$ 时接近 $r=1.0$ 的性能。
- **标签延迟分析**：延迟 200 batch 后 WISE-ATTA 仍多数情况下最优，但 ViT-B-16/ImageNet-K 误差升至 93.0%，凸显延迟鲁棒性待改进。

**最强提升**：在 ImageNet-C FTTA 上以同等 1 label/batch 预算超越 EATTA **1.1 个百分点**；在 ImageNet-C CTTA 上以更少标签击败 3-label 的 HILTTA **0.4 个百分点**。

## 相关工作脉络
1. **TENT (ICLR 2021)**：纯无监督熵最小化 TTA 的代表作，WISE-ATTA 在其基础上引入稀疏监督与批次选择，弥补误差累积问题。
2. **CoTTA (CVPR 2022)**： continual TTA 的代表，强调长期稳定性，本文与其相比在主动监督下达到更低错误率。
3. **SimATTA (ICLR 2024)**：基于聚类与熵的批内样本选择，使用 3 labels/batch，WISE-ATTA 以更少标签实现更好性能。
4. **EATTA (CVPR 2025)**：单标签选择方法，基于预测敏感性，WISE-ATTA 通过引入时序漂移信号超越其表现。
5. **CEMA (ICLR 2024)**：需回放缓冲区与教师模型，WISE-ATTA 在无额外资源需求下达到更优或相当的精度。
6. **HILTTA (TMLR 2024)**：结合主动学习与模型选择，使用 3 labels/batch，本文以更低标注成本取得竞争性能。

## 局限性与未来方向
1. **延迟鲁棒性有限**：当标签返回延迟超过数百 batch 时，模型已严重漂移，旧标签可能有害；需研究时效加权更新与延迟感知选择。
2. **异构批次假设不足**：当前设定假设每批次服从单一分布偏移，现实场景中同一批次内可能混合多种 shift，批次效用需扩展至多模态覆盖评估。
3. **计算预算未联合优化**：仅在标注预算上做约束，未同步优化无监督更新频率；附录 F 表明跳过非选定批次更新影响较小，提示可进一步联合调度。
4. **单标签限制**：每批次最多查询一个标签，极端低预算场景下信息量可能不足。

## 研究启发与可借鉴点
1. **"何时监督"作为独立优化维度**：将预算分配从"静态固定速率"升级为"动态自适应调控"的思路可迁移至其他主动学习/在线学习场景。
2. **预算债务反馈控制器设计**：利用累积偏差驱动阈值松紧的策略简洁且无需知道总长度，可推广至任意持续流式预算分配问题。
3. **漂移作为适应动力学信号**：用 EMA 锚点与当前模型的预测差值捕捉"仍在收敛但尚未稳定"的样本状态，这一区分置信度（entropy）与适应阶段（drift）的思路值得借鉴。
4. **轻量级替代重资源方案**：在不使用回放缓冲区与教师模型的前提下达到 SOTA 竞争性能，证明了在线统计信号的充分性。
5. **延迟鲁棒性分析框架**：将标签延迟作为独立变量系统化评估，为后续部署研究提供了可复用的评测范式。

## 关键术语表
**Active Test-Time Adaptation (ATTA)**：在测试时阶段通过稀疏查询标签来纠正模型、防止无监督适配误差累积的方法。

**Budgeted ATTA**：将标注约束从"每批次"推广至"全局比例 $r$"的设定，迫使方法在时间维度上智能分配预算。

**Budget-Paced Batch Selection**：基于滑动历史分位数与预算债务反馈机制决定何时申请监督的批次选择策略。

**Prediction Drift**：当前模型与 EMA 锚点模型之间预测概率的 $L_2$ 距离，反映样本在适应过程中尚未收敛的动态程度。

**EMA Anchor**：模型参数的指数移动平均版本，作为稳定参考基准用于测量预测漂移。

**Continual TTA (CTTA)**：测试过程中不重置模型、持续适应一条长流的协议。

**Fully TTA (FTTA)**：每次出现新分布偏移时重置回预训练检查点的协议。

**Label Annotation Delay**：从提交样本到标签返回之间的滞后，以测试批次数为度量单位。

## 可复现要素
- **数据集**：ImageNet-C/R/K/A，均为公开基准，标注严重度 level 5 或完整测试集。
- **代码**：已开源，仓库地址 https://github.com/Muhammad-Huzaifaa/WISE-ATTA
- **模型**：ResNet-50-BN 与 ViT-B-16，ImageNet-1K 预训练权重。
- **关键超参**：batch size=64，learning rate=$2.5\times10^{-4}$（C）/$10^{-3}$（R/K）/$5\times10^{-3}$（A），EMA momentum $\mu=0.9$，历史窗口 $W=250$，warmup $M=1$，修正 horizon $H_c=50$，slack $\delta=1$，熵阈值 $\tau_{\mathrm{ent}}=0.4\ln(C)$，$\lambda_{\mathrm{sup}}=0.9,\ \lambda_{\mathrm{ent}}=0.1$。
- **随机种子**：3 次重复（0, 41, 58），报告平均值。
