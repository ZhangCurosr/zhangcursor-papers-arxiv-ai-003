---
title: "VLA-RETAIN-BENCHMARKING-CROSS-DIMENSION-EFFECTS-OF-VISION-TU"
source: https://arxiv.org/pdf/2610.27698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:58:03"
field: "具身智能与VLA模型鲁棒性"
keywords: ["VLA", "vision-language-action", "robotics", "robustness", "fine-tuning", "cross-dimensional effects", "benchmarking"]
innovations: ["提出VLA-Retain配对基准衡量视觉微调的跨维度效应", "揭示目标增益不预测非目标保留的模型依赖性规律", "验证DELTA-style特征保留策略在VLA微调中的有效性但具模型特异性"]
benchmarks: ["LIBERO-Plus"]
---

# 论文速读：VLA-RETAIN-BENCHMARKING-CROSS-DIMENSION-EFFECTS-OF-VISION-TU

## 一句话总结
本文提出VLA-Retain配对基准，系统评估了对VLA视觉模块进行目标域微调后对非目标维度鲁棒性的跨维度影响，揭示了"目标域成功不等于整体鲁棒性保留"的核心结论。

## 研究问题与动机
1. **现有评估范式存在盲区**：现有鲁棒性基准仅评估固定模型，而目标适配研究只报告目标域增益，两者均无法揭示微调对非目标条件的影响。
2. **VLA部署的实际需求**：机器人部署时面临相机位姿、传感器噪声、光照、背景、物体布局等多维变化，模型需在适应新条件的同时保持对其他条件的鲁棒性。
3. **架构异质性带来的评估困难**：不同VLA基线鲁棒性差异大，绝对分数混淆了适应效应与起点差异；非目标衰退可能源于通用遗忘而非扰动特异性干扰。
4. **适配设计空间的探索空白**：训练数据覆盖范围、微调范围、保留策略等设计选择的影响尚未被系统研究。

## 核心贡献（创新点）
1. **提出VLA-Retain配对基准**：在每个模型自身基础检查点与微调检查点之间进行配对比较，分离目标增益与非目标变化，区别于既往工作的单点评估。
2. **引入四维度量体系**：定义目标增益、平均非目标增益、最差非目标增益和 episode 加权总体增益，避免单一聚合指标掩盖维度级衰退。
3. **系统揭示跨维度效应的普遍性与模型依赖性**：发现8/10配置出现非目标回归，且视觉模块参数占比与平均非目标增益呈负相关（Spearman ρ = −0.70）。
4. **实验验证四种缓解策略的模型特异性**：证明混合训练、微调范围、WiSE-FT权重插值、DELTA风格特征保留均不能一致地改善所有模型。

## 方法详解
**模型设定**：将VLA分解为视觉模块 $f_\theta$ 和下游组件 $g_\phi$，微调仅更新 $\theta \rightarrow \theta'$，冻结 $\phi$。

**评估度量**（公式3）：
- 目标增益 $G^{\mathrm{target}}_{m,c} = \Delta_{m,c,t}$
- 平均非目标增益 $G^{\mathrm{off}}_{m,c} = \frac{1}{|\mathcal{D}|-1}\sum_{d \neq t} \Delta_{m,c,d}$
- 最差非目标增益 $G^{\mathrm{worst}}_{m,c} = \min_{d \neq t} \Delta_{m,c,d}$
- 总体增益 $G^{\mathrm{all}}_{m,c} = \frac{\sum_d N_d \Delta_{m,c,d}}{\sum_d N_d}$（episode加权）

**微调配置**：
- 默认：Dual LoRA，rank-32作用于视觉backbone和connector，冻结下游
- 训练协议：AdamW BF16，batch size 32，12,000步，LoRA α=16，dropout=0
- 数据覆盖：Camera-view、Sensor-noise（目标特定），Mixed（完整LIBERO-Plus训练集，覆盖5/7维度）

**保留策略**：
- WiSE-FT：base与Dual-LoRA视觉权重插值，无额外训练
- DELTA-style：基于Taylor近似计算通道敏感度 $q_c$，构造注意力权重 $a_c$，添加特征锚定损失 $\mathcal{L}_{\mathrm{feature}} = \frac{1}{C}\sum_c a_c e_c$

**统计推断**：配对cluster bootstrap（20,000次重采样40个源任务），计算95%置信区间。

## 实验与结果
**数据集**：LIBERO-Plus评估集，10,030个episode，覆盖7个扰动维度（Camera view、Robot state、Language、Lighting、Background、Sensor noise、Object layout）和4个LIBERO任务套件。

**基准模型**：OpenVLA-OFT、π₀.₅、InternVLA-A1.5、NORA-1.5、X-VLA（视觉模块参数占比10.64%–41.26%）。

**核心结果**（Table 3）：

| 模型 | Camera-view G_target | G_off | Sensor-noise G_target | G_off |
|------|---------------------|-------|----------------------|-------|
| OpenVLA-OFT | **+39.3** | **+1.4** | +19.8 | +0.3 |
| π₀.₅ | +22.0 | −0.3 | +10.2 | −1.3 |
| InternVLA-A1.5 | +2.6 | −5.2 | −1.9 | −2.5 |
| NORA-1.5 | +28.7 | −10.1 | +45.3 | −1.0 |
| X-VLA | **+69.4** | **−4.6** | +32.8 | −2.2 |

**关键发现**：
1. **Finding 1**：目标增益不预测保留；8/10配置G_off为负；X-VLA获+69.4目标增益但平均损失4.6pp非目标；语言和目标布局衰退最普遍（9/10配置）
2. **Finding 2**：扣除原始LIBERO衰退后，所有Camera-view微调模型仍至少在一个非目标维度上衰退
3. **Finding 3**：Mixed训练提升所有模型总体增益（X-VLA +13.6pp最高），但每种模型仍在至少一个维度上衰退
4. **Finding 4**：DELTA-style对X-VLA提升G_off从−4.6到+1.2，但对OpenVLA-OFT降至−23.1；Backbone-only优于Dual LoRA仅适用于X-VLA

**最强结果**：X-VLA + DELTA-style获得最高总体增益+10.9pp；OpenVLA-OFT在所有策略下保持正非目标增益。

## 相关工作脉络
1. **鲁棒性基准（LIBERO-Plus、VLA-Arena、LIBERO-X）**：评估固定模型的鲁棒性特征，本文聚焦微调后的变化向量而非绝对分数。
2. **目标适配方法（Li et al. 2026、VLAGuard、Zhang et al. 2026c）**：报道目标域增益，本文补充非目标影响的系统测量。
3. **参数高效适配（LoRA）**：本文采用LoRA作为基础适配方案，但揭示其跨维度效应的模型依赖性。
4. **保留方法（WiSE-FT、DELTA）**：本文将其应用于VLA视觉微调场景，证明保留策略的有效性高度依赖模型架构。
5. **VLM4VLA（Zhang et al. 2026b）**：识别视觉模块为VLA核心瓶颈，与本文聚焦视觉微调的切入点一致。

## 局限性与未来方向
1. **仅评估视觉模块微调**：未探索下游组件（LLM trunk、action decoder）的微调对跨维度效应的影响。
2. **基准限于仿真环境**：LIBERO模拟器的扰动类型和幅度可能与真实部署存在差距。
3. **两种目标域的泛化性**：仅研究Camera-view和Sensor-noise，其他维度（语言、物体布局等）的目标微调效应未知。
4. **超参数搜索有限**：仅比较有限的微调范围/保留策略组合，未全面扫描超参数空间。
5. **未来方向**：探索架构设计与跨维度保留能力的内在关系；开发模型自适应的保留策略；延伸至真实机器人部署验证。

## 研究启发与可借鉴点
1. **配对评估设计**：base-tuned配对比较有效分离适应效应与架构基线差异，可迁移至其他模型的适配研究。
2. **四维度量体系**：目标/平均非目标/最差/总体增益的组合避免单一指标的掩盖效应，适合多维鲁棒性评估。
3. **源域控制实验**：通过original-LIBERO对照组排除通用遗忘干扰，增强因果推断的可靠性。
4. **模型依赖性的启示**：视觉模块参数占比与保留能力负相关（ρ=−0.70），提示架构设计需权衡目标适应与跨维度稳健性。
5. **DELTA-style的VLA适配**：将源网络特征保留思想移植到VLA视觉微调，提供了一条可复用的保留策略实现路径。

## 关键术语表
**VLA（Vision-Language-Action）**：将视觉观测、语言指令和机器人状态映射到可执行动作的多模态机器人控制模型。
**跨维度效应（Cross-dimension effects）**：针对某一扰动维度的微调对其它扰动维度鲁棒性的影响。
**LIBERO-Plus**：包含7个受控扰动维度的VLA鲁棒性基准数据集，扩展自LIBERO。
**Dual LoRA**：同时对视觉backbone和connector应用rank-32 LoRA的微调配置。
**DELTA-style**：基于Taylor近似计算通道敏感度，构造注意力加权特征锚定损失的保留策略。
**WiSE-FT**：通过插值base和微调后视觉权重以改善鲁棒性的后处理策略。
**paired cluster bootstrap**：按源任务族重采样的配对Bootstrap方法，量化任务采样不确定性。
**视觉模块参数占比（Visual share）**：视觉backbone加connector参数占模型总参数的比例。

## 可复现要素
- **数据集**：LIBERO-Plus公开可用（Fei et al. 2026）
- **代码**：https://github.com/BoruiXu/VLA_vision_tuning_benchmark（训练、合并、服务、评估代码及episode级结果）
- **模型检查点**：五个开源VLA的基础检查点公开可用
- **关键超参**：LoRA rank=32, α=16, dropout=0；batch size=32；12,000 optimizer steps；BF16精度
- **评估工具**：vla-evaluation-harness（Choi et al. 2026）
- **统计方法**：20,000次paired cluster bootstrap over 40 source tasks
