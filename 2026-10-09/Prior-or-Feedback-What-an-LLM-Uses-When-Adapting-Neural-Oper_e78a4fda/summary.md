---
title: "Prior-or-Feedback-What-an-LLM-Uses-When-Adapting-Neural-Oper"
source: https://arxiv.org/pdf/2610.12325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:17:39"
field: "科学AI/神经算子适配"
keywords: ["neural operator", "LLM scientific agent", "hyperparameter optimization", "PDE adaptation", "controlled intervention", "decision-level verification"]
innovations: ["提出决策级验证协议，通过单一输入干预测量LLM配置选择变化", "证明LLM在有限预算下同时使用任务相关先验和实验反馈", "在PDEBench 20次试验预算下LLM性能全面优于随机搜索和TPE"]
benchmarks: ["PDEBench"]
---

# 论文速读：Prior or Feedback? What an LLM Uses When Adapting Neural Operators

## 一句话总结
论文在有限试验预算下研究LLM科学代理适配神经算子（FNO）时，是仅依赖初始任务先验还是会响应实验反馈。通过控制输入干预实验，作者证明LLM既利用任务描述形成有价值的冷启动先验，也能根据验证反馈调整后续配置选择。

## 研究问题与动机
1. 科学代理的决策是否真正响应其收集的实验证据？最终的benchmark分数仅衡量整体性能，无法揭示哪些输入塑造了每次决策。
2. 神经算子（如FNO）在目标物理 regime 偏移后性能下降，而适配最佳实践尚未确立，领域科学家缺乏机器学习调参专长。
3. 已有研究表明LLM的优势并非普遍存在：早期收益可能仅反映默认配置，经典优化器可超越独立LLM提议。
4. 在目标regime数据生成成本高、训练尝试次数有限的实际应用场景下，需要评估不同策略在固定小预算下的适配效率。

## 核心贡献（创新点）
1. **有限预算下的竞争性优化**：在相同20次试验预算下，LLM策略在所有36个匹配运行中均取得低于随机搜索的held-out测试nRMSE，在35/36中低于TPE。与已有工作相比，本文消除了默认配置的影响，所有策略均从零开始搜索。
2. **决策级验证协议**：提出对单一输入进行干预并测量动作距离的方法，而非仅观察最终性能差异。与Yagubyan（一致性度量）和Liao（证据来源敏感度测试）互补，本文同时测量变化大小而非仅判断是否变化。
3. **任务相关先验与反馈敏感的双重证明**：切换PDE描述使首次提议的base learning rate翻倍（p=5.2×10⁻⁹），而重新分配验证分数使下一次提议产生约一个类别坐标的变化（+0.085，p=0.015625），证明LLM同时使用任务先验和实验反馈。

## 方法详解
**实验设置**：
- 使用预训练的 Fourier Neural Operator (FNO)，架构为10个输入帧、12个傅里叶模式、宽度20、4层谱层；在PDEBench的1D标量PDE上进行适配。
- 目标PDE族：平流方程（∂ₜu + β∂ₓu = 0）和Burgers方程（∂ₜu + ∂ₓ(u²/2) = ν/π · ∂ₓₓu），共12个适配单元（6个族内+6个跨族），每单元3个随机种子。
- 共享动作空间：12维（8个数值坐标+4个类别坐标），包括优化器{AdamW, Adam}、调度{none, cosine, step}、base learning rate [10⁻⁵, 10⁻²]、4个参数块乘数、weight decay、rollout length {1,2,4,8}、data loss{MSE, rel L₂, MSE+rel L₂}、两个物理loss权重。
- 每次试验从源checkpoint的fresh copy开始，在750条轨迹上训练最多100个epoch，在350条轨迹的验证集上计算nRMSE。

**控制干预方法**：
- **冷启动描述测试**：在未观察到任何验证分数前，分别展示平流方程描述、Burgers方程描述或无描述，测量首次提议的base learning rate变化。
- **反馈重放**：重建已记录的决策点，重新生成下一次提议而不继续搜索。包括重分配干预（保留分数值但置换分配）、记号控制（保留所有分数及分配，仅改写科学记数法格式）和相同提示重采样控制。
- **动作距离度量**：d(a,a') = (1/D)(Σ|āⱼ-ā'ⱼ| + Σℓ[aⱼ≠a'ⱼ])，其中D=12，一个类别坐标变化贡献1/12≈0.083，决策阈值为0.05。

**LLM策略**：使用deepseek-v4-pro，开启thinking和JSON模式，reasoning_effort=max；每次试验采样S=3个JSON配置并执行medoid。

**统计检验**：使用Mann-Whitney检验（冷启动）和精确单侧符号翻转检验（反馈重放，6个单元的中位数，最小可达p=2⁻⁶=0.015625）。

## 实验与结果
**主要性能结果**（Table 2）：
- 在所有36个匹配运行中，LLM的held-out test nRMSE均低于随机搜索；在35/36中低于TPE。
- 最强提升出现在Burgers' ν=0.001跨族场景：LLM=0.01990 vs 随机搜索=0.07670（提升约3.8倍），跨族平均LLM表现显著优于基线。
- 即使在TPE扩展至25次评估（保持相同模型 Proposal 数量）的控制实验中，LLM仍在5/6个比较中胜出，两单元中位数均更低。

**冷启动先验质量**：
- LLM首次提议的验证nRMSE低于对应单元格中91.7%的60个随机搜索配置，仅2/36首次提议低于单元格随机中位数（Figure 3b）。
- 无适配的源checkpoint直接应用于目标 regime 时，nRMSE介于0.12–1.24之间，约为最差适配策略中位数的5–400倍。

**描述干预结果**：
- 切换显示描述从Burgers到平流，使两次实验的中位数base learning rate从0.0005翻倍至0.001（Figure 2a），Mann-Whitney检验 p=5.2×10⁻⁹。
- 平流描述下80%的提议选择0.001，而Burgers描述和无描述条件下仅为39%/40%。

**反馈重放结果**：
- 重新分配干预使中位数trace效应增加+0.085（bootstrap 95% CI [+0.073, +0.108]），接近一个类别坐标变化（1/12≈0.083），全部18个trace效应和6个单元中位数均为正，p=0.015625（Figure 2b）。
- 反应具有选择性：rollout length变化率49.6%、data loss 40.4%、schedule 25.9%，而优化器变化率仅7.8%（与重采样率14.4%接近）。

## 相关工作脉络
1. **LLM用于贝叶斯优化**：Yang et al. (2024)、Zhang et al. (2023)、Liu et al. (2024)、Agarwal et al. (2025) 将LLM用于直接提议超参配置或与经典优化器交互。本文与之定位不同：本文测量问题描述和验证历史对下一步具体配置的影响（而非仅搜索性能），且无默认配置起点。
2. **LLM优势的边界研究**：Rodrigues et al. (2026)、Ferreira et al. (2026)、Huai et al. (2026) 发现LLM优势并非普遍存在，早期增益可能源于默认配置。本文所有策略均从零开始，排除了默认配置的混淆因素。
3. **硬件感知代码优化的先验研究**：Redko et al. (2026) 研究LLM在代码优化中的先验效应，但操纵的是代码内容而非PDE描述和验证历史。
4. **PDE科学代理**：Wuwu et al. (2025) (PINNsAgent)、He et al. (2025) (Lang-PINN)、Li et al. (2026) (CodePDE)、Wang et al. (2026) (OpInf-LLM) 利用LLM构建PINN工作流或生成代码。本文固定架构和下游任务，仅研究配置策略。
5. **神经算子适配与预训练**：McCabe et al. (2023, 2026) (Walrus)、Herde et al. (2024) (Poseidon)、Hao et al. (2024) (DPOT) 通过扩大预训练覆盖来提升泛化。本文保持预训练checkpoint固定，研究有限预算下的配置效率。
6. **行为可复现性与审计**：Yagubyan (2026) 用结构化动作距离测量LLM一致性；Liao (2026) 测试证据来源敏感性；本文结合两者设计，同时测量变化大小。

## 局限性与未来方向
1. 研究范围局限于单个LLM（deepseek-v4-pro）和两种1D PDE族，结论需在不同模型族、科学领域、高维PDE和更大预算下验证。
2. 干预仅证明LLM的决策取决于任务描述和观测分数，但未解释决策机制，未证明响应的合理性或物理理解，也未证明这种依赖是否解释了性能优势。
3. 仅验证了决策响应性，未评估响应是否"合理"或符合领域知识。
4. 未来方向：将决策级验证协议推广到更多模型和家庭，探究响应性质的解释机制，以及是否与 Held-out 性能优势直接相关。

## 研究启发与可借鉴点
1. **决策级验证协议可直接迁移**：本文对单一输入进行控制干预并测量动作距离的方法，可应用于其他科学AI代理的行为审计，无需访问模型内部。
2. **冷启动先验质量评估**：首次提议在随机搜索池中的排名分布可作为评估LLM"常识"或任务迁移能力的新指标，值得纳入基准测试设计。
3. **反馈敏感性的坐标级别分析**：区分不同动作维度（rollout length、data loss、schedule vs optimiser）对干预的响应差异，为理解LLM决策逻辑提供细粒度洞察。
4. **固定架构+配置搜索的范式**：在数据昂贵、训练次数受限的场景下，固定模型架构仅搜索配置的策略具有实际应用价值，可迁移至其他物理模拟模型适配任务。
5. **Medoid采样策略**：从S=3个采样配置中选择medoid的做法平衡了多样性和稳定性，可作为LLM超参提议的稳定输出机制。

## 关键术语表
**Neural Operator (神经算子)**：学习函数空间之间映射的神经网络，可处理不同分辨率的PDE解，代表作为FNO。
**FNO (Fourier Neural Operator)**：基于傅里叶变换的神经算子架构，通过谱层学习PDE解算子。
**nRMSE (normalised Root Mean Squared Error)**：归一化均方根误差，用于衡量预测解与真实解之间的误差。
**Cold-start prior (冷启动先验)**：LLM在未观察到任何验证分数时，仅凭任务描述形成的初始配置偏好。
**Adaptation cell (适配单元)**：配对一个源checkpoint和一个目标regime的实验配置，本文共12个（6个族内+6个跨族）。
**Action distance (动作距离)**：衡量两个配置之间差异的标准化度量，D=12维中一个类别坐标变化贡献1/12≈0.083。
**Reassignment intervention (重分配干预)**：保留评估过的配置和验证分数但置换其分配的干预，用于测试反馈敏感性。
**PDEBench**：公开的PDE科学机器学习基准数据集，包含多种PDE族和预训练checkpoint。

## 可复现要素
- **数据集**：PDEBench（公开，[Takamoto et al., 2022c,a,b]），预训练checkpoint和轨迹数据已发布。
- **代码**：搜索代码开源在 https://github.com/julian-8897/budgeted-search-ai4science；运行记录归档于 doi:10.5281/zenodo.23213042。
- **模型**：deepseek-v4-pro（ hosted API），system fingerprint: fp_9954b31ca7_prod0820_fp8_kvcache_20260402。
- **关键超参**：B=20次试验，S=3次采样，TPE使用5次random startup trials；base learning rate范围[10⁻⁵, 10⁻²]，4个block multipliers范围[0.5, 2.0]。
- **评估**：750条轨迹训练，350条验证，1000条PDEBench test prefix用于held-out评估。
