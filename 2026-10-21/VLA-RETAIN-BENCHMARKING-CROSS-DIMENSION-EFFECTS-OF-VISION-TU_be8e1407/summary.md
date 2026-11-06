---
title: "VLA-RETAIN-BENCHMARKING-CROSS-DIMENSION-EFFECTS-OF-VISION-TU"
source: https://arxiv.org/pdf/2610.27698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:58:08"
field: "具身多模态学习"
keywords: ["VLA", "vision tuning", "cross-dimension effects", "robustness benchmark", "retention", "LoRA adaptation", "LIBERO-Plus"]
innovations: ["提出VLA-Retain配对基准测试，首次系统量化视觉微调的跨维度鲁棒性影响", "发现目标域增益与跨维度保持解耦，8/10模型-域配置出现负离靶增益", "证明DELTA-style等保持策略效果高度依赖模型架构，无通用最优策略"]
benchmarks: ["LIBERO-Plus", "VLA-Retain"]
---

# 论文速读：VLA-RETAIN: BENCHMARKING CROSS-DIMENSION EFFECTS OF VISION TUNING IN VLAS

## 一句话总结
本文提出VLA-Retain配对基准测试，系统评估了5个开源VLA模型在针对单一扰动维度（Camera-view或Sensor-noise）进行视觉模块微调后，对其余6个非目标维度鲁棒性的跨维度影响。研究发现目标域增益往往伴随非目标维度的性能衰退，且这种衰退不能用源域遗忘完全解释，跨维度保持能力因模型架构和训练策略而异。

## 研究问题与动机
1. **核心问题**：现有VLA适应性研究仅报告目标域增益，而鲁棒性基准仅评估未微调模型，二者均无法揭示"针对单一扰动的视觉微调如何影响模型在其他部署条件下的鲁棒性"这一跨维度效应。
2. **现有方法不足一**：当前 targeted adaptation 研究只关注目标域性能提升，忽略适应更新对非目标扰动（语言、物体布局、传感器噪声等）的潜在负面影响。
3. **现有方法不足二**：robustness benchmarks（如LIBERO-Plus、LIBERO-X）仅刻画固定模型的静态鲁棒性剖面，未考虑微调后视觉表征变化对下游所有条件的影响。
4. **评估挑战**：适应效应是多维的，需分离目标增益与离靶变化、隔离架构差异、排除通用遗忘、比较不同微调策略，单一指标或聚合评分均不足以刻画完整影响。

## 核心贡献（创新点）
1. **提出VLA-Retain配对基准**：针对5个开源VLA模型，在同一LIBERO-Plus评估集（10,030个episode、7个扰动维度）上对比base与tuned checkpoint，首次系统量化视觉微调的跨维度效应。
2. **发现目标增益与跨维度保持的解耦**：证明目标域增益不能预测非目标维度的表现，X-VLA在Camera-view上获得+69.4pp提升但平均损失-4.6pp，而OpenVLA-OFT同时获得+39.3pp目标增益和+1.4pp平均离靶增益。
3. **排除通用遗忘解释**：通过original-LIBERO控制和source-adjusted分析证明，非目标衰退不完全是源域遗忘的结果，而是扰动特定干扰。
4. **揭示模型依赖的微调策略选择**：DELTA-style特征保持在X-VLA上将平均离靶增益从-4.6提升至+1.2，但在OpenVLA-OFT上却从+1.4骤降至-23.1，表明无通用最优策略。

## 方法详解
1. **研究框架**：将VLA建模为 $a = g_\phi(f_\theta(x), \ell, s)$，其中 $f_\theta$ 为视觉模块（backbone+connector），$g_\phi$ 为下游多模态与动作组件；微调仅更新 $\theta \to \theta'$ 而冻结 $\phi$，但适配后的视觉模块需处理所有部署条件。
2. **评估指标设计**：定义变化向量 $\Delta_{m,c,d} = S_{m,c,d}^{\text{tuned}} - S_{m,d}^{\text{base}}$（单位：percentage points），并计算四个摘要指标——目标增益 $G^{\text{target}}$、平均离靶增益 $G^{\text{off}}$、最差离靶增益 $G^{\text{worst}}$、 episode加权整体增益 $G^{\text{all}}$。
3. **配对比较设计**：每个tuned checkpoint与其对应base checkpoint在完全相同的10,030个LIBERO-Plus episode上评估，消除基线鲁棒性差异的混杂效应。
4. **统计验证**：采用paired cluster-bootstrap（对40个source task重采样20,000次）计算95%置信区间，分离扰动特定衰退与任务采样噪声；添加original-LIBERO控制以排除通用遗忘。
5. **微调配置**：默认使用Dual LoRA（rank-32作用于visual backbone和connector），分别测试Camera-view、Sensor-noise、Mixed（覆盖5个维度的完整LIBERO-Plus训练集）三种训练数据；策略对比包括Backbone-only、Connector-only、WiSE-FT插值、DELTA-style特征保持。

## 实验与结果
1. **数据集**：LIBERO-Plus评估集共10,030个episode，涵盖7个扰动维度（Camera view: 1,599; Robot state: 1,550; Language: 1,537; Lighting: 1,142; Background: 1,076; Sensor noise: 1,601; Object layout: 1,525）。
2. **评估模型**：5个开源VLA——OpenVLA-OFT（SigLIP+DINOv2, 10.64%视觉参数占比）、$\pi_{0.5}$（SigLIP, 11.40%）、InternVLA-A1.5（Qwen3.5 ViT, 12.34%）、NORA-1.5（Qwen2.5-VL ViT, 16.77%）、X-VLA（DaViT, 41.26%）。
3. **核心结果一（Camera-view微调）**：
   - OpenVLA-OFT：目标+39.3pp，平均离靶+1.4pp，整体+7.5pp（唯一正离靶）
   - X-VLA：目标+69.4pp，平均离靶-4.6pp，最差-10.7pp
   - NORA-1.5：目标+28.7pp，平均离靶-10.1pp，最差-17.8pp
   - 8/10模型-域配置的 $G^{\text{off}}$ 为负
4. **核心结果二（Sensor-noise微调）**：
   - 同样4/5模型出现负离靶增益，范围-1.0至-2.5pp
   - OpenVLA-OFT再次保持+0.3pp平均离靶增益
5. **排除遗忘解释**：除NORA-1.5（-13.4pp）外，所有checkpoint在original-LIBERO上变化≤3.8pp；source-adjusted后所有Camera-view微调模型仍在至少一个维度回归。
6. **训练覆盖度**：Mixed训练提升所有模型的 $G^{\text{all}}$，但每个模型仍在至少一个维度衰退。
7. **策略对比**：
   - WiSE-FT插值改善保持但降低目标增益
   - DELTA-style对X-VLA有效（$G^{\text{off}}$: -4.6→+1.2），对OpenVLA-OFT灾难性（+1.4→-23.1）
   - 最优微调scope因模型而异

## 相关工作脉络
1. **LIBERO-Plus (Fei et al., 2026)**：提供7个扰动维度的鲁棒性基准，但仅评估固定模型，未涉及微调后的跨维度变化。
2. **LIBERO-X (Wang et al., 2026)**：分层测试空间泛化、物体识别和指令理解，关注模型架构特性而非适应效应。
3. **VLAGuard (Yin & Zhang, 2026) / Structure-Aware Robust FT**：针对物理attention劫持的防御性微调，仅报告目标域防御效果。
4. **Li et al. (2026) 低秩视觉更新**：通过LoRA恢复Camera-view鲁棒性，但未评估对非目标条件的溢出影响。
5. **WiSE-FT (Wortsman et al., 2022) / DELTA (Li et al., 2019)**：通用模型适应中的保持方法，本文将其适配至VLA视觉微调并验证跨维度有效性。
6. **定位差异**：本文不挑战目标域增益的正确性，而是补充评估"适配更新如何改变非目标条件的鲁棒性"，填补基准与方法的双重空白。

## 局限性与未来方向
1. **局限性一**：仅评估视觉模块微调，未探索多模态主干或动作解码器的适应性更新对跨维度效应的影响。
2. **局限性二**：LIBERO-Plus为仿真环境，实际部署中的物理传感器噪声、光照变化等可能呈现不同规律。
3. **局限性三**：训练数据覆盖范围受限（Mixed训练未涵盖robot state和object layout两个维度），更广泛的覆盖度效应尚待验证。
4. **未来方向一**：开发模型无关的跨维度保持策略，解决当前策略高度依赖架构的问题。
5. **未来方向二**：探索视觉表征的解耦与正交化，使目标域适应与通用鲁棒性更易分离。
6. **未来方向三**：将评估扩展至真实机器人部署场景，验证仿真基准结论的可迁移性。

## 研究启发与可借鉴点
1. **配对评估设计可迁移**：本文的"base-tuned pair on identical episodes"设计消除了基线差异混杂，适用于任何适应性方法的公平评估，可借鉴到团队的多模态模型微调研究中。
2. **多指标联合报告范式**：目标增益、平均离靶、最差离靶、整体增益四指标联合，避免单一指标掩盖退化，建议作为后续VLA适应性论文的标配报告方式。
3. **Source-adjusted分析框架**：通过减去original-task变化来分离扰动特定衰退与通用遗忘，可作为评估任何领域适应方法可靠性的标准控制手段。
4. **架构占比与保持能力的相关性**：发现视觉模块参数占比与平均离靶增益呈负相关（$\rho = -0.70$），提示大视觉占比模型可能更脆弱，值得在模型选择阶段纳入考量。
5. **策略选择需实证验证**：DELTA-style等保持方法效果高度依赖模型架构，团队在进行类似特征保持设计时应进行多模型交叉验证。

## 关键术语表
**VLA (Vision-Language-Action Model)**：将视觉观察、语言指令和机器人状态映射到可执行动作的多模态机器人控制模型，如OpenVLA、X-VLA等。

**Cross-dimension effect**：针对某一扰动维度进行适配后，模型在其余非目标扰动维度上的性能变化情况。

**Paired benchmark**：在完全相同的评估episode上对比base与tuned checkpoint的基准测试设计，消除基线鲁棒性差异。

**Source-adjusted change**：从离靶增益中减去original-LIBERO任务的变化，用于分离扰动特定衰退与通用遗忘。

**Dual LoRA**：对visual backbone和connector同时应用rank-32 LoRA的微调配置，冻结下游多模态和动作模块。

**WiSE-FT**：通过线性插值融合base与fine-tuned视觉权重以保持鲁棒性的方法（Wortsman et al., 2022）。

**DELTA-style feature preservation**：基于DELTA算法改编的特征保持策略，通过Taylor近似计算通道敏感度并施加加权正则化。

**Cluster bootstrap**：对source task进行重采样的配对bootstrap方法，用于计算评估指标的置信区间。

## 可复现要素
- **数据集**：LIBERO-Plus（公开，https://github.com/senyufe/LIBERO-Plus），包含7个扰动维度的训练/评估数据
- **代码**：公开于https://github.com/BoruiXu/VLA_vision_tuning_benchmark，含训练、合并、服务、评估代码及per-episode结果
- **模型权重**：5个VLA的base checkpoint和LIBERO-Plus数据均公开可用
- **关键超参**：BF16精度、effective batch size=32、12,000 optimizer updates、LoRA rank=32、$\alpha=16$、zero dropout、AdamW优化器
