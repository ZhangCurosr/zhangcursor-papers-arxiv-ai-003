---
title: "TRIANGULAR-RESAMPLING-FOR-LONG-HORIZON-MOTION-GENERATION"
source: https://arxiv.org/pdf/2609.34697v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:13:30"
field: "长程运动生成"
keywords: ["long-horizon motion generation", "diffusion model", "triangular denoising", "rollout training", "distribution matching", "FloodDiffusion", "train-inference mismatch"]
innovations: ["提出三角重采样TR，将rollout训练扩展至三角去噪窗口的部分去噪状态，通过共享阈值GT钳制平衡漂移与一致性", "TR与DMD解耦设计，支持监督与分布匹配两种目标（TR-DMD）", "提出120秒序列×12窗口的时序AUC与退化斜率评测协议，量化长程一致性"]
benchmarks: ["HumanML3D 120s FID AUC", "HumanML3D Matching Distance AUC", "HumanML3D R-precision AUC"]
---

# 论文速读：TRIANGULAR-RESAMPLING-FOR-LONG-HORIZON-MOTION-GENERATION

## 一句话总结
本文提出三角重采样（Triangular Resampling, TR），一种针对三角去噪扩散模型的训练-推理失配后训练方法，通过共享去噪阈值将部分去噪状态锚定到真实运动，在 HumanML3D 上使 120 秒长程生成的 FID AUC 降低 40.9%。

## 研究问题与动机
1. **训练-推理失配**：FloodDiffusion 等三角去噪模型在训练时使用从 ground-truth 直接构造的活动窗口，而推理时反复基于自身预测更新状态，小误差沿序列累积导致长程漂移。
2. **部分去噪状态被忽视**：已有 rollout 训练方法（如 DART、MotionStreamer）仅替换已完成的运动历史，无法覆盖三角去噪窗口中正在去噪的部分去噪状态。
3. **自由 roll-out 易失稳**：直接暴露于模型自生成轨迹的训练会导致状态偏离真实运动分布，缺乏 ground-truth 锚点会严重劣化生成质量。
4. **长程一致性评估缺口**：现有方法多在片段级或过渡级评估，缺少对固定文本指令下持续生成质量的系统性时序评估。

## 核心贡献（创新点）
1. **提出三角重采样（TR）**：将 rollout 训练扩展到三角去噪窗口的全部活跃区域（含部分去噪状态），而非仅替换已完成历史；通过共享去噪阈值控制 ground-truth 锚定与模型预测的过渡，这是与 DART/MotionStreamer 的本质区别。
2. **GT 钳制（GT Clamp）机制**：引入一个每样本共享的去噪阈值 r，低于阈值的状态替换为噪声匹配的 ground-truth，高于阈值则保留模型预测；该设计平衡了 drift 控制与训练-推理一致性，区别于自由 roll-out 和随机位置钳制。
3. **支持监督与分布匹配两种目标**：TR 可扩展为 TR-DMD，在三角 replay 基础上结合 DMD 分布匹配目标（沿 Rolling Forcing 配方），实现非 DMD 与 DMD 两组内同时 SOTA。
4. **系统评估协议**：提出 120 秒序列 × 12 个 10 秒非重叠窗口的评估方法，以 FID AUC 和退化斜率为核心指标，填补长程一致性的量化评测空白。

## 方法详解
**背景（FloodDiffusion 三角去噪）**：活动窗口中每个 token j 的去噪进度由 $\alpha_j(\tau)=\mathrm{clip}(\tau-j/c,\ 0,1)$ 决定，较早 token 更干净、较晚 token 噪声更大，形成三角状去噪前沿；每次 Euler 步进后最前端 token 被提交，窗口滑动一位并注入新噪声。

**三角重采样构造（Section 4.1）**：
- 以概率 $\gamma$（重采样比）对样本进入 replay 区间，重置区间为高斯噪声，前端前缀保持干净。
- 对每个 replay 样本采样一个共享阈值 $r=\mathrm{sigmoid}(u+\log s)$，$u\sim\mathcal{N}(0,1)$，shift 参数 $s$ 控制钳制强度。
- 每步 Euler 更新后按阈值分叉：
$$
\tilde{z}_j^{k+1}=\begin{cases}
\alpha_j^{k+1}z_j^{\mathrm{GT}}+(1-\alpha_j^{k+1})\epsilon_j^{k+1}, & \alpha_j^{k+1}<r\\
z_j^{k+1}, & \alpha_j^{k+1}\ge r
\end{cases}
$$
- replay 状态detach后作为下一步优化输入；阈值 $s$ 小则早释放，大则长锚定。

**监督目标 TR（Section 4.2）**：从 replay 状态 $\tilde{z}$ 反推有效噪声 $\hat{\epsilon}_j=(\tilde{z}_j-\alpha_j z_j^{\mathrm{GT}})/\max(1-\alpha_j,\epsilon)$，得到速度目标 $v_j^\star=z_j^{\mathrm{GT}}-\hat{\epsilon}_j$，损失为
$$
\mathcal{L}_{\mathrm{TR}}(\theta)=\mathbb{E}\!\left[\frac{1}{|\mathcal{B}(\tau)|}\sum_{j\in\mathcal{B}(\tau)}\|v_{\theta,j}(\tilde{z},\alpha(\tau),c)-v_j^\star\|_2^2\right]
$$

**TR-DMD 目标**：在 replay 状态下以 detach 方式采样一个离散去噪阶段，用可微读取得到清洁预测 $\hat{z}_{0,j}=\mathrm{sg}(z_{\alpha_j})+(1-\alpha_j)v_{\theta,j}(\mathrm{sg}(z_\alpha),\alpha,c)$，再以 Rolling Forcing 的 DMD 更新配方优化 fake-score / generator。

## 实验与结果
- **数据集**：训练用 HumanML3D + BABEL；评测用 HumanML3D 测试集 256 条 prompt，生成 120 秒序列，分 12 个 10 秒非重叠窗口计算指标。
- **主结果（无 DMD 组）**：TR $(\gamma=0.25,s=0.6)$ 得 FID AUC = 1.153，较 TR-off（1.951）降 **40.9%**，退化斜率 0.235/min vs 0.526/min（**降 55.3%**）；Matching Distance AUC 3.786，R-precision AUC 0.655。
- **主结果（DMD 组）**：TR-DMD $(\gamma=1,s=0.6)$ 得 FID AUC = 1.324，优于 Rolling Forcing（1.351），为该组 SOTA。
- **消融**：完整自 roll-out（$s=0$）FID AUC 高达 16.024，证明 GT 锚定至关重要；$s=0.6$ 为最优；$\gamma=0.25$ 最优，$\gamma=1$ 次之，中间值 $\gamma=0.5$ 反而更差（非单调）；课程学习（curriculum）和随机位置钳制均不及固定 $s=0.6$。
- **人工评测**：TR 在 30 条 prompt 的成对偏好测试中，运动质量 BT 得分 0.302（TR-off: -0.272，Rolling Forcing: -0.030）；文本对齐 BT 得分 0.317（TR-off: -0.211，Rolling Forcing: -0.106），两项均为最高。

## 相关工作脉络
1. **FloodDiffusion (Cai et al., 2026)**：本文底座模型，三角去噪窗口 + 双向注意力；TR 不改变骨架，仅优化后训练阶段的状态分布。
2. **DART (Zhao et al., 2025)**：运动原语级 rollout，渐进 curriculum 从 GT 历史过渡到全 rollout；差异：DART 替换已完成历史，TR 覆盖部分去噪状态。
3. **MotionStreamer (Xiao et al., 2025)**：Two-Forward 逐个 latent 生成替换；差异：TR 在三角窗口的联合去噪前沿上做 GT 钳制，不依赖逐 token 生成。
4. **Resampling Forcing (Guo et al., 2025)**：视频领域的 teacher-free self-resampling；差异：TR 应用于三角去噪且含共享阈值钳制，后者仅 resample 噪声-corrupted GT frames。
5. **Self Forcing / Rolling Forcing (Huang et al., 2025; Liu et al., 2026)**：视频 DMD 类方法，前者用短 roll-out + 截断梯度，后者用交错噪声滚动窗口；差异：TR 专为 FloodDiffusion 三角日程设计，共享阈值贯穿整个 replay。
6. **TEACH / DoubleTake / FlowMDM / T2LM**：长程运动生成的组合或位置编码方案；差异：本文聚焦训练态分布优化而非生成架构。

## 局限性与未来方向
1. **计算开销**：多步 replay 使后训练耗时约增加 3.3×（单步 1.95s vs 0.58s），限制大规模扩展。
2. **仅验证于 FloodDiffusion 骨架**：未泛化到其他 denoising schedule 或其他 backbone（如 causal DiT）。
3. **超参敏感**：$\gamma$ 与 $s$ 的耦合关系尚未完全解析，课程学习提升有限。
4. **DMD 变体(TR-DMD)未在监督基础上进一步提升**：TR-DMD 的 FID AUC（1.324）不如同等 $\gamma=1,s=0.6$ 的监督 TR（1.251），DMD 兼容性未带来额外保真增益。

## 研究启发与可借鉴点
1. **共享阈值 GT 钳制范式可迁移**：任何具有"部分完成 + 部分噪声"结构的滚动窗口模型（如视频流式扩散）均可套用此思想，用统一阈值平衡 drift 与训练-推理一致。
2. **时序 AUC 与退化斜率指标**：将长序列按窗口切片再聚合 AUC/斜率，是评估长程一致性的有效方案，可直接复用于视频/音频长程生成任务。
3. **$\gamma$ 非单调现象启示**：过高 replay 率可能导致模型过度拟合已 drift 的状态，过低则信号不足；类似"curriculum-like"选择可能普适，值得在其它 rollout 训练场景中验证。
4. **与 DMD 的模块化组合**：TR 的 replay 构造与优化目标解耦，便于替换不同目标（flow-matching、score-matching、DMD 等），为多目标后训练框架提供模板。

## 关键术语表
- **Triangular Resampling (TR)**：一种后训练方法，通过 replay 三角去噪窗口并施加 GT 阈值钳制来缓解长程误差累积。
- **FloodDiffusion**：基于三角去噪日程的流式运动生成扩散模型，使用双向注意力联合去噪活动窗口。
- **GT Clamp（Ground-truth 钳制）**：将去噪进度低于共享阈值的 latent 替换为噪声匹配的原始运动，防止 replay 状态偏离过远。
- **Shift 参数 $s$**：控制阈值分布位置的对数位移，越大则更多状态保留 GT 锚定，越小则更早释放给模型 rollout。
- **FID AUC**：FID 沿生成时间轴的归一化曲线下面积，用于综合衡量长程生成质量。
- **Degradation Slope**：FID 随时间窗线性拟合的斜率，反映质量退化速率。
- **Rolling Forcing**：视频 DMD 方法，使用交错噪声的滚动窗口 + 自 forcing 更新进行分布匹配。
- **Bradley-Terry Score**：成对偏好测试中用于汇总人类选择的对数强度分值，正值表示偏好该方法。

## 可复现要素
- **数据集**：HumanML3D（公开）+ BABEL（公开）；263-dim 表示，20 fps。
- **代码/权重**：论文声明将开源实现、训练配置、frozen prompt manifest、checkpoint 标识符及评测脚本；当前未提供直接链接。
- **关键超参**：TR 主配置 $(\gamma=0.25,\ s=0.6)$；TR-DMD $(\gamma=1,\ s=0.6)$；后训练 30k 步；TR-DMD 额外 1200 次 DMD 迭代；EMA decay=0.99；fake-score 学习率 $4\times10^{-7}$，generator $1.5\times10^{-6}$；Batch size=64。
- **硬件**：TR 单卡 NVIDIA H200；TR-DMD 双卡 H200。
- **Inference**：120 秒序列，NFE 与原始 FloodDiffusion 一致（论文未明确单步 NFE，沿用官方设置）。
