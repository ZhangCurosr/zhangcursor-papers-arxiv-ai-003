---
title: "PartiCam-Camera-Controlled-Video-Generation-with-Reward-Guid"
source: https://arxiv.org/pdf/2609.39504v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 03:01:39"
field: "视频生成与相机控制"
keywords: ["camera-controlled video generation", "sequential Monte Carlo", "particle filtering", "reward-guided diffusion", "training-free guidance", "novel view synthesis"]
innovations: ["全局SMC与局部PF-Restart双层无训练框架，避免单轨迹漂移", "审美/相机奖励分离策略，全局保多样性、局部保几何精度", "对训练型基线（TrajectoryCrafter等）提供测试时补救，缓解OOD失效"]
benchmarks: ["Tanks and Temples静态NVS基准", "NVSSolver动态单目视频重拍基准"]
---

# 论文速读：PartiCam-Camera-Controlled-Video-Generation-with-Reward-Guid

## 一句话总结
本文提出 PartiCam，一种无需训练的相机控制视频生成方法，通过结合全局 Sequential Monte Carlo (SMC) 轨迹探索与局部粒子滤波重采样（PF-Restart），在测试时引导预训练视频扩散模型生成严格遵循指定相机轨迹的高质量视频，同时显著降低相机漂移。

## 研究问题与动机
- **相机控制的稀缺性与高成本**：训练型相机控制方法依赖大规模配对数据，且微调模块强绑定特定骨干网络（架构、隐空间、条件接口），在新型骨干出现后快速过时；无训练方法虽骨干无关，但采样过程中的早期误差会持续累积。
- **Score Modulation 的累积漂移问题**：NVSSolver 等主流无训练方法仅在单条去噪轨迹上操作，早期噪声采样若将模型带入不良生成路径，后续步骤难以纠正，导致相机轨迹随时间漂移（表现为 ATE 偏高）。
- **随意重启的不稳定性**：Stochastic Restart 虽可帮助跳出不良轨迹，但无约束重启会丢弃去噪过程中积累的有用结构，造成生成质量剧烈波动且方差极大。
- **训练-free 作为数据生成器的价值**：可靠的测试时引导过程可生成带相机标注的视频对，进而蒸馏回模型权重，类比 LLM 的 Rejection Sampling Fine-tuning，形成"生成→过滤→训练"的正循环。

## 核心贡献（创新点）
- **全局 SMC + 局部 PF-Restart 的双层无训练框架**：与 NVSSolver 单轨迹调制不同，本文维护 K 个粒子轨迹群体并周期性地做全局重加权/重采样，同时在局部对每个粒子做奖励引导的 Noising-Denoising 重启，兼顾探索多样性与细粒度纠错。
- **双层奖励分离设计**：全局 SMC 使用审美奖励（Aesthetic Reward）保持粒子群体多样性，避免轨迹聚集；局部 PF-Restart 使用相机几何奖励（Camera Reward）精确纠正单条轨迹漂移，两者分离使几何精度与视觉质量取得最佳权衡。
- **对训练型基线同样有效**：与仅针对 NVSSolver 的工作不同，本文方法可直接叠加在 TrajectoryCrafter、ReCamMaster 等训练型相机控制方法之上，在困难 OOD 轨迹下显著缓解其视觉伪影与轨迹偏差。
- **SOTA 级定量与定性结果**：在静态（Tanks & Temples）与动态（单目视频）两个基准上均超越现有训练-free（NVSSolver）与训练型（TrajectoryCrafter、MotionCtrl）方法，静态场景 ATE 降至 0.526（相对 NVSSolver Post 提升约 32%），动态场景 ATE 降至 0.807（相对 NVSSolver Post 提升约 65%）。
- **不依赖可微奖励与反向传播**：与 TDS、Ψ-Sampler 等需对奖励求梯度的方法不同，PartiCam 兼容任意非可微奖励（如基于 VGGT 的相机估计误差），适配 CogVideo、Wan 等最新视频扩散骨干。

## 方法详解
- **问题形式化**：给定视频扩散模型 $p_\theta(\mathbf{z}_{0:T})$，目标是采样满足奖励 $R(\mathbf{x}_0)$ 的轨迹，奖励为 $\sum_i \alpha_i R_i(\mathbf{x}_t)$，其中 $R_i$ 可包含相机对齐项与美学项。
- **Score Modulation 基础**：继承 NVSSolver 框架，通过 Warping 输入视图至目标相机位姿并注入去噪更新，获得调制后分数，作为 SMC/PF-Restart 的底层传播算子。
- **全局 SMC 轨迹探索**：维护 $K$ 个粒子 $\{(\mathbf{z}_t^{(k)}, w_t^{(k)})\}$，每步按反向扩散转移 $p_\theta(\mathbf{z}_{t-1}|\mathbf{z}_t^{(k)})$ 传播，并按 Twisted Potential $\psi_t(\mathbf{z}_t) \approx \exp(\beta R_t(\hat{\mathbf{x}}_t))$ 计算增量权重 $ \tilde{w}_{t-1}^{(k)} = w_t^{(k)} \exp(\beta R_{t-1} - \beta R_t) $；当 ESS 低于阈值时执行系统重采样。
- **局部 PF-Restart 细化**：在指定时间步 $\mathcal{T}_{ref}$，对每个粒子 $\mathbf{z}_t^{(k)}$ 施加前向扩散加噪并随即去噪，得到 $N_r$ 个候选 $\tilde{\mathbf{z}}_t^{(k,n)}$；以相机奖励评分 $r_t^{(k,n)} \propto \exp(\beta R_t(\tilde{\mathbf{x}}_t^{(k,n)}))$ 做 Cat 分布采样替换原粒子，实现局部高奖励区域聚拢。
- **双层奖励配置**：默认全局用审美奖励（保留多样性）、局部用相机奖励（纠正漂移）；实验表明全局也换相机奖励可进一步降 ATE（0.28 vs 0.37）但牺牲视觉多样性（FID 略升）。
- **工程设置**：$K=2$、$N_r=2$、总步数 100；局部细化从第 8 步起至第 32 步，每 4 步执行一次、每次重复 4 轮；全局 SMC 从第 8 步至第 40 步、每 4 步一次；整体推理时间不高于 NVSSolver。

## 实验与结果
- **数据集**：静态场景采用 Tanks and Temples 六个场景加 NVSSolver 指定的三个场景；动态场景采用九个单目视频（城市与自然）。
- **评估指标**：相机精度使用 Particle-SFM 估计轨迹后计算 ATE、RPE-T、RPE-R；视觉质量使用 FID。
- **静态场景（Table 1）**：Ours 取得 FID=121.56、ATE=0.526、RPE-T=0.122、RPE-R=0.144，全面超越 NVSSolver Post（ATE=0.767, RPE-T=0.156）与 TrajectoryCrafter 等基线；相对 NVSSolver Post 的 ATE 降低约 32%，RPE-T 降低约 22%。
- **动态场景（Table 2）**：Ours 取得 FID=31.86、ATE=0.807、RPE-T=0.061、RPE-R=0.414，最强提升在平移误差上：RPE-T 仅 0.061，相对 NVSSolver Post（0.725）降幅达约 92%，ATE 相对 MotionCtrl 降幅约 76%。
- **骨干泛化（Table 3）**：在 CogVideo、SVD、TrajectoryCrafter 三个骨干上均取得稳定增益；尤其对已训练过的 TrajectoryCrafter，叠加后 ATE 从 0.767 降至 0.712，证明对 OOD 轨迹的泛化补救能力。
- **消融（Figure 5 / 正文）**：
  - 纯 Restart（无奖励）：ATE 1.09，方差大。
  - PF-Restart：$N_r=2$ 时 ATE=0.76；$N_r=4$ 时 ATE=0.54，FID 维持 114.15。
  - 纯全局 SMC（K=4）：ATE=1.03，RPE-T=0.31，FID 反降（135.84 vs 133.96），说明仅有群体多样性而无局部纠偏会损害质量。
  - 全组合（K=2, $N_r=2$）：ATE=0.37，RPE-T=0.09，FID=103.82，实现 6× 平移误差与 4× ATE 的联合下降。
  - 奖励设计：全局相机奖励可将 ATE 进一步压至 0.28（提升 24%），但 FID 从 102.51 略升至 103.06；默认配置在几何精度与视觉质量间取得更优平衡。
  - 与 SCG/TDS/Ψ-Sampler 对比：SCG 在审美奖励下 RPE-T=0.71、ATE=1.49 最差；相机奖励下仍仅 RPE-T=0.25、ATE=1.25；双级 SMC 架构是核心优势来源。

## 相关工作脉络
- **NVSSolver [78]**：单轨迹 Score Modulation 无训练相机控制基线，本文在其基础上叠加双层 SMC+PF-Restart，核心差异是从"单轨迹纠偏"升级为"群体探索+局部重采样"。
- **TrajectoryCrafter [82]**：基于 CogVideo 微调的训练型相机控制方法，在 OOD 困难轨迹上仍存在漂移与伪影；本文以测试时引导作为后处理，在不重新训练的前提下对其误差进行补偿。
- **MotionCtrl [66]**：统一运动控制框架，动态场景下对遮挡与大幅度轨迹偏差鲁棒性不足；本文的相机奖励能直接对齐 VGGT 估计的几何轨迹，显著提升 RPE-T。
- **TDS [70] / Ψ-Sampler [77]**：基于可微奖励与梯度回传的 SMC/DPP 采样，无法直接使用非可微相机奖励，且对 CogVideo/Wan 等大模型推理负担重；本文方案免梯度、兼容任意黑盒奖励。
- **SCG [35]**：单粒子符号化 Music 生成框架，仅依赖反向 SDE 采样缺乏群体多样性，实验显示其在相机任务上表现明显弱于本文双级架构。
- **Restart / RePaint / ZigZag [73][47][4]**：局部 annealing 机制，均缺乏跨粒子 reward-weighted 选择，本文将其形式化为粒子滤波重采样，赋予统计意义与稳定性保障。

## 局限性与未来方向
- **推理预算增加**：维持 K 个粒子与局部 $N_r$ 候选需多次去噪前向调用，尽管作者称整体仍控制在 NVSSolver 时长内，但在更长的视频或更高帧率下开销仍会放大。
- **奖励函数依赖人工调参**：相机奖励与审美奖励的权重 $\alpha_i, \beta$ 需任务适配；在遮挡严重、运动模糊的场景下，单一目视奖励可能不够鲁棒。
- **粒子数与性能的非单调关系**：消融显示 $N_r$ 从 2 增至 4 时 ATE 反而从 0.37 升至 0.46，说明局部重启预算与全局控制信号的配合尚未完全理解，缺乏自适应调度机制。
- **未来方向**：可扩展至其他条件生成任务（文本、姿态、深度），并与 Rejection Sampling Fine-tuning 结合形成"生成→过滤→蒸馏"的闭环；引入自适应 ESS 阈值与动态 $K/N_r$ 调度可进一步压缩无效计算。

## 研究启发与可借鉴点
- **双层 SMC 架构的通用性**：全局多样性 + 局部精细搜索的分离设计可移植到任何"单轨迹易漂移、奖励可黑盒评估"的扩散采样任务（如 3D 形态生成、物理仿真引导的图像合成）。
- **审美/几何奖励分离的实践价值**：在相机控制中全局保多样性、局部保几何精度的策略，为多目标奖励下的扩散引导提供了清晰的设计范式，值得在其他多约束任务中复现。
- **将训练型方法的 OOD 失败转化为测试时补救机会**：即便已有 TrajectoryCrafter 等强基线，本文仍能在困难轨迹上取得提升，提示团队在已有微调模型上做"推理时校准"是性价比极高的优化路径。
- **粒子数非线性收益的经验规律**：$K=2, N_r=2$ 为甜点而非越大越好，这对后续工作设定算力预算、设计自适应调度器具有直接参考价值。
- **与 Rejection Sampling Fine-tuning 的串联**：本文生成的相机标注视频可作为蒸馏数据，未来可尝试将 PartiCam 输出作为训练样本反哺下游模型，形成端到端正循环。

## 关键术语表
- **Particle Filtering**：通过维护一组带权重的样本（粒子）近似后验分布的序贯蒙特卡洛方法，在本文用于同步探索多条去噪轨迹并依奖励重采样。
- **Sequential Monte Carlo (SMC)**：在扩散逆过程中逐步对粒子群体进行传播、权重更新与重采样，使样本分布逐步逼近奖励加权的目标分布。
- **PF-Restart**：Particle Filter Restart 的缩写，指在每个粒子处通过短时加噪-去噪生成局部候选并用奖励选择替换，实现细粒度轨迹纠错。
- **Score Modulation**：在去噪步中对扩散模型的分数（score）进行调制，本文沿用 NVSSolver 的 Warping 方式将目标相机位姿作为先验注入。
- **Twisted Potential**：SMC 中的重要性加权因子 $\psi_t(\mathbf{z}_t) \approx \exp(\beta R_t)$，用于在每一步对粒子群施加奖励倾向。
- **Effective Sample Size (ESS)**：衡量粒子群体有效多样性的指标，当 ESS 低于阈值时触发重采样以避免粒子退化。
- **Aesthetic Reward**：衡量生成视频视觉质量的奖励函数，本文用于全局 SMC 阶段以保持粒子多样性。
- **Camera Reward**：基于 VGGT 估计轨迹误差的奖励函数，衡量生成视频与目标相机路径的几何对齐程度，用于局部 PF-Restart。

## 可复现要素
- **数据集**：静态场景使用 Tanks and Temples（6 场景）及 NVSSolver 指定的 3 个额外场景；动态场景使用 9 个单目视频；协议与 NVSSolver 一致。论文未提及私有数据。
- **代码开源**：论文未明确声明代码仓库链接（仅说明详见 Supplementary / Algorithm 1）。
- **模型权重**：基于 CogVideo [33]、SVD [7]、TrajectoryCrafter [82] 等公开骨干；奖励部分依赖 VGGT [65] 与 Particle-SFM [88]。
- **关键超参**：$K=2$（全局粒子数）、$N_r=2$（局部候选数）、扩散步数 100；局部细化启用区间 [8, 32]，全局 SMC 启用区间 [8, 40]，二者均每 4 步执行一次，局部每步重复 4 轮。
