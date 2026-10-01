---
title: "ROBOFL-FEDERATED-EXPERT-ASSEMBLY-FOR-WORLD-ACTION-MODELS"
source: https://arxiv.org/pdf/2609.34968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:18:26"
field: "联邦机器人学习"
keywords: ["联邦学习", "世界-动作模型", "Mixture-of-Experts", "参数高效微调", "机器人操作", "跨路径蒸馏"]
innovations: ["任务隔离联邦专家组装协议MOSAIC，分离局部专长形成与全局路由学习", "跨路径FARD蒸馏：利用理解-生成共识引导动作路由", "路径一致性PCEA聚合：Hellinger重心加权完整专家残差重构全局适配器"]
benchmarks: ["RoboTwin 2.0", "RLBench", "Franka Real-World"]
---

# 论文速读：ROBOFL

## 一句话总结
ROBOFL 提出一种基于任务隔离的联邦专家组装框架 MOSAIC，将各客户端独立训练的 LoRA 适配器直接部署为服务端 MoE 专家分支，通过跨路径路由蒸馏（FARD）和路径一致性专家聚合（PCEA）机制，在保留各机构数据隐私的同时实现异构任务专长融合；在 RoboTwin 2.0 和真实 Franka 机械臂上均超越现有联邦及中心化 PEFT 基线。

## 研究问题与动机
1. **物理交互数据稀缺且分布孤岛化**：高质量机器人轨迹采集成本高、需要硬件与专家遥操作，导致数据在各机构间呈任务隔离状态，难以集中共享。
2. **联邦适配聚合引发专家能力退化**：对客户端独立训练的 LoRA 适配器进行朴素平均会混淆不兼容更新，而将 MoE 式路由纳入联邦聚合则会稀释局部 specialization 并导致路由不一致。
3. **非独立同分布（Non-IID）异构性加剧优化困难**：各客户端训练任务家族基本互不相交，标签分布差异大，导致局部最优相互漂移，简单平均会抹平专有领域知识。
4. **通信与计算资源约束**：现有 MoE 联邦方法要求客户端训练多专家子网并上传路由参数，造成通信开销巨大（如 FedVLA 需 2.3 GiB），难以部署到资源受限机构。

## 核心贡献（创新点）
1. **任务隔离的联邦专家组装协议 MOSAIC**：客户端仅训练单个 LoRA 适配器并上传，服务端将其直接安装为固定专家槽位，通过服务端路由器学习跨专家的输入依赖组合，而非平均参数——与 FedVLA 等方法的本质区别在于分离了局部专长形成与全局路由学习。
2. **洞察引导的动作路由蒸馏 FARD**：利用理解路径（U）与生成路径（G）的路由共识构建 detached teacher，以 Jensen-Shannon 散度蒸馏至动作路径（A）路由器——与现有单路径路由方法相比，本质区别是引入三路径一致性约束而非仅依赖单一模态信号。
3. **路径一致性专家聚合 PCEA**：收集三路径解耦的 Hellinger 重心证据，以此加权完整专家更新并重构为低秩全局适配器，用于个性化再分配——与 FedAvg 或 FedMoE 的逐参数平均相比，本质区别是 preserve 完整专家残差结构并通过跨路径共识进行加权。
4. **极端非独立同分布场景下的实测验证**：在 8 机构任务隔离的 Franka 机械臂实验中，ROBOFL 以单适配器通信开销（40.85M 参数）超越中心化 InternVLA-A1 达 12.23%——与 FedVLA 等基线相比，通信开销降低最高 86.81%。

## 方法详解
**整体架构**：基于三路径世界-动作模型（World-Action Model, WAM），包含理解（U）、生成（G）、动作（A）三条路径，通过统一掩码自注意力耦合。每条路径在每一层保留独立参数，但共享注意力计算。

**Adapter Slotting（适配器槽位化）**：第 $\tau$ 轮中，客户端 $k$ 上传其 LoRA 因子 $(A_{m,k}^{\tau,0}, B_{m,k}^{\tau,0})$，直接覆写模块 $m$ 的专家槽位 $k$，不进行任何平均：
$$
(A_{m,k}^{\tau,0}, B_{m,k}^{\tau,0}) \leftarrow (A_{m,k}^{\mathrm{loc},\tau}, B_{m,k}^{\mathrm{loc},\tau})
$$
客户端与槽位的映射固定，但 token 到专家的路由保持灵活。

**Routed Expert Integration（路由专家集成）**：对于 token 表示 $z$，适配后的投影为：
$$
h_m(z) = W_m^0 z + \gamma_m \sum_{e=1}^{E} \pi_{m,e}(z) B_{m,e} A_{m,e} z, \quad \gamma_m = \alpha_m / r
$$
服务端联合精炼选择路由器与各专家因子，共享 action-head 权重均匀平均后同步回各客户端。

**Global Adapter Conversion（全局适配器转换）**：令 $\Delta W_{m,e}^{\mathrm{srv},\tau}$ 为服务端精炼后的完整专家残差，PCEA 通过解耦路由证据将其转换为低秩约束全局适配器。

**Personalized Redistribution（个性化再分配）**：客户端 $k$ 接收的初始参数为：
$$
\Delta W_{m,k}^{\mathrm{init},\tau+1} = \Pi_r \left( \frac{1}{2} \Delta W_{m,k}^{\mathrm{srv},\tau} + \frac{1}{2} \Delta W_{m}^{\mathrm{glob},\tau} \right)
$$
其中 $\Pi_r$ 为截断 SVD 近似（秩 $\leq r$），实现个性化参数聚合。

**FARD 设计**：定义几何均值 teacher 及其可靠性：
$$
a_{m,b} = \sum_e \sqrt{q_{m,b,e}^U q_{m,b,e}^G}, \quad r_{m,b} = a_{m,b}\left[1 - \frac{H(q_{m,b}^{UG})}{\log E}\right]_+
$$
蒸馏损失为：
$$
\mathcal{L}_{\mathrm{FARD}} = \mathrm{mean}_m \left[ \frac{1}{|\mathcal{V}_m|} \sum_{b \in \mathcal{V}_m} \mathrm{sg}(r_{m,b}) D_{\mathrm{JS}}(\mathrm{sg}(q_{m,b}^{UG}), q_{m,b}^A) \right]
$$
服务端总目标：
$$
\mathcal{L}_{\mathrm{MoSAIC}} = \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{gen}}\mathcal{L}_{\mathrm{gen}} + \lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}} + \lambda_{\mathrm{FARD}}^{(\tau)}\mathcal{L}_{\mathrm{FARD}}
$$

**PCEA 设计**：收集三路径解耦 Hellinger 证据：
$$
g_{m,b,e} = \left(\frac{\sqrt{q_{m,b,e}^U} + \sqrt{q_{m,b,e}^G} + \sqrt{q_{m,b,e}^A}}{3}\right)^2
$$
计算共识加权全局权重并重构：
$$
\Delta W_m^{\mathrm{glob},\tau} = \Pi_r \left(\sum_e \widetilde{w}_{m,e} \Delta W_{m,e}^{\mathrm{srv},\tau}\right)
$$

## 实验与结果
**数据集与基准**：
- RoboTwin 2.0：50 个双机械臂操作任务，8 客户端任务隔离（7 个 server 任务）
- RLBench：8 个单视角任务，4 客户端 + 2 个 server 任务
- Franka 真实机械臂：6 个单臂操作任务，4 客户端 + 2 个 server 任务

**关键超参**：
- 预训练权重：InternVLA-A1-3B
- LoRA rank $r=16$, $\alpha=32$（effective scaling $\alpha/r=2$），dropout=0.05
- 通信轮数：100 轮，每轮客户端 500 步（RoboTwin）/ 200 步（RLBench）/ 300 步（Franka）
- 客户端 batch size=32，服务端 batch size=8/16/16
- 优化器：AdamW，LR 从 $1\times10^{-4}$ 余弦衰减至 $1\times10^{-5}$

**主要结果**：
| 平台 | ROBOFL | FedAvg | 中心化 InternVLA-A1 | 相对提升 |
|------|--------|--------|---------------------|----------|
| RoboTwin 2.0（50任务） | **83.12%** | 78.92% | 81.60% | +4.20% vs FedAvg, +1.52% vs 中心化 |
| RLBench（8任务） | **41.25%** | 34.25%（ForgeVLA最佳FL） | 52.75% | FL最佳，略低于中心化 |
| Franka 真实机械臂 | **59.17%** | - | 46.94% | **+12.23% vs 中心化** |

**通信效率**：客户端每轮通信 311.63 MiB（40.85M 参数），相比 FedMoE（1.2 GB）和 FedVLA（2.36 GB）降低最高 **86.81%**，GPU 内存峰值从 15.59 GiB 降至 8.33 GiB。

**消融实验**：Vanilla MoSAIC（81.94%）→ +FARD（82.42%）→ +FARD+PCEA（83.12%），PCEA 将 ≥95% 成功率任务从 14 提升至覆盖更多高成功率任务。

## 相关工作脉络
1. **FedVLA（Miao et al., 2025）**：结合指令解析、双门控 MoE 与专家驱动聚合的联邦 VLA，但客户端需训练多专家子网并上传路由——ROBOFL 本质区别在于客户端仅训练单适配器，路由仅在服务端学习。
2. **FedMoE（Mei et al., 2024）**：联邦个性化学习中的异构 MoE，各客户端维护专属子 MoE——ROBOFL 优势在于极低客户端通信开销（降低 86.81%）。
3. **ForgeVLA（Zhou et al., 2026）**：针对无标注视觉-动作日志的联邦 VLA，通过指令恢复与对比规划处理弱监督——ROBOFL 定位为有标注任务隔离场景，侧重跨路径路由对齐。
4. **LoRA-MoE（Wu et al., 2024）**：在适配器间进行路由的 MoE 架构，但未考虑联邦设置中的隐私与异构性——ROBOFL 引入任务隔离协议与跨路径蒸馏机制。
5. **Centraled PEFT WAMs（InternVLA-A1, Motus）**：集中式参数高效微调基准——ROBOFL 在 RoboTwin 2.0 和 Franka 上达到或超越中心化性能，同时保持联邦隐私保护。
6. **FLoRA / FRLoRA（Wang et al., 2024; Yan et al., 2025）**：低秩聚合与共享研究——ROBOFL 不直接聚合低秩因子，而是通过完整专家残差的共识加权转换。

## 局限性与未来方向
1. **单一骨干网络架构**：所有实验使用单一 InternVLA-A1-3B 与单一对齐 MoT 架构，未探索跨架构泛化。
2. **任务异构性假设严格**：聚焦任务家族互不相交场景，跨 embodiment（不同机器人形态）与跨模态异构性仍是开放问题。
3. **隐私保护非形式化**：仅交换适配参数，未提供差分隐私或安全聚合等 formal privacy guarantees，参数更新仍可能在对抗设置下泄露信息。
4. **单 seed 实验限制统计显著性**：仿真结果仅使用单个训练 seed 与评估 seed，缺乏多 seed 置信区间。
5. **低秩转换的信息损失**：PCEA 中的截断 SVD 可能导致专家特定成分衰减（如 Franka 实验中"stack cups"任务放置位置偏差）。

## 研究启发与可借鉴点
1. **路由-专家分离设计范式**：将"局部专长形成"与"全局路由学习"解耦的思路可迁移至其他联邦 MoE 场景，如多语言大模型联邦微调，避免跨语言专家的参数混淆。
2. **跨模态一致性蒸馏机制**：FARD 利用理解-生成共识引导动作路由的设计，可扩展至多模态联邦学习（如语音-文本-视觉任务隔离场景），作为路由正则化的通用技术。
3. **完整残差加权聚合**：PCEA 对完整专家更新进行共识加权而非逐参数平均的思路，适用于任何需要将多个低秩适配器组合为全局适配器的场景。
4. **极端非独立同分布实验协议**：任务完全隔离的 8 机构 Franka 设置可作为联邦机器人学习的基准测试协议，供后续工作比较。
5. **个人化再分配策略**：服务端精炼专家与全局适配器各取 50% 混合的个性化方案，可直接复用于其他联邦自适应场景（如医疗图像分析的机构个性化）。

## 关键术语表
**MOSAIC（Mixture of Slotted Adapters）**：将各客户端训练的 LoRA 适配器直接安装为服务端 MoE 固定专家槽位，服务端联合精炼路由器与专家参数的联邦组装协议。
**FARD（Foresight-to-Action Routing Distillation）**：利用理解路径（U）与生成路径（G）的路由几何均值共识作为 detached teacher，蒸馏至动作路径（A）路由器的跨路径对齐机制。
**PCEA（Path-Consensus Expert Aggregation）**：收集三路径解耦的 Hellinger 重心证据，以此加权完整专家残差并重构低秩全局适配器的聚合方法。
**Task-Silo 协议**：各机构仅训练所属任务家族数据的独立适配器、不上报原始轨迹的联邦学习组织模式。
**World-Action Model（WAM）**：统一感知、语言理解、未来状态预测与机器人控制的三路径架构（理解/生成/动作）。
**Flow-Matching Action Policy**：基于流匹配的损失函数，预测条件速度场以实现 action token 的隐式扩散生成。
**LoRA（Low-Rank Adaptation）**：通过低秩分解 $(A, B)$ 参数化权重增量 $\Delta W = BA$ 的参数高效微调方法。
**Hellinger-Barycenter**：基于平方根概率均值的重心距离度量，用于聚合多路径路由分布的几何平均。

## 可复现要素
- **数据集**：RoboTwin 2.0（公开）、RLBench（公开）、Franka 真实机械臂实验（自定义任务配置，公开于附录 Table 9）
- **代码开源状态**：论文声明计划作为 supplementary materials 开源源代码、启动配置与评估脚本
- **预训练权重**：InternVLA-A1-3B checkpoint
- **关键超参**：LoRA rank=16, α=32, dropout=0.05；λ_gen=0.01, λ_aux=0.001, λ_FARD=0.01；top-k routing=4（RoboTwin）/ 2（RLBench/Franka）；训练轮数=100；客户端步数=500/200/300；服务端步数=500/200/300；batch size=32/8-16；optimizer=AdamW（LR=1e-4→1e-5, β=(0.9,0.95), wd=0.01）
