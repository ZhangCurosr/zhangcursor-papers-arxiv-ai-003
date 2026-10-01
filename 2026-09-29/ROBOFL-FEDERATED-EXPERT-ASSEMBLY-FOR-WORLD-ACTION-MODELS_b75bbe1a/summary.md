---
title: "ROBOFL-FEDERATED-EXPERT-ASSEMBLY-FOR-WORLD-ACTION-MODELS"
source: https://arxiv.org/pdf/2609.34968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:18:37"
field: "联邦机器人学习"
keywords: ["federated learning", "world-action models", "mixture-of-experts", "parameter-efficient fine-tuning", "robotic manipulation", "vision-language-action"]
innovations: ["将客户端LoRA适配器直接安装为服务器MoE专家槽位，避免朴素平均稀释专业化", "利用理解-生成路径共识蒸馏指导动作路由器，提升跨路径路由一致性", "基于三路径Hellinger中点聚合专家残差，重建低秩全局适配器用于个性化重分配"]
benchmarks: ["RoboTwin 2.0", "RLBench", "Franka real-world"]
---

# 论文速读：ROBOFL: FEDERATED EXPERT ASSEMBLY FOR WORLD ACTION MODELS

## 一句话总结
ROBOFL 提出了一种面向世界动作模型（WAM）的联邦专家组装框架，通过将各客户端独立训练的 LoRA 适配器直接安装为服务器 MoE 的固定专家槽位，结合跨路径路由蒸馏（FARD）与路径一致性专家聚合（PCEA），在保持客户端单适配器通信成本的前提下，实现了异构任务环境下的高效联邦学习。

## 研究问题与动机
1. **物理交互数据稀缺且昂贵**：机器人演示需要硬件平台、专家遥操作、重复重置、安全监控与同步多模态感知，高质量轨迹既稀缺又具有机构专有价值。
2. **数据隐私与所有权约束**：机器人轨迹可能暴露私人环境、用户日常习惯及专有操作流程，高获取成本使机构不愿集中共享原始数据。
3. **朴素联邦聚合失效**：在任务级强非 IID 场景下，本地最优解漂移严重，直接平均客户端 LoRA 更新会混合仅在各自任务族内有效的变换，稀释专业化。
4. **MoE 路由联邦化困难**：各客户端在窄任务分布上独立训练的 MoE 专家分支会获得异构专业化，平均专家参数稀释知识，平均路由器则破坏已有分配，导致性能退化。

## 核心贡献（创新点）
1. **任务隔离联邦专家组装（MOSAIC）**：客户端仅训练并上传单个 LoRA 适配器，服务器将其直接安装为固定专家槽位构成层式 MoE，路由学习完全在服务器端进行。与 FedVLA/FedMoE 的本质区别在于客户端无需维护多专家结构，通信开销降至单适配器级别。
2. **视野到动作的路由蒸馏（FARD）**：利用理解（U）与生成（G）路径的离体共识构建教师分布，通过 Jensen-Shannon 散度蒸馏至动作（A）路径路由器。与直接训练动作路由器的本质区别在于借助三路径互补证据提升路由稳定性。
3. **路径一致性专家聚合（PCEA）**：基于三路径 Hellinger 中点证据对服务器精炼后的完整专家残差进行加权聚合，重建为低秩全局适配器用于个性化重分配。与直接平均 LoRA 因子的本质区别在于保持 ΔW 结构完整性后再进行低秩投影，避免因子分解引入的近似误差。

## 方法详解
**MOSAIC 核心机制**：
- **适配器槽位安装**：第 τ 轮客户端 k 上传其 LoRA 因子 $(A_{m,k}^{\tau,0}, B_{m,k}^{\tau,0})$，直接覆盖模块 m 的槽位 k，无平均操作。客户端-槽位映射固定，专家因子每轮刷新，路由器持久保留于服务器。
- **路由专家集成**：前向传播公式为 $h_m(z) = W_m^0 z + \gamma_m \sum_{e=1}^{E} \pi_{m,e}(z) B_{m,e} A_{m,e} z$，其中 $\gamma_m = \alpha_m / r$，$\pi_m$ 为 renormalized top-$k_{route}$ 权重。服务器在混合任务数据上联合精炼路由器与专家因子。
- **全局适配器转换**：定义服务器精炼残差 $\Delta W_{m,e}^{srv,\tau} = \gamma_m B_{m,e}^{srv,\tau} A_{m,e}^{srv,\tau}$，经 PCEA 转换为低秩全局适配器 $\Delta W_m^{glob,\tau}$。
- **个性化重分配**：客户端 k 接收 $\Delta W_{m,k}^{init,\tau+1} = \Pi_r(\frac{1}{2}\Delta W_{m,k}^{srv,\tau} + \frac{1}{2}\Delta W_{m}^{glob,\tau})$，其中 $\Pi_r$ 为截断 SVD 近似（秩 ≤ r）。

**FARD 路由蒸馏**：
- 定义几何平均教师分布：$a_{m,b} = \sum_e \sqrt{q_{m,b,e}^U q_{m,b,e}^G}$，$q_{m,b}^{UG} = \frac{\sqrt{q_{m,b}^U q_{m,b}^G}}{a_{m,b}}$。
- 可靠性度量：$r_{m,b} = a_{m,b}[1 - \frac{H(q_{m,b}^{UG})}{\log E}]_+$，结合路径一致性与分布集中度。
- 蒸馏损失：$\mathcal{L}_{FARD} = \text{mean}_m[\frac{1}{|\mathcal{V}_m|} \sum_{b \in \mathcal{V}_m} \text{sg}(r_{m,b}) D_{JS}(\text{sg}(q_{m,b}^{UG}), q_{m,b}^A)]$，梯度仅通过动作分布反向传播。

**PCEA 聚合**：
- 收集离体 Hellinger 证据：$g_{m,b,e} = (\frac{\sqrt{q_{m,b,e}^U} + \sqrt{q_{m,b,e}^G} + \sqrt{q_{m,b,e}^A}}{3})^2$。
- 跨步骤累加：$s_{m,e} = \sum_b g_{m,b,e}$，$w_{m,e} = s_{m,e} / \sum_j s_{m,j}$，$\bar{a}_m = \sum_e s_{m,e} / N_m$。
- 计算全局权重 $\bar{w}_e$ 与匹配权重 $\widetilde{w}_{m,e} = \frac{(\sqrt{w_{m,e}} + \sqrt{\bar{w}_e})^2}{\sum_j(\sqrt{w_{m,j}} + \sqrt{\bar{w}_j})^2}$。
- 转换公式：$\Delta W_m^{glob,\tau} = \Pi_r(\sum_e \widetilde{w}_{m,e} \Delta W_{m,e}^{srv,\tau})$，实现使用 QR 约化 + 小 SVD。

**服务器目标函数**：$\mathcal{L}_{MOSAIC} = \mathcal{L}_{action} + \lambda_{gen}\mathcal{L}_{gen} + \lambda_{aux}\mathcal{L}_{aux} + \lambda_{FARD}^{(\tau)}\mathcal{L}_{FARD}$，其中 $\lambda_{FARD}$ 采用 10 轮 warmup 调度。

## 实验与结果
**数据集与设置**：
- **RoboTwin 2.0**：50 任务，8 客户端（每客户端 3-8 任务）+ 1 服务器组（7 任务），100 通信轮次。
- **RLBench**：8 任务，4 客户端 + 1 服务器组（2 任务），4 专家 / top-2 路由。
- **Franka 真实机器人**：6 任务，4 客户端 + 1 服务器组，3 轮 × 20 次试验。
- 所有实验从 **InternVLA-A1-3B** 初始化，LoRA rank=16, α=32, dropout=0.05。

**主要结果**：
| 基准 | ROBOFL | 最佳联邦基线 | 集中式参考 | 关键提升 |
|------|--------|-------------|-----------|---------|
| RoboTwin 2.0 整体 | **83.12%** | FedAvg 78.92% | InternVLA-A1 81.60% | +4.20% vs FedAvg, +1.52% vs 集中式 |
| RLBench 整体 | 41.25% | ForgeVLA 34.25% | InternVLA 52.75% | 最优联邦方法，低于集中式 |
| Franka 真实任务 | **59.17%** | ForgeVLA 35.56% | InternVLA 46.94% | +12.23% vs 集中式, +23.61% vs ForgeVLA |
| 通信开销（RoboTwin） | 40.85M 参数 / 311.63 MiB | FedVLA 312.93M / 2362.85 MiB | — | 客户端通信↓86.81%, GPU内存↓46.6% |

- **消融**：Vanilla MoSAIC（81.94%）→ +FARD（82.42%）→ +FARD+PCEA（83.12%），FARD 提升 ≥90% 成功率任务覆盖率，PCEA 进一步提升 ≥95% 覆盖率。
- **路由耦合分析**：指令替换产生强路由-动作耦合，FARD 使路由更选择性但不导致策略脆化。
- RLBench 上 ROBOFL 低于集中式基线，归因于任务多样性不足、互补证据有限。

## 相关工作脉络
1. **FedVLA（Miao et al., 2025）**：联合指令场景解析、双门控 MoE 与专家驱动聚合，但客户端需训练多专家 MoE 并通信路由器，ROBOFL 将路由移至服务器、客户端仅通信单适配器。
2. **FedMoE（Mei et al., 2024）**：客户端级子 MoE 个性化联邦学习，但本地 MoE 训练导致专家专业化稀释，ROBOFL 通过固定槽位保持原始专家完整性。
3. **ForgeVLA（Zhou et al., 2026）**：解决弱/缺失语言监督的联邦 VLA，采用指令恢复与对比规划，但未处理 MoE 路由一致性与低秩聚合结构保持问题。
4. **Mixture of LoRA Experts（Wu et al., 2024）**：单模型内路由适配器，非联邦设置；ROBOFL 将其扩展至联邦场景并引入跨路径一致性约束。
5. **FLoRA / FRLoRA / FedSA-LoRA**：研究低秩聚合与共享策略，但未利用多路径路由共识指导聚合权重，ROBOFL 的 PCEA 提供结构化聚合替代方案。
6. **FedAvg（McMahan et al., 2017）**：标准权重平均基线，ROBOFL 在其基础上通过专家保留与路由学习实现显著超越（RoboTwin +4.20%）。

## 局限性与未来方向
1. **单一骨干与架构**：所有实验仅使用 InternVLA-A1-3B 与单一对齐 MoT 架构，跨架构泛化能力待验证。
2. **同质性假设**：仅针对任务异构性，跨 embodiment、跨传感器套件与跨动作空间的联邦学习仍是开放问题。
3. **统计显著性不足**：仿真结果基于单一训练/评估种子，真实机器人结果基于固定脚本协议，缺乏多种子置信区间。
4. **无形式化隐私保证**：仅交换适配器参数而非原始轨迹，但未提供差分隐私或安全聚合等隐私保护机制，参数更新在对抗设置下仍可能泄露信息。
5. **未来方向**：跨 embodiment 联邦、多种子统计评估、隐私增强机制、动态槽位分配策略。

## 研究启发与可借鉴点
1. **跨路径一致性路由蒸馏**：FARD 利用理解-生成共识引导动作路由的思路，可迁移至其他多路径模型（如 VLA、WAM）的路由稳定性问题，尤其适用于存在互补证据路径的架构。
2. **残差结构保留聚合**：PCEA 保持 ΔW 结构完整性再低秩投影的方法，可应用于其他需要聚合低秩适配器的联邦场景，避免因子分解近似误差。
3. **职责分离的联邦 MoE 设计**：客户端仅训练单适配器、服务器负责复杂组装与路由学习的架构，在资源受限的联邦机器人学习中具有高度可移植性，可推广至边缘设备 federated fine-tuning。
4. **Hellinger 中点作为一致性度量**：PCEA 使用的几何平均与 Hellinger 中点可作为多分布融合的有效工具，适用于路由权重聚合、专家选择一致性度量等场景。

## 关键术语表
**World-Action Model (WAM)**：统一感知、语言理解、未来状态预测与机器人控制的策略模型，如 InternVLA-A1，采用理解-生成-动作三路径对齐架构。

**MOSAIC (Mixture of Slotted Adapters)**：将客户端独立训练的 LoRA 适配器直接安装为服务器 MoE 固定专家槽位的组装机制，保持本地专业化同时实现服务器端路由学习。

**FARD (Foresight-to-Action Routing Distillation)**：利用理解路径与生成路径的离体共识构建教师分布，通过 JS 散度蒸馏至动作路径路由器的跨路径路由对齐机制。

**PCEA (Path-Consensus Expert Aggregation)**：基于三路径 Hellinger 中点证据对服务器精炼专家残差进行加权聚合，重建为低秩全局适配器用于个性化重分配的方法。

**Task-Silo Protocol**：机构间任务家族隔离的联邦学习协议，每个客户端拥有 disjoint 的任务集，形成强非 IID 但语义清晰的联邦场景。

**LoRA (Low-Rank Adaptation)**：参数高效微调方法，通过低秩矩阵分解 $\Delta W = BA$ 近似权重更新，大幅降低通信与存储开销。

**Mixture-of-Experts (MoE)**：由多个专家网络与门控路由器组成的稀疏激活架构，输入-dependent 选择专家子集进行计算。

**Hellinger 距离**：用于度量概率分布相似性的距离度量，PCEA 中用于构建三路径路由共识的几何平均。

## 可复现要素
- **数据集**：RoboTwin 2.0（公开）、RLBench（公开）、Franka 真实机器人实验（内部数据集）
- **代码/权重**：论文声明计划作为补充材料发布源码、启动配置与评估脚本；预训练权重 InternVLA-A1-3B 公开；模型权重未声明开源
- **关键超参**：LoRA rank=16, α=32, dropout=0.05, λ_gen=0.01, λ_aux=0.001, λ_FARD=0.01（10轮warmup），top-k路由=4（RoboTwin）/2（RLBench/Franka）
- **优化器**：AdamW, LR=1e-4→1e-5 cosine decay, betas=(0.9,0.95), weight_decay=0.01, gradient norm clipping=1.0
- **通信设置**：100 通信轮次，客户端/服务器步数=500/200/300（RoboTwin/RLBench/Franka），客户端 batch=32，服务器 batch=8/16/16
