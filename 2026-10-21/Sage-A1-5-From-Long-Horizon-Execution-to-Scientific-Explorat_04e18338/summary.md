---
title: "Sage-A1-5-From-Long-Horizon-Execution-to-Scientific-Explorat"
source: https://arxiv.org/pdf/2610.28578v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:23:26"
field: "智能体长周期推理与科学发现"
keywords: ["Agentic Model", "Long-horizon Reasoning", "Capability Composition", "Scientific Discovery", "Reinforcement Learning", "Multi-agent Orchestration", "Unified Post-training"]
innovations: ["提出自适应智能体执行框架，将任务编排与验证内化为模型能力而非依赖外部固定规则", "构建统一结果-过程后训练管线，动态分配多领域RL奖励与多教师过程监督权重", "展示从孤立任务求解到开放科学发现的范式转变，35B模型匹敌万亿参数级模型"]
benchmarks: ["FS-O", "FS-R", "BrowseComp", "SWE-Bench Pro", "MLE-Bench Lite", "HLE w/ tools", "CMT", "CritPt", "DeepResearch Bench II", "PinchBench"]
---

# 论文速读：Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition

## 一句话总结
Sage-A1.5 是一个 35B MoE 智能体模型，通过**自适应能力组合**（智能体编排 + 智能体验证）将长周期任务执行与开放科学探索内化为模型自身能力，在多项科学推理、搜索与工程基准上达到或超越同等规模模型，并在部分基准上匹敌万亿参数级模型。

## 研究问题与动机
1. **长周期执行的累积误差问题**：单智能体逐步执行时，上下文持续增长、探索缓慢、早期错误影响后续决策；更长的执行并不必然带来更好结果。
2. **外部规则依赖的局限**：任务委派、并行协作、证据审查与策略调整通常依赖外部框架的固定规则，而非模型的自主能力。
3. **单一任务求解能力不足**：软件工程、ML优化与科学研究需要模型管理相关子任务、比较不同方案、在新证据出现时修订决策。
4. **训练目标单一**：现有方法侧重"能力覆盖"（domain coverage），忽视"能力组合"（capability composition）——即能力如何在执行中被组织、协调与自适应。

## 核心贡献（创新点）
1. **提出自适应智能体执行框架（Adaptive Agentic Execution）**，将任务编排与证据驱动的自我修正作为模型训练的显式目标，与已有工作仅训练单任务求解能力形成本质区别。
2. **构建统一训练管线**，融合多领域SFT与混合奖励RL（outcome-process post-training），突破已有工作跨域能力割裂、难以统一的问题。
3. **展示从孤立任务求解到开放科学发现的范式转移**，通过多维科学案例（量子光学、计算化学、凝聚态物理等）证明模型可进行长周期、可检查的科学研究。

## 方法详解
**1. 自适应智能体执行（Capability Composition 维度）**
- **智能体编排（Agentic Orchestration）**：主智能体决定任务分解策略，将子任务委派给独立上下文的子智能体并行执行；支持两种组合模式：探索性组合（∨逻辑，多假设并行探索）与构建性组合（∧逻辑，互补子任务并行完成）；子智能体返回结构化反馈（结果+自我评估+支持证据+遗留问题）。
- **智能体验证（Agentic Verification）**：模型在推理过程中自主判断何时需要验证、验证什么、如何基于证据调整后续行动；训练时通过 Supervisor 在停滞/关键节点引入验证信号，推理时无需外部监督器；验证触发通过系统提示 `Verification protocol:enabled` 条件激活。

**2. 工程任务后训练**
- **MLE任务**：构建 Kaggle-MLE Factory 自动化流水线生成竞赛任务；采用窗口化交互强化学习（WIRL），将多次交互分组为共享结果的执行窗口；异步策略优化公式：`L(φ) = 1/|B| Σ_u 1/max(N_u,1) Σ_t m_t [w_t l_t(φ;R_u) + λ_KL K_t - λ_H H_t]`，其中 `l_t` 为 PPO clipped loss，`w_t` 为stop-gradient修正权重。
- **SWE任务**：基于 SWE-rebench-V2 与自建任务；严格泄露控制：移除暴露参考解的 Git 对象、强制流量经代理重定向；训练数据包含成功轨迹与过滤后的失败尝试。

**3. 科学研究三阶段管线**
- 科学问题筛选：基于难度（低/中/高）、任务类型（知识/推理/研究导向）、工具收益三维度评估；
- 工具链优化：隔离代码沙箱 + 工具结果回放机制；
- 轨迹蒸馏与训练：多维过滤（解正确性/推理深度/工具成功率）后对 Qwen3.5-35B-A3B 进行扩展多轮SFT。

**4. 长上下文推理的递归任务演化**
- 基于原子能力组合（证据定位、跨形式对齐、条件评估等）的十轴难度增强；
- 参考模型反馈引导递归演化，生成 3,512 道困难QA对（成功率≤0.75），最终保留 1,949 条轨迹用于SFT。

**5. 统一训练（Stage I + Stage II）**
- Stage I（多领域SFT）：`L_SFT(θ) = (1-ρ) E[q_task][l_θ] + ρ E[q_proto][l_θ]`，结合任务导向与协议导向数据；
- Stage II（统一结果-过程后训练）：`L_post(x) = λ_out(x) L_RL(x) + λ_proc(x) L_OPD(x)`，根据样本交互特征动态分配结果奖励与教师过程监督权重；OPD损失为反向KL：`L_OPD(x) = 1/N_τ Σ_t m_t D_KL(p_t || p_t^T)`。

**6. 训练基础设施（pVisor）**
- Agent原生沙箱框架：按需加载环境、内存卸载/恢复；实验显示卸载后驻留内存降低约80%。

## 实验与结果
**评测基准**（12项）：GAIA、BrowseComp、DeepResearch Bench II、XBench-DS-2510、MLE-Bench Lite、SciCode、SWE-Bench Pro/Multilingual、IFBench、AA-LCR、ALE-CLI、PinchBench、FS-O/FS-R、HiPhO、HLE w/ tools、CMT、CritPt、ChemCOTBench、MatTools、OmniDocBench。

**关键结果**：
- **科学推理**：FS-O 达 81.0（±2.0），FS-R 达 48.3（±2.9）；CMT 达 80.0，CritPt 达 20.0；均大幅优于同等规模模型 Ornith-1.5（FS-R仅6.7、CMT仅26.0）。
- **搜索能力**：BrowseComp 达 81.7（+6.2 vs A1），DeepResearch Bench II 达 43.7。
- **工程任务**：SWE-Bench Pro 达 61.2，MLE-Bench Lite 金牌率 50.0%（比基线 +34.8pp），Any Medal 达 72.7%。
- **通用能力**：PinchBench 达 90.0（所有35B及更大模型中最高），ALE-CLI 达 19.1（35B类别第一）。
- **Agentic Orchestration 消融**：BrowseComp 从 81.7 → 82.8（+1.1pp），DeepResearch 从 43.7 → 47.5（+3.8分）。
- **Agentic Verification 消融**：BrowseComp 从 81.0 → 85.5（+4.5pp），同时平均输出token减少5.52%、轮次减少3.15%，验证机制在提升性能的同时降低了推理成本。
- **科学教师模型**（Table 2）：FS-R 从 2.5 → 70.0，CMT 从 20.0 → 84.0，CritPt 从 0.9 → 25.7，提升极为显著。

## 相关工作脉络
1. **Agents-A1 [1]**：同为35B MoE智能体模型，通过扩展智能体视角（horizon scaling）而非参数缩放提升长周期任务性能；本文在其基础上引入能力编排与验证，实现从"单任务求解"到"能力组合"的跃迁。
2. **Kimi-K3 [4] / GLM-5.3 [6] / GPT-5.6-sol [5]**：万亿参数级闭源/开源模型；Sage-A1.5以35B参数在多项科学、搜索、工程基准上匹敌或超越这些大规模模型。
3. **ExoMind [20]**：受扩展心智启发的科学智能体系统；本文的科学问题筛选方法受其启发，但引入三维度评估（难度/类型/工具收益）与递归任务演化。
4. **SWE-bench [11] / MLE-Bench [15]**：软件工程与ML工程评测基准；本文构建的隔离环境、泄露控制与轨迹选择策略直接面向这两类基准的训练需求。
5. **On-Policy Distillation (OPD) [24] / Nemotron-Cascade 2 [25]**：教师-学生蒸馏框架；本文将其推广至多教师、多监督模式（结果/过程/混合）的统一训练管线。

## 局限性与未来方向
1. 当前编排与验证训练聚焦搜索与研究综合，超长轨迹仍积累较重上下文开销。
2. 统一后训练模型未能完全保留各领域教师的峰值性能（跨域蒸馏损失）。
3. 科学案例仍限于计算模拟与历史数据回放，尚未连接主动物理实验。
4. 未来方向：通过中间验证点压缩探索历史为阶段记录；扩展编排至更广泛工程工作流；与科学家合作连接真实实验验证。

## 研究启发与可借鉴点
1. **能力组合维度的训练目标设计**：将任务组织、并行委派、证据整合、策略修订作为显式训练目标，可为后续研究提供"超越单任务求解"的能力发展框架。
2. **统一结果-过程后训练（mixed outcome-process post-training）**：根据样本交互特征动态分配结果奖励与多教师过程监督权重，实现跨域能力融合而不陷入灾难性遗忘。
3. **递归任务演化方法**：通过原子能力组合定义难度增强轴、以参考模型反馈引导递归演化，可用于生成高质量的长上下文推理训练数据。
4. **验证触发的条件化设计**：通过系统提示开关（`Verification protocol:enabled`）按需激活验证能力，避免推理时的固定规则触发，提升灵活性。
5. **隔离环境的防泄露工程实践**：Git对象移除、代理重定向、独立评分沙箱等措施，可为智能体训练的数据完整性保障提供参考。

## 关键术语表
**Agentic Orchestration（智能体编排）**：模型组织多智能体执行围绕共享目标的能力，包括任务分解、并行委派、上下文隔离与结果集成。
**Agentic Verification（智能体验证）**：模型在长周期推理中自主识别验证时机、评估已有证据与探索策略、并据此调整后续行动的内生能力。
**Capability Composition（能力组合）**：与能力覆盖（domain coverage）并列的第二维能力发展，关注问题求解能力在执行过程中的组织、协调与自适应方式。
**Windowed Interaction Reinforcement Learning (WIRL)**：将多次工具交互分组为共享实验结果奖励的执行窗口，在此基础上进行策略优化的RL方法。
**On-Policy Distillation (OPD)**：在多教师框架下，以学生访问的相同上下文条件，对学生与教师下一步token分布施加反向KL散度的过程监督。
**Science-enhanced Teacher**：通过三阶段管线（问题筛选→工具链优化→轨迹蒸馏）增强的科学研究能力教师模型。
**Recursive Task Evolution（递归任务演化）**：基于原子能力组合与参考模型性能反馈引导的递归任务改写，用于生成难度递增的长上下文推理数据。
**pVisor**：Agent原生沙箱框架，支持按需加载环境、内存卸载/恢复，使智能体执行上下文与模型推理周期解耦。

## 可复现要素
- **数据集**：部分开源（SWE-rebench-V2、LongBlocks、HLE w/ tools 等），大量任务为自建（Kaggle-MLE Factory、科研问题集、真实工作流等）；训练数据未公开。
- **代码**：论文未提及代码开源。
- **权重**：Sage-A1.5 为 35B MoE（3B active）模型，论文未明确权重开源声明。
- **关键超参**：论文未详细列出具体数值（如学习率、epochs、KL系数等）。
