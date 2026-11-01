---
title: "SAFE-PERSISTENT-AND-EVOLVING-AGENT-HAR-NESS-FOR-UNDERSTANDIN"
source: https://arxiv.org/pdf/2610.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:21:35"
---

# 论文速读：SAFE-PERSISTENT-AND-EVOLVING-AGENT-HAR-NESS-FOR-UNDERSTANDIN

## 一句话总结
本文提出 E-Ledger，一个面向部分可观测企业工作流的安全持久化多智能体 Harness，并结合 WorldAbduct 归纳演化框架，通过四视角轨迹诊断与靶向环境验证，将隐含规则动态写入静态世界账本，在 WOW 基准上将安全任务完成率（STCR）较最强基线提升 5–15 个百分点，且策略安全率（PSR）保持 95%+。

## 研究问题与动机
1. **策略合规硬约束**：企业动作受严格权限、审批流与跨记录约束支配，本地看似合理的操作可能因未满足前置条件而违规。
2. **部分可观测性与隐藏副作用**：工具反馈仅暴露交互表层，跨记录联动、延迟效应或级联状态变更往往不在响应中体现，导致状态易过时。
3. **长 Horizon 状态追踪需求**：多实体、多读写任务要求持续维护已确认事实与未竟子目标，信息丢失会引发重复查询或错误执行。
4. **现有 Harness 演化局限**：既有方法多依赖文本提示蒸馏或仅记录可观测轨迹，缺乏对工具接口之外隐含转换规则的主动发现与实证机制。

## 核心贡献（创新点）
1. **E-Ledger 多智能体安全 Harness 架构**：将运行时控制（策略代码化、动态账本）与环境特定知识解耦，通过不可绕过的代码审批层确保动作合规；与依赖模型自身推理的 ReAct 等基线相比，其将自然语言约束转化为确定性可执行门控，从根本上消除 “reasoning loophole”。
2. **WorldAbduct 四视角归纳演化框架**：从状态一致性、世界观察缺口、策略门正确性、目标判断四个正交视角诊断执行轨迹，同时输出直接 Harness 更新与隐含规则假设；区别于 MemoHarness 的文本蒸馏或 WorldEvolver 的静态库构建，该方法通过假设-验证闭环主动扩展世界模型。
3. **隐藏规则溯因验证机制**：设计独立的 Hidden-Rule Abduction Agent，在新鲜沙盒中为每条假设设计靶向交互序列，仅当环境证据支持时才将规则 $r:(a,s,c)\longrightarrow\Delta s$ 写入静态账本 $\mathcal{W}_S$；与直接采纳 LLM 生成规则的方法相比，显著降低错误假设引入导致的安全风险。
4. **跨领域泛化验证**：在企业主观benchmark WOW 及科学发现环境 ScienceWorld、DiscoveryWorld 上验证，证明该范式对组织规则与物理隐式动力学均适用。

## 方法详解
- **任务建模**：环境建模为 POMDP $M=(U,P,S,A,T,O,\Omega)$，目标函数 $J(\tau)=C_U(\tau)\prod_{t=0}^{L-1}P(s_t,a_t)$，要求轨迹既达成目标（或安全拒绝不可行任务），又全程满足策略 $P(s_t,a_t)=1$。
- **World Ledger 双层结构**：
  - 静态层 $\mathcal{W}_S$：存储已验证的隐含规则与跨记录副作用，格式为 $r:(a,s,c)\longrightarrow\Delta s$。
  - 动态层 $\mathcal{W}_D$：维护当前任务相关的证据支持型状态快照，增量更新并保留冲突字段以待后续验证。
- **多智能体协同执行流程**：
  1. Task Agent 基于 $U,P,A,\mathcal{W}_S,\mathcal{W}_D^t,h_t$ 输出动作 $a_t$ 及预测状态变化 $\widehat{\Delta s}_t$。
  2. Policy-Context Agent 从 $\mathcal{W}_D^t$ 提取策略所需的证据字段（如用户活跃工单数、资产成本）。
  3. Code Approval Layer 将文本策略编译为确定性 Python 代码 $P_{\text{code}}$，返回 `{execute, query, probe, block}` 四类决策；非 execute 决策返回 Task Agent 修正。
  4. State-Tracking Agent 依据工具反馈 $o_{t+1}$ 与 $\mathcal{W}_S$ 更新 $\mathcal{W}_D^{t+1}=\mathcal{R}_\theta(\mathcal{W}_D^t,a_t,o_{t+1},\mathcal{W}_S)$，保留支持证据并记录冲突。
- **四视角演化诊断**：
  - 状态一致性 $\mathcal{I}_s=\mathcal{E}_\phi(\widehat{\Delta s}_t,o_{t+1},\mathcal{W}_D^{t+1},\mathcal{W}_S)$：核对预测、反馈与账本更新是否自洽。
  - 世界观察缺口 $\mathcal{I}_o=\mathcal{E}_\phi(\tau,P_{\text{code}},\mathcal{W}_D)$：扫描完整轨迹，定位导致策略敏感字段发生未记录变化的可疑前置动作，产出隐含规则假设。
  - 策略门正确性 $\mathcal{I}_p=\mathcal{E}_\phi(\tilde{a},\mathcal{W}_D,P_{\text{code}})$：审计被拦截的非执行提案，修复上下文绑定或提示词缺陷。
  - 目标判断 $\mathcal{I}_g=\mathcal{E}_\phi(U,P,\tau,\rho_{\text{eval}})$：基于评估器信号诊断过早终止、目标遗漏或误判。
- **溯因验证循环**：假设进入 Abduction Agent，在独立环境执行 ReAct 序列，每步输出 $\{act, revise, verified, unresolved\}$；仅 `verified` 结果以规则、适用条件与证据形式沉淀至 $\mathcal{W}_S$。

## 实验与结果
- **数据集与划分**：主基准 WOW（50 任务，10 类型，2:1:2 划分）；泛化验证 ScienceWorld（150 任务）与 DiscoveryWorld（40  episode，Easy）。训练轨迹驱动演化，验证集选优，测试集冻结评估。
- **评估指标**：WOW 报告 STCR、TCR、PSR、Steps、Cost；ScienceWorld 报告 Success、Score、Steps；DiscoveryWorld 报告 Success、Procedural、Steps。
- **基线**：ReAct、MemoHarness（六组件提示词蒸馏）、WorldEvolver（可观测状态变化库）。
- **核心结果（GLM-5.3-Flash）**：WorldAbduct 达到 STCR 75.0%、TCR 80.0%、PSR 95.0%，Steps 20.9，Cost $0.52。相较最强基线 MemoHarness，STCR 提升 10 个百分点（65.0→75.0）。
- **跨 Backbone 表现**：在 GPT-5.6-Terra、Kimi-K3、GLM-5.3-Flash、DeepSeek-V4-Pro 上均取得最高 STCR，较 MemoHarness 整体提升 5–15 个百分点；WorldAbduct 的安全优势主要来源于经过环境验证的隐含规则将 PSR 稳定维持在 95%+。
- **消融结论**：移除 $\mathcal{W}_S$ 导致 STCR 从 75.0% 骤降至 40.0%；移除溯因验证（Without Abduction）使 PSR 从 95.0% 降至 85.0%，验证环节直接保障安全性；World Observation Gap 单独使用即可达到 95.0% PSR，但与 Goal Judgment 等
