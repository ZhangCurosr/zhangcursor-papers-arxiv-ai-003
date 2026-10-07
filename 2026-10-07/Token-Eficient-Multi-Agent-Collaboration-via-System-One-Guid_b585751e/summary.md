---
title: "Token-Eficient-Multi-Agent-Collaboration-via-System-One-Guid"
source: https://arxiv.org/pdf/2610.08155v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:54:19"
field: "多智能体系统效率优化"
keywords: ["Multi-Agent Systems", "Token Efficiency", "System One", "LLM Orchestration", "Computational Division of Labor"]
innovations: ["提出System One控制器与LLM推理的解耦架构，将MAS协调决策从生成式LLM中分离", "设计决策-证据闭环机制，通过轻量级reader定位evidence支持有界协调决策", "引入累积性通信预算约束B_src控制重复传输开销"]
benchmarks: ["MMLU", "GSM8K", "AQuA", "MultiArith", "SVAMP", "HumanEval", "WebSRC"]
---

# 论文速读：Token-Eficient-Multi-Agent-Collaboration-via-System-One-Guid

## 一句话总结
提出 S1-MAS 框架，利用预训练 System One 模型（Laya）作为轻量级协调控制器，将多智能体系统中的有界协调决策与开放式推理解耦，在保持高准确率的同时显著降低 GPT token 消耗与端到端延迟。

## 研究问题与动机
- **协调决策的 Token 开销瓶颈**：现有 LLM-based MAS 将任务选择、角色分配、消息路由等协调操作委托给强大 LLM，随着交互次数增加产生大量不必要的 token 消耗。
- **有界决策与开放推理的本质差异**：MAS 中的协调操作（如选择下一个任务、判断证据是否充分）本质上是有界决策问题，无需自由形式生成，用 LLM 处理是计算资源的浪费。
- **现有方法的局限性**：AgentVerse、DyLAN、SelfOrg 等方法虽提升了协作灵活性，但仍依赖 LLM 生成控制指令，未解决协调开销问题。
- **System One 模型的潜力**：预训练的 System One 模型专为受限动作空间中的快速结构化决策设计，有望替代 LLM 执行协调任务。

## 核心贡献（创新点）
- **计算劳动分工视角**：首次将 MAS 中的协调与推理视为可分离的计算劳动，提出有界协调决策与开放式 LLM 推理的解耦框架。
- **S1-MAS 框架**：引入轻量级 System One 控制器（Laya）+ 轻量级证据检索器（Qwen3-0.6B）+ LLM workers 的三层架构，实现无需任务特定训练的自适应协作。
- **决策-证据闭环机制**：通过 inspection query → evidence localization → task selection 的闭环，使协调决策基于已有交互证据而非盲目生成。
- **通信预算控制**：设计累积性 source token 预算 $B_{src}$，限制重复传输历史响应造成的上下文处理开销。
- **全面的效率-性能平衡**：在 7 个 benchmark 上实现 state-of-the-art 准确率的同时，较基线减少 44.9%–97.2% token 消耗与 37.8%–93.0% 延迟。

## 方法详解
- **状态定义**：系统状态 $S_t = (q, \mathcal{H}_t, \mathcal{A}_t, B_{src} - u_t)$，包含问题、归档的历史任务-响应对、检查记录、剩余通信预算。
- **检查查询生成**：Controller 从未检查的问题条件集合 $\mathcal{C}(q)$ 中选择目标，通过与当前候选答案配对形成 inspection query $\chi_t$。
- **证据定位**：Lightweight reader（Qwen3-0.6B）在工作者响应中定位与检查条件相关的证据片段，生成 evidence record $\ell_t$。
- **可行任务构建**：确定性编译器根据状态 $S_t$ 和 $\chi_t$ 筛选满足前置条件且符合剩余预算的任务，形成可行菜单 $\mathcal{W}_t$。
- **任务操作类型**：Solve（独立求解）、Decompose（分解子问题）、Reverse（逆向推理）、Review（审查验证）、Repair（修复缺陷）、Contrast（对比选择）。
- **System One 引导的选择**：Controller 使用 $S_t, \chi_t, \ell_t$ 从 $\mathcal{W}_t$ 中选择下一任务 $j_{t+1}$ 或终止决策 finish。
- **角色与来源分配**：选定任务决定工作者角色及可访问的源响应，localized evidence 作为 attention cue 高亮相关内容。
- **通信预算约束**：$\sum_{i=1}^{t} \mathrm{tok}(\tilde{e}(j_i)) \leq B_{src}$，限制传输的 source token 总量。
- **自适应停止与答案交付**：当 finish 可选时 Controller 可终止协作；delivery rule 基于多数投票或最新响应返回归档答案，无需额外 LLM 调用。

## 实验与结果
- **数据集**：MMLU（知识推理）、GSM8K/AQuA/MultiArith/SVAMP（数学推理）、HumanEval（代码生成）、WebSRC（结构化网页问答）。
- **基线方法**：Vanilla、CoT、SC-CoT、Chain/Tree/Complete/Random 拓扑、AgentVerse、MACNET、SelfOrg、MAD-M²(S)、DyLAN。
- **准确率结果**：S1-MAS 在 6 个主要 benchmark 上均取得最高平均准确率 93.97%，较 DyLAN（93.12%）提升 0.85 个百分点，较 AgentVerse（92.43%）提升 1.54 个百分点。
- **Token 效率**：相比 5 个 MAS 基线，GPT token 消耗降低 78.3%–94.0%；平均 input token 从 2,532–9,128 降至 455。
- **延迟优化**：平均端到端延迟 16.8 秒/问题，较基线降低 58.2%–91.7%。
- **WebSRC 结果**：S1-MAS 获得 90.11% exact match，较 AgentVerse（89.78%）、DyLAN（89.89%）略优，token 消耗仅 4,847（较 DyLAN 降低 51.2%）。
- **消融实验**：移除 System One 控制器改用随机选择，平均准确率从 84.17% 降至 82.34%；改用 GPT-4o-mini 做协调决策，token 消耗增加 75.5% 但准确率仅 83.14%（低于 S1-MAS 的 84.17%）；移除 Reader 导致准确率下降 1.76 个百分点。

## 相关工作脉络
- **AgentVerse**：招募 specialized roles 进行协作问题解决，但未解决协调决策的 LLM 依赖问题。
- **DyLAN**：动态调整 agent 参与度的自适应协作框架，仍依赖 LLM 生成协调指令。
- **SelfOrg**：基于 peer contribution 估计重组通信的多智能体框架，计算开销较高。
- **MACNET**：通过有向无环图显式建模通信结构的 MAS，侧重于拓扑设计而非效率优化。
- **MAD-M²(S)**：通过记忆掩码优化通信内容的 MAS，但协调决策过程本身仍有高开销。
- **System One 模型（Laya/Jev）**：专为受限动作空间的结构化决策设计，本文首次将其系统性地应用于通用 LLM-based MAS 的联合任务编排与证据感知通信。

## 局限性与未来方向
- **最大执行步数限制**：当前设置 $T_{max}=4$，复杂任务可能需要更多协作轮次。
- **证据定位的准确性依赖**：Lightweight reader 的定位质量直接影响协调决策效果，对噪声或模糊内容的处理能力有待验证。
- **任务操作空间的固定性**：Solve/Decompose/Reverse/Review/Repair/Contrast 六类操作可能无法覆盖所有协作场景。
- **多页面 Web 检索未探索**：当前 WebSRC 实验仅涉及单页面，多页面信息整合能力待验证。
- **领域泛化性**：实验主要集中在数学推理、知识问答和代码生成，其他领域（如创意写作、法律分析）的有效性需进一步研究。

## 研究启发与可借鉴点
- **有界决策与开放推理的解耦思路**：可将此范式迁移至其他需要频繁协调的 LLM 应用（如长程规划、多轮对话管理），用轻量级模型处理结构化决策。
- **证据定位作为注意力机制**：Lightweight reader 的 localized evidence 作为 attention cue 的设计，可借鉴于减少 LLM 上下文处理开销。
- **累积性通信预算控制**：$B_{src}$ 约束机制可推广至其他多智能体系统的资源管理，防止重复传输导致的上下文膨胀。
- **System One 模型的决策质量验证**：消融实验证明轻量级控制器优于随机路由和 GPT 路由，为 Small Decision Model 在 Agent 系统中的有效性提供了实证支持。
- **无需任务特定训练**：S1-MAS 在零训练条件下即实现高效协作，该零样本泛化能力对下游应用部署具有重要价值。

## 关键术语表
**System One**：基于 Kahneman 双过程理论，指快速、自动、结构化的决策系统，对应本文使用的 Laya 等预训练决策模型。
**Inspection Query**：由 Controller 生成的检查查询，配对问题条件与候选答案以定位需验证的内容。
**Evidence Localization**：通过 Lightweight Reader 在工作者响应中定位与检查条件相关的文本片段的机制。
**Communication Budget**：累积性 source token 预算 $B_{src}$，限制多轮协作中历史响应的重复传输开销。
**Feasible Task Menu**：由确定性编译器根据当前状态筛选的可执行任务集合 $\mathcal{W}_t$。
**Output Identifier**：用于跨响应比较的输出标识符，可以是选项标签、短文本答案或可执行代码。
**Adaptive Stopping**：基于证据充分性的动态终止策略，Controller 可选择 finish 结束协作。
**Delivery Rule**：确定性的答案交付规则，从归档中选择最终答案而无需额外 LLM 调用。

## 可复现要素
- **数据集**：MMLU、GSM8K、AQuA、MultiArith、SVAMP、HumanEval、WebSRC（均为公开基准）。
- **代码/权重**：论文未提及开源代码；使用 Laya（Convai Innovations）和 Qwen3-0.6B 作为预训练模型。
- **关键超参**：最大执行步数 $T_{max}=4$，通信预算 $B_{src}=2,048$ tokens，输出限制 $B_{out}=2,048$ tokens，温度 $temp=1.0$。
- **实验环境**：主实验使用 GPT-4o，消融实验使用 GPT-4o-mini。
