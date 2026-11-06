---
title: "When-the-Critic-Falls-Behind-Co-Evolving-Rubrics-and-Harness"
source: https://arxiv.org/pdf/2610.36826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:55:59"
field: "Agent 持续自演化与诊断"
keywords: ["agent self-improvement", "harness evolution", "rubric co-evolution", "continual learning", "LLM agents"]
innovations: ["提出 RuHarness 实现规标与 harness 双向协同演化，动态追踪迁移的行为弱点", "设计结果盲行为评估与六坐标协同候选生成机制，解耦诊断信号与任务结果", "构建时变诊断效用与保守后验排名，解决小样本准则优先级排序的可靠性问题"]
benchmarks: ["EvoAgentBench", "SWE-Bench", "LiveCodeBench", "BrowseComp-Plus", "OmniMath", "GDP-Val"]
---

# 论文速读：When-the-Critic-Falls-Behind-Co-Evolving-Rubrics-and-Harness

## 一句话总结
提出 **RuHarness**，一种规标（rubric）与执行框架（harness）协同演化的持续智能体自改进框架，通过迭代地从执行轨迹中发现并重新评估行为标准、优先排序未解决的弱点，并用动态规标前沿指导有针对性的 harness 更新，在保持任务目标固定的同时实现自适应监督。

## 研究问题与动机
1. **静态诊断信号无法追踪迁移的弱点分布**：现有 harness 演化方法依赖相对固定的任务级反馈（仅指示成功/失败），随着演化推进，原有的主导失败模式被缓解后，更细微或新的失败模式出现，而固定诊断标准逐渐失去区分性和指导价值。
2. **反复优化易导致过拟合**：多次 harness 优化可能逐步过拟合演化任务中暴露的特定失败模式，导致在未见任务上的泛化能力下降。
3. **缺乏细粒度行为诊断机制**：现有方法仅提供任务级结果反馈，缺乏对导致失败的行为缺陷的细粒度分析，难以在持续演化中定位需要优先改进的具体行为问题。
4. **已有 rubric 工作聚焦策略/响应而非 harness**：现有自适应规标研究（如 EvoLM、EvoRubrics）主要将规标与策略或响应协同演化，将动态 rubric 应用于持续 harness 演化这一场景尚未被充分探索。

## 核心贡献（创新点）
1. **提出 RuHarness 框架，实现 rubric–harness 双向协同演化**：通过交替执行 rubric 演化（从轨迹发现/重新评估行为标准）和 harness 演化（按规标前沿定向更新），形成闭环；相比已有工作仅演化 harness，本文首次将动态诊断信号本身作为可演化组件纳入持续自改进循环。
2. **设计结果盲（outcome-blind）行为评估机制**：新发现的规标由仅访问轨迹行为证据、不接触最终任务结果的评判器评估，防止成功/失败标签直接泄露到准则评价中；相比直接利用任务结果评价准则，该设计使准则质量评估更纯粹、更具诊断性。
3. **引入时变诊断效用函数与保守后验排名**：构建包含区分度、覆盖率、预测有效性、新颖性和冗余惩罚的五维效用指标，并采用 Dirichlet 后验下界 q-分位数进行保守排序；相比直接使用点估计排名，该方法在小样本准则上更稳健，避免高估少量观察支持的高效用准则。
4. **提出六坐标协同候选生成与验证门控更新机制**：将可编辑 harness 抽象为 Equip/Instruct/Act/Observe/Remember/Terminate 六个功能坐标，基于规标前沿联合规划多坐标更新候选，并用独立验证集进行门控选择；相比逐坐标独立更新，该设计保留多坐标间的交互关系，并通过验证集解耦候选选择与演化轨迹。
5. **设计规标生命周期管理（四状态机）**：新准则经历 probation→active→dormant→retired 的生命周期流转，结合出生检验（birth check）过滤低诊断价值准则；相比简单累积所有发现的准则，该机制保证只有具备持续诊断价值的准则参与前沿构建。

## 方法详解
**交替演化流程**：每轮演化 t，当前 harness $H_t$ 在演化任务集 $D_{\text{evo}}$ 上执行收集轨迹 $\mathcal{T}_t$，然后：
- **Rubric 演化**：$R_{t+1} = \mathcal{E}_R(R_t, \mathcal{T}_t)$，从轨迹中发现新行为标准并重新评估已有标准。
- **Harness 演化**：$H_{t+1} = \mathcal{E}_H(H_t, \widehat{R}_{t+1})$，由优先级规标前沿 $\widehat{R}_{t+1}$ 指导 harness 定向更新。

**结果盲行为评估**：对每条轨迹 $\tau$ 和每个准则 $c$，outcome-blind judge 输出 $J(c,\tau) \in \{\text{Satisfied, Violated, N/A}\}$，judge 仅可见行为证据，不接触任务最终结果。

**准则准入与生命周期**：新准则经过出生检验（过滤：80%+为 N/A、失败轨迹上无违规、整体/通过轨迹上违规率均>90%），通过者进入 probation 状态，满足 $P>0$ 晋升为 active，连续多轮触发率<5% 或 N/A 漂移转为 dormant，持续无信息则 retire。

**时变诊断效用**：
$$U_{t+1}(c) = \alpha D_{t+1}(c) + \beta C_{t+1}(c) + \gamma P_{t+1}(c) + \delta N_{t+1}(c) - \lambda \text{Red}_{t+1}(c)$$
其中 $D$ 衡量成功/失败轨迹间区分度，$C$ 为当前失败覆盖率，$P$ 为预测有效性，$N$ 捕捉新颖失败，$\text{Red}$ 惩罚与已有准则的行为冗余。参数设为 $\alpha=\gamma=1.0$，$\beta=\delta=\lambda=0.5$。

**保守后验排名**：基于 $2\times3$ 列联表构建对称 Jeffreys 先验的 Dirichlet 后验，抽取 $B=2000$ 次样本，取第 $q=0.2$ 分位数作为保守效用：
$$\widetilde{U}_{t+1}(c) = Q_{0.2}[p(U_{t+1}(c) \mid \mathcal{T}_t)]$$
然后选取 Top-K（$K=10$）个 active 准则构成规标前沿 $\widehat{R}_{t+1}$。

**Harness 演化**：将可编辑 harness 分为六坐标（Equip/Instruct/Act/Observe/Remember/Terminate），前沿中的每个准则映射到若干坐标，联合规划候选修改 $\Delta_j$，经轻量验证器检查后在独立验证集 $D_{\text{val}}$ 上评估，选择最优候选 $H_t^\star$，仅当 $S(H_t^\star, D_{\text{val}}) - S(H_t, D_{\text{val}}) > \epsilon=0.05$ 时提交更新。

## 实验与结果
**数据集与基线**：使用 **EvoAgentBench**（涵盖信息检索 BrowseComp-Plus、数学推理 OmniMath、软件工程 SWE-Bench、代码实现 LiveCodeBench、知识工作 GDP-Val 五个领域），以 Pass@1 为指标。基线包括 ACE、EvoSkill、Self-Harness、AHE、Meta-Harness。主模型为 **Qwen3.5-35B-A3B** 和 **GLM-4.7-Flash**。

**主要结果**：
- **Qwen3.5-35B-A3B**：RuHarness 在 held-in 平均达 **62.4**、held-out 平均达 **53.7**，分别比最强基线（Self-Harness，48.3/48.3）提升 **1.7/4.5** 个百分点；在 GLM-4.7-Flash 上提升 **2.7/5.3** 个百分点。
- **泛化优势**：在各领域 held-out 上 RuHarness 均取得最高或接近最高的分数，且 held-in–held-out 差距最小，表明改进更不易过拟合演化任务。
- **消融结果**：去除 rubric 模块后 held-in 从 62.4 降至 57.1、held-out 从 53.7 降至 45.8；使用静态 rubric（不演化）表现甚至比无 rubric 更差（held-out 仅 44.4），证明静态规标会阻碍演化；去除前沿选择（使用全部 rubric）也导致性能下降（49.0）。
- **鲁棒性**：在三个随机种子下，RuHarness 在两种模型–任务组合上均保持最佳均值且方差与竞品相当。
- **演化轨迹分析**：基准方法在早期快速上升后趋于 plateau，而 RuHarness 在后期轮次持续提升；held-out 上的基线波动较大，RuHarness 呈现更稳定的上升趋势。

## 相关工作脉络
1. **Self-Harness (Zhang et al., 2026a)**：从执行轨迹挖掘模型特定弱点并提出针对性更新，但使用固定诊断信号，不随行为变化而调整；本文通过动态 rubric 持续重新评估诊断标准。
2. **Meta-Harness (Lee et al., 2026b)**：在累积候选和轨迹中搜索可执行的 harness 代码，优化空间以代码为主；本文采用六坐标结构化抽象并配合规标前沿进行定向引导。
3. **AHE (Lin et al., 2026a)**：通过可观察、可验证的组件更新改进 harness；本文进一步引入了从轨迹中自适应发现准则的 rubric 演化模块。
4. **ACE (Zhang et al., 2026b)**：从交互经验中演化持久上下文知识；本文关注更全面的 harness（含 prompt、工具、记忆、控制逻辑），并通过动态诊断信号指导更新而非仅积累上下文。
5. **EvoLM / EvoRubrics / DynamicRubric (Li et al., 2026b; Ding et al., 2026; Wang et al., 2026a)**：将规标与策略/响应协同演化于 RL 场景；本文首次将动态 rubric 机制引入 agent harness 的持续自演化场景。
6. **Co-Harness (Chen et al., 2026b) / Helix (Fan & Huang, 2026)**：探索 harness 与模型权重的协同演化；本文保持模型固定，专注于 harness 自身的持续演化与诊断信号自适应。

## 局限性与未来方向
1. **单一 agent 框架限制**：所有实验仅在 Deep-Agent 框架下进行，六个坐标的 harness 抽象尚未在更广泛的 agent 架构上验证其通用性。
2. **固定演化轮次的实验设置**：受计算资源限制，实验采用固定演化窗口，未探索更长期或更开放式的自适应场景下的协同演化动态。
3. **可扩展性未知**：Top-K 前沿选择、阈值 $\epsilon$ 等超参在实验设置下有效，但在更大规模或更长演化序列下的稳定性未充分验证。

## 研究启发与可借鉴点
1. **结果盲评估设计**：outcome-blind judge 将行为评估与任务结果解耦，可迁移至其他需要细粒度过程诊断的场景（如多步推理链评估、工具调用策略优化），避免"结果泄露"导致的评估偏差。
2. **保守后验排名策略**：Dirichlet 后验下界分位数排名方法适用于任何小样本准则/特征优先级排序问题，可在推荐系统、安全策略评估等数据稀疏场景中复用。
3. **六坐标 harness 抽象**：将 agent 执行框架结构化为 Equip/Instruct/Act/Observe/Remember/Terminate 六维空间，为后续系统性 harness 诊断与编辑提供了统一接口，可推广至不同 agent 架构。
4. **动态诊断信号的持续演化思想**：本文证明了"静态 rubric 比无 rubric 更差"的反直觉结论，提示我们在任何迭代优化系统中都应持续重新校准诊断信号，而非固化初始评估标准。
5. **验证集与演化集严格分离的设计**：候选选择使用独立 $D_{\text{val}}$，与驱动演化的 $D_{\text{evo}}$ 完全不相交，有效防止候选评估泄露到演化过程中，可推广至其他自改进系统的实验设计。

## 关键术语表
**RuHarness**：Rubric–Harness Co-Evolution 的缩写，本文提出的规标与执行框架协同演化持续自改进框架。
**Rubric（规标）**：由可观察行为描述构成的结构化评价准则集合，用于细粒度诊断 agent 行为弱点。
**Rubric Frontier（规标前沿）**：从活跃规标池中按保守后验诊断效用排名选出的 Top-K 准则集合，用于定向引导 harness 更新。
**Outcome-Blind Assessment（结果盲评估）**：仅基于轨迹行为证据、不接触任务最终成功/失败标签的准则评估机制。
**Diagnostic Utility（诊断效用）**：由区分度、覆盖率、预测有效性、新颖性和冗余惩罚加权组成的时变准则效用指标。
**Harness（执行框架）**：围绕 LLM  governing interaction、memory、tools 和 execution 的外部基础设施，本文将其抽象为六个可编辑功能坐标。
**Validation-Gated Update（验证门控更新）**：仅在候选 harness 在独立验证集上的改进超过阈值时才提交更新的选择机制。
**Rubric Lifecycle（规标生命周期）**：准则经历的 probation → active → dormant → retired 四状态流转机制，保证只有持续具有诊断价值的准则参与演化。

## 可复现要素
- **数据集**：EvoAgentBench（含 BrowseComp-Plus、OmniMath、SWE-Bench、LiveCodeBench、GDP-Val），论文已提供详细的 split 统计（训练/验证/测试），数据集来源均为公开 benchmark。
- **代码/权重开源情况**：论文未明确声明代码开源（截至 arXiv 版本），模型权重使用 Qwen3.5-35B-A3B 和 GLM-4.7-Flash，需从官方渠道获取。
- **关键超参**：$\alpha=\gamma=1.0$，$\beta=\delta=\lambda=0.5$；$q=0.2$（保守后验分位数）；$K=10$（前沿准则数）；$\epsilon=0.05$（更新接受阈值）；每轮最多生成 3 个候选，每个候选最多修改 3 个坐标；Dirichlet 后验采样次数 $B=2000$。
