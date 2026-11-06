---
title: "When-the-Critic-Falls-Behind-Co-Evolving-Rubrics-and-Harness"
source: https://arxiv.org/pdf/2610.36826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:56:09"
field: "Agent 自我进化与持续学习"
keywords: ["Agent Self-Improvement", "Rubric-Harness Co-Evolution", "Continual Learning", "LLM Agents", "Automated Harness Optimization", "Behavioral Diagnostic"]
innovations: ["首次将rubric与harness进化耦合为交替迭代的协同过程，实现诊断监督的自适应追踪", "提出outcome-blind行为评估与四态准则生命周期管理，自动过滤低价值诊断信号", "构建时变诊断效用函数与Dirichlet后验保守排序，实现统计可靠的准则优先级动态调整"]
benchmarks: ["EvoAgentBench"]
---

# 论文速读：When-the-Critic-Falls-Behind-Co-Evolving-Rubrics-and-Harness

## 一句话总结
本文提出 RuHarness，一种 rubric（评估准则）与 harness（智能体执行框架）协同进化的框架，用于 agent 的持续自我改进；通过从执行轨迹中动态发现、重新评估和优先排序行为弱点，并以此指导 harness 的定向更新，实现更强的泛化能力和更稳定的持续改进。

## 研究问题与动机
1. **现有 harness 进化方法的泛化瓶颈**：近期工作开始从执行经验中进化 agent harness，但控制进化的监督信号（rubric）往往相对静态，难以跟踪随进化过程中不断变化的行为弱点分布。
2. **静态监督导致诊断价值衰减**：随着 harness 逐步改进，原先主导的失败模式可能被缓解，而更微妙或此前未见的失败模式会浮现；静态的粗粒度监督会失去区分性和指导价值。
3. **进化过程中缺乏对弱点重要性变化的显式跟踪机制**：现有方法主要优化 task-level feedback（仅指示成功/失败），未能提供关于失败背后行为缺陷的细粒度信息，难以支持持续进化所需的自适应诊断。
4. **harness 优化可能过拟合到特定任务的失败模式**：重复的 harness 优化易在进化任务上过拟合，导致在未见任务上性能下降，需要一种在保持任务目标不变的前提下持续追踪行为弱点的机制。

## 核心贡献（创新点）
1. **提出 RuHarness 协同进化框架**：首次将 rubric 进化与 harness 进化耦合为一个交替迭代的过程，使诊断监督能跟随 agent 行为变化而自适应调整。
2. **设计 outcome-blind 行为评估与准则生命周期管理**：引入对轨迹的行为级盲评机制（不泄露任务结果），并赋予新准则 probation → active → dormant → retired 的四态生命周期，自动过滤低诊断价值的准则。
3. **构建时变诊断效用函数与保守优先级排序**：定义包含区分度、覆盖率、预测有效性、新颖性和冗余惩罚的五维效用指标，并通过 Dirichlet 后验下界分位数实现保守的准则 prioritization。
4. **提出六坐标 harness 抽象与协同候选生成机制**：将可编辑 harness 结构化表示为 Equip/Instruct/Act/Observe/Remember/Terminate 六个功能坐标，支持多坐标联合修改的协调性候选生成。
5. **在 EvoAgentBench 上验证强泛化增益**：在五个 agent 领域（信息检索、知识工作、软件工程、代码实现、数学推理）与两个 backbone 模型上，RuHarness 在 held-out 任务上取得显著优势（GLM-4.7-Flash 提升 5.3 分）。

## 方法详解

### 整体架构（§3.1）
RuHarness 在每一轮进化 $t$ 维护状态 $\boldsymbol{S}_t = (\boldsymbol{H}_t, \boldsymbol{R}_t)$，其中 $H_t$ 为当前 harness，$R_t$ 为准则池。交替执行两个过程：
- **Rubric 进化**：在进化任务集 $D_{\text{evo}}$ 上运行 $H_t$ 收集轨迹 $\mathcal{T}_t$，从中发现新的行为准则并重新评估已有准则，得到更新后的准则池 $R_{t+1} = \mathcal{E}_R(R_t, \mathcal{T}_t)$。
- **Harness 进化**：对准则池进行优先级排序得到前沿准则集 $\widehat{R}_{t+1}$，引导定向的 harness 更新 $H_{t+1} = \mathcal{E}_H(H_t, \widehat{R}_{t+1})$。

### Rubric 进化细节（§3.2）

**Outcome-blind 行为评估**：对每个准则 $c$ 和轨迹 $\tau$，outcome-blind judge 仅根据轨迹中的行为证据给出三值判定：Satisfied / Violated / N/A，不泄露最终任务结果。

**准则准入与生命周期**：
- **Birth Check**：过滤三类无效准则：（i）很少适用（>80% 为 N/A）；（ii）从不触发（在未成功轨迹上无违规）；（iii）过度触发（总体违规率 >90% 且在成功轨迹上也 >90%）。
- 四态生命周期：probation（通过 $P>0$ 转为 active）→ active（触发率<5% 持续3轮或N/A漂移转 dormant；$P<0.10$ 持续3轮转 retired）→ dormant（触发率≥15% 可重激活；休眠4轮转 retired）→ retired（终态）。

**时变诊断效用**（公式4）：
$$U_{t+1}(c) = \alpha D_{t+1}(c) + \beta C_{t+1}(c) + \gamma P_{t+1}(c) + \delta N_{t+1}(c) - \lambda \text{Red}_{t+1}(c)$$
其中：$D$=区分度（失败/成功轨迹上的违规率之差）、$C$=覆盖率（未成功轨迹中被覆盖比例）、$P$=预测有效性、$N$=新颖性（唯一标识的失败轨迹比例）、$\text{Red}$=冗余惩罚（与其他准则的 Jaccard 重叠）。

**保守优先级排序**（公式5-6）：基于 $2\times3$ 列联表的 Dirichlet 后验（Jeffreys prior），抽取 $B=2000$ 个样本计算效用分布，取第 $q=0.2$ 分位数作为保守评分 $\widetilde{U}$，选出 Top-K（$K=10$）准则构成 rubric frontier。

### Harness 进化细节（§3.3）

**六坐标抽象**：将可编辑 harness 表示为六个功能坐标——Equip（工具/技能）、Instruct（指令）、Act（行动策略）、Observe（观察处理）、Remember（记忆管理）、Terminate（完成控制）。

**Frontier-guided 目标定位**：每个准则可映射到多个坐标（一对多），允许联合干预。

**协同候选生成**（公式7-8）：每个候选表示为跨多个坐标的修改集合 $\Delta_j = \{\delta_j^{(k)}\}_{k \in M_j}$，通过更新算子 $\oplus$ 作用于当前 harness：$H_t^{(j)} = H_t \oplus \Delta_j$。候选需通过轻量验证器检查与诊断弱点的一致性。

**验证门控更新**（公式9-10）：候选在独立验证集 $D_{\text{val}}$ 上评估，选取得分最高且 improvement $> \epsilon = 0.05$ 的候选提交更新；否则保留当前 harness。

## 实验与结果

### 实验设置
- **数据集**：EvoAgentBench，覆盖五个领域（Information Retrieval、Knowledge Work、Software Engineering、Code Implementation、Math Reasoning），分为 evolution（held-in）、validation 和 test（held-out）三个独立划分。
- **Backbone 模型**：Qwen3.5-35B-A3B 和 GLM-4.7-Flash，基于 Deep-Agent 框架实例化。
- **评估指标**：Pass@1，各领域使用对应 benchmark evaluator。
- **基线方法**：ACE、EvoSkill、Self-Harness、AHE、Meta-Harness。
- **超参**：每轮最多生成3个候选，$\epsilon=0.05$，$\alpha=\gamma=1.0$、$\beta=\delta=\lambda=0.5$，$q=0.2$，$K=10$，每个候选最多修改3个坐标。

### 主要结果
| 模型 | 方法 | Held-in 平均 | Held-out 平均 |
|---|---|---|---|
| Qwen3.5-35B-A3B | Vanilla Agent | 54.6 | 44.2 |
| Qwen3.5-35B-A3B | Meta-Harness（最强基线） | 60.7 | 46.5 |
| **Qwen3.5-35B-A3B** | **RuHarness** | **62.4** | **53.7** |
| GLM-4.7-Flash | Vanilla Agent | 33.6 | 24.5 |
| GLM-4.7-Flash | Self-Harness（最强基线） | 40.6 | 32.7 |
| **GLM-4.7-Flash** | **RuHarness** | **43.3** | **38.0** |

- **Qwen3.5**：相比最强基线 Meta-Harness，held-in 提升 **+1.7**，held-out 提升 **+4.5**。
- **GLM-4.7**：相比最强基线 Self-Harness，held-in 提升 **+2.7**，held-out 提升 **+5.3**。
- **泛化优势最显著**：在各单独领域中，RuHarness 在 held-out 上普遍取得最佳或次佳成绩，且 held-in–held-out gap 最小。

### 消融分析
- **移除 Rubric 模块**：held-in 从 62.4 降至 57.1，held-out 从 53.7 降至 45.8（↓7.9）。
- **固定 Rubric（无进化）**：性能反而比移除 rubric 模块更差（held-out 仅 44.4），说明静态 rubric 会误导进化。
- **移除 Rubric Frontier**：使用完整准则池而非 Top-K 前沿导致性能下降（held-out 49.0 vs 53.7），证实聚焦高价值准则的重要性。

### 动态分析
- **Rubric Frontier 演进模式**：与初始 frontier 的相似度在前几轮急剧下降后保持稳定低位；相邻轮 frontier 相似度先降后升并稳定在较高水平，反映早期大幅重排、后期局部微调的模式。
- **Prioritized 弱点的渐进缓解**：进入 frontier 的准则违规率在第一轮后无明显变化，但在后续轮中逐渐下降，说明进化关注点能有效转化为行为改进。
- **Committed 更新的长期收益**：通过验证门控的更新能带来持续的违规率下降，而未提交的更新则波动无趋势。

## 相关工作脉络
1. **Self-Harness (Zhang et al., 2026a)**：挖掘模型特异性弱点并提议定向更新；本文相对定位为将"诊断信号本身也纳入进化循环"，而 Self-Harness 仍使用静态的 task-level 反馈。
2. **Meta-Harness (Lee et al., 2026b)**：通过累积候选和轨迹搜索可执行的 harness 代码；本文强调诊断准则的动态适配，而非仅依赖候选搜索空间展开。
3. **AHE (Lin et al., 2026a)**：通过可观测、可验证的组件更新改进 harness；两者均关注 harness 结构优化，但本文额外引入了准则生命周期管理与保守优先级排序。
4. **EvoLM / EvoRubrics / DynamicRubric**：在 LLM policy/response 层面实现 rubric 协同进化；本文将此思想首次拓展到 agent harness 进化场景。
5. **Co-Harness (Chen et al., 2026b) / RRSI (Xia et al., 2026)**：探索 harness-model 协同进化或正则化递归自改进；本文聚焦于 harness 自身的持续改进且不修改模型参数。
6. **Reflection- / Memory-based 方法 (Shinn et al., 2023; Zhong et al., 2023)**：通过存储和精炼交互经验改进未来决策；本文将其扩展至对整个 agent 执行基础设施（prompt/tool/memory/control logic）的系统性进化。

## 局限性与未来方向
1. **单一 Agent 框架限制**：所有实验仅在 Deep-Agent 框架下进行，未在更广泛的 agent 架构上验证六坐标 harness 抽象的通用性。
2. **固定进化周期的约束**：受计算资源限制，实验在固定进化轮次下进行，未探索更长周期和更开放式的适应场景。
3. **未涉及模型参数更新**：本文仅进化外部 harness，未探索与模型权重更新的联合进化路径（如 Co-Harness 所示）。

## 研究启发与可借鉴点
1. **Outcome-blind 行为评估设计**：将准则判定与任务结果解耦，避免监督信号直接泄露最终成功/失败标签，这一设计可迁移到任何需要细粒度行为诊断的 self-improvement 场景。
2. **准则生命周期管理机制**：probation → active → dormant → retired 的四态自动流转机制，为动态维护大规模评估准则池提供了可复用的工程范式，适用于持续学习的评测体系构建。
3. **Dirichlet 后验保守排序策略**：利用下分位数应对小样本统计不确定性，有效防止低置信度准则主导优化方向，该方法可推广至其他基于排序的选择性学习场景。
4. **六坐标 harness 抽象**：将复杂的 agent 执行基础设施解构为六个正交功能坐标，为系统化地分析和修改 agent 行为提供了统一的语言，可应用于跨框架的 harness 对比研究。
5. **验证门控与进化任务的严格分离**：独立验证集专用于候选选择，不与进化轨迹共享，这一设计有效防止了 overfitting 到进化任务，值得在类似迭代优化系统中借鉴。

## 关键术语表
**Rubric（评估准则）**：用于细粒度评估 agent 行为的结构化行为标准，本文中指可动态发现和优先排序的准则集合。
**Harness（执行框架）**：围绕 LLM 构建的交互、记忆、工具和执行控制基础设施，本文中专指可编辑优化的外部执行组件集合。
**Rubric Frontier**：从活跃准则池中经保守排序选出的 Top-K 高优先级准则集合，用于指导下一轮的 harness 定向更新。
**Outcome-blind Assessment**：仅依据轨迹中的行为证据进行准则判定，不泄露最终任务成功/失败标签的评估机制。
**Diagnostic Utility**：综合区分度、覆盖率、预测有效性、新颖性和冗余惩罚的时变准则效用函数，用于量化准则在当前 harness 状态下的诊断价值。
**Coordinated Candidate Generation**：基于准则到坐标的多对多映射，联合规划跨多个 harness 坐标的修改方案的候选生成机制。
**Validation-gated Update**：仅在候选在独立验证集上改善超过阈值时才提交 harness 更新的门控机制。
**Jaccard Similarity of Frontiers**：用于衡量不同进化轮次间 rubric frontier 重叠程度的相似度指标。

## 可复现要素
- **数据集**：EvoAgentBench（包含5个领域的 held-in/held-out 划分，论文未声明公开状态，但基于公开基准构建）
- **代码/权重**：论文未提及开源声明
- **关键超参**：$\epsilon=0.05$（验证提升阈值）、$\alpha=\gamma=1.0$、$\beta=\delta=\lambda=0.5$（效用权重）、$q=0.2$（后验分位数）、$K=10$（frontier大小）、每轮最多3个候选、每个候选最多修改3个坐标、$B=2000$（后验采样数）
- **Backbone**：Qwen3.5-35B-A3B、GLM-4.7-Flash
- **框架**：Deep-Agent
