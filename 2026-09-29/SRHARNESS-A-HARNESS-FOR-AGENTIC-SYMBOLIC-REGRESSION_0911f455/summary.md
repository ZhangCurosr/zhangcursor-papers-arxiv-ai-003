---
title: "SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION"
source: https://arxiv.org/pdf/2609.35501v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:13:08"
field: "符号回归与科学发现"
keywords: ["Symbolic Regression", "Agentic AI", "LLM", "Agent Harness", "Scientific Discovery", "Equation Discovery"]
innovations: ["提出首个面向 Agentic SR 的领域特定 harness 框架，显式分离运行时支撑层与搜索策略", "设计可组合科学动作、持久科学状态和轨迹生命周期管理三大机制，实现异构操作共享评估语义", "通过系统实验证明 harness 增益独立于模型规模和工具可用，同 backbone 下 SA 从 62.16% 提升至 93.69%"]
benchmarks: ["LLM-SRBench (LSR-Synth)", "LSR-Transform", "LSR-Transform-Anon"]
---

# 论文速读：SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION

## 一句话总结
论文提出 SRHarness，一个专为 Agentic Symbolic Regression（符号回归）设计的领域特定运行时框架，通过可组合科学动作、持久科学状态和轨迹生命周期管理三大机制，在 LLM-SRBench 上以相同 LLM backbone 显著超越现有方法（SA 最高达 93.69%），并证明 harness 本身是独立于模型和工具的性能增益来源。

## 研究问题与动机
- 现有 Agentic SR 系统逐渐将 LLM 从"方程提议者"升级为"自主控制器"，但性能不仅依赖底层模型和搜索策略，还高度依赖支持长程科学搜索的运行时基础设施，这一因素长期被忽视。
- 既有系统的支撑机制通常围绕各自搜索策略定制，而非作为独立可复用的层存在，导致跨系统比较时搜索策略差异与基础设施差异相互混淆（张等人 2026c）。
- 符号回归的 agent 需要在对话过程中持续分析观测数据、维护竞争性假设集合、并在证据涌现后 revisiting/revising 已有候选方程，这对 harness 提出了统一的接口、持久状态和长程治理三方面的需求。
- 需要一种不预设特定搜索策略、但提供可复用运行时支持的 domain-specific harness，将执行环境、认知管理和治理三个维度显式分离。

## 核心贡献（创新点）
- **提出 SRHarness 框架**：首个将 agent harness 思想具体化到符号回归领域的系统框架，显式分离 reusable runtime support 与底层模型和搜索策略。与已有工作（如 SR-Scientist、KeplerAgent）的本质区别在于，它不是某种特定搜索策略的实现，而是一个可独立评估、可复用的运行时层。
- **可组合科学动作（Composable Scientific Actions）**：通过表达式视图接口将 agent 指定的科学参数与 harness 管理的执行上下文分离，使异构分析、拟合和搜索操作共享统一的数据访问语义（原始变量、变换视图、候选衍生量）。与 IGSR/LLM-SR 等方法仅在固定数据集列上操作相比，本文允许中间视图在不同阶段间直接复用，无需重复构造数据集引用。
- **持久科学状态与紧凑模型视图**：将已评估候选方程及其证据、溯源信息保存在对话上下文之外，并提供 Pareto 前沿视图和 Top-k 视图两类投影策略。与 SR-Scientist 依赖完整对话历史的方式相比，本设计避免了语境膨胀，同时保留可回溯性。
- **轨迹生命周期管理**：通过 R×C×L×K 调度器协调续接、分支、重启和终止，并跨转换保留溯源信息。与现有方法（多为线性单线程对话）相比，支持更灵活的搜索预算分配和中断恢复。
- **系统性实验与消融**：在 LLM-SRBench 三个 track 上进行多 backbone 对比、与 Codex 的对照实验、以及 action/view/scheduling 三层消融，证明 harness 增益独立于模型规模和工具可用性。

## 方法详解
- **科学动作的形式化**：每个动作表示为 $r = \mathcal{A}(q; \kappa)$，其中 $q$ 为 agent 指定的科学参数（可表达为 raw 变量 $x_i$、变换视图如 $\log x_i$ 或 $x_i/x_j$、候选残差 $y - f(x)$ 等），$\kappa$ 为 harness 管理的执行上下文（数据绑定、train/validation 划分、运行时配置）。候选产出动作遵循"候选-证据契约"，记录表达式、评估值、复杂度、诊断证据和溯源信息。
- **持久科学状态**：维护 $S_t$ 作为累积状态，模型面向视图为 $v_t = P_\rho(S_t)$，$\rho$ 为投影策略。实现两种投影：fit-complexity Pareto 视图（展示 trade-off 前沿）和 Top-k 视图（展示领先历史候选），二者均只暴露紧凑子集而非完整存档。
- **轨迹调度（Algorithm 1）**：配置为 $(R, C, L, K)$——重启轮数、独立分支数、精炼深度、本地响应采样数。每次重启时以 Top-k 视图初始化新对话；每分支内最多 $L$ 步精炼，每步采样 $K$ 个响应，按候选质量选一路续接，其余诊断证据合并入上下文。全程记录 provenance，支持 early stopping 和失败恢复。

## 实验与结果
- **数据集**：LLM-SRBench，包含 LSR-Synth（128 题，物理/化学/生物/材料科学，评估 ID/OOD 数值泛化）和 LSR-Transform（111 题，强调精确符号恢复），以及作者构造的 LSR-Transform-Anon（移除语义描述和变量名）。
- **基线**：PySR、LLM-SR、IGSR、SR-Scientist（均在匹配 backbone 下重跑）。
- **LSR-Synth**：DeepSeek-v4-flash-0731 下，SRHarness 在物理/化学/生物 OOD Acc0.1 分别达 78.00%/78.87%/87.09%，全面优于基线；SA 达 6.20%，而匹配基线均低于 1%。材料科学域接近饱和。
- **LSR-Transform**：SRHarness SA 93.69%（DeepSeek）vs SR-Scientist 62.16%，且表达式复杂度最低（14.64 vs 64.46/42.14/17.50）；所有基线 NMSE 均达 $10^{-14}$ 量级，说明数值拟合不能区分符号恢复质量。
- **LSR-Transform-Anon**：SRHarness SA 72.97% vs SR-Scientist 39.64%，SA 保留率 78% vs 64%；搜索时间几乎不变（12.34 min vs 基线大幅延长），证明优势并非仅依赖语义先验。
- **Codex 对照**：相同 DeepSeek backbone 下，SRHarness SA 72.97% vs Codex 20.72%；SRHarness+DeepSeek 性能可比 Codex+GPT-5.5（68.47%）。将相同工具暴露给 Codex 反而使 SA 从 68.47% 降至 64.86%，表明 harness 价值不来自工具集合本身。
- **效率**：SRHarness 平均 11.11 min/题（DeepSeek），快于 LLM-SR（57.67 min）和 IGSR（24.02 min），API 成本 \$0.03/题。

## 相关工作脉络
- **LLM-guided SR**（LLM-SR、LaSR、SGA、SR-LLM 等）：LLM 作为预设进化/搜索流程中的变异算子或候选生成器，search procedure 由算法主导。本文与之的区别在于将 LLM 提升为自主控制器。
- **Agentic SR**（SR-Scientist、KeplerAgent、MOT-SR、DE、A-SR 等）：LLM 承担控制角色，但支撑机制围绕各自策略定制，缺乏独立 harness 层。本文的定位是将这些机制抽象为可复用的运行时层，独立于搜索策略。
- **Agent harnesses**（Beaver、Ref 等）：面向多模态科学信息处理或游戏环境，缺乏对科学搜索中"候选方程共享评估语义"和"中间视图复用"的支持。本文的 harness 专门针对符号回归的 hypothesis-centric 搜索设计。
- **PySR**：传统进化式符号回归代表，作为非 LLM 基线，用于锚定 agentic 方法相对于成熟传统方法的性能位置。
- **Zhang et al. (2026c)**：指出比较 LLM agent 时必须披露 harness，本文直接回应了这一批评，证明 harness 是独立于模型的工具变量。

## 局限性与未来方向
- 不同 backbone 在 LSR-Synth 上性能呈非单调关系（DeepSeek-v4.1-flash 优于更大规模的 DeepSeek-v4-pro），当前仅观察到与 long-horizon agent benchmark（TerminalBench、DeepSWE）的相关性，缺乏因果解释。
- 消融实验显示，添加更多 fitting 动作虽提升 SA（至 79.28%），但同时增加表达式复杂度（33.16）和资源消耗，工具丰富度与搜索效率的权衡尚需系统研究。
- 轨迹调度参数 $(R, C, L, K)$ 的自动优化未涉及；当前默认配置 R1-C1-L30-K1 的效果较好，但不同 budget 分配策略（更多重启/分支 vs 更长精炼）的效果差异仅通过小规模消融探索。
- 实验仅在 LLM-SRBench 上进行，harness 在其他科学发现场景（如实验规划、多变量微分方程发现）中的可迁移性有待验证。

## 研究启发与可借鉴点
- **harness-as-first-class-component 的设计哲学**：将运行时支撑层从搜索策略中独立出来，可作为后续研究的标准范式；本团队在构建 Agent-driven 科学工作流时可直接迁移此分层思路。
- **表达式视图机制**：允许同一操作作用于 raw/transformed/candidate-derived 三类视图，无需为每种数据形态构造专用接口；可迁移到多步骤数据分析流水线或自动实验设计系统中。
- **Pareto 视图优于单最佳候选视图**：消融证明暴露 trade-off 前沿能显著提升 SA（74.47% vs 66.67%），而单纯移除 best-formula 信号仅轻微下降（72.97%）；这一设计对多目标科学发现任务的候选管理具有参考价值。
- **与 Codex 的对照实验设计**：通过"相同 backbone + 不同 harness"的对照，干净地隔离了 harness 的贡献；本团队在评估自研 agent 框架时也可采用此对照范式。
- **本地响应采样（K>1）的价值**：在相同积累状态下定向多路采样并选择最优续接，可在不增加搜索深度的前提下提升数值精度（95.38%）和降低复杂度（16.30）；适合资源受限但需要高质量拟合的场景。

## 关键术语表
- **Symbolic Regression（符号回归）**：从观测数据中自动发现可解释数学表达式以刻画变量间关系的任务。
- **Agentic Symbolic Regression（Agentic 符号回归）**：将 LLM 作为自主控制器、基于中间证据动态选择科学操作的符号回归新范式。
- **Composable Scientific Actions（可组合科学动作）**：通过统一接口和表达式视图使异构科学操作共享输入/输出语义的动作设计。
- **Persistent Scientific State（持久科学状态）**：跨对话轨迹保存已评估候选方程、证据和溯源信息的结构化状态，独立于瞬态对话上下文。
- **Trajectory Lifecycle Management（轨迹生命周期管理）**：协调长程搜索中续接、分支、重启和终止的运行时管理机制。
- **Symbolic Accuracy（SA）**：衡量发现表达式与目标方程在代数结构上等价的比例，区别于仅基于数值拟合的评估。
- **Expression-based Views（表达式视图）**：对原始变量、变换变量（如 $\log x_i$）和候选衍生量（如残差 $y-f(x)$）的统一访问抽象。
- **LLM-SRBench**：评估 LLM 驱动科学方程发现的基准，含 LSR-Synth（合成数理化生材问题）和 LSR-Transform（变换公式恢复）两个互补 track。

## 可复现要素
- 数据集：LLM-SRBench（公开），LSR-Transform-Anon 由作者构造并随论文发布。
- 代码：开源于 https://github.com/tsinghua-fib-lab/SRHarness。
- 关键超参：R1-C1-L30-K1 调度配置；每问题最多 30 步精炼；output token 上限 DeepSeek 4096/GLM 8192；训练集 80/20 划分（seed=42）；实验 seed=26091320。
- Backbone：DeepSeek-v4-flash-0731、GLM-5.3-flash、DeepSeek-v4.1-flash、DeepSeek-v4-pro-0813、GLM-5.3。
- 动作集：statistics analysis、relationship analysis、read skill、evaluate formula、submit formula、constant fit、call pysr、call sindy、code executor。
- 基线复现：使用官方开源实现，匹配 backbone，调用预算按 token 对齐而非 call 数对齐。
