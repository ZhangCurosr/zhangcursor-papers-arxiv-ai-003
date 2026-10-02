---
title: "WEFT-Scaling-Tool-Use-Post-Training-for-General-Purpose-Agen"
source: https://arxiv.org/pdf/2609.36887v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:29:15"
field: "Agent Post-Training"
keywords: ["tool-use post-training", "agent interaction system", "execution-driven self-evolution", "atomic-turn credit assignment", "prefix-preserving rejection sampling", "MegaMCP", "reinforcement learning for agents"]
innovations: ["将整个agent交互系统（环境、任务、harness、评估器）纳入scaling范畴，通过执行驱动的自演化协同改进各组件", "提出atomic-turn group-relative credit assignment，在同源同状态下采样多segment计算细粒度学习信号", "设计prefix-preserving rejection sampling，以原子turn为粒度保留已验证进度并局部重采样"]
benchmarks: ["BFCL V4", "τ²-Bench", "Claw-Eval", "Toolathlon-Verified", "AutomationBench"]
---

# 论文速读：WEFT-Scaling-Tool-Use-Post-Training-for-General-Purpose-Agen

## 一句话总结
论文提出 WEFT（Whole-system Evolution For Tool-use Post-training）框架，将工具使用 post-training 的扩展对象从"可执行环境"延伸至完整的 agent 交互系统（环境、任务、agent harness、评估器），通过执行驱动的自演化与稳定训练机制，显著提升 LLM agent 的工具调用与多步任务执行能力。

## 研究问题与动机
- **核心问题**：现有工具使用 post-training 工作主要聚焦于"可执行环境"的程序化合成与规模扩展，但仅扩展环境并不保证模型性能的线性提升。
- **动机一**：可靠的 learning signal 依赖 agent 交互系统中所有组件（环境、任务、agent harness、评估器）的协同一致；孤立扩展环境可能引入任务不可行、验证错误或工具实现缺陷等问题，导致错误的学习信号。
- **动机二**：执行失败可能源于环境/任务/验证器缺陷，而非模型策略不足；若将这些失败归因于策略，会惩罚有效行为并引入误导性学习信号。
- **动机三**：需要一种机制，使执行经验既能作为 policy 学习的数据，也能作为识别和修订交互系统缺陷的证据。

## 核心贡献（创新点）
1. **将整个 agent 交互系统纳入 scaling 范畴**：提出 WEFT 端到端框架，同时扩展环境广度（stateful environment synthesis）、任务复杂度（cross-MCP task composition）与交互多样性（multi-view/multi-harness rollouts），区别于仅扩展环境的 EnvScaler、ScaleEnv 等工作。
2. **执行驱动的自演化机制**：利用执行轨迹和状态证据归因失败来源（区分 policy failure 与 environment/task/verifier 问题），定向修订相应组件，并通过 fresh rollout 评估修订效果并发现新问题，形成闭环演化；这是首个将"执行经验用于修订生成数据的基础设施本身"的方法。
3. **Prefix-preserving rejection sampling（SFT 阶段）**：以 atomic-task turn 为拒绝单元，保留已验证成功的 prefix 及其状态，仅重采样失败原子任务，避免端到端采样中因局部失败丢弃已完成进度。
4. **Atomic-turn credit assignment（RL 阶段）**：在同源相同状态下采样多个 atomic-turn segment，基于当前原子任务二元完成结果计算 group-relative advantage，实现比 trajectory-level GRPO 更细粒度的学习信号分配。
5. **MegaMCP 基础设施**：通过共享 MCP 服务进程、隔离各 rollout 私有状态、快照支持状态恢复与重试，显著降低并发 rollout 的数据传输、内存占用与冷启动延迟（上传减少 95.5%，内存减少 77.6%-95.7%，首调用延迟降低 54.1%）。

## 方法详解
**3.1 可扩展 agent 交互系统构建**
- **Stateful Environment Synthesis**：从 MCP tool 文档推断共享实体、读写依赖与状态转换，生成数据库 schema 与 contract，指导 FastAPI backend 构建；通过 E2E 执行验证（正常/边界调用、查询/更新一致性检查）筛选出独立可执行的 MCP server。
- **Cross-MCP Task Composition**：
  - 构建 certified atomic tasks：通过 E2E 执行验证、tool-chain orthogonality 过滤（$\ell_i \preceq \ell_j$ 判定冗余）、tool coverage 度量三个检查筛选原子任务。
  - Coverage-aware sampling：按五档难度分布（Simple/Standard/Moderate/Complex/Expert）控制原子任务数量、MCP 数量与领域广度，优先采样覆盖不足的 domain/MCP。
  - Grounded task graph：将选中的 atomic tasks 通过实体绑定（entity bindings）连接为 DAG，区分给定信息与工具检索信息，确定拓扑执行顺序。
  - Progressive state grounding：三轮初始状态构建（实体初始化→关系补全→上下文丰富），生成可重置的初始快照。
- **Multi-View and Multi-Harness Rollouts**：
  - Task views：Agentic（完整 briefing）与 SimUser（按依赖顺序逐步披露请求，支持 vague/partial/full 三种显式度）。
  - Harness scaling：同一任务路由至 ReAct、OpenClaw、Hermes 三个 harness，保留各自 prompt、memory、planning loop、tool-call protocol。

**3.2 执行驱动的自演化**
- **Verification from Execution Evidence**：对轨迹 $\tau$ 设 K 个 checkpoint，每个 checkpoint 条件满足得 $r_k(\tau)=1$，轨迹得分 $R(\tau) = \frac{1}{K}\sum_{k=1}^K r_k(\tau)$。
- **Attribution-Guided Revision**：归因 agent 分析任务图、轨迹、checkpoint 结果与状态证据，区分 policy failure 与 environment/task/verifier 问题；evolution agent 定向修订对应组件（环境问题→backend、任务/状态问题→task graph 与 initial state、评估问题→verifier）。
- **Dependency revalidation**：修订后重新实例化受影响的环境与任务状态，fresh rollout 评估修复效果，揭示未解决或新问题，交替执行与修订实现迭代演化。

**3.3 稳定 post-training**
- **Prefix-Preserving Rejection Sampling**：teacher 按依赖顺序执行任务，成功 trace 及状态保留；失败时回滚至 turn 前历史与状态，仅重采样该原子任务（最多 k 次重试），所有通过的任务组成 SFT 轨迹。
- **Atomic-turn Credit Assignment**：
  - 每 turn t 在相同历史与相同状态（由 MegaMCP 恢复）下采样 G 个 segment。
  - 原子任务 verifier 给出二元奖励 $r_t^{(i)} \in \{0,1\}$。
  - 组内归一化 advantage：$A_t^{(i)} = \frac{r_t^{(i)} - \bar{r}_t}{\text{sd}(\mathbf{r}_t) + \epsilon}$，其中 $\bar{r}_t = \frac{1}{G}\sum_{i=1}^G r_t^{(i)}$。
  - Clipped GRPO 仅对当前 turn policy tokens 应用 advantage；检测不良模式（无进展重复、格式错误、接口违规）时，对 flagged turn 设置 positive advantage 截断为 0（$\min(A_t^{(i)}, 0)$），保留 negative advantage。
  - Task filtering：用 rubric-guided LLM judge 与 executable verifier 独立评分，保留 Pearson 与 Spearman 相关系数均达标任务。
- **MegaMCP**：
  - Registry：内容寻址存储 MCP metadata、源码与 tool definitions，单调递增版本号。
  - Process sharing：prefork master 预加载通用依赖，按需创建 worker；兼容会话共享 Python 进程，通过静态分析限制进程全局副作用。
  - Private state：每个 rollout 持有独立 database（copy-on-write 克隆初始快照）与 workspace；快照支持状态恢复。
  - Idle-session management：冷却窗口后 offload 空闲 session group，下次调用 reload。
  - 运行时监控：磁盘/内存压力检测、worker 健康迁移、generation fencing 防止非法写入、reaper 幂等完成中断关闭。

## 实验与结果
**数据集与模型**：构建 8,172 个可执行 MCP（64,755 工具）、41,695 个 certified atomic tasks、11,884 个 composed tasks（中位长度 20 atomic-task turns）；对 Qwen3-8B、Qwen3-14B、Qwen3.5-35B-A3B 进行 post-training。

**主要结果（Table 1）**：
- WEFT-8B 与 WEFT-14B 在所有 evaluated matched-size environment-scaling baselines 上全面超越 BFCL V4、τ²-Bench、Claw-Eval 三项基准的 aggregate scores。
- WEFT-14B 较 Agent-World-14B 分别提升 6.41、2.23、12.27 个百分点（BFCL V4、τ²-Bench、Claw-Eval）。
- WEFT-8B 较 Agent-World-8B 分别提升 0.88、0.45、5.28 个百分点。
- BFCL Memory 与 τ²-Bench Telecom 成为跨规模稳定优势项：WEFT-14B 在 BFCL Memory 上较 Agent-World-14B 提升 25.35 点，在 τ²-Bench Telecom 上提升 20.22 点；WEFT-35B-A3B 在 BFCL Memory 达 66.02%（35B-685B 组最高）。
- WEFT-35B-A3B 在长 horizon 基准 Toolathlon-Verified 达 45.99%，AutomationBench 达 26.50%。

**消融实验**：
- Task views + harness scaling（Table 2）：固定任务集下，Agentic + SimUser 比单 view 高 3.18 点；加入 OpenClaw + Hermes 再提升 9.71 点。
- Self-evolution（Figure 6）：3 轮自演化使选定 teacher 轨迹中 tool-error rate 从 1.76% 降至 0.96%（相对下降 45.5%），Toolathlon-Verified、AutomationBench、Claw-Eval 分别提升 5.25、4.33、3.65 点。
- Prefix-preserving sampling（Figure 7）：k=3 时，较 end-to-end sampling 在 Toolathlon-Verified 提升 1.85 点、AutomationBench 提升 2.17 点。
- RL credit assignment（Table 3）：Atomic-turn GRPO 较 SFT 在 τ²-Bench 提升 5.69/4.43 点（8B/14B），而 trajectory-level GRPO 在 τ²-Bench 出现退化（8B -3.17 点、14B -0.84 点）。

**MegaMCP 性能（Figure 9、Table 5-7）**：
- 1,000 tasks 时 sandbox 上传从 773.5 MiB 降至 34.8 MiB（减少 95.5%）。
- 内存占用减少 77.6%（52 instances/8 processes）至 95.7%（50 sessions/1 process）。
- 冷启动首调用延迟从 10.455s 降至 4.794s（减少 54.1%）。

## 相关工作脉络
- **EnvScaler / ScaleEnv / Agent-World**：聚焦程序化合成可执行环境扩展工具覆盖率；本文指出仅扩展环境不够，需同步演化任务、harness、verifier 等整套交互系统。
- **Eigendata / AutoForge / SPADE**：自演化/自适应合成环境相关工作；本文独特之处在于通过 execution trace 归因区分 policy failure 与系统缺陷，定向修订具体组件而非整体重新生成。
- **PivotRL / BPO / GiGPO**：credit assignment 方法；本文采用 atomic-turn group-relative advantage，以当前原子任务二元完成结果为 reward source，而非 trajectory-level terminal return 或 pivot segment comparison。
- **ToolACE / ToolFlow / Magnet**：trajectory 合成与 quality control；本文独特地使用 prefix-preserving rejection sampling，在局部失败时保留已验证 prefix 并重采样失败 turn。
- **ClawGym / WebArena / OSWorld**：agent evaluation benchmarks；本文在 BFCL V4、τ²-Bench、Claw-Eval、Toolathlon-Verified、AutomationBench 上验证，涵盖 tool calling、multi-turn interaction 与 long-horizon workflow。

## 局限性与未来方向
- 实验主要集中在 Qwen3 系列模型，在其他架构（如 LLaMA、DeepSeek）上的泛化性未充分验证。
- Self-evolution 仅演示了 3 轮，未见更深层演化对系统质量与 training signal 稳定性的长期影响分析。
- MegaMCP 的 shared serving 在高并发 warmed 场景下 latency 高于 prewarmed local serving（Table 6 显示 1,000 tasks 下 TTFT p50 为 39.281s vs 0.009s），说明 session provisioning 开销仍有优化空间。
- Verifier consistency filtering 依赖 LLM judge 与 executable verifier 的双重评分，可能受 judge 模型质量影响；未讨论 verrier 本身的规模化验证问题。
- 长 horizon 任务（Toolathlon-Verified、AutomationBench）的绝对性能仍有较大提升空间（45.99%、26.50%），反映复杂多工具工作流的挑战尚未完全解决。

## 研究启发与可借鉴点
- **"执行经验双角色"范式**：将 execution trace 既作为 policy 学习数据又作为系统诊断证据的思路，可迁移至其他 agent post-training 场景（如代码生成 agent、GUI agent）。
- **Prefix-preserving rejection sampling**：对具有可验证中间状态的长 horizon 任务，以原子 turn 为粒度进行局部重采样而非端到端拒绝，显著提升 SFT 数据质量；可借鉴到任何具有可分解子任务的训练 pipeline。
- **Atomic-turn credit assignment**：在同源同状态下采样多个 segment 计算 group-relative advantage，避免 trajectory-level reward 的信用稀释问题；适用于任何具有明确中间验证点的 multi-turn agent 任务。
- **MegaMCP 的隔离共享架构**：将共享工具服务与 rollout 私有状态分离，通过快照支持状态恢复与并发重试，对需要大规模并发的 agent 训练系统具有通用参考价值。
- **Multi-view + Multi-harness 数据增强**：同一任务通过不同 disclosure view 与不同 harness 执行，可低成本扩展训练多样性；适合在合成数据受限场景下挖掘更多训练价值。

## 关键术语表
**WEFT（Whole-system Evolution For Tool-use Post-training）**：一种将工具使用 post-training 的扩展对象从单一环境扩展至完整 agent 交互系统的端到端框架。

**MCP（Model Context Protocol）**：标准化工具接口协议，定义 tool description、input/output schema、state schema 与 execution contract，支持环境合成与 cross-MCP 任务组合。

**Atomic task**：MCP 中最小可独立执行与验证的任务单元，通过有界 tool call 序列实现单一意图，是任务组合与 prefix-preserving sampling 的基本粒度。

**Prefix-preserving rejection sampling**：以 atomic-task turn 为拒绝单元的 SFT 数据构建方法，失败时回滚至 turn 前状态并重采样该 turn，保留已验证 prefix。

**Atomic-turn credit assignment**：RL 阶段以当前原子任务的二元完成结果为 reward source，在同源相同状态采样多个 segment 计算 group-relative advantage 的信用分配方法。

**MegaMCP**：支持大规模并发 rollout 的基础设施，通过共享 MCP 服务进程、隔离各 rollout 私有状态、快照支持状态恢复与重试，降低部署与内存开销。

**Execution-driven self-evolution**：利用执行轨迹与状态证据归因失败来源，定向修订环境/任务/验证器等组件，并通过 fresh rollout 评估修订效果的迭代优化循环。

**Task-disclosure view**：固定任务目标与完成条件，但变化信息揭示方式（Agentic 完整 briefing vs SimUser 逐步披露、vague/partial/full 三种显式度）以扩展交互多样性。

## 可复现要素
- **数据集**：论文构建了 8,172 个 MCP、64,755 工具、41,695 个 certified atomic tasks、11,884 个 composed tasks；论文未明确声明公开，但附录提供了详细构建流程与算法伪代码。
- **代码/权重**：WEFT 代码与模型权重论文未明确开源声明；基线模型（Qwen3-8B/14B/35B-A3B）可从官方渠道获取。
- **关键超参**：
  - SFT：prefix-preserving rejection sampling，每 turn 最多 k=3 次重试。
  - RL：Adam optimizer，lr=10⁻⁶，constant schedule，BF16，PPO clip [0.80, 1.28]，reference KL coefficient 10⁻³，max context 131,072 tokens，max assistant turns 100。
  - RL task selection：从 5,000 candidate tasks 中通过 rubric-executable score consistency 筛选 1,954 tasks。
  - 训练硬件与具体 steps/epoch 数论文未明确提及。
