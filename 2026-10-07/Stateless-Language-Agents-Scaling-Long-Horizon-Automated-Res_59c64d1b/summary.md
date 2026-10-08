---
title: "Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res"
source: https://arxiv.org/pdf/2610.07625v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:23:31"
field: "自动化科学发现与多代理系统"
keywords: ["stateless agents", "automated research", "long-horizon search", "multi-agent coordination", "context management", "code optimization"]
innovations: ["提出有状态搜索+无状态代理原则，将研究状态外置到harness而非agent会话中", "Harness控制的上下文重建实现Advisor全局规划与Worker局部实验的清晰分工", "从共享检查点继续的消融协议精确分离上下文隔离与工作分配的累积贡献"]
benchmarks: ["FrontierSWE", "SOL-ExecBench", "FrontierCS", "Anthropic VLIW SIMD Kernel Optimization"]
---

# 论文速读：Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res

## 一句话总结
本文提出**无状态语言代理（Stateless Language Agents, SLAs）**框架，以"有状态搜索、无状态代理"为核心原则，通过将研究状态保留在外部 harness 中而非代理对话历史中，解决了长周期自动化研究系统中代理重复工作、提前停止等问题，在软件工程、内核优化和算法设计任务上实现了更强的最终性能，同时显著减少 token 消耗。

## 研究问题与动机
1. **长周期推理不等于持续进展**：现有自动化研究系统在长预算下往往出现代理重复对方工作、停滞实验（如 CORAL 在 GPU 内核任务最后 1500 次会话中 98% 无工具调用）的现象，仅靠增加推理量无法保证有效搜索。
2. **评估基准饱和过早**：主流基准（如 26-circle packing）在 5M tokens 内即饱和，无法检验上下文管理和实验选择等会随经验累积而放大效果的组件设计。
3. **经验保留与上下文控制的矛盾**：候选解、评估结果等可复用证据若随对话增长被反复处理，会消耗大量 token 并退化代理行为（Context Rot），但丢弃历史又可能丢失后期仍有价值的发现。
4. **累积证据难以转化为有效探索**：记录的发现不自动决定下一步尝试什么；多代理并行时容易重复相同特征实现（CORAL 轨迹中多个代理从同一父代实现相同功能并得分相同）。

## 核心贡献（创新点）
1. **提出"有状态搜索+无状态代理"原则**：将研究状态（候选解和测量结果）完全保留在外部 harness 中，每个代理每次调用都是全新的独立会话，看到的上下文由 harness 显式构造，而非对话历史的副产品——与 CORAL/SwarmResearch 等将历史携带于代理会话中的方法形成本质区别。
2. **Harness 控制的上下文重建机制**：Advisor 看到的是跨搜索方向的证据摘要（按方向分组的结果、工作状态、进展趋势），Worker 只看到自己的任务、保留候选和局部反馈，实现了全局规划与局部实验的清晰分工。
3. **基于证据的工作分配策略**：Advisor 将全局证据转化为具体实验分配，使并行 Worker 执行不同实验而非重复工作；当代理重定向时从全局最佳重置，新方向即使起点较差也能独立发展。
4. **长周期评估的新范式与消融协议**：首次使用累积 token 作为统一预算轴并在高达 10 亿 token 的长周期下系统评估；提出从共享检查点继续运行的消融协议，分离上下文隔离与工作分配各自的贡献。

## 方法详解
SLA 框架采用交替的全局规划与局部实验循环，每个 epoch 包含以下机制：

1. **Harness 拥有的研究状态**：候选代码、评分结果、实验记录全部由 harness 持久化保存，不因代理调用结束而消失；每次检查点保存完整状态支持恢复和可控消融。

2. **Advisor 上下文重建**：每个 epoch 开始时，harness 为 Advisor 构造结构化证据摘要，包括任务规格、当前全局最佳代码和分数、按搜索方向分组的最近尝试与结果、Worker 状态、进展趋势及任务特定诊断；区分"测量的无改进"与"实现/运行时/正确性的不 conclusive 失败"，避免将实现错误误判为方向穷尽。

3. **Worker 上下文隔离**：每个 trial 中，Worker 仅接收任务规格、其具体分配、已保留的候选代码和紧凑的局部反馈（上次 trial 状态、测量分数、可用性诊断）；Worker 无法访问其他 Worker 的工作空间或完整研究历史，其他方向的证据仅通过 Advisor 分配传递。

4. **Evidence-driven 工作分配**：Advisor 基于跨方向证据决定分散到多个方向或集中攻击有前景的瓶颈；每个 Worker 可在多次本地 trial 中逐步完善分配；即使本地候选落后于全局最佳，harness 仍保留其改进，给替代方法成熟时间；当 Advisor 重定向 Worker 时，Worker 从全局最佳重启，首次成功评估的候选成为新本地基线。

5. **状态性与隔离性强制保障**：Advisor 每 epoch 在新会话启动；Worker 以非特权进程运行，受 Linux Landlock 文件系统限制，网络访问受限，harness 在 trial 间清除保存的会话；候选在 Worker 环境外由受保护的评估器代码独立评估；按角色记录 token 使用并支持检查点。

## 实验与结果
**数据集与任务**：
- **FrontierSWE**（软件工程中）：libexpat→x86-64 Assembly、Git→Zig、Dart→Haskell、Lua Native Compiler（4 个不需要 GPU 的任务）
- **SOL-ExecBench**（GPU 内核优化）：#1（attention softmax/dropout/value matmul 反向传播）、#58（MoE token 排序+前缀和）、#210（fused residual addition+RMS normalization）
- **FrontierCS**（算法设计）：Structured-LWE 密码学任务
- **Anthropic VLIW SIMD 内核优化**

**评估基线**：EvoX、CORAL、SwarmResearch（均使用官方实现、未修改搜索逻辑）

**主要结果数字**：
- SLA 在**所有评估任务上达到最佳全预算结果**
- Anthropic 内核优化（Codex）：SLA 1112.0±8.9 cycles vs. 最强基线 SwarmResearch 1275.7±133.9，**减少 12.8% cycles**；达到最强基线最终性能仅需 **67.9M tokens vs. 986.3M（减少 93.1%）**；Claude Code 配置下减少 84.4%
- FrontierSWE：相比最强 per-task 基线平均提升 **4.95 分**；在 175M（25%预算）时落后于最强基线，但在 700M（全预算）时领先
- SOL-ExecBench #58：达到最强基线性能仅需 **14.5M tokens（减少 95.9%）**
- FrontierCS Structured-LWE：59.5 分 vs. 基线 58.0，提升 1.5 分

**消融结果（从共享检查点继续 100M tokens）**：
- 移除 Advisor 上下文重建：在所有比较中降低平均进展（Anthropic 内核 200M→300M 阶段下降 69.3%）
- 移除 Worker 上下文隔离：在 100M token 延续中降低 5–11 cycles，但在完整 1B token 运行中差距扩大至 **224 cycles**（1336 vs. 1112），说明无聚焦上下文的代价随长周期累积放大
- 移除 Advisor 分配：Worker 重复实现相同功能，在所有比较中降低进展

**Advisor 成本极低**：消耗 **0.24–0.51% tokens** 和 **1.20–2.29% 模型成本**，远低于 SwarmResearch Shepherd 的 8.39–10.70% tokens 和 6.14–7.35% 成本。

**Worker 扩展性**：从 W=1 到 W=15 将 Anthropic 内核运行时间缩短 6.6×，FrontierSWE 平均缩短 7.3×；最佳 Worker 数量因任务而异（Dart 最优 W=3，Git/libexpat 最优 W=7，Lua 最优 W=1）。

## 相关工作脉络
1. **LLM-guided 搜索框架**（FunSearch、AlphaEvolve、OpenEvolve、EvoX）：将 LLM 生成程序嵌入进化搜索循环，每个候选由一次或少数几次模型调用生成——SLA 与其本质区别在于 SLA 使用多步 coding agent 并在长周期中通过状态管理维持进展，而非固定轮次的直接调用。
2. **CORAL**：运行并行 agent 自主选择实验并通过持久化内存和心跳触发反思共享发现，每个 agent 保持单一长运行会话——SLA 的核心差异是将所有研究历史移出 agent 会话，由 harness 统一管理并通过显式分配协调并行工作。
3. **SwarmResearch**：使用 Shepherd agent 通过父选择、agent 类型选择和选择性上下文共享协调 Search Agents，Shepherd 运行长会话——SLA 的 Advisor 是无状态的，每次从当前证据重新推理，且消耗不到 0.6% 的 token。
4. **AI Scientist 系列**（The AI Scientist、Agent Laboratory、AI-Researcher）：自动化科学发现全流程（假设生成、文献综述、实验、写作）——SLA 聚焦于可执行解的开放式搜索，目标是有明确评估器的单一问题，适合隔离研究 agentic 过程如何在长周期中维持进展。
5. **上下文记忆与自改进方法**（Dynamic Cheatsheet、ACE、ReasoningBank）：通过累积经验在 memory/context 中改进 agent——这些方法主要是任务间（inter-task）的，经验跨任务检索/转移；SLA 聚焦任务内（intra-task）单问题长周期优化，核心问题是 Growing record 如何保持可用性和并行 agent 如何分工。

## 局限性与未来方向
1. **长周期运行成本高昂**：大多数配置仅使用单次运行，仅 Anthropic 内核优化与 Codex 和检查点消融进行了三次独立运行复现；无法对所有任务和配置进行全面统计验证。
2. **成本估算不完整**：比较按 token 匹配，但成本估算排除了运行和评估实验的计算开销。
3. **任务类型受限**：所有任务都有可执行的评估器，模糊目标或评估缓慢的目标尚未测试。
4. **消融仅改变上下文而非状态性本身**：所有变体保持 agent 无状态，仅改变进入每个上下文的内容或 Worker 是否接收分配，未能直接对比"有状态 vs 无状态"的根本差异。
5. **未来方向**：将 Advisor 角色拆分为多个各看部分研究状态的 Advisor；将长运行产生的短 trial episodes 直接用于强化学习训练；结合跨任务 memory（如 ACE playbook）作为 Advisor 上下文的种子。

## 研究启发与可借鉴点
1. **上下文显式设计的价值**：SLA 证明了"每个 agent 看到什么"应作为显式设计选择而非对话历史的副产品——这一原则可迁移到任何需要长周期 multi-agent 协作的场景，如持续机器学习工程、自动化代码库维护等。
2. **从检查点继续的消融协议**：通过保存完整研究状态并在相同检查点上运行不同变体，可以精确分离各组件贡献并量化其随预算增长的累积效应——这是一种值得推广的实验方法论。
3. **协调与实现的解耦**：SLA 证明显式协调（Advisor）可低至 0.6% token 且独立选择 Worker 模型能在匹配成本下获得更优任务特定性能——这一架构模式适用于需要长期自主运行的 agent 系统。
4. **短评估周期的误导性**：本文揭示了在 FrontierSWE 等任务上，25% 预算时排名的方法与 100% 预算时完全不同——呼吁社区在多预算点评估并报告，而非仅报告最终或饱和结果。
5. **训练数据生成潜力**：SLA 的短 trial episodes（bounded context + score feedback）可直接作为 RL 训练数据，无需模型训练于完整十亿 token 运行——这为高效利用长周期 agent 运行结果训练下一代模型提供了可行路径。

## 关键术语表
**Stateless Language Agent (SLA)**：每次调用都是全新会话的 agent，不保留任何对话历史，所有研究状态由外部 harness 管理和重建。
**Harness**：负责拥有研究状态（候选解和测量结果）、重建 agent 上下文、执行评估并记录 token 使用的中央控制框架。
**Advisor**：SLA 中负责全局规划的角色，看到跨搜索方向的证据摘要，将证据转化为具体实验分配给并行 Worker。
**Worker**：SLA 中负责局部实现的角色，仅看到自己的分配和保留候选，在隔离工作空间中进行多次本地 trial。
**Cumulative tokens**：输入和输出 token 的累积总和，作为统一且可复现的长周期评估预算轴。
**Context Rot**：随着输入 token 增加导致 LLM 性能退化的现象，是长周期搜索中需要管理的关键问题。
**Research state**：候选代码、评分结果、实验记录的持久化集合，由 harness 拥有而非任何 agent 会话持有。
**Checkpoint continuation ablation**：从相同检查点（研究状态快照）继续运行完整 SLA 和各变体的消融协议，用于分离组件贡献。

## 可复现要素
- **数据集**：FrontierSWE、SOL-ExecBench、FrontierCS、Anthropic VLIW SIMD kernel optimization task——均使用官方公开任务材料和种子解
- **代码/权重是否开源**：论文使用了 Codex (GPT-5.5) 和 Claude Code (Claude Opus 4.8)，SLA 框架实现未明确声明开源，基线使用官方 pinned upstream release
- **关键超参**：默认 Worker 数量 W=3，每 epoch 每 Worker 3 次本地 trial；建议 per-task 调整 Worker 数量；推理 effort 设为 high
- **评估预算**：Anthropic 内核 1B tokens，FrontierSWE 700M tokens per task，SOL-ExecBench 500M tokens per task，FrontierCS 200M tokens
