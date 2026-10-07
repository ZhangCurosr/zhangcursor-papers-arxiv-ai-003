---
title: "Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res"
source: https://arxiv.org/pdf/2610.07625v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:53:42"
field: "自动化科学研究(AutoResearch)"
keywords: ["自动化研究", "无状态代理", "长时程评估", "多agent系统", "上下文管理", "token效率"]
innovations: ["stateful search with stateless agents架构", "harness-controlled context reconstruction机制", "evidence-driven work assignment设计"]
benchmarks: ["FrontierSWE", "SOL-ExecBench", "Anthropic VLIW SIMD Kernel", "FrontierCS Structured-LWE"]
---

# 论文速读：Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res

## 一句话总结
论文提出**无状态语言代理（Stateless Language Agents, SLA）**框架，通过"状态化搜索 + 无状态代理"的设计原则，解决长时程自动化研究中累积上下文导致重复实验、过早停滞等问题，在多项软件工程和内核优化任务上以显著更少的token消耗达到更强最终性能。

## 研究问题与动机
- **长时程运行中的状态与上下文困境**：候选解、评估结果和失败尝试是可重用证据，但回放不断增长的上下文会消耗大量token并劣化agent行为
- **累积证据未能转化为有效探索**：已记录发现不能自动决定下一步实验，在多agent系统中出现重复实现相同功能、互相覆盖工作的问题
- **现有评估无法检验长时程特性**：多数基准测试早期饱和（如26-circle packing在5M tokens内差异仅10^-10），短期预算评估无法反映设计选择（上下文管理、实验选择）的累积效应
- **scaling推理不等于产生进步**：在CORAL的GPU内核任务中，最后改进出现在231M tokens，之后98%的session无工具调用但token仍在消耗

## 核心贡献（创新点）
1. **提出stateful search with stateless agents原则**：将研究状态（候选解、测量结果）与agent上下文分离，harness拥有持久状态并重建每角色专属上下文
2. **Harness-controlled context reconstruction机制**：Advisor接收跨搜索方向的证据摘要，Worker仅接收本地指派与反馈，避免全量历史回放
3. **Evidence-driven work assignment机制**：Advisor基于全局证据分配具体实验给并行Workers，防止重复实现与竞争
4. **长时程AutoResearch评估方法论**：证明短期评估可能误判方法性能，提出多累积token预算评估方案，发现SLA在175M tokens落后但在700M tokens反超
5. **开源框架实现与全面基准测试**：实现SLA框架并在FrontierSWE、SOL-ExecBench、Anthropic内核优化、FrontierCS等多任务验证

## 方法详解
**核心架构**：
- **无状态设计**：每个Agent invocation是全新session，不保留对话历史；研究状态完全由harness管理
- **双角色设计**：
  - **Advisor**：无状态，每次epoch接收harness重建的跨方向证据摘要（任务规格、全局最优解、近期尝试、方向分组结果、进度等），输出并行Worker的具体指派
  - **Worker**：无状态，每次trial仅接收任务规格、自身指派、保留的候选解、本地反馈（上次trial状态、分数、正确性诊断）

**上下文重建机制**（§3.1）：
- Advisor上下文包含：任务规格、当前全局最优代码和分数、证据摘要（按搜索方向分组的近期尝试结果、Worker状态、进度、任务特定诊断），区分"测量无改进"与"实现/运行时/正确性不可判定失败"
- Worker上下文包含：任务规格、指派、保留候选、紧凑本地反馈；无法访问其他Worker工作空间或完整研究历史

**证据驱动工作分配**（§3.2）：
- Advisor根据跨方向证据选择实验策略（diversify/concentrate）
- 限制增量lane数量（最多一个Worker做小修改），其他必须尝试机制级改变
- 保留非最优但进步的局部候选，给予替代方案成熟时间
- Worker重定向时从全局最优重启，首个成功评估候选成为新本地基准

**实现细节**（§3.3）：
- 使用OpenAI Codex或Claude Code
- 通过技术而非请求强制无状态：Advisor每个epoch新开session，Worker在Linux Landlock文件系统限制下作为非特权进程运行，harness清除保存的session
- 独立评估器在保护环境中评估候选，harness记录token使用和检查点

## 实验与结果
**任务与数据集**：
- FrontierSWE：4个软件工程任务（libexpat→x86-64 Assembly、Git→Zig、Dart→Haskell、Lua Native Compiler）
- SOL-ExecBench：3个GPU内核优化任务（#1注意力softmax/dropout、#58 MoE token排序、#210融合残差加法）
- Anthropic VLIW SIMD内核优化
- FrontierCS Structured-LWE密码学任务

**基线方法**：
- EvoX（共演化候选与搜索策略）
- CORAL（多agent共享persistent memory和heartbeat reflection）
- SwarmResearch（Shepherd协调Search Agents）

**主要结果**：
- **最强解**：SLA在所有评估任务上取得最佳full-budget结果（Table 1-3）
- **token效率**：在Anthropic内核优化中，SLA以**93.1%更少tokens**（Codex）/ **84.4%**（Claude Code）达到最强baseline的最终性能；SOL-ExecBench #58达**95.9%** reduction
- **反超现象**：在FrontierSWE，SLA在175M tokens（25%预算）落后所有任务最强baseline，但在700M tokens（full budget）领先全部4个任务
- **消融结果**（Table 4）：从共享checkpoint移除任一设计选择均减少progress；移除Worker isolation成本在100M tokens续跑为5-11 cycles，在全1B tokens跑中达224 cycles
- **协调成本**：Advisor仅消耗**0.24-0.51% tokens**和1.2-2.3%成本，远低于SwarmResearch Shepherd的8.4-10.7% tokens和6.1-7.4%成本

**关键数字**：
- Anthropic内核：SLA达1112.0±8.9 cycles，最强baseline SwarmResearch为1275.7±133.9；匹配目标仅需67.9M tokens vs SwarmResearch 986.3M tokens
- FrontierSWE平均提升：+4.95 points vs最强baseline
- SOL-ExecBench平均提升：+5.5%
- FrontierCS Structured-LWE：59.5 vs 58.0

## 相关工作脉络
1. **LLM-guided search / 进化搜索**：FunSearch、AlphaEvolve、OpenEvolve、EvoX等将LLM生成程序嵌入进化循环；与SLA的区别在于这些方法每个候选仅用1-几次model call，且状态管理方式不同
2. **多agent AutoResearch**：CORAL和SwarmResearch将实验过程置于tool-using coding agents控制下；核心差异在于它们依赖agent自身long-running session存储研究历史，而SLA将状态移至harness
3. **长上下文agent问题**：Context Rot研究（Hong et al.）表明累积输入token劣化LLM性能；SLA通过周期性上下文重建规避此问题
4. **Memory/context engineering**：ACE、Dynamic Cheatsheet、ReasoningBank等方法累积跨任务经验；SLA专注单任务 intra-task 设置，积累问题是特定的候选/结果记录
5. **AI Scientist系统**：The AI Scientist、Agent Laboratory等自动化多阶段科研；SLA可作为其实验引擎
6. **Self-improvement through memory**：Reflexion、ExpeL等方法让agent从过去轨迹学习；SLA使agent无状态但通过harness状态实现progress

## 局限性与未来方向
- **评估成本限制**：长时程运行昂贵（数百至数千美元），多数配置仅单次运行，仅Anthropic内核和checkpoint消融有三次独立复现
- **仅测试可执行评估器任务**：目标模糊或评估缓慢的任务尚未验证
- **消融仅改变上下文内容**：未测试保留跨invocation对话的变体（如native subagent orchestration）
- **未来方向**：
  - 将SLA trials直接用于RL训练（因每次trial有bounded context和评分）
  - 拆分Advisor角色为多个子Advisor各看部分state
  - 结合cross-task memory（如ACE playbook）作为Advisor上下文种子
  - 评估更模糊/慢评估目标的适用性

## 研究启发与可借鉴点
1. **状态-上下文分离原则**：可将"研究状态外置 + 角色专属上下文重建"思路迁移到任何长时程多agent系统，避免context rot和重复工作
2. **明确工作分配优于隐式协调**：证据驱动的显式指派比共享memory/heartbeat reflection更能防止并行agent重复工作，值得在多agent科研/工程系统中借鉴
3. **长时程评估方法论**：证明短期预算评估可能误判方法（SLA在25%预算落后但full budget领先），建议评估AutoResearch系统时使用多token预算曲线而非单一终点
4. **廉价协调层设计**：Advisor消耗<0.6% tokens却显著提升progress，说明轻量显式协调远优于让各agent独立决策；可应用于分布式计算/研究系统
5. **Checkpoint-continuation ablation设计**：从共享checkpoint比较变体，隔离设计选择贡献，是评估复杂框架组件价值的严谨方法

## 关键术语表
**Stateless Agent（无状态代理）**：不在invocation间保留任何状态的agent，每次调用均为全新session，研究状态完全由外部harness管理  
**Harness（控制层）**：拥有研究状态（候选解、测量结果）、重建agent上下文、执行评估的核心框架组件  
**Cumulative Tokens（累积token）**：所有agent角色的输入+输出token总和，作为可控的scaling轴和预算单位  
**Evidence Summary（证据摘要）**：harness按搜索方向分组的近期尝试结果摘要，区分测量无改进与不可判定失败  
**Search Posture（搜索姿态）**：Advisor选择的策略——diversify（多个方向并行）或concentrate（集中资源于有希望的bottleneck）  
**Incremental Lane（增量lane）**：对小修改（priority/threshold tweak等）开放的Worker lane，限制最多一个同时存在  
**Context Reconstruction（上下文重建）**：每次invocation从harness状态为每个角色重建专属上下文的机制  
**Regime（机制流派）**：Worker探索的不同优化技术家族（如Triton、inline CUDA、vectorization等）

## 可复现要素
- **数据集**：FrontierSWE、SOL-ExecBench、Anthropic VLIW SIMD、FrontierCS Structured-LWE（论文提供任务描述和public materials）
- **代码开源**：论文提及SLA框架实现，但未明确说明仓库URL；基线使用pinned upstream release
- **模型**：OpenAI Codex v0.152.1 + GPT-5.5；Claude Code v2.1.258 + Claude Opus 4.8
- **关键超参**：Worker数量W=3（默认），每epoch每Worker本地trial数=3；reasoning effort=high
- **评估协议**：external cumulative-token counter控制预算，三个独立run的mean±std报告（仅Anthropic内核任务）
