---
title: "SELFSEARCH-REWARD-FREE-SEARCH-FOR-SELF-IMPROVING-AGENTS"
source: https://arxiv.org/pdf/2609.37968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:20:18"
field: "Agent 自我改进与自动化搜索"
keywords: ["self-improving agents", "reward-free search", "LLM agent", "automated code generation", "experience sharing"]
innovations: ["提出无需下游评估的 reward-free 自我改进搜索范式，利用 episode records 作为经验载体", "双 lineage（capability/adaptive）并行搜索并通过跨 lineage 共享历史记录实现互补进化", "自改进过程中涌现的调试工具（文本搜索、轨迹读取）可迁移复用至下游任务"]
benchmarks: ["SWE-bench Verified", "SWE-bench Multilingual", "Terminal-Bench 2.1"]
---

# 论文速读：SELFSEARCH

## 一句话总结
SelfSearch 提出了一种**无需下游评估奖励**的 Agent 自我改进搜索方法：Agent 通过读取并学习先前自我改进 episode 的记录（包含推理轨迹、工具调用与代码变更），在完全不接触下游任务评测的情况下迭代修改自身实现。在 SWE-bench、Terminal-Bench 等基准上，该方法实现了任务成功率与执行效率的双重提升，搜索成本仅需 $4.03 即可达到与 Codex 匹敌的 Terminal-Bench 2.1 表现。

## 研究问题与动机
1. **下游评估成本高昂**：现有 Agent 自我改进方法依赖反复的下游任务评测来筛选候选实现，随评测任务量增长成本线性攀升。
2. **搜索与任务耦合**：改进信号绑定于特定任务分布，难以获得通用化的能力进化。
3. **自我改进过程本身的信息未被利用**：每一次修改都会产生包含推理、工具调用、失败分析的轨迹记录，这些"元经验"尚未被系统地用于指导后续改进。
4. **如何在脱离下游评估的条件下实现有效搜索**：假设自我改进过程中习得的调试、代码审查、失败诊断等能力可迁移至下游任务。

## 核心贡献（创新点）
1. **引入 reward-free 搜索范式**：Agent 仅依靠历史 episode 记录和本地工具反馈进行自我修改，不依赖任何下游基准评测信号——与 DGM/HGM 等需要反复运行开发集的评估引导搜索形成本质区别。
2. **Episode Records 作为经验载体**：首次系统性地将"自我改进过程的轨迹记录"结构化保存并在代际间共享，使后续 agent 可复用前期发现的工具缺陷与修复模式。
3. **双 lineage 并行搜索架构**：维护 capability（能力扩展）与 adaptive（自适应修复）两条搜索路径，并通过跨 lineage 共享 episode records 实现互补发现。
4. **发现的工具可在下游任务中复用**：通过自改进发现的文本搜索、行范围文件查看等工具，在 SWE-bench 下游任务中使用率高达 8.6%–40.3%，证明自我改进经验可迁移。
5. **极低成本达到 SOTA 级性能**：仅 $4.03 搜索成本即产出 Terminal-Bench 2.1 上 82.0% 成功率，与九 harness 对比中的最高分 Codex 持平。

## 方法详解
- **Agent 仓库模型**：Agent 全部实现（指令、工具、执行逻辑、策略库）存于可编辑仓库中，模型权重固定不变；同一 agent 同时负责任务求解与自我修改。
- **Episode 记录结构**：每次 episode 产生记录 $e_k$，包含交互轨迹（推理链、工具调用、执行结果）、代码变更 diff、以及检查验证结果；记录以只读方式挂载供后续 episode 访问。
- **搜索流程**（Algorithm 1）：
  1. 初始化 agent $B_0$，在两条方向上各运行一次生成 seed episode records $\mathcal{E}_0$（保留记录，丢弃生成的实现，确保双 lineage 从同一起点出发）。
  2. 每代 $k$：两条 lineage 并行运行 `SelfImprove`，分别接收 $\mathcal{E}_k$ 和对应搜索方向 $d_w$，输出新一代 agent $B_{k+1}^w$ 与 episode 记录 $e_k^w$。
  3. 聚合所有新记录为 $\mathcal{E}_{k+1}$，加入候选归档 $\mathcal{C}$，进入下一代。
- **双方向搜索提示**：
  - **Capability 方向**：识别任务求解中的局限性，开发可复用的新工具或程序。
  - **Adaptive 方向**：改进 action 失败、证据与假设冲突时的策略调整能力。
- **运行时隔离**：模型–工具交互循环运行于不可编辑的固定 runtime 中，agent 通过 `respond(message)` 接口调用，确保推理设置与资源限制不被 agent 篡改。
- **关键公式**：成本降低率 $R(B) = 100(1 - \bar{c}(B)/\bar{c}(B_0))$，shared-success 成本降低率 $R_{\text{shared}}(B) = 100(1 - \sum_{i \in S_B} c_i(B) / \sum_{i \in S_B} c_i(B_0))$。

## 实验与结果
- **数据集**：SWE-bench Verified（120 任务）、SWE-bench Multilingual（60 任务，8 种语言）、Terminal-Bench 2.1（89 任务）。
- **模型配置**：GPT 配置（gpt-5.6-sol 搜索 + gpt-5.6-luna 执行）；DeepSeek 配置（deepseek-v4-pro 搜索 + deepseek-v4-flash 执行），均 medium reasoning effort。
- **主要结果**：
  - **所有六组 model–benchmark 组合下，population-mean success 均超过初始 agent**。
  - **Terminal-Bench 2.1**：DeepSeek $B^c$ 从 65.2% → **73.0%**（+7.8pp）；GPT $B^c$ 从 43.8% → **55.1%**（+11.3pp，为最大提升）。
  - **SWE-bench Multilingual**：DeepSeek $B^a$ 提升 **+5.0pp**，且 shared-success 成本降低 **38.5%**。
  - **SWE-bench Verified**：DeepSeek 双 lineage 均达 **86.7%**（+5.0pp）。
- **与评估引导基线对比**：SelfSearch 在 110 任务（排除 10 开发集）上与 linear search 和 archive search 持平或超越，搜索成本降低 **13.4%–53.2%**。
- **跨模型迁移**：GPT 搜索所得 agent 在 DeepSeek V4 Flash 上达 83.3%/85.0%，超越初始 81.7%；反之亦然。
- **Harness 对比**：$B^c$（DeepSeek）以 $4.03 搜索成本在 Terminal-Bench 2.1 上达成 **82.0%**，与 Codex 并列第一（9 harness 对比）。
- **消融**：移除 episode records 导致 mean success 下降 2.1–2.9pp；固定 improver（始终用 $B_0$）导致下降 1.7–2.9pp，两者均贡献显著。

## 相关工作脉络
1. **DGM (Zhang et al., 2026a)**：使用独立诊断过程提出修改，依赖下游评测引导；SelfSearch 将诊断与修改统一于同一 agent，且无需下游评测。
2. **HGM (Wang et al., 2026)**：分离 meta-agent 负责改进决策；SelfSearch 中 task agent 与 meta-agent 角色统一于单一可编辑实现。
3. **Hyperagents (Zhang et al., 2026b)**：在可编辑程序中显式定义 task agent 与 meta-agent；SelfSearch 不设角色隔离，任一代码均可被修改。
4. **SICA (Robeyns et al., 2025)**：支持 self-modification 但仍需评估反馈；SelfSearch 首次实现纯 reward-free 搜索。
5. **Group-Evolving Agents (Weng et al., 2026)**：通过跨 agent 经验共享指导进化，但经验来自下游任务执行；SelfSearch 的经验来自自我改进过程本身。
6. **Meta-Harness (Lee et al., 2026)**：端到端优化 harness 但依赖 benchmark 梯度；SelfSearch 的搜索信号完全来自本地工具反馈与历史记录。

## 局限性与未来方向
1. **episode records 截断风险**：长轨迹受输出 limit 约束可能被裁剪，导致关键上下文丢失（论文第 4.2 节提及此问题并催生了 structured trajectory reader）。
2. **搜索代数固定为 10**：未探索更深度演化的上限与收益递减拐点。
3. **跨模型迁移幅度不均**：部分配置下迁移增益较小（如 GPT luna 上 DeepSeek 搜索 agent 仅 +1.7pp）。
4. **双 lineage 方向的划分依赖人工 prompt 设计**，search direction 对最终效果的影响未充分消融。
5. **论文自述**：需在多次独立搜索中验证收益的稳定性，以及 SelfSearch 与评估引导搜索结合的效率潜力。

## 研究启发与可借鉴点
1. **无奖赏搜索的设计思路**：将"过程记录"作为替代外部 reward 的信号源，可迁移至强化学习、神经架构搜索等领域中评估代价高昂的场景。
2. **双策略并行 + 经验共享架构**：capability/adaptive 双 lineage 设计为多目标优化提供了轻量实现范式，可推广至并行贝叶斯优化或多臂老虎机。
3. **运行时与代码库的显式分离**：将 inference loop 固定于不可编辑 sandbox 外，agent 仅能修改工具与指令层，为安全可控的 self-modification 提供了工程模板。
4. **工具自发现机制**：agent 因"难以阅读长 episode record"这一具体痛点自动生成了 text-search 与 trajectory-reader 工具，证明**问题驱动的涌现工具设计**比预置工具集更具适应性。
5. **跨 lineage 经验传递**：A 路径发现的工具被 B 路径复用并改进（如 generation 7 交叉引用），为多 agent 协作学习中的知识蒸馏提供了新的记录格式。

## 关键术语表
**SelfSearch**：一种无需下游评估奖励的 Agent 自我改进搜索方法，通过历史 episode 记录指导迭代修改。
**Episode Record**：记录单次自我改进过程的产物，包含推理轨迹、工具调用序列、代码变更 diff 与验证结果。
**Reward-Free Search**：在搜索阶段不使用任何下游任务评测分数作为反馈信号的优化过程。
**Capability Lineage**：搜索方向之一，聚焦于识别能力局限并开发可复用工具/程序。
**Adaptive Lineage**：搜索方向之一，聚焦于改进 action 失败时的策略调整与恢复机制。
**Population-Mean Success**：两条 lineage 最终 agent 在下游任务上成功率的平均值。
**Harness**：指打包好的 Agent 实现（含指令、工具、执行逻辑），可被不同模型加载执行。
**Shared-Success Cost Reduction**：在初始与演化 agent 均解出的任务子集上比较的平均执行成本降幅。

## 可复现要素
- **数据集**：SWE-bench Verified、SWE-bench Multilingual（seed 42 采样的 60 任务列表见 Table 7）、Terminal-Bench 2.1；论文未声明公开，但均为公开 benchmark。
- **代码/权重**：执行框架与 agent 实现细节见 Appendix A–C，论文未提供 GitHub 链接或模型权重；初始 agent 基于 DGM 实现（Zhang et al., 2026a）修改。
- **关键超参**：搜索代数 $K=10$，lineage 数 $W=2$，每 episode 最多 513 次模型调用 / 512 次工具步骤 / 4 小时执行；GPT 单 call 输出 limit 8,192 tokens，DeepSeek 16,384 tokens；temperature=1，medium reasoning effort。
- **搜索成本**：GPT 配置 $6.52，DeepSeek 配置 $4.03。
