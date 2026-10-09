---
title: "VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO"
source: https://arxiv.org/pdf/2610.10183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:26:32"
field: "长视频理解"
keywords: ["long video understanding", "memory-based reasoning", "agentic reinforcement learning", "self-evolving agents", "multimodal large language models", "memory-retrieval co-evolution"]
innovations: ["交替代理强化学习实现记忆构建与检索策略的协同进化", "瓶颈感知进化反馈（BEF）动态分配记忆与检索的训练资源", "能力感知进化反馈（CEF）基于九类视频理解能力重加权训练课程"]
benchmarks: ["Video-MME", "LongVideoBench", "LVBench", "MLVU", "MMVU"]
---

# 论文速读：VIDEOEVOLVE: CO-EVOLVING MEMORY AND RETRIEVAL FOR LONG VIDEO UNDERSTANDING

## 一句话总结
VideoEvolve 提出了一种自进化框架，通过交替代理强化学习（Agentic RL）协同进化长视频理解中的**记忆构建**与**检索策略**，使系统能从下游推理反馈中学习"记住什么"和"如何检索"。实验在五个基准上均取得开放源码方法中的最佳性能。

## 研究问题与动机
- **记忆适应性有限**：现有基于记忆的方法多采用预定义的记忆构建策略，不利用下游推理反馈优化记忆内容，导致关键证据被遗漏后需重复消耗算力回溯原始视频。
- **记忆-检索不对齐**：记忆构建与检索通常分别优化，未显式对齐"记住什么"与"如何检索"之间的关系，造成存储信息有价值但难以可靠检索。
- **单向适应范式缺陷**：现有方法遵循单向适应（Figure 1(a)），记忆构建完成后不再根据检索效果调整，而本文目标是实现两者的联合进化。

## 核心贡献（创新点）
1. **交替代理强化学习的协同进化框架**：通过交替优化 Memory Evolver 和 Retrieval Evolver（冻结一侧训练另一侧），使记忆构建与检索策略互相指导改进；与现有方法仅优化单侧或固定记忆相比，实现了"记住什么"与"如何检索"的动态对齐。
2. **瓶颈感知进化反馈（BEF）**：通过对比 Memory-only 与 Revisit-enabled 条件下的推理表现，诊断当前瓶颈在记忆还是检索侧，动态调整两侧的训练分配比例；现有方法使用固定训练配比，无法自适应跟踪瓶颈变化。
3. **能力感知进化反馈（CEF）**：基于九类视频理解能力（如动作识别、时序推理、复杂情节理解等）的不足程度重加权训练样本，引导进化 toward 待发展 yet 可学习的能力；避免在固定问题集上过拟合，提升泛化性。

## 方法详解

**整体架构**：VideoEvolve 包含四个核心组件：Memory Evolver（学习"记住什么"）、Retrieval Evolver（学习"如何检索"）、交替 Agentic RL（协同进化机制）、BEF 与 CEF（进化引导反馈）。

**Memory Evolver**：
- 构建固定分层 Base Memory（Root–Super–Macro 三级结构，0.5 fps 采样）
- 通过 Observe–Decide–Augment 流程选择性增强记忆：决定观测哪个 Macro、时间区域、帧数（{2,4,8,16}）、信息焦点（action/state, appearance/spatial, text alignment 等）
- Delta 记录由冻结的 Writer 生成，与 Base 合并形成演化后记忆 $M_v^k = \text{Merge}(B_v, \Delta_v^k)$
- 每次更新后从同一 Base 重新构建记忆，而非累积历史 Delta

**Retrieval Evolver**：
- 支持七种检索工具：get_macro_events, get_subgraph, search_nodes, search_by_time, get_keyframes, read_document, view_video
- 通过 Search–Inspect & Revisit–Reason & Answer 流程自适应获取信息
- 初始化采用两阶段 SFT（Stage I: 4,317 条 LongVT 工具使用轨迹；Stage II: 102,732 条 LLaVA 视频轨迹）

**交替 Agentic RL**：
- 第 k 轮：冻结 $R^k$，Memory Evolver 采样多条构建轨迹，由冻结的 $R^k$ 评估候选记忆
- 更新 $C^k \to C^{k+1}$，重建记忆 $M^{k+1}$
- 冻结 $M^{k+1}$，优化 Retrieval Evolver 得 $R^{k+1}$
- 记忆侧奖励：$R_C = R_{\text{rescue}} - \lambda_{\text{reg}} P_{\text{regress}} - \lambda_{\text{inv}} P_{\text{invalid}}$
- 检索侧奖励：$R_R = R_{\text{ans}} - \lambda_{\text{inv}} P_{\text{invalid}}$
- 质量优先组相对优化：任务奖励归一化后，资源效率仅作为同质量轨迹的 tie-breaker

**BEF（瓶颈感知进化反馈）**：
- 在 Probe Questions 上对比 Memory-only 与 Revisit-enabled 条件下的回答正确性
- 计算记忆侧需求 $d_C$ 与检索侧需求 $d_R$，动态调整下一轮训练分配比例 $s^{k+1}$

**CEF（能力感知进化反馈）**：
- 基于九类能力的诊断信号更新平滑需求 $e_{p,c}^k$
- 目标分布：$\tilde{w}_{p,c}^{k+1} = \frac{\rho}{K} + (1-\rho)\frac{e_{p,c}^k}{\sum e_{p,c'}^k}$
- 采样策略混合 frontier（60%）、exploration（20%）、retention（20%）样本

## 实验与结果

**数据集与基准**：
- Video-MME（含 Long 子集）、LongVideoBench、LVBench、MLVU、MMVU
- 训练数据：514 视频（2,056 问题）用于 Agentic RL，128 视频（512 问题）用于 BEF/CEF 诊断

**主要结果**（Table 1 & 2）：
- **VideoEvolve-8B** 在所有五个基准的六项指标上均排名第一（开源方法）：
  - Video-MME（w/ sub）: **73.9%**（vs 次优 ParaVT-8B 69.4%，+4.5pp）
  - LongVideoBench: **70.2%**（vs ParaVT-8B 60.4%，+9.8pp）
  - LVBench: **58.9%**（vs VideoZoomer-7B 41.5%，+17.4pp）
  - MLVU: **72.4%**
  - MMVU: **75.1%**
- 相比工具增强版 Qwen3-VL-8B baseline 提升 **14.7–28.6 个百分点**
- **训练自由版本**（Qwen3.8-27B）在 LVBench 达 **76.7%**，超越 MERIT-GPT 4.9pp

**消融结果**（Table 3）：
- 协同进化 vs 单侧进化：全模型 58.9% vs 仅记忆进化 51.1% vs 仅检索进化 52.7%（LVBench）
- 移除交替更新：LongVideoBench 下降 4.0pp
- BEF+CEF 组合将 MMVU 从 68.4% 提升至 75.1%
- 视频回溯贡献 0.8–1.6pp 提升

## 相关工作脉络

1. **视频检索方法**（VideoAgent, DVD, LVAgent）：关注"何时何地查看视频"，而本文聚焦"持久记住什么以供后续检索"，二者互补。
2. **记忆基方法**（MovieChat, MA-LMM, EgoRAG, HippoMM, WorldMM, MERIT, MemVid, M3-Agent）：现有方法多为预定义构建策略或分别优化记忆与检索，本文首次实现记忆构建与检索的交替协同进化。
3. **Agentic RL 与自进化 Agent**（VideoZoomer, VITAL, LongVT, Ego-R1, EvolveR, SkillRL, Agent0, Evolving-RL）：现有工作主要改进推理策略/技能库，本文将其应用于记忆-检索这对基础轴线的联合进化。
4. **长视频理解 MLLMs**（TimeAware, LongContext Scaling, Adaptive Compression）：本文方法可与这些视觉压缩技术结合，形成更完整的长视频理解系统。

## 局限性与未来方向

**局限性**：
1. 评估仅限离线多选题理解，未验证流式视频、开放式对话或直接音频理解任务
2. 依赖冻结 Writer 的感知质量与有限观测预算，可能遗漏短暂事件或细微视觉细节
3. 策略优化依赖 ground-truth 答案，BEF/CEF 使用固定诊断池与预定义能力分类，覆盖范围受限于可用数据
4. 交替训练需候选轨迹采样、冻结策略评估、反复重建记忆与索引，推理时多步检索与回溯仍有计算开销

**未来方向**：
1. 增量记忆更新以适应流式输入，结合 richer 视听证据与开放式评估
2. 证据级验证与校准不确定性，识别不可靠记忆记录并判断何时需额外观测或 abstain
3. 结合弱监督与自适应能力发现，减少对固定标注池的依赖
4. 更高效记忆/索引更新、策略蒸馏与成本感知执行

## 研究启发与可借鉴点

1. **交替优化范式可迁移**：将记忆构建与使用分离为可交替优化的两个策略，通过下游反馈实现协同进化，可推广至其他需要"存储-检索"配合的任务（如长文档理解、知识图谱问答）。
2. **瓶颈诊断机制设计**：BEF 通过配对条件对比（有/无额外访问）诊断瓶颈所在，这种"条件对比+需求估算"的方法可复用于其他双模块系统的资源分配。
3. **能力感知的课程学习**：CEF 将能力标签转化为动态采样权重，结合 frontier/exploration/retention 三向混合采样，为多能力学习任务提供了可借鉴的课程设计模板。
4. **质量优先的资源 tie-breaker**：将资源效率仅作为同质量轨迹的 tie-breaker 而非直接惩罚，避免了性能-效率的零和权衡，适合部署在资源受限场景。
5. **训练自由变体的实用性**：仅使用预训练模型（Qwen3.8-27B）配合固定 Base Memory 即可取得强基线，为快速部署提供了低成本选项。

## 关键术语表

**Memory Evolver**：学习选择性记忆增强的策略网络，通过观察-决策-增强流程决定从原始视频中提取哪些信息存入 Delta 记忆。

**Retrieval Evolver**：学习自适应检索的策略网络，通过工具调用序列从演化记忆中搜索证据并回答查询。

**Agentic RL**：代理强化学习，将多轮交互与工具使用耦合到策略优化中，使模型能从下游推理结果获得学习信号。

**Bottleneck-Aware Evolution Feedback (BEF)**：通过对比 Memory-only 与 Revisit-enabled 条件的性能差距，诊断当前系统瓶颈在记忆构建还是检索侧，动态调整训练分配。

**Capability-Aware Evolution Feedback (CEF)**：基于九类视频理解能力（如时序推理、复杂情节理解等）的诊断信号，重加权训练样本以优先发展不足 yet 可学习的能力。

**Base Memory**：从低帧率视频观测构建的固定分层记忆（Root-Super-Macro 三级），在进化过程中保持恒定，仅通过 Delta 补充增强。

**Delta Memory**：由 Memory Evolver 选择性增强生成的补充记忆记录，与 Base Memory 合并形成最终可检索记忆。

**Quality-First Group-Relative Optimization**：先对任务奖励进行组内归一化，再将资源效率仅作为同质量轨迹的 tie-breaker 的优化目标设计。

## 可复现要素

- **数据集**：Video-MME-v2（514 视频训练 + 128 视频诊断）、LongVT-derived 轨迹、LLaVA 视频轨迹；评估基准 Video-MME、LongVideoBench、LVBench、MLVU、MMVU
- **代码/权重**：论文未明确声明开源，使用 verl 框架训练
- **关键超参**：
  - 帧预算 $b_F(T)$: 32/64/96/128（依视频时长）
  - 文本预算 $b_T = 4,096$ tokens
  - 探索槽位 $B_N = 12$，每问题新帧 $B_F^R = 64$
  - 回归惩罚 $\lambda_{\text{reg}} = 1$，无效终止惩罚 $\lambda_{\text{inv}} = 0.2$
  - 效率系数 $\beta = 0.05$，KL 正则 $\beta_{\text{KL}} = 0.02$
  - BEF 初始份额 $s^0 = 0.5$，边界 $s_{\min}=0.25, s_{\max}=0.75$
  - CEF EMA 系数 $\alpha = 0.5$，均匀覆盖权重 $\rho = 0.3$
  - 学习率：Memory 2e-6，Retrieval 1e-6
