---
title: "Where-is-Physical-RSI-capability-from-An-Observational-Scali"
source: https://arxiv.org/pdf/2610.26848v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:22:35"
field: "具身智能与 LLM Agent 评估"
keywords: ["physical recursive self-improvement", "LLM-as-policy", "embodied agent", "scaling laws", "VLM evaluation", "robot manipulation", "observational scaling"]
innovations: ["提出五轴指标体系分离首次尝试表现、改进效率、最终得分、预算节省和失败恢复，避免单一终点分掩盖改进过程差异", "提出物理木桶假设并通过六个 VLM 的实证分析揭示空间推理、感知与编程对改进不同维度的差异化约束", "建立观测缩放规律：用公共 benchmark 能力 rank 预测跨模型物理改进排序，用早期轨迹拟合 Weibull 模板实现少轮次预测多轮次"]
benchmarks: ["PhysRSI-eval", "RoboDojo"]
---

# 论文速读：Where-is-Physical-RSI-capability-from-An-Observational-Scali

## 一句话总结
论文系统研究了大模型作为策略在物理操作任务中的递归自我改进（Physical RSI）能力来源、缩放规律与泛化性，提出 PhysRSI-eval 基准和五轴评估指标，并基于六个前沿 VLM 的实证结果提出"物理木桶假设"。

## 研究问题与动机
- **如何量化物理 RSI？** 单一终点分数无法刻画改进何时发生、消耗多少物理交互成本；重复尝试可能只是随机增加成功率，而非有效修订。
- **Physical RSI 能力的来源是什么？** 空间推理、感知、编程和工具使用等不同能力是否以不同方式支撑"首次尝试表现"和"从失败中恢复"？
- **改进是过拟合还是泛化？** 智能体积累的经验是否仅适配特定场景，还是能迁移到 unseen layouts 甚至其他任务？
- **能否用公开能力分数预测物理改进？** 在不运行完整评估的情况下，仅凭公共基准排名能否预判模型的改进幅度和时机？

## 核心贡献（创新点）
- **提出 PhysRSI-eval 五轴评估体系。** 将首次尝试表现（A1）、效率（A2）、成功率（A3）、预算节省（A4）和从失败恢复（A5）分离度量，使改进过程可量化而非仅看终点。
- **提出物理木桶假设（Physical Barrel Hypothesis）。** 不同能力约束不同改进维度：空间推理主导起始表现和提前改进，感知主导最终得分和失败恢复，而编程能力与物理改进相关性最弱。
- **建立观测缩放规律。** 通过公共基准分数的归一化排名可预测跨模型的物理改进相对顺序（包括 T₅₀  gain timing），并用早期物理曲线（Weibull 模板）实现少轮次下对未来轨迹的预测（MAE=0.019 vs 基线 0.112）。
- **实证区分场景适配与经验复用。** 清除历史后晚期成功率从 41.7% 降至 8.3%（Astra 六任务消融）；冻结 workspace 在 unseen layouts 上仍达 41.7% 成功率；跨任务 workspace 初始化使首次成功提前（如 Cover Blocks K9→K1），证明两者共存。

## 方法详解
**实验协议（双阶段固定预算）：**
- 模型权重完全冻结，允许修改 skills/tools/code/notes 等 workspace 内容。
- K0（基线）：每任务独立 3 次运行取均值，无 workspace。
- K1–K17（改进阶段）：每轮一次物理尝试，返回 native score、success flag、failure reason，然后关闭机器人动作进行最多 1800s（GPT）/600s（Claude）的 workspace 修订，直到首次 native success 或耗尽 17 轮。

**五轴指标定义：**
- **A1 = q₀**：K0 平均任务得分（含部分进度）。
- **A2 = 收益时序比**：`[(∑_{k=1}^{H-1} q_k)/(H-1) − q₀] / (q_H − q₀)`，值越大说明改进越早。
- **A3 = 高窗平均分**：找到达到 90% 峰值增益的轮次 τ₉₀，取 τ₉₀ 到 H 的均值（含部分分）。
- **A4 = 预算节省**：对已解决任务 t，计算 `U_t = (W_t + Z_t)/L_t + e_t`（使用步数占比 + 额外尝试），`A₄ = mean[max(0, 1−U_t/H)]`。
- **A5 = 失败恢复率**：`(C_H − C_1)/(1 − C_1)`，K1 失败的任务在预算内被恢复的比例。

**能力关联分析：** 将 17 个公共 benchmark 分数归一化后按 Spatial / Harness / Perception / Coding 四组聚合，以 Spearman 相关考察各组与 A1–A5 的关系。

**预测方法：**
- 跨模型 rank 预测：kernel ridge regression（γ=0.0625，α=10），输入四组能力归一化 rank，预测 T₅₀ 先后顺序。
- 早期轨迹预测：以 Astra K1–K17 拟合 Weibull 模板 g(t)=1−exp[−(t/τ)^p]，再用 Sol 前 Kᵢ 轮校准起始 u、平台 v 和相对速度 ρ，正则化 toward template，预测剩余轮次。

## 实验与结果
**数据集与系统：**
- 18 个 RoboDojo 双臂操作任务，按官方成功率分 6 个难度层（D1–D6）各抽 3 题。
- 6 个系统：GPT-6 Astra、GPT-6.1 Sol、GPT-6 Sol、GPT-5.6 Sol、Claude Opus 5.5、Claude Sonnet 5.5；Gemini 4 在 10 题子集上单独报告。

**主要数值结果（Table 1）：**
| 系统 | Tasks | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|---|
| GPT-6 Astra | 18 | 0.368 | 0.859 | 0.851 | 0.843 | 0.750 |
| GPT-6.1 Sol | 18 | 0.284 | 0.822 | 0.703 | 0.808 | 0.571 |
| Claude Opus 5.5 | 18 | 0.177 | 0.609 | 0.744 | 0.579 | 0.688 |

- Astra 在全部五轴均排名第一，共解决 15/18 任务。
- Opus 的 A3（0.744）高于 GPT-6.1 Sol（0.703），但 A2（0.609 vs 0.822）和 A4（0.579 vs 0.808）明显落后——说明"高分≠早改进≠省预算"。
- GPT-5.6 Sol 累计 16 轮几乎无进展（A3=0.104, A5=0.056）。

**能力关联（Figure 4）：**
- Spatial ↔ A1（ρ=0.94）、A2/A4（ρ=0.83）；Perception ↔ A3/A5（ρ=0.94）；Coding 与所有轴相关性最弱（ρ≤0.37）。

**预测实验：**
- 四组能力 rank 之和的排序完全恢复 6 个系统在 q̄₁₇ 上的观测顺序。
- T₅₀  pairwise 预测全部正确恢复。
- Astra 模板迁移到 GPT-6.1 Sol：用 K1–K5 校准后预测 K6–K17，MAE=0.019，显著优于保持最后一轮的基线（MAE=0.112）。

**泛化实验（Astra 六任务）：**
- No-history 消融：K4–K6 成功率 41.7% → 8.3%。
- 冻结 workspace 在 unseen layouts：41.7%（25/60）；含 raw records 反而 38.3%。
- 跨任务初始化：Sonnet Cover Blocks 首次成功 K9→K1，Opus Organize Table K14→K7，Opus Deposit Coin K15→K8。

**工具库分析（Figure 10）：**
- Opus 工具复用率 0.347，Sonnet 0.167，GPT-6 Sol 0.086；Opus 复用更分散（Top10% 占 48% vs Sonnet 76% vs GPT-6 Sol 83%）。
- Astra 几乎不建工具（99%+ 动作由模型直接生成），提示其内在感知/空间能力更强。

## 相关工作脉络
- **LLM-as-policy 评估基准（如 LIBERO-Agent、CodeActionBench、Robot-World）：** 主要报告单 episode 结果或时间预算内的通过率，不追踪 persistent workspace 下的多轮修订轨迹；PhysRSI-eval 强调"从首次尝试到改进"的动态过程与物理成本。
- **Agent 自我改进基准（如 Evo-Bench、S3Gym、Aspire、EvoHarnessBench）：** 聚焦 harness 代码/策略的自我演化，但缺少真实物理交互循环和 native 评分；本文引入带打分机器人接口的闭环协议。
- **Self-Refine 与 Reflexion：** 文本/代码领域的工作，依赖语言反馈；本文区分"producing an update"和"benefiting from it"，强调物理试错和 workspace 保留的独特价值。
- **Observational Scaling Laws（Ruan et al.，2024）：** 首次将公共 benchmark 能力分数与物理改进排序建立可预测关系；本文将此思路扩展到 embodied physical RSI 场景。
- **能力-表现关联（BLINK、SpatialVLM、EmbSpatial-Bench）：** 证实感知/空间推理对具身任务的重要性，但缺乏对"改进时序"维度的分析；本文揭示不同能力分别制约改进的不同方面。

## 局限性与未来方向
- 仅覆盖 6 个模型+配置，每个任务仅一条改进链，K0 重复无法估计完整链的方差。
- 能力关联分析依赖公共 benchmark 分数（评测设置与物理任务不同），且仅 6 个样本、Spearman 检验的最低 adjusted q=0.11，未能达到常规显著性阈值，瓶颈结论仍为假设。
- 消融和冻结实验仅针对 Astra 及 6 个任务；跨任务迁移仅报告 3 个典型案例。
- First-success stopping 只记录一次成功，无法反映可靠重复执行能力。
- Budget 和 Improvement 使用条件任务集（因各系统已解决问题数不同），跨系统直接比较需谨慎。
- 未来方向：扩大模型数量、独立验证能力瓶颈假设、探索哪些具体 workspace artifact（工具/记忆/笔记）支持跨任务迁移、将观测缩放升级为因果干预实验。

## 研究启发与可借鉴点
- **五轴度量框架可直接迁移：** 任何需要评估"迭代改进过程"（而不仅是终点分数）的 benchmark，均可采用 A1–A5 分离度量的思路，避免单一指标掩盖 trade-off。
- **观测缩放方法论可复用：** 用公共能力 rank 预测新系统相对排名，以及用参考模型的早期曲线迁移预测新模型后期轨迹，均可推广到其他 agent 评测场景。
- **Workspace 干预设计值得借鉴：** "清除历史 vs 冻结 workspace vs 跨任务初始化"的三层次泛化检验协议，为区分过拟合与可复用经验提供了清晰范式。
- **工具复用度量（reuse rate、lineage breadth）可迁移到代码 agent 研究：** 本文的工具 lineage 追踪方法可用于分析 LLM coding agent 的技能积累模式。
- **物理木桶假设的启示：** 团队在构建具身 agent 时，不应只优化单一能力（如编程），而需识别目标任务的具体能力瓶颈（空间/感知/工具）进行针对性增强。

## 关键术语表
- **Physical RSI（Physical Recursive Self-Improvement）：** 指大模型在物理机器人操作中通过反复试错、修订 workspace（代码/工具/笔记）实现策略改进的过程，区别于纯文本/代码领域的 self-improvement。
- **PhysRSI-eval：** 本文提出的基准测试，包含 18 个 RoboDojo 双机械臂操作任务和固定的双阶段评估协议。
- **五轴指标（A1–A5）：** 首次尝试表现（A1）、改进效率/时序（A2）、最终成功率（A3）、物理预算节省（A4）、从失败恢复率（A5）。
- **Physical Barrel Hypothesis（物理木桶假设）：** 类比木桶短板原理，指不同能力（空间推理、感知、编程、工具使用）分别限制改进的不同方面；起始能力强与从失败恢复能力强所需的核心能力不同。
- **T₅₀：** 半增益轮次，即最佳平均得分首次达到 K1–K17 总增益一半时的轮次，用于量化改进时机。
- **Observational Scaling Law：** 通过观察已有系统的公开能力分数与物理改进表现之间的统计关系，预测未见系统的物理改进排名和时序，类似传统 scaling law 但在无训练算力数据下的观测版本。
- **Workspace 干预（Workspace Interventions）：** 包括清除历史（no-history）、冻结 workspace 在未见布局上测试、跨任务 workspace 初始化，用于区分场景适配与可复用经验。
- **Tool Lineage Reuse：** 工具谱系复用，追踪 later tools 是否由 earlier tools 直接复用、修改或衍生而来，用于量化 agent 工具库的积累模式。

## 可复现要素
- **数据集：** 18 个 RoboDojo 双臂操作任务（来自 RoboDojo benchmark），布局固定（layout 0, seed 0）。论文未明确说明是否开源，项目页面为 https://physrsi-eval.github.io/。
- **代码/权重：** 论文未提供开源代码或权重。
- **关键超参：** 每任务最多 H=17 轮；每轮物理步数上限 300–1300 步（25Hz）；推理预算 4200s/轮；修订窗口 GPT 1800s / Claude 600s；Kernel Ridge 参数 γ=0.0625，α=10；Weibull 模板 τ_A=2.18，p_A=1.30；正则化 λ=4。
