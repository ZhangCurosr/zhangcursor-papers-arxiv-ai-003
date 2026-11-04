---
title: "ToolBottleneckBench-Measuring-Tool-Agent-Performance-under-C"
source: https://arxiv.org/pdf/2610.21098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:33:19"
field: "LLM Agent Evaluation"
keywords: ["tool agent", "benchmark", "harness effect", "failure diagnosis", "controlled experiment", "function calling"]
innovations: ["提出在固定 harness 预算下逐个施加 R/V/W/D 四种压力的匹配变体基准，实现压力效应的可归因测量", "设计 failure record（首个未满足义务+终止标志）二元组，分离失败标签与终止原因", "提出修复激活检验与跨配置切换的帕累托对照范式，证明仅 router 与 controller 两项修复胜过所有配置切换"]
benchmarks: ["ToolBottleneckBench", "ToolSandbox", "Harness-Bench", "ToolMenuBench"]
---

# 论文速读：ToolBottleneckBench-Measuring-Tool-Agent-Performance-under-C

## 一句话总结
本文提出了 ToolBottleneckBench，一个在固定 harness 预算下测量 tool-agent 性能的控制基准：固定任务族、逐个施加四种压力（工具目录/可见菜单/认证工作流/返回记录编码），并精确记录每个失败 episode 的终止原因，从而区分模型缺陷与 harness 限制。

## 研究问题与动机
- 现有工具使用基准仅报告单一成功分数，无法区分失败源于模型本身还是运行 harness 的预算/接口限制（如 completion cap 导致的截断）。
- 真实环境中多项负载同时变化，而现有受控实验未记录失败 episode 由哪个 harness 限制终止，导致诊断标签（如"工具选择失败"）可能误导。
- 修复方法的有效性评估缺乏系统性对照：既无"跨压力激活"检验，也无与"更换配置"策略的直接比较。
- 现有基准（如 ToolMenuBench、Harness-Bench、ToolSandbox）在可归因性、精确计分、多压力交叉、修复协议等方面各有缺失。

## 核心贡献（创新点）
- **受控压力基准**：在每个固定任务族上通过确定性编译器逐一改变四种压力（R/V/W/D），生成匹配变体，使压力效应可归因；相比 prior work 同时改变多项负载，本文实现真正的单因素控制。
- **失败记录机制（failure record）**：精确 grader 记录首个未满足义务与终止标志（harness 限制类型）的配对，首次将"失败标签"与"终止原因"分离；这使读者可知失败是截断还是调用超限。
- **修复协议与配置切换对照**：提出激活度量（L, Γ, Σ, Δ）并联合 intersection-union test 检验，同时在同一 stress cells 上直接比较每条修复与所有 21 条配置切换，回答"修复 vs 换模型"的实践问题。
- **完成 cap 扫掠实验**：将 completion cap 提升 16 倍（1,024→16,384 tokens）以隔离截断效应，发现菜单惩罚的 41 pp 交互中绝大部分可归因于截断，揭示了 leaderboard 排名被 serving 预算部分决定的现象。
- **开源全套资源**：发布确定性生成器、精确 grader、全部 26,576 行评分数据与分析代码（MIT License），并附审计证书。

## 方法详解
- **任务族（36 families × 12 intents）**：固定请求、事实值、判定标签、证据引用规则、允许终态与 operational tool 身份；记录 ID  opaque 以防止答案复用；跨 4 类企业领域（会计/邮件/文件/日历）。
- **四种压力**：
  - **注册目录 R**：32 / 512 / 2,048 函数，用冻结检索器返回 16 条候选；评分为 ITT 成功 A·S（可用性×选择）。
  - **可见菜单 V**：8 / 32 / 128 个 schema；schema 字节从 3,481 增至 55,176；加入 route neighbor 干扰项（离线审计 13,158 次调用无返回记录）。
  - **认证工作流 W**：low(2 reads)/medium(4)/high(6 reads)，外加 frontier 链 2/6/14/30 calls；是复合压力（调用数、依赖深度、证据散布、累积上下文同步增长）。
  - **返回编码 D**：raw / compact / typed / linked 四种等价编码，字节数不同。
- **固定 harness 合同**：completion cap=1,024 tokens/请求；episode budget=128,000 tokens；call limit=certified minimum+2；turn limit=call limit+2；infeasibility stop 当 success 在剩余调用内不可达时提前终止。
- **VTS 九义务合取**：Y = A∧S∧P∧Q∧E∧X∧G∧B∧C，按序检查，任一失败即止。
- **失败记录**：（首个未满足义务，终止标志）二元组，终止优先级：截断 > 超预算/调用限额 > turn 限 > infeasibility > 错误结论 > 其他合约违反。
- **修复激活度量**：L = μ_{ℓ0}^0 − μ_{ℓ1}^0（压力损失）；Γ = μ_{ℓ1}^ℓ − μ_{ℓ1}^0（加压下修复收益）；Σ = Γ − (μ_{ℓ0}^ℓ − μ_{ℓ0}^0)（选择性）；Δ = Σ − max_k Σ_k（跨层优势）。三者均正且经 Holm 校正即判定激活。
- **统计设计**：配对 within-family 差（pp），10,000 次 cluster bootstrap 重采样 36 families；主区间与 Holm 多重校正；exact sign-flip test 用于意图级稳健性。

## 实验与结果
- **规模**：21 个配置（7 组织：Anthropic/OpenAI/Google/Alibaba/DeepSeek/Tencent/Z.ai），26,576 个评分条件，20,408 条精确 VTS 打分。
- **主结果**：
  - 平均 VTS = 82.1%。
  - **可见菜单 V**：8→128 导致 VTS 下降 −12.3 pp [−14.6, −9.9]（Holm p<.001），34/36 家族退化；全菜单可见时 −17.5 pp。
  - **工作流 W**：high−low = +0.8 pp [−1.2, 2.8]，无检测到的 pooled 效应；配置间异质大（Gemini 3.6 Flash −30.6 pp vs HY4 Preview +34.7 pp）。
  - **编码 D**：compact−raw = −3.7 pp [−5.7, −1.9]（Holm p=.003）；typed/linked 不显著。
- **截断主导菜单惩罚**：V=128 时截断率从 1.6% 升至 13.6%，391/411 次截断 episode 在第一调用前终止；70.2% 截断首响应含纯 reasoning tokens（主要来自 Gemini 3.8 Flash）。跨配置 Pearson r = −0.95。
- **16 倍 cap 扫掠**：6 个截断重度配置的菜单对比从 −43.7 pp 缩至 −2.7 pp，交互 41.0 pp [36.2, 45.5]，远超预设 5 pp 边际；残余惩罚仅 −2.7 pp。
- **失败终点分布**：3,725/20,408（18.3%）失败；91.5% 终止于 harness 限制；其中 50.0% 超 budget/call 限、28.7% 截断、12.8% infeasibility、仅 5.5% 错误结论。**所有 849 条"selection"标签 episode 均未执行任何调用**，838 条来自截断。
- **失败记录解码**：从 failure record 识别压力仅达 48.9% 平衡准确率，低于调用次数分箱（54.8%）；添加 trace shape 无增益。
- **修复激活**：router（L=79.6 pp, Γ=58.1 pp, Holm p=.001）、routed menu（Γ=4.9 pp, Holm p=.003）、public-handoff controller（Γ=96.6 pp, Holm p<.001）激活；两条 prompt 修复不激活（p=.949）。
- **修复 vs 切换**：router 对全部 21 个 incumbent 均为 success-superior；controller 对 17/17 incumbent success-superior、4/17 cost-superior；routed menu 仅 9/21 Pareto-nondominated。
- **外部环境迁移**：ToolSandbox 中扩大菜单主要增加 token 消耗（+16,916 tokens）而非降低成功（−2.9 pp，p=.224）；VTS 与 ToolSandbox 成功负相关（ρ=−0.8）。

## 相关工作脉络
- **Tool-use benchmarks**（API-Bank, ToolLLM, BFCL, AppWorld, τ-Bench, MCP-Bench）：多数同时改变目录/工作流/观察格式多项负载，分数差异无法归因到单因素；本文通过匹配变体实现单压力对照。
- **受控工具负载研究**（ToolMenuBench, ToolScope, APIFlow-Bench, LOCA-bench, LongFuncEval）：各自聚焦单一维度（菜单大小、长链、输出长度、位置效应）；本文统一框架交叉 R/V/W/D 四种压力，并记录终止标志。
- **Harness 效应研究**（Harness-Bench, HAL, Model or Harness?）：报告 per-model-harness 结果或方差分解；本文进一步量化各压力效应在特定 harness 限制下的共现比例。
- **Failure diagnosis & repair**（DoVer, REFLECT, DAAF, Tool-RobustBench, LayerRAG-Bench）：定位失败阶段或做 counterfactual 修正；本文的独特之处在于引入 "wrong-layer control" 与"修复 vs 配置切换"的直接帕累托比较。
- **工具检索/过滤**（ToolChoiceConfusion, X-MCP-Zero, RAG-MCP, ToolScope）：关注 retriever 设计；本文将 connector-first router 作为修复手段纳入统一评估框架，证明其跨配置 superiority。
- **上下文与截断效应**（Lost in the middle, Context length alone hurts）：理论分析长上下文损失；本文通过 16 倍 cap 扫掠提供实证区分截断与 menu-size 主效应。

## 局限性与未来方向
- **预算固定**：主结果在冻结 harness 预算下成立，仅 completion cap 被扫掠；episode budget 与 call limit 尚未作为独立变量考察。
- **单次实现**：核心容量单元每种配置仅一次运行（family bootstrap 吸收重复间变异，sentinel 重复显示 12.1% 分歧），统计功效受限。
- **合成任务与英语提示**：环境为合成且精确计分（无 partial credit），提示为固定英文文本；现实场景中部分 credit 与多语言分布可能改变压力排序。
- **配置耦合模型与 provider route**：部分 route 含 FP4 量化等特殊设置，结果描述的是"模型+route"对而非纯模型能力。
- **未来方向**：将 episode budget / call limit 作为可变压力、测试 live tool servers、扩展多语言与 partial-credit 场景、探索 public-information 决策策略（如 trace selector）的实际部署。

## 研究启发与可借鉴点
- **单压力控制设计**：确定性编译器生成匹配变体的思路可直接迁移到 RAG 检索精度、上下文工程、agent memory 管理等多因素研究中，实现"压力-效应"的可归因测量。
- **失败记录二元组**：（第一未满足义务，终止标志）的设计比单纯 stage label 更能指导工程修复；建议团队在现有 agent 评测中采用类似格式。
- **修复 vs 换配置的对照范式**：intersection-union 激活检验 + 跨 incumbent Pareto 比较是一套完整的 intervention evaluation pipeline，可复用于 prompt engineering、retriever 替换、serving adapter 选型等场景。
- **截断隔离实验**：通过提高 completion cap 来分离"截断导致"与"菜单过大导致"的范式，对任何受 token 预算限制的 agent benchmark 都有借鉴价值。
- **wrong-layer control**：用非目标层的修复作为阴性对照（如用 canonical delivery 对照 encoding 修复），防止假阳性激活判断，可推广到多干预 ablation 研究。

## 关键术语表
- **Verified Task Success (VTS)**：九项义务合取的成功率指标，任何一项失败即 episode 失败。
- **Assigned Pressure**：通过确定性编译器在固定任务族上单独改变的单一负载（R/V/W/D）。
- **Failure Record**：（首个未满足义务，终止标志）二元组，记录失败归因与 episode 结束原因。
- **Intention-to-Treat (ITT) Success**：目录压力下的评分，将"所需工具从未被检索到"的 episode 计为失败。
- **Completion Cap**：每请求模型最多生成的 token 数（主实验 1,024），截断发生于该限额。
- **Repair Activation**：修复需满足 L>0（压力造成损失）、Γ>0（加压下修复有效）、Σ>0（选择性优于非加压）三项，经 Holm 校正。
- **Routed Menu**：修复手段之一，模型仅在首次执行 operational tool 前看到完整菜单，之后菜单收窄至 operational tools。
- **Public-Handoff Controller**：修复手段，基于公开 handoff 链 deterministic 发出调用，将执行从模型转移至控制器。

## 可复现要素
- **数据集**：36 任务族、26,576 条评分行，含合成企业记录与 4 种压力变体；[公开](https://github.com/zichuanxu/tool-bottleneck-bench)（MIT License）。
- **代码**：确定性生成器、精确 grader、分析代码、manifests 全部开源；[GitHub](https://github.com/zichuanxu/tool-bottleneck-bench)。
- **模型权重**：使用 21 个托管推理端点（Anthropic/OpenAI/Google/Alibaba/DeepSeek/Tencent/Z.ai），需各自 provider credentials；[未开源自有权重]。
- **关键超参**：completion cap=1,024 tokens（扫掠至 16,384）；episode budget=128,000 tokens；call limit=certified minimum+2；turn limit=call limit+2；temperature/seed 随 frozen profile；parallel_tool_calls=false。
