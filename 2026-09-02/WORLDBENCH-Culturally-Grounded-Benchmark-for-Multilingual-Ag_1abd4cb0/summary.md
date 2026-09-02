---
title: "WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag"
source: https://arxiv.org/pdf/2609.01056v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:19:36"
field: "多语言智能体评估"
keywords: ["多语言智能体", "文化接地基准", "状态保持", "约束任务成功率", "多应用工作流", "LLM-as-a-Judge"]
innovations: ["提出CTS指标联合评估任务正确性与环境状态保持", "文化接地原生多语言构建而非翻译派生", "设计含干扰项的多应用沙箱测试床"]
benchmarks: ["WORLDBENCH"]
---

# 论文速读：WORLDBENCH-Culturally-Grounded-Benchmark-for-Multilingual-Ag

## 一句话总结
WORLDBENCH 是一个面向多语言智能体的文化接地型基准测试，包含 1,600 个任务、7 种语言和 8 种文化场景，通过结构化动作接口评估智能体在真实文件工作流中的任务完成能力与环境状态保持能力；实验表明当前前沿模型仅能达到 49.2% 的约束任务成功率（CTS），暴露出长程执行、多语言本地化和状态保持方面的脆弱性。

## 研究问题与动机
1. **上下文接地不足**：现有基准测试任务多为通用指令，缺乏对角色、位置、语言和工作习惯等具体上下文的建模，导致智能体可能基于错误假设行动。
2. **状态保持缺失**：现有评估多关注任务正确性，忽略了智能体在完成目标任务时可能意外修改无关文件或覆盖不应变更的记录，这在真实工作流中至关重要。
3. **多语言覆盖薄弱**：多数多语言基准通过将英语任务翻译来获得多语言支持，但任务场景、用户假设和产物仍受源语言绑定，无法验证智能体在真实文化语境下的表现。
4. **长期执行脆弱性**：真实工作流需要跨多步骤、多工具协调，但现有基准缺乏对长程轨迹中错误累积和状态漂移的评估。

## 核心贡献（创新点）
1. **提出了 WORLDBENCH 基准**：首个文化接地、原生语言构建的多语言智能体基准，1,600 个任务覆盖 7 种语言 8 种文化，区别于翻译派生基准。
   → 与 MAPS、WebArena 等基准的本质区别在于任务从种子到评估均在目标语言/文化语境中本地化构建，而非先英文后翻译。

2. **设计了"人工种子+自动扩展+文化审核"的构建流水线**：从人工撰写的 persona、scenario 和 constraint 种子出发，自动生成任务并经由语言/文化专家审核过滤。
   → 区别于 SWE-bench、AgentBench 等纯合成数据基准，WORLDBENCH 强调文化真实性和人工审计闭环。

3. **引入了 CTS（Constrained Task Success）指标**：将任务正确性（pass rate）与环境保持（preservation rate）联合为二元逻辑度量，揭示"看似正确但破坏工作环境"的失败模式。
   → 与 OfficeBench、OdysseyBench 仅评估最终状态不同，CTS 显式惩罚非目标文件的任何修改。

4. **构建了包含干扰项的多应用测试床**：每个测试床包含 15–20 个干扰物（相似文件名、重叠实体、过期版本），模拟真实办公环境中的信息噪声。
   → 与 OSWorld 等单应用或纯 GUI 基准相比，WORLDBENCH 强调跨应用（电子表格、文档、邮件、日历、Shell）的协同操作。

## 方法详解
**任务形式化**：将智能体任务建模为转移系统 $\mathcal{W} = (S, A, O, \delta)$，其中 $S$ 为环境状态空间，$A$ 为结构化动作空间，$O$ 为观测空间，$\delta: S \times A \to S \times O$ 为转移函数。智能体在每一步选择 $a_t \in A$，环境执行后返回观测 $o_t$，历史记录 $H_t = [(a_1, o_1), \dots, (a_{t-1}, o_{t-1})]$。

**工具环境**：包含五大工具族（Table 1）：Spreadsheet（读写单元格）、Document PDF（读写更新）、Messaging（检查/撰写消息）、Calendar（列出事件/创建不重叠条目）、Shell（沙箱内执行文件系统命令）和 System（全局控制/终止）。

**构建流水线**（Figure 1 / Appendix F）：
- 阶段1：人工撰写种子——每语言/地域设置 20 个 persona，每个 persona 4 个 seed task，共 80 个种子任务。
- 阶段2：自动扩展——生成 30 个增强 persona，每个 5–6 个候选任务，每设置产生 230–260 个候选。
- 阶段3：人工审核——40 名语言/文化专家审核每个候选，评估真实性、文化适宜性、文件充分性和评价标准正确性。
- 阶段4：过滤与平衡——保留后每设置 200 个任务、50 个 persona，最终 1,600 任务、400 persona。

**测试床构造**：电子表格、PDF、文档、日历、文本文件填充任务相关值；每测试床包含 15–20 个干扰物（相似文件名、重叠实体、过时版本）。

**评估协议**（§3.6）：
- 确定性函数：`contain`（目标文件含指定文本）、`not_contain`（目标文件不含禁止文本）、`excel_cell_value`（单元格值精确匹配）、`file_exist`（所需输出文件已创建）、`calendar_no_overlap`（日历事件无时间重叠）。
- LLM-as-a-Judge 函数：`evaluate_email`、`evaluate_note`（对开放性产物按标准评分）。
- CTS 定义：$\text{CTS}(t) = \text{pass}(t) \wedge \text{preserve}(t)$，其中 pass(t) 为所有任务特定评估函数通过，preserve(t) 为所有非目标文件在初始与最终状态间字节级保持不变。

**执行配置**（Appendix E）：最大迭代步数 30，每次输出一条 JSON 动作；生成非 JSON 时重试一次；连续 5 次相同动作触发停滞检测。

## 实验与结果
**评估模型**：9 个前沿 LLM 智能体（Gemini-3.1-Pro, Gemini-3.5-Flash, GPT-5, GPT-4o, Qwen-3-32B, Qwen-3-4B, Llama-3.3-70B, Llama-3.1-8B, EuroLLM-9B）。

**主要结果**（Table 3）：
- 最强模型 **Gemini-3.1-Pro 达到 49.2% CTS**，GPT-5（48.8%）和 Qwen-3-32B（48.0%）紧随其后。
- 所有模型的 Pass Rate 均显著高于 CTS，差距从 10.1（Gemini-3.1-Pro）到 16.8（Llama-3.3-70B）。
- 最弱模型 EuroLLM-9B 仅达 10.8% CTS。
- **Preservation Gap**：所有模型均存在显著的环境保持差距，表明"正确完成任务但破坏无关文件"是普遍失败模式。

**多语言鲁棒性**（Table 4 / Figure 4）：
- 英语设置（EN-US/EN-UK）表现最佳，中文（ZH）普遍最弱。
- Qwen 系列例外：Qwen-3-32B 在中文上达到 43.5%，为该语言设置最高分。
- Locale mismatch 在非英语设置中占失败比例的 21.4%（英语中仅 6.8%）。

**应用类型差异**（Figure 5）：
- Calendar 和 Document 任务得分较高。
- Messaging 和 Shell 任务 CTS 较低且 Preserve Gap 更大。

**失败模式分析**（Figure 8 / Table 10）：
- 强模型的失败主因：**Wrong output**（错误输出）。
- 所有模型的持续问题：**Collateral edit**（附带修改）。
- 弱模型的典型失败：**Iteration-cap hit**（达到迭代上限），EuroLLM-9B 达 43%。
- 人工程序分析揭示六大错误模式（Table 10）：
  - Distractor capture（24.6%）：编辑了与目标相似名称的文件。
  - Shell over-reach（17.8%）：在有专用 app action 时使用 shell 命令，导致批量误改。
  - Locale mismatch（15.2%）：使用错误本地化格式（如美式日期、小数点 vs 逗号）。
  - Redundant inspection（19.1%）：重复读取同一文件。
  - Premature termination（13.7%）：满足部分评估后即终止。
  - Schema violation（9.6%）：输出非 JSON 或动作参数缺失。

**迭代预算敏感性**（Table 8）：
- 将步数上限从 30 增至 50，最强模型 CTS 仅提升 0.3–1.9 个百分点。
- 弱模型虽从预算增加中获益更多，但绝对提升仍不足 2 分，表明问题在于执行质量而非探索时长。

## 相关工作脉络
1. **OfficeBench (Wang et al., 2024)**：多应用办公自动化基准，但以英语为中心，缺乏文化接地和多语言原生构建；WORLDBENCH 通过本土化种子和人工审核填补此空白。
2. **OSWorld (Xie et al., 2024) / WebArena (Zhou et al., 2024)**：聚焦操作系统控制和 Web 导航，WORLDBENCH 补充了文件型工作流（电子表格、文档、邮件、日历）的评估维度。
3. **AgentBench (Liu et al., 2025) / AgentBoard (Ma et al., 2024)**：多轮工具使用基准，但未显式评估状态保持（preservation）；WORLDBENCH 的 CTS 指标首次将"不破坏无关状态"作为硬约束。
4. **MAPS (Hofman et al., 2026)**：多语言智能体基准，但通过翻译获得多语言覆盖；WORLDBENCH 坚持每语言/文化独立构建，避免源语言偏差。
5. **OdysseyBench (Wang et al., 2025b) / WorkArena++ (Boisvert et al., 2025)**：长程办公工作流基准，但主要为英语场景；WORLDBENCH 扩展至 7 种语言并加入干扰项与状态保持评估。
6. **SWE-bench Pro (Deng et al., 2025)**：软件工程长程任务基准；WORLDBENCH 在通用办公领域提供可比但更接地气的跨文化评测。

## 局限性与未来方向
1. **仅限结构化文件工作流**：当前版本排除视觉桌面控制和实时 Web 交互，未来计划扩展以增强可复现性外的真实性。
2. **LLM-as-a-Judge 变异性**：`evaluate_email` 和 `evaluate_note` 依赖 judge 模型，虽与人工标注一致性高（κ=0.76–0.81），但仍引入额外方差；多数投票可将 κ 提升至 0.84。
3. **单次执行**：每个任务每模型仅执行一次，未报告方差或置信区间。
4. **跨地域不可比性声明**：各语言设置独立构建，虽平衡了主题、应用、评估函数和参考轨迹长度分布，但不宜直接归因为"语言效应"，而是多种因素（指令遵循、本地化文档惯例、文化接地、任务变异）的混合。
5. **迭代上限 30 步的限制**：长程任务可能未充分探索，但扩展至 50 步增益有限（Table 8）。

## 研究启发与可借鉴点
1. **状态保持作为硬约束**：CTS 指标揭示了"正确但破坏性"失败模式的普遍性，可作为后续智能体工作流评估的标准维度，建议在本团队基准中引入类似 preservation rate 指标。
2. **干扰项设计提升区分度**：15–20 个语义/名称相似的干扰文件有效区分了"找到正确目标"与"盲目修改"的能力，可在办公类基准中复用该策略。
3. **文化接地而非翻译**：原生语言构建+人工审核的流水线可避免翻译偏差，适合需要评估本地化格式（日期、货币、千位分隔符）的任务场景。
4. **LLM-as-a-Judge 的鲁棒性验证**：使用三个 judge 模型 + 多数投票验证判定一致性（Table 9），为开放型评估提供可复现的验证范式。
5. **错误模式分类框架**：Distractor capture、Locale mismatch、Shell over-reach 等六大模式可作为智能体失败分析的分类体系，辅助定位改进方向。

## 关键术语表
**WORLDBENCH**：文化接地、多语言智能体基准测试，包含 1,600 个任务、7 种语言、8 种文化场景。
**CTS (Constrained Task Success)**：约束任务成功率，联合评估任务正确性与非目标文件保持性的二元指标。
**Pass Rate**：任务特定评估函数全部通过的比例，不衡量环境保持。
**Preservation Rate**：非目标文件在任务执行前后保持不变的比率。
**Distractor Capture**：干扰捕获错误模式，智能体修改了与目标文件相似名称的无关文件。
**Locale Mismatch**：本地化不匹配错误模式，智能体使用了错误地区的格式约定（如日期顺序、小数符号）。
**Iterative Cap Hit**：达到最大迭代步数而未完成或终止，多见于弱模型。
**LLM-as-a-Judge**：使用独立 LLM 作为裁判评估开放性输出（如邮件、笔记）的质量。

## 可复现要素
- **数据集**：1,600 个任务，7 种语言（EN-US, EN-UK, IT, PT, ES, FR, DE, ZH），8 种文化场景。论文未提及是否开源，但从 arXiv 附注推断可能开源。
- **代码/权重**：模型通过官方 API（Gemini、GPT）和本地运行（Llama、Qwen、EuroLLM）评估；推理代码与提示模板见 Appendix B，详细评估函数见 Appendix C/D。
- **关键超参**：迭代上限 30 步，每次输出单条 JSON 动作，judge 温度 0 且输出上限 10 tokens，Agent 温度基于模型文档设置（Appendix E Table 6）。
