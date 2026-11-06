---
title: "What-Do-RTL-Repair-Benchmarks-Actually-Measure"
source: https://arxiv.org/pdf/2610.28595v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 03:00:04"
field: "硬件描述语言与RTL自动综合/验证"
keywords: ["RTL repair", "benchmark evaluation", "formal verification", "simulation oracle", "specification-only baseline", "measurement confound"]
innovations: ["揭示仿真与形式化评测因reset/首周期语义产生相反结论的测量混淆", "提出spec-only对照基线以区分真实修复与规范重生成", "开源RTLRepairDataset+RTLRepairBench及双oracle评测harness"]
benchmarks: ["RTLRepairBench", "RTLRepairDataset", "VerilogEval", "RTLLM"]
---

# 论文速读：What Do RTL Repair Benchmarks Actually Measure

## 一句话总结
本文构建了 RTLRepairDataset（5,784 对）和 RTLRepairBench（459 个测试用例），并在此基础上揭示了当前 RTL 修复基准评测中的两个测量混淆因素：（1）仿真与形式化评分因 reset/首周期语义不同而给出相反结论；（2）"修复"与"仅按规范重新生成"在输出正确性上无法区分，提出了标准化的评测控制方案。

## 研究问题与动机
- **问题一：RTL 行为修复难以可靠评测。** 候选代码可通过编译但行为与规范不符，且模型得分依赖 oracle 合约定义与输入条件，现有工作缺少可复现的行为修复评测框架。
- **问题二：仿真（Tier S）与形式化等价（Tier F）的评测结果可能给出相反结论。** 评测契约（如 reset 序列、首周期比较）的设置会显著影响结论的方向和幅度。
- **问题三：一次"通过"的输出无法证明模型真正利用了输入的 fault RTL。** 模型可能绕过故障 RTL，仅凭 specification 重新生成正确答案，导致"修复得分"虚高。
- **现有方法不足：** 已有工作（RTLCoder、OriGen、VeriReason 等）主要聚焦生成与语法修复，缺少对行为修复评测本身的审计，也缺乏显式 oracle 合约与"spec-only"对照基线。

## 核心贡献（创新点）
1. **构建并开源 RTLRepairDataset + RTLRepairBench**，使用 tool-gated 流水线将"语法失败"与"行为失败"分离；与已有基准的本质区别在于以 hash-bound 配对数据 + 语义 mutation 标签支撑可审计的行为修复评测。
2. **实证识别出两个测量混淆源**：reset/首周期语义导致仿真↔形式化结论反转；spec-only 恢复与修复恢复的得分区间重叠甚至反超（+18.3 pp）；与已有工作的区别在于首次在 RTL 修复评测中把这两个 confound 显式量化。
3. **提出四项评测控制建议**（分离语法/行为 bug；报告 initialization/reset/首周期/unknown-value 语义；加入 spec-only 基线；披露 oracle 覆盖率与 undecided 案例），为后续基准的可复现性和可解释性提供协议级约束。

## 方法详解
- **数据构造 pipeline（tool-gated）：**
  - 训练种子来自 MG-Verilog，held-out 与去污池使用 VerilogEval 和 RTLLM，1,895 个历史训练种子与基准 130 个种子零重叠。
  - 每一对 (broken, fixed) 仅在"类专属 gate"确认注入失败后才保留。
  - RTLRepairDataset：5,784 对训练样本；RTLRepairBench：459 个 held-out 用例，含 374 个 syntactic mutation 与 85 个 behavioral mutation，其中 4 个语义算子未在训练中见过。
- **评测 Harness 与双 Oracle：**
  - **Tier S（仿真）：** 使用 verilator 做 lint、iverilog 跑仿真，候选与 golden module 在相同采样激励下跑 200 个 cycle，逐 cycle 比对输出；200 个 cycle 一致不构成等价性证明。
  - **Tier F（形式化等价）：** 针对每个设计匹配契约：枚举组合逻辑（17 例）、声明 reset 序列（60 例）或指定 preamble（8 例），在明确定义的初始化和 unknown-value 语义下进行等价性证明（Yosys SAT / ABC PDR / SMT induction）。
- **审计实验一（oracle 敏感性）：** 选取 79 个有确定性 Tier F verdict 的案例，对比"带 specification vs 不带 specification"在 Tier S / Tier F 下的修复通过数，发现 cycle-1 skip 和 reset-aware 初始化会改变评分方向。
- **审计实验二（repair vs regeneration）：** 对 274 个 mined model failures，分别输入三种 prompt：(1) broken RTL 无 spec；(2) broken RTL + spec；(3) spec-only，测量三者通过率差异，用以判断 spec-only 能否"模仿"修复成功。

## 实验与结果
- **数据集规模：** 训练 5,784 对；基准 459 个（syntactic 374 / behavioral 85）。
- **模型子集（6 个历史快照）：** GPT-5.5、Claude Opus 4.8、GPT-5.4-mini、GPT-4.1、GPT-4o-mini、Claude Opus 4.6。
- **关键数字：**
  - 在 79 个 tier-F 明确 verdict 案例中：Tier S（cycles 0–199）下 "diag → +spec" 效应为 **−11.4 pp**；但跳过首周期（cycles 1–199）后反转为 **+7.6 pp**；reset-aware 初始化为 **+8.9 pp**；Tier F 同样为 **+8.9 pp**。
  - reset-aware 重评分清除了 39 个 "Tier-S 失败 / Tier-F 证明" 的全部案例（12/12 diag 臂 + 27/30 spec 臂），说明**几乎所有仿真-形式化分歧都能归因于 reset 与首周期语义**。
  - Spec-only 基线对比：在 274 个 mined failures 上，spec-only 减 spec-conditioned 的 Δc1 范围为 **−8.5 pp 至 +18.3 pp**（Claude Opus 4.6 最高 +18.3 pp）。
  - 行为修复通过率（从 87.7%~100% 语法修复的子集中描述性统计）：**10.6% ~ 75.3%**，差距巨大。
- **最强结果：** 语法修复覆盖率 87.7%–100% 的模型子集中，行为修复仍有高达 64.7 pp 的绝对差距（10.6% vs 75.3%），凸显行为修复是更难的任务维度。

## 相关工作脉络
1. **RTLCoder / OriGen / CRAFTRTL / CodeV / VerilogDB**：专注 spec→RTL 生成或语法修复，未对行为修复的评测度量做审计。
2. **RTLFixer / OriGen self-reflection**：用编译器反馈修语法/编译错误，本文关注编译通过的 lint-clean 行为错误。
3. **AutoChip**：同时使用编译器与仿真反馈迭代修订，但未提供 spec-only 对照来区分修复与再生。
4. **VeriReason / CodeV-R1 / Wang et al.**：利用 testbench 奖励/偏好改进功能生成，评测仍缺少 oracle 合约的可审计声明。
5. **VeriDebug / ARSP / VeriRAG**：分别针对调试、语义分区修复、DFT 合规修复；本文的独特定位是把"评测本身"作为被测对象，提供协议级诊断。
6. **VerilogEval / RTLLM**：本工作的 held-out 与去污池来源，但与本文的"tool-gated paired data + 双 oracle 审计"形成互补而非替代。

## 局限性与未来方向
- 结论为描述性而非推断性（单样本生成、历史模型快照、非独立同分布的 case 聚类在 112 个源任务上）。
- 样本量有限（79 个可审计案例、274 个 mined failures），跨模型泛化性待验证。
- 未来工作应扩展到：多次采样、更大更丰富的 RTL 设计库、更广的模型阵容，以及机制归因的严格统计检验。

## 研究启发与可借鉴点
1. **spec-only 对照基线可作为"是否真修复"的最简探针**，任何生成型任务的"修复/编辑"评测都可借鉴这种三轮对照（no-spec / +spec / spec-only）设计。
2. **评测契约必须显式声明**（initialization / reset / 首周期比较 / unknown-value），否则不同实现的分数不可比；建议团队内部评测协议标准化这一层。
3. **tool-gated 数据构造思路**（先注入 mutant → 再用门控只保留确认失败的样本）可迁移到其他"缺陷修复"数据集构建中。
4. **Tier S ↔ Tier F 分歧的归因分析**提示：在硬件/代码评测中，仿真通过不等于等价性成立，建议在论文中同时报告两种 oracle 的结果与分歧原因。
5. **开源全链路可复现资产**（数据集 + 构造代码 + 评测 harness + 聚合证据）是提升基准可信度的最佳实践。

## 关键术语表
- **RTLRepairDataset**：5,784 对 (broken, fixed) 配对训练数据，由 tool-gated 流水线构建。
- **RTLRepairBench**：459 个 held-out 行为修复测试用例，含 syntactic 与 behavioral 两类 mutation。
- **Tier S（Simulation）**：基于 iverilog 的 200-cycle 采样仿真评测，仅能证伪不能证实等价。
- **Tier F（Formal Equivalence）**：基于 Yosys/ABC 的形式化等价性证明，支持组合/ reset / preamble 契约。
- **spec-only baseline**：仅提供规格不输入 fault RTL 的对照实验，用于检测模型是否"绕过修复直接生成"。
- **Δc1**：spec-only 通过率减去 spec-conditioned 修复通过率的差值，衡量"修复 vs 再生"的模糊度。
- **reset-aware initialization**：应用两边沿 reset 启发式初始化后再行评分，消除因状态未初始化导致的仿真-形式化分歧。
- **tool-gated construction**：用 linter/simulator/formal checker 作为 gate，只保留确认注入失败的真实样本进入数据集。

## 可复现要素
- **数据集**：RTLRepairDataset（5,784 对）与 RTLRepairBench（459 个）均已开源。
- **代码/harness**：构造代码与评测 harness 已开源，地址：https://github.com/Evaaaz/RTLRepairBench
- **训练种子来源**：MG-Verilog（snapshot hash 82944845640b…，MIT）；去污池：VerilogEval（c498220，MIT）、RTLLM（41b2689，MIT）。
- **关键超参**：Tier S 仿真步数 200 cycles；6 模型子集使用 greedy decoding，output budget 4,096 tokens。
- **模型版本声明**：所有模型标签为历史 endpoint-output 快照，非不可变版本。
