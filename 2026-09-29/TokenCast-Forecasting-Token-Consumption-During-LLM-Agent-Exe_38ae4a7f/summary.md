---
title: "TokenCast-Forecasting-Token-Consumption-During-LLM-Agent-Exe"
source: https://arxiv.org/pdf/2609.35760v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:14:56"
field: "LLM Agent 系统效率与成本优化"
keywords: ["LLM Agent", "token consumption prediction", "cost forecasting", "segment composition", "prediction intervals", "agent execution"]
innovations: ["可组合的段成本三元组表示与精确合成恒等式", "分阶段前缀-后缀预测器结合直接路径", "无需额外LLM调用的动态任务级token预测"]
benchmarks: ["SWE-bench Verified", "Search-R1", "MMLU-Pro", "LongBench-v2", "LiveClawBench"]
---

# 论文速读：TokenCast: Forecasting Token Consumption During LLM Agent Execution

## 一句话总结
TokenCast 是一种面向 LLM Agent 执行过程的 token 消耗预测方法，通过学习可组合的执行段成本表示（调用次数、上下文增量、残差），在无需额外 LLM 调用的前提下，实现任务开始、调用开始、调用中、任务更新四个时间点的实时 token 预测。在 4 个基准和 6 个 Agent 模型上，相比最强对比方法平均降低 MAE 14.5%，并在离线预算控制重放中节省 21.3% 的 token。

## 研究问题与动机
- **Agent 执行中 token 消耗变化剧烈**：同一任务在不同运行中的 token 消耗可相差达 30 倍，难以在执行前准确预估，影响用户成本规划和资源调度。
- **上下文累积导致成本耦合**：每次 Agent 调用都会将工具反馈和中间结果追加到上下文中，使得后续每次调用的输入长度持续膨胀，消费具有强时序依赖性。
- **现有方法存在适用边界**：
  - 请求级预测方法（如 TRAIL、EGTP、TIE）面向单次模型调用，无法处理 Agent 任务的多轮迭代依赖；
  - 任务级预测方法（如 Self-Prediction）仅支持执行前的一次性预估，无法随执行推进动态修正；
  - 结构化工作流预测（如 Chimera、Pythia）依赖预知的程序结构或语义依赖图，而开放域 Agent 不存在此类先验。
- **预测需兼顾多时间粒度与多预测时机**：实际系统需要在 Task Start（执行前）、Call Start（请求组装后）、In-call Update（生成中）、Task Update（调用完成后）四个时间点分别提供调用级和任务级预测。

## 核心贡献（创新点）
1. **提出可组合的执行段成本表示与精确合成恒等式**：将每个执行段表示为三元组 $\phi = (n, g, b)$（调用次数、净上下文增量、残差消耗），并通过恒等式 $\phi_{A \circ B} = (n_A + n_B, g_A + g_B, b_A + b_B + n_B g_A)$ 实现相邻段的精确合成，显式刻画了上下文增长对后续调用的成本放大效应。
2. **设计分阶段前缀-后缀预测器**：将任务剩余成本预测分解为前缀段表示预测、边界状态预测和后缀段表示预测三部分，结合直接回归路径与组合路径，利用已完成的执行证据动态更新预测，且无需额外 LLM 调用。
3. **实现跨基准、跨模型的泛化预测能力**：通过 5-fold 交叉验证和跨拟合（cross-fitting）策略，使后缀模型训练时暴露于Out-of-fold的边界误差，提升泛化性；在零样本迁移场景下，仅需 20 个目标域任务即可使预测误差低于 Self-Prediction。
4. **提供可校准的预测区间与预算控制验证**：使用独立的分位数 LightGBM 模型生成 90% 预测区间，并经验证在 SWE-bench Verified 上使 MIS 相对 Self-Prediction 降低 32.0%；离线预算控制重放显示，在匹配固定预算完成率的条件下，TokenCast 平均节省 21.3% 的 token。

## 方法详解
- **段表示定义**：对于从输入长度 $L_A$ 开始的段 A，包含 $n_A$ 次调用，总消耗 $C_A$，定义净上下文增量 $g_A$ 和残差 $b_A = C_A - n_A L_A$，则 $\phi_A = (n_A, g_A, b_A)$。
- **组合恒等式**：当段 B 紧接段 A 时（$L_B = L_A + g_A$），组合表示为 $\phi_{A \circ B} = (n_A + n_B, g_A + g_B, b_A + b_B + n_B g_A)$，其中交叉项 $n_B g_A$ 反映了段 A 的上下文增量被段 B 的 $n_B$ 次调用重复计费。
- **预测器架构**：
  - **调用级预测**：Call Start 和 In-call Update 使用独立 LightGBM 模型预测当前调用的完整消耗 $C_k$。
  - **任务级预测**：维持两条路径：
    - 直接路径：直接预测剩余任务消耗 $R_k$；
    - 组合路径：预测前缀段表示 $(\hat{n}_A, \hat{g}_A, \hat{b}_A)$ 及其结束边界，再预测后缀段表示，通过组合恒等式得到剩余消耗。
  - 最终预测 $\hat{T}_k = S_k + \hat{R}_k$，其中 $S_k$ 为已确认消耗。
- **加权损失函数**：为缓解上下文增量和调用次数误差通过组合恒等式放大传播，对前缀段的上下文增量误差和后缀段的调用次数误差施加权重：$\ell_g = n_B |\hat{g}_A - g_A|$，$\ell_n = L_B |\hat{n}_B - n_B|$。
- **跨拟合训练**：任务被分为 F 折，每折的前缀预测由其余折训练的模型生成，后缀训练的残差标签按预测边界重新计算：$b_B^{\text{train}} = C_B - n_B \hat{L}_B$，其中 $\hat{L}_B = L_A + \hat{g}_A^{\text{oof}}$。
- **校正模型**：在直接和组合预测固定后，训练一个校正模型 $h_\psi(\mathbf{q})$ 拟合预测残差，最终预测为 $\hat{C}_{\text{comp}} + h_\psi(\mathbf{q})$。
- **区间校准**：使用独立分位数 LightGBM 模型预测 0.05 和 0.95 分位，通过保留校准集上的区间残差分位数对称展宽原始区间，确保 90% 覆盖。
- **更新机制**：每次调用完成后触发 Task Update，利用已观测证据（已确认消耗、工具结果、调用历史统计）重新预测 $\hat{R}_k$；In-call Update 利用已生成的 token 前缀和流式计时信息细化当前调用预测。

## 实验与结果
- **数据集**：SWE-bench Verified（软件修复）、Search-R1（检索增强问答）、MMLU-Pro（知识推理）、LongBench-v2（长上下文理解），共 240 个任务、11,712 条执行轨迹。
- **Agent 模型**：GPT-5.4、Claude Opus 4.6、Gemini 3.1 Pro、DeepSeek-V4-Pro、Qwen3.8-27B、Llama-3.2-3B-Instruct。
- **基线方法**：TRAIL（隐状态分类）、EGTP（熵加权回归）、TIE（文本嵌入+log-t分布）、Self-Prediction（Agent 自检估）。
- **主要结果**（SWE-bench Verified + GPT-5.4）：
  - Task Start：TokenCast MAE = 144.0k，最强基线 Self-Prediction = 152.0k，降低 5.3%；
  - Call Start：TokenCast = 64.5，最强基线 TIE = 69.5，降低 7.2%；
  - In-call Update：TokenCast = 38.9，最强基线 EGTP = 74.6，降低 **47.9%**；
  - Task Update：TokenCast = 80.0k，最强基线 TRAIL = 115.0k，降低 **30.4%**。
- **跨 96 个组合的平均表现**：TokenCast 相对最强对比方法的 MAE 平均降低 **14.5%**，其中 In-call Update（30.4%）和 Task Update（27.8%）提升显著，Task Start（-2.2%）和 Call Start（1.9%）略有波动。
- **预测区间可靠性**：90% 区间在锚定任务上的平均覆盖率为 90.6%，MIS 较 Self-Prediction 降低 32.0%（如 Task Start MIS 从 3334.6k 降至 1547.9k）。
- **泛化能力**：在 LiveClawBench 零样本跨域迁移中，20 个目标任务微调后 MAE 降至 Self-Prediction 的 0.82 倍；跨模型迁移时，3-10 个目标任务即可超越目标模型单独训练的基线。
- **预算控制**：在 SWE-bench Verified 的 7 个预算阈值重放中，TokenCast 以匹配固定预算的完成率，平均节省 **21.3%** token；Self-Prediction 因预测开销过大（含额外 LLM 调用）反而浪费更多资源。
- **效率**：每次调用后更新预测的平均累计耗时为 32.8 ms/run，仅占 GPT-5.4 执行中位数 129 s 的不到 0.03%。

## 相关工作脉络
- **请求级长度预测**：TRAIL、EGTP、TIE 等通过隐状态、熵统计或文本嵌入预测单次响应长度，但假设输入已知且目标为单次调用，不适用于 Agent 多步迭代的耦合成本。
- **任务级一次性预测**：Self-Prediction 让 Agent 自检任务环境并预估总消耗，但仅在 Task Start 提供单一估计，无法随执行推进更新。
- **结构化工作流预测**：Chimera（量化随机森林）、Pythia（历史轨迹分析）依赖预知的程序依赖图或工作流路径，而开放 Agent 的执行图在执行前不可知。
- **多步推理调度**：DSPy、Parrot、SGLang 等通过编译或语义分析优化 LLM 应用执行，关注调度而非 token 消耗预测。
- **Token 经济性分析**：Salim 等、Zhu 等对已部署 Agent 的 token 使用模式进行事后分析，但不提供事前或事中预测能力。
- **本文定位**：TokenCast 填补了"开放 Agent 任务在多次预测时间点的动态 token 预测"空白，通过组合恒等式显式建模上下文累积成本，且不依赖额外 LLM 调用，区别于 Self-Prediction 的自检方案和请求级方法的任务级扩展。

## 局限性与未来方向
- **训练数据依赖性**：模型性能依赖历史轨迹数据，在极端长尾任务或全新领域上可能泛化不足（尽管实验显示少量目标任务可快速适应）。
- **预测开销未完全消除**：虽然无需额外 LLM 调用，但特征提取和模型推理仍需 CPU/GPU 计算，在高并发场景下可能成为瓶颈。
- **仅预测输入+输出 token**：未显式建模 reasoning tokens（如 Chain-of-Thought）对上下文的贡献差异，部分场景下可能低估总消耗。
- **静态模型部署**：当前预测器在训练后固定，未探索在线增量学习以持续适应漂移的执行模式。
- **未来方向**：作者明确提出将预测器集成到运行时决策框架中，实现基于动态 token 预测的主动预算管控（如提前终止、上下文压缩触发），是后续研究的自然延伸。

## 研究启发与可借鉴点
1. **组合恒等式的代价解耦思想**：将总消耗分解为"起始输入基线 + 增量残差 + 上下文放大项"，这一分解范式可迁移至其他序列决策场景（如多步推理成本预估、分布式 Agent 通信开销建模）。
2. **跨拟合（cross-fitting）在级联预测中的应用**：后缀模型训练时使用 Out-of-fold 的前缀预测边界，有效缓解级联误差传播，可推广至任意多阶段预测流水线。
3. **轻量级树模型替代大模型自检**：用 LightGBM 替代 Self-Prediction 的 LLM 自检，在保持精度的同时彻底消除预测开销，为资源受限场景提供可行方案。
4. **预测区间校准的工程实践**：通过分位数模型+残差分位数展宽的方式实现可靠预测区间，可直接复用到服务调度、成本管控等对不确定性敏感的下游任务。
5. **任务级特征工程的丰富设计**：TF-IDF 任务文本、最近工具操作历史、生成前缀哈希、流式计时等特征的融合策略，为 Agent 行为预测提供了可复用的特征模板。

## 关键术语表
- **TokenCast**：论文提出的 token 消耗预测框架，通过学习可组合的段成本表示实现动态预测。
- **Segment representation $(n, g, b)$**：执行段的三元组表示，分别记录调用次数、净上下文增量和扣除输入基线后的残差消耗。
- **Composition identity**：相邻段合成的精确恒等式 $\phi_{A \circ B} = (n_A + n_B, g_A + g_B, b_A + b_B + n_B g_A)$，显式刻画上下文增长的累积成本。
- **Prediction points**：四个预测时机：Task Start（执行前）、Call Start（请求组装后）、In-call Update（生成中）、Task Update（调用完成后）。
- **Cross-fitting**：将任务分折后，用部分折的预测结果作为另一部分折后缀模型的输入，减少训练-推理分布偏移。
- **Correction model**：在校验直接和组合预测残差后训练的附加模型，用于修正系统误差。
- **WAPE（Weighted Absolute Percentage Error）**：以平均目标消耗为权重的平均绝对百分比误差，用于衡量相对预测精度。
- **Trace completion**：离线重放中 Agent 达到记录终端状态的比例，用于评估预算控制策略的完整性。

## 可复现要素
- **数据集**：SWE-bench Verified、Search-R1、MMLU-Pro、LongBench-v2 均为公开基准；11,712 条执行轨迹通过 DeepSeek Harness 和 OpenHands 收集，论文声明代码已开源。
- **代码**：https://github.com/DEFENSE-SEU/TokenCast 已公开。
- **模型**：GPT-5.4、Claude Opus 4.6、Gemini 3.1 Pro 通过 API 访问；Qwen3.8-27B、Llama-3.2-3B-Instruct 自托管部署（bf16，8×A100）；DeepSeek-V4-Pro 通过 API。
- **关键超参**：LightGBM 为基础预测器（推理 0.8 ms）；TF-IDF 特征维度 12,000；前缀哈希 256 桶；w=3/5/8 的最近工具操作窗口；5-fold 交叉验证；90% 预测区间。
- **训练配置**：每折训练集由剩余折组成，验证集选超参，校准集定区间；3 个随机种子池化结果。
