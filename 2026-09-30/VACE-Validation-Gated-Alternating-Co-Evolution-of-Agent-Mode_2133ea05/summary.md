---
title: "VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode"
source: https://arxiv.org/pdf/2609.37105v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:26:25"
field: "Agent 系统与多组件协同优化"
keywords: ["Agentic RL", "Harness Optimization", "Model-Harness Co-Evolution", "Validation Gate", "Qwen3.5", "OfficeQA", "AutomationBench", "GRPO"]
innovations: ["提出验证门控交替协同演化框架，将 Agentic RL 与轨迹驱动 Harness 修订连接为闭环", "在同一更新 checkpoint 上进行 paired 验证，严格改进门控决定是否采纳 Harness 修订", "实证表明 38.6% 的 Harness 修订在当前 checkpoint 上会降低性能，支撑门控设计的必要性"]
benchmarks: ["OfficeQA", "AutomationBench"]
---

# 论文速读：VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode

## 一句话总结
VACE 提出了一种验证门控的交替协同演化框架，通过"模型权重 RL 训练 → 轨迹驱动 Harness 修订 → 在同一更新 Checkpoint 上验证门控采纳候选 Harness"的闭环，联合优化 Agent 模型与执行编排（Harness）。在 OfficeQA 和 AutomationBench 上，VACE 均显著超越仅训练权重的 RL-only 与无门控交替方法。

## 研究问题与动机
- **核心问题**：Agent 性能同时依赖模型权重（$\theta$）与 Harness（$H$，含指令、技能、工具接口等），二者高度耦合——权重更新会改变模型对 Harness 的使用方式，而 Harness 更新会改变用于训练的执行轨迹，单一组件优化无法建模这种双向交互。
- **已有方法不足**：
  - **Weight-only RL**（如 Agent Lightning、Search-R1、WebAgent-R1）只更新权重，固定 Harness，忽略训练后 Harness 可能不再匹配的能力状态（例如某项技能在早期 checkpoint 有效，但训练后反而限制模型发挥）。
  - **Harness-only/Co-Harness/SIA** 虽涉及两者，但 SIA 先固定 Harness 优化再训练权重，WHALE 无外部验证门控直接采纳每轮 Harness 修订——均存在"修订未必在更新后的 checkpoint 上有效"的风险。
  - **轨迹驱动的修订本身存在风险**：论文统计 44 次 Harness 提议中 17 次在当前 checkpoint 上降低验证性能，说明修订与采纳应为两步决策。
  - **缺乏统一的训练-编排联合优化范式**：现有工作要么分段优化，要么交替但缺少基于同一 checkpoint 的严格验证门控。

## 核心贡献（创新点）
1. **提出 VACE 闭环框架**：将 Agentic RL 与轨迹驱动的 Harness 精炼连接为一个交替迭代闭环，使模型训练为 Harness 优化提供诊断证据，修订后的 Harness 再引导下一轮训练。
2. **引入 Checkpoint 特定的验证门控（Validation Gate）**：在每次 Harness 修订后，用更新后的模型权重固定执行验证任务，比较 incumbent 与 candidate；仅在 candidate 严格优于 incumbent 时才采纳，否则保留原 Harness。
3. **在两套 Agent Benchmark 上系统验证联合优化优于单一组件与调度变体**：在 OfficeQA 和 AutomationBench 上对比了 Static、Harness-only、RL-only、SIA-style、WHALE-style 与 VACE，证明门控交替协同演化带来显著收益。
4. **提供了修订采纳的实证分析**：44 次 Harness 提议中有 38.6%（17/44）被门控拒绝，揭示了"修订≠有效"的事实，支撑门控设计的必要性。

## 方法详解
- **形式化目标**：最大化期望任务奖励 $J(W,H)=\mathbb{E}_{x\sim\mathcal{P}}\mathbb{E}_{\tau\sim p_{W,H}(\cdot|x)}[r(x,\tau)]$，其中 $W$ 为模型权重、$H$ 为 Harness 配置，二者耦合体现在轨迹分布 $p_{W,H}$ 中。
- **三轮交替循环**（Algorithm 1）：
  1. **Stage I（模型训练）**：固定当前 Harness $H_t$，用 Agentic RL 优化器 $\mathcal{M}$（GRPO + DAPO-style 过采样/零优势组过滤）更新权重，产出 $W_{t+1}$ 与轨迹集合 $\mathcal{T}_t$（含失败执行，供诊断）。
  2. **Stage II（Harness 修订）**：Harness 优化器 $S$（基于 HarnessEvolve 的轨迹驱动修订）根据 $\mathcal{T}_t$ 提出一个候选 Harness $H'_t$，复用已收集轨迹，无需额外数据收集。
  3. **Stage III（验证门控）**：固定 $W_{t+1}$，在同一验证集 $\mathcal{V}$ 上分别以 $H_t$ 和 $H'_t$ 评估，计算 $\Delta^H_t = \widehat{R}_{\mathcal{V}}(W_{t+1}, H'_t) - \widehat{R}_{\mathcal{V}}(W_{t+1}, H_t)$；若 $\Delta^H_t > 0$ 则采纳 $H_{t+1}=H'_t$，否则 $H_{t+1}=H_t$（平局保留 incumbent）。
- **关键设计细节**：
  - **验证重放**：incumbent 与 candidate 在同一 checkpoint、相同任务 ID、同一评估器下对比（paired evaluation），控制噪声。
  - **修订范围**：Harness 编辑聚焦于技能与执行引导，评估器、工具语义、环境转移规则保持固定。
  - **测试集不参与优化**：仅在选优后用于最终报告。
- **奖励设置**：OfficeQA 使用二元正确性奖励；AutomationBench 使用密集 partial-credit 奖励。使用 GRPO [Shao et al., 2024] + DAPO-style 过滤全零优势组 [Yu et al., 2025]。

## 实验与结果
- **数据集**：
  - **OfficeQA**：84 train / 53 val / 109 test，binary accuracy。
  - **AutomationBench**：209 tasks（HR 65 / Marketing 72 / Finance 72），拆分 80 train / 58 val / 71 test，部分 credit。
- **模型**：Qwen3.5-9B；基座实现基于 Uni-Agent [Ding et al., 2026]。
- **基线**：Static agent、Harness-only、RL-only（固定 $H_0$）、SIA-style（先 Harness-only 优化后固定 Harness 再 RL）、WHALE-style（与 VACE 相同优化器与交替顺序，但无外部验证门控）。
- **主要结果**（Table 2，均值取 3 次评测）：
  - **OfficeQA 准确率**：VACE 45.26% > RL-only 38.83%（+6.43 pp）> WHALE-style 40.67%（+4.59 pp）> SIA-style 40.67%（+4.59 pp）> Harness-only 35.28%（+9.98 pp）> Static 32.11%（+13.15 pp）。
  - **AutomationBench 平均 partial credit**：VACE 75.19% > RL-only 66.10%（+9.09 pp）> WHALE-style 68.25%（+6.95 pp）> Harness-only 61.62%（+13.58 pp）> SIA-style 65.35%（+9.85 pp）> Static 50.26%（+24.94 pp）。
  - **最强结果**：VACE 在 AutomationBench 达 75.19%（HR 75.18 / Marketing 72.03 / Finance 78.25）。
- **Phase-level 分析**（Table 3）：OfficeQA 12 轮 Harness 试验，7 次接受 / 2 次平局 / 3 次拒绝；AutomationBench 32 轮试验，18 次接受 / 0 次平局 / 14 次拒绝。
- **动态轨迹**（Figure 3）：保留验证分从 OfficeQA 30.28%→47.17%（+16.89 pp），AutomationBench 53.08%→81.42%（+28.34 pp）；进度非单调，权重阶段可能短暂下降。

## 相关工作脉络
1. **Agentic RL**（Search-R1、WebAgent-R1、AgentCPM-Explore、Agent Lightning、ToolRL、RAGEN）：侧重从执行反馈中更新模型权重；VACE 在此基础上将执行轨迹同时用于 Harness 修订，并将修订嵌入交替循环。
2. **Harness / Prompt 优化**（Reflexion、ExpeL、GEPA、Agentic Context Engineering、Darwin Gödel Machine、HarnessEvolve、DSPy、TextGrad）：聚焦静态或迭代改进指令/技能；VACE 的关键区别在于与 RL 训练的闭环耦合与 checkpoint 特定验证。
3. **SIA**（Hebbar et al., 2026）：先用反馈 Agent 优化 Harness，再固定 Harness 进行 RL；与 VACE 的差异在于 SIA 是一阶段先行、再冻结，VACE 是持续交替并逐轮门控验证。
4. **Co-Harness**（Chen et al., 2026b）：交替失败驱动的 Harness 精炼与成功轨迹 SFT；与 VACE 的区别在于 VACE 使用 Agentic RL（而非 SFT）作为权重更新器，并强调在同一 checkpoint 上的 paired 验证门控。
5. **WHALE**（Kim et al., 2026，并发工作）：交替 Online Rejection-Sampling Fine-Tuning 与可执行 Harness 搜索，无外部验证门控；VACE 在此基础上引入验证门控，使Harness 采纳需通过 updated checkpoint 上的性能比较。

## 局限性与未来方向
- **数据集与模型规模有限**：仅使用 Qwen3.5-9B 一个模型尺寸与两个 Agent benchmark（OfficeQA、AutomationBench），对更大模型或更广泛环境的泛化未验证。
- **Harness 编辑范围受限**：仅修改技能与执行指导，工具语义、评估器、环境转移规则固定，限制了整体优化空间。
- **固定交替调度**：VACE 采用固定轮次交替；论文提出可采用反馈驱动调度（feedback-driven scheduling），根据任务结果与训练进度动态决定何时更新模型或 Harness。
- **验证重用噪声**：多次使用同一验证集做 Harness 采纳决策会引入适应性选择效应（adaptive selection effects），且严格改进门控依赖带噪声的点估计，可能在边缘情况下误采纳或误拒绝。
- **计算开销**：每轮 Harness 修订与验证都会增加计算成本（额外的轨迹复用评估）。

## 研究启发与可借鉴点
1. **验证门控机制可迁移**：任何"生成式修订 + 评估决策"的框架（如 LLM 自我改进、工具链优化、SOP 自动修正）均可借鉴"在同一 checkpoint/状态下 paired 比较后采纳"的模式，避免盲目迭代带来的性能退化。
2. **轨迹复用的诊断价值**：VACE 在 RL 阶段收集的失败轨迹直接用于 Harness 修订，无需额外数据收集。这种"一次 rollout、多重收益"的设计对资源受限的 Agent 训练有参考价值。
3. **交替 vs. 一阶段先修的调度启示**：SIA-style 的实验结果（AutomationBench 65.35% vs. VACE 75.19%，差 9.85 pp）表明，持续交替比先行冻结 Harness 更利于探索联合最优；这为 Agent 系统中多组件协同优化的调度策略提供了实证依据。
4. **H/T/R 统计可作为系统健康指标**：论文展示了每轮 Harness 提议的接受/平局/拒绝计数，这类统计可作为后续系统的可观测信号，用于判断修订质量、调节生成策略或触发人工介入。
5. **可与本团队方向结合的创新机会**：若团队在 Agent 规划、工具使用或长程工作流优化方向有积累，可将 VACE 的门控交替范式迁移到代码 Agent、RAG 系统或多 Agent 协作场景中，探索不同评估粒度（任务级 / 步骤级 partial credit）下的收益。

## 关键术语表
- **VACE（Validation-Gated Alternating Co-Evolution）**：论文提出的框架，通过验证门控的交替循环联合优化 Agent 模型权重与 Harness。
- **Harness（H）**：Agent 执行系统的非权重部分，包括指令、技能库、工具接口、执行规则等，决定模型能力如何被调用。
- **Agentic RL**：将强化学习应用于交互式 Agent 的训练，从任务执行轨迹中学习，常见算法包括 GRPO、PPO 等。
- **Checkpoint 特定验证**：在同一模型权重 checkpoint 下对比不同 Harness 的性能，确保比较条件一致、无混入权重变化噪声。
- **严格改进门控（Strict-improvement gate）**：仅当候选 Harness 在验证集上的分数严格高于 incumbent 时才采纳，平局或退化均保留 incumbent。
- **WHALE-style 无门控交替**：与 VACE 相同优化器与交替顺序，但直接采纳每轮 Harness 修订输出，无外部验证过滤。
- **SIA-style 先修后训**：先在 Harness-only 阶段选出最优 Harness 并固定，再在此 Harness 下进行完整 RL 训练。
- **DAPO-style 零优势过滤**：在 GRPO 训练中过滤掉组内所有样本 advantage 均为零的 batch，避免无效梯度更新。

## 可复现要素
- **数据集**：OfficeQA（Singhvi et al., 2025）、AutomationBench（Shepard & Salimans, 2026）；论文使用自定义划分，具体 split 数据见正文 Table 1。OfficeQA 原始数据公开；AutomationBench 论文声明为 arXiv 公开。
- **代码/权重**：论文基于 Uni-Agent（https://github.com/verl-project/uni-agent）实现；模型为 Qwen3.5-9B（公开）；具体 VACE 代码未单独声明开源仓库，论文未明确提及是否开源。
- **关键超参**：使用 GRPO [Shao et al., 2024] + DAPO-style 过滤 [Yu et al., 2025]；RL cap（$B_W$）与 Harness cap（$B_H$）的具体数值论文附录 A Table 5 中以符号表示，详细数值论文未在本节展开；每轮生成 1 个 Harness 候选；验证集大小 OfficeQA 53 / AutomationBench 58。
- **评估协议**：选优策略为"选取优化过程中最高验证分对应的 checkpoint 与 Harness"，报告 3 次独立测试评估均值与标准差。
