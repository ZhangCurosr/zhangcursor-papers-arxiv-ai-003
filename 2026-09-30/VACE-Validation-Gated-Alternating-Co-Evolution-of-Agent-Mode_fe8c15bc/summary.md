---
title: "VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode"
source: https://arxiv.org/pdf/2609.37105v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:26:41"
field: "agent 训练与自我进化"
keywords: ["agentic reinforcement learning", "harness optimization", "model-harness co-evolution", "validation-gated", "trajectory-driven refinement", "LLM agents"]
innovations: ["提出 checkpoint-specific 验证门控机制，拒绝有害 harness 修订再进入下一轮 RL", "构建 agentic RL 与 trajectory-driven harness 精炼的交替共进化闭环", "系统对比 6 种调度策略，量化 gate 带来 4.59-9.09 pp 增益"]
benchmarks: ["OfficeQA", "AutomationBench"]
---

# 论文速读：VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode

## 一句话总结
本文提出 VACE 框架，通过交替执行 agentic RL 与 trajectory-driven harness 优化，并引入 checkpoint-specific 验证门控机制，使模型权重 θ 与执行系统（harness）H 相互适应，在 OfficeQA 和 AutomationBench 上显著优于仅优化权重的 RL-only 方法。

## 研究问题与动机
1. **模型与 harness 耦合性问题**：agentic RL 只更新权重 θ，固定 harness H；harness 优化只改指令/技能，固定权重。两者耦合但从未协同优化——权重更新改变模型使用 harness 的方式，harness 更新又改变后续 RL 的轨迹分布。
2. **harness 修订不一定对已更新模型有效**：基于轨迹提出的技能修订（如增加检索步骤）可能在旧 checkpoint 表现更好，但在训练后的新 checkpoint 反而消耗交互预算、降低性能。
3. **现有 co-optimization 方法缺乏验证门控**：SIA 先修 harness 再固定训练；WHALE 无外门控直接采纳每次修订；本文认为应在每个 RL 阶段后以当前 checkpoint 验证候选 harness，仅采纳验证提升的方案。
4. **单组件优化上限明显**：实验显示 RL-only 在 OfficeQA 仅 38.83%、AutomationBench 66.10%，仍有较大提升空间。

## 核心贡献（创新点）
1. **提出 VACE 交替共进化框架**：将 agentic RL 与 trajectory-driven harness 精炼闭合循环连接，实现模型权重 θ 与 harness H 的反复互相适应；与已有方法本质区别在于显式构建了"训练→提议修订→验证→采纳/拒绝→下一轮训练"的闭环。
2. **引入 checkpoint-specific 验证门控机制**：每次 RL 阶段结束后，用更新后的权重 W_{t+1} 在验证集上同时评估 incumbent H_t 与候选 H'_t，仅当 Δ_t^H = R̂_V(W_{t+1}, H'_t) - R̂_V(W_{t+1}, H_t) > 0 才采纳；区别于 WHALE 无门控直接提交修订的做法。
3. **系统性地对比了 6 种优化调度策略**：在同一组权重/ harness 优化器下公平比较 static、RL-only、Harness-only、SIA-style、WHALE-style、VACE，揭示 validation gate 的关键作用；实验发现 44 次 harnes 提案中 17 次（38.6%）会降低验证性能并被拒绝。
4. **公开完整实验数据与消融分析**：提供阶段级验证轨迹曲线、A/T/R 统计、同 checkpoint 接受/拒绝案例，为后续研究提供可复现基准。

## 方法详解
- **建模**：agent 状态为二元组 (W, H)，任务 x 在策略 p_{W,H} 下生成轨迹 τ，目标为最大化期望奖励 J(W,H) = E_{x~P}E_{τ~p_{W,H}}[r(x,τ)]。
- **交替更新循环（Algorithm 1）**：
  1. **Stage I 模型训练**：(W_{t+1}, T_t) = M(W_t; H_t, D_train)，在当前 harness 下执行 agentic RL（使用 GRPO + DAPO-style 过采样），收集轨迹批次 T_t。
  2. **Stage II 修订提议**：H'_t = S(H_t, T_t)，利用失败轨迹的诊断证据（复用 HarnessEvolve 思路）提出一次修订，无需额外采样。
  3. **Stage III 验证门控**：计算 b_t = R̂_V(W_{t+1}, H_t) 与 c_t = R̂_V(W_{t+1}, H'_t)，Δ_t^H = c_t - b_t；若 Δ_t^H > 0 则 H_{t+1}=H'_t，否则保留 H_t。
- **关键设计**：
  - 验证集 V 用于门控决策但不进入优化目标，测试集完全独立。
  - 权重更新始终保留；门控仅作用于 harness 提议的采纳。
  - 使用 Uni-Agent 连接执行、RL 训练与 harness 精炼。
  - 奖励函数：OfficeQA 为二值正确性（0/1），AutomationBench 为密集 partial credit。
- **调度控制**：每轮限制 RL 步数 B_W 和 harness 提议次数 B_H；tie 时保留 incumbent。

## 实验与结果
- **数据集**：
  - OfficeQA：84 train / 53 val / 109 test，binary accuracy。
  - AutomationBench：209 task 子集（HR 65、Marketing 72、Finance 72），split 为 80/58/71，dense partial credit。
- **基线方法**：Static agent、RL-only、Harness-only、SIA-style、WHALE-style、VACE（共 6 种）。
- **最强结果**：
  - **OfficeQA**：VACE 45.26%（± 三测均值），较 RL-only（38.83%）+6.43 pp，较 WHALE-style（40.67%）+4.59 pp，较 static agent（32.11%）+13.15 pp。
  - **AutomationBench**：VACE 75.19%（整体 partial credit），较 RL-only（66.10%）+9.09 pp，较 WHALE-style（68.25%）+6.95 pp，较 static agent（50.26%）+24.94 pp；其中 Finance 域达 78.25% 最高。
- **验证门控有效性**：44 次提议中 25 次接受、17 次拒绝、2 次 tie；OfficeQA 接受 7/12，AutomationBench 接受 18/32，说明近 40% 修订对已更新模型有害。
- **SIA-style vs VACE**：AutomationBench 上 VACE 75.19% vs SIA 65.35%（+9.85 pp），OfficeQA 上 45.26% vs 40.67%（+4.59 pp）。
- **非单调性**：验证分数在交替中非单调，RL 阶段可能暂时下降，但最终 retained pair 在 OfficeQA 从 30.28% 升至 47.17%，AutomationBench 从 53.08% 升至 81.42%。

## 相关工作脉络
1. **Agent Lightning**：将 agent 执行与 RL 训练分离，支持跨实现更新权重；VACE 在此基础上接入 harness 精炼模块。
2. **HarnessEvolve**：基于轨迹诊断与参考引导错误分析提出修订；VACE 直接复用其 harness optimizer S，并引入外门控。
3. **SIA (Self-improving AI)**：先 harness 优化再固定训练；VACE 改为交替+门控，避免 harness 过早固定错失后续模型能力提升带来的新优化空间。
4. **WHALE**：同时交替优化权重与 harness，但无 outer validation gate；VACE 证明 gate 带来 4.59/6.95 pp 额外增益。
5. **Co-Harness**：交替 failure-driven 修订与 supervised 训练；VACE 使用 RL 而非 SFT，且明确区分训练/验证/测试集。
6. **GRPO/DAPO**：底层 RL 算法；VACE 在其上叠加 harness 层，构成 co-evolution loop。

## 局限性与未来方向
- **单一模型规模**：仅测试 Qwen3.5-9B，未见更小/更大模型或跨架构泛化。
- **数据集范围有限**：仅两个 benchmark（文档推理 + 工作流），未覆盖 code agent、multi-modal agent 等。
- **计算开销**：每轮增加 harness 提议与验证比较，增加 wall-clock 成本；论文未给出效率分析。
- **固定交替调度**：每轮严格执行 W→H 顺序；未来可探索 feedback-driven 动态调度（何时更新权重、何时更新 harness）。
- **验证集复用偏差**：多次使用同一验证集进行门控决策可能引入 adaptive selection effect（Dwork et al., 2015），需更长 horizon 验证 generalization。
- **harness 编辑范围**：仅聚焦 skills 与 execution guidance，tool semantics、environment transition rules 保持固定。

## 研究启发与可借鉴点
1. **Validation-gated co-evolution 通用框架**：任何涉及"离散修订"（prompt/skill/pipeline）+ "连续参数训练"的场景均可套用此模式，先提议再 gate，避免有害修订污染后续训练。
2. **Trajectory reuse 节省采样成本**：用同一批 T_t 既做 RL 梯度更新又做 harness 诊断，无需额外 rollouts；适合 rollout 昂贵的 online RL 场景。
3. **Same-checkpoint 交叉评估**：比较 H_t 与 H'_t 时固定 W_{t+1}，可精准量化修订对当前模型的因果效应；未来可在多个 checkpoint 上做 cross-evaluation 绘制兼容性热力图。
4. **非单调轨迹可视化**：论文 Phase-level 曲线 + A/T/R 统计直观呈现 co-evolution 动态；可借用于其他迭代优化方法的诊断报告。
5. **与强化学习社区对接**：使用 GRPO+DAPO 开源组件 + Uni-Agent 框架，技术栈易于复现；可与本团队已有的 agent 评测 pipeline 快速集成。

## 关键术语表
- **Agent harness (H)**：围绕 LLM 的执行系统，包括指令、技能库、工具接口、执行规则等；决定模型能力如何被使用。
- **Agentic RL**：以 agent 与环境交互轨迹为样本、用强化学习直接优化模型权重的训练范式（如 GRPO）。
- **Validation-gated**：在采纳 harness 修订前，用当前最新权重在验证集上严格比较 incumbent 与候选，仅接受 Δ>0 的方案。
- **Checkpoint-specific**：门控评估时固定使用刚训练完的权重 W_{t+1}，保证比较针对实际将使用该 harness 的模型版本。
- **WHALE-style**：交替优化权重与 harness 但无外层验证门控的调度；本文用以证明 gate 的增量价值。
- **SIA-style**：先完成 harness-only 优化选出最佳 harness，再固定其上进行 RL 训练的调度。
- **Phase-level validation trajectory**：每轮交替后记录的验证分数序列，展示 co-evolution 的动态进展。
- **A/T/R 统计**：Accepted/Tied/Rejected 比例，反映每次 harness 提议的成功率。

## 可复现要素
- **数据集**：OfficeQA（Databricks AI Research，2025）、AutomationBench（Shepard & Salimans，2026）；论文使用自定义分区但未公开划分代码。
- **代码**：依赖 Uni-Agent（GitHub: verl-project/uni-agent）；VACE 本身代码未单独声明开源。
- **权重**：基础模型 Qwen3.5-9B（Qwen Team，2026）。
- **关键超参**：RL 算法为 GRPO + DAPO-style 过采样/零 advantage 组过滤；奖励为 outcome reward（OfficeQA 二值、AutomationBench dense partial credit）；具体 B_W、B_H 值论文未列出（见 Appendix A Table 5 仅给符号）。
- **评估**：每方法取验证集最高分 checkpoint，报告 3 次独立测试均值 ± 标准差。
