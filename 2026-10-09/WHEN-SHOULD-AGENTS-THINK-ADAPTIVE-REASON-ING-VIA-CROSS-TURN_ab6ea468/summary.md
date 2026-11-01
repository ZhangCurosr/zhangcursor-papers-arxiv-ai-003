---
title: "WHEN-SHOULD-AGENTS-THINK-ADAPTIVE-REASON-ING-VIA-CROSS-TURN"
source: https://arxiv.org/pdf/2610.12061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:57:44"
field: "Agent 推理效率优化"
keywords: ["adaptive reasoning", "agent", "reinforcement learning", "reasoning efficiency", "cross-turn estimation", "SFT", "GRPO"]
innovations: ["提出跨轮推理持久性概念，发现参考动作似然下降可作为推理移除的轻量代理信号", "设计 LoGiC 算法实现渐进式推理覆盖检测，仅需似然计算无需重复生成", "构建 cover-aware agentic RL 框架，联合推理掩码与终止监督学习自适应行为"]
benchmarks: ["ScienceWorld", "WebShop", "AppWorld", "DeepSearch"]
---

# 论文速读：WHEN-SHOULD-AGENTS-THINK-ADAPTIVE-REASON-ING-VIA-CROSS-TURN

## 一句话总结
本文提出 RACE（Reasoning Adaptation through Cross-Turn Estimation），一种通过跨轮次估计来实现自适应推理的训练方法，利用参考动作似然下降作为轻量信号识别可移除推理步骤，将其融入 SFT 与 agentic RL，使 Agent 学会在何时推理、何时直接行动。

## 研究问题与动机
- **核心问题**：LLM-based Agent 通常在每个交互轮次都生成推理步骤，但早期推理可能持续支持后续多个轮次的动作，导致不必要的推理成本。
- **现有方法不足**：
  1. 已有自适应推理方法（如 DART、AdaptThink）仅在单轮独立评估推理必要性，未显式建模推理的跨轮持久性（cross-turn persistence）。
  2. 获取轮次级移除信号需要重复生成候选动作进行验证，随着轨迹增长，SFT 数据合成成本高昂；在 on-policy RL 中开销更大。
  3. 现有方法难以扩展到长轨迹场景，计算成本随推理步骤数量线性增长。
- **关键洞察**：移除推理后，后续参考动作的似然下降幅度与动作可恢复率高度相关，可作为轻量代理信号。

## 核心贡献（创新点）
1. **提出跨轮推理持久性概念**：首次系统性刻画 Agent 推理的跨轮影响范围，证明早期推理可支持后续多个轮次动作，并发现参考动作似然下降可作为行动可恢复性的轻量代理信号。
2. **设计 LoGiC 算法**：提出 Likelihood-Guided Progressive Reasoning Cover Detection 过程，仅通过批量 teacher-forced 评分即可识别可移除推理步骤，无需重复自回归解码。
3. **构建端到端训练框架 RACE**：将 LoGiC 推导的推理移除信号同时融入 SFT 和 agentic RL（GRPO），在 SFT 阶段合成自适应轨迹，在 RL 阶段通过推理掩码（masking）和推理终止监督（closure supervision）引导策略学习。
4. **实证验证高效性**：在 ScienceWorld、WebShop、AppWorld、DeepSearch 四个基准上，RACE-SFT-RL 相比标准 SFT+RL 减少推理 token 32.3%~80.2%，同时保持或提升任务成功率。

## 方法详解
**LoGiC（Likelihood-Guided Progressive Reasoning Cover Detection）核心流程**：

1. **Action-Likelihood Initialization**：对完整轨迹 $\tau$，初始化参考动作似然向量 $\ell^{(0)} = [\ell(a_1^*|x_1,r_1), \dots, \ell(a_T^*|x_T,r_T)]$。

2. **Progressive Reasoning Compression**：从第 2 轮开始依次评估每个推理步骤 $r_t$：
   - 构造移除 $r_t$ 后的候选轨迹，保留先前已接受的移除。
   - 计算候选似然向量 $\widetilde{\ell}^{(t)}$，其中 $j<t$ 的似然不变，$j=t$ 取当前压缩上下文下的似然，$j>t$ 在保留后续推理但移除 $r_t$ 的上下文中计算。

3. **Likelihood-Based Verification**：对每个受影响的参考动作 $a_j^*$（$j \in \{t, \dots, T\}$）计算似然下降 $d_j^{(t)} = \ell_j^{(t-1)} - \widetilde{\ell}_j^{(t)}$，当且仅当 $d_j^{(t)} \leq \epsilon$ 对所有 $j$ 成立时才接受移除 $r_t$，更新似然向量和压缩上下文。

**SFT 阶段**：使用 LoGiC 生成的自适应轨迹 $\tilde{\tau}$ 进行监督训练，损失 $\mathcal{L}_{\mathrm{SFT}}$ 仅作用于模型生成的推理和动作 token。

**Agentic RL 阶段（cover-aware）**：
- **推理掩码**：对成功轨迹中 LoGiC 标识的 SKIP 轮次，将推理 token 从 GRPO 目标中掩码，仅保留动作 token 用于任务优化。
- **推理终止监督**：引入 $\mathcal{L}_{\mathrm{close}} = -\frac{1}{\sum T_i}\sum_{i \in \mathcal{T}_{\mathrm{succ}}}\sum_{t \in S_i} \log \pi_\theta(e|x_i^{(t)})$，鼓励策略在 SKIP 轮次输出终止 token $e$。
- 最终 actor 目标：$\mathcal{L}_{\mathrm{actor}} = \mathcal{L}_{\mathrm{GRPO}} + \lambda \mathcal{L}_{\mathrm{close}}$。

## 实验与结果
**数据集**：ScienceWorld（科学推理）、WebShop（网络购物）、AppWorld（应用使用）、DeepSearch（知识密集型推理）。

**基线**：DeepSeek-V4-Pro、GLM-5.3、GPT-5.5、Claude-Opus-4.6、Kimi-K3、Qwen3.8-27B 等 ReAct 风格 Agent；Think Tool、ADaPT、Sparse ReAct、DART、AdaptThink 等推理效率方法。

**主要结果（Qwen3.5-9B  backbone）**：
| 基准 | RACE-SFT-RL Success | Avg Tok. | vs Qwen3.5-9B 提升 | vs RACE-SFT 提升 |
|------|---------------------|----------|---------------------|-------------------|
| ScienceWorld | 73.28 | 1,403 | +54.76 pts, -36.5% tokens | +30.23 pts, -28.6% tokens |
| WebShop | 53.40 | 535 | +37.80 pts, -80.3% tokens | +8.00 pts, -56.6% tokens |
| AppWorld | 65.13 | 1,900 | +18.98 pts, -39.4% tokens | +2.05 pts, -42.9% tokens |
| DeepSearch | 35.00 | 822 | +5.75 pts, -35.8% tokens | +6.75 pts, -55.7% tokens |

**消融实验关键发现**：
- 移除 RL 中的 LoGiC（仅 SFT+GRPO）导致推理 token 暴增：ScienceWorld 从 1,403 增至 2,573，WebShop 从 535 增至 2,670，AppWorld 从 1,900 增至 8,574。
- 移除推理掩码导致成功率下降：ScienceWorld 72.95→73.28，WebShop 46.00→53.40。
- RL 阶段显著提升 skip ratio（如 ScienceWorld 从 37.22% 升至 69.65%）。

**Scaling 分析**：在 Qwen3.5-2B/4B/9B 上验证，RACE 效果随模型规模增大而增强，跨尺度保持有效性。

## 相关工作脉络
1. **AdaptThink（Zhang et al., 2025）**：通过学习 thinking/non-thinking 模式选择实现自适应推理，但未显式建模推理的跨轮持久性；RACE 通过跨轮估计提供更细粒度的推理覆盖控制。
2. **DART（Lee et al., 2026a）**：基于多 draft 一致性进行训练无关的 adaptive routing，仅适用于 standalone query 场景；RACE 针对 multi-turn agent 交互设计，支持 SFT 和 RL 训练。
3. **Agent-omit（Ning et al., 2026）**：通过 agentic RL 学习省略推理和观察，侧重轨迹级过滤；RACE 关注轮次级推理必要性估计，利用似然信号替代生成验证。
4. **Sparse ReAct / Think Tool**：基于 prompting 的策略，无额外训练；RACE 通过端到端训练显式学习自适应行为。
5. **Agentic RL 工作（AgentGym-RL, RAGEN 等）**：聚焦任务性能优化和 credit assignment；本文首次将自适应推理行为整合进 agentic RL 框架。

## 局限性与未来方向
- **阈值敏感性**：LoGiC 依赖超参数 $\epsilon$（论文取 0.001），不同任务/模型可能需要调整，缺乏自适应阈值机制。
- **仅应用于成功轨迹**：RL 阶段只对成功 trajectory 应用 LoGiC，失败轨迹仍用标准 GRPO，可能错过失败路径中的推理效率优化机会。
- **静态覆盖假设**：LoGiC 基于固定参考动作计算似然，未考虑策略更新后动作分布的变化对覆盖判断的影响。
- **评估规模有限**：仅在四个基准上验证，跨域泛化能力待进一步检验。
- **未来方向**：探索动态阈值学习、将 LoGiC 扩展到失败轨迹、结合在线策略更新进行自适应覆盖检测。

## 研究启发与可借鉴点
1. **似然下降作为轻量信号**：用 teacher-forced 评分替代生成验证，大幅降低信号获取成本，可迁移至其他需要 turn-level 决策的 Agent 训练场景（如 tool call 选择、观察跳过）。
2. **推理覆盖（reasoning cover）概念**：将跨轮推理支持形式化为"覆盖"问题，为长期推理持久性建模提供新视角，可延伸至多步规划、memory management 等方向。
3. **Cover-aware RL 设计**：推理掩码+终止监督的组合策略，分离"不强化冗余推理"和"主动学会终止"两个学习目标，对改进其他离散决策的 RL 训练有参考价值。
4. **渐进式压缩算法**：LoGiC 的逐轮递推逻辑（接受移除后更新上下文再评估后续轮次）保证了一致性，类似思想可用于轨迹精简、摘要生成等任务。

## 关键术语表
**Cross-turn reasoning persistence**：早期生成的推理步骤可跨越多个交互轮次持续支持后续动作的现象。
**Reference-action recovery rate**：移除某轮推理后，通过多次采样计算参考动作被恢复的比例，用于衡量推理支持的可靠性。
**Reasoning cover**：轨迹中保留的推理步骤集合，确保所有动作在移除其他推理后仍可被支持。
**LoGiC（Likelihood-Guided Progressive Reasoning Cover Detection）**：基于参考动作似然下降的渐进式推理覆盖检测算法，用于识别可安全移除的推理步骤。
**Reasoning masking**：在 GRPO 训练中对 LoGiC 标识的 SKIP 轮次的推理 token 进行掩码，避免策略被冗余推理强化。
**Reasoning closure supervision**：对 SKIP 轮次施加的监督，强制策略在该位置输出终止 token $e$，直接引导推理结束行为。
**Agentic RL**：将强化学习应用于 Agent 任务训练，通过环境反馈优化策略，典型方法包括 GRPO、PPO 等。

## 可复现要素
- **数据集**：ScienceWorld（公开）、WebShop（公开）、AppWorld（公开）、DeepSearch（基于多个公开 QA 数据集构建）。
- **代码**：GitHub https://github.com/HututuAI/RACE（论文声明开源）。
- **权重**：使用 Qwen3.5-9B 作为 backbone（开源），SFT 和 RL checkpoint 论文未明确声明是否开源。
- **关键超参**：似然阈值 $\epsilon = 0.001$，closure loss 权重 $\lambda = 0.02$（ScienceWorld/WebShop/AppWorld）或 $0.01$（DeepSearch），SFT learning rate $1\times10^{-5}$（除 AppWorld 用 $5\times10^{-6}$），RL learning rate $1\times10^{-6}$，batch size 16，每实例采样 8 条轨迹。
