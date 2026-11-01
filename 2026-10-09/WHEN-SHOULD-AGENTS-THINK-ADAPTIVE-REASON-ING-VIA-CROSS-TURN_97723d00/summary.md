---
title: "WHEN-SHOULD-AGENTS-THINK-ADAPTIVE-REASON-ING-VIA-CROSS-TURN"
source: https://arxiv.org/pdf/2610.12061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:57:38"
field: "大语言模型智能体推理效率优化"
keywords: ["Adaptive Reasoning", "Agent", "Reinforcement Learning", "Reasoning Efficiency", "Cross-turn Estimation"]
innovations: ["提出LoGiC基于似然下降的跨轮次推理覆盖检测，避免重复生成验证", "将推理移除信号同时引入SFT与Agentic RL（推理遮蔽+闭室监督）"]
benchmarks: ["ScienceWorld", "WebShop", "AppWorld", "DeepSearch"]
---

# 论文速读：WHEN-SHOULD-AGENTS-THINK-ADAPTIVE-REASON-ING-VIA-CROSS-TURN ESTIMATION

## 一句话总结
提出了 RACE（Reasoning Adaptation through Cross-Turn Estimation）训练方法，通过跨轮次参考动作似然下降信号识别可安全移除的推理步骤，使 Agent 能自适应地判断"何时推理、何时直接行动"，在四个基准上显著降低推理 token 消耗的同时保持或提升任务性能。

## 研究问题与动机
- 现有 LLM Agent 通常在每个交互轮次前都显式生成推理（如 ReAct 范式），导致长轨迹推理成本高昂。
- 早期推理实际上具有**跨轮次持久性**：某些推理步骤可支持单轮后续行动，某些则可支撑多轮，因此并非每轮都需要新推理。
- 现有自适应推理方法（如逐轮比较有无推理的动作生成差异）未显式建模跨轮次推理影响力，且依赖重复生成验证，数据合成与 on-policy RL 阶段成本极高。
- 关键挑战：如何在**不重复生成候选动作**的前提下，高效估计已有推理对后续行动的充分性，从而决定是否需要新推理。

## 核心贡献（创新点）
1. **发现跨轮次推理似然代理信号**：证明移除额外推理后，后续参考动作的 token 对数似然下降与动作可恢复率高度相关，可作为轻量级跨轮次支持估计信号。
2. **提出 LoGiC（Likelihood-Guided Progressive Reasoning Cover Detection）**：一种渐进式推理覆盖检测方法，仅通过 teacher-forced 似然评分而非重复生成即可识别可安全移除的推理步骤，显著降低信号提取成本。
3. **将 LoGiC 信号同时引入 SFT 与 Agentic RL**：SFT 阶段构建自适应推理轨迹供模型学习"何时跳过推理"；RL 阶段通过推理遮蔽（masking）与推理闭室监督（closure supervision）直接引导策略优化。
4. **系统性实验验证**：在 ScienceWorld、WebShop、AppWorld、DeepSearch 四个代表性基准上，以 Qwen3.5-9B 为骨干，相比标准 SFT+GRPO 分别减少 32.3%、80.2%、46.4%、62.0% 推理 token，同时保持或提升成功率。

## 方法详解
**LoGiC 三阶段流程：**

1. **Action-Likelihood Initialization**：对完整轨迹 τ，将每步参考动作 a*_t 视为固定目标，在原始上下文 x_t 下计算 token 平均对数似然 ℓ(a*_t | x_t)，初始化似然向量 ℓ^(0)。

2. **Progressive Reasoning Compression**：从第 2 轮起逐轮评估是否可移除推理 r_t。保留先前已接受的移除，构建候选轨迹并重新计算当前及后续参考动作的似然向量 ℓ̃^(t)。若 j < t，似然沿用上一轮；若 j = t，在无 r_t 的压缩上下文 x^(t) 下计算；若 j > t，保留后续推理但移除 r_t，计算受影响动作的新似然。

3. **Likelihood-Based Verification**：对每个候选推理 r_t，计算受影响的当前及后续所有参考动作的似然下降 d_j^(t) = ℓ_j^(t−1) − ℓ̃_j^(t)。仅当对所有 j ∈ {t,...,T} 均满足 d_j^(t) ≤ ε 时，才接受移除 r_t，否则保留。接受移除时更新上下文 x^(t+1) 为 (x^(t), a*_t, o_t)；拒绝时保留 r_t，上下文更新为 (x^(t), r_t, a*_t, o_t)。最终保留的推理步骤构成"推理覆盖集"。

**SFT 阶段**：对全推理成功轨迹应用 LoGiC，得到自适应推理轨迹 τ̃ = LoGiC(τ)，在 τ̃ 的推理与动作 token 上执行标准监督损失 L_SFT。

**RL 阶段（Cover-Aware Agentic RL）**：
- **推理遮蔽（Masking）**：对成功轨迹，LoGiC 识别 SKIP 轮集合 S_i，将对应推理 token 从 GRPO 目标中屏蔽，仅保留动作 token 参与策略梯度优化。
- **推理闭室监督（Closure Supervision）**：对 LoGiC 验证的 SKIP 轮，显式施加终止推理的监督信号：L_close = −(1/ΣT_i) Σ Σ_{t∈S_i} log π_θ(e | x_i^(t))，其中 e 为推理终止 token。
- 最终 Actor 目标：L_actor = L_GRPO + λ·L_close。

## 实验与结果
**数据集**：ScienceWorld（科学推理）、WebShop（网络购物）、AppWorld（应用操控）、DeepSearch（知识密集型搜索）。

**骨干模型**：Qwen3.5-9B；SFT 数据由 DeepSeek-V4-Pro-0813 标注的成功轨迹经 LoGiC 转换得到；ε = 0.001，λ 取 0.01~0.02。

**主要结果（RACE-SFT-RL vs. Base Qwen3.5-9B）**：
| 基准 | 成功率提升 | 推理 token 减少 |
|---|---|---|
| ScienceWorld | +54.76 pt（18.52→73.28）| 36.5% |
| WebShop | +37.80 pt（15.60→53.40）| 80.3% |
| AppWorld | +18.98 pt（46.15→65.13）| 39.4% |
| DeepSearch | +5.75 pt（29.25→35.00）| 35.8% |

**RACE-SFT-RL vs. 标准 SFT+GRPO**：推理 token 分别减少 32.3%、80.2%、46.4%、62.0%，成功率差距在 ±1.4 pt 以内。

**关键对比**：RACE 以 9B 模型在多项基准上超越或接近 GPT-5.5（Medium）、Claude-Opus-4.6 等更大/闭源模型；相较同骨干的 AdaptThink 在 WebShop/AppWorld/DeepSearch 上使用更少 token 且性能更优。

**消融**：移除 RL 中的 LoGiC（仅 SFT+GRPO）推理 token 大幅反弹（如 ScienceWorld 从 1,403→2,573）；移除推理遮蔽则多基准成功率下降。

## 相关工作脉络
1. **AdaptThink（Zhang et al., 2025）**：让模型在每轮选择"推理/非推理"模式，通过奖励 bonus 鼓励非推理轮次。RACE 的区别在于显式建模跨轮次推理影响力而非逐轮独立决策，且使用似然下降这一轻量信号而非重复生成。
2. **DART（Lee et al., 2026a）**：基于多 draft 一致性进行推理路由，原设计用于单 query 场景。RACE 将其适配到多步 Agent 交互，并通过跨轮次效应评估提升准确性。
3. **Sparse ReAct / Think Tool**：基于 prompt 的策略，无训练信号。RACE 通过端到端 SFT+RL 学习自适应行为，可泛化到新任务。
4. **Agent-Omit（Ning et al., 2026）**：在 RL 中学习省略观察（observation omission）。RACE 聚焦于省略推理（reasoning omission），两者互补，均利用相似 RL 框架。
5. **传统 ReAct 范式（Yao et al., 2023）**：强制每轮 Think-Act-Observation。RACE 通过训练打破这一硬性约束，实现按需推理。

## 局限性与未来方向
- **ε 阈值超参敏感性**：似然下降阈值 ε 需人工设定，不同任务/模型可能需要调优；缺乏自适应阈值机制。
- **仅作用于成功轨迹**：RL 阶段 LoGiC 仅应用于成功 rollout，失败轨迹仍使用标准 GRPO，可能导致训练信号不对称。
- **离线 SFT 数据依赖**：SFT 阶段依赖高质量标注轨迹（文中使用 DeepSeek-V4-Pro 生成），限制了方法在低资源场景的适用性。
- **未评估对失败轨迹的影响**：自适应推理可能导致某些轮次跳过必要推理而引发错误，论文未深入分析此类风险。
- **未来方向**：探索无阈值自适应截止机制；将 LoGiC 信号延伸至失败轨迹的 credit assignment；适配多模态 Agent 与更长上下文场景。

## 研究启发与可借鉴点
1. **似然下降作为轻量验证信号**：用 teacher-forced 评分替代重复生成进行动作可恢复性估计，大幅降低数据合成与 RL 训练开销，可迁移至其他需要验证推理必要性的场景。
2. **推理覆盖（Reasoning Cover）概念**：将轨迹中的推理步骤视为"覆盖集"，允许某些步骤被安全移除而保持后续行动可行，这一抽象可用于设计更高效的轨迹压缩与蒸馏策略。
3. **RL 阶段的推理遮蔽与闭室监督**：在 GRPO 中屏蔽不必要推理 token、同时对 SKIP 轮施加终止监督，使任务奖励与推理效率目标协同优化，为推理效率 RL 训练提供了可复用的损失设计模板。
4. **渐进式贪心评估**：LoGiC 从第 2 轮逐轮决策并累积接受移除，避免了全局组合搜索的指数复杂度，这种贪心+回溯友好的设计易于工程实现。

## 关键术语表
**RACE（Reasoning Adaptation through Cross-Turn Estimation）**：论文提出的自适应推理训练框架，通过跨轮次似然估计识别可移除推理步骤。
**LoGiC（Likelihood-Guided Progressive Reasoning Cover Detection）**：基于参考动作似然下降的渐进式推理覆盖检测算法，用于提取推理移除信号。
**Cross-turn reasoning persistence**：跨轮次推理持久性，指早期生成的推理可支持后续多个交互轮次的行动。
**Reference-action recovery rate**：参考动作恢复率，衡量移除后续推理后原始动作仍可从压缩上下文中生成的比例。
**Reasoning cover**：推理覆盖集，轨迹中保留的足以支撑所有后续行动的最小推理步骤集合。
**Reasoning-closure supervision**：推理闭室监督，对 LoGiC 识别的 SKIP 轮施加终止推理 token 的显式监督信号。
**GRPO（Group Relative Policy Optimization）**：组相对策略优化，论文采用的 on-policy RL 优化器。

## 可复现要素
- **数据集**：ScienceWorld、WebShop、AppWorld、DeepSearch（各数据集均有公开版本，DeepSearch 基于七个 QA 数据集的组合）
- **代码开源**：GitHub https://github.com/HututuAI/RACE
- **权重/模型**：基于 Qwen3.5-9B，需自行获取权重
- **关键超参**：似然阈值 ε = 0.001；RL 闭室权重 λ = 0.02（ScienceWorld/WebShop/AppWorld）或 0.01（DeepSearch）；SFT 学习率 1e−5（AppWorld 用 5e−6）；RL 学习率 1e−6；batch size 16，每实例采样 8 轨迹；GRPO clip ratio 0.2
- **训练配置**：SFT 有效 batch 32（AppWorld 为 8），最大序列长度 65,536 tokens；RL 每 rollout batch 执行 1 次参数更新
