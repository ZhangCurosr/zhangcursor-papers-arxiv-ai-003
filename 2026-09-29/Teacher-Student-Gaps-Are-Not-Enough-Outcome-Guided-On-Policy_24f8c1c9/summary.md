---
title: "Teacher-Student-Gaps-Are-Not-Enough-Outcome-Guided-On-Policy"
source: https://arxiv.org/pdf/2609.35319v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:13:29"
field: "多轮自主代理的强化学习与蒸馏"
keywords: ["on-policy distillation", "multi-turn agents", "knowledge distillation", "autonomous agents", "supervision weighting", "outcome-guided training"]
innovations: ["发现教师-学生gap与指导收益的不匹配现象并量化分析", "提出基于配对轨迹结果校准的OG-OPD蒸馏框架", "轨迹相对权重与增量式outcome-based增强机制"]
benchmarks: ["ALFWorld", "ScienceWorld", "WebShop"]
---

# 论文速读：Teacher-Student-Gaps-Are-Not-Enough: Outcome-Guided On-Policy Distillation for Multi-Turn Autonomous Agents

## 一句话总结
本文提出OG-OPD（Outcome-Guided On-Policy Distillation），通过配对学生轨迹的最终任务结果校准教师监督权重，解决多轮自主代理蒸馏中"教师-学生分布差异大≠指导有价值"的监督-收益不匹配问题，在ALFWorld、ScienceWorld、WebShop上显著优于基线方法。

## 研究问题与动机
1. **监督-收益不匹配（Supervision-Benefit Mismatch）**：现有方法假设教师-学生token级分布差异越大越需要干预，但实证发现大gap可能是良性（benign），小gap可能关键（outcome-critical）。
2. **局部gap无法预测全局收益**：当前轮次的teacher-student gap仅反映本地差异，但教师指导的实际收益取决于当前学生采纳教师建议后能否完成剩余任务。
3. **多轮交互的误差累积**：单轮动作影响后续观察和状态，错误可能在连续轮次中累积放大，需考虑长期任务结果而非仅当前token差异。
4. **教师行动未必优于学生**：教师偏好的action可能导致学生进入无法完成任务的状态，而学生原action虽与教师不同仍可能成功。

## 核心贡献（创新点）
1. **发现监督-收益不匹配现象**：在ScienceWorld和WebShop上量化分析显示，大gap决策中教师有害（teacher harm）占比10.8%超过教师拯救（teacher rescue）的8.8%，推翻"gap越大越需指导"的直觉。
2. **提出OG-OPD框架**：构建轨迹相对权重（trajectory-relative weighting）与基于结果的校准（outcome-based calibration）相结合的新蒸馏方法，仅在学生原轨迹上增强监督而非引入教师响应作为额外目标。
3. **配对学生延续评估机制**：通过从同一状态出发分别执行学生原action和教师action，比较两者在相同冻结学生策略下的最终任务结果，提供决策依据。
4. **广泛验证与一致提升**：在三个基准（ALFWorld、ScienceWorld、WebShop）和两种师生配置下均取得最高成功率，相对vanilla OPD提升3.6–17.7个百分点。
5. **效率与性能平衡**：在WebShop上Qwen3-1.7B学生达到45.7% SR（vs vanilla OPD的28.0%），训练成本仅增加约42–48%。

## 方法详解
**1. 轨迹相对权重（Trajectory-Relative Weighting）**
- 定义第k轮gap幅度：$\chi_k = \frac{1}{|\mathcal{M}_k(\zeta)|}\sum_{j \in \mathcal{M}_k(\zeta)} |\psi_{k,j}|$，其中$\psi_{k,j}$为token级teacher监督信号
- 以首个有效轮$k_0$为参考，计算对数幅度$\nu_k = \log(\varepsilon_0 + \chi_k)$及其变化$v_k = \nu_k - \nu_{k-1}$
- 预校准权重：$\beta_k = \min\{\beta_{\max}, \exp(\nu_{k_0} - \nu_k)\}$，gap高于参考时降权、低于参考时升权（上限$\beta_{\max}=1.2$）

**2. 候选轮选择（Candidate Selection）**
- 候选集：$\mathcal{K}(\zeta) = \{k \in \mathcal{H}(\zeta) | \nu_k \geq \vartheta_\nu \wedge [k=0 \vee (v_k > 0 \wedge v_k \geq \vartheta_v)]\}$
- 阈值$\vartheta_\nu, \vartheta_v$为批量分位数（$q_\nu=0.5, q_v=0.6$）
- 每轨迹最多检查2个候选，仅选中第一个 replay-valid 且教师action合法不同的轮次

**3. 基于结果的校准（Outcome-Based Calibration）**
- 恢复状态$s_{k^*}$和历史$c_{k^*}$，生成配对延续：$\tilde{\zeta}^S = \mathcal{U}(s_{k^*}, c_{k^*}, z_{k^*}^S; p_\circ, \varsigma)$，$\tilde{\zeta}^T = \mathcal{U}(s_{k^*}, c_{k^*}, z_{k^*}^T; p_\circ, \varsigma)$
- 计算结果差异：$\Delta_\zeta = S(\tilde{\zeta}^T) - S(\tilde{\zeta}^S) \in \{-1, 0, 1\}$
- 仅当$\Delta_\zeta = 1$（学生原action失败、教师action成功）时增强权重：$\gamma_\zeta = [\beta_\uparrow - \beta_{k^*}]_+$
- 校准后权重：$\omega_k = \beta_k + \gamma_\zeta$（$k=k^*$时），否则$\omega_k = \beta_k$

**4. 训练目标**
- 最终损失：$\mathcal{L}_{\text{OG-OPD}}(\theta) = \frac{1}{Z}\sum_{\zeta}\sum_{k \in \mathcal{V}(\zeta)} sg[\omega_k] Q_k(\theta;\zeta)$
- 梯度增量仅来自选中的候选轮：$\nabla_\theta \mathcal{L}_{\text{OG-OPD}} - \nabla_\theta \mathcal{L}_{\text{base}} = \frac{1}{Z}\sum_{\zeta \in \mathcal{D}_{\text{pair}}} sg[\gamma_\zeta] \nabla_\theta Q_{k^*}(\theta;\zeta)$
- 教师响应和配对延续仅用于校准，不作为额外蒸馏目标

## 实验与结果
**数据集与配置**
- ALFWorld（ Seen/Unseen/Hard）、ScienceWorld、WebShop三个基准
- 师生配置1：Qwen3-32B teacher → Qwen3-1.7B student
- 师生配置2：Qwen3-8B-RL teacher（GiGPO训练）→ Qwen3-4B student
- 训练步数200，3个随机种子取均值

**主要结果（Table 1, Qwen3-32B→1.7B）**
| 方法 | ALFWorld Overall SR | ScienceWorld SR | WebShop SR |
|------|---------------------|-----------------|------------|
| Vanilla OPD | 20.1% | 13.0% | 28.0% |
| FutureBridge-OPD（最强基线）| 21.4% | 12.8% | 38.7% |
| **OG-OPD** | **24.2%** | **16.6%** | **45.7%** |

- 相对vanilla OPD提升：ALFWorld +4.1pp、ScienceWorld +3.6pp、WebShop +17.7pp
- 相对最强基线FutureBridge-OPD提升：WebShop +7.0pp

**其他指标（Table 3消融）**
- 去除outcome-based calibration后WebShop SR从45.7%降至32.0%（-13.7pp）
- 去除trajectory-relative weighting后WebShop SR从45.7%降至33.3%（-12.4pp）
- 随机采样候选轮vs gap-based选择：WebShop SR从45.7%降至37.7%（-8.0pp）
- 更高task score和更少interaction rounds同步改善

**训练效率（Table 6）**
- ScienceWorld每步成本：1.42× vanilla OPD（223.58s vs 156.97s）
- WebShop每步成本：1.48× vanilla OPD（99.70s vs 67.21s）
- 实际upweight占比：<1%的recorded turns获得额外权重

## 相关工作脉络
1. **On-Policy Distillation for Agents**：Vanilla OPD（Agarwal et al., 2024）在学员自身轨迹上提供dense教师监督；TCOD（Wang et al., 2026）通过时间课程调节rollout深度；Guided-OPD（Li et al., 2026）交错教师与学生轮次。
2. **Selective Supervision and Reweighting**：TIP（Xu et al., 2026）用熵和gap调整token级监督强度；SOD（Zhong et al., 2026）提出轨迹相对权重规则但依赖相对gap幅度而非任务结果。
3. **FutureBridge-OPD**（Chen et al., 2026）：评估教师指导通过配对延续中教师偏好token比例提升，接受教师响应作为额外蒸馏目标——与OG-OPD的关键区别在于验证信号（token比例vs最终任务结果）和权重注入方式（额外目标vs原轨迹增权）。
4. **Chain-of-Thought Distillation**：Keypoint-based CoT distillation（Feng et al., 2024）学习token权重强调关键推理步骤，与本文的turn-level加权思路相关但应用场景不同。

## 局限性与未来方向
1. **配对延续的单次采样噪声**：仅用1对continuation判断是否增强权重，可能存在随机性；论文提到实证分析用5次匹配试验，但训练时仅用1次。
2. **计算开销增加**：需要状态回放、教师响应生成和配对执行，训练成本增加42–48%，在更大规模或更复杂环境中可能更显著。
3. **候选轮选择限制**：每轨迹最多检查2个候选、仅选中第一个valid轮次，可能遗漏其他有价值的干预点。
4. **固定策略延续假设**：配对延续使用冻结的当前学生策略$p_\circ$，未考虑学生参数更新后的动态变化。
5. **未探索教师only completion的利用**：发现20.6%的大gap决策中教师能完成但学生不能，但未设计专门机制让当前学生学习此类场景。
6. **仅验证三个benchmark**：缺乏在更多样化环境（如实时网页交互、物理仿真）上的泛化性检验。

## 研究启发与可借鉴点
1. **监督-收益解耦思想**：将"教师指导强度"与"实际收益"解耦，通过执行验证替代启发式规则，可迁移至RLHF、DPO等其他对齐场景中的reward建模。
2. **配对轨迹评估范式**：从同一状态出发的paired comparison设计简洁有效，可用于任何需要判断"干预是否有益"的离线/在线学习设置。
3. **轨迹相对归一化**：以首有效轮为参考的动态权重设计，避免大gap自动高权重的偏差，可推广至序列决策的任何蒸馏场景。
4. **增量式权重更新**：仅在校准为正时增强特定轮次权重，保留原有dense监督的同时精准聚焦，兼顾训练稳定性和效率。
5. **与团队成员结合点**：本工作揭示的"gap大≠指导有价值"现象与团队在agent reasoning中的观察一致，可将outcome-guided校准思路引入CoT蒸馏或multi-agent协作训练。

## 关键术语表
**On-Policy Distillation (OPD)**：在学员自身生成的轨迹上进行蒸馏，减少训练-推理分布偏移的核心方法。
**Supervision-Benefit Mismatch**：教师-学生token级gap大但实际指导收益低（甚至有害）的现象。
**Trajectory-Relative Weighting**：以轨迹内首个有效轮为参考的动态权重分配机制，平衡各轮监督强度。
**Outcome-Based Calibration**：通过配对学生延续的最终任务结果判断教师指导是否有益，并据此校准权重。
**Paired Student Continuations**：从同一状态分别执行学生原action和教师action，在相同冻结策略下生成的两条对比轨迹。
**Teacher Rescue / Harm**：教师action拯救了原本失败的任务（rescue）或导致原本成功的任务失败（harm）。
**Gap Magnitude ($\chi_k$)**：第k轮token级teacher-student分布差异的平均绝对值，用于定位潜在干预点。
**Augmented Weight ($\omega_k$)**：经outcome calibration后的最终轮次权重，仅在验证为正时高于预校准权重$\beta_k$。

## 可复现要素
- **数据集**：ALFWorld、ScienceWorld、WebShop（均为公开benchmark）
- **代码**：论文声明提供source code在supplementary material中，Section 3及Appendices A–C含完整算法、优化细节和实现流程
- **模型权重**：使用Qwen3-32B teacher和Qwen3-1.7B/4B student（需自行获取Qwen3模型）
- **关键超参**（Table 5）：
  - $\beta_{\max} = 1.2$（预校准权重上限）
  - $\beta_\uparrow = 1.5$（positive-outcome权重floor）
  - $q_\nu = 0.5, q_v = 0.6$（候选阈值分位数）
  - 每轨迹最多2个candidate checks、1个paired-evaluation position
- **训练配置**：8×A800 GPU，step 200，3个seed，AdamW lr=$10^{-6}$，weight decay=0.01
- **评估协议**：遵循TCOD和FutureBridge-OPD的environment配置、prompt template和evaluation collection
