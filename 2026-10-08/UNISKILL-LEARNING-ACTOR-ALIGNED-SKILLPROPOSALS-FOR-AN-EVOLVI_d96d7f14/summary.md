---
title: "UNISKILL-LEARNING-ACTOR-ALIGNED-SKILLPROPOSALS-FOR-AN-EVOLVI"
source: https://arxiv.org/pdf/2610.10164v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:26:36"
field: "LLM Agent 技能自进化"
keywords: ["agent skill evolution", "contrastive action feedback", "actor-aligned skill proposal", "shared-policy co-evolution", "skill-edit regularization", "ALFWorld", "WebShop"]
innovations: ["用固定成功/失败轨迹的条件 log-likelihood 对比替代 rollout 级技能评估，得到 actor 对齐信号 R_align", "共享策略同时扮演 actor 与 skill proposer，使策略与技能库联合梯度更新", "Skill-Edit Support Regularization 防止 ADD/UPDATE/NO EDIT 三种编辑操作在训练中坍缩到单一类型"]
benchmarks: ["ALFWorld (success rate)", "WebShop (task score & success rate)"]
---

# 论文速读：UNISKILL — 学习 Actor-对齐的技能提议，驱动演进式策略

## 一句话总结

UniSkill 提出了一种**共享策略**（同一组参数 $\pi_\theta$）同时扮演"执行 actor"和"技能提议者"的联合训练框架，使 actor 与技能库能够协同演进。其核心创新是用**对比式行动反馈**（在已收集的成功/失败轨迹上测量替换技能前后的 log-likelihood 差距）替代每次提议都需额外环境 rollouts 的昂贵评估，从而在 ALFWorld 达到 **98.4%**、WebShop 达到 **84.7%** 成功率的同时保持训练稳定。

---

## 研究问题与动机

1. **Skill 效用难以与 actor 改进解耦。** Actor 在持续学习后，用"后续被重用次数"给技能提议打分，会把技能收益与 actor 自身进步混在一起，产生 credit assignment 混淆。
2. **直接评估成本过高。** Evolving-RL 等前置评估方法需要对每条候选技能做额外的 actor rollout，随训练步数线性膨胀；在 actor 尚未更新时评估，又会错过"skill 真正被 actor 使用"时的行为信号。
3. **已有技能提取多为冻结 LLM + 提示调用**（如 Trace2Skill、ReasoningBank），没有把技能生成纳入策略梯度训练，无法与 actor 联合优化。
4. **即便联合训练（Complementary RL、SkillRL）**，提议层面的反馈也主要依赖任务结果预测或轨迹级 reward，很少直接度量"这条技能是否让当前 actor 的行动分布更接近成功轨迹"。

---

## 核心贡献（创新点）

1. **共享策略的 actor–proposer 联合框架。** 同一个 $\pi_\theta$ 既生成环境动作、又生成技能编辑操作（NO EDIT / ADD / UPDATE）与技能内容，策略与技能库同步演进。与 SkillRL/Complementary RL 的本质区别在于：**actor 与 proposer 共享权重、共享一次梯度步**，而不是两阶段或双模型独立优化。
2. **对比式行动反馈（contrastive action feedback）替代 rollout 级评估。** 用已成功与已失败轨迹上、替换技能前后 actor log-likelihood 的差距 $R_{\text{align}}=\Delta^+-\Delta^-$ 作为技能提议的reward；不再为每条提议触发新 rollout。与 Evolving-RL 的本质区别在于：**用固定轨迹做 counterfactual 估计，而非对每条提议重跑环境**。
3. **Skill-Edit Support Regularization 防止编辑操作坍缩。** 当 $R_{\text{align}}$ 为负时可能把某种编辑操作（如 UPDATE）压到几乎不采样；支持正则项对低于 $p_{\min}$ 的操作概率施加软惩罚，保持三种操作均有探索机会。与简单均匀采样的本质区别在于：**只压制过低概率、不强制均匀分布**，允许比例随训练自适应变化。
4. **双层技能入库门槛（$\Delta^+>0$ 且 $R_{\text{align}}>0$）。** 仅有对齐加分还不够，必须确保成功轨迹的 log-likelihood 也提升；以此过滤"只在失败轨迹上拉大差距"的虚假正分。与单层阈值方法的本质区别在于：**同时满足绝对增益与相对增益两个条件**。

---

## 方法详解

### 3.1 形式化设定

- 技能库 $S$，共享策略 $\pi_\theta$ 承担两个角色：
  - **Actor**：在任务 $q$、检索技能 $s\in S$、历史 $H_t$ 下采样动作 $a_t$。
  - **Skill Proposer**：在已完成轨迹 $\tau$ 与同任务反向结果轨迹 $\tau_{\text{opposite}}$ 上，输出 $(\text{op},\tilde{s})$，其中 $\text{op}\in\{\text{NO EDIT, ADD, UPDATE}\}$。

### 3.2 Actor 学习（GRPO）

对每个任务采样 $G$ 条轨迹，定义组内优势：
$$
\widehat{A}_j = \frac{R_j-\mu_q}{\sigma_q+\epsilon}
$$
actor 损失采用带 clip 的 PPO-style 目标 + KL 正则（式 4）：
$$
\mathcal{L}_{\text{actor}}(\theta) = -\mathbb{E}\Big[\frac{1}{G}\sum_j\frac{1}{L_j}\sum_\ell \min(\rho_\theta \widehat{A}_j,\text{clip}(\rho_\theta,1-\varepsilon,1+\varepsilon)\widehat{A}_j)\Big]+\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})
$$

### 3.3 Actor-对齐的技能提议学习

1. **参考轨迹抽取**（式 5）：从同组成功集 $\mathcal{T}^+_q$ 与失败集 $\mathcal{T}^-_q$ 各抽一条，排除 proposer 输入本身的 $\tau,\tau_{\text{opposite}}$。
2. **条件 log-likelihood**（式 6）：
   $$
   J(\tau;s)=\frac{1}{T}\sum_{t=1}^T\frac{1}{|a_t|}\log\pi_\theta(a_t\mid q,s,H_t)
   $$
3. **对比对齐奖励**（式 7–8）：
   $$
   \Delta^\pm=J(\tau^\pm;\tilde{s})-J(\tau^\pm;s),\qquad R_{\text{align}}=\Delta^+-\Delta^-
   $$
   $R_{\text{align}}>0$ 表示候选技能在成功/失败轨迹上拉开了 actor 行动分布的差距。
4. **Skill Critic 判断**：格式合法性 $r_{\text{fmt}}$、编辑操作适当性 $r_{\text{op}}$、内容是否轨迹可支撑 $r_{\text{skill}}$（式 9–10），由冻结 DeepSeek-V4-Pro 充当 critic。
5. **提议总奖励**（式 11）：$r_{\text{prop}}=r_{\text{fmt}}+r_{\text{op}}+r_{\text{skill}}$，做全局批归一化得 $\widehat{A}_{\text{prop}}$。
6. **Proposer 损失**（式 12）：REINFORCE++ 形式，token 级 clip-PPO 目标 + KL。
7. **支持正则**（式 14）：
   $$
   \mathcal{L}_{\text{sup}}(\theta)=\mathbb{E}_x\Big[\frac{1}{|\mathcal{O}(s)|}\sum_{\text{op}}\big[\max(0,\log p_{\min}-\log p_\theta(\text{op}\mid x))\big]^2\Big]
   $$
   最终提议目标（式 15）：
   $$
   \mathcal{L}_{\text{proposer}}=\mathcal{L}_{\text{R++}}+\lambda_{\text{sup}}\mathcal{L}_{\text{sup}}
   $$

### 3.4 联合共进化

总损失（式 16）：
$$
\mathcal{L}_{\text{UniSkill}}=\mathcal{L}_{\text{actor}}+\lambda_{\text{prop}}\mathcal{L}_{\text{proposer}}
$$
入库条件（严格）：$\Delta^+>0$ 且 $R_{\text{align}}>0$，并通过 critic 两项检查；UPDATE 取最高 $R_{\text{align}}$ 的那条。

---

## 实验与结果

- **基准**：ALFWorld（文本式家务导航+操作）、WebShop（模拟电商搜索购买）。
- **基线**：训练-free（ReAct/Reflexion/ExpeL）、RL-only（PPO/RLOO/GRPO/GiGPO）、技能增强 RL（EvolveR/SkillRL/Complementary RL/RetroAgent/Skill1/Evolving-RL）。全部基于 Qwen2.5-7B-Instruct。
- **实现**：共享策略 Qwen2.5-7B-Instruct + AdamW，lr=$1\times10^{-6}$，weight decay=0.01，clip $\varepsilon{=}0.2$；skill critic 冻结 DeepSeek-V4-Pro，检索用 Qwen3-Embedding-0.6B（top-1）；$\eta=0.05$、$p_{\min}=0.1$、$\lambda_{\text{prop}}{=}0.5$、$\lambda_{\text{sup}}{=}0.01$。
- **主结果（Table 1）**：
  - ALFWorld：UniSkill **98.4±0.8%**，优于 Skill1（97.5%）0.9 pp、优于 GiGPO（90.8%）7.6 pp、优于 Evolving-RL（93.0%）5.4 pp。
  - WebShop：task score **90.5±1.4**、success **84.7±0.5%**，优于 SkillRL（72.7%）12.0 pp。
- **训练动态（Fig 1）**：Evolving-RL 早期强增后崩塌；UniSkill 在 250 步内稳步上升，技能库早期快速增长、后期以少量 ADD/UPDATE 持续精炼。
- **跨模型尺寸（Fig 4）**：Qwen2.5-3B-Instruct 上 UniSkill-3B 后期仍继续提升并超过 GRPO-7B，说明 co-evolution 在更小 backbone 上依然有效。
- **效率（Fig 3）**：Evolving-RL 的 per-step 时间被 rollout 级评估主导；UniSkill 用固定轨迹计算 $R_{\text{align}}$，总时间更低。

---

## 相关工作脉络

1. **经验复用型提示提取**（ReasoningBank、Dynamic Cheatsheet、ACE、Trace2Skill）：冻结 LLM 在推理时调用外部记忆，不更新参数，无法与 actor 联合优化。
2. **Skill 增强的 RL**（SkillRL、EvolveR）：把技能库与策略一起训练，但技能提议的 credit 依赖任务结果预测，未度量对当前 actor 行动分布的影响。
3. **Credit 分配问题**（RetroAgent、Skill1、MASA）：前者奖励"正确结果预测"，后者奖励"相对于最高历史技能的蒸馏"，均不衡量"候选技能替换检索技能后 actor 行动 gap 的变化"。
4. **Complementary RL**：用后期 skill 重用的 outcomes 训 proposer，但 actor 在 skill 被 reused 前已被更新，导致 credit 混淆。
5. **Evolving-RL**：前置评估候选技能——对每条提议做额外 rollout，时间开销大；本文在相同实现框架（VERL-AGENT）复现时发现其 100 步后崩塌，与作者公开说明一致。
6. **SkillGraph / ReSkill**：前者建模 skill 间图结构，后者把 skill 创建与 policy 优化统一，但均未使用"成功/失败轨迹上的条件 log-likelihood 对比"作为对齐信号。

---

## 局限性与未来方向

1. **$R_{\text{align}}$ 只是代理指标**。Q2 分析显示正 $R_{\text{align}}$ 提案中仍有约 24.5%（step 25）/ 15.6%（step 75）在实际 rollout 中产生负增益；对齐信号在聚合上有效、但不保证单条提案必然提升。
2. **Critic 虽对模型选择鲁棒（D.2），但仍是外部冻结 LLM**；成本虽小于 rollout 级评估，但在更长轨迹/更大 token budget 下仍会放大。
3. **仅评估 ALFWorld 与 WebShop 两类 bench**，未覆盖代码生成、多轮对话、开放域长程任务等更广泛的 agent 场景。
4. **技能库增长依赖 ADD/UPDATE 的累积**，长期可能出现冗余或冲突技能；论文讨论了偶尔的 UPDATE 持续发生，但没给出"技能合并/删除/版本回滚"的机制。
5. **支持正则 $p_{\min}=0.1$ 是启发式设定**（基于 $G=8$ 时 50% 至少出现一次的边界近似），未做系统敏感性分析。
6. **未来方向**：可探索无 critic 的自动对齐信号、跨任务的技能复用 transfer、以及更大的多模态 agent 环境。

---

## 研究启发与可借鉴点

1. **用固定轨迹做 counterfactual 评估替代在线 rollout**，是降低 skill-evaluation 成本的一般性思路；凡是需要"评估某策略变更对当前 actor 的影响"的场景（如 prompt 版本、工具调用顺序、检索模板）都可借鉴 $J(\tau;s)$ 的形式。
2. **双门槛入库（绝对增益+相对增益）**：$\Delta^+>0$ 且 $R_{\text{align}}>0$ 的组合能有效过滤"只在失败轨迹上拉大差距"的假正例，类似思想可迁移到任何"基于对比 reward 的内存/策略更新"系统。
3. **支持正则而非均匀强制**：对过低概率操作施加软惩罚（式 14）既保留探索又不牺牲自适应比例；在 multi-action 生成（如操作序列、规划步骤）里比 entropy bonus 更温和。
4. **共享策略的 actor–proposer 架构**避免了双模型梯度不一致问题，同时让 proposer 直接"感知"actor 当前能力边界；这对任何需要"提议者评估自身提议在当前策略下是否可用"的系统都很有参考价值。
5. **跨尺寸有效性**（3B 也能 co-evolve）说明该框架不依赖超大参数，便于在资源受限团队中复现与二次开发。

---

## 关键术语表

- **Actor-aligned skill proposal**：提议者生成的、能被当前 actor 实际执行的技能编辑操作与内容。
- **Contrastive action feedback / $R_{\text{align}}$**：用成功与失败轨迹上替换技能前后的条件 log-likelihood 差距之差，衡量提议对齐度。
- **Skill-Edit Support Regularization**：对低于阈值 $p_{\min}$ 的操作概率施加软惩罚，防止编辑操作在训练中坍缩到单一类型。
- **Skillbank**：持续演进的静态技能存储，每个 skill 对应一条可检索的文本化行为指导。
- **NO EDIT / ADD / UPDATE**：三种技能编辑操作；分别表示不修改、新增独立技能、修订已有检索技能。
- **Skill Critic**：冻结的大模型，负责判断编辑操作是否合理、内容是否被轨迹支撑。
- **Group Relative Policy Optimization (GRPO)**：本文 actor 训练采用的组内相对优势 PPO 变体。
- **REINFORCE++**：全局批归一化 advantage 的 REINFORCE 风格优化器，用于 proposer 训练。

---

## 可复现要素

- **数据集**：ALFWorld（valid seen / valid unseen 均已公开）、WebShop（公开模拟环境）；论文未声明自有数据。
- **代码**：已在 GitHub 开源（https://github.com/LimOkii/UniSKill），基于 VERL / VERL-AGENT 框架。
- **权重**：共享策略起点为 Qwen2.5-7B-Instruct 公开权重；critic 为冻结 DeepSeek-V4-Pro；检索用 Qwen3-Embedding-0.6B。
- **关键超参**：lr=$1\times10^{-6}$、weight decay=0.01、clip $\varepsilon{=}0.2$、$\eta{=}0.05$、$p_{\min}{=}0.1$、$\lambda_{\text{prop}}{=}0.5$、$\lambda_{\text{sup}}{=}0.01$、$G{=}8$、250 步训练；详见 Appendix B Table 3。
- **复现状态**：作者声明计划公开实现与配置文件；Evolving-RL 基线已在 VERL-AGENT 下复现并与原作者澄清的稳定性结论一致。

---
