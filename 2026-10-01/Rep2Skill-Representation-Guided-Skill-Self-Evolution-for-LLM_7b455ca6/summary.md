---
title: "Rep2Skill-Representation-Guided-Skill-Self-Evolution-for-LLM"
source: https://arxiv.org/pdf/2609.39149v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:44:13"
field: "LLM Agent 技能自我进化"
keywords: ["agent skill evolution", "representation attribution", "Neural CDE", "self-evolving LLM", "text-space skill optimization", "group-wise rollout"]
innovations: ["首次将内部表示轨迹引入文本技能演化闭环，通过 Neural CDE 建模成功动力学实现轮次级归因", "表示→文本口译机制：将预测误差翻译为结构化诊断并经程序化精确引用校验保证可落地", "组-wise 相对归因：利用同任务多 rollout 的组内差异作为结构化证据替代单次轨迹启发式"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：Rep2Skill: Representation-Guided Skill Self-Evolution for LLM Agents

## 一句话总结
论文提出 REP2SKILL，将 LLM 智能体的内部表示轨迹引入技能优化闭环：通过建模成功执行的表示动力学、定位偏离关键交互轮次，并将其"翻译"为文本化反馈，从而指导可复用文本技能的自我迭代进化。实验在 ALFWorld 和 WebShop 两个 Agent 基准上，REP2SKILL 均稳定超越纯文本方法。

## 研究问题与动机
- **核心问题**：现有 Agent 技能演化方法完全依赖可观测的文本轨迹和任务级结果，导致失败归因模糊、反馈稀疏延迟，且对有限能力的优化器负担极重。
- **表征信号被忽视**：智能体内部表示（hidden states）蕴含丰富的过程性信息（工具使用决策、执行偏差、情感/任务相关状态），但从未进入技能优化回路。
- **单轨迹归因不稳定**：同一任务实例+同一技能下采样的 rollout 行为差异巨大（turn 5 动作分歧率达 82.7%），仅凭单次轨迹修订技能容易受偶然行为干扰。
- **自进化设置下的能力约束**：本文聚焦"同一 LLM 同时充当执行者与优化器"的自进化场景，无更强外部模型辅助，更凸显细粒度归因的必要性。

## 核心贡献（创新点）
1. **首次将内部表示轨迹纳入文本技能演化回路**——与所有现有方法（Trace2Skill/EvoSkill/SkillOpt）仅利用外部可观测轨迹不同，REP2SKILL 把表示空间作为互补信号源。
2. **提出组-wise 表示归因框架**——对同一任务的多 rollout 组内比较，用 Neural CDE 建模成功轨迹的表示动力学并计算预测误差，实现轮次级细粒度归因（AUROC 0.592 vs 文本 0.494）。
3. **表征→文本口译机制（Representation-to-Text Verbalization）**——用分析器 $M_a$ 将表示预测误差与轨迹上下文结合，生成结构化诊断（5 类：suspect_rule / rule_conflict / execution_deviation / missing_rule / insufficient_evidence），并经由程序化校验（引用必须精确匹配原文）保证反馈可落地。
4. **自进化设定下的一致性提升**——在 Qwen3-4B / Qwen3.5-9B、ALFWorld / WebShop 共 4 组实验中均达最优，且在弱优化器场景（SkillOpt 反而降分）中尤为关键；AB 实验证明三个组件（组比较、表示归因、语义口译）缺一不可。

## 方法详解
REP2SKILL 迭代流程 $s_{n} \to s_{n+1}$ 分三步：

**（1）组 wise rollout 采样**
对任务 $x$ 和当前技能 $s_n$，采样 $K=4$ 条 rollout：$\mathcal{G}(x,s_n)=\{\tau^{(k)}\}_{k=1}^K$，其中 $\tau^{(k)}=(o_1,a_1,\dots,o_T,a_T)$。

**（2）表示归因（Representation Attribution）**
- 从冻结 LLM 固定层（decoder layer 24）抽取每轮隐藏状态 $\mathbf{h}_t^{(k)}\in\mathbb{R}^d$，经 PCA 投影至 64 维 $\mathbf{z}_t^{(k)}$（仅用成功 rollout 拟合）。
- 用 Neural CDE（Kidger et al., 2020）建模成功轨迹动力学：
$$\mathbf{y}(u) = \mathbf{y}(u_2) + \int_{u_2}^u f_\phi(\mathbf{y}(v)) \cdot [\gamma_\phi(\dot{\mathbf{X}}(v))\odot \dot{\mathbf{X}}(v)] dv, \quad \widehat{\mathbf{h}}_{t+1}=g_\phi(\mathbf{y}(u_t))$$
以 Hermite 插值构造控制路径 $\mathbf{X}(u)$，Euler 求解，Softplus MLP 作向量场。
- 训练目标：$\min\frac{1}{T-1}\sum_{t=2}^T\|\mathbf{z}_t-\widehat{\mathbf{z}}_t\|_2^2$（AdamW，lr=1e-4，weight decay=1e-5，最多 300 epoch，早停 20）。
- 推理时计算轮次预测误差 $c_t^{(k)}=\|\mathbf{z}_t^{(k)}-\widehat{\mathbf{z}}_t^{(k)}\|_2^2$，取 Top-$K_c$（$K_c=3$）为关键轮次集合 $\mathcal{C}^{(k)}$。

**（3）表示→文本口译 + 技能修订**
- 分析器 $M_a$（同 LLM）接收 $(s_n, x, \tau^{(k)}, \{(t,c_t^{(k)})|t\in\mathcal{C}^{(k)}\})$，生成最多 3 条诊断，每条含：决策上下文、相关技能原文引用、轨迹证据、反证、适用条件、修改假设、是否建议变更。
- 程序化校验：每条引用必须是 $s_n$ 的精确子串（suspect_rule/execution_deviation 引用 1 处，rule_conflict 引用 2 处不同），每条证据必须指向有效步骤并逐字引用。
- 优化器 $\mathcal{O}$ 在候选技能集合中选验证集上最佳者：仅在 $J_{\mathcal{D}_{val}}(\tilde{s})>J_{\mathcal{D}_{val}}(s_n)$ 时接受（validation-gated），否则保留 $s_n$。edit budget 固定 $L=4$。

**总损失**：无端到端可微损失，表示建模阶段用 MSE，技能修订靠离散文本优化+验证门控。

## 实验与结果
**基准**：ALFWorld（39 train / 18 selection / 134 test）、WebShop（50/20/100）；模型 Qwen3-4B、Qwen3.5-9B；3 次独立 seed（42/43/44）。

| 模型 | 方法 | ALFWorld Succ% ↑ | WebShop Score ↑ | WebShop Succ% ↑ |
|---|---|---|---|---|
| Qwen3-4B | No Skill | 22.64 | 41.86 | 13.67 |
| | Trace2Skill | 47.26 | 45.68 | 13.00 |
| | EvoSkill | 43.03 | 49.48 | 13.33 |
| | SkillOpt | 46.51 | 39.04 | 11.67 |
| | **Rep2Skill** | **52.24** | **49.79** | **16.33** |
| Qwen3.5-9B | No Skill | 28.61 | 50.07 | 18.33 |
| | Trace2Skill | 52.74 | 24.77 | 12.00 |
| | EvoSkill | 58.96 | 33.10 | 15.67 |
| | SkillOpt | 62.19 | 52.12 | 18.67 |
| | **Rep2Skill** | **69.90** | **53.56** | **23.33** |

- ALFWorld 相对最强基线提升 **+4.98pp（4B）** / **+7.77pp（9B）**；WebShop 相对最强基线提升 **+4.66pp** / **+1.44pp 分数**。
- 归因评估（GPT-6 Astra 标注正/中/负进展）：Text 基线 AUROC 0.494，纯 Rep. 0.592，Text+Rep 达 **0.838 / 0.876 AUPRC / P@Top-3=88.22%**。
- AB 实验：去掉 verbalization ↓4.48、去掉 analyzer ↓5.22、去掉组比较 ↓5.47（均以 Qwen3-4B/ALFWorld 计）。
- 组内多样性：turn 5 动作分歧 82.7%，98.5% rollout 有唯一 action prefix，支撑组比较必要性。

## 相关工作脉络
1. **Trace2Skill（Ni et al., 2026）**：聚合多轨迹经验生成交叉任务可迁移技能文档；仅用文本，REP2SKILL 额外引入表示动力学信号使归因从粗粒度文本推断升级为轮次级表示偏差检测。
2. **EvoSkill（Alzubi et al., 2026）**：将失败 rollout 归因于缺失能力并修订模块化技能；同样依赖外部可观测轨迹，REP2SKILL 在归因阶段用表示误差替代文本启发式。
3. **SkillOpt（Yang et al., 2026a）**：把技能视为可训练文本状态、通过有界编辑+验证选择更新；REP2SKILL 沿用其 text-space 优化骨架，但以表示归因为先导信号输入。
4. **Activation Oracles / Natural Language Autoencoders（Karvonen et al., 2026; Fraser-Taliente et al., 2026）**：将激活向量直接译为自然语言；REP2SKILL 走"归因→结构化诊断→程序化校验"路线，不依赖端到端激活可微译码。
5. **Representation Engineering（Zou et al., 2023; Tigges et al., 2023）**：揭示 LLM 表示中的线性语义结构；本文将其从静态线性探测推进到动态轨迹建模（Neural CDE）与 Agent 技能闭环结合。
6. **Emotion2skill（Lin et al., 2026）**：用内部情绪信号引导技能选择；与本文同属"表示→技能"思路，但 Emotion2skill 侧重情绪维度选择而非轨迹动力学归因。

## 局限性与未来方向
- **需白盒访问表示**：依赖解码层 hidden states，黑盒模型无法直接使用；可用 proxy model 代理或蒸馏表示信号作为替代路径。
- **模型族局限**：实验仅限 Qwen 系列（full/hybrid attention 各一），跨架构泛化性待验证。
- **额外开销**：约需 50 条成功 trajectory 离线训练 Neural CDE，加上线上每组 4 rollout 的表示抽取与 PCA 投影，引入 modest 额外计算。
- **Neural CDE 超参敏感**：hidden size、层数、Euler 步长等未做系统消融，潜在改进空间。
- **表示层选择**：固定用 layer 24，未讨论不同层/不同头对归因质量的差异。

## 研究启发与可借鉴点
1. **Neural CDE 轨迹建模可迁移**：任何需要"从成功样本中学习动态基准、再检测偏离"的场景（如强化学习 exploration 异常检测、时序行为建模）均可复用该范式。
2. **组-wise 比较 + 相对归因**：与 group-relative optimization（DeepSeekMath, Shao et al., 2024）思想一脉相承，可将"组内差异"作为结构化证据而非噪声，适用于任何采样有方差的过程。
3. **程序化校验保证文本反馈落地**：要求技能引用为精确子串、证据必须逐字匹配轨迹步骤，这一设计显著降低 LLM hallucination 风险，可推广到任何"表示→文本"接口。
4. **表示-文本双空间多样性验证**：同时报告 text-space（TF-IDF、Levenshtein）和 rep-space（cosine）的组内分歧曲线，为"是否需要多 rollout"提供量化依据，实验设计值得借鉴。
5. **self-evolution 设定更贴近部署现实**：同一模型做执行/分析/优化，避免了"强模型指导弱模型"的性能泄漏争议，结论更干净，可作为后续工作的默认对照设定。

## 关键术语表
**Self-evolution（自进化）**：同一 LLM 同时充当任务执行者与技能优化器、无需外部强模型的迭代技能改进范式。
**Representation attribution（表示归因）**：利用内部隐藏状态对轨迹轮次进行细粒度失败相关度打分的过程。
**Neural CDE（神经控制微分方程）**：用连续时间 ODE 建模离散时间序列动力学的框架，本文用于学习成功执行的表示演化规律。
**Group-wise rollout（组-wise rollout）**：对同一任务实例在相同技能下采样多条轨迹形成组，以组内差异作为归因证据。
**Validation-gated update（验证门控更新）**：候选技能仅在验证集表现严格优于当前技能时才被接受，否则回退。
**Representation-to-text verbalization（表示→文本口译）**：将表示预测误差信号结合轨迹上下文翻译为结构化文本诊断的过程。
**PCA latent projection（PCA 潜投影）**：对高维 hidden state 做降维至 64 维并标准化，用于提升轨迹建模效率与泛化。
**Edit budget L（编辑预算）**：每轮技能更新允许的最大文本编辑数，恒定设为 4，充当文本学习率。

## 可复现要素
- **数据集**：ALFWorld（39/18/134 split）、WebShop（50/20/100 split）；官方基准，数据固定种子，论文声明已公开。
- **代码/权重**：论文未提供开源链接（截至发表）；模型为 Qwen3-4B / Qwen3.5-9B 开源权重，vLLM bfloat16 本地推理。
- **关键超参**：group size K=4、critical turns $K_c=3$、edit budget L=4、Neural CDE hidden m=64、PCA 输出 64 维、提取层 layer 24、训练 lr=1e-4、weight decay=1e-5、max 300 epoch、早停 20、actor temp 0.4/0.7、optimizer/analyzer temp 0.6/0.7、top-p=0.8、top-k=20、analyzer max 2048 tokens、optimizer max 4096/8000 tokens。
- **随机 seed**：42, 43, 44；rollout seed 由 (run_seed, task, rollout_idx) 确定性派生。
- **离线表示训练数据**：约 50 条成功轨迹（来自与训练/验证/测试集不相交的任务），用于拟合 PCA 与 Neural CDE。
