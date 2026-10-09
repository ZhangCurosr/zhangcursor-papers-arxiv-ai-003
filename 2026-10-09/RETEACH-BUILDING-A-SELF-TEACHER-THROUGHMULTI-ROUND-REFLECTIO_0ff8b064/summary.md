---
title: "RETEACH-BUILDING-A-SELF-TEACHER-THROUGHMULTI-ROUND-REFLECTIO"
source: https://arxiv.org/pdf/2610.11529v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:19:47"
field: "大语言模型自我蒸馏与推理增强"
keywords: ["self-distillation", "on-policy distillation", "reflection", "reinforcement learning with verifiable rewards", "chain-of-thought reasoning", "self-improvement"]
innovations: ["通过多轮反思-重试从自我生成失败尝试构建自教师，无需参考答案或外部诊断", "结果感知三类加权策略区分初始正确/反思修正/未解决 rollout 并独立加权蒸馏损失", "在六项基准中五项第一，以无特权信息方式超越依赖参考答案的基线方法"]
benchmarks: ["AIME 2024", "AIME 2025", "HMMT 2025", "SciKnowEval-Physics", "SciKnowEval-Material", "ToolAlpaca"]
---

# 论文速读：RETEACH-BUILDING-A-SELF-TEACHER-THROUGHMULTI-ROUND-REFLECTIO

## 一句话总结
论文提出 **ReTeach**，一种通过多轮反思与重试构建自教师的自我蒸馏框架，仅利用模型自身生成的尝试历史和结果级验证器，无需任何参考答案或外部诊断信息；在数学推理、科学问答和工具使用六个基准上，平均准确率 65.50%，超越所有对比基线。

## 研究问题与动机
1. **核心问题**：如何在没有外部特权信息（参考答案、参考解链、外部诊断反馈）的情况下，让模型自主构建一个比自己更强的"自教师"，以提供密集的单 token 蒸馏监督。
2. **现有 RLVR 局限**：GRPO 等方法仅依赖二元结果奖励（scalar reward），无法指出中间推理步骤的错误，导致训练低效。
3. **现有 OPD/OSPD 局限**：Lu & Lab (2025)、OPSD (Zhao et al., 2026a) 等方法需要更强的外部教师，或要求教师可见参考答案/参考解链作为特权上下文，限制了实际可复用性。
4. **现有反思方法局限**：RESD、ReflectionCoder、HERO 等反思类方法往往结合外部工具反馈、永久记忆或地面真值答案，无法仅在结果级验证下工作。

## 核心贡献（创新点）
1. **多轮反思-重试教师构建**：从学生失败 rollout 出发，教师交替进行显式反思（`<reflect>`标签包裹）与重试，直至验证通过或达到最大轮数；与 OPSD/RLSD 等依赖参考答案条件化的本质区别在于，本方法完全不依赖任何外部特权信息，反思内容完全由自我生成的失败尝试驱动。
2. **结果感知选择与类别级加权策略**：将 rollout 分为初始正确（A）、反思修正成功（B）和未解决（C）三类，分别赋予独立权重并做类别内归一化；与 RESD 等仅区分成功/失败的二元处理相比，细粒度地区分了"有验证成功证据"与"无验证成功证据"两类失败样本，避免不可靠监督噪声。
3. **EMA 教师稳定机制**：通过指数移动平均（EMA）更新教师参数以同时保持目标稳定性和追踪学生能力增长；实验表明直接同步学生会导致熵坍缩与梯度爆炸，而冻结教师则丧失适应能力。
4. **跨六项基准的最优性能**：在 GRPO、OPSD、RLSD、CEPO、RESD 等全部对比方法中，ReTeach 在五个基准上排名第一、剩余一个排名第二，且全部对比方法中使用了外部特权信息的也不优于 ReTeach 的整体表现。

## 方法详解

**流程概述**：见 Figure 2，两个阶段：(1) 多轮反思-重试构建教师上下文；(2) 从该教师进行稳定的在线蒸馏。

**组件 1：多轮反思-重试构建教师**

对每个问题 $x$，学生采样 $y \sim \pi_\theta(\cdot|x)$，经验证器 $\mathcal{V}(y)\in\{0,1\}$ 判断。若正确，设 $c(x)=\varnothing$；否则教师启动多轮反思：

- 第 $k$ 轮历史记录 $A_{k-1}=(y, y_1, \ldots, y_{k-1})$。
- 先生成显式反思 $\tau_k \sim \pi_{\theta_{\text{tea}}}(\cdot|x, A_{k-1})$，$\tau_k$ 用 `<reflect></reflect>` 包裹，要求诊断每步失败类型（重复禁止答案/错误公式/计算错误/遗漏条件）并重新推导。
- 再在反思基础上重试 $y_k \sim \pi_{\theta_{\text{tea}}}(\cdot|x, A_{k-1}, \tau_k)$。
- 教师最终上下文 $c(x)=(A_{T-1}, \tau_T)$，仅含历史失败尝试与最终轮反思，不含参考答案或外部反馈。

**组件 2：稳定在线蒸馏（OPD）**

对学生 rollout $y$ 的每一 token 位置 $t$：
$$p_{\text{stu}}^t = \pi_\theta(\cdot|x, y_{<t}),\quad p_{\text{tea}}^t = \text{sg}[\pi_{\theta_{\text{tea}}}(\cdot|x, c(x), y_{<t})]$$
最小化 token 级 Jensen–Shannon 散度：
$$\mathcal{L}_{\text{OPD}}(x,y) = \frac{1}{|y|}\sum_{t=1}^{|y|} D_{\text{JS}}(p_{\text{tea}}^t, p_{\text{stu}}^t)$$

**EMA 更新**：$\theta_{\text{tea}} \leftarrow \lambda\theta_{\text{tea}}+(1-\lambda)\theta$，默认 $\lambda=0.9999$，教师 thinking mode 始终开启，学生禁用。

**结果感知类别加权**：
- **Category A**（初始正确）：$c(x)=\varnothing$，教师与学生提示相同，EMA 提供正则化约束。
- **Category B**（反思修正成功）：$c(x)=(A_{T-1}, \tau_T)$，成功验证提供监督效用证据。
- **Category C**（未解决）：同上 $c(x)$ 但无成功验证，可靠性较低。

整体蒸馏损失：$\mathcal{L}_{\text{distill}}=w_A\mathcal{L}_A + w_B\mathcal{L}_B + w_C\mathcal{L}_C$，各类别内平均后独立加权，不重归一化。

## 实验与结果

**模型与数据集**：基于 Qwen3-8B，LoRA 训练，8×H20 GPU。

| 任务 | 数据集 | 训练数据 |
|------|--------|---------|
| 数学推理（跨数据集） | AIME 2024/2025、HMMT 2025 | OpenThoughts 数学子集，≤30,000 样本 |
| 科学问答（数据集内保留） | SciKnowEval L3 Physics & Materials | 无参考 CoT |
| 工具使用（数据集内保留） | ToolAlpaca | 无参考解链 |

**主要结果（Table 1）**：

| 方法 | Math Avg@12 | Science Avg@16 | ToolUse Avg@16 | **Total Avg** |
|------|------------|---------------|----------------|--------------|
| GRPO | 64.32 | 65.43 | 60.85 | 64.11 |
| OPSD | 63.76 | — | — | 63.76 |
| RLSD | 63.90 | 65.81 | 60.08 | 63.90 |
| CEPO | 64.15 | 66.72 | 61.86 | 64.15 |
| RESD | 62.72 | 62.72 | 60.94 | 62.72 |
| **ReTeach** | **65.62** | **66.71** | **62.72** | **65.50** |

- ReTeach 在 **AIME24 (78.61%)**、**AIME25 (70.46%)**、**HMMT25 (47.78%)**、**SciKnowEval-Physics (64.24%)**、**ToolUse-ToolAlpaca (62.72%)** 五个基准上排名第一；仅在 **SciKnowEval-Material (69.17%)** 排第二（CEPO 70.83% 第一）。
- 相比最强基线 CEPO，平均提升 **+1.35 pp**；相比 GRPO 提升 **+1.39 pp**。

**消融结果（Section 4.3）**：
- 去除显式反思（仅重试无反思）：SciKnowEval-Physics 从 64.24% 降至 62.91%（**-1.33 pp**）。
- EMA vs 同步教师 vs 冻结教师：同步教师导致熵从 0.129 坍缩至 0.036 nats、梯度范数从 0.29 升至 25.27；冻结教师虽稳定但无法适应；EMA 梯度最稳定（0.08–0.14），熵最高且上升（0.21→0.24）。
- 最大轮数 K：K=1→2 提升 0.70 pp，K=2→3 提升 0.14 pp（边际递减），但 K=3 跨种子方差显著降低；最终选用 K=2 平衡成本与收益。
- 未解决样本权重 $w_C$：Math/Science 排除（$w_C=0$）最优；ToolUse 设 $w_C=0.2$ 略有提升，$w_C=0.5$ 反而退化。

## 相关工作脉络
1. **OPSD（Zhao et al., 2026a）**：将教师条件化于参考解链，提供信息不对称；ReTeach 与之本质区别是不依赖任何参考解或答案，仅通过自我反思生成教师上下文。
2. **RLSD（Yang et al., 2026）**：利用验证器接受的 rollout 构建正教师，需参考答案条件化；ReTeach 的教师在结果级验证下仍可通过多轮反思产生有效上下文。
3. **SDPO（Hübotter et al., 2026）**：在二元奖励下从同组成功 rollout 构建教师；ReTeach 更进一步，不依赖同组成功样本，而是主动从失败中迭代生成。
4. **CEPO（Heakl et al., 2026）**：构建对比式 token 级权重（正/负教师各条件于正确/错误响应）；ReTeach 不提供外部条件信息给教师，通过多轮反思替代对比信号。
5. **RESD（Zhang et al., 2026）**：结合持久化 playbook、缓存成功 rollout 和地面真值答案构建教师；ReTeach 同样使用反思驱动的上下文，但不依赖任何外部记忆或真值反馈。
6. **ReflectionCoder（Ren et al., 2025）/ HERO（Liu et al., 2026）**：依赖代码执行反馈或环境观察；ReTeach 完全不依赖外部工具或环境交互，仅用结果级验证器。

## 局限性与未来方向
1. **重试轮数上限需经验调参**：K 增大可提升纠正率并降低跨种子方差，但教师推理成本线性增长（论文取 K=2 而非 K=3 出于效率考虑）。
2. **未解决样本（Category C）处理依赖任务**：Math/Science 上排除最优，ToolUse 上适度加权有收益，任务依赖的权重设计缺乏统一原则。
3. **反思质量受模型本身能力制约**：若模型反思能力不足，多轮反思-重试可能反复陷入同类错误，当前消融仅验证了有无反思，未深入分析反思质量。
4. **教师 thinking mode 始终开启带来训练推理开销**：学生推理时无 thinking，但训练时教师需生成大量反思 token，增加训练时间。
5. **未来方向**：可探索自适应轮数停止策略、反思质量自动评估、将本框架迁移至代码生成/agent 规划等更广泛领域。

## 研究启发与可借鉴点
1. **"反思即上下文"范式**：将反思 `<reflect>` 作为教师上下文的关键组成部分，而非附加信号；可直接迁移至其他自蒸馏场景，用反思序列替代参考答案作为特权信息。
2. **结果感知三类加权策略**：将 rollout 按"初始正确/反思修正/未解决"三分并独立加权，相比二元区分保留了更多细粒度监督信号；可复用于其他 OPD 变体。
3. **EMA 教师更新在反思蒸馏中的必要性**：消融显示同步教师会导致熵坍缩和梯度爆炸（10 倍增长），冻结教师丧失适应能力；建议在类似"教师能力随训练提升"的场景中默认采用 EMA。
4. **工具使用任务的 Category C 有益性**：ToolUse 上 $w_C=0.2$ 优于 $w_C=0$，提示在开放式输出任务中，即使未通过验证的上下文也可能包含部分有用信息；对函数调用类任务可考虑保留弱监督。
5. **与本团队方向结合机会**：可将 ReTeach 的多轮反思-重试框架接入团队已有的 RLVR pipeline，在不增加外部标注成本的前提下替代参考答案条件化；也可将反思诊断模块用于生成过程奖励模型的训练数据。

## 关键术语表
- **On-Policy Distillation (OPD)**：在策略自身采样分布上进行的蒸馏，学生 rollout 与教师条件分布对齐，消除分布偏移并产生 per-token 优势信号。
- **Self-Teacher**：由同一模型通过某种特权上下文（如参考解、反思历史）构建的更强监督源，推理时不暴露该上下文。
- **Reflect-and-Retry**：从失败尝试出发，先生成显式反思诊断错误原因，再基于反思重试，循环直至成功或预算耗尽。
- **Outcome-Level Verification**：仅依赖最终答案的布尔验证器（$\mathcal{V}(y)\in\{0,1\}$），不访问中间推理步骤或参考答案。
- **EMA Teacher**：通过指数移动平均 $\theta_{\text{tea}}\leftarrow\lambda\theta_{\text{tea}}+(1-\lambda)\theta$ 更新的教师参数，兼顾稳定性和适应性。
- **Jensen–Shannon Divergence (JSD)**：对称 KL 散度的变体，用于衡量教师与学生 token 分布差异，此处作为蒸馏损失核心。
- **Category A/B/C**：按 rollout 初始正确性与反思修正结果划分的三类样本，分别用于不同类型的蒸馏加权。
- **Cross-Dataset Evaluation**：在训练集之外的独立基准上评估泛化能力（如用 OpenThoughts 训练、在 AIME/HMMT 上测试）。

## 可复现要素
- **数据集**：OpenThoughts（数学，≤30,000 条）、SciKnowEval L3（Physics & Materials）、ToolAlpaca；论文未明确公开数据集链接，但给出了数据来源名称。
- **代码/权重**：论文未明确声明代码开源，未提及权重发布。
- **关键超参**：LoRA rank=64、$\alpha=128$、targets=`q_proj/k_proj/v_proj/o_proj`（Math）或 `gate_proj/up_proj/down_proj`（Science/ToolUse）；Effective Batch Size=32；训练步数=400；Teacher Temperature=0.7；EMA $\lambda=0.9999$；Max Grad Norm=0.1；最大反思轮数 K=2（默认）；$w_A=w_B=0.5$，$w_C=0$（Math/Science）或 $0.2$（ToolUse）；vLLM 1.1 推理引擎。
