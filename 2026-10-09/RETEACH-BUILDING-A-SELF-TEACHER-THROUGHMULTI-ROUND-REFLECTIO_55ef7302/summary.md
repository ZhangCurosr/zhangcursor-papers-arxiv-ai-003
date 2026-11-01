---
title: "RETEACH-BUILDING-A-SELF-TEACHER-THROUGHMULTI-ROUND-REFLECTIO"
source: https://arxiv.org/pdf/2610.11529v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:19:38"
field: "语言模型自我改进与蒸馏"
keywords: ["self-distillation", "on-policy distillation", "reflective reasoning", "multi-round reflection", "mathematical reasoning", "chain-of-thought"]
innovations: ["多轮反思-重试构建无需参考信息的自我教师", "按修正结果分组的三类加权蒸馏损失", "在六基准上以零特权信息取得最高平均准确率"]
benchmarks: ["AIME 2024", "AIME 2025", "HMMT 2025", "SciKnowEval-Physics", "SciKnowEval-Material", "ToolAlpaca"]
---

# 论文速读：RETEACH-BUILDING-A-SELF-TEACHER-THROUGHMULTI-ROUND-REFLECTIO

## 一句话总结
论文提出 ReTeach，一种仅需自生成尝试和结果级验证的多轮反思与重试自我蒸馏框架，构建了更强的自我教师，在不依赖参考答案或外部反馈的情况下显著提升了大模型在数学推理、科学问答和工具使用任务上的单遍推理能力。

## 研究问题与动机
1. **RLVR 奖励稀疏**：可验证奖励仅提供二元信号，无法说明中间步骤的质量，导致训练效率低。
2. **OPD 依赖强外部分教师**：现有 on-policy 蒸馏受限于教师能力上限，限制了推理能力的进一步突破。
3. **参考答案可能不可用**：OPSD 等自我蒸馏方法依赖参考 CoT 或 ground truth 答案，但在许多实际场景中这类特权信息不存在。
4. **已有反思方法依赖额外信息**：RESD 使用持续 playbook 和缓存成功解，ReflectionCoder/HERO 需要执行反馈，本文探索仅靠自身尝试能否构建有效自我教师。

## 核心贡献（创新点）
1. **多轮反思-重试教师构建**：从零次正确答案的学生 rollout 出发，交替生成显式反思与重试，无需参考解答或外部诊断；与 OPSD/RLSD 依赖参考答案形成本质区别。
2. **结果感知的样本分类与加权**：按初始正确性、反思后是否修正成功将样本分为 A/B/C 三类并分别归一化加权，使未修正示例降权而非直接排除；区别于 RESD 对成败二分类的处理。
3. **六基准一致性优势**：在数学推理、科学 QA、工具使用共六个基准上取得最高平均准确率 65.50%，超越 GRPO +1.39pp、CEPO +1.35pp；五基准第一、一基准第二。

## 方法详解
**多轮反思-重试教师构建（§3.1）：**
- 对问题 $x$，学生首先生成 $y \sim \pi_\theta(\cdot|x)$，由验证器 $\mathcal{V}(y) \in \{0,1\}$ 判定。
- 若失败，教师从第 1 轮起迭代：基于历史尝试 $A_{k-1}=(y,y_1,\dots,y_{k-1})$ 生成显式反思 $\tau_k \sim \pi_{\theta_{tea}}(\cdot|x,A_{k-1})$，再生成新尝试 $y_k \sim \pi_{\theta_{tea}}(\cdot|x,A_{k-1},\tau_k)$。
- 反思需用 `<reflect></reflect>` 标签包裹；前轮反思不作为下一轮输入，仅当前轮重试。
- 终止于验证通过或达到最大轮次 $K$；教师上下文为 $c(x)=(A_{T-1},\tau_T)$。

**稳定 on-policy 蒸馏（§3.2）：**
- 学生在原始 rollout $y$ 的每个 token 位置 $t$ 上接收仅问题 $x$ 的输入；教师在同一 rollout 前缀上接收上下文 $c(x)$ 的输入，梯度停止。
- 目标为 JSD：$\mathcal{L}_{OPD}(x,y)=\frac{1}{|y|}\sum_t D_{JS}(p_{tea}^t,p_{stu}^t)$。
- 教师参数 EMA 更新：$\theta_{tea}\leftarrow\lambda\theta_{tea}+(1-\lambda)\theta$，兼顾稳定性与教师能力演化。

**结果感知三类加权（§3.2）：**
- A（初始正确，$\mathcal{V}(y)=1$）：师生同 prompt，教师仅作为时间滞后参考正则化。
- B（初始错误但反思后修正，$\mathcal{V}(y)=0,\mathcal{V}(y_T)=1$）：教师以 $c(x)=(A_{T-1},\tau_T)$ 为上下文提供纠正性蒸馏目标。
- C（始终未修正）：教师上下文存在但无已验证成功，监督可信度更低。
- 总损失：$\mathcal{L}_{distill}=w_A\mathcal{L}_A+w_B\mathcal{L}_B+w_C\mathcal{L}_C$，缺失类别不贡献梯度且其余类别不重新归一化。

## 实验与结果
**数据集与设置：**
- 基础模型 Qwen3-8B，8×H20 GPU，LoRA rank=64。
- 数学推理：OpenThoughts 子集（≤30k 题），评测 AIME 2024/2025、HMMT 2025（avg@12，开启 thinking mode）。
- 科学 QA：SciKnowEval Physics/Materials L3（within-dataset，avg@16，关闭 thinking mode）。
- 工具使用：ToolAlpaca（within-dataset，avg@16，关闭 thinking mode）。

**主要结果（Table 1）：**
- **跨数据集数学**：ReTeach 65.62% > GRPO 64.32% > CEPO 62.93%，在 AIME24 (78.61)、AIME25 (70.46)、HMMT25 (47.78) 三项均第一。
- **科学 QA**：Physics 64.24%（第一）、Material 69.17%（第二，CEPO 70.83 第一）。
- **工具使用**：62.72%（第一）。
- **总体平均**：ReTeach 65.50% 居首，分别超 GRPO +1.39pp、CEPO +1.35pp。

**消融要点：**
- 移除显式反思：SciKnowEval-Physics 准确率由 64.24% 降至 62.91%（-1.33pp）。
- 教师策略：EMA 最优；tracking teacher 导致熵坍缩与梯度爆炸（norm 峰值 25.27）；frozen teacher 稳定但无法吸收学生提升。
- 重试轮次 $K$：$K=2$ vs $K=1$ +0.70pp，$K=3$ vs $K=2$ +0.14pp（仅改善跨 seed 方差）；默认取 $K=2$ 平衡性能与成本。
- 未修正类别权重 $w_C$：Math/Science 取 0，Tool-Use 取 0.2 有益，0.5 则退化。

## 相关工作脉络
1. **OPSD (Zhao et al., 2026a)**：教师以参考 CoT 为特权上下文；ReTeach 不依赖任何参考解答，反思-重试全程只靠自生成尝试。
2. **RLSD (Yang et al., 2026)**：以 ground truth 答案条件化教师；ReTeach 仅用结果级验证器（0/1），无答案泄露。
3. **CEPO (Heakl et al., 2026)**：构建正负对比教师；ReTeach 无需对比，单次成功反思即提供单一更强的目标分布。
4. **RESD (Zhang et al., 2026)**：结合显式反思、持续 playbook 和缓存成功解；ReTeach 摒弃 cross-example 记忆与持久知识库。
5. **ReflectionCoder (Ren et al., 2025) / HERO (Liu et al., 2026)**：分别依赖代码执行反馈与环境观察；ReTeach 仅依赖 outcome-level verification，无需外部环境交互。
6. **SDPO (Hübotter et al., 2026)**：以同组成功 rollouts 为教师；ReTeach 通过反思重构上下文，避免直接复用其他 rollout 的成功轨迹。

## 局限性与未来方向
1. **重试轮次需人工设定**：$K$ 依赖启发式权衡，未探索自适应终止机制。
2. **仅在 8B 模型上验证**：未扩展到更大参数规模，Scaling Law 未知。
3. **未修正类别（C）的处理保守**：目前 Math/Science 直接丢弃，但 Tool-Use 中小权重有益，分类边界条件尚不清晰。
4. **推理阶段仍为单遍**：反思-重试开销完全转移至训练期，但未评估训练时的 token 预算与延迟成本。
5. **仅依赖单一验证器**：对难以形式化验证的任务（如开放域写作）适用性待检验。

## 研究启发与可借鉴点
1. **反思作为自我修正的显式中间层**：将反思过程结构化（`<reflect>` 标签、强制命名错误类型）可有效提升重试质量，可迁移至 agent 规划、代码生成等需要迭代纠错的场景。
2. **结果-条件分层加权**：按修正结果将样本分组并独立归一化加权，既保留已正确行为又引入纠正性信号，思路可推广到任意存在多状态标签的监督蒸馏任务。
3. **EMA 教师对蒸馏稳定性的关键作用**：对比 tracking/frozen/EMA 三者的熵与梯度范数变化为诊断蒸馏不收敛提供了可量化的监控指标。
4. **对未修正样本保留小权重而非直接丢弃**：在 Tool-Use 上 $w_C=0.2$ 有效，提示部分错误轨迹仍含可迁移信息，后续可在更多任务上系统研究“噪声监督”的边际收益。

## 关键术语表
**On-Policy Distillation (OPD)**：在当前策略采样样本上计算教师-学生分布匹配损失的蒸馏范式，消除分布偏移。  
**Self-Distillation**：模型以自身为教师进行蒸馏，教师通过扩展上下文获得信息优势而非额外能力。  
**Outcome-Level Verification**：仅对最终输出给出二元正确/错误判定的验证机制，不依赖过程监督或参考答案。  
**Exponential Moving Average (EMA) Teacher**：按 $\theta_{tea}\leftarrow\lambda\theta_{tea}+(1-\lambda)\theta$ 平滑更新教师参数，平衡稳定性与适应性。  
**Category-Aware Weighting**：按初始正确性、反思修正成功率将样本划分为 A/B/C 三类并分配独立权重。  
**Chain-of-Thought (CoT)**：引导模型逐步推导中间步骤以提升复杂推理任务表现的 prompting/训练技术。

## 可复现要素
- **数据集**：OpenThoughts 数学子集、SciKnowEval (Physics/Materials L3)、ToolAlpaca；均为公开数据集（论文附录 A 提供获取说明）。
- **代码/权重**：论文未明确声明开源仓库与权重；建议向作者邮箱查询。
- **关键超参**：LoRA rank=64, α=128; 有效 batch size=32; teacher temperature=0.7; student temperature=1.0, top-p=0.95, top-k=20; EMA decay λ=0.9999; 训练步数=400; 最大重试轮次 K=2（默认）; 三类权重 w_A=0.5, w_B=0.5, w_C=0（Math/Science）或 0.2（Tool-Use）。
