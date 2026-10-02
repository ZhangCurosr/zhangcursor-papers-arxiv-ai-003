---
title: "SKILLCOME-GROUP-CONTRAST-SKILL-OPTIMIZA-TION-WITH-DUAL-MEMOR"
source: https://arxiv.org/pdf/2609.37128v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:21:11"
field: "大语言模型技能演化与Agent自我改进"
keywords: ["Skill Evolution", "Group Contrast", "Dual Memory", "LLM Agents", "Non-parametric Optimization", "Trajectory Analysis"]
innovations: ["提出组内对比优化（group contrast）作为可靠优化信号", "引入双记忆系统跨步骤积累泛化模式", "在6个基准5个模型上验证，平均超次优基线5.69个百分点"]
benchmarks: ["SpreadsheetBench", "SearchQA", "LiveMathematicianBench", "ALFWorld", "OfficeQA", "DocVQA"]
---

# 论文速读：SKILLCOME-GROUP-CONTRAST-SKILL-OPTIMIZA-TION-WITH-DUAL-MEMOR

## 一句话总结
SkillCome 是一种基于组对比优化的技能演化方法，通过为每个问题生成多个轨迹（grouped rollouts）并对比成功与失败路径，结合双记忆系统跨步骤积累证据，从而精确识别关键行为分歧并提供更可靠、更具泛化性的技能优化方向。

## 研究问题与动机
1. **单轨迹分析的局限性**：现有方法（如 SkillOpt、Trace2Skill）通常为每个问题只生成单个轨迹，难以判断失败轨迹中哪些操作导致失败，或成功轨迹中哪些操作值得纳入技能。
2. **局部批次噪声敏感**：现有方法主要依赖一个局部批次的轨迹进行分析，有价值的技能模式可能分散在不同步骤中，单个批次数据不足导致优化器忽略这些重复出现的模式。
3. **缺乏跨步骤证据积累**：由于优化器上下文长度有限，单步分析易过拟合特定问题，无法区分"偶发噪声"与"可泛化模式"。

## 核心贡献（创新点）
1. **组 rollout + 组内对比优化**：为每个问题生成 n 个轨迹，在组内对比成功与失败轨迹的行为差异，精确定位导致结果分叉的关键分歧点；与现有单轨迹方法的本质区别在于引入"同题对比"作为优化信号源。
2. **双记忆系统（Dual Memory）**：维护失败记忆（failure memory）和对比记忆（contrast memory），分别记录跨步骤积累的历史失败模式和同组对比模式；与现有技术（如纯单步反射）的本质区别是引入类似梯度下降"动量"的状态跟踪机制。
3. **大规模实证验证**：在 6 个基准、5 个不同规模模型上验证，SkillCome 在全部 26 个 model-benchmark 对上取得最高分，平均相对 SkillOpt 提升 5.69 个百分点，相对"无技能"基线平均增益最高达 19.55%。

## 方法详解
**问题形式化**：给定技能文档 s，目标模型 M_tar 在 s 下执行任务生成轨迹 τ，评估器给出二元得分 R(x, τ) ∈ {0,1}。目标为 max_s Σ_{x∈D} E_τ[R(x, τ)]。迭代流程：训练→优化器提议编辑→验证集评估→仅当 V(s_cand) > V(s) 时接受。

**Grouped Rollout**：从 D_train 采样 m 个问题，对每个问题 x_i 独立生成 n 个轨迹 τ_ij ~ M_tar(· | x_i, s^t)，得分 r_ij。将成功轨迹集 G_i^{t,+} 和失败轨迹集 G_i^{t,-} 分离，混合组 G_mix^t = {G_i^t | G_i^{t,+} ≠ ∅ ∧ G_i^{t,-} ≠ ∅}。

**Dual Memory System**：
- 失败记忆 M_f：记录传统失败反射学到的模式 p_k 及其支持组集合 u_k。
- 对比记忆 M_c：记录组内对比学到的模式，模式规模 K 表示已学习模式数。
- 记忆更新：通过 read-and-update 交互，当前观察可与历史模式关联，支持集 u_k 可扩展而不必改变模式描述。

**Contrast Reflection**：对每个混合组 G_i^t，优化器 M_opt 比较成功/失败轨迹，定位最早行为分歧（first divergence），分析成功行为做了什么、失败行为遗漏了什么，并生成候选编辑 P_{c,i}^t 和对比记忆更新 ΔM_{c,i}^t。

**证据导向的技能修订**：合并所有编辑 P^t = Merge(P_f^t ∪ P_s^t ∪ P_c^t)，按对比编辑 > 失败编辑 > 成功编辑优先级排序，预算限制 L^t 内选择 Δ^t，应用得 s_cand^t。被拒绝的候选技能不撤销其对应的记忆更新。

## 实验与结果
**数据集**：6 个基准——SpreadsheetBench、SearchQA、LiveMathematicianBench、ALFWorld、OfficeQA、DocVQA。训练/验证/测试划分遵循 SkillOpt-Lite 协议（见 Appendix Table 3）。

**模型**：5 个目标模型（GLM-5.3、Qwen3.8-Flash-Next、DeepSeek-V4-Pro-0813、DeepSeek-V4-Flash、Qwen3.6-35B-A3B），前三个自演化，后两个使用更强优化器。

**基线**：No skill、Init skill、Trace2Skill、SkillOpt、SkillOpt-Lite。

**主要结果**：
- SkillCome 在全部 26 个 model-benchmark 对上取得最优，相对次优 SkillOpt 平均提升 **+5.69 个百分点**。
- DeepSeek-V4-Flash 在 OfficeQA 从 5.41% → **50.51%**（+45.10%），超第二名 24.50 个百分点。
- Qwen3.6-35B-A3B 在 LiveMath 从 30.19% → **71.46%**（+41.27%）。
- DeepSeek-V4-Pro-0813 在 ALFWorld 从 66.98% → **89.18%**（+22.20%）。

**消融**：移除双记忆后 SkillCome w/o Memory 仍优于 SkillOpt，但全量 SkillCome 在全部 11 对上都最佳，证明组对比 + 双记忆协同有效。

**跨模型/跨数据集迁移**：SkillCome 学到的技能可跨模型（Flash ↔ Qwen3.6）和跨数据集（SearchQA → HotpotQA、OlympiadBench → Omni-MATH）复用。

## 相关工作脉络
1. **Trace2Skill (Ni et al., 2026)**：并行提取轨迹局部教训并整合为可迁移的程序性指导；定位：关注"如何提取"，SkillCome 关注"如何构建更可靠的优化信号"。
2. **SkillOpt (Yang et al., 2026a)**：结构化优化流程将轨迹转化为有界技能编辑，配合验证接受机制；定位：单轨迹分析，SkillCome 在此基础上引入同题组内对比。
3. **SkillOpt-Lite (Shen et al., 2026b)**：将演化过程部分委托给智能体而非人工管道；定位：流程自动化，SkillCome 改进优化信号质量。
4. **AutoSkill (Yang et al., 2026b)**：经验驱动的终身学习技能自演化框架；定位：宏观框架，SkillCome 聚焦于对比优化与记忆机制的具体设计。
5. **SKILS (Li et al., 2026)**：技能架构、获取、安全与发展路径综述；定位：综述性工作，为 SkillCome 提供理论背景。
6. **Evoskill (Alzubi et al., 2026)**：自动技能发现用于多智能体系统；定位：目标与应用场景相近，但机制上 SkillCome 强调对比与记忆。

## 局限性与未来方向
1. **计算开销**：每个问题需生成 n=8 条轨迹，相比单轨迹方法开销增大 8 倍。
2. **混合组比例有限**：实验中约 32.49% 的组为混合组（含成功与失败），非混合组无法进行组内对比。
3. **记忆系统依赖人工设计模式表示**：当前模式以自然语言描述，可能存在表示瓶颈或丢失细节。
4. **离线演化假设**：当前框架假设可反复访问训练集，未探索在线持续演化场景。

## 研究启发与可借鉴点
1. **组内对比作为优化信号**：将同题成功/失败轨迹配对分析，比单独看轨迹信息量更高；可迁移至 RLHF/RLAIF 等偏好优化场景。
2. **双记忆的状态跟踪机制**：类似优化器中的"动量"，跨步骤累积证据区分信号与噪声；可借鉴到在线学习、持续学习中。
3. **成功的轨迹作为失败分析的参照**：不只是从失败推断"应该改什么"，而是从成功推断"实际有效做法是什么"；这一视角可推广到任何基于轨迹的分析任务。
4. **接受条件与记忆解耦**：技能编辑被拒绝时记忆更新仍保留，避免"因一次失败否定所有有价值观察"；可用于改进其他迭代优化框架。
5. **Prompt 结构化设计值得借鉴**：Appendix A.1 的 Group Contrast Reflection Prompt 设计了清晰的七步分析流程，兼顾可操作性与可解释性。

## 关键术语表
- **SkillCome**：本文提出的基于组对比优化与双记忆的技能演化方法。
- **Group Contrast**：在同一个问题的多个轨迹中对比成功与失败路径的行为差异。
- **Dual Memory**：包含失败记忆（M_f）和对比记忆（M_c）的历史证据积累系统。
- **Grouped Rollout**：对同一问题独立生成 n 条轨迹的实验设计。
- **Optimizer Model (M_opt)**：分析轨迹并提出技能修订的模型角色（可与目标模型相同或不同）。
- **Contrast Reflection**：通过组内对比生成候选技能编辑和对比记忆更新的优化步骤。
- **Mixed Group**：同时包含成功与失败轨迹的问题组，用于触发组内对比分析。
- **Generalizable Skill**：由多个不同问题的支持组共同支撑的可泛化技能模式。

## 可复现要素
- **数据集**：Six benchmarks (SpreadsheetBench, SearchQA, LiveMathematicianBench, ALFWorld, OfficeQA, DocVQA)，划分遵循 SkillOpt-Lite；训练集规模见 Appendix Table 3。
- **代码/权重**：论文声明"code is provided in the supplementary materials"（附录中提供复现代码）；模型权重使用开源模型（DeepSeek-V4、Qwen3.6 等）。
- **关键超参**：rollout 数量 n=8；最大回复长度 16384 tokens；训练轮数 4 epochs 或 10 batches；编辑预算 L^t（论文未明确具体值，见 Appendix B.2）；标准差评估采用 4 次独立运行取平均。
