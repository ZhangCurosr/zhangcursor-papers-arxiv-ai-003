---
title: "Trajectory-Guided-Fault-Localization-for-Agent-Skill-Evoluti"
source: https://arxiv.org/pdf/2610.11858v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:14:02"
field: "Agent Skill Engineering & Evolution"
keywords: ["Agent Skills", "Skill Evolution", "Fault Localization", "Trajectory Analysis", "LLM Agents", "SWE"]
innovations: ["首次将谱-Based故障定位(SBFL)思想引入Agent Skill Evolution，以动作级别替代黑盒LLM推理进行可追溯的edit-site定位", "设计跨重复执行-演化循环-任务三层的证据聚合机制（含Yule's Q历史加权），区分系统性技能缺陷与随机行为", "提出LLM递归定位Agent实现'技能→文件→段落'由粗到细的edit-site定位，支持跨多skill的协同修订"]
benchmarks: ["SWE-Skills-Bench", "CannBot (NPUKernelBench)"]
---

# 论文速读：Trajectory-Guided-Fault-Localization-for-Agent-Skill-Evoluti

## 一句话总结
本文提出 **SkillMorph**，一种基于轨迹引导的故障定位（spectrum-based fault localization）的 Agent Skill 进化方法，通过在生成修订前先显式地将执行证据锚定到具体的技能内容（edit sites），实现了跨任务、跨演化循环的精准技能修订，在 SWE-Skills-Bench 和 CannBot 两个基准上均显著优于原始技能和四种基线方法。

## 研究问题与动机
1. **修订缺乏行为证据锚定**：现有 Skill Evolution 方法直接将执行反馈（trajectory/任务得分）喂给 LLM 生成修订，缺乏将"问题行为"与"需修正的技能内容"显式关联的机制，导致修订往往偏离真实病灶，且无法追溯历史修订效果。
2. **难以区分技能缺陷与随机行为**：Agent 行为具有随机性，单次执行很难区分"技能的系统性缺陷"和"偶然采样偏差"；现有方法基于少量 rollout 归因，容易过拟合单例。
3. **无法处理跨多技能的协同缺陷**：一个任务往往由多个 skill 协同指导，现有方法一次只修订一个 skill，孤立编辑会遗漏其他 skill 中冲突或错误的内容。
4. **缺乏跨演化循环的历史追踪**：修订后若出现回归（regression）或无效修改，现有方法难以通过保留历史证据来识别和纠正。

## 核心贡献（创新点）
1. **轨迹引导的故障定位范式**：首次将谱-Based 故障定位（SBFL）思想引入 Agent Skill Evolution，将异常动作（actions）作为定位实体，而非直接对技能内容做黑盒 LLM 推理，使定位决策可解释、可追溯。
2. **跨三层证据聚合的嫌疑动作识别**：在单一任务内对比失败/成功覆盖率，结合演化循环间的历史变化加权（Yule's Q 变换），再跨任务聚合，实现从"单实例噪声"中提炼"通用技能缺陷"的三级聚合机制。
3. **LLM 递归搜索的定位链**：设计了一个"技能→文件→具体段落"的递归定位 Agent，将嫌疑动作映射到跨多个 skill artifact 的 edit sites，并生成带理由的定位报告。
4. **提案评估与 Patch 合并机制**：在修订生成环节引入三维度评估（嫌疑强度、与现有内容一致性、历史修订效果），通过链式思考（CoT）筛选高价值提案并合并为最终 Patch，避免重复/冲突修改。

## 方法详解
SkillMorph 分三阶段运行，共 L 轮演化：

**阶段一：执行与轨迹抽象**
- 每轮对每个演化任务独立执行 $m$ 次（默认 $m=3$），保留所有轨迹（无论成败）。
- 用 LLM + 层次聚类将原始轨迹 $\tau = \langle (a_1, o_1), \dots, (a_T, o_T) \rangle$ 抽象为状态转移序列 $\hat{\tau} = \langle \mu_0, a_1, \mu_1, \dots, a_K, \mu_K \rangle$，其中 $a_k$ 描述动作与受影响对象，$\mu_{k-1}, \mu_k$ 描述动作前后的观测状态。

**阶段二：嫌疑动作定位（Suspicious Actions Localization）**
- **单任务分析**：计算每个动作 $a$ 的失败覆盖率 $N^-(a)$ 和成功覆盖率 $N^+(a)$。在成功轨迹中，用 LLM 检查动作前后 $h$ 步窗口，判断该动作是否引入了错误（而非仅暴露已有错误），若是则计为失败覆盖。按 Ochiai 形式构建谱矩阵：
$$s_i(a) = \frac{\overline{N}_i^-(a)}{\sqrt{F_i \cdot (\overline{N}_i^-(a) + \overline{N}_i^+(a))}}$$
- **历史分析**：比较当前轮与上一轮的覆盖率变化，用 Yule's Q 变换计算权重 $w_{i,t}^\sigma(a)$，失败覆盖减少则降权（说明修订有效），增加则升权（说明修订无效或引入回归）。
- **跨任务聚合**：将历史调整后各任务的覆盖数求和 $\overline{N}_i^\sigma(a) = \sum_t \widetilde{N}_{i,t}^\sigma(a)$，再按上述公式计算全局嫌疑分数，选取 top-$n$ 动作作为 $A^\star$。

**阶段三：Edit-Site 定位与修订**
- **定位**：对每个嫌疑动作 $a^\star$，用 LLM Agent 递归缩小范围：先看 skill 描述与使用记录→定位到具体 skill→读取 SKILL.md 及引用文件→比对指令与执行结果的差异→输出 Edit Site（行号范围 + 定位理由）。
- **修订生成**：基于三源信息（Edit Site 上下文 + 嫌疑动作轨迹上下文 + 历史修订记录），为每个 Edit Site 生成修订提案。
- **评估与合并**：用 CoT 从三个维度评估提案：(1) 嫌疑强度与支持证据；(2) 与现有内容是否冲突/重复；(3) 是否破坏已有成功案例。筛选后合并为最终 Patch，更新 $S_{i+1}$。

## 实验与结果
- **数据集**：SWE-Skills-Bench（49 个独立 SE skill，每 skill 10 个任务，A/B 双折验证）；CannBot（12 个协同 AscendC Kernel 生成 skill，15 演化任务 + 15 测试任务）。
- **基线**：Original Skills、SkillOpt、EvoSkill、SkillAdaptor、CAP-Evolve。
- **核心结果（SWE-Skills-Bench Overall）**：
  - SkillMorph：**pass@3 = 95.51%，Trial Acc. = 92.24%，pass³ = 88.37%**
  - 相对 Original Skills 的相对提升：**pass@3 +15.84%（82.45%→95.51%），pass³ +30.03%（67.96%→88.37%）**
  - 最强基线 CAP-Evolve：pass@3 = 92.24%，SkillMorph 高出 **3.27pp**
- **核心结果（CannBot）**：
  - SkillMorph：**pass@3 = 93.33%，Trial Acc. = 93.33%，pass³ = 93.33%**
  - 相对 Original Skills 的提升：**pass³ 从 46.67% 翻倍至 93.33%（+100% 相对提升）**
  - pass³ 领先其次强基线 **20–40pp**
- **回归控制**：在 SWE-Skills-Bench 上，原一致成功的 333 个任务中保留 327 个（98.20%），而 CAP-Evolve 仅保留 309 个（92.79%）；在 CannBot 上 SkillMorph 零回归。
- **消融实验**：Single-run 对 pass@3 影响最大（-5.31pp SWE / -13.33pp CannBot），w/o Localization 对 pass³ 影响最大（-6.12pp SWE / -33.33pp CannBot），w/o History 影响最小。
- **实际应用**：与 AI 算子开发团队合作，提交 6 个 PR 均被接受；通过修订减少不必要的搜索操作，正确任务平均耗时减少 41.11%。

## 相关工作脉络
1. **SWE-Skills-Bench（[12]）**：本文使用的基准，提出 agent skill 在 SE 任务上的系统性评测，SkillMorph 在其上证明效果优于原始 skill 和多种演化方法。
2. **SkillOpt（[44]）/ SkillAdam（[17]）**：直接基于 rollout 文本编辑演化单一 skill 文档；SkillMorph 相比它们更强调跨 skill 协同编辑和历史证据追踪。
3. **SkillAdaptor（[46]）**：将失败轨迹归因到最早可操作动作并修订单一 skill；SkillMorph 通过多次执行 + 跨任务聚合区分随机行为与系统性缺陷，并支持多 skill 协同修订。
4. **FAMAS（[11]）**：将谱-Based 故障定位应用于多 Agent 系统的失败归因；本文将其思想扩展到 skill evolution，增加了跨演化循环历史和跨 skill 定位能力。
5. **ContractSkill（[24]）**：通过结构化步骤+后置条件的确定性验证器实现显式定位；但依赖 skill 能被表达为结构化形式且需预定义验证器，泛化性受限；SkillMorph 完全基于自然语言 skill 和轨迹行为证据。
6. **APR 方法（GenProg [36] 等）**：经典程序修复通过测试覆盖率定位可疑代码；SkillMorph 类比此范式，但以"动作"而非"代码行"为定位实体，适应自然语言 skill 的语义特性。

## 局限性与未来方向
1. **执行成本较高**：每轮需多次重复执行任务（$m=3$），导致演化时间主要来自代码 Agent 执行（SWE: ~19.8min/轮，CannBot: ~97.8min/轮），框架分析本身仅占 ~29min。
2. **Top-n 选择敏感**：实验显示 n=1 时 pass³ 仅 69.39%，n≥10 后才稳定；但未探索更精细的动态选择策略。
3. **任务级特异性问题难以完全消除**：双向交叉验证发现 21 个任务在两轮演化后仍全部失败，说明跨任务经验可能掩盖特定任务的即时障碍，需结合任务级自适应机制。
4. **抽象阶段的 LLM 依赖**：轨迹抽象和 success/failure 判定的 LLM 调用质量直接影响后续定位准确性，未对抽象噪声进行鲁棒性分析。
5. **未来方向**：结合 Reflexion/Voyager 式任务级自适应；将已验证的恢复操作固化为可复用 skill 指导；探索动态 $n$ 选择和少次执行的优化策略。

## 研究启发与可借鉴点
1. **SBFL 思想迁移到 Agent Skill 领域**：将"失败/成功覆盖率"对比从程序测试扩展到 Agent 轨迹中的动作级别，是一种通用的"行为证据驱动编辑"范式，可迁移到其他 Agent 能力优化场景（如 prompt 调优、多 Agent 协作协议优化）。
2. **历史加权机制（Yule's Q 变换）**：用相邻演化循环的覆盖率比值变化进行动量加权，既保留有效修订的收益，又对无效/回归修订升权提醒，是演化类方法的通用设计模式。
3. **Success 轨迹中的"引入错误后被恢复"分析**：不仅看最终成败，还用 LLM 检查成功轨迹中每个动作是否引入了可恢复错误——这一思路可用于更精细的 skill 缺陷挖掘。
4. **递归定位 Agent 的"由粗到细"策略**：先从 skill 描述筛选候选 skill→再读 SKILL.md 和相关文件→再精确定位行号，可有效控制 LLM 上下文窗口噪声，适用于任何基于自然语言 artifacts 的根因定位任务。
5. **与工业界 AI Operator 团队的联合部署验证**：6 个 PR 被接受 + 平均耗时减少 41% 的实证数据，为学术界-工业界合作的 skill 工程方向提供了参考模板。

## 关键术语表
**Skill Evolution**：指基于 Agent 执行反馈自动迭代修订 Agent Skill 内容以提升其效能的过程。
**Suspicious Action**：在跨重复执行和跨任务聚合中被判定为更可能反映技能缺陷（而非随机行为）的动作，是定位的核心中间实体。
**Edit Site**：嫌疑动作所对应的需要修订的技能内容（具体到哪个 skill 的哪段/哪几行），由 LLM 定位 Agent 递归搜索得出。
**Spectrum-Based Fault Localization (SBFL)**：源自程序测试的故障定位技术，通过比较程序实体在失败/成功测试中的覆盖率来排序可疑程度；本文将其适配到 Agent 轨迹动作级。
**Ochiai Score**：SBFL 中常用的一种可疑度度量，公式为 $N^-/\sqrt{F \cdot (N^- + N^+)}$，本文沿用其结构用于动作嫌疑评分。
**Pass@3**：3 次独立试验中至少有 1 次成功的任务比例；衡量任务覆盖广度。
**Pass³**：3 次独立试验中全部成功的任务比例；衡量执行一致性/可靠性。
**Trajectory Abstraction**：将原始 Agent 执行轨迹（工具调用序列）用 LLM + 层次聚类抽象为标准化的状态转移序列（$\mu_0, a_1, \mu_1, \dots$），使跨执行的动作可比对。

## 可复现要素
- **数据集**：SWE-Skills-Bench（公开，https://github.com/thanhtong1998/SWE-Skills-Bench）；CannBot/NPUKernelBench（由作者合作提供，非完全公开）。
- **代码**：论文未提及开源代码仓库。
- **模型**：DeepSeek-V4-Flash-0731（代码 Agent 和执行阶段统一使用）。
- **关键超参**：每任务执行次数 $m=3$；演化轮数 $L=3$（SWE）/ $L=5$（CannBot）；每轮选取 top-$n=10$ 嫌疑动作；LLM 检查窗口 $h$ 步（论文未明确具体值）。
- **实验环境**：SWE-Skills-Bench 在 Intel Xeon Gold 6426Y / 512GB RAM Ubuntu 机器；CannBot 在 8×Ascend 910B3 NPU / 1.5TB RAM。
