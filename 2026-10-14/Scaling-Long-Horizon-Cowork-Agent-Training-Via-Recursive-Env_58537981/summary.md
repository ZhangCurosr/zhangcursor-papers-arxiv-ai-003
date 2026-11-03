---
title: "Scaling-Long-Horizon-Cowork-Agent-Training-Via-Recursive-Env"
source: https://arxiv.org/pdf/2610.18067v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:33:22"
---

# 论文速读：Scaling-Long-Horizon-Cowork-Agent-Training-Via-Recursive-Env

## 一句话总结
提出 WorkRover 框架，通过 Harness 驱动的环境联合构建、因子兼容性回放（FCR）递归修复与校验器到回合的信用分配（VTCA），解决长程 cowork agent 训练中长期存在的环境组件失配与终端奖励稀疏问题；在 Qwen3.5-35B-A3B 上 post-train 后于五个外部 cowork-agent 基准上取得全面领先提升。

## 研究问题与动机
1. **环境组件对齐困难**：长程 cowork agent 训练需要 Work Context、Task、Verifier 三者语义一致，但现有合成流水线缺乏协同迭代机制，单一组件修复易引发级联不一致。
2. **隐性缺陷修复成本高**：初始生成环境常存在依赖缺失或校验错误，全量重生成耗时且易改变任务意图；缺乏在冻结任务约束下精准定位并最小化修复的方法。
3. **终端奖励信用分配稀疏**：长轨迹中早期信息收集/工具调用步骤的贡献难以归因，episode-level GRPO 无法区分有用步骤与冗余操作，学习效率受限。
4. **工作区组装缺乏可执行性保障**：现有方法多依赖主题相似度拼装文件与应用，忽视文件间依赖与技能接口兼容性，导致合成任务可执行率低。

## 核心贡献（创新点）
1. **Harness 驱动的语义对齐环境联合构建**：显式维护 Context、Task、Verifier 的版本化依赖图，支持联合构建与可行性验证；与仅按主题相似度拼装的工作区相比，从源头保障文件依赖、技能接口与校验证据的一致性。
2. **因子兼容性回放（FCR）递归进化机制**：冻结候选修复提案后枚举最多 16 种组合，通过依赖闭包展开 $\Gamma(\chi)$ 补全联动更新，选择满足约束且修改版本数最少的 $\chi^\star$；区别于全量重生成或贪心顺序修复，FCR 明确分离提议生成、兼容性选择与回归验证。
3. **Verifier-to-Turn Credit Assignment（VTCA）细粒度信用分配**：在 GRPO 终端优势基础上引入冻结确定性工作区进度检测与精确工具/产物依赖追踪，将正向增量有界分配至先决回合；与依赖中间 Judge 或计划层级归因的方法相比，无需额外推理调用且避免虚假因果。
4. **端到端可训练的长程 cowork 闭环**：结合 SFT 初始化与 VTCA-GRPO，在 3,200 对高质 Task-Verifier 上完成 post-training，五项基准均显著超越 Backbone 且超越同参量级开源模型。

## 方法详解
- **Harness 驱动构建**：环境组织为 $\mathcal{L} = \{\mathcal{P}_i = \langle C_i, T_i, V_i \rangle\}$。Phase I 构建三层 Work Context（文件/能力/App），建立 typed graph 索引；Phase II 由 Skill Matcher 根据任务规格匹配 job/file/app 技能依赖图，组装 Starter WorkEnv 并由独立 Demonstrator Agent 执行获得 Reference State $x_i^\star$；Phase III 从指令推导 Evaluation Contract，合成含证据绑定、权重 $w_k$ 与阈值 $\gamma_k$ 的 Verifier，通过 starter 拒绝、参考态接受、负样本校准的三阶段准入。
- **FCR 递归进化**：诊断区分环境缺陷与策略失误。对 Eligible 组件冻结提案 $\Delta C, \Delta T, \Delta H, \Delta V$，枚举 $\chi \in \{0,1\}^d$（$d\le4$）剔除依赖/资源冲突后，展开为依赖闭包 $\Gamma(\chi)$ 并在匹配 rollout 条件下重放。选择通过独立检查、无任务回归且 $|\Gamma(\chi^\star)|$ 最小的组合，新鲜种子确认后备份或淘汰；被否决候选退回最早受影响构建步骤重新验证。
- **VTCA 信用分配**：轨迹优势 $A_{i,r}^{\text{traj}}$ 经组内标准化。冻结 checker 计算每回合前后 workspace 得分差 $\Delta_{i,r,t,k}$，正值至少一半保留于产生回合，其余等分给精确消费该工具输出/产物版本的先决回合（上限 $\rho \le 1/2$）；负增量完全归属本回合。更新公式为：
  $$\widehat{A}_{i,r,t} = \mathrm{sg}\left[A_{i,r}^{\text{traj}} + \alpha H_{i,r}\delta_{i,r,t}^{\text{crit}} - \eta c_{i,r,t}\right]$$
  其中 $H$ 为完整性/保护状态/安全门控，$c$ 为工具错误与重复状态保存的惩罚，$\alpha,\eta\ge0$。结合 clipped GRPO 完成 turn-balanced 更新，$\alpha=\eta=0$ 时退化为 episode-only 基线。

## 实验与结果
- **训练数据**：SFT 使用 30,115 条经验证通过的轨迹；RL 从 20,000 候选中经四路 35B-A3B pilot 筛选保留 3,200 对 Task-Verifier，按 Work Context 划分。
- **主结果**：WorkRover-35B-A3B 在五个基准上全面超越 Backbone（JobBench +26.27 pp、GDPval v2 +413.25 Elo、AutomationBench +22.67 pp、WorkBuddy Bench (Office) +13.63 pp、OfficeQA Pro +29.32 pp）。在 27–35B 开源组全部第一，JobBench 与 AutomationBench 逼近 GPT-5.5，GDPval 缩小至与 400B+ 模型约 100–350 Elo 差距。
- **消融**：VTCA 较 E-GRPO 在 AutomationBench (+5.7pp)、OfficeQA Pro (+5.4pp)、GDPval (+43 Elo) 持续领先；FCR 自然缺陷 SRR 从 60.7% 升至 70.3%，跨任务回归从 7.0% 降至 0.7%，意图变更从 14.3% 降至 1.3%；构建管线消融显示移除 Relation Expansion 或 Validation Rewrite 后 Task yield 分别跌至 52.6% 与 49.8%。

## 相关工作脉络
1. **环境合成**：对比 EnvScaler、ClawGym、Agent-World 等按任务束或能力字典扩展的流水线，本文以文件依赖图与专业技能匹配为核心，通过 Harness 维护版本化组件依赖，强调可执行性而非表面相似性。
2. **环境修复**：对比 FACET、EchoVerse、Envs-FORGE 的联合修订或错误路由策略，本文 FCR 冻结提案后枚举组合并做依赖闭包展开，明确分离提议生成、兼容性选择与验证，保证修复集在枚举子空间内最小。
3. **过程信用分配**：对比 GRPO、TRACE、DRACO、STAMP、TRCA 等方法，VTCA 完全依赖冻结 checker 的工作区状态差分与精确工具/产物依赖追踪，避免中间 Judge 调用、历史轨迹似然差分或规划层级假设带来的随机性与偏差。
4. **校验可靠性**：对比 ANCHOR、FACET 等共享规范对齐方案，本文通过固定 Contract、反事实 mutant 测试与独立有效替代方案校准 verifier，强调基础设施失效与策略错误的语义分离。

## 局限性与未来方向
- FCR 仅在冻结提案的 $2^d$ 子空间内搜索最小修复，未覆盖全局最优解；提案生成质量与冻结约束直接影响进化上限。
- VTCA 信用分配严格依赖 checker 覆盖率与精确依赖记录，缺失追踪或错误溯源会导致信用误分或遗漏，对不可观测中间状态的任务效果受限。
- 合成漏斗转化率低（20,000→3,200），构建与验证成本较高，限制了更大规模任务池的快速扩展。
- 未探索跨 Work Context 的通用技能迁移、动态 skill 生成以及 >100B 模型的 scaling 行为。

## 研究启发与可借鉴点
1. **依赖闭包展开的最小修复范式**：FCR 的“冻结提案→组合枚举→依赖补全→最小化选择”可迁移至 RAG pipeline 调试、评测 harness 维护等多组件联合系统。
2. **冻结确定性进度探针替代中间
