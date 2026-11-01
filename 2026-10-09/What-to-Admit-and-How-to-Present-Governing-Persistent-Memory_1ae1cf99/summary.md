---
title: "What-to-Admit-and-How-to-Present-Governing-Persistent-Memory"
source: https://arxiv.org/pdf/2610.11188v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:58:54"
field: "LLM Agent 记忆安全与治理"
keywords: ["persistent memory", "sycophancy", "cross-domain leakage", "admission control", "LLM agents", "memory safety", "inference-time governance"]
innovations: ["提出准入-呈现二分治理框架并在同一系统上分离测量两杠杆效应", "两种无需重训的推理时准入设计（FC/PS），通过情境化裁决+确定性编译实现物理排除", "严格预注册+三项互补受控实验设计，揭示准入是主要效应来源而呈现差异不显著"]
benchmarks: ["PersistBench", "MemSyco-Bench"]
---

# 论文速读：What-to-Admit-and-How-to-Present-Governing-Persistent-Memory

## 一句话总结
本文提出将持久化记忆的治理拆分为**准入（admission）**与**呈现（presentation）**两个独立杠杆，在推理时无需重训即可实现：通过情境化裁决将记忆条目按属性判定为可接受/拒绝，并由确定性代码编译为结构化格式后注入生成上下文，从而系统性降低大模型 Agent 的阿谀迎合（sycophancy）和跨域泄露（cross-domain leakage）。

## 研究问题与动机
- **核心问题**：LLM Agent 的持久化记忆可提升个性化，但易诱发阿谀迎合和跨域信息泄露。现有方法未将"是否让记忆进入上下文"与"如何让已准入记忆被表达"这两件事分开治理。
- **现有方法不足 1**：常见检索阈值/相关性过滤器仅以单一的 keep/discard 比特行使粗粒度准入，无法区分"同一记忆在不同查询下的不同适用性"。
- **现有方法不足 2**：prompt 级记忆引导（instruction-level steering）只作用于呈现层，没有准入决策，误导性信息仍会被保留在上下文中。
- **现有方法不足 3**：神经网关（如 MemGate）基于相似度空间过滤，但诱导阿谀的记忆与有益记忆在主题上天然相关，相关性网关难以分离两者；同时实验发现 Gate 在客观事实判断任务上无显著改善（p=1.0），且使个性化失败增加 8.0pp。
- **未研究空白**：尚无工作在同一记忆系统上分离并测量这两个杠杆及其各自效应量。

## 核心贡献（创新点）
1. **提出"准入-呈现"二分治理框架**：将持久化记忆的治理显式分解为准入（决定哪些记忆进入工作上下文）与呈现（决定已准入记忆的呈现方式）两个独立可测的杠杆，并在同一系统上分离测量其效应——现有工作未对此进行解耦分析。
2. **两种无需重训的推理时准入设计（FC / PS）**：FC 将整个记忆条目作为评估单元并映射到五种类型通道；PS 将条目分解为 typed units，映射到六种许可类别（含 deny）。两者均通过确定性代码编译属性为准入决策，被排除内容物理上不出现在 prompt 中——与现有网关的"相似性过滤"本质不同。
3. **情境化裁决（situated adjudication）机制**：单次批量裁决调用同时看到 query、answer target、task mode 和所有 top-k 记忆，先判定回答目标类型（关于用户/第三方/世界）和任务模式（factual/recommendation/planning/support），再对每个条目分配 kind/subject/epistemic status/temporality 属性，由独立策略代码而非裁决器决定准入——避免了裁决器直接做动作决策的风险。
4. **严格的预注册 + 受控实验设计**：所有确认性对比均在数据收集前预注册并冻结资产；引入三项互补实验设计（pipeline-level、verdict-shared rendering、single-variable tightening），可精确归因效应；外部基准以单 judge 评分、独立审计脚本复算所有统计量，9600 条外部判决中仅 2 条缺失且成对排除。
5. **实证发现：准入 > 呈现**：在外部基准上，FC 和 PS 分别将 pooled judged failure 降低 6.7pp（p=2.7×10⁻⁷）和 8.8pp（p=4.1×10⁻¹²）；两个候选呈现格式（PN vs CH）在多重比较校正后无稳健差异；单一变量对比显示收紧准入在呈现固定时进一步降低 17.5pp 跨域失败（p=1.6×10⁻⁴）。

## 方法详解
- **总体架构**：在推理阶段，Agent 检索 top-k 记忆后，进入两层治理：
  - **准入层（admission map α）**：将每条记忆 m 映射为 typed units u(m)（FC 中 u(m)={m} 即整条），再通过确定性策略决定哪些 units 进入工作上下文；被拒绝的内容物理删除，不注入 prompt。
  - **呈现层（presentation map β）**：对已准入的 units 决定 prompt 中的表面形式（分组标题、排序等）。

- **情境化裁决（adjudication）**：单次批量调用接收 query、近期对话、benchmark 提供 evidence、全部 k 条记忆；输出结构化 verdict。三个固定属性：
  1. **属性评估与策略决策分离**：裁决器只分配 kind/subject/epistemic status/temporality 等属性，不做准入动作决策；由确定性代码编译。
  2. **情境感知**：先判断 answer target 类型和 task mode，再对每个 unit 给出 verdict——同一条记忆对不同查询可得到不同准入结果。
  3. **物理排除**：被排除的 units 从 prompt 中完全移除，而非仅加标注。

- **FC（Factor-Compiled admission）**：将每条记忆作为整体评估，分配六个 influence factors（scope、subject match、role、epistemic support、temporal status、utility）；对混合条目 via allowed_content 字段限制可用部分；确定性代码将 factors 映射到五种 typed channels：constraints to follow / personalization signals / non-authoritative user views / supporting context / uncertain context。策略**不对称**：constraints 和 current preferences 采取 fail-open（属性不确定时保留），beliefs 和 unverified facts 采取 fail-closed（不确定时排除）。

- **PS（Permission-Semantic admission）**：先将每条记忆分解为 typed units（如"I prefer brand A because its battery lasts longest" → 一个 preference unit + 一个 factual-claim unit），再对每个 unit 独立分配属性并映射到六种 permission categories：五种 permitted uses（apply constraint / mention user view / etc.）+ deny；deny 的 unit 从 prompt 中删除。

- **两种呈现格式**（同一组准入裁决结果）：
  - **PN（permission-note rendering）**： admitted units 按"该节信息在此回答中可用于什么"分组并标注 section header。
  - **CH（channel rendering）**：admitted units 按五种 typed channel 标题分组。

- **无微调、无重训**：所有操作发生在推理时，不修改存储内容，不重新训练任何模型。

## 实验与结果
- **评估表面**：
  - **Development surface**：PersistBench [15]，三个维度（sycophancy n=122 / beneficial use n=91 / cross-domain leakage n=200），四骨干模型（Qwen3.7-Max、MiniMax-M3、Kimi-K2.6、GPT-OSS-120B）。
  - **External surface**：MemSyco-Bench [21]，四个任务各 300 样本（ofj、csc、mec、pm），DeepSeek-V4-Pro 生成，GLM-5.2 单 judge 评分。
- **主要结果（外部基准）**：

  | 任务 | Bare | Gate | FC | PS |
  |------|------|------|-----|-----|
  | ofj（%） | 45.2 | 45.5 | 23.7 | **20.4** |
  | csc（%） | 33.3 | 22.7 | 27.0 | 25.3 |
  | mec（%） | 91.7 | 85.7 | 85.0 | 84.0 |
  | pm 失败↑ | 14.7 | 22.7 | 22.3 | 20.0 |
  | **pooled** | 46.2 | 44.1 | **39.5** | **37.4** |

  - FC 较 Bare 降低 pooled failure **6.7pp**（p=2.7×10⁻⁷）；PS 降低 **8.8pp**（p=4.1×10⁻¹²）。
  - ofj 是最大效应：PS 降低 24.7pp（p=7.6×10⁻¹⁹），准确率从 56.2%→75.9%。
  - Gate 基线在 ofj 上无显著变化（p=1.0），pooled 亦不显著（p=0.08）；且使 pm 失败增加 8.0pp（p=0.0012）。
  - FC vs PS：pooled 差异不显著（p=0.066）。

- **预算匹配控制（budget-matched）**：各臂每 query 获得与 PS 相同数量的 memory units 后，PS 在 ofj 上仍显著优于 random-selection（23.3% vs 31.7%，Holm 校正 p=0.001）和 relevance-ranking（44.3%，p<0.0001）。
- **呈现比较（PN vs CH）**：九组 task-metric pair 中唯一名义显著项（ofj accuracy 73.7%→77.7%，p=0.043）经 BH 校正后不显著（q=0.39）；交叉骨干主检验 0/4 显著。
- **开发集跨域泄露降低**：Qwen3.7-Max 44.0%→14.5%/18.0%；Kimi-K2.6 37.0%→22.5%/22.0%（均通过 BH 校正）；GPT-OSS-120B 64.0%→70.0%/69.0%（p=0.12/0.20，不显著）。
- **单一变量收紧准入实验**：呈现固定，仅收紧注入集，跨域失败进一步降低 17.5pp（p=1.6×10⁻⁴）。
- **Memory-Evidence Conflict 残留**：所有治理 arm 的 misleading 率 82.3%-85.0%，Bare 为 91.7%，NoMem 为 0%——说明一旦冲突记忆进入上下文，任何呈现措辞均无法修复。
- **边界**：两种设计均增加个性化失败（FC +7.6pp，PS +5.3pp）；PS 未达预注册的效应门控（pooled 改善 8.8pp < 10pp 阈值；pm 下降 -6.0pp > 3pp 允许线）。

## 相关工作脉络
- **Generative Agents [14] / MemGPT [12]**：引入持久化记忆流；本文在其风险维度（sycophancy、leakage）上做缓解而非测量。
- **PersistBench [15] / MemSyco-Bench [21]**：前者形式化风险面（sycophancy median 97.8%、leakage median 53%），后者扩展至五类受控任务；本文用 PersistBench 作开发表面，MemSyco-Bench 作外部表面。
- **MemGate [23]**：query-conditioned 神经网关，在相同 substrate 上 reimplement 后发现其对 topically-relevant 的 sycophancy-inducing 记忆无效（ofj p=1.0），凸显本文情境化裁决的必要性。
- **Context-side structuring [6]**：仅在 prompt 层分区呈现，不改变准入内容；本文发现呈现 alone carry limited weight，与该文结论一致。
- **Storage-side admission [22] / retrieval-layer defenses [9, 18]**：前者在写入侧优化 memory-bank 质量，后者在检索侧做二元相关性决策——两者均不在生成前做基于情境的逐 unit 准入。
- **Deletion-centric [3, 10] / adversarial state-contamination [20]**：前者完全删除存储内容（正交于本文问题）；后者证明 summarization 可洗白 toxic 内容，与本文记忆-证据冲突观察一致。

## 局限性与未来方向
- **个性化失效增加**：两种准入设计均导致 personalized memory use 失败率上升（FC +7.6pp，PS +5.3pp），PS 未达预注册"pm 准确率下降不超过 3pp"的门控。
- **记忆-证据冲突无法修复**：只要冲突记忆留在上下文中，misleading 率始终维持在 82.3%-85.0%，任何呈现措辞均无法降低——说明 admission 的排除能力是关键瓶颈。
- **GPT-OSS-120B 不改善**：该骨干在测试配置下跨域泄露不降反升（64%→70%，p=0.12），虽 supplement S5 中 Two-Phase 配置可改善，但机制未明。
- **fine-grained situational permission 场景泛化不足**：专用 probe（supplement S9）显示 PS 在 object-switch/context-switch/opinion-use 三类权限上未显著优于 equal-budget relevance 基线。
- **外部基准使用单 generator**：开发集有四骨干对比，外部集仅 DeepSeek-V4-Pro，跨模型泛化性待验证。
- **推理时增强（thinking）未评估**：FC/PS 与 extended reasoning 的交互未在 confirmatory 实验中检验。

## 研究启发与可借鉴点
- **两杠杆分离实验设计**：pipeline-level / verdict-shared / single-variable tightening 三层对照可精确定位效应来源，适用于任何记忆治理系统的消融分析框架。
- **属性-策略分离的裁决架构**：裁决器只做属性分配（kind/subject/epistemic/temporality），由确定性代码做准入决策，避免 end-to-end 黑箱策略偏差——可迁移至其他需要安全治理的生成式系统。
- **fail-open vs fail-closed 的非对称策略**：constraints/preferences 不确定时保留（fail-open），beliefs/unverified facts 不确定时排除（fail-closed），这一设计原则可推广至其他信息治理场景。
- **预注册 + 独立审计脚本**：所有资产预冻结、独立脚本从原始 per-sample 标签复算所有统计量、缺失分 pairwise 排除而非 zero-fill——可作为高可信 AI 评测的范式参考。
- **可结合本团队方向**：若团队关注长上下文 Agent 的安全治理，本文的 situated adjudication 可作为检索后-生成前的即插即用模块；其 typed-channel 输出格式也可直接用于后续的 fine-tuning 数据构造。

## 关键术语表
- **Persistent Memory**：Agent 跨会话保存的用户偏好、事实与经历，用于支持个性化交互。
- **Admission（准入）**：决定哪些检索到的记忆条目进入当前生成的工作上下文的决策过程。
- **Presentation（呈现）**：决定已进入上下文的记忆如何组织与措辞表达（分组标题、排序、格式）。
- **Situated Adjudication（情境化裁决）**：单次批量调用同时看到 query、evidence、task mode 和所有记忆，为每个条目分配结构化属性（非动作决策）的裁决过程。
- **FC（Factor-Compiled admission）**：将整个记忆条目作为评估单元，分配 influence factors 并通过确定性代码映射到五种 typed channels 的准入设计。
- **PS（Permission-Semantic admission）**：先将记忆分解为 typed units，再对每个 unit 独立分配属性并映射到六种 permission categories（含 deny）的准入设计。
- **Sycophancy（阿谀迎合）**：Agent 顺应用户已有偏好或错误信念给出附和性回答而非正确回答的现象。
- **Cross-domain Leakage（跨域泄露）**：用户某一领域（如投资偏好）的记忆被错误应用于无关领域（如简历建议）的现象。

## 可复现要素
- **数据集**：PersistBench [15]（开发集）和 MemSyco-Bench [21]（外部集）；论文声明 per-sample binary labels、frozen adjudication/rendering assets 及独立审计脚本均已公开。
- **代码/权重**：论文声明三份预注册文档、所有 arms 的 per-sample 标签、frozen 资产及审计脚本随论文发布（论文未提及具体仓库 URL）。
- **关键超参**：temperature=0（Kimi-K2.6 为 API 强制 0.6；GPT-OSS-120B 为低 reasoning effort）；thinking 关闭（除 supplement 中探索性 pilot）；FC/PS 的确定性策略代码与 prompt 模板冻结。
- **统计方法**：配对 McNemar 检验、Holm 校正、Benjamini-Hochberg 校正；缺失分数 pairwise 排除。
