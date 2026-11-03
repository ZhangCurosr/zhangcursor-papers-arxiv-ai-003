---
title: "Scaling-Long-Horizon-Cowork-Agent-Training-Via-Recursive-Env"
source: https://arxiv.org/pdf/2610.18067v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:33:02"
field: "长horizon agent训练与 Reward Shaping"
keywords: ["long-horizon agent", "reinforcement learning", "environment synthesis", "credit assignment", "cowork agent", "recursive repair"]
innovations: ["Harness驱动的语义对齐环境联合构建与依赖追踪", "FCR因子兼容性重放实现递归环境修复与最小化选择", "VTCA确定性检查与依赖分配实现回合级信用学习信号"]
benchmarks: ["JobBench", "GDPval v2", "AutomationBench", "WorkBuddy Bench", "OfficeQA Pro"]
---

# 论文速读：Scaling-Long-Horizon-Cowork-Agent-Training-Via-Recursive-Env

## 一句话总结
论文提出 WorkRover 框架，通过"Harness驱动的语义对齐环境构建"、"因子兼容性重放（FCR）递归环境修复"和"验证器到回合信用分配（VTCA）"三个核心组件，解决长horizon cowork agent训练中环境缺陷难以迭代修正、终端奖励缺乏细粒度学习信号两大问题，在Qwen3.5-35B-A3B上post-train后于5个cowork agent基准上全面提升。

## 研究问题与动机
1. **环境缺陷的协同修正难题**：cowork环境中工作上下文、任务和验证器必须语义对齐；修正任务缺陷常需同步更新其工作上下文和验证器，但现有构建流水线缺乏迭代修正并保持组件对齐的支持。
2. **终端奖励的信用分配困境**：Agent rollout中有效步骤可能出现在失败轨迹中，冗余动作可能出现在成功轨迹里；早期信息收集步骤仅通过后期workspace变更间接贡献，传统GRPO等终端奖励方法难以提供细粒度学习信号。
3. **环境可扩展性与可靠性矛盾**：单纯扩展合成环境数量无法保证训练反馈可靠性；agent rollout可能暴露初始构建后仍存在的依赖缺失、规范不一致或验证错误。
4. **长horizon任务的可组合性挑战**：需要agent在多文件、多工具、多应用间协调操作，保持跨工件一致性，对环境的可执行性和验证器的证据Grounded性提出高要求。

## 核心贡献（创新点）
1. **Harness驱动的语义对齐环境联合构建**：引入Harness显式追踪工作上下文、任务和验证器的依赖关系，支持三者在统一版本化框架下联合构建与验证；与现有方法仅围绕任务束或环境扩展不同，本文强调组件依赖的一致性与可追踪性。
2. **因子兼容性重放（FCR）递归修复机制**：通过冻结每个组件的一个修复提案，枚举最多16种组合并依赖闭合展开，选择最小已验证修复集；相比全量重新生成，FCR在保持任务意图的前提下将修复成功率从60.7%提升至70.3%，同时将任务意图改变率从14.3%降至1.3%。
3. **验证器到回合信用分配（VTCA）**：利用冻结确定性检查度量workspace进度变化，通过记录的工具输出与artifact依赖将正进度信用有界分配给前置回合；与TRACE、TRCA等方法依赖模型概率或状态比较不同，VTCA直接基于workspace状态差值和精确依赖关系分配信用，无需中间Judge调用。

## 方法详解
**环境构建三阶段流水线**：
- **Phase I 工作上下文构建**：从项目仓库和DeepResearch轨迹收集文件，构建文件上下文（File Context）、能力上下文（Capability Context）和应用上下文（App Context），封装为job/file/app技能库，配套Mock/安装应用的WorkEnv。
- **Phase II 长horizon任务合成**：从技能匹配器出发，基于依赖图组装Starter WorkEnv，独立Demonstrator Agent执行任务获得Reference State，确保任务无预设解决方案路径。
- **Phase III 验证器合成与验证**：基于任务指令生成Evaluation Contract，绑定准则与证据，通过基础设施有效性检查、Canonical State Diff、Functional Checks、Semantic Judge四层验证，校准验证器接受有效解、拒绝无效解。

**FCR修复流程**：
- 诊断区分环境缺陷与agent决策错误，冻结∆C/∆T/∆H/∆V修复提案
- 枚举2^d组合（d≤4），过滤存在冲突的组合后，对剩余组合进行依赖闭合展开Γ(χ)
- 在相同agent策略和执行预算下重放，验证修复组合解决原问题且不引入新缺陷
- 选择更改最少组件版本的修复集Γ(χ*)，经新鲜seed确认后可推广到相关任务包

**VTCA信用分配公式**：
- 轨迹优势A^traj_{i,r}由终端奖励R_{i,r}标准化得到：R_{i,r} = G_{i,r}[(1-λ)S_{B,i,r} + λS_{A,i,r}]，其中G为完整性门控
- 冻结确定性检查计算每回合进度增量Δ_{i,r,t,k} = φ_{i,k}(x_{t+1}) - φ_{i,k}(x_t)
- 正增量至少一半归属产生回合，其余等分给直接使用的工具输出/artifact的前置回合；负增量完全归属产生回合
- 最终回合优势：Â_{i,r,t} = sg[A^traj_{i,r} + αH_{i,r}δ^{crit}_{i,r,t} - ηc_{i,r,t}]，α,η≥0，H为安全门控

## 实验与结果
**数据集与评估基准**：五个外部cowork agent基准——JobBench（职业任务）、GDPval v2（专业交付物）、AutomationBench（跨应用工作流）、WorkBuddy Bench Office（办公工作流）、OfficeQA Pro（文档Grounded推理）。

**训练设置**：SFT使用30,115条轨迹（Qwen3.5-397B-A17B生成），RL独立合成20,000 task-Verifier对，经4次pilot rollout筛选保留3,200对。

**主要结果**：
- **最强结果**：WorkRover-35B-A3B在全部五个基准上超越Qwen3.5-35B-A3B backbone
- **提升幅度**：JobBench +26.27pp（10.62→36.89%）、OfficeQA Pro +29.32pp（19.55→48.87%）、AutomationBench +22.67pp、WorkBuddy Bench +13.63pp、GDPval v2 +413.25 Elo（804.75→1218.00）
- 在27-35B开源模型组中五项基准均排名第一
- VTCA相比E-GRPO在AutomationBench +5.7pp、OfficeQA Pro +5.4pp、GDPval +43 Elo
- FCR维护修复SRR从60.7%提升至70.3%，任务回归率从7.0%降至0.7%

## 相关工作脉络
1. **ClawGym / EnvScaler / Agent-World**：聚焦可执行环境规模扩展或程序化构建，WorkRover额外强调多组件（Context/Task/Verifier）依赖对齐与版本化管理。
2. **ToolForge / MetaTask / CLI-Universe**：以任务束为中心合成可验证任务，WorkRover引入Skill Matcher和依赖闭合展开，确保文件关系与技能匹配的可执行性。
3. **FACET / ANCHOR**：共享规范对齐任务与验证器，WorkRover的Harness提供显式依赖追踪与修复传播机制，支持迭代演化。
4. **Envs-FORGE / EchoVerse**：研究环境联合修订或可执行环境演化，WorkRover将修复提议冻结、组合枚举和最小化选择解耦，降低搜索空间。
5. **TRACE / TRCA / GiGPO / HiPER**：信用分配依赖模型概率、状态比较或规划层次，VTCA直接使用workspace确定性检查和精确工具输出依赖。
6. **DRACO / STAMP**：依赖Judge打分或引用溯源分配信用，VTCA无需中间Judge调用，仅基于workspace快照差值和记录依赖。

## 局限性与未来方向
1. **验证器覆盖率限制**：VTCA的信噪比受限于checker覆盖度和依赖记录完整性，未覆盖的准则仅依赖终端监督。
2. **依赖记录准确性风险**：错误溯源可能导致信用错配，缺失链接会使前置回合无法获得信用。
3. **枚举复杂度上界**：FCR在最坏情况下需测试2^4=16种组合，虽通过冲突过滤剪枝，但大规模组件场景仍可能面临组合爆炸。
4. **35B模型上限**：在JobBench和GDPval上与400B+模型及GPT-5.5、Kimi K3等仍有差距，显示Scaling潜力待进一步挖掘。
5. **环境修复的局部最优**：FCR选择"最小已验证修复"而非全局最优修复，可能遗漏更简洁的非最小修复方案。

## 研究启发与可借鉴点
1. **依赖闭合展开修复策略**：FCR的"冻结提议→组合枚举→依赖闭合→最小化选择"流程可迁移至其他多组件系统的协同修复场景，如代码仓库、数据流水线。
2. **确定性检查替代中间Judge**：VTCA用workspace快照差分替代LLM Judge判断进度，既节省推理成本又避免Judge偏差，适用于任何有状态环境的变化检测任务。
3. **版本化Harness协调多组件**：将Context/Task/Verifier的依赖关系显式化并版本化管理，为多智能体协作系统的环境构建提供工程范式。
4. **任务yield优化思路**：通过关系感知扩展和验证引导重写将通过率从49.8%提升至74.8%，说明"构造后修复"比"纯生成筛选"更高效。
5. **有界信用分配的保守策略**：正信用最多分配一半给前置回合、负信用完全本地化，避免信用过度扩散导致训练信号稀释。

## 关键术语表
**WorkRover**：通过递归环境进化训练长horizon cowork agent的框架，包含Harness驱动构建、FCR修复和VTCA信用分配三大组件。

**Harness**：协调工作上下文、任务和验证器构建与演化的版本化管理层，显式追踪组件依赖关系并管理执行流程。

**Factorial Compatibility Replay (FCR)**：通过枚举组件修复组合、依赖闭合展开和最小化选择实现环境递归修复的机制。

**Verifier-to-Turn Credit Assignment (VTCA)**：利用冻结确定性检查度量workspace进度，通过记录依赖将有界正信用分配给前置回合的训练信号增强方法。

**Evaluation Contract**：基于任务指令和可见workspace条件生成的评估规范，固定准则、证据绑定、权重阈值和关键标志。

**Skill Matcher**：将任务范围映射到工作上下文可用资源，组合job/file/app技能形成内部工作流并构建依赖图的模块。

**Reference State**：由独立Demonstrator Agent从干净初始workspace执行任务获得的参考状态，用于验证器校准和完成证据对比。

**Safe Resolution Rate (SRR)**：衡量缺陷修复成功率且不对相关任务引入新失败的指标。

## 可复现要素
- **数据集**：自行合成，未公开；30,115 SFT轨迹和3,200 RL pair均未开源
- **代码/权重**：论文未提及开源，模型名称WorkRover-35B-A3B非公开下载
- **关键超参**：KL系数0.01、λ≤0.1、正信用分配比例ρ≤1/2、α和η权重系数、GRPO clipping
- **训练硬件**：论文未提及
- **评估配置**：OfficeQA Pro使用Claude Code、temperature 0.7、top-p 0.95、top-k 40、1800秒budget
