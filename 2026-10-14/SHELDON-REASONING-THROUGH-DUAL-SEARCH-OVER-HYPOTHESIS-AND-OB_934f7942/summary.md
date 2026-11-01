---
title: "SHELDON-REASONING-THROUGH-DUAL-SEARCH-OVER-HYPOTHESIS-AND-OB"
source: https://arxiv.org/pdf/2610.18025v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:48:02"
field: "AI科学推理与自主智能体"
keywords: ["scientific reasoning", "test-time scaling", "dual-space search", "hypothesis exploration", "observation space", "LLM agents"]
innovations: ["显式分离并联合搜索假设空间与观察空间，观察驱动新假设生成", "引入对比分析机制主动寻求区分竞争假设的证据", "在无评分函数场景利用观察作为隐式评分信号"]
benchmarks: ["DiagnosisArena", "ORCA-Bench", "ARC-AGI-2", "DOCKSTRING", "ADRS", "QNN Circuit Topology"]
---

# 论文速读：SHELDON: REASONING THROUGH DUAL SEARCH OVER HYPOTHESIS AND OBSERVATION SPACES

## 一句话总结
论文提出 **SHELDON**（Systematic Hypothesis Exploration and Learning through Directed Observation and Navigation），一种基于假设-演绎法的双空间科学推理框架，通过同时探索假设空间与观察空间，指导LLM智能体在诊断、抽象推理和优化任务中发现更广泛的假设与观察，显著提升复杂推理能力。

## 研究问题与动机
- **现有测试时扩展方法的局限性**：现有方法在收集观察证据时通常仅围绕当前假设展开，导致观察发现与现有假设强耦合，偏向局部优化而非探索未知假设区域。
- **演化方法探索范围受限**：如 EvoX 等方法虽扩展了解空间，但其搜索仍锚定在初始种群和衍生候选上，无法跳出已有假设框架。
- **已有研究智能体缺乏假设多样性**：GoS、ScientistOne、Arbor 等方法将证据组织在现有假设下，聚焦高概率假设的调查，忽视假设共同假设外的现象，难以发现真正不同的新假设。
- **缺乏显式建模观察空间**：现有工作将观察视为辅助信息而非独立探索对象，导致无法主动寻找未被当前假设解释的现象，进而发现认知盲区。

## 核心贡献（创新点）
1. **双空间显式建模**：将假设空间与观察空间分别表示并联合搜索，观察作为一等公民驱动新假设生成，与已有工作仅将观察作为辅助证据的本质不同。
2. **假设-观察协同演化机制**：通过 CANDIDATE ANALYSIS 和 COMPARATIVE ANALYSIS 循环，观察不仅支持/反驳假设，还通过未解释观察激发全新假设，避免过早收敛。
3. **无需评分函数的推理能力**：与 EvoX、Arbor、CORAL 等方法依赖显式评分函数不同，SHELDON 可在无评分函数场景（如医疗诊断、根因分析）下利用观察作为隐式评分信号。
4. **大规模实验验证**：在 DiagnosisArena、ORCA-Bench、ARC-AGI-2、QNN拓扑优化、分子优化、ADRS系统等10项任务上，SHELDON 最高超越基线 26.7 分，并发现完美得分的量子神经网络拓扑。

## 方法详解
SHELDON 的核心是**双空间联合搜索**，状态表示为 $S_t = \langle H_t, O_t, wH_t, wO_t, E_t \rangle$：
- **假设空间** $H$：由轴（axis）和区域（region）组织，轴标识候选差异维度（如方法族、假设、参数确定方式），区域描述替代方案。工作假设集 $wH_t$ 包含 RETAINED（保留）和 SETASIDE（搁置）状态的假设。
- **观察空间** $O$：同样由轴和区域组织，轴包括数据来源、时间范围、实体等，工作观察集 $wO_t$ 包含累积证据。观察具有 PROVISIONAL 属性，需多次验证才可用于反驳假设。
- **证据关系** $E_t: H \times O \to \{\text{SUPPORT, REFUTE, NEUTRAL, UNRESOLVED}\}$，类似NLI判断。

**推理流程（Algorithm 1）**：
1. **EXPLORE**：检查案例材料、工具及当前探索状态，定义新假设/观察空间的轴与区域，提出新工作假设和观察集。若存在搁置假设（冷却期过后），可重新考虑以增加多样性。
2. **CANDIDATE ANALYSIS**：对每个工作假设单独深入测试，最多 N=3 次迭代，构建工件、执行评估、寻求支持与反驳证据，更新假设-观察证据关系。
3. **COMPARATIVE ANALYSIS**：对整个工作假设集进行联合分析，寻找能区分假设的证据，识别不同候选的差异、假设或预测行为，执行判别性测试。
4. **EXPLOIT**：基于观察分析假设，决定保留、修复、精炼或搁置（冷却3轮），然后返回 EXPLORE 阶段。

**关键设计**：
- 假设冷却机制：SETASIDE 假设需经过3轮冷却期才能重新考虑，防止过早丢弃潜在有用假设。
- 观察的 Provisional 状态：新观察默认为 provisional，需重复验证才可用于反驳假设，缓解 LLM 判断错误。
- 选择策略：每轮从探索和利用池中选取 k=5 个假设进入下一轮分析。

## 实验与结果
**基准任务**：
- 无评分函数：DiagnosisArena（医疗诊断）、ORCA-Bench（系统根因分析）、ARC-AGI-2（网格变换规则推理）
- 有评分函数：QNN Circuit Topology（量子神经网络拓扑优化）、DOCKSTRING 分子优化（6个靶标）、ADRS 系统优化（5个任务）

**主要结果**：
- **DiagnosisArena**：SHELDON 84.00，超越 RLM（78.00）6分、Codex（76.00）8分。
- **ORCA-Bench**：SHELDON 66.67，超越最强基线 ReAct/Codex/RLM（40.00）26.67分。
- **ARC-AGI-2**：SHELDON 59.17，超越 EvoX（47.50）11.67分。
- **QNN Circuit Topology**：SHELDON 达到完美得分 1.00，超越 EvoX（0.85）和手写参考拓扑（0.917）。
- **分子优化**：在 ADRB1、CDK2、PPARA、F2 四个靶标上超越 Codex、ReAct、EvoX，并在 CDK2、F2、PGR 上超越 DOCKSTRING 库最佳参考。
- **ADRS 系统优化**：在 EPLB、PRISM、Cloudcast、Transaction Scheduling、LLM_SQL 五个任务上均取得最佳或并列最佳均值和最优分数。

**消融实验**（Table 5）：
- 移除观察空间：DiagnosisArena 下降14.0分，ORCA-Bench 下降13.34分
- 移除对比分析：DiagnosisArena 下降6.0分，ORCA-Bench 下降10.0分
- 移除持续探索：ORCA-Bench 下降13.34分，QNN 下降0.267

**探索广度分析**（Figure 10-12）：
- SHELDON 在5/6任务上覆盖更多程序空间（k=20时，最强基线的1.4-20.9倍）
- 生成4.5倍于基线的新颖解决方案（QNN任务）
- 8/12任务的最佳假设来自新探索而非局部精炼

**与 GoS 对比**（ORCA-Bench）：
- SHELDON 发现76.6%（59/77）的真实根因，GoS 仅46.8%（36/77）
- SHELDON 对正确根因的假设调查率达98.3%，GoS 仅63.9%

## 相关工作脉络
1. **演化算法与LLM结合**：FunSearch、AlphaEvolve、EvoX 等方法搜索候选解决方案空间，但搜索锚定于初始种群，SHELDON 额外建模观察空间使观察驱动新假设。
2. **GoS（Luo et al., 2026）**：维护假设-证据因果图，但仅调查最高概率假设；SHELDON 对所有工作假设进行对比分析，并显式建模观察空间。
3. **ScientistOne（Meng et al., 2026）**：将科学声明链接到支持来源和可执行工件，但证据组织围绕现有假设；SHELDON 的观察可作为独立对象激发新假设。
4. **Arbor（Jin et al., 2026）与 CORAL（Qu et al., 2026）**：维护假设树或支持长期探索，但需初始工件和评分函数；SHELDON 可在无评分函数场景运行，通过观察作为隐式评分。
5. **Research Agents**：如 AI Scientist、Agent Laboratory 等自动化科学工作流，但多聚焦文献综述或实验执行；SHELDON 专注双空间搜索以发现新假设和解释。
6. **假说演绎法理论**：Hempel (1945)、Klahr & Dunbar (1988) 的双空间搜索模型，SHELDON 将其工程化为LLM推理框架。

## 局限性与未来方向
- **计算成本较高**：SHELDON 因维护多个假设并进行对比分析，在 DiagnosisArena 和 ORCA-Bench 上花费约 $2.9 和 $6.9，高于单智能体基线（ReAct: $0.02/$0.54，Codex: $0.06/$0.73）。
- **对种子程序的部分依赖**：无种子程序时，LLM_SQL 和 QNN 任务仅恢复79.3%和73.7%的 seeded 改进，表明某些任务仍需要有效初始化。
- **观察空间轴与区域的定义依赖任务理解**：EXPLORE 阶段需合理定义轴和区域，复杂任务的轴设计可能需要领域知识或更多尝试。
- **未来方向**：可扩展至更开放-ended 的科学发现任务；探索自动学习观察空间轴的方法；结合人类反馈进一步校准假设-观察关系。

## 研究启发与可借鉴点
1. **双空间建模的可迁移性**：将"假设空间"与"证据/观察空间"分离建模的思路可应用于代码调试、系统设计、药物发现等领域，避免单空间搜索的局部最优。
2. **对比分析（Comparative Analysis）机制**：主动寻找区分 competing hypotheses 的证据，而非仅验证单一假设，可提升推理的严谨性，适用于任何多假设场景。
3. **Provisional 观察状态设计**：引入"暂定"状态需多次验证才用于反驳，有效缓解 LLM 判断错误，可借鉴到其他需要累积证据的推理框架。
4. **假设冷却与重现机制**：SETASIDE 假设经冷却期后可重新考虑，避免过早放弃潜在有价值假设，平衡探索与利用。
5. **无评分函数的隐式评分**：利用观察作为隐式评分信号，可扩展至缺乏明确评估函数的场景（如开放-ended 科学发现）。

## 关键术语表
- **Hypothetico-deductive method（假说-演绎法）**：科学推理经典方法，从假设出发推导可检验预测，再通过观察证据支持或反驳假设。
- **Dual-space search（双空间搜索）**：Klahr & Dunbar (1988) 提出的科学发现模型，同时搜索假设空间和实验空间。
- **Working hypothesis set（工作假设集）**：当前 actively considered 的假设集合，状态为 RETAINED 或 SETASIDE。
- **Observation space（观察空间）**：由轴和区域组织的潜在观察集合，作为独立探索对象而非仅辅助信息。
- **Evidence relation（证据关系）**：假设与观察之间的四值关系：SUPPORT、REFUTE、NEUTRAL、UNRESOLVED。
- **Provisional observation（暂定观察）**：新观察的初始状态，需重复验证后才可用于反驳假设。
- **Candidate analysis（候选分析）**：对单个假设深入测试，寻求支持与反驳证据。
- **Comparative analysis（对比分析）**：对整体假设集进行联合分析，寻找区分假设的证据。

## 可复现要素
- **数据集**：DiagnosisArena、ORCA-Bench、ARC-AGI-2、DOCKSTRING、ADRS（EPLB、PRISM、Cloudcast、Transaction Scheduling、LLM_SQL）、QNN Circuit Topology，均为公开基准。
- **代码/权重**：代码、数据和artifact已开源，地址：https://github.com/stanford-oval/sheldon
- **关键超参**：rounds=8，n_initial=5，n_explore=3，select_k=5，local_max_evals=3，shelf_cooldown=3，stagnation_rounds=2，max_usd=15（ADRS）/10（DiagnosisArena/ORCA-Bench）/2.5（ARC-AGI-2）
- **模型**：推理使用 GPT-5.6-terra（high reasoning effort），代码实现委托给 GPT-5.6-luna
