---
title: "Trajectory-Guided-Fault-Localization-for-Agent-Skill-Evoluti"
source: https://arxiv.org/pdf/2610.11858v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:30:01"
field: "Agent 自动化调试与技能演化"
keywords: ["Agent Skills", "Skill Evolution", "Fault Localization", "Trajectory Analysis", "LLM Agents", "SWE"]
innovations: ["将谱基故障定位（SBFL）思想从代码推广至 Agent 自然语言动作级别，首次用于技能演化", "提出三粒度证据聚合（跨执行→跨循环→跨任务）与历史方向性权重调整的可疑动作定位机制", "设计可追溯的编辑站点定位与多技能协调修订流程，输出带证据链的 patch"]
benchmarks: ["SWE-Skills-Bench", "CannBot", "NPUKernelBench"]
---

# 论文速读：Trajectory-Guided-Fault-Localization-for-Agent-Skill-Evoluti

## 一句话总结
提出 **SkillMorph**，一种基于轨迹引导的错误定位的 Agent 技能自动演化方法，通过比较多次执行和演化循环中的成功/失败轨迹，识别"可疑动作"并定位到具体技能编辑站点，再协调生成修订补丁，从而替代人工技能调优。

## 研究问题与动机
1. **修订缺乏行为证据锚定**：现有方法直接让 LLM 从执行反馈生成修订，无法追溯"哪些行为有问题→应修改技能哪部分"，导致修订不可检查、可复现性差。
2. **难以区分技能缺陷与偶发行为**：现有方法通常基于有限 rollout 归因，缺乏跨执行一致性的证据，容易拟合个别案例而忽略类内普遍缺陷。
3. **无法协调多技能联合修订**：现有方法大多只修订单个技能文档，但实际任务往往由多个技能协同引导，孤立修订会遗漏冲突或缺失的指引。
4. **人工技能调优成本高、难扩展**：专家需手动审查轨迹、回溯问题动作、迭代修订评估，难以随任务和技能规模增长而扩展。

## 核心贡献（创新点）
1. **首次将谱基故障定位（SBFL）思想引入 Agent 技能演化**：将执行轨迹中的动作视为"程序实体"，比较其失败/成功覆盖率来定位可疑行为，有别于现有工作依赖 LLM 隐式归因的做法。
2. **提出多粒度证据聚合的可疑动作定位机制**：依次在"同任务多次执行→跨演化循环→跨任务"三个维度聚合行为证据，并引入历史方向性权重调整（Yule's Q 变换），避免重复修复已改善的问题。
3. **设计可追溯的编辑站点定位与协调修订流程**：通过 LLM 代理从动作上下文递归缩小搜索范围（技能→文件→内容行），输出带证据链的编辑站点；同时支持跨多个技能/文件的协同修订，避免孤立修改导致新的不一致。
4. **在真实工程场景落地验证**：部署于 AI operator 开发团队的内核生成技能演化，提交 6 个已接受的 PR；在 SWE-Skills-Bench 和 CannBot 两个 benchmark 上全面超越 4 个现有基线。

## 方法详解
### 整体流程
SkillMorph 在 L 个演化循环中迭代执行：`(1) 执行与轨迹抽象 → (2) 可疑动作定位 → (3) 编辑站点定位与修订生成`，并将演化记忆（执行证据、分析结果、修订历史）保留至下一循环。

### 1. 执行与轨迹抽象
- 每个演化循环中，对每个演化任务 t 独立执行 m 次（同技能集 $S_i$），收集全部轨迹 $\mathcal{T}_{i}(t)$，无论成败均保留（成功轨迹中的"先犯错后恢复"同样提供信息）。
- 使用 LLM + 层次聚类算法将原始轨迹 $\tau$ 抽象为有序状态转换序列 $\hat{\tau} = \langle \mu_0, a_1, \mu_1, \ldots, a_K, \mu_K \rangle$，其中 $a_k$ 描述动作与受影响对象，$\mu_{k-1},\mu_k$ 描述执行前后的观测状态。这使得跨执行的"动作"成为可比对的基本单元。

### 2. 可疑动作定位（分三阶段）
**阶段一：单任务分析（Spectrum-based 计数）**
- 定义覆盖率：run 中只要存在动作 $a$ 即计一次覆盖；失败 run 中所有出现动作均计**失败覆盖**；成功 run 中通过 LLM 检查 $a$ 前后 h 步窗口，判断 $a$ 是否**引入错误状态**（若引入则计失败覆盖，否则计成功覆盖）。
- 使用 Ochiai 形式计算可疑得分：
$$
s_i(a) = \frac{\overline{N}_i^-(a)}{\sqrt{F_i \left( \overline{N}_i^-(a) + \overline{N}_i^+(a) \right)}}
$$
其中 $\overline{N}^-$ 和 $\overline{N}^+$ 分别为聚合后的失败/成功覆盖计数，$F_i$ 为含至少一个失败覆盖动作的 run 总数。该得分由"覆盖率（coverage ratio）× 特异性率（specificity ratio）"组成，避免 routine 动作（如读文件）因高频出现而被误判。

**阶段二：历史分析（跨演化循环的权重调整）**
- 计算相邻循环覆盖率比率 $r_{i,t}^\sigma(a) = \frac{N_{i,t}^\sigma(a)+1}{N_{i-1,t}^\sigma(a)+1}$，通过有界变换得到权重：
$$
w_{i,t}^\sigma(a) = \frac{2(N_{i,t}^\sigma(a)+1)}{N_{i,t}^\sigma(a)+N_{i-1,t}^\sigma(a)+2}
$$
- **失败覆盖下降**（修订有效）→ 权重<1 降权，避免反复修改已改善的问题；**失败覆盖上升**（修订无效或引入回归）→ 权重>1 升权；**成功覆盖上升**→ 升权保护有效实践；反之降权。

**阶段三：跨任务聚合**
- 将经历史调整后的各任务计数求和：$\overline{N}_i^\sigma(a) = \sum_{t\in D_e} \widetilde{N}_{i,t}^\sigma(a)$，**先调整后聚合**以避免不同任务趋势互相抵消。
- 保留所有在至少一个任务中有失败覆盖的动作作为候选，取 top-n 最可疑动作为 $A^\star$。

### 3. 编辑站点定位与修订
- **编辑站点定位**：对每个 $a^\star\in A^\star$，LLM 代理沿轨迹中的执行上下文（查询了哪些 skill/file）递归缩小搜索：skill 描述 → SKILL.md → 关联 artifact → 具体行范围，输出 edit site（含文件、行号、rationale）。
- **修订生成**：每个 edit site 的 proposal 由三类信息共同驱动：(a) 当前内容与定位 rationale；(b) 可疑动作及其执行上下文；(c) 修订历史（前一循环的覆盖率变化）。
- **Proposal 评估与合并**：CoT 评估三个维度：证据强度（可疑得分、覆盖范围）、与现有内容一致性（是否重复/冲突）、对成功用例的影响（是否破坏有效指引）。最终将选中的 proposal 合并为 patch，更新 $S_i\to S_{i+1}$。

## 实验与结果
### 基准与设置
- **SWE-Skills-Bench**：49 个独立软件工程项目技能，每技能 10 个任务，划分 A/B 两折做双向泛化测试（A→B / B→A）。
- **CannBot**：12 个协调的多技能系统，用于 AscendC kernel 生成，15 个演化任务 + 15 个 held-out 评估任务。
- 评估指标：pass@3（3 次中至少 1 次成功）、trial-level accuracy（单次 trial 成功率）、pass³（3 次全部成功）。
- 实现：Claude Code + DeepSeek-V4-Flash-0731 作为 code agent，m=3 次/任务，L=3（SWE）/ L=5（CannBot），每轮选 top-10 可疑动作。

### 主要结果
| 方法 | SWE pass@3 | SWE trial-acc | SWE pass³ | CannBot pass@3 | CannBot trial-acc | CannBot pass³ |
|---|---|---|---|---|---|---|
| Base Skills | 82.45% | 75.44% | 67.96% | 80.00% | 66.67% | 46.67% |
| CAP-Evolve（最强基线） | 92.24% | 89.39% | 84.90% | 93.33% | 84.44% | 73.33% |
| **SkillMorph** | **95.51%** | **92.24%** | **88.37%** | **93.33%** | **93.33%** | **93.33%** |

- 相对原始技能：**SWE pass@3 提升 15.84%，pass³ 提升 30.03%；CannBot pass@3 提升 16.67%，pass³ 提升 100%（翻倍）**。
- 相对最强基线 CAP-Evolve：SWE pass@3 提升 3.27pp，**pass³ 提升 3.47pp**；CannBot pass³ 提升 **20pp**。
- 回归控制：SWE 上仅 9 个任务退化（CAP-Evolve 为 36 个），CannBot 上 0 个退化。

### 消融与效率
- **组件消融**：去除重复执行（Single-run）对 pass@3 影响最大（SWE -5.31pp）；去除定位（w/o Localization）对 pass³ 影响最大（SWE -6.12pp）；去除历史记录影响最小。
- **演化循环**：第一 loop 贡献 85–88% 的最终 gain，后续 loop 主要改善执行一致性。
- **运行效率**：SWE 单 loop 约 49min（框架分析 28.8min，agent 执行 19.8min）；CannBot 单 loop 约 127min（框架 29min，agent 执行 97.8min）。

## 相关工作脉络
1. **SkillOpt [44]**：直接对单个 skill 文档做 rollout-guided 文本编辑+验证门控，不显式定位行为→技能内容的对应关系，也无法协调多 skill。
2. **SkillAdaptor [46]**：将失败轨迹归因到最早的可操作 step 并修订一个 skill，单步归因易受 stochasticity 干扰，缺乏跨任务一致性证据。
3. **EvoSkill [1]**：通过 failure-driven mutation 演化结构化 skill folder，使用 Pareto 选择，但未引入谱基故障定位思想来系统地区分缺陷与偶发行为。
4. **CAP-Evolve [6]**：迭代地进行 failure diagnosis→candidate generation→validation-based selection，适用于 prompt/skill/tool 联合演化，但缺少对"同一行为在多次执行中一致出现"这一关键信号的利用。
5. **FAMAS [11]**：在 multi-agent 系统中使用谱基归因对比成功/失败执行，归因到 responsible agent action；本文将其思想扩展至"跨演化循环、跨任务、动作→自然语言技能内容"的定位，并进一步生成可执行的修订。
6. **ContractSkill [24]**：通过 postcondition + 确定性 verifier 实现显式 step 定位，但要求 skill 必须以结构化契约形式表示，通用性受限；SkillMorph 适配自然语言 skill，无需预定义合约。

## 局限性与未来方向
1. **Task-specific 适应仍有局限**：双向交叉验证中发现 21 个任务在原始与演化技能下均全失败，说明跨任务经验积累仍无法覆盖个别任务特有障碍，需结合 task-level 反思机制（如 Reflexion / Voyager）。
2. **超参数敏感性**：top-n 动作选择需权衡——n=1 时 pass³ 仅 69.39%（一致性差），n 过大边际收益递减，最佳 n 依赖场景。
3. **轨迹抽象依赖 LLM**：抽象过程本身引入 LLM 判断误差；success run 中"是否引入错误"的判断窗口 h 也需调优。
4. **演化循环数需经验设定**：当前 SWE 用 3 轮、CannBot 用 5 轮，不同 skill 系统的最优轮数未知。
5. **多技能协调修订的质量控制**：跨 skill 的 patch 合并逻辑（解决 overlap）依赖 LLM CoT 评估，缺乏形式化约束。

## 研究启发与可借鉴点
1. **谱基故障定位思想的迁移**：将 SBFL 中"失败/成功覆盖率"的核心直觉从代码推广到"LLM agent 的自然语言动作"，为其他 agent 系统的故障归因提供了通用框架，可直接借鉴到 multi-agent 系统调试、agent evaluation 等场景。
2. **轨迹抽象 + 状态转换表示**：将 verbose 原始轨迹归一化为有序 $(\mu_{k-1}, a_k, \mu_k)$ 序列，使跨执行比较成为可能；该抽象层可作为通用"trajectory normalizer"复用。
3. **历史方向性权重调整的范式**：用 $w = \frac{2(N_{curr}+1)}{N_{curr}+N_{prev}+2}$ 这种有界变换区分"改善/恶化/中性"趋势，比简单累加或差分更能稳健反映修订有效性，可迁移至任何迭代式系统优化（如超参 tuning、prompt evolution）。
4. **跨多工件的协调修订机制**：将 edit site 定义为"跨多个 skill/file 的行范围集合"，并要求 proposal 明确标注"各工件间的协同约束"，这一设计对文档型知识库维护、配置管理自动化有参考价值。
5. **与团队方向的结合机会**：若团队关注 agent 行为可解释性或自动调试，可将 SkillMorph 的"可疑动作定位→编辑站点映射"模块集成到现有 trace 分析管线中，作为"定位-修正"闭环的前端；同时可用团队内部的 multi-skill 场景复现验证。

## 关键术语表
**Agent Skills**：封装可重用指令、脚本与参考材料的文件包（入口为 SKILL.md），为 code agent 提供特定工作流的专业知识，无需修改底层模型参数。
**Skill Evolution**：基于执行反馈迭代修订 agent 技能集合，以提升 agent 在未见任务上的表现，目标函数为 held-out 任务的成功率期望。
**Spectrum-based Fault Localization (SBFL)**：源自软件测试的经典方法，通过比较实体在失败/成功测试中的覆盖率来排序可疑度；本文将其迁移至 agent 动作级别。
**Edit-Site**：需要修订的技能具体内容区域，以（技能名、文件路径、行号范围）表示，可由单个可疑动作关联多个跨 skill 的站点。
**Suspicious Actions**：在多次执行、跨演化循环、跨任务中表现出高失败覆盖率与高特异性的动作，被认为是技能缺陷的行为表征。
**Trial-level Accuracy**：单次 trial 成功的比例，衡量 agent 执行的稳定性与可重复性。
**Pass@3 / Pass³**：pass@3 表示 3 次独立 trial 中至少 1 次成功；pass³ 表示 3 次全部成功，后者更能反映执行一致性。
**Evolution Memory**：跨循环保存的执行证据、历史覆盖率计数与修订记录的存储结构，支撑跨 loop 的趋势分析与 revision 追溯。

## 可复现要素
- **数据集**：SWE-Skills-Bench（公开，https://github.com/...）、CannBot（关联 NPUKernelBench，公开）、两者均有任务划分协议。
- **代码/权重**：SkillMorph 未声明开源仓库；基线方法使用官方实现复现（SkillOpt、EvoSkill、SkillAdaptor、CAP-Evolve 均有公开代码）。
- **关键超参**：m=3（每任务重复执行次数）、L=3（SWE）/ L=5（CannBot）（演化循环数）、n=10（top-k 可疑动作数）、h（success run 错误检测窗口步数，论文未明确给定数值）。
- **环境**：SWE 实验 Ubuntu 22.04 + Xeon Gold 6426Y + 512GB RAM；CannBot 实验 8×Ascend 910B3 NPU + 1.5TB RAM + CANN 9.1.0。
- **模型**：DeepSeek-V4-Flash-0731（code agent + 轨迹抽象 + 定位 + 修订全流程使用同一模型）。
