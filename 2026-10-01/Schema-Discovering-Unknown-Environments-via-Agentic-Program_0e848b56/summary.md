---
title: "Schema-Discovering-Unknown-Environments-via-Agentic-Program"
source: https://arxiv.org/pdf/2609.39140v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:15:25"
field: "LLM Agent 与环境交互"
keywords: ["Agentic Program Induction", "World Model", "LLM Agent", "Mechanism Discovery", "Interactive Program Synthesis", "Long-horizon Exploration"]
innovations: ["提出交互程序归纳(IPI)范式，将环境机制显式编码为可执行程序并在 hypothesize-certify-plan-act 闭环中持续修订", "Schema harness 通过全历史回测(Certify)与逐动作验证(Act with Verification)防止知识退化、支持程序内搜索规划", "在 ARC-AGI-3/DiG-bench/MazeBench 三个未知环境基准上达到或超越人类水平，消融证实各组件缺一不可"]
benchmarks: ["ARC-AGI-3", "DiG-bench", "MazeBench"]
---

# 论文速读：Schema-Discovering-Unknown-Environments-via-Agentic-Program

## 一句话总结
论文提出 **Schema**，一种让 LLM Agent 将环境机制编码为可执行程序并通过交互式程序归纳（hypothesize → certify → plan → act with verification）持续学习和规划的新范式，在 ARC-AGI-3、DiG-bench 和 MazeBench 三个未知环境基准上均达到或超越人类水平。

## 研究问题与动机
- **核心问题**：LLM Agent 在未知环境中缺乏机制发现能力——前沿模型在从头学习全新环境时仍远逊于人类。
- **现有方法不足**：当前 Agent 将所学记录为 prose（自然语言文本）存放在 context window 或可编辑记忆中；上下文压缩/记忆重写后 prose 会退化，导致 Agent 在复杂新环境中迷失。
- **根本差距**：人类通过观察→实验→推理→抽象为可复用理论来学习，而 Agent 缺乏将经验蒸馏为持久、可执行、可验证表示的机制。
- **目标**：让 Agent 用计算机程序（而非自然语言）显式编码环境表示与转移函数，实现知识的累积与复用。

## 核心贡献（创新点）
1. **提出交互程序归纳（Interactive Program Induction, IPI）范式**：将程序归纳与动作选择统一于单个 LLM Agent，贯穿 hypothesize / certify / plan / act with verification 四个基本操作。*与已有工作本质区别：以往 program induction 多用于离线示例推导，本文将其置于在线交互闭环中，由 Agent 自主决定探索什么、如何修改模型。*
2. **构建 Schema harness**：提供持久程序工作空间与四个接口（写程序、回测历史、程序内搜索、逐动作验证执行）。*与 Code World Models（WorldCoder/CWM/TheoryCoder）本质区别：后者将 LLM 作为预定义学习流程中的算子，Schema 把模型修订、实验设计、规划的全部控制权交给 Agent，状态表示、转移规则、任务目标均由 Agent 自举发现。*
3. **多项基准上达到人类级表现**：ARC-AGI-3 RHAE 58.7%→99.2%（同 base model）、DiG-bench 100% 解出所有公开游戏、MazeBench 超越前50人类中位数。*与并发工作（EWM/OPINE-World/NOOA/Tycho/Twin）定位差异：本文以 IPI 为通用 agent 设计范式，在 ARC-AGI-3 之外额外覆盖 DiG-bench（纯机制发现）与 MazeBench（超长 horizon 知识复用），并给出逐组件 ablation 与行为 trace 分析。*

## 方法详解
Schema 由一个持久工作空间 + 四个操作接口构成，基础模型保持冻结，知识积累体现在可执行程序的迭代修订中。

1. **Hypothesize（假设）：将机制编码为程序 P**
   - 程序包含两部分：(i) **State grounding**——将观测到的对象、属性、关系建模为结构化表示，并携带早期交互中的推断信息；(ii) **Transition rules**——编码动作与对象交互对状态的影响。
   - 通用预测接口：
     $$(\hat{o}_{t+1}, z_{t+1}) = P(z_t, o_t, a_t)$$
     其中 $z_t$ 为程序内部状态，$\hat{o}_{t+1}$ 为对下一观测的预测（含任务完成/失败反馈）。
   - Agent 通过文件管理工具修订程序的状态表示和转移规则，harness 将每次观测到的转移追加到交互历史 $\mathcal{H}_t$。

2. **Certify（认证）：用历史验证程序一致性**
   - 回测（backtest）工具重放历史交互，逐条计算预测 $\hat{o}_{i+1}^P$ 并与真实 $o_{i+1}$ 比较，报告 mismatch 集合：
     $$\mathcal{M}(P, \mathcal{H}_t) = \{ i < t : \hat{o}_{i+1}^P \neq o_{i+1} \}$$
   - 一致性判据：$\text{Cert}(P, \mathcal{H}_t) \iff \mathcal{M}(P, \mathcal{H}_t) = \emptyset$。
   - 关键优势：**修改程序时必须重新通过全量回测**，保证新知识不会与已学机制冲突（如 AR25 案例中未加认证的修订使历史一致率从 97.7% 跌至 65.2%）。

3. **Plan（规划）：在程序内搜索**
   - 程序同时充当模拟器：将预测观测/状态喂回同一接口即可 rollout 而不消耗环境动作。
   - Agent 指定目标谓词 $g$（任务完成 / 中间 waypoint / 待检验机制的情境），选择搜索过程 $\sigma$ 和计算预算 $b$：
     $$\pi \leftarrow \text{SEARCH}_{\sigma}(P, z_t, o_t, g; b), \quad \pi = (a_t, \dots, a_{t+k-1})$$
   - Harness 内置 BFS / DFS / A\* / greedy 等搜索工具，Agent 也可自行编写并加载。

4. **Act with Verification（带验证的执行）**
   - 受模型预测控制启发：在每步执行前用 P 预测 $\hat{o}_{t+1}$，执行 $a_t$ 后将真实转移追加到 $\mathcal{H}_t$。
   - 预测一致则继续执行，**首个 mismatch 即中断计划**，将意外转移返回给 Agent 用于修订 P。
   - 无 verification 的对照：在 LS20 level 2 中，Agent 错误假设补给站可复用，继续执行 42 步直到耗尽能量（此类情况占 58% 的多余动作）。

## 实验与结果
- **ARC-AGI-3**（25 public games, 183 levels）：
  - 主要指标 RHAE（relative human action efficiency，结合完成度与动作效率，超过人类则平方惩罚）。
  - Fable 5 基准：Basic harness 1.2% → Coding harness 58.7% → **Schema 99.2%**；18/25 游戏在 57% 人类动作内完成。
  - Sol (max effort)：Coding harness 62.3% → **Schema 96.7%**，全部 25 游戏通过。
  - Opus 4.8（10  hardest games, 75 levels, 2000 动作/游戏预算）：Schema 72.9% RHAE；prose model 58.8%；w/o certification 62.3%；w/o planning 59.2%；w/o verification 52.4%（Table 1）。

- **DiG-bench**（21 public games, 7 tiers）：
  - GPT-6 Astra medium/high/max effort。
  - Medium effort：basic harness 11/21 games won，Schema **19/21**；lives lost 96 → **34**（≈65% 下降）。
  - Tiers 6–7（最难两级）：basic harness max effort 66.7% → Schema max effort **100%**。
  - 每清一关平均 token 成本 \$3.40 vs basic \$4.00（约降 15%）；前期多投入后期复用机制带来回报。

- **MazeBench**（256 rooms, 100 gems，36h 运行）：
  - Schema (GPT-6 Astra)：33 gems、139 rooms、27,819 actions；**超越 top-50 human median**（29 gems / 137 rooms）。
  - 对比 Codex (GPT-6 Astra)：在相同 20,673 动作下 Schema 30 gems / 133 rooms vs Codex 23 gems / 113 rooms。
  - 10,000 步后后续 10,000 步：Schema 新增 38 rooms / 13 gems；Codex 仅 18 rooms / 5 gems。
  - 程序 compactness：2,012 非注释行 vs prose 6,133 行 notes；约 25,000 步后仍能复用 ~2,600 步学到的 "ice sliding" 规则。

## 相关工作脉络
1. **ReAct / CodeAct / SWE-agent**：提供交互循环与工具接口，但未显式维护环境机制的可执行理论；Schema 把 "persistent executable theory" 作为核心对象。
2. **Reflexion / ACE**：通过 verbal reflections / playbooks 积累策略，属于 prose 形式；Schema 用程序替代自然语言，确保可回测、可模拟。
3. **Voyager / Self-improving harness（Darwin Gödel Machine / Prime Agent）**：扩展 Agent 可修订对象（技能库、prompt、memory），但程序归纳的主导权仍在预定义流程；Schema 由 Agent 自主驱动修订、实验设计与规划。
4. **WorldCoder / CWM / TheoryCoder / PoE-World**：用 LLM 合成/修订 Python 世界模型；这些工作多在预设 state 表示上学习 dynamics，且学习过程固定；Schema 让 Agent 自举 state grounding、transition rules 与 task goal。
5. **DreamCoder / 编程-by-example / sketching**：从 I-O 示例归纳程序；Schema 将其扩展到在线交互场景，每条环境转移既是示例也是 counterexample，并用全历史回测防止局部修复破坏已有知识。
6. **理论强化学习（theory-based RL, Tsividis et al. 2021）**：早期用符号规则指导探索规划；Schema 结合 LLM 的泛化能力，将符号化理论构建完全交给 Agent 在交互中动态形成。

## 局限性与未来方向
- **基础模型依赖**：效果仍建立在 frontier models（Fable 5 / GPT-6 Astra / Opus 4.8）之上，中小模型在 IPI 下的表现未系统评测。
- **程序复杂度**：MazeBench 最终程序达 2,012 行，面对更大/更异构环境可能遭遇程序膨胀或合成爆炸。
- **搜索预算敏感**：Plan 步骤的性能与 $\sigma$、$b$ 等超参密切相关，Auto-tuning 未深入探讨。
- **横向迁移**：三个基准均为 discrete/grid/puzzle 类环境，连续控制或自然语言密集型任务（如 OSWorld、Terminal-bench）下的泛化需进一步验证。
- **未来方向**：跨环境复用已归纳程序、自动抽象出可组合子程序（类似 DreamCoder 的 library learning）、将 IPI 接入更大规模真实任务（如 robotics、embodied AI）。

## 研究启发与可借鉴点
1. **"可回测" 是知识累积的刚需**：Certify 接口要求每次修改通过全量历史一致，这一约束可迁移到任何需要长期记忆的 agent 系统（如 MemGPT-style 工作流），防止"局部优化破坏全局"。
2. **程序作为世界模型比 prose 更适合 planning**：Plan 模块中程序同时充当 simulator 和 planner 目标空间，直接支持 A\* / BFS 搜索；可将此设计用到代码生成 agent、机器人 task planning 等场景。
3. **Act with verification 是低成本试错的实用模式**：受模型预测控制启发的逐动作 prediction check 可自然融入任意 tool-use agent，当出现意外观测时立即中止并触发修订循环。
4. **I-O 一致性 + 主动实验**：DiG-bench 案例展示 Agent 如何用 practice queries 系统性地排除竞争假设（而非试错乱猜），这一 active experimentation 流程可推广到 scientific discovery agent。
5. **Context compaction 下的知识保留**：MazeBench 中程序历经 >20 次 context compaction 仍能复用 2,600 步前学到的 ice sliding 规则，证明程序化表示对压缩友好，值得在长对话 / 长 horizon agent 中采用。

## 关键术语表
- **Interactive Program Induction (IPI)**：由 Agent 主导的"假设→验证→规划→执行"闭环范式，把环境机制显式编码为可执行程序并在交互中持续修订。
- **Schema harness**：实现 IPI 的 agent 基础设施，含持久程序工作空间与 hypothesize / certify / plan / act with verification 四组接口。
- **RHAE (Relative Human Action Efficiency)**：ARC-AGI-3 主指标，综合任务完成度与相对人类的动作效率（超过人类动作量则施加二次惩罚）。
- **State grounding**：程序中将原始观测映射到结构化对象/属性/关系的子程序，决定世界模型的表示粒度。
- **Certify / backtest**：重放历史交互、逐条比对程序预测与真实观测，报告 mismatch 以验证程序一致性。
- **Act with verification**：受模型预测控制启发的执行策略，每步执行前用程序预测，首个 mismatch 即中断并返回新证据。
- **WorldCoder / CWM / TheoryCoder**：先前的 code world model 工作，用 LLM 合成/修订 Python 世界模型；Schema 比它们更强调 Agent 主导与自举。
- **DiG-bench / MazeBench**：两款面向 mechanism discovery 与 long-horizon exploration 的 benchmark，分别基于文本游戏与 3D 迷宫。

## 可复现要素
- **数据集**：ARC-AGI-3（25 public games）、DiG-bench（21 public games）、MazeBench（ASCII track）；论文未声明数据再分发限制外的开源要求，Benchmark 本身为公开评测平台。
- **代码/权重**：Project website https://schema-harness.github.io/；论文正文未明确声明代码开源链接，需查看该网站或仓库确认。Base models（Claude Fable 5 / GPT-6 Astra / Opus 4.8 / Sol）均为闭源商用模型。
- **关键超参**：planning 搜索预算 $b$、搜索过程 $\sigma$（BFS/A\*/greedy 等）、每游戏动作预算（消融实验 2,000 动作/游戏）、MazeBench 36 小时运行时长、reasoning effort（medium/high/max）；论文附录有较详细实现说明，具体数值见 Appendix B/C。
