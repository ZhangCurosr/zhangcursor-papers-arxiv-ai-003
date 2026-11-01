---
title: "R-sub-u-sub-nn-sub-a-sub-bl-sub-e-sub-C-sub-o-sub-mmit-Unt-s"
source: https://arxiv.org/pdf/2610.11593v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:18:13"
field: "AI for Software Engineering"
keywords: ["commit untangling", "coding agents", "runnable commits", "regression repair", "software engineering", "program repair"]
innovations: ["RucTangle：首个保证每次提交后代码可运行的 Agent 驱动提交拆分方法，通过执行反馈闭环修正分组与顺序", "TangleEval：首个同时评测可运行性和下游修复收益的提交拆分框架", "实证证明可运行提交历史可帮助 Coding Agent 修复回归，平均相对提升 16.6% pass@1"]
benchmarks: ["SWE-CI", "TangleEval"]
---

# 论文速读：RucTangle: Runnable Commit Untangling for Coding Agents

## 一句话总结
本文提出了 **RucTangle**，首个保证每次提交后代码仍可运行的 Agent 驱动提交拆分方法，以及 **TangleEval**，首个量化拆分提交历史对代码修复有帮助的评测框架；实验表明，将 RucTangle 生成的历史供修复 Agent 使用可使六个模型中四个的 pass@1 显著提升，平均相对提升 16.6%。

## 研究问题与动机
- **Agent 生成的补丁过于"纠缠"**：Coding Agent 产出的 patch 往往混合多种开发目的（如功能实现 + 重构），且规模是人工补丁的两倍，不利于 Code Review 和 Bug 定位。
- **现有方法忽视可运行性（Runnability）**：既有提交拆分工作（如 SmartCommit、Flexeme、Atomizer）未保证按顺序应用各 commit 后代码仍可运行，但实践中 CI 检查和开发者均会拒绝导致代码不可运行的提交。
- **维护收益缺乏实证验证**：既有研究通过对比拆分结果与开发者原始 commit 的匹配度来间接证明价值，并未直接在真实修复任务上展示拆分后的 commit 历史是否真正帮助后续维护工作。

## 核心贡献（创新点）
1. **RucTangle：首个保证可运行性的 Agent 提交拆分方法**——Agent 分组并提出有序提交计划后，逐 commit 应用并通过测试收集检查可运行性；失败时先尝试重新排序，再启动新 Agent 会话重新分组/拆分，最终仍失败则合并相邻 commit；与仅关注语法匹配的现有方法本质不同。
2. **TangleEval：首个从"实际维护收益"角度评估提交拆分的框架**——同时衡量重建率、可运行性、临时测试失败、提交粒度及功能增益分布，并进一步在回归修复任务上验证历史的效用；不同于以往仅用参考分组匹配度评分的做法。
3. **首次在 coding agent 场景下实证：可运行提交历史能显著提升 Bug 修复性能**——在 453 个引入回归的 patch 上，提供 RucTangle 历史使 Qwen3-Coder-Next 和 Nemotron 的 pass@1 分别提升 4.4pct 和 5.2pct（平均相对 16.6%）。
4. **通过轨迹分析揭示 Agent 如何使用历史**——发现 Agent 会用 commit diff 确认变更位置、用 commit message 判断应保留的行为，但也会被误导性的 commit message 干扰而反复犹豫。

## 方法详解
### RucTangle（四步迭代流程）
1. **提出提交计划**：将 patch 拆分为原子代码变更，Agent 考察变更内容和项目代码，按开发目的分组并提出有序 commit 列表（含 commit message），允许使用 `split` 工具对混合目的的变更进一步拆分，最终通过 `submit` 工具提交计划。
2. **校验计划与代码**：先做结构校验（每个变更恰好分配到一个 commit，每个 commit 至少含一个变更）；再从初始版本 $V_0$ 起依次应用 commit，对每个中间版本 $V_i$ 检查可运行性（以 `pytest` 能否成功收集和加载测试定义：$R(V_i) = 1$）。
3. **修正提交**：
   - 第一步：根据报错信息尝试 **重新排序**（如某模块由后续 commit 才引入，将其提前）。
   - 第二步：启动新 Agent 会话，结合验证错误 **重新分组/拆分/排序**（最多两次修订）。
   - 第三步：若仍不可运行，**合并相邻 commit**（$c_i$ 与 $c_{i+1}$ 合并），逐步向后合并直到版本可运行，保留所有变更。
   - 每轮修正后重新执行 Step 2 校验，全通过后接受。
4. **定位回归**：若 patch 引入回归（$V_\star$ 为 $V_{\text{fail}}$，$T_{\text{fail}}$ 为 $V_0$ 通过、$V_{\text{fail}}$ 失败的测试），使用 `git bisect` 在可运行历史上二分搜索，给出候选引入回归的 commit 及其 message，作为修复 Agent 的上下文。

### TangleEval 评测指标
- **Patch Reconstruction Rate**：输出历史能完整重建原始 patch 的比例。
- **Runnability**：各中间版本通过可运行性检查的比例；以及完全可运行历史（全部 $k$ 个版本均通过）的比例。
- **Intermediate Test Failures（ITF）**：最终版本通过的测试中，在某个 commit 应用后由通过变失败的比例（越低越好）。
- **Commit Count**：平均/中位数 commit 数及单 commit 比例。
- **Test Gain Distribution（MTP / TPE）**：新增通过的测试在不同 commit 上的分布集中度，MTP 越小/TPE 越大表示功能增益越均匀分布在多个 commit 中。
- **Regression Repair（pass@1 / pass@3）**：成功修复所有 $T_{\text{fail}}$ 且不破坏原有通过测试的比率；同时报告 Partial Repair、Full Repair、New Regression 及 Patch Generation 率。

## 实验与结果
- **数据集**：基于 SWE-CI 构建。
  - RQ1（拆分质量）：131 个 Claude Opus 4.6 生成的 patch，来自 100 个 Python 仓库。
  - RQ2（修复辅助）：453 个引入回归的 patch，来自 6 种 Agent（Claude Opus 4.6、GLM-5.1、DeepSeek-V4-Pro、Kimi-K2.6、Qwen3.7-Max、GPT-5.5），共 90 个仓库、108 个任务。
- **对比基线**：FileSplit（按文件）、HunkSplit（按 hunk）、Armchr、Atomizer（Python 适配版）、NoUntangle（整补丁不拆分）。
- **修复 Agent**：Qwen3-Coder-Next、Nemotron-3.5-Lightning-30B-A3B（主实验，各 453 patch × 3 次尝试）；MiMo-V2.6-Flash、Qwen3.7-Flash、DeepSeek-V4-Flash、GPT-6-Luna（150 patch 子集 × 1 次尝试）。
- **RQ1 结果**：
  - RucTangle **100% 重建且 100% 全部版本可运行**；其余基线可运行完整历史比例仅 62.6%–79.4%。
  - RucTangle 的 ITF 仅 **1.7%**，基线为 16.4%–32.4%。
  - RucTangle 的 MTP = 0.552（最低），TPE = 0.562（最高），功能增益分布最均匀；中位数 commit 数 8，单 commit 比例仅 0.8%。
  - 平均耗时 276.3s/patch，低于 Armchr（74.6%）和 Atomizer（18.0%）；平均输出 11.9K tokens。
- **RQ2 结果**：
  - **Qwen3-Coder-Next**：pass@1 从 26.9% → **31.3%**；pass@3 从 40.4% → 43.5%。
  - **Nemotron**：pass@1 从 30.8% → **36.0%**；pass@3 从 41.1% → 47.7%。
  - 6 个模型中 4 个 pass@1 提升，1 个不变（DeepSeek-V4-Flash 61.3%→61.3%），1 个略降（GPT-6-Luna 50.0%→48.7%，因新回归增加）。
  - 平均相对提升 **16.6%**。
- **消融（RucTangle-Original vs Halved）**：将 commit 数减半后 pass@1 从 34.9% 降至 31.8%，新回归率从 25.6% 升至 28.4%，说明更细粒度拆分有一定价值。

## 相关工作脉络
1. **EpiceaUntangler / Flexeme / UTango**：基于时间间隔、词法流、依赖图学习分组的传统方法；与 RucTangle 的差异在于均不使用执行反馈保证可运行性，且未在 Agent 场景下验证维护收益。
2. **SmartCommit**：考虑依赖约束避免编译错误并按序组织 commit；但仅针对 Java 等特定语言设计，不保证运行时可运行性。
3. **Atomizer**：多 Agent LLM 协作的意图驱动拆分，在合成基准上表现最优，但依赖 embedding 相似度（阈值 0.6）且未做运行时校验；RucTangle 通过可运行性检查弥补这一缺口。
4. **Armchr**：结合程序结构与依赖分析的 LLM 辅助拆分；实验中发现其可运行历史比例仅 79.4%，说明依赖分析不足以覆盖运行时状态机约束。
5. **AtomicCommitBench / GitGoodBench**：评测 Agent 重建 commit 历史的基准；前者检验重建与顺序，后者用 LLM 评估，但均未涉及可运行性验证和下游修复任务。
6. **Safe-commit / Crisp / ChgCutter**：识别安全子集或选择相关变更的工具，与 RucTangle 共同强调变更分组中依赖/影响分析的重要性，但同样未将"每次中间版本可运行"作为硬性约束。

## 局限性与未来方向
- **仅面向 Python 项目**：实验局限于 SWE-CI 中的 Python 仓库，其他语言或不同 CI 配置下的适用性尚待验证。
- **不隔离组件贡献**：RucTangle 的修复收益来自"可运行历史 + git bisect 定位"的组合，未单独评估各组件的贡献。
- **部分模型收益不稳定**：DeepSeek-V4-Flash pass@1 无变化，GPT-6-Luna 略降，说明方法对强基线模型帮助有限或可能引入额外噪声。
- **Commit message 质量不可控**：现有实验未独立控制 message 质量，误导性 message 会干扰 Agent 修复决策（如将错误删除误标为"优化"）。
- **未来方向**：训练 History-aware Coding Agent、设计更好的 commit message 生成方法、扩展至多语言和人类开发者协作场景。

## 研究启发与可借鉴点
1. **将执行反馈闭环引入 Agent 软件工程设计**：RucTangle 的"提出→执行检查→根据错误修正"范式可迁移至其他 Agent 生成结构化资产的任务（如 PR 拆分、重构计划）。
2. **可运行性是 commit 分组的硬约束而非软指标**：这一设计原则可直接复用到任何需要顺序执行的变更场景，避免仅靠语义/语法匹配导致中间状态崩溃。
3. **用 git bisect 辅助 Agent 回归定位是一种低成本高效的信息压缩**：将二分搜索结果作为上下文提供给修复 Agent，显著缩小搜索空间，值得在其他 Bug 定位任务中借鉴。
4. **Agent 轨迹分析揭示"误导性历史信息的负面效应"**：commit message 可能与实际代码行为矛盾，未来可在 Agent 训练数据中故意注入噪声以提升鲁棒性。
5. **TangleEval 的多维度评测框架（可运行性 + 临时失败 + 功能增益分布 + 下游修复）**：可作为软件工程 Agent 任务的通用评测模板。

## 关键术语表
- **Commit Untangling**：将混合多种开发目的的纠缠 patch 拆分为每个 commit 聚焦单一目的的较小提交序列。
- **Runnability**：在每次 commit 应用后，`pytest` 仍能成功收集并加载测试的状态；区别于功能正确性，只要求代码可运行不报错。
- **Intermediate Version**：按序应用 commit 序列时，介于初始版本 $V_0$ 和最终版本 $V_\star$ 之间的每一个中间快照。
- **Intermediate Test Failure（ITF）**：最终版本通过的测试，在某个 commit 应用后由通过转为失败的占比，衡量临时回归程度。
- **MTP / TPE**：MTP（Max Test Proportion）衡量功能增益集中在单个 commit 的程度；TPE（Test Pass Entropy）衡量增益在 commit 间的均匀分布程度。
- **Regression**：原本通过的测试在新版本中变为失败，由 patch 引入的回归（introduced regression）。
- **Git Bisect**：在有序版本序列上二分搜索，通过反复运行测试快速定位首次引入失败的 commit。
- **Pass@K**：在 K 次尝试中至少有 1 次成功完成修复的比例；此处成功指修复所有目标回归且不引入新失败。

## 可复现要素
- **数据集**：SWE-CI（HuggingFace，已公开：`skylenage-ai/SWE-CI` 和 `skylenage-ai/SWE-CI-trajectory`）；本文构造的两个子数据集未声明独立公开，但基于公开数据导出。
- **代码/权重**：论文未明确声明开源仓库；使用了 mini-swe-agent（GitHub 开源）、vLLM（开源）、OpenRouter API 及公开大模型（Qwen3、Nemotron、DeepSeek、GPT-6-Luna 等）。
- **关键超参**：每轮 Agent 最多 80 步；temperature=0，top-p=1；Atomizer 使用 BAAI/bge-base-en-v1.5 嵌入，余弦相似度阈值 0.6 合并分组；commit 数上限 20；修复尝试次数主实验 3 次/patch，额外模型 1 次/patch。
