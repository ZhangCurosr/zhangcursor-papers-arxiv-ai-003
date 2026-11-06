---
title: "VeriCUA-Verification-as-A-sub-g-sub-entic-Ex-sub-p-sub-lorat"
source: https://arxiv.org/pdf/2610.40578v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:13:49"
field: "计算机使用代理评估与验证"
keywords: ["computer-using agents", "verification", "agentic exploration", "reward modeling", "evaluation", "VLM judge"]
innovations: ["将CUA验证形式化为代理式主动探索而非单次Judge调用", "环境感知的Rubric构建:先探查应用上下文再生成细粒度检查需求", "源感知证据纪律:区分attested/asserted证据并以源属性约束仲裁规则"]
benchmarks: ["AndroidWorld", "OSWorld", "ScienceBoard", "WebArena", "OSReward"]
---

# 论文速读：VeriCUA: Verification as Agentic Exploration for Computer-Using Agents

## 一句话总结
VeriCUA 将计算机使用代理（CUA）的任务完成验证建模为**代理式探索**过程，通过先检查上下文构建细粒度评分准则（rubric），再自适应选择 GUI/代码工具收集互补证据，最终以显式证据规则裁决任务成功与否；在多个平台（AndroidWorld、OSWorld、ScienceBoard、WebArena、OSReward）的在线与离线验证中均超越了 VLM Judge、Auto-Generated Verifier 等基线，尤其在离线困难样本上提升显著。

## 研究问题与动机
- **现有验证方法依赖局部视图**：VLM Judge 仅能基于已提供的观察片段做判断，而自动生成的检查程序（generated checkers）依赖可访问的机器状态，两者都可能遗漏关键证据或导致裁决无支撑。
- **平台定制成本高昂**：针对不同平台（桌面/移动/Web）和访问设置分别定制验证器，工程成本高且难以复用。
- **轨迹可能"看似成功实则失败"**：代理报告完成任务，但编辑未保存、操作静默失败，或仅满足指令的一部分即可被误判为成功，导致假阳性。
- **缺乏统一的验证范式**：现有方法难以同时处理离线（基于录制轨迹）与在线（基于执行后机器状态）两种场景，也无法灵活利用轨迹日志、屏幕帧、无障碍树、文件/数据库等不同来源的互补证据。

## 核心贡献（创新点）
1. **将 CUA 验证形式化为代理式探索**：提出一个统一的验证工作流，将离线轨迹审计与在线环境检查纳入同一范式，通过主动证据收集替代单次 Judge 调用。
2. **环境感知评分准则构建（Context-Aware Rubric Synthesis）**：验证器在构建需求清单前先探查应用上下文（包名、目录、数据库 Schema 等），使需求具有应用特异性，避免对不可见文件路径或界面元素的错误假设。
3. **自适应混合动作空间（Adaptive Hybrid Action Space）**：验证器可根据未解决需求在 GUI 工具（截图、OCR、区域放大）与代码工具（文件读取、SQLite 查询、ADB dumpsys）之间动态选择检查路径，实现互补证据的交叉验证。
4. **源感知证据规则仲裁（Source-Aware Evidence Discipline）**：通过区分"认证证据（attested）"与"主张证据（asserted）"，并结合源属性（帧时序、持久性），以确定性规则组合多源发现，输出 met/unmet/unknown 三级判定。
5. **跨平台系统评估与错误标签审计**：在 5 个基准/平台上验证方法有效性，并发现 VeriCUA 与 gold label 不一致的轨迹中，约 63% 经盲审模型仲裁后被支持，提示部分 benchmark 标签存在错误。

## 方法详解

**整体流程**（图 1）：验证在代理执行结束后启动，分四阶段：
1. **Context Inspection**：收集应用上下文 $\mathcal{C}$（在线：包/目录/Schema；离线：记录步骤/帧/a11y 树）。
2. **Rubric Synthesis**：利用指令 $g$ 和上下文 $\mathcal{C}$ 生成需求集合 $R = \{(r_i, h_i)\}_{i=1}^n$，其中 $h_i \in \{\text{hard}, \text{soft}\}$ 区分强制/非强制需求。
3. **Active Exploration**：验证器迭代选择检查动作 $\alpha_k = (\upsilon_k, x_k)$，执行后获得观察 $\tilde{o}_k$ 和证据类型 $t_k$：
$$\alpha_k \sim \pi_V(\cdot \mid g, \mathcal{C}, R, \mathcal{E}_{k-1}), \quad \text{Execute}(\alpha_k, S) \to (\tilde{o}_k, t_k)$$
每次检查聚焦于尚未解决的 hard requirement，探索预算耗尽或全部 hard 需求已裁决后停止。
4. **Rule-Based Arbitration**：根据源约束（attested vs asserted）和相关证据规则组合发现，输出 $\text{success}$ / $\text{fail}$ / $\text{abstain}$。

**三种证据视图**（图 2）：
- **Trajectory**：保留的帧、动作日志、推理记录，可用 find_steps / inspect_action 搜索。
- **Surface**：当前机器界面（新截图、UI dump），提供新鲜观察。
- **Persisted State**：文件、数据库、存储，通过 file_read / sqlite_query 等代码工具读取，提供持久化证据。

**证据纪律**：
- 每个发现必须引用已执行的检查调用，harness 验证引用有效性并附加源属性（如帧时序、持久性标签）。
- **Attested 证据**：post-action 帧、文件读取、数据库查询等客观观测，可单独判定 requirement met。
- **Asserted 证据**：代理的完成声明、pre-action 帧，只能引导进一步检查，不能单独标记为 met。
- **仲裁规则**：hard requirement 需 attested support 才能 met；unmet 需来自不同 probe 的 corroborating refutation；否则 unknown。所有 hard met → success；任一 hard unmet → fail；有 unknown 但无 unmet → abstain。

**在线/离线实例化**：
- 在线：GUI 检查 + 代码核查结合，unresolved 任务 abstain（计入 fail）。
- 离线：最多 $K=3$ 条审计路径，forward inspection 搜集支持，backward inspection 搜寻反驳；unresolved 任务使用初始单次 Judge 作为 fallback。

## 实验与结果

**数据集与环境**：
- **在线**：AndroidWorld（116 条，M3A Agents）、OSWorld（361 条，成功率 44.3%）、ScienceBoard（169 条，成功率 23.7%）
- **离线**：WebArena（715 条，3 种 rollout agent）、OSReward full（1019 条，人工标注）、OSReward-Hard（284 条）

**基线**：VLM Judge (TRM)、Auto-Generated Verifier、CUAJudge、OS-Themis

**在线结果**（Table 1）：
| 环境 | 方法 | F1 | Accuracy | False Positive Rate |
|---|---|---|---|---|
| AndroidWorld | VeriCUA | **89.2** | **90.5** | **7.7%** |
| | Auto-Generated | 81.2 | 82.4 | 16.9% |
| | VLM Judge | 74.2 | 78.4 | 15.4% |
| OSWorld | VeriCUA | **75.5** | **88.7** | **13.9%** |
| | Auto-Generated | 74.2 | 75.9 | 25.9% |
| ScienceBoard | VeriCUA | **67.6** | **91.6** | **5.4%** |
| | Auto-Generated | 59.8 | 76.9 | 21.7% |

VeriCUA 在三个在线环境中均取得最高 accuracy 和最低假阳性率，ScienceBoard 上假阳性率仅为生成验证器的 1/4。

**离线结果**（Table 2）：
| 基准 | TRM | CUAJudge | OS-Themis | **VeriCUA** |
|---|---|---|---|---|
| OSReward-Hard Acc | 57.4 | 63.4 | 65.7 | **70.4** |
| OSReward Acc | 82.2 | 83.7 | 80.2 | **87.2** |
| WebArena Acc | 79.6 | 81.3 | 75.4 | **82.7** |
| AndroidWorld Acc | 88.3 | 91.2 | 89.4 | **94.3** |
| OSWorld Acc | 80.3 | 83.4 | 82.1 | **86.3** |

在 OSReward-Hard 子集上优势最大（+4.7 vs OS-Themis），表明主动证据收集对困难样本增益显著。

**额外分析**：
- 混合动作空间（GUI+代码）相比纯 GUI 将 decision rate 从 45% 提升至 82%，recall 从 57.6 升至 87.9，假阳性不变。
- 多 pass 组合（默认 3 pass + vision fallback）将 agreement 提升至 87%、decision rate 至 87%、decided acc 至 96%。
- VeriCUA 拒绝约 80% 的假完成报告（Figure 5b），在 ScienceBoard 上差距最大。
- 切换至 gemini-3.1-pro 后 OSReward-Hard accuracy 进一步提升（Table 4）。

## 相关工作脉络
1. **VLM Judge / TRM**（Pan et al., 2024; Sun et al., 2025）：单次调用基于给定观察评分，无法主动获取更多证据，是 VeriCUA 的主要离线基线。
2. **Auto-Generated Verifier / EvoCUA**（Xue et al., 2026b）：为每个任务生成 evaluator 脚本直接查询机器状态；缺乏上下文探查导致可能查询错误状态位置，VeriCUA 通过 context inspection 改善 checker 构建。
3. **CUAJudge / OS-Themis**（Xue et al., 2026a; Li et al., 2026）：基于 checklist/milestone 的单次结构化判断；VeriCUA 通过多步探索和互补证据来源提升可靠性。
4. **Evidence-First Agentic Verification**（Sanaeyei et al., 2026）：分离调查与验证阶段，检查声称是否得到状态/trace 支撑；VeriCUA 在思想路线上一致，但进一步区分 attested/asserted 证据类型并以源属性约束仲裁。
5. **Agentic Reward Modeling / SeekJudge**（Cui et al., 2026; Wan et al., 2026）：通过迭代交互收集证据验证 GUI agent；VeriCUA 强调平台无关的统一工作流，支持离线录制回放和在线机器状态双重模式。

## 局限性与未来方向
- **计算成本较高**：每次验证 pass 发出数十次 tool call 和 model call，token 消耗约为单次 Judge 的 5–10 倍（在线 39.2K tokens/episode，离线 150K–378K tokens/trajectory），不适合交互式实时使用，更适合数据筛选和奖励分配。
- **离线覆盖率受限**：当录制轨迹未包含某需求的证据时，VeriCUA 只能 abstain 或回退到初始 Judge 判断（WebArena 中 23% 轨迹触发 fallback）。
- **在线访问依赖 instrumented 环境**：需要 root/ADB 权限或可访问的机器状态接口，对无法 instrumentation 的应用受限。
- **随机性**：verifier 基于 LLM 采样，不同 draw 可能有分歧，需要报告 draw-to-draw spread。
- **潜在滥用风险**：同一机制可用于"证明"在实际系统上操作的代理行为合规，因此 VeriCUA 严格限制为只读操作。

## 研究启发与可借鉴点
1. **"探索-裁决"两阶段范式**：将验证拆分为主动证据收集和确定性仲裁两阶段，而非单次 Judge 调用，可显著提升裁决可靠性，尤其适用于需要多维度证据的任务（如文件持久化、数据库写入验证）。
2. **Context Inspection 先于 Rubric 构建**：先探查应用上下文再生成检查需求，可有效避免基于假设的文件路径/界面元素查询，这一思路可迁移到任何需要理解应用语义的评估场景。
3. **Attested vs Asserted 证据分类**：通过源属性严格区分客观观测与代理声称，并以不同证据权重进行仲裁，为构建可解释、可审计的评估框架提供了规范设计。
4. **互补证据源的交叉验证**：GUI 观察与代码状态读取回答不同问题（可见内容 vs 持久化效果），多源交叉可大幅降低假阳性，此设计可用于构建更鲁棒的 Agent 评估管道。
5. **Gold-label Audit 作为副产品**：VeriCUA 与 gold label 的 disagreement 经盲审后约 63% 支持 VeriCUA，提示验证器本身可作为 benchmark 标签审计工具，值得在其他 Agent 基准上复现验证。

## 关键术语表
- **Computer-Using Agent (CUA)**：能够在桌面、移动或 Web 环境中通过软件接口执行任务的 AI 代理。
- **Rubric（评分准则）**：由指令和应用上下文派生的细粒度可检查需求集合，区分 hard（强制）和 soft（可选）需求。
- **Attested Evidence（认证证据）**：来自客观机器状态读取的证据（如文件读取、数据库查询、post-action 帧），可独立支撑裁决。
- **Asserted Evidence（主张证据）**：来自代理自身报告或 pre-action 观察的证据，需与其他 attested 证据交叉验证才能生效。
- **Active Exploration（主动探索）**：验证器根据未解决需求动态选择下一步检查动作的迭代过程，而非静态分析固定轨迹。
- **Hybrid Action Space（混合动作空间）**：验证器可同时使用 GUI 工具（截图/OCR/UI dump）和代码工具（文件/数据库/ADB）两类检查手段。
- **Evidence Discipline（证据纪律）**：绑定每个发现到具体执行检查、附加源属性、并按 attested/asserted 类型和应用约束规则进行仲裁的规范体系。

## 可复现要素
- **数据集**：AndroidWorld（公开）、OSWorld（公开）、ScienceBoard（公开）、WebArena（公开）、OSReward（公开）；论文未声明自行开源新数据集。
- **代码/权重**：论文未明确声明代码开源，实验使用 Gemini 3 Flash Preview / Gemini 3.1 Pro 作为 backbone。
- **关键超参**：离线审计路径数 $K=3$；每路径最多 4 次 tool call 搜集支持 + 至少 4 次搜寻反驳；框架温度：rubric synthesis 主调用 temperature=0，候选生成 temperature=0.7；checker 合成 temperature=0.7；单 pass 预算约 180s timeout + 6 次重试；每次调用无 token cap。
