---
title: "VeriCUA-Verification-as-A-sub-g-sub-entic-Ex-sub-p-sub-lorat"
source: https://arxiv.org/pdf/2610.40578v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:13:44"
field: "计算机使用代理评估与验证"
keywords: ["computer-using agents", "verification", "reward modeling", "agentic exploration", "evidence gathering", "rubric synthesis"]
innovations: ["将CUA验证建模为代理式主动探索的通用框架", "环境感知的Rubric合成与attested/asserted证据分类", "自适应混合动作空间统一GUI与代码态检查"]
benchmarks: ["AndroidWorld", "OSWorld", "ScienceBoard", "WebArena", "OSReward"]
---

# 论文速读：VeriCUA-Verification-as-A-sub-g-sub-entic-Ex-sub-p-sub-lorat

## 一句话总结
VeriCUA 将计算机代理（CUA）的任务完成验证建模为**代理式主动探索**，通过"上下文感知→Rubric 构建→自适应混合动作空间证据采集→规则仲裁"的统一工作流，在在线/离线场景下实现跨平台的高精度验证，显著降低误报率。

## 研究问题与动机
1. **CUA 验证是数据标注、评估和强化学习的关键环节，但现有方法依赖 VLM judge 或自动生成的检查脚本**，两者都只能访问数字环境的"部分视图"，可能遗漏决定性证据。
2. **VLM judge 受限于提供的观测帧**（如仅看终态截图），无法确认操作是否持久化、内容是否保存；生成的检查程序则可能查询错误的状态位置。
3. **平台原生信号（特定应用验证器）** 虽更可靠，但需要为不同平台和访问设置分别定制，工程成本高。
4. **任务完成声明可能是假的**（如编辑未保存、只完成部分指令却报告完成），需要主动收集多源证据才能做出可靠判断。

## 核心贡献（创新点）
1. **将验证形式化为代理式主动探索**：与 TRM、CUAJudge 等一次性 VLM judge 不同，VeriCUA 在构建 Rubric 后主动迭代选择 GUI 或代码工具检查未解决需求，而非单次调用定夺。
2. **环境感知 Rubric 合成**：与 AdaRubric 等方法直接从 instruction 生成检查标准不同，VeriCUA 先探索可用环境/录制中的 context（包名、目录、schema、a11ytrees 等），再构造可操作的需求列表，避免对应用存储路径的错误假设。
3. **自适应混合动作空间**：统一了三类证据源（Trajectory 轨迹记录、Surface 当前界面、Persisted State 持久化存储），而已有工作通常只利用单一视图（如仅 GUI 或仅代码查询）。
4. **源感知证据纪律与规则仲裁**：区分 attested（执行读取得到的可信证据）与 asserted（代理声称的证据）两类观察，要求 hard 需求必须被 attested 支持才算 met，多独立探针 corroborating refutation 才可判 unmet，确保裁决可追溯。

## 方法详解
**共享工作流**（在线/离线均适用）：

1. **上下文检查（Context Inspection）**：探索可用源 $\mathcal{S}=\{S_1,\ldots,S_m\}$（轨迹记录、a11ytrees、机器快照、运行时状态），构建 context $\mathcal{C}$。
2. **Rubric 合成**：结合 instruction $g$ 和 context $\mathcal{C}$，生成可检查需求集合 $R=\{(r_i, h_i)\}$，其中 $h_i\in\{\text{hard, soft}\}$ 区分硬性/软性要求。
3. **主动探索（Active Exploration）**：每步选择检查动作 $\alpha_k = (\upsilon_k, x_k) \sim \pi_V(\cdot \mid g, \mathcal{C}, R, \mathcal{E}_{k-1})$，执行后记录 $(\tilde{o}_k, t_k)$ 加入证据库 $\mathcal{E}_k$，直到硬性需求全部解决或预算耗尽。
4. **规则仲裁**：基于源约束和相关规则，将每个需求状态标记为 met/unmet/unknown；所有 hard 需求均 met → success，任意 hard 需求 unmet → fail，其余 abstain。

**混合动作空间**包含三种视图：
- **Trajectory**：搜索 action log、检查帧、读取 a11ytrees
- **Surface**：新截图、UI dump（volatil 态）
- **Persisted State**：file read、sqlite_query、dumpsys 等（persistent 态）

**证据纪律**：每条 finding 必须关联一次实际执行的读操作，并标注 source attributes（如帧时序、持久性）；attested 证据可单独确认需求，asserted 证据（如代理完成声明）不能单独标记 met。

## 实验与结果
**数据集**：
- 在线：AndroidWorld（116 条）、OSWorld（361 条）、ScienceBoard（169 条）
- 离线：WebArena（715 条）、OSReward（1019 条，Hard 子集 284 条）

**基线**：VLM Judge（TRM）、Auto-Generated Verifier（EvoCUA）、CUAJudge、OS-Themis

**主要结果（在线）**：
| 环境 | VeriCUA Acc | 最高基线 Acc | False Positive ↓ |
|---|---|---|---|
| AndroidWorld | **90.5%** | 82.4%（AutoGen） | **7.7%**（vs 16.9%） |
| OSWorld | **79.5%** | 75.9%（AutoGen） | **13.9%**（vs 25.9%） |
| ScienceBoard | **86.4%** | 76.9%（AutoGen） | **5.4%**（vs 21.7%） |

**离线最强结果（Table 2）**：
- OSReward-Hard：VeriCUA Acc=**70.4**（vs 最强 OS-Themis 65.7），BalAcc=**68.3**
- OSReward 全量：Acc=**87.2**，BalAcc=**86.3**
- WebArena：Acc=**82.7**，BalAcc=**83.1**
- 在 OSReward-Hard 上，硬子集的增益幅度大于全量，说明主动探索对难样本效果显著

**关键分析**：
- 添加代码态检查使 AndroidWorld 决策率从 45% 提升至 82%，Recall 从 57.6% 升至 87.9%，False Positive 不变（3/41）
- 多 pass 组合（3 pass + vision fallback）Decision Rate 和 Agreement 均达 87%，Decided Acc=96%
- 检测到约 **80% 的虚假完成报告**，ScienceBoard 上差异最大
- 用 gemini-3.1-pro 替代 flash 后，OSReward-Hard 准确率进一步提升（见 Table 4）

## 相关工作脉络
1. **VLM Judge 路线**（TRM, Pan et al. 2024）：单次调用对给定观测评分，无主动证据收集；VeriCUA 与其本质区别在于迭代探索和源感知仲裁。
2. **自动生成检查器**（EvoCUA, Xue et al. 2026b）：按任务生成一次执行脚本；VeriCUA 通过 context inspection 确保查询位置正确，false positive 仅为前者的 1/4。
3. **结构化 Rubric 方法**（AdaRubric, Ding 2026; CUAJudge, Xue et al. 2026a）：从 instruction 直接生成 checklist，但不依赖环境探索；VeriCUA 的 Rubric 是在了解应用结构后才合成的。
4. **Milestone Critic 路线**（OS-Themis, Li et al. 2026）：检查有序里程碑；VeriCUA 不预设固定里程碑，而是根据环境动态生成可检查需求。
5. **Evidence-first Agentic Verification**（Sanaeyei et al. 2026）：分离调查与验证阶段；VeriCUA 共享同一工作流处理在线/离线，并通过 source attributes 提供溯源。

## 局限性与未来方向
1. **成本较高**：每次验证执行数十次 tool call 和 model call，token 消耗约 150K–378K（离线）/ 39K（在线），比单次 judge 高一个数量级，不适合交互式场景。
2. **离线场景存在 abstention**：当录制信息不足以解决需求时，VeriCUA 会 abstain（按 fail 计），依赖初始 judgment fallback（约 23% WebArena 轨迹触发）。
3. **在线访问依赖 Instrumentation**：需对目标应用授予 root/ADB 权限以读取持久化状态；对不可 instrument 的环境覆盖率受限。
4. **随机性**：verifier 基于采样生成 rubric/checker，需报告 draw-to-draw 方差；多 pass 组合规则虽减少错误 fail，但非确定性仍需关注。
5. **潜在滥用风险**：同一机制可用于"认证"真实系统上的代理行为，故设计为纯只读。

## 研究启发与可借鉴点
1. **Context-inspection first 范式**：在生成任何检查逻辑前先探索环境结构（包名、目录、schema），这一设计可迁移到任何需要"理解目标应用"的 Agent 任务（如 Agent 训练数据筛选）。
2. **Attested vs Asserted 证据分类**：将"实际执行读取得到的证据"与"代理声称的证据"明确区分，并规定前者才可单独确认需求——这对设计可审计的 Reward Model 有直接参考价值。
3. **Source attributes 绑定机制**：每条 finding 附带帧时序/持久性等元数据，支持确定性聚合和阈值重评估而不重复探索，可作为可复现验证流水线的设计模板。
4. **多 pass 组合优于多数投票**：combination rule 保留一致判定、分歧则 abstain，比简单多数投票产生更少错误 fail（错误 fail 从 20 降至 2），适合需要高 precision 的筛选场景。
5. **可迁移至 reward modeling**：框架天然支持将证据积累过程转化为细粒度 reward signal，可用于 CUAs 的持续强化学习（continual RL）。

## 关键术语表
**VeriCUA**：将 CUA 任务完成验证建模为代理式主动探索的平台无关框架。
**Rubric**：由 instruction 和 context 共同合成的细粒度可检查需求列表，区分 hard/soft。
**Attested Evidence**：通过实际执行读操作（file read、query 等）获得的可信证据，可独立确认需求。
**Asserted Evidence**：代理自身的完成声明或前置帧等间接证据，不能单独标记需求为 met。
**Hybrid Action Space**：统一 GUI 工具（screenshot、a11yread）和代码工具（file read、sqlite_query）的混合检查空间。
**Source Attributes**：每条观察附带的元数据（如帧时序、持久性），约束该观察能支持何种推断。
**Abstention**：当证据不足以解决所有硬性需求时，VeriCUA 不输出裁决，避免假阳性。
**Evidence-first Verification**：先分离调查（收集证据）与验证（基于证据裁决）阶段，再合并判断的验证范式。

## 可复现要素
- **数据集**：AndroidWorld、OSWorld、ScienceBoard（在线）；WebArena、OSReward（离线）——均为公开 benchmark
- **代码/权重**：论文未提及开源仓库链接
- **关键超参**：离线默认 $k=3$ 审计路径、每路径最多 4 tool call；图片缩放到最长边 1280px，仅保留最近 8 帧；在线 rubric 合成 temperature=0.7，checker 合成 temperature=0.7；每次 model call 180s 超时、最多 6 次重试；无 token cap
