---
title: "ServeLearnBench-How-Well-Can-Agents-Self-Improve-from-Servin"
source: https://arxiv.org/pdf/2610.07792v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:53:23"
---

# 论文速读：ServeLearnBench-How-Well-Can-Agents-Self-Improve-from-Serving

## 一句话总结
本文提出 ServeLearnBench，一个用于评估大语言模型 Agent 从服务体验中学习能力的 evolving-environment streaming dataset（EESD）基准，覆盖零售支持、银行审核、销售话术生成三个领域共 53 个环境窗口、4,508 个服务任务和 3,210 个测试任务，首次系统性揭示了 Agent 在隐藏策略下持续学习的能力鸿沟、适应成本与探索瓶颈。

## 研究问题与动机
- 现实部署中 Agent 需要处理的策略、约束和用户偏好会随时间变化，但现行基准要么显式提供目标知识（测试记忆而非发现），要么假设静态环境，要么规模有限，无法系统性衡量 Agent 从交互与反馈中发现、应用和修订**隐藏环境知识**的能力。
- 现有 benchmarks 存在三类不足：① 明示型（如 τ-bench、LLF-Bench）只测保留与应用已知知识，不测从失败中推断；② 发现型但静态（如 DiscoveryWorld、EvoArena）缺少策略演化；③ 支持演化但规模/多样性不足（如 CL-Bench、StreamBench）。
- 亟需一个满足三项标准的理想基准：① 要求从交互与失败中学习（不直接告知知识）；② 环境政策会随时间动态变化；③ 足够规模以区分"发现行为"与"知识获取"两个瓶颈。

## 核心贡献（创新点）
- **提出 EESD 抽象与 ServeLearnBench 基准**：将环境按时间窗口划分，窗口内策略固定但跨窗口可引入/撤回/重激活隐藏规则；与已有 EESD 类工作的本质区别在于评测 slice 将任务严格区分为 Hidden-Dependent（需推断隐藏策略）与 Fully Specified（仅依赖可见信息），从而同时测量"学得多少"与"退化多少"。
- **构建零售/银行/销售三个领域、三难度等级的多面测试床**：9 个 scenario × 53 个环境窗口 × 4,508/3,210 服务/测试任务；区别于 prior 单域/小规模（如 StreamBench 仅连续输入流、无隐藏策略）的设计，首次在同一测试床内测量知识获取（AUC_KA）与行为探索（AUC_E）两条独立能力曲线。
- **系统性评测五种学习 harness 与六种模型的 28 对组合**：覆盖 RAG、Mem0、SkillOpt、Continual Harness (CH)、Prime；区别于 prior 仅比 accuracy 的工作，本文同时报告 token 成本、API 成本、学习速度（AULC₁₀）、Fully Specified 退化、探索多样性与知识保持六类指标。
- **揭示三个关键实证结论**：① 能力鸿沟巨大（Oracle 95.4% vs 最佳 harness 59.6%）；② 持续适应有成本且会损害原有正确行为（GLM-5.3 Flash + CH: Hidden 14.6→49.7，Fully Specified 96.7→82.8）；③ 探索多样性与最终 reward 呈完美秩相关（ρ=1.00），是有效适应的首要瓶颈。
- **开源数据集、代码与评估 dashboard**：GitHub https://github.com/Infini-AI-Lab/ServeLearnBench，Website https://infini-ai-lab.github.io/ServeLearnBench；区别于多数仅公开部分数据的 benchmark，本文公开完整 harness 实现细节与 per-ticket 轨迹。

## 方法详解
- **EESD 形式化**：每个场景定义为环境窗口序列 D = W₀‖W₁‖…‖W_{S−1}，每窗口 W_s 受隐藏状态 θ_s ∈ Θ_g 控制，窗口内固定、窗口边界可演化。每窗口包含服务集与测试集 W_s = D_s^serve ‖ D_s^test，满足 Cov(D_s^serve) = Cov(D_s^test) = C(θ_s) 且两集不交（公式 1），保证测试测迁移而非记忆。
- **任务分类**：Hidden-Dependent（正确行为依赖隐藏状态，Retail/Banking 的规则类与全部 Pitch）vs Fully Specified（仅依赖可见信息，Retail/Banking 单规则可直接判定的任务）。测试同时报告 Avg.H（九场景 Hidden 均值）与 Avg.F（六场景 Fully Specified 均值）。
- **难度等级语义**：L1 测初始获取（新策略首次出现）；L2 测修订与组合（策略收紧/放宽/重作用户偏好，Banking 出现阈值迁移，Retail 出现排序备选）；L3 测非单调演化（策略反转、撤回、再次激活，Retail L3 含 12 条规则的组合与回滚）。
- **五种 harness 设计差异**：① RAG：BM25 检索 8 个最近服务案例注入 prompt，无学习侧模型调用；② Mem0：任务内写记忆 + 任务后记忆提取轮次（extractor model）；③ SkillOpt：每 12 个服务任务为一优化步，前 6 做训练批后 6 做选择集，strict acceptance gate 约 1/9 接受；④ CH：每 25 步 Refiner 调用一次（前 200 步每 25 步、之后每 100 步），四模型调用一次（prompt/memory/skill/subagent）；⑤ Prime：每次任务独立 session，评分后 agent 读 outcome 写 memory/skill/prompt 文件，sandbox + code interpreter。
- **奖励函数**：Retail/Banking 为确定性代码验证（终端数据库 SHA-256 比对 + finish 动作结构化检查，公式 2–3）；Pitch 为 GLM-5.3 Flash judge 结构化 rubric（公式 4–6），按偏好权重覆盖正面属性、惩罚负面/红线属性与 unsupported claims。
- **评估协议**：harness 可见输入严格限制（仅任务可见信息 + 自身 trajectory + 标量 outcome + 单调时间戳 + 自身持久状态），禁止接收窗口标签、隐藏值、策略家族标签、纠正动作；测试任务与反馈不参与任何学习更新。

## 实验与结果
- **评测规模**：6 模型 × 5 harness = 30 对，实际评测 28 对（GPT-5.6 Terra+SkillOpt、Opus 5+SkillOpt 因成本省略），共 252 条学习 lane。
- **能力鸿沟**：Oracle Hidden 平均 95.4%（Retail L3=95.5, Banking L3=99.7, Pitch L3=99.2），Blind 仅 14.1%；最佳 harness（Kimi K3+Prime, Hidden L3=61.7）仍低于 Oracle 的 36.6 个百分点，中位 harness 仅 22.0%。
- **Harness 排名**（四开放权重模型平均）：Prime 61.7 > CH 57.6 > SkillOpt 43.4 > RAG 38.3 > Mem0 37.7。Kimi K3 与 GLM-5.3 表现最强（各 ~49%），GLM-5.3 Flash 最弱但成本最低（$0.010/task）。
- **学习速度 AULC₁₀**：Prime 54.9 (Banking) / 37.7 (Retail)，CH 49.9 / 36.2；SkillOpt 最慢 18.2 / 11.1。
- **Fully Specified 退化**：所有 harness 在 Retail 上均退化（Blind 98.9 vs 89.5–95.5）；除 Prime 95.2 与 Mem0 92.7 外均在 Banking 上退化。GLM-5.3 Flash+CH：Hidden 14.6→49.7，Fully Specified 96.7→82.8（−13.9 点）。
- **探索-奖励完美相关**：AUC_E 秩序与平均 Hidden reward 完全一致（ρ=1.00，跨 harness；ρ=0.80 跨模型），Prime 1.16 > CH 1.02 > RAG 0.88 > Mem0 0.65 > SkillOpt 0.51。
- **知识获取≠探索**：AUC_KA 与 reward 相关仅 ρ=0.60；SkillOpt 发现后保持好（68%）但发现难，RAG 发现易但保持差（38%）。
- **成本**：Prime 最贵（$0.232/task serving + $0.142/test），CH 性价比最优（$0.089/$0.074，reward 55.1/57.7）；Prime 学习侧占比 30–40%，SkillOpt 高达 56–62%。
- **政策撤回测试**：学得快≠撤得快，Prime/CH 撤回速度与学习速度相当（7–10 步达 81–88%），Mem0 撤得慢（4 到 6 步 70% vs 43%），RAG 因 append-only 索引同样慢。
- **训练方法瓶颈**：14B/32B SFT/GRPO 初步试验失败——成功轨迹稀疏，同组全负例时 reward advantage 无区分度。

## 相关工作脉络
- τ-bench（Yao et al., 2025）与 LLF-Bench（Cheng et al., 2023）：前者验证给定策略下的 tool use 最终状态，后者测量语言反馈改进；本文在此基础上增加**隐藏策略、动态演化与跨任务迁移**三重约束。
- StreamBench（Wu et al., 2024）与 LLF-Bench：测量连续改进，但评测对象是显式输入 + 固定 reward；本文引入 Hidden-Dependent/Oracle 双 slice，首次把"发现未知规则"与"应用已知规则"分离测量。
- LifelongAgentBench（Zheng et al., 2025）、AgentCL（Shu et al., 2026）与 ContinualSkillBench（Guan et al., 2026）：测持续能力获取；本文扩展至**同一 agent 在服务过程中在线修订隐式规则**的场景，并报告 Fully Specified 退化指标。
- StateMemBench（Fan et al., 2026）与 CL-Bench（Asawa et al., 2026）：前者验证 memory 是否能区分当前/已废弃事实，后者测 frontier models 在 stateful 环境中的表现；本文进一步把"策略撤回/重激活"纳入时序演化（L3 非单调）。
- EvoArena（Xu et al., 2026）与 EnvHarness（Huang et al., 2026）：动态环境 suite；前者测长期记忆演化、后者把静态 world 转为 adaptive practice；本文的差异化在于**明确定义 EESD 抽象与 Cov 不变性**（公式 1），使得 serve/test 同分布不同实例成为可能。
- GEPA（Agrawal et al., 2026）、SkillOpt（Yang et al., 2026）与 Prime（Karten et al., 2026）：自演化 agent harness；本文是首个将这些 harness 在同一 evolving-policy 测试床上公平对比并同时报告成本/探索/知识保持的工作。

## 局限性与未来方向
- 测试依赖 outcome-only 标量反馈，无 corrected trajectory 信号，限制了 SkillOpt 类方法的 strict gate 接受率（仅 ~1/9）。
- 探索多样性与 reward 的 ρ=1.00 是秩相关，尚未揭示因果机制；不同模型间探索差异更小（0.77–0.92 vs harness 0.51–1.16），harness 设计与模型能力孰轻孰重仍需消融。
- SFT/GRPO 等参数级学习方法因成功轨迹稀疏而失败，表明仅靠 outcome reward 不足以产生 informative gradient，需要过程监督或反事实探索辅助。
- 三领域 53 窗口规模仍有限，L1 仅 4 个非重复 Pitch profile，样本数不足支撑严格统计推断（论文未报告 CI/SE）。
- 未来方向：① 探索-aware harness 设计（将 AUC_E 作为训练/选择信号）；② 知识获取与探索解耦的训练目标（同时优化 AUC_E 与 AUC_KA）；③ 过程反馈（如 step-level correctness）缓解 sparse outcome reward；④ 扩展到多 agent 协作与跨域迁移场景。

## 研究启发与可借鉴点
- **双 slice 评估范式**：Hidden vs Fully Specified 分离测量"学了多少"与"退化多少"，可复用于任何持续学习 benchmark，避免单一 accuracy 指标掩盖 adaptation cost。
- **AUC_E / AUC_KA / AULC₁₀ 三类时序指标**：分别刻画探索广度、知识持久性与学习速度，形成"发现→保持→速度"三维诊断框架，可直接迁移至 self-improving agent 评测。
- **探索多样性作为 harness 设计 target**：ρ=1.00 的强相关提示未来 harness 应优先鼓励 candidate behavior 变异（temperature/sampling/branching），而非仅优化已知正确路径的 retention。
- **Policy 撤回测试（Drop-vs-Learn）**：本文 Figure 9 的"学得快≠撤得快"现象提示 future work 需显式加入 policy retirement 评估，否则隐藏技能会成为 stale belief 负担。
- **Cov 不变性构造 serve/test split**（公式 1）：用同一 hidden policy 覆盖不同 task instance，避免 memorization，可推广至任何 streaming 评测场景。

## 关键术语表
- **EESD（Evolving-Environment Streaming Dataset）**：按时间窗口组织的任务流，窗口内环境状态固定、跨窗口可演化，窗口内服务/测试集共享同一 hidden policy 但实例不交。
- **Hidden-Dependent / Fully Specified**：前者正确行为依赖隐藏环境状态（需推断），后者仅依赖可见任务信息（可直接判定），本文用以分离测量"学习增量"与"行为退化"。
- **AULC₁₀（Average Utility over Learing Curve at 10）**：新策略前 10 次 encounter 的平均正确率，刻画 harness 适应速度。
- **AUC_E（Exploration Area Under Curve）**：同一策略前 K=8 次 encounter 产生的不同 answer 数累积均值，量化行为探索广度。
- **AUC_KA（Knowledge Acquisition AUC）**：首条正确 response 后 5 次后续 ticket 的平均正确率，衡量成功行为是否转化为可复用知识。
- **Serve/Learn split**：服务侧任务产生 outcome feedback 供 harness 学习；测试侧任务冻结 harness 状态并排除于学习之外，保证评测 unbiased。
- **Policy Drop**：已学规则被撤回后 agent 停止应用它的速度，本文与 Policy Learn 对照揭示 stale belief 风险。
- **Cov 不变性**：D_s^serve 与 D_s^test 的行为分量覆盖相同（Cov 相等）但实例不交，确保测试测迁移而非记忆。

## 可复现要素
- **数据集**：EESD 基准开源，9 场景 × 53 窗口 × 4,508/3,210 服务/测试任务；https://github.com/Infini-AI-Lab/ServeLearnBench
- **代码**：benchmark evaluator、harness 适配、per-ticket 轨迹脚本均已开源；附录 D.1–D.4 提供学习速度/探索/成本分解完整计算脚本说明。
- **Website**：交互式 dashboard https://infini-ai-lab.github.io/ServeLearnBench（含 Figure 1/3/4/5/7/8/10/12/13 可视化）。
- **关键超参**：工具调用预算 50/任务、runner guard 100 steps、output token cap 256k；每 task 一次 attempt；Pitch judge 固定 GLM-5.3 Flash + high reasoning efort；SkillOpt 块大小 12（前 6 训练后 6 选择）、CH Refiner 每 25/100 步触发、Prime 每 ticket 独立 session。
- **模型访问**：GPT-5.6 Terra / Opus 5 通过 OpenAI / Anthropic API；其余四模型通过 Fireworks；Fireworks serverless 价格以 2026-09-06 官方定价为准。
- **训练方法**：论文 D.5 附录说明 SFT/GRPO 初步试验仅使用 14B/32B 模型因成功轨迹过稀而失败，未开源训练脚本。
