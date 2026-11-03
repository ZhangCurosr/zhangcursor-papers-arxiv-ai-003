---
title: "Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable"
source: https://arxiv.org/pdf/2610.17893v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:34:43"
field: "多智能体LLM协作训练"
keywords: ["multi-agent collaboration", "LLM post-training", "reinforcement learning", "information reward", "self-play", "transfer learning"]
innovations: ["基于团队惊讶度减少的可验证信息奖励机制", "在可计算环境中通过自博弈训练协作行为并迁移到 unseen 任务", "token级信用分配结合保留通用能力的双正则化训练策略"]
benchmarks: ["HiddenBench", "SILO-BENCH", "SplitQA", "MathChat", "BenchForm", "Einstein Puzzles", "Collaboration Tax"]
---

# 论文速读：Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable

## 一句话总结
论文提出 CoSIR（Collaboration via Surprisal-based Information Rewards），一种基于自博弈的后训练方法，在简化的评分表环境中通过可计算的贝叶斯信息奖励训练 LLM 代理分享、请求和使用信息的行为，并在 7 个未见自然语言协作任务上验证了迁移效果。

## 研究问题与动机
- 多智能体系统的有效协作需要代理分享相关信息、请求缺失信息、将队友贡献整合到决策中，但现有方法难以评估单个消息对团队决策的贡献价值。
- 现有 LLM post-training 主要关注指令遵循和多轮任务完成，不区分个体消息如何贡献于团队决策；仅奖励正确答案或最终结果的方法（如 MA-PoRL、Outcome-RL）无法提供细粒度的协作行为反馈。
- 开放自然语言任务中，相关事实、跨代理的信息分布以及信息对最终决策的贡献难以验证和计算，缺乏可训练的可验证环境。
- 人类集体智慧研究表明团队绩效取决于信息共享质量而非仅个体能力，LLM 代理同样存在隐瞒信息、忽略队友输入、盲目从众等协作失败现象。

## 核心贡献（创新点）
- **可扩展协作训练环境**：构建基于评分表的训练环境，通过信息分配和传递机制保留核心协作行为（分享、请求、使用），同时使信息价值可计算；与现有方法（如 MA-PoRL、MAGRPO）相比，本文不仅依赖最终结果奖励，而是将单条消息的信息贡献显式化为训练信号。
- **可验证的信息奖励机制**：提出基于团队惊讶度（team surprisal）减少量的贝叶斯评估器，精确量化共享评分和请求的信息价值；与纯 outcome-based RL 相比，同一最终结果的不同通信路径可获得不同细粒度反馈。
- **跨任务迁移验证**：在 Qwen3-4B/8B、Gemma-3-4B-it 三个骨干模型上展示 CoSIR 在 7 个未见协作任务（HiddenBench、SILO-BENCH、SplitQA、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax）上的迁移能力，且保持通用能力；与 SPIRAL、MARS 等从合成环境学习推理能力的工作相比，本文直接学习目标为协作行为本身。

## 方法详解
- **评分表环境**：生成 k 个选项、m 个属性的评分表 V ∈ {−1,0,+1}^{k×m}，最佳选项满足 y* = argmax_o Σ_a V_{oa}；每个代理 i 仅获初始简报 B_i（部分单元格），通过多轮对话交换信息并作出私有选择与计算。
- **贝叶斯评估器**：对代理 i 的信息记录 K_i = (B_i, R_i)，以均匀先验计算选项 y 为最佳的精确后验概率 P(y|K_i)，使用动态规划避免枚举所有可能表格；团队惊讶度 Φ(K) = −Σ_i log_2 P̄(y*|K_i)，其中 P̄ 下界为 10^{−4}。
- **共享信息奖励**：消息 M 的信息价值 I_share(M) = ΔΦ(M) = Φ(K_before) − Φ(K_after)，按 Shapley 分配或边际剔除法将值分配到各评分语句；仅当语句由发言者简报或已接收报告支持且与真值一致时才获正向奖励，伪造语句额外惩罚 −0.04，每条语句有 −0.01 沟通成本。
- **请求信息奖励**：假设请求被真实回答，计算潜在信息价值 I_request(q,c) = [ΔΦ(x_{q,c})]_+，乘以持有者下次发言概率 p_next(q) 和折扣因子 δ_c 并 capped 至 2 bits；请求已在待处理或为无效动作时获 −0.10 惩罚。
- **私有答案奖励**：r_i^{answer} = P̂(ŷ_i|K_i) + (1/k)Σ_o 1[ŝ_{io} = s^K_{io}]，衡量代理是否使用可用信息正确选择选项并计算各选项总分；未解析的回答得 0。
- **自博弈训练**：使用共享策略进行自博弈，每回合采样 16 条候选消息 + 最多 2 条演示样本，基于 token 级信用分配（statement spans 直接分配语句奖励，action spans 分配中心化 advantage）更新策略；采用 PPO 变体（ratio clipping [0.8, 1.28]、负 advantage 的 dual clipping bound 3、gradient norm 1.0）；使用参考模型惩罚 L_KL 和通用提示模仿损失 L_anchor（系数各 0.1）以保持基础能力。

## 实验与结果
- **数据集**：训练使用生成的评分表环境（5 种信息配置家族 A–E）；测试使用 7 个未见协作任务：HiddenBench、SILO-BENCH、SplitQA（新增）、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax；通用能力测试：GSM8K、MMLU、IFEval。
- **基线**：Base（初始检查点）、ToM（推理时 Theory-of-Mind）、TalkHier（推理时层次对话框架）、Outcome-RL（与 CoSIR 相同训练但仅奖励最终团队结果）。
- **主要结果**：
  - Qwen3-8B：CoSIR SUM = 384.44，较 Base +31.85，较第二强 TalkHier 高 19.79 分；在 7 个任务中全部优于 Base，HiddenBench 75.00±2.31 最高。
  - Qwen3-4B：CoSIR SUM = 357.87，较 Base +24.70，较 Outcome-RL 高 22.95 分。
  - Gemma-3-4B-it：CoSIR SUM = 235.94，较 Base +27.92，较 Outcome-RL 高 24.86 分。
  - SplitQA：CoSIR 分别提升 Qwen3-8B +2.50、Qwen3-4B +0.36、Gemma-3-4B-it +8.21 分，验证跨选项选择到自由文本生成的迁移。
- **结论**：CoSIR 在所有模型上取得最高总分且各任务均优于 Base；推理时方法在个别任务上更强但 SUM 较低；Outcome-RL 显著低于 CoSIR，证明细粒度信息奖励优于纯结果奖励。
- **行为分析**：训练后代理分享私有信息的比例上升（如 Qwen3-8B HiddenBench 从 68.5%→74.3%），更早披露信息（第一轮从 61.1%→69.0%），忽略队友输入失败率从 26.3% 降至 22.8%，协调失败率从 67.8% 降至 62.0%；通用能力变化在 ±0.8 个百分点内。

## 相关工作脉络
- **LLM Multi-Agent Collaboration（推理时方法）**：ToM、TalkHier 等通过推理时额外调用增强交互，CoSIR 直接训练模型本身而非修改推理流程，两者可在相同代理数和发言限制下对比。
- **Post-Training for LLM Collaboration**：MA-PoRL 奖励正确答案和讨论质量，MAGRPO 将协作形式化为合作多智能体 RL；CoSIR 的核心差异在于将单条消息/请求的信息贡献显式化为可计算奖励信号。
- **Transfer from Synthetic Environments**：SPIRAL、MARS 从零和/合作博弈中学习推理能力并迁移；CoSIR 沿袭该范式但直接学习目标为协作行为（分享、请求、使用信息）而非一般推理或游戏策略。
- **Communication in Multi-Agent Systems**：Cemri et al. (2025) 提出 MAST 失败分类法用于分析 LLM 多代理系统；本文使用该方法量化训练前后协作失败率的变化。

## 局限性与未来方向
- 训练依赖玩具环境中的已知事实和可计算评估器，扩展到开放端到端任务需要评估信息价值的替代方案。
- 训练未能消除所有通信失败类型（如重复步骤 85.6%、信息保留 18.7% 未改善），表明当前奖励设计对某些失败模式效果有限。
- 仅评估了 3 个模型骨干和 7 个协作任务，推断时方法在部分模型-任务组合上仍更强。
- 未来方向包括将信息价值评估扩展到状态不可完全观测的开放任务、探索更高效的信息交换机制、以及结合其他失败类型的缓解策略。

## 研究启发与可借鉴点
- **贝叶斯评估器 + 团队惊讶度**：将信息价值量化为后验概率变化，提供了一种在确定性环境中可计算的细粒度奖励设计范式，可迁移到其他需要评估消息贡献的多智能体训练场景。
- **Token 级信用分配**：对消息中不同成分（评分语句、请求、私有答案）分别分配奖励并映射到相应 token span，比轨迹级奖励更精确，可借鉴用于复杂对话任务的奖励设计。
- **保留通用能力的双正则化**：同时使用参考模型 KL 惩罚和通用提示模仿损失，在强化学习训练协作行为时防止灾难性遗忘，策略值得在其他领域后训练中复用。
- **请求的潜在价值预评估**：不等待实际回复即根据假设真实回答计算请求价值，为主动信息获取提供了可训练的奖励信号，可扩展到更多查询式交互任务。

## 关键术语表
- **Team Surprisal（团队惊讶度）**：各代理对正确选项概率的负对数之和，用于度量团队对正确决策的不确定程度；信息价值由其减少量定义。
- **Bayesian Evaluator（贝叶斯评估器）**：基于代理初始简报和已接收报告，使用均匀先验和动态规划精确计算各选项为最佳的概率的后验评估器。
- **CoSIR（Collaboration via Surprisal-based Information Rewards）**：论文提出的后训练方法，通过团队惊讶度奖励训练代理的协作行为。
- **Self-play（自博弈）**：同一模型扮演所有代理角色进行多轮交互并收集训练数据的方法，避免异构策略对齐问题。
- **Outcome-RL**：与 CoSIR 相同训练框架但仅奖励最终团队结果的对比方法，用于验证细粒度信息奖励的收益。
- **MAST（Multi-Agent System Failure Taxonomy）**：Cemri et al. (2025) 提出的多智能体系统失败分类体系，用于分析训练前后协作失败模式的变化。
- **Shapley Allocation（沙普利分配）**：将消息整体信息价值按各语句边际贡献的平均值分配到单个语句的博弈论方法。
- **SplitQA**：论文新增的迁移评测任务，要求代理从不同文档段落中组合事实生成自由文本答案，测试跨选项选择到开放生成的迁移。

## 可复现要素
- **数据集**：训练使用作者生成的评分表环境（5 种家族配置）；测试使用 HiddenBench、SILO-BENCH、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax 等公开数据集及作者新增的 SplitQA（70 题，基于 HotpotQA）；论文未明确声明代码/数据仓库链接，但 arXiv 论文通常附带补充材料。
- **代码/权重**：论文声明使用 Qwen3-4B、Qwen3-8B、Gemma-3-4B-it 三个开源骨干；训练过程在 GitHub/arxiv 中可能提供（具体 URL 需查阅论文正文），补充材料包含任务生成、解析器、奖励计算等细节。
- **关键超参**：每轮 60 场游戏、16 条候选消息、温度 1.2、最大生成长度 1024 tokens、ratio clipping [0.8, 1.28]、dual clipping 3、gradient norm 1.0、reference KL 系数 0.1、anchor loss 系数 0.1、3 轮训练、学习率 1e−5/1e−5/2e−5；硬件为 2×H100 + 4×Gaudi 2，每轮约 5 小时，总计约 15 小时。
