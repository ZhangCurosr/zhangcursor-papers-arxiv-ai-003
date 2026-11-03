---
title: "Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable"
source: https://arxiv.org/pdf/2610.17893v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:53:10"
field: "多智能体LLM协作训练"
keywords: ["multi-agent collaboration", "post-training", "information reward", "surprisal", "self-play", "LLM fine-tuning", "distributed information"]
innovations: ["基于团队surprisal减少的可验证信息奖励机制", "评分表环境中细粒度的分享/请求/使用信息奖励设计", "自我对弈训练协作行为并验证跨任务泛化能力"]
benchmarks: ["HiddenBench", "SILO-BENCH", "SplitQA", "MathChat", "BenchForm", "Einstein Puzzles", "Collaboration Tax"]
---

# 论文速读：Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable

## 一句话总结
论文提出 CoSIR（Collaboration via Surprisal-based Information Rewards）后训练方法，利用可验证的评分表环境，通过贝叶斯评估器计算信息共享/请求价值（团队 surprisal 的减少），训练 LLM 学会分享、请求和整合队友信息，并在 7 个未见自然语言协作任务上实现有效泛化，同时保留通用能力。

## 研究问题与动机
1. **最终结果反馈不足**：多智能体协作中，任务最终结果难以提供对每条消息价值的细粒度反馈，导致agent可能隐瞒有用发现、未经澄清就推进或忽略队友输入。
2. **现有训练方法缺乏消息级评估**：MA-PoRL 等方法奖励正确答案和讨论修正，但未显式区分每条消息对团队决策的贡献价值。
3. **开放域任务难以验证信息价值**：在自然语言任务中，相关事实、跨 agent 的分布及其对决策的贡献不易量化评估。
4. **需要可计算的信息价值度量**：在简化但保留信息依赖的受控环境中，使分享、请求和整合信息的行为可被精确计算和奖励。

## 核心贡献（创新点）
1. **可拓展的协作训练环境**：构建评分表任务环境，保留"分享证据、请求缺失信息、整合队友输入"的核心协作行为，同时使信息价值可计算。
   — 与已有工作相比，不仅训练推理/游戏策略（如 SPIRAL、MARS），而是专门针对协作行为本身进行训练。
2. **可验证的信息奖励机制**：提出基于团队 surprisal 的信息共享/请求奖励，结合私人答案（选项选择和总量计算）的监督奖励，形成细粒度训练信号。
   — 与 Outcome-RL 仅奖励最终团队结果不同，CoSIR 的奖励直接绑定到每条消息和信息请求。
3. **跨任务泛化验证**：在 Qwen3-8B、Qwen3-4B、Gemma-3-4B-it 三个模型上验证，CoSIR 在所有 7 个未见协作任务上均超越 Base Checkpoint，且 SUM 得分最高。
   — 与 ToM/TalkHier 等推理时方法不同，本文训练模型本身而非仅修改推理流程。
4. **行为分析揭示协作改善**：训练后 agent 更愿意披露私有信息、更少忽略队友输入、更少无澄清就推进，同时通用能力（GSM8K/MMLU/IFEval）基本保持不变。

## 方法详解
1. **任务环境设计**：
   - 生成 $k$ 个选项、$m$ 个属性的评分表 $V \in \{-1, 0, +1\}^{k \times m}$，最优选项满足 $y^* = \arg\max_o \sum_a V_{oa}$。
   - 每个 agent $i$ 获得初始简报 $B_i = \{(o,a), V_{oa}) : (o,a) \in C_i\}$，包含部分单元格评级。
   - Agents 轮流发送自然语言消息（可含多条评级陈述和一条请求），最后各自做出私人选项选择和总量计算。

2. **贝叶斯评估器**：
   - 定义信息记录 $K_i = (B_i, R_i)$，其中 $R_i$ 为收到的评级陈述。
   - 构造部分表 $E_i(K_i)$，优先使用初始简报值，否则用最新收到的陈述。
   - 在 $\Omega = \{v \in \{-1,0,+1\}^{k\times m} : |\arg\max_o \sum_a v_{oa}| = 1\}$ 上定义均匀先验 $p_0(v) = 1/|\Omega|$。
   - 后验概率：$P(y|K_i) = \frac{\sum_{v \in \Omega_y} \prod_{(c,r)\in E_i(K_i)} \mathbf{1}[v_c = r]}{\sum_{v \in \Omega} \prod_{(c,r)\in E_i(K_i)} \mathbf{1}[v_c = r]}$，通过动态规划精确计算（$O(km^2 + k^2m)$）。

3. **团队 Surprisal 与信息价值**：
   - 团队 surprisal：$\Phi(K) = -\sum_{i=1}^n \log_2 \bar{P}(y^*|K_i)$，其中 $\bar{P} = \max\{P, 10^{-4}\}$ 避免 log(0)。
   - 消息信息价值：$\Delta\Phi(x) = \Phi(K_{before}) - \Phi(K_{after})$，衡量接收后团队对正确选项概率的提升（单位：bits）。

4. **奖励设计**：
   - **分享奖励**：$I_{share}(M) = \Delta\Phi(M)$，对支持性陈述（来自初始简报或已接收报告）分配正值，虚假陈述受罚（额外 -0.04），每条陈述有 -0.01 通信成本。使用 Shapley 值风格分配多陈述消息的价值。
   - **请求奖励**：假设请求的内容被 truthful 提供，计算潜在 surprisal 减少：$I_{request}(q,c) = [\Delta\Phi(x_{q,c})]_+$，乘以剩余轮次概率 $p_{next}(q)$ 和折扣因子。
   - **私人答案奖励**：$r_i^{answer} = \bar{P}(\hat{y}_i|K_i) + \frac{1}{k}\sum_o \mathbf{1}[\hat{s}_{io} = s_{io}^K]$，评估选项选择概率和总量计算正确率。

5. **自我对弈训练**：
   - 每个 speaking state 采样 16 个 policy candidate + 最多 2 个 demonstration。
   - 基于 PPO 变体，statement rewards 直接分配给对应 token，action/choice/total 使用 group-relative advantages。
   - 参考模型 KL 惩罚（系数 0.1）和一般提示 imitation loss（系数 0.1）保留通用能力。
   - 训练 3 个 collection round，每轮约 5 小时，使用 AdamW8bit、gradient norm limit 1.0。

## 实验与结果
- **数据集/基准**：7 个未见协作任务：HiddenBench、SILO-BENCH、SplitQA（新构建）、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax；通用能力基准：GSM8K、MMLU、IFEval。
- **模型**：Qwen3-8B、Qwen3-4B、Gemma-3-4B-it（Base vs CoSIR）。
- **基线**：Base、ToM（推理时）、TalkHier（推理时）、Outcome-RL（仅奖励最终结果）。
- **主要结果**：
  - Qwen3-8B：CoSIR SUM = 384.44（+31.85 over Base），超越第二强基线 Outcome-RL 20.72 分。
  - Qwen3-4B：CoSIR SUM = 357.87（+24.70），超越 Outcome-RL 22.95 分。
  - Gemma-3-4B-it：CoSIR SUM = 235.94（+27.92），超越 Outcome-RL 24.86 分。
  - CoSIR 在所有 7 个任务上均优于 Base（三个模型）。
- **行为变化**：
  - HiddenBench 私有事实披露率：Qwen3-8B 从 68.5%→74.3%，回复队友消息从 59.4%→65.8%。
  - MAST 失败分析：协调失败从 67.8%→62.0%，忽略队友输入从 26.3%→22.8%，无澄清就推进从 36.4%→32.0%。
- **通用能力保留**：GSM8K/MMLU/IFEval 变化在 ±0.8% 以内。

## 相关工作脉络
1. **MA-PoRL (Park et al., 2025)**：通过 RL 奖励正确答案和纠正性/说服性讨论；本文与之区别在于奖励直接绑定到信息价值而非仅最终结果。
2. **MAGRPO (Liu et al., 2026b)**：将协作建模为合作式多智能体 RL；本文不依赖 value function，而是用可计算的贝叶斯 surprisal 提供 fine-grained reward。
3. **SPIRAL (Liu et al., 2025) / MARS (Yuan et al., 2025)**：在零和/合作游戏中通过 self-play 学习可迁移推理；本文聚焦协作行为（分享/请求/使用信息）而非通用推理。
4. **TalkHier (Wang et al., 2025a) / ToM (Li et al., 2023)**：推理时通过分层工作流或心智理论增强协作；本文通过后训练改变模型本身能力。
5. **Peng et al. (2025)**：SFT 训练信息分享/寻求 + 环境验证器；本文使用 RL 而非 SFT，且奖励机制更细粒度。

## 局限性与未来方向
1. **评估范围有限**：仅三个模型和七个任务，推理时方法（ToM/TalkHier）和 Outcome-RL 在部分任务组合上仍更强。
2. **未解决所有失败类型**：训练未消除重复步骤（~85%）和信息保留（~18.7%）等问题。
3. **环境依赖性**：奖励基于玩具环境和已知事实的可计算评估器，扩展到开放域任务需解决"全状态不可用时如何评估信息价值"的问题。
4. **未来方向**：将信息价值评估器推广到开放域自然语言任务、探索更复杂的信息分布模式、研究多轮深度协作场景。

## 研究启发与可借鉴点
1. **可验证环境的细粒度奖励设计**：用贝叶斯 surprisal 减少量作为信息共享的价值度量，避免了 reward hacking 同时提供了 dense feedback，可迁移到其他需要评估消息价值的协作场景。
2. **Shapley 值风格的奖励分配**：对包含多条陈述的消息，使用 removal marginal 和 Shapley allocation 分配 credit，解决了多陈述消息的 credit assignment 问题。
3. **自我对弈 + 参考模型惩罚的通用能力保留**：结合 general-prompt imitation loss 和 KL penalty，在训练特定技能时保持基础能力，适合需要专项训练的 post-training 场景。
4. **行为分析 + MAST 失败分类的组合评估**：不仅报告任务分数，还量化披露率、回复率、失败类型变化，为协作训练提供多维度的效果验证。

## 关键术语表
- **Team Surprisal**：团队 surprisal，定义为各 agent 对正确选项后验概率的负对数之和，衡量团队整体"困惑度"。
- **Bayesian Evaluator**：贝叶斯评估器，基于均匀先验和agent可用信息计算每个选项为最优的后验概率。
- **CoSIR**：Collaboration via Surprisal-based Information Rewards，本文提出的后训练方法。
- **Information Value**：信息价值，定义为接收某组评级陈述前后团队 surprisal 的减少量（bits）。
- **Self-play**：自我对弈，同一模型扮演所有 agent 进行训练。
- **MAST**：Multi-Agent System Failure Taxonomy，协作失败分类体系。
- **SplitQA**：新构建的评估任务，要求 agent 结合分属不同文档的事实生成自由形式答案。
- **Outcome-RL**：仅基于最终团队结果给予奖励的基线方法。

## 可复现要素
- **数据集**：训练环境为程序生成的评分表任务；评估基准包括 HiddenBench、SILO-BENCH、SplitQA（新构建，70题来自HotpotQA）、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax、GSM8K、MMLU、IFEval。论文未提及训练数据公开。
- **代码/权重**：论文未明确声明开源；使用 Qwen3-8B、Qwen3-4B、Gemma-3-4B-it 作为 backbone。
- **关键超参**：games per round = 60，policy candidates = 16，temperature = 1.2，max tokens = 1024，epochs per round = 1，samples per step = 64，clip bounds = [0.8, 1.28]，dual clip = 3，gradient norm = 1.0，KL coefficient = 0.1，learning rate = 1e-5 (round 1-2) / 2e-5 (round 3)，Optimizer = AdamW8bit，Hardware = 2× NVIDIA H100 + 4× Intel Gaudi 2。
