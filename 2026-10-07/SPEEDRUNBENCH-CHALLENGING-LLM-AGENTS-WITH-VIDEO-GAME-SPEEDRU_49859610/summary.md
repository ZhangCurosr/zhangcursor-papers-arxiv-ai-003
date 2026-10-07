---
title: "SPEEDRUNBENCH-CHALLENGING-LLM-AGENTS-WITH-VIDEO-GAME-SPEEDRU"
source: https://arxiv.org/pdf/2610.08076v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:51:55"
field: "大语言模型评测与基准构建"
keywords: ["LLM Agent", "Game Benchmark", "Speedrunning", "Strategy Formation", "Long-horizon Reasoning", "Contamination Control"]
innovations: ["提出以速度跑为核心的持续性Agent评测基准SPEEDRUNBENCH，区分完成能力与优化能力", "设计ONLINE/OFFLINE-SEED/OFFLINE-SCRATCH三设置评估框架，分别捕捉实时感知、路线微调与自主发现能力", "引入开源配对游戏控制预训练数据污染，量化记忆与推理的贡献差异"]
benchmarks: ["SPEEDRUNBENCH"]
---

# 论文速读：SPEEDRUNBENCH-CHALLENGING-LLM-AGENTS-WITH-VIDEO-GAME-SPEEDRUN

## 一句话总结
本文提出 SPEEDRUNBENCH，一个以电子游戏速度跑（speedrunning）为核心任务的评测基准，通过三种设置（实时在线感知、离线从无构建路线、离线从种子优化路线）评估前沿 LLM Agent 的策略形成与长程优化能力，揭示当前模型在完成类任务上逼近人类但仍远未达到最优路径的局限。

## 研究问题与动机
1. **完成 ≠ 最优**：现有游戏 Agent 基准（如 VideoGameBench、LMGameBench）仅评估"能否完成游戏"，无法区分完成质量与策略优劣，且面临基准饱和风险。
2. **速度跑的持续挑战性**：速度跑要求在全球已有记录的基础上不断逼近更快时间，不存在静态饱和点，适合长期追踪模型能力演进。
3. **策略形成能力缺失**：当前 Agent 在给定种子轨迹时仅能微调（<12% 提升），难以自主发现绕过设计者预期的捷径、Glitch 或反直觉路由。
4. **预训练数据污染效应**：商业游戏（如 Pokémon）因广泛存在于预训练语料中，模型可能"记忆"而非"推理"，需通过开源配对游戏（Tuxemon）控制该混淆因素。

## 核心贡献（创新点）
1. **提出 SPEEDRUNBENCH 基准**：覆盖 9 款游戏、6 个品类，引入速度跑作为衡量 Agent 策略与优化能力的持久评测范式，区别于仅关注完成状态的现有基准。
2. **定义三设置评估框架**：ONLINE（多模态实时感知）、OFFLINE-SCRATCH（从零发现可行路线）、OFFLINE-SEED（对已知路线迭代优化），分别捕捉不同能力维度。
3. **自改进循环设计**：模拟人类 speedrunner 的"试错-反思-再尝试"流程，Agent 可提交多次候选轨迹并接收评分器反馈（in-turn scorer），实现近-optimal 轨迹的微调。
4. **开源游戏配对机制**：每类主流游戏搭配功能相似但预训练暴露度低的开源版本（如 SuperTux vs Super Mario、Tuxemon vs Pokémon），量化数据污染对性能的影响。

## 方法详解
- **游戏建模**：将游戏视为确定性环境，状态空间 $S$、动作空间 $\mathcal{A}$，从初始状态 $s_0$ 出发，每帧对应一个动作，完整轨迹为帧级动作序列 $\{a_i\}_{i=1}^n$。
- **ONLINE 设置**：Agent 观察当前截图，预测下一动作或短动作序列，需具备实时多模态感知能力；仅少数闭源模型能完成 Pokémon 首徽章。
- **OFFLINE-SEED 设置**：Agent 从人类或脚本生成的参考轨迹出发，在固定 200 轮内提交编辑后的完整轨迹，每次获得二值结果（是否到达终点）与帧数反馈。
- **OFFLINE-SCRATCH 设置**：Agent 无种子轨迹，需自主探索生成首次可行路径，考验路线发现能力；成功率显著低于 SEED 版本。
- **自改进循环**：每轮允许 Agent 提交若干候选轨迹（最多 200 条）并由内建评分器评估，选择最优者提交，近似于 Tool-Assisted Speedrun（TAS）的人工迭代优化。
- **评估指标**：以完成帧数与人类世界纪录（RTA, Real-Time Attack）的比值衡量性能，越低越好；对比基线包括 BFS 生成轨迹、人工记录、在线生成轨迹。

## 实验与结果
- **数据集/游戏**：9 款游戏，涵盖平台（Super Mario Brothers、Super Mario Land、SuperTux）、RPG（Pokémon Blue/Red、Tuxemon）、竞速（Mario Kart 64、SuperTuxKart）、解谜（Astray）、策略（Civilization I）六大品类。
- **评估模型**：10+ 前沿模型，包括 Claude Opus 5、GPT-5.6-sol、Grok 4.6、Gemini-3.7-Flash、Kimi-K3、GLM 5.3、DeepSeek-V4-Pro/Flash、Inkling-Small 等。
- **关键结果**：
  - **ONLINE**：Pokémon 首徽章完成帧数，Opus 5 为 160,021，GPT-5.6-sol 为 283,474，Grok 4.6 为 477,140，人类世界纪录约 40,435，最佳模型慢约 4×。
  - **OFFLINE-SEED**：所有模型在 SuperTux 上仅比种子轨迹快 12% 以内（8,412–9,428 帧），Pokémon 提升不足 1%。
  - **OFFLINE-SCRATCH**：DeepSeek-V4-Pro 在 SuperTux 1-1 达到 1,918 帧（距世界纪录 118%），GLM 5.2 在 Super Mario Land 达到 2,314 帧（距世界纪录 103%）。
  - **加入 In-turn Scorer**：DeepSeek-V4-Pro 在 SuperTux 达到 1,847 帧（接近世界纪录 7%），GLM 5.2 在 Super Mario Land 达到 2,314 帧（3% 内），表明评估预算是关键瓶颈。
- **污染效应**：Pokémon 上所有模型提升 <1%，Tuxemon 上 GLM 5.3 和 Grok 4.6 提升 6.9% 和 5.9%，两款游戏排名相关系数 -0.18，证实预训练数据记忆主导性能。

## 相关工作脉络
1. **Game Playing Benchmarks**：VideoGameBench（Zhang et al., 2025）和 LMGameBench（Hu et al., 2026）仅评估"完成"能力，缺乏对路径效率的细粒度区分。
2. **Speedrunning as Task**：Pokeagent Challenge（Karten et al., 2026）仅聚焦单一 Pokémon 游戏的在线动作预测，未提供离线优化设置。
3. **Reinforcement Learning in Games**：Atari Learning Environment（Bellemare et al., 2013）、MuZero（Schrittwieser et al., 2020）等使用游戏评估 RL 算法，但目标为通用智能而非策略最优化。
4. **Contamination Control**：Tuxemon 作为 Pokémon 的开源等价物，用于分离"熟悉度"与"推理能力"，类似策略见于 LMGameBench 的数据污染筛查。
5. **Long-horizon Reasoning**：Minecraft 类开放世界任务（MineDojo、Voyager）侧重资源收集与技能习得，而非时间优化。

## 局限性与未来方向
1. **现实可行性限制**：部分游戏（如 Civilization I）因状态空间过大，即使加入 scorer 也无法改进，说明当前 Agent 难以处理纯策略类游戏。
2. **预算敏感性**：性能高度依赖评估预算（如 200 轮 vs 无限轮），缺乏对计算成本与实际收益权衡的分析。
3. **视觉依赖差异**：ONLINE 设置要求多模态感知，而 OFFLINE 设置仅需文本/轨迹输入，导致开源小模型在离线设置中更具竞争力，评估维度不一致。
4. **未来方向**：可扩展至更多游戏类型（如即时战略、生存建造）、引入多 Agent 协作/竞争场景、探索神经符号混合优化方法突破局部最优。

## 研究启发与可借鉴点
1. **自改进循环架构**：Agent 通过"候选生成-评估-选择"迭代优化轨迹的设计，可迁移至代码生成、规划、调度等序列优化任务。
2. **污染控制实验设计**：通过开源配对游戏隔离预训练记忆效应的方法，适用于任何依赖公开数据的评测基准构建。
3. **多设置协同评估**：ONLINE/SEED/SCRATCH 三设置分别捕捉感知、微调、发现能力，为全面评估 Agent 分层能力提供范式。
4. **持续进化基准理念**：速度跑作为"永不饱和"的评测目标，为长期追踪模型能力提供了对抗过拟合的基准设计思路。

## 关键术语表
**Speedrunning**：玩家以最短时间完成游戏并挑战世界纪录的社区实践，强调对游戏机制的深度挖掘与反常规路由。
**OFFLINE-SEED / OFFLINE-SCRATCH**：两种离线优化设置，前者从已知轨迹出发微调，后者需从零构建首次可行路径。
**In-turn Scorer**：Agent 每轮可评估多条候选轨迹的内建评分器，显著提升自我优化能力。
**RTA (Real-Time Attack)**：人类玩家真实时间攻击的世界纪录，作为 benchmark 的性能锚点。
**Tool-Assisted Speedrun (TAS)**：借助人工逐帧编辑实现的近乎最优轨迹，用于生成高质量种子。
**Contamination**：训练数据中预存游戏内容导致的"记忆式"性能膨胀，需通过开源游戏对照控制。
**Long-horizon Coherence**：Agent 在数百次迭代中保持轨迹整体有效性的能力，单次错误即导致整条轨迹失效。

## 可复现要素
- **数据集**：9 款游戏，其中 SuperTux、Tuxemon、SuperTuxKart 为开源游戏；部分开源游戏代码已通过匿名链接发布（https://anonymous.4open.science/r/tuxemon-benchmark-anon-A2A4）。
- **代码/权重**：基准测试代码未明确公开，仅开源 Tuxemon 修改版本；模型权重为各商业/开源模型官方权重。
- **关键超参**：迭代轮数 200（部分实验 50–220），每轮候选轨迹数 1–200，推理开销设为各模型最大可用值；使用 Opencode 框架统一实验环境。
