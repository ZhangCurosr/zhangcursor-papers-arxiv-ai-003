---
title: "SPEEDRUNBENCH-CHALLENGING-LLM-AGENTS-WITH-VIDEO-GAME-SPEEDRU"
source: https://arxiv.org/pdf/2610.08076v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:24:03"
field: "LLM Agent游戏评测与策略优化"
keywords: ["LLM Agent", "Video Game Speedrunning", "Benchmark", "Strategy Formation", "Long-horizon Reasoning", "Saturation-resistant Evaluation"]
innovations: ["提出抗饱和的速通基准SPEEDRUNBENCH，分离构造与优化能力", "揭示预训练污染对游戏评测的影响，引入开源对应物控制", "设计ONLINE/OFFLINE-SCRATCH/OFFLINE-SEED三种设置协同评估智能体多维能力"]
benchmarks: ["SPEEDRUNBENCH", "SuperTux 1-1", "Super Mario Land 1-1", "Pokémon Blue/Red", "Tuxemon", "Mario Kart 64", "SuperTuxKart", "Astray", "Civilization I"]
---

# 论文速读：SPEEDRUNBENCH-CHALLENGING-LLM-AGENTS-WITH-VIDEO-GAME-SPEEDRUNNING

## 一句话总结
本文提出SPEEDRUNBENCH基准，通过视频游戏速通任务评估前沿LLM智能体的策略形成能力；在9款游戏中测试了10+个模型，结果显示智能体在简单平台游戏中接近人类世界纪录，但在长程RPG/策略游戏中仍有显著差距，且速通任务具有抗饱和性，可作为持续推动模型能力边界的评测工具。

## 研究问题与动机
1. **前沿LLM智能体能否超越人类已解决的问题**：现有基准多聚焦"完成任务"，但人类专家方案已饱和，需寻找能持续区分模型能力的评测维度。
2. **速通任务为何适合评估策略形成**：速通要求智能体反复优化策略、反思表现、利用积累知识、在长程动作中进行推理以击败自己和他人的记录，而非仅完成游戏。
3. **现有游戏基准的局限性**：如LMGameBench、VideoGameBench等侧重"完成游戏"，无法提供Evergreen评测视角；有限基准总会饱和，而速通的优化目标（更快完成）永不终止。
4. **长程推理与反事实思维的挑战**：速通需要推理超越"可行解"的"最优解"，涉及替代路线的设想和游戏机制的深层理解，这比单纯通关更难。

## 核心贡献（创新点）
1. **提出SPEEDRUNBENCH基准**：覆盖9款游戏（平台跳跃、RPG、赛车、策略、益智等6种类型），包含ONLINE、OFFLINE-SCRATCH、OFFLINE-SEED三种评测设置，首次系统化评估LLM智能体的速通能力。
2. **设计抗饱和的评测任务**：速通以世界纪录为锚点，每次新记录都是上界，永远存在更快的完成方式，解决了传统基准饱和问题，可为持续推动模型能力提供Evergreen评测视角。
3. **揭示"构造路线"与"优化路线"的能力差异**：OFFLINE-SCRATCH（无种子）与OFFLINE-SEED（有种子）的设置分离显示，智能体在构造有效轨迹方面表现更强，但优化已有慢速轨迹时提升有限（如Pokémon种子仅改进<1%）。
4. **发现预训练污染对评测的影响**：通过对比热门游戏《Pokémon》与开源对应物《Tuxemon》，证明 popular title 的预训练暴露会虚高模型表现，两者排名相关系数为负（-0.18），说明流行游戏成绩不可预测陌生游戏表现。

## 方法详解
- **游戏环境建模**：将游戏定义为确定性环境，状态空间S，有限动作空间A（控制器输入集）；从固定初始状态s₀开始，每个动作推进一帧，完整动作序列{aᵢ}∈A决定一次run。
- **ONLINE设置**：智能体观察当前游戏帧s，预测下一动作a∈A（或短序列）；测试多模态实时感知能力，仅限部分视觉模型（Opus 5、GPT-5.6-sol、Grok 4.6）能完成RPG游戏。
- **OFFLINE-SCRATCH设置**：智能体从零开始构造完整的帧索引动作轨迹，无参考轨迹；侧重探索与路线发现能力。
- **OFFLINE-SEED设置**：智能体基于给定种子轨迹（人类RTA、BFS脚本、ONLINE运行等）进行优化；侧重改进能力。
- **自研循环（Auto-research Loop）**：模仿人类反复试玩、定位时间损失、修改一点、再次运行的过程；每轮提交编辑后的轨迹，接收 verdict：是否达目标（o∈{true,false}）、达目标时的帧数f、或未达目标时的最大进度。
- **评分器预算扩展实验**：允许智能体每轮评估最多200条候选轨迹（而非仅提交1条），DeepSeek-V4-Pro在SuperTux 1-1达到1,847帧（世界纪录的107%），GLM 5.2在Super Mario Land 1-1达到2,314帧（世界纪录的103%）。

## 实验与结果
- **数据集/游戏**：9款游戏（Super Mario Brothers、Super Mario Land、SuperTux、Pokémon Blue/Red、Tuxemon、Mario Kart 64、SuperTuxKart、Astray、Civilization I），跨NES、Game Boy、N64、MS-DOS/SNES、浏览器、C++/Browser等平台。
- **评估基线**：人类实时攻击（RTA）世界纪录（SlNNED 2025、EiP25 2026、DylCat 2025等）；种子轨迹来自GPT-5.6-sol/Opus 5 ONLINE运行、BFS脚本、人类轨迹。
- **模型**：10+个前沿模型（Claude Opus 5、GPT-5.6-sol、Grok 4.6、Gemini-3.7-Flash、Kimi-K3、GLM 5.3、GLM 5.2、DeepSeek-V4-Pro-0813、DeepSeek-V4-Flash-0731、Inkling-Small），全部使用Opencode harness，推理 effort 设为最大。
- **主要结果**：
  - 无任何智能体在实用预算内达到人类世界纪录，最佳平均结果慢于纪录2倍以上。
  - SuperTux 1-1（开源平台跳跃）表现最佳，五款模型OFFLINE-SCRATCH达到1,918–2,432帧（世界纪录约1,600帧，即118%-152%）。
  - Pokémon Blue OFFLINE-SEED从Opus 5种子（9,498帧）改进<1%（8,412–9,428帧），而人类纪录约40,435帧，最佳模型仍慢4倍。
  - Super Mario Land 1-1 OFFLINE-SCRATCH：Opus 5为2,835帧，GPT-5.6-sol为3,271帧，Grok 4.6为3,417帧，GLM 5.3为4,113帧；OFFLINE-SEED从6,017帧人类种子改进至5,145–5,896帧。
  - 扩展评分器预算后，DeepSeek-V4-Pro在SuperTux达到1,847帧（世界纪录+7%），GLM 5.2在Super Mario Land达到2,314帧（世界纪录+3%）。
- **关键结论**：智能体在短平快平台游戏中接近世界纪录，但在长程RPG/策略游戏中仍显著落后；给定成功轨迹作为起点会限制智能体优化能力（"priming"效应）。

## 相关工作脉络
1. **LMGameBench (Hu et al., 2026)**：提供感知与记忆脚手架、标准化prompt、筛查预训练数据污染，侧重完成游戏；SPEEDRUNBENCH进一步要求"完成速度"区分，提供 finer-grained discrimination。
2. **VideoGameBench (Zhang et al., 2025)**：评估VLM能否完成流行游戏；未涉及速通优化与反饱和评测。
3. **Gemini 2.5 Pro Pokémon尝试 (Comanici et al., 2025)**：406.5小时完成，属完成任务；SPEEDRUNBENCH量化其与世界纪录的帧数差距。
4. **MineDojo/Voyager (Fan et al., 2022; Wang et al., 2024)**：关注Minecraft中获取物品与学习技能的数量；SPEEDRUNBENCH关注完成速度的优化。
5. **PokeAgent Challenge (Karten et al., 2026a, 2026b)**：NeurIPS 2025竞赛，速通为其中一条赛道，但仅针对单一游戏（Pokémon）与ONLINE设置；SPEEDRUNBENCH扩展至多类型游戏与三种设置。
6. **GlitchBench (Taesiri et al., 2024)**：评估VLM识别渲染屏幕错误的能力；不涉及长程策略形成与速通优化。
7. **经典RL游戏基准（Atari、Go、StarCraft II）**：从像素控制到超人类棋盘游戏；SPEEDRUNBENCH将焦点从"获胜"转向"以最短时间获胜"的策略形成。

## 局限性与未来方向
- **预算限制**：实验固定每模型200轮交互、无in-turn scorer或最多200次评估，实际世界纪录远未触及；更长的优化预算可能进一步提升。
- **仅聚焦特定游戏类型**：当前9款游戏以2D平台跳跃、回合制RPG、赛车、策略为主，尚未覆盖开放世界、多人对战、物理模拟等类型。
- **预训练污染控制不完全**：虽引入开源对应物（SuperTux、Tuxemon）以测量污染，但部分商业游戏（如Mario系列）可能已在预训练中出现，导致成绩虚高。
- **缺少人类专家对比的细粒度分析**：仅对比RTA世界纪录，未与Tool-Assisted Speedrun（TAS）或人类速通社区的迭代优化过程对比。
- **未来方向**：扩展至更多游戏类型；研究多智能体协作速通；探索结合强化学习或世界模型的混合架构；开发可自动验证最优性的 lower bound 计算工具。

## 研究启发与可借鉴点
1. **反饱和评测设计**：以"持续优化"而非"完成任务"为目标，可推广至代码生成（追求更快执行）、数学证明（追求更简洁推导）、机器人控制（追求更低能耗）等需迭代的任务。
2. **种子敏感性与去priming策略**：OFFLINE-SEED的"priming"效应提示，在评测优化能力时应小心控制起始点；可借鉴"随机种子消融"或"无种子对照"设计。
3. **开源对应物控制污染**：为每款热门游戏配备开源同类型替代（如SuperTux vs. Mario、Tuxemon vs. Pokémon），可量化预训练暴露的影响，值得在其他 benchmark 中推广。
4. **评分器预算可扩展性**：允许智能体在单轮内评估多条候选轨迹（如200次）显著提升了性能，提示"内部验证循环"是突破优化瓶颈的关键，可借鉴至Agent自改进研究。
5. **多设置协同评估**：ONLINE（实时感知）+ OFFLINE-SCRATCH（探索）+ OFFLINE-SEED（优化）三种设置分离，揭示了智能体的不同能力维度，为综合性评测提供了框架范式。

## 关键术语表
- **SPEEDRUNBENCH**：本文提出的基准，评估LLM智能体在视频游戏速通中的表现，覆盖9款游戏与三种设置。
- **RTA (Real-Time Attack)**：人类玩家在真实时间中完成游戏的认证记录，作为世界纪录基准。
- **TAS (Tool-Assisted Speedrun)**：借助工具逐帧编辑的最优轨迹，常作为近优种子用于评测。
- **OFFLINE-SCRATCH**：无参考轨迹的速通设置，智能体需从头构造有效动作序列。
- **OFFLINE-SEED**：给定种子轨迹的速通设置，智能体在此基础上迭代优化。
- **Auto-research Loop**：模仿人类反复试玩、定位损失、单点修改、再次运行的自改进循环。
- **抗饱和 (Saturation-resistant)**：指评测任务的目标永不终止（总有更快的完成方式），避免传统基准的饱和问题。
- **Pretraining Contamination**：模型在预训练阶段接触过游戏内容，导致在评测中表现虚高。

## 可复现要素
- **数据集/游戏**：9款游戏，其中SuperTux、Tuxemon、SuperTuxKart、Astray为开源游戏；匿名化代码公开于 https://anonymous.4open.science/r/tuxemon-benchmark-anon-A2A4（GPL-3.0代码，CC-BY-SA资产）。
- **代码/权重**：模型权重为各厂商提供（Claude Opus 5、GPT-5.6-sol、Grok 4.6、Gemini-3.7-Flash、Kimi-K3、GLM 5.3/5.2、DeepSeek-V4-Pro/Flash、Inkling-Small）；评估 harness 使用Opencode（开源）。
- **关键超参**：OFFLINE设置迭代次数200轮；扩展预算实验每轮最多评估200条候选轨迹；推理 effort 设为所有模型的最大可用值。
- **世界纪录来源**：Speedrun.com认证RTA记录（SlNNED 2025、EiP25 2026、DylCat 2025、theodorepringle 2021、abney317 2026）；TAS轨迹来自Bisqwit（2003）等。
- **论文未提及**：具体的GPU硬件配置、单模型训练/微调成本、开源游戏的具体版本哈希。
