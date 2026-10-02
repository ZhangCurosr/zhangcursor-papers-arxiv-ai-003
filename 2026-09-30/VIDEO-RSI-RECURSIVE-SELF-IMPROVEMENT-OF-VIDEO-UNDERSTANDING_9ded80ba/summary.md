---
title: "VIDEO-RSI-RECURSIVE-SELF-IMPROVEMENT-OF-VIDEO-UNDERSTANDING"
source: https://arxiv.org/pdf/2609.37950v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:26:53"
field: "视频理解与智能体系统"
keywords: ["video understanding", "agent harness self-improvement", "frozen model evolution", "active investigation", "accuracy-cost trade-off"]
innovations: ["提出 Video-RSI 框架，以同一冻结 LLM 同时担任 Solver 与 Editor 实现 harness 自改进", "引入主动视频调查与成本感知选择门控的双目标演化循环", "在多基准验证精度-效率权衡，并在 MLVU 较基线提升 4.8pp 同时降帧 27.2%"]
benchmarks: ["MLVU Test", "LongVideoBench", "Video-MME Long", "EgoSchema"]
---

# 论文速读：VIDEO-RSI: RECURSIVE SELF-IMPROVEMENT OF VIDEO UNDERSTANDING AGENTS VIA HARNESS EVOLUTION

## 一句话总结
Video-RSI 提出一个框架，让同一个冻结的LLM同时作为**解题者**与**编辑者**，通过主动重返原始视频调查失败原因，将诊断转化为可复用的代码修订，并在“准确率-视觉成本”双目标约束下筛选并累积改进，从而实现无需更新模型权重的视频理解智能体自我进化。

## 研究问题与动机
- **执行轨迹信息有限**：现有可执行 harness 的运行轨迹仅记录当前策略已获取的证据，无法区分失败是源于“遗漏关键事件”“感知错误”还是“未有效利用已有证据”，导致针对性改进困难。
- **改进缺乏证据支撑**：基于轨迹自动修复制定的代码改动缺乏对原始视频的主动验证，可能在错误诊断上累积无效甚至有害的修改。
- **精度与效率难以兼顾**：单纯追求准确率提升可能引发视觉观察成本（处理帧数）显著上升；而仅优化效率又可能牺牲答案质量。需要一种能同时权衡两者的改进与筛选机制。
- **缺乏可复用、跨问题的程序级进化**：现有工作多聚焦单次问答内的自适应检索，或依赖模型微调；缺少让智能体在离线阶段基于证据系统化地迭代、固化并复用可执行 harness 的闭环。

## 核心贡献（创新点）
1. **提出 Video-RSI 框架，实现 Frozen-Model 下的 Harness 自改进**：同一冻结 LLM 兼任 Solver 与 Editor，离线主动调查训练视频并修订可执行 harness，不改变底层模型权重。*与既有方法（如基于微调或仅调 prompt 的优化）的本质区别在于，改进对象是控制证据获取与处理的程序层，且以自身 frozen 能力完成闭环。*
2. **设计 Active Video Investigation 与 Cost-Aware Harness Evolution 的改进循环**：在提出修订前重返视频获取新证据以支撑失败归因；用准确率与平均视觉帧数双指标对候选 harness 进行带边界的筛选。*与单纯基于轨迹的自动代码生成（如 Meta-Harness/VideoHarness-RSI）的区别在于显式引入“主动调查-假设检验”的证据链。*
3. **在四个视频理解基准上验证效果**：演化后的 harness 在准确率上普遍优于对比基线，并在多数任务上实现更低的帧处理量，展现出更好的精度-效率权衡。*与同类 harness 自改进工作相比，不仅看准确率，还同时报告并优化视觉观测开销。*

## 方法详解
- **问题定义与 Evolvable Harness**：Harness $H$ 是一个可执行程序，给定视频 $V$ 与问题 $q$，由冻结的服务集 $\mathcal{M}$ 执行，输出答案 $\hat{y}$、执行轨迹 $\tau$ 与资源记录 $c$。可修订部分包括提示构建、工具实现、观测处理、记忆管理与执行控制。使用训练集 $\mathcal{D}_{\text{tr}}$，并有一个视频不相交、结果保密的候选筛选集 $\mathcal{D}_{\text{g}}$。
- **Active Video Investigation**：Editor 从失败案例中选择代表性样例，提出诊断查询 $u_k$（如变更时间区间、采样密度、模态等），通过 `Investigate` 函数获得新证据 $o_k$（含视频片段、字幕、OCR 等，且感知请求不含金标答案），不断更新上下文 $\mathcal{T}_k$。调查终止于形成有证据支持的修订假设或无更多有效查询。
- **Cost-Aware Harness Evolution**：将诊断映射为代码修订 $\widetilde{H}_t$，并在 $\mathcal{D}_{\text{g}}$ 上独立评测。定义视觉成本 $C$ 为完整问答中成功视觉模型调用处理帧数的均值。选择门控条件为：
  - **精度提升且成本有界增长**：$\Delta A_t > 0$ 且 $\widetilde{C}_t \le (1+\alpha) C_t$；
  - **成本降低且精度损失有界**：$-\epsilon \le \Delta A_t \le 0$ 且 $\widetilde{C}_t \le (1-\beta) C_t$（且 $C_t>0$）。
  满足任一即接受候选。最终在预留测试集上评估。

## 实验与结果
- **数据集与划分**：四个基准的长视频 splits——MLVU Test、LongVideoBench（Long val）、Video-MME Long、EgoSchema。演化使用 LVBench 的 216 题调查/修订 + 72 题视频不相交开发集筛选。
- **模型与协议**：Solver/Editor 均使用 DeepSeek-V4-Pro，视觉观测用 Qwen3.6-Plus，全部冻结。初始 harness $H_{S0}$ 支持全局/局部查看、字幕、OCR 与记忆。运行 20 次修订尝试，$\alpha=0.1, \beta=0.2, \epsilon=1/|\mathcal{D}_{\text{g}}|$。
- **基线**：MLLMs、Video Agentic Models（VideoAgent、VideoTree、MR.Video、DVD、LVAgent、VCA、FrameThinker、EVA-GRPO、VideoSeek）及 Harness 自改进方法 VideoHarness-RSI。其中 VideoSeek 与 VideoHarness-RSI 用相同模型复现。
- **主要结果（Table 1）**：
  - 在所有四基准上均超越复现基线：
    - **MLVU**：Video-RSI **72.9% / 61.1帧** vs VideoSeek 68.1%/83.9帧 vs VideoHarness-RSI 64.1%/83.2帧；相对 VideoSeek 提升 **+4.8pp**，帧数减少 **27.2%**。
    - **LongVideoBench**：71.1% / 41.1帧，优于 VideoSeek 70.9%/71.9帧与 VideoHarness-RSI 68.4%/92.1帧。
    - **Video-MME**：80.0% / 21.6帧，优于 VideoSeek 78.7%/17.8帧与 VideoHarness-RSI 69.7%/92.8帧。
    - **EgoSchema**：78.2% / 42.4帧，优于 VideoSeek 73.2%/72.3帧与 VideoHarness-RSI 78.0%/66.8帧。
  - 总体最强结果：**Video-MME 80.0% 准确率，EgoSchema 78.2% 准确率**；在精度与帧数上呈现更具竞争力的权衡。
- **消融（Table 2）**：
  - **主动调查 vs 仅轨迹**：在全部四基准上精度更高；MLVU 提升 **+8.2pp** 且帧数相近；调查-enabled 版本 20 次中保留 **5** 次修订，轨迹-only 仅 3 次。
  - **成本感知选择 vs 仅准确率选择**：Accuracy-only gate 虽提升精度但在三个基准上增加帧数；Video-RSI 在更高精度的同时显著降帧，如 Video-MME 帧数约为前者一半。
- **演化动态（Fig.3）**：早期修订主要带来帧数下降，后续修订稳步提升精度；部分步骤在开发集上的提升不一定在每个测试集同步体现，需多轮累积。

## 相关工作脉络
1. **Video Agentic Models（VideoSeek、VideoAgent、VideoTree、DV、MR.Video、LVAgent 等）**：通过迭代检索、层次表示或工具调度在单次问答内自适应观测。Video-RSI 与其区别在于：**不在推理时临时调策略，而是离线迭代并固化整个可执行程序**。
2. **Automated Agent Design / Harness Optimization（ADAS、AFlow、Meta-Harness、Self-Harness）**：以代码为优化对象或在固定模型周围搜索工作流。Video-RSI 与之同类，但**额外引入主动重返视频的证据获取环节**，使 revision 基于更充分的事实检验而非仅凭历史轨迹。
3. **Video Harness-RSI（Xu & Chen, 2026）**：同为程序级、冻结模型的视频 harness 自改进。Video-RSI 的定位差异在于：**聚焦 Editor 如何获得修订证据**，并提出兼顾准确率与视觉成本的边界选择规则。
4. **Self-Evolving Video Agents（EvoGround、Video-Zero、EvoVid）**：通过生成/优化训练信号驱动模型自身演化。Video-RSI 与之相对，**保持模型权重冻结，将改进累积到可执行代码而非训练数据/权重**。
5. **FrameThinker、EVA**：通过 SFT/RL 学习观测策略（帧选择、时序窗口、分辨率）。Video-RSI 不使用训练或微调，而是**用同一 frozen LLM 直接编辑并选择程序**。

## 局限性与未来方向
- 依赖**高质量、可自由检索的原始训练视频**进行主动调查；若视频访问受限或涉及隐私，调查环节的适用性将下降。
- 以**视觉处理帧数**作为主要成本指标，未完全涵盖 Token 消耗、多工具调用开销、时延等综合成本。
- 演化在**离线阶段**进行，最终 harness 被冻结后用于新题；对分布外视频或全新任务类型的泛化仍需验证。
- 当前验证围绕**视频问答代理**，未见扩展到多模态任务组合或实时在线场景。
- 调查预算与修订次数（20 次）受经验设定；在不同规模数据集与更复杂策略空间下的可扩展性待进一步考察。

## 研究启发与可借鉴点
- **离线主动调查再修订的范式**：对任何基于可执行 harness/工具的代理系统，均可引入“重返原始交互对象、收集额外证据后再改代码”的两阶段设计，缓解仅凭执行日志做诊断的信息不足问题。
- **多目标边界选择器（accuracy–cost gate）的设计**：以有界增长/下降的双分支选择规则替代单一指标优化，便于在真实部署中兼顾性能与资源消耗；该思想可迁移到工具调用成本、内存占用、时延等多维权衡场景。
- **统一架构（Solver = Editor）**：证明同一 frozen LLM 可同时承担执行与自改进，有助于简化系统设计与资源分配；可探索将其迁移到文本/代码 agent 的 harness 进化中。
- **结构化观测与关系图的重用机制**：演化后 harness 引入 image reference→timestamp 映射、entity/event/relation 图以支持证据复用；这类中间表示可被其他长上下文理解系统借鉴。
- **保留行为识别与最小化变更**：修订时显式保留成功模式、限定适用条件，有助于控制演化噪声；可在其他 agent 自动生成代码时作为约束引入。

## 关键术语表
- **Harness**：控制视频智能体如何获取观测、处理证据、记忆与管理执行流程的可执行程序。
- **Solver / Editor**：在 Video-RSI 中共享同一冻结 LLM 的两个角色；Solver 负责答题与产生轨迹，Editor 负责调查失败并生成修订后的 harness。
- **Active Video Investigation**：Editor 在修订前通过查询原始视频获取新证据（字幕/OCR/片段重播等）以检验不同失败假设的过程。
- **Cost-Aware Harness Evolution**：以答案准确率与平均视觉处理帧数为双目标，在带边界条件下筛选并累积 harness 修订的演化机制。
- **Visual Cost（Frames）**：一次完整问答执行中，所有成功视觉模型调用所处理帧数的均值；用于衡量视觉观测开销。
- **Trajectory-only Revision**：仅利用既有执行轨迹信息进行修订的对照变体，不进行主动视频调查。
- **Accuracy-Only Gate**：忽略视觉成本、仅按准确率高低接受候选修订的对照选择策略。
- **Structured Perception / Relation Graph**：演化后 harness 中出现的结构化观测接口与实体-事件-关系抽取机制，用于提升证据复用能力。

## 可复现要素
- **数据集**：MLVU Test、LongVideoBench Long val、Video-MME Long、EgoSchema；演化使用 LVBench 的 216 题调查集与 72 题开发集（视频不相交）。基准公开，但**用于候选筛选的开发题与结果对评价方保密**。
- **代码**：已开源，地址 https://github.com/bingjunluo/Video-RSI。
- **权重**：模型权重冻结，仅使用 DeepSeek-V4-Pro 与 Qwen3.6-Plus 的服务调用，**未声明公开具体权重文件**。
- **关键超参**：修订尝试上限 $T=20$；成本选择参数 $\alpha=0.1$（允许成本最多增长 10%）、$\beta=0.2$（要求成本至少减少 20%）、$\epsilon=1/|\mathcal{D}_{\text{g}}|$（允许的最多正确题损失为 1 题）。
