---
title: "VIDEO-RSI-RECURSIVE-SELF-IMPROVEMENT-OF-VIDEO-UNDERSTANDING"
source: https://arxiv.org/pdf/2609.37950v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:27:06"
---

# 论文速读：VIDEO-RSI-RECURSIVE-SELF-IMPROVEMENT-OF-VIDEO-UNDERSTANDING

## 一句话总结
Video-RSI 是在底层模型权重完全冻结的前提下，通过同一语言模型兼任 Solver 与 Editor，让视频理解 Agent 主动重返训练视频诊断失败原因，并将诊断结果转化为可复用代码修订的递归自我改进框架；借助准确率-视觉成本双目标选择门控，进化后的 harness 在四个主流基准上同步提升答题准确率并降低推理帧耗。

## 研究问题与动机
- **执行轨迹信息不足**：现有视频 Agent 的执行轨迹仅记录当前 harness 实际获取的证据，无法区分错误答案是源于“漏看/采样缺失”、“感知误判”还是“未有效利用已有证据”，导致基于轨迹的自动修订缺乏可靠归因依据。
- **模型微调成本高昂**：主流视频 Agent 改进路线（如 FrameThinker、EVA、EvoGround 等）依赖 SFT 或强化学习更新模型权重，训练开销大且难以快速迭代部署。
- **已有代码进化方法缺少主动验证**：VideoHarness-RSI 等方法在冻结 VLM 上进化上下文构造程序，但 Editor 仅依赖历史记录，未对原始视频进行主动证据补充，修订质量受限于既有轨迹的盲区。
- **目标导向**：希望将执行经验沉淀为可复用的程序级改进，在固定模型的前提下同时优化答案准确率与视觉观测成本，使改进可跨问题累积并在离线阶段完成。

## 核心贡献（创新点）
- **提出 Video-RSI 递归 harness 进化框架**：同一冻结语言模型同时承担回答问题的 Solver 与提出修订的 Editor，在不更新任何模型权重的情况下实现视频 Agent 行为模式的持续迭代；与 Meta-Harness、Self-Harness 等通用 harness 优化工作不同，本文聚焦视频场景并提供原始视频重返接口。
- **设计主动视频调查（Active Video Investigation）机制**：Editor 可针对未决失败解释生成诊断性查询（调整时间区域、采样密度、模态等），从原始视频中获取额外观测以区分漏看/误判/利用失败，使代码修订建立在超出执行轨迹的证据之上。
- **提出成本感知 harness 选择门控（Cost-Aware Harness Evolution）**：同时衡量候选 harness 的准确率变化与平均处理帧数，允许“精度提升且成本增长受限”或“成本显著下降且精度损失受限”两类改进被保留，避免单一准确率优化导致的推理开销膨胀。
- **在四个基准上验证离线进化-在线冻结范式的可行性**：进化后 harness 在 MLVU、LongVideoBench、Video-MME、EgoSchema 上均取得优于 VideoSeek 与 VideoHarness-RSI 的结果，消融证明主动调查与成本门控各自贡献显著。

## 方法详解
- **问题形式化**：给定视频 $V$、问题 $q$ 与固定模型服务 $\mathcal{M}$，harness $H$ 执行输出答案 $\hat{y}$、轨迹 $\tau$ 与资源记录 $c$：$(\hat{y}, \tau, c) = \mathrm{Run}(H, V, q; \mathcal{M})$。Editor 可联合修订提示构建、工具实现、观测处理、记忆与执行控制等组件。
- **数据划分与迭代结构**：使用训练集 $\mathcal{D}_{\mathrm{tr}}$ 进行调查与修订；保留一个视频不相交且内容私有化的选择集 $\mathcal{D}_{\mathrm{g}}$ 用于候选筛选。每轮迭代执行 $H_t$ → 收集轨迹 → 主动调查 → 提出候选 $\widetilde{H}_t$ → 在 $\mathcal{D}_{\mathrm{g}}$ 上评估 → 决定是否更新为 $H_{t+1}$。
- **主动视频调查流程**：以当前代码、训练轨迹、标签与历史反馈初始化上下文 $\mathcal{I}_0$。Editor 逐步生成查询 $u_k = \mathrm{Query}_{\mathrm{Editor}}(\mathcal{I}_k)$，通过 $\mathrm{Investigate}$ 接口获取新观测 $o_k$（支持视频片段、语音转录、屏幕 OCR 及不透露金标准的答案回放），累积证据 $\mathcal{I}_{k+1} = \mathcal{I}_k \cup \{(u_k, o_k)\}$ 并迭代，直到形成可复用的修订假设或无更多有效查询。
- **诊断引导的修订生成**：Editor 将新证据映射为代码级改动，明确适用条件与需保留的成功行为。修订可涉及工具输出格式、时间关联保留、查询时机等，并在提交前于训练侧进行调试性试运行。
- **成本感知选择门控**：记 incumbent 与候选的准确率为 $A_t, \widetilde{A}_t$，平均帧耗为 $C_t, \widetilde{C}_t$。接受条件为：
  $$g_t = \underbrace{[\Delta A_t > 0] \wedge [\widetilde{C}_t \leq (1+\alpha)C_t]}_{\text{精度提升且成本增长受限}} \vee \underbrace{[-\epsilon \leq \Delta A_t \leq 0] \wedge [\widetilde{C}_t \leq (1-\beta)C_t] \wedge [C_t > 0]}_{\text{成本下降且精度损失受限}}$$
  实验设置 $\alpha=0.1, \beta=0.2, \epsilon=1/|\mathcal{D}_{\mathrm{g}}|$，仅返回聚合准确率与帧数给 Editor，单样本门控结果保持私有。
- **保留修订的结构化特征**：进化过程中最终留存的关键改动包括（1）结构化感知：自由文本响应替换为目标导向的结构化发现，并将图像引用映射回帧时间戳、显式标注不确定性；（2）关系图证据复用：从累积视觉/转录/OCR 中抽取实体、事件与关系，返回证据缺口以指导稀疏观测，且图查询本身不消耗新帧；（3）选择性观察例程：如 Global-64 扩展全局概览、occurrence 工具定义计数单元并合并重叠证据以避免重复采样。

## 实验与结果
- **评测基准**：MLVU Test、LongVideoBench Long val、Video-MME Long、EgoSchema 官方公开子集；进化阶段使用 216 个 LVBench 问题进行调查修订，另取 72 个视频不相交问题作为固定选择集。
- **模型配置**：Solver/Editor 使用 DeepSeek-V4-Pro，视觉观测由 Qwen3.6-Plus 提供；所有模型权重冻结。初始 harness $H_{S0}$ 支持全局/局部检查、转录本、OCR 与观测记忆。共运行 20 轮修订。
- **主要对比基线**：MLLMs（GPT-4o、Gemini 1.5 Pro、Gemini 2.0 Flash 等）、视频 Agentic Models（VideoAgent、VideoTree、DrVideo、VCA、MR.Video、DVD、FrameThinker、LVAgent、EVA-GRPO 等）、harness 自改进方法（复现 VideoSeek 与 VideoHarness-RSI）。
- **核心结果**（准确率 ↑ / 平均帧数 ↓）：
  - **MLVU**：Video-RSI 72.9 / 61.1，优于 VideoSeek 68.1 / 83.9（+4.8pp，帧数降 27.2%）与 VideoHarness-RSI 64.1 / 83.2
  - **LongVideoBench**：71.1 / 41.1，与 VideoSeek 70.9 / 71.9 相当准确率但帧耗大幅下降，显著优于 VideoHarness-RSI 68.4 / 92.1
  - **Video-MME**：80.0 / 21.6，准确率与帧耗双优，相对 VideoSeek 78.7 / 17.8 精度更高；大幅优于 VideoHarness-RSI 69.7 / 92.8
  - **EgoSchema**：78.2 / 42.4，精度与帧耗均优于 VideoSeek 73.2 / 72.3，与 VideoHarness-RSI 78.0 / 66.8 精度持平但帧耗更低
- **消融结论**：
  - 移除主动调查（仅用轨迹）在四个基准上准确率均下降，MLVU 降幅达 8.2pp，且仅保留 3/20 次修订（vs 主动调查保留 5 次）。
  - 移除成本感知选择（仅用准确率门控）虽在全部基准提升准确率，但在三个基准显著增加帧耗（如 MLVU 从 61.1 升至 81.9，Video-MME 从 21.6 升至 43.1），验证双目标筛选对控制推理开销的关键作用。
- **进化动态**：早期修订主要带来帧耗骤降，后续迭代以小幅非单调波动持续提升准确率；MLVU 线上评估显示选择集收益并非每一步都同步转移，最终精度增益主要由末期修订累积实现。

## 相关工作脉络
- **视频 Agentic 工作**（VideoAgent、VideoTree、VCA、MR.Video、DVD、VideoSeek 等）：聚焦设计检索策略、分层表示与工具链协调；本文与其差异在于不扩展观察策略本身，而是让固定模型自动修订驱动这些策略的可执行代码。
- **自动化 Agent 设计与 Harness 优化**（ADAS、AFlow、Meta-Harness、Self-Harness）：将 Agent 程序视为可搜索空间；本文继承其“冻结模型+程序级优化”范式，但针对视频场景引入原始视频重返与证据归属诊断。
- **VideoHarness-RSI**（Xu & Chen, 2026）：同样在冻结 VLM 上进化上下文构造程序；本文的关键区分是 Editor 具备主动调查接口以获取新观测，并显式将视觉帧成本纳入选择准则。
- **视频自我进化训练路线**（EvoGround、Video-Zero、EvoVid）：通过 proposer-solver 循环或 co-evolution 生成/精炼训练信号以更新模型；本文完全规避模型训练，所有改进沉淀于代码层。
- **观察策略学习**（FrameThinker、EVA）：通过 SFT/RL 让模型学会帧采样与空间分辨率分配；本文避免高昂训练代价，以离线进化+在线冻结部署实现同等目标。

## 局限性与未来方向
- **局限**：进化完全基于离线开发集选择，$\mathcal{D}_{\mathrm{g}}$ 与测试分布之间存在潜在偏移风险；仅验证了 DeepSeek-V4-Pro + Qwen3.6-Plus 组合，跨模型架构的泛化性未充分讨论；视觉成本仅统计成功视觉调用的帧数，未计入 OCR、转录、图查询等文本/逻辑模块的算力开销。
- **未来方向**：将主动调查扩展至跨视频检索与模式对比，以支持更具泛化性的规则提取；探索半在线/在线进化以适应流式或交互式视频场景；将成本度量统一为总 FLOPs、延迟或 API 调用预算；结合多 Editor 并行进化与归档机制提升搜索效率。

## 研究启发与可借鉴点
- **冻结模型下的 harness 自进化范式**：证明不依赖权重更新、仅通过修订可执行程序即可持续拉升下游表现，为低资源/
