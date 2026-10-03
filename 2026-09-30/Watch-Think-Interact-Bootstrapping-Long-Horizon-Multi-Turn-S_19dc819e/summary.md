---
title: "Watch-Think-Interact-Bootstrapping-Long-Horizon-Multi-Turn-S"
source: https://arxiv.org/pdf/2609.37035v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:42:57"
field: "流式视频理解与多轮推理"
keywords: ["streaming video reasoning", "multi-turn video QA", "reinforcement learning", "video large language model", "memory-augmented reasoning", "source-video recall"]
innovations: ["提出闭环Watch-Think-Interact框架，通过时序索引记忆与选择性源视频回忆解决前向压缩导致证据丢失的问题", "构建WTI-82K因果对齐多轮轨迹数据集，覆盖实时/回溯/前瞻三类流式交互", "设计Stream-GDPO轨迹级强化学习，对outcome/format/recall/memory四类奖励分别归一化后聚合以对齐状态化流式决策"]
benchmarks: ["OVO-Bench", "StreamingBench", "Video-MME", "MLVU", "LongVideoBench"]
---

# 论文速读：Watch-Think-Interact-Bootstrapping-Long-Horizon-Multi-Turn-S

## 一句话总结
本文提出 Watch-Think-Interact (WTI) 框架，通过有界主动窗口感知、时序索引记忆与选择性源视频回放的闭环机制，解决长视距多轮流式视频推理中"前向压缩导致证据丢失"的核心问题；结合 WTI-82K 因果对齐交互数据与 Stream-GDPO 轨迹级强化学习，在开源流式模型中刷新 StreamingBench (83.3%) 与 OVO-Bench (73.6%) 的 SOTA。

---

## 研究问题与动机

- **核心问题**：流式视频多轮推理中，问题异步到达，模型仅能观测已到的视频前缀；现有方法的在线状态构建早于后续问题可知，前向压缩丢弃的细节在后续相关问题出现时无法恢复。
- **现有方法的不足**：
  - 多数 Video-LLM 假设离线完整视频可用，不满足因果流式约束；
  - 已有流式模型采用的前向压缩 / 单向 KV 状态 / 压缩视觉 token 会在问题到达前固定状态，导致"后知后觉"类问题（需回溯历史或等待未来证据）性能骤降；
  - 黑盒潜层摘要使模型难以判断"当前记忆是否仍含所需证据、何时该重新回忆"。

---

## 核心贡献（创新点）

1. **闭环流式推理框架 WTI**：将紧凑自然语言记忆与源视频时间范围绑定，支持直接推理与选择性视觉回忆，避免完整历史回放。
   - *与已有工作的本质区别*：前人工作依赖单向压缩或固定状态，WTI 在每步通过 "silence / response / recall" 动作形成行动–反馈–重决策闭环，证据随需可调取。

2. **时序索引记忆 + 主动窗口机制**：记忆条目同时提供语义摘要（直接推理）与源时间范围（精确定位），控制器在边界触发压缩更新，有界上下文不随流增长。
   - *与已有工作的本质区别*：区别于全量 KV 缓存或层次压缩，WTI 的记忆是"时间-来源锚定"的文本索引，支持对未保留区间的精确重访。

3. **WTI-82K 因果对齐多轮轨迹数据集**：构建 82,335 个定时问题、4,812 条轨迹，覆盖 Real-Time、Backward Tracing、Forward/Proactive 三类交互，支持"回答时机、等待、回忆、记忆更新"的联合监督。
   - *与已有工作的本质区别*：不同于孤立视频 QA 对，WTI-82K 以共享流视角组织数据，早期动作塑造后期状态，提供轨迹级训练信号。

4. **Stream-GDPO 轨迹级强化学习**：在完整多轮 rollout 上优化，分别归一化 outcome / format / recall / memory 四类奖励后再聚合，使训练与状态化流式决策对齐。
   - *与已有工作的本质区别*：相比 GSPO 等方法，Stream-GDPO 在 chunk-level 流式环境中实现滚动信用分配，并针对回忆与记忆更新引入专用奖励项。

---

## 方法详解

### 1. 在线问题定义
- 视频按分块顺序到达：$V = (c_1, \dots, c_T)$，步骤 $t$ 只能见前缀 $c_{\leq t}$。
- 多问题 $\mathcal{Q} = \{(q_r, \tau_r, \alpha_r)\}$，其中 $\tau_r$ 为提问时刻、$\alpha_r$ 为答案可给出时刻（推理时隐藏）；模型须在 $\hat{t}_r \geq \tau_r$ 输出答案 $\hat{y}$，且必须基于 $c_{<\hat{t}_r}$。

### 2. WTI 闭环推理框架
- **可见状态**：$s_{t,k} = (W_t, M_t, Q_t, H_t, E_{t,k}^{\text{rec}})$，分别为主动视觉窗口、紧凑文本记忆、当前活跃问题集、交互历史、已回忆证据。
- **子回合决策**：先产生短思考更新 $z_{t,k}$，再由策略 $\pi_\theta$ 选择动作：
  - `silence`：暂不回答，推进当前 chunk 结束；
  - `response(q_r, y)`：回答问题；
  - `recall(τ_s, τ_e)`：召回已观区间（绝对时间戳，不超过当前 chunk 起始时间 $\gamma_t$）。
- **回忆的非终止循环**：recall 将源视频证据加入 $E^{\text{rec}}$，在同一时间步 $t$ 再次更新思考并重新决策，不推进流。
- **记忆压缩**：在控制器触发边界 $b$ 时，将累积记录 $Z_b$ 写入：
  $$M_{\text{post}} = \text{Compact}_\theta(M_{\text{pre}}, Z_b), \quad \|M_{\text{post}}\|_{\text{tok}} \leq B_M$$
  每条记忆包含语义摘要 + 源时间范围。

### 3. Masked Supervised Fine-Tuning (SFT)
- 视觉上下文限制在最近 $K_v=8$ 个 1 秒 chunk（共 16 帧），不暴露全前缀。
- 采用 stream-causal mask：助手 token 仅对更早步骤因果可见；标签 mask 只对助手生成 token 计算 loss。
- 目标函数：
  $$\mathcal{L}_{\text{SFT}}(\theta) = \lambda_{\text{act}} \bar{\ell}_{\text{act}} + \lambda_{\text{state}} \bar{\ell}_{\text{state}}$$
  前者覆盖 silence/response/recall 等稀疏动作目标，后者覆盖思考与记忆更新等较长 span 目标。

### 4. Stream-GDPO 轨迹级强化学习
- **四类奖励设计**：
  - **Outcome**：$R_{\text{out}}^{i,q} = \max_{y \in \mathcal{V}_q} \mathbb{I}(\text{match}(\hat{y}_{i,q}, y))$，通过形式感知验证器检查因果响应时机。
  - **Format**：$R_{\text{fmt}}^{i,q}$ 验证动作是否符合流式语法。
  - **Recall 工具奖励**：$R_{\text{tool}}^{i,q} = \ell_q \cdot u_{i,q} \cdot R_{\text{out}}^{i,q}$，仅在教师召回区间内且语法/时间合法时，配合正确答案才给予奖励。
  - **Memory 更新奖励**：$R_{\text{mem}}^{i} = \frac{1}{2}[S_{\text{time}}(M_{\text{post}}^i)-S_{\text{time}}(M_{\text{pre}}^i) + S_{\text{keep}}(M_{\text{post}}^i)-S_{\text{keep}}(M_{\text{pre}}^i)]$，衡量时间索引质量与关键信息保留的提升。
- **组内归一化优势聚合**：
  $$A_i = \sum_{c \in \{\text{out, fmt, tool, mem}\}} w_c \frac{R_c^i - \mu_c(\mathcal{G})}{\sigma_c(\mathcal{G}) + \epsilon_{\text{norm}}}$$
  所有助手动作回合共享同一轨迹优势，实现轨迹级信用分配。
- **策略优化**：采用 clip-PPO 形式：
  $$\mathcal{I}_{\text{GDPO}}(\theta) = \mathbb{E}_{i,j}[\min(\rho_{i,j} A_i, \tilde{\rho}_{i,j} A_i) - \beta D_{i,j}]$$

---

## 实验与结果

### 数据集与基准
- **数据集**：自建 WTI-82K（82,335 个定时问题 / 4,812 条轨迹，56.9% 轨迹 ≥120s，21.7% ≥240s，平均每条 17.1 题）。
- **评测基准**：OVO-Bench（Real-Time / Backward / Forward 分组 + 加权 Overall）、StreamingBench（10 个视觉理解子类）、Video-MME、MLVU、LongVideoBench。

### 主要结果（开源流式模型对比）
- **OVO-Bench**：WTI-8B 取得 **73.6%** Overall，领先 StreamBridge-7B (62.6%) 约 +11.0pp；Real-Time 76.9%、Backward 73.4%、Forward 70.0%，均在开源流式模型中第一。
- **StreamingBench**：WTI-8B 达 **83.3%**，领先 VST-7B (79.5%)、StreamForest-7B (77.3%)；在 OP、CS、ATP、EU、TR、SU、ACP 等多个子类领先。
- **离线长视频**：Video-MME 66.9%、MLVU 69.9%、LongVideoBench 62.9%，在开源流式模型中保持最佳。

### 消融结论
- **SFT 初始化**：将 OVO-RT 从 64.8% 提升至 74.2%，但对 Backward/Forward 提升有限，证明单一 imitation 不足以学习长程决策。
- **Stream-GDPO vs GSPO**：归一化多奖励后，OVO-BT 从 65.3% → 73.4%，OVO-FT 从 60.7% → 70.0%。
- **记忆与回忆互补**：移除记忆使 OVO-BT 降至 62.8%，移除回忆降至 55.5%，memory-only 仅 48.2%；离线同样证实两者不可替代。

---

## 相关工作脉络

1. **流式视频理解**：VideoLLM-online、StreamForest、StreamBridge 等聚焦"在观察前缀上决定何时回答"，而 WTI 进一步解决"后续问题需要的证据已离开窗口、需回溯或等待"的场景。
2. **长流记忆管理**：Flash-VStream、VideoChat-Flash 等采用视觉 token / 层次压缩或 KV 剪枝；WTI 不同在于记忆是时间锚定的文本索引，并可通过 recall 重新加载源视频，而非永久丢弃。
3. **Agentic 多模态推理**：DeepEyes、MemAgent 等通过工具调用或记忆代理更新状态；WTI 将这些能力引入流式视频，并针对 streaming causality 设计 recall/silence/response 动作空间。
4. **流式评估基准**：OVO-Bench、StreamingBench、SVBench 定义了时间戳问答与多轮交互协议；WTI-82K 在此基础上引入轨迹级因果对齐，使训练与评估环境一致。
5. **RL 对齐 Video-LLM**：GDPO、GSPO 等用于视觉语言模型的对齐优化；Stream-GDPO 扩展至完整多轮 rollout，并针对流式特有的 recall 与 memory 引入独立归一化奖励项。

---

## 局限性与未来方向

- **数据规模与多样性**：WTI-82K 虽覆盖多类别交互，但仅 4,812 条轨迹，场景分布、语言多样性和实时流噪声可能不足；未来可扩展至更大规模、更具挑战性的真实流媒体数据。
- **单次回忆限制**：每个 chunk 最多一次 recall、最多返回 4 个 chunk，对需要多段远距离证据的综合问题可能受限，可探索多级回忆或分层检索。
- **记忆压缩策略**：当前 Compact 由模型端到端学习，缺乏对"哪些信息应永久保留 / 何种压缩损失最小"的理论保证；未来可结合结构化知识图谱或外置向量库。
- **奖励设计手工成分**：$S_{\text{time}}$、$S_{\text{keep}}$ 等记忆质量评估依赖启发式打分器，可能引入偏差；可探索自一致性或人类偏好校准。
- **评估与部署差距**：当前实验为离线模拟流式（chunk-by-chunk），真实流媒体存在丢帧、时钟不同步等噪声，鲁棒性待验证。

---

## 研究启发与可借鉴点

1. **闭环动作空间设计**：将响应决策显式建模为 silence/response/recall 三类动作，并在同一时间步形成"行动–反馈–重决策"循环，值得迁移至其他时序交互任务（如长音频、实时文档检索）。
2. **轨迹级多奖励归一化**：Stream-GDPO 对不同维度奖励分别组内归一化再聚合，避免了高方差 outcome 淹没弱信号；该方法可与团队现有 RLHF/GDPO 管线直接对接。
3. **时间锚定记忆 + 源视频检索**：将"语义摘要"与"时间范围"捆绑作为记忆条目，并在需要时通过绝对时间戳召回原始视觉块，这一设计兼顾了压缩率与证据可追溯性。
4. **SFT 到 RL 的一致性环境**：Masked SFT 与 Stream-GDPO 共享 chunk 划分、记忆预算、回忆工具与动作语法，避免训练–评估分布偏移；可作为后续研究的标准实践。
5. **因果对齐的数据构造范式**：WTI-82K 从证据链出发反向生成 query/answer 时间戳，并自动校验时间边界、答案可及性与动作语法；此类 pipeline 可复用至其他流式多模态数据集构建。

---

## 关键术语表

**Watch-Think-Interact (WTI)**：论文提出的闭环流式视频推理框架，整合主动窗口感知、紧凑时序记忆与源视频选择性回忆。

**Stream-GDPO**：将 GDPO 的多奖励组内归一化扩展至流式 chunk 环境，在完整多轮 rollout 上优化响应时机、回忆与记忆更新的轨迹级 RL 目标。

**WTI-82K**：作者构建的流式多轮交互数据集，含 82,335 个定时问题、4,812 条因果对齐轨迹，覆盖实时、回溯与前瞻三类交互。

**Active visual window ($W_t$)**：当前步骤模型可见的最近 $K_v$ 个视频 chunk（文中设为 8 个 1 秒 chunk，共 16 帧），构成有界感知上下文。

**Time-indexed compact memory ($M_t$)**：带时间边界的文本摘要记忆，每条记录包含语义要点与源视频时间范围，支持后续精确回忆。

**Selective source-video recall**：通过 `recall(τ_s, τ_e)` 动作按需加载已观测但未在当前窗口内的源视频 chunk，不推进时间流。

**OVO-Bench / StreamingBench**：用于评估流式视频理解的基准；OVO-Bench 区分 Real-Time/Backward/Forward 三类任务，StreamingBench 覆盖 10 项细粒度视觉理解子任务。

**Masked SFT**：在流式因果状态下进行的监督微调，仅对助手生成的动作和状态更新 token 计算 loss，并限制视觉输入为最近窗口。

---

## 可复现要素

- **数据集**：WTI-82K 论文声明已构建，但正文未明确说明是否完全开源（建议查阅附录/项目页）；OVO-Bench、StreamingBench、Video-MME、MLVU、LongVideoBench 均为公开基准。
- **代码/权重**：论文宣称 WTI-8B 为开源 SOTA，基座使用 Qwen3-VL-8B-Instruct；具体代码仓库在论文正文末未给出链接，需在附录或项目主页确认。
- **关键超参**：视觉窗口 $K_v=8$（1 秒/chunk，2 FPS）；回忆每 chunk 最多 1 次、最多返回 $K_r=4$ chunk；SFT 学习率 $1 \times 10^{-5}$（冻结视觉编码器，更新 LLM 与投影）；RL 使用 veRL，prompt batch=64，每 prompt 4 rollouts，actor LR $1 \times 10^{-5}$，非对称 clip $(\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28)$，action-branch loss weight=0.15，8×H20 GPU，1 epoch。
- **训练环境**：视频以 1 秒 chunk 流式输入，推理与训练共享相同 chunk 划分、记忆预算、回忆工具与动作语法。

---
