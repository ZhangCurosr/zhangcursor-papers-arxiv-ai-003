---
title: "System1-Arena-Which-Decisions-Can-Be-Left-to-System-1"
source: https://arxiv.org/pdf/2610.21244v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-11-04 04:01:24"
field: "大语言模型评估与决策"
keywords: ["System-1决策", "概率校准", "统一基准", "System1-Arena", "Brier Score", "闭环代理", "成本-延迟权衡"]
innovations: ["提出System1-Arena统一决策基准,将92k题目映射到Boolean/Choice/Score三元接口", "设计无模型定制的两套prompt模板,首次横向比较System-1/闭源/开源LLM的概率决策能力", "揭示最大System-1模型(Vega-27B/Jev)可匹敌快速闭源LLM并在校准与成本上占优"]
benchmarks: ["System1-Arena", "SWE-bench Verified", "MiniWoB++", "WebShop", "ALFWorld", "MuJoCo Control", "BabyAI", "Baba Is AI"]
---

# 论文速读：System1-Arena: Which Decisions Can Be Left to System-1?

## 一句话总结
本文提出 **System1-Arena** 统一决策基准，对 92,125 道题目（105 个任务、16 个领域、138 个闭环环境）进行系统性评测，比较三类模型——专用 System-1 决策模型、闭源生成式 LLM、开源 LLM——在单次前向传播输出概率分布与自回归推理间的性能/成本/延迟权衡，发现最大 System-1 模型（Decision-2.0-Vega-27B、Jev）已能匹配甚至超越快速闭源 LLM，并在概率校准与单位成本上占据显著优势。

---

## 研究问题与动机

1. **System-1 与 System-2 的能力边界不清**：生成式 LLM 通过多步推理（System-2）获得高精度，但代价高昂；专用 System-1 决策模型只需单次前向传播即可输出选项概率分布，两者在真实决策场景中孰优孰劣尚缺乏统一基准。
2. **现有评测分散且格式异构**：CRUXEval、LegalBench、HotpotQA、SWE-bench、RewardBench 等数百个基准各自定义 prompt 格式与评分口径，无法跨任务横向比较模型在"概率决策"维度的一致性。
3. **概率校准与不确定性量化被低估**：多数工作仅报告准确率，忽略 Brier Score 等分布级指标；作者认为在需要风险感知的决策任务中，校准质量与准确率同等重要。
4. **成本-延迟-精度三维权衡缺乏实测数据**：闭源 LLM（如 Claude Opus 5.5）延迟 ~2,300 ms、成本 $10/千次；System-1 模型延迟 <350 ms、成本 <$0.1/千次，但系统性的横评尚未出现。

---

## 核心贡献（创新点）

1. **提出 System1-Arena 统一决策基准**：将 105 个任务、268 个公共来源、138 个闭环环境映射到同一 Boolean/Choice/Score 接口，首次实现跨 16 个领域、非代理 + 代理场景的横向对比。
2. **设计无模型定制的统一提示模板**：仅两套 prompt 分别适配闭源 LLM（JSON 概率输出）与开源 LLM（logits 归一化），不针对任何具体模型调优，保证评测可比性。
3. **建立概率决策的三维评测体系**：同时报告 Accuracy、Brier Score、Latency（ms）、Cost per 1k queries，揭示精度-速度-开销的帕累托前沿。
4. **揭示 System-1 模型的可行性边界**：证明 27B 参数规模的专用决策模型和轻量级 System-1（Jev, ~1B）可在多数非代理任务上匹敌 GPT-6 Luna 等快速闭源模型，并在校准与成本上碾压。
5. **开放全部 92k 道题目与闭环环境标注**：提供完整的数据集清单、label 来源（test execution / solver / expert annotation / original labels）与 tier-C 指令编写规范，促进后续研究复用。

---

## 方法详解

### 统一决策接口
- 每道题包含 `instructions + option descriptions`，类型分三类：
  - **Boolean**：yes/no，选项编码为 "true"/"false"
  - **Choice**：2–255 个选项单选
  - **Score**：2–10 级有序评分
- 所有模型接收相同的结构化输入，不依赖 dataset-specific prompt。

### 概率获取方式
| 模型族 | 概率提取方法 |
|--------|------------|
| System-1 专用模型 | API 直接返回 P(option) 分布 |
| 闭源 LLM（GPT/Claude/Grok） | JSON response 中 `v_k` 字段归一化 |
| 开源 LLM（Qwen/Gemma/Llama） | 在 answer token 起始位置取 logits，对选项标签做 softmax 归一化 |

### 评估协议
- 非代理任务：直接计算 accuracy + Brier Score
- 代理任务（闭环环境）：记录 episode-level success rate 与累积 reward；环境在模型推理期间持续演化，延迟计入性能
- 所有实验使用统一 seed，保留原始 instructions；无明确来源时使用 tier-C 指令编写准则补充说明

### 损失/训练目标（从背景推断）
- System-1 模型通常以 cross-entropy 或 Focal Loss 在大规模 choice 数据上后训练（post-training），直接学习 P(y|x) 而非自回归生成
- 本文未引入新损失函数，重点在于**评测协议**的设计与横评

---

## 实验与结果

### 非代理任务（主要结果，Top 模型）

| 模型 | 准确率 | Brier Score | 延迟 (ms) | 成本/千次 |
|------|--------|-------------|-----------|-----------|
| **Claude Opus 5.5** | **81.1%** | 0.232 | 2,293 | $10.38 |
| GPT-6 Astra | 80.1% | 0.282 | 2,189 | $15.05 |
| Grok 4.7 | 77.5% | 0.306 | 6,694 | $7.57 |
| GPT-6 Sol | 77.4% | 0.340 | 1,446 | $2.83 |
| **Decision-2.0-Vega-27B** | **73.0%** | **0.343** | 228 | 316 GPU-s |
| **Jev 1.13** | **72.6%** | **0.349** | 321 | **$0.052** |
| Microsoft-Decision-1 | 70.9% | 0.369 | 304 | $0.038 |
| OpenAI Decisions API | 68.6% | 0.424 | 140 | $0.111 |
| GPT-6 Luna | 71.2% | 0.461 | 1,276 | $0.143 |
| GPT-5.6 Luna | 70.0% | 0.458 | 1,375 | $0.301 |
| Gemma-4-31B（最佳开源） | 73.9% | 0.474 | 60 | 142 GPU-s |

### 核心发现
1. **Vega-27B (73.0%) 与 Jev (72.6%) 均超过 GPT-6 Luna (71.2%)**，证明最大 System-1 模型可匹敌快速闭源 LLM。
2. **概率校准显著更优**：Vega-27B/Jev 的 Brier Score（0.343/0.349）明显低于 GPT-6 Luna（0.461）；而 Gemma-4-31B 虽然准确率更高（73.9%），但 87% 的题目被赋予 ≥0.999 的预测概率，严重过置信。
3. **成本与延迟优势巨大**：Jev 约 $0.052/千次、延迟 321 ms，较 Claude Opus 5.5 成本降低 **200 倍**、延迟降低 **7 倍**。
4. **开源 LLM 在 logits 归一化下仍可战**：Gemma-4-31B 延迟仅 60 ms，但校准差；说明开源模型经后训练有望兼顾两者。

### 闭环代理任务（代表性结果）
- **MiniWoB++**：Jev 在 `buy-ticket`、`choose-list`、`click-button` 等基础交互任务达到 1.00 成功率；在 `book-flight-nodelay`、`email-inbox` 等需要多步规划任务上降至 0.00–0.20。
- **WebShop**：Jev 成功率 0.29，回报 0.6。
- **ALFWorld**：Jev 成功率 0.46，回报 0.5。
- **MuJoCo 控制**（half_cheetah/hopper/reacher 等）：Jev 得分波动极大（-44.9 ~ 306.6），反映单一前向决策在连续控制任务上的局限。
- **BabyAI / MiniHack**：`goto_obj` 0.70，`mazewalk_9x9` 0.40，其余多数任务为 0。
- **Baba Is AI**：带干扰规则的任务成功率 0.33–0.67，说明规则理解有一定上限。
- **SWE-bench Verified**：Jev 在 GPT-4.1-mini 版本上未完整显示；在 GPT-5.6-Luna 版本上 0.40/0.4。

### 结论
- **System-1 模型在单步、低规划深度的决策任务上完全可用**，且在成本-延迟-校准三角中占据 Pareto 优势。
- **需要多步推理、长期规划、复杂状态跟踪的任务（如 coding、复杂 web 导航、连续控制）仍依赖 System-2 生成式 LLM**。

---

## 相关工作脉络

1. **Decision-2.0 系列（TypeSafe AI, 2026）**：提出 Vega/Lux/Nox/Sol/Kai 等规模梯度 System-1 模型；本文在其基础上扩展评测规模，并首次将其与 Claude Opus / GPT-6 等顶级闭源 LLM 同场竞技。
2. **Microsoft-Decision-1（2026）**：基于 Qwen3.5-9B 后训练的快速决策模型；本文确认其在 Jev 与 Vega-27B 之间的定位（70.9% accuracy）。
3. **Router-Bench / Router-Arena（2024–2026）**：聚焦多模型路由调度；本文与之一脉相承但不再讨论路由，而是直接比较单模型在统一基准上的绝对能力。
4. **RewardBench / HelpSteer2 / CompassVerifier（2024–2026）**：评估奖励模型；本文引入 Brier Score 作为校准指标，补充了"决策质量不仅看准确率"的观点。
5. **Live-CodeBench / BigCodeBench / NL2SQL-BUGs（2024–2025）**：代码推理基准；本文在 loop/coding_controller 子任务中沿用 SWE-bench Verified，揭示 System-1 在此类任务上的短板。
6. **ALFWorld / WebShop / MiniWoB++ / TextWorld（2018–2020）**：经典 agent 环境；本文将其纳入统一 benchmark suite，并首次报告 System-1 模型在其中的成功率。

---

## 局限性与未来方向

1. **闭环任务探索深度不足**：多数游戏/控制任务成功率低或为零，可能受限于单步决策范式，未尝试 multi-round rollout 或 tree search。
2. **开源 LLM 的 logits 方法存在 tokenization 偏差**：答案 token 起始位置可能因分词差异导致概率估计失真，未来需研究 subword-aware calibration。
3. **未覆盖长视野规划任务**：如多轮对话客服、复杂编程调试、机器人连续控制等，System-1 在此类场景的有效性未知。
4. **成本模型简化**：GPU-s 与 $/千次的换算依赖实时云定价，未考虑本地部署与批量预取的 amortized cost。
5. **缺乏 ablation on prompt templates**：两套 prompt 的敏感性未系统评估，可能存在隐式模型偏好。

**未来方向**：
- 将 System-1 作为"快速初筛"嵌入 System-2 流程（如先由 Jev 决策，置信度低再交由 LLM 推理）
- 探索 multi-step System-1 rollout（类似 value iteration 的离散动作空间搜索）
- 扩展至多模态决策（视觉+语言联合输出概率）

---

## 研究启发与可借鉴点

1. **统一接口设计思路可直接迁移**：将异构 benchmark 映射到 Boolean/Choice/Score 三元组的方法，可用于团队内部的多任务评测框架，避免各自为政的 prompt 定制。
2. **三维评测（Accuracy / Brier / Latency×Cost）应成为标配**：在团队选型时同步报告这四维指标，有助于识别"高准确但过置信"或"低成本但校准差"的候选模型。
3. **Jev 1.13 的 cost-effectiveness 值得工程化验证**：若团队有大规模在线决策需求（如推荐、风控），可优先部署 Jev 或 Decision-2.0-Sol-2B 作为第一层过滤器。
4. **logits 归一化 + tier-C prompt 编写规范可作为开源模型微调的 baseline**：未来可用此协议快速复现开源模型的"出人意料的决策能力"。
5. **代理任务的 episode-level 指标设计**：区分 success rate 与 reward 两个维度，有助于识别"能完成任务但收益低"的模型行为模式。

---

## 关键术语表

**System-1 / System-2**：借鉴 Kahneman 认知双过程理论，System-1 指快速、并行、无需显式推理的单次前向传播决策；System-2 指慢速、串行、基于文本生成的多步推理。

**Brier Score**：衡量概率预测校准程度的指标，定义为预测概率与真实标签均方差的均值，越低表示校准越好（范围 0–1）。

**Choice 任务**：2–255 个选项的单选任务，是 System1-Arena 的核心评测类型，覆盖知识问答、法律判断、代码理解等。

**tier-C 指令**：当原始数据集缺少明确 instructions 时，由作者按统一准则编写的补充说明，用于保证 prompt 一致性。

**闭环环境（Closed-loop Environment）**：模型每一步决策都会改变环境状态，且环境在推理期间持续演化的交互设置，延迟会计入性能。

**logits 归一化**：对开源 LLM，在答案 token 起始位置提取 raw logits，对选项标签集合做 softmax，得到近似概率分布的方法。

**Pareto 前沿**：在多目标优化（精度/延迟/成本）中，无法在不恶化其他目标的前提下改进某一目标的解集合；本文 System-1 模型落于该前沿。

---

## 可复现要素

| 要素 | 状态 |
|------|------|
| 数据集（92,125 题、105 任务、138 闭环环境） | 论文声明开源（System1-Arena 应在 arxiv 附 GitHub 链接） |
| 代码 / 评测脚本 | 论文未明确提及，但 System1-Arena 名称暗示配套仓库 |
| 模型权重 | System-1 模型（Jev、Decision-2.0 系列）部分开源；闭源模型需 API 访问 |
| 关键超参 | 两套 prompt 模板（未列出具体 token），logits softmax over options |
| 随机种子 | 未明确声明 |

> 注：论文 arxiv ID 为 2610.21244v1，实际开源状态请以论文附录与官方 GitHub 为准。

---
