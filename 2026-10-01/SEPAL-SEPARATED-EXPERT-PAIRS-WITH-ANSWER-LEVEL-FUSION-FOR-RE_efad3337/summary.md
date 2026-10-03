---
title: "SEPAL-SEPARATED-EXPERT-PAIRS-WITH-ANSWER-LEVEL-FUSION-FOR-RE"
source: https://arxiv.org/pdf/2609.39645v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:46:31"
field: "大语言模型推理与多智能体协作"
keywords: ["Multi-agent collaboration", "Actor-Critic", "preference optimization", "late fusion", "LLM reasoning", "DPO", "self-consistency"]
innovations: ["私有 Actor-Critic 团队隔离修订 + 最终答案级多数投票", "三个角色（Direct/Evidence/Verification）通过训练目标引入结构化多样性而非仅靠采样", "以续写正确率为信号的 Critic/Actor 偏好学习在隔离团队上的独立应用"]
benchmarks: ["MMLU", "BoolQ", "BBH", "SciQ", "ARC"]
---

# 论文速读：SEPAL - SEPARATED EXPERT PAIRS WITH ANSWER-LEVEL FUSION FOR RELIABLE LLM COLLABORATION

## 一句话总结
SEPAL 将三个职责不同的独立 Actor–Critic 团队（Direct、Evidence、Verification）并行运行，各团队内部通过 Critic 引导的私有修订提升答案质量，最终仅对最终答案进行多数投票。该方法在五个开源模型（2B–8B）和五个 QA 基准上，平均相比单一 Actor–Critic 配对提升 **1.81 个百分点**。

## 研究问题与动机
- **共享讨论导致错误扩散**：多智能体辩论中，一个团队的早期错误会通过共享对话进入其他候选者，削弱投票所需的多样性。
- **自一致性缺乏反馈纠错**：Self-consistency 保持候选路径独立但无修正机制，无法修复系统性错误。
- **单对 Actor–Critic 无法产生候选集**：ACC-Collab 通过反馈优化单个答案，缺少可供投票的多候选集合。
- **投票的收益取决于"错误如何聚合"**：改进个别答案的同时，若所有候选走向同一错误方向，投票增益将归零；需要保留有意义的差异。

## 核心贡献（创新点）
- **私有修订 + 最终答案级投票的协作框架**：每个 Actor–Critic 团队独立推理与修订，仅在最终答案处聚合；与 Debate/SoM 等方法不同，团队间不共享推理历史。
- **三个角色分工引入结构化多样性**：Direct（直接推导）、Evidence（证据 grounding）、Verification（独立重解并校验替代选项）——通过训练目标而非采样随机性产生互补。
- **延续并独立应用 ACC-Collab 的偏好学习规则**：每对团队各自经历 Critic DPO → Actor DPO，以续写正确率为反馈质量评估标准。
- **零 Judge 的确定性后期融合**：两级多数投票（≥2 票即采用，否则回退至 Direct），不使用任何额外的评分模型。
- **系统化的消融与决策诊断**：报告了从 R0 到 R4 的每轮收益曲线、角色级准确率、共识覆盖率与 Oracle 选择差距，揭示收益来源与局限性。

## 方法详解
### 3.1 隔离的角色团队
- 三个团队 $i \in \{D, E, V\}$，每个含 Actor $A_i$ 和 Critic $C_i$。
- 初始化：$a_i^0 \sim A_i(\cdot|x)$；后续每轮：Critic 生成反馈 $c_i^t \sim C_i(\cdot|x, a_i^t)$，Actor 修订 $a_i^{t+1} \sim A_i(\cdot|x, a_i^t, c_i^t)$，共 4 轮修订。
- 团队间完全隔离，无信息交互。
- 最终答案 $z_i = g(a_i^4)$，$g$ 为确定性答案解析器。

### 3.3 角色初始化（Role SFT）
- 用 MMLU 辅助训练的 10,000 题，在每个角色下以温度 {0.4, 0.7, 1.0} 生成候选。
- 仅保留答案正确且未截断的候选，取三个角色均有合格答案的题目 ID 交集，确保各角色训练数据均衡。

### 3.4 续写价值偏好学习（Continuation-valued Preference Learning）
- 遵循 ACC-Collab 训练顺序：构建 Critic 偏好 → 训练 Critic → 构建 Actor 偏好 → 训练 Actor。
- 每个 state $(x, a)$ 采样自然反馈 $c^0$、正确导向反馈 $c^+$、错误导向反馈 $c^-$。
- 用 $K=10$ 次 Actor 续写估算反馈质量：$\widehat{R}(c|x,a)=\frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[g(a_k')=y]$。
- 有序 margin 规则（$\epsilon=0.6$）：若 $\widehat{R}(c^+)-\widehat{R}(c^0)\geq\epsilon$ 则保留 $(c^+,c^0)$；否则若 $\widehat{R}(c^0)-\widehat{R}(c^-)\geq\epsilon$ 则保留 $(c^0,c^-)$。
- DPO 损失：$\mathcal{L}_{\text{DPO}}=-\mathbb{E}\log\sigma(\beta[\log\frac{\pi_\theta(u^+|s)}{\pi_{\text{ref}}(u^+|s)}-\log\frac{\pi_\theta(u^-|s)}{\pi_{\text{ref}}(u^-|s)}])+\lambda\mathcal{L}_{\text{NLL}}$，其中 $\beta=0.1, \lambda=1$。
- LoRA rank=256, scaling=512。

### 3.5 无 Judge 的后期融合
$$\widehat{y}=\begin{cases}m,&|\{i:z_i=m\}|\geq 2\\z_D,&\text{otherwise}\end{cases}$$
- 固定回退策略，无需额外训练。

## 实验与结果
- **模型**：Llama-3-8B、Qwen2.5-3B、Gemma-2-2B、Phi-4-mini、Mistral-7B。
- **数据集**：MMLU（训练源）+ BoolQ / BBH / SciQ / ARC（迁移评估）。
- **基线**：Direct、Debate、SoM-2、SoM-4、ACC（单对 Actor–Critic）。
- **主要结果**：
  - SEPAL 在 25 个模型–数据集组合中，**24 个优于 ACC**，平均提升 **+1.81pp**（范围 +1.06~+2.22pp）。
  - **Phi-4-mini 达到最高宏观准确率 80.90%**；**Mistral-7B 增益最大（+2.22pp）**。
  - **唯一负值**：Gemma-2-2B BoolQ（−0.15pp）。
  - 在 BBH、SciQ、ARC 三个迁移集上，所有模型均优于 ACC（20/25 组中 19 组提升）。
- **关键诊断**：
  - 投票结果超越最强单一角色的占比：**21/25**。
  - Oracle-any-role 准确率（86.12%）vs. SEPAL（76.31%），存在 **9.81pp** 的选择差距。
  - 多数覆盖率达 96.41%，完全一致率 70.34%，说明投票有区分度。

## 相关工作脉络
- **Self-consistency (Wang et al., 2023)**：仅采样多样路径做投票，无反馈修正；SEPAL 在采样多样性基础上增加角色化反馈修订。
- **Multi-agent Debate (Du et al., 2024; Liang et al., 2024)**：共享对话历史导致错误传播；SEPAL 通过私有团队隔离避免跨候选污染。
- **ACC-Collab (Estornell et al., 2025)**：单对 Actor–Critic 通过续写正确率学习偏好；SEPAL 将此范式独立复制三遍并引入角色分工。
- **Tree of Thoughts (Yao et al., 2023)**：显式搜索+剪枝多解；SEPAL 在完整候选阶段之前不做中间评估，保留全部候选至投票层。
- **CoMM (Chen et al., 2024b)**：角色提示引入多样性但仍有协同讨论；SEPAL 在聚合前禁止角色间信息交换。
- **Mixture-of-Agents (Wang et al., 2025)**：分层聚合器融合多种模型输出；SEPAL 使用固定多数投票，无额外可训练模块。

## 局限性与未来方向
- **仅三个团队**：受限于最小奇数奇偶约束，团队数不可扩展至偶数投票场景。
- **训练–评估算力未对齐**：与 SoM/Debate 等基线比较时未统一推理调用次数。
- **固定 4 轮修订**：后续分析显示 R1 后边际收益递减（R1→R4 净增益仅 0.40pp），需要验证集驱动的停止策略。
- **Oracle 选择差距 9.81pp**：表明当前投票规则未能充分利用候选质量差异，可训练校准路由器或有改善空间。
- **短答案 QA 任务**：实验集中在选择题形式，复杂生成型任务的泛化性待验证。
- **种子鲁棒性未报告**：仅报告单次运行结果，未重复不同随机种子。

## 研究启发与可借鉴点
- **"私有反馈 + 晚期投票"的范式设计**：将反馈限制在团队内部是防止错误传染的有效机制，可迁移到多模型集成场景。
- **角色分工替代纯随机采样**：通过训练目标（而非仅采样温度）引入结构化多样性，比单纯增加样本数更高效。
- **续写正确率作为反馈质量信号**：用下游任务结果而非人工标注来评估 Critic 反馈，实现了端到端的训练信号链。
- **Oracle 选择差距可作为未来工作指标**：9.81pp 的 gap 明确指出了路由器/选择器的优化空间，可作为后续工作的目标函数基准。
- **可复现性设计优秀**：完整代码、权重、CSV 结果矩阵、单样本决策记录均已开源，且提供了无需推理即可重算结果的脚本。

## 关键术语表
- **SEPAL**：Separated Expert Pairs with Answer-Level Fusion，即本文提出的三团队隔离协作框架。
- **Actor–Critic 对 (ACC)**：Actor 生成答案，Critic 提供修订反馈，通过续写正确率进行偏好学习的训练范式（源自 ACC-Collab）。
- **DPO (Direct Preference Optimization)**：无需显式奖励模型的直接偏好优化方法，通过对比偏好对训练策略。
- **续写价值（Continuation-valued）**：以 Actor 基于反馈的下一轮回答正确率作为 Critic 反馈质量评估指标。
- **Late Fusion / 晚期融合**：各候选独立完成推理后仅在最终答案层聚合的方法论。
- **Oracle-any-role accuracy**：假设最优选择器能从所有角色候选中选出最佳答案时的理论上界准确率。
- **Majority coverage**：最终投票中存在至少两票相同答案的问题比例。
- **Role SFT**：按角色（Direct/Evidence/Verification）对 Actor 进行独立监督微调，初始化不同推理目标。

## 可复现要素
- **数据集**：MMLU、BoolQ、BBH、SciQ、ARC；均为公开基准，BBH 使用 1,260 题分层子集。
- **代码**：开源于 https://github.com/zhansan114514/SEPAL，含测试套件、训练/评估配置、结果 CSV、125 个压缩的样本级决策记录。
- **权重**：使用公开开源模型（Meta-Llama-3-8B-Instruct、Qwen2.5-3B-Instruct、Gemma-2-2b-it、Phi-4-mini-instruct、Mistral-7B-Instruct-v0.3），LoRA adapter 可复现。
- **关键超参**：LoRA rank=256, scaling=512；DPO $\beta=0.1$, $\lambda=1$；margin $\epsilon=0.6$；续写采样 $K=10$；温度 0.7（推理）；训练 3 轮 DPO、1 轮 SFT；max tokens=1024（生成）、4096（训练总限制）。
