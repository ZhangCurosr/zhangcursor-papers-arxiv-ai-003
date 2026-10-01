---
title: "SELF-ADAPTING-GROUP-OF-EXPERTS-FOR-MULTI-AGENT-REASONING"
source: https://arxiv.org/pdf/2609.35412v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:09:27"
field: "多智能体推理与协调"
keywords: ["multi-agent reasoning", "prompt adaptation", "strategy transfer", "sparse DAG", "training-free", "self-consistency"]
innovations: ["基于响应质量的策略捐赠者选择机制（答案一致性+前缀一致性+互惠评审）", "角色保持的免训练 prompt 重写实现跨智能体通用推理策略迁移", "得分驱动的动态稀疏 DAG 协作与跨阶段加权池投票聚合"]
benchmarks: ["MATH", "GSM8K", "GSM-Hard", "AQuA-RAT", "MMLU", "GPQA-Diamond", "MMMU-Pro", "MathVista"]
---

# 论文速读：SELF-ADAPTING-GROUP-OF-EXPERTS-FOR-MULTI-AGENT-REASONING

## 一句话总结
SAGE（Self-Adapting Group of Experts）是一种免训练的 multi-agent 推理框架，通过答案一致性、前缀一致性和互惠同行评审选出策略捐赠者，将其原始系统提示中的通用推理指导迁移给其他智能体（同时保留其角色身份），再沿动态稀疏有向无环图进行多轮协作，最终经加权投票得到答案。该方法在多个小模型骨干上显著超越单模型及 multi-agent 基线。

## 研究问题与动机
- 现有 multi-agent 框架主要通过修改上下文/通信拓扑来调节协作，但智能体的角色 system prompt 通常固定不变，导致当问题需要不同推理技能时，无法针对当前问题自适应调整个别智能体的推理策略。
- 更多讨论并不必然带来更好答案（Smit et al., 2024），已有方法如 GPTSwarm、AgentPrune、G-Designer、SelfOrg 等聚焦于图结构优化，却未触及"如何让某个智能体学到另一种推理视角"这一核心瓶颈。
- 小模型在特定任务上表现优异但对复杂多步推理仍力不从心（Magister et al., 2023），需要通过策略适配而非单纯增大模型规模来提升性能。
- 缺乏一种完全免训练、无需外部 judge 模型即可根据当前问题证据自动选择"最佳推理策略来源"的机制。

## 核心贡献（创新点）
1. **基于响应的策略捐赠者选择**：利用答案一致性（answer agreement）、前缀一致性（prefix consistency）和互惠同行评审构建综合得分，从初始独立响应中选出可作为策略模板的智能体；区别于仅靠嵌入相似性或固定规则选 leader 的 SelfOrg 等方法，这里直接以"响应质量+稳定性"作为选择依据。
2. **角色保持下的提示重写（策略迁移）**：用捐赠者的原始 system prompt 指导其他智能体的 prompt 改写，仅注入通用推理习惯与检查流程，不透露问题或答案；与 MAPRO、SPRIG 等 prompt 优化方法不同，这里无需训练、无需看到问题上下文，仅通过 LLM 将可复用策略跨角色迁移。
3. **基于得分的动态稀疏 DAG 协作**：每轮依据当前得分重建严格降序的稀疏有向无环图，每个智能体至多读取 K 个更高得分父节点的已更新响应；理论证明每轮边数有界、路径深度 ≤ K；与 MOC 的固定随机图或 G-Designer 的 learned GNN 图不同，SAGE 的图完全由当前响应自组织生成且无需训练。
4. **跨阶段加权池投票（Weighted Pool Vote）**：将初始响应、保留的同行评审响应及各轮修订响应统一 pooling，对每阶段 leader 赋予 1.5 倍权重进行加权多数投票；区别于仅取最后一轮答案的策略，该设计能回收在迭代中被"带偏"但仍正确的早期候选答案。

## 方法详解
**整体流程**（Algorithm 1，共 4 阶段）：

1. **初始响应与评分**：N 个智能体用各自模型 $\mathcal{M}_i$ 和 system prompt $s_i$ 独立回答 $x$，得到 $y_i^0$。计算综合得分：
   - **答案一致性**：$q(y,B) = \frac{1}{|B|}\sum_{b\in B}\mathbb{I}[\kappa(b)=\kappa(y)\neq\emptyset]$，衡量该答案在群体中的支持度。
   - **前缀一致性**：取响应前 $\tau=0.6$ 部分令模型续写，若 $\kappa(\bar{y})=\kappa(y)$ 则为 1，衡量答案的可复现稳定性。
   - 综合分：$\rho(y,B)=q(y,B)+\lambda\cdot z_\tau(y)$，$\lambda=0.5$。

2. **互惠同行评审与捐赠者选择**：按初始得分将智能体分为高分组 $H_m$ 和低分组 $L_m$（各采样 $m=2$ 个），组间两两进行 peer review（共享 critic prompt，只输出 KEEP/EDIT 及完整新响应）。对每个被评审智能体保留得分最高的候选响应 $y_i^\dagger$，再从这些保留响应中选得分最高者作为捐赠者 $\ell$，其原始 prompt $s_\ell$ 将被用于策略迁移。

3. **角色保持的提示重写**：对每个非捐赠者 $i$，调用其自身骨干 $\mathcal{M}_i$（温度 0.2）以固定指令重写 $s_i$：$\tilde{s}_i = R_{\mathcal{M}_i}(s_i, s_\ell)$。rewriter 仅输入两个原始 prompt，不接触问题或任何响应内容，确保只迁移"通用推理习惯"而非答案本身。捐赠者 prompt 不变（$\tilde{s}_\ell = s_\ell$）。

4. **稀疏 DAG 多轮协作**：每轮 $t$ 之前，对每个智能体 $i$ 选取最多 $K$ 个得分严格更高的父节点：$P_i^t = \text{TopK}_K\{j:\rho_j^{t-1}>\rho_i^{t-1}\}$，按得分递减（同分按索引递增）顺序更新响应：
   - 有父节点时：$y_i^t \leftarrow \mathcal{M}_i(x;\tilde{s}_i, (y_i^{t-1}, \mathbf{y}_i^t))$
   - 当前 leader（无入边）：$y_i^t \leftarrow \mathcal{M}_i(x;\tilde{s}_i, y_i^{t-1})$
   - 其他源节点：保留上一轮响应。
   每轮结束后重新评分 $\rho_i^t$ 并重建 DAG，直到达到轮次上限 $T=3$ 或所有答案收敛。

5. **加权池投票**：将所有阶段（初始、保留评审、每轮修订）的响应入池 $\mathcal{P}$，对每个答案键 $a$ 计算 $W(a)=\sum_{(t,i,y)\in\mathcal{P}}(1+\beta\cdot\mathbb{I}[i=i_t^\star])\cdot\mathbb{I}[\kappa(y)=a]$，$\beta=0.5$。返回权重最大者；同分时优先 leader 票多者，其次最新阶段，再次低索引。

**理论保证**（Lemma 2.1）：每轮 DAG 无环，每个智能体入度 $\le K$，总边数 $|E^t|\le KN-K(K+1)/2$，任意有向路径长度 $\le K$。

## 实验与结果
- **数据集**：MATH、GSM8K、AQuA-RAT、GSM-Hard、MMLU、GPQA-Diamond（文本推理）；MMMU-Pro、MathVista（视觉语言，仅与 SelfOrg 比较）。
- **骨干模型**：主实验使用 Qwen2.5-1.5B-Instruct 和 Ministral-3-3B-Instruct-2512；扩展实验覆盖 Qwen2.5-Instruct 0.5B–72B。
- **基线**：Single、CoT、MOC、MAD-M²、G-Designer（需微调 GCN）、SelfOrg（最接近竞品）。
- **核心结果（Table 1）**：
  - **Qwen2.5-1.5B**：SAGE AVG = **58.31%**，较最强 multi-agent 基线 MAD-M²（53.40%）提升 **+4.91 个百分点**，且六个基准全面领先；是唯一在该骨干上超越 Single（53.64%）的多智能体方法。
  - **Ministral-3-3B**：SAGE AVG = **75.08%**，较最强基线 MAD-M²（73.79%）提升 **+1.29 个百分点**，同样六基准全面领先。
  - 相对 SelfOrg：Qwen 上 +5.19pt，Ministral 上 +2.01pt。
- **缩放实验**：
  - 智能体数 4→9：Qwen 上 AVG +1.5pt，Ministral 上 +1.8pt。
  - 骨干 0.5B→72B：GSM-Hard 上 SAGE 在每个规模均优于 Single，提升 1.2–5.0pt；GPQA-Diamond 上 0.5B 增益最大（+8.58pt），但在 3B/32B 出现波动。
  - 混合团队（3×Qwen1.5B + 3×Ministral3B + 3×Phi-4-mini）在 GSM8K/MMLU 上达到 93.6/71.4，接近全 Ministral 团队（94.2/76.0），均值超过三种纯同质团队约 +5pt。
- **抗攻击**：9 个 Qwen1.5B 智能体中 3 个被恶意注入错误答案，SAGE 在五基准上均显著优于 SelfOrg（AQuA-RAT +5.8pt，MATH +4.3pt）。
- **消融**：去除 prompt 重写（SAGE-NOREWRITE）导致 AVG 下降 2.10–2.18pt；仅用最后一轮投票（Final-round WPV）导致下降 0.36–0.79pt；随机选捐赠者（SAGE-RANDOMDONOR）AVG 降至 56.72%（Qwen1.5B），验证了捐赠者选择的重要性。

## 相关工作脉络
1. **SelfOrg（Tastan et al., 2026）**：同样在每轮根据响应重建图结构，但依靠响应嵌入与群体质心对齐度决定连通性，且不修改角色 prompt；SAGE 在此基础上增加了策略迁移机制，使协作不只是"谁来听谁的"而是"怎么想"的改变。
2. **G-Designer（Zhang et al., 2025b）**：用 GNN 学习每问题专用的通信拓扑，需在前 200 题上微调；SAGE 完全免训练，靠瞬时响应质量自组织。
3. **MOC（Guan et al., 2026）**：采用固定随机 DAG 并保留多阶上下文；SAGE 的图由响应得分动态决定，信息流严格从高到低，避免低质量消息污染高层智能体。
4. **MAD-M²（Tian et al., 2026）**：通过 token 置信度记忆掩码过滤可能错误的历史消息，但不改变角色指令；SAGE 从根本上改变推理策略而非仅过滤噪声。
5. **SPRIG（Zhang et al., 2026a）** / **MAPRO（Zhang et al., 2026b）**：均在训练/优化层面改进 system prompt；SAGE 的 prompt 重写仅在推理时单次执行，不依赖梯度或外部反馈信号。
6. **Self-consistency（Wang et al., 2023）**：SAGE 的答案一致性评分直接源自该思想，但将其嵌入到 multi-agent 跨智能体的比较中，并结合前缀一致性作为稳定性补充。

## 局限性与未来方向
- **小模型场景增益大、大模型场景收益缩小**：在 3B/32B 等较强骨干上 SAGE 相对 Single 的优势减弱，甚至在 GPQA 上出现倒退，说明策略迁移对"错误模式丰富"的小模型更有价值，对强模型的边际收益有限。
- **互惠评审与 prompt 重写引入额外推理开销**：每问题需多次 LLM 调用（初始 N 次 + 至多 $m^2$ 次评审 + N 次重写 + N×T 次协作），总 token 消耗高于 Single/CoT。
- **角色池固定且规模有限**：当前仅 9 种预设角色，难以覆盖专业领域（如代码生成、法律推理）的复杂任务；角色选择不当时捐赠信号会变弱。
- **对抗鲁棒性测试条件受限**：实验仅评估了"固定错误答案注入"这一非自适应攻击，对更复杂的欺骗性策略（如渐进式误导、角色扮演攻击）未作评估（Ethics Statement 明确限定）。
- **前缀一致性与答案一致性作为质量代理指标存在噪声**：特别是 GPQA 等高难基准上多数智能体答错，一致性信号弱，导致捐赠者选择近乎随机（Figure 6 显示 GPQA 上无显著角色偏好）。
- **未来方向**：可扩展至视觉-语言多模态任务（已有初步实验，增益集中在 MathVista）；探索更丰富的角色池和领域适配；研究自适应轮次 $T$ 和 parent 数 $K$ 的调度策略。

## 研究启发与可借鉴点
1. **"策略迁移而非答案复制"的 prompt 重写范式**：rewriter 不接触问题/答案，仅从捐赠者 prompt 中抽取通用推理习惯（如双重检查变量赋值、边界情况验证），这一设计可迁移到任何需要跨角色知识共享的多 agent 场景，防止信息泄露和答案污染。
2. **双信号评分（一致性+稳定性）**：将答案一致性（横向群体证据）与前缀一致性（纵向自身稳定证据）结合为 $\rho = q + \lambda z_\tau$，为无 judge 场景下的响应质量估计提供了简洁有效的无监督方案，可复用于其他不需要 ground-truth 的聚合任务。
3. **严格降序稀疏 DAG 的理论可证性**：证明了每轮图无环、深度有界、边数有界，这种"自组织有界通信"模式可在资源受限的分布式 agent 系统中推广，避免通信爆炸。
4. **跨阶段加权池投票**：保留所有阶段的候选答案并赋予 leader 额外权重，为多轮推理系统提供了一种"防遗忘"的聚合策略，可被 Debate、ReConcile 等框架借鉴以降低迭代过程中的正确答案丢失率。
5. **混合团队的有效性**：弱骨干占比 2/3 的异构团队仍接近全强骨干团队表现，说明 SAGE 的自适应路由能有效识别并放大少数高质量信号；该结论提示在成本约束下可采用异构部署而非全量部署最强模型。

## 关键术语表
- **SAGE（Self-Adapting Group of Experts）**：本文提出的免训练 multi-agent 推理框架，通过响应驱动的策略捐赠者选择与动态稀疏图协作提升多智能体推理能力。
- **Strategy Donor（策略捐赠者）**：在初始响应阶段被选为"最佳推理策略来源"的智能体，其原始 system prompt 被用作其他智能体 prompt 重写的模板。
- **Answer Agreement（答案一致性）**：衡量某一答案在多个响应中出现的频次比例，源自 self-consistency 思想，作为横向群体支持度的无监督信号。
- **Prefix Consistency（前缀一致性）**：将响应截断为前 $\tau$ 部分后令同一模型续写，检查是否得到相同答案，用于评估响应内部推理链的稳定性。
- **Reciprocal Peer Review（互惠同行评审）**：高分组与低分组智能体两两交叉评审对方响应，决定是否 KEEP 或 EDIT 自己答案，产生候选响应集合用于捐赠者选择。
- **Sparse DAG（稀疏有向无环图）**：每轮根据智能体当前得分严格降序构建的通信拓扑，每个智能体最多连接 $K$ 个更高得分父节点，保证无环与有界深度。
- **Weighted Pool Vote（加权池投票）**：将所有推理阶段（初始、评审保留、各轮修订）的候选答案汇总，对 leader 响应赋予 1.5 倍权重进行加权多数投票以输出最终答案。
- **System Prompt Rewriting（系统提示重写）**：仅以两个原始 role prompt 为输入、不接触问题或答案，由目标智能体自身 backbone 生成融合捐赠者通用推理习惯的新 prompt 的过程。

## 可复现要素
- **数据集**：MATH、GSM8K、AQuA-RAT、GSM-Hard、MMLU、GPQA-Diamond、MMMU-Pro、MathVista——均为公开基准。
- **代码**：论文声明代码已开源（Project Page: https://www.atifquamar.com/sage-page；文末 "Our code is available here"）。
- **权重**：使用公开模型 Qwen2.5-1.5B-Instruct、Ministral-3-3B-Instruct-2512、Qwen2.5-VL-3B-Instruct、Phi-4-mini，均公开可下载。
- **关键超参**：
  - 智能体数 $N=4$（主要实验）/ $N=9$（扩展实验）
  - 每轮父节点上限 $K=2$（主要）/ $K=3$（扩展）
  - 最大轮数 $T=3$
  - 同行评审采样数 $m=2$
  - 前缀比例 $\tau=0.60$
  - 前缀一致性权重 $\lambda=0.50$
  - leader 投票权重系数 $\beta=0.5$
  - 初始/修订/评审温度 0.50；前缀温度 0.60；重写温度 0.20
  - top-p=0.80；top-k=20（Qwen）/ 禁用（Ministral）
  - 重复惩罚 1.10（Qwen）/ 1.0（Ministral）
  - 每调用最多生成 2048 tokens，上下文窗口 32768 tokens
  - 答案提取器：xFinder-qwen1505；评判器：xVerify-0.5B-I
