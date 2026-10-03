---
title: "SEMIFACTUAL-CREDIT-AUGMENTED-POLICYOPTIMIZATION"
source: https://arxiv.org/pdf/2609.40360v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:46:24"
field: "大语言模型推理强化学习"
keywords: ["RLVR", "token-level credit assignment", "semifactual intervention", "GRPO", "spurious dependence", "reasoning robustness", "reinforcement learning for LLMs"]
innovations: ["提出 SCAPO，利用半反事实提示干预的固定回复概率漂移作为 token 级负向信用修正信号融入 GRPO", "首次系统性揭示 LLM token 级虚假依赖的异质性并通过 d-filtered decoding 零权重更新验证其诊断价值", "证明训练早期有限期的负向稳定性修正可显著提升数学推理与 OOD 泛化性能"]
benchmarks: ["AIME 2024–2026", "AMC 2023–2025", "HMMT 2025–2026", "GPQA-Diamond", "NoOp-AIME", "ThinkBench-AIME"]
---

# 论文速读：SEMIFACTUAL-CREDIT-AUGMENTED-POLICY-OPTIMIZATION

## 一句话总结
本文提出 **SCAPO（Semifactual Credit-Augmented Policy Optimization）**，一种因果启发的 GRPO 变体，通过在早期训练阶段利用"半反事实提示干预"（semifactual prompt interventions）测量固定回复中每个 token 的概率漂移，将稳定性信号以仅负向方式融入 GRPO 的优势函数，实现对 token 级信用分配的精化，从而减少模型对任务无关提示特征的虚假依赖。在 Qwen3-4B-Base 和 1.7B-Base 上，SCAPO 较 GRPO 分别将 AIME 2024–2026 准确率提升 5.63 和 4.17 个百分点。

## 研究问题与动机
1. **RLVR 对提示特征敏感**：尽管 RLVR（如 GRPO）显著提升了 LLM 推理能力，但模型输出仍对任务无关的提示特征（paraphrase、typo noise、无关场景包装等）高度敏感，存在 token 级的虚假依赖（spurious dependence）。
2. **GRPO 信用分配过于粗粒度**：GRPO 在同一组 rollout 的每个回复中，将所有 token 分配相同的 trajectory-level advantage，无法区分哪些 token 因依赖虚假特征而脆弱、哪些是真正有效的推理步骤。
3. **正向奖励可能强化虚假关联**：在 GRPO 中，正确答案对应的正向 outcome-derived advantage 会平等地强化回复中所有 token，包括那些因无关提示变化而大幅漂移的脆弱 token。
4. **缺乏细粒度信用信号**：现有 RLVR 方法在 token 级别缺少对"因果不变性"（causal invariance）的显式建模，难以在无需过程监督或外部 reward model 的前提下提升推理鲁棒性与泛化性。

## 核心贡献（创新点）
1. **首次通过半反事实提示干预系统性分析 LLM token 级虚假依赖**：冻结 Qwen3-4B-Base 模型，构造四种半反事实扰动（paraphrase/typo noise/scenario wrap/irrelevant context），量化每个 response token 的概率漂移，揭示不同 token 类别（如 reflection markers、discourse connectives）敏感度显著高于数学符号/数字。
2. **提出 SCAPO 的 token 级信用增强机制**：在 GRPO 框架内引入 detached negative-only stability correction，通过 teacher-forcing 固定回复在原始与扰动提示下的概率漂移来构建组内相对稳定性分数，并将其负向部分加到 GRPO advantage 上，选择性降低不稳定 token 的信用，但不额外奖励稳定性本身。
3. **无需过程监督或外部 Reward Model 的零成本信用细化**：与 CF-GRPO、FIPO 等依赖跨度掩码或未来 KL 的方法不同，SCAPO 仅需固定回复的 teacher-forced 概率比较，无需额外的推理链路标注或并行生成。
4. **在多个尺度模型上取得竞争级数学推理最优结果**：在 Qwen3-4B-Base 和 1.7B-Base 上均优于 GRPO/GSPO/SAPO/CF-GRPO/FIPO，在 AIME、HMMT、BRUMO 等八个数学基准及全部三个 OOD 基准上取得最好结果；SCAPO 在不到 GRPO 一半的优化步数内即达到 GRPO 的最终准确率。
5. **证明 d-filtered decoding 无需更新权重即可大幅提升推理准确率**：冻结模型权重下，通过抑制高漂移 token 候选进行在线解码，Qwen3-4B-Base 在 1,000 题诊断集上的准确率从 15.8% 提升至 30.0%（+14.2 pp），为 token 级信用信号的有效性提供直观佐证。

## 方法详解
### 半反事实干扰构造
对每个数学问题构造 4 类扰动（GPT-5.5 生成，保留数学表达式与答案）：
- **Paraphrase**：改写非数学自然语言部分
- **Typo noise**：单个单词插入/删除/替换/交换一个字符
- **Scenario wrap**：在原问题前后添加无关场景框架
- **Irrelevant context**：在末尾追加无意义干扰字符串（如 `[tag: mx93q]`）

### Token 概率漂移度量
对冻结策略 $\pi_{\mathrm{old}}$ 生成的固定回复 $y_i$，在原始提示 $x^{(0)}$ 和扰动提示 $x^{(k)}$ 下 teacher-force 同一 token $y_{i,t}$，计算有界对称距离：
$$d_{i,t}^{(k)} = \frac{|p_{i,t}^{(0)} - p_{i,t}^{(k)}|}{(p_{i,t}^{(0)} + p_{i,t}^{(k)})/2}, \quad k=1,\dots,K$$
该距离取值范围 $[0,2]$，对低概率 token 保留相对变化敏感性。

### 组内相对稳定性估计
对所有 $G$ 个回复的合法 token 位置，对每种扰动类型 $k$ 独立做组内 z-score 标准化后取负漂移（$-d^{(k)}$），平均后再做整体 z-score 标准化，得到组内相对稳定性分数 $\boldsymbol{u}$：
$$\boldsymbol{u} = \mathrm{zscore}_x\!\left(\frac{1}{K}\sum_{k=1}^{K}\mathrm{zscore}_x(-\boldsymbol{d}^{(k)})\right)$$
保留负向部分 $u_{i,t}^{-} = \min(u_{i,t}, 0)$，确保仅惩罚而非奖励稳定性。

### 带信用增强的策略优化
token 级增强优势：
$$\widetilde{A}_{i,t} = A_i + \lambda \cdot \mathrm{sg}(u_{i,t}^{-})$$
其中 $A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon_A}$ 为 GRPO 的轨迹级 advantage，$\lambda \geq 0$ 为增强强度，$\mathrm{sg}(\cdot)$ 为 stop-gradient。最终优化目标：
$$\mathcal{T}_{\mathrm{SCAPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum T_i}\sum_i\sum_t \min(r_{i,t}\widetilde{A}_{i,t},\; \bar{r}_{i,t}\widetilde{A}_{i,t})\right]$$
$\lambda$ 仅在前 $N_0$ 步启用（4B 模型：$\lambda_0=0.01, N_0=120$；1.7B 模型：$\lambda_0=0.01, N_0=200$），之后恢复标准 GRPO。

## 实验与结果
### 实验设置
- **训练数据**：DAPO-Math-17K（约 17,000 道竞赛级数学题）
- **模型**：Qwen3-4B-Base、Qwen3-1.7B-Base
- **训练步数**：4B 模型 600 步策略优化，1.7B 模型 1,000 步
- **Rollout**：batch size 128，每提示 8 条回复，max response length 16,384
- **评估**：temperature 0.7, top-p 0.9，Pass@16

### 主要结果（Qwen3-4B-Base）
| 基准 | GRPO | SCAPO | 提升 |
|---|---|---|---|
| AIME 24–26 | 22.08% | **27.71%** | **+5.63 pp** |
| AMC 23–25 | 64.71% | **65.65%** | +0.94 pp |
| HMMT 25–26 | 13.69% | **16.17%** | **+2.48 pp** |
| BRUMO 25 | 33.75% | **37.71%** | +3.96 pp |
| GPQA-Diamond（OOD）| 36.26% | **42.45%** | **+6.19 pp** |
| NoOp-AIME（OOD）| 20.56% | **23.61%** | +3.05 pp |
| ThinkBench-AIME（OOD）| 22.64% | **26.11%** | +3.47 pp |

在 4B 规模下，SCAPO 在 **6/8 个数学基准和全部 3 个 OOD 基准**上取得最优；在 1.7B 规模下，**所有 8 个数学基准和全部 3 个 OOD 基准**均最优。

### 消融结论
- **信号来源**：Random shuffle（打乱对齐）和 Counterfactual（答案改变扰动）均劣于 SCAPO，证明漂移与 token 位置正确对齐的重要性
- **符号选择**：Negative-only > All-sign > Positive-only，验证仅惩罚不稳定 token 而不奖励稳定 token 的设计正确性
- **效率**：半反事实探测仅占训练总时长的 1.8%

## 相关工作脉络
1. **GRPO (Shao et al., 2024)**：RLVR 的代表性工作，通过组内相对 reward 构建 trajectory-level advantage；SCAPO 在其基础上增加 token 级稳定性修正，解决 advantage 粗粒度的问题。
2. **CF-GRPO (Khandoga et al., 2026)**：通过 counterfactual 掩码推理跨度来估计 token 级 credit；区别在于 CF-GRPO 依赖额外的 span 重新生成，而 SCAPO 使用 fixed-response teacher-forcing，无需额外生成。
3. **FIPO (Ma et al., 2026)**：使用 discounted future KL 重加权 token advantage；区别在于 FIPO 基于未来 KL 发散，SCAPO 基于半反事实概率漂移，不依赖 future rollout。
4. **GRPO 的下游改进 (GSPO, SAPO, etc.)**：改进 clipping / normalization / importance ratio 的采样与损失设计；SCAPO 则从因果角度引入 token 级信用信号，与这些工作正交可结合。
5. **Token entropy / confidence / eligibility trace 类方法 (Wang et al., 2025b; Mou et al., 2026; Xie et al., 2026)**：利用模型自身不确定性或追踪信号进行 credit assignment；SCAPO 的独特性在于使用**提示扰动下的概率漂移**而非模型内部置信度。
6. **Spurious correlation / robustness 研究 (Mirzadeh et al., 2025; Huang et al., 2025; Fu et al., 2026)**：揭示了 RLVR 对 input perturbation 的敏感性；SCAPO 将这些发现直接转化为训练信号，而非仅停留在诊断层面。

## 局限性与未来方向
1. **模型规模有限**：实验仅在 1.7B 和 4B 密集中验证，尚未在更大模型（如 7B+/MoE）上评估。
2. **领域局限**：训练数据仅为数学推理（DAPO-Math-17K），仅通过 GPQA-Diamond 测试了跨域迁移，对代码、科学推理等其他领域的效果未知。
3. **干预类型固定**：当前仅使用 4 种预定义的扰动类型，不同领域可能需要定制化的扰动策略。
4. **早期阶段限定的信用增强**：$\lambda$ 仅在训练前 $N_0$ 步启用，后期完全依赖标准 GRPO，可能存在持续学习的空间。
5. **扰动依赖 LLM 生成**：使用 GPT-5.5 构造半反事实扰动，增加了对外部模型的依赖（尽管扰动在训练前预计算）。

## 研究启发与可借鉴点
1. **Token 级因果不变性可作为 RLVR 信用信号**：将"固定回复在扰动提示下的概率稳定性"作为训练时的 token 级辅助信号，这一思路可迁移至其他 RLVR 框架（如 PPO、REINFORCE）乃至非数学领域（代码生成、科学推理）。
2. **d-filtered decoding 作为零成本推理增强**：冻结模型下通过抑制高漂移 token 可大幅提升准确率（+14.2 pp），可作为推理时即插即用模块，无需额外训练。
3. **Negative-only correction 的设计范式**：不奖励稳定性、仅惩罚不稳定性的信号设计，有效避免了"伪稳定 token 被过度强化"的风险，这一"负向修正"原则可推广至其他 credit assignment 场景。
4. **早阶段信用塑形 + 后期标准优化的两阶段策略**：$\lambda$ 的有限期使用既引导了早期 trajectory 选择，又避免了后期训练偏移，该调度思想可复用于其他 auxiliary signal 的引入。
5. **与过程监督方法正交**：SCAPO 不依赖 process reward model 或 step-level annotations，可与 Process-SFT、PRM 等方法结合，形成"过程监督 + 因果信用增强"的双层 fine-grained 训练框架。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：基于可验证奖励的强化学习训练范式，通过规则/程序化 verifier 给出二元 reward，无需过程标注即可优化 LLM 推理能力。
- **GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的无 critic 的组内相对优势 RLVR 算法，每组 sampled responses 之间归一化 reward 得到 advantage。
- **Semifactual（半反事实）**：保持结果不变但改变原因的假设性干预，本文中指保留数学问题与答案、仅修改无关提示特征的扰动。
- **Token-level Credit Assignment（Token 级信用分配）**：将轨迹级 outcome reward 细分为每个 token 的独立学习信号，以区分有效推理步骤与虚假依赖 token。
- **Bounded-Symmetric Distance（有界对称距离）**：$d = \frac{|p^{(0)}-p^{(k)}|}{(p^{(0)}+p^{(k)})/2}$，用于度量概率漂移，对绝对值范围有界且对低概率 token 相对变化敏感。
- **d-Filtered Decoding**：推理时在每一步抑制累积概率质量不超过阈值的高漂移 token 候选，以 zero-weight-update 方式提升准确率。
- **Teacher-forcing Probing**：在扰动提示下强制模型输出与原始提示下相同的 token 序列，用于测量固定回复的概率稳定性。
- **Spurious Dependence（虚假依赖）**：模型输出对非因果/任务无关提示特征的异常敏感现象，可能导致分布外泛化性能下降。

## 可复现要素
- **数据集**：DAPO-Math-17K（公开）、AIME/AMC/HMMT/BRUMO/SMT/Omni-Math/Minerva/OlympiadBench/GPQA-Diamond（公开竞赛数据集）；半反事实扰动由 GPT-5.5 预生成
- **代码**：已开源，https://github.com/DtYXs/SCAPO（基于 EasyR1 框架）
- **预训练模型**：Qwen3-4B-Base 和 Qwen3-1.7B-Base（公开）
- **关键超参**：$\lambda_0 = 0.01$，$N_0 = 120$（4B）/ $200$（1.7B），学习率 $10^{-6}$，rollout batch=128，每提示 8 rollout，KL regularization 关闭，token-mean loss reduction，clipping $[0.20, 0.28]$
