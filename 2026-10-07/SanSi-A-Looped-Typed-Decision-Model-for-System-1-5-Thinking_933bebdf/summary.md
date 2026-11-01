---
title: "SanSi-A-Looped-Typed-Decision-Model-for-System-1-5-Thinking"
source: https://arxiv.org/pdf/2610.07730v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-11-01 14:25:03"
field: "大模型高效推理与决策"
keywords: ["System 1.5", "typed decision model", "循环语言模型", "proper scoring rule", "GRPO", "温度校准", "2WikiMultiHopQA"]
innovations: ["提出循环 typed decision 框架实现 System 1.5 思维，支持 anytime 读取", "冻结骨干仅微调 LoRA+readout（61M参数），用计算换精度", "跨分布温度校准显著降低 ECE 至 0.019，并在 2WikiMultiHopQA 上 F1 提升 +7.7"]
benchmarks: ["2WikiMultiHopQA", "59-source decision benchmark", "10,027-item typed decision test set"]
---

# 论文速读：SanSi-A-Looped-Typed-Decision-Model-for-System-1-5-Thinking

## 一句话总结
本文提出 **SanSi**，一种基于循环类型决策（looped typed decision）的 System 1.5 思维模型——在单次前向传播中通过多次循环迭代修正隐藏状态，最终输出选项概率分布而非生成文本；在同一模型上通过冻结骨干仅微调 LoRA + readout（共 61M 参数），即可在多种决策任务与多跳问答生成中实现显著精度提升与良好温度校准。

## 研究问题与动机
1. **现有决策模型依赖文本生成**：传统方法需逐 token 生成答案，计算成本高且无法在同一前向传播中并行评估多个选项的概率。
2. **System 1 与 System 2 之间缺乏"灰度"机制**：纯 System 1（单次前向）速度最快但精度受限；纯 System 2（多步生成）精度高但算力昂贵；缺少一种可在两者间灵活调节预算的方案。
3. **循环结构尚未被系统用于决策任务**：虽然循环 LM（如 Ouro）已在语言建模中验证有效，但将其用于 typed decision 并支持 anytime 读取的研究仍空白。
4. **校准缺失影响可靠性**：高精度模型若缺乏可靠概率输出，在下游应用（如 reward shaping、agent 选择）中难以直接使用。

## 核心贡献（创新点）
1. **提出 System 1.5 循环 typed decision 框架**：通过 $T$ 次循环（$T \in [1,8]$）在单次前向传播内迭代修正隐藏状态，用计算换精度而非参数。
2. **Anytime 读取设计**：一次 8 循环前向传播同时产出 1–8 次循环的所有决策概率，支持按预算灵活截断。
3. **极低参数量微调方案**：冻结 Ouro 骨干（1.4B/2.6B），仅训练 rank-64 LoRA + 每个循环的 rank-16 readout（SanSi 共 61M 参数，占比不到 5%）。
4. **联合 proper scoring rule 训练**：每个循环均使用交叉熵 + Brier score 联合优化，确保概率输出既准确又校准。
5. **在 2WikiMultiHopQA 生成任务中验证闭环效用**：Loop 8 相对未训练基线 F1 提升 +7.7，且通过跨组温度校准将 ECE 降至 0.019±0.005。

## 方法详解
### 1. 基座与循环结构
- 基座采用 **Ouro**（Zhu et al., 2025）——预训练为循环语言模型；提出 **SanSi**（Ouro-1.4B）与 **SanSi-2.6B** 两档尺寸。
- 循环状态更新：$h_t = N(F_\theta(h_{t-1}))$，其中 $N(\cdot)$ 为归一化操作，$F_\theta$ 为冻结的 24 层 transformer 堆栈；归一化状态 $h_t$ 直接作为下一循环输入，不重新注入 token。

### 2. Readout 设计
- 第 $t$ 循环的 logit：$\ell_t = (W + U_t V_t) z_t / \tau_t$，其中 $z_t$ 为循环输出 hidden state，$W$ 为冻结 LM head 中选项字母对应行（如 A/B/C/D），$U_t V_t$ 为 rank-16 低秩校正矩阵，$\tau_t$ 为可训练温度缩放。
- 预测：$p_t = \text{softmax}(\ell_t)$。

### 3. 训练目标
- 每个循环 $t$ 均使用 **交叉熵 + Brier score** 的 proper scoring rule 联合损失（式 3），迫使模型同时优化分类准确性与概率校准。
- 训练规模：1,000 步，每步 16 样本，$T=8$ 次循环；训练数据共 12,800 条决策，涵盖 Classification（2,400）、Multi-step reasoning（5,500）、Uncertain evidence（2,300）、Long documents（1,000）、Sentence pairs（1,600）六类。

### 4. Generator-Verifier 闭环（第5段）
- **Generator**：18.1M 可训练参数（LoRA adapters）。
- **Verifier**：即主模型 seed 0，以训练 prompt 格式调用，提供 reward signal。
- **数据集**：2WikiMultiHopQA（8K train / 500 dev / 3K test），每题含 5 段（supporting + distractors）。
- **训练**：400 GRPO steps；每 step 抽 8 题、每题采 8 个答案；Advantage 在 8 个答案内归一化，每 batch 更新一次。
- **答案匹配**：归一化后相等或 token F1 ≥ 0.8 视为相同；不足 4 个不同答案时补充段落短文本。

## 实验与结果
### 决策任务（Table 1–36，摘要摘要）
- 测试集：10,027 条来自 59 个来源的决策（含 231 个公开 benchmark）。
- Loop 8 在多数任务上显著优于 Loop 1/2/4；Loop 4 与 Loop 8 差距较小（+1.4 平均 F1）。

### 温度校准（Table 37）
| 设置 | Loop 8 Acc | ECE all |
|---|---|---|
| 无校准 | 72.0 ± 0.7% | 0.093 |
| In-distribution 校准 | Mean loops 69.9% | 0.066 |
| **All groups 校准（最优）** | **71.9 ± 0.8%** | **0.025 ± 0.003** |
| Mean loops + temp（all groups） | 71.5 ± 0.8% | **0.019 ± 0.005** |

- 近迁移/远迁移分组均受益，ECE 降低约 4×。

### Generator 性能（Table 38–39）
| Reward | F1 | ΔF1 [95% CI] | Reward AUROC | EM | Contains | Answer words |
|---|---|---|---|---|---|---|
| Not trained | 39.5 | — | — | 32.5 | 34.5% | 2.6 |
| Loop 1 | 29.1 ± 2.4 | −10.5 | 0.779 | 9.6 | 38.5% | 5.5 |
| Loop 2 | 39.5 ± 1.8 | −0.0 | 0.857 | 19.6 | 45.0% | 4.3 |
| Loop 4 | 45.8 ± 4.2 | +6.3 | 0.914 | 29.4 | 49.7% | 3.9 |
| **Loop 8** | **47.3 ± 1.1** | **+7.7** | **0.905** | **32.2** | **49.7%** | **3.6** |

- **按题型（Loop 8）**：Inference +21.1（最大提升）、Bridge comparison +3.4、Compositional +7.6、Comparison −1.2。
- **多 seed 稳定性**：Loop 8 跨三 seed F1 = 46.5–48.5；Loop 4 = 41.0–48.3；Loop 1/2 显著更差。
- Loop 4 与 Loop 8 无法区分（统计上），Loop 1/2 显著劣于其余。

## 相关工作脉络
1. **System 1 vs System 2 思维架构**（Kahneman, 2011；后续 LLM 工作）：本文提出中间态 System 1.5，填补单次前向与多步生成之间的空白。
2. **循环语言模型**（Ouro, Zhu et al., 2025）：本文继承其循环预训练机制，首次将其扩展到 typed decision 任务。
3. **Typed Decision Models**（如 GPTFusion、DirectQA 等）：此类方法通常直接输出概率，但缺乏循环修正能力；本文在 same-forward 内引入多循环迭代。
4. **Proper Scoring Rules 在 LLM 中的应用**：Brier score + CE 联合训练并非首创，但本文将其系统应用于循环 readout 的每个时间步。
5. **GRPO / RLHF for 生成器-验证器闭环**：本文参考 PPO/GRPO 范式，但将 verifier 替换为同模型的循环 typed readout，实现零额外 reward model 开销。
6. **温度校准方法**（Post-hoc temperature scaling, Guo et al., 2017）：本文在其基础上扩展至跨分布分组校准，证明 near/far transfer 均受益。

## 局限性与未来方向
1. **循环次数上限受显存/延迟约束**：当前最大 $T=8$，更深层循环需验证是否持续收益。
2. **决策类别限于 A/B/C/D 字母选项**：对自由文本输出或多选项（>4）任务推广性待验证。
3. **训练数据规模较小**（12,800 条）：虽经合成/筛选，但覆盖广度有限，可能在开放域泛化上存在瓶颈。
4. **未与自博弈（self-play）或工具使用结合**：当前验证器为同模型 seed 0，未来可探索多模型互评。
5. **推理加速未系统研究**：Anytime 读取虽已支持，但未给出各 $T$ 下的具体 FLOPs/延迟对比。

## 研究启发与可借鉴点
1. **循环状态迭代 + 冻结骨干 + 低秩 readout** 可作为通用"决策加速"插件，复用于其他类型化输出任务（如排序、匹配）。
2. **Anytime 读取**设计支持动态预算分配，对端侧部署或在线 agent 场景极具价值。
3. **Proper scoring rule 联合训练**（CE + Brier）应成为任何概率输出模型的默认配置，而非事后校准。
4. **Verifier 复用主模型**的策略大幅降低 reward model 训练成本，可迁移至其他 RL fine-tuning 场景。
5. **跨分布温度校准**（分 near/far transfer 组拟合）值得推广至其他需要可靠不确定性感知的任务。

## 关键术语表
**System 1.5 thinking**：介于单次快速前向（System 1）与多步生成推理（System 2）之间的中间范式，通过循环迭代在保持低延迟的同时提升精度。

**Typed decision model**：不生成文本，而是直接输出预声明选项的概率分布的模型形式。

**Proper scoring rule**：鼓励模型输出真实概率的评分函数，如 Brier score；与交叉熵联合使用可同时优化准确性和校准性。

**Anytime reading**：一次前向传播同时产出多个循环次数的决策，允许按需截断以平衡延迟与精度。

**GRPO（Group Relative Policy Optimization）**：强化学习优化算法，对同一 prompt 的多个采样答案计算组内相对优势再更新策略。

**Ouro**：Zhu et al.（2025）提出的预训练为循环的语言模型基座，本文 SanSi 在此基础上微调。

**Loop depth（$T$）**：循环应用的次数，$T=1$ 退化为单次前向，$T=8$ 为本文最大预算。

**Readout（$U_t V_t$）**：每个循环附加的 rank-16 低秩校正矩阵，用于将循环 hidden state 映射到选项 logits。

## 可复现要素
- **数据集**：训练集 12,800 条决策（合成/筛选）；测试集 10,027 条来自 59 个来源；2WikiMultiHopQA（8K/500/3K）用于生成任务。**论文未明确说明是否全部公开**。
- **代码/权重**：**论文未提及开源声明**；基座 Ouro 模型可参考原论文获取。
- **关键超参**：LoRA rank-64；readout rank-16；$T=8$；训练步数 1,000（决策）/ 400（GRPO）；batch size 16；温度 $\tau_t$ 可训练。
