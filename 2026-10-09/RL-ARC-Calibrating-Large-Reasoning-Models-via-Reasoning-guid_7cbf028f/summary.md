---
title: "RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid"
source: https://arxiv.org/pdf/2610.11352v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:19:58"
field: "大语言模型可靠性与校准"
keywords: ["校准感知训练", "大推理模型", "过度自信缓解", "推理置信度", "强化学习"]
innovations: ["提出推理引导的条件化正则化/惩罚机制联合校准答案置信度", "在保持准确率的同时显著改善 ID/OOD 校准性", "引入推理置信度辅助评估推理过程可靠性"]
benchmarks: ["Big-Math", "GSM8K", "MATH-500", "AMC23", "AIME24", "AIME25", "StrategyQA", "GPQA", "HotpotQA", "SimpleQA", "NQ-Open", "TriviaQA"]
---

# 论文速读：RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty

## 一句话总结
提出 RL-ARC，一种校准感知的强化学习训练框架，通过联合利用推理置信度（reasoning confidence）和答案置信度（answer confidence），在数学推理和分布外（OOD）任务上同时提升模型校准性并缓解准确率-校准性权衡。

## 研究问题与动机
1. **RLVR 导致严重校准退化**：现有大推理模型（LRM）主要通过 RLVR 训练，但 RLVR 仅优化答案正确性，不显式考虑置信度校准，导致模型在 OOD 设置下产生过度自信（overconfidence）。
2. **现有校准方法仍存在局限**：已有校准感知训练方法虽能改善校准，但在分布偏移下仍倾向于对错误答案分配高置信度，且往往以牺牲推理性能为代价（准确率-校准性权衡）。
3. **缺乏对推理过程可靠性的考量**：现有方法仅校准答案置信度，未考虑推理过程本身的可靠性，导致模型可能生成不可靠推理却获得高置信度预测。
4. **分布外泛化能力不足**：事后校准方法在 ID 数据上表现良好，但迁移到 OOD 任务时效果有限，难以适应不同任务类型的校准需求。

## 核心贡献（创新点）
1. **提出 RL-ARC 框架**：首次将推理置信度作为辅助信号引入校准感知 RL 训练，通过条件化目标函数对正确/错误预测采用不同策略。
2. **推理引导的正则化机制**：对正确预测，使用推理置信度对齐答案置信度（最小化两者差异），防止模型仅凭正确答案分配高置信度。
3. **过度自信惩罚机制**：对错误预测，以推理置信度平方作为额外惩罚项，迫使模型在推理和答案均不可靠时降低置信度。
4. **缓解准确率-校准性权衡**：在 Qwen2.5-7B 和 Qwen3-8B 上验证，RL-ARC 在保持与 RLVR 相当准确率的同时显著提升校准指标（ECE、Brier、AUROC）。
5. **跨架构与训练分布的泛化性**：在 Llama-8B（DeepSeek-R1 蒸馏版）和不同训练数据集（Big-Math、HotpotQA）上验证方法通用性。

## 方法详解
**整体框架**：
- 模型 $\pi_\theta$ 在给定问题 $x$ 时生成推理过程、推理置信度 $c_r$、答案 $y$ 和答案置信度 $c_a$。
- 使用 GRPO 作为基础强化学习算法，基于 Big-Math 训练数据集。

**置信度提取**：
- 通过 calibration-aware reasoning training prompt 提示模型同时输出推理置信度和答案置信度（verbalized confidence）。
- 推理置信度 $c_r$：模型对自身推理过程正确性的自我评估。
- 答案置信度 $c_a$：模型对自身预测正确性的自我评估。

**校准感知优化目标**：
$$\max_\theta \mathbb{E}_{(x, y^*) \sim \mathcal{D}, (y, c_a, c_r) \sim \pi_\theta(\cdot|x)} [R_{\text{total}}(z, c_a, c_r)]$$

其中奖励函数为：
$$R_{\text{total}}(z, c_a, c_r) = R(z, c_a) - \Omega_{\text{arc}}(z, c_a, c_r)$$
$$R(z, c_a) = z - (z - c_a)^2, \quad z = \mathbf{1}_{y = y^*}$$

**关键正则化项 $\Omega_{\text{arc}}$**：
$$\Omega_{\text{arc}}(z, c_a, c_r) = \begin{cases} \lambda_{\text{pos}}(c_a - c_r)^2, & z = 1 \quad (\text{正确预测：推理引导正则化}) \\ \lambda_{\text{neg}} c_r^2, & z = 0 \quad (\text{错误预测：过度自信惩罚}) \end{cases}$$

- $\lambda_{\text{pos}}$ 和 $\lambda_{\text{neg}}$ 为超参数，默认值分别为 0.4 和 0.15。
- 正确时鼓励 $c_a \approx c_r$，确保高置信度仅当推理过程也正确时分配。
- 错误时用 $c_r$ 平方惩罚，迫使模型降低对错误答案的置信度。

## 实验与结果
**数据集**：
- 训练：Big-Math（30,000 样本）、HotpotQA（替代训练分布）
- ID 评估：Big-Math、GSM8K、MATH-500、AMC23、AIME24、AIME25
- OOD 评估：StrategyQA、GPQA、HotpotQA、SimpleQA、NQ-Open、TriviaQA

**基线方法**：
- Base（预训练模型直接提取置信度）
- RLVR / RLVR w/ Confidence / RLVR w/ Probability / RLVR w/ Post-hoc
- Behavioral Calibration、RLCR（校准感知 RL 代表方法）

**主要结果（Qwen2.5-7B）**：

| 设置 | 方法 | Acc. (↑) | AUROC (↑) | Brier (↓) | ECE (↓) |
|------|------|----------|-----------|-----------|---------|
| ID Avg. | RLVR | 56.86 | 0.45 | 0.41 | 0.40 |
| ID Avg. | RLCR | 54.82 | 0.58 | 0.25 | 0.22 |
| ID Avg. | **RL-ARC** | **56.53** | **0.60** | **0.23** | **0.18** |
| OOD Avg. | RLVR | 47.58 | 0.50 | 0.50 | 0.50 |
| OOD Avg. | RLCR | 46.02 | 0.54 | 0.28 | 0.20 |
| OOD Avg. | **RL-ARC** | **47.27** | **0.54** | **0.26** | **0.16** |

**关键发现**：
1. **校准显著提升**：RL-ARC 在 OOD 上 ECE 降低 20%（vs RLCR 的 0.20→0.16），Brier 降低 7%。
2. **准确率保持**：ID 上仅比 RLVR 低 0.33%，OOD 上保持与 RLVR 相当水平。
3. **跨模型泛化**：Qwen3-8B 和 Llama-8B（DeepSeek-R1）上均取得类似效果。
4. **不同训练分布**：以 HotpotQA 训练后，在数学推理和 factual QA 任务上均优于 RLCR。
5. **过度自信缓解**：Risk-coverage 曲线显示 RL-ARC 在低覆盖率区域风险显著更低。
6. **推理可靠性提升**：Reasoning Calibration 和 Shallow Reasoning 指标显示推理过程更可靠。

## 相关工作脉络
1. **RLCR (Damani et al., 2026)**：代表性校准感知 RL 方法，仅利用答案置信度，未考虑推理过程可靠性；RL-ARC 在其基础上引入推理置信度作为辅助信号。
2. **Behavioral Calibration (Wu et al., 2026)**：聚合推理过程中的 verbalized confidence 进行校准；RL-ARC 直接利用推理置信度进行条件化正则化/惩罚。
3. **Post-hoc 校准方法**：如 Li et al. (2025)，训练外部置信度估计器；RL-ARC 指出此类方法在 OOD 泛化上受限，主张端到端训练。
4. **概率-based 不确定性估计**：如 Kuhn et al. (2023) 的 semantic uncertainty；RL-ARC 采用 verbalized confidence 更简洁且与推理过程解耦。
5. **Verbalized confidence 研究**：如 Tian et al. (2023)、Xiong et al. (2024)；本文聚焦训练时如何利用这些置信度而非仅评估其质量。

## 局限性与未来方向
1. **实验范围有限**：仅使用两种训练分布（数学推理、复杂推理）和单一 RL 算法（GRPO）；未来可扩展更多数据集和更强 RL 算法。
2. **仅限语言模型**：主要聚焦 Qwen 系列；未来可延伸至多模态推理模型（如 MM-Boundary 相关方向）。
3. **推理正确性评估依赖 LLM-as-a-judge**：可能引入误差；需开发更准确的推理正确性评估方法。
4. **超参数敏感性**：$\lambda_{\text{pos}}$ 和 $\lambda_{\text{neg}}$ 需调优；研究表明正主导设置（$\lambda_{\text{pos}}/\lambda_{\text{neg}} > 1$）效果最佳，但未系统探索边界情况。

## 研究启发与可借鉴点
1. **条件化正则化策略**：对正确/错误样本采用不同辅助目标的设计思路可迁移至其他校准任务（如指令遵循、代码生成）。
2. **推理-答案双置信度解耦**：将推理过程置信度与答案置信度分离建模，有助于诊断模型"碰巧答对"（high answer conf, low reasoning conf）的情况。
3. **超参数正主导偏好**：实验发现 $\lambda_{\text{pos}} > \lambda_{\text{neg}}$ 更优，提示"鼓励正确推理的高置信度"比"惩罚错误推理的高置信度"更重要。
4. **Shallow Reasoning 指标**：可复用为评估模型是否"靠运气答对"的简单有效指标。
5. **与团队方向结合机会**：若团队研究安全/可信 AI，可将 RL-ARC 思路应用于高风险领域（如医疗问答、法律推理）的校准训练。

## 关键术语表
**RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习训练范式，仅以答案正确性作为奖励信号。
**Calibration**：校准性，模型置信度估计与其实际正确率的一致性程度；完美校准时，置信度 c 的预测应有约 c 比例正确。
**Overconfidence**：过度自信，模型对错误预测分配过高置信度的现象，在 OOD 设置下尤为严重。
**Reasoning Confidence ($c_r$)**：推理置信度，模型对自身推理过程正确性的自我评估。
**Answer Confidence ($c_a$)**：答案置信度，模型对自身最终预测正确性的自我评估。
**Brier Score**：布里尔分数，衡量预测概率与真实标签之间均方误差的校准评估指标，越低越好。
**ECE**：Expected Calibration Error，期望校准误差，按置信度分箱计算准确率与平均置信度的加权偏差。
**AURC**：Area Under the Risk-Coverage Curve，风险-覆盖曲线下的面积，衡量选择性预测性能。
**Shallow Reasoning (SR)**：浅层推理，模型给出正确答案但推理过程错误的比例，反映"碰巧答对"现象。

## 可复现要素
- **数据集**：Big-Math（训练）、GSM8K、MATH-500、AMC23、AIME24、AIME25（ID 评估）；StrategyQA、GPQA、HotpotQA、SimpleQA、NQ-Open、TriviaQA（OOD 评估）。部分数据集需自行获取。
- **代码**：论文未提供开源代码链接。
- **模型权重**：未公开；基于 Qwen2.5-7B、Qwen3-8B、Llama-8B（DeepSeek-R1 蒸馏版）实验。
- **关键超参**：$\lambda_{\text{pos}} = 0.4$、$\lambda_{\text{neg}} = 0.15$；rollout group size=8、temperature=0.7、batch size=1536、learning rate=5e-6。
