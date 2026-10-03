---
title: "OpenJev-RLCD-A-Working-RLCD-Implementation"
source: https://arxiv.org/pdf/2609.38850v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 03:00:18"
field: "语言模型校准与选择性预测"
keywords: ["calibration", "reinforcement learning", "proper scoring rule", "reasoning model", "selective prediction", "Qwen3"]
innovations: ["提出方差分解恒等式证明混合目标等价于per-rationale目标加Gini多样性奖励项", "设计'先校准后强化'两阶段训练配方避免System-One坍缩和策略梯度淹没", "以真确评分作为强化学习奖励替代1-bit正确性，使单查询决策校准并可自动化"]
benchmarks: ["GSM8K-Verify", "MMLU-Pro", "ChaosNLI"]
---

# 论文速读：OpenJev-RLCD-A-Working-RLCD-Implementation

## 一句话总结
论文提出了一种可直接使用的**强化学习校准决策（RLCD）**训练方法，通过"先校准后强化"的两阶段配方，使推理模型在生成理由后输出的答案分布具备良好校准性；相比GRPO等方法，RLCD在保持相近准确率的同时，能将选择性预测的自动决策比例从19%提升至81%（GSM8K-Verify，≤5%错误率预算）。

## 研究问题与动机
1. **决策模型的核心价值在于校准**：一个只在自己"确信"时自动决策的系统，必须在其声称置信度时是准确的，否则会产生致命错误。
2. **现有开源复现（如Jev）依赖SFT+温度缩放**：SFT训练读头、TS事后校正，但温度缩放只能校准置信度**级别**，无法修复置信度**排序**（ranking）的错误。
3. **标准RL（GRPO/RLVR）使推理模型严重过度自信**：使用正确答案的1-bit奖励（正确=1，错误=0）不是真确评分规则（proper scoring rule），优化结果使模型对错误答案也高度自信，温度缩放也无法修复。
4. **对样本组求混合再评分会奖励无依据的不一致性**：将多个采样的分布混合后评分，会额外获得一个"多样性/不一致性"奖励项，策略可以通过随机化结论来"作弊"获取该奖励，而非提升真实推理能力。

## 核心贡献（创新点）
1. **方差分解恒等式揭示混合目标的本质缺陷**：$J(p,q) = \mathbb{E}_r J(u(r),q) + \text{Var}_r(u)$，证明对混合样本评分等价于"每理由目标 + 多样性奖励"，后一项不要求分歧有事实依据；并证明**RLVR正是混合目标去掉Gini多样性项**。
2. **提出per-rationale目标（λ=0）**：对每个理由输出单独评分，消除多样性奖励项，其唯一最优解为$u(r)=q(x)$，即单次查询在校准意义上达到最优。
3. **发现并诊断两种端到端优化的失败模式**："System-One坍缩"（模型学会在空理由后 hedge）和"策略梯度淹没读头梯度"（score-function项梯度范数是pathwise项的7–12倍），据此提出**两阶段配方**：先纯路径梯度校准读头，再以KL锚定+弱化score-function项强化推理。
4. **在真实任务上提供严格控制的双阶段对比实验**：在GSM8K-Verify和MMLU-Pro上，两阶段RLCD在准确率、Brier分数和AURC三个指标上全面优于或匹配SFT+TS、RFT/STaR+TS、GRPO+TS；在ChaosNLI上证明当不确定性源于标注者分歧（aleatoric）时，per-rationale目标会主动关闭推理，达到与交叉熵相同的质量上限。

## 方法详解
- **读头（Readout）设计**：模型采样理由$r\sim\pi_\theta(\cdot|x)$直至出现"Answer:"标记，之后读取$K$个选项token的logits，经softmax得到答案分布$u_\theta(x,r)\in\Delta_K$（而非从分布中再采样一个答案）。
- **评分规则**：使用严格真确评分规则（Brier score $J(u,y)=2u_y-\|u\|^2$ 或对数得分），在每次观测到真实结果$Y$后对$u(x,r)$评分得到$R=J(u,Y)$。
- **梯度分解（路径积分 + 评分函数）**：
  - **Pathwise项**（读头梯度）：$\frac{1}{M}\sum_i \nabla_\theta R_i$，仅依赖参数的确定性影响，方差低。
  - **Score-function项**（理由梯度）：$\frac{1}{M}\sum_i (R_i-\bar{R}_{-i})\nabla_\theta\log\pi_\theta(r_i|x)$，使用留一法基线（leave-one-out）降低方差。
- **两阶段训练配方（Algorithm 1）**：
  - **Stage 1（校准读头，400步）**：仅优化pathwise项，$\mathcal{L}=-\frac{1}{M}\sum_i R_i$，rationale仅作为on-policy潜变量不参与强化。
  - **Stage 2（强化推理，400步）**：从stage-1检查点出发，加入score-function项并施加KL锚定：$\mathcal{L}=-\frac{1}{M}\sum_i R_i - c\sum_i\text{sg}(A_i)\log\pi_\theta(r_i|x)$，其中$c=0.3$，$\beta=0.04$，KL参考分布为stage-1模型。
- **方差恒等式的教学意义**：用one-hot答案特例可直观看出，混合目标等于RLVR（期望奖励）+ Gini不纯度（多样性惩罚/奖励），当多样性权重系数$\lambda<1$时最优解偏离$q(x)$，$\lambda=0$时退化为RLVR（mode）。

## 实验与结果
- **数据集**：MMLU-Pro（10选项知识密集型，1500测试）、GSM8K-Verify（数学验证，从GSM8K构建的是/否判断）、ChaosNLI（标注者分歧导致aleatoric不确定性）。
- **模型**：Qwen3-1.7B（non-thinking模式），AdamW lr=$2\times10^{-6}$，8题/步×M=4理由/题。
- **关键数字（Table 1，温度缩放后单查询）**：
  - **GSM8K-Verify**：两阶段RLCD（log-score）Acc=**92.2±0.7%**，Brier=**0.135±0.009**，AURC=**0.034±0.001**，Cov@5%=**81.1±7.2%**，T=1.4；GRPO+KL+TS Acc=91.8±0.6%，Brier=0.150，AURC=0.061，Cov@5%=**19.4±7.3%**，T≈7。
  - **MMLU-Pro**：两阶段RLCD Acc=**47.7±2.0%**，Brier=**0.658±0.015**，AURC=**0.304±0.022**，Cov@20%=**27.6±3.3%**；SFT+TS Acc=42.3%，Cov@20%=15.6%，任何推理基线Cov@20%=0。
- **最强结果**：RLCD在GSM8K-Verify的Cov@5%（81.1%）上以极大优势击败所有基线（GRPO仅19.4%），且p<0.001；准确率与GRPO相当但校准显著更优（Brier低0.015，AURC低0.027）。
- **消融要点（Table 2）**：从同一stage-1检查点分叉，RLCD-RL比纯读头续训在GSM8K-Verify上额外+3.5 Acc和−0.057 Brier；GRPO的正确性奖励在MMLU-Pro上反而恶化Brier（+0.026）和AURC（+0.033）。

## 相关工作脉络
1. **Temperature Scaling（Guo et al., ICML 2017）与置信度 elicitation**：仅校正置信度**级别**，不改变置信度**排序**；本文指出选择性预测（Risk-Coverage）才是体现差异的关键评测维度。
2. **RLVR / GRPO（Shao et al., 2024）**：优化正确采样答案的概率，本质是混合目标去掉Gini多样性项；本文证明这导致过度自信，且正确性奖励会破坏已校准的读头。
3. **LaTRO / TRICE（Chen et al., 2024; Phan et al., 2023）**：将理由视为潜变量、最大化正确回答似然；区别在于本文对**完整分布**使用真确评分而非仅似然，使报告的置信度可直接使用。
4. **REINFORCE with leave-one-out baseline（Kool et al., ICLR 2019；Ahmadian et al., ACL 2024）**：本文沿用的无偏梯度估计器，并在表桌模型上做精确枚举验证（误差<10⁻¹⁵）。
5. **ChaosNLI（Nie et al., EMNLP 2020）**：标注者分歧基准；用于证明当不确定性源于aleatoric时，推理无法改善校准，per-rationale目标会主动关闭推理——这是对RL适用边界的理论+实证刻画。
6. **Selective prediction（Geifman & El-Yaniv, NeurIPS 2017）**：本文评测核心指标AURC和Cov@ϵ的来源，强调校准模型的下游价值在于自动化决策比例的提升。

## 局限性与未来方向
1. **实验规模有限**：仅使用1.7B模型、400–800步更新、3个随机种子；更大模型和更长训练的stage-1/stage-2权衡未知。
2. **与Jev的内部算法无直接对比**：仅行为级复现了Jev的API；训练数据未公开，无法断言本文方法与Jev内部实现的异同。
3. **Stage 2的收益难以预测**：仅在"推理是关键计算"的任务（GSM8K-Verify）上有增益，在知识密集型任务上近似中性；缺乏先验判断何时值得stage 2的理论准则。
4. **任务构造局限**：GSM8K-Verify由Qwen3-0.6B生成提议答案（53%正确率），非标准评测设置；ChaosNLI仅1个种子报告。
5. **未扩展至更多真确评分或任务域**：目前仅验证Brier和对数得分，其他严格真确评分（如Spherical）的效果未知。

## 研究启发与可借鉴点
1. **"方差分解恒等式"是分析多采样奖励的统一工具**：任何对混合分布评分的RL目标都可拆分为"每样本期望 + 多样性项"，多样性项的系数决定了最优解是否仍是$q(x)$。可用于分析reward shaping、collision penalty、deduplication等技巧。
2. **两阶段"先校准后强化"的分离训练思想具有通用性**：当策略梯度和价值/读头梯度量级差异大（7–12倍）时，联合优化容易灾难；分阶段并用KL锚定是稳定训练的通用范式。
3. **真确评分作为RL reward替代1-bit正确性**：不仅使单次查询校准，且能区分"模型多不确定"——可用于构建更丰富的reward signal（如连续reward），替代硬奖惩。
4. **Aleatoric不确定性下的"自动关推理"行为**：per-rationale目标在标注者分歧数据上主动退化读头策略（关闭推理），可作为模型是否具有"自知之明"的行为探针，也可启发未来"自适应推理长度"的研究。
5. **Risk-Coverage（AURC/Cov@ϵ）应成为校准模型的标配评测**：相比单纯accuracy/Brier，该指标更贴近下游自动化决策的真实收益，建议后续工作沿用。

## 关键术语表
- **RLCD（Reinforcement Learning for Calibrated Decisions）**：本文提出的强化学习训练框架，使推理模型在生成理由后输出的答案分布具备统计校准性。
- **Proper Scoring Rule（真确评分规则）**：期望上仅在模型报告真实分布时取得最大值的评分函数（如Brier、对数得分），是校准的理论基础。
- **Mixture Objective（混合目标）**：先对多个采样理由的答案分布求平均$p=\mathbb{E}_r u(r)$，再对$p$评分；等价于per-rationale目标加Gini多样性奖励项。
- **Per-rationale Objective（每理由目标，λ=0）**：对每个理由单独评分并取期望，不含多样性奖励项，唯一最优解为$u(r)=q(x)$。
- **System-One Collapse（系统一坍缩）**：端到端优化per-rationale目标时，模型学会放弃长推理，退化为对所有问题输出相同的空理由+随机hedge的直接分类器。
- **Risk-Coverage / AURC / Cov@ϵ**：按置信度排序计算累积错误率的曲线；Cov@ϵ表示在错误率≤ϵ约束下最多能自动决策多少比例样本，衡量校准的实用性。
- **Leave-one-out Baseline**：REINFORCE估计时用除当前样本外其他样本的均值作为基线，降低方差且保持无偏。
- **Aleatoric Uncertainty（随机/本体不确定性）**：源于数据本身噪声（如标注者分歧）的不确定性，无法通过收集更多信息或更强推理消除；本文用它划定RL的适用边界。

## 可复现要素
- **数据集**：MMLU-Pro（公开）、GSM8K（公开，本文构建GSM8K-Verify）、ChaosNLI（公开）。
- **代码开源**：https://github.com/ZimmyGao/openjev-rlcd，含所有run logs及API兼容server。
- **关键超参**：Qwen3-1.7B non-thinking；AdamW lr=$2\times10^{-6}$，无weight decay，gradient clipping=1.0；batch=8题×M=4理由；rationale max length：MMLU-Pro≤256，GSM8K-Verify≤320；Stage 1：400步路径梯度；Stage 2：400步，score-function权重$c=0.3$，KL锚定$\beta=0.04$；温度缩放T在dev集grid search（$T\in[0.1,50]$按NLL选择）。
