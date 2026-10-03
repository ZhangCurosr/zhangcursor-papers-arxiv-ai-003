---
title: "XU-RS-Explaining-Credal-Width-in-Random-Set-Language-Models"
source: https://arxiv.org/pdf/2609.37594v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:58:38"
field: "可解释AI与不确定性量化"
keywords: ["credal width", "uncertainty attribution", "random-set classifiers", "expected gradients", "explainable AI", "belief functions"]
innovations: ["提出XU-RS框架将credal width归因到输入tokens", "推导概率修正对width梯度的影响机制", "建立参考准备检查与数值精度检查的双重验证协议"]
benchmarks: ["MedQA US"]
---

# 论文速读：XU-RS-Explaining-Credal-Width-in-Random-Set-Language-Models

## 一句话总结
本文提出 XU-RS 框架，利用 Expected Gradients 方法将随机集合语言模型中选定答案的 credal width（信度宽度）归因到输入 tokens 上，并通过零掩码（zero-masking）实验验证了高排名 token 对 width 的影响显著大于随机 token。

## 研究问题与动机
- **不确定性量化缺乏可解释性**：现有不确定性估计仅给出模型的置信程度，但无法说明哪些输入特征导致了该不确定性，难以判断不确定性是否来自与任务相关的特征。
- **随机集合分类器的 credal width 特性**：随机集合分类器为单个答案和答案组分别赋值概率，产生下概率和上概率，二者之差即为 credal width，代表因训练数据有限导致的 epistemic uncertainty，但缺乏对该 width 的输入归因方法。
- **归因基线的选择问题**：Expected Gradients 需要参考输入（reference/baseline），参考 prompt 的准备方式会直接影响所解释的 width 差异，现有方法未系统研究这一问题。
- **数值精度与归因忠实度**：即使 token 排名稳定，归因值的数值精度可能不足，且概率修正（correction）过程会引入额外依赖关系，需要专门的检查机制。

## 核心贡献（创新点）
- **提出 XU-RS 归因框架**：首次将 Expected Gradients 应用于随机集合语言模型的 credal width 归因，将输出层的 width 变化分解到输入 token 级别，区别于此前针对预测分数或熵的归因工作。
- **推导概率修正对 width 归因的影响**：从理论上证明当 surviving total s > 1 时，排除选定答案的集合赋值会通过归一化分母影响该答案的 width，给出完整的梯度公式（Equation 7），揭示了归因中跨集合依赖的本质。
- **建立双重验证协议**：提出参考准备检查（reference preparation check）和数值精度检查（numerical accuracy check），前者识别参考 prompt 准备导致 width 差异偏移，后者通过 completeness residual 和向量一致性评估归因计算的可靠性。
- **在 MedQA 上验证零掩码优势**：在 SmolLM3-3B 和 Llama-2-7B 上实验表明，XU-RS 高排名 token 的 zero-masking 产生的 width 变化显著大于随机 token（平均优势 0.0381 和 0.0450），95% 置信区间均不含零。

## 方法详解
- **随机集合预测与 credal width**：分类器对 4 个答案选项的 14 个子集（单元素、二元组、三元组）分配 mass 值 m(S)，通过 Bel(E) = Σ_{S⊆E} m(S) 和 Pl(E) = Σ_{S∩E≠∅} m(S) 计算下概率和上概率，选定答案 c 的 credal width 为 W_c = Pl({c}) - Bel({c}) = Σ_{S∋c, |S|>1} m(S)。
- **分类器架构**：基于预训练语言模型（SmolLM3-3B / Llama-2-7B）+ LoRA 适配，输出层为 14 个 sigmoid 单元，估计 14 个子集的下概率；通过递归 Möbius 反演 recovery mass 值：m̃_{ {c} } = b_{ {c} }, m̃_S = b_S - Σ_{T⊊S} max(m̃_T, 0)。
- **概率修正（mass correction）**：由于 sigmoid 输出不保证概率关系，需进行修正：q_S = max(0, m̃_S)，s = Σ q_S，r = max(1-s, 0)，最终 m(S) = q_S/(s+r)，m(Θ) = r/(s+r)；当 s > 1 时，分母 s 共享导致其他集合的赋值影响选定答案的 width。
- **Expected Gradients 归因**：构造参考 prompt（来自训练集的问答对），将原始输入 X 和参考嵌入 B 之间线性插值：Z_α = B + α(X-B)，计算 EG = E_{B,α}[(X-B) ⊙ ∇_Z F_c(Z)|_{Z=B+α(X-B)}]，其中 F_c 为选定答案的 width。
- **Token 评分**：对每个 token 的嵌入坐标贡献 e_ij 计算非负范数 a_i = (Σ_j e_ij²)^(1/2) 作为排名依据；完整性检查使用带符号求和 Σ_{i,j} e_ij 与 width 差异比较。
- **零掩码评估**：将排名最高 token 的嵌入置零（R_i X），测量 |F_c(R_i X) - F_c(X)|，与 20 次随机 token zero-masking 的平均变化比较，定义优势 A(x) = D_{i*}(x) - (1/20)Σ_b D_{j_b}(x)。

## 实验与结果
- **数据集与模型**：MedQA US（四选项医学多选题），测试集 1273 题；使用 SmolLM3-3B（准确率 49.3%）和 Llama-2-7B（准确率 45.7%），均通过 LoRA 微调。
- **零掩码结果**：XU-RS 高排名 token 的 zero-masking 平均优势为 SmolLM3 0.0381 [0.0243, 0.0532]、Llama 0.0450 [0.0295, 0.0617]，95% 置信区间均不含零；width 错误检测 AUROC 为 0.634 和 0.598，低于低概率排名的 0.680 和 0.671。
- **概率修正频率**：95.84%-99.92% 测试题的 surviving total s > 1，width 通常处于 u_c/s  regime，所有预测均存在负中间值，平均总修正量 1.23-2.81。
- **数值精度检查**：合成数据集上，增加采样数后 token 排名高度一致（Spearman ρ = 0.9951），但 completeness residual 中位数达 0.0652，超过 0.01 容忍度；使用 FP32 Integrated Gradients 可显著降低残差（最大 1.10×10⁻⁵）。
- **参考准备影响**：对齐配对参考 prompt 导致 width 差异偏移 0.203-0.537，归因值准确解释了 prepared reference 的差异，而非原始 natural 差异。

## 相关工作脉络
- **Random-set classifiers（RS-NN/RS-LLM）**：Manchingal & Cuzzolin (2022, 2025d) 提出的随机集合神经网络框架，本文继承其 belief-output 架构但扩展至 LLM 分类任务，聚焦于 width 归因而非新不确定性表示。
- **Credal deep ensembles / Credal wrapper**：Wang et al. (2024, 2025b) 通过集成或 Bayesian NN 构建 credal sets，本文采用直接预测 set mass 的方式，无需后处理。
- **Uncertainty attribution**：Perez et al. (2022) 结合不确定输入比较和梯度归因；Watson et al. (2023) 使用 Shapley values 分配条件熵变化；Iversen et al. (2025) 将归因方法应用于预测方差；XU-RS 专门针对 credal width 设计。
- **Expected Gradients / Integrated Gradients**：Erion et al. (2021) 提出 EG 通过采样参考和中间点改进 IG；Sanyal & Ren (2021) 的 Discretized IG 保持 token 离散性；本文使用 Captum 的 GradientShap 实现。
- **Attribution evaluation**：Adebayo et al. (2018) 的 sanity checks 指出解释需响应参数变化；Hase et al. (2021) 警告 feature-replacement 可能引入 OOD 输入；本文的 completeness residual 和 reference check 是对此的延伸。

## 局限性与未来方向
- **评估规模有限**：仅在 100 题 MedQA 子集上验证零掩码，未覆盖全测试集；two models 的结论需更大规模验证。
- **Zero-masking 的操作限制**：将嵌入置零而非删除或改写文本，改变的是数值表示而非语义，与真实干预存在差距。
- **合成数据集的 shortcut 问题**：用于数值检查的合成数据存在答案顺序 shortcut 和非对称目标，限制了归因忠实度的评估。
- **未验证 epistemic uncertainty 语义**：width 是否真正代表 epistemic uncertainty 需独立验证，当前仅证明 token 影响 width 的敏感性。
- **未来方向**：使用新问题和受控证据编辑评估；比较 width 归因与熵/答案分数归因的额外价值；结合领域专家进行临床解释。

## 研究启发与可借鉴点
- **归因前的完整性检查**：completeness residual 和 reference preparation check 可作为任何归因方法的标配验证步骤，确保归因值忠实反映目标函数差异。
- **概率修正的梯度传播**：当模型输出需经过非线性修正（如归一化、截断）时，修正操作必须纳入归因计算，否则归因会遗漏重要依赖路径。
- **排名与数值的分离评估**：token 排名稳定不代表归因值精确，应同时报告 Spearman correlation 和向量级指标（cosine similarity、relative Euclidean distance）。
- **参考 prompt 的对齐处理**：当参考与原始输入 token 数量不同时， alignment 会改变 width 差异，需在归因前评估 prepared reference 的 width 以识别偏移。
- **与团队方向结合机会**：XU-RS 的归因框架可迁移至其他基于 belief function 的不确定性模型；reference preparation check 的思路适用于任何需要 baseline 的归因方法。

## 关键术语表
- **Credal width（信度宽度）**：选定答案的上概率与下概率之差，W_c = Pl({c}) - Bel({c})，衡量因 mass 分配于包含 c 的多元素集合而产生的不确定性。
- **Random-set classifier（随机集合分类器）**：基于 Dempster-Shafer 理论，将概率质量分配给答案子集而非单元素的分类器，保留概率未分配信息以表征 epistemic uncertainty。
- **Expected Gradients（期望梯度）**：Extended Integrated Gradients，通过对参考输入和中间点的采样平均估计特征贡献，比 IG 更稳健。
- **Zero-masking（零掩码）**：将选定 token 的嵌入向量置零的操作，用于评估该 token 对模型输出的重要性。
- **Completeness residual（完整性残差）**：归因贡献的带符号总和与目标函数差异之间的绝对差距，用于检验归因计算的数值精度。
- **Reference prompt（参考 prompt）**：用于归因计算的对比输入，定义了解释所针对的 width 差异基准。
- **Mass correction（质量修正）**：将神经网络输出的下概率转换为有效 mass 分布的后处理步骤，包括截断负值、填充 shortfall 和归一化。
- **BetP（pignistic probability）**：将 set mass 均匀分配到成员的答案概率，用于选择最终答案但不承担 width 计算。

## 可复现要素
- **数据集**：MedQA US（Jin et al., 2021），训练集 10,178 题，开发集 1,272 题，测试集 1,273 题；论文声明数据公开可用。
- **代码/权重**：论文未明确声明开源，但提到使用 HuggingFaceTB/SmolLM3-3B 和 meta-llama/Llama-2-7b-hf 模型；附录包含详细实现细节。
- **关键超参**：LoRA rank 16（SmolLM3）/ 8（Llama），LoRA scaling 32/16，dropout 0.05，learning rate 10⁻⁴；EG 采样 512（SmolLM3）/ 1024（Llama）；参考 prompt 数 256/1024。
