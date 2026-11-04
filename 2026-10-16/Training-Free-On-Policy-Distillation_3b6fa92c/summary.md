---
title: "Training-Free-On-Policy-Distillation"
source: https://arxiv.org/pdf/2610.22236v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:33:59"
field: "大语言模型合并与蒸馏"
keywords: ["model merging", "on-policy distillation", "training-free", "reinforcement learning", "task arithmetic", "closed-form solution"]
innovations: ["将逐层蒸馏转化为对 on-policy 输入二阶矩的 RegMean 闭式解，无需梯度更新", "发现 on-policy rollout 的组成（去重循环答案）决定融合模型在难题上的表现", "揭示 RL 专家合并中数值精度（bfloat16 vs float16）是关键被忽视因素"]
benchmarks: ["AIME24", "AIME25", "HMMT25", "MATH-500", "OlympiadBench", "HumanEval+", "MBPP+"]
---

# 论文速读：Training-Free-On-Policy-Distillation

## 一句话总结
本文提出 TF-OPD（Training-Free On-Policy Distillation），一种无需梯度更新的专家模型融合方法：仅需前向传播即可将多个 RL 训练得到的领域专家聚合为一个通用模型，在 Olmo 3 7B 上实现平均性能超越所有单一专家。

## 研究问题与动机
1. **从 RL 专家到通用模型的合并困境**：RLVR（可验证奖励强化学习）催生了大量领域专家（数学、代码、指令遵循等），每个专家在各自领域极强但跨领域退化，用户需要将它们合并为一个全能模型。
2. **现有合并路径的不足**：多域 RL 需要重新训练；on-policy distillation 需要大量 rollout、教师前向传播和梯度更新；权重算术（Task Arithmetic、TIES、DARE）不感知输入分布，会损害专家能力；RegMean 虽有闭式解但使用固定训练语料而非模型自身生成的文本。
3. **核心洞察**：当专家共享同一基础模型时，逐层最小二乘解是蒸馏目标的精确解，所需统计量只需收集专家对自己生成答案的输入二阶矩即可。
4. **关键挑战**：长推理答案中存在重复循环（repetition loops）的失败样本，虽然占比小但 token 数极大，会严重扭曲统计量并损害融合效果。

## 核心贡献（创新点）
1. **证明训练并非必需**：首次展示共享基础模型的专家可通过纯前向计算聚合，无需任何梯度步骤，即可获得平均更优的通用模型。
2. **将逐层蒸馏转化为 RegMean 闭式解**：把蒸馏目标限定为单层线性层的损失，导出与 RegMean 相同的最小二乘形式，但统计量来自专家自身的 on-policy 生成文本而非静态训练语料。
3. **揭示 rollout 组成决定难题表现**：证明去重循环答案 alone 即可将融合模型从 Task Arithmetic 水平提升至专家水平，而 token 级拟合度并非关键因素。
4. **发现数值精度是关键被忽视因素**：RL 更新的步长与 bfloat16 量化步长相当，融合更新经量化后产生可测量失真，需使用 float16/float32 存储和评估。

## 方法详解
**问题设定**：基础模型 $\theta_b$，K 个专家 $\theta_d = \theta_b + \Delta_d$，目标构造 $\theta_\star$。

**逐层目标（公式 1）**：
$$W_\star = \arg\min_W \sum_{d=1}^K w_d \mathbb{E}_{x \sim \mu_d}\|Wx - W_dx\|_2^2$$
其中 $\mu_d$ 是专家 d 处理自身答案时的输入分布。

**闭式解（公式 2）**：
$$W_\star = W_b + \left(\sum_d \Delta_d \widetilde{G}_d\right)\left(\sum_d \widetilde{G}_d\right)^{-1}$$
其中 $\widetilde{G}_d = w_d[\alpha G_d + (1-\alpha)\text{diag}(G_d)]$ 为收缩后的二阶矩矩阵（$\alpha=0.9$）。

**On-policy 统计量采集**：
- 专家回答其领域内几百道训练题（数学：DAPO-Math-17k 的 256 题；代码：DeepCoder 训练集的 256 题）。
- **过滤**：剔除截断答案和重复循环答案（DEFLATE 压缩率 < 5%），保留正确、完整、无循环的答案。
- 专家以 teacher forcing 方式重读保留的答案，通过 forward pre-hook 累积每层输入 Gram 矩阵 $G_d = X^\top X / n$。
- 线性层按架构分组（QKV、attention output、gate/up、down），每组共享同一统计量。

**求解**：
- 每层用 double precision 求解一次线性方程组，非线性层（embedding、LN 参数、bias）用 Task Arithmetic 合并。
- 融合模型以 float32 存储、float16 评估。

**误差诊断（公式 4）**：
$$\varepsilon_d = \frac{\text{tr}[(\Delta_\star - \Delta_d)\widetilde{G}_d(\Delta_\star - \Delta_d)^\top]}{\text{tr}[\Delta_d \widetilde{G}_d \Delta_d^\top]}$$
衡量融合层未能复现教师更新的份额，定位冲突层。

## 实验与结果
**模型**：Olmo 3 7B 系列，Math 专家和 Code 专家各一个。

**主要结果（Table 2）**：

| 模型 | MATH-500 | OlympiadBench | AIME24 | AIME25 | HMMT25 | HumanEval+ | MBPP+ | Mean |
|------|----------|---------------|--------|--------|--------|-----------|-------|------|
| Base | 63.4 | 33.4 | 15.8 | 10.0 | 8.3 | 45.7 | 44.2 | 31.6 |
| Math Specialist | 91.8 | 68.7 | 50.0 | 43.3 | 30.8 | 71.3 | 62.4 | 59.8 |
| Code Specialist | 85.6 | 62.5 | 26.7 | 30.0 | 15.0 | 86.0 | 69.3 | 53.6 |
| **TF-OPD** | 90.8 | 69.4 | 50.0 | 38.3 | 27.5 | 84.1 | 67.5 | **61.1** |

**关键结论**：
- TF-OPD 平均准确率 **61.1%**，高于两个专家（59.8% 和 53.6%），是唯一从未显著低于各领域专家的模型。
- 在 AIME25 上，融合模型（float16 评估）为 40.0%，与数学专家（float16）的 39.2% 基本持平，远超 Task Arithmetic 的 34.2%。
- 每个专家只需约 **few GPU-hours** 的前向计算（生成 rollout + teacher forcing + 一次线性求解），无反向传播。

## 相关工作脉络
1. **On-policy Distillation**（Ross et al., 2011; Agarwal et al., 2024; Gu et al., 2024; Hou et al., 2026）：训练型在线策略蒸馏，需迭代优化；TF-OPD 保留"匹配教师输出"和"模型生成文本"两个要素，去除优化环节。
2. **Model Merging / Task Arithmetic**（Ilharco et al., 2023; Wortsman et al., 2022; Yadav et al., 2023; Yu et al., 2024）：纯权重操作，不感知输入分布，在 RL 专家上显著丢失能力。
3. **RegMean**（Jin et al., 2023）：同形式闭式解，但统计量来自固定训练语料；TF-OPD 使用 on-policy 生成文本，针对性适配推理模型。
4. **Fisher Merging / 谱方法**（Matena & Rafel, 2022; Gargiulo et al., 2025; Marczak et al., 2025; Sun et al., 2026）：引入参数重要性或奇异值分解；TF-OPD 完全不涉及这些复杂计算。
5. **多域 RL / 多任务微调**（Liu et al., 2025; Liang et al., 2025, 2026）：需重新训练或处理 sample 间交互；TF-OPD 提供零训练的替代路径。

## 局限性与未来方向
1. **架构一致性假设**：方法要求所有专家共享基础模型架构，无法直接处理不同架构或尺寸的模型。
2. **数值精度依赖**：融合模型必须用 float16/float32 存储评估，bfloat16 会引入可测量失真，增加了部署成本。
3. **领域权重敏感**：均匀权重下数学专家贡献略被稀释，重新加权可部分恢复但会损害另一领域。
4. **规模未知**：目前仅在 7B 模型上验证，更大规模下是否仍有相同效果尚待检验。
5. **rollout 数量固定**：256 题的经验效果良好，但 optimal sample count 未作系统研究。

## 研究启发与可借鉴点
1. **闭式解替代训练的想法极具启发性**：将蒸馏目标简化为单层最小二乘并导出闭式解，可将计算成本降低数个数量级，值得在其他模型合并场景中探索类似思路。
2. **Rollout 组成质量 > 数量**：去重循环答案 alone 就关闭了性能差距，提示后续工作应更重视数据筛选而非单纯扩大样本量；"重复循环检测 + 压缩率过滤" 可作为通用质量标准。
3. **评估时精度需与方法配套**：论文强调 bfloat16 vs float16 可造成数个百分点差异，提醒在消融实验中严格控制格式一致性，避免格式噪声淹没方法信号。
4. **可与本团队方向结合**：若团队涉及多专家聚合或模型合并，TF-OPD 提供了无需训练的高效 baseline；其诊断指标 $\varepsilon_d$ 可用于分析任意融合方法的冲突分布。

## 关键术语表
**On-policy Distillation**：让教师对模型自身生成的文本进行监督的蒸馏方式，而非固定外部语料。
**Task Arithmetic**：将多个 fine-tuning 的权重更新直接相加（$\Delta = \sum \Delta_d$）合并模型的方法。
**RegMean**：基于输入二阶矩的最小二乘权重合并方法，通过关闭非对角线元素改善条件数。
**Repetition Loop**：长推理模型输出中进入重复模式的失败答案，会被截断至长度上限。
**Gram Matrix / 二阶矩**：输入协方差矩阵 $G = \mathbb{E}[xx^\top]$，在本方法中用于加权各方向上的拟合重要性。
**Forward Hook**：在 PyTorch 中注册的前向传播钩子，用于在不修改模型结构的情况下累积中间张量统计量。
**Teacher Forcing**：推理时将真实答案 token 作为每一步的输入，而非使用模型自身的预测。

## 可复现要素
- **数据集**：DAPO-Math-17k（数学训练集）、DeepCoder 训练集（代码训练集）、AIME24/AIME25/HMMT25/MATH-500/OlympiadBench（数学评测）、HumanEval+/MBPP+（代码评测）——均来自已有公开资源。
- **代码开源**：论文未明确声明开源，但提到算法在 Appendix B 给出伪代码（Algorithm 1）。
- **模型权重**：Olmo 3 系列以 Apache 2.0 发布（allenai/Olmo-3-1025-7B 等）。
- **关键超参**：$\alpha = 0.9$（收缩系数）、$w_d = 1$（等权重）、数学温度 0.7 / top-p 0.9 / 31744 tokens、代码温度 1.0 / top-p 1.0 / 32768 tokens、循环检测阈值 DEFLATE 压缩率 < 0.05、最后 20000 字符窗口。
