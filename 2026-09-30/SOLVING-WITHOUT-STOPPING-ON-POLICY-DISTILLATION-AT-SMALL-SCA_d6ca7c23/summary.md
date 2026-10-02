---
title: "SOLVING-WITHOUT-STOPPING-ON-POLICY-DISTILLATION-AT-SMALL-SCA"
source: https://arxiv.org/pdf/2609.37326v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:21:54"
field: "小模型推理增强"
keywords: ["on-policy distillation", "small language models", "long reasoning", "thinking mode", "stopping behavior", "knowledge distillation", "math reasoning"]
innovations: ["发现 on-policy distillation 在 thinking 模式下不教停止而是过滤已有停止，最小学生几乎丧失停止能力", "提出求解能力与停止能力的解耦诊断框架，将 p@1 变化归因至 answer/accuracy/stop 三个维度", "量化 on-policy distillation 的两个天花板：学生自身 reach(p@n) 和教师性能比例，并系统测量 across 尺寸与模式"]
benchmarks: ["AIME 2026", "AMC23", "GSM8K-200", "DAPO-Math-17k"]
---

# 论文速读：SOLVING-WITHOUT-STOPPING-ON-POLICY-DISTILLATION-AT-SMALL-SCA

## 一句话总结
本文系统分析了小模型范围内（Qwen3-8B→4B/1.7B/0.6B）的 on-policy distillation 在 thinking 模式与 non-thinking 模式下的知识迁移机制，发现蒸馏能有效提升小模型的"求解"能力但无法教会其"何时停止"——thinking 模式下蒸馏实质上是对已有 stop 的筛选而非教学，最小学生甚至失去停止推理的能力。

## 研究问题与动机
1. **核心问题**：在长推理（thinking mode）场景下，on-policy distillation 究竟向小模型传递了什么？小模型是否能同时学会"解题"和"知道何时解完"这两项能力。
2. **现有方法不足**：仅看终点分数（p@1）无法区分"答案错误"和"未停止输出"，容易导致对蒸馏效果的误判。
3. **已有研究空白**：此前工作多关注 overthinking 或 early exit 等停止问题本身，但没有人系统度量 thinking 模式下 on-policy distillation 对 small student 的停止行为到底产生了什么影响。
4. **动机**：small models 几乎完全依赖 distillation 获得推理能力，明确"教了什么、没教什么"对后续方法设计有直接指导意义。

## 核心贡献（创新点）
1. **首次在小规模上全面量化 on-policy distillation 的上界**：提出两个天花板——学生自身预训练 reach（p@n）和教师性能比例，并系统测量 across 所有学生尺寸与两种模式。
2. **发现 thinking 模式下蒸馏不教停止而是过滤停止**：教师的 stop 信号几乎只在学生已停止的位置出现，蒸馏过程只保留正确 stop、移除错误 stop，导致小学生的 stop 能力大幅退化。
3. **构建了解耦诊断框架**：将模型表现分解为"是否作答（answer）""答案是否正确（correctness）""是否停止（stopping）"三个维度，使性能变化可归因到具体行为。
4. **揭示了未截断样本中隐藏的正确值**：即使模型未标记最终答案，正确值在文本中出现的频率远高于随机水平（4–7×），说明小模型经常"做到一半却写完了"。

## 方法详解
- **蒸馏设置**：Teacher = Qwen3-8B，Students = Qwen3 4B / 1.7B / 0.6B，分别在 thinking 模式和 non-thinking 模式下蒸馏。Thinking 模式学生先做 supervised fine-tuning（两种数据来源：5,878 条外部短解 或 1,524 条教师正确解）。
- **训练数据**：13,597 个 DAPO-Math-17k 问题，每步 128 样本，rollout budget = 7,168 tokens，训练 142 步（thinking 模式）/ 212 步（non-thinking 模式）。
- **目标函数（PPO-style）**：每个 token 以 clipped reverse KL divergence 作为 advantage，无任务奖励：
  $$A_t = -\text{clip}\big(\log \pi_{\text{old}}(y_t|x, y_{<t}) - \log \pi_T(y_t|x, y_{<t}),\; -1,\; 1\big)$$
  关键设计：`</think>` token 像其他 token 一样被评分，因此停止压力仅来自教师在该位置的停止概率。
- **评估基准**：AIME 2026（48 samples）、AMC23（48 samples）、GSM8K-200（32 samples），token 上限 30,720（GSM8K-200 为 8,192），temperature=0.6。
- **打分方式**：Lenient scoring（只要任意一个 boxed 答案正确即算对）；另有 strict 和 committed 两种更严格打分。
- **诊断指标**：closure rate（使用 `</think>` 关闭推理的比例）、truncation（触达 token 上限的比例）、stop reliability（停止时答案正确的比例）。

## 实验与结果
**数据集与基准**：DAPO-Math-17k（训练），AIME 2026 / AMC23 / GSM8K-200（评估）。

**主要数值结果（Table 1，三基准均值，lenient p@1）**：
- Non-thinking OPD（教师 p@1=0.595）：4B 从 0.427→0.549，1.7B 从 0.294→0.440，0.6B 从 0.048→0.328。
- Thinking OPD + external-SFT（教师 p@1=0.857）：4B 从 0.532→0.651，1.7B 从 0.358→0.392，0.6B 从 0.238→0.294。
- Thinking OPD + teacher-SFT（教师 p@1=0.857）：4B 从 0.330→0.649，1.7B 从 0.128→0.394，0.6B 从 0.055→0.262。

**关键结论**：
- 蒸馏后每个学生的 p@1 始终低于其起始 p@n（第一天花板），无一越界。
- 蒸馏后 p@1 达到非 thinking 教师的比例：4B 为 92.2%，1.7B 为 73.9%，0.6B 为 55.1%。
- thinking 模式下，蒸馏后 p@1 达到 thinking 教师的比例：4B 约 76%，1.7B 约 46%，0.6B 约 34%。
- 硬题（AIME 2026）差距最大：1.7B 和 0.6B thinking 模式蒸馏后仅达教师 p@1 的十分之一以下。
- **停止率崩溃**：external-SFT 起始点在 step 6 时 closure rate 从 ~0.5~0.6 骤降至 ~0.1~0.2（4B: 0.617→0.211；1.7B: 0.445→0.128；0.6B: 0.583→0.113），在所有 thinking 模式下均一致发生。
- 教师 stop 信号分析（Table 13）：学生在某位置停止时教师同意概率为 0.976（4B）/ 0.764（1.7B）/ 0.673（0.6B）；但在未完成轨迹上教师给出 stop 信号的概率最高仅为 ~10^-7 量级。

## 相关工作脉络
1. **On-policy distillation**（Ross et al., 2011; Agarwal et al., 2024; Gu et al., 2024; Lu & Thinking Machines Lab, 2025）：本文沿用核心配方，不同在于系统测量其在 thinking/non-thinking 两种模式及不同尺寸下的边界。
2. **Off-policy distillation**（Hinton et al., 2015; Kim & Rush, 2016）：从教师输出生成数据训练学生；本文聚焦 on-policy（学生自己生成）。
3. **Stop/过思考研究**（Chen et al., 2025; Zhang et al., 2025; Yang et al., 2026; Davidov et al., 2026）：关注推理长度控制、early exit、abstention；本文从训练角度分析停止行为如何被蒸馏改变。
4. **RL with verifiable rewards**（Yue et al., 2025）：发现单样本提升仍受限于基础模型多采样 reach；本文将此结论扩展到 on-policy distillation 场景并给出两个天花板的精确度量。
5. **Imitation learning gap**（Weihs et al., 2021; de Haan et al., 2019; Swamy et al., 2022）：当学生缺少教师用于决策的信息时存在模仿差距；本文将此概念应用于停止行为分析。
6. **Recent OPD variants**（Jin et al., 2026; Fu et al., 2026; He et al., 2026; Zhou et al., 2026; Xin et al., 2026; Ma, 2026; Kaur et al., 2026）：试图修改 divergence、credit assignment 或 mask stop token；本文指出问题根源在状态而非 token 本身，为这些方法提供定位参照。

## 局限性与未来方向
- **规模局限**：仅覆盖 4B/1.7B/0.6B 三个学生尺寸，更小规模（如 <0.6B）或更大模型的行为未知。
- **任务局限**：仅测试数学推理（DAPO-Math），未涉及代码、科学等其他长推理领域。
- **蒸馏配方固定**：采用单一标准配方，未探索 loss 修改、curriculum 或其他优化策略对停止行为的挽救效果。
- **推理方向**：未来工作应在"响应中间过程"而非"stop token"处介入，例如在教师信号缺失的位置主动注入 stopping guidance。
- **可结合 s1-style test-time forcing**（Muennighoff et al., 2025）：本文 appendix E 已演示强制停止可恢复部分正确答案，可作为后续改进方向。

## 研究启发与可借鉴点
1. **诊断框架可直接复用**：将 p@1 拆解为 answer rate × answer accuracy，并独立测量 stop rate 和 stop reliability，适用于任何长推理蒸馏论文的评估体系。
2. **正确值隐藏检测可作为验证手段**：在未标记答案的截断样本中搜索正确值出现频率（对比 chance），可快速判断模型是否"已解出但未输出"。
3. **rollout budget 与停止行为强耦合**：论文显示关闭率崩溃恰好发生在响应填满 rollout budget 的步骤，提示未来需将 budget 与停止监督联合优化。
4. **teacher stop signal 分析可作为 ablation**：评估教师对学生各位置停止概率的分布，可快速定位蒸馏过程中信息传递的断裂点。
5. **external-SFT vs teacher-SFT 的对比思路**：两种起始数据的差异揭示了格式学习与内容学习的不同效应，可为 cold-start 设计提供方法论参照。

## 关键术语表
- **On-policy distillation**：学生用自己的生成样本训练自身，由更强的教师对每个 token 打分提供 advantage 信号，区别于 off-policy 从教师输出蒸馏。
- **Thinking mode**：模型先进行长链式推理（在 `<think>...</think>` 标签内），然后给出最终答案的模式。
- **Non-thinking mode**：模型跳过显式推理阶段，直接在 chat template 填充的空思考块后作答。
- **Lenient scoring**：只要样本中任意一个标记答案（`\boxed{}` 或 `Answer:` 行）正确即判定为正确，忽略样本是否提前停止。
- **p@n**：对同一问题采 n 次，至少一次成功的比例；p@1 即单次成功率。
- **Stop reliability**：学生停止推理时其答案正确的比例，反映停止位置的准确性。
- **Closure rate**：样本以 `</think>` 关闭推理的比例，衡量学生完成长推理输出的能力。
- **Truncation**：样本在 token 上限处被截断的比例，反映溢出预算的情况。

## 可复现要素
- **数据集**：训练集 DAPO-Math-17k（公开），外部 SFT 数据来自 OpenR1-Math-220k 子集（Apache-2.0，公开），教师 SFT 数据为教师自身生成（公开）；评估集 AIME 2026、AMC23、GSM8K-200 均为公开基准。
- **代码/权重**：教师及 pretrained-only / officially post-trained 学生均为公开 checkpoint；代码、衍生数据和配置声明"upon acceptance"后开源。
- **关键超参**：每步 128 prompts / 4 mini-batches；AdamW lr=3×10^-6，10 warmup steps 后恒定；weight decay=0.01；gradient clip=1.0；advantage clipped reverse KL at ±1，无任务奖励；rollout per prompt，temperature=1.0，budget=7,168 tokens；训练 142 steps（thinking）/ 212 steps（non-thinking）；SFT lr=2×10^-5，2 epochs，cosine schedule。硬件：3 nodes × 4 A100-40GB。
