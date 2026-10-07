---
title: "Thin-Evidence-Thick-Priors"
source: https://arxiv.org/pdf/2610.07798v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:54:14"
field: "LLM算法公平性与可解释性"
keywords: ["LLM公平性", "反事实审计", "先验替代", "机制可解释性", "理财建议偏差", "身份依赖", "clustered bootstrap"]
innovations: ["提出prior substitution curve测量身份依赖随财务证据量连续变化的动态过程", "区分substitution effect与standing gap两类偏差模式并提供针对干预", "证明persona-clustered推断可将会议版9项显著交互降为1项稳健效应"]
benchmarks: ["Llama-3.1-8B-Instruct审计（96,598 prompts）", "Qwen2.5-1.5B-Instruct对照（同设计）"]
---

# 论文速读：Thin-Evidence-Thick-Priors

## 一句话总结
本文通过在 **Llama-3.1-8B-Instruct** 上对投资建议场景进行反事实审计，揭示了当用户披露的财务信息不足时，模型会系统性地依赖性别、家庭规模等身份特征替代缺失的财务证据；披露越充分，身份偏差越小，且"直接询问风险偏好"比提供多条无关财务事实更能抑制此类偏差。

## 研究问题与动机
1. **现实问题**：用户在使用 LLM 获取理财建议时往往无法完整披露财务信息（收入、债务、储蓄、目标等），而模型是否会在证据稀疏时"用身份填补空白"尚不清楚。
2. **现有审计方法的盲区**：已有 LLM 公平性审计（如 name-swap 研究）固定上下文长度、只测量单个信息水平，无法捕捉"身份依赖随证据稀释而放大"这一动态过程——而这正是低金融素养用户最常遭遇的场景。
3. **因果识别需求**：需要在财务 Profile 完全相同的前提下仅改变身份属性，才能将推荐变化归因于身份而非财务差异。
4. **可干预性需求**：需要打开模型内部表征，检验身份效应是否可被 mechanistic interpretability 工具（probe、patching、steering）定位和缓解。

## 核心贡献（创新点）
1. **提出并量化"先验替代曲线"（prior substitution curve）**：将财务披露程度作为连续旋钮（0–8 个事实、7 个嵌套水平），首次系统测量身份依赖如何随证据量线性/非线性变化，而非只在单一上下文点测量。
2. **发现三种不同的身份偏差模式**：家庭规模属于"替代型"（evidence-dependent，证据越缺影响越大）；性别和社会经济出身属于"固定差距"（standing gaps，无论披露多少始终存在）。现有审计框架无法区分二者。
3. **设计"L6 对照条件"分离数量与质量**：在仅披露 1 条事实时比较"风险偏好"与"投资目标"两条线索，证明决策相关信息的一条远胜于通用事实的若干条，为界面设计提供直接启示。
4. **在生成文本层面可视化替代机制**：在无财务事实条件下，98% 的推理文段包含模型虚构的收入、债务、储蓄等财务细节，且虚构内容系统化地与家庭规模挂钩，使替代过程对模型自身的解释文字可读。
5. **给出方法学教训：审计的单位应是 persona 而非 prompt**：使用 persona-clustered 标准误后，之前会议版报告的 9 个显著身份交互项中 8 个不再显著，证明"将重复提示视为独立样本"会严重低估不确定性、夸大统计显著性。

## 方法详解
**实验设计**：
- 构建 100 个合成财务 Profile（受 3 个一致性约束：目标-期限匹配、支出-储蓄匹配、无债时排除偿债目标），23 个种族×宗教组合产生 46 个名字，每个名字配 3 组不同属性，共 138 个 persona，交叉得到 96,600 个 prompt。
- 证据旋钮七个条件（Table 2）：L0（8 个财务事实）→ L1（7 个，不含风险偏好）→ L2（4 个）→ L3（2 个）→ L4（1 个：目标）→ L5（0 个）→ L6（1 个：风险偏好）。嵌套严格，L5 时所有 100 个 Profile 坍缩为同一 prompt 模板。
- 输出格式要求两行：≤20 词推理 + `Equity: NN%`，使用 greedy decoding，Llama-3.1-8B-Instruct。

**关键度量**：
- **Identity swing**：同一 Profile 内所有 persona 对两两绝对差均值（Gini mean差），衡量身份导致的平均偏移。
- **方差分解**：对 demeaned 后的答案计算 persona 贡献的比例，量化身份占解释力的份额。
- **交互项检验**：以 persona 为聚类单位的 mixed effects 模型，Wald 检验 + Bonferroni 校正；剂量编码采用三种方式（step index、withheld facts count、categorical）。
- **双路 cluster bootstrap**：在 persona 和 distinct financial context 两个维度上同时 resample（2,000 次），解决 L3–L5 处提示重复导致标准误过小的问题。

**内部分析**（12 Profile × 20 persona 子集）：
- **Probe**：每层 ridge regression 预测 equity 回答（risk probe）和 gender（identity probe）。
- **Activation patching**：将 donor persona 的最后 token residual stream 替换到 recipient 中，测量绝对变化量。
- **Steering**：在 layers 4, 5, 17, 20, 21 投影去除 gender 方向（余弦惩罚保护 risk 信号），重跑全部 96,600 prompt。

## 实验与结果
**数据集与模型**：
- 合成财务 Profile 100 个，Persona 138 个（46 names × 3 attribute combos），Llama-3.1-8B-Instruct，96,598 条有效响应。
- 并行运行 Qwen2.5-1.5B-Instruct 作为对照（87% 回答为单一值 50%，缺乏可测量方差）。

**主要结果数字**：
- **Prior substitution curve**：Identity swing 从 L0 的 4.78 pp 升至 L5 的 10.34 pp（比值 2.16，95% bootstrap 区间 1.69–2.79）。身份方差占比从 5%（L0）升至 96%（L5）。
- **L6 效应**：仅披露风险偏好时 swing 为 6.75 pp，低于 L4（单事实目标，8.08 pp），与 L1–L3 范围重叠；L6 明显低于 L5（差值 3.59，区间 0.61–6.87），说明一条决策相关信息可消除约 1/3 的身份摇摆。
- **最稳健的交互项**：家庭规模（≥5 dependants：−2.23 pp/step，p = 2×10⁻⁷；4 dependants：−1.38 pp/step，Holm 校正后唯一两项通过）；性别交互不显著（p = 0.32，persona-clustered）。
- **虚构财务文本**：L5 时无财务事实，但 98% 回复提及 horizon、86% 提及 employment/income、84% 提及 income；67% 的 L5 回复断言存在不利财务状况，且大家庭比例更高（89% vs 50%，p = 0.027），虚构内容直接影响推荐（有利虚构 → 47.0%，不利虚构 → 41.2%）。
- **Patch 效果**：平均 2.46 pp（L0）→ 1.77 pp（L1）→ 2.60 pp（L2）→ 2.23 pp（L3）；L5 全部为 0（原因：prompt 重合）。
- **Steering 结果**：移除 gender 方向后各层 swing 均无显著变化（最大差值 0.76 ± 0.72 pp，p = 0.30）；交互系数变化无法通过 paired test 验证（row-level 数据未保留）。
- **Name 效应**：Sipho +11.9 pp，Siti −9.8 pp；name 对 L5 方差解释 32%，6 个非名字属性解释 42%，F(45,92)=1.25, p=0.20 表明 name 本身作用不显著。

**最强结果与提升幅度**：核心发现为身份 swing 从 L0 到 L5 放大 2.16 倍（Bootstrap 1.69–2.79），L6 单条风险偏好可减少 L5 摇摆约 35%。

## 相关工作脉络
1. **Financial advice audit（Foltyn & Olsson 2026; Agliata & Hasso 2026）**：固定上下文切换 gender/race name 发现性别差异，本文将其扩展为连续证据旋钮，首次测量 bias 随信息披露程度变化的曲线而非单点估计。
2. **Counterfactual fairness in LLMs（Kusner et al. 2017; Garg et al. 2019）**：本文与之高度契合但更进一步——在财务场景中实现了严格的 counterfactual cross（财务 Profile 固定，身份随机化），且披露变量本身成为实验自变量。
3. **BBQ / UNQOVER（Parrish et al. 2022; Li et al. 2020）**：二元歧义 vs 消歧上下文的 QA 基准，本文借鉴但转化为连续剂量的金融决策任务，使得"依赖"可被测量为斜率。
4. **Mechanistic interpretability in bias（Vig et al. 2020; Cheng et al. 2023; Yang et al. 2024b）**：gender direction 去除实验直接对话这些方法，本文发现单方向 ablation 不减少 aggregate identity swing，提示偏差可能分布式存储，与 McGrath et al. (2023) 的 self-repair 现象呼应。
5. **Shortcuts in deep networks（Geirhos et al. 2020; McCoy et al. 2019）**：模型用身份作为捷径替代缺失财务证据，符合 shortcut learning 范式——在训练分布中身份与财务相关，但在审计设定下这一规则无数据支撑。
6. **Unfaithful rationales（Turpin et al. 2023; Lanham et al. 2023）**：模型生成的推理文段与其实际决策依据不一致，本文观察到模型不仅生成虚假财务细节，还使这些细节系统化地随身份变化，将 hallucination 与 bias 两个领域交汇。

## 局限性与未来方向
1. **单模型局限**：仅使用 Llama-3.1-8B-Instruct，小模型（Qwen2.5-1.5B）几乎无方差，无法判断曲线形状在其他模型族中的泛化性。
2. **合成数据与地理框架**：财务 Profile 为合成数据，使用美国收入框架；性别二元划分；种族轴不含南亚、中东、北非、原住民、太平洋岛民等群体。
3. **无 placebo 对照**：未设置无关属性（如星期几、喜欢的颜色）随证据旋钮的变化对照，无法完全排除"模型在低证据下对所有剩余信息更敏感"的一般性解释。
4. **低披露水平的 replication 有限**：L5 只有 138 个独立 prompt，L4 有 966 个，曲线细端的精确幅度不确定性较大。
5. **干预缺乏控制**：Steering 实验缺少 norm-matched random direction 和 unrelated attribute direction 作为对照；row-level steered 输出未保留，无法进行 paired persona-clustered 检验。
6. **Prompt 设计局限**：系统提示包含示例推理句（被模型复制到 91% 的 L5 回复中），夸大了"虚构财务"的表现；财务术语有 glossary 定义但身份术语无。
7. **防御性代理 vs 不当替代的边界模糊**：在部署场景中，家庭规模等属性与真实财务变量存在相关性，某些代理使用可能是合理的，本文设计无法对此做规范性区分。

## 研究启发与可借鉴点
1. **"证据旋钮"实验设计**：将信息披露作为连续自变量而非二元开关，适用于任何需要审计模型在 partial information 下行为的公平性研究，可迁移至信贷审批、招聘筛选等场景。
2. **分离替代型 vs 固定差距型偏差**：通过交互项检验区分"证据越缺偏差越大"（substitution）和"始终存在的差距"（standing gap）两类偏差模式，为针对性干预提供分类依据。
3. **Persona-clustered 推断的标准**：反事实审计中，当 treatment 在 persona 层面分配时，应将 persona 而非 prompt 作为统计单元，避免严重低估标准误——这一方法学教训适用于所有 persona-based audit 研究。
4. **L6 式"质量 vs 数量"对照设计**：在需要设计信息收集流程时，比较"一条决策关键信息"与"多条通用信息"的公平性效果，可为 UI/UX 设计提供实证依据。
5. **结合生成文本分析与内部表征**：同时观测模型输出（invented finances）和内部激活（probe/patching/steering），可在行为层面和机制层面交叉验证替代假设，形成更强的因果论证。

## 关键术语表
**Prior substitution curve（先验替代曲线）**：描述身份依赖程度随财务证据数量减少而变化的函数曲线，本文的核心量化指标。
**Identity swing（身份摇摆）**：同一财务 Profile 下不同 persona 的推荐 equity 分配均方差（Gini mean difference），衡量纯身份因素导致的平均推荐偏移量。
**Standing gap（固定差距）**：无论财务披露多少都持续存在的身份偏差（如性别），与证据量无关，需要通过模型层面干预而非信息补充来缓解。
**Substitution effect（替代效应）**：模型在财务证据缺失时用身份属性作为 proxy 推断未提供信息的过程，属于 shortcut learning 的一种表现。
**Two-way cluster bootstrap（双路 cluster bootstrap）**：在 persona 和 distinct financial context 两个维度上同时 resample 的 bootstrap 方法，用于处理低披露条件下提示重复带来的标准误低估问题。
**Activation patching（激活 patching）**：将 donor 的 residual stream 替换到 recipient 的前向传播中，测量特定层的因果效应。
**Risk probe / Identity probe**：分别在每层用 ridge regression 和 logistic regression 预测 equity 答案和 gender 的线性探针，前者有解释力，后者因 prompt 含 gender 词而饱和（layer 0 即 100%）。
**Steering（激活引导）**：在生成过程中投影去除指定方向的 residual stream 分量，本文用于 ablate gender 方向以测试其对身份依赖的影响。

## 可复现要素
- **数据集**：合成财务 Profile 100 个 + Persona 138 个，非公开；代码、数据生成器和模型输出可向作者合理申请获取。
- **模型**：Llama-3.1-8B-Instruct（开源权重），greedy decoding，固定 seed，batched generation with left padding。
- **关键超参**：答案格式限制为两行（reasoning ≤20 词 + `Equity: NN%`）；推理使用 statsmodels；bootstrap 2,000 次；95% percentile interval；Bonferroni 校正因子 9，Holm 校正 across 33 项。
- **内部分析子集**：12 profile × 20 persona = 1,680 forward passes；patching 使用前 30 个 profile；steering 选择 layers 4, 5, 17, 20, 21。
- **代码/权重**：论文声明"available upon reasonable request to the corresponding author"，未提供公开仓库链接。
