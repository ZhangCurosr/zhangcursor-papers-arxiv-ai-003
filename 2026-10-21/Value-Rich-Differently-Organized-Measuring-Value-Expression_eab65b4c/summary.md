---
title: "Value-Rich-Differently-Organized-Measuring-Value-Expression"
source: https://arxiv.org/pdf/2610.28604v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:58:52"
field: "多轮对话与论证分析"
keywords: ["价值观评估", "辩论生成", "Schwartz Basic-10", "跨轮分析", "AI辩论", "价值观组织"]
innovations: ["提出价值观表达与组织的双维诊断框架", "设计覆盖度、对齐度、熵、稳定性、漂移、持续性六个指标", "揭示AI辩论激活更广价值观空间但局部更分散的系统性差异"]
benchmarks: ["ORCHID中文辩论语料库"]
---

# 论文速读：Value-Rich-Differently-Organized-Measuring-Value-Expression

## 一句话总结
本文提出了一个基于Schwartz基本10价值观理论的诊断性指标框架，用于量化比较AI生成辩论与人类参考辩论在"价值观表达"（激活的价值观广度）和"价值观组织"（价值观在多轮对话中的分布与演化）两个维度的系统性差异，发现AI辩论能激活更广泛的价值观空间，但在局部集中度和跨轮次动态上呈现出与人类辩论不同的模式。

## 研究问题与动机
- 现有LLM辩论评估主要关注连贯性、事实性、立场一致性、逻辑有效性等论证质量维度，忽视了辩论中价值观的组织结构特征。
- 现有价值观挖掘工作（如ValueEval、Touché23-ValueEval）将价值观视为独立文本的离散标签，未将其建模为跨多轮辩论对话的分布与轨迹。
- 辩论参与者不仅交换主张和证据，还通过论证展现对公平、安全、自由、尊严等的价值诉求，这些价值诉求的结构性差异可能影响对辩论质量的感知。
- 需要一种理论 grounded 的多粒度指标体系，使研究者能区分"价值观是否存在"和"价值观如何被组织"这两种不同性质的问题。

## 核心贡献（创新点）
- **提出"价值观表达"与"价值观组织"的诊断性区分**：突破了以往仅关注价值观是否出现的二元视角，将评估维度拓展至价值观的分布结构和跨轮演化动态。
- **构建了基于Schwartz Basic-10的多层级价值观表示框架**：以argumentative unit（claim-premise pair）为基本单元，将Soft值分布聚合至话语、立场、辩论三个粒度，与已有研究中将价值观视为孤立文本标签的做法本质不同。
- **设计了一套覆盖覆盖度、对齐度、局部集中度与跨轮动态的指标族**：包括Coverage、Top-k Overlap、JSD、Entropy、Stability、Drift、Persistence七个指标，填补了跨轮价值观动态量化评估的空白。
- **通过配对分析和无源盲评揭示了AI辩论的组织特征**：发现AI辩论激活更多价值观维度但局部熵更高、跨轮稳定性更强且漂移更小，且这些差异不能归因于表层文本因素。

## 方法详解
- **共享价值观空间**：采用Schwartz Basic-10理论作为价值观空间$\mathcal{V}$，包含Self-Direction、Stimulation、Hedonism、Achievement、Power、Security、Conformity、Tradition、Benevolence、Universalism十个维度。
- **论证单元构建**：基于Stab & Gurevych的claim-premise观点，对每轮话语$u_t$（立场$z_t$），通过SAPE-Agent提取前提跨度$\mathcal{P}(u_t)$，与立场特定主张$c_{z_t}$配对构成论证单元$a_{t,i} = (c_{z_t}, p_{t,i})$。
- **Soft值分布分配**：VA-Agent为每个论证单元分配10维Basic-10值分布$\mathbf{p}(a) \in \Delta^{|\mathcal{V}|-1}$，经R次运行取平均后归一化得到$p(v|a)$。
- **三级聚合**：
  - 话语级：$\mathbf{P}_{\text{utt}}(u_t) = \frac{1}{k_t}\sum_{a \in \mathcal{A}(u_t)}\mathbf{p}(a)$
  - 立场级：$\mathbf{P}_{\text{sta}}(d|z) = \frac{1}{n_z}\sum_{u \in \mathcal{U}_z(d)}\mathbf{P}_{\text{utt}}(u)$
  - 辩论级：$\mathbf{P}_{\text{deb}}(d) = \frac{1}{n^*}\sum_{u_t \in \mathcal{U}^*(d)}\mathbf{P}_{\text{utt}}(u_t)$
- **价值观表达指标**：Coverage（阈值$\tau=0.05$下激活的价值观维度数）、Top-k Overlap（Top-3重叠比例）、JSD（与人类参考的分布散度）。
- **价值观组织指标**：
  - 熵$H(\mathbf{P}) = -\sum_v P(v)\log P(v)$，衡量局部集中度；
  - 稳定性$1 - \frac{1}{n_z-1}\sum \text{JSD}(\mathbf{P}_{\text{utt}}(u_{z,i}) \| \mathbf{P}_{\text{utt}}(u_{z,i+1}))$；
  - 漂移$\text{JSD}(\mathbf{P}_{\text{utt}}(u_{z,1}) \| \mathbf{P}_{\text{utt}}(u_{z,n_z}))$；
  - 持续性$\frac{1}{n_z-1}\sum \mathbb{I}[\text{Top}_3(u_{z,i}) \cap \text{Top}_3(u_{z,i+1}) \neq \emptyset]$。
- **配对差异分析**：计算$\Delta\mu(i,g) = \mu(d^M_{i,g}) - \bar{\mu}(d^H_i)$，控制token长度、论点单元数、词汇多样性、BLEU相似度、Self-BLEU等表层变量后进行回归分析。

## 实验与结果
- **数据集**：从ORCHID中文辩论语料库中采样60个辩论话题，每话题1个人类参考辩论，生成对应AI辩论（GPT-5.4、Claude-3.6 Sonnet、Qwen3.6-Plus、DeepSeek-V3.2、R-Debater），按motion类型（fact/value/policy）分层。
- **标注验证**：SAPE-Agent与人工前提标注达到Cohen's $\kappa=0.7124$（micro）、Krippendorff's $\alpha=0.7035$；VA-Agent在多标签F1上达0.82，显著优于BERT（0.25）、SVM（0.20）和直接LLM标注（0.64）。
- **价值观表达结果**：人类参考平均激活4.883个Basic-10值，AI生成辩论激活5.4-5.6个（GPT-5.4最高5.600）；Top-3重叠范围为0.572-0.667，JSD范围为0.102-0.133；DeepSeek-V3.2的JSD最低（0.102）、Top-3重叠最高（0.667）。
- **价值观组织结果（最强差异）**：话语级熵差异最大，人类均值1.252，AI范围1.470-1.553，DeepSeek-V3.2的$\Delta H_{\text{utt}}=+0.301$（$d_z=1.301$）；立场级和辩论级差异逐层缩小；跨轮稳定性人类0.831，AI范围0.915-0.952；Qwen3.6-Plus/DeepSeek-V3.2/R-Debater的漂移极低（0.047-0.078），持续性接近1.0。
- **表面控制后结果**：回归分析表明，控制token数、论点单元数、词汇多样性、BLEU、Self-BLEU后，话语级熵差异仍显著（调整后$\Delta H_{\text{utt}}=+0.252$，95% CI $[0.205, 0.300]$），且motion-type效应显著。
- **盲评验证**：120名辩论队员 judges 中，AI辩论的value clarity评分（GPT-5.4为6.01）高于人类（5.77），但value focus评分（GPT-5.4为5.47）低于人类（5.91），人类在74%的配对比较中被选为更具value focus。

## 相关工作脉络
- **Argument Mining与ValueEval系列（Kiesel et al., 2022, 2023; Mirzakhmedova et al., 2024）**：聚焦从独立论辩文本中识别隐含价值观，本文与之区别在于将价值观建模为多轮对话中的分布与轨迹，而非静态标签。
- **Project Debater（Slonim et al., 2021）与自动化辩论系统**：关注辩论组件构建与自主辩论执行，本文聚焦于辩论产物的价值观结构分析，而非辩论系统本身。
- **Debate Evaluation工作（Liang et al., 2024a; Wang et al., 2025; Sternlicht et al., 2025）**：关注辩论质量的连贯性、说服力、逻辑有效性等多维评估，但未涉及价值观在跨轮次中的组织模式，本文填补了这一空白。
- **ValueNet（Qiu et al., 2022）与FULCRA（Yao et al., 2024）**：研究人类价值观在多轮对话或LLM输出中的分布，但本文针对竞争辩论场景，并引入了跨轮动态指标（Stability、Drift、Persistence）。
- **Schwartz价值观理论应用**：此前较少应用于辩论分析领域，本文将其引入辩论价值观量化评估，建立了理论 grounded 的价值观空间。

## 局限性与未来方向
- Schwartz Basic-10框架无法涵盖文化特异性、意识形态、宗教或领域特定的价值观，其他价值观框架可能揭示互补模式。
- 基准主要基于中文辩论语料（ORCHID），跨语言/跨文化推广受限；附录C的跨语言比较仅为探索性分析。
- 依赖自动前提提取和值标注，对隐式、隐喻性或文化特异性价值观诉求可能存在误差。
- 熵、稳定性等指标为描述性度量，非直接质量评分；高熵可能反映多视角包容性论证，不应简单视为缺陷。
- 未来方向：引入其他价值观理论框架、扩展至多语言辩论、结合辩论胜负与说服力进行综合评估、将框架应用于价值观驱动的对齐调优。

## 研究启发与可借鉴点
- **多粒度分布聚合框架可迁移**：将Soft标签分布从单元级逐级聚合至话语级、会话级、全局级的思路，可复用于对话系统、多轮论证分析等场景的价值观/态度量化评估。
- **跨轮动态指标设计精巧**：Stability、Drift、Persistence三个指标分别从平滑性、趋势偏移、主导属性延续性刻画跨轮演化，可作为对话质量评估的通用设计模板。
- **配对分析与表面控制方法值得借鉴**：按话题、立场、轮次、语言、长度精确匹配的配对设计，辅以token数、BLEU、Self-BLEU等表层变量的回归控制，有效剥离了生成系统与话题等混杂因素。
- **可结合本团队方向**：若团队涉及价值观对齐、多轮对话质量评估或AI辩论生成，可将本框架嵌入评估流水线，作为补充维度的诊断性工具。
- **盲评设计中"clarity"与"focus"的分离发现**：提示我们在评估生成内容时，应区分"表达清晰度"和"内容聚焦度"两个维度，避免单一评分带来的混淆。

## 关键术语表
- **Schwartz Basic-10**：Schwartz提出的十个基本价值观维度（自我导向、刺激、享乐、成就、权力、安全、从众、传统、仁慈、普世主义），作为本文价值观空间的基础。
- **Argumentative Unit（论证单元）**：由立场特定主张（stance-specific claim）与前提跨度（premise span）配对构成的最小论证单位，是值分布分配和聚合的基本粒度的基础。
- **Value Expression（价值观表达）**：辩论激活的Basic-10价值观范围的广度和与人类参考的对齐程度，涵盖Coverage、Top-k Overlap和JSD三个指标。
- **Value Organization（价值观组织）**：价值观在辩论各粒度（话语、立场、辩论）上的集中度分布与跨轮演化动态，涵盖Entropy、Stability、Drift和Persistence四个指标。
- **Value Entropy（价值熵）**：基于Shannon熵公式，衡量任一粒度上值分布的集中或扩散程度，越高表示价值观越分散。
- **Value Stability（价值稳定性）**：同一立场跨轮话语级值分布的JSD均值取反，反映价值观变化的平滑程度，越高表示跨轮切换越少。
- **Value Drift（价值漂移）**：同一立场首轮与末轮话语级值分布的JSD，衡量立场在整个辩论过程中的价值观偏移幅度。
- **Dominant-Value Persistence（主导价值持续性）**：相邻同立场话语的Top-3价值观交集比例，衡量主导价值观是否跨轮持续呈现。

## 可复现要素
- **数据集**：ORCHID中文辩论语料库（Zhao et al., 2023），60个辩论话题（论文声明），已公开可用。
- **代码/权重**：已开源，GitHub链接：https://github.com/Maoyuan-li/Value-Expression-and-Organization。
- **关键超参**：Coverage阈值$\tau=0.05$；Top-k中$k=3$；VA-Agent评分轮次$R$（论文未明确提及，见附录提示"经R次运行取平均"）。
- **生成系统**：GPT-5.4、Claude-3.6 Sonnet、Qwen3.6-Plus、DeepSeek-V3.2、R-Debater（Li et al., 2025）。
- **人类评估**：120名大学辩论队成员 judges，7点Likert量表，来源盲评配对比较。
