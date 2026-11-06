---
title: "Value-Rich-Differently-Organized-Measuring-Value-Expression"
source: https://arxiv.org/pdf/2610.28604v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:58:42"
field: "辩论评估与价值分析"
keywords: ["Schwartz Basic-10", "value expression", "value organization", "debate evaluation", "AI-generated debate", "multi-level value distribution", "ORCHID"]
innovations: ["首次区分并形式化价值表达与价值组织两个正交评估维度", "构建基于Schwartz理论的辩论多粒度价值分布表示与诊断性指标族", "揭示AI辩论广覆盖但低集中、跨轮高稳定性的系统性模式"]
benchmarks: ["ORCHID", "GPT-5.4", "Claude-3.6 Sonnet", "Qwen3.6-Plus", "DeepSeek-V3.2", "R-Debater"]
---

# 论文速读：Value-Rich-Differently-Organized-Measuring-Value-Expression

## 一句话总结
本文提出了一套基于Schwartz Basic-10价值理论的诊断性度量框架，用于评估AI生成辩论与人类辩论在**价值表达**（覆盖了哪些价值观）和**价值组织**（价值观如何在论次间分布和延续）两个维度的差异，发现AI辩论通常激活更广泛的价值维度，但在局部价值集中度和跨轮值结构上与人存在系统性差异。

## 研究问题与动机
- **核心问题**：现有LLM辩论评估主要关注连贯性、事实性、立场一致性等表面/逻辑质量，却鲜少系统性地分析辩论中“价值观”是如何被表达和组织跨多轮论证的。
- **现有方法不足1（维度缺失）**：以往的Value-oriented NLP（如Webis-ArgValues-22、ValueEval）多将价值观视为单篇孤立文本的标签，忽略了辩论的**多轮交互**与**跨轮组织**特性。
- **现有方法不足2（隐含价值）**：辩论中价值很少以显式标签形式出现，而是隐含在论据前提中，难以直接通过关键词匹配捕捉。
- **现有方法不足3（结构粒度）**：价值在辩论中 operates at multiple levels（论据单元、 utterance、 stance、 full debate），现有指标缺乏从局部到全局的多粒度刻画。

## 核心贡献（创新点）
1. **概念区分**：首次明确区分并形式化了**Value Expression**（价值表达广度与对齐）与**Value Organization**（价值集中度与跨轮动态），突破了以往单一“价值是否出现”的二元视角。
2. **理论 grounded 的多粒度框架**：以Schwartz Basic-10理论为共享价值空间，构建了从argumentative unit → utterance → stance → debate四个层级的概率分布表示与聚合方法。
3. **可计算的诊断性指标族**：设计了覆盖Coverage、Top-k Overlap、JSD（表达维度）以及Entropy、Stability、Drift、Persistence（组织维度）的一整套可计算指标。
4. **配对基准与实证发现**：在ORCHID中文辩论语料上构建paired benchmark，对比5个生成系统（GPT-5.4、Claude-3.6 Sonnet、Qwen3.6-Plus、DeepSeek-V3.2、R-Debater）与人类参考，揭示了AI辩论“广覆盖但低集中、跨轮更平滑稳定”的系统性模式。
5. **源盲人工评估验证**：通过120名辩论队成员的双盲评估，证实了指标所捕获的“价值清晰度”与“价值焦点”是可感知的、且AI与人类在此维度呈现分离模式。

## 方法详解
### 1. 共享价值空间
采用Schwartz Basic-10价值观理论，包含10个价值维度：Self-Direction (SD)、Stimulation (STM)、Hedonism (HED)、Achievement (ACH)、Power (POW)、Security (SEC)、Conformity (CON)、Tradition (TRD)、Benevolence (BEN)、Universalism (UNI)。每个辩论被视为一个在该10维空间上的分布。

### 2. 辩论的形式化表示
辩论表示为 $d = (q, \mathcal{Z}, \mathcal{C}, \mathcal{U})$，其中 $q$ 为议题，$\mathcal{Z}=\{+, -\}$ 为正反立场，$\mathcal{C}$ 为立场特定主张，$\mathcal{U}$ 为有序utterance序列。每轮 utterance $u_t$ 具有文本 $x_t$、立场 $z_t$ 和轮次索引 $t$。

### 3. 论据单元构建（SAPE-Agent）
基于claim-premise视图，对每轮 utterance 使用SAPE-Agent（Stance-Aware Premise Extraction Agent）提取前提span $\mathcal{P}(u_t)$，每个前提与立场特定主张 $c_{z_t}$ 配对战构成论据单元：
$$a_{t,i} = (c_{z_t}, p_{t,i})$$
论据单元集合为 $A(u_t) = \{a_{t,1}, ..., a_{t,k_t}\}$。

### 4. 价值标注（VA-Agent）
使用VA-Agent（Value Annotation Agent）为每个论据单元分配Basic-10价值软得分 $s_v(a) \in [0,1]$，经归一化得到概率分布 $\mathbf{p}(a) \in \Delta^{9}$：
$$p(v|a) = \frac{s_v(a)}{\sum_{v' \in V} s_{v'}(a)}$$

### 5. 多粒度聚合
- **Utterance级**：$\mathbf{P}_{utt}(u_t) = \frac{1}{k_t} \sum_{a \in A(u_t)} \mathbf{p}(a)$
- **Stance级**：$\mathbf{P}_{sta}(d|z) = \frac{1}{n_z} \sum_{u \in U_z(d)} \mathbf{P}_{utt}(u)$
- **Debate级**：$\mathbf{P}_{deb}(d) = \frac{1}{n^*} \sum_{u_t \in U^*(d)} \mathbf{P}_{utt}(u_t)$

### 6. 表达维度指标
- **Coverage**：$\sum_{v \in V} \mathbb{I}[P_{deb}(v|d) \geq \tau]$，统计激活的价值维度数
- **Top-k Overlap**：$\frac{|\text{Top}_k(d^H) \cap \text{Top}_k(d^M)|}{k}$，衡量主导价值重合度
- **JSD**：$\text{JSD}(\mathbf{P}_{deb}(d^H) \| \mathbf{P}_{deb}(d^M))$，衡量分布对齐程度

### 7. 组织维度指标
- **Value Entropy**：$H(\mathbf{P}) = -\sum_v P(v)\log P(v)$，刻画价值集中度（高熵=分散）
- **Value Stability**：$1 - \frac{1}{n_z-1}\sum \text{JSD}(\mathbf{P}_{utt}(u_{z,i}) \| \mathbf{P}_{utt}(u_{z,i+1}))$，衡量同立场跨轮变化平滑度
- **Value Drift**：$\text{JSD}(\mathbf{P}_{utt}(u_{z,1}) \| \mathbf{P}_{utt}(u_{z,n_z}))$，衡量立场开首到结尾的价值偏移
- **Dominant-value Persistence**：统计相邻同立场utterance间Top-k主导价值是否有交集

### 8. 配对差异计算
对任意指标 $\mu$，计算 $\Delta\mu(i,g) = \mu(d_{i,g}^M) - \bar{\mu}(d_i^H)$，正值表示AI生成辩论该指标更高，不代表质量更优。

## 实验与结果
### 数据集与基线
- **数据集**：ORCHID中文辩论语料，采样60个议题，每议题匹配1场人类辩论作为reference
- **生成系统**：GPT-5.4、Claude-3.6 Sonnet、Qwen3.6-Plus、DeepSeek-V3.2、R-Debater
- **议题类型**：fact、value、policy 三种motion type

### 核心结果
| 系统 | Coverage | Top-3 Overlap | JSD | $H_{utt}$ | $H_{deb}$ |
|------|----------|---------------|-----|-----------|-----------|
| Human (ref) | 4.883 | 1.000 | 0.000 | 1.252 | 1.651 |
| GPT-5.4 | 5.600 | 0.572 | 0.133 | 1.470 | 1.712 |
| Claude-3.6 Sonnet | 5.450 | 0.628 | 0.128 | 1.476 | 1.756 |
| Qwen3.6-Plus | 5.567 | 0.650 | 0.115 | 1.517 | 1.722 |
| DeepSeek-V3.2 | 5.400 | 0.667 | 0.102 | **1.553** | 1.744 |
| R-Debater | 5.520 | 0.633 | 0.123 | 1.532 | 1.761 |

**关键发现**：
1. **价值表达广度**：所有AI生成辩论的Coverage（5.4–5.6）均高于人类参考（4.883），表明AI辩论激活了更广泛的价值维度
2. **价值对齐度**：Top-3 Overlap仅0.57–0.67，JSD为0.10–0.13，说明广泛覆盖不等于高对齐
3. **局部价值扩散**：AI生成的utterance级熵 $H_{utt}$ 显著更高（1.47–1.55 vs 1.25），effect size $d_z$ 达0.85–1.30，表明单轮内价值分布更分散
4. **跨轮稳定性**：AI辩论的Stability（0.92–0.95）显著高于人类（0.83），Drift更低（0.05–0.22 vs 0.21），Persistence接近完美（0.98–1.0）
5. **控制表面因素后差异仍稳健**：回归控制token长度、论据单元数、lexical diversity、BLEU、Self-BLEU后，$\Delta H_{utt} = +0.252$（95% CI [0.205, 0.300]）仍显著
6. **Motion type效应**：fact motion下entropy差异最大，policy motion相对异构

### 人工评估验证
- 120名辩论队成员参与源盲评估
- **价值清晰度**：GPT-5.4（6.01）、DeepSeek-V3.2（5.93）高于人类（5.77）
- **价值焦点**：人类最高（5.91），在74%的pairwise focus比较中被选中
- **Human-debate likeness**：人类（6.42）> GPT-5.4（5.09）> DeepSeek-V3.2（4.88）
- 证实了"AI辩论价值表达清晰但焦点分散"的模式

### 最强结果
- **Coverage最大**：GPT-5.4（5.600）
- **JSD最小（最对齐）**：DeepSeek-V3.2（0.102）
- **Utterance熵最大**：DeepSeek-V3.2（1.553，$\Delta H_{utt} = +0.301$）
- **Persistence最高**：Qwen3.6-Plus、DeepSeek-V3.2、R-Debater均为1.000

## 相关工作脉络
1. **辩论生成与评估**：Project Debater (Slonim et al., 2021)、LLM多智能体辩论 (Du et al., 2024; Liang et al., 2024b)、检索增强辩论 (Li et al., 2025)、多维度辩论评判 (Liang et al., 2024a; Wang et al., 2025)。本文定位：这些工作关注论据质量/胜方预测，但**未涉及价值观在多轮的分布与组织**。
2. **价值挖掘NLP**：Webis-ArgValues-22 (Kiesel et al., 2022)、SemEval-2023 ValueEval (Kiesel et al., 2023)、Touché23-ValueEval (Mirzakhmedova et al., 2024)。本文定位：这些工作将价值视为**孤立文本标签**，本文扩展至**配对多轮辩论的分布分析**。
3. **社会场景价值建模**：ValueNet (Qiu et al., 2022)、FULCRA (Yao et al., 2024)。本文定位：这些数据集聚焦对话/社会场景，本文专注于**竞争性辩论语境**下的价值组织模式。
4. **基于价值的论证理论**：Bench-Capon (2003) 的value-based argumentation frameworks。本文定位：理论层面借鉴了"论据可接受性取决于所 promoted 的价值及受众价值排序"的思想，但在**计算方法与指标化**上是首创。
5. **LLM-as-judge辩论评估**：Sternlicht et al. (2025)。本文定位：侧重judge模型benchmarking，本文则提供**可解释的、基于价值理论的诊断性度量**而非整体排名。
6. **自动论据质量评估**：Toledo et al. (2019)、Bar-Haim et al. (2021)。本文定位：关注逻辑有效性/说服力，本文补充**价值观组织维度**。

## 局限性与未来方向
- **价值空间局限**：Schwartz Basic-10无法覆盖所有文化特定、意识形态、宗教、修辞或领域特定的价值，其他价值框架可能揭示互补模式。
- **辩论交互简化**：ORCHID为固定轮次配对设计，简化了真实辩论中的打断、观众反馈、身份线索、自发策略转换等动态。
- **跨语言泛化有限**：主基准为中文，附录C的跨语言分析（中英对比）仅为探索性，大规模多语言评估仍是未来方向。
- **自动提取误差**：SAPE-Agent和VA-Agent虽经人工验证（κ=0.71, F1=0.82），但对隐性、隐喻、文化特定价值表达仍可能存在误差。
- **指标描述性而非规范性**：Entropy、Stability、Drift、Persistence是描述性组织指标，非直接质量分数；高熵可能反映包容性论证，不一定是缺陷。
- **不适用于推断模型内部信念**：结果仅反映文本表达的价值，不能解读为模型内部信仰或对齐状态。

## 研究启发与可借鉴点
1. **多粒度价值分布表示可迁移**：将Schwartz Basic-10分布从utterance→stance→debate的层级聚合思路，可直接迁移至其他多轮交互任务（如客户服务对话、医疗问诊、法律辩论）的价值追踪分析。
2. **配对控制设计值得借鉴**：通过topic/motion-type/turn-structure/utterance-length的配对控制，再辅以surface-control回归，有效剥离了文本表层因素对价值指标的混淆，这一实验设计范式可复用于其他AI生成内容的对比评估。
3. **"表达-组织"二分框架具有通用性**：将评估拆分为"是否出现"（expression）与"如何组织"（organization）两个正交维度，可推广至立场强度分布、情感轨迹、论证结构等其他维度的诊断性分析。
4. **SAPE-Agent+VA-Agent双Agent流水线**：先抽取前提span再标注价值分布的解耦设计，既提升了可解释性，又通过多次标注平均降低了方差，这一pipeline结构可适配到其他需要细粒度语义标注的场景。
5. **与交叉生成/对比学习结合的机会**：当前发现AI辩论"广覆盖但低集中"，未来可探索通过loss引入entropy约束或persistence正则，引导生成模型学习人类式的价值聚焦模式，实现"价值组织-aware"的辩论生成优化。

## 关键术语表
- **Value Expression**：辩论中所激活的价值观的广度及其与人类参考的对齐程度，关注"有哪些价值"。
- **Value Organization**：价值观在论据单元、utterance、stance、整场辩论各层级的分布形态与跨轮动态，关注"价值如何被组织"。
- **Schwartz Basic-10**：Schwartz提出的10个基本价值观维度（SD/STM/HED/ACH/POW/SEC/CON/TRD/BEN/UNI），作为跨文本的价值分析共享空间。
- **Argumentative Unit**：由立场特定主张 $c_z$ 与提取的前提span $p$ 配对构成的 $(c_z, p)$ 单元，是价值标注的基本粒度。
- **SAPE-Agent**：Stance-Aware Premise Extraction Agent，基于辩论话题、立场和上下文抽取前提span的LLM agent。
- **VA-Agent**：Value Annotation Agent，为每个论据单元分配Basic-10价值软得分的LLM agent。
- **Value Entropy**：基于Shannon熵计算的价值分布集中度的度量，高熵表示价值分布更分散。
- **Dominant-value Persistence**：同立场相邻utterance间Top-k主导价值的重叠率，反映价值核心在多轮中的延续性。

## 可复现要素
- **数据集**：ORCHID中文辩论语料（Zhao et al., 2023），60个采样议题，公开可用
- **代码**：https://github.com/Maoyuan-li/Value-Expression-and-Organization（已开源）
- **补充材料**：prompts、annotation protocol、human evaluation questionnaire均在附录中提供
- **关键超参**：
  - Coverage阈值 $\tau = 0.05$
  - Persistence计算中 $k=3$
  - VA-Agent多次标注取平均（论文未明确说明R次，见Appendix A）
  - 统计检验：Wilcoxon signed-rank test + Holm correction
