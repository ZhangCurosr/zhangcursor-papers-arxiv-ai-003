---
title: "STRUCTURALCOST-A-controlled-reading-time-dataset-for-modelin"
source: https://arxiv.org/pdf/2610.08208v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:52:18"
field: "计算心理语言学"
keywords: ["cognitive plausibility", "dependency length", "surprisal", "self-paced reading", "language models", "working memory"]
innovations: ["提出全局vs局部对齐区分框架，揭示surprisal在整合位点的系统性低估", "发布STRUCTURALCOST受控数据集，以NLP规模隔离依赖长度与词汇混淆", "发现Mamba架构在长距离依赖上局部拟合最优，支持记忆约束假设"]
benchmarks: ["STRUCTURALCOST", "Natural Stories", "UCL-SPR"]
---

# 论文速读：STRUCTURALCOST: A controlled reading time dataset for modeling human sentence processing difficulty

## 一句话总结
本文构建了 STRUCTURALCOST，一个包含 475 名参与者、40,800 个观测值的自步阅读数据集，通过系统性操纵主谓依赖长度并控制词汇混淆，隔离了长距离句法整合的加工成本；研究发现当前语言模型（n-gram/SSM/Transformer）虽在全局 surprisal-RT 拟合上表现稳健，但在动词整合位点系统性低估人类的记忆整合成本，揭示了现有评估范式的盲点。

## 研究问题与动机
1. **核心问题**：当前语言模型的 surprisal 能否真实模拟人类句子加工中的记忆整合成本，还是仅捕捉了预测成分？
2. **现有方法不足**：自然语料库（如 Natural Stories、UCL-SPR）虽规模大，但词汇频率、长度与句法结构混杂，无法分离依赖长度的独立效应；受控心理语言学实验虽能隔离结构变量，但样本量太小（通常 <100 参与者），不足以支持模型层面的全面评估。
3. **评估标准缺陷**：学界长期依赖全局 surprisal-RT 相关系数（r≈0.5）作为认知合理性的判据，但该指标被词汇 predictability 主导，对结构性难度不敏感。
4. **理论缺口**：依赖位置理论（Dependency Locality Theory）预言线性距离和句法嵌入深度会增加整合成本，但缺乏大规模受控数据验证该预言在现代 LM 中的体现程度。

## 核心贡献（创新点）
1. **发布 STRUCTURALCOST 大规模受控数据集**：450 句子、475 参与者、40,800 观测值，通过"关键句子 + 控制句子"配对设计系统性操纵依赖长度，首次以 NLP 规模复现并扩展了低统计功效的心理语言学发现。
2. **提出全局 vs. 局部对齐的区分框架**：证明全局拟合（所有词位）对架构和规模不敏感（r≈0.51–0.53），而局部拟合（动词整合位点）才能揭示模型对结构成本的敏感性，将评估标准从"预测精度"转向"整合成本编码"。
3. **发现跨架构的系统性盲点**：Transformer、SSM、n-gram 模型均在动词位点低估人类整合成本（预测效应比观测效应小 1–3 个数量级），且模型扩大无改善，指向架构层面的根本限制而非容量问题。
4. **揭示 Mamba 架构的局部优势**：MAMBA-130M 在长距离依赖（dep=9）上达到 r=0.41，是唯一置信区间不与 0 词依赖重叠的模型，支持"近期偏向记忆约束可部分模拟距离衰减"的假设。
5. **验证结构驱动难度的实证发现**：相同线性距离（4 词）下 PP < SRC < ORC 的梯度排序被复现，且双嵌入条件（2×SRC/2×ORC）出现意外的阅读加速反转，挑战了"嵌入深度单调增加成本"的朴素预期。

## 方法详解
1. **实验设计**：60 个句子集，每集 6 个关键句子共享同一主谓对，通过插入 SRC（主语关系从句）、ORC（宾语关系从句）、PP（介词短语）或嵌套双层结构操纵依赖长度（0/4/9 词）；另设 30 个控制句子集仅用 PP 操纵线性距离，用于分离线性距离与层级复杂度。
2. **刺激生成与审核**：使用 LLaMA-3.2 和 Perplexity.ai 生成初始句子，经人工审核确保语义合理、词汇重叠最大化、动词非 cloze-likely、时态一致（嵌入从句过去时、主句现在时），句子长度统一为 13–15 词。
3. **数据收集**：通过 Prolific 在线招募母语英语者，采用 jsPsych 实现的自步阅读范式，每词按键呈现；理解准确率 ≥70% 方可保留数据，最终 475 名参与者；完成后进行 operation span task 测量 WMC。
4. **统计分析**：log₂(RT) 为因变量，混合效应模型（lme4）含参与者、项目、呈现顺序随机截距；预测变量包括 word length、log frequency（WikiText-103）、dependency length；使用 ROPE 分析判断效应实用性（<0.1 SD = 35ms 视为可忽略）。
5. **模型评估框架**：Spearman 相关系数量化 surprisal 与 RT 的全局（所有词位）和局部（动词位点）对齐；嵌套混合模型 M0（基线）→M1（+surprisal）→M2（+dependency length）通过似然比检验量化各预测变量的独立贡献。
6. **转换因子验证**：在 Natural Stories 上估计 ms/bit 转换因子（回归当前词及前三词 surprisal + 溢出效应 + 词汇控制），应用于 STRUCTURALCOST 预测整合成本效应量，与观测 EOI 直接对比。

## 实验与结果
1. **人类行为模式**：动词处平均 RT 随依赖长度变化（0词：490ms → 4词：544ms → 9词：528ms）；ROPE 分析确认 0 vs 4 词和 0 vs 9 词差异显著，但 4 vs 9 词不显著。
2. **结构梯度效应**：相同 4 词距离下，PP（d=−0.08）< SRC（d=−0.15）< ORC（d=−0.19），证明句法嵌入独立于线性距离调制加工难度；双嵌入条件出现反向模式（2×SRC 504ms < SRC 539ms；2×ORC 553ms < ORC 583ms）。
3. **混合效应模型**：dependency length 每增 1 词，RT 增加约 3ms（β=0.0065, t=4.73, p<.001），即使控制 word length（β=0.012）和 log frequency（β=−0.009）。
4. **跨数据集对比**：在 Natural Stories 和 UCL-SPR 上拟合相同模型，dependency length 效应在动词位点反而衰减，而 STRUCTURALCOST 中增强，证明控制设计能有效隔离结构信号。
5. **全局对齐**：所有模型在全部词位上的 Spearman r 集中在 0.51–0.53，与架构/规模无关，χ²(1)>846 确认 surprisal 贡献显著。
6. **局部对齐**：动词位点上，神经模型 r 降至 0.31–0.39（dep=4），n-gram 模型降至 0.14–0.15；dep=9 时神经模型 r=0.26–0.35，且 SRC < ORC 对比弱化。
7. **规模无效应**：各架构内扩大参数未改善局部拟合，GPT-2-XL (r=0.34) < GPT-2-Small (r=0.31)；Mamba-130M 在 dep=9 时达 r=0.41 为最佳。
8. **预测 vs. 观测整合成本**：转换因子验证显示预测效应仅为观测的 1/10–1/1000（如 ORC 观测 93ms，KenLM 预测 1.99ms、Pythia-70M 预测 −1.42ms），无模型能恢复正确方向。

## 相关工作脉络
1. **Surprisal 作为认知代理的基础工作**：Smith & Levy (2013)、Goodkind & Bicknell (2018)、Futrell et al. (2021) 确立 surprisal-RT 相关性，本文指出这是弱诊断，需结合受控设计才能检验结构敏感性。
2. **架构约束改善心理度量拟合的研究**：Kuribayashi et al. (2022) 限制 Transformer 上下文恢复位置效应；De Varda & Marelli (2024)、Timkey & Linzen (2023) 引入归纳偏置；本文验证 Mamba 的选择性状态压缩在长距离依赖上优势，为这类干预提供评估基准。
3. **自然语料库的局限讨论**：Dundee、Provo、MECO、UCL-SPR、Natural Stories 等提供大规模数据但词汇-结构混杂；本文通过控制设计证明：只有正交操纵依赖长度才能分离结构成本，反驳"规模足以替代控制"的观点。
4. **反规模趋势的发现**：Oh et al. (2022, 2023)、Shain et al. (2024) 报告更大模型不一定改善心理度量拟合；本文在受控条件下确认该趋势，并解释为模型"超人类"预测能力稀释了 surprisal 的结构敏感性。
5. **工作记忆与句子加工理论**：Gibson (2001) 依赖位置理论、Lewis et al. (2006) 检索干扰模型；本文实证验证依赖长度效应，并发现双嵌入加速反转，提示现有理论需补充"句法适应"机制。
6. **受控数据集的不足**：Sent. Complexity (Grodner & Gibson, 2005)、Syn. Ambiguity (Huang et al., 2024) 控制良好但样本量小；本文通过 Prolific 在线平台实现 475 参与者，填补"控制+规模"的双重缺口。

## 局限性与未来方向
1. **生态效度有限**：句子为受控构造且仅限英语母语者，跨语言和跨人群推广受限；在线实验未控制阅读环境和设备。
2. **条件梯度不够密集**：缺乏 PP+SRC 组合以进一步分离线性距离与层级贡献；双嵌入反转机制未解，需更细粒度条件验证"句法适应"假设。
3. **WMC 数据未充分利用**：operation span 分数已收集但未在主流分析中建模，个体差异视角的挖掘是自然延伸。
4. **理解准确率粒度不足**：仅记录总体准确率，缺少逐题数据，无法进行 RT-accuracy 的精细联合分析。
5. **LM 架构覆盖不全**：仅评估 n-gram/Transformer/SSM 三类代表性模型，其他架构（如 RWKV、RetNet）或未公开模型的评估仍开放。
6. **理论机制待验证**：双嵌入加速可能源于 within-sentence syntactic adaptation，但缺乏直接证据；需结合 eye-tracking 或 EEG 验证认知过程。

## 研究启发与可借鉴点
1. **受控对照设计范式**："关键句子（结构操纵）+ 控制句子（仅线性距离）"的配对逻辑可有效分离混淆变量，可作为认知评估数据集构建的标准模板。
2. **局部对齐评估框架**：将模型评估从全局相关系数转向理论动机位置（动词整合位点），提供更严格的认知合理性测试；建议团队在未来的模型评估中纳入 local alignment 指标。
3. **ROPE 分析的统计严谨性**：采用实用等价范围（<0.1 SD）区分统计显著与实质显著，避免大样本下的假阳性；建议在团队论文中推广此做法。
4. **混合效应建模流程**：lme4 嵌套模型比较（M0→M1→M2）的流程可直接复用于后续研究，预测变量的独立贡献量化方法具有通用性。
5. **架构选择的启示**：Mamba 类 SSM 在长距离依赖上的局部优势提示，显式记忆约束（recency bias、容量限制）可能是提升认知合理性的有效路径，值得在团队研究中探索 architecturally-constrained LM 的设计。

## 关键术语表
**Surprisal**：信息论中的 surprise 度量，定义为 −log p(word|context)，作为语言模型预测加工难度的代理指标。
**Dependency length**：句法依存中头词与其从属词之间的线性距离（词数），是依赖位置理论的核心预测变量。
**Self-paced reading (SPR)**：被试逐词呈现阅读的任务范式，通过按键控制词显隐以记录反应时，测量在线加工成本的黄金标准。
**Global alignment**：surprisal 与 RT 在所有词位上的整体相关性，反映词汇 predictability 主导的预测精度。
**Local alignment**：surprisal 与 RT 在理论动机位置（如动词整合位点）的相关性，反映模型对结构加工成本的敏感性。
**Working memory capacity (WMC)**：工作记忆容量，通过 operation span task 测量，调节长距离依赖的加工难度和个体差异。
**ROPE (Region of Practical Equivalence)**：实用等价范围分析，将效应量小于 0.1 SD（本研究为 35ms）视为可忽略，区分统计显著与实质显著。
**Integrative cost**：整合成本，指工作记忆中将分离的句法成分（如主谓）合并时的认知负荷，是记忆约束理论的核心概念。

## 可复现要素
- **数据集**：STRUCTURALCOST（450 句子，475 参与者，40,800 观测值）；论文未明确提及公开状态，建议联系作者获取
- **代码**：jsPsych 实现 SPR 任务，lme4 进行混合效应建模；论文未提及代码仓库链接
- **模型**：KenLM n-gram（GPT-2/Pythia 词表）、GPT-2（small/large/XL）、Pythia（70M/1.4B/12B）、Mamba（130M/1.4B/2.8B），均从官方渠道获取
- **关键超参**：n-gram 窗口 = 5 tokens；子词聚合遵循 Pimentel & Meister (2024) 方法；RT 排除标准 100–3000ms；理解准确率阈值 ≥70%
- **依赖解析**：UDPipe + English Universal Dependencies 模型提取 dependency length
- **词汇控制**：log frequency 来自 WikiText-103，word length 为字符数
