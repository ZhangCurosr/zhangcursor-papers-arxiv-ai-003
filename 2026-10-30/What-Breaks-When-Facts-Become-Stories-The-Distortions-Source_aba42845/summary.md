---
title: "What-Breaks-When-Facts-Become-Stories-The-Distortions-Source"
source: https://arxiv.org/pdf/2610.39500v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:15:03"
field: "自然语言生成的事实性与评估"
keywords: ["factuality evaluation", "accountability distortion", "narrative perspective", "LLM judging", "news reconstruction"]
innovations: ["提出warranted/unwarranted区分框架以识别叙事重建中的问责扭曲", "发现accountability distortion在stories中升高6-7倍且67.4%偏向减轻肇事者责任", "发布ReTold-2k数据集及未聚合judgments以支持后续检测器研究"]
benchmarks: ["ReTold-2k", "FRANK"]
---

# 论文速读：What-Breaks-When-Facts-Become-Stories-The-Distortions-Source-Agreement-Metrics-Miss-When-Models-Retell-Real-Events

## 一句话总结
论文揭示了当语言模型将新闻事实重写为故事时，现有基于"源文本一致性"的评估方法会遗漏一种关键的"问责扭曲"(accountability distortion)——这种扭曲会显著减轻肇事者责任（6-7倍增长，67.4%偏向减轻责任），且能逃过claim-level事实核查和语法指标的检测。

## 研究问题与动机
- **核心问题**：当前评估方法将"与源文本的一致性"作为成功标准，但对于重建任务（reconstruction）而言，偏离源文本可能是有意的（如叙事视角转换），这导致两种错误：要么将合理偏离计为错误，要么放任真实扭曲通过。
- **现有方法不足**：BLEU、ROUGE等传统指标仅衡量表面重叠；FRANK等factuality基准也仅检测contradiction，无法捕捉"删除行动者""弱化因果关系"等隐性扭曲。
- **现实关切**：新闻事件报道中责任归属至关重要，叙事视角的转变会改变读者对责任归属的认知（Entman, 1993; Iyengar, 1991），但现有自动评估对此完全盲区。
- **方法论缺口**：缺乏一个能区分"叙事视角允许的偏离"与"误导性扭曲"的可操作判定标准。

## 核心贡献（创新点）
1. **提出" warranted departure vs. misinformation"的区分框架**：定义了十二种叙事视角（perpetrator/victim/observer等），将变化分为"focalization允许的范围"与需合法性判定的范围，此前无工作系统处理这一边界。
2. **发现并量化"问责扭曲"(accountability distortion)**：定义六种扭曲类型（Vanishing Actor, Role Reversal, Causal Fading, Claim Laundering, Selective Omission, Invented Addition），揭示其在stories中较summaries升高6-7倍，且67.4%偏向减轻肇事者责任。
3. **证明现有评估指标对问责扭曲存在系统性盲区**：语法指标反而显示故事比原文"更干净"（更少被动语态），562/1445的故事无contradiction，但其中34%仍含问责扭曲。
4. **发布ReTold-2k数据集与评测协议**：包含2000篇新闻、1926个story输出、71718个judgment（未聚合），以及911个争议案例，支持后续检测器开发。

## 方法详解
**数据构建**：
- 收集2000篇2014-2026年英文新闻，按7个地区、4级话题严重度（neutral/low/medium/high）分层采样。
- 每个article分配一个narrator（12种角色：victim, perpetrator, witness, neutral observer等），由GPT-5.4 mini/Gemini 3.1 Flash-Lite/Claude Haiku 4.5生成summary（压缩比0.55）和story（约200词）。

**Judging框架**：
- 采用leave-one-generator-out协议：每个output由未生成它的两个model familyjudge，temperature=0。
- 六类扭曲的判定条件（Table 1）：
  - Vanishing Actor：伤害行为的agent必须保留
  - Role Reversal：perpetrator与victim位置不变
  - Causal Fading：因果链不断裂
  - Claim Laundering：争议主张必须保留attribution
  - Selective Omission：who did what to whom不得被实质性删除
  - Invented Addition：不得引入源文本外的信息
- 合法性判定：区分"warranted"（由narrator视角允许）与"unwarranted"（视为扭曲）。

**Rule-based指标**：
- 使用spaCy进行句法解析，提取anchor：被动语态/无agent被动、时间表达、地点实体、工具、因果连接词、归因提示。
- 计算密度/保留率/新颖率，不依赖模型判断。

## 实验与结果
**数据集**：
- 2000篇新闻，1926个有效story输出（74个被model拒绝），1923个paired articles。
- 71,718个summary judgments + 23,094个story judgments。

**主要结果**：
- **Table 3（关键）**：Story中accountability distortion显著高于Summary，且随severity上升（Neutral 23.5% → High 40.0%，both judges）；Summary中仅6.0%-9.5%。
- **Table 11（详细类型）**：Vanishing Actor从1.4%升至8.3%（~6倍），Causal Fading从2.4%升至16.9%（~7倍），Role Reversal从无到1.2%。
- **Figure 3（方向）**：67.4%的问责扭曲减轻肇事者责任，Neutral Observer最宽容（mitigate rate 83%），Bereaved family唯一显著amplify（38% mitigate）。
- **Fact-checking盲区**：562/1445故事无contradiction，其中34%含accountability distortion，27个经两个judge一致确认。
- **Judger分歧**：涉及perpetrator的accountability distortion分歧率65.9%，不涉及时仅2.7%；人类作者（3人）在legitimacy上的Fleiss' κ=-0.07（低于chance）。

## 相关工作脉络
1. **Papineni et al. (2002) BLEU / Lin (2004) ROUGE**：基于n-gram重叠的surface指标，本文证明其完全无法检测semantic-level distortion。
2. **Pagnoni et al. (2021) FRANK**：abstractive summarization的factuality benchmark，但仅关注contradiction和entailment，未覆盖"删agent"类扭曲。
3. **Plank (2022) / Sorensen et al. (2024)**：关于human label variation和pluralistic alignment的讨论，本文呼应其"aggregation会抹除争议信号"的观点，选择不聚合judgments。
4. **Entman (1993) / Iyengar (1991)**：framing理论，指出叙事方式影响责任归属认知，本文为此提供empirical evidence and automated detection method。
5. **Hamborg et al. (2019) Giveme5W1H**：five Ws提取系统，本文借鉴其event decomposition思路但扩展至六类distortion类型。
6. **Zheng et al. (2023) MT-Bench/Chatbot Arena**：LLM-as-judge评估，本文采用leave-one-generator-out避免model self-favoritism bias。

## 局限性与未来方向
- **规则指标的局限**：spaCy-based extraction无法捕捉paraphrase，仅做string matching；retention/novelty指标可能低估语义层面的保留。
- **Judgment的subjectivity**：warranted/unwarranted边界本身是contested的（κ=-0.07），不同judge/annotator的strictness差异大，难以建立gold standard。
- **Model family覆盖有限**：仅使用三个family（GPT/Gemini/Claude），未explore open-weight models；refusal集中在Claude（72/74）。
- **Language constraint**：仅限English news，区域差异不显著但generalizability待验证。
- **未来方向**：开发自动化accountability distortion detector；研究narrator视角与distortion方向的关系；探索减轻perpetrator-biased扭曲的generation约束。

## 研究启发与可借鉴点
1. **Leave-one-generator-out judging protocol**：有效避免model self-favoritism bias，可复用于任何generation-evaluation场景。
2. **六类distortion taxonomy**：从five Ws扩展至accountability维度，为factuality评估提供结构化分类框架。
3. **Unaggregated judgment release策略**：承认ground truth的contested性，保留分歧信息而非强行consensus，对annotation-heavy任务有参考价值。
4. **Rule-based + Judge双层验证**：先用无偏rule指标捕捉surface变化，再用judge评估legitimacy，避免单一方法的盲点。
5. **Narrator role作为controlled variable**：通过系统性操纵叙事视角，分离"intended departure"与"misinformation"，为causal inference提供实验设计思路。

## 关键术语表
- **Accountability distortion**：改变"谁对谁做了什么"的扭曲类型，包括删agent、弱化因果、反转角色等，是本文核心发现。
- **Warranted vs. Unwarranted**：前者指由narrator视角合理允许的偏离（如victim不知道的事件细节），后者为误导性扭曲。
- **ReTold-2k**：论文发布的数据集，包含2000篇新闻、1926个story输出、未聚合的judgments及911个争议案例。
- **Vanishing Actor**：六类distortion之一，指删除伤害行为的明确agent（如"the army burned"→"the village burned"）。
- **Causal Fading**：切断因果链（如"Died from the shelling"→"there was shelling and deaths"）。
- **Claim Laundering**：将带attribution的主张改写为无来源的plain fact（如"Police said"→直接陈述）。
- **Leave-one-generator-out**：每个output由未参与生成的其他model family进行judging，避免self-bias。
- **Fleiss' κ**：校正chance的inter-rater agreement度量，本文用于量化human annotator在legitimacy判断上的分歧。

## 可复现要素
- **数据集**：ReTold-2k，部分公开（GitHub: https://github.com/Retold2K/retold2k），包含article index、narrator assignment rule、extractors、per-judge baselines、911个contested judgments；article text因copyright未公开。
- **代码/权重**：Judging prompt、extraction脚本、aggregation script开源；模型权重未提供（使用commercial APIs: GPT-5.4 mini, Gemini 3.1 Flash-Lite, Claude Haiku 4.5）。
- **关键超参**：Compression ratio=0.55（summary）；Story length≈200 words；Temperature=0（judging）；Seed=42（shuffle）。
