---
title: "What-Breaks-When-Facts-Become-Stories-The-Distortions-Source"
source: https://arxiv.org/pdf/2610.39500v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:15:31"
field: "语言模型事实性与叙事评估"
keywords: ["factuality evaluation", "narrative distortion", "accountability bias", "LLM-as-judge", "news retelling", "ReTold-2k"]
innovations: ["提出检测+合法性双层判定，区分视角合理偏离与误导性扭曲", "发现任务自由度驱动问责扭曲，故事重写中施害者责任被系统性减轻", "开源 ReTold-2k 并保留未聚合逐评审争议标签"]
benchmarks: ["ReTold-2k", "NLI contradiction audit"]
---

# 论文速读：What-Breaks-When-Facts-Become-Stories-The-Distortions-Source-Agreement-Metrics-Miss-When-Models-Retell-Real-Events

## 一句话总结
论文系统诊断了"将新闻事实重述为故事"时语言模型产生的**问责扭曲（accountability distortion）**——施害者责任被系统性减轻，并指出当前基于源文本一致性的评估指标（含事实核查与语法测量）均无法捕获此类扭曲。为此作者发布可复现的标注基准 **ReTold-2k**。

## 研究问题与动机
- **评估目标错位**：现有度量把"与源文本一致"当作成功标准，但在重构（reconstruction）任务中，脱离源文本本身就是目标；直接套用会误杀合理偏离，不放任又会让真实扭曲漏过。
- **叙事视角与责任归属的关联**：新闻事件的因果链、施害者位置与受害者位置被叙事框架（framing）理论长期认为是责任归因的关键变量；模型重述是否会系统性转移责任尚待量化。
- **现有指标盲区**：句法/语法层指标甚至能给出故事"比原文更干净"的分数，指控级 NLI 事实核查也能通过，但问责维度却被完全忽略。
- **判定边界本身具有争议性**："偏离是否由视角合理允许"不是简单标注问题，作者希望把这种争议作为信号而非噪声保留下来。

## 核心贡献（创新点）
- **提出"合理偏离 vs. 误导性扭曲"的可判定边界**：以叙述者视角允许的焦点化（focalization）为界，将六类变化区分为 detect 与 warrant 两层判断；与已有工作相比，以往指标只看一致性，本文进一步引入视角合法性判定。
- **发现任务自由度驱动问责扭曲**：Vanishing Actor 与 Causal Fading 在摘要中几乎缺席，在故事里上升 6–7 倍；相同文章同模型下对比揭示出任务约束而非事件严重程度的主效应。
- **证明施害者责任被系统性减轻**：在方向可判定的问责扭曲中，约 2/3（67.4%）减轻施害者责任；该偏斜跨越多类叙述者，并非单一视角所致。
- **开源 ReTold-2k 资源**：提供判定标准、测量流程、三族模型的逐评审 Baseline 与争议案例集，并**保留未聚合的逐评审标签**，供后续检测器对齐评估。
- **展示指标失效的机制路径**：通过规则提取与 judged 结果交叉验证，说明"语法更干净 + 指控无矛盾"并不能排除问责失真。

## 方法详解
- **数据与生成**：2,000 篇 2014–2026 年英语新闻，按 7 个地区 × 4 级话题严重程度分层。三族模型（GPT-5.4 mini、Gemini 3.1 Flash-Lite preview、Claude Haiku 4.5）将每篇改写为：
  - **摘要**：压缩比 0.55，无叙述者视角；
  - **故事**：分配 12 种叙述者之一，约 200 词，从指定视角叙述。
- **六类扭曲定义**（Table 1）：
  - **Vanishing Actor**：有害行为失去施害者；
  - **Role Reversal**：施害者与受害者位置互换；
  - **Causal Fading**：因果链被切断；
  - **Claim Laundering**：归因性声明被改写为事实陈述；
  - **Selective Omission**：超出篇幅必要的实质性内容被删；
  - **Invented Addition**：来源中不存在的事实或评价被加入。
- **双层判定流程**：先 detect 是否发生某类扭曲；再判定（legitimacy）该偏离是否由 assigned perspective 所允许（warranted/unwarranted）。
- **留一评审协议（leave-one-generator-out）**：每个输出由**未生成它的另外两族模型**做评判，温度 0，避免自夸偏差；共 71,718 次摘要评审、23,094 次故事评审。
- **规则提取器（rule-based indicators）**：基于 spaCy，仅依赖句法解析与预定义模式，覆盖 when/where/how 等轴（被动/无施事、因果密度、归因密度、绝对时间、工具、地点实体），用于交叉验证。
- **争议保留设计**：不取多数投票 gold label，逐评审记录 detection、warrant、target、direction、valence 与原文引证，形成 911 个争议样本。

## 实验与结果
- **数据集规模**：2,000 篇文章， story 产出 1,926 份（74 次拒绝生成）；配对可比文章 1,923 篇。
- **问责扭曲随任务自由度剧增**（Table 11）：
  - Vanishing Actor：Summary 1.4% → Story 8.3%；
  - Causal Fading：Summary 2.4% → Story 16.9%；
  - Role Reversal：Summary 0.0% → Story 1.2%。
  - 文章层面仅在 story 中出现差异的样本 661 篇中，587 篇仅在 story 中被检测出（McNemar p < 10⁻⁹⁹），四档严重性均显著（p < 10⁻¹⁵ ~ 10⁻³³）。
- **施害者责任减轻比例**：67.4%（图 3）；Neutral Observer、Rescuer、Perpetrator 等叙述者均更 lenient，仅 Bereaved family 显著 amplify（Table 10）。
- **事实核查盲区**：4 档严重性中 1,445 篇 story 有 NLI 审计，562 篇（约 38.9%）**零矛盾**；其中 34.0% 仍携带检测到的问责扭曲，27 篇获双评审一致。
- **语法指标反向**：故事被动与无施事被动比例均低于原文（Table 2），故事被"更透明"，却掩盖 Vanishing Actor 升高 6 倍的现象（Table 4 示例）。
- **评审争议边界**：涉及施害者的问责扭曲对评审分歧率 65.9%，非涉及者仅 2.7%（Table 5）；三人人工重标 50 例，κ = −0.07（legitimacy 分裂），体现边界争议性而非标注噪声。

## 相关工作脉络
- **Papineni 2002 / Lin 2004 / Pagnoni 2021**：BLEU/ROUGE/FRANK 等以重叠或事实性为主的评价，本文指其无法刻画"视角合理的偏离"。
- **Plank 2022 / Sorensen 2024**：标签变异与多元对齐观点；本文把争议当信号、保留未聚合评审。
- **Entman 1993 / Iyengar 1991 / Fowler 1979**：框架与责任归因传播学基础；本文将其操作化为可机器判定的六类扭曲。
- **Zheng 2023 (MT-Bench / Chatbot Arena)**：LLM-as-judge 自夸偏差；本文用 leave-one-generator-out 缓解。
- **Hamborg 2019 / Genette 1980**：5W1H 事件分解与焦点化叙事理论，构成本文轴分解与视角定义来源。

## 局限性与未来方向
- **语言与体裁局限**：仅考察英语新闻，其他语言、小说化或非新闻体裁的扭曲模式未知。
- **规则提取精度受限**：锚点统计基于字符串匹配，难以覆盖 paraphrase，密度指标可能低估保留。
- **评审严格度差异大**：三族模型与三人作者的 κ 极低，warrant 边界高度争议，难以收敛为单一 gold。
- **叙事者与文章效应未解耦**：每篇文章仅分配一个叙述者，二者效应混叠，难以分离"谁在轻纵"。
- **未来方向**：开发对问责扭曲敏感的自动化检测器；研究指令/任务自由度与施害者责任的定量关系；拓展至多语言与跨体裁；探索对齐策略对 warden 边界的可学习化。

## 研究启发与可借鉴点
- **双层判定架构（detection + legitimacy）**：把"是否变化"与"是否由视角允许"分离，值得迁移到任何基于视角的生成任务评测。
- **留一评审协议**：避免 judge 模型自夸，可在 LLM 评估流水线中常态化使用。
- **争议即信号**：不聚合为多数票 gold，而是保留逐评审标签，便于后续测量"边界难度"本身。
- **规则提取 × 模型判断交叉验证**：对同一批文本并行跑句法规则与 judge，方向一致才强化结论，能削弱单一评测器的盲点。
- **叙事角色矩阵设计**：12 种叙述者按与事件关系分组（perpetrator/victim/observer/professional），可作为后续人物视角研究的模板。

## 关键术语表
- **Accountability distortion**：改变"谁对谁做了什么"的扭曲类型，本文重点对象。
- **Vanishing Actor**：施害者在重述中被隐去或替换为非人主语，导致责任主体消失。
- **Causal Fading**：因果连接词或链条被弱化/切断，使后果显得无源。
- **Claim Laundering**：把"某人称/据报道"的归因句改写为客观事实句。
- **Warranted / Unwarranted**：区分偏离是否由 assigned perspective 所允许的两层判定。
- **Leave-one-generator-out**：由未生成该文本的模型担任 judge 的协议。
- **ReTold-2k**：本文开源的可复现语料与标注资源（含逐评审、争议集与提取器）。
- **Focalization**：叙事学中"谁看/知/感"的限制，本文据以界定合理偏离的边界。

## 可复现要素
- **数据集**：ReTold-2k（https://github.com/Retold2K/retold2k）；文章索引与叙事者分配规则公开，原文因版权未随包提供。
- **代码/权重**：生成使用商业 API；judging prompt、抽取脚本、聚合脚本已公开。
- **关键超参**：摘要压缩比 0.55；故事约 200 词；judge 温度 0；12 种叙述者按模型家族轮换平衡。
- **模型家族**：GPT-5.4 mini（OpenAI）、Gemini 3.1 Flash-Lite preview（Google DeepMind）、Claude Haiku 4.5（Anthropic）。
- **统计检验**：McNemar 精确双侧检验、χ² 检验，p 值均远小于 0.001。
