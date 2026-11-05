---
title: "Same-Judgment-Diferent-Yes-Measuring-Multilingual-Judgment-P"
source: https://arxiv.org/pdf/2610.28066v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:23:38"
field: "多语言大模型评估与可解释性"
keywords: ["multilingual evaluation", "yes/no ambiguity", "negative interrogative", "moral judgment", "LLM benchmarking", "answer habit", "residual profile", "FAR-6 battery"]
innovations: ["FAR-6电池同时操控问句形式×回答约束×判读方式并做双重判读以分离判断与答题习惯", "跨语言consensus-removed残差画像在39/48细胞上实现无需分类器的模型身份识别", "定量分解s-b判断共享分与yes/no keying bias三种答题习惯"]
benchmarks: ["Moral Foundations Vignettes (Clifford et al., 2015)", "Claude/Sonnet/Opus 5", "GPT-6 luna/sol", "Gemini 3.1 Flash-Lite", "Qwen3.8-27B", "Mistral Medium 3.5"]
---

# 论文速读：Same-Judgment-Diferent-Yes-Measuring-Multilingual-Judgment-P

## 一句话总结
本文提出 FAR-6 实验电池，系统操控问句形式、回答约束与答案判读规则，在六语种八模型上采集 110 万条回复，证明语言模型的道德判断故事级画像跨语言具有可识别性（39/48 细胞），而 yes/no 词元在否定疑问句后的映射不可靠——同一判断可以对应不同的 yes。

## 研究问题与动机
- **核心问题**：多语言评估通常将模型回答简化为 yes/no 计数，但词元本身携带模型习惯与语种回答约定，如何区分"判断"与"答题习惯"？
- **否定疑问句的语言学歧义**：英语"Do you not agree?"的 yes/no 可对应不同命题（极性映射 vs. 真值/认同映射），德语 doch、法语 si 等专项反转词进一步使"是/否"不等于判断，既有评估未系统考察此类构造。
- **既有方法的盲点**：现有基准（Moore et al., 2024; Labat et al., 2026; Braun, 2025）仅固定问句形式与判读规则，未在同一批故事上交叉问句形式×回答约束×判读方式，也未对同一条回复做双重判读以分离映射效应与采样噪声。
- **待回答**：不同elicitation与readout方式在何时还原同一判断、何时测量的是答题习惯；模型跨语言是否保留独特的判断签名。

## 核心贡献（创新点）
1. **FAR-6 实验电池**：首次同时操控问句形式（普通极性/否定疑问/选择/或否）、回答约束（自由句/判决词首/是否词首/命题标签）与判读方式（词元固定规则/句子裁判），并在六语种上双表述平衡设计，与已有工作只固定一种问句+一种判读的本质区别在于可分离"判断"与"答题习惯"。
2. **判断画像 vs. 答题习惯的概念区分与测量框架**：定义判断画像（story-level 判断向量）与答题习惯（对词元/位置/术语的系统偏好），提出 s（判断共享分）与 b（keying 偏差）的运算分解，区别于以往仅报告单一 yes/no 比例的做法。
3. **双重判读（same-reply double reading）方法**：对同一条生成回复先用固定词元规则再用句子裁判独立判读，第一次把"映射差异"从"采样波动"中剥离出来，这是现有工作未做的设计。
4. **跨语言模型身份识别**：在去除每语种故事共识后，同一模型另一语种的残差画像在最近邻意义上胜出 39/48 细胞（随机期望 5），无需训练分类器，定位不同于 Sha et al. (2026) 的跨语种分类器路线。
5. **量化三种答题习惯**：yes/no keying bias（偏好 no，均值 −0.11）、second-listed preference（+0.05）、named-term pull（+0.08），揭示选项顺序与术语命名的独立效应，弥补此前研究仅关注顺序偏好的不足。

## 方法详解
- **FAR-6 电池结构**：4 种问句形式 × 4 种回答约束 × 2 种判读方式 × 6 语种 × 每故事两种表述（对立项命名 P 与 not-P 各一次），同一故事集重复生成 3 次以估计采样波动。
- **问句形式（Form）**：F1 普通极性问句（"Was what Finn did wrong?"）、F2 选择问句（"wrong or right?"）、F3 or-not 问句、F4 否定疑问句（"Wasn't what Finn did wrong?"）；F4 是核心对照项。
- **回答约束（Constraint）**：从宽松到严格依次为 free sentence、verdict-first（以评价词开头）、yes/no-first（以 yes/no 词开头）、proposition label（用一个字母代表完整命题，含两种字母顺序）。
- **判读方式（Reading）**：① fixed token/letter rule（R）——将开篇词元按预设映射读为 affirming/not-P；② sentence judge（Claude Sonnet 5）——忽略开篇词元，直接编码整句是否 assert P、assert not-P、conditional/both、none，仅前两类为 decisive。
- **判断与偏差的运算分解**：令 $q_p$、$q_{np}$ 为问题命名参考命题/其对立项时 assert P 的决定性回复比例，则 $s = (q_p + q_{np})/2$ 为判断共享分，$b = (q_p - q_{np})/2$ 为 keying 偏差；双重表述平均使不同 wording 可比。
- **画像比较三量度**：story-level Pearson r（整体趋势）、residual-profile r（去除每语种故事共识与每细胞均值后的相关）、mean $|\Delta|$（保留个体判断变化），均以 95% story-bootstrap 给出区间。
- **数据集**：85 条 Moral Foundations Vignettes（Clifford et al., 2015 英文 wrongness 均值作参考）+20 条自撰故事，英/中/德/日各 105 条，韩/法各 45 条；概念对（concept pair）每语种 4 组（日语 3 组），每对两项表述平衡。
- **模型**：Claude Haiku 4.5、Sonnet 5、Opus 5；GPT-6 luna、GPT-6.1 sol；Gemini 3.1 Flash-Lite；Qwen3.8-27B；Mistral Medium 3.5，共 48 个 model×language 细胞。
- **总回复数**：1,104,574，其中分析约束下 839,230 条；句子裁判三重校验（Claude Sonnet 5 low、GPT-6 astra high、Claude Fable 5 high）两两 agreement 95.5%/97.2%，native reader（中/英/德/日各 100 条）agreement 95.5%，test-retest κ=0.92/0.95。

## 实验与结果
- **收敛性（§4, Figure 2a）**：大多数 readout 对在故事级 r≥0.90 范围内收敛；自由句 vs. 命题标签 r=0.92、vs. 判决词句 r=0.93；否定疑问句自由句 vs. 普通自由句 r=0.94。唯一显著断裂：否定疑问句的 yes/no 词元（预设 R 映射）vs. 普通疑问句词元 r=0.73（[−0.16, 0.89]），mean $|\Delta|=0.22$。
- **同一回复双重判读（§4）**：71,282 条否定疑问 yes/no-first 回复，词元规则与句子裁判的决定性结果只在 median r=0.67 处相遇，mean $|\Delta|=0.25$；但这些被句子裁判解码为决定的句子的 story profile 与自由句 profile 高达 r=0.94（$|\Delta|=0.07$），说明变化在"开篇词元→命题"的映射而非故事判断本身。普通疑问句的同回复双判读 r=0.99，$|\Delta|=0.02$。
- **回复级分歧（Figure 2b）**：普通疑问 yes/no-first 双判读分歧 3.3%，否定疑问上升至 31.3%；三方高 reasoning audit 在否定疑问冲突中 93–96% 支持句子裁判，而在普通疑问冲突中仅约 50% 支持句子裁判（普通疑问分歧一半来自裁判本身）。
- **跨语言模型身份（§5, Figure 3/Table 6）**：命题标签 plain question，39/48 细胞的最近邻为同模型另一语种（uniform-random 期望 5）；free sentence 仅 15/48。同模型跨语言 story-level r=0.88 vs. 同语种不同模型 r=0.81（差 +0.075 [0.047, 0.096]）；残差 r 同模型均值 0.56。在可接受故事上差距最大（0.79 vs. 0.69），在违规故事上消失（0.43 vs. 0.46，全员谴责）。
- **三种答题习惯（§6, Table 4）**：yes/no keying bias 普通问句均值 −0.11（所有细胞偏好 no），交互项解释 0.56 方差；second-listed preference +0.05，named-term pull +0.08。否定问句下 second-listed 基本不变，named-term pull 趋零，词元 keying bias 在 19/48 细胞翻转符号。
- **否定疑问词元编码（§7, Figure 5/Table 5）**：极性映射 share 跨构造族差异巨大——negated attitude verb 中 English "Don't you think?" 0.99、Chinese "难道你不觉得…？" 0.96、Japanese 0.80；negated evaluative adjective 中 English "Wasn't it wrong?" 0.97、French 0.91、Chinese "…没有错吗" 0.21、German 0.58；negated evaluative verb 中 English "Do you not agree?" 0.22、French 0.78。不同模型（如日文 Sonnet/Opus/Qwen 偏极性、Haiku/luna/Mistral 偏真值/认同）呈现分化；同一词元在不同构造下映射到自身句子是由语言、模型、构造共同决定的，而非固定语言规则。

## 相关工作脉络
- **Moore et al. (2024)**：考察 LLM 在价值负载问题上的 consistency，变 paraphrase/topic/format/translation，但未交叉否定疑问与评价术语；本文在 question form × wording polarity × 回答约束维度上扩展。
- **Labat et al. (2026)**：在多语种多项选择上考察 option order/symbol/language，每题只读一次输出；本文对同回复做 token + sentence 双重判读。
- **Huang (2026)**：英语道德困境三工具 crossed design，分离 stance 与 order/label 效应；未涉及否定疑问句与多语种。
- **Sha et al. (2026)**：用分类器从 psychometric profile 识别跨语种模型身份；本文用 consensus-removed profile 的最近邻测试，不依赖训练分类器。
- **Braun (2025); Dominguez-Olmedo et al. (2024); Tjuatja et al. (2024)**：survey acquiescence/label/order bias，其中 negated-agreement 歧义被指出但未在 LLM 评测中解构；本文将其量化为 keying bias。
- **Okada et al. (2026)**：desirability-matched forced-choice 缓解 socially desirable responding，仅限英语人格条目；本文在道德 vignette 上同时处理 keying/position/term 三种习惯。
- **Hua et al. (2025)**：指出英语 benchmark 上的 prompt sensitivity 多来自 rule-based scoring 而非生成文本；本文用 sentence judge 替代纯规则判读，并对比两者差异。

## 局限性与未来方向
- **画像级收敛的粒度局限**：同模型跨语言残差结构在不同模型/语对间变异大；命题标签下模型签名比自由句更锐利（自由句 81% 单元格得分位于端点，只剩较少 graded 结构）。
- **覆盖率与抽样局限**：韩/法仅 45 条故事；undecided sentences（both sides/conditional/no position）不计入 decisive share；韩/法无 native reader 标签，仅 100 条/语种的英/中/德/日有。
- **句子裁判的参考性质**：sentence judge 为一个模型（Claude Sonnet 5），虽经三重校验，但与 human 裁判共享 labeling bias 的可能性未排除；普通疑问下 token vs. 句子冲突中强裁判约半数支持 token，说明判决本身非 ground truth。
- **模型与语种固定**：interval 仅在故事层面 bootstrap，未覆盖模型/语种总体；结论仅适用于 FAR-6 所代表的形式、约束、规则与表述集合。
- **人类对照缺失**：英文 MFV wrongness 均值作为跨语种统一参考；对话语料用于背景锚定但 register/context 与 prompt 不同；"同故事的人类匹配研究"作者明确列为 future work。
- **解释性边界**：s 与 b 是 operational decomposition 而非 pure judgment/pure bias；cross-语境的 identifiability 是被测画像的属性，不证明存在 language-free 的内部状态。
- **未来方向**：加入人类被试在同故事上的匹配对照；扩展至更多语种/模型；将共识移除最近邻方法用于其他心理测量域；探索否定疑问句的模型内学习规律以改进判读规则。

## 研究启发与可借鉴点
- **双重判读分离信号与设计偏差**：对同一条回复分别用规则解码与裁判判读，可将"映射不确定"从"生成不稳定"中剥离——此设计可迁移到任何依赖首词/选项解码的多语种评测。
- **双向表述平衡分离 keying 与判断**：对每个概念对同时提问 P 与 not-P 并取平均，把 wording/keying bias（b）从共享分（s）中剥离，可直接移植到价值/伦理基准构建中。
- **consensus-removed 残差画像用于模型身份指纹**：去语种故事共识 + 去细胞均值后做最近邻识别，无需训练分类器即可在 39/48 细胞上确认模型跨语言稳定性，可为跨语言对齐评估提供零样本指纹方法。
- **三种习惯的独立量化框架**：keying bias / second-listed preference / named-term pull 分别刻画否定倾向、位置倾向与术语回声，三者可正交评估；后续工作可加入"refusal rate""hedge rate"扩展为四种习惯。
- **s-b 分解的通用性**：任何 dichotomous 或多项选择答案均可按 $s=(q_P+q_{\neg P})/2$、$b=(q_P-q_{\neg P})/2$ 运算，得到可加性更清晰的评测分数，优于单 wording 比例。

## 关键术语表
- **Judgment profile**：单一模型在单一语种、单一判读下对全部故事的判断向量，代表该模型在该配置下的"道德立场画像"。
- **Answer habit**：模型在答题形式上的系统性偏好（如偏好 no、偏好第二位选项、偏好问题命名的术语），独立于故事判断本身。
- **FAR-6**：本论文的六因素实验电池（Form × Answer constraint × Reading × language × wording × option order），采集 110 万条多语种回复。
- **Yes/no keying bias（b）**：两种对立表述下 assert P 比例之差的一半；负值表示偏好 no，正值表示偏好 yes。
- **Residual-profile r**：去除每语种故事共识与每细胞均值后，两画像之间的 Pearson 相关，衡量模型特有的偏离共识的结构。
- **Polarity mapping**：将 yes/no 映射为"肯定/否定问题中命名的术语"的规则；与 truth/agreement mapping（yes 确认否定命题）对立，两者在否定疑问句下产生不同解码。
- **Sentence judge**：使用 Claude Sonnet 5 忽略开篇词元、直接编码整句是否 assert P/not-P/conditional/none 的判读机制，作为固定规则的对照。
- **Concept pair**：每语种一对对立评价词（如 wrong/right、oppose/support），两种表述分别命名其中一项，用于平衡 keying 效应。

## 可复现要素
- **数据集**：85 Moral Foundations Vignettes（Clifford et al., 2015，英文 wrongness 均值公开）+20 条自撰故事，英/中/德/日 105 条，韩/法 45 条；共 1,104,574 条回复。论文声明包含 prompts 与 setting 附录，代码/数据未在本摘要中明确说明开源仓库，但 arXiv 论文本身含完整附录与 Table；建议查阅论文附录 A 与附属仓库链接。
- **模型与版本**：Claude Haiku 4.5、Sonnet 5、Opus 5；GPT-6 luna、GPT-6.1 sol；Gemini 3.1 Flash-Lite；Qwen3.8-27B；Mistral Medium 3.5（论文未提及 temperature/top-p 等超参的具体数值，仅注明 Haiku/Sonnet/Opus 用 non-thinking，luna 无 reasoning、sol 低 reasoning、Gemini/Mistral 默认、Qwen 禁用 thinking）。
- **句子裁判**：主裁判 Claude Sonnet 5 low adaptive reasoning；交叉校验 GPT-6 astra high、Claude Fable 5 high；重复运行 test-retest 两次各 300 条。
- **统计方法**：95% story-bootstrap（seed fixed），同 reprompt 3 次内先取 item 内均值再跨 order/keying/concept pair 加权；nearest-neighbor 从 47 个候选细胞中选取，无平局。
