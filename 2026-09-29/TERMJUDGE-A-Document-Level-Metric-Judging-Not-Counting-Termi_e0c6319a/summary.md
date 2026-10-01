---
title: "TERMJUDGE-A-Document-Level-Metric-Judging-Not-Counting-Termi"
source: https://arxiv.org/pdf/2609.35017v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:12:54"
field: "机器翻译评估"
keywords: ["machine translation evaluation", "terminology metric", "LLM-as-a-judge", "document-level evaluation", "terminological variation", "BioMQM"]
innovations: ["提出 TERMJUDGE 文档级术语指标，通过 Q1/Q2 两阶段 LLM judge 区分术语错误与合法变体", "构建 BioMQM-Terms 评测资源，从 MQM 标注中提取术语专用基准", "用可解释 verdict 仲裁 glossary injection 效果的此前未解 tension"]
benchmarks: ["BioMQM-Terms", "STEP", "PARANLP", "IWSLT23"]
---

# 论文速读：TERMJUDGE-A-Document-Level-Metric-Judging-Not-Counting-Termi

## 一句话总结
论文提出 TERMJUDGE，一种文档级术语评测指标，通过 LLM-as-a-judge 的两步判决流程（Q1 错误分类 + Q2 一致性判定）区分术语翻译错误与合法变体，在 BioMQM-Terms 和 STEP 数据集上均取得最优元评测结果。

## 研究问题与动机
- 现有术语评测指标（ glossary conformity、一致性指标等）仅通过与固定参考形式匹配来判定正确性，无法区分翻译错误与人类译员 routinely 产生的合法术语变体（如缩写首次出现、同义替换、语域调整）。
- 通用神经指标（ COMET、MetricX 等）在专业领域术语错误集中时与人工判断的相关性显著下降，且输出单一标量分数，无法提供可解释的判决依据。
- 一致性指标假设"同一概念所有出现必须采用相同译法"，但该假设仅适用于 prescriptive terminologies，忽略了专业文本中 denominative variation 的 constitutive property。
- 缺乏能够在文档上下文中小粒度判定每个术语出现是否 acceptable 的自动指标。

## 核心贡献（创新点）
1. **提出 TERMJUDGE 文档级术语评测指标**：基于 LLM-as-a-judge 架构，通过 reference validation → error classification → document-consistency qualification 三步流程，输出可解释判决而非单一分数。（与现有指标的本质区别：区分错误与合法变体，而非简单匹配计数。）
2. **构建 BioMQM-Terms 评测资源**：从 bio-MQM 语料中提取 729 个生物医学概念，构建包含 13,200 个 term–translation 对齐对的评估基准，并附 bilingual glossary。（与现有 MQM 资源的本质区别：首次将 MQM 标注扩展为术语专用评测集。）
3. **系统性元评测验证**：在 STEP（段落级人工标注）和 BioMQM-Terms（文档级 MQM 分数）上验证，TERMJUDGE 在 SPA 和 $\operatorname { acc } _ { \mathrm { eq } } ^ { * }$ 指标上均排名第一，显著优于 ten 种 baseline。（与现有工作的本质区别：首次在同一任务上同时验证 occurrence-level 和 document-level 表现。）
4. **重新评估 glossary injection 效果**：在 PARANLP 和 IWSLT23 上验证发现，glossary-guided prompting 通过消除 genuine errors 而非压制 valid variation 提升术语翻译质量。（与之前工作的本质区别：用可解释判决解决此前无法仲裁的 tension。）

## 方法详解
**预处理阶段（§3.1）**：
- 输入：SKOS glossary + 源文档 + 翻译输出（无需人工参考译文）
- CONCORDANCER 工具检测术语出现，识别 glossary 变体及 context-derived 变体（ lemma matching 会遗漏约 15%）
- 每出现按 variation category 标注（no variation/graphical/morphosyntactic/reduction/expansion/lexical/combined）
- LLM 调用对齐源–目标 span（按 aligned segment 而非全文，temperature=0，≤20 tokens）

**五步评估管线（§3.2，Figure 1）**：
1. **Reference selection**：对每个源形式 s，确定期望参考 R(s)——glossary 条目优先；无 glossary 时取系统自身最高频翻译作为 pseudo-reference，再由 error judge 验证；若被判为错误，则由 LLM 从 source-side context 生成（不从系统输出生成）
2. **Divergence detection**：对 lemma 做 exact string matching（忽略大小写、冠词、连字符、重音），匹配则标记 CONFORMING（无需 LLM）；不匹配则为 divergence
3. **Q1 错误判决**：error judge 接收 aligned pair + 当前 segment + 前 5 个 aligned segments + glossary entry + prior renderings，判断是否为错误及类型（13 类标签：A1–A7 细粒度 + C–H 聚合类 + no_error）。A 类保留 7 个细码，C–H 每族取最宽松 sub-code（benefit of the doubt），B 族（文档内不一致）排除交由 Q2
4. **Q2 一致性判决**：consistency judge 对 Q1 判 for no_error 的 divergence 判断是否与文档其他翻译一致。10 个标签：7 个 coherent 标签（introduction/explicitation/avoid repetition/document usage/register/facet/synonym merge）→ JUSTIFIED；3 个 inconsistency 标签（neutralisation/synonym inconsistency/gratuitous divergence）→ UNJUSTIFIED（penalty=0.5）
5. **Propagation and scoring**：Q1 按 divergent translation form 去重调用（传播至相同形式的所有出现），Q2 按 occurrence 调用；每个 verdict 映射 penalty（Table 1），文档分数为 mean penalty（lower is better）

**关键设计**：
- Hapax（单出现形式）跳过 Step 2 匹配直接进 Q1，无 Q2（一致性无从定义）
- Q1 传播机制使 LLM 调用数减少 2.55×（IWSLT23 上）
- Judge 模型：gpt-4.1-mini（低成本 + 避免 self-preference bias）
- 判决输出 JSON：{"justification": "...", "label": "..."}，temperature=0

## 实验与结果
**数据集（Table 2）**：
- STEP：地学领域，10 篇文档，3,377 个人工标注出现，17,035 条目 glossary
- BioMQM-Terms：生物医学领域，50 文档，1,320 对齐出现/系统 × 10 系统 = 13,200 对，729 条目 glossary
- PARANLP / IWSLT23：NLP 领域，分别 34/10 文档，8 个系统输出（4 LLM × 2 prompting 条件）

**段级验证（STEP，Table 3）**：
- Binary error detection：balanced accuracy 0.728，precision 0.632，recall 0.609，F1 0.620
- 15-class macro-F1：0.196（细粒度分类仍困难）
- Hapax vs non-hapax：accuracy 0.664 vs 0.787（差 12 点）
- 主要混淆：A3 invented calque 被接受为正确（40/65），家族 F（语法）和 B（纯不一致）漏检多

**文档级元评测（BioMQM-Terms，Table 4）**：
- TERMJUDGE：SPA **0.844**，$\operatorname { acc } _ { \mathrm { eq } } ^ { * }$(seg) **0.606**，均排名第一
- 显著优于所有 baseline（paired permutation test $p < 0.05$）
- Divergence-QE 第二梯队（SPA 0.769–0.803）
- 排除 hapax 后（Table 18–20）SPA 差距缩小但不改变排名结论

**系统排名（PARANLP/IWSLT23，Table 6–7）**：
- base+terms 条件下所有 4 模型均优于 baseline
- TERMJUDGE 显示：conformity +6–12 点，A 类错误率从 11.7%→8.4%（PARANLP）、18.6%→9.3%（IWSLT23）
- Glossary conformity 指标高估 base+terms 效果（仅计数 injected terms）
- CometKiwi_gen 对术语提升几乎无感知

**消融（§5.4）**：
- General QE (SPA 0.729–0.736) → Divergence 架构 (0.769–0.803) → Q1+Q2 判决 (0.844)
- 判决步骤是性能跃升的关键

## 相关工作脉络
1. **Farajian et al. (2018) / TermEval (Haque et al., 2023)**：基于 term-annotated reference 的 term hit rate 计算，依赖预定义形式清单，无法处理 unlisted variation。（TERMJUDGE 的定位：不再依赖固定 reference，改用 LLM 判决。）
2. **WMT terminology shared tasks (ibn Alam et al., 2021; Semenov et al., 2023, 2025)**：glossary-based 匹配指标，从 exact matching 到 lemma-aware，2025 扩展到 one-to-many dictionaries。（区别：仍是 counting 而非 judging，未区分错误与合法变体。）
3. **Semenov & Bojar (2022) / Consistency metrics**：通过 pseudo-reference（dominant translation）评估一致性，使用 Herfindahl–Hirschman index 或 LXR。（区别：假设"一致即正确"，忽略 motivated alternation。）
4. **GEMBA / GEMBA-MQM (Kocmi & Federmann, 2023)**：LLM-as-a-judge 通用质量评测，标注 MQM 错误 span。（区别：无术语追踪能力，不跨文档一致性判定。）
5. **COMET / MetricX (Rei et al., 2020; Juraska et al., 2024)**：fine-tuned neural QE 指标，sentence-level 训练。（区别：在未见专业领域相关性显著下降，且输出不可解释的单一分数。）
6. **Dahan et al. (2026b)**：前作提出 divergence-based 指标，但未解决"counting vs judging"问题，发现 glossary injection 的 tension 无法仲裁。（区别：本文在前作 pipeline 基础上引入 LLM judge，解决该 tension。）

## 局限性与未来方向
- **Judge 模型依赖**：仅使用单一 proprietary model（gpt-4.1-mini），未测试 open-weight judge，判决稳定性 across models/prompts 未验证
- **Pseudo-reference 循环性**：无 glossary 时参考形式来自系统自身输出，虽有 error judge 验证和 LLM 生成回退，但未能完全消除
- **Hapax 难点**：单出现术语 binary accuracy 比非 hapax 低 12 点，缺乏文档内部证据时判决困难
- **细粒度分类能力有限**：15-class macro-F1 仅 0.196，judge 擅长检测"是否错误"但不擅长识别错误机制
- **数据局限性**：验证仅限 EN–FR 语言对和 biomedical/geoscience/NLP 领域，跨语言/跨领域泛化未知
- **上游 pipeline 误差传播**：term detection（PARANLP 上 precision 91.5%/recall 95.5%）和 alignment 误差会传递至判决，但未直接测量 propagation
- **成本**：端到端评测需数千至数万 LLM calls（IWSLT23 约 $7，PARANLP 约 $50），含 preprocessing 总成本更高
- **未来方向**：测试 open-weight judges、跨 prompt 稳定性、扩展至其他语言对和领域、暴露完整 23-code typology

## 研究启发与可借鉴点
1. **Q1/Q2 分离设计**：将"absolute correctness"和"document consistency"分阶段判决，避免单一 LLM call 的位置依赖性，同时减少调用量（Q1 传播机制）。可迁移至其他需要跨文档一致性判定的 NLP 评测任务。
2. **Benefit-of-the-doubt 惩罚策略**：对聚合错误族（C–H）取最宽松 sub-code，保守估计 penalty，避免过惩罚。该方法可在任何基于 error typology 的自动评测中复用。
3. **Glossary injection 的可解释评估框架**：用可解释 verdict distribution（conformity/justified/unjustified/error）替代单一分数，能仲裁此前无法解释的实验现象（如 glossary injection 的 tension）。启示：评测指标应输出诊断性信号而非仅 ranking。
4. **Compact vs full typology 的 controlled comparison**：附录 D.1 展示压缩 typology（13-code）在 binary error detection 上优于完整 23-code（+2.8% accuracy，-78 FP），为 LLM judge 的 label space 设计提供实证依据。
5. **BioMQM-Terms 资源构建方法**：从 MQM 通用标注中手工提取术语、交叉 MeSH thesaurus、error-driven pass 补充，可作为其他领域术语 benchmark 构建的参考模板。

## 关键术语表
- **TERMJUDGE**：本文提出的文档级术语评测指标，通过两步 LLM judge 流程对每个术语出现输出可解释判决（CONFORMING/JUSTIFIED/UNJUSTIFIED/A1–A7/C–H）
- **LLM-as-a-judge**：使用大语言模型作为评判器，替代或辅助人工对翻译质量进行标注和评分的方法论
- **BioMQM-Terms**：本文构建的术语评测资源，从 bio-MQM 语料提取 729 个生物医学概念及 13,200 个 term–translation 对齐对
- **Divergence**：术语翻译输出与期望参考形式 R(s) 的不匹配，需经 LLM judge 判定为 error 或 valid variation
- **Hapax**：在文档中仅出现一次的源形式，因无法计算一致性而被特殊处理（跳过匹配直接进 Q1，无 Q2）
- **Pseudo-reference**：无 glossary 条目时的期望参考形式，由系统自身最高频翻译或 LLM 生成
- **SPA (Soft Pairwise Accuracy)**：元评测指标，衡量 metric 与 human score 在系统排序上的一致性（permutation test）
- **CONCORDANCER**：本文使用的术语检测工具，基于 SKOS glossary 识别 source document 中的术语出现及变体

## 可复现要素
- **数据集**：STEP（Cornejo Cárcamo et al., 2026）、bio-MQM（Zouhar et al., 2024 EN–FR 子集）、PARANLP（Peng et al., 2026）、IWSLT23（Salesky et al., 2023）；BioMQM-Terms glossary 和 aligned pairs 随代码开源
- **代码**：论文声明 open-source（脚注 1），具体仓库见原文
- **Judge 模型**：gpt-4.1-mini，temperature=0
- **预处理**：CONCORDANCER 术语检测、spaCy lemmatization（fr_core_news_lg）、LLM alignment（WMT25 terminology task prompt，6 in-context examples）
- **Variation labeling**：gpt-4.1，temperature=0，基于 Dahan et al. (2026b) prompt 修订版
- **Prompt**：全部 prompts 见 Appendix L，Python str.format 模板，{domain}/{src_form}/{concept_block} 等占位符
- **评估包**：mt-metrics-eval（WMT metrics meta-evaluation）
- **硬件/成本**：未明确 GPU 要求（LLM calls 通过 API），Appendix K 给出 API 成本估算
