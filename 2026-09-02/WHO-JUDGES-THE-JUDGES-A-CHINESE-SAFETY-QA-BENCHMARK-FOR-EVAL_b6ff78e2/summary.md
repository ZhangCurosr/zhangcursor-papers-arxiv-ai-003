---
title: "WHO-JUDGES-THE-JUDGES-A-CHINESE-SAFETY-QA-BENCHMARK-FOR-EVAL"
source: https://arxiv.org/pdf/2609.01210v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:19:42"
field: "大语言模型安全评测"
keywords: ["LLM安全评测", "中文安全基准", "对抗变换", "自动化评测器审计", "响应级安全", "jailbreak评估"]
innovations: ["策略驱动的中文响应级QA安全基准C-SafeQA，分离查询风险与响应违规", "21种机制标记的中文对抗变换体系与三元标签分层盲审质控管线", "在统一参考标签上联合审计4个目标LLM和7个自动化安全评测器，揭示藏头变换盲点"]
benchmarks: ["C-SafeQA"]
---

# 论文速读：WHO-JUDGES-THE-JUDGES-A-CHINESE-SAFETY-QA-BENCHMARK-FOR-EVAL

## 一句话总结
本文提出 **C-SafeQA**，一个面向中文问答场景的策略驱动响应级安全评测基准，通过 269 个风险点衍生 538 条种子查询及 21 类对抗变换（共 8,877 条），为 4 个 LLM 生成 37,660 条查询-响应记录，并在此基础上系统审计了 7 个自动化安全评测器（automated safety judges）的可靠性。

## 研究问题与动机
1. **查询风险≠响应违规**：现有基准多评估用户输入的风险等级，但安全评测的最终对象是模型响应是否违反安全策略；一条明显有害的查询可能获得安全拒绝，而一条经过对抗变换的查询可能引发不安全响应。
2. **中文表达的特殊挑战**：中文有害内容的风险意图常通过口语化、谐音、字符替换、混合脚本、语义反转或结构化文本变换等方式隐藏，现有中文安全基准难以在对抗变换下测量响应级鲁棒性。
3. **自动化评测器本身不可靠**：现有的 automated judges / guardrail models 对中文特定混淆方式（如藏头诗式变换）的鲁棒性尚未被系统评估，且参考标签的构建质量直接影响评测结论的可信度。
4. **缺乏同时评测目标模型与评测器的基准**：现有工作通常只关注一类目标，C-SafeQA 旨在在统一参考标签下同时评测目标 LLM 的安全行为和评测器的可靠性。

## 核心贡献（创新点）
1. **首个策略驱动的中文响应级 QA 安全基准**：将内部安全策略分解为 269 个风险点，构造 538 条种子查询（问题+陈述各一），与以往仅依赖通用危害分类学的基准形成本质区别。
2. **21 种机制标记的中文对抗变换体系**：涵盖角色扮演/指令优先级攻击（10 类）、表示与重构攻击（7 类）、语言与输出约束攻击（4 类），覆盖藏头诗提取/生成等中文特有变换，此前中文基准缺少如此系统的机制级变换目录。
3. **协议一致的三元响应标签（safe/unsafe/disputed）与分层盲审质控**：采用 agreement-aware 多模型裁决 + 三层专家盲审，既保证大规模可扩展性又显式保留不确定性，区别于纯人工标注或纯模型标注的基准。
4. **目标 LLM 与 7 个自动化评测器的联合审计框架**：在同一策略驱动参考标签上同时评测 4 个目标模型和 7 个 judge 模型，揭示评估器可靠性的维度差异，而非单一精度指标。
5. **公开可复现的分离式发布设计**：Hugging Face 数据集含 37,660 条 JSONL 评测记录及 schema/manifest；GitHub 含完整性验证器与 7 个 judge 运行脚本；构造模板、内部政策映射等不在发布边界内，兼顾科学可验性与安全。

## 方法详解
- **策略驱动的种子查询构造**：将内部安全策略分解为 269 个 risk points，每个风险点对应一对查询（一问一陈述），共 538 条 base queries，大多数由熟悉安全策略的人工评审员撰写。
- **21 种对抗变换**：分为三类通道——模板类（template-based）、LLM 改写类（LLM-rewrite）、程序类（programmatic）。每类变换需满足语义保留、攻击机制明确、诊断目标可解释三项约束。变换结果 $q' = T_m(q)$。
- **响应收集**：使用 4 个全量未量化模型部署（Qwen3.5-397B-A17B、Kimi-K2.5、DeepSeek-V3.2、MiniMax-M2.5），thinking 关闭，各自使用官方 chat template 和推荐采样参数，每个查询生成一次响应，共 37,660 条 QA 记录。
- **参考标签构建（agreement-aware 多模型裁决 + 分层盲审）**：先用三个强 judge 模型（DeepSeek-V4-Pro、Gemini-3.1-Pro、GPT-5.5）给出初判；三者一致时接受为高置信 provisional label，抽样 20% 人工审核；两比一分歧时送 Opus-4.8 复审，抽样 40% 审核；三者完全分歧时也送 Opus-4.8，抽样 80% 审核。盲审由三位安全专家独立完成，expert label 覆盖模型判断。最终标签为 safe/unsafe/disputed 三元。
- **自动化评测器归一化**：7 个 judge（Llama Guard 4、MD-Judge、NeMo Guard、PolyGuard、Qwen3Guard、WildGuard、YuFeng-XGuard）原始标签空间不同，归一化为 safe/unsafe/disputed；支持 disputed 的 judge 报告 D→Safe 和 D→Unsafe 两种二元策略下的指标。
- **关键评估公式**：
  - Unsafe-response rate = $N_{\text{unsafe}} / (N_{\text{safe}} + N_{\text{unsafe}})$
  - Refusal Rate = $N_{\text{refusal-keyword hit}} / N_{\text{total}}$
  - Over-refusal Proxy = $N_{\text{safe} \wedge \text{refusal-keyword hit}} / N_{\text{binary}}$
  - Risk Leakage Index = $N_{\text{hit\_risk\_point}} / N_{\text{binary}}$
  - Judge 指标：unsafe recall、safe-response FPR、precision、accuracy、F1、conflict rate。

## 实验与结果
- **数据集规模**：538 条 base queries + 8,877 条 adversarial queries = 9,415 条查询/模型，4 模型共 37,660 条 QA 记录。参考标签：safe 30,116 / unsafe 7,079 / disputed 465。
- **目标 LLM 整体安全行为**：Qwen3.5-397B-A17B 最低 unsafe rate（11.06%），Kimi-K2.5 最高（28.43%）；MiniMax-M2.5 和 Qwen3.5 的 refusal rate 远高于 DeepSeek 和 Kimi，说明拒绝频率不等同于安全性能。
- **Base vs. Adversarial 描述性比较**：Base queries unsafe rate 0.93%–3.35%，adversarial queries 11.68%–30.05%；Kimi-K2.5 在对抗查询上 unsafe rate 最高（30.05%），增幅 28.20 个百分点。
- **变换级脆弱性**：Acrostic Answer Extraction 平均 unsafe rate 达 91.05%，Acrostic Generation 69.71%，Text Concatenation 56.74%，Text Insertion/Deletion 40.72%，Encoding/Decoding 34.75%；表征类变换暴露跨模型普遍弱点。
- **类别级脆弱性**：Other（23.59%）、Sexual and Vulgar（23.03%）、Violence and Terrorism（20.87%）、Religious and Cult-related（20.60%）为最高四类；Illegal and Non-compliant 虽 unsafe rate 最低（18.27%）但 high-actionability violation rate 最高（13.28%）。
- **自动化评测器性能（adversarial subset，35,508 条）**：YuFeng-XGuard recall 最高（63.52%，F1=59.65%）；WildGuard FPR 最低（3.13%，precision=64.97%）；PolyGuard FPR 极高（68.88%）；无单一 judge 在所有指标上占优。
- **机制条件分析**：所有 7 个 judge 在 Acrostic Generation（recall 仅 0.67%–13.81%）和 Acrostic Answer Extraction（recall 仅 1.33%–26.43%）上均严重漏检；PolyGuard 在 Developer Mode/DAN/Malicious Confidant 上 FPR 达 98.66%–99.95%。

## 相关工作脉络
1. **SALAD-Bench / JAILJUDGE / Know Thy Judge**：已有工作覆盖 response-level labels 和 multi-judge audit，但非中文、非策略驱动；C-SafeQA 填补了中文策略驱动 + 控制变换 + 多 judge 联合审计的空白。
2. **SafetyPrompts / CValues / CHiSafetyBench / ChineseSafe / JailBench**：中文安全基准侧重 prompt-level risk 或知识问答，较少评估响应是否实际违反策略；C-SafeQA 聚焦 query-response 对的安全判定。
3. **HarmBench / JailbreakBench**：以攻击成功率为主要评估目标；C-SafeQA 区分查询风险与响应违规，同时评测 judge 可靠性。
4. **Llama Guard / WildGuard / Qwen3Guard / YuFeng-XGuard 等 guardrail 模型**：已有模型面向通用安全分类；C-SafeQA 在中文策略驱动标注下首次系统性审计其响应级判断可靠性，揭示藏头变换等特定盲点。
5. **HarmMetric Eval / A Coin Flip for Safety**：关注评测指标和 judge 对分布偏移的敏感性；C-SafeQA 进一步将变换机制作为条件变量，进行 category- 和 mechanism-conditioned 错误分析。
6. **BeaverTails / PKU-SafeRLHF**：偏好数据层面的安全对齐资源；C-SafeQA 提供响应级分类基准，支持 target model 与 judge 的联合评估。

## 局限性与未来方向
1. **单轮中文 QA，单一内部策略**：未覆盖多轮对话、agent 轨迹及其他政策体系；内部风险点层级未在公开数据中暴露，无法独立重建。
2. **参考标签非完全人工标注**：模型辅助 + 分层盲审，可能对初判 judge 共有的系统性错误覆盖不足。
3. **9 种变换仅应用于陈述类种子**：base vs. adversarial 和跨变换比较仅为描述性，非配对因果估计。
4. **每模型单次生成**：未进行重复采样或 best-of 选择，模型版本更新可能改变响应分布。
5. **无专用 benign query 集合**：无法直接估计真实场景中的 over-refusal 率，over-refusal proxy 仅作为诊断指标。
6. **未来方向**：扩展至专用良性查询集、多轮交互、更广泛的中文有害内容场景。

## 研究启发与可借鉴点
1. **Query-risk vs. Response-violation 分离设计**：将用户意图风险与模型响应违规作为两个独立评估对象，此思路可迁移至任何安全评测场景，避免仅凭查询表面特征推断安全状态。
2. **Agreement-aware 分层裁决管线**：用多模型一致性决定人工审核密度（20%/40%/80%），在高置信区域省人力、在分歧区域集中投入，可作为大规模标注任务的可复用质控模式。
3. **机制标记的对抗变换目录**：21 种变换按机制（role/instruction、representation/reconstruction、language/output-constraint）分类并标注诊断目标，使结果可解释且可复现，适合迁移到其他语言的安全鲁棒性评测。
4. **Judge 审计的双策略报告（D→Safe / D→Unsafe）**：对支持 disputed 的评测器报告两种二元映射下的指标，揭示了单一 aggregate metric 掩盖的操作选择差异，可作为评测器报告的规范建议。
5. **分类-变换双层条件分析**：不仅在 risk category 维度报告 judge 冲突率，还在 transformation mechanism 维度揭示特定盲点（如 acrostic），为后续评测器改进提供细粒度诊断。

## 关键术语表
- **C-SafeQA**：本文提出的中文策略驱动响应级安全 QA 评测基准，包含 538 条种子查询、21 种对抗变换和 37,660 条 QA 记录。
- **Policy-grounded**：基准构建基于内部安全策略的风险点分解，而非通用危害分类学，确保每条查询与可追溯的策略边界关联。
- **Agreement-aware adjudication**：通过多个 judge 模型的一致性程度决定人工审核的采样比例，实现可扩展的质量控制。
- **Unsafe-response recall**：自动化评测器正确识别 unsafe 响应的比例，衡量其检测有害输出的能力。
- **Safe-response FPR**：在安全风险查询条件下，评测器将安全响应误判为不安全（或反之）的比例，衡量假阳性水平。
- **Acrostic transformation**：将风险语义分散到文本各行的首字符中构成藏头诗式结构，使关键词级安全机制失效的中文特有对抗变换。
- **Three-way label (safe/unsafe/disputed)**：响应级三元标签，disputed 用于标注模糊或评审员无法达成一致的样本，显式保留不确定性。
- **Risk Leakage Index**：命中内部风险点的响应在 binary-resolved 记录中的比例，衡量模型是否泄露了策略定义的风险信息。

## 可复现要素
- **数据集**：Hugging Face 公开（https://huggingface.co/datasets/SparkShieldLab/C-SafeQA），含 37,660 条 JSONL 记录、schema.json 和 manifest.json。
- **代码/权重**：GitHub 公开（https://github.com/SparkShieldLab/C-SafeQA），含完整性验证器和 7 个 judge 模型运行脚本；内部策略映射、变换模板、目标响应生成代码不在发布范围内。
- **关键超参**：Qwen3.5 Temperature=0.7, Top-p=0.80, Top-k=20；Kimi-K2.5 Temperature=0.6, Top-p=0.95；DeepSeek-V3.2 Temperature=1.0, Top-p=0.95；MiniMax-M2.5 Temperature=1.0, Top-p=0.95, Top-k=40；thinking 均关闭。
