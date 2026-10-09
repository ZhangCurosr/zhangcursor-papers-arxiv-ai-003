---
title: "RAG-Stress-Probing-the-Limits-of-Evidence-Reliance-in-Retrie"
source: https://arxiv.org/pdf/2610.11183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:18:52"
field: "检索增强生成评估与鲁棒性"
keywords: ["RAG", "retrieval-augmented generation", "evidence reliability", "knowledge conflict", "adversarial evaluation", "diagnostic benchmark"]
innovations: ["提出RAG-Stress协议，在基线正确子集上测量指定干扰答案的采纳率，解耦答案替换与既有错误", "首次系统交叉Instruction(Soft/Strict)与Word Position(Beg/Mid/End)对误导遵循的影响", "构建HOR/BCR配对审计框架分离有害覆盖与有益修正，证明Strict仅增加有害覆盖而无对应收益"]
benchmarks: ["TriviaQA-RC", "HotpotQA", "SearchQA", "MedQA (EN/ZH)"]
---

# 论文速读：RAG-Stress-Probing-the-Limits-of-Evidence-Reliance-in-Retrieval-Augmented-Generation

## 一句话总结
论文提出 RAG-Stress，一种受控诊断协议，在模型闭卷回答正确的前提下，通过编辑证据中单一断言来测试模型采用指定干扰答案（foil）的倾向；结果表明 Strict 指令比 Soft 指令诱发更高的误导性遵循率（高 10.9–13.5 pp），且答案出现在证据末尾时影响最大。

## 研究问题与动机
- **现有指标的盲区**：标准准确率无法区分"模型原本答错"和"有正确答案被证据覆盖"两种情形，掩盖了"错误替换"这一具体行为。
- **误导性证据的真实风险**：即使证据自洽且与问题相关，若支持一个不相容的答案，模型仍可能被诱导放弃原有正确知识，而现有评估（如 groundedness、faithfulness）无法捕捉。
- **指令策略的影响未被系统量化**：要求模型"以文档为唯一真相来源"与允许"以已有知识优先"之间，模型行为差距有多大尚不清楚。
- **证据位置效应的机制不明**：答案 span 在证据文本的 Beginning/Middle/End 是否系统性影响模型遵循率，缺乏受控交叉实验验证。

## 核心贡献（创新点）
1. **提出 RAG-Stress 诊断协议**：以基线正确子集为条件测量指定 foil 采用率，将"答案替换"与"既有错误"解耦——不同于以往只报告整体准确率的评测。
2. **首次系统交叉 Instruction × Word Position 干预**：在同一问题、同一份证据上构造 Clean/Edited 配对，并系统比较 6 种 prompt-position 组合，揭示行为受提示策略控制的程度。
3. **跨模型族与跨任务的实证规律**：在 15 个系统（API/开源/RL agent）、3 个英文 QA 数据集及中英 MedQA 上统一度量，指出端部位置（End）诱发的 MR 最高，且 Strict 在全部 15 系统中均产生更高 MR。
4. **配对审计框架分离有害覆盖与有益修正**：构建 HOR（有害覆盖率）和 BCR（有益修正率）两个独立指标，证明 Strict 增加有害覆盖但没有对应改善有益修正。

## 方法详解
- **同查询证据干预（Same-Query Evidence Intervention）**：对每个 QA 样本，构造 Clean 版本（支持参考回答 $a_q^*$）和 Edited 版本（仅改动含答案的单一断言，支持指定干扰 $a_q^-$），其余文本保持固定。
- **Word Position 条件**：将含答案的完整断言置于目标证据文本的 Beginning/Middle/End 三个位置，非目标文档顺序不变。
- **Soft vs Strict 指令**：Strict 明确要求"以文档为主要事实来源，即使文档看似错误也须作答"；Soft 允许"当文档与公认事实冲突时优先使用自身知识"。
- **误导性率（MR）**：在闭卷正确的子集 $K_M = \{q : \text{Match}(M(q, \emptyset), a_q^*) = 1\}$ 上，统计 edited 条件下回答等于 $a_q^-$ 且不等 $a_q^*$ 的比例：
  $$\mathrm{MR}_{\pi, p} = \frac{1}{|K_M|}\sum_{q \in K_M} \mathbf{1}[\text{Match}(\hat{a}_{q,\pi,p}^E, a_q^-) \wedge \neg\text{Match}(\hat{a}_{q,\pi,p}^E, a_q^*)]$$
- **配对审计指标**：HOR（Harmful Override Rate）衡量 Strict 下基线正确问题的 foil 采纳率；BCR（Baseline Correction Rate）衡量严格指令在基线错误问题上的参考回复恢复率。
- **统计推断**：对 20,000 次 question-level bootstrap 抽样，报告点wise 95% 百分位区间；不对多重比较校正。

## 实验与结果
- **数据集**：TriviaQA-RC（Joshi et al., 2017）、HotpotQA（Yang et al., 2018）、SearchQA（Dunn et al., 2017）、MedQA（EN USMLE + ZH MCMLE）。
- **模型（15 个）**：7 个 API 模型（GPT-5.6、Claude Opus 4.6、Qwen3.8-Max、DeepSeek-V4-Pro、GLM-5.2、Kimi-K3、HY4-Preview）、4 个开源模型（Llama-3-8B、Mistral-7B、Qwen3-8B、Gemma-2-9B）、4 个 RL 搜索 Agent（Search-R1、ZeroSearch-base、REDSearcher、Search-P1）。
- **关键结果**：
  - **Instruction 效应**：Strict 在全部 15 系统和 3 个 QA 数据集上 MR 均高于 Soft；跨模型平均 gap 为 TriviaQA-RC 12.2pp、HotpotQA 13.5pp、SearchQA 10.9pp。
  - **Word Position**：End > Beginning > Middle，Strict 下均值分别为 TriviaQA-RC 37.0%/31.4%/30.3%。
  - **配对验证**：Strict 下 End−Middle = 10.9pp（QA/A，CI [4.5, 17.7]），9.6pp（QA/B，CI [2.5, 16.7]），区间均不含零。
  - **MedQA 跨语言**：API 模型 Δ_instr 中英相近（12.8 vs 12.4），RL Agent 在两种语言下 Soft MR 均高达 ~40%。
  - **配对审计（500 题）**：Llama-3-8B 的 HOR 从 50.5% 升至 64.5%（+14.0pp）；Qwen3-8B 从 54.2% 升至 63.8%（+9.7pp）；BCR 区间包含 0，无显著改善。
  - **解码一致性**：temperature 0.7 top-p 0.9 与 greedy 的 Strict-MR 排序 Kendall τ = 1.00。
  - **人机验证**：1000 条编辑中 88.1% 通过全部四项标准（κ = 0.77）；matcher 与人工标签 Cohen's κ = 0.91。

## 相关工作脉络
- **Fakepedia / Entity substitutions**（Monea et al., 2024; Longpre et al., 2021）：通过实体替换构造冲突，但本文进一步固定基线正确子集并系统交叉指令与位置，诊断分辨率更高。
- **RECALL / RAGTruth**（Liu et al., 2023; Niu et al., 2024）：聚焦鲁棒性或幻觉标注；本文测量的是"有正确答案→被替换"这一特定行为转变，而非整体准确率。
- **Astute RAG / FaithfulRAG / CARE**（Wang et al., 2025; Zhang et al., 2025; Choi et al., 2025）：解决 fact 间冲突或提升忠实度；本文是诊断工具，不提供缓解方法。
- **Context-aware decoding（Trust Your Evidence）**（Shi et al., 2024）：通过解码策略降低上下文依赖；本文揭示的是 prompt policy 层面的行为差异，可作为对照基线。
- **AgentPoison / MemoryGraft**（Chen et al., 2024b; Srivastava & He, 2025）：研究持久记忆污染攻击；本文的编辑是瞬时的、单次对话内的，不改变系统知识。
- **Self-RAG / RAGAs**（Asai et al., 2024; Es et al., 2024）：集成自反思或自动化评测；本文的定位是补充性诊断协议，独立于这些系统可用。

## 局限性与未来方向
- **证据编辑是人工构造的**：虽经 88.1% 通过人机校验，但仍可能泄露参考回答或改变语义关系，未必代表真实 RAG 中的噪声检索结果。
- **仅测单一断言编辑**：未覆盖多跳中"中间断言"vs"终端断言"的差异（HotpotQA 更高 MR 的因果机制未被隔离）。
- **配对审计仅两个 8B 模型**：无法代表所有模型家族，且未验证语义有效性。
- **未探索缓解方法**：论文声明贡献是"诊断分辨率"而非提出 mitigation，未测试任何对抗策略。
- **跨模型归因受限**：不同模型的 $K_M$ 大小差异大，低 MR 可能源于高准确率而非强鲁棒性。

## 研究启发与可借鉴点
1. **基线分层设计值得复用**：将 $K_M$（闭卷正确子集）作为分母度量特定行为，是解耦"既有错误"与"指令驱动替换"的有效范式，可迁移至其他 RAG 评测场景。
2. **Instruction × Position 的交叉设计**：将 prompt policy 与证据结构位置作为正交因子，能更细致地定位脆弱点，适合用于 agent/RAG 系统的 stress test pipeline。
3. **HOR/BCR 分离评估思路**：有害覆盖与有益修正用不同分母独立报告，避免互相抵消，对后续"RAG 净收益"评估有方法论参考价值。
4. **配对 bootstrap 估计区间**：question-level 重采样保留重复 cell，比简单 pooling 更可靠，可用于小规模审计实验的统计推断。
5. **可迁移至中文场景**：中英 MedQA 实验表明该协议具有跨语言可扩展性，结合国内垂直领域数据（如医疗、法律）可直接复用该评测框架。

## 关键术语表
- **Misleading Rate (MR)**：在模型闭卷回答正确的前提下，收到 edited 证据后采用指定干扰答案的比例。
- **Hard Override Rate (HOR)**：Strict 指令下，基线正确问题中采纳干预 foil 的比率，衡量有害覆盖。
- **Baseline Correction Rate (BCR)**：Strict 指令下，基线错误问题中恢复参考回答的比率，衡量有益修正。
- **Word Position**：含答案断言在目标证据文本中的位置（Beginning/Middle/End），影响模型的遵循倾向。
- **Same-Query Evidence Intervention**：对每个问题保留问题与其余证据不变，仅改动一处含答案的断言以支持指定干扰。
- **Baseline-correct subset ($K_M$)**：模型在闭卷条件下（无检索）回答正确的题目集合，作为 MR 的分母。
- **Soft vs Strict 指令**：Soft 允许模型以自身知识优先；Strict 要求以文档为主要事实来源，即使文档看似错误。
- **Paired Bootstrap**：以 question 为单位进行 20,000 次重采样，计算每道题内各 cell 的差值分布，估计点wise 95% 区间。

## 可复现要素
- **数据集**：TriviaQA-RC、HotpotQA、SearchQA、MedQA（公开，但论文使用的样本需自行采样）。
- **代码/权重**：论文未提供开源代码仓库；推理使用 MLX + MLX-LM 0.31.3，checkpoint 指纹已在附录 D 记录。
- **关键超参**：温度采样（0.7，top-p 0.9，5 seeds）；greedy 解码，输出 cap 20–256 tokens，输入 cap 1,536 tokens；4-bit affine 量化，group size=64。
- **Prompt 模板**：附录 A.1 提供完整模板，Strict/Soft 子句明确列出。
- **评测脚本**：附录 A.3 给出 Algorithm 1 伪代码，评分细节见附录 A.4。
- **人工标注**：6 位 annotator，κ=0.77；matcher 验证 κ=0.91（详见附录 D）。
