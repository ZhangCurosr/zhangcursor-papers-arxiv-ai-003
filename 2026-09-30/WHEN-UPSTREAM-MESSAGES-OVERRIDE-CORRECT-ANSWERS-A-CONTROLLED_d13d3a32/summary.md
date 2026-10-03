---
title: "WHEN-UPSTREAM-MESSAGES-OVERRIDE-CORRECT-ANSWERS-A-CONTROLLED"
source: https://arxiv.org/pdf/2609.36855v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:42:32"
field: "多智能体LLM协作可靠性"
keywords: ["multi-agent LLM", "message passing reliability", "answer substitution", "controlled experiment", "chain-of-thought", "pipeline error recovery"]
innovations: ["提出并验证answer substitution现象：下游代理在有充分独立证据时仍会被上游错误消息定向覆盖到其具体错误答案", "建立2×2证据-消息因子设计的item-level因果框架，分离消息帮助与伤害的边界条件", "通过结论反转与最小编辑控制证明结论标签具有独立因果效应，排除social framing等替代解释"]
benchmarks: ["BIRD", "HotpotQA", "LBMusique", "2WikiMultihopQA", "DROP"]
---

# 论文速读：WHEN-UPSTREAM-MESSAGES-OVERRIDE-CORRECT-ANSWERS-A-CONTROLLED

## 一句话总结
本文通过受控实验系统研究了多智能体LLM协作中上游代理的错误消息如何覆盖下游代理已有充分证据支持的正确答案，揭示了"答案替换"（answer substitution）现象，并探索了通过选择性消息门控实现误差恢复的可行性。

## 研究问题与动机
- **核心问题**：在多智能体流水线（draft-review handoff）中，当下游代理已持有足够的独立证据且能独立给出正确答案时，上游代理的一条错误消息是否会导致下游代理放弃正确答案而采纳上游的具体错误答案？
- **现有方法的不足**： prior work on conformity（Qu et al., 2026）、sycophancy（Sharma et al., 2024）、knowledge-conflict（Xie et al., 2024b）多为单模型观察或模拟投票场景，缺乏在真实流水线中固定证据、操纵消息内容的因果分析，无法隔离消息内容本身的定向影响。
- **动机**：实际多智能体系统（如code generation、QA、debate）广泛采用顺序消息传递，错误和幻觉（hallucination）会沿流水线传播；但现有工作未解释为何下游代理在有充分独立证据时仍会"盲从"上游错误。

## 核心贡献（创新点）
1. **首次识别并精确定义"答案替换"（substitution）现象**：当下游代理具备独立解題能力时，一条错误上游消息可定向地将答案切换到上游的具体错误答案（而非随机退化）。
2. **设计了 fixing downstream evidence 的因果实验框架**：通过2×2因子设计（独立证据present/absent × peer message shown/hidden），在五项基准和五类接收模型上逐项测量消息价值τ，分离了help与harm的边界条件。
3. **揭示了消息价值的双刃剑本质**：同一消息在"上游正确→帮助"与"上游错误→伤害"之间呈现系统性分歧；当接收器已有独立证据时，消息的净价值接近于零（加权均值τ = -0.6 pp），因为help与harm几乎相互抵消。
4. **建立了error-gated recovery的可行性与精度阈值**：通过检测上游错误并选择性移除消息/更换接收器，可在上游错误的样本子集上恢复部分准确率（BIRD +21.7 pp，LBM +30.8 pp），并给出了breakeven detector precision（BIRD约66%，L2W约94%）。
5. **通过系列匹配控制排除了社会 framing、primacy、第二次尝试artifact、评分artifact等替代解释**，确认替换行为由消息内容本身驱动。

## 方法详解
- **实验拓扑**：两节点draft-review交接（single message reception），上游agent生成消息m，下游agent结合任务证据e（检索段落、数据库schema、工具输出）产生最终答案。
- **因子设计**：2（独立证据：有/无）× 2（peer消息：显示/隐藏）= 4条件；通过k=3 majority vote判定下游是否"独立可解"（至少2/3次独立运行答对）。
- **核心度量**：
  - 消息价值 τ(s) = Acc_shown(s) − Acc_hidden(s)，τ > 0 表示帮助，τ < 0 表示伤害。
  - 证据-消息交互 Γ = τ(no evidence) − τ(with evidence)；Γ > 0 表示独立证据降低了对peer消息的依赖。
  - c→w率：独立可解题目中，有消息时从正确变为错误的比例。
- **替换定义**：当前提e固定、上游答案为错、且receiver在无消息时可独立答对，但有消息后改为上游的**具体错误答案**时，即为"answer substitution"。
- **结论反转实验（Conclusion reversal）**：保留推理链格式与证据引用（60–73% token overlap），仅将上游消息的最终结论翻转（错误→正确 / 正确→错误），检验receiver是否追踪结论标签本身。
- **匹配控制实验**：
  - Source-label control：将消息来源标签改为"unverified tool output"或移除，检验social framing效应。
  - Delayed-receipt control：receiver先独立作答再查看消息，检验primacy/review artifact。
  - Matched-review control：两分支均执行两次推理，仅隐藏分支用中性占位符，排除"第二次尝试"本身的随机扰动。
- **CoT诊断**：对override cases标注Chain-of-Thought traces，检验receiver是否在"engagement with evidence"的同时仍follow peer。
- **Recovery实验**：用oracle error knowledge区分upstream-wrong/correct items，测试message removal、receiver replacement、generic warning、error-location提示等干预的组合效应（Table 1分解A/B/C）。

## 实验与结果
- **数据集**：BIRD（SQL生成，n=150，上游错误率61.3%）、HotpotQA（multi-hop QA，n=200，50%）、LBMusique（n=160，43.8%）、2WikiMultihopQA/L2W（n=120，13.3%）、DROP（reading comp，n=120，18.3%）。
- **模型**：上游gpt-4o-mini（主要）、gpt-5.4、kimi-k2.6、qwen-plus；下游gpt-4o-mini（主要）、deepseek-v3.2、kimi-k2.6、glm-5、qwen3.6-plus（共5 receiver）。
- **核心结果**：
  - Γ在所有25个cell中显著为正（p < 0.0001），证据显著降低消息价值。
  - τ随上游正确性 bifurcate：上游正确时τ > 0（帮助），上游错误时τ < 0（伤害）。
  - **c→w率最高达32%**（kimi-k2.6 on BIRD）；全局c→w占比11.1%（297/2,667）。
  - **94%的被审计c→w案例**中，receiver采纳了上游的**具体错误答案**（77/82，bootstrap 95% CI [87%, 98%]）。
  - 结论反转实验：所有11个cell中，corrupt correct conclusion降低准确率、correct wrong conclusion恢复准确率（Figure 4），强因果证据。
  - 最小编辑控制（仅改最终答案行+1句）在LBM和HotpotQA上4/6 cells显著，确认结论标签的独立因果效应。
  - Social framing/delayed-receipt/matched-review 控制均不改变c→w模式，排除替代解释。
  - CoT未可靠消除override：gpt-5.4从15%升至32%，部分模型无显著变化。
- **Recovery结果**（Table 1，upstream-wrong items）：
  - Message removal alone：BIRD +10.9 pp、LBM +6.7 pp、L2W +1.7 pp。
  - Receiver replacement alone：BIRD +10.9 pp、LBM +24.1 pp、L2W +22.3 pp。
  - 组合（C）：BIRD +21.7 pp、LBM +30.8 pp、L2W +24.0 pp。
  - 对全量with-evidence items，无条件移除消息在L2W上净损−21.8 pp（因多数上游正确，丢弃了有用信号）。
  - Breakeven detector precision：BIRD约66%，L2W约94%。
  - Solo accuracy 对 recovery 的解释力 R² = 0.77；same-family vs cross-family 无显著差异（BIRD p = 0.50）。

## 相关工作脉络
1. **Qu et al. (2026) Conformity in multi-agent discussion**：考察peer opinion诱导的从众，但无receiver独立证据控制，无法分离消息内容与社会 framing；本文固定证据、逐项操纵消息，实现item-level因果归因。
2. **Cho et al. (2025) Herd behavior via simulated majorities**：注入模拟多数意见，属aggregate-level设计；本文关注single-message handoff，更贴近真实pipeline原子单元。
3. **Xie et al. (2024b) Knowledge conflicts (parametric vs. context)**：研究模型内部参数知识与上下文证据的冲突；本文冲突发生在两个外部输入之间（peer message vs. task evidence）。
4. **Jamshidi et al. (2026) Hallucination cascade**、**Cemri et al. (2025) Failure mode taxonomy**：描述了错误沿pipeline传播，但未解释"为何receiver在有独立证据时仍不纠正"；本文揭示了directed substitution机制。
5. **Yang et al. (2026) Wrong but useful**：发现wrong-answer messages可能携带有用的中间步骤；本文与之互补——指出conclusion label本身具有独立的有害因果效应，即便reasoning chain正确也可能被conclusion override。
6. **Self-correction / CoT faithfulness literature (Huang et al. 2024; Turpin et al. 2023)**：本文CoT分析显示receiver在evidence-engaged状态下仍follow peer，印证了CoT explanation的unfaithfulness（Trace中引用正确证据却得出错误结论）。

## 局限性与未来方向
- **证据质量假设**：所有实验使用oracle-quality证据（gold passages、完整schema）；在检索不完美的实际部署中，substitution发生率可能与本文测量值不同。
- **任务范围限制**：仅验证于具有离散、可验证答案的任务（QA、SQL）；开放生成、iterative debate、更长pipeline（>2节点）中是否适用尚待检验。
- **恢复方法的现实约束**：Recovery实验使用oracle error knowledge作为上限；实际部署需依赖不完美检测器，其成本收益权衡（§6）显示高错误率基准（BIRD）对detector精度容忍度较低。
- **机制未完全裁决**：Controlled experiments缩小了解释空间，但未区分"receiver用peer claim替换了evidence-based推理"与"同时考虑两者但赋予peer过度权重"两种内禀机制。
- **未来方向**：(1) 探索selective message gating协议（按upstream reliability和receiver evidence state动态决定是否传递消息）；(2) 研究更长pipeline和open-ended任务的generalization；(3) 开发无需oracle的在线error detection方法。

## 研究启发与可借鉴点
1. **2×2 factorial design with evidence控制**可作为多智能体通信可靠性的通用实验范式：固定receiver任务与证据，逐项操纵上游消息（shown/hidden/reversed），实现item-level因果归因；该方法可直接迁移到对debate、tool-augmented workflow、multi-hop reasoning pipeline的评估。
2. **Conclusion reversal + minimal-edit control** 提供了剥离"结论标签"与"推理链"贡献的干净实验策略（60–73% token overlap，仅改结论行+≤1句），可推广至对其他LLM影响力机制（如sycophancy、anchoring）的因果分离。
3. **c→w vs w→c 的转移矩阵与τ(s)分解** 揭示了消息价值的不对称性（help和harm在不同subset上抵消导致aggregate τ ≈ 0），提醒工程实践中不应仅看平均指标，而应区分receiver能力分层（solvable vs. unsolvable items）。
4. **Recovery decomposition (A+B+C)** 为pipeline错误恢复提供了可量化的设计参考：message removal与receiver replacement的组合收益显著，且benefit由receiver capability而非family diversity主导（R² = 0.77），提示选择更高solo能力的receiver比刻意追求跨family多样性更有效。
5. **CoT trace annotation protocol**（evidence-engaged vs. no reasoning二分）可用于诊断"模型明明看到了证据却仍follow错误peer"的机制，为后续设计"faithful reasoning enforcement"提供观测手段。

## 关键术语表
- **Answer substitution**：当下游代理已持有充分独立证据可答对题目时，仍因上游错误消息而切换到上游具体错误答案的现象。
- **Message value τ(s)**：在证据条件s下，show消息相对于hide消息的准确率变化（Acc_shown − Acc_hidden）；τ > 0为帮助，τ < 0为伤害。
- **Evidence–message interaction Γ**：Γ = τ(no evidence) − τ(with evidence)，衡量独立证据降低peer消息依赖的程度；Γ > 0 表明证据具有缓冲作用。
- **Draft-review handoff**：两节点顺序交接拓扑，上游生成草案消息，下游结合独立证据与消息产出最终答案，是多智能体pipeline的原子单元。
- **c→w transition**：Receiver在无消息时可独立答对、有消息后答错的项；衡量消息的定向伤害。
- **w→c transition**：Receiver在无消息时答错、有消息后答对的项；衡量消息的帮助。
- **Conclusion reversal**：保留上游消息的推理格式与证据引用（60–73% token overlap），仅将最终结论翻转（错→对 / 对→错）的实验操纵。
- **Breakeven detector precision**：使error-gated message removal净收益为零的检测器精度阈值；低于该值时，误删正确消息的成本超过移除错误消息的收益。

## 可复现要素
- **数据集**：BIRD、HotpotQA、LBMusique、2WikiMultihopQA (L2W)、DROP均为公开benchmark，论文未声明自行采集数据。
- **代码/权重**：论文未明确声明代码开源仓库；模型通过API调用（OpenAI、DeepSeek、Moonshot、Zhipu、Alibaba、Google），未见模型权重发布声明。
- **关键超参**：所有模型temperature = 0，max tokens = 4096，无system prompt（除非特别说明）；独立可解判定采用k=3 majority vote；bootstrap B = 10,000或20,000；配对检验用McNemar / Fisher exact / TOST equivalence (δ = ±5 pp)。
- **Prompt模板**：附录A.26提供了message-shown、message-hidden、conclusion-reversal、delayed-receipt、CoT等条件完整prompt，可直接复现。
- **模型清单**：见附录Table 29，含API identifier与provider；结论编辑用qwen3.7-max，检测器用gemini-2.5-pro-06-17。
