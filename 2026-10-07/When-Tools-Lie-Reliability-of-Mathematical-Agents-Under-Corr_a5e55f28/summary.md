---
title: "When-Tools-Lie-Reliability-of-Mathematical-Agents-Under-Corr"
source: https://arxiv.org/pdf/2610.08097v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:57:45"
field: "工具增强推理智能体的可靠性"
keywords: ["tool-augmented agents", "mathematical reasoning", "robustness", "verification policy", "corrupted feedback", "self-correction"]
innovations: ["可控工具反馈污染框架与四种验证策略系统比较", "证明验证频率比验证器质量更决定鲁棒性", "引入虚假不信任率(FDR)指标分离调用不足与质量不足"]
benchmarks: ["MATH-500", "GSM8K"]
---

# 论文速读：When-Tools-Lie-Reliability-of-Mathematical-Agents-Under-Corr

## 一句话总结
本文构建了一个可控的"工具反馈污染"实验框架，系统评估数学推理智能体在工具返回看似合理但实际错误结果时的鲁棒性；核心发现是：**强制性同上下文反思（mandatory reflection）可将准确率从72.4%完全恢复至100%，而可选验证策略的效果高度依赖模型主动调用频率**。

## 研究问题与动机
- 现有工具增强推理（如ReAct）通常假设模型自身不确定而工具可信，但工具可能因软件bug、状态过时、集成故障或对抗性干扰而**静默返回外观合理但语义错误的结果**，这类"语义失败"比超时或格式错误更难察觉。
- 自我修正（self-correction）方法若缺乏外部 grounding 可能不可靠；外部验证方法（如CRITIC、Chain-of-Verification）引入额外验证通道，但未在受控污染条件下系统比较不同验证策略。
- 核心研究问题：(RQ1) 成熟数学智能体对工具输出语义污染的易感性如何？(RQ2) 无验证、强制同上下文反思、可选新上下文验证、可选结构验证四种策略谁更鲁棒？(RQ3) 检查频率与条件成功率哪个才是鲁棒性的关键相关因素？

## 核心贡献（创新点）
1. **可控污染基准（controlled corruption framework）**：通过隐藏拦截器将工具结果替换为经过独立验证的" plausible incorrect"值，每次试验最多污染一个工具调用——与已有工作对比，前者能精准量化单一错误传播，后者多关注整体幻觉率。
2. **四种验证策略的系统比较**：首次在同一31题基准上对比 B0/B1/B2/B3 四种政策，并严格区分"验证器可用性"与"验证政策"是两个独立组件。
3. **检查行为分解（behavioral decomposition of verification policy）**：将成功概率拆解为 $P(S)=P(C)\cdot P(S|C)+P(\neg C)\cdot P(S|\neg C)$，揭示 Robustness 差异主要来自 **调用频率 $P(C)$** 而非条件成功率 $P(S|C)$。
4. **虚假不信任率（false-distrust rate, FDR）指标**：引入干净条件下将正确工具输出误判为错误的比率（B1为3.2%，B2/B3为0%），提示强制策略需权衡误报成本。

## 方法详解
- **基准构建**：从合成题、MATH-500、GSM8K 中筛选31题（代数、微积分、数论、算术），每个污染试验将工具结果替换为数值扰动/缺解/符号替换等" plausible incorrect"形式，且每项均独立验证为错误。
- **四种Agent策略**：
  - **B0（vanilla）**：普通工具循环，无任何验证步骤。
  - **B1（mandatory reflection）**：提交初稿后**无条件**触发同上下文重新审视 prompt，要求模型核查推理、工具使用与最终答案，允许调用 `challenge_evidence` 标志不一致并重算。
  - **B2（optional fresh-context verification）**：可选调用独立 reviewer 实例，仅可见原题与争议结果，不可见求解过程。
  - **B3（optional structural verification）**：可选调用结构独立验证（如求根后代回验证、微分验证积分）。
  - **B4（full restart，支持性基线）**：仅在明确检测到污染后从头重启。
- **模型**：Claude Haiku 4.5 与 Claude Sonnet 5，同一 API 环境、相同 prompt 与固定参数；496 次主试验（每模型×每政策×31题）。
- **评估**：主要指标为最终答案准确率；采用 problem-level bootstrap 95% CI 与 McNemar 配对检验；另报告 false-distrust rate（干净条件下误拒正确工具输出的比率）。

## 实验与结果
- **B0 脆弱性**：Haiku 从干净100% → 污染72.4%（降幅27.6pp，95% CI: [55.2, 86.2]）；Sonnet 从100% → 81.5%（降幅18.5pp，[66.7, 96.3]）。**更高能力不消除对虚假外部证据的易感性**。
- **B1 强制反思**：Haiku 72.4% → 100%（McNemar p=0.0078，n=28配对题）；Sonnet 81.5% → 100%（p=0.0625，方向一致）。两者均达天花板，误拒率3.2%但不损害最终准确率。
- **B2/B3 可选验证**：Haiku 约79.3%；Sonnet 80.0%/84.6%。**实际提升来自调用，而非验证器本身质量**——B2/B3 仅被调用68.5%/74.5%的试验。
- **RQ3 分解**： pooled 下 B1 检查频率90.9%且条件成功率100%；B2 检查68.5%、条件100%；B3 检查74.5%、条件95.1%。**未被检查的试验成功率仅35.3%/42.9%**，差距主要来自 $P(C)$。
- **B4 支持性结果**：明确检测后全题重启成功率100%（85次尝试，81次可评分）。
- **稳定性**：B0 在15题子集三次重跑一致率95.0%（57/60 cells）。

## 相关工作脉络
- **ReAct**（Yao et al., ICLR 2023）：工具增强推理的经典框架，假设工具可信；本文与其互补，聚焦工具自身" plausible but false"的语义失败。
- **Reflexion / Self-Refine**（Shinn et al., NeurIPS 2023; Madaan et al., 2023）：内在校正方法，本文指出无 grounding 的内在自修正不可靠。
- **CRITIC**（Gou et al., ICLR 2024）与 **Chain-of-Verification**（Dhuliawala et al., ACL 2024 findings）：引入外部验证通道，但未在受控污染下比较"强制 vs. 可选"政策差异。
- **Let's Verify Step by Step**（Lightman et al., ICLR 2024）：过程监督评估中间推理；本文更关注工具反馈层面的污染。
- **Large Language Models Cannot Self-Correct Reasoning Yet**（Huang et al., ICLR 2024）：内在自修正不可靠的实证；本文进一步证明**外部证据本身也可能被污染**。

## 局限性与未来方向
- 样本量小（每 cell <30 观测），统计功效有限；B1 天花板效应（100%）无法判断其是否"最优"。
- 仅测试一家模型家族（Claude），跨平台泛化性待验证。
- 题目难度偏低（可心算复核），B1 的100%恢复未必推广到工具不可替代的长程任务。
- 污染为瞬态（仅首步工具调用被污染，96.4% 案例），持续性故障工具场景尚未覆盖。
- 检查行为未随机化，频率-鲁棒性关联为相关非因果。
- 未来工作：强制始终调用 B2/B3 验证器以隔离"验证器质量"本身的影响。

## 研究启发与可借鉴点
1. **强制策略设计**：对关键计算环节引入"不可跳过"的反思/re-compute 阶段，是当前已知最有效的污染防御手段。
2. **可迁移实验设计**：可控污染框架（interceptor + plausible incorrect substitution + 独立验证）可直接移植到其他工具增强场景（代码执行、检索增强、函数调用）。
3. **指标借鉴**：FDR（虚假不信任率）与 $P(C)$ vs. $P(S|C)$ 分解框架有助于后续研究区分"调用不足"与"验证器质量不足"两类问题。
4. **部署启示**：构建更好验证器只是必要条件而非充分条件；**有效政策必须保证验证的被调用**（triggering mechanism）。
5. **本团队结合机会**：在数学 agent、代码生成 agent、RAG 系统中引入 B1 式 mandatory reflection 作为默认管线，同时监控 FDR 以避免过度校正。

## 关键术语表
- **Plausible incorrect（看似合理的不正确）**：经数值扰动/缺解/符号替换等方式生成的"外观可信但实质错误"的工具返回结果。
- **Mandatory same-context reflection（强制同上下文反思）**：无条件触发的二次审视环节，模型在原推理上下文内核查并重算。
- **Optional fresh-context verification（可选新上下文验证）**：调用独立实例仅看原题与争议结果，不可见原始求解轨迹。
- **Optional structural verification（可选结构验证）**：用数学上不同的操作交叉验证（如求根后代回、微分验证积分）。
- **False-distrust rate (FDR)**：干净条件下将正确工具输出误判为错误的比率，衡量验证策略的误报成本。
- **Checking frequency $P(C)$**：模型主动发起验证行为的概率，是本研究中与鲁棒性最相关的因子。
- **Conditional success $P(S|C)$**：在已发起检查的条件下最终答对的概率。
- **Transient corruption（瞬态污染）**：仅在首个工具调用注入错误，后续调用返回真实值的设定。

## 可复现要素
- **数据集**：31题子集来自合成题、MATH-500、GSM8K；**论文未明确声明公开**，但提供了污染样例与 prompt 模板（Appendix B/C/D）。
- **代码/权重**：论文未声明开源仓库；模型为闭源 API（Claude Haiku 4.5、Sonnet 5）。
- **关键超参**：固定 API 参数（论文未详述具体 temperature/top-p）；每试验最多污染1次工具调用；prompt 模板见 Appendix B。
- **统计方法**：problem-level bootstrap 95% CI（Clopper-Pearson exact binomial）、McNemar 配对检验。
