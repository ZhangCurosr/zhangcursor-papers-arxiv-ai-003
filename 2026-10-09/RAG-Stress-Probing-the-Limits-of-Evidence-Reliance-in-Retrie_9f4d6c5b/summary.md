---
title: "RAG-Stress-Probing-the-Limits-of-Evidence-Reliance-in-Retrie"
source: https://arxiv.org/pdf/2610.11183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:18:33"
field: "RAG 可靠性与鲁棒性评测"
keywords: ["RAG evaluation", "misleading rate", "evidence reliance", "instruction contrast", "counterfactual evidence", "paired audit"]
innovations: ["以闭卷正确子集为条件的误导性接受率 MR，分离替换与既有错误", "指令 Strict/Soft 与词位置 Beginning/Middle/End 的受控交叉诊断", "有害覆盖 HOR 与参考恢复 BCR 的分层配对审计"]
benchmarks: ["TriviaQA-RC", "HotpotQA", "SearchQA", "MedQA"]
---

# 论文速读：RAG-Stress-Probing-the-Limits-of-Evidence-Reliance-in-Retrie

## 一句话总结
本文提出了 RAG-Stress 诊断协议，用于测量 RAG 系统在**已知具备正确闭卷知识**的情况下，受误导证据与强指令影响而**采纳指定干扰答案（foil）的比率（MR）**；并发现 Strict 指令下 MR 显著高于 Soft 指令，且答案出现在证据末尾时最容易引发模型答案替换，但并未带来有益的校正收益。

## 研究问题与动机
- 现有 RAG 评估多聚焦“是否正确”或“是否遵循证据”，**混淆了因错误证据导致的替换行为与已有知识缺失导致的失败**，无法刻画“正确 → 被误导”这一可靠性退化路径。
- 模型可能在无检索时答对，但在检索到**内部一致却支持错误答案的证据**后被覆盖；聚合准确率无法分离出这种“有害覆盖”。
- 先验工作虽研究了冲突/反事实敏感度，但缺乏**同时控制证据断言编辑、指令优先级策略与答案所在词位置**的成套诊断。
- 需要一种可复现、可配对比较的测量范式，用于区分**证据遵从（compliance）**与**事实可靠性（reliability）**。

## 核心贡献（创新点）
- 提出 RAG-Stress 协议，以**闭卷正确子集为条件**测量误导性接受率（MR），把“替换”与“预先失败”解耦。
- 构造 Same-Query 编辑证据对（仅更换含答案的断言），并与 Strict/Soft 指令及 Beginning/Middle/End 词位置做受控交叉。
- 给出配对审计设计，分别报告有害覆盖（HOR）与参考恢复（BCR），避免用不同分母指标相减做净收益推断。
- 提供跨数据集（TriviaQA-RC/HotpotQA/SearchQA/MedQA EN/ZH）、跨模型族（API/open/RL-search agent）与解码配置的稳健性检验。

## 方法详解
- **Same-Query Evidence Intervention**：每题保留问题与参考答案 $a_q^\star$，将支持参考的答案断言替换为同类型实体/选项的支持断言，构造干扰答案 $a_q^-$；同一 Word Position 条件下 clean/edited 仅在含答案断言上不同。
- **Instruction 策略**：Strict 要求以文档为主要事实来源；Soft 允许先验知识覆盖冲突文档，二者仅政策子句不同。
- **Word Position**：目标断言置于目标证据文本的 Beginning / Middle / End，文档顺序固定。
- **基线分层**：$K_M$ 为闭卷答对集合，$U_M=Q\setminus K_M$ 为闭卷答错集合；MR 仅在 $K_M$ 上计算。
- **核心指标**：
  - Clean accuracy：$\operatorname{Acc}_{\pi,p}^C = \frac{1}{|Q|}\sum_{q\in Q}\mathbf{1}[\operatorname{Match}(\hat a_{q,\pi,p}^C,a_q^\star)]$
  - Misleading Rate：$\mathrm{MR}_{\pi,p}=\frac{1}{|K_M|}\sum_{q\in K_M}\mathbf{1}[\operatorname{Match}(\hat a_{q,\pi,p}^E,a_q^-)\land\neg\operatorname{Match}(\hat a_{q,\pi,p}^E,a_q^\star)]$
  - 配对审计：HOR（有害覆盖）在 $K_M$ 上平均，BCR（参考恢复）在 $U_M$ 上平均。
- **不确定性**：question-level bootstrap 估计 95% 区间；不控制多重比较。
- **评分**：大小写、标点、冠词与前缀规范化；gold/foil 同时命中不计入 MR。

## 实验与结果
- **数据集**：TriviaQA-RC、HotpotQA、SearchQA；英语与中文 MedQA（USMLE/MCMLE）。
- **系统**：7 个 API 模型、4 个开源指令模型（Llama-3-8B、Mistral-7B、Qwen3-8B、Gemma-2-9B）、4 个 RL 搜索代理（Search-R1、ZeroSearch-base、RREDSearcher、Search-P1），共 15 个。
- **主要数字（跨模型均值，Strict MR）**：
  - TriviaQA-RC：End 37.0% / Beginning 31.4% / Middle 30.3%
  - HotpotQA：End 41.3% / Beginning 35.8% / Middle 32.8%
  - SearchQA：End 35.0% / Beginning 29.3% / Middle 26.9%
- **指令效应**：Strict 在所有格子均高于 Soft；三数据集平均 MR 的 Strict–Soft 差约 **10.9–13.5 pp**。
- **配对验证**（300 题×2模型）：Strict 下 End–Middle 差为 10.9 pp（[4.5, 17.7]）与 9.6 pp（[2.5, 16.7]），其他区间多包含零。
- **提示措辞稳健性**： paraphrase 下 Strict–Soft 差仍显著（A: 10.8–11.6 pp；B: 7.8–9.3 pp），区间不含零。
- **跨语言**：MedQA EN/ZH 上 Strict 均高于 Soft；API 组 Δ≈12.8/12.4 pp，open 组 Δ≈16.9/18.3 pp，RL 组 Δ≈9.8/9.3 pp。
- **搜索代理**：Soft MR 最高（EN 40.1%、ZH 41.8%），指令差最小。
- **审计 500 题×2 模型**：Strict vs Soft 的 HOR 提升 14.0 / 9.7 pp；BCR 区间包含零，**未证明有益校正提升**。
- **人类校验**：1000 条编辑中 88.1% 全项通过（κ=0.77）；筛后可维持 Δ 方向并微增至 12.8 pp。
- **匹配器校验**：94.2% 一致（κ=0.91）；主要错误来源为别名/格式、语气缓和、多实体、解释性输出。
- **解码**：temperature=0.7/top-p=0.9/5 种子与 greedy 256 token 下 Strict MR 排序一致（Kendall τ=1.00）。

## 相关工作脉络
- **知识冲突与上下文依赖**：Mallen 2023、Longpre 2021、Fakepedia 等研究反事实与来源敏感；本文与它们的关键差异在于**固定闭卷正确子集并测量指定 foil 采纳**。
- **RAG 评估与鲁棒性**：Self-RAG、RAGAs、RAGTruth、RECALL、GaRAGe 等关注 retrieval/generation faithfulness 或 grounding；本文的定位是**补充“从正确到被覆盖”的行为转变诊断**，不替代整体准确率或 groundedness。
- **对抗性证据与记忆**：CF-RAG、AgentPoison、MemoryGraft 研究检索攻击与持久化污染；本文保持查询/索引/记忆不变，仅观察单步编辑断言触发的即时响应变化。
- **位置效应**：Lost in the middle 等研究长上下文位置；本文聚焦**同一文档内答案断言的词位置**对 MR 的影响。
- **冲突缓解方法**：Astute RAG、FaithfulRAG、CARE 等处理冲突/可靠性；本文为**评估诊断层**面，不构成缓解方案。

## 局限性与未来方向
- 编辑有效性以人类抽样校验为主，**未对所有样本做语义级保证**；不同数据集/模型的编辑难度与关系保持度存在差异。
- Word Position 结论基于跨模型均值，**个体模型不完全遵循 End > Beginning > Middle**；内部注意力/召回机制仍未被隔离。
- 配对审计使用 8B 量化模型与局部推理，**外部效度受限于模型规模与部署形态**。
- MR 仅度量指定 foil 采纳，**不覆盖其他错误类型**（如 hedge、hallucination、解释性偏离）。
- 跨数据集比较受题型、多跳结构、基线覆盖率与题目组成差异影响，不宜直接作因果归因。
- 未探索更复杂的检索噪声、多文档排序与实时检索场景。

## 研究启发与可借鉴点
- **基线分层思路**：将评测拆成 $K_M$（先验正确）与 $U_M$（先验错误），对 MR 与 BCR 分别报告，有助于避免“净收益减法”误导。
- **配对审计设计**：同一问题在 clean/edited、不同指令与位置下的联合留存，便于直接观测过渡类型（clean-gold → edited-foil 等）。
- **措辞稳健性控制**：保留同义改写对照，区分“政策含义一致”与“效应量不变”。
- **可迁移到多模态/长上下文 RAG**：把 Word Position 推广到段级/篇章级位置，或用 document slot 做互补实验。
- **可作为 RAG 评测的附加面板**：与 accuracy、faithfulness、grounding 并置，识别“高准确率但高有害覆盖”的系统风险点。

## 关键术语表
- **RAG-Stress**：一种受控诊断协议，用于测量 RAG 系统在已具备正确答案知识时的被误导替换率。
- **Misleading Rate (MR)**：在闭卷正确子集 $K_M$ 上，收到编辑证据后仅命中干扰答案的比例。
- **Strict / Soft instruction**：分别要求“以文档为主要事实来源”与“允许先验知识覆盖冲突文档”的冲突解决策略。
- **Word Position**：含答案断言在目标证据文本中位于 Beginning / Middle / End 的位置条件。
- **Harmful Override Rate (HOR)**：在 $K_M$ 上平均的严格指令 vs 软指令下采纳干扰答案的提升幅度。
- **Baseline Correction Rate (BCR)**：在 $U_M$ 上平均的收到干净证据后参考恢复比例。
- **Paired audit**：对同一问题的 clean/edited、指令与位置组合做联合保存与配对统计的评估流程。
- **Bootstrap 95% CI**：以问题为聚类单位的重采样区间，用于刻画指令/位置对比的不确定性。

## 可复现要素
- **数据集**：TriviaQA-RC、HotpotQA、SearchQA、MedQA（EN/ZH）；通常可在各自公开资源处获取。
- **代码/权重**：论文声明了请求、响应与哈希等重审计材料，支持重打分；主要 15 系统的权重通常来源于各开源/API 官方渠道（论文未统一托管）。
- **关键超参**：输出上限多为短答案；部分实验温度 0.7 / top-p 0.9 / 5 种子采样；greedy 场景 256 token 上限；审计中 20 token 上限、1536 token 输入上限、4-bit 量化、MLX 推理。
- **采样与统计**：QA 位置校验 2,000 次 bootstrap；审计 20,000 次 bootstrap；人类校验 1,000 样本。
