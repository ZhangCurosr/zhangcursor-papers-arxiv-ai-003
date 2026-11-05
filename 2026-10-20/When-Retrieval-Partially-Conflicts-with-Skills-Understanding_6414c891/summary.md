---
title: "When-Retrieval-Partially-Conflicts-with-Skills-Understanding"
source: https://arxiv.org/pdf/2610.25827v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:21:32"
field: "大语言模型检索增强与技能执行"
keywords: ["procedural hallucination", "retrieval-augmented generation", "agent skill execution", "mechanistic interpretability", "value vector intervention", "partial conflict"]
innovations: ["提出SOURCECLASH基准系统化评估部分检索冲突下的程序幻觉", "通过残差替换和注意力分解定位冲突信息的早期生成因果路径", "证明值向量而非注意力权重是检索冲突信息的主要载体"]
benchmarks: ["SOURCECLASH"]
---

# 论文速读：When-Retrieval-Partially-Conflicts-with-Skills-Understanding

## 一句话总结
本文提出了 **SOURCECLASH** 基准，系统研究了语言模型在 Agent Skill 执行过程中因检索内容与选定技能部分冲突而导致的"程序幻觉"现象，揭示了模型在冲突检索下有效答案率大幅下降的根本原因，并通过机制分析定位了导致这一偏好的早期生成残差状态和值向量路径。

## 研究问题与动机
- **核心问题**：当模型已选定一个 Skill（如"纯素烹饪"规范）并获得了检索上下文时，如果检索内容部分兼容（提供有用信息但同时包含违反技能的线索），模型能否选择性使用检索信息、拒绝冲突线索？现实中模型常常采纳与检索一致但违反技能的答案，即"程序幻觉"。
- **现有方法的不足**：已有工作多关注事实正确性或证据支持性，或将检索冲突简化为"完全矛盾"；本文聚焦"部分冲突"场景——检索既有用又含违规线索，评估的是**过程合规性**而非单纯的事实真实性。
- **动机来源**：Agent/Tool-using 系统中，Skill 提供可复用的操作规范，检索提供任务上下文；但检索内容与 Skill 的部分冲突会导致模型偏离正确的执行流程，这一现象缺乏系统性评测与机制理解。

## 核心贡献（创新点）
1. **提出 SOURCECLASH 匹配基准**：构建 1,271 个匹配束（6,355 个提示），覆盖烹饪、网络安全、医疗决策等 10 个程序领域；每条束固定任务、Skill 规范、答案空间，仅改变检索与 Skill 的关系——这是首个系统性评估"部分检索冲突"下技能执行的基准。
2. **揭示程序幻觉的显著且一致的负面影响**：跨 9 个模型，冲突条件下有效答案率平均仅 25.7%，显著低于支持冗余（78.2%）和支持互补（72.8%）条件；大型模型（Qwen3-32B、Llama-3.1-70B）也保持约 50 个百分点的差距。
3. **机制定位：答案偏好差异形成于早期生成阶段**：通过边际间隙分析发现，支持/冲突差异通常在生成的前 3–4 步达到峰值，主要由该时刻的残差状态决定，而非原始 Skill 或检索 span 的残留状态。
4. **证明值向量是冲突信息的载体，注意力权重贡献有限**：注意力分解显示，值向量替换可恢复完整修复效果的 64.8%–82.6%，而注意力权重替换仅贡献 8.5%–15.1%；干预该值通路可使最终有效答案率提升约 5–6 个百分点。
5. **分离冲突信号的"关键信息"与"上下文"两个成分**：通过 CAV 干预发现，检索线索本身及其周围影响"可用性判断"的上下文共同导致程序幻觉，说明冲突不只是局部线索的复制，还涉及模型对线索在 Skill 下"是否可用"的处理。

## 方法详解
- **问题建模**：定义程序幻觉 $\mathcal{H}(y; x, c_s, c_r) = (1 - C_s(y)) \cdot A_r(y)$，其中 $C_s$ 表示答案是否满足 Skill 约束，$A_r$ 表示是否采纳了检索中的违规线索。
- **五元评估设置**：每条束（bundle）展开为 5 个条件：CONFLICT（检索含冲突线索）、SUPPORTIVE-REDUNDANT（检索重申 Skill 约束）、SUPPORTIVE-COMPLEMENTARY（检索提供额外兼容信息）、NO-SKILLS（移除 Skill，仅保留中性检索）、NO-RAG（移除检索，仅保留 Skill）。
- **答案偏好度量（Marginal Gap）**：对每个 bundle 生成 supportive 和 conflict 两个运行，计算 $m_k(t) = s(Y^+|q_t^k) - s(Y^-|q_t^k)$（$Y^+$ 为 Skill 一致答案集，$Y^-$ 为检索支持但 Skill 违规答案集），并定义 $g(t) = m_{sup}(t) - m_{conf}(t)$ 衡量冲突导致的偏好偏移。
- **关键步骤 $t^\star$ 定位**：选择最早使 $|g(t)|$ 接近最大值且趋于稳定的生成步骤作为干预点。
- **残差替换（Residual Replacement）**：在 $t^\star$ 处将冲突运行的残差状态替换为支持运行的对应状态，测量边际修复量 $\Delta_A$。
- **注意力分解**：将注意力输出 $o_{\ell,S}(t^\star) = W^O_\ell [z_{\ell,1,S}, \dots, z_{\ell,H,S}]$ 分解为权重和值向量两部分，分别替换测试其对边际修复的贡献。
- **SVD + 方向干预**：对训练集上的值贡献差 $d_{\ell,S}$ 做 SVD，提取区分冲突/支持的奇异向量，在 held-out 集上删除这些方向以测量修复效果。
- **CAV 概念干预**：训练 logistic regression CAV 分别分离"关键信息"和"上下文"两个成分，测试移除各自的边际修复。

## 实验与结果
- **数据集**：SOURCECLASH，1,271 个匹配束，6,355 个提示，覆盖 10 个领域（烹饪、网络安全、教育、金融、HR、IT、法律审查、物流、制造、医疗决策）。
- **评估模型**：9 个模型（Llama-3-8B、Gemma-3-12B、Llama-2-13B、Qwen3-14B、DeepSeek-V3.2、GPT-5-mini、Gemini-2.5-Flash、Qwen3-32B、Llama-3.1-70B-Instruct）。
- **主要结果**：
  - 冲突条件下有效答案率平均 **25.7%**，支持冗余 **78.2%**，支持互补 **72.8%**，差距约 50 个百分点。
  - Qwen3-32B：支持冗余 84.6% → 冲突 33.5%；Llama-3.1-70B：支持冗余 86.8% → 冲突 38.0%。
  - NO-SKILLS 条件几乎为零（~1.4%），NO-RAG 约 32.7%，说明 Skill 是必要但检索冲突显著干扰执行。
- **机制分析**（4 个开源模型）：
  - 边际间隙峰值：prefill 阶段 <6%，生成阶段平均在 3.0–4.4 步。
  - 残差替换：$t^\star$ 处替换修复量最大（如 Gemma-3-12B 为 4.963），远高于检索 span（0.465）或 Skill span（≈0）。
  - 注意力分解：值向量替换恢复完整效果的 64.8%–82.6%，注意力权重仅 8.5%–15.1%。
  - 值方向干预：FULL 区域修复最强（如 Gemma-3-12B 为 +2.552），RAG 区域也有效但较弱，说明冲突信号散布于全 prompt。
  - CAV 干预：移除"关键信息"修复 +0.291 至 +0.626，移除"上下文"修复 +0.220 至 +0.436。
  - 生成后验证：Llama-3-8B 有效答案率提升 5.94 个百分点，Qwen3-14B 提升 5.48 个百分点。

## 相关工作脉络
- **幻觉研究**：TruthfulQA、Halueval、Factscore 等关注事实/证据层面的幻觉；本文聚焦"过程合规性"幻觉，目标行为是选择性检索使用而非事实正确性。
- **检索增强生成与知识冲突**：Self-RAG、WikiContradict、TruthfulRAG、CARE 等研究检索冲突；但多假设检索完全矛盾或支持单一事实，本文强调"部分冲突"——检索既有用又含违规线索的复杂场景。
- **指令遵循与 Agent 评估**：GuideBench、ComplexBench、SRA-Bench 评估指令/多约束/技能检索能力；本文聚焦 Skill 已选定后的执行阶段，引入匹配的 source-relation 对比设计。
- **知识定位与编辑**：Knowledge Neurons、Knowledge Editing 研究事实知识在模型内部的表征；本文将注意力扩展到"过程性知识 vs 检索信息"的动态冲突，并用残差替换和注意力分解定位因果路径。

## 局限性与未来方向
- **场景局限**：仅研究 Skill 已选定后的执行阶段，未覆盖 Skill 发现、渐进加载、多 Skill 协调、实时工具输出、多轮交互等更复杂 Agent 场景。
- **数据局限**：基准为合成匹配数据，虽然人工审核确保质量，但可能与真实检索日志或用户数据存在分布差异。
- **评估方式**：采用规范化字面匹配（answer atom matching）+ 人工验证，未来可扩展语义评估。
- **机制分析范围**：当前干预仅在早期生成步骤，未探索训练时或更长推理链中的修复方法。

## 研究启发与可借鉴点
- **部分冲突的评估设计**：五元匹配束（CONFLICT / SUPPORTIVE-REDUNDANT / SUPPORTIVE-COMPLEMENTARY / NO-SKILLS / NO-RAG）可作为评估 Agent 系统选择性检索能力的通用框架。
- **程序幻觉的定义与度量**：$\mathcal{H} = (1 - C_s) \cdot A_r$ 的形式可迁移到其他"过程 vs 证据"冲突场景（如合规审查、安全策略执行）。
- **机制分析方法**：边际间隙 + 残差替换 + 注意力分解 + SVD 方向 + CAV 的组合流程，可用于诊断其他检索-指令冲突类问题。
- **干预策略启示**：值向量是冲突信息的主要载体，可通过在 $t^\star$ 处干预值通路或训练去偏 value representation 来提升 Skill 合规性，为推理时修正提供新方向。
- **团队结合点**：若团队研究 Agent 技能执行、RAG 系统或指令遵循，SOURCECLASH 可作为基准，残差/值向量干预思路可作为提升 Skill 合规性的技术手段。

## 关键术语表
- **Procedural Hallucination（程序幻觉）**：模型采纳与检索一致但违反选定 Skill 规范的答案，导致输出在检索层面受支持但在过程层面无效。
- **SOURCECLASH**：本文提出的匹配基准，包含 1,271 个 bundle、6,355 个提示，用于系统化评估检索-技能冲突下的程序幻觉。
- **Marginal Gap（边际间隙）**：$g(t) = m_{sup}(t) - m_{conf}(t)$，衡量冲突检索相对于支持检索导致的答案偏好偏移量。
- **Decisive Step $t^\star$**：支持/冲突边际间隙达到最大值且趋于稳定的最早生成步骤，作为后续残差替换和注意力分解的干预点。
- **Value Vector vs Attention Weight**：注意力机制中，值向量携带"写入了什么信息"，注意力权重决定"从哪些 token 取信息"；本文发现值向量是冲突信息的主要载体。
- **CAV（Concept Activation Vector）**：通过 logistic regression 训练的概念方向向量，用于分离和干预"关键信息"与"上下文"对答案偏好的影响。

## 可复现要素
- **数据集**：SOURCECLASH，论文声明将在发表后公开（包含 train/dev/test 拆分、prompt 模板、评估脚本）。
- **代码/权重**：基准记录、评估脚本、干预代码将开源；使用开源模型（Llama-3-8B、Gemma-3-12B、Llama-2-13B、Qwen3-14B）及 API 模型（GPT-5-mini、Gemini-2.5-Flash 等）。
- **关键超参**：greedy decoding（do_sample=False, max_new_tokens=512, seed=42）；边际间隙阈值 α=0.7 用于选择 $t^\star$；FP32 精度；约 200 GPU-hours 计算量。
- **评估协议**：确定性 answer atom 匹配 + 300 样本人工验证（Cohen's κ=0.9255）。
