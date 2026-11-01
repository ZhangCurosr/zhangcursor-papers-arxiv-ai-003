---
title: "Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constra"
source: https://arxiv.org/pdf/2610.12281v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:57:15"
field: "非编码调控变异功能解释与AI可信赖性"
keywords: ["noncoding variant interpretation", "allele-specific binding", "agentic genomics", "transcription factor binding", "hallucination mitigation", "evidence-constrained reasoning", "DNABERT", "ADASTRA"]
innovations: ["确定性科学+智能体推理架构：所有生物分类由规则代码完成，LLM仅规划与报告，通过架构约束根除幻觉", "假设驱动的调查循环：观测依赖分支使同一规划器在相同位点产生结构不同的调查轨迹，证据充足即停止", "证伪作为一等公民：CAN_CLAIM_EXPERIMENTAL标记防止计算证据被提升为实验确认"]
benchmarks: ["rs6983267 8q24 locus (4 TFs)", "AlphaGenome Atlas AVI cross-reference", "Frontier LLM baseline (GPT-4o, Claude 4.6 Sonnet, Gemini 2.5 Pro)"]
---

# 论文速读：Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constra

## 一句话总结
ARGUS 是一个证据约束型智能体框架，通过将确定性生物计算与 LLM 推理严格分离，对非编码调控区 SNP 的转录因子结合变化进行假设驱动的证据调查，在 rs6983267 8q24 癌症风险位点上实现了四条从相同规划器出发但路径迥异的调查轨迹，有效避免了现有 LLM 因幻觉而错误归类结合变化的问题。

## 研究问题与动机
- 超过 90% 的 GWAS 疾病相关变异位于非编码调控区，但其功能解释仍是基因组医学的核心瓶颈；现有资源只区分"模型预测了效应"，却无法区分"独立证据支持该预测"。
- 通用 LLM 在解释 TF 结合变化时容易幻觉：作者早期使用 Qwen 2.5 7B 分析 rs6983267 时，SP1 的两个等位基因结合概率分别为 0.007 和 0.0009（均远低于 0.5 活性阈值），但 LLM 却报告"结合增强，log_odds_ratio = 3.02"——将两个接近零的数的比值误判为生物学显著事件。
- DNABERT 类模型存在饱和盲点：当两个等位基因预测概率均 >0.95 时，模型无法分辨等位基因差异（如 FOXA1：pref = 0.995，palt = 0.995），产生假阴性预测。
- 当前 agentic AI 系统在生物研究中缺乏可审计的假设检验机制，未能区分预测与解释、计算证据与实验证据。

## 核心贡献（创新点）
- **确定性科学+智能体推理（deterministic science, agentic reasoning）原则**：所有生物分类由基于规则的代码完成，LLM 只负责选择调查动作和撰写报告，不得覆盖任何预分类事实；与既往工作本质区别在于将幻觉风险通过架构约束而非提示词控制来根除。
- **假设驱动的调查循环与观测依赖分支**：规划器根据当前不确定性状态动态选择证据源（ADASTRA → JASPAR → ENCODE cCRE），中间结果决定下一步行动，同一规划器在同一位点产生四条结构不同的轨迹；区别于固定 DAG 流程。
- **证伪作为智能体科学的一等设计模式**：将每个预测视为可检验假设，通过 ADASTRA 等直接实验证据实现可证伪性，并以 CAN_CLAIM_EXPERIMENTAL 标记防止计算证据被提升为实验确认；区别于仅做预测的 AlphaGenome Atlas 等系统。
- **LLM 辅助规划可降低工具调用量而不影响结论**：LLM 规划器（Claude Sonnet 4.6）在相同验证器和停止规则下与固定优先级规划器得出完全一致 verdict，但以 9 次工具调用替代 13 次（节省约 31%），主动拒绝不改变结论的冗余证据源。

## 方法详解
- **双阶段架构**：第一阶段为确定性预测阶段（复用 DeepVRegulome pipeline），第二阶段为智能体调查阶段，二者由 LangGraph 编排。
- **预测阶段**：用户查询经确定性 regex 路由到 variant/gene/region/TF 四种模式；ARGUS 库含 458 个 DNABERT 基 TF 结合模型，通过 GTEx 表达谱 + ENCODE/ChIP-Atlas 1.93 亿 peaks + 疾病相关注释筛选相关 TF；每个模型输入 510bp 序列（以变异为中心，分别替换 ref/alt 等位基因），输出 p_ref 和 p_alt。
- **确定性分类器**：基于 0.5 活性阈值分为四类——LOF/GOF（一个等位基因 ≥0.5、另一个 <0.5）、binding strengthened/weakened（两者均 ≥0.5 但定量偏移）、binding retained（两者均 ≥0.5 且变化可忽略）、no confident binding（两者均 <0.5）；另设饱和标志（两者均 >0.95），此类情况直接阻止假阳性结合变化标签。
- **调查循环公式**：$a_t = \pi(S_t, \mathcal{T})$，$o_t = T_{a_t}(S_t)$，$S_{t+1} = V(S_t, o_t)$，其中 π 为规划策略，T 为证据工具集，V 为确定性验证器。
- **五模块分工**：State（仅存储事实和轨迹，不做决策）→ Planner（基于当前不确定性选择证据源；可选固定优先级或 LLM 辅助规划）→ Tools（ADASTRA v6.1.1 / JASPAR 2024 / ENCODE cCRE）→ Verifier（对每种证据源实现确定性的多分支决策树）→ Stopping Rule（在规划器返回 ABSTAIN 时给出最终 epistemic status）。
- **证据源细节**：
  - **ADASTRA**：本地 TSV 查找，返回 FDR 校正 p 值、实验数、reads 数、偏好等位基因和 motif 一致性；缺失记录为"无数据"而非"证明无覆盖"。
  - **JASPAR**：等位基因特异性 motif 打分（|Δ_rel_score| > 0.05 判定方向差异），仅为启发式阈值，证据标记为 experimental=False、direct_to_claim=False。
  - **ENCODE cCRE**：tabix 索引 BED 查询，报告变异是否重叠调控元件（pELS/pPLS/CTCF/DNase-only），仅提供背景上下文。
- **Verifier 10 分支决策树（ADASTRA）**：覆盖无数据/显著 ASB（饱和模型 vs 非饱和）/显著 ASB 与预测方向一致或相反/不显著但统计效力充足/不显著且效力不足等情形。
- **证据约束报告器**：默认配置下 LLM 仅用于最终报告生成，接收结构化 facts 块，受 CAN_CLAIM_EXPERIMENTAL 标记约束，不可修改任何 verdict 或重新分类实验事件。

## 实验与结果
- **数据集与位点**：聚焦 rs6983267（chr8:127401060 G>T, GRCh38），8q24 癌症风险位点，调查 4 个 TF（FOXA1, KLF6, RAD21, SP1），所有观察来自真实数据库查询，无模拟数据。
- **FOXA1（rescued，3 步）**：DNABERT 预测 p_ref = 0.995, p_alt = 0.995（饱和 retained），ADASTRA 返回显著 ASB（ref 偏好等位基因，FDR = 0.030, 15 个 ChIP-seq 实验, 475 reads）→ verifier 标记为 contradict DVR，assign rescued（模型假阴性，独立实验证据揭示真实等位基因效应）。
- **KLF6（abstained，8 步）**：DVR 预测 strengthened（p_ref = 0.596, p_alt = 0.915）；ADASTRA 无此 SNP 数据（unavailable）；JASPAR motif MA1517.2 方向相反（Δ = −0.147，alt 降低 motif score）；cCRE 定位至 proximal enhancer-like（pELS）含 CTCF 结合 → 混合间接证据（JASPAR 矛盾 + cCRE 支持）→ abstained，CAN_CLAIM_EXPERIMENTAL = False。
- **RAD21（abstained，8 步）**：饱和 retained（p_ref = 0.999, p_alt = 0.9995）；ADASTRA 返回有统计效力但不显著 ASB（FDR = 0.65, 5 实验, 104 reads）→ verifier 标记为 neutral（不显著不等同证明无效应）；JASPAR 无匹配 motif；cCRE 提供上下文 → abstained。
- **SP1（abstained，8 步）**：饱和 retained（p_ref = p_alt = 0.9995）；ADASTRA 无此 SNP 数据；JASPAR motif 方向虽相反但因非方向性预测而不 adjudicated（unavailable for claim）；cCRE 上下文 → abstained。
- **关键对比**：FOXA1 与 SP1 均以相同饱和 retained 预测进入循环，但 FOXA1 因存在直接实验证据而被 rescued，SP1 因缺乏直接证据而 abstained，体现了观察依赖分支的核心价值。
- **LLM vs 固定规划器**：五个假设（4 个 rs6983267 TF + BACH1/rs2981578）均得出相同 verdict；LLM 规划器以 9 次工具调用替代固定规划器的 13 次（节省 ~31%），LLM 调用平均消耗 ~1200 input tokens + 250 output tokens，约 $0.01/假设。
- **AlphaGenome Atlas 交叉参考**：rs6983267 AVI 评分 0.570；FOXA1 max |Δ| = 0.463（与 ADASTRA 发现一致），KLF6 Δ = −0.240（与 JASPAR 方向一致），均在计算层面提供额外佐证。
- **基线 LLM 表现**：GPT-4o、Claude 4.6 Sonnet、Gemini 2.5 Pro 在相同位点上产生自信但不准确的分析，混淆已知 8q24/MYC 生物学与变异特异性 TF 结合声明，未区分预测与证据，未对不可验证声明 abstain。

## 相关工作脉络
- **DeepVRegulome（Dutta et al., 2025）**：上游 DNABERT 基预测框架，ARGUS 沿用其 458 个模型作为预测引擎；区别在于 DeepVRegulome 仅做预测，ARGUS 解决下游"是否应相信该预测"的解释问题。
- **AlphaGenome Atlas（Cheng et al., 2026）**：预计算 90 亿 SNP 的变体影响预测；ARGUS 将其作为可选交叉参考但不纳入规划器，因其 AVI 评分对全 TF 相同、无法产生观测依赖分支。
- **Agentic Genomics（Corpas et al., 2026）**：综述 agentic AI 在基因组学中的应用；ARGUS 与之定位不同，强调通过架构约束而非自动化流程来解决 LLM 幻觉问题。
- **Lin et al.（2025）**：基准测试显示 LLM 在癌症变体分类中易将弱证据过度分类为显著；ARGUS 的确定性分类器+verifier 架构直接解决了这一故障模式。
- **ADASTRA（Abramov et al., 2021）**：等位基因特异性 TF 结合数据库，ARGUS 作为首选直接实验证据源；论文引用其作为唯一能实现直接证伪的证据来源。
- **JASPAR 2024（Rauluseviciute et al., 2024）**：motif 数据库，ARGUS 作为二级间接证据源；与 ADASTRA 的本质区别在于计算证据无法独立确认或证伪实验 TF 结合。

## 局限性与未来方向
- 评估规模有限：仅五个 TF 假设、两个位点，演示系统行为和轨迹分歧，但非预测精度基准或受控幻觉减少测量。
- 规划策略为固定优先级，未引入信息增益或自适应策略；计划未来扩展为基于期望信息增益的正式停止准则和跨 TF 依赖跟踪的假设图。
- 证伪电池仅限三个证据源；计划扩展至 eQTL（GTEx）、染色质开放性（ATAC-seq）、文献（PubMed）和独立模型预测（AlphaGenome Atlas CHIP_TF）。
- DNABERT 模型存在已知盲点（饱和 + 非核心 motif 位置敏感度不足）；饱和 rescue 路径仅解决最严重情况，系统性地表征模型失效模式仍在进行中。
- ENCODE cCRE 数据从本地索引 BED 文件查询（SCREEN API 不可达），缺少 biosample 特异性活性注释。
- 组织匹配（模型训练上下文 vs 证据来源）、ADASTRA 实验聚合、ref/alt 等位基因标准化等仍是生物学解释的局限。

## 研究启发与可借鉴点
- **确定性-智能体分离架构可直接迁移**到其他需要 LLM 处理科学证据的领域（如药物反应预测、蛋白质功能注释），用 verifer 模块替代 prompt-level 约束来消除幻觉。
- **观测依赖分支设计**（证据充足即停止、不足则继续）比固定 DAG 流程更高效；可借鉴到任何多源证据融合的 agent 系统中，作为减少冗余推理的标准范式。
- **CAN_CLAIM_EXPERIMENTAL 标记模式**可作为通用防过声明工具：在跨模态证据整合中，用数据结构级别的分类标记（experimental/computational、direct/indirect）防止低质量证据被提升为高置信结论。
- **饱和预警+ADASTRA rescue 路径**提供了处理高置信模型假阴性的通用策略模板，可应用于其他存在"预测天花板"效应的序列模型解释任务。
- **LLM 规划 vs 固定规划的成本-效果对比方法**（相同 verifier、不同 action-selection policy）可作为评估 plannner 效率的标准实验设计，避免结论被 LLM 推理质量污染。

## 关键术语表
- **DVR（Deterministic Variant Regulatory）**：基于 DNABERT 的 TF 结合预测框架，输出 ref/alt 等位基因的结合概率并施加确定性分类（LOF/GOF/strengthened/weakened/retained/no binding）。
- **ADASTRA**：allele-specific binding 数据库，整合数千个 ChIP-seq 实验的等位基因特异性 TF 结合数据，提供 FDR 校正 p 值、实验数和 reads 覆盖，是 ARGUS 的首选直接实验证据源。
- **JASPAR CORE**：脊椎动物 PSSM motif 数据库，ARGUS 用于等位基因特异性 motif 打分，作为计算级间接证据（experimental=False, direct_to_claim=False）。
- **ENCODE cCRE**：candidate cis-regulatory elements 注释，标记启动子样、增强子样、CTCF 结合和 DNase-only 区域，提供变异调控上下文但无法直接判定 TF 结合变化。
- **饱和（saturation）**：DNABERT 模型对两个等位基因均预测概率 >0.95 的情况，模型动态范围已达上限，无法分辨等位基因差异，需依赖 ADASTRA 等实验数据进行 rescue。
- **Rescued**：当 ADASTRA 直接实验证据与 DVR 饱和保留预测矛盾时，verifier 赋予的终止状态，表示模型假阴性被独立实验数据纠正。
- **Abstained**：在穷尽可用证据源后仍无法得出 supported/contradicted/rescued 结论时的终止状态，表明该假设在当前证据下无法判定。
- **CAN_CLAIM_EXPERIMENTAL**：报告器使用的布尔标记，仅在存在直接实验证据（experimental=True 且 direct_to_claim=True）时才为 True，防止将计算一致性宣传为实验确认。

## 可复现要素
- **代码**：开源，https://github.com/duttaprat/ARGUS
- **数据集**：ADASTRA Mabel v6.1.1（本地 TSV）、JASPAR 2024 release、ENCODE cCRE（GRCh38, SCREEN）、GTEx（组织表达谱）、ENCODE/ChIP-Atlas ChIP-seq peaks（193M+ peaks across 700 TFs）、AlphaGenome Atlas（SDK v0.9.0, 2026-09-08）
- **模型**：458 个 DNABERT-based TF 结合模型（基于 DeepVRegulome pipeline）
- **LLM 规划器**：Claude Sonnet 4.6（Anthropic API credits 支持）
- **关键超参**：活性结合阈值 = 0.5；饱和阈值 = 0.95；JASPAR motif 方向差异阈值 = |Δ| > 0.05；LLM 规划器工具调用预算 = 3/hypothesis
- **序列窗口**：510bp（以变异为中心）
- **语言/框架**：Python, LangGraph (LangChain)
- **论文未提及**：具体硬件配置、训练数据划分、GPU 型号
