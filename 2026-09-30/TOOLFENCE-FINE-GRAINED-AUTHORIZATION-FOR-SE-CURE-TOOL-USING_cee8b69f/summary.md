---
title: "TOOLFENCE-FINE-GRAINED-AUTHORIZATION-FOR-SE-CURE-TOOL-USING"
source: https://arxiv.org/pdf/2609.37196v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 20:00:00"
field: "LLM Agent 安全"
keywords: ["LLM Agent", "Prompt Injection", "Tool Security", "Fine-grained Authorization", "Provenance Tracking", "Within-tool Attack"]
innovations: ["提出 Capability-level 授权单元，将 judge 决策从 per-call 提升至可复用的能力形状", "确定性监控（零 LLM 开销）+ 运行时能力授予 + 跨会话缓存的三段式防御框架", "参数感知 AgentDojo 切分，首次系统量化 within-tool 攻击防御缺口"]
benchmarks: ["AgentDojo", "Cross-tool Escalation", "Within-tool Hijacking"]
---

# 论文速读：TOOLFENCE: FINE-GRAINED AUTHORIZATION FOR SECURE TOOL-USING LLM AGENTS

## 一句话总结
本文提出 ToolFence，一种针对工具调用型 LLM Agent 的推理时防御框架，通过编译"类型化授权蓝图"+ 确定性监控 + 运行时能力授予，将防护粒度从"工具级"细化到"参数溯源级"，在近乎零安全损耗（整体 ASR 降至 0.20%）的前提下，以 1.63× 运行开销显著优于 CaMeL 的 12.95×，有效抵御跨工具提升与 within-tool 参数劫持两类攻击。

## 研究问题与动机
1. **Within-tool 攻击未被充分防御**：现有工具选择类防御（如 RepeatPrompt、ToolFilter、SecInfer）只能阻止调用未授权工具，但攻击者可复用合法工具并篡改敏感参数（收件人、金额、URL 等），在 Qwen3-max/AgentDojo 上 Tool Filter 的 cross-tool ASR 降至 0–1.74%，而 within-tool ASR 仍高达 2.38%–14.46%。
2. **强隔离代价过高**：CaMeL 通过数据流追踪与能力执行隔离实现强安全保障，但需定制执行流水线，带来 10.67×–12.95× 运行时开销，难以落地。
3. **静态工具过滤损害效用**：预先静态白名单无法预见所有合法能力；被拒正当调用会导致任务中断。
4. **现有防御停留在"内容检测"而非"效果授权"**：输入过滤（DataSentinel、PiGuard）和共识推理（SecInfer）只判断内容是否可疑，未对可执行操作的权限边界进行约束。

## 核心贡献（创新点）
1. **识别并形式化 within-tool hijacking 威胁**：通过参数感知（parameter-aware）AgentDojo 切分，量化了现有防御在参数级劫持上的漏洞。
2. **将授权单元从 per-call 提升到 capability level**：首次提出以"工具+效应类+参数溯源约束"的形状化能力作为一次裁决、多次复用的授权单位。
3. **确定性快路径 + 能力级慢路径的结合**：蓝图匹配由无 LLM 的确定性监控完成（零推理成本），仅在被蓝图覆盖的请求缺失时触发能力授予，使 judge 调用次数减少 43%。
4. **跨会话能力缓存（cross-session capability cache）**：以值无关的 signature 存储已授权能力形状，在不泄露敏感值的前提下摊销 judge 调用成本。
5. **在 AgentDojo 上实现安全–效用–效率三重 SOTA**：Qwen3-max 整体 ASR 0.20%（vs. CaMeL 0.42%）、保留 38.90% 干净效用、运行开销仅 1.63×，显著优于所有基线。

## 方法详解
- **前置编译（Policy Architect）**：在 agent 消费任何外部内容之前，由隔离的 LLM（仅见认证请求 $q$ 与工具 schema）编译得到类型化授权蓝图 $\mathcal{P} = (\mathcal{M}, \mathcal{C})$。其中 $\mathcal{M}$ 为工具清单（标记效应类别与参数敏感标志），$\mathcal{C} = \{c_1,\dots,c_n\}$ 为能力集合。
- **能力元组定义**：$c = (\text{tool}, e, \{p_j \mapsto \beta_j\}, \kappa, r)$，其中 $e \in \{\text{READ, COMMUNICATION, FINANCIAL, ...}\}$，$\beta_j \in \{\text{LITERAL, DERIVED, TEMPLATE, FREE}\}$ 分别表示参数值必须来自用户请求原文、来自某源工具输出、匹配认证模板、或非敏感自由选取；$\kappa$ 为调用预算，$r$ 为可复用标志。
- **确定性监控（Deterministic Monitor）**：对每次 proposed call $a=(\text{tool}, \mathbf{x})$，无需调用 LLM 即可判定：① 能力匹配：存在 $c \in \mathcal{C}$ 覆盖所有参数；② 绑定合规：每个参数值满足其 $\beta_j$（LITERAL 须出现在 $q$ 中，DERIVED 须出现在已执行源工具的输出中，TEMPLATE 须匹配模式且源工具已执行）；③ 预算充足（$\kappa$ 未耗尽或 $r=\text{TRUE}$）。三者均满足则走快路径直接派派。
- **Read Auto-Allow**：所有参数溯源均为 USER 的只读调用自动放行；若任何参数源自模型生成或未授权工具输出，则转交 judge。
- **动态运行时能力授予（Runtime Capability Grant）**：当 call 不命中蓝图时，控制器为每个参数计算溯源标签 $\pi(p_j) \in \{\text{USER, TOOL, MODEL}\}$，并构造候选能力 $\hat{c}$。若任一敏感参数溯源为 MODEL（即模型自主生成），直接拒绝（$\hat{c}=\bot$），防止"注入标准模板"绕开 judge；否则向 judge 提交值无关的形状，judge 输出 GRANT/DENY。GRANT 时将 $\hat{c}$ 加入 $\mathcal{C}$ 并复用；DENY 则回退为 per-call judge 判决。
- **跨会话能力缓存（Cross-Session Cache）**：signature 为 $\operatorname{sig}(c)=(\text{tool}, e, \{p_j \mapsto (\beta_j, \text{source\_tools}_j)\})$，仅存形状不存值。新会话启动时自动注入缓存能力，使同形状能力在跨会话中被确定性放行，同时每次调用的实际值仍须在本会话内重新验证。
- **Fail-closed 安全属性**：任何未经 $\mathcal{C} \cup \mathcal{G}$ 覆盖且 judge 未 GRANT 的动作均不可执行；解析失败的 judge 响应默认 DENY。

## 实验与结果
- **数据集**：AgentDojo benchmark（629 对注入任务），作者构建了参数感知切分：524 对跨工具提升（cross-tool escalation）、85 对 within-tool 劫持、20 对 ambiguous。覆盖 Workspace、Slack、Travel、Banking 四领域。
- **攻击策略**：Direct、Ignore Previous、Important Instructions、InjecAgent、System Message、Tool Knowledge，共 6 种，ASR 为均值。
- **模型**：主实验 Qwen3-max、GPT-4o；附录报告 Mistral-small-3.1-24B。
- **基线**：Repeat Prompt、Spotlighting、Tool Filter、PromptArmor、SecInfer（$K=5$, temperature=0.7）、CaMeL。
- **Qwen3-max 主结果**：ToolFence 整体 ASR=0.20%（No Defense=21.20%），cross-tool=0.10%，within-tool=0.80%，Clean Utility=38.90%，U@A=32.45%。相比 CaMeL（ASR=0.42%，Clean U.=32.67%，Runtime=12.95×）在保持更低 ASR 的同时效用更高、开销仅为 1.63×。
- **GPT-4o 结果**：整体 ASR=0.90%，within-tool=2.10%，Clean U.=82.70%，U@A=73.80%。
- **Mistral-small-3.1-24B 结果**：整体 ASR=0.13%，within-tool=0.60%，Clean U.=55.80%，U@A=52.30%。
- **Ablation（Qwen3-max）**：
  - Blueprint + Deterministic Monitor：ASR=0.45%，U@A=25.60%，Runtime=1.45×（过于保守）；
  - + Runtime Grant：U@A↑至 30.40%，ASR=0.18%，Judge/Task=1.84；
  - + Cache：Judge/Task 降至 1.05（↓42.9%），U@A=31.80%；
  - Full（+ Read Auto-Allow）：U@A=32.45%，ASR=0.20%，Judge/Task=0.82，Runtime=1.90×。
- **失败分析**：①保守拒绝导致效用损失（如用户授权读取文件后发现的 URL 属 TOOL 溯源，无法自动放行 get_webpage）；②当所有参数溯源合法时，provenance-only 无法区分模型在多个合法候选中的偏好是否受注入影响（如 Travel 场景下模型被诱导选择特定餐厅）。

## 相关工作脉络
1. **PromptArmor / DataSentinel / PiGuard**（输入过滤类）：在内容进入模型前标记并移除可疑片段，但无法区分"语义连贯的注入"与"正常业务文本"，对 within-tool 攻击残留高 ASR。本文定位：从内容过滤转为效果授权。
2. **SecInfer**（多路径共识）：通过 K=5 条推理路径聚合决策，但所有路径共享同一被注入的观察，within-tool 攻击沿所有路径复现，共识无法消除。本文定位：不在聚合层面修补，而在执行层阻断敏感参数的非法溯源。
3. **Tool Filter / Spotlighting / Repeat Prompt**（工具级白名单或提示重置）：仅控制工具可见性，不能约束工具内部的敏感参数。本文定位：将授权粒度从 tool-level 细化至 parameter-level provenance。
4. **CaMeL**（数据流隔离+限制执行）：通过程序合成与隔离解释器实现强隔离，但每次调用的数据流分析开销大。本文定位：以"能力形状缓存 + 确定性监控"替代每步流分析，在安全性相近（ASR 0.20% vs 0.42%）下将开销从 12.95× 降至 1.63×。
5. **StruQ / SecAlign / Jatmo**（微调类）：需修改模型参数，难以应用于黑盒/闭源模型。本文定位：推理时防御，无需访问模型权重。
6. **IPI / InjecAgent / ToolHijacker / ASB**（攻击侧工作）：本文在 AgentDojo 上复现并扩展评估，首次显式区分跨工具提升与 within-tool 劫持，并提出针对性防御。

## 局限性与未来方向
1. **无法感知"合法来源内的语义操纵"**：当所有参数均来自已授权溯源时（如餐厅名合法来源于授权 tool output），provenance 监控无法检测模型是否被注入诱导在多个合法候选间做出偏颇选择（Travel 案例）。
2. **保守拒绝损害部分合法任务**：用户授权读取文件后，从文件中提取的 URL 被标记为 TOOL 溯源，后续的 get_webpage 调用需人工确认，增加交互摩擦。
3. **能力授予 prompt 依赖 judge 模型判断**：虽然 judge 调用频率已大幅降低，但在复杂任务中 judge 仍可能出现误判；parser-fail 时默认 DENY 虽安全但可能误杀。
4. **未覆盖非工具型 DoS 攻击**：AgentDojo 中 20 对 ambiguous 样本主要为非工具拒绝服务攻击，本文机制对此类攻击无特殊处理。
5. **跨会话缓存的安全性边界**：虽然 signature 不含值，但若某能力形状在历史会话中被恶意利用过（虽被 judge 拒绝则不会入库），仍需确保 cache 注入不会引入额外风险——论文未深入讨论此场景。

## 研究启发与可借鉴点
1. **"Capability 而非 Call" 的授权粒度**：将一次性 judge 决策转化为可复用的能力形状，大幅摊薄成本——该思路可迁移至任何需要反复执行同类操作的 Agent 安全系统。
2. **确定性监控作为 LLM 决策的前置过滤**：在每次 LLM 调用前先经规则引擎（无 LLM）检查，可在不牺牲安全性的前提下把开销降到接近零——适用于高并发 Agent 部署。
3. **参数溯源标签（USER / TOOL / MODEL）的三元划分**：简单而有效的安全抽象，可推广到文档处理、代码生成等场景中区分"用户意图来源"与"系统/第三方来源"。
4. **跨会话值无关 signature 缓存**：以 $(tool, effect, binding\_type, source\_tool)$ 为 key 的缓存策略，在不泄漏敏感值的前提下复用授权决策——这一设计可直接用于企业内部 Agent 平台的策略管理。
5. **参数感知 benchmark 切分方法**：通过枚举"认证任务–注入任务"的 executable ground-truth 工具调用差异来分类 attack，比自然语言描述更精确——可作为后续安全评测的标准做法。

## 关键术语表
**Within-tool hijacking**：攻击者复用已授权工具但篡改其敏感参数（收件人、金额、URL 等），从而绕过工具级白名单的攻击方式。
**Parameter-aware split**：基于认证任务与注入任务的 executable 工具调用差异，将测试样本划分为 cross-tool escalation / within-tool hijacking / ambiguous 三类。
**Capability**：以 (tool, effect, 参数溯源约束) 为核心形状的授权单元，一次 judge 裁决后可被确定性监控多次复用。
**Provenance**：参数值的来源标签，分为 USER（来自认证请求）、TOOL（来自已执行源工具输出）、MODEL（模型自主生成）三类。
**Deterministic Monitor**：无需调用 LLM 的规则检查器，负责验证每次 tool call 是否满足蓝图中对应能力的绑定与预算约束。
**Cross-Session Capability Cache**：以值无关 signature 存储已授权能力形状的过程级缓存，跨会话复用可降低 judge 调用频次。
**Fail-closed**：系统在无法确认合法性时默认拒绝执行的策略，ToolFence 中体现为解析失败、预算耗尽、或敏感参数溯源为 MODEL 时均 DENY。

## 可复现要素
- **数据集**：AgentDojo（开源），作者额外提供了参数感知切分脚本（见 Appendix C.1 及 C.2）；切分细节见附录 Table 3。
- **代码**：作者声明将发布评估 artifacts 与 safeguards；主代码与 baseline 实现见附录 C.2 的详细描述（vLLM v0.29.0、Mistral tool-call parser）。
- **模型**：Qwen3-max、GPT-4o (gpt-4o-2024-08-06)、Mistral-small-3.1-24B-Instruct-2503。
- **关键超参**：SecInfer 使用 $K=5$、temperature=0.7；judge prompt 输出限制为 JSON with 25-word reason；读操作自动放行（Read Auto-Allow）默认开启。
- **随机种子**：固定种子，报告 95% bootstrap 置信区间（5 seeds）。
