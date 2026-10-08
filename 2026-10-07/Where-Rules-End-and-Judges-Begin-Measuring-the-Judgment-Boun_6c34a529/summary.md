---
title: "Where-Rules-End-and-Judges-Begin-Measuring-the-Judgment-Boun"
source: https://arxiv.org/pdf/2610.07657v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:05:18"
field: "多智能体系统安全"
keywords: ["Multi-Agent Systems", "Prompt Injection", "LLM-as-a-Judge", "Defense-in-Depth", "Guardrail", "Security Evaluation"]
innovations: ["提出DEFER级联流水线首次系统测量MAS中规则与裁判的决策边界", "提出裁判回放技术实现离线跨面板评估", "发现风险评估门存在反选择性且裁判跨域不可迁移"]
benchmarks: ["Agent Security Bench", "InjecAgent", "TAMAS"]
---

# 论文速读：Where-Rules-End-and-Judges-Begin-Measuring-the-Judgment-Boundary-in-Multi-Agent-Systems-Security

## 一句话总结
本文提出 **DEFER**（DEterministic-First Enforcement with Residual judgment）框架，将28项确定性检查与LLM裁判面板组合为级联流水线，首次系统测量了多智能体系统中"规则决策"与"裁判决策"的边界，发现规则可拦截78%的攻击，而相同裁判仅靠自身判断无法降低攻击成功率。

## 研究问题与动机
- **核心问题**：在由多种防御组成的多智能体（MAS）安全流水线中，多少验证工作可由确定性规则完成，剩余部分才需要LLM裁判介入？
- **现有方法不足**：
  - 既有MAS防御多为单类攻击/单通道设计并孤立评估，缺乏跨控制组件的集成流水线。
  - 组合后的最终裁决通常交由LLM裁判，存在被注入内容说服、决策不稳定、审计困难、延迟与成本高四大缺陷。
  - 已有分层设计（如LlamaFirewall、MCP-Guard、CASCADE）仅在单个agent内组合检测器，未覆盖跨工具调用、共享内存与agent间通信的多阶段流水线。
  - AgentTrust等前作虽有"规则vs裁判"的划分思想，但未在有状态的多智能体工作流中测量该边界。

## 核心贡献（创新点）
1. **DEFER级联流水线**：将来自文献的5项成熟防御原则实例化为28项检查，按"最便宜且最确定→最贵且最不确定"排序，将裁判作为最后手段。与已有工作的本质区别在于：本文不提供新的防御机制，而是对已有机制进行跨控制类型的编排，并测量其效果。
2. **边界测量方法论**：引入"判决分率"（judged fraction）作为核心度量指标，以及"裁判回放"（judge replay）技术——从日志重建输入并离线重判，无需重新运行agent即可评估任意裁判面板。
3. **四域多基线实验**：在CyberOps、医疗、金融、法律四个独立领域（共300个攻击变体+35个良性场景），并与Agent Security Bench、InjecAgent、TAMAS三个第三方基准交叉验证，证明规则的"确定性优先"模式具有跨领域通用性。
4. **可复现性保障**：提供完整测试床、所有领域配置、攻击套件、良性场景、逐试验日志，以及单命令重生成全部表格与图表的reproducibility工具；每轮裁判投票均缓存，无需GPU或API即可复现实验结果。

## 方法详解
DEFER将MAS中不受信任内容的入口归为**两个集成表面**：工具编排（tool orchestration）与内存管理（memory management），并在此之上组织五项防御原则，形成28项检查的级联流水线。

**两项集成表面**：
- **工具编排面**：agent→工具调用、工具响应→agent、handoff（阶段间移交）、proposal justification（提案理由）。
- **内存管理面**：共享存储中的写入与读取操作。

**五项防御原则（5个原则28项检查）**：

- **P1 Authorized Interface（授权接口）**：3项检查，全为内容无关规则。工具注册表白名单、响应schema/时序/replay检查、每工具MAC验证、启动时配置哈希快照。
- **P2 Capability Scoping（能力范围界定）**：3项检查。每阶段静态manifest+动作上限；参数校验（通配符、资产关键性、action target vs incident evidence一致性，cosine≥0.5）；输出敏感内容分类器（phase-aware模式+cosine>0.7拒收）。
- **P3 Verified Execution（验证执行）**：11项检查，含4项有状态检查。顺序：handoff校验→operational context→intent chain→cross-incident ledger→replay detection→adaptive consent→risk score→auto-gates（P3.9）→LLM panel→pre-execution hash重验。 consequential proposal进入此路径。
- **P4 Memory Integrity（内存完整性）**：6项检查。schema校验、证据相似度（cosine硬下限0.3，阈值0.5）、metadata校验、漂移检测（距质心≤1.5×最大观测值）、写replay检测、规则矛盾检查。
- **P5 Access-Controlled Isolation（访问控制隔离）**：5项检查。阶段→存储静态策略、字段级脱敏、查询范围限制（cosine≥0.2）、探测模式监控、读取结果sanitize（8条模式去指令化）。

**决策层级**：
- 内容无关规则（13项）：注册表、manifest、access policy、哈希、replay ledger等。
- 内容相关规则（8项）：handoff短语、context/intent规则、风险评分、元数据校验等，可被攻击者通过选择写入内容规避。
- 相似度阈值（6项）：嵌入模型固定后的cosine比较。
- LLM裁判（1项）：唯一基于"判断"的检查。

**级联顺序**：P1→P2→（consequential则P3）→P4（写）/P5（读/写）。每次检查失败则关闭（fail-closed）：拒绝并转入人工审查队列。

**关键公式/逻辑**：
- P2.2 target-in-evidence：`substring(t, e) ∨ cosine(E(t), E(e)) ≥ 0.5`
- P4.2 write similarity：`cosine(E(c), E(e)) ≥ θ(0.5)，硬下限0.3`
- P4.4 drift：`distance(centroid_s, E(c)) ≤ 1.5 × max_seen`
- P3.9 auto-gate：`alignment < 0.2 → deny; alignment > 0.7 ∧ precedent > 0.7 ∧ proportionality > 0.8 ∧ scope < 0.05 → approve`

## 实验与结果
**数据集**：
- 自建套件：4域×75变体=300变体（含35个良性场景），每个变体3次trial。
- 第三方基准：Agent Security Bench（255案例×2 trials）、InjecAgent子集（50案例×3 trials）、TAMAS（250实例×3 trials）。

**主要结果（Qwen3-235B主模型，LOCAL4裁判面板）**：

| 配置 | CyberOps ASR | 四域平均ASR | 良性拒绝率 |
|------|-------------|------------|-----------|
| FLAT（无防护） | 32.0% | 30.0% | 0% |
| JUDGEONLY（仅裁判） | 34.7% | — | 4.5% |
| NOJUDGE（永不裁判） | 1.3% | — | 54.3% |
| **FULL（DEFER）** | **2.2%** | **3.0%** | **9.9%** |

- 规则首次拦截占比：CyberOps中**97.8%**的被拦截攻击由确定性检查最先拦截；四域平均77.9%。
- 同裁判在JUDGEONLY下攻击成功率为34.7%，与无防护的32.0%几乎无差异；加入规则后降至2.2%。
- 裁判仅决定**26.4%**的攻击相关提案（CyberOps），即"判决分率"远低于100%。
- 在ASB上：FLAT ASR=28.8%，DEFER降至**1.8%**；ACL阻止0%注入动作（因注入使用agent有权调用的工具）。
- 在TAMAS上：三类工具类攻击ASR从43.5%降至**21.9%**，其中规则仅最先拦截10/202次。

**最强结果与提升幅度**：
- CyberOps：ASR从32.0%降至2.2%（**降低29.8个百分点**）；JUDGEONLY相比FLAT反而上升2.7个百分点（不显著）。
- 跨域：ASR从30.0%降至3.0%（**降低27个百分点**）。
- P3移除后ASR从2.2%升至16.9%（增幅最大）；P1移除后从2.2%升至4.9%（最小，因仅AP-15变体涉及）。

**裁判行为发现**：
- LOCAL4两两Cohen's κ：0.41~0.65，均值0.51；4/4一致仅39.0%通过。
- 跨域迁移差：Gemma在CyberOps批准75%良性但在Legal仅批准39.7%。
- DIV3L（3线系多样）vs LIN3（同系3）：ASB上DIV3L放行18/405，LIN3放行127/405（弱3.5倍）。

## 相关工作脉络
1. **AgentTrust**（Yang, 2026）：首次将agent动作划分为"词法威胁（规则可判）"和"语义威胁（需裁判）"，结论与本文"边界形状相同"，但本文在有状态多智能体流水线中测量规则承担的安全份额更大。
2. **LlamaFirewall**（Chennabasappa et al., 2025）：报告分类器+推理审计员的单独与组合效果，但仅限单agent内检测器组合，不跨工具/内存/跨agent消息。
3. **MCP-Guard / CASCADE**（Xing et al., 2026; Turgut & Gumus, 2026）：从模式匹配升级到LLM仲裁者检测注入文本，但仍局限于单个agent的文本检测，未覆盖MAS的授权决策流水线。
4. **Progent / FIDES / RTBAS / Datalog Reference Monitors**：在工具边界强制特权和信息流控制，属于"确定性一侧"的工作；本文测量的是两条路径之间的边界宽度而非单边优势。
5. **CaMeL / 设计模式**（Debenedetti et al., 2025）：通过设计换取可证明隔离，牺牲一定灵活性；本文定位在两者之间——确定性检查足够处用规则，不够处用裁判。
6. **InjecAgent / ASB / TAMAS**：第三方攻击基准。本文在此基础上扩展了四域+良性 companion+可开关各防御的配置空间，填补了MAS级评测的空白。

## 局限性与未来方向
- **静态攻击上限**：评估基于静态攻击，自适应攻击者在问题陈述之外，确定性层的安全性为上限估计。
- **单拓扑**：仅测试了host-mediated垂直拓扑，水平拓扑、多租户部署未涉及。
- **合成环境**：工具为stub，内存为合成数据，绝对成功率不可直接迁移到生产环境；配置间比较有效。
- **基线不完整**：JUDGEONLY不强制执行manifest（低估P2）；ACL对所有agent展示全部16工具；未与Progent、CaMeL等外部防御在相同套件上对比；前沿裁判仅用于judged-write arm。
- **覆盖盲区**：AP-3（confused deputy via forged response）无任何主模型发起攻击；AP-14（memory-read injection）在CyberOps中无Qwen3-235B发起。
- **P3.6 replay检测**：按(skeleton)匹配而非canonical action identity，存在语义replay漏洞。
- **未来方向**：适应性攻击评估、水平拓扑扩展、内存写入来源证明（provenance）、更细粒度的replay检测（canonical action identity）、跨域自适应裁判微调。

## 研究启发与可借鉴点
1. **"确定性优先，裁判兜底"的级联架构**可作为MAS安全流水线的通用设计范式，核心经验是：先评估可确定性解决的问题，裁判只处理意图层面的模糊判断。
2. **Judge Replay技术**极具工程价值：记录裁判输入后离线重判，可无限比较不同面板配置而无需重跑agent，大幅降低评估成本，值得在本团队实验中复用。
3. **风险评估门（auto-gate）的"反选择"教训**：基于攻击者可塑造的证据计算的风险评分，会系统性偏向批准攻击提案（90%攻击 vs 36%良性），应改为基于结构化字段而非自由文本计算风险。
4. **裁判多样性远优于同系堆叠**：3个同线系裁判（LIN3）比1个单独裁判弱3.5倍，提示多裁判面板的多样性（lineage diversity）是关键设计参数。
5. **状态必须设过期**：replay检测以action skeleton为key且永不过期，导致第2起良性incident起拒绝率飙升至88%；以1次incident为窗口的过期策略可将良性完成率恢复至35%~69%。

## 关键术语表
- **DEFER（DEterministic-First Enforcement with Residual judgment）**：本文提出的级联安全流水线，28项确定性检查在前、LLM裁判面板在后，按"能确定则确定、不能确定则裁判"排序。
- **Judgment Boundary（裁判边界）**：MAS安全流水线中"规则足以裁决"与"必须依赖LLM判断"的临界点，本文核心测量对象。
- **Judged Fraction（判决分率）**：到达LLM裁判面板的工具提案占全部提案的比例，本文核心评估指标之一。
- **Judge Replay（裁判回放）**：从日志重建裁判输入并离线重判的评估技术，可复现任意裁判面板决策而不需重新运行agent。
- **Injection Channel（注入通道C1–C5）**：攻击者可控内容进入MAS的五个通道：task input、tool response、memory content、handoff、proposal justification。
- **Attacker Tier（攻击者层级）**：T1（内容攻击者）、T2（被 compromising agent）、T3（基础设施攻击者），对应不同能力的对手模型。
- **Integration Surface（集成表面）**：MAS中不受信任内容的两个入口面：工具编排面与内存管理面。
- **Canary（金丝雀值）**：攻击者嵌入每条恶意内容的唯一标识值，用于机械追踪攻击效果是否实现。

## 可复现要素
- **数据集**：自建攻击套件（300变体+35良性场景）已开源；第三方基准ASB、InjecAgent、TAMAS为公开基准。
- **代码/权重**：测试床、harness、四域配置、攻击与良性套件、逐trial日志均在artifact中；实验代码开源：`github.com/shaswata09/DEFER`；所有裁判投票已缓存，无需GPU/API即可重生成全部表格与图表。
- **关键超参**：主agent temperature=0.7（带seed）；裁判temperature=0；cosine阈值0.5（P2.2/P4.2硬下限0.3）；P4.4漂移距离上限1.5×最大观测；P5.3查询cosine下限0.2；P3.9 auto-gate阈值：approve需alignment>0.7∧precedent>0.7∧proportionality>0.8∧scope<0.05；replay检测窗口24h（skeleton key永不过期，实验中显示需改为≤1次incident）；裁判quorum=3/4；本地面板：Mistral-Small-3.2-24B、Gemma-4-31B-it、gpt-oss-120b、Llama-4-Scout-17B-16E。
