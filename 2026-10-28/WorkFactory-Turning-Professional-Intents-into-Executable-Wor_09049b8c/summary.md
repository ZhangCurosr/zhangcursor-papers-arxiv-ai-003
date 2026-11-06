---
title: "WorkFactory-Turning-Professional-Intents-into-Executable-Wor"
source: https://arxiv.org/pdf/2610.37161v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:02:43"
field: "工作代理训练与任务合成"
keywords: ["working agent", "agent task synthesis", "information support graph", "execution-based repair", "supervised fine-tuning", "GDPval-AA", "professional work", "scaling laws"]
innovations: ["从预期输出反向规划信息单元与载体位置的信息支撑图", "独立探针执行诊断驱动、工作保留约束的执行修复循环", "110K任务多职业语料上的工作代理数据可扩展性与U型执行长度规律"]
benchmarks: ["GDPval-AA v2.1", "JobBench", "Workspace-Bench", "OmegaUse-OfficeVal"]
---

# 论文速读：WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training

## 一句话总结
WorkFactory是从职业意图出发，通过信息支撑图（information support graph）反向规划任务所需的信息依赖与位置，再经独立探针执行修复后可执行职业工作环境的框架；基于此构建了112,686个跨612职业的训练任务，用于微调Qwen系列模型得到WorkPilot家族，在GDPval-AA、JobBench等四项专业工作基准上全面超越基座模型，其中WorkPilot-27B在GDPval-AA v2.1达到1612 Elo的局部前沿水平。

## 研究问题与动机
1. **教师轨迹可靠性依赖任务信息完整性**：工作代理的SFT训练需要教师模型给出可靠的示范轨迹，但若任务所需信息缺失或不一致（如供应商报价未注明运费归属），教师就会引入未支撑的假设或中断实质工作，削弱监督质量。
2. **信息供给与待求解工作的边界难以维持**：构建任务环境时既要把必要前提全部暴露在求解器可见输入中，又不能把中间/最终结果直接给出——前者保证可执行，后者保留实质性工作供教师完成。这一边界在跨文件、跨模态、跨来源的信息分布中极易被破坏。
3. **现有合成框架在信息层面的显式规划不足**：GraphForge从真实文件证据图出发编译任务和评分准则，WorkGenesis基于准则做失败归因，而两者均未从"预期输出→所需信息单元→载体位置"这一反向链路显式建模依赖关系，导致材料化后的环境仍频繁出现断裂推导、跨文件不一致等问题。
4. **多尺度工作代理的数据可扩展性缺乏系统刻画**：目前关于专业工作任务规模增加如何影响模型性能、执行长度、工具调用可靠性等关键行为指标的经验证据有限。

## 核心贡献（创新点）
1. **信息支撑图反向规划**：提出图$\mathcal{G}_{\text{info}}=(\mathcal{V}_{\text{info}},\mathcal{V}_{\text{carrier}},\mathcal{E}_{\text{info}},\mathcal{E}_{\text{loc}})$显式表示输出依赖的信息单元、关系类型与载体位置，使"所有前提可见、所有推导留给求解器"的边界在设计阶段就明确——区别于GraphForge/WorkGenesis仅做素材组装和准则审计，本工作从输出反向构造依赖链。
2. **独立探针执行修复循环**：用一个独立于计划和图的探针求解器只基于求解器可见输入尝试任务，由其诊断的blocking缺陷驱动修复，并设工作保留审查约束修复不得改变核心目标/操作/交付物——区别于直接artifact级规划的"构造完即放行"，本方法用执行反馈作为环境质量的最终仲裁者。
3. **110K规模多职业任务语料与数据可扩展性刻画**：构建覆盖612职业、23大类职业组的112,686个任务与教师轨迹；系统刻画训练数据规模对GDPval-AA Elo、平均执行长度、工具调用失败率的非线性影响，发现模型依赖的收益递减、执行长度先增后降等规律。
4. **WorkPilot多尺度模型训练与前沿对标**：在Qwen3.5-4B、Qwen3.6-27B、Qwen3.6-35B-A3B上SFT，WorkPilot-27B在GDPval-AA达到1612 Elo（局部估计，比DeepSeek-V4.1-Flash高12点，较GLM-5.3低34点），WorkPilot-4B在4B尺度超过Apodex-1.1-mini；并在成本-性能图上落在非支配前沿。
5. **组件分析与教师选择消融**：对比artifact级规划、信息级规划、信息级规划+执行修复三个5K条件，证明后两者的互补收益（缺陷率下降、Elo提升81点）；并证明在不同教师（GLM-5.3 vs DeepSeek-V4-Flash）下训练均可显著增益，但增益幅度受教师轨迹风格影响。

## 方法详解

### 3.1 概览
三条流水线：意图合成→信息级规划+材料化→探针修复→教师收集轨迹→SFT。

### 3.2 多源职业意图合成（§3.2、附录A）
三类来源生成candidate intents并经LLM审查入库共享库$\mathcal{L}$：
- **职业分层驱动**：从BLS 2018标准职业分类（SOC）扩展粗粒度职业→情境化细粒度角色，搜索网页填充工作场景与动机。
- **技能文档驱动**：从公开技能目录抽取每项技能的"目的-适用条件-前置-操作"四元组，当技能A的输出能作为技能B的输入时组合为多步工作流（例如报价提取→年度现金流建模→NPV比较→ sourcing推荐）。
- **交付物驱动**：以真实职场交付物为线索，用instruction backtranslation反推合理目标。
LLM审查三维度：内在一致/可行、职业适当性、作为真实职场活动的合理性。

### 3.3 图引导的环境构建（§3.3、附录B）

**信息支撑图定义**：
$$
\mathcal{G}_{\text{info}} = (\mathcal{V}_{\text{info}}, \mathcal{V}_{\text{carrier}}, \mathcal{E}_{\text{info}}, \mathcal{E}_{\text{loc}})
$$
- $\mathcal{V}_{\text{info}}$：信息单元节点，含语义内容、实体与范围、供给状态（supplied / externally acquired / solver-derived / internal to construction）。
- $\mathcal{V}_{\text{carrier}}$：载体节点（user request、reference artifacts、external resources、future deliverables、internal construction requirements）。
- $\mathcal{E}_{\text{info}}$：支撑关系边，记录association/transformation/evidence/constraint四种关联。
- $\mathcal{E}_{\text{loc}}$：位置绑定，把每个信息单元绑定到载体，并标记是否solver-visible。

**规划步骤**：从预期输出反向追溯每步计算/判断所需的信息单元，落到载体与位置上。内部分层约束：所有任务特定的前提/义务必须可通过求解器可见输入获取或推导；仅construction内部约束不暴露。

**载体绑定举例**（供应商选择示例）：
- 输入绑定：需求预测与供应商报价→参考PDF；定价规则→用户请求。
- 输出绑定：供应商NPV分析与sourcing推荐→未来成本对比工作簿。
- 隐藏：construction内部需求不参与求解。

**材料化**：图翻译为artifact级计划$\mathcal{P}=(u,\mathcal{A},\mathcal{D})$（用户请求、工件规格、无环依赖），由environment worker按reuse/adapt/synthesize三类路由一次性生成所有工件；随后程序性检查（文件数/类型）+ LLM内容检查（请求对各参考文件的引用一致性）双重校验，最多3次重试，未通过则丢弃。

### 3.4 基于执行的环境修复（§3.4、附录C）

**诊断**：独立探针求解器只利用当前solver-visible请求与资源（不访问图、计划、历史诊断），尝试完成最低要求交付物；按严重度标注四类发现：
- blocker（阻断完成最小交付物）→触发修复
- correctness（具体任务错误）
- efficiency（流程建议）
- open（非约束风险/提示）
仅blocker触发修复；效率/open仅作记录不强制处理。

**修复与重新验证**：refiner直接编辑请求$u$与现有参考文件内容，约束：
- 文件数量、类型、依赖结构不变；
- 信息支撑图仍为初始构建规范；
- 修复需通过work-preservation审查：核心目标、本质操作、实质性交付物要求保持完整。
每轮修复后重新过construction checks+work-preservation review+探针新执行；上限3轮；未解决blocker则丢弃。

**缺陷八类分类**（附录C.1 Table 6）：Missing evidence / Cross-file inconsistency / Broken derivation / Underspecified request / Leakage or shortcut / Execution feasibility / Self-inconsistency / Factual truth error。

### 3.5 语料与训练数据（§3.5、附录D）

**教师轨迹**：GLM-5.3独立在干净沙盒中执行（不访问图/计划/历史），得到初始rollout；轻过滤剔除重复推理/重复tool call/缺失交付物；对最长5%的rollout再作pairwise rejection sampling（DeepSeek-V4-Flash-0731固定baseline做对比，教师胜出保留；最多4次尝试）。最终94,077任务走GDPval-AA Stirrup harness、18,609走Claude Code harness。

**反污染**：Qwen3-Embedding-8B对训练请求与评测请求做cosine≥0.85与10-gram重叠≥20%双条件flag，并比对文件hash；人工复核未发现评测复用。

**SFT损失**（masked autoregressive）：
$$
\mathcal{T}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{s\in\mathcal{S}}|M_s|}\sum_{s\in\mathcal{S}}\sum_{\ell\in M_s}\log\pi_\theta(a_{s,\ell}\mid H_{s,<\ell})
$$
其中$M_s$为监督assistant token位置；已失败的tool call token span从$M_s$移除，但failure indicator保留在历史中以维持恢复上下文的可解释性。

**训练超参**（附录D Table 9）：4 epochs、256K序列、global batch=32 packed、peak LR=$3\times10^{-5}$ cosine到$2\times10^{-6}$、$\beta_1=0.9,\beta_2=0.95$、weight decay=0.01、clip=1.0、BF16；视觉与aligner冻结仅更新语言模型。

## 实验与结果

### 数据集与基线
- **评估基准**：GDPval-AA v2.1（220任务，9行业44职业）、JobBench、Workspace-Bench、OmegaUse-OfficeVal。
- **基线分组**：frontier（GLM-5.3/DeepSeek-V4.1-Flash/Qwen3.8-Max/GPT-5.6 Sol/Kimi-K3）、open≥300B（DeepSeek-V4-Pro/Flash/GLM-5.2/MiniMax-M3）、open≈30B（Qwen3.8-27B/Apodex-1.1-mini/Nex-N2.5-mini）。
- **评测协议**：GDPval-AA用Stirrup+GLM-5.3/DeepSeek-V4-Flash两锚点Bradley-Terry局部Elo估计（95% bootstrap CI）；其余用Claude Code harness+Claude Sonnet 4.6 judge。每model-task 3次独立rollout取均值。

### 主要结果（Table 2）
| 模型 | GDPval-AA Elo | JobBench | Workspace-Bench | OmegaUse-OfficeVal |
|---|---|---|---|---|
| Qwen3.5-4B → WorkPilot-4B | 661→1375 (+714) | 4.67→39.87 (+35.20) | 26.92→51.30 (+24.38) | 1.71→11.77 (+10.06) |
| Qwen3.6-35B-A3B → WorkPilot-35B-A3B | 879*→1565 (+686) | 28.91→62.35 (+33.44) | 47.12→56.53 (+9.41) | 7.14→21.18 (+14.04) |
| Qwen3.6-27B → WorkPilot-27B | 975→1612 (+637) | 37.81→63.05 (+25.24) | 49.41→59.18 (+9.77) | 10.18→26.61 (+16.43) |

**最强结果**：WorkPilot-27B以1612 Elo位列GDPval-AA局部前沿，接近GLM-5.3(1646)与DeepSeek-V4.1-Flash(1600)；WorkPilot-4B超越Apodex-1.1-mini(1189)与Nex-N2.5-mini(1127)。

### 可扩展性（§4.3、Figure 5）
- **性能递减回报**：35B-A3B从879到1565，前5K任务贡献447点，75K→110K仅+13点；30K即捕获88%增益。
- **执行长度U型**：Q35B-A3B均值从24.87→162.88(2K)→85.90(110K)；触发250上限比例从21.9%→2.7%。
- **工具失败率**：code执行失败16.0%→6.3%；web-search失败22.4%(2K)→10.4%(110K)；web-fetch维持40%-48%。

### 组件分析（§4.4、Table 3，5K条件对比）
| 条件 | 缺失证据 | 跨文件不一致 | 断裂推导 | 未指定请求 | 泄漏/捷径 | 执行可行 | 自相矛盾 | 事实错误 | GDPval-AA |
|---|---|---|---|---|---|---|---|---|---|
| artifact规划 | 9.62% | 32.82% | 24.86% | 7.22% | 1.92% | 10.80% | 3.28% | 4.52% | 1245 |
| +信息图 | 3.64% | 25.74% | 6.86% | 0.62% | 3.36%↑ | 6.64% | 4.46%↑ | 2.56% | 1293 (+48) |
| +执行修复 | 2.02% | 4.84% | 3.26% | 0.28% | 0.84% | 0.48% | 1.74% | 1.30% | 1326 (+33) |

修复使跨文件不一致从25.74%骤降至4.84%，执行可行性缺陷从6.64%→0.48%。

### 教师选择（§4.5、Table 10）
GLM-5.3教师→1483 Elo；DeepSeek-V4-Flash教师→1248 Elo，均显著优于879基座，说明工作增益不绑定特定教师，但轨迹风格（GLM教师更多tool call/交互轮次）影响最终收益。

### 成本-性能（§4.6、Figure 6）
WorkPilot-4B/35B-A3B/27B单位任务推理成本约$0.012/$0.031/$0.396，均落在非支配前沿。

### 构造成本（附录D.2 Table 8）
平均每任务11.656M tokens：信息级规划4.737M + 材料合成3.275M + 执行修复3.644M。

## 相关工作脉络
1. **Synthetic task/environment合成**：SWE-Gym/SWE-smith/R2E-Gym面向软件工程；DIVE/AgentSynth/TaskCraft面向通用agent。WorkFactory面向"专业工作"的异构信息整合与领域专属交付物。
2. **NexForge [40]**：从能力需求翻译为任务指令，再以workspace blueprint实例化；侧重需求→任务文档链路，缺少信息层显式图规划与执行探针修复。
3. **Synthetic Computers [12]**：规划artifact间依赖引导workspace合成；偏重长期生产力模拟而非职业工作可执行性边界。
4. **GraphForge [31]**：在组装的真实文件workspace上构建evidence图→推导任务陈述与evidence-anchored rubric→执行反馈修订。差异：GraphForge以真实文件为基底编译evidence图；WorkFactory从预期输出反向规划跨reuse/synthesis的材料依赖与位置，并通过独立探针约束修复保持"预期工作"。
5. **WorkGenesis [42]**：在检索文件旁合成同伴材料，用rubric-based failure attribution修订请求/材料/rubric。差异：WorkGenesis做rubric审计循环；WorkFactory直接以探针做blocking诊断并约束修复不改变意图。
6. **Agent-World [11]、DIVE [7]**：状态化交互环境与exploration-based合成。WorkFactory聚焦于"工作交付物"而非纯交互探索。

## 局限性与未来方向
1. **纯文本训练限制**：教师GLM-5.3仅支持文本输入，Qwen原生视觉/对齐组件被冻结，训练与评测均在text-only接口下进行；未能直接学习视觉证据的利用。
2. **仅SFT，未做RL**：当前仅用教师轨迹做masked autoregressive SFT，未引入高质量任务+执行反馈的强化学习路线。
3. **信息图显式化的代价**：信息级规划本身是token消耗最大的环节（4.737M/token任务），且会引入leakage与self-inconsistency小幅上升，说明显式化并不自动保证材料化阶段的一致性。
4. **probe修复预算有限**：3轮上限下仍有约0.48%任务因unresolved blocker被弃用；对于极复杂任务可能收敛不足。
5. **成本估算不含judge/工具/沙箱/编排开销**：Figure 6的非支配结论仅在"任务求解模型推理成本"口径成立。
6. **未来方向**：作者明确提及的多模态专业任务扩展、RL训练路线，以及更细粒度的执行反馈（beyond blocking repair）利用。

## 研究启发与可借鉴点
1. **信息支撑图设计范式**：将"输出→信息单元→关系类型→载体位置→可见性"五元建模形式化，可作为其他领域（科学报告生成、法律案例分析、医疗病历整理）构建可执行工作任务的通用模板。
2. **探针执行修复的"解耦验证"思路**：把环境构造与执行验证拆开，用独立、不可见计划的探针求解器暴露blocking缺陷，再以work-preservation约束引导修复——这种"构造-验证-修复"三段式可直接迁移到代码仓库合成、多智能体协作环境构建等场景。
3. **执行长度U型曲线观察**：小数据时模型倾向过度探索（长度上升+高频触上限），大数据后模型学会更高效执行（长度下降+失败率降低）。这一模式可作为评估agent训练数据效率的新指标，不限于GDPval-AA。
4. **教师选择敏感度**：相同task set、GLM vs DeepSeek教师导出1483 vs 1248 Elo的差距（+235点），提示数据质量不只在"量"而在"轨迹风格分布"；未来可考虑多教师混合或风格校准。
5. **反污染三件套**：embedding相似度+10-gram重叠+文件hash三重检测，可作任何"合成数据→公开benchmark"场景的标准防泄漏协议。

## 关键术语表
- **Information support graph ($\mathcal{G}_{\text{info}}$)**：显式记录任务预期输出所依赖的信息单元、关系类型（association/transformation/evidence/constraint）及其载体位置的图结构，用于从输出反向规划环境。
- **Carrier**：信息单元的存放位置，分为user request/reference artifacts/external resources/future deliverables/internal construction requirements五种。
- **Solver-visible**：对求解器可见的载体与信息；所有任务特定前提必须进入此类，internal construction requirements保持隐藏。
- **Probe solver**：独立于规划与修复的求解器，仅用solver-visible输入尝试完成任务，负责诊断blocking缺陷与生成质量报告。
- **Blocker finding**：探针报告中等别最高的缺陷类别，指阻断完成"最小要求交付物"的问题，只有blocker触发修复。
- **Work-preservation review**：修复后由LLM对照原始意图与初始计划审查，确保核心目标、本质操作、实质性交付物要求未被改变。
- **Rejection sampling (pairwise)**：对最长5%的教师rollout，由judge对比教师与baseline deliverable，胜者保留；任务最多4次教师尝试，全败则弃用。
- **GDPval-AA v2.1 Elo**：基于GLM-5.3与DeepSeek-V4-Flash两锚点的Bradley-Terry局部估计Elo，经三个校准节点映射到v2.1官方量表。

## 可复现要素
- **数据集**：作者公开部分任务数据与模型权重（GitHub: https://github.com/IQuestLab/WorkFactory）；全文未声明完整112,686任务开放。
- **代码/权重**：WorkPilot-4B/27B/35B-A3B模型权重随Apache 2.0许可开源（以Qwen基座许可为前提）；构建pipeline代码未明确开源路径。
- **关键超参**：4 epochs、序列长256K、batch=32、peak LR=$3\times10^{-5}$、cosine至$2\times10^{-6}$、warmup=10 iter、$\beta_1=0.9,\beta_2=0.95$、weight decay=0.01、梯度裁剪=1.0、BF16；视觉与aligner冻结。
- **模型配置**：intent/规划/材料化/修复均用DeepSeek-V4-Flash-0731；教师为GLM-5.3；探针/teacher上下文窗口分别为1M/256K tokens，无compaction（Stirrup用官方机制，Claude Code threshold=0.7）。
- **探针参数**：sampling temperature=1.0, top-p=1.0；修复预算3轮。
- **训练任务数**：112,686 tasks；94,077走Stirrup、18,609走Claude Code。
