---
title: "Reading-Right-Acting-Wrong-Diagnosing-Decision-Failures-of-S"
source: https://arxiv.org/pdf/2610.18579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:32:23"
field: "Agent评估与可解释性"
keywords: ["System One models", "agent decision diagnostics", "paired measurement", "reading versus deciding", "behavioral evaluation", "probe sufficiency", "cost-aware routing"]
innovations: ["配对测量框架：在同一决策状态上独立回答动作选择与事实探针，分离理解失败与决策失败", "探针充分性形式化准则：通过∩G(s)≠∅检验探针集是否足以归因决策错误", "三阶段诊断链路：控制状态→真实状态probe-then-decide→decision-oriented probes，定位composition vs. sub-goal缺失"]
benchmarks: ["ALFWorld", "BabyAI-Text", "TextWorld cooking", "τ²-bench", "AgentRM", "SWE-smith coding traces", "Mind2Web"]
---

# 论文速读：Reading-Right-Acting-Wrong: Diagnosing Decision Failures of System One Models in Agent Loops

## 一句话总结
本文针对Agent循环中"System One"快速决策模型的行为诊断，提出配对测量框架：在相同状态下独立回答动作选择与事实探针，揭示强系统的决策错误主要源于"读对但决策错"（而非误读信息），且这类错误集中在特定困难状态上，可通过"先提问再编程组合"的方式修复。

## 研究问题与动机
1. **System One模型的决策黑箱困境**：此类模型一次通过评分候选动作集返回分布，无推理文本可检查，当动作选错时无法判断是"读错了环境信息"还是"读对了却决策失误"。
2. **最终动作准确率掩盖了错误类型**：传统评估仅报告D(s)=1/0，无法区分U=0（误读）与U=1D=0（读对决策错）两类本质不同的失败，前者指向状态表示/上下文预算问题，后者指向决策本身问题。
3. **现有基线缺乏行为诊断能力**：已有Agent基准（ALFWorld、τ²-bench、AgentRM等）仅测任务成功率或单步动作正确率，未将"理解"与"决策"解耦测量。
4. **快速模型的部署可靠性未知**：System One模型强调"per dollar intelligence"，但其在不同域的表现差异大、校准误差跨度达2.6%–26.6%，亟需知道何时可信任、何时需升级。

## 核心贡献（创新点）
1. **配对测量框架（Paired Measurement）**：在每个决策状态s上，独立回答next-action问题与一组factual probes，交叉得到U1D1/U1D0/U0D1/U0D0四类可观测结果——与已有工作仅报告动作准确率的本质区别在于显式分离"理解"与"决策"两个阶段。
2. **65,228条记录的冻结基准（Frozen Benchmark）**：覆盖模拟环境（ALFWorld、BabyAI-Text、TextWorld）、客服策略（τ²-bench）、工具调用、网页导航、代码轨迹六类域，gold标签仅由Agent可见信息计算，所有请求/选项顺序/标签在分析前哈希密封——与公共训练数据过拟合风险的区别在于提供可复现、可扩展的评测基础。
3. **探针充分性判定准则（Probe Sufficiency Criterion）**：形式化定义∩_{s∈S}G(s)≠∅时可从U=1推断决策失败，无需访问模型内部——与行为测试/探测分类器的区别在于仅需环境标签即可完成诊断资格判定。
4. **三阶段诊断实验链路**：先构建"探针充分性由构造保证"的控制状态分离composition vs. misreading，再在真实状态上做probe-then-decide，最后设计decision-oriented probes逼近规划需求——与既有消融实验的本质区别在于每阶段信息递增、可归因。

## 方法详解
1. **决策状态定义**：状态s=(g,h₁:ₜ,R,A(s))，其中g为任务目标，h₁:ₜ为已执行动作与观测历史，R为环境/策略规则，A(s)为完整候选动作集（含名称与描述，无截断），gold集合G(s)⊆A(s)仅由Agent可见信息计算（模拟器中最短路径首动作集合）。
2. **事实探针设计**：闭合问题族，答案由模拟器状态/数据库/数据标签确定，包括feedback outcome（最后动作成败）、current state（位置/持有物）、progress（完成子目标）、constraint（动作是否被允许）、evidence（哪个观测establishes某事实）。每个问题独立回答，不与另一问题的输入/答案合并。
3. **四类联合结果**：Uᵢ=1当且仅当所有适用探针答对；Dᵢ=1当且仅当选定动作∈G(s)。U1D0表示"读对决策错"，U0D1表示"读错但碰巧对"，二者各自不能推出"纯决策失败"或"捷径"（见Table 5）。
4. **探针充分性检验（§2.4）**：固定goal/rules/context，令P(s)为正确探针向量。对共享同一探针向量的状态组S，若∩_{s∈S}G(s)≠∅则探针集充分；两两相交不足，必须检验整个等价类。
5. **诊断实验协议**：Stage 1在原始轨迹上独立回答所有问题；Stage 2用系统自身Stage 1答案拼接成新请求重测决策（绿色路径）；代码组合阶段将模型回答的事实输入确定性规则引擎直接输出动作。
6. **成本与延迟测量**：准确率跑批量请求计时不适用，故设置163项延迟面板（每难度单元1项，严格串行），报告p50/p95；成本按GPU小时或API账单折算。

## 实验与结果
1. **基准规模**：65,228条记录（59,234 test / 5,994 dev），其中19,663条含配对探针；覆盖ALFWorld unseen/seen、BabyAI-Text (BALROG)、TextWorld cooking (TALES levels)、τ²-bench airline/retail、SWE-smith coding traces、AgentRM/Mind2Web离线轨迹等。
2. **评估系统**：17个快速系统（JEV、OpenAI Decisions两个hosted API；Clef-27B/Kev-9B/CLM-8B/Laya/Eikos-27B/Perplexity Decider-27B/SimpleJev-27B/Vega-27B/Torchcast-27B/Quyet-1.0-Large十个open-weight；Cygnet/Decider-chat-31B两个frozen-LLM readout）+ Gemma-4-31B/Qwen3.8-27B/Qwen3.5-9B三个likelihood readout + GPT-6.1 Sol推理参考。
3. **主发现（§5.1）**：对Top-6系统，73–85%的决策错误发生在U=1状态；准确率越高，该占比越大（Kev-9B 49%→Gemma 80.9%→GPT-6.1 85.2%）。条件概率P(D=1|U=1)为0.79–0.90，P(D=1|U=0)为0.53–0.70，互不为代理。
4. **错误结构化（§5.3）**：14个 competent fast systems 产生的53,382次U1D0错误中，85%集中在至少3个系统共同失败的states；全部14系统在同一141个状态上均犯U1D0错误（零假设下期望为0），说明错误附着于状态属性而非模型特异性噪声。
5. **受控消融（§7）**：去除history后ALFWorld强系统跌至14–24%，证明trace被阅读；trace长度>7步时JEV从0.89降至0.48，Gemma稳定在0.72–0.82；决定性观测与决策间的距离（0–15步干扰）对任何系统无显著影响。
6. **诊断实验一：充分探针生成状态（§6.1）**：1,200状态（household + customer service两域各120×4 ladder position），4条rule fact决定动作，4条distractor为干扰。JEV读事实97–100%准确，但trace上决策仅0.791；append正确4 facts升至0.974，append等量无关文本反降至0.751。用自己的答案拼接恢复+17.7点；代码组合达到0.996。GPT-6.1从trace仅0.928→facts-only 1.000，但其86个错误中仅26个U=1，大部分为规则条件误判。
7. **诊断实验二：真实状态probe-then-decide（§6.2）**：2,783配对状态上，benchmark探针问题本身仅+0.6–5.4点，系统自身答案仅+0–9.1点（SimpleJev-27B例外+9.1），正确事实+2.9–10.1点；大多数系统增益被其自身答案破坏2–4%的正确状态抵消。
8. **诊断实验三：decision-oriented probes（§6.3）**：644 ALFWorld状态上约26个问题/状态。JEV事实答对99.7%，但决策-oriented facts仅+5.3点，next sub-goal单块+16.3点，facts+sub-goal共+21.3点至0.929；41%的单独决策错误为U=1但sub-goal错。Clef/Gemma/Qwen3.8-27B受益小，因近半数错误在事实+sub-goal均正确时仍发生（规划失败而非信息缺失）。
9. **校准差异（§7）**：ECE跨度2.6%（Cygnet）至26.6%（Vega-27B）；Gemma likelihood readout均置信度0.98但准确率仅0.79；confidence可驱动cascade但对uncalibrated readout几乎不触发。
10. **成本-准确率前沿（Figure 2）**：JEV $0.13/千状态达0.763；Eikos-27B $0.19/千达0.795；Gemma-4-31B约5倍成本达0.811；GPT-6.1 $642（约73×JEV）达0.884。

## 相关工作脉络
1. **System One模型家族**（TypeSafe AI 2026; OpenAI Decisions 2026; Kwok et al. 2026 CLM; Palmer 2026 Kev; Cloudflare Clef; Convai Laya等）：本文与它们的关系是首个系统性横向比较14个同类choice-only模型的"理解-决策"解耦行为，而非仅报告aggregate score。
2. **Agent基准**（ALFWorld Shridhar et al. 2021; BabyAI Chevalier-Boisvert et al. 2019; TextWorld Côté et al. 2018; τ²-bench Yao et al. 2025; AgentRM Xia et al. 2025; Mind2Web Deng et al. 2023）：本文复用这些轨迹来源但引入paired factual probes，gold仅由agent-visible信息计算，解决已有基准无法区分"误读"与"决策错"的问题。
3. **Knowing-vs-doing研究**（Sobotka et al. 2026; Deb & Krishnan 2026 STOCK-TAKE; Arghal et al. 2026; Schmied et al. 2026）：本文定位差异在于不依赖free-text rationale或hidden state probing，纯behavioral通过closed questions在exact same input上推断competence。
4. **Compositionality gap**（Press et al. 2023; Zhou et al. 2023 least-to-most; Khot et al. 2023 decomposed prompting; Gao et al. 2023 PAL）：本文与之的区别是sub-questions发给同一个choice-only模型并行回答，composition由调用程序完成，符合System One设计范式。
5. **Behavioral testing/故障归因**（Ribeiro et al. 2020 CheckList; Zhang et al. 2025; Liu et al. 2026; In et al. 2026）：本文采用matched minimal pairs与difficulty ladders而非事后explanation faithfulness测量，避免chain-of-thought不忠实问题。
6. **Calibration-aware routing**（Guo et al. 2017; Kadavath et al. 2022; Chen et al. 2024 FrugalGPT; Li et al. 2026 JEV-as-judge）：本文扩展至18个choice interface的native confidence校准对比，证明是否可用confidence做cascade取决于具体readout。

## 局限性与未来方向
1. **纯行为证据局限**：配对结果仅表明输入-输出关联，不揭示内部计算路径；U=1不能证明模型真正"使用"了该事实。
2. **自然状态探针不充分**：benchmark探针为测量理解而设计，未覆盖决策所需全部信息，U1D0混合了决策失败与遗漏事实；充分性仅在构造状态中保证。
3. **标签严格性偏差**：gold集合仅含最短路径首动作，5–42%的 simulator 错误经exact lookahead实为optimal；原始ALFWorld cohort中303/8,197条因co-located receptacle产生location label误差。
4. **部署非架构比较**：系统间serve stack/readout/input limit不同（如Quyet-1.0-Large拒绝>10选项，Vega-27B在长状态上OOM），比较的是deployed choice interface而非model architecture。
5. **单步快照非闭环**：评估冻结traces上的单次决策，未做closed-loop执行；后续研究（§6）仅覆盖部分系统。
6. **推理参考不可比**：GPT-6.1逐问题单独调用并verbalize distribution，其calibration无法与其他系统比较。

## 研究启发与可借鉴点
1. **配对测量框架可直接迁移**：任何"输入相同状态、独立测理解与动作"的场景（如judge模型评估、工具调用选择、robotic policy evaluation）均可套用U/D交叉分类，低成本获得错误归因信号。
2. **探针充分性检验可作为诊断前置条件**：在归因决策失败前，先检查∩G(s)≠∅，避免将信息缺失误诊为推理缺陷；这一形式化条件仅需环境oracle，不需模型权重。
3. **"先提问后编程组合"作为System One部署范式**：对规则明确的任务（§6.1），将wide trace拆成narrow parallel probes + 代码组合可达到0.96–1.00，且confidence cascade可仅在<0.7置信度时触发第二轮，兼顾成本与精度——适用于rule-based customer service、workflow automation。
4. **sub-goal缺失作为开放任务瓶颈信号**：当事实探针补充仅+5点而sub-goal声明+16点时，说明瓶颈在规划而非理解；团队若遇到类似症状可优先引入planner或deliberating model。
5. **校准作为cascade阈值依据需谨慎**：同一准确率下ECE可差10倍（Cygnet 2.6% vs Vega-27B 26.6%），选择路由模型时必须独立验证confidence informativeness。

## 关键术语表
**System One models**：一次性返回候选动作概率分布的快速决策模型，不生成文本推理，典型如JEV、Clef-27B、GPT-6.1 Sol。
**Paired measurement**：在相同决策状态s上独立回答动作选择与事实探针，交叉得U/D四象限结果的评估范式。
**U (measured-fact performance)**：状态级指标，U=1当且仅当所有适用factual probe均答对；不等同于complete understanding。
**D (decision correctness)**：状态级指标，D=1当且仅当选定动作∈G(s)；gold集合仅由agent-visible信息计算。
**Probe sufficiency criterion**：对共享探针向量P(s)的状态组S，若∩_{s∈S}G(s)≠∅则探针集充分，此时U1D0可归因于决策失败。
**Factual probes**：闭合问题族，答案由模拟器/数据库/标签确定，含feedback outcome、current state、progress、constraint、evidence五类。
**Decision-oriented probes**：面向决策规划的问题组（约26问/状态），含location、inventory、last-seen object、processing status、unvisited receptacles、next sub-goal。
**Coverage**：因input limit超配导致问题记为unsupported的比例；低于50%的系统被排除出对应分析。

## 可复现要素
- **数据集**：基准冻结于哈希密封manifest，ALFWorld/BabyAI-Text/TextWorld轨迹由environment expert rollout生成；τ²-bench、SWE-smith、AgentRM、Mind2Web、TOUCAN等公共数据复用；confirmation cohorts从新seed生成。论文未提供官方下载链接，但附录D说明所有raw requests/output distributions/timestamps已保留供复现比较。
- **代码/权重**：14个open-weight系统（Clef-27B、Kev-9B、CLM-8B、Laya、Eikos-27B、Perplexity Decider-27B、SimpleJev-27B、Vega-27B、Torchcast-27B、Quyet-1.0-Large、Cygnet recipe、Decider-chat-31B recipe）均在HuggingFace发布；JEV/OpenAI Decisions为hosted API；GPT-6.1 Sol通过OpenAI API访问。论文未提供统一evaluation harness开源链接。
- **关键超参**：prompt模板与chat format依各模型官方示例；batching策略用于准确率跑，串行单请求用于延迟面板；temperature对System One模型不适用（native distribution），Gemma/Qwen likelihood readout使用temperature=schema-specified。
- **硬件**：本地系统在单H100 80GB上服务；hosted模型外部账单计费。
- **版本锁定**：所有open模型pin至发布revision，JEV pin至jev-1.13.0；五日后rerun显示96.3% agreement on 2,783 states。
