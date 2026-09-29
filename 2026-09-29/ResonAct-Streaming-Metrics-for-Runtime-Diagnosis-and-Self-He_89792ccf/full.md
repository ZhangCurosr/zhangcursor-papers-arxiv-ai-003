# ResonAct: Streaming Metrics for Runtime Diagnosis and Self-Healing in Multi-Agent Systems

Tarun Chintada<sup>1,2∗</sup>, Neelamadhav Gantyat<sup>1</sup>, Ishaan Romil<sup>1,3∗</sup>, Renuka Sindhgatta<sup>1</sup>, Soujanya Soni<sup>1</sup>, Sameep Mehta<sup>1</sup>

<sup>1</sup>IBM, India

<sup>2</sup>Indian Institute of Technology Patna, India

<sup>3</sup>International Institute of Information Technology Hyderabad, India

tarunchintada1@gmail.com, neelamadhav@in.ibm.com, ishaan.romil@research.iiit.ac.in, {renuka.sr, soujanya.soni}@ibm.com, sameepmehta@in.ibm.com

## Abstract

Multi-agent systems (MAS) are increasingly used to automate enterprise workflows involving multiple specialized agents, external tools, and long-running task execution. Failures may arise from tool degradation, context propagation errors, coordination breakdowns, or repeated agent interactions that prevent task completion. While existing observability frameworks provide traces and logs, diagnosis and remediation are largely performed after execution completes, limiting opportunities for recovery during runtime. We present ResonAct, a runtime self-healing framework that enables continuous monitoring, diagnosis, and remediation of multi-agent systems through streaming operational metrics. ResonAct ingests execution traces, agent interactions, and tool invocations into a streaming analytics layer that continuously derives task progress, context health, and tool reliability metrics. These metrics serve as runtime control signals for detecting anomalous execution patterns and localizing root causes using a structured failure model. Based on the diagnosed failure, ResonAct dynamically selects remediation policies and performs actions. The framework operates as an external control plane, enabling intervention without modifying application agents or orchestration logic. We evaluate ResonAct across enterprise workflow scenarios and AppWorld benchmarks. The results show that the streaming metric-based analysis identifies execution degradations and localizes faults. Furthermore, policy-driven remediation improves task completion rates by up to 10.00 percentage points, with detection precision ranging from 70.59% to 82.91%, recall from 63.09% to 100%, recovery rates from 10.48% to 46.67%, and runtime overhead ranging from −0.25% to 14.12% across the evaluated configurations.

## Introduction

Large Language Model (LLM)-based multi-agent systems (MAS) are increasingly being used to automate complex workflows, such as software development, tool orchestration, and data-intensive tasks (Qian et al. 2024; Luo et al. 2026). By coordinating specialized agents with external tools over extended execution horizons, these systems can perform tasks that require multiple interdependent decisions and actions. However, increasing autonomy also introduces reliability challenges. Failures in MAS can arise not only from conventional software faults, but also from interactions among agents, tool failures, context propagation errors, coordination breakdowns, and deviations from the intended execution trajectory (Cemri et al. 2025; Huang et al. 2025).

Recent empirical work provides evidence that failures in MAS are both diverse and systematic (Cemri et al. 2025). Failures arise due to execution-time behaviors such as step repetition, loss of conversation history, reasoning-action mismatch, task derailment, and premature termination. Existing operational approaches provide mechanisms for observing and recording such executions through traces, logs, and other telemetry (Moshkovich and Zeltyn 2025). However, failure detection and root cause analysis are often carried out after an execution has failed or produced an undesirable outcome (Solomon et al. 2025). This post-hoc paradigm is particularly problematic for long-running enterprise agentic workflows, where anomalous behavior can incur substantial computational cost. Recent work has also shown that agentic workflows can incur substantial and highly variable token consumption as a result of recursive behavior, context growth, and repeated tool interactions, with large cost differences possible even for identical inputs (Bai et al. 2026). Consequently, post-execution analysis can result in unnecessary computational expenditure and increased execution latency. Hence, conventional observability mechanisms primarily provide visibility into system behavior but do not provide a general mechanism for intervening in an ongoing agent execution.

To address these challenges, we present ResonAct (Resonate and Act), a runtime self-healing framework for LLMbased multi-agent systems. ResonAct continuously ingests execution traces, agent interactions, and tool invocations through a streaming analytics layer and derives dynamic health signals, including task progress, context reliability, and tool health. These signals are used to identify emerging execution anomalies and localize potential failure causes. Policy-driven remediation actions are then applied during execution, enabling automated intervention without requiring modifications to individual agent implementations. Reson-Act further separates monitoring, failure analysis, and remediation policies from the application agents and orchestrators through an external control plane.

Contributions: To summarize, we make the following contributions:

(1) We propose ResonAct, a runtime self-healing framework for LLM-based multi-agent systems that enables continuous failure detection, diagnosis, and policy-driven remediation during task execution.

(2) We introduce a streaming metrics-based control mechanism that transforms execution telemetry, including agent interactions, tool invocations, and execution traces, into metrics representing task progress, context health, tool reliability, and execution state. These signals enable detection and localization of emerging failures before task termination.

(3) We develop a policy-driven remediation mechanism that maps diagnosed failure conditions to targeted runtime interventions, enabling the system to recover from execution failures while preserving the underlying task objective.

(4) We evaluate ResonAct across enterprise workflow scenarios and the AppWorld benchmark, measuring failure detection, task completion and recovery, and runtime overhead. Our evaluation shows improved task completion and recovery while maintaining low operational overhead.

## Related Work

We position ResonAct within prior work on multi-agent systems, MAS failure analysis, and runtime observability and self-healing.

## Multi-Agent Systems

LLM-based multi-agent frameworks provide abstractions for agent specialization, communication, task decomposition, and workflow orchestration. Frameworks including AutoGen (Wu et al. 2024), CAMEL (Li et al. 2023), MetaGPT (Hong et al. 2024), CrewAI (Moura and Contributors 2024), and LangGraph (Wang and Duan 2024) support increasingly complex agentic workflows. While these frameworks provide mechanisms for coordinating agents and executing tasks, runtime reliability is generally not their primary focus. In particular, they do not provide general mechanisms for continuously detecting failures, diagnosing their causes, and applying automated recovery actions during execution.

## Failure Analysis and Reliability

Recent work has established that MAS failures are diverse and arise from interactions among agents, system components, and execution state. Multi-Agent System Failure Taxonomy (MAST) provides an empirically grounded taxonomy of 14 failure modes, developed from 150 execution traces and evaluated on 1,600 traces across seven frameworks (Cemri et al. 2025). Complementary work studies failure propagation, fault injection, memory inconsistency, and agent collaboration reliability (Lin et al. 2026; Jia et al. 2026; Wang et al. 2023; Zhang et al. 2024). Collectively, these studies demonstrate that MAS reliability requires systematic failure detection and diagnosis rather than isolated application-level fixes. However, they primarily focus on characterizing, analysing, or injecting failures rather than providing mechanisms for automated runtime recovery.

## Observability and Self-Healing Systems

Observability provides visibility into system behavior through logs, metrics, traces, and events (Silva Fontes, Andrikopoulos, and Yumi Nakagawa 2026). AIOps extends observability with automated fault detection and root cause analysis, demonstrating the value of closed-loop management for deterministic software and infrastructure (Pei et al. 2025). However, LLM-based MAS introduces additional failure modes involving agent reasoning, coordination, context propagation, and tool-mediated execution that are not directly addressed by traditional infrastructure-oriented approaches.

Recent work addresses runtime observability specifically for MAS. AgentOps introduces a pipeline spanning observation, metric collection, issue detection, root-cause analysis, and optimization, highlighting the need for runtime intervention in agentic workflows (Moshkovich and Zeltyn 2025; Al-Sayyad, Huang, and Pal 2026). LumiMAS provides platformagnostic runtime monitoring using an LSTM autoencoder for anomaly detection, followed by failure classification and root-cause analysis using the MAST taxonomy (Solomon et al. 2025). SentinelAgent models MAS execution as a dynamic interaction graph and combines graph-based anomaly detection with an LLM-powered oversight agent for runtime analysis and intervention (He et al. 2025).

Existing work demonstrates the feasibility ofruntime monitoring for MAS, but difers from ResonAct in its detection and intervention mechanisms. ResonAct uses interpretable streaming operational metrics as runtime control signals and does not require model training or per-step LLM judgement.

## The ResonAct Framework

ResonAct operates as an external control plane alongside application agents and their orchestrator, separating runtime monitoring, failure analysis, and remediation policies from task-specific agent reasoning and tool implementations. As illustrated in Figure 1, ResonAct forms a closed control loop comprising: (1) runtime event capture, (2) streaming metric construction, (3) failure detection, (4) stability filtering, (5) remediation selection, and (6) corrective guidance injection. The components communicate through an event stream and can therefore operate independently of the underlying agent framework.

## Failure Taxonomy and Runtime Events Capture

ResonAct builds on existing empirical characterizations of MAS failures (Cemri et al. 2025) and organizes runtime conditions into three operational categories:

• Common Failures: transient errors, timeouts, and retry failures;

• Silent Failures: context decay, state divergence, redundant execution, and other degradations without explicit errors;

• Critical Failures: coordination breakdowns, unbounded loops, and cascading failures threatening task completion or resource bounds;

Additionally, three principal forms of execution evidence: agent interactions, tool invocations, and LLM execution

![](images/15ed485f8812989e092443879f1ef7c5e5e900ad82db95edbdc542af06638033.jpg)  
Figure 1: Architecture of the proposed ResonAct framework.

traces are used. In the current implementation, events are generated through an observability platform<sup>1</sup> and forwarded to an event-streaming platform<sup>2</sup>.

## Streaming Metric Construction

Individual runtime events are often insuficient to determine whether an agent workflow is progressing normally or approaching a failure state. Hence, the event stream is processed to incrementally compute metrics that summarize execution behavior. An execution context c groups the runtime events associated with one attempt to fulfil a user request, including executions that complete, fail, time out, or are otherwise terminated. It defines the scope over which runtime metrics are computed. The metric state for context c at time t is

$$
{ \bf M } _ { c } ( t ) = [ m _ { 1 } ( c , t ) , m _ { 2 } ( c , t ) , \dots , m _ { K } ( c , t ) ] ,\tag{1}
$$

where K is the number of available runtime metrics and each $m _ { j } ( c , t )$ is computed from the events observed for c up to time t.

The metrics are updated incrementally as events arrive and are organized along two axes. The first specifies the unit of analysis: query-level metrics summarize the state of an entire execution context, agent-level metrics characterize an individual agent, and tool-level metrics characterize the reliability and performance of an external tool. The second specifies the computation mechanism: count-based metrics aggregate structured runtime events, embedding-based metrics compare semantic representations, and LLM-as-judge metrics assess properties that cannot be derived directly from structured telemetry. Query-, agent-, and tool-level metrics are updated continuously as runtime events arrive and are evaluated at configurable intervals, such as every x seconds or after a configurable number of execution steps. Table 1 summarizes the metrics used to construct the runtime failure signatures. Metric computation is independent of remediation policy. A metric describes observed execution behavior but does not determine whether an intervention should occur. This separation allows metrics to be reused across failure detectors and enables remediation policies to change without modifying the streaming analytics layer. Detailed metric definitions are provided in the Appendix.

<table><tr><td>Category</td><td>Failure Mode</td><td>Driving Metric</td><td>Detection Rule</td></tr><tr><td rowspan="3">Critical</td><td>agent_loop</td><td>agent_loop_index</td><td>≥ τL</td></tr><tr><td>unbounded_loop</td><td>average_steps</td><td>&gt; P95</td></tr><tr><td>task_incomplete</td><td>task_completion_rate</td><td>&lt; τC</td></tr><tr><td rowspan="3">Silent</td><td>redundant_execution</td><td>unnecessary_path_ratio</td><td>≥ πU</td></tr><tr><td>context_propagation_gap</td><td>delegation_efficiency</td><td>&lt; TD</td></tr><tr><td>decision_repetition</td><td>agent_loop_index</td><td>recurrent</td></tr><tr><td rowspan="3">Common</td><td>tool_degradation</td><td>tool_health / latency</td><td>fail-rate ≥ TT</td></tr><tr><td>agent_degradation</td><td>agent_health / latency</td><td>fail-rate ≥ τA</td></tr><tr><td>API call drift</td><td>call_efficiency</td><td>Moving-Avg. ratio ≥ τP</td></tr></table>

Table 1: Metric-driven runtime failure signatures considered.

## Failure Assessment

At evaluation cycle k, occurring at time $t _ { k } ,$ the Failure Assessment component reads the latest metric snapshot and applies each failure-specific detector $d _ { f } \mathbf { \cdot }$

$$
\mathbf { M } _ { c } ^ { ( k ) } = \mathbf { M } _ { c } ( t _ { k } ) , \qquad d _ { c , f } ^ { ( k ) } = d _ { f } \Big ( \mathbf { M } _ { c } ^ { ( k ) } \Big ) \in \{ 0 , 1 \} .\tag{2}
$$

The evaluation cadence is configurable and may be defined using a wall-clock interval or a specified number of execution steps. The detection conditions are defined in Table 1. If $d _ { c , f } ^ { ( \bar { k } ) } = 1$ , the component constructs the candidate failure signal

$$
F _ { c , f } ^ { ( k ) } = \left( f , c , \mathbf { M } _ { c } ^ { ( k ) } \right) ,\tag{3}
$$

which identifies the failure type and execution context and retains the supporting metric evidence. To suppress transient

violations, a failure must be detected for r consecutive evaluation cycles:

$$
S _ { c , f } ^ { ( k ) } = \prod _ { \ell = 0 } ^ { r - 1 } d _ { c , f } ^ { ( k - \ell ) } .\tag{4}
$$

Thus, $S _ { c , f } ^ { ( k ) } = 1$ only when the failure is detected in the current cycle and the preceding r − 1 cycles. The corresponding signal $F _ { c , f } ^ { ( k ) }$ is then treated as stable and forwarded for further diagnosis. The persistence parameter r controls the trade-of between early intervention and resistance to transient detections.

## Runtime Reflection and Diagnosis

When $S _ { c , f } ^ { ( k ) } = 1$ , the stable failure signal $F _ { c , f } ^ { ( k ) }$ is passed to the diagnosis component. Because an expected execution path is not generally available at runtime, an LLM reflects over the failure signal and available execution evidence, including recent agent interactions, tool outcomes, execution history, and the current metric state. It produces

$$
D _ { c , f } ^ { ( k ) } = ( q , e ) ,\tag{5}
$$

where q characterizes the observed failure pattern and e summarizes the supporting evidence, including likely implicated agents, tools, or execution paths. This output represents an evidence-based interpretation, not a definitive root-cause determination.

## Remediation Decision Engine

The Remediation Decision Engine consumes the stable failure signal $F _ { c , f } ^ { ( k ) }$ and diagnostic output $D _ { c , f } ^ { ( k ) }$ to produce

$$
A _ { c , f } ^ { ( k ) } = \left( b , u , s , g , \rho _ { a } \right) ,\tag{6}
$$

where $b \in \{ 0 , 1 \}$ indicates whether to intervene, u identifies the target agent or tool, s specifies the remediation strategy, g contains the corrective guidance, and $\rho _ { a } ~ \in ~ \{ 0 , 1 \}$ indicates whether a retry is authorized. The engine supports LLM-based and deterministic remediation policies. The LLM-based policy generates a context-specific decision from the diagnostic output, while the deterministic policy maps recognized failure types to predefined, bounded actions. Deterministic remediation is also used when the LLM is unavailable or fails to generate the output. Guidance is published for runtime injection only when $\dot { b } = 1$

## Runtime Guidance Injection

The remediation decision $A _ { c , f } ^ { ( k ) }$ is published as a guidance message on a dedicated remediation stream (when b = 1). An agent-side integration component retrieves guidance associated with execution context c and incorporates it into the applicable agent’s prompt before the next invocation or retry. The guidance may instruct the agent to avoid a failing execution path, invoke an alternative agent or tool, recover missing context, or restrict further retries. ResonAct does not modify an LLM generation already in progress; intervention occurs only at an agent invocation boundary.

<table><tr><td>RQ</td><td>Metric</td><td>Definition</td></tr><tr><td rowspan="2">RQ1</td><td>Recall</td><td>Labelled failed executions detected / all labelled failed execu- tions</td></tr><tr><td>Precision FPR</td><td>Correctly detected failed executions / all detected executions Labelled healthy executions detected as failures / all labelled healthy executions</td></tr><tr><td rowspan="2">RQ2</td><td>Completion Recovery</td><td>Successfully completed tasks / all evaluated tasks Failed executions completed after remediation / detected failed</td></tr><tr><td>Runtime</td><td>executions Relative latency increase over the corresponding baseline</td></tr><tr><td rowspan="2">RQ3</td><td>CPU</td><td>Change in CPU utilization relative to the corresponding base- line</td></tr><tr><td>Memory</td><td>Change in memory utilization relative to the corresponding baseline</td></tr></table>

Table 2: Evaluation metrics for detection, recovery, overhead.

## Experiments

We evaluate ResonAct framework by answering the following three research questions:

RQ1: Detection. Can streaming metrics reliably detect multi-agent failures during task execution?

RQ2: Remediation. Does policy-driven remediation improve task completion and recovery from detected failures? RQ3: Eficiency. Can task reliability improve while maintaining low runtime and resource overhead?

Models: We use two compact, open-weight LLMs, granite-4.1-8b (Granite-8B)<sup>3</sup> and qwen3-8b (Qwen3-8B)<sup>4</sup>, to evaluate the framework. We focus intentionally on 8B-scale models because they are more representative of practical industry deployments, where constraints on latency, inference cost, memory footprint, and on-premises serving often make larger models less feasible. This enables us to assess the efectiveness of the proposed architecture and remediation framework under realistic resource-constrained deployment settings.

## Evaluation Metrics

Table 2 defines the metrics used for RQ1–RQ3. Detection is evaluated against independently labelled execution outcomes. A task is considered recovered only when a detected failed execution completes successfully after remediation.

## Experimental Setup

We present our results on two multi-agent system settings.

AppWorld MAS The original AppWorld benchmark employs a single ReAct agent that independently reasons over tasks and interacts directly with all application APIs within a unified conversation loop. We extend this architecture to a distributed multi-agent system (MAS) comprising an OrchestratorAgent and nine application-specific worker agents. The orchestrator delegates app-related subtasks to specialized agents, each restricted to a single application, provisioned with the required access credentials, and equipped with application-specific API knowledge. Agentto-environment interactions are handled through a traced execution layer that enables eficient tool attribution.

All experiments were performed using the benchmark’s test\_normal evaluation split, providing a standardized and consistent basis for comparing the performance of the MAS and MAS+ResonAct configurations.

Sales MAS The setup consists of ten agents that model an end-to-end B2B sales fulfillment workflow. A Host Agent orchestrates execution by decomposing natural-language requests and routing subtasks to specialized agents (via the A2A protocol), passing intermediate outputs between agents until all requested business functions are completed. The system comprises ten domain-specific agents covering order management, inventory tracking, payments, order fulfillment, warehouse operations, pricing, vendor management, procurement, shipping, and returns, each encapsulating its own tools and business logic and exposed as an independent HTTP service.

The validation dataset is a 600-query ground-truth benchmark spanning all ten domain agents, with queries evenly split across Medium, Dificult, and Complex dificulty tiers. It was deliberately engineered so that approximately 60% of the queries represent failure-oriented stress scenarios by incorporating policy edge cases, missing context, and multiagent dependency chains, making it the canonical evaluation set for measuring system correctness under realistic stress conditions.

MAS with ResonAct To improve robustness, we integrate the ResonAct remediation layer into the AppWorld and Sales use case, where streamed agent events are analyzed to detect persistent failures and automatically generate corrective guidance that is injected into both worker and orchestrator reasoning loops, enabling adaptive self-healing while remaining fully optional and preserving the baseline MAS when disabled.

## Results and Discussion

We evaluate ResonAct along three dimensions: (RQ1) the reliability of its metric-driven failure detector, (RQ2) its efect on end-to-end task completion and recovery, and (RQ3) the runtime cost introduced by continuous diagnosis and remediation. We evaluate detection, remediation, and eficiency end-to-end across Sales MAS and AppWorld.

RQ1: Runtime Failure Detection Detection performance varies across workloads and model backends. On Sales MAS, Granite-8B achieves 76.21% precision and 69.15% recall, while Qwen3-8B achieves 72.11% precision and 63.09% recall. The corresponding F1 scores are 72.51% and 67.30%.

AppWorld has a greater coverage of failures. Recall reaches 100% for both models, with precision of 82.91% for Granite-8B and 70.59% for Qwen3-8B. However, this sensitivity is accompanied by elevated false-positive rates, particularly for Granite-8B (72.97%). These results expose a sensitivity-selectivity trade-of: ResonAct captures a larger fraction of failures on AppWorld but can also intervene on healthy executions.

![](images/d449fe388c61d48f6a6cc53a474dfaab43e8f1306ac33417ccdcefb40a2e4cc4.jpg)  
Figure 2: Operational failure-type distribution across Sales MAS and AppWorld MAS.

Figure 2 further shows that failure composition difers substantially across workloads. Agent degradation and agent loops constitute the dominant combined failure classes, whereas AppWorld contains a comparatively larger proportion of agent failures. This variation helps explain why a single detector configuration does not behave identically across benchmarks.

RQ1 Summary. ResonAct detects failures across both workloads and model families, with precision ranging from 70.59 - 82.91% and recall from 63.09 - 100%. The variation in FPR indicates that workload-aware detector calibration remains important.

RQ2: Runtime Remediation and Task Completion As shown in Table 3, ResonAct improves task completion across all four benchmark–model configurations. The largest gain occurs on Sales MAS with Qwen3-8B, increasing from 73.67% to 83.67% (+10.00 pp), while Granite-8B improves from 80.83% to 88.00% (+7.17 pp). Recovery among detected failed executions reaches 37.97% and 46.67%, respectively.

AppWorld also shows positive gains. Granite-8B improves from 2.01% to 7.14% (+5.13 pp), with 32.06% recovery, while Qwen3-8B improves from 11.31% to 16.07% (+4.76 pp), with 10.48% recovery. The lower recovery rates indicate that longer, tool-intensive AppWorld trajectories are harder to repair than the Sales workflows.

Post-trigger execution is substantially more reliable. In the Granite-8B Sales run, 43 of 44 initiated top-level retries succeed (97.7%), including 7/7 Medium, 19/20 Dificult, and 17/17 Complex cases. This distinguishes retry success from recovery: the former measures executions where a retry is actually initiated, whereas recovery considers all detected failed executions.

RQ2 Summary. ResonAct improves completion by 4.76– 10.00 percentage points across all evaluated configurations, with stronger recovery on Sales than AppWorld.

RQ3: Runtime and Resource Overhead ResonAct introduces additional computation through monitoring, diagnosis, and remediation. On Sales MAS, mean execution time increases from 22.43 s to 23.70 s for Granite-8B (+5.66%) and from 23.58 s to 24.06 s for Qwen3-8B (+2.04%). On AppWorld, Granite-8B remains essentially unchanged (122.32 s vs. 122.01 s, −0.25%), whereas Qwen3-8B increases from 143.54 s to 163.82 s (+14.12%).

<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Model</td><td colspan="4">Detection (%)</td><td colspan="2">Completion (%)</td><td rowspan="2">Gain</td><td colspan="2">Recovery Avg Time (s)</td><td rowspan="2">Overhead (%)</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>FPR</td><td>Base ResonAct</td><td>(pp)</td><td>(%) Base</td><td>ResonAct</td></tr><tr><td rowspan="2">AppWorld MAS</td><td>Granite-8B</td><td>82.91</td><td>100.00</td><td>90.65</td><td>72.97</td><td>2.01</td><td>7.14</td><td>+5.13</td><td>32.06</td><td>122.32 122.01</td><td>-0.25</td></tr><tr><td>Qwen3-8B</td><td>70.59</td><td>100.00</td><td>82.76</td><td>41.67</td><td>11.31</td><td>16.07</td><td>+4.76</td><td>10.48</td><td>143.54 163.82</td><td>+14.12</td></tr><tr><td rowspan="2">Sales MAS</td><td>Granite-8B</td><td>76.21</td><td>69.15</td><td>72.51</td><td>53.83</td><td>80.83</td><td>88.00 +7.17</td><td></td><td>46.67 22.43</td><td>23.70</td><td>+5.66</td></tr><tr><td>Qwen3-8B</td><td>72.11</td><td>63.09</td><td>67.30</td><td>51.18</td><td>73.67</td><td>83.67 +10.00</td><td></td><td>37.97 23.58</td><td>24.06</td><td>+2.04</td></tr></table>

Table 3: End-to-end detection, task completion, recovery, and runtime performance. Runtime overhead is computed relative to the corresponding baseline execution time.

![](images/092c0dc4516a00b54bd8c5686a7a3ed68275d07e2eba4e57bd0b1db5f16f4be5.jpg)  
Figure 3: Dashboard reporting failures through ResonAct.

Resource overhead remains modest. Across the Sales runs, aggregate CPU utilisation increases by approximately 4.84%, while memory consumption decreases by approximately 2.55%. For AppWorld with Qwen3-8B, CPU utilisation increases by 4.18 percentage points and RAM utilisation by 2.33 percentage points, while mean physical RAM usage decreases by 3.36%. Thus, the additional cost is driven primarily by longer execution and increased compute activity rather than sustained memory growth.

The overhead should be considered together with the reliability gains: Sales achieves completion improvements of 7.17-10.00 pp with only 2.04–5.66% additional runtime, while AppWorld shows greater model-dependent variation.

RQ3 Summary. Runtime overhead ranges from negligible to 14.12%, while CPU and memory changes remain comparatively modest.

Overall Implications. ResonAct consistently improves task completion while exposing a clear trade-of between detection quality, recovery dificulty, and runtime cost. Recovery is stronger on the structured Sales workload, whereas AppWorld’s longer and more heterogeneous trajectories make both remediation and overhead more variable. The high posttrigger retry success further suggests that failure selection and calibration are currently more limiting than the execution of remediation itself. These findings support workloadaware detector calibration and selective intervention policies that apply additional computation when recovery is likely to improve the final outcome.

## Deployment Plan

ResonAct is integrated with IBM watsonx Orchestrate (WXO)<sup>5</sup> as a technical preview for streaming runtime diagnosis in an enterprise agent platform. The operational motivation is based on an analysis of 904 issues from an internal WXO support repository, including issue descriptions and discussion threads. Approximately 160 issues were mapped to capabilities involving runtime event capture, metric computation, failure detection, or remediation. Recurring classes included agent and flow loops, intermittent tool failures, non-deterministic routing, context-propagation gaps, and incomplete traces (Figure 4). This analysis establishes the operational relevance of the targeted failure classes. In the current detection-only preview, WXO runtime events are streamed through Confluent Kafka<sup>6</sup> to ResonAct’s metricconstruction and failure-detection components. The implemented integration incrementally computes the agent-loop index and per-tool failure rates, enabling potential execution loops and sustained tool degradation to be identified during execution. Figure 3 shows how the resulting signals are surfaced in the technical-preview dashboard.

![](images/9f713a15de3655f2226298796398bb5da9e0c8255cc1b85aded0d28adb5e351f.jpg)  
Figure 4: Distribution of identified operational failure classes

Progression beyond the current preview is planned with a controlled evaluation of remediation efectiveness and operational overhead. In particular, the latency, token consumption, and cost of LLM-based decision generation will be weighed against improvements in task recovery for pilot customers. Because remediation can alter an execution path and impact the task outcome, its usefulness will also be assessed with platform users. The evidence through pilot customers will guide the safeguards of the deployment.

## Conclusion

We presented ResonAct, an external control plane that derives continuous operational metrics from multi-agent execution events and uses them to drive failure-specific runtime interventions. Its streaming architecture enables detection and remediation at agent invocation boundaries without embedding control logic in individual agents. Integration with the agent runtime platform provides a technically grounded path to controlled operational deployment.

## References

AlSayyad, A.; Huang, K. Y.; and Pal, R. 2026. AgentTrace: A Structured Logging Framework for Agent System Observability. arXiv:2602.10133.

Bai, L.; Huang, Z.; Wang, X.; Sun, J.; Mihalcea, R.; Brynjolfsson, E.; Pentland, A.; and Pei, J. 2026. How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks. arXiv:2604.22750.

Cemri, M.; Pan, M. Z.; Yang, S.; Agrawal, L. A.; Chopra, B.; Tiwari, R.; Keutzer, K.; Parameswaran, A.; Klein, D.; Ramchandran, K.; Zaharia, M.; Gonzalez, J. E.; and Stoica, I. 2025. Why Do Multi-Agent LLM Systems Fail? In Advances in Neural Information Processing Systems, volume 38.

He, X.; Wu, D.; Zhai, Y.; and Sun, K. 2025. SentinelAgent: Graph-based Anomaly Detection in Multi-Agent Systems. arXiv:2505.24201.

Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y.; Wang, J.; Zhang, C.; Wang, Z.; Yau, S. K. S.; Lin, Z.; Zhou, L.; Ran, C.; Xiao, L.; Wu, C.; and Schmidhuber, J. 2024. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. In The Twelfth International Conference on Learning Representations.

Huang, J.-T.; Zhou, J.; Jin, T.; Zhou, X.; Chen, Z.; Wang, W.; Yuan, Y.; Lyu, M. R.; and Sap, M. 2025. On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, 26202–26226. PMLR.

Jia, J.; Deng, Z.; Chen, Z.; Wang, Y.; and Zheng, Z. 2026. MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems. arXiv:2602.19843.

Li, G.; Hammoud, H. A. A. K.; Itani, H.; Khizbullin, D.; and Ghanem, B. 2023. CAMEL: Communicative Agents for ”Mind” Exploration of Large Language Model Society. In Thirty-seventh Conference on Neural Information Processing Systems.

Lin, B.; Yang, K.; Tan, Z.; Lai, Y.; Zhang, C.; Zhang, G.; Yu, X.; Yu, M.; Wang, X.; Zhang, Y.; and Wang, Y. 2026. AgentAsk: Multi-Agent Systems Need to Ask. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 28055–28077. San Diego, California, United States: Association for Computational Linguistics. ISBN 979-8-89176-390-6.

Luo, Y.; Li, G.; Fan, J.; and Tang, N. 2026. Data Agents: Levels, State of the Art, and Open Problems. arXiv:2602.04261.

Moshkovich, D.; and Zeltyn, S. 2025. Taming Uncertainty via Automation: Observing, Analyzing, and Optimizing Agentic AI Systems. arXiv:2507.11277.

Moura, J.; and Contributors. 2024. CrewAI: Framework for orchestrating role-playing, autonomous AI agents. https: //github.com/crewAIInc/crewAI.

Pei, C.; Wang, Z.; Liu, F.; Li, Z.; Liu, Y.; He, X.; Kang, R.; Zhang, T.; Chen, J.; Li, J.; Xie, G.; and Pei, D. 2025. Flowof-Action: SOP Enhanced LLM-Based Multi-Agent System for Root Cause Analysis. arXiv:2502.08224.

Qian, C.; Liu, W.; Liu, H.; Chen, N.; Dang, Y.; Li, J.; Yang, C.; Chen, W.; Su, Y.; Cong, X.; Xu, J.; Li, D.; Liu, Z.; and Sun, M. 2024. ChatDev: Communicative Agents for Software Development. arXiv:2307.07924.

Silva Fontes, G.; Andrikopoulos, V.; and Yumi Nakagawa, E. 2026. Open Source Software Ecosystem for Cloud Observability: An Overview and Trends. In Proceedings ofthe 14th IEEE/ACM International Workshop on Software Engineering for Systems-of-Systems and Software Ecosystems, SESoS ’26, 9–15. New York, NY, USA: Association for Computing Machinery. ISBN 9798400723957.

Solomon, R.; Levi, Y. Y.; Vaknin, L.; Aizikovich, E.; Baras, A.; Ohana, E.; Giloni, A.; Bose, S.; Picardi, C.; Elovici, Y.; and Shabtai, A. 2025. LumiMAS: A Comprehensive Framework for Real-Time Monitoring and Enhanced Observability in Multi-Agent Systems. arXiv:2508.12412.

Wang, J.; and Duan, Z. 2024. Agent AI with LangGraph: A Modular Framework for Enhancing Machine Translation Using Large Language Models. arXiv:2412.03801.

Wang, W.; Dong, L.; Cheng, H.; Liu, X.; Yan, X.; Gao, J.; and Wei, F. 2023. Augmenting Language Models with Long-Term Memory. arXiv:2306.07174.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A. H.; White, R. W.; Burger, D.; and Wang, C. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations. In First Conference on Language Modeling.

Zhang, K.; Peng, L.; Wang, C.; Go, A.; and Liu, X. 2024. LLM Cascade with Multi-Objective Optimal Consideration. arXiv:2410.08014.

## Appendix

## Metric Definitions and Computation

A query/context is identified by its context\_id. Metrics are computed over recent observation windows, with default windows of 20 queries, 50 calls per tool, and 50 calls per worker agent unless otherwise specified.

Agent loop index. For execution context c, with A denoting the set of worker agents, the agent loop index is

$$
L ( c ) = \frac { N _ { \mathrm { r e p e a t e d } } ( c ) } { N _ { \mathrm { c a l l s } } ( c ) } ,\tag{7}
$$

where $N _ { \mathrm { c a l l s } } ( c )$ is the number of worker-agent calls and $N _ { \mathrm { r e p e a t e d } } ( c )$ is the number of calls to an agent after its first occurrence within c. Equivalently,

$$
N _ { \mathrm { r e p e a t e d } } ( c ) = \sum _ { a \in \mathcal { A } } \operatorname* { m a x } ( N _ { a } ( c ) - 1 , 0 ) .\tag{8}
$$

The metric is zero when no worker-agent calls are observed. A high value indicates repeated activation but does not establish that the repetition is erroneous.

Average steps. For query q, execution steps are defined as

$$
S ( q ) = N _ { \mathrm { a g e n t } } ( q ) + N _ { \mathrm { t o o l } } ( q ) ,\tag{9}
$$

where the two terms count worker-agent calls and tool invocations. The window-level average is

$$
\overline { { S } } = \frac { 1 } { | Q | } \sum _ { q \in Q } S ( q ) .\tag{10}
$$

ResonAct uses this metric to identify executions whose step count exceeds the expected workload distribution. The current implementation uses the empirical $P _ { 9 5 }$ as the upper reference for unbounded execution.

## Execution and Path Eficiency

Unnecessary path ratio. The unnecessary path ratio provides a proxy for wasted execution by combining repeated worker-agent calls and failed tool invocations:

$$
U ( q ) = \frac { N _ { \mathrm { r e p e a t e d ~ a g e n t } } ( q ) + N _ { \mathrm { f a i l e d ~ t o o l } } ( q ) } { N _ { \mathrm { a g e n t } } ( q ) + N _ { \mathrm { t o o l } } ( q ) } .\tag{11}
$$

The metric is zero when no counted execution occurs. It is intentionally a heuristic measure: legitimate repeated agent calls may be classified as unnecessary, and repeated successful tool calls do not contribute to the numerator.

Delegation eficiency. Delegation eficiency measures successful worker completion relative to delegation activity:

$$
D ( q ) = \frac { N _ { \mathrm { s u c c e s s f u l ~ w o r k e r } } ( q ) } { N _ { \mathrm { d e l e g a t e d } } ( q ) } .\tag{12}
$$

Here, $N _ { \mathrm { d e l e g a t e d } }$ counts worker-agent delegation events and N<sub>successful worker</sub> counts worker completions with successful outcomes. The metric provides evidence about the efectiveness of information and work transfer across agent boundaries, but does not match individual assignments to completions or independently verify answer quality.

Compactness. Compactness measures execution length relative to the shortest positive-step completed context in the current window. Let

$$
S _ { \mathrm { m i n } } = \operatorname* { m i n } _ { \substack { q \in Q _ { \mathrm { c o m p l e t e d } } , S ( q ) > 0 } } S ( q ) .\tag{13}
$$

For a completed context q with $S ( q ) > 0$

$$
C ( q ) = \frac { S _ { \mathrm { m i n } } } { S ( q ) } .\tag{14}
$$

The value is zero when the context is incomplete, has no steps, or no positive-step completed baseline exists. Compactness therefore measures relative execution eficiency rather than optimality.

API call eficiency. API call eficiency measures recent tool-call volume relative to a longer moving baseline. Let $Q _ { s } \subseteq Q _ { \ell }$ denote the short and long windows of execution contexts, ordered by their most recent update time. Using $N _ { \mathrm { t o o l } } ( q )$ for the number of tool invocations in context q, define

$$
\mu _ { s } = \frac { 1 } { | Q _ { s } | } \sum _ { q \in Q _ { s } } N _ { \mathrm { t o o l } } ( q ) , \qquad \mu _ { \ell } = \frac { 1 } { | Q _ { \ell } | } \sum _ { q \in Q _ { \ell } } N _ { \mathrm { t o o l } } ( q ) .\tag{15}
$$

For $\mu _ { \ell } > 0$ , the metric is

$$
E _ { \mathrm { A P I } } = { \frac { \mu _ { s } } { \mu _ { \ell } } } .\tag{16}
$$

The short and long windows contain up to 5 and 20 contexts, respectively, by default; available contexts are used while the history accumulates. When $\mu _ { \ell } = 0$ , the metric is reported as unavailable.

Values above 1 indicate increased recent tool-call volume relative to the longer baseline, while values below 1 indicate reduced volume. The metric is a drift signal: it does not independently establish task success, monetary cost, or execution optimality. Changes in task dificulty can also change the ratio.

## Agent and Tool Health

Tool health and latency. For tool t, the failure rate over its observation window is

$$
F _ { t } = \frac { N _ { \mathrm { f a i l e d } } ( t ) } { N _ { \mathrm { t e r m i n a l } } ( t ) } ,\tag{17}
$$

with success rate $1 - F _ { t }$ . The latest status is up for a successful completion and down for a failed completion. Tool latency is measured from invocation to terminal completion or failure:

$$
\ell _ { t } = T _ { \mathrm { t e r m i n a l } } - T _ { \mathrm { i n v o c a t i o n } } .\tag{18}
$$

ResonAct considers health and latency jointly because a low latency can also result from a rapid failure.

Agent health and latency. For worker agent a, health is defined from recent completion outcomes:

$$
F _ { a } = \frac { N _ { \mathrm { f a i l e d ~ c o m p l e t i o n s } } ( a ) } { N _ { \mathrm { c o m p l e t i o n s } } ( a ) } .\tag{19}
$$

Agent latency is obtained from the recorded duration of completed worker-agent executions. Health and latency provide complementary signals for detecting sustained agent degradation. The host/orchestrator is excluded from these metrics.

## Semantic and Quality Signals

Tool-routing confidence. For a selected tool $t ^ { * }$ and query q, ResonAct computes semantic similarity between the query and each tool schema using cosine similarity:

$$
s _ { t } = \cos \left( E ( q ) , E ( \mathrm { s c h e m a } _ { t } ) \right) .\tag{20}
$$

The selected-tool confidence is then obtained through a temperature-scaled softmax:

$$
P ( t ^ { * } \mid q ) = \frac { \exp ( s _ { t ^ { * } } / T ) } { \sum _ { t \in \mathcal { T } } \exp ( s _ { t } / T ) } .\tag{21}
$$

The current implementation uses the $\mathtt { a l 1 - M i n i L M - L 6 - v 2 }$ embedding model with a default temperature of $T ~ = ~ 0 . 1$ . The score reflects how strongly the selected tool semantically matches the query relative to the other tools in the registry; it is not a calibrated probability of correctness.

LLM-judge scores. For completed queries, ResonAct optionally uses an LLM judge to score communication quality, planning quality, coordination, and final-answer quality. Communication, planning, and answer quality are independently scored on a 1–5 scale. Coordination is computed as

$$
J _ { \mathrm { c o o r d } } = \frac { J _ { \mathrm { c o m m u n i c a t i o n } } + J _ { \mathrm { p l a n n i n g } } } { 2 } .\tag{22}
$$

Scores may be normalized by dividing by 5. These metrics provide qualitative evidence about execution quality but are model-based assessments rather than independently verified correctness labels.

Agent activation accuracy. Agent activation accuracy evaluates whether the observed ordered agent/tool execution path belongs to a set of accepted paths associated with the query intent:

$$
A = \frac { N _ { \mathrm { u t t e r a n c e s ~ w i t h a c c e p t e d ~ p a t h s } } } { N _ { \mathrm { s c o r e d u t t e r a n c e s } } } .\tag{23}
$$

The implementation groups queries by semantic intent using MiniLM embeddings and a default similarity threshold of 0.75. During bootstrapping, an LLM may provide an optimal reference path. Once suficient observations are collected, frequently observed paths are incorporated into the accepted-path set. The default bootstrap threshold is 50 utterances per intent, and the retained path probability mass is 0.80. Consequently, the metric measures agreement with a learned execution-path reference rather than accuracy against an independently labeled ground truth.

These metrics are computed continuously and aggregated at the context, agent, and system levels.