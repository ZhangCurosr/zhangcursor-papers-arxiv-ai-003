# What Coding Agents Leave Unsaid: Measuring Task-State Gaps Between Execution and Reports

Guoyang Xu Rui Zhang Xiuyu Wu Hao Chen

Tencent

nikolaxu@tencent.com

## Abstract

Human supervisors rely on coding agents’ reports to assess progress, verify repairs, and decide when to intervene. But a supervisor cannot investigate a failure they were never told occurred. Failed attempts, workarounds, and unresolved risks may disappear from routine reports, leaving supervisors unaware of issues that warrant scrutiny. We conduct CTXGAP, a measurement study quantifying the gap in taskstate recoverability between agents’ accumulated user-facing reports and their full recorded execution contexts. Across 37 coding tasks and 4,799 valid probes, agents answer 93.6% of task-state questions on demand; full-context readers recover 92.7%, while report-only read ers recover 52.0%, a 40.7-point gap. At the final checkpoints of 30 verified fixes, the gap remains 34.8 points. Human annotation identifies states mentioned without enough supporting detail as the largest category of unrecovered information, and model-assessed supervisionrelevant states retain substantial gaps. Full terminal scrollback improves recovery to 73.6% but still trails full context. Under matched report lengths, context-informed rewrites improve recovery by about eight points, without a detectable targeted-over-natural advantage. Positive gaps persist across harnesses and independent DeepSeek execution and measurement. CTXGAP establishes report-based task-state recoverability as a complementary evaluation dimension. Benchmark and code: https:// github.com/Kaguraaaaaa/Ctxgap.

## 1 Introduction

A supervisor cannot investigate afailure they were never told occurred. Human supervisors increasingly delegate long-running tasks to coding agents, relying on their reports to assess progress, verify outcomes, and decide whether to intervene. During execution, however, agents may encounter failed attempts, adopt workarounds, or deviate from initial task requirements. These events can be consequential for supervision, yet may be omitted or insufficiently explained in subsequent reports. A supervisor—or another language model asked to review the work—may therefore lack the information needed to recognize potential problems, even when the agent reports successful completion.

This raises a fundamental question: how much task-state information recoverable from an agent’s full execution context remains recoverablefrom its user-facing reports? We introduce CTXGAP to quantify this gap using identical task-state probes evaluated against accumulated reports and recorded execution contexts. In one archived Django trajectory, the agent identified a compatibility risk: with certain custom user models, a password change followed by access from another device could trigger HTTP 500 rather than a normal logout. Its accumulated progress reports did not convey this condition. An observer reading the execution context recovered the trigger; an observer reading only the reports did not (Appendix A.3). The agent encountered the condition, but its reporting channel

<table><tr><td>View / setting Accuracy Acc(T) Gap 95% CI</td></tr><tr><td>Main measurement (37 tasks; 4,799 probes) Reports U .520 .927.407[.373, .438]</td></tr><tr><td>Scrollback V .736 .927.191 Actor fork (aux.) Acc(A) = .936 [.929, .944]</td></tr><tr><td>Fixed at k = 4 † .571 .919.348[.282, .413]</td></tr><tr><td>Cross-harness check (12 shared issues)</td></tr><tr><td>Claude Code .525 .932 .407 [.336, .478]</td></tr><tr><td>Pi .550 .940 .390[.289, .491]</td></tr><tr><td>DSH .429 .919.490[.417, .563]</td></tr></table>

Table 1: Execution–report task-state recoverability. U: accumulated reports; V: full terminal scrollback; T: full recorded execution context. Gap is Acc(T) − Acc(view); actor self-answering is auxiliary. <sup>†</sup>Final checkpoints of 30 hidden-test-fixed tasks (1,172 probes). Main estimates use category-weighted probes and issue-cluster bootstrap CIs; cross-harness rows use task-equal means, distinct trajectories/probes, and perissue t intervals. The scrollback gap is a point estimate; reading-view intervals are in Appendix B.

![](images/dcb6866010a9382e4ab6b408ada90b0950a279ceb6d2b0c7a25f6156f441e3dd.jpg)  
Figure 1: CTXGAP overview. Execution-grounded task state can be unrecoverable from accumulated reports; paired probes quantify the 40.7-point main-sample gap between $T _ { \mathrm { f u l l } }$ and U (37 tasks; 4,799 valid probes). Panel 4 illustrates two archived cases—an unreported Django compatibility risk and an earlier pytest verification failure— with original probe judgments U: 0/3 and $T _ { \mathrm { f u l l } } \mathrm { : }$ 3/3 (Appendix A.3). Panels 1–3 illustrate the paired-view measurement; Panel 5 shows the evaluated application settings.

## did not preserve it.

Task success, on-demand answerability, and supervisory visibility are distinct. A coding agent may complete a repair and answer executionspecific questions when asked, yet omit the states that would prompt a supervisor to ask them. Issueresolution benchmarks measure outcomes; they do not reveal how much task state reaches supervisors through routine reports. CTXGAP measures this reporting dimension without requiring supervisors to inspect entire execution logs.

Checkpoint-aligned measurement. At checkpoint k, $U _ { k }$ contains accumulated user instructions and agent reports, while $T _ { \mathrm { f u l l } }$ contains the recorded execution context. CTXGAP extracts trajectorygrounded task-state atoms, constructs probes, and evaluates identical questions with the same observer under each view. Category-aware weighting and independent grading, evidence-support, and cross-model audits test the instrument. Repeated checkpoints trace recovery over time (Figure 1).

Finding 1: routine reports expose far less state than direct queries. Across 37 tasks and 4,799 valid probes, actors answer 93.6% correctly when directly queried at isolated checkpoint forks; a full-context observer recovers 92.7%, but accumulated reports support only 52.0% recovery (a 40.7-point view gap, 95% CI [37.3, 43.8]). The reporting channel leaves much of the state recoverable on demand unavailable to a supervisor who relies on routine reports. At the final checkpoint of 30 hidden-test-verified repairs, the gap remains 34.8 points [28.2, 41.3], with positive task gaps in 28/30 cases.

Finding 2: the gap includes supervisionrelevant states. States independently labeled useful for supervisory decisions by HY4-preview and DeepSeek retain report–context gaps of 35.3 and 36.4 points; their jointly relevant subset retains 36.1 points (Section 4.4). The information deficit is also not limited to completely unmentioned events: two human annotators identify mentioned but insufficiently specified as the largest category among sampled unrecovered states (55.4% and 59.2% weighted estimates). A separate mentiononly intervention provides no detectable recovery gain (∆ = −.002, 95% CI [−.027, .025]). Merely mentioning an event does not communicate enough evidence to answer questions about it.

Finding 3: more terminal visibility reduces but does not close the gap. On the same 4,799 probes, full terminal scrollback V yields 73.6% recovery, versus 52.0% from accumulated reports U and 92.7% from full execution context $T _ { \mathrm { f u l l } }$ . Reading terminal-visible content therefore improves recovery by 21.5 points, but leaves a 19.1-point residual context gap. U represents report-mediated oversight and handoffs; V measures access to the fuller rendered execution history, not what a person necessarily reads or remembers (Section 4).

Reporting and robustness checks. With matched total-report lengths, context-informed natural and targeted rewrites raise recovery by 8.3 and 8.5 points over original reports, without a detectable targeted–natural advantage (Section 5.1). Positive gaps also recur across three harnesses (.390–.490), an independently executing DeepSeek model (.527 on 12 issues), and a fully independent DeepSeek atom-to-grade measurement chain (.398 [.348, .456] on the original 37 trajectories; Section 6).

Our contributions are threefold. (1) A measurement target: task-state recoverability through routine reports, separately from task success and on-demand answerability. (2) A paired-view measurement: identical execution-grounded probes evaluated from reports and full context, with explicit terminal-visibility and relevance checks. (3) Empirical evidence: a large report–context gap persists after successful repairs, for supervisionrelevant states, and even with terminal scrollback; independent execution and measurement checks support its recurrence.

## 2 Related Work

Agent reports and supervisory observability. Kraishan and Jitkajornwanich (2026) analyze 5,851 coding-agent sessions, measuring how executed actions appear in self-reports and how well logs can be reconstructed from them. Wang and Wang (2026) study the difficulty of diagnosing code failures from rendered Blender outputs. Related work examines human–AI context gaps and user correction (Te’eni and Katz, 2026; Tang et al., 2026). Together, these studies motivate assessing the task state conveyed by agents’ reports, beyond action mentions or task outcomes.

Question-based assessment. QAEval and QuestEval assess summaries through question answering (Deutsch et al., 2021; Scialom et al., 2021); FActScore examines atomic factual support (Min et al., 2023), and KnowledgeGain measures reader learning (Soós et al., 2026). AMA-Bench studies memory over agent trajectories (Zhao et al., 2026). CTXGAP uses question answering as a measurement tool rather than proposing a new QA metric: it compares task-state recoverability between paired report and execution views at successive checkpoints, with grading audits informed by Hashemi et al. (2024).

Decision-relevant omissions and reporting interventions. MED-OMIT connects missing facts in medical summaries to simulated downstream decisions (Schumacher et al., 2025). For coding tasks, CTXGAP compares instructions issued from report-only, fact-restored, and matched-known views, while separately evaluating reports under added-text and total-length budgets.

Coding-agent evaluation. SWE-bench measures issue-resolution outcomes (Jimenez et al., 2024); SWE-agent and OpenHands build tool-using software-agent runtimes (Yang et al., 2024; Wang et al., 2024). ContextBench studies code-context retrieval (Li et al., 2026), and AgentLens examines agent trajectories and communication (Podivilov et al., 2026). Harness design also shapes execution (Ning et al., 2026). We measure report–execution recoverability across Claude Code, Pi, and DSH (Anthropic, 2025; Zechner, 2026; DeepSeek-AI, 2026), using each trajectory’s own matched probes.

## 3 Measurement Instrument and Dataset

## 3.1 Checkpoint-aligned task state

We distinguish a canonical task (one unique SWEbench issue ID), a trajectory (one selected execution run ID), a checkpoint, and a probe. The 76-task inventory selects one trajectory per unique issue; its 37-task headline subset therefore contains 37 unique issues, 37 trajectories, and 148 checkpoints (23 development, 14 held-out). The subset was not randomly drawn (Appendix A.4). SWE-bench Verified tasks supply repository issues and hidden tests. Sessions investigate without edits, implement fixes, test and report, then assess risk. We record each trajectory prefix and workspace snapshot, defining the information available at measurement time and enabling later task-outcome checks.

Let $T _ { k }$ denote execution context, including tool interactions, narration, and reasoning; $U _ { k }$ denotes accumulated user instructions and reports. A reading ladder adds narration, execution headers, tool outputs, and terminal scrollback. Complete context includes more than the rendered terminal record. Appendix A specifies views, checkpoints, and collection checks.

## 3.2 Atomic probes and scoring

An atom is a factual proposition grounded in the recorded checkpoint, with a task-state category (e.g., failure, progress, risk, or decision) and an extractor-assigned importance level. The instrument generates a probe and reference answer for each sampled eligible atom (importance 2 or 3). These levels prioritize potentially useful state but are not human-validated decision relevance. The probes refer to each actor’s own execution, so the estimator concerns recoverability of that actor’s state, not a shared exam across agents. A validity filter retains 4,799 of 4,855 probes whose answers are at least partly recoverable from full context but not fully recoverable from the task description. Because the filter uses scored responses, this is a selected analysis population. Atom eligibility, selection effects, identity, and prompt-version audits are specified in Appendix A.2 and A.

An observer answers each probe from its assigned view. A separate judge assigns full, partial, or no credit against the reference, and scores are averaged over repeated samples. To account for category-dependent probe sampling, a probe receives weight

$$
w _ { i } = \frac { N _ { r k c } } { n _ { r k c } } ,\tag{1}
$$

where $N _ { r k c }$ is the number of eligible atoms and $n _ { r k c }$ the number of sampled probes in category c at checkpoint k of run r. Accuracy is the weighted mean of the probe scores. The appendix specifies eligibility, scoring values, and failure handling.

The observer comparison uses the same model and scoring procedure across views. An additional measurement asks the actor to answer from an isolated checkpoint fork, where the harness supports this operation. We define

$$
\mathrm { G a p } _ { \mathrm { a c t o r } } = \operatorname { A c c } ( A ) - \operatorname { A c c } ( U ) ,
$$

$$
\mathrm { G a p } _ { \mathrm { v i e w } } = \operatorname { A c c } ( T ) - \operatorname { A c c } ( U ) .\tag{2}
$$

(3)

Here T denotes the full-context observer and A an actor answering from an isolated checkpoint fork. Our primary outcome is $\mathrm { G a p } _ { \mathrm { v i e w } }$ : the pairedobserver difference between the full-context and report views, used throughout the main study and cross-harness comparisons. The auxiliary $\mathrm { G a p } _ { \mathrm { a c t o r } }$ compares actor self-answering with the report observer where a fork is available; it is used for actorspecific diagnostics, not as the headline estimate. A reasoning-free public-view contrast is specified separately in Appendix A.10.

## 3.3 Conveyance and statistical units

A semantic judge labels atoms as explicit, implicit, or absent in reports. The separately recorded reported\_in\_text field is an extraction provenance flag (whether the atom originates in assistant\_text), not a report-mention detector; it is not used to validate omission. We audit classification precedence and independently annotate report mention and probe-conditioned answer support (Appendix B.4).

Table 4 distinguishes issues, trajectories, checkpoints, and probes. Development/held-out assignment and uncertainty clustering use the canonical issue ID, not the run ID: main-sample and reporting-intervention intervals bootstrap issues, whereas cross-agent intervals use per-issue Student t estimates and issue pairing. The complete interval specifications and sample sizes appear in Appendix A. Hidden tests measure checkpoint task outcomes, while human annotation and alternative scoring examine the probe-based instrument.

## 3.4 Measurement validity and scope

We examine whether the result is sensitive to probe eligibility, reference phrasing, answer grading, and report-evidence classification (Table 2). The separate human audits assess answer scoring and semantic support. Detailed prompt checks and replication scope are in Appendices A.3 and A.10.

Reference phrasing and literal-detail sensitivity. Holding observer answers fixed, a stratified 300-probe reference-paraphrase audit changes the sampled view gap by −.010 (task-cluster 95% CI [−.0417, .0236]), while 18.3% of report-only and 11.7% of full-context item grades change (Appendix E). In 159 probes with concordant labels from two LLM families, 74 labeled non-literal retain a .369 gap [.290, .444] (Appendix K.1).

Changing who asks and who answers. On 413 atom-matched instances from 37 tasks, crossing HY4-preview and DeepSeek as probe generators and observers yields positive gaps in all four cells (.345–.400), under a common answering and GLMgrading protocol. The pattern also holds for 374 pairs independently labeled as probing the same state. A separate observer-only replication likewise retains positive gaps. The full factorial protocol and its scope appear in Appendix K.

<table><tr><td>Component</td><td>Validity evidence</td><td>Study scope</td></tr><tr><td>Probe eligibility</td><td>4,799 of 4,855 retained; filter changes  $\mathrm { G a p } _ { \mathrm { v i e w } }$  by +0.56 pp</td><td>Eligibility determined after full-context scor- ing</td></tr><tr><td>Answer scoring</td><td>Alternative rubric: +1.1 pp  $[ - 1 . 3 , + 3 . 6 ] ;$  dual-rater machine-human difference: +1.39 pp  $[ - 1 . 3 7 , + 4 . 5 4 ]$ </td><td>Rubric and dual-rater audits on separate sam- ples</td></tr><tr><td>Conveyance validity</td><td>195 stratified atoms: mention  $\kappa = . 6 3 2 .$  , sufficiency  $\kappa = . 6 3 8$ </td><td>Mentioned but insufficient: 55.4% / 59.2% of weighted gap mass</td></tr></table>

Table 2: Validity evidence for probe eligibility, answer scoring, and report conveyance. Brackets indicate 95% taskcluster CIs; the dual-rater contrast uses task-equal aggregation. Full protocols appear in Appendices G and A.10.

## 4 RQ1: Magnitude, Persistence, and Anatomy

## 4.1 Reports leave substantial state unrecovered

Main estimate and success-conditioned recovery. The primary, category-weighted probe-level view gap is $\mathrm { G a p } _ { \mathrm { v i e w } } = . 4 0 6 6 [ . 3 7 3 3 , . 4 3 8 1 ]$ (37 tasks, 4,799 valid probes; Table 1). Objective hiddentest outcomes label 30 tasks as fixed at checkpoint 4. On their final-checkpoint probes $( n = 1 , 1 7 2 )$ reports recover .5705 versus .9185 from full context: $\mathrm { G a p } _ { \mathrm { v i e w } } = . { \bf 3 4 7 9 } [ . 2 8 1 6 , . 4 1 2 6 ]$ . The gap is positive for 28/30 tasks. Across all checkpoints of these eventually fixed tasks, the gap is .4118 [.3716, .4478]; earlier checkpoints do not necessarily represent completed fixes. The final checkpoint follows the protocol’s explicit residual-risk assessment, making this a measured gap after successful repair. Expanded-cohort results appear in Appendix A.4; detailed success and anomaly sensitivities in Appendix A.5.

Report-mediated and terminal-visible recovery. On identical probes, accumulated reports U (6,564 mean characters) recover .5205, full terminal scrollback V (25,477 characters) recovers .7359, and full execution context $T _ { \mathrm { f u l l } }$ recovers .9271. Thus the report–context gap is .4066, while the terminal– context gap remains .1912. Reading full scrollback closes 53.0% [47.6, 57.8] of the report–context gap (Table 14); it does not establish what users actually inspect. A structured summary alone (3,015 characters) recovers .452. These views also differ in content and organization, while matched-total-length experiments separately test reporting alternatives (Section 5.1).

## 4.2 Report evidence, persistence, and normalization

Unrecovered atoms rise across checkpoints (9.4, 19.0, 21.0, 22.3), including 18.9 earlier-turn atoms at checkpoint 4 (Table 15). Meanwhile, normalized actor gap peaks at .472 at checkpoint 2 and falls to .359 at checkpoint 4: counts of unrecovered state and normalized recovery tell different stories (Figure 3).

Facts, failures, and progress contribute most gap mass, followed by environmental state; the remaining categories are in the appendix. The decision relevance of these states is evaluated separately below.

Mentioned, but not answerable. In a Django checkpoint, execution records identify two end-to-end test scripts: fake\_psql.py captures subprocess environment variables, while e2e\_check.py exercises the database-shell path under empty and password-bearing configurations. The report mentions a fake psql test but omits the scripts’ identities and distinct roles. Asked which scripts were created and what each did, the fullcontext observer answers correctly in all three samples; the report-only observer answers none. Both human annotators classify the report as mentioned but insufficiently supported (Appendix A.3).

## 4.3 Mention, support, and recovery

Mention is not recoverability. On 195 stratified gap atoms, two annotators show 85.6% / 81.5% raw agreement $( \kappa = . 6 3 2 / . 6 3 8 )$ . Weighted gapmass estimates place mentioned but insufficient (55.4% / 59.2%) above not mentioned (42.1% / 33.7%); only 2.5% / 3.7% is sufficient yet unrecovered. Mention-based coverage can therefore overstate answer support; confidence intervals and an independent calibration appear in Appendix B.4. Adding mentions alone is not enough. Across 99 human-screened mentioned-but-insufficient candidates, adding a mention produces no detectable improvement over the original report (−.002, taskcluster 95% CI [−.027, +.025]). Adding referenceguided pre-checkpoint evidence instead improves recovery relative to mention-only by +.189 (threeobserver mean; [+.137, +.241]). The evidence additions are reference-guided; their lexical overlap

## Reading-depth measurement · 37 tasks · 4,799 probes · task-bootstrap 95% Cls

![](images/32eff3e1df5c791ee839e757936eab1439a7c976aa2401f14b534a1f049b786d.jpg)  
Mean characters supplied per checkpoint· Purple labels: actor-minus-view gap Recent-tool-call views branch from U; they are not cumulative additions to narration

Figure 2: Recovery by reading depth across 11 views (37 tasks, 4,799 probes). Error bars are 95% task-cluster bootstrap CIs. Purple annotations indicate gaps relative to actor self-answer accuracy (0.936); views need not be nested, and the two rightmost views contain internal execution context. Detailed estimates appear in Table 14.  
![](images/c72c66e9f038186e72ebef18815e2bfbb3218932481c52cff9392ceb48500511.jpg)

![](images/6db4a28032dcdaef32301752f19df2251292b6a881e7763f449820b9945bfd3b.jpg)  
Figure 3: RQ1 information-transfer diagnosis. (a) Stratified inverse-probability-weighted gap-mass estimates from two independent annotators (195 atoms), with task-cluster bootstrap 95% CIs. (b) Unaligned atom counts grow across checkpoints while normalized $\mathrm { G a p } _ { \mathrm { a c t o r } }$ is non-monotonic. Automatic semantic classes are detailed in Appendix B.3.

with the probe answers is analyzed in Appendix D.

## 4.4 Recoverability of decision-relevant state

The headline instrument measures executionsupported facts, not only facts a supervisor would need. To examine this distinction, we independently asked HY4-preview and DeepSeek to label 200 probes sampled across state categories and importance levels as decision-relevant, irrelevant, or uncertain, without showing report-observer answers or gap scores. Among 187 probes with both labels, agreement is 86.6% $( \kappa = . 6 3 7 ) ;$ ; disagreements are retained rather than voted away. The labels measure model-assessed supervisory relevance.

The gap persists within relevant state. On the scored stratified sample, relevant-probe $\mathrm { G a p } _ { \mathrm { v i e w } }$ is .353 [.266, .435] under HY4 labels $( n = 1 5 1 )$ )

and .364 [.259, .463] under DeepSeek labels $( n =$ 135); the subset both label relevant gives .361 [.260, .462] (n = 125). For the latter, Acc(U) = .553 and $\mathrm { A c c } ( T _ { \mathrm { f u l l } } ) = . 9 1 4$ . All intervals exclude zero. Post-stratification to the main probe distribution also yields positive gaps (Appendix C). A substantial gap therefore remains within modelassessed supervisory-relevant state.

## 5 Measurement Applications

## 5.1 Comparing reporting designs

We evaluate reports at fixed checkpoints with the same probes, comparing original reports against full-context-informed rewrites under a matched total-length protocol and, separately, natural, structured, and targeted summary additions. The two budget designs answer different questions.

<table><tr><td>Report</td><td>NI recovery</td><td>nNI</td><td>Overall</td><td>Nall</td><td>Chars</td><td>Ncost</td></tr><tr><td>Reports only</td><td>0.017</td><td>54</td><td>0.512</td><td>120</td><td>8,480.5</td><td>120</td></tr><tr><td>Natural</td><td>0.031</td><td>54</td><td>0.518</td><td>120</td><td>1,573</td><td>120</td></tr><tr><td>Structured</td><td>0.281</td><td>54</td><td>0.597</td><td>120</td><td>3,184</td><td>120</td></tr><tr><td>Targeted (matched)</td><td>0.110</td><td>50</td><td>0.564</td><td>112</td><td>1,460.5</td><td>112</td></tr><tr><td>Targeted</td><td>0.602</td><td>50</td><td>0.686</td><td>112</td><td>5,693.5</td><td>112</td></tr></table>

Table 3: Held-out reporting results. Recovery is the mean over available checkpoint-level scores. Non-inferable (NI) and overall outcomes have different denominators; cost is the median number of report characters.

Matched total-report-length replay (E2). On 129 paired checkpoint units from 37 runs (4,175 valid probes), the original accumulated reports (A), natural rewrites (B), and targeted rewrites (C) achieve Acc = .6716, .7545, and .7566, respectively. Natural and targeted rewrites raise recovery over original reports by 8.29 points [+5.72, +10.42] and 8.50 points [+6.34, +10.58]. The direct targeted–natural contrast is C − B = +0.20 points [−1.48, +1.78], with no detectable advantage for targeting. A tighter 113-unit length-tolerance subset gives +0.19 points [−1.71, +1.99]. Both rewrites are generated from the full recorded execution context; these results quantify the recovery obtainable through counterfactual, context-informed reports at approximately matched total length (Appendix I).

Added-summary comparisons. A separate experiment retains the original reports and appends summaries generated with different instructions (Table 3; Figure 7). Natural summarization changes non-inferable recovery from .017 to .031 (54 units), structured reporting reaches .281, and categorytargeted reporting reaches .602 at greater character cost. Matched-budget targeting achieves .110 on 50 paired units while using fewer median characters (1,460.5 vs. 1,573); its non-inferable recovery advantage is +.077 [.030, .133]. Targets are selected from development-set gap categories without exposing held-out answers.

For the added-text comparison, we match realized addition lengths at complete semantic boundaries while keeping prior reports and probes fixed. At 10% tolerance (47 units, 25 tasks), natural versus targeted recovery is .6728 versus .6967, a +2.39-point difference [+1.05, +3.57]. At prespecified 15% tolerance (79 units, 30 tasks), the difference is +2.52 points [+1.45, +3.49]. These added-text results and the total-length replay involve different baselines and budgets; protocols, coverage, and sampling details appear in Appendices J and J.4.

## 5.2 Exploratory simulated supervision

A model supervisor generates next-step instructions from accumulated reports, reports plus a newly supplied task-state atom, or reports plus a previously known atom. Adding previously unrecoverable state changes instructions more frequently than adding known state (+18.9 points, task-cluster 95% CI [+10.2, +28.0]); evidence of fact-specific uptake is more limited (Appendix H).

## 6 Cross-Harness and Cross-Model Validation

Different harnesses. With HY4-preview fixed as executor, Claude Code (Anthropic, 2025), Pi (Zechner, 2026), and DSH (DeepSeek-AI, 2026) all exhibit report–execution gaps on 12 shared issues (Table 1). Each harness generates its own trajectories and probes (Appendix L).

Independent DeepSeek execution. DeepSeek-v4- flash also runs the complete coding-agent workflow under fixed DSH, on the same 12 issues as HY4-preview. Its own trajectories, tool calls, and reports yield a 52.7-point gap [46.0, 60.9] versus 52.2 [42.8, 61.3] for HY4, using the same observer protocol (Appendix L.2).

Independent DeepSeek measurement. On the original 37 frozen trajectories, a separately built DeepSeek pipeline performs atom extraction, probe/reference generation, observation, and grading. It measures a 39.8-point gap [34.8, 45.6], alongside the frozen HY4 result of 40.7 points. The two pipelines construct different probes; a fixedprobe Judge swap also preserves positive gaps (Appendix K.2).

## 7 Discussion

What supervision can recover. Task completion, on-demand answerability, and report-mediated visibility answer different questions. Actors recover 93.6% of evaluated state on demand, yet accumulated reports support 52.0% recovery. The deficit persists among states labeled relevant to supervisory decisions and is not eliminated by the broader terminal-visible history (73.6% versus 92.7% from full context). Supervisors cannot be expected to anticipate every failure, workaround, or changed assumption worth asking about; reports are one channel through which such issues become discoverable. Reporting recoverability therefore warrants measurement alongside task success, with both report-only and terminal-visible views made explicit.

Evaluating reporting policies. The answersupport audit shows why listing an activity differs from communicating its implications. Contextinformed rewrites improve recovery within approximately matched total lengths, demonstrating room for better communication using the recorded execution evidence. Yet targeted rewriting shows no detectable additional advantage over natural rewriting under this control. CTXGAP provides a common instrument for evaluating such designs on the state they convey, rather than judging a reporting policy by length or style alone.

## 8 Conclusion

CTXGAP reveals a persistent gap between execution-grounded task state and what codingagent reports make recoverable. Direct actor queries reach 93.6% accuracy, yet report-only recovery reaches 52.0%; even among 30 successfully repaired tasks, the final-checkpoint gap remains 34.8 points. The gap extends to states independently labeled relevant to supervision and recurs across execution and measurement settings. Measuring this communication channel alongside task success makes a previously hidden part of agent performance visible and gives reporting strategies a concrete evaluation target.

## Limitations

Measurement construct and model dependence. The primary pipeline uses HY4-preview for atom extraction, probe/reference generation, observation, and much of grading. This can favor executiongrounded detail, including exact identifiers. Human answer-scoring and mention/support audits, a cross-generator–observer comparison, and an independent DeepSeek extraction-to-grading chain test aspects of this dependence. The estimates still measure probe recoverability under specified modelgenerated questions and references; the two chains sample different states. The reference-guided evidence diagnostic is additionally conditioned on answer-bearing text.

Supervision and reporting interventions. Decision relevance and probe answering are modelassessed; simulated instruction changes have limited fact-specific uptake. Checkpoint reporting interventions measure offline recovery rather than human action or code outcomes. Added-text comparisons retain prior reports and impose realizedlength tolerances. The total-length replay selects 129 paired units and uses complete execution context to create both rewrites; its approximately eightpoint gains over original reports are not an autonomous reporting effect. The 19 additional units were not observed and cannot form an all-148 sensitivity analysis. These results motivate humansupervision studies with a distinct evaluation target.

Sampling and transfer. The non-random 37-task Python SWE-bench Verified cohort underrepresents difficult issues from the 76-task inventory.

Harness and executor checks span other trajectories and probe sets, supporting recurrence across the tested settings. The main cohort combines recorded prompt versions v3.0/v3.2; the original v3.0 source revision is unavailable, while logged templates and cached scores support the documented checks.

Reproducibility and interpretation. The anonymous numerical supplement reconstructs statistics from cached answers and grades. Releasing full report and execution texts, especially private reasoning, requires separate review; a reasoning-free public execution view represents a different input condition. Automated semantic non-conveyance, literal report mention, and ongoing unresolved task status are distinct annotations. Further protocol and release details appear in the appendix.

## Ethical Considerations

The study uses public SWE-bench Verified tasks executed in sandboxed repositories. Human annotators assessed probe validity, condition-blinded model answers, and a separate stratified sample of report mention and evidence sufficiency. No behavioral study with end users was conducted. The anonymous score-based replication materials are available for numerical verification; publication of text-complete materials remains subject to privacy and redistribution approval, particularly for reasoning-bearing records. Intervention results concern simulated supervision and task-state communication.

## References

Anthropic. 2025. Claude Code: Best practices for agentic coding. Anthropic Engineering. Accessed 2026- 10-11.

DeepSeek-AI. 2026. DeepSeek Harness: Everything is a plugin. Software repository. Developer preview; accessed 2026-10-11.

Daniel Deutsch, Tania Bedrax-Weiss, and Dan Roth. 2021. Towards question-answering as an automatic metric for evaluating the content quality of a summary. Transactions ofthe Associationfor Computational Linguistics, 9:774–789.

Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. 2024. LLM-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13806–13834. Association for Computational Linguistics.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations (ICLR).

Obada Kraishan and Kulsawasd Jitkajornwanich. 2026. Plans they abandon, reports they author: The narrative layer of autonomous agents. Preprint, arXiv:2609.12205.

Han Li, Letian Zhu, Bohan Zhang, Rili Feng, Jiaming Wang, Yue Pan, Earl T. Barr, Federica Sarro, Zhaoyang Chu, and He Ye. 2026. Contextbench: A benchmark for context retrieval in coding agents. Preprint, arXiv:2602.05892.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, et al. 2026. Code as agent harness. Preprint, arXiv:2605.18747.

Andrey Podivilov, Vadim Lomshakov, Sergey Savin, Matvei Startsev, Roman Pozharskiy, Maksim Parshin, and Sergey Nikolenko. 2026. Agentlens: Productionassessed trajectory reviews for coding agent evaluation. Preprint, arXiv:2607.06624.

Elliot Schumacher, Daniel Rosenthal, Dhruv Naik, Varun Nair, Luladay Price, Geoffrey Tso, and Anitha Kannan. 2025. MED-OMIT: Extrinsically-focused evaluation metric for omissions in medical summarization. In Proceedings of the 4th Machine Learning for Health Symposium, volume 259 of Proceedings of Machine Learning Research, pages 897–922. PMLR.

Thomas Scialom, Paul-Alexis Dray, Sylvain Lamprier, Benjamin Piwowarski, and Jacopo Staiano. 2021. Questeval: Summarization asks for fact-based evaluation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Dominik Soós, Meng Jiang, and Jian Wu. 2026. Knowledgegain: Evaluating and optimizing science news generation for reader learning. Preprint, arXiv:2605.31099.

Ningzhi Tang, Chaoran Chen, Gelei Xu, Yiyu Shi, Yu Huang, Collin McMillan, Tao Dong, and Toby Jia-Jun Li. 2026. How coding agents fail their users: A large-scale analysis of developer-agent misalignment in 20,574 real-world sessions. Preprint, arXiv:2605.29442.

Dov Te’eni and Adi Katz. 2026. Reliable vibe coding: The human-AI context gap in software development. In HHAI 2026: Proceedings ofthe 5th International Conference on Hybrid Human-Artificial Intelligence.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, et al. 2024. Open-Hands: An open platform for AI software developers as generalist agents. Preprint, arXiv:2407.16741.

Yinghao Wang and Cheng Wang. 2026. The observability gap: Why output-level human feedback fails for LLM coding agents. Preprint, arXiv:2603.26942.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems.

Mario Zechner. 2026. Pi: A minimal, extensible coding agent harness. Software repository. Accessed 2026- 10-11.

Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, and Jishen Zhao. 2026. AMA-bench: Evaluating long-horizon memory for agentic applications. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 162781–162809. PMLR.

<table><tr><td>Experiment</td><td>Tasks</td><td>Units</td></tr><tr><td>Main sample</td><td>37</td><td>148 checkpoints</td></tr><tr><td>State probes</td><td>37</td><td>4,799 probes</td></tr><tr><td>Supervision contrast</td><td>72</td><td>284 probe-linked units</td></tr><tr><td>Reporting development</td><td>46</td><td>184 checkpoints</td></tr><tr><td>Reporting held-out</td><td>30</td><td>120 checkpoints</td></tr><tr><td>Cross-harness</td><td>12</td><td>36 trajectories</td></tr><tr><td>Cross-executor</td><td>12</td><td>24 trajectories</td></tr></table>

Table 4: Tasks are unique issue IDs; units differ by experiment. The 76-task inventory selects one run per issue. Cross-harness has three trajectories per issue; cross-executor has two (reusing the DSH–HY4 runs). Splits and clusters use issue IDs.

<table><tr><td>Measurement</td><td>Estimate</td><td>95% CI</td></tr><tr><td>Actor A</td><td>0.936</td><td>[0.929, 0.944]</td></tr><tr><td>Full context T</td><td>0.927</td><td>[0.913, 0.938]</td></tr><tr><td>Reports U</td><td>0.520</td><td>[0.491, 0.551]</td></tr><tr><td> $\mathrm { G a p } _ { \mathrm { a c t o r } }$ </td><td>0.416</td><td>[0.386, 0.444]</td></tr><tr><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>0.407</td><td>[0.373, 0.438]</td></tr></table>

Table 5: Main-sample weighted accuracies and gaps. Main $\mathrm { G a p } _ { \mathrm { v i e w } }$ interval uses 10,000 issue-clustered percentile bootstrap draws of the category-weighted pooled estimator (seed 0); other cells retain their recorded interval protocols.

## A Collection and Measurement Protocol

## A.1 Tasks, instructions, and snapshots

Each SWE-bench Verified instance supplies a fixed repository commit and issue description. The common instruction sequence in Table 6 structures the investigation, implementation, verification, and risk-assessment phases. At each checkpoint, CTX-GAP records the trajectory prefix and stores a workspace snapshot as a git commit object. Prefix fidelity is checked segment by segment against the final trajectory.

Hidden outcome evaluation restores the snapshot, resets test-patch files to their base-commit state, applies the benchmark’s test patch, and executes the FAIL\_TO\_PASS and PASS\_TO\_PASS sets. These tests establish whether the issue is fixed and whether existing behavior regresses at the checkpoint. Individual state atoms use evidence in the recorded trajectory, including commands, outputs, and file contents, as their reference material.

<table><tr><td>Checkpoint</td><td>Instruction</td></tr><tr><td>1</td><td>Investigate; do not modify code</td></tr><tr><td>2</td><td>Implement a fix</td></tr><tr><td>3</td><td>Run tests and report the result</td></tr><tr><td>4</td><td>Assess remaining risk</td></tr></table>

Table 6: Common instruction sequence and protocol parameters.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Minimum atom importance</td><td>2</td></tr><tr><td>Full / partial / no credit</td><td>1 / 0.5 / 0</td></tr><tr><td>Probe answer samples per view</td><td>3</td></tr><tr><td>Measurement thinking mode</td><td>Disabled</td></tr><tr><td>Headline view-gap bootstrap resamples</td><td>10,000</td></tr><tr><td>Confidence level</td><td>95%</td></tr><tr><td>Cross-agent degrees of freedom</td><td>11</td></tr><tr><td>Low-cluster descriptive threshold</td><td>8 tasks</td></tr><tr><td>Verified fork prefixes</td><td>86 / 86</td></tr><tr><td>Fork tool calls / modifications</td><td>0/0</td></tr><tr><td>Split seed Development / held-out tasks</td><td>20261003 46 / 30</td></tr></table>

Table 7: Fixed measurement parameters and reported collection checks.

Actor self-answering forks the checkpoint session into an isolated copy. The collection checks compare the forked message prefix with the recorded one and verify that reasoning is carried forward. The self-answer fork makes no tool calls or workspace modifications. Report and fullcontext observers are fresh readers of their supplied materials. Recorded fork-check counts and protocol parameters appear in Table 7.

## A.2 Atom eligibility, sampling, identity, and freeze boundaries

Recorded importance and interpretive guidance. The frozen records contain an extractor-assigned ordinal importance field. The main sampler includes only scores ≥ 2: importance 2 accounts for 43.3% of eligible atoms and importance 3 for 56.7%; the sampled-probe shares are 36.6% and 63.4%, respectively. Importance is not a humanvalidated measure of actionability. A subsequent audit of original request logs recovered the same importance rule in sampled v3.0 and v3.2 extraction calls: level 3 denotes information whose absence could change a user’s instructions or judgment (e.g., failures, assumptions, remaining risks, constraints, or unexpected actions); level 2 helps explain progress; level 1 is trivial detail. The rubric is grounded in recorded extraction requests; versionspecific prompt conformance is assessed below.

Atomicity and duplication boundaries. Each archived record is an evidence-linked factual proposition with category, local atom ID, provenance, and origin checkpoint. Compound extraction statements may retain multiple related details when required by a single generated question (e.g., a command and its result); the original pipeline cannot be asserted to enforce perfect one-fact-per-record atomicity. The same underlying event may generate different probes within or across checkpoints. Sampling and estimation therefore refer to recorded eligible atom/probe units, not a deduplicated universe of unique real-world events. The archived percheckpoint atom key is (run, k, atom\_id); a probe is indexed by (run, k, qid). We do not collapse two probes merely because their natural-language strings overlap. Any semantic duplicate rate would require a separate adjudication.

Probe sampling and validity. At each $( r , k )$ the sampler draws question/reference pairs about eligible atoms within ten recorded task-state categories, with greater representation of importance-3 records than their underlying share. The frozen main weighting compensates at the $( r , k , c )$ category level with $w = N _ { r k c } / n _ { r k c } ,$ rather than assuming uniform importance sampling. The archived data establish this weighting and sampled rows. Original request logs also preserve the prompts actually sent, but the audit has not established a portable global question-count cap or reconstructed a single shared source-code revision across both cohorts. The 4,855 generated probes precede view answering; the 4,799-valid mask is determined $a f -$ ter the three-sample task-description/full-context grading using $\mathit { \Pi } _ { \overline { { s } } _ { T } } ^ { } > 0$ and $\overline { { s } } _ { \mathrm { p r i o r } } < 1$ (54 and 2 exclusions, respectively). Eligibility is therefore outcome-conditioned.

Longitudinal matching and frozen stages. Atoms have a turn\_origin and may contain an explicit supersedes link. Those fields support provenance/updates but do not alone certify the truth or continuing relevance of a proposition. The longitudinal recovery-loss diagnostic also requires a comparable earlier scored instance $\begin{array} { r l r } { \mathrm { \Omega } ( k ^ { \prime } } & { { } < } & { k ; } \end{array}$ prior $U \ \mathrm { r e c o v e r y } \geq . 5 .$ current U recovery < .5); text identity was checked for its 121 candidate links. Historical unresolved status is not inferred solely from low report-view accuracy; resolution and supersession would require evidence-grounded state adjudication. The 76-task development/heldout registry was frozen on 2026-10-06 (split seed

20261003). The 37-run execution/probe records are archived separately: per-run metadata identifies v3.0 for 23 tasks and v3.2 for 14 tasks, while complete original v3.0 source and an exact global extraction freeze timestamp remain unavailable. The cross-generator analysis selected 442 atoms before DeepSeek probe generation and answering.

## A.3 Logged-request instrument conformance audit

Version provenance and scope. Per-run metadata identify v3.0 in 23 headline tasks (15 development, 8 held-out) and v3.2 in 14 (8 development, 6 held-out). The subsequent source file prompts\_v3.py declares v3.2; the original v3.0 source revision is unavailable, so we audit the requests that were actually logged rather than treating the later file as the universal frozen prompt. The p2 collection batch is the mixed-version within-batch comparison available for a full per-run check: 18 v3.0 and 14 v3.2 runs, 32 in total.

Comparison procedure. Hashing an unmodified request prefix is invalid because it includes rundependent issue text. Instead, the zero-API audit extracts narrow static-template windows from logged requests: the importance rubric and its immediate instructions for atom extraction; the fixed probe instruction preceding injected candidate material; the judge instruction before question material; and, for report-only, prior-only, and fullcontext observers, the instruction between the end of the supplied material and the question list. Each window is compared for every one of the 32 runs. This test addresses the checked prompt skeletons, not all possible dynamic content or endpoint defaults.

<table><tr><td>Checked stage</td><td>Runs checked</td><td>Distinct windows</td></tr><tr><td>Atom extraction</td><td>32</td><td>1</td></tr><tr><td>Probe generation</td><td>32</td><td>1</td></tr><tr><td>Judge</td><td>32</td><td>1</td></tr><tr><td>Observer: reports (U)</td><td>32</td><td>1</td></tr><tr><td>Observer: execution (T)</td><td>32</td><td>1</td></tr><tr><td>Observer: task prior</td><td>32</td><td>1</td></tr></table>

Table 8: Per-run logged-request conformance within the mixed-version p2 batch (18 v3.0, 14 v3.2). The common hash in each row applies only to the extracted static-template window, not to the entire request.

Measurement logic and frozen-estimate check. The checked importance rubric retains the three levels (3: could alter user instructions or judgment; 2:

helps explain progress; 1: trivial), and the logged category, provenance, and supersession fields agree in the sampled original requests. Within $\mathtt { p 2 }$ archived atom-count and checkpoint-probe records and the frozen category-level weighting rule reveal no prompt-version-dependent branch. Recomputing all 4,799 eligible headline probe scores gives $\mathrm { A c c } ( U ) = . 5 2 0 5 0 0 , \mathrm { A c c } ( T _ { \mathrm { f u l l } } ) = . 9 2 7 0 8 1$ , and $\mathrm { G a p } _ { \mathrm { v i e w } } = . 4 0 6 5 8 2 . \mathrm { F o r v } 3 . 0 ( n = 2 , 9 8 1 \mathrm { p r o b e s } )$ , the view gap is .386075; for v3.2 (n = 1,818), it is .438893. These retrospective strata differ in task mix and neither constitute a randomized version comparison nor demonstrate statistical equivalence.

Scope of the conformance check. One common template-window hash at each stage is evidence of agreement for those six static segments, not of byte-identical end-to-end requests or all sourcecode revisions. The v3.2-only additional reading views were not part of this six-stage audit or the primary U versus $T _ { \mathrm { f u l l } }$ contrast. The original v3.0 source and some request-side sampling defaults are unavailable; the audit establishes conformance of the logged template windows and reproducibility of frozen scores.

Observed evidence–report–probe–answer alignment For the frozen human-audit item S-036 (probe Q03, local atom A0024), a Django task at checkpoint 4 (anonymized display; the unredacted artifact is not distributed), tool-use evidence records writing a fake\_psql.py executable that prints MY\_EXPORTED\_VAR, PGPASSWORD, and PATH; a second e2e\_check.py drives DatabaseClient.runshell([]) with a fake executable, testing both empty and passwordcontaining environments. The accumulated report states that the agent used a fake psql for end-to-end verification and inspected subprocess environment inheritance, but does not give the two scripts’ identities and individual functions. The actual frozen question asks for both names and roles; its archived reference supplies them. Across the three original report-view judging samples the outcome is unknown, unknown, unknown, while the three full-context outcomes are correct, correct, correct. In the independent 195-case human audit both raters label mention=yes and sufficiency=partial. The numeric replication archive retains the recorded judgments and scores; the full original answer texts are outside that archive. The example demonstrates mention without full answer support;

it does not establish that all details are necessary to make the final engineering decision.

Additional trajectory-grounded examples: risk and failed verification. The following two cases come from distinct issues in the 37-task headline cohort. For each, the archived question and reference are linked to checkpoint execution evidence; the reported scores are the original three observer judgments for the accumulated report view U and checkpoint full-context view $T _ { \mathrm { f u l l } }$ . These examples illustrate what the instrument tests, not a humanvalidated decision-relevance rate.

Django: a password-change compatibility risk. In django\_\_django-16631, checkpoint 4, probe Q40 (risk, importance 3) asks why an HTTP 500 risk for custom user models was more extensive than initially believed. The recorded execution analysis (segment S0382) identifies a conditional call to get\_session\_auth\_fallback\_hash() inside get\_user(): a custom user class implementing get\_session\_auth\_hash() but not inheriting AbstractBaseUser may lack the new fallback method. A mismatching stored session hash can trigger this path not only after secret-key rotation but also when a password change is followed by a request from another device; the resulting AttributeError produces HTTP 500 rather than the previous logout behavior. The accumulated user-facing reports did not convey that HTTP 500 trigger condition. The three original U labels are wrong/unknown/unknown (mean score 0), whereas the three $T _ { \mathrm { f u l l } }$ labels are all correct (mean 1). This is a concrete unrecoverable risk condition; it is not evidence that a human supervisor actually made a wrong decision.

Pytest: an aborted baseline and restored patch. In pytest-dev\_\_pytest-10051, checkpoint 4, probe Q08 (failure, importance 3) concerns a failed earlier baseline-comparison attempt. The recorded command (S0137) first backed up the modified files and then ran git checkout on src/\_pytest/logging.py and testing/logging/test\_fixture.py before attempting pytest under set -u. The shell failed with ZSH\_VERSION: unbound variable (S0138), after the files had been reverted but before tests ran; the baseline output was empty. Subsequent actions restored the patch from temporary backups and reran the baseline without set -u (S0141-S0142). The cumulative reports describe the eventual baseline results without conveying this intervening failure and recovery sequence. The original U judgments are wrong/unknown/unknown (mean 0), and the $T _ { \mathrm { f u l l } }$ judgments are all correct (mean 1). A later successful verification does not erase a historical execution event; conversely, its non-recovery alone does not prove an outstanding unresolved blocker.

## A.4 Main-cohort provenance and sample sensitivity

Selection mechanism. We define a canonical task as a unique repository–issue ID (the SWEbench instance\_id); a trajectory is a particular execution identified by a run ID. The authoritative units.jsonl registry contains 76 distinct canonical issues, one selected trajectory per issue (76 run IDs), and four checkpoint rows per trajectory (304 rows). Other collection/calibration trajectories may exist for the same issue but are not additional inventory tasks. The development/heldout split is by canonical issue ID: 46/30 issues, with all trajectories of an issue assigned to its single split. The frozen headline selects 37 disjoint issues/trajectories (p2: 32; pilot: 5; 23 development, 14 held-out), each with four scored checkpoints (148). The remaining 39 distinct issues/selected trajectories come from a9 (22), cal (4), mid (4), and p3 (9). Selection followed available analysis/annotation batches, not a preregistered random 37-of-76 draw. Separately recorded incomplete collection runs are not the same as the 39 issues outside the headline cohort.

Aggregation reconciled before cohort comparisons. The 76-task archived view-ablation output supplies three distinct estimators: weighted micro $\mathrm { G a p } _ { \mathrm { v i e w } } = . 4 3 7 8 0 4$ , equal-task macro ${ \mathrm { G a p } } _ { \mathrm { v i e w } } =$ .433916 (75 contributing tasks in its paired subset), and unweighted micro $\mathrm { G a p } _ { \mathrm { v i e w } } = . 4 4 2 4 7 8$ . We use the same weighted-micro rule as the frozen 37-task headline for every number in Table 9; the three alternative 76-task aggregations are not interchangeable.

<table><tr><td>Subset</td><td>Valid probes Acc(U)</td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td></tr><tr><td>Frozen 37</td><td>4,799</td><td>.5205</td><td>.9271 .4066</td></tr><tr><td>Additional 39</td><td>5,387</td><td>.4748 .9372</td><td>.4624</td></tr><tr><td>Expanded 76</td><td>10,186</td><td>.4950 .9328</td><td>.4378</td></tr></table>

Table 9: Cohort sensitivity under the same categoryweighted micro estimator. The 76-task row is an expanded observational population, not a replacement for the frozen 37-task headline.

Observed selection and composition. The hardest difficulty category (1–4 hours) represents 10.8% of the 37-task cohort versus 25.0% of the full 76- task inventory, whereas tasks labeled under 15 minutes are 40.5% versus 23.7%. Thus the headline cohort overrepresents easier tasks. Equal-task view gaps are .3906 in the original 37 and .4393 in the complementary cohort (38 of 39 tasks contribute nonempty paired measurements). Reweighting their difficulty compositions to the 76-task mix yields .3987 and .4561, respectively; a simple difficulty-composition adjustment therefore does not eliminate the observed difference. These are post-hoc, nonrandomized checks, not a causal decomposition.

Attribution limits. The additional 39 runs were measured later and all have v3.2 metadata, while the frozen 37 include both v3.0 and v3.2 and were collected earlier. The batches also differ in tasks and possibly collection conditions. Thus their .4624 versus .4066 weighted gaps cannot be attributed to prompt revision, collection batch, time, or difficulty individually. The report-only accuracy changes more across cohorts than full-context accuracy (.4748 vs. .5205, and .9372 vs. .9271, respectively), but this descriptive observation does not identify the cause. The 76-task result shows that a positive gap persists across the expanded recorded inventory; it does not establish random-sample representativeness or a population-level correction for the 37-task estimate. The original 86/86 fork-prefix verification concerns tested fork instances, not all 148 headline checkpoints.

## A.5 Outcome-conditioned and probe-binding robustness

Hidden-test task outcomes and time of measurement. The original checkpoint snapshots have objective fix-status records from hidden-test execution (truth\_k4.json). All 37 headline runs have a final status: 30 fixed and seven not fixed. Table 10 separates eventual repair success from the final checkpoint. In particular, k1–k3 probes in an eventually fixed run cannot be presented as evidence of post-repair reporting. At k4, the user has already asked the agent to assess risks, so this test is about residual recoverability after a fixed outcome, not whether unprompted success reports suppress known risks.

<table><tr><td>Subset</td><td>Valid n Acc(U)</td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$  [95% CI]</td></tr><tr><td>Fixed; all k</td><td>3,914</td><td>.5129 .4118 [.3716,.4478]</td></tr><tr><td>Fixed;  $k = 4$ </td><td>1,172</td><td>.5705 .3479[.2816,.4126]</td></tr><tr><td>All runs; k = 4</td><td>1,450</td><td>.3451 [.2902,.3979]</td></tr></table>

Table 10: View-gap recovery by outcome and check point, using the primary category-weighted probe estimator and issue-clustered bootstrap. Final-checkpoint fixed tasks have $\mathrm { A c c } ( T _ { \mathrm { f u l l } } ) = . 9 1 8 5 ;$ their gap is positive for 28 of 30 issues. The 30 eventually fixed tasks have $\mathrm { A c c } ( T _ { \mathrm { f u l l } } ) = . 9 2 4 6$ across all checkpoints.

<table><tr><td>Subset</td><td>Valid n</td><td>Tasks</td><td> $\mathrm { G a p } _ { \mathrm { v i e w } } \ [ 9 5 \% \mathrm { C I I } ]$ </td></tr><tr><td>Primary cohort</td><td>4,799</td><td>37</td><td>.4066 [.3733,.4381]</td></tr><tr><td>Drop affected run</td><td>4,653</td><td></td><td>36.4141 [.3824,.4439]</td></tr><tr><td>Drop 86 mismatches</td><td>4,713</td><td></td><td>37 .4110 [.3796,.4404]</td></tr></table>

Table 11: Sensitivity to the probe–atom binding inconsistency. All estimates use the original frozen scores, original probe-category weights, and the same issueclustered protocol; no probe IDs or references are rewritten.

Misbound probe–atom IDs. A provenance audit found 87 generated probes whose recorded category disagrees with the category of the atom referenced by their ID, all from pilot\_django\_\_django-15268. The archived probe-generation requests/outputs and saved probes agree on the IDs; the inconsistency originates in the generation-stage binding. Of these 87, 86 pass the main validity mask and one does not. Questions, reference answers, and frozen observer grades are retained. The category-weighted estimator uses the probe category; consequently this audit probes category-binding sensitivity rather than replacing answers or repairing IDs by text similarity. Table 11 reports two exclusion checks. The positive gap remains after either exclusion; the affected run has a smaller within-run gap (.2256), so its inclusion reduces the headline point estimate in this sample. These comparisons are not a confidence bound or a correction for other possible extraction errors.

Reproduction of the headline interval. The canonical pooled, category-weighted $\mathrm { G a p } _ { \mathrm { v i e w } }$ resamples complete issues (37 clusters) with replacement using random.Random(0) and rng.choice, recomputing the ratio of pooled weighted score differences to pooled weights for each draw. With 10,000 draws and percentile-order indices ⌊.025N⌋ and ⌊.975N⌋ − 1, the main 95% interval is [.3733, .4381], rounding to [.373, .438] as reported in the paper. A task-equal macro estimator uses different aggregation weights and is not substituted for the primary probe-weighted estimate. Other audit-specific intervals may use their separately stated resampling counts.

## A.6 View construction

The complete context $T _ { k }$ includes reasoning, tool calls, tool outputs, and narration available at checkpoint k. The report view $U _ { k }$ includes the actor’s final reports for completed turns together with user instructions. The latest-report view retains only the most recent report. The terminal view $V _ { k }$ is a capture of default-visible scrollback, where long outputs can be collapsed or truncated. The condition $T _ { k } \setminus R$ removes reasoning blocks from the complete context; R denotes reasoning.

The consumption ladder successively adds narration, execution headers, and progressively more tool outputs to the reports before reaching terminal scrollback. Each consumption tier contains the preceding tier’s supplied material. The recenttool-call conditions branch directly from $U _ { k }$ and are analyzed separately. Terminal rendering and complete execution context are different representations, so their relation is described by the view construction rather than a text-subset assumption. Only the terminal-based actor supports the entire rendered ladder; the cross-agent comparison uses report and full-context views available for every actor.

## A.7 Probes, weighting, and missing observations

Extraction assigns each atom a category and an importance rating. Eligible atoms meet the importance threshold in Table 7. Probes ask questions grounded in those atoms and provide reference answers and distractor structure. A probe passes the execution-dependence filter when its task-description-only accuracy is below full credit and its full-context accuracy is above zero. These accuracies are the same recorded answers used in final measurement, not independent screening runs. Of 4,855 probes, 4,799 are retained; 54 fail the $T -$ score threshold and two are excluded by the priorscore threshold. Conditioning on $T > 0$ increases the observed full-context accuracy by 0.0085 and report accuracy by 0.0029, for a net +0.0056 in $\mathrm { G a p } _ { \mathrm { v i e w } }$ relative to the unfiltered population. This small but nonzero selection effect is part of the frozen protocol and is not an external validation of probe quality.

A separate judge labels each answer correct (1), partial (0.5), wrong (0), or unknown (0). Repeated samples are averaged before the weighted aggregate. Inverse-sampling weights use eligible-atom counts divided by sampled-probe counts within each run, checkpoint, and category. Empty materials, failed judge calls, and unparseable answers are recorded as reader or measurement failures rather than ordinary wrong answers.

The legacy reported\_in\_text flag records extraction provenance (assistant\_text in the atom’s source types), not mention in the user-facing report. It must not be interpreted as an independent omission measure. Semantic conveyance separately classifies report evidence as explicit, implicit, or absent; a stratified independent human audit distinguishes mention from probe-conditioned answer sufficiency. These constructs and measurement populations remain separate.

## A.8 Intervals and comparison units

The main estimate has 37 unique issue IDs and 37 selected trajectories; the expanded inventory has 76 issue IDs and 76 selected trajectories. For the RQ2 link1 analysis, the recorded 72 runs represent 72 task IDs, with 286 (run, k, qid) units (284 in the paired null contrast). Its bootstrap first aggregates by issue, then resamples 72 issues, not the 284 probe-linked units. The cross-harness grid contains 12 unique issues and 36 trajectories (one per harness per issue); 11 issue IDs overlap the 76- task inventory. Cross-executor validation uses the same 12 issues and 24 model-specific trajectories, reusing the 12 DSH–HY4 trajectories rather than treating them as new independent tasks. Across analyses, canonical issue ID defines splits and statistical clusters; within-issue checkpoints/probes and additional runs are not independent issue samples. Main-sample, supervision, and reportingdesign intervals use issue-clustered percentile bootstrap with the settings in Table 7. An interval based on fewer contributing tasks than the stated threshold is descriptive. Cross-agent estimates average per-task values over the shared task set. For task values $x _ { j }$ , the interval is

$$
{ \bar { x } } \pm t _ { . 9 7 5 , n - 1 } { \frac { s _ { x } } { \sqrt { n } } } .\tag{4}
$$

For paired contrasts, $x _ { j }$ is the difference between agents on task $j .$ . Each actor produces its own trajectory and probes; pairing is therefore at the task level. The same Student t formula is used for view gaps, omission diagnostics, and conditionalrecovery contrasts. Cross-agent intervals are unadjusted for multiplicity.

## A.9 Conformance and workflow

Cross-agent conformance checks cover prefix fidelity, workspace snapshots, instruction integrity, and the availability of each context view. Terminal scrollback is unavailable for harnesses without a terminal interface. The three harnesses share the HY4-preview model backend while varying the agent harness (Claude Code version 2.1.154, DSH, and Pi); this is not a comparison of foundation models. Model use and request parameters were audited from the recorded 37-run logs (Table 12). The atom extractor, probe generator, observers, and judges use HY4-preview. Probe generation produces the question and reference answer in the same model call, and the judge sees the reference; the observer does not. These stages therefore have correlated model-dependent errors despite the observer being reference-blind. Effective service-side defaults for omitted request parameters remain unknown.

<table><tr><td>Recorded stage</td><td>Calls Explicit temperature</td></tr><tr><td>Atom extraction</td><td>226 none (226 unknown)</td></tr><tr><td>Probe + reference generation</td><td>520 0.0 (211); 309 unknown</td></tr><tr><td>Observer answers</td><td>8,2580.7 (697); 7,561</td></tr><tr><td>Judge / coverage / audits</td><td>unknown 9,929 0.0 (1,064); 8,865 unknown</td></tr></table>

Table 12: Archived 37-run model-call audit, including associated coverage and rescoring work. All stages use HY4-preview; the reference is emitted in the probegeneration call. No $\mathsf { t o p \_ p o r t o p \_ s o p \_ s o r t }$ k values are logged. “Unknown” means omitted from the recorded request, not zero or a verified server default. Counts are logged calls, not independent judged probes.

Pipeline dependency and instrument audit. Across 37 runs, recorded stage tags group 226 atom-extraction, 520 probe-generation, 8,258 observer, and 9,929 judge/coverage/semantic-review calls; reference answers are generated jointly with probes. HY4-preview fills these model roles, while independent grading and the complete DeepSeek pipeline provide complementary checks. Recorded temperatures are 0.0 for 211 probe and 1,064 judge calls, and 0.7 for 697 observer calls; omitted values and top\_p/top\_k are unknown. Three identified reference inconsistencies (P0001, P0024, P0028) imply an illustrative item-specific adjustment of at most about 0.04 percentage points to Acc(U). The separate human audits evaluate grading and report evidence support.

## A.10 Public benchmark artifacts and replication scope

The public benchmark and replication materials are available at https://github.com/ Kaguraaaaaa/Ctxgap. The approved CTX-GAP release provides two distinct reproduction paths. First, the numerical-replication materials preserve archived probe-level scores, validity masks, and aggregation inputs. Running reproduce\_all.py checks the original weighted estimates, including the main-sample view gap of 0.406582 (37 tasks, 4,855 candidate probes, 4,799 valid), from frozen judgments without regenerating model responses. This is numerical reproduction of recorded evaluations, not an independent rerun of the original measurement.

Public text benchmark. The separate, approved text benchmark contains task/checkpoint records, probe questions and reference answers, report views U, terminal-visible views V , and privacyprocessed execution materials for public evaluation. It also provides the observer and judge prompt templates, their provenance documentation, and a model-calling evaluator (evaluate.py) for new measurements on published inputs. The prompt provenance audit checks logged system prompts, fixed template elements, and recorded model settings, but does not claim byte-for-byte identity of every dynamically constructed request. The 23- development/14-evaluation division is a predefined, public task split: because references are published, the latter is not a hidden test set.

Scope of public-text evaluation. The original full context $T _ { \mathrm { f u l l } }$ contains private reasoning and execution records that are not distributed; the processed public context $T _ { \mathrm { p u b l i c } }$ is not the identical input. Consequently, running evaluate.py on released text produces a new public-input measurement, which must be reported separately from the historical 40.7-point full-context–report gap. The numerical-replication results must not be described as proof of end-to-end model re-evaluation. The release excludes restricted original execution records and private reasoning.

Historical recovery versus unresolved state. An earlier failure remains a recoverable historical event even after a successful rerun. Accordingly, longitudinal unrecovered/unaligned counts refer to past execution-supported state, not verified open blockers. The prospective rubric in protocol/HISTORICAL\_STATE\_RULE.md separately checks event evidence, checkpoint validity, decision relevance, and positive evidence of continued unresolved status or supersession. It was not retrospectively applied as a human label to every frozen persistence atom.

## B Reading Depth and Gap Anatomy

## B.1 Reading conditions

Table 14 retains the complete reading ladder, recenttool-call branches, and summary additions. The ladder’s accuracies and actor-gap values are computed from the same main-sample probes. The full-context observer and the actor have separate accuracy measurements. Accordingly, the residual actor gap at full context differs from the view gap between full context and reports.

The reading results describe recoverability when the observer is given a specified view. Terminal scrollback captures rendered information, while complete tool interactions and reasoning extend the supplied context further. Natural and structured summary additions are evaluated at the same checkpoint, so their results can be related to the original report view without changing the underlying state.

Paired view decomposition. To avoid mixing probe-micro and task-macro aggregates, Table 13 recomputes all four views on the same 4,799 valid probes (37 runs, 148 checkpoints) using the frozen category-weighted micro estimator $w ~ = ~ N _ { r k c } / n _ { r k c }$ , where c is the probe category. Its telescoping contrasts sum exactly to $\mathrm { G a p } _ { \mathrm { v i e w } } = 0 . 4 0 6 5 8 2$ . The V − U contrast measures the change in accuracy when the observer receives terminal scrollback instead of reports; $T _ { \mathrm { m i n u s } R } - V$ and $T _ { \mathrm { f u l l } } - T _ { \mathrm { m i n u s } R }$ compare further specified contexts. Although scrollback is interfaceaccessible, accessibility does not imply it was read or retained. Because these views are not all literal nested text subsets, the percentages are accuracygap decomposition, not a causal partition of information locations or user behavior. Clustering and 2,000 task-bootstrap draws preserve the matched units; 56 of 4,855 probes fail the frozen validity rule, no checkpoint or run is dropped.

<table><tr><td>View (weighted micro)</td><td>Accuracy</td><td>Share [95% CI]</td></tr><tr><td>Reports U</td><td>.5205</td><td></td></tr><tr><td>Terminal scrollback V</td><td></td><td>.7359 53.0% [47.6, 57.8]</td></tr><tr><td>Execution, no reasoning  $T _ { \mathrm { m i n u s } R }$ </td><td></td><td>.8902 37.9% [33.1, 43.7]</td></tr><tr><td>Full execution  $T _ { \mathrm { f u l l } }$ </td><td>.9271</td><td>9.1% [6.0, 12.1]</td></tr><tr><td>Full-minus-report gap</td><td>.4066</td><td>100%</td></tr></table>

Table 13: Exact telescoping decomposition of the weighted main-sample report-to-full-context gap $( n =$ 4,799 probes, 37 tasks). Each view row’s share is the accuracy increase from the preceding row divided by .406582; step increases are .215436, .154222 and .036924, respectively. Share intervals use paired taskcluster bootstrap (2,000 resamples). The displayed view accuracies are weighted micro means. A share is not the fraction of facts demonstrably contained only in that view.

<table><tr><td>k</td><td>Fresh</td><td>Earlier</td><td>Unaligned</td><td> $\operatorname { G a p } _ { \mathrm { a c t o r } }$ </td><td>95% CI</td></tr><tr><td>1</td><td>9.423</td><td>0.014</td><td>9.436</td><td>0.426</td><td>[0.375, 0.480]</td></tr><tr><td>2</td><td>11.546</td><td>7.480</td><td>19.027</td><td>0.472</td><td>[0.439, 0.505]</td></tr><tr><td>3</td><td>3.057</td><td>17.984</td><td>21.041</td><td>0.436</td><td>[0.390, 0.482]</td></tr><tr><td>4</td><td>3.411</td><td>18.869</td><td>22.280</td><td>0.359</td><td>[0.317, 0.403]</td></tr></table>

Table 15: Unaligned state and actor gap by checkpoint. Unaligned counts are fresh plus earlier-turn atoms; the accuracy gap is a separate normalized measurement.

<table><tr><td>View</td><td>Mean chars</td><td>Accuracy</td><td>95% CI</td><td> $\mathrm { G a p } _ { \mathrm { a c t o r } }$ </td></tr><tr><td>Latest report</td><td>2,565.1</td><td>0.239</td><td>[0.219, 0.263]</td><td>0.697</td></tr><tr><td>All turn reports</td><td>6,564.4</td><td>0.520</td><td>[0.491, 0.551]</td><td>0.416</td></tr><tr><td>+ Narration</td><td>7,532.8</td><td>0.545</td><td>[0.520, 0.574]</td><td>0.391</td></tr><tr><td>+ Exec. headers</td><td>13,064.8</td><td>0.639</td><td>[0.610, 0.669]</td><td>0.297</td></tr><tr><td>+ Last 3 outputs</td><td>14,553.5</td><td>0.657</td><td>[0.635, 0.683]</td><td>0.279</td></tr><tr><td>+ Last 10 outputs</td><td>18,256.3</td><td>0.690</td><td>[0.666, 0.715]</td><td>0.247</td></tr><tr><td>+ Full scrollback</td><td>25,476.5</td><td>0.736</td><td>[0.715, 0.757]</td><td>0.200</td></tr><tr><td>Context w/o thinking</td><td>107,388.0</td><td>0.890</td><td>[0.882, 0.900]</td><td>0.046</td></tr><tr><td>Full context</td><td>167,389.0</td><td>0.927</td><td>[0.913, 0.938]</td><td>0.009</td></tr><tr><td>Reports + last tool call</td><td>6,849.2</td><td>0.525</td><td>[0.496, 0.558]</td><td>0.411</td></tr><tr><td>Reports + last two tool calls</td><td>7,179.2</td><td>0.525</td><td>[0.493, 0.563]</td><td>0.411</td></tr><tr><td>Reports + natural summary</td><td></td><td>0.528</td><td>[0.501, 0.559]</td><td>0.408</td></tr><tr><td>Reports + structured summary</td><td></td><td>0.597</td><td>[0.564, 0.633]</td><td>0.339</td></tr><tr><td>Structured summary alone</td><td>3,014.7</td><td>0.452</td><td>[0.415, 0.489]</td><td>0.484</td></tr></table>

Table 14: Reading conditions and summary additions. The two recent-tool-call conditions branch from reports; ful context extends beyond terminal visibility. Context cost is the mean number of characters.

## B.2 Accumulation and semantic composition

The frozen semantic classifier assigns 71.1% of weighted gap mass to not conveyed (79.8% under omission-first precedence). This category describes semantic insufficiency; the independent mention/support audit in Appendix B.4 directly evaluates literal mention. Table 15 separates current-turn and earlier-turn unaligned atoms. The total unaligned count is their sum; the source’s broader atom-population total is not an unaligned count. The normalized actor gap varies across checkpoints even as the absolute unrecovered count increases.

Table 17 retains the legacy semantic-versusprovenance diagnostic solely for computational traceability; the provenance column is not a reporting-based omission estimate. Table 19 derives all category shares from the same semanticreport population as the main decomposition. Each category’s typed components exhaust its gap mass, and summing category mass recovers the overall semantic composition. Figure 4 presents the category-level panels, including the smaller plan and constraint categories; the overall anatomy and accumulation are in Figure 3.

Table 18 records the historical provenance-flag statistic separately from semantic non-conveyance. Only the semantic judgment concerns the reporting view; the provenance flag cannot establish whether an earlier-turn fact was ever mentioned in a final report.

<table><tr><td>Component</td><td>Share</td><td>Atoms</td></tr><tr><td>Not conveyed</td><td>0.711</td><td>1397</td></tr><tr><td>Reported, not recovered</td><td>0.151</td><td>294</td></tr><tr><td>Wrong report answer 2</td><td>0.108</td><td>213</td></tr><tr><td>Latest-only partial†</td><td>0.028</td><td>65</td></tr><tr><td>Stale</td><td>0.002</td><td>5</td></tr><tr><td>Total</td><td>1.000</td><td>1974</td></tr></table>

Table 16: Automatic semantic decomposition of mainsample gap mass. “Not conveyed” is the classifier’s non-conveyance class, not literal non-mention; human mention and sufficiency are measured separately (Appendix B.4). Counts are gap atoms; shares are weighted. <sup>∗</sup>Wrong-answer labels take precedence over non-conveyance. <sup>†</sup>Same-checkpoint partial recovery under the latest-report view.

<table><tr><td>Source class</td><td>Semantic Provenance-flag split</td></tr><tr><td>Not conveyed</td><td>0.711 0.867</td></tr><tr><td>Reported, not recovered</td><td>0.151 0.020</td></tr><tr><td>Wrong report answer</td><td>0.108 0.108</td></tr><tr><td>Latest-only partial</td><td>0.028 0.002</td></tr><tr><td>Stale</td><td>0.002 0.002</td></tr></table>

Table 17: Semantic-class judgments and extractionprovenance flags. The provenance flag indicates extraction from assistant\_text, a distinct measurement from report conveyance.

## B.3 Semantic non-conveyance: classification-order sensitivity

We retain the frozen semantic decomposition of 1,974 typed gap atoms. For atom i, its contribution to gap mass is $w _ { i } [ \operatorname { A c c } _ { i } ( A ) - \operatorname { A c c } _ { i } ( U ) ]$ , using the original sampling weight and partial-credit scoring; it is not the separate gap\_amount field. The total weighted mass is 2,089.42. Recomputing this quantity with the frozen labels exactly reproduces all five reported classes and their counts (Table 20).

![](images/a604d99b8f32c973ca65a5b85dca142ef85f6c5a643716a7fd81cfe57a9875f9.jpg)

![](images/73c3e8ae697eefbf29ac7379f81da03fd44f17fe1bd6a8d692a0d25514505ec7.jpg)

Figure 4: Supplementary RQ1 category decomposition. (c) Within-category gap composition and (d) each category’s contribution to total semantic gap mass. This complements the omission and accumulation evidence in Figure 3.
<table><tr><td>Quantity</td><td>Share</td></tr><tr><td>Old gap mass</td><td>0.620</td></tr><tr><td>Old, not traced to assistant_text</td><td>0.979</td></tr><tr><td>Old, judged not conveyed</td><td>0.942</td></tr></table>

Table 18: Older-turn atom diagnostics. The sourceprovenance flag does not measure literal mention in reports; only the frozen semantic judgment refers to report conveyance.

In the frozen precedence, a wrong reportreader answer is classified as misled before checking whether the corresponding fact was conveyed. Of 213 atoms labeled misled, 171 have semantic\_conveyed=False. Giving omission priority reassigns exactly these 171 atoms (8.71 percentage points of weighted gap mass), raising the omission share from .7112 to .7984 and reducing misled from .1082 to .0211. The frozen proportions remain the reported decomposition; the wrong-answer share depends on classification precedence.

Interpretation of the classification audit. The 1,974 conveyance labels originate from an LLM judge. The precedence test checks rule sensitivity on those fixed labels; the separate 195-atom annotation evaluates mention and answer support. Code reproduction verifies the frozen classification and its weighting, while the human audit probes its semantic interpretation.

Decay is distinct from latest-report insufficiency. The original assign\_types reproduces the frozen counts and weighted shares, but its “decayed” category is cross-sectional: it uses the latestreport-only view $U _ { \mathrm { l a t e s t } }$ at the same checkpoint, not a prior checkpoint. Its zero-score fallback makes the effective condition $0 < \mathrm { A c c } ( U _ { \mathrm { l a t e s t } } ) < 0 . 5$ after preceding branches. Hence the frozen 65 cases describe partial recovery from the latest report, not established temporal decay. Treating zero as an observed score instead of a missing value yields 307 cases. Both figures use the same 1,974 atoms and gap mass $\begin{array} { r } { \sum _ { i } w _ { i } [ \operatorname { A c c } _ { i } ( A ) - \operatorname { A c c } _ { i } ( U ) ] = } \end{array}$ 2089.420608.

Longitudinal recoverability audit. For the third rule, a matching atom must have an earlier scored checkpoint $k ^ { \prime } <$ k with $\mathrm { A c c } ( U _ { k ^ { \prime } } ) \geq 0 . 5$ and a current score $\mathrm { A c c } ( U _ { k } ) < 0 . 5$ . Historical scores were reconstructed from the full checkpoint measurement files rather than the gap-only atom file (which cannot reveal earlier successes). Of 359 eligible atoms, 196 have a comparable earlier checkpoint; 121 meet this recovery-loss condition, 75 have prior observations but no earlier successful recovery, and 163 lack an earlier comparable observation. Tightening the recovery threshold to .75 yields 63 cases (3.26% of weighted gap mass), indicating threshold dependence. A subsequent full identity check confirmed the text identity of all 121 longitudinal recovery-loss candidates across their matching earlier checkpoints (121/121). This validates the temporal join key, not the factual validity of every claim or the absence of observer variability. A score of $\operatorname { A c c } ( A ) \geq . 5$ at the current checkpoint also does not independently certify that the underlying fact is still true. Thus neither cross-sectional definition should be read as temporal decay, and the 121 longitudinal cases are candidatesfor recovery loss, not direct evidence of human forgetting. The omitted class remains 1,397 atoms (71.12%) under all three alternatives because it is determined by the separate semantic\_conveyed decision; the semantic validity of that decision is not established by this sensitivity test.

<table><tr><td></td><td></td><td></td><td>Not</td><td>Reported,</td><td>Wrong report</td><td>Latest-only</td><td></td></tr><tr><td>Category</td><td>Atoms</td><td>Total share</td><td>conveyed</td><td>unrecovered</td><td>answer</td><td>partial</td><td>Stale</td></tr><tr><td>Facts</td><td>632</td><td>0.428</td><td>0.687</td><td>0.187</td><td>0.098</td><td>0.028</td><td>0.000</td></tr><tr><td>Failures</td><td>373</td><td>0.166 0.159</td><td>0.894 0.545</td><td>0.028 0.221</td><td>0.074 0.169</td><td>0.004 0.052</td><td>0.000 0.012</td></tr><tr><td>Progress Environment</td><td>332 205</td><td>0.082</td><td>0.825</td><td>0.068</td><td>0.090</td><td>0.014</td><td>0.004</td></tr><tr><td>Risks</td><td>128</td><td>0.050</td><td>0.666</td><td>0.157</td><td>0.149</td><td>0.027</td><td>0.000</td></tr><tr><td>Decisions</td><td>99</td><td>0.037</td><td>0.678</td><td>0.177</td><td>0.094</td><td>0.051</td><td>0.000</td></tr><tr><td>Rejected options</td><td>82</td><td>0.035</td><td>0.730</td><td>0.116</td><td>0.113</td><td>0.041</td><td>0.000</td></tr><tr><td>Assumptions</td><td>48</td><td>0.017</td><td>0.732</td><td>0.054</td><td>0.176</td><td>0.038</td><td>0.000</td></tr><tr><td>Plans</td><td>47</td><td>0.016</td><td>0.793</td><td>0.106</td><td>0.081</td><td>0.020</td><td>0.000</td></tr><tr><td></td><td>28</td><td>0.010</td><td>0.550</td><td>0.389</td><td>0.061</td><td>0.000</td><td>0.000</td></tr><tr><td>Constraints</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 19: Complete semantic category decomposition using the main-sample population. "Not conveyed" denotes automatic semantic non-conveyance. All values except counts are proportions.

<table><tr><td>Type</td><td colspan="2">Frozen share (n) Omission first (n)</td></tr><tr><td>Not conveyed</td><td>.7112 (1397)</td><td>.7984 (1568)</td></tr><tr><td>Wrong report answer</td><td>.1082 (213)</td><td>.0211 (42)</td></tr><tr><td>Reported, not recovered</td><td>.1506 (294)</td><td>.1506 (294)</td></tr><tr><td>Latest-only partial</td><td>.0277 (65)</td><td>.0277 (65)</td></tr><tr><td>Stale</td><td>.0022 (5)</td><td>.0022 (5)</td></tr></table>

Table 20: Classification-precedence sensitivity on the same 1,974 semantic gap atoms, keeping all original weights and scores. Only 171 atoms move from misled to omitted; weighted mass is conserved and each column sums to one at full precision (displayed shares are rounded).

<table><tr><td>Operational rule</td><td> $\mathbf { \ddot { \delta p e c a y } } ^ { * } n$ </td><td>Mass (%) Residual n</td></tr><tr><td>Frozen latest-only</td><td>65</td><td>2.77 294</td></tr><tr><td>Zero included</td><td>307</td><td>16.04 52</td></tr><tr><td>Prior recovery lost</td><td>121</td><td>6.29 238†</td></tr></table>

Table 21: Sensitivity of the conveyed, non-misled, nonstale pool (359 atoms) under alternative decay rules. The residual includes reported-but-unrecovered atoms; <sup>†</sup>also includes 163 atoms without a comparable earlier checkpoint. All rules keep omission fixed at 1,397 atoms (71.12% of total weighted gap mass). The longitudinal row is a recovery-loss diagnostic, not an adjudicated forgetting rate.

## B.4 Independent mention and sufficiency annotation

To test whether the frozen automated not conveyed class means literal non-mention, two independent annotators reviewed the same original frozen report view $U _ { k }$ and task-state probes, rather than the later anonymized release view. A fixed 195-item stratified sample covered 70 frozen omitted, 40 misled, 40 longitudinal recovery-loss candidates, 40 corresponding reported-but-unrecovered cases, and all five stale cases. Every sampled item was uniquely joined back to the 1,974-item frozen population by task, checkpoint, atom identifier, and question. All 195 joins were reported as exact, including three recovered atom texts. The original total weighted gap mass, $\begin{array} { r } { \sum _ { i } w _ { i } [ \operatorname { A c c } _ { i } ( A ) - } \end{array}$ $\operatorname { A c c } _ { i } ( U ) ] = 2 0 8 9 . 4 2 0 6 0 8$ , and the frozen omitted share (.711247) were reproduced before reweighting the human annotations.

Annotators separately labeled mention (whether the report refers to the specific state) and probeconditioned sufficiency (whether report evidence supports a complete reference answer). Reader recovery was read from the existing scored answers rather than independently human-labeled. For mention, raw agreement was 85.6% and Cohen’s $\kappa = . 6 3 2 ;$ for sufficiency, 81.5% and κ = .638 (each n = 195). These labels differ in construct and category count from an earlier three-way conveyance audit $( \kappa = . 4 8 1 )$ ; their kappa coefficients do not by themselves establish that instructions improved.

The stratified sample is not self-weighting. The five strata are omitted (70/1397), misled (40/213), longitudinal recovery-loss candidates (40/121), reported-but-unrecovered (40/238), and stale (5/5), with $\pi _ { h } = n _ { h } / N _ { h }$ ranging from .0501 to 1.0000. For each annotated item, its frozen gap-mass contribution is multiplied by $1 / \pi _ { h } ;$ category shares are normalized by reconstructed weighted mass. Taskcluster bootstrap resamples tasks while retaining the stratified estimator. An independent computation recovered the 55.4%/59.2% point estimates within rounding/Monte Carlo variation. The two human estimators are reported separately, never treated as an adjudicated label. Their estimated not-mentioned shares differ by 8.34 percentage points (A: .4205; B: .3371), reflecting disagreement whose specific source has not been established.

Within the frozen omitted stratum, both annotators find substantial mention without sufficient detail (Table 23). This invalidates the literal reading of “71.1% not conveyed” as “71.1% never mentioned.” The frozen percentage remains an exact automated-classification result, while the humanweighted estimates describe a different measurement construct. Because both 95% intervals for the mentioned-but-insufficient share include .5, the data support a large point-estimated contribution without establishing a majority at this confidence level. The small share labeled sufficiently supported but not recovered is a fraction of gap mass, not an estimated recovery-error probability given sufficient reports. Neither quantity by itself identifies a causal mechanism.

## C AI-Labeled Supervisory Relevance

Annotation and estimands. A 200-probe audit sample was drawn across category × importance groups from a privacy-screened eligible pool. Two models (HY4-preview and DeepSeek) separately labeled whether each state would be useful for a supervisor deciding next actions, assessing task progress, or assessing risk; uncertain responses and failed labels were not silently recoded as irrelevant. They did not see report-observer grades, referenceconditioned gap labels, or the opposite model’s answer. Of 200 probes, 187 have a paired valid label; 162 pairs agree (86.63%), 25 disagree, with Cohen’s $\kappa = . 6 3 7 3$ . HY4 had two failed labels and DeepSeek eleven. The underlying observerscore analysis contains 195 valid sampled probes; group-specific denominators therefore differ.

Population reweighting and limitations. The balanced sample is not representative of the main 37-task probe composition. Category-byimportance post-stratification yields the following gap estimates and task-cluster intervals for relevant probes: HY4 .4188 [.231, .589], DeepSeek .4797 [.256, .652], and jointly relevant .4726 [.235, .648]. Corresponding post-stratified Acc(U)/Acc(T) are .4980/.9168, .4414/.9211, and .4492/.9218. For irrelevant probes, reweighted gaps are .5439 [.424, .655] (HY4) and .4492 [.331, .572] (DeepSeek). Several post-strata contain only one or very few samples, yielding unstable large weights and much wider intervals. The relative ranking of relevant and irrelevant gap reverses under DeepSeek after weighting and is not a finding. These analyses establish a gap conditional on model-defined relevance, not an estimated proportion of human-useful omissions or downstream decision errors; independent human relevance validation remains outstanding.

## D Targeted Evidence Supplementation and Lexical Answerability

Conditional candidate pool and manipulation. Of 104 candidates drawn on the basis of the two annotators’ mentioned-but-insufficient labels in the 195-item study, five with known data defects were excluded before analysis. The remaining 99 candidates span 33 tasks. For each frozen probe, three inputs keep the original $U _ { k }$ intact: original reporting (O); $U _ { k }$ with related mention/metadata that is not designed to answer the probe (M); and $U _ { k }$ with genuine pre-checkpoint tool-use/tool-result evidence retrieved with access to the probe/reference (E). The evidence addition was constructed under an approximately 220-token-per-candidate budget; this is not a global checkpoint-level report budget. The 99-candidate pool is conditioned on human annotation and is not an inverse-probability-weighted sample of the 1,974 gap atoms. Moreover, mention and evidence were not rigorously matched for length or content. The contrast estimates the performance of these particular additions and cannot identify the isolated causal effect of evidence sufficiency.

Paired observer results. The full candidate pool has 297 scored sample–observer units per condition (99 probes × three observers; 891 responses overall), with blind cross-model grading and full/partial/no credit of 1/.5/0. One truncated HY4 mention answer was recovered and graded under the same condition-blind assignment before final analysis. Task-cluster bootstrap intervals use the 33 original tasks, not 297 independent tasks. The three observers have different baseline recovery levels; per-observer changes are shown separately. The equal-observer-average paired improvement $\Delta _ { E - M } = + . 1 8 9$ (95% CI $[ + . 1 3 7 , + . 2 4 1 ] )$ , whereas $\Delta _ { M - O } = - . 0 0 2 $ (95% $\operatorname { C I } \left[ - . 0 2 7 , + . 0 2 5 \right] )$ . The mention-only contrast has an interval spanning zero.

<table><tr><td rowspan="2">State for report-only probe</td><td colspan="2">Annotator A</td><td colspan="2">Annotator B</td></tr><tr><td>Mass share</td><td>95% CI</td><td>Mass share</td><td>95% CI</td></tr><tr><td>Not mentioned</td><td>.4205</td><td>[.314,.535]</td><td>.3371</td><td>[.247,.426]</td></tr><tr><td>Mentioned, insufficient</td><td>.5543</td><td>[.439,.658]</td><td>.5917</td><td>[.493,.691]</td></tr><tr><td>Sufficient, not recovered</td><td>.0252</td><td>[.011,.045]</td><td>.0371</td><td>[.019,.061]</td></tr><tr><td>Uncertain</td><td>.0000</td><td></td><td>.0341</td><td>[.004,.073]</td></tr></table>

Table 22: Independent human measurement of report mention and probe-conditioned evidence sufficiency in 195 stratified gap atoms. Mass shares are inverse-stratum-probability weighted estimates relative to the frozen 1,974- atom gap population; 95% intervals use task-cluster bootstrap. Annotators are shown separately; both confidence intervals for “mentioned, insufficient” include 50%. Values quantify where gap mass is associated with information states, not causal contributions or accuracy conditional on sufficiency.

<table><tr><td></td><td colspan="2">A</td><td colspan="3">B</td></tr><tr><td>Frozen omitted</td><td>Share</td><td></td><td>CI Share</td><td></td><td>CI</td></tr><tr><td>Not mentioned</td><td></td><td></td><td></td><td>.5254 [.385,.669].3927 [.273,.517]</td><td></td></tr><tr><td>Mentioned, insufficient .4746 [.332,.616] .5748 [.445,.700]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sufficient</td><td>.0000</td><td></td><td></td><td>.0000</td><td></td></tr></table>

Table 23: Human distinction within thefrozen omitted stratum (70 sampled atoms). Uncertain residual is not displayed; two separately weighted raters are not a consensus.

<table><tr><td>Group</td><td></td><td>n Acc(U) Acc(T)</td><td>Gap [95% CI]</td></tr><tr><td>HY4 relevant</td><td>151</td><td>.553</td><td>.906 .353 [.266,.435]</td></tr><tr><td>HY4 irrelevant</td><td>42</td><td>.368</td><td>.900.532[.421,.627]</td></tr><tr><td>DS relevant</td><td>135</td><td>.546</td><td>.910.364 [.259,.463]</td></tr><tr><td>DS irrelevant</td><td>49</td><td>.377</td><td>.873 .496 [.379,.606]</td></tr><tr><td>Both relevant</td><td>125</td><td>.553</td><td>.914 .361 [.260,.462]</td></tr><tr><td>All scored</td><td>195</td><td>.508</td><td>.901 .393 [.316,.467]</td></tr></table>

Table 24: Descriptive recoverability in the stratified audit sample. Labels are AI-assessed decision relevance, not human ground truth. CI calculations cluster by task.

Answer-overlap validity diagnostic. Character 8-gram overlap between added evidence and reference averages .188 (90th percentile .511; maximum .815). Eleven of 99 evidence additions have overlap $\ge ~ . 5 0 ;$ mention additions average .044 $\mathrm { ( n o n e \ge . 5 0 ) }$ . Across observers, high-overlap cases yield $\Delta _ { E - M } = + . 5 9 1 ( 9 5 \% \mathbf { C I } [ + . 4 0 9 , + . 7 7 3 ] )$ , compared $\mathrm { w i t h + . 1 3 8 [ + . 0 9 7 , + . 1 8 3 ] }$ after excluding these cases. Because this grouping is a post hoc sensitivity analysis, neither contrast is a population lower or upper bound. A descriptive linear relation between per-unit paired improvement and overlap has slope +.882 (task-cluster bootstrap CI $[ + . 5 8 5 , + 1 . 1 4 7 ] \rangle$ , with intercept +.023 $[ - . 0 2 7 , + . 0 7 7 ]$ . The quartile trend in Table 26 is monotonic. Lexical overlap may represent direct answer extractability as well as underlying evidence support; these observational contrasts do not distinguish copying from reasoning or prove a causal role for lexical similarity.

## E Reference-Paraphrase Construct-Validity Audit

Unmodified report/reference diagnostic (A). In an unfiltered 4,855-probe audit of the 37-task U view, reference–report character 8-gram overlap averages .364, .317, .226, and .199 for probes scored correct, partial, wrong, and unknown, respectively. Among wrong/unknown cases, only 28 have reference content-word coverage in $U _ { k }$ of at least .8. These lexical statistics describe reference–report overlap; the paraphrase experiment below tests the sensitivity of actual grades to reference wording.

Fixed-answer reference paraphrases (C; main sample). A 300-probe stratified sample spanning all 37 main-study tasks (with 60% enriched for $U / T _ { \mathrm { f u l l } }$ grade disagreement) was regraded after reference paraphrasing while preserving the original observer answers. The original sample scores are $\mathrm { A c c } ( U ) = . 4 9 1 7 , \mathrm { A c c } ( T _ { \mathrm { f u l l } } ) = . 8 9 3 3$ , and $\mathrm { G a p } _ { \mathrm { v i e w } } ~ = ~ . 4 0 1 7 ;$ regraded scores are .5117, .9033, and .3917. The paired gap change is −.0100, with task-cluster bootstrap CI $[ - . 0 4 1 7 , + . 0 2 3 6 ]$ Per-item grades change in 18.3% of U answers and 11.7% of full-context answers. Thus no large aggregate shift was detected in this particular sensitivity check, but the interval does not establish equivalence, the deliberate sampling enrichment prevents direct substitution for the headline estimator, and item-level sensitivity remains material. Reported automated quality checks indicate 99.7% language consistency and 99.0% reference-slot preservation, not independent human proof of semantic equivalence.

<table><tr><td>Observer</td><td>0</td><td>M</td><td>E E 一 M [95% CI]</td><td>M 一 O [95% CI]</td></tr><tr><td>HY4-preview</td><td>.303</td><td>.258</td><td>.535 +.278 [.205,.347]</td><td>-.045 [-.111,.020]</td></tr><tr><td>DeepSeek</td><td>.323</td><td>.343</td><td>.525 +.182 [.119,.245]</td><td> $+ . 0 2 0 \ [ - . 0 1 6 , . 0 5 6 ]$ </td></tr><tr><td>GLM</td><td>.475</td><td>.495</td><td>.601 +.106 [.037,.177]</td><td>+.020 [-.043,.077]</td></tr></table>

Table 25: Targeted intervention on all 99 selected candidates (33 tasks); full-pool estimates with task-clustered intervals. All figures are conditional on the human-screened pool, not estimates of overall gap composition.

<table><tr><td>Quartile</td><td>Mean overlap</td><td> $E - M$ </td><td>95% CI</td></tr><tr><td>Q1</td><td>.005</td><td>+.028</td><td>[-.036,.086]</td></tr><tr><td>Q2</td><td>.053</td><td>+.113</td><td>[.018,.213]</td></tr><tr><td>Q3</td><td>.208</td><td>+.167</td><td>[.080,.257]</td></tr><tr><td>Q4</td><td>.479</td><td>+.440</td><td>[.300,.594]</td></tr></table>

Table 26: Exploratory evidence–reference lexicaloverlap gradient in the 99-candidate intervention pool. Quartiles should not be interpreted as randomized evidence-quality groups.

Paraphrase checks within the intervention pool (B/C). In a separate check using candidates from the 99-item intervention pool, evidence was paraphrased with factual identifiers and values retained (B), or reference wording was paraphrased while the observer answer stayed fixed (C). Reported paired score changes for B were HY4 +.0625 [−.011, +.150], DeepSeek −.1000 [−.182, −.011], and GLM $- . 0 4 2 6 \quad [ - . 1 2 2 , + . 0 4 4 ] ;$ for C they were $\begin{array} { r l r } { \mathrm { H Y 4 } } & { { } + . 0 3 1 9 } & { [ - . 0 2 1 , + . 0 8 2 ] . } \end{array}$ DeepSeek −.0800 [−.156, +.019], and GLM −.0521 [−.102, −.010]. These exploratory directions differ by model and condition. The interventionpool checks are distinct from the 300-probe main-measurement C audit and cannot establish invariance of the full measurement pipeline.

## F Importance Post-Stratification

The original estimator weights probe scores by eligible-atom counts within each checkpoint and task-state category. The capped probe sampler draws within categories and samples highimportance atoms somewhat more frequently than their population share. We therefore recompute the same frozen endpoints using weights stratified jointly by category and importance. This sensitivity analysis uses the original answers and judgments, with no additional model calls; the 37-run main sample and the 12-task cross-agent grid remain separate.

<table><tr><td>Metric</td><td>Frozen Adjusted</td><td></td><td>∆ [95% CI]</td></tr><tr><td> $\mathrm { G a p } _ { \mathrm { a c t o r } }$ </td><td>.4159</td><td></td><td>.4244 +.0085 [.0018, .0154]</td></tr><tr><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>.4066</td><td> $. 4 1 5 3 \ + . 0 0 8 8$ </td><td>[.0021, .0153]</td></tr></table>

Table 27: Importance post-stratification in the 37-run main sample. Intervals use a run-clustered bootstrap. Frozen values are reproduced by the category-only estimator before changing weights.

<table><tr><td>Harness</td><td>Orig. Adjusted</td><td>CI (∆)</td></tr><tr><td>Claude Code .4070</td><td></td><td>.4216 [.0053, .0260]</td></tr><tr><td>Pi</td><td>.3903</td><td>.4026 [-.0003, .0238]</td></tr><tr><td>DSH</td><td>.4898</td><td>.5144 [.0130, .0362]</td></tr></table>

Table 28: View-gap sensitivity for each harness in the separate 12-task sample. Intervals use task-clustered bootstrap of the within-harness change; no cross-harness ranking is inferred.

Importance-3 atoms account for .567 of the eligible atom population but .634 of the sampled probes; importance-2 shares are .433 and .366, respectively. The combined category-by-importance stratification contains 20 strata, with weights from .88 to 2.38. The smallest stratum, constraints at importance 2, contains 17 probes and receives weight .88. The original category-only estimator uses weights from 1.01 to 1.82. Table 27 shows that adjusting for importance increases the main-sample actor gap by .0085; Table 28 shows positive within-harness shifts across all three implementations; it does not equalize their task-state category distributions. The post-stratified results are sensitivity estimates; all original headline results remain unchanged.

## G Human Calibration and Scoring Audits

## G.1 Annotation populations

The initial conveyance audit asked independent annotators to label atoms as explicit, implicit, or absent in the reports. Its reported Fleiss’ kappa is recorded with the other audit results below. A separate, later 195-atom audit directly assesses mention and answer support (Appendix B.4). A subsequent automated sufficiency audit inspected reported-butunrecovered probes by matching the information required by each reference to quoted report evidence. Its population includes probes from several anatomy classes. Table 30 gives the subset belonging to the reported-but-unrecovered class; Table 29 records the distinct audit sizes.

<table><tr><td>Audit</td><td>Items</td><td>Annotators</td></tr><tr><td>Conveyance wave 1</td><td>20</td><td>3</td></tr><tr><td>Probe-sufficiency machine audit</td><td>403</td><td></td></tr><tr><td>Stratified human sufficiency</td><td>45</td><td>2</td></tr><tr><td>Human-consensus subset</td><td>32</td><td>2</td></tr><tr><td>Question-aligned re-scoring</td><td>240</td><td></td></tr><tr><td>Disputed items retained as errata</td><td>14</td><td></td></tr></table>

Table 29: Audit sample sizes and recorded defects from the study.

<table><tr><td>Comparison</td><td>Agreement</td><td>κ</td></tr><tr><td>Human-human</td><td>0.711</td><td>0.441</td></tr><tr><td>Machine-annotator 1</td><td>0.822</td><td>0.554</td></tr><tr><td>Machine-annotator 2</td><td>0.711</td><td>0.447</td></tr><tr><td>Machine-human consensus</td><td>0.875</td><td>0.721</td></tr></table>

Table 31: Human calibration of probe sufficiency. Agreement and kappa are distinct measures.

Most items in that class have partial answer support: the report contains the general fact but omits details used by the reference. Other items have full support, no support, or contradictory evidence. These categories distinguish atom-level mention from probe-level answerability. An observer failure on a fully supported item can also involve a probe or reference defect, which the manual review inspects separately.

<table><tr><td>Audit label</td><td>Class share</td><td>Gap share</td><td>Items</td></tr><tr><td>Partial support</td><td>0.797</td><td>0.120</td><td>235</td></tr><tr><td>Full support</td><td>0.173</td><td>0.026</td><td>50</td></tr><tr><td>No support</td><td>0.024</td><td>0.004</td><td>8</td></tr><tr><td>Contradictory</td><td>0.005</td><td>0.001</td><td>1</td></tr><tr><td>Total</td><td>0.999</td><td>0.151</td><td>294</td></tr></table>

Table 30: Probe sufficiency within the reported-butunrecovered class. The original probe audit covers a broader set of 403 probes; this table refers to its 294- item class subset. Class shares and weighted mass are reproduced at the precision reported by the study.

<table><tr><td>Diagnostic</td><td>Value</td></tr><tr><td>Wave-1 Fleiss kappa</td><td>0.565</td></tr><tr><td>Annotator 1: stricter / looser</td><td>5/2</td></tr><tr><td>Annotator 1: exact p</td><td>0.453125</td></tr><tr><td>Annotator 2: stricter / looser</td><td>11/0</td></tr><tr><td>Annotator 2: exact p</td><td>0.000977</td></tr></table>

Table 33: Additional annotation diagnostics. Exact tests compare machine-stricter and machine-looser discordances.

<table><tr><td>Annotator / subset</td><td>Upgrades</td><td>Partial items</td><td>Rate</td></tr><tr><td>Annotator 1</td><td>5</td><td>35</td><td>0.143</td></tr><tr><td>Annotator 2</td><td>11</td><td>35</td><td>0.314</td></tr><tr><td>Consensus</td><td>3</td><td>24</td><td>0.125</td></tr><tr><td>Mean annotator rate</td><td></td><td></td><td>0.229</td></tr></table>

Table 32: Partial-to-full reclassification in the calibration sample. Consensus uses the subset on which both humans agree.

## G.2 Human judgments and reclassification

The stratified human sufficiency sample was annotated independently. Table 31 reports raw agreement and Cohen’s kappa; Table 32 gives the partialto-full reclassification counts and their denominators. The annotators differ in how often they accept an answer as fully supported. The consensus rate is calculated only among machine-partial items in the human-consensus subset; it is distinct from either individual annotator’s rate.

The machine-versus-human discordances include both upgrades and downgrades. The original exact-test results are retained in Table 33. Human comments identify references that request exact commands, import names, or line numbers beyond the substantive question. Such detail can influence both a sufficiency judgment and the scoring of an observer’s answer.

## G.3 Sensitivity calculations

Table 34 and Figure 5 apply the consensus and mean-annotator upgrade rates to the partial-support mass of the complete audited population. The adjusted fully supported mass is the machine fully supported mass plus the partial mass multiplied by the upgrade rate. The class-restricted audit in Table 30 has a different denominator and a different fully supported mass.

These scenarios characterize how reclassification changes the measured residue. The human sample was stratified, so the scenarios are sensitivity calculations rather than probability intervals or weighted population estimates. The category distinction and its sample definition are retained alongside the calculations.

<table><tr><td>Scenario</td><td>Upgrade rate</td><td>Gap share</td></tr><tr><td>Machine</td><td>0.000</td><td>0.034</td></tr><tr><td>Consensus adjustment</td><td>0.125</td><td>0.054</td></tr><tr><td>Mean-annotator adjustment</td><td>0.229</td><td>0.070</td></tr></table>

Table 34: Sensitivity of the fully supported residue over the complete audited population. This population differs from the class subset in Table 30. These are reclassification scenarios.

## G.4 Alternative scoring

The re-scoring audit samples probes from both high- and low-report-accuracy strata and scores both report-reader and actor answers with a question-aligned rubric. This rubric asks whether the answer conveys what the question requests. Table 35 retains the original and revised arm accuracies and their difference. Both accuracies increase, and the paired gap change is small relative to the measured gap in this audit sample.

The sample was deliberately stratified by reportreader accuracy. Its results therefore describe sensitivity to the alternative rubric within that audit design. The reported main-sample measurements use the original scoring protocol; documented defects remain in the item-level audit.

<table><tr><td>Quantity</td><td>Original</td><td>Re-scored</td><td>Change</td></tr><tr><td>Acc(U)</td><td>0.514</td><td>0.540</td><td>+0.026</td></tr><tr><td>Acc(A)</td><td>0.935</td><td>0.972</td><td>+0.037</td></tr><tr><td>Gap</td><td>0.421</td><td>0.432</td><td>+0.011</td></tr><tr><td>Gap ∆ 95% CI</td><td></td><td></td><td>[-0.013, +0.036]</td></tr></table>

Table 35: Question-aligned re-scoring on the original study stratified audit sample. The gap-change interval is reported for the paired difference.

## G.5 Dual-rater probe and answer audit

The final audit comprises 405 question items, including 120 already reviewed in the pilot and 285 additional items. Forty-five deliberately forced cases are excluded from population estimates; the remaining 360 sampled probes span 37 tasks and are analyzed using the recorded sampling weights. Pilot items are part of the final set rather than an independent replication. On the 285 newly audited probes, both annotators marked every question answerable (raw agreement 1.000); referencecorrectness agreement was 0.989. Because these fields have little or no response variation, their kappa values are not informative. Their agreement on whether exact literal detail was required was 0.744 (Cohen’s $\kappa = 0 . 4 4 2 )$ . The ambiguous “overspecification” field was discontinued after low pilot agreement and was not part of the final audit.

Each item supplies two condition-blinded answers (report-only and full-context). Both annotators independently scored all 810 condition-blinded answers (two per item), with raw agreement 0.888, Cohen’s κ = 0.783 and weighted $\kappa = 0 . 8 5 1$ . The final human scoring rule averages the two annotators’ item scores; it is not an adjudicated consensus. All reported contrasts apply the same partial-credit convention to machine and human scores.

Let i index one of the 360 non-forced probes and t index a task. Denote the machine score under view $v \in \{ U , T \}$ by $m _ { i } ^ { v }$ and the two annotators’ mean by $h _ { i } ^ { v } = ( h _ { i , A } ^ { v } + h _ { i , B } ^ { v } ) / 2 \colon$ ; all judgments use correct = 1, partial = 0.5, and wrong/unknown = 0. The per-probe difference between machine and human view gaps is

$$
d _ { i } = ( m _ { i } ^ { T } - m _ { i } ^ { U } ) - ( h _ { i } ^ { T } - h _ { i } ^ { U } ) .\tag{5}
$$

With the original probe weight $w _ { i }$ and the probe set $I _ { t }$ for task t, the two aggregation rules are

$$
\begin{array} { r l } & { \Delta _ { \mathrm { t a s k } } = \displaystyle \frac { 1 } { 3 7 } \sum _ { t = 1 } ^ { 3 7 } \frac { \sum _ { i \in I _ { t } } w _ { i } d _ { i } } { \sum _ { i \in I _ { t } } w _ { i } } , } \\ & { \Delta _ { \mathrm { p o o l e d } } = \displaystyle \frac { \sum _ { i } w _ { i } d _ { i } } { \sum _ { i } w _ { i } } . } \end{array}\tag{6}
$$

(7)

The pre-registered primary analysis gives each task equal weight, whereas the pooled estimate weights tasks in proportion to their total probe weight. Both use the same $d _ { i } ;$ they differ only in aggregation. The task-clustered percentile bootstrap uses 2,000 resamples and seed 0.

<table><tr><td>Human rule</td><td>Task-paired ∆</td><td>95% CI</td></tr><tr><td>Two-rater mean (partial=0.5)</td><td>+0.0139</td><td>[-0.0137,+0.0454]</td></tr></table>

Table 36: Machine-minus-human view-gap contrast under a common partial-credit rubric. The frozen sample contains 360 non-forced probes from 37 tasks. The CI uses 2,000 task-clustered bootstrap samples (seed 0).

After correcting all machine and human partial judgments to receive 0.5 credit, the weighted machine and two-annotator-mean view gaps are 0.4097 and 0.4045, respectively. The prespecified task-equal paired contrast is $\Delta _ { \mathrm { t a s k } } = + 0 . 0 1 3 9$ (task-clustered 95% CI $[ - 0 . 0 1 3 7 , + 0 . 0 4 5 4 ] \rangle$ ). Its interval includes zero; it does not establish equivalence, nor does it provide a correction to the mainsample actor gap.

![](images/6ffe6a71681efb0ea19da2832f87bea2bfd6dadbcf8d041541e3e46656cd72c5.jpg)  
Figure 5: Observer-side residue under machine, consensus-adjusted, and mean-annotator-adjusted judgments, using the complete audited population.

<table><tr><td>Measurement</td><td>Estimate</td><td>95% CI</td></tr><tr><td>Restored omission  $r _ { G }$ </td><td>0.718</td><td></td></tr><tr><td>Matched-known rN</td><td>0.525</td><td></td></tr><tr><td>Paired G — N</td><td>0.189</td><td>[0.102, 0.280]</td></tr></table>

Table 37: Instruction changes relative to the report-only arm. Arm rates are unit-level; the paired contrast uses task means.

## G.6 Documented probe defects

For item P0001 in p2\_django\_\_django-15957 at the final checkpoint, the report identifies an ordering-test result as earlier and not rerun, whereas the reference treats it as confirmed in the final turn. For P0024 in p2\_pytest-dev\_\_pytest-7982 at the testing checkpoint, the probe and reference name reproduction directories differently from the report even though the corresponding test nodes express the same substantive finding. The original audit also records P0028 as a reference contradiction. Item identifiers and the audited defect count are retained for data repair; the complete population’s defect frequency was not estimated by this sample.

## H Supervision Interventions and Continuations

Simulated instruction-change diagnostic. The paired instruction-change comparison is visualized in Figure 6; it measures instruction variation rather than demonstrated improvements in code outcomes.

## H.1 Arms and endpoints

The unit is a task–checkpoint–atom triple. The report-only condition supplies U; the restoredinformation condition adds the omitted atom; the matched-known condition adds an atom already available to the reader. The observer produces its next instruction under a fixed prompt. The supervision endpoint records whether that instruction changes relative to the report-only instruction. Inferability is evaluated separately by asking whether the omitted fact can be reconstructed from the reports.

Table 38 retains the inferability sample, fullsample instruction changes, uptake flags, and their intersection. The matched-control analysis in the main text uses the paired subset recorded in its frozen contrast report. Unit-level arm means and task-level paired means have different weighting, so their displayed difference need not equal the paired estimate.

<table><tr><td>Quantity</td><td>Count Base</td><td></td><td>Rate</td></tr><tr><td>Non-inferable omissions</td><td>241</td><td>286</td><td>0.843</td></tr><tr><td>Instructions changed</td><td>206</td><td>286</td><td>0.720</td></tr><tr><td>Uptake flag</td><td>149</td><td>286</td><td>0.521</td></tr><tr><td>Changed + uptake</td><td>138</td><td>286</td><td>0.483</td></tr><tr><td>Fact-targeting check</td><td>1</td><td>7</td><td>0.143</td></tr><tr><td>Behavior: informed wins</td><td>4</td><td>14</td><td></td></tr><tr><td>Behavior: control wins</td><td>0</td><td>14</td><td></td></tr><tr><td>Solved fraction range</td><td></td><td></td><td>0.73-0.88</td></tr></table>

Table 38: Supervision diagnostics. The full inferability sample and the matched-control sample are separate. Uptake flags and their intersection with instruction change are reported separately.

## H.2 Fact targeting and continuation outcomes

The manual manipulation check assessed whether restored-information instructions explicitly targeted the supplied fact. The inspected instructions were specific and actionable, but explicit targeting was uncommon. The continuation experiment then compared agent behavior under informed and control instructions. Only a small subset of pairs was discordant, favoring the informed condition. The recorded success range indicates that many checkpoints were already solved, limiting the available variation for studying final outcomes.

![](images/51fcf4f21be5f9346f2e24c8754f6c6c797e47d572c425b784f4c68396bc2b0e.jpg)

![](images/4d9cf6cc14b6ab8302ea2c0b1f1465b11fb2170db5ee9738b4828ccf3a39231c.jpg)  
Figure 6: Simulated supervision: simulated supervisory instruction changes under report-only (U), omitted-fact reveal (G), and matched-known reminder (N) conditions. The paired contrast is $G - N = + 1 8 . 9$ percentage points (task-clustered 95% CI [+10.2, +28.0]). This is an instruction-change endpoint, not a measured improvement in final task success.

Instruction change, fact targeting, subsequent behavior, and final success are distinct endpoints. The matched-known control captures responses to additional already-available information; variation in relevance and actionability can remain. The observations in Table 38 preserve these separate levels of the experiment.

## H.3 Exploratory action-dependence diagnostic

The frozen matched-known contrast has 284 distinct (run, k, qid) units from 72 tasks. Original report-only and restored-fact instructions are preserved in the experiment records; the matchedknown instructions were recovered from complete generation logs. There are no missing instructions for the 284 matched units. The additional diagnostic extracts action indications from these existing $U , G ,$ and N instruction texts and applies a lexical proxy for dependence on the supplied fact. Its reported task-clustered G–N contrast is +0.127 (95% $\mathrm { C I } \left[ + 0 . 0 4 2 , + 0 . 2 1 3 \right] )$ .

This lexical proxy diagnoses whether generated instructions refer to the supplied fact; it does not measure action causality. The one-generation-perarm design also leaves sampling variability unestimated. The original instruction-change and facttargeting analyses retain their separate endpoints.

## I Matched Total-Report-Length Replay (E2)

The replay compares original cumulative agent reports (A), a natural rewrite (B), and a targeted rewrite (C). Both rewrites use the full recorded execution context; they are counterfactual alternatives to the report available to the original user. Its primary paired, length-filtered population contains 129 checkpoint units from 37 runs, with 4,175 valid probes. The observation stage includes 61 failed calls treated as missing draws, not as scored failures; available replicate counts (three/two/one) are 118/9/2 in A, 107/19/3 in B, and 109/17/3 in C. The final judgment stage completed 1,100/1,100 calls without failure. Unit recovery uses the mean of available observer samples, followed by the recorded paired task-cluster contrast.

<table><tr><td colspan="2">Report condition Accuracy</td></tr><tr><td>A: original accumulated U B: natural rewrite C: targeted rewrite</td><td>.6716 .7545 .7566</td></tr><tr><td>Paired contrast</td><td>∆ 95% CI</td></tr><tr><td>B - A +.0829 C-A +.0850</td><td> $[ + . 0 5 7 2 , + . 1 0 4 2 ]$   $[ + . 0 6 3 4 , + . 1 0 5 8 ]$ </td></tr></table>

Table 39: E2 equal-total-length replay: 129 paired units, 37 runs, 4,175 valid probes. Task-cluster intervals. $C - B$ is not significant; A is the original agent report, whereas B and C access full execution context. The E2 subset differs from the main-sample measurement population.

The narrower 113-unit sensitivity subset constrains the between-arm difference to at most 10% of original report length and gives $C \mathrm { ~ - ~ } B \mathrm { ~ = ~ }$ $+ . 0 0 1 9 [ - . 0 1 7 1 , + . 0 1 9 9 ]$ . Both rewrites use $T _ { \mathrm { f u l l } } .$ whereas A is the original agent report; thus $B - A$ and $C - A$ jointly reflect available execution evidence and reporting choices. The analysis covers the 129 observed paired units; the remaining 19 units of the broader 148-unit pool were not evaluated. The interval for $C - B$ includes zero and does not establish policy equivalence.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Development gap atoms</td><td>2,116</td></tr><tr><td>Target category coverage threshold</td><td>0.70</td></tr><tr><td>Selected category coverage</td><td>0.764</td></tr><tr><td>Matched generation budget (tokens)</td><td>1,032</td></tr><tr><td>Observed matched median tokens</td><td>951</td></tr><tr><td>Development / held-out checkpoints</td><td>184 /120</td></tr></table>

Table 40: Development selection and generation budgets recorded in the study.

## J Reporting Interventions

## J.1 Task split and category selection

The split is performed at task level, stratified by repository and difficulty. Table 4 gives development and held-out sizes, and Table 7 gives the fixed seed. The supplied split uses a development majority, rather than equal halves. Category priorities are selected from development data before generating targeted held-out reports.

For category c, the selection score is

$$
\operatorname { S c o r e } ( c ) = P ( c \mid \operatorname { g a p } ) P ( { \mathrm { n o n - i n f e r a b l e } } \mid c , \operatorname { g a p } ) .\tag{8}
$$

Categories are ordered by this score, and the shortest prefix reaching the fixed coverage threshold is retained. The selected set contains failures, facts, progress, and environmental state. Table 40 records the threshold, observed development coverage, and token budget.

## J.2 Report generation and budget

The natural condition asks for a summary of current progress. The structured condition follows a fixed schema. Targeted generation uses the selected category priorities and full checkpoint context; the budget-matched version applies the natural-summary token budget. The generator does not receive held-out gap labels, atom text, probe questions, reference answers, inferability labels, or hidden-test outcomes.

The category ranking is a corpus-level training choice. At generation time it is fixed, while the trajectory provides task-specific content. The experiments evaluate this learned priority scheme; a manually specified category checklist is a separate design. Budget matching uses a token constraint, while the cost plot reports observed median characters. Both budget and observed length are retained in the tables.

A separate 116-unit report-budget diagnostic completed without unresolved failures; 22 units required format-fallback processing. With partial judgments scored as 0.5, the recorded accuracies are U 0.396, full context 0.867, random 0.188, Ext 0.390, and gold atom 0.852. Here Ext denotes the extractive diagnostic arm; its precise span-selection rule is not established by the available report and is not treated as an oracle. The nominal budget was non-binding, and Ext did not outperform U on its recorded endpoint. This diagnostic does not establish an additional budget-constrained advantage and uses a different population from the held-out reporting experiment.

## J.3 Recovery and availability

The primary reporting endpoint is recovery of previously omitted non-inferable state, measured by probe accuracy under the report condition. This differs from the instruction-change endpoint of the supervision experiment. Some checkpoints contain no eligible non-inferable probes; targeted conditions are also missing for some checkpoints. The eight unobserved targeted and matched outputs are precisely four checkpoints each for the Django task 15098 and SymPy task 22714. The available heldout result tables establish the missing cells but do not record whether the cause was a generator error, aborted collection, or lost output. We consequently do not assign a mechanism or impute their results; the $n = 1 1 2$ estimand conditions on observed outputs. Table 3 reports available outcome and cost denominators separately.

Contrasts use the shared eligible units for each pair and average differences at task level. Table 41 gives all non-inferable contrasts, and Table 42 retains the overall recovery contrasts. Naturalsummary additions by category are reported in Table 43 and Figure 8; these evaluate additions to reports in the main sample and have a different population from held-out report-design comparisons.

![](images/b1af691b1d10e48fa9e1006b055d3695d0cd372f37f4de4d59a7029678b3c0f6.jpg)  
Figure 7: Report cost and recovery on held-out checkpoints. On this held-out added-summary comparison, targeted reports have higher recovery than natural summaries, including under a matched generation budget; this is not the equal-total-length E2 contrast. Points use available-unit recovery means and median report lengths; endpoint and cost sample sizes are reported separately in Table 3.

<table><tr><td>Contrast</td><td>Units Tasks</td><td></td><td>∆95%CI</td><td></td></tr><tr><td>TM − Nat</td><td>50</td><td>25</td><td>0.077</td><td>[0.030, 0.133]</td></tr><tr><td>Targeted — Nat</td><td>50</td><td>25</td><td>0.568</td><td>[0.455, 0.672]</td></tr><tr><td>Struct — Nat</td><td>54</td><td>27</td><td>0.250</td><td>[0.164, 0.347]</td></tr><tr><td>TM — Struct</td><td>50</td><td>25</td><td>-0.188</td><td>[-0.292, -0.097]</td></tr><tr><td>Targeted — TM</td><td>50</td><td>25</td><td>0.492</td><td>[0.378, 0.597]</td></tr><tr><td>Targeted — Struct</td><td>50</td><td>25</td><td>0.303</td><td>[0.167, 0.420]</td></tr></table>

Table 41: Task-paired non-inferable recovery contrasts. TM denotes targeted reporting under the naturalsummary budget; Nat denotes natural summaries, and Struct denotes structured summaries.

<table><tr><td>Contrast</td><td>Units</td><td>Tasks</td><td>Delta</td><td>95% CI</td></tr><tr><td>matched – nat</td><td>112</td><td>28</td><td>0.049</td><td>[0.036, 0.062]</td></tr><tr><td>targeted – nat</td><td>112</td><td>28</td><td>0.171</td><td>[0.149, 0.194]</td></tr></table>

Table 42: Task-paired overall recovery contrasts.

<table><tr><td>Category</td><td>Probes</td><td>Before</td><td>After</td></tr><tr><td>Failures</td><td>437</td><td>0.823</td><td>0.810</td></tr><tr><td>Rejected options</td><td>120</td><td>0.643</td><td>0.640</td></tr><tr><td>Environment</td><td>326</td><td>0.617</td><td>0.576</td></tr><tr><td>Progress</td><td>987</td><td>0.390</td><td>0.369</td></tr><tr><td>Assumptions</td><td>102</td><td>0.384</td><td>0.394</td></tr><tr><td>Facts</td><td>1866</td><td>0.376</td><td>0.377</td></tr><tr><td>Decisions</td><td>298</td><td>0.329</td><td>0.327</td></tr><tr><td>Plans</td><td>129</td><td>0.310</td><td>0.289</td></tr><tr><td>Risks</td><td>415</td><td>0.285</td><td>0.275</td></tr><tr><td>Constraints</td><td>119</td><td>0.261</td><td>0.274</td></tr></table>

Table 43: Within-category actor gap before and after adding a natural summary.

![](images/e0b54d49b9d7e20894af0453f40d59f44538d039b2a5a12084f8901768f43dcd.jpg)  
Figure 8: Within-category actor gap before and after adding a natural summary to reports. All categories are shown; the horizontal axis measures a within-category accuracy difference.

## J.4 Realized added-text budget matching

Unlike a summary-only diagnostic, this replay matches the original RQ3 information view: each observer receives all accumulated reports U plus one additional summary. The unchanged U is byteidentical across natural/targeted arms at each checkpoint. The additional summaries use paired sample identifiers $s 0  s 0 , s 1  s 1$ , and $s 2  s 2 .$ with the same probes and observer/judge templates. Original RQ3 chars aggregates lengths over all three samples; it is not a single selected summary. The separate summary-only replay (87 units; targeted minus natural −2.71 points) lacks U and is not compared as a causal counterfactual for RQ3.

<table><tr><td>Tolerance</td><td>Units</td><td>Tasks</td><td>Natural</td><td>Targeted</td><td>∆ [95% CI]</td></tr><tr><td>≤ 10%</td><td>47</td><td>25</td><td>.6728</td><td>.6967</td><td>+.0239 [.0105,.0357]</td></tr><tr><td>≤ 15%</td><td>79</td><td>30</td><td>.6811</td><td>.7063</td><td>+.0252 [.0145,.0349]</td></tr></table>

Table 44: Overall recoverability under matched realized added-summary character budgets, preserving original accumulated reports U. Intervals use task-cluster bootstrap with three paired samples. This outcome is not the non-inferable-state subset in Table 3.

A within-unit additional-text budget is set from the paired summary lengths. Cutting is restricted to complete semantic units: no mid-sentence cuts, padding, or expansions are used. Length is audited on realized additional material, while identical U and prompt scaffolding are retained. All three samples must pass the designated tolerance, leading to 47 units/25 tasks at ≤ 10% relative length discrepancy and 79 units/30 tasks at ≤ 15%. In the stricter set, mean additional budget is 1,358 characters, mean added-text retention is .838 (natural) and .865 (targeted); in the broader set, it is 1,321 characters with retention .837/.845. These are approximately matched reading budgets, not an exactly equalized word/token input.

The 10% filter drops five tasks; their original gap averaged .3487 compared with .3968 among retained tasks, so the matched cohort is not a random draw. The broader tolerance restores all 30 tasks. The main 10% analysis reports complete observation and judging; for the 15% extension, the execution report records 472 successful outcomes among 474 planned unit–arm–sample observations, with two repeated failures, while all 79 units remain represented. Exact effects of nonrandom failure and selection are not established. The positive estimates do not quantify the share of an earlier unmatched advantage attributable to length because truncation, material composition, and remeasurement also change.

## K Cross-model Probe and Observer Validity

Study design and protocol control. We examine two sources of model dependence without changing the frozen execution/report views. An observeronly check replaces the original HY4-preview observer with DeepSeek-v4-flash on HY4-generated probes, reusing the original prompting protocol. Across 37 tasks, DeepSeek-observer gaps are .4296 when HY4-preview judges its answers and .4152 when GLM-5.3-flash judges them. This check uses the original HY4-generated probes and a separate evaluation protocol.

<table><tr><td>Probe</td><td>Observer</td><td>Acc(U)</td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td></tr><tr><td>HY4</td><td>HY4</td><td>.5206</td><td>.8995</td><td>.3789</td></tr><tr><td>HY4</td><td>DeepSeek</td><td>.5012</td><td>.8462</td><td>.3450</td></tr><tr><td>DeepSeek</td><td>HY4</td><td>.4879</td><td>.8874</td><td>.3995</td></tr><tr><td>DeepSeek</td><td>DeepSeek</td><td>.4564</td><td>.8547</td><td>.3983</td></tr></table>

Table 45: Cross-generator/observer comparison under one answering and judging protocol, on the same 413 fully paired atoms (37 tasks; GLM judge). All entries are unweighted atom-level means; the original mainresult $\mathrm { G a p } _ { \mathrm { v i e w } } = . 4 0 6 6$ uses a different population and scoring protocol.

A cross-generator study independently generates probes and references with DeepSeek from 442 frozen task-state atoms (148 task–checkpoint cells, 37 tasks), without exposing the corresponding HY4 probe/reference texts to the generator. For the four-way comparison we use one observer protocol in all four generator–observer cells: up to three atom-targeted questions per call, the same prompt template and view, temperature zero, one answer sample, per-question identifiers, and the same GLM-5.3-flash judge (1 for correct, .5 for partial, 0 for wrong/unknown). All four generator– observer cells are evaluated anew under this protocol. All generation and sampling decisions precede the evaluated answers. The reported values are unweighted item means on paired atoms, with task-cluster bootstrap intervals; they should not be equated to the original 4,799-probe, differently weighted and graded headline estimate $( \mathrm { G a p } _ { \mathrm { v i e w } } = . 4 0 6 6 )$

Complete four-way pairs. The same four generator–observer combinations have both U and $T _ { \mathrm { f u l l } }$ scores for 413 of the 442 frozen atoms, spanning all 37 tasks. The execution audit attributes the 29 unavailable four-way pairs to 17 missing answer/grade cases and 12 atoms without a paired HY4 probe. Table 45 shows the complete-pair values; the pre-answering semantic-comparison labels are used only in the sensitivity check below.

Semantic-alignment sensitivity. Of these 413 pairs, 374 were pre-labeled same\_state=yes; the remainder comprise 28 labeled no and 11 unassessed. The gaps within the 374 aligned pairs are .3810 (HY4 probe/HY4 observer), .3489 (HY4/DeepSeek), .3984 (DeepSeek/HY4), and .3837 (DeepSeek/DeepSeek). Their task-cluster paired comparisons are reported in Table 47; the intervals for all four between-model contrasts include zero.

<table><tr><td colspan="2">Within-atom contrast (sec-  $\Delta \mathrm { G a p } _ { \mathrm { v i e w } }$  ond – first)</td><td>95% CI</td></tr><tr><td>DS vs. HY4 probe; HY4 ob- server</td><td>+.0206[-.0216,.0644]</td><td></td></tr><tr><td>DS vs. HY4 probe; DS ob- server</td><td>+.0533</td><td>[.0049,.1076]</td></tr><tr><td>DS vs. HY4 observer; HY4</td><td></td><td>-.0339[-.0744,.0048]</td></tr><tr><td>probe DS vs. HY4 observer; DS probe</td><td></td><td>-.0012 [-.0352,.0351]</td></tr></table>

Table 46: Cross-generator/observer contrasts on 413 complete atom-matched instances. Percentile 95% CIs use 2,000 bootstrap resamples clustered by task (37 tasks; fixed seed). The second contrast excludes zero nominally, without a multiple-comparison adjustment.
<table><tr><td>Aligned-pair contrast</td><td> $\Delta \mathrm { G a p } _ { \mathrm { v i e w } }$  95% CI</td></tr><tr><td>DS vs. HY4 probe; HY4 ob- server</td><td>+.0174[-.0276,.0615]</td></tr><tr><td>DS vs. HY4 probe; DS ob-</td><td>+.0348 [-.0151,.0892]</td></tr><tr><td>server DS vs. HY4 observer; HY4</td><td>-.0321 [-.0766,.0090]</td></tr><tr><td>probe DS vs. HY4 observer; DS probe</td><td>-.0147[-.0517,.0247]</td></tr></table>

Table 47: Same-state matched subset $( n \ = \ 3 7 4 ,$ , 37 tasks), with the same paired, task-cluster bootstrap estimator.

Validity considerations. The 413-pair analysis yields a generator contrast of +.0533 conditional on the DeepSeek observer, compared with +.0348 in the semantically aligned subset, whose interval includes zero. Probe specificity varies by generator. Atom extraction in this factorial comparison is HY4-based; the separately evaluated DeepSeek chain also replaces extraction and grading (Appendix K.2).

## K.1 Literal-detail dependence (LLM-label sensitivity)

We test whether the recoverability contrast is confined to probes requiring exact strings (e.g., file paths, commands, or identifiers). Two model families (HY4-preview and DeepSeek-v4-flash) independently label literal-detail dependence from each probe’s question and reference, blinded to report/full-context answers and gap scores. In a 400-probe validation pool, 192 HY4 and 186 DeepSeek judgments were usable; 179 had both usable judgments, with agreement $\kappa ~ = ~ . 7 7 6 4 .$ Twenty disagreements are excluded from the twomodel-concordant analysis, leaving 159 probes across 21 tasks. These are LLM-assessed categories, not verified human labels or a random sample of all 4,799 headline probes.

<table><tr><td>LLM label</td><td>Probe/task</td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>95% CI</td></tr><tr><td>Literal-dependent</td><td>85/19</td><td></td><td>.4785[.4011,.5509]</td></tr><tr><td>Non-literal</td><td>74/19</td><td></td><td>.3690[.2902,.4436]</td></tr></table>

Table 48: Recovery by two-model-concordant literaldetail label, under frozen probe weights and 10,000 task-cluster bootstrap draws (seed 0). The two groups jointly cover 21 tasks and overlap on 17 tasks.

The task-paired difference (literal minus nonliteral) on the 17 tasks covered by both groups is +.1096 [ −.0081, +.2331 ], so the group difference is not distinguished from zero. The nonliteral subset alone retains a positive .3690 gap [.2902, .4436]. Each group covers only 19 tasks with unbalanced per-task item counts, and concordance/validity selection limits generalization. An exploratory eight-signal regex proxy (three or more matches indicating literal dependence) has precision .583 and recall .741 against the 159 concordant LLM labels; we therefore do not treat proxy threshold effects as validated group differences. This subset supports the existence of a gap among LLM-assessed non-literal probes, not a quantitative decomposition of the headline 40.7-point gap.

## K.2 Independent DeepSeek measurement chain

Protocol and comparison. To test measurementchain dependence beyond the preceding probegenerator/observer factorial, we rebuild the entire task-state measurement pipeline with DeepSeekv4-flash: state-atom extraction, probe/reference generation, report/full-context observation, and answer grading. On the same frozen 37 task trajectories and four checkpoints (148 cells), this pipeline constructs its own atoms and probes; views are paired within each pipeline, not across two pipelines’ questions. The baseline is the frozen HY4-preview instrument, recomputed through the authoritative eligible-probe filter, category weights, and task-cluster bootstrap (10,000 samples, fixed seed). It exactly reproduces the original mainsample .406582 gap (4,799 valid of 4,855 probes), providing a scoring-path check rather than a new headline result.

<table><tr><td>Pipeline (tasks)</td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>95% task CI</td></tr><tr><td>HY4 measurement (37)</td><td>.4066</td><td>[.373,.438]</td></tr><tr><td>DS measurement (37)</td><td>.3985</td><td>[.348,.456]</td></tr><tr><td>HY4 on DS execution (12)</td><td>.5468</td><td>[.462,.643]</td></tr><tr><td>DS on DS execution (12)</td><td>.4885</td><td>[.431,.549]</td></tr></table>

Table 49: Independent measurement-chain checks. The 37-task rows share frozen trajectories; the 12-task rows share independently collected DeepSeek execution trajectories. Each chain constructs different atoms and probes. The 12-task intervals are sensitive to individual tasks.

Between-chain estimands. The difference between the two separately probe-weighted pooled gaps is not the same estimand as an equal-taskweight paired contrast. Table 51 reports both with task-cluster bootstrap intervals, keeping each chain’s frozen eligibility rules and weights. For 37 tasks, the pooled difference is −.0081, whereas the equal-task paired contrast is −.0294, reflecting their different task weighting. On 12 DeepSeekexecutor trajectories, the corresponding estimates are −.058 and −.029. Because the pipelines construct different atoms and probes, these contrasts summarize their measured gaps rather than isolated effects of individual model roles.

Answer scoring and attribution checks. A fixed-probe cross-judge audit samples 296 probes from 36 tasks (276 eligible, 111 DeepSeek judging requests). View gaps remain positive under both original and DeepSeek graders (approximately .454 and .448); the task-paired change is −.014 [ −.051, +.025 ]. This equal-task contrast is likewise not simply the subtraction of the two pooled gaps.

Coverage and failure accounting. The DeepSeek chain extracts 8,144 atoms and generates 4,986 probes, compared with 6,887 atoms and 4,855 probes for HY4. The DeepSeek run excludes 186 probes (158 with full-context score zero, 28 with prior-only score one), versus 56 in the frozen HY4 measurement (54 and 2, respectively); its observer additionally omits 34 question identifiers, compared with zero for HY4. The two pipelines differ in probe eligibility, coverage, and answer completeness. The independent 12-task DeepSeek-executor check reuses previously collected DSH/DeepSeek trajectories. The resulting positive gaps extend the measurement-chain robustness evidence; Section 7 discusses their supervisory interpretation.

## L Cross-Harness Applicability and Diagnostics

## L.1 Shared-task comparisons

Claude Code, Pi, and DSH are measured on the same task set under the same instructions, checkpoint rule, underlying model, and instrument. The view gap compares report and full-context observers for every agent. DSH lacks the actor-fork interface, so the shared comparison uses the view metric. The complete accepted task–agent grid provides the per-task values used in all tables here.

Table 53 reports task-paired view-gap differences using Student t intervals; each includes zero. Because trajectory-generated questions vary across harnesses, their point estimates describe the observed within-harness mixtures and should not be interpreted as relative reporting quality.

<table><tr><td>Pair</td><td>Difference</td><td>95% CI</td></tr><tr><td>ClaudeCode – Pi</td><td>0.017</td><td>[-0.122, 0.155]</td></tr><tr><td>ClaudeCode – DSH</td><td>-0.083</td><td>[-0.172, 0.006]</td></tr><tr><td>Pi - DSH</td><td>-0.100</td><td>[-0.207, 0.008]</td></tr></table>

Table 53: Task-paired view-gap differences. Two-sided Student t intervals, df = 11, without multiplicity adjustment.

## L.2 Cross-executor validation

This is a cross-executor experiment: DeepSeekv4-flash performs complete coding-agent tasks, including its own tool interactions and reports, rather than substituting only for a probe-answering model. We hold the DeepSeek Harness (DSH), 12 task identifiers, the four-checkpoint definition, and the v3.2 measurement instrument fixed while changing the executing foundation model from HY4-preview to DeepSeek-v4-flash. Existing HY4 DSH trajectories are reused; the DeepSeek arm contains 12 completed trajectories, each with four checkpoints. Each arm has its own trajectory-grounded atoms and probes, so this is a task-paired transfer test, not a common-item model leaderboard. Report-only and full-context observers use the same scoring logic. The actor self-answering fork is unavailable for this collection; only the view gap is evaluated, with no surrogate actor scores.

Reproduction and sampling. The original DSH HY4 reference cells use the authoritative task-equal computation with three observer-answer samples per probe, yielding Acc(U) = .4289, Acc(T<sub>full</sub>) = .9187, and $\mathrm { G a p } _ { \mathrm { v i e w } } ~ = ~ . 4 8 9 8$ ; the 12 per-run records reproduce those frozen cells exactly. For a comparable executing-model comparison we reaggregate both arms using only the first answer sample (s = 0), rather than comparing the historical three-sample HY4 mean with single-sample DeepSeek. Eligibility requires prior-only recovery below one and full-context recovery above zero; within each checkpoint and category, probe scores are weighted by eligible-atom counts divided by sampled-probe counts. We first aggregate within tasks and then bootstrap tasks. The paired sample contains 1,826 valid probes from 1,890 HY4 candidates and 1,782 from 1,859 DeepSeek candidates; different executed states and question sets explain why these item counts need not match.

<table><tr><td></td><td colspan="4">DeepSeek independent measurement</td><td colspan="4">HY4 frozen measurement</td></tr><tr><td>Checkpoint</td><td>Valid n</td><td> $\operatorname { A c c } ( U )$ </td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>Valid n</td><td> $\operatorname { A c c } ( U )$ </td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td></tr><tr><td>k = 1</td><td>829</td><td>.557</td><td>.933</td><td>.376</td><td>789</td><td>.496</td><td>.917</td><td>.421</td></tr><tr><td>k = 2</td><td>1,215</td><td>.506</td><td>.937</td><td>.431</td><td>1,223</td><td>.468</td><td>.922</td><td>.453</td></tr><tr><td>k = 3</td><td>1,340</td><td>.507</td><td>.932</td><td>.425</td><td>1,337</td><td>.501</td><td>.940</td><td>.439</td></tr><tr><td>k = 4</td><td>1,416</td><td>.574</td><td>.938</td><td>.364</td><td>1,450</td><td>.579</td><td>.924</td><td>.345</td></tr></table>

Table 50: Per-checkpoint independent measurement-chain results on the same 37 frozen task trajectories (weighted valid-probe estimates). Both pipelines independently sample probes, so the rows are not matched question sets.

![](images/dbfa31bea70f31f4054621f3ba85632069265d7599e31535a0e96c83732fbf49.jpg)

![](images/1117890277b62fd4e731613cb9b0df9fe4b7d5107d1ea7874d8cf0498cc7617b.jpg)  
Figure 9: Execution–report view gaps across three coding-agent harnesses on 12 shared tasks, using one trajectory per harness and task. Error bars show task-level uncertainty; the instrument compares different trajectory-specific state probes.

<table><tr><td>Subset (DS – HY4)</td><td>Pooled ∆</td><td>Task-equal ∆</td></tr><tr><td></td><td>-.0081</td><td>-.0294</td></tr><tr><td>37 tasks</td><td> $^ { [ - . 0 5 8 , + . 0 4 7 ] } _ { - . 0 5 8 }$ </td><td> $[ - . 0 6 4 , + . 0 0 7 ]$  -.029</td></tr><tr><td>12 DS-executor tasks</td><td> $[ - . 1 4 9 , + . 0 2 9 ]$ </td><td> $[ - . 0 9 6 , + . 0 3 1 ]$ </td></tr></table>

Table 51: Distinct estimands for the DeepSeek-minus-HY4 measurement-chain contrast. Intervals appear below point estimates. Pooled ∆ subtracts chain-weighted aggregates; task-equal ∆ averages within-task differences (positive in 12/37 and 5/12 tasks). CIs use taskcluster resampling; probe sets differ.

<table><tr><td>Harness</td><td>Acc(U)</td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\mathrm { G a p } _ { \mathrm { v i e w } }$ </td><td>95% CI</td></tr><tr><td>Claude Code</td><td>.525</td><td>.932</td><td></td><td>.407[.336, .478]</td></tr><tr><td>Pi</td><td>.550</td><td>.940</td><td></td><td>.390[.289, .491]</td></tr><tr><td>DSH</td><td>.429</td><td>.919</td><td></td><td>.490[.417, .563]</td></tr></table>

Table 52: Execution–report recoverability for 12 unique issue IDs and 36 harness-specific trajectories (95% CIs: per-issue t, $d f = 1 1 )$ . The executor is HY4-preview throughout; each harness produces distinct trajectories and probes.

<table><tr><td>Executing model</td><td> $\operatorname { A c c } ( U )$ </td><td> $\operatorname { A c c } ( T _ { \operatorname { f u l l } } )$ </td><td> $\operatorname { G a p } _ { \mathrm { v i e w } }$ </td></tr><tr><td>HY4-preview  $( s = 0 )$ </td><td>.4264</td><td>.9487</td><td>.5223</td></tr><tr><td>DeepSeek-v4-flash  $( s = 0 )$ </td><td>.4212</td><td>.9482</td><td>.5270</td></tr></table>

Table 54: Cross-executor recoverability under fixed DSH, 12 tasks per model, single answer sample and task-equal aggregation. Task-bootstrap 95% CIs are [.4278, .6130] (HY4) and [.4596, .6085] (DeepSeek); paired $\Delta _ { \mathrm { D S - H Y 4 } } = + . 0 0 4 7 [ - . 1 0 8 9 , + . 1 5 3 5 ]$

Execution comparability and interpretation. Per-task means for HY4/DeepSeek are 376/345 atoms, 158/155 probes, 10,551/8,965 report-view characters, 572,569/711,948 full-context characters, 123/108 API requests, and 1,291/1,325 events. These descriptive statistics do not establish matched state difficulty, task completion, or a pure model effect. The positive gaps independently support the presence of an execution–report recoverability contrast in both tested executing models; the wide paired difference interval does not demonstrate equivalence or a ranking in reporting quality. The actor fork is not reported and downstream issue-resolution outcomes were not compared in this analysis.

## L.3 Additive decomposition

The cross-agent taxonomy partitions probe rows into omitted, reported-but-missed, misled, partially recovered, unclassified, and balancing components. The balancing component contains rows where the report reader scores at least as well as the fullcontext reader. The signed contributions sum to the view gap. Table 55 reports contributions as accuracy proportions and lists omission share, prevalence, and conditional recovery failure separately.

Omission is the largest positive contributor for every actor. Its contribution measures how much accuracy difference comes from omitted state; omission prevalence measures how often state is absent; omission share normalizes within positive typed mass. The share is averaged per task, so it is not generally the ratio of the displayed mean contribution to the displayed mean gap.

<table><tr><td>Quantity</td><td>Claude Code</td><td>Pi</td><td>DSH</td></tr><tr><td>View gap</td><td>0.407</td><td>0.390</td><td>0.490</td></tr><tr><td>Not conveyed</td><td>0.251</td><td>0.194</td><td>0.284</td></tr><tr><td>Reported, unrecovered</td><td>0.075</td><td>0.074</td><td>0.105</td></tr><tr><td>Wrong answer</td><td>0.030</td><td>0.053</td><td>0.049</td></tr><tr><td>Partial recovery</td><td>0.060</td><td>0.075</td><td>0.058</td></tr><tr><td>Unclassified</td><td>0.001</td><td>0.000</td><td>0.000</td></tr><tr><td>Balancing term</td><td>-0.010</td><td>-0.006</td><td>-0.006</td></tr><tr><td>Non-conveyance share</td><td>0.594</td><td>0.443</td><td>0.556</td></tr><tr><td>Non-conveyance rate</td><td>0.432</td><td>0.329</td><td>0.468</td></tr><tr><td>Failure given conveyed</td><td>0.125</td><td>0.127</td><td>0.193</td></tr></table>

Table 55: Cross-agent decomposition and diagnostics. Contributions are accuracy proportions; nonconveyance share is the mean per-task share of positive typed mass.

<table><tr><td>Pair</td><td>Difference</td><td>95% CI</td></tr><tr><td>ClaudeCode – Pi</td><td>0.057</td><td>[-0.073, 0.186]</td></tr><tr><td>ClaudeCode – DSH</td><td>-0.033</td><td>[-0.121, 0.055]</td></tr><tr><td>Pi - DSH</td><td>-0.089</td><td>[-0.198, 0.019]</td></tr></table>

Table 56: Task-paired omission contribution differences. Two-sided Student t intervals, df = 11, without multiplicity adjustment.

<table><tr><td>Pair</td><td>Difference</td><td>95% CI</td></tr><tr><td>ClaudeCode – Pi</td><td>0.104</td><td>[-0.011, 0.218]</td></tr><tr><td>ClaudeCode – DSH</td><td>-0.036</td><td>[-0.146, 0.074]</td></tr><tr><td>Pi - DSH</td><td>-0.140</td><td>[-0.239, -0.040]</td></tr></table>

Table 57: Task-paired omission prevalence differences. Two-sided Student t intervals, df = 11, without multiplicity adjustment.

<table><tr><td>Pair</td><td>Difference</td><td>95% CI</td></tr><tr><td>ClaudeCode – Pi</td><td>-0.001</td><td>[-0.087, 0.085]</td></tr><tr><td>ClaudeCode – DSH</td><td>-0.067</td><td>[-0.136, 0.001]</td></tr><tr><td>Pi - DSH</td><td>-0.066</td><td>[-0.161, 0.029]</td></tr></table>

Table 58: Task-paired conditional recovery-failure differences. Two-sided Student t intervals, df = 11, without multiplicity adjustment.

## L.4 Conditional recovery

Conditional recovery failure concerns conveyed atoms that the reader fails to recover. It is larger for DSH in the observed means. The paired diagnostics show that prevalence and conditional recovery need not follow the same uncertainty pattern as total view gaps. The tables retain the estimates and intervals for each quantity, allowing the reporting and recovery stages to be examined separately.

All cross-harness intervals use the same tasklevel Student t calculation; typed contributions and the balancing term sum to the signed view gap. Because each harness yields its own category distribution and probes, the displayed gaps describe the tested trajectories rather than a categorystandardized ranking. Resource totals are separately reported because batching, repeated reads, report length, and harness turn structure also affect usage.

## M Measurement Cost Accounting

## M.1 Observable usage and counting rules

Collection usage records the coding agents’ requests while producing trajectories; measurement usage records the benchmark calls that score those trajectories. Requests are identified by hashes. Effective usage takes the last successful usagebearing attempt for each unique request. Actual observable usage counts all attempts that return usage, including retries. Failed calls without a usage block are recorded in coverage counts.

Fresh input, cache-read input, and cachecreation input are separate token classes. Total input volume, when needed, is their sum. Thinking tokens are a subset of output tokens and are not added again. The present cost tables retain the fresh-input, output, and cache-read columns reported by the study. Cache-creation totals were not given in the original cost tables.

<table><tr><td>Quantity</td><td>Count</td></tr><tr><td>Cells with available usage</td><td>34</td></tr><tr><td>Cells with partial usage</td><td>2</td></tr><tr><td>Cells with unavailable usage</td><td>0</td></tr><tr><td>Partial cell: usable / total logs</td><td>628 / 631</td></tr><tr><td>Partial cell: usable / total logs Collection unique requests</td><td>530 / 531</td></tr><tr><td>Collection attempts</td><td>2,545</td></tr><tr><td></td><td>2,626</td></tr><tr><td>Failed / no-usage requests Claude Code / DSH / Pi requests</td><td>35 847 / 740 / 958</td></tr></table>

Table 59: Usage coverage and request counts recorded in the study.

The accounting covers the cross-agent collection runs and their measurement directories. Table 59 records usage coverage and request counts. Partially observed cells contain a small number of failed or empty calls without usage; observable totals include only usage-bearing records.

<table><tr><td>Actor</td><td>Eff. input</td><td>Eff. output</td><td>Cache read</td></tr><tr><td>Claude Code</td><td>1,554,998</td><td>846,179</td><td>40,115,328</td></tr><tr><td>DSH</td><td>1,520,598</td><td>649,127</td><td>40,074,112</td></tr><tr><td>Pi</td><td>3,482,598</td><td>1,867,266</td><td>34,649,536</td></tr><tr><td>Total</td><td>6,558,194</td><td>3,362,572</td><td>114,838,976</td></tr><tr><td>Actor</td><td>Actual input</td><td></td><td>Actual output</td></tr><tr><td>Claude Code</td><td>1,554,998</td><td></td><td>846,179</td></tr><tr><td>DSH</td><td></td><td>1,520,598</td><td>649,127</td></tr><tr><td>Pi</td><td></td><td>3,953,682</td><td>2,388,585</td></tr><tr><td>Total</td><td></td><td>7,029,278</td><td>3,883,891</td></tr></table>

Table 60: Collection traffic by coding agent. Effective usage keeps the final successful request; actual usage includes retries with reported tokens. Cache reads are separate.

## M.2 Collection and measurement

Table 60 separates effective collection usage from actual observable usage. Retry-related differences occur in Pi’s collection records; the other actors have identical effective and actual totals in the supplied accounting. Collection token volume depends on context management and interaction structure as well as report verbosity.

Table 61 gives effective measurement usage by pipeline stage. Table 62 groups the same measurement traffic by purpose. Core traffic includes extraction, probes, report and actor observers, judging, and semantic conveyance. Sensitivity traffic covers additional context views and summary arms. Other experiments and audit overhead remain separate from the core measurement account.

<table><tr><td>Stage</td><td>Input</td><td>Output</td><td>Cache read</td></tr><tr><td>Judge</td><td>22,134,005</td><td>925,035</td><td>3,107,264</td></tr><tr><td>Probes</td><td>21,678,906</td><td>814,186</td><td>941,440</td></tr><tr><td>Observer</td><td>17,420,919</td><td>5,541,630</td><td>32,200,704</td></tr><tr><td>Atoms</td><td>5,524,555</td><td>588,513</td><td>536,576</td></tr><tr><td>Semantic conveyance</td><td>2,266,075</td><td>492,617</td><td>168,000</td></tr><tr><td>Coverage</td><td>1,294,524</td><td>167,853</td><td>1,078,784</td></tr><tr><td>Audit overhead</td><td>218,783</td><td>1,021,353</td><td>0</td></tr><tr><td>Other</td><td>3,123,195</td><td>164,096</td><td>3,574,592</td></tr><tr><td>Total</td><td>73,660,962</td><td>9,715,283</td><td>41,607,360</td></tr></table>

Table 61: Effective measurement usage by pipeline stage.

<table><tr><td>Purpose</td><td>Input</td><td>Output</td></tr><tr><td>Core</td><td>53,491,784</td><td>3,674,329</td></tr><tr><td>Sensitivity</td><td>16,827,200</td><td>4,855,505</td></tr><tr><td>Other</td><td>2,813,018</td><td>150,674</td></tr><tr><td>Supervision</td><td>310,177</td><td>13,422</td></tr><tr><td>Audit overhead</td><td>218,783</td><td>1,021,353</td></tr></table>

Table 62: Effective measurement usage by purpose. Core includes atoms, probes, report/actor observers, judging, and semantic conveyance.

<table><tr><td>Metric</td><td>Collection</td><td>Measurement</td><td>Total</td></tr><tr><td>Input</td><td>6,558,194</td><td>73,660,962</td><td>80,219,156</td></tr><tr><td>Output</td><td>3,362,572</td><td>9,715,283</td><td>13,077,855</td></tr><tr><td>Input+output</td><td>9,920,766</td><td>83,376,245</td><td>93,297,011</td></tr><tr><td>Cache read</td><td>114,838,976</td><td>41,607,360</td><td>156,446,336</td></tr></table>

Table 63: Effective observable token totals. Cache reads are a separate token class, not added to the input+output headline.

## M.3 Interpreting the totals

The effective input-plus-output total in Table 63 excludes cache reads, which are listed separately. Measurement traffic exceeds collection traffic because each trajectory is processed by multiple components and observers. The accounting includes recorded benchmark calls outside the final headline analysis; the purpose breakdown identifies that broader scope.

Observed usage is an account of logged traffic. Different harnesses batch and revisit context differently, and their reports vary in length. Comparing communication recoverability at a common task protocol therefore addresses a different question from comparing resource efficiency.