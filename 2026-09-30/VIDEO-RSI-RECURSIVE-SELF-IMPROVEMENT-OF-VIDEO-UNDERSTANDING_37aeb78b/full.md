# VIDEO-RSI: RECURSIVE SELF-IMPROVEMENT OF VIDEO UNDERSTANDING AGENTS VIA HARNESS EVOLUTION

Bingjun Luo Tsinghua University

Jialin Guo Harbin Engineering University

Siqi Li Tsinghua University

## ABSTRACT

Video understanding agents acquire evidence through an executable harness that controls what they observe and how they use those observations. However, execution traces contain only the evidence acquired by the current harness, leaving competing explanations for failure unresolved and limiting the basis for self-improvement. We introduce Video-RSI, a framework for recursive selfimprovement in which a video understanding agent uses its own language model to revise its harness. Through active video investigation, the model revisits the original training videos to test competing failure explanations with additional observations, grounding proposed changes in evidence beyond the existing trace. Cost-aware harness evolution turns these diagnoses into reusable revisions and determines which revisions to retain by considering both answer accuracy and visual cost. Across our evaluation settings on video understanding benchmarks, the evolved agent improves accuracy while processing fewer frames and achieves competitive accuracy-efficiency trade-offs against existing video understanding agents. These results demonstrate the potential for video understanding agents to improve their own evidence acquisition and use through harness evolution. Code is available at https://github.com/bingjunluo/Video-RSI.

## 1 INTRODUCTION

Video understanding agents must decide what to observe before they can decide how to answer. In long videos, relevant events may be brief, temporally distant, or surrounded by visually similar distractions. Recent agents address this challenge through iterative retrieval, temporal navigation, and tool-guided observation (Wang et al., 2024; Zhang et al., 2025b; Lin et al., 2026). At each step, the agent must choose where to look, which details to inspect, and whether the evidence is sufficient to answer. These decisions are governed by an executable harness that controls evidence acquisition, processes observations, and supports subsequent reasoning. The harness shapes both what the model sees and how it uses that evidence, linking answer accuracy to the cost of visual observation. Improving this executable layer is therefore an opportunity to strengthen video understanding without changing model weights.

Automated agent design uses feedback to improve agent programs (Hu et al., 2025; Lee et al., 2026), and recent work extends code-level evolution to video understanding (Xu & Chen, 2026). Self-Harness further shows that a frozen model can revise its own operating harness in terminal environments (Zhang et al., 2026a). We study this form of harness self-improvement for video understanding. The language model that answers video questions also investigates failures and revises the code governing its behavior. As shown in Figure 1, adaptive evidence gathering changes what an agent observes within a question, while harness self-improvement changes the procedure it uses across subsequent questions. This makes execution experience a basis for revising how evidence is acquired and organized, allowing useful changes to accumulate in the program. Our objective is to turn execution experience into reusable improvements that increase answer accuracy while reducing visual observation, with the underlying models fixed.

This objective requires both evidence for a revision and a criterion for deciding whether to retain it. An execution trace contains only the observations acquired by the current harness. A missing event may therefore reflect a sampling gap, a perceptual error, or a failure to use available evidence, each requiring a different modification. For instance, an event absent from the recorded observations does not establish whether the agent failed to inspect it or misinterpreted what it saw. Revisiting the video can help distinguish these explanations before the code is changed. Yet correcting a failure does not necessarily produce a better harness. Denser sampling may improve an answer while increasing visual cost across many questions, whereas better evidence reuse may avoid redundant inspection. Useful self-improvement must connect a supported failure diagnosis to an assessment of the revised harness’s accuracy and visual cost over complete executions.

![](images/4efe080779ce05751e52b9a5b90be747b111e3b1c61adb6c93aa7d1162df0713.jpg)  
Figure 1: From answering video questions to improving the harness. (a) Direct MLLM inference uses supplied observations. (b) Video agents adapt evidence acquisition within a question. (c) Video-RSI uses the same frozen language model to investigate execution failures, revisit training videos, and revise the harness offline. Revisions are selected by accuracy and visual cost, producing a reusable harness for subsequent questions.

We introduce Video-RSI, a framework in which the same frozen language model serves as both Solver and Editor to recursively improve a video agent’s executable harness. Active Video Investigation revisits training videos to test failure explanations and inform code revisions. Cost-Aware Harness Evolution evaluates the resulting candidates by answer accuracy and visual cost, retaining revisions that become the starting point for further improvement. These mechanisms link evidence for what to change with criteria for which changes should accumulate. The resulting loop can revise observation strategies and evidence-processing routines together, assessing their combined effect on the behavior of the complete agent. Evolution takes place offline. The selected harness is then frozen for evaluation on new questions. Across four video understanding benchmarks, the evolved harness improves accuracy while processing fewer frames than its initial version and achieves competitive accuracy–efficiency trade-offs against existing video agents. On MLVU, active investigation also yields higher final accuracy than trace-only revision at similar inference-time frame usage.

Our contributions are threefold:

• We introduce Video-RSI, a framework for video harness self-improvement in which the same language model solves video questions, investigates failures, and revises its executable harness while model weights remain fixed.

• We develop an improvement loop that links active video investigation to diagnosis-guided code revision and accuracy–cost selection, addressing both the evidence for a modification and its value for subsequent executions.

• We evaluate Video-RSI across four video understanding benchmarks and analyze investigation and evolution dynamics, showing improvements over the initial harness in both accuracy and frame usage and the benefit of investigation over trace-only revision in the MLVU comparison.

## 2 RELATED WORK

## 2.1 AGENTIC VIDEO UNDERSTANDING

Video agents address long videos by deciding which evidence to acquire and how to organize it for reasoning. Iterative retrieval and hierarchical representations allow VideoAgent and VideoTree to concentrate on question-relevant content (Wang et al., 2024; 2025b), while VCA guides exploration with intrinsic rewards and a bounded visual memory (Yang et al., 2025). Beyond frame selection, document retrieval (Ma et al., 2025) and multi-granular tool use (Zhang et al., 2025b; Lin et al., 2026) connect video observations to successive reasoning steps. DVD combines global browsing, clip search, and direct frame inspection. VideoSeek provides complementary overview, skim, and focus operations. MR.Video uses MapReduce to reconcile entities and aggregate question-specific analyses across segments (Pang & Wang, 2025), while LVAgent combines the observations and reasoning of multiple agents through dynamic collaboration (Chen et al., 2025).

Another line of work learns observation policies through model training. FrameThinker uses supervised fine-tuning and reinforcement learning to coordinate reasoning with additional frame acquisition (He et al., 2026). EVA further learns to allocate temporal windows, frame counts, and spatial resolution, and uses failure cases to guide the generation of additional training questions (Zhang et al., 2026d). Video-RSI instead keeps the underlying models fixed and evolves the executable harness that governs evidence acquisition and use. Its improvement loop revisits training videos to diagnose failures before making program changes that apply across questions.

## 2.2 AUTOMATED AGENT DESIGN AND SELF-IMPROVEMENT

Automated agent design treats the programs surrounding language models as an optimization space. ADAS searches over agents expressed in code, while AFlow searches executable workflows (Hu et al., 2025; Zhang et al., 2025a). These formulations allow changes to the composition and control of model calls beyond isolated prompt edits. Meta-Harness extends this direction by using historical code, evaluation scores, and execution traces to optimize harnesses around a fixed model (Lee et al., 2026). This program-level view motivates our choice of an executable harness as the object of improvement, encompassing both evidence-processing tools and the logic that uses their outputs.

Self-improvement also concerns which system proposes the changes. STOP studies an improvement program that modifies itself (Zelikman et al., 2024), and the Darwin Godel Machine combines self-¨ modification with exploration over an archive of agents (Zhang et al., 2026b). Self-Harness uses the same frozen model to propose harness revisions and validates candidates through regression testing (Zhang et al., 2026a). Video-RSI builds on this code-level self-improvement setting, with the same language model serving as Solver and Editor. Our focus is the evidence available to that Editor. An execution trace records only what the current harness observed, so investigating the original video can distinguish failure explanations that would otherwise motivate different revisions.

## 2.3 SELF-EVOLVING VIDEO AGENTS

Video self-improvement has been explored through the generation and refinement of training signals. EvoGround develops a proposer–solver learning loop for temporal grounding, Video-Zero organizes questioner–solver co-evolution around local temporal evidence, and EvoVid emphasizes temporal information in self-evolution (Jung et al., 2026; Zhang et al., 2026c; Huang et al., 2026). These methods improve video capabilities through model training. Video-RSI instead accumulates improvements in executable code while keeping model weights fixed.

VideoHarness-RSI is a closely related approach to video harness evolution (Xu & Chen, 2026). It evolves context-construction programs that can coordinate retrieval, evidence representations, and auxiliary model calls around frozen vision–language models, and promotes candidates that improve validation accuracy. Video-RSI shares this program-level search setting and focuses on how the Editor obtains evidence for a revision. Active investigation of the original videos tests failure explanations before code is edited. Our selection rule also accounts for visual cost, admitting accuracy gains with bounded cost growth or cost reductions with bounded accuracy loss. Together, investigation and selection connect reusable harness changes to their effects on answer quality and observation requirements.

![](images/55a389216a4569a5366ebbd862ff957739a3c5b856a3b50671f6c39f4f7b5e58.jpg)  
Figure 2: Overview of Video-RSI. The Solver and Editor share a frozen language model. Active video investigation revisits training videos to diagnose failures and guide harness revisions. Costaware harness evolution selects candidates using accuracy and visual cost on a private development set, carrying accepted revisions into subsequent iterations.

## 3 VIDEO-RSI

Video-RSI evolves the executable harness of a video question-answering agent while keeping the underlying models fixed. An Editor revisits the original video to investigate failures, uses the resulting diagnosis to revise the harness, and submits the candidate to a performance–cost gate. Accepted revisions become the starting point for the next evolution attempt. Figure 2 provides an overview.

## 3.1 PROBLEM FORMULATION AND EVOLVABLE HARNESS

A harness H governs how an agent acquires and uses evidence to answer a video question. Given a video V , a question q with answer choices, and fixed model services M, the Solver produces an answer yˆ, an execution trace τ, and a resource record c:

$$
( \hat { y } , \tau , c ) = \mathrm { R u n } ( H , V , q ; \mathcal { M } ) .\tag{1}
$$

The trace records tool calls, observations, and decisions. The Editor can jointly revise the harness code for prompt construction, tool implementation, observation processing, memory, and execution control.

These components connect evidence acquisition to decision-making by determining what to inspect, how observations are represented and retained, and when to gather more evidence or answer. Their coordination matters because even a correct observation can lose its temporal or entity associations before reaching the answering step. Harness evolution can therefore change both the observations acquired and how they support reasoning.

We use a training set ${ \mathcal { D } } _ { \mathrm { t r } }$ for investigation and revision and a fixed, video-disjoint selection set $\mathcal { D } _ { \mathrm { g } }$ Selection examples and per-item outcomes remain private to the evaluator.

## 3.2 FRAMEWORK OVERVIEW

Video-RSI uses the same frozen language model as Solver and Editor, accumulating improvements in executable code (Xu & Chen, 2026). At each attempt, the Solver executes $H _ { t }$ on training questions. The Editor examines its traces and revisits videos to test failure explanations before proposing a candidate ${ \widetilde { H } } _ { t }$ . Independent evaluation on the private selection set determines whether the candi date replaces $H _ { t }$ for the next attempt. Investigation occurs offline. Additional observations guide reusable code changes, while the selected harness answers subsequent questions using their own inputs.

## 3.3 ACTIVE VIDEO INVESTIGATION

An execution trace may not reveal whether an incorrect answer stems from missing an event, misperceiving it, or failing to use existing evidence. Active investigation lets the Editor revisit the original video to distinguish these explanations before revising the harness.

From failure explanations to diagnostic queries. The Editor examines training outcomes, traces, and code, prioritizing representative failures that additional evidence could clarify. Queries vary the temporal region, sampling density, modality, or perceptual question to test competing explanations. For example, denser inspection can reveal whether a short event fell between sampled frames. The interface supports video observations, speech transcripts, on-screen text, and reruns of the unmodified incumbent. Perception requests omit the gold answer.

Let $\mathcal { T } _ { 0 }$ contain the current code, training traces, available labels, and prior feedback. The Editor selects queries $u _ { k }$ and accumulates returned evidence $O _ { k } .$

$$
\begin{array} { r l } & { u _ { k } = \mathrm { Q u e r y } _ { \mathrm { E d i t o r } } ( \mathcal { T } _ { k } ) , } \\ & { o _ { k } = \mathrm { I n v e s t i g a t e } ( u _ { k } ; H _ { t } , \mathcal { D } _ { \mathrm { t r } } , \mathcal { M } ) , } \\ & { \mathcal { T } _ { k + 1 } = \mathcal { T } _ { k } \cup \{ ( u _ { k } , o _ { k } ) \} . } \end{array}\tag{2}
$$

Each query specifies a training example and the observation needed to test an explanation. The Editor interprets the accumulated evidence to choose the next query, while $H _ { t }$ remains unchanged.

From new evidence to a revision hypothesis. New observations can support, contradict, or leave an explanation unresolved, redirecting subsequent queries. Finding a missed event supports revising temporal coverage. If it was already observed, the diagnosis instead examines how its evidence was represented and used. Investigation ends with a supported revision hypothesis or an unresolved case when no further informative query is available.

The Editor compares relevant cases to identify a reusable change, its applicable conditions, and successful behaviors to preserve. Training annotations may locate informative intervals during diagnosis, but the revision must use cues available to the Solver on new questions. The diagnosis therefore connects observed failures to a procedure for finding and using evidence without knowing the answer or event location in advance.

## 3.4 COST-AWARE HARNESS EVOLUTION

Diagnosis-guided revision. The Editor maps the diagnosis to code changes, specifying when they apply and which successful behaviors to preserve. Revisions can span the harness components in Section 3.1, such as changing inspection cues or preserving temporal relationships in observations and memory. A reusable change may require coordinated edits to a tool’s output and the code that consumes it. The Editor can debug candidates through training-side trials, distinct from the pre-edit investigation, before submitting a candidate for selection.

Cost-aware selection. After execution checks, the evaluator compares the candidate and incumbent on the same private selection examples under matched conditions. It measures answer accuracy and visual cost, defined as mean frames processed by successful visual-model calls over complete question-answering executions. This end-to-end assessment matters because a more elaborate tool can reduce later inspection, while a locally useful repair can increase overall visual usage.

Let $A _ { t } , C _ { t }$ denote the incumbent’s accuracy and mean frame count on the selection set, and $\widetilde { A } _ { t } , \widetilde { C } _ { t }$ the corresponding candidate measurements. With $\Delta A _ { t } = \widetilde { A } _ { t } - A _ { t }$ , the gate admits two improvement directions:

$$
\begin{array} { r } { g _ { t } = \underbrace { [ \Delta A _ { t } > 0 ] \wedge [ \widetilde { C } _ { t } \leq ( 1 + \alpha ) C _ { t } ] } _ { \mathrm { a c c u r a c y ~ g a i n ~ w i t h ~ b o u n d e d ~ c o s t ~ g r o w t h } } } \\ { \vee \underbrace { [ - \epsilon \leq \Delta A _ { t } \leq 0 ] \wedge [ \widetilde { C } _ { t } \leq ( 1 - \beta ) C _ { t } ] \wedge [ C _ { t } > 0 ] } _ { \mathrm { c o s t r e d u c t i o n ~ w i t h ~ b o u n d e d ~ a c c u r a c y ~ l o s s } } . } \end{array}\tag{3}
$$

Here, $\alpha \geq 0$ bounds relative cost growth, $0 < \beta < 1$ sets the required relative cost reduction, and $\epsilon \geq 0$ bounds accuracy loss. Their values are fixed during evolution and specified in the experimental setup. If $g _ { t } = 1$ , the candidate becomes $H _ { t + 1 }$ . Otherwise, $H _ { t + 1 } = H _ { t }$ . The Editor receives only aggregate accuracy and frame counts to guide subsequent revisions. Per-example gate traces remain private. The selected harness is finally evaluated on held-out data. Algorithm 1 summarizes the complete evolution process.

```latex
Algorithm 1 Recursive harness evolution in Video-RSI
Input: initial harness ${ \overline { { H _ { 0 } } } } .$ , fixed services M, training set $\mathcal { D } _ { \mathrm { t r } } ,$ private gate ${ \mathcal { D } } _ { \mathrm { g } } .$ attempt budget $\overline { { T } } .$
Output: final accepted harness $H _ { T }$
for $t = 0 , \ldots , T - 1$ do
Execute $H _ { t }$ on training examples and inspect its traces.
Initialize investigation context $\mathcal { I } _ { 0 }$ and set $\overset { \cdot } { k } = 0 .$
while further investigation is warranted and resources permit do
Select a query to distinguish unresolved failure explanations.
Acquire evidence and construct $\mathcal { T } _ { k + 1 }$ using Eq. (2).
Revise the supported explanations and remaining questions.
Set $k \gets k + \hat { 1 }$
end while
Synthesize a diagnosis and plan a reusable behavior change.
Edit $H _ { t }$ into a candidate $\widetilde { H } _ { t }$ and run training-side checks.
Validate and compare the candidate with $\bar { H _ { t } }$ on ${ \mathcal { D } } _ { \mathrm { g } } .$
Accept or retain the incumbent according to Eq. $( 3 ) .$
end for
return $H _ { T }$
```

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks and data splits. We evaluate on the long-video splits of Video-MME (Fu et al., 2025) and LongVideoBench (Wu et al., 2024) with subtitles, the official public subset of EgoSchema (Mangalam et al., 2023), and MLVU test (Zhou et al., 2025). For evolution, we use 216 LVBench questions (Wang et al., 2025a) for investigation and revision, and another 72 questions as a fixed, video-disjoint development set for candidate selection. Development examples and per-question outcomes remain private. The Editor receives only aggregate feedback.

Models and evolution protocol. The Solver and Editor use DeepSeek-V4-Pro, with Qwen3.6- Plus for visual observations. All model weights are frozen. The initial harness $H _ { S 0 }$ supports global and local video inspection, transcripts, OCR, and observation memory. We run 20 revision attempts using Eq. (3) with $\begin{array} { r } { \dot { \alpha } = 0 . 1 , \beta = \bar { 0 } . 2 , } \end{array}$ and $\epsilon = 1 / | \mathcal { D } _ { \mathrm { g } } |$ , allowing at most one fewer correct answer in the cost-reduction branch. The final accepted harness is frozen for the main evaluation, while intermediate accepted versions are evaluated to analyze evolution dynamics. Active-investigation and trajectory-only variants share training examples, annotation access, initialization, models, and per-attempt resource limits, differing only in pre-edit access to new video observations.

Baselines. We compare with MLLMs, video agentic models, and harness self-improvement methods. We reproduce VideoSeek (Lin et al., 2026) and VideoHarness-RSI (Xu & Chen, 2026) using the same model (DeepSeek-V4-Pro) as Video-RSI. Results for the remaining methods are taken from their original publications. The initial harness $H _ { S 0 }$ serves as the reference for evolution gains. Trajectory-only revision and an accuracy-only gate assess the contributions of investigation and candidate selection.

Evaluation metrics. We report multiple-choice accuracy and mean visual frames per question. For our evaluations, frame usage sums all frames processed by successful visual-model calls during each complete execution, then averages across questions. Offline Editor observations are excluded.

Table 1: Video understanding results. Acc. denotes accuracy (%) and Frames denotes visual frame usage. Bold indicates column-best values. † denotes official M-Avg.
<table><tr><td rowspan="2">Method</td><td colspan="2">MLVU Test</td><td colspan="2">LongVideoBench Long val</td><td colspan="2">Video-MME Long</td><td colspan="2">EgoSchema</td></tr><tr><td></td><td>Acc.↑ Frames↓ Acc.↑</td><td></td><td></td><td></td><td></td><td></td><td>Frames↓ Acc.↑ Frames↓ Acc.↑ Frames↓</td></tr><tr><td colspan="9">MLLMs</td></tr><tr><td>GPT-40</td><td>54.9†</td><td>0.5 fps</td><td>60.9</td><td>256</td><td>65.3</td><td>384</td><td>70.4</td><td>一</td></tr><tr><td>Gemini 1.5 Pro</td><td></td><td></td><td>58.6</td><td>256</td><td>67.4</td><td>1233</td><td></td><td></td></tr><tr><td>Gemini 2.0 Flash</td><td></td><td></td><td>45.7</td><td>256</td><td>63.0</td><td>1233</td><td>71.2</td><td>一</td></tr><tr><td>Qwen2.5-VL-72B</td><td></td><td></td><td></td><td></td><td>53.2</td><td>256</td><td>一</td><td>一</td></tr><tr><td>GPT-5</td><td></td><td></td><td>64.5</td><td>384</td><td>67.9</td><td>384</td><td>一</td><td>一</td></tr><tr><td colspan="9">Video Agentic Models</td></tr><tr><td>VideoAgent</td><td>一</td><td></td><td>一</td><td></td><td>46.4</td><td>24.6</td><td>60.2</td><td>8.4</td></tr><tr><td>VideoTree</td><td></td><td></td><td></td><td></td><td>53.1</td><td>98.0</td><td>66.2</td><td>62.4</td></tr><tr><td>DrVideo</td><td></td><td></td><td></td><td></td><td>51.7</td><td>493.2</td><td>66.4</td><td></td></tr><tr><td>VCA</td><td></td><td></td><td></td><td></td><td>54.2</td><td>18.1</td><td>73.6</td><td>7.2</td></tr><tr><td>MR.Video</td><td></td><td></td><td>61.6</td><td>2816</td><td>63.4</td><td></td><td>73.8</td><td>一</td></tr><tr><td>DVD</td><td></td><td></td><td>68.6</td><td></td><td>67.3</td><td>4932</td><td>76.6</td><td>一</td></tr><tr><td>FrameThinker</td><td></td><td></td><td></td><td></td><td>47.6</td><td>24.1</td><td></td><td>一</td></tr><tr><td>LVAgent</td><td></td><td></td><td>一</td><td></td><td>74.3</td><td></td><td>一</td><td>一</td></tr><tr><td>EVA-GRPO</td><td></td><td></td><td></td><td></td><td>48.4</td><td>22.8</td><td></td><td></td></tr><tr><td>VideoSeek</td><td>68.1</td><td>83.9</td><td>70.9</td><td>71.9</td><td>78.7</td><td>17.8</td><td>73.2</td><td>72.3</td></tr><tr><td colspan="9">Harness Self-Improvement</td></tr><tr><td>VideoHarness-RSI</td><td>64.1</td><td>83.2</td><td>68.4</td><td>92.1</td><td>69.7</td><td>92.8</td><td>78.0</td><td>66.8</td></tr><tr><td>Video-RSI (Ours)</td><td>72.9</td><td>61.1</td><td>71.1</td><td>41.1</td><td>80.0</td><td>21.6</td><td>78.2</td><td>42.4</td></tr></table>

## 4.2 MAIN RESULTS

Table 1 places Video-RSI alongside MLLMs, video agentic models, and harness self-improvement methods. Our analysis focuses on the reproduced VideoSeek and VideoHarness-RSI implementa tions. Published results provide broader context. With its harness evolved on LVBench and frozen for evaluation, Video-RSI achieves higher accuracy than both reproduced baselines across all four benchmarks, with lower frame usage in all but the VideoSeek comparison on Video-MME.

Relative to VideoSeek, the accuracy–cost balance varies across benchmarks. On MLVU, Video-RSI gains 4.8 percentage points while using 27.2% fewer frames, combining better answers with reduced visual usage. On LongVideoBench, the main benefit is substantially fewer frames at similar accuracy. On EgoSchema, both accuracy and frame usage improve. On Video-MME, higher accuracy accompanies a moderate increase in frame usage. Thus, the advantage is not a uniform reduction in observation, but a favorable balance between answer quality and visual usage across different evaluation settings.

Compared with VideoHarness-RSI, Video-RSI achieves higher accuracy with fewer frames on every benchmark. The accuracy gain is largest on Video-MME, whereas EgoSchema shows closely comparable accuracy with substantially lower frame usage. The evolved harness thus offers two practical benefits by improving answer quality and reducing the observations needed for comparable performance. The controlled comparisons in Section 4.3 further examine how active investigation and cost-aware selection contribute to the resulting harness.

## 4.3 ABLATION STUDY

Table 2 examines how harness evolution depends on the evidence available before revision and the criterion used to retain candidates. The trajectory-only variant restricts the Editor to existing execution traces, while active investigation allows additional video observations before editing. Both use 20 revision attempts under the shared conditions in Section 4.1. The accuracy-only gate variant examines selection without the visual-cost criterion. We evaluate the resulting harnesses on all four benchmarks, using the initial harness as a reference.

Table 2: Ablations of active investigation and candidate selection. Acc. is accuracy (%). Frames is mean visual frames per question.
<table><tr><td rowspan="2">Variant</td><td colspan="2">MLVU Test</td><td colspan="2">LongVideoBench Long val</td><td colspan="2">Video-MME Long</td><td colspan="2">EgoSchema</td></tr><tr><td>Acc.↑ Frames↓</td><td></td><td>Acc.↑</td><td>Frames↓</td><td>Acc.↑ Frames↓</td><td></td><td>Acc.↑ Frames↓</td><td></td></tr><tr><td>Initial harness</td><td>62.4</td><td>68.5</td><td>67.7</td><td>51.6</td><td>73.4</td><td>38.6</td><td>71.4</td><td>66.0</td></tr><tr><td>Trajectories only</td><td>64.7</td><td>60.3</td><td>67.2</td><td>43.1</td><td>73.7</td><td>28.5</td><td>76.6</td><td>51.8</td></tr><tr><td>Accuracy-only gate</td><td>67.1</td><td>81.9</td><td>69.5</td><td>66.4</td><td>74.9</td><td>43.1</td><td>77.4</td><td>60.5</td></tr><tr><td>Video-RSI</td><td>72.9</td><td>61.1</td><td>71.1</td><td>41.1</td><td>80.0</td><td>21.6</td><td>78.2</td><td>42.4</td></tr></table>

Active investigation improves the resulting harness. Active investigation yields higher final accuracy than trajectory-only revision on every benchmark. On MLVU, the gain reaches 8.2 percentage points with similar inference-time frame usage. On the other three benchmarks, higher accuracy accompanies fewer frames. This pattern links additional evidence during offline revision to better deployed behavior without a consistent increase in inference-time observation. The investigationenabled run also retains five of twenty revisions, compared with three for trajectory-only revision. Alongside the endpoint results, these observations support using evidence beyond the existing trace to guide reusable harness changes.

Cost-aware selection improves the accuracy–cost balance. The accuracy-only gate improves accuracy over the initial harness on all four benchmarks, but increases frame usage on three. Video-RSI achieves higher accuracy than this variant while using fewer frames on every benchmark. On Video-MME, it uses roughly half as many frames. Thus, the cost-aware run obtains visual savings without sacrificing endpoint accuracy in these evaluations. The comparison shows why accuracy gains alone do not fully characterize a useful harness revision. Selecting for both objectives can retain more efficient behavior while preserving improvements in answer quality.

## 4.4 EVOLUTION DYNAMICS

Figure 3 tracks the retained harness over 20 revision attempts on the development set and MLVU. The MLVU evaluations were performed retrospectively after evolution and did not inform candidate selection. On the development set, the first accepted revision produces most of the net reduction in frame usage, while subsequent revisions steadily improve accuracy with smaller, nonmonotonic changes in visual cost. The run thus combines an early efficiency gain with continued refinement of answer quality.

The MLVU trajectory reveals that improvements on the selection set need not transfer at every step. The first accepted revision reduces frame usage but lowers MLVU accuracy, despite improving both development metrics. Accuracy recovers with the next revision, and the final two accepted revisions account for most of the net MLVU accuracy gain. The final harness therefore reflects cumulative refinement beyond the initial efficiency improvement. An early development gain alone would give an incomplete picture of its evaluation behavior.

Accuracy and visual cost also follow different paths during later refinement. From $H _ { S 2 }$ onward, MLVU accuracy rises together with frame usage, yet the final harness remains more accurate and uses fewer frames than $\bar { H } _ { S 0 }$ . This pattern is consistent with the selection rule, which permits bounded cost increases for accuracy gains, rather than requiring frame usage to decrease at every update. Later revisions can use more frames while retaining a net efficiency advantage over the initial harness, improving the final accuracy–cost balance.

![](images/cac36928b8759162396da6ef8c004fc77df3cbcffad1ebb94afca3e9945ddafc.jpg)  
Figure 3: Evolution over 20 revision attempts. Accuracy and mean frames per question on $^ { ( \mathrm { a } , \mathrm { b } ) }$ the development set and $^ { ( \mathrm { c } , \mathrm { d } ) }$ MLVU. Curves track the retained harness. Vertical guides mark accepted revisions, and flat intervals indicate an unchanged incumbent.

## 4.5 ANALYSIS OF EVOLVED HARNESSES

Comparing the initial harness with the retained revisions reveals three changes in how the Solver acquires and uses evidence:

Structured perception. In $H _ { S 2 }$ , free-form visual responses give way to goal-directed observations with structured findings, image references, and explicit uncertainty. The harness maps image references to frame timestamps, preserving the link between a finding and its temporal evidence. This interface lets follow-up reasoning target unresolved details and their supporting frames without changing model weights.

Evidence reuse through relation graphs. Introduced in $H _ { S 4 }$ , the graph mechanism extracts entities, events, and relations from accumulated visual, transcript, and OCR observations. For eligible relational or temporal questions, it returns evidence gaps that can guide further inspection. Graph queries themselves process no new frames, making evidence organization an intermediate step between existing observations and additional acquisition.

Selective observation routines. The Global-64 route in $H _ { S 3 }$ expands the initial 32-frame overview for selected whole-video questions. The occurrence tool in $H _ { S 5 }$ defines counting units, verifies candidate events, and consolidates overlapping evidence to avoid duplicate counts. These routines coexist with transcript-based localization and observation reuse. The final harness improves accuracy while using 10.8–44.1% fewer frames than $H _ { S 0 }$ across the four benchmarks, showing that richer observation tools can coexist with lower aggregate visual usage.

These retained changes illustrate how evolution modifies both evidence acquisition and its use in reasoning. The reported savings measure visual frames rather than total model computation.

## 5 CONCLUSION

We presented Video-RSI, a framework for improving video understanding agents through harness evolution with fixed model weights. The same language model serves as Solver and Editor, revisiting training videos to test failure explanations and translating the resulting diagnoses into reusable code changes. Cost-aware selection retains revisions according to both answer accuracy and visual usage. Across four benchmarks, the evolved harness improves accuracy while using fewer frames than it initial version, with ablations supporting the contributions of active investigation and cost-aware selection. Analysis of the retained code reveals changes in structured perception, evidence reuse, and selective observation. Together, these findings support harness evolution as a practical route to improving how frozen-model video agents acquire and use evidence.

## AI USE STATEMENT

We used generative AI tools to assist with code editing, figure and table preparation, and reviewing the manuscript for clarity of expression. The authors reviewed the AI-assisted content and take responsibility for the final manuscript, code, and scientific claims.

## REFERENCES

Boyu Chen, Zhengrong Yue, Siran Chen, Zikang Wang, Yang Liu, Peng Li, and Yali Wang. LVAgent: Long video understanding by multi-round dynamical collaboration of MLLM agents. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20237– 20246, October 2025. doi: 10.1109/ICCV51701.2025.01882. URL https://doi.org/10. 1109/ICCV51701.2025.01882.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24108–24118, 2025.

Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, and Yu Cheng. Framethinker: Learning to think with long videos via multi-turn frame spotlighting. In International Conference on Learning Representations, volume 2026, pp. 74904–74933, 2026.

Shengran Hu, Cong Lu, and Jeff Clune. Automated Design of Agentic Systems. In International Conference on Learning Representations, volume 2025, pp. 21344–21377, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html.

Shiqi Huang, Ziyue Wang, Zhongrong Zuo, Han Qiu, Qi She, and Bihan Wen. EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models, 2026. URL https://arxiv.org/ abs/2605.21931.

Minjoon Jung, Byoung-Tak Zhang, and Lorenzo Torresani. EvoGround: Self-Evolving Video Agents for Video Temporal Grounding, 2026. URL https://arxiv.org/abs/2605. 13803.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-Harness: End-to-End Optimization of Model Harnesses, 2026. URL https://arxiv.org/ abs/2603.28052.

Jingyang Lin, Jialian Wu, Jiang Liu, Ximeng Sun, Ze Wang, Xiaodong Yu, Jiebo Luo, Zicheng Liu, and Emad Barsoum. VideoSeek: Long-horizon video agent with tool-guided seeking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5465–5475, June 2026. URL https://openaccess.thecvf.com/ content/CVPR2026/html/Lin\_VideoSeek\_Long-Horizon\_Video\_Agent\_ with\_Tool-Guided\_Seeking\_CVPR\_2026\_paper.html.

Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. DrVideo: Document retrieval based long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18936–18946, June 2025. doi: 10.1109/CVPR52734.2025.01764. URL https://openaccess.thecvf.com/ content/CVPR2025/html/Ma\_DrVideo\_Document\_Retrieval\_Based\_Long\_ Video\_Understanding\_CVPR\_2025\_paper.html.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In Advances in Neural Information Processing Systems, volume 36, pp. 46212–46244, 2023.

Ziqi Pang and Yu-Xiong Wang. MR. Video: MapReduce as an effective principle for long video understanding. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 6640–6672. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-0231. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, and Jie Tang. LVBench: An Extreme Long Video Understanding Benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22958–22967, October 2025a. URL https: //openaccess.thecvf.com/content/ICCV2025/html/Wang\_LVBench\_An\_ Extreme\_Long\_Video\_Understanding\_Benchmark\_ICCV\_2025\_paper.html.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. VideoAgent: Long-form video understanding with large language model as agent. In Computer Vision – ECCV 2024, volume 15138 of Lecture Notes in Computer Science, pp. 58–76. Springer Nature Switzerland, 2024. doi: 10.1007/978-3-031-72989-8 4. URL https://doi.org/10.1007/ 978-3-031-72989-8\_4.

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3272–3283. IEEE, 2025b.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for longcontext interleaved video-language understanding. In Advances in Neural Information Processing Systems, volume 37, pp. 28828–28857, 2024.

Guoyang Xu and Hao Chen. VideoHarness-RSI: Recursive harness self-improvement for long-video understanding with frozen vision-language models, 2026. URL https://arxiv.org/abs/ 2608.24302.

Zeyuan Yang, Delin Chen, Xueyang Yu, Maohao Shen, and Chuang Gan. VCA: Video curious agent for long video understanding. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20168–20179, 2025.

Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (STOP): Recursively self-improving code generation. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=46Zgqo4QIU.

Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. Self-Harness: Harnesses That Improve Themselves, 2026a. URL https: //arxiv.org/abs/2606.09498.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Open-¨ ended evolution of self-improving agents. In International Conference on Learning Representations, volume 2026, pp. 104223–104294, 2026b.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, XiongHui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In International Conference on Learning Representations, volume 2025, pp. 34040–34077, 2025a.

Ruixu Zhang, Deyi Ji, Lanyun Zhu, Xuanyi Liu, Yuxin Meng, Ruihang Chu, and Yujiu Yang. Video-Zero: Self-Evolution Video Understanding, 2026c. URL https://arxiv.org/abs/2605. 14733.

Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. Deep video discovery: Agentic search with tool use for long-form video understanding. Advances in Neural Information Processing Systems, 38:89863–89895, 2025b.

Yaolun Zhang, Ruohui Wang, Jiahao Wang, Yepeng Tang, Xuanyu Zheng, Haonan Duan, Hao Lu, Hanming Deng, and Lewei Lu. EVA: Efficient reinforcement learning for end-to-end video agent. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12289–12299, 2026d.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: Benchmarking multi-task long video understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13691–13701, 2025.