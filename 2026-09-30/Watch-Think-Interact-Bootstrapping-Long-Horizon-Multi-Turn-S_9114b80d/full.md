# Watch-Think-Interact: Bootstrapping Long-Horizon Multi-Turn Streaming Video Reasoning with Reinforcement Learning

Ziheng Huang<sup>1∗</sup>, Yicheng Bao<sup>1∗</sup>, Xueheng Li<sup>2∗</sup>, Zhenkun Gao<sup>1∗</sup>, Bangwei Liu<sup>1∗</sup>, Kunquan Li<sup>3</sup>, Yuxiang Shen<sup>3</sup>, Bangyan Li<sup>1</sup>, Xuejiao Wang<sup>1</sup>, Changbo Wang<sup>1</sup>, Gaoqi He<sup>1†</sup>

<sup>1</sup>East China Normal University <sup>2</sup>University of Science and Technology of China <sup>3</sup>Xiamen University 51275901129@stu.ecnu.edu.cn gqhe@cs.ecnu.edu.cn

## Abstract

Multi-turn streaming video reasoning must answer asynchronous questions from observed prefixes under a bounded online state. Existing methods form this state before future questions are known, making details discarded by forwardonly compression irrecoverable when their relevance emerges later. We introduce Watch-Think-Interact (WTI), a unified approach that preserves on-demand access to observed source evidence under a bounded online state. Our solution comprises a closed-loop inference framework, causally aligned interaction data, and trajectory-level optimization. At inference, WTI couples compact natural-language memory with source-video time ranges, supporting direct reasoning and selective visual recall. Its policy answers when evidence sufices, waits when evidence has not appeared, or recalls an observed interval and redecides at the same time step without replaying the full history. WTI-82K organizes 82,335 timed questions into 4,812 trajectories aligning query times, answerable moments, evidence, state updates, and actions. It supervises when to answer, wait, recall, and update memory across a shared stream. Stream-GDPO optimizes complete multi-turn rollouts using separately normalized outcome, format, recall, and memory rewards. It aligns training with stateful streaming decisions whose early actions shape later states. WTI sets a new opensource state of the art, reaching 83.3% on StreamingBench and 73.6% on OVO-Bench while leading its Real-Time, Backward, and Forward groups.

## Introduction

Streaming video assistance requires a model to answer asynchronous questions as a video unfolds, using only the observed prefix and an evolving interaction state. Recent Video Large Language Models (Video-LLMs) have advanced offline video question answering, but most assume access to a complete video or pre-collected clip before inference (Li et al. 2024; Wu et al. 2024; Fu et al. 2025a). In streaming settings, frames arrive continuously and future frames are unavailable. For each question, the model should respond as soon as the active visual context and retained history provide suficient evidence; otherwise, it must first supplement its state by revisiting previously observed visual evidence or continuing to observe until the required evidence becomes available to the model.

Recent streaming Video-LLMs model response states or online decisions over observed prefixes (Chen et al. 2024; Qian et al. 2025; Fu et al. 2025b; Chen et al. 2025; Xia et al. 2025; Wang et al. 2026; Guan et al. 2026; Yang et al. 2026). Other work uses online cache and memory management for streaming inputs (Xu et al. 2026; Zhang et al. 2025a; Zeng et al. 2025), or visual-token and hierarchical compression for pre-collected videos (Shen et al. 2024; Li et al. 2026). In causal streaming, however, the retained state is formed before subsequent questions are known; visual details excluded by forward-only compression therefore become unavailable when later questions reveal their relevance. Further reasoning over that state cannot recover omitted evidence; opaque latent summaries also obscure what remains and when it was observed in the stream.

We therefore introduce Watch-Think-Interact (WTI), a closed-loop framework coupling active-window perception, compact time-indexed memory, response timing, and selective source-video recall. Each memory entry pairs a semantic summary for direct reasoning with a source time range that localizes finer visual evidence when the summary is insuficient. For every active question, a learned policy answers when the visible state sufices, remains silent when required evidence has not appeared, or recalls a relevant observed interval. Recall returns evidence without advancing the stream, updates the visible state, and triggers a new decision at the same time step. This action–feedback–redecision loop avoids replaying the full observed history while preserving on-demand access to source evidence. Figure 1 illustrates how compact memory and selective recall reconnect the online state with source evidence.

To train this closed-loop behavior, we construct WTI-82K, which organizes 82,335 timed questions into 4,812 causally aligned multi-turn trajectories linking query times, answerable moments, supporting evidence, state updates, and interaction actions. Relative to the 8-second active visual window, 56.9% of trajectories span at least 120 seconds (15× the window), and 21.7% span at least 240 seconds (30×), so most trajectories extend far beyond the visible context. Masked SFT initializes the interaction protocol, while Stream-GDPO extends GDPO (Liu et al. 2026) into a trajectory-level reinforcement-learning objective for complete multi-turn rollouts, optimizing response timing, sourcevideo recall, and memory updates throughout the stream. WTI sets a new state of the art among open-source streaming models, reaching 83.3% on StreamingBench (Lin et al. 2024) and 73.6% weighted overall accuracy on OVO-Bench (Niu et al. 2025).

![](images/03f96c8b5024646c155516bded2c3f89b29abc73df9a8c0c15352ec639582f3d.jpg)  
Figure 1: Motivation for Watch-Think-Interact. In causal streaming video, asynchronous questions may require current, past, or not-yet-observed evidence. WTI uses a bounded active window for current perception, compact time-indexed memory for direct reasoning and temporal localization, and selective recall to reload a past visual interval.

In summary, our contributions are as follows:

• We introduce Watch-Think-Interact, a closed-loop framework for bounded multi-turn streaming reasoning. Its time-indexed memory and selective recall recover source evidence when later questions reveal its relevance.

• We construct WTI-82K, comprising 82,335 timed questions in 4,812 causally aligned trajectories. It supplies supervision for answering, waiting, recall, and memory updates across a shared stream.

• We develop Stream-GDPO for trajectory-level reinforcement learning over complete multi-turn rollouts. Its separate normalization of outcome, format, recall, and memory rewards aligns optimization with stateful streaming decisions throughout each rollout.

• WTI sets a new open-source state of the art on StreamingBench and OVO-Bench. It leads OVO-Bench’s Real-Time, Backward, and Forward groups while retaining strong ofline long-video performance.

## Related Work

Streaming Video Understanding. Video-LLMs have progressed from ofline long-video reasoning, where the full video is available before inference, to online benchmarks requiring prefix-only answers, timestamped queries, and temporal multi-turn interaction (Li et al. 2024; Wu et al. 2024; Fu et al. 2025a; Niu et al. 2025; Huang et al. 2025; Yang et al. 2025). Streaming systems add response-state modeling, feedback policies, proactive interaction, streaming thinking, or instruction tuning (Chen et al. 2024; Qian et al. 2025; Wang et al. 2026; Guan et al. 2026; Xia et al. 2025). These methods decide when to answer from the observed prefix; WTI addresses later turns whose source-linked evidence has left the active visual window.

Long-Stream Memory. Long-stream methods use visualtoken or hierarchical compression, compact KV states, recurrent windows, and event memory to keep streaming context tractable (Shen et al. 2024; Li et al. 2026; Xu et al. 2026; Zhang et al. 2025a; Zeng et al. 2025). Their forward-only histories compress evidence before later questions arrive and provide no source-linked mechanism for reloading omitted visual details. WTI instead makes compact memory timeindexed and source-recallable.

Agentic Multimodal Reasoning. Agentic multimodal models acquire evidence through tools or additional visual inspection (Zheng et al. 2026; Hong et al. 2026; Gao et al. 2026), while memory agents learn to update or revisit compact state (Yu et al. 2026; Shi et al. 2026; Wang et al. 2026). WTI brings both capabilities to streaming video by deciding whether to wait, answer, or recall time-bounded history while transferring compact state at controller-defined boundaries.

![](images/999b60370ee70208f3f4ff8327f80a873d5c87cee7fa7ef20bb496d0cb05e15c.jpg)  
Figure 2: Overview of Watch-Think-Interact. For each arriving chunk, WTI updates a short thinking state and chooses to remain silent, answer, or recall. Recall returns visual evidence and loops back to the decision at the same time step; controller-triggered memory updates separately compress elapsed observations into a bounded temporal index. Stream-GDPO optimizes the resulting complete interaction trajectories.

## Method

## Problem Definition

We study online multi-turn streaming video reasoning, where a video arrives as an ordered stream of chunks:

$$
V = ( c _ { 1 } , \dots , c _ { T } ) .\tag{1}
$$

At step $t ,$ the observed prefix is $c _ { \leq t } : = ( c _ { 1 } , \ldots , c _ { t } )$ ; future chunks are unavailable.

The same stream may contain multiple questions arriving at diferent times:

$$
\mathcal { Q } = \{ ( q _ { r } , \tau _ { r } , \alpha _ { r } ) \} _ { r = 1 } ^ { R } .\tag{2}
$$

Here $\tau _ { r }$ and $\alpha _ { r }$ are the question and answer timestamps used for data construction and evaluation, with $1 \le \tau _ { 1 } \le \cdots \le \tau _ { R }$ and $\tau _ { r } \leq \alpha _ { r } \leq T$ . Although $\alpha _ { r }$ , the time when suficient visual evidence becomes available, is hidden at inference, the model must output one answer yˆ at $\hat { t } _ { r } \ge \tau _ { r } . \mathrm { A }$ response before $\alpha _ { r }$ violates streaming causality; otherwise it must be grounded in $c _ { < \hat { t } _ { r } }$

## Watch-Think-Interact Framework

We instantiate an online multimodal agent that receives chunk $c _ { t }$ at step t and acts at assistant subturn k from the bounded model-visible state

$$
s _ { t , k } = ( W _ { t } , M _ { t } , Q _ { t } , H _ { t } , E _ { t , k } ^ { \mathrm { r e c } } ) .\tag{3}
$$

Here $W _ { t }$ is the active visual window, $M _ { t }$ compact textual memory, $Q _ { t }$ the active question set, $H _ { t }$ interaction history, and $E _ { t , k } ^ { \mathrm { r e c } }$ recalled evidence. Time-indexed memory carries evidence boundaries beyond $W _ { t } ,$ while the observed prefix remains in a controller-managed source archive accessible only through recall.

Each assistant subturn combines a streaming thinking update with an interaction decision. The model first emits a short update $z _ { t , k }$ , then its learned policy selects $a _ { t , k } \sim \pi _ { \theta } ( \cdot \ |$ $s _ { t , k } , z _ { t , k } )$ from

$$
\begin{array} { r l } & { a _ { t , k } \in \mathcal { A } _ { \mathrm { i n t } } = \{ \mathrm { s i l e n c e , r e s p o n s e } ( q _ { r } , y ) , } \\ & { ~ \mathrm { r e c a l l } ( \tau _ { s } , \tau _ { e } ) \} . } \end{array}\tag{4}
$$

The actions silence and response respectively delay output and answer the current question, both terminating the current chunk; recall requests earlier evidence without advancing the stream.

A recall interval $[ \tau _ { s } , \tau _ { e } ]$ uses absolute source-video seconds and is valid only when $0 \leq \tau _ { s } \leq \tau _ { e } < \gamma _ { t }$ , where $\gamma _ { t }$ is the start time of $c _ { t }$ . The environment adds the returned evidence to $E _ { t , k + 1 } ^ { \mathrm { r e c } }$ , after which the policy updates $z _ { t , k + 1 }$ and decides again at the same step t. Recall is therefore a non-terminal action–feedback–redecision loop. Our protocol permits at most one recall invocation per current chunk, returning at most $K _ { r } = 4$ observed chunks.

At a controller-triggered memory boundary $b , Z _ { b }$ collects model-generated records since the previous boundary, and the model rewrites the pre-update memory:

![](images/e689a8bf69d57765842ee4456c040b83b89c0a687cde049eb4f4ccb7c002b4bd.jpg)  
Figure 3: WTI-82K construction pipeline. WTI-82K converts videos into streaming-causal trajectories by standardizing clips, aligning chunk-level evidence chains, generating QAs from aligned evidence, placing query and answerable times, and checking timestamp bounds, answer availability, action grammar, and evidence-interval fields. The resulting samples cover Real-Time, Backward Tracing, and Proactive interactions under the same visibility constraints as inference.

$$
M _ { \mathrm { p o s t } } = \mathrm { C o m p a c t } _ { \theta } ( M _ { \mathrm { p r e } } , Z _ { b } ) .\tag{5}
$$

Under the fixed token budget $\| M _ { \mathrm { p o s t } } \| _ { \mathrm { t o k } } \le B _ { M }$ , each line stores a concise semantic note and source time range. The semantic note supports direct reasoning when suficient, while the time range anchors selective inspection of finer sourcevideo evidence.

WTI can process arbitrarily long streams without expanding the model-visible context: every decision uses a fixed active window, budgeted textual state, and at most four returned chunks, while the controller-managed source archive grows with the observed stream.

## Masked Supervised Fine-Tuning

Masked SFT matches the causal, bounded inference state (Xu et al. 2026). At step t, its dense visual context is limited to the recent window

$$
W _ { t } = \{ c _ { \operatorname* { m a x } ( 1 , t - K _ { v } + 1 ) } , \ldots , c _ { t } \} ,\tag{6}
$$

where $K _ { v }$ is the visual window size; we use $K _ { v } \ = \ 8$ one-second chunks (16 frames). Without exposing the full prefix, compact memory $M _ { t } .$ , active questions $Q _ { t }$ , and teacher-forced interaction history $H _ { t }$ provide the remaining inference-time textual state.

We serialize stream inputs, assistant outputs, and environment observations temporally. A stream-causal mask restricts assistant tokens to earlier steps and their causal prefix, while a label mask applies loss only to assistant-generated tokens; environment outputs remain conditioning context. Because actions are sparse and state-update spans are longer, within each group we average token losses first per target type present in the trajectory and then across present types. This yields $\bar { \ell } _ { \mathrm { a c t } }$ for silence, response, and recall targets and $\bar { \ell } _ { \mathrm { s t a t e } }$ for thinking and compact-memory targets. The bucketweighted SFT objective is

$$
{ \mathcal { L } } _ { \mathrm { S F T } } ( \theta ) = \lambda _ { \mathrm { a c t } } { \bar { \ell } } _ { \mathrm { a c t } } + \lambda _ { \mathrm { s t a t e } } { \bar { \ell } } _ { \mathrm { s t a t e } } .\tag{7}
$$

Here $\lambda _ { \mathrm { { a c t } } }$ and $\lambda _ { \mathrm { s t a t e } }$ balance sparse interaction-action targets against longer thinking and memory-update targets. This stage cold-starts the interaction protocol in the inference-time causal state by supervising teacher action and state-update targets, while stream inputs, environment observations, and recall returns remain fixed conditioning context.

## Stream-GDPO Reinforcement Learning

Scalarizing multi-reward RL before normalization can let high-variance outcomes dominate weaker process signals. GDPO instead normalizes each reward component within a sampled group before combining advantages (Liu et al. 2026); Stream-GDPO extends this principle to complete online rollouts in the inference-time chunk environment. Each rollout contains one stream with one or more timed questions; policy actions use the current online state, while chunk arrivals and recall returns are environment feedback. Thus, early answers, unnecessary silence, or missed recalls afect the states available to later decisions.

Stream-GDPO combines question-level outcome, format, and recall-use signals with one rollout-level memory signal. For question q in rollout i, the answer outcome is computed by a form-aware verifier:

$$
R _ { \mathrm { o u t } } ^ { i , q } = \operatorname* { m a x } _ { y \in \mathcal { V } _ { q } } \mathbb { I } ( \operatorname* { m a t c h } ( \hat { y } _ { i , q } , y ) ) ,\tag{8}
$$

<table><tr><td>Model</td><td colspan="6">Real-Time</td><td colspan="4">Backward</td><td colspan="4">Forward</td><td>Overall</td></tr><tr><td></td><td>OCR</td><td>ACR</td><td>ATR</td><td>STU FPD</td><td>OJR</td><td>Avg.</td><td>EPM</td><td>ASI HLD</td><td></td><td>Avg.</td><td>REC</td><td>SSR CRR</td><td></td><td>Avg.</td><td>Avg.</td></tr><tr><td></td><td colspan="9">Proprietary models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>69.8</td><td>64.2</td><td>71.6</td><td>51.1</td><td>70.3</td><td>59.8</td><td>64.5</td><td>57.9 75.7</td><td>48.7</td><td>60.8</td><td>27.6</td><td>73.2</td><td>59.4</td><td>53.4</td><td>59.5</td></tr><tr><td>Gemini-1.5-Pro</td><td>85.9</td><td>67.0</td><td>79.3</td><td>58.4</td><td>63.4 62.0</td><td>69.3</td><td>58.6</td><td>76.4</td><td>52.6</td><td>62.5</td><td>35.5</td><td>74.2</td><td>61.7</td><td>57.2</td><td>63.0</td></tr><tr><td></td><td colspan="9">Open-source offline models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LongVU-7B</td><td>53.7</td><td>53.2</td><td>62.9</td><td>47.8</td><td>68.3 59.8</td><td>57.6</td><td>40.7</td><td>59.5</td><td>4.8</td><td>35.0</td><td>12.2</td><td>69.5</td><td>60.8</td><td>47.5</td><td>46.7</td></tr><tr><td>LLaVA-OV-7B</td><td>66.4</td><td>57.8</td><td>73.3</td><td>53.4</td><td>71.3 62.0</td><td>64.0</td><td>54.2</td><td>55.4</td><td>21.5</td><td>43.7</td><td>25.6</td><td>67.1</td><td>58.8</td><td>50.5</td><td>52.7</td></tr><tr><td>LLaVA-Video-7B</td><td>69.1</td><td>58.7</td><td>68.8</td><td>49.4 74.3</td><td>59.8</td><td>63.5</td><td>56.2</td><td>57.4</td><td>7.5</td><td>40.4</td><td>34.1</td><td>70.0</td><td>60.4</td><td>54.8</td><td>52.9</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>69.8</td><td>59.6</td><td>66.4</td><td>50.6 72.3</td><td>61.4</td><td>63.3</td><td>51.2</td><td>64.2</td><td>9.7</td><td>41.7</td><td>34.1</td><td>67.6</td><td>60.8</td><td>54.2</td><td>53.1</td></tr><tr><td>Qwen2.5-VL-7B</td><td>67.8</td><td>55.1</td><td>67.2</td><td>42.1 66.3</td><td>60.9</td><td>58.9</td><td>51.5</td><td>58.8</td><td>32.2</td><td>47.5</td><td>34.1</td><td>67.6</td><td>60.8</td><td>63.6</td><td>57.3</td></tr><tr><td>Qwen3-VL-8B</td><td>75.2</td><td>58.7</td><td>72.4</td><td>57.3</td><td>70.3 59.2</td><td>64.8</td><td>56.6</td><td>69.6</td><td>38.7</td><td>54.4</td><td>38.8</td><td>67.6</td><td>52.5</td><td>63.5</td><td>61.4</td></tr><tr><td>Open-source streaming models</td><td colspan="14"></td></tr><tr><td>StreamForest-7B</td><td>68.5</td><td>53.2</td><td>71.6</td><td>47.8</td><td>65.4</td><td>60.9</td><td>61.2</td><td>58.9 64.9</td><td>32.3</td><td>52.0</td><td>32.8</td><td>70.6</td><td>57.1</td><td>53.5</td><td>55.6</td></tr><tr><td>Streamo-7B</td><td>77.2</td><td>66.1</td><td>76.7</td><td>45.5</td><td>66.3</td><td>72.8</td><td>67.4</td><td>55.6 58.1</td><td>33.9</td><td>49.2</td><td>30.8</td><td>57.6</td><td>82.5</td><td>57.0</td><td>57.9</td></tr><tr><td>VST-7B</td><td>80.5</td><td>55.1</td><td>72.4</td><td>55.1</td><td>76.2</td><td>64.1</td><td>67.2</td><td>56.9 64.9</td><td>48.4</td><td>56.7</td><td>33.0</td><td>66.9</td><td>62.1</td><td>54.0</td><td>59.3</td></tr><tr><td>ViSpeak-7B</td><td>75.2</td><td>58.7</td><td>71.6</td><td>51.1</td><td>74.3</td><td>66.9 66.3</td><td>59.9</td><td>48.7</td><td>64.0</td><td>57.5</td><td>33.8</td><td>68.5</td><td>60.4</td><td>54.3</td><td>61.1</td></tr><tr><td>StreamBridge-7B</td><td>84.6</td><td>71.6</td><td>74.1</td><td>49.4</td><td>75.3</td><td>72.8 71.3</td><td>67.7</td><td>57.4</td><td>79.0</td><td>68.1</td><td>19.2</td><td>64.3</td><td>61.7</td><td>48.4</td><td>62.6</td></tr><tr><td>WTI-8B (Ours)</td><td>90.6</td><td>76.2</td><td>78.5</td><td>66.9</td><td>76.2</td><td>75.5</td><td>76.9</td><td>67.0 74.3</td><td>82.8</td><td>73.4</td><td>42.2</td><td>73.1</td><td>77.1</td><td>70.0</td><td>73.6</td></tr></table>

Table 1: Results on OVO-Bench. Scores are grouped by Real-Time, Backward, and Forward tasks; Avg. gives the corresponding group average, and Overall gives the benchmark average.

where $\mathcal { { V } } _ { q }$ is the valid-answer set; the verifier checks causal response timing, while $R _ { \mathrm { f m t } } ^ { i , q }$ validates whether the action follows the required streaming grammar.

For recall supervision, let $\ell _ { q } , u _ { i , q } \in \{ 0 , 1 \}$ indicate respectively whether the teacher recalls within question $q ^ { * } { \bf s }$ decision interval and whether rollout i makes a syntactically and temporally valid recall there. We gate the recall-tool reward by the answer outcome:

$$
R _ { \mathrm { t o o l } } ^ { i , q } = \ell _ { q } u _ { i , q } R _ { \mathrm { o u t } } ^ { i , q } .\tag{9}
$$

This rewards teacher-aligned recall only when followed by a correct, causally valid response.

For compact-memory writing, let $S _ { \mathrm { t i m e } } ( M )$ score time indexing and $S _ { \mathrm { k e e p } } ( M )$ salient-fact retention. For an update $M _ { \mathrm { p r e } } ^ { i }  M _ { \mathrm { p o s t } } ^ { i } ,$ its improvement reward is

$$
\begin{array} { c l c r }  { { \displaystyle R _ { \mathrm { m e m } } ^ { i } = \frac { 1 } { 2 } \Big [ S _ { \mathrm { t i m e } } ( M _ { \mathrm { p o s t } } ^ { i } ) - S _ { \mathrm { t i m e } } ( M _ { \mathrm { p r e } } ^ { i } ) } } \\ { { + S _ { \mathrm { k e e p } } ( M _ { \mathrm { p o s t } } ^ { i } ) - S _ { \mathrm { k e e p } } ( M _ { \mathrm { p r e } } ^ { i } ) \Big ] . } } \end{array}\tag{10}
$$

Averaging over timed questions gives $R _ { \mathrm { o u t } } ^ { i } , R _ { \mathrm { f m t } } ^ { i }$ , and $R _ { \mathrm { t o o l } } ^ { i } ,$ while $R _ { \mathrm { m e m } } ^ { i }$ remains rollout-level because each memory update persists across subsequent decisions and can afect multiple later questions. Let $\mathcal { C } \ = \ \{ \mathrm { o u t , f m t , t o o l , m e m } \}$ with weights $w _ { \mathrm { o u t } } ~ = ~ w _ { \mathrm { f m t } } ~ = ~ 1 , ~ w _ { \mathrm { t o o l } } ~ = ~ \lambda _ { \mathrm { t o o l } }$ , and $w _ { \mathrm { m e m } } = \lambda _ { \mathrm { m e m } } .$ The two $\lambda$ coeficients control the relative strengths of recall-use and memory-quality feedback, while $\epsilon _ { \mathrm { n o r m } } > 0$ below stabilizes normalization when a component has low group variance. For each rollout group ${ \mathcal { G } } .$ Stream-GDPO forms the trajectory advantage by component-wise normalization:

$$
A _ { i } = \sum _ { c \in \mathcal { C } } w _ { c } \frac { R _ { c } ^ { i } - \mu _ { c } ( \mathcal { G } ) } { \sigma _ { c } ( \mathcal { G } ) + \epsilon _ { \mathrm { n o r m } } } .\tag{11}
$$

We assign $A _ { i }$ to all assistant action turns in the rollout, giving trajectory-level credit to silence/response, recall, and compact-memory decisions. The policy is optimized with the clipped objective

$$
\mathcal { I } _ { \mathrm { G D P O } } ( \theta ) = \mathbb { E } _ { i , j } [ \operatorname* { m i n } ( \rho _ { i , j } A _ { i } , \tilde { \rho } _ { i , j } A _ { i } ) - \beta D _ { i , j } ] .\tag{12}
$$

Here $\rho _ { i , j }$ is the action-turn probability ratio between π and $\pi _ { \theta _ { \mathrm { o l d } } } , ~ \tilde { \rho } _ { i , j } ~ = ~ \mathrm { c l i p } ( \rho _ { i , j } , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } ) , ~ D _ { i , j }$ is the reference-policy KL penalty at $s _ { i , j } ,$ , and $\beta$ controls its strength. We minimize $\mathcal { L } _ { \mathrm { G D P O } } = \overset { \sim } { - } \mathcal { J } _ { \mathrm { G D P O } }$ , retaining trajectory-level credit without separate turn- or token-level advantages.

## WTI-82K: Causally Aligned Streaming Interaction Data

Prior online video benchmarks and streaming instruction datasets (Niu et al. 2025; Yang et al. 2025; Wang et al. 2026; Xia et al. 2025) define timestamped queries, multi-turn interaction, proactive responses, and streaming task annotations. Building on this setting, we construct WTI-82K as stateful streaming interaction trajectories rather than isolated video-QA pairs. It organizes $8 \hat { 2 } , 3 3 5$ timed questions into 4,812 trajectories that align observed chunks, query times, answerable moments, supporting evidence, state updates, and interaction actions under the causal context available at each moment. Relative to the 8-second active window, 56.9% of trajectories span at least 120 seconds and 21.7% span at least 240 seconds. Each trajectory contains 17.1 questions on average, and 49.9% of questions fall into historical, future, multistate, or temporal-reasoning categories. This yields repeated decisions over a shared evolving stream rather than isolated current-frame questions. As shown in Figure 3, our pipeline extracts temporal evidence, generates questions from aligned

<table><tr><td>Model OP CR CS ATP EU TR PR S SU ACP CT Avg.</td></tr><tr><td>Proprietary models</td></tr><tr><td>GPT-40 77.1 80.5 83.976.5 70.2 83.8 66.7 62.2 69.1 49.2 73.3</td></tr><tr><td>Gemini-1.5-Pro 79.0 80.5 83.5 79.7 80.0 84.7 77.8 64.2 72.0 48.775.7</td></tr><tr><td>Open-source offline models</td></tr><tr><td>LLaVA-OV-7B 80.4 74.2 76.080.7 72.7 71.7 67.6 65.5 65.7 45.1 71.1</td></tr><tr><td>Qwen2.5-VL-7B 77.9 76.6 78.6 80.9 76.7 77.0 80.6 65.5 65.7 52.9 73.3</td></tr><tr><td>Qwen3-VL-8B 79.9 77.3 81.1 84.3 76.7 77.9 79.6 71.1 69.6 46.3 75.2</td></tr><tr><td>MiniCPM-o-4.5-9B 80.9 88.3 82.085.0 78.9 85.4 78.7 67.1 75.6 56.078.2</td></tr><tr><td>Open-source streaming models</td></tr><tr><td>VideoLLM-online-8B 39.1 40.1 34.5 31.1 46.0 32.4 31.5 34.2 42.5 27.9 36.0</td></tr><tr><td>ViSpeak-7B 79.8 71.1 81.4 78.8 74.5 70.1 63.9 64.2 71.4 28.0 70.4</td></tr><tr><td>StreamBridge-7B 84.7 82.7 88.9 89.8 77.4 85.4 84.3 69.9 71.7 35.8 77.0</td></tr><tr><td>StreamForest-7B 83.1 82.8 82.7 84.3 77.5 78.2 76.9 69.1 75.6 54.4 77.3</td></tr><tr><td>VST-7B 85.4 82.0 86.4 89.1 74.2 87.2 82.4 73.1 73.9 47.3 79.5</td></tr><tr><td>WTI-8B (Ours) 88.1 79.7 92.7 90.1 79.2 92.5 82.4 80.9 83.2 40.4 83.3</td></tr></table>

Table 2: StreamingBench real-time visual understanding results.

Temporal evidence extraction. We first filter videos for clear visual content and temporally locatable events (Gao et al. 2026). For each retained video, we extract chunk-level evidence and link facts that persist or change across chunks. This yields minimal supporting intervals for evidence-chain selection and time-anchored question generation.

evidence, places each query and answerable moment on the streaming timeline, and checks timestamp bounds, answer availability, action grammar, and evidence-interval fields.

Candidate question generation. Given the aligned evidence, we use Qwen3.5-397B to generate questions and reference answers from local facts and cross-chunk evidence chains. The candidates cover immediate perception, historical evidence, temporal progression, and proactive interaction, so each trajectory includes both current-prefix questions and questions requiring long-range state maintenance.

<table><tr><td colspan="3">Model MLVU V-MME LVB</td></tr><tr><td colspan="3">Offline models</td></tr><tr><td>LongVA-7B LLaVA-OV-7B LongVU-7B</td><td>56.3 64.7 65.4 66.7</td><td>52.656.3 58.2 一 60.6 63.5 60.7</td></tr><tr><td colspan="3">Qwen3-VL-8B Streaming models</td></tr><tr><td>StreamForest-7B StreamBridge-7B</td><td>70.0 69.6</td><td>61.4 64.4</td></tr><tr><td>VST-7B WTI-8B (Ours)</td><td>69.9</td><td>64.9 58.0 66.9 62.9</td></tr></table>

Table 3: Long-video results.

Query-time and answer-time placement. Following timestamped online-video protocols (Niu et al. 2025; Yang et al. 2025; Wang et al. 2026), we convert each candidate into a streaming sample by assigning query and answerable timestamps from its supporting evidence interval. Real-Time samples are answerable at the query time from the current visible prefix. Backward Tracing samples are also answerable at the query time, but their supporting evidence lies in previously observed chunks and therefore tests history maintenance and recall. Proactive samples are not answerable when the query appears and require the model to wait until the earliest answerable moment, when decisive future evidence has arrived. Automated checks cover temporal bounds, answer availability, action grammar, and evidence localization before final serialization. The resulting set exposes answernow, recall-before-answer, and wait-for-evidence behaviors under the visibility constraints used for evaluation. The supplementary material reports the data statistics and question distribution in detail.

## Experiments

## Experimental Setup

Implementation Details. WTI is initialized from Qwen3- VL-8B-Instruct (Bai et al. 2025a). Masked SFT trains on chunk-level trajectories with the vision encoder frozen and updates the language model and multimodal projector at a learning rate of $1 \times 1 0 ^ { - 5 }$ . Stream-GDPO starts from the SFT checkpoint and uses veRL (Sheng et al. 2025) for one epoch on 8 H20 GPUs, with prompt batch size 64, 4 rollouts per prompt, actor learning rate $1 \times 1 0 ^ { - 5 }$ , asymmetric clipping $( \epsilon _ { \mathrm { l o w } } = 0 . 2 , \epsilon _ { \mathrm { h i g h } } = 0 . 2 8 )$ , and action-branch loss weight 0.15. Videos are streamed as 1-second chunks at 2 FPS with an 8-chunk (16-frame) active window; recall is permitted at most once per current chunk and returns at most four observed chunks. Compact-and-reprefill is triggered at the active-context boundary during inference. SFT, RL, and evaluation share the same chunking, memory budget, recall tool, and action grammar; objective comparisons therefore do not change the streaming environment.

Benchmarks. We evaluate OVO-Bench Real-Time, Backward, Forward, and weighted overall accuracy, followed by StreamingBench real-time visual understanding (Niu et al. 2025; Lin et al. 2024). StreamingBench includes Object Perception (OP), Causal Reasoning (CR), Commonsense (CS), Action and Temporal Perception (ATP), Event Understanding (EU), Temporal Reasoning (TR), Proactive Reasoning (PR), Spatial Understanding (SU), Attribute and Character Perception (ACP), and Counting (CT). We also report Video-MME (Fu et al. 2025a), MLVU (Zhou et al. 2025), and LongVideoBench (Wu et al. 2024) oficial aggregate metrics for general long-video ability.

Baselines. We compare with proprietary models, offline/static Video-LLMs, and online or streaming Video-LLMs, including GPT-4o (Hurst et al. 2024), Gemini-1.5- Pro (Team et al. 2024), LongVA (Zhang et al. 2025b), LongVU (Shen et al. 2024), LLaVA-series models (Li et al. 2025; Zhang et al. 2025c, 2024), Qwen2.5-VL (Bai et al.

![](images/c5d1bc1630d5a061d0ab86da0a293b49d6f889fe201a0a0b2dcc53fd2d5b0f60.jpg)

![](images/b7fdd47a90f0a17533a6361da9f881ca3759c0732ebd3af5f806ffd3802cf95d.jpg)

![](images/7c39909bb6875f6f9667b131dcf0d3dc0f9c150e53ae65cfd1fc37261f6f43f3.jpg)

![](images/2b01989653f0353581d4268e56b7550631d437002b4f35866c7c5d6c6fa3fad7.jpg)

Figure 4: Training-time GSPO/Stream-GDPO diagnostics. Panels report outcome accuracy, BT recall triggering, post-recall outcome accuracy, and memory quality; faint and bold curves show raw and rolling values.
<table><tr><td>Config.</td><td colspan="7">Stream. OVO-RT OVO-BT OVO-FT OVO Avg. V-MME MLVU LVB</td></tr><tr><td>Base model</td><td>75.2</td><td>64.8</td><td>54.4</td><td>63.5</td><td>61.4</td><td>63.5</td><td>66.7 60.7</td></tr><tr><td>+ SFT</td><td>75.3</td><td>74.2</td><td>60.3</td><td>60.9</td><td>65.7</td><td>64.7</td><td>67.6 59.8</td></tr><tr><td>+ GSPO</td><td>80.0</td><td>76.2</td><td>65.3</td><td>60.7</td><td>67.8</td><td>65.6</td><td>68.7 61.7</td></tr><tr><td>+ GDPO</td><td>83.3</td><td>76.9</td><td>73.4</td><td>70.0</td><td>73.6</td><td>66.9</td><td>69.9 62.9</td></tr></table>

<table><tr><td>Variant</td><td>OVO-BT V-MME MLVU LVB</td></tr><tr><td>Full</td><td>73.4 66.9 69.9 62.9</td></tr><tr><td>w/o memory 62.8</td><td>63.5 44.7 61.6</td></tr><tr><td>w/o recall</td><td>55.5 48.3 54.2 57.9</td></tr><tr><td>memory only 48.2</td><td>47.7 46.557.1</td></tr></table>

Table 4: Training-objective ablation across streaming and ofline long-video benchmarks; GDPO denotes Stream-GDPO.  
Table 5: Component ablation across OVO-BT and ofline long-video benchmarks.

2025b), Qwen3-VL (Bai et al. 2025a), MiniCPM-o (Cui et al. 2026), and recent streaming systems (Chen et al. 2024; Fu et al. 2025b; Wang et al. 2026; Zeng et al. 2025; Xia et al. 2025; Guan et al. 2026). All online evaluations are causal, and ofline/static baselines are restricted to the observed prefix in online benchmarks under the benchmark protocol.

## Main Results

Tables 1–3 report OVO-Bench Real-Time, Backward, and Forward results, StreamingBench categories, and aggregate ofline long-video performance. On OVO-Bench, WTI achieves the best Overall, Real-Time, Backward, and Forward scores among open-source streaming baselines: 73.6%, 76.9%, 73.4%, and 70.0%, respectively; its overall score exceeds StreamBridge-7B (62.6%) and ViSpeak-7B (61.1%). The gains span all three streaming modes: time-indexed memory and recall support historical reasoning, while activewindow perception and multi-turn textual state retain Real-Time and Forward performance.

On StreamingBench, WTI reaches 83.3%, ahead of VST-7B (79.5%), StreamForest-7B (77.3%), and StreamBridge-7B (77.0%), and leads on OP, CS, ATP, EU, TR, SU, and ACP. These category gains show that WTI’s improvement extends from historical reasoning to current-prefix perception. On ofline long-video benchmarks, WTI retains strong general ability, scoring 66.9% on Video-MME, 69.9% on MLVU, and 62.9% on LongVideoBench. Among the reported streaming baselines, these are the best Video-MME and LongVideoBench scores; on MLVU, WTI remains within 0.1 point of StreamForest-7B (69.9% vs. 70.0%). Together, these results establish state-of-the-art open-source streaming performance with strong ofline long-video ability; the ablations below evaluate compact memory and sourcevideo recall separately.

## Ablations

Table 4 separates protocol initialization from trajectory-level policy optimization. Masked SFT keeps StreamingBench at

75.3% and raises OVO-RT from 64.8% to 74.2%, but reaches only 60.3% on OVO-BT and 60.9% on OVO-FT; its Forward score remains below the base model’s 63.5%. Thus, imitation initializes recent-prefix behavior but remains substantially below the complete rollout-optimized model on Backward and Forward reasoning, motivating trajectory optimization in which early actions shape later states throughout the rollout.

Figure 4 compares the training dynamics of Stream-GDPO and GSPO (Zheng et al. 2025). Stream-GDPO sustains stronger BT recall triggering, post-recall accuracy, and memory quality, and raises OVO-BT from 65.3% to 73.4% and OVO-FT from 60.7% to 70.0%, lifting the overall score to 73.6%. It separately normalizes outcome, format, recall, and memory signals before aggregation. These diagnostic gains extend beyond final-answer accuracy.

Table 5 shows that compact memory and recall are complementary rather than interchangeable. Removing memory lowers OVO-BT from 73.4% to 62.8%, while removing recall lowers it to 55.5%; the memory-only variant reaches only 48.2%. Ofline, removing recall reduces Video-MME from 66.9% to 48.3%, while removing memory reduces MLVU from 69.9% to 44.7%. The memory-only result supports this division of labor: compact text indexes history but cannot substitute for visual evidence, whereas recall restores details omitted by compression.

## Conclusion

We introduced Watch-Think-Interact (WTI), a causal framework for long-horizon multi-turn streaming video reasoning that combines bounded active-window perception, timeindexed memory, source-video recall, and response timing. WTI-82K supplies causally aligned trajectories, while Stream-GDPO optimizes complete online rollouts in the same chunk-level environment used at inference. WTI sets a new open-source state of the art on StreamingBench and OVO-Bench while retaining strong ofline long-video performance; ablations verify complementary contributions from both memory and recall.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025b. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Chen, J.; Lv, Z.; Wu, S.; Lin, K. Q.; Song, C.; Gao, D.; Liu, J.- W.; Gao, Z.; Mao, D.; and Shou, M. Z. 2024. VideoLLM-online: Online Video Large Language Model for Streaming Video. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18407–18418.

Chen, J.; Zeng, Z.; Lin, Y.; Li, W.; Ma, Z.; and Shou, M. Z. 2025. LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 29083–29095.

Cui, J.; Xu, B.; Wang, C.; Yu, T.; Sun, W.; Xu, Y.; Wang, T.; He, Z.; Ma, W.; Cai, T.; et al. 2026. MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction. arXiv preprint arXiv:2604.27393.

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; Chen, P.; Li, Y.; Lin, S.; Zhao, S.; Li, K.; Xu, T.; Zheng, X.; Chen, E.; Shan, C.; He, R.; and Sun, X. 2025a. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 24108–24118.

Fu, S.; Yang, Q.; Li, Y.-M.; Peng, Y.-X.; Lin, K.-Y.; Wei, X.; Hu, J.-F.; Xie, X.; and Zheng, W.-S. 2025b. ViSpeak: Visual Instruction Feedback in Streaming Videos. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 21778–21788.

Guan, Y.; Yin, L.; Liang, D.; Ju, J.; Luo, Z.; Luan, J.; Liu, Y.; and Bai, X. 2026. Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously. In European Conference on Computer Vision (ECCV).

Hong, J.; Zhao, C.; Zhu, C.; Lu, W.; Xu, G.; and XingYu. 2026. DeepEyesV2: Toward Agentic Multimodal Model. In The Fourteenth International Conference on Learning Representations.

Huang, Z.; Li, X.; Li, J.; Wang, J.; Zeng, X.; Liang, C.; Wu, T.; Chen, X.; Li, L.; and Wang, L. 2025. Online Video Understanding: OVBench and VideoChat-Online. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3328–3338.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; and Li, C. 2025. LLaVA-OneVision: Easy Visual Task Transfer. Transactions on Machine Learning Research.

Li, K.; Wang, Y.; He, Y.; Li, Y.; Wang, Y.; Liu, Y.; Wang, Z.; Xu, J.; Chen, G.; Luo, P.; Wang, L.; and Qiao, Y. 2024. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 22195–22206.

Li, X.; Wang, Y.; Yu, J.; Zeng, X.; Zhu, Y.; Huang, H.; Gao, J.; Li, K.; He, Y.; Wang, C.; Qiao, Y.; Wang, Y.; and Wang, L. 2026. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. In The Fourteenth International Confer ence on Learning Representations.

Lin, J.; Fang, Z.; Chen, C.; Wan, Z.; Luo, F.; Li, P.; Liu, Y.; and Sun, M. 2024. StreamingBench: Assessing the Gap for

MLLMs to Achieve Streaming Video Understanding. arXiv preprint arXiv:2411.03628.

Liu, S.-Y.; Dong, X.; Lu, X.; Diao, S.; Belcak, P.; Liu, M.; Chen, M.-H.; Yin, H.; Wang, Y.-C. F.; Cheng, K.-T.; Choi, Y.; Kautz, J.; and Molchanov, P. 2026. GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization. arXiv:2601.05242.

Niu, J.; Li, Y.; Miao, Z.; Ge, C.; Zhou, Y.; He, Q.; Dong, X.; Duan, H.; Ding, S.; Qian, R.; Zhang, P.; Zang, Y.; Cao, Y.; He, C.; and Wang, J. 2025. OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18902–18913.

Qian, R.; Ding, S.; Dong, X.; Zhang, P.; Zang, Y.; Cao, Y.; Lin, D.; and Wang, J. 2025. Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 24045–24055.

Shen, X.; Xiong, Y.; Zhao, C.; Wu, L.; Chen, J.; Zhu, C.; Liu, Z.; Xiao, F.; Varadarajan, B.; Bordes, F.; et al. 2024. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2025. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, 1279–1297.

Shi, Y.; Chen, Y.; Wang, S.; Li, S.; Cai, H.; Qi GU; Wang, X.; and Zhang, A. 2026. Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents. In The Fourteenth International Conference on Learning Representations.

Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Wang, H.; Feng, B.; Lai, Z.; Xu, M.; Li, S.; Ge, W.; Dehghan, A.; Cao, M.; and Huang, P. 2026. StreamBridge: Turning Your Ofline Video Large Language Model into a Proactive Streaming Assistant. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Wu, H.; Li, D.; Chen, B.; and Li, J. 2024. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding. In Globerson, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C., eds., Advances in Neural Information Processing Systems, volume 37, 28828–28857. Curran Associates, Inc.

Xia, J.; Chen, P.; Zhang, M.; Sun, X.; and Zhou, K. 2025. Streaming Video Instruction Tuning. arXiv preprint arXiv:2512.21334.

Xu, R.; Xiao, G.; Chen, Y.; He, L.; Peng, K.; Lu, Y.; and Han, S. 2026. StreamingVLM: Real-Time Understanding for Infinite Video Streams. In The Fourteenth International Conference on Learning Representations.

Yang, Z.; Hu, Y.; Du, Z.; Xue, D.; Qian, S.; Wu, J.; Yang, F.; Dong, W.; and Xu, C. 2025. SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding. In Yue, Y.; Garg, A.; Peng, N.; Sha, F.; and Yu, R., eds., International Conference on Learning Representations, volume 2025, 42522– 42556.

Yang, Z.; Zhang, K.; Hu, Y.; Wang, B.; Qian, S.; Wen, B.; Yang, F.; Gao, T.; Dong, W.; and Xu, C. 2026. LiveStar: Live Streaming Assistant for Real-World Online Video Understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yu, H.; Chen, T.; Feng, J.; Chen, J.; Dai, W.; Yu, Q.; Zhang, Y.-Q.; Ma, W.-Y.; Liu, J.; Wang, M.; and Zhou, H. 2026. MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. In The Fourteenth International Conference on Learning Representations.

Zeng, X.; Qiu, K.; Zhang, Q.; Li, X.; Wang, J.; Li, J.; Yan, Z.; Tian, K.; Tian, M.; Zhao, X.; Wang, Y.; and Wang, L. 2025. StreamForest: Eficient Online Video Understanding with Persistent Event Memory. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 75804–75835. Curran Associates, Inc.

Zhang, H.; Wang, Y.; Tang, Y.; Liu, Y.; Feng, J.; and Jin, X. 2025a. Flash-VStream: Eficient Real-Time Understanding for Long Video Streams. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 21059–21069.

Zhang, P.; Zhang, K.; Li, B.; Zeng, G.; Yang, J.; Zhang, Y.; Wang, Z.; Tan, H.; Li, C.; and Liu, Z. 2025b. Long Context Transfer from Language to Vision. Transactions on Machine Learning Research.

Zhang, Y.; Li, B.; Liu, h.; Lee, Y. j.; Gui, L.; Fu, D.; Feng, J.; Liu, Z.; and Li, C. 2024. LLaVA-NeXT: A Strong Zero-shot Video Understanding Model.

Zhang, Y.; Wu, J.; Li, W.; Li, B.; Ma, Z.; Liu, Z.; and Li, C. 2025c. LLaVA-Video: Video Instruction Tuning With Synthetic Data. Transactions on Machine Learning Research.

Zheng, C.; Liu, S.; Li, M.; Chen, X.-H.; Yu, B.; Gao, C.; Dang, K.; Liu, Y.; Men, R.; Yang, A.; et al. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071.

Zheng, Z.; Yang, M.; Hong, J.; Zhao, C.; Xu, G.; Yang, L.; Shen, C.; and XingYu. 2026. DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning. In The Fourteenth International Conference on Learning Representations.

Zhou, J.; Shu, Y.; Zhao, B.; Wu, B.; Liang, Z.; Xiao, S.; Qin, M.; Yang, X.; Xiong, Y.; Zhang, B.; Huang, T.; and Liu, Z. 2025. MLVU: Benchmarking Multi-task Long Video Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 13691–13701.

Gao, Z.; Bao, Y.; Peng, J.; Li, X.; Huang, T.; Liu, B.; Li, K.; Gan, Z.; Hu, T.; Xie, C.; et al. 2026. VideoSearcher: Empowering Video Deep Research with Multi-Tool Agentic Reasoning via Reinforce ment Learning. arXiv preprint arXiv:2607.02927.

Gao, Z.; Wang, X.; Tan, X.; and Xie, Y. 2026. TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models. arXiv preprint arXiv:2602.18884.

Wang, S.; Liu, B.; Gao, Z.; Ma, L.; Wang, X.; Xie, Y.; and Tan, X. 2026. Explore with Long-Term Memory: A Benchmark and Multimodal LLM-Based Reinforcement Learning Framework for Embodied Exploration. arXiv preprint arXiv:2601.10744.