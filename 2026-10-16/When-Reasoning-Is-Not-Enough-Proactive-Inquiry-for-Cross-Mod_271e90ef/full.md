# When Reasoning Is Not Enough: Proactive Inquiry for Cross-Modal Safety in Multimodal Large Language Models

Wei Cai, Jian Zhao, Yuchen Yuan, and Xuelong Li<sup>∗</sup>, Fellow, IEEE

Abstract—Multimodal Large Language Models (MLLMs) face a subtle safety vulnerability: cross-modal inputs that are individually benign can elicit unsafe behavior once jointly interpreted. We identify two complementary failure modes: implicit reasoning risk, where long reasoning chains branch into unsafe trajectories even though the request is fully specified, and intent-level risk, where cross-modal ambiguity silently commits the model to a hazardous premise. Path-level optimization addresses only the former, while proactive-inquiry methods designed for unimodal text are misaligned with the latter: they rely on entropy-only triggers and efficiency-biased rewards, and no multimodal multi-turn training data exist for them. We propose SAFE-PRO, a unified framework with three components: a Risk-Conditioned Uncertainty (RCU) signal that combines predictive entropy with a learned cross-modal risk score—trained on contrastive reasoning branches at no extra annotation cost—to decide both when the model asks and where it optimizes; a two-layer defense that interleaves risk-guided path optimization with safety-grounded clarification under a composite reward in which safety dominates efficiency and an over-refusal penalty prevents pathological caution; and SSUI-Dialog, a cross-modal multi-turn dialogue dataset with safety-critical intent ambiguity and dual-intent (safe/unsafe) supervision. We show that, when the risk score is positively associated with gradient magnitude, risk-guided step sampling strictly reduces the variance of the inner-loop gradient estimator over a range of concentration levels, and we bound the probability of an unsafe completion under RCU gating. Across seven benchmarks and two base MLLMs, SAFE-PRO raises the average safety score of SRPO from 76.18 to 82.77 and lowers its average harm score from 12.67 to 7.64 on Qwen2.5-VL, roughly halves over-refusal on MOSSBench (38.4 → 19.7), and outperforms all compared models, including GPT-4o and Claude-3.5-Sonnet, on every single-turn safety metric and on over-refusal.

Index Terms—Multimodal large language models, cross-modal safety alignment, proactive clarification, reinforcement learning.

## 1 INTRODUCTION

M <sup>ULTIMODAL</sup> <sup>Large</sup> <sup>Language</sup> <sup>Models</sup> <sup>(MLLMs)</sup> <sup>have</sup>rapidly become foundational components in realment broadens, concerns about their safety alignment have intensified. While substantial progress has been made in aligning unimodal Large Language Models (LLMs) [6], [7], safety alignment under multimodal settings remains notably fragile [8], [9], [10]. Recent studies [11], [12], [13] reveal that MLLMs are particularly susceptible to a subtle but dangerous class of attacks: cross-modal inputs in which each modality is individually benign, yet whose semantic combination elicits unsafe behavior.

Fig. 1(a) illustrates a canonical case: an image of two household products—“84” disinfectant (a hypochlorite bleach) and a toilet cleaner containing hydrochloric acid— paired with the request to “clean the bathroom according to the instructions,” where the instructions prescribe using both products. Neither input is harmful in isolation, yet their semantic composition implies a mixing procedure that releases toxic chlorine gas. Mainstream MLLMs frequently proceed to compute ingredient quantities as if the request were routine [11], [14]. Here the request is fully specified, and the danger lies in recognizing the hazard during reasoning. We term this failure implicit reasoning risk: the model’s long reasoning chain branches into a disadvantageous trajectory whose errors compound silently until they produce a hazardous answer. Path-level optimization methods that encourage favorable branches and penalize unfavorable ones at every intermediate state can substantially mitigate this problem [15], [16].

In extending and stress-testing this path-level paradigm, however, we identified a second failure mode that path optimization alone cannot resolve. Consider Fig. 1(b): the same image paired with the more naturalistic request “follow the instructions on the labels,” which does not say whether one or both products are to be used. To continue reasoning, the model must assume which product(s) the user intends to use; if it assumes the unsafe interpretation, every subsequent step, however carefully optimized, marches toward a hazardous answer. We term this intent-level risk: the crossmodal input is ambiguous in a safety-critical direction, and any internal assumption silently commits the model to one branch of a moral fork. Recent work on reasoning LLMs [17], [18], [19] documents related missing-premise failures in the unimodal setting; we argue that the problem is more acute under multimodality, where joint image–text semantics frequently leave room for multiple plausible intents.

The natural remedy—training models to issue clarification questions when uncertain [20], [21], [22], [23]—does not transfer cleanly to cross-modal safety. First, existing methods trigger questions using predictive entropy [23], [24], which signals only that the model is unsure, not that the uncertainty lies on a safety-critical axis; this yields both over-interruption on benign inputs and under-interruption on hazardous ones. Second, prevailing interactive frameworks balance task success against interaction efficiency, penalizing extra turns [22], [23]; in safety-critical settings this trade-off inverts, since a missed clarification at a risky branch is far more costly than a redundant one. Third, recent multi-turn multimodal safety benchmarks [25], [26], [27] target risks that escalate or shift across turns, but no public dataset supervises safety-grounded clarification of an ambiguous cross-modal premise. These gaps—signal, reward, and data—explain why path-level optimization and proactive inquiry, taken alone, each leave half of the problem unaddressed.

![](images/3b935031dd2bb3f4f9f378cbfa8933233928e001eea9ae11a11e97eee8653e3a.jpg)  
Fig. 1. Two complementary failure modes in cross-modal safety. (a) Implicit reasoning risk: even when the request is fully specified, the model’s long reasoning chain may branch into unsafe trajectories. (b) Intent-level ambiguity: when the cross-modal input under-specifies the user’s goal, the model is forced to assume a premise, and a wrong assumption can deterministically commit reasoning to an unsafe path. SAFE-PRO addresses both via a two-layer defense unified by a risk-conditioned uncertainty signal.

We propose the Safety-Aware Framework for Explicit and Proactive reasoning (SAFE-PRO), a unified two-layer defense that integrates reasoning path optimization with proactive inquiry through a single risk-aware mechanism. At the core of SAFE-PRO is a risk-conditioned uncertainty (RCU) signal $u ( s ) \ = \ \alpha \mathrm { P E } ( s ) + \beta \mathcal { R } ( s )$ that combines normalized predictive entropy with a learned cross-modal risk score, the latter trained on the contrastive branch pairs already produced by SRPO exploration [15] and therefore requiring no additional annotation. Conditioned on this signal, an outer loop issues a clarification question when the RCU score is high, under a redesigned reward in which safety strictly dominates efficiency and an explicit overrefusal penalty prevents pathological caution; an inner loop performs risk-guided contrastive path optimization during training, concentrating its supervision on high-risk steps. To train and evaluate the framework, we construct SSUI-Dialog, a cross-modal multi-turn dialogue dataset with safety-critical intent ambiguity, by extending the multiagent pipeline of our prior SSUI dataset [15] to inject safetycritical premise ambiguity and to simulate both safe-intent and unsafe-intent user replies.

Our main contributions are summarized as follows:

We identify intent-level risk as a class of cross-modal safety failures complementary to implicit reasoning risk, and show that path-level optimization alone is insufficient to resolve it.

We propose SAFE-PRO, a unified two-layer defense driven by a risk-conditioned uncertainty signal. We prove that, when the risk score is positively associated with gradient magnitude, risk-guided step sampling strictly reduces the variance of the innerloop gradient estimator for a range of concentration levels, and we bound the probability of an unsafe completion under RCU gating, making explicit the role of the risk-score weight.

We construct SSUI-Dialog, a cross-modal multi-turn dialogue dataset with safety-critical premise ambiguity and dual-intent supervision, and introduce the Cross-modal Safety with Helpful Clarification (CSHC) metric to jointly evaluate safety, helpfulness, and clarification economy.

Evaluations on seven benchmarks (five single-turn safety benchmarks, SSUI-Dialog, and MOSSBench) with two base MLLMs show that SAFE-PRO raises the average safety score by 6.59 points and lowers the average harm score by 5.03 points relative to SRPO, roughly halves over-refusal on MOSSBench, and outperforms leading commercial MLLMs, including GPT-4o and Claude-3.5-Sonnet, on every evaluated metric.

![](images/ca36504acd6a7c780e499dda0352dedb8ab480f1c0522b0e93717d31b502f972.jpg)  
Fig. 2. The construction pipeline of SSUI-Dialog. Stage 1 injects safety-critical premise ambiguity into SSUI queries; Stage 2 generates the corresponding clarification questions; Stage 3 simulates dual user replies (safe-intent and unsafe-intent); Stage 4 composes full dialogue trajectories; Stage 5 performs human verification; and Stage 6 unifies the verified trajectories into the released format. The stages reuse the multi-agent infrastructure of SSUI [15] with new ambiguity-injection and dual-simulation modules.

## 2 RELATED WORK

## 2.1 Cross-Modal Safety in MLLMs

Safety alignment of Multimodal Large Language Models [1], [2], [3], [4], [5], [28] has emerged as a distinct and harder problem than its unimodal predecessor. Early benchmarks such as MM-SafetyBench [8], FigStep [9], and the visual jailbreak studies of Qi et al. [10] demonstrated that visual inputs can elicit harmful responses that text-only safety alignment fails to block. Subsequent work shifted attention to genuinely cross-modal attacks, in which neither modality is individually unsafe: SIUO [12], VLSBench [29], and MSSBench [11] systematically construct cross-modal benchmarks where each input passes unimodal safety filters yet jointly induces unsafe outputs. MLLMGuard [30] and SafeBench [31] broaden the taxonomy to dozens of risk categories. Most recently, safety evaluation has moved to multi-turn multimodal dialogue: SafeMT [25] and MTMCS-Bench [27] benchmark risks that emerge or escalate across turns, LLaVAShield [26] pairs a multi-turn multimodal benchmark with a dialogue-level moderation model, and MINT-Safe [32] provides multi-image multi-turn dialogues for safety alignment. These resources focus on detecting or refusing risks that arise across turns; to our knowledge, none of them trains or evaluates safety-grounded clarification of an ambiguous premise. Methodologically, defenses span RLHF [7], DPO [33], SimPO [34], ORPO [35], and

MLLM-specific protection mechanisms [11], [13], [16]. A common limitation of these methods is that they treat safety as a property of the final output rather than of the intermediate reasoning process, leaving the model vulnerable to error propagation in long reasoning chains [36]. They also presuppose a fully specified user intent, an assumption that frequently breaks in practice.

## 2.2 Interactive and Proactive Language Models

A complementary thread enhances language models with multi-turn interactivity. Early prompting-based methods [37], [38] steered LLMs to issue clarification questions via templates, but generalized poorly. Learning-based approaches subsequently emerged: CLAM [21] trains on synthetically perturbed ambiguous queries; CIRCLE [39] applies reinforcement learning to teach when to ask; STaR-GATE [20] trains LLMs to elicit user preferences through clarification; CLAMBER [40] provides a benchmark for ambiguity identification; CollabLLM [22] reframes LLMs as active collaborators rewarded for long-term conversational contribution; and PIR [23] trains reasoning LLMs to ask during reasoning, combining uncertainty-aware supervised fine-tuning with GRPO against a simulated user and a reward that balances task success, helpfulness, and efficiency. These works establish that proactive inquiry is both feasible and beneficial in unimodal text settings. However, they were not designed for safety-critical multimodal scenarios:

![](images/8f7cf4e7a23c056f3f45e12d78faf6728e2784f255aa766b62b8f42d5c8d535f.jpg)  
Fig. 3. Three-tier safety taxonomy (3 primary, 19 secondary, and 68 tertiary categories) that SSUI-Dialog inherits from SSUI; reproduced from [15].

their triggering signals are modality-agnostic and safetyunaware, their reward structures penalize the very behavior we wish to encourage at risky branches, and they neither train nor evaluate on multimodal inputs. To our knowledge, no prior work trains MLLMs to issue safety-grounded clarifications for cross-modal inputs, which is the primary methodological focus of SAFE-PRO.

## 3 THE SSUI-DIALOG DATASET

Training and evaluating models for intent-level cross-modal safety requires data that existing benchmarks do not supply: cross-modal inputs that are simultaneously ambiguous in a safety-critical direction and paired with naturalistic multi-turn dialogues resolving that ambiguity. We construct SSUI-Dialog to fill this gap.

## 3.1 Construction Pipeline

The construction pipeline (Fig. 2) consists of six stages, three of which (Stages 1–3) are new relative to the original SSUI pipeline.

Stage 1: Safety-Critical Premise Identification. For each SSUI sample $( { \overset { \cdot } { I } } , T , \tau ^ { * } , A )$ , we identify the safety-critical premise—the explicit fact in $T$ whose removal renders the query genuinely ambiguous about whether the safe or unsafe interpretation is intended. For the bathroom-cleaning example of Fig. 1, the safety-critical premise is the explicit naming of both products; removing it yields “follow the instructions on the labels,” which admits multiple interpretations. A Premise Identifier agent (GPT-4o), prompted with $T$ and $\tau ^ { * } ,$ extracts the minimal text span whose removal triggers safety-critical ambiguity. The resulting ambiguous query $T _ { \mathrm { a m b } }$ replaces $T$ in the trajectory.

Stage 2: Clarification Question Generation. A Clarifier agent (GPT-4o) generates a safety-grounded clarification question $q ^ { * }$ targeting the removed premise under three constraints: it must be answerable in fewer than 20 words, must not leak the unsafe interpretation, and must be phrased neutrally so that both safe and unsafe replies are plausible. We sample three candidates per instance and retain the highest-rated by an LLM judge on a 5-point fluency-andneutrality scale.

Stage 3: Dual-Intent User Reply Simulation. For each clarification question $q ^ { * }$ , we simulate two distinct user replies. The safe-intent reply $r ^ { + }$ describes a usage that does not trigger the hazard (“I’ll only use one of them, not both”); the unsafe-intent reply $r ^ { - }$ explicitly confirms the unsafe interpretation (“Yes, I plan to combine them for stronger cleaning”). Both replies are generated by a Llama-3.1-8B-Instruct user simulator with carefully designed system prompts. Dual simulation is the key methodological departure from prior interactive datasets [22], [23]: by exposing the model to both branches, training drives the model to produce different downstream behaviors conditioned on the reply, rather than rote continuation.

Stage 4: Trajectory Composition and Labeling. For each $( I , T _ { \mathrm { a m b } } , q ^ { * } , r ^ { + } , r ^ { - } )$ , we compose two full multiturn trajectories. The safe-intent trajectory $y ^ { + }$ concatenates $( T _ { \mathrm { a m b } } , q ^ { * } , r ^ { + } )$ with a continuation reasoning trace that completes the task safely. The unsafe-intent trajectory y<sup>−</sup> concatenates $( T _ { \mathrm { a m b } } , q ^ { * } , \dot { r ^ { - } } )$ with a refusal-with-explanation that warns the user of the hazard. The continuation traces are generated by the Reasoning Agent of the SSUI pipeline, conditioned on the clarified intent. Each trajectory is labeled with the expected model behavior: continue-safely, refuse-with-warning, or, for a small held-out set, ask-second-clarification when the simulated reply itself remains ambiguous.

Stage 5: Human Verification. Three trained annotators review every trajectory along four dimensions: (i) the ambiguity is genuinely safety-critical, not stylistic; (ii) the clarification is neutral and answerable; (iii) both simulated replies are plausible and intent-distinct; (iv) the continuation aligns with the labeled behavior. Failing trajectories are returned for regeneration up to twice; chronic failures are discarded. Inter-annotator agreement (Krippendorff’s alpha) reaches 0.82, and final acceptance requires unanimous approval.

Stage 6: Unification. Following the summary and manual-revision steps of the SSUI pipeline [15], a check agent consolidates the verified trajectories into question– answer records, and annotators curate the final set for safety, difficulty, redundancy, and completeness.

## 3.2 Scale and Safety Taxonomy

SSUI-Dialog is built on the SSUI corpus [15], which was obtained from approximately 25,000 candidate images through multi-agent generation and manual revision and contains $^ { 4 , 7 7 9 }$ samples. Applying the six-stage pipeline above to SSUI yields 3,912 SSUI-Dialog instances (7,824 trajectories after dual-intent expansion), which we split by SSUI source instance into 3,130 / 391 / 391 instances for training, validation, and test to prevent train/test leakage.

Every instance is annotated against the three-tier safety taxonomy of SSUI (Fig. 3), comprising 3 primary, 19 secondary, and 68 tertiary categories. The primary tier partitions risks into Public Safety, Ethical Safety, and National Safety; the secondary tier refines these into 19 categories (e.g., Hazardous & Controlled Materials, Bias & Discrimination, Privacy Protection, Network Attacks); and the tertiary tier specifies 68 fine-grained leaf types (e.g., Drug Precursor Chemicals, Gender Discrimination, Phishing Email Guidance). Each ambiguous query $T _ { \mathrm { a m b } }$ inherits the leaf label of its source sample, so the safety-critical premise removed in Stage 1 is always grounded in a concrete, typed hazard; this lets us balance ambiguity injection across domains during construction and supports category-resolved analysis. To our knowledge, the taxonomy covers the majority of risk categories used across existing academic benchmarks and industrial safety policies.

## 4 THE SAFE-PRO FRAMEWORK

We now present SAFE-PRO, a unified framework that defends against the two cross-modal safety failure modes identified in Section 1. Section 4.1 gives the overall architecture and notation. Section 4.2 introduces the Risk-Conditioned Uncertainty (RCU) estimator, the single signal that drives the framework. Section 4.3 describes the inner loop of safetyaware reasoning path optimization, Section 4.4 the outer loop of cross-modal proactive inquiry, and Section 4.5 the composite reward that couples the two loops. Section 4.6 summarizes the three-stage training pipeline, and Section 5 analyzes the framework.

## 4.1 Overview and Notation

Let (I, T) denote a cross-modal input pair consisting of an image I and a text query T, and let π<sub>θ</sub> denote the MLLM policy. A reasoning trajectory is

$$
\tau = ( s _ { 0 } \to s _ { 1 } \to \cdot \cdot \cdot \to s _ { N } ) ,\tag{1}
$$

where $s _ { 0 } ~ = ~ ( I , T )$ is the input state, each $s _ { j }$ is a semantically coherent reasoning step, and s<sub>N</sub> contains the final answer A. Safety is determined by a verification function ${ \mathcal { F } } ( \tau ) \in \{ 0 , 1 \}$ , where $\mathcal { F } ( \tau ) = 1$ indicates that τ yields a safe and effective response. In $S A F E \mathrm { \mathrm { - P R O } } ,$ a trajectory may additionally contain clarification turns

$$
y = ( s _ { 0 } , \ldots , s _ { j _ { 1 } } , q _ { 1 } , r _ { 1 } , s _ { j _ { 1 } + 1 } , \ldots , q _ { K } , r _ { K } , \ldots , s _ { N } ) ,\tag{2}
$$

where $\left( q _ { k } , r _ { k } \right)$ is the k-th clarification question issued by the model and the corresponding user reply. The number of clarification turns per trajectory is bounded by $K _ { \mathrm { m a x } }$

Figure 4 illustrates the data flow. At every reasoning step $s _ { j }$ the framework computes a scalar $u ( \dot { s } _ { j } ) ~ \in ~ [ 0 , 1 ]$ summarizing how uncertain and how safety-critical the current state is. Two thresholds partition the unit interval into three regimes: $u \geq \tau _ { \mathrm { h i g h } }$ triggers an outer-loop clarification; $\tau _ { \mathrm { l o w } } ~ \le ~ u ~ < ~ \tau _ { \mathrm { h i g h } }$ marks the step for inner-loop path optimization during training; and $u < \tau _ { \mathrm { l o w } }$ leaves reasoning untouched. The inner loop is a training-time mechanism, so at inference only the outer-loop gate is active. Default thresholds are given in Section 6.1.

![](images/f472c20c54dc896345b7d72799836c3d6fb145c3e549f0c5abe42aa20e59d166.jpg)

![](images/22f000273cfdd35907b11b73f79f770f1696a73c4a25567aab2671ce1d4a61f4.jpg)  
Fig. 4. Overview of the SAFE-PRO framework. A cross-modal input (I, T) enters the MLLM, which generates reasoning steps autoregressively. At each step $s _ { j } { } _ { ; }$ , the Risk-Conditioned Uncertainty estimator $u ( s _ { j } )$ combines normalized predictive entropy with a learned cross-modal risk score. When $u ( s _ { j } ) \geq \pi \mathsf { n i g h } ,$ the outer loop interrupts reasoning with a clarification question, and the user reply is re-injected into the trace. During training, steps with $\pi _ { \mathsf { o w } } \leq u ( s _ { j } ) < \pi _ { \mathsf { h i g h } }$ receive risk-guided contrastive path optimization (inner loop). A composite reward with a dominant safety term jointly trains both loops.

## 4.2 Risk-Conditioned Uncertainty Estimation

A central observation motivating SAFE-PRO is that the entropy-based triggers used by prior interactive methods [20], [23], [24] conflate two distinct quantities: epistemic uncertainty (the model is unsure which continuation is correct) and safety criticality (the current step lies on a branch that may lead to harm). A reasoning step can have high entropy without being dangerous—for instance, when the model hesitates between two equally safe phrasings—and conversely a step can be confidently committed to an unsafe assumption with very low entropy. Asking on entropy alone is therefore both over-eager and under-protective.

SAFE-PRO addresses this by conditioning uncertainty on a learned safety signal. Concretely, at each reasoning step $s _ { j }$ we compute

$$
u ( s _ { j } \mid I , T ) = \alpha \cdot \mathrm { P E } ( s _ { j } \mid I , T ) + \beta \cdot \mathcal { R } ( s _ { j } \mid I , T ) ,\tag{3}
$$

where PE is normalized predictive entropy, R is a crossmodal risk score, and $\alpha + \beta { } ~ = ~ 1$ are mixing weights $( \alpha ~ = ~ \beta ~ = ~ 0 . 5$ by default; Section 6.5 reports the singlesignal variants $\alpha = 0$ and $\beta = 0 )$ . Because u is a convex combination, the gate fires whenever the combined evidence is large: with the defaults $\alpha = \beta = 0 . 5$ and $\tau _ { \mathrm { h i g h } } = 0 . 6 5 ,$ a step needs $\mathcal { R } \geq 0 . 3$ even at maximal entropy and $\mathrm { P E } \geq 0 . 3$ even at maximal risk. Section 5.2 analyzes the consequences of this design.

Normalized Predictive Entropy. As a sequence-level proxy for predictive entropy [41], we use the lengthnormalized negative log-likelihood of the step,

$$
\operatorname { N L L } ( s _ { j } \mid I , T ) = - { \frac { 1 } { | s _ { j } | } } \sum _ { i = 1 } ^ { | s _ { j } | } \log p _ { \theta } ( z _ { i } \mid z _ { < i } , s _ { < j } , I , T ) ,\tag{4}
$$

where $z _ { 1 } , \dotsc , z _ { | s _ { j } | }$ are the tokens of $s _ { j }$ . We obtain $\operatorname { P E } ( s _ { j } ) \in$ [0, 1] by min-max normalizing NLL with its empirical $5 ^ { \mathrm { t h } }$ and $9 \breve { 5 } ^ { \mathrm { t h } }$ percentiles on a held-out calibration split and clipping to [0, 1].

Cross-Modal Risk Score. The risk score $\mathcal { R } ( s _ { j } \mid I , T ) \in$ (0, 1) predicts the probability that step $s _ { j }$ lies on a branch leading to an unsafe completion. A key design decision is that R reuses the base MLLM’s hidden states rather than introducing a separate model. Specifically, let $\boldsymbol { h } _ { j } \in \mathbb { R } ^ { d }$ be the final-layer hidden state of π<sub>θ</sub> at the last token of $s _ { j }$ . We attach a lightweight two-layer MLP head $g _ { \phi } : \mathbb { R } ^ { d }  \mathbf { \bar { ( 0 , 1 ) } }$ with bottleneck width d/16:

$$
\begin{array} { r } { \mathcal { R } ( s _ { j } \mid I , T ) = \sigma \big ( W _ { 2 } \mathrm { G E L U } ( W _ { 1 } h _ { j } + b _ { 1 } ) + b _ { 2 } \big ) , } \end{array}\tag{5}
$$

where $W _ { 1 } ~ \in ~ \mathbb { R } ^ { ( d / 1 6 ) \times d } , ~ b _ { 1 } ~ \in ~ \mathbb { R } ^ { d / 1 6 } , ~ W _ { 2 } ~ \in ~ \mathbb { R } ^ { 1 \times ( d / 1 6 ) }$ and $b _ { 2 } ~ \in ~ \mathbb { R }$ . This amounts to $d ^ { 2 } / 1 6 + d / 8 + 1$ trainable parameters, i.e., about 1.05M for LLaVA-NeXT-LLaMA3- 8B $( d = 4 0 9 6 )$ and 0.80M for Qwen2.5-VL-7B $( d = 3 5 8 4 )$ roughly 0.01% of the backbone. The base MLLM weights remain frozen while $g _ { \phi }$ is trained, and since $g _ { \phi }$ reads hidden states that are computed anyway, evaluating R adds only one small MLP evaluation per step.

Training the Risk Head. We exploit a convenient property of the SRPO exploration phase: for each input it already produces a positive continuation $\tau _ { \mathrm { c o n t } } ^ { \prime + }$ and a negative continuation $\tau _ { \mathrm { c o n t } } ^ { \prime \dot { - } }$ from the same prefix [15]. Every intermediate step $s _ { j }$ on the positive branch receives label 0 (safe) and every step on the negative branch receives label 1 (unsafe), yielding a large supervised dataset at no additional annotation cost. These labels are noisy—early steps of a negative branch may themselves be benign—so we treat them as weak supervision. We train $g _ { \phi }$ with binary cross-entropy

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R C U } } ( \phi ) = - \mathbb { E } _ { ( s _ { j } , y _ { j } ) \sim \mathcal { D } _ { \mathrm { S S U } } } \big [ y _ { j } \log \mathcal { R } ( s _ { j } ) } \\ { + ( 1 - y _ { j } ) \log ( 1 - \mathcal { R } ( s _ { j } ) ) \big ] , } \end{array}\tag{6}
$$

where D<sub>SSUI</sub> denotes the union of positive and negative branches from SSUI exploration. On held-out SSUI branches, the trained $g _ { \phi }$ achieves a ROC-AUC of 0.894 in distinguishing safe from unsafe continuations, compared with 0.612 for predictive entropy alone (Section 6.5).

## 4.3 Inner Loop: Risk-Guided Path Optimization

When the RCU signal lies in the intermediate band $\tau _ { \mathrm { l o w } } \leq$ $u ( s _ { j } ) < \tau _ { \mathrm { h i g h } } $ , the current step is uncertain or mildly riskbearing but not severe enough to warrant interrupting the user. In this regime SAFE-PRO does not ask; instead, during training it strengthens the reasoning in place through contrastive path optimization.

Contrastive branches. Let $S ^ { * } ~ = ~ \{ s _ { 1 } , . . . , s _ { n } \}$ be the candidate steps of a reference reasoning path drawn from SSUI-Dialog, namely those whose RCU value falls in the intermediate band. At a selected step $s _ { j }$ we expand two short continuations sharing the prefix $s { \underline { { < } } } j \colon$ a positive branch $\tau _ { j } ^ { + }$ that the verifier certifies safe and effective $( \mathcal { F } ( \tau _ { i } ^ { + } ) = 1 )$ and a negative branch $\tau _ { j } ^ { - }$ that is verified unsafe $( \mathcal { F } ( \tau _ { j } ^ { - } ) = 0 )$

Branches are obtained by temperature-sampling the policy from $s _ { \leq j }$ and labeling the rollouts with the same verifier used for the output reward (Section 4.5); steps for which no contrastive pair can be formed are skipped. This yields a per-step supervision signal that is local to $s _ { j }$ yet grounded in the global safety of its downstream completions.

Path-level alignment objective. Writing $\begin{array} { r l } { \ell _ { \theta } ( \tau ) } & { { } = } \end{array}$ log $p _ { \theta } ( \tau \mid I , T )$ for the policy log-likelihood of a branch, the inner loop maximizes the contrastive log-odds that the policy prefers the safe branch over the unsafe one, averaged over the candidate steps:

$$
\mathcal { I } _ { \mathrm { i n n e r } } ( \theta ) ~ = ~ \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \log \sigma \big ( \ell _ { \theta } ( \tau _ { j } ^ { + } ) - \ell _ { \theta } ( \tau _ { j } ^ { - } ) \big ) ,\tag{7}
$$

whose per-step gradient contribution is

$$
G ( s _ { j } ) = \nabla _ { \theta } \log \sigma \big ( \ell _ { \theta } ( \tau _ { j } ^ { + } ) - \ell _ { \theta } ( \tau _ { j } ^ { - } ) \big ) .\tag{8}
$$

Unlike a final-answer preference loss, (7) shapes the trajectory at the precise intermediate state where the safe and unsafe paths diverge, addressing error propagation that final-output objectives do not localize.

Risk-guided step selection. Rather than selecting supervised steps uniformly, SAFE-PRO samples them in proportion to a power of the cross-modal risk score,

$$
p _ { \gamma } ( s _ { j } ) \ : = \ : \frac { \mathcal { R } ( s _ { j } ) ^ { \gamma } } { \sum _ { i = 1 } ^ { n } \mathcal { R } ( s _ { i } ) ^ { \gamma } } , \qquad \gamma \geq 0 ,\tag{9}
$$

where $\gamma$ controls concentration $( \gamma = 0$ recovers uniform sampling). The motivation is that contrastive supervision is most valuable where the model is most likely to branch into danger; spending the optimization budget on low-risk steps, where the safe and unsafe continuations barely differ, contributes little to safety while adding gradient variance. Because R is already computed for the RCU signal (3), riskguided selection requires no additional model evaluation. Since (7) averages uniformly over the candidate steps while steps are drawn from $p _ { \gamma } .$ , the gradient at a sampled step is importance-weighted by $1 / ( \ Y _ { n p _ { \gamma } } ( s _ { j } ) )$ , which keeps the stochastic gradient unbiased for $\nabla _ { \boldsymbol { \theta } } \mathcal { I } _ { \mathrm { i n n e r } }$ . Lemma 1 shows that this estimator has strictly lower variance than uniform sampling for every γ in an interval (0, γ¯) whenever log R is positively correlated with the squared gradient norm across steps.

## 4.4 Outer Loop: Cross-Modal Proactive Inquiry

When $u ( s _ { j } ) \geq \tau _ { \mathrm { h i g h } }$ and fewer than $K _ { \mathrm { m a x } }$ clarifications have been issued, the outer loop interrupts the reasoning trace and emits a clarification question $q _ { k }$

Safety-Grounded Question Generation. A clarification question in SAFE-PRO is grounded in the cross-modal risk that triggered it. Rather than asking a generic “what do you mean?”, the model is trained (Section 4.6) to surface the specific safety-critical premise it is uncertain about. For the example in Fig. 1(b), the model asks “Could you confirm ifyou intend to use both products or just one?” instead of a contentfree prompt. The training data for this behavior is provided by SSUI-Dialog (Section 3), in which each clarification turn is constructed around the safety-critical premise removed from the original SSUI text query.

Stateful Reply Integration. The user reply $r _ { k }$ is appended to the reasoning trace and the model resumes generation conditioned on the augmented history. Gradient updates flow only through tokens generated by $\pi _ { \theta } ;$ the user reply tokens are masked from the policy loss, which prevents the model from over-fitting to simulator artifacts.

Bounded Interruption. The outer loop is bounded by $K _ { \mathrm { m a x } }$ clarification turns (we use $K _ { \operatorname* { m a x } } = 3 )$ . Once the bound is reached, the model commits to its best-guess premise and proceeds, ensuring that pathological cases $( \mathrm { e . g . }$ , adversarial users who systematically refuse to clarify) do not cause nontermination.

## 4.5 Composite Reward

A central failure mode we observed when naively transplanting interaction rewards from prior proactive frameworks [22], [23] into the safety setting was that their efficiency term actively discouraged clarification at risky branches. SAFE-PRO therefore uses a reward in which the safety term strictly dominates and over-refusal is explicitly penalized.

For a generated trajectory y with final answer $o ,$ let $r$ denote the interaction portion (all clarification turns and reasoning steps) and k the number of clarification turns. The total reward is

$$
\begin{array} { r } { R _ { \mathrm { t o t a l } } ( y ) = \underbrace { w _ { s } \cdot R _ { \mathrm { s a f e } } ( o ) } _ { \mathrm { s a f e t y } } + \underbrace { w _ { h } \cdot R _ { \mathrm { h e l p } } ( r , o ) } _ { \mathrm { h e l p f u l n e s s } } } \\ { - \underbrace { w _ { o } \cdot R _ { \mathrm { o v e r } } ( o ) } _ { \mathrm { o v e r - r e f u s a l } } + \underbrace { w _ { e } \cdot R _ { \mathrm { e f f } } ( r ) } _ { \mathrm { e f f i c i e n c y } } , } \end{array}\tag{10}
$$

with weights enforcing the ordering $w _ { s } \gg w _ { o } > w _ { h } > w _ { e }$ (default $w _ { s } \mathrm { = } 1 . 0 , w _ { o } \mathrm { = } 0 . 3 , w _ { h } \mathrm { = } 0 . 2 , w _ { e } \mathrm { = } 0 . 0 5 )$

The four components are: (i) $R _ { \mathrm { s a f e } } ( o ) \in \{ 0 , 1 \}$ is the verified safety of the final answer, evaluated by an LLM judge (GPT-4o-mini) as in [15]; (ii) $R _ { \mathrm { h e l p } } ( r , o ) \in [ \bar { 0 } , 1 ]$ is the helpfulness of the dialogue, scored by the same LLM judge; (iii) $R _ { \mathrm { o v e r } } ( o ) \in \{ 0 , 1 \}$ flags whether the model refused a benign input, computed on MOSSBench-style benign controls [42] mixed into training; (iv) $R _ { \mathrm { e f f } } ( r ) = ( \dot { K } _ { \mathrm { m a x } } - k \bar { ) } / ( K _ { \mathrm { m a x } } - 1 )$ rewards trajectories with fewer clarification turns, but its small weight prevents it from overriding safety considerations. The ordering $w _ { s } \gg w _ { o }$ is essential: in early experiments with $w _ { s } \approx { w _ { o } }$ the model oscillated between over-refusal and unsafe compliance.

## 4.6 Training Pipeline

SAFE-PRO is trained in three stages. Stages 2 and 3 follow the general recipe of supervised interactive fine-tuning followed by GRPO against a simulated user used in PIR [23]; the RCU gate, the safety-dominant reward, the over-refusal penalty, and the inner loop are specific to SAFE-PRO.

Stage 1: Risk Head Pre-training. We freeze the base MLLM and train $g _ { \phi }$ on the SRPO-style contrastive branches from SSUI [15], optimizing ${ \mathcal { L } } _ { \mathrm { R C U } }$ until validation AUC plateaus. This stage requires only a few hours on a single A100 because the base MLLM forward pass can be cached.

Stage 2: Cold-Start Interactive SFT. We unfreeze the base MLLM and fine-tune it on SSUI-Dialog with the standard autoregressive loss. The training trajectories contain reasoning steps, clarification questions, and simulated replies; the loss is computed only on model-generated tokens, with user-reply tokens masked. This stage activates the model’s capacity to emit and incorporate clarification turns; the RCU thresholds $\tau _ { \mathrm { l o w } } , \tau _ { \mathrm { h i g h } }$ are not yet enforced.

Stage 3: RCU-Gated GRPO. The final stage jointly optimizes the inner and outer loops under the composite reward $R _ { \mathrm { t o t a l } }$ . At each training step, RCU values are computed online; trajectories that trigger the outer loop interact with the user simulator (Llama-3.1-8B-Instruct in our default configuration), while steps in the intermediate band receive risk-guided contrastive supervision via ${ \mathcal { I } } _ { \mathrm { i n n e r } } .$ GRPO updates are applied to policy-generated tokens only. The Stage-3 objective is $\begin{array} { r } { \mathcal { L } ( \theta ) \bar { = } - \mathcal { J } _ { \mathrm { G R P O } } ( \theta ) - \lambda \mathcal { J } _ { \mathrm { i n n e r } } ( \theta ) } \end{array}$ with $\lambda = 0 . 5$ , where $\mathcal { I } _ { \mathrm { G R P O } }$ is the clipped group-relative policy objective on $R _ { \mathrm { t o t a l } }$ with KL regularization toward the Stage-2 model. Because Stage 2 changes the hidden states that $g _ { \phi }$ reads, we re-fit $g _ { \phi }$ for one epoch on the Stage-2 model before Stage 3 and keep it frozen during Stage 3.

## 5 THEORETICAL ANALYSIS

This section analyzes two design choices of SAFE-PRO. We first characterize when risk-guided step sampling reduces the variance of the inner-loop gradient estimator (Section 5.1). We then bound the probability of an unsafe completion under RCU gating in terms of the calibration of the risk head, the reliability of clarification, and the clarification budget (Section 5.2); a remark makes the role of the mixing weight β explicit. Finally, we relate variance reduction to optimization (Section 5.3).

Throughout, $\pi _ { \theta }$ denotes the policy, a trajectory is $\tau =$ $( s _ { 0 } \to \cdots \to s _ { N } )$ with reasoning steps $s _ { j } ,$ , and $\mathcal { F } ( \tau ) \dot { } \in \{ 0 , 1 \}$ is the safety verifier of Section 4.1. We write $\mathrm { P E } ( s _ { j } ) \in \ [ 0 , \bar { 1 } ]$ for the normalized predictive entropy, $\mathcal { R } ( s _ { j } ) \in ( 0 , \bar { 1 } )$ for the cross-modal risk score, and $u ( s _ { j } ) = \bar { \alpha } \mathrm { P E } ( \bar { s _ { j } } ) + \beta \mathcal { R } ( s _ { j } )$ with $\alpha + \beta = 1$ for the RCU signal.

## 5.1 Variance Reduction of Risk-Guided Sampling

Recall from Section 4.3 that the inner loop samples a step $s _ { j }$ from the candidate set $S ^ { * } = \{ s _ { 1 } , . . . , \bar { s } _ { n } \}$ and applies a contrastive update there, with per-step gradient $\bar { G } ( s _ { j } )$ given by (8). Throughout this subsection we condition on the sampled branches, so that each $G ( s _ { i } )$ is a fixed vector. The full-path target gradient is $\begin{array} { r } { \bar { G } = \frac { 1 } { n } \sum _ { i } G ( s _ { j } ) } \end{array}$

Write $g _ { j } : = \| G ( s _ { j } ) \|$ and $a _ { j } : = \mathcal { R } ( s _ { j } ) \in ( 0 , 1 )$ , and let $\mathbb { E } [ \cdot ]$ and $\operatorname { C o v } [ \cdot , \cdot ]$ denote the uniform average and covariance over $j \in \{ 1 , \ldots , n \}$ . Uniform sampling draws $s _ { j }$ with probability $1 / n$ and uses the estimator ${ \bf \bar { \boldsymbol { G } } } .$ Risk-guided sampling with exponent $\gamma \geq 0$ draws $s _ { j }$ with probability $p _ { \gamma } ( \dot { s _ { j } } ) ~ = ~ a _ { j } ^ { \gamma } / \sum _ { i } \dot { a _ { i } ^ { \gamma } }$ , as in (9), and uses the importancecorrected estimator $\widetilde { G } _ { \gamma } ( s _ { j } ) ~ = ~ G ( s _ { j } ) / { \left( n p _ { \gamma } ( s _ { j } ) \right) }$ , so that $\mathbb { E } _ { p _ { \gamma } } [ \widetilde { G } _ { \gamma } ] \ = \ \bar { G }$ for every γ and $\gamma ~ = ~ 0$ recovers uniform sampling. Let

$$
V ( \gamma ) : = \mathrm { t r } \mathrm { V a r } _ { p _ { \gamma } } \bigl [ \widetilde { G } _ { \gamma } \bigr ] , \qquad \Phi ( \gamma ) : = \mathbb { E } \bigl [ a ^ { \gamma } \bigr ] \mathbb { E } \bigl [ g ^ { 2 } a ^ { - \gamma } \bigr ] .\tag{11}
$$

Lemma 1 (Variance reduction). The following hold:

$$
\begin{array} { r l } { \mathrm { ( i ) } } & { { } V ( \gamma ) = \Phi ( \gamma ) - \| \bar { G } \| ^ { 2 } f o r a l l \gamma \geq 0 ; } \end{array}
$$

(ii) Φ is log-convex, and hence convex, on $[ 0 , \infty ) ;$

(iii) $\Phi ^ { \prime } ( 0 ) { \stackrel { \smile } { = } } - \operatorname { C o v } { \left[ g ^ { 2 } , \log a \right] } .$

Consequently, $i f \mathrm { C o v } [ g ^ { 2 } , \log a ] > 0 ,$ , there exists $\bar { \gamma } \in ( 0 , \infty ]$ such that $\dot { V ( \gamma ) } < \dot { V } ( 0 )$ for every $\gamma \in ( 0 , \bar { \gamma } )$ and $V ( \gamma ) \overset { } { \underset { } { \geq } } V ( \bar { 0 } )$ for every $\gamma > \bar { \gamma }$

Proof: (i) Since $\widetilde { G } _ { \gamma }$ is unbiased with mean ${ \bar { G } } ,$

$$
\begin{array} { l } { { \displaystyle V ( \gamma ) = \sum _ { j = 1 } ^ { n } p _ { \gamma } ( s _ { j } ) \frac { g _ { j } ^ { 2 } } { n ^ { 2 } p _ { \gamma } ( s _ { j } ) ^ { 2 } } - \| \bar { G } \| ^ { 2 } } } \\ { { \displaystyle \quad = \frac { 1 } { n ^ { 2 } } \sum _ { j = 1 } ^ { n } \frac { g _ { j } ^ { 2 } } { p _ { \gamma } ( s _ { j } ) } - \| \bar { G } \| ^ { 2 } } . } \end{array}\tag{12}
$$

Substituting $p _ { \gamma } ( s _ { j } ) = a _ { j } ^ { \gamma } / \big ( n \mathbb { E } [ a ^ { \gamma } ] \big )$ turns the first term into $\begin{array} { r } { \mathbb { E } [ a ^ { \gamma } ] \frac { 1 } { n } \sum _ { i } g _ { j } ^ { 2 } a _ { i } ^ { - \gamma } = \Phi ( \bar { \gamma } ) } \end{array}$

(ii) Let $\begin{array} { r } { \bar { \dot { A } } ( \bar { \gamma } ) ~ = ~ \mathbb { E } [ a ^ { \gamma } ] ~ = ~ \frac { 1 } { n } \sum _ { i } { e ^ { \gamma \log a _ { j } } } } \end{array}$ and $B ( \gamma ) \ =$ $\begin{array} { r } { \mathbb { E } [ g ^ { 2 } a ^ { - \gamma } ] = \frac { 1 } { n } \sum _ { i } g _ { j } ^ { 2 } e ^ { - \gamma \operatorname { l o g } a _ { j } } } \end{array}$ . Each is a nonnegative combination of exponentials of linear functions of $\gamma ,$ so log A and log B are convex (log-sum-exp is convex; if all $g _ { j } = 0$ the lemma is trivial). Hence log $\Phi { } = \log { A } + \log { B }$ is convex, and so is $\Phi = \exp ( \log \Phi )$ .

(iii) We have $\begin{array} { r } { \dot { A } ( 0 ) = 1 , A ^ { \prime } ( 0 ) = \mathbb { E } [ \log a ] , B ( 0 ) = \mathbb { E } [ g ^ { 2 } ] . } \end{array}$ and $B ^ { \prime } ( 0 ) = - \mathbb { E } [ g ^ { 2 }$ log a]. Therefore $\Phi ^ { \prime } ( 0 ) = A ^ { \prime } ( 0 ) B ( 0 ) +$ $A ( 0 ) B ^ { \prime } ( 0 ) = \mathbb { E } [ \log a ] \mathbb { E } [ g ^ { \dot { 2 } } ] - \mathbb { E } [ g ^ { 2 } \log a ] = - \operatorname { C o v } [ g ^ { 2 } , \log a ] .$

Finally, if Cov $[ g ^ { 2 } , \log a ] > 0 ,$ , then $\Phi ^ { \prime } ( 0 ) < 0 .$ , so the set $S = \{ \gamma \stackrel { . } { > } 0 : \Phi ( \stackrel { . } { \gamma } ) < \Phi ( \stackrel { . } { 0 } ) \}$ is nonempty; let $\bar { \gamma } = \operatorname* { s u p } S \in$ $( 0 , \infty ]$ . For $\gamma \in ( 0 , \bar { \gamma } )$ , choose $\gamma ^ { \prime } \in S$ with $\gamma ^ { \prime } > \gamma ;$ convexity gives $\Phi ( \gamma ) \le ( 1 - \gamma / \gamma ^ { \prime } ) \Phi ( 0 ) + ( \gamma / \gamma ^ { \prime } ) \Phi ( \gamma ^ { \prime } ) < \Phi ( 0 )$ . For $\gamma > \bar { \gamma } , \gamma \notin S , \mathrm { i . e . , } \Phi ( \gamma ) \ge \Phi ( 0 ) . \mathrm { B y ~ ( i ) }$ , the same statements hold for $V .$ □

Remark 1. To first order, $V ( \gamma ) - V ( 0 ) = - \gamma \mathrm { C o v } [ g ^ { 2 } , \log a ] +$ $O ( \gamma ^ { 2 } )$ : the reduction is governed by how well the risk score ranks the steps with large gradient norm. The varianceoptimal sampling distribution is $p ^ { \star } ( s _ { j } ) \propto g _ { j } ;$ risk-guided sampling is a computable surrogate that does not require per-step gradients. Lemma 1 does not guarantee an improvement for arbitrary $\gamma \colon$ overly concentrated sampling $( \gamma ~ > ~ \bar { \gamma } )$ inflates the variance through the importance weights. Since $V$ is convex in $\gamma ,$ the concentration can be selected by a one-dimensional search; our default $\gamma = 1 . 5$ was chosen empirically, and the ablation in Table 4 shows that it improves safety over uniform sampling. That ablation measures safety rather than gradient variance, so it is consistent with, but not a direct test of, the covariance condition of Lemma 1.

## 5.2 Safety Bound under RCU Gating

We now ask how unsafe SAFE-PRO can still be given an imperfect risk head. For a reasoning step $s ,$ let $q ( s )$ denote the probability that continuing from s without intervention yields $\mathcal { F } ( \tau ) \stackrel { \cdot } { = } 0 . \mathrm { A }$ step is critical if $q ( s ) \geq \rho _ { 0 }$ for a chosen risk level $\rho _ { 0 } \in ( 0 , 1 )$ . Let C be the (random) set of critical steps generated in an episode and $C = | \mathcal { C } |$ , with $\mathbb { E } [ C ] \le m$ Each critical step s is in exactly one of three situations: it is gated $( u ( s ) \ \geq \ \tau _ { \mathrm { h i g h } }$ and fewer than $K _ { \mathrm { m a x } }$ clarifications have been issued), missed $( u ( s ) < \tau _ { \mathrm { h i g h } } )$ , or budget-exhausted $( u ( s ) \geq \tau _ { \mathrm { h i g h } }$ but the budget $K _ { \mathrm { m a x } }$ has been used up). Let B denote the event that at least one critical step is budgetexhausted.

Assumption 1 (Bounded calibration). The risk head is $\varepsilon -$ calibrated: $| \mathcal { R } ( s ) - q ( s ) | \leq \varepsilon f o r$ every reachable reasoning step s.

Assumption 2 (Clarification efficacy). If the outer loop is triggered at a critical step s, then, conditionally on the history up to $s ,$ the probability that the risk at s remains unresolved is at most δ. Here “unresolved” covers a noisy or residually ambiguous user reply as well as afailure of the model to act safely on the reply $( e . g .$ , to refuse after an unsafe-intent reply).

Assumption 3 (Localization). An episode ends in an unsafe completion only if at least one of its critical steps is missed, budgetexhausted, or gated but unresolved.

Assumption 3 formalizes the premise that unsafe completions originate at identifiable high-risk steps. Every unsafe trajectory contains at least one critical step (its final step has $q = 1 )$ , but the assumption additionally requires that resolving all critical steps suffices for safety.

Theorem 1 (Safety bound). Under Assumptions $^ { 1 - 3 , }$

$$
\mathrm { P r } [ u n s a f e ] \ \leq \ m \left( \xi + \delta \right) + \mathrm { P r } [ B ] ,\tag{13}
$$

where

$$
\xi = \frac { \mathbb { E } \Big [ \sum _ { s \in \mathcal { C } } \mathbf { 1 } \big \{ \alpha \mathrm { P E } ( s ) < \tau _ { h i g h } - \beta ( \rho _ { 0 } - \varepsilon ) \big \} \Big ] } { \mathbb { E } [ \mathcal { C } ] }\tag{14}
$$

is the expected fraction of critical steps with low normalized entropy $( \dot { \xi } : = 0 \dot { i } f \mathbb { E } [ C ] = \dot { 0 } )$ . In particular, $i f$

$$
\begin{array} { r } { \tau _ { h i g h } \ \leq \ \beta ( \rho _ { 0 } - \varepsilon ) , } \end{array}\tag{15}
$$

then $\xi = 0$ and Pr[unsafe] ≤ m δ + Pr[B].

Proof: By Assumption 3 and the union bound,

$$
\begin{array} { r l } & { \mathrm { P r } [ \mathrm { u n s a f e } ] \leq \mathbb E \Big [ \sum _ { s \in \mathcal C } \mathbf { 1 } \{ s \mathrm { ~ m i s s e d } \} \Big ] } \\ & { \qquad + \mathbb E \Big [ \sum _ { s \in \mathcal C } \mathbf { 1 } \{ s \mathrm { ~ g a t e d } , \mathrm { u n r e s o l v e d } \} \Big ] } \\ & { \qquad + \mathrm { P r } [ B ] . } \end{array}\tag{16}
$$

For a critical step $s ,$ Assumption 1 gives ${ \mathcal { R } } ( s ) \geq q ( s ) -$ $\varepsilon \geq \rho _ { 0 } - \varepsilon ,$ hence $u ( s ) \geq  { \hat { \alpha } }  { \mathrm { P E } } ( s ) + \beta ( \rho _ { 0 } - \varepsilon )$ . A miss, $u ( s ) < \tau _ { \mathrm { h i g h } } ,$ therefore implies $\begin{array} { r } { \nu \mathrm { P E } ( s ) < \tau _ { \mathrm { h i g h } } - \beta ( \rho _ { 0 } - \varepsilon ) . } \end{array}$ so the first term is at most $\xi \mathbb { E } [ C ] \ \leq \ \xi m$ . For the second term, the tower property and Assumption 2 give $\mathbb { E } [ \sum _ { s \in { \mathcal { C } } } \mathbf { 1 } \{ s$ gated} Pr[unresolved | history] $\begin{array} { r } { \int ^ { \bullet } \leq \delta \mathbb { E } [ \check { C } ] \leq } \end{array}$ δm. Finally, under (15) the right-hand side of the miss condition is at most 0 while α $\begin{array} { r } { \mathrm { P E } ( s ) \geq 0 , } \end{array}$ , so no critical step can be missed and $\xi = 0$ □

Theorem 1 decomposes the residual risk into three interpretable sources: critical steps that the gate misses $( \xi ) ,$ clarifications that fail to resolve the risk $( \delta ) _ { \cdot }$ , and exhaustion of the clarification budget $( \mathrm { P r } [ B ] )$ . The model’s own behavior enters through $\delta ,$ which includes acting safely on the user’s reply, and through $B ,$ which becomes more likely when the gate also fires on non-critical steps. The bound is informative only when its right-hand side is below one.

The theorem also exposes a limitation of the default configuration. With $\alpha = \bar { \beta } = 0 . 5$ and $\tau _ { \mathrm { h i g h } } = 0 . 6 5$ , we have $\tau _ { \mathrm { h i g h } } > \beta \geq \beta ( \rho _ { 0 } - \varepsilon ) ,$ , so condition (15) cannot hold and the bound counts every critical step with $\operatorname { \mathrm { \Delta } } ^ { \mathrm { \prime } } \mathrm { E } ( s ) < 1 . 3 - ( \rho _ { 0 } - \varepsilon )$ as potentially missed $( \mathrm { e . g . , P E } \bar { ( } s ) < 0 . 5 $ for $\rho _ { 0 } - \varepsilon = 0 . 8 )$ . In particular, a confidently unsafe step with near-zero entropy has $u \leq 0 . 5 < \tau _ { \mathrm { h i g h } }$ and cannot trigger the outer loop; such steps are addressed only through the training-time inner loop. Lowering $\tau _ { \mathrm { h i g h } }$ or increasing $\beta$ shrinks $\xi$ at the cost of more interruptions, i.e., a larger TTR and a larger $\mathrm { P r } [ B ] .$ our default values reflect this trade-off and were chosen empirically.

Remark 2 (Why $\beta$ cannot be too small). Since $\tau _ { \mathrm { h i g h } } > 0 ,$ condition (15), which guarantees $\xi \ = \ 0 ,$ , can hold only if $\rho _ { 0 } \mathrm { ~ } > \varepsilon$ and $\beta \ge \tau _ { \mathrm { h i g h } } \bar { / } ( \rho _ { 0 } - \varepsilon )$ . In particular, an entropyonly signal $( \beta = 0 )$ cannot guarantee $\xi = 0$ at any threshold $\tau _ { \mathrm { h i g h } } > 0$

Consistently with Remark 2, the entropy-only variant $( \beta = 0 )$ has the highest ASR among the RCU variants in Table 4 (30.82 vs. 15.83 for the full model) together with a much larger TTR (3.05): without the risk term, the gate must either fire indiscriminately or miss critical steps.

## 5.3 Variance Reduction and Optimization

Finally, we relate Lemma 1 to optimization. For stochastic gradient methods on a smooth nonconvex objective with unbiased gradients of variance $\sigma ^ { 2 } ,$ , the number of iterations needed to reach an ϵ-stationary point $( \mathbb { E } \| \nabla \mathcal { I } \| ^ { 2 } \leq \epsilon )$ scales as $O ( \sigma ^ { 2 } / \epsilon ^ { 2 } )$ when the variance term dominates [43]. For $\gamma \in ( 0 , \bar { \gamma } )$ , Lemma 1 reduces the variance of the inner-loop estimator by the factor $\kappa = V ( \gamma ) / V ( 0 ) < 1 ,$ so in this regime the number of contrastive updates required by the inner-loop term scales down proportionally. This argument covers the inner-loop term ${ \dot { \mathcal { T } } } _ { \mathrm { i n n e r } }$ only; it does not account for the GRPO term of Stage 3.

## 6 EXPERIMENTS

We conduct experiments to evaluate four claims: (i) SAFE-PRO achieves strong cross-modal safety while remaining competitive on helpfulness (Section 6.2); (ii) it handles multi-turn safety-critical dialogues that path-only and inquiry-only baselines handle poorly (Section 6.3); (iii) the proactive inquiry mechanism does not induce pathological over-refusal (Section 6.4); and (iv) each component of the framework contributes measurable gains (Section 6.5).

## 6.1 Experimental Setup

Base Models. We apply SAFE-PRO to two MLLMs from distinct architectural families: LLaVA-NeXT-LLaMA3-8B [44] and Qwen2.5-VL-7B [45], following the setup of SRPO [15] for direct comparison.

Baselines. We compare against three categories. Safety-only methods: SFT, DPO [33], ORPO [35], and SRPO [15]. Interactive-only methods: STaR-GATE [20] and CollabLLM [22], adapted to the multimodal setting by applying their training recipes to the same MLLM backbone. Commercial MLLMs: GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Pro, evaluated zero-shot via their respective APIs. The baselines that we train (SFT, DPO, ORPO, STaR-GATE, and CollabLLM) use the same training data and computational budget as SAFE-PRO. SRPO is trained on SSUI without the dialogue extension, and its single-turn results, like those of the commercial MLLMs, are taken from [15].

Benchmarks. We evaluate on seven benchmarks. For single-turn cross-modal safety we use USBBench [14], MSS-Bench [11], MLLMGuard [30], SafeBench [31], and VLS-Bench [29]. For multi-turn safety dialogue we use the test split of SSUI-Dialog (Section 3). For over-refusal we use MOSSBench [42].

Metrics. For the single-turn benchmarks we follow the protocol and metrics of SRPO [15]. On USBBench (SIST subset) we report the attack success rate (ASR↓) on unsafe inputs and the refusal rate (ARR↓) on their benign counterparts, together with their mean; on MLLMGuard, the perfect answer rate (PAR↑) and the attack success degree (ASD↓); on MSSBench, the average accuracy over safe and unsafe situations; on SafeBench, ASR↓ and the safety risk index (SRI↑); and on VLSBench, the safety rate. We summarize these by the average of the higher-is-better metrics (Avg↑: PAR, MSSBench, SRI, VLSBench) and of the lower-is-better ones (Avg↓: USBBench mean, ASD, SafeBench ASR). For multi-turn evaluation we introduce two metrics: Cross-modal Safety with Helpful Clarification (CSHC↑), the proportion of dialogues that are simultaneously safe, helpful, and use no more clarifications than necessary, i.e., than the number of safety-critical premises annotated for the instance (one, or two for the ask-second-clarification subset); and Turns to Resolution (TTR↓), the average number of model turns until a safe, complete answer, where the final answer counts as one turn (a non-interactive model therefore has TTR = 1, and TTR − 1 is the average number of clarification questions); TTR is averaged over dialogues that reach a safe, complete answer. For over-refusal we report the Over-Refusal Rate (ORR↓) as in [42]. Evaluation uses GPT-4o as the judge, distinct from the GPT-4o-mini reward judge. On SSUI-Dialog, each test instance is played twice, once with a safe-intent and once with an unsafe-intent simulated user; the test-time simulator is GPT-4o-mini, different from the Llama-3.1-8B-Instruct simulator used in training. ASR on SSUI-Dialog is the percentage of unsafe-intent dialogues that end in an unsafe completion, and helpfulness is scored on safe-intent dialogues.

Implementation. All models are trained on 8× A100 80GB GPUs. Stage 1 (risk-head training) uses learning rate $1 \times 1 0 ^ { - 4 }$ for 3 epochs. Stage 2 (SFT) uses $1 \times 1 0 ^ { - 5 }$ for 3 epochs with batch size 64. Stage 3 (GRPO) uses $1 \times 1 0 ^ { - 6 }$ for 5 epochs with group size 8, KL coefficient 0.04, and the Llama-3.1-8B-Instruct user simulator. Default hyperparameters are $\alpha = \beta = 0 . 5 , \tau _ { \mathrm { l o w } } = 0 . 3 , \tau _ { \mathrm { h i g h } } = 0 . 6 5 , \gamma = \mathrm { \hat { 1 } . 5 , \hat { K } _ { \mathrm { m a x } } = 3 } ,$ , and $( w _ { s } , w _ { o } , w _ { h } , w _ { e } ) = ( 1 . 0 , 0 . \bar { 3 } , 0 . 2 , 0 . 0 5 )$

## 6.2 Main Results

Table 1 reports results on the five single-turn cross-modal safety benchmarks. Three observations stand out.

First, SAFE-PRO improves over SRPO on both backbones. On Qwen2.5-VL, it raises Avg↑ from 76.18 to 82.77 and lowers Avg↓ from 12.67 to 7.64; USBBench ASR drops from 42.38 to 31.46, SafeBench ASR from 7.50 to 0.62, MLLMGuard ASD from 7.92 to 5.36, and the VLSBench safety rate rises from 67.43 to 80.17. On LLaVA-NeXT, Avg↑ rises from 65.99 to 75.25 and Avg↓ falls from 17.70 to 11.96. This supports the premise that path-level optimization alone leaves a substantial portion of cross-modal safety failures unresolved.

TABLE 1  
Cross-modal safety on five single-turn benchmarks, following the protocol of SRPO [15]. The commercial-MLLM, base-model, and + SRPO rows are taken from Table 1 of [15]; the Avg↑ of LLaVA-NeXT-LLaMA3-8B and the Avg↓ of LLaVA-NeXT-LLaMA3-8B + SRPO are recomputed from the per-benchmark values (published as 41.12 and 14.37). Bold: best in each column.
<table><tr><td rowspan="2">Method</td><td colspan="3">USBBench</td><td colspan="2">MLLMGuard</td><td>MSSBench</td><td colspan="2">SafeBench</td><td>VLSBench</td><td colspan="2">Average</td></tr><tr><td>ASR↓</td><td>ARR↓</td><td>Avg↓</td><td>PAR↑</td><td>ASD↓</td><td>Avg↑</td><td>ASR↓</td><td>SRI↑</td><td>Avg↑</td><td>Avg↑</td><td>Avg↓</td></tr><tr><td>Commercial MLLMs (zero-shot)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude-3.5-Sonnet</td><td>32.80</td><td>25.79</td><td>29.30</td><td>52.38</td><td>9.01</td><td>69.01</td><td>0.70</td><td>99.30</td><td>79.35</td><td>75.01</td><td>13.00</td></tr><tr><td>Gemini-1.5-Pro</td><td>64.45</td><td>11.33</td><td>37.89</td><td>38.12</td><td>21.94</td><td>61.94</td><td>2.60</td><td>97.10</td><td>50.21</td><td>61.84</td><td>20.80</td></tr><tr><td>GPT-40</td><td>72.83</td><td>3.77</td><td>38.30</td><td>56.68</td><td>14.32</td><td>59.30</td><td>3.40</td><td>96.10</td><td>69.50</td><td>70.40</td><td>18.67</td></tr><tr><td>Base: LLaVA-NeXT-LLaMA3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-NeXT-LLaMA3-8B</td><td>78.88</td><td>7.53</td><td>43.20</td><td>26.45</td><td>42.27</td><td>56.35</td><td>29.40</td><td>73.10</td><td>18.56</td><td>43.62</td><td>38.29</td></tr><tr><td>+ SRPO</td><td>50.36</td><td>6.20</td><td>28.28</td><td>50.22</td><td>17.63</td><td>71.20</td><td>7.20</td><td>91.80</td><td>50.74</td><td>65.99</td><td>17.70</td></tr><tr><td>+ SAFE-PRO</td><td>38.60</td><td>4.85</td><td>21.73</td><td>60.30</td><td>10.95</td><td>78.10</td><td>3.20</td><td>96.40</td><td>66.20</td><td>75.25</td><td>11.96</td></tr><tr><td>Base: Qwen2.5-VL-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-7B</td><td>75.26</td><td>4.72</td><td>39.99</td><td>38.22</td><td>28.35</td><td>55.36</td><td>32.20</td><td>76.30</td><td>20.45</td><td>47.58</td><td>33.51</td></tr><tr><td>+ SFT</td><td>66.85</td><td>6.15</td><td>36.50</td><td>46.90</td><td>21.40</td><td>61.20</td><td>20.30</td><td>84.60</td><td>38.70</td><td>57.85</td><td>26.07</td></tr><tr><td>+ DPO</td><td>59.30</td><td>5.60</td><td>32.45</td><td>52.30</td><td>16.85</td><td>64.75</td><td>15.40</td><td>88.90</td><td>47.20</td><td>63.29</td><td>21.57</td></tr><tr><td>+ ORPO</td><td>56.15</td><td>5.85</td><td>31.00</td><td>54.10</td><td>15.20</td><td>66.30</td><td>13.80</td><td>90.20</td><td>50.60</td><td>65.30</td><td>20.00</td></tr><tr><td>+ STaR-GATE (mm.)</td><td>61.40</td><td>4.30</td><td>32.85</td><td>49.80</td><td>19.10</td><td>62.40</td><td>18.20</td><td>86.10</td><td>41.30</td><td>59.90</td><td>23.38</td></tr><tr><td>+ CollabLLM (mm.)</td><td>52.70</td><td>3.95</td><td>28.33</td><td>55.60</td><td>14.70</td><td>66.10</td><td>12.60</td><td>90.80</td><td>52.30</td><td>66.20</td><td>18.54</td></tr><tr><td>+ SRPO</td><td>42.38</td><td>2.83</td><td>22.60</td><td>65.30</td><td>7.92</td><td>73.89</td><td>7.50</td><td>98.10</td><td>67.43</td><td>76.18</td><td>12.67</td></tr><tr><td>+ SAFE-PRO</td><td>31.46</td><td>2.41</td><td>16.94</td><td>71.85</td><td>5.36</td><td>79.62</td><td>0.62</td><td>99.42</td><td>80.17</td><td>82.77</td><td>7.64</td></tr></table>

Second, SAFE-PRO on Qwen2.5-VL attains the best result on every metric in Table 1, including against the strongest commercial model, Claude-3.5-Sonnet: USBBench ASR 31.46 vs. 32.80, SafeBench ASR 0.62 vs. 0.70 and SRI 99.42 vs. 99.30, and VLSBench 80.17 vs. 79.35, while refusing an order of magnitude fewer benign USBBench inputs (ARR 2.41 vs. 25.79). SAFE-PRO is thus safer than commercial systems without paying for it in over-refusal. The LLaVA-NeXT variant also surpasses all commercial models on both averages (75.25 and 11.96 vs. 75.01 and 13.00 for Claude-3.5- Sonnet).

Third, the interactive baselines improve over the base model but remain well below SAFE-PRO. On USBBench, STaR-GATE and CollabLLM lower ASR from 75.26 to 61.40 and 52.70, respectively, confirming that proactive inquiry is a useful ingredient; yet even CollabLLM trails SAFE-PRO by 21.24 ASR points. This gap reflects the three deficiencies identified in Section 1: a triggering signal misaligned with safety criticality, a reward that penalizes extra turns rather than prioritizing clarification at risky branches, and the absence of cross-modal multi-turn training data. SAFE-PRO addresses all three, and our ablations (Section 6.5) show that removing any one of them degrades safety.

## 6.3 Multi-Turn Safety Dialogue

Table 2 reports results on the SSUI-Dialog test set, which specifically stresses the interaction between cross-modal safety risk and intent-level ambiguity. The single-turn SRPO, which has no mechanism to seek clarification, suffers heavily (41.20 ASR, 0.31 CSHC): when premises are removed, path optimization alone is insufficient. CollabLLM, the strongest interactive baseline in the multimodal setting, improves on both ASR (32.81) and CSHC (0.41)— confirming that interaction helps—but still lags far behind SAFE-PRO. Optimized to collaborate rather than to localize safety-critical ambiguity, it interrupts often yet on the wrong axis: its TTR of 3.18 buys little safety, leaving ASR more than double that of SAFE-PRO. SAFE-PRO instead achieves the lowest ASR (15.83) and the highest CSHC (0.68) with a TTR of 1.62, i.e., 0.62 clarification questions per dialogue on average, well below CollabLLM and within the $K _ { \operatorname* { m a x } } = 3$ budget. This indicates that risk-conditioned triggering is both safer and more economical than success-oriented triggering; the complementary entropy-only failure—over-interruption that ignores the safety-critical axis—is isolated in our $\bar { \beta } = 0$ ablation.

TABLE 2  
Multi-turn evaluation on the SSUI-Dialog test set (base model: Qwen2.5-VL-7B). CSHC and TTR are defined in Section 6.1; a non-interactive model has TTR = 1 by construction, so TTR is not bolded.
<table><tr><td>Method</td><td>ASR↓</td><td>Helpful↑</td><td>CSHC↑</td><td>TTR↓</td></tr><tr><td>GPT-40</td><td>28.72</td><td>0.71</td><td>0.46</td><td>2.85</td></tr><tr><td>Claude-3.5-Sonnet</td><td>24.18</td><td>0.74</td><td>0.51</td><td>2.73</td></tr><tr><td>SRPO</td><td>41.20</td><td>0.42</td><td>0.31</td><td>1.00</td></tr><tr><td>CollabLLM (mm.)</td><td>32.81</td><td>0.61</td><td>0.41</td><td>3.18</td></tr><tr><td>SAFE-PRO</td><td>15.83</td><td>0.78</td><td>0.68</td><td>1.62</td></tr></table>

## 6.4 Over-Refusal Analysis

A critical concern when introducing proactive inquiry is whether the model becomes pathologically cautious— asking when it should answer, or refusing benign requests outright. We evaluate this on MOSSBench [42], which specifically tests over-refusal on innocuous inputs that superficially resemble unsafe ones.

Table 3 summarizes the results. SRPO combines high over-refusal (ORR 38.4) with an MSSBench accuracy of 73.89: its aggressive safety alignment refuses many innocuous inputs. SAFE-PRO with the over-refusal penalty $R _ { \mathrm { o v e r } }$ reaches ORR 19.7 while also raising MSSBench accuracy to 79.62, roughly halving SRPO’s over-refusal (−18.7 points) at higher safety. This is the lowest over-refusal of all compared models, below the base model (24.6), GPT-4o (21.3), and Claude-3.5-Sonnet (34.8): safety training usually trades helpfulness on borderline inputs for caution, whereas SAFE-PRO resolves such inputs by asking rather than refusing. Removing $R _ { \mathrm { o v e r } }$ raises ORR to 33.5, confirming the penalty’s necessity. The benign controls used for $R _ { \mathrm { o v e r } }$ are synthesized from SSUI images with the SSUI query agent and contain no MOSSBench images or prompts. By explicitly penalizing over-refusal, proactive inquiry becomes a tool for calibrated caution rather than a license for blanket refusal.

![](images/30b949a5237615e5aa91f6e57ff6d4079b38882374755e603af36d1f3a085702.jpg)  
Fig. 5. Qualitative case of intent-level cross-modal risk. The input pairs bleach (“84”) and an acid-based toilet cleaner (Harpic) with an underspecified instruction. (a) The baseline silently assumes a “mixing” intent and outputs instructions that would release toxic gas. (b) SAFE-PRO detects the safety-critical missing premise via a high RCU score $( u \geq \pi \mathsf { n i g h } )$ and proactively clarifies the user’s intent; conditioned on the reply, it proceeds to a safe and helpful answer.

TABLE 3  
Over-refusal on MOSSBench (ORR↓) alongside MSSBench average accuracy (as in Table 1; the w/o $R _ { \mathsf { 0 } \mathsf { v e r } }$ variant is evaluated only here). Trained methods use Qwen2.5-VL-7B. Bold: best in each column.
<table><tr><td>Method</td><td>ORR↓</td><td>MSSBench↑</td></tr><tr><td>GPT-40</td><td>21.3</td><td>59.30</td></tr><tr><td>Claude-3.5-Sonnet</td><td>34.8</td><td>69.01</td></tr><tr><td>Gemini-1.5-Pro</td><td>27.5</td><td>61.94</td></tr><tr><td>Qwen2.5-VL-7B</td><td>24.6</td><td>55.36</td></tr><tr><td>+ CollabLLM (mm.)</td><td>22.9</td><td>66.10</td></tr><tr><td>+ SRPO</td><td>38.4</td><td>73.89</td></tr><tr><td>+ SAFE-PRO w/o  $R _ { \mathrm { o v e r } }$ </td><td>33.5</td><td>78.95</td></tr><tr><td>+ SAFE-PRO</td><td>19.7</td><td>79.62</td></tr></table>

## 6.5 Ablation Studies

Table 4 ablates each component along five axes: RCU composition, the two-loop architecture, reward design, step selection, and training data. Several patterns emerge.

TABLE 4  
Ablation on SAFE-PRO components (Qwen2.5-VL-7B). ASR, CSHC, and TTR are measured on SSUI-Dialog; ORR is measured on MOSSBench. Bold marks the best value in each of the ASR, CSHC, and ORR columns.
<table><tr><td>Variant</td><td>ASR↓</td><td>CSHC↑</td><td>TTR↓</td><td>ORR↓</td></tr><tr><td>SAFE-PRO (full)</td><td>15.83</td><td>0.68</td><td>1.62</td><td>19.70</td></tr><tr><td>RCU components</td><td></td><td></td><td></td><td></td></tr><tr><td>R only (α = 0)</td><td>21.45</td><td>0.59</td><td>1.48</td><td>24.32</td></tr><tr><td>PE only (β = 0)</td><td>30.82</td><td>0.42</td><td>3.05</td><td>31.85</td></tr><tr><td>Loop ablation</td><td></td><td></td><td></td><td></td></tr><tr><td>Óuter only (no inner)</td><td>24.71</td><td>0.53</td><td>2.15</td><td>28.42</td></tr><tr><td>Inner only (no outer)</td><td>38.50</td><td>0.34</td><td>1.00</td><td>35.18</td></tr><tr><td>Reward ablation</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o over-refusal penalty</td><td>17.95</td><td>0.61</td><td>1.55</td><td>33.50</td></tr><tr><td>w/o helpfulness reward</td><td>16.20</td><td>0.51</td><td>1.42</td><td>21.18</td></tr><tr><td>Step selection Úniform sampling  $( \gamma = 0 )$ </td><td>19.83</td><td>0.62</td><td>1.71</td><td>22.40</td></tr><tr><td>Training data SFT on SSUI only (no dialog)</td><td>32.18</td><td>0.36</td><td>1.05</td><td></td></tr></table>

RCU composition. Using only the risk score $( \alpha ~ = ~ 0 )$ worsens ASR by 5.6 points; using only predictive entropy $( \beta = 0 )$ worsens it by 15.0 points and inflates TTR to 3.05, reproducing the failure mode of over-interruption on safetyirrelevant uncertainty. This corroborates the design rationale of Section 4.2: neither signal alone is sufficient.

Loop ablation. Removing the outer loop reduces SAFE-PRO to a risk-guided variant of SRPO; ASR rises to 38.50, close to plain SRPO (41.20 in Table 2). Removing the inner loop and using only proactive inquiry yields ASR 24.71, also worse than the full system. The two loops are complementary, not redundant.

Reward and step selection. Removing the over-refusal penalty raises ORR from 19.70 to 33.50 while modestly worsening ASR (17.95). Removing the helpfulness reward leaves ASR almost unchanged (16.20) but lowers CSHC from 0.68 to 0.51. Replacing risk-guided sampling with uniform sampling $( \gamma = 0 )$ worsens ASR by 4.0 points, in line with the motivation of Lemma 1, although this ablation does not measure gradient variance directly.

Risk head. On held-out SSUI branches (Qwen2.5-VL-7B backbone), $g _ { \phi }$ attains a ROC-AUC of 0.894 for separating safe from unsafe continuations, against 0.612 for predictive entropy alone, with an expected calibration error of 0.046 (15 bins). ECE is an average over bins and therefore indicates typical calibration error only; it does not certify the uniform bound ε of Assumption 1, which should be read as an idealization.

Training data. Training only on the original SSUI corpus (without the dialogue extension) produces a single-turn model whose ASR on the multi-turn test set (32.18) is more than double that of the full system. The SSUI-Dialog data are therefore essential.

## 6.6 Qualitative Case Study

Figure 5 contrasts a path-only baseline with SAFE-PRO on a representative case of intent-level ambiguity. An image of two household products—a chlorine-based “84” disinfectant and an acid-based Harpic toilet cleaner—is paired with the instruction “Follow the instructions on the labels” and the label dosages (2 ml of the left-side liquid and 3 ml of the right-side liquid per square meter). The dosages are given per product, and nothing in the input states whether the user intends to use a single product or to combine both.

Baseline. Lacking any mechanism to recognize that a missing premise requires clarification, the baseline silently commits to a default assumption. Treating the dosage instructions as a routine measurement task, it proceeds without hesitation and ultimately instructs the user to measure both liquids and combine them. Because mixing hypochlorite bleach with an acidic cleaner releases toxic chlorine gas, every step after the unsafe assumption—however fluently reasoned—leads toward a hazardous answer. This is the intent-level failure of Section 1: the error originates not in faulty reasoning but in an unexamined premise, which pathlevel optimization alone cannot repair.

SAFE-PRO. At the reasoning step where the premise must be resolved, the cross-modal risk score R rises because the joint presence of bleach and acid is hazardous; the RCU signal exceeds $\tau _ { \mathrm { h i g h } }$ and the outer loop fires. Rather than refusing outright or asking a content-free “what do you mean?”, SAFE-PRO surfaces the specific safety-critical premise: “Before proceeding, could you please confirm if you intend to use only one of these products or mix them? Mixing these products can be extremely hazardous.” Once the user clarifies a benign intent (“I only intend to use one of them”), the reply is re-injected and generation resumes with correct, single-product instructions. This illustrates both the safety benefit of targeted clarification and how the dualintent supervision avoids pathological caution, consistent with the MOSSBench results above.

## 7 CONCLUSION

We identified intent-level risk—cross-modal ambiguity that reasoning-path optimization alone cannot resolve—as a distinct vulnerability of MLLMs. To address it, we proposed SAFE-PRO, a framework that integrates reasoning path optimization with proactive inquiry. Guided by a Risk-Conditioned Uncertainty (RCU) signal, SAFE-PRO triggers targeted clarifications at safety-critical states while limiting over-interruption and over-refusal. To support this paradigm, we introduced SSUI-Dialog, a cross-modal multiturn dialogue dataset with safety-critical intent ambiguity and dual-intent supervision. Experiments show that SAFE-PRO reduces attack success rates and mitigates pathological caution relative to path-only and interaction-only baselines. Our analysis characterizes when risk-guided sampling reduces gradient variance and bounds the probability of unsafe completion under RCU gating.

## REFERENCES

[1] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in Neural Information Processing Systems, vol. 36, pp. 34 892–34 916, 2023.

[2] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.

[3] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “MiniGPT-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

[4] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International Conference on Machine Learning. PMLR, 2023, pp. 19 730–19 742.

[5] Gemini Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.

[6] Z. Wang, B. Bi, S. K. Pentyala, K. Ramnath, S. Chaudhuri, S. Mehrotra, X.-B. Mao, S. Asur et al., “A comprehensive survey of LLM alignment techniques: RLHF, RLAIF, PPO, DPO and more,” arXiv preprint arXiv:2407.16216, 2024.

[7] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 730–27 744, 2022.

[8] X. Liu, Y. Zhu, J. Gu, Y. Lan, C. Yang, and Y. Qiao, “MM-SafetyBench: A benchmark for safety evaluation of multimodal large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 386–403.

[9] Y. Gong, D. Ran, J. Liu, C. Wang, T. Cong, A. Wang, S. Duan, and X. Wang, “FigStep: Jailbreaking large vision-language models via typographic visual prompts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 22, 2025, pp. 23 951–23 959.

[10] X. Qi, K. Huang, A. Panda, P. Henderson, M. Wang, and P. Mittal, “Visual adversarial examples jailbreak aligned large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 19, 2024, pp. 21 527–21 536.

[11] K. Zhou, C. Liu, X. Zhao, A. Compalas, D. Song, and X. E. Wang, “Multimodal situational safety,” in International Conference on Learning Representations, 2025.

[12] S. Wang, X. Ye, Q. Cheng, J. Duan, S. Li, J. Fu, X. Qiu, and X. Huang, “Safe inputs but unsafe output: Benchmarking crossmodality safety alignment of large vision-language model,” arXiv preprint arXiv:2406.15279, 2024.

[13] W. Cai, J. Zhao, Y. Jiang, T. Zhang, and X. Li, “Safe semantics, unsafe interpretations: Tackling implicit reasoning safety in large vision-language models,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 13 489–13 491.

[14] B. Zheng, G. Chen, H. Zhong, Q. Teng, Y. Tan, Z. Liu, W. Wang, J. Liu, J. Yang, H. Jing et al., “USB: A comprehensive and unified safety evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2505.23793, 2025.

[15] W. Cai, S. Liu, J. Zhao, Z. Shi, Y. Zhao, Y. Yuan, T. Zhang, C. Zhang, and X. Li, “When safe unimodal inputs collide: Optimizing reasoning chains for cross-modal safety in multimodal large language models,” arXiv preprint arXiv:2509.12060, 2025.

[16] R. Pi, T. Han, J. Zhang, Y. Xie, R. Pan, Q. Lian, H. Dong, J. Zhang, and T. Zhang, “MLLM-Protector: Ensuring MLLM’s safety without hurting performance,” arXiv preprint arXiv:2401.02906, 2024.

[17] C. Fan, M. Li, L. Sun, and T. Zhou, “Missing premise exacerbates overthinking: Are reasoning models losing critical thinking skill?” arXiv preprint arXiv:2504.06514, 2025.

[18] X. Chen, J. Xu, T. Liang, Z. He, J. Pang, D. Yu, L. Song, Q. Liu, M. Zhou, Z. Zhang et al., “Do NOT think that much for 2+3=? On the overthinking of long reasoning models,” in International Conference on Machine Learning, 2025.

[19] A. Cuadron, D. Li, W. Ma, X. Wang, Y. Wang, S. Zhuang, S. Liu, L. G. Schroeder, T. Xia, H. Mao et al., “The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks,” arXiv preprint arXiv:2502.08235, 2025.

[20] C. Andukuri, J.-P. Franken, T. Gerstenberg, and N. D. Goodman,¨ “STaR-GATE: Teaching language models to ask clarifying questions,” arXiv preprint arXiv:2403.19154, 2024.

[21] L. Kuhn, Y. Gal, and S. Farquhar, “CLAM: Selective clarification for ambiguous questions with generative language models,” arXiv preprint arXiv:2212.07769, 2022.

[22] S. Wu, M. Galley, B. Peng, H. Cheng, G. Li, Y. Dou, W. Cai, J. Zou, J. Leskovec, and J. Gao, “CollabLLM: From passive responders to active collaborators,” in International Conference on Machine Learning, 2025.

[23] X. Chen, F. Jiang, Y. Zhang, H. Chen, S. Yan, W. Xie, M. Yang, and S. Huang, “Reasoning while asking: Transforming reasoning large language models from passive solvers to proactive inquirers,” arXiv preprint arXiv:2601.22139, 2026.

[24] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson et al., “Language models (mostly) know what they know,” arXiv preprint arXiv:2207.05221, 2022.

[25] H. Zhu, J. Dai, J. Ji, H. Li, C. Cai, P. Wen, C.-M. Chan, B. Chen, Y. Yang, S. Han, and Y. Guo, “SafeMT: Multi-turn safety for multimodal language models,” arXiv preprint arXiv:2510.12133, 2025.

[26] G. Huang, Q. Peng, G. Xu, Y. Huang, Y. Lu, and Y. Shen, “LLaVAShield: Safeguarding multimodal multi-turn dialogues in vision-language models,” arXiv preprint arXiv:2509.25896, 2025.

[27] Z. Liu, D. Kim, Y. Wan, X. Yuan, Z. Tan, F. Mo, and M. Jiang, “MTMCS-Bench: Evaluating contextual safety of multimodal large language models in multi-turn dialogues,” arXiv preprint arXiv:2601.06757, 2026.

[28] W. Cai, J. Zhao, Y. Yuan, T. Zhang, M. Zhu, H. Tang, and X. Li, “Visual attention reasoning via hierarchical search and self-verification,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 8986–8997.

[29] X. Hu, D. Liu, H. Li, X. Huang, and J. Shao, “VLSBench: Unveiling visual leakage in multimodal safety,” arXiv preprint arXiv:2411.19939, 2024.

[30] T. Gu, Z. Zhou, K. Huang, L. Dandan, Y. Wang, H. Zhao, Y. Yao, Y. Yang, Y. Teng, Y. Qiao et al., “MLLMGuard: A multi-dimensional safety evaluation suite for multimodal large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 7256– 7295, 2024.

[31] Z. Ying, A. Liu, S. Liang, L. Huang, J. Guo, W. Zhou, X. Liu, and D. Tao, “SafeBench: A safety evaluation framework for multimodal large language models,” arXiv preprint arXiv:2410.18927, 2024.

[32] H. Zhu, J. Chen, C. Cai, S. Sun, H. Li, Y. Zhou, C.-M. Chan, P. Wen, L. Li, Y. Guo, and S. Han, “Towards multi-modal multi-turn safety: From agentic interaction to strategic alignment,” arXiv preprint arXiv:2601.04736, 2026.

[33] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in Neural Information Processing Systems, vol. 36, pp. 53 728–53 741, 2023.

[34] Y. Meng, M. Xia, and D. Chen, “SimPO: Simple preference optimization with a reference-free reward,” Advances in Neural Information Processing Systems, vol. 37, pp. 124 198–124 235, 2024.

[35] J. Hong, N. Lee, and J. Thorne, “ORPO: Monolithic preference optimization without reference model,” in Proceedings of the 2024

Conference on Empirical Methods in Natural Language Processing, 2024.

[36] Z. Ling, Y. Fang, X. Li, Z. Huang, M. Lee, R. Memisevic, and H. Su, “Deductive verification of chain-of-thought reasoning,” Advances in Neural Information Processing Systems, vol. 36, pp. 36 407–36 433, 2023.

[37] G. Kim, S. Kim, B. Jeon, J. Park, and J. Kang, “Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 996–1009.

[38] F. Faltings, M. Galley, K. Brantley, B. Peng, W. Cai, Y. Zhang, J. Gao, and W. B. Dolan, “Interactive text generation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 4450–4468.

[39] P. Erbacher and L. Soulier, “CIRCLE: Multi-turn query clarifications with reinforcement learning,” arXiv preprint arXiv:2311.02737, 2023.

[40] T. Zhang, P. Qin, Y. Deng, C. Huang, W. Lei, J. Liu, D. Jin, H. Liang, and T.-S. Chua, “CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 746–10 766.

[41] A. Malinin and M. Gales, “Uncertainty estimation in autoregressive structured prediction,” in International Conference on Learning Representations, 2021.

[42] X. Li, H. Zhou, R. Wang, T. Zhou, M. Cheng, and C.-J. Hsieh, “MOSSBench: Is your multimodal language model oversensitive to safe queries?” arXiv preprint arXiv:2406.17806, 2024.

[43] S. Ghadimi and G. Lan, “Stochastic first- and zeroth-order methods for nonconvex stochastic programming,” SIAM Journal on Optimization, vol. 23, no. 4, pp. 2341–2368, 2013.

[44] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “LLaVA-NeXT: Improved reasoning, OCR, and world knowledge,” Blog post, Jan. 2024. [Online]. Available: https://llava-vl.github.io/blog/2024-01-30-llava-next/

[45] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.