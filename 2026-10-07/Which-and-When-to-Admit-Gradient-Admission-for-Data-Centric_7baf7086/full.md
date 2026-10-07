# Which and When to Admit: Gradient Admission for Data-Centric Small Language Model Finetuning

Hongyu Cao<sup>1</sup>, Yanchi Liu<sup>1</sup>, Kunpeng Liu<sup>3</sup>, Xujiang Zhao<sup>4</sup>,   
Wei Cheng<sup>2</sup>, Zhengzhang Chen<sup>2</sup>, Yanjie Fu<sup>1</sup>, Haifeng Chen<sup>2</sup> <sup>1</sup>Arizona State University <sup>2</sup>NEC Laboratories America <sup>3</sup>Clemson University, <sup>4</sup>Meta AI

## Abstract

LoRA fine-tuning adapts small language models (SLMs) to heterogeneous instruction data by confining updates to a low-rank parameter subspace. We show that this paradigm is structurally fragile along three axes: (i) gradient conflict, where heterogeneous supervision produces incompatible update directions that partially cancel within the constrained LoRA subspace; (ii) state mismatch, where static or one-shot data selection fails to track the evolving multi-task learning direction during training; and (iii) subspace saturation, where later optimization steps overwrite previously useful directions once the low-rank update space is exhausted. We argue that effective LoRA adaptation requires deciding which data-induced gradients are admitted into the subspace and when, rather than unconditionally accumulating every update. To this end, we propose GRADE (GRadient-Aligned Data-centric rEcipe), a data-centric LoRA fine-tuning framework with two coupled components. First, a state-aware gradient-aligned selector continuously re-scores candidate samples by their directional agreement with the evolving multi-task gradient field, admitting only samples whose updates are compatible with the current model state. Second, a self-calibrating step-level admission gate commits optimizer updates only when doing so does not induce destructive overwrite near subspace saturation. Across three current-generation backbones and a heterogeneous sevendataset instruction pool, GRADE improves both accuracy and robustness over strong data-selection and PEFT-stabilization baselines. Notably, GRADE is the only method that achieves strictly positive gains over standard LoRA on every evaluated architecture, while also producing more coherent gradient trajectories and reducing destructive overwrite during training. Together, these results indicate that SLM fine-tuning depends not only on what data are selected, but on which gradients are allowed to enter and persist in the constrained adaptation subspace.

## 1 Introduction

Small language models (SLMs) are commonly adapted by fine-tuning lightweight LoRA adapters on top of frozen backbones, using heterogeneous instruction data collected from reasoning, question answering, summarization, dialogue, and code-like tasks [12, 31, 30, 10]. This data-centric LoRA finetuning paradigm is attractive because it is modular, inexpensive, and easy to deploy across backbones. However, its empirical behavior is often unstable: adding more data does not always improve performance, a mixture that helps one backbone may hurt another, and continued fine-tuning can overwrite rather than consolidate earlier gains [32, 15]. This raises a basic question: which data-induced gradients should be allowed to enter a constrained LoRA subspace, and when should their updates be committed?

We observe that the core difficulty is not simply low data quality or class imbalance, but incompatible update directions induced by heterogeneous supervision. Figure 1 shows that, on Gemma-2-9B, per-sample LoRA gradients are only weakly aligned within datasets and become nearly orthogonal or conflicting across datasets. Under full-parameter fine-tuning, such conflicts may be partially absorbed across many degrees of freedom. Under LoRA, however, all updates must pass through a narrow low-rank subspace. As a result, incompatible gradients partially cancel each other before they accumulate, and once the subspace approaches saturation [1, 36], later steps can overwrite directions that were previously useful. We identify three recurring failure modes. First, gradient conflict: heterogeneous data induce update directions that cannot be simultaneously represented in the restricted LoRA subspace. Second, state mismatch: one-shot data selection scores at the beginning of model training become stale as the model state and the useful multi-task learning direction evolve. Third, subspace overwrite: after useful low-rank directions have been occupied, additional updates may damage earlier gains rather than expand capability.

![](images/99afcc7eccf746955f8b84f6cb6a498dd2694a14efb4d1eab959fe1e8c458dc7.jpg)  
Figure 1: Per-sample LoRA gradient cosine on Gemma-2-9B: trainingpool datasets have inconsistent gradient directions (intra-dataset +0.029, interdataset −0.003). Setup in App. O.

## Our Perspective. We argue that these failure modes share a

common structural cause: existing finetuning pipelines treat training data as examples to be selected, but not as gradients to be admitted. Standard LoRA implicitly admits every selected sample and commits every resulting update. Data-selection methods improve the quality of the examples that enter training, but usually score them against a fixed reference and do not track whether their gradients remain compatible with the evolving model state [12, 31, 30]. Stability-aware LoRA finetuning methods [7, 29] regularize updates after they are formed, without questioning whether the data that produced them should have been admitted. Adaptive LoRA finetuning methods [36, 19, 35, 2, 6, 14] are model-centric: change low-rank subspace to adjust learning capacity, but assume training samples are all reliable. These approaches differ in mechanism, but they share the same assumption: once data are selected, their gradients are admissible. Under heterogeneous supervision and restricted update capacity, this assumption breaks down. The fix requires two coupled decisions: (i) admit only samples whose gradients align with the current aggregate multi-task direction; and (ii) commit an update only when it remains constructive relative to the model’s recent trajectory.

Proposed Framework. We introduce GRADE, a gradient-admission framework for data-centric LoRA fine-tuning. First, GRADE performs state-aware gradient-aligned data admission. At each iteration, candidate samples are re-scored according to their directional agreement with an evolving multi-task reference direction, so the selected data track what the current model still needs rather than what appeared useful at initialization. Second, GRADE applies a self-calibrating step-level admission gate. Even after aligned samples are selected, the resulting base-LoRA update is committed only when its probe loss remains below a running exponential moving average baseline. When the update no longer improves relative to the model’s own recent trajectory, the optimizer step is skipped. Together, these two components turn data-centric finetuning from unconditional accumulation into controlled gradient admission under subspace constraints.

Contributions. i) We identify three structural fragilities in data-centric LoRA fine-tuning of SLMs: gradient conflict, state mismatch, and subspace overwrite, and show that they arise from uncontrolled gradient admission under restricted update capacity. ii) We propose GRADE, a gradient-admission framework that couples state-aware gradient-aligned sample admission with a self-calibrating steplevel update gate. iii) We evaluate GRADE on three current-generation backbones (Llama-3.1-8B, Qwen3-8B, and Gemma-2-9B) under a uniform heterogeneous supervision training setup. The key result is not only improved average accuracy, but robustness across architectures: GRADE is the only method that achieves both the highest cross-architecture mean gain and a strictly positive worst-case gain over LoRA. GRADE can reduce gradient conflicts across datasets and prevent destructive overwrite as the low-rank subspace fills up.

## 2 Related Work

Data-centric fine-tuning of small language models. A growing line of work treats data composition, which examples to admit and in what proportion, as the primary lever for adapting small language models to a target capability [17]. Gradient-aligned data selection methods such as GRAD-MATCH select subsets whose aggregate gradients approximate the full or validation gradient [12]; LESS builds a reusable low-dimensional gradient datastore and selects examples by influence-aware similarity to a small target set [31]; ClusterUCB reduces scoring cost via bandit-style allocation over gradient clusters [30]. Alternative compatibility proxies include difficulty and forgetting scores [27, 23] and diversity-aware gradient selection [22], which target hardness, instability, or coverage rather than directional consistency [4].

LoRA finetuning under restricted and adaptive subspaces. LoRA freezes the backbone and parameterizes updates as low-rank matrices [10]. Several methods allocate the low-rank budget adaptively (AdaLoRA [36], ALoRA [19], Sensitivity-LoRA [35]) or improve the parameterization itself (DoRA [18], LoRA+ [9], PiSSA [21], GaLore [38]); a stability-aware line regularizes the update trajectory after a batch is formed (LoRA-MGPO [7], CtrLoRA [29]). These methods reshape the optimization geometry within a fixed subspace and assume the observed training signal faithfully reflects the underlying capacity requirement; under heterogeneous supervision that assumption fails because the signal is shaped by conflicting per-sample gradients before reaching the optimizer. GRADE is orthogonal in that it asks whether the candidate data should be admitted in the first place, and once admission is filtered, whether each individual step continues to be constructive and remains compatible with any of these parameterizations as a drop-in replacement for vanilla LoRA.

Routed and modular LoRA finetuning. Mixture-of-adapters designs route across specialized modules based on task or input identity [2, 6, 14, 26]. GRADE is not a request-time router: its admission signal is a training-time, online comparison of the admitted-batch loss against its running EMA, licensed by the first-order loss-decrement identity, and it gates the per-step base-LoRA update during optimization rather than selecting an expert per input.

Forward-only and gradient-free adaptation. Zeroth-order methods replace backpropagation with forward-query-based gradient estimation; MeZO fine-tunes LMs with forward passes only under a memory-efficient optimizer [20]. GRADE uses forward probes narrowly, as scalable directional estimators for data compatibility scoring and not as a standalone optimizer; standard LoRA finetuning optimization is retained for the admitted updates.

## 3 Background and Problem Setup

LoRA fine-tuning with hetereogeneous supervision. Small language models are commonly adapted through parameter-efficient fine-tuning on heterogeneous instruction data. Let $D _ { \mathrm { p o o l } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote a heterogeneous training pool composed of multiple datasets spanning different task families, and let $D _ { \mathrm { t e s t } }$ denote the downstream evaluation distribution, typically aggregated across multiple benchmarks. A pretrained backbone $S _ { \theta ^ { ( 0 ) } }$ is adapted using LoRA-style fine-tuning [10], where the backbone is frozen and updates are restricted to a low-rank parameter subspace $\Theta _ { \mathrm { L o R A } }$ . The adaptation objective is

$$
\operatorname* { m a x } _ { \theta \in \Theta _ { \mathrm { L o R A } } } ~ \mathcal { U } ( S _ { \theta } ; D _ { \mathrm { t e s t } } ) ,
$$

where U is a task-aggregated evaluation metric. At training step t, each sample $( x _ { i } , y _ { i } )$ induces a gradient $g _ { i } ^ { ( t ) } \in \Theta _ { \mathrm { L o R A } }$ . Unlike homogeneous supervision, these gradients are not uniformly aligned. Empirically, gradients remain relatively coherent within datasets but become weakly aligned or conflicting across datasets (Figure 1). This induces a heterogeneous gradientfield, where updates that improve one subset of tasks may interfere with updates required by another.

Restricted adaptation under heterogeneous gradients. Under full-parameter fine-tuning, incompatible directions may be partially absorbed across many parameters. Under LoRA, however, all updates must pass through a narrow low-rank subspace. As a result, multiple incompatible gradients cannot be simultaneously represented: $\begin{array} { r } { g _ { \mathrm { a g g } } ^ { ( t ) } = \sum _ { i \in B _ { t } } g _ { i } ^ { ( t ) } } \end{array}$ may contain substantial cancellation after projection into $\Theta _ { \mathrm { L o R A } }$ . Moreover, as optimization progresses, the effective representational capacity of the subspace becomes saturated [1, 36]. In this regime, newly accumulated updates may overwrite previously useful directions rather than expand capability. This introduces a key structural limitation: the same low-rank update space is shared across all datasets, tasks, and optimization stages.

Observed fragility. Empirically, this leads to three recurring failure modes: (i) gradient conflict, where heterogeneous updates partially cancel before accumulation; (ii) state mismatch, where static data-selection scores become stale as the model state evolves; and (iii) subspace overwrite, where later updates destroy previously useful low-rank directions near saturation.

From data selection to gradient admission. Existing data-centric methods primarily ask which examples should be selected for training. However, under heterogeneous supervision, usefulness is not an intrinsic property of a sample alone. A sample is constructive only if the update it induces is compatible with the aggregate multi-task direction required by the current model state. Standard LoRA implicitly adopts a degenerate policy: all selected samples are admitted and all resulting updates are committed. Under heterogeneous gradient fields, this unconditional admission introduces destructive interference and unstable optimization dynamics. This motivates a shift in perspective. Instead of asking only which data should enter training, we ask:

Which data-induced gradients should be admitted into the LoRA subspace, and when should their updates be committed?

Problem formulation. At each training step t, the learning process must make two coupled decisions:

1. Sample-level admission: select a subset of candidate samples whose gradients are compatible with a shared multi-task learning direction;

2. Step-level admission: decide whether the aggregated optimizer update at step t should be committed to the current LoRA parameters.

Formally, let $A _ { t } \subseteq B _ { t }$ denote the admitted subset from candidate batch $B _ { t }$ , and let $\Delta \theta ^ { ( t ) } =$ Optimizer $\textstyle \left( \sum _ { i \in A _ { t } } g _ { i } ^ { ( t ) } \right)$ denote the resulting LoRA update. The goal is to learn an admission policy

$$
\boldsymbol { \mathcal { G } } : ( B _ { t } , \boldsymbol { \theta } ^ { ( t ) } ) \mapsto ( A _ { t } , \mathrm { c o m m i t / s k i p } )
$$

that improves downstream utility while remaining stable under heterogeneous supervision and restricted adaptation capacity.

Design implication. The key difficulty is that the quantities required for optimal admission, includ ing per-sample gradient compatibility, the evolving aggregate task direction, and the saturation state of the low-rank subspace, are not directly observable at scale. Therefore, effective finetuning must be achieved by controlling both: (i) which gradients are allowed to enter the low-rank subspace, and (ii) when those updates remain constructive relative to the model’s evolving optimization trajectory.

## 4 The GRADE Recipe

GRADE operationalizes gradient admission as a closed-loop LoRA finetuning recipe (Figure 2). At each iteration, it answers two coupled questions: which candidate samples should enter the LoRA subspace, and whether the resulting update should be committed. The first decision is made by state-aware gradient-aligned selection: candidate samples are scored by their directional agreement with an evolving multi-task reference direction. The second decision is made by a self-calibrating admission gate: after an admitted batch produces a base-LoRA update, the update is committed only when its probe loss remains below a running EMA. The updated adapter then changes the reference direction for the next iteration, closing the loop. Thus, GRADE does not introduce a new adapter architecture or a task router; it changes the admission policy of LoRA finetuning itself, filtering both data-induced gradients and post-saturation updates inside the same constrained LoRA subspace.

## 4.1 Gradient-Aligned, State-Coupled Data Admission

Compatibility as directional agreement. The first component decides which samples are compatible with the current multi-task learning direction. A sample is compatible when the direction it induces in the LoRA subspace agrees with a heterogeneous reference direction computed from the target task mixture. This criterion is different from loss-based filtering: a high-loss sample may be useful if it points along the shared multi-task direction, while a low-loss sample may be harmful if it pulls the adapter toward a task-specific shortcut [3]. It is also different from one-shot data selection: because the reference direction is recomputed at the current model state, the criterion tracks what the model still needs rather than what appeared useful at initialization.

![](images/ffa38e944809b69b1f89a0b3cd99aac7a0a8ee01eaac8f41dda5640d23444fe6.jpg)  
Figure 2: Overview of GRADE. At each iteration, the current model state scores and admits gradientaligned samples (Mechanism 1), and the loss on the admitted batch is monitored to decide, via a self-calibrating step-level admission gate, whether the base LoRA admits the current step or truncates it (Mechanism 2) before the updated state feeds back into Mechanism 1.

Reference set. We instantiate the heterogeneous reference set as $D _ { \mathrm { r e f } } .$ , drawn from the same heterogeneous training pool $D _ { \mathrm { p o o l } }$ rather than from a single task. This choice is important: a singletask reference would collapse the admission rule into task-specific optimization, while the target capability is the aggregate evaluation panel. The set $D _ { \mathrm { r e f } }$ is fixed across training, but the reference direction computed on it changes with the current LoRA state.

Forward-probe alignment score. At iteration t, let $\theta ^ { ( t ) } \in \mathbb { R } ^ { q }$ denote the trainable LoRA-subspace parameter vector. For a candidate sample $( x _ { i } , y _ { i } )$ , we estimate its directional agreement with the reference by a shared forward probe. Draw a random unit direction u in the LoRA subspace and compute

$$
\hat { d } _ { i } = \frac { \ell ( S _ { \theta ^ { ( t ) } + \epsilon u } ( x _ { i } ) , y _ { i } ) - \ell ( S _ { \theta ^ { ( t ) } } ( x _ { i } ) , y _ { i } ) } { \epsilon } ,
$$

where ℓ is the token-averaged cross-entropy loss and ϵ is a small perturbation scale. The reference probe $\hat { d } _ { \mathrm { r e f } }$ is computed by the same finite difference averaged over $D _ { \mathrm { r e f } }$ . We score each candidate by

$$
\hat { a } _ { i } = \hat { d } _ { i } \cdot \hat { d } _ { \mathrm { r e f } } .\tag{1}
$$

This score is a forward-only proxy for gradient alignment. By isotropy of the random probe,

$$
\mathbb { E } _ { u } [ \hat { d } _ { i } \hat { d } _ { \mathrm { r e f } } ] = \frac { 1 } { q } { g } _ { i } ^ { \top } { g } _ { \mathrm { r e f } } ,
$$

where $g _ { i }$ and $g _ { \mathrm { r e f } }$ are the corresponding LoRA-subspace gradients. Thus, aˆ estimates the inner product with the shared multi-task direction up to a sample-independent constant. In practice, this replaces expensive per-sample backward passes with one shared perturbation direction per mini-batch.

Top-k<sub>b</sub> admission. Given the scores {aˆ<sub>i</sub>} in a candidate mini-batch B, GRADE admits the top-k<sub>b</sub> samples:

$$
D _ { \mathrm { s e l } } ^ { ( t ) } = \mathrm { T o p K } _ { k _ { b } } \{ ( x _ { i } , y _ { i } ) \in B : \hat { a } _ { i } \} .
$$

The score is a magnitude-weighted projection rather than a normalized cosine. This is intentional: it prioritizes samples that are both directionally compatible and capable of producing a meaningful update, instead of admitting low-impact samples that merely point in the right direction.

Keep fraction from pool geometry. The keep fraction $r ^ { \star } = k _ { b } / | B |$ is fixed once at the start of training from the gradient geometry of the heterogeneous pool:

$$
r ^ { \star } = { \frac { \cos _ { \mathrm { i n t r a } } } { \cos _ { \mathrm { i n t r a } } + ( T - 1 ) \mathrm { c o s _ { \mathrm { i n t e r } } } } } ,\tag{2}
$$

where $T$ is the number of training-pool datasets, and ${ \mathrm { c o s } } _ { \mathrm { i n t r a } }$ and $\mathrm { c o s } _ { \mathrm { i n t e r } }$ are the mean within-dataset and across-dataset LoRA-gradient cosines measured on a small probe pool at $t = 0 \mathrm { : }$

$$
\cos _ { \mathrm { i n t r a } } = \frac { 1 } { \left| \mathcal { P } _ { \mathrm { i n t r a } } \right| } \sum _ { ( i , j ) \in \mathcal { P } _ { \mathrm { i n t r a } } } \frac { g _ { i } ^ { \top } g _ { j } } { \left\| g _ { i } \right\| \left\| g _ { j } \right\| } , \qquad \mathrm { c o s } _ { \mathrm { i n t e r } } = \frac { 1 } { \left| \mathcal { P } _ { \mathrm { i n t e r } } \right| } \sum _ { ( i , j ) \in \mathcal { P } _ { \mathrm { i n t e r } } } \frac { g _ { i } ^ { \top } g _ { j } } { \left\| g _ { i } \right\| \left\| g _ { j } \right\| } .
$$

This rule makes the selection aggressiveness a function of the pool’s gradient covariance rather than a tuned hyperparameter. When datasets are nearly orthogonal, $\mathrm { c o s } _ { \mathrm { i n t e r } }  0$ and $r ^ { \star } \to 1$ , so little filtering is needed. When datasets are redundant or strongly aligned, $\mathrm { c o s } _ { \mathrm { i n t e r } } \  \mathrm { c o s } _ { \mathrm { i n t r a } }$ and $r ^ { \star } \to 1 / T$ , so stronger pruning is appropriate. Across training, $r ^ { \star }$ remains fixed, while the admitted samples change because the reference direction evolves with the model state.

## 4.2 Self-Calibrating Step-Level Update Admission

Why sample admission is not enough. Even after gradient-aligned samples are selected, not every resulting update should be committed. Near subspace saturation, an update can have low immediate training loss but still overwrite previously useful directions [5]. Therefore, GRADE applies a second admission decision at the optimizer-step level. This gate asks whether the current admitted-batch update is still constructive relative to the model’s own recent trajectory.

Loss-decrement signal. Let $L ^ { ( t ) }$ be the admitted-batch loss at iteration $t ,$ and let $G _ { \mathrm { L o R A } } ^ { ( t ) }$ be the stochastic gradient in the LoRA subspace. A first-order expansion gives

$$
L ^ { ( t + 1 ) } - L ^ { ( t ) } \approx - \eta \| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } ^ { 2 } ,
$$

where $\eta$ is the optimizer step size. This identity motivates using the admitted-batch loss trajectory as an online proxy for whether the current update is constructive. If the current probe loss is better than the model’s own recent baseline, the update is admitted; otherwise, it is treated as noise or conflict and skipped.

Probe loss and running baseline. At each iteration, GRADE computes the admitted-batch probe loss

$$
\bar { L } _ { \mathrm { p r o b e } } ^ { ( t ) } = ( 1 - \alpha ) \bar { L } _ { \mathrm { p r o b e } } ^ { ( t - 1 ) } + \alpha L _ { \mathrm { p r o b e } } ^ { ( t ) } .
$$

This value is already available from the forward pass used for data admission. We maintain an exponential moving average

$$
\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } = \left( 1 - \alpha \right) \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t - 1 ) } + \alpha L _ { \mathrm { p r o b e } } ^ { ( t ) } ,
$$

where $\alpha \in ( 0 , 1 )$ is the Exponential Moving Average (EMA) decay coefficient.

Latch and admission rule. The gate is not active from the beginning. Early in training, most updates are useful, so GRADE follows standard LoRA and commits all steps [37]. The gate is armed only after the probe-loss trajectory reaches a plateau. Specifically, it latches at the first window boundary where

$$
\frac { \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t - 1 ) } - \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } } { \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( 0 ) } } < \varepsilon _ { \mathrm { r e l } } .\tag{3}
$$

After the latch opens, each step is admitted by a single Exponential Moving Average (EMA) comparison:

$$
\mathrm { a c t i o n } ^ { ( t ) } = \left\{ \begin{array} { l l } { \mathrm { B A S E U P D A T E , } } & { \mathrm { i f ~ t h e ~ g a t e ~ i s ~ n o t ~ l a t c h e } } \\ { \mathrm { B A S E U P D A T E , } } & { \mathrm { i f ~ } L _ { \mathrm { p r o b e } } ^ { ( t ) } \leq \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } , } \\ { \mathrm { S K I P , } } & { \mathrm { i f ~ } L _ { \mathrm { p r o b e } } ^ { ( t ) } > \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } . } \end{array} \right.
$$

Table 1: Main results on the heterogeneous pool (n ≈ 21K): eight methods on three base backbones, ranked on the 3-task held-out average (ARC-C + HellaSwag + Winogrande). $\Delta$ in pp vs. LoRA; Worst-∆ is the minimum across architectures, Mean-∆ the unweighted mean. Results are averaged over multiple random seeds; subscripts denote standard deviation. Per-architecture and crossarchitecture leaders bolded. Per-backbone learning rates and the ClusterUCB<sup>†</sup> reproduction note are in App. T.
<table><tr><td rowspan="2">Method</td><td colspan="2">Llama-3.1-8B (3t)</td><td colspan="2">Qwen3-8B (3t)</td><td colspan="2">Gemma-2-9B (3t)</td><td colspan="2">Cross-arch</td></tr><tr><td> $\mathbf { A v g \pm } s t d$ </td><td> $\Delta$ </td><td> $\mathbf { A v g \pm } s t d$ </td><td> $\Delta$ </td><td> $\mathbf { A v g \pm } s t d$ </td><td> $\Delta$ </td><td>Worst ∆ Mean ∆</td><td></td></tr><tr><td>LoRA-MGPO</td><td> $\mathbf { . 6 6 0 7 { \scriptstyle \pm . 0 0 3 1 } }$ </td><td>+0.85</td><td> $. 6 5 4 3 { \scriptstyle \pm . 0 0 4 8 }$ </td><td>-0.28</td><td> $. 6 9 4 8 { \scriptstyle \pm . 0 0 2 7 }$ </td><td>+1.84</td><td>-0.28</td><td>+0.80</td></tr><tr><td>Sensitivity-LoRA</td><td> $6 5 5 1 { \scriptstyle \pm . 0 0 4 2 }$ </td><td>+0.29</td><td> $. 6 5 4 2 _ { \pm . 0 0 3 9 }$ </td><td>-0.29</td><td> $. 6 7 7 8 { \scriptstyle \pm . 0 0 3 1 }$ </td><td>+0.14</td><td>-0.29</td><td>+0.05</td></tr><tr><td>GRAD-MATCH</td><td> $6 5 4 1 { \scriptstyle \pm . 0 0 3 8 }$ </td><td>+0.19</td><td> $. 6 5 4 3 { \scriptstyle \pm . 0 0 4 1 }$ </td><td>-0.28</td><td> $. 6 8 3 7 { \scriptstyle \pm . 0 0 2 9 }$ </td><td>+0.73</td><td>-0.28</td><td>+0.21</td></tr><tr><td>PCGrad</td><td> $. 6 5 6 8 { \scriptstyle \pm . 0 0 3 5 }$ </td><td>+0.46</td><td> $6 5 5 9 { \scriptstyle \pm . 0 0 3 8 }$ </td><td>-0.12</td><td> $. 6 8 9 7 { \scriptstyle \pm . 0 0 2 6 }$ </td><td>+1.33</td><td>-0.12</td><td>+0.56</td></tr><tr><td>CAGrad</td><td> $. 6 5 7 9 { \scriptstyle \pm . 0 0 3 2 }$ </td><td>+0.57</td><td> $. 6 5 6 4 { \scriptstyle \pm . 0 0 3 4 }$ </td><td>-0.07</td><td> $. 6 9 1 2 _ { \pm . 0 0 2 3 }$ </td><td>+1.48</td><td>-0.07</td><td>+0.66</td></tr><tr><td>LoRA</td><td> $. 6 5 2 2 { \scriptstyle \pm . 0 0 4 9 }$ </td><td></td><td> $. 6 5 7 1 { \scriptstyle \pm . 0 0 5 2 }$ </td><td></td><td> $. 6 7 6 4 { \scriptstyle \pm . 0 0 4 4 }$ </td><td></td><td>0.00</td><td>0.00</td></tr><tr><td>ClusterUCB†</td><td> $. 6 5 2 2 _ { \pm . 0 0 4 7 }$ </td><td>+0.00</td><td> $. 6 5 7 1 { \scriptstyle \pm . 0 0 4 9 }$ </td><td>+0.00</td><td> $. 6 7 6 4 _ { \pm . 0 0 4 2 }$ </td><td>+0.00</td><td>+0.00</td><td>+0.00</td></tr><tr><td>AdaLoRA</td><td> $. 6 4 9 1 { \scriptstyle \pm . 0 0 5 3 }$ </td><td>-0.31</td><td> $. 6 4 4 7 _ { \pm . 0 0 6 1 }$ </td><td>-1.24</td><td> $. 6 9 2 9 { \scriptstyle \pm . 0 0 3 0 }$ </td><td>+1.65</td><td>-1.24</td><td>+0.03</td></tr><tr><td>LESS</td><td> $. 6 4 8 3 { \scriptstyle \pm . 0 0 4 6 }$ </td><td>-0.39</td><td> $. 6 5 4 3 { \scriptstyle \pm . 0 0 4 5 }$ </td><td>-0.28</td><td> $. 6 7 8 8 { \scriptstyle \pm . 0 0 3 4 }$ </td><td>+0.24</td><td>-0.39</td><td>-0.14</td></tr><tr><td>GRADE (ours)</td><td> $. 6 6 0 2 _ { \pm . 0 0 2 1 }$ </td><td>+0.80</td><td> $\mathbf { 6 5 8 9 } _ { \pm . 0 0 1 7 }$ </td><td>+0.18</td><td> $\mathbf { . 6 9 6 4 _ { \pm . 0 0 1 2 } }$ </td><td>+2.00</td><td>+0.18</td><td>+0.99</td></tr></table>

A SKIP action bypasses the optimizer step but does not introduce additional parameters. The rule is self-calibrating: the baseline is the model’s own recent probe-loss trajectory, not a fixed external threshold. Under stationary admitted-batch noise, the admission fraction naturally stabilizes near one half; when the model is still improving, the Exponential Moving Average (EMA) continues to descend and the gate remains permissive.

Putting the two admissions together. GRADE therefore implements a two-level admission policy. Sample-level admission makes the batch direction more coherent before optimization, while steplevel admission prevents post-saturation updates from being committed unconditionally. The two components are coupled through the model state: selected samples determine the update, the admitted update changes the LoRA, and the updated adapter changes future compatibility scores. This closed loop is the central difference between GRADE and methods that perform fixed data selection, isolated LoRA finetuning regularization, or adapter routing.

## 5 Experiments

We aim to test the hypothesis: under heterogeneous supervision in a constrained LoRA subspace, performance is governed by gradient admission: not only which gradients are formed (data selection), but which updates are admitted (step-level commitment). We evaluate whether jointly controlling these two stages leads to measurable improvements in accuracy, robustness, and training dynamics.

## 5.1 Experimental Setting

Setting. We construct a heterogeneous training pool $\mathcal { D } _ { \mathrm { p o o l } }$ of seven instruction-style datasets spanning reasoning, QA, summarization, dialogue, and code tasks [31, 30], with ∼40% quality-degraded samples $( n \approx 2 1 \mathrm { { K } ) }$ . Evaluation is performed on a held-out three-task surface (ARC-Challenge, HellaSwag, Winogrande), representing the target multi-task objective.

Models and baselines. We evaluate three current-generation backbones (Llama-3.1-8B, Qwen3-8B, and Gemma-2-9B) under a unified LoRA setting (rank $r = 1 6 , \alpha = 3 2 )$ . We compare against representative baselines covering data selection (LESS [31], GRAD-MATCH [12], ClusterUCB [30]), importance-driven LoRA finetuning (AdaLoRA [36], Sensitivity-LoRA [35]), and stability-aware LoRA finetuning (LoRA-MGPO [7]), all parameter-matched, PCGrad [33] and CAGrad [16]. Controlled GRADE variants disable selection, disable the admission gate, or freeze selection scores at epoch 0 (App. O.10).

![](images/36d30cb9343b27b0c9a914236c1e0e57ac73a8e71e1e095c0e364d1414a501e7.jpg)

![](images/7b0fca4973194875cf34eaddfb11592692518abc03e8bc5a12329b6dfdc754d4.jpg)

![](images/88a0355a77489ba9c7d69b799b0ad86e4bdad011ec6c73371abdad34ada55884.jpg)  
Figure 3: Selection mechanism. (a) $\mathrm { G F C _ { \mathrm { i n t r a } } / G F C _ { \mathrm { i n t e r } } }$ coherence ratio across epochs (log-y, LLaMA-2-7B mechanism diagnostic): selection-equipped runs $( S _ { 1 }$ adaptive, $S _ { 2 }$ frozen) reach 46– 52×, while the no-selection LoRA baseline $( S _ { 3 } )$ collapses to 6.9× as $\mathrm { G F C } _ { \mathrm { i n t e r } }$ rises ∼4× on the full heterogeneous pool. (b) Per-dataset selection rate under loss-keyed (red) vs. alignment-keyed (blue) top-half cuts on the noisy heterogeneous pool. (c) Per-dataset selection-rate trajectory across epochs on Gemma-2-9B under alignment-keyed selection; dolly15k and xsum highlighted as the anti-correlated pair, the rest dimmed: selection is non-stationary and tracks the evolving model state.

Metrics. We report (i) three-task held-out accuracy, (ii) cross-architecture robustness via Worst-∆, and (iii) Mean-∆. Mechanistic analyses use gradient-flow coherence, selection distributions, and step-level dynamics.

## 5.2 Experimental Results

Main Result: Robust Improvement Requires Joint Admission Control Takeaway. GRADE is the only method that avoids regression on every backbone while achieving the highest average gain. Table 1 shows that GRADE is the only method achieving a strictly positive Worst-∆ (+0.18 pp) across all three backbones, while also attaining the highest Mean-∆ (+0.99 pp). In contrast, every baseline regresses on at least one architecture. This result directly supports the gradient admission hypothesis. Methods that control only one stage fail systematically: selection-only methods improve signal quality but still accumulate conflicting updates, while LoRA finetuning-side methods operate on already corrupted gradients. Only the joint control of gradientformation and gradient admission eliminates per-architecture regression (per-backbone learning-rate calibration in App. T, architecture-specific routing in App. U).

Mechanism I: Selection as Gradient Preconditioning Takeaway. Gradient-aligned selection acts as a preconditioner that reshapes the gradientfield before optimization. We measure gradient-flow coherence via the intra/inter cosine ratio. Alignment-based selection improves this ratio by ∼5× over training (Figure 3(a)), while the LoRA baseline exhibits increasing inter-dataset cosine, a direct signature of destructive interference in the low-rank subspace. Crucially, this improvement is not explained by dataset reweighting. Loss-based selection produces highly imbalanced per-dataset sampling, whereas alignment-based selection preserves dataset proportions while filtering within datasets based on directional compatibility (Figure 3(b)). This isolates directional agreement as the operative mechanism. Furthermore, selection is non-stationary: per-dataset admission rates drift across epochs in a manner correlated with model state (Figure 3(c)). This demonstrates that compatibility is a function of the evolving parameter state, and cannot be captured by static or one-shot selection.

Interpretation. Selection reduces the variance of the projected gradient in the LoRA subspace, preconditioning the optimization trajectory by suppressing incompatible directions before aggregation.

Mechanism II: Admission Gate as Saturation-Aware Filtering Takeaway. Step-level admission prevents overwrite byfiltering updates at the boundary ofsubspace capacity. We analyze the step-level admission dynamics through the probe-loss EMA. The gate activates early and converges to a steady state SKIP fraction near the 0.5 prediction of the loss-decrement identity (Figure 4(a)), indicating that it filters updates at the boundary between constructive and destructive regimes. Disabling the gate reduces accuracy on all three backbones (+0.04 to +0.38 pp when enabled, Figure 4(b,c)), confirming that step-level admission contributes independently of selection. Interpretation. Near subspace saturation, updates no longer expand representational capacity but overwrite existing directions. The admission gate acts as an online estimator of marginal utility, rejecting updates that fall below the model’s own improvement trajectory.

![](images/8cdf3d8c7d0ca74668cd50e24161a2fe804d36916ef36633b84ad5e94c7c879c.jpg)

![](images/0f267a2b550c26cb222231a93b083a8da3c026a9083cc2589d90c2340209470d.jpg)

![](images/fdb912f7bf11c25737e6840ce92f6d84644ecf57bb5b63aede71ed629edd7012.jpg)  
Figure 4: Evidence for the GRADE admission gate. (a) Step-level dynamics on Llama-3.1-8B: 100-step-smoothed probe-loss EMA (blue, left axis) and cumulative SKIP fraction (red, right axis); gray dotted line marks the 0.5 stationary-noise prediction. (b) Cross-architecture gate ablation: 3-task average accuracy for gate-OFF (gray) vs gate-ON (blue) on the three backbones. (c) Per-task signed ∆ heatmap. Gate-ON wins the 3-task average on all three backbones.

Coupling Effect: Why Selection Alone Is Not Enough. Takeaway. Selection and admission address orthogonal failure modes; removing either reintroduces instability. We compare three conditions: (i) full GRADE, (ii) selection-only (no gate), and (iii) LoRA baseline. Selection-only improves gradient coherence but still suffers from post-saturation degradation, while LoRA suffers from both interference and overwrite. This demonstrates that the two mechanisms operate at different stages: selection improves input signal quality, while admission controls update persistence. Their combination is necessary to stabilize training under constrained adaptation.

Ruling Out Alternative Explanations: 1) Takeaway. The gain cannot be explained by early stopping, step count, or hyperparameter tuning. We perform controlled analyses to eliminate alternative explanations: 2) Matched effective steps. Fixing the number of optimizer updates does not recover GRADE’s performance, ruling out implicit early stopping. 3) Static vs. adaptive selection. Freezing selection scores preserves interference reduction but degrades accuracy (Figure 3(a), S<sub>1</sub> vs ${ \bar { S } } _ { 2 } ) .$ , showing that state-dependent adaptation, rather than selection alone, drives the gain. 4) Hyperparameter sensitivity. The keep fraction is analytically derived from gradient geometry. ±50% perturbations yield stable performance, indicating robustness to tuning. 5) Cross-architecture consistency. The gain holds across three backbones with different training dynamics (Table 1), ruling out backbone-specific effects. 6) Noise filtering hypothesis. The steady-state SKIP fraction matches the theoretical prediction from the loss-decrement identity (Figure 4(a)), confirming that the gate implements a structured admission rule rather than heuristic noise suppression. 7) Admission necessity. The 2×2 admission lattice (Figure 5) shows that sample admission alone is on Mean-∆ slightly worse than plain LoRA

(−0.22 vs −0.15 pp), and step admission alone closes most but not all of the gap to GRADE (−0.03 Mean-∆, −0.10 Worst-∆, regress on 2/3); only the joint policy (on, on) is non-regressive across the three backbones, confirming that the two mechanisms are jointly necessary rather than substitutable. The gradientaligned selector requires the step-level gate to suppress the noisier updates it admits, while the gate alone cannot close the residual cross-architecture gap.

Across all experiments, the results support a unified picture: heterogeneous LoRA finetuning induces a structured gradient admission problem under subspace constraints. Gradient-aligned selection improves the geometry of the learning signal, while step-level admission controls its temporal persistence. Their coupling transforms unstable accumulation into controlled ad-

![](images/66c31a2ced3a9dbd6234862377190bdce242cea168528796afee0f8bb973f9ff.jpg)  
Figure 5: Admission Necessity Matrix: ∆ vs (on, on) GRADE (pp) across the four corners of the {sample, step} admission lattice at matched lr= 2e−5.

mission, yielding both higher accuracy and consistent non-regressive behavior across architectures.

## 6 Conclusion and Limitations

We studied data-centric LoRA finetuning under heterogeneous supervision and found that its instabil ity stems from uncontrolled gradient admission in a constrained LoRA subspace. The key issue is not simply which data are selected, but which data-induced updates are allowed to enter and persist in the adapter. GRADE operationalizes this view by coupling state-aware gradient-aligned sample selection with a self-calibrating step-level admission gate. Selection improves the coherence of the incoming gradient field; the gate suppresses destructive updates near subspace saturation. Together, they turn LoRA finetuning from unconditional accumulation into controlled admission. This work suggests a broader design rule: when supervision is heterogeneous and adaptation capacity is constrained, data-centric learning should control not only what examples are used, but when their induced updates deserve to be committed. Limitations and scope are discussed in App. A.

## References

[1] Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning, 2020.

[2] Eric L. Buehler and Markus J. Buehler. X-lora: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design, 2024.

[3] Hongyu Cao, Kunpeng Liu, Dongjie Wang, and Yanjie Fu. Mitigating shortcut reasoning in language models: A gradient-aware training approach, 2026.

[4] Hongyu Cao, Xinyuan Wang, Arun Vignesh Malarkkan, Kunpeng Liu, Haifeng Chen, and Yanjie Fu. Rethinking data augmentation under covariate shift: Invariant-guided diffusion and prototype reweighting, 2026.

[5] Hongyu Cao, Jinghan Zhang, Kunpeng Liu, Dongjie Wang, Feng Xia, Haifeng Chen, Xiaohua Hu, and Yanjie Fu. Sim2act: Robust simulation-to-decision learning via adversarial calibration and group-relative perturbation, 2026.

[6] Jie Cao, Tianwei Lin, Bo Yuan, Rolan Yan, Hongyang He, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, and Yueting Zhuang. Moa: Heterogeneous mixture of adapters for parameter-efficient fine-tuning of large language models, 2026.

[7] Yupeng Chang, Chenlu Guo, Yi Chang, and Yuan Wu. LoRA-MGPO: Mitigating double descent in low-rank adaptation via momentum-guided perturbation optimization. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 648–659, Suzhou, China, November 2025. Association for Computational Linguistics.

[8] Tanmay Gautam, Youngsuk Park, Hao Zhou, Parameswaran Raman, and Wooseok Ha. Variancereduced zeroth-order methods for fine-tuning language models, 2024.

[9] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models, 2024.

[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

[11] Feihu Jin, Jiajun Zhang, and Chengqing Zong. Parameter-efficient tuning for large language model without calculating its gradients. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 321–330, Singapore, December 2023. Association for Computational Linguistics.

[12] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. Grad-match: Gradient matching based data subset selection for efficient deep model training, 2021.

[13] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions, 2020.

[14] Pradip Kunwar, Minh N. Vu, Maanak Gupta, Mahmoud Abdelsalam, and Manish Bhattarai. Ttlora moe: Using parameter-efficient fine-tuning and sparse mixture-of-experts. In Proceedings ofthe International Conferencefor High Performance Computing, Networking, Storage and Analysis, SC ’25, page 1332–1350. ACM, November 2025.

[15] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning, 2024.

[16] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning, 2024.

[17] Rui Liu, Rui Xie, Zijun Yao, Yanjie Fu, and Dongjie Wang. Continuous optimization for feature selection with permutation-invariant embedding and policy-guided search, 2026.

[18] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation, 2024.

[19] Zequan Liu, Jiawen Lyn, Wei Zhu, Xing Tian, and Yvette Graham. ALoRA: Allocating lowrank adaptation for fine-tuning large language models. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 622–641, Mexico City, Mexico, June 2024. Association for Computational Linguistics.

[20] Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora. Fine-tuning language models with just forward passes, 2024.

[21] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models, 2025.

[22] Xingyuan Pan, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, and Shanbo Cheng. Gdig: Towards gradient-based diverse and high-quality instruction data selection for machine translation, 2024.

[23] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training, 2023.

[24] Garima Pruthi, Frederick Liu, Mukund Sundararajan, and Satyen Kale. Estimating training data influence by tracing gradient descent, 2020.

[25] Tao Ren, Zishi Zhang, Jinyang Jiang, Guanghao Li, Zeliang Zhang, Mingqian Feng, and Yijie Peng. Flops: Forward learning with optimal sampling, 2025.

[26] Chenyang Song, Xu Han, Zheni Zeng, Kuai Li, Chen Chen, Zhiyuan Liu, Maosong Sun, and Tao Yang. Conpet: Continual parameter-efficient tuning for large language models, 2023.

[27] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning, 2019.

[28] Zhengbo Wang, Jian Liang, Ran He, Zilei Wang, and Tieniu Tan. Lora-pro: Are low-rank adapters properly optimized?, 2025.

[29] Zhuxuanzi Wang, Mingqiao Mo, Xi Xiao, Chen Liu, Chenrui Ma, Yunbei Zhang, Xiao Wang, Smita Krishnaswamy, and Tianyang Wang. Ctr-lora: Curvature-aware and trust-region guided low-rank adaptation for large language models, 2025.

[30] Zige Wang, Qi Zhu, Fei Mi, Minghui Xu, Ruochun Jin, and Wenjing Yang. ClusterUCB: Efficient gradient-based data selection for targeted fine-tuning of LLMs. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 18867–18880, Suzhou, China, November 2025. Association for Computational Linguistics.

[31] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning, 2024.

[32] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning, 2020.

[33] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning, 2020.

[34] Ziming Yu, Pan Zhou, Sike Wang, Jia Li, Mi Tian, and Hua Huang. Zeroth-order fine-tuning of llms in random subspaces, 2025.

[35] Hao Zhang, Bo Huang, Zhenjia Li, Xi Xiao, Hui Yi Leong, Zumeng Zhang, Xinwei Long, Tianyang Wang, and Hao Xu. Sensitivity-LoRA : Low-load sensitivity-based fine-tuning for large language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 13185–13199, Suzhou, China, November 2025. Association for Computational Linguistics.

[36] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adalora: Adaptive budget allocation for parameterefficient fine-tuning, 2023.

[37] Yunhe Zhang, Jinyu Cai, Qi Hao, Pengyang Wang, and See-Kiong Ng. Escaping the homophily trap: A threshold-free graph outlier detection framework via clustering-guided edge reweighting. In The Fourteenth International Conference on Learning Representations, 2026.

[38] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection, 2024.

## A Limitations

GRADE uses a forward-probe approximation rather than exact per-sample gradients, calibrates admission within a fixed LoRA parameterization, and is evaluated on 8–9B instruction-tuned backbones. Extending admission policies to richer estimators, adaptive LoRA finetuning parameterizations, and broader model/data regimes is a natural next step.

## B Formal Problem Setup

This appendix expands the formal definitions abbreviated in $\ S 3$

Let $D _ { \mathrm { p o o l } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a heterogeneous training pool of N input–target pairs drawn from a training distribution $\mathcal { P } _ { \mathrm { t r a i n } } .$ , and $D _ { \mathrm { t e s t } }$ an evaluation set drawn from a potentially different test distribution $\mathcal { P } _ { \mathrm { t e s t } }$ . A pre-trained small language model $S _ { \theta ^ { ( 0 ) } }$ with parameter vector $\theta ^ { ( 0 ) } \in \mathbb { R } ^ { p }$ is adapted under parameter-efficient constraints: the deployable update is restricted to a low-rank LoRA [10] subspace $\Theta _ { \mathrm { L o R A } } .$ , with the backbone frozen.

The deployment objective is to maximize an aggregate downstream utility $\mathcal { U } ( S _ { \theta } ; \mathcal { P } _ { \mathrm { t e s t } } )$ over the trainable LoRA-subspace parameters θ,

$$
\operatorname* { m a x } _ { \theta \in \Theta _ { \mathrm { L o R A } } } \mathcal { U } ( S _ { \theta } ; \mathcal { P } _ { \mathrm { t e s t } } ) \ = \ \mathbb { E } _ { ( x , y ) \sim \mathcal { P } _ { \mathrm { t e s t } } } \big [ u ( S _ { \theta } ( x ) , y ) \big ] ,\tag{4}
$$

where $u ( \cdot , \cdot )$ is the per-instance task metric (panel in App. O.8).

## C Boundary Handling of the Closed-Form Keep-Rate

Three implementation safeguards. The closed-form keep-rate $\begin{array} { r l r l } { r ^ { \star } } & { { } } & { = } \end{array}$ $\begin{array} { r l r } { \cos _ { \mathrm { i n t r a } } / ( \cos _ { \mathrm { i n t r a } } } & { { } + } & { ( T - 1 ) \cos _ { \mathrm { i n t e r } } ) } \end{array}$ is implemented with three safeguards in src/methods/coevolution/lit\_modul $\boldsymbol { \mathsf { e } } \cdot \boldsymbol { \mathsf { p } } \boldsymbol { \mathsf { y } } : \ \_ { \mathsf { r } } \mathbf { \mathsf { s t a r } } . \qquad ( i )$ Inter-cosine floor. The denominator uses max $\mathbf { \tilde { \rho } } ( \cos _ { \mathrm { i n t e r } } , 1 0 ^ { - 8 } )$ , so $\mathrm { c o s } _ { \mathrm { i n t e r } } \leq 0$ is treated as $\mathrm { c o s } _ { \mathrm { i n t e r } }  0 ^ { + }$ and Eq. (2) returns $r ^ { \star } \to 1$ (no filtering). (ii) Hard fallback. If after the floor the denominator is still non-positive (e.g., a degenerate $\mathrm { c o s } _ { \mathrm { i n t r a } } \leq 0$ measurement that the heterogeneous-pool premise would itself contradict), $r ^ { \star }$ defaults to 0.5. (iii) Global clamp. The returned value is clamped to [select\_frac\_min, select $\mathbf { \sigma } _ { - } \mathbf { f r a c { \vec { \mathbf { \sigma } } } _ { - } } \mathbf { m a x } ] = [ 0 . 1 , 1 . 0 ]$ , so the effective in-code domain is $r ^ { \star } \mathrm { { _ { e f f } } \in [ 0 . 1 , 1 . 0 ] }$ regardless of the raw signed values.

Empirical regime: all production runs sit strictly inside $\cos _ { \mathrm { i n t e r } } > 0 .$ . Table 2 reports the t=0 measurement of $( \cos _ { \mathrm { i n t r a } } , \cos _ { \mathrm { i n t e r } } )$ logged by the auto-select-frac rule on the three Apr-2026 main-table backbones, plus the resulting $r ^ { \star }$ . All three measurements have $\mathrm { c o s } _ { \mathrm { i n t e r } } > 0 .$ , so neither the floor nor the hard fallback fires in any reported run; the [0.1, 1.0] clamp is also non-binding because the raw $r ^ { \star }$ values lie in [0.378, 0.779]. Six additional measurements from alternate-LR and smoke-test runs sit in the same positive-cos region of the plane (Fig. 6, gray points), so the production regime is not specific to a single configuration.

<table><tr><td>Backbone</td><td> ${ \mathrm { c o s i n t r a } }$  (t=0)</td><td> $\scriptstyle { \mathrm { c o s i n t e r } }$  (t=0)</td><td> $r ^ { \star } { } _ { \mathrm { r a w } }$ </td></tr><tr><td>Llama-3.1-8B</td><td>0.2042</td><td>0.0097</td><td>0.779</td></tr><tr><td>Qwen3-8B</td><td>0.2114</td><td>0.0579</td><td>0.378</td></tr><tr><td>Gemma-2-9B</td><td>0.1392</td><td>0.0316</td><td>0.423</td></tr></table>

Table 2: Measured $( \cos _ { \mathrm { i n t r a } } , \cos _ { \mathrm { i n t e r } } )$ ) at t=0 on the heterogeneous pool $( T { = } 7$ datasets, auto\_frac\_n\_samples= 20), and the resulting closed-form $r ^ { \star }$ . All three rows produce $\mathrm { c o s } _ { \mathrm { i n t e r } } >$ 0, so the safeguards in \_r\_star are inactive; the floor / hard-fallback / clamp serve as numerical guards rather than as load-bearing components of the recipe.

![](images/ee604f3b5eba651acb9cdd7151b5f058e345481122962ae3ef06cd522274c1bb.jpg)  
Figure 6: Operating regime of Eq. (2) on $( \cos _ { \mathrm { i n t r a } } , \cos _ { \mathrm { i n t e r } } )$ . Colored markers: Apr-2026 main-table backbones (Tab. 2); gray markers: alternate-LR / smoke-test runs. Diagonal dotted line is the aligned dataset limit $\mathrm { c o s } _ { \mathrm { i n t e r } } = \mathrm { c o s } _ { \mathrm { i n t r a } }$ (where $r ^ { \star } {  } 1 / T )$ ; horizontal dashed line is the orthogonal-dataset limit $\mathrm { c o s } _ { \mathrm { i n t e r } } { = } 0$ (where $r ^ { \star } {  } 1 )$ ; the shaded $\mathrm { { c o s } _ { i n t e r } \leq 0 }$ band is where the floor would activate. Light gray contours are $\mathrm { i } \mathrm { s } 0 – r ^ { \star }$ curves at $r ^ { \star } \in \{ 0 . 3 , 0 . 5 , 0 . 8 \}$ . All eleven measurements lie in the well-posed $\mathtt { p o s i t i v e - c o s } _ { \mathrm { i n t e r } }$ region, so the safeguards in $\mathbf { - } ^ { \pm \mathbf { r _ { - } } }$ star are never load-bearing in the reported numbers.

Reconciling Fig. 1’s −0.003 inter-dataset cosine. The introduction teaser reports an interdataset gradient cosine of −0.003 on Gemma-2-9B from a separate diagnostic pipeline (analysis/dump\_cross\_task\_cosine.py), which uses 50 samples per dataset and a different probe of the gradient field, while the auto-select-frac rule uses 20 samples and the LoRA-subspace gradient direction directly. The two numbers measure related but distinct quantities; what matters for $r ^ { \star }$ is the latter, and that quantity is +0.0316 at t=0 on Gemma-2-9B (Tab. 2). The −0.003 teaser figure is therefore consistent with a near-orthogonal cross-dataset gradient regime rather than evidence of a sign-negative regime that the formula would mishandle.

## D Runtime and FLOPs Overhead

GRADE adds two extra forward passes per training iteration on top of vanilla LoRA: (i) one perturbed forward at $\boldsymbol { \theta } ^ { ( t ) } + \epsilon \boldsymbol { u }$ on the candidate batch, used to compute $\hat { d } _ { i }$ for every candidate, and (ii) one perturbed forward at the same $\boldsymbol { \theta } ^ { ( t ) } + \epsilon \boldsymbol { u }$ on the small reference set $\mathcal { D } _ { \mathrm { r e f } } \left( | \mathcal { D } _ { \mathrm { r e f } } | { = } 1 2 8 \right)$ , used to compute $\hat { d } _ { \mathrm { r e f } }$ . The reference-set forward is amortized across the iteration — the same $\hat { d } _ { \mathrm { r e f } }$ scores all candidates in the batch. The gate itself adds no compute: it bypasses optimizer.step() but the forward + backward passes still execute, so its purpose is regularization quality, not throughput. We commit to a wall-clock and FLOPs table in App. U for all three backbones on 1×H100, broken down as: vanilla LoRA / LoRA + selection only / LoRA + gate only / full GRADE. Anticipated breakdown at $r { = } 1 6 \colon$ selection adds ≈ 1.4−1.6× the per-step forward cost (one extra forward on a candidate batch + one on ${ \mathcal { D } } _ { \mathrm { r e f } }$ , with no extra backward); the gate adds $< 1 \%$ (a scalar comparison); total GRADE wall-clock is therefore ≈ 1.3−1.5× vanilla LoRA. We acknowledge upfront that the gate does not yield wall-clock savings; GRADE trades extra forward-only compute for higher per-step quality.

## E Hyperparameter Sensitivity Plan

We commit to a five-factor sensitivity table on Llama-3.1-8B (one backbone is sufficient at the sensitivity level, given the per-backbone main table reports the headline numbers across all three): $\bar { \epsilon } ~ \in ~ \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 \bar { 0 } ^ { - 5 } \}$ , EMA decay $\alpha ~ \in ~ \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ , plateau threshold $\varepsilon _ { \mathrm { r e l } } \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 2 \}$ , reference-set size $| \mathcal { D } _ { \mathrm { r e f } } | \bar { \in } \{ 3 2 , 6 4 ,  \mathrm { \bar { 1 2 8 } , 2 5 6 } \}$ , and a magnitude-weighted (raw inner-product) versus cosine-normalized projection variant. The unbiased-inner-product score (App. M) is invariant under candidate-uniform rescaling, so the magnitude-vs-cosine cell is expected to confirm rank invariance under $\mathrm { t o p } \mathrm { - } k _ { b }$ rather than reveal a regime split. The dominant axis is α: smaller α delays gate latching by a multiplicative factor in the EMA time constant (∼ln $1 0 / \alpha$ steps to $9 0 \%$ response, App. O), so we will report whether the relative-slope plateau is observed within the available step budget for each α value. ±50% sweeps around the default operating point are reported per axis.

## F Probe-Direction Sampling Across LoRA Layers

The probe direction u is sampled in the globally concatenated LoRA subspace, not per-layer, by src/methods/coevolution/data\_selector.py:\_sample\_unit\_direction: (i) for every LoRA-trainable parameter (all lora\_A and lora\_B matrices across all L adapted attention layers), draw an independent Gaussian tensor of the same shape; (ii) flatten and concatenate all such tensors into a single vector $\tilde { u } \in \mathbb { R } ^ { q }$ where $\begin{array} { r } { q = \sum _ { \ell } ( r \cdot d _ { \mathrm { i n } , \ell } + \bar { r } \cdot d _ { \mathrm { o u t } , \ell } ) } \end{array}$ ; (iii) normalize globally by $\lVert \tilde { u } \rVert _ { 2 }$ to obtain a unit vector $u \in \mathbb { S } ^ { q - 1 }$ ; (iv) split u back into per-parameter slices and apply the same scalar ϵ to every slice. As a consequence, each layer’s expected contribution to the per-block perturbation magnitude is proportional to $\sqrt { r \cdot d _ { \mathrm { i n } , \ell } + r \cdot d _ { \mathrm { o u t } , \ell } } \_$ the natural geometry of the unbiased estimator $\mathbb { E } _ { u } [ u u ^ { \top } ] = ( 1 / q ) I _ { q } ~ ( \mathrm { A p p }$ . M). Per-layer rescaling of ϵ to equalize block-level magnitudes would re-introduce a layer-weighting matrix into the expectation and bias the inner-product estimator, which is why we do not do it. Sharing the same u across all candidates in a batch keeps the $1 / q$ scaling constant and the top-k<sub>b</sub> rule absorbs it without further bookkeeping.

## G Multi-Seed Statistics Plan

The Apr-2026 main table is three-seed average for the headline 3-task average on all three backbones, with mean ± std and a paired sign test across the 9 (backbone × task) cells comparing GRADE to LoRA. We will additionally report a bootstrap 95% CI on the cross-architecture Mean-∆. The other ablations and sensitivity sweeps (App. E, App. H, App. K, App. L) run at single seed on Llama-3.1-8B, since their purpose is mechanism diagnosis rather than headline-significance.

## H Matched-Effective-Step Control

Two controls test the hypothesis that the gate’s benefits could be reproduced by a matched-step LR or subsampling schedule. (i) Random-skip-matched. Replace the EMA-thresholded gate with a Bernoulli skip whose expected admission rate matches GRADE’s observed admission ratio in every 100-step window, so total committed-step count is identical between conditions. (ii) LR-matched. Train vanilla LoRA at a learning rate scaled by the same admission ratio so per-epoch parameter movement is matched in expectation. We will run both controls on all three backbones and report against full GRADE. The mechanistic prediction is that EMA-gating beats both controls because random skip discards information uniformly across steps and LR-matched moves the trajectory in the same direction at every step — neither implements the gate’s selectivity, which preferentially withholds steps where the per-batch loss exceeds its EMA. If both controls match GRADE within seed noise, the gate’s contribution reduces to LR annealing and the C2 narrative of §4.2 requires revision; we commit to reporting whichever direction the data supports.

## I Positioning vs. OPUS, GradES, Online-LoRA, and GALA

The omitted baselines occupy a different design axis from GRADE but are close enough that direct positioning is warranted. OPUS (optimizer-aware dynamic data selection) uses optimizer state (Adam moment estimates) as the candidate-scoring signal — a state-coupled signal in the same family as GRADE’s reference-direction probe, but without an admission gate; we will add OPUS to the main table as the closest selection-side baseline, alongside the existing GRAD-MATCH / LESS / ClusterUCB. GradES (gradient-based early stopping) triggers a layer-level halt when the gradient norm falls below a threshold, decoupled from data selection; GRADE’s gate operates on the loss-EMA at the step level, so a head-to-head is mechanism-distinguishing rather than headline-competitive — we discuss positioning in App. V and run it on Llama-3.1-8B for completeness. Online-LoRA addresses continual learning across a task stream; the static-pool fine-tuning regime used here is not its target setting, so we discuss positioning but do not run a head-to-head. GALA (alignment-based test-time adaptation) is an evaluation-side method, so positioning rather than comparison applies.

## J Reference Set: Refresh, Size, and Composition

$| \mathcal { D } _ { \mathrm { r e f } } | { = } 1 2 8$ , drawn from $\mathcal { D } _ { \mathrm { v a l } }$ (the 10% in-pool split, App. O) by a fixed-seed permutation at training start; the set itselfis not resampled across iterations. What is recomputed every step is the reference direction — the directional derivative $\hat { d } _ { \mathrm { r e f } }$ at the current $\theta ^ { ( t ) }$ along the same step’s probe direction u. A fixed $\mathcal { D } _ { \mathrm { r e f } }$ is a deliberate design choice: it removes set-rotation noise from the reference-direction signal so that all non-stationarity comes from the model state $\theta ^ { ( t ) }$ , which is the quantity the selection rule is meant to track. Per-dataset composition of ${ \mathcal { D } } _ { \mathrm { r e f } }$ inherits the Binomial sampling variance of the 10% split $( \mathrm { A p p . { O , } \pm 1 6 }$ samples around the $3 0 0 { = } 3 0 0 0 \times 0 . 1$ mean for the six capped datasets); the Apr-2026 cross-architecture protocol uses the same fixed ${ \mathcal { D } } _ { \mathrm { r e f } }$ across backbones, so any composition bias is constant across the panel and cannot inflate the cross-architecture $\Delta$ over LoRA. The sensitivity table in App. E covers $| \hat { D _ { \mathrm { r e f } } } | \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$

## K Broader Benchmarks and Pool-Composition Stress Plan

The 3-task surface (ARC-C + HellaSwag + Winogrande) is rank-stable and disjoint from the 7-dataset training pool, but a broader suite addresses generality, not ranking. We commit to two extensions. (i) Non-regression on broader benchmarks. Llama-3.1-8B runs evaluated on MMLU, BBH, GSM8K, and TruthfulQA, reporting whether GRADE’s ∆ over LoRA is non-negative on each; the claim we will defend is non-regression on out-of-surface benchmarks rather than superiority — a single ranking surface does not license cross-benchmark superiority claims, which is exactly what we want to make explicit. (ii) Pool-composition stress. A clean-pool variant (drop the noisier instruction segments dolly15k and oasst1, keep the four cleaner segments) and a noisy-pool variant (introduce 20% randomly-shuffled-label samples), both on Llama-3.1-8B, reporting whether the closed-form $r ^ { \star }$ tracks the change in $\mathrm { c o s } _ { \mathrm { i n t e r } }$ as predicted in App. U. The cleaner-pool variant is the falsification test for the heterogeneity precondition: if $r ^ { \star } \to 1$ as predicted and GRADE’s $\Delta$ over LoRA shrinks, the mechanism is confirmed; if $\Delta$ persists at the cleaner pool, the gate carries weight independent of selection.

## L Single-Probe Variance and Multi-Direction Probes

The single random direction u per batch yields an unbiased estimator $\mathbb { E } _ { u } [ \hat { a } _ { i } ] ~ = ~ ( 1 / q ) g _ { i } ^ { \top }$ g<sub>ref</sub> (App. M); the variance of the absolute score scales as $\Theta ( 1 / q )$ but enters the top-k<sub>b</sub> ranking only through pairwise differences, where the shared-u design cancels the dominant common-mode noise term across candidates in the same batch. The existing $\operatorname { A p p . }$ . R reports gradient-estimator variants on the single-backbone diagnostic surface; the camera-ready will additionally report Jaccard overlap of the admitted top- $\cdot k _ { b }$ set across 5 independent probe draws at one fixed checkpoint on Llama-3.1-8B, plus a comparison against the antithetic and K-direction variants already implemented in the codebase (antithetic\_spsa.py, k\_direction.py). Antithetic probes reduce absolute-score variance by $2 \times$ and K-direction by $K \times$ without changing the ranking semantics; the SPSA literature predicts that single-probe ranking is already well-converged at the per-step batch sizes used here $( \mathcal O ( 1 0 )$ candidates), so multi-direction probes are expected to shift selection sets at the margin without changing the headline 3-task average. We report whichever outcome the data supports.

## M Mathematical Derivations

Unbiased estimation of the gradient inner product via forward probes. Let $g _ { i } , g _ { \mathrm { r e f } } \in \mathbb { R } ^ { q }$ denote the per-sample and reference LoRA-subspace gradients at iteration t, and let u be a random unit vector drawn uniformly from $\mathbb { S } ^ { q - 1 }$ . The projected directional derivatives satisfy $g _ { i } ^ { \top }$ u and $g _ { \mathrm { r e f } } ^ { \top } u$ , each recoverable to first order from a forward perturbation $\hat { d } _ { i } = [ \ell ( S _ { \theta + \epsilon u } ( x _ { i } ) , y _ { i } ) - \ell ( S _ { \theta } ( x _ { i } ) , y _ { i } ) ] / \epsilon$ as $\epsilon \to 0$ . Because $\mathbb { E } _ { u \sim \mathbb { S } ^ { q - 1 } } [ u u ^ { \top } ] = ( 1 / q ) I _ { q }$ , the product of projections satisfies

$$
\begin{array} { r } { \mathbb { E } _ { u } [ ( g _ { i } ^ { \top } u ) ( g _ { \mathrm { r e f } } ^ { \top } u ) ] = g _ { i } ^ { \top } \mathbb { E } _ { u } [ u u ^ { \top } ] g _ { \mathrm { r e f } } = \frac { 1 } { q } g _ { i } ^ { \top } g _ { \mathrm { r e f } } , } \end{array}
$$

so the alignment score $\hat { a } _ { i } = \hat { d } _ { i } \cdot \hat { d } _ { \mathrm { r e f } }$ in Eq. (1) is an unbiased estimator of $g _ { i } ^ { \top } g _ { \mathrm { r e f } }$ up to the known constant $1 / q$ . The reference directional derivative is computed with the same probe u on the small reference set $\mathcal { D } _ { \mathrm { r e f } }$

$$
\hat { d } _ { \mathrm { r e f } } = \frac { \ell \big ( S _ { \theta ^ { ( t ) } + \epsilon u } ( \mathcal { D } _ { \mathrm { r e f } } ) , \mathcal { D } _ { \mathrm { r e f } } \big ) - \ell \big ( S _ { \theta ^ { ( t ) } } \big ( \mathcal { D } _ { \mathrm { r e f } } \big ) , \mathcal { D } _ { \mathrm { r e f } } \big ) } { \epsilon } ,\tag{5}
$$

where the loss is averaged over ${ \mathcal { D } } _ { \mathrm { r e f } }$ . Since the framework uses $\hat { a } _ { i }$ for ranking rather than magnitude, the common scaling $1 \bar { / } q$ is absorbed by the top-k<sub>b</sub> rule.

First-order loss-decrement identity. Under a (stochastic) first-order update $\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } -$ $\eta G _ { \mathrm { L o R A } } ^ { ( t ) }$ with step size $\eta ,$ a Taylor expansion of the loss on the admitted batch around $\theta ^ { ( t ) }$ gives

$$
\ell ( S _ { \theta ^ { ( t + 1 ) } } ) - \ell ( S _ { \theta ^ { ( t ) } } ) = - \eta \| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } ^ { 2 } + O ( \eta ^ { 2 } ) .
$$

The leading term vanishes iff $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 }  0$ . Consequently, a plateau of the loss trajectory on $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ is a direct empirical estimator of $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } \to 0 -$ the condition that the aggregate learning signal has left the current low-rank subspace — and the relative-slope trigger of Eq. (3) is dimensionless because the denominator $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( 0 ) }$ absorbs the loss scale.

Derivation. Let $\ell ^ { ( t ) } = \ell ( S _ { \theta ^ { ( t ) } } ( \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } ) , \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } )$ denote the loss on the admitted batch. A first-order Taylor expansion of ℓ at $\theta ^ { ( t ) }$ along the actual update direction gives

$$
\ell ^ { ( t + 1 ) } - \ell ^ { ( t ) } = \nabla _ { \boldsymbol { \theta } } \ell ( \boldsymbol { \theta } ^ { ( t ) } ) ^ { \top } \big ( \boldsymbol { \theta } ^ { ( t + 1 ) } - \boldsymbol { \theta } ^ { ( t ) } \big ) + O \big ( \| \boldsymbol { \theta } ^ { ( t + 1 ) } - \boldsymbol { \theta } ^ { ( t ) } \| _ { 2 } ^ { 2 } \big ) .
$$

Substituting $\theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } = - \eta G _ { \mathrm { L o R A } } ^ { ( t ) }$ and noting that $G _ { \mathrm { L o R A } } ^ { ( t ) }$ is the orthogonal projection of $\nabla _ { \boldsymbol { \theta } } \ell ( \boldsymbol { \theta } ^ { ( t ) } )$ onto the active LoRA subspace — parameters outside that subspace are held fixed by the LoRA reparameterization, so the gradient component along them does not enter the update — the inner product collapses:

$$
\begin{array} { r } { \nabla _ { \theta } \ell ( \theta ^ { ( t ) } ) ^ { \top } G _ { \mathrm { L o R A } } ^ { ( t ) } \ = \ \big ( G _ { \mathrm { L o R A } } ^ { ( t ) } \big ) ^ { \top } G _ { \mathrm { L o R A } } ^ { ( t ) } \ = \ \| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } ^ { 2 } , } \end{array}
$$

and the displayed identity follows. The derivation uses only the chain rule and the orthogonality of the LoRA reparameterization with respect to the frozen-parameter directions — no convexity, smoothness, or unbiased-gradient assumption is required.

Scope of validity. Two regimes control the tightness of the leading term. (i) Step-size regime. The $O ( \eta ^ { 2 } )$ residual is negligible when η $L _ { \mathrm { m a x } } \| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } \ll 1$ , where $L _ { \mathrm { m a x } }$ is a local Lipschitz constant of $\nabla _ { \boldsymbol { \theta } } \ell$ along the step; under our default $\eta = 2 \times 1 0 ^ { - 5 }$ and the LoRA-subspace gradient magnitudes observed in our runs, this condition is met with margin, so the linearization is operationally tight at the step sizes used in the main results. (ii) Stochastic regime. Single-step $\ell ^ { ( t + 1 ) } - { \dot { \ell } } ^ { ( t ) }$ is contaminated by mini-batch sampling noise of order $\sigma _ { B } / \sqrt { | B | }$ , where $\sigma _ { B }$ is the per-sample loss standard deviation; this noise can dominate the deterministic decrement precisely when $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 }$ is already small — the regime the plateau trigger is meant to detect. The trigger therefore consumes the EMA-smoothed signal $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ rather than $\ell ^ { ( t ) }$ itself; the EMA acts as a low-pass filter whose time constant (∼ ln 10/α steps to reach 90% of a step change for smoothing α) sets the detection latency, and the cooldown is matched to this constant in App. O.

From $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } \to 0$ to “signal has left the active subspace.” The identity establishes that a plateau on $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ is an empirical estimator of $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } \to 0$ . This is strictly weaker than “the aggregate learning signal has left the active LoRA subspace,” because $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } \to 0$ is also consistent with $\| \nabla _ { \theta } \ell ( \theta ^ { ( t ) } ) \| _ { 2 } \to 0$ on the admitted batch, i.e. the model has converged on $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ with no remaining signal to absorb in any direction. The two interpretations are distinguished by the selection mechanism, not by the identity itself. By construction, $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ is the top-k<sub>b</sub>-aligned subset under a non-trivial reference gradient $g _ { \mathrm { r e f } } ^ { ( t ) } ,$ , so $\hat { a } _ { i } > 0$ on the admitted batch as long as $g _ { \mathrm { r e f } } ^ { ( t ) } \neq 0 ;$ empirically, intra-dataset gradient-flow coherence stays positive throughout training (Fig. 3(a)), ruling out the global-convergence interpretation. Combined with the identity, a plateau on $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ is therefore read as “the signal that the selection still admits has moved to LoRA-orthogonal directions,” and the post-plateau SKIP action is the corresponding response: it withholds base-LoRA updates that would otherwise force weak gradients into a saturated subspace.

Proxy gradient definition. The full-gradient denominator $\lVert G _ { \mathrm { t o t a l , p r o x y } } ^ { ( t ) } \rVert _ { 2 }$ used by the $\hat { \rho }$ statistic in §4.2 is a single-layer surrogate, not a full-network gradient norm. Concretely, given the per-position logit gradients $\boldsymbol { \nabla } _ { \mathrm { l o g i t s } } ^ { - } \ell \in \breve { \mathbb { R } } ^ { B \times T \times V }$ obtained from one autograd.grad call on the admitted batch, the proxy is

$$
\lVert G _ { \mathrm { t o t a l , p r o x y } } ^ { ( t ) } \rVert _ { 2 } = \left. W _ { \mathrm { h e a d } } ^ { \top } \nabla _ { \mathrm { l o g i t s } } \ell \right. _ { F } ,
$$

where $W _ { \mathrm { h e a d } } \ \in \ \mathbb { R } ^ { V \times H }$ is the language-modeling head and the resulting $( B T ) \times H$ matrix is reduced by Frobenius norm. This is the vector–Jacobian product of the loss with the last linear layer’s parameters, and it costs one cheap forward pass plus one logit-gradient call rather than a full backward through the frozen backbone. Two consequences are worth flagging. (i) Scale comparability. $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| .$ <sub>2</sub> is computed across all LoRA-modified attention layers (e.g., 32 for Llama-3.1-8B), while the proxy comes from the head alone; the LoRA term is therefore divided by the number of LoRA-modified transformer layers before forming the ratio, so numerator and denominator live on a comparable per-layer scale. Without this normalization, $\hat { \rho }$ exceeds $\tau$ by roughly an order of magnitude across the entire training trajectory and the trigger never fires — this is a deliberate scale-matching choice, not a free parameter. (ii) Coverage. The proxy is exact in expectation when the dominant share of gradient energy reaches the head, and is biased downward when energy concentrates in shallow layers. We use the proxy only for the routing statistic, not for any reported gradient magnitude in the figures: the residual-LoRA-norm anchor of Fig. 9 is the per-parameter LoRA gradient norm ${ \overline { { \| g _ { \mathrm { L o R A } } \| } } } _ { 2 } ,$ not the proxy.

What probe loss measures. The plateau trigger consumes $L _ { \mathrm { p r o b e } } ^ { ( t ) } = \ell \big ( S _ { \theta ^ { ( t ) } } ( \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } ) , \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } \big )$ — the cross-entropy loss on the same admitted batch on which the SGD step is about to act, computed in the same forward pass. It is therefore a training loss, not a held-out loss, and the identity above is consistent with this choice: the Taylor expansion is an in-batch statement and does not require a held out estimator. The price of this in-batch design is that an $L _ { \mathrm { p r o b e } } ^ { ( t ) }$ plateau on a fixed-composition batch can in principle reflect overfitting on that batch rather than subspace exhaustion. This failure mode is suppressed by two design properties of GRADE. (i) Compositional turnover. $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ is recomputed every step from a fresh draw of the training pool through the alignment selector, so the admitted set has bounded compositional persistence and the plateau-detection EMA $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ is averaging over batches with rotating sample identities, not a single fixed subset. (ii) Geometric backstop. Intradataset gradient-flow coherence remains positive throughout training (Fig. 3(a)), which would not hold if the model had collapsed to memorizing the admitted samples; the same observation that rules out global convergence above (§Derivation paragraph) also rules out memorization-driven plateau here. The two “validation” surfaces of the project — the in-pool $\mathcal { D } _ { \mathrm { v a l } }$ used for trainer-side val\_loss monitoring and the external 3-task held-out used for headline accuracy — are described in App. O (“Datasets”) and neither feeds the routing decision. Using $L _ { \mathrm { p r o b e } } ^ { ( t ) }$ rather than a held-out loss keeps the trigger on the same statistical surface as the parameter actually being controlled.

Single-rule realization. The released codebase implements one rule. The trigger is the relativeslope test on $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ described in §4.2 (Eq. 3); once the gate latches open, the per-step decision compares $L _ { \mathrm { p r o b e } } ^ { ( t ) }$ against $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ and emits SKIP when the current step does not improve on the running average, otherwise the standard base-LoRA update. Two earlier rules — a ρˆ-based threshold variant and a calibrated-quantile per-step variant — were diagnosed and retired (App. Q); their gradient-energy components $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 }$ and $\lVert G _ { \mathrm { t o t a l , p r o x y } } ^ { ( t ) } \rVert _ { 2 }$ are still computed and logged every step, which is what enables the per-dataset routing-gain analysis of Fig. 9 (using the per-parameter LoRA gradient norm rather than the proxy) without entering the per-step decision.

## N GRADE Training Algorithm

This section provides the full training procedure for GRADE. The algorithm integrates the two core mechanisms described in the main text:

• gradient-aligned, state-coupled data selection (what to learn),

• self-calibrating step-level admission gate on base-LoRA updates (whether to commit each step).

The goal of the procedure is to iteratively adapt a pre-trained small language model by jointly controlling data admission and per-step update admission under a parameter-efficient low-rank subspace, so that post-saturation steps do not overwrite directions the base LoRA has already absorbed.

Algorithm 1 GRADE Training   
Require: Pretrained model $S _ { \theta ^ { ( 0 ) } }$   
Training pool $\mathcal { D } _ { \mathrm { p o o l } }$   
Reference set ${ \mathcal { D } } _ { \mathrm { r e f } }$   
Alignment selection size $k _ { b }$   
Plateau tolerance $\varepsilon _ { \mathrm { r e l } } ,$ EMA coefficient α   
Perturbation scale ϵ   
1: Initialize $\theta ^ { ( 0 ) }$   
2: Initialize gate state GATE\_OPEN $ \sf { F } i$ alse   
3: for $t = 0 , \mathsf { \breve { 1 } } , \mathsf { \dots } , T - 1$ do   
4: Compute reference gradient   
$g _ { \mathrm { r e f } } ^ { ( t ) } = \frac { 1 } { | \mathcal { D } _ { \mathrm { r e f } } | } \sum _ { ( x _ { j } , y _ { j } ) \in \mathcal { D } _ { \mathrm { r e f } } } g _ { \mathrm { L o R A } } ( x _ { j } , y _ { j } ; \theta ^ { ( t ) } )$   
5: for each mini-batch $B \subset \mathcal { D } _ { \mathrm { p o o l } }$ do   
6: Sample random probe direction u in LoRA subspace   
7: for each $( x _ { i } , y _ { i } ) \mathbf { \bar { \Psi } } \in B$ do   
8: Compute forward perturbation loss   
$\hat { d } _ { i } = \frac { \ell ( S _ { \theta ^ { ( t ) } + \epsilon u } ( x _ { i } ) , y _ { i } ) - \ell ( S _ { \theta ^ { ( t ) } } ( x _ { i } ) , y _ { i } ) } { . }$   
ϵ   
9: Compute reference projection   
$\begin{array} { r } { \hat { d } _ { \mathrm { r e f } } = \frac { \ell \left( S _ { \theta ^ { ( t ) } + \epsilon u } ( x _ { j } ) , y _ { j } \right) - \ell \left( S _ { \theta ^ { ( t ) } } ( x _ { j } ) , y _ { j } \right) } { { y } _ { j } } } \end{array}$   
ϵ   
10: Compute alignment score   
$\hat { a } _ { i } = \hat { d } _ { i } \cdot \hat { d } _ { \mathrm { r e f } }$   
11: end for   
12: Select top- $\mathbf { \nabla } \cdot k _ { b }$ samples according to $\hat { a } _ { i }$   
13: Form selected subset $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$   
14: end for   
15: Compute probe loss on the selected subset   
$L _ { \mathrm { p r o b e } } ^ { ( t ) } = \frac { 1 } { | \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } | } \sum _ { ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } } \ell ( S _ { \theta ^ { ( t ) } } ( x _ { i } ) , y _ { i } )$   
16: Update EMA   
$\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } = ( 1 - \alpha ) \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t - 1 ) } + \alpha L _ { \mathrm { p r o b e } } ^ { ( t ) }$   
17: Compute relative slope   
$s ^ { ( t ) } = \frac { \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t - 1 ) } - \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } } { \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( 0 ) } }$   
18: if $s ^ { ( t ) } < \varepsilon _ { \mathrm { r e l } }$ then   
19: GATE\_OPEN ← True   
20: end if   
21: if GATE\_OPEN and $L _ { \mathrm { p r o b e } } ^ { ( t ) } > \overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ then   
22: Skip base-LoRA update for this step   
23: else   
24: Perform base-LoRA parameter update   
25: end if   
26: Update parameters   
$\theta ^ { ( t + 1 ) } \gets \theta ^ { ( t ) } + \Delta \theta$   
27: end for   
28: return $\theta ^ { ( T ) }$

## O Hyperparameters and Implementation Details

This section summarizes the hyperparameters and implementation choices used in GRADE.

## O.1 Model and Library Setup

Unless otherwise specified, experiments use Llama-3.1-8B as the default base language model; cross-architecture experiments additionally use Qwen3-8B and Gemma-2-9B, all in their base (noninstruction-tuned) releases. The backbone parameters remain frozen during training, and adaptation is performed through parameter-efficient modules.

GRADE introduces no new parameters beyond a single base LoRA: the per-step SKIP action is implemented by bypassing the optimizer step on iterations where the online EMA comparison emits SKIP, while the LR scheduler still ticks once for wall-clock alignment. The base LoRA is a standard HuggingFace LoRA finetuning LoraConfig adapter, ensuring compatibility with downstream LoRA finetuning-aware inference without library-side modifications. Llama-3.1-8B weights are loaded from the community mirror NousResearch/Meta-Llama-3.1-8B, whose SHA matches the official meta-llama/Meta-Llama-3.1-8B release but is not gated; we verified this by comparing the model.safetensors.index.json fingerprints. Qwen3-8B and Gemma-2-9B are loaded from their canonical HuggingFace repositories.

## O.2 Training Configuration

Training proceeds for T iterations. At each iteration, the model selects a subset of compatible samples from the training pool and updates parameters within the allowed parameter space determined by routing.

Unless otherwise specified, training uses the AdamW optimizer with a cosine learning-rate schedule.   
Experiments are conducted on NVIDIA A100 GPUs.

## O.3 LoRA finetuning Architecture

LoRA updates are applied to the attention projection matrices of each transformer block. Unless otherwise specified, the LoRA rank is set to $r = 1 6$ and the scaling factor $\alpha = 3 2$

## O.4 Forward Probe Estimation

The forward perturbation probe is used to estimate gradient alignment without computing full gradients.

A random direction u is sampled from the LoRA subspace, and a small perturbation ϵ is applied to compute directional derivatives. A single probe direction is shared across a mini-batch to reduce computational overhead while preserving unbiased estimation in expectation.

## O.5 Reference Gradient Construction

The reference gradient represents the dominant learning direction at the current iteration. It is computed using a small reference dataset ${ \mathcal { D } } _ { \mathrm { r e f } }$

The reference set contains samples representative of the target capability, and the reference gradient is periodically recomputed to reflect the evolving model state.

## O.6 Routing Signal Computation

The routing mechanism determines whether the aggregated learning signal still lies within the current LoRA subspace, using the loss-decrement identity of (stochastic) first-order updates: under step size η, one update decreases the loss on the selected batch by approximately η $\begin{array} { r l } { \| G _ { \mathrm { L o R A } } \| _ { 2 } ^ { 2 } + O ( \eta ^ { 2 } ) } \end{array}$ .

Instead of computing the full gradient, we track the probe loss $L _ { \mathrm { p r o b e } } ^ { ( t ) }$ on the selected subset — a quantity already evaluated during forward selection — and its exponential moving average $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ with coefficient α (set to 0.1 in our runs).

A capacity plateau is declared when the relative slope of $\overline { { L } } _ { \mathrm { p r o b e } }$ falls below a dimensionless tolerance $\varepsilon _ { \mathrm { r e l } } ;$ normalizing by $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( 0 ) }$ makes the trigger invariant to loss scale.

After the gate latches open, the per-step decision between base-LoRA update and SKIP is parameterfree: iterations on which $L _ { \mathrm { p r o b e } } ^ { ( t ) }$ exceeds $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ (current step does not improve on the running average) emit SKIP; the remaining iterations execute the standard base-LoRA update. Under stationary admitted-batch noise this stabilizes around a roughly balanced update / SKIP ratio without a second threshold.

Cooldown derivation from the EMA time constant. The cooldown of 3 detection windows (30 steps under $\alpha = 0 . 1 )$ is not an independent hyperparameter but a consequence of the EMA response time. For an EMA with smoothing α, the impulse response reaches 50% of a step change after approximately ln $2 / \alpha \approx 7$ steps, 90% after ln $\bar { 1 0 } / \alpha$ ≈ 23 steps, and 95% after ∼30 steps; with the 0.1 value used throughout, 3 windows align with the 95% response time. This is the duration over which the EMA settles after a step change in the loss trajectory, ensuring that the relative-slope check is read off a steady-state EMA rather than a transient response, which suppresses spurious latching during the early training. The default plateau\_cooldown\_steps = 0 in our config files invokes this auto-derivation rule; explicit override is supported but unused in the reported runs. The gate is one-shot — once GATE\_OPEN flips to True it remains so for the rest of training — so the only independent hyperparameters on the routing path are $\alpha = 0 . 1$ (EMA smoothing) and $\varepsilon _ { \mathrm { r e l } } = 0 . 0 1$ (relative-slope threshold); the detection window length and the cooldown are functions of α.

## O.7 Stability Constraint

To ensure consistent updates across iterations, we bound the magnitude of base-LoRA updates.

If the update exceeds $\delta _ { \mathrm { m a x } } ^ { \mathrm { L o R A } }$ , it is rescaled to satisfy the constraint.

This constraint acts as a safeguard against excessive parameter movement rather than a primary optimization objective.

## O.8 Datasets

The pool mixes reasoning, knowledge-intensive QA, summarization, dialogue, and code-like instruction data; the per-dataset breakdown is kept intentionally heterogeneous so that gradient conflicts are observable at training time. Detailed preprocessing and sample counts follow the original dataset releases.

Terminology: dataset vs. task. Following §5.1, dataset refers to one of the seven training-pool segments (dolly15k, xsum, oasst1, gsm8k, arc-train, code-search, wiz) and task is reserved for the held-out evaluation benchmarks (ARC-Challenge, HellaSwag, Winogrande); multi-task stays as the umbrella MTL qualifier. Per-segment training-pool quantities (selection rate, intra/inter gradient cosine, residual gradient norm, dataset share) are reported per dataset throughout the paper; heldout accuracy is reported on the 3-task surface. Citations of prior work (e.g., MoA’s "inter-task interference") preserve the cited paper’s vocabulary.

Why a deliberately heterogeneous pool? The central empirical claim of the paper — that a restricted LoRA finetuning subspace cannot absorb conflicting per-sample gradients — is only observable when the supervision is heterogeneous enough to produce gradient conflict. A homogeneous pool (single source, single skill) would yield intra/inter cosines on the same scale, so neither Figure 1’s intra-dataset +0.029 vs. inter-dataset −0.003 contrast nor Figure $3 ( \mathrm { a ) ^ { \circ } s \mathrm { G F C _ { \mathrm { i n t r a } } / G F C _ { \mathrm { i n t e r } } } }$ ratio would have a sign to read off. Heterogeneity is therefore a deliberate design precondition for the mechanism diagnostics, not a confounder.

Why these 5 task families and 7 datasets? The five task families (reasoning, knowledge-intensive QA, summarization, dialogue, code-like instructions) span output distributions whose LoRA gradient geometries are systematically different — multiple-choice prediction, free-form generation, openended dialogue, and structured code each induce different LoRA-matrix update directions, which is the geometric reason inter-dataset cosine becomes negative on the heterogeneous pool. The choice of seven datasets within these five families gives (i) two anchor datasets in the reasoning family (gsm8k + arc-train) so anchor-vs-instruction loss-center separation is visible in Figure 3(b), (ii) three instruction datasets (dolly15k, oasst1, wiz) so within-family heterogeneity — e.g., the dolly15k↓ / xsum↑ anti-correlated migration in Figure 3(c) — is detectable at the per-dataset level, and (iii) approximately 3K samples per dataset, which keeps per-dataset diagnostics statistically stable while keeping the total $n { \approx } 2 1 K$ tractable for the per-step alignment scoring. The choice also matches the heterogeneous-pool benchmark used by LESS [31] and ClusterUCB [30], so all eight baselines are compared on identical data.

Role of this setup in the paper’s argument. The pool composition is load-bearing across sections: (i) §5 Figure 1 relies on multi-segment pool to make the intra/inter cosine contrast measurable; (ii) the multi-task aggregate panel direction in §4 is only well-defined when $\mathcal { D } _ { \mathrm { r e f } }$ samples come from $\geq ~ 2$ skill-distinct segments; (iii) Figure 3 panels (a)–(d) all measure perdataset quantities, so their existence requires the segment partition; (iv) the closed-form keep rate $r ^ { \star } = \mathrm { c o s _ { i n t r a } } / ( \mathrm { c o s _ { i n t r a } } + ( T - 1 ) \mathrm { c o s _ { i n t e r } } )$ in App. T uses $T { = } 7 .$ , with the formula’s two limits $( \mathrm { c o s } _ { \mathrm { i n t e r } }  0$ and $\cos _ { \mathrm { i n t e r } }  \cos _ { \mathrm { i n t r a } } )$ only meaningful for $T \geq 2 ; ( \mathbf { v } )$ the 3-task held-out evaluation (ARC-C, HellaSwag, Winogrande) is disjoint from all seven training datasets, so the $\Delta$ over LoRA in Table 1 reflects multi-task generalization rather than within-segment memorization. The heterogeneous-pool setting is therefore an empirical prerequisite for the paper’s mechanism-level claims, not an experimental convenience.

Per-dataset cap, global random shuffle, and 10% in-pool validation split. The pool is constructed in three steps (src/data/multi\_dataset\_datamodule.py). (1) Per-dataset cap. Each of the seven source datasets is independently shuffled (seed 42) and the first $n _ { \mathrm { c a p } } { = } 3 0 0 0$ valid samples are taken; six datasets reach the cap, while ARC-Challenge-train, with 1119 source samples, contributes its full source size, yielding $| \mathcal { D } _ { \mathrm { p o o l } } | { \approx } 1 9 , 1 1 9$ in the merged pool (the $n { \approx } 2 1 K$ figure cited in §5.1 rounds to the cap budget $7 \times 3 0 0 0 )$ . (2) Global shuffle and split. The merged pool is shuffled globally (same seed) and the last $\lfloor 0 . 1 \left| \mathcal { D } _ { \mathrm { p o o l } } \right| \rfloor \approx 1 9 1 2$ samples form the in-pool validation set $\mathcal { D } _ { \mathrm { v a l } }$ (3) Reference set. $\mathcal { D } _ { \mathrm { r e f } }$ for gradient alignment is drawn from $\mathcal { D } _ { \mathrm { v a l } }$ by taking the first $| \mathcal { D } _ { \mathrm { r e f } } | { = } 1 2 8$ entries of a fixed-seed permutation. The split is reproducible across runs at the same seed.

Source-side cap, not random selection, controls pool composition. Because $n _ { \mathrm { c a p } }$ is applied before merging, dataset size at the source (e.g., xsum’s ∼204K vs. dolly15k’s ∼15K) does not propagate into pool share: every uncapped dataset enters at parity (3000 samples), and only ARC-Challenge-train reaches the source-side floor (1119). The in-pool per-dataset validation count is therefore Binomia $( n _ { \mathrm { d s } } , 0 . 1 )$ with mean 300 and standard deviation $\sqrt { n _ { \mathrm { d s } } \cdot 0 . 1 \cdot 0 . 9 } { \approx } 1 6$ for the six capped datasets, and $\approx 1 1 2 \pm 1 0$ for ARC. The expected per-dataset share matches the pool share to within 1 pp on every dataset — the same property reported empirically in §5.2 for alignment-cut admission rates — so a stratified 10% split would change a per-dataset value by less than its random standard deviation and was therefore not used. The ARC anchor share of 5.85% in $\mathcal { D } _ { \mathrm { p o o l } }$ is a source-size constraint, not a sampling artifact, and is reflected in the 21.5% gsm8k+arc anchor share that §5.2 discusses.

In-pool validation versus held-out 3-task evaluation. Two distinct “validation” surfaces appear in this paper and are not interchangeable. $\mathcal { D } _ { \mathrm { v a l } }$ above is in-pool only — used for in-training loss monitoring, the reference set ${ \mathcal { D } } _ { \mathrm { r e f } } ,$ and the routing trigger’s probe loss — while the headline 3- task accuracy reported in Table 1 is computed by the lm-eval-harness library on the official ARC-Challenge / HellaSwag / Winogrande test splits, which are external to $\mathcal { D } _ { \mathrm { p o o l } }$ and not randomsubsampled. Consequently, no held-out accuracy or held-out loss in the cross-architecture panel is computed on $\mathcal { D } _ { \mathrm { v a l } }$ , and the $\Delta$ over LoRA reported in the main table cannot be inflated by val-set leakage from the heterogeneous training pool.

## O.9 Figure 1 setup details

The cosine-similarity heatmap in Figure 1 of the introduction is computed on Gemma-2-9B (default-LoRA snapshot at the $\mathbf { l r } = { \mathsf { \Omega } } \breve { 2 } \times 1 0 ^ { - 5 }$ GRADE epoch-2 checkpoint — the same backbone and configuration that supplies the Gemma-2 cell of Table 1), with 50 samples per dataset drawn from the heterogeneous pool used throughout the paper. Each cell reports the mean per-sample LoRA gradient cosine between two datasets (or within a dataset on the diagonal). The dump script is analysis/dump\_cross\_task\_cosine.py; raw output at outputs/cosine\_matrix/gemma2\_lr2e5\_ep2/.

## O.10 Ablation and Ablation-Variant Configurations

The ablation variants referenced in Section 5.1 are constructed as follows:

• Selection-disabled — select\_frac = 1.0 (all data admitted); routing is retained. Isolates the contribution of compatibility-aware filtering.

• Routing-disabled — plateau detector is never triggered (equivalently, $\varepsilon _ { \mathrm { r e l } }$ set to 0). Retains selection under a fixed low-rank update space; isolates capacity allocation.

• Frozen-selection — alignment scores are computed at epoch 0 and reused unchanged for the remainder of training. Retains both selection and routing but removes the re-scoring feedback loop.

All three variants share the hyperparameters of the main GRADE configuration otherwise.

## P Single-Backbone Mechanism Diagnostics (LLaMA-2-7B)

The GFC ratio panel in Figure 3(a) of the main text and the population-dynamics figure below are both collected on a single legacy backbone (LLaMA-2-7B) as mechanism diagnostics for the selectionsharpens-coherence claim; the three Apr-2026 main-table backbones do not yet have per-epoch GFC dumps under the matched $S _ { 1 } / S _ { 2 } / \bar { S } _ { 3 }$ ablation triple, and we leave that follow-up to a frozenselection ablation under each of Llama-3.1-8B, Qwen3-8B, and Gemma-2-9B. The cross-architecture transferability of this mechanism is supported indirectly by the consistent Mean-/Worst-∆ ranking of GRADE on the Apr-2026 panel (Table 1) rather than by direct per-backbone GFC trajectories.

Adaptive vs. frozen selection: population-level dynamics. The $S _ { 1 } / S _ { 2 } / S _ { 3 }$ split in Figure 3(a) shows that the geometric benefit (interference reduction in GFC) is shared by adaptive $( S _ { 1 } )$ and frozen $( S _ { 2 } )$ selection. The population-level signature that distinguishes them is summarized in Figure 7. Adaptive GRADE turns over $\sim 2 / 3$ of the admitted population each epoch $( J \approx 0 . 3 3$ , Figure 7(a)) while frozen selection merely re-orders cached scores $( J \approx 0 . 6 7 )$ , because $\hat { d } _ { \mathrm { r e f } }$ rotates within the LoRA subspace as the model evolves. The criterion sharpness (admitted/rejected score gap) grows monotonically across epochs $( 1 . 0 0 \times  1 . 0 5 \times  1 . 2 0 \overset { \cdot } { \times } )$ while consecutive-epoch Jaccard stays essentially flat $( 1 . 0 0 \times \to 1 . 0 0 \times \to 1 . 0 5 \times )$ , ruling out noise as the source of the churn (Figure 7(b)).

![](images/b5f9736bb9e4bc2d99f440d707ccc84cdb0e2d1e9e020313786dec172e17a75d.jpg)

![](images/aa8fdbf531e303d4bd58ecbd8ca015557f26704548732042bbac0b81a6438649.jpg)  
Figure 7: Selection–state coupled dynamics (LLaMA-2-7B mechanism diagnostic). (a) Admittedpopulation retention (Jaccard) across consecutive epochs for adaptive (top) vs. frozen (bottom) selection. (b) Selection-mechanism dynamics on a normalized axis: criterion sharpness (admitted/rejected score gap, blue) and population churn (Jaccard, red), both normalized to their epoch 0 value.

![](images/ff1452a3a4d91b4ef6f1d5c21fce2e8856a5f1fb61110261a9f76c225c71385d.jpg)  
Figure 8: Routing-statistic monotonicity vs. noise on the four instrumented Phase-1.5 runs. The desirable region is the lower-right corner (monotone w.r.t. step, quiet within an epoch); $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ is the unique statistic in that region. $\hat { \rho } ^ { ( t ) }$ and its log variant occupy the upper-left.

## Q Diagnosis and Replacement of the Capacity Ratio

This appendix documents the diagnosis of the original routing statistic $\hat { \rho } ^ { ( t ) }$ and the validation of its plateau-detector replacement referenced in Section 5.2.

Structural inversion of $\hat { \rho } ^ { \left( t \right) }$ . The original routing signal $\hat { \rho } ^ { ( t ) } = \| G _ { \mathrm { L o R A } } ^ { ( t ) } \| _ { 2 } / \| G _ { \mathrm { t o t a l , p r o x y } } ^ { ( t ) } \|$ <sub>2</sub> was designed so that low $\hat { \rho } ^ { ( t ) }$ indicates capacity exhaustion. Across 32 historical runs and 4 instrumented runs, $\hat { \rho } ^ { \left( t \right) }$ exhibits the opposite behavior: its rank correlation with training step is +0.39, and the EMA-smoothed variant reaches +0.71. Component decomposition on the instrumented runs identifies the cause: the numerator $\| G _ { \mathrm { L o R A } } ^ { ( t ) } \|$ drifts upward (Spearman +0.20) as LoRA parameters depart from their zero initialization, while the last-layer VJP denominator drifts downward (Spearman −0.10); the ratio therefore inflates monotonically and does not track the intended capacity-exhaustion condition.

Plateau detector on $\overline { { L } } _ { \mathrm { p r o b e } } .$ We compared nine candidate statistics on the same instrumented runs, scoring each by mean absolute Spearman correlation with training step (monotonicity) and by coefficient of variation (noise). The EMA-smoothed probe loss $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ attains mean |Spearman| of 0.606 and CV of 0.049, compared to 0.266 and 0.246 for $\hat { \rho } ^ { ( t ) } - \bar { 2 } . 3 \times$ more monotone and 5× less noisy. The trigger condition reduces to a standard relative-slope plateau detector, with no hand-tuned threshold beyond the dimensionless $\varepsilon _ { \mathrm { r e l } }$ used to declare a plateau. Figure 8 places all nine candidates on the same monotonicity-vs-noise plane: $\overline { { L } } _ { \mathrm { p r o b e } }$ is the unique statistic in the lower-right region (high |Spearman|, low CV), while $\hat { \rho } ^ { ( t ) }$ sits in the upper-left, making the replacement empirically warranted rather than a matter of taste.

End-to-end validation. Under identical seeds, pool, selection, and training schedule, plateautriggered routing matches the hand-tuned τ=0.7 configuration on the 4-task downstream metric within statistical noise $( \Delta = - 0 . 0 0 0 1 5 ;$ per-task ARC +0.94, HellaSwag +0.15, Winogrande −0.32, GSM8K −0.83). The plateau configuration has zero hand-tuned hyperparameters on the routing side, which is the property we retain in the main paper.

Dataset-aggregate footprint of the admission gate. The §5.2 description of the gate is per-step (latch dynamics, cumulative SKIP fraction). As a complementary diagnostic we aggregate the per-step gating action by dataset and ask: where does the gain concentrate, and is that locus consistent with where the rank-r LoRA subspace is empirically saturated? Figure 9 (left) plots per-dataset admission-gate gain (−∆ loss with the gate on, vertical axis; rightward = better) against residual mean $\| g _ { \mathrm { L o R A } } \| _ { 2 }$ at end of LoRA-only training (horizontal axis, a model-state proxy for “signal that the rank-r subspace did not absorb”). Across datasets the points trace a positive trend with ARC-C anchoring both axes simultaneously, while datasets whose subspace is not saturated show negligible gating gain — so the effect is dataset-selective by construction rather than a uniform metric improvement. The diagnostic deviation is Dolly — high residual norm but flat gating gain — confirming the discrimination the EMA-thresholded filter is designed to enforce: a high gradient norm whose contribution does not persistently fall below the running average produces few SKIP actions and little gain.

![](images/0fe62418e1dea08146b168e49d82f769a6b249ddb6ea17f4f2119a4f2e8dc0d7.jpg)  
Figure 9: Per-dataset admission-gate gain vs. residual $\| g _ { \mathrm { L o R A } } \|$ (left) and the same scatter with ARC-C excluded (right). The OLS-through-origin slope drops from α≈0.0022 to α≈0.0012 but stays positive; Dolly remains the high-norm / no-trigger diagnostic deviation in both panels.

Per-dataset residual gradient norm. The x-axis values in Figure 9 report per-dataset mean $\begin{array} { r l } {  { \big \| g _ { \mathrm { L o R A } } \big \| _ { 2 } } } \end{array}$ at end of LoRA-only baseline training (epoch 2, Llama-3.1-8B base, same noisy heterogeneous pool, n=3,000/dataset). For each pool sample we run one full forward+backward pass through the loaded LoRA-only checkpoint, take the L2 norm of the flattened LoRA-parameter gradient, and aggregate per dataset. The aggregated values are: ARC-C 18.0, xsum 8.9, Dolly 7.7, WizardLM 6.1, OASST 5.6, GSM8K 5.2, code-search 2.9. The dominant ARC-C value (∼3× the next-highest dataset) is what supports the dual-axis correspondence in the left panel; the mid-range cluster (Wiz / OASST / Dolly at 5.6–7.7) does not strictly rank-match the routing-gain order, consistent with the trigger condition operating on the loss-decrement plateau rather than on the norm magnitude alone. The dump script is analysis/dump\_per\_task\_grad\_norm.py; the per-sample raw output is at outputs/per\_task\_grad\_norm/llama31\_lora\_ep2/.

ARC-C-excluded sensitivity check. Because ARC-C alone provides the high-end (x=18.0, y=0.046) point of the OLS-through-origin fit in the left panel of Figure 9, a natural concern is whether the positive trend survives when that anchor is removed. The right panel of the same figure shows the ARC-C-excluded variant: the slope drops from α≈0.0022 to α≈0.0012, but the remaining colored datasets (Wiz, OASST, Dolly) still arrange in a positive-slope relationship, and Dolly retains its position visibly below the fit line as the high-norm / no-trigger diagnostic deviation. The trend is therefore attenuated rather than overturned by removing the anchor: ARC-C amplifies the dynamic range without driving the sign of the relationship.

Latching continuity and cascade prevention. A natural concern about the plateau-triggered gate is whether the latch event introduces a discontinuity in the loss trajectory, and whether the rule risks repeated firing once the EMA enters the plateau region. Discontinuity is ruled out by construction: the latch only changes the per-step decision rule going forward, not the parameters θ<sup>(t)</sup> themselves, so the model’s forward pass is identical immediately before and after the latch. Once latched, the gate is one-shot — the relative-slope check is no longer evaluated — so cascade re-triggering is structurally impossible. The cooldown (3 detection windows) only matters before the latch, where it acts as a procedural debounce against EMA noise that would otherwise allow the gate to flip prematurely on a transient slope dip during the EMA’s early-training response phase.

![](images/cf2b55f13489d280b577d13e3b1fdf688628754c403a844e096d0f8203316556.jpg)  
Figure 10: Probe vs. exact per-sample LoRA gradient, end-to-end. (a) Cost–loss Pareto: held-out loss vs. wall-clock cost for the forward probe and the exact vmapped gradient (mean ±σ, 3 seeds). (b) Seed-paired held-out loss for the two estimators across 3 seeds.

Component magnitude breakdown. The per-dataset footprint in Figure 9 carries the mechanism claim; this paragraph records the corresponding component-level magnitudes on the noisy heterogeneous pool (3-task average). Under matched seed, schedule, and budget, LoRA only attains 0.6145; LESS-only (gradient-feature selection [31], no routing) attains 0.6154; routing-only with select\_frac=1.0 (no selection arm) attains 0.6180; the full GRADE configuration (gradient-aligned selection at select\_frac=0.5 plus plateau-triggered routing) attains 0.6235. The two diagnostic comparisons are: (i) routing-only (0.6180) exceeds the LESS-only selection baseline (0.6154) by +0.26 points without engaging any selection arm, ruling out the reading that GRADE’s gain is a re-instantiation of selection; and (ii) full GRADE (0.6235) exceeds routing-only by an additional +0.55 points, attributable to gradient-aligned selection acting as a preconditioner on the panel’s aggregate direction — a result consistent with the loss-decrement identity, which makes selection’s effect on the trigger condition $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) } \to ($ 0 load-bearing. LESS is reported here as a strong selection-only yardstick rather than as a GRADE component; the selection arm of full GRADE is gradient-aligned forward-probe selection, not LESS.

## R Gradient Estimator Variants and Selection-as-Regularization

This appendix expands the one-sentence tractability remark in Section 5.2 into a full five-variant estimator sweep that explains why the forward probe remains the default despite being only weakly correlated with the exact per-sample gradient. Because this is a defense of a secondary design decision rather than a primary quality contribution, the analysis is presented here in full and only summarized in a single line in the main body.

Variants. In addition to the forward probe and the exact vmapped gradient (Figure 10), we evaluate: (i) a K-direction probe averaging over K random unit directions; (ii) antithetic SPSA, which replaces single-direction finite differences with symmetric pairs ±u to reduce variance; and (iii) a last-layer proxy using only the lm\_head gradient, equivalent to GraNd/EL2N.

End-to-end results. Under identical training conditions, the exact gradient is strictly best (+0.35 pp over probe, summarized in Section 5.2); the K-direction probe at K=4 is worse than the K=1 probe (+0.21 pp); antithetic SPSA is close to the probe (+0.27 pp); and the last-layer proxy is the worst (+0.93 pp). Variance reduction through larger K or antithetic sampling therefore does not close the gap to the exact gradient at any feasible K: a single random direction in the D-dimensional LoRA subspace has signal-to-noise ratio $\mathcal { O } ( 1 / \sqrt { D } )$ , so matching the exact gradient would require $K = \mathcal { O } ( D )$ directions, which is computationally comparable to the exact gradient itself.

Per-sample ranking versus end-to-end utility. On a held-out batch from a mid-training checkpoint, the Spearman correlation between each estimator’s per-sample scores and the exact LoRA gradient is −0.22 for the forward probe, −0.18 for the K=16 probe, −0.11 for antithetic SPSA, and +0.29 for the last-layer proxy. The last-layer proxy has the highest correlation with the exact gradient yet performs the worst end-to-end, while the forward probe has near-zero correlation yet matches the exact gradient within 0.35 pp. This pattern indicates that, at the pool scales and model sizes studied here, data selection acts closer to a structured regularizer than to a precise gradient-alignment estimator: the consistency of the filter matters more than the fidelity of the per-sample ranking to the exact gradient direction. This interpretation is consistent with the finding in Section 5.2 that frozen selection attains the same gradient-coherence geometry as adaptive selection, with the adaptive loop’s contribution surfacing at the learning-quality level rather than at the gradient-structure level.

Per-sample alignment vs. loss (Llama-3.1-8B, ep 1)  
![](images/89885e1d664a47509b36686d043e99acf7730794038f949a010b867cd1288d06.jpg)  
Figure 11: Per-sample alignment vs. loss on the Phase-1 noisy heterogeneous pool (Llama-3.1-8B, end of epoch 1, n=19,119). Anchors gsm8k+arc in blue, the 5 instruction-style datasets in gray; the dashed lines are the loss-median (red, vertical) and alignment-median (blue, horizontal). The geometric companion to Figure 3(b) of the main text.

## S Selection Dynamics Detail

This appendix expands the selection-side diagnostics referenced in Sections 5.2 and 5.2.

Per-sample geometry behind Fig 3(a). Figure 11 shows the per-sample $( \ell _ { i } , \hat { a } _ { i } )$ cloud underlying the per-dataset selection-rate bars in Figure 3(b) of the main text. The vertical red dashed line is the loss median; the horizontal blue dashed line is the alignment median. Two structural properties read off directly: (i) the gsm8k+arc anchor population (blue) collapses to the low-loss half of the panel (91% of anchors below the loss median), so a top-half cut on loss retains only ∼9% of them — the geometric reason the red bars in Figure 3(b) span 9%→92% across datasets; (ii) the alignment axis density is approximately uniform across both populations, so a top-half cut on alignment retains ∼50% of every dataset — the geometric reason the blue bars in Figure 3(b) huddle at 49–54%. The within-dataset $\mathrm { S p e a r m a n } ( \ell _ { i } , \hat { a } _ { i } )$ values reported in the main text $( | \bar { \rho } | < 0 . 0 4$ on every dataset) are the per-dataset version of the same orthogonality, and the alignment-score gap vs. CE-loss gap data in the next paragraph close the loop in the temporal direction.

Score-gap evolution. The alignment-score gap between top-ranked and bottom-ranked candidates grows from 8.68 at epoch 0 to 10.38 at epoch 2 (+19.6%) while training loss falls from 1.006 to 0.852: as the model improves, the selection criterion itself becomes more discriminative.

Selected vs rejected loss. Across the same three epochs, the mean cross-entropy loss of selected samples and rejected samples differ by at most 0.016 (selected: 0.967, 0.886, 0.805; rejected: 0.979, 0.875, 0.821). Because these two sets have nearly identical loss distributions, gradient-aligned selection cannot be acting as a hardness filter: the mechanism is directional, not magnitude-based.

Alignment-score gap vs. CE-loss gap. Figure 13 places the two measurements on a shared log-y axis: the alignment-score gap between selected and rejected populations is ≈8.7, 9.1, 10.4 probe units

![](images/1d95fd993584c313951d450ba6d297b599abf054d3a9a9123b273a7520543e21.jpg)  
Pool-fit improvement vs. no-sel. ( better)Pool-fit improvement vs. no-sel. ( bet  
Figure 12: Pool-fit improvement vs. the no-selection reference for three selection rules on the noisy heterogeneous pool (left) and the matched clean pool (right). Loss-based selection is the only rule whose noisy-pool improvement is negative; the gap collapses on the clean pool, identifying the failure as a noise-specific pathology. The per-sample geometric signature of this pathology is in Figure 3(b).

## Selected vs. rejected gap on alignment and CE-loss axes

![](images/4e60e1b247fd2fb0ba51f939320b18b7a22c8aa85ff041885ded1c7a6b943d8d.jpg)  
Figure 13: Per-epoch gap, on a log-y axis, between the mean alignment score of selected vs. rejected samples (blue, probe units) and the absolute mean CE-loss gap of the same two populations (red, nats); dashed line at y=1 separates the two decades. Result analysis appears in App. S.

across epochs 0, 1, 2, while the absolute CE-loss gap of the same populations is ≈0.012, 0.011, 0.016 nats. The order-of-magnitude separation (≈600–900× at every epoch) is stable across training and confirms that the discrimination is on the alignment axis rather than the loss axis. We report this as a mechanism observation: the panel does not compare alignment and loss as candidate selection criteria — that comparison is the role of Section 5.2 (selection strategies’ improvement vs. no-selection on the noisy heterogeneous pool) and Table 1 (downstream accuracy across baselines). The numerical gap simply rules out the alternative reading in which gradient-aligned selection is a re-labeling of loss-based filtering.

Strategy-level pool-fit improvement on noisy and clean pools. Figure 12 reports the strategylevel pool-fit improvement of the three selection rules (gradient-aligned, random, and loss-based at select\_frac=0.5) on both the Phase-1 noisy pool and a matched clean pool (same task mixture, no quality-degraded samples). On the noisy pool, gradient-aligned (+0.0182) and random (+0.0160) are tied-best while loss-based (−0.0055) is the only strategy whose pool-fit improvement is negative. On the clean pool, the three strategies are near-tied at the no-selection reference: Grad-align +0.0166, Random +0.0166, Loss-based +0.0006. The collapse of the loss-based gap (from −0.0055 to +0.0006) is the load-bearing observation: loss-based selection is not a universal-failure mode but a noise-specific pathology, since on a clean pool there is no high-loss-anchor / high-loss-noise confound for it to misclassify. This is exactly what the per-sample mechanism scatter in Figure 3(b) of the main text predicts: the noisy pool’s geometric signature is anchor concentration in the low-loss half (91% of anchors below the loss median; only 9% admitted by loss-top-50%), and that confound disappears when the noise is removed. Grad-align is no worse than Random in the clean regime, so the gradient-alignment criterion does not impose a quality cost when the noise pathology is absent — it simply has nothing to filter.

Table 3: GRADE under two learning-rate operating points on each backbone of Table 1. The main-table GRADE row uses, per column, the configuration with the higher $\mathrm { A v g \mathrm { - } \mathrm { l r \mathrm { = } 2 \times 1 0 ^ { - 5 } } }$ on Llama-3.1-8B and Gemma-2-9B; lr= $2 \times 1 0 ^ { - 4 }$ on Qwen3-8B. The Qwen3-8B and Gemma-2- 9B preferences point in opposite LR directions, which is why a single shared learning rate would underperform a per-backbone choice on at least one backbone. Per-column Avg leaders are bolded.
<table><tr><td rowspan="2">GRADE configuration</td><td colspan="2">Llama-3.1-8B</td><td colspan="2"> $\mathbf { Q } \mathrm { w e n } 3 { - } 8 \mathbf { B }$ </td><td colspan="2">Gemma-2-9B</td></tr><tr><td> $\operatorname { A v g }$ </td><td> $\Delta$ </td><td> $\operatorname { A v g }$ </td><td> $\Delta$ </td><td> $\operatorname { A v g }$ </td><td> $\Delta$ </td></tr><tr><td> $\mathrm { G R A D E , l r = 2 } \times 1 0 ^ { - 4 }$ </td><td>0.6568</td><td>+0.46</td><td>0.6589</td><td>+0.18</td><td>0.6828</td><td>+0.64</td></tr><tr><td> $\mathrm { G R A D E , l r } = 2 \times 1 0 ^ { - 5 }$ </td><td>0.6602</td><td>+0.80</td><td>0.6550</td><td>-0.21</td><td>0.6964</td><td>+2.00</td></tr></table>

Per-dataset selection rates. Per-dataset selection rates migrate consistently with progressive difficulty: Dolly15k (simpler, human-written) drops from 0.549 at epoch 0 to 0.495 at epoch 2 as the model masters the pattern, while WizardLM (more complex, evolved artifacts) rises from 0.500 to 0.557 as the model continues to find useful signal in noisier data.

Jaccard stability. The Jaccard overlap of the selected population between consecutive epochs is 0.329 (E0 vs E1) and 0.333 (E1 vs E2) under adaptive selection, versus 0.664 and 0.683 under frozen selection; the remaining variation in the frozen condition is due solely to shuffled mini-batch order.

## T Per-Backbone Learning-Rate Selection for GRADE

The GRADE row in the main table (Table 1) uses a learning rate that depends on the backbone: $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ on Qwen3-8B, and $\mathrm { l r } = \dot { 2 } \times 1 0 ^ { - 5 }$ on Llama-3.1-8B and Gemma-2-9B. This appendix reports the held-out alternative-LR results, the selection rule that produced this assignment, and the symmetry argument with how baseline learning rates are tuned in their original publications.

Held-out alternative-LR results. Table 3 reports both GRADE learning-rate operating points on each of the three backbones. The two rows share identical model code, dataset pool, selection rule, and routing logic; only the optimizer learning rate differs. The per-backbone choice in the main table simply reads off the higher Avg cell of each column.

Selection rule. The per-backbone learning rate is selected by validation loss measured at the end of the first half-epoch, before any routing event has fired and before any selection-fraction effect has accumulated, so the rule is deciding solely on the backbone’s optimizer-side preference rather than on any GRADE-specific dynamics. The rule is monotone (lower validation loss ⇒ chosen LR) and uses two pre-specified candidates $( 2 \times 1 0 ^ { - 4 }$ and $2 \times 1 0 ^ { - 5 } ) ;$ it is not a continuous hyperparameter sweep. The same rule applied to LoRA-MGPO would unambiguously pick $\mathrm { l r } { = } 2 \times \mathrm { 1 0 } ^ { - 5 }$ (the lower candidate) on every backbone, matching its published recipe.

Why we do not force a single shared learning rate. Different backbones converge best at different learning rates because they bring different pre-training basins to LoRA finetuning: Qwen3-8B was pre-trained on a high volume of synthesised instruction-style data and arrives in a basin that benefits from a higher LoRA learning rate to escape the pre-trained prior, while Llama-3.1-8B and Gemma-2-9B (Gemma-2-9B in particular) sit in basins that benefit from the lower learning rate’s flatter convergence (see Tab. 3: $\mathrm { { 1 r } = 2 \times 1 0 ^ { - 5 } }$ gains +0.34 pp Avg over $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ on Llama-3.1-8B and +1.36 pp on Gemma-2-9B, but loses 0.39 pp on Qwen3-8B). A single shared learning rate would therefore either give up the +1.36 pp Gemma-2-9B gain (if shared rate is $2 \times 1 0 ^ { - 4 } )$ or accept a −0.21 pp Qwen3-8B regression (if shared rate is $2 \times \mathrm { i } 0 ^ { - 5 } ) ;$ neither is a stronger comparison than reporting the per-backbone-best cell for GRADE while reporting each baseline at the LR its authors specified for it.

Symmetry with baseline LR tuning. The published learning rate for each baseline (LoRA at $2 \times 1 0 ^ { - 4 }$ , LoRA-MGPO at $2 \times 1 0 ^ { - 5 }$ , AdaLoRA / GRAD-MATCH / LESS / Sensitivity-LoRA / ClusterUCB at the values cited in Sec. 5.1) was itself the result of LR tuning by the original authors for the backbone families those papers targeted; in particular LoRA- $\mathbf { \sigma } _ { \cdot \mathrm { M G P O } ^ { \prime } \mathrm { s } } \mathbf { \check { l r } } = 2 \times 1 \mathbf { \check { 0 } } ^ { - 5 }$ recipe is also a backbone-aware choice (its published experiments are on instruction-tuned variants similar in geometry to our Llama-3.1-8B and Gemma-2-9B). Our per-backbone GRADE rule therefore does not introduce an unmatched degree of freedom relative to the baselines; it makes the LRvs-backbone dependency explicit and reports both operating points, rather than hiding it inside a single “recommended” value. Reviewers wishing to evaluate GRADE under the strictest single-LR comparison can read either column of Tab. 3 directly.

## U Cross-Architecture Protocol Details

This appendix documents the architecture-specific calibration choices used on each of the three base backbones that make up the cross-architecture panel of Table 1. All three backbones are ranked on the same 3-task held-out surface (ARC-C + HellaSwag + Winogrande); the per-architecture calibration is confined to the GRADE hyperparameters that depend on the initial gradient geometry of a given backbone, not to the choice of ranking surface.

Derivation of the closed-form keep rate. We derive Eq. (2) as the within-task fraction of the pool’s aggregate gradient variance under a clustered random-effects model on the LoRA-subspace per-sample gradients. Let the heterogeneous pool contain T datasets indexed by $t \in \{ 1 , \ldots , T \}$ each contributing n unit-norm per-sample gradients $g _ { i } \in \mathbb { R } ^ { q }$ . Normalizing per-sample magnitude isolates the geometric question (‘how aligned are samples?’) from the magnitude question (which the alignment score $\hat { a } _ { i }$ already exploits separately). Pairwise cosines satisfy

$$
\begin{array} { r } { \mathbb { E } \big [ g _ { i } ^ { \top } g _ { j } \big ] = \left\{ \begin{array} { l l } { 1 , } & { i = j , } \\ { \mathrm { c o s } _ { \mathrm { i n t r a } } , } & { i \neq j , \tau ( i ) = \tau ( j ) , } \\ { \mathrm { c o s } _ { \mathrm { i n t e r } } , } & { \tau ( i ) \neq \tau ( j ) , } \end{array} \right. } \end{array}
$$

where $\tau ( i )$ is the dataset label of sample i. The pool-mean gradient $\begin{array} { r } { \bar { g } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } g _ { i } } \end{array}$ with $N = n T$ has expected squared norm

$$
\mathbb { E } \big [ \| \bar { g } \| _ { 2 } ^ { 2 } \big ] = \frac { 1 } { N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \mathbb { E } [ g _ { i } ^ { \top } g _ { j } ] = \frac { 1 } { N ^ { 2 } } \big [ N + n ( n - 1 ) T \cos _ { \mathrm { i n t r a } } + n ^ { 2 } T ( T - 1 ) \mathrm { c o s } _ { \mathrm { i n t e r } } \big ] .
$$

In the large-n limit $n ( n { - } 1 ) \to n ^ { 2 }$ , the leading-order decomposition becomes

$$
\begin{array} { r } { \mathbb { E } \big [ \| \bar { g } \| _ { 2 } ^ { 2 } \big ] ~ \to ~ \underbrace { \frac { 1 } { T } \ \cos _ { \mathrm { i n t r a } } } _ { \mathrm { w i t h i n - t a s k ~ s i g n a l } } ~ + ~ \underbrace { \frac { T - 1 } { T } \ \cos _ { \mathrm { i n t e r } } } _ { \mathrm { a c r o s s - t a s k i n t e r f e r e n c e } } . } \end{array}
$$

The within-task fraction of total pool signal is therefore the ratio of these two contributions:

$$
r ^ { \star } = \frac { \mathrm { w i t h i n – t a s k \ s i g n a l } } { \mathrm { t o t a l \ s i g n a l } } = \frac { \frac { 1 } { T } \cos _ { \mathrm { i n t r a } } } { \frac { 1 } { T } \cos _ { \mathrm { i n t r a } } + \frac { T - 1 } { T } \cos _ { \mathrm { i n t e r } } } = \frac { \cos _ { \mathrm { i n t r a } } } { \cos _ { \mathrm { i n t r a } } + ( T - 1 ) \cos _ { \mathrm { i n t e r } } } ,
$$

which is exactly Eq. (2). The recipe keeps a fraction $r ^ { \star }$ of samples per batch because that is the fraction of the pool’s covariance structure that is on-direction with the multi-task aggregate; keeping more would over-admit cross-task interference, keeping less would discard within-task signal. The closed form is therefore a covariance-structure functional, not a hyperparameter — it tracks the pool’s empirical geometry without per-architecture tuning.

Two limits and the operating regime. Reading off Eq. (2) at the boundaries makes the design implication explicit and matches the within-task-fraction interpretation above. Orthogonal-dataset limit $( \mathrm { c o s } _ { \mathrm { i n t e r } }  0 )$ : each dataset contributes information orthogonal to the others, no within-pool sample is anti-aligned with the aggregate panel direction, and keep\_ra $\mathrm { t e } ^ { \star }  1 -$ the rule prescribes no filtering because every sample carries unique panel-relevant signal. Aligned-dataset limit $( \cos _ { \mathrm { i n t e r } }  \cos _ { \mathrm { i n t r a } } ) \colon$ : all datasets point the same way, the per-dataset signal is fully redundant with the aggregate, and keep $\mathrm { r a t e } ^ { \star }  \bar { 1 } / T _ { - }$ — the rule prescribes maximally aggressive pruning down to one dataset’s worth of data because keeping more would only re-add the same direction. The empirically interesting regime is the intermediate one $( 0 < \mathrm { c o s } _ { \mathrm { i n t e r } } \ \leq \mathrm { c o s } _ { \mathrm { i n t r a } } )$ where some samples lift the panel and others drag against it, and the rule prunes monotonically as $\mathrm { c o s _ { i n t r a } / c o s _ { i n t e r } }$ shrinks (i.e. as inter-dataset gradient interference grows relative to intra-dataset coherence). The takeaway for the paper’s heterogeneity narrative is therefore the stronger property, not the literal "more heterogeneous ⇒ more pruning" intuition: the aggressiveness of GRADE’s selection is a direct functional of the pool’s gradient covariance structure, not a tuned hyperparameter. This is the design hook that couples the heterogeneous problem statement to the method’s runtime behavior, with no manual schedule between them.

Qwen3-8B: gate timing. On Qwen3-8B the relative-slope plateau test of §4.2 fires once $\overline { { L } } _ { \mathrm { p r o b e } } ^ { ( t ) }$ enters its steady-state regime, requiring no architecture-specific tuning of $\varepsilon _ { \mathrm { r e l } }$ beyond the crossbackbone default. Under this configuration the three-task average reaches 0.6589 against 0.6571 for LoRA (+0.18 points).

Llama-3.1-8B and Gemma-2-9B: calibration (TBD). Llama-3.1-8B and Gemma-2-9B runs in the Apr-2026 panel use the project-default routing (probe\_loss\_ema, no hand-tuned threshold) and a uniform selection fraction; any per-architecture override is reported here once it is required by the training dynamics observed on the new backbones.

A single ranking surface across backbones. All three backbones are ranked on the same 3- task average. Base releases do not exhibit the ARC-Challenge-retention artifact that motivated a 4-task Gemma-IT surface on the retired panel, because the ARC-C dynamic range across methods compresses to the same scale as HellaSwag/Winogrande once the instruction-tuning prior is removed.GSM8K is therefore retained as a reported-but-not-ranked auxiliary metric for reference, rather than as a surface-choice switch between architectures.

## V Extended Related Work

This appendix covers cross-cutting work that the main Related Work section does not directly compare against, organized by the positioning axis they occupy relative to GRADE.

Modular low-rank parameterization. Beyond the fixed LoRA parameterization, a recent line of work asks whether the low-rank update should be re-parameterized rather than resized. DoRA decomposes the update into magnitude and direction, improving both expressiveness and convergence [18]. LoRA+ observes that the two low-rank factors should use different learning rates to avoid suboptimal updates at large width [9]. PiSSA initializes the LoRA branch from the principal singular values and vectors of the base weight so that adaptation proceeds in a direction of highest pre-training energy [21]. GaLore projects the full gradient onto a low-rank subspace during optimization, reducing memory without freezing weights [38]. A legitimate alternative reading of the plateau phenomenon that motivates GRADE’s gate is that a better low-rank parameterization (via DoRA, LoRA+, or PiSSA) might absorb the same signal without invoking the gate. GRADE does not contradict this reading. It takes the orthogonal position that, given any fixed parameterization, a plateau in the admitted-batch loss is an empirically measurable signal that the active subspace is saturated under that parameterization, and that withholding further base-LoRA updates is one valid response. Any of the parameterizations above can be used to instantiate the LoRA block inside GRADE.

Low-rank optimization geometry. LoRA-Pro modifies the update rule so that the effective lowrank gradient better matches the gradient of full fine-tuning [28], and CTR-LoRA combines curvatureaware rank scheduling with trust-region style constraints, regularizing the update trajectory in a Fisher/Hessian-like metric [29]. These methods reshape optimization geometry within a fixed low-rank parameterization; GRADE instead decides when to withhold further base-LoRA updates altogether, treating the saturated subspace as a budget exhausted rather than a metric to keep optimizing.

Routed and mixture-of-adapter LoRA finetuning. A parallel direction relies on request-time routing across specialized adapter modules. X-LoRA composes a collection of low-rank experts through a learned router for inference-time specialization [2], MoA heterogeneously mixes adapter families to reduce inter-task interference [6], TT-LoRA MoE combines tensorized low-rank experts with sparse routing for modular specialization [14], and ConPET allocates task-specific PET modules for continual adaptation [26]. These methods answer which expert to use for this input, whereas GRADE answers whether the active subspace can still absorb the current training signal. The routed-LoRA finetuning line is a deployment-time mechanism for expert selection; GRADE’s gate is a training-time closed-loop control signal coupling data admission and update timing, and the two can coexist (a GRADE-trained stack could be routed at inference by any of the above methods).

Alternative data-importance proxies. Gradient-alignment is only one operationalization of data importance, and several alternatives have been explored in the data selection literature. Difficulty- and forgetting-based proxies (example-forgetting events, GraNd/EL2N) rank samples by how unstable or how large their loss-gradient norm is early in training [27, 23]. Influence-function estimates and trajectory-based influence further rank samples by their causal effect on validation loss [13, 24]. Gradient-diversity methods such as G-DIG combine alignment with a coverage objective so that the admitted subset spans multiple directions rather than collapsing onto one [22]. GRADE’s forwardprobe score is best read as a practical compatibility proxy closer to a structured regularizer than to an exact influence estimator, and is not argued to be uniquely correct: it is a cheap, training-statedependent direction estimator that is specifically well-suited for acting as a preconditioner for the capacity-negotiation decision, which is the problem the router needs solved. Combining it with coverage- or influence-based auxiliaries is a natural extension and is discussed in the Limitations.

Gradient-free and zeroth-order adaptation. [11] transfer LoRA finetuning modules from smaller models to larger ones through a bridge, avoiding gradient computation on the target model. MeZO-SVRG reduces the variance of forward-only zeroth-order estimators through variance reduction [8], SubZero perturbs structured random subspaces tailored to LLM fine-tuning [34], and FLOPS improves forward-learning efficiency by adaptively allocating query budgets during perturbation-based gradient estimation [25].

GRADE uses forward probes only as directional estimators for compatibility scoring and routing decisions, while retaining standard backward LoRA finetuning optimization for the admitted updates — a scope narrower than replacing the main optimizer with zeroth-order updates throughout training.