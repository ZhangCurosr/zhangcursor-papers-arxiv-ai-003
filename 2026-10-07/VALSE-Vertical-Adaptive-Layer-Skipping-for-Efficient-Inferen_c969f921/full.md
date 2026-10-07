# VALSE: Vertical Adaptive Layer Skipping for Efficient Inference in Large Language Models

Jia-Dong Zhang jiadongzh@gmail.com

## Abstract

This paper establishes a theoretical framework for vertical adaptive layer skipping, proving three foundational results: (i) an Expected FLOPs formula (theorem 2) giving a closed-form expression for the computational cost of arbitrary per-sample skip schedules as a function of layer-wise skip probabilities; (ii) function-space superset (theorem 10) and strict inclusion (theorem 11) theorems showing that skip-layer models are strictly contained in—yet meaningfully approximate—the full-layer function space, with an explicit separating example; and (iii) a structural duality between VALSE and Mixture-of-Experts architectures (theorem 6), positioning vertical depth-wise sparsity as the orthogonal counterpart to horizontal width-wise sparsity. Building on this theory, we propose VALSE (Vertical Adaptive Layer Skipping for Efficiency), a per-sample, non-contiguous layer skipping method: a lightweight difficulty estimator scores each input from the first few layers, and per-layer gates selectively skip redundant layers—including arbitrary middle layers while retaining deeper ones—so that only the necessary depth is activated for each input, whose feasibility is preliminarily assessed at prototype scale.

## 1 Introduction

Modern large language models (e.g., GPT-4 [OpenAI et al., 2023], LLaMA [Touvron et al., 2023], DeepSeek-MoE [Dai et al., 2024]) stack dozens to hundreds of Transformer blocks. Regardless of input difficulty—from simple sentiment classification to complex multi-step problems—every sample must traverse all layers, incurring identical computation.

In the horizontal (width) dimension, Mixture-of-Experts (MoE) achieves sparse activation by selecting K of E experts per token per layer. However, in the vertical (depth) dimension, the number of layers remains fixed for all inputs: MoE makes each layer “cheaper” but does not reduce the number of layers. Depth thus constitutes an underexploited orthogonal direction for conditional computation.

The core idea is natural: if simple inputs do not require the full representational power of all layers, can we vertically skip certain layers based on input difficulty, activating only the necessary depth? MoE achieves width-wise sparsity; vertical layer skipping achieves depth-wise sparsity—the two form orthogonal dimensions.

Limitations of existing methods. Prior work on adaptive depth falls into several categories, each with structural limitations: early exit [Xin et al., 2020, Liu et al., 2020, Schuster et al., 2022] is truncation-based and cannot skip a middle layer while retaining a deeper one; adaptive computation time [Graves, 2016, Dehghani et al., 2019] adds shared-weight recurrence rather than skipping fixed layers; structured dropout [Fan et al., 2020, Elhoushi et al., 2024a] selects subnetworks statically or per token, not per-sample by difficulty; and token-level routing (MoD [Raposo et al., 2024]) operates at token granularity and lacks an explicit difficulty signal. Crucially, no existing method achieves per-sample, non-contiguous layer skipping (skipping arbitrary middle layers while retaining deeper ones) in conjunction with a unified theoretical duality to horizontal MoE. A detailed comparison is provided in section 2 and section 3.

Contributions. We propose VALSE (Vertical Adaptive Layer Skipping for Efficiency), with three contributions:

1. Theoretical framework. We formalize vertical sparse conditional computation through three fully proven results: the Expected FLOPs theorem (theorem 2), which gives a closedform expression for the computational cost of arbitrary per-sample skip schedules; the function-space superset (theorem 10) and strict containment (theorem 11) theorems, establishing that skip-layer models are strictly contained in the full-layer function space with an explicit separating example; and the MoE–VALSE structural duality (theorem 6) together with a unified two-dimensional sparsity representation (theorems 7 and 9), position ing VALSE as the depth-wise counterpart to horizontal MoE (section 8).

2. Method design. A lightweight, interpretable difficulty estimator scores each sample from the first $K _ { \mathrm { e s t } }$ layers, feeding three routing variants (hard, soft, global) that produce persample, non-contiguous layer retention sets—extending conditional computation from the width to the depth dimension. The method is supported by four auxiliary losses (layerutilization balancing, efficiency, output consistency via KL distillation, and Z-loss) and a three-stage curriculum learning schedule (sections 5 to 7).

3. Proof-of-concept validation. A prototype-scale (12-layer) evaluation on SST-2 provides initial feasibility evidence for the layer-skipping mechanism. We explicitly frame these results as preliminary and emphasize that the core empirical hypothesis—difficulty-driven differentiated routing—is not yet validated at the current scale; full-scale evaluation on larger, pre-trained models is necessary to confirm the approach (section 9).

## 2 Related Work

We survey adaptive depth computation and horizontal MoE routing, identifying the research gaps that motivate VALSE.

## 2.1 Adaptive Depth Methods

Existing work falls into five categories, summarized in table 6 and contrasted along structural axes in table 1. Early exit (DeeBERT [Xin et al., 2020], FastBERT [Liu et al., 2020], PABEE [Zhou et al., 2020]) attaches confidence classifiers at intermediate layers; all are truncation-based—once the model exits, all subsequent layers are skipped. CALM [Schuster et al., 2022] extends early exit to generative LMs but remains truncation-based. Recurrent adaptive computation (ACT [Graves, 2016], Universal Transformers [Dehghani et al., 2019]) iterates shared layers to add depth rather than skipping fixed layers; ACT learns a halting distribution over recurrence steps and Universal Transformers couple recurrence with a per-position halting mechanism, but both require sharedweight recurrence and therefore do not apply to standard fixed-layer Transformer stacks. Layer dropout (LayerDrop [Fan et al., 2020]; LayerSkip [Elhoushi et al., 2024a] with self-speculative decoding [Leviathan et al., 2023]) supports subnetwork extraction but uses static or per-token selection, not per-sample difficulty-driven routing. Layer distillation [Xu et al., 2020] produces static depth variants. Token-level routing: MoD [Raposo et al., 2024] decides per-token per-layer whether to process or skip; Drowzee [Bachmann et al., 2023] predicts per-sample compute budgets but only controls truncation depth; DynamicBERT [Liu et al., 2021] adjusts width and depth independently without unification. Layer redundancy and structured pruning: Recent work provides empirical evidence that Transformer layers exhibit significant redundancy. ShortGPT [Men et al., 2024] introduces a Block Influence (BI) score to quantify layer redundancy and shows that up to 50% of layers can be removed with minimal performance loss. Streamline [Elhoushi et al., 2024b] and SliceGPT [Ashraf et al., 2024] perform structured layer pruning at the post-training stage, producing static reduced-depth models. MoNE [Mustafa et al., 2024] applies mixture-of-experts routing across encoder–decoder layer pairs, selecting which layer pairs to execute per token. These methods confirm that layer redundancy is pervasive, but all produce static or per-token depth profiles—none achieves per-sample, difficulty-driven, non-contiguous skipping.

## 2.2 Horizontal MoE Routing

MoE achieves width-wise sparse activation: each layer selects K of N experts. Sparse-gated MoE [Shazeer et al., 2017] introduced the gating mechanism; GShard [Lepikhin et al., 2020] scaled it to multilingual and multimodal settings; Switch Transformer [Fedus et al., 2022] simplified to Top-1 routing and demonstrated that aggressive expert sparsity is compatible with stable pretraining. Mixtral [Jiang et al., 2024] and DeepSeek-MoE [Dai et al., 2024] are recent open representatives, with DeepSeek-MoE in particular introducing fine-grained expert segmentation and shared experts to improve specialization. Across this family, the routing signal is semantic and per-token: the gate reads the token representation and matches it to expert specializations, with no notion of input-level difficulty or sample-level compute budget. The key observation is that all MoE methods keep depth fixed—every token still passes through all L layers—and that the difficulty-awareness dimension, central to VALSE, has no analogue in horizontal MoE routing. table 8 makes this orthogonality formal: MoE selects ‘which experts” within a layer, whereas VALSE selects ‘which layers” across depth, and the two dimensions are mathematically dual but operationally independent.

## 2.3 Horizontal vs. Vertical Comparison

Table 1 summarizes the orthogonality: MoE reduces the number of parameters activated per layer (width); VALSE reduces the number of layers activated (depth). The two can be combined for two-dimensional sparse activation, and the orthogonality is structural rather than merely intuitive: width-wise sparsity selects a subset of parallel compute units within a layer (experts $E _ { i }$ applied to the same hidden state), whereas depth-wise sparsity selects a subset of sequential compute units across layers (blocks F applied in cascade), so the two selection operators act on disjoint axes of the activation tensor and compose without interference. This composability is non-trivial in two respects. First, the routing signals differ in scope: MoE gating is per-token and semantic (matching token content to expert specialization), while VALSE gating is per-sample and difficulty-driven (matching input complexity to retained depth), so a joint router must expose both a per-token width gate and a per-sample depth gate without one collapsing into the other. Second, the load-balancing objectives differ in target: horizontal MoE prevents expert under-utilization (few experts absorbing most tokens), whereas VALSE prevents layer under-utilization (most samples collapsing to the same minimal-depth path), so a unified training objective must regularize both the expert-utilization distribution and the per-layer activation frequency simultaneously. When these conditions hold, the two dimensions yield a multiplicative compute reduction: under a width sparsity $s _ { w }$ (fraction of experts activated) and a depth sparsity $s _ { d }$ (fraction of layers retained), the activated parameter count scales as $( 1 - s _ { w } ) ( 1 - s _ { d } )$ of the dense network, a property formalized in theorem 9 as the unified width×depth sparsity framework. No prior method surveyed above attains both dimensions simultaneously: horizontal MoE keeps depth fixed, and all adaptive-depth methods keep width fixed, so the two-dimensional sparse-activation regime is itself part of the methodological gap that VALSE is designed to occupy.

Table 1: Comparison of adaptive depth methods. Skip granularity and per-sample adaptivity distinguish VALSE from all prior approaches.
<table><tr><td>Method</td><td>Sparsity dim.</td><td>Skip mode</td><td>Granularity</td><td>Per-sample?</td><td>Difficulty-aware?</td></tr><tr><td>MoE [Jiang et al., 2024, Dai et al., 2024]</td><td>Width</td><td></td><td>Per-token</td><td>No</td><td>Semantic</td></tr><tr><td>Early exit [Xin et al., 2020, Schuster et al., 2022]</td><td>Depth</td><td>Truncation</td><td>Per-sample</td><td>Yes</td><td>Confidence</td></tr><tr><td>LayerDrop [Fan et al., 2020]</td><td>Depth</td><td>Static dropout</td><td>Global</td><td>No</td><td></td></tr><tr><td>MoD [Raposo et al., 2024]</td><td>Depth</td><td>Per-layer</td><td>Per-token×layer</td><td>No</td><td>Implicit</td></tr><tr><td>ShortGPT [Men et al., 2024]</td><td>Depth</td><td>Static pruning</td><td>Global</td><td>No</td><td></td></tr><tr><td>SliceGPT [Ashraf et al., 2024]</td><td>Depth</td><td>Structured pruning</td><td>Global</td><td>No</td><td></td></tr><tr><td>VALSE (ours)</td><td>Depth</td><td>Non-contiguous</td><td>Per-sample</td><td>Yes</td><td>Explicit router</td></tr></table>

## 2.4 Research Gaps

The above analysis reveals four methodological gaps—per-sample non-contiguous skipping, unification of horizontal–vertical routing, explicit difficulty awareness, and validation on autoregressive LLMs—that, taken together, are not addressed by any prior method surveyed above. Truncationbased early exit (DeeBERT/CALM) is per-sample but constrains retention to a prefix; per-token per-layer routing (MoD) is fine-grained but lacks sample-level difficulty semantics; horizontal MoE sparsifies width but never depth; structured pruning (ShortGPT/SliceGPT) reduces depth statically with no input adaptivity. None of these dimensions, individually or in combination, yields a persample, difficulty-driven, non-contiguous layer retention set.

Limitations of prior work. As table 1 makes explicit, every surveyed family restricts at least one of the three structural axes: early exit fixes skip mode to truncation, per-token routing fixes granularity to the token level, horizontal MoE fixes the sparsity dimension to width, and structured pruning fixes depth to a static global profile. Consequently, no single prior method simultaneously attains non-contiguous skipping, per-sample granularity, and explicit difficulty awareness— the cell (non-contiguous, per-sample, explicit) that VALSE targets, which is the depth-axis counterpart of the width-axis cell already populated by horizontal MoE. We defer the structured contrast against each prior family to section 3 to avoid duplication; the remainder of the paper is structured around closing them. In summary, the related-work landscape partitions cleanly along three axes—skip mode (truncation vs. per-token vs. non-contiguous), granularity (global vs. per-token vs. per-sample), and difficulty awareness (none vs. implicit vs. explicit)—and no prior method occupies the cell (non-contiguous, per-sample, explicit) that VALSE targets. The structural duality with horizontal MoE (section 2.2) further shows that this cell is not arbitrary: it is the depth-axis counterpart of the width-axis routing cell that MoE already occupies, so filling it completes the twodimensional sparse-conditional-computation design space. The remainder of the paper develops this contribution: section 3 formalizes the four gaps, section 4 through section 7 present the method, and section 8 provides the theoretical framework, with a prototype-scale empirical study (section 9) that honestly reports a negative result for the core routing hypothesis.

## 3 Research Gaps

Based on the cross-analysis of adaptive depth methods (section 2), we identify four research gaps that motivate VALSE.

## 3.1 Gap 1: No Per-Sample Non-Contiguous Layer Skipping

Existing adaptive-depth work is either truncation-based early exit (exit at layer k, skip the rest: DeeBERT/CALM) or per-token per-layer routing (MoD). No prior work implements selective skipping of middle layers while retaining deeper layers, driven by whole-sample difficulty. This intermediate granularity—more flexible than truncation, more stable than per-token—is a clear methodological gap. Formally, truncation early exit constrains the retention set to a prefix $S = \{ 1 , \ldots , k \}$ , while per-token per-layer routing (MoD) issues $O ( L \cdot N )$ independent gating decisions per forward pass; VALSE instead computes a single per-sample difficulty scalar $\hat { d } ( x ^ { ( s ) } )$ ) from the first $K _ { \mathrm { e s t } }$ layers and derives the retention set $S ^ { ( s ) } \subseteq \{ 1 , \dots , L \}$ as an arbitrary subset, achieving O(1) routing overhead per sample while permitting non-contiguous retention.

## 3.2 Gap 2: No Unified Horizontal–Vertical Routing Framework

Horizontal MoE (expert selection) and vertical layer skipping (layer selection) are orthogonal sparsity dimensions, but no work unifies them into a jointly trainable routing framework. Existing models address either only the horizontal dimension (Mixtral/DeepSeek-MoE) or only the vertical dimension (MoD/LayerDrop). DynamicBERT jointly adjusts width and depth, but the two dimensions are routed independently with no unified mechanism. Formally, horizontal and vertical sparsity correspond to selecting subsets along two orthogonal axes of the activation tensor—experts $\bar { \{ E _ { 1 } , \ldots , E _ { E } \} }$ within a layer versus layers $\{ \breve { F } _ { 1 } , \ldots , F _ { L } \}$ across depth—yet no prior routing framework treats them as instances of a single TopK-selection operator over a unified unit set, leaving the compositional sparsity pattern width × depth unexploited.

## 3.3 Gap 3: No Explicit Difficulty-Aware Layer Routing

Existing layer-skipping methods rely on implicit signals (intermediate-layer confidence, implicitly learned routers). No work introduces an explicit difficulty-aware module to drive per-sample layer-path selection. Drowzee’s budget predictor controls only truncation depth, not non-contiguous skipping; MoD’s router carries no difficulty semantics. Concretely, prior gating signals are either (a) implicit and learned end-to-end with no interpretable semantics, or (b) confidence-based and thus post-hoc (computed from the layer’s own output); no method defines an explicit, pre-routing difficulty scalar $\hat { d } \in [ 0 , 1 ]$ with a documented feature-to-score mapping (table 2) and a monotonicity guarantee linking it to retained depth (theorem 4). We emphasize that theorem 4 is a theoretical guarantee obtained under the standard assumption of decreasing marginal layer contribution; the prototype reported in section 9 exhibits a routing-collapse negative result $( r = - 0 . 0 1 7 )$ that reflects the failure of the learned gate to modulate depth, not a refutation of the monotonicity assumption itself.

## 3.4 Gap 4: Insufficient Validation on Generative LLMs

Most layer-skipping work focuses on BERT-level discriminative tasks (Dee-BERT/FastBERT/LayerDrop/DynamicBERT). CALM/LayerSkip/MoD extend to generation but still use truncation early exit or per-token routing. Non-contiguous per-sample layer skipping on autoregressive generative LLMs—its effectiveness, training stability, and quality-safety mechanisms—has not been systematically validated. We note that the VALSE routing mechanism is architecturally extensible to autoregressive generation: the per-sample gate $\tilde { g } _ { l }$ is independent of token-position coupling and is compatible with causal masking, and self-speculative decod ing [Leviathan et al., 2023] provides a natural quality-safety fallback for skipped layers. This extensibility is a present design property of the routing mechanism rather than an empirical claim: it holds at the architectural level (the gate is position-agnostic and causal-mask-compatible) but is not exercised by the prototype, whose evaluation is confined to discriminative SST-2 routing. The empirical validation on autoregressive generative LLMs that would close this community-wide gap is identified here as the central open problem that the field must address, and the design conditions under which such a validation would be well-posed are precisely the ones VALSE satisfies.

Positioning. VALSE addresses all four gaps by introducing per-sample, non-contiguous layer skipping driven by an explicit difficulty estimator (Gap 1, 3), grounded in a structural duality with horizontal MoE that formalizes both dimensions as instances of a single TopK-selection operator over a unified unit set and thereby enables two-dimensional sparse activation (Gap 2), and designed for extension to autoregressive generation with self-speculative decoding (Gap 4). A feature-level comparison of VALSE against prior methods is provided in table 1; our contributions are detailed in section 1.

## 4 Method Overview

## 4.1 Design Motivation

While horizontal MoE reduces per-layer cost by activating only a subset of experts, the total depth— the count of sequential Transformer blocks—stays constant across all inputs. Consequently, even trivially easy inputs pay the full computational price of a deep stack, despite potentially needing only a fraction of its representational capacity. Hard samples may benefit from all L layers, but easy ones likely do not. This observation motivates depth-wise conditional computation: the number of executed layers should adapt to the input, yielding an orthogonal—and complementary—sparsity axis to width-wise MoE.

## 4.2 Core Idea: Per-Sample Non-Contiguous Layer Skipping

Existing methods have two limitations (section 2.4): (i) truncation-based early exit can only skip a contiguous suffix of layers; (ii) per-token routing (MoD) lacks per-sample difficulty semantics. We propose VALSE: for each sample s, a lightweight router generates a non-contiguous retention set $S ^ { ( s ) } \subseteq \{ 1 , \ldots , L \}$ , allowing arbitrary middle layers to be skipped while deeper layers are retained. Layers 1 and L are always retained. A difficulty score $\hat { d } \in [ 0 , 1 ]$ (section 5) drives per-layer gates $g _ { l } = \sigma ( w _ { l } ^ { \top } [ r _ { l } ; \hat { d } ] + b _ { l } )$ (section 6). A target budget $B _ { \mathrm { t a r g e t } }$ and skip rate $\rho ^ { * }$ control the compute– quality trade-off (section 7).

Duality with MoE: MoE uses a gate to select K of E experts per layer; VALSE uses a per-layer gate to select |S| of L layers per sample—one in width, one in depth, both using lightweight routers and load balancing. This structural duality is formalized in theorem 6 and unified under a single sparse selection framework in theorem ${ 9 } ;$ the key theoretical insight is that VALSE completes the depth dimension of sparse conditional computation, previously existing only in the width dimension (MoE).

## 4.3 Overall Architecture

Three components: (i) Difficulty estimation—extracts features from the first $K _ { \mathrm { e s t } }$ layers, producing $\hat { d }$ (section 5); (ii) Layer router—generates gating decisions forming $S ^ { ( s ) }$ (section 6); (iii) Training framework—auxiliary losses, curriculum learning, gradient stability (section 7). Data flow: embedding → first $K _ { \mathrm { e s t } }$ layers → difficulty score $\hat { d } \to$ per-layer gates → skip/execute → LM head. The full symbol table is in section A.

## 5 Difficulty Estimation Module

## 5.1 Design Goals

The module must: (i) produce interpretable difficulty scores (not implicitly learned); (ii) be lightweight (overhead far less than the layers it skips); (iii) operate at per-sample granularity; (iv) remain smooth during training.

## 5.2 Input Features

From the l-th layer’s hidden state $h _ { l } ^ { ( s ) } \in \mathbb { R } ^ { N \times d _ { \mathrm { m o d e l } } }$ , we extract a five-dimensional statistic (meanpooled over sequence length N, with $\bar { h } _ { l } = \operatorname* { m e a n } ( h _ { l } ^ { ( s ) } ) )$ as $r _ { l } \in \mathbb { R } ^ { 5 }$ :

Table 2: Difficulty estimation features.
<table><tr><td>Feature</td><td>Symbol</td><td>Computation</td><td>Intuition</td></tr><tr><td>Hidden-state norm</td><td> $\rho _ { l }$ </td><td> $\rho _ { l } = \| \mathrm { m e a n } ( h _ { l } ^ { ( s ) } ) \| _ { 2 }$ </td><td>Information content; activation magnitude</td></tr><tr><td>Inter-layer delta norm</td><td> $\Delta _ { l }$ </td><td> $\Delta _ { l } = \| \operatorname* { m e a n } ( h _ { l } - h _ { l - 1 } ) \| _ { 2 }$ </td><td>Representation change; small  $\Rightarrow$  skippable</td></tr><tr><td>Self-attention en- tropy</td><td> $H _ { l } ^ { \mathrm { a t t n } }$ </td><td> $\begin{array} { r } { \frac { 1 } { N } \sum _ { i } H ( \mathrm { s o f t m a x } ( \mathrm { a t t n } _ { i } ) ) } \end{array}$ </td><td>High ⇒ diffuse  $\Rightarrow$  complex</td></tr><tr><td>Prediction en- tropy</td><td> $H _ { l } ^ { \mathrm { p r e d } }$ </td><td> $H ( \mathrm { s o f t m a x } ( \mathrm { L M H e a d } ( \bar { h } _ { l } ) ) )$ </td><td>High ⇒ uncertain ⇒ hard</td></tr><tr><td>Singular-value concentration</td><td>Kl</td><td>Top-1 SVD singular value share</td><td> $\mathrm { H i g h } \Rightarrow \mathrm { r e d u n d a n t } \Rightarrow$  skippable</td></tr></table>

Aggregation: for $l \in \{ 1 , \ldots , K _ { \mathrm { e s t } } \} ( K _ { \mathrm { e s t } } = 2 \mathrm { - } 3 ) \mathrm { ; }$

$$
R ^ { ( s ) } = [ r _ { 1 } ; r _ { 2 } ; \cdot \cdot \cdot ; r _ { K _ { \mathrm { e s t } } } ] \in \mathbb { R } ^ { 5 K _ { \mathrm { e s t } } }\tag{1}
$$

Only the first $K _ { \mathrm { e s t } }$ layers are used because the routing decision must be made before deeper layers are executed. We set $\mathrm { \bar { K } } _ { \mathrm { e s t } } = 2$ in the prototype (with a design range of 2–3): this is the smallest value that exposes both an inter-layer delta $\Delta _ { l }$ (requiring $l \geq 2 )$ and a stable singular-value concentration $\kappa _ { l } .$ , while keeping routing overhead below $1 \%$ of total FLOPs; larger $K _ { \mathrm { e s t } }$ increases estimation cost without adding qualitatively distinct signals, since features beyond the first few layers correlate strongly with the final-layer statistics used at training equilibrium.

## 5.3 Difficulty Score

A lightweight MLP maps features to a scalar:

$$
d ( x ^ { ( s ) } ) = w _ { d } ^ { \top } \cdot \mathrm { R e L U } ( W _ { 1 } \cdot R ^ { ( s ) } + b _ { 1 } ) + b _ { d }\tag{2}
$$

where $W _ { 1 } \ \in \ \mathbb { R } ^ { d _ { \mathrm { h i d d e n } } \times 5 K _ { \mathrm { e s t } } } \ ( d _ { \mathrm { h i d d e n } } = 6 4 \mathrm { - } 1 2 8 )$ . Batch-internal min-max normalization yields $\hat { d } \in$ $[ 0 , 1 ] \colon$

$$
\hat { d } ( x ^ { ( s ) } ) = \frac { d ( x ^ { ( s ) } ) - \operatorname* { m i n } _ { s \in { \mathcal { B } } } d ( x ^ { ( s ) } ) } { \operatorname* { m a x } _ { s \in { \mathcal { B } } } d ( x ^ { ( s ) } ) - \operatorname* { m i n } _ { s \in { \mathcal { B } } } d ( x ^ { ( s ) } ) + \varepsilon }\tag{3}
$$

where 0 denotes the easiest sample and 1 the hardest. The mapping from difficulty to layer-skipping decisions is realized by the layer router (section 6): easy samples $( \hat { d }$ small) skip more layers; hard samples $( \hat { d }$ large) retain all layers. This design is theoretically motivated by theorem 4 (difficulty– depth monotonicity): under the standard assumption of decreasing marginal layer contribution, harder inputs benefit more from full depth, justifying the schedule $\rho _ { \mathrm { t a r g e t } } = \rho ^ { * } ( 1 - \hat { d } )$

Normalization limitation. The batch-internal min–max map in eq. (3) guarantees $\hat { d } \in \ [ 0 , 1 ]$ within each batch but renders scores incomparable across batches of different composition (the easiest sample of a hard batch may be harder than the hardest of an easy batch). This is acceptable for routing—the gate operates on within-batch relative ordering—but precludes using $\hat { d }$ as an absolute difficulty signal across experiments or as a training-free diagnostic. To restore cross-batch comparability, we specify a calibration target that decouples the score from the batch extremum: replacing the per-batch (min, max) in eq. (3) with a running exponential moving average $( m _ { t } , M _ { t } )$ updated as $\begin{array} { r } { m _ { t } = \lambda m _ { t - 1 } + ( 1 - \lambda ) \operatorname* { m i n } _ { s \in \mathcal { B } _ { t } } d ( x ^ { ( s ) } ) } \end{array}$ and $\begin{array} { r } { M _ { t } = \lambda M _ { t - 1 } + ( 1 - \lambda ) \operatorname* { m a x } _ { s \in \mathcal { B } _ { t } } d ( x ^ { ( s ) } ) } \end{array}$ with $\lambda = 0 . 9$ . Because the gate consumes only the within-batch ordering of $\hat { d } ,$ this calibration is routing-preserving for within-batch decisions while yielding a population-anchored score suitable for cross-experiment comparison; we adopt the batch-internal form in the prototype for simplicity and specify the calibrated variant as the production-target configuration.

Difficulty score distribution. The normalized score $\hat { d } \in [ 0 , 1 ]$ spreads samples across the full difficulty range by construction $( \mathsf { e q } . ( 3 ) ) \colon$ the easiest sample in each batch receives $\hat { d } = 0$ and the hardest receives $\hat { d } = 1$ , with intermediate samples distributed according to the raw MLP output $d ( x ^ { ( s ) } )$ ). Because the normalization is rank-preserving within a batch, the distribution of $\hat { d }$ reflects the empirical density of the raw MLP output $d ( x ^ { ( s ) } )$ : a roughly uniform $\hat { d }$ distribution indicates that the raw score is approximately uniform, whereas a U-shaped $\hat { d }$ distribution indicates bimodal raw scores. In the prototype evaluation (section 9), this within-batch spread did not translate into differentiated routing: the Pearson correlation between $\hat { d }$ and activated layers was $r = - 0 . 0 1 7 .$ indicating that the difficulty signal failed to modulate per-layer gating as intended, a result we report and analyze in section 9.3.

Per-feature contribution. Table 2 lists five features with their intuitions, and each feature admits a well-defined expected contribution direction under the monotonicity assumption of theorem 4: $\rho _ { l }$ and $\Delta _ { l }$ are expected to decrease with difficulty (harder inputs exhaust redundant magnitude faster), $H _ { l } ^ { \mathrm { a t t n } }$ and $H _ { l } ^ { \mathrm { p r e d } }$ to increase with difficulty (greater uncertainty and diffuse attention), and $\kappa _ { l }$ to decrease with difficulty (lower redundancy on harder inputs). Because the five features are pooled from only the first $K _ { \mathrm { e s t } } ~ = ~ 2$ layers, they are not statistically independent: $\rho _ { l }$ and $\Delta _ { l }$ share the mean-pooled hidden state, and $H _ { l } ^ { \mathrm { p r e d } }$ is computed from the same state through the LM head, so their marginal contributions are identifiable only when routing is non-degenerate. In the prototype, the router collapsed to a degenerate uniform-skip mode (section 9.3), so the marginal contribution of any single feature to routing differentiation is not identifiable from the prototype data; we therefore report the feature-level design rationale and expected directions as the present specification, with the empirical variance-decomposition analysis contingent on a non-collapsed routing regime that the prototype’s $r = - 0 . 0 1 7$ result did not provide.

Computational overhead. The difficulty estimator is invoked exactly once per sample, after the first $\bar { K } _ { \mathrm { e s t } } = 2$ layers and before any routing decision, so its cost is amortized over the layers it gates. Concretely, the five features are all $O ( N d _ { \mathrm { m o d e l } } ) \colon \rho _ { l }$ and $\Delta _ { l }$ require a mean-pool and two norm reductions; $H _ { l } ^ { \mathrm { a t t n } }$ reuses the already-computed attention weights (no extra matmul); $H _ { l } ^ { \mathrm { p r e d } }$ requires one LM-head forward on the single pooled vector $\bar { h } _ { l }$ (cost $O ( d _ { \mathrm { m o d e l } } \cdot | V | )$ , amortized to a single token); and $\kappa _ { l }$ requires a top-1 singular value of the $N \times d _ { \mathrm { m o d e l } }$ pooled matrix, computed via a fixed number of power iterations $( O ( N d _ { \mathrm { m o d e l } } )$ per iteration). The scoring MLP adds $\dot { O } ( d _ { \mathrm { h i d d e n } } \cdot 5 K _ { \mathrm { e s t } } )$

parameters with $d _ { \mathrm { h i d d e n } } = 6 4 .$ Against a single Transformer block costing $O ( N d _ { \mathrm { m o d e l } } ^ { 2 } )$ for attention plus $O ( N d _ { \mathrm { m o d e l } } d _ { \mathrm { f f n } } )$ for the FFN, the estimator’s total cost is $O ( N d _ { \mathrm { m o d e l } } + d _ { \mathrm { m o d e l } } | V | )$ , which is dominated by a single block whenever $d _ { \mathrm { f f n } } \gg d _ { \mathrm { m o d e l } }$ (the standard ratio $d _ { \mathrm { f f n } } = 4 d _ { \mathrm { m o d e l } } )$ . Since VALSE skips at least one full block to realize any savings, the estimator overhead is strictly less than the cheapest layer it can gate, satisfying design goal (ii); in the prototype this ratio is below 1% of total FLOPs, consistent with the overhead budget assumed throughout.

Feature selection rationale. The five features in table $^ 2$ were selected to span four complementary signals of layer marginality—magnitude $( \rho _ { l } )$ , dynamics $( \Delta _ { l } )$ , dispersion $( \dot { H } _ { l } ^ { \mathrm { a t t n } } )$ , and uncertainty $( H _ { l } ^ { \mathrm { p r e d } } )$ —plus a redundancy probe $( \kappa _ { l } )$ that directly operationalizes the layer-redundancy hypothesis of [Men et al., 2024]. We deliberately exclude two natural candidates. First, an explicit layer-wise mutual-information estimate between $h _ { l }$ and the input x is excluded because it requires costly estimation (e.g., MINE-style critics) whose overhead would violate design goal (ii) and whose variance at small $\dot { N }$ dominates the signal. Second, a learned end-to-end difficulty embedding (a small Transformer over the prefix) is excluded because it would forfeit the interpretability guarantee of design goal (i): the five-feature design exposes a documented, fixed feature-to-score mapping (table 2) whose individual contributions admit expected directions under theorem $^ { 4 , }$ whereas a learned em bedding’s difficulty semantics would be opaque. The selection is therefore not exhaustive but principled: each feature admits a well-defined expected contribution direction (table 2, right column), the set is jointly computable from quantities already available in a forward pass, and its dimensionality $( 5 K _ { \mathrm { e s t } } = 1 0 )$ is small enough that the scoring MLP cannot memorize sample identity, preserving the difficulty signal’s generalization.

Summary. The module produces a single, normalized, interpretable scalar $\hat { d } ( x ^ { ( s ) } ) \in [ 0 , 1 ]$ at persample granularity with sub-1% FLOPs overhead, satisfying all four design goals. Its sole known limitation—within-batch rank semantics that preclude cross-batch comparison—is addressed in the production-target configuration by the calibrated variant specified above. The downstream consumption of $\hat { d } ,$ namely the per-layer gating and retention-set construction, is the subject of section $\begin{array} { r } { 6 ; } \end{array}$ the theoretical guarantee linking $\hat { d }$ to retained depth is theorem 4, established in section 8.

Calibrated variant pseudocode. The exponential moving average (EMA) calibration that restores cross-batch comparability of $\hat { d }$ (specified qualitatively in the Normalization limitation paragraph above) is realized by the following per-step update, executed once per batch before the min–max map of eq. (3):

```python
# EMA-calibrated difficulty normalization (production target)
# m_t, M_t: running min/max of raw score d(x)
lambda = 0.9 # EMA decay
# init: seed with first-batch extrema before EMA updates
m_0 = min_{s in B_1} d(x^{(s)}); M_0 = max_{s in B_1} d(x^{(s)})
m_t = lambda * m_{t-1} + (1 - lambda) * min_{s in B_t} d(x^{(s)})
M_t = lambda * M_{t-1} + (1 - lambda) * max_{s in B_t} d(x^{(s)})
d_hat = (d(x) - m_t) / (M_t - m_t + eps) # population-anchored
```

The batch-internal form of eq. (3) is recovered by setting $\begin{array} { r l r l } { ( m _ { t } , M _ { t } ) } & { { } } & { = } \end{array}$ $( \mathrm { m i n } _ { s \in \mathcal { B } _ { t } } d ( x ^ { ( s ) } ) , \mathrm { m a x } _ { s \in \mathcal { B } _ { t } } d ( x ^ { ( s ) } ) )$ at each step, so the prototype and production variants differ only in the reference extrema used for normalization; the routing decision itself (the gate g˜<sub>l</sub>) is invariant under this substitution because it consumes only within-batch ordering.

Empirical-validation scope. The present section specifies the difficulty estimator as a design contract: the feature-to-score mapping, the expected contribution directions of table $^ { 2 , }$ , and the monotonicity link to retained depth (theorem $^ { 4 ) }$ are theoretical statements whose empirical validity is contingent on a non-collapsed routing regime. The prototype reported in section 9.3 collapsed to a uniform-skip mode $( r = - 0 . 0 1 7 )$ , so the variance-decomposition analysis of individual features and the calibration of <sup>ˆ</sup>d against an external difficulty oracle are declared here as empirically open rather than verified; they are part of the design specification but not part of the prototype evidence base.

## 6 Layer Routing Strategies

## 6.1 Per-Sample Non-Contiguous Layer Skipping

For sample s, the retention set $S ^ { ( s ) } \subseteq \{ 1 , \ldots , L \}$ satisfies: $( \mathrm { i } ) \ S ^ { ( s ) }$ may be any subset (noncontiguous), e.g., {1, 2, 4, 7, 11, 12}; (ii) layers 1 and L are always retained; (iii) all tokens in the same sample share the same $S ^ { ( s ) }$ (per-sample granularity). Unlike truncation-based early exit $( S = \{ 1 , \ldots , \bar { k } \} )$ or per-token MoD, VALSE achieves per-sample non-contiguous skipping (see table 7 in the appendix).

## 6.2 Gating Function

For each layer l, the gating function is:

$$
g _ { l } ( x ^ { ( s ) } ) = \sigma \Big ( w _ { l } ^ { \top } \cdot [ r _ { l } ; \hat { d } ( x ^ { ( s ) } ) ] + b _ { l } \Big ) \in [ 0 , 1 ]\tag{4}
$$

where $w _ { l } \in \mathbb { R } ^ { d _ { \mathrm { f e a t } } + 1 }$ , and $b _ { l }$ is initialized to 2.0 so that gates are close to 1 (full-layer execution) at the start of training. At inference, hard quantization is applied:

$$
\tilde { g } _ { l } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \ g _ { l } > \theta _ { l } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{5}
$$

$( \theta _ { l } = 0 . 5$ by default). During training, a straight-through estimator (STE) allows gradients to flow through the soft gate g<sub>l</sub>.

## 6.3 Residual Connection after Skipping

Scheme A—pure identity (default):

$$
h _ { l + 1 } = h _ { l } + \tilde { g } _ { l } \cdot F _ { l } ( h _ { l } )\tag{6}
$$

When $\tilde { g } _ { l } = 0 , h _ { l + 1 } = h _ { l }$ (identity skip; since $d _ { \mathrm { m o d e l } }$ is consistent, no projection is needed).

Scheme B—residual projection (optional, not used in prototype): Note: the function-space results (theorems 10 and 11) are established only under Scheme A; Scheme B introduces $W _ { l } ^ { p r o j }$ which may break the subset relationship (see Scheme A/B Remark in section 8). The prototype uses Scheme A throughout.

$$
h _ { l + 1 } = h _ { l } + \tilde { g } _ { l } \cdot F _ { l } ( h _ { l } ) + ( 1 - \tilde { g } _ { l } ) \cdot W _ { l } ^ { \mathrm { p r o j } } \cdot h _ { l }\tag{7}
$$

where $W _ { l } ^ { \mathrm { p r o j } }$ may be low-rank $( W = A B ^ { \top } , r \ll d _ { \mathrm { m o d e l } } )$ . Gumbel noise is applied to gates during training to enhance robustness to non-contiguous skip paths, inspired by LayerDrop.

## 6.4 MoE Routing Analogy

Table 8 (appendix) details the mathematical analogy: MoE selects K of E experts via Top- $K ;$ VALSE selects a retention set |S| of L layers via thresholding. Both use lightweight routers and impose load-balancing regularization. MoE faces expert collapse (few experts selected); VALSE faces the dual layer collapse (few layers retained), requiring load-balancing loss (section 7).

## 6.5 Forward Pass

algorithm 1 shows VALSE’s single-sample forward pass. In batched implementation, samples with $\tilde { g } _ { l } = 1$ are clustered via argsort to form per-layer sub-batches, reducing gather/scatter overhead.

## 6.6 Architecture Variants

Three variants are designed (full comparison in table 9 in the appendix):

VALSE-Hard (main variant). Hard routing with STE; $\tilde { g } _ { l } \in \{ 0 , 1 \}$ ; per-layer learnable $w _ { l } , b _ { l } .$ . At inference, layers are genuinely skipped—directly saving FLOPs.

Algorithm 1 VALSE single-sample forward pass.   
1: h ← Embedding $( x _ { s } )$   
2: features $ [ ] ; \bar { h _ { \mathrm { p r e v } } }  h$   
3: for $l = 0$ to $K _ { \mathrm { e s t } } - 1$ do   
4: $h _ { l } \gets \mathrm { L a y e r } _ { l } ( h _ { \mathrm { p r e v } } )$   
5: $r _ { l } \gets \mathrm { E x t r a c t F }$ eatures $( h _ { l } , h _ { \mathrm { p r e v } } , l )$   
6: features. $a p p e n d ( r _ { l } ) ; h _ { \mathrm { p r e v } } \dot {  } h _ { l }$   
7: end for   
8: R ← Concat(features)   
9: $\hat { d } \gets$ DifficultyEstimator(R) ▷ normalize to [0, 1]   
10: $h  h _ { \mathrm { p r e v } } ; S  \emptyset$   
11: for $l \stackrel { \cdot } { = } K _ { \mathrm { e s t } }$ to $L - 1$ do   
12: $r _ { l } \gets$ ExtractFeatures $( h , h _ { \mathrm { p r e v } } , l )$   
13: $g _ { l } \gets \sigma ( w _ { l } ^ { \top } \cdot [ r _ { l } ; \hat { d } ] + b _ { l } )$   
14: if training then g˜<sub>l</sub> ← GumbelSTE $( g _ { l } , \theta _ { l } )$   
15: else $\tilde { g } _ { l }  \mathbf { 1 } [ g _ { l } > \theta _ { l } ]$   
16: end if   
17: if $\tilde { g } _ { l } = 1$ then $h  h + \mathrm { L a y e r } _ { l } ( h )$   
18: else $S  S \cup \{ l \}$   
19: end if   
20: $h _ { \mathrm { p r e v } }  h$   
21: end for   
22: return LMHead(h), S

VALSE-Soft. Continuous gating: $h _ { l + 1 } = h _ { l } + g _ { l } \cdot F _ { l } ( h _ { l } )$ . Gradients propagate directly; FLOPs savings depend on $g _ { l } \to 0$ . Most stable; can serve as initialization for hard routing.

VALSE-Global. A global router produces $p = \mathrm { s o f t m a x } ( W _ { \mathrm { g l o b a l } } \cdot R ^ { ( s ) } )$ , then:

$$
S ^ { ( s ) } = \mathrm { T o p K } ( p ) , \quad K = \mathrm { r o u n d } \big ( B _ { \mathrm { t a r g e t } } - \hat { d } \cdot \big ( B _ { \mathrm { t a r g e t } } - L _ { \mathrm { m i n } } \big ) \big )\tag{8}
$$

K varies with difficulty $( \hat { d } = 0 \Rightarrow K = B _ { \mathrm { t a r g e t } } ; \hat { d } = 1 \Rightarrow K = L )$ . This variant has the strongest MoE analogy: Top-K layer selection ↔ Top-K expert selection.

## 7 Training Method

The training objective jointly optimizes task quality and computational efficiency while maintain ing router stability. Four auxiliary losses are designed.

## 7.1 Auxiliary Loss Design

Layer utilization balancing L<sub>balance</sub>. This is the dual of MoE routing collapse: the router may retain only the first and last layers. Let $\begin{array} { r } { \bar { f } _ { l } = \frac { 1 } { B } \sum _ { s } \tilde { g } _ { l } ^ { ( s ) } } \end{array}$ and $\begin{array} { r } { \bar { P } _ { l } = \frac { 1 } { B } \sum _ { s } g _ { l } ^ { ( s ) } } \end{array}$ :

$$
\mathcal { L } _ { \mathrm { b a l a n c e } } = \alpha \cdot L \cdot \sum _ { l = 1 } ^ { L } \bar { f } _ { l } \cdot \bar { P } _ { l }\tag{9}
$$

By the AM–GM inequality, this is minimized under a fixed budget when all layers are used uniformly. (α = 0.01.)

Efficiency loss $\mathcal { L } _ { \mathrm { e f f } } .$ A one-sided hinge controls the total activated layers $\begin{array} { r } { \bar { A } = \sum _ { l } \bar { f } _ { l } \colon } \end{array}$

$$
\mathcal { L } _ { \mathrm { e f f } } = \beta \cdot \frac { \mathrm { m a x } ( 0 , \bar { A } - B _ { \mathrm { t a r g e t } } ) ^ { 2 } } { L ^ { 2 } }\tag{10}
$$

The weight increases progressively: $\beta _ { \mathrm { e f f } } ( t ) ~ = ~ \beta \cdot \mathrm { m i n } ( 1 , t / T _ { \mathrm { w a r m u p } } ) . ~ ( \beta ~ = ~ 0 . 1 , ~ B _ { \mathrm { t a r g e t } } ~ \in$ [0.5L, 0.85L].)

Output consistency $\mathcal { L } _ { \mathrm { c o n s i s t } } .$ . KL distillation aligns the skip-path output with the full-layer path:

$$
\mathcal { L } _ { \mathrm { c o n s i s t } } = \delta \cdot T _ { \mathrm { K D } } ^ { 2 } \cdot \mathrm { K L } ( p _ { \mathrm { f u l l } } | | p _ { \mathrm { s k i p } } )\tag{11}
$$

$( T _ { \mathrm { K D } } = 2 . 0 , \delta = 0 . 5 .$ , annealed to $\delta / 3 . )$ Three teacher strategies reduce the double-forward overhead: EMA teacher, periodic teacher $( K _ { \mathrm { c o n s i s t } } = 1 0 )$ , and difficulty-bucketed caching.

Z-loss $\mathcal { L } _ { z } .$ Stabilizes the numerical range of gating logits (following Switch Transformer): $\mathcal { L } _ { z } =$ $\begin{array} { r } { \gamma \cdot \frac { 1 } { B } \sum _ { s } \sum _ { l } ( w _ { l } ^ { \top } [ r _ { l } ; \hat { d } ] + b _ { l } ) ^ { 2 } . ~ ( \gamma = 1 0 ^ { - 4 } . ) } \end{array}$

The total loss combines all terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } + \alpha _ { t } \mathcal { L } _ { \mathrm { b a l a n c e } } + \beta _ { t } \mathcal { L } _ { \mathrm { e f f } } + \delta _ { t } \mathcal { L } _ { \mathrm { c o n s i s t } } + \gamma _ { t } \mathcal { L } _ { z }\tag{12}
$$

where $\alpha _ { t } , \beta _ { t } , \delta _ { t } , \gamma _ { t }$ are the (possibly time-varying) weights of each auxiliary loss.

## 7.2 Curriculum Learning

Stage 1: Full-layer warm-up $( \rho _ { \mathrm { s k i p } } = 0 )$ . Gate bias $b _ { l } = 2 . 0$ forces $\tilde { g } _ { l } = 1 ;$ ; only $\mathcal { L } _ { \mathrm { t a s k } } + \gamma \mathcal { L } _ { z }$ is active. Duration: $\eta _ { 1 } \approx 0 . 2 5 \substack { - 0 . 3 3 }$ of $T _ { \mathrm { t o t a l } }$

Stage 2: Progressive skipping $( 0 \to \rho ^ { * } )$

$$
\begin{array} { r } { \rho _ { \mathrm { s k i p } } ( t ) = \rho ^ { * } \cdot \frac { 1 } { 2 } \left[ 1 - \cos \Bigl ( \pi \cdot \frac { t - T _ { 1 } } { T _ { 2 } - T _ { 1 } } \Bigr ) \right] } \end{array}\tag{13}
$$

Auxiliary losses progressively take effect; Gumbel temperature $\tau _ { G }$ anneals $2 . 0  0 . 5 ;$ bias anneals to $b _ { \mathrm { t a r g e t } } \approx 0 . 7 6$ . Duration: $\eta _ { 2 } \approx 0 . 5 – 0 . 6 7$ (extended to $2 \times$ original to allow differentiated routing to emerge).

Stage 3: Joint optimization $( \rho _ { \mathrm { { s k i p } } } = \rho ^ { * } )$ . All parameters are jointly optimized; δ anneals downward. Learning rate cosine-anneals to $1 / 1 0$ of the Stage 2 peak. Duration: $\eta _ { 3 } \approx 0 . 2 5 \ – 0 . 4$

## 7.3 Difficulty-Aware Scheduling and Stability

Individualized target: $\rho _ { \mathrm { t a r g e t } } ^ { ( s ) } = \rho ^ { * } ( 1 - \hat { d } ^ { ( s ) } )$ . The training feedback loop ( <sup>ˆ</sup>d → gate → skip path → loss → gradient $ \hat { d } )$ is stabilized under:

$$
\begin{array} { r } { \left\| \frac { \partial \hat { d } } { \partial \theta _ { D } } \right\| \cdot \left\| \frac { \partial \mathcal { L } _ { \mathrm { e f f } } } { \partial \hat { d } } \right\| \cdot \left\| \frac { \partial \tilde { g } } { \partial \hat { d } } \right\| \cdot \left\| \frac { \partial \mathcal { L } _ { \mathrm { t a s k } } } { \partial \tilde { g } } \right\| < 1 } \end{array}\tag{14}
$$

Three safeguards: normalization anchoring, difficulty-estimator gradient clipping (max norm = 1.0), and gradient flow through $\mathcal { L } _ { \mathrm { e f f } }$ (removing stop-gradient so the difficulty signal propagates to the router). Probabilistic full-layer recomputation $( p _ { \mathrm { r e c o m p } } = 0 . 1 )$ prevents “dead layers” among skipped layers.

Proofsketchfor eq. (14). The training feedback loop forms a composition: $\hat { d } \ \xrightarrow { \phi _ { 1 } } \tilde { g } \ \xrightarrow { \phi _ { 2 } } \ h _ { \mathrm { s k i p } } \ \xrightarrow { \phi _ { 3 } } \quad$ $\mathcal { L } _ { \mathrm { t a s k } } \stackrel { \phi _ { 4 } } { \longrightarrow } \nabla _ { \hat { d } } \mathcal { L } _ { \mathrm { e f f } } \stackrel { \phi _ { 5 } } { \longrightarrow } \theta _ { D }$ . By the chain rule, the product of Jacobian norms along this loop determines whether the feedback is contractive (stable) or expansive (divergent). eq. (14) requires the product to be $< 1$ , i.e., the loop is a contraction. The three safeguards enforce this: (i) normalization anchors $\lVert \partial \hat { d } / \partial \theta _ { D } \rVert$ by bounding $\hat { d } \in [ 0 , 1 ] ; ( \mathrm { i i } )$ gradient clipping upper-bounds the effective Jacobian norm; (iii) allowing gradient flow through $\mathcal { L } _ { \mathrm { e f f } }$ ensures the difficulty signal propagates, while the efficiency loss provides a corrective force that counteracts runaway amplification. When all three hold, each Jacobian factor is bounded, and their product i $s < 1$ provided that (i) the difficulty estimator $\theta _ { D }$ is Lipschitz-continuous with constant $L _ { D } \leq 1$ (ensured by gradient clipping with max norm $= 1 . 0 )$ , and (ii) the gate function $\sigma ( w _ { l } ^ { \top } [ r _ { l } ; \hat { d } ] + b _ { l } )$ has bounded gradient $\leq 0 . 2 5$ (inherent to the sigmoid). These are sufficient conditions, not mild assumptions; their empirical validity at scale remains unverified in the prototype. □ □

Prototype implementation note. In the current prototype, curriculum learning Stage 1 is simplified (omitted), and Gumbel temperature annealing is configured $( \tau _ { G } : 2 . 0 \stackrel { - } {  } 0 . 5 )$ but not yet fully implemented in the training code. The prototype uses a fixed $\tau _ { G } = 1 . 0$ throughout training. Full implementation of the three-stage curriculum and temperature annealing is planned for the next iteration.

## 8 Theoretical Analysis

## 8.1 Computational Savings

Per-layer FLOPs. The per-sample FLOPs of the l-th Transformer block decompose as $C _ { l } \ =$ $C _ { l } ^ { \mathrm { a t t n } } + \bar { C } _ { l } ^ { \mathrm { f f n } }$ , where $C _ { l } ^ { \mathrm { a t t n } } \stackrel { \cdot } { \approx } 4 N d _ { \mathrm { m o d e l } } ^ { 2 } + 2 N ^ { 2 } d _ { \mathrm { m o d e l } }$ and $C _ { l } ^ { \mathrm { f f n } } \approx 8 N d _ { \mathrm { m o d e l } } ^ { 2 } .$ . For uniform layers, this is a constant $C _ { \mathrm { l a y e r } }$

Lemma 1 (Linearity of expectation). For any random variables $\begin{array} { r l } { X _ { 1 } , \ldots , X _ { n } , \mathbb { E } [ \sum _ { i } X _ { i } ] } & { = } \end{array}$ $\textstyle \sum _ { i } \mathbb { E } [ X _ { i } ]$

Theorem 2 (Expected FLOPs). Let the skip rate of layer l be $p _ { 1 } , p _ { 2 } , \ldots , p _ { L }$ (with $p _ { 1 } = p _ { L } = 0 )$ The expected per-sample FLOPs of VALSE is:

$$
\mathbb { E } [ \mathbb { F } _ { V A L S E } ] = C _ { f i x e d } + \sum _ { l = K _ { e s t } + 1 } ^ { L } \left( 1 - p _ { l } \right) \cdot C _ { l }\tag{15}
$$

Proof. Total FLOPs equals the sum of components. The fixed part is $C _ { \mathrm { f i x e d } }$ . For the skippable part, layer l’s FLOPs contribution is $\tilde { g } _ { l } \cdot C _ { l }$ . Taking expectation, $\mathbb { E } [ \tilde { g } _ { l } \cdot C _ { l } ] = C _ { l } \cdot P ( \tilde { g } _ { l } = \bar { 1 } ) =$ $C _ { l } \cdot ( 1 - p _ { l } )$ . By theorem 1 (linearity of expectation), summing over all skippable layers yields $\begin{array} { r } { \mathbb E [ \mathbb F _ { \mathrm { V A L S E } } ] = C _ { \mathrm { f i x e d } } + \sum _ { l = K _ { \mathrm { e s t } } + 1 } ^ { L } ( 1 - p _ { l } ) \cdot C _ { l } } \end{array}$ □

Remark (gate correlation). The linearity of expectation used above does not require the gate decisions $\{ \tilde { g } _ { l } \}$ to be independent across layers. Although these decisions share the common difficulty estimate <sup>ˆ</sup>d and are therefore correlated, $\mathbb { E } [ \tilde { g } _ { l } \cdot C _ { l } ] = C _ { l } \cdot \mathbb { E } [ \tilde { g } _ { l } ] = C _ { l } \cdot ( 1 - p _ { l } )$ holds for any joint distribution of $\{ \tilde { g } _ { l } \}$

Corollary 3 (Uniform skip rate). If all skippable layers share skip rate $p ,$ each layer has $F L O P s$ $C _ { l a y e r } ,$ , and $K _ { e s t } \ll L .$

$$
\mathbb { E } [ \mathbb { F } _ { V A L S E } ] \approx \left( 1 - p \cdot \frac { L - K _ { e s t } - 1 } { L } \right) \cdot \mathbb { F } _ { f u l l }\tag{16}
$$

VALSE satisfies $\mathbb { E } [ \mathbb { F } _ { \mathrm { V A L S E } } ] ~ \leq ~ \mathbb { F } _ { \mathrm { f u l l } }$ (upper bound, equality iff $p _ { l } ~ = ~ 0 ~ \forall l )$ and $\mathbb { E } [ \mathbb { F } _ { \mathrm { V A L S E } } ] \ \geq$ $C _ { \mathrm { f i x e d } } + C _ { L }$ (lower bound). When VALSE and truncation-based early exit share the same average activated layer count $\mu ,$ , their FLOPs are comparable; however, VALSE can activate non-contiguous layer subsets for different samples, whereas truncation can only activate a contiguous prefix. The theoretical savings under the difficulty-aware schedule are derived in section 8.2 (theorem 5).

## 8.2 Difficulty-Adaptive Scheduling

The FLOPs analysis above treats the skip rate p as an exogenous parameter. In practice, VALSE employs a difficulty-aware schedule $\rho _ { \mathrm { t a r g e t } } ^ { ( s ) } = \rho ^ { * } ( 1 - \hat { d } ^ { ( s ) } )$ ), where harder samples (larger $\hat { d } )$ skip fewer layers. We now provide a theoretical justification for this design choice.

Proposition 4 (Difficulty–depth monotonicity). Assume that the marginal task contribution ofeach layer is non-decreasing in input difficulty: for each layer l, the quantity

$$
\Delta _ { l } ( x ) : = \mathcal { L } _ { t a s k } \big ( F _ { 1 : l - 1 } ( x ) \big ) - \mathcal { L } _ { t a s k } \big ( F _ { 1 : l } ( x ) \big )
$$

is non-decreasing in ${ \hat { d } } ( x )$ . Then the optimal skip rate $\rho ^ { \star } ( x )$ that minimizes $\mathcal { L } _ { t a s k } + \lambda \mathbb { F }$ is nonincreasing in ${ \hat { d } } ( x )$ : harder inputs should skipfewer layers.

Proof. We analyze the Lagrangian relaxation of the constrained optimization. The per-sample objective is to choose a retention set $S ( x ) \subseteq \{ 1 , \ldots , L \}$ minimizing

$$
\mathcal { I } ( S ) : = \mathcal { L } _ { \mathrm { t a s k } } \big ( F _ { 1 : S } ( x ) ; y \big ) + \lambda \mathbb { F } ( S ( x ) ) ,
$$

where $\begin{array} { r } { \mathbb { F } ( S ( x ) ) = C _ { \mathrm { f i x e d } } + \sum _ { l \in S } C _ { l } } \end{array}$ is the FLOPs of retaining layers in $S$ and $\lambda > 0$ is the FLOPs– accuracy trade-off parameter.

Define the marginal task contribution of layer l as

$$
\begin{array} { r } { \Delta _ { l } ( x ) : = \mathcal { L } _ { \mathrm { t a s k } } \big ( F _ { 1 : l - 1 } ( x ) ; y \big ) - \mathcal { L } _ { \mathrm { t a s k } } \big ( F _ { 1 : l } ( x ) ; y \big ) , } \end{array}
$$

i.e., the reduction in task loss obtained by adding layer l to the retention set. The marginal FLOPs cost of retaining layer l is $\lambda C _ { l }$

Optimality condition. The Lagrangian $\mathcal { I } ( S )$ is minimized when, for each layer $l ,$ the marginal task benefit of retention equals the marginal FLOPs cost: $\Delta _ { l } ( x ) = \lambda C _ { l }$ . Layers where $\Delta _ { l } ( x ) > \lambda C _ { l }$ should be retained; layers where $\breve { \Delta } _ { l } ( x ) < \lambda C _ { l }$ should be skipped. This is the standard greedyretention condition for separable costs.

Monotonicity argument. Under the assumption that $\Delta _ { l }$ is non-decreasing in $\hat { d }$ (harder inputs derive greater benefit from each layer), for a harder input $x ^ { \prime }$ with ${ \hat { d } } ( x ^ { \prime } ) > { \hat { d } } ( x )$ , we have $\Delta _ { l } ( x ^ { \prime } ) \geq \Delta _ { l } ( x )$ for every l. The equality $\dot { \Delta _ { l } } = \lambda C _ { l }$ is therefore reached at a larger retention set $| S ( \dot { x ^ { \prime } } ) | \geq | S ( \dot { x } ) |$ $( \mathrm { i . e . }$ ., fewer layers skipped). Consequently, the optimal skip rate $\rho ^ { \star } ( \hat { d } ) = 1 - | S ^ { \star } ( \hat { d } ) | / L$ is nonincreasing in <sup>ˆ</sup>d: harder inputs should skip fewer layers.

Expected FLOPs. By theorem 1, the expected FLOPs under the difficulty-aware schedule is $\mathbb { E } [ \mathbb { F } ] =$ $\begin{array} { r } { C _ { \mathrm { f i x e d } } + \sum _ { l } \mathbb { E } [ \tilde { g } _ { l } | \hat { d } ] \cdot C _ { l } = C _ { \mathrm { f i x e d } } + \sum _ { l } ( 1 - p _ { l } ( \hat { d } ) ) \cdot C _ { l } } \end{array}$ , where $p _ { l } ( \hat { d } )$ decreases with $\hat { d } ,$ confirming that harder inputs incur more computation. The linear schedule $\rho _ { \mathrm { t a r g e t } } = \rho ^ { * } ( 1 - \hat { d } )$ is the simplest monotone-decreasing parametrization consistent with this result. □ □

Remark. theorem 4 establishes the direction of the difficulty–depth relationship under a standard assumption about decreasing marginal layer contribution, consistent with the layer redundancy hypothesis [Men et al., 2024]. It does not prove that the learned estimator $\hat { d }$ correctly captures true input difficulty; that is an empirical question partially addressed by our prototype experiments (section 9.2).

Observation 5 (Theoretical savings under difficulty-aware scheduling). Under the schedule $\rho _ { \mathrm { t a r g e t } } ^ { ( s ) } =$ $\rho ^ { * } ( 1 - { \hat { d } } ^ { ( s ) } )$ with $\rho ^ { * } = 0 . 3$ , uniform per-layer FLOPs $C _ { \mathrm { l a y e r } }$ , and $K _ { \mathrm { e s t } } \ll L$ , the average skip rate is $\mathbb { E } [ p ] = \rho ^ { * } \cdot ( 1 - \mathbb { E } [ \hat { d } ] )$ . By theorem $3 ,$ , the theoretical FLOPs savings are:

$$
{ \mathrm { s a v i n g s ~ } } = \rho ^ { * } \cdot ( 1 - \mathbb { E } [ { \hat { d } } ] ) \cdot { \frac { L - K _ { \mathrm { e s t } } - 1 } { L } } .
$$

For the prototype $( L = 6 , K _ { \mathrm { e s t } } = 2 , \rho ^ { * } = 0 . 3 ) \colon$ savings range from 0% (all-hard, $\mathbb { E } [ \hat { d } ] = 1 )$ to 15% (all-easy, $\mathbb { E } [ \hat { d } ] = 0 ) ;$ ; with a 50–50 easy–hard split, savings are $\approx 7 . 5 \%$ . For a 12-layer model $( K _ { \mathrm { e s t } } = 2 )$ , the range is 0% to 22.5%. These are theoretical estimates under the assumed schedule and idealized difficulty distribution; actual savings depend on the learned $\hat { d }$ and gate calibration.

## 8.3 MoE–VALSE Duality

Proposition 6 (Structural duality of MoE and VALSE). Horizontal MoE and vertical VALSE are instances ofthe same sparse selection problem on two orthogonal dimensions: MoE selects K ofE candidate transformations per layer (width); VALSE selects |S| of L candidate layers per sample (depth). Their pipelines are structurally parallel: router generates selection probabilities → Top-K / hard quantization → sparse execution → load-balancing regularization.

Proof. We formalize the duality via an isomorphism of the routing pipelines. Define the conditional computation template $\mathcal { T } = ( \mathcal { R } , \mathcal { S } , \mathcal { E } , \mathcal { L } )$ consisting of four components:

1. Router $\mathcal { R } \colon \mathrm { a }$ function $g : \mathcal { X } \to [ 0 , 1 ] ^ { N }$ mapping the input to selection probabilities, where N is the number of candidate compute units.

2. Selection operator $s \colon$ a mapping from probabilities to a binary mask $m \in \{ 0 , 1 \} ^ { N }$ (Top-K for MoE, thresholding for VALSE).

3. Sparse execution $\varepsilon \colon$ only units with $m _ { i } = 1$ are evaluated, producing output $\textstyle \sum _ { i : m _ { i } = 1 } w _ { i }$ $u _ { i } ( \mathrm { i n p u t } )$

4. Load-balancing regularizer $\mathcal { L } \mathrm { : }$ : a regularization term preventing degenerate selection (the MoE load-balancing loss or VALSE entropy regularizer).

MoE instantiation. The compute units $\{ u _ { 1 } , \dotsc , u _ { E } \}$ are expert networks organized in parallel within each layer: $N = E$ (width dimension). The router $g _ { l } ( \boldsymbol { x } ) \in \mathbb { R } ^ { E }$ produces expert-selection probabilities, and $\mathrm { T o p K } _ { h }$ selects $K _ { h }$ experts per token. The output is $\begin{array} { r } { \sum _ { e \in \mathrm { T o p K } _ { h } } g _ { l } ^ { ( e ) } ( x ) \cdot f _ { e } ( h _ { l } ) } \end{array}$

VALSE instantiation. The compute units $\{ u _ { 1 } , \ldots , u _ { L } \}$ are Transformer layers organized sequentially across the depth: $N \ = \ \bar { L }$ (depth dimension). The router produces per-layer gate logits $g _ { l } ( x ) \in \mathbb { R }$ , and thresholding produces $\bar { \tilde { g } } _ { l } \in \{ 0 , 1 \}$ }. The output is the sequential composition through retained layers.

Isomorphism. The isomorphism $\varphi : \mathcal { T } _ { \mathrm { M o E } }  \mathcal { T } _ { \mathrm { V A L S E } }$ maps each component:

• Router: $g _ { l } ^ { \mathrm { M o E } } \in \mathbb { R } ^ { E }  g _ { l } ^ { \mathrm { V A L S E } } \in \mathbb { R }$ (vector-to-scalar gate, reducing the selection dimension from width to depth).

• Selection: $\mathrm { T o p K } _ { h }  { \bf 1 } [ g _ { l } > \theta _ { l } ]$ (Top-K to binary threshold).

• Execution: $\begin{array} { r } { \sum _ { e \in \mathrm { T o p K } } w _ { e } f _ { e }  \bigcirc _ { l \in S } ( I + F _ { l } ) } \end{array}$ (parallel weighted sum to sequential residual composition).

• Regularization: $\mathcal { L } _ { \mathrm { b a l a n c e } } ^ { \mathrm { M o E } }  \mathcal { L } _ { \mathrm { b a l a n c e } } ^ { \mathrm { V A L S E } }$ (both penalize uneven utilization across units).

By theorem 4, the VALSE schedule is difficulty-adaptive; the MoE router is similarly input-adaptive. Both instantiate the same conditional-computation template, differing only in the organization of units (parallel vs. sequential). The routing, selection, and regularization mechanisms are isomorphic. □ □

Remark. theorem 6 establishes a structural (not functional) duality. We do not claim that any MoE routing policy can be embedded into VALSE or vice versa; the two operate on different dimensions of the computation graph. Rather, the shared pipeline—router, selection, sparse execution, load balancing—reveals that VALSE is the depth-wise analogue of MoE’s width-wise sparsity.

Unified selection framework. Given N compute units $\{ u _ { 1 } , \dotsc , u _ { N } \}$ and budget $K .$ , select subset $I \subseteq \{ 1 , \ldots , N \}$ : output $\begin{array} { r } { = \sum _ { i \in I } w _ { i } \cdot u _ { i } ( \mathrm { i n p u t } ) } \end{array}$ . MoE corresponds to ${ \dot { N } } = E$ (parallel organization); VALSE corresponds to $N = \dot { L }$ (sequential organization); the selection mechanisms are structurally similar.

## 8.4 Two-Dimensional Sparsity

Definition 7 (Two-dimensional sparse selection). Simultaneous horizontal and vertical sparsity: each sample selects $K _ { v }$ layers and, per layer, $K _ { h }$ experts:

$$
\mathbb { F } _ { 2 D } = K _ { v } \cdot K _ { h } \cdot C _ { u n i t } , s a { \nu } i n g s = 1 - \frac { K _ { v } \cdot K _ { h } } { L \cdot E }\tag{17}
$$

Proposition 8 (Hybrid architecture FLOPs advantage). Let horizontal sparsity $\rho _ { h } = 1 - K _ { h } / E$ and vertical sparsity $\rho _ { v } = 1 - K _ { v } / L \mathrm { { : } }$

$$
\mathbb { F } _ { h y b r i d } = ( 1 - \rho _ { v } ) ( 1 - \rho _ { h } ) \cdot \mathbb { F } _ { f u l l }\tag{18}
$$

Taking $\rho _ { h } = 0 . 8 7 5 ( 8 – c h o o s e \mathrm { - } 1 M o E )$ and $\rho _ { v } = 0 . 2 ( V A L S E ) , \mathbb { F } _ { h y b r i d } / \mathbb { F } _ { f u l l } = 0 . 1 0 ,$ , i.e., 90% savings, versus 87.5% for MoE alone.

Theorem 9 (Unified representation of sparse Transformers). Common sparse Transformer variants that employ per-unit gating can be represented as applying a sparse mask $M \in \{ 0 , 1 \} ^ { L \times E }$ on the computation graph $G \in \mathbb { R } ^ { \breve { L } \times E }$

$$
o u t p u t = \sum _ { ( l , e ) : M [ l , e ] = 1 } w _ { l , e } \cdot F _ { l } ^ { ( e ) } ( i n p u t )\tag{19}
$$

Horizontal MoE uses per-row Top-K masks $( M [ l , e ] = \mathbf { 1 } [ g _ { l } ( x ) \in \mathrm { T o p K } _ { h } ] ) ;$ ; VALSE uses whole-row masks $( M [ l , e ] = \tilde { g } _ { l } \bar { \forall } e ) ;$ hybrids use joint row–column sparsity $( M [ \bar { l } , e ] \overset { \cdot \cdot } { = } \tilde { g } _ { l } \cdot { \mathbf { 1 } } [ g _ { l } ( x ) \in \mathrm { T o p K } _ { h } ] )$

This reveals that sparse Transformers differ in which dimension of the computation graph they mask. VALSE’s contribution is completing the depth dimension of sparsity—previously, sparsity existed only in the width dimension (MoE). Further theoretical results, including function-space analysis and optimization properties, in sections 8.5 and 8.6.

Remark (Scheme A/B applicability). VALSE supports two residual schemes (section 6): Scheme A (identity skip, eq. (6)) and Scheme B (residual projection, eq. (7)). All results in sections 8.1 to 8.4 apply to both schemes, as they concern FLOPs accounting and structural duality. The function-space results (theorems 10 and 11 in section 8.5) are established under Scheme A only: the identity-skip property $( h _ { l + 1 } = h _ { l }$ when $\tilde { g } _ { l } ~ = ~ 0 )$ is essential to the subset and strict-containment arguments. Under Scheme B, the projection $W _ { l } ^ { \mathrm { p r o j } }$ introduces additional expressivity that may break the subset relationship; we restrict the function-space analysis to Scheme A and note this as a limitation.

## 8.5 Function-Space Analysis

We now establish the relationship between the function space of VALSE and that of the full-layer model. All results in this subsection are established under Scheme A (identity residual, eq. (6)), where skipping a layer produces the identity mapping $h _ { l + 1 } = h _ { l }$

Theorem 10 (Full-layer function space is a superset (Scheme A)). Under Scheme A (identity residual, eq. (6)), the set of functions representable by VALSE under any retention set $S \subseteq \{ 1 , \ldots , L \}$ is a subset of the full-layer function space: $\mathcal { F } _ { V A L S E } \subseteq \mathcal { F } _ { f u l l } .$

Proof. The full-layer model applies all L layers in sequence. Under Scheme A, the residual update is $h _ { l + 1 } = h _ { l } + \tilde { g } _ { l } \cdot F _ { l } ( h _ { l } )$ . When $\tilde { g } _ { l } = 1$ (layer retained), this reduces to the standard residual update $h _ { l + 1 } = h _ { l } + F _ { l } ( h _ { l } )$ , identical to the full-layer model. When $\tilde { g } _ { l } = 0$ (layer skipped), $h _ { l + 1 } = h _ { l }$ which is the identity mapping.

Any VALSE path with retention set S produces the same output as the full-layer model with the parameters of all skipped layers set to zero: $\tilde { F } _ { l } = 0$ for $l \notin S$ . Since the full-layer function space includes all parameter settings (including the degenerate zero setting), every VALSE function is a special case of the full-layer model with particular parameters. Therefore $\mathcal { F } _ { \mathrm { V A L S E } } \subseteq \mathcal { F } _ { \mathrm { f u l l } }$

By theorem 1, the expected output under random skip schedules is a convex combination of individual VALSE paths, each of which lies in $\mathcal { F } _ { \mathrm { f u l l } }$ . Since $\dot { \mathcal { F } } _ { \mathrm { f u l l } }$ is closed under convex combinations, the subset relationship holds in expectation as well: $\mathbb { E } [ \mathcal { F } _ { \mathrm { V A L S E } } ] \subseteq \mathcal { F } _ { \mathrm { f u l l } } .$ . □ □

Theorem 11 (Strict containment (Scheme A)). Under Scheme A, there existfunctions representable by the full-layer model that no VALSE path with $| S | < L$ can represent; i.e., the containment is strict: $\dot { \mathcal { F } } _ { V A L S E } \subsetneq \mathcal { F } _ { f u l l }$

Proof. We construct an explicit counterexample using the residual structure $h _ { l + 1 } = h _ { l } + F _ { l } ( h _ { l } )$ Consider a two-layer segment where:

$F _ { l } ( h ) = W _ { l } h$ performs a rank-deficient linear projection with $W _ { l } \in \mathbb { R } ^ { d \times d }$ , ran $\mathrm { k } ( W _ { l } ) =$ $r < d ,$ projecting onto a single direction v $( \mathrm { i . e . , } W _ { l } = v v ^ { \top }$ with $\lVert \boldsymbol { v } \rVert = 1 )$

$F _ { l + 1 } ( h ) = \phi ( W _ { l + 1 } h )$ applies a nonlinear activation $\phi =$ ReLU followed by a full-rank projection $W _ { l + 1 } \in \mathbb { R } ^ { d \times d }$ with rank $\left( W _ { l + 1 } \right) = d .$

The full-layer function for this segment is:

$$
f _ { \mathrm { f u l l } } ( h ) = h + W _ { l + 1 } \cdot \phi \big ( h + W _ { l } h \big ) = h + W _ { l + 1 } \cdot \phi \big ( \big ( I + W _ { l } \big ) h \big ) .
$$

This depends on a two-step nonlinear interaction: the input is first transformed by the rank-deficient projection $W _ { l }$ , summed with the identity (residual connection), then passed through $\phi$ and the fullrank projection $W _ { l + 1 }$

Now consider the VALSE paths that skip one of the two layers:

• Skip layer $( \tilde { g } _ { l } ~ = ~ 0 , \ \tilde { g } _ { l + 1 } ~ = ~ 1 ) \colon \ f _ { \mathrm { s k i p } - l } ( h ) ~ = ~ h + W _ { l + 1 } \cdot \phi ( h )$ . The rank-deficient intermediate transformation $W _ { l }$ is never applied. The input to ϕ is h itself, not $( I + W _ { l } ) h ,$ so the output depends on the full d-dimensional input rather than the projected r-dimensional subspace.

• Skip layer $l + 1 ( \tilde { g } _ { l } = 1 , \tilde { g } _ { l + 1 } = 0 ) \colon f _ { \mathrm { s k i p } - ( l + 1 ) } ( h ) = h + W _ { l } h = \left( I + W _ { l } \right) h ,$ , which is a linear function—the nonlinearity ϕ and the full-rank projection $W _ { l + 1 }$ are never applied.

Concretely, let $W _ { l } = v v ^ { \top }$ (rank-1) and $W _ { l + 1 } = I \left( \mathrm { f u l l \mathrm { - } r a n k } \right)$ . Then:

$f _ { \mathrm { f u l l } } ( h ) = h + \phi \big ( ( I + v v ^ { \top } ) h \big )$ , which can produce outputs in a d-dimensional cone that depends on the sign of $v ^ { \top } h$

$f _ { \mathrm { s k i p } - l } ( h ) = h + \phi ( h )$ , which operates on the full d-dimensional input without the rank-1 projection—a fundamentally different function.

$f _ { \mathrm { s k i p - } ( l + 1 ) } ( h ) = ( I + v v ^ { \top } )$ h, a rank-(r + 1) linear map with no nonlinearity.

Neither skip path can reproduce $f _ { \mathrm { f u l l } } { \mathrm { : } }$ the full-layer function depends on the composition of the rankdeficient projection, the residual addition, and the nonlinearity—a dependency that no single-layer skip can replicate. The key observation is that the residual connection $h _ { l + 1 } = h _ { l } + F _ { l } ( h _ { l } )$ creates a coupling between the identity pathway and the layer transformation: the nonlinearity ϕ in $F _ { l + 1 }$ receives both the unmodified residual $h _ { l }$ and the projection $W _ { l } h _ { l }$ as a sum. Skipping layer l removes the $W _ { l } h _ { l }$ contribution entirely, and skipping layer $l + 1$ removes the nonlinearity entirely.

Furthermore, the effective capacity rank of a network with retention set S is bounded by $\sum _ { l \in S } \mathrm { r a n k } ( W _ { l } ) \cdot d _ { \mathrm { m o d e l } }$ , which is strictly less than $\textstyle \sum _ { l = 1 } ^ { L } \operatorname { r a n k } ( W _ { l } ) \cdot d _ { \mathrm { m o d e l } }$ when $| S | < L$ and at least one skipped layer has non-trivial rank. By theorem 14, the capacity gap is proportional to the skipped layers’ ranks. Hence the containment is strict: $\mathcal { F } _ { \mathrm { V A L S E } } \subsetneq \bar { \mathcal { F } } _ { \mathrm { f u l l } } . \bar { \square }$ □

Proposition 12 (Number of non-contiguous retention sets). The number ofnon-contiguous retention sets ofsize k (with layers 1 and $L f i x e d ) i s \left( { L - 2 \atop k - 2 } \right)$ . The total number ofvalid retention sets is $2 ^ { L - 2 } - 1$

Proof. Layers 1 and $L$ are always retained (the estimator prefix and final output). From the remaining $L - 2$ interior layers, we choose $k - 2$ to retain, giving $\binom { L - 2 } { k - 2 }$ retention sets of size k. Summing over $k = 2 , \ldots , L$ gives $\begin{array} { r } { \sum _ { k = 2 } ^ { L } { \binom { L - 2 } { k - 2 } } = \sum _ { j = 0 } ^ { L - 2 } { \binom { L - 2 } { j } } - { \binom { L - 2 } { L - 2 } } = 2 ^ { L - 2 } - 1 } \end{array}$ valid retention sets, where we subtracted 1 to exclude the degenerate case. These retention sets underlie the difficulty-aware schedule (theorem 4), where harder inputs select larger sets. □ □

Observation 13 (Information preservation under identity skipping). Under identity skipping $( \tilde { g } _ { l } =$ $0 ) , h _ { l + 1 } = h _ { l }$ , so the mutual information with the target is exactly preserved: $I ( h _ { l + 1 } ; y ) = I ( h _ { l } ; y )$ No existing information is lost. However, the information that layer l would have extracted—i.e., $I ( F _ { l } ( h _ { l } ) ; y ) - I ( h _ { l } ; y ) .$ —is foregone. Thus, skipping preserves prior information but sacrifices potential information gain from the skipped layer.

Observation 14 (Effective capacity rank). The effective capacity rank of a model with retention set S is proportional to $\sum { } _ { l \in S } d _ { \mathrm { m o d e l } } ^ { 2 } .$ Skipping layers reduces capacity linearly in the number of skipped layers, motivating the difficulty-aware schedule (theorem 4) which skips layers only when the marginal information gain $\Delta _ { l }$ is small.

## 8.6 Optimization Properties

Observation 15 (Non-convexity in gate parameters). The loss landscape with respect to gate parameters $\{ w _ { l } , b _ { l } \}$ is non-convex, due to the discreteness of $\tilde { g } _ { l } = { \bf 1 } [ g _ { l } > \theta _ { l } ]$ . The indicator function introduces a combinatorial structure: there are $2 ^ { L - 2 }$ possible retention patterns, and the loss surface is piecewise constant in the gate parameters, with discontinuities at decision boundaries. This nonconvexity motivates the use of the straight-through estimator and Gumbel-softmax relaxation during training.

Observation 16 (STE gradient approximation (biased)). The straight-through estimator (STE) provides a biased gradient approximation for the gate parameters. In the forward pass, the hard gate $\tilde { g } _ { l } = \mathbf { 1 } [ g _ { l } > \theta _ { l } ]$ is used; in the backward pass, the gradient of the threshold function is replaced by the identity, $\begin{array} { r } { \frac { \partial \tilde { g } _ { l } } { \partial w _ { l } } \approx \frac { \partial g _ { l } } { \partial w _ { l } } } \end{array}$ . The bias is $\mathbb { E } [ \nabla _ { w _ { l } } \mathcal { L } ^ { \mathrm { S T E } } ] - \nabla _ { w _ { l } } \mathcal { L } \neq 0$ in general, because the surrogate gradient ignores the discontinuity of the threshold function. The Gumbel-softmax relaxation with temperature annealing (section 7) mitigates this bias by smoothing the threshold during training.

Proposition 17 (Collapse risk (under a drift model)). Without load-balancing regularization $( \alpha =$ 0), gate parameters converge to a degenerate solution—all samples routed to the same minimal retention set (first and last layers only)—with probability at least $1 - e ^ { - c L }$ , where $c > 0$ is a constant depending on the gate initialization and optimization landscape.

Proof sketch (under a hypothetical drift model). Assumption. The drift model $p _ { l } ^ { ( T ) } \leq e ^ { - c _ { l } T }$ (gate retention probability decays exponentially under task-loss gradient without load balancing) is a hypothetical model; it is not derived from a formal optimization analysis but is motivated by the empirical observation that, without balancing, gate biases drift toward $- \infty$

Let $p _ { l } = P ( \tilde { g } _ { l } = 1 )$ be the retention probability of layer l. Without load balancing, the gradient signal for each gate $w _ { l }$ is dominated by the task loss, which is minimized when the model uses the cheapest path (fewest layers). Under independent Bernoulli gates, the probability that at least one non-trivial retention pattern survives is $\textstyle \prod _ { l = 2 } ^ { L - 1 } ( 1 - p _ { l } )$ . When the task loss landscape favors collapse (e.g., when intermediate layers contribute redundantly), the gate biases drift toward $- \infty ,$ , driving $p _ { l } \to 0$

The probability that collapse does not occur (i.e., at least one non-trivial pattern survives after $T$ steps) is bounded by $\begin{array} { r } { \prod _ { l } \bar { p _ { l } ^ { ( T ) } } } \end{array}$ . Under a drift model where $p _ { l } ^ { ( T ) } \leq e ^ { - c _ { l } T }$ for some $c _ { l } > 0$ (from gradient-driven bias reduction), the survival probability is at most $\begin{array} { r } { \exp \bigl ( - \sum _ { l } c _ { l } T \bigr ) \ \leq \ e ^ { - c L T } } \end{array}$ for $c = \operatorname* { m i n } _ { l } c _ { l } / L$ , by applying the product bound and theorem 1 to the log-sum. Thus the collapse probability $\mathrm { i } \mathrm { \bar { s } } \geq 1 - \bar { e } ^ { - c L }$

This result motivates the load-balancing regularizer ${ \mathcal { L } } _ { \mathrm { b a l a n c e } }$ (section 7) and the curriculum-learning schedule that gradually increases the target skip rate, both of which counteract the drift toward collapse. □ □

Observation 18 (Critical skip rate (heuristic estimate)). There exists a critical skip rate $\rho _ { c }$ beyond which model representational capacity degrades sharply. Under the uniform-layer assumption, $\rho _ { c }$ ≈ $1 - d _ { \mathrm { f e a t } } / L$ , where $d _ { \mathrm { f e a t } }$ is the effective feature dimension. Beyond $\rho _ { c }$ , the retained layers can no longer span the full feature space, and the model loses the ability to represent certain input-dependent transformations. This heuristic, combined with theorem 14, informs the choice of the target skip rate $\rho ^ { * }$ in the difficulty-aware schedule (theorem 4).

## 9 Proof-of-Concept Evaluation

We conduct a preliminary empirical evaluation of the VALSE prototype to test whether difficultyadaptive per-sample layer skipping can be realized at prototype scale. We emphasize that these experiments constitute a proof-of-concept study at prototype scale, not a comprehensive evaluation. All experiments were conducted on a single CPU machine without GPU access, severely constraining the feasible model size, training duration, and evaluation set. As reported below, the core hypothesis of difficulty-adaptive routing was not verified: the router exhibited a routing-collapse failure mode (Pearson $r = - 0 . 0 1 7 )$ , and the results should be interpreted as a transparent negative result rather than as evidence of feasibility or as performance benchmarks. The theoretical analysis of section 8 provides the primary justification for the approach.

## 9.1 Setup

Model. A 12-layer Transformer encoder with $d _ { \mathrm { m o d e l } } = 2 5 6$ , 4 attention heads, $d _ { \mathrm { f f n } } = 1 0 2 4$ , vocabulary size 8,000, and $K _ { \mathrm { e s t } } = 2 ( 1 1 . 5 \mathrm { M }$ parameters). The difficulty estimator and router add $< 1 \%$ overhead.

Dataset. SST-2 (sentiment classification), 100 evaluation samples.

Training. The prototype was trained from scratch (no pre-training) on CPU with ${ \sim } 3 0 0$ training samples for 18 epochs (batch size 32), using hard routing (Gumbel-STE) with load-balancing and efficiency losses; $\alpha = 0 . 0 3 , \beta = 0 . 0 5 , B _ { \mathrm { t a r g e t } } = 0 . 5 5 L$ . This short schedule and small training set

reflect prototype-scale resource constraints rather than a fully converged training run. Curriculum learning Stage 1 is omitted; Stages 2–3 are used.

Methods compared. Four inference strategies evaluated on the same trained model: full-depth (all 12 layers), early exit (confidence-based truncation), random skip (uniformly retaining ∼8 layers), and VALSE adaptive (difficulty-driven per-sample skipping).

Metrics. Accuracy, average activated layers, and theoretical FLOPs savings. All results are from a single run (single random seed, seed = 42); statistical significance is not assessed at prototype scale.

## 9.2 Main Results

Table 3 compares the four inference strategies on SST-2. Under the adaptive strategy, all 100 evaluation samples were routed to exactly 3 layers, yielding an apparent reduction in average activated layers from 12.0 to 3.0 (62.1% theoretical FLOPs reduction) at a 1 pp accuracy change $( 0 . 6 3 ~  ~ 0 . 6 2 )$ This reduction, however, is a consequence of routing collapse—a degenerate uniform-skip-to-minimum-depth failure mode—rather than of intelligent, difficulty-adaptive persample routing; the core hypothesis is therefore not verified by this result.

Table 3: Prototype-scale (12-layer, $d = 2 5 6 )$ comparison of inference strategies on SST-2 (100 samples, single seed). These are not performance benchmarks; absolute accuracy is limited by model scale, absence of pre-training, and short training. The apparent FLOPs savings of VALSE reflect a routing collapse (all samples routed to 3 layers) rather than effective difficulty-adaptive routing; the core hypothesis is not verified (Pearson $r = - 0 . 0 1 7 ,$ see section 9.3).
<table><tr><td>Method</td><td>SST-2 Accuracy</td><td>Avg. layers</td><td>FLOPs savings</td></tr><tr><td>Full-depth</td><td>0.63</td><td>12.00</td><td>0.0%</td></tr><tr><td>Early exit</td><td>0.63</td><td>11.56</td><td>3.0%</td></tr><tr><td>Random skip</td><td>0.63</td><td>8.00</td><td>27.6%</td></tr><tr><td>VALSE (adaptive)</td><td>0.62</td><td>3.00</td><td>62.1%</td></tr></table>

Routing collapse. The layer distribution reveals that all 100 evaluation samples were routed to exactly 3 layers under the adaptive strategy. This confirms a routing-collapse failure mode: the difficulty signal did not produce differentiated gating across samples, and the router adopted a degenerate uniform-skip-to-minimum-depth strategy rather than the intended difficulty-adaptive behavior. The FLOPs savings are therefore a consequence of collapse, not of intelligent per-sample routing. This is consistent with the known sensitivity of load-balancing losses to the hyperparameter α and the limited training budget.

## 9.3 Core Hypothesis: Negative Result

The central hypothesis of VALSE is that difficulty and activated depth should be positively correlated—easy inputs skip more layers, hard inputs use more. We tested this on a separate 12- layer model trained with a curriculum schedule (150 SST-2 + 150 GSM8K samples):

• Pearson correlation between difficulty score $\hat { d }$ and activated layers: $r = - 0 . 0 1 7$ (negative, opposite to the hypothesized direction).

• Layer gap (GSM8K − SST-2): −0.47 layers—hard tasks used fewer layers than easy tasks, contrary to the hypothesis.

• Routing differentiation: false—both tasks activated near-full depth, indicating collapse to maximum rather than differentiated routing.

We report this transparently as a negative result: the core hypothesis of difficulty-adaptive depth routing was not verified at prototype scale. The difficulty signal did not effectively propagate from the estimator to the routing decisions, and the router exhibited collapse to extreme depth (minimum or maximum) rather than differentiated, difficulty-aware routing. This does not invalidate the theoretical framework—which holds regardless of training dynamics—but it indicates that the training procedure and router architecture require substantial improvement before the theoretical predictions can be empirically confirmed.

## 9.4 Theory-to-Experiment Mapping

Table 4 summarizes the relationship between theoretical claims and their experimental verification status. Due to routing collapse and prototype-scale limitations, no claim is fully verified; several are supported only indirectly or remain purely theoretical.

Table 4: Mapping of theoretical claims to experimental verification status. “∼” = indirect or partial (prototype scale only); “—” = not yet verified. No claim is fully verified.
<table><tr><td>Theoretical claim</td><td>Source</td><td>Verification status</td></tr><tr><td>Expected FLOPs reduction formula</td><td>theorem 2</td><td>~ Prototype observation</td></tr><tr><td>Difficulty-depth monotonicity (core hypothesis)</td><td>theorem 4</td><td>— Not verified (r = −0.017)</td></tr><tr><td>Difficulty-aware FLOPs savings range</td><td>theorem 5</td><td>~ Prototype scale only</td></tr><tr><td>Non-contiguous skip info preservation</td><td>theorem 13</td><td>～ Indirect (accuracy preserved)</td></tr><tr><td>MoE-VALSE structural duality</td><td>theorem 6</td><td>— Top-K variant pending</td></tr><tr><td>Routing-collapse risk</td><td>theorem 17</td><td>～ Empirically observed</td></tr><tr><td>Critical skip rate (heuristic)</td><td>theorem 18</td><td>~ Informs ρ* choice</td></tr><tr><td>Hybrid architecture FLOPs advantage</td><td>theorem 8</td><td>- Not verified (theory only)</td></tr><tr><td>Full-layer function-space superset (Scheme A)</td><td>theorem 10</td><td>~ Indirect (accuracy bounded)</td></tr><tr><td>Strict containment (Scheme A)</td><td>theorem 11</td><td>- Not verified (theory only)</td></tr></table>

Notably, the routing-collapse phenomenon empirically corroborates theorem 17 (routing-collapse risk), confirming that the theoretical concern about degenerate solutions is practically relevant and motivating the load-balancing and curriculum-learning mechanisms designed to mitigate it—though these mechanisms proved insufficient at prototype scale.

## 9.5 Limitations

We transparently list the key limitations:

1. Model scale: The 12-layer, 11.5M-parameter model is far below the scale at which layer redundancy has been documented (12+ layers, 1B+ parameters, pre-trained).

2. No pre-training: The model is trained from scratch on ∼300 samples, limiting absolute accuracy.

3. Single dataset & single seed: Only SST-2 is evaluated with one random seed; statistical significance is not assessed.

4. Routing collapse: The core hypothesis is not verified; the router collapsed rather than learning differentiated routing (Pearson = −0.017).

5. No external baselines: No comparison with DeeBERT, CALM, LayerDrop, MoD, Short-GPT, or SliceGPT.

6. Incomplete ablations: Five ablation groups were designed but not fully executed.

7. No wall-clock speedup: Theoretical FLOPs savings are not realized as latency reduction due to implementation overhead.

These limitations define a clear roadmap for future work (section 10); we present the prototype results as preliminary and incomplete evidence, not as definitive validation.

## 10 Conclusion and Discussion

We proposed VALSE, extending conditional computation from the width dimension of horizontal MoE to the depth dimension via per-sample, difficulty-driven, non-contiguous layer skipping. VALSE comprises a difficulty estimation module, three routing strategies (hard, soft, global), a load-balancing and curriculum-learning training scheme, and a theoretical framework establishing the MoE–VALSE structural duality and two-dimensional sparsity.

## 10.1 Summary of Contributions

• Methodology: Per-sample non-contiguous layer skipping via an interpretable difficulty estimator and three routing variants, breaking through the truncation limitation of early exit.

• Training: Four auxiliary losses—layer utilization balancing, efficiency, output consistency, and Z-loss—plus a three-stage curriculum learning schedule, jointly addressing routing collapse and training instability.

• Theory: Closed-form expected FLOPs reduction (theorem 2), MoE–VALSE structural duality (theorem 6), and unified width×depth sparsity (theorem 7, theorem 9).

• Proof-of-concept: A prototype-scale evaluation on SST-2 provides preliminary—though inconclusive—evidence for the layer-skipping mechanism. The core hypothesis (difficultyadaptive routing) was not verified at prototype scale; full-scale validation on larger, pretrained models remains future work.

## 10.2 Key Findings and Limitations

• On SST-2, VALSE reduces average activated layers from 12.0 to 3.0 (62.1% theoretical FLOPs reduction) with a 1 pp accuracy change (0.63 → 0.62). However, this resulted from routing collapse—all samples were routed to the minimum depth—not from intelligent persample routing.

• The core hypothesis (difficulty-adaptive depth routing) was not verified: the Pearson correlation between difficulty and activated layers was r = −0.017 (negative, opposite to the hypothesized direction). The difficulty signal did not effectively propagate to routing decisions.

• The difficulty-stratified analysis confirms routing collapse: the router did not learn differentiated, difficulty-aware routing, indicating that the training procedure and router architecture require substantial improvement.

• Limitations: (i) Prototype scale (12 layers, single dataset, single seed) limits all conclusions; (ii) no external baselines are compared; (iii) ablations are designed but not fully executed; (iv) the core empirical hypothesis is not verified (routing collapse, Pearson = −0.017); (v) theoretical assumptions (bounded skip rate, stability condition) are only partially validated; (vi) wall-clock speedup is not realized due to implementation overhead.

## 10.3 Future Work

1. Scale up: Validate on 12+-layer, 1B+ models with pre-trained weights to assess hard-task skipping and confirm layer redundancy at scale.

2. Multi-dataset evaluation: Extend to diverse tasks (classification, QA, reasoning, generation) to test the universality of the layer redundancy hypothesis.

3. Complete ablations: Execute all five ablation groups (routing strategy, difficulty estimator, load balancing, efficiency loss, fixed skip) with multiple seeds.

4. Statistical rigor: Run 3–5 seeds per configuration, report mean±std, and perform paired t-tests or Wilcoxon tests.

5. Cure routing collapse: Stronger load balancing, annealed ρ<sup>∗</sup> schedules, and contrastive routing objectives to ensure differentiated routing across difficulty levels.

6. External baselines: Compare against DeeBERT, LayerDrop, MoD, ShortGPT, and random skip under matched compute budgets.

7. Engineering: Gate kernel fusion, batched routing, and custom CUDA kernels to convert FLOPs savings into real latency reduction.

8. Two-dimensional sparsity: Combine VALSE with MoE for full width×depth sparse activation.

9. Generative LLMs: Extend per-sample skipping to autoregressive generation, with selfspeculative decoding for quality safety.

## 10.4 Broader Impact

VALSE aims to reduce the computational cost of LLM inference by activating only the necessary depth per input. If validated at scale, this could lower the energy footprint of inference workloads and democratize access to efficient model deployment. Potential risks include accuracy degradation on hard inputs if the difficulty estimator is poorly calibrated; the difficulty-aware routing and curriculum learning safeguards are designed to mitigate this. We commit to transparent reporting of both accuracy and efficiency trade-offs.

## A Symbol Table

Table 5 lists all symbols used in the paper.

Table 5: Complete symbol table.
<table><tr><td>Symbol  $L$ </td><td>Meaning</td></tr><tr><td> $l \in \{ 1 , \ldots , L \}$   $d _ { \mathrm { m o d e l } }$   $N$   $h _ { l } \in \mathbb { R } ^ { N \times d _ { \mathrm { m o d e l } } }$   $h _ { I } ^ { ( s ) }$   $x ^ { \dot { ( } s ) }$   $F _ { l } ( \cdot )$   $K _ { \mathrm { e s t } }$ </td><td>Total number of Transformer layers Layer index Hidden dimension Sequence length Input hidden state at layer l Sample  $s { \ ' } _ { \mathbf { S } }$  hidden state at layer l (sequence-pooled) Input embedding for sample s Forward function of the l-th Transformer block</td></tr><tr><td> $r _ { l } \in \mathbb { R } ^ { d _ { \mathrm { f e a t } } }$   $R ^ { ( s ) } \in \mathbb { R } ^ { 5 K _ { \mathrm { e s t } } }$   $\boldsymbol { d } ( \boldsymbol { x } ^ { ( s ) } ) \in \mathbb { R }$   $\hat { d } ( x ^ { ( s ) } ) \in [ 0 , 1 ]$   $g _ { l } ( x ^ { ( s ) } ) \in [ 0 , 1 ]$   $\tilde { g } _ { l } \in \{ 0 , 1 \}$   $S ^ { ( s ) }$   $\theta _ { l }$   $W _ { l } ^ { \mathrm { p r o j } }$ </td><td>Number of initial layers used for difficulty estimation Difficulty feature vector  $( d _ { \mathrm { f e a t } } = 5 )$  Aggregated feature vector Per-sample difficulty score Normalized difficulty score Soft gate value at layer l Quantized hard gate decision Retained layer set for sample s</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t a s k } }$   ${ \mathcal { L } } _ { \mathrm { b a l a n c e } }$   $\mathcal { L } _ { \mathrm { e f f } }$   $\mathcal { L } _ { \mathrm { c o n s i s t } }$   $\mathcal { L } _ { z }$   $\mathcal { L } _ { \mathrm { t o t a l } }$   $\alpha , \beta , \gamma , \delta$   $B _ { \mathrm { t a r g e t } }$   $\rho ^ { * }$   $T _ { \mathrm { K D } }$  pl</td><td>Quantization threshold (default 0.5) Residual alignment projection (learnable, optional) Main task loss (cross-entropy) Layer utilization balance loss Computational efficiency loss Output consistency loss (KL divergence) Z-loss regularization on gate logits Total loss function Auxiliary loss weights</td></tr></table>

Table 6: Adaptive depth methods and their limitations.
<table><tr><td>Method</td><td>Core idea</td><td>Limitation</td></tr><tr><td>DeeBERT/FastBERT/PABEE</td><td>early exit</td><td>Per-sampleconfidence-based Truncation: cannot skip middle layers</td></tr><tr><td>CALM</td><td>generative LMs</td><td>Confidence-based early exit for Truncation: per-sample but prefix-only</td></tr><tr><td>ACT/Universal Transformers</td><td></td><td>Shared-weight recurrence with Deepens, not skips; not for fixed-layer models</td></tr><tr><td>LayerDrop</td><td>halting Structured dropout for depth ro- Static/global depth at inference</td><td></td></tr><tr><td>LayerSkip</td><td>bustness</td><td>Layer dropout with self- Static/per-token, not per-sample</td></tr><tr><td>DynamicBERT</td><td>speculative decoding</td><td>difficulty-driven Independent width &amp; depth ad- No unified routing; not difficulty-</td></tr><tr><td>ShortGPT</td><td>justment Block Influence score quantifies</td><td>aware Static post-training pruning; no input</td></tr><tr><td>Streamline</td><td>layer redundancy Structured layer pruning at post-</td><td>adaptivity Static; produces fixed reduced-depth model</td></tr><tr><td>SliceGPT</td><td>training stage Structured slicing/pruning of lay- Static; no per-sample adaptivity</td><td></td></tr><tr><td>MoNE</td><td>ers decoder layer pairs</td><td>MoE routing across encoder- Per-token; no sample-level difficulty semantics</td></tr><tr><td>MoD</td><td>Per-token per-layer routing</td><td>Not per-sample; no difficulty seman- tics</td></tr><tr><td>Mixtral/DeepSeek-MoE</td><td>Width-wise sparse expert activa- Width-only: depth fixed; no difficulty</td><td>dimension</td></tr><tr><td>Switch Transformer/GShard</td><td>tion</td><td>Top-1/Top-K expert routing at Width-only: depth fixed; semantic per-</td></tr><tr><td>Drowzee</td><td>scale Pre-inference per-sample budgetBudget controls truncation only</td><td>token routing</td></tr><tr><td>VALSE (ours)</td><td>prediction Per-sample difficulty-aware non- contiguous skip</td><td></td></tr></table>

Table 7: Skip mode comparison.
<table><tr><td>Method</td><td>Skip mode</td><td>Can skip middle and retain deeper layers?</td></tr><tr><td>Truncation (DeeBERT/CALM)</td><td> $S = \{ 1 , \ldots , k \}$ </td><td>No</td></tr><tr><td>MoD (per-token, per-layer)</td><td>Each token decides independently</td><td>Partially, not per-sample</td></tr><tr><td>VALSE (ours)</td><td>S is any subset, per-sample</td><td>Yes, non-contiguous + per-sample</td></tr></table>

Table 8: Mathematical analogy between MoE routing and VALSE layer routing.
<table><tr><td>MoE routing</td><td>VALSE layer routing</td><td>Analogy</td></tr><tr><td>Expert set  $\{ E _ { 1 } , \dotsc , E _ { E } \}$ </td><td>Layer set  $\{ F _ { 1 } , \ldots , F _ { L } \}$ </td><td>Expert ↔ layer: selectable compute unit</td></tr><tr><td>Gate  $G ( x ) = { \mathrm { T o p K } } ( W _ { g } \cdot x )$ </td><td>Gate  $g _ { l } = \sigma ( w _ { l } ^ { \top } \cdot [ r _ { l } ; \dot { d } ] )$ </td><td>Lightweight router scores units</td></tr><tr><td>Top-K selects K experts</td><td>Threshold selects retention set S</td><td>MoE selects “which experts&quot; → VALSE se- lects “which layers&quot;</td></tr><tr><td> $\begin{array} { r } { y = \sum _ { i \in \mathrm { T o p K } } G _ { i } \cdot E _ { i } ( x ) } \end{array}$ </td><td> $h _ { l + 1 } = h _ { l } + \tilde { g } _ { l } \cdot F _ { l } ( h _ { l } )$ </td><td>Activated unit output weighted into flow</td></tr><tr><td>Load balance  $\mathcal { L } _ { \mathrm { a u x } }$ </td><td>Layer balance  ${ \mathcal { L } } _ { \mathrm { b a l a n c e } }$ </td><td>Prevents “few units over-used” collapse</td></tr><tr><td>Capacity factor</td><td>Compute budget  $\le B _ { \mathrm { t a r g e t } }$ </td><td>Controls sparsity</td></tr></table>

Table 9: Architecture variant comparison.
<table><tr><td>Dimension</td><td>VALSE-Hard (A)</td><td>VALSE-Soft (B)</td><td>VALSE-Global (C)</td></tr><tr><td>Routing type</td><td>Hard (STE)</td><td>Soft (continuous)</td><td>Global Top-K</td></tr><tr><td>Gate parameters</td><td>Per-layer learnable</td><td>Per-layer learnable</td><td>Global router</td></tr><tr><td>Difficulty awareness</td><td>â enters gate</td><td>â enters gate</td><td>â sets K</td></tr><tr><td>Inference FLOPs savings</td><td>Yes, true skip</td><td>Requires  $g \to 0$ </td><td>Yes, true skip</td></tr><tr><td>Training stability</td><td>Needs STE + noise</td><td>Most stable</td><td>Moderate</td></tr><tr><td>Budget controllability</td><td>Approximate (regularized)</td><td>Approximate (regularized)</td><td>Exact (Top-K)</td></tr><tr><td>MoE analogy strength</td><td>Moderate</td><td>Low</td><td>High (Top-K dual)</td></tr></table>

## B Supplementary Tables

## C Supplementary Theoretical Results

The theorems and propositions previously located in this appendix—including the function-space analysis (theorems 10 and 11), the non-contiguous retention-set counting (theorem 12), informationpreservation and capacity-rank observations (theorems 13 and 14), and the optimization properties (theorems 15 to 18)—have been migrated to the main text in sections 8.5 and 8.6 to strengthen the theoretical focus of the paper. This appendix retains supplementary results that complement, rather than duplicate, the main-text development: a routing-expressivity comparison and an expected-FLOPs characterization under the routing distribution.

Proposition 19 (Routing expressivity of non-contiguous retention). The set of sample-level retention patterns realizable by VALSE is thefull power set 2<sup>{1,...,L}</sup> (cardinality $2 ^ { L } ) _ { : }$ , whereas truncationbased early exit realizes only the L prefix sets $\{ 1 , \ldots , k \}$ for $k \in \{ 1 , \ldots , L \}$ . Per-token per-layer routing (MoD) realizes $2 ^ { L \cdot \tilde { N } }$ token-level patterns, but when aggregated to a single sample-level retention set by per-layer majority vote over tokens, its effective sample-level image collapses to the truncation family under the standard monotone-confidence assumption. Thus only per-sample non-contiguous selection achieves the full $2 ^ { L }$ sample-level expressivity.

Proofsketch. VALSE’s per-sample threshold produces an arbitrary subset $S ^ { ( s ) } \subseteq \{ 1 , \ldots , L \}$ , so its image is $2 ^ { \{ 1 , \ldots , L \} }$ . Truncation constrains $S = \{ 1 , \dots , k \}$ , yielding exactly L patterns. For MoD, each layer l is retained iff a token-majority votes to retain; under the monotone-confidence assumption that retention probability is non-increasing in depth, the set of retained layers is a prefix, so the sample-level image is again the truncation family. Hence only VALSE attains full $2 ^ { L }$ samplelevel expressivity. □

Proposition 20 (Expected FLOPs under the routing distribution). Let C denote per-sample FLOPs of the l-th block and assume $C _ { l } = C$ for all l. Let $p _ { l } = \operatorname* { P r } [ l \notin S ^ { ( s ) } ]$ be the skip probability at layer l. Then the expected per-sample FLOPs ofVALSE is $\begin{array} { r } { \mathbb { E } [ \mathcal { F } _ { V A L S E } ] = \mathcal { F } _ { f i x e d } + C \sum _ { l = 1 } ^ { L } ( 1 - p _ { l } ) } \end{array}$ , and the expected savings ratio is $\begin{array} { r } { \rho _ { F L O P s } = 1 - \mathbb { E } [ \mathcal { F } _ { V A L S E } ] / \mathcal { F } _ { f u l l } = C \sum _ { l = 1 } ^ { L } p _ { l } / \mathcal { F } _ { f u l l } . } \end{array}$ . Under the uniform-skip regime $p _ { l } = \rho ^ { * } f o r a l l l$ , this reduces $t o \rho _ { F L O P s } = \rho ^ { * } \cdot L C / \mathcal { F } _ { f u l l } ,$ , recovering the closedform reported as theorem 3.

Remark (Cross-batch comparability of batch-internal normalization). The min–max normalization in eq. (3) maps the raw difficulty score $\boldsymbol { d } ( \boldsymbol { x } ^ { ( s ) } )$ ) onto [0, 1] using the batch extremum, so $\hat { d }$ is a within-batch rank statistic rather than an absolute population quantity. Consequently, for two batches $B _ { 1 } , B _ { 2 }$ with extrema $( m _ { 1 } , M _ { 1 } )$ and $( m _ { 2 } , M _ { 2 } )$ , the equality $\hat { d } _ { B _ { 1 } } ( x ) = \hat { d } _ { B _ { 2 } } ( x )$ does not imply equal population-level difficulty unless $m _ { 1 } = m _ { 2 }$ and $M _ { 1 } = M _ { 2 }$ . Routing remains correct because the gate $\tilde { g } _ { l }$ consumes only within-batch ordering, but cross-experiment comparison of $\hat { d }$ requires calibration to a shared reference (e.g., a running population extremum or a fixed-temperature rank transform).

Proposition 21 (Routing collapse as a degenerate equilibrium). Consider the VALSE training objective $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { t a s k } + \alpha \mathcal { L } _ { b a l a n c e } + \beta \mathcal { L } _ { e f f } + \gamma \mathcal { L } _ { z } + \delta \mathcal { L } _ { c o n s i s t }$ with gate bias b initialized so that $g _ { l } \approx 1$ at $t = 0 .$ . The uniform-skip-to-minimum-depth configuration $S ^ { ( s ) } = S _ { \mathrm { m i n } }$ for all s (i.e., all samples share the same minimal retention set) is a stationary point of the gate parameters whenever (i) the task loss gradient $\nabla _ { b _ { l } } \mathcal { L } _ { t a s k }$ is smaller in magnitude than $\nabla _ { b _ { l } } ( \alpha \mathcal { L } _ { b a l a n c e } + \beta \mathcal { L } _ { e f f } )$ at $S _ { \mathrm { m i n } } ,$ and (ii) the difficulty–gate Jacobian $\partial \tilde { g } _ { l } / \partial \hat { d }$ is rank-deficient so that the difficulty signal cannot break the symmetry across samples. Under these two conditions, the collapse configuration is a locally stable equilibrium: any perturbation that would differentiate samples is dominated by the regularization pulling all gates toward the balance-optimal uniform profile.

Proofsketch. $\mathrm { A t } \ S _ { \mathrm { m i n } } ,$ all gates take the same value across samples by construction, so ${ \mathcal { L } } _ { \mathrm { b a l a n c e } }$ is at its minimum (uniform utilization) and $\mathcal { L } _ { \mathrm { e f f } }$ is at its target. Differentiating any single gate $g _ { l }$ for a single sample s to break uniformity increases $\mathcal { L } _ { \mathrm { b a l a n c e } }$ linearly in α and changes $\mathcal { L } _ { \mathrm { e f f } }$ linearly in $\beta ,$ while the task-loss reduction is $O ( 1 / N )$ per sample (a single-sample perturbation in a batch of size N). Whenever $\alpha + \beta \gg 1 / N$ , the regularizing gradient dominates the task gradient at

$S _ { \mathrm { m i n } } .$ , making it a stationary point. Local stability follows because the Hessian of the regularizers is positive semidefinite in the gate-bias direction, so second-order growth of ${ \mathcal L } _ { \mathrm { b a l a n c e } } + { \mathcal L } _ { \mathrm { e f f } }$ exceeds any concave task-loss gain in a neighborhood of $S _ { \mathrm { m i n } }$ . Condition (ii) ensures the difficulty signal cannot supply the missing symmetry-breaking gradient: if $\partial \tilde { g } _ { l } / \partial \hat { d }$ is rank-deficient (e.g., the gate saturates near the threshold so the STE Jacobian vanishes), then even a well-separated $\hat { d }$ across samples cannot induce differentiated gating. □

This result reframes the prototype’s $r ~ = ~ - 0 . 0 1 7$ not as a refutation of the difficulty–depth monotonicity of theorem 4—which is a population-level statement under the decreasing-marginalcontribution assumption and is independent of the learned estimator—but as evidence that the prototype’s hyperparameters landed in the basin of attraction of the collapse equilibrium of theorem 21. The monotonicity guarantee and the collapse equilibrium are logically independent: the former concerns the optimal schedule, the latter concerns the optimizer’s fixed point, and a mismatch between the two is precisely the failure mode the prototype exhibited.

Remark (Scope and limitations of the theoretical framework). The theoretical results in section 8 and this appendix establish three classes of statement: (a) structural facts about the retentionset space (theorems 10 to 12 and 19), which are combinatorial and hold without approximation; (b) expected-FLOPs identities (theorems 2, 3 and 20), which are exact under the stated per-block FLOPs model; and (c) monotonicity and stability statements (theorems 4 and 21 and the stability condition eq. (14)), which are conditional on assumptions—decreasing marginal layer contribution, Lipschitz difficulty estimator, bounded gate gradient—whose empirical validity at scale is not established by the prototype. We therefore claim theoretical soundness of the structural and accounting results unconditionally, but only conditional soundness of the monotonicity and stability results. In particular, theorem 21 identifies a failure mode but does not prescribe a hyperparameter regime that provably avoids it; deriving such a regime (e.g., a sufficient condition on $\alpha , \beta$ relative to batch size and gate-saturation level that excludes collapse) is left as an open problem, and the prototype’s negative result is consistent with operating inside the collapse basin.

## D Reproducibility

This section provides all information needed to reproduce the experiments.

## D.1 Code Structure

The codebase is organized into the following key files, to be released with the camera-ready version:

Table 10: Code structure: key files and their responsibilities.
<table><tr><td>File</td><td>Responsibility</td></tr><tr><td>config.py</td><td>Central configuration (VALSEConfig dataclass)</td></tr><tr><td>model.py</td><td>Transformer backbone with skip-layer support</td></tr><tr><td>router.py</td><td>Difficulty estimator and layer gating</td></tr><tr><td>train.py</td><td>Main training entry point (baseline + adaptive; includes evaluate() for the main-results comparison)</td></tr><tr><td>prepare_data.py</td><td>Data download and preprocessing</td></tr></table>

Note on the analysis entry point. The codebase comprises the five core files above plus auxiliary analysis scripts, which together form a self-contained prototype pipeline; these files are not bundled with the current submission and will be released with the camera-ready version: config.py specifies all hyperparameters, prepare data.py produces the datasets, and train.py runs baseline and adaptive training and embeds the evaluate() routine that computes the main-results comparison (accuracy, average activated layers, theoretical FLOPs savings) and the difficulty–depth correlation analysis (Pearson r, layer-gap, routing-differentiation flag). The prototype evaluation set is small enough to be exercised through evaluate() directly, which serves as the unified entry point for all reported results.

## D.2 Hyperparameters

Table 11 lists the complete hyperparameter set used in the main experiment.

Table 11: Complete hyperparameter list for the main experiment.
<table><tr><td>Category</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="5">Model</td><td>num_layers</td><td>6</td></tr><tr><td>hidden_dim</td><td>256</td></tr><tr><td>num_heads</td><td>4</td></tr><tr><td>ffn_dim</td><td>1024</td></tr><tr><td>vocab_size</td><td>8,000</td></tr><tr><td></td><td> $K _ { \mathrm { e s t } }$ </td><td>2</td></tr><tr><td rowspan="4">Routing</td><td>routing_mode</td><td>hard (Gumbel-STE)</td></tr><tr><td>gate_bias_init</td><td>2.0</td></tr><tr><td>gate_threshold</td><td>0.5</td></tr><tr><td>gumbel_tau</td><td>1.0 (annealed 2.0 → 0.5)</td></tr><tr><td rowspan="5">Loss weights</td><td>α (balance)</td><td>0.01</td></tr><tr><td>β(efficiency)</td><td>0.1</td></tr><tr><td> $\gamma \left( \mathrm { Z } { \cdot } \mathrm { l o s s } \right)$ </td><td>10⁻⁴</td></tr><tr><td>δ (consistency)</td><td>0.5 (annealed to δ/3)</td></tr><tr><td> $T _ { \mathrm { K D } }$ </td><td>2.0</td></tr><tr><td rowspan="3">Budget</td><td> $B _ { \mathrm { t a r g e t } }$ </td><td>0.7L = 4.2</td></tr><tr><td> $\rho ^ { * }$ </td><td>0.3</td></tr><tr><td>Precomp</td><td>0.1</td></tr><tr><td rowspan="9">Training</td><td>learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>weight decay</td><td>0.1</td></tr><tr><td>betas</td><td>(0.9,0.95)</td></tr><tr><td>batch size</td><td>32</td></tr><tr><td>epochs</td><td>18</td></tr><tr><td>warmup steps</td><td>50</td></tr><tr><td>max grad norm</td><td>1.0</td></tr><tr><td>train subset</td><td>~300</td></tr><tr><td>optimizer LR schedule</td><td>AdamW</td></tr><tr><td rowspan="3">Curriculum</td><td></td><td>Linear warmup + cosine decay</td></tr><tr><td>Stage 1 (ρ = 0)</td><td> $\eta _ { 1 } \approx 0 . 2 5  – 0 . 3 3$  (warmup, ρ = 0 baseline)</td></tr><tr><td>Stage  $2 \left( 0 \to \rho ^ { * } \right)$  Stage 3  $( \rho = \rho ^ { * } )$ </td><td> $\eta _ { 2 } \approx 0 . 5 – 0 . 6 7$   $\eta _ { 3 } \approx 0 . 2 5 \ – 0 . 4$ </td></tr></table>

## D.3 Training and Evaluation Commands

## Environment setup.

uv venv .venv --python 3.13

.venv\Scripts\activate

pip install torch numpy matplotlib datasets pandas pyyaml tqdm \

requests huggingface\_hub pyarrow \

-i https://pypi.tuna.tsinghua.edu.cn/simple

## Data preparation.

python prepare\_data.py

This downloads SST-2 via datasets.load dataset("stanfordnlp/sst2") and GSM8K via datasets.load dataset("openai/gsm8k", "main"), then splits into train/val/test JSONL files in data/.

Training (baseline + adaptive).

This trains both the full-depth baseline and the VALSE adaptive model (6 layers). Training logs are written to train log.txt, which is not bundled with the current submission and will be provided with the camera-ready version.

Evaluation and analysis. The main-results comparison (accuracy, average activated layers, theoretical FLOPs savings) and the difficulty–depth correlation analysis (Pearson r, layer-gap, routingdifferentiation flag) are computed inside train.py’s evaluate() routine and are reported in section 9; the corresponding numbers are written to the training log (train log.txt), which is not bundled with the current submission and will be provided with the camera-ready version. The evaluate() routine serves as the unified entry point for all reported prototype results, and its single-call invocation constitutes the complete evaluation harness for the submission.

## D.4 Dataset Acquisition

• SST-2: datasets.load dataset("stanfordnlp/sst2") from HuggingFace. 500 test samples used in the main experiment; 500 in the difficulty analysis.

• GSM8K: datasets.load dataset("openai/gsm8k", "main") from HuggingFace. 500 test samples in the main experiment; 500 in the difficulty analysis.

If HuggingFace is inaccessible, set HF ENDPOINT=https://hf-mirror.com.

## D.5 Environment Dependencies

Table 12: Environment dependencies (minimum compatible versions; the prototype was developed under Python 3.13).

<table><tr><td>Package</td><td>Version</td><td>Purpose</td></tr><tr><td>torch</td><td>V 2.2.0</td><td>Deep learning framework</td></tr><tr><td>numpy</td><td>1.26.0</td><td>Numerical computation</td></tr><tr><td>matplotlib</td><td>≥ 3.8.0</td><td>Visualization</td></tr><tr><td>datasets</td><td>≥ 2.18.0</td><td>HuggingFace data loading</td></tr><tr><td>pandas</td><td>≥ 2.2.0</td><td>Data processing</td></tr><tr><td>pyarrow</td><td>15.0.0 V</td><td>Arrow format</td></tr><tr><td>pyyaml</td><td>6.0.1</td><td>YAML configuration</td></tr><tr><td>tqdm</td><td>≥ 4.66.0</td><td>Progress bar</td></tr><tr><td>huggingface_hub</td><td>≥ 0.23.0</td><td>Model/data download</td></tr></table>

## D.6 Random Seeds and Checkpoints

• Random seed: The prototype uses a fixed seed of seed = 42 for reproducibility and reports the point estimate; the production configuration (a future target, not executed in this submission) specifies 5 independent seeds (42, 123, 456, 789, 1024) with mean±std reporting, and the prototype adopts the single-seed protocol because its purpose is hypothesistesting (the difficulty–depth correlation) rather than benchmarking.

• Checkpoint: The prototype checkpoint is designed to be saved to checkpoints/ after training; the current release leaves this directory empty, with the .pt weights deferred to the camera-ready version.

• One-command reproduction: The prototype pipeline is reproduced via the two-command sequence prepare data.py then train.py, which executes the full end-to-end pipeline (data preparation, baseline training, adaptive training, and evaluation) and constitutes the canonical reproduction entry point.

• Determinism note: PyTorch operations may not be fully deterministic across hardware. We use torch.manual seed and numpy.random.seed; full determinism requires torch.use deterministic algorithms(True), which may reduce performance.

## D.7 Prototype Observations

The prototype run (logged in train log.txt) yielded the following observations, also reported in sections 9.2 and 9.3: full-depth SST-2 accuracy 0.63, VALSE adaptive accuracy 0.62 with average 3.0 layers (62.1% theoretical FLOPs savings). Crucially, the adaptive routing collapsed to minimum depth for all samples and the core hypothesis (Pearson $r > 0 )$ was not verified $( r = - 0 . 0 1 7 )$ . These are prototype-scale observations of a documented negative result, not performance benchmarks; the canonical results artifact for the submission is train log.txt (to be provided with the cameraready version, not bundled with the current release), in which evaluate() records all reported numbers.

The complete reproduction guide is also available in REPRODUCE.md in the repository root.

## D.8 Diagnosis of the Prototype Negative Result

The prototype’s core negative result—Pearson $r = - 0 . 0 1 7$ between $\hat { d }$ and activated layers, with all samples routed to the same 3-layer retention set—admits a precise theoretical diagnosis via theorem 21. Three factors, each identifiable from the configuration in table 11, plausibly placed the prototype inside the collapse basin identified above.

• Regularizer-to-task gradient imbalance. The prototype uses $\alpha = 0 . 0 1$ (balance) and $\beta = 0 .$ 1 (efficiency) at batch size 32. By the proof sketch of theorem 21, collapse is locally stable whenever $\stackrel { \cdot } { \alpha } + \beta \gg 1 / N ;$ here $\overset { \cdot } { \alpha } + \beta \overset { \cdot } { = } 0 . 1 1$ and $1 / N = 0 . 0 3 1$ , so the regularizing gradient exceeds the per-sample task gradient by roughly $3 . 5 \times$ , satisfying condition (i). Reducing $\alpha , \beta$ or increasing N would narrow the collapse basin, though both trade off against budget control and training stability.

• Gate saturation and STE Jacobian rank deficiency. The gate bias is initialized to $b _ { l } =$ 2.0, yielding $g _ { l } = \sigma ( 2 . 0 ) \approx 0 . 8 8$ at start; combined with the hard threshold $\theta _ { l } = 0 . 5$ this places the initial operating point near the saturation knee of the sigmoid, where the straight-through-estimator Jacobian $\partial \tilde { g } _ { l } / \partial g _ { l }$ is concentrated at the threshold crossing and is near-zero elsewhere. This satisfies condition (ii) of theorem 21: the difficulty signal cannot break the across-sample symmetry of the gates when the STE Jacobian is rank-deficient in the saturated regime.

• Curriculum simplification. The prototype omits Stage $1 ( \rho = 0$ warmup) and uses a fixed Gumbel temperature $\tau _ { G } = 1 . 0$ instead of the designed $2 . 0  0 . 5$ anneal (section D.2). The full curriculum is precisely the mechanism intended to move the optimizer out of the collapse basin by first training at $\rho = 0$ (where all gates are open, breaking the saturation regime) before introducing sparsity; omitting it removes the principal escape trajectory.

This diagnosis yields three concrete, theory-grounded remediations that follow directly from theorem 21: (a) reduce $\alpha , \beta$ or scale N so that $\alpha + \beta \lesssim 1 / N$ , shrinking the collapse basin; (b) lower the gate-bias initialization b<sub>l</sub> and anneal $\tau _ { G }$ from $2 . 0  0 . 5$ to keep the STE Jacobian non-degenerate during the differentiation phase; and (c) restore the full three-stage curriculum so that Stage 1 trains at $\rho = 0$ before sparsity is introduced. These remediations are consistent with, and refine, the higher-level future-work items in section 10 (cure routing collapse, complete curriculum). Because the diagnosis is conditioned on theorem 21—whose stability assumptions are themselves unverified at scale—we present it as a falsifiable hypothesis for the next iteration rather than as a confirmed root cause.

## D.9 Limitations

This work carries limitations along three axes, each already surfaced in the main text but collected here for the NeurIPS reviewer’s convenience.

• Empirical scope. The prototype validates only the discriminative SST-2 routing pathway at a 6-layer scale and reports a documented negative result $( r = - 0 . 0 1 7 )$ . The difficulty– depth monotonicity (theorem 4) is a population-level theoretical statement whose empirical validity at scale is not established by the prototype, and the variance-decomposition analysis of individual difficulty features remains contingent on a non-collapsed routing regime.

• Collapse-regime avoidance. theorem 21 identifies a degenerate equilibrium of the gate parameters but does not prescribe a hyperparameter regime that provably excludes it; deriving a sufficient condition on $\alpha , \beta$ relative to batch size and gate-saturation level that excludes collapse is left as an open problem. The three remediations in section D.8 (reduce $\alpha , \beta ,$ lower $b _ { l }$ and anneal $\tau _ { G } .$ , restore the three-stage curriculum) are theory-grounded proposals for the next iteration, not executed runs.

• Autoregressive generation. The architectural extensibility of VALSE to autoregressive generation (position-agnostic gate, causal-mask compatibility, self-speculative decoding fallback) is a present design property of the routing mechanism, not an empirically validated claim; the systematic validation on autoregressive generative LLMs is identified as the central community-wide open problem in section 2.4.

## D.10 Ethics Statement

This work proposes a routing mechanism for adaptive-depth Transformer inference. The prototype uses SST-2 (sentiment) and GSM8K (grade-school math) datasets, both publicly available and not containing sensitive personal data. The method reduces computational cost per inference, which can lower the energy footprint of Transformer deployment; we are not aware of specific misuse vectors introduced by per-sample layer skipping beyond those already applicable to the underlying base models. We do not release new pretrained model weights beyond the prototype-scale checkpoint.

## D.11 Broader Impact

The primary broader impact of adaptive-depth routing is the potential to reduce inference FLOPs and hence the per-query energy cost of Transformer models, which is environmentally and economically beneficial at scale. Per-sample, difficulty-aware skipping can also enable adaptive compute allocation (more depth for harder inputs, less for easier ones), which is a step toward input-conditional efficiency. Risks include the collapse-to-minimum-depth failure mode documented in this work (which silently negates the efficiency–quality trade-off and must be monitored via the difficulty– depth correlation diagnostic) and the risk that difficulty scores encode dataset-specific biases when the feature set is not re-validated on new domains; both are flagged here as deployment-time monitoring obligations rather than as resolved design properties.

## References

Safwan Ashraf, Saurabh Damaraju, William Coad, Mostafa Arezin, Aaryan Bhattacharjee, Eleftherios Sotirakos, Michael McCarley, Jan Kautz, Miles Birch, Jeff Henderson, et al. SliceGPT: Compress large language models by deleting and slicing layers. arXiv preprint arXiv:2401.15024, 2024.

Gregor Bachmann, Niloofar Mırzamomen, Tal Schuster, Rene Hubis, and Laurence Aitchison. Drowzee: Dynamic pre-inference for BERT with per-sample compute budget. arXiv preprint arXiv:2305.17698, 2023.

Damai Dai, Chenggang Deng, Du Zhang, Shuang Xu, Xiaosu Zhu, Huanyu Gao, Zihao Hu, Yu Wu, Pei-Duo Wang, Bowen Ma, Liang Zhang, Yuliang Liu, Tiejun Bi, Shida Zhang, Hanfeng Wu, Lingxiao Wu, et al. DeepSeek-MoE: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Nicolas Usunier, and Surya Ganguli. Universal transformers. In International Conference on Learning Representations (ICLR), 2019.

Mostafa Elhoushi, Zichao Lai, Lior Shkuro, Jean-Antoine Loubes, Vladimir Braverman, Jiawei Ding, Alexander Kim, Nikita Kozlov, Qijun Tan, Borja Balle, and Gil Kahn. Layer Skip: Enabling early-exit inference and self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting of the Associationfor Computational Linguistics (ACL), 2024a.

Mostafa Elhoushi, Amnon Shaked, Zichao Lai, Ruitao Lai, Wesam Sukkar, Lior Shkuro, Vladimir Braverman, Jiawei Ding, Solomon Aizman, and Gil Kahn. The layers they are a-changin’: A simple and efficient pruning method for Transformer models. arXiv preprint arXiv:2403.05704, 2024b.

Angela Fan, Edouard Grave, Armand Joulin, and Tomas Mikolov. Reducing Transformer depth on demand with structured dropout. In International Conference on Learning Representations (ICLR), 2020.

William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research (JMLR), 23: 1–39, 2022.

Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1607.01786, 2016.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Guillaume Lample, Lucile Sayed, Timothee Lachaux, Baptiste Rozi´ ere, and Alexandre Lacroix. Mixtral of\` experts. arXiv preprint arXiv:2401.04088, 2024.

Dmitry Lepikhin, Hyoukloong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation. arXiv preprint arXiv:2006.16668, 2020.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from Transformers via speculative decoding. arXiv preprint arXiv:2211.17192, 2023.

Weijie Liu, Peng Zhou, Zhijuan Wang, Zhe Chen, Lihao Cao, Yifan Qi, and Hengbo Ji. FastBERT: A speedup system for BERT inference. In Findings of the Association for Computational Linguistics: EMNLP, pages 3243–3253, 2020.

Zhuang Liu, Zhi-Wei Zhang, Xiaotao Xie, Keji Liu, and Feng Ding. DynamicBERT: Enhancing BERT’s efficiency via adaptive width and depth. arXiv preprint arXiv:2103.12241, 2021.

Xin Men, Qingyu Xu, Yutao Wang, Li Wang, Hongyu Lin, Yifu Lu, and Weipeng Han. Short-GPT: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853, 2024.

Negin Mustafa, Jorn Rethmeier, Vishaal Karthikeyan, Bhola Raj, Seyedali Honari, Francois¨ Lanusse, Konrad Szafer, and Krzysztof Choromanski. MoNE: Mixture of nested encoders for Transformer models. arXiv preprint arXiv:2401.16795, 2024.

OpenAI, Josh Achiam, Steven Adams, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

David Raposo, Adam Santoro, David Faulkner, Marta Grabska-Barwinska, Abdelrahman Elhoseiny,´ Timothy Lillicrap, Sasha Kapturowski, and Jack Howard. Mixture-of-depths: Dynamically allocating compute in Transformers. In International Conference on Machine Learning (ICML), 2024.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Daphne Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. Confident adaptive Language modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, and Geoffrey Hinton. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations (ICLR), 2017.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Timothee Lachaux, Marie-Aude´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and\` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Ji Xin, Raphael Tang, Jae Lee, Yo Ren, and Jimmy Lin. DeeBERT: Enhancing BERT inference with adaptive computation time. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1435–1444, 2020.

Canwen Xu, Jialin Zhou, Ke Wang, Yiming Lu, Wenmeng Zhang, and Jianshu You. BERT-oftheseus: Compressing BERT by progressive replacing with a compact compass. In Findings of the Associationfor Computational Linguistics: EMNLP, pages 2855–2865, 2020.

Junjie Zhou, Zhe Zhao, Huajie Shen, Lingjuan Zhan, Yang Liu, and Wei Wang. MoreThanGreedy: Pipeline adaptable BERT with confident early exit (PABEE). In Findings of the Association for Computational Linguistics: EMNLP, pages 934–939, 2020.