# Scaling Laws for Looped Mixture of Experts

Yanbei Chen<sup>∗</sup>, Anirudh Goyal<sup>†</sup>, Raghuraman Krishnamoorthi<sup>†</sup>

Meta AI

<sup>∗</sup>Corresponding author, <sup>†</sup>Senior authors

Looped transformers and Mixture-of-Experts (MoE) ofer complementary routes to eficient scaling: recurrence increases computational depth at fixed parameters, while MoE sparsity expands total capacity at fixed active compute. Yet existing scaling laws model recurrence or sparsity in isolation. In this work, we introduce Loop Scaling Laws, the first scaling law to jointly model recurrence and sparsity alongside model size and data. At its core is a bounded, sparsity-conditional recurrence mapping that characterizes the efective-parameter gain from looping and how sparsity raises this gain. The laws predict the held-out loss of looped models more accurately than prior alternatives, and recover the standard dense and MoE scaling laws as special cases. Beyond prediction, the fitted laws provide a principled foundation for designing looped MoE models under compute and memory constraints. Downstream evaluations further demonstrate the complementary benefits of the two axes: sparsity delivers ∼3× active-parameter eficiency, recurrence yields ∼2× total-parameter eficiency on reasoning, and joint scaling further advances the performance frontier. As a practical extension, we show these gains hold at trillion-token scale: at matched training compute, a looped MoE with law-derived recurrence matches a ∼2× larger non-looped MoE on the reasoning benchmarks, while enabling test-time scaling through recurrence.

Date: Sep 30, 2026

Correspondence: yanbeichen@meta.com

∞Meta

## 1 Introduction

Scaling language models has conventionally relied on larger models and more data (Kaplan et al., 2020; Hofmann et al., 2022). Beyond model and data, two additional axes enable eficient scaling. Mixture-of-Experts (MoE) introduces sparsity, expanding total capacity at fixed active compute, and is now widely adopted by frontier models from cloud (DeepSeek-AI, 2024; Yang et al., 2025) to edge (Chen et al., 2026; Apple, 2026). Recently, looped transformers (Dehghani et al., 2018) introduce recurrence, reusing shared weights across passes to increase computational depth at fixed stored parameters and delivering substantial gains in reasoning (Saunshi et al., 2025; Geiping et al., 2025; Zhu et al., 2025; Jeddi et al., 2026). Looped MoE models naturally combine these complementary axes.

Yet the joint scaling behavior of recurrence and sparsity remains underexplored. Scaling laws for looped transformers capture recurrence, modeling its capacity gain with linear or power-law forms that grow without bound (Prairie et al., 2026; Schwethelm et al., 2026). MoE laws capture sparsity, varied through expert count, but omit recurrence (Clark et al., 2022; Ludziejewski et al., 2025; Abnar et al., 2025). Recent and concurrent looped MoE studies examine sparsity while fixing recurrence at two passes in their primary scaling analyses (Lee et al., 2026; Wang et al., 2026), leaving the interaction between the two axes unmodeled. Consequently, a unified scaling law that jointly characterizes recurrence and sparsity in looped MoE models has yet to be established.

Establishing such a law requires understanding how recurrence and sparsity shape the efective capacity gain from looping. This raises two central questions: (Q1) How much efective capacity does each recurrent pass add, and does this gain grow indefinitely? (Q2) Does greater MoE sparsity increase and sustain this gain across recurrent passes? Two observations in Figure 1(b) point toward the answers. First, increasing recurrence shows diminishing returns: loss drops sharply over the first few passes, and then approaches a plateau within the observed range. Second, greater sparsity, varied by expert count E, improves and sustains this gain: with more experts, looping achieves lower loss over more passes. Figure 1(a) explains this interaction intuitively: in a dense looped transformer, every recurrent pass reuses the same weights; whereas in a looped MoE, the router can send a token to diferent experts across passes, allowing each pass to reach a new set of parameters.

![](images/ce15445f68bb2e5e7f89ba288bede19b06e65d87fb94503c813d70ffa328e6ef.jpg)  
(a)

![](images/765ab235d5e0e8f632706f9c450255344294eb2d482606f7343dbd6083af8aa5.jpg)  
(b)

![](images/6ffb3838aeac5b250647fa0a1807a36e636cbec77d65d3195ef1302c564a7e3d.jpg)  
(c)  
Figure 1 (a) Dense looped transformers reuse the same FFNs, whereas looped MoE models can route tokens to diferent experts across recurrent passes. (b) At fixed active parameters and training tokens, increasing recurrence yields diminishing returns, with the dense model (E=1) saturating earlier than MoE models (E>1). (c) Our Loop Scaling Laws model a bounded efective-parameter gain conditioned on recurrence R and sparsity (varied by expert count E), unlike prior linear and power-law mappings that assume unbounded gains.

To answer the above questions, we introduce Loop Scaling Laws, the first unified law over model size, data, recurrence, and sparsity. The law treats each recurrent pass as adding efective parameters with diminishing returns: each pass adds less than the last, with its total capacity gain approaching a finite asymptote rather than growing without bound. This asymptote also depends on sparsity: more experts raise and sustain the gain over more recurrent passes, so recurrence and sparsity interact within a single form, as illustrated in Figure 1(c). The law subsumes the standard dense, dense-looped, and non-looped MoE scaling laws as special cases. Empirically, it predicts held-out loss more accurately than prior alternatives with unbounded mappings, extrapolating well to unseen recurrence and expert counts.

Beyond prediction, the fitted law provides a principled recipe for looped MoE model design: it selects the compute- and memory-optimal recurrence and expert count under given budgets for resource-constrained deployment. Our empirical results across 14 downstream benchmarks further confirm the complementary gains of sparsity and recurrence. By scaling sparsity, an MoE model can surpass larger dense models with 1/3 the active parameters, while scaling recurrence enables a looped model to match non-looped models with 1/2 the total parameters. Jointly scaling both further advances the frontier beyond either axis alone. Together, these results establish joint scaling of recurrence and sparsity as a new parameter-eficient scaling paradigm. We extend this paradigm to practical, trillion-token training: at matched compute, a looped MoE model (0.3B active/1.3B total) with law-derived recurrence matches the reasoning performance of a larger non-looped MoE (0.6B active/2.9B total), trading additional inference compute for approximately 2× parameter eficiency. We also show that the looped MoE can unlock on-demand test-time scaling by varying recurrence at inference.

## 2 Related Work

Looped transformers (also known as recurrent-depth or recursive transformers) increase computational depth by repeatedly applying shared layers without proportional growth in stored parameters. Looped architectures have evolved from single-layer recurrence in Universal Transformers (Dehghani et al., 2018) to full-model recurrence (Zhu et al., 2025; Giannou et al., 2023) and middle-block reuse for recursive and latent-reasoning models (Saunshi et al., 2025; Zeitoun et al., 2026; McLeish et al., 2025; Geiping et al., 2025). Training objectives range from final-pass next-token prediction loss (Saunshi et al., 2025; Geiping et al., 2025; McLeish et al., 2025) to intermediate-pass supervision (Bae et al., 2024), self-distillation (Goyal et al., 2026), and shortcut consistency (Jeddi et al., 2026). Prior work shows that recurrent depth provides parameter-eficient training- and test-time scaling, especially on reasoning tasks (Saunshi et al., 2025; Geiping et al., 2025; Zhu et al., 2025; Jeddi et al., 2026). Motivated by these findings, we model recurrence as an explicit scaling axis to characterize how performance scales with looping.

Table 1 Comparison of scaling laws for training LLMs. ✓ indicates variables explicitly modeled by each law. $N / F$ denotes model size $N \ /$ training FLOPs F. S/E denotes MoE sparsity S (varied by expert expansion E).
<table><tr><td></td><td></td><td colspan="5">Scaling axes</td><td></td><td colspan="2">Optimality</td></tr><tr><td>LLM Category</td><td>Scaling law</td><td>N/F</td><td></td><td>D</td><td>R</td><td>S/E</td><td>Recurrence mapping</td><td>Compute Memory</td><td></td></tr><tr><td>Dense</td><td>Chinchilla (2022)</td><td>√</td><td>1</td><td></td><td>X</td><td>×</td><td></td><td>√</td><td>X</td></tr><tr><td rowspan="2">MoE</td><td>Unified routed (2022)</td><td>√</td><td>×</td><td>×</td><td></td><td>√</td><td></td><td>×</td><td>X</td></tr><tr><td>Joint MoE (2025)</td><td>√</td><td>√</td><td></td><td>×</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td rowspan="2">Looped</td><td>Parcae (2026/04)</td><td>√</td><td></td><td>r</td><td></td><td>X</td><td>linear; unbounded†</td><td>√</td><td>×</td></tr><tr><td>Iso-Depth (2026/05)</td><td>√</td><td>√</td><td>√</td><td></td><td>×</td><td>power law; unbounded</td><td>√</td><td>×</td></tr><tr><td rowspan="2">Looped MoE</td><td>Sparse Layers (2026/05)</td><td>√</td><td>×</td><td>×</td><td></td><td>×</td><td>fixed recurrence (R=2)</td><td>√</td><td>×</td></tr><tr><td>SMELT (2026/09)</td><td>√</td><td>√</td><td>X</td><td></td><td>√</td><td>fixed recurrence (R=2)</td><td>√</td><td>×</td></tr><tr><td rowspan="2"></td><td>Loop scaling laws (ours)</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>bounded; sparsity-dependent</td><td>√</td><td>√</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup>Parcae uses the fully unrolled parameter count in its trained-recurrence scaling analysis and separately fits a saturating law over test-time depth.

Mixture of experts (MoE) models increase total capacity without proportionally increasing per-token computation by routing each token to a sparse subset of expert networks (Shazeer et al., 2017; Fedus et al., 2022). MoE has become a well-established scaling paradigm across deployment regimes, from open-weight and proprietary frontier models deployed in the cloud, e.g., Mixtral (Jiang et al., 2024), DeepSeek-V3 (Dai et al., 2024; DeepSeek-AI, 2024), Qwen3MoE (Yang et al., 2025), and Gemini (Gemini Team, 2024), to resource-constrained models deployed on edge devices, e.g., MobileMoE (Chen et al., 2026) and AFM 3 Core Advanced (Apple, 2026). Recent works have also explored looped MoE models, where looped MoE is shown to outperform non-looped dense transformers (Csordás et al., 2024), and MoE efectively improves looping performance (Lee et al., 2026) or vice versa (Wang et al., 2026), suggesting the complementary architectural advantages of unifying MoE and looping. We further formulate a unified scaling law that quantifies their scaling benefits in a single functional form.

Scaling laws characterize predictable power-law relationships among loss, model size, data, and compute, providing a principled foundation for compute-optimal language model training (Kaplan et al., 2020; Hofmann et al., 2022). MoE scaling laws further incorporate routed expert count (Clark et al., 2022), expert granularity (Krajewski et al., 2024), sparsity (Abnar et al., 2025), enabling optimization under memory constraints (Ludziejewski et al., 2025), eficiency leverage (Tian et al., 2026), and on-device constraints (Chen et al., 2026). More recently, Parcae and Iso-Depth model recurrence for dense looped transformers (Prairie et al., 2026; Schwethelm et al., 2026), while concurrent work SMELT analyzes looped MoE models with recurrence fixed in their primary scaling analyses (Wang et al., 2026). As summarized in Table 1, prior laws model either recurrence or sparsity, but not their joint efect. Our unified MoE loop scaling law jointly models model size, data, recurrence, and sparsity, while recovering the prior laws for dense and MoE models as equivalent reduced forms.

## 3 Loop Scaling Laws

## 3.1 Preliminaries

Scaling law. The standard Chinchilla-style scaling law (Hofmann et al., 2022; Kaplan et al., 2020) models the relationship between model parameters N, training tokens $D _ { : }$ , and model loss $\mathcal { L }$ as

$$
\mathcal { L } ( N , D ) = A N ^ { \alpha } + B D ^ { \beta } + c ,\tag{1}
$$

where $A , B , \alpha , \beta ,$ c are fitted coeficients, $\alpha , \beta < 0$ are the model and data scaling exponents, c is the irreducible loss. Training compute is $F _ { \mathrm { t r a i n } } = 6 N D$ , and inference compute is $F _ { \mathrm { i n f } } = 2 N$ per token.

![](images/326439b744d64928561af3a1e02051d40e19763c936f5833b7412e54c2f89861.jpg)  
(a)

![](images/240cef79e3b4db7be06483a896522e161b2ac332dc60c0aced6ed243f129b837.jpg)  
(b)

Recurrence mappings for $N _ { \mathbf { e f f } }$
<table><tr><td>Mapping|</td><td> $N _ { \mathrm { e f f } } - N$ </td><td>I  $R \to \infty$ </td></tr><tr><td>Linear</td><td> $( R { - } 1 ) N _ { \mathrm { l o o p } }$ </td><td>∞</td></tr><tr><td>Power law</td><td> $( \dot { R } ^ { \varphi } - \dot { 1 } ) \dot { N } _ { \mathrm { l o o p } }$ </td><td>∞</td></tr><tr><td>Bounded</td><td> $\left| \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 { - } e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right) \right.$ </td><td> $\left| \kappa _ { 1 } N _ { \mathrm { l o o p } } \right.$ </td></tr></table>

<table><tr><td colspan="4">Held-out prediction RMSE (↓)</td></tr><tr><td>Evaluation</td><td>Linear</td><td>Power law</td><td>Bounded</td></tr><tr><td>Held-out R</td><td>0.2566</td><td>0.0313</td><td>0.0092</td></tr><tr><td>Held-out N</td><td>0.1106</td><td>0.0211</td><td>0.0128</td></tr><tr><td>Held-out D</td><td>0.0882</td><td>0.0073</td><td>0.0049</td></tr></table>

(c)  
Figure 2 (a) Illustration of efective-parameter gain under linear, power-law, bounded recurrence mappings. (b) Predictive loss by diferent recurrence mappings: each mapping is fitted on $R \leq 8$ and applied to predict the held-out-R up to $R = 1 6$ . (c) Formula on recurrence mappings and their efective-parameter gains (top) and their evaluation on held-out recurrence $R ,$ model $N ,$ and data D with RMSE scores (bottom).

Parameter counts and compute. A looped model reuses a block of parameters $N _ { \mathrm { l o o p } }$ over R recurrent passes with fixed model parameters N. In a forward pass, its unrolled parameter count is

$$
N _ { \mathrm { u n r o l l } } ( R ) = N _ { \mathrm { a c t } } + ( R - 1 ) N _ { \mathrm { l o o p } } .\tag{2}
$$

For dense looped models, the parameters are fixed as $N _ { \mathrm { a c t } } { = } N _ { \mathrm { t o t a l } } { = } N$ . For looped MoE models, the active and total parameters $N _ { \mathrm { a c t } }$ and $N _ { \mathrm { t o t a l } }$ are fixed. In both cases, R=1 recovers the corresponding non-looped baseline. Under full backpropagation through all recurrent passes, the training and per-token inference compute are $F _ { \mathrm { t r a i n } } = 6 N _ { \mathrm { u n r o l l } } ( R ) D$ and $F _ { \mathrm { i n f } } = 2 N _ { \mathrm { u n r o l l } } ( R )$ , respectively. While looping recurrence trades compute for quality at fixed parameters, MoE sparsity trades parameters for quality at fixed compute. We explore these complementary scaling axes in the following.

## 3.2 Loop Scaling Law

The standard scaling law depends only on model N and data D, and thus cannot capture the scaling behaviour of looping recurrence R. Prior scaling laws for looped transformers address this limitation by replacing N with a recurrence-dependent efective parameter count $N _ { \mathrm { e f f } } ( R )$ . Parcae’s training scaling law (Prairie et al., 2026) uses the fully unrolled parameter count: a linear mapping on $R ;$ whereas Iso-Depth (Schwethelm et al., 2026) learns a power-law mapping on R with exponent φ:

$$
N _ { \mathrm { e f f } } ^ { \mathrm { l i n e a r } } ( R ) = N + ( R - 1 ) N _ { \mathrm { l o o p } } , \qquad N _ { \mathrm { e f f } } ^ { \mathrm { p o w e r } } ( R ) = N + ( R ^ { \varphi } - 1 ) N _ { \mathrm { l o o p } } .\tag{3}
$$

$N _ { \mathrm { e f f } } ^ { \mathrm { l i n e a r } } ( R )$ assigns a constant efective-parameter gain $N _ { \mathrm { l o o p } }$ per recurrence; $N _ { \mathrm { e f f } } ^ { \mathrm { p o w e r } } ( R )$ yields diminishing gains with $0 { < } \varphi { < } 1$ . Nevertheless, both mappings are unbounded: lim $_ { R \to \infty } N _ { \mathrm { e f f } } ( R ) = \infty$ , where increasing recurrence substitutes for parameters indefinitely (Figure 2(a)).

A bounded recurrence mapping. However, increasing recurrence yields diminishing returns. Empirically, the model loss drops substantially over the earlier passes, and flattens gradually within the evaluated range (Figure 2(b)). This behavior is consistent with weight sharing: each additional pass increases computational depth without introducing new learned parameters, leading to diminishing marginal gains. We therefore introduce the following bounded monotone mapping:

$$
N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( R ) = N + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right) , \qquad \kappa _ { 1 } > 0 , \ \kappa _ { 2 } > 0\tag{4}
$$

where $\kappa _ { 1 }$ sets the asymptotic efective-parameter gain from looping: $\kappa _ { 1 } N _ { \mathrm { l o o p } }$ , while $\kappa _ { 2 }$ controls how quickly this limit is approached as R increases.

Boundary properties. The bounded mapping in Eq. 4 recovers the non-looped baseline at $R { = } 1$ , and approaches a finite efective-parameter bound as R→∞:

$$
N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( 1 ) = N , \qquad \operatorname* { l i m } _ { R \to \infty } N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( R ) = N + \kappa _ { 1 } N _ { \mathrm { l o o p } } .\tag{5}
$$

![](images/05302ce14553f5df834f09c4f35df814b9393d0c00d656d605477bdd25e95eba.jpg)  
(a)

![](images/feb9e44ffa1ee15835fc2c6e769535aed8e41b25083a2f9fc52d5970fdae4d34.jpg)  
(b)

MoE recurrence mappings for $N _ { \mathrm { e f f } }$
<table><tr><td>Mapping</td><td> $N _ { \mathrm { e f f } } - N _ { \mathrm { a c t } }$ </td></tr><tr><td>Linear</td><td> $( R { - } 1 ) N _ { \mathrm { l o o p } }$ </td></tr><tr><td>Power law</td><td> $( R ^ { \varphi } - 1 ) \bar { N } _ { \mathrm { l o o p } }$ </td></tr><tr><td>Bounded  $N _ { \mathrm { e f f } } ( R )$ </td><td> $\kappa _ { 1 } \tilde { N } _ { \mathrm { l o o p } } \left( 1 { - } e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right)$ </td></tr><tr><td>Bounded</td><td> $N _ { \mathrm { e f f } } ( R , m ) \Big | \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \Big ( 1 { - } e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \Big )$ </td></tr></table>

<table><tr><td colspan="5">Held-out prediction RMSE (↓)</td></tr><tr><td>Evaluation</td><td>Linear</td><td>Power</td><td> $N _ { \bf e f f } ( R ) \ N _ { \bf e f f } ( R , m )$ </td><td></td></tr><tr><td>Held-out R</td><td>0.2847</td><td>0.0480</td><td>0.0190</td><td>0.0100</td></tr><tr><td>Held-out E</td><td>0.0896</td><td>0.0113</td><td>0.0075</td><td>0.0047</td></tr><tr><td>Held-out N</td><td>0.0656</td><td>0.0077</td><td>0.0065</td><td>0.0050</td></tr><tr><td>Held-out D</td><td>0.0824</td><td>0.0074</td><td>0.0058</td><td>0.0043</td></tr></table>

(c)  
Figure 3 (a) Illustration of efective-parameter gain with sparsity-conditional recurrence mapping. (b) Predictive loss under diferent MoE recurrence mappings fitted on runs with $R \leq 8 ;$ markers denote observations used for fitting. (c) Formula on MoE recurrence mappings and their efective-parameter gains (top), and their evaluation on held-ou recurrence $R ,$ expert count $E ,$ model size $N ,$ , and data D with RMSE scores (bottom).

Thus, looping contributes at most $\kappa _ { 1 } N _ { \mathrm { l o o p } }$ additional efective parameters, an asymptotic property of the mapping rather than a guarantee beyond the observed recurrence range.

Loop scaling law. For a looped transformer with R recurrence loops, we replace the model-size term N in standard scaling law Eq. 1 with a recurrence-dependent efective parameter count $N _ { \mathrm { e f f } } ( R )$

$$
{ \mathcal { L } } ( N , D , R ) = A N _ { \mathrm { e f f } } ( R ) ^ { \alpha } + B D ^ { \beta } + c ,\tag{6}
$$

where $N _ { \mathrm { e f f } } ( R )$ introduces recurrence-specific coeficients alongside the base scaling-law coeficients $\{ A , \alpha , B , \beta , c \}$ in Eq. 3, the linear mapping uses linear constant and the power-law mapping uses $\varphi ,$ while our bounded map ping in $\mathrm { E q . 4 }$ uses $\left\{ \kappa _ { 1 } , \kappa _ { 2 } \right\}$ . Figure $2 ( \mathrm { a } )$ illustrates the normalized efective-parameter gain $( N _ { \mathrm { e f f } } ( R ) - N ) / N _ { \mathrm { l o o p } }$ under these mappings. The efective parameter count $N _ { \mathrm { e f f } } ( R )$ is distinct from the unrolled parameter count $N _ { \mathrm { u n r o l l } } ( R )$ : the latter determines compute, whereas the former models the efective parameter capacity attributed to weight-tied recurrence.

Reduced form $\scriptstyle { \mathcal { L } } | _ { R = 1 }$ . For any recurrence mapping satisfying $N _ { \mathrm { e f f } } ( 1 ) = N$ , the loop scaling law in Eq. 6 reduces to the standard non-looped scaling law in Eq. 1 at R=1 (as derived in Proposition A.1).

Comparison of recurrence mappings. We derive the fitted laws with the same parametric fitting and experimental sweep over $( R , N , D )$ , as detailed in Appendix C.3. Figure 2(b) compares the predictive loss over recurrence for loop scaling laws under diferent recurrence mappings, where each law is fitted on the sweep over N, D, $R \leq 8$ and extrapolated to the held-out $R { = } 1 6$ . The linear mapping substantially overestimates the efective-parameter gain, while the power-law mapping captures diminishing gains but remains overly optimistic beyond the fitted range. In contrast, the bounded mapping captures the loss plateau. The held-out evaluation over unseen N, D, R in Figure $2 ( \mathrm { c ) }$ further shows the law with bounded recurrence mapping achieves the lowest held-out RMSE, indicating it better captures the scaling trends across model size, data, and recurrence.

## 3.3 MoE Loop Scaling Law

The loop scaling law in Section 3.2 assumes dense looped models. In looped MoE models, sparse routing may select diferent experts across recurrent passes, allowing tokens to traverse diferent parameter paths despite weight sharing (Figure 1(a)). Motivated by this expert-path diversity (see analysis in Appendix B), we introduce MoE sparsity as a condition for recurrence mapping, allowing sparsity to modulate the efective-parameter gain from looping.

A sparsity-conditional recurrence mapping. Let $m = N _ { \mathrm { a c t } } / N _ { \mathrm { t o t a l } } \in ( 0 , 1 ]$ denote the MoE activeparameter ratio, where smaller m indicates greater sparsity and $m { = } 1$ is the dense case. For top-k routing over $E _ { \mathrm { r o u t e } }$ experts, m $\approx k / E _ { \mathrm { r o u t e } }$ . We extend the dense bounded mapping in Eq. 4 to looped MoE models by replacing N with $N _ { \mathrm { a c t } }$ and parameterizing its coeficients as functions of m:

$$
N _ { \mathrm { e f f } } ( R , m ) = N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) , \quad \kappa _ { j } ( m ) = \kappa _ { j } m ^ { - \theta } ~ ( j = 1 , 2 ) .\tag{7}
$$

The fitted sparsity-scaling exponent $\theta \geq 0$ quantifies how strongly sparsity amplifies the parameter gain from recurrence, with $\theta = 0$ recovering a sparsity-independent recurrence mapping. As sparsity increases (m decreases), the factor $m ^ { - \theta }$ lifts the recurrence curve vertically through $\kappa _ { 1 } ( m )$ and stretches it horizontally through $\kappa _ { 2 } ( m )$ , thus raising the asymptotic efective-parameter gain that is approached over more recurrent passes, as illustrated in Figure $\mathrm { 3 ( a ) }$

Boundary properties. The sparsity-conditional mapping in Eq. 7 recovers the non-looped baseline at $R { = } 1$ and approaches a finite efective-parameter bound as $R {  } { \infty }$ . It also recovers the dense bounded mapping in Eq. 4 at $m { = } 1$ , and approaches a linear-in-R limit as m→0:

$$
\begin{array} { l l } { { \displaystyle N _ { \mathrm { e f f } } ( 1 , m ) = N _ { \mathrm { a c t } } , } } & { { \displaystyle \operatorname* { l i m } _ { R \to \infty } N _ { \mathrm { e f f } } ( R , m ) = N _ { \mathrm { a c t } } + \kappa _ { 1 } m ^ { - \theta } N _ { \mathrm { l o o p } } , } } \\ { { \displaystyle N _ { \mathrm { e f f } } ( R , 1 ) = N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( R ) , } } & { { \displaystyle \operatorname* { l i m } _ { m \to 0 } N _ { \mathrm { e f f } } ( R , m ) = N _ { \mathrm { a c t } } + \frac { \kappa _ { 1 } } { \kappa _ { 2 } } ( R - 1 ) N _ { \mathrm { l o o p } } , \quad \theta > 0 . } } \end{array}\tag{8}
$$

Thus, at any fixed $m { > } 0$ , looping contributes at most $\kappa _ { 1 } m ^ { - \theta } N _ { \mathrm { l o o p } }$ additional efective parameters; in the extreme-sparsity limit $m  0$ (expert count $E \to \infty )$ , the gain grows linearly with R. These asymptotic limits are properties rather than guarantees beyond the observed recurrence range.

MoE loop scaling law. For a looped MoE model with R recurrence loops and active-parameter ratio m, we replace the active-parameter term $N _ { \mathrm { a c t } }$ in MoE scaling law (Clark et al., 2022; Ludziejewski et al., 2025) with the sparsity-conditional efective parameter count $N _ { \mathrm { e f f } } ( R , m )$ in Eq. 7:

$$
{ \mathcal { L } } ( N _ { \mathrm { a c t } } , D , R , E , m ) = A { \hat { E } } ^ { \delta } N _ { \mathrm { e f f } } ( R , m ) ^ { \alpha + \gamma \ln { \hat { E } } } + B { \hat { E } } ^ { \omega } D ^ { \beta + \zeta \ln { \hat { E } } } + c ,\tag{9}
$$

where $N _ { \mathrm { e f f } } ( R , m )$ introduces the fitted coeficients $\{ \kappa _ { 1 } , \kappa _ { 2 } , \theta \}$ alongside the base MoE scaling-law coeficients $\left\{ A , \alpha , B , \beta , c , \delta , \gamma , \omega , \zeta , E _ { \mathrm { s t a r t } } , E _ { \mathrm { m a x } } \right\}$ . Following Clark et al. (2022), $\hat { E }$ is a monotonic transformation: $\hat { E } ^ { - 1 } =$ $\left( E - 1 + \left( E _ { \mathrm { s t a r t } } ^ { - 1 } - E _ { \mathrm { m a x } } ^ { - 1 } \right) ^ { - 1 } \right) ^ { - 1 } + E _ { \mathrm { m a x } } ^ { - 1 } .$ , where E denotes the number of experts under top-1 routing (Clark et al., 2022; Ludziejewski et al., 2025). For top-k routing, we define the efective expert expansion as $E = E _ { \mathrm { r o u t e } } / k _ { \mathrm { : } }$ consistent with the parameterization of Chen et al. (2026).

Reduced forms. The MoE loop scaling law in Eq. 9 subsumes prior scaling laws as special cases: $( 1 ) \ \mathcal { L } | _ { E = 1 }$ at $E = 1$ , it recovers the dense loop scaling law in Eq. 6 after reparameterization; (2) $\mathcal { L } | _ { R = 1 } , \mathrm { a t } R = 1$ , it recovers the standard MoE scaling law; and (3) $\scriptstyle { \mathcal { L } } | _ { R = 1 , E = 1 }$ , at $E = 1 , R = 1$ , it recovers the standard dense scaling law in Eq. 1. Full derivations are given in Proposition A.2.

Comparison of MoE recurrence mappings. We derive the fitted MoE loop scaling laws using the same parametric fitting and experimental sweep over $( R , N _ { \mathrm { a c t } } , D , E )$ , as detailed in Appendix C.3. Figure 3(b) illustrates the predictive loss over recurrence R and MoE sparsity varied by expert count $E ,$ where each law is fitted on the sweep over varying $N _ { \mathrm { { a c t } } } , D , E $ , and $R { \le } 8$ . The same fitted laws are evaluated on held-out recurrence $R { = } 1 6$ in Figure $3 ( \mathrm { c } )$ . As observed in the dense comparison, the linear mapping overestimates recurrence gains and the power-law mapping remains optimistic; importantly, neither captures how saturation varies with sparsity. In contrast, our sparsity-conditional bounded mapping tracks the loss plateau across expert counts. Figure $3 ( \mathrm { c } )$ further shows the law with sparsity-conditional bounded mapping in Eq. 7 achieves the lowest held-out RMSE across all held-out axes, as compared to the laws with linear, power-law in Eq. 3 and sparsity-independent mappings Eq. 4. This confirms our formulated MoE loop scaling law in Eq.9 has a better predictive fit and captures how sparsity modulates the asymptotic gain from recurrence.

## 4 Experiments and Findings

## 4.1 Scaling Experiments and Analysis

Scaling experiments and fitted law. We jointly sweep over four scaling axes: model parameters $N _ { \mathrm { a c t } } \in$ {0.3, 0.6, 1.0}B, training tokens $D \in \{ 1 0 0 , 2 0 0 , \dots , 5 0 0 \} \mathrm { B }$ , expert count $E \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ , and recurrence

![](images/728b70848b7204542a80209383a8a258e42e5567ed1ee0c34044a6c055f8674f.jpg)

![](images/afbd52edcc99450d00a74a3ab47402f2c5aa257f317b678a4423958169fdeb4d.jpg)  
(a)  
(b)  
(c)  
(d)  
Figure 4 (a,b) Efective-parameter multiplier over recurrence at $N _ { \mathrm { a c t } } { = } 1 . 0 \mathrm { B }$ , and its asymptote over expert count. (c,d) Recurrence IsoFLOP profiles for dense (E=1) and MoE (E=8) models; markers denote minima.

$R \in \{ 1 , 2 , 3 , 4 , 6 , 8 \}$ . We use standard middle-cycle looping (Geiping et al., 2025) to loop over the middle block, leaving the first two prelude layers and last two coda layers unshared. As our formulation is agnostic to specific looping strategies, we leave other strategies to future work. The full training and law fitting details are provided in Appendix C.

How do recurrence and sparsity improve efective capacity from looping? To quantify the efective capacity gain from the fitted MoE loop scaling law, we define the efective-parameter multiplier:

$$
\rho _ { \mathrm { e f f } } ( R , m ) = \frac { N _ { \mathrm { e f f } } ( R , m ) } { N _ { \mathrm { a c t } } } = 1 + \kappa _ { 1 } ( m ) \frac { N _ { \mathrm { l o o p } } } { N _ { \mathrm { a c t } } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) .\tag{10}
$$

Its asymptote is $\rho _ { \mathrm { e f f } } ( \infty , m ) = 1 + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } / N _ { \mathrm { a c t } }$ , which characterizes the maximum efective-parameter multiplier relative to $N _ { \mathrm { { a c t } } } . \ \mathrm { { F i g u r e } \ 4 ( a ) }$ shows that, at fixed $N _ { \mathrm { a c t } }$ , the efective-parameter multiplier increases with larger recurrence but approaches a finite asymptote, while greater sparsity raises this asymptote and delays saturation over recurrent passes; Figure 4(b) shows this asymptote itself rises with expert count. Alternatively, this gain can be measured relative to the recurrent-block parameters $N _ { \mathrm { l o o p } }$ as the efectiveparameter gain $g _ { \mathrm { e f f } } = ( N _ { \mathrm { e f f } } - N _ { \mathrm { a c t } } ) / N _ { \mathrm { l o o p } }$ (see Figures $1 ( \mathrm { c } ) , 2 ( \mathrm { a } )$ , and $\mathrm { 3 ( a ) ) }$ , which yields the same scaling trends over recurrence and sparsity, as summarized below.

Finding: Scaling recurrence adds efective-parameter gain up to a finite, sparsity-dependent asymptote, while scaling sparsity raises and sustains this gain over more recurrent passes.

How do recurrence and sparsity push the IsoFLOP frontier? We examine the scaling behavior over recurrence and sparsity on an IsoFLOP basis based on the fitted law. Figures $^ \mathrm { 4 ( c , d ) }$ show predicted loss against $N _ { \mathrm { a c t } }$ under fixed training compute: $5 \times 1 0 ^ { 2 0 }$ and $1 0 ^ { 2 1 }$ FLOPs. When increasing sparsity from $E { = } 1$ (dense) to E=8 (MoE), i.e., comparing Figure 4(c) and (d), lower loss is attained under the same compute budget, indicating that higher sparsity pushes the IsoFLOP performance frontier. When increasing recurrence at fixed sparsity, higher recurrence can outperform lower recurrence at the same active parameter count $N _ { \mathrm { a c t } }$ with additional compute; for example, R=2 at $1 0 ^ { 2 1 }$ FLOPs achieves lower loss than R=1 at $5 \times 1 0 ^ { 2 0 }$ FLOPs for both E=1 and $E { = } 8 .$ , as summarized below.

Finding: Higher sparsity improves the IsoFLOP frontier at matched compute, while recurrence pushes this frontier further at the same parameter size when given more compute.

## 4.2 Compute- and Memory-Optimal Recurrence and Sparsity

Given the fitted law, which recurrence and sparsity should be selected under resource constraints? Our MoE loop scaling law provides a principled foundation for optimizing architectures under practical resource constraints of training compute, and deployment memory. At fixed weight memory, recurrence R trades additional compute for better performance; whereas at fixed compute, sparsity trades additional weight memory for improved performance. These complementary tradeofs motivate our following analyses that apply the fitted law to predict the compute-optimal recurrence at fixed sparsity, memory-optimal sparsity at fixed recurrence, and their joint optimum

![](images/baa23bb2d361c428b55edbfe3ea9a9ffc24d3576d4caa89eed7f811a9724ebb1.jpg)  
(a)

![](images/ee5eec7079ebd4d9c1033e26c378ac49ba1510282842811f78d6b6a50eee5356.jpg)  
(b)

![](images/95b7e499f71d522a4c75ea97b0e8da12913c850175a666623f354afe53e6c439.jpg)  
(c)  
Figure 5 Predicted compute-optimal recurrence $R ^ { \star }$ of dense (E=1) and MoE (E=8) models. (a) Recurrence with compute-optimal loss across training-compute budgets. (b) IsoFLOP profiles over recurrence at $\scriptstyle \bar { F } _ { \mathrm { t r a i n } } = 5 \times 1 0 ^ { 2 1 }$ FLOPs; stars mark $R ^ { \star }$ . (c) $R ^ { \star }$ across compute and active model size for E=8 (top) and E=1 (bottom).

Compute-optimal recurrence at fixed sparsity. At fixed model sparsity (fixed expert count $E )$ , increasing recurrence R leaves the model weights $N _ { \mathrm { t o t a l } }$ and weight memory $\mathcal { M } _ { \mathrm { w e i g h t } }$ unchanged, but increases the training compute through unroll parameters $N _ { \mathrm { u n r o l l } } ( R )$ . Therefore, recurrence can be compute-optimal if it achieves the minimal loss among recurrences trained under the same compute, while providing a meaningful loss reduction over the preceding recurrence:

$$
R ^ { \star } = \arg \operatorname* { m i n } _ { R } \mathcal { L } \big ( N _ { \mathrm { a c t } } , D , R , E , m \big ) , \qquad \mathrm { s . t . } \quad 6 N _ { \mathrm { u n r o l l } } ( R ) D = \bar { F } _ { \mathrm { t r a i n } } , \quad \Delta \mathcal { L } ( R ) \geq \epsilon .\tag{11}
$$

$\bar { F } _ { \mathrm { t r a i n } }$ is the given training-compute budget. For $R \geq 2 , \Delta \mathcal { L } ( R ) = \mathcal { L } _ { R - 1 } - \mathcal { L } _ { R }$ measures the predicted loss reduction from the additional recurrent pass $R { - } 1 {  } R$ . The tolerance ϵ specifies the minimum meaningful loss reduction. In practice, we set ϵ to the fitted-law RMSE, below which predicted improvements cannot be reliably distinguished from fitting error. If no pass satisfies $\Delta \mathcal { L } ( R ) \geq \epsilon , R ^ { \star } { = } 1$ . Figure 5 shows three trends on compute-optimal $R ^ { \star } \colon ( { \mathrm { a } } ) \ R ^ { \star }$ increases with more training-compute budget and is higher for smaller active models; (b) along each IsoFLOP profile, the selected $R ^ { \star }$ lies at or near the loss minimum, with subsequent recurrent passes providing negligible gains; and (c) across the $F _ { \mathrm { t r a i n } } – N _ { \mathrm { a c t } }$ design space, greater sparsity shifts the compute-optimal regime toward higher recurrence. These trends can be summarized below.

Finding: Larger training compute budgets and greater sparsity favor higher recurrence.

Memory-optimal sparsity at fixed recurrence. At fixed recurrence $R ,$ increasing sparsity through expert count E expands the total parameters $N _ { \mathrm { t o t a l } }$ and thus weight memory, while leaving the active compute unchanged. Therefore, sparsity can be memory-optimal by selecting the expert count E that achieves the minimal loss when trained under the same compute:

$$
\begin{array} { r } { E ^ { \star } = \underset { E } { \mathrm { a r g } \mathrm { m i n } } \ : \mathcal { L } ( N _ { \mathrm { a c t } } , D , R , E , m ) , \quad \mathrm { s . t . } \ : F _ { \mathrm { t r a i n } } = \bar { F } _ { \mathrm { t r a i n } } , \ : \ : \mathcal { M } _ { \mathrm { w e i g h t } } \leq M _ { \mathrm { b u d g e t } } , \ : \ : \Delta \mathcal { L } ( E ) \geq \epsilon . } \end{array}\tag{12}
$$

$M _ { \mathrm { b u d g e t } }$ is the weight-memory budget and $\mathcal { M } _ { \mathrm { w e i g h t } } = b _ { w } N _ { \mathrm { t o t a l } } / 8$ , where $b _ { w }$ is the weight precision in bits. Total deployment memory may include KV cache and other runtime state, but their optimization is outside our scope; we therefore consider weight memory only. For expert counts $\{ E _ { i } \}$ and $i \ge 2 , \Delta \mathcal { L } ( E _ { i } ) = \mathcal { L } _ { E _ { i - 1 } } - \mathcal { L } _ { E _ { i } }$ measures the loss reduction from expert expansion $E _ { i - 1 } {  } E _ { i }$ . The same tolerance ϵ as in Eq. 11 requires each expansion yields a meaningful loss reduction. If no expansion satisfies $\Delta \mathcal { L } ( E _ { i } ) \geq \epsilon , E ^ { \star } { = } E _ { 1 } { = } 1$ . Figure 6(a)

![](images/e70af0d2ecd3a3a353b1abb7408f4a0c1da646e8b14cbb2ed011d293dfa8d273.jpg)  
(a)

![](images/2716ee356bb351e9e141500b7cf753a9b3549907be513f44098f56ae4465f0a4.jpg)  
(b)

![](images/1a2e78dfe32805f92e20ac9247e5a43733eec0aa6d22a7f1eba9cedfc40c3846.jpg)  
(c)  
Figure 6 Predicted memory-optimal expert count $E ^ { \star }$ and joint optimum $( N _ { \mathrm { a c t } } ^ { \star } , E ^ { \star } , R ^ { \star } )$ at $\scriptstyle \bar { F } _ { \mathrm { t r a i n } } = 5 \times 1 0 ^ { 2 1 }$ FLOPs. (a) $E ^ { \star }$ across bf16 weight-memory budgets at R=4 (top) and R=1 (bottom). (b) $E ^ { \star }$ across INT4 weight memory and recurrence. (c) Predicted joint optimum $( N _ { \mathrm { a c t } } ^ { \star } , E ^ { \star } , R ^ { \star } )$ across INT4 weight memory.

and (b) show two trends on memory-optimal E<sup>⋆</sup>: (a) $E ^ { \star }$ increases with more memory budget, while higher recurrence favors larger $E ^ { \star }$ under the same memory; and (b) across the $\mathcal { M } _ { \mathrm { w e i g h t } } – R$ design space, larger memory budgets and higher recurrence shift the optimal regime toward greater sparsity. These trends are summarized below.

Finding: Larger weight memory budgets and higher recurrence favor greater sparsity.

Joint optimum on recurrence and sparsity. Given a training-compute budget and a memory budget, the recurrence R and sparsity (varied by expert count E) can be jointly compute- and memory-optimal if they achieve the minimal loss under the same resource constraints:

$$
\begin{array} { r l } & { ( N _ { \mathrm { a c t } } ^ { \star } , E ^ { \star } , R ^ { \star } ) = \displaystyle \operatorname* { a r g m i n } _ { N _ { \mathrm { a c t } } , E , R } \angle ( N _ { \mathrm { a c t } } , D , R , E , m ) } \\ & { \mathrm { s . t . ~ c o m p u t e : ~ } 6 N _ { \mathrm { u n r o l l } } ( R ) D = \bar { F } _ { \mathrm { t r a i n } } , ~ \Delta \mathcal { L } ( R ) \ge \epsilon ; ~ \mathrm { m e m o r y : ~ } \mathcal { M } _ { \mathrm { w e i g h t } } \le M _ { \mathrm { b u d g e t } } , ~ \Delta \mathcal { L } ( E ) \ge \epsilon . } \end{array}\tag{13}
$$

Given a training compute budget $\bar { F } _ { \mathrm { t r a i n } }$ and weight memory budget $M _ { \mathrm { b u d g e t } }$ , we use the fitted scaling law to identify the predicted optimal looped MoE configuration from a model ladder spanning active model size, expert count, and recurrence (Appendix C.2), which guides model design under resource constraints. For each candidate configuration $( N _ { \mathrm { a c t } } , E , R )$ , we set $D = \bar { F } _ { \mathrm { t r a i n } } / [ 6 N _ { \mathrm { u n r o l l } } ( R ) ]$ , retain candidates satisfying the compute and memory constraints, and select the optimum with lowest predicted loss. Figure 6(c) shows that under fixed training compute, tighter memory favors higher recurrence and smaller expert counts, e.g., selecting a 1.0B active model with $E { = } 8$ and R=2 at 3 GB; whereas larger memory favors greater sparsity and less recurrence, e.g., selecting a 1.6B active model with $E { = } 1 6$ and $R { = } 1$ at 10 GB, as summarized below.

Finding: At fixed compute, tight memory favors recurrence, and more memory favors sparsity.

## 4.3 Downstream Scaling with Recurrence and Sparsity

Empirical scaling on downstream tasks. We extend our scaling analysis of looped MoE models to downstream tasks, covering 14 benchmarks spanning five categories: reasoning (BBH, GSM8K), science (ARC-C/E, OpenBookQA), commonsense (HellaSwag, PIQA, SIQA, WinoGrande), reading (BoolQ, DROP), and knowledge (MMLU, Natural Questions, TriviaQA). We report Overall as the mean over 14 benchmarks. Motivated by prior work showing that recurrent depth can preferentially improve latent reasoning (Saunshi et al., 2025; Geiping et al., 2025; Zhu et al., 2025), we also report Reasoning as the mean of GSM8K and BBH. In Figure 7, we evaluate downstream performance as a function of training compute, comparing models with active sizes of 0.3B, 0.6B, and 1.0B under varying expert count and recurrence at matched training FLOPs. The complete benchmark suite and evaluation protocols are provided in Appendix D.1.

![](images/ee38529afcc3f7cdc7db984efed5c4150bf36819e47c39f32044062569a6b3d4.jpg)  
(a)  
(b)  
(c)  
(d)  
Figure 7 Downstream scaling of sparsity and recurrence. (a,b) Scaling E at R=1; (c,d) Scaling R at E=8, with the dense baseline $E { = } 1 , R { = } 1$ . (a,c) report Overall performance, and (b,d) report Reasoning performance; dashed lines mark compute-matched comparisons.

Table 2 Downstream results of A0.6B-2.9B MoE and A0.3B-1.3B LoopMoE across test-time recurrence R, trained at matched compute $\mathrm { F L O P s \approx 1 . 5 { \times } 1 0 ^ { 2 2 } }$ . Relative inference compute is normalized to the non-looped baseline. $\Delta$ reports gains over LoopMoE at R=1. Evaluation protocols are provided in Appendix D.1.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2"> $\mathbf { \left| _ { R } \right| _ { \overline { { \mathbf { \Lambda } } } } } \mathbf { \frac { \partial \mathbf { F } _ { \mathrm { i n f } } ^ { 0 . 3 \mathrm { B } } ( R ) } { \partial \mathbf { \Lambda } } } \mathbf { \Lambda } ^ { \left| \right.} $   $\overline { { F _ { \mathrm { i n f } } ^ { 0 . 6 \mathrm { B } } ( 1 ) } }$ </td><td colspan="2">Reasoning</td><td colspan="2">Science</td><td rowspan="2"></td><td colspan="4">Commonsense Reasoning</td><td colspan="2">Reading</td><td colspan="4">Knowledge</td></tr><tr><td></td><td>|BBH3 GSM8K8</td><td>ARC-C25 ARC-E OBQA</td><td></td><td></td><td></td><td>HS PIQA SIQA</td><td></td><td>Wino</td><td>BoolQ DROP3</td><td>MMLU5 NQ5 TQA5</td><td></td><td></td><td>Overall (∆)</td></tr><tr><td>A0.6B-2.9B MoE</td><td>|1|</td><td>1.0×</td><td>29.8</td><td>42.9</td><td>50.0</td><td>66.9</td><td>40.8</td><td>68.1 77.2</td><td></td><td>51.6</td><td>64.4</td><td>70.1</td><td>39.7</td><td>46.9</td><td>13.8</td><td>39.8</td><td>50.2</td></tr><tr><td rowspan="5">A0.3B-1.3B LoopMoE</td><td></td><td>0.4×</td><td>18.6</td><td>10.5</td><td>36.1</td><td>57.3</td><td>33.8</td><td>52.7 70.7</td><td></td><td>43.8</td><td>54.6</td><td>58.1</td><td>23.0</td><td>29.2</td><td>5.7</td><td>18.2</td><td>36.6</td></tr><tr><td></td><td>0.8×</td><td>26.7</td><td>30.0</td><td>42.2</td><td>60.3</td><td>38.2</td><td>59.9</td><td>72.5</td><td>46.4</td><td>58.5</td><td>62.8</td><td>33.7</td><td>37.6</td><td>8.3</td><td>24.1</td><td>43.0 (+6.4)</td></tr><tr><td></td><td>1.1×</td><td>30.1</td><td>41.0</td><td>44.9</td><td>60.9</td><td>39.6</td><td>62.4</td><td>73.2</td><td>50.2</td><td>61.3</td><td>65.9</td><td>38.4</td><td>41.6</td><td>8.9</td><td>25.6</td><td>46.0 (+9.4)</td></tr><tr><td></td><td>1.5×</td><td>31.5</td><td>41.5</td><td>46.5</td><td>63.9</td><td>39.0</td><td>62.9</td><td>73.9</td><td>51.1</td><td>60.6</td><td>64.4</td><td>39.3</td><td>42.2</td><td>9.8</td><td>27.1</td><td>46.7(+10.1)</td></tr><tr><td>12345</td><td>1.8×</td><td>31.8</td><td>41.2</td><td>47.5</td><td>64.3</td><td>40.4</td><td>63.1</td><td>74.0</td><td>51.1</td><td>61.0</td><td>64.3</td><td>39.0</td><td>42.2</td><td>9.9</td><td>27.0</td><td>46.9 (+10.3)</td></tr></table>

How do recurrence and sparsity translate into downstream performance? Figure 7 shows the two scaling axes provide distinct but complementary gains. In panels $^ { ( \mathrm { a } , \mathrm { b } ) }$ , increasing sparsity at fixed R=1 consistently improves both Overall and Reasoning across all three model sizes. These sparsity gains can ofset $\mathrm { a \sim 3 \times }$ increase in active model size: at $5 \times 1 0 ^ { 2 0 }$ FLOPs, the 0.3B MoE models with $E { \ge } 8$ outperform the larger 1.0B dense model on both Overall and Reasoning. In panels (c,d), increasing recurrence at fixed $E { = } 8$ produces more reasoning-oriented gains. These recurrence gains can ofset $\mathbf { a } \sim 2 \times$ increase in total parameters on Reasoning: at $1 0 ^ { 2 1 }$ FLOPs, the 0.3B model with R≥4 matches or exceeds the 0.6B model with $R { = } 1$ , while the 0.6B model with $R { \geq } 3$ matches the 1.0B model with R=1. Moreover, models that jointly increase sparsity and recurrence $( E { = } 8 , R { > } 1 )$ consistently outperform the dense baseline $( E { = } 1 , R { = } 1 )$ at matched training compute, showing that their gains accumulate when scaled together. Thus, sparsity and recurrence provide complementary routes to parameter-eficient scaling, with sparsity achieving ${ \sim } 3 \times$ active-parameter eficiency on overall performance and recurrence achieving ${ \sim } 2 \times$ total-parameter eficiency on reasoning.

A practical case study of looped MoE. As a practical extension of our scaling analysis on downstream tasks, we explore looped MoE in comparison to an approximately $2 \times$ larger non-looped MoE. Specifically, taking an A0.6B-2.9B MoE (0.6B active/2.9B total, $E { = } 8 , R { = } 1 )$ as the non-looped reference, we use the fitted law and apply our recurrence-selection criterion with the compute constraint $( \mathrm { E q . 1 1 } )$ to derive the recurrence for an A0.3B-1.3B looped MoE (0.3B active/1.3B total, E=8). We then train the two MoE models at matched compute at trillion-token scale (∼3T tokens for A0.3B-1.3B LoopMoE and ∼6T tokens for A0.6B-2.9B MoE), and compare their downstream performance. Further derivation and training details are in Appendix D.2.

How does a looped MoE compare to a 2× larger non-looped MoE at matched compute? Table 2 compares A0.3B-1.3B LoopMoE and A0.6B-2.9B MoE at matched training compute. Despite using ∼2× fewer active and total parameters, A0.3B-1.3B LoopMoE at $R \in \{ 4 , 5 \}$ matches A0.6B-2.9B MoE on reasoning tasks (BBH, GSM8K) by using more inference compute, while trailing on Overall performance. Beyond parameter eficiency, LoopMoE supports test-time scaling on demand: varying recurrence from R=1 to R=5 raises inference compute from low to high and boosts Overall performance by a substantial +10.3 points (36.6 → 46.9; Table 2).

## 5 Conclusion

We introduced Loop Scaling Laws, a unified predictive law that jointly models recurrence and sparsity alongside model size and data, closing a pressing gap between MoE and looped model scaling theories. Its bounded, sparsity-conditional mapping captures the diminishing recurrence gains observed within the evaluated range and models how sparsity raises the fitted capacity. The fitted law guides looped MoE model design by selecting recurrence and sparsity under compute and memory constraints, providing practical recipes for resource-constrained settings.

More broadly, our work reframes the scaling design space, positioning recurrence and sparsity as complementary axes for the next-generation parameter-eficient scaling paradigm. As efective scaling relies not only on how many model parameters are added, but also on how efectively additional computation can use them, these two axes should be co-designed according to resource constraints. By combining both axes properly, looped MoE models can advance the performance–eficiency frontier while enabling on-demand test-time scaling through recurrence.

To unlock the greater potential of jointly scaling recurrence and sparsity, we highlight several future directions. For modeling, more advanced looping strategies may be explored to enrich recurrent latent representations and boost the efective-capacity gain further. For scaling, looped MoE models could be generalized across broader scales, spanning from small models on the edge to large models in the cloud. For inference, the memory and latency of looped models could be further improved by optimizing recurrent states or adding early-exit gating, thus scaling recurrence only when necessary.

## References

Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs FLOPs: Scaling laws for optimal sparsity for mixture-of-experts language models. arXiv preprint arXiv:2501.12370, 2025.

Apple. Introducing the third generation of Apple’s foundation models. Apple Machine Learning Research, June 2026. URL https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models. Published June 8, 2026.

Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed recursive transformers: Efective parameter sharing with layer-wise LoRA. arXiv preprint arXiv:2410.20672, 2024.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Yanbei Chen, Hanxian Huang, Ernie Chang, Jacob Szwejbka, Digant Desai, Zechun Liu, Vikas Chandra, and Raghuraman Krishnamoorthi. MobileMoE: Scaling on-device mixture of experts. arXiv preprint arXiv:2605.27358, 2026.

Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hofmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, 2022.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising dificulty of natural yes/no questions. In Proceedings of NAACL-HLT, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber, Christopher Potts, and Christopher D Manning. MoEUT: Mixtureof-experts universal transformers. Advances in Neural Information Processing Systems, 2024.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal Transformers. arXiv preprint arXiv:1807.03819, 2018.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of NAACL-HLT, 2019.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 2022.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jefrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighof, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. arXiv preprint arXiv:2301.13196, 2023.

Sahil Goyal, Swayam Agrawal, Gautham Govind Anil, Prateek Jain, Sujoy Paul, and Aditya Kusupati. ELT: Elastic looped transformers for visual generation. arXiv preprint arXiv:2604.09168, 2026.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. LoopFormer: Elastic-depth looped transformers for latent reasoning via shortcut modulation. arXiv preprint arXiv:2602.11451, 2026.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, 2017.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, et al. Scaling laws for fine-grained mixture of experts. arXiv preprint arXiv:2402.07871, 2024.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 2019.

Ryan Lee, Jacob Biloki, Edward J. Hu, and Jonathan May. Sparse layers are critical to scaling looped language models. arXiv preprint arXiv:2605.09165, 2026.

Jan Ludziejewski, Maciej Pióro, Jakub Krajewski, Maciej Stefaniak, Michał Krutul, Jan Małaśnicki, Marek Cygan, Piotr Sankowski, Kamil Adamczewski, Piotr Miłoś, et al. Joint MoE scaling laws: Mixture of experts can be memory eficient. arXiv preprint arXiv:2502.05172, 2025.

Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. Teaching pretrained language models to think deeper with retrofitted recurrence. arXiv preprint arXiv:2511.07384, 2025.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018.

Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, and Daniel Y. Fu. Parcae: Scaling laws for stable looped language models. arXiv preprint arXiv:2604.12946, 2026.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Proceedings of EMNLP-IJCNLP, 2019.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. arXiv preprint arXiv:2502.17416, 2025.

Kristian Schwethelm, Daniel Rueckert, and Georgios Kaissis. How much is one recurrence worth? Iso-Depth scaling laws for looped language models. arXiv preprint arXiv:2604.21106, 2026.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.

Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. Towards greater leverage: Scaling laws for eficient mixture-of-experts language models. In International Conference on Learning Representations, 2026.

Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, and Jian Li. SMELT: Scaling laws for compute-matched MoE looped transformers. arXiv preprint arXiv:2609.01343, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Abbas Zeitoun, Lucas Torroba-Hennigen, and Yoon Kim. Hyperloop Transformers. arXiv preprint arXiv:2604.21254, 2026.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.

## APPENDIX

## A Reduced Forms of Loop Scaling laws

## A.1 Dense Loop Scaling Law

Proposition A.1 (Dense loop scaling law equivalence). At $R { = } 1$ , the dense loop scaling law recovers the standard non-looped scaling law.

Proof.

Reduced form $\scriptstyle { \mathcal { L } } | _ { R = 1 }$ . With recurrence fixed at $R { = } 1$ , the recurrence mapping $N _ { \mathrm { e f f } } ( R ) ~ \mathrm { ( E q . ~ 4 ) }$ recovers the non-looped baseline: $N _ { \mathrm { e f f } } ( 1 ) = N$ . Specifically,

$$
\left. N _ { \mathrm { e f f } } ( R ) \right| _ { R = 1 } = N + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( 1 - 1 ) / \kappa _ { 2 } } \right) = N .
$$

The loop scaling laws reduce to

$$
\begin{array} { c } { { { \mathcal { L } ( N , D , R ) } | _ { R = 1 } = A N _ { \mathrm { e f f } } ( 1 ) ^ { \alpha } + B D ^ { \beta } + c } } \\ { { = A N ^ { \alpha } + B D ^ { \beta } + c } } \\ { { = { \mathcal { L } } ( N , D ) . } } \end{array}\tag{14}
$$

This reduced form is equivalent to the standard non-looped scaling law in $\operatorname { E q . 1 }$ (Hofmann et al., 2022; Kaplan et al., 2020). □

## A.2 MoE Loop Scaling Law

Proposition A.2 (MoE loop scaling law equivalence). The MoE loop scaling law recovers the dense loop scaling law at E=1, the non-looped MoE scaling law at $R { = } 1$ , and the standard dense scaling law at $E { = } 1$ and $R { = } 1$ , with coeficient reparameterization where required.

Proof.

Reduced form $\scriptstyle { \mathcal { L } } | _ { E = 1 }$ . With expert count at $E { = } 1$ , we recover the dense case, for which $m { = } 1$ and $N _ { \mathrm { a c t } } { = } N$ Since $\kappa _ { j } ( m ) | _ { m = 1 } = \kappa _ { j } m ^ { - \theta } | _ { m = 1 } = \kappa _ { j } 1 ^ { - \theta } = \kappa _ { j }$ for $j = 1 , 2$ , the recurrence mapping in Eq. 7 therefore reduces to the dense looped model formula:

$$
\begin{array} { r l } & { N _ { \mathrm { e f f } } ( R , m ) | _ { E = 1 } = \ \left[ N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) \right] \Big | _ { m = 1 , N _ { \mathrm { a c t } } = N } } \\ & { \quad \quad \quad = N + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right) = N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( R ) . } \end{array}
$$

Let $\hat { E } _ { 1 } \equiv \hat { E } | _ { E = 1 }$ . We reparameterize the coeficients as $\widetilde { A } = A \hat { E } _ { 1 } ^ { \delta } , \ : \widetilde { \alpha } = \alpha + \gamma \ln \hat { E } _ { 1 } , \ : \widetilde { B } = B \hat { E } _ { 1 } ^ { \omega }$ , and $\widetilde { \beta } = \beta + \zeta \ln \hat { E } _ { 1 }$ . The MoE loop scaling law in $\operatorname { E q . 9 }$ then reduces to

$$
\begin{array} { r l } & { \mathcal { L } ( N _ { \mathrm { a c t } } , D , R , E , m ) | _ { E = 1 } = A \hat { E } _ { 1 } ^ { \delta } N _ { \mathrm { e f f } } ( R , 1 ) ^ { \alpha + \gamma \ln \hat { E } _ { 1 } } + B \hat { E } _ { 1 } ^ { \omega } D ^ { \beta + \zeta \ln \hat { E } _ { 1 } } + c } \\ & { \quad \quad \quad = \widetilde { A } N _ { \mathrm { e f f } } ^ { \mathrm { b o u n d e d } } ( R ) ^ { \widetilde { \alpha } } + \widetilde { B } D ^ { \widetilde { \beta } } + c } \\ & { \quad \quad \quad = \mathcal { L } ( N , D , R ) . } \end{array}\tag{15}
$$

This reduced form is equivalent to the loop scaling law in Eq. 6 for dense looped transformers.

Reduced form $\scriptstyle { \mathcal { L } } | _ { R = 1 }$ . With recurrence fixed at R=1, the recurrence mapping in Eq. 7 reduces to

$$
\begin{array} { r l } & { \left. N _ { \mathrm { e f f } } ( R , m ) \right| _ { R = 1 } = \left. \left[ N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) \right] \right| _ { R = 1 } } \\ & { \qquad = N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { 0 } \right) = N _ { \mathrm { a c t } } . } \end{array}
$$

The MoE loop scaling law in Eq. 9 then reduces to

$$
\begin{array} { r l } & { \mathcal { L } ( N _ { \mathrm { a c t } } , D , R , E , m ) | _ { R = 1 } = A \hat { E } ^ { \delta } N _ { \mathrm { e f f } } ( 1 , m ) ^ { \alpha + \gamma \ln \hat { E } } + B \hat { E } ^ { \omega } D ^ { \beta + \zeta \ln \hat { E } } + c } \\ & { \phantom { \mathcal { L } ( N _ { \mathrm { a c t } } , D , R , E , m ) | _ { R = 1 } } = A \hat { E } ^ { \delta } N _ { \mathrm { a c t } } ^ { \alpha + \gamma \ln \hat { E } } + B \hat { E } ^ { \omega } D ^ { \beta + \zeta \ln \hat { E } } + c } \\ & { \phantom { \mathcal { L } ( N _ { \mathrm { a c t } } , D , E ) } = \mathcal { L } ( N _ { \mathrm { a c t } } , D , E ) . } \end{array}\tag{16}
$$

This reduced form is equivalent to the MoE scaling law for non-looped MoE models (Ludziejewski et al., 2025).

Reduced form $\scriptstyle { \mathcal { L } } | _ { R = 1 , E = 1 }$ . Setting both R=1 and E=1 gives m=1, N =N, and $\kappa _ { j } ( m ) | _ { m = 1 } = \kappa _ { j } m ^ { - \theta } | _ { m = 1 } =$ $\kappa _ { j } 1 ^ { - \theta } = \kappa _ { j } \mathrm { f o r } j = 1 , 2$ . The recurrence mapping in Eq. 7 thus reduces to the dense non-looped baseline:

$$
\begin{array} { r l } & { N _ { \mathrm { e f f } } ( R , m ) | _ { R = 1 , E = 1 } = \left. \left[ N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) \right] \right| _ { R = 1 , m = 1 , N _ { \mathrm { a c t } } = N _ { \mathrm { b r } } } } \\ & { \qquad = N + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { 0 } \right) = N . } \end{array}
$$

Using the same reparameterized coeficients $\widetilde { A } = A \hat { E } _ { 1 } ^ { \delta } , \widetilde { \alpha } = \alpha + \gamma \ln \hat { E } _ { 1 } , \widetilde { B } = B \hat { E } _ { 1 } ^ { \omega }$ , and $\widetilde { \beta } = \beta + \zeta \ln \hat { E } _ { 1 }$ as in Eq. 15, the MoE loop scaling law in Eq. 9 then reduces to

$$
\begin{array} { r l } & { \mathcal { L } ( N _ { \mathrm { a c t } } , D , R , E , m ) | _ { R = 1 , E = 1 } = A \hat { E } _ { 1 } ^ { \delta } N _ { \mathrm { e f f } } ( 1 , 1 ) ^ { \alpha + \gamma \ln \hat { E } _ { 1 } } + B \hat { E } _ { 1 } ^ { \omega } D ^ { \beta + \zeta \ln \hat { E } _ { 1 } } + c } \\ & { \quad \quad \quad \quad = \widetilde { A } N ^ { \widetilde { \alpha } } + \widetilde { B } D ^ { \widetilde { \beta } } + c } \\ & { \quad \quad \quad \quad = \mathcal { L } ( N , D ) . } \end{array}\tag{17}
$$

This reduced form is equivalent to the standard non-looped scaling law in Eq. 1 (Hofmann et al., 2022; Kaplan et al., 2020). □

## B Expert-Path Diversity Across Recurrence

Expert-path diversity. The sparsity-conditional mapping is motivated by MoE sparse routing: a token may access diferent experts across recurrent passes. We measure this directly from the router decisions of trained looped MoE models. For an input token t, looped MoE layer ℓ, and pass r, let $\mathcal { E } _ { t , \ell , \prime }$ be the top-k experts selected from $E _ { \mathrm { r o u t e } }$ routed experts. We define the expert-path diversity as

$$
\Psi ( R ) = \frac { 1 } { k } \mathbb { E } _ { t , \ell } \left[ \left| \bigcup _ { r = 1 } ^ { R } \mathcal { E } _ { t , \ell , r } \right| \right] ,\tag{18}
$$

where the union deduplicates experts reused across recurrent passes, the expectation averages over tokens and the $n _ { \ell , \mathrm { l o o p } }$ looped layers, and normalizing by k expresses the expert-path diversity per active expert. At $R { = } 1$ (non-looped) or $E _ { \mathrm { r o u t e } } { = } k { = } 1$ (dense), $\Psi { = } 1$ by definition, as a single pass or a single expert does not have path diversity over recurrent passes.

Table B.1 Expert-path diversity Ψ across varying recurrence R and expert count $E \ ( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td>E</td><td>R=1</td><td>R=2</td><td>R=4</td><td>R=8</td></tr><tr><td>1</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td></tr><tr><td>2</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 1 2 1 2 \pm 0 . 0 6 9 2$ </td><td> $1 . 2 1 6 5 \pm 0 . 0 4 6 5$ </td><td> $1 . 5 0 8 8 \pm 0 . 0 5 2 9$ </td></tr><tr><td>4</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 1 9 1 7 \pm 0 . 0 9 7 4$ </td><td> $1 . 2 9 7 3 \pm 0 . 0 7 0 3$ </td><td> $1 . 5 3 0 0 \pm 0 . 0 7 8 5$ </td></tr><tr><td>8</td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $1 . 2 0 9 9 \pm 0 . 0 8 6 8$ </td><td> $1 . 4 1 5 9 \pm 0 . 0 9 0 1$ </td><td> $1 . 7 0 3 5 \pm 0 . 0 8 6 1$ </td></tr></table>

Empirical measures of Ψ. We report Ψ for looped models across varying recurrence and expert count, where Ψ is computed on the same validation set of 800 prompts from diferent tasks, e.g., math, code, and knowledge. Table B.1 shows that at fixed recurrence, expert-path diversity Ψ increases with larger expert count consistently. This indicates that greater sparsity lets each recurrent pass reach a broader set of expert parameters. As Ψ is only an empirical routing statistic, we introduce sparsity as a condition for modeling the modulated efective-parameter gain (Eq. 7).

## C Scaling Experiments and Parametric Fitting

## C.1 Training Details

Model training setups. We follow the experimental setup similar to recent work on MoE scaling laws (Chen et al., 2026), using the same pre-training data. All models are trained from scratch on the same open-licensed, web-heavy data mixture, supplemented with math, code, knowledge, and science data. We train with a context length of 2,048 and sequence packing, tied input-output embeddings, and RoPE with base frequency 500,000. We use the llama tokenizer with a vocabulary of 202k. We use AdamW with $\beta _ { 1 } { = } 0 . 9 , ~ \beta _ { 2 } { = } 0 . 9 5$ $\epsilon { = } 1 0 ^ { - 1 5 }$ , weight decay 0.1, and gradient clipping at 1.0. Model weights are trained in BF16, while optimizer states, gradients, and MoE router computations are maintained in FP32. All scaling-sweep experiments run on 8 nodes (64 NVIDIA H100 96 GB GPUs) with a global batch size of 3,072 and a sequence length of 2,048, corresponding to approximately 6.29 million tokens per batch.

Looped MoE setups. For MoE models, we adopt sigmoid gating with per-token top-k normalization, auxiliary-loss-free load balancing with bias-update rate $\lambda _ { \mathrm { l b } } { = } 1 0 ^ { - 3 }$ , router z-loss with coeficient $\lambda _ { z } { = } 1 0 ^ { - 4 }$ , and drop-and-pad token dispatch with capacity factor 1.5. For the scaling sweep experiments used for loop scaling laws in Sections 3.2–4.1, we apply the next-token prediction loss to the output of the final recurrent pass. For the looped MoE in Table 2, we add per-loop supervision (Bae et al., 2024) to allow the same trained checkpoint to be evaluated at diferent recurrence at test time.

## C.2 Model Ladder and Scaling Sweep

Model ladder setups. We construct a model ladder with five active model scales $N _ { { \mathrm { a c t } } } \in \{ 0 . 3 , 0 . 6 , 1 . 0 , 1 . 6 , 2 . 4 \} \mathrm { B }$ Each rung fixes its own model configuration $d _ { m } , n _ { h } , n _ { \ell } , n _ { \ell , \mathrm { l o o p } } ,$ and $N _ { \mathrm { l o o p } }$ . Here, $d _ { m }$ is the model dimension; $n _ { h }$ and n<sub>KV</sub> are the numbers of attention and KV heads; $n _ { \ell }$ and $n _ { \ell , \mathrm { l o o p } }$ are the numbers of total and recurrent layers; $d _ { h }$ is the head dimension; $N _ { \mathrm { a c t } } , N _ { \mathrm { t o t a l } }$ denote the active and total parameter counts and $N _ { \mathrm { l o o p } }$ is the parameter count of the recurrent block. All configurations use n =4 and $d _ { h } { = } 6 4$ , with $n _ { h } = d _ { m } / d _ { h }$ . Since we adopt the standard middle-block looping (Geiping et al., 2025), the first two and last two layers remain unshared, giving $n _ { \ell , \mathrm { l o o p } } = n _ { \ell } - 4$ . Table C.1 summarizes the model ladder setups.

Table C.1 Model scaling ladder over five model scales. Parameter counts are in billions (B). $N _ { \mathrm { a c t } }$ and $N _ { \mathrm { t o t a l } }$ include embedding parameters, whereas $N _ { \mathrm { l o o p } }$ includes only parameters in the recurrent block.
<table><tr><td rowspan="2"> $N _ { \mathrm { { a c t } } }$  (B) |</td><td colspan="4">Architecture dimensions</td><td rowspan="2">1  $N _ { \mathrm { l o o p } } ~ \mathrm { { ( B ) } }$ </td><td colspan="5"> $N _ { \mathrm { t o t a l } } ( E ) \ ( \mathrm { B } )$ </td></tr><tr><td> $d _ { m }$ </td><td> $n _ { h }$ </td><td>nl</td><td> $^ { n _ { \ell , \mathrm { l o o p } } }$  I</td><td>一  $E { = } 1$ </td><td> $E { = } 2$ </td><td> $E { = } 4$ </td><td> $E { = } 8$ </td><td> $E { = } 1 6$ </td></tr><tr><td>0.3</td><td>768</td><td>12</td><td>20</td><td>16</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.8</td><td>1.3</td><td>2.5</td></tr><tr><td>0.6</td><td>1024</td><td>16</td><td>26</td><td>22</td><td>0.3</td><td>0.6</td><td>0.9</td><td>1.6</td><td>2.9</td><td>5.5</td></tr><tr><td>1.0</td><td>1280</td><td>20</td><td>32</td><td>28</td><td>0.7</td><td>1.0</td><td>1.6</td><td>2.9</td><td>5.4</td><td>10.5</td></tr><tr><td>1.6</td><td>1536</td><td>24</td><td>38</td><td>34</td><td>1.2</td><td>1.6</td><td>2.7</td><td>4.8</td><td>9.1</td><td>17.7</td></tr><tr><td>2.4</td><td>1792</td><td>28</td><td>44</td><td>40</td><td>1.8</td><td>2.4</td><td>4.1</td><td>7.5</td><td>14.3</td><td>27.8</td></tr></table>

Scaling sweep runs. Our experimental scaling sweep uses $N _ { \mathrm { a c t } } \in \{ 0 . 3 , 0 . 6 , 1 . 0 \} \mathrm { B }$ across training tokens $D \in \{ 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 , 5 0 0 \} \mathrm { B }$ , expert counts $E \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ which vary sparsity and active-parameter ratios m, and recurrences $R \in \{ 1 , 2 , 3 , 4 , 6 , 8 \}$ . Following Kaplan et al. (2020), the fitted parameter counts exclude embeddings.

Model optimization design space. For the resource-constrained joint optimization of recurrence and sparsity in Figure 6(c) (Section 4.2), we evaluate the fitted law over five model scales on the model ladder: $N _ { { \mathrm { a c t } } } \in \{ 0 . 3 , 0 . 6 , 1 . 0 , 1 . 6 , 2 . 4 \} \mathrm { B }$ , expert counts $E \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ , and recurrences $R \in \{ 1 , \ldots , 1 0 \}$ . This gives 300 candidate configurations, where $N _ { \mathrm { a c t } } \in \{ 1 . 6 , 2 . 4 \} \mathrm { B }$ , expert counts $E > 1 6$ , and recurrences R>8 lie in the extrapolation range of the fitted law.

## C.3 Parametric Fitting

Fitting the dense loop scaling laws. For fitting the dense loop scaling law, we instantiate the efective parameter count $N _ { \mathrm { e f f } } ( R )$ in $\operatorname { E q }$ . 6 using one of the following recurrence mappings:

$$
N _ { \mathrm { e f f } } ( R ) = \left\{ \begin{array} { l l } { N + ( R - 1 ) N _ { \mathrm { l o o p } } , } & { \mathrm { l i n e a r , } } \\ { N + ( R ^ { \varphi } - 1 ) N _ { \mathrm { l o o p } } , } & { \mathrm { p o w e r ~ l a w , } } \\ { N + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right) , } & { \mathrm { b o u n d e d . } } \end{array} \right.\tag{19}
$$

Here, N is the stored model parameter count and $N _ { \mathrm { l o o p } }$ is the parameter count of the recurrent block, as defined in Section 3.1. The linear mapping coincides with the unrolled parameter count $N _ { \mathrm { u n r o l l } } ( R )$ , whereas the power-law and bounded mappings model the efective-parameter gain from looping. The linear mapping introduces no recurrence-specific fitted coeficient, while the power-law and bounded mappings introduce $\varphi$ and $\left\{ \kappa _ { 1 } , \kappa _ { 2 } \right\}$ , respectively.

At R=1, the dense loop scaling law recovers the standard non-looped scaling law (Proposition A.1, Eq. 14). We therefore derive the loop scaling law using a staged fitting procedure: first establish the non-looped scaling law with base coeficients $\{ A , \alpha , B , \beta , c \}$ using the $R { = } 1$ subset of the scaling sweep across model size N and data $D _ { \ast }$ , and then extend it to the loop scaling law to jointly estimate all coeficients using the scaling sweep across $R , N .$ , and $D .$ For a fair comparison in Figure 2, we apply the same fitting procedure to the loop scaling law under the linear and power-law mappings (Eq. 3) and the bounded mapping $\left( \mathrm { E q . ~ 4 } \right)$ , and evaluate each formulation on the same held-out R, N, and $D _ { : }$ , where the held-out slices are $R { = } 1 6 , \ N { = } 1 . 0 8$ , and $D \in ( 4 0 0 , 5 0 0 ] \mathrm { B }$ . Figure $2 ( \mathrm { c ) }$ reports the RMSE on each held-out slice after fitting on the remaining runs.

Fitting the MoE loop scaling laws. For fitting the MoE loop scaling law, we instantiate the efective parameter count $N _ { \mathrm { e f f } } ( R , m )$ in Eq. 9 using one of the following recurrence mappings:

$$
N _ { \mathrm { e f f } } ( R , m ) = \left\{ \begin{array} { l l } { N _ { \mathrm { a c t } } + ( R - 1 ) N _ { \mathrm { l o o p } } , } & { \mathrm { l i n e a r , } } \\ { N _ { \mathrm { a c t } } + ( R ^ { \varphi } - 1 ) N _ { \mathrm { l o o p } } , } & { \mathrm { p o w e r ~ l a w , } } \\ { N _ { \mathrm { a c t } } + \kappa _ { 1 } N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } } \right) , } & { \mathrm { b o u n d e d , } } \\ { N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right) , } & { \mathrm { s p a r s i t y - c o n d i t i o n a l . } } \end{array} \right.\tag{20}
$$

For the sparsity-conditional mapping, $\kappa _ { j } ( m ) = \kappa _ { j } m ^ { - \theta }$ for $j \in \{ 1 , 2 \}$ . Here, $N _ { \mathrm { a c t } }$ is the active parameter count, $N _ { \mathrm { l o o p } }$ is the active parameter count of the recurrent block, and $m = N _ { \mathrm { a c t } } / N _ { \mathrm { t o t a l } }$ is computed from the active and total parameter counts as defined in Section 3.1. The linear mapping coincides with $N _ { \mathrm { u n r o l l } } ( R )$ while the linear, power-law, and bounded recurrence mappings are independent of $m ;$ their outer MoE scaling laws retain the dependence on expert count $E .$ The linear mapping introduces no recurrence-specific fitted coeficient, while the power-law, bounded, and sparsity-conditional mappings introduce $\varphi , \ \{ \kappa _ { 1 } , \kappa _ { 2 } \}$ , and $\{ \kappa _ { 1 } , \kappa _ { 2 } , \theta \}$ , respectively.

At R=1, the MoE loop scaling law recovers the standard non-looped MoE scaling law (Proposition A.2, Eq. 16). We derive the MoE loop scaling law using a staged fitting procedure: first establish the non-looped MoE scaling law with base coeficients $\left\{ A , \alpha , B , \beta , c , \delta , \gamma , \omega , \zeta , E _ { \mathrm { s t a r t } } , E _ { \mathrm { m a x } } \right\}$ using the $R { = } 1$ subset of the scaling sweep across active model size $N _ { \mathrm { a c t } }$ , training tokens $D _ { : }$ and expert count $E ;$ and then we extend it to the MoE loop scaling law to jointly estimate all coeficients using the full sweep across the four scaling axes: $R ,$ $N _ { \mathrm { a c t } } , \ D$ , and E. For a fair comparison in Figure 3, we apply the same fitting procedure to the MoE loop scaling law under the linear, sparsity-independent power-law, bounded, and sparsity-conditional recurrence mappings, and evaluate each formulation on the same held-out $R , E , N _ { \mathrm { a c t } }$ , and $D _ { ; }$ , where the held-out slices are $R { = } 1 6 , E { = } 1 6 , N _ { \mathrm { a c t } } { = } 1 . 0 8$ , and D∈(400, 500]B. In Figure 3(c), we report the RMSE scores on the held-out runs of recurrence $R ,$ expert count $E ,$ model $N _ { \mathrm { a c t } }$ , and data $D ,$ after fitting each law on the remaining runs.

Numerical optimization and final fitted coeficients. We fit the final MoE loop scaling law using more than 1,000 loss observations from the scaling sweep across four axes $N _ { \mathrm { a c t } } , D , E , R$ (Appendix C.2). We initialize the scaling-law fit using scipy.optimize.curve\_fit with nonlinear least squares and an MSE objective, and then refine the fit using scipy.optimize.minimize with L-BFGS-B optimization and a log-Huber objective. Table C.2 reports the final fitted coeficients of the MoE Loop Scaling Law in Eq. 9, using all the scaling sweep runs over recurrence $R ,$ expert count $E ,$ model $N _ { \mathrm { a c t } }$ , and data D.

Table C.2 Fitted coeficients of MoE Loop Scaling Law in Eq. 9. RMSE: root-mean-square error of the fit.
<table><tr><td colspan="5">Base scaling law coefficients</td><td colspan="6">MoE coefficients</td><td colspan="3">Recurrence coefficients</td><td></td><td>Fit</td></tr><tr><td>A</td><td>α</td><td></td><td>B</td><td>β</td><td>c |</td><td>δ</td><td>γ</td><td>ω</td><td>ζ</td><td> $E _ { \mathrm { s t a r t } }$ </td><td> $E _ { \mathrm { m a x } }$ </td><td> $\kappa _ { 1 }$ </td><td>κ2</td><td></td><td>θ| |RMSE</td></tr><tr><td>0.8085</td><td>-0.1951 3.4291</td><td></td><td>-0.7548 1.3555</td><td></td><td></td><td>-0.1935</td><td>-0.0137</td><td></td><td></td><td></td><td>-0.2056 0.0881 1.3174 57.1201 |0.3682 1.4037</td><td></td><td></td><td>0.3286</td><td>0.0037</td></tr></table>

Sparsity exponent ablation. We ablate a less constrained variant of Eq. 7 with separate sparsity exponents, $\kappa _ { 1 } ( m ) = \kappa _ { 1 } m ^ { - \theta _ { 1 } }$ and $\kappa _ { 2 } ( m ) = \kappa _ { 2 } m ^ { - \theta _ { 2 } }$ . The RMSE remains 0.0037 on the final fitted law. As adding extra exponent yields no meaningful gain, we adopt the shared-θ formulation.

## C.4 Bootstrap Results of the MoE Loop Scaling Law

To quantify uncertainty in our fitted MoE loop scaling law in Table C.2, we use a bootstrapping protocol similar to prior works (Hofmann et al., 2022; Ludziejewski et al., 2025). We repeatedly sample 90% of the observations without replacement 100 times, and refit the law on each sampled subset. For the fitted coeficients, we report the full-data estimates and 90% bootstrap percentile confidence intervals (CIs), defined by the 5th and 95th percentiles over the 100 refit runs.

Table C.3 Bootstrap results on the MoE loop scaling law. (a) Coeficients fitted on full-data or the 90% sampled subsets with intervals spanning the 5th–95th percentiles over 100 refit runs. (b) Derived joint optimum under the compute and memory budgets (with INT4 weights) used in Figure 6(c) on the 100 refit runs.  
(a) Coeficient uncertainty
<table><tr><td>Coefficient</td><td>Estimate</td><td>90% CI</td></tr><tr><td>κ1</td><td>0.3682</td><td>[0.3658, 0.3723]</td></tr><tr><td>κ2</td><td>1.4037</td><td>[1.3896, 1.4243]</td></tr><tr><td>θ</td><td>0.3286</td><td>[0.3183, 0.3350]</td></tr></table>

(b) Joint-optimum uncertainty
<table><tr><td>Compute budget Memory budget</td><td></td><td> $N _ { \mathbf { a c t } } ^ { \star }$  90%CI</td><td>E* 90% CI</td><td>R* 90% CI</td></tr><tr><td> $5 \times 1 0 ^ { 2 1 }$ </td><td>1 GB</td><td>[1.0, 1.0]B</td><td>[2, 2]</td><td>[2, 2]</td></tr><tr><td> $5 \times 1 0 ^ { 2 1 }$ </td><td>3 GB</td><td>[1.0, 1.0]B</td><td>[8, 8]</td><td>[2, 2]</td></tr><tr><td> $5 \times 1 0 ^ { 2 1 }$ </td><td>10 GB</td><td>[1.6, 1.6]B</td><td>[16, 16]</td><td>[1, 1]</td></tr></table>

Table C.3(a) presents the bootstrap results for coeficients of the sparsity-conditional recurrence mapping. The narrow intervals of 100 refits indicate these coeficient estimates remain stable under bootstrapping. Table C.3(b) shows the 90% bootstrap CIs of the joint optimum for resource-constrained architecture selection in Figure 6(c) (Section 4.2). The collapsed intervals indicate the selected configurations remain the same across 100 refits, and confirm that the fitted law yields robust estimates for architecture optimization under compute and memory budgets.

## D Downstream Scaling and Evaluation

## D.1 Downstream Evaluation Protocols

Table D.1 Evaluation setups of 14 foundational benchmarks. Task names and metrics follow lm-eval.
<table><tr><td>Category</td><td>Benchmark</td><td>Task name</td><td></td><td>n-shot Metric</td></tr><tr><td rowspan="2">Reasoning</td><td>BBH</td><td>leaderboard_bbh</td><td>3</td><td>acc_norm</td></tr><tr><td>GSM8K</td><td>gsm8k_cot</td><td>8</td><td>exact_match,flexible-extract</td></tr><tr><td rowspan="3">Science</td><td>ARC-C</td><td>arc_challenge</td><td>25</td><td>acc_norm</td></tr><tr><td>ARC-E</td><td>arc_easy</td><td>0</td><td>acc_norm</td></tr><tr><td>OBQA</td><td>openbookqa</td><td>0</td><td>acc_norm</td></tr><tr><td rowspan="4">Commonsense Reasoning</td><td>HellaSwag</td><td>hellaswag</td><td>0</td><td>acc_norm</td></tr><tr><td>PIQA</td><td>piqa</td><td>0</td><td>acc_norm</td></tr><tr><td>SIQA</td><td>social_iqa</td><td>0</td><td>acc</td></tr><tr><td>WinoGrande</td><td>winogrande</td><td>0</td><td>acc</td></tr><tr><td rowspan="2">Reading</td><td>BoolQ</td><td>boolq</td><td>0</td><td>acc</td></tr><tr><td>DROP</td><td>drop</td><td>3</td><td>f1</td></tr><tr><td rowspan="3">Knowledge</td><td>MMLU</td><td>mmlu</td><td>555</td><td>acc</td></tr><tr><td>NQ</td><td>nq_open</td><td></td><td>exact_match</td></tr><tr><td>TQA</td><td>triviaqa</td><td></td><td>exact_match</td></tr></table>

Evaluation protocols. We evaluate the models in Section 4.3 on 14 widely used benchmarks spanning five core competencies: (1) reasoning (BBH (Suzgun et al., 2023), GSM8K (Cobbe et al., 2021)), (2) science (ARC-C/E (Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018)), (3) commonsense (HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), WinoGrande (Sakaguchi et al., 2020)), (4) reading (BoolQ (Clark et al., 2019), DROP (Dua et al., 2019)), and (5) knowledge (MMLU (Hendrycks et al., 2020), Natural Questions (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017)). We use the Language Model Evaluation Harness (lm-eval) (Gao et al., 2024) with the vLLM backend, bfloat16 model precision, and greedy decoding for generation-based tasks. Detailed benchmark-specific few-shot settings and metrics are given in Table D.1. For compactness in Table 2, we use the following benchmark abbreviations: BBH: BIG-Bench Hard, ARC-C/E: ARC-Challenge/Easy, OBQA: OpenBookQA, HS: HellaSwag, Wino: WinoGrande, NQ: Natural Questions, TQA: TriviaQA.

## D.2 Compute-Matched Trillion-Token Experiment

Scaling-law-guided model design. In Section 4.3, we compare a looped MoE model with a non-looped MoE reference under matched training compute. Given fixed compute $\mathrm { F L O P s } .$ we derive the compute-optimal recurrence for the looped MoE model by evaluating each candidate recurrence on an IsoFLOP basis, as summarized in Algorithm D.1.

Algorithm D.1: IsoFLOP recurrence selection   
Input: compute budget $\bar { F } _ { \mathrm { t r a i n } }$ or reference model $( N _ { \mathrm { r e f } } , D _ { \mathrm { r e f } } )$ ; target looped MoE architecture $( N _ { \mathrm { a c t } } , N _ { \mathrm { l o o p } } , E , m )$   
recurrence set $\mathcal { R } ;$ tolerance ϵ.   
1. If a reference model is provided, set $\bar { F } _ { \mathrm { t r a i n } } = 6 N _ { \mathrm { r e f } } D _ { \mathrm { r e f } } .$   
2. For each $R \in \mathcal R$ of the target looped MoE, compute its unrolled parameter count and compute-matched training-token   
budget:   
$N _ { \mathrm { u n r o l l } } ( R ) = N _ { \mathrm { a c t } } + ( R - 1 ) N _ { \mathrm { l o o p } } , \qquad D _ { \mathrm { t a r g e t } } ^ { R } = { \bar { F } } _ { \mathrm { t r a i n } } / [ 6 N _ { \mathrm { u n r o l l } } ( R ) ]$   
3. Compute the looping efective parameter count and the predicted loss using Eqs. 7 and 9:   
$N _ { \mathrm { e f f } } ( R , m ) = N _ { \mathrm { a c t } } + \kappa _ { 1 } ( m ) N _ { \mathrm { l o o p } } \left( 1 - e ^ { - ( R - 1 ) / \kappa _ { 2 } ( m ) } \right)$   
$\mathcal { L } _ { R } = \mathcal { L } ( N _ { \mathrm { a c t } } , D _ { \mathrm { t a r g e t } } ^ { R } , R , E , m )$   
4. Select the compute-optimal recurrence according to Eq. 11:   
$R ^ { \star } = \arg \operatorname* { m i n } _ { R } \mathcal { L } ( N _ { \mathrm { a c t } } , D _ { \mathrm { t a r g e t } } ^ { R } , R , E , m ) ,$   
s.t. $6 N _ { \mathrm { u n r o l l } } ( R ) D _ { \mathrm { t a r g e t } } ^ { R } = \bar { F } _ { \mathrm { t r a i n } } , \quad \Delta \mathcal { L } ( R ) \geq \epsilon .$   
Output: Compute-optimal recurrence $R ^ { \star }$

We set ϵ to the fitted-law RMSE, which measures the predictive discrepancy in the model loss. A marginal improvement below this resolution is treated as indistinguishable from fitting error and thus insuficient to justify the cost of an additional recurrent pass.

Matched-compute training and test-time evaluation. For Table 2, the procedure above selects $R ^ { \star } { = } 5$ for the A0.3B-1.3B LoopMoE $( E { = } 8 )$ , which we compare with an A0.6B-2.9B non-looped MoE $( E { = } 8 , R { = } 1 )$ The models are trained for ∼3T and ∼6T tokens, respectively, which correspond to approximately $1 . 5 \times 1 0 ^ { 2 2 }$ FLOPs. We calculate the compute FLOPs as $F _ { \mathrm { t r a i n } } = 6 N _ { \mathrm { u n r o l l } } ( R ) D$ , where D is the training tokens and $N _ { \mathrm { u n r o l l } } ( R )$ is the unrolled parameter count at recurrence R. A0.3B-1.3B LoopMoE is trained as described in Appendix C.1. As LoopMoE gives valid output per recurrent pass, we evaluate the model by varying recurrence $R \in \{ 1 , 2 , \ldots , 5 \}$ at test time, to switch inference efort from low to high on demand. To compare the inference compute of A0.3B-1.3B LoopMoE and A0.6B-2.9B non-looped MoE, we report their relative inference compute ratio as $F _ { \mathrm { i n f } } ^ { 0 . 3 \mathrm { B } } ( R ) / F _ { \mathrm { i n f } } ^ { 0 . 6 \bar { \mathrm { B } } } ( 1 )$ , where $F _ { \mathrm { i n f } } ( R ) = 2 N _ { \mathrm { u n r o l l } } ( R )$