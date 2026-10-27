# Why Are LLMs Confidently Wrong? Correcting Overconfident Errors via Causal Head Intervention

Jing Ren<sup>1,#</sup> Bowen Li<sup>1,#</sup> Ziqi Xu<sup>1,\*</sup> Xuechao Yang<sup>1</sup> Feng Xia<sup>1</sup>

<sup>1</sup>RMIT University

{S4065838, S3890442}@student.rmit.edu.au {ziqi.xu, xuechao.yang, feng.xia}@rmit.edu.au

<sup>#</sup>Equal contribution <sup>\*</sup>Corresponding author

<sup>§</sup> https://aisuko.github.io/confix

## Abstract

Modern large language models (LLMs) often exhibit overconfident errors, assigning high confidence to incorrect predictions and thereby undermining reliability in real-world use. Existing calibration methods typically rely on post-hoc adjustment or retraining, operating at the output level without addressing the underlying causes of miscalibration. We propose CoCHI (Class-oriented Causal Head Intervention), a framework that improves calibration by identifying and intervening on internal components responsible for distinct confidence behaviours. By analysing predictions through correctness-confidence patterns, CoCHI enables targeted intervention on model internals, reducing overconfident errors while preserving correct predictions. Our approach provides a mechanistic perspective on calibration, linking reliability to identifiable structures within the model rather than treating it as an output-level issue. Experiments across multiple LLMs show consistent improvements in calibration metrics, including Expected Calibration Error, Negative Log-Likelihood, and Brier Score, while maintaining accuracy on five multiple-choice question answering benchmarks. These results suggest that effective calibration requires not only adjusting outputs, but also addressing the underlying causal mechanisms within the model.

## 1 Introduction

Modern neural models have achieved remarkable performance across a wide range of tasks. However, they frequently produce overconfident errors, assigning high confidence to incorrect predictions [1, 2, 3, 4] and thereby undermining reliability in high-stakes applications. This issue is further amplified in large language models (LLMs) [5, 6, 7], which have become a dominant paradigm in modern AI and are widely deployed in domains such as healthcare, education, and scientific reasoning, where incorrect yet confident outputs can lead to significant downstream risks [8, 9, 10, 11, 12].

To address this problem, a large body of work has focused on calibration, aiming to align predicted confidence with empirical accuracy [13, 14, 15]. Existing approaches, such as temperature scaling and its variants, primarily operate at the output level by adjusting logits or confidence scores, while more recent methods introduce learned calibrators or inference-time interventions [16]. Despite their effectiveness in some settings, these approaches share a fundamental limitation: they treat miscalibration as a surface-level phenomenon and do not address the internal mechanisms that give rise to overconfident errors. As a result, they often fail to generalise and provide limited insight into why models are confidently wrong.

In this work, we take a different perspective and ask: what internal components of a model are responsible for overconfident errors, and can we intervene on them directly? Recent advances in mechanistic interpretability [17, 18, 19, 20] have made significant progress in moving beyond correlational post-hoc explanations toward causally grounded, intervention-based understanding of how LLMs implement computations internally. In particular, causal intervention methods such as causal head gating [17], causal mediation analysis [18, 21], and causal tracing [19] allow us to identify internal components that are causally responsible for specific model behaviours by systematically perturbing internal activations and observing the resulting change in the model’s output distribution. However, existing interpretability approaches largely focus on post-hoc analysis and explanation, rather than leveraging these insights to actively improve model reliability. Consequently, the connection between internal computational circuits and miscalibration remains largely unexplored.

To bridge this gap, we propose CoCHI (Class-oriented Causal Head Intervention), a framework that improves calibration by directly targeting the internal causes of overconfident errors in large language models. Our key insight is that miscalibration is not merely an output-level issue, but arises from specific internal components that systematically promote incorrect yet highly confident predictions or introduce noise into otherwise correct ones. To capture this, we decompose model predictions into four calibration classes based on correctness and confidence, and perform class-conditioned causal analysis to identify attention heads associated with distinct behaviours. Based on this analysis, CoCHI constructs a structured intervention that selectively suppresses heads responsible for overconfident errors while removing non-contributing components in correct predictions. This results in a minimal, targeted modification to the model’s internal computation that improves calibration without altering model parameters, providing a mechanism-aware alternative to conventional post-hoc approaches.

## Contributions. Our main contributions are as follows:

• We propose a structural causal mask for calibration, operating directly on model internals rather than relying on post-hoc output adjustment.

• We partition model predictions into four fine-grained classes based on correctness and confidence, enabling precise identification of attention heads that causally promote overconfident errors or fail to support correct predictions.

• We introduce CoCHI (Class-oriented Causal Head Intervention), an inference-time framework that performs class-aware causal analysis and constructs a structured mask to selectively suppress problematic attention heads.

• Extensive experiments across five multiple-choice question answering benchmarks show that CoCHI consistently improves calibration metrics while preserving accuracy, outperforming traditional post-hoc calibration methods.

## 2 Related Work

Uncertainty Estimation and Overconfident Errors. A large body of work studies uncertainty estimation in neural models, including Bayesian approaches [22, 23, 24, 25, 26], deep ensembles [27], and post-hoc techniques such as temperature scaling [16], which rescale logits using a learned temperature parameter. In the context of LLMs, uncertainty estimation and calibration have received increasing attention [28]. Recent efforts extend post-hoc calibration to LLMs, including adaptive temperature scaling [29] and learned calibrators such as Thermometer [30], aiming to better align predicted confidence with empirical accuracy. Other approaches adjust hidden representations, for example by training a linear layer on top of the final hidden states for confidence prediction [31], or exploit prompt-based strategies and self-consistency to estimate uncertainty [32, 33, 34, 35, 15]. Despite their effectiveness in some settings, these methods primarily operate at the output or representation level, but recent work such as Inference-Time Intervention [36] shows that undesirable behaviours (e.g., hallucination) are governed by internal mechanisms. These methods primarily enhance truthful representations, while overlooking the role of mechanisms that specifically drive overconfident errors, where incorrect predictions are assigned high confidence scores. Recent work has shown that attention heads encode rich signals about model uncertainty, enabling unsupervised uncertainty estimation without additional training [37]. However, these approaches primarily focus on extracting uncertainty signals from internal representations, without modifying the model’s behaviour.

This limitation motivates our approach, which targets the structural causal components responsible for overconfident errors and performs structured intervention to improve calibration.

Circuit Discovery in Mechanistic Interpretability Circuit discovery aims to identify the minimal subgraph of a language model responsible for a specific behaviour. The seminal work of [38] manually identified the Indirect Object Identification (IOI) circuit in GPT-2 Small, comprising 28 attention heads across seven functional classes. To scale this process, Conmy et al. [39] proposed Automatic Circuit Discovery (ACDC), an iterative pruning method based on activation patching, which successfully rediscovered known circuits such as Greater-Than and recovered most components of the IOI circuit. Subsequent approaches, including Efficient Attribution Patching (EAP), further improved the efficiency and scalability of automated circuit discovery [40]. Beyond circuit discovery, mechanistic interpretability has increasingly incorporated tools from causal inference [41]. These approaches leverage counterfactual interventions and structured perturbations to uncover how structural heads contribute to model outputs [42, 43, 44, 45]. In particular, causal head gating identifies functional roles of attention heads by assigning graded scores for facilitation, interference, or irrelevance based on their causal effect on task performance, going beyond purely correlational analyses [17]. Recent mechanistic studies have shown that verbalised confidence in LLMs is governed by dedicated internal circuits that are largely decoupled from factual correctness [46]. These findings suggest that overconfidence arises not from miscalibration at the output layer, but from specific internal mechanisms that consistently generate confident responses regardless of their accuracy [47]. Building on this insight, we identify and suppress attention heads that contribute to high-confidence errors, leading to improved calibration and reliability.

## 3 Preliminaries

We briefly introduce the key components for identifying the causal roles of attention heads in transformer models via differentiable intervention.

Transformer with Head-wise Gating. Consider a transformer model with L layers and H attention heads per layer. For each attention head (ℓ, h), we introduce a continuous gating variable $G _ { \ell , h } \in [ 0 , 1 ]$ that scales the output of the head:

$$
Z _ { \ell , h } = G _ { \ell , h } \cdot \left( A _ { \ell , h } V _ { \ell , h } \right) ,\tag{1}
$$

where $A _ { \ell , h }$ denotes the attention weights (after softmax) and $V _ { \ell , h }$ denotes the value projections.

The gated head outputs are concatenated and projected as in standard multi-head attention. By varying $G _ { \ell , h }$ , we perform soft interventions on individual heads without modifying model parameters.

Gate Optimisation. The gating variables are optimised while keeping model parameters frozen. Given a dataset D, we minimise the negative log-likelihood objective:

$$
\begin{array} { r } { \mathcal { L } ( G ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \log P ( y \mid x ; G ) , } \end{array}\tag{2}
$$

where $P ( y \mid x ; G )$ denotes the model output under gating configuration $G .$

Optionally, a regularisation term can be introduced to encourage sparsity or density of the gating values. This formulation allows us to assess the causal contribution of each head by observing how scaling its output affects model behaviour.

Dual Optimisation and Role Identification. To disambiguate causal importance from incidental effects, we optimise two sets of gates:

• $G ^ { + } ;$ : optimised under a retention bias, encouraging gates towards 1,

• $G ^ { - } \colon$ optimised under a sparsity bias, encouraging gates towards 0.

The divergence between $G ^ { + }$ and $G ^ { - }$ reveals the causal role of each head: heads that remain active under sparsity are necessary for the task, while those suppressed even under retention bias are likely harmful.

![](images/40ef2271a2fb130a4e8bf54ad4fa19cd18039b732d47194746d12f063cae5ca4.jpg)  
Figure 1: The framework of CoCHI.

Causal Head Taxonomy. Based on the optimised gates, each attention head is assigned a causal role. Facilitating heads contribute positively to performance and remain active under sparsity $( G ^ { - } )$ indicating necessity. Interfering heads degrade performance and are suppressed even under retention bias $( G ^ { + } )$ , indicating harmful contributions. Irrelevant heads have negligible impact, with gating values varying under different regularisation regimes. This taxonomy provides a causal, rather than correlational, characterisation of model internals, validated through targeted ablation and intervention.

## 4 Methodology

We reframe calibration as a problem of causal intervention on model heads. Specifically, our framework identifies and suppresses causal heads responsible for overconfident errors, while removing sources of noise in otherwise correct predictions. This perspective provides a mechanistic view of miscalibration, linking reliability to identifiable structural components within the model. An overview of CoCHI is shown in Figure 1.

## 4.1 Problem Setup

Let $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ denote a pretrained language model that maps an input x to a probability distribution $p _ { \theta } ( \cdot \mid x )$ over a fixed answer space $\mathsf { \bar { y } } .$ The model prediction is defined as:

$$
{ \hat { y } } ( x ) = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } p _ { \theta } ( y \mid x ) .\tag{3}
$$

The corresponding confidence is defined as:

$$
c ( x ) = \operatorname* { m a x } _ { y \in \mathcal { V } } p _ { \theta } ( y \mid x ) .\tag{4}
$$

Let $y ^ { * }$ denote the ground-truth label. A prediction is correct if $\hat { y } ( x ) = y ^ { * }$ . We focus on the failure mode of overconfident errors, defined as incorrect predictions with high confidence:

$$
\hat { y } ( x ) \neq y ^ { * } , \quad c ( x ) \geq \tau _ { c } ,\tag{5}
$$

where $\tau _ { c }$ is a predefined confidence threshold.

Our goal is to design an inference-time intervention that reduces such errors by targeting their underlying internal causes, thereby improving calibration while preserving predictive accuracy.

## 4.2 Calibration-Class Decomposition

Rather than treating calibration as a global property of output probabilities, we decompose model behaviour into four calibration classes based on correctness and confidence:

$$
\mathcal { C } = \{ \mathsf { c o r r e c t \_ h i g h } , \mathsf { c o r r e c t \_ l o w } , \mathsf { w r o n g \_ h i g h } , \mathsf { w r o n g \_ l o w } \} .
$$

Formally, this defines a partition over predictions induced by correctness $( \hat { y } ( x ) = y ^ { \ast } \ \mathrm { o r } \ \hat { y } ( x ) \neq$ $y ^ { * } )$ and confidence $( c ( x ) \ge \tau _ { c } \mathrm { o r } c ( x ) < \tau _ { c } )$ . This decomposition isolates overconfident errors (wrong\_high) as a distinct subset, enabling us to analyse the structural causal head that give rise to miscalibrated confidence, rather than conflating them with other types of errors.

## 4.3 Class-Conditioned Causal Head Attribution

Following Causal Head Gating (CHG) [17], we attribute model behaviour to individual attention heads by estimating their causal contribution. Let H denote the set of all attention heads across l layers. For each head $h \in \mathcal H$ , CHG computes two scores with respect to a target behaviour defined on a dataset $\mathcal { D } \colon$ a necessary score $s _ { h } ^ { \mathrm { n e c } }$ , measuring the drop in performance when the head is ablated, and a sufficient score $s _ { h } ^ { \mathrm { s u f } }$ , measuring how well the head alone can reproduce the behaviour when isolated.

In our setting, we apply CHG in a class-conditioned manner. For each calibration class $c \in { \mathcal { C } }$ , we construct a subset $\mathcal { D } _ { c }$ consisting of inputs whose predictions fall into class c, and perform CHG using the model’s own predictions as the target signal. This yields a necessary score $s _ { h } ^ { \mathrm { n e c } } ( c )$ and a sufficient score $s _ { h } ^ { \mathrm { s u f } } ( c )$ for each head. The necessary score measures how strongly a head persists when the optimisation is encouraged to minimise the number of active heads, whereas the sufficient score measures how strongly a head remains selected when the optimisation is encouraged to retain all heads compatible with good task performance.

CoCHI is trained in three sequential stages. First, an unregularised optimisation stage $( \lambda = 0 )$ is performed to obtain a shared task-relevant initialisation:

$$
z _ { l , h } ^ { 0 } \xrightarrow [ ] { \mathrm { s t a g e } 1 } \hat { z } _ { l , h } .\tag{6}
$$

The resulting logits $\hat { z } _ { l , h }$ are then copied into two separate optimisation branches:

$$
\hat { z } _ { l , h }  z _ { l , h } ^ { + } , \quad \hat { z } _ { l , h }  z _ { l , h } ^ { - } ,\tag{7}
$$

corresponding to the positive-L1 and negative-L1 stages, respectively.

During each forward pass, the learned logit is converted into a multiplicative gate using a sigmoid function and applied directly to the attention head output:

$$
\tilde { x } _ { l , h } = \sigma ( z _ { l , h } ) \cdot x _ { l , h } ,\tag{8}
$$

where $x _ { l , h } \in \mathbb { R } ^ { T \times d _ { h } }$ denotes the output of head h at layer $l ,$ and $T$ is the sequence length. A gate value close to 0 suppresses the head, while a value close to 1 preserves its contribution.

The optimisation objective is defined as:

$$
\mathcal { L } = - \frac { 1 } { \vert S \vert } \sum _ { t \in S } \log p _ { \theta } ( x _ { t + 1 } \mid x _ { \le t } ) - \lambda \cdot \frac { 1 } { L H } \sum _ { l , h } \mathrm { c l a m p } ( z _ { l , h } , - 4 , 4 ) ,\tag{9}
$$

where the first term is the next-token negative log-likelihood and the second term is the regularisation objective. The logits are clamped to [−4, 4] before regularisation to stabilise optimisation and encourage near-binary gate values.

Two complementary optimisation stages are then performed. In the positive-L1 stage $( \lambda = + 0 . 1 )$ , the optimiser is encouraged to suppress unnecessary heads, so only heads whose removal significantly increases the NLL retain large gate values. These heads correspond to necessary heads. In the

negative-L1 stage $( \lambda = - 0 . 1 )$ , the optimiser instead encourages larger gate values, identifying heads that are sufficient to maintain low task loss.

At the end of optimisation, the final sigmoid gate values are recorded as:

$$
s _ { h } ^ { \mathrm { n e c } } ( c ) = \sigma ( z _ { l , h } ^ { + } ) ^ { ( \mathrm { f i n a l } ) } ,\tag{10}
$$

$$
s _ { h } ^ { \mathrm { s u f } } ( c ) = \sigma ( z _ { l , h } ^ { - } ) ^ { ( \mathrm { f i n a l } ) } ,\tag{11}
$$

which define the necessary and sufficient scores, respectively.

A binary threshold $\tau = 0 . 5$ is then applied to derive discrete causal roles:

$$
\mathrm { r o l e } _ { l , h } = \left\{ \begin{array} { l l } { \mathbf { f a c i l i t a t i n g } , } & { s _ { h } ^ { \mathrm { n e c } } ( c ) \geq \tau ~ \land ~ s _ { h } ^ { \mathrm { s u f } } ( c ) \geq \tau , } \\ { \mathrm { i n t e r f e r i n g } , } & { s _ { h } ^ { \mathrm { n e c } } ( c ) < \tau ~ \land ~ s _ { h } ^ { \mathrm { s u f } } ( c ) \geq \tau , } \\ { \mathrm { i r r e l e v a n t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{12}
$$

A head is considered facilitating for class c if both scores exceed a threshold $\tau ,$ indicating that it is both necessary and sufficient for producing the corresponding behaviour; otherwise, it is treated as non-facilitating, encompassing both interfering and irrelevant heads.

This procedure yields two sets per class, $\mathcal { H } _ { c } ^ { \mathrm { f a c } }$ and ${ \mathcal { H } } _ { c } ^ { \mathrm { n o n - f a c } }$ . Crucially, this class-conditioned decomposition associates specific structural causal head with distinct calibration behaviours and forms the basis for targeted intervention in subsequent steps.

## 4.4 Canonical-Union Mask Construction

We construct a targeted structural causal mask for calibration by combining two complementary types of heads. First, from the wrong\_high class, we extract the facilitating heads:

$$
\mathcal { H } ^ { \mathrm { e r r } } = \mathcal { H } _ { \mathrm { w r o n g \_ h i g h } } ^ { \mathrm { f a c } } ,\tag{13}
$$

which contribute to generating high-confidence incorrect predictions. These heads can be interpreted as forming the targeted structural causal mask that amplify erroneous outputs.

Second, from the correct\_high class, we extract the non-facilitating heads:

$$
\mathcal { H } ^ { \mathrm { n o i s e } } = \mathcal { H } _ { \mathrm { c o r r e c t \_ h i g h } } ^ { \mathrm { n o n - f a c } } ,\tag{14}
$$

which do not support correct predictions and may introduce variability in confidence estimation.

We then define the candidate mask set as the union $\mathcal { H } ^ { \mathrm { m a s k } } = \mathcal { H } ^ { \mathrm { e r r } } \cup \mathcal { H } ^ { \mathrm { n o i s e } }$ . To avoid excessive intervention, we rank heads by their attribution strength:

$$
r ( h ) = \operatorname* { m a x } \big ( | s _ { h } ^ { \mathrm { n e c } } | , | s _ { h } ^ { \mathrm { s u f } } | \big ) ,\tag{15}
$$

and select the top-k heads to form a compact mask ${ \mathcal { H } } ^ { \mathrm { t o p } - k }$ . This ranking focuses the intervention on the most influential components while preserving the overall functionality of the model.

## 4.5 Inference-Time Intervention

Given the identified set of influential heads ${ \mathcal { H } } ^ { \mathrm { t o p } - k }$ , we perform a structured intervention on the model’s internal computation at inference time. Rather than modifying model parameters or postprocessing outputs, our approach directly alters the forward pass by selectively suppressing specific attention heads.

Let $z _ { h } ( x )$ denote the output of attention head h for input x. We define a binary gating operator $g _ { h } \in \{ 0 , 1 \}$ for each head as:

$$
g _ { h } = \left\{ { \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } h \in { \mathcal { H } } ^ { \mathrm { t o p } - k } , } \\ { 1 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{16}
$$

and obtain the modified head output:

$$
\tilde { z } _ { h } ( x ) = g _ { h } \cdot z _ { h } ( x ) .\tag{17}
$$

This yields a perturbed model $f _ { \theta } ^ { \mathrm { m a s k e d } } ( x )$

This operation can be interpreted as a form of causal editing on the model’s computation graph, where selected edges (i.e., head outputs) are removed to disrupt specific functional circuits. In particular, masking heads in $\mathcal { H } ^ { \mathrm { e r r } }$ suppresses pathways responsible for overconfident errors, while masking heads in ${ \mathcal { H } } ^ { \mathrm { n o i s e } }$ reduces spurious variation in otherwise correct predictions.

The intervention is both local and minimal, modifying only a small subset of heads and thus preserving most of the model’s representational capacity. The calibration classes are used exclusively during mask construction. At inference time, the resulting global mask is applied uniformly to all test inputs without requiring ground-truth labels or sample-specific class identification. From a computational perspective, the overhead is negligible, as the intervention involves only element-wise gating during the forward pass. Since no parameters are updated, the method can be seamlessly applied to pretrained models in a plug-and-play manner.

## 5 Experiments

## 5.1 Implementation Details

In practice, we apply CoCHI on a small subset of the data (typically fewer than 50 samples), with balanced sampling across calibration classes to ensure stable attribution estimates. We use a fixed confidence threshold $\tau _ { c } = 0 . 5$ to define calibration classes, and a threshold τ to identify facilitating heads. The number of masked heads k is treated as a hyperparameter; we find that small values $( { \bf e . g . } , k \in [ 6 , 8 ] )$ provide a favourable trade-off between calibration improvement and accuracy preservation. The penalty coefficient was empirically tuned across 15 model–dataset runs, yielding a stable trade-off that limits degradation on correct\_high calibration $( \Delta \mathrm { E C E } < 0 . 0 1 5 )$ while still effectively suppressing overconfident errors on incorrect predictions. We evaluate our method on three representative open-source large language models: LLaMA-3.1-8B [7], Gemma-2-9B [48], and Mistral-7B-v0.1 [49]. We adopt LLaMA-3.1-8B as the primary backbone for ablation and hyperparameter studies. All experiments are conducted in a zero-shot setting, ensuring that improvements in calibration are solely attributable to the proposed inference-time intervention. Further experimental details are provided in Appendix A.

## 5.2 Evaluation Protocols

Datasets. We evaluate our method on five widely used multiple-choice question answering benchmarks spanning diverse domains, including general knowledge, science, and medicine, to assess both generalisation and domain-specific robustness of calibration. Measuring Massive Multitask Language Understanding (MMLU) [50] is a large-scale multi-domain benchmark covering 57 subjects such as STEM, humanities, and social sciences, designed to evaluate general knowledge and reasoning ability. MedMCQA [51] is a large medical QA dataset derived from real-world entrance examinations, focusing on domain-specific knowledge in healthcare. OpenBookQA [52] is a science QA benchmark requiring multi-hop reasoning over a small set of core science facts. MedQA-USMLE [53] is a chal lenging medical benchmark based on USMLE exams, evaluating clinical reasoning and professional knowledge. SciQ [54] is a science QA dataset constructed from educational materials, focusing on factual understanding and basic reasoning. All datasets are formatted as four-choice multiple-choice tasks, and we adopt a unified evaluation protocol.

Baselines. We compare our method against several representative calibration approaches, including both post-hoc and inference-time techniques. Temperature Scaling (TS) [16] is a standard post-hoc method that learns a single scalar temperature $T$ on a held-out validation set to rescale logits. Inference-Time Intervention (ITI) [36] modifies model activations during inference using a fixed intervention strength $\alpha ,$ enabling reliability improvements without retraining. Thermometer (Therm.) [30] is a learned calibration method that trains a lightweight MLP-based calibrator on top of model outputs to better align predicted confidence with empirical accuracy. Adaptive Temperature Scaling (ATS) [29] extends TS by dynamically adapting the temperature based on input features, allowing more flexible calibration across samples.

Metrics. We evaluate performance in terms of both accuracy and calibration. Prediction quality is measured by accuracy, defined as the proportion of correctly predicted answers. Calibration is assessed using Expected Calibration Error (ECE) [16], which measures the discrepancy between predicted confidence and empirical accuracy across confidence bins; Negative Log-Likelihood (NLL), which evaluates the quality of predicted probability distributions by penalising overconfident incorrect predictions; and the Brier Score (BS), which computes the mean squared error between predicted probabilities and ground-truth labels. Lower values of ECE, NLL, and Brier Score indicate better calibration. Together, these metrics provide a comprehensive assessment of both predictive correctness and the reliability of confidence estimates.

Table 1: Results on four benchmarks (MMLU, MedMCQA, OpenBookQA, and MedQA-USMLE). Best results are highlighted in bold.
<table><tr><td rowspan="2"></td><td colspan="4">MMLU</td><td colspan="4">MedMCQA</td></tr><tr><td>Accuracy ↑</td><td>ECE↓</td><td>NLL↓</td><td>BS↓</td><td>Accuracy ↑</td><td>ECE↓</td><td>NLL↓</td><td>BS↓</td></tr><tr><td colspan="8">Gemma-2-9B</td></tr><tr><td>TS ITI Therm. ATS CoCHI</td><td>0.6653±0.0050 0.6693±0.0060 0.6653±0.0050 0.6653±0.0050</td><td>0.0660±0.0173 0.0605±0.0062 0.1831±0.0567 0.3784±0.0052 0.0601±0.0011</td><td>0.5060±0.0335 0.5029±0.0233 1.8729±1.1455 0.9191±0.0066 0.4982±0.0141</td><td>0.1683±0.0125 0.1675±0.0105 0.2302±0.0429 0.3553±0.0030</td><td>0.5607±0.0186 0.5673±0.0200 0.5607±0.0186 0.5607±0.0186</td><td>0.0556±0.0261 0.0451±0.0140 0.2849±0.0106 0.3534±0.0783</td><td>0.6114±0.0151 0.6085±0.0154 2.9242±0.4188 4.1559±1.3606</td><td>0.2122±0.0066 0.2122±0.0070 0.3293±0.0165 0.3702±0.0396 0.2070±0.0027</td></tr><tr><td colspan="8">0.6730±0.0023 0.1654±0.0056 0.5873±0.0023 0.0380±0.0263 LLaMA-3.1-8B</td></tr><tr><td>0.6007±0.0061 0.5853±0.0685</td><td>0.6007±0.0061</td><td>0.0569±0.0040 0.2014±0.1191 0.1717±0.0778 0.3265±0.0257</td><td>0.5205±0.0138 0.7161±0.1590 1.9024±0.7247 1.7325±1.5251</td><td>0.1936±0.0065 0.5753±0.0297 0.2580±0.0643 0.3747±0.0893 0.2415±0.0354 0.5753±0.0297 0.3330±0.0157</td><td>0.5753±0.0301</td><td>0.0557±0.0124 0.3766±0.2287 0.2327±0.0356 0.3307±0.0727</td><td>0.6173±0.0142 1.2043±0.7682 1.8116±0.8973 2.5476±2.9442</td><td>0.2144±0.0052 0.3831±0.1870 0.2907±0.0208 0.3495±0.0501</td></tr><tr><td colspan="8">0.6007±0.0061 0.6120±0.0626 0.0490±0.0089 0.5168±0.0236 0.1743 ±0.0008</td></tr><tr><td>TS</td><td></td><td>0.0902±0.0182</td><td>0.5754±0.0130</td><td>Mistral-7B</td><td>0.5773±0.0081</td><td>0.0365±0.0090</td><td>0.6064±0.0032</td><td>0.2114±0.0012</td></tr><tr><td>0.5987±0.0121 0.3980±0.1152 0.5987±0.0121</td><td>0.3693±0.2666 0.2988±0.0449</td><td>1.2168±0.8397 1.4277±0.9584</td><td></td><td>0.1974±0.0037 0.3845±0.2056 0.3177±0.0274</td><td>0.4867±0.0081 0.3293±0.0939 0.4867±0.0081</td><td>0.0604±0.0308 0.3985±0.3280 0.2668±0.0776</td><td>0.6424±0.0339 1.4918±1.2039</td><td>0.2249±0.0133 0.4317±0.2469</td></tr><tr><td colspan="8">OpenBookQA</td></tr><tr><td>ATS CoCHI</td><td>0.5987±0.0121 0.6100±0.0040</td><td>0.3379±0.0290 0.0854±0.0015</td><td>2.5765±1.4883 0.5649±0.0008</td><td>0.3462±0.0184 0.1941±0.0005</td><td>0.4867±0.0081 0.4847±0.0023</td><td>0.3817±0.1376 0.0383±0.0035</td><td>2.6632±0.6919 3.6701±2.4950 0.6168±0.0012</td><td>0.3385±0.0372 0.4032±0.0945 0.2146±0.0006</td></tr><tr><td>一 Method Accuracy ↑</td><td colspan="4"></td><td colspan="4">MedQA-USMLE</td></tr><tr><td>0.6080±0.0087</td><td></td><td>ECE↓</td><td>NLL↓</td><td>BS↓ Gemma-2-9B</td><td>Accuracy ↑</td><td>ECE↓</td><td>NLL↓</td><td>BS↓</td></tr><tr><td colspan="8">0.0686±0.0119 0.5410±0.0076 0.1848±0.0054 0.7714±0.0124</td></tr><tr><td>ITI Therm. ATS CoCHI</td><td>0.6100±0.0111 0.6080±0.0087 0.6080±0.0087 0.6090±0.0766</td><td>0.0682±0.0064 0.2635±0.0266 0.3350±0.0165 0.0434±0.0116</td><td>0.5438±0.0070 2.4829±0.5178 1.9401±1.8405 0.5353±0.0268</td><td>0.1848±0.0043 0.2930±0.0223 0.3399±0.0090 0.1840±0.0092</td><td>0.7927±0.0151 0.7810±0.0114 0.7740±0.0164 0.7893±0.0223</td><td>0.1725±0.0143 0.1767±0.0298 0.1863±0.0131 0.1723±0.0070</td><td>0.5091±0.0163 1.7598±0.2096 2.2281±0.2476 0.5081±0.0046</td><td>0.1708±0.0072 0.1757±0.0205 0.1866±0.0141 0.1694±0.0019</td></tr><tr><td colspan="8">LLaMA-3.1-8B 0.0618±0.0106 0.5776±0.0125 0.1853±0.0066 0.7360±0.0087 0.1198±0.0200</td></tr><tr><td>ITI Therm. ATS CoCHI 0.6700±0.0040</td><td>0.5200±0.1618 0.6827±0.0170 0.6827±0.0170</td><td>0.2272±0.2848 0.1654±0.0405 0.3346±0.0658 0.0523±0.0193</td><td>0.9131±0.6744 1.5188±0.5910 2.6951±1.5194 0.5069±0.0090</td><td>0.2952±0.2009 0.2180±0.0302 0.3195±0.0403 0.1730±0.0035</td><td>0.5387±0.0908 0.7360±0.0087 0.7360±0.0087 0.7307±0.0186</td><td>0.1611±0.0761 0.1954±0.0372 0.2432±0.0119 0.0961±0.0169</td><td>0.6333±0.0739 1.9680±0.4174 3.1290±0.1205 0.5066±0.0121</td><td>0.2226±0.0332 0.2124±0.0213 0.2430±0.0082 0.1700±0.0048</td></tr><tr><td colspan="8">Mistral-7B</td></tr><tr><td>TS ITI</td><td>0.5100±0.0100 0.3567±0.0996</td><td>0.0673±0.0383 0.3583±0.2315</td><td>0.6016±0.0130 1.0117±0.3433</td><td>0.2076±0.0033 0.3684±0.1309</td><td>0.6707±0.0064 0.5193±0.0574</td><td>0.1489±0.0077 0.1522±0.0597</td><td>0.5720±0.0115 0.6011±0.0402</td><td>0.1987±0.0089 0.2089±0.0195</td></tr></table>

## 5.3 Main Results

The results in Table 1 show that CoCHI consistently improves calibration across all datasets and models while maintaining competitive accuracy. On datasets such as MMLU and MedMCQA, CoCHI achieves clear reductions in ECE together with improved NLL and Brier Score. Similar trends are observed on OpenBookQA and MedQA-USMLE, where CoCHI attains the lowest or near-lowest calibration errors without sacrificing predictive performance. In several cases, CoCHI also improves accuracy, particularly on LLaMA-3.1-8B and Mistral-7B.

In contrast, existing methods show limited or unstable behaviour across datasets. Post-hoc methods such as TS and ATS provide only marginal improvements and can severely degrade calibration in some settings, while Thermometer often increases NLL, indicating unreliable probability estimates. ITI also exhibits high variance across models and datasets. The advantage of CoCHI is especially evident on Mistral-7B, where it consistently achieves lower ECE together with strong NLL and Brier Score performance. Overall, these results suggest that intervening on targeted causal attention heads

provides more effective and stable calibration than conventional output-level adjustment methods.   
Additional results are provided in Appendix B.1.

## 5.4 Ablation Study

The ablation results in Table 2 highlight the importance of structured, causallyinformed masking. Naive strategies, such as masking only wrong-high facilitating heads or applying a broad canonical-union mask, severely degrade both accuracy and calibration (accuracy 0.2044/0.2189; ECE 0.4745/0.3850), suggesting that indiscriminate head removal disrupts essential model functionality. More targeted variants also fail to achieve a satisfactory trade-off:

Table 2: Ablation study of different masking strategies on MMLU.
<table><tr><td>Mask Rule</td><td>Masked Acc ↑</td><td>Masked ECE ↓</td></tr><tr><td>Wrong-high facilitating only</td><td>0.2044±0.0038</td><td>0.4745±0.1477</td></tr><tr><td>Correct-high non-facilitating only</td><td>0.2389±0.0398</td><td>0.1518±0.0474</td></tr><tr><td>Correct-high interfering only</td><td>0.4267±0.0462</td><td>0.0694±0.0488</td></tr><tr><td>Canonical union (broad)</td><td>0.2189±0.0271</td><td>0.3850±0.1816</td></tr><tr><td>Random size-matched</td><td>0.2111±0.0077</td><td>0.4330±0.2736</td></tr><tr><td>CoCHI</td><td>0.6120±0.0626</td><td>0.0490±0.0089</td></tr></table>

masking correct-high interfering heads lowers ECE to 0.0694 but substantially reduces accuracy to 0.4267, while random size-matched masking performs poorly (accuracy 0.2111, ECE 0.4330). In contrast, CoCHI achieves the best balance, with accuracy 0.6120 and ECE 0.0490. These results show that effective intervention depends not only on identifying problematic heads, but also on selecting a compact set of high-impact components through canonical-union design and top-k ranking.

## 5.5 Sensitivity Analysis

Table 3: Sensitivity analysis of the confidence threshold $\tau _ { c }$ on MMLU.

Table 3 presents the sensitivity analysis of the confidence threshold $\tau _ { c } .$ Overall, the proposed method remains relatively stable across different threshold settings, indicating that its performance is not highly sensitive to the choice of $\tau _ { c } .$ . Among all configurations, $\tau _ { c } = 0 . 5$ achieves the best overall trade-off, yielding the lowest ECE, while also obtaining competi-

<table><tr><td> $\tau _ { c }$ </td><td>Accuracy ↑</td><td>ECE↓</td><td>NLL↓</td><td>BS↓</td></tr><tr><td>0.2</td><td> $0 . 5 9 3 3 \pm 0 . 0 2 1 7$ </td><td> $0 . 0 5 5 5 \pm 0 . 0 0 9 6$ </td><td> $0 . 5 3 8 5 \pm 0 . 0 0 7 5$ </td><td> $0 . 1 8 0 8 \pm 0 . 0 0 3 2$ </td></tr><tr><td>0.3</td><td> $0 . 6 0 6 0 \pm 0 . 0 0 5 9$ </td><td> $0 . 0 6 4 3 \pm 0 . 0 1 0 2$ </td><td> $0 . 5 3 1 9 \pm 0 . 0 0 4 3$ </td><td> $0 . 1 7 8 1 \pm 0 . 0 0 1 7$ </td></tr><tr><td>0.4</td><td> $0 . 5 8 6 0 \pm 0 . 0 1 1 8$ </td><td> $0 . 0 6 6 5 \pm 0 . 0 1 0 8$ </td><td> $0 . 5 2 7 6 \pm 0 . 0 0 5 8$ </td><td> $0 . 1 7 6 1 \pm 0 . 0 0 2 5$ </td></tr><tr><td>0.5</td><td> $0 . 6 1 2 0 { \scriptstyle \pm 0 . 0 6 2 6 }$ </td><td> $0 . 0 4 9 0 { \scriptstyle \pm 0 . 0 0 8 9 }$ </td><td> $0 . 5 1 6 8 { \pm } 0 . 0 2 3 6$ </td><td> $0 . 1 7 4 3 \pm 0 . 0 0 0 8$ </td></tr><tr><td>0.7</td><td> $0 . 5 8 7 3 \pm 0 . 0 3 7 7$ </td><td> $0 . 0 7 0 5 \pm 0 . 0 0 9 9$ </td><td> $0 . 5 5 4 8 \pm 0 . 0 2 1 2$ </td><td> $0 . 1 8 7 3 \pm 0 . 0 0 8 3$ </td></tr><tr><td>0.9</td><td> $0 . 5 9 3 3 \pm 0 . 0 0 8 1$ </td><td> $0 . 0 6 9 6 \pm 0 . 0 2 7 7$ </td><td> $0 . 5 2 5 5 \pm 0 . 0 0 5 9$ </td><td> $0 . 1 7 4 8 \pm 0 . 0 0 2 2$ </td></tr></table>

tive Accuracy, NLL, and Brier Score values. In contrast, very high thresholds $( \mathrm { e . g . , } \tau _ { c } = 0 . 7$ and 0.9) lead to noticeable degradation in calibration quality, reflected by increased ECE and NLL. Lower thresholds such as 0.2 provide reasonable calibration performance but slightly reduce predictive accuracy. These results suggest that moderate confidence thresholds are more effective for balancing classification performance and uncertainty calibration.

## 5.6 Efficiency Analysis

Table 4 presents an efficiency comparison across calibration methods, highlighting a clear trade-off between calibration cost and inference overhead. CoCHI requires GPU computation during a one-time calibration phase (69.9s), but produces an extremely compact artifact (40 B) and introduces zero infer-

Table 4: Efficiency comparison of calibration methods using 40 calibration samples on MMLU.
<table><tr><td>Method</td><td>GPU</td><td>Cal. time (s) ↓</td><td>Artifact size ↓</td><td>Inf.</td><td>Prunable</td></tr><tr><td>TS</td><td>No</td><td>0.5</td><td>4 B</td><td>Negligible</td><td>No</td></tr><tr><td>ATS</td><td>No</td><td>25.8</td><td>16 B</td><td> $\mathrm { N e g l i g i b l e }$ </td><td>No</td></tr><tr><td>ITI</td><td>Yes</td><td>18.4</td><td>16 KB</td><td> $O ( k \cdot d )$ </td><td>No</td></tr><tr><td>Therm.</td><td>No</td><td>19.4</td><td>1.0 MB</td><td> $O ( { \dot { H } } \cdot { \dot { C } } )$ </td><td>No</td></tr><tr><td>CoCHI</td><td>Yes</td><td>69.9</td><td>40 B</td><td>None</td><td>Yes</td></tr></table>

ence overhead, as the intervention is implemented by directly zeroing selected attention heads. In contrast, methods such as ITI and Thermometer incur additional computation at every forward pass $( { \mathrm { e . g . , ~ } } O ( k \cdot d )$ and $O ( H \cdot C )$ , respectively), leading to ongoing runtime costs and significantly larger artifacts (up to 1.0 MB). Lightweight post-hoc approaches such as TS and ATS have minimal calibration cost (0.5s and 25.8s) and negligible inference overhead, but do not modify the model’s internal behaviour and therefore lack structural impact.

A key advantage of CoCHI is its prunability: the masked heads can be permanently removed, yielding a smaller model with no runtime wrapper. Overall, CoCHI shifts the computational burden from continuous inference to a one-time structural calibration step, offering a more deployment-efficient solution for improving model reliability. More results are available in Appendix B.2.

## 5.7 Case Study

Figure 2 presents a per-class calibration comparison between the unmasked baseline and our method across five datasets, focusing on correct predictions with low confidence (left) and wrong predictions with high confidence (right). While our method is primarily designed to address wrong\_high errors, we observe consistent improvements in both regimes. In the correct\_low subset, our method reduces ECE across all datasets (e.g., 0.10 on MMLU and 0.09 on MedMCQA), indicating improved confidence alignment for underconfident correct predictions. In derconfident correct predictions. In

![](images/710524b6d8c911c6a5cba5897075d709d9769f36a09c6e634bf8dc8a9250d7fd.jpg)

![](images/8203e816a2787021ea3c9a7eaac0181ccc1935ef55f6bae1392485da3bf07dac.jpg)  
Figure 2: ECE comparison between the unmasked model0.09 on MedMCQA), indicating imand CoCHI on correct\_low and wrong\_high subsets.proved confidence alignment for un-

the more challenging wrong\_high subset, our method also consistently lowers ECE, albeit with smaller margins. This suggests that suppressing causally relevant heads not only mitigates overconfident errors but also improves overall confidence stability.

## 6 Conclusion

In this work, we study overconfident errors in large language models and argue that miscalibration should be addressed not only at the output level, but also through the internal mechanisms that give rise to unreliable confidence estimates. We propose CoCHI, a class-oriented causal head intervention framework that decomposes predictions by correctness and confidence, identifies attention heads associated with distinct calibration behaviours, and performs targeted inference-time masking without updating model parameters. Across multiple LLMs and five multiple-choice QA benchmarks, CoCHI consistently improves calibration metrics while maintaining competitive predictive accuracy. The ablation and sensitivity analyses further show that reliable calibration depends on structured causal selection rather than indiscriminate head removal. Moreover, while CoCHI primarily targets wrong\_high errors, it also yields improvements in correct\_low cases. These findings support a mechanism-aware view of calibration, linking model reliability to identifiable internal structures rather than treating it solely as a post-hoc output adjustment problem.

Limitations. This work focuses on multiple-choice question answering, where confidence can be naturally defined over a fixed answer space. Extending CoCHI to open-ended generation, long-form reasoning, and multi-turn interaction requires further investigation, especially when correctness and confidence are harder to define. In addition, our current intervention operates at the attention-head level, which provides an efficient and interpretable unit for masking, but may not capture all sources of miscalibration arising from MLP layers, residual streams, or cross-layer interactions.

## Acknowledgements

We acknowledge the RMIT Advanced Compute Environment (RACE) for providing the highperformance computing resources and infrastructure that supported this research. We also thank Patrick Taylor for his valuable technical support and assistance.

## References

[1] Yasin Abbasi-Yadkori, Ilja Kuzborskij, András György, and Csaba Szepesvári. To believe or not to believe your LLM: iterative prompting for estimating epistemic uncertainty. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 6aebba00fff5b6de7b488e496f80edd7-Abstract-Conference.html.

[2] Mark Steyvers, Heliodoro Tejeda Lemus, Aakriti Kumar, Catarina G. Belém, Sheer Karny, Xinyue Hu, Lukas William Mayer, and Padhraic Smyth. What large language models know and what people think they know. Nature Machine Intelligence, 7(2):221–231, 2025. doi: 10.1038/S42256-024-00976-7. URL https://doi.org/10.1038/s42256-024-00976-7.

[3] Jing Ren, Jiapeng Du, Bowen Li, Ziqi Xu, Xin Zheng, Hong Jia, Suyu Ma, Xiwei Xu, and Feng Xia. Spiking graph predictive coding for reliable OOD generalization. In Proceedings ofthe ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduled for April 13-17, 2026, rescheduled for June 29 - July 3, 2026, pages 9056–9067. ACM, 2026. doi: 10.1145/3774904.3793044. URL https://doi.org/10.1145/3774904.3793044.

[4] Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, and Xiuzhen Zhang. Hallucinate less by thinking more: Aspect-based causal abstention for large language models. In Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, pages 32555–32563. AAAI Press, 2026. doi: 10.1609/AAAI.V40I38.40532. URL https://doi.org/10.1609/aaai.v40i38.40532.

[5] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303. 08774. URL https://doi.org/10.48550/arXiv.2303.08774.

[6] DeepSeek-AI. Deepseek-v3 technical report. CoRR, abs/2412.19437, 2024. doi: 10.48550/ ARXIV.2412.19437. URL https://doi.org/10.48550/arXiv.2412.19437.

[7] Llama Team. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV. 2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783.

[8] Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, and Pascale Fung. Hallulens: LLM hallucination benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 24128–24156. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.ACL-LONG.1176. URL https://doi.org/10.18653/v1/2025.acl-long.1176.

[9] Abhilasha Ravichander, Shrusti Ghela, David Wadden, and Yejin Choi. Halogen: Fantastic LLM hallucinations and where to find them. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 1402–1425. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.ACL-LONG.71. URL https://doi.org/10.18653/v1/ 2025.acl-long.71.

[10] Shengqiong Wu, Hao Fei, Liangming Pan, William Yang Wang, Shuicheng Yan, and Tat-Seng Chua. Combating multimodal LLM hallucination via bottom-up holistic reasoning. In Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications ofArtificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2025, Philadelphia, PA, USA, February 25 - March 4, 2025, pages 8460–8468. AAAI Press, 2025. doi: 10.1609/AAAI.V39I8.32913. URL https://doi. org/10.1609/aaai.v39i8.32913.

[11] Akira Okutomi. Stable but miscalibrated: A kantian view on overconfidence from filters to large language models. CoRR, abs/2510.14925, 2025. doi: 10.48550/ARXIV.2510.14925. URL https://doi.org/10.48550/arXiv.2510.14925.

[12] Yaqi Yang, Ziqi Xu, Jie Li, Chenglong Ma, Jeffrey Chan, Mark Sanderson, Xin Zheng, and Yongli Ren. Mitigating bias in large language model based question answering through causal front door prompting. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2026, MelbourneVICAustralia, July 20-24, 2026, pages 2208–2218. ACM, 2026. doi: 10.1145/3805712.3809679. URL https://doi.org/10.1145/3805712.3809679.

[13] Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Joon Oh. Calibrating large language models using their generations only. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 15440–15459. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.824. URL https://doi.org/10.18653/v1/ 2024.acl-long.824.

[14] Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Cats: Calibrated test-time scaling for efficient LLM reasoning. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/hash/9f9ecbf4062842df17ec3f4ea3ad7f54-Abstract-Conference.html.

[15] Jing Ren, Bowen Li, Ziqi Xu, Xikun Zhang, Haytham M. Fayek, and Xiaodong Li. When to trust: A causality-aware calibration framework for accurate knowledge graph retrievalaugmented generation. In Proceedings ofthe ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduledfor April 13-17, 2026, rescheduledfor June 29 - July 3, 2026, pages 3859–3869. ACM, 2026. doi: 10.1145/3774904.3792358. URL https: //doi.org/10.1145/3774904.3792358.

[16] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. URL http://proceedings.mlr.press/ v70/guo17a.html.

[17] Andrew Nam, Henry Conklin, Yukang Yang, Tom Griffiths, Jonathan D. Cohen, and Sarah-Jane Leslie. Causal head gating: A framework for interpreting roles of attention heads in transformers. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ a7f530e11fa19e9551b7a51dbd0f336f-Abstract-Conference.html.

[18] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart M. Shieber. Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 92650b2e92217715fe312e6fa7b90d82-Abstract.html.

[19] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html.

[20] Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Kui Yu, Thuc Duy Le, and Jixue Liu. Linking model intervention to causal interpretation in model explanation. Pattern Recognit., 173:112814, 2026. doi: 10.1016/J.PATCOG.2025.112814. URL https://doi.org/10.1016/j.patcog.2025. 112814.

[21] Ziqi Xu, Debo Cheng, Jiuyong Li, Jixue Liu, Lin Liu, and Ke Wang. Disentangled representation for causal mediation analysis. In Thirty-Seventh AAAI Conference on Artificial Intelligence,

AAAI 2023, Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 10666–10674. AAAI Press, 2023. doi: 10.1609/AAAI.V37I9.26266. URL https://doi.org/10.1609/aaai.v37i9.26266.

[22] David John Cameron Mackay. Bayesian methodsfor adaptive models. California Institute of Technology, 1992.

[23] Alex Graves. Practical variational inference for neural networks. In John Shawe-Taylor, Richard S. Zemel, Peter L. Bartlett, Fernando C. N. Pereira, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 24: 25th Annual Conference on Neural Information Processing Systems 2011. Proceedings ofa meeting held 12-14 December 2011, Granada, Spain, pages 2348–2356, 2011. URL https://proceedings.neurips.cc/ paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html.

[24] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In Proceedings ofthe 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1613–1622. JMLR.org, 2015. URL http://proceedings.mlr.press/ v37/blundell15.html.

[25] José Miguel Hernández-Lobato and Ryan P. Adams. Probabilistic backpropagation for scalable learning of bayesian neural networks. In Proceedings ofthe 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1861–1869. JMLR.org, 2015. URL http://proceedings. mlr.press/v37/hernandez-lobatoc15.html.

[26] Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning, ICML 2011, Bellevue, Washington, USA, June 28 - July 2, 2011, pages 681–688. Omnipress, 2011. URL https: //icml.cc/2011/papers/398\_icmlpaper.pdf.

[27] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 6402–6413, 2017. URL https://proceedings.neurips. cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html.

[28] Yuxi Xia, Pedro Henrique Luz de Araujo, Klim Zaporojets, and Benjamin Roth. Influences on LLM calibration: A study of response agreement, loss functions, and prompt styles. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 3740–3761. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.ACL-LONG.188. URL https://doi.org/10.18653/v1/2025.acl-long.188.

[29] Johnathan Xie, Annie S. Chen, Yoonho Lee, Eric Mitchell, and Chelsea Finn. Calibrating language models with adaptive temperature scaling. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 18128–18138. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.EMNLP-MAIN.1007. URL https://doi.org/10.18653/v1/2024. emnlp-main.1007.

[30] Maohao Shen, Subhro Das, Kristjan H. Greenewald, Prasanna Sattigeri, Gregory W. Wornell, and Soumya Ghosh. Thermometer: Towards universal calibration for large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pages 44687–44711. PMLR / OpenReview.net, 2024. URL https://proceedings.mlr.press/v235/shen24c. html.

[31] Xin Liu, Muhammad Khalifa, and Lu Wang. Litcab: Lightweight language model calibration over short- and long-form responses. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=jH67LHVOIO.

[32] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 5433–5442. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.330. URL https://doi.org/10.18653/v1/ 2023.emnlp-main.330.

[33] Min Zhang, Jianfeng He, Taoran Ji, and Chang-Tien Lu. Don’t go to extremes: Revealing the excessive sensitivity and calibration limitations of llms in implicit hate speech detection. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12073– 12086. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG. 652. URL https://doi.org/10.18653/v1/2024.acl-long.652.

[34] Kaitlyn Zhou, Dan Jurafsky, and Tatsunori Hashimoto. Navigating the grey area: How expressions of uncertainty and overconfidence affect language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 5506–5524. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.335. URL https://doi.org/10.18653/v1/2023. emnlp-main.335.

[35] Sabrina J. Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. Reducing conversational agents’ overconfidence through linguistic calibration. Transactions of the Association for Computational Linguistics, 10:857–872, 2022. doi: 10.1162/TACL\_A\_00494. URL https: //doi.org/10.1162/tacl\_a\_00494.

[36] Kenneth Li, Oam Patel, Fernanda B. Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html.

[37] Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Mrinmaya Sachan, Preslav Nakov, and Artem Shelmanov. Uncertainty-aware attention heads: Efficient unsupervised uncertainty quantification for llms. CoRR, abs/2505.20045, 2025. doi: 10.48550/ARXIV.2505.20045. URL https://doi.org/10.48550/arXiv.2505.20045.

[38] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum? id=NpsVSN6o4ul.

[39] Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html.

[40] Aliyah R. Hsu, Georgia Zhou, Yeshwanth Cherapanamjeri, Yaxuan Huang, Anobel Y. Odisho, Peter R. Carroll, and Bin Yu. Efficient automated circuit discovery in transformers using contextual decomposition. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=41HlN8XYM5.

[41] Judea Pearl. Causality. Cambridge university press, 2009.

[42] Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 9574–9586, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 4f5c422f4d49a5a807eda27434231040-Abstract.html.

[43] Jean Kaddour, Aengus Lynch, Qi Liu, Matt J. Kusner, and Ricardo Silva. Causal machine learning: A survey and open problems. Foundations and Trends in Optimization, 9(1-2):1–247, 2025. doi: 10.1561/2400000052. URL https://doi.org/10.1561/2400000052.

[44] Zhengxuan Wu, Atticus Geiger, Joshua Rozner, Elisa Kreiss, Hanson Lu, Thomas Icard, Christopher Potts, and Noah D. Goodman. Causal distillation for language models. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 4288–4295. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022. NAACL-MAIN.318. URL https://doi.org/10.18653/v1/2022.naacl-main.318.

[45] Xinshu Li, Shiyi Yang, Ziqi Xu, Feng Xia, Quan Z. Sheng, and Lina Yao. Causal abstraction learning for multi-modal grounded planning. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD 2026, Jeju Island, Korea, August 9-13, 2026, pages 2790–2801. ACM, 2026. doi: 10.1145/3770855.3817995. URL https://doi.org/10.1145/3770855.3817995.

[46] Tianyi Zhao, Yinhan He, Wendy Zheng, Yujie Zhang, and Chen Chen. Wired for overconfidence: A mechanistic perspective on inflated verbalized confidence in llms. CoRR, abs/2604.01457, 2026. doi: 10.48550/ARXIV.2604.01457. URL https://doi.org/10.48550/arXiv.2604. 01457.

[47] Abhinav Joshi, Areeb Ahmad, and Ashutosh Modi. Calibration across layers: Understanding calibration evolution in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 14686–14714. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025. EMNLP-MAIN.742. URL https://doi.org/10.18653/v1/2025.emnlp-main.742.

[48] Gemma Team. Gemma 2: Improving open language models at a practical size. CoRR, abs/2408.00118, 2024. doi: 10.48550/ARXIV.2408.00118. URL https://doi.org/10. 48550/arXiv.2408.00118.

[49] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023. doi: 10.48550/ARXIV.2310.06825. URL https://doi.org/10.48550/arXiv.2310. 06825.

[50] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

[51] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on Health, Inference, and Learning, CHIL 2022, 7-8 April 2022, Virtual Event, volume 174 of Proceedings ofMachine Learning Research, pages 248–260. PMLR, 2022. URL https: //proceedings.mlr.press/v174/pal22a.html.

[52] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL https://doi.org/10.18653/v1/d18-1260.

[53] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14), 2021. ISSN 2076-3417. doi: 10.3390/app11146421. URL https://www.mdpi.com/2076-3417/11/14/6421.

[54] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings ofthe 3rd Workshop on Noisy User-generated Text, NUT@EMNLP 2017, Copenhagen, Denmark, September 7, 2017, pages 94–106. Association for Computational Linguistics, 2017. doi: 10.18653/V1/W17-4413. URL https://doi.org/10.18653/v1/ w17-4413.

## A Dataset Preparation and Experimental Pipeline

This section details the data preparation and experimental pipeline to ensure reproducibility.

## A.1 Datasets

We use five publicly available multiple-choice question answering datasets (as shown in Table 5) from HuggingFace, covering general knowledge, science, and medical domains. All datasets are formatted as four-option multiple-choice questions with a single correct answer.

Table 5: Statistical summary of the datasets.
<table><tr><td>Dataset</td><td>Domain</td><td># Questions</td></tr><tr><td>MMLU</td><td>Multi-domain knowledge</td><td>14,042</td></tr><tr><td>MedQA-USMLE</td><td>Medical exam</td><td>11,451</td></tr><tr><td>OpenBookQA</td><td>Science</td><td>~4,200</td></tr><tr><td>SciQ</td><td>Science</td><td>13,679</td></tr><tr><td>MedMCQA</td><td>Medical exam</td><td>187,005</td></tr></table>

## A.2 Data Cleaning and Standardisation

All datasets are converted into a unified format containing: question ID, dataset source, question text, four answer choices (A–D), and the correct label. We remove samples that: (i) do not contain exactly four options, or (ii) have missing or invalid answer labels.

SciQ Adjustment. In SciQ, the correct answer appears in a fixed position. To avoid positional bias, we cyclically rotate answer options to ensure a uniform distribution.

## A.3 Prompt Formatting

All inputs follow a unified zero-shot prompt:

Answer the following multiple choice question with a single letter: A, B, C, or D.

Question: [question text]

A. [choice 1]

B. [choice 2]

C. [choice 3]

D. [choice 4]

Answer:

No demonstrations or additional context are provided.

## A.4 Training Example Selection

From the fit pool, we sample a balanced calibration subset:

• up to 10 examples per class,

• maximum of 40 samples per dataset.

This simulates a low-resource calibration setting.

## A.5 Top-k Head Selection

We determine the number of masked heads k via grid search over $k \in \{ 1 , \ldots , 2 0 \}$ . Selection is based on minimising Expected Calibration Error (ECE), subject to an accuracy drop constraint of at most 1% relative to the unmodified model. A single k is selected per model and applied across all datasets to capture model-level calibration properties. The selected values are $k = \bar { 1 } 0$ for LLaMA-3.1-8B, and $\bar { k } = 8$ for Gemma-2-9B and Mistral-7B.

Table 6: Results on SciQ. Best results are highlighted in bold.
<table><tr><td>Method</td><td>Acc↑</td><td>ECE↓</td><td>NLL↓</td><td>BS↓</td></tr><tr><td colspan="5">Gemma-2-9B</td></tr><tr><td>TS ITI Therm. ATS</td><td>0.9173±0.0012</td><td>0.9173±0.0012 0.0579±0.0184 0.4699±0.4226 0.0641±0.0094 0.9253±0.0083 0.1027±0.0091 0.2337±0.0076 0.0617±0.0025 0.9223±0.0021 0.0609±0.0066 1.0836±0.1824 0.0599±0.0068  $0 . 0 6 0 7 { \scriptstyle \pm 0 . 0 0 6 7 }$  0.0563±0.0199</td><td>1.1578±0.1183 0.0602±0.0065</td><td>0.2309±0.12760.0548±0.0472</td></tr><tr><td colspan="5">CoCHI 0.9353±0.1069 LLaMA-3.1-8B</td></tr><tr><td>TS ITI</td><td>0.8790±0.0210 0.0354±0.0109 0.2377±0.0145 0.0721±0.0045 0.8647±0.0311 0.0359±0.01760.3032±0.0329 0.0921±0.0139</td><td>0.8920±0.0131 0.0738±0.0112 0.9514±0.2711 0.0807±0.0065</td><td></td><td></td></tr><tr><td>Therm. ATS CoCHI</td><td>0.8790±0.0210 0.0932±0.0219 1.4970±0.2712 0.0928±0.0195</td><td></td><td></td><td></td></tr><tr><td></td><td colspan="4">0.9020±0.0020 0.0330±0.0010 0.2266±0.0012 0.0676±0.0004</td></tr><tr><td></td><td>Mistral-7B</td><td></td><td></td><td></td></tr><tr><td>TS</td><td>0.8667±0.01500.0764±0.00480.3057±0.04020.0944±0.0045</td><td></td><td></td><td></td></tr><tr><td>ITI</td><td>0.5720±0.1512 0.2095±0.1429 0.6363±0.1672 0.2249±0.0680</td><td></td><td></td><td></td></tr><tr><td>Therm.</td><td></td><td>0.8467±0.0310 0.0986±0.0084 1.4162±0.0701 0.0978±0.0073</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ATS</td><td></td><td>0.8667±0.0150 0.1056±0.0139 1.6920±0.2413 0.1049±0.0134</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoCHI</td><td></td><td>0.8773±0.0046 0.0730±0.0116 0.3030±0.0144 0.0903±0.0046</td><td></td><td></td></tr></table>

## B Additional Results

## B.1 Main results

Table 6 confirms the effectiveness of our method on SciQ, demonstrating consistent improvements in calibration across all models while maintaining strong predictive performance. On Gemma-2-9B, our method achieves the best overall calibration, reducing ECE and improving both NLL and Brier Score , while matching the highest accuracy. On LLaMA-3.1-8B, the gains are even more pronounced: our method simultaneously achieves the best performance across all four metrics, improving accuracy, lowering ECE, and reducing both NLL and Brier Score. Similar trends are observed on Mistral-7B, where our method achieves the best results across all metrics, outperforming the strongest baseline (ATS) in both calibration and accuracy. These results indicate that the proposed intervention consistently improves confidence alignment without sacrificing predictive quality.

In contrast, baseline methods exhibit unstable and inconsistent behaviour. ITI achieves competitive accuracy on Gemma-2-9B but significantly worsens calibration, while on Mistral-7B it severely degrades both accuracy and calibration. Post-hoc methods such as TS and ATS provide moderate or inconsistent calibration improvements but often fail to jointly optimise NLL and Brier Score, with ATS in particular showing substantial degradation in probabilistic metrics (e.g., NLL exceeding 1.0 across models). Thermometer similarly leads to large increases in NLL, indicating unreliable probability estimates. Overall, these results reinforce our central claim: calibration is better improved by targeting internal causal mechanisms rather than relying on output-level adjustments, leading to more stable and reliable performance across diverse models.

Table 7: Calibration time (s; mean ± std) of CoCHI across three model families, averaged over 15 runs per model. Costs include class-conditioned forward passes and exclude model loading, and scale with model size while remaining practical.
<table><tr><td>Model</td><td>Cal. time (s) ↓</td><td>Min</td><td>Max</td><td>Peak GPU (GiB)</td></tr><tr><td>Mistral-7B-v0.1</td><td>39±21</td><td>21</td><td>80</td><td>28.1</td></tr><tr><td>Llama-3.1-8B</td><td>58±20</td><td>41</td><td>93</td><td>30.4</td></tr><tr><td>Gemma-2-9B</td><td>88±23</td><td>72</td><td>159</td><td>46.2</td></tr></table>

## B.2 Efficiency Analysis

Table 7 shows that the calibration cost of our method scales predictably with model size while remaining practical. The average calibration time increases from 39 s for Mistral-7B to 58 s for LLaMA-3.1-8B and 88 s for Gemma-2-9B, reflecting the expected growth in forward-pass complexity, yet all models complete calibration within roughly 1–1.5 minutes. The variance is moderate (std ≈ 20–23 s), indicating stable computational behaviour. Peak GPU memory ranges from 28.1 GiB to 46.2 GiB, increasing with model size but remaining within the capacity of standard high-end GPUs. Notably, these costs include multiple class-conditioned forward passes, which are essential to our method, yet remain comparable to inference-time intervention baselines. Overall, the results indicate that the calibration procedure introduces only moderate overhead, scales with model size, and is practical for real-world deployment.