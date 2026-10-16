# TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models

Ya-Shi Zhang<sup>1,2,†</sup>, Jian Tang<sup>2,3</sup>, Pierre-André Noël<sup>1</sup>, Oleksiy Ostapenko<sup>1</sup>

<sup>1</sup>ServiceNow AI Research, <sup>2</sup>Mila - Quebec AI Institute, <sup>3</sup>HEC Montréal <sup>†</sup>Work done at ServiceNow

Masked difusion language models expose an inference-time choice: which positions become visible after each denoising pass. Many existing decoding methods use hand-designed rules or require visibility to grow monotonically. We present a Tiny Inference-time Policy for Scaling (TIPS), an approximately 600K-parameter controller that learns both unmasking and remasking while keeping the underlying language model frozen. TIPS uses visibility-aware confidence features and a shallow transformer to select which tokens should be visible, while a fixed schedule controls the expected visible fraction. We train the controller through supervised warmup followed by reinforcement learning from terminal correctness rewards. On GSM8K and MATH-500 with LLaDA-8B-Instruct, the full-canvas and two-block TIPS variants achieve 58.46% and 57.97% micro-averaged accuracy at a budget of 32 decoder evaluations, exceeding the strongest baseline in our main comparison by 12.1 and 11.6 percentage points, respectively. Both variants are trained at a budget of 32 evaluations and remain competitive at higher budgets without retraining. These results demonstrate that learning when to reveal and reconsider tokens can improve the accuracy–compute trade-of without updating the decoder backbone.

Correspondence: ya-shi.zhang@mila.quebec

servicenow

## 1 Introduction

Masked difusion language models (dLLMs) generate responses by predicting unresolved positions in parallel and exposing selected predictions as context (Sahoo et al., 2024; Nie et al., 2025). This flexibility creates a decoding problem: under a limited number of denoising passes, a sampler must decide which predictions to commit and whether earlier commitments should be reconsidered. Revealing more positions makes faster progress toward a complete response, but those predictions also condition subsequent generations. An early commitment can therefore influence later predictions even after the surrounding context changes. The resulting visibility trajectory—which tokens become context, when they appear, and whether they can be revised—is an important component of the accuracy–compute trade-of (Kim et al., 2025a). We study how to learn this trajectory while leaving the pretrained token model unchanged.

Existing methods address diferent parts of this problem. Confidence-based and uncertainty-aware samplers select positions using hand-designed rules (Wu et al., 2026; Ben-Hamu et al., 2025; Huang et al., 2026a). Learned unmasking policies instead optimize position selection around a frozen denoiser, but typically preserve monotonic visibility: once a token is exposed, the policy cannot withdraw it (Jazbec et al., 2026; Xu et al., 2026; Hong et al., 2026a). Conversely, remasking methods allow revision through prescribed transitions or heuristic criteria (Wang et al., 2025; Hong et al., 2026b; Zhai et al., 2026), while other approaches learn remasking behavior by updating the language model itself (Huang et al., 2026b). These developments motivate a complementary question: can a small auxiliary policy learn both commitment and revision without modifying the dLLM?

We introduce Tiny Inference-time Policy for Scaling (TIPS), an approximately 600K-parameter controller assigning to each response position a binary next-step visibility action. At masked positions, the action reveals the decoder’s proposal or leaves the position masked; at visible positions, it retains the current token or remasks it for subsequent regeneration. A two-layer transformer predicts visibility preferences from confidence features, mask indicators, and normalized decoding time.

TIPS separates decoding order from pace. A fixed power-law schedule specifies the expected visible fraction, while a shared shift of the policy logits matches this target without changing the policy’s relative preferences. Thus, the expected amount of context grows according to the schedule even as individual commitments remain reversible. Our training recipe first provides dense, reference-derived supervision of visibility decisions, then optimizes final-answer correctness through group-relative policy optimization (Shao et al., 2024). Only the auxiliary controller is updated in either stage; training requires neither gradients through the dLLM nor its sequence likelihood. The same visibility-control formulation applies to full-canvas and semi-autoregressive decoding (section 3).

## We summarize our contributions as follows:

• We introduce TIPS, an approximately 600K-parameter policy that learns both unmasking and remasking while keeping the dLLM backbone frozen.

• We instantiate TIPS with full-canvas and blockwise decoding on LLaDA-8B-Instruct. At most decoder NFE budgets, we surpass previous state-of-the-art methods in accuracy on GSM8K and MATH-500 by up to 13.25 percentage points, otherwise remaining competitive.

• We investigate how TIPS uses commitment and revision through ablations and other supplementary experiments. These analyses highlight the importance of remasking, suggesting its benefits are not only token replacement, but also temporarily withdrawing harmful context. We further characterize the scope and limits of TIPS through transfer experiments.

## 2 Background

We focus on finite-step decoding with a pretrained masked difusion language model (dLLM). For fundamental principles and backbone training objectives, we refer readers to Austin et al. (2021); Sahoo et al. (2024); Sh et al. (2024); Nie et al. (2025); Ou et al. (2025); Arriola et al. (2025).

## 2.1 Finite-step decoding

Let $x _ { t } \in \mathcal V ^ { L }$ denote the partially generated response after t denoising passes, let

$$
\mathcal { M } _ { t } = \{ i : x _ { t , i } = [ \mathrm { M A S K } ] \} , \qquad S _ { t } = \{ i : x _ { t , i } \neq [ \mathrm { M A S K } ] \}\tag{1}
$$

be the masked and visible positions, respectively. A dLLM predicts a vocabulary distribution at every position and the reverse process samples a candidate token at each masked position:

$$
p _ { t , i } ( \boldsymbol { v } ) = p _ { \theta } ( \boldsymbol { v } \mid \boldsymbol { q } , \boldsymbol { x } _ { t } ) , \qquad \widetilde { \boldsymbol { x } } _ { t , i } \sim p _ { t , i } ( \cdot ) , \quad i \in \mathcal { M } _ { t } .\tag{2}
$$

Note that token sampling is often controlled ad-hoc via temperature scaling, where temperature 0 denotes top-1 or greedy decoding. A decoding algorithm then chooses the set $S _ { t + 1 } \subseteq \{ 1 , \dots , L \}$ of positions that will be visible in the next state and applies

$$
x _ { t + 1 , i } = \left\{ \begin{array} { l l } { [ \mathrm { M A S K } ] , } & { i \notin { S _ { t + 1 } } , } \\ { x _ { t , i } , } & { i \in \mathcal { S } _ { t } \cap \mathcal { S } _ { t + 1 } , } \\ { \widetilde { x } _ { t , i } , } & { i \in \mathcal { M } _ { t } \cap \mathcal { S } _ { t + 1 } . } \end{array} \right.\tag{3}
$$

A visible token may be kept or re-masked, while a masked position may be revealed or left masked. Many decoding algorithms impose the monotonic constraint $S _ { t } \subseteq S _ { t + 1 }$ . Under this restriction, tokens that are already unmasked can no longer be modified. Re-masking and editing methods remove this monotonicity constraint, allowing an earlier prediction to return to the masked state or directly transition to another vocabulary token after the surrounding context changes.

Hence, masked-difusion inference separates two choices: what token to place, and which positions should be visible. The first choice is supplied by the dLLM backbone. The second is supplied by the decoding algorithm. Because visible tokens become context at subsequent passes, diferent visibility trajectories can produce diferent final responses for the same model and number of forward passes.

```latex
Algorithm 1 TIPS: Inference
Require: Prompt q, response length L, decoder budget $T \geq 1$
Require: Frozen decoder $p _ { \theta } , \mathrm { T I P S } \ \pi _ { \phi } ,$ fixed schedule parameter $\gamma \in \mathbb { R }$
Ensure: Fully visible response $x _ { T }$ using exactly T dLLM evaluations
1: $x _ { 0 } \gets ( [ \mathrm { M A S K } ] , \dots , [ \mathrm { M A S K } ] )$
2: for $t = 0 , \ldots , T - 2$ do ▷ No early stopping; empty when $T = 1$
3: $p _ { t } \gets p _ { \theta } ( \cdot \mid q , x _ { t } )$ ▷ One dLLM evaluation
4: $m _ { t , i } \gets \mathbf { 1 } \{ x _ { t , i } = [ \mathrm { M A S K } ] \}$ for all i
5: Compute $C _ { t }$ using equation (6)
6: $b _ { t } \gets \pi _ { \phi } ( C _ { t } , m _ { t } , t / T )$
7: $\kappa _ { t } \gets \big ( ( t + 1 ) / T \big )$ exp(γ)
8: Compute $\begin{array} { r } { \delta _ { t } \ \mathrm { s . t . } \ \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sigma ( b _ { t , i } + \delta _ { t } ) = \kappa _ { t } } \end{array}$
9: $\rho _ { t , i } \gets \sigma ( b _ { t , i } + \delta _ { t } ) $ for all i
10: Sample $a _ { t , i } \sim$ Bernoull $. ( \rho _ { t , i } )$ for all i
11: Update all positions simultaneously: [MASK], $a _ { t , i } = 0 ,$
x<sub>t+1,i</sub> = x<sub>t,i</sub>, $a _ { t , i } = 1 , ~ m _ { t , i } = 0 ,$
(xt,i, $a _ { t , i } = 1 , m _ { t , i } = 1 .$
12: end for
13: $p _ { T - 1 }  p _ { \theta } ( \cdot \mid q , x _ { T - 1 } )$ ▷ Final dLLM evaluation
14: $\widehat { x } _ { T - 1 , i } \gets \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { T - 1 , i } ( v )$ for all i x<sub>T−1,i</sub>, x<sub>T−1,i</sub> ̸= [MASK],
$x _ { T , i } = \left\{ \begin{array} { l l } { \ l _ { - 1 , \imath } , } & { \ l ^ { \mathrm { \ l } _ { - 1 } } - \mathrm { \ l } _ { 1 } , \ i } \\ { \ l _ { \widehat { x } _ { T - 1 , i } , } } & { \ l { x _ { T - 1 , i } } = \mathrm { [ M A S K ] } . } \end{array} \right.$
16: return $x _ { T }$
```

## 2.2 The decoding path as a Markov decision process

A decoding path can be viewed as a finite-horizon Markov decision process (Jazbec et al., 2026). At decision step t, the state contains the prompt, current partially generated response, and denoising time. One dLLM evaluation produces the candidate distributions in equation (2), and an action is represented by the next visible set $\boldsymbol { S } _ { t + 1 }$ or – equivalently – by a binary next-step visibility decision per position, denoted $a _ { t } \in \{ 0 , 1 \} ^ { L }$ , where 1 denotes unmasked. The transition then applies equation (3). For non-zero temperatures, the transition is stochastic when conditioning on the action; with greedy decoding it is deterministic given the state and action. We assume greedy decoding from here on out.

## 3 Method

TIPS is a lightweight auxiliary policy network $\pi _ { \phi }$ that controls the inference trajectory of a frozen dLLM. Following the Bernoulli action parameterization of Jazbec et al. (2026), the policy maps decoding-state features to position-wise binary decisions. TIPS extends unmasking-only decisions to next-step visibility: an action can reveal or defer a masked position, or retain or remask an already visible token. TIPS is also fixed-horizon: it does not terminate early when all tokens are unmasked. Subsequent policy actions may remask tokens for revision. See algorithm 1 for details.

One dLLM forward pass produces probabilities over non-mask tokens $p _ { t , i } ( v ) : = p _ { \theta } ( v \mid q , x _ { t } )$ . Define mask indicator $m _ { t , i } = 1 \{ x _ { t , i } = [ \mathrm { M A S K } ] \}$ and greedy candidate $\widehat { x } _ { t , i } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { t , i } ( v )$

Featurization. For every position, define the active token:

$$
\begin{array} { r } { v _ { t , i } ^ { \mathrm { a c t } } = \bigg \{ \frac { \widehat { x } _ { t , i } , \quad m _ { t , i } = 1 , } { m _ { t , i } = 0 , } } \end{array}\tag{4}
$$

with corresponding probability $P _ { t , i } ^ { \mathrm { a c t } } = p _ { t , i } ( v _ { t , i } ^ { \mathrm { a c t } } )$ . At a masked position, the active token is the decoder’s greedy proposal. At an unmasked position, it is the already decoded token, even if the token is not the decoder’s

argmax. This is done to ensure visibility consistency: always measuring support for the token whose visibility the policy controls. For more details, refer to appendix B.2.3. We also define the strongest alternative

$$
v _ { t , i } ^ { \mathrm { a l t } } = \underset { v \in \mathcal { V } \backslash \{ v _ { t , i } ^ { \mathrm { a c t } } \} } { \arg \operatorname* { m a x } } p _ { t , i } ( v ) ,\tag{5}
$$

with probability $P _ { t , i } ^ { \mathrm { a l t } } = p _ { t , i } \big ( \boldsymbol { v } _ { t , i } ^ { \mathrm { a l t } } \big )$ , which is the most probable token other than the active token.

Now, we construct the two-dimensional per-token feature

$$
C _ { t , i } = \left[ \log \frac { P _ { t , i } ^ { \mathrm { a c t } } } { 1 - P _ { t , i } ^ { \mathrm { a c t } } } , \log \frac { P _ { t , i } ^ { \mathrm { a c t } } } { P _ { t , i } ^ { \mathrm { a l t } } } \right] .\tag{6}
$$

The first coordinate is the active token’s log-odds, whose parametrization can distinguish changes near the lowand high-probability extremes more clearly than raw probability. The second coordinate measures relative support of the active token against the most likely competing token. This log-ratio is negative if and only if the decoder prefers a diferent token over the current visible token. It is also important to mention the two features are on the same numerical scale.

A two-layer RoPE-DiT (Peebles & Xie, 2023; Su et al., 2024) policy network produces per-position visibility logits $b _ { t } = \pi _ { \phi } ( C _ { t } , m _ { t } , t / T )$ , representing the policy’s relative preference for each position’s visibility (see appendix C.1). These logits determine which positions are preferred while also allowing an external unmasking schedule to control how many positions become visible.

Scheduled visibility. We observe in appendix B.2.5 a network that is free to determine the total visible fraction leads to a lazy strategy we hereafter refer to as finalization deferral: keeping most positions masked until the final forward pass, where all are unmasked at once. This prevents the decoder from conditioning on progressively richer contexts, and leads to worse task performance. Therefore, we introduce a scalar shift to the output logits so that the adjusted logits adhere to the schedule in expectation.

For normalized decoding time $\tau \in [ 0 , 1 ]$ , define the family $\kappa ( \tau ) = \tau ^ { \mathrm { e x p } ( \gamma ) } , \gamma \in \mathbb { R }$ . For $\gamma > 0$ , the curve is convex: visibility grows slowly at first and more rapidly toward the end of decoding. For $\gamma < 0$ , it is concave, exposing more tokens early; $\gamma = 0$ recovers a linear schedule. In all cases, $\kappa ( 0 ) = 0$ and $\kappa ( 1 ) = 1$ , corresponding to completely masked and completely unmasked states at the beginning and end of decoding, respectively. Thus, for a budget of T dLLM evaluations and parameter $\gamma ,$ the target fraction of visible tokens for the next state is $\kappa _ { t } = \left( ( t + 1 ) / T \right) ^ { \mathrm { e x p } ( \gamma ) }$ , with $t = 0 , \dots , T - 2$ . To achieve this ratio in practice without changing local preference, we find a shared scalar $\delta _ { t }$ such that $\begin{array} { r } { \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sigma ( b _ { t , i } + \delta _ { t } ) = \kappa _ { t } } \end{array}$ , with calibrated next-state visibility probabilities $\rho _ { t , i } = \sigma ( b _ { t , i } + \delta _ { t } )$ . As the left hand side is continuous and strictly monotonic in $\delta _ { t } ,$ it has a unique solution, which we solve for numerically via Newton iteration. For more details, refer to equation (15). The same scalar is added to every position, preserving the policy’s preference ordering. The realized number of visible positions remains stochastic as TIPS samples $a _ { t , i } \sim \mathrm { B e r n o u l l i } ( \rho _ { t , i } )$ , where $a _ { t , i }$ has the state-independent interpretation of

$$
a _ { t , i } = 1 \iff \mathrm { p o s i t i o n } \ i \ \mathrm { i s \ v i s i b l e \ i n } \ x _ { t + 1 } ,\tag{7}
$$

while following the schedule in expectation, i.e., $\begin{array} { r } { \mathbb { E } \left[ \frac { 1 } { L } \sum _ { i = 1 } ^ { L } a _ { t , i } \Big | x _ { t } \right] = \kappa _ { t } } \end{array}$

Policy optimization. As in Jazbec et al. (2026), we use conditionally independent Bernoulli actions with a tractable factorized likelihood. For TIPS’s next-step visibility actions, the joint log-probability is

$$
\log \operatorname* { P r } _ { \phi } ( a _ { t } \mid s _ { t } ) = \sum _ { i = 1 } ^ { L } \left[ a _ { t , i } \log \rho _ { t , i } + ( 1 - a _ { t , i } ) \log ( 1 - \rho _ { t , i } ) \right] ,\tag{8}
$$

where $s _ { t } = ( C _ { t } , m _ { t } , t / T )$ denotes the policy state. During training, we sample a group of G visibility trajectories with final responses to obtain a binary reward $R _ { g }$ from a verifier. We can then optimize the policy network

with GRPO (Schulman et al., 2017; Shao et al., 2024):

$$
\mathcal { L } _ { \mathrm { R L V R } } ( \phi ) = - \frac { 1 } { G ( T - 1 ) } \sum _ { g = 1 } ^ { G } \sum _ { t = 0 } ^ { T - 2 } \operatorname* { m i n } \Big ( w _ { t , g } ( \phi ) A _ { g } , \mathrm { c l i p } \left( w _ { t , g } ( \phi ) , 0 . 5 , 1 . 5 \right) A _ { g } \Big ) ,\tag{9}
$$

where $\begin{array} { r } { w _ { t , g } ( \phi ) = \frac { \mathrm { P r } _ { \phi } ( a _ { t } ^ { ( g ) } | s _ { t } ^ { ( g ) } ) } { \mathrm { P r } _ { \mathrm { o l d } } ( a _ { t } ^ { ( g ) } | s _ { t } ^ { ( g ) } ) } } \end{array}$ is the likelihood ratio and $\begin{array} { r } { A _ { g } = R _ { g } - \frac { 1 } { G } \sum _ { g = 1 } ^ { G } R _ { g } } \end{array}$ is the group-relative advantage.

Supervised warmup. If reference response data is available, then we can optionally warmup the policy network before policy optimization by providing dense supervision of network logits. We collect intermediate decoding states from rollouts, then construct a target using the reference tokens. Given a reference response y and a masked position, we define the probability of the reference token and the strongest non-reference token as $q _ { t , i } = p _ { t , i } ( y _ { i } )$ and $\widehat { q } _ { t , i } = \operatorname* { m a x } _ { v \in \mathcal { V } \backslash \{ y _ { i } \} } p _ { t , i } ( v )$ , respectively. Now, we assign a visibility utility to each position:

$$
u _ { t , i } = \left\{ \begin{array} { l l } { + 1 , } & { m _ { t , i } = 0 , x _ { t , i } = y _ { i } , } \\ { - 1 , } & { m _ { t , i } = 0 , x _ { t , i } \neq y _ { i } , } \\ { \frac { q _ { t , i } - \widehat { q } _ { t , i } } { q _ { t , i } + \widehat { q } _ { t , i } } , } & { m _ { t , i } = 1 . } \end{array} \right.\tag{10}
$$

At visible positions, this utility favors retaining tokens that match the reference response and remasking those that are mismatching. At masked positions, it measures the decoder’s support for the reference token relative to its strongest competitor. The masked utility can also be written as tanh $\textstyle \left( { \frac { 1 } { 2 } } \log q _ { t , i } / { \widehat { q } } _ { t , i } \right) \in [ - 1 , 1 ]$ . The utilities are converted into visibility targets $\rho _ { t , i } ^ { \star } = \mathrm { s g } [ \sigma ( \beta u _ { t , i } + \delta _ { t } ^ { \star } ) ]$ , where $\beta > 0$ controls sharpness and sg[·] is the stopgrad operator. We minimize the binary cross-entropy between the policy probabilities and the targets (details in appendix C.4)

$$
\mathcal { L } _ { \mathrm { w a r m u p } } ( \phi ) = - \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \left[ \rho _ { t , i } ^ { \star } \log \rho _ { t , i } + ( 1 - \rho _ { t , i } ^ { \star } ) \log ( 1 - \rho _ { t , i } ) \right] .\tag{11}
$$

Semi-autoregressive decoding. TIPS generalizes to blockwise decoding trivially by treating each block as the full canvas (Arriola et al., 2025). For a total budget of T dLLM evaluations split across B blocks, we allocate $T / B$ calls to each block. For each block, TIPS is active for the first $T / B - 1$ calls before the deterministic final forward pass that unmasks all remaining mask tokens. We train and evaluate two variants of TIPS, full-canvas and $B = 2 .$ , referred to as $\mathrm { T I P S _ { f u l l } }$ and $\mathrm { T I P S _ { b l o c k } }$

## 4 Experiments

## 4.1 Setup

Backbone and tasks. All experiments use the frozen LLaDA-8B-Instruct decoder with a 256-token response length (Nie et al., 2025) unless stated otherwise. We train the models on the GSM8K and MATH training sets and evaluate on the GSM8K test set and MATH-500, containing 1,319 and 500 problems, respectively (Cobbe et al., 2021; Hendrycks et al., 2021; Lightman et al., 2024). For all methods, we report the accuracy over the 1,819 problems (i.e., micro-average over both test sets).

Training. We train TIPS across three seeds with fixed decoding budget $T _ { \mathrm { t r a i n } } = 3 2$ , beginning with the supervised warmup followed by 935 RLVR updates. For the choice of hyper-parameter γ, we first roll out the policy on a validation set without supervised warmup or logit calibration and choose γ by least-squares projection of its mean expected visibility trajectory onto the family $\tau ^ { \mathrm { e x p } ( \gamma ) }$ (details in appendix A.1). We get $\gamma _ { \mathrm { f u l l } } \approx 0 . 1 4 5$ and $\gamma _ { \mathrm { b l o c k } } \approx 0 . 1 8 7$

Baselines. We reproduce and compare with a broad suite of baselines. To preserve visual fidelity, we show only a subset of methods, identified by the highest-performing or competitive baselines at each evaluated decoder-call budget to include them in figure 1 and table 1.

Table 1 Micro-averaged test accuracy (%) on GSM8K and MATH-500 (1,819 problems) at diferent decoder-call budgets. For compactness, we display a subset of baselines that preserves the highest or competitive accuracy at every evaluated budget across the full comparison suite. TIPS accuracy entries show the mean with ± standard error underneath, across three training-seed means; each seed mean averages three inference draws. AUC is the normalized area under the accuracy versus $\mathrm { l o g _ { 2 } \mathrm { - b u d g e t } }$ curve over budgets 4–256, computed using trapezoidal quadrature and reported on the same 0–100 scale as accuracy. TIPS AUC values are computed from the mean accuracy curves. Bold indicates the highest accuracy among all evaluated configurations in each budget column; in the AUC column, it indicates the highest value among the displayed methods. Blocks denotes number of blocks × tokens per block. The policy from Jazbec et al. (2026) corresponds to the expert-steering variant. Fast-dLLM’s unavailable 4-call accuracy is imputed as its 8-call accuracy.
<table><tr><td>Method</td><td>Blocks</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td><td>AUC</td></tr><tr><td>Fast-dLLM</td><td>8×32</td><td>17.48</td><td>17.48</td><td>14.73</td><td>37.93</td><td>65.59</td><td>67.01</td><td>67.01</td><td>40.83</td></tr><tr><td>WINO</td><td>1×256</td><td>18.14</td><td>21.06</td><td>34.80</td><td>46.34</td><td>49.26</td><td>49.37</td><td>49.37</td><td>39.10</td></tr><tr><td>UNCODE</td><td>1×256</td><td>12.26</td><td>14.35</td><td>21.50</td><td>29.74</td><td>53.66</td><td>63.66</td><td>67.73</td><td>37.15</td></tr><tr><td>Jazbec et al. (2026)</td><td>1×256</td><td>5.63</td><td>7.42</td><td>9.18</td><td>12.39</td><td>47.72</td><td>64.80</td><td>64.63</td><td>29.44</td></tr><tr><td> ${ \sf T I P S } _ { \mathrm { f u l l } }$ </td><td>1×256</td><td>16.46 ±1.79</td><td>29.12 ±1.68</td><td>48.05 ±0.73</td><td>58.46 ±0.10</td><td>63.09 ±0.50</td><td>65.32 ±0.55</td><td>66.28 ±0.71</td><td>50.90</td></tr><tr><td> ${ \mathsf { T } } 1 { \mathsf { P } } { \mathsf { S } } _ { \mathrm { b l o c k } }$ </td><td>2×128</td><td>12.53 ±0.45</td><td>20.95 ±0.99</td><td>42.12 ±2.04</td><td>57.97 ±1.23</td><td>64.49 ±0.80</td><td>67.12 ±0.28</td><td>68.18 ±0.36</td><td>48.83</td></tr></table>

This includes WINO (Hong et al., 2026b), Fast-dLLM’s confidence-based decoder (Wu et al., 2026), and UNCODE (Huang et al., 2026a). We additionally include the unmasking-only policy from Jazbec et al. (2026) as a learned-policy comparator. Results for all evaluated baseline configurations appear in appendix B.1, with implementation details in appendix C.6.

All methods use the exact same setting except for block decomposition, in which we followed the settings found in the method’s original manuscripts. For each method, we evaluate the performance at exponentially increasing budgets $T \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . Fixed-horizon methods will use exactly the requested number of decoder calls. For Fast-dLLM, T is a target mean NFE (refer to appendix A.2); other adaptive methods use decoder-call caps and may terminate earlier.

Metrics. We primarily measure the micro-averaged accuracy of each method reported across diferent decoder-call budgets. In table 1, we additionally report the standard error of our method’s accuracies across three seeds. Further details are given in appendix B.1.

Additionally, to summarize performance across decoding budgets using a single numerical value, we also estimate the area under the accuracy log-budget curve (AUC). Let $\operatorname { A c c } ( k )$ be the method’s accuracy at NFE budgets $k \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ , we use a piecewise-linear interpolation in $\log _ { 2 }$ NFE and trapezoidal quadrature, which simplifies to

$$
\mathrm { A U C } = \frac { 1 } { \log _ { 2 } 2 5 6 - \log _ { 2 } 4 } \int _ { \log _ { 2 } 4 } ^ { \log _ { 2 } 2 5 6 } \mathrm { A c c } ( 2 ^ { u } ) \mathrm { d } u = \frac { 1 } { 6 } \sum _ { k \in \{ 4 , 8 , \ldots , 1 2 8 \} } \frac { \mathrm { A c c } ( k ) + \mathrm { A c c } ( 2 k ) } { 2 } .
$$

If the method is adaptive and finishes decoding before using up the allocated NFE budget, its highest NFE accuracy is continued as a constant until $k = 2 5 6$ as the additional budget leaves the method’s output and hence performance unchanged.

## 4.2 Results

Results. figure 1 and table 1 presents our main results. TIPS substantially improves the accuracy–compute tradeof at low and moderate decoding budgets, with $\mathrm { T I P S } _ { \mathrm { f u l l } }$ achieving the highest accuracy among all methods at 8–32 call caps. Notably, $\mathrm { T I P S _ { f u l l } }$ at a budget of 16 calls already exceeds WINO’s accuracy at a budget cap of 32 calls. The two TIPS variants exhibit complementary strengths across decoding budgets. While $\mathrm { T I P S } _ { \mathrm { f u l l } }$ performs better at 4–32 call caps, $\mathrm { T I P S _ { b l o c k } }$ achieves higher accuracy at 64–256 call caps and is often the highest accuracy among the methods in this range. At 256 call cap, $\mathrm { T I P S _ { b l o c k } }$ reaches 68.18%, compared with 67.73% for UNCODE and 67.01% for Fast-dLLM: TIPS remains competitive at high budget.

![](images/faf5335545d090054845ad70d4ee98fce3fa89223ccb8295d4d32d6617e00e2c.jpg)  
Figure 1 Accuracy–compute tradeoff on mathematical reasoning. Micro-averaged test accuracy on GSM8K and MATH-500 (1,819 problems) versus total decoder-calls, shown on a logarithmic scale. We compare $\mathrm { T I P S } _ { \mathrm { f u l l } } ~ ( 1 \times 2 5 6 )$ and $\mathrm { T I P S } _ { \mathrm { b l o c k } } ~ ( 2 \times 1 2 8 )$ against Fast-dLLM $( 8 \times 3 2 )$ , WINO, UNCODE, and the learned unmasking policy of Jazbec et al. (2026) with expert-steering; the latter three use $1 \times 2 5 6$ full-canvas decoding. The displayed baselines include a highest or competitive accuracy baseline at every evaluated budget from the full comparison suite. TIPS curves show means with shaded bands indicating ±1 standard error across three training-seed means, each averaged over three inference draws. Baselines are shown as point estimates. Other adaptive baselines may use fewer calls than the specified budget.

By looking at the reported AUC values and inspecting figure 1 visually, we see that $\mathrm { T I P S } _ { \mathrm { f u l l } }$ and $\mathrm { T I P S _ { b l o c k } }$ achieves substantially higher AUC than the other displayed baselines, with Fast-dLLM being the next best at 40.83. Overall, this demonstrates that both methods are efective at many budget levels without the need to re-fit, and show the value of learning a lightweight controller for the decoding trajectory.

## 4.3 Supplementary Experiments

We investigate whether TIPS can transfer beyond its decoder backbone during training, diferent decoding geometries, and whether it can efectively continue decoding from states generated by another sampler. These experiments use one seed and, whenever applicable, are evaluated with decoder NFE budgets $T \in \{ 1 6 , 3 2 , 6 4 \}$ Unless explicitly mentioned, the mathematical reasoning results are micro-averaged over the test set of GSM8K and MATH-500 as before. Additional experiments and results appear in appendix B.

Zero-shot transfer to Dream-7B. We apply $\mathrm { T I P S } _ { \mathrm { f u l l } }$ policy network to Dream-7B-Instruct without retraining and compare the performance to Fast-dLLM on a single 256-token canvas (Ye et al., 2025). As seen in figure 2 and table 2, these gains show that non-trivial visibility decisions can transfer to a diferent decoder backbone without the need for re-training.

Zero-shot transfer across block sizes. We reuse $\mathrm { T I P S } _ { \mathrm { f u l l } }$ for $2 \times 1 2 8 , 4 \times 6 4$ , and $8 \times 3 2$ decoding, keeping the total response length and decoder budget fixed, in order to examine TIPS’ ability to generalize to diferent block sizes which, in turn, also shed light on its performance on diferent response lengths. Also seen in figure 2 and table $2 , \mathrm { T I P S _ { f u l l } }$ can preserve accuracy at slightly smaller block sizes before eventual degradation at $8 \times 3 2$

Handoff from a heuristic sampler. We evaluate whether TIPS can efectively continue decoding from intermediate states generated by another sampler. To do so, we run Fast-dLLM’s confidence-thresholding sampler with default threshold of 0.9 for a fraction $q \in [ 0 , 1 ]$ of the total forward calls then switch to TIPS. Across both TIPS variants and various budgets, 16 out of the 18 settings improve over the baseline confidence-thresholding sampler, demonstrating a consistent ability to recover from states generated from other decoding methods. For $T = 6 4$ specifically, as seen in table 3, all settings q are able to improve over continued heuristic decoding. However, available comparisons indicate that discarding the inherited canvas and restarting TIPS with only the remaining $( 1 - q ) T$ calls can yield higher accuracy. Thus, handof improves on continued heuristic decoding, but does not necessarily outperform restarting. See appendix B.3.4 for details.

Table 2 Zero-shot transfer. Test micro-accuracy (%) on GSM8K and MATH-500, one seed. Block-size transfer uses the same frozen 1×256-trained policy. T is the decoder-call budget.
<table><tr><td>Method / geometry  $T = 1 6 T = 3 2$   $T = 6 4$ </td></tr><tr><td>Dream-7B: zero-shot transfer,  $1 \times 2 5 6 .$ </td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$  35.18 46.56 52.39 Fast-dLLM 22.87 27.43 29.96</td></tr><tr><td>LLaDA: block-size transfer.</td></tr><tr><td></td></tr><tr><td> $1 \times 2 5 6$  (training geometry) 47.00 59.10 62.67</td></tr><tr><td> $2 \times 1 2 8$  36.78 55.25 62.89 23.80 46.95 59.81</td></tr><tr><td> $4 \times 6 4$  22.43 34.03 55.25</td></tr><tr><td> $8 \times 3 2$ </td></tr></table>

Table 3 Handoff from a heuristic sampler at $T =$ 64. Test micro-accuracy (%), one seed, switching from non-adjusted confidence thresholding to TIPS after budget fraction q. Both variants outperform continued heuristic decoding $( q = 1 )$ at every tested intermediate q.
<table><tr><td>Fraction q</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td> $\mathrm { T I P S _ { b l o c k } }$ </td></tr><tr><td>0 (TIPS)</td><td>63.39</td><td>64.54</td></tr><tr><td> $1 / 4$ </td><td>40.85</td><td>38.70</td></tr><tr><td> $1 / 2$ </td><td>31.34</td><td>38.15</td></tr><tr><td> $3 / 4$ </td><td>25.73</td><td>40.19</td></tr><tr><td>1 (Non-adj. Fast-dLLM)</td><td>23.91</td><td>34.91</td></tr></table>

![](images/bde9d7ab5c68e1de1d323ff05fe989f242b94d384007ad08848d290422f63b5d.jpg)  
(a) Zero-shot DREAM transfer

![](images/df770d2a9b7bf9a3804a826e7e972e44e28e98f19557a72feabceba805e56c13.jpg)  
(b) Block-size transfer  
Figure 2 Generalization of $\mathrm { T I P S } _ { \mathrm { f u l l } }$ . Left: the LLaDA-trained full-canvas policy transfers to Dream-7B. Right: $\mathrm { T I P S } _ { \mathrm { f u l l } }$ is applied to diferent block sizes on LLaDA, with decoding budget shared across blocks

We further apply $\mathrm { T I P S } _ { \mathrm { f u l l } }$ after non-adjusted Fast-dLLM has completed its response: four additional decoder evaluations raise test micro-accuracy from 67.01% to 67.29%, correcting five initially incorrect answers without making any initially correct answer incorrect in this run (Appendix B.3.5).

## 4.4 Ablations

We ablate TIPS on a held-out validation set of 512 GSM8K and 512 MATH training examples, reporting micro-averaged accuracy. Its diferent dataset composition makes these accuracies not directly comparable to test results. Ablations cover supervised warmup, remasking, visibility-aware input features, the choice of and sensitivity to γ, policy architecture and runtime, and warmup and RL decoding budgets. Full experimental details and accuracy–compute curves appear in Appendix B.2.

## 5 Related Work

Training-free decoding order and adaptive parallelism. Masked difusion allows many reveal orders, which can materially afect generation quality (Sahoo et al., 2024; Kim et al., 2025a). Samplers often prescribe how many positions to reveal at each pass, selecting their identities using decoder statistics. LLaDA uses confidence-ranked unmasking by default (Nie et al., 2025). More structured approaches spread reveals across dilated position groups or compare short candidate decoding paths before committing tokens (Luxembourg et al., 2026; Lee et al., 2026). Adaptive-parallelism methods include Fast-dLLM, which combines confidencethresholded decoding with approximate block-wise KV caching (Wu et al., 2026), and EB-Sampler, which dynamically reveals multiple positions subject to an entropy-based error tolerance (Ben-Hamu et al., 2025). Other methods refine the selection criterion: UNCODE/PC-Sampler calibrates uncertainty scores using position- and token-informativeness priors (Huang et al., 2026a); KLASS incorporates the temporal KL stability of token distributions (Kim et al., 2025b); and DAPD uses a self-attention-derived dependency graph to select weakly coupled positions for parallel decoding (Kim et al., 2026).

Learned unmasking schedules. A closely related line of work learns a controller around a pretrained denoiser. Jazbec et al. (2026) formulate masked-difusion sampling as a Markov decision process and use reinforcement learning to train a one-layer transformer that maps decoder confidences to monotonic unmasking decisions. Their transitions preserve already visible tokens, whereas TIPS extends the action semantics to reversible next-step visibility. Where-to-Unmask trains a supervised ranker to imitate a ground-truth probability-margin oracle (Asano et al., 2026). Adaptive Order Policies reweight masked-difusion loss terms by policy probabilities to favor positions at which the denoiser is likely to be correct, considering both frozen-denoiser and joint-training settings (Mohamud et al., 2026). Self-Aware Scheduling (SAS) derives a dense trajectory reward from a bound on sequential decoding mismatch and trains an order policy with group-relative policy optimization while keeping the denoiser fixed (Xu et al., 2026). UPO similarly learns a stochastic position-selection policy for a frozen masked difusion model, using a KL-regularized, GRPO-style objective around an explicit heuristic reference policy (Hong et al., 2026a). Its actions select only currently masked positions, preserving monotonicity.

Re-masking and revision. Re-masking methods relax monotonicity by allowing earlier token commitments to be withdrawn. P2 and ReMDM allow visible positions to return to the mask state and be regenerated, with ReMDM deriving its sampler from a discrete difusion process with a custom reverse transition (Peng et al., 2025; Wang et al., 2025). Running Confidence Remasking (RCR), introduced alongside MDPO, revisits positions with poor running confidence (He et al., 2025). WINO uses a training-free parallel draft-and-verify procedure to reveal candidates and re-mask suspicious visible tokens (Hong et al., 2026b), while CoRe identifies revision targets by testing prediction stability under masked-context perturbations (Zhai et al., 2026). These approaches prescribe revision rules rather than learning a separate visibility controller. RemeDi instead modifies the language model to jointly predict tokens and confidence scores, then trains its re-masking behavior through supervised and reinforcement-learning stages (Huang et al., 2026b). TIPS also uses a supervised warmup phase followed by reinforcement learning.

Reinforcement learning for diffusion language models. Policy optimization has also been applied directly to the dLLM. MDPO updates the masked token model along progressive denoising trajectories using intermediate rewards (He et al., 2025); mask-aware policy gradients account for both token and masking decisions (Raajesh et al., 2026); and SPG constructs upper- and lower-bound surrogates for the otherwise intractable sequence log-likelihood (Wang et al., 2026). TIPS instead applies policy gradients only to its auxiliary Bernoulli visibility policy, hence training requires neither sequence likelihood nor updates to decoder parameters.

## 6 Discussion

We introduced TIPS, a small policy that learns unmasking and remasking around a frozen dLLM, separating visibility control from token prediction while a fixed schedule regulates expected progress. On GSM8K and MATH-500, TIPS improves low- and moderate-budget accuracy and remains competitive at higher budgets without retraining; full-canvas and blockwise variants ofer complementary strengths. Ablations, corruption recovery, heuristic-prefix handofs, remasking and editing statistics, and transfer experiments highlight the value of warmup and remasking, suggest that revision may help by withdrawing harmful context rather than only replacing tokens, and reveal transferable but domain-dependent behavior.

Limitations. Our evaluation centers on 256-token mathematical reasoning, and the supplementary transfer experiments use a single seed; longer generations and broader task coverage remain important tests of generality. TIPS also uses a fixed decoding horizon rather than an adaptive stopping rule, preserving further revision but foregoing savings from early termination. Developing stopping criteria that account for possible future revisions, and evaluating the resulting accuracy–latency trade-of, are important directions for future work. Decoder-call budgets also do not directly capture wall-clock cost.

## References

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block Difusion: Interpolating between autoregressive and difusion language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= tyEyYT267x.

Hikaru Asano, Tadashi Kozuno, Kuniaki Saito, and Yukino Baba. Where-to-unmask: Ground-truth-guided unmasking order learning for masked difusion language models, 2026. URL https://arxiv.org/abs/2602.09501.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising difusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pp. 17981–17993, 2021.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas S. Nolte, and Brian Karrer. Accelerated sampling from masked difusion models via entropy bounded unmasking. In Advances in Neural Information Processing Systems, volume 38, pp. 62365–62391, 2025. doi: 10.52202/085713-1874. URL https://doi.org/10.52202/085713-1874.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Haoyu He, Katrin Renz, Yong Cao, and Andreas Geiger. MDPO: Overcoming the training-inference divide of masked difusion language models, 2025. URL https://arxiv.org/abs/2508.13148.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/ forum?id=7Bywt2mQsCe.

Chunsan Hong, Seonho An, Min-Soo Kim, and Jong Chul Ye. Improving discrete difusion unmasking policies beyond explicit reference policies. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=on6cb46OhD.

Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, and Jiangchao Yao. Wide-in, narrow-out: Revokable decoding for eficient and efective DLLMs. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=XtLQHlNLxy.

Pengcheng Huang, Tianming Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Tong Xiao, Zulong Chen, and Maosong Sun. Empirical analysis of decoding biases in masked difusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6853–6876, San Diego, California, United States, July 2026a. Association for Computational Linguistics. doi: 10.18653/v1/2026.acl-long.311. URL https://aclanthology.org/2026.acl-long.311/.

Zemin Huang, Yuhang Wang, Zhiyang Chen, and Guo-Jun Qi. Don’t settle too early: Self-reflective remasking for difusion language models. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=BsZeTuB5fD.

Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, Michael Kirchhof, João Monteiro, Victor Turrisi, Jason Ramapuram, and Marco Cuturi. Learning unmasking policies for difusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=3aXoaet48C.

Bumjun Kim, Dongjae Jeon, Moongyu Jeon, and Albert No. DAPD: Dependency-aware parallel decoding via attention for difusion LLMs. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=LZro8iqjiA.

Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham M. Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked difusions. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 30749–30768. PMLR, 2025a. URL https://proceedings.mlr.press/v267/kim25ah.html.

Seo Hyun Kim, Sunwoo Hong, Hojung Jung, Youngrok Park, and Se-Young Yun. KLASS: KL-guided fast inference in masked difusion models. In Advances in Neural Information Processing Systems, volume 38, pp. 102436–102470, 2025b. doi: 10.52202/085713-3087. URL https://doi.org/10.52202/085713-3087.

Sanghyun Lee, Seungryong Kim, Jongho Park, and Dongmin Park. Lookahead unmasking elicits reliable decoding in difusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=OdM5JRVEiK.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=v8L0pN6EOi.

Omer Luxembourg, Haim Permuter, and Eliya Nachmani. Plan for speed: Dilated scheduling for masked difusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview. net/forum?id=hcODkuS1cF.

Jama Hussein Mohamud, Mohsin Hasan, Mirco Ravanelli, and Yoshua Bengio. Adaptive order policies for masked difusion, 2026. URL https://arxiv.org/abs/2606.00295. Accepted at the ReALM-GEN Workshop at ICLR 2026.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. In Advances in Neural Information Processing Systems, volume 38, pp. 56354–56392, 2025. doi: 10.52202/085713-1689. URL https://doi.org/10.52202/085713-1689.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete difusion secretly models the conditional distributions of clean data. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=sMyXP8Tanm.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, October 2023. URL https://openaccess.thecvf.com/ content/ICCV2023/html/Peebles\_Scalable\_Difusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Avishek Joey Bose, Alexander Tong, and Pranam Chatterjee. Path planning for masked difusion model sampling, 2025. URL https://arxiv.org/abs/2502.03540.

Haran Raajesh, Kulin Shah, Adam Klivans, and Philipp Krähenbühl. Mask-aware policy gradients for difusion language models. In The Third Conference on Language Modeling, 2026. URL https://openreview.net/forum?id= mbjRRWwZDP.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander M. Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. In Advances in Neural Information Processing Systems, volume 37, pp. 130136–130184, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked difusion for discrete data. In Advances in Neural Information Processing Systems, volume 37, pp. 103131–103167, 2024.

Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063. URL https://doi.org/10.1016/j.neucom.2023.127063.

Chenyu Wang, Paria Rashidinejad, DiJia Su, Song Jiang, Sid Wang, Siyan Zhao, Cai Zhou, Shannon Zejiang Shen, Feiyu Chen, Tommi Jaakkola, Yuandong Tian, and Bo Liu. SPG: Sandwiched policy gradient for masked difusion language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=18j5Q49GwN.

Guanghan Wang, Yair Schif, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete difusion models with inference-time scaling. In Advances in Neural Information Processing Systems, volume 38, pp. 163488–163545, 2025. doi: 10.52202/085713-4927. URL https://doi.org/10.52202/085713-4927.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. FastdLLM: Training-free acceleration of difusion LLM by enabling KV cache and parallel decoding. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=3Z3Is6hnOT.

Jiawei Xu, Minghui Liu, Aakriti Agrawal, Yifan Chen, and Furong Huang. Scheduling thoughts: Learning the order of thought in difusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=j4UhHaxCpq.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Kevin Zhai, Sabbir Mollah, Zhenyi Wang, and Mubarak Shah. CORE: Context-robust remasking for difusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum? id=bmKHxLWkz9.

## A Expanded methodology

## A.1 Fitting the visibility schedule

We select $\gamma$ by fitting the visibility trajectory of a separately trained, uncalibrated pilot policy. Another feasible approach is to optimize γ via hyperparameter search on the validation set. We first train this pilot from scratch with RLVR and terminal correctness rewards, without supervised warmup or logit calibration. We then freeze the pilot and collect N trajectories on the held-out validation prompts, again without calibration. Let $T _ { \mathrm { { f i t } } }$ be the rollout budget, including the deterministic final decoder call, and let $b _ { t , i } ^ { ( j ) }$ be the pilot’s logit at position i and transition t in trajectory j. The empirical mean expected next-state visible fraction is

$$
\bar { v } _ { t + 1 } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left[ \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sigma \left( b _ { t , i } ^ { ( j ) } \right) \right] , \qquad t = 0 , \dots , T _ { \mathrm { f i t } } - 2 .\tag{12}
$$

Because each Bernoulli action specifies whether a position is visible in the next state, the inner average is that state’s conditional expected visible fraction. Using probabilities instead of sampled mask counts removes the additional noise from the next Bernoulli sample, while retaining the states visited by the uncalibrated policy.

We fit the endpoint-anchored family $\kappa _ { \gamma } ( \tau ) = \tau ^ { \exp ( \gamma ) }$ by solving the one-dimensional least-squares problem

$$
\widehat { \gamma } \in \underset { \gamma \in \mathbb { R } } { \arg \operatorname* { m i n } } \ \frac { 1 } { T _ { \mathrm { f i t } } - 1 } \sum _ { t = 0 } ^ { T _ { \mathrm { f i t } } - 2 } \left[ \bar { v } _ { t + 1 } - \left( \frac { t + 1 } { T _ { \mathrm { f i t } } } \right) ^ { \exp ( \gamma ) } \right] ^ { 2 } .\tag{13}
$$

This is a least-squares projection of the pilot’s empirical schedule onto a continuous, monotone family with $\kappa _ { \gamma } ( 0 ) = 0$ and $\kappa _ { \gamma } ( 1 ) = 1$ . The policy observes the current time $t / T _ { \mathrm { f i t } }$ , but the fit uses the next-state time $( t + 1 ) / T _ { \mathrm { f i t } }$ . We exclude deterministic finalization because its full visibility is imposed by the decoding algorithm. The fit uses neither reference answers nor correctness labels for the validation prompts. We then fix $\widehat { \gamma }$ during subsequent calibrated policy training and evaluate the same normalized schedule at other inference budgets. The pilot supplies the schedule shape; it is not the final calibrated policy.

For blockwise decoding, we apply the same procedure to the active block, using its length and local decoder-call budget in place of L and $T _ { \mathrm { { f i t } } }$ . We average the expected visible fractions over validation trajectories and blocks at matching local times, exclude each block’s deterministic finalization, and fit one shared $\gamma$ for that decoding geometry.

## A.2 Fast-dLLM budget matching

For a target budget T, we tune the confidence-decoding threshold using test-set rollouts until the average NFE lies in $[ T , T + 0 . 5 ]$ . The one-sided tolerance gives the baseline up to 0.5 additional decoder calls on average.

We use this mean-adjusted setting with $8 \times 3 2$ decoding for the main results and $1 \times 2 5 6$ decoding for Dream transfer. Results without this adjustment are labeled Non-adjusted Fast-dLLM or Non-adjusted confidence thresholding, as appropriate, and use decoder call caps. In the main comparison, threshold adjustment covers $T = 8 , 1 6 , 3 2 , 6 4 , 1 2 8 ;$ at $T = 2 5 6$ , we reuse the non-adjusted results, whose runs terminate well below their caps, and impute the unavailable 4-call result from the 8-call result. The 4-call result is unavailable because during our sweeps, there was no confidence-threshold hyperparameter found such that the expected NFE was within the range of [4, 4.5].

It is important to note that unless otherwise specified, Fast-dLLM refers to the mean-adjusted version, while the non-adjusted version with standard confidence threshold of 0.9 is referred to as the non-adjusted Fast-dLLM.

## B Additional Experiments

## B.1 Main results with all baselines

Table 4 expands the main comparison to all evaluated baseline configurations. Fast-dLLM uses the meanadjusted $8 \times 3 2$ setting described in section 4.1. We also retain Non-adjusted Fast-dLLM with the validationselected $1 \times 2 5 6$ configuration and the $8 \times 3 2$ shared-cap configuration; i.e. their thresholds are not adjusted to match mean NFE.

Across the 256-token configurations, $\mathrm { T I P S } _ { \mathrm { f u l l } }$ has the highest mean accuracy at budgets 8–32, while $\mathrm { T I P S _ { b l o c k } }$ leads at 128 and 256. Fast-dLLM leads at the target budget of 64, reaching 65.59% compared with 64.49% for $\mathrm { T I P S _ { b l o c k } }$ . The non-adjusted $8 \times 3 2$ configuration reaches 37.82% at its 64-call cap. The $4 \times 3 2$ UPO configuration generates only 128 response tokens and is shown separately from the 256-token comparisons.

## B.2 Ablations

Protocol. We evaluate the design choices of $\mathrm { T I P S _ { f u l l } }$ on the 1,024-example validation set comprising 512 GSM8K and 512 MATH problems, chosen then excluded from the training data. The same prompts were also used without correctness labels to fit the native schedule exponent. Unless stated otherwise, each comparison uses training seed 123, evaluation seed 42, a 256-token response length, and inference budgets $T \in \{ 1 6 , 3 2 , 6 4 \}$ Accuracy is the number of correct answers divided by 1,024. Because this equally weighted dataset mixture difers from the 1,319:500 test mixture, its absolute accuracies should not be compared directly with the main test results. The feature-width and finalization-deferral studies are historical exceptions, with their protocols specified separately below.

## B.2.1 Supervised warmup and reinforcement learning

We compare the complete supervised warmup–RLVR recipe with its supervised parent and a policy trained with RLVR alone. The RLVR-trained policies use the same 935-update continuation length, while the warmup-only policy receives no RLVR updates. As shown in Table 5 and Figure 3, both stages contribute to the final performance. Warmup followed by RLVR improves over warmup alone by 14.55, 17.87, and 14.45 percentage points at $T = 1 6 , 3 2 , 6 4$ , respectively. Warmup also provides a useful initialization for RLVR: although the gain over RLVR alone is only 0.49 points at $T = 1 6 .$ , it increases to 8.50 and 7.81 points at $T = 3 2$ and 64. Thus, the supervised policy is not simply suficient on its own; supervised initialization and terminal-reward optimization complement one another.

## B.2.2 The contribution of remasking

To separate dependence on an already learned behavior from the value of the action space itself, we evaluate two controls. The first disables remasking at inference for the frozen reference policy. The second retrains a monotone policy with supervised warmup and RLVR. In both cases, visible tokens retain their identities, and only masked positions can become newly visible. If V positions are already visible and $M = L - V$ remain masked, the reveal target is clipped to [0, M] from $L \kappa _ { t } - V$ , so the intervention does not demand an infeasible decrease in visibility.

Table 4 Complete accuracy–budget comparison on mathematical reasoning. Micro-averaged test accuracy (%) on GSM8K and MATH-500 (1,819 problems). All baseline configurations are retained, with blocks denoting number of blocks × tokens per block. TIPS entries give the mean and, underneath, its standard error across three training-seed means; each seed mean averages three inference draws. Baselines are point estimates. Bold indicates the highest reported mean in each budget column among the 256-token configurations. Dashes indicate unreported cells. The non-adjusted Fast-dLLM shared-cap adaptation is retained for reference, and the updated mean-budget adaptation is added as a separate row. The final row uses only 128 response tokens and is displayed separately, without entering the matched comparison.
<table><tr><td>Method / configuration</td><td>Blocks</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr><td colspan="9">Single-canvas configurations: 256 response tokens</td></tr><tr><td>Random / uniform</td><td>1×256</td><td>11.45</td><td>16.91</td><td>25.16</td><td>33.06</td><td>38.34</td><td>41.82</td><td>42.57</td></tr><tr><td>High confidence (fitted)</td><td>1×256</td><td>16.22</td><td>16.82</td><td>23.42</td><td>34.52</td><td>39.36</td><td>32.71</td><td>28.26</td></tr><tr><td>High confidence (linear)</td><td>1×256</td><td>5.06</td><td>12.26</td><td>30.84</td><td>37.82</td><td>33.32</td><td>27.32</td><td>29.19</td></tr><tr><td>Probability margin</td><td>1×256</td><td>9.84</td><td>17.59</td><td>31.61</td><td>39.53</td><td>35.51</td><td>27.54</td><td>30.40</td></tr><tr><td>Predictive entropy</td><td>1×256</td><td>4.45</td><td>7.81</td><td>20.51</td><td>32.49</td><td>25.73</td><td>19.79</td><td>28.59</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>1×256</td><td>17.70</td><td>22.37</td><td>31.67</td><td>33.32</td><td>33.48</td><td>33.48</td><td>33.48</td></tr><tr><td>EB-Sampler</td><td>1×256</td><td>12.04</td><td>16.88</td><td>19.85</td><td>21.72</td><td>25.40</td><td>27.21</td><td>31.39</td></tr><tr><td>UNCODE / PC-Sampler</td><td>1×256</td><td>12.26</td><td>14.35</td><td>21.50</td><td>29.74</td><td>53.66</td><td>63.66</td><td>67.73</td></tr><tr><td>KLASS</td><td>1×256</td><td>5.06</td><td>12.09</td><td>29.14</td><td>37.16</td><td>36.72</td><td>38.87</td><td>38.32</td></tr><tr><td>DAPD Jazbec et al. (2026)</td><td>1×256</td><td>8.36</td><td>14.79</td><td>21.37</td><td>28.95</td><td>50.47</td><td>50.82</td><td>50.82</td></tr><tr><td>(expert steering, retrained)</td><td>1×256</td><td>5.63</td><td>7.42</td><td>9.18</td><td>12.39</td><td>47.72</td><td>64.80</td><td>64.63</td></tr><tr><td>Jazbec et al. (2026) (vanilla, retrained)</td><td>1×256</td><td>5.81</td><td>9.05</td><td>16.40</td><td>38.34</td><td>48.30</td><td>47.32</td><td>47.90</td></tr><tr><td>UPO</td><td>1×256</td><td>5.94</td><td>7.15</td><td>8.03</td><td>10.23</td><td>12.37</td><td>17.59</td><td>49.70</td></tr><tr><td>SAS</td><td>1×256</td><td>9.84</td><td>20.95</td><td>33.53</td><td>41.56</td><td>43.65</td><td>43.65</td><td>43.76</td></tr><tr><td>ReMDM</td><td>1×256</td><td>0.55</td><td>4.93</td><td>21.73</td><td>32.40</td><td>41.14</td><td>40.52</td><td>33.53</td></tr><tr><td>RCR</td><td>1×256</td><td>8.34</td><td>19.72</td><td>33.33</td><td>38.67</td><td>35.79</td><td>33.88</td><td>36.93</td></tr><tr><td>WINO</td><td>1×256</td><td>18.14</td><td>21.06</td><td>34.80</td><td>46.34</td><td>49.26</td><td>49.37</td><td>49.37</td></tr><tr><td>CoRe</td><td>1×256</td><td>5.28</td><td>12.09</td><td>28.75</td><td>36.12</td><td>32.77</td><td>27.38</td><td>29.03</td></tr><tr><td colspan="9">Block-wise configurations: 256 response tokens</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>8×32</td><td>7.20</td><td>7.75</td><td>10.12</td><td>11.76</td><td>37.82</td><td>66.85</td><td>67.01</td></tr><tr><td>Fast-dLLM</td><td>8×32</td><td></td><td>17.48</td><td>14.73</td><td>37.93</td><td>65.59</td><td>67.01</td><td>67.01</td></tr><tr><td>KLASS (native block-wise)</td><td>4×64</td><td>14.07</td><td>21.99</td><td>28.70</td><td>40.68</td><td>59.98</td><td>66.03</td><td>66.90</td></tr><tr><td>UPO (block-trained; shared cap)</td><td>8×32</td><td>7.26</td><td>7.92</td><td>9.46</td><td>11.05</td><td>11.16</td><td>22.54</td><td>66.85</td></tr><tr><td>SAS (native block-wise)</td><td>8×32</td><td></td><td>17.48</td><td>23.75</td><td>33.26</td><td>53.22</td><td>61.46</td><td>66.14</td></tr><tr><td>WINO (rescored block-wise)</td><td>2×128</td><td>8.61</td><td>10.28</td><td>13.03</td><td>38.34</td><td>63.39</td><td>65.16</td><td>65.16</td></tr><tr><td>TIPSfull</td><td>1×256</td><td>16.46 ±1.79</td><td>29.12 ±1.68</td><td>48.05 ±0.73</td><td>58.46 ±0.10</td><td>63.09 ±0.50</td><td>65.32 ±0.55</td><td>66.28 ±0.71</td></tr><tr><td>TIPSblock</td><td>2×128</td><td>12.53 ±0.45</td><td>20.95 ±0.99</td><td>42.12 ±2.04</td><td>57.97 ±1.23</td><td>64.49 ±0.80</td><td>67.12 ±0.28</td><td>68.18 ±0.36</td></tr><tr><td colspan="9">Different response length: excluded from column-wise ranking UPO (128-token response) 4×32</td></tr></table>

As seen in figure 3, disabling remasking reduces accuracy by 7.03–12.99 percentage points across the three budgets. Retraining recovers part of this loss, but the monotone policy remains 0.78, 8.40, and 6.93 points below reversible TIPS at $T = 1 6 , 3 2 , 6 4$ , respectively. These results support retaining the ability to reconsider visible context.

Table 5 Training stages and reversible visibility. Validation-set micro-accuracy (%) over 1,024 problems, with one training seed and one evaluation seed. RLVR continuations use the same number of updates.
<table><tr><td>Configuration</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>Warmup → RLVR</td><td>43.16</td><td>56.15</td><td>58.20</td></tr><tr><td>Supervised warmup only</td><td>28.61</td><td>38.28</td><td>43.75</td></tr><tr><td>RLVR only</td><td>42.68</td><td>47.66</td><td>50.39</td></tr><tr><td>Remasking disabled (frozen)</td><td>36.13</td><td>44.53</td><td>45.21</td></tr><tr><td>Monotone policy (retrained)</td><td>42.38</td><td>47.75</td><td>51.27</td></tr></table>

![](images/3f3720b375329742f2df85206f3dd724798686f087638f5d144dc6d676b37cd2.jpg)  
(a) Training stages

![](images/c4b7b7b88d77105e954ee8cbe42bff0955cb897b3f3f64665fc0d8a2c5beb644.jpg)  
(b) Remasking controls  
Figure 3 Supervised initialization and reversible visibility. Validation-set accuracy versus decoder-call budget on a logarithmic axis. Each curve is one trained policy evaluated with one inference seed.

## B.2.3 Visibility-consistent confidence features

At an already visible position, the policy controls whether the current token remains in context, rather than whether the decoder’s newly preferred token should be inserted. A global-top-1 confidence can consequently describe the wrong token. When analyzing our trajectories, the global argmax difered from the visible token in 73,930 of 3,407,617 recorded visible-position events, or 2.17%. This observation motivates anchoring the first feature to the current visible token and excluding that token when constructing the strongest alternative. While not necessarily an ablation, we believe that ensuring the policy inputs remain consistent with the current decoding state is important for the semantic correctness of our method.

## B.2.4 Confidence-feature dimensionality study

We additionally report an earlier study of feature width $K \in \{ 1 , 2 , 1 0 , 5 0 \}$ . This experiment uses a one-layer full-canvas policy trained with RLVR at $T _ { \mathrm { t r a i n } } = 3 2$ . Inputs are raw full-vocabulary probabilities. The first coordinate measures support for the active token, using the greedy proposal at masked positions and the current token at visible positions; larger widths append the strongest alternative probabilities. The $K = 1$ and $K = 2$ variants therefore isolate whether to expose the strongest competitor.

Figure 4 reports all 14 recorded inference budgets. For each fixed trained policy, we average five seeds. Adding the strongest alternative improves the curve broadly: $K = 2$ exceeds $K = 1$ at every budget above one call, including gains of 3.56 points at $T = 1 6 , 2 . 1 4$ points at $T = 3 2$ , and 2.92 points at $T = 6 4$ . Expanding the input to ten or fifty probabilities does not improve over $K = 2 .$ . In this setting, information about the most competitive alternative is more useful than a much wider description of the probability distribution, supporting the choice of a two-channel feature.

Table 6 Feature-width study. Test micro-accuracy (%), reported as mean ± sample standard deviation across five evaluation seeds for each fixed policy.
<table><tr><td>Feature width K</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td><td> $T = 2 5 6$ </td></tr><tr><td>1</td><td> $4 0 . 1 2 \pm 0 . 9 3$ </td><td> $4 8 . 4 4 \pm 1 . 0 3$ </td><td> $5 1 . 3 8 \pm 1 . 0 0$ </td><td> $5 6 . 0 1 \pm 0 . 9 6$ </td></tr><tr><td>2</td><td> $4 3 . 6 8 \pm 1 . 2 7$ </td><td> $5 0 . 5 9 \pm 0 . 4 8$ </td><td> $5 4 . 3 0 \pm 0 . 6 4$ </td><td> $5 7 . 6 8 \pm 0 . 4 8$ </td></tr><tr><td>10</td><td> $3 8 . 9 0 \pm 0 . 3 5$ </td><td> $4 5 . 9 4 \pm 0 . 6 0$ </td><td> $4 9 . 9 7 \pm 0 . 5 3$ </td><td> $5 5 . 3 5 \pm 1 . 0 8$ </td></tr><tr><td>50</td><td> $4 0 . 1 3 \pm 0 . 4 2$ </td><td> $4 7 . 1 2 \pm 1 . 1 2$ </td><td> $4 9 . 9 7 \pm 0 . 7 6$ </td><td> $5 4 . 0 3 \pm 1 . 1 0$ </td></tr></table>

![](images/9a2db99f289fcf500289b713bebbc4a44c427f7b9be3644918d8fa59beb6c5c2.jpg)  
Figure 4 Confidence-feature width. Test micro-accuracy versus log decoder-call budget for $K \in \{ 1 , 2 , 1 0 , 5 0 \}$ rawprobability features. Markers include intermediate budgets as well as powers of two. Shading is one standard error across five evaluation seeds.

## B.2.5 Occupancy control and finalization deferral

We revisit an earlier RLVR-only comparison of single-layer policies with one 256-token response canvas, $T = 3 2$ decoder calls, and three training seeds. Policies are trained with no occupancy constraint, a soft schedule loss, or fixed power-law calibration, while retaining reversible actions and deterministic finalization.

The soft schedule loss imposes a penalty that is multiplied against the binary terminal correctness reward. Specifically, if, by the finalization pass, the state has masked fraction $m \in [ 0 , 1 ]$ , then the assigned terminal correctness reward is multiplied by (1 − m). For example, if the response is correct but only 50% of the canvas was unmasked by the finalization pass, the reward becomes 0.5 for that trajectory.

Figure 5 shows that the uncalibrated trajectory levels of before the final call. After the last learned transition, 57.11% of positions remain masked, versus 48.01% with the soft loss and 6.89% with fixed calibration (Table 7). The subsequent jump is imposed by deterministic completion, not learned visibility decisions: this is finalization deferral. We find the soft schedule loss, while non-constraining, insuficient for alleviating the finalization deferral phenomenon. Fixed calibration reduces this terminal workload and is accompanied by higher accuracy, from 45.90% to 50.83%.

![](images/d37d996422c7deac33dfcee9415935da5f29e27bbc822510f00e7cda73da0ee6.jpg)  
Figure 5 Finalization deferral in RLVR-only policies. Realized visible-token fraction at $T = 3 2 \colon$ : mean ± one standard deviation over training seeds 123, 124, and 125, with evaluation seed 42 and example-weighted pooling of GSM8K test and MATH-500. The shaded final interval and dashed terminal segments denote deterministic finalization. The origin is the prescribed all-mask state.

Table 7 Occupancy control reduces the finalization burden. At $L = 2 5 6$ and $T = 3 2 ,$ mask counts are means ± one standard deviation across three training seeds, measured at $\tau = 3 1 / 3 2$ with evaluation seed 42.
<table><tr><td>Occupancy control</td><td>Masks before finalization Masked (%) Accuracy (%)</td><td></td><td></td></tr><tr><td>None</td><td> $1 4 6 . 2 0 \pm 1 5 . 6 8$ </td><td>57.11</td><td>45.90</td></tr><tr><td>Soft schedule loss</td><td> $1 2 2 . 9 1 \pm 9 . 9 6$ </td><td>48.01</td><td>47.18</td></tr><tr><td>Fixed power-law calibration</td><td> $1 7 . 6 5 \pm 0 . 0 2$ </td><td>6.89</td><td>50.83</td></tr></table>

## B.2.6 Schedule shape and logit calibration

We freeze the reference policy and vary the inference schedule $\kappa ( \tau ) = \tau ^ { \mathrm { e x p } ( \gamma ) }$ . Table 8 and figure 6 distinguishes changes in $\gamma$ from removing calibration altogether. The fitted value is $\gamma = 0 . 1 4 5 4 4 \mathrm { : }$ ; the other evaluated settings are −0.2, 0, and 0.2. Note that $\gamma = 0$ is a calibrated linear schedule, not an uncalibrated policy.

The efect of schedule shape depends on the inference budget. Moving from the fitted value to $\gamma = 0 . 2$ raises accuracy at $T = 1 6$ from 43.16% to 45.70%, but reduces it at $T = 3 2$ from 56.15% to 53.22%; the $T = 6 4$ diference is only 0.29 points. More front-loaded visibility at $\gamma = - 0 . 2$ is worse at all three budgets, particularly at $T = 1 6$ . Therefore, nearby schedules can preserve much of the performance, but the schedule is not irrelevant and the best measured setting need not be the same at every budget.

Removing calibration. We next bypass the scalar occupancy-matching shift while keeping the checkpoint and deterministic finalization unchanged. Accuracy falls to 26.66%, 25.20%, and 25.59% at $T = 1 6 , 3 2 , 6 4$ respectively. In this intervention, additional decoder calls no longer yield the improvement observed with calibrated TIPS. The experiment establishes the importance of calibration for the learned policy’s use of intermediate states.

Table 8 Inference-time schedule interventions. Validation-set accuracy (%) for frozen $\mathrm { T I P S } _ { \mathrm { f u l l } }$ . All γ settings retain occupancy calibration; the final row bypasses the scalar logit shift entirely.
<table><tr><td>Frozen-policy setting</td><td> $T = 1 6$   $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>Fitted:  $\gamma = 0 . 1 4 5 4 4$ </td><td>43.16</td><td>56.15 58.20</td></tr><tr><td> $\gamma = - 0 . 2$ </td><td>30.86</td><td>47.95 54.49</td></tr><tr><td> $\gamma = 0 \ \mathrm { ( l i n e a r ) }$ </td><td>39.55</td><td>50.49 55.66</td></tr><tr><td> $\gamma = 0 . 2$ </td><td>45.70</td><td>53.22 57.91</td></tr><tr><td>Calibration disabled</td><td>26.66</td><td>25.20 25.59</td></tr></table>

![](images/77bb80839e5f2ef96b6f92b05c2dd7baf1dda692de6cfd277c0bb08f7d125d7c.jpg)  
(a) Schedule-shape sensitivity

![](images/0e75b8d8f3dd6b6d0844772bbb8994c42f0985dda48be28147f2f137a4867093.jpg)  
(b) Calibration intervention  
Figure 6 Schedule shape and occupancy calibration have distinct effects. Left: validation-set accuracy at the four measured γ values, with a separate curve for each budget. Right: the fitted schedule, calibrated linear schedule, and calibration-disabled policy versus log budget. All settings use the same frozen checkpoint.

## B.2.7 Policy architecture and runtime

We replace the two-layer DiT with either a one-layer DiT or a parameter-matched per-token multilayer perceptron, while retaining the training and evaluation protocol. The per-token network cannot exchange information between response positions within the controller. It nevertheless reaches 54.88% accuracy at $T = 6 4$ , compared with 58.20% for the two-layer DiT. Thus, a substantial fraction of the policy’s efectiveness can be retained with local processing of contextual decoder features. Cross-position processing still helps: the two-layer DiT exceeds the MLP at all three budgets and exceeds the one-layer DiT at $T = 3 2$ and 64.

Table 9 also reports the corresponding controller costs. At $T = 3 2 .$ , cumulative policy time per generated example is 8.06 ms for the two-layer DiT, 6.88 ms for the one-layer DiT, and 6.22 ms for the MLP. These measurements sum the controller’s work across the complete rollout.

Table 9 Architecture and controller cost. Validation-set accuracy $( \% )$ and total CUDA-measured policy time per generated example at $T = 3 2 .$ . Timings accumulate all 31 policy decisions; they were collected on an H100 80GB with float32 policy weights and batch size at most 16. The decoder and feature-extraction cost are excluded from the policy-time column.
<table><tr><td>Policy</td><td>Parameters</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td><td>Policy time (ms)</td></tr><tr><td>Two-layer DiT</td><td>628,739</td><td>43.16</td><td>56.15</td><td>58.20</td><td>8.06</td></tr><tr><td>One-layer DiT</td><td>331,395</td><td>45.12</td><td>53.03</td><td>55.86</td><td>6.88</td></tr><tr><td>Per-token MLP</td><td>634,835</td><td>38.18</td><td>48.24</td><td>54.88</td><td>6.22</td></tr></table>

## B.2.8 Training with a longer decoding budget

We compare $\mathrm { T I P S _ { f u l l } }$ trained with $T _ { \mathrm { t r a i n } } = 3 2$ and $T _ { \mathrm { t r a i n } } = 1 2 8$ on the same validation problems. Unfortunately, computational constraints prevented a fully matched experiment, and the $T _ { \mathrm { t r a i n } } = 1 2 8$ configuration uses a diferent γ of approximately 0.8.

The $T _ { \mathrm { t r a i n } } = 3 2$ policy is more accurate at every measured inference budget, with gains of 4.00, 6.84, and 4.98 percentage points at $T = 1 6 , 3 2 , 6 4$ , respectively. A possible explanation is that a longer training trajectory requires more visibility decisions to share the same terminal correctness signal, making credit assignment more dificult.

Table 10 Training-budget comparison for $\mathrm { T I P S } _ { \mathrm { f u l l } }$ . Validation-set accuracy (%) at the indicated inference budgets.
<table><tr><td>Training configuration</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td> $T _ { \mathrm { t r a i n } } = 3 2$ </td><td>43.16</td><td>56.15</td><td>58.20</td></tr><tr><td> $T _ { \mathrm { t r a i n } } = 1 2 8$ </td><td>39.16</td><td>49.32</td><td>53.22</td></tr></table>

![](images/1e818f3d269682022a80c6b2e6129c1f7512434ccf89466a972eebcd4569c3f9.jpg)  
(a) Policy architecture

![](images/7d4a149a0b3c8d2f7bb84cc7b9de841bc66299ac04208ae292e90b946ed53975.jpg)  
(b) Training decoding budget  
Figure 7 Controller architecture and training budget. Validation-set accuracy versus log decoder-call budget.

## B.2.9 Depth-matched comparison with the learned unmasking policy

To examine whether the comparison with Jazbec et al. (2026) is explained by controller depth, we evaluate one- and two-layer policies for both methods. We also vary the depth of $\mathrm { T I P S _ { b l o c k } }$ without changing its $2 \times 1 2 8$ decoding geometry. These depth-pair experiments use the same 1,024-example validation set and one training/evaluation seed pair. Within each method, the intended architectural change is the number of

DiT layers; the training recipe is otherwise retained. Across methods, matching depth almost matches the parameter count: TIPS has 331,395 parameters at one layer and 628,739 at two, compared with 331,265 and 628,609 for the unmasking-only policy.

TIPS retains its advantage at either matched depth (Table 11). At $T = 3 2$ , the one-layer full-canvas TIPS policy reaches 53.03%, compared with 43.85% for the one-layer comparison policy; at two layers, the respective accuracies are 56.15% and 40.33%. Moreover, doubling the comparison policy’s depth does not reproduce the improvement from TIPS. Its accuracy falls from 32.32% to 20.70% at $T = 1 6$ , and from 43.85% to 40.33% at $T = 3 2$ , while the two depths become similar at larger decoder budgets. These single-seed comparisons show that the reported advantage is not removed by matching controller depth, without attributing it to any one of TIPS’s other design choices.

For this depth comparison, we use the vanilla (non-expert-steered) variant of Jazbec et al. (2026) at both depths, motivated by its stronger low-budget performance in our reproduction. table 1 instead reports the expert-steered variant, which achieves higher accuracy at large budgets. The validation split additionally places greater weight on the harder MATH dataset (50%, compared with 27.48% in the test mixture): it involves high-school competition-level mathematics requiring complex, multi-step symbolic reasoning, whereas GSM8K comprises elementary- and middle-school-level math word problems involving only basic arithmetic and simple multi-step algebraic manipulation. Thus, diferences from table 1 reflect changes in both the baseline variant and evaluation population.

The compute interpretation also difers between methods. Both TIPS depths use their entire allocated horizon, whereas the comparison policy may terminate early. At cap 64, its deeper controller consumes 50.30 decoder calls on average, compared with 35.42 for the one-layer controller, despite their nearly identical accuracies. We therefore report mean calls alongside the cap-based accuracy curves.

Table 11 Matched policy-depth comparisons. Validation-set accuracy (%) at the requested decoder-call caps. Each TIPS decoding geometry is unchanged when depth is varied.
<table><tr><td>Method</td><td>Layers</td><td>Parameters</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>1</td><td>331,395</td><td>17.38</td><td>25.98</td><td>45.12</td><td>53.03</td><td>55.86</td><td>59.47</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>2</td><td>628,739</td><td>18.26</td><td>28.42</td><td>43.16</td><td>56.15</td><td>58.20</td><td>59.67</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td>1</td><td>331,395</td><td>12.70</td><td>20.90</td><td>40.43</td><td>52.64</td><td>58.79</td><td>61.33</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td>2</td><td>628,739</td><td>15.04</td><td>23.54</td><td>41.60</td><td>53.03</td><td>59.18</td><td>61.33</td></tr><tr><td>Jazbec et al. (2026)</td><td>1</td><td>331,265</td><td>7.32</td><td>14.16</td><td>32.32</td><td>43.85</td><td>43.75</td><td>43.95</td></tr><tr><td>Jazbec et al. (2026)</td><td>2</td><td>628,609</td><td>7.23</td><td>11.62</td><td>20.70</td><td>40.33</td><td>43.65</td><td>44.82</td></tr></table>

Table 12 Consumed decoder calls in the depth comparison. Mean calls per validation sample. TIPS uses every allocated call, whereas the comparison policy may stop before reaching the cap. Equal caps therefore do not always imply equal consumed compute.
<table><tr><td>Method</td><td>Cap 16</td><td>Cap 32</td><td> $\mathrm { C a p }$  64</td><td> $\mathrm { C a p }$  128</td></tr><tr><td>TIPS (either geometry or depth)</td><td>16.00</td><td>32.00</td><td>64.00</td><td>128.00</td></tr><tr><td>Jazbec et al., one layer</td><td>16.00</td><td>29.52</td><td>35.42</td><td>35.92</td></tr><tr><td>Jazbec et two layers  $\mathrm { a l . , }$ </td><td>16.00</td><td>31.95</td><td>50.30</td><td>51.38</td></tr></table>

![](images/f94db600c39ed63afec816e7dc00326cda7a15ce6afbfcb670bb4cc42ecb9088.jpg)  
Figure 8 Full-canvas comparison at matched policy depth. Validation-set accuracy for one- and two-layer TIPS and the learned unmasking policy of Jazbec et al. (2026). The horizontal axis is the requested decoder-call cap; consumed calls for adaptive policies are reported in Table 12.

![](images/c033cbaf711ce73f6627847e8e143e518bf86e5f1923776f6c7d93a00797ff2e.jpg)  
Figure 9 Policy depth under two-block decoding. Both curves use the same 2 × 128 geometry and exact global decoder-call budget. The one-layer variant approaches the two-layer reference at larger budgets and matches it at $T = 1 2 8$ in this single-seed comparison.

## B.3 Supplementary analyses of visibility and revision

Protocol. These experiments use $\mathrm { T I P S } _ { \mathrm { f u l l } }$ and $\mathrm { T I P S _ { b l o c k } }$ checkpoints with training seed 123, one evaluation seed, and the full GSM8K and MATH-500 test populations. Except for the explicitly selected corruptionreference pools, each reported mathematical-reasoning accuracy is micro-averaged over all 1,819 problems. The decoder remains frozen and uses greedy token proposals. We preserve each checkpoint’s native schedule and timing convention, including the block policy’s next-state local time input. All budgets count decoder calls, including deterministic finalization.

## B.3.1 Does TIPS select useful reveal and remask positions?

We examine a low-budget case, $\mathrm { T I P S _ { b l o c k } }$ at $T = 1 6 .$ , by intervening on one decision along an otherwise on-distribution trajectory. At within-block phases 0.25, 0.5, and 0.75 in each of the two blocks, we preserve the numbers of revealed and remasked positions but randomize the reveal positions, remask positions, or both. The random selections remain within the corresponding eligible masked or visible sets. Each continuation is evaluated on all test problems, including cases where an intervention leaves the action unchanged.

As seen in figure 10 and table 13, randomizing both sets reduces accuracy at all six sites in this case study. For example, at phase 0.25 of the first block, accuracy falls from 44.42% to 40.52%; at phase 0.5 of the second block, it falls to 40.79%. The separate interventions are less uniform: randomizing reveal or remask positions at phase 0.75 of the first block slightly improves accuracy. Thus, the selected example provides evidence that the locations chosen by TIPS can matter beyond their counts, particularly early in the trajectory, while retaining cases where a particular native decision is not beneficial. It is a focused case study rather than a summary of all geometries and budgets.

Table 13 Position-selection case study: $\mathrm { T I P S } _ { \mathrm { b l o c k } } , T = 1 6$ . Test micro-accuracy (%) for a count-preserving intervention at one decision. All six sites for this selected setting are included. Block indices are one-based.
<table><tr><td>Block</td><td>Phase</td><td>Native</td><td>Random reveal</td><td>Random remask</td><td>Random both</td></tr><tr><td>1</td><td>0.25</td><td>44.42</td><td>41.29</td><td>43.65</td><td>40.52</td></tr><tr><td>1</td><td>0.50</td><td>44.42</td><td>43.32</td><td>42.33</td><td>42.33</td></tr><tr><td>1</td><td>0.75</td><td>44.42</td><td>44.91</td><td>44.58</td><td>43.87</td></tr><tr><td>2</td><td>0.25</td><td>44.42</td><td>41.89</td><td>43.05</td><td>41.34</td></tr><tr><td>2</td><td>0.50</td><td>44.42</td><td>40.79</td><td>43.76</td><td>40.79</td></tr><tr><td>2</td><td>0.75</td><td>44.42</td><td>42.99</td><td>44.53</td><td>43.21</td></tr></table>

![](images/3bb3ff6f85ebb4b8af10e2125b96e5b3f4bd1c911c9dd23930edc680058cdbb5.jpg)  
Figure 10 Count-preserving position randomization for $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ at $T = 1 6 .$ . Positive values mean that native position selection yields higher final accuracy. All six intervention sites are shown, with one-based block indices. Reveal counts and remask counts are separately preserved at the intervened decision.

## B.3.2 Recovery from controlled token corruption

We test whether a policy can recover from deliberately introduced token errors. For each policy, we first generate one $T = 3 2$ response per test problem and retain verifier-accepted references. At within-block phase 0.75, we construct matched clean and corrupted partial states, retaining only references with at least eight eligible visible tokens. We replace $k \in \{ 1 , 2 , 4 , 8 \}$ visible tokens with uniformly drawn vocabulary tokens, excluding the original token, the mask, and tokenizer-defined special tokens. Corruption sets are nested across k, and every k for a given policy and intervention block uses the same reference responses.

The intervention leaves eight calls for the full-canvas policy. For the two-block policy, corrupting the first block leaves four calls in that block and sixteen in the second, while corrupting the second block leaves four calls. Table 14 gives the resulting populations.

Token restoration and answer correctness. TIPS frequently restores the exact substituted token, even though neither the policy nor the decoder receives the reference during continuation. As seen in table 15, for first-block corruption, $\mathrm { T I P S _ { b l o c k } }$ restores 76.53% of substituted tokens at k = 1 and 66.55% at $k = 8 ;$ the corresponding second-block rates are 66.35% and 57.23%. The full-canvas rates range from 42.50% to 46.41%. Exact restoration and answer correctness nevertheless measure diferent outcomes: an alternative token can preserve a correct solution, and restoring one token need not repair every consequence of the corruption. All corrupted conditions have lower answer accuracy than their matched clean continuations.

Restoration pathway and collateral drift. Most finally restored tokens last return through a learned reveal rather than deterministic finalization (Table 16). For example, with eight corrupted tokens in the first block, 5,011 of the 5,830 finally restored tokens return through learned reveals, and 819 return at finalization. However, corruption can also afect positions that were not substituted. Table 17 compares untouched-token disagreement with the reference against its paired clean-control rate. Excess disagreement is positive in all twelve corrupted conditions and generally grows with k. These measurements show both recovery and propagation of perturbations.

Table 14 Reference selection for controlled corruption. One $T = 3 2$ reference attempt is made per test problem for each policy. We retain verifier-accepted responses with at least eight eligible visible tokens at the intervention site. The same eligible pool is used for all k within a row.
<table><tr><td>Policy / active block</td><td>Accepted references</td><td>Eligible</td><td>Too few tokens</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td> $1 0 4 7 / 1 8 1 9$ </td><td>1047</td><td>0</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } } ,$  block 1</td><td>1095/1819</td><td>1095</td><td>0</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } } ,$  block 2</td><td>1095/1819</td><td>948</td><td>147</td></tr></table>

Table 15 Controlled corruption and recovery. Answer correctness is measured over eligible problems; exact restoration is measured over the k substituted tokens per problem. $k = 0$ denotes the reconstructed clean continuation. ∆ compares answer accuracy with that matched clean control.
<table><tr><td>Policy / active block</td><td>k</td><td>Correct / eligible</td><td>Answer (%)</td><td>Restored (%)</td><td>∆(pp)</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>0</td><td>1038/1047</td><td>99.14</td><td></td><td>+0.00</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>1</td><td>1030/1047</td><td>98.38</td><td>42.50</td><td>-0.76</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>2</td><td>1022/1047</td><td>97.61</td><td>42.50</td><td>-1.53</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>4</td><td>1007/1047</td><td>96.18</td><td>43.79</td><td>-2.96</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>8</td><td>979/1047</td><td>93.51</td><td>46.41</td><td>-5.64</td></tr><tr><td>TIPSblock, block 1</td><td>0</td><td>1076/1095</td><td>98.26</td><td></td><td>+0.00</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>1</td><td>1041/1095</td><td>95.07</td><td>76.53</td><td>-3.20</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>2</td><td>1045/1095</td><td>95.43</td><td>75.39</td><td>-2.83</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>4</td><td>1015/1095</td><td>92.69</td><td>71.80</td><td>-5.57</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>8</td><td>1001/1095</td><td>91.42</td><td>66.55</td><td>-6.85</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>0</td><td>948/948</td><td>100.00</td><td></td><td>+0.00</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>1</td><td>935/948</td><td>98.63</td><td>66.35</td><td>-1.37</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>2</td><td>925/948</td><td>97.57</td><td>65.14</td><td>-2.43</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>4</td><td>899/948</td><td>94.83</td><td>63.79</td><td>-5.17</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>8</td><td>838/948</td><td>88.40</td><td>57.23</td><td>-11.60</td></tr></table>

![](images/ab23f18f3ea99a5c46aca2142feb2d7ba65c0f660cae35b01c474ff71574694d.jpg)  
(a) Exact substituted-token restoration

![](images/7f3b39f6441a364fc64946846a75873be278699bad8b27044c459a74607c6883.jpg)  
(b) Final-answer correctness  
Figure 11 Recovery from controlled corruption. The horizontal axis is the number of substituted tokens. Each curve uses its own accepted-reference population and remaining decoding horizon, as detailed in the text. Panel (a) weights corrupted tokens; panel (b) weights eligible problems.

Table 16 Where restoration occurs. The learned-reveal and finalization columns partition the finally restored tokens by their last restoration event. The last column counts tokens restored temporarily but not present in their reference form at termination.
<table><tr><td>Policy / active block</td><td>k</td><td>Restored</td><td>Learned reveal</td><td>Finalization</td><td>Transient only</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>1</td><td>445</td><td>434</td><td>11</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>2</td><td>890</td><td>857</td><td>33</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>4</td><td>1834</td><td>1767</td><td>67</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td>8</td><td>3887</td><td>3747</td><td>140</td><td>4</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>1</td><td>838</td><td>736</td><td>102</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>2</td><td>1651</td><td>1442</td><td>209</td><td>2</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>4</td><td>3145</td><td>2724</td><td>421</td><td>3</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 1</td><td>8</td><td>5830</td><td>5011</td><td>819</td><td>7</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>1</td><td>629</td><td>536</td><td>93</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>2</td><td>1235</td><td>1067</td><td>168</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>4</td><td>2419</td><td>2094</td><td>325</td><td>0</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$  block 2</td><td>8</td><td>4340</td><td>3647</td><td>693</td><td>1</td></tr></table>

Table 17 Drift at untouched positions. We compare identical initially visible, nonspecial positions in each corrupted/clean pair, excluding substitutions. The first two rates and $\Delta _ { \mathrm { t o k e n } }$ weight token positions; $\Delta _ { \mathrm { e x a m p l e } }$ averages per-problem drift fractions. Diferences are in percentage points. A token mismatch does not necessarily change the final answer.
<table><tr><td></td><td>Policy / active block</td><td>k</td><td>Corrupt (%)</td><td>Clean (%)</td><td> $\Delta _ { \mathrm { t o k e n } }$ </td><td> $\Delta _ { \mathrm { e x a m p l e } }$ </td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td></td><td>1</td><td>0.15</td><td>0.06</td><td> $+ 0 . 0 8$ </td><td>+0.08</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td></td><td>2</td><td>0.19</td><td>0.06</td><td> $+ 0 . 1 3$ </td><td>+0.13</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td></td><td>4</td><td>0.30</td><td>0.06</td><td> $+ 0 . 2 4$ </td><td>+0.24</td></tr><tr><td> $\mathrm { T I P S _ { f u l l } }$ </td><td></td><td>8</td><td>0.54</td><td>0.07</td><td> $+ 0 . 4 7$ </td><td>+0.49</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 1</td><td>1</td><td>0.30</td><td>0.14</td><td> $+ 0 . 1 6$ </td><td>+0.16</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 1</td><td>2</td><td>0.37</td><td>0.14</td><td>+0.23</td><td>+0.23</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 1</td><td>4</td><td>0.73</td><td>0.15</td><td>+0.58</td><td>+0.58</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 1</td><td>8</td><td>1.20</td><td>0.15</td><td>+1.05</td><td>+1.06</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 2</td><td>1</td><td>0.13</td><td>0.02</td><td>+0.11</td><td>+0.13</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 2</td><td>2</td><td>0.23</td><td>0.02</td><td>+0.21</td><td>+0.24</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 2</td><td>4</td><td>0.41</td><td>0.02</td><td>+0.39</td><td>+0.53</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } , }$ </td><td>block 2</td><td>8</td><td>0.85</td><td>0.02</td><td>+0.82</td><td>+1.21</td></tr></table>

![](images/a08f872a5306fd578dd02d19854359d6093fcc2d611bb670beeed3315b7c8145.jpg)  
Figure 12 Excess drift at initially untouched positions. Token-weighted disagreement with the reference after a corrupted continuation, minus disagreement at the identical untouched positions in its matched clean continuation. Block numbers are one-based.

## B.3.3 Remasking is more frequent than token editing

We distinguish a remasking event from a completed token edit. A remasking episode starts when a visible token becomes masked and ends when that position next becomes visible; it is an edit only if the returning token difers from the original. Table 18 reports remasking under four denominators, preventing trajectory-level incidence from being confused with a per-position rate. Every observed trajectory contains at least one remask, but the fraction of visible position-opportunities that are remasked is approximately 18% for the full-canvas policy and 31–38% for the block policy.

Most remasking episodes do not change token identity. Across the measured budgets, 91.52–95.17% of full-canvas episodes and 92.91–98.16% of blockwise episodes return the original token. Every recorded episode resolves by the end of decoding, and the mean delay before re-reveal is 2.25–3.13 calls for full-canvas decoding and 1.20–1.41 calls for blockwise decoding (Table 19). Thus, remasking often temporarily withdraws context rather than replacing its content. Together with the remasking ablation, these observations motivate the hypothesis that hiding unreliable context can help other positions even when the hidden token ultimately returns unchanged. Another potential explanation is simply due to sampling noise, as each re-masking decision is sampled from a Bernoulli distribution informed by the policy network. The episode counts do not isolate that mechanism from the efects of actual token edits

Table 18 How often TIPS remasks. All 1,819 test problems are included in each row. The four rates use, respectively, visible position-opportunities, all active position-opportunities, learned steps containing a remask, and trajectories containing a remask. A position-opportunity is one active position at one learned decision. Deterministic finalization is excluded.
<table><tr><td>Policy</td><td>T</td><td>Remasks</td><td>Visible (%)</td><td>All positions (%)</td><td>Steps (%)</td><td>Trajectories (%)</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>503,803</td><td>18.04</td><td>7.21</td><td>93.29</td><td>100.00</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>1,119,399</td><td>17.97</td><td>7.75</td><td>96.57</td><td>100.00</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>2,457,136</td><td>18.71</td><td>8.38</td><td>97.97</td><td>100.00</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>16</td><td>324,263</td><td>30.50</td><td>9.95</td><td>85.31</td><td>100.00</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>32</td><td>929,911</td><td>34.23</td><td>13.31</td><td>92.91</td><td>100.00</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>64</td><td>2,315,040</td><td>38.11</td><td>16.04</td><td>96.02</td><td>100.00</td></tr></table>

Table 19 What happens after remasking. An episode begins when a visible token is remasked and ends at its next reveal. Changed and same-token percentages use completed episodes; every recorded episode completes. Delay is the mean number of global forward-call intervals until re-reveal. Finalized counts episodes ending at deterministic finalization. Special is the fraction of remask events involving tokenizer-defined special tokens, without opportunity adjustment.
<table><tr><td>Policy</td><td>T</td><td>Changed</td><td>Changed (%)</td><td>Same (%)</td><td>Delay</td><td>Finalized</td><td>Special (%)</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>42,714</td><td>8.48</td><td>91.52</td><td>2.245</td><td>29,553</td><td>45.36</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>72,426</td><td>6.47</td><td>93.53</td><td>2.699</td><td>16,413</td><td>36.39</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>118,743</td><td>4.83</td><td>95.17</td><td>3.126</td><td>8,332</td><td>31.96</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td>16</td><td>22,988</td><td>7.09</td><td>92.91</td><td>1.198</td><td>48,771</td><td>25.16</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td>32</td><td>31,264</td><td>3.36</td><td>96.64</td><td>1.294</td><td>31,638</td><td>18.69</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td>64</td><td>42,633</td><td>1.84</td><td>98.16</td><td>1.412</td><td>17,001</td><td>14.76</td></tr></table>

## B.3.4 Continuing heuristic trajectories with TIPS

We extend the main-text handof experiment to all budgets and switching fractions. A non-adjusted confidencethreshold sampler based on Fast-dLLM first runs for $q T$ global decoder calls, with $q \in \{ 0 , 1 / 4 , 1 / 2 , 3 / 4 , 1 \}$ after which TIPS uses the remaining budget. The heuristic reveals eligible masked proposals with confidence strictly above 0.9 and, when none qualifies, reveals the highest-confidence eligible proposal for each example. TIPS resumes at the corresponding point of its native schedule for normalized time input and logit calibration purposes.

For a matched budget comparison, all arms use exactly T real forward passes, including calls that leave the canvas unchanged and deterministic finalizations. In particular, the $q = 1$ endpoint is this fixed-horizon heuristic. We retain both endpoints and all three hybrid settings in Table 20.

Sixteen of the eighteen hybrid conditions improve over the matched heuristic endpoint. At $T = 6 4$ , full-canvas handof at $q = 1 / 4$ raises accuracy from 23.91% to 40.85%, while blockwise handof at $q = 3 / 4$ raises it from 34.91% to 40.19%. The paired counts show that these gains can include both repaired and newly incorrect final answers: the former full-canvas condition changes 491 heuristic failures into correct answers while changing 183 heuristic successes into failures. No hybrid exceeds pure TIPS at the same budget. The result therefore supports TIPS’s ability to improve continuations from another sampler, rather than recommending a heuristic prefix over starting with TIPS.

The results also support a comparison at the remaining budget. For $T = 6 4$ and $q = 1 / 2$ , handof leaves 32 calls for TIPS and yields 31.34% and 38.15% accuracy for $\mathrm { T I P S } _ { \mathrm { f u l l } }$ and $\mathrm { T I P S _ { b l o c k } }$ , respectively. Their from-scratch 32-call runs instead achieve 57.56% and 60.20%, suggesting that restarting from an all-mask canvas can be preferable to retaining the heuristic state.

Table 20 Complete heuristic-to-TIPS handoff results. Test micro-accuracy (%) at all global switching fractions. $q = 0$ is pure TIPS; q = 1 is the matched fixed-horizon non-adjusted confidence-threshold heuristic. Every row uses exactly T real decoder calls, including no-op and finalization calls.
<table><tr><td>Policy</td><td>T</td><td> $q = 0$ </td><td> $q = 1 / 4$ </td><td> $q = 1 / 2$ </td><td> $q = 3 / 4$ </td><td> $q = 1$ </td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>46.23</td><td>24.02</td><td>16.99</td><td>17.26</td><td>17.70</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>57.56</td><td>31.45</td><td>20.56</td><td>19.79</td><td>19.74</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>63.39</td><td>40.85</td><td>31.34</td><td>25.73</td><td>23.91</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>16</td><td>44.42</td><td>17.98</td><td>20.45</td><td>20.62</td><td>15.94</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>32</td><td>60.20</td><td>24.85</td><td>22.76</td><td>26.06</td><td>20.67</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>64</td><td>64.54</td><td>38.70</td><td>38.15</td><td>40.19</td><td>34.91</td></tr></table>

![](images/9102288eb20f5319126149120d0239d415427d8bf4e63116682f29e193282f9f.jpg)  
(a) $\mathrm { T I P S } _ { \mathrm { f u l l } }$

![](images/901ce45eca54f3bd89a606ab0e2343532d72619e6a75793b90082ca21044b2df.jpg)  
(b) $\mathrm { T I P S _ { b l o c k } }$  
Figure 13 Heuristic prefixes followed by TIPS. Test micro-accuracy at every evaluated switching fraction, including pure TIPS (q = 0) and the matched heuristic (q = 1). Each curve has a fixed total decoder-call budget.

Table 21 Paired final-answer changes after handoff. Counts compare each hybrid with the matched pure-heuristic endpoint on the same 1,819 problems. Correctness refers to terminal answers. Both beneficial and adverse transitions are retained.
<table><tr><td>Policy</td><td>T</td><td>q</td><td>Wrong → correct</td><td>Correct → wrong</td><td>∆ (pp)</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>0.25</td><td>267</td><td>152</td><td>+6.32</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>0.50</td><td>91</td><td>104</td><td>-0.71</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16</td><td>0.75</td><td>42</td><td>50</td><td>-0.44</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>0.25</td><td>394</td><td>181</td><td>+11.71</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>0.50</td><td>137</td><td>122</td><td>+0.82</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>32</td><td>0.75</td><td>50</td><td>49</td><td>+0.05</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>0.25</td><td>491</td><td>183</td><td>+16.93</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>0.50</td><td>284</td><td>149</td><td>+7.42</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>64</td><td>0.75</td><td>88</td><td>55</td><td>+1.81</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>16</td><td>0.25</td><td>186</td><td>149</td><td>+2.03</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>16</td><td>0.50</td><td>195</td><td>113</td><td>+4.51</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>16</td><td>0.75</td><td>143</td><td>58</td><td>+4.67</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>32</td><td>0.25</td><td>246</td><td>170</td><td>+4.18</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>32</td><td>0.50</td><td>184</td><td>146</td><td>+2.09</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>32</td><td>0.75</td><td>163</td><td>65</td><td>+5.39</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>64</td><td>0.25</td><td>316</td><td>247</td><td>+3.79</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>64</td><td>0.50</td><td>205</td><td>146</td><td>+3.24</td></tr><tr><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>64</td><td>0.75</td><td>167</td><td>71</td><td>+5.28</td></tr></table>

## B.3.5 Repairing completed non-adjusted Fast-dLLM responses

We next test whether TIPS can improve responses that another sampler has already completed, rather than continue a partially decoded trajectory. We first run non-adjusted Fast-dLLM on frozen LLaDA-8B-Instruct with $8 \times 3 2$ decoding, confidence threshold 0.9, no caching, and a global cap of 256 decoder calls. We then apply the frozen $\mathrm { T I P S _ { f u l l } }$ policy to every completed 256-token canvas on the full GSM8K and MATH-500 test sets, using exactly four additional decoder evaluations. Repair retains the native schedule and follows the final four calls of a virtual 32-call horizon: three policy transitions receive normalized times 28/32, 29/32, and $3 0 / 3 2$ , with next-state visibility targets $( ( t + 1 ) / 3 2 ) ^ { \mathrm { e x p ( \gamma _ { \mathrm { f u l l } } ) } }$ for $t \in \{ 2 8 , 2 9 , 3 0 \}$ , followed by deterministic finalization. The decoder and policy remain frozen, and correctness feedback is used only for scoring. As shown in Table 22 and Figure 14, micro-accuracy increases from 67.01% to 67.29%, a gain of 0.27 percentage points. TIPS corrects five of the 600 initially incorrect answers—four on GSM8K and one on MATH-500—while all 1,219 initially correct answers remain correct. This single-seed result shows a modest improvement from revisiting completed responses, extending revision beyond unfinished trajectories.

Table 22 Four-call repair of completed non-adjusted Fast-dLLM responses. Test accuracy before and after applying $\mathrm { T I P S } _ { \mathrm { f u l l } }$ to the same saved responses, with training and evaluation seed 123. $\mathrm { W } {  } \mathrm { C }$ and $\mathrm { C } \to \mathrm { W }$ count wrong-to-correct and correct-to-wrong final-answer transitions, respectively. Every response is repaired, regardless of its original correctness. Non-adjusted Fast-dLLM has a 256-call global cap but terminates naturally on every problem. Diferences are computed from unrounded accuracies.
<table><tr><td></td><td></td><td colspan="3">Accuracy (%)</td><td colspan="2">Answer transitions</td></tr><tr><td>Dataset</td><td>N</td><td>Non-adjusted Fast-dLLM</td><td> $+ \mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td> $\Delta \ \mathrm { ( p p ) }$ </td><td>W→C</td><td>C→W</td></tr><tr><td>GSM8K</td><td>1,319</td><td>79.30</td><td>79.61</td><td>+0.30</td><td>4</td><td>0</td></tr><tr><td>MATH-500</td><td>500</td><td>34.60</td><td>34.80</td><td>+0.20</td><td>1</td><td>0</td></tr><tr><td>Micro-average</td><td>1,819</td><td>67.01</td><td>67.29</td><td>+0.27</td><td>5</td><td>0</td></tr></table>

![](images/4dc6c7b6a61f664b4b9427615c82d8e74323448d31bd55e3e1c916c73390a869.jpg)  
Figure 14 Accuracy changes after four additional repair calls. Points show the paired accuracy change from completed non-adjusted Fast-dLLM responses to their $\mathrm { T I P S } _ { \mathrm { f u l l } }$ revisions. Whiskers are 95% paired bootstrap intervals from 10,000 resamples, sampling problems within each benchmark at its original size. The dashed line marks no accuracy change.

## B.4 Generalization beyond the training configuration

Protocol. We evaluate transfer with one training lineage and one evaluation seed, a 256-token response, and budgets $T \in \{ 1 6 , 3 2 , 6 4 \}$ . Mathematical-reasoning results use the complete GSM8K and MATH-500 test sets. Coding results use all 164 HumanEval and 500 MBPP test tasks and report pass@1 from one generation per task. Policy weights remain frozen for the zero-shot experiments; coding-trained policies are evaluated separately. Baseline settings are fixed independently of coding test outcomes. We retain a reduced baseline roster – selected based on achieving the best non-TIPS results in mathematical reasoning in table 1 – due to computational constraints.

## B.4.1 Zero-shot transfer to Dream-7B

We apply the LLaDA-trained $\mathrm { T I P S } _ { \mathrm { f u l l } }$ policy to Dream-v0-Instruct-7B without retraining. This comparison uses a single 1×256 canvas for both TIPS and Fast-dLLM and preserves the Dream logit-to-position alignment. As the policy observes confidence features rather than token identities or decoder hidden states, its input interface does not require a shared vocabulary or hidden-state dimension.

The transferred policy reaches 35.18%, 46.56%, and 52.39% test micro-accuracy at $T = 1 6 , 3 2 , 6 4$ , compared with 22.87%, 27.43%, and 29.96% for the matched full-canvas Fast-dLLM baseline. Table 23 expands the main-text comparison with per-dataset scores and actual mean decoder calls. These results demonstrate transfer to another frozen backbone in this setting; they do not require adapting the policy on Dream outputs or selecting its parameters from Dream test scores.

Table 23 Dream transfer by dataset and consumed compute. Columns correspond to requested budgets 16, 32, and 64. Both methods decode a single 256-token canvas. The pooled row weights individual problems equally.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Accuracy (%)</td><td colspan="3">Mean decoder calls</td></tr><tr><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td></tr><tr><td>GSM8K</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>41.77</td><td>55.72</td><td>63.84</td><td>16.00</td><td>32.00</td><td>64.00</td></tr><tr><td>GSM8K</td><td>Fast-dLLM</td><td>25.32</td><td>30.78</td><td>34.57</td><td>15.48</td><td>32.75</td><td>66.33</td></tr><tr><td>MATH-500</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>17.80</td><td>22.40</td><td>22.20</td><td>16.00</td><td>32.00</td><td>64.00</td></tr><tr><td>MATH-500</td><td> $_ \mathrm { F a s t - d L L M }$ </td><td>16.40</td><td>18.60</td><td>17.80</td><td>17.50</td><td>30.70</td><td>58.21</td></tr><tr><td>Micro-average</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>35.18</td><td>46.56</td><td>52.39</td><td>16.00</td><td>32.00</td><td>64.00</td></tr><tr><td>Micro-average</td><td> $_ \mathrm { F a s t - d L L M }$ </td><td>22.87</td><td>27.43</td><td>29.96</td><td>16.03</td><td>32.19</td><td>64.10</td></tr></table>

![](images/23944b5965c066ed7c962d04942144afac63074f4ae3d61cd863447823f86c29.jpg)  
Figure 15 Zero-shot backbone transfer on a matched full canvas. Test micro-accuracy for the frozen LLaDA-trained $\mathrm { T I P S } _ { \mathrm { f u l l } }$ policy and Fast-dLLM on Dream-7B. Both use $1 \times 2 5 6$ decoding. The horizontal axis is the decoder-call budget; actual baseline consumption is reported in Table 23.

## B.4.2 Zero-shot transfer from mathematics to coding

We next apply the math-trained full-canvas and blockwise policies to coding without adaptation. Unlike the backbone-transfer experiment, this changes the task domain while retaining LLaDA. Table 24 reports every retained baseline individually: UNCODE/PC-Sampler on $1 \times 2 5 6$ , WINO on $1 \times 2 5 6$ and $2 \times 1 2 8$ , and Non-adjusted Fast-dLLM on $1 \times 2 5 6$ and $8 \times 3 2$ . These baselines retain their fixed mathematical-reasoning presets or assets rather than being tuned on coding test scores.

Transfer is mixed. At $T = 1 6 , \mathrm { T I P S _ { f u l l } }$ achieves the highest measured HumanEval score among these methods, $1 6 . 4 6 \%$ , but it does not maintain that advantage as the budget grows. At $T = 6 4$ , Non-adjusted Fast-dLLM with $8 \times 3 2$ decoding reaches 40.24% on HumanEval, compared with 22.56% and 29.27% for the full and block TIPS policies. On MBPP, the two policies reach 29.00% and 33.00%, compared with 38.20% for that baseline. Thus, the mathematical-reasoning policy transfers nontrivially, but the gains observed in its training domain do not translate into consistent superiority on coding.

Table 24 Zero-shot coding transfer. TIPS uses frozen math-trained policies. Pass@1 (%), one generation per test task, against each of the five retained baselines individually. Baseline presets/assets are inherited from mathematica reasoning. The 8×32 baseline uses a shared global cap. Adaptive methods may consume fewer than T calls.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Geometry</td><td colspan="3">HumanEval (164)</td><td colspan="3">MBPP (500)</td></tr><tr><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>UNCODE / PC-Sampler</td><td>1 × 256</td><td>4.27</td><td>6.10</td><td>16.46</td><td>0.00</td><td>2.80</td><td>15.20</td></tr><tr><td>WINO</td><td>1 × 256</td><td>15.24</td><td>21.95</td><td>24.39</td><td>16.20</td><td>26.80</td><td>32.40</td></tr><tr><td>WINO</td><td>2 × 128</td><td>12.20</td><td>23.78</td><td>30.49</td><td>1.80</td><td>11.40</td><td>29.80</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>1 × 256</td><td>15.24</td><td>21.95</td><td>23.78</td><td>16.80</td><td>23.40</td><td>26.20</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>8 × 32</td><td>14.63</td><td>32.93</td><td>40.24</td><td>1.20</td><td>13.40</td><td>38.20</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td> $1 \times 2 5 6$ </td><td>16.46</td><td>23.78</td><td>22.56</td><td>10.60</td><td>21.80</td><td>29.00</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td> $2 \times 1 2 8$ </td><td>13.41</td><td>27.44</td><td>29.27</td><td>8.00</td><td>23.60</td><td>33.00</td></tr></table>

(a) HumanEval  
![](images/03cf33092cb3ade7103a34f3a7f367247f279930e957dd4e78de88c6d5301a47.jpg)

![](images/91fab1c95469f8312174c12c7bfc6c2c0c101237fe0dca13424bd7e1c059557c.jpg)  
(b) MBPP  
Figure 16 Zero-shot coding transfer against individual baselines. One-generation pass@1 versus log decoder-call cap. TIPS uses its frozen math-trained policies.

## B.4.3 Training TIPS for coding

We separately train full-canvas and blockwise policies for coding while keeping the decoder frozen, using supervised warmup followed by an RLVR continuation to the fixed 935-update endpoint. Table 25 compares these policies with the same five baselines, so changes relative to the zero-shot table reflect the policy’s domain adaptation.

Coding training improves the full-canvas policy in every measured dataset–budget condition. Its HumanEval pass@1 increases to 19.51%, 25.61%, and 27.44%, while its MBPP score increases to 24.00%, 31.80%, and 37.80%. The largest gains over the corresponding math-trained full-canvas policy occur on MBPP, reaching 13.40 percentage points at $T \ = \ 1 6$ The coding-trained full policy exceeds every retained baseline on HumanEval at $T = 1 6$ and on MBPP at $T = 1 6$ and 32. Nevertheless, Non-adjusted Fast-dLLM with $8 \times 3 2$ decoding remains stronger on HumanEval at $T = 3 2$ and 64, and slightly stronger on MBPP at $T = 6 4$

The block policy responds less consistently to coding training: it improves at $T = 1 6$ on both tasks but falls below its zero-shot counterpart at $T = 6 4$ . These results support domain-specific policy training, especially for full-canvas decoding, while leaving room for a coding-specific recipe and hyperparameter study. We do not infer that the present coding configuration is optimal. Table 26 additionally reports actual mean calls, since adaptive baselines may finish below their requested caps.

Table 25 Coding-trained TIPS. TIPS policies are trained separately on coding; the decoder remains frozen. Pass@1 (%), one generation per test task, against each of the five retained baselines individually. Baseline presets/assets and TIPS settings are inherited from mathematical reasoning and untuned. The $8 \times 3 2$ baseline uses a shared global cap. Adaptive methods may consume fewer than T calls.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Geometry</td><td colspan="3">HumanEval (164)</td><td colspan="3">MBPP (500)</td></tr><tr><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>UNCODE /PC-Sampler</td><td>1 × 256</td><td>4.27</td><td>6.10</td><td>16.46</td><td>0.00</td><td>2.80</td><td>15.20</td></tr><tr><td>WINO</td><td>1 × 256</td><td>15.24</td><td>21.95</td><td>24.39</td><td>16.20</td><td>26.80</td><td>32.40</td></tr><tr><td>WINO</td><td>2 × 128</td><td>12.20</td><td>23.78</td><td>30.49</td><td>1.80</td><td>11.40</td><td>29.80</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>1 × 256</td><td>15.24</td><td>21.95</td><td>23.78</td><td>16.80</td><td>23.40</td><td>26.20</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>8 × 32</td><td>14.63</td><td>32.93</td><td>40.24</td><td>1.20</td><td>13.40</td><td>38.20</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>1 × 256</td><td>19.51</td><td>25.61</td><td>27.44</td><td>24.00</td><td>31.80</td><td>37.80</td></tr><tr><td> $\mathrm { T I P S } _ { \mathrm { b l o c k } }$ </td><td> $2 \times 1 2 8$ </td><td>16.46</td><td>26.83</td><td>21.95</td><td>13.00</td><td>23.60</td><td>32.40</td></tr></table>

(a) HumanEval  
![](images/e1a93a42ae58c1973fba90674de63af2e819e56b5b41232bf59bc7e26b2de418.jpg)

![](images/65230f877958a8d3e65dba7e813d7a917d7d5c06f7ea63c86e240d4e852e1fd3.jpg)  
(b) MBPP  
Figure 17 Coding-trained TIPS against individual baselines. The decoder is unchanged; only the TIPS policies are trained for coding. Baselines are identical to those in Figure 16. Curves show one evaluation sample.

Table 26 Consumed compute in coding experiments. Mean decoder calls at the indicated caps, for the same conditions as Tables 24 and 25. Both math-trained and coding-trained TIPS use the exact requested horizon.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Geometry</td><td colspan="3">HumanEval (164)</td><td colspan="3">MBPP (500)</td></tr><tr><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>UNCODE / PC-Sampler</td><td> $1 \times 2 5 6$ </td><td>16.00</td><td>32.00</td><td>64.00</td><td>16.00</td><td>32.00</td><td>64.00</td></tr><tr><td>WINO</td><td> $1 \times 2 5 6$ </td><td>15.84</td><td>25.26</td><td>31.24</td><td>15.94</td><td>25.20</td><td>33.71</td></tr><tr><td>WINO</td><td> $2 \times 1 2 8$ </td><td>16.00</td><td>24.84</td><td>31.19</td><td>16.00</td><td>28.73</td><td>41.09</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td> $1 \times 2 5 6$ </td><td>13.21</td><td>18.83</td><td>21.01</td><td>14.95</td><td>23.45</td><td>28.65</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td> $8 \times 3 2$ </td><td>15.93</td><td>28.51</td><td>37.82</td><td>16.00</td><td>31.76</td><td>53.07</td></tr><tr><td>TIPS (both policies)</td><td>native</td><td>16.00</td><td>32.00</td><td>64.00</td><td>16.00</td><td>32.00</td><td>64.00</td></tr></table>

## B.4.4 Zero-shot transfer across block sizes

We reuse the full-canvas-trained policy for target geometries $1 \times 2 5 6 , 2 \times 1 2 8 , 4 \times 6 4$ , and $8 \times 3 2$ , without retraining or fitting a geometry-specific schedule. The total response length remains 256 tokens, and the global decoder budget is divided equally across blocks, including one deterministic finalization per block. Consequently, increasing the number of blocks changes both the visible-context structure and the number of learned decisions available within each block.

Transfer is strongest when suficient within-block compute remains available. At $T = 6 4$ , the original fullcanvas and transferred two-block configurations reach 62.67% and 62.89% micro-accuracy, respectively; the four- and eight-block configurations retain 59.81% and 55.25%. At $T = 1 6$ , however, accuracy decreases from 47.00% on the full canvas to 22.43% with eight blocks. In the latter case, each block receives only two calls: one learned decision and one finalization. The results therefore show useful transfer to smaller blocks, particularly at larger budgets, while also exposing the cost of reducing the number of decisions per block.

Table 27 Frozen-policy block-size transfer. All rows use the full-canvas-trained checkpoint, without block-specific policy training. Values are test accuracy (%). The total 256-token response and total decoder-call budget remain fixed.
<table><tr><td>Target geometry</td><td>Dataset</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td> $1 \times 2 5 6$ </td><td>GSM8K</td><td>57.39</td><td>71.72</td><td>74.75</td></tr><tr><td> $1 \times 2 5 6$ </td><td>MATH-500</td><td>19.60</td><td>25.80</td><td>30.80</td></tr><tr><td> $1 \times 2 5 6$ </td><td>Micro-average</td><td>47.00</td><td>59.10</td><td>62.67</td></tr><tr><td> $2 \times 1 2 8$ </td><td>GSM8K</td><td>44.73</td><td>66.41</td><td>75.74</td></tr><tr><td> $2 \times 1 2 8$ </td><td>MATH-500</td><td>15.80</td><td>25.80</td><td>29.00</td></tr><tr><td> $2 \times 1 2 8$ </td><td>Micro-average</td><td>36.78</td><td>55.25</td><td>62.89</td></tr><tr><td> $4 \times 6 4$ </td><td>GSM8K</td><td>27.90</td><td>57.01</td><td>72.02</td></tr><tr><td> $4 \times 6 4$ </td><td>MATH-500</td><td>13.00</td><td>20.40</td><td>27.60</td></tr><tr><td> $4 \times 6 4$ </td><td>Micro-average</td><td>23.80</td><td>46.95</td><td>59.81</td></tr><tr><td> $8 \times 3 2$ </td><td>GSM8K</td><td>26.46</td><td>40.71</td><td>66.26</td></tr><tr><td> $8 \times 3 2$ </td><td>MATH-500</td><td>11.80</td><td>16.40</td><td>26.20</td></tr><tr><td> $8 \times 3 2$ </td><td>Micro-average</td><td>22.43</td><td>34.03</td><td>55.25</td></tr></table>

![](images/52e941fd0660cab03e98b0c41aa667a2508c21d817716f5af14963b350563352.jpg)  
Figure 18 Frozen-policy transfer across decoding geometries. Every curve uses the same full-canvas-trained checkpoint and the same global 256-token response length. The decoder-call budget is shared across blocks.

## B.4.5 Sensitivity to decoder proposal temperature

We vary token-proposal temperature over $\{ 0 , 0 . 1 , 0 . 5 , 1 . 0 \}$ for the frozen full-canvas policy, leaving policy-action sampling and the schedule unchanged. Finalization remains greedy in every condition. Small proposal noise is competitive: temperature 0.1 gives 48.32% at $T = 1 6$ and 63.33% at $T = 6 4$ , slightly above the corresponding greedy scores of 47.00% and 62.67%, while greedy proposals are better at $T = 3 2$ . Increasing temperature to 1.0 reduces accuracy at all three budgets, with the largest loss at $T = 1 6$ (37.77% versus 47.00%). These measurements support greedy decoding as a strong default.

Table 28 Sensitivity to proposal temperature. Frozen $\mathrm { T I P S } _ { \mathrm { f u l l } }$ on the pooled math test set. Only token-proposal temperature changes; finalization remains greedy. Values are single-seed micro-accuracy (%).
<table><tr><td>Proposal temperature</td><td> $T = 1 6$ </td><td> $T = 3 2$ </td><td> $T = 6 4$ </td></tr><tr><td>0.0</td><td>47.00</td><td>59.10</td><td>62.67</td></tr><tr><td>0.1</td><td>48.32</td><td>58.11</td><td>63.33</td></tr><tr><td>0.5</td><td>46.89</td><td>56.02</td><td>62.62</td></tr><tr><td>1.0</td><td>37.77</td><td>51.35</td><td>60.75</td></tr></table>

![](images/c6124cf62f8c95b51d59224896b30d60b8c52134c23a4b19e7f8890d66ec5312.jpg)  
Figure 19 Sensitivity to proposal temperature. Test micro-accuracy for the same frozen $\mathrm { T I P S } _ { \mathrm { f u l l } }$ policy. Only token proposals change temperature; finalization remains greedy, and visibility actions retain the same sampling rule.

## B.5 Training diagnostics

We report training traces separately from downstream test performance. In the mathematical-reasoning run, supervised warmup comprises an initial 3,230-update phase followed by a final 4,074-update phase after refreshing the state dataset. Figure 20 shows the recorded optimized loss in each phase.

Figure 21 additionally reports recomputed mathematical-reasoning validation observations on a fixed dataset of 36,363 states from 1,173 reserved training examples. These are post-hoc diagnostic evaluations and use a diferent population from the 1,024-example validation ablations. Figure 22 retains the coding run’s recorded training and validation loss curves. All three figures concern supervised warmup of the small policy network.

![](images/6373bc6123c6ecbadf460245ee36b964ab8b616dc01991d08ff18d9c986a70b7.jpg)

![](images/48329422d9cdce215bba38c0340113a09e462dfece6518d6cfb3a967c5d82e03.jpg)  
Figure 20 Supervised warmup on mathematics. Recorded training soft binary cross-entropy in the initial and final warmup phases. Faint traces are raw observations; the darker curves are exponential moving averages with smoothing coeficient 0.1. The phases have separate optimizer-step axes.

![](images/e3d9d322fbe977cefbff5602995a20880ccaf49cc07b884bbe04a56acf77465f.jpg)

![](images/ef4eb63ad1c3daa7ff1f184cb855d1551bbd761a8d5de3cd6f90a43211a39d0a.jpg)  
Validation: 1,173 reserved training examples • fixed 36,363-state bank • no historical online validation log  
Figure 21 Recomputed warmup validation curves. Recorded mathematical-reasoning training loss and validation observations recomputed on a fixed state bank, with exponential moving averages for visualization.

![](images/e21d3e876689a744c002570777338c38efbde596a675b893613a0b8bfaef56c2.jpg)

![](images/c61d89c1033793b676b49e21e693fea705ec4989dc65375838ea2b32e896e9c7.jpg)  
Figure 22 Supervised warmup on coding. Supplied training and validation soft binary cross-entropy traces for the initial and final warmup phases. The displayed averages summarize within-run loss observations.

RLVR trajectories. Figure 23 shows raw terminal correctness rewards and exponential moving averages for the mathematical-reasoning and coding runs. Each nominal update contains 128 terminal outcomes, and each continuation ends at nominal update 935. Table 29 summarizes the first and last 50 updates without smoothing. For mathematics, mean reward increases from approximately 38% to 50% for both geometries. For coding, it increases from 16.28% to 37.55% for the full policy and from 13.66% to 27.41% for the block policy. In the coding full-canvas run, two nominal updates are skipped by the original optimizer rule, leaving 933 optimizer steps; the plot retains all 935 nominal updates and their recorded outcomes.

Table 29 RLVR training diagnostics. Mean raw terminal-correctness reward (%) over the first and last 50 recorded updates of each run.
<table><tr><td>Domain</td><td>Policy</td><td>First 50 updates</td><td>Last 50 updates</td></tr><tr><td>Math</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>38.19</td><td>50.73</td></tr><tr><td>Math</td><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>37.55</td><td>50.41</td></tr><tr><td>Coding</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td>16.28</td><td>37.55</td></tr><tr><td>Coding</td><td> $\mathrm { T I P S _ { b l o c k } }$ </td><td>13.66</td><td>27.41</td></tr></table>

![](images/1ffc794a3370a6549b5a20c9885acf078526cbdc2721b893dc5826feffc1b19b.jpg)  
(a) Mathematical reasoning

![](images/9b793f758ed9745410efc06ef0f6846ba0eb77f4b70c1d2b908c19bed8eab1e3.jpg)  
(b) Coding  
Figure 23 RLVR training curves. Terminal correctness reward per nominal update. Faint lines show the raw observations; darker lines show exponential moving averages with smoothing coeficient 0.1.

## B.6 Illustrative decoding trajectories

We show four trajectories with verifier-accepted final answers—one for each policy and mathematical-reasoning dataset—and one unsuccessful full-canvas trajectory. All use $T = 3 2$ . These examples were selected for readable outputs and visible ordinary-token edits. In particular, an accepted terminal answer does not certify every intermediate reasoning step or establish that a displayed edit caused the success.

The canvases retain the recorded token-level displays. A square denotes a masked position; a prefixed + marks a reveal, a prefixed − a remask, and a prefixed ∗ a changed re-reveal. These prefixes are display annotations. Each snapshot is labeled by the number of decoder calls already consumed, while active-block indices in the recorded canvases are zero-based. Deterministic finalizations are identified explicitly, and earlier finalized blocks remain frozen. The prompt instructions are omitted from the figures, while the task is given in each caption.

![](images/3c8d93fc9a1b70a54ad561ccf56dc52c13d4f78bae3895775a6dac16d93d7fa5.jpg)  
Figure 24 Accepted full-canvas trajectory on GSM8K. Janet’s ducks lay 16 eggs daily; she uses three for breakfast and four for mufins, then sells the remainder for two dollars each. The final response correctly computes (16 −3−4)×2 = 18 dollars. The snapshots show progressive visibility, remasking, and finalization.

![](images/7607f7f309afea2cde7f410f07d26fdfe2f2bdf260165f38fdbee6980ced3bb6.jpg)  
Figure 25 Accepted full-canvas trajectory on MATH-500. The task asks how many positive whole-number divisors 196 has. The response arrives at $1 9 6 = 2 ^ { 2 } \times 7 ^ { 2 }$ and the accepted answer $( 2 + 1 ) ( 2 + 1 ) = 9$ . Token-level reveal and remask annotations are retained in the displayed intermediate and final canvases.

![](images/64a4e5442f9c5242a6f9168431bc36e3beed3f997c6532cbe5332e1a68016695.jpg)  
Figure 26 Accepted blockwise trajectory on GSM8K. A robe requires two bolts of blue fiber and half as much white fiber, for a total of three bolts. The first block contains the solution and is finalized after 16 calls; the second block is then decoded under the same global budget. This example also illustrates that TIPS continues to the allocated horizon even when the answer-bearing text appears earlier.

[block 0] +\n□+ find+ the□+ between□+ lines□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□   
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□[block 1] □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□   
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Before forward 1   
[initial response: 256 masked positions] □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□   
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□   
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

After forward 1/32 · active block 0 · local policy time 0.0625

After forward 4/32 · active block 0 · local policy time 0.25

[block 0] \n−□□−□−□+ between+ two−□□□□□□−□□−□□□□□□□□□−□□□□□□□□□□□□□□□□□+ first□□□−□+\[−□x□+ −□y−□+   
-+z+\]□+This□□−□□□□□+,□□□+ -□□+\n□+The−□□□□□+\[+6+x+ =−□□−□+ -+4−□+\]+\n□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□[block 1] □□□   
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□   
□□□□□

After forward 9/32 · active block 0 · local policy time 0.5625

[block 0] −□+To find+ the+ angle−□ two−□+,−□□□□□□\* of−□+ lines□□□□□□ the□□□□□□□□□□□□−□+\n\nThe□−□ is+:\n+\[−□+x−□□−□y+ =+ -−□+\]\nThis□□+ the−□+ vector (2, +3−□−□+1−□−□+\n+The+ second−□ is:+\n\[6+x =+ -y−□−□−□+z+\]+\n+This□□ the direction+ vector (+6,+ -−□−□−□4+).−□\n□□□□□□□□□□□□□□□□□□[block 1] □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

After forward 16/32 · active block 0 · local policy time — · DETERMINISTIC FINALIZATION [block 0] \nTo find the angle between the lines, we first identify the direction vectors of the+ lines. The direction vectors are obtained by equating the variables and taking the coefficients of+ the given equations.\n\nThe first line is:\n\[2x =+ 3y = -z\]\nThis gives us the direction vector (2, +3, -1).\n\n+The second line is:\n\[6x = -y = -4z\]\nThis gives us the direction vector (6, -1, -4).\n\nThe angle between the lines can be found using the dot product of their direction vectors. The[block 1] □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

After forward 17/32 · active block 1 · local policy time 0.0625

[block 0] \nTo find the angle between the lines, we first identify the direction vectors of the lines. The direction vectors are obtained by equating the variables and taking the coefficients of the given equations.\n\nThe first line is:\n\[2x = 3y = -z\]\nThis gives us the direction vector (2, 3, -1).\n\nThe second line is:\n\[6x = -y = -4z\]\nThis gives us the direction vector (6, -1, -4).\n\nThe angle between the lines can be found using the dot product of their direction vectors. The[block 1] □□+ of□+ vectors□□□□□+1□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□+<|endoftext|>

After forward 32/32 · active block 1 · local policy time — · DETERMINISTIC FINALIZATION

[block 0] \nTo find the angle between the lines, we first identify the direction vectors of the lines. The direction   
vectors are obtained by equating the variables and taking the coefficients of the given equations.\n\nThe first line   
is:\n\[2x = 3y = -z\]\nThis gives us the direction vector (2, 3, -1).\n\nThe second line is:\n\[6x = -y = -4z\]\nThis   
gives us the direction vector (6, -1, -4).\n\nThe angle between the lines can be found using the dot product of their   
direction vectors. The[block 1] dot product of two vectors, (2, 3, -1) and (6, -1, -4), is:\n\[(2)(6) + (3)(-1) +   
(-1)(-4) = 12 - 3 + 4 = 13\]\n\nSince the dot product is not+ zero, the+ direction vectors are not+ col+linear,+ meaning   
they are perpendicular to each\* other.+ Therefore, the angle between the two lines is 90   
degrees.\n</reasoning>\n\n<answer>\n\boxed{90^\circ}\n</answer><|eot\_id|><|endoftext|>

Figure 27 Verifier-accepted blockwise answer with flawed intermediate reasoning. The task asks for the angle between the lines 2x = 3y = −z and 6x = −y = −4z. The final answer of $9 0 ^ { \circ }$ is accepted, but the displayed derivation uses incorrect direction vectors and an invalid inference from their dot product. We retain the example to distinguish final-answer verification from verification of the reasoning trace.

![](images/2081de603021ad64d29c54bda77434984c7305ca41b7822a20d6937ccfc6083d.jpg)  
Figure 28 Unsuccessful full-canvas revision on GSM8K. Josh buys a house for \$80,000, spends \$50,000 on repairs, and increases its value by 150%. Despite repeated visibility changes, the response ends with an incorrect calculation and a generated answer of \$40,000. The trajectory demonstrates that remasking is an opportunity to revise, not a guarantee of correct reasoning or a correct final answer.

## C Experimental and implementation details

This section specifies the architecture, numerical implementation, and training procedure for the mathematicalreasoning checkpoints used in the main comparison. We distinguish the full-canvas policy, $\mathrm { T I P S } _ { \mathrm { f u l l } }$ , from the two-block policy, $\mathrm { T I P S _ { b l o c k } }$ , while identifying the supervised initialization shared by the two variants. Evaluation populations, compute accounting, and statistical summaries are specified in appendix D.

## C.1 Frozen decoder and policy architecture

We use the frozen LLaDA-8B-Instruct decoder (Nie et al., 2025), identified as GSAI-ML/LLaDA-8B-Instruct. Only the visibility controller is optimized: decoder parameters are excluded from the optimizer, and the recorded decoder weights remain unchanged throughout training.

The controller is a two-layer RoPE-DiT (Peebles & Xie, 2023; Su et al., 2024) with hidden width 128, two attention heads per layer, and feed-forward width 512. A linear projection maps each two-dimensional confidence feature to the hidden width. A 128-dimensional sinusoidal time embedding is processed by a two-layer multilayer perceptron with SiLU activation, and a learned two-state embedding distinguishes masked from visible positions. The DiT layers use time- and state-conditioned adaptive layer normalization, bidirectional self-attention over response-position inputs, and GELU feed-forward activations. Rotary position embeddings are applied to attention queries and keys, with base 10,000. A final layer normalization and scalar projection produce one visibility logit per position.

We use zero dropout and no separate conditioning on the absolute decoding budget. The adaptive normalization parameters are initialized to zero, the shared output bias is initialized to zero, and learned state-dependent logit ofsets are initialized to −2 for masked positions and +4 for visible positions. Each final controller has 628,739 trainable parameters; fixed bufers are excluded from this count. The two response blocks of $\mathrm { T I P S _ { b l o c k } }$ share controller weights, so response-block count does not multiply the number of parameters.

Table 30 Configuration of the final mathematical-reasoning policies. The architecture is shared; decoding geometry, local timing, and fitted schedule difer.
<table><tr><td>Setting</td><td> $\mathrm { T I P S } _ { \mathrm { f u l l } }$ </td><td> $\mathrm { T I P S _ { b l o c k } }$ </td></tr><tr><td>Response geometry</td><td>1 × 256</td><td>2 × 128</td></tr><tr><td>DiT layers / hidden width</td><td>2/128</td><td>2/128</td></tr><tr><td>Attention heads / head width</td><td>2/64</td><td>2/64</td></tr><tr><td>Feed-forward width</td><td>512</td><td>512</td></tr><tr><td>Confidence channels</td><td>2</td><td>2</td></tr><tr><td>Time-embedding width</td><td>128</td><td>128</td></tr><tr><td>Dropout</td><td>0</td><td>0</td></tr><tr><td>Trainable policy parameters</td><td>628,739</td><td>628,739</td></tr><tr><td>Training decoder-call budget</td><td>32</td><td>32</td></tr><tr><td>Schedule power  $\alpha = \exp ( \gamma )$ </td><td>1.1565491821</td><td>1.2060408032</td></tr><tr><td>Schedule parameter γ</td><td>0.1454407285</td><td>0.1873429312</td></tr></table>

## C.2 Confidence features and visibility updates

We compute the features in equation (6) from the decoder distribution before applying the visibility transition. At a masked position, the active token is the greedy proposal; at a visible position, it is the token already stored on the canvas. The strongest alternative is obtained from the two highest-probability vocabulary entries: it is the second entry when the active token is first, and the first entry otherwise. Thus, alternative probabilities are not renormalized over a reduced vocabulary.

For numerical stability, feature calculations are performed in float32 with $\epsilon = 1 0 ^ { - 6 }$ . The active probability is clipped to $[ \epsilon , 1 - \epsilon ]$ for its log-odds, whereas both probabilities are clipped only from below by ϵ for the log-ratio. The implementation uses the ordering returned by the vocabulary top-two operation without an additional tie-breaking rule.

All position-wise visibility decisions are applied synchronously. A newly revealed position receives its proposal from the pre-transition decoder evaluation; a visible position selected to remain visible retains its existing token, even when the decoder now prefers another token. At deterministic finalization, one fresh decoder evaluation fills the remaining masks greedily and leaves already visible tokens unchanged. This finalization does not invoke the policy or contribute an action likelihood.

Block-local execution. Let B be the number of response blocks and $H = T / B$ the number of decoder calls allocated to each block. Each block receives $H - 1$ learned transitions and one deterministic finalization. The prompt and finalized prefix remain immutable, while future blocks remain masked until they become active. For local learned step $j \in \{ 0 , \ldots , H - 2 \}$ , both variants target occupancy $( ( j + 1 ) / H ) ^ { \alpha }$ within the active block. The saved policies use diferent time inputs: $\mathrm { T I P S } _ { \mathrm { f u l l } }$ receives $j / H$ , whereas ${ \mathrm { T I P S } } _ { \mathrm { b l o c k } }$ receives $( j + 1 ) / H$ with the clock reset at each block. We preserve these checkpoint-specific conventions during evaluation.

## C.3 Numerical calibration of the visibility logits

For the set I of positions eligible for scheduling, define

$$
f ( \delta ) = \frac { 1 } { | { \cal Z } | } \sum _ { i \in { \cal Z } } \sigma ( b _ { i } + \delta ) - \bar { \kappa } , \qquad \bar { \kappa } = \mathrm { c l i p } ( \kappa , 1 0 ^ { - 6 } , 1 - 1 0 ^ { - 6 } ) .\tag{14}
$$

The scalar shift is initialized as $\begin{array} { r } { \delta ^ { ( 0 ) } = \log \mathrm { i t } ( \bar { \kappa } ) - | \mathcal { T } | ^ { - 1 } \sum _ { i \in \mathcal { T } } b _ { i } } \end{array}$ . We first perform ten float32 Newton iterations,

$$
\begin{array} { l } { { \displaystyle \delta ^ { ( r + 1 ) } = \delta ^ { ( r ) } - \mathrm { c l i p } \left( \frac { f ( \delta ^ { ( r ) } ) } { \operatorname* { m a x } \{ f ^ { \prime } ( \delta ^ { ( r ) } ) , 1 0 ^ { - 8 } \} } , - 1 0 , 1 0 \right) , } } \\ { { \displaystyle f ^ { \prime } ( \delta ) = \frac { 1 } { | { \cal Z } | } \sum _ { i \in { \cal Z } } \sigma ( b _ { i } + \delta ) \big [ 1 - \sigma ( b _ { i } + \delta ) \big ] . } } \end{array}\tag{15}
$$

For rows whose resulting absolute residual is nonfinite or at least $1 0 ^ { - 6 }$ , we select a safeguarded ten-iteration solution. Its initial bracket is $\begin{array} { r } { [ \mathrm { l o g i t } ( \bar { \kappa } ) - \mathrm { m a x } _ { i \in \mathcal { T } } b _ { i } , \mathrm { l o g i t } ( \bar { \kappa } ) - \mathrm { m i n } _ { i \in \mathcal { T } } b _ { i } ] } \end{array}$ . Each safeguarded iteration updates the bracket and chooses the lower-residual candidate between a clipped Newton proposal and the bracket midpoint. Both solution paths remain in the automatic-diferentiation graph, so gradients pass through the selected calibration computation during training. The detached teacher solve used for supervised targets is a separate procedure.

## C.4 Supervised warmup

Reference responses. The training input contains 12,776 problems: 6,280 from GSM8K and 6,496 from MATH. We use each problem’s original dataset answer or solution as its reference response. Each reference view ends at the first EOS token, or has EOS appended when none is present, and is then capped at 256 tokens. All training problems are retained, including the 2,113 whose reference view exceeds the canvas length before truncation. Reference inclusion is not conditioned on acceptance by the generated-answer verifier. Padded positions are marked invalid for the direct supervised loss.

Teacher and loss. The teacher uses the reference-token utility $u _ { t , i }$ in equation (10). The masked-position margin is computed by subtracting the strongest non-reference logit from the reference-token logit, which preserves the sign of the margin when the reference is the decoder’s top prediction. In the implementation, validposition utilities are transformed to $( u _ { t , i } + 1 ) / 2$ and multiplied by 10 before applying the occupancy-matching ofset. Consequently, the sharpness in the raw-utility convention of the main text is $\beta = 5$

Binary cross-entropy is averaged over reference-valid positions within each state and then equally over the states in the minibatch. Reference tokens and reference margins are used only to construct supervision; the student receives the decoder-derived confidence features, mask states, and time inputs.

Initial and refreshed state banks. Warmup has two successive stages, both using full-canvas decoding at $T _ { \mathrm { t r a i n } } = 3 2$ . The first stage trains from a fresh initialization on an initial bank of 396,056 states. We then refresh the bank with 99,014 additional states and continue from the first-stage policy using the combined 495,070-state bank. The optimizer, learning-rate schedule, and local update counter restart for the second stage. Each stage visits its bank once, with seed-dependent packet order and within-packet state order. Thus, the two stages together comprise 7,304 optimizer updates.

Table 31 Training stages per seed. State presentations count supervised examples processed, not distinct training problems.
<table><tr><td>Stage</td><td>Updates</td><td>State presentations</td><td>Initialization</td></tr><tr><td>Initial supervised warmup</td><td>3,230</td><td>396,056</td><td>Fresh policy</td></tr><tr><td>Refreshed supervised warmup</td><td>4,074</td><td>495,070</td><td>Initial warmup endpoint</td></tr><tr><td>Full-canvas RLVR</td><td>935</td><td></td><td>Refreshed warmup endpoint</td></tr><tr><td>Two-block RLVR</td><td>935</td><td></td><td>Same warmup endpoint</td></tr></table>

## C.5 RLVR and optimization settings

For each training seed, the final supervised policy initializes separate full-canvas and two-block RLVR runs. Each run uses a global batch of 128 trajectories, formed from 16 prompt groups with eight trajectories per prompt, and a decoding horizon of 32 calls. Rollouts and likelihood replay use microbatches of 16 trajectories. Terminal reward is binary final-answer correctness, and the advantage for a trajectory is its reward minus the mean reward of its eight-trajectory group.

At each learned step, the action log-probability sums Bernoulli log-probabilities over the positions on which the policy acts. The new-to-old joint action-likelihood ratio is clipped to [0.5, 1.5] in the GRPO surrogate, which is averaged over learned steps and trajectories. Finalization contributes neither an action nor a likelihood term. Hence, a training rollout contains 31 learned transitions for the full-canvas policy and 30 for the two-block policy. Both final mathematical-reasoning branches complete 935 nominal updates and 935 optimizer steps. Each branch therefore uses 935 × 128 × 32 = 3,829,760 example-level decoder evaluations for training rollouts; this count excludes bootstrap training and supervised state collection.

Table 32 Optimization settings.
<table><tr><td>Setting</td><td>Supervised warmup</td><td>RLVR</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Adam moments</td><td>(0.9,0.999)</td><td>(0.9,0.99)</td></tr><tr><td>Adam numerical constant</td><td>10-8</td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td>0</td><td>0.1; 0 for bias/norm</td></tr><tr><td>Learning-rate warmup</td><td>100 updates</td><td>100 updates</td></tr><tr><td>Subsequent schedule</td><td>Cosine decay</td><td>Cosine decay</td></tr><tr><td>Global gradient-norm clipping</td><td>0.2</td><td>0.2</td></tr><tr><td>Trajectories per prompt</td><td></td><td>8</td></tr><tr><td>Global trajectory batch</td><td></td><td>128</td></tr><tr><td>Likelihood-ratio clipping</td><td></td><td>[0.5, 1.5]</td></tr></table>

We evaluate the fixed 935-update endpoints for training seeds 123, 124, and 125. The checkpoint indices 1, 2, and 3 correspond to these seeds, respectively, for both geometries. Full-canvas and block policies with the same index share their supervised warmup ancestral weights, but their RLVR weights are trained separately.

## C.6 Baseline configurations

The mathematical-reasoning comparison contains the configurations listed in table 33. Multiple configurations of one method difer in response geometry or in the saved training or decoding variant; they are therefore retained as separate comparisons.

Table 33 Baseline configurations. A geometry denotes number of blocks × tokens per block. Separate entries within a row are distinct saved configurations. The 4 × 32 UPO configuration uses a 128-token response and is recorded only at T = 128.
<table><tr><td>Method</td><td>Configurations</td></tr><tr><td>Random / uniform</td><td>1 × 256</td></tr><tr><td>High confidence</td><td>1 × 256, fitted schedule; 1 × 256, linear schedule</td></tr><tr><td>Probability margin</td><td>1 × 256</td></tr><tr><td>Predictive entropy</td><td>1 × 256</td></tr><tr><td>Fast-dLLM</td><td>8 × 32, mean-adjusted as described in section 4.1</td></tr><tr><td>Non-adjusted Fast-dLLM</td><td>1 × 256; 8 × 32</td></tr><tr><td>EB-Sampler</td><td>1 × 256</td></tr><tr><td>UNCODE / PC-Sampler</td><td>1 × 256</td></tr><tr><td>KLASS</td><td>1 × 256; 4 × 64</td></tr><tr><td>DAPD</td><td>1 × 256</td></tr><tr><td>Jazbec et al. (2026)</td><td>1 × 256, expert-steering; 1 × 256, vanilla</td></tr><tr><td>UPO</td><td>1 × 256; 8 × 32, block-trained; 4 × 32, 128-token response</td></tr><tr><td>SAS</td><td>1 × 256; 8 × 32</td></tr><tr><td>ReMDM</td><td>1 × 256</td></tr><tr><td>RCR</td><td>1 × 256</td></tr><tr><td>WINO</td><td>1 × 256; 2 × 128</td></tr><tr><td>CoRe</td><td>1 × 256</td></tr></table>

## D Evaluation protocols

We first specify the mathematical-reasoning evaluation of the main TIPS policies, including its validation/test separation and aggregation over trained policies. We then describe the populations, interventions, and uncertainty measures used in the supplementary experiments, whose statistical units difer from those of the main result.

## D.1 Datasets, validation split, and prompts

The mathematical-reasoning test population is the union of all 1,319 GSM8K test problems and all 500 MATH-500 problems (Cobbe et al., 2021; Hendrycks et al., 2021; Lightman et al., 2024). The 1,024-problem validation set contains 512 GSM8K and 512 MATH problems drawn from the training sources and excluded from the gradient-training input. The recorded training, validation, and test identifier sets are pairwise disjoint.

Table 34 Problem populations for the main mathematical-reasoning training and evaluation. The validation population is also used, without correctness labels, to fit the schedule.
<table><tr><td>Population</td><td>GSM8K</td><td>MATH / MATH-500</td><td></td><td>Total</td></tr><tr><td>Gradient-training input</td><td>6,280</td><td></td><td>6,496</td><td>12,776</td></tr><tr><td>Validation and schedule fitting</td><td>512</td><td></td><td>512</td><td>1,024</td></tr><tr><td>Final mathematical-reasoning test</td><td>1,319</td><td></td><td>500</td><td>1,819</td></tr></table>

The same validation prompts support both label-free schedule fitting and design ablations. Their equal task mixture also difers from the 1,319:500 test mixture, so their absolute accuracies are not directly comparable to main-test accuracies.

Prompt construction. The main TIPS test prompts are zero-shot. We place the task instruction and question in one user message, separated by two newline characters, apply the tokenizer’s chat template with add\_generation\_prompt=True, and append the literal prefix <reasoning>. The GSM8K instruction is:

You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer

in a \boxed{}.   
Respond in the following format:   
<reasoning>   
Your reasoning here   
</reasoning>   
<answer>   
\boxed{...}   
</answer>

The saved MATH-500 instruction has the same text, with a literal double-quotation mark and a newline after the closing </answer> tag. We retain this exact task-specific prompt rather than normalizing it after evaluation. The full tokenized prompt is preserved. The response canvas is a separate fixed-length sequence of 256 tokens.

## D.2 Decoder-call budgets and actual consumption

The main budgets are $T \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . For TIPS, these are exact horizons: each example consumes all T calls, including deterministic finalization, even if an intermediate canvas is already fully visible. Full-canvas decoding uses T − 1 learned transitions and one finalization; two-block decoding uses $T - 2$ learned transitions and two finalizations. For example, the two-block policy still makes two learned transitions at $T = 4$

The horizontal coordinate in the main comparison denotes a decoder-call cap, not necessarily the number of calls consumed by an adaptive baseline. The displayed 8 × 32 Fast-dLLM configuration uses the mean-adjusted confidence threshold to match budget in expectation (refer to appendix A.2). Other naturally terminating decoding can consume fewer calls than its cap. We therefore distinguish capped accuracy comparisons from measurements plotted against mean actual decoder calls. Controller evaluations and numerical calibration do not add decoder calls, although their runtime is not represented by this metric. Equal decoder-call counts alone do not imply equal wall-clock time or floating-point operation counts across response geometries.

## D.3 Output processing and answer verification

For the main TIPS evaluation, we decode the complete 256-token response canvas with skip\_special\_- tokens=True, without an additional stop-string truncation. GSM8K reference targets are converted to finite numerical values, using the final #### target when a full dataset answer is supplied. MATH targets are normalized with the mathematical-answer parser, which extracts an answer from boxed, framed, or answer-tagged reference text when applicable. Generated responses are evaluated by the corresponding task-specific extraction and correctness functions. Each verification has a five-second timeout. Missing or nonfinite predictions, answer mismatches, timeouts, and verifier exceptions count as incorrect; they are not removed from the evaluation denominator.

## D.4 Randomness, aggregation, and uncertainty

For each TIPS geometry and budget, the main evaluation uses three trained policies, with seeds 123, 124, and 125, and three inference streams per policy, with identifiers 42, 43, and 44. Token proposals are greedy, whereas visibility actions are independent Bernoulli draws conditional on the policy probabilities.

Let $c _ { s r i } ( T ) \in \{ 0 , 1 \}$ denote correctness on problem i for training seed s and inference stream r, and let $N = 1 { , } 8 1 9$ . We first compute each trained policy’s accuracy, averaging its three inference draws, and then average the three policy accuracies:

$$
A _ { s } ( T ) = \frac { 1 } { 3 N } \sum _ { r = 1 } ^ { 3 } \sum _ { i = 1 } ^ { N } c _ { s r i } ( T ) , \qquad \bar { A } ( T ) = \frac { 1 } { 3 } \sum _ { s = 1 } ^ { 3 } A _ { s } ( T ) .\tag{16}
$$

This is a problem-level micro-average over GSM8K and MATH-500, not an equally weighted average of their dataset-level accuracies. For the main table and shaded bands, we report the standard error across the three

trained-policy means:

$$
\mathrm { S E } ( T ) = \frac { 1 } { \sqrt { 3 } } \left[ \frac { 1 } { 2 } \sum _ { s = 1 } ^ { 3 } \bigl ( A _ { s } ( T ) - \bar { A } ( T ) \bigr ) ^ { 2 } \right] ^ { 1 / 2 } .\tag{17}
$$