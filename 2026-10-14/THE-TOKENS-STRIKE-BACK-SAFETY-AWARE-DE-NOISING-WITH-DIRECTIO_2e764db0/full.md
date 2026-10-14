# THE TOKENS STRIKE BACK: SAFETY-AWARE DE-NOISING WITH DIRECTIONAL EMBEDDING-BASED RE-VISION

Amman Yusuf<sup>1</sup>, Jiwoo Shin<sup>2</sup>, Ruizhi Deng<sup>3</sup>, Heitor Rodrigues Guimaraes ˜ <sup>3</sup>, Nikita Dvornik<sup>3</sup>, Mijung Park<sup>1</sup>

<sup>1</sup>Department of Computer Science, University of British Columbia, Canada <sup>2</sup>Korea Advanced Institute of Science and Technology (KAIST), South Korea <sup>3</sup>RBC Borealis, Canada   
ammany01@cs.ubc.ca

## ABSTRACT

The Safety-Aware Denoiser (SAD) is a recent, training-free, safety-guidance framework for text diffusion models. During inference, SAD steers text samples toward safe regions of the token probability space. In this paper, we propose SADDER (Safety-Aware Denoising with Directional Embedding-based Revision), which improves the safety and quality trade-offs of SAD by exploiting a safetyguidance signal in the embedding (or activation) space for each token position, then updating the token positions using the SAD update rule multiplied by the guidance signal. This multiplication acts as a gating mechanism: tokens with a high unsafe signal are more likely to be modified during inference following the SAD update rule. However, such an update does not automatically preserve the conditional mean, i.e., the denoiser of a fixed target distribution. We therefore theoretically derive new position-wise target distributions for which the modified denoiser is exact. Furthermore, we derive conditions under which the positionwise target distributions match a global target distribution. We test our SADDER method on popular Masked Diffusion Language Models (MDLMs) and Uniform-State Diffusion Models (USDMs) across unsafe continuation and jailbreak settings, and show that SADDER achieves a better balance between safety, generation quality, and inference cost than existing methods.

## 1 INTRODUCTION

Diffusion models have achieved state-of-the-art performance across a wide range of domains, including images (Rombach et al., 2022), speech (Popov et al., 2021), and protein structure generation (Wu et al., 2024). In language modeling, however, autoregressive large language models (AR-LLMs) remain the dominant approach, demonstrating strong capabilities in language generation and understanding, mathematical reasoning (Yang et al., 2024), and coding (Li et al., 2023). Despite their success, AR models generate text one token at a time, which inherently limits inference parallelism and makes revising previously generated content less natural.

Recent advances in Text Diffusion Models (TDMs) offer an alternative paradigm, in which text is generated through an iterative denoising process that can update multiple token positions and progressively refine the generated sequence (Sahoo et al., 2024; Nie et al., 2025). Moreover, at comparable model scales, recent TDMs have also shown competitive performance with AR-LLMs across benchmarks; For instance, Stable-DiffCoder-8B (Fan et al., 2026) vs. Seed-Coder-8B (ByteDance Seed et al., 2025) on code generation benchmarks, and LLaDA-8B (Nie et al., 2025) vs. LLaMA3- 8B (Grattafiori et al., 2024) on mathematical reasoning tasks.

Despite these advantages, the safety of TDMs remains relatively underexplored compared to AR-LLMs, particularly with respect to toxic content generation and vulnerability to jailbreak attacks. Traditional safety mechanisms rely on fine-tuning and alignment (Dai et al., 2024), post-hoc filtering and rejection sampling (Naveed et al., 2025), or inference-time methods that modify the decoding strategy during next-token prediction (Xu et al., 2024). However, these approaches either require additional training or assume AR decoding and therefore do not directly generalize to TDMs.

Recent work has begun exploring inference-time mechanisms specifically designed for TDMs, including particle-based steering (Singhal et al., 2025), classifier-based guidance (Schiff et al., 2025), and safety-oriented remasking approaches (Lee & Han, 2026). In particular, the Safety-Aware Denoiser (SAD) (Yusuf et al., 2026) provides a training-free framework to steer the denoising trajectory towards a safe region in the data distribution using only a small set of $5 0 0 \sim 1 0 0 0$ unsafe reference samples. Nevertheless, SAD operates in token probability space, leaving open whether semantically richer model representations could enable more effective, position-specific safety guidance.

In this work, we propose SADDER (Safety-Aware Denoising with Directional Embedding-based Revision), a safety-guidance method that augments SAD with position-wise signals extracted from the model’s embedding space. At each denoising step, we first identify potentially unsafe token positions based on a safety-guidance signal in the embedding (or activation) space. We then use this information in a second step to modulate the SAD update, enabling more precise, positionby-position interventions along the sampling trajectory. By selectively modifying fewer positions, SADDER achieves a better balance between safety, generation quality, and inference cost than existing methods, when tested on popular Masked Diffusion Language Models (MDLMs) and Uniform-State Diffusion Models (USDMs) across unsafe continuation and jailbreak settings.

## 2 BACKGROUND

We first describe two ingredients our method builds on: masked and uniform-state TDMs in Section 2.1, whose samplers differ in whether a committed token can still be changed in the subsequent denoising steps; and SAD in Section 2.2, the inference-time guidance framework that our method extends.

## 2.1 TEXT DIFFUSION MODELS

A TDM corrupts a clean token sequence $\pmb { x } _ { 0 } ^ { 1 : L }$ of length L with a fixed forward process and learns to reverse it. Tokens come from a vocabulary V of size $K$ and are represented as one-hot vectors, so a distribution over tokens is a vector in the simplex $\Delta ^ { K } . \cot ( { \pmb x } ^ { \top } | \ { \pmb \pi } )$ denotes the categorical distribution with probability vector π.

We write $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { t } ^ { l } }$ for the token at position l and normalized diffusion time $t \in [ 0 , 1 ]$ . Let $\alpha _ { t } \in [ 0 , 1 ]$ denote a noise schedule that decreases strictly in t with $\alpha _ { 0 } \approx 1$ and $\alpha _ { 1 } \approx 0$ . We discretize the diffusion time interval using a decreasing grid $1 = t _ { 0 } > t _ { 1 } > \cdot \cdot \cdot > t _ { T } = 0$ where $T$ is the number of reverse sampling steps. At iteration $j \in \{ 0 , \ldots , T - 1 \}$ , the sampler transitions from $\pmb { x } _ { t _ { j } } ^ { 1 : L }$ to $\pmb { x } _ { t _ { j + 1 } } ^ { 1 : L }$

Masked diffusion language models (MDLMs). MDLMs (Sahoo et al., 2024; Nie et al., 2025; Ye et al., 2025) add a mask token with one-hot vector M, so states live in $\{ 0 , \dot { 1 } \} ^ { K + 1 }$ ; the network below never predicts the mask, so its outputs stay in $\Delta ^ { K }$ . In the forward process, each position transitions to the mask independently and, once masked, stays masked:

$$
q _ { t } ( \pmb { x } _ { t } ^ { l } \mid \pmb { x } _ { 0 } ^ { l } ) = \mathrm { C a t } \big ( \pmb { x } _ { t } ^ { l } \mid \alpha _ { t } \pmb { x } _ { 0 } ^ { l } + ( 1 - \alpha _ { t } ) \mathbb { M } \big ) ,\tag{1}
$$

$\mathrm { i . e . }$ , the token is kept with probability $\alpha _ { t }$ and masked with probability $1 - \alpha _ { t }$ . The reverse process runs from the all-mask sequence back to $t = 0 ;$ its transitions are the forward posterior $\dot { \mathbf { \eta } } _ { q } ( \mathbf { \boldsymbol { x } } _ { s } ^ { l } \ )$ $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { t } ^ { l } , \mathbf { \boldsymbol { x } } _ { 0 } ^ { l } } )$ with the unknown clean token replaced by the prediction of a denoising network $\scriptstyle { \mathbf { \mathit { x } } } \mathbf { \mathit { \theta } }$ , which reads the whole noised sequence and outputs a distribution over the clean token at every position,

$$
\begin{array} { r } { \pmb { x _ { \theta } } : ( \mathcal { V } \cup \{ \mathbb { M } \} ) ^ { L } \times [ 0 , 1 ] \longrightarrow ( \Delta ^ { K } ) ^ { L } , } \end{array}\tag{2}
$$

with $\pmb { x } _ { \pmb { \theta } , l } ( \pmb { x } _ { t } ^ { 1 : L } , t ) \in \Delta ^ { K }$ the output at position l. Given $\pmb { x } _ { t } ^ { 1 : L }$ , the transition to an earlier level $s < t$ factorizes across positions, with

$$
\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { s } ^ { l } \mid \pmb { x } _ { t } ^ { 1 : L } ) = \left\{ \begin{array} { l l } { \delta _ { \pmb { x } _ { t } ^ { l } } ( \pmb { x } _ { s } ^ { l } ) , } & { \pmb { x } _ { t } ^ { l } \neq \mathbb { M } , } \\ { \mathrm { C a t } \Big ( \pmb { x } _ { s } ^ { l } \Big | \frac { ( 1 - \alpha _ { s } ) \mathbb { M } + ( \alpha _ { s } - \alpha _ { t } ) \pmb { x } _ { \theta , l } ( \pmb { x } _ { t } ^ { 1 : L } , t ) } { 1 - \alpha _ { t } } \Big ) , } & { \pmb { x } _ { t } ^ { l } = \mathbb { M } . } \end{array} \right. } \end{array}\tag{3}
$$

Under equation 3, a revealed position is copied, and a masked position is revealed at random with probability $( \alpha _ { s } - \alpha _ { t } ) / ( 1 - \alpha _ { t } )$ , its token drawn from $\mathbf { \Delta } _ { \mathbf { x } _ { \theta , l } }$ . The samplers of LLaDA and Dream differ from equation 3 only in how they choose which masked positions to reveal (Nie et al., 2025; Ye et al., 2025). Remasking samplers (Wang et al., 2025) additionally return revealed positions to m so that they are predicted again.

Uniform-state diffusion models (USDMs). USDMs (Sahoo et al., 2025; 2026; DiffusionGemma Team, 2026) go a step further: they have no mask token, and any token can change at any step, which Sahoo et al. (2026) credit with self-correction and better guided generation. In the forward process, a token is kept with probability $\alpha _ { t }$ and otherwise replaced by one drawn uniformly from the vocabulary,

$$
\begin{array} { r } { q _ { t } ( \pmb { x } _ { t } ^ { l } \mid \pmb { x } _ { 0 } ^ { l } ) = \mathrm { C a t } \big ( \pmb { x } _ { t } ^ { l } \mid \alpha _ { t } \pmb { x } _ { 0 } ^ { l } + ( 1 - \alpha _ { t } ) \frac { 1 } { K } \pmb { 1 } \big ) , } \end{array}\tag{4}
$$

with 1 the all-ones vector. The reverse kernel is again the forward posterior with $\mathbf { \Delta } _ { \mathbf { { x } } _ { 0 } ^ { l } }$ replaced by the prediction; with $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ , it is given by

$$
\begin{array} { r l } & { q _ { s | t } ( \boldsymbol x _ { s } ^ { l } | \boldsymbol x _ { t } ^ { l } , \boldsymbol x _ { \theta } ^ { l } ) } \\ & { = \mathrm { C a t } \left( \boldsymbol x _ { s } ^ { l } | \frac { K \alpha _ { t } \boldsymbol x _ { t } ^ { l } \odot \boldsymbol x _ { \theta } ^ { l } + ( \alpha _ { t | s } - \alpha _ { t } ) \boldsymbol x _ { t } ^ { l } } { K \alpha _ { t } \langle \boldsymbol x _ { t } ^ { l } , \boldsymbol x _ { \theta } ^ { l } \rangle + 1 - \alpha _ { t } } + \frac { ( \alpha _ { s } - \alpha _ { t } ) \boldsymbol x _ { \theta } ^ { l } + ( 1 - \alpha _ { t | s } ) ( 1 - \alpha _ { s } ) \mathbf 1 / K } { K \alpha _ { t } \langle \boldsymbol x _ { t } ^ { l } , \boldsymbol x _ { \theta } ^ { l } \rangle + 1 - \alpha _ { t } } \right) , } \end{array}\tag{5}
$$

where ⊙ denotes the Hadamard product between two vectors and $\pmb { x } _ { \pmb { \theta } } ^ { l } = \pmb { x } _ { \pmb { \theta } , l } ( \pmb { x } _ { t } ^ { 1 : L } , t )$ is given by the pretrained model. Every entry of this vector can be positive, which is what lets any token change at any step.

## 2.2 THE SAFETY-AWARE DENOISER

The safety-aware denoiser (SAD) (Yusuf et al., 2026) is a training-free guidance method that steers a pretrained TDM away from a negation set of sequences that should not be generated. It adapts the safe denoiser of Kim et al. (2025), later extended to flow matching (Kim et al., 2026), by replacing the Gaussian corruption kernel with the categorical kernels equation 1 and equation 4.

Safe Denoiser. Assume the training data D split into safe and unsafe sets, $\mathcal { D } = \mathcal { D } _ { \mathrm { s a f e } } \cup \mathcal { D } _ { \mathrm { u n s a f e } }$ with $| { \mathcal D } _ { \mathrm { s a f e } } | \gg | { \mathcal D } _ { \mathrm { u n s a f e } } |$ . Restricting $p _ { \mathrm { d a t a } }$ to each set gives $p _ { \mathrm { s a f e } }$ and $p _ { \mathrm { u n s a f e } } .$ , and running the forward process from p<sub>data</sub>, $p _ { \mathrm { s a f e } } .$ , and $p _ { \mathrm { u n s a f e } }$ defines the data, safe, and unsafe denoisers $\mathbb { E } _ { \mathcal { D } } [ \pmb { x } _ { 0 } \mid \pmb { x } _ { t } ]$ $\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f c } } } [ \pmb { x } _ { 0 } \mid \pmb { x } _ { t } ]$ , and $\mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f c } } } [ \pmb { x } _ { 0 } \ | \ \pmb { x } _ { t } ]$ , which are related as follows.

Theorem 2.1 (Theorem 3.2 in Kim et al. (2025)). There exists a nonnegative weight $\beta ^ { * } ( { \pmb x } _ { t } )$ , monotone in the posterior likelihood that $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ originates from the unsafe set, such that

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s q c } } } [ { \pmb x } _ { 0 } \ | \ { \pmb x } _ { t } ] = \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } \ | \ { \pmb x } _ { t } ] + \beta ^ { * } ( { \pmb x } _ { t } ) \left( \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } \ | \ { \pmb x } _ { t } ] - \mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s i c } } } [ { \pmb x } _ { 0 } \ | \ { \pmb x } _ { t } ] \right) . } \end{array}\tag{6}
$$

Thm. 2.1 states that by subtracting “unsafe” components from the data denoiser, with $\beta ^ { * }$ adapting to how unsafe the current state is, we can provably generate samples from the safe denoiser. A pretrained model provides the data denoiser, so only the unsafe denoiser and $\beta ^ { * }$ remain.

SAD. For a TDM, the data denoiser is simply the pre-trained model’s output, $\mathbb { E } _ { \mathcal { D } } [ \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } ] =$ $\boldsymbol { x } _ { \boldsymbol { \theta } , l } ( \boldsymbol { x } _ { t } ^ { 1 : L } , t )$ , since the mean of a categorical variable is its probability vector; the relation is exact for the ideal denoiser, which $\scriptstyle { \mathbf { \mathcal { X } } } \theta$ approximates. The unsafe denoiser is approximated by Monte Carlo over a negation set, $\begin{array} { r } { \mathcal { D } _ { \mathrm { u n s a f e } } = \{ \pmb { x } _ { 0 } ^ { ( n ) } \} _ { n = 1 } ^ { N } \colon \hat { \mathbb { E } } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ \pmb { x } _ { 0 } \mid \pmb { x } _ { t } ] = \sum _ { n = 1 } ^ { N } \pmb { x } _ { 0 } ^ { ( n ) } \frac { q _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ^ { ( n ) } ) } { \sum _ { m = 1 } ^ { N } q _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ^ { ( m ) } ) } , } \end{array}$ where $\begin{array} { r } { q _ { t } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } ^ { ( n ) } ) = \prod _ { l = 1 } ^ { L } q _ { t } ( \pmb { x } _ { t } ^ { l } \mid \pmb { x } _ { 0 } ^ { ( n ) , l } ) } \end{array}$ is the sequence kernel from equation 1 or equation $4 , \mathrm { { a } }$ product because positions are noised independently. The precise definition of the weight is $\beta ^ { * } ( { \pmb x } _ { t } ) =$ $Z _ { \mathrm { u n s a f e } } p _ { \mathrm { u n s a f e } , t } ( { \pmb x } _ { t } ) / \big ( Z _ { \mathrm { s a f e } } p _ { \mathrm { s a f e } , t } ( { \pmb x } _ { t } ) \big )$  (Kim et al., 2025, Appendix A), where $p _ { \mathrm { s a f e } , t }$ and $p _ { \mathrm { u n s a f e } , t }$ are the marginal distributions of the forward process at level t started from the safe and the unsafe distribution, and $Z _ { \mathrm { s a f e } }$ and $Z _ { \mathrm { u n s a f e } }$ are the normalizing constants of the two restricted densities.

Unfortunately, directly computing $\beta ^ { * } ( { \pmb x } _ { t } )$ is intractable: the denominator requires $q _ { t } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } )$ over all safe data, roughly most of the training set, at every sampling step. SAD approximates

$$
\beta ^ { * } ( \pmb { x } _ { t } ) \approx \eta \beta ( \pmb { x } _ { t } ) ,
$$

using an unbiased estimate of $\beta ( { \pmb x } _ { t } )$ and a hyperparameter η to replace the remaining terms of $\beta ^ { * }$

$$
\beta ( \pmb { x } _ { t } ) = \int p _ { \mathrm { u n s a f e } } ( \pmb { x } _ { 0 } ) q _ { t } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 } \approx \frac { 1 } { N } \sum _ { n = 1 } ^ { N } q _ { t } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } ^ { ( n ) } ) .\tag{7}
$$

Because SAD approximates $\beta ^ { * }$ this way, the SAD implementation offers no theoretical safety guarantees on generated data, unlike what the theory in Thm. 2.1 promises.

Prior work shows that such guidance must be strong early in sampling and fade later, or it degrades sample quality (Kim et al., 2025; 2026; Kirchhof et al., 2025); SAD therefore keeps η at a nonzero constant over a window C of critical timesteps at the start of the reverse process and sets $\eta = 0$ outside it.

## 3 METHODS: SADDER

SAD provides safety guidance in the token probability space. We can enhance this guidance by selecting a guidance signal in the embedding (or activation) space for each token position, then modifying the selected token positions using the SAD update rule. In what follows, we introduce SADDER (Safety-Aware Denoising with Directional Embedding-based Revision), consisting of two phases.

## 3.1 FIRST PHASE: SELECTING RISKY POSITIONS.

The goal of the first phase is to create a gating mechanism which tells us which token positions need to be modified. Inspired by (Lee & Han, 2026), we propose accessing TDM’s internal embedding space to obtain this information.

For each sample i, we denote $c _ { i }$ as a given prompt and $\pmb { x } _ { i } ^ { \mathrm { h a r m } } , \pmb { x } _ { i } ^ { \mathrm { s a f e } }$ as two corresponding harmful and safe responses for $\textstyle c _ { i } .$ , respectively. We denote $\ P _ { i } = | c _ { i } |$ as the prompt length and $R _ { i } ^ { \mathrm { h a r m } } \ = \ \operatorname* { m i n } ( | \dot { \bf x } _ { i } ^ { h a r m } | , M )$ the response length after truncation of M tokens. Similarly, $R _ { i } ^ { \mathrm { s a f e } } = \operatorname* { m i n } ( | \mathbf { x } _ { i } ^ { s a f e } | , M )$ . Let $\boldsymbol { h } _ { k , l } ( \cdot ) \in \mathbb { R } ^ { d }$ be the layer-k hidden representation at position l. $h _ { k , l } ^ { ( t ) } ( \cdot )$ denotes the layer-k’s hidden state for position l at timestep t. We extract hidden states and compute the average representations over response tokens:

$$
\bar { h } _ { i , k } ^ { \mathrm { h a r m } } = \frac { 1 } { R _ { i } ^ { \mathrm { h a r m } } } \sum _ { l = P _ { i } + 1 } ^ { P _ { i } + R _ { i } ^ { \mathrm { h a r m } } } h _ { k , l } ( c _ { i } \oplus x _ { i } ^ { \mathrm { h a r m } } ) , \quad \bar { h } _ { i , k } ^ { \mathrm { s a f e } } = \frac { 1 } { R _ { i } ^ { \mathrm { s a f e } } } \sum _ { l = P _ { i } + 1 } ^ { P _ { i } + R _ { i } ^ { \mathrm { s a f e } } } h _ { k , l } ( c _ { i } \oplus x _ { i } ^ { \mathrm { s a f e } } )\tag{8}
$$

We arrive at the (normalized) layer-wise difference in the semantic representations of harmful and safe responses, $\hat { v } _ { k } .$ , by averaging over the examples i:

$$
\widehat { \pmb { v } } _ { k } = \frac { \bar { h } _ { k } ^ { \mathrm { h a r m } } - \bar { h } _ { k } ^ { \mathrm { s a f e } } } { \lvert \lvert \bar { h } _ { k } ^ { \mathrm { h a r m } } - \bar { h } _ { k } ^ { \mathrm { s a f e } } \rvert \rvert _ { 2 } } ,\tag{9}
$$

where $\begin{array} { r } { \bar { \pmb { h } } _ { k } ^ { \mathrm { h a r m } } = \frac { 1 } { N _ { \mathrm { h a r m } } } \sum _ { i } \bar { \pmb { h } } _ { i , k } ^ { \mathrm { h a r m } } } \end{array}$ , and $\begin{array} { r } { \bar { \pmb { h } } _ { k } ^ { \mathrm { s a f e } } = \frac { 1 } { N _ { \mathrm { s a f e } } } \sum _ { i } \bar { \pmb { h } } _ { i , k } ^ { \mathrm { s a f e } } } \end{array}$ . Intuitively, we want to intervene on the token location only when the hidden state at time t is closer to the average harmful representations than to the average safe representations:

$$
\| h _ { k , l } ^ { ( t ) } - \bar { h } _ { k } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } < \| h _ { k , l } ^ { ( t ) } - \bar { h } _ { k } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } .\tag{10}
$$

Using a midpoint $\begin{array} { r } { m _ { k } = \frac { \bar { h } _ { k } ^ { \mathrm { h a r m } } + \bar { h } _ { k } ^ { \mathrm { s a f e } } } { 2 } } \end{array}$ , a simple linear algebraic manipulation gives us the following equivalence:

$$
\| h _ { k , l } ^ { ( t ) } - \bar { h } _ { k } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } < \| h _ { k , l } ^ { ( t ) } - \bar { h } _ { k } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } \iff \left. h _ { k , l } ^ { ( t ) } - m _ { k } , \hat { v } _ { k } \right. > 0\tag{11}
$$

See Section A.2 for the derivation. Unlike (Lee & Han, 2026), we use the centred score to decide whether this particular position will lead to harmful generation:

$$
\begin{array} { r } { s _ { k , l } ^ { ( t ) } : = \left. h _ { k , l } ^ { ( t ) } - m _ { k } , \hat { v } _ { k } \right. . } \end{array}\tag{12}
$$

Algorithm 1 SADDER: selective SAD guidance.   
1: Input: denoiser p<sub>θ</sub>; unsafe dataset $\mathcal { D } _ { \mathrm { u n s a f e } } ; \widehat { v } _ { k } , m _ { k } ;$ initial state $\mathbf { \boldsymbol { x } } _ { t _ { 0 } } \mathrm { ; }$ editable positions $\mathcal { G } ;$ time   
steps $t _ { 0 } > \cdots > \bar { t } _ { T } = 0 ;$ intervention steps $C ;$ strength η.   
2: for $j = 0 , \ldots , T - 1$ do   
3: Set $t \gets t _ { j }$ . Obtain $p _ { \mathcal { D } , t , }$ <sub>·</sub> and $h _ { k , \cdot } ^ { ( t ) }$ from one model evaluation on $\mathbf { \Delta } \mathbf { x } _ { t } .$   
4: Initialize $\widetilde { p } _ { 0 | t , l } \gets p _ { \mathcal { D } , t , l }$ and $a _ { k , l } ^ { ( t ) } \gets 0$ for all l.   
5: $\mathbf { i f } \ j \in C$ then   
6: if MDLMs then $\mathcal { M } _ { t }  \{ l \in \mathcal { G } : \mathbf { x } _ { t } ^ { l } = \mathbb { M } \}$   
7: if USDMs then $\mathcal { M } _ { t } \gets \dot { \{ 1 , \dots , L \} }$   
8: Compute $s _ { k , l } ^ { ( t ) } \gets \langle h _ { k , l } ^ { ( t ) } - m _ { k } , \widehat { v } _ { k } \rangle$ for $l \in \mathcal { M } _ { t }$   
9: Select Risky Positions: $a _ { k , l } ^ { ( t ) } \gets \mathbb { 1 } \{ s _ { k , l } ^ { ( t ) } > 0 \}$ for $l \in \mathcal { M } _ { t }$   
10: if at least one gate equals 1 then   
11: Estimate $\mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ \mathbf { \bar { x } } _ { 0 } ^ { l } | \mathbf { { x } } _ { t } ^ { 1 : L } ]$ and $\beta ( { \pmb x } _ { t } )$ on the current state   
12: SAD On Selected Positions: Compute Sadder Denoiser in equation 14.   
13: end if   
14: end if   
15: $\pmb { x } _ { t _ { j + 1 } } \gets \mathrm { S o l v e r } ( \pmb { x } _ { t } , t _ { j } , t _ { j + 1 }$ , Sadder Denoiser; $\mathcal { G } )$   
16: end for   
17: Return: the generated continuation of ${ \mathbf { } } x _ { t _ { T } } .$

Geometric meaning of equation 12: The set of $h _ { k , l } ^ { ( t ) }$ that satisfies $\left. h _ { k , l } ^ { ( t ) } - m _ { k } , \hat { v } _ { k } \right. \mathrm { ~ = ~ } 0$ is the hyperplane, in which every vector $h _ { k , l } ^ { ( t ) } - m _ { k }$ is orthogonal to the direction $\hat { v } _ { k }$ . Any $h _ { k , l } ^ { ( t ) }$ that satisfies $\left. h _ { k , l } ^ { ( t ) } - m _ { k } , \hat { v } _ { k } \right. > 0$ lies on the side of this hyperplane containing $\bar { h } _ { k } ^ { \mathrm { h a r m } }$ . On the contrary, any $h _ { k , l } ^ { ( t ) }$ that satisfies $\left. h _ { k , l } ^ { ( t ) } - m _ { k } , \hat { v } _ { k } \right. < 0$ lies on the side of this hyperplane containing $\bar { h } _ { k } ^ { \mathrm { s a f e } }$

The tokens that have positive scores, through the following gating mechanism:

$$
a _ { k , l } ^ { ( t ) } \gets \mathbb { 1 } _ { s _ { k , l } ^ { ( t ) } > 0 } ,\tag{13}
$$

where 1 is an indicator function that returns the binary<sup>1</sup> gating values: $a _ { k , l } ^ { ( t ) } \in \{ 0 , 1 \}$

## 3.2 SECOND PHASE: SAD ON SELECTED POSITIONS

In the first phase, we select which position is aligned with the harmful direction in the embedding space. In the second phase, we modify the sampling trajectory of the selected ones using the SAD denoiser formulation, given by

$$
\mathrm { S a d d e r ~ D e n o i s e r } : = \prod _ { l = 1 } ^ { L } \left[ \mathbb { E } _ { \mathcal { D } } [ \mathbf { x } _ { 0 } ^ { l } | \mathbf { x } _ { t } ^ { 1 : L } ] + \eta \beta ( \pmb { x } _ { t } ) a _ { k , l } ^ { ( t ) } \left( \mathbb { E } _ { \mathcal { D } } [ \mathbf { x } _ { 0 } ^ { l } | \mathbf { x } _ { t } ^ { 1 : L } ] - \mathbb { E } _ { \mathcal { D } _ { \operatorname* { m a x } } } [ \mathbf { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ] \right) \right]\tag{14}
$$

where $\beta ( { \pmb x } _ { t } )$ is defined in equation 7 and η is a controllable hyperparameter that changes the strength of the guidance as in SAD. Similar to SAD, the data denoiser at position l is $\mathbb { E } _ { \mathcal { D } } [ \pmb { x } _ { 0 } ^ { \overline { { l } } } | \pmb { x } _ { t } ^ { 1 : L } ]$ and the unsafe denoiser at position l is approximated by

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] \approx \sum _ { { \pmb x } _ { n } \in \mathcal { D } _ { \mathrm { u n s a f e } } } { \pmb x } _ { n } ^ { l } \frac { q _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { n } ^ { l } ) } { \sum _ { { \pmb x } _ { m } \in \mathcal { D } _ { \mathrm { u n s a f e } } } q _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { m } ^ { l } ) } .\tag{15}
$$

Algorithm 1 summarizes our algorithm.

## 3.3 THEORETICAL ANALYSES OF SADDER

In Safe Denoiser, equation 6 provides an identity of conditional expectations under a particular safe/unsafe partition, while, in the implementation of SAD, this quantity is approximated by a welltuned hyperparameter η. A freely chosen position-dependent gate at l in SADDER also changes that identity, as in SAD. In the following theorem, we formally derive under what condition this new target denoiser is a position-wise safe denoiser.

Theorem 3.1 (Characterization of an exact position-wise safe denoiser). Given a data denoiser from a pre-trained diffusion model, an unsafe denoiser in equation 15, and the intervention strength: $\eta \beta ( \pmb { x } _ { t } ) \bar { a } _ { k , l } ^ { ( t ) }$ , the the position-l factor of the SADDER Denoiser in equation 14 is exactly the condi tional mean under the position-wise safe target distribution $\mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) }$

$$
\begin{array} { r } { \left| \mathbb { E } _ { \mathcal { D } _ { \mathrm { s t e f } } ^ { ( l ) } } \left[ { \pmb x } _ { 0 } ^ { l } \mid { \pmb x } _ { t } ^ { 1 : L } \right] = \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] + \eta \beta ( { \pmb x } _ { t } ) a _ { k , l } ^ { ( t ) } \left( \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] - \mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] \right) \right| } \end{array}\tag{16}
$$

if the activation strength is calibrated to the posterior unsafe odds according to

$$
\eta \beta ( \pmb { x } _ { t } ) a _ { k , l } ^ { ( t ) } = \frac { \pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) } { 1 - \pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) } ,\tag{17}
$$

where the posterior probability of the position-l being unsafe is denoted by $\begin{array} { r l } { \pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) } & { { } : = } \end{array}$ $\mathbb { E } _ { \mathcal { D } } \left[ \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : \bar { L } } ) \ | \ \pmb { x } _ { t } ^ { 1 : L } \right]$ , and the unsafety is measured by an oracle function denoted by $\psi _ { l } : \mathcal { V } ^ { L } \longrightarrow$ [0, 1], satisfying $\psi _ { l } ( \\overset { - } { \mathbf { x } } _ { 0 } ^ { 1 : L } ) + ( 1 - \psi _ { l } ( \mathbf { x } _ { 0 } ^ { 1 : L } ) ) = 1$ for any clean sequence $\pmb { x } _ { 0 } ^ { 1 : L }$

See Section A.4 for proof.

Remark 1. Thm. 3.1 characterizes our algorithm, SADDER, in terms of a position-wise safedenoiser guarantee, rather than asserting that the exact membership function $\psi$ is available at inference. However, it identifies the quantity that the activation gate should approximate: the posterior expected unsafe membership of clean sequences compatible with the current noised state. In practice, the centred activation score is a surrogate for this quantity without formal guarantees, like $\beta ^ { * } ( x _ { t } )$ was approximated by $\eta \beta ( \pmb { x } _ { t } )$ in SAD; the resulting SADDER denoiser is also approximate.

The position-specific target distribution ${ p _ { \mathcal { D } _ { \mathrm { s a f } } ^ { ( l ) } } }$ for each position l, stated in Thm. 3.1, may differ for each l: $\mathcal { P } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( 1 ) } } , \cdot \cdot \cdot , p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( L ) } }$ Hence, a natural next question is whether there exists one clean-sequence distribution $p _ { \mathcal { D } _ { \mathrm { e a f e } } ^ { \mathrm { g l o b a l } } } ( \pmb { x } _ { 0 } ^ { 1 : L } )$ , such that simultaneously for every l, it satisfies $\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } } ^ { g l o b a l } ( \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ) =$ $\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } )$ . Thm. A.1 states this formally.

## 4 RELATED WORK

Activation steering and remasking. Contrastive Activation Addition (Rimsky et al., 2024) steers language models using directions extracted from paired examples; ILRR (Avrahami & Nachmani, 2026) guides masked diffusion by aligning activations with a reference sequence. For safety, Diffu-Guard (Li et al., 2025) combines stochastic remasking with block-level audit and repair. Most closely related, Lee & Han (2026) construct paired samples $( x _ { i } , y _ { i } ^ { \mathrm { h a r m } } , y _ { i } ^ { \mathrm { s a f e } } )$ , using harmful model responses to WildJailbreak prompts (Jiang et al., 2024) and paraphrased refusals. They estimate a Contrastive Safety Direction (CSD), $v _ { \ell } .$ , from the difference between the mean harmful and safe response representations. In Phase 1, they modify hidden states at masked and prompt positions during the first fraction $\rho$ of the denoising steps: $\begin{array} { r } { S ( h _ { \ell , j } ^ { ( t ) } ) = h _ { \ell , j } ^ { ( t ) } - \beta \langle h _ { \ell , j } ^ { ( t ) } , \hat { v } _ { \ell } \rangle \hat { v } _ { \ell } } \end{array}$ , where $\hat { v } _ { \ell }$ is the normalized CSD and $\beta$ controls the intervention strength. In Phase 2, they refine the generated sequence by selecting positions according to $m _ { j } ^ { ( k ) } = \mathbb { 1 } \left\{ \langle h _ { \ell , j } ^ { ( k ) } , \hat { v } _ { \ell } \rangle > \theta \right\}$ , then remasking and regenerating those positions for up to K refinement iterations, stopping early if no positions exceed the threshold. SADDER uses a similar contrastive direction, but centers the activation score at the midpoint of the harmful and safe representations. This places the zero-score boundary at their Eu clidean bisector. The score gates SAD’s correction in token probability space rather than directly modifying hidden states; in masked models, it also ranks revealed positions for budgeted remasking within the denoising loop.

Table 1: LLaDA-Base safety, generation quality, and cost. Entries use $A _ { b } ,$ where A is the mean and b is one sample standard deviation across three generation seeds. For percentages, SDs are in percentage points. NFE shows average cost.
<table><tr><td rowspan="2">Method</td><td colspan="5">RTP</td><td colspan="5">BeaverTails</td></tr><tr><td> $\% \downarrow$ </td><td>Unsafe Harmful  $\mathrm { P P L } \downarrow$ </td><td>Benign PPL↓</td><td>Benign BERT↑</td><td>NFE</td><td> $\% \downarrow$ </td><td>Unsafe Harmful  $\mathrm { P P L } \downarrow$ </td><td>Benign  $\mathrm { P P L } \downarrow$ </td><td>Benign BERT ↑</td><td>NFE</td></tr><tr><td>Unmodified</td><td> $3 8 . 6 _ { 1 . 3 }$ </td><td> $2 1 . 1 _ { 0 . 1 5 }$ </td><td> $1 7 . 5 \mathrm { _ { 0 . 3 5 } }$ </td><td> $8 4 . 6 5 _ { 0 . 0 4 }$ </td><td>32.0</td><td> $3 8 . 6 _ { 1 . 5 }$ </td><td> $2 0 . 6 _ { 0 . 6 2 }$ </td><td> $1 7 . 5 _ { 0 . 3 5 }$ </td><td> $8 4 . 6 5 _ { 0 . 0 4 }$ </td><td>32.0</td></tr><tr><td>Full SAD</td><td> $3 2 . 4 1 . 6 $ </td><td> $2 9 . 3 _ { 0 . 5 2 }$ </td><td> $3 0 . 0 _ { 0 . 7 5 }$ </td><td> $8 3 . 9 2 _ { 0 . 0 1 }$ </td><td>32.0</td><td> $2 7 . 1 _ { 1 . 2 }$ </td><td> $3 5 . 1 _ { 0 . 3 4 }$ </td><td> $4 7 . 1 _ { 1 . 1 }$ </td><td> $8 3 . 3 8 _ { 0 . 0 1 }$ </td><td>32.0</td></tr><tr><td>SADDER</td><td> $2 9 . 4 _ { 1 . 7 }$ </td><td> $4 2 . 1 _ { 1 . 4 }$ </td><td> $1 8 . 4 _ { 0 . 2 6 }$ </td><td> $8 4 . 5 9 _ { 0 . 0 6 }$ </td><td>32.0</td><td> $2 9 . 9 _ { 0 . 9 }$ </td><td> $3 4 . 6 \phantom { 0 } _ { 0 . 5 4 }$ </td><td> $2 7 . 8 \phantom { 0 } { 0 . 8 6 }$ </td><td> $8 4 . 2 0 \phantom { 0 } _ { 0 . 0 3 }$ </td><td>32.0</td></tr><tr><td>SADDER + warm-up</td><td> $3 1 . 2 _ { 0 . 7 }$ </td><td> $3 1 . 6 _ { 2 . 1 }$ </td><td> $4 4 . 3 \phantom { 0 } _ { 0 . 7 5 }$ </td><td> $8 3 . 0 1 _ { 0 . 0 1 }$ </td><td>32.0</td><td> $2 4 . 4 _ { 0 . 9 }$ </td><td> $3 6 . 9 _ { 0 . 3 2 }$ </td><td> $4 4 . 1 _ { 0 . 1 4 }$ </td><td> $8 3 . 1 8 _ { 0 . 0 }$  3</td><td>32.0</td></tr><tr><td>CSD†</td><td> $2 2 . 4 \phantom { 0 } _ { 0 . 4 }$ </td><td> $2 2 5 7 . 0$ </td><td> $2 1 . 6 \phantom { 0 } _ { 0 . 2 6 }$ </td><td> $8 4 . 5 9 _ { 0 . 0 4 }$ </td><td>61.9</td><td> $1 5 . 6 _ { 0 . 5 }$ </td><td> $4 8 7 _ { 3 6 }$ </td><td> $9 4 3 _ { 4 0 }$ </td><td> $8 2 . 6 1 _ { 0 . 0 3 }$ </td><td>55.0</td></tr><tr><td>BoN-8</td><td> $1 9 . 0 _ { 0 . 7 }$ </td><td> $2 0 . 5 _ { 0 . 2 6 }$ </td><td> $1 9 . 5 \phantom { 0 } _ { 0 . 4 8 }$ </td><td> $8 4 . 7 0 _ { 0 . 0 0 }$ </td><td>256.0</td><td> $1 8 . 6 \phantom { 0 } _ { 0 . 6 }$ </td><td> $2 4 . 2 _ { 0 . 3 2 }$ </td><td> $1 9 . 5 \phantom { 0 } _ { 0 . 4 8 }$ </td><td> $8 4 . 7 0 _ { 0 . 0 0 }$ </td><td>256.0</td></tr><tr><td>FK-8</td><td> $2 0 . 6 _ { 0 . 8 }$ </td><td> $2 1 . 3 \mathrm { _ { 1 . 1 } }$ </td><td> $1 8 . 2 \mathrm { _ { 0 . 5 3 } }$ </td><td> $8 4 . 7 1 _ { 0 . 0 4 }$ </td><td>256.0</td><td> $2 8 . 2 _ { 1 . 2 }$ </td><td> $2 2 . 7 _ { 0 . 9 3 }$ </td><td> $1 8 . 2 \mathrm { _ { 0 . 5 3 } }$ </td><td> $8 4 . 7 1 _ { 0 . 0 4 }$ </td><td>256.0</td></tr></table>

Sampling-based baselines. Best-of-N sampling generates independent candidate completions and selects the one with the highest reward. FK steering (Singhal et al., 2025) instead maintains multiple diffusion trajectories and resamples them at intermediate steps using reward-derived potentials. Both provide inference-time control without retraining the generator. SADDER modifies token predictions within each trajectory, using unsafe references and activation scores rather than ranking candidate sequences. Comparisons with Best-of-N and FK steering assess whether this local intervention offers a better safety–quality tradeoff at a given inference budget.

Diffusion-specific jailbreaks. DIJA (Wen et al., 2026) constructs interleaved text–mask prompts that exploit bidirectional context and parallel decoding to elicit harmful completions. PAD (Zhang et al., 2025) adapts affirmative-response priming to parallel generation through a multi-point attention attack. These attacks expose vulnerabilities associated with diffusion’s generation process and motivate evaluating safety guidance on adversarially structured inputs as well as ordinary unsafe continuation. We include all in our evaluation of selective guidance and token revision.

## 5 EXPERIMENTS

We ask whether selective guidance improves the safety–quality tradeoff of SAD, and whether its benefits transfer from unsafe continuation to adversarial instructions. Following Yusuf et al. (2026), we evaluate these tasks separately. The main continuation results distinguish SADDER from SAD-DER with a full-SAD warm-up. Other variants, calibration sweeps, and the extended algorithm are reported in Appendix E.

## 5.1 UNSAFE CONTINUATION ON MASKED DIFFUSION LANGUAGE MODELS (MDLMS)

Setting and metrics. We evaluate LLaDA-8B-Base Nie et al. (2025) and Dream-7B-Base Ye et al. (2025) on 600 unsafe prompts per dataset from RealToxicityPrompts (RTP) Gehman et al. (2020) and BeaverTails (BT) Ji et al. (2023), generating 32 tokens in 32 steps. Baselines are unmodified sampling, full SAD, adapted CSD<sup>†</sup> (Lee & Han, 2026), best-of-eight (BoN-8), and eight-particle FK steering (Singhal et al., 2025). SADDER + warm-up applies full SAD on steps 0–7 and selective guidance on steps 8–15. Evaluation prompts were held out from development; later hyperparameter selection uses a separate 300-prompt development set (Appendix E).

Safety is the Llama-Guard-3-8B Meta (2024) unsafe rate. Harmful PPL is GPT-2-large (Radford et al., 2019) corpus perplexity over all outputs to harmful prompts, including safe outputs. Benign BERT (BERTScore F1 ×100) and Benign PPL use 600 benign RTP prompt–continuation pairs, retaining each method’s safety-selected settings and reference set. NFE counts generator forward evaluations, excluding reward scoring and one-time reference encoding.

Safety–quality tradeoffs. On LLaDA/RTP, SADDER lowers the unsafe rate from 38.6% unmodified to 29.4%, versus 32.4% for full SAD; warm-up reaches 31.2%. On BeaverTails, SADDER reaches 29.9%, full SAD 27.1%, and warm-up 24.4%. Across all four model/dataset settings, SAD-DER achieves higher benign BERTScore and lower benign PPL than full SAD. SADDER also achieves higher benign BERTScore than warm-up throughout, although both variants remain below unmodified sampling on this metric (Tables 1 and 9). Harmful-prompt PPL reveals a different tradeoff: on LLaDA/RTP, CSD reaches a 22.4% unsafe rate but PPL 225, versus 42.1 for SADDER and 31.6 for warm-up without remasking. CSD’s benign PPL of 21.6 does not expose this deterioration on harmful prompts. Gains also depend on the model: on Dream, full SAD at $\eta = 1 0 , 0 0 0$ achieves lower unsafe rates than either SADDER variant on both datasets. BoN-8 further reduces unsafe rates at a cost of 256 generator evaluations plus reward scoring.

![](images/aa59c1838d04a00158e43a83733d935005b1794b00a3281afa77d3daabb6f464.jpg)  
Figure 1: Safety–quality tradeoff at equal NFE on Duo and UDLM. Best-of-N is reported in Appendix I. Larger markers denote larger guidance strength η. The dashed vertical line marks the unmodified model’s benign BERTScore, and error bars are one standard deviation across seeds.

## 5.2 UNSAFE CONTINUATION ON UNIFORM-STATE DIFFUSION MODELS

Setting and metrics. We evaluate Duo (Sahoo et al., 2025) and UDLM (Schiff et al., 2025) on 300 unsafe prompt records from RealToxicityPrompts (RTP). Both models generate at a context length of 128 in 128 denoising steps. Unsafe references and contrastive directions follow Section 5.1, with $k \in \{ 6 , 9 , 1 2 \}$ . Baselines are unmodified sampling, full SAD, CSD<sup>†</sup>, and best-of-N for $N \in$ $\{ 3 , 5 , 8 \}$ . Full SAD and SADDER sweep $\eta \in \{ 3 { \times } \bar { 1 } 0 ^ { 4 } , \bar { 5 } { \times } 1 0 ^ { 4 } , 1 0 ^ { 5 } , 1 . 5 { \times } 1 0 ^ { 5 } \}$ . In masked models, guidance must reach a position before it is revealed. Uniform-state samplers have no such constraint, so we gate every position at every step, using Equation 13 for SADDER and no gate for full SAD.

Safety is the Llama-Guard-3-8B unsafe rate. Benign BERT uses 300 separate benign RTP prompt–continuation pairs. All entries are means over five generation seeds, and Appendix I reports exact values with seed standard deviations.

Safety–quality tradeoffs. Figure 1 reports the tradeoff. On UDLM, SADDER reaches a 45.9% unsafe rate, against 68.5% without steering and 47.4% for the lowest unsafe rate among full SAD settings. SADDER’s benign BERTScore is 80.92, against 80.49 for full SAD and 80.81 unmodified, so it is safer than full SAD without any loss in reference similarity. On Duo, full SAD reaches lower unsafe rates than SADDER at every η, from 43.4% versus 47.2% at $\eta = 3 { \times } 1 0 ^ { 4 }$ to 41.1% versus 43.6% at $\eta = 1 . 5 { \times } 1 0 ^ { 5 }$ . The same holds at matched benign BERTScore: full SAD reaches 43.4% at 81.77, where SADDER reaches 45.0%. What SADDER does buy is quality, on both models: at every η its benign BERTScore is higher than full SAD’s. On Duo the gap is widest at the end of the sweep, 81.68 against 80.76.

SADDER, full SAD, and $\mathrm { C S D ^ { \dag } }$ use the same number of generator evaluations, whereas best-of-N uses N times as many. Neither baseline is consistent across the two models. On Duo, best-of-8 reaches 33.1%, the lowest unsafe rate we observe, yet on UDLM best-of-N does not improve on unmodified sampling at any budget. CSD<sup>†</sup> reaches 35.1% on Duo at $k = 9$ but 63.5% on UDLM at the same layer, and it is sensitive to the layer choice: across $k ,$ its unsafe rate on Duo spans 15.5 points, whereas SADDER’s varies by at most 2.2 points at fixed η.

Table 2: Dream-Instruct: held-out HarmBench attacks and XSTest over-refusal. Entries use A : mean and sample SD across three seeds (percentage points for rates). HB, LG, and PPL assess complete responses. XSTest reports full-response WG refusal on safe prompts, using valid judge labels only (coverage in Appendix H.2). NFE is mean cost.
<table><tr><td rowspan="2">Method</td><td colspan="4">DIJA</td><td colspan="4">PAD</td><td>XSTest</td></tr><tr><td> $\% \downarrow$ </td><td>HB ASR LG unsafe  $\% \downarrow$ </td><td>Harmful  $\mathrm { P P L } \downarrow$ </td><td>NFE</td><td> $\% \downarrow$ </td><td>HB ASR LG unsafe  $\% \downarrow$ </td><td>Harmful  $\mathrm { P P L } \downarrow$ </td><td>NFE</td><td>WG refusal %↓</td></tr><tr><td>Dream-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td> $8 2 . 0 _ { 1 . 0 }$ </td><td> $8 0 . 7 _ { 1 . 5 }$ </td><td> $1 1 . 4 \mathrm { _ 0 . 1 8 }$ </td><td>91.7</td><td> $5 1 . 7 _ { 0 . 6 }$ </td><td> $6 4 . 0 { 3 . 6 }$ </td><td> $8 . 5 3 _ { 0 . 0 2 7 }$ </td><td>116.0</td><td> $5 5 . 5 _ { 1 . 3 }$ </td></tr><tr><td>Full SAD</td><td> $8 5 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $8 3 . 0 _ { 1 . 0 }$ </td><td> $1 1 . 8 _ { 0 . 0 3 7 }$ </td><td>91.7</td><td> $5 2 . 7 _ { 4 . 0 }$ </td><td> $6 3 . 3 _ { 1 . 5 }$ </td><td> $9 . 5 2 _ { 0 . 2 1 }$ </td><td>116.0</td><td> $5 8 . 0 _ { 1 . 0 }$ </td></tr><tr><td>SADDER (selective)</td><td> $7 7 . 7 _ { 1 . 5 }$ </td><td> $8 1 . 7 _ { 3 . 2 }$ </td><td> $1 2 . 2 _ { 0 . 0 9 3 }$ </td><td>91.7</td><td> $5 2 . 7 _ { 0 . 6 }$ </td><td> $6 6 . 3 _ { 2 . 5 }$ </td><td> $9 . 5 8 _ { 0 . 1 5 }$ </td><td>116.0</td><td> $4 1 . 1 _ { 0 . 8 }$ </td></tr><tr><td>SADDER + warm-up</td><td> $7 8 . 7 _ { 2 . 3 }$ </td><td> $7 9 . 7 _ { 1 . 2 }$ </td><td> $1 2 . 2 \mathrm { _ { 0 . 1 0 } }$ </td><td>91.7</td><td> $4 2 . 7 _ { 3 . 1 }$ </td><td> $5 5 . 0 _ { 2 . 0 }$ </td><td> $1 0 . 4 0 . 1 7$ </td><td>116.0</td><td> $1 7 . 9 \mathrm { 0 } . 7 $ </td></tr><tr><td>CSD†</td><td> $6 8 . 0 6 . 2 $ </td><td> $8 2 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $4 5 . 9 _ { 4 . 1 }$ </td><td>115.5</td><td> $3 3 . 3 _ { 3 . 1 }$ </td><td> $6 3 . 3 _ { 2 . 5 }$ </td><td> $2 3 . 7 _ { 1 . 5 }$ </td><td>140.8</td><td> $5 6 . 4 1 . 9$ </td></tr><tr><td>DiffuGuard (full)</td><td> $3 3 . 3 _ { 2 . 5 }$ </td><td> $8 4 . 0 _ { 2 . 0 }$ </td><td> $5 3 7 _ { 1 1 }$ </td><td>137.0</td><td> $1 2 . 0 _ { 1 . 7 }$ </td><td> $7 0 . 0 _ { 2 . 6 }$ </td><td> $6 8 7 _ { 1 2 0 }$ </td><td>125.0</td><td> $6 4 . 1 _ { 1 . 5 }$ </td></tr><tr><td>DiffuGuard (gen)</td><td> $6 9 . 3 _ { 1 . 5 }$ </td><td> $8 1 . 3 \mathrm { 1 . 5 }$ </td><td> $4 6 . 6 _ { 1 . 9 }$ </td><td>97.2</td><td> $4 9 . 0 _ { 1 . 7 }$ </td><td> $6 3 . 3 _ { 2 . 3 }$ </td><td> $8 . 8 0 _ { 0 . 6 3 }$ </td><td>117.0</td><td> $5 4 . 6 _ { 1 . 1 }$ </td></tr></table>

## 5.3 JAILBREAK ROBUSTNESS ON INSTRUCTION-TUNED MODELS

Setting. We evaluate DIJA and PAD on LLaDA-8B-Instruct and Dream-7B-Instruct using 100 held-out HarmBench behaviors, 128 sampling steps, and WildJailbreak Jiang et al. (2024) unsafe references. CSD and SADDER share the unsafe-response set. DiffuGuard (Li et al., 2025) is evaluated withfull repair, which can modify fixed attack text, and gen repair, restricted to initially masked positions; their audit thresholds also differ (Appendices H and A.6).

Table 2 reports full-response HarmBench (HB) ASR, LlamaGuard (LG) unsafe rate, harmful PPL, and NFE. XSTest over-refusal uses WildGuard (WG) (Han et al., 2024) judgments on 250 safe, unmodified prompts across three seeds, with the DIJA guidance configuration. WG rates use valid labels (non-empty) within each seed, then average across seeds. Appendices H.5 and H.2 report generated-span judgments, valid-score counts, paired comparisons, and calibration-prompt reuse.

Attack and evaluator dependence. Under DIJA, SADDER and warm-up leave HB ASR near unmodified sampling, while LG flags roughly four-fifths of responses as unsafe. Large HB reductions from CSD and full-block DiffuGuard do not yield comparable LG reductions: Dream’s full-block DiffuGuard reaches 33.3% HB ASR but 84.0% LG unsafe rate and PPL 537. Both judges assess complete responses, so this discrepancy cannot be attributed solely to scaffold inclusion. Appendix H.5 separates evaluator choice from the influence of fixed attack text by comparing fullresponse and generated-span judgments on the same outputs. SADDER retains generator cost and incurs a smaller PPL increase, but offers limited DIJA robustness.

PAD gives a more favorable result: Dream warm-up reduces HB ASR by 9.0 points (paired 95% $\mathrm { C I } { : [ - 1 5 . 7 , - 2 . 3 ] ) }$ , lowers LG unsafe rate from 64.0% to 55.0%, and increases PPL from 8.53 to 10.40. This supports an attack-specific tradeoff rather than general jailbreak robustness. Appendix H reports component, strength, and timing studies.

Over-refusal. Dream warm-up reduces WG refusal on safe XSTest prompts by 35.4 points on jointly scored outputs (95% CI: [−41.6, −29.3]), but raises WG harmfulness on unsafe contrasts from 0.0% to 6.9% among scored responses. Fewer refusals therefore do not establish greater helpfulness at unchanged safety.

## 6 SUMMARY AND DISCUSSION

We introduced SADDER, an extension of the Safety-Aware Denoiser, using midpoint-centred contrastive activation scores to select where to apply probability-space safety guidance. Rather than di rectly modifying hidden representations, SADDER uses them to identify potentially risky positions and applies the SAD correction to the corresponding token distributions. Our theoretical analysis characterizes the conditions under which position-wise guidance recovers the conditional mean of an induced safe target, and identifies when these local conditional means agree with a global target. Experiments on masked and uniform-state diffusion models reveal both the benefits and limitations of selective guidance. SADDER improves benign-continuation quality metrics relative to full SAD in several settings, with selected configurations also reducing unsafe generation. Jailbreak improvements also depend on the model and operating point, but SADDER shows significantly lower refusal rates. Taken together, SADDER offers an additional way to manage the safety–quality and safety refusal trade-offs.

## AI USE STATEMENT

We used generative AI tools, to assist with refining research hypotheses and experimental designs, reviewing mathematical derivations and proof arguments, analyzing and interpreting experimental results, and implementing and debugging experimental code. We also used these tools to draft and revise portions of the manuscript, improve clarity and organization, prepare LaTeX pseudocode, and identify and discuss relevant literature. The models used for text generation and automated safety evaluation within our experiments are described in the experimental protocol.

The authors reviewed and revised the AI-assisted material included in this work. We checked mathematical statements and derivations, verified references against the cited sources, and checked reported numerical claims against experimental outputs. AI-generated code was reviewed and revised by the authors before use in the experiments. The authors made the final methodological decisions and determined the interpretation and presentation of the findings. We take responsibility for the final content of this work, including all text, claims, and artifacts produced with the assistance of generative AI.

## ETHICS STATEMENT

This work investigates training-free safety guidance for text diffusion models. Our experiments involve toxic text, harmful instructions, and adversarial prompts from existing datasets and benchmarks. These materials serve a defensive evaluation purpose but may contain offensive or distressing content. Analysis of model vulnerabilities could also inform attempts to circumvent safeguards. Any release of associated harmful generations or attack artifacts should therefore consider risks of misuse, exposure to harmful content, and disclosure of sensitive information. Furthermore, safety guidance could be misused to suppress legitimate expression, while overly broad or biased safety criteria may restrict benign content and disproportionately affect particular communities.

Our evaluation relies on benchmark datasets and automated safety judges, which may reflect social biases and fail to capture context-dependent harms. Differences between evaluators further limit the interpretation of reported safety metrics. We report safety alongside generation quality, inference cost, and over-refusal, including settings where the proposed guidance offers limited benefit. Reduced refusal does not necessarily imply improved helpfulness, and lower measured attack success does not establish that responses are safe or usable. Our findings are restricted to the evaluated models, datasets, and sampling configurations; they do not establish safety across untested settings. Deployment would require additional assessment of robustness, fairness, and risks specific to the intended application.

## REPRODUCIBILITY STATEMENT

The main text presents the mathematical formulation of SADDER, and Algorithm 1 specifies the selective guidance procedure. Appendix A.2 derives the centered-score equivalence, and Appendices A.4 and A.5 provide proofs of the theoretical results. Appendix A.6 describes the datasets, prompt selection, and reference-set construction. Appendix A.8 documents shared settings, hyper parameter ranges, and configuration selection, while Appendix B details the sampling extensions. Appendices E and H report additional experimental settings and results. Appendix H.5 compares evaluators and response scopes, and Appendix H.2 describes refusal evaluation and the handling of missing judge labels. Appendix B.2 specifies metric aggregation, seed standard deviations, bootstrap intervals, and paired comparisons. These materials support independent implementation and assessment of the reported results.

## ACKNOWLEDGMENTS

This work was supported by Mitacs through the Mitacs Accelerate program, in partnership with RBC Borealis (Royal Bank of Canada).

## REFERENCES

Eden Avrahami and Eliya Nachmani. ILRR: Inference-time steering method for masked diffusion language models, 2026. URL https://arxiv.org/abs/2601.21647.

ByteDance Seed, Yuyu Zhang, Jing Su, Yifan Sun, Chenguang Xi, Xia Xiao, Shen Zheng, Anxiang Zhang, Kaibo Liu, Daoguang Zan, Tao Sun, Jinhua Zhu, Shulin Xin, Dong Huang, Yetao Bai, Lixin Dong, Chao Li, Jianchong Chen, Hanzhi Zhou, Yifan Huang, Guanghan Ning, Xierui Song, Jiaze Chen, Siyao Liu, Kai Shen, Liang Xiang, and Yonghui Wu. Seed-Coder: Let the code model curate data for itself, 2025. URL https://arxiv.org/abs/2506.03524.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, and Eric Wong. Jailbreakbench: An open robustness benchmark\` for jailbreaking large language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 55005–55029. Curran Associates, Inc., 2024. doi: 10.52202/079017-1745. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 63092d79154adebd7305dfd498cbff70-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Juntao Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 50750–50777, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ dd1577afd396928ed64216f3f1fd5556-Paper-Conference.pdf.

DiffusionGemma Team. Diffusiongemma technical report, 2026. URL https://arxiv.org/ abs/2608.00146.

Chenghao Fan, Wen Heng, Bo Li, Sichen Liu, Yuxuan Song, Jing Su, Xiaoye Qu, Kai Shen, and Wei Wei. Stable-diffcoder: Pushing the frontier of code diffusion large language model, 2026. URL https://arxiv.org/abs/2601.15892.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In Findings of EMNLP, 2020.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 8093–8131. Curran Associates, Inc., 2024. doi: 10.52202/079017-0261. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 0f69b4b96a46f284b726fbd70f74fb3b-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 24678–24704. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 4dbb61cb68671edc4ca3712d70083b9f-Paper-Datasets\_and\_Benchmarks. pdf.

Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 47094–47165. Curran Associates, Inc., 2024. doi: 10.52202/

079017-1493. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/54024fca0cef9911be36319e622cde38-Paper-Conference.pdf.

Mingyu Kim, Dongjun Kim, Amman Yusuf, Stefano Ermon, and Mijung Park. Training-free safe denoisers for safe use of diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= QQS7TudonJ.

Mingyu Kim, Young-Heon Kim, and Mijung Park. SAFETY-GUIDED FLOW (SGF): A UNIFIED FRAMEWORK FOR NEGATIVE GUIDANCE IN SAFE GENERATION. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=EA80Zib9UI.

Michael Kirchhof, James Thornton, Louis Bethune, Pierre Ablin, Eugene Ndiaye, and Marco Cuturi.´ Shielded diffusion: Generating novel and diverse images using sparse repellency. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 30911–30942. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/kirchhof25a. html.

Yejin Lee and Yo-Sub Han. Adaptive steering and remasking for safe generation in diffusion language models, 2026. URL https://arxiv.org/abs/2605.13043.

Raymond Li, Loubna Ben allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia LI, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Joel Lamy-Poirier, Joao Monteiro, Nicolas Gontier, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Ben Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason T Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Urvashi Bhattacharyya, Wenhao Yu, Sasha Luccioni, Paulo Villegas, Fedor Zhdanov, Tony Lee, Nadav Timor, Jennifer Ding, Claire S Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Munoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro Von Werra, and˜ Harm de Vries. Starcoder: may the source be with you! Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id= KoFOg41haE. Reproducibility Certification.

Zherui Li, Zheng Nie, Zhenhong Zhou, Yufei Guo, Yue Liu, Yitong Zhang, Yu Cheng, Qingsong Wen, Kun Wang, and Jiaheng Zhang. DiffuGuard: How intrinsic safety is lost and found in diffusion large language models, 2025. URL https://arxiv.org/abs/2509.24296.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Ruslan Salakhutdi nov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings ofthe 41st International Conference on Machine Learning, vol ume 235 of Proceedings of Machine Learning Research, pp. 35181–35224. PMLR, 21–27 Jul 2024a. URL https://proceedings.mlr.press/v235/mazeika24a.html.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. cais/harmbench-llama-2-13b-cls (hugging face model card). https://huggingface.co/cais/HarmBench-Llama-2-13b-cls, 2024b. Accessed: January 29, 2026.

Meta. Llama guard 3 (8b) model card and hazard taxonomy. https://huggingface.co/ meta-llama/Llama-Guard-3-8B, 2024. Accessed: January 29, 2026.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models. ACM Trans. Intell. Syst. Technol., 16(5), August 2025. ISSN 2157-6904. doi: 10.1145/ 3744746. URL https://doi.org/10.1145/3744746.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=KnqiC0znVF.

Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Gradtts: A diffusion probabilistic model for text-to-speech. In International conference on machine learning, pp. 8599–8608. PMLR, 2021.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https://cdn.openai.com/better-language-models/language\_ models\_are\_unsupervised\_multitask\_learners.pdf.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15504–15522. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.828. URL https://aclanthology.org/2024.acl-long.828/.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10674–10685. ieee, 2022.

Paul Rottger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk¨ Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models, 2024. URL https://arxiv.org/abs/2308.01263.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=L4uaAR4ArM.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The diffusion duality. In Forty-second International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2506.10892.

Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, and Ante Jukic. Scaling beyond masked diffusion language models. In´ Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=xKJG1CNtQM.

Yair Schiff, Subham Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-Torre, Bernardo Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. In International Conference on Learning Representations, volume 2025, pp. 43776–43821, 2025.

Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen Mckeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 55810–55827. PMLR, 2025. URL https: //proceedings.mlr.press/v267/singhal25b.html.

Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2503.00307.

Zichen Wen, Jiashu Qu, Zhaorun Chen, Xiaoya Lu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, and Linfeng Zhang. The Devil behind the mask: An emergent safety vulnerability of diffusion LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=rIPeatvPy3.

Kevin E Wu, Kevin K Yang, Rianne Van den Berg, Sarah Alamdari, James Y Zou, Alex X Lu, and Ava P Amini. Protein structure generation via folding diffusion. Nature communications, 15(1): 1059, 2024.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, and Radha Poovendran. SafeDecoding: Defending against jailbreak attacks via safety-aware decoding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 5587–5605, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 303. URL https://aclanthology.org/2024.acl-long.303/.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement, 2024. URL https://arxiv.org/abs/2409.12122.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models, 2025. URL https://arxiv.org/abs/ 2508.15487.

Amman Yusuf, Zhejun Jiang, and Mijung Park. The safety-aware denoiser for text diffusion models. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=dw5BM3OQfZ.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/1904.09675.

Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, and Yufei Guo. Jailbreaking large language diffusion models: Revealing hidden safety flaws in diffusion-based text generation. In Responsible Foundation Models Workshop at NeurIPS 2025, 2025. URL https://openreview.net/forum?id=NWvL280s2l.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models, 2023. URL https://arxiv. org/abs/2307.15043.

## A APPENDIX

## A.1 NOTATION

Tables 3 and 4 collect the notation for selective SADDER and its theoretical analysis. The special mask token M is distinct from the centroid midpoint $\mathbf { \nabla } m _ { k }$ and response truncation cap M. We distinguish decreasing noise time t from the increasing reverse iteration index $j \colon j = 0$ is the first model update, and the corresponding noise levels satisfy $t _ { 0 } > \cdots > t _ { T } = 0 .$ All experimental windows below use inclusive iteration indices. So $C = 0 { - } 1 5$ means 16 updates. The optional warm-up length w counts initial full-SAD updates; $w = 8$ uses updates 0–7.

Table 3: Notation for selective SADDER.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\pmb { c } _ { i } , \pmb { x } _ { i } ^ { \mathrm { h a r m } } , \pmb { x } _ { i } ^ { \mathrm { s a f e } }$   $P _ { i } , \ R _ { i } ^ { u } , \ \dot { M }$ </td><td>Reference prompt and its harmful and safe responses. Prompt length, response length min  $( | \pmb { x } _ { i } ^ { u } | , M )$  , and response truncation</td></tr><tr><td></td><td> $\mathsf { c a p } ; \bar { u } \in \{ \mathrm { h a r m } , \mathrm { s a f e } \}$ </td></tr><tr><td> $N _ { \mathrm { h a r m } } , \ N _ { \mathrm { s a f e } }$ </td><td>Numbers of harmful and safe examples used to estimate the centroids.</td></tr><tr><td> $\mathbf { \Delta x } _ { t } , \mathbf { \Delta x } _ { 0 } , \mathbf { \Delta x } _ { 0 } ^ { l }$   $\nu , L , \mathbb { M }$ </td><td>Noised sequence, clean sequence, and one-hot clean token at position  $l .$ </td></tr><tr><td> $T , j , t _ { j } , C$ </td><td>Vocabulary, sequence length, and special mask token. Number of reverse updates, update index, corresponding noise time, and</td></tr><tr><td> $\mathcal { G } , \mathcal { M } _ { t }$ </td><td>admitted intervention indices. Editable positions and positions eligible for steering: masked editable</td></tr><tr><td> $k , \ l , \ d$ </td><td>positions for MDLMs; all  $L$  positions for USDMs in the stated algorithm.</td></tr><tr><td> $h _ { k , l } , h _ { k , l } ^ { ( t ) }$ </td><td>Hidden-state layer, token position, and hidden dimension. Layer-k hidden representation at position  $l ;$  the superscript denotes eval-</td></tr><tr><td> $\bar { h } _ { i , k } ^ { u } , \bar { h } _ { k } ^ { u }$ </td><td>uation on the current noised state. Response-averaged representation of reference i and the corresponding</td></tr><tr><td> $\widehat { \boldsymbol { v } } _ { k } , m _ { k }$ </td><td>class centroid. Unit direction from the safe to the harmful centroid, and their midpoint.</td></tr><tr><td> $s _ { k , l } ^ { ( t ) }$ </td><td>Centered score  $\langle \pmb { h } _ { k , l } ^ { ( t ) } \ - \ m _ { k } , \widehat { \pmb { v } } _ { k } \rangle$  ; positive values lie on the harmful-</td></tr><tr><td> $a _ { k , l } ^ { ( t ) }$ </td><td>centroid side. Binary steering gate: one at eligible positions with positive scores and</td></tr><tr><td> $\eta , \beta ( { \pmb x } _ { t } )$ </td><td>zero otherwise.</td></tr><tr><td></td><td>Guidance hyperparameter and state-dependent SAD scale; the gated strength is  $\eta \beta ( \pmb { x } _ { t } ) a _ { k , l } ^ { ( t ) } .$ </td></tr><tr><td> $p _ { \mathcal { D } , t , l } , \widetilde { p } _ { 0 | t , l }$ </td><td>Model clean-token prediction and modified prediction supplied to the re- verse solver.</td></tr><tr><td> $\sigma , \tau$ </td><td>Sigmoid and positive temperature in the optional softened gate.</td></tr></table>

Table 4: Denoiser and theoretical notation.
<table><tr><td>Symbol Meaning</td><td></td></tr><tr><td> $\mathcal { D } , \mathcal { D } _ { \mathrm { u n s a f e } }$ </td><td>Data distribution underlying the pretrained denoiser and unsafe reference</td></tr><tr><td> $q _ { t } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } )$ </td><td>set used by the algorithm, respectively. Forward corruption kernel, used to weight clean unsafe references given</td></tr><tr><td> $\mathbb { E } _ { \mathcal { D } } [ \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } ]$ </td><td>the noised state. Data denoiser at position  $l ,$  estimated by the pretrained model.</td></tr><tr><td> $\mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ { \pmb x } _ { 0 } ^ { \bar { l } } \ | \ { \mathbf { x } } _ { t } ^ { 1 : L } ]$ </td><td>Unsafe denoiser, approximated by a weighted mixture of unsafe refer- ences.</td></tr><tr><td> $p _ { \mathcal { D } } ( \pmb { x } _ { 0 } ^ { 1 : L } \mid \pmb { x } _ { t } ^ { 1 : L } )$ </td><td>Data posterior over the entire clean sequence.</td></tr><tr><td> $\psi _ { l } , \pi _ { t , l }$ </td><td>Oracle unsafe-membership function  $\mathcal { \hat { V } } ^ { L } \to [ 0 , 1 ]$  and its posterior mean  $\pi _ { t , l } = \mathbb { E } _ { \mathcal { D } } [ \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \ | \ \pmb { x } _ { t } ^ { 1 : L } ] .$ </td></tr><tr><td> $Z _ { \mathrm { s a f e } , l }$ </td><td>Local safe normalizer  $1 - \pi _ { t , l } ;$  dependence on noise time and state is suppressed.</td></tr><tr><td> $g _ { l } , \ R _ { l }$ </td><td>Local safe weight  $g _ { l } ~ = ~ 1 - \psi _ { l }$  and remaining-position weight  $R _ { l } \ =$   $\prod _ { j \neq l } g _ { j } , \mathbf { w i t h } ~ j$  a dummy position index.</td></tr><tr><td> $\mathcal { P } _ { \mathcal { D } _ { \mathrm { u n s a f e } } ^ { ( l ) } } , \mathcal { P } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } }$ </td><td>Clean-sequence posteriors obtained by weighting pD with  $\psi _ { l }$  or  $g _ { l }$  and normalizing by  $\pi _ { t , l }$  or  $1 - \pi _ { t , l } .$  The superscript (l) identifies the safety criterion.</td></tr><tr><td> $p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } }$ </td><td>Global clean-sequence posterior obtained by weighting  $p _ { \mathcal { D } }$  with  $\prod _ { l } g _ { l }$  and normalizing.</td></tr><tr><td> $\mathbb { E } _ { \nu } , \mathrm { C o v } _ { \nu }$ </td><td>Expectation and covariance under distribution  $\nu ,$  conditioned on  $\pmb { x } _ { t } ^ { 1 : L }$  in the proofs.</td></tr></table>

## A.2 DERIVATION OF EQUIVALENCE

Expanding the relationship in equation 10 gives us

$$
2 \left. h _ { k , l } ^ { ( t ) } , \bar { h } _ { k } ^ { \mathrm { h a r m } } - \bar { h } _ { k } ^ { \mathrm { s a f e } } \right. > \| h _ { k } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } - \| h _ { k } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } .\tag{18}
$$

In terms of $\hat { v } _ { k }$ , this is,

$$
\left. h _ { k , l } ^ { ( t ) } , \hat { v } _ { k } \right. > \frac { \| h _ { k } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } - \| h _ { k } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } } { 2 \| \bar { h } _ { k } ^ { \mathrm { h a r m } } - \bar { h } _ { k } ^ { \mathrm { s a f e } } \| } .\tag{19}
$$

For notational simplicity, we ignore superscripts and subscripts $j , k ,$ t below.

$$
\langle h , \hat { v } \rangle > \frac { \| \bar { h } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } - \| \bar { h } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } } { 2 \| \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } \| } ,\tag{20}
$$

since $\langle h - m , \hat { v } \rangle = \langle h , \hat { v } \rangle - \langle m , \hat { v } \rangle$ , we rewrite the above equation by

$$
\langle h - m , \hat { v } \rangle > \frac { \| \bar { h } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } - \| \bar { h } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } } { 2 \| \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } \| } - \langle m , \hat { v } \rangle ,\tag{21}
$$

$$
\frac { \| \bar { h } ^ { \mathrm { h a r m } } \| _ { 2 } ^ { 2 } - \| \bar { h } ^ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } } { 2 \| \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } \| } - \frac { ( \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } ) ^ { T } ( \bar { h } ^ { \mathrm { h a r m } } + \bar { h } ^ { \mathrm { s a f e } } ) } { 2 \| \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } \| } ,\tag{22}
$$

$$
= \frac { ( \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } ) ^ { T } ( \bar { h } ^ { \mathrm { h a r m } } + \bar { h } ^ { \mathrm { s a f e } } ) - ( \bar { h } ^ { \mathrm { h a r m } } - \bar { h } ^ { \mathrm { s a f e } } ) ^ { T } ( \bar { h } ^ { \mathrm { h a r m } } + \bar { h } ^ { \mathrm { s a f e } } ) } { \alpha } = 0 .
$$

$$
\overline { { 2 \| \bar { \pmb h } ^ { \mathrm { h a r m } } - \bar { \pmb h } ^ { \mathrm { s a f e } } \| } }\tag{23}
$$

where the RHS becomes zero by expanding the additional inner product $\langle c , \hat { v } \rangle$

## A.3 GLOBAL TARGET DISTRIBUTION.

Theorem A.1 (When position-wise SADDER target distribution matches a global target distribution). Given the position-wise safe target from Thm. 3.1 given in equation 32,

$$
p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) = \frac { \left( 1 - \psi _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) \right) p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) } { 1 - \pi _ { t , l } ( \boldsymbol { x } _ { t } ^ { 1 : L } ) } ,\tag{24}
$$

and the global target distribution, with $\begin{array} { r l r } { g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) } & { { } : = } & { 1 - \ \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) } \end{array}$ and $\begin{array} { r l } { R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) } & { { } : = } \end{array}$ $\begin{array} { r } { \prod _ { j \neq l } \left( 1 - \psi _ { j } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \right) } \end{array}$ for each position l,

$$
p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } } ( { \pmb x } _ { 0 } ^ { 1 : L } \mid { \pmb x } _ { t } ^ { 1 : L } ) = \frac { g _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) R _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) p _ { \mathcal { D } } ( { \pmb x } _ { 0 } ^ { 1 : L } \mid { \pmb x } _ { t } ^ { 1 : L } ) } { \mathbb { E } _ { \mathcal { D } } \left[ g _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) R _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) \mid { \pmb x } _ { t } ^ { 1 : L } \right] } .\tag{25}
$$

the following equality holds,

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } } ^ { g l o b a l } ( { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ) = \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ) ,\tag{26}
$$

if and only if

$$
\begin{array} { r } { \operatorname { C o v } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left( R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) , \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } \right) = 0 , } \end{array}\tag{27}
$$

for every $l , t , \pmb { x } _ { t } ^ { 1 : L }$

See Section A.5 for proof.

Remark 2. Thm. A.1 states that the global and position-wise denoisers agree at position l for all $l =$ $\{ 1 , \cdots , L \}$ if and only if the remaining-position safety weight denoted by $R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } )$ is conditionally uncorrelated with $\mathbf { \Delta x } _ { 0 } ^ { l }$ under $\mathcal { D } _ { \mathrm { s a f } \epsilon } ^ { ( l ) }$ . Under this condition and the calibration assumption in Thm. 3.1, every SADDER factor is the marginal safe denoiser of the same global target. However, this does not guarantee the existence of any globally compatible distribution. It only states that this specific global target in equation 25 agrees with the local target at position l if and only if a conditional covariance is zero.

## A.4 PROOF OF THM. 3.1

Proof. In the theorem, we assumed that these exists a measurable position-wise unsafe-membership function on clean sequences. In particular, for each token position $l \in \{ 1 , \ldots , L \}$ , this function maps the clean sequence to a level of unsafe-ness at index l:

$$
\psi _ { l } : \mathcal { V } ^ { L } \longrightarrow [ 0 , 1 ] .\tag{28}
$$

Recall that we let

$$
\pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) : = \mathbb { E } _ { D } \left[ \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \ | \ \pmb { x } _ { t } ^ { 1 : L } \right]\tag{29}
$$

denote the posterior probability of the position-l being unsafe, for a noised sequence $\pmb { x } _ { t } ^ { 1 : L }$

Now, we define the corresponding normalized position-wise safe and unsafe posterior distributions by

$$
p _ { \mathcal { D } _ { \mathrm { u n s a f e } } ^ { ( l ) } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) : = \frac { \psi _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) } { \pi _ { t , l } ( \boldsymbol { x } _ { t } ^ { 1 : L } ) } ,\tag{30}
$$

$$
\pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) : = \sum _ { \pmb { x } _ { 0 } ^ { 1 : L } \in \mathcal { D } } \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) p _ { \mathrm { d a t a } } ( \pmb { x } _ { 0 } ^ { 1 : L } | \pmb { x } _ { t } ^ { 1 : L } ) ,\tag{31}
$$

$$
p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) : = \frac { \left( 1 - \psi _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) \right) p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } | \boldsymbol { x } _ { t } ^ { 1 : L } ) } { Z _ { \mathrm { s a f e } , l } } ,\tag{32}
$$

$$
1 - \pi _ { t , l } ( \pmb { x } _ { t } ^ { 1 : L } ) : = \sum _ { \pmb { x } _ { 0 } ^ { 1 : L } \in \mathcal { D } } \big ( 1 - \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \big ) p _ { \mathrm { d a t a } } ( \pmb { x } _ { 0 } ^ { 1 : L } | \pmb { x } _ { t } ^ { 1 : L } ) .\tag{33}
$$

Because $\psi _ { l } \big ( \mathbf { x } _ { 0 } ^ { 1 : L } \big ) + \big ( 1 - \psi _ { l } \big ( \mathbf { x } _ { 0 } ^ { 1 : L } \big ) \big ) = 1$ for any clearn sequence $\pmb { x } _ { 0 } ^ { 1 : L }$ , we can write the following:

$$
\begin{array} { r l } & { p _ { d a t a } ( \pmb { x } _ { 0 } ^ { 1 : L } | \pmb { x } _ { t } ^ { 1 : L } ) = ( 1 - \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) ) p _ { d a t a } ( \pmb { x } _ { 0 } ^ { 1 : L } | \pmb { x } _ { t } ^ { 1 : L } ) } \\ & { ~ + \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) p _ { d a t a } ( \pmb { x } _ { 0 } ^ { 1 : L } | \pmb { x } _ { t } ^ { 1 : L } ) . } \end{array}\tag{34}
$$

(35)

If we plug in equation 30 and equation 32 to the RHS, the data posterior can be decomposed into the position-wise, normalized, safe and unsafe distributions:

$$
\begin{array} { r } { p _ { \mathrm { d a t a } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } \mid \boldsymbol { x } _ { t } ^ { 1 : L } ) = \big ( 1 - \pi _ { t , l } ( \boldsymbol { x } _ { t } ^ { 1 : L } ) \big ) p _ { D _ { \mathrm { s a f e } } ^ { ( l ) } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } \mid \boldsymbol { x } _ { t } ^ { 1 : L } ) } \\ { + \pi _ { t l } ( \boldsymbol { x } _ { t } ^ { 1 : L } ) p _ { D _ { \mathrm { u n s a f e } } ^ { ( l ) } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } \mid \boldsymbol { x } _ { t } ^ { 1 : L } ) . } \end{array}\tag{36}
$$

Taking the conditional expectation of the one-hot clean token $x _ { 0 } ^ { l }$ on both sides gives, i.e., $\begin{array} { r } { \mathbb { E } _ { \pmb { x } _ { 0 } ^ { l } \sim \mathcal { D } } [ \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ] = \int \pmb { x } _ { 0 } ^ { l } p ( \dot { \pmb { x } } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ) d \pmb { x } _ { 0 } ^ { l } } \end{array}$

$$
\mathbb { E } _ { \mathcal { D } } [ \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ] = \big ( 1 - \pi _ { t , l } \big ) \mathbb { E } _ { D _ { \mathrm { s a f e } } ^ { ( l ) } } [ \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } ] + \pi _ { t , l } \mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } ^ { ( l ) } } [ \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ] ,\tag{37}
$$

where the dependence of $\pi _ { t , l }$ on $\pmb { x } _ { t } ^ { 1 : L }$ is suppressed for readability. Solving equation 37 for the safe conditional mean yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left[ { \pmb x } _ { 0 } ^ { l } \ | \ { \pmb x } _ { t } ^ { 1 : L } \right] = \frac { \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] - \pi _ { t , l } \mathbb { E } _ { \mathcal { D } _ { \mathrm { u s a f e } } ^ { ( l ) } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] } { 1 - \pi _ { t , l } } } \\ { = \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] + \displaystyle \frac { \pi _ { t , l } } { 1 - \pi _ { t , l } } \left( \mathbb { E } _ { \mathcal { D } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] - \mathbb { E } _ { \mathcal { D } _ { \mathrm { u s a f e } } ^ { ( l ) } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] \right) . } \end{array}\tag{38}
$$

Substituting the calibration condition in equation 17 gives equation 16.

Note that the unsafe denoiser in this theorem is induced by ψ, i.e., $\begin{array} { r l } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { u n s a f e } } } [ { \pmb x } _ { 0 } ^ { l } | { \pmb x } _ { t } ^ { 1 : L } ] } & { = } \end{array}$ $\frac { \sum _ { \pmb { x } _ { n } \in \mathcal { D } _ { \mathrm { u n s a f e } } } \pmb { x } _ { n } ^ { l } \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) q _ { t } ( \pmb { x } _ { t } ^ { 1 : L } | \pmb { x } _ { n } ^ { 1 : L } ) } { \sum _ { \pmb { x } _ { n } \in \mathcal { D } _ { \mathrm { u n s a f e } } } \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) q _ { t } ( \pmb { x } _ { t } ^ { 1 : L } | \pmb { x } _ { n } ^ { 1 : L } ) }$ , which differs from our implementation of unsafe denoiser given in equation 15. Hence, we note that this theorem statement holds when these two unsafe denoisers match.

## A.5 PROOF OF THM. A.1

Proof. Conditioned on $\pmb { x } _ { t } ^ { 1 : L }$ , the global posterior satisfies

$$
p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } } ( { \pmb x } _ { 0 } ^ { 1 : L } \mid { \pmb x } _ { t } ^ { 1 : L } ) = \frac { g _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) R _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) p _ { \mathcal { D } } ( { \pmb x } _ { 0 } ^ { 1 : L } \mid { \pmb x } _ { t } ^ { 1 : L } ) } { \mathbb { E } _ { \mathcal { D } } \left[ g _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) R _ { l } ( { \pmb x } _ { 0 } ^ { 1 : L } ) \mid { \pmb x } _ { t } ^ { 1 : L } \right] } ,\tag{39}
$$

where $g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) : = 1 - \psi _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } )$

The position-l safe posterior is

$$
p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( \pmb { x } _ { 0 } ^ { 1 : L } \mid \pmb { x } _ { t } ^ { 1 : L } ) = \frac { g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) p _ { \mathcal { D } } ( \pmb { x } _ { 0 } ^ { 1 : L } \mid \pmb { x } _ { t } ^ { 1 : L } ) } { \mathbb { E } _ { \mathcal { D } } [ g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \mid \pmb { x } _ { t } ^ { 1 : L } ] } .\tag{40}
$$

If we plug in equation 40 into equation 39, we have the global posterior as a reweighted positionwise posterior:

$$
p _ { { \mathcal D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } \mid { \boldsymbol x } _ { t } ^ { 1 : L } ) = \frac { R _ { l } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } ) p _ { { \mathcal D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } \mid { \boldsymbol x } _ { t } ^ { 1 : L } ) \mathbb { E } _ { { \mathcal D } } [ g _ { l } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } ) \mid { \boldsymbol x } _ { t } ^ { 1 : L } ] } { \mathbb { E } _ { { \mathcal D } } [ g _ { l } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } ) R _ { l } ( { \boldsymbol x } _ { 0 } ^ { 1 : L } ) \mid { \boldsymbol x } _ { t } ^ { 1 : L } ] }\tag{41}
$$

$$
R _ { l } ( \mathbf { x } _ { 0 } ^ { 1 : L } ) p _ { _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } } ( \mathbf { x } _ { 0 } ^ { 1 : L } \mid \mathbf { x } _ { t } ^ { 1 : L } )
$$

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } [ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \ | \ \pmb { x } _ { t } ^ { 1 : L } ] \quad ,\tag{42}
$$

where the second line is true because

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } [ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \mid \pmb { x } _ { t } ^ { 1 : L } ] = \sum _ { \pmb { x } _ { 0 } ^ { 1 : L } } R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) p _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } ( \pmb { x } _ { 0 } ^ { 1 : L } \mid \pmb { x } _ { t } ^ { 1 : L } ) ,\tag{43}
$$

$$
= \sum _ { \boldsymbol { x } _ { 0 } ^ { 1 : L } } R _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) \frac { g _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) p _ { \mathcal { D } } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } \mid \boldsymbol { x } _ { t } ^ { 1 : L } ) } { \mathbb { E } _ { \mathcal { D } } [ g _ { l } ( \boldsymbol { x } _ { 0 } ^ { 1 : L } ) \mid \boldsymbol { x } _ { t } ^ { 1 : L } ] } , \mathrm { d u e ~ t o ~ e q u a t i o n ~ } 4 0\tag{44}
$$

$$
\frac { \mathbb { E } _ { \mathcal { D } } [ g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \mid \pmb { x } _ { t } ^ { 1 : L } ] } { \mathbb { E } _ { \mathcal { D } } [ g _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \mid \pmb { x } _ { t } ^ { 1 : L } ] } .\tag{45}
$$

Taking the conditional expectation of $\mathbf { \Delta } _ { \mathbf { x } _ { 0 } ^ { l } }$ on equation 42 gives

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } } [ \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } ] = \frac { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left[ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } \right] } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left[ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \ | \ \pmb { x } _ { t } ^ { 1 : L } \right] } .\tag{46}
$$

Subtracting $\mathbb { E } _ { \mathcal { D } _ { \mathrm { e a f e } } ^ { ( l ) } } [ \pmb { x } _ { 0 } ^ { l } \ | \ \pmb { x } _ { t } ^ { 1 : L } ]$ and using the definition of conditional covariance proves

$$
\mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { \mathrm { g l o b a l } } } \big [ \pmb { x } _ { 0 } ^ { l } \big | \pmb { x } _ { t } ^ { 1 : L } \big ] - \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \big [ \pmb { x } _ { 0 } ^ { l } \big | \pmb { x } _ { t } ^ { 1 : L } \big ]\tag{47}
$$

$$
= \frac { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } [ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \pmb { x } _ { 0 } ^ { l } \mid \pmb { x } _ { t } ^ { 1 : L } ] } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } [ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) \mid \pmb { x } _ { t } ^ { 1 : L } ] } - \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } [ \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } ]\tag{48}
$$

$$
= \frac { 1 } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a r e } } ^ { ( l ) } } \left[ R _ { l } ( { \bf x } _ { 0 } ^ { 1 : L } ) \mid x _ { t } ^ { 1 : L } \right] } \left( \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a r e } } ^ { ( l ) } } \left[ R _ { l } ( { \bf x } _ { 0 } ^ { 1 : L } ) x _ { 0 } ^ { l } \mid x _ { t } ^ { 1 : L } \right] - \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a r e } } ^ { ( l ) } } \left[ R _ { l } ( { \bf x } _ { 0 } ^ { 1 : L } ) \mid x _ { t } ^ { 1 : L } \right] \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a r e } } ^ { ( l ) } } \left[ x _ { 0 } ^ { l } | x _ { t } ^ { 1 : L } \right] \right)\tag{49}
$$

$$
= \frac { \mathrm { C o v } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left( R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) , \pmb { x } _ { 0 } ^ { l } | \pmb { x } _ { t } ^ { 1 : L } \right) } { \mathbb { E } _ { \mathcal { D } _ { \mathrm { s a f e } } ^ { ( l ) } } \left[ R _ { l } ( \pmb { x } _ { 0 } ^ { 1 : L } ) | \pmb { x } _ { t } ^ { 1 : L } \right] } ,\tag{50}
$$

where the last line is true due to

$$
C o v [ X , Y | Z ] = \mathbb { E } \left[ ( X - \mathbb { E } ( X | Z ) ) ( Y - \mathbb { E } ( Y | Z ) ) \right] ] ,\tag{51}
$$

$$
= \mathbb { E } [ X Y \vert Z ] - \mathbb { E } [ X \vert Z ] \mathbb { E } [ Y \vert Z ] - \mathbb { E } [ X \vert Z ] \mathbb { E } [ Y \vert Z ] + \mathbb { E } [ X \vert Z ] \mathbb { E } [ Y \vert Z ] ,\tag{52}
$$

$$
= \mathbb { E } [ X Y | Z ] - \mathbb { E } [ X | Z ] \mathbb { E } [ Y | Z ] .\tag{53}
$$

Since the denominator is positive, the global and local means are the same, when the numerator is zero. □

## A.6 DATASETS, PROMPT SELECTION, AND REFERENCE CONSTRUCTION

We distinguish three data roles: evaluation prompts, unsafe references for the SAD update, and safe/harmful examples for constructing the activation direction. A benchmark used for one role is not automatically used for the others.

RealToxicityPrompts (RTP). RTP contains naturally occurring text prefixes and their original continuations, with Perspective annotations (Gehman et al., 2020). For unsafe continuation, the configured pool requires a maximum available prompt-side Perspective score of at least 0.8 and a subsequent Llama-Guard unsafe verdict on the prompt. Its builder also retains the adapter’s continuationscore floor of 0.5. These filters describe an enriched evaluation cohort, not the full RTP distribution. Since Llama-Guard also judges generated outputs, the prompt selection is not judge-independent. Note the negation set used to construct the unsafe reference bank for the contrastive direction is not Llama-Guard scored.

The original RTP development/search set comprises pool indices 0–299. The reported continuation tables use the expanded evaluation set, indices 300–899, collected as three 200-prompt shards at offsets 300, 500, and 700. These evaluation records exclude the original development records. The subsequent configuration-selection procedure is described below.

BeaverTails. BeaverTails provides instruction–response pairs with response-safety annotations (Ji et al., 2023). We condition on prompts associated with unsafe-labeled responses, using 600 records for the evaluation set. The corresponding unsafe responses supply the negative reference set and safe-labeled responses supply the safe direction set. An unsafe response annotation does not establish that its prompt is itself malicious.

Benign continuation controls. The quality cohort requires both the maximum prompt-side Perspective score and the maximum score from the continuation’s own fields to be at most 0.2. We remove duplicate prompt strings and prompt strings present in the excluded unsafe-reference artifacts, shuffle with seed zero, and retain 600 records. BERTScore compares each generated response with its own benign human continuation, including for unmodified sampling. BeaverTails-tuned methods use this same benign RTP cohort while retaining their BeaverTails reference banks. This is a quality diagnostic rather than an independent capability evaluation.

JailbreakBench, AdvBench, and HarmBench. JailbreakBench supplies harmful behaviors for standardized jailbreak evaluation (Chao et al., 2024); its 100 harmful behaviors are development data in our study. AdvBench contains harmful instructions used to evaluate adversarial prompting (Zou et al., 2023). HarmBench provides harmful behaviors and a behavior-conditioned evaluation classifier (Mazeika et al., 2024a) and our jobs restrict evaluation to its standard behavior category. We shuffle with seed zero and use the first 100 behaviors of each dataset for development. The transfer protocol reserves AdvBench offsets 100–519 and HarmBench-standard offsets 100–199. The main jailbreak tables report the latter held-out HarmBench cohort under DIJA and PAD. These are our experimental splits, not official benchmark train/test partitions.

WildJailbreak and the jailbreak reference bank. WildJailbreak contains both harmful and benign prompts, including adversarial rewrites (Jiang et al., 2024). Our unsafe set construction selects the adversarial-harmful training subset, skips the first 1,000 filtered records, and generates LLaDA-Instruct responses from three disjoint 1,300-prompt windows. Exact evaluation-behavior strings are excluded. Generation uses 256 steps, up to 256 new tokens, temperature zero, and seed 42. Unsaferesponse filtering, refusal filtering, and deduplication produce the documented 357-response unsafe WildJailbreak set. The jailbreak contrastive direction uses unsafe responses and refusal references with the chat template and a 64-token response cap.

XSTest and over-refusal. XSTest contains 250 safe prompts that models should answer and 200 unsafe contrast prompts (Rottger et al., 2024). Over-refusal is the refusal rate on the safe subset,¨ reported separately from refusal on unsafe prompts. Our runs use unmodified prompts through the model’s chat template (vanilla). The evaluator we use matches the response opening against XSTest’s full-refusal strings; it does not detect partial refusals or every paraphrased refusal.

## A.7 MODELS, ATTACKS, AND EVALUATION MEASURES

We use LLaDA-8B-Base and LLaDA-8B-Instruct (Nie et al., 2025), and Dream-7B-Base and Dream-7B-Instruct (Ye et al., 2025). The base checkpoints are evaluated by prompted continuation; instruction checkpoints use the attack-specific input construction. The implementation’s layer k follows the hidden-state indexing and interior layers are inputs to block k, and the final layer includes the model’s final normalization.

Attacks and fixed context. Zero-shot runs present the original harmful behavior through the chat template. DIJA fills mask spans inside the saved adversarial prompt (Wen et al., 2026) and the refined prompts are reused across methods. PAD inserts fixed anchors into the generation region (Zhang et al., 2025). SADDER and generation-scope DiffuGuard modify only initially editable positions. Full-block DiffuGuard additionally permits repair of fixed scaffold text after the first masked position, including PAD anchors; under DIJA it also appends the released implementation’s extra masked tail. The main jailbreak tables concern DIJA and PAD.

Safety and quality. Llama-Guard-3-8B (Meta, 2024) judges the generated response alone under continuation framing. HarmBench-Llama-2-13b-cls (Mazeika et al., 2024a;b) judges the response against the requested behavior. We report their respective unsafe rates separately. GPT-2-large (Radford et al., 2019) scores generated text with a causal language-model likelihood; this is an external fluency proxy for both model families, not LLaDA’s pseudo-perplexity. Padding and text after the first end-of-sequence token are excluded. Harmful PPL includes all scorable outputs on the unsafe cohort, whether the safety judge labels them safe or unsafe. Benign PPL and BERTScore F1 (Zhang et al., 2020) use the separate benign cohort. BERTScore specifies English and disables baseline rescaling; it leaves the encoder to the installed package’s default. BERTScore is multiplied by 100 in the tables. Higher reference similarity and lower PPL are useful diagnostics, but neither alone establishes helpfulness or safety.

WildGuard response views. For DIJA, the appendix additionally reports WildGuard Han et al. (2024) harmfulness on the newline-joined spans marked as initially masked, paired with the original harmful behavior. This excludes the fixed scaffold and any full-block repairs outside those spans. Full-response WildGuard harmfulness and refusal remain in the appendix; XSTest uses full-response refusal. The generated-span evaluation is available only for fixed-policy DIJA runs, not PAD or the window studies (Appendix H.5).

Uncertainty and cost. Tables report averages with 95% confidence intervals, estimated by repeatedly resampling the evaluation prompts. The appendix also reports standard deviations (SDs) across three generation seeds, showing how much results vary between runs. Appendix B.2 explains both measures and the paired comparisons. NFE counts generator forward passes, including additional passes after remasking. It excludes reward scoring, safety judges, and preparation of the reference set, so it measures generator computation rather than total runtime.

## A.8 HYPERPARAMETERS AND CONFIGURATION SELECTION

Table 5 gives the shared settings; Table 6 summarizes the tested hyperparameter ranges by model and parameter; Table 7 lists the selected continuation operating points. The remaining variant settings and results follow in Appendix E. Full SAD does not depend on the hidden-state layer.

Table 5: Shared sampling and guidance settings.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Continuation sampling</td><td>32 new tokens; 32 reverse steps; temperature 1; seeds 42, 43, 44. 128 nominal steps; nominal length 128; temperature 1; seeds 42, 43, 44,</td></tr><tr><td>Jailbreak sampling</td><td>except the explicitly labeled one-seed pilot. DIJA fills existing template spans.</td></tr><tr><td>Unsafe set</td><td>Continuation: up to 2,000 responses from the matching dataset after exact prompt-overlap exclusion. Jailbreak: documented 357-response Wild- Jailbreak set.</td></tr><tr><td>Direction construction</td><td>Clean-reference response means. Later jobs use all valid reference rows; the selected earlier LLaDA/RTP selective row uses the 256-reference di- rection protocol.</td></tr><tr><td>Centered selection</td><td>Strict score threshold  $s _ { k , l } ^ { ( t ) } > 0$  on masked editable positions.</td></tr><tr><td>Remasking</td><td>Off for the core method; top-b uses  $b = 1 ;$  aligned remasking reopens all positive-score revealed positions.</td></tr><tr><td>Warm-up</td><td>Off for selective-only SADDER. Main continuation warm-up:  $w = 8 ,$  followed by selective guidance through step 15.</td></tr><tr><td>Matched controls</td><td>Count-matched random selection or reopening; matrix-control seed zero.</td></tr><tr><td>Search budgets</td><td>BoN and FK with 2, 4, or 8 candidates/particles; BoN-2 composition with SAD-based steering.</td></tr></table>

Table 6: Hyperparameter values considered, consolidated by model. Values are unions across the sweeps, not a single fully crossed grid; a single value denotes a fixed setting. k is the scoring layer, η the SAD strength, w the number of full-SAD warm-up steps, and $[ s , e ]$ the inclusive intervention window.
<table><tr><td>Model</td><td>Parameter</td><td>Values</td></tr><tr><td>LLaDA-Base</td><td>Strength η</td><td>{0, 1, 3, 10, 30, 100, 300, 1000, 3000}</td></tr><tr><td></td><td>Layer k</td><td>{8, 16, 24, 31}</td></tr><tr><td></td><td>Window  $[ s , e ]$ </td><td>[0, 7], [0, 15], [0, 31]</td></tr><tr><td></td><td>Warm-up w</td><td>{8, 16}</td></tr><tr><td></td><td>BoN / FK budget</td><td>{2, 4, 8}</td></tr><tr><td>Dream-Base</td><td>Strength η</td><td>{0, 0.1, 0.3, 1, 3, 10, 100, 3000, 10000, 30000, 100000, 300000}</td></tr><tr><td></td><td>Layer k</td><td>27</td></tr><tr><td></td><td>Window [s, e]</td><td>[0, 7], [0, 15]</td></tr><tr><td></td><td>BoN / FK budget</td><td>{2, 4, 8}</td></tr><tr><td>LLaDA-Instruct</td><td>Strength η</td><td>{1, 3, 10, 30, 100, 300, 1000, 3000, 10000}</td></tr><tr><td></td><td>Layer k</td><td>{9, 16, 24}</td></tr><tr><td></td><td>Window [s, e]</td><td>[0, 15], [0,31], [0,63], [0,127], [32, 63], [64,95], [96, 127]</td></tr><tr><td>Dream-Instruct</td><td>Strength η</td><td>{0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000}</td></tr><tr><td></td><td>Layer k</td><td></td></tr><tr><td></td><td>Window [s, e]</td><td> $\{ 1 , \ldots , 2 8 \}$  [0, 31]</td></tr></table>

Development settings. Base models use RTP: the original search uses indices 0–299; additional strength development uses 64 prompts at indices 900–963, seeds 42/43, and window [0, 7] (k = 31 for LLaDA, 27 for Dream). Instruction-model development uses 100 JBB behaviors; LLaDA additionally uses the first 100 AdvBench and HarmBench behaviors. These runs use seed 42 or seeds 42/43/44, as documented with the results. Sampling uses 32 steps for Base models and 128 nominal steps for Instruct models.

Table 7: Selected main-table continuation settings.
<table><tr><td>Model</td><td>Data</td><td>Method</td><td>k</td><td>η</td><td>Last step</td><td>w</td><td>Reopening</td></tr><tr><td>LLaDA</td><td>RTP</td><td>Full SAD</td><td></td><td>3000</td><td>7</td><td>0</td><td>none</td></tr><tr><td></td><td>RTP</td><td>SADDER (selective)</td><td>31†</td><td>100</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>RTP</td><td>SADDER + warm-up</td><td>24</td><td>3000</td><td>15</td><td>8</td><td>none</td></tr><tr><td>LLaDA</td><td>BT</td><td>Full SAD</td><td>一</td><td>3000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>BT</td><td>SADDER (selective)</td><td>31</td><td>3000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>BT</td><td>SADDER + warm-up</td><td>16</td><td>3000</td><td>15</td><td>8</td><td>none</td></tr><tr><td>Dream</td><td>RTP</td><td>Full SAD</td><td>一</td><td>10000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>RTP</td><td>SADDER (selective)</td><td>27</td><td>10000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>RTP</td><td>SADDER + warm-up</td><td>27</td><td>3000</td><td>15</td><td>8</td><td>none</td></tr><tr><td>Dream</td><td>BT</td><td>Full SAD</td><td>一</td><td>10000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>BT</td><td>SADDER (selective)</td><td>27</td><td>10000</td><td>15</td><td>0</td><td>none</td></tr><tr><td></td><td>BT</td><td>SADDER + warm-up</td><td>27</td><td>10000</td><td>15</td><td>8</td><td>none</td></tr></table>

All windows begin at step zero. BT denotes BeaverTails.

## B IMPLEMENTATION AND EVALUATION DETAILS

Motivation for remasking. The safety improvements reported by CSD-based steering and Diffu-Guard (Lee & Han, 2026; Li et al., 2025) motivate examining remasking as an optional extension to SADDER. Reopening previously generated positions allows guidance to revise token choices that would otherwise remain fixed. Uniform-state diffusion provides a related capability through its native token replacement transitions, allowing repeated revision without an explicit mask token. As such, we wanted to evaluate selective SAD with risk-based remasking and matched random controls in the appendix.

## B.1 OPTIONAL REMASKING AND WARM-UP

Algorithm 2 extends the selective-only main algorithm. Let $\mathcal { R } _ { t } = \{ l \in \mathcal { G } : \pmb { x } _ { t } ^ { l } \neq \mathbb { M } \}$ be the revealed editable positions. The remasking policy r determines the reopened set:

$$
\begin{array} { r } { \mathcal { Q } _ { t } ( r ) = \left\{ \begin{array} { l l } { \emptyset , } & { r = \mathrm { n o n e } , } \\ { \mathrm { T o p } _ { \operatorname* { m i n } ( b , | \mathcal { R } _ { t } | ) } \{ s _ { k , l } ^ { ( t ) } : l \in \mathcal { R } _ { t } \} , } & { r = \mathrm { t o p } \ – b , } \\ { \{ l \in \mathcal { R } _ { t } : s _ { k , l } ^ { ( t ) } > 0 \} , } & { r = \mathrm { a l i g n e d } . } \end{array} \right. } \end{array}\tag{54}
$$

Here Top returns position indices with the largest scores; we use $b = 1$ in the top-b experiments. This ranking policy can reopen a position even if every score is non-positive. Aligned remasking instead reopens every positive-score revealed position, without a top-b cap. After replacing selected tokens by M, the model predictions and hidden states are recomputed. The steering gate is then evaluated on this changed state; reopening does not force a position to be steered. Warm-up length w applies full SAD on admitted denoising steps with index $j < w$ , followed by selective steering for $j \geq w$ . Remasking is disabled during warm-up and on the final sampling step. Outside the intervention set C, neither guidance nor remasking is applied. Setting $w = 0$ disables warm-up; setting r = none disables remasking. Thus the selective-only main method is exactly the case $w = 0 , r = \mathrm { n o n e }$

Algorithm 2 SADDER with optional remasking and full-SAD warm-up.   
1: Input: inputs of Algorithm 1; warm-up length $w \geq 0 ;$ policy $r \in$ {none, top-b, aligned};   
budget $b \geq 0 .$   
2: $w \stackrel { - } { = } 0$ and r = none recover Algorithm 1.   
3: for $j = 0 , \ldots , T - 1$ do   
4: Set $t \gets t _ { j }$ . Evaluate $p _ { \mathcal { D } , t , }$ <sub>·</sub> and $h _ { k , \cdot } ^ { ( t ) }$ together on the current $\mathbf { \nabla } _ { \mathbf { x } _ { t } . }$   
5: Initialize $\widetilde { p } _ { 0 | t , l } \gets p _ { \mathcal { D } , t , l }$ and $a _ { k , l } ^ { ( t ) } \gets 0$ for all l.   
6: if $j \in C$ then   
7: Initialize $\mathcal { Q } _ { t }  \mathcal { O } .$   
8: $\mathbf { i f } \ j \geq$ w and $r \neq$ none and $j < T - 1$ then   
9: $\mathcal { R } _ { t }  \{ l \in \dot { \mathcal { G } } : \mathbf { x } _ { t } ^ { l } \neq \mathbb { M } \}$   
10: Compute $s _ { k , l } ^ { ( t ) } \gets \langle h _ { k , l } ^ { ( t ) } - m _ { k } , \widehat { v } _ { k } \rangle$ for $l \in \mathcal { R } _ { t } .$   
11: Select $\mathcal { Q } _ { t } ( r )$ using Equation equation 54.   
12: Set $\pmb { x } _ { t } ^ { l } \gets \dot { \mathbb { M } }$ for $\bar { l \in { \bar { \mathcal { Q } } } _ { t } } .$   
13: end if   
14: if $\mathcal { Q } _ { t } \neq \mathcal { O }$ then   
15: Recompute $p _ { \mathcal { D } , t , }$ and $h _ { k , \cdot } ^ { ( t ) }$ together on the changed $\mathbf { \nabla } _ { \mathbf { x } _ { t } . }$   
16: Reset $\widetilde { p } _ { 0 | t , \cdot } \gets p _ { \mathcal { D } , t , \cdot }$   
17: end if   
18: $\mathcal { M } _ { t }  \{ l \in \mathcal { G } : \mathbf { x } _ { t } ^ { l } = \mathbb { M } \}$   
19: $\mathbf { i f } \ j < w$ then   
20: Full-SAD warm-up: $a _ { k , l } ^ { ( t ) } \gets 1$ for $l \in \mathcal { M } _ { t }$   
21: else   
22: Compute $s _ { k , l } ^ { ( t ) } \gets \langle h _ { k , l } ^ { ( t ) } - m _ { k } , \widehat { v } _ { k } \rangle$ for $l \in \mathcal { M } _ { t }$   
23: Select risky positions: $a _ { k , l } ^ { ( t ) } \gets \mathbf { 1 } \{ s _ { k , l } ^ { ( t ) } > 0 \}$ for $l \in \mathcal { M } _ { t }$   
24: end if   
25: if at least one gate equals 1 then   
26: Estimate $\widehat { p } _ { \mathcal { D } _ { \mathrm { u n s a f e } } , t } ,$ <sub>·</sub> and $\widehat { \beta } ( \pmb { x } _ { t } )$ on this same state; set $\eta _ { t } \gets \eta \widehat { \beta } ( { \pmb x } _ { t } )$   
27: SAD update on selected positions: update $\widetilde { p } _ { 0 \vert t , l }$ using Equation equation 14 for each l   
with $a _ { k , l } ^ { ( t ) } = 1$   
28: end if   
29: end if   
30: $\pmb { x } _ { t _ { j + 1 } } \gets \mathrm { S o l v e r } ( \pmb { x } _ { t } , t _ { j } , t _ { j + 1 } , \widetilde { p } _ { 0 | t , \cdot } ; \mathcal { G } )$   
31: end for   
32: Return: the generated continuation of ${ \mathbf { } } x _ { t _ { T } } .$

## B.2 METRIC ESTIMATES AND UNCERTAINTY

We compute uncertainty from the saved evaluations. Let $i = 1 , \ldots , n$ index evaluation records and $s = 1 , \ldots , S$ index the observed generation seeds. The main continuation evaluations have $n = 6 0 0$ and $S = 3 ;$ ; the main HarmBench evaluations have $n = 1 0 0$ and $S = 3$

Unsafe rates and other averages. Let $y _ { i , s } \in \{ 0 , 1 \}$ indicate whether the output for prompt record i and seed s is judged unsafe. On a complete, balanced evaluation, the unsafe percentage is

$$
{ \widehat { r } } _ { \mathrm { u n s a f e } } = { \frac { 1 0 0 } { n S } } \sum _ { i = 1 } ^ { n } \sum _ { s = 1 } ^ { S } y _ { i , s } .\tag{55}
$$

The same expression defines HarmBench ASR, refusal rate, or short-output rate when $y _ { i , s }$ is the corresponding event indicator. A mean repetition score uses the per-output repetition score instead. Each metric retains its own evaluation population: for example, XSTest over-refusal uses only safe prompts. When observation counts differ, rate estimates use the total number of valid observations as their denominator.

Quality metrics. Let $\mathcal { T } _ { s }$ contain the valid records for a quality metric at seed s. The reported mean averages a metric computed within each seed:

$$
\widehat { Q } _ { s } = f \left( \frac { \sum _ { i \in \mathcal { T } _ { s } } A _ { i , s } } { \sum _ { i \in \mathcal { T } _ { s } } D _ { i , s } } \right) , \qquad \widehat { Q } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \widehat { Q } _ { s } .\tag{56}
$$

For BERTScore, $A _ { i , s }$ is the response-level F1 score, $D _ { i , s } = 1$ , and $f ( z ) = z ;$ the displayed values are multiplied by 100. For PPL, $A _ { i , s }$ is the total negative log likelihood of scored tokens, $D _ { i , s }$ is their count, and $f ( z ) = \exp ( z )$ . Thus PPL is an average of seed-level corpus perplexities, not an average of response perplexities. Missing or unscorable measurements are excluded; they are not assigned zero.

95% confidence intervals. To estimate uncertainty due to the evaluation prompts, we randomly sample prompts from the evaluated set, allowing the same prompt to be selected more than once. Each resampled set has the same number of prompts as the original set. Whenever a prompt is selected, we keep its responses from all three seeds together. We calculate the metric again for each resampled set and repeat this process 5,000 times. The 2.5th and 97.5th percentiles of these 5,000 estimates form the reported 95% confidence interval.

Seed SD and paired comparisons. For seed-level estimates $\widehat { \theta } _ { s }$ , the reported sample SD is

$$
\mathrm { S D } _ { \mathrm { s e e d } } = \sqrt { \frac { 1 } { S - 1 } \sum _ { s = 1 } ^ { S } ( \widehat { \theta } _ { s } - \bar { \theta } _ { \mathrm { s e e d } } ) ^ { 2 } } , \qquad \bar { \theta } _ { \mathrm { s e e d } } = \frac { 1 } { S } \sum _ { s } \widehat { \theta } _ { s } .\tag{57}
$$

It describes variation across runs and is not a confidence interval; it is unavailable for a single seed. To compare methods A and B, retain matched prompts and seeds and use identical cluster weights for both:

$$
\widehat { \Delta } = \widehat { \theta } _ { \mathrm { A } } - \widehat { \theta } _ { \mathrm { B } } , \qquad \widehat { \Delta } ^ { * ( b ) } = \widehat { \theta } _ { \mathrm { A } } ^ { * ( b ) } - \widehat { \theta } _ { \mathrm { B } } ^ { * ( b ) } .\tag{58}
$$

## C CONTINUATION RESULTS WITH CONFIDENCE INTERVALS

Table 8 reports the main LLaDA continuation estimates with pointwise 95% prompt-cluster bootstrap intervals from 5,000 resamples, retaining all observed seeds within each sampled prompt. These intervals describe prompt uncertainty; the main-table subscripts describe variation across three generation seeds. The selected configurations and evaluation cohorts are identical in both tables.

Table 8: LLaDA-Base safety, generation quality, and cost.
<table><tr><td rowspan="2">Method</td><td colspan="5">RTP</td><td colspan="5">BeaverTails</td></tr><tr><td>Unsafe %↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT↑ NFE</td><td></td><td>Unsafe %↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT↑</td><td>NFE</td></tr><tr><td>Unmodified</td><td>38.6</td><td>21.1 [35.7,41.5] [20.2,22.1] [16.9,18.2] [84.50,84.79]</td><td>17.5</td><td>84.65</td><td>32.0</td><td>38.6</td><td>20.6 [35.6,41.6] [19.6,21.6] [16.9,18.2] [84.50,84.79]</td><td>17.5</td><td>84.65</td><td>32.0</td></tr><tr><td>Full SAD</td><td>32.4</td><td>29.3 [29.4,35.5] [27.6,31.1] [27.7,32.9] [83.75,84.09]</td><td>30.0</td><td>83.92</td><td>32.0</td><td>27.1</td><td>35.1 [24.0,30.4] [33.2,37.0] [42.9,52.7] [83.20,83.55]</td><td>47.1</td><td>83.38</td><td>32.0</td></tr><tr><td>SADDER (selective)</td><td>29.4</td><td>42.1 [26.3,32.5] [37.4,47.8] [17.7,19.2] [84.45,84.74]</td><td>18.4</td><td>84.59</td><td>32.0</td><td>29.9</td><td>34.6 [26.8,33.2] [32.2,37.3] [26.0,30.0] [84.04,84.35]</td><td>27.8</td><td>84.20</td><td>32.0</td></tr><tr><td>SADDER + warm-up</td><td>31.2</td><td>31.6 [28.1,34.5] [29.7,33.5] [38.4,52.5] [82.81,83.20]</td><td>44.3</td><td>83.01</td><td>32.0</td><td>24.4</td><td>36.9 [21.2,27.8] [34.2,39.9] [40.2,49.7] [82.99,83.35]</td><td>44.1</td><td>83.18</td><td>32.0</td></tr><tr><td>CSD†</td><td>22.4 [20.3,24.7]</td><td>225</td><td>21.6</td><td>84.59 [204,249] [20.6,22.7] [84.44,84.73]</td><td>61.9</td><td>15.6 [13.8,17.3]</td><td>487</td><td>943 [440,541] [868,1030] [82.48,82.73]</td><td>82.61</td><td>55.0</td></tr><tr><td>BoN-8</td><td>19.0</td><td>20.5 [16.7,21.5] [19.6,21.4] [18.7,20.2] [84.56,84.84]</td><td>19.5</td><td>84.70</td><td>256.0</td><td>18.6</td><td>24.2 [16.3,20.9] [23.0,25.4] [18.7,20.2] [84.56,84.84]</td><td>19.5</td><td>84.70</td><td>256.0</td></tr><tr><td>FK-8</td><td>20.6</td><td>21.3 [18.1,23.0] [20.4,22.3] [17.5,18.8] [84.57,84.85]</td><td>18.2</td><td>84.71</td><td>256.0</td><td>28.2</td><td>22.7 [25.4,30.9] [21.7,23.7] [17.5,18.8] [84.57,84.85]</td><td>18.2</td><td>84.71</td><td>256.0</td></tr></table>

## D DREAM-BASE CONTINUATION RESULTS

Table 9 reports the complete Dream-Base comparison under the same continuation protocol as LLaDA (Section 5.1). Unsafe rate and harmful PPL use the RTP or BeaverTails harmful cohort; benign PPL and BERTScore use the separate benign RTP cohort. Entries are means with 95% prompt-cluster intervals; NFE measures generator cost.

Table 9: Dream-Base safety, generation quality, and cost.
<table><tr><td rowspan="2">Method</td><td colspan="5">RTP</td><td colspan="5">BeaverTails</td></tr><tr><td>Unsafe %↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT↑ NFE</td><td></td><td>Unsafe %↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT↑ NFE</td><td></td></tr><tr><td>Unmodified</td><td>46.8 [43.8,49.7]</td><td>18.3 [17.6,19.1] [15.0,16.0] [84.57,84.86]</td><td>15.5</td><td>84.71</td><td>32.0</td><td>38.7</td><td>12.7 [35.8,41.6] [12.2,13.2] [15.0,16.0] [84.57,84.86]</td><td>15.5</td><td>84.71</td><td>32.0</td></tr><tr><td>Full SAD</td><td>33.4 [30.1,36.8]</td><td>26.0 [24.3,27.8]</td><td>23.3 [22.0,24.7][82.96,83.35]</td><td>83.16</td><td>32.0</td><td>22.7</td><td>23.7 [19.7,25.7] [21.8,25.8] [21.7,24.5] [83.44,83.80]</td><td>23.0</td><td>83.62</td><td>32.0</td></tr><tr><td>SADDER (selective)</td><td>35.9 [32.8,39.0]</td><td>29.1 [27.5,30.7]</td><td>18.1 [17.5,18.7] [84.45,84.73]</td><td>84.59</td><td>32.0</td><td>34.1</td><td>19.3 [31.3,36.8] [18.4,20.3] [19.2,20.7] [84.34,84.64]</td><td>20.0</td><td>84.49</td><td>32.0</td></tr><tr><td>SADDER + warm-up</td><td>34.7 [31.6,38.2]</td><td>27.1 [25.5,28.8]</td><td>21.3 [20.3,22.3] [83.57,83.94]</td><td>83.76</td><td>32.0</td><td>27.7</td><td>23.6 [24.8,30.8] [22.1,25.1][19.6,21.5] [84.06,84.37]</td><td>20.5</td><td>84.21</td><td>32.0</td></tr><tr><td>CSD†</td><td>32.9</td><td>1180 [30.6,35.4] [1100,1270] [3270,3540] [81.42,81.61]</td><td>3400</td><td>81.51</td><td>60.2</td><td>38.6</td><td>17.8 [35.7,41.4] [16.7,19.1] [25.3,29.4] [84.36,84.64]</td><td>27.2</td><td>84.50</td><td>55.8</td></tr><tr><td>BoN-8</td><td>24.6 [22.0,27.3]</td><td>17.5 [16.7,18.2]</td><td>16.9 [16.3,17.5] [84.68,84.95]</td><td>84.81</td><td>256.0</td><td>20.0</td><td>15.0 [17.8,22.2] [14.3,15.7] [16.3,17.5] [84.68,84.95]</td><td>16.9</td><td>84.81</td><td>256.0</td></tr><tr><td>FK-8</td><td>25.1</td><td>17.9 [22.5,27.7] [17.2,18.6]</td><td>15.7</td><td>84.76 [15.3,16.2] [84.62,84.90]</td><td>256.0</td><td>27.8</td><td>13.9 [25.2,30.4] [13.4,14.5] [15.3,16.2] [84.62,84.90]</td><td>15.7</td><td>84.76</td><td>256.0</td></tr></table>

## E CONTINUATION SETTINGS AND COMPONENT VARIANTS

Settings are selected by unsafe rate; benign quality is evaluated afterward. Tables 10 and 11 report layer, strength, and the last admitted step (all windows begin at zero). The warm-up variant uses full SAD on steps 0–7 and selective guidance on 8–15, with aligned remasking on LLaDA and no remasking on Dream. Equation equation 54 defines the alternative reopening policies.

Table 10: LLaDA selected continuation settings on development set.
<table><tr><td>Dataset</td><td>Method</td><td>Setting</td></tr><tr><td rowspan="7">RTP</td><td>Full SAD</td><td>31/3000/7</td></tr><tr><td>SADDER</td><td>31/100/15</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>24/100/15</td></tr><tr><td>SADDER w/ aligned remasking</td><td>16/100/15</td></tr><tr><td>SADDER + warm-up + aligned remasking 24/3000/15;</td><td></td></tr><tr><td>CSD† BoN-8</td><td>k = 31, j = 0: 15</td></tr><tr><td>FK-8</td><td>N = 8 N = 8</td></tr><tr><td>BeaverTails Full SAD</td><td></td><td>31/3000/15</td></tr><tr><td rowspan="7"></td><td></td><td></td></tr><tr><td>SADDER</td><td>31/3000/15</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>24/100/7</td></tr><tr><td>SADDER w/ aligned remasking</td><td>16/3000/15</td></tr><tr><td>SADDER + warm-up + aligned remasking 16/3000/15; aligned</td><td></td></tr><tr><td>CSD†</td><td> $k = 2 4 , j = 0 \colon 7$ </td></tr><tr><td>BoN-8 FK-8</td><td>N = 8 N = 8</td></tr></table>

Table 11: Dream selected continuation settings on development set.
<table><tr><td>Dataset</td><td>Method</td><td>Setting</td></tr><tr><td rowspan="9">RTP</td><td>Full SAD</td><td>27/10000/15</td></tr><tr><td>SADDER</td><td>27/10000/15</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>27/100/15</td></tr><tr><td>SADDER w/ aligned remasking</td><td>24/1e5/15</td></tr><tr><td>SADDER + warm-up</td><td>27/3000/15; none</td></tr><tr><td>CSD†</td><td>k = 24, j = 0: 15</td></tr><tr><td>BoN-8</td><td>N = 8</td></tr><tr><td>FK-8</td><td>N = 8</td></tr><tr><td>BeaverTails Full SAD</td><td>27/10000/15</td></tr><tr><td rowspan="9"></td><td>SADDER</td><td>27/10000/15</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>27/100/15</td></tr><tr><td>SADDER w/ aligned remasking</td><td>27/100/15</td></tr><tr><td></td><td></td></tr><tr><td>SADDER + warm-up</td><td>27/10000/15; none</td></tr><tr><td>CSD†</td><td>k = 27, j = 0: 15</td></tr><tr><td>BoN-8</td><td>N = 8</td></tr><tr><td>FK-8</td><td>N = 8</td></tr></table>

SADDER variants. These tables extend the main comparison with top-1 and aligned remasking, using the same metrics and uncertainty conventions.

Table 12: LLaDA-Base component variants.
<table><tr><td>Method</td><td>Unsafe % ↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT ↑</td><td>NFE</td></tr><tr><td>RTP</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td>38.6</td><td>21.14</td><td>17.53 [16.90, 18.21]</td><td>84.65 [84.50, 84.79]</td><td>32.0</td></tr><tr><td></td><td>[35.7, 41.5] 32.4</td><td>[20.21, 22.13] 29.30</td><td>29.96</td><td>83.92</td><td></td></tr><tr><td>Full SAD</td><td>[29.4,35.5] 29.4</td><td>[27.61, 31.13] 42.08</td><td>[27.65, 32.89] 18.41</td><td>[83.75, 84.09] 84.59</td><td>32.0</td></tr><tr><td>SADDER (selective)</td><td>[26.3, 32.5] 28.3</td><td>[37.39, 47.75] 43.25</td><td>[17.70, 19.17] 41.75</td><td>[84.45,84.74] 83.38</td><td>32.0</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>[25.2, 31.3] 30.6</td><td>[38.63, 48.89] 35.81</td><td>[37.34, 47.13] 35.88</td><td>[83.17,83.58] 83.99</td><td>47.0</td></tr><tr><td>SADDER w/ aligned remasking</td><td>[27.8, 33.4] 27.2</td><td>[33.47, 38.38] 38.75</td><td>[33.53, 38.57] 33.33</td><td>[83.82, 84.16] 84.13</td><td>46.6</td></tr><tr><td>SADDER + warm-up + aligned remasking</td><td>[24.7, 29.9] 22.4</td><td>[36.22, 41.88] 225.01</td><td>[31.08, 35.91] 21.59</td><td>[83.96, 84.28] 84.59</td><td>40.0</td></tr><tr><td>CSD†</td><td>[20.3, 24.7] 19.0</td><td>[204.37, 248.84] 20.46</td><td>[20.58, 22.67] 19.47</td><td>[84.44, 84.73] 84.70</td><td>61.9</td></tr><tr><td>BoN-8</td><td>[16.7, 21.5] 20.6</td><td>[19.57, 21.39] 21.29</td><td>[18.74, 20.21] 18.17</td><td>[84.56, 84.84] 84.71</td><td>256.0</td></tr><tr><td>FK-8</td><td>[18.1, 23.0]</td><td>[20.37, 22.27]</td><td>[17.53, 18.84]</td><td>[84.57, 84.85]</td><td>256.0</td></tr><tr><td>BeaverTails</td><td>38.6</td><td>20.57</td><td>17.53</td><td>84.65</td><td></td></tr><tr><td>Unmodified</td><td>[35.6, 41.6] 27.1</td><td>[19.61, 21.57] 35.05</td><td>[16.90, 18.21] 47.08</td><td>[84.50, 84.79] 83.38</td><td>32.0</td></tr><tr><td>Full SAD</td><td>[24.0, 30.4] 29.9</td><td>[33.23, 37.03] 34.63</td><td>[42.86, 52.67] 27.80</td><td>[83.20, 83.55] 84.20</td><td>32.0</td></tr><tr><td>SADDER (selective)</td><td>[26.8, 33.2] 30.6</td><td>[32.22, 37.33] 31.58</td><td>[25.95, 30.01] 23.78</td><td>[84.04, 84.35] 84.10</td><td>32.0</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>[27.1,34.1] 27.5</td><td>[28.46, 35.55] 37.12</td><td>[22.34, 25.39] 35.91</td><td>[83.90, 84.29] 83.69</td><td>39.0</td></tr><tr><td>SADDER w/ aligned remasking</td><td>[24.4, 30.8] 24.7</td><td>[33.51, 41.64] 31.36</td><td>[33.64, 38.41] 42.23</td><td>[83.50, 83.86] 83.65</td><td>42.2</td></tr><tr><td>SADDER + warm-up + aligned remasking</td><td>[21.8,27.7] 15.6</td><td>[29.70, 33.08] 486.52</td><td>[39.06, 46.69] 942.73</td><td>[83.47, 83.82]</td><td>37.8</td></tr><tr><td>CSD†</td><td>[13.8, 17.3]</td><td>[439.90, 541.43]</td><td>[868.26, 1027.02]</td><td>82.61 [82.48, 82.73]</td><td>55.0</td></tr><tr><td>BoN-8</td><td>18.6 [16.3, 20.9]</td><td>24.23 [23.04, 25.42]</td><td>19.47 [18.74, 20.21]</td><td>84.70 [84.56, 84.84]</td><td></td></tr><tr><td>FK-8</td><td>28.2 [25.4, 30.9]</td><td>22.66 [21.66, 23.74]</td><td>18.17 [17.53, 18.84]</td><td>84.71 [84.57, 84.85]</td><td>256.0 256.0</td></tr></table>

Table 13: Dream-Base component variants.
<table><tr><td>Method</td><td>Unsafe % ↓</td><td>Harmful PPL↓</td><td>Benign PPL↓</td><td>Benign BERT↑</td><td>NFE</td></tr><tr><td>RTP</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td>46.8 [43.8, 49.7]</td><td>18.31 [17.61, 19.06]</td><td>15.45 [14.96, 15.98]</td><td>84.71 [84.57, 84.86]</td><td>32.0</td></tr><tr><td></td><td>33.4</td><td>26.02</td><td>23.31</td><td>83.16</td><td>32.0</td></tr><tr><td>Full SAD</td><td>[30.1, 36.8] 35.9</td><td>[24.30, 27.83] 29.09</td><td>[21.99, 24.68] 18.10</td><td>[82.96, 83.35] 84.59</td><td></td></tr><tr><td>SADDER</td><td>[32.8, 39.0] 37.4</td><td>[27.54, 30.68] 22.54</td><td>[17.48, 18.73] 16.92</td><td>[84.45, 84.73] 84.57</td><td>32.0</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>[34.3, 40.5] 40.9</td><td>[21.28, 23.80] 23.06</td><td>[16.37, 17.53] 17.53</td><td>[84.43, 84.71] 84.52</td><td>47.0</td></tr><tr><td>SADDER w/ aligned remasking</td><td>[38.0, 43.7] 34.7</td><td>[22.15, 24.00] 27.11</td><td>[16.92, 18.16] 21.26</td><td>[84.37, 84.66] 83.76</td><td>44.8</td></tr><tr><td>SADDER + warm-up</td><td>[31.6, 38.2] 32.9</td><td>[25.52, 28.82] 1182.90</td><td>[20.25, 22.29] 3397.41</td><td>[83.57, 83.94] 81.51</td><td>32.0</td></tr><tr><td>CSD†</td><td>[30.6, 35.4]</td><td>[1101.51, 1271.65]</td><td>[3266.61, 3538.51]</td><td>[81.42,81.61]</td><td>60.2</td></tr><tr><td>BoN-8</td><td>24.6 [22.0, 27.3]</td><td>17.46 [16.72, 18.18]</td><td>16.87 [16.30, 17.46]</td><td>84.81</td><td></td></tr><tr><td>FK-8</td><td>25.1</td><td>17.85</td><td>15.73</td><td>[84.68, 84.95] 84.76</td><td>256.0</td></tr><tr><td></td><td>[22.5, 27.7]</td><td>[17.16, 18.57]</td><td>[15.27, 16.24]</td><td>[84.62, 84.90]</td><td>256.0</td></tr><tr><td>BeaverTails</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td>38.7 [35.8, 41.6]</td><td>12.67 [12.17, 13.21]</td><td>15.45 [14.96, 15.98]</td><td>84.71 [84.57, 84.86]</td><td>32.0</td></tr><tr><td>Full SAD</td><td>22.7</td><td>23.68</td><td>23.03</td><td>83.62</td><td></td></tr><tr><td></td><td>[19.7, 25.7] 34.1</td><td>[21.80, 25.75] 19.32</td><td>[21.69, 24.45] 19.95</td><td>[83.44, 83.80]</td><td>32.0</td></tr><tr><td>SADDER</td><td>[31.3, 36.8]</td><td>[18.41, 20.32]</td><td>[19.18,20.73]</td><td>84.49</td><td></td></tr><tr><td></td><td>34.3</td><td>19.16</td><td>18.35</td><td>[84.34, 84.64]</td><td>32.0</td></tr><tr><td>SADDER w/ top-1 remasking</td><td>[31.5, 37.1]</td><td></td><td></td><td>84.39</td><td></td></tr><tr><td></td><td></td><td>[18.23, 20.15]</td><td>[17.63, 19.09]</td><td>[84.24, 84.55]</td><td>47.0</td></tr><tr><td>SADDER w/ aligned remasking</td><td>33.7</td><td>17.78</td><td>21.07</td><td>84.54</td><td></td></tr><tr><td></td><td>[31.0, 36.5]</td><td>[16.93, 18.65]</td><td>[20.28, 21.92]</td><td>[84.39, 84.68]</td><td>40.4</td></tr><tr><td></td><td>27.7</td><td>23.56</td><td>20.52</td><td>84.21</td><td></td></tr><tr><td>SADDER + warm-up</td><td>[24.8, 30.8]</td><td>[22.08,25.14]</td><td>[19.60, 21.49]</td><td>[84.06, 84.37]</td><td>32.0</td></tr><tr><td></td><td>38.6</td><td>17.81</td><td>27.19</td><td>84.50</td><td></td></tr><tr><td>CSD†</td><td>[35.7, 41.4]</td><td>[16.65, 19.13]</td><td>[25.30, 29.38]</td><td></td><td>55.8</td></tr><tr><td></td><td>20.0</td><td></td><td></td><td>[84.36, 84.64]</td><td></td></tr><tr><td>BoN-8</td><td></td><td>14.95</td><td>16.87</td><td>84.81</td><td></td></tr><tr><td></td><td>[17.8, 22.2]</td><td>[14.26, 15.68]</td><td>[16.30, 17.46]</td><td>[84.68, 84.95]</td><td>256.0</td></tr><tr><td>FK-8</td><td>27.8 [25.2, 30.4]</td><td>13.94 [13.39, 14.54]</td><td>15.73 [15.27, 16.24]</td><td>84.76 [84.62, 84.90]</td><td>256.0</td></tr></table>

![](images/8b3a4f0c8ad2fccab883015e34e7ec944b3336a7ee25e7f2926b44199ec2409f.jpg)

## F PAIRED COMPARISONS AND COMPONENT DIAGNOSTICS

The following figure and benign-quality table compare the selected component configurations; their LLaDA warm-up entries include aligned remasking. Paired differences compare SADDER with the separately selected full-SAD operating point on identical prompt–seed observations. Negative unsafe-rate and PPL differences, and positive BERTScore differences, favour SADDER. Quality comparisons use common valid pairs within each cohort. Intervals are conditional on the selected settings and observed seeds.

Figure 2: Selected SADDER component configurations minus full SAD on 600 harmful prompts per dataset and three seeds. LLaDA warm-up includes its aligned remasking, Dream does not. Points and bars show unsafe-rate differences (pp) and pointwise 95% paired bootstrap CIs; negative favors SADDER.  
Table 14: SADDER variants: paired benign-quality changes versus full SAD.
<table><tr><td>Model/data</td><td>Method</td><td>∆ benign PPL</td><td>∆ benign BERT</td><td></td></tr><tr><td>LLaDA/RTP</td><td>SADDER (selective)</td><td>-11.55 [−14.49, −9.26]</td><td></td><td>0.672 [0.527, 0.813]</td></tr><tr><td></td><td>SADDER + warm-up</td><td>3.42 [0.49, 6.19]</td><td></td><td>0.206 [0.076, 0.332]</td></tr><tr><td>LLaDA/BT</td><td>SADDER (selective)</td><td>-19.28 [-24.82, −15.03]</td><td></td><td>0.822 [0.681, 0.967]</td></tr><tr><td></td><td>SADDER + warm-up</td><td>-4.85 [−8.89, -1.18]</td><td></td><td>0.270 [0.133, 0.407]</td></tr><tr><td>Dream/RTP</td><td>SADDER (selective)</td><td>-5.22 [−6.55, -3.88]</td><td></td><td>1.429 [1.258, 1.605]</td></tr><tr><td rowspan="2">Dream/BT</td><td>SADDER + warm-up</td><td>-2.06 [−3.25, −0.91]</td><td></td><td>0.596 [0.447,0.740]</td></tr><tr><td>SADDER (selective)</td><td>-3.08 [−4.51, -1.70]</td><td></td><td>0.868 [0.709, 1.030]</td></tr><tr><td></td><td>SADDER + warm-up</td><td>-2.51 [−3.66, −1.38]</td><td></td><td>0.594 [0.482, 0.703]</td></tr></table>

Paired random controls. Tables 15 and 16 compare each selected SADDER component configuration with full SAD and random controls at the same layer, strength, and intervention window. LLaDA warm-up here includes aligned remasking. Random selection replaces the steering gate with a random subset of the gate-implied size, preserving the remasking and warm-up policies. Random reopening replaces top-1 remasking with one random position, or aligned remasking with a random subset of the threshold-implied size, preserving steering and warm-up. Counts are computed on each trajectory’s current state and may diverge across trajectories. The full-SAD comparator steers all masked positions throughout the window, without additional remasking or a selective phase. Differences are SADDER minus the comparator; negative values favor SADDER. Dashes indicate that a matched control is unavailable or reopening is not used. Each interval uses 5,000 paired prompt-cluster resamples of the same 600 prompt records and three seeds, conditional on the selected configuration.

Table 15: LLaDA-Base: paired unsafe-rate differences (pp), means [95% CI].
<table><tr><td>SADDER variant</td><td></td><td>vs full SAD vs random selection vs random reopening</td><td></td></tr><tr><td>RTP</td><td></td><td></td><td></td></tr><tr><td>Selective + top-1 remasking</td><td>-4.1 [−6.1, −2.1] −3.3 [−5.4, −1.3] -5.2 [−8.5, −1.7]</td><td></td><td> $- 3 . 4 \left[ - 6 . 3 , - 0 . 6 \right]$ </td></tr><tr><td>+ aligned remasking + warm-up + aligned</td><td>-2.9 [−6.3, 0.4]</td><td></td><td> $- 1 . 7 \ [ - 3 . 5 , 0 . 1 ]$   $- 4 . 8 \left[ - 6 . 9 , - 2 . 8 \right]$ </td></tr><tr><td>BeaverTails</td><td>-5.8 [−9.0, −2.7]</td><td></td><td></td></tr><tr><td>Selective</td><td>2.8 [−1.1, 6.6]</td><td>-0.7 [−3.9, 2.4]</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>+ top-1 remasking</td><td>-7.9 [−11.8, -4.1]</td><td></td><td> $- 6 . 1 \ [ - 9 . 1 , - 3 . 1 ]$ </td></tr><tr><td>+ aligned remasking</td><td> $0 . 4 \ [ - 3 . 3 , 4 . 1 ]$ </td><td></td><td> $1 . 9 \ [ - 0 . 6 , 4 . 6 ]$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>+ warm-up + aligned</td><td>−2.4 [−5.4, 0.3]</td><td></td><td>-1.8 [−4.5, 0.8]</td></tr></table>

Table 16: Dream-Base: paired unsafe-rate differences (pp), means [95% CI].
<table><tr><td>SADDER variant</td><td></td><td></td><td></td><td>vs full SAD vs random selection vs random reopening</td></tr><tr><td>RTP</td><td></td><td></td><td></td><td></td></tr><tr><td>Selective</td><td>2.5 [−0.3, 5.3]</td><td></td><td>-2.7 [−5.5, 0.1]</td><td></td></tr><tr><td>+ top-1 remasking</td><td>-3.9 [−7.4, −0.4]</td><td></td><td>-2.6 [−5.3, 0.3]</td><td>-2.8 [−5.7, 0.2]</td></tr><tr><td>+ aligned remasking −3.4 [−6.2, −0.6]</td><td></td><td></td><td>0.6 [−1.7, 2.7]</td><td>−2.2 [−4.5, 0.2]</td></tr><tr><td>+ warm-up</td><td>0.3 [−1.6, 2.2]</td><td></td><td>0.4 [−1.4, 2.3]</td><td></td></tr><tr><td>BeaverTails</td><td></td><td></td><td></td><td></td></tr><tr><td>Selective</td><td>11.4 [7.8, 14.8]</td><td></td><td>3.4 [0.8, 5.9]</td><td></td></tr><tr><td>+ top-1 remasking</td><td>4.6 [1.1, 8.0]</td><td></td><td>5.3 [2.7, 7.9]</td><td>-1.6 [−4.5, 1.2]</td></tr><tr><td>+ aligned remasking</td><td>4.0 [0.6, 7.4]</td><td></td><td>2.6 [0.1, 5.1]</td><td>0.2 [−2.1, 2.5]</td></tr><tr><td>+ warm-up</td><td>5.1 [2.6, 7.5]</td><td></td><td>0.1 [−1.7, 1.8]</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## G BON-2/4/8, FK-2/4/8, AND COMPOSITION

We compare BoN and FK at budgets 2, 4, and 8, with harmful and benign quality measured on the corresponding cohorts. The FK adaptation resamples every four steps, using one clean-state draw per particle and a difference potential on log safety rewards. RoBERTa supplies the search reward; LlamaGuard judges final outputs. NFE excludes reward scoring.

Table 17: LLaDA: BoN and FK at budgets 2, 4, and 8.
<table><tr><td>Method</td><td>Unsafe %</td><td>Harmful PPL</td><td></td><td>Benign PPL Benign BERT</td><td>NFE</td></tr><tr><td colspan="6">RTP</td></tr><tr><td>BoN-2</td><td>30.6 [27.8, 33.4]</td><td>20.90 [20.00, 21.82]</td><td>17.51 [16.89, 18.15]</td><td>84.73 [84.59, 84.88]</td><td>64</td></tr><tr><td>BoN-4</td><td>24.7 [22.3, 27.2]</td><td>20.50 [19.68, 21.40]</td><td>18.70 [18.00, 19.41]</td><td>84.75 [84.61, 84.89]</td><td>128</td></tr><tr><td></td><td>19.0</td><td>20.46</td><td>19.47</td><td>84.70 [84.56, 84.84]</td><td></td></tr><tr><td>BoN-8 FK-2</td><td>[16.7, 21.5] 32.4</td><td>[19.57, 21.39] 21.47</td><td>[18.74, 20.21] 17.59</td><td>84.67 [84.52, 84.82]</td><td>256</td></tr><tr><td>FK-4</td><td>[29.7, 35.2] 26.2</td><td>[20.53, 22.46] 20.82 [19.92, 21.71]</td><td>[16.96, 18.26] 17.84</td><td>84.68</td><td>64</td></tr><tr><td>FK-8</td><td>[23.6, 28.8] 20.6 [18.1, 23.0]</td><td>21.29 [20.37, 22.27]</td><td>[17.22, 18.49] 18.17 [17.53, 18.84]</td><td>[84.54, 84.83] 84.71 [84.57,84.85]</td><td>128 256</td></tr><tr><td colspan="6">BeaverTails</td></tr><tr><td>BoN-2</td><td>32.1 [29.3, 34.8]</td><td>22.32 [21.38, 23.33] 23.76</td><td>17.51 [16.89, 18.15]</td><td>84.73 [84.59, 84.88]</td><td>64</td></tr><tr><td>BoN-4</td><td>24.1 [21.7, 26.6]</td><td>[22.68, 24.93] 24.23</td><td>18.70 [18.00, 19.41] 19.47</td><td>84.75 [84.61, 84.89] 84.70</td><td>128</td></tr><tr><td>BoN-8</td><td>18.6 [16.3, 20.9] 36.2</td><td>[23.04, 25.42] 21.24</td><td>[18.74, 20.21] 17.59</td><td>[84.56, 84.84] 84.67</td><td>256</td></tr><tr><td>FK-2</td><td>[33.3, 39.2] 34.6</td><td>[20.26, 22.26] 21.85</td><td>[16.96, 18.26] 17.84</td><td>[84.52, 84.82] 84.68</td><td>64</td></tr><tr><td>FK-4</td><td>[31.8, 37.4] 28.2</td><td>[20.88, 22.94]</td><td>[17.22, 18.49] 18.17</td><td>[84.54, 84.83] 84.71</td><td>128</td></tr><tr><td>FK-8</td><td>[25.4, 30.9]</td><td>22.66 [21.66, 23.74]</td><td>[17.53, 18.84]</td><td>[84.57, 84.85]</td><td>256</td></tr></table>

Table 18: Dream: BoN and FK at budgets 2, 4, and 8.
<table><tr><td>Method</td><td>Unsafe %</td><td>Harmful PPL</td><td></td><td>Benign PPL Benign BERT</td><td>NFE</td></tr><tr><td colspan="6">RTP</td></tr><tr><td>BoN-2</td><td>37.0 [34.2, 39.9]</td><td>17.67 [17.01, 18.35]</td><td>15.39 [14.90, 15.90]</td><td>84.76 [84.62, 84.91]</td><td>64</td></tr><tr><td>BoN-4</td><td>30.6 [27.9, 33.3]</td><td>17.95 [17.27, 18.64]</td><td>15.78 [15.29, 16.29]</td><td>84.76 [84.62, 84.90]</td><td>128</td></tr><tr><td>BoN-8</td><td>24.6 [22.0, 27.3]</td><td>17.46 [16.72, 18.18]</td><td>16.87 [16.30, 17.46]</td><td>84.81 [84.68, 84.95]</td><td>256</td></tr><tr><td>FK-2</td><td>39.7 [36.8, 42.7]</td><td>17.90 [17.23, 18.56]</td><td>15.40 [14.92, 15.90]</td><td>84.69 [84.54, 84.83]</td><td>64</td></tr><tr><td>FK-4</td><td>31.3 [28.6, 33.9]</td><td>17.90 [17.20, 18.61]</td><td>15.31 [14.83, 15.83]</td><td>84.72 [84.57, 84.86]</td><td>128</td></tr><tr><td>FK-8</td><td>25.1 [22.5, 27.7]</td><td>17.85 [17.16, 18.57]</td><td>15.73 [15.27, 16.24]</td><td>84.76 [84.62,84.90]</td><td>256</td></tr><tr><td colspan="6">BeaverTails</td></tr><tr><td>BoN-2</td><td>32.5 [29.8, 35.1]</td><td>12.81 [12.27, 13.38]</td><td>15.39 [14.90, 15.90]</td><td>84.76 [84.62, 84.91]</td><td>64</td></tr><tr><td>BoN-4</td><td>26.3 [23.7, 28.8]</td><td>13.58 [13.00, 14.22] 14.95</td><td>15.78 [15.29, 16.29] 16.87</td><td>84.76 [84.62,84.90] 84.81</td><td>128</td></tr><tr><td>BoN-8</td><td>20.0 [17.8, 22.2]</td><td>[14.26, 15.68]</td><td>[16.30, 17.46]</td><td>[84.68, 84.95] 84.69</td><td>256</td></tr><tr><td>FK-2</td><td>36.3 [33.4, 39.1]</td><td>13.21 [12.63, 13.81]</td><td>15.40 [14.92, 15.90]</td><td>[84.54, 84.83]</td><td>64</td></tr><tr><td>FK-4</td><td>32.7 [29.9, 35.4]</td><td>13.76 [13.20, 14.37]</td><td>15.31 [14.83, 15.83]</td><td>84.72 [84.57, 84.86]</td><td>128</td></tr><tr><td></td><td>27.8</td><td>13.94</td><td>15.73</td><td>84.76</td><td></td></tr><tr><td>FK-8</td><td>[25.2, 30.4]</td><td>[13.39, 14.54]</td><td>[15.27, 16.24]</td><td>[84.62,84.90]</td><td>256</td></tr></table>

Composition with BoN. SADDER can guide each candidate before BoN-2 selection. In the LLaDA/RTP run, this reduces unsafe rate from 30.61% to 26.7% at the same 64 generator evaluations. Its unsafe-prompt PPL increases from 20.90 to 28.80. The two tables distinguish separate composition studies. PPL is evaluated on harmful prompts; benign quality was not measured for these hybrids.

Table 19: Composition.
<table><tr><td>Model/data</td><td>Method</td><td></td><td>Unsafe % Unsafe-prompt PPL NFE</td><td></td></tr><tr><td>Dream/BT</td><td>BoN-2</td><td>32.5 [29.8, 35.1]</td><td>12.81 [12.27, 13.38]</td><td>64</td></tr><tr><td rowspan="2">Dream/RTP</td><td>BoN-2</td><td>37.0 [34.2, 39.9]</td><td>17.67 [17.01, 18.35]</td><td>64</td></tr><tr><td>SADDER + BoN-2 34.0 [30.9, 37.1]</td><td></td><td>21.05 [19.98, 22.11]</td><td>64</td></tr><tr><td rowspan="2">LLaDA/BT</td><td>BoN-2</td><td>32.1 [29.3, 34.8]</td><td>22.32 [21.38, 23.33]</td><td>64</td></tr><tr><td>SADDER + BoN-2 28.5 [25.3, 31.7]</td><td></td><td>27.54 [25.89, 29.39]</td><td>64</td></tr><tr><td rowspan="2">LLaDA/RTP BoN-2</td><td></td><td>30.6 [27.8, 33.4]</td><td>20.90 [20.00, 21.82]</td><td>64</td></tr><tr><td>SADDER + BoN-2 26.7 [23.7, 29.8]</td><td></td><td>28.80 [26.80, 31.41]</td><td>64</td></tr></table>

## H JAILBREAK ROBUSTNESS AND ABLATIONS

## H.1 LLADA-INSTRUCT JAILBREAK RESULTS

Table 20 gives the LLaDA-Instruct comparison corresponding to the Dream-Instruct results in Table 2. Both tables use the same metrics and reporting conventions. XSTest uses the DIJA-configured defense; for LLaDA generation-scope DiffuGuard, this is $\lambda = 0 . 3 5$

Table 20: LLaDA-Instruct: held-out HarmBench attacks and XSTest over-refusal. Entries use $A _ { b } \colon$ mean and sample SD across three seeds (percentage points for rates). HB, LG, and PPL assess complete responses. XSTest reports full-response WG refusal on safe prompts, using valid judge labels only (coverage in Appendix H.2). NFE is mean cost.
<table><tr><td rowspan="2">Method</td><td colspan="4">DIJA</td><td colspan="4">PAD</td><td>XSTest</td></tr><tr><td>HB ASR LG unsafe  $\% \downarrow$ </td><td> $\% \downarrow$ </td><td>Harmful  $\mathrm { P P L } \downarrow$ </td><td>NFE</td><td> $\% \downarrow$ </td><td> $\% \downarrow$ </td><td>HB ASR LG unsafe Harmful  $\mathrm { P P L } \downarrow$ </td><td>NFE</td><td>WG refusal  $\% \downarrow$ </td></tr><tr><td>LLaDA-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td> $7 8 . 7 _ { 3 . 1 }$ </td><td> $8 2 . 0 _ { 2 . 6 }$ </td><td> $9 . 3 4 _ { 0 . 1 3 }$ </td><td>91.7</td><td> $5 8 . 7 _ { 2 . 5 }$ </td><td> $7 5 . 7 _ { 2 . 3 }$ </td><td> $6 . 0 6 _ { 0 . 1 7 }$ </td><td>116.0</td><td> $2 0 . 7 _ { 1 . 1 }$ </td></tr><tr><td>Full SAD</td><td> $7 7 . 0 _ { 2 . 0 }$ </td><td> $8 1 . 0 _ { 0 . 0 }$ </td><td> $1 1 . 0 \phantom { 0 } . 1 3 \phantom { 0 }$ </td><td>91.7</td><td> $6 3 . 3 \mathrm { _ 7 . 5 }$ </td><td> $7 8 . 7 _ { 4 . 0 }$ </td><td> $8 . 2 4 _ { 0 . 2 1 }$ </td><td>116.0</td><td> $2 0 . 9 _ { 1 . 9 }$ </td></tr><tr><td>SADDER (selective)</td><td> $7 9 . 0 _ { 1 . 0 }$ </td><td> $7 9 . 3 _ { 1 . 5 }$ </td><td> $1 3 . 3 _ { 0 . 0 7 6 }$ </td><td>91.7</td><td> $5 7 . 7 _ { 4 . 0 }$ </td><td> $7 3 . 7 _ { 3 . 8 }$ </td><td> $1 3 . 2 _ { 0 . 2 6 }$ </td><td>116.0</td><td> $1 8 . 8 \phantom { 0 } 0 . 6$ </td></tr><tr><td>SADDER + warm-up</td><td> $7 8 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $8 0 . 7 _ { 0 . 6 }$ </td><td> $1 2 . 8 \mathrm { _ 0 . 1 9 }$ </td><td>91.7</td><td> $5 7 . 0 _ { 4 . 0 }$ </td><td> $6 8 . 3 _ { 2 . 5 }$ </td><td> $1 1 . 4 _ { 0 . 1 9 }$ </td><td>116.0</td><td> $1 6 . 3 _ { 2 . 0 }$ </td></tr><tr><td>CSD†</td><td> $4 8 . 7 _ { 4 . 7 }$ </td><td> $7 9 . 7 _ { 4 . 9 }$ </td><td> $3 6 . 1 _ { 0 . 9 2 }$ </td><td>115.7</td><td> $3 0 . 3 { } _ { 3 . 5 }$ </td><td> $6 9 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $1 9 . 7 _ { 1 . 3 }$ </td><td>142.0</td><td> $2 9 . 7 _ { 1 . 4 }$ </td></tr><tr><td>DiffuGuard (full)</td><td> $4 2 . 7 _ { 4 . 5 }$ </td><td> $8 1 . 7 _ { 2 . 5 }$ </td><td> $2 5 5 _ { 8 . 9 }$ </td><td>137.0</td><td> $3 4 . 3 { _ { 3 . 8 } }$ </td><td> $7 3 . 7 _ { 4 . 2 }$ </td><td> $2 3 1 _ { 7 . 2 }$ </td><td>125.0</td><td> $4 2 . 8 \phantom { 0 } . 8$ </td></tr><tr><td>DiffuGuard (gen)</td><td> $7 0 . 7 _ { 3 . 5 }$ </td><td> $8 3 . 7 _ { 2 . 5 }$ </td><td> $4 1 . 8 _ { 1 . 3 }$ </td><td>97.8</td><td> $5 7 . 3 _ { 2 . 1 }$ </td><td></td><td> $7 4 . 7 _ { 3 . 5 } 6 . 2 2 _ { 0 . 1 4 }$ </td><td>117.0</td><td> $2 4 . 4 _ { 1 . 0 }$ </td></tr></table>

## H.2 XSTEST OVER-REFUSAL

We evaluate refusal on 250 safe XSTest prompts and separately on 200 unsafe contrasts, using unmodified prompts and seeds 42, 43, and 44. Over-refusal is the refusal rate on safe prompts only. String matching uses full-refusal openings after lowercasing, trimming surrounding whitespace, and normalizing typographic apostrophes. WildGuard separately judges response refusal and harmfulness from each prompt–response pair (Han et al., 2024).

Table 21 reports the string-based measurements. Tables 23 and 24 report WildGuard means with seed-SD subscripts and 95% prompt-cluster bootstrap intervals below. The n columns count valid labels out of 750 safe-prompt or 600 unsafe-prompt outputs. Missing WildGuard labels are excluded separately for each method; they are not counted as safe or non-refusing. Each point estimate averages the valid-label rates within the three seeds. Intervals use 5,000 resamples of prompt clusters, retaining all seeds.

Refusal detection and coverage. The detectors assign different refusal rates to the same responses. For LLaDA warm-up on unsafe prompts, opening-string matching gives 53.5% refusal and WildGuard gives 97.5%; the corresponding Dream rates are 69.5% and 91.5%. So a reduction in stereotyped refusal openings can overstate the reduction in refusal behavior. WildGuard’s response-harmfulness labels provide a separate check: Dream warm-up has 6.9% harmful responses on unsafe XSTest prompts, compared with 0.0% for unmodified sampling, among outputs with valid labels. Coverage also varies with the intervention. Paired comparisons use the intersection of valid labels, rather than subtracting rates computed on different subsets. On safe prompts, warmup changes WildGuard refusal by −5.9 percentage points on LLaDA (95% CI: [−9.7, −2.1]; 593 matched outputs) and −35.4 points on Dream ([−41.6, −29.3]; 602 matched outputs).

DiffuGuard settings. Generation-scope DiffuGuard uses λ = 0.35 on LLaDA and λ = 0.5/0.75 for Dream’s DIJA/PAD settings. Both Dream thresholds are evaluated on XSTest. On vanilla prompts, full-block and generation-scope repair regions coincide, so these rows compare thresholds rather than repair scopes. Calibration reuses XSTest-safe prompts for the trigger diagnostic in Appendix A.6; these measurements include calibration prompts.

Table 21: XSTest string-based refusal: means [95% CI] and seed SDs.
<table><tr><td>Method</td><td>Safe: over-refusal%</td><td>SD</td><td>Unsafe: refusal%</td><td>SD</td></tr><tr><td colspan="5">LLaDA-Instruct</td></tr><tr><td>Unmodified</td><td>9.7 [6.5, 13.2]</td><td>0.9</td><td>93.7 [90.7,96.3]</td><td>1.0</td></tr><tr><td>Full SAD</td><td>10.0 [6.8, 13.5]</td><td>1.1</td><td>93.7 [90.5, 96.3]</td><td>1.3</td></tr><tr><td>SADDER (selective)</td><td>7.9 [5.1, 10.9]</td><td>0.8</td><td>85.2 [80.3,89.7]</td><td>0.3</td></tr><tr><td>SADDER + warm-up</td><td>5.2 [2.8, 8.0]</td><td>0.7</td><td>53.5 [46.7, 60.3]</td><td>0.0</td></tr><tr><td>CSD†</td><td>10.3 [7.1, 13.7]</td><td>0.8</td><td>93.7 [90.7, 96.3]</td><td>1.0</td></tr><tr><td>DiffuGuard (full)</td><td>4.0 [2.5, 5.6]</td><td>1.1</td><td>28.2 [24.7, 31.8]</td><td>1.6</td></tr><tr><td>DiffuGuard (gen) (λ = 0.35)</td><td>10.7 [7.3, 14.3]</td><td>0.5</td><td>92.2 [88.8, 95.2]</td><td>1.3</td></tr><tr><td>SADDER (top-1)</td><td>8.4 [5.5, 11.7]</td><td>0.8</td><td>85.5 [80.8, 89.8]</td><td>0.5</td></tr><tr><td>SADDER (aligned)</td><td>8.5 [5.6, 11.7]</td><td>0.9</td><td>85.3 [80.5, 89.8]</td><td>0.8</td></tr><tr><td>Random selection</td><td>10.0 [6.8, 13.5]</td><td>0.7</td><td>86.2 [82.7, 89.5]</td><td>2.9</td></tr><tr><td colspan="5">Dream-Instruct</td></tr><tr><td>Unmodified</td><td>44.9 [39.2, 50.5]</td><td>0.6</td><td>98.8 [97.3, 100.0]</td><td>0.3</td></tr><tr><td>Full SAD</td><td>46.0 [40.0, 51.9]</td><td>0.7</td><td>98.5 [96.5, 100.0]</td><td>0.0</td></tr><tr><td>SADDER (selective)</td><td>26.1 [20.9, 31.2]</td><td>0.5</td><td>92.0 [88.2, 95.5]</td><td>0.5</td></tr><tr><td>SADDER + warm-up</td><td>7.1 [4.1, 10.4]</td><td>0.2</td><td>69.5 [63.0, 76.0]</td><td>0.0</td></tr><tr><td>CSD†</td><td>45.2 [39.5, 50.8]</td><td>0.8</td><td>98.8 [97.3, 100.0]</td><td>0.3</td></tr><tr><td>DiffuGuard (full)</td><td>6.7 [4.9, 8.5]</td><td>1.2</td><td>13.5 [10.8, 16.3]</td><td>1.0</td></tr><tr><td>DiffuGuard (gen) (λ = 0.75)</td><td>43.1 [37.5, 48.5]</td><td>0.5</td><td>98.8 [97.3, 100.0]</td><td>0.3</td></tr><tr><td>DiffuGuard (gen) (λ = 0.5)</td><td>41.9 [36.4, 47.2]</td><td>0.6</td><td>98.0 [96.0, 99.5]</td><td>0.5</td></tr><tr><td>SADDER (top-1)</td><td>34.7 [28.8, 40.5]</td><td>0.2</td><td>96.5 [93.5, 99.0]</td><td>0.0</td></tr><tr><td>SADDER (aligned)</td><td>29.5 [24.0, 34.9]</td><td>0.5</td><td>96.5 [93.8, 98.7]</td><td>0.9</td></tr><tr><td>Random selection</td><td>22.5 [18.7, 26.7]</td><td>2.3</td><td>74.2 [70.0, 78.2]</td><td>4.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 22: Paired string-based refusal changes versus unmodified sampling (pp).
<table><tr><td>Method</td><td>Safe prompts</td><td>Unsafe prompts</td></tr><tr><td>LLaDA-Instruct Full SAD 0.3 [−0.9, 1.5] 0.0 [−1.0, 1.0]</td></tr></table>

Table 23: LLaDA: WildGuard judgments on XSTest.
<table><tr><td rowspan="2">Method</td><td colspan="3">Safe prompts</td><td rowspan="2"></td><td colspan="3">Unsafe prompts</td></tr><tr><td>Over-refusal%</td><td>Harmful%</td><td>n</td><td>Refusal%</td><td>Harmful%</td><td>n</td></tr><tr><td></td><td> $2 0 . 7 _ { 1 . 1 }$ </td><td> $3 . 3 _ { 0 . 8 }$ </td><td></td><td></td><td> $9 7 . 9 _ { 1 . 0 }$ </td><td> $1 . 2 _ { 0 . 6 }$ </td><td></td></tr><tr><td>Unmodified</td><td>[16.2, 25.6]</td><td>[1.3, 5.7] 603</td><td></td><td></td><td> $[ 9 6 . 0 , 9 9 . 5 ]$ </td><td>[0.2, 2.7] 585</td><td></td></tr><tr><td>Full SAD</td><td>20.91.9 [16.4, 25.7]</td><td>3.10.5 [1.2, 5.5]</td><td>603</td><td></td><td> $9 7 . 9 _ { 1 . 0 }$  [96.0,99.5]</td><td>1.20.6 [0.2, 2.7] 585</td><td></td></tr><tr><td>SADDER (selective)</td><td> $1 8 . 8 \phantom { 0 } 0 . 6$  [14.5, 23.4]</td><td>2.50.8 [0.9,4.3]</td><td>644</td><td></td><td> $9 8 . 8 _ { 0 . 3 }$  [97.3, 100.0]</td><td>0.70.3 [0.0, 1.7] 584</td><td></td></tr><tr><td>SADDER + warm-up</td><td> $1 6 . 3 _ { 2 . 0 }$  [12.4, 20.8]</td><td>2.50.0 [1.0, 4.4]</td><td>710</td><td></td><td> $9 7 . 5 \mathrm { _ 0 . 5 }$  [95.2, 99.3]</td><td> $0 . 5 \mathrm { _ 0 . 5 }$  [0.0, 1.4] 592</td><td></td></tr><tr><td>CSD†</td><td> $2 9 . 7 _ { 1 . 4 }$  [24.5, 35.1]</td><td>3.70.7 [2.1, 5.7] 589</td><td></td><td></td><td> $9 8 . 1 _ { 1 . 1 }$   $[ 9 6 . 5 , 9 9 . 5 ]$ </td><td>1.00.5 [0.2, 2.2] 585</td><td></td></tr><tr><td>DiffuGuard (full)</td><td>42.80.8 [38.5, 47.1]</td><td>7.61.7</td><td>[5.5, 9.9] 750</td><td></td><td> $8 4 . 5 _ { 1 . 5 }$   $[ 8 1 . 3 , 8 7 . 5 ]$ </td><td>14.41.6 [11.4, 17.5] 599</td><td></td></tr><tr><td>DiffuGuard (gen) (λ = 0.35)</td><td>24.41.0 [19.7, 29.4]</td><td>4.62.4 [2.4, 7.1] 595</td><td></td><td></td><td> $9 8 . 3 _ { 0 . 3 }$   $[ 9 6 . 7 , 9 9 . 6 ]$ </td><td>0.90.3 [0.2, 1.9] 586</td><td></td></tr><tr><td>SADDER (top-1)</td><td> $2 0 . 8 _ { 0 . 2 }$   $[ 1 6 . 3 , 2 5 . 5 ]$ </td><td>3.40.8 [1.5, 5.5] 650</td><td></td><td></td><td> $9 8 . 8 _ { 0 . 3 }$   $[ 9 7 . 3 , 9 9 . 8 ]$ </td><td>0.70.3 [0.0, 1.6] 583</td><td></td></tr><tr><td>SADDER (aligned)</td><td> $2 0 . 0 _ { 0 . 7 }$ </td><td> $3 . 4 _ { 0 . 5 }$ </td><td>645</td><td></td><td> $9 8 . 0 _ { 0 . 5 }$  [95.9,99.5]</td><td> $0 . 9 _ { 0 . 3 }$  [0.0, 2.2] 586</td><td></td></tr><tr><td>Random selection</td><td> $\stackrel { [ 1 5 . 4 , 2 4 . 6 ] } { 2 3 . 5 }$  [18.7, 28.5]</td><td> $\begin{array} { c } { { [ 1 . 5 , 5 . 7 ] } } \\ { { 3 . 7 _ { 1 . 0 } } } \end{array}$ </td><td>[1.9, 5.8] 622</td><td></td><td>99.50.0 [98.6, 100.0]</td><td>0.00.0 [0.0, 0.0] 588</td><td></td></tr></table>

Table 24: Dream: WildGuard judgments on XSTest.
<table><tr><td></td><td colspan="3">Safe prompts</td><td colspan="3">Unsafe prompts</td></tr><tr><td>Method</td><td>Over-refusal%</td><td>Harmful%</td><td>n</td><td>Refusal%</td><td>Harmful%</td><td>n</td></tr><tr><td>Unmodified</td><td> $5 5 . 5 _ { 1 . 3 }$ </td><td> $2 . 2 _ { 0 . 5 }$ </td><td></td><td> $9 9 . 5 _ { 0 . 0 }$ </td><td> $0 . 0 \ L _ { 0 . 0 }$ </td><td></td></tr><tr><td></td><td> $[ 4 9 . 1 , 6 1 . 5 ]$   $5 8 . 0 _ { 1 . 0 }$ </td><td> $[ 0 . 8 , 3 . 8 ]$   $2 . 7 _ { 0 . 3 }$ </td><td>629</td><td> $[ 9 8 . 5 , 1 0 0 . 0 ]$   $9 9 . 5 _ { 0 . 0 }$ </td><td> $[ 0 . 0 , 0 . 0 ]$   $0 . 0 \ L _ { 0 . 0 }$ </td><td>600</td></tr><tr><td>Full SAD</td><td> $[ 5 1 . 6 , 6 4 . 4 ]$   $4 1 . 1 _ { 0 . 8 }$ </td><td> $[ 1 . 1 , 4 . 7 ]$   $3 . 2 _ { 0 . 5 }$ </td><td>619</td><td> $[ 9 8 . 5 , 1 0 0 . 0 ]$  99.10.3</td><td> $[ 0 . 0 , 0 . 0 ]$   $0 . 0 _ { 0 . 0 }$ </td><td>600</td></tr><tr><td>SADDER (selective)</td><td> $[ 3 4 . 3 , 4 7 . 9 ]$  17.90.7</td><td> $[ 1 . 1 , 5 . 6 ]$  1.60.2</td><td>562</td><td>[97.7, 100.0] 91.51.3</td><td> $[ 0 . 0 , 0 . 0 ]$   $6 . 9 _ { 1 . 8 }$ </td><td>572</td></tr><tr><td>SADDER + warm-up</td><td>[13.5, 22.7] 56.41.9</td><td>[0.4, 2.9] 2.11.0</td><td>709</td><td>[88.0,94.5] 99.50.0</td><td>[4.2, 10.0]  $0 . 0 \ L _ { 0 . 0 }$ </td><td>597</td></tr><tr><td>CSD†</td><td>[50.0, 62.4] 64.11.5</td><td>[0.8, 3.7] 14.32.6</td><td>630</td><td>[98.5, 100.0] 75.80.6</td><td> $[ 0 . 0 , 0 . 0 ]$   $2 3 . 2 _ { 1 . 0 }$ </td><td>600</td></tr><tr><td>DiffuGuard (full)</td><td>[60.0, 68.1] 53.11.5</td><td> $[ 1 1 . 2 , 1 7 . 6 ]$   $2 . 5 \phantom { 0 } _ { 0 . 5 }$ </td><td>750</td><td> $\stackrel { [ 7 2 . 2 , 7 9 . 3 ] } { 9 9 . 5 }$ </td><td> $[ 1 9 . 8 , 2 6 . 7 ]$   $0 . 0 \ L _ { 0 . 0 }$ </td><td>600</td></tr><tr><td>DiffuGuard (gen) (λ = 0.75)</td><td>[46.9, 58.9] 54.61.1</td><td> $[ 0 . 9 , 4 . 4 ]$   $3 . 2 _ { 0 . 4 }$ </td><td>641</td><td>[98.5, 100.0] 99.20.3</td><td> $[ 0 . 0 , 0 . 0 ]$   $0 . 2 _ { 0 . 3 }$ </td><td>600</td></tr><tr><td>DiffuGuard (gen) (λ = 0.5)</td><td>[48.8, 60.3] 41.20.1</td><td> $[ 1 . 5 , 5 . 1 ]$  3.00.7</td><td>665</td><td>[97.8,100.0] 99.00.5</td><td> $[ 0 . 0 , 0 . 5 ]$   $0 . 0 \ L _ { 0 . 0 }$ </td><td>600</td></tr><tr><td>SADDER (top-1)</td><td>[34.7, 47.4] 38.11.3</td><td> $[ 1 . 2 , 5 . 0 ]$  2.50.7</td><td>670</td><td> $\begin{array} { c } { { [ 9 7 . 7 , 1 0 0 . 0 ] } } \\ { { 9 8 . 0 _ { 0 . 9 } } } \end{array}$ </td><td> $[ 0 . 0 , 0 . 0 ]$   $0 . 0 \ L _ { 0 . 0 }$ </td><td>600</td></tr><tr><td>SADDER (aligned)</td><td> $[ 3 1 . 8 , 4 4 . 1 ]$   $3 9 . 2 _ { 0 . 5 }$ </td><td> $[ 0 . 8 , 4 . 3 ]$   $2 . 9 \phantom { } _ { 0 . 1 }$ </td><td>653</td><td>[96.0, 99.5] 98.10.7</td><td> $[ 0 . 0 , 0 . 0 ]$   $1 . 3 \phantom { 0 } _ { 0 . 7 }$ </td><td>599</td></tr><tr><td>Random selection</td><td>[33.2, 45.2]</td><td>[1.0, 5.1] 526</td><td></td><td>[96.1, 99.6]</td><td>[0.2, 2.7]471</td><td></td></tr></table>

Table 25: LLaDA: paired WildGuard refusal changes (pp).
<table><tr><td>Method</td><td>Safe prompts</td><td>Pairs</td><td>Unsafe prompts</td><td>Pairs</td></tr><tr><td>Full SAD</td><td>−0.0 [−1.2, 1.4]</td><td>600</td><td>0.0 [0.0, 0.0]</td><td>585</td></tr><tr><td>SADDER (selective)</td><td>−2.4 [−5.8, 0.9]</td><td>579</td><td> $0 . 7 \ [ - 0 . 9 , 2 . 4 ]$ </td><td>581</td></tr><tr><td>SADDER + warm-up</td><td>-5.9 [−9.7, −2.1]</td><td>593</td><td>−0.7 [−2.9, 1.4]</td><td>581</td></tr><tr><td>CSD†</td><td>9.3 [5.9, 12.8]</td><td>588</td><td>0.2 [−0.3, 0.7]</td><td>585</td></tr><tr><td>DiffuGuard (full)</td><td>16.7 [12.4, 21.0]</td><td>603</td><td>-13.7 [−16.8, -10.6]</td><td>585</td></tr><tr><td>DiffuGuard (gen) (λ = 0.35)</td><td>3.2 [1.3, 5.3]</td><td>589</td><td>0.3 [−0.5, 1.4]</td><td>585</td></tr><tr><td>SADDER (top-1)</td><td>−1.0 [−4.6, 2.4]</td><td>585</td><td>0.7 [−0.7, 2.4]</td><td>583</td></tr><tr><td>SADDER (aligned)</td><td>-1.9 [-4.9,0.9]</td><td>584</td><td>−0.2 [−2.4, 1.9]</td><td>583</td></tr><tr><td>Random selection</td><td>1.1 [−2.0, 4.2]</td><td>568</td><td>1.4 [0.2, 2.9]</td><td>580</td></tr></table>

Table 26: Dream: paired WildGuard refusal changes (pp).
<table><tr><td>Method</td><td>Safe prompts Pairs</td><td></td><td>Unsafe prompts</td><td>Pairs</td></tr><tr><td>Full SAD</td><td> $1 . 8 \ [ - 0 . 7 , 4 . 3 ]$ </td><td>619</td><td>0.0 [0.0, 0.0]</td><td>600</td></tr><tr><td>SADDER (selective)</td><td>-8.6 [−14.1, -3.3]</td><td>532</td><td>−0.3 [−1.1, 0.0]</td><td>572</td></tr><tr><td>SADDER + warm-up</td><td>-35.4 [−41.6, -29.3]</td><td>602</td><td> $- 8 . 0 \left[ - 1 1 . 5 , - 5 . 0 \right]$ </td><td>597</td></tr><tr><td>CSD†</td><td>1.0 [0.3, 1.8]</td><td>629</td><td>0.0 [0.0, 0.0]</td><td>600</td></tr><tr><td>DiffuGuard (full)</td><td>7.6 [2.7, 12.7]</td><td>629</td><td>-23.7 [-27.3, -20.0]</td><td>600</td></tr><tr><td>DiffuGuard (gen) (λ = 0.75)</td><td>−1.4 [−3.1, 0.2]</td><td>629</td><td>0.0 [0.0, 0.0]</td><td>600</td></tr><tr><td>DiffuGuard (gen) (λ = 0.5)</td><td>−0.8 [−2.8, 1.4]</td><td>629</td><td>-0.3 [−0.8, 0.0]</td><td>600</td></tr><tr><td>SADDER (top-1)</td><td> $- 1 2 . 5 \ [ - 1 7 . 7 , - 7 . 3 ]$ </td><td>625</td><td>−0.5 [−1.3, 0.0]</td><td>600</td></tr><tr><td>SADDER (aligned)</td><td> $- 1 4 . 1 \ [ - 1 9 . 0 , - 9 . 2 ]$ </td><td>603</td><td>−1.5 [−3.2, −0.3]</td><td>599</td></tr><tr><td>Random selection</td><td>-6.7 [−11.9, −1.7]</td><td>493</td><td>-1.3 [−2.7, −0.2]</td><td>471</td></tr></table>

## H.3 HELD-OUT ATTACK COMPARISONS

Table 27 reports the main jailbreak-table estimates with pointwise 95% prompt-cluster bootstrap confidence intervals, using 5,000 resamples and retaining all observed seeds within each sampled prompt. These intervals describe prompt uncertainty; the main-table subscripts describe variation

across the three generation seeds. The XSTest over-refusal is the offline string-based matching metric rather than the WildGuard over-refusal metric.

Table 27: Held-out HarmBench attacks and XSTest over-refusal.
<table><tr><td>Method</td><td colspan="4">DIJA</td><td colspan="4">PAD</td><td>XSTest</td></tr><tr><td></td><td>HB ASR %↓</td><td>LG unsafe %↓</td><td>Harmful PPL↓</td><td>NFE</td><td>HB ASR %↓</td><td>LG unsafe %↓</td><td>Harmful PPL↓</td><td>NFE</td><td>Over-refusal %↓</td></tr><tr><td>LLaDA-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td>78.7 [72.0,85.0] [75.0,88.3] [8.59,10.1]</td><td>82.0</td><td>9.34</td><td>91.7</td><td>58.7</td><td>75.7 [50.3,66.7] [68.3,82.3]</td><td>6.06 [5.68,6.47]</td><td>116.0</td><td>9.7 [6.5,13.2]</td></tr><tr><td>Full SAD</td><td>77.0 [69.3,84.3] [73.7,87.7] [10.0,12.1]</td><td>81.0</td><td>11.0</td><td>91.7</td><td>63.3</td><td>78.7 [56.3,70.3] [72.0,85.0] [7.79,8.74]</td><td>8.24</td><td>116.0</td><td>10.0 [6.8,13.5]</td></tr><tr><td>SADDER</td><td>79.0</td><td>79.3</td><td>13.3</td><td>91.7</td><td>57.7</td><td>73.7</td><td>13.2</td><td>116.0</td><td>7.9</td></tr><tr><td>(selective) SADDER</td><td>78.3</td><td>80.7</td><td>[71.7,86.3] [71.7,86.3] [11.6,15.7] 12.8</td><td></td><td>57.0</td><td>[50.3,65.0] [66.3,80.7] [10.3,17.7] 68.3</td><td>11.4</td><td></td><td>[5.1,10.9] 5.2</td></tr><tr><td>+ warm-up</td><td>[71.3,85.0]</td><td></td><td>[73.0,87.3] [11.1,15.1]</td><td>91.7</td><td></td><td>[49.0,65.0] [60.3,76.0]</td><td>[9.97,13.4]</td><td>116.0</td><td>[2.8,8.0]</td></tr><tr><td>CSD†</td><td>48.7 [40.7,57.0] [72.7,86.3]</td><td>79.7</td><td>36.1 [31.1,41.4]</td><td>115.7</td><td>30.3</td><td>69.3 [24.0,37.0] [61.7,76.7] [17.3,22.6]</td><td>19.7</td><td>142.0</td><td>10.3 [7.1,13.7]</td></tr><tr><td>DiffuGuard (full)</td><td>42.7</td><td>81.7 [35.7,50.0] [75.7,87.0]</td><td>255 [239,272]</td><td>137.0</td><td>34.3</td><td>73.7 [28.0,41.0] [67.3,79.7]</td><td>231</td><td>125.0</td><td>4.0 [2.5,5.6]</td></tr><tr><td>DiffuGuard (gen)</td><td>70.7 [63.7,77.7]</td><td>83.7 [77.0,89.7]</td><td>41.8 [32.0,53.7]</td><td>97.8</td><td>57.3 [49.0,65.0]</td><td>74.7 [67.0,81.7]</td><td>[214,251] 6.22 [5.85,6.59]</td><td>117.0</td><td>10.7 [7.3,14.3]</td></tr><tr><td>Dream-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unmodified</td><td>82.0</td><td>80.7</td><td>11.4 [75.3,88.3] [73.3,87.3] [10.6,12.3]</td><td>91.7</td><td>51.7</td><td>64.0</td><td>8.53 [43.0,60.3] [55.7,72.0] [7.95,9.21]</td><td>116.0</td><td>44.9 [39.2,50.5]</td></tr><tr><td>Full SAD</td><td>85.3</td><td>83.0</td><td>11.8</td><td>91.7</td><td>52.7</td><td>63.3</td><td>9.52</td><td>116.0</td><td>46.0</td></tr><tr><td>SADDER</td><td>77.7</td><td>81.7</td><td>[79.0,91.3] [75.3,90.0] [10.9,12.8] 12.2</td><td>91.7</td><td>52.7</td><td>[43.7,61.7] [54.7,71.0] 66.3</td><td>[9.01,10.1] 9.58</td><td>116.0</td><td>[40.0,51.9] 26.1</td></tr><tr><td>(selective)</td><td></td><td></td><td>[70.3,84.7] [74.7,88.0] [11.3,13.1]</td><td></td><td></td><td>[43.7,61.7] [57.3,75.0] [8.86,10.4]</td><td></td><td></td><td>[20.9,31.2]</td></tr><tr><td>SADDER + warm-up</td><td>78.7</td><td>79.7</td><td>12.2 [71.7,85.7] [72.3,86.7] [11.3,13.2]</td><td>91.7</td><td>42.7</td><td>55.0 [33.7,51.7] [45.7,64.0]</td><td>10.4 [9.60,11.4]</td><td>116.0</td><td>7.1 [4.1,10.4]</td></tr><tr><td>CSD†</td><td>68.0</td><td>82.3</td><td>45.9</td><td>115.5</td><td>33.3</td><td>63.3</td><td>23.7</td><td>140.8</td><td>45.2</td></tr><tr><td></td><td></td><td></td><td>[60.7,75.7] [75.7,88.7] [37.6,55.9]</td><td></td><td></td><td>[26.0,41.0] [54.7,71.0] [19.2,29.8]</td><td></td><td></td><td>[39.5,50.8]</td></tr><tr><td>DiffuGuard (full)</td><td>33.3 [27.3,39.7] [78.0,89.3]</td><td>84.0</td><td>537 [494,587]</td><td>137.0</td><td>12.0</td><td>70.0 [8.0,16.0] [64.0,76.0]</td><td>687 [631,748]</td><td>125.0</td><td>6.7 [4.9,8.5]</td></tr></table>

## H.4 WILDGUARD HARMFULNESS AND REFUSAL

Tables 28 and 29 report full-response WildGuard harmfulness and refusal on the same HarmBench outputs, including remasking and random-selection controls. Entries show the mean with a seed-SD subscript and a 95% prompt-cluster bootstrap interval below. The n columns count valid labels out of 300 outputs per attack and method. Missing labels are excluded, and rates average the validoutput rate within each of the three seeds. Table 30 compares selective SADDER and warm-up against full SAD and random selection on jointly scored outputs. Negative harmfulness differences indicate fewer harmful responses; negative refusal differences indicate fewer refusals, which alone does not establish a safety improvement. The prompt–response pairs use the original behavior and the complete response, including any retained attack scaffold.

Comparison across safety judges. The judges differ in their assessment of these gains. On Dream/PAD, warm-up lowers HB ASR from 51.7% to 42.7%, while WG harmfulness increases from 66.0% to 69.0%; selective SADDER reaches 80.0% WG harmfulness. On LLaDA/PAD, warm-up reduces both HB ASR (58.7% to 57.0%) and WG harmfulness (80.0% to 76.3%). Full-block Diffu-Guard on Dream/DIJA illustrates a larger disagreement: 33.3% HB ASR accompanies 98.3% WG harmfulness and PPL 537. The apparent safety benefit depends on the judge as well as the model and attack, particularly when the intervention disrupts fluency.

Table 28: LLaDA: WildGuard judgments on HarmBench.
<table><tr><td rowspan="2">Method</td><td colspan="3">DIJA</td><td colspan="3">PAD</td></tr><tr><td>Harmful%</td><td>Refusal%</td><td>n</td><td>Harmful%</td><td>Refusal%</td><td>n</td></tr><tr><td rowspan="2">Unmodified</td><td> $9 6 . 0 _ { 2 . 6 }$ </td><td> $1 . 7 _ { 0 . 6 }$ </td><td></td><td> $8 0 . 0 _ { 3 . 0 }$ </td><td> $1 5 . 3 \mathrm { { 1 . 5 } }$ </td><td></td></tr><tr><td> $[ 9 2 . 3 , 9 9 . 0 ]$ </td><td> $[ 0 . 0 , 4 . 3 ]$ </td><td>300</td><td> $[ 7 3 . 7 , 8 6 . 0 ]$ </td><td>[10.3, 21.0] 300</td><td></td></tr><tr><td rowspan="2">Full SAD</td><td> $9 6 . 3 _ { 0 . 6 }$ </td><td> $2 . 0 \phantom { 0 } _ { 0 . 0 }$ </td><td></td><td> $8 5 . 7 _ { 3 . 8 }$ </td><td>11.33.8</td><td></td></tr><tr><td> $[ 9 2 . 3 , 9 9 . 3 ]$ </td><td> $[ 0 . 0 , 5 . 0 ]$ </td><td>300</td><td>[80.0,90.7]</td><td>[6.7, 16.3]</td><td>300</td></tr><tr><td rowspan="2">SADDER (selective)</td><td> $9 6 . 7 _ { 0 . 6 }$ </td><td> $2 . 0 \phantom { 0 } _ { 0 . 0 }$ </td><td></td><td>81.02.6</td><td>15.04.0</td><td></td></tr><tr><td> $\left[ 9 2 . 7 , 9 9 . 7 \right]$ </td><td>[0.0, 5.0] 300</td><td></td><td>[74.3, 87.0]</td><td>[10.0, 20.7]</td><td>300</td></tr><tr><td rowspan="2"></td><td> $9 6 . 7 _ { 1 . 5 }$ </td><td>2.01.0</td><td></td><td> $7 6 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td>19.01.7</td><td></td></tr><tr><td>SADDER + warm-up [93.0, 99.3]</td><td>[0.0, 4.7]</td><td>300 </td><td></td><td>[68.7, 83.3] [12.7, 26.0] 300</td><td></td></tr><tr><td rowspan="2">CSD†</td><td> $9 7 . 0 _ { 1 . 0 }$ </td><td>1.70.6</td><td></td><td> $7 7 . 0 _ { 4 . 4 }$ </td><td>19.72.1</td><td></td></tr><tr><td> $[ 9 3 . 7 , 9 9 . 3 ]$ </td><td> $[ 0 . 0 , 4 . 3 ]$ </td><td>300</td><td> $[ 7 0 . 7 , 8 3 . 0 ]$ </td><td>[14.0, 26.0]</td><td>300</td></tr><tr><td rowspan="2">DiffuGuard (full)</td><td> $9 7 . 7 _ { 2 . 1 }$ </td><td> $0 . 7 _ { 1 . 2 }$  [0.0, 1.7] 300</td><td></td><td> $8 5 . 0 _ { 4 . 0 }$ </td><td>11.73.5 [7.7, 16.0]</td><td>300</td></tr><tr><td> $[ 9 5 . 3 , 9 9 . 7 ]$   $9 7 . 0 _ { 1 . 0 }$ </td><td>1.00.0</td><td></td><td>[80.0, 89.7]  $7 8 . 0 _ { 2 . 6 }$ </td><td> $1 7 . 3 _ { 1 . 2 }$ </td><td></td></tr><tr><td rowspan="2">DiffuGuard (gen)</td><td>[93.7, 99.7]</td><td> $[ 0 . 0 , 3 . 0 ]$ </td><td>300</td><td>[71.3, 84.3]</td><td>[11.7, 23.0]</td><td>300</td></tr><tr><td>97.30.6</td><td> $1 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td></td><td>81.33.2</td><td>15.32.9</td><td></td></tr><tr><td rowspan="2">SADDER (top-1)</td><td>[94.0, 99.7]</td><td></td><td>300</td><td>[74.3, 87.3]</td><td>[9.7, 21.3] 300</td><td></td></tr><tr><td>96.70.6</td><td> $\stackrel { [ 0 . 0 , 3 . 7 ] } { 2 . 0 _ { 0 . 0 } }$ </td><td></td><td>83.31.5</td><td>13.02.0</td><td></td></tr><tr><td rowspan="2">SADDER (aligned)</td><td>[92.7,99.7]</td><td> $[ 0 . 0 , 5 . 0 ]$ </td><td>300</td><td>[77.7, 88.7]</td><td>[8.3, 18.0]</td><td>300</td></tr><tr><td>96.31.2</td><td> $1 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td></td><td>77.33.2</td><td>19.03.5</td><td></td></tr><tr><td rowspan="2">Random selection</td><td>[92.3, 99.3]</td><td>[0.0, 3.3] 300</td><td></td><td> [71.7, 82.7] [14.0, 24.0] 300</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 29: Dream: WildGuard judgments on HarmBench.
<table><tr><td></td><td colspan="3">DIJA</td><td colspan="3">PAD</td></tr><tr><td>Method</td><td>Harmful%</td><td>Refusal%</td><td>n</td><td>Harmful%</td><td>Refusal%</td><td>n</td></tr><tr><td></td><td> $9 7 . 3 _ { 0 . 6 }$ </td><td> $0 . 7 _ { 0 . 6 }$ </td><td></td><td> $6 6 . 0 _ { 2 . 0 }$ </td><td> $3 0 . 3 _ { 2 . 5 }$ </td><td></td></tr><tr><td>Unmodified</td><td>[94.0, 99.7]  $9 7 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $[ 0 . 0 , 2 . 0 ]$   $1 . 0 \phantom { 0 } _ { 0 . 0 }$ </td><td>300</td><td> $[ 5 7 . 7 , 7 4 . 0 ]$   $6 6 . 7 _ { 0 . 6 }$ </td><td>[22.3, 38.7] 300  $2 9 . 3 _ { 0 . 6 }$ </td><td></td></tr><tr><td>Full SAD</td><td>[94.0,100.0]  $9 6 . 7 _ { 1 . 5 }$ </td><td>[0.0, 3.0] 1.30.6</td><td>300</td><td> $8 0 . 0 _ { 1 . 7 }$ </td><td>[58.3, 74.7] [21.3, 37.7] 300 16.71.5</td><td></td></tr><tr><td>SADDER (selective)</td><td>[93.3, 99.3] 96.30.6</td><td>[0.0,3.7] 1.70.6</td><td>300</td><td>[72.7,86.7] 69.01.7</td><td>[10.0, 23.7] 300 28.01.7</td><td></td></tr><tr><td>SADDER + warm-up</td><td>[92.7, 99.3] 98.00.0</td><td>[0.0, 4.3] 0.70.6</td><td>300</td><td>[60.0, 77.0] 65.31.5</td><td>[20.0, 37.0] 300 30.72.5</td><td></td></tr><tr><td>CSD†</td><td>[95.0, 100.0] 98.30.6</td><td>[0.0, 2.0]</td><td>300</td><td>[57.0, 73.3]</td><td>[22.7, 39.0] 300</td><td></td></tr><tr><td>DiffuGuard (full)</td><td>[95.7,100.0] 98.30.6</td><td>0.70.6 [0.0, 1.7]</td><td>299</td><td>69.32.1 [62.7,75.7][</td><td> $2 9 . 7 _ { 2 . 1 }$  [23.3, 36.3]</td><td>300</td></tr><tr><td>DiffuGuard (gen)</td><td>[95.7, 100.0] 97.70.6</td><td>0.70.6 [0.0, 2.0]</td><td>300</td><td>65.02.0 [56.7,72.7]</td><td> $3 0 . 0 _ { 1 . 0 }$  [22.3, 38.0] 300</td><td></td></tr><tr><td>SADDER (top-1)</td><td>[94.3, 100.0] 97.70.6</td><td>1.30.6 [0.0, 3.7]</td><td>300</td><td>80.01.7 [72.7,87.0]</td><td>17.01.0 [10.7, 24.0]</td><td>300</td></tr><tr><td>SADDER (aligned)</td><td>[94.3, 100.0]</td><td>1.00.0 [0.0, 3.0]</td><td>300</td><td>72.01.7 [63.3,80.0]</td><td>25.01.0 [17.3, 33.0] 300</td><td></td></tr><tr><td>Random selection</td><td>97.01.0 [93.7,99.7]</td><td>1.00.0 [0.0, 3.0]</td><td>300</td><td>71.02.0 [63.3, 78.0]</td><td>27.01.7 [20.3, 34.3]</td><td>300</td></tr></table>

Table 30: Paired WildGuard differences on HarmBench (pp).
<table><tr><td>Comparison</td><td>∆ harmful%</td><td>∆ refusal%</td><td>Pairs</td></tr><tr><td>LLaDA / DIJA</td><td></td><td></td><td></td></tr><tr><td>Selective vs. full SAD</td><td>0.3 [−0.7, 1.7]</td><td>0.0 [0.0, 0.0]</td><td>300</td></tr><tr><td>Selective vs. random</td><td>0.3 [−1.7, 3.0]</td><td>0.7 [0.0, 1.7]</td><td>300</td></tr><tr><td>Warm-up vs. full SAD</td><td>0.3 [−0.7, 1.7]</td><td>−0.0 [−1.0, 1.0]</td><td>300</td></tr><tr><td>Warm-up vs. random LLaDA / PAD</td><td>0.3 [−1.3, 2.3]</td><td>0.7 [0.0, 1.7]</td><td>300</td></tr><tr><td>Selective vs. full SAD Selective vs. random</td><td>-4.7 [−11.0, 1.7] 3.7 [−2.0, 9.3]</td><td>3.7 [−2.3, 9.7] -4.0 [−9.3, 1.7]</td><td>300</td></tr><tr><td>Warm-up vs. full SAD</td><td>-9.3 [−16.7, -2.0]</td><td>7.7 [0.0, 15.0]</td><td>300 300</td></tr><tr><td>Warm-up vs. random Dream / DIJA</td><td>-1.0 [−6.7, 4.3]</td><td>0.0 [−5.3, 5.7]</td><td>300</td></tr><tr><td>Selective vs. full SAD</td><td>-0.7 [−2.0, 0.7]</td><td>0.3 [0.0, 1.0]</td><td>300</td></tr><tr><td>Selective vs. random</td><td>-0.3 [−1.7, 0.7]</td><td>0.3 [0.0, 1.0]</td><td>300</td></tr><tr><td>Warm-up vs. full SAD</td><td>-1.0 [−2.7, 0.0]</td><td>0.7 [0.0, 2.0]</td><td>300</td></tr><tr><td>Warm-up vs. random</td><td>-0.7 [−2.0, 0.7]</td><td>0.7 [0.0, 2.0]</td><td>300</td></tr><tr><td>Dream / PAD</td><td></td><td></td><td></td></tr><tr><td>Selective vs. full SAD</td><td>13.3 [5.0, 22.0]</td><td>−12.7 [−21.3, −4.3]</td><td>300</td></tr><tr><td>Selective vs. random</td><td>9.0 [1.3, 16.7]</td><td>-10.3 [−18.0, -3.0]</td><td>300</td></tr><tr><td>Warm-up vs. full SAD</td><td>2.3 [−7.3, 12.0]</td><td>−1.3 [−11.3, 8.7]</td><td></td></tr><tr><td></td><td></td><td></td><td>300</td></tr><tr><td>Warm-up vs. random</td><td>-2.0 [−7.3, 3.3]</td><td>1.0 [−4.0, 6.0]</td><td></td></tr><tr><td></td><td></td><td></td><td>300</td></tr></table>

## H.5 DIJA JUDGE AND RESPONSE-VIEW SENSITIVITY

Separating generated content from the attack scaffold. DIJA completions contain both fixed attack text and model-generated spans. A harmfulness judgment on the complete response can therefore depend on text supplied by the attacker. We compare HarmBench (HB), LlamaGuard (LG), and WildGuard (WG) on two response views: the complete response, including the scaffold, and only the spans marked as initially masked in the saved completion segmentation (fills). Nonempty spans are trimmed and joined by newlines before judging. HB measures behavior-specific attack success, LG unsafe content, and WG response harmfulness. These are distinct evaluator criteria, even when the response view is held fixed. The main table retains full-response HB/LG for attacks and uses WG only for XSTest. Generated-span attack scores from all three judges are reported here. All response-view comparisons use the same generated completions and defence settings, so differences between views reflect the scoring procedure.

This view isolates the initially editable content, rather than judging the assembled attack response. Removing intervening text can change the context needed to interpret a span. For full-block Diffu-Guard, repairs outside the initially masked spans are also excluded. Generated-span scores thus complement complete-response judgments; they do not measure the safety of every token that a defence may modify.

Judge disagreement persists after removing fixed text. Tables 31 and 32 separate judge choice from response-view choice. On unmodified sampling, removing the scaffold lowers LG unsafe rate from 82.0% to 53.3% on LLaDA and from 80.7% to 48.7% on Dream. HB falls more modestly, from 78.7% to 73.0% and from 82.0% to 76.0%, respectively; WG remains above 96%. Full-response scores therefore cannot be interpreted as measurements of generated spans alone. Conversely, ex cluding fixed text does not make the judges interchangeable.

Warm-up illustrates the consequence for method comparisons. Against paired unmodified outputs, LG reports reductions of 11.7 points on LLaDA (95% CI: [−19.7, −4.3]) and 9.0 points on Dream ([−16.0, −2.0]), while the corresponding HB and WG intervals include zero. Dream selective SAD-DER also improves HB relative to full SAD by 8.0 points ([−13.3, −2.7]), but increases LG unsafe rate by 15.0 points ([8.0, 22.0]). These are evaluator-dependent effects, not a consistent safety ordering across judges. The paired estimates in Table 33 retain the same prompts and seeds and use pointwise 95% intervals.

Table 31: LLaDA: DIJA judge and response-view comparison on 100 held-out HarmBench behaviors, three seeds. Full includes the fixed scaffold; fills excludes it. Rates (%) are means with seed-SD subscripts. HB/LG have 300 scored outputs per row; WG has 300 except Dream aligned remasking (299).

$$
7 8 . 7 _ { 3 . 1 }
$$

$$
7 3 . 0 _ { 3 . 6 }
$$

$$
8 2 . 0 _ { 2 . 6 }
$$

$$
5 3 . 3 _ { 2 . 5 }
$$

$$
7 7 . 0 _ { 2 . 0 }
$$

$$
9 6 . 0 _ { 2 . 6 }
$$

$$
9 6 . 3 _ { 1 . 5 }
$$

$$
7 1 . 7 _ { 2 . 9 }
$$

$$
8 1 . 0 \phantom { 0 } _ { 0 . 0 }
$$

$$
5 6 . 3 _ { 6 . 8 }
$$

$$
9 6 . 3 _ { 0 . 6 }
$$

$$
9 7 . 3 _ { 1 . 2 }
$$

$$
7 9 . 0 _ { 1 . 0 }
$$

$$
7 5 . 7 _ { 2 . 3 }
$$

$$
7 9 . 3 _ { 1 . 5 }
$$

$$
5 4 . 0 _ { 2 . 6 }
$$

$$
9 6 . 7 _ { 0 . 6 }
$$

$$
9 6 . 3 _ { 1 . 5 }
$$

$$
\mathrm { S A D D E R + w a r m - u p }
$$

$$
7 8 . 3 _ { 0 . 6 }
$$

$$
7 1 . 3 _ { 2 . 5 }
$$

$$
8 0 . 7 _ { 0 . 6 }
$$

$$
4 1 . 7 _ { 3 . 5 }
$$

$$
9 6 . 7 _ { 1 . 5 }
$$

$$
9 7 . 3 _ { 2 . 1 }
$$

$$
\mathrm { C S D ^ { \dag } }
$$

$$
4 8 . 7 _ { 4 . 7 }
$$

$$
3 0 . 7 _ { 5 . 1 }
$$

$$
7 9 . 7 _ { 4 . 9 }
$$

$$
3 3 . 7 _ { 4 . 2 }
$$

$$
9 7 . 0 _ { 1 . 0 }
$$

$$
9 6 . 3 _ { 1 . 5 }
$$

$$
4 2 . 7 _ { 4 . 5 }
$$

$$
3 2 . 3 _ { 0 . 6 }
$$

$$
8 1 . 7 _ { 2 . 5 }
$$

$$
4 9 . 0 _ { 3 . 6 }
$$

$$
9 7 . 7 _ { 2 . 1 }
$$

$$
9 7 . 0 _ { 0 . 0 }
$$

$$
7 0 . 7 _ { 3 . 5 }
$$

$$
5 5 . 7 _ { 4 . 0 }
$$

$$
8 3 . 7 _ { 2 . 5 }
$$

$$
4 9 . 0 _ { 5 . 2 }
$$

$$
9 7 . 0 _ { 1 . 0 }
$$

$$
7 8 . 7 _ { 3 . 1 }
$$

$$
9 7 . 7 _ { 0 . 6 }
$$

$$
7 2 . 7 { } _ { 3 . 5 }
$$

$$
8 0 . 7 _ { 2 . 1 }
$$

$$
5 7 . 0 _ { 2 . 0 }
$$

$$
9 7 . 3 \mathrm { _ 0 . 6 }
$$

$$
7 9 . 3 _ { 1 . 2 }
$$

$$
7 5 . 0 _ { 3 . 5 }
$$

$$
9 6 . 0 _ { 2 . 0 }
$$

$$
8 0 . 0 _ { 2 . 0 }
$$

$$
7 6 . 3 _ { 1 . 2 }
$$

$$
7 2 . 3 _ { 4 . 0 }
$$

$$
5 7 . 3 \mathrm { _ 0 . 6 }
$$

$$
8 0 . 3 _ { 5 . 5 }
$$

$$
9 6 . 7 _ { 0 . 6 }
$$

$$
5 1 . 7 _ { 3 . 1 }
$$

$$
9 6 . 3 _ { 1 . 5 }
$$

$$
9 6 . 3 _ { 1 . 2 }
$$

$$
9 7 . 3 _ { 1 . 2 }
$$

Table 32: Dream: DIJA judge and response-view comparison on 100 held-out HarmBench behaviors, three seeds. Full includes the fixed scaffold; fills excludes it. Rates (%) are means with seed-SD subscripts. HB/LG have 300 scored outputs per row; WG has 300 except Dream aligned remasking (299).
<table><tr><td>Method</td><td>HB full</td><td>HB fills</td><td>LG full</td><td>LG fills</td><td>WG full</td><td>WG fills</td></tr><tr><td>Unmodified</td><td> $8 2 . 0 \mathrm { 1 . 0 }$ </td><td> $7 6 . 0 3 . 6 $ </td><td> $8 0 . 7 _ { 1 . 5 }$ </td><td> $4 8 . 7 _ { 2 . 5 }$ </td><td> $9 7 . 3 \mathrm { _ 0 . 6 }$ </td><td> $9 7 . 3 \mathrm { _ 0 . 6 }$ </td></tr><tr><td>Full SAD</td><td> $8 5 . 3 _ { 0 . 6 }$ </td><td> $8 0 . 3 _ { 2 . 3 }$ </td><td> $8 3 . 0 _ { 1 . 0 }$ </td><td> $4 2 . 3 _ { 3 . 1 }$ </td><td> $9 7 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $9 6 . 3 _ { 1 . 5 }$ </td></tr><tr><td>SADDER (selective)</td><td> $7 7 . 7 _ { 1 . 5 }$ </td><td> $7 2 . 3 _ { 1 . 5 }$ </td><td> $8 1 . 7 _ { 3 . 2 }$ </td><td> $5 7 . 3 _ { 2 . 1 }$ </td><td> $9 6 . 7 _ { 1 . 5 }$ </td><td> $9 7 . 3 _ { 1 . 2 }$ </td></tr><tr><td>SADDER + warm-up</td><td> $7 8 . 7 _ { 2 . 3 }$ </td><td> $7 5 . 7 _ { 3 . 1 }$ </td><td> $7 9 . 7 _ { 1 . 2 }$ </td><td> $3 9 . 7 _ { 1 . 2 }$ </td><td> $9 6 . 3 _ { 0 . 6 }$ </td><td> $9 6 . 0 _ { 0 . 0 }$ </td></tr><tr><td>CSD†</td><td> $6 8 . 0 6 . 2 $ </td><td> $4 5 . 0 _ { 3 . 5 }$ </td><td> $8 2 . 3 _ { 0 . 6 }$ </td><td> $4 1 . 7 _ { 2 . 1 }$ </td><td> $9 8 . 0 _ { 0 . 0 }$ </td><td> $9 7 . 3 _ { 0 . 6 }$ </td></tr><tr><td>DiffuGuard (full)</td><td> $3 3 . 3 _ { 2 . 5 }$ </td><td> $2 2 . 0 _ { 2 . 6 }$ </td><td> $8 4 . 0 _ { 2 . 0 }$ </td><td> $4 4 . 7 _ { 3 . 1 }$ </td><td> $9 8 . 3 _ { 0 . 6 }$ </td><td> $9 8 . 3 _ { 1 . 2 }$ </td></tr><tr><td>DiffuGuard (gen)</td><td> $6 9 . 3 _ { 1 . 5 }$ </td><td> $5 4 . 3 _ { 1 . 5 }$ </td><td> $8 1 . 3 _ { 1 . 5 }$ </td><td> $5 5 . 7 _ { 4 . 0 }$ </td><td> $9 8 . 3 _ { 0 . 6 }$ </td><td> $9 8 . 0 _ { 1 . 0 }$ </td></tr><tr><td>SADDER (top-1)</td><td> $7 8 . 0 _ { 2 . 6 }$ </td><td> $7 1 . 0 _ { 1 . 7 }$ </td><td> $8 0 . 0 _ { 2 . 6 }$ </td><td> $5 8 . 3 _ { 1 . 5 }$ </td><td> $9 7 . 7 _ { 0 . 6 }$ </td><td> $9 7 . 3 _ { 0 . 6 }$ </td></tr><tr><td>SADDER (aligned)</td><td> $8 5 . 3 \phantom { 0 } _ { 0 . 6 }$ </td><td> $7 7 . 7 _ { 3 . 1 }$ </td><td> $8 2 . 7 _ { 0 . 6 }$ </td><td> $5 5 . 7 _ { 1 . 5 }$ </td><td> $9 7 . 7 _ { 0 . 6 }$ </td><td> $9 7 . 7 _ { 0 . 6 }$ </td></tr><tr><td>Random selection</td><td> $7 9 . 3 _ { 1 . 2 }$ </td><td> $7 4 . 3 _ { 1 . 5 }$ </td><td> $8 2 . 0 _ { 2 . 0 }$ </td><td> $5 3 . 7 _ { 4 . 0 }$ </td><td> $9 7 . 0 _ { 1 . 0 }$ </td><td> $9 7 . 0 _ { 1 . 0 }$ </td></tr></table>

Table 33: Generated-span DIJA score differences (percentage points), with 95% prompt-cluster bootstrap intervals. Comparisons use the same prompts and seeds; full SAD and random selection have no warm-up.
<table><tr><td>Comparison</td><td></td><td>HB</td><td></td><td>LG</td><td>WG</td></tr><tr><td colspan="6">LLaDA</td></tr><tr><td>Selective vs Unmodified</td><td></td><td>2.7 [−2.0, 7.3]</td><td>0.7 [−6.7, 7.7]</td><td></td><td>0.0 [−2.3, 2.0]</td></tr><tr><td>Selective vs Full SAD</td><td></td><td>4.0 [−1.3, 9.3]</td><td>-2.3 [−8.7, 3.7]</td><td></td><td>-1.0 [−2.7, 0.3]</td></tr><tr><td>Selective vs Random selection</td><td></td><td>3.3 [−1.3, 7.7]</td><td>2.3 [−2.7, 7.3]</td><td></td><td>-1.0 [−2.7, 0.7]</td></tr><tr><td>Warm-up vs Unmodified</td><td></td><td>-1.7 [−7.7, 4.0]</td><td>-11.7 [−19.7, -4.3]</td><td></td><td>1.0 [−1.0, 3.0]</td></tr><tr><td>Warm-up vs Full SAD</td><td></td><td>-0.3 [−6.0, 5.3]</td><td>-14.7 [−21.7, -8.0]</td><td></td><td>0.0 [−1.3, 1.3]</td></tr><tr><td>Warm-up vs Random selection</td><td></td><td>-1.0 [−6.0, 3.7]</td><td>-10.0 [−16.3, -4.0]</td><td></td><td>0.0 [−1.3, 1.3]</td></tr><tr><td>Dream</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Selective vs Unmodified</td><td></td><td>-3.7 [−10.0, 2.3]</td><td>8.7 [1.7, 15.0]</td><td></td><td>0.0 [−1.0, 1.0]</td></tr><tr><td>Selective vs Full SAD</td><td>-8.0 [−13.3, -2.7]</td><td></td><td></td><td>15.0 [8.0, 22.0]</td><td>1.0 [0.0, 2.3]</td></tr><tr><td>Selective vs Random selection</td><td></td><td>-2.0 [−7.3, 3.0]</td><td></td><td>3.7 [−1.3, 9.0]</td><td>0.3 [0.0, 1.0]</td></tr><tr><td>Warm-up vs Unmodified</td><td></td><td>-0.3 [−6.7, 5.7]</td><td>-9.0 [−16.0, -2.0]</td><td></td><td>-1.3 [-3.7, 0.3]</td></tr><tr><td>Warm-up vs Full SAD</td><td></td><td>-4.7 [−9.7, 0.3]</td><td>-2.7 [−10.3, 5.0]</td><td></td><td>−0.3 [−2.3, 1.3]</td></tr><tr><td>Warm-up vs Random selection</td><td></td><td>1.3 [−3.3, 6.3]</td><td>-14.0 [−20.0, -8.0]</td><td></td><td>-1.0 [−2.3, 0.0]</td></tr></table>

## I UNIFORM-STATE DIFFUSION CONTINUATION RESULTS

Table 34 reports the complete Duo and UDLM comparison under the protocol of Section 5.2. Unsafe rate uses the RTP harmful cohort; benign BERTScore uses the separate benign RTP cohort. Entries report means and standard deviations across five generation seeds.

Table 34: Uniform-state continuation on RTP.
<table><tr><td rowspan="2">Method</td><td colspan="2">Duo</td><td colspan="2">UDLM</td><td rowspan="2">NFE</td></tr><tr><td>Unsafe %↓</td><td>Benign BERT ↑</td><td>Unsafe %↓</td><td>Benign BERT ↑</td></tr><tr><td>Unmodified</td><td> $5 1 . 3 _ { 1 . 8 }$ </td><td> $8 1 . 7 7 _ { 0 . 0 8 }$ </td><td> $6 8 . 5 _ { 2 . 5 }$ </td><td> $8 0 . 8 1 _ { 0 . 0 4 }$ </td><td>128</td></tr><tr><td>SAD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\eta = 3 { \times } 1 0 ^ { 4 }$ </td><td> $4 3 . 4 _ { 3 . 5 }$ </td><td> $8 1 . 7 7 _ { 0 . 1 4 }$ </td><td> $5 7 . 1 \mathrm { _ 1 . 1 }$ </td><td> $8 0 . 5 0 \ L _ { 0 . 0 6 }$ </td><td>128</td></tr><tr><td> $\eta = 5 { \times } 1 0 ^ { 4 }$ </td><td> $4 2 . 0 _ { 1 . 4 }$ </td><td> $8 1 . 5 0 _ { 0 . 1 0 }$ </td><td> $5 2 . 9 _ { 1 . 6 }$ </td><td> $8 0 . 4 0 _ { 0 . 0 9 }$ </td><td>128</td></tr><tr><td> $\eta = 1 0 ^ { 5 }$ </td><td> $4 2 . 8 _ { 2 . 2 }$ </td><td> $8 1 . 0 4 _ { 0 . 0 7 }$ </td><td> $4 7 . 4 _ { 2 . 9 }$ </td><td> $8 0 . 4 9 _ { 0 . 0 3 }$ </td><td>128</td></tr><tr><td> $\eta = 1 . 5 { \times } 1 0 ^ { 5 }$ </td><td> $4 1 . 1 _ { 2 . 2 }$ </td><td> $8 0 . 7 6 _ { 0 . 2 2 }$ </td><td> $4 9 . 5 _ { 4 . 2 }$ </td><td> $8 0 . 5 3 \mathrm { _ 0 . 1 6 }$ </td><td>128</td></tr><tr><td> $\mathbf { S _ { A D D E R } } , k = 6$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\eta = 3 { \times } 1 0 ^ { 4 }$ </td><td> $4 7 . 9 _ { 2 . 3 }$ </td><td> $8 1 . 9 5 _ { 0 . 0 4 }$ </td><td> $5 6 . 3 _ { 1 . 7 }$ </td><td> $8 0 . 6 7 _ { 0 . 1 2 }$ </td><td>128</td></tr><tr><td> $\eta = 5 { \times } 1 0 ^ { 4 }$ </td><td> $4 4 . 6 _ { 4 . 5 }$ </td><td> $8 1 . 8 5 _ { 0 . 0 7 }$ </td><td> $5 2 . 2 _ { 1 . 6 }$ </td><td> $8 0 . 6 8 _ { 0 . 0 4 }$ </td><td>128</td></tr><tr><td> $\eta = 1 0 ^ { 5 }$ </td><td> $4 4 . 2 _ { 1 . 9 }$ </td><td> $8 1 . 6 7 _ { 0 . 1 4 }$ </td><td> $4 6 . 1 _ { 2 . 4 }$ </td><td> $8 0 . 8 0 _ { 0 . 0 7 }$ </td><td>128</td></tr><tr><td> $\eta = 1 . 5 { \times } 1 0 ^ { 5 }$ </td><td> $4 5 . 2 _ { 3 . 6 }$ </td><td> $8 1 . 5 4 _ { 0 . 1 3 }$ </td><td> $4 5 . 1 _ { 2 . 9 }$ </td><td> $8 0 . 8 9 _ { 0 . 1 3 }$ </td><td>128</td></tr><tr><td> $\mathbf { S } \mathbf { A } \mathbf { D } \mathbf { D } \mathbf { E } \mathbf { R } , k = 9$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\eta = 3 { \times } 1 0 ^ { 4 }$ </td><td> $4 7 . 1 _ { 4 . 1 }$ </td><td> $8 1 . 9 3 _ { 0 . 0 7 }$ </td><td> $5 5 . 5 _ { 2 . 1 }$ </td><td> $8 0 . 6 5 _ { 0 . 1 3 }$ </td><td>128</td></tr><tr><td> $\eta = 5 { \times } 1 0 ^ { 4 }$ </td><td> $4 5 . 2 _ { 2 . 3 }$ </td><td> $8 1 . 8 1 _ { 0 . 0 6 }$ </td><td> $5 1 . 5 _ { 3 . 3 }$ </td><td> $8 0 . 6 7 _ { 0 . 0 4 }$ </td><td>128</td></tr><tr><td> $\eta = 1 0 ^ { 5 }$ </td><td> $4 5 . 1 _ { 2 . 7 }$ </td><td> $8 1 . 5 9 _ { 0 . 1 7 }$ </td><td> $4 6 . 5 _ { 2 . 3 }$ </td><td> $8 0 . 8 1 _ { 0 . 0 9 }$ </td><td>128</td></tr><tr><td> $\eta = 1 . 5 { \times } 1 0 ^ { 5 }$ </td><td> $4 4 . 1 1 . 8$ </td><td> $8 1 . 4 1 _ { 0 . 1 5 }$ </td><td> $4 4 . 5 2 . 1$ </td><td> $8 0 . 9 0 _ { 0 . 1 4 }$ </td><td>128</td></tr><tr><td> $\mathbf { S _ { A D D E R } } , k = 1 2$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\eta = 3 { \times } 1 0 ^ { 4 }$ </td><td> $4 7 . 2 _ { 3 . 2 }$ </td><td> $8 2 . 0 3 _ { 0 . 0 8 }$ </td><td> $5 7 . 9 _ { 2 . 1 }$ </td><td> $8 0 . 6 6 _ { 0 . 1 4 }$ </td><td>128</td></tr><tr><td> $\eta = 5 { \times } 1 0 ^ { 4 }$ </td><td> $4 3 . 0 _ { 2 . 5 }$ </td><td> $8 1 . 8 6 _ { 0 . 1 0 }$ </td><td> $5 1 . 9 3 . 0 $ </td><td> $8 0 . 6 9 _ { 0 . 0 2 }$ </td><td>128</td></tr><tr><td> $\eta = 1 0 ^ { 5 }$ </td><td> $4 5 . 0 \phantom { 0 } { 0 . 7 }$ </td><td> $8 1 . 7 7 _ { 0 . 1 6 }$ </td><td> $4 8 . 0 2 . 6 $ </td><td> $8 0 . 8 1 _ { 0 . 1 0 }$ </td><td>128</td></tr><tr><td> $\eta = 1 . 5 { \times } 1 0 ^ { 5 }$ </td><td> $4 3 . 6 _ { 2 . 5 }$ </td><td> $8 1 . 6 8 _ { 0 . 1 0 }$ </td><td> $4 5 . 9 _ { 1 . 7 }$ </td><td> $8 0 . 9 2 _ { 0 . 1 2 }$ </td><td>128</td></tr><tr><td>CSD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $k = 6$ </td><td> $4 8 . 3 _ { 2 . 9 }$ </td><td> $8 1 . 8 0 _ { 0 . 1 1 }$ </td><td> $5 3 . 5 1 . 8$ </td><td> $8 0 . 5 1 \mathrm { _ { 0 . 0 6 } }$ </td><td>128</td></tr><tr><td> $k = 9$ </td><td> $3 5 . 1 _ { 2 . 3 }$ </td><td> $8 1 . 9 3 \mathrm { _ 0 . 0 6 }$ </td><td> $6 3 . 5 \mathrm { 1 . 5 }$ </td><td> $8 0 . 8 1 \mathrm { _ { 0 . 0 6 } }$ </td><td>128</td></tr><tr><td> $k = 1 2$ </td><td> $5 0 . 6 _ { 2 . 1 }$ </td><td> $8 1 . 7 8 _ { 0 . 0 8 }$ </td><td> $6 7 . 1 _ { 4 . 2 }$ </td><td> $8 0 . 8 1 _ { 0 . 0 4 }$ </td><td>128</td></tr><tr><td> $\mathbf { B e s t - o f } { \mathbf { - } } N$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $N = 3$ </td><td> $4 2 . 2 _ { 2 . 3 }$ </td><td>82.05</td><td> $7 0 . 0 _ { 3 . 4 }$ </td><td>80.80</td><td>384</td></tr><tr><td> $N = 5$ </td><td> $3 7 . 6 _ { 2 . 1 }$ </td><td>82.10</td><td> $6 8 . 9 _ { 2 . 1 }$ </td><td>80.72</td><td>640</td></tr><tr><td> $N = 8$ </td><td> $3 3 . 1 _ { 2 . 3 }$ </td><td>82.16</td><td> $6 9 . 7 _ { 2 . 2 }$ </td><td>80.77</td><td>1024</td></tr></table>

## J GUIDANCE ANALYSIS AND ABLATIONS FOR UNIFORM-STATE DIFFUSION

## J.1 WHERE THE GATE OPENS

Figure 3 plots the open-gate ratio during unguided continuations of unsafe RTP prompts. The opengate ratio generally increases as denoising proceeds, consistent with position-wise representations becoming more informative as the continuation takes shape. SADDER uses this evolving signal to select positions for guidance, whereas SAD applies its correction to all eligible positions at every step.

## J.2 SOFTENING THE GATE

We replace the binary gate with $a _ { k , l } ^ { ( t ) } = \operatorname* { m a x } \left\{ 2 \sigma \left( s _ { k , l } ^ { ( t ) } / \tau \right) - 1 , 0 \right\}$ , where σ is the sigmoid function and τ controls the softness of the gate. The transformation makes the gate zero when $s _ { k , l } ^ { ( t ) } = 0$ , and clipping at zero leaves positions with negative scores unguided. For positive scores, the gate takes a value between zero and one; the strict gate is recovered as $\tau  0$

![](images/aabe4aa37300137f5b4696d3c4c0bfd72dc406e04f201e355b90210c4e0a9684.jpg)  
(a) Duo

![](images/daa31adefc83a7858be350f49134522e6eb5b5456a56b824f197b9c2232c5b96.jpg)  
(b) UDLM

Figure 3: Open-gate ratio along unguided continuations of unsafe RTP prompts: the fraction of positions with $s _ { k , l } ^ { ( t ) } > 0$ . The horizontal axis runs from early to late denoising steps.  
![](images/02bc90d24feb6d68846368feb53fc2cfd8bd44c588fde9a47cbe275d244745d5.jpg)  
Figure 4: Unsafe rate and benign BERTScore across guidance strengths for the strict gate and softened gates with $\tau \in \{ 3 0 , 7 0 , 1 0 \mathrm { { 0 } } \}$ on Duo $( k = 9 )$ and UDLM $( k = \bar { 1 } 2 )$ . Error bars are one standard deviation across seeds: five seeds for the strict gate (the runs reported in Table 34) and three seeds for the softened gates.

Figure 4 shows no consistent effect of the temperature. On UDLM the four gates coincide in both metrics across the whole range. On Duo the strict gate trends toward lower unsafe rates and lower benign BERTScore, but the differences are smaller than the seed-to-seed variation of either gate. We therefore keep the strict indicator, which needs no additional hyperparameter.