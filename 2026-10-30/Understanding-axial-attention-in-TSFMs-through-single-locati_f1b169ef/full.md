# Understanding axial attention in TSFMs through single location regression

Andrei Tiberiu Pantea<sup>∗</sup> CMAP, École polytechnique, IP Paris Electricité de France R&D Palaiseau, France

Aymeric Dieuleveut CMAP, École polytechnique, IP Paris Palaiseau, France

## Abstract

Recent advances in Time Series Foundation Models (TSFMs) increasingly rely on axial attention mechanisms, but despite their empirical success, the theoretical properties of axial attention remain largely unexplored. To bridge this gap, we extend the Single Location Regression framework introduced in Marion et al. [2025] to 2D structured data, in order to study sparse information settings where a target depends on a small number of tokens. We formulate an analytically tractable axial attention-type predictor and rigorously analyze its statistical properties and training dynamics. We prove that under a specific asymptotic regime and with knowledge of the underlying oracle parameters, our predictor achieves Bayes optimality, whereas a class of semi-linear simplifications strictly fails to do so, underlining the role of the inner non-linearity. We prove that for a specific temperature scaling, Projected Gradient Descent converges globally to the optimal oracle parameters. Extensive numerical experiments validate our theoretical findings and highlight the critical role of inverse temperature scheduling in empirical convergence.

## 1 Introduction

Recent years have witnessed a surge in the development of Time Series Foundation Models (TSFMs) leveraging transformer architectures [Vaswani et al., 2017] (e.g., Chronos-2 [Ansari et al., 2025], Moirai 2.0 [Liu et al., 2025], TimesFM [Das et al., 2024], TabPFN-TS [Hoo et al., 2025], Toto [Cohen et al., 2025], PatchTST-FM [Wen et al., 2026]). However, applying standard global attention to high-dimensional multivariate time series presents a significant computational bottleneck. Standard attention treats inputs as a flattened 1D sequence, which scales quadratically with the total number of tokens and fails to explicitly leverage the natural 2D grid structure (time steps × channels) of multivariate data. To address this, axial attention introduced in Ho et al. [2019], and specifically for multivariate time series in Zhang and Yan [2023], has emerged as a powerful alternative already implemented in some of the state-of-the-art TSFMs, e.g., Chronos-2, TabPFN-TS and Toto, but also in task specific models such as CEReBrO [Dimofte et al., 2025], an EEG foundation model that utilizes alternating intra-channel temporal and inter-channel spatial attention and STAIT (Spatio-Temporal Alternating Iterative Transformer) [Cui et al., 2026] for remote sensing.

The axial-attention layer combines two token-mixing half-layers, as described in Figure 1. Formally, we consider our input time series as $M \times N$ matrices of tokens $\mathbb { X } = ( X _ { i j } ) _ { 1 \le i \le M , 1 \le j \le N }$ , taking values in the embedding (or latent) space $\mathbb { R } ^ { d }$ , where M can be seen as the number of channels and N as the number of time series patches. Generically, given as hyperparameters two real numbers $\lambda _ { r } , \lambda _ { c } ,$ referred to as inverse temperatures, and an activation function $\sigma : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , the standard axial self-attention layer is parametrized by weight matrices $Q _ { r } , K _ { r } \in \mathbb { R } ^ { d \times p _ { r } } , V _ { r } \in \mathbb { R } ^ { d \times d }$

![](images/c96bd91fc7d57ca21dfc95ea7a1276491670e4f1dd56bc537c77a974e2157f8d.jpg)  
Figure 1: Schematic view of an axial attention layer. Raw input tokens are first processed by a time-wise self-attention layer that isolates features along the rows. A subsequent channel-wise attention layer vertically blends these extracted features, effectively capturing global cross-channel and temporal context.

$Q _ { c } , K _ { c } , V _ { c } \in \mathbb { R } ^ { d \times p _ { c } }$ , and $O _ { c } \in \mathbb { R } ^ { p _ { c } \times o _ { c } }$ . The subscripts r and c represent row-wise and, respectively, column-wise operations. The mapping is defined as, for any $j \in [ N ]$

$$
T _ { \lambda _ { r } , \lambda _ { c } , \sigma } ^ { ( Q _ { r , c } , K _ { r , c } , V _ { r , c } , O _ { c } ) } ( \mathbb { X } ) _ { : j } = \sigma \Bigl ( \lambda _ { c } \underbrace { \mathbb { X } _ { : j } ^ { ( 1 / 2 ) } Q _ { c } } _ { M \times p _ { c } } \underbrace { K _ { c } ^ { T } \left( \mathbb { X } _ { : j } ^ { ( 1 / 2 ) } \right) ^ { T } } _ { p _ { c } \times M } \Bigr ) \underbrace { \mathbb { X } _ { : j } ^ { ( 1 / 2 ) } V _ { c } } _ { M \times p _ { c } } \underbrace { O _ { c } } _ { p _ { c } \times o _ { c } } ,\tag{1}
$$

where the initial row-wise computation is, for any $i \in [ M ]$

$$
\mathbb { X } _ { i \cdot } ^ { ( 1 / 2 ) } = \sigma ( \lambda _ { r } \underbrace { \mathbb { X } _ { i \cdot } Q _ { r } } _ { N \times p _ { r } } \underbrace { K _ { r } ^ { T } \mathbb { X } _ { i \cdot } ^ { T } } _ { p _ { r } \times N } ) \underbrace { \mathbb { X } _ { i \cdot } V _ { r } } _ { N \times d } ,\tag{2}
$$

where we use the notations $[ n ] = \{ 1 , 2 , \dots , n \}$ for any positive integer $^ { n , }$ and $W _ { i }$ <sub>·</sub> (resp. $W _ { \cdot j } )$ to denote the i-th line (resp. j-th column) of a matrix $W$ . Note that in (2) (and throughout the paper when applied to a matrix) the activation function is used row-wise. When σ is the softmax function, this structure is one of the building blocks used in Chronos-2, TabPFN-TS and Toto, stripped of residual connections and normalization layers. Note that the order in which the row attention and column attention are performed has no impact on the results in our paper; we choose row attention followed by column attention just for presentation purposes.

Although the empirical success of these axial architectures is evident across various domains, their theoretical properties remain largely unexplored. For example, in Berthelier et al. [2026], the authors have shown that pretrained Chronos-2 is not always capable of identifying simple relationships between targets and covariates in the token-wise sparse information setting. Our goal is to improve the theoretical understanding of axial attention by analyzing their ability to approximate and learn the optimal predictor in this setting. To this end, we build on the work of Marion et al. [2025], conducted in the context of classical transformers and motivated by natural language processing (NLP) tasks. In that paper, the authors introduced a new class of statistical models, namely Single Location Regression (SLR) models, through which they aimed to investigate the ability of standard attention layers to recover relevant information in a token-wise sparse information setting, where the target depends only on a relatively small number of tokens. They have provided theoretical guarantees for a modification of the standard transformer architecture, proving that one can recover optimal predictors for SLRs. This naturally raises a fundamental, yet unanswered, question: Can we achieve the same statistical and optimization performance with axial attention as we do with normal global attention? The answer is not obvious, since the information flow is much more restricted in an axial transformer, where each token only attends to tokens with which it shares an axis, as opposed to the free flow of information between any two tokens in a standard transformer.

Building upon the foundation established by Marion et al. [2025], we introduce a SLR model and extend the theoretical analysis of transformer architectures to axial attention mechanisms, as a method of assessing the theoretical capabilities of current state-of-the-art TSFMs. Our main contributions are structured as follows:

• In Section 2, we introduce an SLR statistical model for the token-wise sparse information setting adapted to 2D structured data (e.g., multivariate time series, tables). We then explain the connection between such a model and multivariate time series anomaly analysis, and finally, we introduce a trainable predictor and show that it corresponds to a slightly simplified axial attention layer.

• In Section 3, we analyze the performance of the predictor introduced and prove that there exists an asymptotic regime in which it is statistically optimal for the SLR model investigated, when using the oracle parameters. Furthermore, we show that removing the inner non-linearity in our predictor renders it suboptimal.

• In Section 4, we propose an optimization algorithm for our predictor and provide theoretical guarantees for its convergence, despite the non-convexity of the problem. We further conduct extensive numerical studies which reveal a complex optimization landscape that confirm our theoretical findings.

Our structure closely follows that of Marion et al. [2025], but our motivation (the study of TSFMs) and the technical challenges are, nevertheless, very different. Standard global attention benefits from symmetric token coupling, which facilitates the derivation of closed-form analytical solutions for the expected risk. By contrast, axial attention operates in a sequential manner. This decoupled architecture breaks the global symmetry and induces a cascading non-linear dependence between axes. As a result, the optimization landscape exhibits significant complexity, rendering standard closed-form approaches to statistical optimality and convergence of optimization methods intractable and necessitating novel analytical techniques.

## 2 SLR model and simplified axial attention predictor

In this section, we introduce an SLR model and underline its connection to multivariate time series anomaly analysis. Afterwards, we consider a simplification of axial attention that renders the statistical analysis in Section 3 tractable.

Single location regression model. We focus on the following statistical model, where the output $Y \in \mathbb { R }$ is assumed to be given by

$$
Y = X _ { I _ { 0 } J _ { 0 } } ^ { T } v ^ { * } + \xi ,\tag{SLR}
$$

where $\xi \sim \mathcal { N } ( 0 , \epsilon ^ { 2 } )$ is independent noise and $I _ { 0 } , J _ { 0 }$ are discrete random variables over $[ M ]$ and $[ N ]$ , respectively. No restriction on the distribution of $( I _ { 0 } , J _ { 0 } )$ is imposed unless otherwise specified. Conditionally on $( I _ { 0 } , J _ { 0 } )$ , for $( \ell , \gamma ^ { 2 } , \epsilon ^ { 2 } , k ^ { * } , v ^ { * } ) \in \mathbb { R } _ { + } \times \mathbb { R } _ { + } ^ { * } \times \mathbb { R } _ { + } ^ { * } \times \mathbb { S } ^ { d - 1 } \times \mathbb { S } ^ { d - 1 }$

$$
\begin{array} { r } { \left\{ \begin{array} { l l l } { X _ { I _ { 0 } J _ { 0 } } } & { \sim } & { \mathcal { N } \left( \ell k ^ { * } , \gamma ^ { 2 } I _ { d } \right) , } \\ { X _ { i j } } & { \sim } & { \mathcal { N } \left( 0 , I _ { d } \right) , \mathrm { \ f o r \ } ( i , j ) \neq ( I _ { 0 } , J _ { 0 } ) , } \end{array} \right. } \end{array}
$$

where $X _ { i j }$ are independent for all $( i , j ) \in [ M ] \times [ N ]$

The parameters of (SLR) are the shift magnitude ℓ, the localization and noise variances $\gamma ^ { 2 }$ and $\epsilon ^ { 2 } { . }$ , the unobserved projection vector $v ^ { * }$ and the localization vector $k ^ { * }$ , with both vectors taking values on the d-dimensional unit sphere $\mathbb { S } ^ { d - 1 }$ . The target variable Y depends exclusively on a single embedded token within the $M \times { \bar { N } }$ input matrix, positioned at random discrete coordinates $( { \bar { I } } _ { 0 } , J _ { 0 } )$ This active token is statistically identifiable because its expected value is shifted along the direction $k ^ { * }$ Consequently, the primary mathematical challenge of this task stems from its dynamic sparsity.

Remark 1. Unlike in Marion et al. [2025], we consider ℓ to be decoupled from d throughout our analysis, but note that for $\ell = \sqrt { { d } / { 2 } }$ and $\gamma ^ { 2 } = 1 / 2$ , all tokens have the same expected norm, i.e. $\mathbb { E } [ \| X _ { I _ { 0 } J _ { 0 } } \| _ { 2 } ^ { 2 } ] = \mathbb { E } [ \| X _ { i j } \| _ { 2 } ^ { 2 } ] = d .$ making them indistinguishable just based on their magnitude.

Connection to multivariate time series anomaly analysis. The Single Location Regression (SLR) model detailed above provides a natural theoretical framework for multivariate time series anomaly analysis, particularly when operating on learned representation embeddings X. We consider systemwide anomalous events identifiable through correlated behaviors across multiple sensors, as pictured in Figure 2. We operate under the assumption that the embedding layer has already captured these correlated dynamics, localizing the signature of the system-wide anomaly into the embedding of a single primary driving sensor $I _ { 0 }$ at a specific time step $J _ { 0 }$ . By investigating whether axial attention, which restricts information flow, can successfully recover the true parameters $k ^ { * }$ and $v ^ { * }$ , we can theoretically assess its ability to isolate and leverage localized anomalies.

Furthermore, using axial attention layers instead of standard global attention is attractive also from a computational point of view. The global attention mechanism treats the input as a flattened 1D sequence of length $M \times N$ and this coupling requires $\mathcal { O } ( M ^ { 2 } N ^ { 2 } )$ operations per layer, while axial attention only requires $\mathcal { O } ( M N ^ { 2 } + M ^ { 2 } \dot { N } )$ (see Ho et al. [2019] for a derivation).

![](images/8939dd655833944e0633351c2e9901fc8a5417105dc070838f85574f70a2f226.jpg)  
Figure 2: Under normal operating conditions, the background embeddings behave as uninformative noise, or as a standard token, modeled by the standard normal distribution $\mathcal { N } ( 0 , I _ { d } )$ . When an anomaly occurs in one sensor, its specific signature is encoded into the token $X _ { I _ { 0 } J _ { 0 } } ,$ characterized by a shifted mean $\ell k ^ { * }$ and a localization variance $\gamma ^ { 2 } I _ { d } .$ . The regression target Y represents a downstream consequence or system-level anomaly score that depends strictly on identifying this specific embedded state via the projection vector $v ^ { * }$ . Our goal is to investigate whether we can recover this target through an axial attention layer.

An analytically tractable simplification of axial attention for SLR. In what follows, we introduce a class of predictors that are direct simplifications of the axial attention layer. For an activation $\sigma \colon  { \mathbb { R } ^ { d } } \to \dot {  { \mathbb { R } ^ { d } } }$ and inverse temperatures $\lambda _ { r } , \lambda _ { c } \in ( 0 , \infty )$ , let

$$
\begin{array} { r } { T _ { \lambda _ { r } , \lambda _ { c } , \sigma } ^ { ( k , v ) } ( \mathbb { X } ) : = \sigma ( \lambda _ { c } \underbrace { \mathbb { X } ^ { ( 1 / 2 ) } k } _ { M \times 1 } ) ^ { T } \ \underbrace { \mathbb { X } ^ { ( 1 / 2 ) } v } _ { M \times 1 } , \ \mathrm { ~ w h e r e ~ } \mathbb { X } _ { i } ^ { ( 1 / 2 ) } : = \sigma ( \lambda _ { r } \underbrace { \mathbb { X } _ { i } . k } _ { N \times 1 } ) ^ { T } \ \underbrace { \mathbb { X } _ { i \cdot , \cdot } } _ { N \times d } , } \end{array}\tag{3}
$$

for $i \in [ M ]$ and learnable parameters $k , v \in \mathbb { S } ^ { d - 1 }$ . Note that we slightly overload the notation and use $\mathbb { X } ^ { ( 1 / 2 ) }$ to denote the half step of both our predictor and of axial attention.

In the context of TSFMs, sequences are typically appended with designated [REG] tokens to act as readout tokens or attention sinks (see Ansari et al. [2025], for example). Assuming that the input X is appended with a $( N + 1$ )th column of row-aggregation [REG\_ROW] tokens and a column aggregation [REG\_COL] token at position $( M + 1 , N + \mathbf { \bar { 1 } } )$ , we have the following lemma

Lemma 1. For $p _ { r } = p _ { c } = 1 , o _ { c } = 1 , O _ { c } = 1$ , together with shared keys $K _ { r } = K _ { c }$ and $V _ { r } = I _ { d } ,$ , we have thefollowing equivalence:

$$
T _ { \lambda _ { r } , \lambda _ { c } , \sigma } ^ { ( Q _ { r , c } , K _ { r , c } , V _ { r , c } , O _ { c } ) } ( \mathbb { X } ) _ { M + 1 , N + 1 } = T _ { \lambda _ { r } a , \lambda _ { c } a ^ { ( 1 / 2 ) } , \sigma } ^ { ( K _ { c } , V _ { c } ) } ( \mathbb { X } ) ,
$$

for some constants a and $a ^ { ( 1 / 2 ) }$ . The proof and further comments on the constraints are in Appendix A.

The main bottleneck in trying to analyze σ = softmax in (3) is the softmax operation that induces a global coupling between the tokens, which acts as a major roadblock to rigorous statistical analysis. To bypass this issue without losing the core behavioral properties of the network, we substitute softmax with the component-wise nonlinear function er $\begin{array} { r } { \dot { \mathrm { f } ( x ) } = \frac { 2 } { \sqrt { \pi } } \int _ { 0 } ^ { x } e ^ { - u ^ { 2 } } \mathrm { d } u } \end{array}$ , following Marion et al. [2025]. This adjustment leads to the definition of the analytically tractable predictor that we investigate throughout the remainder of the paper,

$$
\begin{array} { r } { T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) : = T _ { \lambda _ { r } , \lambda _ { c } , \mathrm { e r f } } ^ { ( k , v ) } ( \mathbb { X } ) = \mathrm { e r f } ( \lambda _ { c } \mathbb { X } ^ { ( 1 / 2 ) } k ) ^ { T } \mathbb { X } ^ { ( 1 / 2 ) } v , \mathrm { w h e r e } \mathbb { X } _ { i } ^ { ( 1 / 2 ) } : = \mathrm { e r f } ( \lambda _ { r } \mathbb { X } _ { i } . k ) ^ { T } \mathbb { X } _ { i } . } \end{array}\tag{4}
$$

## 3 Statistical analysis

Having established the predictor $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) }$ , we now turn to evaluating its statistical properties.

## 3.1 Asymptotic optimality of the oracle predictor

In this section, we follow the approach of Marion et al. [2025]: we first observe that our predictor only depends on the alignments between $k , v , k ^ { * }$ and $v ^ { * }$ , projecting the high-dimensional risk landscape down to five scalars, which is essential to the analysis of the training dynamics in Section 4. Second, under the assumption of orthogonality between $\bar { k } ^ { * }$ and $v ^ { * }$ , we obtain a simplified formula for the risk. Finally, using this formula, we find an asymptotic regime in which our predictor is optimal. Although we follow the same structure and maintain similar notation as Marion et al. [2025], in order to optimize readability, the sequential nature of axial attention makes the results and the proofs more complicated as we lack a closed form formula for the risk. Detailed proofs for all theoretical claims in this section can be found in Appendix B.

We quantify the model’s performance via its risk:

$$
\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k , v ) : = \mathbb { E } \left[ \left( Y - T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) \right) ^ { 2 } \right] .\tag{5}
$$

Furthermore, we make the following assumption throughout the rest of the paper.

Assumption 1. The vectors $k ^ { * } , v ^ { * } \in \mathbb { S } ^ { d - 1 }$ are orthogonal to each other i.e. $k ^ { * } { } ^ { T } v ^ { * } = 0$

Note that enforcing orthogonality between $k ^ { * }$ and $v ^ { * }$ is a mild requirement in high-dimensional embedding spaces, commonly found in TSFMs. The following lemma establishes a reduced representation of the risk of our axial estimator based on the alignments between $k , v , k ^ { * }$ and $v ^ { * }$

Lemma 2. There exists afunction $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } : \mathbb { R } ^ { 5 }  ]$ R such thatfor any $( k , v ) \in \left( \mathbb { S } ^ { d - 1 } \right) ^ { 2 }$

$$
\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k , v ) = \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu , \theta , \eta , \rho ) ,\tag{6}
$$

where $\kappa = k ^ { T } k ^ { * } , \nu = v ^ { T } v ^ { * } , \theta = v ^ { T } k ^ { * } , \eta = k ^ { T } v ^ { * }$ and $\rho = k ^ { T } v$ . An expressionfor $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ is given in Appendix B.

Next, the Bayes risk for the regression task in (SLR) is bounded below by the variance $\epsilon ^ { 2 }$ of the noise ξ. Building on Lemma 2, Proposition 3 demonstrates that our oracle estimator $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k ^ { * } , v ^ { * } ) }$ recovers this theoretical lower bound under a specific asymptotic regime.

Proposition 3. Assume a joint asymptotic scaling where $\lambda _ { r } \ell \to \infty , \lambda _ { c } \ell \to \infty$ and $\begin{array} { r l } { \lambda _ { r } } & { { } = } \end{array}$ $o \left( M ^ { - \frac { 1 } { 2 } } N ^ { - \frac { 1 } { 2 } } \right)$ . Then,

$$
\begin{array} { r } { \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k ^ { * } , v ^ { * } )  \epsilon ^ { 2 } . } \end{array}
$$

Consequently, the oracle axial predictor $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k ^ { * } , v ^ { * } ) }$ achieves asymptotic Bayes optimality.

Note that Proposition 3 does not impose any scaling on $d ,$ which is expected given Lemma 2. We provide a non-rigorous intuition for the imposed scalings. First, we wish $X _ { I _ { 0 } J _ { 0 } }$ to pass through the first layer asymptotically unaltered, that is,

$$
X _ { I _ { 0 } J _ { 0 } } \approx \mathbb { X } _ { I _ { 0 } } ^ { ( 1 / 2 ) } = \mathrm { e r f } ( \underbrace { \lambda _ { r } X _ { I _ { 0 } J _ { 0 } } ^ { T } k ^ { * } } _ { \sim \lambda _ { r } \ell } ) X _ { I _ { 0 } J _ { 0 } } + \sum _ { j \neq J _ { 0 } } \mathrm { e r f } ( \lambda _ { r } X _ { I _ { 0 } j } ^ { T } k ^ { * } ) X _ { I _ { 0 } j } .
$$

Therefore, we need $\lambda _ { r } \ell \to \infty$ to ensure that the argument of the first erf goes to ∞ and $\lambda _ { r } \to 0$ to ensure that the others go to 0. In an analogous manner, we wish $X _ { I _ { 0 } J _ { 0 } }$ to pass unaltered through the second layer, which leads to $\lambda _ { c } \ell \to \infty$ . Lastly, in the expansion of $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( \check { k } ^ { * } , v ^ { * } ) }$ , there appear $( M - 1 ) N$ noise terms of the form $C \cdot \mathrm { e r f } ( \lambda _ { r } X _ { i j } ^ { T } k ^ { * } ) X _ { i j } ^ { T } v ^ { * }$ , with $C < 1$ , one for each $( i , j )$ with $\dot { \iota } \neq I _ { 0 }$ . Multiple scalings could make the sum of these terms ${ \bf g 0 }$ to $0 ,$ one of them being $\lambda _ { r } = o \left( M ^ { - \frac { 1 } { 2 } } N ^ { - \frac { 1 } { 2 } } \right)$ , as shown in the proof in Appendix B.

## 3.2 Removing the inner non-linearity is suboptimal

To fully appreciate the asymptotic optimality of our oracle predictor, it is instructive to compare it against semi-linear models, which replace the inner non-linearity with a fixed linear combination layer governed by $\boldsymbol { \beta } \in \mathbb { R } ^ { N }$ . This justifies the necessity for the inner non-linearity in the axial attention layer in order to achieve asymptotic optimality. More precisely, consider the semi-linear predictor

$$
T _ { \beta , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) : = \mathrm { e r f } ( \lambda _ { c } \mathbb { X } ^ { ( 1 / 2 ) } k ) ^ { T } \mathbb { X } ^ { ( 1 / 2 ) } v , ~ \mathrm { w h e r e } ~ \mathbb { X } _ { i } ^ { ( 1 / 2 ) } : = \beta ^ { T } \mathbb { X } _ { i } . ~ .\tag{7}
$$

Then we have the following sub-optimality statement.

Proposition 4. Let $p _ { j } = \mathbb { P } ( J _ { 0 } = j ) f o r j \in [ N ]$ . Denote by $S _ { i } = \beta ^ { T } \mathbb { X } _ { i }$ <sub>·</sub> Then, for any function $f \colon  { \mathbb { R } ^ { d M } } \to$ R in $L ^ { 2 } ( \dot { \mathbb { R } } ^ { d M } )$ , the semi-linear predictor achieves a risk strictly larger than ϵ<sup>2</sup> for any choice of $\cdot \beta \neq 0$ and hyperparameters $M , N , \ell , \lambda _ { c }$ i.e.

$$
\mathbb { E } \left[ ( Y - f ( S _ { 1 } , S _ { 2 } , \ldots , S _ { M } ) ) ^ { 2 } \right] \ge \epsilon ^ { 2 } + \gamma ^ { 2 } \left( 1 - \sum _ { j _ { 0 } = 1 } ^ { N } p _ { j _ { 0 } } \frac { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } } { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \ne j _ { 0 } } \beta _ { j } ^ { 2 } } \right) > \epsilon ^ { 2 } ,
$$

if there is no $j \in [ N ]$ with $p _ { j } = 1$ . We provide a proofofthis statement in Appendix B.2.

Remark 2. Discarding the outer non-linearity leads to a less restrictive information flow compared to our predictor, and it amounts to a simplification of the standard attention mechanism. Indeed,

$$
T _ { \lambda _ { r } , \beta } ^ { ( k , v ) } ( \mathbb { X } ) : = \sum _ { i = 1 } ^ { M } \beta _ { i } \sum _ { j = 1 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } X _ { i j } ^ { T } k \right) X _ { i j } ^ { T } v
$$

is a modification of the trainable predictor proposed in Marion et al. [2025], which is asymptotically optimal for $\beta _ { i } = 1 , i \in [ M ]$ , under the joint scaling $\lambda _ { r } = o \left( M ^ { - \frac { 1 } { 2 } } N ^ { - \frac { 1 } { 2 } } \right)$ and $\lambda _ { r } \ell \to \infty$

## 4 Gradient descent dynamics

The last section of our analysis is concerned with the theoretical training dynamics of our predictor (4) for the risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k , v )$ . While in practice training minimizes an empirical risk evaluated over a finite dataset, we limit our investigation of the empirical risk minimization to numerical simulations, which confirm our findings about the theoretical risk.

## 4.1 Convergence of gradient descent

Lemma 2 shows that on the manifold $\left( \mathbb { S } ^ { d - 1 } \right) ^ { 2 }$ the risk can be described through 5 alignment metrics. We want to use this fact and formula since it renders the analysis tractable after further constraints. Therefore, it is natural to employ Projected Riemannian Gradient Descent (PGD) as an optimization algorithm for two main reasons: it keeps the parameters on the manifold and it guarantees that the Euclidean projection back onto the sphere remains well-defined by preventing the parameters from reaching the origin. We emphasize that this geometric constraint is necessary because our reduced risk formulation (6) is strictly valid only on this manifold.

Algorithm 1 Projected Riemannian Gradient Descent   
Input: Initialization $( k _ { 0 } , v _ { 0 } ) \in ( \mathbb { S } ^ { d - 1 } ) ^ { 2 }$ , step size $\alpha > 0 ,$ , inverse temperatures sequence $( \lambda _ { r } ^ { t } , \lambda _ { c } ^ { t } ) _ { t \geq 0 }$   
1: for $t \geq 0$ do   
2: $\boldsymbol { k } _ { t + 1 } ^ { - }  \mathrm { P r o j } _ { \mathbb { S } ^ { d - 1 } } ( \boldsymbol { k } _ { t } - \alpha ( I _ { d } - \boldsymbol { k } _ { t } \boldsymbol { k } _ { t } ^ { T } ) \nabla _ { k } \mathcal { R } _ { \lambda _ { r } ^ { t } , \lambda _ { c } ^ { t } } ( \boldsymbol { k } _ { t } , \boldsymbol { v } _ { t } ) )$   
3: $\boldsymbol { v } _ { t + 1 }  \operatorname { P r o j } _ { \mathbb { S } ^ { d - 1 } } ( \boldsymbol { v } _ { t } - \alpha \big ( \boldsymbol { I } _ { d } - \boldsymbol { v } _ { t } \boldsymbol { v } _ { t } ^ { T } \big ) \nabla _ { \boldsymbol { v } } \mathcal { R } _ { \lambda _ { r } ^ { t } , \lambda _ { c } ^ { t } } \big ( k _ { t } , \boldsymbol { v } _ { t } \big ) )$   
4: end for

Analyzing the dynamics of the resulting 5-dimensional system is intractable due to the highly complex nature of $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ . To enable a rigorous analysis, we restrict our focus to scenarios in which the parameters are initialized on the submanifold M defined by

$$
\begin{array} { r } { \mathcal { M } : = \left\{ \left( k , v \right) \in \left( \mathbb { S } ^ { d - 1 } \right) ^ { 2 } ~ \middle | ~ k ^ { T } v ^ { * } = 0 , ~ v ^ { T } k ^ { * } = 0 , ~ k ^ { T } v = 0 \right\} . } \end{array}\tag{8}
$$

Under Assumption 1, the true underlying parameters $( k ^ { * } , v ^ { * } )$ naturally reside within M and, crucially, the training parameters remain on the submanifold throughout the optimization process if they are initialized on it, from Lemma 6 in Appendix C. Due to this invariance, we only need to characterize the dynamics of the system restricted to M when initialized on this manifold. Although in practice this cannot be enforced, we show empirically in Section 4.2 that penalizing a proxy for the distance to the manifold leads to similar dynamics. This lower-dimensional analysis is mathematically tractable and leads to the primary optimization result of this paper, which guarantees asymptotic recovery of the oracle parameters.

Theorem 5. Consider $\tau > 0$ . Then, there exists $\bar { \lambda } > 0$ and $\bar { \alpha } > 0$ such thatfor constant inverse temperatures $\bar { \lambda } > \lambda _ { r } > 0$ and $\lambda _ { c } = \tau \lambda _ { r } ,$ for any step size $\alpha \leq \bar { \alpha }$ andfor a generic initialization $( k _ { 0 } , \bar { v } _ { 0 } ) \in \mathcal { M } ,$

$$
( k _ { t } , v _ { t } ) \to \pm ( k ^ { * } , v ^ { * } ) \quad a s \ t \to \infty .
$$

The convergence to $( - k ^ { * } , - v ^ { * } )$ is not a problem, as by symmetry around $0 , T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k ^ { * } , v ^ { * } ) } = T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( - k ^ { * } , - v ^ { * } ) }$

Remark 3. Although some assumptions made throughout the paper are identical to the ones in Marion et al. [2025], our main result Theorem 5 is qualitatively different from theirs. In their simplified standard (single-axis) attention layer case, convergence is always reached, regardless of the inverse temperatures. On the contrary, here, as shown analytically in the proof of Theorem 5 and numerically in Section 4.2 and Appendix E.4, the values of the inverse temperatures greatly impact the optimization landscape and convergence is not attainable in general. In particular, global convergence is only achieved below a specific threshold λ<sup>¯</sup>.

Remark 4 (Training and inference temperatures). The inverse temperatures used during optimization and inference serve different purposes and are therefore allowed to differ. During training, we use inverse temperatures $\left( \lambda _ { r } ^ { \mathrm { t r a i n } } , \lambda _ { c } ^ { \mathrm { t r a i n } } \right)$ satisfying the conditions of Theorem 5. In particular, they are sufficiently small and satisfy $\lambda _ { c } ^ { \mathrm { t r a i n } } = \tau \lambda _ { r } ^ { \mathrm { t r a i n } }$ , which guarantees recovery of $( k ^ { * } , v ^ { * } )$ up to their joint sign for generic initializations on $\mathcal { M } .$ . After training, the recovered parameters are frozen and the inverse temperatures are reset to inference values $( \lambda _ { r } ^ { \mathrm { i n f } } , \lambda _ { c } ^ { \mathrm { i n f } } )$ satisfying the scaling conditions of Proposition 3. The resulting predictor is then asymptotically Bayes optimal.

Thus, the two results provide a sequential two-stage guarantee: parameter recovery is performed in an optimization-friendly temperature regime, followed by prediction in a statistically optimal inference regime. This separation is also used in our experiments, where the evaluation risk is computed using fixed inverse temperatures independently of the schedule used for optimization.

## 4.2 Understanding the dynamics - numerical experiments

In practice, one cannot use the theoretical risk for optimization using PGD, since there is no closed form formula for it. Furthermore, one also does not have knowledge of what values for $\lambda _ { r }$ and $\lambda _ { c }$ are small enough. In our experiments, we bypass these issues by using batch stochastic PGD for the optimization process and implementing a decreasing schedule for the inverse temperatures.

![](images/a3c393d6f2c4618d143df5f32d6f55698d16d61aad297c72fa0371eb2b1614e7.jpg)

![](images/15966fddbe006ddfe4a3c5ce541bdf6a776020a292aa0059f725a8e917585756.jpg)

![](images/0e9f673507c668da874f02e8ef62895204db6ce333d8b9be538176cd490d09ee.jpg)

![](images/e2e284ef8714e56b91230a501c52210cc73849ed6c87338a3add803292bdf21f.jpg)  
Figure 3: Dynamics of stochastic PGD to the oracle parameters over 20 independent runs. The solid curve represents the empirical median across the runs and the bands represent the 0.05-0.95 and, respectively, the 0.25-0.75 quantile intervals (although not always visible). Left: The empirical risk obtained by our predictor on the training batches of data using both the training parameters $k ,$ v and the oracle parameters $k ^ { * } , v ^ { * }$ . For computing the oracle and model losses we use the initial inverse temperatures. Middle-left: The alignment between the training and oracle parameters $\kappa = k ^ { T } k ^ { * }$ $\nu = v ^ { T } v ^ { * }$ . Middle-right: The distance from the manifold $\bar { \mathcal { M } }$ of the training parameters $( k , v )$ Right: The schedule used for $\lambda = \lambda _ { r } = \lambda _ { c } .$ . The hyperparameters of the experiment are $M = N = 5$ $d \bar { = } 2 0 , \gamma = 1 , \ell = 1 0 0$ and $\lambda _ { t } = 5 \cdot 1 0 ^ { - 2 } / ( 1 + \operatorname* { m a x } ( t - 2 \cdot 1 0 ^ { 4 } , 0 ) \cdot 1 0 ^ { - 4 } )$

Figure 3 presents numerical results of the optimization obtained for initializations on the manifold using stochastic PGD. We perfectly recover the oracle parameters, up to numerical error, and there is no departure from the manifold, again, up to numerical error and noise, as was predicted by theory.

We emphasize the fact that the schedule for the inverse temperatures is crucial for the dynamics. To understand this, in Figure 4 we run an experiment in which we train 4 predictors using different schedulers: 20k warm-up steps with $\lambda _ { c } = \bar { \lambda } _ { r } = 0 . 1$ , followed by different values of constant inverse temperatures $\lambda = \lambda _ { c } = \lambda _ { r }$ . The leftmost plots show two different behaviors for ν and κ: ν converges fast, usually before the warm up period ends, while the convergence of κ is highly dependent on the value of λ. The heat maps present the respective partial derivatives of $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for the different values of λ. They show that for λ small enough, minus the gradient always points towards the global minima when far from the origin. For large values of λ, the dynamics push away from the origin much faster, but the gradients vanish and have the wrong sign close to the boundaries $\kappa = \pm 1$ . As a consequence, we see that for those values, κ does not converge to one, while it does for $\lambda = 0 . 0 1$ The warm start is needed, as for small λ, the dynamics would remain stuck around the axis $\kappa = 0$ where the values of the partial derivatives are extremely small, as visible on the plot.

![](images/24690385bd168d521b3013f2ad0fa5d8a133e0d568371ef2ec8bd83dfb82397d.jpg)

![](images/a1055749266da492caf6330aa3641883f8be831779f96ec8a76e1d8aa29ce114.jpg)  
(a) Alignment κ and the partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda , \lambda } ^ { < }$

![](images/736857dedee6842219ccd992eafd4d784b216e1c8bd1865c71ede202d6eceade.jpg)

![](images/3fbd7116c1df6aa490c3f9fb7556875fb8e8ec1b9cec56bfa09bc5484fd83f8c.jpg)  
(b) Alignment ν and the partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda , \lambda } ^ { < }$

Figure 4: Leftmost: Alignment dynamics during the training of our predictor with an inverse temperature warm-up period of 20k steps using $\lambda _ { c } = \lambda _ { r } = 0 . 1$ , followed by constant $\lambda _ { c } = \lambda _ { r } = \lambda$ for $\bar { \lambda } \in \{ 0 . 0 1 , 0 . 0 3 , \bar { 0 } . \bar { 0 } 5 , 0 . 0 7 \}$ . We plot the median across 20 runs for each value of λ. The dashed vertical line represents the warm up period. Center-left to right: The numerically computed gradients of $\mathcal { R } _ { \lambda _ { r } , \lambda _ { r } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { r } = \lambda _ { c } = \bar { \lambda \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 , 0 . 1 \} }$ $N = 5$ and $\ell = 1 0 0$ . The gradient, given by (32), is approximated using the Monte Carlo method with a sample size of 10,000.

One last and very important problem remains: in practice we cannot enforce the initialization to be on the manifold M, since that would require knowing the oracle parameters $k ^ { * }$ and $v ^ { * }$ . We can ensure that $k ^ { T } v = 0$ at initialization and penalize their inner product throughout the training. Furthermore, a noisy proxy for $v ^ { T } k ^ { * }$ is $\textstyle { \frac { 1 } { \ell } } v ^ { T } \sum _ { i , j } X _ { i j }$ . Therefore, in the case of initialization outside the manifold, we impose $k ^ { T } v = 0$ at initialization and optimize for the penalized risk

$$
\begin{array} { r } { \bar { \mathcal { R } } _ { \lambda _ { r } , \lambda _ { c } } ( k , v ) = \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k , v ) + \mu _ { 1 } ( k ^ { T } v ) ^ { 2 } + \mu _ { 2 } \mathbb { E } \big [ \big ( v ^ { T } \sum _ { i , j } X _ { i j } \big ) ^ { 2 } \big ] , } \end{array}\tag{9}
$$

where $\mu _ { 1 } , \mu _ { 2 } \in ( 0 , \infty )$ are fixed penalization coefficients. A similar use of regularization to encourage the orthogonality conditions defining an invariant manifold is explored by Maulen Soto et al. [2025] for attention-based clustering, using a different penalty to handle initializations outside the manifold.

Indeed, Figure 5 presents the results obtained when optimizing for this penalized empirical risk from a generic random initialization on $( \mathbb { S } ^ { d - 1 } ) ^ { 2 }$ . The top row (Figure 5a) displays the training dynamics for our trainable predictor (4), employing the erf non-linearity. We observe that the penalty terms effectively guide the parameters towards the invariant manifold M throughout the training and the convergence is slower, even though we have used a higher initial inverse temperature. The lower row (Figure 5b) presents the identical experimental setup applied to the predictor (3) with $\sigma = \operatorname { s o f t m a x }$ Although the training dynamics are not identical, both models converge on the manifold M and there are phase transitions in the empirical loss curves for both models. In the case of the erf model, the phase transition appears to be correlated with ν reaching near-perfect alignment, while in the case of the softmax model, the phase transition presents itself when both ν and κ reach near-perfect alignment.

As we have shown in the present experiments, the inverse temperature schedule is of the utmost importance to the performance of the training procedure, and finding a suitable one that performs well across all initializations adds a layer of complexity to the optimization process. A more in-depth analysis of and further details related to the numerical experiments can be found in Appendix E.

![](images/8fda3f416b80f5f69130bc04d5b39a13bf4d1f0d65b10b030d7d537106a047bd.jpg)

![](images/f28f900f5931cc6b672ee939f4cb8440d567c0e5c4916fd852f9cdd30468f39d.jpg)

![](images/339f863eb259d3fdec6c39abb0125326f9939011e097f29fbde91726b229fd37.jpg)

![](images/1295ebf7c0fc30ec9433a930553d40bbb8251d310867067c880da180b699c39c.jpg)  
(a) Training dynamics of the trainable predictor (4) (i.e. (3) with $\sigma = \operatorname { e r f } )$

![](images/5235f1c6fabdea91ac6180b8bef3308c8fb4c1bba46b2905fabe8152502661ba.jpg)

![](images/eb3e1b45eee76dbd7d633bc3a762efca041d23e7d3cf6ebf7badd1cf87ca8150.jpg)

![](images/9fc3d623bb589dbdde71fed0c2055c7eb35dd5ee18d1dbe16160c02cbaf61644.jpg)

![](images/4ce3961351dd56186fa82dadc7785bf7a0eb9d7a5827a25b517484cfd8be1241.jpg)  
(b) Training dynamics of the softmax predictor (3) with σ = softmax.

Figure 5: Convergence of stochastic PGD over 20 runs with random initializations on $\mathbb { S } ^ { d - 1 } \times \mathbb { S } ^ { d - 1 }$ optimizing for the penalized empirical risk (9). The solid curve represents the empirical median across the runs. The hyperparameters of the experiment are $M = \bar { N } = 5 , d = 2 0 , \bar { \gamma } = 1 , \ell = 1 0 0$ $\mu _ { 1 } = 1 0 , \mu _ { 2 } = 1$ . The inverse temperature schedules are $\lambda _ { t } = 1 0 ^ { - 1 } / ( 1 + \operatorname* { m a x } ( t - 5 \cdot 1 0 ^ { 3 } , 0 ) \cdot 1 0 ^ { - 4 } )$ for the erf predictor and $\lambda _ { t } = 1 0 ^ { - 1 } / ( 1 + \operatorname* { m a x } ( t - 2 \cdot 1 0 ^ { 4 } , 0 ) \cdot 1 0 ^ { - 4 } )$ for the softmax predictor. For computing the oracle and model losses we use the initial inverse temperatures.

## 5 Conclusion and discussion

We presented a rigorous theoretical framework for understanding axial attention in TSFMs by extending the SLR model to 2D grids. We proved that an analytically tractable simplification of axial attention achieves asymptotic Bayes optimality for the oracle parameters, whereas semi-linear models in which the inner non-linearity is removed strictly fail to do so. Furthermore, we established that Projected Gradient Descent converges to these optimal parameters for small enough inverse temperatures. Extensive numerical experiments validate our theoretical findings and highlight the critical role of inverse temperature scheduling in converging to the global minima within the nonconvex optimization landscape.

Our statistical framework assumes fixed temporal and spatial dimensions, independent background tokens, and a target that depends on a single localized token. Relaxing these assumptions provides several natural directions for future work. In particular, incorporating autoregressive temporal dynamics and dense cross-channel correlations would better reflect the structure of real multivariate time series, while allowing the dimensions or number of informative tokens to grow would broaden the range of sparse information settings covered by the model. These extensions would introduce additional dependencies into both the statistical model and the optimization landscape and would therefore require analytical tools beyond those used in the present work.

It is also necessary to explicitly reiterate the architectural constraints that make the analysis possible. Our results are heavily dependent on a rank-1 constraint applied to the query and key projection matrices, as well as the substitution of the standard softmax activation with the erf function and the shared-key constraint $K _ { r } = K _ { c }$ The theoretical results in this paper apply to this simplified erf predictor, while our experiments with the corresponding softmax predictor provide encouraging evidence that similar optimization behavior persists beyond the activation function used in the analysis. Full TSFMs additionally contain realistic projection matrices, residual connections, normalization layers, and multiple axial-attention blocks; the implications of our results for these architectures are therefore qualitative. Nevertheless, our Single Location Regression model preserves the core structural challenge of identifying and exploiting a sparsely located informative token, making it highly relevant for the theoretical study of axial attention in TSFMs.

Finally, while our current focus is restricted to a single informative token, the same framework suggests a natural extension to multiple-token regression through multi-head attention. If the target depends on several informative locations, different heads could learn distinct localization directions and specialize in selecting different tokens, while their outputs could be combined to recover the target. Under suitable identifiability and separation assumptions, parts of the statistical analysis could then be applied head-wise, following an approach similar to the one developed here. The optimization problem would, however, become substantially richer: the heads would be trained jointly, their roles would be interchangeable, and several heads could compete for or collapse onto the same informative location. We leave this extension, together with the study of the corresponding statistical and optimization guarantees, for future work.

AI use statement. We have used generative AI tools to assist with writing and cleaning code, and with editing the manuscript. The authors reviewed all AI-assisted outputs and take responsibility for the final content of this work.

Acknowledgments. We thank Pierre Marion, whose insightful comments helped us improve the final version of this paper. The work of Andrei Tiberiu Pantea and Aymeric Dieuleveut is supported by French State aid managed by the Agence Nationale de la Recherche (ANR) under the France 2030 program ANR-23-IACL-0005 (Hi! PARIS).

## References

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821, 2025.

Gaspard Berthelier, Mariia Baranova, Andrei-Tiberiu Pantea, Etienne Le Naour, Adrien Petralia, Tahar Nabil, and Themis Palpanas. Investigating simple target-covariate relationships for Chronos-2 and TabPFN-TS. In 1st ICLR Workshop on Time Series in the Age of Large Models, 2026.

Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Ramé, Qiqi Ren, Afshin Rostamizadeh, Jean Ogier du Terrail, Anna-Monica Toon, Kan Wang, Stephan Xie, Zongzhe Xu, Viktoriya Zhukova, David Asker, Ameet Talwalkar, and Othmane Abou-Amal. This time is different: An observability perspective on time series foundation models, 2025.

Yukun Cui, Jiangshe Zhang, Haowen Bai, Zixiang Zhao, Lilun Deng, Shuang Xu, and Chunxia Zhang. STAIT: A spatio-temporal alternating iterative transformer for multi-temporal remote sensing image cloud removal. Remote Sensing, 18(4), 2026. ISSN 2072-4292. doi: 10.3390/rs18040596.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10148–10167. PMLR, 21–27 Jul 2024.

Alexandru Dimofte, Glenn Anta Bucagu, Thorir Mar Ingolfsson, Xiaying Wang, Andrea Cossettini, Luca Benini, and Yawei Li. Cerebro: Compact encoder for representations of brain oscillations using efficient alternating attention. arXiv preprint arXiv:2501.10885, 2025.

O Duranthon, Pierre Marion, Claire Boyer, Bruno Loureiro, and Lenka Zdeborová. Statistical advantage of softmax attention: Insights from single-location regression. In The Fourteenth International Conference on Learning Representations, 2026.

Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. Axial attention in multidimensional transformers. arXiv preprint arXiv:1912.12180, 2019.

Shi Bin Hoo, Samuel Müller, David Salinas, and Frank Hutter. From tables to time: Extending TabPFN-v2 to time series forecasting. arXiv preprint arXiv:2501.02945, 2025.

Chenghao Liu, Taha Aksu, Juncheng Liu, Xu Liu, Hanshu Yan, Quang Pham, Silvio Savarese, Doyen Sahoo, Caiming Xiong, and Junnan Li. Moirai 2.0: When less is more for time series forecasting. arXiv preprint arXiv:2511.11698, 2025.

Pierre Marion, Raphaël Berthier, Gérard Biau, and Claire Boyer. Attention layers provably solve single-location regression. In The Thirteenth International Conference on Learning Representations, 2025.

Rodrigo Maulen Soto, Pierre Marion, and Claire Boyer. Attention-based clustering. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 66455–66506. Curran Associates, Inc., 2025. doi: 10.52202/085713-2231.

Michael Shub. Global stability ofdynamical systems. Springer Science & Business Media, 2013.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Yunshi Wen, Wesley M Gifford, Chandra Reddy, Lam M Nguyen, Jayant Kalagnanam, and Anak Agung Julius. Revisiting the generic transformer: Deconstructing a strong baseline for time series foundation models. arXiv preprint arXiv:2602.06909, 2026.

A. Yu, D.C. Maddix, B. Han, X. Zhang, A.F. Ansari, O. Shchur, C. Faloutsos, A.G. Wilson, M.W. Mahoney, and Y. Wang. Understanding transformers for time series: Rank structure, flow-of-ranks, and compressibility. In International Conference on Learning Representations, 2026.

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The Eleventh International Conference on Learning Representations, 2023.

## A Proof of Lemma 1 and further discussion about architectural constraints

We recall the definition of the standard axial self-attention layer.

$$
T _ { \lambda _ { r } , \lambda _ { c } , \sigma } ^ { ( Q _ { r , c } , K _ { r , c } , V _ { r , c } , O _ { c } ) } ( \mathbb { X } ) _ { : j } = \sigma \left( \lambda _ { c } \underbrace { \mathbb { X } _ { \cdot j } ^ { ( 1 / 2 ) } Q _ { c } } _ { M \times p _ { c } } \underbrace { K _ { c } ^ { T } \left( \mathbb { X } _ { \cdot j } ^ { ( 1 / 2 ) } \right) ^ { T } } _ { p _ { c } \times M } \right) \underbrace { \mathbb { X } _ { : j } ^ { ( 1 / 2 ) } V _ { c } } _ { M \times p _ { c } } \underbrace { Q _ { c } } _ { p _ { c } \times o _ { c } } ,
$$

where the initial row-wise computation is:

$$
\mathbb { X } _ { i \cdot } ^ { ( 1 / 2 ) } = \sigma ( \lambda _ { r } \underbrace { \mathbb { X } _ { i \cdot } Q _ { r } } _ { N \times p _ { r } } \underbrace { K _ { r } ^ { T } \mathbb { X } _ { i \cdot } ^ { T } } _ { p _ { r } \times N } ) \underbrace { \mathbb { X } _ { i \cdot } V _ { r } } _ { N \times d }
$$

In order to justify our simplified estimator (3), we explicitly map it to the mechanism of an actual axial attention block. As mentioned in the main paper, sequences are typically appended with designated [REG] tokens to act as either readout tokens or attention sinks in the context of TSFMs. Following this convention, we assume that the input’s final column consists of specialized [REG\_ROW] tokens. To collapse the 2D grid into a single regression scalar, we then incorporate an additional [REG\_COL] token at the specific index $( M + \mathbf { \bar { 1 } } , N \mathbf { \bar { + } } 1 )$ , which acts as a global aggregator during the subsequent column-wise operation. We further assume the two [REG] tokens do not attend to themselves, which is a very light assumption, since a priori these tokens contain no information. Isolating the output for this specific token yields the following

$$
\mathbb { X } _ { i , N + 1 } ^ { ( 1 / 2 ) } = \sum _ { l = 1 } ^ { N } \sigma ( \lambda _ { r } X _ { [ \mathrm { R E G \_ R O W } ] } ^ { T } Q _ { r } K _ { r } ^ { T } \mathbb { X } _ { i \cdot } ^ { T } ) _ { l } X _ { i l } ^ { T } V _ { r } = \sigma ( \lambda _ { r } a K _ { r } ^ { T } \mathbb { X } _ { i \cdot } ^ { T } ) \mathbb { X } _ { i \cdot } \in \mathbb { R } ^ { d }
$$

if we assume that the projection $V _ { r } \in \mathbb { R } ^ { d \times d }$ is the identity matrix and encapsulate the query interaction as a vector $a : = X _ { [ \mathrm { R E G \_ R O W } ] } ^ { T } Q _ { r } \in \mathbb { R } ^ { 1 \times p _ { r } }$ . By applying an analogous definition for the column-wise query $a ^ { ( 1 / 2 ) } : = X _ { [ \mathrm { R E G \_ C O L } ] } ^ { \dot { T } } \dot { Q } _ { c } \in \mathbb { R } ^ { 1 \times p _ { c } }$ , the output of the final attention block becomes:

$$
T _ { \lambda _ { r } , \lambda _ { c } , \sigma } ^ { ( Q _ { r , c } , K _ { r , c } , V _ { r , c } , O _ { c } ) } ( \mathbb { X } ) _ { M + 1 , N + 1 } = \sum _ { i = 1 } ^ { M } \sigma \left( \lambda _ { c } a ^ { ( 1 / 2 ) } K _ { c } ^ { T } \mathbb { X } _ { i , N + 1 } ^ { ( 1 / 2 ) } \right) _ { i } \left( \mathbb { X } _ { i , N + 1 } ^ { ( 1 / 2 ) } \right) ^ { T } V _ { c } O _ { c }
$$

To perfectly align this with our theoretical model, we enforce rank-1 restrictions on the projections, setting $p _ { r } = p _ { c } = 1$ . With these constraints in place, along with $o _ { c } = 1 , O _ { c } = 1$ , and shared keys $K _ { r } = K _ { c }$ , the full axial mechanism cleanly reduces to our formulated predictor:

$$
T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( Q _ { r , c } , K _ { r , c } , V _ { r , c } , O _ { c } ) } ( \mathbb { X } ) _ { M + 1 , N + 1 } = T _ { \lambda _ { r } a , \lambda _ { c } a ^ { ( 1 / 2 ) } , \sigma } ^ { ( K _ { c } , V _ { c } ) } ( \mathbb { X } )
$$

Discussion. In this paragraph, we address the main architectural constraints and modifications imposed in our analysis.

• REG tokens. The addition of REG tokens is not a relevant modification, since any transformer layer whose output must be a single value, be it scalar of vector, will implement a global aggregation mechanism for the input tokens and will be trained for it. REG tokens remove the inductive biases that come with pooling methods or positional decoding and allow for a learned aggregation mechanism.

• Rank-1 restriction on projection matrices. This constraint is not at all harsh in the context of SLR, where there exists a single informative direction in the latent space, namely $k ^ { * }$ Moreover, Yu et al. [2026] has shown that patch-based TSFMs admit low-rank approximations of the projection matrices.

• Shared keys. $K _ { r } = K _ { c }$ is indeed a strong constraint in practice. However, in the particular case of the studied SLR, there is a single special direction in the embedding space, $k ^ { * }$ , and, heuristically, token selection should be made according to this direction.

• Non-linearity. For the theoretical analysis, we have restricted the value of σ to erf, maintaining the same convention as in Marion et al. [2025]. However, only a couple of properties of the erf function are essential to the present proofs, such as global boundedness of erf and of its derivatives, the asymptotics l $\begin{array} { r } { \operatorname* { i m } _ { x  \infty } \operatorname { e r f } ( x ) = 1 } \end{array}$ and li $\operatorname { 1 } _ { x \to - \infty } \operatorname { e r f } ( x ) = - 1 , \operatorname { e r f } ( x ) = 0$ etc. Extending the proofs and the numerical analysis to other activation functions seems feasible and would be an interesting further research direction. Furthermore, the statistical role of the activation function in single-location regression is investigated by Duranthon et al. [2026], who establish advantages of softmax over alternative activations in a single-axis setting. Understanding how these comparisons extend to sequential axial attention is an interesting direction for future work.

## B Proofs for Section 3

In what follows, we shall denote by

$$
\mathcal { R } ^ { < } ( \kappa , \nu ) : = \mathcal { R } ^ { < } ( \kappa , \nu , 0 , 0 , 0 ) ,
$$

the value of the risk when restricted to M.

## B.1 Proofs of Lemma 2 and of Proposition 3

We recall the formula for the risk

$$
\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ( k , v ) = \mathbb { E } \left[ \left( Y - \sum _ { i = 1 } ^ { M } \mathrm { e r f } \left( \lambda _ { c } \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } X _ { i j } ^ { T } k ) X _ { i j } ^ { T } k \right) \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } X _ { i j } ^ { T } k ) X _ { i j } ^ { T } v \right) ^ { 2 } \right]\tag{10}
$$

Let us now denote by $G _ { i j } ^ { u } = X _ { i j } ^ { T } u$ , for $u \in \mathbb { R } ^ { d }$ . Note that we consider $X _ { I _ { 0 } J _ { 0 } } = \ell k ^ { * } + \gamma \tilde { X } _ { I _ { 0 } J _ { 0 } }$ and $G _ { I _ { 0 } J _ { 0 } } ^ { u } = \tilde { X } _ { I _ { 0 } J _ { 0 } } ^ { T } u$ . Then, our predictor only depends on $\kappa , \theta$ and the alignment random variables $G _ { i j } ^ { v } , G _ { i j } ^ { v ^ { * } } , G _ { i j } ^ { k }$ which are i.i.d. and follow the distribution

$$
\binom { G _ { i j } ^ { k } } { G _ { i j } ^ { v } } \sim \mathcal { N } \left( 0 , \left( \begin{array} { l l l } { 1 } & { \rho } & { \eta } \\ { \rho } & { 1 } & { \nu } \\ { \eta } & { \nu } & { 1 } \end{array} \right) \right) .
$$

We further assume, without loss of generality, that $I _ { 0 } = J _ { 0 } = 1$ . Then $Y = \gamma G _ { 1 1 } ^ { v ^ { * } } + \xi$ and, dropping the subscripts of $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } }$ , one indeed has

$$
\begin{array} { r l } & { { \mathcal { R } } ( k , v ) = { \mathbb { E } } \left[ \left( \gamma G _ { 1 1 } ^ { v ^ { * } } + \xi - T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( { \mathbb X } ) \right) ^ { 2 } \right] } \\ & { \quad \quad = \epsilon ^ { 2 } + { \mathbb { E } } \left[ \left( \gamma G _ { 1 1 } ^ { v ^ { * } } - T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( { \mathbb X } ) \right) ^ { 2 } \right] } \\ & { \quad \quad = \epsilon ^ { 2 } + \gamma ^ { 2 } - 2 \gamma { \mathbb { E } } \left[ G _ { 1 1 } ^ { v ^ { * } } T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( { \mathbb X } ) \right] + { \mathbb { E } } \left[ T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( { \mathbb X } ) ^ { 2 } \right] . } \end{array}\tag{11}
$$

where

$$
\begin{array} { l } { { \displaystyle T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) = \sum _ { i = 2 } ^ { M } \mathrm { e r f } \left( \lambda _ { c } \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { k } \right) \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { v } } } \\ { { \displaystyle \qquad + \mathrm { e r f } \left( \lambda _ { c } \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \kappa \ell + \gamma G _ { 1 1 } ^ { k } \right) + \lambda _ { c } \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { k } \right) . } } \\ { { \displaystyle \qquad \cdot \left( \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \theta \ell + \gamma G _ { 1 1 } ^ { v } \right) + \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { v } \right) } } \end{array}\tag{12}
$$

Since Equation (11) and Equation (12) only depend on the 5 parameters $\kappa , \nu , \rho , \eta , \theta$ , this ends the proof of Lemma 2. On the manifold M, where $\rho = \theta = \eta = 0$ , we can simplify the expression by dropping the θℓ term in the last row of Equation (12). We continue our proof by analyzing each term in Equation (11) individually.

The first term can be rewritten as

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ G _ { 1 1 } ^ { \nu ^ { * } } T _ { \lambda _ { r } , \lambda _ { \lambda } } ^ { ( k , \nu ) } ( \mathbb { X } ) \right] = } \\ { \displaystyle = \sum _ { i = 2 } ^ { M } \mathbb { E } \left[ \mathrm { e r f } \left( \lambda _ { c } \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { 1 j } ^ { k } \right) \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { \nu } G _ { 1 1 } ^ { \nu ^ { * } } \right] } \\ { \displaystyle ~ + \mathbb { E } \left[ \mathrm { e r f } \left( \lambda _ { c } \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \kappa \ell + \gamma G _ { 1 1 } ^ { k } \right) + \lambda _ { c } \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { k } \right) . } \\ { \displaystyle ~ \cdot \left( \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \theta \ell + \gamma G _ { 1 1 } ^ { \nu } \right) G _ { 1 1 } ^ { \nu ^ { * } } + \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { \nu } G _ { 1 1 } ^ { \nu ^ { * } } \right) \right] } \end{array}\tag{13}
$$

On the manifold M, this becomes:

$$
\begin{array} { r l } & { \mathbb { E } [ G _ { 1 1 } ^ { v ^ { * } } T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) ] = } \\ & { \quad = \displaystyle \sum _ { i = 2 } ^ { M } \mathbb { E } [ \mathrm { e r f } ( \lambda _ { c } \displaystyle \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { k } ) \displaystyle \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { v } G _ { 1 1 } ^ { v ^ { * } } ] } \\ & { \quad + \mathbb { E } [ \mathrm { e r f } ( \lambda _ { c } \mathrm { e r f } ( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } ) ( \kappa \ell + \gamma G _ { 1 1 } ^ { k } ) + \lambda _ { c } \displaystyle \sum _ { j = 2 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { 1 j } ^ { k } ) G _ { 1 j } ^ { k } ) . } \end{array}
$$

$$
\begin{array} { r l } & { \quad ( \mathrm { c o l } ( \mathcal { A } ( \mathcal { A } ( \mathcal { A } ( \mathcal { A } ) , \mathcal { C } ) ) \otimes \mathcal { C } ) ) \times \frac { \mathcal { C } } { \mathcal { C } } ( \mathcal { A } ( \mathcal { A } ( \mathcal { A } ) , \mathcal { C } ) \otimes \mathcal { C } ) \frac { \mathcal { C } } { \mathcal { C } } ( \mathcal { A } ( \mathcal { A } ( \mathcal { A } ) , \mathcal { C } ) \otimes \mathcal { C } ) \frac { \mathcal { C } } { \mathcal { C } } ) \Bigg | } \\ & { = \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \Bigg | \Bigg ( \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \mathcal { C } \mathcal { C } \mathcal { C } \Bigg ) \Bigg | \frac { \mathcal { C } } { \mathcal { C } } \Bigg | \Bigg ( \mathcal { C } - \frac { \mathcal { C } } { \mathcal { C } } \frac { \mathcal { C } } { \mathcal { C } } \mathcal { C } \mathcal { C } \Bigg ) \Bigg | \frac { \mathcal { C } } { \mathcal { C } } \Bigg | \Bigg ( \mathcal { C } - \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) } \\ &  \quad + \mathcal { C } \Bigg [ \Bigg ( \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) \Bigg ( \mathcal { C } + \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) \Bigg ( \mathcal { C } + \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) \Bigg ) \Bigg | \tilde { \mathcal { C } } \Bigg | + \mathcal { C } ^ { 2 } \Bigg ( \mathcal { C } + \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) \Bigg | \frac { \mathcal { C } } { \mathcal { C } } \Bigg | \Bigg ( \mathcal { C } + \frac { \mathcal { C } } { \mathcal { C } } \Bigg ) \Bigg | \frac  \mathcal { C }  \end{array}\tag{14}
$$

where we have used the independence between $G _ { 1 1 } ^ { v ^ { * } }$ and $G _ { i j } ^ { v } , G _ { i j } ^ { k }$

Due to the independence between $G _ { i j } ^ { \cdot }$ and $G _ { r s } ^ { \ast }$ for $( i , j ) \neq ( r , s )$ , there appear no cross-terms when expanding the expectation of $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) ^ { 2 }$ . Therefore, under $\rho = 0$ , the second term simplifies as

$$
\begin{array} { l } { { \displaystyle { \mathbb E } [ T _ { s , \nu , \nu } ^ { ( k , \nu ) } ( { \mathbb X } ^ { 2 } ) ] ^ { 2 } } \ ~ } \\ { { \displaystyle = \ \sum _ { i = 2 , j = 1 } ^ { N } { \sum _ { k = 1 } ^ { N } \{ \mathrm { E r f } ^ { 2 } ( \lambda _ { c , \sum ^ { k = 1 } } \mathrm { e r f } ( \lambda _ { \nu } C _ { i i } ^ { k } ) C _ { i i j } ^ { k } )  ^ { \alpha _ { 0 } } \} } \cdot \ \mathrm { e r f } ^ { 2 } ( \lambda _ { \nu } C _ { i j } ^ { k } ) ( C _ { i j } ^ { \nu } ) ^ { 2 } \Bigg ] } } \\ { { \displaystyle ~ + \ { \mathbb E } [ \mathrm { e r f } ^ { 2 } ( \lambda _ { c , \mathrm { e r f } } ( \lambda _ { c , \mathbb { Y } } C _ { 1 } ^ { k } ) ( { \mathbb X } ^ { 2 } + \lambda _ { \nu } ) C _ { 1 } ^ { k } ) ( \kappa \ell + \gamma G _ { 1 } ^ { k } ) + \lambda _ { \nu } \sum _ { j = 2 } ^ { N } \mathrm { e r f } ( \lambda _ { \nu } G _ { 1 } ^ { k } ) G _ { 1 j } ^ { k } ) ^ { 2 } } \ ~ } \\ { { \displaystyle ~ + \ ( \mathrm { e r f } ^ { 2 } ( \lambda _ { c } \kappa \ell + \lambda _ { c } \gamma G _ { 1 } ^ { k } ) ( \ell \ell + \gamma G _ { 1 } ^ { \nu } ) ^ { 2 } ) ^ { 2 } + \sum _ { j = 2 } ^ { N } \mathrm { e r f } ^ { 2 } ( \lambda _ { c } G _ { 1 j } ^ { k } ) ( G _ { 1 j } ^ { \nu } ) ^ { 2 } } \ ~ } \\   \displaystyle ~ + 2 \theta \ell \ \mathrm { e r f } ( \lambda _ { c } \kappa \ell + \lambda _ { c } \gamma G _ { 1 1 } ^ { k } ) \sum _ { j = 2 } ^ { N } \mathrm { e r f } ( \lambda _ { c } G _ { 1 j } ^ { k } ) \end{array}\tag{15}
$$

On the manifold M $( \rho = \theta = \eta = 0 )$ , this becomes

$$
\begin{array} { r l } & { \mathbb { E } \Big [ T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) ^ { 2 } \Big ] } \\ & { = \displaystyle \sum _ { i = 2 } ^ { M } \sum _ { j = 1 } ^ { N } \mathbb { E } \left[ \mathrm { e r f } ^ { 2 } \left( \lambda _ { c } \sum _ { j = 1 } ^ { N } \mathrm { e r f } \big ( \lambda _ { r } G _ { i j } ^ { k } \big ) G _ { i j } ^ { k } \right) \cdot \mathrm { e r f } ^ { 2 } \big ( \lambda _ { r } G _ { i j } ^ { k } \big ) \right] } \end{array}
$$

$$
\begin{array} { r l } { \displaystyle } & { + \mathbb { E } \left[ \mathrm { e r f } ^ { 2 } \left( \lambda _ { r } \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \kappa \ell + \gamma G _ { 1 1 } ^ { k } \right) + \lambda _ { c } \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { k } \right) \right. } \\ & { \left. \vphantom { \Bigg ( } \mathrm { \lambda _ { r } } . \left( \gamma ^ { 2 } \mathrm { e r f } ^ { 2 } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) + \sum _ { j = 2 } ^ { N } \mathrm { e r f } ^ { 2 } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) \right) \right] } \end{array}
$$

In what follows, we shall prove that under the joint scaling $\lambda _ { r } \ell \to \infty , \lambda _ { c } \ell \to \infty$ and $\lambda _ { r } ~ =$ $o \left( M ^ { - \frac { 1 } { 2 } } N ^ { - \frac { 1 } { 2 } } \right)$ , the risk converges to $\epsilon ^ { 2 } .$ For the oracle parameters, $\kappa = \nu = 1$ . Moreover, erf $\dot { \left( \lambda _ { r } x \right) } x \geq 0$ for all $x \in \mathbb { R }$ and $| \mathrm { e r f } ( x ) | \leq 1$ . Therefore, Equation (14) implies

$$
\begin{array} { r l } & { \mathbb { E } \left[ G _ { 1 1 } ^ { v ^ { * } } T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) } ( \mathbb { X } ) \right] \geq \gamma \mathbb { E } \left[ \mathrm { e r f } \left( \lambda _ { c } \mathrm { e r f } \left( \lambda _ { r } \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \ell + \gamma G _ { 1 1 } ^ { k } \right) \right) \mathrm { e r f } \left( \lambda _ { r } \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \right] } \\ & { \phantom { = \gamma \mathbb { P } \left( \gamma G _ { 1 1 } ^ { k } \leq \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) } - \gamma \mathbb { P } \left( \gamma G _ { 1 1 } ^ { k } \leq - \ell \right) } \\ & { = \gamma \mathbb { E } \left[ \mathrm { e r f } \left( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } ( G ) G \right) \mathrm { e r f } ( G ) \right] - \gamma \Phi \left( \frac { - \ell } { \gamma } \right) , } \end{array}
$$

where $G \sim \mathcal N \left( \lambda _ { r } \ell , \lambda _ { r } ^ { 2 } \gamma ^ { 2 } \right)$ and Φ is the cumulative distribution function of $\mathcal { N } ( 0 , 1 )$

Next, since $| \mathrm { e r f } ( x ) | \leq 1$ and $\operatorname { s r f } ^ { 2 } ( x ) \leq { \frac { 4 } { \pi } } x ^ { 2 }$ from Lemma 12 for all $x \in \mathbb { R }$ , one has

$$
\begin{array} { r } { \displaystyle \sum _ { i = 2 } ^ { M } \displaystyle \sum _ { j = 1 } ^ { N } \mathbb { E } [ \mathrm { e r f } ^ { 2 } ( \lambda _ { c } \displaystyle \sum _ { j = 1 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { i j } ^ { k } ) G _ { i j } ^ { k } ) \cdot \mathrm { e r f } ^ { 2 } ( \lambda _ { r } G _ { i j } ^ { k } ) ] \leq \displaystyle \sum _ { i = 2 } ^ { M } \sum _ { j = 1 } ^ { N } \mathbb { E } [ \mathrm { e r f } ^ { 2 } ( \lambda _ { r } G _ { i j } ^ { k } ) ] } \\ { \leq \displaystyle \sum _ { i = 2 } ^ { M } \displaystyle \sum _ { j = 1 } ^ { N } \frac { 4 \lambda _ { r } ^ { 2 } } { \pi } \mathbb { E } [ ( G _ { i j } ^ { k } ) ^ { 2 } ] } \\ { = \displaystyle \frac { 4 } { \pi } \lambda _ { r } ^ { 2 } ( M - 1 ) N  0 , } \end{array}
$$

due to the scaling $\lambda _ { r } = o \left( M ^ { - \frac { 1 } { 2 } } N ^ { - \frac { 1 } { 2 } } \right)$

Finally, substituting $\kappa = 1$ and using $| \mathrm { e r f } ( x ) | \le 1$ and $\operatorname { e r f } ^ { 2 } ( x ) \leq { \frac { 4 } { \pi } } x ^ { 2 }$ again from Lemma 12, one has that

$$
\begin{array} { r l } & { \mathbb { E } \Bigg [ \mathrm { e r f } ^ { 2 t } \Bigg ( \lambda _ { r } \mathrm { e f f } ( \lambda _ { r } \mathrm { e f f } ( \lambda _ { 1 } ) \mathrm { e } ^ { \lambda } + \lambda _ { r } \mathrm { o f } \mathcal { C } _ { 1 1 } ^ { k } ) \Bigg ( \Lambda _ { t } \mathrm { e } ^ { \lambda } + \gamma \mathcal { Q } _ { 1 1 } ^ { k } \Bigg ) + \lambda _ { r } \sum _ { k = 0 } ^ { \infty } \mathrm { e r f } ( \lambda _ { r } \mathcal { Q } _ { 1 2 } ^ { k } ) \mathcal { Q } _ { 1 2 } ^ { k } \Bigg ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \int _ { - 2 } ^ { \infty } \mathrm { e r f } ^ { 2 } \Bigg ( \lambda _ { r } \mathrm { e f f } ^ { 2 } ( \lambda _ { r } \mathrm { e f f } ^ { k } + \lambda _ { r } \mathrm { o f } \mathcal { Q } _ { 1 1 } ^ { k } ) + \frac { \lambda _ { r } ^ { 2 } } { \lambda _ { r } ^ { 2 } - \sigma ^ { 2 } } \mathrm { e r f } ^ { 2 } ( \lambda _ { r } \mathcal { Q } _ { 1 2 } ^ { k } ) \Bigg ) \Bigg ] } \\ & { \leq \gamma ^ { 2 } \mathbb { E } \Bigg [ \mathrm { e r f } ^ { 2 } \Bigg ( \lambda _ { r } \mathrm { e f f } ( \lambda _ { r } \mathrm { e f f } ( \lambda _ { r } ) + \lambda _ { r } \mathrm { e f f } \mathcal { Q } _ { 1 1 } ^ { k } ) ( \ell + \gamma \mathcal { Q } _ { 1 1 } ^ { k } ) + \lambda \sum _ { r = 0 } ^ { \infty } \mathrm { e r f } ( \lambda _ { r } \mathcal { Q } _ { 1 2 } ^ { k } ) \mathcal { Q } _ { 1 2 } ^ { k } \Bigg ) } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { e r f } ^ { 2 t } ( \lambda _ { r } \mathrm { e f f } ( \lambda _ { r } ) + \sum \end{array}
$$

where $G \sim \mathcal N \left( \lambda _ { r } \ell , \lambda _ { r } ^ { 2 } \gamma ^ { 2 } \right)$

Aggregating everything and considering the oracle parameters $k ^ { * } , v ^ { * }$ , one now has that

$$
\mathcal { R } ( k ^ { * } , v ^ { * } ) \leq \epsilon ^ { 2 } + \gamma ^ { 2 } - 2 \gamma ^ { 2 } \mathbb { E } \left[ \operatorname { e r f } \left( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } ( G ) G \right) \mathrm { e r f } ( G ) \right] + 2 \gamma ^ { 2 } \Phi \left( \frac { - \ell } { \gamma } \right)
$$

$$
+ \gamma ^ { 2 } \mathbb { E } [ \mathrm { e r f } ^ { 2 } ( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } ( G ) G + \lambda _ { c } \sum _ { j = 2 } ^ { N } \mathrm { e r f } ( \lambda _ { r } G _ { 1 j } ^ { k } ) G _ { 1 j } ^ { k } ) ] \mathrm { e r f } ^ { 2 } ( G ) ] + o ( 1 ) ,
$$

where $G \sim \mathcal N ( \lambda _ { r } \ell , \lambda _ { r } ^ { 2 } \gamma ^ { 2 } )$

To conclude the proof of Proposition 3, it is enough to prove

$$
\mathbb { E } [ \mathrm { e r f } ( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } ( G ) G ) \mathrm { e r f } ( G ) ]  1 ,
$$

as $\ell \to \infty$ implies Φ $\left( \frac { - \ell } { \gamma } \right) \to 0$ . Since $x \mapsto$ erf $\begin{array} { r } { \left( \frac { \lambda _ { c } } { \lambda _ { r } } \mathbf { e r f } ( x ) x \right) \mathbf { e r f } ( x ) } \end{array}$ is increasing on $( 0 , \infty )$ and less than 1 in absolute value, it is enough to consider

$$
\begin{array} { r l } & { \mathbb { E } \left[ \operatorname { e r f } \left( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } ( G ) G \right) \mathrm { e r f } ( G ) \right] \geq \mathrm { e r f } \left( \frac { \lambda _ { c } } { \lambda _ { r } } \mathrm { e r f } \left( \frac { \lambda _ { r } \ell } { 2 } \right) \frac { \lambda _ { r } \ell } { 2 } \right) \mathrm { e r f } \left( \frac { \lambda _ { r } \ell } { 2 } \right) \mathbb { P } \left( G \geq \frac { \lambda _ { r } \ell } { 2 } \right) } \\ & { \quad \quad - \mathbb { P } ( G < 0 ) } \\ & { \quad \quad = \mathrm { e r f } \left( \frac { \lambda _ { c } \ell } { 2 } \mathrm { e r f } \left( \frac { \lambda _ { r } \ell } { 2 } \right) \right) \mathrm { e r f } \left( \frac { \lambda _ { r } \ell } { 2 } \right) \left( 1 - \Phi \left( - \frac { \ell } { 2 \gamma } \right) \right) } \\ & { \quad \quad \quad - \Phi \left( - \frac { \ell } { \gamma } \right) . } \end{array}
$$

The conclusion now follows since $\lambda _ { r } \ell \to \infty , \ell \to \infty , \lambda _ { c } \ell \to \infty$

## B.2 Proof of Proposition 4

The proof aims to make explicit the loss of information from X to $( S _ { 1 } , S _ { 2 } , \ldots , S _ { M } )$ . To this end, one has

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( Y - f ( S _ { 1 } , S _ { 2 } , \ldots , S _ { M } ) \right) ^ { 2 } \right] \ge \mathbb { E } \left[ \mathrm { V a r } \left( Y | S _ { 1 } , S _ { 2 } , \ldots , S _ { M } \right) \right] } \\ & { \qquad \ge \mathbb { E } \left[ \mathrm { V a r } \left( Y | S _ { 1 } , S _ { 2 } , \ldots , S _ { M } , I _ { 0 } , J _ { 0 } \right) \right] . } \end{array}
$$

Now, consider $i _ { 0 } \in [ M ]$ and $j _ { 0 } \in [ N ]$ . Conditional on $( I _ { 0 } , J _ { 0 } ) = ( i _ { 0 } , j _ { 0 } ) , S _ { i }$ is independent of $( Y , S _ { i _ { 0 } } )$ , for $i \neq i _ { 0 }$ . Then, one has

$$
\mathrm { V a r } \left( Y | S _ { 1 } , S _ { 2 } , \ldots , S _ { N } , I _ { 0 } = i _ { 0 } , J _ { 0 } = j _ { 0 } \right) = \mathrm { V a r } \left( Y | S _ { i _ { 0 } } , I _ { 0 } = i _ { 0 } , J _ { 0 } = j _ { 0 } \right) .\tag{16}
$$

For multivariate Gaussian random variables $( U , V )$ , one has

$$
\mathrm { V a r } ( U | V ) = \Sigma _ { U U } - \Sigma _ { U V } \Sigma _ { V V } ^ { - 1 } \Sigma _ { V U } ,\tag{17}
$$

where $\Sigma _ { A B }$ stands for $\mathrm { C o v } ( A , B )$ , for random vectors A and B.

Conditioned on $I _ { 0 } = i _ { 0 }$ and $J _ { 0 } = j _ { 0 }$ , one has:

$\begin{array} { r } { \bullet \ { X } _ { i _ { 0 } j _ { 0 } } = \ell k ^ { * } + \gamma Z _ { j _ { 0 } } } \\ { \mathrm { v a r i a b l e s } \ { Z } _ { 1 } , \ldots , { Z } _ { N } ; } \end{array}$ and $X _ { i _ { 0 } j } = Z _ { j }$ for $j \in [ N ] \setminus \{ j _ { 0 } \}$ with standard normal random

$$
\begin{array} { r } { \bullet \ S _ { i _ { 0 } } = \beta _ { j _ { 0 } } \ell k ^ { * } + \gamma \beta _ { j _ { 0 } } Z _ { j _ { 0 } } + \sum _ { j \neq j _ { 0 } } \beta _ { j } Z _ { j } ; } \end{array}
$$

$$
\bullet \mathrm { ~ } Y = \gamma Z _ { j _ { 0 } } ^ { T } v ^ { * } + \xi .
$$

One then has the following:

$$
\begin{array} { r l } & { \bullet \operatorname { V a r } ( Y ) = \gamma ^ { 2 } + \epsilon ^ { 2 } ; } \\ & { \bullet \operatorname { C o v } ( S _ { i _ { 0 } } , Y ) = \gamma ^ { 2 } \beta _ { j _ { 0 } } v ^ { * } } \\ & { \bullet \operatorname { C o v } ( S _ { i _ { 0 } } , S _ { i _ { 0 } } ) = \left( \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \neq j _ { 0 } } \beta _ { j } ^ { 2 } \right) I } \end{array}
$$

Introducing in (17), one obtains

$$
\operatorname { V a r } \left( Y | S _ { i _ { 0 } } , I _ { 0 } = i _ { 0 } , J _ { 0 } = j _ { 0 } \right) = \gamma ^ { 2 } + \epsilon ^ { 2 } - \frac { \gamma ^ { 4 } \beta _ { j _ { 0 } } ^ { 2 } } { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \neq j _ { 0 } } \beta _ { j } ^ { 2 } } \| v ^ { * } \| ^ { 2 } = \epsilon ^ { 2 } + \gamma ^ { 2 } \left( 1 - \frac { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } } { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \neq j _ { 0 } } \beta _ { j } ^ { 2 } } \right) .\tag{18}
$$

Finally,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \operatorname { V a r } \left( Y \vert S _ { 1 } , S _ { 2 } , \dots , S _ { M } , I _ { 0 } , J _ { 0 } \right) \right] = \displaystyle \sum _ { j _ { 0 } = 1 } ^ { N } \frac { M } { i _ { 0 } = 1 } \mathbb { P } ( I _ { 0 } = i _ { 0 } \vert J _ { 0 } = j _ { 0 } ) p _ { j _ { 0 } } \operatorname { V a r } \left( Y \vert S _ { i _ { 0 } } , I _ { 0 } = i _ { 0 } , J _ { 0 } = j _ { 0 } \right) } \\ & { \quad \quad \quad = \displaystyle \sum _ { j _ { 0 } = 1 } ^ { N } p _ { j _ { 0 } } \left( \epsilon ^ { 2 } + \gamma ^ { 2 } \left( 1 - \frac { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } } { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \neq j _ { 0 } } \beta _ { j } ^ { 2 } } \right) \right) } \\ & { \quad \quad \quad = \epsilon ^ { 2 } + \gamma ^ { 2 } \left( 1 - \displaystyle \sum _ { j _ { 0 } = 1 } ^ { N } p _ { j _ { 0 } } \frac { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } } { \gamma ^ { 2 } \beta _ { j _ { 0 } } ^ { 2 } + \sum _ { j \neq j _ { 0 } } \beta _ { j } ^ { 2 } } \right) } \\ & { \quad \quad \quad \quad \geq \epsilon ^ { 2 } , } \end{array}
$$

with equality in the last step only if there exists $j _ { 0 }$ such that $p _ { j _ { 0 } } = 1$ and $\beta _ { j } = 0$ for $j \neq j _ { 0 }$

## C Proof of Theorem 5

In this section, we establish the global convergence of the Projected Gradient Descent (PGD) dynamics to the true oracle parameters. We follow the proof of Theorem 5 in Marion et al. [2025] as closely as possible while adapting it to our model $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , \bar { v } ) }$ . The only result which requires a completely different and much more involved proof is that of Proposition 9. Throughout these proofs, $C > 0$ denotes a generic constant that may vary from line to line but is strictly independent of the step size α and the iteration step t. Moreover, we shall drop the subscript $\lambda _ { r } , \lambda _ { c }$ and assume an implicit dependence on the inverse temperatures.

The theoretical guarantees of our optimization trajectory are established through the following five primary results. We first establish the invariance of the manifold to PGD steps.

Lemma 6. The manifold M is invariant under the PGD dynamics (Algorithm 1), in the sense that if $( k _ { t } , v _ { t } ) \in \mathcal { M } ,$ , then $( k _ { t + 1 } , v _ { t + 1 } ) \in \mathcal { M }$

By restricting our analysis to the manifold M, we can project the high-dimensional dynamics down to the scalar alignment variables $\kappa _ { t } = k _ { t } ^ { T } k ^ { * }$ and $\nu _ { t } = \bar { v _ { t } ^ { T } } v ^ { * }$ . We then map the PGD updates to an autonomous dynamical system:

Lemma 7. When initialized on the manifold M, the PGD iterations (Algorithm 1) can be reformulated as the autonomous discrete dynamical system $\left( \kappa _ { t + 1 } , \nu _ { t + 1 } \right) \ = \ g ( \kappa _ { t } , \nu _ { t } )$ , where $\dot { g \colon [ - 1 , 1 ] ^ { 2 } }  [ - 1 , 1 ] ^ { 2 }$ is given by

$$
g ( \kappa , \nu ) = \left( \frac { \kappa - \alpha \big ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) \big ) ( 1 - \kappa ^ { 2 } ) } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \kappa ^ { 2 } ) } } , \frac { \nu - \alpha \big ( \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa , \nu ) \big ) \big ( 1 - \nu ^ { 2 } \big ) } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \nu ^ { 2 } ) } } \right) .\tag{19}
$$

With the dynamics formalized, we establish that the risk strictly decreases, forcing convergence to the stationary points:

Proposition 8. For a sufficiently small step size α, the risk $\mathcal { R } ^ { < }$ <sup><</sup> is strictly decreasing along the PGD iterates unless it is at afixed point. Furthermore, $i f ( \kappa , \nu )$ is an accumulation point ofthe sequence $( \kappa _ { t } , \nu _ { t } ) _ { t \geq 0 } ,$ , it must satisfy:

$$
( 1 - \kappa ^ { 2 } ) \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) = 0 \quad a n d \quad ( 1 - \nu ^ { 2 } ) \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa , \nu ) = 0 .\tag{20}
$$

We then rigorously classify the geometry of these accumulation points, isolating the oracle states as the unique global minima:

Proposition 9. For $\lambda _ { c }$ and $\lambda _ { r }$ small enough, the fixed points of the dynamics, given by (20), on $[ - 1 , \bar { 1 } ] ^ { 2 }$ are $( \pm 1 , \pm 1 )$ and $( 0 , 0 )$ . Specifically:

$( \pm 1 , \mp 1 )$ are global maxima.

• (±1, ±1) are global minima.

• (0, 0) is a saddle point.

Finally, we characterize the local geometry around the saddle point to guarantee that random initializations escape it:

Proposition 10. For a sufficiently small step size α, the mapping g is a local diffeomorphism around $( 0 , 0 )$ , whose Jacobian matrix has one eigenvalue in (0, 1) and one in $( 1 , \infty )$ . Furthermore, g is injective on $[ - 1 , 1 ] ^ { 2 }$

The global convergence to the oracle states then follows directly from the exclusion of the saddle point and maxima. The proofs for all stated results are provided in the subsections below.

## C.1 Proof of Lemma 6

To prove that the manifold M is invariant under the Projected Gradient Descent (PGD) dynamics, we must first establish that the gradient of the risk does not "push" the parameters outside of M. We do this by proving an auxiliary symmetry result regarding the cross-term derivatives.

Lemma 11. At any point $( \kappa , \nu , \theta , \eta , \rho )$ on the manifold M $( i . e .$ , where $\theta = \eta = \rho = 0 )$ , the partial derivatives ofthe risk with respect to the cross-terms vanish:

$$
\partial _ { \theta } \mathcal { R } ^ { < } = \partial _ { \eta } \mathcal { R } ^ { < } = \partial _ { \rho } \mathcal { R } ^ { < } = 0 .
$$

The proof follows by recalling that the risk can be reduced to a function $\mathcal { R } ^ { < }$ depending only on the covariance structure of the Gaussian projections $G _ { i j } ^ { k } = X _ { i j } ^ { T } k , G _ { i j } ^ { v } = X _ { i j } ^ { T } v$ , and $G _ { i j } ^ { v ^ { * } } = X _ { i j } ^ { T } v ^ { * }$ . The relevant covariance matrix is:

$$
\binom { G _ { i j } ^ { k } } { G _ { i j } ^ { v } } \sim \mathcal { N } \left( 0 , \left( \begin{array} { l l l } { 1 } & { \rho } & { \eta } \\ { \rho } & { 1 } & { \nu } \\ { \eta } & { \nu } & { 1 } \end{array} \right) \right)
$$

and the alignment between v and $k ^ { * }$ is given by $\theta = v ^ { T } k ^ { * }$

It is clear from Equation 10 that the risk is invariant to the sign inversion of $( \boldsymbol { v } , \boldsymbol { v } ^ { * } )$ to $( - v , - v ^ { * } )$ This amounts to

$$
\mathcal { R } ^ { < } ( \kappa , \nu , \theta , \eta , \rho ) = \mathcal { R } ^ { < } ( \kappa , \nu , - \theta , - \eta , - \rho ) .
$$

Taking the partial derivative with respect to θ now yields the following result:

$$
\partial _ { \theta } \mathcal { R } ^ { < } ( \kappa , \nu , \theta , \eta , \rho ) = - \partial _ { \theta } \mathcal { R } ^ { < } ( \kappa , \nu , - \theta , - \eta , - \rho ) .
$$

Evaluating this at the manifold M where $\theta = \eta = \rho = 0 $ , we immediately obtain $\partial _ { \theta } \mathcal { R } ^ { < } ( \kappa , \bar { \nu } , 0 , 0 , 0 ) = 0$ . Identical arguments hold for $\partial _ { \eta } \mathcal { R } ^ { < }$ and $\partial _ { \rho } \mathcal { R } ^ { < }$ <sup><</sup>, concluding the proof of Lemma 11.

We now return to the proof of Lemma 6. By the chain rule, the Euclidean gradients of the risk with respect to k and v can be expanded in the basis of the parameter vectors:

$$
\begin{array} { r l } & { \nabla _ { k } \mathcal { R } = ( \partial _ { \boldsymbol { \kappa } } \mathcal { R } ^ { < } ) \boldsymbol { k } ^ { * } + ( \partial _ { \boldsymbol { \eta } } \mathcal { R } ^ { < } ) \boldsymbol { v } ^ { * } + ( \partial _ { \boldsymbol { \rho } } \mathcal { R } ^ { < } ) \boldsymbol { v } } \\ & { \nabla _ { \boldsymbol { v } } \mathcal { R } = ( \partial _ { \boldsymbol { \nu } } \mathcal { R } ^ { < } ) \boldsymbol { v } ^ { * } + ( \partial _ { \boldsymbol { \theta } } \mathcal { R } ^ { < } ) \boldsymbol { k } ^ { * } + ( \partial _ { \boldsymbol { \rho } } \mathcal { R } ^ { < } ) \boldsymbol { k } } \end{array}
$$

Applying Lemma 11, on the manifold $\mathcal { M }$ (where $\theta = \eta = \rho = 0 )$ , the gradients simplify strictly to:

$$
\begin{array} { r } { \nabla _ { k } \mathcal { R } ( k _ { t } , v _ { t } ) = \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa _ { t } , \nu _ { t } ) k ^ { * } } \\ { \nabla _ { v } \mathcal { R } ( k _ { t } , v _ { t } ) = \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa _ { t } , \nu _ { t } ) v ^ { * } } \end{array}\tag{21}
$$

Recall the PGD updates for $\left( k _ { t + 1 } , v _ { t + 1 } \right)$ :

$$
\begin{array} { r l } & { k _ { t + 1 } = \frac { k _ { t } - \alpha ( I _ { d } - k _ { t } k _ { t } ^ { T } ) \nabla _ { k } \mathcal { R } } { \| k _ { t } - \alpha ( I _ { d } - k _ { t } k _ { t } ^ { T } ) \nabla _ { k } \mathcal { R } \| _ { 2 } } : = \frac { \tilde { k } _ { t + 1 } } { c _ { k } } } \\ & { v _ { t + 1 } = \frac { v _ { t } - \alpha ( I _ { d } - v _ { t } v _ { t } ^ { T } ) \nabla _ { v } \mathcal { R } } { \| v _ { t } - \alpha ( I _ { d } - v _ { t } v _ { t } ^ { T } ) \nabla _ { v } \mathcal { R } \| _ { 2 } } : = \frac { \tilde { v } _ { t + 1 } } { c _ { v } } } \end{array}
$$

where $c _ { k } , c _ { v } > 0$ are the respective normalization constants. To prove $( k _ { t + 1 } , v _ { t + 1 } ) \in \mathcal { M } ,$ , we must verify the three orthogonality conditions: $( v ^ { * } ) ^ { T } k _ { t + 1 } = 0 , ( k ^ { * } ) ^ { T } \dot { v } _ { t + 1 } = \stackrel { . } { 0 }$ , and $v _ { t + 1 } ^ { T } k _ { t + 1 } = 0$

Condition 1: $( v ^ { * } ) ^ { T } k _ { t + 1 } = 0$ . Using the simplified gradient (21), we project $\tilde { k } _ { t + }$ <sub>1</sub> onto $v ^ { * }$

$$
( v ^ { * } ) ^ { T } \tilde { k } _ { t + 1 } = ( v ^ { * } ) ^ { T } k _ { t } - \alpha \partial _ { \kappa } \mathcal { R } ^ { < } \left( ( v ^ { * } ) ^ { T } k ^ { * } - ( v ^ { * } ) ^ { T } k _ { t } ( k _ { t } ^ { T } k ^ { * } ) \right) .
$$

Because $( k _ { t } , v _ { t } ) \in \mathcal { M }$ , we have $( v ^ { * } ) ^ { T } k _ { t } = 0$ . By Assumption 1, the oracle parameters are orthogonal, so $( v ^ { * } ) ^ { T } \dot { k } ^ { * } = \stackrel { \cdot } { 0 }$ . Thus:

$$
( v ^ { * } ) ^ { T } \tilde { k } _ { t + 1 } = 0 - \alpha \partial _ { \kappa } \mathcal { R } ^ { < } ( 0 - 0 ) = 0 \implies ( v ^ { * } ) ^ { T } k _ { t + 1 } = 0 .
$$

Condition 2: $( k ^ { * } ) ^ { T } v _ { t + 1 } = 0$ . By an exactly symmetric argument using $\tilde { v } _ { t + 1 }$ and the fact that $( k ^ { * } ) ^ { T } v _ { t } = 0$ , we obtain $( k ^ { * } ) ^ { T } v _ { t + 1 } = 0$

Condition 3: $v _ { t + 1 } ^ { T } k _ { t + 1 } = 0$ . We compute the inner product of the unnormalized updates $\tilde { v } _ { t + 1 } ^ { T } \tilde { k } _ { t + 1 }$

$$
\begin{array} { r l } & { \tilde { v } _ { t + 1 } ^ { T } \tilde { k } _ { t + 1 } = \left( v _ { t } - \alpha \partial _ { \nu } \mathcal { R } ^ { < } ( v ^ { * } - \nu _ { t } v _ { t } ) \right) ^ { T } \left( k _ { t } - \alpha \partial _ { \kappa } \mathcal { R } ^ { < } ( k ^ { * } - \kappa _ { t } k _ { t } ) \right) } \\ & { \qquad = v _ { t } ^ { T } k _ { t } - \alpha \partial _ { \kappa } \mathcal { R } ^ { < } \left( v _ { t } ^ { T } k ^ { * } - \kappa _ { t } v _ { t } ^ { T } k _ { t } \right) - \alpha \partial _ { \nu } \mathcal { R } ^ { < } \left( ( v ^ { * } ) ^ { T } k _ { t } - \nu _ { t } v _ { t } ^ { T } k _ { t } \right) } \\ & { \qquad + \alpha ^ { 2 } \partial _ { \kappa } \mathcal { R } ^ { < } \partial _ { \nu } \mathcal { R } ^ { < } \left( ( v ^ { * } - \nu _ { t } v _ { t } ) ^ { T } ( k ^ { * } - \kappa _ { t } k _ { t } ) \right) . } \end{array}
$$

Since $( k _ { t } , v _ { t } ) \in \mathcal { M }$ , we already know $v _ { t } ^ { T } k _ { t } = 0 , v _ { t } ^ { T } k ^ { * } = 0 .$ , and $( \boldsymbol { v } ^ { * } ) ^ { T } \boldsymbol { k } _ { t } = 0$ . Thus, all the linear terms vanish. For the quadratic term, we expand:

$$
( v ^ { * } - \nu _ { t } v _ { t } ) ^ { T } ( k ^ { * } - \kappa _ { t } k _ { t } ) = ( v ^ { * } ) ^ { T } k ^ { * } - \kappa _ { t } ( v ^ { * } ) ^ { T } k _ { t } - \nu _ { t } v _ { t } ^ { T } k ^ { * } + \nu _ { t } \kappa _ { t } v _ { t } ^ { T } k _ { t } .
$$

Substituting the orthogonalities $( v ^ { * } ) ^ { T } k ^ { * } = 0 , ( v ^ { * } ) ^ { T } k _ { t } = 0 , v _ { t } ^ { T } k ^ { * } = 0$ , and $v _ { t } ^ { T } k _ { t } = 0$ yields the desired orthogonality.

Therefore, $\tilde { v } _ { t + 1 } ^ { T } \tilde { k } _ { t + 1 } = 0$ , meaning $v _ { t + 1 } ^ { T } k _ { t + 1 } = 0$ . Having verified all three conditions, we conclude that $( k _ { t + 1 } , v _ { t + 1 } ) \in \mathcal { M } . \mathrm { q . e . d }$

## C.2 Proof of Lemma 7

Using the simplified gradient (21) valid on the manifold $\mathcal { M } ,$ we compute the unnormalized update for k:

$$
\tilde { k } _ { t + 1 } = k _ { t } - \alpha ( I _ { d } - k _ { t } k _ { t } ^ { T } ) ( \partial _ { \kappa } \mathcal { R } ^ { < } ) k ^ { * } = k _ { t } - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( k ^ { * } - \kappa _ { t } k _ { t } ) .
$$

Projecting this unnormalized update onto $k ^ { * }$ yields the numerator for $\kappa _ { t + 1 } \colon$

$$
( \boldsymbol { k } ^ { * } ) ^ { T } \tilde { \boldsymbol { k } } _ { t + 1 } = ( \boldsymbol { k } ^ { * } ) ^ { T } \boldsymbol { k } _ { t } - \alpha ( \partial _ { \boldsymbol { \kappa } } \mathcal { R } ^ { < } ) ( ( \boldsymbol { k } ^ { * } ) ^ { T } \boldsymbol { k } ^ { * } - \kappa _ { t } ( \boldsymbol { k } ^ { * } ) ^ { T } \boldsymbol { k } _ { t } ) = \kappa _ { t } - \alpha ( \partial _ { \boldsymbol { \kappa } } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) .
$$

To find the normalization constant $c _ { k } ~ = ~ \lVert \tilde { k } _ { t + 1 } \rVert _ { 2 }$ , we apply the Pythagorean theorem. Since $\left( k ^ { * } - \kappa _ { t } k _ { t } \right)$ is the projection of $k ^ { * }$ onto the orthogonal complement of $k _ { t }$ , it is strictly orthogonal to $k _ { t }$ . Thus:

$$
\begin{array} { r l } & { \| \tilde { k } _ { t + 1 } \| _ { 2 } ^ { 2 } = \| k _ { t } \| _ { 2 } ^ { 2 } + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } \| k ^ { * } - \kappa _ { t } k _ { t } \| _ { 2 } ^ { 2 } } \\ & { \qquad = 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) . } \end{array}
$$

Dividing the numerator by $\| \tilde { k } _ { t + 1 } \| _ { 2 }$ directly yields $g _ { 1 } ( \kappa _ { t } , \nu _ { t } )$ . Analogously, one obtains the result for $g _ { 2 } ( \kappa _ { t } , \nu _ { t } )$ .

## C.3 Proof of Proposition 8

The theoretical risk $\mathcal { R } ^ { < }$ is composed of smooth error functions evaluated over Gaussian expectations, ensuring it is ${ \mathcal { C } } ^ { \infty }$ on the compact square $[ - 1 , 1 ] ^ { 2 }$ . Consequently, it is Λ-smooth, meaning we can bound its variation:

$$
\begin{array} { l } { \displaystyle \mathcal { R } ^ { < } ( \kappa _ { t + 1 } , \nu _ { t + 1 } ) - \mathcal { R } ^ { < } ( \kappa _ { t } , \nu _ { t } ) \leq ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( \kappa _ { t + 1 } - \kappa _ { t } ) + ( \partial _ { \nu } \mathcal { R } ^ { < } ) ( \nu _ { t + 1 } - \nu _ { t } ) } \\ { \displaystyle \qquad + \frac { \Lambda } { 2 } \left[ ( \kappa _ { t + 1 } - \kappa _ { t } ) ^ { 2 } + ( \nu _ { t + 1 } - \nu _ { t } ) ^ { 2 } \right] . } \end{array}\tag{22}
$$

Using the discrete mapping g from Lemma 7, we rewrite the step difference for κ:

$$
\kappa _ { t + 1 } - \kappa _ { t } = \kappa _ { t } \left( \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) } } - 1 \right) - \frac { \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) } }
$$

$$
\begin{array} { l } { \displaystyle = - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) } \\ { \displaystyle \quad + \left( \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) } } - 1 \right) ( \kappa _ { t } - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) ) . } \end{array}\tag{23}
$$

As a consequence,

$$
\begin{array} { r l r } { | \kappa _ { t + 1 } - \kappa _ { t } + \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) | \leq \left| \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) } } - 1 \right| \left| \kappa _ { t } - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) \right| } & { } & \\ { \leq \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) \left| \kappa _ { t } - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) \right| } & { } & \\ { \leq C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) , } & { ( 2 4 } \end{array}
$$

where the second inequality holds by Lemma 14 and the last bound holds since the function $( \kappa , \nu ) \mapsto$ $\left| \kappa - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ( 1 - \kappa ^ { 2 } ) \right|$ is uniformly bounded for all $\alpha \leq 1$ . This bound has two implications. First,

$$
\begin{array} { r l } { ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( \kappa _ { t + 1 } - \kappa _ { t } ) + \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) = ( \partial _ { \kappa } \mathcal { R } ^ { < } ) \left( ( \kappa _ { t + 1 } - \kappa _ { t } ) + \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) \right) } & { } \\ { \leq | ( \partial _ { \kappa } \mathcal { R } ^ { < } ) | \left| ( \kappa _ { t + 1 } - \kappa _ { t } ) + \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) \right| } & { } \\ { \leq C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) , } & { } \end{array}\tag{25}
$$

where we use the fact that $| \partial _ { \kappa } \mathcal { R } ^ { < } |$ is bounded and the bound (24). Second, since the square function is Lipschitz on compact sets, we have

$$
\begin{array} { r } { \left| ( \kappa _ { t + 1 } - \kappa _ { t } ) ^ { 2 } - \left( \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ( 1 - \kappa _ { t } ^ { 2 } ) \right) ^ { 2 } \right| \leq C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) . } \end{array}
$$

Thus,

$$
\begin{array} { r l } & { ( \kappa _ { t + 1 } - \kappa _ { t } ) ^ { 2 } \leq \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) ^ { 2 } + C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) } \\ & { \qquad \leq C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) . } \end{array}\tag{26}
$$

We obtain analogous bounds for $\nu ,$ namely

$$
( \partial _ { \nu } \mathcal { R } ^ { < } ) ( \nu _ { t + 1 } - \nu _ { t } ) + \alpha ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) \leq C \alpha ^ { 2 } ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } )\tag{27}
$$

and

$$
( \nu _ { t + 1 } - \nu _ { t } ) ^ { 2 } \leq C \alpha ^ { 2 } ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) .\tag{28}
$$

Plugging the bounds (25), (26), (27), (28) into Equation (22), we obtain the desired inequality

$$
\begin{array} { r l } & { \mathcal { R } ^ { < } ( \kappa _ { t + 1 } , \nu _ { t + 1 } ) - \mathcal { R } ^ { < } ( \kappa _ { t } , \nu _ { t } ) \leq - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) - \alpha ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) } \\ & { \qquad + C \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) + C \alpha ^ { 2 } ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) . } \end{array}
$$

By choosing the step size $\begin{array} { r } { \alpha \leq \frac { 1 } { 2 C } } \end{array}$ , this ensures that

$$
\mathcal { R } ^ { < } ( \kappa _ { t + 1 } , \nu _ { t + 1 } ) - \mathcal { R } ^ { < } ( \kappa _ { t } , \nu _ { t } ) \leq - \frac { \alpha } { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) - \frac { \alpha } { 2 } ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) .
$$

This shows that the risk is decreasing along the PGD iterates. Next, introducing $\mathcal { R } _ { \operatorname* { m i n } } ^ { < } ~ =$ min $1 _ { ( \kappa , \nu ) \in [ - 1 , 1 ] ^ { 2 } } \mathcal { R } ^ { < } ( \kappa , \nu )$ and using a telescopic sum, we have, for all $T \geq 0$

$$
\begin{array} { r l } & { \mathcal { R } ^ { < } ( \kappa _ { 0 } , \nu _ { 0 } ) - \mathcal { R } _ { \operatorname* { m i n } } ^ { < } \geq \mathcal { R } ^ { < } ( \kappa _ { 0 } , \nu _ { 0 } ) - \mathcal { R } ^ { < } ( \kappa _ { T } , \nu _ { T } ) } \\ & { \qquad \geq \displaystyle \frac { \alpha } { 2 } \sum _ { t = 0 } ^ { T - 1 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) + ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } ) . } \end{array}
$$

Since the left-hand side is finite, and the terms of the sum are nonnegative, we conclude that the series converges as $T \to \infty$ . In particular, the generic term $( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa _ { t } ^ { 2 } ) + ( \partial _ { \nu } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \nu _ { t } ^ { 2 } )$ of the series converges to 0 as $t \to \infty$ . Therefore, the accumulation points $\left( \kappa _ { \infty } , \nu _ { \infty } \right)$ satisfy

$$
\left\{ \begin{array} { c c c } { { \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa _ { \infty } , \nu _ { \infty } ) = 0 } } & { { \mathrm { o r } } } & { { \kappa _ { \infty } ^ { 2 } = 1 } } \\ { { \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa _ { \infty } , \nu _ { \infty } ) = 0 } } & { { \mathrm { o r } } } & { { \nu _ { \infty } ^ { 2 } = 1 . } } \end{array} \right.
$$

Inspecting identity (23), we observe that the convergence of the general term also implies $\kappa _ { t + 1 } - \kappa _ { t } $ 0. Similarly, $\nu _ { t + 1 } - \nu _ { t } \to 0$ . q.e.d.

## C.4 Proof of Proposition 9

First we recall the formula of the risk

$$
\begin{array} { r l } & { \mathbb { E } ^ { \mathcal { K } } ( s , i , j ) = } \\ & { \quad - \sigma ^ { 2 } \| s \| \displaystyle \left[ s c t \Big ( \lambda _ { x } e t ( \lambda _ { x } , s \xi - \lambda _ { x } \zeta _ { 0 } ^ { \lambda } ) \Big ) ( \alpha + \gamma ( \xi _ { 0 } ^ { \lambda } ) + \lambda _ { x } \displaystyle \sum _ { j = 1 } ^ { N } \alpha + \Gamma _ { 0 } ^ { \lambda } \Big ( \lambda _ { x } \xi _ { j } ^ { \lambda } \Big ) \mathcal { G } _ { j } ^ { \lambda } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \end{array}\tag{29}
$$

where C does not depend on κ and $\nu .$

$$
V ( x _ { 1 } , x _ { 2 } , \ldots , x _ { N } ) : = \operatorname { e r f } \left( { \frac { \lambda _ { c } } { \lambda _ { r } } } \sum _ { j = 1 } ^ { N } \operatorname { e r f } ( x _ { j } ) x _ { j } \right)\tag{30}
$$

and by

$$
U ( x _ { 1 } , x _ { 2 } , . . . , x _ { N } ) : = \ V ( x _ { 1 } , . . . , x _ { N } ) \mathrm { e r f } ( x _ { 1 } )\tag{31}
$$

Furthermore, we shall make a slight abuse of notation and denote by

$$
\begin{array} { l } { { \displaystyle V ( \kappa ) : = V \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } , \lambda _ { r } G _ { 1 2 } ^ { k } , \ldots , \lambda _ { r } G _ { 1 N } ^ { k } \right) } } \\ { { \displaystyle \quad = \mathrm { e r f } \left( \lambda _ { c } \mathrm { e r f } \left( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } \right) \left( \kappa \ell + \gamma G _ { 1 1 } ^ { k } \right) + \lambda _ { c } \sum _ { j = 2 } ^ { N } \mathrm { e r f } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) G _ { 1 j } ^ { k } \right) } } \end{array}
$$

and by

$$
U ( \kappa ) : = V ( \kappa ) \cdot \mathrm { e r f } ( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } ) .
$$

One can easily observe that $V ( \kappa )$ is symmetric in all $G _ { 1 j } ^ { k }$ and in κ since $x \mathrm { e r f } ( x )$ is symmetric around 0 and that $| V ( \kappa ) | \leq 1$ . Moreover, this implies that ${ \bar { U } } ( \kappa )$ is strictly increasing in κ and that $| U ( \kappa ) | \leq 1$ . Finally, the risk can be rewritten as

$$
\mathcal { R } ^ { < } ( \kappa , \nu ) = - 2 \gamma ^ { 2 } \nu \mathbb { E } \left[ U ( \kappa ) \right] + \gamma ^ { 2 } \mathbb { E } \left[ U ( \kappa ) ^ { 2 } \right] + \mathbb { E } \left[ V ( \kappa ) ^ { 2 } \sum _ { j = 2 } ^ { N } \mathrm { e r f } ^ { 2 } \big ( \lambda _ { r } G _ { 1 j } ^ { k } \big ) \right] + C .
$$

Now, $( \kappa , \nu )$ is a stationary point of the PGD dynamics if

$$
\left\{ \begin{array} { l l } { \partial _ { \boldsymbol { \kappa } } \mathcal { R } ^ { < } ( { \boldsymbol { \kappa } } , { \boldsymbol { \nu } } ) = 0 \quad \mathrm { o r } \quad \kappa ^ { 2 } = 1 } \\ { \partial _ { \boldsymbol { \nu } } \mathcal { R } ^ { < } ( { \boldsymbol { \kappa } } , { \boldsymbol { \nu } } ) = 0 \quad \mathrm { o r } \quad \nu ^ { 2 } = 1 . } \end{array} \right.
$$

The partial derivatives of the risk are

$$
\begin{array}{c} \left. \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) = 2 \gamma ^ { 2 } \mathbb { E } \left[ \partial _ { \kappa } U ( \kappa ) ( U ( \kappa ) - \nu ) \right] + 2 \mathbb { E } \left[ \partial _ { \kappa } V ( \kappa ) V ( \kappa ) \sum _ { j = 2 } ^ { N } \mathrm { e r f } ^ { 2 } \big ( \lambda _ { r } G _ { 1 j } ^ { k } \big ) \right]  \\ { \partial _ { \nu } \mathcal { R } ^ { < } ( \kappa , \nu ) = - 2 \gamma ^ { 2 } \mathbb { E } \left[ U ( \kappa ) \right] . } \end{array} \right.\tag{32}
$$

We split our analysis in different cases.

Case 1. $\partial _ { \nu } \mathcal { R } ^ { < } ( \kappa , \nu ) = 0$ This implies that $\mathbb { E } [ U ( \kappa ) ] = 0$ . It is clear that $\mathbb { E } [ U ( 0 ) ] = 0$ and due to the fact that $U$ is strictly increasing, $\kappa = 0$ is also the unique such point.   
Now,

$$
S ( \kappa ) : = \gamma ^ { 2 } \mathbb { E } \left[ U ( \kappa ) ^ { 2 } \right] + \mathbb { E } \left[ V ( \kappa ) ^ { 2 } \sum _ { j = 2 } ^ { N } \operatorname { e r f } ^ { 2 } \left( \lambda _ { r } G _ { 1 j } ^ { k } \right) \right]\tag{33}
$$

is jointly symmetric in $\left( \kappa , \left( G _ { 1 j } ^ { k } \right) _ { j \in [ N ] } \right)$ . Taking the expectation over $\big ( G _ { 1 j } ^ { k } \big ) _ { j \in [ N ] } ,$ we obtain that the term in (33) is symmetric in $\kappa ,$ so its derivative must satisfy $\partial _ { \kappa } S ( 0 ) = 0$ . This implies that

$$
0 = \partial _ { \kappa } \mathcal { R } ^ { < } ( 0 , \nu ) = - 2 \gamma ^ { 2 } \nu \mathbb { E } \left[ \partial _ { \kappa } U ( 0 ) \right]
$$

But $U$ is strictly increasing, which implies that $\nu = 0$

Therefore, the only critical point of $\mathcal { R } ^ { < }$ <sup><</sup> inside of $[ - 1 , 1 ] ^ { 2 }$ is (0, 0). Moreover, since the Hessian

$$
\mathcal { H } _ { \mathcal { R } ^ { < } } ( 0 , 0 ) = \binom { \partial _ { \kappa \kappa } \mathcal { R } ^ { < } } { - 2 \gamma ^ { 2 } \mathbb { E } \left[ \partial _ { \kappa } U ( 0 ) \right] } \quad \frac { - 2 \gamma ^ { 2 } \mathbb { E } \left[ \partial _ { \kappa } U ( 0 ) \right] } { 0 } ,
$$

we obtain that (0, 0) is a saddle point.

Case $2 . \nu = - 1$ It is clear that $\partial _ { \kappa } \mathcal { R } ^ { < } ( 0 , - 1 ) > 0$ , since $\mathbb { E } [ U ( \kappa ) ^ { 2 } ]$ and $\mathbb { E } [ V ( \kappa ) ^ { 2 } ]$ are even functions and $\mathbb { E } [ U ( \kappa ) ]$ is an increasing function in κ. Moreover, on $\kappa \in [ 0 , 1 ]$ , the function $\mathcal { R } ^ { < } ( \kappa , - 1 )$ is clearly increasing in κ since $\mathbb { E } [ U ( \kappa ) ]$ and $\mathbb { E } [ V ^ { 2 } ( \kappa ) ]$ ] are.

Let us now denote by $Z _ { 1 } : = \kappa \ell + \gamma G _ { 1 1 } ^ { k }$ and by $Z _ { j } : = G _ { 1 j } ^ { k } , \mathrm { f o r } j \in \{ 2 , 3 , \ldots , N \}$ . Then, one has

$$
\begin{array} { r } { \{ \partial _ { \kappa } U ( \kappa ) = \partial _ { \kappa } U ( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } , \lambda _ { r } G _ { 1 2 } ^ { k } , \ldots , \lambda _ { r } G _ { 1 N } ^ { k } ) = \lambda _ { r } \ell \partial _ { 1 } U ( \lambda _ { r } Z _ { 1 } , \ldots , \lambda _ { r } Z _ { N } ) , } \\ { \partial _ { \kappa } V ( \kappa ) = \partial _ { \kappa } V ( \lambda _ { r } \kappa \ell + \lambda _ { r } \gamma G _ { 1 1 } ^ { k } , \lambda _ { r } G _ { 1 2 } ^ { k } , \ldots , \lambda _ { r } G _ { 1 N } ^ { k } ) = \lambda _ { r } \ell \partial _ { 1 } V ( \lambda _ { r } Z _ { 1 } , \ldots , \lambda _ { r } Z _ { N } ) . } \end{array}\tag{34}
$$

By directly applying the chain rule to the expectation, the formula for the partial derivative of the risk becomes

$$
\begin{array} { r l } & { \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , - 1 ) = \partial _ { \kappa } \mathbb { E } \left[ \gamma ^ { 2 } \left( U ( \kappa ) ^ { 2 } + 2 U ( \kappa ) \right) + V ( \kappa ) ^ { 2 } \underset { j = 2 } { \overset { N } { \sum } } \mathrm { e r f } ^ { 2 } ( \lambda _ { r } Z _ { j } ) \right] } \\ & { \qquad = \lambda _ { r } \ell \mathbb { E } \left[ \gamma ^ { 2 } \partial _ { 1 } \left( U ( \lambda _ { r } Z ) ^ { 2 } + 2 U ( \lambda _ { r } Z ) \right) + \partial _ { 1 } \left( V ( \lambda _ { r } Z ) ^ { 2 } \right) \underset { j = 2 } { \overset { N } { \sum } } \mathrm { e r f } ^ { 2 } ( \lambda _ { r } Z _ { j } ) \right] } \\ & { \qquad = 2 \lambda _ { r } \ell \mathbb { E } \left[ \gamma ^ { 2 } \partial _ { 1 } U ( \lambda _ { r } Z ) \left( 1 + U ( \lambda _ { r } Z ) \right) + V ( \lambda _ { r } Z ) \partial _ { 1 } V ( \lambda _ { r } Z ) \underset { j = 2 } { \overset { N } { \sum } } \mathrm { e r f } ^ { 2 } ( \lambda _ { r } Z _ { j } ) \right] , } \end{array}\tag{35}
$$

where $U ( \lambda _ { r } Z )$ stands for $U ( \lambda _ { r } Z _ { 1 } , . . . , \lambda _ { r } Z _ { N } )$ and the same for the other functions.

We will now show that if $\lambda _ { c } = \tau \lambda _ { r }$ , for $\lambda _ { r }$ small enough and the other hyperparameters fixed, the expectation above is strictly positive for all $\kappa \in [ - 1 , 1 ]$ ]. We shall make use of the Taylor expansions

$$
\operatorname { e r f } ( x ) = { \frac { 2 } { \sqrt { \pi } } } x + { \frac { \operatorname { e r f } ^ { \prime \prime \prime } ( \xi _ { x } ) } { 6 } } x ^ { 3 } = { \frac { 2 } { \sqrt { \pi } } } x + { \mathcal { O } } ( x ^ { 3 } ) ,
$$

for some $\xi _ { x } \in [ 0 , x ]$ , and

$$
e ^ { - x } = 1 - e ^ { - \zeta _ { x } } x = 1 + \mathcal { O } ( x ) ,
$$

for some $\zeta _ { x } \in [ 0 , x ]$ . Note that $| \mathrm { e r f } ^ { \prime \prime \prime } ( x ) | < 2 . 5$ , for any $x \in \mathbb { R }$ and $| e ^ { - \zeta _ { x } } | \leq 1$ for any $x \in [ 0 , \infty )$ This in turn implies the following asymptotic scalings

$$
V ( \lambda _ { r } Z ) = \mathrm { e r f } \left( \tau \lambda _ { r } \sum _ { i = 1 } ^ { N } Z _ { i } \mathrm { e r f } ( \lambda _ { r } Z _ { i } ) \right) = \frac { 4 \tau } { \pi } \lambda _ { r } ^ { 2 } \sum _ { i = 1 } ^ { N } Z _ { i } ^ { 2 } + \mathcal { O } ( \lambda _ { r } ^ { 3 } ) ,
$$

$$
\begin{array} { r l } & { V ( \lambda , Z ) = \ : V ( \lambda , Z ) \mathrm { e r f } ( \lambda , Z _ { 1 } ) = \frac { 8 \pi ^ { 2 } } { \gamma \lambda ^ { 3 } } \sum _ { i = 1 } ^ { N } Z _ { i } ^ { N } \sum _ { \lambda = 1 } ^ { N } Z _ { i } ^ { N } + O ( \lambda _ { i } ^ { \lambda } ) , } \\ & { \ : \ : } \\ & { \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : \ : \ : \widetilde { \mathcal { H } } _ { 1 } V ( \lambda , Z ) = \frac { 2 \pi ^ { 2 } } { \gamma \lambda ^ { 3 } } e ^ { - \alpha ^ { 2 } ( \sum _ { i = 1 } ^ { N } \lambda _ { i } Z _ { i } + 2 Z _ { i } ) ( \lambda _ { i } - Z _ { i } ) ) ^ { 2 } } \ : \ : \left( \frac { 2 } { \sqrt { \lambda ^ { 3 } } } x , Z _ { i } ( \lambda - \lambda ^ { 2 } Z _ { i } + \mathrm { e r f } ( \lambda , Z _ { 1 } ) ) \right. } \\ & { \ : \ : \ : \ : \ : } \\ & { = \frac { 2 \pi ^ { 2 } } { \gamma \lambda ^ { 3 } } \left( 1 - \mathcal { O } ( \lambda _ { i } ^ { 2 } ) \right) \left( \frac { 2 } { \sqrt { \pi } ^ { 2 } } \lambda , Z _ { i } ( 1 - \mathcal { O } ( \lambda _ { i } ^ { 2 } ) ) + \frac { 2 } { \sqrt { \pi } ^ { 3 } } x , Z _ { i } \right) } \\ & { \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : \ : \ : \ : \frac { 8 \pi } { \gamma \lambda ^ { 3 } } \sum _ { i = 1 } ^ { N } \mathcal { O } ( \lambda _ { i } ^ { 3 } ) , } \\ & { \ : \ : \ : \ : \ : \ : } \\ &  \ : \ : \ : \ : \ : \ : \frac { 8 \pi ^ { 2 } } { \gamma \lambda ^ { 4 } } \ : \ : \frac { 2 } { \lambda ^ { 4 } } \ : V ( \lambda , Z ) \mathrm { e r f } ( \lambda , \end{array}\tag{36}
$$

From these computations, it is also clear that every term $\mathcal { O } ( \lambda _ { r } ^ { n } ) = C ( Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { N } ) \lambda _ { r } ^ { n }$ , where $\mathbb { E } [ | C ( Z _ { 1 } , \dots , Z _ { N } ^ { \cdot } ) | ] < \infty$

Finally, (35) becomes

$$
\partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , - 1 ) = \frac { 1 6 \tau \ell \gamma ^ { 2 } } { \pi \sqrt { \pi } } \lambda _ { r } ^ { 3 } \mathbb { E } \left[ \sum _ { i = 1 } ^ { N } Z _ { i } ^ { 2 } + 2 Z _ { 1 } ^ { 2 } \right] + \mathcal { O } ( \lambda _ { r } ^ { 4 } ) .
$$

Since the coefficient of the leading term is positive, this concludes the proof that there exists $\lambda _ { r }$ small enough such that $\partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , - 1 ) \bar { > } 0$ for all $\kappa \in [ - 1 , 1 ]$

Case $3 . \ \nu = 1 \ \mathrm { ~ \normalfont ~ { ~ B y ~ } ~ }$ leveraging the global symmetry of the theoretical risk function, specifically $\mathcal { R } ^ { < } ( - \kappa , - \nu ) = \mathcal { R } ^ { \overset { . } { < } } ( \kappa , \nu )$ , an analogous argument applies along the opposite boundary $\nu = 1$ . The identical asymptotic smoothness guarantees $\partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , 1 ) < 0$ for all $\kappa \in [ - 1 , 1 ]$

Finally combining the results of the three cases, one obtains the conclusion.

## C.5 Proof of Proposition 10

We verify the claims of the proposition sequentially.

Local diffeomorphism and eigenvalues at the origin. Let $x = \left( \kappa , \nu \right)$ . Because the gradient of the risk vanishes at the origin $( \partial _ { \kappa } \mathcal { R } ^ { \breve { < } } ( 0 , 0 ) = 0 )$ and $\mathcal { R } ^ { < }$ is smooth, the squared partial derivative satisfies $( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } = \mathcal { O } (  { \lVert { \boldsymbol { x } } \rVert } _ { 2 } ^ { 2 } )$ . This allows us to write the Taylor expansion for the first component of $^ { g , }$ denoted $g _ { 1 } ( \kappa , \nu )$

$$
\begin{array} { l } { g _ { 1 } ( \kappa , \nu ) = \displaystyle \frac { \kappa - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ( 1 - \kappa ^ { 2 } ) } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \kappa ^ { 2 } ) } } } \\ { = \displaystyle \frac { \kappa - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ( 1 - \kappa ^ { 2 } ) } { \sqrt { 1 + \mathcal { O } ( \| x \| _ { 2 } ^ { 2 } ) } } } \\ { = ( \kappa - \alpha ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ( 1 - \kappa ^ { 2 } ) ) ( 1 + \mathcal { O } ( \| x \| _ { 2 } ^ { 2 } ) ) } \\ { = \kappa - \alpha \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) + \mathcal { O } ( \| x \| _ { 2 } ^ { 2 } ) . } \end{array}
$$

Applying the same expansion to the second component $g _ { 2 } ( \kappa , \nu )$ yields the Jacobian matrix of $g$ evaluated at the origin, $J _ { g } ( 0 , 0 ) = I _ { 2 } - \alpha H _ { \mathcal { R } ^ { < } } ( 0 , 0 )$ . Because det $\left( H _ { \mathcal { R } ^ { < } } ( 0 , 0 ) \right) = - 4 \gamma ^ { 4 } \left( \mathbb { E } [ \partial _ { \kappa } U ( 0 ) ] \right) ^ { 2 }$ from the proof of Proposition 9, the eigenvalues of $H _ { \mathcal { R } ^ { < } } ( 0 , 0 )$ have strictly opposite signs. Consequently, for a sufficiently small step size $\alpha > 0 , J _ { g } ( 0 , 0 )$ possesses one eigenvalue strictly inside the interval (0, 1) and another strictly greater than 1. This confirms that $J _ { g } ( 0 , 0 )$ is invertible, which by the Inverse Function Theorem ensures $g$ is a local diffeomorphism around (0, 0).

Differentiability and non-degenerate Jacobian on $[ - 1 , 1 ] ^ { 2 }$ . The mapping g is globally differentiable on $[ - 1 , 1 ] ^ { \circ }$ as it is a composition of smooth elementary functions. To prove its Jacobian is nowhere degenerate, we express the mapping as $g ( x ) = x + \alpha h ( x )$ . The first coordinate of h takes the form:

$$
\begin{array} { r l r } {  { h _ { 1 } ( \kappa , \nu ) = \frac { 1 } { \alpha } ( g _ { 1 } ( \kappa , \nu ) - \kappa ) } } \\ & { } & { = \underbrace { \frac { \kappa } { \alpha } ( \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \kappa ^ { 2 } ) } } - 1 ) } _ { \mathrm { T e m ~ 1 } } - \underbrace { \frac { \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ( 1 - \kappa ^ { 2 } ) } { \sqrt { 1 + \alpha ^ { 2 } ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \kappa ^ { 2 } ) } } } _ { \mathrm { T e m ~ 2 } } . } \end{array}
$$

We must prove that the gradient of $h _ { 1 } ( \kappa , \nu )$ remains uniformly bounded for all $\alpha \leq 1$ . Term 2 is a composition of functions with uniformly bounded gradients on the compact domain, so its gradient is trivially bounded. For Term 1, we introduce the auxiliary function $\begin{array} { r } { q ( a , b ) = \frac { a } { \alpha } \left( \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } b } } - 1 \right) } \end{array}$ and define the supremum $B = \mathrm { s u p } _ { [ - 1 , 1 ] ^ { 2 } } ( \partial _ { \kappa } \mathcal { R } ^ { < } ( \kappa , \nu ) ) ^ { 2 } ( 1 - \kappa ^ { 2 } )$ . Term 1 can be expressed as $q ( \kappa , ( \partial _ { \kappa } \mathcal { R } ^ { < } ) ^ { 2 } ( 1 - \kappa ^ { 2 } ) )$ . Since the inner function is smooth and independent of α, we only need to bound the partial derivatives of $q .$ . By Lemma 14, for any $\alpha \leq 1 { : }$

$$
\begin{array} { l } { \displaystyle | \partial _ { a } q ( a , b ) | = \frac { 1 } { \alpha } \left| \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } b } } - 1 \right| \leq \alpha b \leq B , } \\ { \displaystyle | \partial _ { b } q ( a , b ) | = \left| - \frac { \alpha a } { 2 ( 1 + \alpha ^ { 2 } b ) ^ { 3 / 2 } } \right| \leq \frac { \alpha } { 2 } \leq \frac { 1 } { 2 } . } \end{array}
$$

Since both partial derivatives are uniformly bounded, the gradient of $h _ { 1 }$ is uniformly bounded. A symmetric argument applies to $h _ { 2 } ,$ confirming that the Jacobian matrix $J _ { h } ( \kappa , \nu )$ is uniformly bounded over $\alpha \leq 1$ . Because $J _ { g } ( \kappa , \nu ) = I _ { 2 } + \bar { \alpha ^ { \prime } } J _ { h } ( \kappa , \nu )$ , selecting a sufficiently small α ensures the eigenvalues of $J _ { g }$ remain strictly positive, guaranteeing the Jacobian is nowhere degenerate.

Injectivity. The uniform bound on $J _ { h }$ implies that h is B -Lipschitz continuous on $[ - 1 , 1 ] ^ { 2 }$ for some constant $B _ { L } > 0$ independent of α (for small α). Choose $\alpha < 1 / B _ { L }$ . Suppose there exist two distinct points $x , y \in [ - 1 , 1 ] ^ { 2 }$ such that $g ( x ) = g ( y )$ . This implies:

$$
x + \alpha h ( x ) = y + \alpha h ( y ) \implies x - y = \alpha ( h ( y ) - h ( x ) ) .
$$

Taking the Euclidean norm of both sides yields:

$$
\| x - y \| _ { 2 } = \alpha \| h ( y ) - h ( x ) \| _ { 2 } \leq \alpha B _ { L } \| x - y \| _ { 2 } .
$$

Because α $B _ { L } < 1$ , this inequality forces $\| x - y \| _ { 2 } < \| x - y \| _ { 2 }$ , which is a strict contradiction. Therefore, g must be injective on the domain $[ - 1 , \dot { 1 } ] ^ { 2 }$

## C.6 Final details.

From Proposition 8, the distance between successive iterates $\lVert ( \kappa _ { t + 1 } , \nu _ { t + 1 } ) - ( \kappa _ { t } , \nu _ { t } ) \rVert _ { 2 }$ tends to zero. Because the sequence $( \kappa _ { t } , \nu _ { t } ) _ { t \geq 0 }$ lies within the compact set $[ - 1 , 1 ] ^ { 2 }$ , the set of its accumulation points must be connected. Given that the set of available fixed points is finite and isolated $\{ ( 0 , 0 ) , ( \pm 1 , \pm 1 ) , ( \pm 1 , \mp 1 ) \}$ , the sequence cannot oscillate and must uniquely converge to exactly one of them.

It remains to be shown that the probability of converging to the saddle point (0, 0) or the maxima $( \pm 1 , \mp 1 )$ is zero. Because the risk strictly decreases at every step (Proposition 8), convergence to the global maxima is impossible unless initialized exactly upon them.

To systematically rule out the saddle point $( 0 , 0 )$ , we apply the Center-Stable Manifold Theorem [Shub, 2013, Theorem III.7]. Because $J _ { g } ( 0 , \stackrel { . } { 0 } )$ has exactly one eigenvalue with magnitude strictly greater than 1 (Proposition 10), the local center-stable manifold $\hat { W } _ { l o c } ^ { c s }$ associated with the origin has codimension 1. In the 2-dimensional space $[ - 1 , 1 ] ^ { 2 }$ , a codimension-1 manifold is a curve, and its Lebesgue measure is strictly zero.

The stable set $W ^ { s }$ , consisting of all initializations that converge to (0, 0) under iterations of $^ { g , }$ is contained within the countable union of pre-images of $W _ { l o c } ^ { c s } { . }$

$$
W ^ { s } \subset \bigcup _ { T \geq 0 } g ^ { - T } ( W _ { l o c } ^ { c s } ) .\tag{37}
$$

Because g is injective and differentiable, its inverse preserves sets of measure zero. The countable union of measure-zero sets is also of measure zero.

Therefore, the set of all initializations on M that converge to the saddle point (0, 0) has Lebesgue measure zero. We conclude that for a generic random initialization $( k _ { 0 } , v _ { 0 } ) \in \mathcal { M }$ , the PGD sequence almost surely bypasses the saddle point and the maxima, converging strictly to the true oracle structure (±1, ±1).

## D Technical lemmas

In this section we list the technical lemmas used throughout the proofs in the Appendix.

Lemma 12. For $u \in \mathbb { R } ,$

$$
e r f ^ { 2 } ( u ) \leq \frac { 4 } { \pi } u ^ { 2 } .
$$

Proof. The proof relies on the concavity of erf on $( 0 , \infty )$ . Since erf is an odd function, it is enough to prove the statement for $u \geq 0$ . Since erf is concave,

$$
\operatorname { e r f } ( u ) \leq \operatorname { e r f } ( 0 ) + \operatorname { e r f } ^ { \prime } ( 0 ) u = { \frac { 2 } { \sqrt { \pi } } } u ,
$$

which implies the conclusion.

Lemma 13. For $G \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ and $t \in \mathbb { R }$

$$
\mathbb { E } [ \operatorname { e r f } ( t + G ) ] = \operatorname { e r f } \left( { \frac { t } { \sqrt { 1 + 2 \sigma ^ { 2 } } } } \right) .
$$

Proof. One has

$$
\begin{array} { r l } { \mathcal { L } ( \mathbf { r i } ( t ) + \mathcal { L } ) = } & { \int _ { 0 } ^ { \infty } \mathrm { d } w ( t ) \mathrm { d } t \leq \mathcal { O } ( \mathrm { l i } t ) } \\ & { = \int _ { 0 } ^ { \infty } \mathrm { b } \left\{ \int _ { 0 } ^ { 2 } w ^ { \mathrm { i d } ( t ) } w ^ { \mathrm { i d } ( t ) } \mathrm { d } t \right\} } \\ & { = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } \mathrm { d } w \left\{ \int _ { 0 } ^ { 2 } w ^ { \mathrm { i d } ( t ) } w ^ { \mathrm { i d } ( t ) } \mathrm { d } t \right\} } \\ & { = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } \mathrm { d } w ^ { \mathrm { i d } ( t ) } \mathrm { d } t = \mathcal { O } ( \mathrm { r i } ( \mathrm { s . e . c . } ) ) ^ { \prime } \mathrm { d } w \mathrm { d } t } \\ & { = \frac { w _ { 0 } ^ { 2 } } { w _ { 0 } } \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } \mathrm { d } w ^ { \mathrm { i d } ( t ) } \mathrm { d } t \leq \mathcal { O } ( \mathrm { r i } ( \mathrm { s . e . c . } ) ) } \\ &  = \frac { w _ { 0 } ^ { 2 } } { w _ { 0 } } \int _ { 0 } ^ { t } \int _ { - \infty } ^ { \infty } e ^ { - \left( \mathrm { e } ^ { \mathrm { i } ( t ) } \left\{ \frac { w _ { 0 } ^ { 2 } } { 1 + t ^ { 2 } } - w \right\} \left\{ \frac { w _ { 0 } ^ { 2 } } { 1 + t ^ { 2 } } \right\} \right\} } \\ & { = \frac { w _ { 0 } ^ { 2 } } { w _ { 0 } } \int _ { 0 } ^ { t } \int _ { - \infty } ^ { \infty } \int _ { 0 } ^ { \infty } \mathrm { d } w \left\{ \frac { w _ { 0 } ^ { 2 } } { 1 + t ^ { 2 } } \right\} \mathrm { d } t } \\ &  = \frac  w _ { 0 } ^   \end{array}
$$

Lemma 14. $F o r u \ge 0 ,$

$$
\left| { \frac { 1 } { \sqrt { 1 + u } } } - 1 \right| \leq u .
$$

Proof. For $u \geq 0$ , the term inside the absolute value is at most 0, so we need to prove that

$$
u + \frac { 1 } { \sqrt { 1 + u } } \geq 1 .
$$

This statement is obvious for $u \geq 1$ . For $u \in [ 0 , 1 )$ , it is equivalent to

$$
{ \frac { 1 } { \sqrt { 1 + u } } } \geq 1 - u \iff { \frac { 1 } { 1 + u } } \geq ( 1 - u ) ^ { 2 } \iff 1 \geq ( 1 - u ^ { 2 } ) ( 1 - u ) ,
$$

which is clearly true since $u \in [ 0 , 1 )$ . One has equality only when $u = 0$

## E Additional Experimental Details and Results

In this section, we provide supplementary empirical results that expand upon the optimization dynamics discussed in Section 4. Specifically, we analyze the training dynamics of the predictors without manifold penalties in Section E.2, provide the full results for the experiments in Figure 4 in Section E.3 and further investigate the optimization landscape for decoupled $\lambda _ { c }$ and $\lambda _ { r }$ in Section E.4, and for different values of ℓ and N in Section E.5.

## E.1 Implementation details and hyperparameters

Code availability. The complete codebase used to run all numerical simulations and generate the figures presented in this work can be found at https://github.com/anditp/ Single-Location-Regression.git. All the numerical experiments were run on a MacBook Pro M1 on CPU. Each individual experiment (20 runs) in the paper takes less than an hour to run.

Distance to the invariant manifold. Throughout our experiments, we track the optimization trajectories to determine if they remain on, or converge to, the invariant manifold M. Recall that M is defined by the strict orthogonality constraints:

$$
\begin{array} { r } { \mathcal { M } = \left\{ ( k , v ) \in \left( \mathbb { S } ^ { d - 1 } \right) ^ { 2 } ~ \middle | ~ k ^ { T } v ^ { * } = 0 , ~ v ^ { T } k ^ { * } = 0 , ~ k ^ { T } v = 0 \right\} . } \end{array}
$$

Consequently, for any parameter pair $\left( k , v \right) \in \left( \mathbb { S } ^ { d - 1 } \right) ^ { 2 }$ , we compute its distance to M continuously during training using the root sum of squares of the corresponding inner products:

$$
\begin{array} { r } { d _ { \mathcal { M } } ( k , v ) : = \sqrt { ( k ^ { T } v ^ { * } ) ^ { 2 } + ( v ^ { T } k ^ { * } ) ^ { 2 } + ( k ^ { T } v ) ^ { 2 } } . } \end{array}\tag{38}
$$

Hyperparameters. Table 1 summarizes the specific values and schedules used across all numerical experiments presented in Section 4 and Appendix E. For the penalized empirical risk experiments, we introduced the weighting coefficients $\mu _ { 1 }$ and $\mu _ { 2 }$ to enforce the structural constraints.

Table 1: Summary of hyperparameters used across the numerical experiments.
<table><tr><td>Parameter</td><td>Fig. 3 (M initialization)</td><td>Fig. 4, Fig. 13 (Constant λs)</td><td>Fig. 5 (Penalized)</td><td>Fig. 6 (Unpenalized)</td></tr><tr><td>Grid height (M)</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Grid width (N)</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Embedding dim. (d)</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td>Shift magnitude (l)</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Special token variance  $( \gamma ^ { 2 } )$ </td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Noise variance  $( \epsilon ^ { 2 } )$ </td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Penalty weight  $( \mu _ { 1 } )$ </td><td>0</td><td>0</td><td>10</td><td>0</td></tr><tr><td>Penalty weight (µ2)</td><td>0</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Learning rate (α)</td><td>1e-4</td><td>1e-4</td><td>1e-4 (a) / 1e-3 (b)</td><td>1e-4 (a) / 1e-3 (b)</td></tr><tr><td>Number of iterations</td><td>100,000</td><td>100,000</td><td>100,000</td><td>100,000</td></tr><tr><td>Batch size</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Independent runs</td><td>20</td><td>20 /λ value</td><td>20</td><td>20</td></tr></table>

The batches for Stochastic PGD are generated dynamically and resampled at each step, since we are aiming to mimic PGD on the theoretical risk.

## E.2 Optimization without Manifold Penalties

In Section 4, we demonstrated that optimizing the penalized empirical risk allows both the erf and softmax predictors to converge to the oracle parameters from generic random initializations. To isolate the effect of the penalty terms, we conduct an ablation study where both predictors are trained using the strictly unpenalized risk from generic initializations outside of M.

Figure 6 presents the optimization trajectories for both $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) }$ (erf predictor) and $T _ { \lambda _ { r } , \lambda _ { c } } ^ { ( k , v ) }$ ,softmax (softmax predictor) under these unconstrained conditions. Without the explicit geometric regularization provided by the penalties, the trajectories fail to always converge. For the erf predictor there are runs which get trapped in high loss regions. On the other hand, the optimization of the softmax predictor appears better behaved, with most of the runs converging in a similar manner as for the penalized version; however, there are some runs that get trapped in regions with high positive ν and high negative κ, far from M. This empirical finding underscores the practical necessity of the regularization terms when operating outside the theoretical guarantees of Theorem 5.

![](images/4c9d69040641309883266ee8d1c511f0fb1299cb25500bdc73e40cf09cf89b5c.jpg)

![](images/ed1d10b56b196c1cdd0e8b8e94eb06738df155948b6b97d016fdfb3405fa6794.jpg)

![](images/7a72cad78c4a8db598eb4996c2a261961cf5cca230222a7f338fda00b9c400e0.jpg)  
(a) Dynamics for the erf predictor.

![](images/52c952d7bab4bfe19ebc081aa4f26b9a02e2d4b48a95077c949389b91b8ed78c.jpg)

![](images/204da85b16959a021ff48fda8c1452dba36dea22e69a3d6fa40dae4a98f9f971.jpg)

![](images/9a19317472c7ec50bc7d223bd3eea3770d90814c8ecde5e23aa7c004659a0614.jpg)

![](images/36d6452d7f67a5a97aecc3bf01a9f0fd1b7b088d2f073111bd048d12fddf236e.jpg)  
(b) Dynamics for the softmax predictor.

![](images/67d655d59539b7c99c818b67898561674f99229866b01f51d8c3af8fc79d8bcf.jpg)

Figure 6: Optimization dynamics from 20 random initializations without the alignment penalties $( \mu _ { 1 } = \mu _ { 2 } = 0 )$ The solid curve represents the empirical median across the runs. The bands represent the 0.05-0.95 and, respectively, the 0.25-0.75 quantile intervals. The parameters used are $\bar { M } = N = 5 , d = 2 0 , \gamma = 1 , \ell = 1 0 0 \mathrm { ~ a n d ~ } \lambda _ { t } = 1 0 ^ { - 1 } / ( \bar { 1 } + \operatorname* { m a x } ( t - 5 \cdot 1 0 ^ { 4 } , 0 ) \cdot 1 0 ^ { - 4 } )$ , for the erf predictor and $\lambda _ { t } = 1 0 ^ { - 1 } / ( 1 + \operatorname* { m a x } ( t - 2 \cdot 1 0 ^ { 4 } , 0 ) \cdot \dot { 1 0 ^ { - 4 } } )$ for the softmax predictor. For computing the oracle loss we use the initial inverse temperatures.

## E.3 Full results for the experiment in Figure 4

In Section 4, we discussed the optimization dynamics under various constant inverse temperatures following an initial warm-up phase. Here, we present the full empirical results for this experiment.

As illustrated in Figures 7, 8, 9, 10, the training procedure begins with a 20,000 step warm up using $\lambda _ { c } = \lambda _ { r } = \mathbf { \bar { 0 } } . 1$ . After this period, the inverse temperatures are held constant at values $\lambda \in \mathbf { \bar { \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 \} } }$ . We observe that without a meticulously chosen lambda schedule, not all the runs converged in the simulated 100,000 steps. For lower values of λ, more runs got stuck around the origin compared to the more consistent convergence obtained for larger λs.

These full results empirically validate the need to employ a decreasing schedule for the inverse temperatures. A static λ forces a compromise between quickly escaping the saddle point at the origin and converging precisely at the global minima. A schedule dynamically adapts the landscape to first push the parameters away from the saddle point, and subsequently allows for fine-tuned convergence to the exact oracle parameters.

![](images/43014cf4bec07cfe82a9b5162841856fed08a76ea7407e30d2d75d89946853af.jpg)

![](images/10414eae770b70e5de95af4b9df2bf5cfed9d8a16d894085a06ae7404f0bee63.jpg)

![](images/ea42d878a8ef6ed865bbba261b0e4e3dae5681fcb1c1e9bed787c2b26822199f.jpg)

![](images/6ea5bfaaba2cc97f32f3dbf3c8b6c0ad301d781d255b378ebca8fb559b5e2e20.jpg)  
Figure 7: Training dynamics of the predictor with a 20k warm-up phase $\lambda _ { c } = \lambda _ { r } = \lambda = 0 .$ 1 followed by constant inverse temperatures $\bar { \lambda _ { c } } = \lambda _ { r } = 0 . 0 1$ . The solid curve represents the empirical median across 20 runs and the bands represent the 0.05-0.95 and, respectively, the $0 . 2 5 – 0 . 7 5$ quantile intervals (although not always visible). The parameters used are $M \dot { = } N = \dot { 5 } .$ , d = 20, γ = 1, ℓ = 100.

![](images/f4706f6dff5b4a610360c9f27846f0196f13807a1391f86b9f2d9b0d99b84db9.jpg)

![](images/cd5e86809ae22acceae659b9818e54ceea6bd48095492aa695b8e4182d70379a.jpg)

![](images/273c2ed475d0abc8940ebb4fc398f9e01d1699ff9baef2b27c281ff98cf32138.jpg)

![](images/e992c80c1b58bd9f76abd19844f3cd3039dfbe130d3468149cf0028325f1fdc3.jpg)  
Figure 8: Training dynamics of the predictor with a 20k warm-up phase $\lambda _ { c } = \lambda _ { r } = \lambda = 0 . 1$ followed by constant inverse temperatures $\lambda _ { c } = \lambda _ { r } = 0 . 0 3$ . The solid curve represents the empirical median across 20 runs and the bands represent the $0 . 0 5 { - } 0 . 9 5$ and, respectively, the 0.25-0.75 quantile intervals (although not always visible). The parameters used are $M \overset { \cdot } { = } N = \overset { \cdot } { 5 } , d = 2 0 , \gamma = \overset { \cdot } { 1 } , \ell = 1 0 0$

![](images/69285d2f26cba7ca970bafb4071dc41aa3e59ed5dfe05966109daf8d34489346.jpg)

![](images/d0aac75580eb993b66f0c9c07db70809b6c91ca4e96726b22a48d891789fad7e.jpg)

![](images/3cc62fd88a701ba572956034c0f23be9eb32cd3fc120686f1eead874d4050473.jpg)

![](images/ef05fa2444277287b3c2292381e2593cbf0bc16a3062918da343f7ff51ecea93.jpg)  
Figure 9: Training dynamics of the predictor with a 20k warm-up phase $\lambda _ { c } = \lambda _ { r } = \lambda = 0 . 1$ followed by constant inverse temperatures $\lambda _ { c } = \lambda _ { r } = 0 . 0 5$ . The solid curve represents the empirical median across 20 runs and the bands represent the $0 . 0 5 { - } 0 . 9 5$ and, respectively, the 0.25-0.75 quantile intervals (although not always visible). The parameters used are $M = N = 5$ , d = 20, γ = 1, ℓ = 100.

![](images/87be3772441f16c79481b305ae2f547e6e008bd009ee9096e0108b0002c3828d.jpg)

![](images/691798a8fe2c474f2bdf6d7584d87e38d62f2094d377dede46029fd16a59eb04.jpg)

![](images/bdac893f95b693c0ea54dff01030aded301fbdb70795ddd7688f6abad677cb2d.jpg)

![](images/9345b2be260ddc1e16634d2c071529c6ba57bd6064ab5fafd8880591115400c7.jpg)  
Figure 10: Training dynamics of the predictor with a 20k warm-up phase $\lambda _ { c } = \lambda _ { r } = \lambda = 0 . 1$ followed by constant inverse temperatures $\lambda _ { c } = \lambda _ { r } = 0 . 0 7$ . The solid curve represents the empirical median across 20 runs and the bands represent the 0.05-0.95 and, respectively, the 0.25-0.75 quantile intervals (although not always visible). The parameters used are $M = N = 5 , d = 2 0 , \gamma = 1$ $\ell = 1 0 0$

## E.4 Gradient analysis for decoupled inverse temperatures

This subsection isolates the effects of decoupling the two inverse temperatures. We visualize the gradient vector field of the reduced risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ evaluated on the invariant manifold $\mathcal { M }$ for multiple configurations of $( \lambda _ { r } , \lambda _ { c } )$

Figures 11, 12 present the numerically computed gradients for various mixed combinations of $\lambda _ { c }$ and $\lambda _ { r } . \mathrm { \bf ~ B y }$ breaking the symmetry, we observe how the magnitude and direction of the gradients shift across the landscape. Specifically, two interesting phenomena appear:

$\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { i } } ^ { < }$ behaves symmetrically with respect to both $\lambda _ { r }$ and $\lambda _ { c } ;$

• The same is not true for $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ . The shape of the gradient and most importantly of the 0 level curve of $\partial _ { \kappa } \mathcal { R } ^ { < }$ varies significantly with the pairs $( \lambda _ { r } , \lambda _ { c } )$

This analysis leads to another method of controlling the training dynamics, namely through the ratio $\tau = \lambda _ { c } / \lambda _ { r }$ . Indeed, Figure 13 presents the optimization dynamics obtained for constant $\lambda _ { c } = 0 . 0 1$ and $\lambda _ { r } = 0 . 1$ . While 100,000 steps are not enough for convergence at the oracle parameters, the runs consistently move towards convergence. Moreover, the training dynamics are different from the ones studied in the main paper, as can be seen from the much faster increase of $| \kappa |$ , closely mirroring the behavior of $| \nu |$

K  
![](images/4674cc5b417f8d9d3cf30cdb3dbea47aa293b67071cea21ed14acd8b1f1cb565.jpg)  
K

![](images/2f7a9fdbe6fca7c7a3010c81df851f3c723d2db72b518045e0fe87cccda16efc.jpg)  
K

![](images/bbc60c89d36555425ac56cffe72eac9b8372ce2c8c5987d2c26ed19daf7d7865.jpg)  
K

![](images/b6369d9819fc1831afd28a9fb5d08d65ab6e11cd2e52b35d8f2211a05fb5362a.jpg)  
K

(a) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for λ<sub>r</sub> = 0.1 and $\lambda _ { c } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.  
![](images/47386380423d28ff75b07837fbecccc250e2ed56ddcb27e19e7f092a74b796e7.jpg)

λr = 0.1, λc = 0.01 λr = 0.1, λc = 0.03 λr = 0.1, λc = 0.05 λr = 0.1, λc = 0.07 $\lambda _ { r } = 0 . 1 , \lambda _ { c } = 0 . 1$  
![](images/cd2c5f44cf9a53e708c8d773b7e5c5fee7e50797395fc6709163ab073336262d.jpg)  
K

![](images/dc0d3ad441e8a9e8ac50b6961e8e69cab3b8c4ed82e7dfe379fd6cd3305d0478.jpg)  
K

![](images/22e84061d6918628643b492c15adcb3e473ac747aad0f94c8bebea2e9864f5b8.jpg)

![](images/a362ec9da668b96de4429b10ea64d888a61c69039bbf60286835155792ad8c09.jpg)  
K

![](images/8fe388f4faacda1ff55d679158860ce61321db5dcce7c022b4db69f6a6097e00.jpg)  
K  
(b) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = 0 . 1$ and $\lambda _ { c } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.  
K

![](images/5e9450d6f25be5a07ed6ef7e68d3297027f1ead610b0a17bc610076d64778a92.jpg)  
K

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 3
$$

![](images/3d3a259f185dbcb701e29674397884376126bb6f94bf710a1e3e0462908dc05b.jpg)  
K

![](images/d359f6909d25f9786ca87effc70a3fc2f2f82bc78c15bb5225138c16a7e2c8fe.jpg)  
K

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 7
$$

![](images/8f5487a8e0cbb9296e13c13a9949ee941e6bf385668691440d2e824b70d33597.jpg)  
K

![](images/a4b54b93c15063f30e86c8ae6af0d86d85c6abec84126a9fad463453fcbc31d9.jpg)  
K  
(c) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = 0 . 0 1$ and λ<sub>c</sub> ∈ {0.01, 0.03, 0.05, 0.07, 0.1}.

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 1
$$

![](images/b8beb2f663aa4e206a0c51ae4e5e5767528691860521025d45834ba15410ca79.jpg)  
K

![](images/ba84f1da04b7612b15d75cceae29b74e96fdc9da6948d8098ed956d88f934aaf.jpg)  
K

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 5
$$

![](images/648d5bc48139f0344baf55e8b4325ef2af51a64b72f2fc5bd0543d2616bacad5.jpg)  
K

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 7
$$

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 1
$$

![](images/a7bce99daa1489e23adf0faac651820ad66b159c3c5c4f4c0f9d37cfc245a9f7.jpg)  
K

![](images/63d28f9787c204d0181d45713dd2ef972ee4604ad099e8c7011e72d0140c5632.jpg)  
(d) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = 0 . 0 1$ and $\lambda _ { c } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.

Figure 11: Numerically computed gradients of the theoretical risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { r } \in \{ 0 . 1 , 0 . 0 1 \}$ and varying values of $\lambda _ { c } .$ . The parameters used are $M = N = 5$ and $\ell = 1 0 0$ . The gradients, given by Equation (32), are approximated using the Monte Carlo method with a sample size of $1 0 { , } 0 0 0$ on a $1 0 0 \times 1 0 0$ discretization grid.

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 1
$$

![](images/a85b1ce2af74a2c429b22333f57ff5d38b1033767de6f36ae9796c785078137b.jpg)  
K

$$
\lambda _ { r } = 0 . 0 3 , \lambda _ { c } = 0 . 1
$$

![](images/f525fdc985f7cf1f993e18c8333d413c4aa4cacf157d1fbdc7b72af6ee0ab46b.jpg)  
K

$$
\lambda _ { r } = 0 . 0 5 , \lambda _ { c } = 0 . 1
$$

![](images/a15daaf121f421a79f4e3ec7931744ff71b5b034373eb920e3060f53e8d340b8.jpg)  
K

![](images/17c59b7ee791831e62533e41f5a355f8bf87b24142833f25e26acbf3357a2cb2.jpg)  
K

![](images/8b0a25351d4b39c9daa694b8a20b90b1a81a88fa00b4d004db49bb0e2df2f312.jpg)  
K  
(a) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { c } = 0 . 1$ and $\lambda _ { r } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.

![](images/8f0e24276f723a844228292c5aea294bf178eceeddc35529e58f6141ee5b3b2d.jpg)

$$
\lambda _ { r } = 0 . 0 3 , \lambda _ { c } = 0 . 1
$$

![](images/c2a161411b1ff4ccff70d8481b996a4f4981ae39b8a31571f6d708b73d89a916.jpg)

$$
\lambda _ { r } = 0 . 0 5 , \lambda _ { c } = 0 . 1
$$

K  
K  
![](images/5b77fcf4875042862eef3af9cd45eab5eb9700d3fa9a8233d63cc9586f455b57.jpg)

$$
\lambda _ { r } = 0 . 0 7 , \lambda _ { c } = 0 . 1
$$

K  
![](images/c7aa35823b47d198bacfa218e46ee95be2d7513982170653d0c02dfca99961a4.jpg)

$$
\lambda _ { r } = 0 . 1 , \lambda _ { c } = 0 . 1
$$

K  
![](images/ad39878c5f85c48f329b4a6db1699bd99009a7d9443609401cdd952c93219c32.jpg)  
K  
(b) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { c } = 0 . 1$ and $\lambda _ { r } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 1
$$

![](images/5f44ee0d12f9ab1d6f9eaeba648093b87fadb1bedffc6f47d9226174a1d7a865.jpg)  
K

$$
\lambda _ { r } = 0 . 0 3 , \lambda _ { c } = 0 . 0 1
$$

![](images/7e02d38c7e8b8d468b2594c84f1f967c42fcc1df83475d1a679c9bce9ea6d37e.jpg)  
K

$$
\lambda _ { r } = 0 . 0 5 , \lambda _ { c } = 0 . 0 1
$$

![](images/c9702aabfe440a7bbcd3e0f2d1ec700e98e63490cffa91b5c95adc1a69c240ff.jpg)  
K

$$
\lambda _ { r } = 0 . 0 7 , \lambda _ { c } = 0 . 0 1
$$

![](images/24a22b7c64dde825adf2b1d4c59d9e34893cbc42d8324fe7b9639a24a468616f.jpg)  
K

![](images/3d4188471f16793ec5f2369a794571a1976a6cf7136fbc8d0aab8dfa9f03690e.jpg)  
K  
(c) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { c } = 0 . 0 1$ and $\lambda _ { r } \in$ {0.01, 0.03, 0.05, 0.07, 0.1}.

$$
\lambda _ { r } = 0 . 0 1 , \lambda _ { c } = 0 . 0 1
$$

![](images/73c43955bde21d891d647d4046de6508ffdd696e0bf68e4d142c680400b7791a.jpg)  
K

$$
\lambda _ { r } = 0 . 0 3 , \lambda _ { c } = 0 . 0 1
$$

![](images/47fab8e873664d7efc61bcf8bf65b784058780f7e9ee7fb0b69d297fcef16f5a.jpg)  
K

$$
\lambda _ { r } = 0 . 0 5 , \lambda _ { c } = 0 . 0 1
$$

![](images/5bc04832a4992035402a6d7f5f3909bef97f8d8d6899f10b55f6cb70f48b8f83.jpg)  
K

$$
\lambda _ { r } = 0 . 0 7 , \lambda _ { c } = 0 . 0 1
$$

![](images/db6eb874bad0ca986235e811d3fdcd6a5cd3d6394f59518fb2a754ef9e68e473.jpg)  
K

![](images/226c054c29791da6fb74456c2fb2c8d41ee86696a4bd9db65896d3d9e26f9e06.jpg)  
(d) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { c } = 0 . 0 1$ and λ<sub>r</sub> ∈ {0.01, 0.03, 0.05, 0.07, 0.1}.

Figure 12: Numerically computed gradients of the theoretical risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { c } \in \{ 0 . 1 , 0 . 0 1 \}$ and varying values of $\lambda _ { r } .$ The parameters used are $M = N = 5 { \mathrm { ~ a n d } } { \dot { \ell } } = 1 0 0$ . The gradients, given by Equation (32), are approximated using the Monte Carlo method with a sample size of $1 0 { , } 0 0 0$ on a $1 0 0 \times 1 0 0$ discretization grid.

![](images/ca37c279463707a81859a7aa01e94b0e5ce1f4645345793633ad688cbe1f2f7e.jpg)

![](images/988fcfc3a635fe55da830890b7b84459a18aa9b6733884bee8f667c41e4e63af.jpg)

![](images/7f3d125640c99773bb84ee766302b8ca7b2e56d02cb165b4b62df737c02fae19.jpg)  
Figure 13: Training dynamics of the predictor with constant $\lambda _ { c } = 0 . 0 1$ and $\lambda _ { r } = 0 . 1$ . The solid curve represents the empirical median across the runs and the bands represent the 0.05-0.95 and, respectively, the 0.25-0.75 quantile intervals (although not always visible). The parameters used are $M = N = 5 , d = 2 0 , \gamma = 1 , \ell = 1 0 0 .$

## E.5 Gradient analysis for other hyperparameters N and ℓ

In this section, we investigate the behavior of the partial derivatives of $\mathcal { R } ^ { < }$ when varying N and ℓ, for fixed $\lambda _ { r } = \lambda _ { c } = 0 . 1$

![](images/c693830d19937b2f7233c6bbf01a2e314f39944f46c1bd4369e8c4eaedd8ad90.jpg)

![](images/52f6131a612c7baecfa16269ea7a341467ae1d8c3f32ca75f9ffc5e2d0cb4725.jpg)

![](images/2b80b393f6d63efe5947c29b7c2fca6fd31468aa9c72941739e5f13b2c05187e.jpg)  
(a) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for λ<sub>r</sub> = λ<sub>c</sub> = 0.1, ℓ = 10 and N ∈ {5, 50, 500}.

![](images/6a85e7a112f184e851ef53d421d817fde463e435e160ccf41b13a4f13a9d64cf.jpg)

![](images/6c2b08e46265d3e330ddf43e4bcd165bc9d80acbf435fbbbd47f1088ea54fb2c.jpg)

![](images/cd679a2e0613929bc5ec4acf38053d5726ba77cd13257e5be6c7aeb22cc548d6.jpg)  
(b) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1 , \ell = 1 0$ and $N \in \{ 5 , 5 0 , 5 0 0 \}$

Figure 14: Numerically computed gradients of the theoretical risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1$ $\ell = 1 0$ and varying values of N. The gradients, given by Equation $( 3 2 ) ,$ are approximated using the Monte Carlo method with a sample size of $1 0 { , } 0 0 0$ on a $\mathrm { { 1 0 0 } ^ { - } \times 1 0 0 }$ discretization grid.

The values considered are not following the asymptotics required for Bayes optimality of the predictor, but better reflect real-world scenarios in which TSFMs are trained to take tens of channels and hundreds of time patches as input, while an individual token’s shift in the mean cannot be too large

due to normalization layers. We restrict our analysis to exploring the gradient across multiple pairs of values (N, ℓ), as conducting full trainings for each pair would be computationally expensive.

![](images/5bf5e6c86c434f191f2c2d6bfb3c0b2326d53d3af3d534efc0017bcee375d469.jpg)

![](images/232a214a9bc7733e2e97b86e8af4119a0ab78a97ce8b6a4d668a5906fa10ce55.jpg)

![](images/832503ea5eb203d608d29be84493642718849b1e6b54ea0c5df9f06b52fc5434.jpg)  
(a) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for λ<sub>r</sub> = λ<sub>c</sub> = 0.1, ℓ = 100 and N ∈ {5, 50, 500}.

![](images/8a9ca35b9f538f1da7ff81ea4c949ce178bc8047f44a765c6a404c903b6a88aa.jpg)

![](images/21c9e77acb63a766234a752858c2be7cd0358a2e01fc582a0984a0ca4ae35c0c.jpg)

![](images/f18dceb3696fe99b5caa28203d59278d879189846ad53ad9f3992ae0565eeb3e.jpg)  
(b) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1 , \ell = 1 0 0$ and $N \in \{ 5 , 5 0 , 5 0 0 \}$

Figure 15: Numerically computed gradients of the theoretical risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1$ $\ell = 1 0 0$ and varying values of N. The gradients, given by Equation $( 3 2 )$ , are approximated using the Monte Carlo method with a sample size of 10,000 on a $\mathrm { i 0 0 } \times \mathrm { 1 0 0 }$ discretization grid.

As can be seen in Figures 14, 15, 16, the effect of ℓ overshadows that of N. Although for small ℓ, a higher N appears to eliminate the problem of escaping the origin for κ, this disappears for larger ℓ values.

Furthermore, the higher ℓ is, the higher the values of $\partial _ { \nu } \mathcal { R } ^ { < }$ and the steeper its sign change around $\kappa = 0$

![](images/e5aa5522677c77c6ba5fe0aebd5fccd4685b704847ddc4f2e409ee563aa4f75e.jpg)

![](images/5558abc5b3583b99bb20070968ab21cbcd0188ee000199cefefa0799c6d3da15.jpg)

![](images/5f2828c5dcc754c0fef11dcb8fd8c8377968f0ce0af50ec4d2602edc4674e0f6.jpg)  
(a) Partial derivative $\partial _ { \kappa } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for λ<sub>r</sub> = λ<sub>c</sub> = 0.1, ℓ = 1000 and $N \in \{ 5 , 5 0 , 5 0 0 \}$

![](images/83152310227d5dfbde10181163e0958307e8eda87300e4de3d08d357712a703b.jpg)

![](images/4786c8db1af53741f8f7aec21791a4c31a9d2ec17d9b35d0d6809f465d29ad26.jpg)

![](images/82b280e2a13cb53b985af50e6a7a136ed68dac98afbaf55e5c58127ca1504cf9.jpg)  
(b) Partial derivative $\partial _ { \nu } \mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < }$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1 , \ell = 1 0 0 0$ and $N \in \{ 5 , 5 0 , 5 0 0 \}$

Figure 16: Numerically computed gradients of the theoretical risk $\mathcal { R } _ { \lambda _ { r } , \lambda _ { c } } ^ { < } ( \kappa , \nu )$ for $\lambda _ { r } = \lambda _ { c } = 0 . 1$ $\ell = 1 0 0 0$ and varying values of N. The gradients, given by Equation (32), are approximated using the Monte Carlo method with a sample size of 10,000 on a $1 0 \bar { 0 } \times 1 0 0$ discretization grid.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We provide proofs for all the theoretical claims made and extensive numerical experiments both in the main text and in the supplemental material for all the empirically observed claims.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: While not having a separate "Limitations" section in the paper, we discuss limitations of our work in the "Conclusions and limitations" section.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All the theoretical results and claims are accompanied by full proofs in the supplemental material.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide the full code used for the numerical experiments in the paper.   
Furthermore, we provide the complete set of parameters for each experiment in the paper.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code is given as a resource in the supplemental materials.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The hyperparameters used in all the experiments are listed in the supplemental material.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Empirical distributions are always plotted alongside the main curves.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Details regarding the compute resource are given in Appendix E.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: All the experiments are conducted using synthetically generated data. The authors don’t see any possible harm arising from the present research.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: The work in this paper is purely theory-oriented, aiming to better understand existing architectures. The authors see no societal impact related to the present paper.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: The authors don’t see any risk in the models used for this research.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [N/A]

Justification: No existing asset is used.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The only asset released is code for the experiments in the main paper. The code is well documented in the opinion of the authors.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs were used only for writing, editing, or formatting purposes.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.