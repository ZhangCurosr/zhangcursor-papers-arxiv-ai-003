# Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity

Sahil Joshi<sup>∗†</sup>

Prithvi Dixit<sup>∗‡</sup>

Aryan Katiyar<sup>§</sup>

Agniva Chowdhury<sup>†</sup> Joseph E. Gonzalez<sup>‡</sup> Anshumali Shrivastava<sup>†</sup>

Aditya Desai<sup>¶</sup> <sup>‖</sup> Kumar Krishna Agrawal<sup>¶</sup> <sup>‡</sup>

## Abstract

Large Language Model inference over long contexts is increasingly bottlenecked by the cost of attending to a growing Key-Value (KV) cache. Sparse attention reduces this cost by accessing only a fraction of the cache, but existing methods face a trade-of between deterministic retrieval, which captures high-scoring tokens but discards residual attention mass, and sampling, which improves coverage at the cost of higher variance. Hybrid methods combine top-k retrieval with uniform sampling, yet still exhibit a substantial gap from dense attention at aggressive sparsity. We introduce Stratified Sparse Attention (SSA), a unified framework combining deterministic retrieva with score-guided sampling. SSA deterministically attends to high-scoring tokens while sampling the residual using proxy attention scores and correcting for their inclusion probabilities, thereby recovering attention mass beyond the retrieval set. SSA encompasses top-k, top-p, and hybrid top-k+sampling as special cases. Across RULER, LongBench, LOFT, and AIME, and five models ranging from 3B to 27B parameters, SSA consistently improves over the baselines at matched sparsity. On RULER-HARD with Ministral-3-8B, SSA improves accuracy over the strongest practical baseline from 81.8% to 84.2% at 32K and from 63.8% to 69.3% at 128K under 50× attention sparsity, recovering 94% and 85% of dense-attention performance, respectively. With optimized CUDA kernels, SSA achieves up to 2.96× decode speedup over FlashInfer at 1M context, demonstrating that its accuracy gains translate into practical long-context inference acceleration. We open-source our code at https://github.com/skylight-org/SSA-Paper-Code.

## 1 Introduction

Large language models (LLMs) are rapidly expanding into long-context workloads, spanning large document collections [23], image analysis [10, 31], software engineering [15, 5], databases [48], video processing [33], and long-horizon agents that continuously accumulate observations, tool outputs, and interaction histories [3, 4]. As context lengths approach millions of tokens, eficient inference becomes increasingly challenging. During autoregressive decoding, each newly generated token attends to an ever-growing key-value (KV) cache, requiring keys and values to be repeatedly transferred from GPU high-bandwidth memory (HBM) onto on-chip SRAM for attention computation [35, 20]. At long contexts, this makes decoding increasingly memory-bandwidth bound: the attention cost per decoding step grows linearly with context length. Sparse attention addresses this bottleneck by attending to only a small fraction of the KV cache [39, 27, 46], reducing both memory trafic and attention latency. However, realizing these gains in practice requires sparse attention to satisfy two competing objectives: faithfully approximate dense attention at aggressive sparsity while remaining eficient enough to translate sparsity into end-to-end inference speedups [29, 25].

![](images/c348743b836a8ec50580691c1bf4bff1569bc975885bffec642692ef2f7ea18a.jpg)

![](images/613a4217b76e683aa45707708b8d818040a87f8f571d5215342021b97a6f5e50.jpg)  
Figure 1: Relative $L _ { 2 }$ error when varying the top-k:sampling budget allocation at 50× sparsity (left) and varying sparsity with a fixed 3:2 top-k:sampling budget split (right).

Existing sparse-attention methods can be broadly categorized by how they allocate a limited attention budget. First, approximate retrieval methods use proxy attention scores to deterministically select high-scoring tokens, typically through top-k retrieval. Representative methods include PQCache [46], and Quest [39], with recent approaches such as Tactic [50] extending this paradigm to top-p selection. While efective when attention is highly concentrated, deterministic retrieval can miss substantial attention mass when the distribution is difuse or long-tailed [6, 9]. Second, sampling-based methods such as MagicPig [6] use importance sampling to approximate dense attention, providing broader coverage but introducing sampling variance that can lead to a larger gap from dense attention at limited budgets (see Table 5 in Appendix). Finally, hybrid methods such as vAttention [9] combine deterministic top-k retrieval with uniform sampling [42], preserving high-confidence tokens while extending coverage beyond the top-k set. Although this combination improves the trade-of between exploitation and coverage, a substantial gap to dense attention remains at aggressive sparsity (see Table 5 in Appendix). See Section 2 for more details. These observations motivate a fundamental question: Can we design a more efective sampling strategy that substantially narrows the sparse-to-dense accuracy gap?

Our preliminary studies reveal a key limitation of uniform sampling: while it provides coverage beyond the deterministic top-k set, it ignores the highly non-uniform distribution of residual attention mass<sup>1</sup>. As shown in fig. 1, score-guided importance sampling (the principle that SSA uses) consistently achieves lower relative $L _ { 2 }$ error than uniform sampling. Motivated by this observation, we introduce Stratified Sparse Attention (SSA), a unified framework that combines deterministic retrieval of high-confidence heavy hitters with score-guided stochastic exploration of the remaining context. Rather than treating all residual tokens equally, SSA uses proxy attention scores to preferentially explore promising regions while preserving stochastic coverage of the long tail. This enables the sparse attention budget to better capture the attention mass that would otherwise be lost under aggressive sparsity. We evaluate SSA across diverse models, including Llama-3.1-8B-Instruct [12],

![](images/458c500b8f1956318ea7971607e23082a835ed4ce9881040c03c4424a8f4b82e.jpg)  
(a) Qwen3.5-4B: Error

![](images/f5e7aef1a2c97cc9b41611fab6f4791acfcf85fb2c2dd617338076fd77c76289.jpg)  
(b) Qwen3.5-9B: Error  
Figure 2: Average attention error across diferent context lengths and sparsity levels for Qwen3.5-9B and Qwen3.5-4B on LOFT [22].

Qwen3.5-4B, Qwen3.5-9B, Ministral-3-3B-Instruct-2517, and Ministral-3-8B-Instruct-2517 [26], and across RULER [16], LongBench [2], LOFT [22], and AIME [28]. Across these evaluations, SSA consistently improves over deterministic, sampling-based, and hybrid sparse-attention baselines, substantially narrowing the gap to dense attention at extreme sparsity. Our contributions can be summarized as follows:

I. Unified Attention Paradigm: We introduce a unified sparse-attention framework that partitions tokens into deterministic and stochastic strata using a generic proxy score. By varying how tokens are retained or sampled, the framework recovers top-k, top-p, and hybrid top-k+sampling as special cases of importance-sampling-inspired sparse attention.

II. Theoretical Guarantees: We provide rigorous bounds for our proposed algorithm, characterizing how closely it approximates dense attention under a limited attention budget.

III. Eficient Kernel Support: We develop eficient CUDA/Triton kernels that realize the computational benefits of our approach in practice, delivering end-to-end decoding speedups over dense attention.

## 2 Related Work

In the context of SSA, the extensive prior work can be organized into three broad and comprehensive research directions:

Head-level compression and KV eviction: One line of work reduces KV-cache overhead by sharing or compressing KV heads. Multi-Query Attention (MQA) [37] shares a single KV head across all query heads, while Grouped-Query Attention (GQA) [1] and Multi-Head Latent Attention (MLA) [7] provide more flexible compression to balance eficiency and model quality. A complementary line of work dynamically evicts less important KV entries during inference. Methods such as SnapKV [24], DynamicKV [49], KVZip [19], KeyDif [30], CurDKV [34], and CapKV [44] use diferent importance signals and eviction policies to reduce cache size and subsequent attention computation. These approaches are complementary to sparse attention: KV compression reduces the tokens or representations retained across decoding steps, whereas sparse attention dynamically selects which retained tokens to access for each query. The two can therefore be composed to further reduce memory and decoding costs.

Approximate top-k/top-p methods: A substantial body of sparse-attention work focuses on find ing the tokens with the largest attention scores. Methods such as Quest [39], PQCache [46], SqueezedAttention [14], Adamas [43], SOCKET [18], Sketch-and-Walk [21], RetrievalAttention [27], Loki [38], HashAttention [8], and Double Sparsity [45] employ diferent proxy scores or retrieval mechanisms to approximate the top-k tokens under dense attention. These methods are efective at recovering high-attention “heavy hitters,” but a fixed top-k selection discards all tokens outside the retrieval set. Consequently, when non-negligible attention mass is distributed across the residual tail, or when approximation error causes relevant tokens to fall just below the top-k boundary, this discarded mass can create a substantial gap from dense attention at aggressive sparsity. Tactic [50] extends this retrieval paradigm from top-k to top-p selection by modeling the decay of attention scores with a power-law distribution and estimating the number of highest-scoring tokens required to satisfy a target attentionmass threshold.

Sampling-based and Hybrid methods: MagicPig [6] addresses the limitations of deterministic top-k attention by using LSH [17] to approximately sample tokens and importance correction to estimate dense attention. vAttention [9] combines deterministic and stochastic selection through a verified hybrid estimator that retains top-k tokens while uniformly sampling from the residual KV cache. KNN-Attention [13] also explores top-k plus sampling via lazy Gumbel sampling, but primarily targets training rather than inference acceleration for existing models. Unlike uniform hybrid sampling, which treats all residual tokens equally, SSA uses approximate attention scores to guide stochastic selection toward tokens more likely to carry residual attention mass (see fig. 1).

## 3 Background and Motivation

## 3.1 Sparse Attention

We begin by describing the dense attention computation. For clarity, we consider a single query $\mathbf { q } \in \mathbb { R } ^ { d }$ attending to a KV cache $\mathbf { K } : = ( \mathbf { k } _ { 1 } , \ldots , \mathbf { k } _ { N } ) ^ { \top } \in \mathbb { R } ^ { N \times d }$ and $\mathbf { V } : = ( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { N } ) ^ { \top } \in \mathbb { R } ^ { N \times d }$

Full Scaled Dot-Product Attention (SDPA): Let $s _ { i } = \mathbf { q } ^ { \top } \mathbf { k } _ { i } / \sqrt { d }$ be the similarity score between q and $\mathbf { k } _ { i } .$ , and $\mathbf { a } = ( a _ { 1 } , \ldots , a _ { N } )$ with $a _ { i } = w _ { i } / \sum _ { j = 1 } ^ { N } w _ { j }$ be the corresponding attention weights where $w _ { i } = \exp ( s _ { i } )$ . We write SDPA output $\mathbf { o } \in \mathbb { R } ^ { d }$ as

$$
\mathbf { o } = { \frac { \sum _ { i = 1 } ^ { N } w _ { i } \mathbf { v } _ { i } } { \sum _ { j = 1 } ^ { N } w _ { j } } } = \sum _ { i = 1 } ^ { N } a _ { i } \mathbf { v } _ { i } = \mathbf { V } ^ { \mathsf { T } } \mathbf { a } .\tag{1}
$$

Sparse Attention with deterministic top-k Selection: Let heavy set $\mathcal { H } \subseteq [ N ]$ denote a deterministic top-k subset from the N cached tokens. Sparse attention over the set H is defined as $\begin{array} { r } { \mathbf { o } _ { \mathcal { H } } = \frac { \sum _ { i \in \mathcal { H } } w _ { i } \mathbf { v } _ { i } } { \sum _ { j \in \mathcal { H } } w _ { j } } } \end{array}$ . While deterministic top-k selection concentrates computation on tokens deemed most relevant, tokens outside H do not contribute to the sparse output.

Sparse Attention with Importance Sampling: Let sampled set ${ \mathcal { S } } \subseteq [ N ]$ be a non-empty random set of selected tokens. For each token i, let $Z _ { i } = \mathbf { 1 } \{ i \in S \}$ indicate whether it is selected, and let $\pi _ { i } = \operatorname* { P r } ( i \in \mathcal { S } ) > 0$ be its inclusion probability. Importance-based sparse attention estimates the dense output by weighting each selected token by the inverse of its inclusion probability:

$$
\widehat { \mathbf { o } } = \frac { \sum _ { i = 1 } ^ { N } ( Z _ { i } / \pi _ { i } ) w _ { i } \mathbf { v } _ { i } } { \sum _ { i = 1 } ^ { N } ( Z _ { i } / \pi _ { i } ) w _ { i } } = \frac { \sum _ { i \in \mathcal { S } } ( w _ { i } / \pi _ { i } ) \mathbf { v } _ { i } } { \sum _ { i \in \mathcal { S } } w _ { i } / \pi _ { i } } .\tag{2}
$$

The factors $1 / \pi _ { i }$ account for diferences in how likely tokens are to be selected. The estimator in eq. 2 is an instance of inverse-probability weighting. We next recall the standard draw-based formulation of importance sampling.

## 3.2 Importance Sampling

Importance sampling [41] is a Monte Carlo technique for estimating a quantity using samples drawn from a proposal distribution. Given a target distribution $p ( x )$ and a function $f ( x )$ , consider the expectation $\begin{array} { r } { \mu = \mathbb { E } _ { x \sim p } [ f ( x ) ] = \sum _ { x } p ( x ) f ( x ) } \end{array}$ . Instead of sampling directly from $p ( x )$ , importance sampling draws samples $x _ { 1 } , \ldots , x _ { m } \sim q ( x )$ from a proposal distribution $q ( x )$ and rewrites the expectation as $\begin{array} { r } { \mu = \mathbb { E } _ { x \sim q } \left[ \frac { p ( x ) } { q ( x ) } f ( x ) \right] } \end{array}$ . This yields the Monte Carlo estimator

$$
{ \hat { \mu } } = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } { \frac { p ( x _ { j } ) } { q ( x _ { j } ) } } f ( x _ { j } ) .
$$

The ratio $p ( x ) / q ( x )$ is the importance weight, which corrects for sampling from q rather than p. When q places greater probability on samples with large contributions to the expectation, importance sampling can achieve lower variance than uniform sampling with the same sample budget.

## 4 Introducing Stratified Sparse Attention (SSA)

As already mentioned, deterministic top-k attention [11, 47] can omit meaningful attention mass outside the selected set. Uniform sampling on residual tokens provides broader coverage, but does not use information about which remaining tokens are more likely to matter. We introduce Stratified Sparse Attention (SSA), a unified framework that retains tokens with high proxy scores and samples the residual tokens according to those scores. SSA divides a fixed attention budget between two well-known complementary objectives: exploitation, by deterministically retaining high-confidence heavy hitters, and exploration, by sampling the residual context according to proxy attention scores. This score-aware stratification preserves heavy hitters while providing meaningful stochastic coverage toward more promising residual tokens. We next formalize the SSA mechanism described Algorithm 1.

## 4.1 Final Algorithm

SSA operates in three stages that divide the sparse-attention budget k between deterministic exploitation and stochastic exploration.

(i) Proxy-based budget allocation (Steps 1-4): SSA converts the proxy scores $\tilde { s } _ { i }$ into probabilities $q _ { i }$ and finds λ such that $\textstyle \sum _ { i = 1 } ^ { N }$ min $\{ 1 , \lambda q _ { i } \} = k$ . Tokens satisfying $\lambda q _ { i } \geq 1$ form the deterministically retained heavy set H. The remaining $m = k - | \mathcal { H } |$ is allocated to importance sampling from the residual set $\mathcal { R } = [ N ] \backslash \mathcal { H }$ . Thus, the proxy distribution determines the split between deterministic allocation and residual sampling automatically; SSA does not require a separate choice of heavy-set size.

(ii) Score-guided residual sampling (Steps 5-10): SSA normalizes the proxy probabilities over R to obtain proposal probabilities $\begin{array} { r } { p _ { i } = q _ { i } / \sum _ { j \in \mathcal { R } } q _ { j } } \end{array}$ , makes m multinomial draws with replacement, and retains the unique keys in S. Let ${ \mathcal { K } } = \mathcal { H } \cup \mathcal { S }$ , and define $Z _ { i } = \mathbf { 1 } \{ i \in \mathcal { K } \}$ to indicate whether token i is selected. Its inclusion probability is $\pi _ { i } = \operatorname* { P r } ( Z _ { i } = 1 )$ . For $i \in { \mathcal { H } } , \pi _ { i } = 1$ . For $i \in \mathcal { R }$ , the token is selected if it appears in at least one of the m draws, giving $\pi _ { i } = 1 - ( 1 - p _ { i } ) ^ { m }$

(iii) Exact scoring and correction (Steps 11-15): For each token in $\kappa ,$ , SSA computes its exact score $\ell _ { i } = \mathbf { q } ^ { \top } \mathbf { k } _ { i } / \sqrt { d }$ and corrected weight $\hat { w } _ { i } = \exp ( \ell _ { i } ) / { \pi } _ { i }$ . It returns the normalized weighted sum ${ \hat { \mathbf { o } } } .$ Since repeated draws are retained only once, $| { \boldsymbol { \mathcal { K } } } | \leq k$

```latex
Algorithm 1 Stratified Sparse Attention (SSA)
1: Input: Proxy scores $\{ \tilde { s } _ { i } \} _ { i = 1 } ^ { N }$ , query q, keys
$\{ \mathbf { k } _ { i } \} _ { i = 1 } ^ { N }$ , values $\{ { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { N }$ , total budget k
2: $\begin{array} { r } { q _ { i } \gets \frac { \exp ( \tilde { s } _ { i } ) } { \sum _ { j = 1 } ^ { N } \exp ( \tilde { s } _ { j } ) } , } \end{array}$
3: Find $\begin{array} { r } { \lambda > 0 \mathrm { ~ s . t . ~ } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } ( 1 , \lambda q _ { i } ) = k } \end{array}$
4: Define heavy set $\mathcal { H } = \{ i : \lambda q _ { i } \geq 1 \}$ , residual
set ${ \mathcal { R } } = [ N ] \backslash { \mathcal { H } } ,$ heavy size $h = | { \mathcal { H } } |$ , sampling
budget $m = k - h$
5: Normalize the proposal over $\textstyle { \mathcal { R } } : p _ { i } = { \frac { q _ { i } } { \sum _ { j \in { \mathcal { R } } } q _ { j } } }$
for $i \in \mathcal { R }$
6: Draw with replacement: $J _ { 1 } , J _ { 2 } , \ldots J _ { m } \sim \mathbf { p }$
7: $\mathcal { S } \gets \mathrm { u n i q u e } \{ J _ { 1 } , J _ { 2 } , \dots , J _ { m } \}$
8: $\pi _ { i }  1 - ( 1 - p _ { i } ) ^ { m }$ for $i \in S$
9: ${ \mathcal { K } }  { \mathcal { H } } \cup { \mathcal { S } }$
10: $\pi _ { i } \gets 1$ for $i \in \mathcal { H }$
11: for $i \in \mathcal { K }$ do
12: $\ell _ { i } \gets \mathbf { q } ^ { \top } \mathbf { k } _ { i } / \sqrt { d }$
13: ${ \hat { w } } _ { i } \gets \exp ( \ell _ { i } ) / \pi _ { i }$
14: end for
15: return $\hat { \mathbf { o } } \gets \frac { \sum _ { i \in \mathcal { K } } \hat { w } _ { i } \mathbf { v } _ { i } } { \sum _ { i \in \mathcal { K } } \hat { w } _ { i } }$
```

Table 1: Average RULER-HARD performance at 50× attention sparsity across models and context lengths. Full per-task results are provided in the Appendix.
<table><tr><td>Model</td><td>Method</td><td>32K</td><td>64K</td><td>128K</td></tr><tr><td rowspan="5">Llama-3.1-8B</td><td>Dense</td><td>88.76</td><td>86.52</td><td>71.26</td></tr><tr><td>PQ</td><td>72.51</td><td>67.53</td><td>39.73</td></tr><tr><td>vAttn (PQ)</td><td>66.71</td><td>66.22</td><td>39.45</td></tr><tr><td>Oracle-top-k</td><td>78.80</td><td>71.91</td><td>56.10</td></tr><tr><td>SSA</td><td>83.67</td><td>77.09</td><td>47.53</td></tr><tr><td rowspan="5">Ministral-3-3B</td><td>Dense</td><td>86.69</td><td>82.19</td><td>81.49</td></tr><tr><td>PQ</td><td>70.81</td><td>69.88</td><td>66.48</td></tr><tr><td>vAttn (PQ)</td><td>69.26</td><td>65.77</td><td>63.06</td></tr><tr><td>Oracle-top-k</td><td>72.81</td><td>75.09</td><td>71.87</td></tr><tr><td>SSA</td><td>82.05</td><td>75.70</td><td>69.65</td></tr><tr><td rowspan="5">Ministral-3-8B</td><td>Dense</td><td>89.33</td><td>86.13</td><td>81.45</td></tr><tr><td>PQ</td><td>81.83</td><td>78.02</td><td>62.99</td></tr><tr><td>vAttn (PQ)</td><td>78.69</td><td>75.19</td><td>63.78</td></tr><tr><td>Oracle-top-k</td><td>86.61</td><td>83.41</td><td>75.83</td></tr><tr><td>SSA</td><td>84.17</td><td>79.91</td><td>69.25</td></tr><tr><td rowspan="5">Qwen3.5-4B</td><td>Dense</td><td>91.95</td><td>91.11</td><td>89.45</td></tr><tr><td>PQ</td><td>91.22</td><td>90.72</td><td>88.72</td></tr><tr><td>vAttn (PQ)</td><td>90.56</td><td>90.61</td><td>89.28</td></tr><tr><td>Oracle-top-k</td><td>91.39</td><td>90.39</td><td>89.39</td></tr><tr><td>SSA</td><td>91.45</td><td>90.83</td><td>89.50</td></tr><tr><td rowspan="5">Qwen3.5-9B</td><td>Dense</td><td>93.45</td><td>92.22</td><td>91.28</td></tr><tr><td>PQ</td><td>92.00</td><td>91.08</td><td>90.95</td></tr><tr><td>vAttn (PQ)</td><td>91.67</td><td>91.30</td><td>90.61</td></tr><tr><td>Oracle-top-k</td><td>92.56</td><td>92.28</td><td>91.39</td></tr><tr><td>SSA</td><td>93.33</td><td>92.36</td><td>91.45</td></tr></table>

## 4.2 Theoretical Insights

We now characterize how accurately SSA approximates dense attention. Our analysis of Algorithm 1 is designed for the budget parameter $k \in \{ 1 , \ldots , N - 1 \}$ . When $k = N$ , every token is retained (see step 3), so SSA trivially returns dense attention exactly. The residual set is then empty and requires no sampling analysis. For $k < N$ , the residual set is nonempty $i . e . , m \ge 1$ . Let $a _ { i }$ be the dense attention weights from Section 3, and $\| \mathbf V \| _ { 2 }$ denote the spectral norm of the value matrix. In addition, we introduce the following quantities for our analysis,

$$
A _ { \mathcal { R } } = \sum _ { i \in \mathcal { R } } a _ { i } , \qquad \bar { a } _ { i } = \frac { a _ { i } } { A _ { \mathcal { R } } } , \qquad \rho = \operatorname* { m a x } _ { i \in \mathcal { R } } \frac { \bar { a } _ { i } } { p _ { i } } ,
$$

where $A _ { \mathcal { R } }$ is the true attention mass left for sampling, while $\rho$ measures how well the residual proposal covers that mass. For finite proxy scores, $q _ { i } > 0$ and hence $\begin{array} { r } { p _ { i } = q _ { i } / \sum _ { j \in \mathcal { R } } q _ { j } > 0 } \end{array}$ for every $i \in \mathcal { R }$ Since $\begin{array} { r } { \sum _ { i \in \mathcal { R } } p _ { i } ( \bar { a } _ { i } / p _ { i } ) = \sum _ { i \in \mathcal { R } } \bar { a } _ { i } = 1 } \end{array}$ , the largest ratio must satisfy $\rho \geq 1$ . The equality holds precisely when every ratio is equal to one, $i . e . , p _ { i } = \bar { a } _ { i }$ for all $i \in \mathcal { R }$

The budget-allocation rule also limits the efect of duplicate residual draws. Indeed, steps 3–4 of Algorithm 1 give $\begin{array} { r } { k = h + \lambda \sum _ { i \in \mathcal { R } } q _ { i } } \end{array}$ as each heavy token contributes 1 to the budget equation, and each residual token contributes $\lambda q _ { i } < 1$ . Since $m = k - h$ , it follows that $\begin{array} { r } { m = \lambda \sum _ { i \in \mathcal { R } } q _ { i } } \end{array}$ . Combining this with $\begin{array} { r } { p _ { i } = q _ { i } / \sum _ { j \in \mathcal { R } } q _ { j } } \end{array}$ gives $m p _ { i } = \lambda q _ { i } < 1$ for every $i \in \mathcal { R }$ . Here mp is the expected number of times token i appears among the m draws. The largest expected multiplicity among the residual

tokens is therefore defined as

$$
c : = \operatorname* { m a x } _ { i \in \mathcal { R } } m p _ { i } < 1 .
$$

A small value of c corresponds to a difuse proposal with few repetitions, whereas a value closer to one indicates a more concentrated proposal and a greater possibility of repeated draws.

$\operatorname { N o w } ,$ to use the proxy-quality parameter $\rho$ in the error bound, we need to relate the proposal probability $p _ { i }$ to the inclusion probability $\pi _ { i }$ used by Algorithm 1. A residual token is retained if it appears in at least one of the m draws, so $\pi _ { i } = 1 - ( 1 - p _ { i } ) ^ { m }$ . To account for duplicate draws, we define

$$
\kappa ( c ) : = \frac { 1 - e ^ { - c } } { c } .
$$

The following lemma quantifies their efect on the inclusion probabilities.

Lemma 1. Let m $\geq 1$ , let $p _ { i } > 0$ for $i \in \mathcal { R }$ , and let c and $\kappa ( c )$ be defined above. For every $i \in \mathcal { R }$ $\kappa ( c ) m p _ { i } \leq \pi _ { i } \leq m p _ { i }$ . Consequently, $\begin{array} { r } { \frac { a _ { i } } { \pi _ { i } } \leq \frac { A _ { \mathcal { R } } \rho } { \kappa ( c ) m } } \end{array}$

Now, we are ready to use the lemma to bound SSA’s approximation error.

Theorem 2. For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \epsilon _ { D } ( \delta ) : = \frac { 2 \sqrt { 2 } A _ { \mathcal { R } } \rho } { \kappa ( c ) } \sqrt { \frac { \log ( 4 / \delta ) } { m } } . \ I f \epsilon _ { D } ( \delta ) < 1 } \end{array}$ , then, with probability at least $1 - \delta$ over the residual sampling,

$$
\| \widehat { \mathbf { o } } - \mathbf { o } \| _ { 2 } \leq \frac { 4 \sqrt { 2 } A _ { \mathcal { R } } \rho } { \kappa ( c ) \left( 1 - \epsilon _ { D } ( \delta ) \right) } \sqrt { \frac { \log ( 4 / \delta ) } { m } } \| \mathbf { V } \| _ { 2 } .\tag{3}
$$

Since $\kappa ( c ) > 1 - e ^ { - 1 }$ , deduplication contributes only a universal constant factor to the approximation and sample-complexity bounds. The following corollary gives a direct $( \varepsilon , \delta )$ guarantee and a suficient sample size $\begin{array} { r } { m = \mathcal { O } \left( A _ { \mathcal { R } } ^ { 2 } \rho ^ { 2 } \varepsilon ^ { - 2 } \log \frac { 1 } { \delta } \right) } \end{array}$ that automatically ensures $\epsilon _ { D } ( \delta ) < 1$

Corollary 3. Consider Algorithm 1 with an integer budget $1 \leq k < N$ and finite attention and proxy scores. Let $\varepsilon , \delta \in ( 0 , 1 )$ . I $\begin{array} { r } { \dot { \bar { r } } m \geq 2 1 A _ { \mathcal { R } } ^ { 2 } \rho ^ { 2 } \frac { ( 2 + \varepsilon ) ^ { 2 } } { \varepsilon ^ { 2 } } } \end{array}$ log <sup>4</sup><sub>δ</sub> , then, with probability at least $1 - \delta$

$$
\left\| \widehat { \mathbf { o } } - \mathbf { o } \right\| _ { 2 } \leq \varepsilon \| \mathbf { V } \| _ { 2 } .\tag{4}
$$

Remark 4. Note that the approximation error in Theorem 2 scales as $A \mathcal { R } \rho \big / \sqrt { m }$ . The residual mass A<sub>R</sub> decreases when the retained heavy tokens capture more of the true attention mass, while ρ increases when the proxy underestimates an important residual token. Increasing the residual budget m reduces the stochastic error at the standard Monte Carlo rate $m ^ { - 1 / 2 }$ . For a fixed total budget, SSA therefore balances the reduction in $A _ { \mathcal { R } }$ obtained from retaining more heavy tokens against the larger sampling budget obtained from retaining fewer of them.

Remark 5. In both Theorem 2 and Corollary 3, we express the bounds using $\| \mathbf V \| _ { 2 }$ . The proofs only require a bound on the individual value-vector norms, so the same results hold with ma $\boldsymbol { \mathfrak { r } } _ { i \in [ N ] } \parallel \mathbf { v } _ { i } \parallel _ { 2 }$ in its place, without changing the constants.

## 4.3 Kernel Design

At long contexts, decode attention is memory-bandwidth bound because each step reads the entire KV cache from HBM [35, 20]. SSA reduces this memory trafic by selecting a sparse subset of tokens before loading their KV states. While Algorithm 1 determines the division between the heavy and residual strata adaptively, our optimized GPU implementation uses a fixed heavy fraction. Developing an equally eficient implementation of the adaptive algorithm remains future work. As shown in Table 4, the practical fixed-budget variant closely matches the performance of the theoretical formulation. The residual proposal, inclusion-probability correction, and resulting attention estimator remain unchanged. Our SSA decode step consists of four stages, where only the first three scan the full context using a few bytes per token.

Proxy scoring: Each key is represented by a 16-byte PQ code (16 subspaces, 256 centroids) [46], compared to 1 KiB for its fp16 key-value pair at head dimension 256. For each query, a lookup table computes the proxy score $\tilde { s } _ { i }$ for every token. Scores are scaled and masked for sink and local tokens within the same kernel, avoiding an additional pass.

Heavy-set selection: We select H using radix top-k [36], which returns an unsorted index set since the estimator in eq. 2 is invariant to the ordering of selected tokens.

Residual sampling: Direct sampling from the full residual distribution becomes expensive at long context lengths, as constructing and sampling from the distribution requires processing probabilities across the entire residual KV cache. To reduce this overhead, we use a hierarchical sampler that progressively narrows the search space. Each draw first selects a 1024-token chunk, then a 32-token sub-tile, and finally scans at most 32 tokens to select a key. We remove duplicate samples with a per-row hash table and reweight each unique sample by $1 / \pi _ { i }$ . In practice, we clamp $\pi _ { i }$ at 10<sup>−4</sup> to avoid excessively large weights, at the cost of introducing a small bias. Finally, we use a counter-based random number generator to obtain fresh samples across replays while retaining compatibility with CUDA graph capture.

Sparse decode: The final stage gathers the keys and values in H ∪ S from the paged KV cache and evaluates eq. 2 using an online softmax. To exploit more parallelism, we divide each query head across multiple thread blocks and combine their partial softmax results using a numerically stable log-sum-exp reduction. At 1M context and batch size 1, using 64 splits is 1.58-1.71× faster than using 32 splits across sparsity levels. Increasing the number of splits to 128 or 256, however, reduces performance because the additional parallelism no longer ofsets the increased launch and merge overhead.

## 5 Experiments

We conduct an extensive evaluation of SSA against Oracle-top-k, deterministic top-k, pure sampling, and hybrid baselines across multiple datasets, models, context lengths, and generation lengths. In addition, note that we slightly overload the notation h. In Algorithm 1 and the theoretical analysis, $h =$ |H| denotes the number of heavy tokens, whereas in the implementation and experimental discussion, $h \in [ 0 , 1 ]$ denotes the fraction of the total budget assigned to the heavy set. We describe the evaluation setup below.

## 5.1 Setup

Baselines: We compare SSA against three representative sparse-attention baselines: vAttention [9], PQCache [46], and Oracle-top-k. Oracle-top-k performs exact attention-score computation before selecting the top-k tokens and is therefore impractical for acceleration; we include it as an upper-bound reference for deterministic top-k selection. For all methods, we use 128 sink tokens and 128 local tokens, along with the hyperparameters recommended by the respective authors. Additional comparisons with MagicPig [6], HashAttention [8], Double Sparsity [45], and Quest [39] are provided in Table 5 in the Appendix. These results motivate our use of PQCache, the strongest approximate top-k baseline in this comparison, to construct SSA’s proposal distribution throughout our experiments.

Table 2: Kernel optimizations make SSA faster than dense decoding even at batch size 1. We report decode-time attention-layer latency at 1M context length on a single H200 GPU. Reference implements the same algorithm without the kernel optimizations described in Section 4.3. At batch size 1, it reaches only 0.37-0.47× the speed of FlashInfer and surpasses FlashInfer only at batch size 8. In contrast, optimized SSA is faster than FlashInfer across all measured batch sizes and sparsity levels. FlashInfer latencies across the two runs agree within 1.2%.
<table><tr><td colspan="2"></td><td colspan="3">Latency (ms)</td><td colspan="2">SSA speedup</td></tr><tr><td>Sparsity</td><td>Batch</td><td>FlashInfer</td><td>Reference</td><td>SSA</td><td></td><td>vs. FlashInfer vs. reference</td></tr><tr><td rowspan="3">50×</td><td>1</td><td>1.41</td><td>2.99</td><td>0.79</td><td>1.79×</td><td>3.80×</td></tr><tr><td>8</td><td>11.90</td><td>7.18</td><td>4.27</td><td>2.78×</td><td>1.68×</td></tr><tr><td>16</td><td>24.47</td><td>12.57</td><td>8.26</td><td>2.96×</td><td>1.52×</td></tr><tr><td rowspan="3">20×</td><td>1</td><td>1.41</td><td>3.34</td><td>1.01</td><td>1.40×</td><td>3.32×</td></tr><tr><td>8</td><td>11.91</td><td>8.74</td><td>5.76</td><td>2.07×</td><td>1.52×</td></tr><tr><td>16</td><td>24.53</td><td>15.51</td><td>11.32</td><td>2.17×</td><td>1.37×</td></tr><tr><td rowspan="3">10×</td><td>1</td><td>1.42</td><td>3.83</td><td>1.30</td><td>1.10×</td><td>2.95×</td></tr><tr><td>8</td><td>12.01</td><td>11.06</td><td>7.91</td><td>1.52×</td><td>1.40×</td></tr><tr><td>16</td><td>24.55</td><td>19.82</td><td>15.81</td><td>1.55×</td><td>1.25×</td></tr></table>

Datasets and Models: We evaluate SSA on four benchmark suites: RULER (32K-128K) [16], Long-Bench (10K-16K) [2], LOFT (32K-128K) [22], and AIME (32K token generation) [28]. Our evaluation spans five models: Llama-3.1-8B-Instruct [12], Qwen3.5-4B and Qwen3.5-9B [40], and Ministral-3-3B-Instruct-2517 and Ministral-3-8B-Instruct-2517 [26]. Dataset subsets and additional evaluation details are provided in Table 6 in Appendix.

## 5.2 Efectiveness of SSA

RULER-HARD: Table 1 reports average RULER-HARD scores at 50× attention sparsity. SSA consistently outperforms both PQ and vAttention(PQ) across all models and context lengths. Since these methods use the same PQ scores and attention budget, the comparison highlights the benefit of allocating part of the budget to score-guided sampling rather than exclusively to top-k retrieval or uniform sampling. The gains are particularly pronounced when deterministic top-k struggles. At 32K, SSA improves Llama-3.1-8B from 72.51 (PQ) to 83.67, approaching the dense score of 88.76. Similarly, on Ministral-3-3B, SSA improves the score from 70.81 to 82.05, compared with 86.69 for dense attention. Oracle-top-k, which selects tokens using exact attention scores, remains stronger at 128K for Llama-3.1-8B and Ministral-3-3B, and across all context lengths for Ministral-3-8B. On Qwen3.5-4B and Qwen3.5-9B, all sparse methods remain within two points of dense attention.

AIME 2025: Table 3 reports avg@16 accuracy for long-form generation. SSA remains robust as sparsity increases, with particularly large gains in the aggressive-sparsity regime. On Qwen3.5-9B at 50× sparsity, SSA achieves 59.8, compared with 48.3 for oracle top-k, 36.7 for PQ, and 35.8 for vAttention(PQ). At 100× sparsity, SSA retains an accuracy of 51.5, while vAttention(PQ) drops to 3.5. At lower sparsity, vAttention(PQ) performs better at 5× and at 10× on Qwen3.5-4B. We further study the efect of the heavy fraction h in Table 13. Hybrid allocations with h ∈ [0.2, 0.8] achieve 61.0-64.0, substantially outperforming both pure sampling (h = 0), which achieves 40.2-42.5, and pure top-k (h = 1), which achieves 49.4-53.8. These results demonstrate the benefit of combining deterministic retrieval with score-guided importance sampling, particularly under aggressive sparsity.

<table><tr><td>Sparsity</td><td colspan="3">Qwen3.5-9B Oracle-top-k c PQ vAttn(PQ)</td><td>SSA</td><td colspan="3">Qwen3.5-4B Oracle-top-k PQ vAttn(PQ)</td><td>SSA</td></tr><tr><td>Dense</td><td colspan="3">66.3</td><td></td><td colspan="3">54.2</td><td></td></tr><tr><td>5×</td><td>63.8</td><td></td><td>65.4</td><td>61.0</td><td></td><td></td><td></td><td>52.7</td></tr><tr><td>10×</td><td>59.4</td><td>63.5 53.8</td><td>62.7</td><td>64.0</td><td>49.8 47.9</td><td>49.4 41.5</td><td>54.4 51.9</td><td>51.0</td></tr><tr><td>20×</td><td>55.4</td><td>49.4</td><td>58.3</td><td>63.5</td><td>42.7</td><td>33.8</td><td>50.2</td><td>53.1</td></tr><tr><td>50×</td><td>48.3</td><td>36.7</td><td>35.8</td><td>59.8</td><td>36.5</td><td>25.4</td><td>27.5</td><td>50.0</td></tr><tr><td>100×</td><td>42.5</td><td>28.1</td><td>3.5</td><td>51.5</td><td>31.9</td><td>13.1</td><td>1.7</td><td>42.5</td></tr></table>

<table><tr><td>Method</td><td>50×</td><td>20×</td></tr><tr><td>Dense</td><td>41.56</td><td>41.56</td></tr><tr><td>PQ</td><td>40.60</td><td>40.98</td></tr><tr><td>vAttn (PQ)</td><td>40.39</td><td>41.47</td></tr><tr><td>SSA</td><td>40.92</td><td>41.56</td></tr><tr><td>Oracle-top-k</td><td>41.25</td><td>41.40</td></tr></table>

Table 3: AIME 2025 avg@16 accuracy (left). LongBench performance with Qwen3.5-27B (right).

![](images/9d9878cca0438c488d5b609eec3702f32f4e9479285233fc233619a664a4ce60.jpg)

![](images/8295634bd0e86a0e75280ba62b0deed4aa587f45777f1b35fdd0a8ba416671bf.jpg)

![](images/9b1bfaff65db6820992d92c33477fb6006d43ebe7b154634719db46ac3e57809.jpg)  
Figure 3: SSA accelerates decode at long contexts, with larger gains at higher batch sizes. Decode attention speedup on a single H200 (16 query heads, 4 KV heads, head dimension 256, fp16 KV). (a) At batch size 1, SSA overtakes FlashInfer beyond 256K context and reaches 1.79× speedup at 1M and 50× sparsity. (b) At 1M context, speedup increases with batch size, reaching 2.96× at batch 16 and 50× sparsity. (c) Overhead of two-level residual sampling relative to 1,024-token chunk-only sampling as the number of draws m increases.

LOFT and LongBench: On LOFT with Qwen3.5-9B and Qwen3.5-4B (see fig. 2), SSA has the lowest attention error among the non-oracle methods at every context length and sparsity level, while subspan-EM varies by at most 0.04 across methods (see fig. 5). On LongBench with Qwen3.5-27B (Table 3), all sparse methods stay within one point of dense attention at both 50× and 20× sparsity.

## 5.3 Decode Eficiency

Sparse attention accelerates decoding only when its selection overhead is smaller than the memory trafic it saves. We evaluate this trade-of for SSA across context lengths and batch sizes, and separately measure the sampling overhead absent in top-k methods.

Eficiency analysis: SSA becomes increasingly advantageous as context length and batch size grow. At batch size 1, SSA is slower than FlashInfer at 128K, achieving approximately 0.60-0.74× its speed. As context length increases, this overhead is increasingly amortized, and SSA reaches 1.79×, 1.39×, and 1.07× speedups at 1M for 50×, 20×, and 10× sparsity, respectively (see fig. 3a). The gains further increase with batch size, reaching 2.96× at batch 16 for 50× sparsity (see fig. 3b); our kernel optimizations are particularly important at batch 1, improving SSA by 2.95-3.80× over the reference implementation (Table 2). Finally, although residual sampling introduces overhead absent from top-k methods, our two-level sampler reduces this cost by 1.21-3.16× over chunk-only routing as the number of draws m increases (see fig. 3c), keeping sampling latency at 0.30-1.18 ms at 1M context and h = 0.4 across the evaluated sparsity levels.

![](images/833452fde5e19513dec4548e7edf2a3fa9f97787d5e452ad265e429d643e6545.jpg)  
Figure 4: Illustration of Stratified Sparse Attention algorithm.

## 6 Conclusion

We introduce Stratified Sparse Attention (SSA), a unified framework encompassing top-k, pure sampling, and hybrid sparse attention while substantially outperforming existing methods and reducing the accuracy gap to dense attention. SSA is compatible with any approximate top-k method that provides proxy scores, allowing it to directly benefit from advances in approximate retrieval. An important direction for future work is to make SSA verifiable by incorporating user-controlled guarantees on approximation quality.

## Acknowledgments

The work was primarily supported by Rice Ken Kennedy Institute (K2I) Generative AI Cluster Funding. The Sky Computing Lab is supported by gifts from Accenture, Algorithmic SuperIntelligence Labs, Amazon, AMD, Anyscale, Broadcom, cmpnd, Google, IBM, Intel, Intesa Sanpaolo, Lambda, Lightspeed, Mirendil, NVIDIA, Samsung SDS, and VESSL.

## References

[1] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

[2] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li. LongBench: A Bilingual, Multitask Benchmark for Long Context Understand-

ing. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, Aug. 2024.

[3] W. Cao, X. Yin, B. Dhingra, and S. Zhou. Coding Agents are Efective Long-Context Processors. arXiv preprint arXiv:2603.20432, 2026.

[4] M. Chen, L. Wang, and B. Qu. The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents. arXiv preprint arXiv:2608.06663, 2026.

[5] X. Chen, X. Hu, Y. Huang, H. Jiang, W. Ji, Y. Jiang, Y. Jiang, B. Liu, H. Liu, X. Li, X. Lian, G. Meng, X. Peng, H. Sun, L. Shi, B. Wang, C. Wang, J. Wang, T. Wang, J. Xuan, X. Xia, Y. Yang, Y. Yang, L. Zhang, Y. Zhou, and L. Zhang. Deep learning-based software engineering: Progress, challenges, and opportunities. CoRR, abs/2410.13110, 2024.

[6] Z. Chen, R. Sadhukhan, Z. Ye, Y. Zhou, J. Zhang, N. Nolte, Y. Tian, M. Douze, L. Bottou, Z. Jia, and B. Chen. MagicPIG: LSH Sampling for Eficient LLM Generation. In International Conference on Learning Representations, 2025.

[7] DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Eficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, 2024.

[8] A. Desai, S. Yang, A. Cuadron, A. Klimovic, M. Zaharia, J. E. Gonzalez, and I. Stoica. HashAttention: Semantic Sparsity for Faster Inference. In International Conference on Machine Learning, 2025.

[9] A. Desai, K. K. Agrawal, S. Yang, A. Cuadron, L. G. Schroeder, M. Zaharia, J. E. Gonzalez, and I. Stoica. vAttention: Verified Sparse Attention via Sampling. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= zzTDulLys0.

[10] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[11] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, and J. Gao. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. In International Conference on Learning Representations, 2024.

[12] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.

[13] T. Haris. kNN Attention Demystified: A Theoretical Exploration for Scalable Transformers. In The Thirteenth International Conference on Learning Representations, 2025.

[14] C. R. C. Hooper, S. Kim, H. Mohammadzadeh, M. Maheswaran, S. Zhao, J. Paik, M. W. Mahoney, K. Keutzer, and A. Gholami. Squeezed Attention: Accelerating Long Context Length LLM Inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025.

[15] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy, and H. Wang. Large Language Models for Software Engineering: A Systematic Literature Review. ACM Trans. Softw. Eng. Methodol., 33(8), 2024.

[16] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg.

RULER: What’s the Real Context Size of Your Long-Context Language Models? In First Conference on Language Modeling (COLM), 2024.

[17] P. Indyk and R. Motwani. Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pages 604–613, 1998.

[18] S. Joshi, A. Chowdhury, W. Bellinger, A. Kanakamedala, E. Singh, H. A. D. Le, A. Desai, and A. Shrivastava. SOCKET: SOft Collision Kernel EsTimator for Sparse Attention. arXiv preprint arXiv:2602.06283, 2026.

[19] J.-H. Kim, J. Kim, S. Kwon, J. W. Lee, S. Yun, and H. O. Song. KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[20] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Eficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

[21] H. A. D. Le, S. Joshi, Z. Yang, Z. Xu, and A. Shrivastava. Scout before you attend: Sketchand-walk sparse attention for eficient LLM inference. In Forty-third International Conference on Machine Learning, 2026.

[22] J. Lee, A. Chen, Z. Dai, D. Dua, D. S. Sachan, M. Boratko, Y. Luan, S. M. R. Arnold, V. Perot, S. Dalmia, H. Hu, X. Lin, P. Pasupat, A. Amini, J. R. Cole, S. Riedel, I. Naim, M.-W. Chang, and K. Guu. Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? arXiv preprint arXiv:2406.13121, 2024.

[23] Y. Li, S. Wang, H. Ding, and H. Chen. Large language models in finance: A survey. In Proceedings of the 4th ACM International Conference on AI in Finance, 2023. doi: 10.1145/3604237.3626869.

[24] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. SnapKV: LLM Knows What You are Looking for Before Generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[25] X. Lin, Y. Zhang, S. Ge, L. Ren, B. Patra, V. Chaudhary, H. Peng, and X. Song. S2-ATTENTION: HARDWARE-AWARE CONTEXT SHARDING AMONG ATTENTION HEADS. In Sparsity in LLMs (SLLM): Deep Dive into Mixture of Experts, Quantization, Hardware, and Inference, 2025. URL https://openreview.net/forum?id=AhlE339I7o.

[26] A. H. Liu, K. Khandelwal, S. Subramanian, V. Jouault, A. Rastogi, A. Sadé, A. Jefares, A. Jiang, A. Cahill, A. Gavaudan, A. Sablayrolles, A. Héliou, et al. Ministral 3. arXiv preprint arXiv:2601.08584, 2026.

[27] D. Liu, M. Chen, B. Lu, H. Jiang, Z. Han, Q. Zhang, Q. Chen, C. Zhang, B. Ding, K. Zhang, C. Chen, F. Yang, Y. Yang, and L. Qiu. RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[28] Mathematical Association of America. American Invitational Mathematics Examination (AIME) 2025. https://maa.org/maa-invitational-competitions/, 2025. Accessed: 2026-05- 20.

[29] M. Pagliardini, D. Paliotta, M. Jaggi, and F. Fleuret. Fast attention over long sequences with dynamic sparse flash attention. In Advances in Neural Information Processing Systems, volume 36, pages –, 2023.

[30] J. Park, D. Jones, M. J. Morse, R. Goel, M. Lee, and C. Lott. KeyDif: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[31] W. Peebles and S. Xie. Scalable Difusion Models with Transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[32] I. Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, 22(4):1679–1706, 1994. doi: 10.1214/aop/1176988477.

[33] L. Ruan and Q. Jin. Survey: Transformer based Video-Language Pre-training. AI Open, 3:1–13, 2022. https://www.sciencedirect.com/science/article/pii/ S2666651022000018.

[34] A. Sengupta, S. Chaudhary, and T. Chakraborty. Value-Guided KV Compression for LLMs via Approximated CUR Decomposition. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[35] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[36] A. Shanbhag, H. Pirk, and S. Madden. Eficient top-k query processing on massively parallel hardware. In Proceedings ofthe 2018 International Conference on Management ofData, SIGMOD ’18. Association for Computing Machinery, 2018. ISBN 9781450347037. URL https://doi. org/10.1145/3183713.3183735.

[37] N. Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

[38] P. Singhania, S. Singh, S. He, S. Feizi, and A. Bhatele. Loki: Low-rank Keys for Eficient Sparse Attention. In Advances in Neural Information Processing Systems, 2024.

[39] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. QUEST: Query-Aware Sparsity for Eficient Long-Context LLM Inference. In International Conference on Machine Learning, 2024.

[40] Q. Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

[41] S. T. Tokdar and R. E. Kass. Importance sampling: a review. Wiley Interdisciplinary Reviews: Computational Statistics, 2, 2010.

[42] J. S. Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software (TOMS), 11(1), 1985.

[43] S. Yan, G.-Q. Jiang, Y. Zhang, X. Ma, R. Zhu, C. Cao, and J. Xu. Adamas: Hadamard Sparse Attention for Eficient Long-Context Inference. arXiv preprint arXiv:2510.18413, 2025.

[44] J. Yang, C. Tang, L. Zhen, and J. Lv. Rethinking KV cache eviction via a unified informationtheoretic objective. In Forty-third International Conference on Machine Learning, 2026.

[45] S. Yang, Y. Sheng, J. E. Gonzalez, I. Stoica, and L. Zheng. Post-Training Sparse Attention with Double Sparsity. arXiv preprint arXiv:2408.07092, 2024.

[46] H. Zhang, X. Ji, Y. Chen, F. Fu, X. Miao, X. Nie, W. Chen, and B. Cui. PQCache: Product Quantization-based KVCache for Long Context LLM Inference. Proceedings of the ACM on Management of Data, 3(3):1–30, 2025.

[47] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Ré, C. W. Barrett, Z. Wang, and B. Chen. $\mathrm { H _ { 2 } O } { \mathrm { : } }$ Heavy-Hitter Oracle for Eficient Generative Inference of Large Language Models. In Advances in Neural Information Processing Systems, 2023.

[48] X. Zhou, Z. Sun, and G. Li. DB-GPT: Large Language Model Meets Database. Data Science and Engineering, 9:102 – 111, 2024.

[49] X. Zhou, W. Wang, M. Zeng, J. Guo, X. Liu, L. Shen, M. Zhang, and L. Ding. DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs. arXiv preprint arXiv:2412.14838, 2024.

[50] K. Zhu, T. Tang, Q. Xu, Z. Jin, Y. Gu, Z. Zeng, R. Kadekodi, L. Zhao, A. Li, A. Krishnamurthy, and B. Kasikci. Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs. In The Fourteenth International Conference on Learning Representations, 2026.

## A Appendix

## A.1 Proof of Lemma 1

Proof. Token i is included if it appears in at least one of the m independent draws, so

$$
\pi _ { i } = 1 - ( 1 - p _ { i } ) ^ { m } .
$$

By the union bound,

$$
\pi _ { i } = \operatorname* { P r } \left( \bigcup _ { \ell = 1 } ^ { m } \{ J _ { \ell } = i \} \right) \leq \sum _ { \ell = 1 } ^ { m } \operatorname* { P r } ( J _ { \ell } = i ) = m p _ { i } .
$$

For the lower bound, $1 - p _ { i } \le e ^ { - p _ { i } }$ gives

$$
\pi _ { i } = 1 - ( 1 - p _ { i } ) ^ { m } \geq 1 - e ^ { - m p _ { i } } .
$$

The function $g ( x ) = ( 1 - e ^ { - x } ) / x$ is decreasing for $x > 0$ . Since $m p _ { i } \le c ,$

$$
1 - e ^ { - m p _ { i } } = m p _ { i } g ( m p _ { i } ) \geq m p _ { i } g ( c ) = \kappa ( c ) m p _ { i } .
$$

This proves the two-sided bound on $\pi _ { i }$ .

Finally, the definition of $\rho$ gives $a _ { i } = A _ { \mathcal { R } } \bar { a } _ { i } \leq A _ { \mathcal { R } } \rho p _ { i }$ . Combining this with the lower bound on $\pi _ { i }$ yields

$$
\frac { a _ { i } } { \pi _ { i } } \leq \frac { A _ { \mathcal { R } } \rho p _ { i } } { \kappa ( c ) m p _ { i } } = \frac { A _ { \mathcal { R } } \rho } { \kappa ( c ) m } .
$$

## A.2 Proof of Theorem 2

We first establish the following dimension-free bounded-diferences inequality used in the proof.

Lemma 6 (Vector-valued bounded diferences). Let $X _ { 1 } , \ldots , X _ { m }$ be independent random variables, and let $f$ be a measurable function taking values in $\mathbb { R } ^ { r }$ . Suppose that changing only the $\ell ^ { t h }$ argument changes f by at most $b _ { \ell }$ in Euclidean norm i.e.,

$$
\left\| f ( x _ { 1 } , \dots , x _ { \ell } , \dots , x _ { m } ) - f ( x _ { 1 } , \dots , x _ { \ell } ^ { \prime } , \dots , x _ { m } ) \right\| _ { 2 } \leq b _ { \ell }
$$

for every $\ell \in [ m ]$ and every pair of input sequences difering only in coordinate $\ell ,$ where $b _ { \ell } \geq 0$ and $\Sigma _ { \ell = 1 } ^ { m } b _ { \ell } ^ { 2 } > 0$ . Then, for every $t > 0$ 2

$$
\operatorname* { P r } \left( \| f ( X _ { 1 } , \ldots , X _ { m } ) - \mathbb { E } [ f ( X _ { 1 } , \ldots , X _ { m } ) ] \| _ { 2 } \ge t \right) \le 2 \exp \left( - { \frac { t ^ { 2 } } { 2 \sum _ { \ell = 1 } ^ { m } b _ { \ell } ^ { 2 } } } \right) .\tag{5}
$$

Proof. Reveal the variables one at a time and define

$$
M _ { \ell } = \operatorname { \mathbb { E } } [ f ( X _ { 1 } , \dots , X _ { m } ) \mid X _ { 1 } , \dots , X _ { \ell } ] , \qquad \ell = 0 , \dots , m .
$$

Here

$$
M _ { 0 } = \mathbb { E } [ f ( X _ { 1 } , \dots , X _ { m } ) ] , \qquad M _ { m } = f ( X _ { 1 } , \dots , X _ { m } ) .
$$

The sequence $( M _ { \ell } ) _ { \ell = 0 } ^ { m }$ is a vector-valued martingale.

We first bound its increments. Fix the first $\ell - 1$ variables and let

$$
g _ { \ell } ( x ) = \mathbb { E } _ { X _ { \ell + 1 } , \ldots , X _ { m } } [ f ( X _ { 1 } , \ldots , X _ { \ell - 1 } , x , X _ { \ell + 1 } , \ldots , X _ { m } ) ] .
$$

Averaging over the same future variables preserves the bounded-diferences condition:

$$
\| g _ { \ell } ( x ) - g _ { \ell } ( x ^ { \prime } ) \| _ { 2 } \leq b _ { \ell } .
$$

Let $X _ { \ell } ^ { \prime }$ be an independent copy of $X _ { \ell } ,$ independent of all the original variables. Since $M _ { \ell } = g _ { \ell } ( X _ { \ell } )$ and $M _ { \ell - 1 }$ averages $g _ { \ell }$ over the $\ell ^ { t h }$ variable, Jensen’s inequality gives

$$
\begin{array} { r l } & { \| M _ { \ell } - M _ { \ell - 1 } \| _ { 2 } = \left\| \mathbb { E } \big [ g _ { \ell } ( X _ { \ell } ) - g _ { \ell } ( X _ { \ell } ^ { \prime } ) \ | \ X _ { 1 } , \ldots , X _ { \ell } \big ] \right\| _ { 2 } } \\ & { \qquad \leq \mathbb { E } \big [ \| g _ { \ell } ( X _ { \ell } ) - g _ { \ell } ( X _ { \ell } ^ { \prime } ) \| _ { 2 } \ | \ X _ { 1 } , \ldots , X _ { \ell } \big ] } \\ & { \qquad \leq b _ { \ell } . } \end{array}
$$

The Euclidean specialization of the martingale inequality in [32, Theorem 3.5], applied to $( M _ { \ell } { - } M _ { 0 } ) _ { \ell = 0 } ^ { m }$ therefore yields

$$
\operatorname* { P r } \left( \| M _ { m } - M _ { 0 } \| _ { 2 } \geq t \right) \leq 2 \exp \left( - \frac { t ^ { 2 } } { 2 \sum _ { \ell = 1 } ^ { m } b _ { \ell } ^ { 2 } } \right) .
$$

Substituting $M _ { m }$ and $M _ { 0 }$ proves the claim.

Now, we are ready to prove Theorem 2.

Proof of Theorem 2. We consider the sparse regime with an integer budget $1 \le k < N$ and finite attention and proxy scores. The case $k = N$ can be handled by evaluating dense attention exactly. Recall that H and $\mathcal { R } = [ N ] \backslash$ H be the heavy and residual sets produced by SSA, and let $h = | \mathcal { H } |$ and $m = k - h .$ . As shown in Section 4.2, $\mathcal { R } \neq \emptyset$ and $m \geq 1$ . Since the proxy scores are finite, $p _ { i } > 0$ for every $i \in \mathcal { R }$

Let $J _ { 1 } , \ldots , J _ { m }$ denote the independent residual draws:

$$
\operatorname* { P r } ( J _ { \ell } = i ) = p _ { i } , \qquad i \in \mathcal { R } , \qquad \ell \in [ m ] .
$$

For each residual token, define its occupancy indicator

$$
Z _ { i } = { \bf 1 } \{ J _ { \ell } = i \ \mathrm { f o r ~ s o m e } \ \ell \in [ m ] \} , \qquad i \in \mathcal { R } .
$$

We know that its inclusion probability is given by $\pi _ { i } = \operatorname* { P r } ( Z _ { i } = 1 ) = 1 - ( 1 - p _ { i } ) ^ { m }$ , so $\mathbb { E } [ Z _ { i } ] = \pi _ { i }$ Heavy tokens are retained deterministically and have inclusion probability one.

Note that although the residual indicators $\{ Z _ { i } \} _ { i \in \mathcal { R } }$ are dependent, they are deterministic functions of the independent draws $J _ { 1 } , \ldots , J _ { m }$ . We therefore apply Lemma 6 to these draws. Write

$$
D : = \sum _ { i = 1 } ^ { N } w _ { i } , \qquad a _ { i } = \frac { w _ { i } } { D } , \qquad \mathbf { o } = \sum _ { i = 1 } ^ { N } a _ { i } \mathbf { v } _ { i } .
$$

Define the estimated denominator and numerator by

$$
\widehat { D } : = \sum _ { i \in \mathcal { H } } w _ { i } + \sum _ { i \in \mathcal { R } } \frac { Z _ { i } w _ { i } } { \pi _ { i } } ,
$$

$$
{ \widehat { \mathbf { N } } } : = \sum _ { i \in \mathcal { H } } w _ { i } \mathbf { v } _ { i } + \sum _ { i \in \mathcal { R } } \frac { Z _ { i } w _ { i } } { \pi _ { i } } \mathbf { v } _ { i } .
$$

The SSA output is therefore

$$
\widehat { \mathbf { o } } = \frac { \widehat { \mathbf { N } } } { \widehat { D } } .
$$

Since $m \geq 1$ and all weights are positive, $\widehat { D } > 0$ for every realization of the residual draws. ${ \mathrm { I f ~ } } \| \mathbf { V } \| _ { 2 } = 0$ , all value vectors vanish, and ${ \widehat { \mathbf { o } } } = \mathbf { o } = \mathbf { 0 }$ , so the theorem is immediate. Therefore, we assume $\| \mathbf V \| _ { 2 } > 0$

By Lemma 1,

$$
\frac { a _ { i } } { \pi _ { i } } \leq \frac { A _ { \mathcal { R } } \rho } { \kappa ( c ) m } , \qquad i \in \mathcal { R } .
$$

For brevity, define

$$
\beta : = \frac { A _ { { \mathcal R } } \rho } { \kappa ( c ) m } .\tag{6}
$$

Thus $a _ { i } / \pi _ { i } \leq \beta$ for every $i \in \mathcal { R }$

Next, we define

$$
\Delta _ { D } : = \frac { \widehat { D } - D } { D } .\tag{7}
$$

The contributions from H are evaluated exactly and cancel in $\widehat { D } - D$ . Consequently,

$$
\Delta _ { D } = \sum _ { i \in \mathcal { R } } a _ { i } \left( \frac { Z _ { i } } { \pi _ { i } } - 1 \right) .\tag{8}
$$

Since $\mathbb { E } [ Z _ { i } ] = \pi _ { i }$

$$
\mathbb { E } [ \Delta _ { D } ] = 0 .
$$

Consider two draw sequences $\mathbf { J } = \left( J _ { 1 } , \ldots , J _ { m } \right)$ and $\mathbf { J } ^ { \prime }$ difering only in coordinate $\ell ,$ where the draw changes from r to s. Only $Z _ { r }$ and $Z _ { s }$ can change, and each changes by at most one. Therefore,

$$
\begin{array} { c } { \displaystyle { \left| \Delta _ { D } ( \mathbf { J } ) - \Delta _ { D } ( \mathbf { J } ^ { \prime } ) \right| \leq \displaystyle \frac { a _ { r } } { \pi _ { r } } + \frac { a _ { s } } { \pi _ { s } } } } \\ { \displaystyle { \leq 2 \beta } . } \end{array}\tag{9}
$$

Applying Lemma 6 with $r = 1$ and $b _ { \ell } = 2 \beta$ gives, for every $t > 0$

$$
\mathrm { P r } \left( | \Delta _ { D } | \geq t \right) \leq 2 \exp \left( - \frac { t ^ { 2 } } { 8 m \beta ^ { 2 } } \right) .
$$

Taking $t = 2 \sqrt { 2 m } \beta \sqrt { u }$ , we obtain, for every $u > 0$

$$
\operatorname* { P r } \left( | \Delta _ { D } | \geq 2 { \sqrt { 2 m } } \beta { \sqrt { u } } \right) \leq 2 e ^ { - u } .\tag{10}
$$

Substituting eq. 6 yields

$$
\operatorname* { P r } \left( | \Delta _ { D } | \geq { \frac { 2 { \sqrt { 2 } } A _ { \mathcal { R } } \rho } { \kappa ( c ) } } { \sqrt { \frac { u } { m } } } \right) \leq 2 e ^ { - u } .\tag{11}
$$

Now, we define

$$
\mathbf { E } : = \frac { \widehat { \mathbf { N } } - \mathbf { o } \widehat { D } } { D } .\tag{12}
$$

Because $\textstyle \sum _ { i = 1 } ^ { N } a _ { i } = 1$ and $\begin{array} { r } { { \mathbf { o } } = \sum _ { i = 1 } ^ { N } a _ { i } { \mathbf { v } } _ { i } } \end{array}$ , we have

$$
\sum _ { i = 1 } ^ { N } a _ { i } ( \mathbf { v } _ { i } - \mathbf { o } ) = \mathbf { 0 } .
$$

It follows that

$$
\begin{array} { l } { { \displaystyle { \bf E } = \sum _ { i \in { \mathcal { H } } } a _ { i } ( { \bf v } _ { i } - { \bf 0 } ) + \sum _ { i \in { \mathcal { R } } } \frac { Z _ { i } a _ { i } } { \pi _ { i } } ( { \bf v } _ { i } - { \bf 0 } ) } } \\ { { \displaystyle ~ = \sum _ { i \in { \mathcal { R } } } a _ { i } \left( \frac { Z _ { i } } { \pi _ { i } } - 1 \right) ( { \bf v } _ { i } - { \bf 0 } ) } . } \end{array}\tag{13}
$$

Again using $\mathbb { E } [ Z _ { i } ] = \pi _ { i }$ , we obtain

$$
\mathbb { E } [ \mathbf { E } ] = \mathbf { 0 } .
$$

Let $\mathbf { e } _ { i }$ denote the $i ^ { t h }$ standard basis vector of R $N$ . Since the value vectors are the rows of V, for every $i \in [ N ]$ ，

$$
\| \mathbf { v } _ { i } \| _ { 2 } = \| \mathbf { V } ^ { \top } \mathbf { e } _ { i } \| _ { 2 } \leq \| \mathbf { V } ^ { \top } \| _ { 2 } \| \mathbf { e } _ { i } \| _ { 2 } = \| \mathbf { V } \| _ { 2 } .
$$

Moreover, o is a convex combination of the value vectors, so

$$
\| \mathbf { o } \| _ { 2 } \leq \sum _ { i = 1 } ^ { N } a _ { i } \| \mathbf { v } _ { i } \| _ { 2 } \leq \| \mathbf { V } \| _ { 2 } .
$$

Therefore,

$$
\| \mathbf { v } _ { i } - \mathbf { o } \| _ { 2 } \leq \| \mathbf { v } _ { i } \| _ { 2 } + \| \mathbf { o } \| _ { 2 } \leq 2 \| \mathbf { V } \| _ { 2 } .\tag{14}
$$

Now consider draw sequences J and $\mathbf { J } ^ { \prime }$ that difer only in one draw, changed from r to s. As before, only $Z _ { r }$ and $Z _ { s }$ can change. Hence,

$$
\begin{array} { l } { \displaystyle \| \mathbf { E } ( \mathbf { J } ) - \mathbf { E } ( \mathbf { J } ^ { \prime } ) \| _ { 2 } \leq \frac { a _ { r } } { \pi _ { r } } \| \mathbf { v } _ { r } - \mathbf { o } \| _ { 2 } + \frac { a _ { s } } { \pi _ { s } } \| \mathbf { v } _ { s } - \mathbf { o } \| _ { 2 } } \\ { \leq 4 \beta \| \mathbf { V } \| _ { 2 } . } \end{array}\tag{15}
$$

Applying Lemma 6 with $r = d$ and $b _ { \ell } = 4 \beta \| \mathbf { V } \| _ { 2 }$ gives, for every $t > 0$

$$
\operatorname* { P r } \left( \| \mathbf { E } \| _ { 2 } \geq t \right) \leq 2 \exp \left( - { \frac { t ^ { 2 } } { 3 2 m \beta ^ { 2 } \| \mathbf { V } \| _ { 2 } ^ { 2 } } } \right) .
$$

Taking $t = 4 { \sqrt { 2 m } } \beta { \sqrt { u } } \| \mathbf { V } \| _ { 2 }$ gives, for every $u > 0$

$$
\operatorname* { P r } \left( \| \mathbf { E } \| _ { 2 } \geq 4 { \sqrt { 2 m } } \beta { \sqrt { u } } \| \mathbf { V } \| _ { 2 } \right) \leq 2 e ^ { - u } .\tag{16}
$$

Equivalently,

$$
\operatorname* { P r } \left( \| \mathbf { E } \| _ { 2 } \geq { \frac { 4 { \sqrt { 2 } } A _ { \mathcal { R } } \rho } { \kappa ( c ) } } { \sqrt { \frac { u } { m } } } \| \mathbf { V } \| _ { 2 } \right) \leq 2 e ^ { - u } .\tag{17}
$$

Now, fix the failure probability $\delta \in ( 0 , 1 )$ and set

$$
u = \log { \frac { 4 } { \delta } } .
$$

Then $2 e ^ { - u } = \delta / 2$ . By $\mathrm { e q } .$ . 11, with probability at least $1 - \delta / 2$

$$
| \Delta _ { D } | \leq \frac { 2 \sqrt { 2 } A _ { \mathcal { R } } \rho } { \kappa ( c ) } \sqrt { \frac { \log ( 4 / \delta ) } { m } } = \epsilon _ { D } ( \delta ) .
$$

Similarly, by eq. 17, with probability at least $1 - \delta / 2$

$$
\| \mathbf { E } \| _ { 2 } \leq { \frac { 4 { \sqrt { 2 } } A _ { \mathcal { R } } \rho } { \kappa ( c ) } } { \sqrt { { \frac { \log ( 4 / \delta ) } { m } } } } \| \mathbf { V } \| _ { 2 } .
$$

A union bound shows that both inequalities hold simultaneously with probability at least $1 - \delta$ By the definition of $\Delta _ { D }$

$$
\frac { \widehat { D } } { D } = 1 + \Delta _ { D } .
$$

Consequently,

$$
\widehat { \mathbf { o } } - \mathbf { o } = \frac { \widehat { \mathbf { N } } } { \widehat { D } } - \mathbf { o } = \frac { \widehat { \mathbf { N } } - \mathbf { o } \widehat { D } } { \widehat { D } } = \frac { \mathbf { E } } { 1 + \Delta _ { D } } .\tag{18}
$$

On the simultaneous concentration event, the assumption $\epsilon _ { D } ( \delta ) < 1$ implies

$$
1 + \Delta _ { D } \geq 1 - | \Delta _ { D } | \geq 1 - \epsilon _ { D } ( \delta ) > 0 .
$$

Therefore,

$$
\begin{array} { r l } & { \| \widehat { \mathbf o } - \mathbf o \| _ { 2 } = \displaystyle \frac { \| \mathbf E \| _ { 2 } } { 1 + \Delta _ { D } } \leq \displaystyle \frac { \| \mathbf E \| _ { 2 } } { 1 - \epsilon _ { D } ( \delta ) } } \\ & { \qquad \leq \displaystyle \frac { 4 \sqrt { 2 } A _ { \mathcal { R } } \rho } { \kappa ( c ) \big ( 1 - \epsilon _ { D } ( \delta ) \big ) } \sqrt { \displaystyle \frac { \log ( 4 / \delta ) } { m } } \| \mathbf { V } \| _ { 2 } . } \end{array}
$$

This proves eq. 3 with probability at least $1 - \delta .$

## A.3 Proof of Corollary 3

Proof. Let

$$
x : = \epsilon _ { D } ( \delta ) = \frac { 2 \sqrt { 2 } A _ { \mathcal { R } } \rho } { \kappa ( c ) } \sqrt { \frac { \log ( 4 / \delta ) } { m } } .
$$

The assumed lower bound on m gives

$$
x \leq \frac { 2 \sqrt { 2 } } { \sqrt { 2 1 } \kappa ( c ) } \frac { \varepsilon } { 2 + \varepsilon } .
$$

Under the Algorithm 1 allocation rule, $c < 1$ , so $\kappa ( c ) > 1 - e ^ { - 1 }$ . Since

$$
\frac { 8 } { ( 1 - e ^ { - 1 } ) ^ { 2 } } < 2 1 ,
$$

we have

$$
x \leq \frac { \varepsilon } { 2 + \varepsilon } < 1 .
$$

Thus the denominator-stability condition in Theorem 2 holds.

The numerator of the bound in Theorem 2 equals $2 x \| \mathbf { V } \| _ { 2 }$ . Therefore, with probability at least $1 - \delta .$

$$
\| \widehat { \mathbf { o } } - \mathbf { o } \| _ { 2 } \leq \frac { 2 x } { 1 - x } \| \mathbf { V } \| _ { 2 } \leq \frac { 2 \varepsilon / ( 2 + \varepsilon ) } { 1 - \varepsilon / ( 2 + \varepsilon ) } \| \mathbf { V } \| _ { 2 } = \varepsilon \| \mathbf { V } \| _ { 2 } .
$$

## A.4 Additional Experiments

![](images/677975130899a49c16e6e54f50873e40c09008ff07413441b6001edcfc714a3b.jpg)  
Context Length (Bars: Sparsity)  
(a) Ministral-3-8B: Subspan-EM

![](images/8e05894799a94e53a6877b1e76fa1a883985395923a79ca9364b0c587ea47e9b.jpg)  
(b) Ministral-3-8B: Error

![](images/b8a56c8cc5dd542aee4b92aff54008f3258a2c2f81a288e69ab799496ba63d77.jpg)  
(c) Qwen3.5-4B: Subspan-EM

![](images/784a836553d970c6c275e92cea6b36809159e172882eb64dbfe8fdfd91fbc51b.jpg)  
(d) Qwen3.5-9B: Subspan-EM  
Figure 5: Average Subspan-EM and error across diferent context lengths and sparsity levels on LOFT.

Table 4: Comparison of theoretical and practical SSA variants on average RULER-HARD-32K performance.
<table><tr><td>Model</td><td>Sparsity</td><td>SSA-Theory</td><td>SSA-Practical</td></tr><tr><td>Llama-3.1-8B</td><td>50×</td><td>81</td><td>83</td></tr><tr><td>Llama-3.1-8B</td><td>20×</td><td>88</td><td>88</td></tr><tr><td>Ministral-3-8B</td><td>50×</td><td>82</td><td>84</td></tr><tr><td>Ministral-3-8B</td><td>20×</td><td>89</td><td>88</td></tr></table>

Table 5: Average performance on RULER-HARD-32K across diferent models at 50× sparsity.
<table><tr><td>Model</td><td>Dense</td><td>PQCache</td><td>vAttn(PQ)</td><td>Quest</td><td>HashAttn</td><td>Double Sparsity</td><td>MagicPig</td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td>88.67</td><td>85.80</td><td>85.67</td><td>42.03</td><td></td><td>39.29</td><td>18.72</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>66.25</td><td>39.30</td><td>39.59</td><td>34.11</td><td></td><td>31.21</td><td>17.22</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>87.89</td><td>68.95</td><td>64.42</td><td>64.17</td><td>49.48</td><td>44.62</td><td>16.28</td></tr></table>

Table 6: Evaluation datasets and subsets used in our experiments.
<table><tr><td>Dataset</td><td># Samples Subsets</td><td></td></tr><tr><td>RULER-HARD</td><td>100</td><td>fwe, qa1, qa2, niah_multikey_2, niah_multikey_3, vt</td></tr><tr><td>LongBench</td><td>200</td><td>NrtvQA, Qasper, MF-en, HotpotQA, 2WikiMQA, Musique, DuReader, GovRep, QMSum, MultiNews, VCSum, TREC, Trivi- aQA, SAMSum, PCount, PR-en, LCC, RBP</td></tr><tr><td>LOFT</td><td>100</td><td>HotpotQA, Musique, NQ, Qampari, Quest</td></tr><tr><td>AIME</td><td>16</td><td></td></tr></table>

Table 7: Performance on RULER-HARD with Llama-3.1-8B.
<table><tr><td>Sparsity</td><td>Context</td><td>Method</td><td>FWE</td><td>VT</td><td>NM2</td><td>NM3</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td rowspan="10">50×</td><td rowspan="5">32K</td><td>Dense</td><td>93.33</td><td>99.2</td><td>100</td><td>100</td><td>88</td><td>52</td><td>88.76</td></tr><tr><td>PQ</td><td>77.67</td><td>95.4</td><td>67</td><td>71</td><td>76</td><td>48</td><td>72.51</td></tr><tr><td>vAttn (PQ)</td><td>83.67</td><td>94.6</td><td>52</td><td>58</td><td>66</td><td>46</td><td>66.71</td></tr><tr><td>Oracle-top-k</td><td>81.00</td><td>97.8</td><td>87</td><td>76</td><td>79</td><td>52</td><td>78.80</td></tr><tr><td>SSA</td><td>91.00</td><td>98.0</td><td>83</td><td>92</td><td>85</td><td>53</td><td>83.67</td></tr><tr><td rowspan="5">64K</td><td>Dense</td><td>86.33</td><td>98.8</td><td>98</td><td>99</td><td>84</td><td>53</td><td>86.52</td></tr><tr><td>PQ</td><td>80.00</td><td>92.2</td><td>65</td><td>50</td><td>71</td><td>47</td><td>67.53</td></tr><tr><td>vAttn (PQ)</td><td>83.33</td><td>94.0</td><td>40</td><td>68</td><td>67</td><td>45</td><td>66.22</td></tr><tr><td>Oracle-top-k</td><td>79.67</td><td>95.8</td><td>78</td><td>56</td><td>77</td><td>45</td><td>71.91</td></tr><tr><td>SSA</td><td>85.33</td><td>96.2</td><td>67</td><td>84</td><td>77</td><td>53</td><td>77.09</td></tr><tr><td rowspan="5">128K</td><td>Dense PQ</td><td>73.33</td><td>66.2</td><td>91</td><td>67</td><td>83</td><td>47</td><td>71.26</td></tr><tr><td></td><td>66.00</td><td>39.4</td><td>27</td><td>15</td><td>57</td><td>34</td><td>39.73</td></tr><tr><td>vAttn (PQ)</td><td>71.67</td><td>55.0</td><td>21</td><td>20</td><td>41</td><td>28</td><td>39.45</td></tr><tr><td>Oracle-top-k</td><td>61.00</td><td>66.6</td><td>71</td><td>32</td><td>66</td><td>40</td><td>56.10</td></tr><tr><td>SSA</td><td>73.00</td><td>54.2</td><td>30</td><td>30</td><td>59</td><td>39</td><td>47.53</td></tr><tr><td rowspan="9">20×</td><td rowspan="4">32K</td><td>PQ</td><td>87.33</td><td>98.6</td><td>97</td><td>93</td><td>84</td><td>52</td><td>85.32</td></tr><tr><td>vAttn (PQ)</td><td>93.33</td><td>99.4</td><td>97</td><td>99</td><td>84</td><td>52</td><td>87.46</td></tr><tr><td>Oracle-top-k</td><td>90.33</td><td>98.8</td><td>100</td><td>99</td><td>84</td><td>52</td><td>87.36</td></tr><tr><td>SSA</td><td>93.67</td><td>99.0</td><td>97</td><td>100</td><td>87</td><td>54</td><td>88.45</td></tr><tr><td rowspan="4">64K</td><td>PQ</td><td>86.33</td><td>96.2</td><td>91</td><td>91</td><td>77</td><td>55</td><td>82.76</td></tr><tr><td>vAttn (PQ)</td><td>86.00</td><td>98.4</td><td>87</td><td>98</td><td>79</td><td>57</td><td>84.23</td></tr><tr><td>Oracle-top-k</td><td>85.33</td><td>98.0</td><td>94</td><td>95</td><td>81</td><td>52</td><td>84.22</td></tr><tr><td>SSA</td><td>85.33</td><td>98.6</td><td>94</td><td>99</td><td>81</td><td>54</td><td>85.32</td></tr><tr><td></td><td>PQ</td><td>69.00</td><td>65.8</td><td>66</td><td>46</td><td>68</td><td>34</td><td>58.13</td></tr><tr><td rowspan="4">128K</td><td>vAttn (PQ)</td><td>73.33</td><td>63.6</td><td>61</td><td>51</td><td>69</td><td>39</td><td>59.49</td></tr><tr><td>Oracle-top-k</td><td>67.00</td><td>63.2</td><td>85</td><td>51</td><td>72</td><td>42</td><td>63.37</td></tr><tr><td>SSA</td><td>75.00</td><td>69.0</td><td>70</td><td>63</td><td></td><td>44</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>76</td><td></td><td>66.17</td></tr></table>

Table 8: Performance on RULER-HARD with Ministral-3-3B.
<table><tr><td>Sparsity</td><td>Context</td><td>Method</td><td>FWE</td><td>VT</td><td>NM2</td><td>NM3</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td rowspan="10">50×</td><td rowspan="5">32K</td><td>Dense</td><td>96.33</td><td>91.8</td><td>98</td><td>99</td><td>79</td><td>56</td><td>86.69</td></tr><tr><td>PQ</td><td>92.67</td><td>60.2</td><td>91</td><td>56</td><td>73</td><td>52</td><td>70.81</td></tr><tr><td>vAttn (PQ)</td><td>90.33</td><td>72.2</td><td>67</td><td>77</td><td>65</td><td>44</td><td>69.26</td></tr><tr><td>Oracle-top-k</td><td>93.67</td><td>62.2</td><td>99</td><td>55</td><td>75</td><td>52</td><td>72.81</td></tr><tr><td>SSA</td><td>95.67</td><td>88.6</td><td>92</td><td>90</td><td>72</td><td>54</td><td>82.05</td></tr><tr><td rowspan="5">64K</td><td>Dense</td><td>74.33</td><td>99.8</td><td>93</td><td>96</td><td>75</td><td>55</td><td>82.19</td></tr><tr><td>PQ</td><td>72.67</td><td>94.6</td><td>77</td><td>51</td><td>69</td><td>55</td><td>69.88</td></tr><tr><td>vAttn (PQ)</td><td>76.00</td><td>95.6</td><td>54</td><td>75</td><td>52</td><td>42</td><td>65.77</td></tr><tr><td>Oracle-top-k</td><td>75.33</td><td>98.2</td><td>89</td><td>68</td><td>68</td><td>52</td><td>75.09</td></tr><tr><td>SSA</td><td>75.00</td><td>99.2</td><td>76</td><td>87</td><td>67</td><td>50</td><td>75.70</td></tr><tr><td rowspan="5">128K</td><td>Dense PQ</td><td>92.33</td><td>98.6</td><td>95</td><td>87</td><td>66</td><td>50</td><td>81.49</td></tr><tr><td></td><td>77.67</td><td>99.2</td><td>66</td><td>43</td><td>66</td><td>47</td><td>66.48</td></tr><tr><td>vAttn (PQ)</td><td>89.33</td><td>98.0</td><td>54</td><td>42</td><td>54</td><td>41</td><td>63.06</td></tr><tr><td>Oracle-top-k</td><td>88.00</td><td>98.2</td><td>79</td><td>51</td><td>66</td><td>49</td><td>71.87</td></tr><tr><td>SSA</td><td>91.67</td><td>98.2</td><td>60</td><td>67</td><td>60</td><td>41</td><td>69.65</td></tr><tr><td rowspan="8">20×</td><td rowspan="4">32K</td><td>PQ</td><td>93.33</td><td>67.6</td><td>98</td><td>86</td><td>79</td><td>54</td><td>79.66</td></tr><tr><td>vAttn (PQ)</td><td>95.33</td><td>90.8</td><td>97</td><td>98</td><td>76</td><td>52</td><td>84.86</td></tr><tr><td>Oracle-top-k</td><td>96.00</td><td>73.0</td><td>99</td><td>93</td><td>75</td><td>52</td><td>81.33</td></tr><tr><td>SSA</td><td>95.67</td><td>94.2</td><td>99</td><td>97</td><td>79</td><td>56</td><td>86.81</td></tr><tr><td rowspan="4">64K</td><td>PQCache</td><td>77.00</td><td>98.4</td><td>87</td><td>84</td><td>70</td><td>56</td><td>78.73</td></tr><tr><td>vAttn (PQ)</td><td>73.00</td><td>99.8</td><td>86</td><td>96</td><td>73</td><td>53</td><td>80.13</td></tr><tr><td>Oracle</td><td>77.00</td><td>99.2</td><td>93</td><td>91</td><td>67</td><td>52</td><td>79.87</td></tr><tr><td>SSA</td><td>75.33</td><td>99.8</td><td>86</td><td>96</td><td>73</td><td>56</td><td>81.02</td></tr><tr><td rowspan="4">128K</td><td>PQ</td><td>88.33</td><td>99.2</td><td>71</td><td>65</td><td>70</td><td>50</td><td></td><td>73.92</td></tr><tr><td>vAttn (PQ)</td><td>89.33</td><td>98.4</td><td>59</td><td>77</td><td>65</td><td>44</td><td></td><td>72.12</td></tr><tr><td>Oracle-top-k</td><td>91.00</td><td>98.4</td><td>85</td><td>75</td><td>68</td><td></td><td>50</td><td>77.90</td></tr><tr><td>SSA</td><td>90.33</td><td>98.6</td><td>75</td><td>81</td><td>67</td><td>46</td><td></td><td>76.32</td></tr></table>

Table 9: Performance on LongBench with Qwen3.5-27B (10K-16K context).
<table><tr><td></td><td>Sparsity Method</td><td>NrtvQA</td><td>Qasper</td><td>MF-en</td><td>MF-zh</td><td>HotpotQA</td><td>2WikiMQA</td><td></td><td>Musique DuReader</td><td></td><td>GovRep</td><td>QMSum</td></tr><tr><td rowspan="5">50×</td><td>Dense</td><td>20.49</td><td>29.64</td><td>38.94</td><td>42.97</td><td>34.83</td><td>34.64</td><td>26.69</td><td></td><td>25.92</td><td>31.91 31.92</td><td>21.96</td></tr><tr><td>PQ</td><td>20.13</td><td>28.69</td><td>38.60</td><td>42.01</td><td>33.27</td><td>31.92</td><td>20.61</td><td>25.54</td><td></td><td></td><td>21.64</td></tr><tr><td>vAttn(PQ)</td><td>21.97</td><td>25.33</td><td>39.38</td><td>43.53</td><td>32.00</td><td>30.44</td><td>21.45</td><td>26.91</td><td></td><td>30.68</td><td>22.42</td></tr><tr><td>SSA</td><td>20.42</td><td>27.51</td><td>38.87</td><td>42.13</td><td>34.64</td><td>32.79</td><td>24.88</td><td>24.82</td><td></td><td>31.96</td><td>22.72</td></tr><tr><td>Oracle-top-k</td><td>20.51</td><td>28.97</td><td>38.22</td><td>42.12</td><td>34.79</td><td>33.77</td><td>22.50</td><td>25.76</td><td></td><td>32.86</td><td>22.52</td></tr><tr><td rowspan="4">20×</td><td>PQ</td><td>20.61</td><td>29.55</td><td>38.41</td><td>41.76</td><td>34.51</td><td>33.38</td><td>23.90</td><td>25.74</td><td></td><td>31.63</td><td>22.31</td></tr><tr><td>vAttn(PQ)</td><td>21.55</td><td>27.17</td><td>40.14</td><td>43.67</td><td>33.39</td><td>34.18</td><td>25.65</td><td>26.70</td><td></td><td>32.20</td><td>22.27</td></tr><tr><td>SSA</td><td>20.90</td><td>28.80</td><td>39.73</td><td>42.91</td><td>34.41</td><td>33.97</td><td>25.24</td><td>25.93</td><td></td><td>31.36</td><td>22.58</td></tr><tr><td>Oracle-top-k</td><td>20.74</td><td>29.28</td><td>38.68</td><td>42.87</td><td>34.90</td><td>34.57</td><td>23.46</td><td>25.82</td><td></td><td>32.77</td><td>21.97</td></tr><tr><td colspan="10">Sparsity Method MultiNews VCSUM</td></tr><tr><td rowspan="5">50×</td><td>Dense</td><td>23.33</td><td>14.50</td><td>TREC 72.50</td><td>TriviaQA 89.65</td><td>SAMSum 45.63</td><td>PCount 5.16</td><td>PRe-en 88.12</td><td>PRe-zh 57.33</td><td>LCC 67.96</td><td>RB-P 59.12</td><td>Avg. 41.56</td></tr><tr><td>PQ</td><td>24.37</td><td>14.50</td><td>71.25</td><td>89.68</td><td>45.92</td><td>6.37</td><td>87.40</td><td></td><td>56.60</td><td>66.65 54.97</td><td>40.60</td></tr><tr><td>vAttn(PQ)</td><td>23.17</td><td>14.62</td><td>71.25</td><td>89.04</td><td>43.76</td><td>6.85</td><td>85.25</td><td>59.21</td><td></td><td>66.46 53.98</td><td>40.39</td></tr><tr><td>SSA</td><td>23.08</td><td>14.35</td><td>72.50</td><td>89.77</td><td>44.32</td><td>5.16</td><td>86.25</td><td>59.69</td><td>64.55</td><td>58.01</td><td>40.92</td></tr><tr><td>Oracle-top-k</td><td>24.62</td><td>14.66</td><td>72.50</td><td>89.66</td><td>46.91</td><td>5.28</td><td>87.81</td><td>53.90</td><td>68.56</td><td>59.06</td><td>41.25</td></tr><tr><td rowspan="4">20×</td><td>PQ</td><td>24.20</td><td>14.79</td><td>70.00</td><td>89.46</td><td>45.83</td><td>5.10</td><td>89.06</td><td>54.21</td><td>67.87</td><td>57.24</td><td>40.98</td></tr><tr><td>vAttn(PQ)</td><td>23.41</td><td>15.05</td><td>75.00</td><td>90.52</td><td>45.12</td><td>4.68</td><td>86.88</td><td>57.23</td><td>66.02</td><td>58.56</td><td>41.47</td></tr><tr><td>SSA</td><td>24.01</td><td>14.67</td><td>72.50</td><td>90.18</td><td>46.93</td><td>4.76</td><td>88.12</td><td>56.81</td><td>66.85</td><td>60.47</td><td>41.56</td></tr><tr><td>Oracle-top-k</td><td>24.75</td><td>14.29</td><td>71.25</td><td>89.66</td><td>46.25</td><td>5.26</td><td>87.71</td><td>55.04</td><td>68.84</td><td>59.90</td><td>41.40</td></tr></table>

Table 10: Performance on RULER-HARD with Qwen3.5-4B at 50× sparsity.
<table><tr><td>Context</td><td>Method</td><td>FWE</td><td>VT</td><td>NM2</td><td>NM3</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td rowspan="5">32K</td><td>Dense</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>89</td><td>63</td><td>91.95</td></tr><tr><td>PQ</td><td>97.33</td><td>100</td><td>100</td><td>100</td><td>88</td><td>62</td><td>91.22</td></tr><tr><td>vAttn (PQ)</td><td>96.33</td><td>100</td><td>100</td><td>100</td><td>89</td><td>58</td><td>90.56</td></tr><tr><td>Oracle-top-k</td><td>99.33</td><td>100</td><td>100</td><td>100</td><td>88</td><td>61</td><td>91.39</td></tr><tr><td>SSA</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>89</td><td>60</td><td>91.45</td></tr><tr><td rowspan="5">64K</td><td>Dense</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>88</td><td>59</td><td>91.11</td></tr><tr><td>PQ</td><td>98.33</td><td>100</td><td>100</td><td>100</td><td>88</td><td>58</td><td>90.72</td></tr><tr><td>vAttn (PQ)</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>87</td><td>57</td><td>90.61</td></tr><tr><td>Oracle-top-k</td><td>99.33</td><td>100</td><td>100</td><td>100</td><td>86</td><td>57</td><td>90.39</td></tr><tr><td>SSA</td><td>99.00</td><td>100</td><td>100</td><td>100</td><td>88</td><td>58</td><td>90.83</td></tr><tr><td rowspan="5">128K</td><td>Dense</td><td>96.67</td><td>100</td><td>100</td><td>100</td><td>84</td><td>56</td><td>89.45</td></tr><tr><td>PQ</td><td>95.33</td><td>100</td><td>100</td><td>100</td><td>82</td><td>55</td><td>88.72</td></tr><tr><td>vAttn (PQ)</td><td>96.67</td><td>100</td><td>100</td><td>100</td><td>82</td><td>57</td><td>89.28</td></tr><tr><td>Oracle-top-k</td><td>97.33</td><td>100</td><td>100</td><td>100</td><td>82</td><td>57</td><td>89.39</td></tr><tr><td>SSA</td><td>97.00</td><td>100</td><td>100</td><td>100</td><td>82</td><td>58</td><td>89.50</td></tr></table>

Table 11: Performance on RULER-HARD with Qwen3.5-9B at 50× sparsity.
<table><tr><td>Context</td><td>Method</td><td>FWE</td><td>VT</td><td>NM2</td><td>NM3</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td rowspan="5">32K</td><td>Dense</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>93</td><td>68</td><td>93.45</td></tr><tr><td>PQ</td><td>94.00</td><td>100</td><td>100</td><td>100</td><td>91</td><td>67</td><td>92.00</td></tr><tr><td>vAttn (PQ)</td><td>98.00</td><td>100</td><td>100</td><td>100</td><td>89</td><td>63</td><td>91.67</td></tr><tr><td>Oracle-top-k</td><td>98.33</td><td>100</td><td>99</td><td>100</td><td>91</td><td>67</td><td>92.56</td></tr><tr><td>SSA</td><td>100.00</td><td>100</td><td>100</td><td>100</td><td>93</td><td>67</td><td>93.33</td></tr><tr><td rowspan="5">64K</td><td>Dense</td><td>99.33</td><td>100</td><td>100</td><td>100</td><td>88</td><td>66</td><td>92.22</td></tr><tr><td>PQ</td><td>95.67</td><td>99.8</td><td>100</td><td>100</td><td>86</td><td>65</td><td>91.08</td></tr><tr><td>vAttn (PQ)</td><td>98.00</td><td>99.8</td><td>100</td><td>99</td><td>88</td><td>63</td><td>91.30</td></tr><tr><td>Oracle-top-k</td><td>98.67</td><td>100</td><td>100</td><td>100</td><td>89</td><td>66</td><td>92.28</td></tr><tr><td>SSA</td><td>99.33</td><td>99.8</td><td>100</td><td>100</td><td>89</td><td>66</td><td>92.36</td></tr><tr><td rowspan="5">128K</td><td>Dense</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>87</td><td>61</td><td>91.28</td></tr><tr><td>PQ</td><td>97.67</td><td>100</td><td>98</td><td>100</td><td>88</td><td>62</td><td>90.95</td></tr><tr><td>vAttn (PQ)</td><td>98.67</td><td>100</td><td>98</td><td>100</td><td>84</td><td>63</td><td>90.61</td></tr><tr><td>Oracle-top-k</td><td>99.33</td><td>100</td><td>99</td><td>100</td><td>88</td><td>62</td><td>91.39</td></tr><tr><td>SSA</td><td>99.67</td><td>100</td><td>100</td><td>100</td><td>88</td><td>61</td><td>91.45</td></tr></table>

Table 12: Performance on RULER-HARD with Ministral-3-8B.
<table><tr><td>Context</td><td>Sparsity</td><td>Method</td><td>FWE</td><td>VT</td><td>NM2</td><td>NM3</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td rowspan="5">32K</td><td rowspan="5">50×</td><td>Dense</td><td>94.00</td><td>100.00</td><td>99.00</td><td>100.00</td><td>80.00</td><td>63.00</td><td>89.33</td></tr><tr><td>PQ</td><td>88.00</td><td>100.00</td><td>97.00</td><td>67.00</td><td>80.00</td><td>59.00</td><td>81.83</td></tr><tr><td>vAttn (PQ)</td><td>89.33</td><td>98.80</td><td>85.00</td><td>87.00</td><td>64.00</td><td>48.00</td><td>78.69</td></tr><tr><td>SSA</td><td>94.00</td><td>100.00</td><td>88.00</td><td>91.00</td><td>74.00</td><td>58.00</td><td>84.17</td></tr><tr><td>Oracle-top-k</td><td>90.67</td><td>100.00</td><td>99.00</td><td>90.00</td><td>81.00</td><td>59.00</td><td>86.61</td></tr><tr><td rowspan="5">64K</td><td rowspan="5">50×</td><td>Dense</td><td>81.00</td><td>99.80</td><td>100.00</td><td>100.00</td><td>75.00</td><td>61.00</td><td>86.13</td></tr><tr><td>PQ</td><td>84.33</td><td>99.80</td><td>89.00</td><td>71.00</td><td>68.00</td><td>56.00</td><td>78.02</td></tr><tr><td>vAttn (PQ)</td><td>81.33</td><td>99.80</td><td>71.00</td><td>87.00</td><td>63.00</td><td>49.00</td><td>75.19</td></tr><tr><td>SSA</td><td>81.67</td><td>99.80</td><td>80.00</td><td>95.00</td><td>69.00</td><td>54.00</td><td>79.91</td></tr><tr><td>Oracle-top-k</td><td>84.67</td><td>99.80</td><td>100.00</td><td>82.00</td><td>75.00</td><td>59.00</td><td>83.41</td></tr><tr><td rowspan="5">128K</td><td></td><td>Dense</td><td>92.67</td><td>100.00 100.00</td><td></td><td>90.00</td><td>62.00</td><td>44.00</td><td>81.45</td></tr><tr><td>50×</td><td>PQ</td><td>79.33</td><td>99.60</td><td>63.00</td><td>32.00</td><td>63.00</td><td>41.00</td><td>62.99</td></tr><tr><td>vAttn (PQ)</td><td></td><td>89.67 100.00</td><td>36.00</td><td>61.00</td><td></td><td>59.00</td><td>37.00</td><td>63.78</td></tr><tr><td>SSA</td><td>90.67</td><td>99.80</td><td>50.00</td><td>71.00</td><td>62.00</td><td>42.00</td><td></td><td>69.25</td></tr><tr><td>Oracle-top-k</td><td></td><td>85.00 100.00</td><td>93.00</td><td>67.00</td><td>64.00</td><td></td><td>46.00</td><td>75.83</td></tr><tr><td rowspan="4">32K</td><td rowspan="4">20×</td><td>PQ</td><td>92.33</td><td>100.00</td><td>99.00</td><td>97.00</td><td>83.00</td><td>60.00</td><td>88.56</td></tr><tr><td>vAttn (PQ)</td><td>95.00</td><td>100.00</td><td>96.00</td><td>100.00</td><td>78.00</td><td>61.00</td><td>88.33</td></tr><tr><td>SSA</td><td>93.33</td><td>100.00</td><td>97.00</td><td>99.00</td><td>77.00</td><td>63.00</td><td>88.22</td></tr><tr><td>Oracle-top-k</td><td>92.00</td><td>100.00</td><td>99.00</td><td>98.00</td><td>78.00</td><td>63.00</td><td>88.33</td></tr><tr><td rowspan="4">64K</td><td></td><td>PQ</td><td>83.67</td><td>99.80</td><td>99.00</td><td>97.00</td><td>74.00</td><td>60.00</td><td>85.58</td></tr><tr><td rowspan="4">20×</td><td>vAttn (PQ)</td><td>83.33</td><td>99.80</td><td>94.00</td><td>99.00</td><td>72.00</td><td>57.00</td><td></td><td>84.19</td></tr><tr><td>SSA</td><td>81.67</td><td>99.80</td><td>95.00</td><td>100.00</td><td>73.00</td><td>58.00</td><td></td><td>84.58</td></tr><tr><td>Oracle-top-k</td><td>83.00</td><td>99.80</td><td>100.00</td><td>95.00</td><td>71.00</td><td>60.00</td><td></td><td>84.80</td></tr><tr><td rowspan="4">20×</td><td>PQ</td><td>85.33</td><td>100.00</td><td>88.00</td><td>80.00</td><td>66.00</td><td></td><td>44.00</td><td>77.22</td></tr><tr><td>vAttn (PQ)</td><td>91.67</td><td>99.80</td><td>77.00</td><td>89.00</td><td>62.00</td><td></td><td>43.00</td><td>77.08</td></tr><tr><td>SSA</td><td>93.33</td><td>99.80</td><td>81.00</td><td>89.00</td><td>62.00</td><td></td><td>42.00</td><td>77.86</td></tr><tr><td>Oracle-top-k</td><td>90.67</td><td>100.00</td><td>99.00</td><td>84.00</td><td></td><td>64.00</td><td>44.00</td><td>80.28</td></tr></table>

Table 13: Ablation on the heavy-hitter fraction h for SSA on AIME 2025. A fraction h of the sparse attention budget is allocated to deterministic heavy hitters, while the remaining 1−h is allocated to importance sampling. Thus, h = 0 corresponds to pure importance sampling and h = 1 corresponds to PQCache (pure top-k retrieval).
<table><tr><td>Heavy Fraction h</td><td>20×</td><td>10×</td></tr><tr><td>0.0</td><td>40.2</td><td>42.5</td></tr><tr><td>0.2</td><td>61.0</td><td>63.8</td></tr><tr><td>0.4</td><td>63.5</td><td>64.0</td></tr><tr><td>0.6</td><td>62.7</td><td>62.7</td></tr><tr><td>0.8</td><td>63.8</td><td>63.1</td></tr><tr><td>1.0 (PQ)</td><td>49.4</td><td>53.8</td></tr></table>