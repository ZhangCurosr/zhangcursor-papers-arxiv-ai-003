# Virtual Head Attention

Xibo Ding<sup>1,2∗</sup>, Guoxia Wang<sup>1∗</sup>, Jinle Zeng<sup>1</sup>,   
Jiabin Yang<sup>1</sup>, Sirui Tao<sup>2</sup>, Chao Yang<sup>2</sup>, Dianhai Yu<sup>1,2†</sup> <sup>1</sup>Baidu Inc. <sup>2</sup>Peking University   
{dingxibo, wangguoxia, yudianhai}@baidu.com

## Abstract

The design of attention mechanisms in long-context large language models faces an “impossible triangle” among KV cache size, training and prefill FLOPs, and model quality. Grouped-Query Attention (GQA) achieves strong quality but incurs high KV cache overhead; alternative approaches such as Multi-head Latent Attention (MLA) and Multi-matrix Factorization Attention (MFA) can compress the cache, but MLA suffers from quality loss while MFA substantially increases training and prefill compute. We propose Virtual Head Attention (VHA), which simultaneously improves all three vertices through two lightweight linear operations: Q Premix applies a near-identity transformation to queries within each KV group to recover virtual head diversity from halved physical query heads, and Linear Postmix fuses inter-head features via a low-rank I + AB<sup>⊤</sup> residual structure. Since VHA introduces no nonlinear operations, Premix can be folded into the query projection and Postmix into the output projection at inference time, reducing the model to a standard GQA-2 architecture that directly reuses all existing GQA inference optimizations. With only two KV heads, the group size in large models readily reaches 64, fully exploiting Tensor Core parallelism during decoding. Across four scales (0.6B, 1.7B, 4B dense, and 8B-A1B MoE) with 4K-context 50Btoken pretraining, VHA consistently surpasses the standard GQA baseline at every scale (+0.05% to +0.81%), while reducing KV cache by 4× vs. GQA-8 at dense scales and 2× vs. GQA-4 at the MoE scale, and lowering training and prefill FLOPs by 7.9% (1.7B, S=4096). On an industrial 30B-A3B MoE model trained on 500B tokens at 8K context, VHA outperforms both MLA and GQA-4 by +0.70 / +2.60 % on a 12-benchmark base-model evaluation suite; in a separate matched 10B-A1B MoE comparison after long-context mid-training, VHA attains an average RULER score of 84.4 over 8K–128K versus 73.0 for GQA, while using two rather than four KV heads in the full-attention layers. The source code is available at https://github.com/PaddlePaddle/VHA.

## 1 Introduction

Large language models are rapidly scaling toward longer contexts, with context windows expanding from 2K tokens to 128K and beyond [Grattafiori et al., 2024, Yang et al., 2024]. As sequence lengths grow, the attention mechanism [Vaswani et al., 2017] becomes a central efficiency bottleneck in both inference and training. During inference, the key-value (KV) cache grows linearly with sequence length, consuming substantial GPU memory and constraining both serving concurrency and generation latency. During training and prefill, the FLOPs of attention computation scale quadratically with sequence length. Since the design of the attention mechanism jointly determines KV cache size, computational cost, and model expressiveness, it is a critical factor governing the overall efficiency and quality of large language models.

The design of existing attention mechanisms faces three mutually constraining objectives, forming an “impossible triangle”: KV cache size, training and prefill FLOPs, and model quality. Consider the widely adopted Grouped-Query Attention (GQA) [Ainslie et al., 2023]: GQA with 8 KV heads (GQA 8) achieves strong quality, but incurs a KV cache of 4 KB per layer per token (in BF16). Directly reducing the number of KV heads to 2 (GQA-2) compresses the cache to 1 KB, but dramatically enlarges the group size so that far more query heads must share the same key-value pairs, causing attention diversity to drop sharply and resulting in a 1.00% quality degradation at the 1.7B scale. Multi-head Latent Attention (MLA) [Liu et al., 2024] applies low-rank compression to reduce KV cache to 1.12 KB, but yields the lowest quality across multiple model scales. Multi-matrix Factorization Attention (MFA) [Hu et al., 2025] reduces the cache to 1 KB by enlarging the head dimension, achieving quality close to GQA-8, but at the cost of a 49.8% increase in training and prefill FLOPs (1.7B, S=4096). No existing approach achieves a satisfactory balance across all three dimensions.

We propose Virtual Head Attention (VHA), which simultaneously improves all three vertices of the impossible triangle through two lightweight linear operations. The core idea of VHA is to halve the number of physical query heads while setting the number of KV heads to only 2. Q Premix then applies a near-identity linear transformation to queries within each KV group, expanding the reduced physical query heads into virtual query heads to recover attention diversity. Linear Postmix applies a low-rank identity-residual transformation (I + AB<sup>⊤</sup>, r=H<sup>′</sup> /2, e.g., r=4 at 1.7B) on the attention output, fusing cross-group inter-head features. Since VHA introduces no nonlinear operations, Premix can be directly folded into the query projection matrix and Postmix into the output projection matrix at inference time, reducing the model to a standard GQA-2 architecture that reuses all existing GQA inference optimizations in mainstream serving frameworks such as vLLM [Kwon et al., 2023] and SGLang [Zheng et al., 2024]. Moreover, with only 2 KV heads, the group size in large models readily reaches 64, fully exploiting Tensor Core parallelism during the decoding phase.

Across four model scales (0.6B, 1.7B, 4B dense, and 8B-A1B MoE) with 4K-context 50B-token pretraining, VHA consistently surpasses the standard GQA baseline at every scale (+0.05% to +0.81%), while reducing KV cache by 4× vs. GQA-8 at dense scales and 2× vs. GQA-4 at the MoE scale, and lowering training and prefill FLOPs by 7.9% (1.7B, S=4096). On an industrial 30B A3B MoE model trained on 500B tokens at 8K context, VHA outperforms both MLA and GQA-4 by +0.70 / +2.60 % on a 12-benchmark base-model evaluation suite, demonstrating scalability to industrial model scale. A separate matched 10B-A1B MoE comparison further extends this advantage to 128K context under more aggressive KV compression. Our main contributions are as follows:

1. We propose VHA, employing Q Premix and Linear Postmix, two lightweight linear operations that simultaneously reduce KV cache and training/prefill FLOPs while improving model quality.

2. VHA uses only linear operations and can be fully absorbed into a standard GQA-2 model at inference time, reusing existing serving frameworks such as vLLM and SGLang with zero additional overhead.

3. Direct inspection of the trained Premix and Postmix matrices reveals two complementary structures (group-conditioned query specialization and structured low-rank cross-head communication) that jointly close the gap of the doubly-compressed configuration; component ablations confirm that neither module is sufficient on its own.

4. Comprehensive evaluations across five scales (0.6B to 30B-A3B MoE) against MHA, GQA, MFA, and MLA establish VHA as the best attention mechanism that is Pareto-optimal on KV cache, training/prefill FLOPs, and downstream quality simultaneously; matched long-context evaluations through 128K further validate this advantage under aggressive KV compression.

## 2 Related Work

## 2.1 KV Cache Efficient Attention

Standard Multi-Head Attention (MHA) [Vaswani et al., 2017] maintains independent key and value projections per head, causing the KV cache to grow linearly with the number of heads. Multi-Query Attention (MQA) [Shazeer, 2019] shares a single KV head across all query heads, drastically reducing the cache at the cost of quality. Grouped-Query Attention (GQA) [Ainslie et al., 2023] interpolates between MHA and MQA by allowing multiple query heads to share a group of KV heads, and has been widely adopted in LLaMA 2/3 [Touvron et al., 2023, Grattafiori et al., 2024] and Qwen2 [Yang et al., 2024].

Several recent designs further compress the KV cache. Multi-head Latent Attention (MLA) [Liu et al., 2024] projects KV into a shared low-rank latent space and absorbs the up-projection matrices into Q and O at inference, but requires custom kernels and a non-standard inference pipeline. Multi-matrix Factorization Attention (MFA) [Hu et al., 2025] enlarges the head dimension while reducing the number of KV heads, preserving model capacity under a tight cache budget but substantially increasing training and prefill FLOPs. Tensor Product Attention (TPA) [Zhang et al., 2025] decomposes QKV into context-dependent tensor products. Grouped-head Latent Attention (GTA) [Sun et al., 2025] extends MLA’s global latent space to a group-wise design. Simulated Attention Score (SAS) [Zheng et al., 2025] projects a small number of low-dimensional KV heads into higher-dimensional spaces to simulate more attention heads, an idea related to VHA’s virtual head expansion; however, SAS operates on the KV side whereas VHA virtualizes on the query side. Low-Rank Key-Value Attention (LRKV) [O’Neill et al., 2026] shares a full-rank KV basis across heads and restores head-specific flexibility through low-rank residuals, a form reminiscent of VHA’s Postmix $( I + A B ^ { \top } ) ;$ yet LRKV acts in KV space and its additional structure remains at inference time. Cross-Layer Attention (CLA) [Brandon et al., 2024] and YOCO [Sun et al., 2024] reduce memory by sharing or reusing KV caches across layers.

Unlike all of the above, VHA employs a richer training-time parameterization that merges exactly back into standard GQA at deployment, requiring no custom kernels or specialized inference procedures.

## 2.2 Inter-Head Communication and Virtual Heads

Talking-Heads Attention [Shazeer et al., 2020] applies learned linear projections across heads before and after softmax, pioneering inter-head communication. Cordonnier et al. [2020] analyze multi-head attention from a collaborative perspective, revealing substantial redundancy among heads. More recently, Explicit Multi-head Attention (MEA) [Peng et al., 2026] introduces a head-level linear composition module for explicit cross-head interaction, and Interleaved Head Attention [Duvvuri et al., 2026] interleaves head representations to enable information exchange. In contrast to these methods, VHA does not mix across heads in the attention weight space; instead, it virtualizes in the query space via Premix and fuses inter-head features in the output space via a low-rank Postmix, with all operations being strictly linear and fully absorbable at inference.

## 2.3 Reparameterization

RepVGG [Ding et al., 2021] uses a multi-branch training architecture $( 3 { \times } 3 + 1 { \times } 1 +$ identity) that merges into a single convolution at inference, demonstrating that decoupled training-time parameterization provides implicit regularization at zero inference cost. LoRA [Hu et al., 2022] performs parameter-efficient fine-tuning via low-rank decomposition $( W + B A )$ VHA extends this reparameterization paradigm from a single weight matrix to the head dimension of attention: Premix structurally factorizes the per-virtual-head query projections $( \widetilde { W } _ { Q } ^ { ( i , g ) } = W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { p r e m i x } } )$ while Postmix introduces a low-rank identity-residual mixing on the head axis $( I + A B ^ { \top } )$ . Both modules are mathematically equivalent in the forward pass to a standard GQA configuration, yet the decoupled parameterization yields distinct optimization dynamics during training and merges back into a standard GQA-2 architecture at inference with zero overhead.

## 3 Method: Virtual Head Attention

## 3.1 Preliminaries

Grouped-Query Attention (GQA) [Ainslie et al., 2023] partitions $H _ { q }$ query heads into $H _ { k v }$ groups, where heads within each group share a common key-value pair. Given input $\boldsymbol { X } \in \mathbb { R } ^ { S \times h }$ , GQA computes $Q = X W _ { Q } , K \stackrel { \cdot } { = } X W _ { K } , V = X W _ { V }$ , where $W _ { Q } ^ { \bullet } \in \mathbb { R } ^ { h \times H _ { q } d } , \dot { W _ { K } } \in \mathbb { R } ^ { h \times H _ { k v } d } , \dot { W _ { V } } \in$ $\mathbb { R } ^ { h \times H _ { k v } d }$ . Each query head attends to its group’s key-value pair via scaled dot-product attention [Vaswani et al., 2017], and all head outputs are concatenated and projected by $W _ { O } \in \overset { \bullet } { \mathbb { R } } ^ { H _ { q } d \times h }$ . During inference, K and V are cached, consuming $4 H _ { k v } d$ bytes per layer per token in BF16. Reducing $H _ { k v }$ compresses the cache but enlarges the group size $G = H _ { q } / H _ { k v }$ , forcing more query heads to share the same key-value pair and degrading attention diversity.

![](images/7a1b039cf62aee21ac8031b293682d02cf2d6e28715a7f5c4176666fd9f7c327.jpg)  
Figure 1: Overview of Virtual Head Attention. (a) VHA improves the trade-off among KV cache size, training/prefill FLOPs, and model quality by combining Q Premix and Linear Postmix during training, while absorbing both linear modules into standard GQA projections at inference. (b) Schematic of the KV cache–FLOPs–quality impossible triangle faced by existing attention designs. (c) Method comparison at 1.7B and 4B scales, plotting KV cache size against training FLOPs, with bubble area proportional to 8-task average accuracy.

## 3.2 Overall Design

VHA halves the number of physical query heads while setting $H _ { k v } = 2$ , then recovers model expressiveness through two lightweight linear operations. Specifically, $H _ { q } ^ { \prime } = H _ { q } / 2$ physical query heads and $H _ { k v } = 2 \mathrm { K V }$ heads yield $H _ { v } = H _ { k v } \times H _ { q } ^ { \prime } = H _ { q }$ virtual heads (matching the original GQA query head count), with $W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h }$ identical in size to the original GQA output projection. The resulting KV cache is $4 \times 2 \times d$ bytes per layer per token, 4× smaller than GQA-8 at dense scales and 2× smaller than GQA-4 at MoE scales.

On top of this configuration, VHA introduces Q Premix, which expands $H _ { q } ^ { \prime }$ physical query heads into $H _ { v }$ virtual query heads to restore attention diversity, and Linear Postmix, which applies a low-rank identity-residual transformation on the attention output to fuse inter-head features.

Inference absorption principle. VHA uses only linear operations (no nonlinear activations), so that at inference time Premix can be folded into $W _ { Q }$ and Postmix into $W _ { O }$ , reducing the entire model to a standard GQA-2 architecture that directly reuses all existing GQA optimizations in frameworks such as vLLM [Kwon et al., 2023] and SGLang [Zheng et al., 2024].

## 3.3 Q Premix: Virtual Query Head Expansion

Q Premix applies a near-identity linear transformation to physical query heads within each KV group, expanding them into virtual query heads. For KV group $\bar { g } \in \{ 1 , \dotsc , \bar { H } _ { k v } \}$ , each physical query head $Q _ { i } \in \mathbb { R } ^ { S \times d } , i \in \left\{ 1 , \dots , H _ { q } ^ { \prime } \right\}$ , is transformed as:

$$
\tilde { Q } _ { i , g } = Q _ { i } W _ { g } ^ { \mathrm { p r e m i x } } , \quad W _ { g } ^ { \mathrm { p r e m i x } } \in \mathbb R ^ { d \times d }\tag{1}
$$

where $W _ { g } ^ { \mathrm { p r e m i x } }$ is a per-group transformation matrix. We initialize $W _ { q } ^ { \mathrm { p r e m i x } } = I + \alpha \cdot \epsilon , \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with a small scaling factor α. This near-identity initialization ensures that virtual query heads start as slight perturbations of the physical query heads, preserving training stability at the early stage. As training progresses, each $W _ { g } ^ { \mathrm { p r e m i x } }$ gradually deviates from the identity to produce increasingly diverse attention patterns across virtual heads. Through this operation, $H _ { q } ^ { \prime }$ physical query heads are expanded into $H _ { v } = H _ { k v } \times H _ { q } ^ { \prime }$ virtual query heads.

Structured factorization view. Premix is mathematically equivalent in the forward pass to a standard $H _ { v }$ -head GQA whose query projections are structurally factorized. Specifically, defining

$$
\widetilde { \boldsymbol { W } } _ { \boldsymbol { Q } } ^ { ( i , { g } ) } = \boldsymbol { W } _ { \boldsymbol { Q } } ^ { ( i ) } \boldsymbol { W } _ { \boldsymbol { g } } ^ { \mathrm { p r e m i x } } \in \mathbb { R } ^ { \boldsymbol { h } \times \boldsymbol { d } } ,\tag{2}
$$

the entire Premix-then-attend computation reduces, for any input X and any parameter values, to

$$
\mathrm { s o f t m a x } \left( \tilde { Q } _ { i , g } K _ { g } ^ { \top } / \sqrt { d } \right) V _ { g } = \mathrm { s o f t m a x } \left( X \widetilde { W } _ { Q } ^ { ( i , g ) } K _ { g } ^ { \top } / \sqrt { d } \right) V _ { g } .\tag{3}
$$

Thus VHA is a standard GQA with $H _ { v } = H _ { k v } H _ { q } ^ { \prime }$ virtual query heads and $H _ { k v }$ KV heads, where the per-virtual-head query projections are constrained to lie on the structured low-dimensional manifold $\{ W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { p r e m i x } } \}$

Why this factorization is useful. The structured parameterization brings two benefits over an unconstrained $H _ { v }$ -head GQA. First, it is parameter- and compute-efficient: the Q path uses $H _ { q } ^ { \prime } h d +$ $H _ { k v } d ^ { 2 }$ parameters and a comparable number of FL ${ \mathrm { . O P s } } ,$ , versus $H _ { v } h d$ for a fully decoupled $H _ { v }$ -head Q projection. At the 1.7B scale, this yields a roughly 50% reduction in Q-projection parameters and FLOPs (the primary source of VHA’s training-FLOPs reduction). Second, although the forward pass is algebraically equivalent to training a single ${ \widetilde { W } } _ { Q } ^ { ( i , g ) }$ , the gradients of $W _ { Q } ^ { ( i ) }$ and $W _ { g } ^ { \mathrm { p r e m i x } }$ are coupled across all groups and physical heads, yielding a different optimization landscape and providing implicit regularization analogous to LoRA [Hu et al., 2022] and RepVGG [Ding et al., 2021] (see Appendix G for a formal analysis).

Inference-time absorption. Once training is complete and parameters are frozen, for each virtual head $( i , g )$ we precompute $\widetilde { W } _ { Q } ^ { ( i , g ) } = W _ { Q } ^ { ( i ) }$ · W<sup>premix</sup>, after which VHA reduces to a standard GQA-2 with $H _ { v }$ independent query projections and $\check { H _ { k v } } = 2 \mathrm { K V }$ heads.

## 3.4 Linear Postmix: Low-Rank Identity-Residual Mixing

While Premix restores virtual head diversity within each KV group, the outputs across different groups still lack cross-group information exchange. Linear Postmix addresses this through a low-rank identity-residual structure.

Let $Y \in \mathbb { R } ^ { S \times H _ { v } \times d }$ denote the attention output. Postmix computes:

$$
\boldsymbol { Y } ^ { \prime } = \boldsymbol { Y } + B \boldsymbol { A } ^ { \top } \boldsymbol { Y } = \left( \boldsymbol { I } + B \boldsymbol { A } ^ { \top } \right) \boldsymbol { Y }\tag{4}
$$

where $A , B \in \mathbb { R } ^ { H _ { v } \times r }$ and $r$ is the bottleneck rank (default $r = H _ { a } ^ { \prime } / 2 , { \mathrm e . g . , r { = } } 4$ at 1.7B). The operation acts along the head dimension independently for each token and head-dimension channel, applying the transformation $I + B A ^ { \top } \in \mathbb R ^ { H _ { v } ^ { \bot } \times H _ { v } }$ to the head axis.

Initialization. A is initialized with small random values and B with zeros, ensuring $B A ^ { \top } = 0$ at the start of training so that Postmix acts as the identity and does not disturb the initial training dynamics. As training progresses, B gradually learns effective cross-group mixing patterns.

Linear redundancy and reparameterization benefit. Mathematically, Postmix can be absorbed into $W _ { O }$ . Using the row-major Kronecker identity $\operatorname { v e c } _ { \operatorname { r o w } } ( M Y ) ^ { \top } = \operatorname { v e c } _ { \operatorname { r o w } } ^ { - } ( Y ) ^ { \top } ( M ^ { \top } \otimes I _ { d } )$ [Roger and Charles, 1994]:

$$
\mathrm { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } \left( \left( I _ { H _ { v } } + A B ^ { \top } \right) \otimes I _ { d } \right) W _ { O } = \mathrm { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } W _ { O } ^ { \prime }\tag{5}
$$

where $W _ { O } ^ { \prime } = \left( \left( I _ { H _ { v } } + A B ^ { \top } \right) \otimes I _ { d } \right) W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h }$ . Thus, linear Postmix is theoretically redundant (see Appendix H for the complete derivation). However, ablation experiments (Section 4.4) show that adding Postmix on top of Premix yields a +1.19 % improvement at 1.7B, analogous to the implicit regularization observed in RepVGG [Ding et al., 2021] and the low-rank decomposition in LoRA [Hu et al., 2022]: the separated gradient paths of A and B provide a different optimization landscape, while the low-rank constraint $\bar { ( r } \ll \bar { H _ { v } ) }$ acts as an additional regularizer preventing cross-head mixing from overfitting.

Inference-time absorption. At inference, we precompute $W _ { O } ^ { \prime } = \big ( ( I _ { H _ { v } } + A B ^ { \top } ) \otimes I _ { d } \big ) W _ { O }$ , after which VHA reduces to standard GQA-2. Algorithm 1 (Appendix F) summarizes the complete training forward pass and the one-time inference-absorption procedure.

## 3.5 Efficiency Analysis

Training and prefill FLOPs. By halving the physical $H _ { q }$ , VHA directly reduces Q projection FLOPs. The additional cost of Premix $( 2 H _ { k v } H _ { a } ^ { \prime } d ^ { 2 } )$ and Postmix $( 4 H _ { v } r d )$ is negligible compared to the FLOPs saved by halving the Q projection. Table 5 quantifies this trade-off in detail across all four model scales (S=4096); VHA emerges as the best Pareto-optimal point on KV cache and FLOPs at every scale (Section 4.5).

Table 1: Pretraining accuracy across four model scales. We report zero-shot accuracy (%) on eight standard multiple-choice benchmarks. CACHE is KV cache per layer per token in BF16 (KB). Bold marks the best per scale per column; ties are bolded jointly. VHA achieves the highest average accuracy at every scale while using the smallest KV cache. Per-method full configurations and benchmark citations are in Appendix D; per-method training-FLOPs comparison is in Table 5.
<table><tr><td>Scale</td><td>Attn</td><td>Q-K-V</td><td>Cache</td><td>ARC-C</td><td>BoolQ</td><td>HSwag</td><td>OBQA</td><td>PIQA</td><td>Wino</td><td>SciQ</td><td>SIQA</td><td>Avg</td></tr><tr><td rowspan="5">0.6B</td><td>MHA</td><td>12-12-12</td><td>6.00</td><td>29.10</td><td>60.40</td><td>44.88</td><td>33.20</td><td>68.66</td><td>51.85</td><td>74.50</td><td>39.15</td><td>50.22</td></tr><tr><td>GQA</td><td>16-8-8</td><td>4.00</td><td>29.86</td><td>58.01</td><td>44.77</td><td>35.80</td><td>68.17</td><td>52.33</td><td>75.70</td><td>38.69</td><td>50.42</td></tr><tr><td>MLA</td><td>14-14-14</td><td>1.12</td><td>29.27</td><td>55.96</td><td>43.20</td><td>33.80</td><td>68.23</td><td>50.67</td><td>76.40</td><td>38.59</td><td>49.52</td></tr><tr><td>MFA</td><td>17-1-1</td><td>1.00</td><td>30.80</td><td>60.18</td><td>45.47</td><td>32.80</td><td>69.04</td><td>54.70</td><td>73.20</td><td>39.05</td><td>50.66</td></tr><tr><td>VHA</td><td>8-2-2</td><td>1.00</td><td>31.91</td><td>61.19</td><td>45.26</td><td>35.60</td><td>68.44</td><td>50.99</td><td>75.60</td><td>39.36</td><td>51.04</td></tr><tr><td rowspan="5">1.7B</td><td>MHA</td><td>12-12-12</td><td>6.00</td><td>35.24</td><td>58.53</td><td>51.86</td><td>35.20</td><td>71.33</td><td>54.06</td><td>80.30</td><td>39.36</td><td>53.24</td></tr><tr><td>GQA</td><td>16-8-8</td><td>4.00</td><td>35.15</td><td>62.11</td><td>51.98</td><td>35.60</td><td>71.71</td><td>55.25</td><td>81.10</td><td>39.82</td><td>54.09</td></tr><tr><td>MLA</td><td>12-12-12</td><td>1.12</td><td>31.57</td><td>61.31</td><td>49.22</td><td>36.20</td><td>70.51</td><td>54.22</td><td>77.60</td><td>39.30</td><td>52.49</td></tr><tr><td>MFA</td><td>19-1-1</td><td>1.00</td><td>34.47</td><td>59.45</td><td>52.16</td><td>37.80</td><td>71.82</td><td>54.85</td><td>81.50</td><td>40.43</td><td>54.06</td></tr><tr><td>VHA</td><td>8-2-2</td><td>1.00</td><td>35.92</td><td>60.64</td><td>51.71</td><td>39.00</td><td>71.38</td><td>54.70</td><td>81.70</td><td>40.94</td><td>54.50</td></tr><tr><td rowspan="5">4B</td><td>MHA</td><td>20-20-20</td><td>10.00</td><td>38.31</td><td>60.21</td><td>57.52</td><td>39.60</td><td>73.88</td><td>56.99</td><td>85.40</td><td>41.56</td><td>56.68</td></tr><tr><td>GQA</td><td>32-8-8</td><td>4.00</td><td>39.16</td><td>62.32</td><td>58.39</td><td>39.40</td><td>73.83</td><td>57.06</td><td>83.80</td><td>41.20</td><td>56.90</td></tr><tr><td>MLA</td><td>28-28-28</td><td>1.12</td><td>37.20</td><td>61.65</td><td>56.55</td><td>38.80</td><td>73.29</td><td>56.75</td><td>84.90</td><td>41.86</td><td>56.38</td></tr><tr><td>MFA</td><td>33-1-1</td><td>1.00</td><td>38.40</td><td>62.20</td><td>57.67</td><td>39.80</td><td>73.88</td><td>57.70</td><td>83.60</td><td>42.27</td><td>56.94</td></tr><tr><td>VHA</td><td>16-2-2</td><td>1.00</td><td>38.91</td><td>64.74</td><td>58.67</td><td>39.20</td><td>74.43</td><td>58.80</td><td>84.60</td><td>42.34</td><td>57.71</td></tr><tr><td rowspan="5">8B-A1B</td><td>MHA</td><td>20-20-20</td><td>10.00</td><td>35.67</td><td>59.63</td><td>53.98</td><td>37.40</td><td>72.25</td><td>53.04</td><td>82.70</td><td>42.12</td><td>54.60</td></tr><tr><td>GQA</td><td>32-4-4</td><td>2.00</td><td>34.73</td><td>61.53</td><td>54.03</td><td>38.60</td><td>71.55</td><td>55.09</td><td>82.60</td><td>40.74</td><td>54.86</td></tr><tr><td>MLA</td><td>28-28-28</td><td>1.12</td><td>33.11</td><td>60.86</td><td>52.29</td><td>36.60</td><td>72.63</td><td>53.28</td><td>82.00</td><td>39.56</td><td>53.79</td></tr><tr><td>MFA</td><td>33-1-1</td><td>1.00</td><td>35.07</td><td>62.48</td><td>54.08</td><td>36.60</td><td>72.80</td><td>55.17</td><td>81.80</td><td>40.48</td><td>54.81</td></tr><tr><td>VHA</td><td>16-2-2</td><td>1.00</td><td>36.69</td><td>62.17</td><td>53.78</td><td>37.00</td><td>71.71</td><td>54.06</td><td>82.40</td><td>41.45</td><td>54.91</td></tr></table>

Decoding inference. Autoregressive decoding is memory-bound [Pope et al., 2023], so KV cache size directly governs latency and serving concurrency. With KV cache reduced by 4× vs. GQA-8 (dense) or 2× vs. GQA-4 (MoE), VHA achieves a corresponding 2–4× theoretical speedup in memory-bound decoding. Since Premix and Postmix are fully absorbed at inference, VHA’s decoding cost is identical to standard GQA-2.

## 4 Experiments

## 4.1 Experimental Setup

Baselines. We compare VHA with MHA, GQA, MFA, and MLA across all scales. Following common practice, GQA uses 8 KV heads at dense scales (0.6B/1.7B/4B) and 4 KV heads at MoE scales (8B-A1B and 30B-A3B). VHA halves the physical query heads relative to GQA and uses H =2, yielding 4× KV cache compression at dense scales and 2× at MoE scales.

Small-scale pretraining. We pretrain Qwen3-architecture models at four scales (dense at 0.6B, 1.7B, and 4B parameters, plus an 8B-A1B MoE) on 50B tokens of FineWeb-Edu with BF16 mixed precision and a 4K context length. We report zero-shot accuracy on eight standard benchmarks.

Industrial-scale pretraining. We pretrain an ERNIE-style 30B-A3B MoE on an internal multidomain pretraining corpus (web text, books, code, and multilingual data) with FP8 mixed precision on 500B tokens at 8K context. We evaluate on 12 standard base-model benchmarks spanning general knowledge, math & STEM, coding, and multilingual capabilities. Full model configurations, benchmark details, and training hyperparameters are in Appendix D.

Long-context mid-training. We additionally compare VHA and GQA on a matched MiMo-2.5- Flash-style 10B-A1B MoE backbone that interleaves full and sliding-window attention at a 1:3 ratio. The two models share the backbone, data, optimization, and sliding-window layers; only the full-attention layers differ, using two KV heads for VHA and four for GQA. We evaluate through 128K, with the stage-wise recipe and benchmark definitions reported in Appendix A.

Table 2: Recovering from a doubly-compressed shape (1.7B). GQA 16-8-8 is the standard GQA-8 baseline; GQA 16-2-2 reduces only $H _ { k v }$ to obtain a 1 KB KV cache; GQA 8-2-2 additionally halves the physical query heads to match VHA’s shape but lacks Premix and Postmix. ∆ is the gap to the GQA 16-8-8 baseline.
<table><tr><td>Method</td><td>Cache</td><td>ARC-C</td><td>BoolQ</td><td>HSwag</td><td>OBQA</td><td>PIQA</td><td>Wino</td><td>SciQ</td><td>SIQA</td><td>Avg</td><td>∆</td></tr><tr><td>GQA 16-8-8</td><td>4.00</td><td>35.15</td><td>62.11</td><td>51.98</td><td>35.60</td><td>71.71</td><td>55.25</td><td>81.10</td><td>39.82</td><td>54.09</td><td></td></tr><tr><td>GQA 16-2-2</td><td>1.00</td><td>33.19</td><td>58.96</td><td>50.92</td><td>36.00</td><td>71.27</td><td>54.22</td><td>79.60</td><td>40.53</td><td>53.09</td><td>-1.00</td></tr><tr><td>GQA 8-2-2</td><td>1.00</td><td>33.02</td><td>59.11</td><td>49.98</td><td>34.00</td><td>71.44</td><td>53.28</td><td>77.80</td><td>40.43</td><td>52.38</td><td>-1.71</td></tr><tr><td>VHA 8-2-2</td><td>1.00</td><td>35.92</td><td>60.64</td><td>51.71</td><td>39.00</td><td>71.38</td><td>54.70</td><td>81.70</td><td>40.94</td><td>54.50</td><td>+0.41</td></tr></table>

Table 3: Pretraining accuracy at the 30B-A3B MoE scale. We report accuracy (%) averaged within four task categories (General, Math & STEM, Coding, Multilingual) over 12 base-model benchmarks; AVG is the macro-average over the four task categories. CACHE is KV cache per layer per token in BF16 (KB). Bold marks the best per column. Per-benchmark scores and benchmark citations are in Appendix C.
<table><tr><td>Method</td><td>Q-K-V</td><td>Cache</td><td>General</td><td>Math &amp; STEM</td><td>Coding</td><td>Multilingual</td><td>Avg</td></tr><tr><td>GQA</td><td>48-4-4</td><td>2.00</td><td>52.86</td><td>43.09</td><td>51.85</td><td>43.26</td><td>47.76</td></tr><tr><td>MLA</td><td>28-28-28</td><td>1.12</td><td>55.57</td><td>43.49</td><td>54.37</td><td>45.20</td><td>49.66</td></tr><tr><td>VHA</td><td>24-2-2</td><td>1.00</td><td>56.25</td><td>46.40</td><td>53.23</td><td>45.56</td><td>50.36</td></tr></table>

## 4.2 Main Results

Cross-scale consistency. VHA achieves the highest average accuracy at every scale, surpassing the GQA baseline by +0.05% to +0.81% while using the smallest KV cache: 1.00 KB per layer per token, tied with MFA, 4× smaller than GQA-8 at dense scales and 2× smaller than GQA-4 at the MoE scale. The advantage holds across both dense and MoE architectures. A separate matched-harness study at 0.6B confirms the same ordering across repeated runs, with VHA reaching 50.40±0.07 (n=4) versus GQA’s 50.00±0.34 (n=3); full results are in Appendix B.

Trade-offs against each baseline. At equal KV cache (1.00 KB), VHA dominates MFA at every scale (+0.10% to +0.77%) while reducing training and prefill FLOPs by 7.9%, compared with MFA’s +49.8% increase at the 1.7B scale (S=4096; Table 5). Full-cache MHA falls below GQA-8 at every dense scale despite using a ≥ 1.5× larger cache, confirming that grouping the queries is essentially free. MLA underperforms at every scale, suggesting that latent KV compression is less effective than head-level virtualization at the 50B-token training budget used here. Per-method training loss curves at all four scales are reported in Appendix E.

Recovering from a doubly-compressed shape. Reducing GQA’s KV head count from 8 to 2 (GQA 16-2-2) costs −1.00% in average accuracy at the 1.7B scale; further halving the physical query heads to match VHA’s shape (GQA 8-2-2) costs another −0.71%, totaling −1.71% vs. the GQA-8 baseline. With the same (8, 2, 2) shape, VHA recovers all of this loss and exceeds the original baseline by +0.41%, despite using only 25% of GQA-8’s KV cache and 50% of its physical query heads. The doubly-compressed (8, 2, 2) shape that on its own forfeits −1.71% thus becomes a winning configuration once Premix and Postmix are introduced.

## 4.3 Large-scale and long-context validation

At the 30B-A3B MoE scale (500B tokens; FP8 mixed precision; full setup in Appendix D), Table 3 shows that VHA 24-2-2 attains the highest 12-benchmark average while using only 50% of GQA-4’s KV cache and 89% of MLA’s, confirming that the trends observed at 0.6B∼8B-A1B carry over to the production-scale MoE regime. The category-level breakdown shows that gains are broadly distributed: General +3.39 / +0.68%, Math & STEM +3.31 / +2.91%, Coding +1.38 / −1.14%, Multilingual +2.30 / +0.36% (VHA vs. GQA / MLA), indicating head-level virtualization scales gracefully across heterogeneous downstream tasks. Per-benchmark scores for all 12 evaluations are in Appendix C.

Long-context validation. At the final 128K checkpoint, VHA outperforms the matched GQA baseline on RULER at every evaluated length from 8K to 128K, with a length-average of 84.4 versus 73.0 (+11.4 points). The advantage extends to complementary evaluations at 128K: VHA outperforms GQA on LongBench-v2 (24.2 vs. 18.9) and BABILong (38.2 vs. 31.1), while remaining on par on InfiniteBench (31.5 vs. 31.8); averaged across lengths, VHA wins, ties, and loses 3/1/1

Table 4: VHA ablations. Left, middle: component contribution at 1.7B and 4B (each row uses VHA’s shape, 8-2-2 or 16-2-2, with KV cache 1.00 KB / layer / token vs 4.00 KB for the GQA-8 baseline; Premix and Postmix are toggled individually). Right: activation ablation at 1.7B (replacing the linear formulation with SiLU on either or both modules). Avg (%) is over the eight benchmarks of Table 1; ∆ is the average delta vs the GQA-8 baseline (component) or the linear default (activation); the linear VHA default is bold. Each module on its own only partially recovers the doubly-compressed loss from Table 2; the full VHA configuration exceeds the GQA-8 baseline at both scales, and SiLU consistently underperforms the linear default.  
Component @ 1.7B
<table><tr><td>Variant</td><td>Avg</td><td>∆</td></tr><tr><td>GQA</td><td>54.09</td><td></td></tr><tr><td>Premix only</td><td>53.31</td><td>-0.78</td></tr><tr><td>Postmix only</td><td>53.95</td><td>-0.14</td></tr><tr><td>VHA</td><td>54.50</td><td>+0.41</td></tr></table>

Component @ 4B
<table><tr><td>Variant</td><td>Avg</td><td>∆</td></tr><tr><td>GQA</td><td>56.90</td><td></td></tr><tr><td>Premix only</td><td>56.62</td><td>-0.28</td></tr><tr><td>Postmix only</td><td>57.22</td><td>+0.32</td></tr><tr><td>VHA</td><td>57.71</td><td>+0.81</td></tr></table>

<table><tr><td colspan="3">Activation @ 1.7B</td></tr><tr><td>Variant</td><td>Avg</td><td> $\Delta$ </td></tr><tr><td>+ SiLU on Premix</td><td>53.83</td><td>-0.67</td></tr><tr><td>+ SiLU on Postmix</td><td>53.79</td><td>-0.71</td></tr><tr><td>+ SiLU on both</td><td>53.67</td><td>-0.83</td></tr><tr><td>Linear (default)</td><td>54.50</td><td>1</td></tr></table>

Table 5: Computational efficiency across four model scales (S=4096). For each scale we list per-method query/key/value head counts, KV cache per layer per token (BF16, KB), training FLOPs per token aggregated over all transformer layers (G), and the relative change vs. the per-scale GQA baseline (GQA-8 dense, GQA-4 MoE). Bold highlights VHA, the method that achieves the best joint KV cache+FLOPs Pareto optimum at every scale. MHA’s lower FLOPs at three scales reflect its smaller query head count under our parameter-matched configurations $( H _ { q } = 1 2$ at 0.6B/1.7B, $H _ { q } = 2 0$ at 4B/8B-A1B versus GQA’s $H _ { q } { = } 1 6$ or $H _ { q } = 3 2 )$ , at the cost of a 6–10× larger KV cache.
<table><tr><td></td><td colspan="4">0.6B</td><td colspan="4">1.7B</td><td colspan="4">4B</td><td colspan="4">8B-A1B</td></tr><tr><td>Method</td><td>Q-K-V</td><td>Cache</td><td>FLOPs</td><td>∆%</td><td>Q-K-V</td><td>Cache</td><td>FLOPs</td><td>∆%</td><td>Q-K-V</td><td>Cache</td><td>FLOPs</td><td>∆%</td><td>Q-K-V</td><td>Cache FLOPs</td><td></td><td>∆%</td></tr><tr><td>MHA</td><td>12-12-12</td><td>6.00</td><td>6.87</td><td>-9.3</td><td>12-12-12</td><td>6.00</td><td>9.91</td><td>-6.6</td><td>20-20-20</td><td>10.00</td><td>21.66</td><td>-11.1</td><td>20-20-20</td><td>10.00</td><td>8.76</td><td>-14.7</td></tr><tr><td>GQA</td><td>16-8-8</td><td>4.00</td><td>7.57</td><td>0</td><td>16-8-8</td><td>4.00</td><td>10.61</td><td>0</td><td>32-8-8</td><td>4.00</td><td>24.38</td><td>0</td><td>32-4-4</td><td>2.00</td><td>10.27</td><td>0</td></tr><tr><td>MLA</td><td>14-14-14</td><td>1.12</td><td>8.25</td><td>+8.9</td><td>12-12-12</td><td>1.12</td><td>10.42</td><td>-1.8</td><td>28-28-28</td><td>1.12</td><td>25.09</td><td>+2.9</td><td>28-28-28</td><td>1.12</td><td>11.39</td><td>+10.9</td></tr><tr><td>MFA</td><td>17-1-1</td><td>1.00</td><td>11.54</td><td>+52.3</td><td>19-1-1</td><td>1.00</td><td>15.90</td><td>+49.8</td><td>33-1-1</td><td>1.00</td><td>36.04</td><td>+47.9</td><td>33-1-1</td><td>1.00</td><td>17.82</td><td>+73.5</td></tr><tr><td>VHA</td><td>8-2-2</td><td>1.00</td><td>6.96</td><td>-8.1</td><td>8-2-2</td><td>1.00</td><td>9.78</td><td>-7.9</td><td>16-2-2</td><td>1.00</td><td>22.52</td><td>-7.6</td><td>16-2-2</td><td>1.00</td><td>9.60</td><td>-6.5</td></tr></table>

HELMET categories. Together, these results demonstrate that VHA preserves its quality advantage through 128K while halving the KV heads and KV cache of the full-attention layers relative to GQA-4; complete results are in Appendix A.

## 4.4 Ablations

We ablate VHA’s design choices at the 1.7B scale (additionally at 4B for the component contribution), isolating (i) the contribution of Premix and Postmix, (ii) the Postmix rank, and (iii) whether nonlinearity helps either module.

Reading the $\Delta$ column of Table 4 (left and middle), the two modules address complementary deficits of the doubly-compressed shape: only their joint VHA configuration consistently achieves the largest gains at both scales at 25% of GQA-8’s KV cache; we provide a mechanistic interpretation of this complementarity in Section 5.

Postmix rank. With Premix fixed at the 1.7B scale, sweeping the Postmix rank gives Avg 54.50/54.24/54.15 for $r \in \{ 4 , 3 2 , 1 2 8 \}$ : larger r monotonically lowers training loss but degrades downstream evaluation, indicating that the low-rank constraint provides useful implicit regularization. We therefore fix $r = H _ { q } ^ { \prime } / 2$ across all scales (Section 3.4 / Appendix D).

Activation. Replacing the linear formulation with SiLU on either or both modules consistently degrades quality at 1.7B (Table 4, right), regardless of where the non-linearity is inserted. Beyond the empirical loss, every non-linear variant forfeits inference-time absorption: with both modules linear the entire VHA computation collapses to standard GQA-2 at serving time, whereas any non-linearity must remain in the runtime kernel. We therefore keep both modules strictly linear, simultaneously achieving the highest empirical quality and lossless absorption into standard GQA-2 at inference.

## 4.5 Computational Efficiency

Table 5 quantifies the joint KV cache–FLOPs trade-off across the four model scales of Table 1. VHA is the only method that simultaneously matches the smallest KV cache (1.00 KB) and reduces training/prefill FLOPs below the per-scale GQA baseline (−8.1 / −7.9 / −7.6 / −6.5 % across scales).

![](images/9b5865d499ad5cf7a537b432c65f933b8621bb1482427a1a5bd2056949a5c429.jpg)

![](images/92dfe09ea455e5c65b953a2469a7aef2f7f19fc8b841d561b15e3f5811eb9da0.jpg)

Figure 2: Q Premix weight analysis on a trained Qwen-1.7B VHA model. (a) Per-layer Frobenius deviation $\| \bar { W } _ { g } ^ { \mathrm { p r e m i x } } - I \| _ { F }$ per KV group (group mean overlaid). (b) Per-layer mean pairwise distance $\lVert W _ { g _ { 1 } } ^ { \mathrm { p r e m i x } } - \bar { W } _ { g _ { 2 } } ^ { \mathrm { p r e m i x } } \rVert _ { F }$ between KV groups.  
![](images/f25d9f7549ac176d19472602ed48d76494cd08d7fd6bfcc93f8fad6dfb7e7d16.jpg)  
Figure 3: Postmix head-mixing analysis on a trained Qwen-4B VHA model. Sankey diagrams of $| \check { A } \check { B } ^ { \intercal } |$ across five representative layers, with ribbons flowing from source heads (left, colored by KV group) to recombined output heads (right); width and color saturation proportional to $| A B ^ { \top } | _ { j , i }$

MFA matches the cache but pays a +48–74 % FLOPs penalty; MLA’s latent compression yields a slightly larger cache (1.12 KB) and raises FLOPs at three of four scales (up to +10.9 % at 8B-A1B). MHA appears competitive on FLOPs only because its smaller query head count under our parameter matched configurations comes with a $6 { \bar { - } } 1 0 \times$ larger KV cache. Combined with VHA’s complete inference absorption to standard GQA-2 (Section 3.4), this makes VHA the best Pareto-optimal point on the cache–FLOPs–quality triangle in our experiments.

## 5 Analysis: How VHA Reparameterization Works

Beyond task accuracy, VHA should exploit the limited information available under aggressive KV sharing. We therefore inspect the Premix and Postmix matrices of trained VHA models (Premix on the 1.7B, Postmix on the 4B model), asking whether (i) Premix learns a group-conditioned query geometry, (ii) Postmix learns structured low-rank cross-head communication, and (iii) the two jointly close the doubly-compressed gap of Section 4.4.

Premix learns non-trivial, group-differentiated query transforms. Premix introduces a learnable per-group query transform $\bar { W } _ { g } ^ { \mathrm { p r e m i x } }$ before RoPE, initialized near identity. Two trivial outcomes would falsify our hypothesis: (a) $W _ { g } ^ { \mathrm { p r e m i x } }$ remains near identity after training, rendering the module a no-op; or (b) all KV groups converge to the same matrix, degenerating Premix into a group-agnostic shared adapter without group-conditioned specialization. Figure 2 refutes both: panel (a) shows that $\| W _ { g } ^ { \mathrm { p r e m i x } } - I \| _ { F }$ is substantially nonzero and most pronounced in mid-to-late layers (peak 5.85), and panel (b) shows that the inter-group distance $\lVert W _ { g _ { 1 } } ^ { \mathrm { p r e m i x } } - W _ { g _ { 2 } } ^ { \mathrm { p r e m i x } } \rVert _ { F }$ is of comparable magnitude (peak 6.13), yielding a ratio well below the $\sqrt { 2 }$ value expected under fully independent groups and far above the zero expected under a shared deviation. This quantifies that the two KV groups learn substantially but not independently different query transforms. Premix therefore learns differentiated query probing directions per KV group under a fixed minimal KV cache, realizing genuine group-conditioned virtual-head expansion rather than mere parameter augmentation.

Postmix learns structured low-rank cross-head communication. Postmix recombines the $H _ { v }$ attention output heads via the low-rank residual $I + B A ^ { \top }$ , where entry $( j , i )$ of $| A B ^ { \top } |$ encodes the contribution of source head $j$ to recombined head i. The central question is whether $\dot { A } B ^ { \top }$ merely performs a per-head rescaling (a strictly diagonal pattern) or genuinely redistributes information across heads (off-diagonal flow). Figure 3 visualizes $| A B ^ { \top } |$ for the trained 4B model and reveals pronounced off-diagonal flow, ruling out the trivial per-head-rescaling outcome. The dominant pattern is within-group redistribution: Postmix primarily rebalances information among heads sharing the same KV cache, with a sparse but consistent set of cross-group bridges providing inter-group connectivity without additional KV heads.

Joint reparameterization closes the doubly-compressed gap. Tying these analyses back to the component ablation in Table 4 clarifies the three numbers reported there. Premix-only retains the group-conditioned query geometry of Figure 2 but lacks Postmix’s cross-group repair pathway, hence the −0.78% gap at 1.7B. Postmix-only owns the cross-head mixing channels of Figure 3 but operates over collapsed, unspecialized virtual heads, hence the −0.14% gap. Only the joint configuration realizes both structures simultaneously, yielding a +0.41% advantage over GQA-8 at 25% of its KV cache. VHA’s effectiveness therefore arises from the functional complementarity of its two modules: Premix and Postmix are not interchangeable, and neither suffices alone.

## 6 Conclusion

We have presented Virtual Head Attention (VHA), which mitigates the tension among KV cache size, training/prefill FLOPs, and model quality through two lightweight linear operations: Q Premix and Linear Postmix. By halving physical query heads and recovering expressiveness via virtual head expansion and low-rank cross-group mixing, VHA reduces the KV cache to its architectural minimum while simultaneously lowering training FLOPs and improving model quality. Since all added operations are strictly linear, VHA fully absorbs into standard GQA-2 at inference, reusing existing serving optimizations with zero additional overhead. Analysis of trained models reveals that Premix and Postmix are functionally complementary and neither suffices alone, explaining why the joint configuration consistently surpasses the GQA baseline across all five evaluated scales (0.6B to 30B-A3B MoE). A matched 10B-A1B evaluation further shows that VHA maintains this advantage over GQA through 128K context.

## Acknowledgments

We thank Wenbin Chang for valuable input and helpful discussions. We also thank the anonymous reviewers, area chairs, and senior area chairs for their insightful comments and constructive suggestions.

This work was supported by the National Engineering Research Center of Deep Learning Technology and Application and the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (Grant No. JYB2025XDXM202).

## References

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan-Kelley. Reducing transformer key-value cache size with cross-layer attention. Advances in Neural Information Processing Systems, 37:86927–86957, 2024.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 conference ofthe north American chapter ofthe associationfor computational linguistics: Human language technologies, volume 1 (long and short papers), pages 2924–2936, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. Multi-head attention: Collaborate instead of concatenate. arXiv preprint arXiv:2006.16362, 2020.

Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 13733–13742, 2021.

Sai Surya Duvvuri, Chanakya Ekbote, Rachit Bansal, Rishabh Tiwari, Devvrit Khatri, David Brandfonbrener, Paul Liang, Inderjit Dhillon, and Manzil Zaheer. Interleaved head attention. arXiv preprint arXiv:2602.21371, 2026.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In Proceedings ofthe 2025 Conference ofthe Nations ofthe Amer icas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

Jingcheng Hu, Houyi Li, Yinmin Zhang, Zili Wang, Shuigeng Zhou, Xiangyu Zhang, and Heung Yeung Shum. Multi-matrix factorization attention. In Findings of the Association for Computational Linguistics: ACL 2025, pages 25114–25126, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-ofexperts language model. arXiv preprint arXiv:2405.04434, 2024.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in neural information processing systems, 36:21558–21572, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2381–2391, 2018.

James O’Neill, Robert Clancy, Mariia Matskevichus, and Fergal Reid. Low-rank key value attention. arXiv preprint arXiv:2601.11471, 2026.

OpenAI. MMMLU: Multilingual massive multitask language understanding, 2024. URL https: //huggingface.co/datasets/openai/mmmlu.

Runyu Peng, Yunhua Zhou, Demin Song, Kai Lv, Bo Wang, Qipeng Guo, and Xipeng Qiu. Explicit multi-head attention for inter-head interaction in large language models. arXiv preprint arXiv:2601.19611, 2026.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. Proceedings ofmachine learning and systems, 5:606–624, 2023.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Horn Roger and R Johnson Charles. Topics in matrix analysis, 1994.

Angelika Romanou, Negar Foroutan, Anna Sotnikova, Zeming Chen, Sree Harsha Nelaturu, Shivalika Singh, Rishabh Maheshwary, Micol Altomare, Mohamed A Haggag, Alfonso Amayuelas, et al. Include: Evaluating multilingual language understanding with regional knowledge. arXiv preprint arXiv:2411.19799, 2024.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4463–4473, 2019.

Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

Noam Shazeer, Zhenzhong Lan, Youlong Cheng, Nan Ding, and Le Hou. Talking-heads attention. arXiv preprint arXiv:2003.02436, 2020.

Luoyang Sun, Cheng Deng, Jiwen Jiang, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, and Jun Wang. Gta: Grouped-head latent attention. arXiv preprint arXiv:2506.17286, 2025.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. Advances in Neural Information Processing Systems, 37:7339–7361, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024.

Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings ofthe 3rd Workshop on Noisy User-generated Text, pages 94–106, 2017.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 4791–4800, 2019.

Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Zhen Qin, Yang Yuan, Quanquan Gu, and Andrew Chi-Chih Yao. Tensor product attention is all you need. arXiv preprint arXiv:2501.06425, 2025.

Chuanyang Zheng, Jiankai Sun, Yihang Gao, Yuehao Wang, Peihao Wang, Jing Xiong, Liliang Ren, Hao Cheng, Janardhan Kulkarni, Yelong Shen, et al. Sas: Simulated attention score. arXiv preprint arXiv:2507.07694, 2025.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

## Limitations and Future Work

Our largest validation is at the 30B-A3B MoE scale trained on 500B tokens; scaling VHA to larger MoE models and substantially more training data (e.g., 20T tokens) remains an important direction. We further validate VHA through 128K on a 10B-A1B MoE, while broader long-context architectures and training recipes remain to be explored. VHA’s behavior under reinforcement learning training paradigms is also worth exploring. Finally, VHA is orthogonal to other KV cache compression techniques such as quantization and sparsification, and combining them may yield further inference savings.

## A Long-Context Evaluation

## A.1 Model and Stage-wise Training

We conduct a controlled long-context comparison on a MiMo-2.5-Flash-style 10B-A1B MoE backbone that interleaves full and sliding-window attention at a 1:3 ratio. The GQA and VHA models share the same backbone, data, optimization settings, and sliding-window layers. They differ only in the full-attention layers: GQA uses four KV heads, whereas VHA uses two; all sliding-window layers use eight KV heads in both models. Thus, VHA halves the KV cache of the full-attention layers without changing the sliding-window component.

The 8K pretraining stage uses 500B tokens and ends at 60K steps. Long-context mid-training then uses 100B tokens at 32K followed by another 100B tokens at 128K. These stages add 12K and 24K steps, yielding cumulative checkpoints at 72K and 96K steps, respectively. The RoPE base is increased from $1 0 ^ { \overline { { 4 } } }$ at 8K to $6 . 4 \times 1 0 ^ { 5 }$ at 32K and $5 \times 1 0 ^ { 6 }$ at 128K. We report intermediate results from the 32K checkpoint and final results from the 128K checkpoint.

## A.2 RULER and HELMET

We evaluate RULER and HELMET from 8K to 128K. RULER Overall is the mean over 13 tasks: eight Retrieval, two Aggregation, two QA, and one Variable Tracking task. HELMET is reported by category because its categories measure distinct capabilities.

At the final checkpoint, VHA leads RULER Overall at every evaluated length, with an average of 84.4 across lengths versus 73.0 for GQA. The largest gap occurs in Variable Tracking, where VHA scores 92.6/87.2/83.0 at 32K/64K/128K versus 3.2/4.2/3.0 for GQA. HELMET is more heterogeneous: averaged across lengths, VHA wins RAG, LongQA, and Summarization, ties Reranking, and trails on ICL.

## A.3 Complementary Long-Context Evaluations

We additionally report LongBench-v2, InfiniteBench, BABILong, a short-context eight-task average, and language-modeling bits per byte on the four ProLong domains. At the final checkpoint, VHA wins seven of the eight reported metrics; InfiniteBench is nearly tied.

## B Additional Robustness Experiments

The following experiments use a more efficient training implementation than the main 50B-token runs. Absolute scores therefore need not match the main tables, but all methods within each comparison use the same implementation and training recipe.

## B.1 Repeated-run Robustness at 0.6B

We repeat the 0.6B experiment four times for VHA and three times for GQA. We report every run using zero-shot BF16 evaluation, with normalized accuracy used where available. Avg is the unweighted mean over eight tasks, and ± denotes the sample standard deviation across run-level averages. VHA achieves a higher mean with lower observed dispersion.

## B.2 Sensitivity to Premix Initialization

We train the 1.7B model from scratch with the same recipe while changing only the Premix initialization: the default near-identity initialization (α=0.1), a larger identity-anchored perturbation (α=1.0), and fully random initialization without the identity anchor. All three runs remain stable, with no

Table 6: Long-context results on RULER and HELMET (%). RULER Overall averages all 13 tasks. Bold marks the better result within each matched checkpoint and evaluation length; exact ties are unbolded.
<table><tr><td></td><td></td><td></td><td colspan="4">RULER</td><td colspan="6">HELMET</td></tr><tr><td>Checkpoint</td><td>Length</td><td>Model</td><td>Overall</td><td>Retr.</td><td>Aggr.</td><td>QA</td><td>VT</td><td>RAG</td><td>LongQA</td><td>Summ.</td><td>ICL</td><td>Rerank</td></tr><tr><td>32K</td><td>8K</td><td>GQA</td><td>80.9</td><td>92.4</td><td>73.1</td><td>56.5</td><td>53.6</td><td>24.5</td><td>9.0</td><td>7.2</td><td>60.7</td><td>0.3</td></tr><tr><td></td><td></td><td>VHA</td><td>85.7</td><td>95.9</td><td>78.0</td><td>60.1</td><td>69.8</td><td>29.6</td><td>11.0</td><td>6.6</td><td>57.1</td><td>0.4</td></tr><tr><td></td><td>16K</td><td>GQA</td><td>73.7</td><td>90.0</td><td>51.8</td><td>53.7</td><td>27.6</td><td>22.6</td><td>11.0</td><td>8.0</td><td>70.4</td><td>0.0</td></tr><tr><td></td><td></td><td>VHA</td><td>83.2</td><td>93.3</td><td>71.5</td><td>55.4</td><td>80.8</td><td>32.3</td><td>10.5</td><td>7.2</td><td>68.6</td><td>0.0</td></tr><tr><td></td><td>32K</td><td>GQA</td><td>69.7</td><td>81.9</td><td>51.9</td><td>49.0</td><td>49.4</td><td>22.3</td><td>17.0</td><td>8.1</td><td>78.3</td><td>0.0</td></tr><tr><td></td><td></td><td>VHA</td><td>73.5</td><td>83.5</td><td>56.8</td><td>51.8</td><td>71.4</td><td>33.4</td><td>11.0</td><td>7.6</td><td>75.1</td><td>0.5</td></tr><tr><td>128K</td><td>8K</td><td>GQA</td><td>79.1</td><td>89.5</td><td>87.9</td><td>46.0</td><td>44.6</td><td>27.4</td><td>12.5</td><td>5.9</td><td>58.7</td><td>0.6</td></tr><tr><td></td><td></td><td>VHA</td><td>90.1</td><td>96.7</td><td>92.8</td><td>59.4</td><td>94.0</td><td>34.2</td><td>12.5</td><td>5.6</td><td>46.1</td><td>0.1</td></tr><tr><td></td><td>16K</td><td>GQA</td><td>78.2</td><td>88.1</td><td>95.1</td><td>46.7</td><td>28.4</td><td>28.4</td><td>13.0</td><td>6.4</td><td>66.4</td><td>0.5</td></tr><tr><td></td><td></td><td>VHA</td><td>88.5</td><td>93.5</td><td>98.9</td><td>55.9</td><td>93.4</td><td>34.3</td><td>14.5</td><td>6.3</td><td>61.0</td><td>0.7</td></tr><tr><td></td><td>32K</td><td>GQA</td><td>72.6</td><td>83.3</td><td>93.6</td><td>43.2</td><td>3.2</td><td>26.7</td><td>21.0</td><td>6.7</td><td>75.6</td><td>0.4</td></tr><tr><td></td><td></td><td>VHA</td><td>85.3</td><td>89.6</td><td>96.0</td><td>53.7</td><td>92.6</td><td>36.9</td><td>18.5</td><td>6.9</td><td>72.1</td><td>0.4</td></tr><tr><td></td><td>64K</td><td>GQA</td><td>69.5</td><td>81.0</td><td>86.4</td><td>39.5</td><td>4.2</td><td>27.8</td><td>19.0</td><td>7.9</td><td>80.1</td><td>0.7</td></tr><tr><td></td><td></td><td>VHA</td><td>81.3</td><td>85.6</td><td>90.2</td><td>52.2</td><td>87.2</td><td>39.5</td><td>20.0</td><td>7.7</td><td>79.6</td><td>0.4</td></tr><tr><td></td><td>128K</td><td>GQA</td><td>65.7</td><td>75.3</td><td>92.5</td><td>31.5</td><td>3.0</td><td>30.4</td><td>13.0</td><td>7.8</td><td>82.6</td><td>2.0</td></tr><tr><td></td><td></td><td>VHA</td><td>76.8</td><td>79.9</td><td>92.5</td><td>45.5</td><td>83.0</td><td>42.5</td><td>15.0</td><td>9.5</td><td>82.2</td><td>2.6</td></tr></table>

Table 7: Complementary evaluation results. LB-v2, InfBench, and BABI denote LongBench-v2, InfiniteBench, and BABILong; Short is the short-context eight-task average. The remaining columns report bits per byte (BPB) on ProLong domains. Bold marks the better matched result.
<table><tr><td>Checkpoint</td><td>Model</td><td>LB-v2 ↑</td><td>InfBench ↑</td><td>BABI↑</td><td>Short ↑</td><td>arXiv ↓</td><td>Books↓</td><td>Code ↓</td><td>Web↓</td></tr><tr><td rowspan="2">32K</td><td>GQA</td><td>28.0</td><td>29.9</td><td>36.5</td><td>62.7</td><td>0.568</td><td>0.812</td><td>0.294</td><td>0.763</td></tr><tr><td>VHA</td><td>22.3</td><td>27.3</td><td>37.4</td><td>63.7</td><td>0.568</td><td>0.812</td><td>0.295</td><td>0.765</td></tr><tr><td>128K</td><td>GQA</td><td>18.9</td><td>31.8</td><td>31.1</td><td>62.6</td><td>0.582</td><td>0.819</td><td>0.311</td><td>0.768</td></tr><tr><td></td><td>VHA</td><td>24.2</td><td>31.5</td><td>38.2</td><td>64.1</td><td>0.564</td><td>0.804</td><td>0.296</td><td>0.760</td></tr></table>

gradient explosion, divergence, NaN, or Inf. Their early loss and pre-clipping gradient norms are similar (Table 9), while the default near-identity initialization gives the highest downstream average (Table 10).

## C Per-benchmark Scores for the 30B-A3B MoE Evaluation

Table 11 reports per-benchmark scores for all evaluated models on the 30B-A3B MoE suite. Categoryaveraged numbers are reproduced in Table 3 in the main text. Model configurations: GQA 48-4-4, MLA 28-28-28, VHA 24-2-2. Bold marks the best per row.

## D Training Configuration and Benchmark Citations

## D.1 Model Architecture and Training Hyperparameters

Table 12 lists the backbone architecture and training hyperparameters for all five scales. Small-scale models follow the Qwen3 architecture; the 30B-A3B model uses an ERNIE-style MoE architecture with auxiliary-free routing (noaux\_tc) and z-loss $( \lambda { = } 4 \times 1 0 ^ { - 4 } )$ . We use AdamW [Loshchilov and Hutter, 2017] with $\beta _ { 1 } { = } 0 . \bar { 9 } , \beta _ { 2 } { = } 0 . 9 5$ , weight decay 0.1, and gradient clipping at 1.0.

## D.2 Evaluation Benchmarks

Small-scale benchmarks (8 tasks). We report zero-shot accuracy on the following benchmarks: ARC-Challenge [Clark et al., 2018], BoolQ [Clark et al., 2019], HellaSwag [Zellers et al., 2019],

Table 8: Repeated-run results at 0.6B (%). Every run is reported. Bold in the summary rows marks the higher across-run mean.
<table><tr><td>Model</td><td>Run</td><td>ARC-C</td><td>BoolQ</td><td>HSwag</td><td>OBQA</td><td>PIQA</td><td>Wino</td><td>SciQ</td><td>SIQA</td><td>Avg</td></tr><tr><td>VHA</td><td>01</td><td>30.12</td><td>60.18</td><td>44.46</td><td>33.80</td><td>68.77</td><td>52.25</td><td>74.20</td><td>39.56</td><td>50.42</td></tr><tr><td>VHA</td><td>02</td><td>29.35</td><td>59.27</td><td>44.36</td><td>32.80</td><td>68.55</td><td>55.96</td><td>74.00</td><td>39.56</td><td>50.48</td></tr><tr><td>VHA</td><td>03</td><td>29.27</td><td>59.36</td><td>44.79</td><td>33.80</td><td>67.14</td><td>52.25</td><td>76.00</td><td>40.23</td><td>50.35</td></tr><tr><td>VHA</td><td>04</td><td>30.12</td><td>56.09</td><td>44.68</td><td>34.80</td><td>68.55</td><td>52.64</td><td>76.10</td><td>39.76</td><td>50.34</td></tr><tr><td>VHA mean</td><td></td><td>29.72</td><td>58.73</td><td>44.57</td><td>33.80</td><td>68.25</td><td>53.28</td><td>75.08</td><td>39.78</td><td>50.40±0.07</td></tr><tr><td>GQA</td><td>01</td><td>29.44</td><td>58.59</td><td>44.49</td><td>32.40</td><td>68.28</td><td>51.14</td><td>75.50</td><td>38.79</td><td>49.83</td></tr><tr><td>GQA</td><td>02</td><td>30.12</td><td>57.86</td><td>44.55</td><td>32.60</td><td>68.34</td><td>50.91</td><td>74.10</td><td>39.87</td><td>49.79</td></tr><tr><td>GQA</td><td>03</td><td>30.20</td><td>60.55</td><td>44.61</td><td>35.00</td><td>68.99</td><td>51.93</td><td>72.50</td><td>39.30</td><td>50.39</td></tr><tr><td>GQA mean</td><td></td><td>29.92</td><td>59.00</td><td>44.55</td><td>33.33</td><td>68.54</td><td>51.33</td><td>74.03</td><td>39.32</td><td>50.00±0.34</td></tr></table>

Table 9: Early training trajectory under different Premix initializations. Each entry is loss / preclipping gradient norm.
<table><tr><td>Step</td><td>α=0.1</td><td>α=1.0</td><td>Random</td></tr><tr><td>10</td><td>12.2976 / 14.74</td><td>12.3072 / 14.80</td><td>12.3132 / 14.98</td></tr><tr><td>20</td><td>11.1501 / 15.97</td><td>11.1581 / 16.26</td><td>11.1576 / 16.25</td></tr><tr><td>30</td><td>9.9346 / 4.98</td><td>9.9389 / 5.01</td><td>9.9400 / 5.07</td></tr><tr><td>50</td><td>9.1168 / 2.67</td><td>9.1243 / 3.76</td><td>9.1255 / 2.03</td></tr><tr><td>100</td><td>7.9670 / 2.96</td><td>7.9788 / 2.93</td><td>7.9669 / 4.02</td></tr><tr><td>200</td><td>6.4777 / 2.99</td><td>6.4878 / 2.21</td><td>6.4898 / 1.65</td></tr><tr><td>500</td><td>4.9822 / 1.70</td><td>4.9990 / 2.47</td><td>5.0198 / 2.85</td></tr><tr><td>1000</td><td>3.8647 / 1.17</td><td>3.8851 / 0.94</td><td>3.8888 / 0.89</td></tr><tr><td>2000</td><td>3.1699 / 0.27</td><td>3.1842 / 0.33</td><td>3.1967 / 0.54</td></tr></table>

OpenBookQA [Mihaylov et al., 2018], PIQA [Bisk et al., 2020], Winogrande [Sakaguchi et al., 2021], SciQ [Welbl et al., 2017], Social IQA [Sap et al., 2019].

Large-scale benchmarks (12 tasks). We report accuracy with the shot counts listed in Table 11 on the following benchmarks: MMLU [Hendrycks et al., 2020], MMLU-Redux [Gema et al., 2025], MMLU-Pro [Wang et al., 2024], BBH [Suzgun et al., 2023], GPQA [Rein et al., 2023], GSM8K [Cobbe et al., 2021], MATH [Hendrycks et al., 2021], HumanEval+ [Liu et al., 2023], MBPP [Austin et al., 2021], MBPP+ [Liu et al., 2023], MMMLU [OpenAI, 2024], INCLUDE [Romanou et al., 2024].

## E Training Loss Curves

Figure 4 shows per-method training loss curves at four model scales (50B tokens, 4K context). VHA achieves the lowest convergence loss at the 4B scale and remains competitive at 0.6B, 1.7B, and 8B-A1B, where its gap from the best-converging method is within 0.01 nats.

## F VHA Algorithm

Algorithm 1 provides a self-contained reference for VHA’s training forward pass and the one-time inference-absorption procedure that reduces the trained model to a standard GQA-2 deployment.

## G Premix as Structured Factorization of Virtual-Head Query Projections

We show that, in the forward pass, Q Premix is mathematically equivalent to a standard multi-head attention with $H _ { v } = H _ { k v } H _ { q } ^ { \bar { \prime } }$ virtual query heads, whose per-virtual-head query projections are constrained to a structured low-dimensional manifold. The training-time benefit of the separated parameterization arises from the coupled gradient structure induced by this factorization.

Setup. Let $W _ { Q } ^ { \mathrm { p h y s } } \in \mathbb { R } ^ { h \times H _ { q } ^ { \prime } d }$ be the physical query projection (with $H _ { q } ^ { \prime }$ heads, each of dimension d), and let $W _ { g } ^ { \mathrm { p r e m i x } } \in \mathbb { R } ^ { d \times d }$ be the Premix matrix for KV group $g \in \{ 1 , \ldots , H _ { k v } \}$ . For physical head $i \in \{ 1 , \ldots , H _ { q } ^ { \prime } \}$ within group g, the virtual query is

Table 10: Downstream accuracy under different Premix initializations (%). Bold marks the best result per row.
<table><tr><td>Task</td><td> $\alpha { = } 0 . 1$ </td><td> $\alpha { = } 1 . 0$ </td><td>Random</td></tr><tr><td>ARC-C</td><td>34.81</td><td>34.13</td><td>34.39</td></tr><tr><td>BoolQ</td><td>60.06</td><td>58.04</td><td>57.03</td></tr><tr><td>HellaSwag</td><td>51.80</td><td>51.72</td><td>51.76</td></tr><tr><td>OpenBookQA</td><td>37.60</td><td>37.00</td><td>36.00</td></tr><tr><td>PIQA</td><td>71.06</td><td>71.33</td><td>72.14</td></tr><tr><td>Winogrande</td><td>55.72</td><td>54.85</td><td>53.28</td></tr><tr><td>SciQ</td><td>80.70</td><td>80.30</td><td>80.30</td></tr><tr><td>Social IQA</td><td>40.58</td><td>40.23</td><td>39.97</td></tr><tr><td>Avg</td><td>54.04</td><td>53.45</td><td>53.11</td></tr></table>

Table 11: Per-benchmark scores on the 30B-A3B MoE evaluation suite (accuracy %).
<table><tr><td>Category</td><td>Benchmark</td><td>Shot</td><td>GQA</td><td>MLA</td><td>VHA</td></tr><tr><td rowspan="4">General</td><td>MMLU</td><td>5-shot</td><td>55.97</td><td>59.34</td><td>60.24</td></tr><tr><td>MMLU-Redux</td><td>5-shot</td><td>63.05</td><td>65.25</td><td>66.10</td></tr><tr><td>MMLU-Pro</td><td>5-shot</td><td>36.20</td><td>41.12</td><td>40.03</td></tr><tr><td>BBH</td><td>3-shot CoT</td><td>56.23</td><td>56.57</td><td>58.61</td></tr><tr><td rowspan="3">Math &amp; STEM</td><td>GPQA</td><td>10-shot CoT</td><td>34.62</td><td>31.62</td><td>34.11</td></tr><tr><td>GSM8K</td><td>4-shot CoT</td><td>56.48</td><td>55.42</td><td>56.25</td></tr><tr><td>MATH</td><td>4-shot CoT</td><td>38.18</td><td>43.42</td><td>48.84</td></tr><tr><td rowspan="3">Coding</td><td>HumanEval+</td><td>0-shot</td><td>50.00</td><td>52.44</td><td>47.56</td></tr><tr><td>MBPP</td><td>3-shot</td><td>45.00</td><td>48.60</td><td>49.20</td></tr><tr><td>MBPP+</td><td>3-shot</td><td>60.54</td><td>62.07</td><td>62.93</td></tr><tr><td rowspan="2">Multilingual</td><td>MMMLU</td><td>5-shot</td><td>39.82</td><td>42.69</td><td>42.95</td></tr><tr><td>INCLUDE</td><td>5-shot</td><td>46.69</td><td>47.71</td><td>48.16</td></tr></table>

$$
{ \tilde { Q } } _ { i , g } = Q _ { i } W _ { g } ^ { \mathrm { p r e m i x } } = X W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { p r e m i x } } ,\tag{6}
$$

where $W _ { Q } ^ { ( i ) } \in \mathbb { R } ^ { h \times d }$ is the i-th slice of $W _ { Q } ^ { \mathrm { p h y s } }$ . The attention output for virtual head $( i , g )$ is

$$
\mathrm { h e a d } _ { i , g } \ : = \ : \mathrm { s o f t m a x } \left( \frac { \tilde { Q } _ { i , g } K _ { g } ^ { \top } } { \sqrt { d } } \right) V _ { g } .\tag{7}
$$

Forward equivalence to standard multi-head attention. Define

$$
\widetilde { \boldsymbol W } _ { \boldsymbol Q } ^ { ( i , g ) } \triangleq \boldsymbol W _ { \boldsymbol Q } ^ { ( i ) } \boldsymbol W _ { \boldsymbol g } ^ { \mathrm { p r e m i x } } \in \mathbb R ^ { h \times d } .\tag{8}
$$

By the associativity of matrix multiplication, $\tilde { Q } _ { i , g } = X \widetilde { W } _ { Q } ^ { ( i , g ) }$ , and therefore

$$
\mathrm { { h e a d } } _ { i , g } = \mathrm { { s o f t m a x } } \left( { \frac { X { \widetilde W } _ { Q } ^ { ( i , g ) } K _ { g } ^ { \top } } { \sqrt { d } } } \right) V _ { g }\tag{9}
$$

holds as a pointwise identity for every input X and every parameter assignment. Hence VHA is, at the level of input-output behavior, a standard multi-head attention with $H _ { v }$ query heads and $H _ { k v }$ KV heads, in which the i-th virtual head of group g uses the projection matrix $\tilde { \widetilde { W } } _ { Q } ^ { ( i , g ) }$

The factorization defines a low-dimensional parameter manifold. The standard $H _ { v }$ -head Q projection has $H _ { v } h d = H _ { k v } H _ { q } ^ { \prime }$ hd free parameters. The Premix factorization uses

$$
H _ { q } ^ { \prime } h d \ + \ H _ { k v } d ^ { 2 }\tag{10}
$$

parameters, parameterized by $( W _ { Q } ^ { \mathrm { p h y s } } , \{ W _ { g } ^ { \mathrm { p r e m i x } } \} _ { g = 1 } ^ { H _ { k v } } )$ . The set of attainable $\{ \widetilde { W } _ { Q } ^ { ( i , g ) } \}$ tuples is therefore a structured submanifold of the unconstrained parameter space:

$$
\mathcal { M } = \left\{ \left\{ W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { p r e m i x } } \right\} _ { i , g } : W _ { Q } ^ { ( i ) } \in \mathbb { R } ^ { h \times d } , W _ { g } ^ { \mathrm { p r e m i x } } \in \mathbb { R } ^ { d \times d } \right\} .\tag{11}
$$

Table 12: Model architecture and training hyperparameters across all five scales. “–” denotes not applicable.
<table><tr><td></td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B-A1B (MoE)</td><td>30B-A3B (MoE)</td></tr><tr><td colspan="6">Architecture</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td><td>BF16</td><td>BF16</td><td>FP8</td></tr><tr><td>Num layers</td><td>28</td><td>28</td><td>36</td><td>24</td><td>46</td></tr><tr><td>Hidden size</td><td>1024</td><td>2048</td><td>2560</td><td>2048</td><td>1536</td></tr><tr><td>FFN dim</td><td>3072</td><td>6144</td><td>9728</td><td>768×64</td><td>1024×128</td></tr><tr><td>Num experts</td><td>一</td><td>一</td><td>一</td><td>64</td><td>128 (+1 shared)</td></tr><tr><td>Experts per token</td><td>一</td><td>一</td><td>一</td><td>4</td><td>8</td></tr><tr><td>Vocab size</td><td>151936</td><td>151936</td><td>151936</td><td>151936</td><td>102400</td></tr><tr><td>RoPE base</td><td> $1 0 ^ { 6 }$ </td><td> $1 0 ^ { 6 }$ </td><td> $1 0 ^ { 6 }$ </td><td> $1 0 ^ { 6 }$ </td><td> $1 0 ^ { 6 }$ </td></tr><tr><td>Tied embeddings</td><td> $\checkmark$ </td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td colspan="6">Training</td></tr><tr><td>Training tokens</td><td>50B</td><td>50B</td><td>50B</td><td>50B</td><td>500B</td></tr><tr><td>Context length</td><td>4096</td><td>4096</td><td>4096</td><td>4096</td><td>8192</td></tr><tr><td>Peak learning rate</td><td> $2 . 5 { \times } 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Min learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td><td>wsd</td></tr><tr><td>Warmup steps</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td>Total steps</td><td>24000</td><td>24000</td><td>24000</td><td>24000</td><td>40000</td></tr><tr><td>Batch size (tokens)</td><td>2M</td><td>2M</td><td>2M</td><td>2M</td><td>13.1M</td></tr><tr><td>Adam  $\beta _ { 1 }$ </td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td></tr><tr><td>Adam  $\beta _ { 2 }$ </td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td colspan="6">VHA-specific</td></tr><tr><td>Premix init scale α</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Premix init noise σ</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Postmix rank r</td><td>4</td><td>4</td><td>8</td><td>8</td><td>12</td></tr><tr><td colspan="6">Baseline-specific</td></tr><tr><td>MLA latent dim  $c _ { k v }$ </td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>MLA latent dim  $c _ { q }$ </td><td>512</td><td>1536</td><td>1536</td><td>1536</td><td>1536</td></tr><tr><td>MFA head dim  $d ^ { \prime }$ </td><td>256</td><td>256</td><td>256</td><td>256</td><td>一</td></tr></table>

At the 1.7B scale $( h { = } 2 0 4 8 , d { = } 1 2 8 , H _ { a } ^ { \prime } { = } 8 , H _ { k v } { = } 2 , H _ { v } { = } 1 6 )$ , the factorized parameterization uses ${ \sim } 2 . 1 3 \mathbf { M }$ parameters, compared to ∼4.19M for the unconstrained $H _ { v }$ -head $\mathrm { Q }$ projection, $\mathbf { a } \sim 5 0 \%$ reduction. The corresponding Q-projection FLOPs are reduced by a similar factor, as the additional cost $2 H _ { k v } H _ { q } ^ { \prime } d ^ { 2 }$ of applying Premix is much smaller than the savings 2h $H _ { q } ^ { \prime } ( H _ { k v } - 1 ) d$ achieved by halving the physical head count. This is the principal source of VHA’s reduction in training and prefill FLOPs.

Reparameterization benefit via gradient coupling. Although Eq. 9 shows that the forward pass is algebraically equivalent to training a single matrix $\widetilde { W } _ { Q } ^ { ( i , g ) }$ per virtual head, the gradients under the factorized parameterization are coupled across groups and physical heads:

$$
\frac { \partial \mathcal { L } } { \partial W _ { Q } ^ { ( i ) } } = \sum _ { g = 1 } ^ { H _ { k v } } X ^ { \top } \frac { \partial \mathcal { L } } { \partial \tilde { Q } _ { i , g } } \left( W _ { g } ^ { \mathrm { p r e m i x } } \right) ^ { \top } ,\tag{12}
$$

$$
\frac { \partial \mathcal { L } } { \partial W _ { g } ^ { \mathrm { p r e m i x } } } = \sum _ { i = 1 } ^ { H _ { q } ^ { \prime } } \left( X W _ { Q } ^ { ( i ) } \right) ^ { \top } \frac { \partial \mathcal { L } } { \partial \tilde { Q } _ { i , g } } .\tag{13}
$$

![](images/c0b26e30104afa3bc05d4a902dcc8407d1c71c1632face0d607248db2daa864c.jpg)  
Figure 4: Training loss curves across four model scales. Each panel compares MHA, GQA, MLA, MFA, and VHA on FineWeb-Edu (50B tokens, 4K context). All curves are plotted from step 4K onward after the learning-rate warm-up; smoothing is applied with an exponential moving average (α=0.93). VHA (blue, solid) achieves the lowest loss at convergence at the 4B scale.

Each $W _ { Q } ^ { ( i ) }$ aggregates contributions from all $H _ { k v }$ groups, modulated by the corresponding Premix matrices, while each $W _ { g } ^ { \mathrm { p r e m i x } }$ receives gradient signal through all $H _ { q } ^ { \prime }$ physical heads. SGD on the factorized parameters therefore traverses the manifold M along directions induced by this coupled gradient structure, which differs from the trajectories obtained by independently optimizing $H _ { v }$ unconstrained virtual-head matrices. This is the same source of implicit regularization that underlies LoRA [Hu et al., 2022] and RepVGG [Ding et al., 2021]: a forward-equivalent factorization of a large weight tensor yields different optimization dynamics and, empirically, different (and often better) final solutions.

Inference-time absorption. Since the forward equivalence in Eq. 9 holds for any frozen parameter values, at inference we simply precompute $\widetilde { W } _ { Q } ^ { ( i , { \dot { g } } ) } = W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { \scriptscriptstyle F } }$ <sup>premix</sup> for all (i, g) pairs and discard $( i , g )$ $W _ { Q } ^ { \mathrm { p h y s } }$ and $\{ W _ { q } ^ { \mathrm { p r e m i x } } \}$ . The model then runs as a standard GQA-2 with $H _ { v }$ independent query projections, ${ \cal H } _ { k v } = 2$ KV heads, and the same output projection $W _ { O }$ , fully compatible with existing GQA serving kernels.

Unified view with Postmix. Premix and Postmix are two instantiations of the same head-axis reparameterization principle. Premix factorizes the per-virtual-head query projection on the Q side $( \widetilde { W } _ { Q } ^ { ( i , g ) } = W _ { Q } ^ { ( i ) } W _ { g } ^ { \mathrm { p r e m i x } } ) ;$ Postmix factorizes the merged output projection on the O side $( W _ { O } ^ { \prime } = \big ( ( I _ { H _ { v } } + A B ^ { \top } ) \otimes I _ { d } \big ) W _ { O } ,$ , see Appendix H). Both are forward-equivalent to a standard GQA in their merged form, both gain training-time benefits from coupled gradients of the separated parameters, and both fold back into a standard GQA-2 architecture at inference with zero overhead.

## H Proof of Postmix Linear Redundancy

We prove that Linear Postmix can be mathematically absorbed into the output projection $W _ { O } .$ , and discuss why the separated parameterization nonetheless improves training.

Algorithm 1 Virtual Head Attention (VHA)   
Training Forward Pass   
Require: Input $X \in \mathbb { R } ^ { S \times h } ;$ parameters $W _ { Q } , W _ { K } , W _ { V } , W _ { O } , \{ W _ { g } ^ { \mathrm { p r e m i x } } \} _ { g = 1 } ^ { H _ { k v } } , A , B$   
1: $Q  X W _ { Q } \in \mathbb { R } ^ { S \times H _ { q } ^ { \prime } \times d }$ ▷ Physical query heads $( H _ { q } ^ { \prime } { = } H _ { q } / 2 )$   
2: $K \gets X W _ { K } \in \mathbb { R } ^ { S \times H _ { k v } \times d } , \quad V \gets X W _ { V } \in \mathbb { R } ^ { S \times H _ { k v } \times d }$   
3: for each KV group $g = 1 , \ldots , H _ { k v }$ do   
4: $\tilde { Q } _ { g } \gets Q W _ { g } ^ { \mathrm { p } }$ remix ▷ Q Premix: virtual head expansion   
5: end for   
6: $\tilde { Q } \gets \mathrm { c o n c a t } ( \tilde { Q } _ { 1 } , \dots , \tilde { Q } _ { H _ { k v } } ) \in \mathbb { R } ^ { S \times H _ { v } \times }$ d ▷ $H _ { v } { = } H _ { k v } { \times } H _ { q } ^ { \prime }$ virtual heads   
7: $Y _ { g } \gets \mathrm { s o f t m a x } \left( \tilde { Q } _ { g } K _ { g } ^ { \top } / \sqrt { d } \right) V _ { g }$ for each group g ▷ Grouped attention   
8: $Y \gets \mathrm { c o n c a t } ( Y _ { 1 } , \dots , Y _ { H _ { k v } } )$   
9: $\boldsymbol { Y ^ { \prime } }  ( \boldsymbol { I _ { H _ { v } } } + \boldsymbol { B } \boldsymbol { A } ^ { \top } ) \boldsymbol { Y }$ ▷ Linear Postmix: cross-group mixing   
10: output ← reshape $\left( Y ^ { \prime } , [ S , H _ { v } d ] \right) W _ { O }$

Inference Absorption (one-time precomputation)   
Require: $W _ { Q } \in \mathbb { R } ^ { h \times H _ { q } ^ { \prime } d }$ (sliced into per-head $W _ { Q } ^ { ( i ) } \in \mathbb { R } ^ { h \times d } ) ; \{ W _ { g } ^ { \mathrm { p r e m i x } } \} _ { g = 1 } ^ { H _ { k v } }$ each $\in \mathbb { R } ^ { d \times d }$   
$W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h } ; A , B \in \mathbb { R } ^ { H _ { v } \times r }$   
1: for each $( i , g ) \in \{ 1 , \ldots , H _ { q } ^ { \prime } \} \times \{ 1 , \ldots , H _ { k v } \}$ do   
2: $\widetilde { W } _ { Q } ^ { ( i , g ) }  W _ { Q } ^ { ( i ) } \ : W _ { g } ^ { \mathrm { p r e m i x } ^ { * } } \in \ : \mathbb { R } ^ { h \times d }$ ▷ Fold Premix into the $( i , g )$ -th virtual query projection   
3: end for   
4: $W _ { Q } ^ { \mathrm { a b s } }  \mathrm { c o n c a t } \Big ( \big \{ \widetilde { W } _ { Q } ^ { ( i , g ) } \big \} _ { i , g } \Big ) \ \in \ \mathbb { R } ^ { h \times H _ { v } d }$ ▷ $H _ { v } { = } H _ { k v } { \times } H _ { q } ^ { \prime }$   
5: $W _ { O } ^ { \mathrm { a b s } }  ( ( I _ { H _ { v } } + A B ^ { \top } ) \otimes \bar { I _ { d } } ) W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h }$ ▷ Fold Postmix into $W _ { O }$   
6: Discard $W _ { Q } , \{ W _ { g } ^ { \mathrm { p r e m i x } } \} , A , B ;$ run standard $\mathrm { G Q A } { - } 2$ with $W _ { Q } ^ { \mathrm { a b s } } , W _ { K } , W _ { V } , W _ { O } ^ { \mathrm { a b s } }$

Setup. For a single token, let $Y \in \mathbb { R } ^ { H _ { v } \times d }$ be the concatenated attention output across all $H _ { v }$ virtual heads. Postmix computes:

$$
\boldsymbol { Y ^ { \prime } } = ( \boldsymbol { I _ { H _ { v } } } + \boldsymbol { B \boldsymbol { A } } ^ { \top } ) \boldsymbol { Y }\tag{14}
$$

where A, $B \in \mathbb { R } ^ { H _ { v } \times r }$ . The final output is obtained by applying the output projection:

$$
{ \mathrm { o u t p u t } } ^ { \top } = { \mathrm { v e c } } _ { \mathrm { r o w } } ( Y ^ { \prime } ) ^ { \top } W _ { O } \in \mathbb { R } ^ { 1 \times h }\tag{15}
$$

where $W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h }$ and $\mathrm { v e c } _ { \mathrm { r o w } } ( \cdot )$ flattens a matrix in row-major order (i.e., Y .reshape $( H _ { v } d )$ in PyTorch, concatenating head outputs).

Absorption via Kronecker product. Using the row-major identity: for $M \in \mathbb { R } ^ { H _ { v } \times H _ { \tau } }$ and $Y \in$ $\mathbb { R } ^ { H _ { v } \times d }$

$$
\operatorname { v e c } _ { \mathrm { r o w } } ( M Y ) ^ { \top } = \operatorname { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } \left( M ^ { \top } \otimes I _ { d } \right)\tag{16}
$$

Setting $M = I _ { H _ { v } } + B A ^ { \top } ( \mathbf { s o } M ^ { \top } = I _ { H _ { v } } + A B ^ { \top } )$ [Roger and Charles, 1994]:

$$
\begin{array} { r l } & { \mathrm { v e c } _ { \mathrm { r o w } } ( Y ^ { \prime } ) ^ { \top } = \mathrm { v e c } _ { \mathrm { r o w } } \big ( ( I _ { H _ { v } } + B A ^ { \top } ) Y \big ) ^ { \top } } \\ & { \qquad = \mathrm { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } \big ( ( I _ { H _ { v } } + A B ^ { \top } ) \otimes I _ { d } \big ) } \end{array}\tag{17}
$$

Substituting into the output computation:

$$
\begin{array} { r l } & { \mathrm { o u t p u t } ^ { \top } = \mathrm { v e c } _ { \mathrm { r o w } } ( Y ^ { \prime } ) ^ { \top } W _ { O } } \\ & { \quad \quad = \mathrm { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } \big ( ( I _ { H _ { v } } + A B ^ { \top } ) \otimes I _ { d } \big ) W _ { O } } \\ & { \quad \quad = \mathrm { v e c } _ { \mathrm { r o w } } ( Y ) ^ { \top } W _ { O } ^ { \prime } } \end{array}\tag{18}
$$

where $W _ { O } ^ { \prime } = \left( \left( I _ { H _ { v } } + A B ^ { \top } \right) \otimes I _ { d } \right) W _ { O } \in \mathbb { R } ^ { H _ { v } d \times h }$

Since $I _ { H _ { v } } + B A ^ { \top }$ is generically invertible (it equals the identity at initialization and remains nearidentity during training), $W _ { O } ^ { \prime }$ has the same number of free parameters as $W _ { O }$ . Therefore, for any choice of ${ \bar { W _ { O } , A , B } } .$ , there exists a single matrix $W _ { O } ^ { \prime }$ that produces the identical output, confirming that Postmix is theoretically redundant.

Reparameterization benefit. Despite this redundancy, the separated parameterization $\{ W _ { O } , A , B \}$ provides training advantages analogous to those observed in RepVGG [Ding et al., 2021]:

1. Factored gradient paths. The gradients with respect to A, B, and $W _ { O }$ follow different paths through the computation graph. This factorization provides a form of implicit regularization, as the effective weight update to $W _ { O } ^ { \prime }$ at each step is constrained by the low-rank structure of $A B ^ { \top }$

2. Identity initialization. With B initialized to zero, $B A ^ { \top } = 0$ and $Y ^ { \prime } = Y$ at the start of training. Postmix begins as a no-op and gradually learns useful cross-group mixing patterns, avoiding disruption to the early training dynamics.

3. Low-rank constraint. The bottleneck rank $r \ll H _ { \tau }$ limits the expressiveness of the cross-head mixing, acting as a structural regularizer. This prevents Postmix from overfitting, consistent with the ablation finding that larger ranks $\left( r { = } 3 2 , 1 2 8 \right)$ degrade evaluation performance despite lower training loss.