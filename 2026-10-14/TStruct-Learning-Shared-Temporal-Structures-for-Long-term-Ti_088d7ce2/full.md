# TStruct: Learning Shared Temporal Structures for Long-term Time-Series Forecasting

Yangze Li1, Zhengyang Zhou1,2,, Qihe Huang1, Pengkun Wang1,2, Binwu Wang1,2, Yang Wang1,2,

1University of Science and Technology of China (USTC), Hefei, China 2Suzhou Institute for Advanced Research, USTC, Suzhou, China liyangze@mail.ustc.edu.cn

## Abstract

Long-term time series forecasting requires temporal dependency modeling that remains stable and generalizable under temporal distribution shifts. However, real-world time series are often noisy and non-stationary, making deep learning models prone to fitting incidental temporal interactions in the training data. A key question is whether there exist temporal relationship structures that recur across different temporal positions and variables. We analyze temporal relationship matrices on commonly used time series datasets and find that samples from different temporal positions and different variables exhibit shared dominant temporal relationship structures. This indicates that temporal relationship structures contain patterns that recur stably across samples and variables. To learn such shared structures, we propose Temporal Structure Network (TStruct), a simple yet effective network for directly learning shared and generalizable temporal dependencies. Specifically, TStruct directly parameterizes a set of shared temporal dependency matrices and generates normalized combination weights using dynamic sample information and static variable priors. These weights represent a soft allocation of each sample-variable pair over shared temporal dependency structures, enabling the model to generate adaptive temporal dependency matrices under shared structural constraints, rather than fitting incidental interactions in a fully unconstrained manner. Notably, this simple structure-aware design consistently outperforms several strong baselines across all datasets, achieving superior accuracy while maintaining high computational efficiency. Our code is available at https://github.com/yangzeli/TStruct/.

## 1 Introduction

Long-term time series forecasting (LTSF) has widespread applications in real-world scenarios such as energy scheduling, traffic management, weather forecasting, and financial risk control [46, 39, 44, 28]. The goal of LTSF is to accurately model future dynamics over a long prediction horizon. The challenge lies not only in fitting historical observations but also in learning temporal dependency structures that remain effective across future time distributions. However, real-world time series data often have low signal-to-noise ratios (SNR), non-stationarity, and temporal distribution shifts, and local fluctuations or incidental interactions in the training data may not generalize well for future predictions [8, 7, 38, 37]. Therefore, a central question in LTSF is: how can we identify and model temporal dependency structures that remain stable across shifting temporal distributions?

![](images/e93b4bbd507ebfa71447e4971d227f8276fcde7c05fa9145fa60df698af03a9a.jpg)  
Figure 1: Patch-level temporal relationship structures. (a) Construction of patch-level relationship matrices and qualitative comparison between attention-based and TStruct-based temporal dependency modeling. (b,c) Concentrated singular value spectra across samples and variables. (d) Higher spectral concentration in real data than in patch-shuffled and IID Gaussian controls supports reusable temporal structures.

In recent years, deep learning methods have driven significant progress in long-term time series forecasting [46, 39, 47, 44, 28, 45, 11, 10, 17, 1, 18, 33, 24, 12, 27, 14, 26]. One category of methods implicitly models temporal dependencies through model architectures. For example, recurrent models pass historical information via hidden states [22, 2], convolutional and TCN models use shared kernels to extract local or multi-scale temporal patterns [23, 35, 6], Transformer-based models estimate pairwise interactions between time tokens explicitly using attention mechanisms [28, 45, 4, 29], while Linear/MLP methods achieve efficient prediction by mixing time or feature dimensions [44, 43, 36, 20, 13]. Despite their strengths in expressive power and efficiency, these methods still face a trade-off between adaptive temporal dependency modeling and forecasting generalization. Highly flexible interaction mechanisms may fit incidental relationships in low-SNR data, while overly fixed transformations struggle to capture heterogeneous temporal dependencies across different samples and variables [18, 17, 37]. A recent line of work directly parameterizes stable temporal patterns, most notably periodicity, through global temporal priors, suggesting that certain temporal structures need not be entirely learned implicitly by generic neural architectures [19, 21, 42, 25, 3]. However, such explicit parameterization has mainly focused on periodic or time-indexed patterns in the input sequence, leaving open whether temporal dependency structures themselves also contain shared and explicitly learnable patterns.

To investigate whether temporal dependencies contain shared structures, we analyze patch-level temporal relationship matrices across commonly used time series datasets. As shown in Figure 1(a), for each sample-variable sequence, we divide the input window into temporal patches and construct a matrix that captures the relationships between relative patch positions. This representation shifts the analysis from raw value fluctuations to relationship patterns among relative patch positions, allowing us to examine whether these patterns recur across samples and variables. We then collect these matrices separately across samples from different temporal positions and across variables for spectral analysis. Figure 1(b,c) shows clear spectral concentration in both settings. This consistent spectral concentration indicates the presence of shared dominant temporal relationship patterns across samples and variables. Figure 1(d) further examines these patterns using patch-shuffled and IID Gaussian controls under the same analysis pipeline. Independent patch shuffling preserves each matrix's relationship values and spectrum while disrupting the alignment of relative patch positions across samples and variables. The resulting decrease in spectral concentration indicates that shared relationship patterns recur at corresponding relative temporal positions. Compared with IID Gaussian inputs, real data also exhibit stronger concentration, indicating shared regularities beyond the random-input reference. Further details are provided in Appendix A.

These observations motivate learning reusable temporal structures and adapting their contributions to each sample and variable. We therefore propose the Temporal Structure Network (TStruct). Specifically, TStruct directly parameterizes a set of shared temporal dependency matrices and uses dynamic sample information and static variable priors to generate normalized combination weights. These weights serve as a soft allocation of each sample-variable pair over the shared dependency structures, combining the shared matrices into a sample- and variable-specific temporal dependency matrix. In this way, TStruct models heterogeneous temporal dependencies under the constraint of shared structures, balancing generalizable structure and adaptive flexibility. In addition, we introduce the Parameter-efficient Channel Adaptation Head (PCAHead), which utilizes a shared low-rank weight generation mechanism to dynamically produce specific prediction heads for each variable.

The contributions of this paper are summarized as follows:

• We reveal that patch-level temporal relationship matrices in real world time series contain dominant structures shared across samples from different temporal positions and across variables.

• We propose TStruct, a concise and effective shared-adaptive temporal dependency modeling method that learns shared temporal dependency matrices and uses dynamic sample information and static variable priors to construct sample- and variable-specific dependencies.

• We conduct extensive experiments to validate the effectiveness of TStruct. The results demonstrate that this simple design consistently outperforms several strong baselines on standard long-term forecasting benchmarks and remains effective under diverse generalization and robustness evaluations, while also exhibiting high computational efficiency.

## 2 Related Work

## 2.1 Temporal Dependency Modeling for Time Series Forecasting

Temporal dependency modeling aims to characterize how information interacts across different temporal positions, and is a central problem in long-term time series forecasting. In recent years, Transformer-based and Linear/MLP-based methods have received increasing attention due to their strong performance in long-range dependency modeling and efficient forecasting. Transformer-based methods, such as Informer, Autoformer, PatchTST, Crossformer, and Pathformer [46, 39, 28, 45, 4], usually use attention mechanisms to dynamically estimate pairwise interactions among temporal tokens or patches for each input, thereby modeling long-range temporal dependencies. Their highly flexible input-dependent interactions provide strong adaptivity, but may also capture sample-specific incidental relations in low-SNR time series. Linear/MLP-based methods, such as DLinear, TSMixer, FiTS, SparseTSF, and TimeBase [44, 5, 43, 20, 13], achieve simple and efficient forecasting through shared temporal or feature-wise mixing. These methods demonstrate the strong competitiveness of fixed shared transformations in LTSF, but their adaptability to heterogeneous temporal dependencies remains limited. In contrast, TStruct neither relies on a single fixed mixing rule nor freely estimates interactions for each input. Instead, it generates normalized information aggregation weights through shared temporal dependency structures and sample-variable-conditioned weights, enabling adaptive temporal dependency modeling under shared structural constraints.

## 2.2 Explicit Temporal Structure Parameterization

Recent studies attempt to explicitly parameterize stable structures in time series, rather than relying entirely on generic neural networks to learn them implicitly from data. CycleNet, TQNet, TimeEmb, MOFÓ, and GTR [19, 21, 42, 25, 3] mostly model recurring periodic patterns in data through global learnable periodic representations. These studies suggest that directly parameterizing reusable temporal structures can help improve the generalization ability of time series forecasting. However, existing methods mainly focus on periodic patterns in the input sequence or temporal representations. Temporal dependency structures can contain periodicity, but are not equivalent to periodicity itself; they may also characterize more general relationships among relative temporal positions and patchlevel interaction patterns [9, 28, 40]. Unlike these methods, TStruct shifts the explicit parameterization target from input-level periodic patterns to temporal dependency structures themselves.

## 3 Methodology

Problem Definition. The problem of LTSF can be formulated as follows. Let $\begin{array} { r l } { \mathbf { X } } & { { } = } \end{array}$ $\left\{ \mathbf { x } _ { t - L + 1 } , \ldots , \mathbf { x } _ { t } \right\} \in \mathbb { R } ^ { L \times C }$ denote the sequence of historical observations, where L represents the lookback window length and C denotes the number of variables. At each time step t, the observation vector is defined as $\mathbf { \bar { x } } _ { t } = \{ x _ { t } ^ { ( 1 ) } , \dots , x _ { t } ^ { ( C ) } \}$ , where $x _ { t } ^ { ( c ) }$ represents the value of the c-th variable. Our objective is to learn a mapping function $\mathcal { F } _ { \theta }$ parameterized by θ to predict the future values for the next $H$ time steps, denoted as $\mathbf { Y } = \{ \mathbf { x } _ { t + 1 } , \dots , \mathbf { x } _ { t + H } \} \in \mathbb { R } ^ { H \times C }$ . In this work, we focus on how to learn temporal dependency structures within $\mathcal { F } _ { \theta }$ that remain stable and generalizable across different samples and variables.

## 3.1 Overview

As shown in Figure 2, TStruct contains three components. Following RevIN [15], TStruct applies instance-wise normalization before encoding and restores the original scale after prediction. Patch Embedding first converts each univariate sequence into patch-level representations, enabling local pattern extraction and reducing the length of temporal modeling [28, 41]. The Temporal Structure Encoder (TSEncoder) then learns shared temporal dependency structures and adaptively combines them using dynamic sample information and static variable priors. The resulting dependency scores are normalized into patch-level aggregation weights, allowing the model to capture sample- and variable-specific temporal dependencies under shared structural constraints. Finally, PCAHead provides a parameter-efficient prediction head that accounts for variable heterogeneity by generating variable-aware forecasting projections.

![](images/07a3198f4b6df9895e66e06cc8494bf67fb463c986a4224af44944fb09e6f624.jpg)  
Figure 2: Overall architecture of TStruct. (a) The Temporal Structure Mixer (TSM) generates sampleand variable-specific temporal dependencies by adaptively combining shared temporal dependency structures with dynamic sample information and static variable priors. (b) The PCAHead handles variable heterogeneity by generating variable-aware prediction weights in a parameter-efficient manner.

## 3.2 Patch Embedding

Patch-based representation has been widely used to capture local temporal patterns and reduce the effective sequence length [28]. Given the input sequence of the c-th variable $\mathbf { x } ^ { ( c ) } \in \mathbb { R } ^ { L }$ , we first segment it into $N$ non-overlapping patches $\mathbf { x } _ { \mathrm { p } } ^ { ( c ) } \in \mathbb { R } ^ { N \times P }$ , where $P$ is the patch length and $N = \checkmark / P$ is the number of patches. For simplicity, we assume that L is divisible by ${ \bf \bar { \rho } } _ { P } .$ Subsequently, a learnable linear projection $\mathbf { W } _ { \mathrm { e m b } } \in \mathbb { R } ^ { P \times \hat { D } }$ , shared across all variables, maps these patches into a D-dimensional latent representation:

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { p } } ^ { ( c ) } = \mathrm { U n f o l d } ( \mathbf { x } ^ { ( c ) } ) , \quad \mathbf { E } ^ { ( c ) } = \mathbf { x } _ { \mathrm { p } } ^ { ( c ) } \mathbf { W } _ { \mathrm { e m b } } , } \end{array}\tag{1}
$$

where $\mathbf { E } ^ { ( c ) } \in \mathbb { R } ^ { N \times D }$ denotes the patch-level representation of the c-th variable. This representation provides the basis for TSEncoder to model temporal dependencies among relative patch positions.

## 3.3 Temporal Structure Encoder

The TSEncoder is designed to model patch-level temporal dependencies in a shared yet adaptive manner. Given the patch-level representation $\mathbf { E } \in \mathbb { R } ^ { B \times C \times N \times D }$ , where $B , C , N$ , and D denote the batch size, number of variables, number of patches, and hidden dimension, respectively, TSEncoder learns temporal dependencies among the $N$ relative patch positions. We first parameterize a set of globally shared temporal dependency structures:

$$
\begin{array} { r } { \boldsymbol { B } = \{ \mathbf { B } _ { k } \} _ { k = 1 } ^ { K } , \quad \mathbf { B } _ { k } \in \mathbb { R } ^ { N \times N } , } \end{array}\tag{2}
$$

where K is the number of shared temporal structures. Each $\mathbf { B } _ { k }$ represents a learnable patch-level dependency score pattern.

We initialize the vectorized structure matrices orthogonally, providing distinct initial directions with unit Frobenius norm. This controls the initial scale of the dependency scores before softmax normalization. The importance of initialization is also consistent with findings on learning dynamics in deep linear networks [32]. A detailed initialization study is provided in Appendix C.2.

These shared structures are then combined using sample-variable-specific weights. Specifically, we introduce a learnable static variable prior $\mathbf { Q } _ { \mathrm { s t a t i c } } \mathbf { \check { \Lambda } } \in \mathbb { R } ^ { \pmb { \dot { C } } \times K }$ , which captures variable-specific temporal preferences. A complementary dynamic query is generated by averaging each sample-variable representation along the patch dimension and passing the result through a lightweight router:

$$
\mathbf { Q } _ { \mathrm { d y n a m i c } } = \mathrm { R o u t e r } \left( \mathrm { M e a n } _ { N } ( \mathbf { E } ) \right) ,\tag{3}
$$

where $\mathbf { Q } _ { \mathrm { d y n a m i c } } \in \mathbb { R } ^ { B \times C \times K }$ . The static and dynamic queries are then fused to obtain normalized combination coefficients:

$$
\begin{array} { r } { \pmb { \alpha } _ { b , c } = \mathrm { S o f t m a x } \left( \mathbf { Q } _ { \mathrm { s t a t i c } } ^ { ( c ) } + \mathbf { Q } _ { \mathrm { d y n a m i c } } ^ { ( b , c ) } \right) , } \end{array}\tag{4}
$$

where ${ \pmb { \alpha } } _ { b , c } \in \mathbb { R } ^ { K }$ and the softmax is applied over the K shared structures. The coefficients are nonnegative and sum to one, so the resulting dependency score matrix is a convex combination of the shared bases.

For each sample-variable pair, we compose a temporal dependency score matrix from the shared structures:

$$
\mathbf { S } _ { b , c } = \sum _ { k = 1 } ^ { K } \alpha _ { b , c , k } \mathbf { B } _ { k } , \quad \mathbf { S } _ { b , c } \in \mathbb { R } ^ { N \times N } .\tag{5}
$$

The composed scores are converted into normalized temporal aggregation weights:

$$
\mathbf { A } _ { b , c } = \operatorname { S o f t m a x } _ { \operatorname { r o w } } \left( \mathbf { S } _ { b , c } \right) .\tag{6}
$$

Each row of $\mathbf { A } _ { b , c }$ assigns nonnegative weights summing to one to the source patches. Temporal aggregation therefore forms weighted averages of projected historical features, constraining the operator to select and combine information without arbitrary signed amplification.

Finally, the normalized temporal dependency matrix is used to aggregate patch-level features:

$$
\widetilde { \mathbf { E } } _ { b , c } = \mathbf { A } _ { b , c } \left( \mathbf { E } _ { b , c } \mathbf { W } _ { \mathrm { p r o j } } \right) ,\tag{7}
$$

where $\mathbf { W } _ { \mathrm { p r o j } } \in \mathbb { R } ^ { D \times D }$ is a learnable feature projection. The aggregated representation is further processed by a feed-forward network with residual connections:

$$
\begin{array} { r } { \mathbf { Z } _ { b , c } = \widetilde { \mathbf { E } } _ { b , c } + \mathrm { M L P } \left( \widetilde { \mathbf { E } } _ { b , c } \right) . } \end{array}\tag{8}
$$

Together, the shared bases and normalized aggregation weights restrict temporal mixing to a common family of dependency patterns learned across samples and variables. This provides an inductive bias toward stable, reusable temporal structures, while allowing their contributions to be adjusted to the input context.

## 3.4 Parameter-efficient Channel Adaptation Head

PCAHead maps the encoded features to the prediction horizon while balancing variable heterogeneity and parameter efficiency. A fully shared prediction head is efficient but may be insufficient for variable-specific forecasting patterns, whereas assigning an independent head to each variable leads to a parameter cost proportional to C. PCAHead therefore adopts a shared base projection with variable-specific low-rank residual adaptation.

Given the encoder output, we flatten the patch and feature dimensions into $\mathbf { Z } \in \mathbb { R } ^ { B \times C \times D _ { \mathrm { i n } } }$ , where $D _ { \mathrm { i n } } = N D$ . A shared base projection gives

$$
\mathbf { Y } _ { \mathrm { b a s e } } = \mathbf { Z } \mathbf { W } _ { \mathrm { b a s e } } , \quad \mathbf { W } _ { \mathrm { b a s e } } \in \mathbb { R } ^ { D _ { \mathrm { i n } } \times H } .\tag{9}
$$

For variable-specific adaptation, we define shared low-rank projection banks $\mathbf { U } _ { k } \in \mathbb { R } ^ { D _ { \mathrm { i n } } \times r }$ and $\mathbf { V } _ { k } \in \mathbb { R } ^ { r \times H }$ , where $r \ll D _ { \mathrm { i n } }$ . For each variable c, PCAHead learns normalized coefficients $\pmb { \alpha } _ { c } = \mathrm { S o f t m a x } ( \mathbf { s } ^ { ( c ) } )$ and assembles

$$
\mathbf { W } _ { \mathrm { d o w n } } ^ { ( c ) } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } \mathbf { U } _ { k } , \quad \mathbf { W } _ { \mathrm { u p } } ^ { ( c ) } = \sum _ { k = 1 } ^ { K } \alpha _ { c , k } \mathbf { V } _ { k } .\tag{10}
$$

The final prediction is

$$
\hat { \mathbf { Y } } ^ { ( c ) } = \mathbf { Y } _ { \mathrm { b a s e } } ^ { ( c ) } + \mathbf { Z } ^ { ( c ) } \mathbf { W } _ { \mathrm { d o w n } } ^ { ( c ) } \mathbf { W } _ { \mathrm { u p } } ^ { ( c ) } .\tag{11}
$$

This reduces the head complexity from $O ( C D _ { \mathrm { i n } } H )$ to $O ( D _ { \mathrm { i n } } H + K r ( D _ { \mathrm { i n } } + H ) + C K )$ . We zero-initialize the up-projection bank so that the residual branch starts from zero, allowing the model to first rely on the shared base prediction and then gradually learn variable-specific corrections.

## 4 Experiment

## 4.1 Experiment Setup

Following prior studies, we use Mean Squared Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics. All methods are evaluated under prediction lengths $F = \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ The datasets and baseline models used for evaluation are summarized below. Dataset statistics are provided in Appendix A.1, while implementation details and hyperparameter settings are provided in Appendix B.

Datasets. We evaluate TStruct on seven widely used long-term forecasting benchmarks, including Weather, Traffic, Electricity [40], and four ETT subsets [46]. These datasets cover different domains, variable scales, and temporal patterns, providing a comprehensive evaluation of forecasting performance and generalization ability.

Baselines. We compare TStruct with eight strong and recent forecasting baselines, covering different temporal dependency modeling paradigms. Specifically, PatchTST [28] and PhaseFormer [29] represent Transformer-based methods; DLinear [44] represents simple linear forecasting methods; CycleNet [19], TQNet [21], TimeEmb [42], and GTR [3] explicitly model temporal patterns or periodic priors; and DUET [31] is included as a recent strong forecasting baseline. For all baselines, we report the best results reported in their papers or accompanying code repositories, with the corresponding experimental configurations detailed in Appendix B.

## 4.2 Main results

Table 1 reports the average results over all prediction horizons, with detailed comparisons in Table 10. TStruct achieves the best average MSE on all seven benchmarks and the best average MAE on six, ranking first in 13 of the 14 dataset-metric comparisons. These results demonstrate the effectiveness of shared-adaptive temporal dependency modeling against strong Transformer-based, Linear-based, and explicit temporal-pattern modeling baselines.

On ETT and Weather, TStruct shows clear advantages over existing methods. These datasets often contain non-stationary temporal dynamics and distribution variations, where overly flexible models may fit incidental temporal interactions and overly fixed transformations may be insufficiently adaptive. The consistent improvements on these datasets indicate that TStruct can better capture stable temporal dependency structures while preserving sample-specific adaptability.

Table 1: Average multivariate time series forecasting results over all prediction horizons. The best results are marked in bold red, and the second-best results are marked in blue underline.
<table><tr><td>Methods</td><td>TStruct (ours)</td><td>DUET [2025]</td><td></td><td>PhaseFormer [2026]</td><td>GTR [2026]</td><td>TimeEmb [2025]</td><td>TQNet [2025]</td><td>CycleNet [2024]</td><td>PatchTST [2023]</td><td></td><td>DLinear [2023]</td></tr><tr><td>Datasets</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td></td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>ETTh1</td><td>0.390 0.415</td><td>0.398 0.419</td><td>|0.403</td><td>0.415</td><td>0.415 0.419</td><td>0.418 0.433</td><td>0.420 0.428</td><td>0.415 0.426</td><td>0.417</td><td>0.431</td><td>0.425 0.439</td></tr><tr><td>ETTh2</td><td>0.331 0.381</td><td>0.334 0.383</td><td>0.347</td><td>0.389</td><td>0.346 0.394</td><td>0.353 0.396</td><td>0.358 0.399</td><td>0.345 0.394</td><td>0.351</td><td>0.395</td><td>0.470 0.468</td></tr><tr><td>ETTm1</td><td>0.337 0.367</td><td>0.338 0.369</td><td>0.347</td><td>0.374</td><td>0.345 0.377</td><td>0.345 0.376</td><td>0.353 0.388</td><td>0.352 0.378</td><td>0.349</td><td>0.381</td><td>0.356 0.378</td></tr><tr><td>ETTm2</td><td>0.243 0.304</td><td>0.248 0.307</td><td>0.251</td><td>0.314</td><td>0.255 0.315</td><td>0.248 0.308</td><td>0.258 0.318</td><td>0.249 0.311</td><td>0.256</td><td>0.314</td><td>0.259 0.324</td></tr><tr><td>Weather</td><td>0.213 0.249</td><td>0.218 0.252</td><td>0.223</td><td>0.261</td><td>0.227 0.264</td><td>0.218 0.257</td><td>0.230 0.266</td><td>0.237 0.277</td><td>0.224</td><td>0.262</td><td>0.242 0.293</td></tr><tr><td>Electricity</td><td>0.151 0.244</td><td>0.157 0.248</td><td>0.161</td><td>0.250</td><td>0.160 0.254</td><td>0.158 0.253</td><td>0.162 0.256</td><td>0.157 0.250</td><td>0.162</td><td>0.254</td><td>0.167 0.264</td></tr><tr><td>Traffic</td><td>0.381 0.252</td><td>0.393 0.256</td><td>0.387</td><td>0.250</td><td>0.397 0.274</td><td>|0.399 0.272</td><td>0.404 0.282</td><td>0.406</td><td>0.280 0.396</td><td>0.266</td><td>0.418 0.287</td></tr><tr><td>1st Count</td><td>7 6</td><td>0</td><td>0 0</td><td>2</td><td>0 0</td><td>0 0</td><td>0 0</td><td>0</td><td>0 0</td><td>0</td><td>0 0</td></tr></table>

Table 2: Ablation study on the main components of TStruct. Results are averaged over all prediction horizons. Detailed results for each prediction horizon are provided in Table 11.
<table><tr><td>Dataset</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Weather Electricity</td><td colspan="2">Traffic</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td></tr><tr><td>①</td><td>0.425</td><td>0.444</td><td>0.350</td><td>0.395</td><td>0.376 0.391</td><td>0.259</td><td>0.318</td><td>0.229 0.265</td><td>0.162</td><td>0.254</td><td>0.403 0.270</td></tr><tr><td>②</td><td>0.403</td><td>0.424</td><td>0.341</td><td>0.384 0.347</td><td>0.372</td><td>0.248</td><td>0.308</td><td>0.218 0.254</td><td>0.153</td><td>0.246</td><td>0.382 0.253</td></tr><tr><td>③</td><td>0.394</td><td>0.417</td><td>0.339</td><td>0.386 0.341</td><td>0.368</td><td>0.248</td><td>0.307</td><td>0.215 0.251</td><td>0.152</td><td>0.245</td><td>0.382 0.253</td></tr><tr><td>TStruct</td><td>0.390</td><td>0.416</td><td>0.331</td><td>0.381</td><td>0.337 0.367</td><td>0.243</td><td>0.303</td><td>0.213 0.249</td><td>0.151</td><td>0.244</td><td>0.381 0.252</td></tr></table>

On Electricity and Traffic, Notably, without explicit cross-variable interactions, TStruct achieves the best average MSE on both datasets, with the best average MAE on Electricity and the second-best on Traffic. This result is consistent with our spectral analysis in Appendix A.2: Electricity and Traffic exhibit stronger spectral concentration in patch-level temporal relationship matrices across both samples and variables, suggesting a higher degree of shared temporal dependency structures. Such structures can be effectively captured by TStruct through shared temporal dependency matrices and sample-variable-conditioned adaptation.

## 4.3 Ablation Study

We evaluate the effectiveness of three key components in TStruct. The full model is compared against three variants: ① TSM → MHA replaces the Temporal Structure Mixer with multi-head attention; ② TSM → Linear replaces it with a shared linear temporal mixer; and ③ PCAHead → Linear replaces PCAHead with a shared linear forecasting head. All TStruct variants in the ablation study are trained with the same objective and training protocol. The summary of these ablation results is reported in Table 2.

On ETT and Weather, TStruct clearly outperforms both ① and ② on these datasets. These datasets exhibit stronger temporal distribution variations, where forecasting requires both shared structures for generalization and adaptive modeling for sample- and variable-specific heterogeneity. MHA provides flexible input-dependent interactions but may fit incidental relations, while a shared linear mixer is stable but insufficiently adaptive. The superior performance of TStruct indicates that shared-adaptive temporal dependency modeling is especially beneficial in such settings.

On Electricity and Traffic, The gap between TStruct and ② becomes smaller on these datasets, while TStruct still clearly outperforms ①. This is consistent with our spectral analysis in Appendix A: Electricity and Traffic exhibit stronger cross-sample and cross-variable spectral concentration in patch-level temporal relationship matrices, suggesting more pronounced shared temporal dependency structures. Therefore, simple shared temporal mixing is already highly competitive on these datasets whereas fully input-dependent attention remains less stable.

![](images/01c5bd7d9c30c1e84cc53a0a4ac27a98bee56cd69ce0ff13c4e16481f1b3cf91.jpg)  
(a) Earliest samples.

![](images/c25dec5722e4561329d4e28aac0cba6ef482064abd5b95f02ee4af4081a1fe0e.jpg)  
(b) Random samples.  
Figure 3: Generalization analysis on ETTh1. (a) Models are trained with the earliest timestampordered training samples, evaluating temporal distribution shift. (b) Models are trained with randomly sampled training instances, evaluating sample efficiency.

Finally, replacing PCAHead with a shared linear head leads to moderate but consistent degradation, showing that variable-aware prediction further improves performance under variable heterogeneity.

## 4.4 Generalization Analysis

To evaluate generalization, we conduct two reduced-training experiments on ETTh1. We compare the proposed TSM with two variants that replace TSM with Linear or MHA, keeping all other components and experimental settings identical. The earliest sample setting evaluates temporal distribution shift, while random sample setting evaluates sample efficiency.

Temporal Distribution Shift. As shown in Figure 3(a), models are trained using the earliest 20%, 40%, 60%, 80%, 100% of timestamp-ordered training samples, while validation and test sets remain unchanged. MHA consistently performs worse than Linear and TSM, indicating that fully input-dependent attention is more sensitive to temporally biased training data. TSM is close to Linear when only the earliest 20% samples are used, but clearly outperforms Linear from 40% onward, showing that shared-adaptive temporal dependency modeling improves generalization once sufficient shared structures can be estimated.

Sample Efficiency. As shown in Figure 3(b), TSM achieves the best performance even with only 20% randomly sampled training data and quickly converges around 40%. Notably, TSM trained with 40% data already outperforms MHA and Linear trained with the full training set. This suggests that shared temporal structures improve data efficiency by reducing the need to relearn temporal interactions from individual samples.

## 4.5 Effect of Look-back Length

To examine how different methods utilize historical context, we vary the look-back length from 48 to 720 and compare TStruct with explicit temporal-pattern parameterization methods, including CycleNet [19], TQNet [21], TimeEmb [42], and GTR [3]. These baselines mainly rely on global periodic representations, which can remain effective even with short input windows because the periodic priors are largely inputindependent. However, as the look-back length increases, their improvements become limited and may even degrade when overly long histories introduce noisy or less relevant observations.

![](images/15e9d3f17873054a76b568582ece75b2c20f2058e7185f90c81c2133189cbdd1.jpg)  
Figure 4: Effect of look-back length on ETTh1.

![](images/7b272243c53222a91e6c70d3b8db5daf40a209968b895cfb1894cb55e1263218.jpg)

![](images/8780848c8f4be4b792b79c74322650cebe878ed7f6f6f39ce92e0da367fe38b3.jpg)

![](images/596919b4662c01b7ebdf84cbaf34d6b2e2c19af011aca462234aaa8ce0ad0464.jpg)  
(b) ACF of selected variables

(a) Learned static prior Qstatic  
![](images/163a74b24ec251ff6fba91b78799252c461703a531d31cda6d9db50f5a0175b3.jpg)  
(c) Representative learned structures  
Figure 5: Interpretability analysis on the Weather dataset. Different variables exhibit different ACF patterns and assign different weights to the learned shared temporal structures through $Q _ { \mathrm { s t a t i c } } .$ The representative learned structures show distinct patch-level dependency patterns, including broader recurrent relations and local diagonal-dominant continuity.

In contrast, TStruct consistently benefits from longer look-back windows. This suggests that TStruct does not merely rely on fixed global periodic patterns; instead, it explicitly models patch-level temporal dependency structures, which become richer as more historical context is available. This further indicates that patch-level temporal dependency structures can provide a more extensible modeling target than fixed global periodic priors when longer historical context is available.

## 4.6 Interpretability of Learned Temporal Structures

To examine whether the learned structures are interpretable, we visualize the learned static prior, the autocorrelation functions (ACF) of representative variables, and the corresponding temporal dependency structures on the Weather dataset. As shown in Figure 5, Var 16 exhibits clear periodic autocorrelation, while Var 14 shows rapidly decaying autocorrelation, indicating that these two variables have different temporal characteristics. The learned static prior $\mathbf { Q } _ { \mathrm { s t a t i c } }$ assigns different preferences to these variables over the shared structures. Specifically, Var 16 places higher weight on Structure #1, which captures broader recurrent relations among patches, whereas Var 14 assigns higher weight to Structure #2, which shows strong diagonal dominance and reflects local temporal continuity. This suggests that $\mathbf { Q } _ { \mathrm { s t a t i c } }$ captures variable-specific temporal preferences rather than using the same dependency pattern for all variables. Overall, these observations indicate that TStruct learns reusable temporal dependency structures and adapts them to variables with different temporal characteristics through the static prior. For additional visualizations of learned structure matrices on other datasets, please refer to Appendix D.

## 5 Conclusion

In this paper, we studied LTSF from the perspective of temporal dependency structures. By analyzing patch-level temporal relationship matrices, we found that temporal dependencies contain shared dominant structures across samples and variables. This observation suggests that stable temporal relationships can be explicitly modeled as reusable structures rather than treated as fully input-specific interactions. Based on this finding, we proposed TStruct, which learns shared temporal dependency structures and generates sample-variable-specific dependency scores within the shared structural space. These scores are further converted into normalized soft allocation weights for temporal information aggregation, enabling adaptive dependency modeling under shared structural constraints. Experiments show that TStruct consistently improves forecasting accuracy over strong baselines.

Ablation and generalization analyses further verify that the gains come from shared-adaptive temporal dependency modeling, while look-back length and interpretability analyses show that TStruct can exploit longer contexts and learn meaningful temporal structures.

## References

[1] Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=gerNCVqqtR.

[2] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael K Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=ARAxPPIAhq.

[3] Fanpu Cao, Lu Dai, Jindong Han, and Hui Xiong. Enhancing multivariate time series forecasting with global temporal retrieval. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=QUJBPSfyui.

[4] Peng Chen, Yingying ZHANG, Yunyao Cheng, Yang Shu, Yihang Wang, Qingsong Wen, Bin Yang, and Chenjuan Guo. Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=1Jk0CMP2aW.

[5] Si-An Chen, Chun-Liang Li, Sercan O Arik, Nathanael Christian Yoder, and Tomas Pfister. TSMixer: An all-MLP architecture for time series forecast-ing. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview .net/forum?id= wbpxTuXgm0.

[6] Luo donghao and wang xue. ModernTCN: A modern pure convolution structure for general time series analysis. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=vpJMJerXHU.

[7] Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods. Econometrica: journal of the Econometric Society, pages 424–438, 1969.

[8] C.W.J. Granger and P. Newbold. Spurious regressions in econometrics. Journal of Econometrics, 2(2):111–120, 1974. ISSN 0304-4076. doi: https://doi.org/10.1016/0304-4076(74)90034-7. URLhttps://www.sciencedirect.com/science/article/pii/0304407674900347.

[9] James D Hamilton. Time series analysis, volume 2. Princeton Princeton University Press, 1994.

[10] Lu Han, Han-Jia Ye, and De-Chuan Zhan. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. IEEE Trans. Knowl. Data Eng., 36(11):7129–7142, 2024.

[11] Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, and Yang Wang. CrossGNN: Confronting noisy multivariate time series via cross interaction refinement. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=x0zlW2vUYc.

[12] Qihe Huang, Zhengyang Zhou, Yangze Li, Kuo Yang, Binwu Wang, and Yang Wang. Many minds, one goal: Time series forecasting via sub-task specialization and inter-agent cooperation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=Uon41HfqR3.

[13] Qihe Huang, Zhengyang Zhou, Kuo Yang, Zhongchao Yi, Xu Wang, and Yang Wang. Timebase: The power of minimalism in efficient long-term time series forecasting. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview .net/ forum?id=GhTdNOMfOD

[14] Qihe Huang, Zhengyang Zhou, Yangze Li, Jiaming Ma, Kuo Yang, Binwu Wang, Xu Wang, and Yang Wang. STORM: Synergistic cross-scale spatio-temporal modeling for weather forecasting. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=JLF6XDnscF.

[15] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift In International Conference on Learning Representations, 2022. URL https : //openreview . net/forum?id=cGDAkQo1COp.

[16] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster), 2015.

[17] Zhe Li, Shiyi Qi, Yiduo Li, and Zenglin Xu. Revisiting long-term time series forecasting: An investigation on linear mapping, 2026. URL https://arxiv.org/abs/2305.10721.

[18] Zida Liang, Jiayi Zhu, and Weiqiang Sun. Why attention fails: The degeneration of transformers into mlps in time series forecasting, 2025. URL https ://arxiv. org/abs/2509 .20942.

[19] Shengsheng Lin, Weiwei Lin, Xinyi HU, Wentai Wu, Ruichao Mo, and Haocheng Zhong. Cyclenet: Enhancing time series forecasting through modeling periodic patterns. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=clBiQUgj4w.

[20] Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, and Junjie Yang. SparseTSF: Modeling long-term time series forecasting with \*1k\* parameters. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=54NSH001Fe.

[21] Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, and Weiwei Lin. Temporal query network for efficient multivariate time series forecasting. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=e24CueVty2.

[22] Shengsheng Lin, Weiwei Lin, Wentai Wu, Feiyu Zhao, Ruichao Mo, and Haotong Zhang. Segrnn: Segment recurrent neural network for long-term time-series forecasting. IEEE Internet Things J., 13(5):9861–9871, 2026.

[23] Minhao LIU, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia LAI, Lingna Ma, and Qiang Xu. SCINet: Time series modeling and forecasting with sample convolution and interaction. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https ://openreview .net/forum?id= AyajSjTAzmg.

[24] Peiyuan Liu, Beiliang Wu, Yifan Hu, Naiqi Li, Tao Dai, Jigang Bao, and Shu-Tao Xia. Timebridge: Non-stationarity matters for long-term time series forecasting. In Forty-second International Conference on Machine Learning, 2025. URL https ://openreview . net/forum? id=pyK00ZZ51z.

[25] Jiaming Ma, Binwu Wang, Qihe Huang, Guanjun Wang, Pengkun Wang, Zhengyang Zhou, and Yang Wang. Mofo: Empowering long-term time series forecasting with periodic pattern modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=sbvLts2HqR.

[26] Jiaming Ma, Guanjun Wang, Sheng Huang, Kuo Yang, Binwu Wang, Pengkun Wang, and Yang Wang. Spatiotemporal causal decoupling model for air quality forecasting. In ICASSP 2025- 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

[27] Jiaming Ma, Qihe Huang, Haofeng Ma, Guanjun Wang, Sheng Huang, Zhengyang Zhou, Pengkun Wang, Xu Wang, Binwu Wang, and Yang Wang. PHAT: Modeling period heterogeneity for multivariate time series forecasting. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=lr4RlISR6x.

[28] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview .net/forum? id=Jbdc0vT0col.

[29] Yiming Niu, Jinliang Deng, and Yongxin Tong. Phaseformer: From patches to phases for efficient and effective time series forecasting. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=Lk9SqMQzhX.

[30] Xiangfei Qiu, Jilin Hu, Lekui Zhou, Xingjian Wu, Junyang Du, Buang Zhang, Chenjuan Guo Aoying Zhou, Christian S. Jensen, Zhenli Sheng, and Bin Yang. TFB: towards comprehensive and fair benchmarking of time series forecasting methods. Proc. VLDB Endow., 17(9):2363– 2377, 2024.

[31] Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang. DUET: dual clustering enhanced multivariate time series forecasting. In KDD (1), pages 1185–1196. ACM, 2025.

[32] Mark Rhee, Dhruva Karkada, and Jamie Simon. Deep linear networks are a surprisingly useful toy model of weight-space dynamics. Learning Mechanics, 2026. URL https : // learningmechanics.pub/deep-linear-nets.

[33] Mingtian Tan, Mike A Merrill, Vinayak Gupta, Tim Althoff, and Thomas Hartvigsen. Are language models actually useful for time series forecasting? In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https : //openreview net/forum?id=DV15UbHCY1

[34] Hao Wang, Lichen Pan, Yuan Shen, Zhichao Chen, Degui Yang, Yifei Yang, Sen Zhang, Xinggao Liu, Haoxuan Li, and Dacheng Tao. FreDF: Learning to forecast in the frequency domain. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=4A9IdSa1ul.

[35] Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: Multi-scale local and global context modeling for long-term series forecasting. In The Eleventh International Conference on Learning Representations, 2023. URL https ://openreview . net/forum?id=zt53IDUR1U.

[36] Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, and JUN ZHOU. Timemixer: Decomposable multiscale mixing for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024. URL https : //openreview.net/forum?id=7oLshfEIC2.

[37] Zheng Wang, Kaixuan Zhang, Wanfang Chen, Xiaonan Lu, Longyuan Li, and Tobias Schlagenhauf. Characteristic root analysis and regularization for linear time series forecasting. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=JTtwGRACte.

[38] Gerhard Widmer and Miroslav Kubat. Learning in the presence of concept drift and hidden contexts. Mach. Learn., 23(1):69–101, 1996.

[39] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems 2021. URL https://openreview.net/forum?id=J4gRj6d5Qm.

[40] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview .net/forum? id=ju\_Uqw3840q.

[41] Xingjian Wu, Xiangfei Qiu, Hanyin Cheng, Zhengyu Li, Jilin Hu, Chenjuan Guo, and Bin Yang. Enhancing time series forecasting through selective representation spaces: A patch perspective. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=BirE0jYKt0.

[42] Mingyuan Xia, Chunxu Zhang, Zijian Zhang, Hao Miao, Qidong Liu, Yuanshao Zhu, and Bo Yang. Timeemb: A lightweight static-dynamic disentanglement framework for time series forecasting. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=sLfMvrkn6T.

[43] Zhijian Xu, Ailing Zeng, and Qiang Xu. FITS: Modeling time series with \$10k\$ parameters. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=bWcnvZ3qMb.

[44] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In AAAI, pages 11121–11128. AAAI Press, 2023.

[45] Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=vSVLM2j9eie.

[46] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI, pages 11106–11115. AAAI Press, 2021.

[47] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In ICML, volume 162 of Proceedings of Machine Learning Research, pages 27268–27286. PMLR, 2022.

## Appendix

## Catalogue

1. Introduction.   
2. Related Work   
2.1 Temporal Dependency Modeling for Time Series Forecasting . . . .3   
2.2 Explicit Temporal Structure Parameterization .. ..   
3. Methodology   
3.1 Overview   
3.2 Patch Embedding   
3.3 Temporal Structure Encoder.   
3.4 Parameter-efficient Channel Adaptation Head . .   
4. Experiment   
4.1 Experiment Setup   
4.2 Main Results   
4.3 Ablation Study   
4.4 Generalization Analysis .   
4.5 Effect of Look-back Length ..   
4.6 Interpretability of Learned Temporal Structures .   
5. Conclusion .   
Appendix.   
A. Dataset and Temporal Structure Analysis ..   
B. Implementation Details. . 18   
C. Additional Experimental Analyses 19   
D. Additional Interpretability Visualizations .. 24   
E. Detailed Experimental Results 28   
F. Limitations and Future Directions 30

## A Dataset and Temporal Structure Analysis

## A.1 Dataset Details

We evaluate TStruct on seven widely used long-term forecasting benchmarks, including four ETT subsets, Weather, Electricity, and Traffic. The statistics of these datasets are summarized in Table 3. The ETT datasets [46] contain temperature and power load records collected from electricity transformers in two regions of China from 2016 to 2018. They include two temporal granularities: hourly records, denoted as ETTh1 and ETTh2, and 15-minute records, denoted as ETTm1 and ETTm2. The Weather dataset [40] contains 21 meteorological variables recorded at 10-minute intervals in 2020, including air temperature, humidity, visibility, and other weather-related indicators. The Electricity dataset [40] consists of hourly electricity consumption records from 321 clients, measured in kilowatt-hours (kWh), and is derived from the UCI Machine Learning Repository. The Traffic dataset [40] records hourly road occupancy rates collected by 862 sensors on freeways in the San Francisco Bay Area, reflecting large-scale traffic flow dynamics.

Table 3: Detailed descriptions of datasets.
<table><tr><td>Dataset</td><td>Variables</td><td>Timestamps</td><td>Split</td><td>Prediction Length</td><td>Sampling</td><td>Information</td></tr><tr><td>ETTh1</td><td>7</td><td>14400</td><td>6:2:2</td><td>{96,192,336,720}</td><td>Hourly</td><td>Electricity</td></tr><tr><td>ETTh2</td><td>7</td><td>14400</td><td>6:2:2</td><td>{96,192,336,720}</td><td>Hourly</td><td>Electricity</td></tr><tr><td>ETTm1</td><td>7</td><td>57600</td><td>6:2:2</td><td>{96,192,336,720}</td><td>15min</td><td>Electricity</td></tr><tr><td>ETTm2</td><td>7</td><td>57600</td><td>6:2:2</td><td>{96,192,336,720}</td><td>15min</td><td>Electricity</td></tr><tr><td>Weather</td><td>21</td><td>52696</td><td>7:1:2</td><td>{96,192,336,720}</td><td>10min</td><td>Weather</td></tr><tr><td>ECL</td><td>321</td><td>26304</td><td>7:1:2</td><td>{96,192,336,720}</td><td>Hourly</td><td>Electricity</td></tr><tr><td>Traffic</td><td>862</td><td>17544</td><td>7:1:2</td><td>{96,192,336,720}</td><td>Hourly</td><td>Transport</td></tr></table>

## A.2 Temporal Structure Analysis

Patch-level temporal relationship matrix. Real-world time series often have low-SNR, and temporal interactions observed in individual windows may reflect incidental fluctuations. We hypothesize that these observations nevertheless contain relatively stable temporal structures associated with the underlying evolution, which can be extracted from recurring relationships across windows and variables. To examine this hypothesis, we construct patch-level temporal relationship matrices that describe relationships among relative temporal positions. For each dataset, we first standardize each variable and randomly sample B input windows of length $L .$ Given the b-th input window $\mathbf { X } _ { b } \in \mathbb { R } ^ { L \times C }$ , we apply instance-level normalization along the temporal dimension:

$$
\widetilde { \mathbf { X } } _ { b } = \frac { \mathbf { X } _ { b } - \mathrm { M e a n } _ { L } ( \mathbf { X } _ { b } ) } { \mathrm { S t d } _ { L } ( \mathbf { X } _ { b } ) + \epsilon } ,\tag{12}
$$

where $\mathrm { M e a n } _ { L } ( \cdot )$ and $\mathrm { S t d } _ { L } ( \cdot )$ are computed along the temporal dimension for each variable.

Then, for each variable $c ,$ the normalized sequence is divided into $N$ non-overlapping patches with patch length $P \colon$

$$
\mathbf { X } _ { b , c } ^ { p } = \operatorname { U n f o l d } ( \widetilde { \mathbf { X } } _ { b } ^ { ( c ) } ) \in \mathbb { R } ^ { N \times P } ,\tag{13}
$$

where $N = L / P$ for non-overlapping patches, and we assume $L$ is divisible by $P .$

Each patch vector is further normalized by its $\ell _ { 2 } \cdot$ -norm:

$$
\overline { { \mathbf { X } } } _ { b , c } ^ { p } [ i ] = \frac { \mathbf { X } _ { b , c } ^ { p } [ i ] } { \| \mathbf { X } _ { b , c } ^ { p } [ i ] \| _ { 2 } + \epsilon } , \quad i = 1 , \ldots , N .\tag{14}
$$

We then construct the patch-level temporal relationship matrix as

$$
\mathbf { G } _ { b , c } = \overline { { \mathbf { X } } } _ { b , c } ^ { p } \left( \overline { { \mathbf { X } } } _ { b , c } ^ { p } \right) ^ { \top } \in \mathbb { R } ^ { N \times N } ,\tag{15}
$$

where ${ \bf G } _ { b , c } [ i , j ]$ measures the normalized relationship strength between the ¿-th and j-th relative patch positions.

Shared temporal structures across samples and variables. To examine shared temporal structures, we stack the vectorized relationship matrices across samples and across variables. For the sampleside analysis, each variable c is fixed and the matrices from all B windows are collected; for the variable-side analysis, each sample b is fixed and the matrices from all $C$ variables are collected:

$$
\begin{array} { r } { \mathcal { G } _ { c } ^ { ( \mathrm { s a m p l e } ) } = \left[ \mathrm { v e c } ( \mathbf { G } _ { 1 , c } ) , \ldots , \mathrm { v e c } ( \mathbf { G } _ { B , c } ) \right] ^ { \top } \in \mathbb { R } ^ { B \times N ^ { 2 } } , } \\ { \mathcal { G } _ { b } ^ { ( \mathrm { v a r } ) } = \left[ \mathrm { v e c } ( \mathbf { G } _ { b , 1 } ) , \ldots , \mathrm { v e c } ( \mathbf { G } _ { b , C } ) \right] ^ { \top } \in \mathbb { R } ^ { C \times N ^ { 2 } } . } \end{array}\tag{16}
$$

Before performing SVD, the rows of both matrices are normalized to compare temporal relationship patterns on a common scale:

$$
\overline { { \mathcal { G } } } [ j ] = \frac { \mathcal { G } [ j ] } { \| \mathcal { G } [ j ] \| _ { 2 } + \epsilon } ,\tag{17}
$$

where $\mathcal { G }$ denotes either stacked matrix and j indexes its rows. This normalization removes differences in vector magnitude while preserving the directions of the temporal relationship representations.

Let $\{ \sigma _ { c , i } ^ { ( \mathrm { s a m p l e } ) } \} _ { i } \mathrm { a n d } \{ \sigma _ { b , i } ^ { ( \mathrm { v a r } ) } \}$ i denote the ordered singular values of $\overline { { \mathcal { G } } } _ { c } ^ { ( \mathrm { s a m p l e } ) }$ and $\overline { { \mathcal { G } } } _ { b } ^ { ( \mathrm { v a r } ) }$ , respectively. The averaged spectral energies are defined as

$$
\bar { e } _ { i } ^ { ( \mathrm { s a m p l e } ) } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \left( \sigma _ { c , i } ^ { ( \mathrm { s a m p l e } ) } \right) ^ { 2 } , \qquad \bar { e } _ { i } ^ { ( \mathrm { v a r } ) } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left( \sigma _ { b , i } ^ { ( \mathrm { v a r } ) } \right) ^ { 2 } .\tag{18}
$$

These spectra indicate the extent to which temporal relationship patterns are concentrated in a small number of directions across samples and variables, respectively.

Spectral concentration metrics. We quantify spectral concentration by measuring how many singular components are required to explain 95% of the spectral energy. Given an averaged spectral energy sequence $\{ \bar { e } _ { i } \} _ { i = 1 } ^ { R }$ , the 95% energy rank is defined as

$$
r _ { 9 5 } = \operatorname* { m i n } \left\{ r : \frac { \sum _ { i = 1 } ^ { r } \bar { e } _ { i } } { \sum _ { i = 1 } ^ { R } \bar { e } _ { i } } \geq 0 . 9 5 \right\} .\tag{19}
$$

Here $R$ denotes the reference dimension of the full vectorized representation:

$$
R = \left\{ \begin{array} { l l } { \mathrm { m i n } ( C , N ^ { 2 } ) , } & { \mathrm { f o r } \mathcal { G } _ { b } ^ { \mathrm { ( v a r ) } } , } \\ { \mathrm { m i n } ( B , N ^ { 2 } ) , } & { \mathrm { f o r } \mathcal { G } _ { c } ^ { \mathrm { ( s a m p l e ) } } . } \end{array} \right.\tag{20}
$$

To make the degree of spectral concentration more intuitive, we report the rank compression factor:

$$
\kappa _ { 9 5 } = \frac { R } { r _ { 9 5 } } .\tag{21}
$$

A larger $\kappa _ { 9 5 }$ indicates that 95% of the spectral energy is captured by a smaller fraction of the reference dimension, corresponding to stronger spectral concentration. However, spectral concentration can also arise from the Gram construction itself: symmetry introduces redundant coordinates in the vectorized representation, while patch normalization produces a common unit diagonal across matrices. A reference level is therefore needed to assess the additional concentration associated with temporal structure. To this end, we evaluate two null controls through the same analysis pipeline and compare their $\kappa _ { 9 5 }$ values with those of the real data.

Quantitative analysis. Table 4 reports the spectral concentration statistics under $B \ = \ 2 0 0 0$ $L = 7 2 0$ , and $P = \mathrm { \bar { 2 4 } } \left( N = 3 0 \right)$ , together with two null controls evaluated through the same pipeline. (1) Patch shuffle evaluates the role of temporal ordering by independently permuting patch positions for each sample-variable pair, giving $\mathbf { G } _ { b , c } ^ { \mathrm { s h u f f e } } = \Pi _ { b , c } \mathbf { \tilde { G } } _ { b , c } ^ { \ast } \mathbf { I I } _ { b , c } ^ { \top ^ { \ast } }$ . This preserves the spectrum and similarity values of each Gram matrix while disrupting the correspondence between relative patch positions across matrices. Comparing the resulting concentration with that of real data therefore assesses whether the observed concentration depends on temporal relationships recurring at aligned patch positions. (2) ID Gaussian provides a complementary reference by replacing the observed windows with independent Gaussian noise. Applying the same analysis to these inputs gives the spectral concentration that can arise without temporal dependencies, against which the additional concentration in real data can be assessed.

Table 4: Shuffle denotes independent patch permutations and IID denotes independent Gaussian windows. Results are averaged over repeated runs of each null control.  
(a) Sample side: across temporal windows
<table><tr><td>Dataset</td><td>Patch size P</td><td>Patch number N</td><td>Variables C</td><td>Real</td><td>Shuffle</td><td>IID</td><td>Real/Shuffle</td><td>Real/IID</td></tr><tr><td>ETTh1</td><td>24</td><td>30</td><td>7</td><td>5.59</td><td>2.80</td><td>2.61</td><td>2.00×</td><td>2.14×</td></tr><tr><td>ETTh2</td><td>24</td><td>30</td><td>7</td><td>5.73</td><td>2.68</td><td>2.61</td><td>2.14×</td><td>2.20×</td></tr><tr><td>ETTm1</td><td>24</td><td>30</td><td>7</td><td>4.74</td><td>2.41</td><td>2.61</td><td>1.96×</td><td>1.82×</td></tr><tr><td>ETTm2</td><td>24</td><td>30</td><td>7</td><td>4.27</td><td>2.43</td><td>2.61</td><td>1.76×</td><td>1.64×</td></tr><tr><td>Weather</td><td>24</td><td>30</td><td>21</td><td>5.92</td><td>2.44</td><td>2.61</td><td>2.43×</td><td>2.27×</td></tr><tr><td>Electricity</td><td>24</td><td>30</td><td>321</td><td>90.00</td><td>8.04</td><td>2.61</td><td>11.20×</td><td>34.50×</td></tr><tr><td>Traffic</td><td>24</td><td>30</td><td>862</td><td>64.29</td><td>6.12</td><td>2.61</td><td>10.50×</td><td>24.64×</td></tr></table>

(b) Variable side: across variables within a window
<table><tr><td>Dataset</td><td>Patch size P</td><td>Patch number N</td><td>Variables C</td><td>Real</td><td>Shuffle</td><td>IID</td><td>Real/Shuffle</td><td>Real/IID</td></tr><tr><td>ETTh1</td><td>24</td><td>30</td><td>7</td><td>1.75</td><td>1.17</td><td>1.00</td><td>1.50×</td><td>1.75×</td></tr><tr><td>ETTh2</td><td>24</td><td>30</td><td>7</td><td>1.40</td><td>1.00</td><td>1.00</td><td>1.40×</td><td>1.40×</td></tr><tr><td>ETTm1</td><td>24</td><td>30</td><td>7</td><td>1.40</td><td>1.00</td><td>1.00</td><td>1.40×</td><td>1.40×</td></tr><tr><td>ETTm2</td><td>24</td><td>30</td><td>7</td><td>1.40</td><td>1.00</td><td>1.00</td><td>1.40×</td><td>1.40×</td></tr><tr><td>Weather</td><td>24</td><td>30</td><td>21</td><td>2.33</td><td>1.11</td><td>1.11</td><td>2.11×</td><td>2.11×</td></tr><tr><td>Electricity</td><td>24</td><td>30</td><td>321</td><td>64.20</td><td>12.84</td><td>1.66</td><td>5.00×</td><td>38.60×</td></tr><tr><td>Traffic</td><td>24</td><td>30</td><td>862</td><td>287.33</td><td>8.62</td><td>2.90</td><td>33.33×</td><td>99.00×</td></tr></table>

## Table 4 reveals four findings:

(i) Shared structures at aligned temporal positions. Across all seven datasets, Real/Shuffle is substantially greater than one on both the sample and variable sides. The reduction in concentration after shuffling supports shared temporal relationship patterns that recur at corresponding relative patch positions across windows and variables.

(ii) Concentration beyond the random-input reference. Real/IID is also substantially greater than one across all datasets on both sides. Temporal relationships in real time series are therefore more concentrated than those obtained from independent random inputs under the same analysis pipeline, supporting their representation through a compact set of shared patterns.

(iii) Larger gains in Electricity and Traffic. Electricity and Traffic exhibit substantially larger Real/Shuffle and Real/IID gains than ETT and Weather on both sides. Their temporal relationships show more pronounced concentration relative to their respective controls, with the larger Real/Shuffle gains indicating a greater dependence of this concentration on the original patch-position alignment.

(iv) Shared regularities preserved under shuffling. For Electricity and Traffic, shuffled-data concentration remains well above the corresponding IID reference on both sides, whereas the two controls yield closer values for ETT and Weather. This suggests that shared regularities preserved under patch permutation coexist with reusable temporal relationships tied to the original relative positions.

From temporal relationships to dependency parameterization. The above analysis provides evidence of shared temporal relationship structures across windows and variables. To connect these historical relationships with forecasting, we consider an idealized model in which historical and future patch representations are related by a shared temporal evolution operator.

For a sample-variable context i, let $\mathbf { X } _ { i } \in \mathbb { R } ^ { N \times d }$ and $\mathbf { Y } _ { i } \in \mathbb { R } ^ { M \times d }$ denote row-normalized historical and future patch representations in a common state space, where $N = L / P$ and $M = H / P$ assuming both lengths are divisible by P. Suppose

$$
\mathbf { Y } _ { i } = \mathbf { T } \mathbf { X } _ { i } + \varepsilon _ { i } , \qquad \mathbb { E } [ \varepsilon _ { i } \mid \mathbf { X } _ { i } ] = 0 , \qquad \mathbf { T } \in \mathbb { R } ^ { M \times N } ,\tag{22}
$$

where T is shared across the contexts under consideration and $\varepsilon _ { i }$ captures residual variation. The historical auto-relation and history-future cross-relation then satisfy

$$
\mathbf { G } _ { i } = \mathbf { X } _ { i } \mathbf { X } _ { i } ^ { \top } , \qquad \mathbf { F } _ { i } = \mathbf { Y } _ { i } \mathbf { X } _ { i } ^ { \top } = \mathbf { T } \mathbf { G } _ { i } + \varepsilon _ { i } \mathbf { X } _ { i } ^ { \top } ,\tag{23}
$$

and hence $\mathbb { E } [ \Gamma _ { i } \mid \mathbf { X } _ { i } ] = \mathbf { T } \mathbf { G } _ { i }$

Motivated by the observed spectral concentration, suppose the historical relation family admits an approximate representation through a mean structure and r shared directions:

$$
\mathbf { G } _ { i } = \pmb { \mu } _ { G } + \sum _ { \ell = 1 } ^ { r } \beta _ { i , \ell } \mathbf { Q } _ { \ell } + \pmb { \Delta } _ { i } ,\tag{24}
$$

where $\Delta _ { i }$ denotes the approximation residual. It follows that

$$
{ \mathbf { \Gamma } } _ { i } =  { \mathbf { T } } \pmb { \mu } _ { G } + \sum _ { \ell = 1 } ^ { r } \beta _ { i , \ell } (  { \mathbf { T } }  { \mathbf { Q } } _ { \ell } ) + \left(  { \mathbf { T } }  { \mathbf { A } } _ { i } + \varepsilon _ { i }  { \mathbf { X } } _ { i } ^ { \top } \right) .\tag{25}
$$

Thus, under shared temporal evolution, the historical directions $\mathbf { Q } _ { \ell }$ give rise to shared directions $\mathbf { T Q } _ { \ell }$ in the history-future relation family. When the residual terms are small, the same coefficients $\beta _ { i , \ell }$ organize both families. This provides a conditional link between reusable historical relationships and a compact description of relationships relevant to forecasting.

Motivated by this connection, TStruct parameterizes temporal aggregation through shared structure matrices:

$$
\mathbf { A } _ { i } = \mathrm { S o f t m a x } _ { \mathrm { r o w } } \left( \sum _ { k = 1 } ^ { K } \alpha _ { i , k } \mathbf { B } _ { k } \right) , \qquad \widetilde { \mathbf { Z } } _ { i } = \mathbf { A } _ { i } \mathbf { Z } _ { i } ,\tag{26}
$$

where $\mathbf { Z } _ { i }$ denotes projected historical features. Learned jointly across training observations under forecasting supervision, the shared matrices $\mathbf { B } _ { k }$ define a compact family of temporal aggregation operators $\mathbf { A } _ { i }$ on historical patch coordinates. This parameterization is intended to capture stable temporal dependency patterns relevant to forecasting, while the forecasting head maps the aggregated features to the future horizon. The connection motivates the shared structural parameterization of $\mathbf { A } _ { i }$ without requiring equality with $\mathbf { G } _ { i }$ or $\mathbf { { { T } } } _ { i } ,$ or inheritance of their spectra. By drawing on patterns shared across observations, TStruct aims to reduce reliance on incidental relationships in noisy time series and improve forecasting generalization. These intended benefits are evaluated in Section 4.4 and Appendix C.4.

## B Implementation Details

All experiments performed in this study are conducted on a workstation equipped with NVIDIA RTX PRO 6000 GPUs with 96GB memory. TStruct is implemented in PyTorch 2.8.0. We fix the random seed to 2026 for our runs. Following the standard evaluation protocol [30], we set drop\_last=False for the data loader during testing to ensure that all test samples are evaluated.

For the main forecasting experiments, TStruct uses a fixed input look-back length of $L = 7 2 0$ . We train TStruct using the Adam optimizer [16] for at most 25 epochs with an early stopping patience of 10 epochs. The best checkpoint is selected according to the validation loss.

Baseline configurations.To compare the best performance of each method within the evaluated configurations, we retain the training objectives used by the respective methods and allow look-back selection where specified below. The sources and configurations of the results in Table 1 are as follows: TStruct. We report our results using the Huber+FFT objective described below and a fixed look-back length of $L = 7 2 0$ . DUET. We use the paper reported results obtained with the L1 objective, selecting the best performing look-back length from $L \stackrel { \cdot } { \in } \{ 9 6 , 1 9 2 , 3 3 6 , 5 1 2 , 7 2 0 \}$ for each dataset horizon setting. PhaseFormer. We use the paper reported results obtained with the Huber objective and $L = 7 2 0$ . Other baselines. We run their released implementations with their native training objectives and select the best performing look-back length from $L \in \{ 9 6 , 1 9 2 , 3 3 6 , 5 1 2 , 7 2 0 \}$ for each dataset horizon setting.

Training objective. We optimize TStruct with a hybrid objective that combines Huber loss and FFT loss [34]. Given the prediction Ý and ground truth Y, the Huber loss is defined as

$$
\mathcal { L } _ { \mathrm { H u b e r } } = \frac { 1 } { \vert \Omega \vert } \sum _ { i \in \Omega } \ell _ { \delta } ( \hat { y } _ { i } - y _ { i } ) ,\tag{27}
$$

where Ω denotes all predicted entries and we set $\delta = 0 . 1$ in all experiments. The element-wise Huber loss is defined as

$$
\ell _ { \delta } ( e ) = \left\{ { \begin{array} { l l } { \frac { 1 } { 2 } } e ^ { 2 } , } & { | e | \leq \delta , } \\ { \delta \left( | e | - \frac { 1 } { 2 } \delta \right) , } & { | e | > \delta . } \end{array}  \right.\tag{28}
$$

The FFT loss measures the discrepancy between predictions and ground truth in the frequency domain:

$$
\mathcal { L } _ { \mathrm { F F T } } = \left. \mathrm { r F T } ( \hat { \mathbf { Y } } , \mathrm { d i m } = 1 ) - \mathrm { r F F T } ( \mathbf { Y } , \mathrm { d i m } = 1 ) \right. _ { 1 } ,\tag{29}
$$

where rFF $\mathrm { T } ( \cdot , \mathrm { d i m } = 1 )$ denotes the real-valued fast Fourier transform along the temporal dimension, and $\| \cdot \| _ { 1 }$ is implemented as the mean absolute value over all frequency-domain entries. The final training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { H u b e r } } + \lambda _ { \mathrm { F F T } } \mathcal { L } _ { \mathrm { F F T } } , } \end{array}\tag{30}
$$

where $\lambda _ { \mathrm { F F T } }$ is selected from {0.0, 0.1, 0.2, 0.3}.

Evaluation metrics. Following prior LTSF studies [46, 39], we use Mean Squared Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics:

$$
\mathrm { M S E } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } , \quad \mathrm { M A E } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } | \hat { y } _ { i } - y _ { i } | .\tag{31}
$$

Hyperparameter search space. We tune the hyperparameters over the following search space. The patch size is selected from $P \in \{ 2 4 , 4 8 , 7 \dot { 2 } , 1 2 \bar { 0 } , 1 8 0 , 3 6 0 \}$ , the number of encoder layers from {1, 3}, the hidden dimension from $D \in \{ 1 2 8 , 2 5 6 , 5 1 2 \}$ , and the number of shared temporal structures from $K \in \{ 1 , 2 , 4 , 5 , 8 , 1 6 , 3 2 \}$ . For regularization, the MLP dropout rate is selected from $\{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ , the TSM dropout rate from {0.0, 0.1, 0.2, 0.3, 0.4}, and the PCAhead dropout rate from {0.0, 0.1, 0.2}. For optimization, the batch size is selected from {16, 64, 256, 512}, and the learning rate η is selected from {0.001, 0.0015, 0.002, 0.005, 0.015}.

## C Additional Experimental Analyses

## C.1 Random Seed Stability

Table 5: Stability analysis under different random seeds.
<table><tr><td colspan="2">Seed</td><td colspan="2">2024</td><td colspan="2">2025</td><td colspan="2">2026</td><td colspan="2">2027</td><td colspan="2">2028</td><td colspan="2">Mean</td><td colspan="2">Std</td></tr><tr><td>Dataset</td><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.345</td><td>0.380</td><td>0.344</td><td>0.380</td><td>0.345</td><td>0.381</td><td>0.344</td><td>0.390</td><td>0.345</td><td>0.380</td><td>0.345</td><td>0.382</td><td>0.001</td><td>0.004</td></tr><tr><td>192</td><td>0.382</td><td>0.404</td><td>0.383</td><td>0.406</td><td>0.383</td><td>0.405</td><td>0.384</td><td>0.406</td><td>0.384</td><td>0.406</td><td>0.383</td><td>0.405</td><td>0.001</td><td>0.001</td></tr><tr><td>336</td><td>0.407</td><td>0.421</td><td>0.405</td><td>0.421</td><td>0.405</td><td>0.420</td><td>0.407</td><td>0.422</td><td>0.407</td><td>0.421</td><td>0.406</td><td>0.421</td><td>0.001</td><td>0.001</td></tr><tr><td>720</td><td>0.424</td><td>0.453</td><td>0.426</td><td>0.453</td><td>0.431</td><td>0.456</td><td>0.428</td><td>0.456</td><td>0.424</td><td>0.450</td><td>0.427</td><td>0.454</td><td>0.003</td><td>0.003</td></tr><tr><td>Avg.</td><td>0.390</td><td>0.415</td><td>0.390</td><td>0.415</td><td>|0.391</td><td>0.416</td><td>|0.391</td><td>0.419</td><td>0.390</td><td>0.414</td><td>0.390</td><td>0.416</td><td>0.001</td><td>0.002</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.137</td><td>0.179</td><td></td><td></td><td>0.137</td><td>0.181</td><td>0.138</td><td>0.180</td><td>0.138</td><td>0.180</td><td>0.137</td><td>0.180</td><td>0.001</td><td>0.001</td></tr><tr><td>192</td><td>0.181</td><td>0.225</td><td>0.137 0.182</td><td>0.179 0.225</td><td>0.181</td><td>0.226</td><td>0.180</td><td>0.225</td><td>0.181</td><td>0.225</td><td>0.181</td><td>0.225</td><td>0.001</td><td>0.000</td></tr><tr><td>336</td><td>0.231</td><td>0.267</td><td>0.231</td><td>0.268</td><td>0.230</td><td>0.267</td><td>0.231</td><td>0.269</td><td>0.230</td><td>0.268</td><td>0.231</td><td>0.268</td><td>0.001</td><td>0.001</td></tr><tr><td>720</td><td>0.306</td><td>0.325</td><td>0.305</td><td>0.325</td><td>0.304</td><td>0.324</td><td>0.305</td><td>0.325</td><td>0.305</td><td>0.326</td><td>0.305</td><td>0.325</td><td>0.001</td><td>0.001</td></tr><tr><td>Avg.</td><td>0.214</td><td>0.249</td><td>0.214</td><td>0.249</td><td>0.213</td><td>0.250</td><td>0.214</td><td>0.250</td><td>0.214</td><td>0.250</td><td>0.214</td><td>0.249</td><td>0.000</td><td>0.000</td></tr></table>

To examine the stability of TStruct under different random initializations and stochastic training processes, we repeat experiments with five random seeds, i.e., 2024, 2025, 2026, 2027, and 2028. Table 5 reports the results on ETTh1 and Weather, including the mean and standard deviation across seeds.

The results show that TStruct is highly stable across different random seeds. The standard deviation is consistently very small, mostly ranging from 0.000 to 0.002 for both MSE and MAE. This indicates that the performance of TStruct is not sensitive to random initialization or stochastic optimization, and the observed improvements are reproducible rather than caused by favorable random seeds.

Table 6: Sensitivity analysis under different initialization strategies.
<table><tr><td colspan="2">Initialization</td><td colspan="2">Random</td><td colspan="2">Orthogonal</td><td colspan="2">Xavier</td><td colspan="2">Kaiming</td></tr><tr><td>Dataset</td><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.366</td><td>0.398</td><td>0.345</td><td>0.380</td><td>0.345</td><td>0.380</td><td>0.345</td><td>0.380</td></tr><tr><td>192</td><td>0.421</td><td>0.439</td><td>0.383</td><td>0.405</td><td>0.383</td><td>0.405</td><td>0.383</td><td>0.405</td></tr><tr><td>336</td><td>0.426</td><td>0.442</td><td>0.407</td><td>0.420</td><td>0.407</td><td>0.420</td><td>0.407</td><td>0.420</td></tr><tr><td>720</td><td>0.497</td><td>0.503</td><td>0.433</td><td>0.456</td><td>0.434</td><td>0.459</td><td>0.430</td><td>0.458</td></tr><tr><td>Avg.</td><td>0.428</td><td>0.446</td><td>0.392</td><td>0.415</td><td>0.392</td><td>0.416</td><td>0.391</td><td>0.416</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.142</td><td>0.187</td><td>0.137</td><td>0.181</td><td>0.137</td><td>0.180</td><td>0.137</td><td>0.180</td></tr><tr><td>192</td><td>0.187</td><td>0.230</td><td>0.181</td><td>0.226</td><td>0.182</td><td>0.226</td><td>0.181</td><td>0.225</td></tr><tr><td>336</td><td>0.247</td><td>0.282</td><td>0.230</td><td>0.267</td><td>0.231</td><td>0.267</td><td>0.230</td><td>0.268</td></tr><tr><td>720</td><td>0.327</td><td>0.347</td><td>0.304</td><td>0.324</td><td>0.304</td><td>0.325</td><td>0.305</td><td>0.325</td></tr><tr><td>Avg.</td><td>0.226</td><td>0.262</td><td>0.213</td><td>0.250</td><td>0.214</td><td>0.250</td><td>0.213</td><td>0.250</td></tr></table>

## C.2 Initialization Study

We study the effect of different initialization strategies for the shared temporal structures B. Specifically, we compare Random, Orthogonal, Xavier, and Kaiming initialization, as reported in Table 6. The results show that the Random baseline yields noticeably higher forecasting errors on both ETTh1 and Weather, whereas Orthogonal, Xavier, and Kaiming achieve consistently strong and comparable performance.

All initialization strategies are applied directly to the $K \times N \times N$ basis tensor. Orthogonal initialization gives the vectorized bases mutually orthogonal directions with unit Frobenius norm, while Xavier and Kaiming scale the sampling variance according to the tensor dimensions. These strategies control the initial magnitude of the dependency scores, which are subsequently converted into aggregation weights through softmax. This provides a plausible explanation for their favorable performance: appropriately scaled initial scores moderate arbitrary preferences over patch positions, allowing temporal dependency patterns to develop through forecasting supervision. The role of initialization scale is also consistent with observations on learning dynamics in deep linear networks [32].

We use Orthogonal initialization as the default. It provides diverse initial structure directions with a common scale, reducing initial redundancy among the shared bases while achieving consistently strong empirical performance.

## C.3 Training Objective Comparison

We investigate the effect of the training objective on forecasting performance by evaluating TStruct, DUET, and PhaseFormer under the same two objectives, Huber+FFT and plain MSE. The remaining experimental settings follow Appendix B.

Table 7 shows that TStruct ranks first in MSE on all seven datasets and in MAE on six under both objectives, with PhaseFormer leading in MAE on Traffic. Huber+FFT further improves TStruct's MSE and MAE across all seven datasets compared with plain MSE. Its strong performance under plain MSE supports the effectiveness of shared temporal structure modeling, while the hybrid objective provides additional gains.

## C.4 Noise Injection Robustness

To evaluate robustness under local abnormal perturbations, we conduct noise injection experiments on ETTh1. For each normalized input window, whose values are standardized to approximately zero mean and unit variance, we randomly select a continuous segment and inject Gaussian noise into this segment. The noise ratio controls the length of the perturbed segment relative to the input window, while the noise strength controls the magnitude of the injected Gaussian noise. The start position of the noisy segment is randomly sampled within each input window.

We compare TSM and MHA across different noise ratios and strengths, with the results summarized in Table 8. TSM achieves lower MSE and MAE throughout both experiments. As the noise ratio

Table 7: Forecasting performance under two training objectives with $L = 7 2 0$ for all methods. The lowest MSE and MAE for each dataset within each objective are highlighted in bold.
<table><tr><td>Loss</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td></td><td colspan="3">Huber+FFT</td><td colspan="2"></td><td colspan="2">Plain MSE</td><td colspan="2"></td></tr><tr><td>Method</td><td>TStruct</td><td>DUET</td><td></td><td>PhaseFormer</td><td>TStruct</td><td>DUET</td><td></td><td>PhaseFormer</td><td></td></tr><tr><td>Datasets</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.390 0.415</td><td>0.430</td><td>0.450</td><td>0.417 0.425</td><td>0.410</td><td>0.432</td><td>0.425 0.447</td><td>0.431</td><td>0.437</td></tr><tr><td>ETTh2</td><td>0.331 0.381</td><td>0.364</td><td>0.410</td><td>0.374 0.401</td><td>0.349</td><td>0.400</td><td>0.358 0.405</td><td>0.358</td><td>0.402</td></tr><tr><td>ETTm1</td><td>0.337 0.367</td><td>0.432</td><td>0.426</td><td>0.351 0.378</td><td>0.345</td><td>0.373</td><td>0.382 0.409</td><td>0.358</td><td>0.384</td></tr><tr><td>ETTm2</td><td>0.243 0.304</td><td>0.282</td><td>0.342</td><td>0.253 0.314</td><td>0.246</td><td>0.311</td><td>0.279 0.338</td><td>0.266</td><td>0.329</td></tr><tr><td>Weather</td><td>0.213 0.249</td><td>0.218</td><td>0.252</td><td>0.231 0.271</td><td>0.217 0.157</td><td>0.258</td><td>0.249 0.293</td><td>0.231</td><td>0.275</td></tr><tr><td>Electricity</td><td>0.151 0.244</td><td>0.158</td><td>0.251</td><td>0.159 0.249</td><td></td><td>0.251</td><td>0.158 0.256</td><td>0.161</td><td>0.252</td></tr><tr><td>Traffic</td><td>0.381 0.252</td><td>0.387</td><td>0.267</td><td>0.387 0.249</td><td>0.385</td><td>0.264</td><td>0.388 0.270</td><td>0.393</td><td>0.259</td></tr></table>

increases from 0.05 to 0.30, its MSE ranges from 0.392 to 0.400, remaining 0.042–0.047 below MHA. At a noise strength of 8, the MSE gap reaches 0.075, with TSM obtaining 0.404 compared with 0.479 for MHA.

Table 8: Noise injection robustness on ETTh1 under varying noise ratios (left) and strengths (right). ∆ denotes the error reduction of TSM relative to MHA.
<table><tr><td>Setting</td><td colspan="4">Varying noise ratio</td><td>Setting</td><td colspan="4">Varying noise strength</td></tr><tr><td>Method</td><td colspan="2">TSM</td><td>MHA</td><td>Δ</td><td></td><td>Method</td><td>TSM</td><td>MHA</td><td></td><td>Δ</td></tr><tr><td>Ratio</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>Strength</td><td>MSE</td><td>MAE MSE</td><td>MAE</td><td>MSE MAE</td></tr><tr><td>0.05</td><td>0.392</td><td>0.417</td><td>0.434 0.449</td><td>0.042</td><td>0.033</td><td>2</td><td>0.393 0.417</td><td>0.437</td><td>0.452 0.044</td><td>0.035</td></tr><tr><td>0.10</td><td>0.394</td><td>0.418</td><td>0.441 0.455</td><td>0.047</td><td>0.036</td><td>3</td><td>0.394 0.418</td><td>0.441</td><td>0.455</td><td>0.047 0.036</td></tr><tr><td>0.20</td><td>0.397</td><td>0.425</td><td>0.443 0.457</td><td>0.046</td><td>0.033</td><td>5</td><td>0.396 0.420</td><td>0.434</td><td>0.449</td><td>0.038 0.029</td></tr><tr><td>0.30</td><td>0.400</td><td>0.427</td><td>0.442 0.458</td><td>0.042</td><td>0.031</td><td>8</td><td>0.404 0.426</td><td>0.479</td><td>0.473</td><td>0.075 0.047</td></tr></table>

These results support the robustness of shared temporal structure modeling under local perturbations. By aggregating temporal information through structures learned across observations, TSM provides a shared basis for processing corrupted inputs, which may help explain its consistently lower forecasting errors across the tested noise conditions.

## C.5 Hyperparameter Sensitivity

We further analyze the sensitivity of TStruct to two key hyperparameters: the number of shared temporal structures K and the number of patches N. Experiments are conducted on the ETT benchmarks under different prediction horizons.

Figure 6 reports the forecasting performance under different values of K. Overall, TStruct remains stable across a wide range of K. A very small K may limit the diversity of shared temporal structures, making it difficult to represent heterogeneous temporal dependencies. In contrast, an overly large K may introduce redundant structures and increase optimization difficulty. The results show that a moderate number of shared structures is sufficient for strong performance, and we use K = 4 as the default setting in our main experiments.

Figure 7 shows the sensitivity to the number of patches N, which controls the resolution of the patch-level temporal dependency matrix. TStruct achieves robust performance within a broad middle range, especially when $\bar { N } \in [ 1 0 , 3 0 ]$ . When N is too small, the temporal relationship matrix becomes overly coarse and may underfit fine-grained temporal dependencies. When N is too large, each patch contains fewer time points, which may make the learned dependency structures more sensitive to local noise and increase the risk of overfitting. These results indicate that TStruct is robust to reasonable patch-level resolutions while benefiting from a balanced choice of N.

## C.6 Efficiency and Complexity Analysis

We provide an additional efficiency analysis on the Electricity dataset to evaluate the accuracyefficiency trade-off of TStruct. Since TStruct models temporal dependencies at the patch level, we compare it with representative patch-based Transformer baselines, including Crossformer and PatchTST. All models are evaluated under the same setting with input length $L = 7 2 0 .$ patch size and stride $P = S = 4 8$ , and hidden dimension $D = 5 1 2$ . We report forecasting accuracy, parameter number, multiply-accumulate operations (MACs), GPU memory usage, training time per step, and inference time per batch in Table 9.

Complexity analysis. The efficiency of TStruct mainly comes from its structured temporal dependency modeling. In attention-based patch-level temporal mixing, the dependency score matrix is computed from input-dependent query-key interactions, which introduces a score estimation cost of $\dot { O (} N ^ { 2 } D )$ , where N is the number of patches and D is the hidden dimension. In contrast, TSM constructs temporal dependency scores by combining K shared temporal structures, with a score construction cost of ${ \mathcal { O } } { \dot { ( N ^ { 2 } K ) } }$ . Since $K \ll D$ , TSM provides a more compact way to estimate temporal dependency scores and avoids fully input-dependent pairwise attention computation.

After obtaining the normalized dependency matrix, TSM still performs patch-level feature aggregation, so the overall temporal mixing remains quadratic in N. The key difference is that TSM replaces unconstrained input-dependent attention score estimation with shared-adaptive structure composition. This makes temporal dependency modeling more structured and computationally efficient while preserving sample- and variable-specific adaptability.

Empirical efficiency. As shown in Table 9, TStruct achieves the best MSE while requiring fewer parameters, lower MACs, less GPU memory, and faster training and inference than the patch-based Transformer baselines. Compared with PatchTST, TStruct reduces parameters from 5.496M to 3.324M and MACs from 23.134G to 11.779G, while reducing MSE from 0.171 to 0.151. Compared with Crossformer, TStruct also achieves substantially lower memory usage and faster runtime. These results show that TStruct improves forecasting accuracy with a favorable efficiency profile by replacing fully input-dependent attention with shared-adaptive temporal dependency modeling.

Table 9: Efficiency comparison on the Electricity dataset. Crossformer [45] and PatchTST [28] are included as representative patch-based Transformer baselines. All models are evaluated under the same setting. We report MSE, parameters, MACs, GPU memory, training time per step, and inference time per batch. Bold indicates the best result, and underline indicates the second-best result.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=2>MSE  Params (M)</td><td rowspan=1 colspan=1>MACs (G)</td><td rowspan=1 colspan=1>Memory (GB)</td><td rowspan=1 colspan=1>Train Time (s)</td><td rowspan=1 colspan=1>Infer Time (ms)</td></tr><tr><td rowspan=1 colspan=1>Crossformer</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>36.936</td><td rowspan=1 colspan=1>47.543</td><td rowspan=1 colspan=1>22.932</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>75.474</td></tr><tr><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>5.496</td><td rowspan=1 colspan=1>23.134</td><td rowspan=1 colspan=1>12.848</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>326.133</td></tr><tr><td rowspan=1 colspan=1>TStruct</td><td rowspan=1 colspan=1>0.151</td><td rowspan=1 colspan=1>3.324</td><td rowspan=1 colspan=1>11.779</td><td rowspan=1 colspan=1>7.143</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>56.059</td></tr><tr><td rowspan=1 colspan=1>Improvement (↓)</td><td rowspan=1 colspan=1>1.13×</td><td rowspan=1 colspan=1>1.65×</td><td rowspan=1 colspan=1>1.96×</td><td rowspan=1 colspan=1>1.80×</td><td rowspan=1 colspan=1>4.56×</td><td rowspan=1 colspan=1>1.35×</td></tr></table>

![](images/ea977feebc43599830a1b66130723e25f2924b090442157227eeb45688ac800d.jpg)

![](images/65f628803aa3b55b0c48152564929c8570c0f910ea3cca02b40e3880ccfced19.jpg)

![](images/2682c5a663702348d749de942b83eb447b45835a3f76be27d0c5622b2263e67b.jpg)

![](images/6ca6b68e7c7811891053c0a93d66776a5ecc41cdf73060fae31679ea7732137b.jpg)

![](images/3cdc330f8d619a47350dbcb450327e9277e8de5ba8093f21163fcc9adc0355eb.jpg)

![](images/894b8a78a3ff48a272cc729d2d07c7145efddfca946fb474d4ae1d6256ed3f82.jpg)

Figure 6: Hyperparameter sensitivity analysis for the number of shared temporal structures K on the ETT benchmarks. Bar plots denote MSE and line plots denote MAE under different prediction horizons.  
![](images/237903087167da7a348cd9f017de4ccb2d314c8b0512a066ee288b4175ce9ca5.jpg)

![](images/2ed37e3a306b691395b74b890d53924dd3f3468ce34f5fdd36e7badf39c56a28.jpg)

![](images/3c63279afa0cb205e4cbcd6dca383b8e02f0f6a334b1354266b3d0120ce0d649.jpg)

![](images/242dbf9761b31a1df491d366e9fe8d0ddf3d1c49082c4b55ac096c558728ef28.jpg)

![](images/143dc36a6378c5ff2ef221fb799c313fb2f10de3e6b7af52465866226cfc7b84.jpg)  
Figure 7: Hyperparameter sensitivity analysis for the number of patches N on the ETT benchmarks. TStruct remains stable within a broad middle range of patch-level resolutions, while overly small or overly large N may lead to underfitting or overfitting.

## D Additional Interpretability Visualizations

We provide additional visualizations of the learned temporal structures across all benchmark datasets. Figures 8 to 14 directly visualize the learned shared temporal structure matrices before softmax normalization, as the concentration of weights after softmax can make finer structural differences difficult to distinguish visually. Red and blue indicate higher and lower relative aggregation preferences, respectively, rather than positive and negative dependencies. The visualizations show that the learned structures are not random mixing patterns. Instead, they exhibit structured and reusable temporal dependency patterns across datasets, including diagonal-dominant local continuity, repeated stripe-like interactions, smooth long-range transitions, and more abstract patch-level dependency patterns.

Several observations can be made. First, many structures contain clear diagonal or near-diagonal patterns, indicating that neighboring patches often preserve strong local temporal continuity. Second, repeated stripe-like or checkerboard patterns appear in several datasets, suggesting recurrent interactions between relative patch positions. Third, some structures exhibit smooth low-to-high transitions across rows or columns, reflecting broader temporal dependency tendencies beyond local neighborhoods. These patterns show that TStruct does not rely on a single predefined temporal prior, such as periodicity alone.

The learned structures also vary across datasets. The ETT datasets often contain a mixture of local continuity, alternating recurrent patterns, and smooth transition structures, reflecting heterogeneous temporal dynamics. Weather, Electricity, and Traffic exhibit more regular banded or repeated structures, indicating that TStruct can capture dataset-specific temporal regularities through shared dependency matrices. Overall, these visualizations provide qualitative evidence that the temporal structures learned by TStruct are interpretable, structured, and adaptive to different datasets.

![](images/a7e33b4f9e3c31c6838c59ba47ed10863f25097bbdaaa16dfd71cce5322059d1.jpg)

![](images/173a72bd80987207dc551a8faa6f5b332cd25c2eda92e89c676f165d4bb059e4.jpg)

![](images/0b810738c0c9486079271dcbeabf0f0525e6dbc4e6fc6ad2e4271b37ea63dfce.jpg)  
Figure 8: Learned Temporal Structures on ETTh1.

![](images/f3f9f97f0458ba26e91c61ea899a172e2b3f42bf6d8b9937dc79c6adb497ccd9.jpg)

![](images/02efe8dfcd27cb8b3c5da76760e51237ede5efbf02289070968406d78ab88e30.jpg)

![](images/2a3b5755c8420220fd455368fcd0c12b0fb2066ce81b2e8c874a6a254d54f4a4.jpg)

![](images/8e0d2843f0b337cfe4204ffaf0421ad6957f47d9addcb2d4c01eb000b7d83f49.jpg)

Figure 9: Learned Temporal Structures on ETTh2.  
![](images/9aafab71190e6495a08fcb8c6957072f3524bfb55b1d153041150523d8825973.jpg)

![](images/26ce3b5b95a5e080830c1b069277c3d1c052f3bbf6d67d5c892d024ea2a17878.jpg)  
Figure 11: Learned Temporal Structures on ETTm2.

![](images/3912710d3984ec6743f1784da15637e9257b5406fe9cebfdfd4374138e322b69.jpg)

![](images/3f79dd57b6edcae0ec94af79c2186ff3313427cdf5a8bc0be2f6508bd90a1970.jpg)

![](images/355bdbbd620b2fc543a5370aa7fdf5a0ddc3aba480c8c429bbc4c768cf364085.jpg)

![](images/50c0205d20d15c7f1e03e6a3c58545e087d089916d28f49ae6d32e2653e67bc7.jpg)

![](images/defb695940c1723b48708ce587c41e2f104bb5b20199ea69c0bed71de9f4ae0c.jpg)

![](images/6935e0927f00c394c1b39e43d3a5a1a6ae2d53674d271a6b1bec761690f40e8b.jpg)

![](images/617c24a3abf631584bef524c435b71dcb3442b9d9ad30cda85321dcb147a2882.jpg)

![](images/6d1c96c9f6cf8f3c1f6610403ccf520859354b23ff59fdc0d3c69a042cbfdf16.jpg)

![](images/0e644a46fe05c6f9f2400d470c184ec0f864835307eb3778caece792d58f2f0f.jpg)

![](images/55bf46bfa0dc13e44ddd6ff8d0308dd1345f03256337f81dbf87c14cebe18c58.jpg)

![](images/348c062808c354a4cc09c735984a625ad291b4f909dc97ffc45402985088df4e.jpg)

![](images/c59e536414f8459226a493afbcf50d042d86b127cc3868290493a46e5dc94020.jpg)

![](images/6b44046867cdb844fa06acce7b547aaa593b7bc47687e4978b39d9b08ccd7326.jpg)  
Figure 12: Learned Temporal Structures on Weather.

![](images/fa1ba591f691c90741da76ce2e1c81939662f5f0c043c6aa1df6f7c86daa83eb.jpg)

![](images/5fc893170f247df6521e47e1f33251c1dfbd82d2225b3835d07ca0ac044f6fb0.jpg)

![](images/25c0328e39a6eab3ed61e0ca36aeac7e2f9031804d9becb20d093b5d0b88c2fd.jpg)

![](images/e5b8e294f8dfae9278801cfaba3d08ad3bb1737208c4bed96c3cd45392add558.jpg)

![](images/3ac768a028bccb83d1826c03d43d3877e0ce2c082054c32ce5e01024a390c9bb.jpg)

![](images/6664280e39d52a30a32db27c9a920d0ab483a2e5dd07d418ac9e7ca184513bde.jpg)

![](images/acfcaecd5843d5ca6475fd18ba50d08374e8d4a51d9b7129965d26af3dc9f3bf.jpg)  
Figure 13: Learned Temporal Structures on Electricity.

![](images/fa1f879cd7f88fdc2c1fd3f46bd84771c08807a68518bd50096a9cdbd07be0e5.jpg)  
Figure 14: Learned Temporal Structures on Traffic.

## E Detailed Experimental Results

This section provides the complete experimental results corresponding to the summarized tables in the main paper. While the main text reports average results over all prediction horizons for compactness, here we present the detailed results under each prediction length H ∈ {96, 192, 336, 720}. These results provide a more fine-grained comparison across horizons and further support the consistency of TStruct under different forecasting settings.

## E.1 Detailed Forecasting Results

Table 10 reports the full forecasting results of TStruct and all baseline methods on seven benchmarks. The results include both MSE and MAE under each prediction horizon, providing the detailed counterpart to the averaged results reported in the main paper.

Table 10: Full multivariate time series forecasting results. The best results are marked in bold red, and the second-best results are marked in blue underline
<table><tr><td rowspan=1 colspan=2>Methods</td><td rowspan=1 colspan=1>TStruct(ours)</td><td rowspan=1 colspan=1>DUET[2025]</td><td rowspan=1 colspan=1>PhaseFormer[2026]</td><td rowspan=1 colspan=1>GTR[2026]</td><td rowspan=1 colspan=2>TimeEmb   TQNet[2025]    [2025]</td><td rowspan=1 colspan=1>CycleNet[2024]</td><td rowspan=1 colspan=1>PatchTST[2023]</td><td rowspan=1 colspan=1>DLinear[2023]</td></tr><tr><td rowspan=1 colspan=2>Datasets|Metric |</td><td rowspan=1 colspan=1>MSE MAE|</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSEMAE</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSE MAE</td><td rowspan=1 colspan=1>MSE MAE</td></tr><tr><td rowspan=5 colspan=1>ETTh1</td><td rowspan=4 colspan=1>96192336720</td><td rowspan=4 colspan=1>0.3450.3800.3830.4050.4070.4210.4270.454</td><td rowspan=1 colspan=1>0.3520.384</td><td rowspan=1 colspan=1>|0.3590.382</td><td rowspan=1 colspan=1>|0.3670.391</td><td rowspan=1 colspan=1>0.3720.400</td><td rowspan=1 colspan=1>|0.3710.393</td><td rowspan=1 colspan=1>|0.374 0.396</td><td rowspan=1 colspan=1>|0.3750.399</td><td rowspan=1 colspan=1>0.3790.403</td></tr><tr><td rowspan=1 colspan=1>0.3980.409</td><td rowspan=1 colspan=1>0.3970.404</td><td rowspan=1 colspan=1>0.4090.420</td><td rowspan=1 colspan=1>0.4130.427</td><td rowspan=1 colspan=1>0.4100.418</td><td rowspan=1 colspan=1>0.4060.415</td><td rowspan=1 colspan=1>0.4140.421</td><td rowspan=1 colspan=1>0.4080.419</td></tr><tr><td rowspan=1 colspan=1>0.4140.426</td><td rowspan=1 colspan=1>0.4250.424</td><td rowspan=1 colspan=1>0.4320.434</td><td rowspan=1 colspan=1>0.438 0.443</td><td rowspan=1 colspan=1>0.4320.433</td><td rowspan=1 colspan=1>0.4310.430</td><td rowspan=1 colspan=1>0.431 0.436</td><td rowspan=1 colspan=1>0.4400.440</td></tr><tr><td rowspan=1 colspan=1>0.4290.455</td><td rowspan=1 colspan=1>0.4310.450</td><td rowspan=1 colspan=1>0.4510.463</td><td rowspan=1 colspan=1>0.4490.462</td><td rowspan=1 colspan=1>0.4660.468</td><td rowspan=1 colspan=1>0.4500.464</td><td rowspan=1 colspan=1>0.4490.466</td><td rowspan=1 colspan=1>0.4710.493</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.3900.415</td><td rowspan=1 colspan=1>0.3980.419|</td><td rowspan=1 colspan=1>0.4030.415</td><td rowspan=1 colspan=1>|0.4150.419|</td><td rowspan=1 colspan=1>0.418 0.433 |</td><td rowspan=1 colspan=1>|0.420 0.428 |</td><td rowspan=1 colspan=1>0.4150.426||</td><td rowspan=1 colspan=1>0.417 0.431 |</td><td rowspan=1 colspan=1>|0.4250.439</td></tr><tr><td rowspan=5 colspan=1>ETTh2</td><td rowspan=4 colspan=1>96192336720</td><td rowspan=2 colspan=1>0.2670.3300.3220.371</td><td rowspan=1 colspan=1>0.2700.336</td><td rowspan=1 colspan=1>|0.2750.3380.274</td><td rowspan=1 colspan=1>0.341|</td><td rowspan=1 colspan=1>0.2900.348</td><td rowspan=1 colspan=1>|0.290 0.349 |</td><td rowspan=2 colspan=1>0.2710.337|0.3320.380</td><td rowspan=2 colspan=1>0.274 0.3370.3480.384</td><td rowspan=2 colspan=1>|0.3000.3640.3870.423</td></tr><tr><td rowspan=1 colspan=1>0.3320.374</td><td rowspan=1 colspan=1>0.3410.376</td><td rowspan=1 colspan=1>0.3400.385</td><td rowspan=1 colspan=1>0.3470.387</td><td rowspan=1 colspan=1>0.3670.397</td></tr><tr><td rowspan=1 colspan=1>0.3570.400</td><td rowspan=1 colspan=1>0.3530.397</td><td rowspan=1 colspan=1>0.3690.405</td><td rowspan=1 colspan=1>0.3630.409</td><td rowspan=1 colspan=1>0.3760.411</td><td rowspan=1 colspan=1>0.3630.406</td><td rowspan=1 colspan=1>0.3620.408</td><td rowspan=1 colspan=1>0.3770.416</td><td rowspan=1 colspan=1>0.4900.487</td></tr><tr><td rowspan=1 colspan=1>0.3810.426</td><td rowspan=1 colspan=1>0.3820.425</td><td rowspan=1 colspan=1>0.4020.436</td><td rowspan=1 colspan=1>0.4050.439</td><td rowspan=1 colspan=1>0.3990.439</td><td rowspan=1 colspan=1>0.4120.443</td><td rowspan=1 colspan=1>0.4150.449</td><td rowspan=1 colspan=1>0.406 0.441</td><td rowspan=1 colspan=1>0.7040.597</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.331 0.381</td><td rowspan=1 colspan=1>0.3340.383|</td><td rowspan=1 colspan=1>0.3470.389</td><td rowspan=1 colspan=1>|0.346 0.394||</td><td rowspan=1 colspan=1>0.3530.396</td><td rowspan=1 colspan=1>|0.358 0.399 |</td><td rowspan=1 colspan=1>0.345 0.394||</td><td rowspan=1 colspan=1>0.351 0.395|</td><td rowspan=1 colspan=1>|0.4700.468</td></tr><tr><td rowspan=5 colspan=1>ETTm1</td><td rowspan=2 colspan=1>96192</td><td rowspan=2 colspan=1>0.2750.3250.3200.354</td><td rowspan=2 colspan=1>0.2790.3330.3200.358</td><td rowspan=2 colspan=1>|0.2930.3440.3230.361</td><td rowspan=2 colspan=1>|0.2830.3370.3230.365</td><td rowspan=1 colspan=1>|0.293 0.346</td><td rowspan=1 colspan=1>|0.294 0.345 |</td><td rowspan=2 colspan=1>0.2990.348|0.3340.367</td><td rowspan=2 colspan=1>0.289 0.3430.3290.368</td><td rowspan=1 colspan=1>|0.3000.345</td></tr><tr><td rowspan=1 colspan=1>0.3260.366</td><td rowspan=1 colspan=1>0.3360.372</td><td rowspan=1 colspan=1>0.3360.366</td></tr><tr><td rowspan=2 colspan=1>336720</td><td rowspan=1 colspan=1>0.3550.378</td><td rowspan=1 colspan=1>0.3480.377</td><td rowspan=1 colspan=1>0.3580.381</td><td rowspan=1 colspan=1>0.3550.384</td><td rowspan=1 colspan=1>0.3560.382</td><td rowspan=1 colspan=1>0.3680.392</td><td rowspan=1 colspan=1>0.3640.387</td><td rowspan=1 colspan=1>0.3620.390</td><td rowspan=1 colspan=1>0.3670.386</td></tr><tr><td rowspan=1 colspan=1>0.4000.411</td><td rowspan=1 colspan=1>0.4050.408</td><td rowspan=1 colspan=1>0.4120.410</td><td rowspan=1 colspan=1>0.4200.422</td><td rowspan=1 colspan=1>0.4050.410</td><td rowspan=1 colspan=1>0.4120.443</td><td rowspan=1 colspan=1>0.4100.411</td><td rowspan=1 colspan=1>0.4160.423</td><td rowspan=1 colspan=1>0.4190.416</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.3370.3670.338</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.3470.374|</td><td rowspan=1 colspan=1>0.3450.377</td><td rowspan=1 colspan=1>|0.345 0.376|</td><td rowspan=1 colspan=1>|0.353 0.388 |</td><td rowspan=1 colspan=1>0.352 0.378||</td><td rowspan=1 colspan=1>0.349 0.381 ||</td><td rowspan=1 colspan=1>0.356 0.378</td></tr><tr><td rowspan=5 colspan=1>ETTm2</td><td rowspan=3 colspan=1>96192336</td><td rowspan=3 colspan=1>0.1530.2410.2090.2820.2640.321</td><td rowspan=1 colspan=1>0.1610.248</td><td rowspan=1 colspan=1>|0.1630.256</td><td rowspan=1 colspan=1>|0.1610.252</td><td rowspan=1 colspan=1>0.1630.251</td><td rowspan=1 colspan=1>|0.1660.255</td><td rowspan=1 colspan=1>|0.1590.247</td><td rowspan=1 colspan=1>|0.1650.255</td><td rowspan=1 colspan=1>0.1640.255</td></tr><tr><td rowspan=1 colspan=1>0.2140.286</td><td rowspan=1 colspan=1>0.2190.293</td><td rowspan=1 colspan=1>0.2210.292</td><td rowspan=1 colspan=1>0.2190.290</td><td rowspan=1 colspan=1>0.2220.294</td><td rowspan=1 colspan=1>0.2140.286</td><td rowspan=1 colspan=1>0.2210.293</td><td rowspan=1 colspan=1>0.2240.304</td></tr><tr><td rowspan=1 colspan=1>0.2670.321</td><td rowspan=1 colspan=1>0.2690.326</td><td rowspan=1 colspan=1>0.2780.326</td><td rowspan=1 colspan=1>0.2650.320</td><td rowspan=1 colspan=1>0.2770.337</td><td rowspan=1 colspan=1>0.2680.326</td><td rowspan=1 colspan=1>0.2760.327</td><td rowspan=1 colspan=1>0.2770.337</td></tr><tr><td rowspan=1 colspan=1>720</td><td rowspan=1 colspan=1>0.3440.372</td><td rowspan=1 colspan=1>0.3480.374</td><td rowspan=1 colspan=1>0.3510.379</td><td rowspan=1 colspan=1>0.3590.388</td><td rowspan=1 colspan=1>0.3430.372</td><td rowspan=1 colspan=1>0.366 0.386</td><td rowspan=1 colspan=1>0.3530.384</td><td rowspan=1 colspan=1>0.3620.381</td><td rowspan=1 colspan=1>0.3710.401</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.2430.3040</td><td rowspan=1 colspan=1>.2480.307</td><td rowspan=1 colspan=1>|0.2510.3140</td><td rowspan=1 colspan=1>.2550.315</td><td rowspan=1 colspan=1>0.2480.308 |</td><td rowspan=1 colspan=1>|0.258 0.318</td><td rowspan=1 colspan=1>|0.2490.311</td><td rowspan=1 colspan=1>|0.2560.314 |</td><td rowspan=1 colspan=1>0.2590.324</td></tr><tr><td rowspan=3 colspan=1>Weather</td><td rowspan=2 colspan=1>96192</td><td rowspan=2 colspan=1>0.1370.1810.1810.226</td><td rowspan=1 colspan=1>0.1460.191</td><td rowspan=1 colspan=1>0.1480.195</td><td rowspan=1 colspan=1>|0.1470.199</td><td rowspan=1 colspan=1>0.1430.193</td><td rowspan=1 colspan=1>0.155 0.201</td><td rowspan=1 colspan=1>|0.1640.220</td><td rowspan=1 colspan=1>0.1500.200</td><td rowspan=1 colspan=1>0.1700.230</td></tr><tr><td rowspan=1 colspan=1>0.1880.231</td><td rowspan=1 colspan=1>0.1930.237</td><td rowspan=1 colspan=1>0.1920.240</td><td rowspan=1 colspan=1>0.1880.237</td><td rowspan=1 colspan=1>0.2000.242</td><td rowspan=1 colspan=1>0.2090.258</td><td rowspan=1 colspan=1>0.1910.239</td><td rowspan=1 colspan=1>0.2160.273</td></tr><tr><td rowspan=1 colspan=1>336</td><td rowspan=1 colspan=1>0.2300.267</td><td rowspan=1 colspan=1>0.2340.268</td><td rowspan=1 colspan=1>0.2420.278</td><td rowspan=1 colspan=1>0.2430.281</td><td rowspan=1 colspan=1>0.2360.274</td><td rowspan=1 colspan=1>0.2480.284</td><td rowspan=1 colspan=1>0.2550.292</td><td rowspan=1 colspan=1>0.2420.279</td><td rowspan=1 colspan=1>0.2580.307</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>720</td><td rowspan=1 colspan=1>0.3040.324</td><td rowspan=1 colspan=1>0.3050.319</td><td rowspan=1 colspan=1>0.3090.332</td><td rowspan=1 colspan=1>0.3240.337</td><td rowspan=1 colspan=1>0.3060.325</td><td rowspan=1 colspan=1>0.3180.335</td><td rowspan=1 colspan=1>0.3200.338</td><td rowspan=1 colspan=1>0.3120.330</td><td rowspan=1 colspan=1>0.3230.362</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.213 0.249</td><td rowspan=1 colspan=1>0.2180.252|</td><td rowspan=1 colspan=1>0.2230.261|</td><td rowspan=1 colspan=1>0.2270.264</td><td rowspan=1 colspan=1>0.2180.257 |</td><td rowspan=1 colspan=1>|0.230 0.266</td><td rowspan=1 colspan=1>|0.2370.277|</td><td rowspan=1 colspan=1>0.224 0.262|</td><td rowspan=1 colspan=1>|0.242 0.293</td></tr><tr><td rowspan=3 colspan=1>Electricity</td><td rowspan=3 colspan=1>96192336</td><td rowspan=2 colspan=1>0.123 0.2160.1410.232</td><td rowspan=1 colspan=1>0.1280.219|</td><td rowspan=1 colspan=1>0.1290.221</td><td rowspan=1 colspan=1>0.1270.222</td><td rowspan=1 colspan=1>0.1290.225</td><td rowspan=1 colspan=1>|0.1290.224 |</td><td rowspan=1 colspan=1>0.1280.223</td><td rowspan=1 colspan=1>|0.1300.222</td><td rowspan=1 colspan=1>|0.1400.237</td></tr><tr><td rowspan=1 colspan=1>0.1450.235</td><td rowspan=1 colspan=1>0.1480.238</td><td rowspan=1 colspan=1>0.1470.242</td><td rowspan=1 colspan=1>0.145 0.241</td><td rowspan=1 colspan=1>0.151 0.245</td><td rowspan=1 colspan=1>0.1430.237</td><td rowspan=1 colspan=1>0.148 0.240</td><td rowspan=1 colspan=1>0.1540.251</td></tr><tr><td rowspan=1 colspan=1>0.1550.250</td><td rowspan=1 colspan=1>0.1630.255</td><td rowspan=1 colspan=1>0.1650.257</td><td rowspan=1 colspan=1>0.1650.260</td><td rowspan=1 colspan=1>0.1610.257</td><td rowspan=1 colspan=1>0.1660.261</td><td rowspan=1 colspan=1>0.1590.254</td><td rowspan=1 colspan=1>0.1670.261</td><td rowspan=1 colspan=1>0.1690.268</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>720</td><td rowspan=1 colspan=1>0.1850.279</td><td rowspan=1 colspan=1>0.1930.281</td><td rowspan=1 colspan=1>0.2010.285</td><td rowspan=1 colspan=1>0.1990.292</td><td rowspan=1 colspan=1>0.1970.289</td><td rowspan=1 colspan=1>0.2010.294</td><td rowspan=1 colspan=1>0.1970.287</td><td rowspan=1 colspan=1>0.2020.291</td><td rowspan=1 colspan=1>0.2040.301</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.151 0.244</td><td rowspan=1 colspan=1>0.1570.248</td><td rowspan=1 colspan=1>|0.1610.250|</td><td rowspan=1 colspan=1>0.1600.254|</td><td rowspan=1 colspan=1>0.158 0.253</td><td rowspan=1 colspan=1>|0.162 0.256 |</td><td rowspan=1 colspan=1>0.1570.250</td><td rowspan=1 colspan=1>|0.162 0.254 |</td><td rowspan=1 colspan=1>0.1670.264</td></tr><tr><td rowspan=4 colspan=1>Traffic</td><td rowspan=4 colspan=1>96192336720</td><td rowspan=4 colspan=1>0.3520.2350.3680.2440.3830.2530.4220.277</td><td rowspan=2 colspan=1>0.3600.2380.3830.249</td><td rowspan=2 colspan=1>|0.3610.2380.3730.243</td><td rowspan=1 colspan=1>|0.3670.257</td><td rowspan=1 colspan=1>|0.3740.268</td><td rowspan=1 colspan=1>|0.3620.260 |</td><td rowspan=1 colspan=1>0.3810.266|</td><td rowspan=1 colspan=1>0.3670.251</td><td rowspan=1 colspan=1>|0.3950.275</td></tr><tr><td rowspan=1 colspan=1>0.3820.266</td><td rowspan=1 colspan=1>0.3870.268</td><td rowspan=1 colspan=1>0.3840.269</td><td rowspan=1 colspan=1>0.3940.273</td><td rowspan=1 colspan=1>0.3850.259</td><td rowspan=1 colspan=1>0.4070.280</td></tr><tr><td rowspan=1 colspan=1>0.3950.259</td><td rowspan=1 colspan=1>0.3850.248</td><td rowspan=1 colspan=1>0.3990.275</td><td rowspan=1 colspan=1>0.4010.275</td><td rowspan=1 colspan=1>0.4230.297</td><td rowspan=1 colspan=1>0.4060.279</td><td rowspan=1 colspan=1>0.3980.265</td><td rowspan=1 colspan=1>0.4170.286</td></tr><tr><td rowspan=1 colspan=1>0.4350.278</td><td rowspan=1 colspan=1>0.4280.270</td><td rowspan=1 colspan=1>0.4410.298</td><td rowspan=1 colspan=1>0.4340.275</td><td rowspan=1 colspan=1>0.4470.303</td><td rowspan=1 colspan=1>0.4410.300</td><td rowspan=1 colspan=1>0.4340.287</td><td rowspan=1 colspan=1>0.4540.308</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.3810.2520.393</td><td rowspan=1 colspan=1>0.256</td><td rowspan=1 colspan=1>0.3870.250</td><td rowspan=1 colspan=1>0.3970.274|</td><td rowspan=1 colspan=1>0.399 0.272</td><td rowspan=1 colspan=1>|0.404 0.282 |</td><td rowspan=1 colspan=1>0.406 0.280||</td><td rowspan=1 colspan=1>0.396 0.266|</td><td rowspan=1 colspan=1>|0.418 0.287</td></tr><tr><td rowspan=1 colspan=2>1st Count</td><td rowspan=1 colspan=1>32  23</td><td rowspan=1 colspan=1>3  5</td><td rowspan=1 colspan=1>0   7</td><td rowspan=1 colspan=1>0  0</td><td rowspan=1 colspan=1>1   2</td><td rowspan=1 colspan=1>0   0</td><td rowspan=1 colspan=1>0   0</td><td rowspan=1 colspan=1>0   0</td><td rowspan=1 colspan=1>0   0</td></tr></table>

## E.2 Detailed Ablation Results

Table 11 reports the full ablation results under different prediction horizons. These results complement the averaged ablation table in the main paper and show that the benefits of the proposed Temporal Structure Mixer and PCAHead remain consistent across forecasting horizons.

Table 11: Complete ablation results of TStruct under different prediction horizons. TSM→MHA replaces the Temporal Structure Mixer with multi-head attention, TSM→Linear replaces it with a shared linear temporal mixer, and PCAHead→Linear replaces the proposed prediction head with a linear head.
<table><tr><td colspan="2">Datasets</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Weather</td><td colspan="2">Electricity</td><td colspan="2">Traffic</td></tr><tr><td>Method</td><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">TSM → MHA</td><td>96 192</td><td>0.367</td><td>0.399</td><td>0.285</td><td>0.350</td><td>0.298</td><td>0.340</td><td>0.191</td><td>0.272</td><td>0.143</td><td>0.189</td><td>0.131</td><td>0.225</td><td>0.377</td><td>0.255</td></tr><tr><td></td><td>0.421</td><td>0.440</td><td>0.360</td><td>0.395</td><td>0.344</td><td>0.369</td><td>0.222</td><td>0.293</td><td>0.188</td><td>0.232</td><td>0.148</td><td>0.240</td><td>0.390</td><td>0.262</td></tr><tr><td>336</td><td>0.434</td><td>0.447</td><td>0.360</td><td>0.402</td><td>0.424</td><td>0.426</td><td>0.266</td><td>0.322</td><td>0.260</td><td>0.295</td><td>0.167</td><td>0.258</td><td>0.405</td><td>0.272</td></tr><tr><td>720</td><td>0.478</td><td>0.491</td><td>0.393</td><td>0.433</td><td>0.437</td><td>0.429</td><td>0.356</td><td>0.383</td><td>0.326</td><td>0.344</td><td>0.203</td><td>0.291</td><td>0.439</td><td>0.289</td></tr><tr><td>Avg.</td><td>0.425</td><td>0.444</td><td>0.350</td><td>0.395</td><td>0.376</td><td>0.391</td><td>|0.259</td><td>0.318</td><td>| 0.229</td><td>0.265</td><td>| 0.162</td><td>0.254</td><td>0.403</td><td>0.270</td></tr><tr><td rowspan="5">TSM → Linear</td><td>96</td><td>0.364</td><td>0.393</td><td>0.271</td><td>0.327</td><td>0.278</td><td>0.331</td><td>0.158</td><td>0.243</td><td>0.140</td><td>0.185</td><td>0.125</td><td>0.218</td><td>0.352</td><td>0.235</td></tr><tr><td>192</td><td>0.395</td><td>0.413</td><td>0.338</td><td>0.373</td><td>0.334</td><td>0.362</td><td>0.218</td><td>0.285</td><td>0.187</td><td>0.232</td><td>0.143</td><td>0.234</td><td>0.368</td><td>0.245</td></tr><tr><td>336</td><td>0.418</td><td>0.428</td><td>0.365</td><td>0.404</td><td>0.359</td><td>0.382</td><td>0.266</td><td>0.322</td><td>0.236</td><td>0.273</td><td>0.157</td><td>0.250</td><td>0.383</td><td>0.254</td></tr><tr><td>720</td><td>0.433</td><td>0.461</td><td>0.391</td><td>0.431</td><td>0.418</td><td>0.413</td><td>0.351</td><td>0.381</td><td>0.309</td><td>0.327</td><td>0.187</td><td>0.280</td><td>0.423</td><td>0.277</td></tr><tr><td>Avg.</td><td>0.403</td><td>0.424</td><td>0.341</td><td>0.384</td><td>0.347</td><td>0.372</td><td>0.248</td><td>0.308</td><td>|0.218</td><td>0.254</td><td>0.153</td><td>0.246</td><td>| 0.382</td><td>0.253</td></tr><tr><td rowspan="5">PCAHead → Linear</td><td>96</td><td>0.348</td><td>0.382</td><td>0.268</td><td>0.331</td><td>0.275</td><td>0.325</td><td>0.160</td><td>0.248</td><td>0.137</td><td>0.179</td><td>0.124</td><td>0.217</td><td>0.353</td><td>0.236</td></tr><tr><td>192</td><td>0.385</td><td>0.405</td><td>0.330</td><td>0.374</td><td>0.323</td><td>0.354</td><td>0.212</td><td>0.283</td><td>0.184</td><td>0.227</td><td>0.142</td><td>0.232</td><td>0.369</td><td>0.245</td></tr><tr><td>336</td><td>0.411</td><td>0.423</td><td>0.364</td><td>0.405</td><td>0.357</td><td>0.379</td><td>0.267</td><td>0.322</td><td>0.230</td><td>0.268</td><td>0.156</td><td>0.250</td><td>0.383</td><td>0.254</td></tr><tr><td>720</td><td>0.431</td><td>0.459</td><td>0.393</td><td>0.432</td><td>0.408</td><td>0.412</td><td>0.354</td><td>0.376</td><td>0.310</td><td>0.328</td><td>0.186</td><td>0.279</td><td>0.423</td><td>0.277</td></tr><tr><td>Avg.</td><td>0.394</td><td>0.417</td><td>0.339</td><td>0.386</td><td>0.341</td><td>0.368</td><td>0.248</td><td>0.307</td><td>0.215</td><td>0.251</td><td>0.152</td><td>0.245</td><td>0.382</td><td>0.253</td></tr><tr><td rowspan="5">TStruct</td><td>96</td><td>0.345</td><td>0.380</td><td>0.267</td><td>0.330</td><td>0.275</td><td>0.325</td><td>0.155</td><td>0.241</td><td>0.137</td><td>0.181</td><td>0.123</td><td>0.216</td><td>0.352</td><td>0.235</td></tr><tr><td>192</td><td>0.383</td><td>0.405</td><td>0.322</td><td>0.371</td><td>0.320</td><td>0.354</td><td>0.209</td><td>0.282</td><td>0.181</td><td>0.226</td><td>0.141</td><td></td><td>0.232</td><td>0.368 0.244</td></tr><tr><td>336</td><td>0.407</td><td>0.421</td><td>0.357</td><td>0.400</td><td>0.355</td><td>0.378</td><td>0.265</td><td>0.320</td><td>0.230</td><td>0.267</td><td>0.155</td><td>0.249</td><td>0.383</td><td>0.253</td></tr><tr><td>720</td><td>0.427</td><td>0.454</td><td>0.381</td><td>0.426</td><td>0.400</td><td>0.411</td><td>0.344</td><td>0.372</td><td>0.304</td><td>0.324</td><td>0.185</td><td>0.279</td><td>0.422</td><td>0.277</td></tr><tr><td>Avg.</td><td>0.390</td><td>0.416</td><td>0.331</td><td>0.381</td><td>0.337</td><td>0.367</td><td>0.243</td><td>0.303</td><td>0.213</td><td>0.249</td><td>0.151</td><td>0.244</td><td>0.381</td><td>0.252</td></tr></table>

## F Limitations and Future Directions

TStruct models temporal dependencies through a compact set of shared temporal structure matrices, with controlled initialization and softmax normalized composition and aggregation. This parameterization provides an effective way to learn reusable temporal structures for LTSF. Their adaptation across patch resolutions and transfer across datasets and domains at scale remain directions for further exploration.

Patch resolution dependency. The learned temporal structures are tied to a specific patch resolution. Since each temporal dependency matrix is parameterized over N patch positions, changing the patch size or look-back length can alter the matrix dimensions or the temporal meaning of its coordinates. Although TStruct remains effective under different look-back lengths, directly sharing temporal structures across different patch resolutions remains non-trivial. A promising direction is to develop resolution-adaptive temporal structures that can be transferred, interpolated, or composed across different patch scales.

Temporal structure foundation models. Even after fine-tuning, time series foundation models can struggle to outperform small models trained on individual datasets. One possible explanation is the difficulty of learning input representations that generalize across heterogeneous time series, motivating the exploration of temporal dependency structures as an alternative focus of pretraining. Our analysis provides evidence of reusable temporal relationships across temporal positions and variables, while their transfer and generalization across datasets and domains at scale remain untested. A potential direction is to develop a time series foundation model built around a pretrained dictionary of shared temporal dependency structures, with downstream forecasting tasks selecting and composing these structures to form temporal aggregation operators.