# SPIKELITE: LIGHTWEIGHT SPIKING NEURAL NETWORKS FOR TIME-SERIES FORECASTING

Bang Hu, Changze Lv, Mingjie Li, Xiaoqing Zheng, Wei cao, Fan Zhang School of Computer Science, Fudan University, Shanghai, China

## ABSTRACT

Spiking neural networks (SNNs) offer an energy-efficient paradigm for time-series forecasting through spike-driven computation. However, recent SNN forecasters often pursue higher accuracy through increasingly complex attention mechanisms, or specialized neuronal dynamics, weakening the lightweight motivation of SNNs. We introduce SpikeLite, a spiking forecasting framework built around two modules: a Frequency-Selective Spiking Encoder (FSSE) for frequency-sensitive temporal encoding and a Sparse Spiking Channel Attention (SSCA) module for selective crosschannel interaction. FSSE exploits the low-pass filtering behavior of LIF dynamics to reorganize each input sequence into frequency-sensitive components while collectively preserving the input at the decomposition stage. SSCA then learns a binary mask from encoded channel representations and uses it to selectively exchange information within spike-driven self-attention, retaining informative cross-channel interactions while suppressing redundant ones. When explicit channel interaction is unnecessary, SpikeLite uses the lighter FSSE-only channel-independent path. Experiments under the SeqSNN and SpikF protocols cover four standard multivariate and eight long-term forecasting benchmarks. SpikeLite achieves the best aggregate performance under both protocols, with an average R<sup>2</sup> of 0.790 and RSE of 0.440, and lowest average MSE/MAE of 0.343/0.345 in long-term forecasting. Moreover, evaluation on the ECL dataset shows that SpikeLite achieves the lowest reported energy consumption, further demonstrating its potential for energy-efficient time-series forecasting.

## 1 Introduction

Spiking neural networks (SNNs) are regarded as the third generation of neural networks, emulate biological neurona dynamics and communicate through discrete spike events, offering a biologically inspired and potentially energyefficient alternative to artificial neural networks (ANNs ) [1]. Recent advances have extended SNNs to many tasks, including image classification [2, 3], object detection [4, 5], semantic segmentation [6], speech recognition [7, 8], and autonomous driving and control [9].

Multivariate time-series forecasting is an important predictive task with applications in energy systems, traffic manage ment, weather prediction, and financial analysis [10, 11]. Motivated by the energy-efficiency of SNNs, recent studies have begun to develop SNN-based forecasting models. SeqSNN was the first work to systematically apply SNNs to time-series forecasting, evaluating spiking convolutional, recurrent, and Transformer architectures within a unified forecasting framework [12]. Following this direction, SpikF introduces frequency-domain selection for long-term forecasting [13], TS-LIF develops a dual-compartment spiking neuron to model multi-scale temporal dynamics [14], and SpikeSTAG combines graph learning with spike-based temporal processing for spatio-temporal forecasting [15]. These advances improve the forecasting accuracy of SNNs, but increasingly complex neuronal dynamics, attention modules, and spatial–temporal backbones also add architectural and computational complexity, potentially compromising the energy-efficiency advantage that motivates spike-based modeling. This raises a central question: how can SNN-based forecasters preserve their accuracy gains while maintaining energy-efficient computation?

We revisit this question through two fundamental aspects of multivariate forecasting: input-channel representation and cross-channel modeling. Existing spiking forecasters primarily focus on converting continuous observations into spike sequences, while making limited use of the latent characteristics embedded in the observed series. In particular, low-frequency variations often reflect slowly evolving trends, whereas higher-frequency variations capture rapid local changes and shorter-period patterns [16, 17]. At the same time, modeling relationships among variables requires more than dense interactions among all channels: useful information should be exchanged selectively while redundant channel connections are suppressed.

Based on these considerations, we introduce SpikeLite, a lightweight spiking framework for time-series forecasting that combines frequency-sensitive encoding with selective cross-channel interaction. Its Frequency-Selective Spiking Encoder (FSSE) uses parallel LIF branches with learnable decay factors to organize each input sequence into frequency sensitive temporal components while collectively preserving the original signal. The Sparse Spiking Channel Attention (SSCA) module learns a binary channel-interaction mask to retain informative spike-driven exchanges and suppress redundant connections. We evaluate the model under the SeqSNN and SpikF protocols on four standard and eight longterm forecasting benchmarks, where SpikeLite achieves the strongest overall performance among spiking forecaster while requiring the lowest estimated energy consumption. Our contributions are summarized as follows:

• We introduce SpikeLite, a lightweight spiking forecasting framework that combines frequency-sensitive input representation with selective cross-channel interaction.

• We develop FSSE, which exploits heterogeneous LIF dynamics to construct frequency-sensitive components and aggregate them into compact channel representations, and SSCA, which uses a learned binary mask to suppress redundant spike-driven channel interactions.

• We establish comprehensive evaluations under two spiking forecasting protocols. It achieves the best aggregate accuracy among the compared ANN and SNN methods and the lowest estimated energy consumption in the reported energy comparison.

## 2 Related Work

## 2.1 Input-Channel Representation

Input-channel representation determines what information from the historical window is available for forecasting. Conventional models process historical observations through temporal convolutions, recurrent updates, or direct linear projections [18, 19, 20]. To capture more complex temporal dependencies, other methods adopt elaborate Transformer backbones [16, 17, 21]. Although deeper backbones improve modeling capacity, recent work shows that lightweight representations can remain highly competitive. LTSF-Linear directly maps the lookback window to the forecast horizon, while DLinear improves this design by separately projecting moving-average trends and residual components [20]. Other methods organize the input through progressive decomposition and autocorrelation [16], Fourier or wavelet representations [17], or local temporal patches [21]. However, such representations are designed for conventional continuous-valued networks rather than spiking models.

Spiking forecasters additionally need to encode continuous observations into discrete spike trains for spike-driven processing. Existing methods have successfully developed spiking representations of time-series inputs through temporal encoding and neuronal dynamics[12]. However, their primary focus is on how continuous observations are converted into effective spike sequences, while the intrinsic characteristics already present in the observed signals remain less explored. SpikeLite addresses this limitation through FSSE, which incorporates frequency characteristics into the spiking encoding process and produces frequency-sensitive representations for each input channel.

## 2.2 Cross-Channel Modeling

After individual channel representations are constructed, forecasting models differ in how they exchange information across variables. Channel-independent designs, such as DLinear and PatchTST, process each variable separately, often using a shared predictor structure or shared parameters across channels [20, 21]. This strategy provides a simple and efficient prediction path, but may overlook useful cross-variable dependencies. Channel-dependent designs instead introduce explicit interactions among variables through convolution, graph propagation, or attention. For example, Crossformer uses two-stage attention to capture cross-dimension dependencies [22], while iTransformer treats variables as tokens and applies self-attention to learn their multivariate correlations [23].

Dense interaction, however, assumes that all channel pairs are equally useful and can propagate irrelevant information. Prior studies show that indiscriminate mixing may introduce noise or oversmoothing, motivating mechanisms that select informative relationships [24, 25]. SNN-based forecasters also support cross-channel modeling: SeqSNN applies spiking self-attention to channel-wise embeddings [12], and SpikeSTAG combines adaptive graph learning with spiking temporal processing for spatio-temporal forecasting [15]. These methods demonstrate the value of variable interaction, but their interaction structures are either dense or tied to specialized graph architectures. SpikeLite instead introduces

SSCA, which learns a binary mask over channel pairs and uses it to retain informative spike-driven interactions while suppressing redundant connections.

## 3 Preliminaries

## 3.1 Problem Formulation

We first define the multivariate time-series forecasting task. Given a historical multivariate sequence ${ \bf X } \ =$ $[ \mathbf { x } _ { 1 } , \mathbf { \eta } _ { \cdot } \mathbf { \cdot } \mathbf { \cdot } , \mathbf { x } _ { L } ] ^ { \top } \textbf { \in } \mathbb { R } ^ { L \times C }$ , where L is the lookback length, C is the number of channels, and $\mathbf { x } _ { l } \in \mathbb { R } ^ { C }$ contains all channel observations at position l, multivariate time-series forecasting aims to predict the next $P$ observations, $\mathbf { Y } = [ \mathbf { x } _ { L + 1 } , \dots , \mathbf { x } _ { L + P } ] ^ { \top } \in \mathbb { R } ^ { P \times C }$ . A forecasting model $f _ { \theta }$ with learnable parameters θ maps the historical sequence to the prediction ${ \widehat { \mathbf { Y } } } = f _ { \theta } ( \mathbf { X } )$ . Throughout the paper, l indexes the original observation sequence, whereas $t \in \{ 1 , \ldots , T _ { s } \}$ indexes the internal simulation steps of a spiking layer.

## 3.2 Spiking Neural Computation

Spiking neural networks (SNNs) model the evolution of neuronal membrane potentials and transmit information through discrete spike events. Among the various spiking neuron models, the leaky integrate-and-fire (LIF) neuron is widely adopted because it provides a simple mechanism for capturing temporal state dynamics. Given an input current I<sup>t</sup> at simulation step t, an LIF neuron performs leaky integration, thresholding, and reset according to

$$
\left\{ \begin{array} { l l } { { U ^ { t } = \lambda U ^ { t - 1 } + { \bf { I } } ^ { t } - \vartheta { \bf { S } } ^ { t - 1 } , } } \\ { { \bf { S } } ^ { t } = \Theta \left( U ^ { t } - \vartheta \right) , } \\ { { \bf { G } } ^ { t } = U ^ { t } { \bf { S } } ^ { t } . } \end{array} \right.\tag{1}
$$

Here, λ denotes the membrane decay factor. The last term implements a delayed subtractive reset. During training, the derivative of the Heaviside function is approximated with a surrogate gradient.

Based on these neuronal dynamics, spiking self-attention (SSA) extends pairwise representation interaction to spikedriven networks [3, 12]. Given an input representation X, three learnable projections followed by spiking neurons produce binary query, key, and value representations,

$$
\mathbf { Q } = \operatorname { S N } ( \mathbf { X } \mathbf { W } _ { Q } ) , \quad \mathbf { K } = \operatorname { S N } ( \mathbf { X } \mathbf { W } _ { K } ) , \quad \mathbf { V } = \operatorname { S N } ( \mathbf { X } \mathbf { W } _ { V } ) ,\tag{2}
$$

where $\operatorname { S N } ( { \mathord { \cdot } } )$ denotes the spiking-neuron activation. SSA then aggregates pairwise interactions among these spike representations without relying on the softmax normalization used in conventional self-attention:

$$
\mathrm { S S A } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { S N } \left( \alpha \mathbf { Q } \mathbf { K } ^ { \top } \mathbf { V } \right) ,\tag{3}
$$

where α scales the magnitude of the aggregated interactions to maintain an appropriate activation range for the subsequent spiking neuron.

## 4 Method

## 4.1 Overview

As illustrated in Fig. 1, SpikeLite is a lightweight spiking framework for multivariate time-series forecasting that combines the Frequency-Selective Spiking Encoder (FSSE) with Sparse Spiking Channel Attention (SSCA). Given $\mathbf { X } \in \mathbb { R } ^ { L \times C }$ , FSSE applies parallel LIF dynamics with learnable decay factors to each input channel, and reorganizes their event-gated responses into a compact representation $\mathbf { Z } \in \mathbb { R } ^ { C \times \check { D } }$ , where $\mathbf { Z } _ { c , \phantom { } }$ represents the c-th channel. FSSE then repeats $\mathbf { Z }$ over $T _ { s }$ simulation steps and converts it into binary spike states $\mathbf { S } _ { \mathrm { i n } } ^ { ( t ) } \in \{ 0 , 1 \} ^ { C \times D }$ . SSCA uses a learned binary interaction mask to selectively exchange information across channels while suppressing redundant connections; when omitted, the model follows the lighter FSSE-only path. A lightweight prediction head maps the resulting representation to $\widehat { \mathbf { Y } } = g _ { \psi } ( \widetilde { \mathbf { Z } } ) \in \mathbb { R } ^ { P \times C }$

## 4.2 Frequency-Selective Spiking Encoder

Frequency-selective LIF dynamics. As defined in Section 3, the LIF neuron integrates the current input with its decayed membrane state. To analyze this dynamics in the frequency domain, we temporarily remove thresholding and

![](images/a63ae4b5d1ef582aec9c043c2316370f83c2a81083d3df7cd5d1d42f8e61a34a.jpg)  
Figure 1: Overview of SpikeLite. FSSE constructs frequency-sensitive representations for individual input channels, while SSCA selectively exchanges information across channels. The lower panel details mask generation and masked spiking attention: the mask is estimated from the encoded channel representations and applied to the attention scores before value aggregation.

reset, yielding the linearized equation $U _ { l } = \tau U _ { l - 1 } + X _ { l }$ . Its z-transform gives $H ( z ) = U ( z ) / X ( z ) = 1 / ( 1 - \tau z ^ { - 1 } )$ Evaluating this transfer function on the unit circle, $z = e ^ { \mathrm { i } \omega }$ , produces

$$
H _ { \tau } ( e ^ { \mathrm { i } \omega } ) = \frac { 1 } { 1 - \tau e ^ { - \mathrm { i } \omega } } , \qquad \left| H _ { \tau } ( e ^ { \mathrm { i } \omega } ) \right| = \frac { 1 } { \sqrt { 1 + \tau ^ { 2 } - 2 \tau \cos \omega } } ,\tag{4}
$$

where ω denotes the angular frequency. Thus, leaky integration behaves as a first-order low-pass filter: a larger τ more strongly favors low-frequency components, whereas a smaller τ produces a flatter response and retains a broader frequency range. Different decay factors therefore induce distinct frequency sensitivities to the same input, as illustrated in Fig. 2.

The complete LIF neuron restores thresholding and reset, making the response event-dependent. FSSE instantiates K parallel branches for each input variable, with branch-specific learnable decay factors. For branch k and variable c, the dynamics along the original observation axis are

$$
U _ { l , c } ^ { ( k ) } = \tau _ { c } ^ { ( k ) } U _ { l - 1 , c } ^ { ( k ) } + X _ { l , c } - \vartheta _ { c } ^ { ( k ) } S _ { l - 1 , c } ^ { ( k ) } ,\tag{5}
$$

$$
S _ { l , c } ^ { ( k ) } = \Theta \Big ( U _ { l , c } ^ { ( k ) } - \vartheta _ { c } ^ { ( k ) } \Big ) ,\tag{6}
$$

where $U _ { l , c } ^ { ( k ) }$ is the membrane state, $S _ { l , c } ^ { ( k ) } \in \{ 0 , 1 \}$ is the emitted event, and $\vartheta _ { c } ^ { ( k ) }$ is the firing threshold. Both $U _ { 0 , c } ^ { ( k ) }$ and $S _ { 0 , c } ^ { ( k ) }$ are initialized to zero. The last term in Eq.(5) implements the delayed subtractive reset.

FSSE gates each membrane response with its emitted spike: $G _ { l , c } ^ { ( k ) } = U _ { l , c } ^ { ( k ) } S _ { l , c } ^ { ( k ) }$ . Thus, $G _ { l , c } ^ { ( k ) }$ is zero when branch k is inactive and retains the corresponding membrane state when the branch fires, combining the frequency-selective neuronal dynamics with event-based transmission.

Each decay factor is parameterized as $\tau _ { c } ^ { ( k ) } = \sigma ( \rho _ { c } ^ { ( k ) } )$ , where $\rho _ { c } ^ { ( k ) }$ is trainable and σ denotes the logistic sigmoid. We initialize the branches with descending decay factors to establish a nominal slow-to-fast ordering. During training, the decay factors are freely optimized without any explicit constraint on their relative ordering. Despite the absence of an explicit ordering constraint, the learned decay factors remain broadly consistent with the initial slow-to-fast ordering.

Frequency-sensitive encoding. To obtain informative frequency-sensitive components without discarding the original signal, FSSE applies successive differences to the event-gated responses of the parallel LIF branches. Let $\mathbf { G } ^ { ( k ) } \in \mathbb { R } ^ { L \times C }$ collect the responses of branch k over the historical window. FSSE constructs $K + 1$ components as

![](images/da09eb0c5611210dfb4bd18192ce4c35ca40e5e687c0c5682cdd4fd96cc6a663.jpg)  
(b) Frequency-Selective Spiking Encoder (FSSE)

$$
\mathbf { B } ^ { ( j ) } = \left\{ \begin{array} { l l } { \mathbf { G } ^ { ( 1 ) } , } & { j = 0 , } \\ { \mathbf { G } ^ { ( j + 1 ) } - \mathbf { G } ^ { ( j ) } , } & { 1 \leq j < K , } \\ { \mathbf { X } - \mathbf { G } ^ { ( K ) } , } & { j = K . } \end{array} \right.\tag{7}
$$

The first component retains the response of the first LIF branch, the intermediate components describe the response differences between adjacent branches, and the final component preserves the residual between the input and the last branch response. Because neighboring branch responses are subtracted successively, the components form a telescoping decomposition:

$$
\sum _ { j = 0 } ^ { K } \mathbf { B } ^ { ( j ) } = \mathbf { G } ^ { ( 1 ) } + \sum _ { j = 1 } ^ { K - 1 } \left( \mathbf { G } ^ { ( j + 1 ) } - \mathbf { G } ^ { ( j ) } \right) + \left( \mathbf { X } - \mathbf { G } ^ { ( K ) } \right) = \mathbf { X } .\tag{8}
$$

$\{ \mathbf { B } ^ { ( j ) } \} _ { j = 0 } ^ { K }$ while preserving the input collectively. Each component is independently projected along the observation axis and aggregated into a compact channel representation:

$$
\begin{array} { r l } & { \mathbf { z } _ { c } ^ { ( j ) } = \mathbf { W } ^ { ( j ) } \mathbf { B } _ { : , c } ^ { ( j ) } + \mathbf { b } ^ { ( j ) } \in \mathbb { R } ^ { D } , } \\ & { \mathbf { Z } _ { c , : } = \displaystyle \sum _ { j = 0 } ^ { K } \alpha _ { j } \mathbf { z } _ { c } ^ { ( j ) } , \qquad \mathbf { Z } \in \mathbb { R } ^ { C \times D } , } \end{array}\tag{9}
$$

where $\mathbf { W } ^ { ( j ) } \in \mathbb { R } ^ { D \times L }$ and $\alpha _ { j }$ is a learnable scale for component j. The resulting Z is passed to the spike-driven interaction stage described in Section 4.3.

## 4.3 Sparse Spiking Channel Attention

FSSE produces frequency-sensitive representations for each input variable while collectively preserving the original signal. Because forecasting can also depend on relationships among variables, SSCA selectively exchanges information across these representations. A standard spiking self-attention block considers all channel pairs, whereas SSCA learns a sample-dependent binary mask $\mathbf { M } \in \{ 0 , \dot { 1 } \} ^ { C \times \mathbf { \zeta } }$ to retain informative connections during spike-driven aggregation and suppress redundant interactions.

Binary mask generation. Before spike replication, SSCA applies a real-valued Fourier transform to each row of Z along the representation dimension:

$$
\mathbf { F } _ { c } = | \mathrm { r F F T } \left( \mathbf { Z } _ { c , : } \right) | \in \mathbb { R } ^ { F } , \qquad F = \left\lfloor \frac { D } { 2 } \right\rfloor + 1 .\tag{10}
$$

These descriptors are used only to estimate the binary channel-interaction mask. The subsequent attention operation is performed on the spike-form channel states obtained after replicating Z over the $T _ { s }$ simulation steps, rather than on the spectral descriptors.

To measure the similarity between channel descriptors, SSCA maps each spectral descriptor into a low-dimensional relation space using a trainable projection matrix:

$$
\mathbf { r } _ { c } = \mathbf { W } _ { r } \mathbf { F } _ { c } \in \mathbb { R } ^ { r } , \qquad \mathbf { W } _ { r } \in \mathbb { R } ^ { r \times F } .\tag{11}
$$

For a pair of channels $( i , j )$ , we compute the squared relation distance and its inverse affinity:

$$
\begin{array} { r } { d _ { i j } = \big \| { \bf r } _ { i } - { \bf r } _ { j } \big \| _ { 2 } ^ { 2 } + \epsilon , \qquad a _ { i j } = d _ { i j } ^ { - 1 } . } \end{array}\tag{12}
$$

The affinity is normalized independently for each source channel:

$$
p _ { i j } = \mathrm { c l i p } \left( \gamma \frac { a _ { i j } } { \operatorname* { m a x } _ { q \neq i } a _ { i q } } , 0 , 1 \right) , \qquad i \neq j .\tag{13}
$$

Channels with similar spectral descriptors therefore obtain larger affinity values and are more likely to remain connected. We obtain the binary interaction mask by thresholding the normalized affinity while retaining all self-connections:

$$
\overline { { { m } } } _ { i j } = \left\{ { \begin{array} { l l } { 1 , } & { i = j , } \\ { \mathbb { I } ( p _ { i j } > \eta ) , } & { i \neq j . } \end{array} } \right.\tag{14}
$$

To optimize this discrete mask, we use a straight-through estimator:

$$
m _ { i j } = \left\{ \begin{array} { l l } { 1 , } & { i = j , } \\ { \sec { ( m _ { i j } - p _ { i j } ) } + p _ { i j } , } & { i \neq j , } \end{array} \right. \quad \mathbf { M } = [ m _ { i j } ] ,\tag{15}
$$

where $\eta$ is the sparsification threshold and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. The forward value of M is binary, while its backward gradient is propagated through the continuous affinity $p _ { i j }$

Masked spike-driven attention. The spike-form variable states produced by FSSE are used as the input tokens of sparse SSA. For attention head m and simulation step t, the corresponding queries, keys, and values are denoted by $\bar { \mathbf { Q } } ^ { ( t , m ) } , \mathbf { K } ^ { ( t , m ) }$ , and $\mathbf { V } ^ { ( t , m ) }$ . SSCA applies the learned mask directly to the pairwise interaction matrix:

$$
\mathbf { A } _ { \mathrm { S S C A } } ^ { ( t , m ) } = \kappa \left( \mathbf { Q } ^ { ( t , m ) } \mathbf { K } ^ { ( t , m ) \top } \right) \odot \mathbf { M } , \qquad \mathbf { O } ^ { ( t , m ) } = \mathbf { A } _ { \mathrm { S S C A } } ^ { ( t , m ) } \mathbf { V } ^ { ( t , m ) } .\tag{16}
$$

where κ is the attention scaling factor. Consistent with spike-driven SSA, no softmax normalization is applied. A zero entry $M _ { i j } = 0$ removes the contribution from variable j to variable i, whereas $M _ { i j } = 1$ preserves the corresponding interaction.

The masked outputs across all attention heads and simulation steps are aggregated through the SSA output projection, yielding the channel-interaction-enhanced representation $\widetilde { \mathbf { Z } } .$ This representation is subsequently mapped by the prediction head to the final forecast $\hat { \mathbf { Y } } \in \mathbb { R } ^ { P \times \bar { C } }$

## 5 Experiments

## 5.1 Experimental Settings

We evaluate SpikeLite under two representative forecasting settings: the standard multivariate forecasting protocol introduced by SeqSNN [12] and the long-term forecasting protocol adopted by SpikF [13]. Across the two settings, we compare SpikeLite with a broad collection of ANN- and SNN-based forecasting models on datasets covering traffic, electricity, weather, and other application domains.

Protocols and metrics. Under the SeqSNN protocol [12], we evaluate standard multivariate forecasting with prediction horizons of 6, 24, 48, and 96, using $R ^ { 2 }$ and elative squared error (RSE) as evaluation metrics. Under the SpikF protocol [13], we evaluate long-term forecasting with a lookback length of 96 and prediction horizons of 96, 192, 336, and 720, using mean squared error (MSE) and mean absolute error (MAE). Main-text results are averaged over the four prediction horizons, while the complete horizon-wise results are provided in Appendix B. Dataset statistics and complete experimental configurations are summarized in Appendix A.

Compared methods. For the SeqSNN protocol, we compare against statistical predictors, including ARIMA and Gaussian Process regression, and ANN forecasters, including Autoformer, PatchTST, and iTransformer. ARIMA, Gaussian Process, Autoformer, iTransformer, and iSpikformer results are taken from SeqSNN [12]. TS-TCN, TS-GRU, and TS-former results are taken from TS-LIF [14]. PatchTST, AGCRN, and STAEformer are implemented and evaluated under the SeqSNN data splits and evaluation procedure, using their original architectures [21, 26, 27].

For the SpikF protocol (long-term forecasting), we compare SpikeLite with the ANN baselines iTransformer, RLinear, PatchTST, Crossformer, TimesNet, DLinear, SCINet, and Autoformer, together with SpikF. Their reported results are taken from the SpikF benchmark [13], while SpikeLite is evaluated under the same protocol.

## 5.2 Main Results

SpikeLite achieves the best overall forecasting performance under both the standard multivariate and long-term forecasting settings.

Table 1: Experimental results on standard multivariate forecasting under the SeqSNN protocol [12]. Bold and underlined values denote the best and second-best results, respectively. ↑/↓ indicates that higher/lower values are better.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">METR-LA</td><td colspan="4">PEMS-BAY</td><td colspan="4">Solar</td><td colspan="4">Electricity</td><td colspan="2">Summary</td></tr><tr><td>6</td><td>24</td><td>48</td><td>96</td><td>6</td><td>24</td><td>48</td><td>96</td><td>6</td><td>24</td><td>48</td><td>96</td><td>6</td><td>24</td><td>48</td><td>96</td><td></td><td>Avg. Rank</td></tr><tr><td rowspan="2">ARIMA</td><td>R²↑.687 .441 .282</td><td></td><td></td><td></td><td></td><td></td><td>2.265 .741 .723</td><td>.692</td><td></td><td>.670</td><td>.951.847.725</td><td></td><td></td><td>.689</td><td>.963</td><td>3.960</td><td>.914.863</td><td>.713</td><td>9.69</td></tr><tr><td>RSE↓.575 .742.889</td><td></td><td></td><td></td><td></td><td></td><td>.902 .532 .548</td><td>.562</td><td></td><td>.612</td><td>.202.365</td><td></td><td>.588.589</td><td></td><td></td><td>.522.534.564 .599</td><td></td><td>.583</td><td>9.44</td></tr><tr><td rowspan="2">GP</td><td>R² ↑ .685 .437 .265 .233 .732 .712 .689</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.665.944.836.711.675</td><td></td><td></td><td></td><td>5.962 .968 .912 .852 .705</td><td></td><td></td><td></td><td>11.03</td></tr><tr><td>RSE↓.572 .738.912 .925.544.532</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.577</td><td></td><td></td><td></td><td>7.592 .225.388.612 .575</td><td>.603</td><td>3.612</td><td>.633</td><td>.642</td><td>.605</td><td>10.28</td></tr><tr><td rowspan="2">Autoformer</td><td>R² ↑.762 .548 .411 .282 .782 .711 .689</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>9.668 .960.852 .791 .701 .980.977 .975 .963 .753</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.00</td></tr><tr><td>RSE↓.569.692 .785.872.452.543</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.577</td><td>.565</td><td></td><td>5.212 .432.622 .685</td><td></td><td></td><td>.481.506</td><td>.566</td><td>.548</td><td>.569</td><td>8.91</td></tr><tr><td rowspan="2">PatchTST</td><td>R² ↑ .819 .618 .434 .284 .873 .718 .686 .612</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2 .962 .868 .795 .729</td><td></td><td></td><td></td><td>.980.977</td><td>7.975</td><td>.963</td><td></td><td>6.50</td></tr><tr><td></td><td></td><td>RSE↓ .456 .666 .792 .875 .385 .532</td><td></td><td></td><td></td><td></td><td></td><td>2.569</td><td>.577</td><td>.201</td><td>.369</td><td>.467.588</td><td></td><td>.264.325</td><td>.326</td><td></td><td>.768 .441.490</td><td>6.41</td></tr><tr><td rowspan="2">AGCRN</td><td>R² ↑ .764 .562 .442 .287.787 .709</td><td></td><td></td><td></td><td></td><td></td><td></td><td>.694</td><td></td><td>.659</td><td>9.958 .854 .788 .704 .987 .982</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>RSE↓.511.696.785</td><td></td><td></td><td></td><td>.874.543</td><td>.543</td><td>.566</td><td>.574</td><td></td><td></td><td>4.207.432 .625 .656</td><td></td><td>.223</td><td>.265</td><td>.978 .296</td><td>.966.758 .531.520</td><td></td><td>6.66 7.56</td></tr><tr><td rowspan="2">STAEformer R2 ↑ .774 .573 .396</td><td></td><td></td><td></td><td></td><td></td><td>.279</td><td>9.803.716</td><td></td><td>6.691.647</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>RSE↓.499.686.817</td><td></td><td></td><td></td><td>.881</td><td></td><td></td><td>1.397.534.565</td><td></td><td>.580</td><td>.210</td><td></td><td>7.961.857 .790.713 .431.621.601</td><td></td><td>.977 .973 .271 .303</td><td></td><td>.970.959</td><td>.755</td><td></td><td>8.38 7.81</td></tr><tr><td colspan="2">iTransformer R2 ↑ .829 .623 .439</td><td></td><td></td><td></td><td></td><td>.285</td><td></td><td>.887.719</td><td></td><td>.685</td><td>.668</td><td>3.964.879</td><td></td><td></td><td>.799.738</td><td>.979</td><td>.977</td><td>.326 7.975</td><td></td><td>.481.513</td><td></td></tr><tr><td colspan="2"></td><td></td><td>RSE↓.436.648.780</td><td></td><td></td><td>.878</td><td>.362</td><td>.547</td><td></td><td>.561</td><td></td><td>.584.191</td><td>.348</td><td></td><td>.448.563</td><td>.259</td><td>.305</td><td>.335</td><td>.427</td><td>.964.776 .479</td><td>4.84 5.09</td></tr><tr><td colspan="2">TS-TCN</td><td></td><td></td><td></td><td></td><td>.328</td><td>.897</td><td>.759</td><td></td><td>.698</td><td></td><td></td><td>.652 .964 .884 .762 .720</td><td></td><td></td><td>.980.971 .968</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">RSE↓.459.656.757</td><td>R²↑.810 .605 .473</td><td></td><td></td><td></td><td>.857</td><td>.354</td><td>.527</td><td>.559</td><td></td><td>.633 .189</td><td></td><td>.325</td><td>.484.523</td><td></td><td>.264 .316 .318</td><td></td><td></td><td>.962 .360</td><td>.777 .474</td><td>5.41 4.22</td></tr><tr><td colspan="2">TS-GRU</td><td></td><td></td><td></td><td></td><td></td><td></td><td>9.874.742</td><td></td><td>2.684.649</td><td></td><td>.938.878 .779 .722</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">RSE↓ .412 .651 .795.853.384.530.587.637.253 .349</td><td>R2 ↑ .848 .618 .430 .329</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.426.527</td><td></td><td>.991 .981 .216 6.240</td><td>.236</td><td>.983</td><td>.976.776 .271.460</td><td></td><td>5.47 5.19</td></tr><tr><td colspan="2">iSpikformer R2 ↑ .817 .618 .440 .279 .879 .744 .687 .674 .961 .876 .795 .738</td><td></td><td>RSE↓ .475 .668 .752 .905 .376.536.569</td><td></td><td></td><td></td><td></td><td></td></table>

Under the SeqSNN protocol, SpikeLite achieves the best overall performance, with an average R<sup>2</sup> of 0.790 and an average RSE of 0.440. Across the 16 dataset–horizon settings for each metric, SpikeLite obtains the largest number of first- and second-place results.

Among the four benchmarks, the clearest improvement appears on Electricity, whose pronounced periodic structure is well aligned with FSSE’s reorganization of the input into multi-scale frequency-sensitive components. SpikeLite also performs strongly on METR-LA and PEMS-BAY, where correlated sensor measurements require both temporal modeling and information exchange across variables. Among the compared methods, TS-former ranks second and shows a slight advantage over SpikeLite on Solar, possibly due to its stronger emphasis on local temporal dynamics.

Under the SpikF protocol, SpikeLite achieves the best average MSE and MAE, demonstrating that its performance advantage also holds under the long-term forecasting setting.

SpikeLite extends a consistent advantage on Electricity to ECL, substantially outperforming all compared methods in long-term forecasting. Across the ETT and Traffic benchmarks, SpikeLite remains competitive with the strongest ANN and SNN baselines. The ETT datasets exhibit multi-scale temperature and load variations, whereas Traffic contains recurring patterns among spatially correlated sensor measurements; FSSE and SSCA jointly model these temporal and cross-channel characteristics. Exchange is more challenging due to its less regular long-horizon variations, yet SpikeLite remains close to the best-performing methods.

Table 2: Long-term forecasting performance under the SpikF protocol [13]. Results are averaged across four prediction horizons (96, 192, 336, and 720) and three random seeds. Lower MSE and MAE indicate better performance. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">iTransformer</td><td colspan="2">RLinear</td><td colspan="2">PatchTST</td><td colspan="2">Crossformer</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">SCINet</td><td colspan="2">Autoformer</td><td colspan="2">SpikF</td><td colspan="2">SpikeLite</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>EMAE</td><td>MSE MAE</td><td></td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td></td><td>MSE EMAE</td><td></td><td>MSE MAE</td><td></td><td>MSE</td><td>MAE</td><td></td><td>MSE MAE</td><td>MSE MAE</td><td></td></tr><tr><td>ECL</td><td>.178</td><td>.270</td><td>.219</td><td>.298</td><td>.205</td><td>.290</td><td>.244</td><td>.334</td><td>.192</td><td>.295</td><td>.212</td><td>.300</td><td>.268</td><td>.365</td><td>.227</td><td>.338</td><td>.183</td><td>.275</td><td>.175</td><td>.263</td></tr><tr><td>Weather</td><td>.258</td><td>.278</td><td>.272</td><td>.291</td><td>.259</td><td>.281</td><td>.259</td><td>.315</td><td>.259</td><td>.287</td><td>.265</td><td>.317</td><td>.292</td><td>.363</td><td>.338</td><td>.382</td><td>.245</td><td>.265</td><td>.245</td><td>.265</td></tr><tr><td>ETTh1</td><td>.454</td><td>.447</td><td>.446</td><td>.434</td><td>.469</td><td>.454</td><td>.529</td><td>.522</td><td>.458</td><td>.450</td><td>.456</td><td>.452</td><td>.747</td><td>.647</td><td>.496</td><td>.487</td><td>.440</td><td>.428</td><td>.443</td><td>.429</td></tr><tr><td>ETTh2</td><td>.383</td><td>.407</td><td>.374</td><td>.398</td><td>.387</td><td>.407</td><td>.942</td><td>.684</td><td>.414</td><td>.427</td><td>.559</td><td>.515</td><td>.954</td><td>.723</td><td>.450</td><td>.459</td><td>.372</td><td>.394</td><td>.374</td><td>.394</td></tr><tr><td>ETTm1</td><td>.407</td><td>.410</td><td>.414</td><td>.407</td><td>.387</td><td>.400</td><td>.513</td><td>.496</td><td>.400</td><td>.406</td><td>.403</td><td>.407</td><td>.485</td><td>.481</td><td>.588</td><td>.517</td><td>.388</td><td>.385</td><td>.387</td><td>.389</td></tr><tr><td>ETTm2</td><td>.288</td><td>.332</td><td>.286</td><td>.327</td><td>.281</td><td>.326</td><td>.757</td><td>.610</td><td>.291</td><td>.333</td><td>.350</td><td>.401</td><td>.571</td><td>.537</td><td>.327</td><td>.371</td><td>.281</td><td>.320</td><td>.278</td><td>.319</td></tr><tr><td>Traffic</td><td>.428</td><td>.282</td><td>.626</td><td>.378</td><td>.481</td><td>.304</td><td>.550</td><td>.304</td><td>.620</td><td>.336</td><td>.625</td><td>.383</td><td>.804</td><td>.509</td><td>.628</td><td>.379</td><td>.497</td><td>.296</td><td>.479</td><td>.294</td></tr><tr><td>Exchange</td><td>.360</td><td>.403</td><td>.354</td><td>.414</td><td>.367</td><td>.404</td><td>.940</td><td>.707</td><td>.416</td><td>.443</td><td>.354</td><td>.414</td><td>.750</td><td>.626</td><td>.613</td><td>.539</td><td>.360</td><td>.402</td><td>.366</td><td>.404</td></tr><tr><td>Avg.</td><td>.345</td><td>.354</td><td>.374</td><td>.368</td><td>.355</td><td>.358</td><td>.592</td><td>.497</td><td>.381</td><td>.372</td><td>.403</td><td>.399</td><td>.609</td><td>.531</td><td>.458</td><td>.434</td><td>.346</td><td>.346</td><td>.343</td><td>.345</td></tr><tr><td>Avg. Rank</td><td>3.56</td><td>3.56</td><td>5.12</td><td>5.00</td><td>4.38</td><td>4.19</td><td>8.25</td><td>8.31</td><td>5.50</td><td>5.50</td><td>5.94</td><td>7.12</td><td>9.38</td><td>9.38</td><td>8.38</td><td>8.38</td><td>2.44</td><td>1.75</td><td>2.06</td><td>1.81</td></tr></table>

## 5.3 Frequency-Selective Mechanism Analysis

We further examine whether the parallel LIF branches and the resulting FSSE components exhibit distinct responses to temporal frequencies. Following the frequency-response analysis in Section 4.2, we apply sinusoidal inputs to the PEMS setting and sweep the frequency from 0.1 to 0.5 cycles per sample. The higher-frequency portion of this sweep provides a clearer view of the differences among the learned responses, as shown in Fig. 3.

In Fig. 3(a), the FSSE components exhibit distinct gain profiles: earlier components respond more strongly at lower frequencies, whereas later components and the residual remain responsive over a broader frequency range. Fig. 3(b) shows a similar overall trend in firing probability. The firing-probability profiles broadly follow the nominal slow-to-fast initialization order, while allowing channel-dependent deviations after training. These results are consistent with the heterogeneous decay factors learned by FSSE and demonstrate that heterogeneous LIF dynamics and event gating produce distinct frequency-sensitive responses.

![](images/76a4238e25e08c60285d3e5d9a6120115dabdb0a7532cc714a782a3fcb75715b.jpg)

![](images/6557bb0a83c65f7228fb18a986b0831faa6298ef6784d0c602b5c3eaa45ef0bc.jpg)  
Figure 3: Frequency-sweep analysis of FSSE: component-wise fundamental gain and branch-wise firing probability.

## 5.4 Ablation Study

To assess the contributions of FSSE and SSCA, we perform an ablation study under the SpikF protocol. The main-text analysis focuses on Traffic, where dependencies among observed variables make cross-channel modeling particularly relevant. The full model is compared against w/o FSSE and w/o SSCA with all other components and training settings unchanged. Figure 4 reports the Traffic results, with the corresponding Weather results provided in Appendix D.

Removing either component degrades performance across all four Traffic horizons. Without FSSE, the average MSE and MAE increase from 0.479 and 0.294 to 0.489 and 0.299, indicating that frequency-sensitive encoding improves the representation of each variable’s temporal variations. Removing SSCA produces a larger increase, to 0.602 MSE and 0.338

![](images/519b5aeaf48beed0a3008abf81753c6760b732fa54832a5538f86eff8d7283be.jpg)  
Figure 4: Ablation results on Traffic under the SpikF protocol. Lower values indicate better forecasting performance.

MAE, showing that selective cross-channel interaction contributes substantial predictive information on Traffic. Together, these results support the complementary roles of FSSE in temporal encoding and SSCA in modeling dependencies among variables.

## 5.5 Energy Analysis

Following the established energy-estimation protocol based on 45-nm CMOS technology [28], where a multiply– accumulate operation is assigned an energy cost of 4.6 pJ and an accumulate operation is assigned 0.9 pJ, we estimate the inference cost of the encoder backbone. The task-specific prediction heads are excluded for all methods to ensure a consistent comparison.

Table 3 shows that SpikeLite uses only 67.2K parameters and 0.02G operations, corresponding to an estimated energy consumption of 95.30 µJ per sample, which is the lowest among the compared models. Compared with SpikF, SpikeLite reduces energy consumption by approximately 19.0%, while the savings over the ANN-based baselines are approximately 53.6% relative to DLinear and 97.1% relative to iTransformer.

These results indicate that the energy advantage of SpikeLite is attributed to its spike-driven encoder and compact channel interaction pathway, which reduce the number of costly operations required during inference while maintaining forecasting accuracy.

Table 3: Model complexity, estimated energy, and forecasting accuracy on ECL at a prediction horizon of 720.
<table><tr><td>Model</td><td>Params</td><td>OPs</td><td>Energy (µJ)</td><td>MSE@720</td></tr><tr><td>SpikF</td><td>1.2K</td><td>0.13G</td><td>117.66</td><td>.219</td></tr><tr><td>DLinear</td><td>0.14M</td><td>45M</td><td>205.19</td><td>.245</td></tr><tr><td>iTransformer</td><td>1.6M</td><td>0.72G</td><td>3289.29</td><td>.225</td></tr><tr><td>SpikeLite</td><td>67.2K</td><td>0.02G</td><td>95.30</td><td>.218</td></tr></table>

## 6 Conclusion

We presented SpikeLite, a lightweight spiking framework for multivariate time-series forecasting. FSSE uses heteroge neous LIF dynamics to reorganize each input sequence into frequency-sensitive components, while SSCA selectively models informative cross-channel interactions through a sparse binary mask. Experiments under the SeqSNN and SpikF protocols show that SpikeLite achieves consistently strong forecasting accuracy across standard and long-term benchmarks, while requiring the lowest estimated energy among the compared models. The limitations and future directions are discussed in Appendix H.

## 7 Reproducibility Statement

The supplementary material documents the dataset characteristics, temporal properties, preprocessing and evaluation protocols, and implementation and training configurations (Appendix A). It also provides the complete horizon-wise results, seed-stability analysis, ablations, frequency diagnostics, and energy-accounting details. Results produced by our implementation are averaged over three independent random seeds unless stated otherwise. All corresponding mean±std results are reported in Appendix C. Numerical results copied from published benchmark reports are identified in the corresponding comparison description and are retained under their original evaluation protocol; they are not presented as new three-seed measurements. The source code, configuration files, and analysis scripts will be released upon publication.

## 8 Ethics Statement

Our experiments use publicly available multivariate time-series benchmarks covering traffic sensors, electricity and solar measurements, weather observations, transformer measurements, and exchange rates. We collect no new human or animal data, and our experiments do not use individual-level identifiers or attempt to infer personal attributes. The datasets are used for research in accordance with their original licenses and documentation. Although aggregate traffic, energy, and exchange rates can support sensitive operational or mobility-related inferences in downstream systems, such deployments should follow the applicable data-use restrictions, privacy requirements, and access-control practices.

## References

[1] Wolfgang Maass. Networks of spiking neurons: The third generation of neural network models. Neural Networks, 10(9):1659–1671, 1997.

[2] Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothée Masquelier, and Yonghong Tian. Deep residual learning in spiking neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 21056–21069, 2021.

[3] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng Yan, Yonghong Tian, and Li Yuan. Spikformer: When spiking neural network meets transformer. In International Conference on Learning Representations, 2023.

[4] Seijoon Kim, Seongsik Park, Byunggook Na, and Sungroh Yoon. Spiking-yolo: Spiking neural network for energy-efficient object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11270–11277, 2020.

[5] Qiaoyi Su, Yuhong Chou, Yifan Hu, Jianing Li, Shijie Mei, Ziyang Zhang, and Guoqi Li. Deep directly-trained spiking neural networks for object detection. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6555–6565. IEEE, 2023.

[6] Youngeun Kim, Joshua Chough, and Priyadarshini Panda. Beyond classification: Directly training spiking neural networks for semantic segmentation. Neuromorphic Computing and Engineering, 2(4):044015, 2022.

[7] Jibin Wu, Emre Yılmaz, Malu Zhang, Haizhou Li, and Kay Chen Tan. Deep spiking neural networks for large vocabulary automatic speech recognition. Frontiers in Neuroscience, 14:199, 2020.

[8] Thomas Pellegrini, Romain Zimmer, and Timothée Masquelier. Low-activity supervised convolutional spiking neural networks applied to speech commands recognition. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 97–103. IEEE, 2021.

[9] Rui-Jie Zhu, Ziqing Wang, Leilani Gilpin, and Jason K. Eshraghian. Autonomous driving with spiking neural networks. In Advances in Neural Information Processing Systems, volume 37, 2024.

[10] Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang. DUET: Dual clustering enhanced multivariate time series forecasting. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1185–1196, 2025.

[11] Lu Han, Xu-Yang Chen, Han-Jia Ye, and De-Chuan Zhan. SOFTS: Efficient multivariate time series forecasting with series-core fusion. In Advances in Neural Information Processing Systems, volume 37, 2024.

[12] Changze Lv, Yansen Wang, Dongqi Han, Xiaoqing Zheng, Xuanjing Huang, and Dongsheng Li. Efficient and effective time-series forecasting with spiking neural networks. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 33624–33637, 2024.

[13] Wenjie Wu, Dexuan Huo, and Hong Chen. SpikF: Spiking fourier network for efficient long-term prediction. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 67342–67368. PMLR, 2025.

[14] Shibo Feng, Wanjin Feng, Xingyu Gao, Peilin Zhao, and Zhiqi Shen. TS-LIF: A temporal segment spiking neuron network for time series forecasting. In International Conference on Learning Representations, 2025.

[15] Bang Hu, Changze Lv, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Wei Cao, and Fan Zhang. SpikeSTAG: Spatial-temporal forecasting via GNN-SNN collaboration. arXiv preprint arXiv:2508.02069, 2025.

[16] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430, 2021.

[17] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 27268–27286. PMLR, 2022.

[18] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 95–104. ACM, 2018.

[19] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

[20] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 11121–11128, 2023.

[21] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

[22] Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In International Conference on Learning Representations, 2023.

[23] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In International Conference on Learning Representations, 2024.

[24] Jialin Chen, Jan Eric Lenssen, Aosong Feng, Weihua Hu, Matthias Fey, Leandros Tassiulas, Jure Leskovec, and Rex Ying. From similarity to superiority: Channel clustering for time series forecasting. In Advances in Neural Information Processing Systems, volume 37, 2024.

[25] Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, and Yang Wang. Crossgnn: Confronting noisy multivariate time series via cross interaction refinement. In Advances in Neural Information Processing Systems, volume 36, pages 46885–46902, 2023.

[26] Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent network for traffic forecasting. In Advances in Neural Information Processing Systems, volume 33, 2020.

[27] Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Jinliang Deng, Quanjun Chen, and Xuan Song. Spatiotemporal adaptive embedding makes vanilla transformer sota for traffic forecasting. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 4125–4129, 2023.

[28] Mark Horowitz. 1.1 computing’s energy problem (and what we can do about it). In 2014 IEEE International Solid-State Circuits Conference Digest ofTechnical Papers (ISSCC), pages 10–14. IEEE, 2014.

## A Experimental Details

This appendix records the dataset characteristics, evaluation rules, implementation choices, additional forecasting results, and diagnostic experiments used in the paper. The two protocols are kept separate because they use different datasets, lookback windows, horizons, metrics, and model interfaces.

## A.1 Datasets and temporal characteristics

Table 4 lists the dimensionality exposed to the forecasting model. The sensor-network datasets contain substantial cross-variable structure, whereas the physical and financial datasets are more heterogeneous in their temporal regularity. All datasets used here are regularly sampled benchmark series with complete input windows; explicit missing-value handling is not part of the current pipeline.

Table 4: Datasets used by the two evaluation protocols. “Variables/nodes” denotes the input dimensionality after protocol preprocessing.
<table><tr><td>Protocol</td><td>Dataset</td><td>Domain</td><td></td><td>Variables/nodes Main temporal characteristics</td></tr><tr><td>SeqSNN</td><td>METR-LA</td><td>traffic sensors</td><td>207</td><td>Road-speed measurements with spatial correlation, daily repetition, and short-term congestion changes.</td></tr><tr><td></td><td>PEMS-BAY</td><td>traffic sensors</td><td>325</td><td>Large sensor network with correlated locations, recur- ring traffic cycles, and rapidly changing local conditions.</td></tr><tr><td></td><td>Solar</td><td>solar power</td><td>137</td><td>Diurnal structure with weather-driven local fluctuations and intermittent high-frequency changes.</td></tr><tr><td></td><td>Electricity</td><td>electricity load</td><td>321</td><td>Heterogeneous consumption channels with periodic structure and channel-specific temporal profiles.</td></tr><tr><td>SpikF</td><td>ECL</td><td>electricity load</td><td>321</td><td>Long-horizon electricity demand with periodic and het-</td></tr><tr><td></td><td>Weather</td><td>meteorological series</td><td>21</td><td>erogeneous channel behavior. Smooth physical variables with seasonal trends and multi-scale local variation.</td></tr><tr><td></td><td>ETTh1/ETTh2</td><td>transformer temperature</td><td>7</td><td>Hourly transformer measurements combining trend, pe- riodicity, and regime-dependent variation.</td></tr><tr><td></td><td>ETTm1/ETTm2</td><td>transformer temperature 7</td><td></td><td>Higher-frequency ETT benchmarks with finer local fluc- tuations.</td></tr><tr><td></td><td>Traffic</td><td>road sensors</td><td>862</td><td>High-dimensional traffic network with recurring pat- terns and dense cross-sensor dependencies.</td></tr><tr><td></td><td>Exchange</td><td>exchange rates</td><td>8</td><td>Financial series with weaker periodicity, nonstationary long-range behavior, and less stable correlations.</td></tr></table>

METR-LA and PEMS-BAY are traffic-network settings in the standard multivariate protocol; their spatial correlations make cross-channel information potentially useful. Solar and Electricity are dominated by periodic energy-generation or consumption patterns, but individual channels still exhibit different amplitudes and local variations. In the long-term protocol, ECL repeats the electricity domain with a distinct preprocessing pipeline, while the ETT datasets expose trend and periodic components at different sampling frequencies. Traffic stresses both long-range temporal structure and channel interaction. Exchange is a contrasting case: its irregular and weakly periodic behavior makes small absolute metric differences produce noticeable ranking changes.

## A.2 Evaluation protocols and preprocessing

Under the standard multivariate protocol, we evaluate METR-LA, PEMS-BAY, Solar, and Electricity at horizons {6, 24, 48, 96} using $R ^ { 2 }$ and RSE. The released SeqSNN data splits, normalization, and dataset-specific input windows are retained. The long-term protocol uses a lookback window of 96 and horizons {96, 192, 336, 720}, evaluated with MSE and MAE. The complete horizon-wise table is given in Section B.

## A.3 Evaluation metrics

For the standard multivariate benchmarks, the four datasets have different scales, dimensionalities, and horizon lengths. We therefore use the global scale-normalized metrics adopted by the protocol. With ground truth $y _ { i }$ , prediction $\widehat { y } _ { i }$ , and global mean y¯, they are

$$
R ^ { 2 } = 1 - \frac { \sum _ { i } ( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } } { \sum _ { i } ( y _ { i } - \bar { y } ) ^ { 2 } } , \qquad \mathrm { R S E } = \frac { \sqrt { \sum _ { i } ( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } } } { \sqrt { \sum _ { i } ( y _ { i } - \bar { y } ) ^ { 2 } } } .\tag{17}
$$

This normalization makes errors comparable across datasets whose raw units and variances differ. Higher $R ^ { 2 }$ and lower RSE are better.

For long-term forecasting, we report

$$
\mathrm { M S E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } , \qquad \mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | y _ { i } - \widehat { y } _ { i } | ,\tag{18}
$$

where N counts all forecast entries after flattening the horizon and channel dimensions. Dataset averages are computed after averaging the four horizons within each dataset.

## A.4 Implementation and reproducibility

The SeqSNN configurations use $D = 1 2 8$ for the compact channel representation. The released “spikelinear” configurations use a window of 12 for PEMS-BAY and dataset-specific windows for the remaining standard benchmarks. FSSE uses per-channel learnable decay factors and thresholds, surrogate-gradient scale 4, RevIN, component normalization, and horizon scaling. SSCA uses four attention heads, $T _ { s } = 4$ simulation steps, query–key scale 0.125, mask rank 8, $\gamma = 1$ , and mask threshold 0.3; stochastic mask sampling is disabled in both training and evaluation.

We use Adam with zero weight decay, batch size 32, learning rates specified by the released configurations, and early stopping on validation loss. Main table entries follow the configured repeated runs

## B Complete Long-Term Forecasting Results

Table 5 reports all dataset–horizon combinations under the long-term protocol. Bold and underlined values denote the best and second-best entries, respectively.

The horizon-wise results reveal differences that are obscured by the dataset averages. On ECL, SpikeLite obtains the lowest MSE and MAE at all four horizons, indicating that its advantage is maintained as the prediction length increases. On Weather, its average results are comparable to SpikF, with small differences that change direction across horizons. Among the ETT datasets, SpikeLite achieves the lowest average MSE and MAE on ETTm2, whereas the results on ETTh1, ETTh2, and ETTm1 are close to those of the strongest competing methods rather than uniformly better at every horizon.

The table also clarifies where SpikeLite is less competitive. On Traffic, it improves on SpikF at every horizon but trails iTransformer, which obtains the lowest errors on this dataset. On Exchange, the gap to SpikF is small at shorter horizons, while the larger error at horizon 720 raises SpikeLite’s dataset average. These cases qualify the aggregate results in the main text: the overall performance does not imply consistent superiority on every dataset or prediction horizon.

## C Stability Across Random Seeds

To assess the robustness of SpikeLite to random initialization, we repeat the experiments with three independent random seeds under both the standard multivariate and long-term forecasting protocols. For a metric value $q \in$

Table 5: Full horizon-wise long-term forecasting results under the SpikF protocol. Each entry reports MSE and MAE. Lower values are better. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td rowspan=1 colspan=20>Dataset  HorizonSpikeLite   SpikF  iTransformer RLinear  PatchTST Crossformer TimesNet  DLinear   SCINet  AutoformerMSEMAEMSEMAEMSEMAEMSEMAEMSEMAEMSEMAEMSEMAEMSEMAEMSEMAEMSEMAE</td></tr><tr><td rowspan=1 colspan=20>ECL    96     .145.236.156.252.148 .240 .201.281 .181 .270 .219.314.168.272.197.282.247 .345 .201 .317</td></tr><tr><td rowspan=1 colspan=1>192    .161</td><td rowspan=1 colspan=11>.250.169.262.162 .253 .201.283.188.274.231.322</td><td rowspan=1 colspan=8>.184.289 .196.285.257.355 .222 .334</td></tr><tr><td rowspan=1 colspan=1>336    .175</td><td rowspan=1 colspan=11>.264.188.281.178 .269 .215.298.204.293.246.337</td><td rowspan=1 colspan=8>.198.300.209.301.269.369 .231 .338</td></tr><tr><td rowspan=1 colspan=12>720    .218.300.219.306.225 .317 .257.331 .246.324.280 .363</td><td rowspan=1 colspan=8>.220.320.245.333.299.390 .254 .361</td></tr><tr><td rowspan=1 colspan=12>ECL Avg.       .175.263.183.275.178 .270 .219.298.205.290 .244 .334</td><td rowspan=1 colspan=8>.192.295.212.300.268.365 .227 .338</td></tr><tr><td rowspan=1 colspan=12>Weather 96     .158.196.163.200.174 .214 .192.232.177.218.158 .230</td><td rowspan=1 colspan=8>.172.220.196.255.221 .306.266 .336</td></tr><tr><td rowspan=1 colspan=1>192    .209</td><td rowspan=1 colspan=1>.243</td><td rowspan=1 colspan=1>.209</td><td rowspan=1 colspan=1>.241</td><td rowspan=1 colspan=4>.221 .254 .240.271</td><td rowspan=1 colspan=1>.225</td><td rowspan=1 colspan=3>.259.206 .277</td><td rowspan=1 colspan=2>.219.261</td><td rowspan=1 colspan=1>.237</td><td rowspan=1 colspan=1>.296</td><td rowspan=1 colspan=4>.261 .340 .307 .367</td></tr><tr><td rowspan=1 colspan=1>336    .266</td><td rowspan=1 colspan=1>.284</td><td rowspan=1 colspan=1>.266</td><td rowspan=1 colspan=1>.283</td><td rowspan=1 colspan=4>.278 .296 .292.307</td><td rowspan=1 colspan=1>.278</td><td rowspan=1 colspan=2>.297 .272</td><td rowspan=1 colspan=1>.335</td><td rowspan=1 colspan=1>.280</td><td rowspan=1 colspan=1>.306</td><td rowspan=1 colspan=1>.283</td><td rowspan=1 colspan=1>.335</td><td rowspan=1 colspan=4>.309.378.359 .395</td></tr><tr><td rowspan=1 colspan=1>720    .346</td><td rowspan=1 colspan=1>.337</td><td rowspan=1 colspan=2>.344.334</td><td rowspan=1 colspan=4>.358 .347 .364.353</td><td rowspan=1 colspan=1>.354</td><td rowspan=1 colspan=1>.348</td><td rowspan=1 colspan=1>.398</td><td rowspan=1 colspan=1>.418</td><td rowspan=1 colspan=1>.365</td><td rowspan=1 colspan=1>.359</td><td rowspan=1 colspan=1>.345</td><td rowspan=1 colspan=1>.381</td><td rowspan=1 colspan=4>.377.427 .419 .428</td></tr><tr><td rowspan=1 colspan=1>Weather Avg.    .245</td><td rowspan=1 colspan=1>.265</td><td rowspan=1 colspan=2>.245.265</td><td rowspan=1 colspan=4>.258 .278 .272.291</td><td rowspan=1 colspan=1>.259</td><td rowspan=1 colspan=1>.281</td><td rowspan=1 colspan=1>.259</td><td rowspan=1 colspan=1>.315</td><td rowspan=1 colspan=1>.259</td><td rowspan=1 colspan=1>.287</td><td rowspan=1 colspan=1>.265</td><td rowspan=1 colspan=1>.317</td><td rowspan=1 colspan=4>.292.363 .338 .382</td></tr><tr><td rowspan=1 colspan=1>ETTh1  96    .383</td><td rowspan=1 colspan=3>.391 .379.391</td><td rowspan=1 colspan=5>.386 .405 .386.395.414</td><td rowspan=1 colspan=3>.419.423 .448</td><td rowspan=1 colspan=4>.384 .402.386.400</td><td rowspan=1 colspan=4>.654 .599 .449 .459</td></tr><tr><td rowspan=1 colspan=1>192    .437</td><td rowspan=1 colspan=1>.422</td><td rowspan=1 colspan=1>.432</td><td rowspan=1 colspan=1>.421</td><td rowspan=1 colspan=1>.441</td><td rowspan=1 colspan=1>.436</td><td rowspan=1 colspan=1>.437</td><td rowspan=1 colspan=1>.424</td><td rowspan=1 colspan=1>.460</td><td rowspan=1 colspan=1>.445</td><td rowspan=1 colspan=1>.471</td><td rowspan=1 colspan=1>.474</td><td rowspan=1 colspan=1>.436</td><td rowspan=1 colspan=1>.429</td><td rowspan=1 colspan=1>.437</td><td rowspan=1 colspan=1>.432</td><td rowspan=1 colspan=2>.719.631</td><td rowspan=1 colspan=2>.500 .482</td></tr><tr><td rowspan=1 colspan=1>336    .472</td><td rowspan=1 colspan=1>.445</td><td rowspan=1 colspan=1>.473</td><td rowspan=1 colspan=1>.441</td><td rowspan=1 colspan=1>.487</td><td rowspan=1 colspan=1>.458</td><td rowspan=1 colspan=1>.479</td><td rowspan=1 colspan=1>.446</td><td rowspan=1 colspan=1>.501</td><td rowspan=1 colspan=1>.466</td><td rowspan=1 colspan=1>.570</td><td rowspan=1 colspan=1>.546</td><td rowspan=1 colspan=1>.491</td><td rowspan=1 colspan=1>.469</td><td rowspan=1 colspan=1>.481</td><td rowspan=1 colspan=1>.459</td><td rowspan=1 colspan=2>.778.659</td><td rowspan=1 colspan=2>.521 .496</td></tr><tr><td rowspan=1 colspan=1>720    .478</td><td rowspan=1 colspan=1>.460</td><td rowspan=1 colspan=1>.474</td><td rowspan=1 colspan=1>.459</td><td rowspan=1 colspan=1>.503</td><td rowspan=1 colspan=1>.491</td><td rowspan=1 colspan=1>.481</td><td rowspan=1 colspan=1>.470</td><td rowspan=1 colspan=1>.500</td><td rowspan=1 colspan=1>.488</td><td rowspan=1 colspan=1>.653</td><td rowspan=1 colspan=1>.621</td><td rowspan=1 colspan=1>.521</td><td rowspan=1 colspan=1>.500</td><td rowspan=1 colspan=1>.519</td><td rowspan=1 colspan=1>.516</td><td rowspan=1 colspan=2>.836.699</td><td rowspan=1 colspan=2>.514.512</td></tr><tr><td rowspan=1 colspan=1>ETTh1 Avg.     .443</td><td rowspan=1 colspan=1>.429</td><td rowspan=1 colspan=1>.440</td><td rowspan=1 colspan=1>.428</td><td rowspan=1 colspan=1>.454</td><td rowspan=1 colspan=1>.447</td><td rowspan=1 colspan=1>.446</td><td rowspan=1 colspan=1>.434</td><td rowspan=1 colspan=1>.469</td><td rowspan=1 colspan=1>.454</td><td rowspan=1 colspan=1>.529</td><td rowspan=1 colspan=1>.522</td><td rowspan=1 colspan=1>.458</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.456</td><td rowspan=1 colspan=1>.452</td><td rowspan=1 colspan=2>.747.647</td><td rowspan=1 colspan=2>.496.487</td></tr><tr><td rowspan=1 colspan=1>ETTh2  96     .294</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.290</td><td rowspan=1 colspan=2>.336.297</td><td rowspan=1 colspan=2>.349 .288</td><td rowspan=1 colspan=1>.338</td><td rowspan=1 colspan=1>.302</td><td rowspan=1 colspan=1>.348</td><td rowspan=1 colspan=2>.745 .584</td><td rowspan=1 colspan=2>.340 .374</td><td rowspan=1 colspan=1>.333</td><td rowspan=1 colspan=1>.387</td><td rowspan=1 colspan=4>.707 .621 .346 .388</td></tr><tr><td rowspan=1 colspan=1>192    .366</td><td rowspan=1 colspan=1>.385</td><td rowspan=1 colspan=1>.367</td><td rowspan=1 colspan=1>.385</td><td rowspan=1 colspan=1>.380</td><td rowspan=1 colspan=2>.400 .374</td><td rowspan=1 colspan=1>.390</td><td rowspan=1 colspan=1>.388</td><td rowspan=1 colspan=1>.400</td><td rowspan=1 colspan=1>.877</td><td rowspan=1 colspan=1>.656</td><td rowspan=1 colspan=1>.402</td><td rowspan=1 colspan=1>.414</td><td rowspan=1 colspan=1>.477</td><td rowspan=1 colspan=1>.476</td><td rowspan=1 colspan=1>.860</td><td rowspan=1 colspan=1>.689</td><td rowspan=1 colspan=1>.456</td><td rowspan=1 colspan=1>.452</td></tr><tr><td rowspan=1 colspan=1>336    .409</td><td rowspan=1 colspan=1>.418</td><td rowspan=1 colspan=1>.414</td><td rowspan=1 colspan=1>.420</td><td rowspan=1 colspan=1>.427</td><td rowspan=1 colspan=2>.432.415</td><td rowspan=1 colspan=1>.426</td><td rowspan=1 colspan=1>.426</td><td rowspan=1 colspan=1>.433</td><td rowspan=1 colspan=1>1.043</td><td rowspan=1 colspan=1>.731</td><td rowspan=1 colspan=1>.452</td><td rowspan=1 colspan=1>.452</td><td rowspan=1 colspan=1>.594</td><td rowspan=1 colspan=1>.541</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>.744</td><td rowspan=1 colspan=1>.482</td><td rowspan=1 colspan=1>.486</td></tr><tr><td rowspan=1 colspan=1>720    .426</td><td rowspan=1 colspan=1>.438</td><td rowspan=1 colspan=1>.416</td><td rowspan=1 colspan=1>.436</td><td rowspan=1 colspan=1>.427</td><td rowspan=1 colspan=2>.445 .420</td><td rowspan=1 colspan=1>.440</td><td rowspan=1 colspan=1>.431</td><td rowspan=1 colspan=1>.446</td><td rowspan=1 colspan=1>1.104</td><td rowspan=1 colspan=1>.763</td><td rowspan=1 colspan=1>.462</td><td rowspan=1 colspan=1>.468</td><td rowspan=1 colspan=1>.831</td><td rowspan=1 colspan=1>.657</td><td rowspan=1 colspan=1>1.249</td><td rowspan=1 colspan=1>.838</td><td rowspan=1 colspan=1>.515</td><td rowspan=1 colspan=1>.511</td></tr><tr><td rowspan=1 colspan=1>ETTh2 Avg.     .374</td><td rowspan=1 colspan=1>.394</td><td rowspan=1 colspan=1>.372</td><td rowspan=1 colspan=1>.394</td><td rowspan=1 colspan=1>.383</td><td rowspan=1 colspan=2>.407 .374</td><td rowspan=1 colspan=1>.398</td><td rowspan=1 colspan=1>.387</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=1>.942</td><td rowspan=1 colspan=1>.684</td><td rowspan=1 colspan=1>.414</td><td rowspan=1 colspan=1>.427</td><td rowspan=1 colspan=1>.559</td><td rowspan=1 colspan=1>.515</td><td rowspan=1 colspan=1>.954</td><td rowspan=1 colspan=1>.723</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.459</td></tr><tr><td rowspan=1 colspan=1>ETTm1  96     .311</td><td rowspan=1 colspan=1>.345</td><td rowspan=1 colspan=1>.317</td><td rowspan=1 colspan=1>.345</td><td rowspan=1 colspan=1>.334</td><td rowspan=1 colspan=2>.368 .355</td><td rowspan=1 colspan=1>.376</td><td rowspan=1 colspan=1>.329</td><td rowspan=1 colspan=1>.367</td><td rowspan=1 colspan=1>.404</td><td rowspan=1 colspan=1>.426</td><td rowspan=1 colspan=1>.338</td><td rowspan=1 colspan=1>.375</td><td rowspan=1 colspan=1>.345</td><td rowspan=1 colspan=1>.372</td><td rowspan=1 colspan=1>.418</td><td rowspan=1 colspan=3>.438 .505 .475</td></tr><tr><td rowspan=1 colspan=1>192    .369</td><td rowspan=1 colspan=1>.377</td><td rowspan=1 colspan=1>.372</td><td rowspan=1 colspan=1>.372</td><td rowspan=1 colspan=1>.379</td><td rowspan=1 colspan=1>.391</td><td rowspan=1 colspan=1>.391</td><td rowspan=1 colspan=1>.392</td><td rowspan=1 colspan=1>.367</td><td rowspan=1 colspan=1>.385</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.451</td><td rowspan=1 colspan=1>.374</td><td rowspan=1 colspan=1>.387</td><td rowspan=1 colspan=1>.380</td><td rowspan=1 colspan=1>.389</td><td rowspan=1 colspan=1>.439</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.553</td><td rowspan=1 colspan=1>.496</td></tr><tr><td rowspan=1 colspan=1>336    .396</td><td rowspan=1 colspan=1>.395</td><td rowspan=1 colspan=1>.401</td><td rowspan=1 colspan=1>.394</td><td rowspan=1 colspan=1>.426</td><td rowspan=1 colspan=1>.420</td><td rowspan=1 colspan=1>.424</td><td rowspan=1 colspan=1>.415</td><td rowspan=1 colspan=1>.399</td><td rowspan=1 colspan=1>.410</td><td rowspan=1 colspan=1>.532</td><td rowspan=1 colspan=1>.515</td><td rowspan=1 colspan=1>.410</td><td rowspan=1 colspan=1>.411</td><td rowspan=1 colspan=1>.413</td><td rowspan=1 colspan=1>.413</td><td rowspan=1 colspan=1>.490</td><td rowspan=1 colspan=1>.485</td><td rowspan=1 colspan=1>.621</td><td rowspan=1 colspan=1>.537</td></tr><tr><td rowspan=1 colspan=1>720    .473</td><td rowspan=1 colspan=1>.439</td><td rowspan=1 colspan=1>.461</td><td rowspan=1 colspan=1>.430</td><td rowspan=1 colspan=1>.491</td><td rowspan=1 colspan=1>.459</td><td rowspan=1 colspan=1>.487</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.454</td><td rowspan=1 colspan=1>.439</td><td rowspan=1 colspan=1>.666</td><td rowspan=1 colspan=1>.589</td><td rowspan=1 colspan=1>.478</td><td rowspan=1 colspan=1>.450</td><td rowspan=1 colspan=1>.474</td><td rowspan=1 colspan=1>.453</td><td rowspan=1 colspan=1>.595</td><td rowspan=1 colspan=1>.550</td><td rowspan=1 colspan=1>.671</td><td rowspan=1 colspan=1>.561</td></tr><tr><td rowspan=1 colspan=1>ETTm1 Avg.     .387</td><td rowspan=1 colspan=1>.389</td><td rowspan=1 colspan=1>.388</td><td rowspan=1 colspan=1>.385</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=2>.410 .414</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=1>.387</td><td rowspan=1 colspan=1>.400</td><td rowspan=1 colspan=1>.513</td><td rowspan=1 colspan=1>.496</td><td rowspan=1 colspan=1>.400</td><td rowspan=1 colspan=1>.406</td><td rowspan=1 colspan=1>.403</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=1>.485</td><td rowspan=1 colspan=2>.481 .588</td><td rowspan=1 colspan=1>.517</td></tr><tr><td rowspan=1 colspan=1>ETTm2 96     .175</td><td rowspan=1 colspan=1>.254</td><td rowspan=1 colspan=1>.175</td><td rowspan=1 colspan=1>.251</td><td rowspan=1 colspan=1>.180</td><td rowspan=1 colspan=2>.264 .182</td><td rowspan=1 colspan=1>.265</td><td rowspan=1 colspan=1>.175</td><td rowspan=1 colspan=1>.259</td><td rowspan=1 colspan=1>.287</td><td rowspan=1 colspan=1>.366</td><td rowspan=1 colspan=1>.187</td><td rowspan=1 colspan=1>.267</td><td rowspan=1 colspan=1>.193</td><td rowspan=1 colspan=1>.292</td><td rowspan=1 colspan=1>.286</td><td rowspan=1 colspan=3>.274 .255 .339</td></tr><tr><td rowspan=1 colspan=1>192    .240</td><td rowspan=1 colspan=1>.295</td><td rowspan=1 colspan=1>.242</td><td rowspan=1 colspan=1>.296</td><td rowspan=1 colspan=1>.250</td><td rowspan=1 colspan=1>.309</td><td rowspan=1 colspan=1>.246</td><td rowspan=1 colspan=1>.304</td><td rowspan=1 colspan=1>.241</td><td rowspan=1 colspan=1>.302</td><td rowspan=1 colspan=1>.414</td><td rowspan=1 colspan=1>.492</td><td rowspan=1 colspan=1>.249</td><td rowspan=1 colspan=1>.309</td><td rowspan=1 colspan=1>.284</td><td rowspan=1 colspan=1>.362</td><td rowspan=1 colspan=1>.399</td><td rowspan=1 colspan=2>.445 .281</td><td rowspan=1 colspan=1>.340</td></tr><tr><td rowspan=1 colspan=1>336    .302</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.302</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.311</td><td rowspan=1 colspan=1>.348</td><td rowspan=1 colspan=1>.307</td><td rowspan=1 colspan=1>.342</td><td rowspan=1 colspan=1>.305</td><td rowspan=1 colspan=1>.343</td><td rowspan=1 colspan=1>.597</td><td rowspan=1 colspan=1>.542</td><td rowspan=1 colspan=1>.321</td><td rowspan=1 colspan=1>.351</td><td rowspan=1 colspan=1>.369</td><td rowspan=1 colspan=1>.427</td><td rowspan=1 colspan=1>.637</td><td rowspan=1 colspan=2>.591 .339</td><td rowspan=1 colspan=1>.372</td></tr><tr><td rowspan=1 colspan=1>720    .395</td><td rowspan=1 colspan=1>.390</td><td rowspan=1 colspan=1>.405</td><td rowspan=1 colspan=1>.397</td><td rowspan=1 colspan=1>.412</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=1>.407</td><td rowspan=1 colspan=1>.398</td><td rowspan=1 colspan=1>.402</td><td rowspan=1 colspan=1>.400</td><td rowspan=1 colspan=1>1.730</td><td rowspan=1 colspan=1>1.042</td><td rowspan=1 colspan=1>.408</td><td rowspan=1 colspan=1>.403</td><td rowspan=1 colspan=1>.554</td><td rowspan=1 colspan=1>.522</td><td rowspan=1 colspan=1>.960</td><td rowspan=1 colspan=2>.735 .433</td><td rowspan=1 colspan=1>.432</td></tr><tr><td rowspan=1 colspan=1>ETTm2 Avg.     .278</td><td rowspan=1 colspan=1>.319</td><td rowspan=1 colspan=1>.281</td><td rowspan=1 colspan=1>.320</td><td rowspan=1 colspan=1>.288</td><td rowspan=1 colspan=2>.327 .286</td><td rowspan=1 colspan=1>.327</td><td rowspan=1 colspan=1>.281</td><td rowspan=1 colspan=2>.326 .757</td><td rowspan=1 colspan=1>.610</td><td rowspan=1 colspan=1>.291</td><td rowspan=1 colspan=1>.333</td><td rowspan=1 colspan=1>.350</td><td rowspan=1 colspan=1>.401</td><td rowspan=1 colspan=1>.571</td><td rowspan=1 colspan=3>.537 .327 .371</td></tr><tr><td rowspan=1 colspan=1>Traffic  96    .453</td><td rowspan=1 colspan=1>.285</td><td rowspan=1 colspan=1>.477</td><td rowspan=1 colspan=1>.286</td><td rowspan=1 colspan=1>.395</td><td rowspan=1 colspan=2>.268 .649</td><td rowspan=1 colspan=2>.389.462</td><td rowspan=1 colspan=2>.295 .522</td><td rowspan=1 colspan=1>.290</td><td rowspan=1 colspan=1>.593</td><td rowspan=1 colspan=1>.321</td><td rowspan=1 colspan=1>.650</td><td rowspan=1 colspan=1>.396</td><td rowspan=1 colspan=4>.788 .499 .613 .388</td></tr><tr><td rowspan=1 colspan=1>192    .468</td><td rowspan=1 colspan=1>.287</td><td rowspan=1 colspan=1>.481</td><td rowspan=1 colspan=1>.289</td><td rowspan=1 colspan=1>.417</td><td rowspan=1 colspan=2>.276 .601</td><td rowspan=1 colspan=2>.366.466</td><td rowspan=1 colspan=2>.296 .530</td><td rowspan=1 colspan=1>.293</td><td rowspan=1 colspan=1>.617</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.598</td><td rowspan=1 colspan=1>.370</td><td rowspan=1 colspan=1>.789</td><td rowspan=1 colspan=3>.505 .616 .382</td></tr><tr><td rowspan=1 colspan=1>336    .481</td><td rowspan=1 colspan=1>.293</td><td rowspan=1 colspan=1>.499</td><td rowspan=1 colspan=1>.295</td><td rowspan=1 colspan=1>.433</td><td rowspan=1 colspan=2>.283 .609</td><td rowspan=1 colspan=1>.369</td><td rowspan=1 colspan=1>.482</td><td rowspan=1 colspan=1>.304</td><td rowspan=1 colspan=1>.558</td><td rowspan=1 colspan=1>.305</td><td rowspan=1 colspan=1>.629</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.605</td><td rowspan=1 colspan=1>.373</td><td rowspan=1 colspan=1>.797</td><td rowspan=1 colspan=3>.508.622.337</td></tr><tr><td rowspan=1 colspan=1>720    .514</td><td rowspan=1 colspan=1>.311</td><td rowspan=1 colspan=1>.533</td><td rowspan=1 colspan=1>.312</td><td rowspan=1 colspan=1>.467</td><td rowspan=1 colspan=2>.302.647</td><td rowspan=1 colspan=1>.387</td><td rowspan=1 colspan=1>.514</td><td rowspan=1 colspan=1>.322</td><td rowspan=1 colspan=1>.589</td><td rowspan=1 colspan=1>.328</td><td rowspan=1 colspan=1>.640</td><td rowspan=1 colspan=1>.350</td><td rowspan=1 colspan=1>.645</td><td rowspan=1 colspan=1>.394</td><td rowspan=1 colspan=1>.841</td><td rowspan=1 colspan=3>.523 .660 .408</td></tr><tr><td rowspan=1 colspan=1>Traffic Avg.      .479</td><td rowspan=1 colspan=3>.294 .497 .296</td><td rowspan=1 colspan=3>.428 .282 .626</td><td rowspan=1 colspan=5>.378.481 .304 .550 .304</td><td rowspan=1 colspan=1>.620</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>.625</td><td rowspan=1 colspan=5>.383 .804 .509 .628 .379</td></tr><tr><td rowspan=1 colspan=1>Exchange96     .084</td><td rowspan=1 colspan=6>.204.084.201 .086 .206 .093</td><td rowspan=1 colspan=2>.217.088</td><td rowspan=1 colspan=3>.205 .256 .367</td><td rowspan=1 colspan=3>.107 .234.088</td><td rowspan=1 colspan=5>.218 .267 .396 .197 .323</td></tr><tr><td rowspan=1 colspan=1>192    .180</td><td rowspan=1 colspan=3>.300.180.300</td><td rowspan=1 colspan=3>.177 .299 .184</td><td rowspan=1 colspan=2>.307.176</td><td rowspan=1 colspan=2>.299 .470</td><td rowspan=1 colspan=1>.509</td><td rowspan=1 colspan=2>.226 .344</td><td rowspan=1 colspan=1>.176</td><td rowspan=1 colspan=1>.315</td><td rowspan=1 colspan=4>.351 .459 .300 .369</td></tr><tr><td rowspan=1 colspan=1>336    .336</td><td rowspan=1 colspan=2>.417.334</td><td rowspan=1 colspan=1>.417</td><td rowspan=1 colspan=1>.331</td><td rowspan=1 colspan=2>.417 .351</td><td rowspan=1 colspan=1>.432</td><td rowspan=1 colspan=1>.301</td><td rowspan=1 colspan=2>.3971.268</td><td rowspan=1 colspan=1>.883</td><td rowspan=1 colspan=1>.367</td><td rowspan=1 colspan=1>.448</td><td rowspan=1 colspan=1>.313</td><td rowspan=1 colspan=1>.427</td><td rowspan=1 colspan=1>1.324</td><td rowspan=1 colspan=3>.853 .509 .524</td></tr><tr><td rowspan=1 colspan=1>720    .864</td><td rowspan=1 colspan=2>.695.841</td><td rowspan=1 colspan=1>.690</td><td rowspan=1 colspan=1>.847</td><td rowspan=1 colspan=2>.691 .886</td><td rowspan=1 colspan=1>.714</td><td rowspan=1 colspan=1>.901</td><td rowspan=1 colspan=2>.7141.767</td><td rowspan=1 colspan=1>1.068</td><td rowspan=1 colspan=1>.964</td><td rowspan=1 colspan=1>.746</td><td rowspan=1 colspan=1>.839</td><td rowspan=1 colspan=1>.695</td><td rowspan=1 colspan=1>1.058</td><td rowspan=1 colspan=3>.7971.447.941</td></tr><tr><td rowspan=1 colspan=1>Exchange Avg.    .366</td><td rowspan=1 colspan=6>.404 .360 .402 .360 .403 .378</td><td rowspan=1 colspan=4>.417 .367 .404 .940</td><td rowspan=1 colspan=1>.707</td><td rowspan=1 colspan=8>.416 .443 .354 .414 .750 .626 .613 .539</td></tr></table>

{R<sup>2</sup>, RSE, MSE, MAE}, we report the sample mean and standard deviation across the three runs:

$$
s ( q ) = \sqrt { \frac { 1 } { N - 1 } \sum _ { n = 1 } ^ { N } \left( q _ { n } - \bar { q } \right) ^ { 2 } } , \qquad N = 3 .\tag{19}
$$

Table 6 reports the three-seed results under the SpikF protocol. The “Avg.” column is first computed within each run by averaging the four prediction horizons, after which the mean and standard deviation are calculated across the three runs. Table 7 provides the corresponding results under the standard multivariate protocol.

The results show that SpikeLite is stable across random initializations. Under the SpikF protocol, the standard deviations are at most 0.0012 for both MSE and MAE, and several entries remain identical after rounding. Under the standard multivariate protocol, most standard deviations are at most 0.001, with the largest observed deviation equal to 0.003. Therefore, the main performance trends and relative rankings are not sensitive to the particular random seed.

Table 6: Three-seed stability under the SpikF protocol. Each entry reports the mean ± sample standard deviation over three independent runs. The $\operatorname { A v g } .$ column is computed within each run over the four prediction horizons.
<table><tr><td>Dataset</td><td>Metric</td><td>96</td><td>192</td><td>336</td><td>720</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>ECL</td><td>MSE</td><td> $. 1 4 5 3 { \pm } . 0 0 0 6$ </td><td> $. 1 6 1 0 \pm . 0 0 0 0$ </td><td> $. 1 7 5 0 { \pm } . 0 0 0 0$ </td><td> $. 2 1 8 7 \pm . 0 0 1 2$ </td><td> $. 1 7 5 0 { \pm } . 0 0 0 3$ </td></tr><tr><td></td><td>MAE</td><td> $. 2 3 6 3 { \pm } . 0 0 0 6$ </td><td> $. 2 5 0 0 { \pm } . 0 0 0 0$ </td><td> $. 2 6 4 0 { \pm } . 0 0 0 0$ </td><td> $. 3 0 0 7 { \scriptstyle \pm . 0 0 1 2 }$ </td><td> $. 2 6 2 8 \pm . 0 0 0 3$ </td></tr><tr><td>Weather</td><td>MSE</td><td> $. 1 5 8 7 \pm . 0 0 1 2$ </td><td> $. 2 0 9 0 { \pm } . 0 0 0 0$ </td><td> $. 2 6 6 0 { \pm } . 0 0 0 0$ </td><td> $. 3 4 6 0 \pm . 0 0 0 0$ </td><td> $. 2 4 4 9 \pm . 0 0 0 3$ </td></tr><tr><td></td><td>MAE</td><td> $. 1 9 7 7 { \scriptstyle \pm . 0 0 1 2 }$ </td><td> $. 2 4 4 0 \pm . 0 0 0 0$ </td><td> $. 2 8 5 3 { \pm } . 0 0 0 6$ </td><td> $. 3 3 8 0 \pm . 0 0 0 0$ </td><td> $. 2 6 6 3 { \pm } . 0 0 0 3$ </td></tr><tr><td>ETTm2</td><td>MSE</td><td> $. 1 7 5 3 { \pm } . 0 0 0 6$ </td><td> $. 2 4 0 0 \pm . 0 0 0 0$ </td><td> $. 3 0 2 3 \pm . 0 0 0 6$ </td><td> $. 3 9 5 0 \pm . 0 0 0 0$ </td><td> $. 2 7 8 2 \pm . 0 0 0 3$ </td></tr><tr><td></td><td>MAE</td><td> $. 2 5 4 0 \pm . 0 0 0 0$ </td><td> $. 2 9 5 0 \pm . 0 0 0 0$ </td><td> $. 3 3 6 0 { \pm } . 0 0 0 0$ </td><td> $. 3 9 0 0 { \pm } . 0 0 0 0$ </td><td> $. 3 1 8 8 \pm . 0 0 0 0$ </td></tr><tr><td></td><td>MSE</td><td> $. 4 5 3 0 \pm . 0 0 0 0$ </td><td> $. 4 6 8 0 \pm . 0 0 0 0$ </td><td> $. 4 8 1 0 \pm . 0 0 0 0$ </td><td> $. 5 1 4 0 \pm . 0 0 0 0$ </td><td> $. 4 7 9 0 \pm . 0 0 0 0$ </td></tr><tr><td>Traffic</td><td>MAE</td><td> $. 2 8 5 0 \pm . 0 0 0 0$ </td><td> $. 2 8 7 0 \pm . 0 0 0 0$ </td><td> $. 2 9 3 0 { \pm } . 0 0 0 0$ </td><td> $. 3 1 1 0 \pm . 0 0 0 0$ </td><td> $. 2 9 4 0 { \pm } . 0 0 0 0$ </td></tr></table>

Table 7: Three-seed stability under the standard multivariate protocol. Each entry reports the mean ± sample standard deviation over three independent runs.
<table><tr><td>Dataset</td><td>Metric</td><td>Horizon 6</td><td>Horizon 24</td><td>Horizon 48</td><td>Horizon 96</td></tr><tr><td rowspan="2">PEMS-BAY</td><td> $R ^ { 2 } \uparrow$ </td><td> $. 8 9 9 \pm . 0 0 1$ </td><td> $. 7 8 1 \pm . 0 0 1$ </td><td> $. 7 1 5 \pm . 0 0 0$ </td><td> $. 6 3 9 \pm . 0 0 3$ </td></tr><tr><td> $R S E \downarrow$ </td><td> $. 3 6 1 \pm . 0 0 0$ </td><td> $. 5 0 6 \pm . 0 0 1$ </td><td> $. 5 7 7 \pm . 0 0 0$ </td><td> $. 6 4 9 \pm . 0 0 2$ </td></tr><tr><td rowspan="2">METR-LA</td><td> $R ^ { 2 } \uparrow$ </td><td> $. 8 4 7 \pm . 0 0 1$ </td><td> $. 6 2 2 \pm . 0 0 0$ </td><td> $. 4 5 2 \pm . 0 0 0$ </td><td> $. 3 1 6 \pm . 0 0 1$ </td></tr><tr><td> $R S E \downarrow$ </td><td> $. 4 1 4 \pm . 0 0 1$ </td><td> $. 6 4 9 \pm . 0 0 0$ </td><td> $. 7 8 1 \pm . 0 0 2$ </td><td> $. 8 7 3 \pm . 0 0 2$ </td></tr><tr><td rowspan="2">Solar</td><td> $R ^ { 2 } \uparrow$ </td><td> $. 9 6 2 \pm . 0 0 0$ </td><td> $. 8 8 5 \pm . 0 0 1$ </td><td> $. 8 1 0 \pm . 0 0 1$ </td><td> $. 7 6 7 \pm . 0 0 1$ </td></tr><tr><td> $R S E \downarrow$ </td><td> $. 2 0 1 \pm . 0 0 0$ </td><td> $. 3 4 8 \pm . 0 0 0$ </td><td> $. 4 4 6 \pm . 0 0 0$ </td><td> $. 4 9 6 \pm . 0 0 0$ </td></tr><tr><td rowspan="2">Electricity</td><td> $R ^ { 2 } \uparrow$ </td><td> $. 9 9 3 \pm . 0 0 0$ </td><td> $. 9 9 1 \pm . 0 0 1$ </td><td> $. 9 8 8 \pm . 0 0 0$ </td><td> $. 9 8 3 \pm . 0 0 0$ </td></tr><tr><td> $R S E \downarrow$ </td><td> $. 1 4 6 \pm . 0 0 0$ </td><td> $. 1 6 7 \pm . 0 0 1$ </td><td> $. 1 9 6 \pm . 0 0 0$ </td><td> $. 2 3 3 \pm . 0 0 0$ </td></tr></table>

## D Additional Module Ablations

The main text focuses on Traffic because its sensor variables exhibit strong useful interactions. Table 8 gives the horizon-wise Traffic and Weather results for removing FSSE or SSCA while keeping all other settings fixed.

Table 8: Horizon-wise ablations under the long-term protocol. Lower MSE and MAE are better.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Variant</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">Traffic</td><td>SpikeLite</td><td>.453</td><td>.285</td><td>.468</td><td>.287</td><td>.481</td><td>.293</td><td>.514</td><td>.311</td></tr><tr><td>w/o FSSE</td><td>.463</td><td>.291</td><td>.479</td><td>.292</td><td>.491</td><td>.298</td><td>.524</td><td>.315</td></tr><tr><td>w/o SSCA</td><td>.615</td><td>.345</td><td>.581</td><td>.327</td><td>.589</td><td>.330</td><td>.624</td><td>.349</td></tr><tr><td rowspan="3">Weather</td><td>SpikeLite</td><td>.158</td><td>.197</td><td>.209</td><td>.244</td><td>.266</td><td>.285</td><td>.346</td><td>.338</td></tr><tr><td>w/o FSSE</td><td>.172</td><td>.210</td><td>.223</td><td>.256</td><td>.278</td><td>.297</td><td>.358</td><td>.350</td></tr><tr><td>w/o SSCA</td><td>.179</td><td>.215</td><td>.224</td><td>.256</td><td>.278</td><td>.296</td><td>.356</td><td>.357</td></tr></table>

On Traffic, removing FSSE raises the averaged MSE/MAE from (.479, .294) to (.489, .299), while removing SSCA produces the larger degradation to (.602, .338). Weather shows smaller module gaps, consistent with weaker benefits from explicit cross-channel exchange. These results support the conditional design: FSSE supplies the input representation, while SSCA is most useful when channel interaction carries predictive information.

## E Frequency-Selective Mechanism Diagnostics

We analyze FSSE before its temporal projections and before SSCA. For a component $b _ { n , c } ^ { ( j ) } [ l ]$ from test window n and channel $c ,$ we remove the temporal mean, apply a Hann window $w [ l ]$ , and compute

$$
P _ { n , c } ^ { ( j ) } ( f ) = \left| \mathrm { r F F T } \left( w [ l ] \left( b _ { n , c } ^ { ( j ) } [ l ] - \bar { b } _ { n , c } ^ { ( j ) } \right) \right) \right| ^ { 2 } .\tag{20}
$$

g  
![](images/74c900244c18a62bec4e7e1a52452402ff628b1e990ffc6ce5e9082e7394992f.jpg)

![](images/c59fdfc7dec5220a16c95d4602f1672899f9710fda0f072084c5fea1c50e3657.jpg)

![](images/0063e1eaf07d6cb208c99dad34eafab95076f9402d2a93c73444d85f72968f27.jpg)

![](images/a4fbe92f2b2dabf750b4cebf3aa8b97c36f03e9ca06c322e8afd001930c35b5c.jpg)

![](images/a9f26210c4bdc89e7ed5b2b21ccccbba24062d0bae187538b6df4111f7cafdda.jpg)

![](images/637e8f8b13c88bd3f8f84198d14a6cabd84f2b8e09f38adb471a0e3d61abd8a6.jpg)

![](images/32242fa9dd21aca36cf64d7d32df6eba33bd3a16f0908d3801157e477f22acc4.jpg)

![](images/92159d4ae1b0678e799232a8db56e282fa0501988731a93d6e8371134911e65f.jpg)  
Figure 5: Frequency diagnostics for FSSE. Panels (a)–(c): real-data component spectra. Panel (d): spectral centroids. Panel (e): initialized and trained decay factors. Panel (f): branch-order violations and decay–centroid concordance. Panels (g)–(h): PEMS sinusoidal fundamental gain and firing probability.

The spectral centroid used in the real-data summary is

$$
\mu _ { f } ^ { ( j ) } = \frac { \sum _ { f > 0 } f P ^ { ( j ) } ( f ) } { \sum _ { f > 0 } P ^ { ( j ) } ( f ) } .\tag{21}
$$

Figure 5 makes the panel mapping explicit: panels (a)–(c) show component spectra for Electricity, PEMS-BAY, and ECL; (d) compares component spectral centroids; (e) compares initialized and trained decay factors; (f) reports order violations and decay–centroid concordance; and (g)–(h) show the PEMS sine-sweep fundamental gain and LIF firing probability. The branches are frequency-sensitive but are not forced into disjoint bands, which is consistent with the nonlinear threshold, reset, and event-gating operations.

Before learnable component scaling and temporal projection, the successive difference construction is exactly reconstructive: $\begin{array} { r } { \sum _ { j = 0 } ^ { K } \mathbf { B } ^ { ( j ) } = \mathbf { X } } \end{array}$ .. The largest floating-point reconstruction error in the diagnostic is $1 . 6 7 \times 1 0 ^ { - 6 }$ . Thus, FSSE reorganizes the input response without discarding the signal at the decomposition stage.

## F Energy Accounting

We report the encoder-level energy estimate for the ECL dataset with a prediction horizon of 720. We follow the established 45-nm CMOS accounting convention, assigning $E _ { \mathrm { M A C } } = 4 . 6$ pJ to a dense multiply–accumulate operation and $E _ { \mathrm { A C } } = 0 . 9 \ : \mathrm { p J }$ to an accumulation. The task-specific prediction head is excluded for every model in the comparison.

For SpikeLite, the dense operation count is decomposed into the FSSE encoder, temporal projections, and, when enabled, SSCA:

$$
N _ { \mathrm { d e n s e } } = 3 K L C + ( K + 1 ) C L D\tag{22}
$$

$$
+ \mathbb { I } _ { \mathrm { S S C A } } \left[ 2 T _ { s } C D + T _ { s } ( 3 C D ^ { 2 } + 2 C ^ { 2 } D + C D ^ { 2 } ) + N _ { \mathrm { m a s k } } \right] ,\tag{23}
$$

where $N _ { \mathrm { m a s k } }$ accounts for the rFFT and low-rank affinity calculation. Measured spike rates are used to convert spike-driven terms into effective accumulation operations. The final energy estimate is

$$
\begin{array} { r } { E = E _ { \mathrm { M A C } } N _ { \mathrm { M A C } } + E _ { \mathrm { A C } } N _ { \mathrm { A C } } . } \end{array}\tag{24}
$$

## G Toward Hardware Realization

SpikeLite uses hardware-compatible primitives, but the following discussion is an implementability analysis rather than a measured hardware result.

FSSE dataflow. Each input channel and LIF branch maintains one membrane state and one previous-event state. Leaky integration, thresholding, subtractive reset, and event gating are local operations with no all-to-all communication. Successive differences and the terminal residual require only additions and subtractions. Inactive events can be omitted from a compressed event stream, while an active event carries its membrane payload; this maps the encoder to a mixed event-control and membrane-value data path.

SSCA dataflow. SSCA can tile the Q/K/V projections over channels and hidden dimensions. The binary mask is generated once per input window and stored as a bit matrix or index list. Masked channel blocks can then be skipped by a scheduler during the $T _ { s }$ simulation steps. For large C, blockwise generation avoids materializing a dense score matrix; for dense masks, the implementation can fall back to regular tiled attention.

## H Limitations and Future Work

Limitations. The current benchmarks provide regularly sampled, fully observed windows, so SpikeLite does not yet model missing values, irregular timestamps, or online arrival of observations. The reported results therefore establish the method under complete-window forecasting rather than under data-imputation or irregular-sampling conditions.

Future work. We will implement FSSE and SSCA on programmable or fabricated neuromorphic hardware and measure actual energy, latency, memory traffic, and sparse execution efficiency. We will also study irregular time series, missing-data forecasting, online prediction, and cross-dataset transfer in the future work.