# SpikeSSL: A Universal Spike Inference Framework with Dynamics-Informed State-Space Layers

Chenghao Yue<sup>1,∗</sup> Siming Xing<sup>1,∗</sup> Shuran Liu<sup>1</sup> Angran Li<sup>2</sup> Yuanlong Zhang<sup>1,†</sup> <sup>1</sup>School of Life Sciences, Tsinghua University, Beijing, China <sup>2</sup>School of Future Information Innovation, Fudan University, Shanghai, China {yuech26,xsm25,liusr24}@mails.tsinghua.edu.cn arli25@m.fudan.edu.cn; yuanlongzhang@tsinghua.edu.cn <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

## Abstract

Two-photon calcium imaging is a standard tool for recording large neural populations in vivo, yet inferring spikes accurately across the growing diversity of calcium indicators remains an open problem. Existing supervised methods achieve reasonable in-domain accuracy but generalize poorly to unseen indicators, because different indicators induce distinct fluorescence kinetics and signal statistics while existing architectures remain relatively simple generic temporal regressors without dynamics-matched inductive bias. We propose SpikeSSL, a universal spike inference framework whose temporal backbone is a bank of bidirectional IIR statespace layers broadly motivated by calcium dynamics. A multi-modal conditioning encoder maps indicator identity, sampling rate, and trace-level signal statistics into a global conditioning vector that modulates the backbone via Adaptive Layer Normalization, while a heteroscedastic variance head provides calibrated per-frame uncertainty. On a benchmark with five fixed evaluation splits built from 33 public ground-truth datasets, SpikeSSL achieves state-of-the-art performance in both in-domain and zero-shot leave-one-indicator-out settings. We also develop a biophysical simulation pipeline capable of generating paired fluorescence-spike traces with systematically varied kinetic parameters, spike statistics, response nonlinearities, baseline drift, and noise. Using this pipeline, we synthesize approximately 11,000 simulated traces. Augmenting training with these data effectively closes the cross-indicator domain gap and improves zero-shot generalization. Code is publicly available at https://github.com/detimage123/SpikeSSL.

## 1 Introduction

Two-photon calcium imaging is a widely used tool for recording neural population activity in vivo [4, 29]. In this modality the observed fluorescence trace is a noisy, nonlinear, and temporally filtered proxy whose relation to the underlying spike train depends heavily on the calcium indicator, optical noise, and cell-type biophysics [22, 16]. Spike deconvolution, the task of recovering the latent spike train from the observed calcium fluorescence, is therefore an essential computational step. Existing spike inference methods broadly fall into two paradigms. Model-based approaches such as OASIS [5] and MLSpike [3] invert explicit generative models of calcium dynamics, but they require indicator-specific kinetic parameters whose optimal values may diverge from measured quantities. Deep learning-based methods such as CASCADE [16] and ENS<sup>2</sup> [28] learn the inverse mapping from paired fluorescence-electrophysiology recordings and achieve stronger in-domain performance.

Nevertheless, universal spike inference remains difficult, as models trained on observed indicators can degrade markedly on unseen ones. The central difficulty is that modern calcium indicators induce substantially different fluorescence dynamics. The underlying forward model relating calcium dynamics to fluorescence, parameterized by indicator-specific rise and decay time constants among other factors [23, 3, 2, 26], varies dramatically across widely used sensors. Figure 1(a,b) illustrates both the differences between different indicators and the data variation within the same indicator. Figure 1(c) further shows that the domain gap between indicator families is large in per-window feature space. At the same time, existing CNN-based supervised models treat spike inference as a generic temporal regression task without inductive bias tailored to calcium dynamics. The forward model itself motivates a structurally matched design, as the first-order linear recurrence governing calcium decay shares the mathematical form of scan operations in deep state-space layers [8, 7, 21], with the state-decay coefficient serving as a direct analogue of the indicator time constant. If the network’s temporal backbone is built from such recurrences, cross-indicator generalization becomes a more structured problem.

![](images/6976b4f1c997ddecebedf70e0ca1efbdaa7abfea123ea53d18115b208320446f.jpg)

![](images/3c863a7fbf4c6e0abba6c16b067010f504776dd1e6083775b3a7a0d8ec4deda0.jpg)

![](images/5a277cd586609f996f3be09d6cfbddd2b902fea477bc6ee36614a3caaff424f3.jpg)  
Figure 1: Indicator diversity and coverage gaps. (a) Representative fluorescence and spike snippets show markedly different response shapes across calcium indicators and among related GCaMP variants. (b) Indicator medians separate in kinetic and signal-statistic space, while simulated traces populate sparse regions between real datasets. (c) A t-SNE projection of per-window features shows partially overlapping but systematically displaced indicator domains.

Informed by this observation, we propose SpikeSSL, a universal spike inference framework built on dynamics-informed state-space layers. The temporal backbone consists of bidirectional IIR scan layers spanning a basis of decay rates from fast transient detection to slow baseline tracking. A multimodal conditioning encoder maps indicator identity, sampling rate, and trace-level signal statistics into a global conditioning vector, which modulates the backbone via Adaptive Layer Normalization as a built-in inductive bias rather than a post-hoc adaptation. In addition, a heteroscedastic variance head provides calibrated per-frame uncertainty and yields wider confidence intervals for out-ofdistribution predictions. Although this architecture substantially improves universal inference, the scarcity of paired real recordings still limits cross-indicator generalization. We therefore design a biophysical simulation pipeline that samples spike trains from stochastic point processes and neuron model dynamics, then passes them through a calcium forward model with varied kinetics, response nonlinearities, depletion, baseline drift, and observation noise. The resulting synthetic traces augment training and successfully bridge the domain gap between real indicator datasets, as illustrated in Figure 1(c).

Our main contributions are as follows.

• We propose SpikeSSL, a universal spike inference framework whose bidirectional IIR statespace layers are broadly motivated by calcium indicator decay dynamics. A multi-modal conditioning encoder maps indicator identity, sampling rate, and trace-level signal statistics into a global conditioning vector to modulate the backbone, together with a heteroscedastic variance head for calibrated uncertainty estimation.

• On a benchmark with five fixed evaluation splits built from 33 public ground-truth datasets, SpikeSSL achieves state-of-the-art performance in both in-domain and zero-shot LOIO settings, establishing its universal capability across diverse experimental conditions.

• We develop a biophysical simulation pipeline that generates synthetic calcium traces with systematically varied kinetic parameters and firing statistics. We synthesize approximately 11,000 simulated traces covering a wide range of indicator kinetic regimes, effectively closing the cross-indicator gap in zero-shot transfer.

## 2 Related Work

## 2.1 Spike Inference from Calcium Imaging

Spike inference methods divide into model-based and data-driven approaches. Model-based approaches invert explicit calcium dynamics models, including linear deconvolution [25], Bayesian approaches [23, 24, 14], the biophysical MLSpike model [3], and the OASIS active-set method [5]. These methods require indicator-dependent kinetic parameters whose optimal values often diverge from biophysically measured quantities [18]. Data-driven methods, from early SVM approaches [19] to the deep architectures advanced through the Spikefinder benchmark [1], have converged on end-to-end learned representations. CASCADE [16] provides noise-optimized training across 27 datasets and expands coverage via calibrated noise augmentation, but does not vary kinetic parameters across indicators. $\mathrm { E N S } ^ { 2 }$ [28] offers a compact 1D U-Net that achieves competitive accuracy with fewer parameters. Existing supervised methods rely on largely generic temporal backbones, and their performance degrades on unseen indicators. The Spikefinder benchmark [1] underscored this pattern, showing that improvements on one indicator rarely transferred to others, and coverage of the CASCADE database [16] remains uneven, with GCaMP8 variants and red-shifted sensors [26, 18] underrepresented relative to established GECIs, leaving models vulnerable to silent degradation on newly developed indicators.

## 2.2 Cross-Domain Generalization

Cross-domain generalization studies how models trained on multiple source domains transfer to unseen targets without target-domain labels [12, 9, 27]. In spike inference, domains are naturally defined by calcium indicators, and the challenge is accurate inference on held-out indicators with kinetics and statistics unseen during training. This problem is exacerbated by the limited diversity of paired ground-truth recordings. Although the database assembled by Rupprecht et al. [16] broadened coverage across indicators and experimental settings, and has since been extended with recordings for newly developed indicators and additional tissue types [18, 17], important gaps remain in indicator kinetic space. Recent work shows that CASCADE trained on one domain, such as cortex or GCaMP6- dominated data, transfers suboptimally to spinal cord recordings or GCaMP8 indicators [17, 18], highlighting the need for stronger cross-indicator generalization. Simulation-based augmentation offers a complementary strategy, where a calibrated forward model can generate paired fluorescence– spike traces spanning a systematic grid of kinetic parameters, extending the training distribution into regimes where real recordings are absent. This direction has received limited exploration in spike inference despite its potential to bridge coverage gaps without additional electrophysiology experiments.

## 3 Methodology

In this section, we present SpikeSSL, a universal spike inference framework with dynamics-informed state-space layers that integrates a bidirectional IIR backbone, multi-modal conditioning, and output heads for spike rate and uncertainty estimation. We then describe the biophysical simulation pipeline used for cross-indicator data augmentation.

## 3.1 Overall Architecture

SpikeSSL is a dynamics-informed network for spike inference whose architecture draws broad motivation from the biophysics of calcium indicators. The model takes a univariate fluorescence trace $y \in \mathbb { R } ^ { T }$ together with indicator identity and sampling rate, and produces a per-frame spike rate $\hat { s } \in \mathbb R _ { > 0 } ^ { T }$ along with calibrated uncertainty $\mathsf { \bar { \sigma } } ^ { 2 } \in \mathbb { R } _ { > 0 } ^ { T } .$ . The trace is first projected to model dimension D via a $1 \times 1$ convolution, while a conditioning encoder maps indicator identity, sampling rate, and trace-level signal statistics through dedicated branches into a global conditioning vector $\bar { c } \in \mathbb { R } ^ { 6 4 }$

![](images/5556fb47df1ea270161842301bb0aeffb1b14324f1726334ccbeb78a4d0c369f.jpg)  
Figure 2: Architecture of SpikeSSL. A fluorescence trace is projected by a pointwise convolution and processed by a stack of SpikeSSL blocks. Each block contains a bidirectional IIR scan with learnable decay rates that capture the exponential decay dynamics of calcium indicators, modulated by AdaLN conditioning on indicator identity, sampling rate, and trace-level signal statistics, followed by a Temporal Mixer for local temporal mixing and a SwiGLU FFN for channel mixing. A dilated refinement module aggregates multi-scale temporal context. Three parallel output heads produce a spike rate estimate, a binary spiking gate, and a per-frame uncertainty estimate.

The projected trace then passes through N stacked blocks, each containing a bidirectional IIR scan with AdaLN conditioning, a Temporal Mixer, and a SwiGLU feed-forward network (FFN). After the stack, a dilated refinement module of three residual convolutional layers with dilation schedule 1, 2, and 4 expands the receptive field from local to medium-range context, complementing the long-range multi-scale decay basis encoded by the IIR layers.

Three output heads operate on the refined features. The mean head predicts the non-negative spike rate $\hat { \mu } [ t ]$ via a convolutional projection followed by ReLU. A sigmoid gate head $g [ t ] \stackrel { \smile } { = } \sigma ( w _ { q } ^ { \top } \grave { h } [ t ] )$ classifies each frame as spiking or silent, is detached from the mean head’s gradient during training, and is trained with a separate binary cross-entropy loss, preventing co-adaptation between the two heads. At inference, $\hat { s } [ t ] = \hat { \mu } [ t ]$ if $g [ t ] > 0 . 5$ and $\hat { s } [ t ] = 0$ otherwise, eliminating low-amplitude noise in silent periods. A variance head produces per-frame uncertainty estimates $\tilde { \sigma ^ { 2 } } [ t ]$ as described in Section 3.4.

## 3.2 Dynamics-Informed IIR Backbone

The central design principle of SpikeSSL is to ground the network’s temporal processing in the dynamics of calcium indicators. Simple first-order exponential decay is among the most common and general models of calcium dynamics, and although the underlying biophysics can be substantially more complex, this recurrence still provides a useful inductive bias. The IIR scan used in deep state-space layers [8, 7, 21, 10, 6, 15] follows a structurally similar recurrence,

$$
h [ t ] = \alpha \cdot h [ t - 1 ] + ( 1 - \alpha ) \cdot z [ t ] ,\tag{1}
$$

where $\alpha = \sigma ( \alpha _ { \mathrm { l o g i t } } )$ is a per-channel decay rate parameterized via the sigmoid function, and z is the input projected from the trace embedding. The resemblance to exponential calcium decay motivates initializing and conditioning the decay rates to span the physiologically relevant range. By initializing the base logits as $\alpha _ { \mathrm { l o g i t } } \sim \mathrm { l i n s p a c e } ( 0 , 4 )$ , the model spans a multi-scale bank of decay rates from fast time constants of a few frames to slow trends unfolding over hundreds of frames. This range deliberately covers the full spread of indicator kinetics observed in the benchmark, providing a structured inductive bias that reflects the underlying biophysics rather than leaving the network to discover relevant temporal scales from data statistics alone.

To capture both causal context from past fluorescence and anti-causal context from future decay shape, SpikeSSL employs bidirectional IIR scans. Each layer computes a forward scan $h ^ { \mathrm { f w d } }$ and a time-reversed backward scan $h ^ { \mathrm { b w d } }$ , concatenates them along the channel dimension, and projects

back to the model dimension,

$$
h = W _ { \mathrm { o u t } } \left[ h ^ { \mathrm { f w d } } \parallel h ^ { \mathrm { b w d } } \right] + b _ { \mathrm { o u t } } .\tag{2}
$$

## 3.3 Multi-Modal Conditioning

Different indicators exhibit different kinetics, and SpikeSSL must adapt its decay rates accordingly. We achieve this through a multi-modal conditioning encoder, Adaptive Layer Normalization (AdaLN) [13], and per-sample decay rate modulation.

The conditioning vector $c \in \mathbb { R } ^ { d _ { c } }$ fuses three sources of information,

$$
c = \mathrm { M L P } \big ( e _ { \mathrm { i n d } } \ | | \ f _ { \mathrm { f s } } \ | | \ f _ { \mathrm { s t a t s } } \big ) ,\tag{3}
$$

where $e _ { \mathrm { i n d } }$ is a learned embedding of the discrete indicator identity, $f _ { \mathrm { f s } } = \mathrm { M L P } ( \log f _ { s } )$ encodes the sampling rate, and $f _ { \mathrm { s t a t s } }$ encodes trace-level signal statistics. Each source is independently projected to a shared dimension before concatenation and fusion.

The signal statistics component extracts a 7-dimensional summary vector from the input trace,

$$
f _ { \mathrm { s t a t s } } = \left[ \hat { \sigma } _ { y } , \ \hat { \gamma } _ { y } , \ \hat { \kappa } _ { y } , \ \rho _ { 1 } , \ \rho _ { 3 } , \ \rho _ { 5 } , \ \rho _ { 1 0 } \right] ,\tag{4}
$$

comprising the standard deviation, skewness, excess kurtosis, and normalized autocorrelation at lags 1, 3, 5, and 10. These statistics are computed deterministically with no learnable parameters and serve as an implicit proxy for the indicator’s SNR and kinetic properties. Slowly decaying indicators produce high autocorrelation, while noisy or fast indicators exhibit low autocorrelation and high kurtosis. This module enables SpikeSSL to adapt its temporal dynamics even when the indicator type is not specified.

The conditioning vector modulates the IIR sub-block via AdaLN with zero-initialized projection,

$$
{ \mathrm { A d a L N } } ( x , c ) = \operatorname { L N } ( x ) \odot ( 1 + \gamma ( c ) ) + \beta ( c ) ,\tag{5}
$$

where $( \gamma , \beta ) = W _ { c } c$ is a linear projection initialized to zero, so that at initialization the model reduces to standard Layer Normalization. AdaLN is applied only to the temporal IIR sub-block, while the subsequent convolution and feed-forward sub-blocks use standard normalization. This concentrates the indicator-conditioned modulation on the component that directly encodes decay dynamics.

In addition to AdaLN, the conditioning vector directly shifts the IIR decay logits,

$$
\alpha _ { \mathrm { e f f } } = \sigma \big ( \alpha _ { \mathrm { l o g i t } } + W _ { \alpha } c \big ) ,\tag{6}
$$

where $W _ { \alpha }$ is zero-initialized. This allows each sample’s effective decay rates to be continuously modulated by its indicator identity and signal properties, enhancing the model’s ability to adapt to diverse kinetic regimes.

## 3.4 Heteroscedastic Uncertainty Estimation

SpikeSSL produces a per-frame variance estimate $\hat { \sigma } ^ { 2 } [ t ]$ alongside the spike rate prediction. The variance head consists of a single convolutional projection from the shared backbone features plus a per-indicator learned bias,

$$
\log \hat { \sigma } ^ { 2 } [ t ] = w _ { v } ^ { \top } h [ t ] + b _ { \mathrm { i n d } } ,\tag{7}
$$

where $b _ { \mathrm { i n d } }$ is an indicator-specific scalar bias. Both the projection and bias are zero-initialized. A key emergent property is that the variance output serves as an automatic out-of-distribution detector, as in-domain predictions yield tight confidence intervals while zero-shot predictions on unseen indicators produce inflated variance.

## 3.5 Biophysical Simulation Data Generation

To populate underrepresented regimes while preserving realistic action-potential-driven fluorescence, we augment training with simulated calcium traces generated by an enhanced MLSpike-based forward model [3]. The simulator complements real paired recordings by expanding kinetic coverage rather than replacing empirical data, and samples parameters across ranges covering observed indicators and plausible unseen sensors. As illustrated in Fig. 3, the simulation pipeline contains three components: spike train synthesis, calcium transient generation, and observation noise modeling. For each trial, we sample a recording duration and target firing rate, then generate spike trains with two complementary strategies. One strategy draws spike counts and inter-spike intervals from stochastic point-process models, allowing sparse, irregular, quasi-periodic, and burst-like firing. The other integrates the Izhikevich neuron model [11] to produce spikes from continuous membrane potential dynamics. Together, these strategies provide a broad repertoire of neuronal activity regimes before the fluorescence forward model is applied.

![](images/8f9caa49fcbae3f335d0ffc1cbc08242daad27cfa5e5e8f3fdde839b17190fc9.jpg)  
Figure 3: Hierarchical Workflow of the Calcium Data Simulator.

The generated spikes are then passed through a modified calcium simulator. Compared with standard MLSpike, our simulator introduces stochastic amplitude jitter, a recovery variable that captures depletion and interval dependent response scaling, and indicator specific cooperative binding kinetics. Finally, we form observable fluorescence by adding slow baseline drift and Gaussian measurement noise to the modeled calcium response, producing traces with variability in baseline stability and signal quality. These mechanisms broaden the range of firing patterns, response amplitudes, noise conditions, and indicator dynamics beyond the real training set alone, helping bridge held out kinetic regimes during zero shot transfer. More details are provided in the supplementary material.

## 3.6 Training Objective

The total loss combines three components, namely a heteroscedastic reconstruction loss, structural penalty terms, and a gate detection loss,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { s t r u c t } } + \lambda _ { g } \mathcal { L } _ { \mathrm { g a t e } } . } \end{array}\tag{8}
$$

The reconstruction loss follows the β-NLL formulation [20], which precision-weights the per-frame Huber loss by the predicted variance,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c o n } } = \displaystyle \frac { 1 } { \hat { \sigma } ^ { 2 } } \cdot \mathrm { H u b e r } _ { \delta } ( \hat { s } , s ) + \frac { 1 } { 2 } \log \hat { \sigma } ^ { 2 } . } \end{array}\tag{9}
$$

High-confidence frames receive amplified gradients while noisy frames are down-weighted, and the entropy term ${ \textstyle \frac { 1 } { 2 } } \log { \hat { \sigma } ^ { 2 } }$ prevents the degenerate solution $\hat { \sigma } ^ { 2 } \to \bar { \infty }$

The structural penalties impose three asymmetric constraints that are not attenuated by the variance estimate, namely an over-prediction penalty that penalizes $\hat { s } [ t ] > s [ t ]$ more heavily than underprediction, a peak under-prediction penalty that pulls up missed high-amplitude spike events, and a mean bias penalty that prevents systematic offset between predicted and true mean spike rates,

$$
{ \mathcal { L } } _ { \mathrm { s t r u c t } } = \beta _ { \mathrm { o v e r } } \cdot { \mathcal { L } } _ { \mathrm { o v e r } } + \beta _ { \mathrm { p e a k } } \cdot { \mathcal { L } } _ { \mathrm { p e a k } } + \lambda _ { \mu } \cdot { \mathcal { L } } _ { \mathrm { b i a s } } .\tag{10}
$$

The gate loss trains the binary spike-or-silent classifier via cross-entropy, $\mathcal { L } _ { \mathrm { g a t e } } = \mathrm { B C E } ( g [ t ] , \mathbf { 1 } [ s [ t ] >$ ϵ]), with gradients detached from the mean head to prevent co-adaptation.

Table 1: In-domain resluts. All methods are trained and evaluated on the full ground-truth database with per-dataset train and test splits. Bold / Underline indicate the 1st / 2nd best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">V1-OGB</td><td colspan="3">V1-GCaMP6s</td><td colspan="3">V1-GCaMP8</td><td colspan="3">Other-RCaMP</td><td colspan="3">SC-GCaMP6s</td><td rowspan="2">Mean Corr ↑</td><td rowspan="2">Params</td></tr><tr><td>Corr ↑</td><td>Err ↓</td><td>Bias ↓</td><td>Corr ↑</td><td>Err ↓</td><td></td><td>Bias↓ Corr↑</td><td>Err ↓</td><td>Bias ↓</td><td>Corr↑</td><td>Err↓</td><td>Bias↓</td><td>Corr ↑</td><td>Err↓</td><td>Bias ↓</td></tr><tr><td>OASIS</td><td>0.298</td><td>1.157</td><td>-0.481</td><td>0.332</td><td>0.973</td><td>-0.883</td><td>0.493</td><td>0.878</td><td>-0.659</td><td>0.504</td><td>1.054</td><td>-0.508</td><td>0.365</td><td>0.975</td><td>-0.964</td><td>0.398</td><td></td></tr><tr><td>MLSpike</td><td>0.363</td><td>4.379</td><td>3.244</td><td>0.224</td><td>1.017</td><td>-0.928</td><td>0.558</td><td>0.935</td><td>-0.887</td><td>0.354</td><td>1.369</td><td>-0.761</td><td>0.307</td><td>1.005</td><td>-0.919</td><td>0.361</td><td></td></tr><tr><td>CASCADE [16]</td><td>0.613</td><td>1.117</td><td>-0.250</td><td>0.361</td><td>1.718</td><td>0.520</td><td>0.825</td><td>3.246</td><td>3.108</td><td>0.675</td><td>1.060</td><td>-0.388</td><td>0.149</td><td>1.005</td><td>-0.639</td><td>0.524</td><td>34K</td></tr><tr><td>ENS2 [28]</td><td>0.641</td><td>1.032</td><td>-0.368</td><td>0.358</td><td>2.265</td><td>1.354</td><td>0.775</td><td>1.827</td><td>1.596</td><td>0.695</td><td>0.884</td><td>-0.509</td><td>0.256</td><td>1.116</td><td>-0.154</td><td>0.545</td><td>145K</td></tr><tr><td>SpikeSSL</td><td>0.743</td><td>0.810</td><td>-0.067</td><td>0.579</td><td>0.945</td><td>-0.741</td><td>0.915</td><td>0.565</td><td>0.017</td><td>0.731</td><td>1.269</td><td>0.234</td><td>0.736</td><td>0.764</td><td>-0.704</td><td>0.741</td><td>3.9M</td></tr></table>

Table 2: Zero-shot LOIO results. Each column group holds out one of the five evaluation splits during training and evaluates on it without target-domain data. Bold / Underline indicate the 1st / 2nd best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">V1-OGB</td><td colspan="3">V1-GCaMP6s</td><td colspan="3">V1-GCaMP8</td><td colspan="3">Other-RCaMP</td><td colspan="3">SC-GCaMP6s</td><td>Mean</td></tr><tr><td>Corr↑</td><td>Err ↓</td><td>Bias ↓</td><td>Corr↑</td><td>Err ↓</td><td>Bias↓</td><td>Corr↑</td><td>Err↓</td><td>Bias ↓</td><td>Corr↑</td><td>Err↓</td><td>Bias↓</td><td>Corr↑</td><td>Err ↓</td><td>Bias↓</td><td>Corr ↑</td></tr><tr><td>OASIS</td><td>0.298</td><td>1.157</td><td>-0.481</td><td>0.332</td><td>0.973</td><td>-0.883</td><td>0.493</td><td>0.878</td><td>-0.659</td><td>0.504</td><td>1.054</td><td>-0.508</td><td>0.365</td><td>0.975</td><td>-0.964</td><td>0.398</td></tr><tr><td>MLSpike</td><td>0.363</td><td>4.379</td><td>3.244</td><td>0.224</td><td>1.017</td><td>-0.928</td><td>0.558</td><td>0.935</td><td>-0.887</td><td>0.354</td><td>1.369</td><td>-0.761</td><td>0.307</td><td>1.005</td><td>-0.919</td><td>0.361</td></tr><tr><td>CASCADE</td><td>0.560</td><td>1.208</td><td>-0.144</td><td>0.304</td><td>2.571</td><td>1.614</td><td>0.757</td><td>3.723</td><td>3.602</td><td>0.551</td><td>1.086</td><td>-0.300</td><td>0.126</td><td>1.083</td><td>-0.442</td><td>0.460</td></tr><tr><td>ENS2</td><td>0.574</td><td>1.005</td><td>-0.366</td><td>0.310</td><td>3.401</td><td>2.654</td><td>0.716</td><td>2.038</td><td>1.715</td><td>0.635</td><td>0.966</td><td>-0.454</td><td>0.178</td><td>1.092</td><td>-0.391</td><td>0.483</td></tr><tr><td>SpikeSSL</td><td>0.650</td><td>1.979</td><td>0.675</td><td>0.509</td><td>0.944</td><td>-0.677</td><td>0.853</td><td>0.513</td><td>-0.091</td><td>0.717</td><td>1.172</td><td>0.606</td><td>0.542</td><td>0.911</td><td>-0.892</td><td>0.654</td></tr></table>

Table 3: LOIO results with simulation augmentation (“+ Sim.”). Training is augmented with biophysically simulated calcium traces. Bold / Underline indicate the 1st / 2nd best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">V1-OGB</td><td colspan="3">V1-GCaMP6s</td><td colspan="3">V1-GCaMP8</td><td colspan="3">Other-RCaMP</td><td colspan="3">SC-GCaMP6s</td><td>Mean</td></tr><tr><td>Corr↑</td><td>Err↓</td><td>Bias ↓</td><td>Corr↑</td><td>Err↓</td><td>Bias ↓</td><td>Corr↑</td><td>Err↓</td><td>Bias↓</td><td>Corr↑</td><td>Err↓</td><td>Bias ↓</td><td>Corr ↑</td><td>Err↓</td><td>Bias↓</td><td>Corr ↑</td></tr><tr><td>CASCADE + Sim.</td><td>0.552</td><td>2.418</td><td>1.423</td><td>0.439</td><td>1.522</td><td>0.691</td><td>0.716</td><td>1.408</td><td>1.018</td><td>0.642</td><td>1.421</td><td>0.614</td><td>0.405</td><td>3.206</td><td>2.084</td><td>0.551</td></tr><tr><td>ENS2 + Sim.</td><td>0.486</td><td>1.634</td><td>0.592</td><td>0.447</td><td>2.146</td><td>1.418</td><td>0.796</td><td>1.882</td><td>1.487</td><td>0.588</td><td>1.764</td><td>0.756</td><td>0.461</td><td>2.291</td><td>1.054</td><td>0.556</td></tr><tr><td>SpikeSSL + Sim.</td><td>0.697</td><td>1.441</td><td>0.419</td><td>0.533</td><td>0.931</td><td>-0.697</td><td>0.902</td><td>0.523</td><td>0.175</td><td>0.708</td><td>0.849</td><td>-0.505</td><td>0.678</td><td>0.771</td><td>-0.653</td><td>0.704</td></tr></table>

## 4 Experiments

## 4.1 Datasets

We evaluate on the ground-truth database assembled by Rupprecht et al. [16], comprising 33 datasets with 1,960 paired neuron recordings spanning 11 calcium indicators, two species (mouse and zebrafish), multiple brain regions (V1, CA3, olfactory bulb, telencephalon) and spinal cord, with both excitatory neurons and inhibitory interneurons (PV, SST, VIP). The corpus covers chemical dyes (OGB-1, Cal-520), legacy GECIs (GCaMP5k, 6f, 6s, 7f [2]), next-generation GCaMP8 variants (8f, 8m, 8s [26]), and red-shifted indicators (R-CaMP, jRCaMP1a, jGECO1a). Recent spinal cord recordings [17] introduce a major tissue-type shift from conventional cortical preparations. We additionally augment training with biophysically simulated calcium traces stored in multiple generated batches (Section 3.5 and Appendix A.4) used exclusively during training.

## 4.2 Benchmark and Metrics

We compare SpikeSSL against four baselines spanning the major paradigms, namely OASIS [5], MLSpike [3], CASCADE [16], and ENS<sup>2</sup> [28]. The underlying recordings are publicly available, and we evaluate them under five fixed evaluation splits. In the in-domain setting, all splits are included during training with per-dataset train and test subsets. In the LOIO setting, five representative experimental conditions, namely V1-OGB, V1-GCaMP6s, V1-GCaMP8, Other-RCaMP, and SC-GCaMP6s, are each held out in turn and evaluated without any target-domain training data. Across this benchmark, indicator decay time constants span from ∼50 ms for GCaMP8f to ∼600 ms for GCaMP6s, creating fundamentally different inverse problems for each condition.

We evaluate all methods using Pearson correlation between the predicted and ground-truth spike rate [22, 16, 28], computed per neuron and averaged across neurons within each dataset. We additionally report normalized error and mean bias (signed difference in temporal means) to assess systematic over- or under-prediction.

![](images/959889292c55d7fdeaea57475b7737914699ad49ea23ba246e25581ca25fcdaa.jpg)  
Figure 4: Qualitative LOIO comparison on GCaMP8s, GCaMP6s, and OGB-1 test neurons. Each column shows a different method, and the grey curve denotes the ground-truth smoothed spike rate.

Table 4: Ablation study under LOIO. Mean correlation is averaged over all five held-out evaluation splits.

<table><tr><td>Method</td><td>Mean Corr (↑)</td></tr><tr><td>V1 (− AdaLN)</td><td>0.601</td></tr><tr><td>V2 (− Stats)</td><td>0.623</td></tr><tr><td>V3 (− Bi-IIR)</td><td>0.594</td></tr><tr><td>V4 (− Var.)</td><td>0.608</td></tr><tr><td>V5 (− Gate)</td><td>0.621</td></tr><tr><td>SpikeSSL</td><td>0.654</td></tr></table>

![](images/7c38f39581f8247e44990e97add9c68edd6179a2ea9f17af314508775126a0c5.jpg)  
(a)

![](images/037b23505f19001c565afcbd4ed184416f21f97d0900d3048543454dc415f998.jpg)  
(b)

![](images/20604f3a76c5361df4e6e7c03a2999601bdd9d560e5b5cdd6e779499a88c3c0b.jpg)  
(c)  
Figure 5: Uncertainty calibration of SpikeSSL. Panel (a) shows reliability curves. Panel (b) compares variance distributions. Panel (c) plots std dev against absolute error.

## 4.3 Implementation Details

SpikeSSL uses 8 stacked blocks with model dimension D=192 and conditioning dimension $d _ { c } { = } 6 4 .$ We train with Adam using an initial learning rate of $1 0 ^ { - 3 }$ . The model converges within 20 epochs. For the loss hyperparameters, we set the over-prediction penalty $\beta _ { \mathrm { o v e r } } { = } 3 . 0$ , peak under-prediction weight $\beta _ { \mathrm { u n d e r } } { = } 0 . 3$ , Huber threshold δ=0.1, and mean bias weight $\lambda _ { \mu } { = } 0 . 5$ . The gate loss weight $\lambda _ { g } \mathrm { = } 5 . 0$ is set deliberately large to suppress false-positive spike predictions in silent periods.

## 4.4 In-Domain Results

Table 1 reports in-domain performance with all evaluation splits included during training. SpikeSSL achieves state-of-the-art performance, substantially outperforming all baselines. These results demonstrate that the dynamics-informed IIR backbone and multi-modal conditioning provide strong inductive bias even in the fully supervised setting where all indicator types are observed during training. The consistently lower error and bias values confirm that SpikeSSL produces well-calibrated spike rate estimates across the full diversity of indicators.

## 4.5 Zero-Shot LOIO Results

Under the LOIO protocol defined in Section 4.2, cross-indicator generalization remains difficult for all methods, but SpikeSSL achieves the highest mean correlation across all five held-out evaluation splits in Table 2. This result indicates that the IIR backbone and indicator-aware conditioning provide a meaningful structural advantage under domain shift. When training is augmented with biophysically simulated calcium traces, zero-shot performance improves further across all held-out splits in Table 3, with particularly large gains on conditions whose kinetics are sparsely covered by the real training set. Figure 4 shows that SpikeSSL preserves sharper spike structure than the classical baselines on representative unseen indicators.

Table 5: Param-matched control. Scaling fails to close the gap.
<table><tr><td>Method Mean Corr (↑) #Params</td></tr><tr><td>CASCADE-L 0.554</td></tr><tr><td>0.4M CASCADE-H 0.533 4.6M</td></tr><tr><td>ENS2-L 0.576 2.1M</td></tr><tr><td>ENS²-H 0.564 4.2M</td></tr><tr><td>SpikeSSL-L 0.649 7.2M</td></tr><tr><td>SpikeSSL 0.654 3.9M</td></tr></table>

![](images/95001c218572083ea6f714cbaec9029af3dd71686ca12d60f5b0af8dab00e399.jpg)  
Figure 6: Linearity and activity-regime analysis. First three: deviation from linearity; Last: normalized error in flat, sparse, and dense windows.

## 4.6 Ablation Studies

We perform ablations under the LOIO protocol and report mean correlation across all five held-out splits in Table 4. V1 replaces AdaLN with a concatenation-based conditioning scheme, and V2 removes the signal-statistics branch from the conditioning vector. Together they confirm that both conditioning mechanisms contribute to cross-indicator adaptation. V3 replaces the bidirectional IIR module with a standard temporal convolutional block, supporting the value of the dynamics-informed multi-scale backbone. V4 removes the heteroscedastic variance head and β-NLL reweighting, while V5 removes the binary gate head and its cross-entropy loss, confirming that both heads improve predictive quality beyond their primary roles in uncertainty calibration and spike denoising.

## 4.7 Discussion

Uncertainty Calibration SpikeSSL’s heteroscedastic variance head provides calibrated per-frame uncertainty in addition to point estimates. Figure 5 shows that confidence intervals remain close to empirical coverage, that predicted variance increases under LOIO shift, and that higher predicted standard deviation is associated with larger absolute error. Taken together, these observations suggest that the variance head captures meaningful epistemic uncertainty rather than acting as a generic smoothing term. At the same time, this analysis is limited to the five-split LOIO benchmark considered here, so whether calibration remains equally reliable under more extreme acquisition shifts, such as substantially different noise statistics or sampling regimes, remains an open question.

Architectural Gains Beyond Parameter Scaling Table 5 is intended to test whether SpikeSSL’s gains can be explained simply by model size. We scale CASCADE and ENS<sup>2</sup> to larger parameter budgets and also evaluate a larger SpikeSSL variant. Increasing the baseline parameter counts indeed improves their performance, but they still do not reach SpikeSSL. In contrast, further scaling SpikeSSL does not yield additional gains, suggesting that its advantage comes from the dynamicsinformed IIR backbone, indicator-aware conditioning, and uncertainty-aware objective rather than from simply stacking more parameters.

Linearity and Response Scale of Spike Rate Estimation A useful spike inference model should keep predicted firing rates close to a linear, correctly scaled transform of the ground truth. Figure 6(a) summarizes this property on OGB-1, GCaMP6f, and GCaMP8, where SpikeSSL shows the smallest deviation from linearity across distinct kinetic regimes. Figure 6(b) further stratifies local 1-s windows into flat, sparse, and dense activity regimes, showing lower error for SpikeSSL during both silent and active periods. These results are consistent with the multi-scale IIR backbone, whose fast and slow channels help preserve response scale across indicator kinetics.

## 5 Conclusion

We presented SpikeSSL, a universal spike inference framework that narrows the cross-indicator generalization gap through two complementary mechanisms. A bidirectional IIR backbone with a multi-scale decay basis spanning the full range of physiologically relevant indicator time constants, combined with multi-modal conditioning, provides a structured inductive bias for the inverse problem. Simulation-augmented training further bridges gaps in the kinetic parameter space of unseen indicators. On five fixed evaluation splits built from public ground-truth datasets, SpikeSSL achieves state-of-the-art performance in both in-domain and zero-shot settings without target-domain ground truth. By providing a single model that generalizes across indicator types without retraining,

SpikeSSL can reduce the experimental burden of ground-truth data collection and facilitate the adoption of new calcium indicators in neuroscience laboratories.

## References

[1] Philipp Berens, Jeremy Freeman, Thomas Deneux, Nikolay Chenkov, Thomas McColgan, Artur Speiser, Jakob H Macke, Srinivas C Turaga, Patrick Mineault, Peter Rupprecht, et al. Community-based benchmark ing improves spike rate inference from two-photon calcium imaging data. PLoS Computational Biology, 14(5):e1006157, 2018.

[2] Tsai-Wen Chen, Trevor J Wardill, Yi Sun, Stefan R Pulver, Sabine L Renninger, Amy Baohan, Eric R Schreiter, Rex A Kerr, Michael B Orger, Vivek Jayaraman, et al. Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature, 499(7458):295–300, 2013.

[3] Thomas Deneux, Attila Kaszas, Gergely Szalay, Gergely Katona, Tamás Lakner, Amiram Grinvald, Balázs Rózsa, and Ivo Vanzetta. Accurate spike estimation from noisy calcium signals for ultrafast three-dimensional imaging of large neuronal populations in vivo. Nature Communications, 7(1):12190, 2016.

[4] Winfried Denk, James H Strickler, and Watt W Webb. Two-photon laser scanning fluorescence microscopy. Science, 248(4951):73–76, 1990.

[5] Johannes Friedrich, Pengcheng Zhou, and Liam Paninski. Fast online deconvolution of calcium imaging data. PLoS Computational Biology, 13(3):e1005423, 2017.

[6] Daniel Y Fu, Tri Dao, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos: Towards language modeling with state space models. In International Conference on Learning Representations, 2023.

[7] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2024.

[8] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022.

[9] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2021.

[10] Ankit Gupta, Albert Gu, and Jonathan Berant. Diagonal state spaces are as effective as structured state spaces. In Advances in Neural Information Processing Systems, volume 35, 2022.

[11] Eugene M. Izhikevich. Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6): 1569–1572, 2003.

[12] Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. Domain generalization via invariant feature representation. In International Conference on Machine Learning, pages 10–18, 2013.

[13] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[14] Eftychios A Pnevmatikakis, Josh Merel, Ari Pakman, and Liam Paninski. Bayesian spike inference from calcium imaging data. In 2013 Asilomar Conference on Signals, Systems and Computers, pages 349–353. IEEE, 2013.

[15] Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena hierarchy: Towards larger convolutional language models. In International Conference on Machine Learning, 2023.

[16] Peter Rupprecht, Stefano Carta, Adrian Hoffmann, Mayumi Echizen, Antonin Blot, Alex C Kwan, Yang Dan, Sonja B Hofer, Kazuo Kitamura, Fritjof Helmchen, and Rainer W Friedrich. A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. Nature Neuroscience, 24(9):1324–1337, 2021.

[17] Peter Rupprecht, Wei Fan, Steve J Sullivan, Fritjof Helmchen, and Andrei D Sdrulla. Spike rate inference from mouse spinal cord calcium imaging data. Journal of Neuroscience, 45(18), 2025.

[18] Peter Rupprecht, Márton Rózsa, Xusheng Fang, Karel Svoboda, and Fritjof Helmchen. Spike inference from calcium imaging data acquired with gcamp8 indicators. bioRxiv, pages 2025–03, 2025.

[19] Takuya Sasaki, Naoya Takahashi, Norio Matsuki, and Yuji Ikegaya. Fast and accurate detection of action potentials from somatic calcium fluctuations. Journal ofNeurophysiology, 100(3):1668–1676, 2008.

[20] Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, and Georg Martius. On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In International Conference on Learning Representations, 2022.

[21] Jimmy T.H. Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for sequence modeling. In International Conference on Learning Representations, 2023.

[22] Lucas Theis, Philipp Berens, Emmanouil Froudarakis, Jacob Reimer, Miroslav Román Rosón, Tom Baden, Thomas Euler, Andreas S Tolias, and Matthias Bethge. Benchmarking spike rate inference in population calcium imaging. Neuron, 90(3):471–482, 2016.

[23] Joshua T Vogelstein, Brendon O Watson, Adam M Packer, Rafael Yuste, Bruno Jedynak, and Liam Paninski. Spike inference from calcium imaging using sequential monte carlo methods. Biophysical Journal, 97(2): 636–655, 2009.

[24] Joshua T Vogelstein, Adam M Packer, Timothy A Machado, Tanya Sippy, Baktash Babadi, Rafael Yuste, and Liam Paninski. Fast nonnegative deconvolution for spike train inference from population calcium imaging. Journal ofNeurophysiology, 104(6):3691–3704, 2010.

[25] Emre Yaksi and Rainer W Friedrich. Reconstruction of firing rate changes across neuronal populations by temporally deconvolved Ca<sup>2+</sup> imaging. Nature Methods, 3(5):377–383, 2006.

[26] Yan Zhang, Márton Rózsa, Yajie Liang, Daniel Bushey, Ziqiang Wei, Jihong Zheng, Daniel Reep, Gerard Joey Broussard, Arthur Tsang, Getahun Tsegaye, et al. Fast and sensitive gcamp calcium indicators for imaging neural populations. Nature, 615(7954):884–891, 2023.

[27] Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):4396–4429, 2022.

[28] Zhanhong Zhou, Hei Matthew Yip, Katya Tsimring, Mriganka Sur, Jacque Pak Kan Ip, and Chung Tin. Effective and efficient neural networks for spike inference from in vivo calcium imaging. Cell Reports Methods, 3(5), 2023.

[29] Warren R. Zipfel, Rebecca M. Williams, and Watt W. Webb. Nonlinear magic: Multiphoton microscopy in the biosciences. Nature Biotechnology, 21(11):1369–1377, 2003.

## A Supplementary Material

This appendix consolidates the technical details and extended validation in one continuous supplement. Appendix A.1 expands the conditioning encoder, dilated refinement module, Temporal Mixer, and SwiGLU FFN. Appendix A.2 documents the recording sets and fixed evaluation splits, Appendix A.4 specifies the biophysical simulation pipeline, and Appendix A.5 details benchmark construction. The remaining subsections provide additional analyses of conditioning, domain breadth, data scaling, ablations, adaptation regimes, learned dynamics, simulation, and robustness.

## A.1 Supplementary architecture modules

This subsection expands three modules that are named in the main text but not visualized in Figure 2, namely the conditioning encoder, the dilated refinement module, and the Temporal Mixer and SwiGLU FFN sub-blocks within each SpikeSSL block.

![](images/f2370701a07213680d8dec6d4c53e7936836fc1091bcd9323ad3de50ac6e654a.jpg)  
Figure 7: Conditioning encoder architecture. Three modality-specific branches independently encode their inputs before fusing them into a common representation. The indicator embedding branch maps discrete indicator identity to a 64-dimensional learned vector, where ID 0 is reserved for unknown or unseen indicators. The sampling-rate branch encodes log $f _ { s }$ through a two-layer MLP. The signal-statistics branch encodes the 7-dimensional descriptor comprising standard deviation, skewness, excess kurtosis, and autocorrelation at lags 1, 3, 5, and 10 through a two-layer MLP. The three 64-dimensional outputs are concatenated and fused by a 192 → 64 → 64 MLP to produce the global conditioning vector $c \in \mathbb { R } ^ { 6 4 }$

## A.1.1 Conditioning encoder internals

The conditioning encoder maps three modality-specific inputs through dedicated branches into the global conditioning vector $c \in \mathbb { R } ^ { 6 4 }$ . The indicator branch uses a learned embedding table over 14 IDs, where ID 0 is reserved for unknown or unseen indicators and the remaining IDs cover named indicators including OGB-1, GCaMP6f, GCaMP6s, GCaMP8 variants, R-CaMP, jRCaMP1a, jGECO1a, and XCaMPgf, each projected to 64 dimensions by a linear layer. The sampling-rate branch maps log $f _ { s }$ through a two-layer MLP of hidden width 64, falling back to log 30 when the frame rate is unavailable. The signal-statistics branch maps a 7-dimensional descriptor comprising standard deviation, skewness, excess kurtosis, and autocorrelation at lags 1, 3, 5, and 10 through a separate two-layer MLP of width 64. The three 64-dimensional branch outputs are concatenated and fused by a final 192 → 64 → 64 MLP to produce $c ,$ keeping each modality isolated until the fusion stage so that indicator-specific metadata and data-driven statistics contribute through clearly separated pathways.

## A.1.2 Dilated refinement module

After the stack of conditioned SpikeSSL blocks and a final LayerNorm, the model applies a residual dilated convolution stack. With the default setting of three refinement layers, the dilation schedule is 1, 2, and 4, where each layer consists of a 1D convolution with kernel size 3 and matching dilationdependent padding, followed by batch normalization, GELU, dropout, and a residual addition, expanding the temporal receptive field without downsampling. Whereas the bidirectional IIR scan captures long-range indicator-specific decay, the refinement stack provides complementary shortto-medium-range local correction capacity at full temporal resolution, functioning as a lightweight post-backbone smoother and sharpener.

![](images/f9cfb945dd7dea40f09d57ea0cc5e3a670633354dd4882514a17cb63d9cc6658.jpg)  
Figure 8: (a) SpikeSSL block sub-modules. The Temporal Mixer applies a depthwise 1D convolution (kernel width 7) followed by a pointwise convolution, GELU activation, and a residual connection, providing local temporal mixing without indicator-conditioned normalization. The SwiGLU FFN projects to twice the channel width, splits into gate and value tensors, computes SiLU(gate) ⊙ value, and projects back to model dimension D, followed by a residual connection. (b) Dilated refinement module comprising three residual layers with dilation factors 1, 2, and 4 that successively expand the temporal receptive field at full resolution, each containing a kernel-3 1D convolution, batch normalization, GELU, and dropout.

## A.1.3 Temporal Mixer and SwiGLU FFN inside each block

Each SpikeSSL block contains three sequential sub-modules. The first is the AdaLN-conditioned bidirectional IIR layer described in the main text. The second and third are deliberately left unconditioned. The Temporal Mixer applies LayerNorm to the block state, followed by a depthwise 1D convolution of kernel width 7, a pointwise 1D convolution, GELU activation, dropout, and a residual addition, providing local temporal mixing without an additional indicator-conditioned normalization path.

The SwiGLU FFN applies LayerNorm and projects the state to twice the expanded width, splits the projection into gate and value tensors, computes SiLU(gate) ⊙ value, projects back to model dimension D, and adds the result residually. With the default configuration D=192 and expansion factor 2, the intermediate width is 384. AdaLN modulation is confined to the IIR branch, where indicator-specific decay kinetics have the clearest physical interpretation, while the Temporal Mixer and SwiGLU FFN serve as standard local and channel mixers.

## A.2 Benchmark data composition and split definitions

Table 6 provides an overview of all benchmark datasets. Specifically, Table 7 categorizes the ground-truth recording sets by indicator family. Furthermore, Table 8 lists the five fixed evaluation splits.

The recording-level inventory makes the source of benchmark heterogeneity explicit: the corpus combines chemical dyes and genetically encoded indicators across species, anatomical regions, induction methods, and sampling rates rather than treating an indicator as a single acquisition setup. Frame rates span approximately 2.5–500 Hz, and the recordings cover cortical, hippocampal, olfactory, and spinal preparations collected by multiple laboratories. Consequently, the benchmark tests variation in acquisition and biological context in addition to nominal sensor identity. Recordinglevel separation also prevents overlapping windows from the same acquisition from being divided between training and evaluation.

Table 6: Overview of all ground truth recording sets.
<table><tr><td>Set</td><td>Calcium indicator</td><td>Induction method</td><td>Animal model</td><td>Anatomical region</td><td>Frame rate (Hz)</td><td>N</td><td>Source paper</td></tr><tr><td>1</td><td>OGB-1</td><td>Acute injection</td><td>Mouse</td><td>V1</td><td>11.3</td><td>11</td><td>Theis et al., 2016</td></tr><tr><td>2</td><td>OGB-1</td><td>Injection + tg(tdTomato- CaMKIIα)</td><td>Mouse</td><td>V1</td><td>15.6</td><td>16</td><td>Kwan and Dan, 2012</td></tr><tr><td>3</td><td>Cal-520</td><td>Acute injection</td><td>Mouse</td><td>S1</td><td>500.0</td><td>8</td><td>Tada et al., 2014</td></tr><tr><td>4</td><td>OGB-1</td><td>Acute injection</td><td>Zebrafish</td><td>pDp</td><td>7.7</td><td>15</td><td>Rupprecht et al., 2021</td></tr><tr><td>5</td><td>Cal-520</td><td>Acute injection</td><td>Zebrafish</td><td>pDp</td><td>7.8</td><td>5</td><td>Rupprecht et al., 2021</td></tr><tr><td>6</td><td>GCaMP6f</td><td>tg(NeuroD)</td><td>Zebrafish</td><td>aDp</td><td>30.0</td><td>8</td><td>Rupprecht et al., 2021</td></tr><tr><td>7</td><td>GCaMP6f</td><td>tg(NeuroD)</td><td>Zebrafish</td><td>dD</td><td>30.0</td><td>10</td><td>Rupprecht et al., 2021</td></tr><tr><td>8</td><td>GCaMP6f</td><td>tg(NeuroD)</td><td>Zebrafish</td><td>OB</td><td>30.0</td><td>9</td><td>Rupprecht et al., 2021</td></tr><tr><td>9</td><td>GCaMP6f</td><td>AAV</td><td>Mouse</td><td>V1</td><td>60.1</td><td>11</td><td>Chen et al., 2013</td></tr><tr><td>10</td><td>GCaMP6f</td><td>tg(Emx1)</td><td>Mouse</td><td>V1</td><td>160.1</td><td>23</td><td>Huang et al., 2019</td></tr><tr><td>11</td><td>GCaMP6f</td><td>tg(Cux2)</td><td>Mouse</td><td>V1</td><td>158.3</td><td>25</td><td>Huang et al., 2019</td></tr><tr><td>12</td><td>GCaMP6s</td><td>tg(tetOs)</td><td>Mouse</td><td>V1</td><td>151.6</td><td>6</td><td>Huang et al., 2019</td></tr><tr><td>13</td><td>GCaMP6s</td><td>tg(Emx1)</td><td>Mouse</td><td>V1</td><td>157.5</td><td>26</td><td>Huang et al., 2019</td></tr><tr><td>14</td><td>GCaMP6s</td><td>AAV</td><td>Mouse</td><td>V1</td><td>60.1</td><td>7</td><td>Chen et al., 2013</td></tr><tr><td>15</td><td>GCaMP6s</td><td>AAV</td><td>Mouse</td><td>V1</td><td>59.1</td><td>9</td><td>Theis et al., 2016</td></tr><tr><td>16</td><td>GCaMP6s</td><td>AAV</td><td>Mouse</td><td>V1</td><td>59.1</td><td>9</td><td>Theis et al., 2016</td></tr><tr><td>17</td><td>GCaMP5k</td><td>AAV</td><td>Mouse</td><td>V1</td><td>50.0</td><td>9</td><td>Akerboom et al., 2012</td></tr><tr><td>18</td><td>R-CaMP1.07</td><td>tg(Grik4-cre) + AAV</td><td>Mouse</td><td>CA3</td><td>20.0</td><td>4</td><td>Schoenfeld et al., 2021</td></tr><tr><td>19</td><td>R-CaMP1.07</td><td>AAV</td><td>Mouse</td><td>S1</td><td>15.0</td><td>9</td><td>Bethge et al., 2017</td></tr><tr><td>20</td><td>jRCaMP1a</td><td>AAV</td><td>Mouse</td><td>V1</td><td>15.0</td><td>10</td><td>Dana et al., 2016</td></tr><tr><td>21</td><td>jRGECO1a</td><td>AAV</td><td>Mouse</td><td>V1</td><td>29.8</td><td>11</td><td>Dana et al., 2016</td></tr><tr><td>22</td><td>OGB-1</td><td>Injection + tg(GFP-GIN)</td><td>Mouse</td><td>V1 (SST)</td><td>15.6</td><td>5</td><td>Kwan and Dan, 2012</td></tr><tr><td>23 24</td><td>OGB-1 GCaMP6f</td><td>Injection + tg(tdTomato-PV)</td><td>Mouse Mouse</td><td>V1 (PV) V1 (PV)</td><td>15.6 26.6</td><td>7</td><td>Kwan and Dan, 2012</td></tr><tr><td></td><td></td><td>tg(PV-cre) + AAV</td><td>(in vitro)</td><td></td><td></td><td>13</td><td>Khan et al., 2018</td></tr><tr><td>25 26</td><td>GCaMP6f</td><td>tg(SOM-cre) + AAV</td><td>Mouse (in vitro)</td><td>V1 (SST)</td><td>26.6</td><td>17</td><td>Khan et al., 2018</td></tr><tr><td></td><td>GCaMP6f</td><td>tg(VIP-cre) + AAV</td><td>Mouse (in vitro)</td><td>V1 (VIP)</td><td>26.6</td><td>11</td><td>Khan et al., 2018</td></tr><tr><td>27 28</td><td>GCaMP6f</td><td>tg(tdTomato-PV) + AAV</td><td>Mouse</td><td>V1 (PV)</td><td>30.0</td><td>4</td><td>Rupprecht et al., 2021</td></tr><tr><td></td><td>XCaMPgf</td><td>AAV</td><td>Mouse</td><td>V1</td><td>122</td><td>8</td><td>Zhang et al., 2023</td></tr><tr><td>29</td><td>GCaMP7f</td><td>AAV</td><td>Mouse</td><td>V1</td><td>122</td><td>22</td><td>Zhang et al., 2023</td></tr><tr><td>30</td><td>GCaMP8f</td><td>AAV</td><td>Mouse</td><td>V1</td><td>122</td><td>36</td><td>Zhang et al., 2023</td></tr><tr><td>31</td><td>GCaMP8m</td><td>AAV</td><td>Mouse</td><td>V1</td><td>122</td><td>42</td><td>Zhang et al., 2023</td></tr><tr><td>32 33</td><td>GCaMP8s Mixed</td><td>AAV AAV</td><td>Mouse</td><td>V1</td><td>122 122</td><td>39</td><td>Zhang et al., 2023</td></tr><tr><td></td><td>(GCaMP7f /8/</td><td></td><td>Mouse</td><td>V1 (Interneurons)</td><td></td><td>17</td><td>Zhang et al., 2023</td></tr><tr><td>40</td><td>XCaMPgf) GCaMP6s</td><td>tg(VGlut2-Cre)</td><td>Mouse</td><td>Spinal cord</td><td>~2.5</td><td>21</td><td>Rupprecht et al., 2025</td></tr><tr><td>41</td><td>GCaMP6s</td><td>tg(Viaat-Cre)</td><td>Mouse</td><td>Spinal cord</td><td>~2.5</td><td>23</td><td>Rupprecht et al., 2025</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Ground-truth recording sets used in the benchmark.
<table><tr><td>Category</td><td>Indicator</td><td>Recording sets</td><td></td><td>N</td></tr><tr><td>Chemical dye</td><td>OGB-1</td><td></td><td>DS01, DS02, DS04, DS22, DS23</td><td>64</td></tr><tr><td>Chemical dye</td><td>Cal-520</td><td></td><td>DS03, DS05</td><td>13</td></tr><tr><td>Legacy GECI</td><td>GCaMP5k</td><td></td><td>DS17</td><td>9</td></tr><tr><td>Fast GCaMP6</td><td>GCaMP6f</td><td></td><td>DS06-DS11, DS24-DS27</td><td>124</td></tr><tr><td>Slow GCaMP6</td><td>GCaMP6s</td><td></td><td>DS12-DS16</td><td>57</td></tr><tr><td>GCaMP7</td><td>GCaMP7f</td><td></td><td>DS29</td><td>22</td></tr><tr><td>GCaMP8 variants</td><td>GCaMP8f,</td><td>GCaMP8m,</td><td>DS30, DS31, DS32</td><td>117</td></tr><tr><td>Red-shifted GECI</td><td>GCaMP8s R-CaMP</td><td></td><td>DS18, DS19</td><td></td></tr><tr><td>Red-shifted GECI</td><td></td><td></td><td>DS20</td><td>13</td></tr><tr><td>Red-shifted GECI</td><td>jRCaMP1a</td><td></td><td>DS21</td><td>9</td></tr><tr><td>Other GECI</td><td>jGECO1a XCaMPgf</td><td></td><td>DS28</td><td>11 8</td></tr><tr><td>Interneurons</td><td>GCaMP7f,</td><td>GCaMP8f,</td><td>DS33</td><td>17</td></tr><tr><td></td><td>GCaMP8m,</td><td>GCaMP8s,</td><td></td><td></td></tr><tr><td>Spinal cord (excit.)</td><td>XCaMPgf</td><td></td><td>DS40</td><td>21</td></tr><tr><td></td><td>GCaMP6s</td><td></td><td></td><td></td></tr><tr><td>Spinal cord (inhib.)</td><td>GCaMP6s</td><td></td><td>DS41</td><td>23</td></tr></table>

Grouping these recordings by indicator family shows how the 33 source datasets contribute complementary kinetic regimes. The fixed splits below then isolate five representative transfer settings while preserving recording-level separation.

Table 8: Fixed evaluation splits. Source-side recordings are excluded from training under LOIO; evaluation recordings are held out in all settings.
<table><tr><td>Split</td><td>Held-out under LOIO</td><td>Eval. recordings</td><td>Contrast</td></tr><tr><td>OGB-1</td><td>DS01, DS22, DS23</td><td>DS02, DS04</td><td>Chemical-dye transfer</td></tr><tr><td>GCaMP6s</td><td>DS12, DS13, X-DS11, X-DS12</td><td>DS14, DS15, DS16</td><td>Slow-decay GECI</td></tr><tr><td>GCaMP8 variants</td><td>DS30, DS31</td><td>DS32</td><td>GCaMP8f/8m→8s</td></tr><tr><td>Red-shifted</td><td>DS18, DS19, DS20</td><td>DS21</td><td>Cross-indicator transfer</td></tr><tr><td>Spinal cord</td><td>DS41</td><td>DS40</td><td>Brain-to-spinal-cord</td></tr></table>

## A.3 Loss function ablation under +sim LOIO

Table 9 compares four loss configurations on the five +sim LOIO splits. Adding each auxiliary term consistently improves mean correlation, and the full three-component setup achieves the highest score on every split.

Table 9: Mean Pearson correlation under +sim LOIO for SpikeSSL trained with progressive loss additions. All runs use identical data and hyperparameters, and only the active loss terms differ. Bold: best per column.
<table><tr><td>Loss configuration</td><td>V1-OGB</td><td>V1-GCaMP6s</td><td>V1-GCaMP8</td><td>Other-RCaMP</td><td>SC-GCaMP6s</td><td>Mean</td></tr><tr><td>Huber regression only</td><td>0.527</td><td>0.445</td><td>0.861</td><td>0.703</td><td>0.622</td><td>0.632</td></tr><tr><td>+ variance head (β-NLL)</td><td>0.536</td><td>0.461</td><td>0.872</td><td>0.712</td><td>0.641</td><td>0.644</td></tr><tr><td>+ gate BCE</td><td>0.540</td><td>0.467</td><td>0.876</td><td>0.717</td><td>0.651</td><td>0.650</td></tr><tr><td>Full (+ mean-rate)</td><td>0.547</td><td>0.476</td><td>0.886</td><td>0.725</td><td>0.668</td><td>0.660</td></tr></table>

## A.4 Biophysical simulation pipeline specification

## A.4.1 Spike Time Simulation

To construct realistic spike timings, we implemented two distinct strategies, namely drawing interspike intervals from predefined probability distributions and simulating membrane potential dynamics via the Izhikevich neuron model.

In the first strategy, the total spike count for each trial is determined by a target mean firing rate $\lambda _ { \mathrm { s p i k e } }$ and a randomly generated recording duration T. Three stochastic point-process models cover distinct firing regimes. For standard stochastic firing, the spike count follows a Poisson distribution $N \sim \bar { \mathrm { P o i s s o n } } ( \lambda _ { \mathrm { s p i k e } } T )$ and inter-spike intervals (ISIs) are drawn independently. A Gamma distribution $\Delta t _ { k } \sim \dot { \Gamma } ( \dot { \kappa } , \theta )$ generalizes this to model the relative refractory period, reducing to the standard exponential when $\kappa = 1$ and penalizing short ISIs when $\kappa > 1$ . For quasi-periodic firing, a linear interpolation introduces stochastic perturbations into a uniform spike train, $\Delta t _ { k } =$ $( 1 - \bar { r } ) { \cdot } T _ { \mathrm { p e r i o d } } + r { \cdot } \bar { T } _ { \mathrm { r a n d } }$ , where $T _ { \mathrm { p e r i o d } } = 1 / \lambda$ and ${ \bf { \bar { \rho } } } _ { r } \in [ 0 , 1 ]$ controls the degree of randomness. For bursty firing, burst onsets follow a Poisson process with rate $\lambda _ { \mathrm { b u r s t } }$ , the spike count per burst follows $n _ { i } \sim \mathrm { P o i s s o n } ( \mu _ { \mathrm { b u r s t } } )$ , and intra-burst ISIs are generated by stochastic jitter around a sub-millisecond mean interval.

In the second strategy, we numerically integrate the Izhikevich neuron model over a duration T to efficiently generate biophysically realistic spike trains. By capturing the rapid depolarization and slow recovery of the membrane potential, this model faithfully reproduces complex firing dynamics. To reflect physiological stochasticity, the driving current I(t) is formulated as a multi-scale composite synaptic drive:

$$
I ( t ) = I _ { \mathrm { b a s e } } + I _ { \mathrm { s l o w } } ( t ) + I _ { \mathrm { m e d } } ( t ) + I _ { \mathrm { f a s t } } ( t ) ,
$$

where $I _ { \mathrm { b a s e } }$ denotes the baseline current profile, while the slow, med, and fast terms represent the respective frequency components of the synaptic perturbations.The temporal evolution of the fast membrane potential v and the slow recovery variable u under this driving current is governed by the following system of ordinary differential equations:

$$
\begin{array} { l } { \displaystyle { \frac { d v } { d t } = 0 . 0 4 v ^ { 2 } + 5 v + 1 4 0 - u + I , } } \\ { \displaystyle { \frac { d u } { d t } = a ( b v - u ) . } } \end{array}
$$

Upon reaching the spike apex $( v \geq 3 0 m V )$ , the state variables are instantaneously reset: $v  c$ and $u  u + d .$ . The dimensionless parameter set $( a , b , c , d )$ governs these intrinsic dynamics, enabling the simulation of distinct neuronal subclasses, including Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH), and Fast Spiking (FS) neurons.

## A.4.2 Fluorescence Signal Simulation

In the MLSpike framework, the normalized intracellular calcium concentration $c ( t )$ is modeled as a first-order linear system driven by the spike train $\begin{array} { r } { s ( t ) = \sum _ { k } \delta ( t - t _ { k } ) } \end{array}$ , which is a set of Dirac functions placed at spike times $t _ { k }$ . The calcium level decays exponentially with a time constant τ , governed by the differential equation:

$$
\frac { d c } { d t } = - \frac { 1 } { \tau } c + s ( t ) .
$$

However, the assumption of a constant unitary response deviates from actual biological processes and contradicts experimentally observed data. To address this, we define the input $s ( t )$ as a sequence of weighted discrete events, where the calcium influx elicited by the k-th spike at time $t _ { k }$ is scaled by a dynamic weight w $\left( t _ { k } \right)$ . Consequently, the dynamics of $c ( t )$ are modified to:

$$
\frac { d c } { d t } = - \frac { 1 } { \tau } c + s ( t ) * w ( t ) ,
$$

where the dynamic weight is formulated as:

$$
w ( t ) = R ( t ) \cdot \operatorname* { m a x } ( 0 , 1 + \epsilon _ { k } ) .
$$

Here, $\epsilon _ { k } \sim \mathcal N ( 0 , \sigma _ { A } ^ { 2 } )$ represents the stochastic amplitude jitter drawn from a normal distribution, reflecting the natural variability of calcium influx per spike. The variable $R ( t ) \in [ 0 , 1 ]$ serves as a depletion variable representing the fraction of the available calcium response pool. When an action potential occurs at $t = t _ { k }$ , the available pool is instantaneously depleted by a constant proportion d (the depletion factor):

$$
R ( t _ { k } ^ { + } ) = R ( t _ { k } ) \cdot ( 1 - d ) ,
$$

where $R ( t _ { k } ^ { + } )$ denotes the instantaneous value of $R ( t )$ immediately after the spike occurs at time $t _ { k }$ During the inter-spike intervals $( t \neq t _ { k } )$ , the temporal evolution is a continuous exponential recovery towards its maximum capacity of 1:

$$
\frac { d R } { d t } = \frac { 1 } { \tau _ { r e c } } ( 1 - R ( t ) ) ,
$$

where $\tau _ { r e c }$ is the recovery time constant.

Subsequent to the influx of free calcium, the observable fluorescence signal is fundamentally determined by the concentration of calcium-bound indicator molecules. The binding between calcium ions and indicator molecules [B] obeys mass-action kinetics, as illustrated by the reaction:

$$
\mathrm { [ C a ] } ^ { n } + \mathrm { [ B ] } \underset { k _ { \mathrm { o f f } } } { \overset { k _ { \mathrm { o n } } } {  } } \mathrm { [ C a ^ { n } B ] } ,
$$

where $k _ { \mathrm { o n } }$ and $k _ { \mathrm { o f f } }$ are the association and dissociation rates, respectively, and n is the Hill coefficient representing the degree of cooperativity. This formulation can accommodate both genetically encoded calcium indicators (GECIs, e.g., GCaMP, where $n > 1 )$ and synthetic chemical dyes $( \mathbf { e } . \mathbf { g } . , \mathrm { O G B }  – 1$ where $n = 1 )$ . According to the law of mass action, the rate of change for the bound product concentration is given by:

$$
\frac { d \left[ C a ^ { n } B \right] } { d t } = k _ { \mathrm { o n } } \left[ C a \right] ^ { n } \left[ B \right] - k _ { \mathrm { o f f } } \left[ C a ^ { n } B \right] .
$$

By defining the normalized bound indicator fraction $p ( t )$ as:

$$
p = \frac { 1 } { \gamma } \frac { [ C a ^ { n } B ] - [ C a ^ { n } B ] _ { 0 } } { [ B ] _ { T } - [ C a ^ { n } B ] _ { 0 } } ,
$$

the temporal evolution of $p ( t )$ under standard constant-rate assumptions is governed by the following parameterized differential equation:

$$
\frac { d p } { d t } = \frac { 1 } { \tau _ { o n } } \left[ 1 + \gamma ( ( c _ { 0 } + c ) ^ { n } - c _ { 0 } ^ { n } ) \right] \left( \frac { ( c _ { 0 } + c ) ^ { n } - c _ { 0 } ^ { n } } { 1 + \gamma ( ( c _ { 0 } + c ) ^ { n } - c _ { 0 } ^ { n } ) } - p \right) ,
$$

where $\tau _ { o n }$ is the binding time constant at baseline, $c _ { 0 }$ is the normalized resting calcium level, and γ is the saturation parameter limiting the maximum bound fraction.

To extend this standard formulation and mechanistically characterize the conformational locking of the indicator molecules, we relax the constant-rate assumption for $k _ { \mathrm { o f f } }$ . Specifically, the dissociation rate is modeled as a dynamically decelerating variable that explicitly depends on the instantaneous occupancy fraction $p ( t )$

$$
k _ { \mathrm { o f f } } ( t ) = k _ { \mathrm { o f f , 0 } } \cdot \operatorname* { m a x } \left( 0 . 1 , 1 - \alpha \cdot p ( t ) \right) ,
$$

where $k _ { \mathrm { o f f } , 0 }$ is the basal dissociation rate and α determines the deceleration magnitude.

Next, we apply a phenomenological polynomial mapping to the dynamically derived variable $p ( t )$ This transformation yields the nonlinearly scaled bound fraction $p _ { \mathrm { f i n a l } } ( t )$

$$
\begin{array} { r } { p _ { \mathrm { f i n a l } } ( t ) = \left\{ \begin{array} { l l } { c _ { 3 } p ( t ) ^ { 3 } + c _ { 2 } p ( t ) ^ { 2 } + c _ { 1 } p ( t ) , } & { \mathrm { i f ~ } p ( t ) \leq 1 } \\ { 1 + \kappa \cdot ( p ( t ) - 1 ) , } & { \mathrm { i f ~ } p ( t ) > 1 } \end{array} \right. } \end{array}
$$

where the constraint $c _ { 1 } = 1 - c _ { 2 } - c _ { 3 }$ ensures gain conservation at $p = 1$ , and $\kappa = 3 c _ { 3 } + 2 c _ { 2 } + c _ { 1 }$ represents the tangent slope at $p = 1$ , facilitating a smooth linear extrapolation that guarantees numerical stability outside the standard dynamic range.

Furthermore, to account for the slow continuous baseline drifts inherent to realistic scenarios, we model the baseline fluorescence $B ( t )$ as a Wiener process:

$$
\frac { d B ( t ) } { d t } = \eta \cdot \xi _ { B } ( t ) ,
$$

where $\eta$ scales the amplitude of the baseline drift and $\xi _ { B } ( t )$ denotes standard Gaussian white noise. Ultimately, the observed fluorescence intensity $F ( t )$ is formulated as the integrated effect of the drifting baseline, the dynamic bound indicator concentration, and the measurement noise:

$$
F ( t ) = B ( t ) \left( 1 + A \cdot p ( t ) \right) + \sigma \cdot \xi _ { F } ( t ) ,
$$

where A represents the maximum fluorescence transient amplitude, and $\boldsymbol { \sigma } \cdot \boldsymbol { \xi } _ { F } ( t )$ accounts for the additive Gaussian noise with variance $\sigma ^ { 2 }$

## A.5 Detailed benchmark construction

The benchmark uses fixed recording-set partitions rather than random neuron-level resampling. In the in-domain setting, the source-side recordings listed in Table 8 are included during training and the corresponding evaluation recordings are held out for testing. In the LOIO setting, the sourceside recordings from the target evaluation family are removed from training, while the evaluation recordings remain unchanged. All other real source-domain recordings remain available for training, so the only controlled difference between LOIO runs is the held-out indicator or tissue family.

The five splits in Table 8 are the evaluation units behind the in-domain and LOIO benchmark tables in the main paper. The held-out evaluation recordings are fixed at the recording-set level, which prevents different methods from being compared under different neuron-level samples.

Before training and testing, each fluorescence trace is windowed into 128-frame segments with 30 frames of left context, 30 frames of right context, and overlap size 2. We use raw fluorescence as the input representation, do not apply offline denoising, and do not resample traces before evaluation. When simulation augmentation is enabled, simulated traces are appended to the training pool only and are never included in any evaluation split.

## A.6 Additional results and analyses

## A.6.1 Representative transfer functions

Figure 9 indicates that the dominant failure mode is not a uniform offset but an indicator-dependent distortion of gain. For OASIS and MLSpike, the predicted rate increases too slowly once the true rate enters the moderate and high regime, which is consistent with the strong saturation visible in the curves. CASCADE and $\mathrm { E N S } ^ { 2 }$ recover more dynamic range, but their mappings still bend away from the identity and exhibit broader inter-neuron variability, indicating that rate calibration remains unstable across indicators. SpikeSSL stays closer to the identity over a wider operating range and maintains a narrower spread across neurons, suggesting that its advantage comes from preserving response gain more faithfully rather than from isolated corrections at a few firing-rate values.

![](images/61b635d381cf039add97e96cafe85369e22aab37359bf9872ac0bda39b525d7b.jpg)  
Figure 9: Representative transfer functions for five commonly used indicators. The shaded region denotes inter-neuron variability, and the dashed diagonal indicates the identity mapping. Indicators are selected because they span distinct kinetic regimes.

## A.6.2 Long-trial LOIO visualizations

To complement the per-window correlation scores reported in the main benchmark, we provide long-trial visualisations that span the full recording duration. We include one representative neuron from each of the six held-out calcium indicators in the LOIO evaluation, namely the chemical dye OGB-1, the early and improved fast green GECIs GCaMP5k and GCaMP7f, the slow green GECI GCaMP6s, the red-shifted GECI jGECO1a, and the next-generation fast green GECI GCaMP8f. For every neuron we display the recorded fluorescence trace together with spike-rate predictions from all six methods over the entire recording, with the smoothed ground-truth spike rate overlaid in grey for reference. The Pearson correlation coefficient r reported in each panel is computed over the full trace. These visualisations corroborate the in-domain and LOIO benchmarks in the main paper, and additionally show that SpikeSSL with simulation augmentation tracks both isolated bursts and prolonged elevated firing across indicators with markedly different kinetics, while the remaining gap that simulation augmentation closes is most pronounced for indicators whose kinetics deviate substantially from the supervised training pool, such as jGECO1a and GCaMP5k.

![](images/38c295daba83043e6fd46b48463799cdf6ca31f7998f8cc0cbe1ac5337e5d56c.jpg)  
Figure 10: LOIO spike inference on a representative OGB-1 neuron (chemical dye with slow calcium decay kinetics). The top row shows the recorded $\Delta F / F$ trace. Subsequent rows display spike-rate estimates produced by OASIS, MLSpike, CASCADE, ENS<sup>2</sup>, SpikeSSL, and SpikeSSL+Sim., each overlaid with the smoothed ground-truth spike rate (grey). The Pearson correlation coefficient r is computed over the full recording duration.

![](images/abf2f7f20fe545a97ccdceccc5e59c651ad533d798ca8b7848c0a0584e26e48a.jpg)  
Figure 11: LOIO spike inference on a representative GCaMP5k neuron (an early-generation fast green GECI). Panel layout follows Figure 10.

![](images/539f601015e3b52e61f8adda48b385df1fe2dec06ba5c4fecd7ef97f0c72c69f.jpg)  
Figure 12: LOIO spike inference on a representative GCaMP7f neuron (an improved fast green GECI with enhanced sensitivity). Panel layout follows Figure 10.

![](images/481ab7498661bc010f49059076c907f8dd8f506eb8c0219d7e9ed46572dd4c33.jpg)  
Figure 13: LOIO spike inference on a representative GCaMP6s neuron (a slow GECI recorded in a low-activity regime). Panel layout follows Figure 10.

![](images/d908900d27b73f98160bce9a1576f759f0db52e782cf84aeff02639615835b63.jpg)  
Figure 14: LOIO spike inference on a representative jGECO1a neuron (a red-shifted GECI with distinct spectral and kinetic properties). Panel layout follows Figure 10.

![](images/fae7190cb9f13ede3da4515157073be5ec15b481ad4bc3caeedc5be04b7ab596.jpg)  
Figure 15: LOIO spike inference on a representative GCaMP8f neuron (a next-generation fast GECI with high signal-to-noise ratio). Panel layout follows Figure 10.

The following subsections present additional controls, robustness studies, and mechanistic analyses that complement the main results. All experiments use the fixed recording-set partitions described in Appendix A.5. Unless stated otherwise, correlation denotes Pearson correlation of predicted and ground-truth spike rates, higher is better, and error and absolute bias are lower is better.

## A.7 Test-Time Conditioning Uses No Target-Domain Pool

The seven trace statistics are computed independently from the current input window. Thus, LOIO inference uses one unlabeled input window, no target-domain pool, no target labels, and no parameter update. Pooling statistics across multiple target windows does not improve performance (Table 10), supporting that the method is per-window conditional inference rather than few-shot adaptation.

Table 10: Effect of pooling unlabeled target windows for trace statistics under LOIO.
<table><tr><td></td><td>Per-window</td><td>Pool 5</td><td>Pool 20</td><td>Pool all</td></tr><tr><td>Mean correlation</td><td>0.654</td><td>0.649</td><td>0.650</td><td>0.650</td></tr></table>

LOIO uses the reserved unknown indicator code (ID=0), never a held-out dye label. Removing the discrete-ID branch changes mean LOIO correlation from 0.654 to 0.647. In contrast, in-domain performance falls from 0.741 to 0.703 without the ID branch, and shuffling the observed ID at inference gives 0.578. Therefore, the ID chiefly captures regularities of observed domains, while zeroshot transfer is supported by sampling rate, input-window statistics, and the trace itself (Table 11).

Table 11: Indicator-ID intervention. The LOIO baseline uses the unknown ID=0.
<table><tr><td>Setting</td><td>V1-OGB</td><td>V1-GCaMP6s</td><td>V1-GCaMP8</td><td>Other-RCaMP</td><td>SC-GCaMP6s</td><td>Mean</td></tr><tr><td>LOIO baseline (ID=0)</td><td>0.650</td><td>0.509</td><td>0.853</td><td>0.717</td><td>0.542</td><td>0.654</td></tr><tr><td>LOIO without ID branch</td><td>0.646</td><td>0.500</td><td>0.849</td><td>0.708</td><td>0.532</td><td>0.647</td></tr><tr><td>In-domain full model</td><td>0.743</td><td>0.579</td><td>0.915</td><td>0.731</td><td>0.736</td><td>0.741</td></tr><tr><td>In-domain shuffled ID</td><td>0.587</td><td>0.457</td><td>0.791</td><td>0.619</td><td>0.437</td><td>0.578</td></tr><tr><td>In-domain without ID branch</td><td>0.700</td><td>0.546</td><td>0.881</td><td>0.694</td><td>0.694</td><td>0.703</td></tr></table>

For completeness, the structural loss uses valid-frame mask $\begin{array} { r } { w _ { t } \in \{ 0 , 1 \} , Z = \sum _ { t } w _ { t } , Q = \{ t : s _ { t } \geq } \end{array}$ $q _ { 0 . 9 } ( s ) \}$ }, and Huber loss $h _ { \delta } \colon$

$$
L _ { \mathrm { o v e r } } = Z ^ { - 1 } \sum _ { t } w _ { t } \operatorname* { m a x } ( \hat { s } _ { t } - s _ { t } , 0 ) ^ { 2 } ,\tag{11}
$$

$$
L _ { \mathrm { p e a k } } = | Q | ^ { - 1 } \sum _ { t \in Q } h _ { \delta } ( \operatorname* { m a x } ( s _ { t } - \hat { s } _ { t } , 0 ) ) ,\tag{12}
$$

$$
L _ { \mathrm { b i a s } } = \left( Z ^ { - 1 } \sum _ { t } w _ { t } \hat { s } _ { t } - Z ^ { - 1 } \sum _ { t } w _ { t } s _ { t } \right) ^ { 2 } ,\tag{13}
$$

$$
L _ { \mathrm { s t r u c t } } = \beta _ { \mathrm { o v e r } } L _ { \mathrm { o v e r } } + \beta _ { \mathrm { p e a k } } L _ { \mathrm { p e a k } } + \lambda _ { \mu } L _ { \mathrm { b i a s } } .\tag{14}
$$

We use β<sub>over</sub> = 3.0, β<sub>peak</sub> = 0.3, λ<sub>µ</sub> = 0.5, δ = 0.1, and q = 0.9.

## A.8 Breadth of the Universal Benchmark

The benchmark aggregates ground-truth recordings from multiple laboratories, sensors, species, brain regions, cell types, and acquisition protocols rather than a single experimental setup (Table 6). LOIO removes every recording of the held-out family and applies no target-domain update. To characterize the resulting domain variation, we measured isolated-spike rise time, decay time, and SNR in the public recordings. Large within-family dispersion and systematic cross-family shifts are evident in Table 12. These event-level summaries are distinct from the baseline-aware exponential τ analysis in Table 19.

Table 12: Isolated-spike kinetic and SNR dispersion in public recordings. Times are seconds.
<table><tr><td>Indicator family</td><td>n</td><td>Rise  $\mu \pm \sigma$ </td><td> $\operatorname { D e c a y } \mu \pm \sigma$ </td><td> $\mathrm { S N R } ~ \mu \pm \sigma$ </td></tr><tr><td>Cal-520</td><td>6</td><td> $0 . 0 8 6 { \pm } 0 . 0 8 5$ </td><td> $0 . 2 5 3 { \scriptstyle \pm 0 . 0 9 4 }$ </td><td> $5 . 5 3 3 { \pm } 2 . 2 9 5$ </td></tr><tr><td>GCaMP5k</td><td>36</td><td> $0 . 0 7 9 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $0 . 0 7 3 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $3 . 9 3 4 { \scriptstyle \pm 0 . 9 0 4 }$ </td></tr><tr><td>GCaMP6f</td><td>907</td><td> $0 . 0 6 1 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 1 1 9 { \pm } 0 . 1 2 0$ </td><td> $5 . 5 7 0 { \pm } 4 . 3 5 8$ </td></tr><tr><td>GCaMP6s</td><td>436</td><td> $0 . 1 0 4 { \pm } 0 . 0 9 5$ </td><td> $0 . 1 7 7 { \scriptstyle \pm 0 . 1 7 3 }$ </td><td> $5 . 7 1 3 { \pm } 3 . 0 1 1$ </td></tr><tr><td>GCaMP6s spinal</td><td>180</td><td> $0 . 1 1 2 { \pm } 0 . 1 6 4$ </td><td> $0 . 1 5 5 { \pm } 0 . 1 9 9$ </td><td> $5 . 2 1 0 { \scriptstyle \pm 2 . 9 0 6 }$ </td></tr><tr><td>GCaMP7f</td><td>57</td><td> $0 . 0 3 0 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 0 7 0 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $6 . 1 2 5 { \pm } 4 . 3 9 8$ </td></tr><tr><td>GCaMP8 variants</td><td>2,967</td><td> $0 . 0 1 8 { \pm } 0 . 0 2 0$ </td><td> $0 . 0 8 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $6 . 4 1 7 { \scriptstyle \pm 4 . 0 1 7 }$ </td></tr><tr><td>Interneurons mixed</td><td>102</td><td> $0 . 0 3 7 { \pm } 0 . 0 5 6$ </td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $3 . 8 8 6 { \scriptstyle \pm 0 . 9 3 6 }$ </td></tr><tr><td>OGB-1</td><td>330</td><td> $0 . 0 9 4 { \pm } 0 . 0 7 3$ </td><td> $0 . 1 9 3 { \pm } 0 . 1 5 5$ </td><td> $7 . 1 4 1 { \pm } 9 . 0 7 5$ </td></tr><tr><td>R-CaMP</td><td>260</td><td> $0 . 0 8 2 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 2 3 5 { \scriptstyle \pm 0 . 2 0 1 }$ </td><td> $1 0 . 8 3 1 { \pm } 1 1 . 6 6 6 $ </td></tr><tr><td>Unknown</td><td>619</td><td> $0 . 0 9 6 { \pm } 0 . 1 2 2$ </td><td> $0 . 1 3 8 { \pm } 0 . 1 4 3$ </td><td> $5 . 6 8 9 { \pm } 4 . 2 8 3$ </td></tr><tr><td>XCaMPgf</td><td>37</td><td> $0 . 0 2 8 { \pm } 0 . 0 3 1$ </td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $4 . 5 1 1 \pm 2 . 4 4 5$ </td></tr><tr><td> $\operatorname { j } \operatorname { G E C O 1 a }$ </td><td>96</td><td> $0 . 0 6 6 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 1 2 6 { \pm } 0 . 0 9 2$ </td><td> $6 . 1 1 0 { \scriptstyle \pm 3 . 6 7 1 }$ </td></tr><tr><td> $\mathbf { j R C a M P l a }$ </td><td>56</td><td> $0 . 1 4 6 { \pm } 0 . 1 6 6$ </td><td> $0 . 2 6 1 { \pm } 0 . 1 9 8$ </td><td> $1 1 . 6 2 7 { \pm } 2 6 . 4 2 2$ </td></tr></table>

For example, rise-time standard deviation exceeds the corresponding mean for GCaMP6f, GCaMP8 variants, and XCaMPgf. Across families, mean decay ranges from 0.050 s to 0.261 s and mean SNR from 3.886 to 11.627. We use “universal” in this operational sense: a single model transfers across this diverse, multi-laboratory benchmark without target labels or retraining.

## A.9 Data Scaling and Statistical Robustness

Increasing the amount of real data, synthetic data, or indicator-family coverage improves mean LOIO correlation (Table 13). These controls indicate that data quantity is the primary limitation in the present benchmark.

Table 13: Data-scaling study under fixed LOIO evaluation.
<table><tr><td>Factor</td><td>Metric</td><td colspan="4">Scale</td><td></td></tr><tr><td>Real-recording fraction</td><td>Correlation</td><td>25%: 0.564</td><td>50%: 0.614</td><td>75%: 0.637</td><td>100%: 0.654</td><td></td></tr><tr><td></td><td>Error</td><td>25%: 1.791</td><td>50%: 1.534</td><td>75%: 1.133</td><td>100%: 1.123</td><td></td></tr><tr><td></td><td>Absolute bias</td><td>25%: 1.187</td><td>50%: 1.081</td><td>75%: 0.583</td><td>100%: 0.588</td><td></td></tr><tr><td>Synthetic-data fraction</td><td>Correlation</td><td>0%: 0.654</td><td>25%: 0.667</td><td>50%: 0.679</td><td>75%: 0.692</td><td>100%: 0.704</td></tr><tr><td></td><td>Error</td><td>0%: 1.123</td><td>25%: 1.118</td><td>50%: 1.113</td><td>75%: 1.108</td><td>100%: 1.103</td></tr><tr><td></td><td>Absolute bias</td><td>0%: 0.588</td><td>25%: 0.564</td><td>50%: 0.539</td><td>75%: 0.514</td><td>100%: 0.490</td></tr><tr><td>Indicator families</td><td>Correlation</td><td>4: 0.543</td><td>8: 0.583</td><td>13: 0.654</td><td></td><td></td></tr><tr><td></td><td>Error</td><td>4:1.140</td><td>8:1.204</td><td>13: 1.123</td><td></td><td></td></tr><tr><td></td><td>Absolute bias</td><td>4: 0.871</td><td>8:0.811</td><td>13: 0.588</td><td></td><td></td></tr></table>

Across seeds {42, 2719, 8888, 1337, 2025}, mean LOIO correlation is $0 . 6 6 0 \pm 0 . 0 1 6 .$ , close to the seed-42 result of 0.654; in-domain correlation is 0.739 ± 0.004, close to 0.741. Table 14 reports all per-split results. Mean absolute bias is used in the final column.

Table 14: Five-seed robustness results. Values are mean ± standard deviation.
<table><tr><td></td><td colspan="3">LOIO</td><td colspan="3">In-domain</td></tr><tr><td>Split</td><td>Corr.</td><td>Error</td><td>Abs. bias</td><td>Corr.</td><td>Error</td><td>Abs. bias</td></tr><tr><td>V1-OGB</td><td>0.666±0.030</td><td>1.096±0.513</td><td>0.309±0.574</td><td>0.729±0.017</td><td>0.875±0.062</td><td>0.131±0.180</td></tr><tr><td>V1-GCaMP6s</td><td>0.507±0.022</td><td>0.951±0.008</td><td>0.780±0.069</td><td>0.575±0.009</td><td>0.948±0.005</td><td>0.779±0.031</td></tr><tr><td>V1-GCaMP8</td><td>0.864±0.013</td><td>0.531±0.015</td><td>0.054±0.213</td><td>0.917±0.005</td><td>0.487±0.044</td><td>0.036±0.059</td></tr><tr><td>Other-RCaMP</td><td>0.724±0.013</td><td>1.141±0.157</td><td>0.445±0.267</td><td>0.748±0.019</td><td>1.025±0.137</td><td>0.149±0.130</td></tr><tr><td>SC-GCaMP6s</td><td>0.539±0.018</td><td>0.889±0.046</td><td>0.840±0.082</td><td>0.728±0.009</td><td>0.825±0.036</td><td>0.784±0.050</td></tr><tr><td>Mean</td><td>0.660±0.016</td><td>0.922±0.116</td><td>0.486±0.074</td><td>0.739±0.004</td><td>0.832±0.021</td><td>0.376±0.069</td></tr></table>

## A.10 Ablations

Removal and cumulative build-up studies were repeated across the same five seeds. The bidirectional IIR backbone, multi-modal conditioning, gate head, and variance head each contribute to the full model (Table 15).

Table 15: Five-seed removal and cumulative build-up ablations under LOIO.
<table><tr><td>Removal variant</td><td>Corr.</td><td>Error</td><td>Abs. bias</td><td>Build-up stage</td><td>Corr.</td><td>Error</td><td>Abs. bias</td></tr><tr><td>-AdaLN</td><td>0.604±0.014</td><td>1.276±0.103</td><td>0.456±0.061</td><td>1D U-Net backbone</td><td>0.481±0.022</td><td>1.173±0.101</td><td>0.788±0.088</td></tr><tr><td>—statistics</td><td>0.626±0.012</td><td>1.213±0.091</td><td>0.556±0.073</td><td>Transformer backbone</td><td>0.484±0.020</td><td>1.197±0.104</td><td>0.618±0.074</td></tr><tr><td>—bidirectional IIR</td><td>0.597±0.016</td><td>1.377±0.120</td><td>0.719±0.086</td><td>Bidirectional IIR backbone</td><td>0.539±0.018</td><td>1.191±0.101</td><td>0.757±0.083</td></tr><tr><td>—variance head</td><td>0.611±0.013</td><td>1.243±0.112</td><td>1.005±0.105</td><td>+ indicator ID</td><td>0.550±0.017</td><td>1.167±0.096</td><td>0.696±0.079</td></tr><tr><td>-gate</td><td>0.624±0.011</td><td>1.153±0.084</td><td>0.402±0.057</td><td>+ sampling rate</td><td>0.554±0.016</td><td>1.175±0.098</td><td>0.672±0.077</td></tr><tr><td>Full SpikeSSL</td><td>0.660±0.016</td><td>0.922±0.116</td><td>0.486±0.074</td><td>+ signal statistics</td><td>0.576±0.015</td><td>1.145±0.091</td><td>0.703±0.081</td></tr><tr><td></td><td></td><td></td><td></td><td>+ AdaLN</td><td>0.594±0.014</td><td>1.153±0.087</td><td>0.696±0.076</td></tr><tr><td></td><td></td><td></td><td></td><td>+ gate</td><td>0.612±0.013</td><td>1.117±0.082</td><td>0.665±0.071</td></tr><tr><td></td><td></td><td></td><td></td><td>+ variance head</td><td>0.660±0.016</td><td>0.922±0.116</td><td>0.486±0.074</td></tr></table>

## A.11 Universal, Specific, and Few-Shot Regimes

For each of 13 indicator families, the specific-model comparison holds out one recording and trains on the remaining recordings of that indicator. Universal models pool all indicator families and use exactly the same held-out recording for evaluation. SpikeSSL benefits from pooling, unlike CASCADE (Table 16).

Table 16: Per-indicator comparison of specific and universal training.
<table><tr><td>Indicator</td><td>CASCADE-specific</td><td>SpikeSSL-specific</td><td>CASCADE-universal</td><td>SpikeSSL-universal</td></tr><tr><td>OGB-1</td><td>0.663</td><td>0.760</td><td>0.583</td><td>0.728</td></tr><tr><td>Cal-520</td><td>0.721</td><td>0.633</td><td>0.647</td><td>0.693</td></tr><tr><td>GCaMP5k</td><td>0.687</td><td>0.520</td><td>0.688</td><td>0.695</td></tr><tr><td>GCaMP6f</td><td>0.500</td><td>0.626</td><td>0.416</td><td>0.562</td></tr><tr><td>GCaMP6s</td><td>0.262</td><td>0.435</td><td>0.279</td><td>0.628</td></tr><tr><td>GCaMP7f</td><td>0.776</td><td>0.508</td><td>0.412</td><td>0.816</td></tr><tr><td>GCaMP8f</td><td>0.753</td><td>0.928</td><td>0.815</td><td>0.933</td></tr><tr><td>GCaMP8m</td><td>0.799</td><td>0.939</td><td>0.809</td><td>0.928</td></tr><tr><td>GCaMP8s</td><td>0.884</td><td>0.927</td><td>0.767</td><td>0.912</td></tr><tr><td>R-CaMP</td><td>0.309</td><td>0.555</td><td>0.644</td><td>0.691</td></tr><tr><td>jRCaMP1a</td><td>0.689</td><td>0.538</td><td>0.662</td><td>0.769</td></tr><tr><td>jGECO1a</td><td>0.736</td><td>0.613</td><td>0.672</td><td>0.790</td></tr><tr><td>XCaMPgf</td><td>0.797</td><td>0.537</td><td>0.446</td><td>0.833</td></tr><tr><td>Mean</td><td>0.660</td><td>0.655</td><td>0.603</td><td>0.768</td></tr></table>

For few-shot adaptation, 20 target-domain neurons are set aside as a support pool and all remaining traces form the evaluation set. For $k > 0 ,$ models are fine-tuned from the k = 0 universal LOIO weights using k labeled support neurons. Fine-tuning first exceeds the zero-shot value at $k = 5$ and reaches 0.719 at k = 20 (Table 17).

Table 17: Few-shot fine-tuning from zero-shot LOIO pretrained weights.
<table><tr><td>k</td><td>V1-OGB</td><td>V1-GCaMP6s</td><td>V1-GCaMP8</td><td>Other-RCaMP</td><td>SC-GCaMP6s</td><td>Mean</td></tr><tr><td>0</td><td>0.622</td><td>0.466</td><td>0.843</td><td>0.696</td><td>0.493</td><td>0.624</td></tr><tr><td>1</td><td>0.488</td><td>0.324</td><td>0.766</td><td>0.649</td><td>0.606</td><td>0.567</td></tr><tr><td>3</td><td>0.509</td><td>0.389</td><td>0.888</td><td>0.683</td><td>0.545</td><td>0.603</td></tr><tr><td>5</td><td>0.521</td><td>0.412</td><td>0.895</td><td>0.718</td><td>0.648</td><td>0.639</td></tr><tr><td>10</td><td>0.621</td><td>0.459</td><td>0.862</td><td>0.721</td><td>0.714</td><td>0.675</td></tr><tr><td>20</td><td>0.707</td><td>0.548</td><td>0.892</td><td>0.725</td><td>0.722</td><td>0.719</td></tr></table>

## A.12 Disentangling Indicator and Acquisition Shifts

We construct mini-domain evaluations that hold an indicator fixed while varying one additional factor, excluding other shift types from each data pool. In the species split, GCaMP6f mouse V1 recordings (DS09–11 and X-DS09/10) transfer to zebrafish recordings (DS06–08). The cell-type split fixes mouse V1 and GCaMP6f while changing the labeled population. The tissue split fixes GCaMP6s while transferring from brain to spinal cord. SpikeSSL gives the best correlation in all three cases (Table 18).

Table 18: Mini-domain shifts with fixed indicator.
<table><tr><td rowspan="2">Method</td><td colspan="3">GCaMP6f species</td><td colspan="3">GCaMP6f cell type</td><td colspan="3">GCaMP6s tissue</td></tr><tr><td>Corr.</td><td>Error</td><td>Bias</td><td>Corr.</td><td>Error</td><td>Bias</td><td>Corr.</td><td>Error</td><td>Bias</td></tr><tr><td>OASIS</td><td>0.191</td><td>0.982</td><td>-0.945</td><td>0.266</td><td>0.985</td><td>-0.974</td><td>0.365</td><td>0.975</td><td>-0.964</td></tr><tr><td>MLSpike</td><td>0.085</td><td>1.130</td><td>-0.706</td><td>0.098</td><td>1.015</td><td>-0.897</td><td>0.307</td><td>1.005</td><td>-0.919</td></tr><tr><td>CASČADE</td><td>0.540</td><td>1.304</td><td>0.539</td><td>0.574</td><td>0.977</td><td>-0.144</td><td>0.226</td><td>1.083</td><td>-0.442</td></tr><tr><td>ENS2</td><td>0.476</td><td>1.266</td><td>0.229</td><td>0.584</td><td>0.898</td><td>-0.680</td><td>0.278</td><td>1.092</td><td>-0.391</td></tr><tr><td>SpikeSSL</td><td>0.636</td><td>0.952</td><td>-0.442</td><td>0.673</td><td>0.931</td><td>-0.798</td><td>0.542</td><td>0.911</td><td>-0.592</td></tr></table>

## A.13 Dynamics Readouts of the Conditioned IIR Bank

We compare four complementary quantities in Table 19. Reference τ is a catalogue decay time, measured τ is fitted from isolated real fluorescence responses at 30 Hz using a joint baseline and exponential fit, and $\| W _ { \alpha } \| _ { F }$ and ACF τ are extracted from a separately trained single-indicator SpikeSSL model. $W _ { \alpha }$ maps the condition vector to shifts of the channel-wise α bank. Its Frobenius norm measures the strength of that influence, not a time constant. The norms vary across indicators, while the three kinetically similar GCaMP8 variants have similar values. ACF τ is positively associated with measured τ (Spearman $\rho = 0 . 8 8 6 , p = 5 . 6 \times 1 0 ^ { - 5 } )$ , although not in fixed proportion. These complementary readouts support indicator-dependent dynamics rather than a fixed shared kernel.

Table 19: Kinetic measurements and learned dynamics readouts.
<table><tr><td>Indicator</td><td>Ref. τ (ms)</td><td> $\| W _ { \alpha } \| _ { F }$ </td><td>Measured τ (ms)</td><td>ACF τ (ms)</td></tr><tr><td>GCaMP8f</td><td>60</td><td>5.35</td><td>175</td><td>145</td></tr><tr><td>GCaMP8s</td><td>70</td><td>5.94</td><td>314</td><td>165</td></tr><tr><td>GCaMP8m</td><td>80</td><td>6.11</td><td>136</td><td>132</td></tr><tr><td>GCaMP7f</td><td>100</td><td>0.60</td><td>223</td><td>154</td></tr><tr><td>XCaMPgf</td><td>120</td><td>0.39</td><td>179</td><td>152</td></tr><tr><td>GCaMP6f</td><td>180</td><td>5.44</td><td>212</td><td>140</td></tr><tr><td>Cal-520</td><td>200</td><td>0.41</td><td>878</td><td>178</td></tr><tr><td>GCaMP5k</td><td>250</td><td>0.50</td><td>547</td><td>163</td></tr><tr><td>R-CaMP</td><td>280</td><td>0.25</td><td>535</td><td>171</td></tr><tr><td>jRCaMP1a</td><td>280</td><td>0.74</td><td>756</td><td>163</td></tr><tr><td> $\operatorname { j G E C O 1 a }$ </td><td>300</td><td>0.81</td><td>572</td><td>162</td></tr><tr><td>GCaMP6s</td><td>600</td><td>4.64</td><td>912</td><td>192</td></tr><tr><td>OGB-1</td><td>750</td><td>2.59</td><td>854</td><td>191</td></tr></table>

## A.14 Simulation: Benefits and Failure Modes

We repeated simulation-augmented LOIO across five seeds and characterized the target gap in the exact conditioning space of SpikeSSL: log sampling rate, standard deviation, skewness, excess kurtosis, and autocorrelation at lags 1, 3, 5, and 10. After robust scaling, we compute a 512-projection sliced Wasserstein-1 distance. $\Delta W _ { 1 }$ is the real-only target gap minus the equal-sampled real-plussimulation gap, so a positive value denotes improved alignment. AC averages the four autocorrelation terms and shape averages standard deviation, skewness, and kurtosis.

Table 20: Simulation gain and alignment analysis.
<table><tr><td>Split</td><td>No-sim corr.</td><td>+Sim corr.</td><td>∆ corr.</td><td>∆SWD</td><td> $\Delta W _ { 1 }$  AC</td><td> $\Delta W _ { 1 }$  shape</td><td> $\Delta W _ { 1 }$  kurt.</td></tr><tr><td>V1-OGB</td><td>0.666±0.030</td><td> $0 . 7 0 4 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td>+0.038</td><td>+0.013</td><td>-0.111</td><td>+0.008</td><td>-0.132</td></tr><tr><td>V1-GCaMP6s</td><td>0.507±0.022</td><td> $0 . 5 3 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>+0.024</td><td>+0.065</td><td>+0.197</td><td>-0.031</td><td>+0.094</td></tr><tr><td>V1-GCaMP8</td><td>0.864±0.013</td><td> $0 . 9 1 4 { \pm } 0 . 0 1 5$ </td><td>+0.050</td><td>-0.090</td><td>-0.124</td><td>-0.157</td><td>-0.260</td></tr><tr><td>Other-RCaMP</td><td>0.724±0.013</td><td> $0 . 7 1 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>-0.012</td><td>+0.018</td><td>+0.110</td><td>-0.051</td><td>-0.180</td></tr><tr><td>SC-GCaMP6s</td><td>0.539±0.018</td><td>0.677±0.010</td><td>+0.138</td><td>+0.126</td><td>+0.187</td><td>+0.065</td><td>+0.202</td></tr><tr><td>Mean</td><td>0.660±0.016</td><td>0.708±0.011</td><td>+0.048</td><td></td><td></td><td></td><td></td></tr></table>

SC-GCaMP6s obtains the largest gain, alongside the largest overall, temporal, and shape gap reductions. Other-RCaMP is a diagnostic non-uniform case: simulation improves autocorrelation alignment but worsens shape, especially kurtosis, consistent with incomplete coverage of red-indicator nonlinear statistics. Its −0.012 change is within five-seed variability, so it does not establish harmful negative transfer. Conversely, GCaMP8 improves despite a larger marginal SWD, consistent with label-aware synthetic supervision regularizing the inverse mapping beyond a label-agnostic distributional metric. These results motivate target-relative weighting or rejection of synthetic windows when conditional dynamics are mismatched.

## A.15 Threshold, Artifact, and Runtime Sensitivity

The inference gate threshold was swept with the trained model fixed. The default $\tau = 0 . 5$ maximizes mean correlation, the task’s primary metric; normalized error is smallest at $\tau = 0 . 6$ , and absolute bias reaches its minimum at $\bar { \tau } = 0 . \bar { 7 }$ (Table 21).

Table 21: Gate-threshold sensitivity. Each entry is mean across five LOIO splits.
<table><tr><td>Threshold</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td></tr><tr><td>Correlation</td><td>0.628</td><td>0.651</td><td>0.654</td><td>0.654</td><td>0.648</td><td>0.640</td><td>0.628</td></tr><tr><td>Normalized error</td><td>1.454</td><td>1.322</td><td>1.214</td><td>1.123</td><td>1.046</td><td>1.256</td><td>1.392</td></tr><tr><td>Absolute bias</td><td>0.831</td><td>0.739</td><td>0.659</td><td>0.588</td><td>0.525</td><td>0.464</td><td>0.483</td></tr></table>

Table 22 characterizes two common non-biological perturbations. Per-window statistics do not make SpikeSSL invariant to artifacts: baseline drift is the dominant degradation, while sparse motion is milder. Standard motion correction and baseline detrending are recommended before inference.

Table 22: LOIO robustness to baseline drift and sparse motion artifacts.
<table><tr><td>Condition</td><td>V1-OGB</td><td>V1-GCaMP6s</td><td>V1-GCaMP8</td><td>Other-RCaMP</td><td>SC-GCaMP6s</td><td>Mean</td></tr><tr><td>Clean</td><td>0.650</td><td>0.509</td><td>0.853</td><td>0.717</td><td>0.542</td><td>0.654</td></tr><tr><td>Baseline drift</td><td>0.636</td><td>0.392</td><td>0.605</td><td>0.665</td><td>0.478</td><td>0.555</td></tr><tr><td>Sparse motion artifacts</td><td>0.618</td><td>0.466</td><td>0.835</td><td>0.702</td><td>0.500</td><td>0.624</td></tr><tr><td>Drift + motion</td><td>0.620</td><td>0.409</td><td>0.673</td><td>0.684</td><td>0.463</td><td>0.570</td></tr></table>

SpikeSSL is designed primarily for offline analysis because the bidirectional IIR block uses bounded future context within each window. Nevertheless, sliding-window inference avoids holding an entire recording in memory. On an A100, one forward pass is approximately 29 times faster than the arrival of a 30-Hz frame window, so overlapping-window processing can keep pace with many acquisition settings while remaining noncausal within each window.