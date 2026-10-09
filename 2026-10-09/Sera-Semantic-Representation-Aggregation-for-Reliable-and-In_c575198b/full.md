# Sera: Semantic Representation Aggregation for Reliable and Interpretable Batery Health Forecasting

Jiawei Li Infocomm Technology Cluster   
Singapore Institute of Technology Singapore   
jiawei.li@singaporetech.edu.sg   
Zuming Liu   
College of Smart Energy   
Shanghai Jiao Tong University   
Shanghai, China   
zmliu@sjtu.edu.cn

Fang Liu School of Science and Technology Singapore University of Social Sciences Singapore liufang@suss.edu.sg

Man-Fai Ng   
Institute of Advanced Intelligence and   
Computing (IAIC), Agency for Science,   
Technology and Research (A\*STAR), 1 Fusionopolis Way, #16–16 Connexis,   
Singapore 138632, Republic of Singapore ngmf@a-star.edu.sg Wei Zhang<sup>∗</sup> Infocomm Technology Cluster   
Singapore Institute of Technology Singapore   
wei.zhang@singaporetech.edu.sg   
Zhi Wei Seh   
Institute of Materials Research and   
Engineering (IMRE), Agency for Science,   
Technology and Research (A\*STAR), 2   
Fusionopolis Way, Innovis #08-03,   
Singapore 138634, Republic of Singapore   
sehzw@a-star.edu.sg

## Ab<sub>s</sub>t<sub>rac</sub>t

Battery state of health (SoH) forecasting is important for battery management, but remains challenging due to nonlinear degradation and heterogeneity across batteries. Existing data-driven approaches primarily use temporal models to learn from numerical battery time series, and higher-level degradation characteristics are often not explicitly represented. These characteristics, however, can provide degradation guidance to support reliable forecasting and make the influence of degradation more interpretable. In this paper, we propose Sera, a semantic representation aggregation framework that complements temporal modelling with degradation semantics. Guided by battery domain expertise, Sera extracts degradation semantics from time series and constructs two complementary representations using rule-based knowledge and LLM-based interpreta tion. The representations are independently encoded and integrated with the representation learned by temporal models through gated aggregations. Experiments on the mainstream benchmark across multiple prediction horizons and diferent temporal models show that Sera consistently improves forecasting performance, achieving up to a 37.3% reduction in prediction error over the temporal baseline and enhanced generalizability. Counterfactual analysis examines how forecasts respond to changes in degradation semantics to assess interpretability. The results show that prediction responses are consistent with the meanings of key degradation descriptors across tested horizons. Together, these findings demonstrate that structured degradation semantics and efective aggregation can improve forecasting accuracy and support reliable and interpretable battery health forecasting for advanced battery management.

Battery health, semantic representation, large language model, representation learning, rule-based diagnostics.

ACM Reference Format: Jiawei Li, Fang Liu, Wei Zhang, Zuming Liu, Man-Fai Ng, and Zhi Wei Seh. 2026. Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting. In . ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Predictive health management is increasingly important in manufacturing and industrial systems, where component health directly afects operational reliability, maintenance planning, and asset utilization. Batteries exemplify this need because their growing use in electric vehicles, energy storage, and industrial equipment requires monitoring and managing performance changes associated with repeated operation and electrochemical aging. Battery degradation reduces available capacity and operational eficiency, shortens service life, and, in severe cases, increases safety risks such as overheating and thermal runaway. These concerns make state of health (SoH), measured as the ratio of a battery’s current capacity to its initial capacity, an important indicator for characterizing battery health changes [18, 19]. While SoH estimation characterizes the current battery condition, SoH forecasting predicts its future evolution to inform battery management and lifecycle planning [13].

SoH forecasting is challenging because battery degradation is nonlinear, heterogeneous, and afected by multiple interacting factors. Electrochemical aging mechanisms, operating conditions, temperature, and usage patterns can correspond to diferent degradation trajectories [5, 17]. The trajectories difer even among batteries of the same model, limiting generalization of SoH models from one set of cells to unseen cells [1, 2]. The challenge becomes more evident when the prediction horizon increases, since the model must infer future degradation from a fixed amount of observed history. Besides, practical forecasting requires not only accurate temporal modelling but also representations that characterize the underlying degradation state to support reliable predictions across batteries and interpretable forecasting behavior.

Existing studies predict battery health using physics based models, data-driven methods, and hybrid solutions [6, 10]. Deep sequential models, such as LSTM and its variants, learn degradation patterns from historical measurements through temporal modelling [12], while Transformer based approaches use attention mecha nisms to capture both spatial and temporal dependencies [4, 8]. Transfer learning has also been investigated to improve generalization under domain shift [3, 20]. However, data-driven models often do not explicitly represent higher-level degradation characteristics, such as trend and severity, and possible degradation mechanisms. This limits the ability of the models to explicitly leverage such characteristics for prediction. As such, other approaches incorporate battery domain knowledge into SoH models. For example, empirical observations suggest that SoH generally declines with aging, electrical engineering provides equivalent circuit models to describe battery behavior, and materials science explains mechanisms such as lithium inventory loss and electrode degradation [7, 11, 15]. Such knowledge can be encoded through equations, parameters, or constraints, providing additional guidance and a more comprehensive understanding of battery degradation for SoH prediction.

Although temporal and knowledge-enhanced models have advanced battery health forecasting, existing methods predominantly represent degradation information numerically, which may not explicitly convey higher-level degradation characteristics. Our key idea is to introduce text-based semantic representations as another modality for characterizing degradation and enable more explicit and descriptive expression of degradation knowledge. These representations capture degradation trends and change points, severity and risk, and possible underlying mechanisms. By integrating semantic descriptions with numerical time series, we seek to leverage the strengths of diferent representations to improve forecasting accuracy and reliability and to enhance interpretability.

Based on this idea, we propose Sera, a semantic representation aggregation framework for reliable and interpretable battery health forecasting with both temporal and semantic representations. Given the battery time series data and domain knowledge, Sera constructs two semantic representations. The rule-based representation provides consistent degradation interpretation through predefined rules, and the LLM-based representation interprets relationships among multiple rules. These representations are encoded separately from the temporal representation and integrated through gated aggregations, which adaptively control the contribution of each representation. The aggregated representation is used to predict future SoH across multiple prediction horizons. Experiments on a mainstream benchmark show that Sera improves forecasting accuracy with up to a 37.3% reduction in mean absolute error (MAE) over the temporal baseline and improved generalization to unseen batteries. Counterfactual analysis further shows that prediction responses are consistent with the meanings of key degradation descriptors, supporting the interpretability of the forecasting behavior.

The main contributions of this work are threefold. First, we introduce a structured semantic representation that translates battery degradation evidence into interpretable knowledge-derived descriptors and makes such information available to SoH models. Second, we develop Sera, which integrates rule- and LLM-based semantic representations with temporal modelling through gated aggregation, providing a general framework for adaptively incorporating semantic and temporal representations. Third, we evaluate Sera on a real-world dataset across multiple prediction horizons and temporal encoders, and use counterfactual semantic interventions to assess the consistency between forecast responses and the meanings of degradation semantics.

The remainder of this paper is organized as follows. Section 2 defines the forecasting problem. Section 3 presents Sera. Section 4 reports the experimental setup, forecasting results, and interpretability analyses. Section 5 concludes the paper.

## 2 Batter<sub>y</sub> SoH Forecastin<sub>g</sub> Problem Formulation

In this section, we formulate battery health forecasting as a multivariate sequence-to-sequence prediction problem and investigate the forecasting task for a future horizon. For a battery with � observed cycles, each comprising charging and discharging time series, let ${ \bf x } _ { t } \in \dot { \mathbb { R } } ^ { F }$ denote the feature vector extracted from cycle �, where � is the number of input features. These features include cycle-level battery measurements and derived health indicators, such as voltage, current, temperature, and internal resistance. Given the most recent � cycles ending at cycle $t \geq L ,$ the historical observation window is defined as,

$$
\begin{array} { r } { \mathbf { X } _ { t - L + 1 : t } = \left[ \mathbf { x } _ { t - L + 1 } , \mathbf { x } _ { t - L + 2 } , \ldots , \mathbf { x } _ { t } \right] \in \mathbb { R } ^ { L \times F } , } \end{array}\tag{1}
$$

where � denotes the observation window length. Let �<sub>�</sub> denote the SoH of the battery at cycle �. The forecasting target is the future SoH trajectory over the next � cycles,

$$
\mathbf { y } _ { t + 1 : t + H } = [ y _ { t + 1 } , y _ { t + 2 } , \ldots , y _ { t + H } ] \in \mathbb { R } ^ { H } .\tag{2}
$$

The forecasting model therefore learns the mapping,

$$
\hat { \mathbf { y } } _ { t + 1 : t + H } = f _ { \theta } \left( \mathbf { X } _ { t - L + 1 : t } \right) ,\tag{3}
$$

where $f _ { \theta } ( \cdot )$ denotes a forecasting model parameterized by �. The model predicts the future SoH trajectory as $\hat { \mathbf { y } } _ { t + 1 : t + H }$ , which is expected to closely approximate the ground truth trajectory y<sub>�+1:�+�</sub>.

## 3 Sera M<sub>e</sub>th<sub>o</sub>d<sub>o</sub>l<sub>ogy</sub>

In this section, we introduce the overall architecture of Sera, followed by semantic representation construction, temporal modelling, aggregation, and model training.

## 3<sub>.</sub>1 D<sub>es</sub>i<sub>g</sub>n O<sub>ve</sub>r<sub>v</sub>i<sub>ew o</sub>f Sera

The overall architecture of Sera is illustrated in Fig. 1, following a four-layer pipeline comprising input, representation, aggregation, and forecasting. At the input layer, a window of time series provides battery measurements that are processed through three branches. The temporal branch learns degradation patterns directly from numerical measurements and provides the baseline for forecasting. The rule-based branch applies battery domain knowledge to extracted degradation evidence to construct semantic descriptions of the battery state. The LLM-based branch interprets the degradation evidence to generate contextual descriptions within the same semantic schema. The representations from the three branches are then adaptively aggregated to predict future SoH.

Semantic representation is one of the key contributions of Sera and is constructed through the rule- and LLM-based branches at the representation layer. The rule-based branch constructs deterministic degradation descriptions using predefined rules grounded in battery domain knowledge. These rules are expected to provide consistent and useful degradation information to guide SoH forecasting. However, they are fixed and not exhaustive, and therefore ofer limited flexibility in capturing diverse battery dynamics. To address this limitation, the LLM-based branch leverages the language understanding capability of LLMs to interpret degradation evidence more flexibly and provide richer semantic representations.

![](images/d6114b00b69ccdeb24f50cbab0b8ee703018885e9ae39372141c2d6b5025abd3.jpg)  
Fi<sub>g</sub>ure 1: Overall architecture of Sera. Given a window of batt<sub>e</sub>r<sub>y</sub> tim<sub>e se</sub>ri<sub>es, se</sub>m<sub>a</sub>nti<sub>c</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>es a</sub>r<sub>e e</sub>xtr<sub>ac</sub>t<sub>e</sub>d t<sub>o co</sub>n<sub>s</sub>tr<sub>uc</sub>t <sub>ru</sub>l<sub>e-</sub>b<sub>ase</sub>d <sub>an</sub>d LLM<sub>-</sub>b<sub>ase</sub>d d<sub>egra</sub>d<sub>a</sub>ti<sub>on seman</sub>ti<sub>cs.</sub> T<sub>empora</sub>l <sub>an</sub>d <sub>seman</sub>ti<sub>c</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s enco</sub>d<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y an</sub>d th<sub>en</sub> a<sub>gg</sub>re<sub>g</sub>ated for SoH forecastin<sub>g</sub>.

At the aggregation layer, the temporal, rule-based, and LLMbased information is independently encoded and integrated through a gated aggregation. The aggregation design enables Sera to adaptively control the contribution of diferent information sources rather than combining them deterministically. Finally, the aggregated representation is passed to the forecasting layer, where the prediction module generates the future SoH trajectory within the specified prediction horizon. The components of Sera are described in detail below.

## 3.2 Structured Semantic Re<sub>p</sub>resentation

Guided by battery domain knowledge, we select seven semantic rules to characterize battery degradation, including trend, risk level, mechanism, severity, change point, key metric, and metric value. The rule definitions, candidate values, and associated degradation features are summarized in Table 1. The rules for trend and change point characterize degradation progression and its changes, while those for risk level and severity indicate degradation risk and severity. The mechanism rule describes possible underlying degradation processes, while key metric and metric value associate the semantic descriptions with quantitative degradation features. Together, these rules define the structured semantic representation as,

$$
{ \bf z } = \left[ z _ { \mathrm { t r e n d } } , z _ { \mathrm { r i s k } } , z _ { \mathrm { m e c h } } , z _ { \mathrm { s e v } } , z _ { \mathrm { c p } } , z _ { \mathrm { m e t r i c } } , z _ { \mathrm { v a l u e } } \right] .\tag{4}
$$

To assign values to these semantic rules, we preprocess the battery time series into a feature vector s that captures degradation characteristics required for semantic construction. These characteristics are derived from temporal changes and statistical variations over the observation window. Specifically, we compute the slope and acceleration ratio of the capacity-based SoH sequence to characterize degradation progression, together with statistical summaries of temperature and internal resistance to characterize operating variability. These derived features provide the quantitative basis for assigning values to the semantic rules. For example, the SoH slope and acceleration ratio are used to distinguish among stable, mild\_decay, linear\_decay, and accelerating\_fade for the trend rule. The resulting s provides a common input to both semantic construction branches.

The rule-based branch implements the semantic rules using predefined thresholds grounded in domain knowledge. Specifically, the preprocessed features in s are compared against their corresponding thresholds to determine the candidate values described above. This threshold-based assignment produces a deterministic semantic representation for a given feature vector. The mapping is expressed as,

$$
{ \bf z } ^ { \mathrm { r u l e } } = f _ { \mathrm { r u l e } } ( { \bf s } ) ,\tag{5}
$$

where $f _ { \mathrm { r u l e } } ( \cdot )$ applies the same predefined rules across all input windows. Although this provides consistent semantic guidance, fixed thresholds may not capture contextual relationships among features, motivating the LLM-based branch.

The LLM branch interprets the same feature vector s through a prompt specification $\mathcal { P }$ that defines the semantic rules and their candidate values. The prompt instructs the LLM to consider the extracted features jointly when assigning values to the semantic rules, while restricting its output to the common schema, yielding,

$$
\begin{array} { r } { \mathbf { z } ^ { \mathrm { l l m } } = f _ { \mathrm { l l m } } ( \mathbf { s } , \mathcal { P } ) , } \end{array}\tag{6}
$$

where $f _ { \mathrm { l l m } } ( \cdot )$ denotes the LLM-based semantic mapping process. Drawing on knowledge learned from natural language, this branch provides greater flexibility in capturing contextual relationships among degradation features than predefined threshold-based mappings in Eq. (5). The resulting representation provides complementary semantic guidance while retaining the same semantic rules and candidate values as the rule-based branch.

## 3.3 Re<sub>p</sub>resentation Encodin<sub>g</sub> and A<sub>gg</sub>re<sub>g</sub>ation

Given the raw battery time series and the rule- and LLM-based semantic representations, Sera encodes each input along its corresponding branch to obtain temporal and semantic embeddings. These three embeddings are then integrated through gated aggregation for subsequent SoH prediction.

3.3.1 Temporal Encoding. In the first branch, we perform temporal modelling using an encoder that processes the battery time series window $\mathbf { X } \in \overline { { \mathbb { R } } } ^ { L \times F }$ to obtain a temporal embedding. We use PatchTST [9] as the temporal encoder in this study. Its patch-based design groups consecutive observations into local segments, preserving patterns within each patch while reducing the number of tokens processed by attention compared with using one token per cycle. Specifically, the input window is divided into � patches of length �, each projected into a �-dimensional embedding. A learnable classification (CLS) token, used here to summarize the input sequence, is prepended to the patch embeddings, and positional information is added to retain their temporal order. The resulting sequence of size $( M + 1 ) \times D$ is processed by $L _ { \mathrm { e n c } }$ Transformer encoder layers. The final CLS token output provides the temporal embedding as,

Table 1: Structured semantic re<sub>p</sub>resentation schema for batter<sub>y</sub> de<sub>g</sub>radation. Each semantic com<sub>p</sub>onent is described b<sub>y</sub> its candidate values<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> source features<sub>,</sub> and de<sub>g</sub>radation meanin<sub>g</sub>.
<table><tr><td>Rule</td><td>Candidate Values</td><td>Source Features</td><td>Description</td></tr><tr><td>Trend</td><td>stable, mild_decay, linear_decay, accelerating_fade</td><td>SoH slope, SoH acceleration ratio</td><td>Overall degradation tendency observed within the input win- dow.</td></tr><tr><td>Risk Level</td><td>low, medium, high</td><td>SoH slope, temperature std. (Tstd), IR std. (σIR)</td><td>Estimated level of degradation risk based on evidence from multiple battery indicators. Dominant electrochemical or thermal degradation behaviour</td></tr><tr><td>Mechanism</td><td>SEI_growth, lithium_plating, cathode_degradation, thermal_aging, impedance_rise, mixed stable</td><td>SoH slope, σIR, Tstd, charge time slope</td><td>inferred from the observed feature patterns.</td></tr><tr><td>Severity</td><td>mild, moderate, severe</td><td>Magnitude of SoH slope, mean SoH level</td><td>Strength of the degradation evidence observed within the input window.</td></tr><tr><td>Change Point</td><td>yes, no</td><td>SoH acceleration ratio, SoH acceler- ation flag</td><td>Indication of a possible transition in the observed degradation behaviour.</td></tr><tr><td>Key Metric</td><td>SoH_slope, Tstd, IRstd, CTslp, QDslp</td><td>Selected diagnostic indicator</td><td>Most informative feature supporting the current degradation interpretation.</td></tr><tr><td>Metric Value</td><td>Numerical (R)</td><td>Corresponding raw feature value</td><td>Numerical value of the selected diagnostic metric obtained from the extracted battery features.</td></tr></table>

$$
\mathbf { h } _ { \mathrm { t s } } = \mathbf { E } _ { \mathrm { c l s } } ^ { ( L _ { \mathrm { e n c } } ) } \in \mathbb { R } ^ { D } .\tag{7}
$$

Note that Sera is not restricted to PatchTST and can accommodate other temporal encoders to generate temporal embeddings. We evaluate several alternative architectures in the experiments.

3.3.2 Semantic Representation Encoding. The second and third branches independently encode the rule- and LLM-based semantic representations. These representations are generated through diferent methods and provide diferent interpretations of the degradation evidence. We use two separate encoders with the same architecture and independent parameters to retain their distinct information. In each branch, the structured semantic text is first encoded into a numerical vector, then processed by an MLP projection, a ReLU activation, and a final linear mapping to obtain the semantic embedding. The resulting semantic embeddings are,

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { r u l e } } = \mathsf { L i n e a r } \left( \mathsf { R e L U } \left( \mathsf { P r o j } \left( \mathsf { E m b e d } \left( \mathbf { z } ^ { \mathrm { r u l e } } \right) \right) \right) \right) \in \mathbb { R } ^ { D } , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { l l m } } = \mathsf { L i n e a r } \left( \mathsf { R e L U } \left( \mathsf { P r o j } \left( \mathsf { E m b e d } \left( \mathbf { z } ^ { \mathrm { l l m } } \right) \right) \right) \right) \in \mathbb { R } ^ { D } . } \end{array}\tag{9}
$$

Here, Embed(·) converts the structured semantic representation into a numerical vector, and Proj(·) denotes the MLP projection with intermediate dimension $d _ { \mathrm { p r o j } } = 6 4$ . The final linear mapping produces semantic embeddings of dimension $D ,$ matching the temporal embedding h<sub>ts</sub> for subsequent gated aggregation.

3.3.3 Adaptive Gated Aggregation. Sera integrates the temporal and semantic embeddings through adaptive gated aggregation. The three embeddings are adaptively weighted and aggregated as,

$$
\mathbf { h } _ { \mathrm { a g g } } = \mathbf { g } _ { \mathrm { t s } } \odot \mathbf { h } _ { \mathrm { t s } } + \mathbf { g } _ { \mathrm { r u l e } } \odot \mathbf { h } _ { \mathrm { r u l e } } + \mathbf { g } _ { \mathrm { l l m } } \odot \mathbf { h } _ { \mathrm { l l m } } ,\tag{10}
$$

where $\mathbf { g } _ { k } \in \mathbb { R } ^ { D }$ is the input-dependent gating vector for embed ding $\mathbf { h } _ { k } ,$ , with � ∈ {ts, rule, llm}. Each gating vector is computed as $\mathbf { g } _ { k } = \sigma ( \mathbf { W } _ { k } \mathbf { h } _ { k } )$ , where ${ \bf W } _ { k } \in \mathbb { R } ^ { D \times D }$ contains learned parameters and $\sigma ( \cdot )$ denotes the sigmoid function. The operator ⊙ denotes element-wise multiplication. Each gate assigns a factor between zero and one to each embedding dimension, allowing the contributions of the three representations to adapt to individual input windows. Factors approaching zero suppress the corresponding dimensions, and factors approaching one retain them. The aggregated embedding $\mathbf { h } _ { \mathrm { a g g } }$ is subsequently used for SoH prediction.

## 3.4 SoH Forecastin<sub>g</sub> and Model Trainin<sub>g</sub>

The aggregated embedding $\mathbf { h } _ { \mathrm { a g g } }$ is passed to an MLP prediction head to generate the future SoH trajectory over � cycles as,

$$
\hat { \mathbf { y } } _ { t + 1 : t + H } = \mathbf { W } _ { \mathrm { o u t } } \mathsf { R e L U } \left( \mathbf { W } _ { \mathrm { h i d } } \mathbf { h } _ { \mathrm { a g g } } + \mathbf { b } _ { \mathrm { h i d } } \right) + \mathbf { b } _ { \mathrm { o u t } } ,\tag{11}
$$

where ${ \bf W } _ { \mathrm { h i d } }$ $\mathbf { W _ { \mathrm { o u t } } }$ and their corresponding biases are learnable parameters, and $\hat { \mathbf { y } } _ { t + 1 : t + H } \in \mathbb { R } ^ { H }$ . The model is trained using a composite objective that combines prediction accuracy with constraints on the temporal behaviour of the predicted trajectory,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { m o n o } } \mathcal { L } _ { \mathrm { m o n o } } + \lambda _ { \mathrm { s m o o t h } } \mathcal { L } _ { \mathrm { s m o o t h } } , } \end{array}\tag{12}
$$

where $\lambda _ { \mathrm { m o n o } }$ and $\lambda _ { \mathrm { s m o o t h } }$ control the contributions ofthe two penalty terms. The base term uses a horizon-weighted SmoothL1 (Huber) loss that assigns larger weight to predictions farther into the future, reflecting the increasing dificulty of longer-horizon forecasting.

$$
\mathcal { L } _ { \mathrm { b a s e } } = \frac { 1 } { B H } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { H } w _ { j } \mathsf { S m o o t h L } \mathsf { 1 } _ { \beta } ( \hat { y } _ { i , j } , y _ { i , j } ) , w _ { j } = \left( \frac { j } { H } \right) ^ { \alpha } ,\tag{13}
$$

where � is the batch size, $\beta$ is the transition threshold of the SmoothL1 loss, and $\alpha > 0$ controls the weighting across the prediction horizon. Larger weights are assigned to later prediction steps, which encourages the model to maintain forecasting accuracy as the prediction distance increases. Since battery SoH generally declines with aging, a monotonicity penalty discourages increases between consecutive predictions, and we have,

$$
\mathcal { L } _ { \mathrm { m o n o } } = \frac { 1 } { B ( H - 1 ) } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { H - 1 } \operatorname* { m a x } \left( 0 , \hat { y } _ { i , j + 1 } - \hat { y } _ { i , j } \right) .\tag{14}
$$

Electrochemical aging accumulates slowly over multiple cycles, so that battery SoH typically changes gradually. A smoothness penalty

discourages large changes between consecutive predictions with,

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { B ( H - 1 ) } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { H - 1 } \left| \hat { y } _ { i , j + 1 } - \hat { y } _ { i , j } \right| .\tag{15}
$$

Together, these loss terms encourage accurate forecasts with monotonically and gradually decreasing SoH trajectories.

## 4 Ex<sub>p</sub>eriments

This section evaluates the forecasting performance of Sera. We first describe the experimental setup, followed by the main results and further analyses.

## 4.1 Ex<sub>p</sub>erimental Setu<sub>p</sub>

We describe the dataset and sample construction, implementation settings of Sera, and comparison variants.

4.1.1 Dataset and Forecasting Sample Construction. We evaluate Sera on the MIT Battery Degradation Dataset [14], which contains 124 commercial LFP/graphite cells cycled to end-of-life under fast charging conditions at $3 0 ^ { \circ } \mathrm { C } .$ . The tested cells are A123 Systems APR18650M1A cells with a nominal capacity of 1.1 Ah and a nominal voltage of 3.3 V. The dataset provides cycle-level measurements of capacity, voltage, current, temperature, and internal resistance throughout the battery degradation process.

The batteries are divided by cell identity into 70%, 15%, and 15% for training, validation, and testing, respectively, ensuring that no cell appears in more than one set. Forecasting samples are generated from each battery trajectory using sliding windows. For a battery trajectory of length � , a valid sample requires both an �-cycle observation window and its subsequent �-cycle forecasting target. The �-th sample is constructed as,

$$
\mathcal { W } _ { i } = ( \mathbf { X } _ { i : i + L - 1 } , \mathbf { y } _ { i + L : i + L + H - 1 } ) ,\tag{16}
$$

where � = 1+�� for nonnegative integers � satisfies $i \le T { - } L { - } H { + } 1$ and � is the sliding stride. We set $w = 1$ , advancing the window by one cycle and producing $N = T - L - H + 1$ samples for each battery with $T \geq L + H$ . For example, with $L = 3 0$ and $H = 9 0$ , the first sample uses cycles 1–30 to predict cycles 31–120.

We fix the look-back length at $L = 3 0$ cycles and evaluate prediction horizons $H \in \{ 3 0 , 5 0 , 7 0 , 9 0 \}$ , keeping the observed history unchanged. The relative forecasting distance is characterized by the forecast-to-observation ratio,

$$
\rho = { \frac { H } { L } } .\tag{17}
$$

The four forecasting settings correspond to $\rho \in \{ 1 . 0 , 1 . 6 7 , 2 . 3 3 , 3 . 0 \}$ Increasing � extends the forecast farther beyond the observation window, allowing us to assess forecasting robustness across prediction distances under a fixed history length.

4.1.2 Sera Configurations. For the temporal branch, PatchTST is used with patch length $P = 5$ and representation dimension $D = 1 2 8 .$ The Transformer module of PatchTST $L _ { \mathrm { e n c } }$ contains three encoder layers with four attention heads. For the semantic branches, the rule- and LLM-based representations use separate encoders with the same schema, where the intermediate projection dimension $d _ { \mathrm { p r o j } }$ is set to 64. The resulting semantic embeddings are mapped to the same dimension � before adaptive gated aggregation. All models are trained using Adam with a learning rate of $1 0 ^ { - 3 }$ and a batch size of 32. Training is performed for up to 80 epochs with gradient clipping at 1.0 and early stopping based on the validation loss with a patience of 10 epochs. For the training objective, the horizon weighting parameter is set to $\alpha = 1 . 0$ , while $\lambda _ { \mathrm { m o n o } } = 0 . 1$ and $\lambda _ { \mathrm { s m o o t h } } = 0 . 0 5$ . The SmoothL1 threshold $\beta$ is set to 0.01. The same training configuration is used across all prediction horizons for consistent evaluation.

Table 2: Com<sub>p</sub>arison of forecastin<sub>g p</sub>erformance in terms <sub>o</sub>f t<sub>es</sub>t MAE <sub>an</sub>d RMSE <sub>across</sub> dif<sub>eren</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> h<sub>or</sub>i<sub>zons.</sub> L<sub>ower</sub> <sub>va</sub>l<sub>ues</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> b<sub>e</sub>tt<sub>er</sub> <sub>per</sub>f<sub>ormance,</sub> <sub>w</sub>ith th<sub>e</sub> b<sub>es</sub>t <sub>re-</sub> <sub>su</sub>lt<sub>s</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>e</sub>d i<sub>n</sub> b<sub>o</sub>ld<sub>.</sub> All <sub>va</sub>l<sub>ues</sub> <sub>are</sub> $\times 1 0 ^ { - 3 }$
<table><tr><td></td><td colspan="2"> $H = 3 0$ </td><td colspan="2"> $H = 5 0$ </td><td colspan="2"> $H = 7 0$ </td><td colspan="2"> $H = 9 0$ </td></tr><tr><td>Model</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>TS</td><td>1.556</td><td>2.309</td><td>2.317</td><td>2.942</td><td>3.043</td><td>3.818</td><td>3.565</td><td>4.727</td></tr><tr><td>Rule only</td><td>11.384</td><td>12.904</td><td>12.064</td><td>13.578</td><td>10.962</td><td>12.174</td><td>11.641</td><td>13.172</td></tr><tr><td>LLM only</td><td>12.157</td><td>14.225</td><td>12.806</td><td>14.755</td><td>11.526</td><td>13.296</td><td>12.150</td><td>14.515</td></tr><tr><td> $\mathrm { T S } + \mathrm { R u l e }$ </td><td>0.912</td><td>1.369</td><td>1.883</td><td>2.463</td><td>2.350</td><td>3.135</td><td>2.593</td><td>3.349</td></tr><tr><td> $\mathrm { T S } + \mathrm { L L M }$ </td><td>1.408</td><td>2.036</td><td>2.108</td><td>2.744</td><td>2.522</td><td>3.202</td><td>2.946</td><td>3.869</td></tr><tr><td>SERA (ours)</td><td>1.015</td><td>1.323</td><td>1.688</td><td>2.122</td><td>2.142</td><td>2.851</td><td>2.234</td><td>2.971</td></tr></table>

4.1.3 Comparison Models. We compare Sera with several comparison models to evaluate the contribution of diferent semantic sources. TS uses only the time series or temporal representation and serves as the forecasting baseline. TS + Rule augments the temporal representation with rule-based semantics, while $\mathrm { T S } + \mathrm { L L M }$ incorporates LLM-based semantics. Sera represents the complete model, which aggregates both rule- and LLM-based semantic representations with the temporal representation. We also include Rule only and LLM only to examine whether semantic representations can support forecasting without temporal information. All variants use the same input features, data partitions, and training settings for consistent comparison.

To examine whether the proposed semantic representation and aggregation approach depends on a particular temporal model, we further evaluate Sera with PatchTST, a standard Transformer (TFM) [16], and LSTM as the temporal encoders. Forecasting performance is measured using MAE and root mean squared error (RMSE). A single NVIDIA RTX 4090 GPU is used.

## 4<sub>.</sub>2 F<sub>o</sub>r<sub>ecas</sub>tin<sub>g</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> An<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

We report the experimental results and evaluate the forecasting performance of Sera under the following settings.

4.2.1 Overall Forecasting Performance. We first evaluate whether incorporating structured degradation semantics improves SoH forecasting beyond the temporal representation alone. Table 2 presents the results of Sera, the temporal baseline, and diferent semantic configurations across four prediction horizons, using PatchTST as the temporal encoder. Overall, Sera achieves the lowest RMSE at all tested horizons and the lowest MAE for long horizons with � ∈ {50, 70, 90}. Compared with baseline TS, Sera reduces MAE from 1.6 to 1.0, 2.3 to 1.7, 3.0 to 2.1, and 3.6 to $2 . 2 ~ ( \mathrm { a l l } \times 1 0 ^ { - 3 } )$ at $H = 3 0 , 5 0 , 7 0$ , and 90, respectively. These results indicate that structured degradation semantics complement the temporal representation with explicit text-based degradation information, helping the model better capture the evolving battery state and produce more accurate SoH predictions. An exception occurs at $H = 3 0$ where TS + Rule achieves the lowest MAE of $0 . 9 \times 1 0 ^ { - 3 }$ , compared with $1 . 0 \times 1 0 ^ { - 3 }$ for Sera. One possible explanation is that the temporal and rule-based representations already capture the dominant degradation information needed for shorter prediction horizon, while the additional LLM representation afects the eficiency of the forecasting module. As the prediction horizon increases, the comprehensive information provided by both semantic sources becomes more useful, and Sera achieves the best MAE and RMSE.

![](images/9105ca5bd77e69e0c6a20060e94906f9c1d3b8225b9954091528ae7ff4c6e818.jpg)

![](images/3c54ef91836d3093fdfb7253b5059f77ce140840549c187580e3e33f1bd4727e.jpg)  
(a) Semantic contribution  
(b) Generalization gap  
Fi<sub>gu</sub>r<sub>e</sub> 2: S<sub>e</sub>m<sub>a</sub>nti<sub>c co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s</sub> <sub>across pre</sub>di<sub>c</sub>ti<sub>on</sub> h<sub>or</sub>i<sub>zons.</sub> Fi<sub>g.</sub> 2<sub>a: re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>n</sub> t<sub>es</sub>t MAE <sub>re</sub>l<sub>-</sub> ative to the TS baseline. Fi<sub>g</sub>. 2b: <sub>g</sub>eneralization <sub>g</sub>a<sub>p,</sub> measured <sub>as</sub> th<sub>e</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>es</sub>t <sub>an</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> MAE<sub>.</sub>

The results also clarify the role of semantic information. Both TS + Rule and TS + LLM generally improve upon TS, showing that each semantic source provides useful degradation guidance. In contrast, Rule only and LLM only produce MAE values from 0.011 to 0.014, substantially higher than the models that utilize temporal information. This implies that semantic descriptors alone do not retain suficient information about the absolute SoH level and datacentric trajectory dynamics. They therefore serve more efectively as complementary guidance to the temporal representation than as standalone predictors.

4.2.2 Semantic Contribution and Generalization. We further examine the contribution of semantic information to forecasting and the model’s generalization to unseen batteries. Fig. 2a reports the percentage reduction in test MAE relative to the TS baseline, calculated as $R = \left( { \mathrm { M A E } _ { \mathrm { T S } } } { - } { \mathrm { M A E } _ { \mathrm { v a r i a n t } } } \right) / { \mathrm { M A E } _ { \mathrm { T S } } } { \times } 1 0 0 \% { \mathrm { ~ A } }$ larger positive value indicates a greater relative improvement over TS. Aligned with the results in Table 2, the contribution of semantic information varies with the prediction horizon. At the short horizon � = 30, TS + Rule achieves the largest MAE reduction of 41.4%, compared with 34.8% for Sera, suggesting that rule-based semantics already provide efective guidance for short term forecasting. For the relative longer horizons $H \in \{ 5 0 , 7 0 , 9 0 \}$ , Sera consistently outperforms both individual semantic configurations, and its MAE reduction increases from 27.1% to 29.6% and then to 37.3%. Its advantage over TS + Rule is 8.4%, 6.8%, and 10.1% across these horizons, respectively.

Table 3: Com<sub>p</sub>arison of test MAE across tem<sub>p</sub>oral encoders at $H = 3 0$ <sub>an</sub>d $H = 9 0 .$ <sub>.</sub> L<sub>ower va</sub>l<sub>ues are</sub> b<sub>e</sub>tt<sub>er, an</sub>d th<sub>e</sub> b<sub>es</sub>t <sub>resu</sub>lt<sub>s are</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>e</sub>d i<sub>n</sub> b<sub>o</sub>ld<sub>.</sub> All <sub>va</sub>l<sub>ues are</sub> $\times 1 0 ^ { - 3 }$
<table><tr><td rowspan="3">Model</td><td colspan="2">LSTM</td><td colspan="2">TFM</td><td colspan="2">PatchTST</td></tr><tr><td> $H = 3 0$ </td><td> $H = 9 0$ </td><td> $H = 3 0$ </td><td> $H = 9 0$ </td><td> $H = 3 0$ </td><td> $H = 9 0$ </td></tr><tr><td>TS</td><td>1.394</td><td>3.578</td><td>1.599</td><td>3.421</td><td>1.556</td><td>3.565</td></tr><tr><td> $\mathrm { T S } + \mathrm { R u l e }$ </td><td>1.505</td><td>3.097</td><td>1.375</td><td>2.708</td><td>0.912</td><td>2.593</td></tr><tr><td> $\mathrm { T S } + \mathrm { L L M }$ </td><td>1.535</td><td>3.336</td><td>2.013</td><td>3.246</td><td>1.408</td><td>2.946</td></tr><tr><td>SERA (ours)</td><td>1.229</td><td>3.098</td><td>1.350</td><td>2.636</td><td>1.015</td><td>2.234</td></tr></table>

These results suggest that combining rule- and LLM-based semantics and temporal representation becomes increasingly beneficial as the prediction horizon extends.

Fig. 2b provides a diferent view through the gap between test and training MAE. A gap near zero suggests generalized performance on training and unseen batteries and a positive gap may indicate overfitting. The TS baseline shows a larger gap as � increases, reaching $1 . 7 \times 1 0 ^ { - 3 }$ at $H = 9 0$ , whereas Sera maintains a relatively small gap across all horizons. A possible reason is that structured semantics capture degradation characteristics shared across batteries, reducing the model’s dependence on battery specific temporal patterns. The results suggest that semantic information supports both forecasting accuracy and generalization to unseen batteries.

4.2.3 Temporal Encoder Sensitivity. We further examine the impact of temporal encoder choice. Table 3 compares the TS baselines and corresponding comparison models using LSTM, TFM, and PatchTST as the temporal encoder at $H = 3 0$ and $H = 9 0$ . Although incorporating semantic representations improves forecasting across all three temporal encoders, the gains vary with the temporal encoder and prediction horizon. With LSTM as the temporal encoder, Sera achieves the lowest MAE at $H = 3 0$ and performs nearly identically to TS + Rule at $H = 9 0 .$ . With TFM, Sera achieves the lowest MAE at both horizons. A similar benefit is observed with PatchTST, where Sera reduces the MAE from 3.6×10<sup>−3</sup> for TS to $2 . 2 \times 1 0 ^ { - 3 }$ at � = 90. These results suggest that the benefit of semantic representations is not specific to a particular temporal encoder and that semantic information can be efectively aggregated with diferent temporal representations. Meanwhile, the overall forecasting performance also varies across temporal encoders, with Sera generally achieving better performance with PatchTST than with LSTM and TFM. The varying gains further indicate that the efectiveness of semantic information depends partly on the temporal representations with which it is aggregated.

4.2.4 Case Study. To complement the aggregate forecasting results, Fig. 3 illustrates the full SoH trajectories of two representative batteries with short and long lifespans. The trajectories provide a detailed view of prediction behavior across degradation stages that is not reflected by aggregate error metrics. In both cases, Sera remains close to the ground truth SoH during gradual degradation and accelerated decline stages. Specifically, for the short-lifespan case in Fig. 3a, the trajectories predicted by diferent SoH models remain close during the early cycles. However, after approximately

![](images/fee93bb9b201409f079aba9a6a2a262bef481b24298083f1b3c3833c7f29f893.jpg)

![](images/836f04b2793674a4bad82457d7ec7ea304230adc2357392ae61c9908a46d622e.jpg)  
(a) Short cycle battery  
(b) Long cycle battery  
Figure 3: Qualitative comparison ofSoH trajectories for repre sentative short- (Fig. 3a) and long-lifespan (Fig. 3b) batteries. The results compare the ground truth trajectories with predi<sub>c</sub>ti<sub>ons</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>empora</sub>l b<sub>ase</sub>li<sub>ne, seman</sub>ti<sub>c var</sub>i<sub>an</sub>t<sub>s, an</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s w</sub>ith t<sub>empora</sub>l <sub>an</sub>d <sub>seman</sub>ti<sub>c represen</sub>t<sub>a</sub>ti<sub>ons.</sub>

300 cycles, TS gradually deviates from the ground truth and increasingly overestimates SoH, while Sera follows the accelerated decline more closely. For the long-lifespan battery in Fig. 3b, the main diference occurs between approximately 800 and $1 { , } 6 0 0$ cycles, where TS underestimates SoH while Sera better captures the gradual decline. These patterns illustrate how incorporating degradation semantics can help the model capture diferent degradation behaviors across battery life stages. As also shown in the figures, the Rules and LLM curves fall below the ground truth during the middle stage in Fig. 3a, while exhibiting larger fluctuations and a later transition to accelerated degradation in Fig. 3b. These deviations suggest that semantic descriptors alone do not fully capture the numerical SoH trajectory, supporting their use to guide rather than replace the temporal representation in Sera.

## 4.3 Inter<sub>p</sub>retabilit<sub>y</sub> Anal<sub>y</sub>sis

Beyond forecasting accuracy, it is important to interpret the influ ence of structured semantic representations on the predictions of Sera and we perform counterfactual semantic analysis.

4.3.1 Counterfactual Semantic Analysis. Counterfactual analysis examines whether changes in the semantic representation lead to expected changes in the forecast while the observed time series remains unchanged. This allows us to isolate the influence of semantic information from that of the temporal input and assess whether Sera uses the semantic descriptors consistently with their degradation meaning. Given an input window X and its ruleand LLM-based semantic representations ${ \mathbf z } ^ { \mathrm { r u l e } }$ and $\mathbf { z } ^ { \mathrm { { l l m } } }$ , the original prediction is,

$$
\begin{array} { r } { \hat { \mathbf { y } } = f ( \mathbf { X } , \mathbf { z } ^ { \mathrm { r u l e } } , \mathbf { z } ^ { \mathrm { l l m } } ) . } \end{array}\tag{18}
$$

We then modify one or multiple rules in both semantic representations while keeping X fixed, producing counterfactual representations ${ \mathbf z } ^ { \mathrm { r u l e ^ { \prime } } }$ and $\mathbf { z } ^ { \mathrm { { l l m } ^ { \prime } } }$ . Applying the same intervention to both representations maintains a consistent hypothetical degradation state. The intervention changes the corresponding prediction to,

$$
\hat { \mathbf { y } } ^ { \prime } = f ( \mathbf { X } , \mathbf { z } ^ { \mathrm { r u l e ^ { \prime } } } , \mathbf { z } ^ { \mathrm { l l m ^ { \prime } } } ) .\tag{19}
$$

Here, the intervention efect at prediction step $t ^ { \prime }$ is measured by,

$$
\Delta _ { t ^ { \prime } } = \hat { y } _ { t + t ^ { \prime } } ^ { \prime } - \hat { y } _ { t + t ^ { \prime } } , \quad t ^ { \prime } \in \{ 1 , . . . , H \} .\tag{20}
$$

A positive value indicates an increase in predicted SoH after the intervention, and a negative value indicates a decrease. For example, changing the value of the Trend rule from stable in ${ \mathbf z } ^ { \mathrm { r u l e } }$ and $\mathbf { z } ^ { \mathrm { { l l m } } }$ to accelerating\_fade in ${ \mathbf z } ^ { \mathrm { r u l e ^ { \prime } } }$ and $\mathbf { z } ^ { \mathrm { { l l m } ^ { \prime } } }$ is expected to lower predicted SoH, as battery degradation typically accelerates near the battery’s end of life with low SoH. If the predicted $\hat { \mathbf { y } } ^ { \prime }$ with such intervention decreases relative to the predicted yˆ without intervention, the response agrees with the expectation based on battery domain knowledge. We call this agreement semantic consistency and use it to interpret the alignment of the model’s response with the rule’s degradation meaning.

We evaluate 11 interventions covering individual semantic fields, degradation slope, and coherent optimistic or pessimistic combinations. For each prediction horizon, each intervention is evaluated over $n _ { \mathrm { c f } } = 2 0 0$ randomly sampled counterfactual test windows. We report semantic consistency as the percentage of windows whose predictions change in the expected direction. We also report the mean prediction shift at the final cycle of the horizon as,

$$
\bar { \Delta } _ { H } = \frac { 1 } { n _ { \mathrm { c f } } } \sum _ { i = 1 } ^ { n _ { \mathrm { c f } } } \Delta _ { H } ^ { i } ,\tag{21}
$$

where $\Delta _ { H } ^ { i }$ is the SoH shift for the �-th sampled window. The sign of $\bar { \Delta } _ { H }$ shows the average SoH response direction, while $| \bar { \Delta } _ { H } |$ measures the magnitude of the intervention efect.

4.3.2 Semantic Consistency. Table 4 reports the semantic consistency and final-cycle shift in predicted SoH for each intervention. We first consider the optimistic bundle intervention, which jointly modifies multiple semantic rules toward a healthier degradation state, including a stable trend, low risk, a stable mechanism, and a mild SoH decline. Conversely, the pessimistic bundle intervention modifies these rules toward a worsening degradation state, including an accelerating trend, high risk, Li plating, and a steep SoH decline. These bundled interventions assess whether the resulting prediction changes align with the intended degradation semantics. The optimistic intervention increases predicted SoH and achieves 69.0–90.5% semantic consistency across the prediction horizons, while the pessimistic intervention decreases predicted SoH and achieves 100% consistency. Both interventions generally shift SoH predictions in the expected direction based on domain knowledge, with stronger consistency observed for the pessimistic bundle. One possible explanation is that worsening degradation semantics are more compatible with the degradation trajectory already reflected in the observed time series, whereas imposing healthier rules may partially conflict with this temporal evidence.

To distinguish the contributions of individual rules within the bundles, we next examine interventions that modify one semantic rule at a time. Among the individual interventions, Slope → Steep has the largest efect on the predicted SoH and achieves 100% semantic consistency across all horizons. It also yields negative $\bar { \Delta } _ { H }$ values, indicating lower predicted SoH when the slope is intervened toward steeper degradation. The mean SoH decrease grows from 6.9% at $H = 3 0$ to 25.5% at $H = 9 0$ , indicating a stronger efect at longer prediction horizons as degradation efects accumulate over time. Trend → Accelerating also achieves high semantic consistency of 87.0–100.0%, but produces smaller SoH decreases of 0.25–1.23% only.

T<sub>a</sub>bl<sub>e</sub> 4<sub>:</sub> C<sub>oun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s o</sub>f dif<sub>eren</sub>t <sub>seman</sub>ti<sub>c</sub> i<sub>n-</sub> terventions. Higher semantic consistency (%) indicates that more <sub>p</sub>redictions chan<sub>g</sub>e in the ex<sub>p</sub>ected direction after interventions. Positive (negative) $\bar { \Delta } _ { H }$ <sub>va</sub>l<sub>ues</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> <sub>an</sub> i<sub>ncrease</sub> (decrease) in predicted SoH, with larger absolute values indicating stronger e<sup>f</sup>ects. $\bar { \Delta } _ { H }$ <sub>va</sub>l<sub>ues are</sub> i<sub>n un</sub>it<sub>s o</sub>f $1 0 ^ { - 3 }$
<table><tr><td></td><td colspan="2"> $H = 3 0$ </td><td colspan="2"> $H = 5 0$ </td><td colspan="2"> $H = 7 0$ </td><td colspan="2"> $H = 9 0$ </td></tr><tr><td>Semantic Edit</td><td> $\%$ </td><td> $\bar { \Delta } _ { H }$ </td><td> $\%$ </td><td> $\bar { \Delta } _ { H }$ </td><td>%</td><td> $\bar { \Delta } _ { H }$ </td><td>%</td><td> $\bar { \Delta } _ { H }$ </td></tr><tr><td>Optimistic Bundle</td><td>90.5</td><td>+5.8</td><td>69.0</td><td>+7.5</td><td>86.5</td><td>+12.9</td><td>70.5</td><td>+11.9</td></tr><tr><td>Pessimistic Bundle</td><td>100.0</td><td>-71.3</td><td>100.0</td><td>-116.8</td><td>100.0</td><td>-146.1</td><td>100.0</td><td>-251.3</td></tr><tr><td>Slope → Mild</td><td>75.5</td><td>+5.6</td><td>84.5</td><td>+7.5</td><td>82.0</td><td>+13.2</td><td>71.0</td><td>+12.1</td></tr><tr><td>Slope → Steep</td><td>100.0</td><td>-68.9</td><td>100.0</td><td>-129.7</td><td>100.0</td><td>-142.3</td><td>100.0</td><td>-255.1</td></tr><tr><td>Trend → Stable</td><td>78.5</td><td>+0.9</td><td>93.0</td><td>+2.2</td><td>91.0</td><td>+4.2</td><td>61.0</td><td>+7.7</td></tr><tr><td>Trend → Accel.</td><td>87.0</td><td>-2.5</td><td>94.5</td><td>-3.9</td><td>87.5</td><td>-5.6</td><td>100.0</td><td>-12.3</td></tr><tr><td>Mech. → Stable</td><td>86.0</td><td>+1.0</td><td>64.0</td><td>-0.2</td><td>44.5</td><td>-2.8</td><td>26.5</td><td>-3.0</td></tr><tr><td>Mech. → SEI</td><td>93.5</td><td>-1.1</td><td>32.0</td><td>+0.0</td><td>22.5</td><td>+0.6</td><td>83.5</td><td>-2.5</td></tr><tr><td>Mech. → Li Plating</td><td>99.5</td><td>-3.8</td><td>51.0</td><td>+1.6</td><td>99.5</td><td>-9.2</td><td>99.0</td><td>-13.4</td></tr><tr><td>Risk → Low</td><td>50.5 19.0</td><td>-1.3</td><td>39.5</td><td>-1.4</td><td>85.5</td><td>+4.4</td><td>66.5</td><td>+3.1</td></tr><tr><td> $\mathrm { R i s k } \to \mathrm { H i g h }$ </td><td></td><td>+2.6</td><td>34.5</td><td>-0.1</td><td>58.5</td><td>+0.2</td><td>37.0</td><td>-0.3</td></tr></table>

These results indicate that, under the tested settings, the quantitative slope intervention has a stronger influence on the prediction than the categorical trend intervention. We also test mechanism interventions. For example, Mechanism → Li Plating produces neg ative $\bar { \Delta } _ { H }$ values and high semantic consistency in most cases, as Li plating represents a detrimental degradation mechanism associated with accelerated capacity loss. For risk interventions, Risk → High achieves only 19.0–58.5% semantic consistency and produces relatively small $\bar { \Delta } _ { H }$ values, suggesting that its influence may depend more strongly on the battery state encoded in the temporal representation. Overall, the counterfactual analysis demonstrates that the learned semantic information meaningfully influences the SoH forecast and provides an interpretable characterization of the efects of diferent degradation semantics on the predictions.

## 5 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

This paper presents Sera, a framework for battery SoH forecasting that integrates structured text-based degradation semantics derived from battery domain knowledge with numerical battery time series. Sera employs a three-branch architecture that independently learns a temporal representation and rule- and LLM-based semantic rep resentations, which are fused through adaptive gated aggregation for SoH forecasting. Compared with the temporal baseline, Sera reduces SoH forecasting errors by up to 37.3% and improves generalization to unseen batteries. Sera also achieves consistent forecasting gains across diferent temporal encoders. Counterfactual semantic analysis provides insight into the influence of degradation semantics by examining the direction and magnitude of prediction changes under controlled interventions. The results show more consistent responses to semantic rules that directly characterize degradation progression. Overall, Sera improves SoH forecasting accuracy and interpretability by incorporating explicit degradation knowledge and semantics into temporal modelling efectively.

## A<sub>c</sub>k<sub>now</sub>l<sub>e</sub>d<sub>gmen</sub>t<sub>s</sub>

This research is supported by A\*STAR under its MTC Individual Research Grants (IRG) (Award M23M6c0113), the Shanghai Sci-tech

Co-research Program (Award 25HB2702600), and MTC Programmatic (Award M23L9b0052 and Award M24N6b0043).

## R<sub>e</sub>f<sub>erences</sub>

[1] Xingguang Chen, Tao Sun, Xin Lai, Yuejiu Zheng, and Xuebing Han. 2024. Transfer learning strategies for lithium-ion battery capacity estimation under domain shift diferences. Journal of Energy Storage 90 (2024), 111860.

[2] Samuel Filgueira da Silva, Mehmet Fatih Ozkan, Faissal El Idrissi, and Marcello Canova. 2026. Conformalized Transfer Learning for Li-ion Battery State of Health Forecasting under Manufacturing and Usage Variability. arXiv preprint arXiv:2603.24475 (2026).

[3] Chenglong Duan, Hung Le, and Dazhong Wu. 2025. Lithium-ion battery stateof-health estimation using intra-domain and cross-domain transfer learning: mitigating domain shift based on Wasserstein distance. Journal ofEnergy Storage 132 (2025), 117601.

[4] Xinyu Gu, Khay Wai See, Penghua Li, Kangheng Shan, Yunpeng Wang, Liang Zhao, Kai Chin Lim, and Neng Zhang. 2023. A novel state-of-health estimation for the lithium-ion battery using a convolutional neural network and transformer model. Energy 262 (2023), 125501.

[5] MM Kabir and Dervis Emre Demirocak. 2017. Degradation mechanisms in Li-ion batteries: a state-of-the-art review. International Journal ofEnergy Research 41, 14 (2017), 1963–1986.

[6] Dhivya Dharshini Kannan, Wei Li, Wei Zhang, Jianbiao Wang, Zhi Wei Seh, and Man-Fai Ng. 2026. When smaller wins: Dual-stage distillation and paretoguided compression of liquid neural networks for edge battery prognostics. In International Conference on Pattern Recognition. Springer, 414–429.

[7] Wei Li, Wei Zhang, Yi Xie, Yonggang Wen, and Qingyu Yan. 2026. Operando Thermal Reconstruction for Lithium-Ion Batteries via Physics-Informed Generative Difusion Models. IEEE Transaction on Transportation Electrification (TTE) 12, 1 (2026), 1869–1879.

[8] Tianfeng Long, Pengcheng Zhang, Xiaoqi Liu, Huaqing Shang, Meiling Yue, Xuesong Shen, and Jianwen Meng. 2025. Improved SOH prediction of lithium-ion batteries based on multi-dimensional feature analysis and transformer framework. IEEE Open Journal of Vehicular Technology (2025).

[9] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730 (2022).

[10] Tsuyoshi Oji, Yanglin Zhou, Song Ci, Feiyu Kang, Xi Chen, and Xiulan Liu. 2021. Data-driven methods for battery soh estimation: Survey and a critical analysis. IEEE Access 9 (2021), 126903–126916.

[11] Yuan Qiu, Wei Li, Wei Zhang, Yi Zhou, Fang Liu, Jianbiao Wang, and Zhi Wei Seh. 2026. Interpretable Battery Aging without Extra Tests via Neural-Assisted Physics-based Modelling. arXiv preprint arXiv:2604.01229 (2026).

[12] Jiantao Qu, Feng Liu, Yuxiang Ma, and Jiaming Fan. 2019. A neural-networkbased method for RUL prediction and SOH monitoring of lithium-ion battery. IEEE access 7 (2019), 87178–87191.

[13] Concetta Semeraro, Mariateresa Caggiano, Abdul-Ghani Olabi, and Michele Dassisti. 2022. Battery monitoring and prognostics optimization techniques: Challenges and opportunities. Energy 255 (2022), 124538.

[14] Kristen A Severson, Peter M Attia, Norman Jin, Nicholas Perkins, Benben Jiang, Zi Yang, Michael H Chen, Muratahan Aykol, Patrick K Herring, Dimitrios Fraggedakis, et al. 2019. Data-driven prediction of battery cycle life before capacity degradation. Nature energy 4, 5 (2019), 383–391.

[15] Wen Yang Tan,Jiawei Li, Fang Liu, Wei Zhang, Sumei Sun, Peng Cheng Wang, and Elisa YM Ang. 2026. TIDE: Trustworthy and Interpretable Battery Degradation Estimation with Contextual Learning and Symbolic Distillation. arXiv preprint arXiv:2607.14640 (2026).

[16] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[17] Jens Vetter, Petr Novák, Markus Robert Wagner, Claudia Veit, K-C Möller, JO Besenhard, Martin Winter, Margret Wohlfahrt-Mehrens, Christoph Vogler, and Abderrezak Hammouche. 2005. Ageing mechanisms in lithium-ion batteries. Journal ofpower sources 147, 1-2 (2005), 269–281.

[18] Rui Xiong, Linlin Li, and Jinpeng Tian. 2018. Towards a smarter battery management system: A critical review on battery state of health monitoring methods. Journal ofPower Sources 405 (2018), 18–29.

[19] Bo Yang, Yucun Qian, Qiang Li, Qian Chen, Jiyang Wu, Enbo Luo, Rui Xie, Ruyi Zheng, Yunfeng Yan, Shi Su, et al. 2024. Critical summary and perspectives on state-of-health of lithium-ion battery. Renewable and Sustainable Energy Reviews 190 (2024), 114077.

[20] Zhuang Ye and Jianbo Yu. 2021. State-of-health estimation for lithium-ion batteries using domain adversarial transfer learning. IEEE Transactions on Power Electronics 37, 3 (2021), 3528–3543.