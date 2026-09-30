# State transport routing for short-horizon adaptation in multi-horizon photovoltaic forecasting

Xu Yuqing<sup>a</sup>, Zhou Liguo<sup>a</sup>, Sun Ze<sup>a</sup>, Yu Lei<sup>a</sup>, Jiang Mingming<sup>a</sup>

<sup>a</sup>Huaibei Normal University, China

## Abstract

Recent power measurements provide valuable information for photovoltaic (PV) power forecasting, but directly extrapolating short-term trends can introduce substantial errors over longer forecast horizons. To address this challenge, we propose state transport routing (STR), a lightweight adapter that refines the predictions of a frozen forecasting model. STR combines the original forecast with two complementary trajectories derived from the latest measured power level and its recent trend. A horizon-conditioned router adjusts their contributions over the first 120 min, while leaving subsequent predictions unchanged. Experiments on four public PV datasets show that STR consistently outperforms a parameter-matched residual adapter. On PVDAQ, the same approach improves five neural forecasting backbones, reducing all-horizon normalized mean absolute error by 0.0201–0.2364 percentage points, with paired 95% confidence intervals excluding zero. No reliable improvement is observed for LightGBM. These findings demonstrate the potential of structured state adaptation to improve short-term forecasting across diferent neural architectures without retraining the underlying models or altering their longer-horizon predictions.

Keywords: Photovoltaic power forecasting, Multi-horizon forecasting, State transport routing, Forecast adaptation, Forecast post-processing

## 1. Introduction

Accurate photovoltaic (PV) power forecasting is essential for power system operation and energy management. For example, day-ahead work has combined on-site weather observations with numerical forecasts [? ], while ultra-short-term work has used historical power and meteorological data [?

]. Recent power measurements provide valuable information about the current operating state, allowing forecasting models to respond to rapid changes in power generation. However, short-term fluctuations may not persist over longer horizons, and directly extrapolating recent trends can lead to substantial prediction errors [1, 2]. Therefore, multi-horizon forecasting models need to incorporate recent operating information while preserving the reliability of their longer-term predictions.

Recent forecasting models capture temporal dynamics in diferent ways. N-HiTS and TimeMixer use hierarchical interpolation and multiscale mixing, respectively, while PatchTST and iTransformer employ attention-based representations [3–6]. Simpler architectures, such as DLinear and TSMixer, ofer alternative trade-ofs between forecasting accuracy and computational cost [7, 8]. Rather than modifying these forecasting architectures, this study focuses on adapting the output of an already trained model: recent power measurements are used to refine its short-horizon predictions, while longerhorizon predictions remain unchanged.

Simple state-based corrections provide useful short-term information, but each has clear limitations. Persistence keeps the forecast close to the latest observation, while a local-slope extrapolation extends the recent trend into the future. The former ignores subsequent evolution, whereas the latter may overextend a transient ramp. A generic residual adapter ofers greater flexibility, but does not explicitly distinguish between these diferent forecasting assumptions. This motivates an adaptation mechanism that can select among them at each forecast horizon while limiting its influence to the short term.

To address this problem, we propose state transport routing (STR), which adapts the predictions of a pretrained forecasting model without modifying its parameters. STR considers three candidate trajectories: the original model prediction, the latest observed power level, and an extrapolation of the recent power trend. A horizon-conditioned router combines these trajectories and learns an additional residual correction for the first 120 min. Beyond this interval, the predictions remain identical to those of the original model. STR operates directly on the forecast output using causally available information, without requiring access to the backbone’s internal representations.

We evaluate STR by addressing three research questions: (RQ1) Does explicit state routing provide an advantage over a conventional residual correction? (RQ2) Can the same STR design improve diferent frozen neural forecasting models? (RQ3) How does its efectiveness vary across forecast horizons, model architectures, and operating conditions?

The main contributions of this study are as follows:

1. We propose STR, a lightweight adapter that combines the original model forecast with two trajectories derived from recent power measurements. Horizon-conditioned routing adjusts their contributions over the first 120 min, while leaving subsequent predictions unchanged.

2. We evaluate STR on four public PV datasets and compare it with a nominally parameter-matched residual adapter. The results show that explicitly incorporating state trajectories improves forecasting accuracy beyond the conventional residual correction used in this comparison.

3. We investigate the applicability of STR to five diferent neural forecasting models under a common PVDAQ protocol. We also examine its limitations, including the lack of a reliable improvement for Light-GBM and the performance degradation observed under low-volatility conditions in Solar-Energy.

## 2. Related work

## 2.1. Multi-horizon PV forecasting

Multi-horizon PV forecasting requires models to capture both the daily variation in solar generation and short-term changes caused by weather and operating conditions. Solar geometry provides information about the daily generation pattern [9, 10], while weather forecasts can provide additional information about future conditions.

To model these dynamics, previous PV forecasting studies have explored Transformer-based and recurrent architectures [11–14]. Other forecasting models, including N-HiTS, TimesNet, TimeMixer, TiDE and TimeXer, employ hierarchical, multiscale or exogenous-variable modeling strategies [3, 4, 15–17]. Recent studies in Energy have also investigated stacking and temporal-scale decoupling to improve forecasting performance [18, 19].

While these studies focus on improving the forecasting models themselves, the present work considers how their predictions can be adapted after training. Rather than developing another forecasting architecture, STR uses recent power measurements to refine short-horizon predictions while preserving the original forecasts at longer horizons.

## 2.2. Forecast combination and short-range state information

Recent power measurements provide a simple basis for short-term forecasting. Persistence assumes that the latest observed power level remains unchanged, while trend extrapolation extends recent power variations into the future [1, 2]. Although these approaches can capture the current operating state, their assumptions may become less reliable as the forecast horizon increases.

Forecast combination and stacking ofer another way to improve prediction accuracy by exploiting complementary forecasts [20–22]. Similar strategies have been explored in PV forecasting through combinations of neural and tree-based models [23, 24].

Rather than combining multiple independently trained forecasting models, STR combines the output of a single frozen backbone with two simple trajectories derived from recent power measurements. Their contributions are adjusted according to the forecast horizon, and the resulting correction is restricted to the short-term prediction interval.

## 2.3. Output adaptation and positioning of STR

PV forecasting has used error-correction stages alongside a primary predictor [23]. Such a stage can refine an output without changing the primary model. Stacking, by contrast, combines several complete forecasts [18, 20, 21]. Our comparison uses a single frozen predictor and an additive residual adapter as its control; that control does not expose distinct assumptions about the latest level or trend.

STR keeps this output interface and adds two state trajectories to the frozen forecast. A horizon-conditioned router combines the three paths and adds a residual correction through 120 min; later outputs return the backbone prediction exactly.

To examine the contribution of the explicit state trajectories, we compare STR with a nominally parameter-matched residual adapter under the same experimental protocol.

## 3. Method

## 3.1. Problem formulation and design objective

At each forecast origin t, the available power measurements are denoted by $\mathbf { x } _ { t } ~ = ~ ( x _ { t - L + 1 } , \dots , x _ { t } )$ , where L is the historical input length. Let $\mathbf { c } _ { t }$ represent the covariates available at the time of prediction. A pretrained forecasting model, referred to as the backbone, generates an H-step forecast:

$$
\mathbf { b } _ { t } = f _ { \theta } ( \mathbf { x } _ { t } , \mathbf { c } _ { t } ) .\tag{1}
$$

The backbone parameters θ are kept fixed throughout adaptation. STR operates on its predictions rather than modifying the forecasting model itself. Its objective is to incorporate recent power information into short-horizon predictions while preserving the original forecast at longer horizons.

## 3.2. Candidate state transports

STR constructs three candidate trajectories: the frozen backbone forecast and two state-derived paths that carry the latest observed power forward, either unchanged or according to its recent trend. For forecast step h, they are defined as

$$
p _ { t , h } ^ { ( 1 ) } = b _ { t , h } ,\tag{2}
$$

$$
p _ { t , h } ^ { ( 2 ) } = x _ { t } ,\tag{3}
$$

$$
p _ { t , h } ^ { ( 3 ) } = x _ { t } + h \frac { x _ { t } - x _ { t - 3 } } { 3 } .\tag{4}
$$

The first trajectory retains the backbone prediction, which describes the future evolution learned from historical data and available covariates. The second assumes that power remains at its latest observed level, whereas the third extends the recent power trend into the future. Together, these trajectories provide complementary alternatives for adapting the original forecast.

The slope-based trajectory is a simple extrapolation of recent observations rather than a physical model of power ramps. Here, h denotes the number of forecast steps, so the extrapolation follows the native sampling interval of each dataset.

To determine how these trajectories should be combined, STR uses a local-state descriptor $\mathbf { u } _ { t }$ containing the latest 12 power measurements, recent diferences and local variability. Station identity is also included for pooled datasets. The forecast horizon is represented separately by a learned embedding.

The available information difers across datasets. For GEFCom, the descriptor additionally includes the first two forecast-predictor vectors and their change. On PVDAQ, solar geometry and archived GFS forecasts are supplied to the frozen backbone, whereas the core router uses recent power information and the horizon embedding. Gatton and Solar-Energy use power history without additional weather inputs. Thus, direct weather conditioning is not required for the core STR formulation.

## 3.3. Horizon-conditioned routing and exact bypass

The local-state descriptor is first projected into a hidden representation and combined with the horizon embedding:

$$
\mathbf { z } _ { t , h } = \mathrm { G E L U } ( W _ { u } \mathbf { u } _ { t } + \mathbf { e } _ { h } ) .\tag{5}
$$

A linear output layer then produces three routing logits $\mathbf { a } _ { t , h } \in \mathbb { R } ^ { 3 }$ and an additive residual $d _ { t , h }$ . The routing weights are obtained using a softmax function, $\pmb { \alpha } _ { t , h } = \mathrm { s o f t m a x } ( \mathbf { a } _ { t , h } )$

For horizons within the adaptation window, the final prediction is given by

$$
\hat { y } _ { t , h } = \sum _ { j = 1 } ^ { 3 } \alpha _ { t , h } ^ { ( j ) } p _ { t , h } ^ { ( j ) } + d _ { t , h } , \qquad h \le K .\tag{6}
$$

The routing weights determine the contribution of each candidate trajectory at every forecast step, while the residual term provides an additional learned correction.

Adaptation is restricted to the first 120 min. For all subsequent horizons, STR directly returns the original backbone prediction:

$$
\widehat { y } _ { t , h } = b _ { t , h } , \qquad h > K .\tag{7}
$$

Here, K is the forecast step corresponding to 120 min at the native sampling interval. This design allows STR to adjust short-horizon predictions without changing the backbone output for the third and fourth hours. The complete architecture is illustrated in Fig. 1.

## 3.4. Matched mechanistic control

To examine whether the explicit state trajectories contribute to the improvement, we compare STR with a residual adapter using the same descriptor, hidden projection, nominal parameter count and optimization settings. The control removes the candidate-state combination and instead learns an additive correction to the backbone forecast:

$$
\hat { y } _ { t , h } ^ { \mathrm { c o n t r o l } } = b _ { t , h } + d _ { t , h } , \qquad h \leq K .\tag{8}
$$

Both methods use the same long-horizon bypass defined in Eq. (7). This comparison isolates the contribution of the explicit trajectories within the implemented adapter architecture. Because the routing logits are inactive in the control, matching the nominal parameter count does not imply identical efective capacity for every possible residual-adapter design.

![](images/a8e5f3cf660430633229634e810cb470f6fe3081e37eb07b5ad68ec8bdc8d035.jpg)  
Figure 1: Overview of STR. (a) The backbone forecast and recent power observations are used to construct three candidate trajectories. (b) A horizon-conditioned router combines these trajectories and adds a residual correction. (c) Predictions are adapted within the first 120 min, while subsequent backbone outputs are preserved exactly. The illustrated trajectories and routing weights are schematic.

## 3.5. Backbones and optimization

STR is trained on top of a pretrained forecasting backbone, whose parameters remain fixed during adaptation. For GEFCom, we use a compact forecast-conditioned convolutional network with dilation rates of 1, 2, 4 and 8. Gatton, Solar-Energy and PVDAQ use TimeMixer as the backbone. The corresponding STR adapters contain 2,312, 1,628, 2,780 and 1,204 trainable parameters, respectively. For the three TimeMixer-based models, these account for only 0.047%, 0.077% and 0.034% of the backbone parameter counts.

All adapters are trained for 24 epochs using AdamW, with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 }$ and a batch size of 256. Gradient clipping is applied with a maximum norm of one. We evaluate the models on the selection set every two epochs and retain the checkpoint with the lowest MAE.

To account for training variability, we use three random seeds (2021– 2023). GEFCom, Gatton and Solar-Energy include three backbone seeds, whereas PVDAQ uses a single pretrained backbone checkpoint with three independently trained adapters. Therefore, the variation across PVDAQ runs reflects adapter training rather than variability from retraining the entire forecasting model.

## 3.6. Backbone interface and transfer protocol

STR operates on the forecast output of a pretrained model, using only its prediction vector $\mathbf { b } _ { t } \in \mathbb { R } ^ { H }$ and causally available state information. It does not require access to the backbone’s hidden representations, and the backbone parameters remain frozen during adaptation.

To examine its applicability across diferent forecasting architectures, we apply the same three-trajectory design and routing mechanism to each backbone. A separate STR adapter is trained for each model, without sharing learned router weights between architectures. This experiment therefore evaluates whether the STR design can be reused across diferent models, rather than whether a trained adapter can be transferred directly between them.

The common PVDAQ evaluation protocol is described in the following section.

## 4. Experimental design

## 4.1. Datasets and chronological partitions

Table 1 defines the four tasks. Native targets and normalizations are retained because capacity metadata and sampling structure difer across datasets.

The four datasets are divided chronologically into training, selection and test sets. For GEFCom, the training period covers April 2012 to September 2013, followed by a selection period from October to December 2013 and a test period in June 2014. Gatton uses January–June 2020 for training, July 2020 for selection and September–October 2020 for testing. Solar-Energy uses the first 70% of its 52,560 ten-minute source rows for training and the final 20% for testing. The intervening 10% is divided into three equal chronological blocks; Table 1 reports the first as selection, while the other two are reserved for the source protocol’s inner and outer fusion stages. For PVDAQ 2107, the corresponding periods are 2019–2022, 2023 and January–October 2024.

Solar-Energy’s table counts are eligible (series, forecast-origin) windows, not raw time rows. Each window has 144 history steps and 24 target steps, with origins spaced six rows (one hour) apart. The training block contains

Table 1: Public datasets and frozen forecasting tasks. Counts are retained training/selection/test forecast-origin windows, one per series and origin.
<table><tr><td>Dataset</td><td>Series Step Horizons</td><td></td><td>Origins Available</td><td>information</td></tr><tr><td>GEFCom</td><td>3 60 min</td><td>4</td><td>34,458/5,793/1,887 Power, calendar,</td><td>12 issued predictors</td></tr><tr><td>Gatton</td><td>1 15 min</td><td>16</td><td>16,897/2,961/5,613 Power history;</td><td>3.275 MW nameplate</td></tr><tr><td>Solar-Energy</td><td>137 10 min</td><td></td><td>24 32,743/39,593/239,613 Per-series power</td><td>history</td></tr><tr><td>PVDAQ 2107</td><td>1 15 min</td><td></td><td>16 139,869/17,197/29,211 Power, solar</td><td>geometry, archived GFS</td></tr></table>

836,385 eligible windows; the frozen stratified cap of 239 per series retains 32,743 for fitting. Selection and test retain all 39,593 and 239,613 eligible windows, respectively. The window counts therefore do not follow the rawrow split percentages.

Dataset-specific preprocessing and evaluation conventions are retained. Gatton’s one-minute measurements are aggregated into 15-minute means using complete intervals, following a sign audit performed on the training data only. Because verified plant capacities are unavailable for Solar-Energy, performance is evaluated using train-standardized macro MAE. For PVDAQ, power is normalized by the reported 893 kW DC nameplate capacity, and the daylight metric is restricted to solar elevations above 5<sup>◦</sup>.

## 4.2. Mechanism evaluation

To assess the contribution of the three candidate trajectories, we compare STR with a residual adapter on all four datasets. Both methods use the same frozen backbone predictions, state descriptor, training procedure and longhorizon bypass. The residual adapter removes the trajectory combination and instead learns an additive correction to the original forecast. Their nominal parameter counts are also matched.

This comparison examines whether explicit state routing provides an advantage over the residual correction used in this study, independently of

STR’s performance relative to other forecasting models.

## 4.3. Backbone-transfer evaluation

To investigate whether STR can be applied to diferent forecasting architectures, we evaluate it on five neural backbones using the same PV-DAQ dataset and experimental protocol. These include TimeMixer, N-HiTS, TSMixer, DLinear and iTransformer. All models use the same power history, solar geometry, historical GFS information and archived future GFS forecasts. LightGBM is included to examine whether the adaptation approach also benefits a tree-based forecasting model. Its input features are derived from the same raw information. All backbone parameters and predictions are kept fixed throughout the experiment.

For each backbone, we independently train an STR adapter and a nominally parameter-matched residual adapter. Both receive the frozen 16-step forecast and the latest 12 power observations. The STR architecture, routing window (K = 8), optimizer, learning rate, batch size and 24-epoch training schedule are kept unchanged across models. We use the same three random seeds (2021–2023), with a separate adapter trained for each backbone and seed. The chronological data split, 29,211 test origins and evaluation masks are also identical across experiments. All adapters are frozen before their test predictions are generated, and no backbone-specific tuning is performed to improve the transfer results.

The backbone scores in this experiment difer slightly from those reported in the external benchmark because the two comparisons use diferent seed-aggregation procedures. Here, TSMixer, iTransformer and DLinear use their original selected seed-2021 checkpoints, with baseline nMAE values of 4.9492%, 4.9934% and 5.7695%, respectively. In contrast, the external benchmark reports the mean performance of three independently trained backbone seeds, giving 4.9427%, 5.0252% and 5.7550%. Both sets of results are retained, as they refer to the same data preprocessing and forecast origins but diferent aggregation procedures.

## 4.4. External benchmark

To assess the forecasting performance of STR against existing methods, we conduct a common benchmark across four public PV datasets. The comparison includes 13 learning-based methods: LightGBM, XGBoost, Cat-Boost, N-HiTS, TimeMixer, STR, PatchTST, iTransformer, TimeXer, TiDE, TimesNet, DLinear and TSMixer. Within each dataset, all methods follow the same chronological data split and use identical forecast origins, history lengths, prediction targets and available input information.

Seven additional neural architectures are implemented using the oficial Time-Series-Library repository at commit 4e938a1, while XGBoost and Cat-Boost use their released Python packages. Learning rates and model checkpoints are selected exclusively on the selection set. All newly trained methods are evaluated with three random seeds (2021–2023), and all reported results are obtained from executed models rather than copied from published studies.

AMPDNet and PV-Client are included in the GEFCom comparison, where their available implementations support the corresponding data interface. Dataset-specific naive forecasts are also retained as reference methods. The frozen backbone and residual adapter are used only for the mechanism evaluation and are not included in the external benchmark rankings.

The GEFCom H0–H3 weather experiment is reported separately, as it involves retraining six methods with a common expanded weather-input interface. Inference latency is reported separately for CPU-based tree models and GPU-based neural models to avoid direct comparisons across diferent hardware configurations.

## 4.5. Metrics and paired uncertainty

For datasets with capacity-normalized targets, forecasting error is measured using normalized mean absolute error (nMAE):

$$
\mathrm { n M A E } _ { h } = \frac { 1 } { N _ { h } } \sum _ { t = 1 } ^ { N _ { h } } \frac { | P _ { t , h } - \hat { P } _ { t , h } | } { C } .\tag{9}
$$

Here, $P _ { t , h }$ and $\hat { P } _ { t , h }$ denote the observed and predicted power at forecast horizon $h .$ , respectively. $N _ { h }$ is the number of evaluated predictions, and C is the corresponding normalization capacity. Gatton uses a 3,275 kW nameplate denominator for its net-export proxy, and PVDAQ uses the reported 893 kW DC nameplate for measured AC output. GEFCom supplies already normalized power, so its error is MAE on the published unit-scale target; we do not infer an unreported physical capacity for its three zones.

For Solar-Energy, each of the 137 series is centered and divided by its own training-period standard deviation. We calculate MAE within each series at each horizon and then give the series equal weight. Its train-standardized macro MAE is therefore not capacity-normalized nMAE. Let $E _ { d , h }$ denote the resulting dataset-specific scaled MAE: published normalized-target MAE for GEFCom, capacity-normalized MAE for Gatton and PVDAQ, and trainstandardized macro MAE for Solar-Energy. The reported all-horizon error is the unweighted mean of $E _ { d , h }$ . The tabulated score $1 0 0 ( 1 - E _ { d } )$ is a descriptive transformation; Solar-Energy’s 86.517% is not an engineering accuracy relative to plant capacity. Errors are compared within each dataset, not as interchangeable physical percentages across datasets.

We use paired temporal resampling to estimate uncertainty in the diferences between forecasting methods. On PVDAQ, circular seven-day blocks are sampled from the 305 test dates, with the same dates retained for each paired comparison. For the external benchmark, a reported model mean first scores each seed’s predictions on the same daylight origin–horizon mask and then averages those seed errors. The table’s mean diference is the comparator mean minus the STR mean. The paired diference instead averages each model’s seed predictions at every origin and horizon, scores those mean predictions on the same mask, and subtracts STR error from comparator error. Both diferences are multiplied by 100 for display. In symbols, with e(p) denoting the masked, equal-horizon MAE, these estimands are

$$
\Delta _ { \mathrm { m e a n } } = 1 0 0 \left[ \frac { 1 } { S _ { c } } \sum _ { s } e ( { \bf p } _ { c , s } ) - \frac { 1 } { S _ { \mathrm { S T R } } } \sum _ { s } e ( { \bf p } _ { \mathrm { S T R } , s } ) \right] ,\tag{10}
$$

$$
\Delta _ { \mathrm { p a i r } } = 1 0 0 \left[ e ( \overline { { \bf p } } _ { c } ) - e ( \overline { { \bf p } } _ { \mathrm { S T R } } ) \right] .\tag{11}
$$

The external paired intervals use 5,000 circular seven-day block draws. The candidate backbones have three independently selected seed checkpoints, whereas the three STR adapter seeds share one frozen TimeMixer checkpoint. Thus the pairing is by forecast origin and mask, not by matched checkpoint seed. Since absolute error is nonlinear, the two diferences can even have opposite signs. The backbone-transfer experiment likewise compares three-seed mean adapter predictions with each matched frozen backbone and three-seed residual control; positive baseline-minus-STR or control-minus-STR diferences favor STR.

For the weather experiment, paired confidence intervals are obtained from 4,000 block-resampling draws after averaging predictions across the three seeds.

To examine performance under diferent operating conditions, recentpower volatility is calculated as the mean absolute first diference over the latest 12 observations. The thresholds used to define volatility groups are determined from the training data.

## 4.6. Eficiency

We evaluate the computational overhead introduced by STR in terms of trainable parameters, training time and inference latency. Since the backbone remains frozen, we report the additional parameters and training time required by the adapter separately from those of the complete forecasting system.

Inference latency is measured with a batch size of one, using 30 warm-up runs followed by 200 timed repetitions. The backbone and its STR-enhanced version are evaluated in the same session to measure the additional inference time introduced by the adapter.

Training times are reported according to their original experimental settings. Latency results for GPU-based neural models and CPU-based tree models are presented separately, as their execution times are not directly comparable across diferent hardware configurations.

## 5. Results

## 5.1. Structured routing versus matched residual correction

STR has lower dataset-specific scaled MAE than the nominally parameter-matched residual adapter on all four tasks (Table 2). Each paired 95% interval excludes zero. This comparison isolates the implemented trajectory design from the broader benchmark of complete forecasting methods.

Table 2: STR versus the matched residual control. Entries are dataset-specific scaled MAE multiplied by 100; reductions and intervals use the same units. GEFCom uses source-normalized power, Gatton and PVDAQ use nameplate-normalized MAE, and Solar-Energy uses train-standardized macro MAE. Positive reductions favor STR.
<table><tr><td>Dataset</td><td>STR error (×100)</td><td>Residual (×100)</td><td>Reduction (×100)</td><td>Paired 95% CI (×100)</td></tr><tr><td>GEFCom</td><td>2.3137</td><td>2.5131</td><td>0.1994</td><td>[0.1226, 0.2797]</td></tr><tr><td>Gatton</td><td>5.2732</td><td>5.2946</td><td>0.0214</td><td>[0.0118, 0.0328]</td></tr><tr><td>Solar-Energy</td><td>13.4834</td><td>13.5478</td><td>0.0644</td><td>[0.0493, 0.0803]</td></tr><tr><td>PVDAQ</td><td>4.9255</td><td>4.9504</td><td>0.0250</td><td>[0.0182, 0.0325]</td></tr></table>

Table 3: PVDAQ transfer of the STR formulation. Baselines are matched frozen checkpoints; residual and STR errors are means of three adapter seeds. Positive diferences favor STR. Paired intervals use seed-averaged predictions and can difer from diferences of separately scored seed means.
<table><tr><td colspan="6">Backbone Base (%) Residual (%) STR (%) Added parameters</td></tr><tr><td>TimeMixer</td><td>4.9455</td><td></td><td>4.9504</td><td>4.9255</td><td>1,204</td></tr><tr><td>N-HiTS</td><td>4.8116</td><td></td><td>4.8022</td><td>4.7888</td><td>1,204</td></tr><tr><td>TSMixer</td><td></td><td>4.9492</td><td>4.9387</td><td>4.8950</td><td>1,204</td></tr><tr><td>DLinear</td><td></td><td>5.7695</td><td>5.7381</td><td>5.5330</td><td>1,204</td></tr><tr><td>iTransformer</td><td></td><td>4.9934</td><td>4.9718</td><td>4.8789</td><td>1,204</td></tr><tr><td colspan="6">Non-neural boundary case</td></tr><tr><td></td><td>LightGBM</td><td>4.5405</td><td>4.5412</td><td>4.5421</td><td>1,204</td></tr><tr><td colspan="2">Backbone TimeMixer</td><td>Base-STR (pp)</td><td></td><td>Paired 95% CI Residual-STR (pp)</td><td>Paired 95% CI</td></tr><tr><td colspan="2">N-HiTS</td><td>+0.0201</td><td>[+0.0133,+0.0292]</td><td>+0.0250</td><td>[+0.0188,+0.0328]</td></tr><tr><td colspan="2"></td><td>+0.0229</td><td>[+0.0113,+0.0364]</td><td>+0.0134</td><td>[+0.0075,+0.0199]</td></tr><tr><td colspan="2">TSMixer</td><td>+0.0541</td><td>[+0.0320,+0.0815]</td><td>+0.0437</td><td>[+0.0265,+0.0627]</td></tr><tr><td colspan="2">DLinear</td><td>+0.2364</td><td>[+0.1821,+0.3057]</td><td>+0.2050</td><td>[+0.1588,+0.2638]</td></tr><tr><td colspan="2">iTransformer</td><td>+0.1145</td><td>[+0.0849,+0.1522]</td><td>+0.0929</td><td>[+0.0682,+0.1242]</td></tr><tr><td colspan="4">Non-neural boundary case</td><td>-0.0010</td><td>[-0.0012,+0.0000]</td></tr></table>

## 5.2. Transfer across frozen neural backbones

We next examine whether the same STR design can be used with diferent frozen neural backbones. On PVDAQ, adding STR reduces all-horizon nMAE for TimeMixer, N-HiTS, TSMixer, DLinear and iTransformer (Table 3 and Fig. 2). The corresponding paired 95% confidence intervals exclude zero both against each original backbone and against its matched residual adapter. Improvements in normalized accuracy range from 0.0201 to 0.2364 percentage points. Each backbone uses an independently trained STR adapter; the learned weights are not transferred between models.

We also evaluate LightGBM to examine whether the same approach benefits a tree-based model. Its observed accuracy change is −0.0017 percentage points, with a paired interval of [−0.0023, +0.0002]. Because the interval crosses zero, this experiment does not establish an improvement for Light-GBM.

![](images/cf501bb9d705c54c5bc6ccda01783f00cf3b89fe39408352f1afc1388a41885e.jpg)  
Figure 2: Reduction in all-horizon nMAE after adding STR to each frozen PVDAQ backbone, with paired confidence intervals. Positive values indicate lower error with STR; the dashed line marks zero. LightGBM is shown separately from the five neural backbones.

## 5.3. Horizon-resolved adaptation

The benefits of STR vary with forecast horizon (Table 4). For TimeMixer, most of the gain occurs at 15 and 30 min, while the estimates from 90 to 120 min are slightly negative and their intervals cross zero. N-HiTS improves from 30 to 120 min, although its 15-min interval includes zero. DLinear improves at all eight routed horizons. TSMixer shows larger gains at the earliest steps but negative diferences at 90, 105 and 120 min; for iTransformer, the gain decreases with lead time and its 120-min interval crosses zero. Thus, a reduction in overall error does not imply an improvement at every individual horizon. Only the first eight of the sixteen equally weighted PVDAQ horizons are adapted. Because the remaining eight predictions are unchanged, the all-horizon error reduction is exactly half the mean reduction over the routed horizons, apart from display rounding. This is a direct consequence of the evaluation metric and the bypass rule, not an additional empirical finding.

## 5.4. Exact long-horizon preservation

For all evaluated adapters and seeds, predictions from 135 to 240 min are identical to the corresponding frozen backbone outputs. This confirms

Table 4: Frozen routed-horizon nMAE reductions (percentage points) on PVDAQ. Positive values favor STR. Full paired intervals and relative improvements are retained in the supplementary CSV. These mean-prediction profile estimates are distinct from the meanof-seed metrics in the transfer summary.
<table><tr><td>Backbone</td><td>15</td><td>30</td><td>45</td><td>60</td><td>75</td><td>90</td><td>105</td><td>120</td></tr><tr><td>TimeMixer</td><td>+0.2843</td><td>+0.0472</td><td>+0.0097</td><td>+0.0061</td><td>+0.0021</td><td>-0.0007</td><td>-0.0054</td><td>-0.0073</td></tr><tr><td>N-HiTS</td><td>-0.0082</td><td>+0.0521</td><td>+0.0608</td><td>+0.0528</td><td>+0.0499</td><td>+0.0414</td><td>+0.0498</td><td>+0.0798</td></tr><tr><td>TSMixer</td><td>+0.7846</td><td>+0.2475</td><td>+0.0796</td><td>+0.0010</td><td>-0.0230</td><td>-0.0486</td><td>-0.0671</td><td>-0.0834</td></tr><tr><td>DLinear</td><td>+0.5954</td><td>+0.5048</td><td>+0.5196</td><td>+0.5120</td><td>+0.5182</td><td>+0.4660</td><td>+0.4164</td><td>+0.3155</td></tr><tr><td>iTransformer</td><td>+0.8984</td><td>+0.4369</td><td>+0.2464</td><td>+0.1436</td><td>+0.0806</td><td>+0.0480</td><td>+0.0234</td><td>-0.0021</td></tr><tr><td>LightGBM</td><td>-0.0022</td><td>+0.0013</td><td>+0.0001</td><td>-0.0026</td><td>-0.0007</td><td>-0.0019</td><td>-0.0057</td><td>-0.0057</td></tr></table>

that the implementation follows Eq. (7). The equality is a property of the architecture, not evidence that STR improves third- or fourth-hour accuracy; any errors in the original backbone forecasts are also retained.

## 5.5. External benchmark comparison

Supplementary Table S.1 compares the original TimeMixer-based STR system with 12 other learning methods on each task, using each task’s own error scale. STR ranks third on GEFCom, first on Gatton, third on Solar-Energy and fifth on PVDAQ. LightGBM leads GEFCom and PVDAQ; XG-Boost leads Solar-Energy. These ranks place the adapter among complete forecasting methods without selecting the best adapted backbone after the transfer experiment.

On PVDAQ, the three tree ensembles have the lowest all-horizon nMAE: LightGBM 4.5405%, XGBoost 4.5753% and CatBoost 4.6068% (Supplementary Table S.2). N-HiTS reaches 4.8116%, followed by the original TimeMixer-based STR instance at 4.9255%. TSMixer and TimeMixer reach 4.9427% and 4.9455%. The five adapted backbones in the transfer study are separate matched comparisons, not candidates from which this benchmark selected an STR result.

Supplementary Table S.3 adds origin-paired comparisons. Negative comparator-minus-STR values favor the comparator. LightGBM, XGBoost and CatBoost have intervals below zero; TimeMixer, PatchTST, DLinear and TiDE have intervals above zero. TSMixer and iTransformer change sign between the diference of separately scored seed means and the diference scored after averaging seed predictions. Their paired intervals, like those for TimesNet and TimeXer, cross zero. N-HiTS has a lower mean error, but no frozen paired interval was available.

## 5.6. Eficiency and recorded training cost

Supplementary Table S.2 reports full-system batch-one latency. Among the GPU models, DLinear and TSMixer have the smallest measured medians. In the benchmark timing session, the same frozen TimeMixer checkpoint takes 17.421 ms alone and 18.327 ms with STR, a diference of medians of 0.906 ms (5.2%). A separate transfer timing session on the same RTX A6000 and checkpoint set gives 17.563 and 19.052 ms, respectively, a 1.49 ms (8.5%) diference of medians. Both use batch one, 30 warm-ups and 200 randomized timed calls, but their measurements belong to diferent sessions. Across the five neural backbones in the transfer session, the added full-system median latency is 1.14–1.49 ms. Tree timings use eight CPU threads and are not ranked against GPU timings. The low fitting cost applies to the adapter after a backbone already exists; it does not make complete-system inference free.

Full training-time records and same-session full-system latency comparisons appear in Supplementary Tables S.4 and S.5. The seven newly trained models follow the same fitting protocol, whereas the remaining timings come from earlier recorded sessions. The representative STR adapter fit takes 25.8 s with the TimeMixer backbone already trained and frozen. This measures the cost of fitting the adapter only, not the training time of a complete STR forecasting system.

## 5.7. Boundary and failure analysis

Recent-power volatility modifies the observed efect (Supplementary Table S.6). STR reduces error in all three training-defined groups on GEF-Com, Gatton and PVDAQ. Solar-Energy’s low-volatility group instead has a 0.000407 increase in train-standardized macro MAE; its middle and high groups improve. Thresholds remain training-derived even where many histories have no measured change.

Because confidence intervals were not frozen for the individual volatility groups, Supplementary Table S.6 reports only point estimates and sample counts. In the consecutive-block analysis, STR improves on the residual control in all 18 evaluated blocks. Its comparison with the original backbone is positive in all but one PVDAQ block, where the diference is only −2.6 × $1 0 ^ { - 7 }$ . GEFCom includes one test month, and certified calendar dates are unavailable for Solar-Energy; these results therefore cannot establish stability across seasons.

Several results also limit the scope of the findings. The paired interval for LightGBM crosses zero, and TSMixer shows worse performance at some routed horizons. In the exploratory weather experiment, the added weather information does not produce a stable overall benefit for TimeMixer (Supplementary Section S3). We retain these outcomes under the original split and routing window rather than adjusting the experiment after observing them.

## 6. Discussion

## 6.1. Mechanism and backbone transferability

STR improves on the implemented residual control across four tasks. The control learns b + d, whereas STR exposes the backbone forecast, latest-level persistence and local-slope extrapolation before its additive residual. These paths encode diferent assumptions about how the observed state may develop; their explicit separation may help the router use a transient ramp without forcing the entire correction into one residual. The nominally matched control has inactive routing logits, so this experiment supports the tested design comparison rather than a claim about every possible residual adapter.

Five neural backbones gain in aggregate on the common PVDAQ queue, each with independently fitted STR weights. DLinear gains more than TimeMixer; one possible explanation is that the frozen models leave different amounts of recent-state error for an output adapter to correct. Model capacity itself was not isolated. LightGBM’s paired interval crosses zero, so a reliable improvement is not established for that ensemble. Its feature splits might already capture some relevant interactions, but one tree result cannot define a boundary for all tree models. The external ranks describe the original TimeMixer-based instance and do not change these matched transfer estimates.

## 6.2. Horizon-dependent behavior

Most routed-horizon gains occur early, while TSMixer and some other backbones regress at individual later steps within the routing window. Recent level and slope information may lose value as lead time grows, which is consistent with the mixed horizon profile. The exact bypass at 120 min prevents the adapter from changing 135–240 min predictions. It preserves their existing errors as well as their values; it neither improves third- or fourth-hour accuracy nor establishes 120 min as an optimal cutof for every task.

## 6.3. Computational considerations

The PVDAQ adapter adds 1,204 trainable parameters and has a representative 25.8-s adapter-only fit after backbone training. Full-system inference still runs the backbone. The benchmark session measures 0.906 ms (5.2%) additional TimeMixer median latency, and the separate transfer session measures 1.49 ms (8.5%); the latter session spans 1.14–1.49 ms across the five neural backbones. The added fraction is larger for smaller backbones. These measurements describe computational overhead on the recorded hardware; energy consumption was not measured.

## 6.4. Weather conditioning

DLinear W1 gains under temporal pairing, but one of its three adapter seeds regresses. TimeMixer shows no stable aggregate gain, and W2 and W3 give negative or mixed outcomes. Future weather may help when the frozen backbone leaves errors associated with changing conditions, but these data do not establish that mechanism. The 56 origins with conflicting power and radiation trends ofer an exploratory clue only: weather enters the additive residual as well as the routing logits, and this small subset cannot identify which component caused the change. Weather is therefore an optional research extension, not a requirement of STR.

## 6.5. Limitations and future work

The transfer study tests five neural backbones and LightGBM at one PVDAQ site. Solar-Energy’s low-volatility group regresses, and the four datasets use diferent error scales. Their public test periods were exposed during development, so the intervals do not substitute for a new independent evaluation. Archived forecasts follow the study’s issue-time rules, but historical operational receipt logs are unavailable; valid time and receipt time remain diferent concepts. A prospectively held-out study across multiple sites and seasons, with receipt records and direct energy measurements where eficiency is claimed, is needed to assess deployment and broader transfer.

## 7. Conclusions

We introduced STR to adapt the short-horizon output of a frozen PV forecasting model. It combines the original forecast with trajectories based on the latest measured power and recent power trend, while preserving all predictions beyond 120 min.

STR reduces error relative to the matched residual adapter on four public PV tasks. On PVDAQ, separately trained STR adapters also improve five diferent neural backbones, with positive paired confidence intervals for their aggregate results. These experiments establish reuse of the STR design across the tested backbones, not transfer of trained router weights.

No reliable gain is observed for LightGBM, and some volatility groups and individual horizons show worse results. Direct weather conditioning also provides no consistent improvement across the tested backbones. STR is therefore a short-horizon adaptation option for the neural models evaluated here, rather than a replacement forecasting architecture or a method shown to benefit all model families.

## Data availability

PVDAQ measurements are available through the Open Energy Data Initiative [25], and historical GFS forecasts through the NCAR Geoscience Data Exchange [26]. GEFCom2014 is described by Hong et al. [27]. The reproducibility package documents the Gatton and Solar-Energy source manifests and preprocessing steps. Derived code and prediction tables will be released after the relevant source licenses and repository metadata have been reviewed.

## Funding

This research received no external funding.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

ChatGPT assisted with language revision. The authors reviewed the manuscript and remain responsible for its content.

## References

[1] J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F. Martinez-de Pison, F. Antonanzas-Torres, Review of photovoltaic power forecasting, Solar Energy 136 (2016) 78–111. doi:10.1016/j.solener.2016.06.069.

[2] R. H. Inman, H. T. Pedro, C. F. Coimbra, Solar forecasting methods for renewable energy integration, Progress in Energy and Combustion Science 39 (6) (2013) 535–576. doi:10.1016/j.pecs.2013.06.002.

[3] C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza, M. Mergenthaler-Canseco, A. Dubrawski, N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting, arXiv preprint arXiv:2201.12886 (2022). URL https://arxiv.org/abs/2201.12886

[4] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, J. Zhou, TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting, arXiv preprint arXiv:2405.14616 (2024). URL https://arxiv.org/abs/2405.14616

[5] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, arXiv preprint arXiv:2211.14730 (2022). URL https://arxiv.org/abs/2211.14730

[6] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long, iTransformer: Inverted Transformers Are Efective for Time Series Forecasting, arXiv preprint arXiv:2310.06625 (2023). URL https://arxiv.org/abs/2310.06625

[7] A. Zeng, M. Chen, L. Zhang, Q. Xu, Are Transformers Efective for Time Series Forecasting?, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, 2023, pp. 11121–11128. doi:10.1609/aaai.v37i9.26317.

[8] V. Ekambaram, A. Jati, N. Nguyen, P. Dayama, J. Kalagnanam, TSMixer: An All-MLP Architecture for Time Series Forecasting, arXiv preprint arXiv:2303.06053 (2023). URL https://arxiv.org/abs/2303.06053

[9] I. Reda, A. Andreas, Solar position algorithm for solar radiation applications, Solar Energy 76 (5) (2004) 577–589. doi:10.1016/j.solener.2003.12.003.

[10] W. F. Holmgren, C. W. Hansen, M. A. Mikofski, pvlib python: a python package for modeling solar energy systems, Journal of Open Source Software 3 (29) (2018) 884. doi:10.21105/joss.00884.

[11] G. Piantadosi, S. Dutto, A. Galli, S. De Vito, C. Sansone, G. Di Francia, Photovoltaic power forecasting: A Transformer based framework, Energy and AI 18 (2024) 100444. doi:10.1016/j.egyai.2024.100444.

[12] K. Tao, J. Zhao, Y. Tao, Q. Qi, Y. Tian, Operational day-ahead photovoltaic power forecasting based on transformer variant, Applied Energy 373 (2024) 123825. doi:10.1016/j.apenergy.2024.123825.

[13] J. Kim, J. Obregon, H. Park, J.-Y. Jung, Multi-step photovoltaic power forecasting using transformer and recurrent neural networks, Renewable and Sustainable Energy Reviews 200 (2024) 114479. doi:10.1016/j.rser.2024. 114479.

[14] Y. Ma, F. Li, H. Zhang, G. Fu, M. Yi, Two-stage photovoltaic power forecasting method with an optimized transformer, Global Energy Interconnection 7 (2024) 812–824. doi:10.1016/j.gloei.2024.11.011.

[15] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, M. Long, TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis, in: International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=ju\_Uqw384Oq

[16] A. Das, W. Kong, R. Sen, Y. Zhou, Long-term Forecasting with TiDE: Timeseries Dense Encoder, arXiv preprint arXiv:2304.08424 (2023). URL https://arxiv.org/abs/2304.08424

[17] Y. Wang, H. Wu, J. Dong, G. Qin, H. Zhang, Y. Liu, Y. Qiu, J. Wang, M. Long, TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables, arXiv preprint arXiv:2402.19072 (2024). URL https://arxiv.org/abs/2402.19072

[18] Y. Cao, P. Yong, J. Yu, Z. Yang, Stacking algorithm based framework with strong generalization performance for ultra-short-term photovoltaic power forecasting, Energy 322 (2025) 135599. doi:10.1016/j.energy.2025.135599.

[19] H. Sun, Y. Wan, M. Liu, H. Zhang, SPI-Net: Temporal scale decoupling and physics-inspired calibration network for non-stationary photovoltaic power forecasting, Energy 359 (2026) 141422. doi:10.1016/j.energy.2026.141422.

[20] J. M. Bates, C. W. J. Granger, The Combination of Forecasts, Journal of the Operational Research Society 20 (4) (1969) 451–468. doi:10.1057/jors. 1969.103.

[21] D. H. Wolpert, Stacked generalization, Neural Networks 5 (2) (1992) 241–259. doi:10.1016/s0893-6080(05)80023-1.

[22] L. Breiman, Stacked regressions, Machine Learning 24 (1) (1996) 49–64. doi: 10.1007/bf00117832.

[23] R. Zhang, G. Li, S. Bu, G. Kuang, W. He, Y. Zhu, S. Aziz, A hybrid deep learning model with error correction for photovoltaic power forecasting, Frontiers in Energy Research 10 (2022) 948308. doi:10.3389/fenrg.2022.948308.

[24] G. Xiong, J. Zhang, X. Fu, J. Chen, A. W. Mohamed, Seasonal shortterm photovoltaic power prediction based on gsk–bigru–xgboost considering correlation of meteorological factors, Journal of Big Data 11 (2024) 164. doi:10.1186/s40537-024-01037-x.

[25] Open Energy Data Initiative, PVDAQ, 2023 Solar Data Prize: System 2107, Farm Solar Array, public measurement and metadata collection; accessed September 2026 (2026). URL https://oedi-data-lake.s3.amazonaws.com/pvdaq/ 2023-solar-data-prize/2107\_OEDI/metadata/2107\_system\_metadata. json

[26] NCAR Geoscience Data Exchange, NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive, dataset d084001; accessed September 2026 (2026). doi:10.5065/D65D8PWK.

[27] T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, R. J. Hyndman, Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond, International Journal of Forecasting 32 (3) (2016) 896–913. doi:10.1016/j.ijforecast.2016.02.001.