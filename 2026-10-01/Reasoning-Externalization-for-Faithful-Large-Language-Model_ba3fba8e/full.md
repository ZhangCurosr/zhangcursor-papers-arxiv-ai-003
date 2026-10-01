# Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions

Sujung Kim<sup>1,†</sup>, Seung Hwan Cho<sup>1,†</sup>, Sangjin Park<sup>2,\*</sup> and Young-Min Kim<sup>2</sup>

<sup>1</sup>Department of Industrial Data Engineering, Hanyang University, Republic of Korea

<sup>2</sup>School of Interdisciplinary Industrial Studies, Hanyang University, Republic of Korea

## Abstract

In finance, interpreting machine learning predictions is essential, yet the numerical outputs of explainable AI can be dificult for non-experts to understand. While large language models (LLMs) can translate these outputs into natural language, they may produce errors when inferring numerical changes and feature relations. We propose an LLM narrative framework for cross-sectional stock return prediction that combines temporal Shapley additive explanations (SHAP) evidence with historical regime analogs. Temporal evidence tracks changes in the normalized global SHAP importance of an XGBoost model over six months. Historical analogs are past periods with similar changes in SHAP importance, their model performance and subsequent market returns are provided as comparative context. Using this framework, we conduct a controlled study of progressive reasoning externalization, sequentially providing raw SHAP sequences, deterministic temporal descriptors, and feature relations. Each generated claim is verified against provenance-linked evidence. Across Qwen3, externalizing numerical and relational reasoning improved evidence faithfulness as well as temporal and relational accuracy. Evidence faithfulness increased from 0.696 to 0.996 for Qwen3-32B-Instruct. While historical analogs did not improve structured automatic faithfulness, they received higher human-rated usefulness scores. These results suggest that externalizing verifiable reasoning enhances narrative faithfulness and that historical context adds interpretive value. The code is available at https://github.com/sugenre/reasoning-externalization-xai.

## Keywords

Explainable AI, Large Language Models, Narratives, Cross-sectional Stock Prediction

## 1. Introduction

Machine learning based predictive models are increasingly being adopted in the financial domain; however, their inherently black box nature makes it dificult to understand the rationale underlying their predictions. Model interpretability is considered essential in finance for both investment decision making and regulatory compliance [1], motivating the adoption of explainable AI (XAI) methods. Among the most widely used XAI methods, Shapley additive explanations (SHAP) quantitatively measure the contribution of individual features to model predictions [2]. However, SHAP outputs derived from complex financial data are inherently numerical and can be dificult for non expert users to understand [3]. Translating such quantitative explanations into insights that can support decision making therefore remains a challenging task [4].

To address this limitation, recent studies have used large language models (LLMs) to translate SHAP outputs into natural language explanations, improving understandability [5]. However, returngenerating processes in financial markets vary over time due to business cycles, policy changes, crises, and other structural shifts [6]. Therefore, a SHAP explanation based on data from a single point in time cannot fully convey how the model interprets current market conditions. Historical information on how the model behaved under similar market conditions, and how the market subsequently evolved, can provide valuable context for financial practitioners [7]. Such temporal and historical context has not yet been systematically incorporated into LLM-based explanations. Providing it is itself challenging, as LLMs struggle to reason over time-series data [8]. Narratives generated from this evidence therefore require verification of their faithfulness after generation [9]. This verification is essential in finance, where model reliability directly afects economic outcomes and regulatory compliance [1].

In this study, we propose a framework that provides LLMs with temporal XAI evidence to generate financial analysis narratives in a cross-sectional stock return prediction setting. An XGBoost model is trained on preprocessed CRSP<sup>1</sup> data, and its SHAP values are used to construct temporal trajectories of feature attributions. Historical periods with similar attribution trajectories are then retrieved and provided as historical context. Using this framework, we conduct a controlled study ofhow progressively externalizing numerical and relational reasoning afects narrative faithfulness. The main contributions of this study are as follows.

• Evidence-grounded narrative framework. Temporal SHAP evidence is linked to historical regimes with similar changes in SHAP importance, and each generated claim is verified against a provenance-linked evidence registry.

• Controlled study of reasoning externalization. Deterministic temporal descriptors and feature relations substantially reduce temporal and relational reasoning errors and improve evidence faithfulness across Qwen3 scales.

<sup>•</sup> Role of historical analogs. Historical analogs do not improve structured automatic faithfulness but increase human-rated usefulness, serving as complementary interpretive context.

## 2. Related Work

## 2.1. SHAP Based Feature Analysis in Financial Prediction

SHAP is one of the most widely used XAI methods grounded in game theory [2]. By providing both global attributions for the overall model and local attributions for individual predictions, SHAP has been extensively studied across various domains [10], including financial return prediction. Goswami and Uddin analyzed 166 asset-pricing characteristics and found that momentum and trading related features exhibited high contributions, with portfolio analyses further demonstrating their substantial economic significance [11]. Wang showed that, in neural network models, momentum- and tradingrelated features contributed most strongly to the prediction of abnormal stock returns, whereas investor sentiment features exhibited the highest contributions in predicting excess stock returns [12]. However, these studies primarily conducted static SHAP analyses over the full sample period or at specific points in time.

To address the limitations of static analysis, Lundberg et al. proposed a method that uses SHAP to quantify each feature’s contribution to predictive loss and tracks these contributions over time to identify the causes of model performance degradation [13]. Rather than directly comparing shifts in input distributions, Mougan et al. introduced the concept of Explanation Shift, which detects changes in how a model utilizes features by comparing historical and current SHAP based distributions [14]. Similarly, Jenett et al. applied XGBoost and SHAP to the prediction of Real Estate Investment Trust returns and volatility, showing that the importance of key variables changes across distinct market regimes, including the global financial crisis, the low interest rate period, and the COVID-19 pandemic. They further analyzed nonlinear relationships between explanatory variables and returns using Accumulated Local Efects [6]. These studies examined dynamic changes in SHAP importance primarily from the perspective of monitoring and detecting model changes, but did not exploit temporal patterns in these changes to identify historically similar market regimes or support downstream analysis.

## 2.2. LLM-Based Explanation and Context-Augmented Financial Analysis

Although SHAP based outputs provide detailed model explanations, they can be dificult for non expert users to understand and interpret. To address this issue, Zeng and Zhu proposed a pipeline that organizes SHAP outputs into a structured format and generates natural language explanations through prompt engineering, thereby improving the clarity and usability of model explanations [3]. Zytek et al. introduced Explingo, which consists of a Narrator that translates ML explanations into natural language and a Grader that evaluates the generated explanations, with the aim ofproducing high quality narratives [4]. Martens et al. proposed XAIstories, which transforms SHAP and counterfactual explanations into LLM generated narratives, and showed that these narratives helped users summarize and understand AI decisions more accurately than raw SHAP outputs [5]. Geng et al. found that providing SHAP based feature rankings to an LLM resulted in better performance than allowing the LLM to infer feature importance independently, suggesting that LLMs should be used as controlled narrative interfaces [15]. Beyond SHAP, Wang translated explanations of a temporal graph convolutional network for stock trend prediction into natural language financial reports using LLMs, and evaluated the reports with a factual sensitivity protocol [16]. Nevertheless, concerns remain regarding the faithfulness of generated narratives. Lukassen et al. pointed out that prior studies have predominantly evaluated the textual quality of generated narratives while providing limited validation of their practical usefulness for decision making [17]. Pratama and Tseng further identified fidelity failures in LLM generated credit risk reports, including reversals of SHAP value signs, omission of dominant features, and inclusion of features that were not provided in the underlying evidence [9].

A growing body of research has also investigated the use of contextual information in financial analysis and prediction. Teixeira et al. proposed Labeled Guide Prompting, which combines structured outputs from Bayesian Networks with LLMs to automatically generate credit risk reports [18]. Kim et al. provided GPT-4 with financial statements to predict the direction of future earnings changes and demonstrated that LLMs can generate useful narrative insights regarding future earnings [19]. Fatouros et al. introduced MarketSenseAI 2.0, which integrates news, financial statements, and macroeconomic data within a multiagent LLM architecture to support stock analysis and investment decision making [20]. These studies primarily used textual or numerical information observed at a given point in time as contextual input to LLMs. Other studies have explored the use of historically similar cases as contextual information. Khanna et al. proposed a framework that combines macroeconomic indicators with text embeddings to retrieve similar historical periods and provides the retrieved cases as context for LLM based prediction [7]. In their framework, similarity is measured based on macroeconomic variables, while temporal patterns in feature attributions derived from an ML model are not used.

## 3. Methodology

## 3.1. Data Preparation and Stock Return Prediction

Using monthly CRSP data obtained through WRDS, we constructed 30 stock level characteristics covering price and trend, momentum, volatility and risk, and trading volume and liquidity. Each characteristic was defined as either a month end value or a monthly aggregate according to its economic interpretation and the measurement convention of the underlying raw data. For delisting observations, both regular returns and delisting returns were incorporated. Missing characteristic values were imputed using the monthly cross sectional median, after which the remaining missing values were set to zero following [21]. To mitigate the influence of extreme observations while maintaining consistent preprocessing across time, each characteristic was winsorized using the 1st and 99th percentile thresholds estimated from the training sample, and the same thresholds were subsequently applied to the validation and test periods [22]. Preserving the temporal ordering of the data, the sample was divided into a training period from January 1995 to December 2009, a validation period from January 2010 to December 2011, and a test period from January 2012 to November 2023.

Using the resulting monthly stock characteristics, we formulated a cross sectional stock return prediction task in which the feature vector $\mathbf { x } _ { i , t }$ of stock � observed in month � is used to predict its realized return $r _ { i , t + 1 }$ in the following month. Monthly prediction is a widely adopted setting in cross sectional asset pricing research based on firm characteristics [21] and is also well suited to aligning characteristics with heterogeneous update frequencies at a common time point. Because the primary focus of this study is not short term price fluctuations themselves but rather the model’s feature dependence structure and its temporal variation, both the return prediction task and the subsequent SHAP based temporal analysis were conducted at a monthly frequency.

![](images/8951bbe72284ec57ce802a691c000126bcb8305d66337b3e63ca5e46189a10e6.jpg)  
Figure 1: Overview of the proposed framework for generating evidence-grounded LLM narratives of stock return predictions.

## 3.2. Temporal XAI Evidence Construction

Tree based models can flexibly capture nonlinear relationships and interactions among features in cross sectional return prediction [21], while TreeSHAP enables eficient computation of feature attributions for individual predictions [13]. Let the SHAP value for stock �, feature �, and month � be denoted by $\phi _ { i , j , t }$ The monthly global attribution was computed by first taking the cross sectional mean of the absolute SHAP values for each feature and then normalizing it by the sum of these mean absolute attributions across all features to obtain $P _ { j , t }$ . The resulting $P _ { j , t }$ represents the relative importance of feature $j$ in the model’s global attribution for month �. Let $P _ { t }$ denote the 30 dimensional global attribution vector for month �; the temporal evidence over the most recent six months was then constructed as $S _ { t }$

$$
A _ { j , t } = \frac { 1 } { N _ { t } } \sum _ { i } \left| \phi _ { i , j , t } \right| ,\tag{1}
$$

$$
P _ { j , t } = \frac { A _ { j , t } } { \sum _ { k } A _ { k , t } } ,\tag{2}
$$

$$
S _ { t } = [ P _ { t - 5 } , P _ { t - 4 } , \ldots , P _ { t } ] \in \mathbb { R } ^ { 6 \times 3 0 } .\tag{3}
$$

## 3.3. Historical SHAP Based Analog Augmentation

To augment temporal XAI evidence with historical comparative context, we retrieved past market regimes exhibiting similar patterns of change in feature importance. While raw market variables such as prices and returns can characterize similarity in market states, changes in global SHAP importance reflect how the relative contribution structure of features evolves over time under a fixed predictive model. We therefore retrieved historical regimes with similar SHAP importance dynamics and provided the model’s predictive performance and subsequent universe returns in those periods as comparative context.

Using $P _ { t }$ , we defined the monthly change in attribution as in Equation (4) and used it as the retrieval representation. For a temporal window of length �, the query trajectory at month � was constructed as in Equation (5). Because each $\Delta P _ { t }$ is a 30 dimensional vector, $\Delta \dot { P } _ { t } \in \dot { \mathbb { R } } ^ { 3 0 }$ , the distance between query trajectory ${ Q } _ { t } ^ { ( L ) }$ and historical candidate trajectory ${ Q } _ { s } ^ { ( L ) }$ was computed using Euclidean distance.

$$
\Delta P _ { t } = P _ { t } - P _ { t - 1 } ,\tag{4}
$$

$$
Q _ { t } ^ { ( L ) } = [ \Delta P _ { t - L + 1 } , \dots , \Delta P _ { t } ] ,\tag{5}
$$

$$
d ( t , s ) = \left. \operatorname { v e c } \left( Q _ { t } ^ { ( L ) } \right) - \operatorname { v e c } \left( Q _ { s } ^ { ( L ) } \right) \right. _ { 2 } .\tag{6}
$$

To ensure that the query and candidate trajectories did not overlap and that the outcome following each candidate was observable before the query period, historical candidates were restricted to $s \leq$ $t - ( L + 1 )$ . The top � candidates with the smallest distances were selected as historical analogs. The sensitivity to � and �, as well as the final retrieval configuration, was determined empirically.

Each historical analog was associated with its trajectory distance, monthly RankIC at the corresponding period, and the equal weighted universe return in the subsequent month. These outcomes were not treated as direct forecasts or causal evidence. Rather, they served as comparative historical context indicating the model’s predictive performance and subsequent universe returns during past periods exhibiting similar changes in feature attribution.

## 3.4. Structured Evidence Registry

The generated prediction, feature attribution, temporal descriptors, feature relations, and historical analog evidence were stored in a unified structured evidence registry. Each evidence unit was assigned a unique provenance identifier specifying the case, evidence type, and associated feature or relation. For example, case\_id#TMP:momentum12 denotes the temporal numerical evidence for momentum12 in a given case.

All experimental conditions shared the same underlying registry, while the visibility of each evidence unit was predetermined for each condition. Thus, diferences across conditions were created by varying only the scope of information exposed to the LLM, rather than by recomputing the underlying evidence. The prompt presented each evidence item together with its corresponding identifier, which was subsequently used to verify generated claims against the source evidence.

## 3.5. Reasoning Externalization and Narrative Generation

To control the reasoning burden placed on the LLM during temporal XAI narrative generation, numerical and relational reasoning were progressively externalized. Motivated by findings that delegating computation from the LLM to an external interpreter improves reasoning accuracy [23], these computations were performed deterministically before generation and provided as evidence rather than derived by the LLM. The experimental conditions were designed according to the dependency structure of the required reasoning steps, while the system prompt and output schema were kept unchanged across conditions.

C0 evaluates whether the model suppresses unsupported temporal inference when no temporal evidence is available. Under C1, the LLM must infer temporal trends and feature relations directly from the raw temporal sequence. C2 externalizes deterministic numerical computation by explicitly providing temporal statistics. The six-month attribution change was defined as $\Delta _ { 6 } P _ { j , t } = P _ { j , t } - P _ { j , t - 5 }$ The temporal slope was estimated by OLS over the six monthly attribution values $P _ { j , t - 5 : t }$ , using month indices $0 , \ldots , 5$ . Trend direction was labeled as increasing, decreasing, or flat according to the sign of the six month OLS slope. C3 further provides OVERTAKES relations, thereby removing the burden of relational inference from the LLM. Feature � was defined to overtake feature � when $P _ { a , t - 5 } < P _ { b , t }$ −5 but $P _ { a , t } > P _ { b , t }$ . C3-Analog is a separate augmentation condition that combines C3 with historical context. Each output consisted of natural language text together with structured claims. Every factual claim was required to reference evidence IDs available under the corresponding condition, while the model was allowed to abstain when no supporting evidence was available.

Table 1  
Experimental Conditions and Evidence Provided
<table><tr><td>Condition</td><td>Evidence provided</td></tr><tr><td>C0-Evidence Control</td><td>Prediction, local SHAP, and current global attribution</td></tr><tr><td>C1-Raw</td><td>C0 + raw 6-month global attribution sequence</td></tr><tr><td>C2-Numerical</td><td> ${ \mathsf { C } } 1 + \Delta 6 { \mathsf { P } } ;$  slope, and trend direction</td></tr><tr><td>C3-Relational</td><td>C2 + deterministic OVERTAKES relations</td></tr><tr><td>C3-Analog</td><td>C3 + historical analog evidence</td></tr></table>

The primary model was Qwen3-32B-Instruct [24], the largest dense model that could be executed under a consistent inference protocol in our experimental environment. Qwen3-14B and Qwen3-8B were additionally included to assess whether the observed efects remained consistent across model scales within the same model family, while their corresponding Base checkpoints were used to examine sensitivity to post training. Ministral-8B-Instruct-2410 [25] and Llama-3.1-8B-Instruct [26] were further included to evaluate whether the efects ofreasoning externalization remained consistent across diferent model families.

## 4. Experiments and Results

## 4.1. Evaluation Protocol

## 4.1.1. Prediction Performance Evaluation

Return prediction performance was evaluated using RankIC, RankICIR, RMSE, and MAE. For each month �, RankIC was computed as the cross sectional Spearman correlation between the predicted next-month return $\widehat { r } _ { i , t + 1 }$ and the realized next month return $r _ { i , t + 1 }$ across stocks. We report the average monthly RankIC over the full test period. RankICIR was calculated as the mean monthly RankIC divided by its standard deviation, measuring the stability of cross-sectional ranking performance over time.

$$
\mathrm { R a n k I C } _ { t } = \mathrm { S p e a r m a n } _ { i } ( \widehat { r } _ { i , t + 1 } , r _ { i , t + 1 } ) ,\tag{7}
$$

$$
\mathrm { R a n k I C } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { R a n k I C } _ { t } ,\tag{8}
$$

$$
\mathrm { R a n k I C I R } = \frac { \mathrm { m e a n } ( \mathrm { R a n k I C } _ { t } ) } { \mathrm { s t d } ( \mathrm { R a n k I C } _ { t } ) } .\tag{9}
$$

## 4.1.2. Deterministic Automatic Evaluation

The generated outputs were evaluated by directly matching their structured semantic fields against the evidence registry. Our primary metric, Structured Evidence Faithfulness (EF), was defined as the proportion of factual claims that referenced evidence permitted under the corresponding experimental condition and matched the ground truth subject–predicate–value tuple.

Because the evaluation operates on structured fields generated alongside the natural language narrative, EF measures the consistency between structured factual claims and their supporting evidence rather than the overall semantic truthfulness of unrestricted free form text. Output format reliability was additionally evaluated using the parse rate. For C0, in which no temporal evidence was provided, we further evaluated whether the model appropriately refrained from unsupported inference using Evidence Control Abstention Accuracy.

## 4.1.3. LLM-as-a-Judge

Because the deterministic evaluation verifies only the correspondence between structured claim fields and the evidence registry, without directly assessing the narrative text itself, we conducted a complementary LLM-as-a-Judge evaluation on the C3-Relational and C3-Analog conditions. Using month stratified sampling, we selected a common set of 400 case\_ids and paired the narratives from the two conditions at the case level. Claude Opus 5, which belongs to a diferent model family from the generation models (Qwen, Llama, and Mistral), was used as the judge. Model and condition identifiers were omitted from the evaluation prompts. Each item was evaluated once (� = 1) through the Message Batches API.

The evaluation comprised three criteria, with diferent levels of information exposure for each criterion. First, the judge was shown only the narrative text, without any supporting evidence, and asked to determine whether the normalized global SHAP importance (�) of the focal feature had increased, decreased, or remained unchanged over the preceding six months, or whether the direction could not be determined from the text alone. Textual Temporal Transfer Accuracy was computed as the exact match accuracy between the resulting judgment and trend\_direction in the evidence registry.

For the second and third criteria, the judge was provided with both the narrative and the evidence block corresponding to the relevant condition. Evidence Scope Adherence was defined as the extent to which the narrative avoided claims beyond the scope of the provided evidence, whereas Narrative Synthesis was defined as the extent to which multiple evidence items were coherently integrated into an interpretation while remaining within the supported evidence scope. Both criteria were evaluated using pre defined anchored 1–5 rating rubrics.

## 4.1.4. Human Evaluation

To assess the practical utility of historical analog context, we conducted a human evaluation comparing narratives generated by Qwen3-32B-Instruct under the C3-Relational and C3-Analog conditions. Ten evaluators each assessed two cases. For each case, the evaluator reviewed one C3-Relational narrative and one C3-Analog narrative, resulting in four narrative evaluations per evaluator. Evaluators were non expert users with an interest in stock investment and basic familiarity with artificial intelligence concepts. Condition labels were hidden, and both the case order and the order of the two narratives within each pair were randomized. Evaluators rated each narrative using five criteria: Understandability, Decision Usefulness, Contextual Usefulness, Reliance Calibration, and Adoption Intention, on five point Likert scales. This evaluation was designed to examine whether historical analogs provide additional value for users’ interpretation and use of the narratives, separately from automatic faithfulness evaluation.

## 4.2. Experimental Setup and Prediction Model Selection

All experiments were conducted on Ubuntu 24.04 using an NVIDIA RTX PRO 6000 GPU with 96 GB of memory. XGBoost 3.2.0 and SHAP 0.51.0 were used for prediction and XAI analysis, respectively, while LLM inference was performed using vLLM 0.11.2. The same decoding configuration was applied across all LLM conditions.

Because the primary objective was not to compete for state-of-the-art predictive performance, but rather to evaluate the faithfulness of LLM-generated narratives grounded in XAI evidence derived from a fixed prediction model, we compared two linear baselines, Linear Regression and Ridge Regression, with two tree-based models, Random Forest and XGBoost, which can capture nonlinear relationships and feature interactions while supporting TreeSHAP-based attribution.

The tree based models achieved higher RankIC values than the linear models, suggesting that capturing nonlinear efects and feature interactions may be important for modeling the relationship between financial characteristics and next month returns in this prediction setting. Compared with

Table 2  
Out-of-Sample Prediction Performance
<table><tr><td>Model</td><td>RankIC ↑</td><td>RankICIR↑</td><td>RMSE↓</td><td>MAE↓</td></tr><tr><td>Linear Regression</td><td>0.0170</td><td>0.2380</td><td>0.1631</td><td>0.0845</td></tr><tr><td>Ridge Regression</td><td>0.0209</td><td>0.2810</td><td>0.1631</td><td>0.0848</td></tr><tr><td>Random Forest</td><td>0.0357</td><td>0.4900</td><td>0.1627</td><td>0.0841</td></tr><tr><td>XGBoost</td><td>0.0361</td><td>0.5380</td><td>0.1660</td><td>0.0859</td></tr></table>

Random Forest, XGBoost achieved 1.4% higher RankIC and 9.8% higher RankICIR. However, Random Forest achieved lower RMSE and MAE, indicating that XGBoost did not dominate across all predictive metrics. XGBoost was selected using validation period performance and then fixed for all test period XAI analyses. Test results were used only for final out of sample evaluation. TreeSHAP was applied to interpret feature contributions under this predictive model.

## 4.3. Historical Analog Characterization

Before incorporating historical analogs as contextual evidence, we assessed whether regimes retrieved based on changes in SHAP importance provided useful comparative context for subsequent universe returns. Sensitivity analyses were conducted across temporal window lengths $L \ \in \ \{ 3 , 6 , 9 \}$ and numbers of retrieved analogs $K \in \{ 1 , 3 , 5 \}$

For comparison, we constructed a market state based retrieval method that retained the same retrieval conditions but replaced the representation with changes in the cross sectional market state derived from the raw variables. We additionally evaluated a recent return baseline and a random baseline that selected historical regimes at random. For each method, we compared the aggregated subsequent universe returns of the retrieved historical regimes with the realized next month universe return using mean absolute error (MAE). Table 3 reports the results across the evaluated retrieval configurations.

Table 3  
Mean Absolute Error of Subsequent Universe Returns across Retrieval Settings
<table><tr><td>Method</td><td colspan="3">K = 1</td><td colspan="3">K = 3</td><td colspan="3">K = 5</td></tr><tr><td></td><td> $L = 3$ </td><td> $L = 6$ </td><td> $L = 9$ </td><td> $L = 3$ </td><td> $L = 6$ </td><td> $L = 9$ </td><td> $L = 3$ </td><td> $L = 6$ </td><td> $L = 9$ </td></tr><tr><td>SHAP-Trajectory</td><td>0.05146</td><td>0.05135</td><td>0.04845</td><td>0.04180</td><td>0.04260</td><td>0.04091</td><td>0.04012</td><td>0.03996</td><td>0.04121</td></tr><tr><td>Market-State</td><td>0.04752</td><td>0.04822</td><td>0.04899</td><td>0.04225</td><td>0.04076</td><td>0.04057</td><td>0.04098</td><td>0.03997</td><td>0.04147</td></tr><tr><td>Recent-Return</td><td>0.04544</td><td>0.04128</td><td>0.03972</td><td>0.04544</td><td>0.04128</td><td>0.03972</td><td>0.04544</td><td>0.04128</td><td>0.04042</td></tr><tr><td>Random</td><td>0.04902</td><td>0.04858</td><td>0.04874</td><td>0.04236</td><td>0.04205</td><td>0.04186</td><td>0.04047</td><td>0.04040</td><td>0.04114</td></tr></table>

For SHAP-Trajectory retrieval, MAE generally decreased as the number of retrieved analogs increased. This pattern is consistent with the possibility that aggregating multiple historical analogs reduces sensitivity to any single retrieved period. Among the evaluated SHAP-Trajectory configurations, $K = 5$ and $L = 6$ minimized validation MAE, yielding a value of 0.03996. Accordingly, we selected this configuration for SHAP-Trajectory retrieval before evaluating C3-Analog on the test period.

However, SHAP-Trajectory retrieval was not consistently superior to the alternative methods. The recent return baseline achieved lower MAE than SHAP-Trajectory in six of the nine evaluated configurations. Under the selected $K = 5$ and $L = 6$ configuration, SHAP-Trajectory achieved an MAE of 0.03996, which was nearly identical to the 0.03997 obtained by market-state retrieval. We therefore do not interpret these results as evidence of consistent superiority of SHAP-Trajectory retrieval over the alternative methods. Instead, the retrieved analogs are used as historical comparative context, providing information on model performance and subsequent equal-weighted universe returns during past periods characterized by similar changes in model feature contributions.

## 4.4. Reasoning Externalization and Narrative Faithfulness

Table 4 reports cross-model performance under C1–C3 in terms of evidence faithfulness, temporal accuracy, and relational accuracy, together with output parse rates.

## Table 4

Cross-Model Consistency of Reasoning Externalization
<table><tr><td>Model</td><td>Parse</td><td>EF C1</td><td>EF C2</td><td>EF C3</td><td>Temp. C1</td><td>Temp. C2</td><td>Temp. C3</td><td>Rel. C1</td><td>Rel. C2</td><td>Rel. C3</td></tr><tr><td>Qwen3-8B</td><td>0.9363</td><td>0.7219</td><td>0.7662</td><td>0.9195</td><td>0.6318</td><td>0.8784</td><td>0.8458</td><td>0.2155</td><td>0.3585</td><td>0.9295</td></tr><tr><td>Qwen3-8B-Instruct</td><td>1.0000</td><td>0.8259</td><td>0.9704</td><td>0.9686</td><td>0.3604</td><td>0.6125</td><td>0.5750</td><td>0.8625</td><td>0.8875</td><td>1.0000</td></tr><tr><td>Qwen3-14B</td><td>0.9779</td><td>0.7570</td><td>0.8061</td><td>0.9984</td><td>0.6979</td><td>0.9916</td><td>0.9916</td><td>0.3375</td><td>0.2059</td><td>0.9979</td></tr><tr><td>Qwen3-14B-Instruct</td><td>1.0000</td><td>0.7194</td><td>0.8844</td><td>0.9995</td><td>0.4854</td><td>0.9479</td><td>0.9521</td><td>0.4271</td><td>0.4979</td><td>1.0000</td></tr><tr><td>Qwen3-32B-Instruct</td><td>0.9988</td><td>0.6960</td><td>0.9032</td><td>0.9964</td><td>0.3396</td><td>0.9896</td><td>0.9313</td><td>0.5250</td><td>0.5729</td><td>1.0000</td></tr><tr><td>Ministral-8B-Instruct</td><td>0.5433</td><td>0.8733</td><td>0.8536</td><td>0.9204</td><td>0.5506</td><td>0.6273</td><td>0.5246</td><td>0.7341</td><td>0.8636</td><td>0.9836</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.4525</td><td>0.6555</td><td>0.5762</td><td>0.5437</td><td>0.1905</td><td>0.4020</td><td>0.4098</td><td>0.5450</td><td>0.4975</td><td>0.7639</td></tr></table>

Across all Qwen3 Instruct models, numerical externalization from C1 to C2 increased Structured Temporal Accuracy, with improvements of 0.6500, 0.4625, and 0.2521 for the 32B, 14B, and 8B models, respectively. When relational evidence was provided in C3, Relational Accuracy increased over C2 by 0.4271, 0.5021, and 0.1125, respectively, showing the same directional efect regardless of model scale. Ministral and Llama exhibited relatively low parse rates of 0.5433 and 0.4525, respectively, resulting in diferent numbers of valid samples across conditions; they were therefore excluded from direct quantitative comparisons. The Qwen3 base models also showed increases in Structured Temporal Accuracy from C1 to C2 of 0.2466 for 8B and 0.2937 for 14B, indicating that the efect of numerical externalization was not limited to instruction-tuned models. Meanwhile, Qwen3 Base models generally achieved higher temporal reasoning accuracy than their Instruct counterparts of the same size, but lower relational reasoning accuracy, suggesting that the efect of instruction tuning may difer across reasoning types.

## Table 5

Structured Claim Performance by Reasoning Condition (Qwen3-32B-Instruct)
<table><tr><td>Condition</td><td>EF↑</td><td>Temporal Acc. ↑</td><td>Relational Acc. ↑</td><td>C0 Abstention</td></tr><tr><td>C0-Evidence Control</td><td>0.9104</td><td></td><td></td><td>0.8386</td></tr><tr><td>C1-Raw</td><td>0.6960</td><td>0.3396</td><td>0.5250</td><td></td></tr><tr><td>C2-Numerical</td><td>0.9032</td><td>0.9896</td><td>0.5729</td><td>一</td></tr><tr><td>C3-Relational</td><td>0.9964</td><td>0.9313</td><td>1.0000</td><td>一</td></tr><tr><td>C3-Analog</td><td>0.9891</td><td>0.9000</td><td>1.0000</td><td>一</td></tr></table>

For Qwen3-32B-Instruct, externalizing numerical information from C1 to C2 increased EF from 0.6960 to 0.9032, corresponding to an improvement of 0.2072, while Structured Temporal Accuracy increased from 0.3396 to 0.9896, an improvement of 0.6500. This substantial increase indicates that explicitly providing deterministic temporal descriptors markedly improved the model’s ability to represent temporal changes relative to requiring such information to be inferred from raw attribution sequences. When relational evidence was additionally structured and provided in C3, Relational Accuracy increased from 0.5729 under C2 to 1.0000, corresponding to an improvement of 0.4271. EF also increased from 0.9032 to 0.9964, while Structured Temporal Accuracy remained high at 0.9313. Under the C0-Evidence Control condition, abstention accuracy was 0.8386, indicating that unsupported inference was frequently, though not perfectly, suppressed when temporal evidence was unavailable. C3-Analog maintained perfect Relational Accuracy at 1.0000, while EF and Structured Temporal Accuracy were slightly lower than those under C3-Relational, decreasing from 0.9964 to 0.9891 and from 0.9313 to 0.9000, respectively. These results suggest that historical analog augmentation did not provide an additional improvement in structured automatic faithfulness, motivating a separate evaluation of its narrative level and human level utility.

## 4.5. LLM-as-a-Judge Evaluation

To assess narrative-level faithfulness, we compare C3-Relational and C3-Analog in terms of temporal information transfer, evidence-scope adherence, and narrative synthesis. Table 6 summarizes the results.

## Table 6

LLM-as-a-Judge Evaluation of Narrative Level Faithfulness (Mean ± SD)
<table><tr><td>Metric</td><td>C3-Relational</td><td>C3-Analog</td></tr><tr><td>Textual Temporal Transfer Accuracy</td><td>0.9975</td><td>0.9975</td></tr><tr><td>Deterministic Temporal Accuracy</td><td>0.8625</td><td>0.7425</td></tr><tr><td>Evidence Scope Adherence</td><td> $\mathbf { 4 . 9 3 0 0 \pm 0 . 2 5 5 5 }$ </td><td> $\underline { { 4 . 8 9 0 0 } } \pm 0 . 3 2 1 2$ </td></tr><tr><td>Narrative Synthesis</td><td> $\mathbf { 2 . 0 2 7 5 \pm 0 . 1 6 3 7 }$ </td><td> $1 . 9 9 7 5 \pm 0 . 0 8 6 7$ </td></tr></table>

Textual Temporal Transfer Accuracy was 0.9975 (399/400) for both conditions, indicating that the addition of historical analogs did not materially impair the communication of temporal change information for the focal feature. In contrast, Deterministic Temporal Accuracy was lower, at 0.8625 for C3-Relational and 0.7425 for C3-Analog. Further analysis showed that all 158 cases classified as failures by the deterministic metric lacked a temporal\_trend claim but contained a temporal\_change claim. This discrepancy therefore reflects a diference in claim-type-specific aggregation rather than an actual loss of temporal information in the narrative.

Evidence Scope Adherence remained high in both conditions, at approximately 4.9 out of5, suggesting that the inclusion of historical analogs did not increase the generation of claims beyond the scope of the provided evidence. By contrast, Narrative Synthesis scores were low, at approximately 2 out of 5. This suggests that evidence constrained generation is efective in maintaining faithfulness to the provided evidence, but may limit the flexibility with which multiple evidence items are integrated into a narrative.

## 4.6. Human Evaluation of Historical Analog Context

To assess whether historical analogs improve the perceived usefulness of generated narratives, we compare C3-Relational and C3-Analog across five human-evaluation criteria covering understandability, decision usefulness, contextual usefulness, reliance calibration, and adoption intention. Table 7 summarizes the mean ratings and between-condition diferences.

## Table 7

Human Evaluation of Historical Analog Context (Mean ± SD)
<table><tr><td>Item</td><td>C3-Relational</td><td>C3-Analog</td><td>∆</td></tr><tr><td>Q1 Understandability</td><td> $\underline { { 3 . 2 0 0 0 } } \pm 1 . 2 3 9 7$ </td><td> $\mathbf { 3 . 6 0 0 0 } \pm 1 . 2 7 3 2$ </td><td>+0.4000</td></tr><tr><td>Q2 Decision Usefulness</td><td> $\underline { { 3 . 3 5 0 0 } } \pm 1 . 2 2 5 8$ </td><td> $\mathbf { 3 . 8 0 0 0 } \pm 1 . 1 5 1 7$ </td><td>+0.4500</td></tr><tr><td>Q3 Contextual Usefulness</td><td> $3 . 2 0 0 0 \pm 1 . 5 0 7 9$ </td><td> $\mathbf { 3 . 7 0 0 0 \pm 1 . 2 6 0 7 }$ </td><td>+0.5000</td></tr><tr><td>Q4 Reliance Calibration</td><td> $3 . 2 5 0 0 \pm 1 . 1 6 4 2$ </td><td> $\mathbf { 3 . 6 5 0 0 \pm 1 . 1 3 6 7 }$ </td><td>+0.4000</td></tr><tr><td>Q5 Adoption Intention</td><td> $2 . 8 0 0 0 \pm 1 . 3 6 1 1$ </td><td> $\mathbf { 3 . 3 5 0 0 \pm 1 . 3 8 7 0 }$ </td><td>+0.5500</td></tr><tr><td>Overall</td><td>3.1600</td><td>3.6200</td><td>+0.4600</td></tr></table>

The human evaluation results showed that C3-Analog achieved higher mean scores than C3-Relational across all evaluation criteria. The overall mean increased from 3.1600 to 3.6200, corresponding to a mean diference of +0.4600. Adoption Intention exhibited the largest diference (+0.5500), followed by Contextual Usefulness (+0.5000). These higher mean ratings were observed even though EF and Structured Temporal Accuracy under C3-Analog were slightly lower than those under C3-Relational in the preceding automatic evaluation. This suggests that historical analog information may provide users with useful comparative context for interpreting and potentially using current predictions, even when it does not further improve structured accuracy. Historical analogs should therefore be interpreted primarily as complementary information that enhances human-level contextual usefulness rather than as a mechanism for improving automatic faithfulness.

## 5. Conclusion

This study examined whether the reliability of financial XAI narratives can deteriorate when LLMs are required to directly derive numerical changes and feature relations from raw XAI evidence, and evaluated an approach that externalizes such reasoning burdens into verifiable evidence. SHAP based temporal evidence was structured into numerical and relational information, while factual claims were linked to Evidence IDs, enabling deterministic evaluation of faithfulness and reasoning accuracy without relying on LLM-as-a-Judge.

The results showed that explicitly providing attribution changes and trend information improved faithfulness and temporal reasoning relative to providing only raw temporal sequences, while externalizing feature relations further improved relational reasoning. These findings suggest that structuring error prone reasoning steps into verifiable evidence can improve the reliability of financial XAI narratives. Future work should examine whether these findings generalize to other predictive models and financial decision making tasks.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS 2025-00554384), and the Technology Development Program (No. RS 2024-00513926) funded by the Ministry of SMEs and Startups (MSS, Korea).

## References

[1] S. Baviskar, Conditional adversarial fragility in financial machine learning under macroeconomic stress, 2025. doi:10.48550/arXiv.2512.19935. arXiv:2512.19935.

[2] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, Advances in neural information processing systems 30 (2017).

[3] X. Zeng, K. Zhu, Enhancing the interpretability of shap values using large language models, arXiv preprint arXiv:2409.00079 (2024).

[4] A. Zytek, S. Pido, S. Alnegheimish, L. Berti-Equille, K. Veeramachaneni, Explingo: Explaining ai predictions using large language models, 2024 IEEE International Conference on Big Data (BigData) (2024) 1197–1208.

[5] D. Martens, J. Hinns, C. Dams, M. Vergouwen, T. Evgeniou, Tell me a story! narrative-driven xai with large language models, Decision Support Systems 191 (2025) 114402. URL: https://www. sciencedirect.com/science/article/pii/S016792362500003X. doi:https://doi.org/10.1016/j. dss.2025.114402.

[6] H. Jenett, C. Nagl, M. Nagl, S. M. Price, W. Schaefers, Dynamics of reit returns and volatility: Analyzing time-varying drivers through an explainable machine learning approach, The Journal of Real Estate Finance and Economics 72 (2026) 1–40.

[7] S. Khanna, A. Berger, M. Chopra, D. Berghaus, R. Sifa, History rhymes: Macro-contextual retrieval for robust financial forecasting, 2025 IEEE International Conference on Big Data (BigData) (2025) 7196–7203.

[8] M. Strong, A. Vlachos, TSVer: A benchmark for fact verification against time-series evidence, in: C. Christodoulopoulos, T. Chakraborty, C. Rose, V. Peng (Eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Suzhou, China, 2025, pp. 29906–29926. URL: https://aclanthology.org/2025.emnlp-main.1519/. doi:10.18653/v1/2025.emnlp-main.1519.

[9] G. R. Pratama, K.-K. Tseng, Accurate ensembles, fragile narratives: Multi-scale stacking and a fidelity audit of llm-generated explanations for credit risk, arXiv preprint arXiv:2608.08126 (2026).

[10] J. Kim, S. Park, Groupsegment-shap: Shapley value explanations with group-segment players for multivariate time series, arXiv preprint arXiv:2601.06114 (2026).

[11] B. Goswami, A. Uddin, Significance of predictors: revisiting stock return predictions using explainable ai, Annals of Operations Research 357 (2026) 223–257.

[12] C. Wang, Stock return prediction with multiple measures using neural network models, Financial Innovation 10 (2024) 72.

[13] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, S.-I. Lee, From local explanations to global understanding with explainable ai for trees, Nature machine intelligence 2 (2020) 56–67.

[14] C. Mougan, K. Broelemann, D. Masip, G. Kasneci, T. Thiropanis, S. Staab, Explanation shift: How did the distribution shift impact the model?, arXiv preprint arXiv:2303.08081 (2023).

[15] W. Geng, D. Liu, L. Li, Y. Wang, Evaluating large language models as post hoc explainability interfaces for credit risk models, Expert Systems 43 (2026) e70351.

[16] T. Wang, Enhancing stock market prediction with temporal graph neural networks and large language model-based explainability, Procedia Computer Science 274 (2025) 147–160. URL: https://www.sciencedirect.com/science/article/pii/S1877050925037366. doi:https://doi.org/ 10.1016/j.procs.2025.12.015.

[17] F. Lukassen, J. Herrmann, C. Weisser, A. Silbersdorf, B. Saefken, T. Kneib, Quality without usefulness: Llm-generated xai narratives as trust heuristics rather than decision aids, arXiv preprint arXiv:2605.26770 (2026).

[18] A. C. Teixeira, V. Marar, H. Yazdanpanah, A. Pezente, M. Ghassemi, Enhancing credit risk reports generation using llms: An integration of bayesian networks and labeled guide prompting, 2023, pp. 340–348.

[19] A. Kim, M. Muhn, V. Nikolaev, Financial statement analysis with large language models, arXiv preprint arXiv:2407.17866 (2024).

[20] G. Fatouros, K. Metaxas, J. Soldatos, M. Karathanassis, Marketsenseai 2.0: Enhancing stock analysis through llm agents, 2025 IEEE International Conference on Data Mining Workshops (ICDMW) (2025) 883–892.

[21] S. Gu, B. Kelly, D. Xiu, Empirical asset pricing via machine learning, The Review of Financial Studies 33 (2020) 2223–2273. doi:10.1093/rfs/hhaa009.

[22] P. Gray, M. Limkriangkrai, W. Xu, An examination of the characteristics versus covariance debate for contemporary asset-pricing models: Australian evidence, Accounting & Finance 64 (2024) 3781–3802. doi:10.1111/acfi.13279.

[23] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, G. Neubig, PAL: Program-aided language models, in: A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, J. Scarlett (Eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 10764–10799. URL: https://proceedings.mlr.press/ v202/gao23f.html.

[24] A. Yang, et al., Qwen3 technical report, arXiv preprint arXiv:2505.09388 (2025). doi:10.48550/ arXiv.2505.09388. arXiv:2505.09388.

[25] Mistral AI, Ministral 8b, Model Card, 2024. URL: https://huggingface.co/mistralai/ Ministral-8B-Instruct-2410.

[26] A. Grattafiori, et al., The llama 3 herd of models, arXiv preprint arXiv:2407.21783 (2024). doi:10. 48550/arXiv.2407.21783. arXiv:2407.21783.

## A. Stock Level Characteristic Definitions

For the price-based characteristics, we define the adjusted price variables as

$$
\begin{array} { r } { c _ { t } = \left\{ 1 , \begin{array} { l l } { \mathrm { C F A C P R } _ { t } = 0 \mathrm { ~ o r ~ m i s s i n g , } } \\ { \mathrm { C F A C P R } _ { t } , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

$$
\mathrm { a d j p } _ { t } = \frac { \left| \mathrm { P R C } _ { t } \right| } { c _ { t } } , \qquad \mathrm { a d j h i } _ { t } = \frac { \left| \mathrm { A S K H I } _ { t } \right| } { c _ { t } } , \qquad \mathrm { a d j l o } _ { t }  &  = \frac { \left| \mathrm { B I D L O } _ { t } \right| } { c _ { t } } .
$$

Table 8: Definition and Construction of Stock-Level Characteristics
<table><tr><td>Feature</td><td>Definition / Formula</td><td>Measurement &amp; Source</td></tr><tr><td>Price Level</td><td></td><td></td></tr><tr><td>lowest_bid</td><td> $\left| \mathrm { B I D L O } _ { t } \right|$ </td><td>Month t CRSP: BIDLO</td></tr><tr><td>highest_ask</td><td> $| \mathrm { A S K H I } _ { t } |$ </td><td>Month t</td></tr><tr><td></td><td></td><td>CRSP: ASKHI</td></tr><tr><td>close_price</td><td> $| P R C _ { t } |$ </td><td>Month t CRSP: PRC</td></tr><tr><td>ma3</td><td> $\mathrm { m e a n } ( a d j p _ { t - 2 : t } )$  , minimum 2 observations</td><td>Trailing 3 months CRSP: PRC, CFACPR</td></tr><tr><td>ma12</td><td> $\mathrm { m e a n } ( a d j p _ { t - 1 1 : t } )$  , minimum 6 observations</td><td>Trailing 12 months CRSP: PRC, CFACPR</td></tr><tr><td>Trend &amp; Momentum</td><td></td><td></td></tr><tr><td>price_distance_ma12</td><td> $\frac { a d j p _ { t } - m a 1 2 _ { t } } { m a 1 2 _ { t } }$ </td><td>Month t, relative to 12M MA</td></tr><tr><td></td><td></td><td>CRSP: PRC, CFACPR</td></tr><tr><td>high_12m_ratio</td><td> $\frac { a d j p _ { t } } { \operatorname* { m a x } ( a d j p _ { t - 1 1 : t } ) }$ </td><td>Trailing 12 months</td></tr><tr><td></td><td></td><td>CRSP: PRC, CFACPR</td></tr><tr><td>ma_cross</td><td> $+ 1 \mathrm { i f } m a 3 _ { t } > m a 1 2 _ { t } ;$  otherwise -1</td><td></td></tr><tr><td></td><td></td><td>3M vs. 12M MA CRSP: PRC, CFACPR</td></tr><tr><td>momentum12</td><td> $\frac { a d j p _ { t } } { a d j p _ { t - 1 2 } } - 1$ </td><td></td></tr><tr><td></td><td></td><td>12-month lag CRSP: PRC, CFACPR</td></tr><tr><td>mom1m</td><td> $R E T _ { t - 1 }$ </td><td></td></tr><tr><td></td><td></td><td>Previous month CRSP: RET</td></tr><tr><td></td><td></td><td></td></tr><tr><td>mom6m</td><td> $\exp \left( \sum _ { k = 1 } ^ { 6 } \log ( 1 + R E T _ { t - k } ) \right) - 1$ </td><td>Previous 6 months</td></tr><tr><td></td><td></td><td>CRSP: RET</td></tr><tr><td></td><td></td><td></td></tr><tr><td>macd</td><td> $E M A _ { 1 2 } ( a d j p ) - E M A _ { 2 6 } ( a d j p )$ </td><td>12M / 26M EMA</td></tr><tr><td></td><td></td><td>CRSP: PRC, CFACPR</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>signal_line</td><td> $E M A _ { 9 } ( m a c d )$ </td><td>9M EMA</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>CRSP: PRC, CFACPR</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

## Oscillator

Table 8 (continued)
<table><tr><td>Feature</td><td>Definition / Formula</td><td>Measurement &amp; Source</td></tr><tr><td>stochastic</td><td> $1 0 0 \times { \frac { \mathrm { a d j p } _ { t } - \operatorname* { m i n } _ { \tau = t - 1 3 , \dots , t } { \mathrm { a d j p } _ { \tau } } } { \operatorname* { m a x } _ { \tau = t - 1 3 , \dots , t } \mathrm { a d j p } _ { \tau } - \operatorname* { m i n } _ { \tau = t - 1 3 , \dots , t } { \mathrm { a d j p } _ { \tau } } } }$ </td><td>Trailing 14 months; minimum 6 valid observations</td></tr><tr><td></td><td> $d _ { t } = \mathrm { a d j p } _ { t } - \mathrm { a d j p } _ { t - 1 } ,$   $g _ { t } = \operatorname* { m a x } ( d _ { t } , 0 ) , \qquad \ell _ { t } = \operatorname* { m a x } ( - d _ { t } , 0 ) ,$   $G _ { t } = \frac { 1 } { 1 4 } \sum _ { k = 0 } ^ { 1 3 } g _ { t - k } ,$ </td><td></td></tr><tr><td>rsi</td><td> $L _ { t } = \frac { 1 } { 1 4 } \sum _ { k = 0 } ^ { 1 3 } \ell _ { t - k } ,$ </td><td>Trailing 14 months CRSP: PRC, CFACPR</td></tr><tr><td></td><td> $R S I _ { t } = 1 0 0 - \frac { 1 0 0 } { 1 + G _ { t } / L _ { t } } .$ </td><td></td></tr><tr><td>Volatility &amp; Risk volatility_12</td><td> $\mathrm { s t d } ( R E T _ { t - 1 1 : t } )$ </td><td>Trailing 12 months</td></tr><tr><td>parkinson_vol</td><td> $\sqrt { \frac { 1 } { 4 \ln 2 } \mathrm { { m e a n } _ { \tau = t - 1 1 , . . . , t } \left[ \ln ^ { 2 } \left( \frac { \mathrm { { a d j h i } _ { \tau } } } { \mathrm { { a d j l o } _ { \tau } } } \right) \right] } }$ </td><td>CRSP: RET Trailing 12 months</td></tr><tr><td>bb_width</td><td> $\frac { 4 \mathrm { { s t d } } ( a d j p _ { t - 1 1 : t } ) } { m a 1 2 _ { t } }$ </td><td>CRSP: ASKHI, BIDLO, CFACPR Trailing 12 months CRSP: PRC, CFACPR</td></tr><tr><td>atr</td><td> $\mathrm { T R } _ { t } = \mathrm { m a x } \{ \mathrm { a d j h i } _ { t } - \mathrm { a d j l o } _ { t } ,$  |adjhit − adjpt−1|, |adjlot − adjpt−1|},</td><td>14-month EMA</td></tr><tr><td>beta60m</td><td> $\mathrm { A T R } _ { t } = \mathrm { E M A } _ { 1 4 } ( \mathrm { T R } _ { t } )$   $\beta _ { i , t } = \frac { \mathrm { { C o v } } _ { \tau = t - 6 0 , . . . , t - 1 } ( { R E T } _ { i , \tau } , { V W R E T } { D _ { \tau } } ) } { { \mathrm { { V a r } } _ { \tau = t - 6 0 , . . . , t - 1 } ( { V W R E T } { D _ { \tau } } ) } }$ </td><td>PRC, CFACPR Trailing 60 months</td></tr><tr><td>retskew12m</td><td> $\mathrm { s k e w } \big ( R E T _ { t - 1 2 : t - 1 } \big )$ </td><td>CRSP: RET, VWRETD Previous 12 months</td></tr><tr><td>maxret12m</td><td> $\operatorname* { m a x } ( R E T _ { t - 1 2 : t - 1 } )$ </td><td>CRSP: RET Previous 12 months</td></tr><tr><td>Volume &amp; Liquidity</td><td></td><td>CRSP: RET</td></tr><tr><td>volume</td><td> $V O L _ { t }$ </td><td>Month t</td></tr><tr><td>volume_ma12</td><td> $\mathrm { m e a n } ( V O L _ { t - 1 1 : t } )$ </td><td>CRSP: VOL Trailing 12 months</td></tr><tr><td>volume_ratio</td><td> $\frac { V O L _ { t } } { v o l u m e \_ m a 1 2 _ { t } }$ </td><td>CRSP: VOL Month t, relative to 12M mean</td></tr><tr><td></td><td></td><td>CRSP: VOL</td></tr><tr><td>obv_normalized</td><td> $O B V _ { t } = \sum _ { \tau \leq t } \mathrm { s i g n } ( R E T _ { \tau } ) V O L _ { \tau } ,$ </td><td>Expanding window</td></tr><tr><td></td><td> $O B V _ { t } ^ { n o r m } = \frac { O B V _ { t } - \mathrm { E x p M e a n } ( O B V ) _ { t } } { \mathrm { E x p S t d } ( O B V ) _ { t } + 1 0 ^ { - 1 2 } } .$ </td><td>CRSP: RET, VOL</td></tr><tr><td>turnover ratio</td><td> $\frac { V O L _ { t } } { S H R O U T _ { t } }$ </td><td>Month t</td></tr></table>

Continued on next page

Table 8 (continued)
<table><tr><td>Feature</td><td>Definition / Formula</td><td>Measurement &amp; Source</td></tr><tr><td>amihud</td><td>|RETt|</td><td>Month t</td></tr><tr><td></td><td>dolvolt</td><td>CRSP: RET, PRC, VOL</td></tr><tr><td>bidask</td><td> $A S K _ { t } - B I D _ { t }$ </td><td>Month-end t</td></tr><tr><td></td><td> $\overline { { ( A S K _ { t } + B I D _ { t } ) / 2 } }$ </td><td>CRSP: BID, ASK</td></tr></table>