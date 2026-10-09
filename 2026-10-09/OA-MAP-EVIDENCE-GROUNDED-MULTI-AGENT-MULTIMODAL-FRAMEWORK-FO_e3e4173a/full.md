# OA-MAP: EVIDENCE-GROUNDED MULTI-AGENT MULTIMODAL FRAMEWORK FOR INTERPRETABLE KNEE OSTEOARTHRITIS PROGRESSION

Sixu Chen<sup>1,2</sup> Mingrui Yang<sup>1</sup> Qiang Guan<sup>2</sup> Xiaojuan Li<sup>1</sup>

<sup>1</sup> Cleveland Clinic, Cleveland, OH, USA

<sup>2</sup> Kent State University, Kent, OH, USA

## ABSTRACT

Knee osteoarthritis (KOA) progression prediction can support patient monitoring, requiring the integration of multimodal data and multidomain expertise. Moreover, isolated risk estimates provide limited insight underlying a prediction. To automate the progression assessment workflow and reduce manual effort while providing interpretable findings and supporting evidence, we present OA-MAP, an autonomous multi-agent framework for evidence-grounded assessment of structural and pain progression in KOA. The system incorporates modality-specific agents including MRI, X-ray, and clinical agents, together with a coordinator agent. This framework can autonomously recruit specialist agents, select tools for prediction and analysis, and retrieve literature as external evidence based on user request and available patient information. An uncertainty-informed human-in-the-loop mechanism enables clinicians to review and correct intermediate findings, triggering recomputation of affected results. We evaluate the prediction models using 600 participants from the FNIH Osteoarthritis Biomarkers Consortium cohort. On the test set of 100 participants, the fusion models achieve AUROCs of 0.80 for structural progression and 0.68 for pain progression. A case study illustrates how OA-MAP combines risk estimates with intermediate findings, cross-modal conflicts, literature support, and uncertainty indicators to support interactive review.

## 1 INTRODUCTION

Knee osteoarthritis (KOA) is a prevalent degenerative joint disease, affecting an estimated 364.6 million people worldwide in 2019 Yang et al. (2023); Wang et al. (2025). Early identification of patients at risk of progression can inform patient monitoring and support participant selection for clinical trials Tiulpin et al. (2019). Previous studies have investigated the prediction of structural and pain progression using imaging and clinical data, including multimodal approaches Tiulpin et al. (2019); Panfilov et al. (2025). Despite the effictiveness of these methods, they primarily focus on improving final prediction accuracy, while overlooking the intermediate decision-making process, clinical consistency, and interpretability of the results. Without this information, clinicians may find it difficult to assess the basis and reliability of an individual result. Practical implementation presents another challenge. Using these models often requires preparing model-specific inputs, selecting an appropriate model for the available modalities and intended prediction endpoint, and coordinating multiple processing steps. These tasks require technical expertise and additional effort, creating barriers to routine clinical use. Together, these challenges motivate an integrated framework that simplifies model execution while providing interpretable predictions, traceable evidence, and explicit uncertainty assessment.

Large language model (LLM) agents offer a means of coordinating such workflows by combining language-based reasoning with tool use. These agents can interpret user requests, select and execute actions, and incorporate external information into subsequent decisions Yao et al. (2022). Applications include collaborative biological research Swanson et al. (2025), tool-assisted chest radiograph interpretation Fallahpour et al. (2025), and the selection and execution of clinical risk calculators Jin et al. (2025). These developments suggest the potential of agent-based interfaces to connect clinical requests with specialized computational tools. Multi-agent systems extend this approach by distributing tasks among agents with distinct responsibilities, such as data analysis, evidence retrieval, and result verification. This division of labor can facilitate the integration of heterogeneous data and specialized tools, while allowing agents to review intermediate outputs and identify inconsistencies across different stages of a workflow.

However, reliability remains a central challenge. LLM agents can generate unsupported statements, misinterpret clinical information, and use tools incorrectly. Evaluations in clinical decision-making settings have documented errors in data interpretation and instruction following, including attempts to invoke nonexistent tools Hager et al. (2024). Multi-agent collaboration does not inherently overcome these limitations. Consensus-seeking discussion can induce conformity, causing agents to abandon initially correct judgments and converge on a shared error Cui et al. (2026). Moreover, when agents are differentiated primarily through role prompts, their apparent diversity may not translate into independent reasoning or complementary evidence, particularly if they share the same underlying model and information. Agreement among agents may therefore reflect shared biases or propagated errors and should not, by itself, be treated as independent corroboration. These limitations motivate agent architectures that combine clearly defined responsibilities with verifiable evidence, explicit handling of disagreement, and transparent uncertainty assessment.

To address these challenges, we propose a multi-agent framework for evidence-grounded assessment of structural and pain progression in KOA. The framework integrates specialized prediction models, automated workflow orchestration, literature-based evidence assessment, and clinician review. Given a patient ID and a clinical request, our framework retrieves patient records from a memory module and coordinates the relevant modality-specific agents. Each specialist selects and executes tools appropriate to the available data and requested target endpoint. This automation reduces the need for users to manually prepare inputs and coordinate model execution.

Each specialist grounds its assessment in modality-specific model outputs and retrieves relevant PubMed literature to examine whether published associations between patient features and progression support or conflict with the relationships learned by the models. Specialists are differentiated through their data modalities, computational tools, and associated modality-specific literature evidence, providing a substantive basis for complementary assessments. The coordinator obtains the final progression label from a fusion model using all available modality data and reports it alongside modality-specific predictions, model explanations, literature assessments, and uncertainty indicators.

By presenting intermediate findings, model explanations, and traceable literature evidence alongside KOA progression predictions, the framework enables clinicians to examine the basis of each assessment while helping to limit unsupported agent interpretations.

We further introduce an uncertainty score that combines model-derived uncertainty indicators, disagreement across modalities, and literature-based evidence assessments to comprehensively evaluate the reliablity of the prediction result. This score also guides a human-in-the-loop module in which clinicians can inspect intermediate findings and correct suspected errors. The agents then propagate these revisions through the affected predictions, literature assessments, uncertainty calculations, and final report. This interactive process allows clinicians to examine the basis of a prediction and assess how corrections to intermediate findings change the final assessment.

Our contributions are fourfold:

• We provide interpretable assessments of structural and pain progression in KOA by integrating model-based explanations with external literature evidence. The resulting reports present intermediate findings, supporting evidence, conflicts, and evidence gaps, offering clinicians a more comprehensive view of each patient’s predicted progression risk and the basis for the assessment.

• We develop an agent-based workflow that automates patient-data retrieval, modality- and endpoint-specific tool selection, prediction, and report generation in response to clinical requests, reducing the need for manual coordination across separate processing pipelines.

• We design modality-grounded coordination that differentiates agents through distinct source data inputs, specialized tools, and modality-specific evidence pools. This grounds agent specialization in their information sources and capabilities, rather than role descriptions alone, with the aim of preserving modality-specific evidence and complementary assessments.

• We incorporate uncertainty-informed human-in-the-loop that enables correction of intermediate findings when uncertainty score is high. These revisions trigger recomputation of affected predictions, literature assessments, uncertainty indicators, and reports, allowing clinician feedback to directly influence the final assessment.

## 2 RELATED WORK

## 2.1 LLM AGENT

LLM-based agents have been investigated as coordinators of specialist reasoning and external tools in biomedical research and healthcare Wu et al. (2023). In biological discovery, the Virtual Lab organizes an LLM principal investigator and specialist agents into a research team, with human feedback guiding the development of a computational pipeline for nanobody design Swanson et al. (2025). In medical reasoning, MedAgents uses role-based expert discussions to develop and revise answers to medical questions Tang et al. (2024). MDAgents further adapts the collaboration structure to task complexity, selecting between individual and group reasoning Kim et al. (2024).

Another line of work connects language agents to executable medical tools. MMedAgent learns to select specialized tools for tasks across multiple medical modalities Li et al. (2024a). MedRAX integrates chest X-ray analysis tools with multimodal LLMs and dynamically invokes these tools to address medical queries Fallahpour et al. (2025). AgentMD curates executable clinical calculators from PubMed and selects and applies relevant calculators to patient information for risk assessment Jin et al. (2025) Xiong et al. (2024). These systems Joseph et al. (2025); Leung et al. (2020); Joseph et al. (2022) provide a foundation for coordinating prediction models, image analysis, and external knowledge within an agent workflow. Building on these capabilities, our framework focuses on KOA progression assessment, where agents coordinate modality-specific analysis and assess the literature support for influential feature–progression relationships.

## 2.2 KNEE OSTEOARTHRITIS

Knee osteoarthritis (KOA) prediction research has showned that multi-modal data provide complementary information for predicting structural and symptomatic progression Hu et al. (2023); Li et al. (2024b). Integrating these modalities has therefore become an important direction in KOA progression modeling Tiulpin et al. (2019); Panfilov et al. (2025); Cai et al. (2020).

Interpretability has also received explicit attention. Logistic-regression models with nomograms have been developed for separate radiographic and pain progression outcomes Li et al. (2024b). DeepKOA uses Grad-CAM to visualize image regions associated with its predictions Hu et al. (2023). Explainable machine learning has further combined quantitative MRI and clinical variables with SHAP, regression coefficients, and permutation importance to examine influential predictors Harari et al. (2025). These methods expose model associations and salient findings. Our framework extends this model-level interpretability by evaluating feature–progression relationships against external literature and examining agreement and conflicts across modality-specific predictions, providing a more comprehensive assessment of the evidence underlying each prediction.

## 3 METHOD

Figure 1 illustrates the overall architecture of the proposed framework. Given a patient ID and a clinical request, our framework automatically retrieves available patient records from the memory module. Based on the requested task and available modalities, the framework activates the appropriate modality-specific agents and supplies each with the required inputs. Each specialist agent performs modality-specific analysis, prediction, and literature evidence assessment. The processed modality agent outputs are then passed to the coordinator for fusion-model inference, uncertainty assessment, and final structured report generation.

## 3.1 MEMORY MODULE

The memory module organizes patient information into structured records indexed by patient ID. Clinical records include age, sex, body mass index, symptom and functional assessments, and medication-use information. Imaging records contain 3D DESS MRI images, together with X-

![](images/9206e8a07465d775218f7673ad98cff7697c101d344a5e9117665360a4f40475.jpg)  
Figure 1: Overview of the proposed multi-agent framework for knee osteoarthritis progression assessment.

ray-derived radiographic measurements. Each record retains knee laterality, visit information, and modality availability, enabling our framework to retrieve the relevant data for the requested prediction task.

## 3.2 MODALITY-SPECIFIC AGENT

Our framework includes three modality-specific agents including clinical agent, MRI agent and X-ray agent. Each agent receives the corresponding modal patient data retrieved from the memory module. It invokes a logistic-regression model appropriate to its own modality and the requested target, then retrieves external literature to assess the relationships underlying the prediction. For a given endpoint, let $\mathbf { z } _ { m }$ denote the preprocessed feature vector for modality $m ,$ , and let $f _ { m }$ denote the selected prediction model:

$$
p _ { m } = f _ { m } ( \mathbf { z } _ { m } ) , \qquad \hat { y } _ { m } = \mathbb { I } [ p _ { m } \geq \tau _ { m } ] ,
$$

where $p _ { m }$ is the model’s progression score, $\tau _ { m }$ is its training-derived decision threshold, and I[·] is the indicator function. The label $\hat { y } _ { m } = 1$ denotes a predicted progressor for the selected endpoint, whereas $\hat { y } _ { m } = 0$ denotes a predicted non-progressor.

Literature retrieval focuses on features with the largest contributions to the patient’s prediction. For the logistic-regression models, feature importance is measured by $\vert \beta _ { m j } z _ { m j } \vert$ , where $\beta _ { m j }$ is the coefficient of feature j. This ranking guides top-K feature selection under the predefined retrieval budget. The selected features are mapped to clinical concepts, anatomical terms, and endpoint-specific PubMed queries. Queries expand from specific feature–anatomy combinations to broader predictor concepts, and article titles and abstracts are retrieved through NCBI E-utilities.

Findings are assessed for comparability in measurement, anatomy, temporal role, and prediction endpoint. Those failing these criteria are labeled not comparable, while comparable findings with mixed, statistically nonsignificant, or directionally unclear associations are labeled inconclusive. For the remaining findings, association directions are aligned with the model’s feature and outcome coding and compared with the corresponding logistic-regression coefficient signs. Concordant and opposing directions are labeled supporting and conflicting, respectively.

These assessments characterize literature support for the model’s feature relationships and inform the coordinator’s evidence-based uncertainty calculation. By linking model-derived feature associations to verifiable literature evidence, this process provides clinicians with a traceable basis for interpreting patient-specific progression risk and identifying discrepancies that warrant further clinical review.

MRI agent. The MRI agent automatically coordinates the tools required to convert a 3D DESS MRI into an endpoint-specific progression assessment. It manages the dependencies between image analysis, feature extraction, prediction, and literature retrieval, using intermediate tool outputs to determine the appropriate inputs for subsequent stages. Given an MRI volume, the agent first invokes segmentation tools to identify 17 anatomical ROIs: six femoral regions, two patellar regions, seven tibial regions, and the medial and lateral menisci.

The agent uses each eligible ROI’s anatomical label to automatically select and invoke the appropriate abnormality classifier. It routes the fourteen cartilage ROIs to the cartilage classifier and the two meniscal ROIs to the meniscal-morphology classifier. For each selected crop, the agent uses the Qwen2.5-VL-3B-Instruct visual encoder and supplies the resulting representation, together with the region identity, to the appropriate classifier. For crop $I _ { r }$ and abnormality field $k ,$

$$
\mathbf { h } _ { r } = V ( I _ { r } ) , q _ { r k } = \mathrm { s o f t m a x } ( C _ { k } ( \mathbf { h } _ { r } , r ) ) _ { 1 } ,
$$

where V denotes the visual encoder and $C _ { k }$ produces binary presence logits. The cartilage classifier returns two scores per ROI, representing nonzero cartilage-size and cartilage-depth grades. The meniscal classifier returns three morphology scores per ROI, corresponding to the anterior horn, body, and posterior horn. The agent retains these 34 continuous abnormality-presence scores for progression prediction and uses validation-derived thresholds to identify regions for visual review.

To complement the classifier outputs, the MRI agent invokes quantitative measurement tools on the segmentation results. These tools produce four directional meniscal-extrusion measurements and 16 geometric descriptors, comprising cartilage volume and thickness descriptors across four compartments, two joint-gap measurements, and two meniscal-coverage ratios. The agent assembles their outputs into a common representation:

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { M R I } } = [ \mathbf { q } _ { 3 4 } ; \mathbf { e } _ { 4 } ; \mathbf { g } _ { 1 6 } ] \in \mathbb { R } ^ { 5 4 } . } \end{array}
$$

Conditioned on the requested progression target, the agent selects the corresponding prediction model and supplies x<sub>MRI</sub> in the required feature order. The agent then uses the model’s feature contributions to guide literature retrieval and assess external support for the influential feature– progression relationships. It forwards the prediction, MRI findings, and literature assessments to the coordinator for integration.

The MRI agent supports human-in-the-loop refinement by incorporating clinician corrections to ROI-level abnormality scores. Following each correction, it updates the corresponding input features and reruns the affected prediction and literature-assessment steps. The revised outputs are then propagated to uncertainty assessment and final report generation.

X-ray Agent. The X-ray agent receives radiographic information, comprising 18 baseline features that describe global and compartmental OA grades, joint-space measurements, osteophytes, sclerosis, and alignment. Based on the requested structural or pain target, the agent automatically selects the corresponding prediction model and supplies the features in the required order.

The agent uses the prediction model’s feature contributions to guide subsequent evidence retrieval in PubMed literature. It identifies influential radiographic features, maps them to clinical concepts, and invokes the PubMed retrieval tool with queries matched to the anatomical compartment and prediction target. The retrieved findings are interpreted and compared with the model’s feature relationships to identify supporting, conflicting, or inconclusive evidence. The agent then forwards a structured modal output containing the prediction score, decision threshold, predicted label, feature explanations, and assessed literature evidence to the coordinator agent.

Clinical Agent. The clinical agent takes clinical demographic data as input. Its input comprises six baseline variables: age, sex, body mass index, target-knee WOMAC pain, WOMAC physical-function limitation, and frequent knee-pain medication use. Given the requested target, the agent automatically selects the corresponding clinical prediction model and invokes the further PubMed-based evidence retrieval.

## 3.3 COORDINATOR AGENT

The coordinator agent invokes an endpoint-specific fusion model using all available modalities and integrates structured outputs from the available modality-specific agents to produce the final assessment.

It then performs three important tasks. First, it identifies disagreements among modality-specific predictions and conflicts between model relationships and retrieved PubMed evidence and quantifies uncertainty from these complementary sources. Second, it flags uncertain or conflicting findings that warrant clinician review, providing the relevant predictions and evidence for human assessment. Third, it consolidates the integrated prediction, modality-specific findings, literature assessments, and uncertainty estimates into a structured final report.

Uncertainty Quantification. The coordinator agent quantifies three sources of uncertainty: proximity to the prediction boundary, disagreement between modality-specific predictions, and insufficient or conflicting literature evidence. Let $\mathcal { M }$ denote the selected modalities, and let $p _ { m } , \tau _ { m }$ , and $\hat { y } _ { m }$ denote the prediction score, classification threshold, and predicted label returned by modality-specific agent m, respectively.

Prediction uncertainty is determined by the distance between the model score and its classification threshold. We normalize this distance by $1 - \tau _ { m }$ for scores at or above the threshold and by $\tau _ { m }$ for scores below it. Uncertainty is then defined as one minus this normalized distance:

$$
u _ { m } = \left\{ \begin{array} { l l } { 1 - \displaystyle \frac { p _ { m } - \tau _ { m } } { 1 - \tau _ { m } } , } & { p _ { m } \ge \tau _ { m } , } \\ { 1 - \displaystyle \frac { \tau _ { m } - p _ { m } } { \tau _ { m } } , } & { p _ { m } < \tau _ { m } , } \end{array} \right.\tag{1}
$$

$$
U _ { \mathrm { p r e d } } = \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } u _ { m } .
$$

The uncertainty score $u _ { m }$ is highest $( u _ { m } \ = \ 1 )$ when the model score equals the classification threshold. As the score moves away from the threshold toward 0 or 1, uncertainty decreases and model confidence increases. The coordinator averages these uncertainty scores across all selected modalities to obtain $U _ { \mathrm { p r e d } }$ , where a higher value indicates greater average uncertainty in the modality specific predictions.

Modality-conflict uncertainty measures disagreement among all available modality-specific agents. The coordinator compares each pair of agents and averages their conflict scores:

$$
U _ { \mathrm { c o n f i c t } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( m , n ) \in \mathcal { P } } \mathbb { I } [ \hat { y } _ { m } \neq \hat { y } _ { n } ] | p _ { m } - p _ { n } | ,\tag{2}
$$

where $\mathcal { P }$ contains all distinct pairs of agents with available predictions. A pair contributes zero when its predicted labels agree. When the labels disagree, its contribution is the absolute difference between the two prediction scores. Thus, the overall conflict score is zero when all agents predict the same label, while higher values indicate stronger disagreement among their predictions. This component is calculated whenever at least two agents provide predictions; otherwise, it is recorded as unavailable.

Literature-evidence uncertainty summarizes the support for the feature relationships underlying each modality-specific prediction. Let $\kappa _ { m }$ contain the K features with the largest absolute log-odds contributions, $a _ { m j } = \beta _ { m j } z _ { m j }$ . For feature j, let $N _ { m j }$ denote the number of eligible papers with verified source passages, comparable measurements and temporal roles, and a direct match to the feature and prediction target. Each eligible paper ℓ contributes at most one directional assessment per feature:

$$
r _ { m j \ell } = \left\{ \begin{array} { l l } { + 1 , } & { \mathrm { s u p p o r t s ~ t h e ~ m o d e l ~ c o e f f i c i e n t ~ d i r e c t i o n } , } \\ { 0 , } & { \mathrm { i n c o n c l u s i v e } , } \\ { - 1 , } & { \mathrm { c o n f i c t s ~ w i t h ~ t h e ~ m o d e l ~ c o e f f i c i e n t ~ d i r e c t i o n } . } \end{array} \right.\tag{3}
$$

A paper receives $r _ { m j \ell } = 0$ when its eligible findings are mixed, statistically nonsignificant, or directionally unclear. Papers without eligible comparable findings are excluded from $N _ { m j }$

The coordinator combines directional consistency with a saturating measure of evidence sufficiency:

$$
\begin{array}{c} \begin{array} { l l } { { \displaystyle D _ { m j } = \left\{ \frac { 1 } { 2 } \left( 1 + \frac { 1 } { N _ { m j } } \sum _ { \ell = 1 } ^ { N _ { m j } } r _ { m j \ell } \right) , \right.} } & { { N _ { m j } > 0 , } } \\ { { 0 , } } & { { N _ { m j } = 0 , } } \\ { { S _ { m j } = 1 - \exp ( - N _ { m j } / \lambda ) , } } & { { \lambda = 2 , } } \\ { { e _ { m j } = D _ { m j } S _ { m j } . } } & { { } } \end{array}    \end{array}\tag{4}
$$

Directional consistency equals 1 when all eligible papers support the coefficient direction, 0 when all conflict, and 0.5 when all are inconclusive. The sufficiency term discounts agreement based on a small number of papers. A feature without eligible evidence has $D _ { m j } = S _ { m j } \stackrel {  } { = } e _ { m j } = 0$

Feature-level support is aggregated using the magnitude of each feature’s contribution to the patient’s prediction:

$$
E _ { m } = \frac { \sum _ { j \in \mathcal { K } _ { m } } \left| a _ { m j } \right| e _ { m j } } { \sum _ { j \in \mathcal { K } _ { m } } \left| a _ { m j } \right| } ,\tag{5}
$$

$$
U _ { \mathrm { e v i d e n c e } } = 1 - { \frac { 1 } { \vert { \mathcal { M } } \vert } } \sum _ { m \in { \mathcal { M } } } E _ { m } .
$$

Supporting papers increase a feature’s evidence support, while conflicting papers reduce it. A feature without eligible papers receives zero support, but its weight remains in the calculation so that missing evidence is accounted for. If none of the selected features has eligible papers, or if all contribution weights are zero, we set $E _ { m } = 0$ . Overall, stronger literature support produces a higher $E _ { m }$ and lower evidence uncertainty, whereas missing or conflicting evidence produces lower support and higher uncertainty.

When all three components are available, the coordinator computes overall uncertainty and a complementary confidence index:

$$
\begin{array} { l l } { \displaystyle U _ { \mathrm { o v e r a l l } } = \frac { U _ { \mathrm { p r e d } } + U _ { \mathrm { c o n f i c t } } + U _ { \mathrm { e v i d e n c e } } } { 3 } , } \\ { \displaystyle C _ { \mathrm { o v e r a l l } } = 1 - U _ { \mathrm { o v e r a l l } } . } \end{array}\tag{6}
$$

All uncertainty indices range from 0 to 1, with higher values indicating greater uncertainty. A high overall uncertainty score corresponds to low confidence in the prediction, suggesting that the result should be interpreted cautiously and may warrant further assessment by a clinician. The coordinator reports each component separately so that clinicians can identify whether the uncertainty comes from a model score close to its classification threshold, disagreement among modality-specific agents, or insufficient or conflicting literature evidence.

Human-in-the-Loop Review. The coordinator agent uses overall uncertainty to automatically identify predictions that warrant clinician review. When the overall uncertainty score is available, the review flag is defined as

$$
H _ { \mathrm { r e v i e w } } = \mathbb { I } [ U _ { \mathrm { o v e r a l l } } > \tau _ { \mathrm { r e v i e w } } ] ,\tag{7}
$$

where $\tau _ { \mathrm { r e v i e w } }$ is a predefined review threshold. A flag of $H _ { \mathrm { r e v i e w } } = 1$ prompts the coordinator to request clinician assessment of the intermediate findings underlying the prediction.

For cases with MRI data, the coordinator presents the ROIs flagged as abnormal, together with their anatomical labels, predicted abnormalities, and classifier scores. The clinician visually inspects these regions and can retain each original abnormality score or replace it with a reviewed binary value: 0 for normal and 1 for abnormal.

Once the clinician submits corrections, our system automatically updates the corresponding entries in the feature representation and trigger recomputation of affected predictions, literature assessments, uncertainty indicators, and reports. This human-in-the-loop process incorporates clinician corrections directly into the model inputs, allowing expert review to change the final progression score and potentially the predicted label. The agents automatically propagate these corrections, ensuring that clinician feedback is reflected in the updated assessment.

Final Report Generation. The coordinator agent integrates the fusion-model prediction, modalityspecific outputs, literature assessments, and uncertainty estimates into a structured multi-modal prediction report. The report presents the final outcome alongside its supporting model findings, external evidence, and unresolved discrepancies. It contains four components:

• Primary prediction: The predicted outcome, Progressor or Non-progressor, together with the continuous LR progression score and the corresponding classification threshold.

• Multimodal assessment: Predictions from the available modality-specific agents and a summary of agreement or disagreement among them.

• Model explanation: The features contributing most strongly to the prediction, including their standardized values, logistic-regression coefficients, and contributions to the model log-odds.

• Evidence and uncertainty assessment: Literature support and conflicts concerning the influential feature–progression relationships, together with inconclusive findings, evidence gaps, and links to the source articles. The report also presents the overall uncertainty and its individual components to help clinicians identify findings that warrant further review.

The coordinator examines numerical results and evidence references against the structured tool outputs before presenting the report, maintaining consistency between the final assessment and the underlying agent outputs.

## 4 EXPERIMENT

## 4.1 DATASET

We use the Foundation for the National Institutes of Health (FNIH) Osteoarthritis Biomarkers Consortium cohort, a nested case–control sample of 600 participants from the Osteoarthritis Initiative, with one index knee per participant. The cohort includes 297 progressors and 303 non-progressors for each of two separately defined endpoints: Structural and pain progression. Structural progression is defined as a reduction of at least 0.7 mm in minimum medial tibiofemoral joint-space width from baseline to the 24-, 36-, or 48-month visit. Pain progression is defined as an increase of at least 9 points in the WOMAC pain score, normalized to a 0–100 scale, relative to baseline at the 24-, 36-, or 48-month visit. This increase must occur at two or more visits between months 24 and 60 to establish persistent worsening.

Baseline three-dimensional DESS MRI images, radiograph-derived measurements, and clinical and demographic variables are used to predict the two endpoints. A split stratified by the four FNIH progression groups reserves 100 knees for testing, with 50 progressors and 50 non-progressors for each endpoint. The remaining 500 knees are used to train progression prediction models.

## 4.2 PROGRESSION PREDICTION PERFORMANCE

Table 1: AUROC for structural and pain progression prediction using different input modalities.

<table><tr><td>Input</td><td>Structural AUROC</td><td>Pain AUROC</td></tr><tr><td>Clinical</td><td>0.60</td><td>0.63</td></tr><tr><td>X-ray</td><td>0.62</td><td>0.58</td></tr><tr><td>MRI</td><td>0.80</td><td>0.58</td></tr><tr><td>Fusion</td><td>0.80</td><td>0.68</td></tr></table>

Table 1 reports the AUROC of the modality-specific and fusion models for structural and pain progression prediction. Fusion achieved the highest AUROC for both structural progression (0.80) and pain progression (0.68), in line with prior multi-modal KOA studies.

<table><tr><td>Structured Multimodal Prediction Report</td></tr><tr><td></td></tr><tr><td>Assessment: Baseline</td></tr><tr><td>Endpoint: Structural progression</td></tr></table>

## 1. Primary prediction

Final classification: Progressor. The fusion logistic-regression (LR) score is 0.8100, above the trainingderived threshold of 0.4505.

## 2. Multimodal assessment

<table><tr><td>Agent/model</td><td>LR score Threshold Prediction</td><td></td><td></td></tr><tr><td>MRI agent</td><td>0.7580</td><td></td><td>0.4563 Progressor</td></tr><tr><td>X-ray agent</td><td>0.5120</td><td></td><td>0.5320 Non-progressor</td></tr><tr><td>Clinical agent</td><td>0.5524</td><td></td><td>0.5222 Progressor</td></tr><tr><td>Fusion model</td><td>0.8100</td><td></td><td>0.4505 Progressor</td></tr></table>

The MRI and clinical predictions agree with the final classification. The X-ray score is 0.0200 below its threshold, producing a discordant classification that is retained for review. The final classification is determined by the fusion model, rather than majority voting.

## 3. Model explanation

Features are ranked by absolute contribution to model log-odds. Here, $z _ { j }$ expresses the patient’s feature value relative to the training-set mean in standard-deviation units, and $\dot { \beta } _ { j }$ is the learned LR weight for that feature. Their product, $\beta _ { j } z _ { j } ,$ is the feature’s additive contribution to model log-odds: positive values increase the prediction score, while negative values decrease it.

## 3.1. Fusion model: top five contributions

<table><tr><td>Feature</td><td> $z _ { j }$ </td><td> $\beta _ { j }$   $\beta _ { j } z _ { j }$ </td></tr><tr><td>Kellgren-Lawrence grade</td><td>+1.170-0.492 -0.576</td><td></td></tr><tr><td>Medial meniscus anterior horn abnormality probability</td><td>+3.408 +0.147 +0.501</td><td></td></tr><tr><td>Lateral femoral cartilage bone-scaled volume index</td><td></td><td>+1.793 +0.194 +0.348</td></tr><tr><td>Medial joint-space-narrowing grade</td><td></td><td>+1.158+0.248+0.288</td></tr><tr><td>Medial meniscal tibial-footprint coverage ratio</td><td></td><td>+0.695+0.350+0.244</td></tr></table>

3.2. MRI agent: top five contributions
<table><tr><td>Feature</td><td> $z _ { j }$ </td><td> $\beta _ { j }$   $\beta _ { j } z _ { j }$ </td></tr><tr><td>Medial meniscus anterior horn abnormality probability</td><td>+3.408+0.075 +0.255</td><td></td></tr><tr><td>Lateral femoral cartilage bone-scaled volume index</td><td>+1.793 +0.099+0.178</td><td></td></tr><tr><td>Anterior medial tibial cartilage full-thickness-loss probability</td><td>+2.070 +0.080+0.166</td><td></td></tr><tr><td>Medial meniscus body abnormality probability</td><td>+2.066 +0.060+0.124</td><td></td></tr><tr><td>Anterior medial tibial cartilage lesion-presence probability</td><td></td><td>+1.494 +0.077 +0.115</td></tr></table>

## 3.3. X-ray agent: top five contributions

<table><tr><td>Feature</td><td> $z _ { j }$   $\beta _ { j }$ </td><td> $\beta _ { j } z _ { j }$ </td></tr><tr><td>Kellgren-Lawrence grade</td><td>+1.176 -0.537 -0.631</td><td></td></tr><tr><td>Minimum medial joint-space width</td><td>-0.596+0.927 -0.553</td><td></td></tr><tr><td>Medial joint-space-narrowing grade</td><td>+1.164 +0.415 +0.483</td><td></td></tr><tr><td>Femoral-tibial alignment angle</td><td>-3.164 -0.131 +0.415</td><td></td></tr><tr><td>Medial tibial sclerosis grade</td><td>+0.425 +0.583 +0.248</td><td></td></tr></table>

3.4. Clinical agent: top five contributions
<table><tr><td>Feature</td><td>zj</td><td> $\beta _ { j }$ </td></tr><tr><td>Female sex indicator</td><td>-1.205 -0.260 +0.313</td><td></td></tr><tr><td>Baseline WOMAC physical function</td><td>+0.950+0.218 +0.207</td><td></td></tr><tr><td>Baseline age</td><td>-0.992 +0.209 -0.207</td><td></td></tr><tr><td>Baseline body mass index</td><td>-1.792 +0.070 -0.125</td><td></td></tr><tr><td>Frequent knee-pain medication use</td><td></td><td>-0.652 -0.045 +0.029</td></tr></table>

## 4. Literature evidence assessment

The agent checks whether PubMed findings are comparable with the model features and prediction task in terms of measurement, anatomy, timing, outcome. For comparable findings, it compares the reported association direction with the sign of the corresponding LR coefficient. Agreement indicates literature support for the model’s feature–progression relationship, but does not confirm whether an individual patient will progress.

<table><tr><td>Modality</td><td>Evidence assessment</td></tr><tr><td>MRI</td><td>Support for medial meniscal abnormalities was qualified by inconclusive findings and limited comparability. The remaining MRI features lacked clear comparable support, primarily due to differences in measurements or endpoints.</td></tr><tr><td>X-ray</td><td>Evidence supported the model relationship for medial joint-space-narrowing grade but conflicted with that for minimum medial joint-space width. Findings for the remaining features were inconclusive or not comparable.</td></tr><tr><td>Clinical</td><td>BMI-related findings were inconclusive. Retrieved findings for sex, age, physical function, and medication use were classified as not comparable to the corresponding model relationships.</td></tr></table>

These assessments characterize literature support for modality-specific feature relationships and inform evidence uncertainty. They do not establish the correctness of the patient-level fusion prediction.

## 5. Uncertainty assessment

<table><tr><td>Component</td><td>Value</td></tr><tr><td>Prediction uncertainty</td><td>0.7814</td></tr><tr><td>Modality-conflict uncertainty</td><td>0.0955</td></tr><tr><td>Literature evidence uncertainty</td><td>0.8596</td></tr><tr><td>Overall uncertainty</td><td>0.5788</td></tr><tr><td>Complementary confidence</td><td>0.4212</td></tr></table>

## 6. Human review and correction

Findings warranting review include the X-ray cross modality disagreement, the medial meniscal and anterior medial tibial cartilage abnormalities, and feature relationships with conflicting or insufficient evidence. Clinician corrections can be passed through the feature-override workflow to update predictions and the report. As the overall uncertainty exceeds the predefined threshold of 0.5, the case is flagged as Clinician Review Required.

## 5 DISCUSSION

In this study, we present a multi-agent framework for knee osteoarthritis progression assessment, covering both structural and pain progression. The framework integrates MRI, radiographic, and clinical information through modality-specific agents and a coordinator agent. It automates patientdata retrieval, task-specific tool selection, progression prediction, literature evidence assessment, and structured report generation.

The framework connects automated prediction with interpretable findings, literature evidence, and clinician oversight. Feature-level evidence assessment identifies agreement, potential conflicts, and gaps between modelled relationships and literature findings. The coordinator reports these assessments alongside prediction uncertainty and cross-modal disagreement, providing a transparent basis for reviewing the final prediction that helps determine when a prediction should be trusted, questioned, or interpreted cautiously. The human-in-the-loop workflow further allows clinicians to correct intermediate MRI findings, with the agents propagating these corrections through subsequent predictions and reports. This enables clinical feedback to directly influence the final assessment while retaining automated coordination of the analysis.

Future work will expand the framework by incorporating additional analysis tools and prediction models and extending its coverage to more clinical endpoints. These extensions will broaden the range of tasks supported by the agents within the similar coordinated workflow.

## REFERENCES

Guoqi Cai, F Cicuttini, Dawn Aitken, LL Laslett, Z Zhu, Tania Winzenberg, and Graeme Jones. Comparison of radiographic and mri osteoarthritis definitions and their combination for prediction of tibial cartilage loss, knee symptoms and total knee replacement: a longitudinal study. Osteoarthritis and Cartilage, 28(8):1062–1070, 2020.

Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, and Cong Zuo. Free-mad: Consensus-free multiagent debate. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 31977–31997, 2026.

Adibvafa Fallahpour, Jun Ma, Alif Munim, Hongwei Lyu, and Bo Wang. Medrax: Medical reasoning agent for chest x-ray. arXiv preprint arXiv:2502.02673, 2025.

Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios Kaissis, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nature medicine, 30(9):2613–2622, 2024.

Ryan Harari, Stacy E Smith, Sara M Bahouth, Lawrence Lo, Meera Sury, Ming Yin, Lena F Schaefer, William Wells, Jamie Collins, and Jeffrey Duryea. Predicting knee osteoarthritis progression using explainable machine learning and clinical imaging data. Available at SSRN 5383084, 2025.

Jiaping Hu, Chuanyang Zheng, Qingling Yu, Lijie Zhong, Keyan Yu, Yanjun Chen, Zhao Wang, Bin Zhang, Qi Dou, and Xiaodong Zhang. Deepkoa: a deep-learning model for predicting progression in knee osteoarthritis using multimodal magnetic resonance images from the osteoarthritis initiative. Quantitative Imaging in Medicine and Surgery, 13(8):4852, 2023.

Qiao Jin, Zhizheng Wang, Yifan Yang, Qingqing Zhu, Donald Wright, Thomas Huang, Nikhil Khandekar, Nicholas Wan, Xuguang Ai, W John Wilbur, et al. Agentmd: Empowering language agents for risk prediction with large-scale clinical tool learning. Nature Communications, 16(1): 9377, 2025.

Gabby B Joseph, Charles E McCulloch, Michael C Nevitt, Thomas M Link, and Jae Ho Sohn. Machine learning to predict incident radiographic knee osteoarthritis over 8 years using combined mr imaging features, demographics, and clinical factors: data from the osteoarthritis initiative. Osteoarthritis and Cartilage, 30(2):270–279, 2022.

Gabby B Joseph, Charles E McCulloch, Michael C Nevitt, Nancy E Lane, Sharmila Majumdar, and Thomas M Link. Machine learning models for clinical and structural knee osteoarthritis prediction: recent advancements and future directions. Osteoarthritis and cartilage open, pp. 100654, 2025.

Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik S Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae W Park. Mdagents: An adaptive collaboration of llms for medical decision-making. Advances in Neural Information Processing Systems, 37: 79410–79452, 2024.

Kevin Leung, Bofei Zhang, Jimin Tan, Yiqiu Shen, Krzysztof J Geras, James S Babb, Kyunghyun Cho, Gregory Chang, and Cem M Deniz. Prediction of total knee replacement and diagnosis of osteoarthritis by using deep learning on knee radiographs: data from the osteoarthritis initiative. Radiology, 296(3):584–593, 2020.

Binxu Li, Tiankai Yan, Yuanting Pan, Jie Luo, Ruiyang Ji, Jiayuan Ding, Zhe Xu, Shilong Liu, Haoyu Dong, Zihao Lin, et al. Mmedagent: Learning to use medical tools with multi-modal agent. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pp. 8745–8760, 2024a.

Xiaoyu Li, Chunpu Li, and Peng Zhang. Predictive models of radiographic progression and pain progression in patients with knee osteoarthritis: data from the fnih oa biomarkers consortium project. Arthritis Research & Therapy, 26(1):112, 2024b.

Egor Panfilov, Simo Saarakkala, Miika T Nieminen, and Aleksei Tiulpin. End-to-end prediction of knee osteoarthritis progression with multimodal transformers. IEEE Journal ofBiomedical and Health Informatics, 29(9):6276–6286, 2025.

Kyle Swanson, Wesley Wu, Nash L Bulaong, John E Pak, and James Zou. The virtual lab of ai agents designs new sars-cov-2 nanobodies. Nature, 646(8085):716–723, 2025.

Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 599–621, 2024.

Aleksei Tiulpin, Stefan Klein, Sita MA Bierma-Zeinstra, Jérôme Thevenot, Esa Rahtu, Joyce van Meurs, Edwin HG Oei, and Simo Saarakkala. Multimodal machine learning-based knee osteoarthritis progression prediction from plain radiographs and clinical data. Scientific reports, 9(1):20038, 2019.

Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Jiaming Ji, Wenting Chen, Xiang Li, and Yixuan Yuan. A survey of llm-based agents in medicine: How far are we from baymax? Findings of the Association for Computational Linguistics: ACL 2025, pp. 10345–10359, 2025.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 6233–6251, 2024.

Guangmin Yang, Jue Wang, Yun Liu, Haojie Lu, Liu He, Changsheng Ma, and Zhe Zhao. Burden of knee osteoarthritis in 204 countries and territories, 1990–2019: results from the global burden of disease study 2019. Arthritis care & research, 75(12):2489–2500, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.