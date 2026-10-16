# TIR-EC: Incentivizing Tool-Use Reinforcement Learning for Post-Arrival Emergency Care Prediction

Pengfei Hu<sup>1</sup>\* Chenwei Wu<sup>2</sup>\* Qiao Jin<sup>3</sup> Jun Deng<sup>4</sup> Zhiyong Lu<sup>3</sup> Liyue Shen<sup>2†</sup> Yue Ning<sup>1†</sup>

<sup>1</sup>Stevens Institute of Technology <sup>2</sup>University of Michigan, Ann Arbor

<sup>3</sup>National Institutes of Health <sup>4</sup>Yale University

{phu9,yue.ning}@stevens.edu {chenweiw,liyues}@umich.edu

{qiao.jin,zhiyong.lu}@nih.gov jun.deng@yale.edu

## Abstract

Large language models (LLMs) can reason over both structured measurements and freetext records, which makes them attractive for emergency department (ED) prediction. However, most existing LLMs still fall short of conventional ML methods like trees on EHRbased prediction tasks. Our controlled analysis traces this gap to two weaknesses: LLMs do not select informative features, and they misjudge what numerical values imply for risk even when given the right features. We propose TIR-EC, a tool-integrated reinforcement learning framework that addresses both. TIR-EC exposes EHR feature groups as tools the LLM can call. It trains the LLM with rewards from a gradient-boosted tree teacher, which provides a reference set of informative tools and a graded risk target. TIR-EC also distills task-specific clinical skills into the model weights through a reverse curriculum that gradually removes the skills from the prompt during RL process. Evaluation on three clinical prediction tasks shows that our 4B agent achieves the highest macro F1 of all compared models, outperforms conventional ML baselines, and is competitive with much larger frontier LLMs. Further analysis shows that TIR-EC selects more useful evidence with fewer tool calls and combines measurements more effectively.

## 1 Introduction

The emergency department (ED) is a high-stakes, time-critical scenario where predicting outcomes such as hospital admission, clinical deterioration, and return visits from electronic health records (EHRs) can support timely decision-making (Xie et al., 2022; Chen et al., 2023; Wu et al., 2025). Unlike conventional machine learning models, large language models (LLMs) can jointly reason over structured EHR data and free-text clinical notes, like how a clinician proceeds, and therefore its application in ED have attracted growing research attention (Shi et al., 2024; Alcaraz et al., 2025; Kruse et al., 2025).

However, recent studies show that most LLMs, both general-purpose and medical, still underperform simple and cheap conventional models such as gradient-boosted trees on EHR-based clinical prediction in spite of model scaling, prompting, or fine-tuning (Chen et al., 2026; Sanatizadeh et al., 2026). To understand why LLMs underperform on EHR-based clinical prediction, and more importantly, how they could be improved, we run a controlled preliminary analysis on ED encounters comparing general and medical LLMs with conventional models, yielding two major findings.

First, we observe that more EHR features do not consistently improve LLM predictions, mainly due to poor feature selection. EHR features vary widely in format, reliability, and relevance to a particular patient outcome (Shickel et al., 2017; Kansal et al., 2025). Supervised models learn their task-specific relevance through tree splits or linear coefficients, whereas LLMs must infer relevance from context autonomously. As shown in Fig. 1 (a), in disposition prediction, Gradient Boosting Tree Models (GBM) and Logistic Regression (LR) clearly improve as more feature groups become available, while LLMs show smaller and less consistent gains with fluctuations. Allowing the LLM to pick features like an agent selects tools to call before prediction does not resolve this problem, while supplying LLMs the top-ranked groups identified by GBM improves performance. These comparisons identify feature selection as a bottleneck for LLMs to leverage EHR features, and motivate explicit guidance on which evidence is useful for the prediction task.

Second, preliminary analysis indicates that the

![](images/644c464eb91acdbfb57de149240fe315bcfd066176af2cda01afe50dbc11dca2.jpg)  
Figure 1: Preliminary analysis on 2000 MC-MED patients (Kansal et al., 2025). (a) Disposition AUROC given k of 12 numerical feature groups, averaged over 10 runs. (b) 46 features’ dataset level vs. model-stated change in risk. (c) One decompensation patient. (d) Error decomposition on 100 randomly sampled patients for whom the GBM predicts correctly but Qwen3-4B free selection fails, human-labeled.

LLM cannot reliably translate clinical numerical values into task-specific risk. Even when supplied with GBM-selected features, the LLM still falls well short of GBM, raising the question why it fails to leverage these features. For each predictive numerical feature, we compare the change in model-assigned risk between patients with high and low feature values against the corresponding difference in observed outcome rates in the dataset. Fig. 1 shows that the LLM’s risk estimates mostly fall in the over- and under-reaction region, and some in the wrong direction. Such task-specific relationships, like the heart rate at which deterioration becomes likely, are learned from data by ML models but are not captured by LLM’s medical priors (Chen et al., 2026; Moghaddam et al., 2026). Error analysis in Fig. 1(d) further shows that among LLM prediction errors over 100 cases, most come from mis-weighting evidence that the LLM had already retrieved. In the example shown in Fig. 1(c), the LLM dismisses borderline vital signs as showing no immediate instability, while tree models correctly route the same values to a high-risk prediction. This pattern reveals that even when informative evidence is provided, the LLM may misjudge its predictive significance. Effective EHR prediction therefore requires learning not only which features to select, but also how their numerical values relate to the target outcome.

Guided by these findings, we propose TIR-EC, a tool-integrated reinforcement learning (RL) framework that teaches LLMs both which EHR features to use and how to interpret them for EHR-based clinical task prediction. To address feature selection, we treat EHR feature groups as tools that the LLM can optionally call, so that selection becomes an action optimized with RL. Since expert tool-use trajectories are unavailable, we use tree-importance to derive a reference set of informative tools against which the agent’s selections are rewarded. To address feature understanding, we curate experience in the form of skills, consisting of short taskspecific rules each stating which evidence to inspect and how its values shift risk. We then distill these skills into the model weights through reverse curriculum learning. Training starts with the full skills in context and gradually removes them, so that the agent predicts without any skills at inference time. Experiments on disposition, decompensation, and revisit prediction show that: (1) with only a 4B backbone, TIR-EC outperforms both conventional ML models and frontier LLMs; (2) TIR-EC improves the LLM in both feature selection and feature understanding, the two weaknesses identified above; and (3) the two abilities are mutually beneficial: as the LLM understands features better, it also selects them better.

Our contributions are threefold: 1 We identify two key weaknesses behind LLMs’ underperformance on EHR prediction, poor feature selection and unreliable numerical understanding. 2 Based on these findings, we propose TIR-EC, reformulating feature selection as an agentic tool call task and feature understanding as a skill distillation task.

3 Empirical results show that TIR-EC achieves state-of-the-art performance, improves both feature selection and feature understanding, and reveals that the two are mutually beneficial.

## 2 Related Work

## 2.1 Emergency Care Prediction

Emergency care supports a broad range of timesensitive clinical prediction tasks (Xie et al., 2022; Chen et al., 2023). Unlike longitudinal EHR tasks that aggregate medical concepts (e.g., medications) from prior visits (Rasmy et al., 2021; Wornow et al., 2023), ED prediction centers on one episode and must respect when each observation becomes available. Existing work can be categorized into two settings: (1) Arrival-Time Prediction: uses patient demographics and initial vital signs to estimate critical illness, arrival triage, or resource allocations (Colakca et al., 2024; Sebastian et al., 2025; Solaiman et al., 2026). (2) Post-Arrival Prediction: updates risk after care begins using repeated vital signs, laboratory results, orders, imaging, and physiological signals (Alcaraz et al., 2025; Sun et al., 2024; Kansal et al., 2025). Recent LLM studies in emergency care have largely focused on triage (Cui et al., 2026), where inputs are compact and consistently recorded. Still, the later setting is more demanding for LLMs, since evidence is distributed across heterogeneous sources, includes high-frequency measurements, and requires numerical and temporal interpretation (Kansal et al., 2025; Alcaraz et al., 2025). Here, we focus on this setting by treating functions as retrieval actions over structured and unstructured clinical data.

## 2.2 LLMs and Agents for EHR

Current EHR research uses pretrained representations, retrieval, automatic feature construction, multi-agent coordination, and executable interfaces to reason over structured records (Steinberg et al., 2021; Wornow et al., 2023; Cao et al., 2026). Toolintegrated reasoning methods teach models to interleave reasoning with external actions (Yao et al., 2023; Patil et al., 2024; Qin et al., 2024), while tool-integrated RL optimizes these actions from task-level feedback (Li et al., 2025; Xu et al., 2026). However, most of them strengthen text reasoning around coded clinical concepts already present or retrieved, and they overlook how to teach LLMs to identify useful evidence and interpret retrieval before making predictions. For instance, knowing heart rate matters does not tell the model what value to use or how it affects risk. Hence, our goal is to address both by using tree models to supervise evidence selection and anomaly identification.

## 3 Methodology

In this work, we introduce TIR-EC to learn which clinical evidence to retrieve and how to use it for prediction. Figure 2 gives an overview: the agent queries clinical tools, tree-based supervision scores its evidence choices and predictions, clinical guidance fades during training, and our customized reward updates the policy from grouped rollouts.

## 3.1 Task Definition

Let $q$ identify a prediction task, c denote the evidence cutoff, h denote the prediction horizon, and $y \in \{ 0 , 1 \}$ denote the observed outcome. The clinical data $E _ { \leq c }$ is visible to the model and clinical functions. The initial view $x _ { 0 } = g _ { 0 } ( E _ { \leq c } )$ contains demographics, triage vital signs, and the chief complaint. A set of clinical functions $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ provides selective access to the remaining record before the cutoff. At interaction step k, the policy writes short reasoning $r _ { k }$ , selects a function $a _ { k } \in \mathcal T$ , and receives its observation $o _ { k }$ . After at most K calls, it returns a probability $p \in [ 0 , 1 ]$ for the positive outcome. We denote the resulting trajectory as

$$
\begin{array} { c } { \tau = ( x _ { 0 } , r _ { 1 } , a _ { 1 } , o _ { 1 } , \dots , r _ { m } , a _ { m } , o _ { m } , } \\ { r _ { m + 1 } , p ) , \qquad m \le K . } \end{array}\tag{1}
$$

In each rollout, the model writes untagged reasoning and issues a <tool\_call> block; the environment returns evidence in a <tool\_response> block. The final probability is enclosed in an <answer> block. Appendix B.2 describes the full interaction protocol.

The policy thus controls both evidence acquisition and prediction. Training maximizes the expected terminal reward:

$$
\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ R ( \tau ) \right] .\tag{2}
$$

## 3.2 Reward Design and RL Training

The outcome label indicates whether the final decision is correct, but not which evidence was useful. Prior to LLM training, we therefore train a GBM (Natekin and Knoll, 2013) on structured features available to provide patient risk and identify task-relevant functions. We map its feature importance to the functions that retrieve each feature family and retain those whose normalized importance exceeds γ, yielding a reference set $B _ { q }$ . These signals are incorporated into the following reward components for policy optimization:

![](images/899be65d1bf6669ab67bd330972a8ba5c0a262712775934947771a13876b2fd0.jpg)  
Figure 2: Overview of TIR-EC. The agent retrieves patient evidence through clinical tools and predicts an outcome. Tree models provide tool targets and risk signals, while clinical skill guidance fades during training. Alternative trajectories for the same patient share retained guidance and receive rewards for agentic RL updates.

Format reward. $R _ { \mathrm { f o r m a t } }$ checks the required output and call format:

$$
R _ { \mathrm { f o r m a t } } ( \tau ) = \left\{ \begin{array} { l l } { 1 , \quad \mathrm { i f ~ a n s w e r ~ a n d ~ a l l ~ f u n c t i o n } } \\ { \hfill \mathrm { c a l l s ~ a r e ~ p a r s e a b l e } , } \\ { 0 , \quad \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

Class-safe prediction reward. Let $p _ { i } ^ { \mathrm { t r e e } }$ be the calibrated risk from the gradient boosting model for case i. Directly copying this probability can teach the wrong class when the model is mistaken, while the hard label alone provides no graded notion of risk. We keep the tree confidence only on the label’s side of the decision boundary:

$$
\begin{array} { r } { t _ { i } = \left\{ \begin{array} { l l } { \mathrm { m a x } ( 0 . 5 , p _ { i } ^ { \mathrm { t r e e } } ) , } & { y _ { i } = 1 , } \\ { \mathrm { m i n } ( 0 . 5 , p _ { i } ^ { \mathrm { t r e e } } ) , } & { y _ { i } = 0 , } \end{array} \right. } \\ { R _ { \mathrm { p r e d } } ( \tau _ { i } ) = 1 - ( p _ { i } - t _ { i } ) ^ { 2 } . \quad } \end{array}\tag{4}
$$

Hence, an estimate on the opposite side is moved to 0.5 instead of reversing the observed class.

Function matching reward. Let $C _ { i }$ be the set of structured functions called in $\tau _ { i } .$ . We remove

functions whose information is already shown in $x _ { 0 }$ and compare the remaining set with $B _ { q } \colon$

$$
R _ { \mathrm { m a t c h } } ( \tau _ { i } ) = \frac { | C _ { i } \cap B _ { q } | } { | C _ { i } \cup B _ { q } | } .\tag{5}
$$

This guides which functions are useful for a task, but it does not require a sequence. Functions that return detailed values or text are excluded because the decision tree does not observe these contents.

Execution reward. $R _ { \mathrm { e x e c } }$ checks the interaction as a whole:

$$
R _ { \mathrm { e x e c } } ( \tau ) = \mathbb { I } _ { \mathrm { e x e c } } ( \tau ) ,\tag{6}
$$

where $I _ { \mathrm { e x e c } } ( \tau )$ indicates whether the trajectory is complete and all calls are parseable, within budget, and error-free. Unlike the format term, it depends on whether the calls execute successfully and stay within the interaction budget.

Total reward. We combine the prediction and function signals into a correctness reward $R _ { \mathrm { c o r r } } ,$ then add checks for output format and execution:

$$
\begin{array} { r l } & { R _ { \mathrm { c o r r } } = \alpha R _ { \mathrm { p r e d } } + ( 1 - \alpha ) R _ { \mathrm { m a t c h } } , } \\ & { R ( \tau ) = \lambda _ { f } R _ { \mathrm { f o r m a t } } + \lambda _ { c } R _ { \mathrm { c o r r } } + \lambda _ { e } R _ { \mathrm { e x e c } } , } \end{array}\tag{7}
$$

where $\alpha$ balances prediction and function selection, while $\lambda _ { f } , \lambda _ { c } ,$ and $\lambda _ { e }$ weight the three reward groups. Concretely, we optimize the policy with

Group Relative Policy Optimization (GRPO) (Shao et al., 2024). Appendix B.1 details the grouprelative advantage and policy optimization objective.

## 3.3 Skill Curriculum

For each task $q ,$ physicians can provide a skill document $ { \boldsymbol { S } } _ { q }$ containing general guidance for selecting and interpreting evidence. To internalize the skill content into the trained LLM, we use a reverse curriculum learning method with random dropout. At each optimizer step s, skill sentences are retained independently with probability

$$
q _ { s } = \operatorname* { m a x } \Bigl ( 0 , 1 - \delta \left\lfloor \frac { s } { \Delta } \right\rfloor \Bigr ) ,\tag{8}
$$

where $\Delta$ is the update interval and $\delta$ is the retention, decreasing at each update. Training starts with the complete skill and removes sentences in stages until no guidance remains.

We sample one subset for the G trajectories of a case, so group-relative rewards compare policy behavior under identical guidance. Validation and test prompts never contain skills; gradual removal therefore encourages the policy to internalize the guidance rather than rely on the prompt.

## 4 Experiments

## 4.1 Experimental Setups

Dataset & Task Formulation. We conduct experiments on MC-MED (Kansal et al., 2025), containing 118,385 adult emergency-department visits from 70,545 patients, using the patient-disjoint chronological split of 78%/11%/11% for training, validation, and test. The dataset includes demographics, triage information, diagnoses, medications, orders, lab results, radiology reports, physiological signals, and waveform summaries. We evaluate three binary tasks: disposition, predicting admission versus discharge at 1, 2, or 4 hours; decompensation, predicting HR > 110, MAP < 65, or $\mathrm { S p O _ { 2 } } < 9 0 \%$ within 60, 90, or 120 minutes after a 15-minute assessment; and revisit, predicting ED return within 3, 7, or 14 days after discharge. More details are shown in Appendix A.1.

Implementation Details. We use Qwen3- 4B (Qwen Team, 2025a) without thinking mode as the backbone and train one shared policy across the three tasks. The agent selects up to 12 tool calls from 18 feature groups, with multi-turn interaction and RL implemented using verl, verl-tool, and vLLM. Skill sentences are fully retained for the first 20 steps, then their retention probability decreases by 0.05 every 20 steps until reaching zero at step 400. No skills are used for validation or test. Training uses 4 NVIDIA H100 80GB GPUs. Additional hyperparameters are provided in the Appendix A.2.

Baselines. We compare TIR-EC with the following baselines: (i) ML and DL Models: XGBoost (Chen and Guestrin, 2016), BEHRT (Li et al., 2020), MC-BEC (Chen et al., 2023), TransformerEHR (Yang et al., 2023), and TabM (Gorishniy et al., 2025); (ii) Small and Domain-Specific LLMs: Llama 3.3 70B (Meta, 2024), Gemma 4 31B (Gemma Team, 2026), GPT-OSS 120B (OpenAI, 2025), m1- 7B (23K) (Huang et al., 2026), HuatuoGPT-o1- 8B (Chen et al., 2025), Baichuan-M1-14B (Wang et al., 2025), MedGemma-27B (Sellergren et al., 2025), Qwen3.5-27B (Qwen Team, 2026a), and QwQ-32B (Qwen Team, 2025b); (iii) Large and Frontier LLMs: DeepSeek V4 Flash (DeepSeek-AI, 2026), Gemini 3.7 Flash (Google DeepMind, 2026), GLM-5.2 (Z.ai, 2026), GPT-5.6 Sol (OpenAI, 2026), Claude Opus 5 (Anthropic, 2026), Grok 4.6 (SpaceXAI, 2026), Qwen 3.8 2.4T (Qwen Team, 2026b), and MiniMax M2.7 (MiniMax, 2026); and (iv) Backbone and Interface Controls: Qwen3-4B (Qwen Team, 2025a) under both textonly and agentic settings.

## 4.2 Main Results

At all metrics, TIR-EC beats ML and achieves best or competitive scores with large frontier LLMs with only 4B parameters. As shown in Tab. 1, TIR-EC achieves the highest macro F1 of all models and AUROC second to Opus 5. Its advantage is largest on decompensation and revisit, obtaining the best F1 on both tasks and the best AUPRC on decompensation. Compared with its own backbone, TIR-EC improves F1, AUPRC, and AUROC on every task by a wide margin. Apart from TIR-EC, only four frontier LLMs (GPT-5.6 Sol, Claude Opus 5, Grok 4.6, and Gemini 3.7 Flash) surpass the ML baselines in macro AUPRC and AUROC, and all of them except Claude Opus 5 still fall behind these baselines in macro F1.

Existing LLMs struggle with decompensation and revisit, and neither scale nor medical training fixes this. On decompensation and revisit, the baselines fail in two opposite ways. Most generalpurpose LLMs, frontier models included, are too conservative. They rarely predict a positive outcome, so their precision can be high while their recall stays very low. Many medical and reasoningoriented LLMs do the reverse, flagging almost every patient as positive, which gives high recall but ranking quality that is often near chance. Medical specialization does not help either, as we observe that medical LLMs generally rank patients worse than general-purpose models of similar size. In contrast, TIR-EC alleviates both failure modes.

<table><tr><td></td><td colspan="5">Disposition</td><td colspan="5">Decompensation</td><td colspan="5">Revisit</td><td colspan="5">Macro avg.</td></tr><tr><td>Model</td><td>P</td><td>R</td><td>F1</td><td>AUPRC AUROC</td><td></td><td>P</td><td>R</td><td>F1</td><td>AUPRC</td><td>AUROC</td><td>P</td><td></td><td>R</td><td>F1</td><td>AUPRC AUROC| F1</td><td></td><td></td><td></td><td>AUPRC AUROC</td><td></td></tr><tr><td colspan="9">ML &amp; DL Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>XGBoost</td><td>|62.384.371.6</td><td></td><td></td><td></td><td>80.2</td><td>27.9</td><td>41.7</td><td>33.5</td><td>26.9</td><td></td><td></td><td></td><td>60.6</td><td>40.4</td><td>26.6</td><td>60.9</td><td>|48.5</td><td></td><td></td><td>68.8</td></tr><tr><td>BEHRT</td><td>70.2 76.7 73.3</td><td></td><td></td><td>75.9 82.3</td><td>82.8</td><td>21.0</td><td>61.1</td><td>31.2</td><td>28.8</td><td></td><td>65.2 65.8</td><td>|30.3 24.3</td><td>55.1</td><td>33.8</td><td>23.7</td><td></td><td>57.5</td><td>46.1</td><td>43.1 44.9</td><td>68.7</td></tr><tr><td>MC-BEC</td><td>70.5 77.5 73.8</td><td></td><td></td><td>82.6</td><td>83.0</td><td>20.4</td><td>55.6</td><td>29.9</td><td>33.0</td><td></td><td>64.0</td><td>22.3</td><td>51.0</td><td>31.1</td><td>27.8</td><td>59.3</td><td></td><td>44.9</td><td>47.8</td><td>68.8</td></tr><tr><td>TransformerEHR</td><td></td><td>66.1 68.3 67.2</td><td></td><td>68.1</td><td>72.2</td><td>24.3</td><td>25.0</td><td>24.7</td><td>19.6</td><td></td><td>53.6</td><td>21.4</td><td>63.3</td><td>32.0</td><td>21.2</td><td></td><td>52.6</td><td>41.3</td><td>36.3</td><td>59.5</td></tr><tr><td>TabM</td><td></td><td>76.1 62.5 68.7</td><td></td><td>76.9</td><td>79.8</td><td>19.1</td><td>72.2</td><td>30.2</td><td>29.7</td><td></td><td>66.1</td><td>22.4</td><td>34.7</td><td>27.2</td><td>23.9</td><td>55.5</td><td></td><td>42.0</td><td>43.5</td><td>67.1</td></tr><tr><td colspan="17">Small &amp; Domain-Specific LLMs</td></tr><tr><td>Llama 3.3 70B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma 4 31B</td><td></td><td>56.4 93.8 70.4</td><td></td><td>65.7</td><td>72.5 84.4</td><td>40.0</td><td>5.6</td><td>9.8</td><td>24.1</td><td></td><td>58.4</td><td>14.3</td><td>8.2</td><td>10.4</td><td>18.5</td><td>44.7</td><td></td><td>30.2</td><td>36.1</td><td>58.5</td></tr><tr><td>GPT-OSS 120B</td><td></td><td>78.2 71.7 74.8</td><td></td><td>81.3 83.9</td><td>84.4</td><td>66.7 50.0</td><td>11.1 5.6</td><td>19.1 10.0</td><td>29.8 23.7</td><td></td><td>61.7</td><td>21.4</td><td>6.1</td><td>9.5</td><td>24.2 24.3</td><td>57.7 59.7</td><td></td><td>34.5</td><td>45.1</td><td>67.9 67.6</td></tr><tr><td>m1-7B (23K)</td><td></td><td>73.9 78.2 76.0 48.098.3</td><td>64.5</td><td>60.2</td><td>61.6</td><td>15.5</td><td>94.6</td><td>26.6</td><td>14.0</td><td></td><td>58.6 44.7</td><td>28.6 22.3</td><td>8.3 92.7</td><td>12.9 36.0</td><td>23.9</td><td></td><td>47.8</td><td>33.0 42.4</td><td>44.0 32.7</td><td>51.4</td></tr><tr><td>HuatuoGPT-o1-8B</td><td></td><td>48.896.6 64.9</td><td></td><td>53.2</td><td>56.9</td><td>20.6</td><td>44.6</td><td>28.2</td><td>17.2</td><td></td><td>54.7</td><td>24.0</td><td>85.3</td><td>37.5</td><td>21.5</td><td></td><td>51.2</td><td>43.5</td><td>30.6</td><td>54.3</td></tr><tr><td>Baichuan-M1-14B</td><td>59.8</td><td>85.7 70.4</td><td></td><td>65.9</td><td>72.9</td><td>26.8</td><td>54.1</td><td>35.9</td><td>21.0</td><td></td><td>61.5</td><td>22.5</td><td>93.6</td><td>36.3</td><td>22.5</td><td></td><td>48.1</td><td>47.5</td><td>36.5</td><td>60.8</td></tr><tr><td>MedGemma-27B</td><td>58.9</td><td>84.6 69.4</td><td></td><td>65.3</td><td>71.1</td><td>23.8</td><td>45.9</td><td>31.3</td><td>21.5</td><td></td><td>59.2</td><td>22.5</td><td>100.0</td><td>36.7</td><td>23.1</td><td></td><td>48.9</td><td>45.8</td><td>36.6</td><td>59.7</td></tr><tr><td>Qwen3.5-27B</td><td>62.1 95.1 75.2</td><td></td><td></td><td>61.5</td><td>72.5</td><td>16.5</td><td>90.5</td><td>28.0</td><td>24.5</td><td></td><td>56.9</td><td>26.2</td><td>67.0</td><td>37.6</td><td>27.8</td><td>57.1</td><td></td><td>46.9</td><td>37.9</td><td>62.2</td></tr><tr><td>QwQ-32B</td><td>58.4 85.9 69.5</td><td></td><td></td><td>61.9</td><td>68.9</td><td>15.1</td><td>100.0</td><td>26.2</td><td>19.2</td><td></td><td>53.6</td><td>22.4</td><td>97.2</td><td>36.4</td><td>22.5</td><td>50.3</td><td></td><td>44.0</td><td>34.5</td><td>57.6</td></tr><tr><td>Large &amp; Frontier LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepSeek V4 Flash</td><td></td><td>69.387.077.2</td><td></td><td>85.3</td><td>86.0</td><td>62.5</td><td>13.9</td><td>22.7</td><td>28.8</td><td></td><td>59.1</td><td>|26.9</td><td>36.7</td><td>31.0</td><td>25.4</td><td>58.7</td><td></td><td>43.6</td><td>46.5</td><td>67.9</td></tr><tr><td>Gemini 3.7 Flash</td><td></td><td>78.4 78.8 78.6</td><td></td><td>87.1</td><td>88.4</td><td>40.0</td><td>33.3</td><td>36.4</td><td>37.4</td><td></td><td>64.0</td><td>66.7</td><td>4.1</td><td>7.7</td><td>33.9</td><td></td><td>64.8</td><td>40.9</td><td>52.8</td><td>72.4</td></tr><tr><td>GLM-5.2</td><td>83.5</td><td>76.1 78.377.2 71.7 77.1</td><td></td><td>84.3 88.9</td><td>85.4 89.2</td><td>50.0 83.3</td><td>11.1 13.9</td><td>18.2 23.8</td><td>28.3 37.6</td><td></td><td>59.2 69.2</td><td>29.6 33.3</td><td>32.7 2.0</td><td>31.1 3.9</td><td>26.1 31.4</td><td></td><td>58.3 62.5</td><td>42.2 34.9</td><td>46.2 52.6</td><td>67.6 73.6</td></tr><tr><td>GPT-5.6 Sol Claude Opus 5</td></table>

Table 1: Results on three emergency-care prediction tasks. All LLMs are evaluated in the same agentic feature selection setup as TIR-EC, except Qwen’s "Full Features" rows intakes all features for comparison.

Both the feature selection and understanding components of TIR-EC matter. Removing either the tool-matching reward, which guides feature selection, or the skill curriculum, which targets feature understanding, lowers performance across all three tasks. Removing the GBM probability target mainly hurts ranking quality.

## 5 Analysis

## 5.1 Does TIR-EC better select features?

To separate selection from interpretation, we hold the model fixed and vary only who chooses the evidence. Under a budget of k feature groups, the agent requests up to k groups and predicts from what it retrieved. As a control, the same model predicts from k groups drawn at random.

TIR-EC’s feature selection outperforms random selection at every budget, and the improvement is largest when the evidence is scarce (low budget), as shown in Fig. 3 (a). Within only a few picks, it comes close to the accuracy it reaches with a full budget. The base model shows the reverse pattern, as its own picks fall below random selection beyond its first request.

We then remove the budget limit and compare TIR-EC with open-weight general and medical LLMs, in terms of accuracy against the number of tool calls each model makes (Fig. 3 (b)). TIR-EC attains the highest AUROC while issuing fewer calls than most baselines; and models that call more tools are not necessarily more accurate.

## 5.2 Does TIR-EC better interpret features?

To separate reading from selection, we give every model the same k randomly drawn feature groups for the same patients. Any difference in accuracy then comes from how the evidence is interpreted (Fig. 5).

![](images/3c87ee4a4e7e959fe552c81e243bee9b3eacc07de9ac813654ab1a0f86c2bb27.jpg)

![](images/621e5469b14191e22f30059e966ce0642accd147710b6ccd4f5eea47d8df3a7f.jpg)  
Figure 3: Selective evidence acquisition. (a) Disposition AUROC when the agent may request up to k feature groups (solid) or is given k random groups (dotted). (b) Macro AUROC versus mean tool calls per patient.

With a single group, TIR-EC is no better than its base model. As groups are added, the two diverge: TIR-EC improves steeply and nears its ceiling with about half of the groups, while the base model gains only slowly. TIR-EC’s advantage lies not in reading any single measurement better, but in combining many measurements into one judgment. Medical-domain models do not show this ability.

## 5.3 Are feature selection and interpretation independent abilities?

Sec. 5.1 and 5.2 left two observations about the base model: its own choices are worse than random, and it extracts little from additional evidence. We ask whether these are connected. For each feature group and each model, we measure two quantities: how much the group improves that model’s own predictions when present, and how often the model requests the group under a budget of four. If selection is guided by what a model can use, the two should have a high correlation.

![](images/75ddc1438b78a5b0c23af2eceef21a42780807ccc5a95457a530aab0d8b07e58.jpg)  
Figure 4: Feature interpretation helps selection. Each arrow is one feature group, from Qwen3-4B (tail) to TIR-EC (head). Only commonly selected features are shown. Lines: least-squares fit across groups.

![](images/3fe7ed189fdf13b75532f3ba6e3cff79327cc6975bab17305eb13f8cb9b56641.jpg)  
Figure 5: Feature interpretation. Every model receives the same k randomly drawn feature groups for each patient. Lines show macro AUROC over the three tasks, averaged over 10 random draws at each k.

As shown in Fig. 4, the base model barely requests the features it understands most. It requests blood pressure (BP) for most patients even though this is the group it benefits from least, and almost never requests orders. After training, requests closely track usable value. TIR-EC asks for the evidence it can turn into accuracy. Training also changed what is usable. Nearly every group contributes more to TIR-EC than to its base model, and requests shifted toward the groups that gained most. As "orders" became more useful to the model, the model went from almost never requesting it to requesting it routinely. Therefore, interpretation and selection improve together: as a group becomes more useful to the agent, the agent asks for it more.

![](images/22e5d8b8cc8b578b5e5cfaea381b66087b1b5ebb64f6087eea0118b9ac331903.jpg)  
Figure 6: Barcode plot of tool call (top features for GBM) patterns for (Left) TIR-EC, (Middle) GBM, (Right) Qwen-4B base.

## 5.4 Is TIR-EC merely mimicking the GBM teacher?

Both of TIR-EC’s training signals come from a tree teacher. Does TIR-EC simply reproduce the teacher’s choices? To answer this, we visualize tool calls that TIR-EC, the tree, and the base model make for 100 patients per task (Fig. 6). The five tools on the right return textual details and only the LLMs can access them.

TIR-EC adapts its evidence to the task and to the patient: the base model asks for everything, and the tree always puts high importance on the same few groups. TIR-EC keeps what the teacher taught it. For each task, the groups the tree relies on are among those TIR-EC requests most often. It then goes further and opens the detailed tools the tree cannot access. For disposition, for example, both rely on the orders group, but TIR-EC also reads which orders and diagnoses lie behind it.

## 5.5 Case Study

How do the advantages of TIR-EC show up for individual patients? Table 2 compares TIR-EC with GPT-5.6 Sol, the strongest model on disposition, on two illustrative cases. In the first case, an older woman with an ankle fracture has one abnormal lab value and otherwise stable findings. GPT-5.6 Sol anchors on the fracture and the low hemoglobin and predicts admission. TIR-EC weighs the same abnormalities against the rest of the record, which are mostly normal labs and orders aimed at diagnosis and pain relief rather than inpatient care. It then predicts discharge, which matches the outcome. This shows that TIR-EC judges risk from the whole record rather than only from its most alarming finding, which is the feature integration ability measured in Sec. 5.2. In the second case, a man arrives after a motor vehicle crash with stable vital signs. Both models begin by checking the same vital signs and both correctly predict no deterioration. TIR-EC stops once those signs are reassuring. GPT-5.6 Sol goes on to review orders, labs, imaging, and waveforms before reaching the same answer. TIR-EC requests enough evidence rather than exhausting the feature list, illustrating the selective acquisition measured in 5.1.

![](images/7b82246d49324592f49cd68a8aa2582e24e084988a8105842e63e37a6ab51436.jpg)  
Table 2: Case study of TIR-EC versus GPT.

## 6 Conclusion

We traced LLMs’ underperformance on EHRbased emergency care prediction to poor feature selection and unreliable numerical interpretation. We proposed TIR-EC, which learns to select EHR features as tools through tree-guided reinforcement learning and distills clinical skills into the model via a reverse curriculum. A 4B TIR-EC agent outperforms conventional ML baselines and is competitive with much larger frontier LLMs. Our analysis also shows that feature selection and interpretation improve together.

## 7 Limitations

Single-dataset evaluation. Our experiments are conducted on MC-MED, which provides a rich multimodal view of emergency care but represents a particular data collection setting and clinical workflow. Hospitals may differ in patient populations, measurement practices, documentation styles, and available data sources. As a result, the learned evidence-selection policy and risk interpretation may not transfer directly to other institutions. External validation across health systems is an important next step.

Task and label dependence. We study three post-arrival prediction tasks: disposition, decompensation, and revisit. These outcomes cover different stages of emergency care, but they do not represent the full range of clinical decisions. Some labels, especially disposition and revisit, can also reflect local practice patterns, resource availability, and downstream care in addition to patient physiology. Extending TIR-EC to other outcomes and prediction horizons would provide a broader test of its generality.

Dependence on tree-guided supervision. TIR-EC uses tree models to provide task-level guidance for evidence selection and graded prediction signals during training. This supervision is useful because it transfers data-driven statistical structure into the LLM, but it can also inherit dataset-specific correlations from the teacher model. In addition, our function-matching target identifies informative evidence sources at the task level rather than specifying an optimal retrieval path for every individual patient. Future work may study more adaptive or patient-specific forms of supervision.

Tool interface design. Our agent operates over a predefined set of clinical functions that expose different parts of the patient record. This design makes evidence acquisition explicit and measurable, but real EHR systems differ substantially in data schemas, tool granularity, and access constraints. Deploying the same policy in another system may therefore require remapping or redesigning the available functions, even when the underlying prediction task remains unchanged.

Residual numerical reasoning errors. Training improves how the model combines numerical evidence at the population level, but it does not guarantee correct numerical reasoning for every trajectory. Our analysis still finds occasional cases in which values are retrieved correctly but compared with a threshold in the wrong direction. Such local errors are especially important in clinical applications and suggest that learned numerical interpretation may benefit from additional verification mechanisms.

Retrospective evaluation. Our results measure predictive performance and tool-use behavior on retrospective records. They do not establish that the system improves clinician decisions, workflow efficiency, or patient outcomes when used prospectively. Clinical deployment would require prospective evaluation under real workflow constraints, including calibration, latency, alert burden, and human interaction with model outputs.

## 8 Ethical Considerations

TIR-EC is developed as a research framework for studying evidence selection and prediction from electronic health records. The reported results should not be interpreted as evidence that the system is ready for direct use in patient care.

Clinical safety. Incorrect predictions may lead to inappropriate escalation or false reassurance if model outputs are used without review. This risk is particularly relevant because strong aggregate performance does not eliminate individual failures, including the numerical interpretation errors observed in our trajectory analysis. Any clinical use would require independent validation, calibrated decision thresholds, appropriate human oversight, and mechanisms for detecting unreliable outputs.

Bias and distribution shift. Clinical records and outcome labels reflect both patient conditions and the health-care processes that produced them. As a result, learned prediction patterns may reproduce differences in access, treatment, documentation, or resource use across patient groups and institutions. Tree-guided supervision does not remove this issue and may transfer such correlations into the trained policy. Deployment therefore requires subgroup evaluation and continued monitoring under the target population and clinical setting.

Interpretability of agent behavior. TIR-EC produces observable tool calls and short naturallanguage reasoning, which can help inspect what evidence the model used. However, these trajectories should not be treated as faithful explanations. For clinical use, evidence provenance and verifiable checks should be preferred over relying on generated rationales alone.

Intended use. The goal of this work is to study how reinforcement learning can improve evidence acquisition and interpretation for clinical prediction. We do not recommend using TIR-EC as a substitute for clinician judgment or as a standalone basis for admission or discharge decisions.

## References

Juan Miguel Lopez Alcaraz, Hjalmar Bouma, and Nils Strodthoff. 2025. Enhancing clinical decision support with physiological waveforms—a multimodal benchmark in emergency care. Computers in biology and medicine, 192:110196.

Anthropic. 2026. Introducing Claude Opus 5. Accessed October 12, 2026.

Lang Cao, Qingyu Chen, and Yue Guo. 2026. Ehr-rag: Bridging long-horizon structured electronic health records and large language models via enhanced retrieval-augmented generation. arXiv preprint arXiv:2601.21340.

Canyu Chen, Jian Yu, Shan Chen, Che Liu, Zhongwei Wan, Shuang Zhou, Yuan Luo, Rui Zhang, Danielle Bitterman, Fei Wang, et al. 2026. Clinicalbench: Can llms beat traditional ml models in clinical prediction? In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 10567–10577.

Emma Chen, Aman Kansal, Julie Chen, Boyang Tom Jin, Julia Reisler, David E Kim, and Pranav Rajpurkar. 2023. Multimodal clinical benchmark for emergency care (mc-bec): A comprehensive benchmark for evaluating foundation models in emergency medicine. Advances in Neural Information Processing Systems, 36:45794–45811.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, and Benyou Wang. 2025. Towards medical complex reasoning with LLMs through medical verifiable problems. In Findings of the Association for Computational Linguistics: ACL 2025, pages 14552–14573. Association for Computational Linguistics.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM.

Cansu Colakca, Mehmet Ergın, Habibe Selmin Ozensoy, Alp Sener, Selahattin Guru, and Ayhan Ozhasenekler. 2024. Emergency department triaging using chatgpt based on emergency severity index principles: a cross-sectional study. Scientific reports, 14(1):22106.

Limei Cui, Mengting Wang, Ye Xu, Wei Wei, and Xuehong Zhao. 2026. Diagnostic accuracy of large language models for emergency department triage: a systematic review and meta-analysis. BMC Emergency Medicine.

DeepSeek-AI. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence.

Gemma Team. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Google DeepMind. 2026. Gemini 3.7 Flash model card. Accessed October 12, 2026.

Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. 2025. TabM: Advancing tabular deep learning with parameter-efficient ensembling. In The Thirteenth International Conference on Learning Representations.

Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, and Yuyin Zhou. 2026. m1: Unleash the potential of test-time scaling for medical reasoning with large language models. In Proceedings of the Fifth Machine Learningfor Health Symposium, volume 297 of Proceedings ofMachine Learning Research, pages 369–383. PMLR.

Aman Kansal, Emma Chen, Boyang Tom Jin, Pranav Rajpurkar, and David A Kim. 2025. Mc-med, multimodal clinical monitoring in the emergency department. Scientific Data, 12(1):1094.

Maya Kruse, Shiyue Hu, Nicholas Derby, Yifu Wu, Samantha Stonbraker, Bingsheng Yao, Dakuo Wang, Elizabeth M Goldberg, and Yanjun Gao. 2025. Large language models with temporal reasoning for longitudinal clinical summarization and prediction. In EMNLP (Findings), pages 20715–20735.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025. ToRL: Scaling tool-integrated RL. Preprint, arXiv:2503.23383.

Yikuan Li, Shishir Rao, José Roberto Ayala Solares, Abdelaali Hassaine, Rema Ramakrishnan, Dexter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. 2020. BEHRT: Transformer for electronic health records. Scientific Reports, 10(1):7155.

Meta. 2024. Llama 3.3 70B Instruct model card. Accessed October 12, 2026.

MiniMax. 2026. The MiniMax-M2 series: Mini activations unleashing max real-world intelligence. Preprint, arXiv:2605.26494.

Arya Hadizadeh Moghaddam, Drew Ross, Mohsen Nayebi Kerdabadi, Dongjie Wang, and Zijun Yao. 2026. Reprompt: Recurrent prompt tuning for integrating structured ehr encoders with large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 26766–26778.

Alexey Natekin and Alois Knoll. 2013. Gradient boosting machines, a tutorial. Frontiers in neurorobotics, 7:21.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

OpenAI. 2026. GPT-5.6 system card. Accessed October 12, 2026.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2024. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, volume 2024, pages 9695–9717.

Qwen Team. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qwen Team. 2025b. QwQ-32B: Embracing the power of reinforcement learning.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

Qwen Team. 2026b. Qwen3.8-Max: A new bar for coding and cowork.

Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. 2021. Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine, 4(1):86.

Aida Sanatizadeh, Sorouralsadat Fatemi, Reza Mousavi, and Ahmed Abbasi. 2026. Generalization or memorization? multi-agent vs. baseline llms and automl models for tabular classification. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 40099–40132.

Joshua Sebastian, Karma Tobden, and KMA Solaiman. 2025. Llm-assisted emergency triage benchmark: Bridging hospital-rich and mci-like field simulation. arXiv preprint arXiv:2509.26351.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. 2025. MedGemma technical report. arXiv preprint arXiv:2507.05201.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Jieyu Zhang, Hang Wu, Yuanda Zhu, Joyce C Ho, Carl Yang, and May Dongmei Wang. 2024. Ehragent: Code empowers large language models for fewshot complex tabular reasoning on electronic health records. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 22315–22339.

Benjamin Shickel, Patrick James Tighe, Azra Bihorac, and Parisa Rashidi. 2017. Deep ehr: a survey of recent advances in deep learning techniques for electronic health record (ehr) analysis. IEEEjournal of biomedical and health informatics, 22(5):1589–1604.

KMA Solaiman, Joshua Sebastian, and Karma Tobden. 2026. Benchmarking early deterioration prediction across hospital-rich and mci-like emergency triage under constrained sensing. arXiv preprint arXiv:2602.20168.

SpaceXAI. 2026. Introducing Grok 4.6. Accessed October 12, 2026.

Ethan Steinberg, Ken Jung, Jason A Fries, Conor K Corbin, Stephen R Pfohl, and Nigam H Shah. 2021. Language models are an effective representation learning technique for electronic health record data. Journal ofbiomedical informatics, 113:103637.

Liwen Sun, Abhineet Agarwal, Aaron Kornblith, Bin Yu, and Chenyan Xiong. 2024. ED-copilot: Reduce emergency department wait time with language model diagnostic assistance. In Forty-first International Conference on Machine Learning.

Bingning Wang, Haizhou Zhao, Huozhi Zhou, Liang Song, Mingyu Xu, Wei Cheng, Xiangrong Zeng, Yupeng Zhang, Yuqi Huo, Zecheng Wang, Zhengyun Zhao, et al. 2025. Baichuan-M1: Pushing the medical capability of large language models. arXiv preprint arXiv:2502.12671.

Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason Fries, and Nigam Shah. 2023. Ehrshot: An ehr benchmark for few-shot evaluation of foundation models. Advances in Neural Information Processing Systems, 36:67125–67137.

Chaoyi Wu, Pengcheng Qiu, Jinxin Liu, Hongfei Gu, Na Li, Ya Zhang, Yanfeng Wang, and Weidi Xie. 2025. Towards evaluating and building versatile large language models for medicine. npj Digital Medicine, 8(1):58.

Feng Xie, Jun Zhou, Jin Wee Lee, Mingrui Tan, Siqi Li, Logasan S/O Rajnthern, Marcel Lucas Chee, Bib has Chakraborty, An-Kwok Ian Wong, Alon Dagan, et al. 2022. Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data, 9(1):658.

Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, and Hongkun Yu. 2026. Incentivizing agentic reasoning in llm judges via tool-integrated reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 156446–156468.

Zhichao Yang, Avijit Mitra, Weisong Liu, Dan Berlowitz, and Hong Yu. 2023. TransformEHR: Transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records. Nature Communications, 14(1):7857.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. 2025. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244.

Z.ai. 2026. GLM-5.2: Built for long-horizon tasks. Accessed October 12, 2026.

## Appendix

## A Experimental Details

## A.1 Benchmark Datasets.

We conduct our experiments on MC-MED (Kansal et al., 2025), a multi-modal emergency-care dataset containing 118,385 adult emergency-department visits from 70,545 patients. We use the patientdisjoint chronological split, with 78%, 11%, and 11% of the patients assigned to the training, validation, and test sets, respectively. The available encounter data include demographics, triage information, chief complaints, prior diagnoses, home medications, orders, laboratory results, radiology reports, continuously monitored physiological measurements, and waveform-availability summaries. For each prediction instance, we retain only evidence observed by the task-specific cutoff. Here we consider three binary prediction tasks: disposition, which predicts admission versus discharge at 1, 2, or 4 hours after arrival; decompensation, which predicts whether an initially stable patient will develop heart rate above 110, mean arterial pressure below 65, or oxygen saturation below 90% within the following 60, 90, or 120 minutes after a 15-minute assessment period; and revisit, which predicts whether a discharged patient will return to the emergency department within 3, 7, or 14 days. Each encounter–cutoff or encounter–horizon pair is treated as one prediction instance.

## A.2 Implementation Details.

We use Qwen3-4B (Qwen Team, 2025a) without thinking mode as the backbone and train one shared policy across the three prediction tasks. Multi-turn tool interaction and reinforcement learning are implemented with verl, verl-tool, and vLLM. We sample 10,000 training instances for disposition, decompensation, and revisit. At each optimizer step, a batch of 64 prompts generates G = 8 trajectories per prompt, with the actor micro-batch size set to 1 per GPU. We train for 500 optimizer steps with a learning rate of $1 \times 1 0 ^ { - 6 }$ . The allowable prompt and response lengths are up to 6,144 and 8,192 tokens, and each trajectory may make at most 12 clinical-function calls. For policy optimization, we set $\epsilon _ { \mathrm { l o w } } = 0 . 2 , \epsilon _ { \mathrm { h i g h } } = 0 . 2 8$ , the KL coefficient $\beta = 1 0 ^ { - 4 }$ , and the entropy coefficient $\eta = 1 0 ^ { - 4 }$ For reward computation, we use $\alpha = 0 . 7$ to combine prediction and function-matching rewards, with $( \lambda _ { f } , \lambda _ { c } , \lambda _ { e } ) = ( 0 . 1 , 0 . 8 , 0 . 1 )$ for format, correctness, and execution rewards. During training, clinical-skill sentences are fully retained for the first 20 steps, after which their retention probability decreases by 0.05 every 20 steps and reaches zero after step 400. Validation and test prompts do not contain these skill sentences. We sample training trajectories with temperature 1.0 and top-p 1.0, and use greedy decoding for evaluation. Checkpoints are saved every 10 steps, and the taskspecific checkpoint is selected according to validation F1. All training experiments are conducted on 4 NVIDIA H100 80GB GPUs.

## B Tool-Use Training Details

## B.1 RL Training with GRPO

We optimize the policy with Group Relative Policy Optimization (GRPO) (Shao et al., 2024). For each case, the current policy samples a group of G trajectories and evaluates them with Eq. 7.

Group-relative advantage. Let $\mu _ { R }$ and $\sigma _ { R }$ be the mean and standard deviation of the $G$ rewards for the same case. We assign trajectory i the standardized advantage

$$
A _ { i } = \frac { R ( \tau _ { i } ) - \mu _ { R } } { \sigma _ { R } + \epsilon } .\tag{9}
$$

This compares alternative evidence paths for one patient and avoids training a separate value model.

Policy objective. For token $z _ { i , t }$ generated by the policy, let $\rho _ { i , t } = \pi _ { \theta } ( z _ { i , t } \mid z _ { i , < t } ) / \pi _ { \mathrm { o l d } } ( z _ { i , t } \mid z _ { i , < t } )$ Following DAPO (Yu et al., 2025), we use asymmetric clipping, $\bar { \rho } _ { i , t } = \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon _ { l } , 1 + \epsilon _ { h } )$ , and aggregate the loss over generated tokens:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { R L } } = \mathrm { ~ - ~ } \frac { 1 } { \sum _ { i , t } m _ { i , t } } \sum _ { i , t } m _ { i , t } \operatorname* { m i n } ( \rho _ { i , t } A _ { i } , \bar { \rho } _ { i , t } A _ { i } ) } } \\ & { + \beta { \mathcal { D } _ { \mathrm { K L } } } - \eta \mathcal { H } , } \end{array}\tag{10}
$$

where $m _ { i , t }$ selects valid tokens generated by the model. Function observations are masked because the environment, not the policy, produces them; overlong trajectories are also masked. The upper clipping range is larger than the lower range, following DAPO, while $\beta$ and $\eta$ control KL regularization and entropy.

## B.2 TIR Rollout

Figure 7 summarizes the interaction contract used during rollout. It is an explanatory template rather than a verbatim copy of the runtime prompt; each task inserts its outcome definition, cutoff, horizon, and tool descriptions. The interface contains summary tools for broad signals and detail tools for measured values or clinical text. The model can therefore inspect compact evidence first and request the underlying record only when needed.

![](images/6774d62c5f7026d1928a2e7e7853d0a5c9c79dc3520f90b559ecb6a0d9863465.jpg)  
Figure 7: General prompt template and interaction protocol for TIR rollout. The figure summarizes the decision process rather than reproducing the runtime prompt verbatim. Blue denotes model reasoning, green a tool call, brown an environment observation, and purple the final prediction.

The model receives $x _ { 0 }$ and the function schemas, but not a serialized copy of $E _ { \leq c } .$ . It may write a short plan in natural language and then emit one <tool\_call> block containing a JSON function name and an empty argument object. We disable native thinking mode, so reasoning is not wrapped in a <think> tag. Generation pauses at the closing function call tag. The environment parses the call, runs it on the record available by the cutoff, and appends the result as a user message inside <tool\_response>. The model can then revise its plan and request another source. The interaction ends when the model emits one <answer>p</answer> block or reaches the call budget K. Empty observations are preserved because a test or order that is absent by the cutoff can be informative. The prompt asks the model to stop early when further evidence is unlikely to change the prediction.

## C Prompts and Clinical Tools

This section documents the prompts and tool interface used by TIR-EC. We separate the shared instructions, task definitions, training-time skills, and tool catalog. The prompt cards consolidate repeated instructions for readability while preserving the task definitions, numerical thresholds, and output contract. Bracketed fields are filled for each case; they are not patient examples. Appendix B.2 describes how these components form a multi-turn interaction.

## C.1 Shared Prediction Prompt

The system message combines shared evidence-use instructions with one task definition from Appendix C.2. The chat template supplies the function names, descriptions, and argument schemas. The user message supplies the cutoff, horizon, and initial patient view, followed by the probability-output instruction. During training, the retained skill sentences are appended to this user message; evaluation omits them.

## Shared System Instructions

You evaluate an emergency-department encounter using the evidence available at the specified cutoff. You receive an initial patient view, a prediction task, and clinical tools for requesting additional evidence.

[Function names, descriptions, and schemasfrom Appendix C.4.] Each tool has a single purpose and takes no arguments.   
The environment supplies the current case and cutoff.

Reason only from the initial view and observations returned by the tools. Do not invent measurements, findings, or events. Treat an empty response as informative rather than filling in missing values. Use the evidence available by the cutoff to predict the specified outcome; do not treat a future event as an observed fact.

1. Review the available evidence and identify what is still needed.

2. Select a relevant tool, or stop if further evidence is unlikely to change the prediction. Prefer a few targeted calls to an exhaustive search.

3. Interpret the returned evidence before deciding whether to call another tool or answer. Do not repeat the same unsuccessful or empty call.

You may write short, untagged reasoning between calls. Make at most [tool budget] calls; the experiments use a budget of 12. Keep a turn available for the final answer.

Finish with exactly one <answer>p</answer>, where p ∈ [0, 1] is the probability of the positive outcome. Put no text after this block.

## Initial User Message

Case ID: [identifier] Cutoff: [timestamp] Horizon: [task horizon]

Demographics: [age, gender, race, payor, and arrival mode]

Triage Vitals: [temperature, HR, RR, BP/MAP, SpO<sub>2</sub>, and acuity]

Chief Complaint: [presenting complaint]

No external prediction has been provided. Use the available tools to gather additional evidence, then return the probability of [admission / decompensation / revisit]. Write a decimal number between 0 and 1 inside the final <answer> block, not a category word.

![](images/9588237675a427a3c1b74104364bc7acecb5e68a48b0211a2f2ca45cde808aa8.jpg)  
Text-only comparison. The text-only baseline replaces tool access with clinical data placed directly in the user message. Its instructions state that no tools are available and ask the model to reason over the supplied record before returning the task prediction. This changes how evidence is provided; it is not an additional component of TIR-EC.

## C.3 Training-Time Skill Prompts

The curriculum supplies disposition and decompensation skills, but no additional revisit skill. These cards group the runtime sentences by purpose. During training, sentences are sampled independently, kept in order, and shared across a case’s rollout group. Evaluation uses no skills. The numerical rules are benchmark-specific training guidance, not clinical recommendations or definitions of the outcome.

<table><tr><td>Disposition Skill</td></tr><tr><td>Select Evidence Request get_order_counts for the total and category breakdown of orders, and get_med_counts for home-medication</td></tr><tr><td>burden. Use the triage values already shown instead of requesting them again. Interpret the Pace of Care</td></tr><tr><td>Interpret order counts relative to the time since arrival. A typical total is roughly 12 orders at one hour and 20 at four hours. Calculate orders per hour: a moderate total at four hours represents a slow work-up and leans toward discharge, whereas the same total at one hour is less informative.</td></tr><tr><td>Combine Counts and History More than approximately five medication orders together with more than 34 total orders leans toward admission. More</td></tr><tr><td>than approximately 30 total orders also leans toward admission without home medications.</td></tr><tr><td>The presence of home medications and more than approximately four orders per hour leans toward admission even when the total is small. No home medications, at most approximately 22 total orders, and at most four laboratory orders leans</td></tr><tr><td>toward discharge. A greater home-medication burden leans toward admission; older age provides only a mild upward adjustment.</td></tr><tr><td>Read Order Categories Explicitly identify Code Status, Blood Bank, Nursing Transfusion, or REHAB categories when present; the skill treats</td></tr><tr><td></td></tr><tr><td>Do not base disposition primarily on triage vital signs. Never infer disposition from race or ethnicity. Support severity</td></tr><tr><td>them as strong admission signals. A Diet order is a weak signal and should receive little weight. Keep Claims Grounded</td></tr></table>

<table><tr><td>Decompensation Skill</td></tr><tr><td>Select Evidence Request get_vital_hr, get_vital_spo2, get_vital_bp, and get_vital_rr when the corresponding measurements</td></tr><tr><td>are needed. These tools return minimum, maximum, mean, and last values. Inspect peak HR, lowest SpO2, lowest MAP, and peak RR: averages can hide abnormal readings. Compare Values Explicitly</td></tr><tr><td>Before calling a value abnormal, state the measurement and its reference range. The skill uses HR outside 60–100 beats/min,  $\mathrm { S p O _ { 2 } } < 9 4 \%$  , RR outside 12–20 breaths/min, MAP &lt; 65 mmHg, SBP &lt; 100 mmHg, or temperature outside</td></tr><tr><td> $3 6 { - } 3 8 \ ^ { \circ } \mathrm { C } .$  Do not use terms such as hypoxia, tachycardia, or hypotension without a supporting comparison. Combine Physiological Signals</td></tr><tr><td>Low risk: peak HR ≤ 95, lowest  $\mathrm { S p O _ { 2 } } \geq 9 4 \%$  , and lowest MAP &gt; 77 mmHg. The skill identifies this group as common and below the overall event rate.</td></tr><tr><td>Elevated risk: peak HR &gt; 100 or lowest  $\mathrm { S p O _ { 2 } } < 9 4 \%$  , together with at least two abnormalities among  $\mathrm { H R } > 1 0 0 ,$  SpO2 &lt; 94%, MAP &lt; 70 mmHg, and RR &gt; 22.</td></tr><tr><td>Highest risk: the elevated-risk conditions hold, and mean  $\mathrm { H R } > 1 0 2$  or lowest  $\mathrm { S p O _ { 2 } } \leq 9 2 \%$  Avoid Unsupported Shortcuts</td></tr><tr><td>ESI 2 alone is not a high-risk finding. Do not infer physiological risk from monitoring volume, race, or ethnicity. Normal current vital signs imply lower risk, not zero future risk; the outcome concerns the window after the cutoff.</td></tr><tr><td>Keep Claims Grounded Support each risk claim with a retrieved value. State plainly when all measurements fall within their reference ranges.</td></tr></table>

## C.4 Clinical Tool Catalog and Responses

The interface exposes 18 functions: 12 structured-summary functions and 6 functions for detailed values or text. All use an empty argument object. The environment binds the case identifier, cutoff, and task, so the model chooses an evidence source rather than writing a database query. Detailed-value and text functions are excluded from the function-matching reward. Demographics and triage measurements are already present in the initial view and are also excluded from that reward comparison.

<table><tr><td>Function</td><td>Returned Evidence</td></tr><tr><td>get_vital_hr</td><td>Heart-rate minimum, maximum, mean, last value, and number of measurements.</td></tr><tr><td>get_vital_spo2</td><td>The same statistics for oxygen saturation.</td></tr><tr><td>get_vital_bp</td><td>Separate statistics for mean, systolic, and diastolic blood pressure.</td></tr><tr><td>get_vital_rr</td><td>Respiratory-rate statistics.</td></tr><tr><td>get_vital_temp</td><td>Temperature statistics.</td></tr><tr><td>get_triage_vitals</td><td>Triage temperature, HR, RR, SpO2, SBP, DBP, and acuity.</td></tr><tr><td>get_order_counts</td><td>Total number of orders and counts by order type.</td></tr><tr><td>get_lab_counts</td><td>Number of laboratory component results, distinct panels, and results with an</td></tr><tr><td>get_pmh_counts</td><td>abnormality flag. Total prior-history records, number of top categories shown, and number of recent</td></tr><tr><td>get_med_counts</td><td>distinct categories in the history digest. Total active home-medication records, number of classes shown, and number of</td></tr><tr><td>get_rad_counts</td><td>recent distinct medications in the digest. Number of radiology studies and number with an impression.</td></tr><tr><td>get_demographics</td><td>Age, gender, race, ethnicity, arrival mode, payor, and chief complaint.</td></tr></table>

<table><tr><td>Function</td><td>Returned Evidence</td></tr><tr><td>get_waveform_intensity</td><td>Precomputed waveform duration and segment count for each channel type; no raw waveform signal.</td></tr><tr><td>get_lab_values</td><td>Up to 24 most recent results: panel, component, value, units, abnormality flag, and result time.</td></tr><tr><td>get_orders_detail</td><td>Up to 15 most frequent procedure names and their counts.</td></tr><tr><td>get_diagnoses_detail</td><td>Up to 10 frequent prior comorbidity categories with counts, followed by up to 10 additional recent distinct categories.</td></tr><tr><td>get_medications_detail</td><td>Up to 10 recent distinct home medications and their medication classes.</td></tr><tr><td>get_rad_impressions</td><td>Radiology study names, de-identified impression text, and result times.</td></tr></table>

Time and content restrictions. Timestamped records are filtered by availability: physiological measurements use measurement time, orders use order time, and laboratories and radiology use result time. Prior diagnoses and home medications use their recorded dates; home medications must be active at the cutoff. Final disposition, lengths of stay, and subsequent-visit outcomes are not returned. For disposition, explicit admission, bed-request, and ward-transfer orders are filtered from the order tools. Waveform availability is a precomputed encounter-level summary without per-segment time anchors, rather than a time-resolved signal.

Observation rendering. Results are rendered as key–value pairs, a summary table, or a table of records. A response header identifies the function, underlying row count, and cutoff. Detail extractors can retain fewer rows than this header count; the generic record renderer displays at most 50 rows. Empty results remain visible. The example below shows the interface with placeholders, not a real patient or a measured prediction.

<table><tr><td colspan="2">Tool Schema and Observation Format</td></tr><tr><td>Schema Supplied to the Model</td><td>Environment Observation (Placeholders)</td></tr><tr><td>Name: get_vital_hr Description: Heart-rate aggregates (min/max/mean/last/n)</td><td>&lt;tool_response&gt; get_vital_hr (rows: [n],</td></tr><tr><td>over (arrival, cutoff].</td><td></td></tr><tr><td rowspan="3">Parameters: {&quot;type&quot;: &quot;object&quot;, &quot;properties&quot;: {}}</td><td>cutoff: [timestamp])</td></tr><tr><td>measure min max mean last n</td></tr><tr><td>HR [min] [max] [mean] [last] [n] &lt;/tool_response&gt;</td></tr></table>