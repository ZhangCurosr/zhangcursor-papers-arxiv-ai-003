# When to Stop Asking: Factor Coverage Bias and LLM Determinability Judgments with Underspecified Information in Medicine

Yusuke Watanabe<sup>1,2,∗</sup>, Yohei Kobashi<sup>3</sup>, Taisei Tosaki<sup>1</sup>, Takeshi Kojima<sup>3</sup>, Yusuke Iwasawa<sup>3</sup>, Yasushi Okuno<sup>1</sup>, Yutaka Matsuo<sup>3</sup>

<sup>1</sup>Kyoto University, Department of Biomedical Data Intelligence <sup>2</sup>Kyoto University, Department of Cardiovascular Medicine <sup>3</sup>The University of Tokyo usukewatanabe@gmail.com

## Abstract

Clinical decision-making often begins with underspecified information and requires not only gathering missing information but also knowing when to stop. This calls for judging determinability (whether the available informa tion is sufficient for a decision) by considering whether missing information could change the conclusion, not simply the amount of avail able decision-relevant information. Reliance on this amount as a cue for determinability, which we termfactor coverage bias, could un dermine the trustworthiness of large language models (LLMs) through premature conclusions or excessive information gathering, such as unnecessary invasive tests. Yet whether LLMs exhibit this bias remains insufficiently stud ied. To address this gap, we construct a controlled pairwise benchmark in which LLMs identify the determinable case within each determinable–undeterminable pair. We add a decision-relevant factor to either side while keeping each case’s determinability unchanged, and similarly add irrelevant text to distinguish this bias from context-length effects. Adding a relevant factor made LLMs more likely to select the augmented case as determinable even when it remained undeterminable, an effect that persisted with increased reasoning effort, while adding irrelevant text made them less likely to select the augmented case, an effect that reasoning substantially reduced. These findings identify factor coverage bias as a failure mode distinct from context-length effects and a poten tial risk to LLM trustworthiness. Our work provides an evaluation framework for LLM trustworthiness under underspecification. <sup>1</sup>

## 1 Introduction

Recent LLMs have reached near-ceiling performance on benchmarks based on medical licensing examinations (Tordjman et al., 2025; Liu et al., 2026). The challenge is shifting from knowledge to reasoning as experts do in practice (Li et al., 2024; Watanabe et al., 2026). Examinations, however, typically provide the information needed for a decision, whereas clinical practice often involves underspecified information (Hersh et al., 2013; Tobin-Schnittger et al., 2018), limiting what these examinations can reveal about reasoning under such conditions (Hager et al., 2024). In clinical practice, experts tend to focus information gathering on what is needed for a decision, while trainees tend to collect more unnecessary information (Bowen, 2006; McQuade et al., 2024).

![](images/0392d3fc2810e39915d70030f164699ec4f3c68e6fc6cf25e2c17b7570655f86.jpg)  
Figure 1: Concept of factor coverage bias. Determinability depends on whether missing information could change the conclusion. An LLM exhibiting factor coverage bias tends to use the amount of available decision-relevant information as a cue for determinability. This can lead to errors in deciding when to stop gathering information: premature conclusions when many factors are available but still insufficient, and excessive information gathering when few factors already suffice.

When inputs are underspecified, as in patientfacing history taking and decision support for physicians outside their specialty, LLMs need not only to gather missing information but also to know when to stop, avoiding premature conclusions and excessive information gathering. In clinical practice, unnecessary questioning in emergencies or unnecessary invasive testing can harm patients (Grady and Redberg, 2010), so gathering every potentially relevant detail is not necessarily appropriate even if the final conclusion is correct. Prior work examines abstention under insufficient information and gath ering additional information (Machcha et al., 2026; Li et al., 2024), but stopping once information is sufficient has received less attention. Watanabe et al. (2026) defined determinability by whether missing information could change the conclusion and demonstrated failures in LLM determinability judgments, without fully examining which input cues drive them. Failures in determinability judgment may undermine LLM trustworthiness beyond medicine, including coding assistance, legal consultation, and customer support, where robust responses to underspecified inputs require clarifying user intent and gathering necessary information, yet asking about every potentially relevant detail burdens users and reduces efficiency (Fang et al., 2026; Vincent et al., 2026; Mass et al., 2022).

LLM biases, such as sycophancy (Sharma et al., 2024), can undermine trustworthiness by reducing robustness. LLMs are also sensitive to context length and distraction by irrelevant content (Shi et al., 2023; Amiraz et al., 2025; Du et al., 2025). Inputs grow through both relevant information and irrelevant text, especially during retrieval or multi-turn interactions. However, more relevant information does not necessarily imply determinability. Few factors can suffice, while many can still be insufficient. LLMs may still use the amount of available decision-relevant information as a cue for determinability, a tendency we term factor coverage bias (Figure 1). This bias could lead to premature conclusions or excessive information gathering and may require different mitigation from context-length effects, because filtering irrelevant text would not prevent reliance on the amount of relevant information. Adding relevant information also lengthens the context, but whether factor coverage biases determinability judgments independently of context length remains insufficiently studied.

To address these gaps, we construct a controlled pairwise benchmark of 1,080 pairs based on clinical scoring systems to examine the bias clearly. Each pair contains a determinable and an undeterminable case, both with missing information, and LLMs identify which is determinable. We test for factor coverage bias by adding a decision-relevant factor to either side while keeping each case’s determinability unchanged and measuring how the judgment shifts. We similarly add irrelevant text of comparable length to either side to distinguish this bias from context-length effects. Across recent open-weight and proprietary LLMs, adding a relevant factor shifts judgments toward determinable even when the augmented case remains undeterminable, an effect that persists with increased reasoning effort, while adding irrelevant text shifts judgments toward undeterminable, an effect that reasoning substantially reduces. These differences in direction and response to reasoning identify factor coverage bias as a failure mode distinct from context-length effects. Our main contributions are as follows.

• We define and demonstrate factor coverage bias, the tendency to use the amount of available decision-relevant information as a cue for determinability, as a potential risk to LLM trustworthiness.

• We distinguish factor coverage bias from context-length effects through their different directions and responses to reasoning, informing future efforts to develop trustworthy LLMs.

• We release a controlled pairwise benchmark for evaluating factor coverage bias in LLMs.

## 2 Related Work

LLM behavior under underspecified input. Underspecified inputs are a practical challenge in LLM deployment (Yang et al., 2026; Laban et al., 2025). Missing information in the input is distinct from gaps in the model’s knowledge (Ren et al., 2026). Prior work has evaluated LLM responses to underspecified inputs by examining whether they ask clarifying questions (Zhang et al., 2024; Dong et al., 2025; Li et al., 2025; Luo et al., 2025; Fang et al., 2026), abstain from answering (Kirichenko et al., 2025), or flag what is missing (Vincent et al., 2026). Few evaluate whether the conclusion could still change given what is missing (Watanabe et al.,

2026), and systematic biases in this judgment remain insufficiently understood. Our work characterizes these biases.

Heuristics in LLM judgment. Input-driven biases undermine the robustness and trustworthiness of LLMs. Known examples include sycophancy (Sharma et al., 2024) and evaluators’ preference for longer responses (Zheng et al., 2023; Dubois et al., 2024). Adding irrelevant text or increasing context length can reduce accuracy (Shi et al., 2023; Amiraz et al., 2025; Du et al., 2025), but the direction of the resulting answer shifts remains underexplored. Increasing supporting evidence shifts answers toward the supported side (Xie et al., 2024; Naphade, 2026), but these studies do not examine information sufficiency judgments. In these studies, the amount of relevant information and context length also change together, leaving their effects difficult to distinguish, although they may require different mitigation. We address these gaps by comparing additions of relevant information and irrelevant text while holding determinability fixed and examining how judgments shift.

LLM benchmarks in medicine. Medical benchmarks for LLMs have largely assumed complete information, as in those derived from licensing examinations (Jin et al., 2021; Pal et al., 2022). Work on incomplete information has evaluated abstention (Machcha et al., 2026; Cocchieri et al., 2026), requests for additional information (Arora et al., 2025), and interactive information gathering (Li et al., 2024; Schmidgall et al., 2026). Clin-DetBench (Watanabe et al., 2026) evaluates determinability judgment directly, but because its items are constructed by progressively removing information from complete cases, factor coverage and determinability are correlated by construction, so the benchmark cannot detect whether a model responds to factor coverage rather than to determinability. We manipulate the two independently and evaluate this failure mode.

## 3 Methodology

## 3.1 Motivation

We adopt a pairwise format because it controls for each model’s propensity to abstain, which confounds single-case judgments, and because pairwise comparison amplifies superficial cues (Jeong et al., 2025), making bias easier to detect. Case presentations are constructed from clinical scoring systems. The aim is not to evaluate the ability to apply scoring rules; rather, because the judgment is determined by item values and a threshold, the correct label can be constructed uniquely, which yields a robust experimental setting.

Table 1: $\mathrm { C H A D S _ { 2 } }$ scoring system. Total scores range from 0 to $\begin{array} { r } { 6 ; } \end{array}$ we use a threshold of $\theta = 2 .$
<table><tr><td>Item</td><td>Points if positive</td></tr><tr><td>Congestive heart failure</td><td>1</td></tr><tr><td>Hypertension</td><td>1</td></tr><tr><td> $\mathrm { A g e } \geq 7 5$ </td><td>1</td></tr><tr><td>Diabetes mellitus</td><td>1</td></tr><tr><td>Prior stroke or TIA</td><td>2</td></tr></table>

## 3.2 Preliminary: Clinical Scoring Systems

Clinical scoring systems are widely used rule-based tools that assign points to clinical findings and guide decisions such as treatment initiation or risk stratification according to whether the total score reaches a threshold. A representative example, CHADS<sub>2</sub>, assesses the risk of stroke in patients with atrial fibrillation. It assigns one point each to congestive heart failure (C), hypertension (H), age $\ge 7 5 ~ ( \mathrm { A } )$ , and diabetes (D), and two points to a history of stroke or transient ischemic attack $\left( \mathbf { S } _ { 2 } \right)$ , and classifies a total score $\geq 2 \ ( \mathrm { i . e . }$ , threshold $\theta = 2 )$ as high risk (Table 1). We use 15 scoring systems that are recommended in clinical guidelines and have explicit thresholds, each with 4–9 scored items (Table A.2).

## 3.3 Definition of Determinability

When the values of some scored items are missing, the exact total score cannot be computed, but the judgment can still be determined if all possible total scores fall on the same side of the threshold. Let $S _ { \mathrm { m i n } }$ and $S _ { \mathrm { m a x } }$ denote the minimum and maximum possible total scores given the available information, obtained by assigning each item with a missing value its minimum and maximum possible points, respectively. With threshold θ, a case is determinable if $S _ { \mathrm { m i n } } ~ \ge ~ \theta$ or $S _ { \mathrm { m a x } } \ < \ \theta _ { \mathrm { \tiny { i } } }$ , and undeterminable if $S _ { \mathrm { m i n } } < \theta \le S _ { \mathrm { m a x } }$

For example, in $\mathrm { C H A D S _ { 2 } } \left( \theta = 2 \right)$ , a patient aged 80 with a history of heart failure has a total score of at least 2 regardless of the remaining missing items and is therefore determinable. In contrast, a patient with hypertension but without heart failure or diabetes, whose age and stroke history are missing, has a possible total score ranging from 1 to 4 and is therefore undeterminable. Determinability thus depends on whether missing information could change the conclusion, rather than the amount of available relevant information.

![](images/aa1fe0e81c48d77f641305fc97db671cdd871be44c8d7b3e4da57fab397faff7.jpg)  
Figure 2: Overview of our pairwise benchmark. LLMs identify which case is determinable. We add a decisionrelevant factor to either the determinable or the undeterminable case. We similarly add irrelevant text to either case, yielding four conditions in total. Each addition preserves both cases’ determinability. We compare the resulting judgment shifts to distinguish factor coverage bias from context-length effects.

## 3.4 Task Formulation

Figure 2 provides an overview of our benchmark. Each pair contains a determinable and an undeterminable case constructed from the same clinical scoring system. The model identifies, in a forced binary choice, which case contains sufficient information to settle the judgment under that system. Both cases have missing scored items. One is determinable, meaning that the judgment is the same for all possible values of the missing items, while the other is undeterminable, meaning that the judgment depends on those values.

## 3.5 Dataset Construction

Materials. Case presentations are constructed from pre-written text chunks that explicitly indicate whether each documented scored item is positive (scoring) or negative (non-scoring). Table A.3 illustrates these chunks using CHADS . Irrelevant sentences describe medically plausible details, such as drug allergies, current medications, or family history, that do not affect the score (Table A.4). A board-certified internal medicine physician checked the chunks against item definitions and verified that irrelevant sentences neither directly nor implicitly indicated any scored item. For example, irrelevant sentences were checked for mentions of antidiabetic drugs in scoring systems that include diabetes. Determinability labels are assigned automatically by computing the possible total score range $( S _ { \mathrm { m i n } } , S _ { \mathrm { m a x } } )$ and comparing it with the threshold. The prompt template and examples of the actual prompts are provided in Appendix D.

Baseline pairs. A baseline pair consists of a determinable case and an undeterminable case with the same number of missing items and comparable length to minimize differences in information amount and context length at baseline. Eight pairs were sampled uniformly at random within each scoring system, yielding 120 baseline pairs, with correct answers balanced between A and B. Each baseline pair is used in nine conditions, including baseline, yielding 1,080 pairs in total (120 × 9). Baseline cases average 41 tokens, with a mean prompt length of 140 tokens (Table A.5).

Adding a relevant factor. To test for factor coverage bias, we examine whether adding relevant information shifts judgments toward the augmented case as determinable. A relevant factor is realized as a positive or negative chunk for a previously missing scored item (Table A.3), which we add to either case of a baseline pair. Baseline pairs are constructed so that the undeterminable case remains undeterminable after the addition, while the determinable case also retains its label. This increases both the amount of available relevant information and the context length without changing either case’s determinability. The addition averages 8 tokens.

Adding irrelevant text. To distinguish factor coverage bias from context-length effects, we similarly add irrelevant sentences (Table A.4) to either case without changing its determinability. The length-matched condition adds one sentence with the same token count as the added relevant factor for the same scoring system (mean 8 tokens). To examine larger length increases, the 2× and 4× conditions extend the case to approximately twice and four times its original length, adding 43 and 127 tokens on average, respectively. Token counts by condition are reported in Table A.6.

## 3.6 Evaluation

For each model and each condition, we use the error rate against the correct label (1 − accuracy) as the metric. The effect of an addition is quantified by attraction. Because the correct answer is always the determinable side, a tendency to select the side to which text was added increases errors when the addition is made to the undeterminable side and decreases errors when it is made to the determinable side. We therefore define attract $\mathrm { i o n } = \mathrm { E r r _ { U n d } - E r r _ { D e t } }$ , where $\mathrm { E r r _ { U n d } }$ and $\mathrm { E r r } _ { \mathrm { D e t } }$ are the error rates when the text is added to the undeterminable and determinable side, respectively. Attraction is positive when the judgment is drawn toward the added side, negative when the added side is avoided, and near 0 when an addition merely degrades both sides equally (nondirectional noise). We obtain 95% confidence intervals (CIs) by a pair-level paired bootstrap (10,000 resamples, percentile method).

Table 2: Error rates. Det/Und denote the determinable/undeterminable case to which text is added; values in parentheses are changes from baseline. Irrelevant text is length-matched to the added relevant factor. Reasoning indicates the setting used, with max for GPT-5.5 and high for Opus 4.5 representing the highest reasoning levels evaluated; for the Qwen models, none/on correspond to thinking off/on.
<table><tr><td rowspan="2"></td><td rowspan="2">Reasoning</td><td rowspan="2">Baseline</td><td colspan="2">Add relevant factor</td><td colspan="2">Add irrelevant text (matched)</td></tr><tr><td>Det</td><td>Und</td><td>Det</td><td>Und</td></tr><tr><td rowspan="2">Model GPT-5.5</td><td>none</td><td>0.342</td><td>0.233 (−0.108)</td><td>0.525 (+0.183)</td><td>0.475 (+0.133)</td><td>0.267 (−0.075)</td></tr><tr><td>max</td><td>0.017</td><td>0.017 (±0.000)</td><td>0.050 (+0.033)</td><td>0.025 (+0.008)</td><td>0.025 (+0.008)</td></tr><tr><td rowspan="2">Opus 4.5</td><td>none</td><td>0.425</td><td>0.358 (-0.067)</td><td>0.525 (+0.100)</td><td>0.558 (+0.133)</td><td>0.392 (−0.033)</td></tr><tr><td>high</td><td>0.125</td><td>0.042 (−0.083)</td><td>0.233 (+0.108)</td><td>0.142 (+0.017)</td><td>0.133 (+0.008)</td></tr><tr><td rowspan="2">Qwen3.5-27B</td><td>none</td><td>0.342</td><td>0.225 (−0.117)</td><td>0.492 (+0.150)</td><td>0.542 (+0.200)</td><td>0.183 (−0.158)</td></tr><tr><td>on</td><td>0.100</td><td>0.050 (–0.050)</td><td>0.242 (+0.142)</td><td>0.208 (+0.108)</td><td>0.108 (+0.008)</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>none</td><td>0.375</td><td>0.350 (–0.025)</td><td>0.433 (+0.058)</td><td>0.617 (+0.242)</td><td>0.275 (−0.100)</td></tr><tr><td>on</td><td>0.167</td><td>0.100 (−0.067)</td><td>0.233 (+0.067)</td><td>0.192 (+0.025)</td><td>0.217 (+0.050)</td></tr><tr><td rowspan="2">Qwen3.6-35B-A3B</td><td>none</td><td>0.375</td><td>0.367 (−0.008)</td><td> $0 . 5 0 8 \left( + 0 . 1 3 3 \right)$ </td><td>0.500 (+0.125)</td><td>0.308 (−0.067)</td></tr><tr><td>on</td><td>0.200</td><td>0.108 (−0.092)</td><td> $0 . 2 9 2 \ : ( + 0 . 0 9 2 )$ </td><td>0.233 (+0.033)</td><td>0.192 (−0.008)</td></tr></table>

## 4 Experiments

## 4.1 Experimental Setup

We evaluated GPT-5.5, Claude Opus 4.5, and three open-weight models, Qwen3.5-27B, Qwen3.5-9B, and Qwen3.6-35B-A3B, through their providers’ APIs, with the Qwen models accessed via OpenRouter. Reasoning effort was varied through the thinking level for Opus 4.5 (none/low/medium/high), the reasoning-effort parameter for GPT-5.5 (none/low/high/max), and thinking off/on for the Qwen models (reported as none/on). Temperature was left at the provider default, and each instance was queried once. The prompt was a single zero-shot template containing the scoring system, the decision target, and the two cases (Figure A.2); no judgment procedure was supplied, because our question is whether models make the judgment autonomously.

## 4.2 Overall Error Rates

Table 2 reports error rates and changes from baseline. Enabling reasoning reduced error rates across all five models and all conditions shown, with baseline error rates falling from 0.342–0.425 to

0.017–0.200 (see also Figure A.1 in the appendix). Adding a relevant factor to the determinable case reduced or left unchanged the error rate, whereas adding it to the undeterminable case increased the error rate in every model and reasoning setting. With reasoning enabled, adding a relevant factor to the undeterminable case produced the highest error rate for every model among the conditions shown. Without reasoning, irrelevant text showed the opposite pattern, increasing errors when added to the determinable case and decreasing errors when added to the undeterminable case; these changes were generally smaller with reasoning. The next section quantifies these directional shifts as attraction. Results including intermediate reasoning levels and the 2× and 4× conditions are provided in Table A.1.

## 4.3 Factor Coverage Bias Is Distinct from Context-Length Effects

Figure 3 shows the attraction (Section 3.6) and its 95% CI for each model with reasoning off and on, for the addition of one relevant factor (left) and the addition of length-matched irrelevant text (right); on denotes the highest level available (max for GPT-5.5 and high for Opus 4.5).

For the addition of a relevant factor, attraction was positive across all models and reasoning levels, indicating a tendency to select the case with added relevant information as determinable even with reasoning enabled. For GPT-5.5 with reasoning, the 95% CI included zero, but the point estimate remained positive (0.033). In contrast, adding irrelevant text produced negative attraction in all five models without reasoning, indicating a tendency to regard the augmented case as undeterminable. Reasoning substantially reduced this effect. Because the length-matched addition averaged only about 8 tokens, we also added irrelevant text to extend cases to approximately twice and four times their original length (Figure 4). Without reasoning, these larger additions amplified the shift toward undeterminable. Reasoning substantially reduced this effect, although statistically significant negative attraction remained in some settings, including Opus 4.5 at both lengths.

![](images/4ea01ddadfedf434ffd410e4f24a553285e1185dead13d08451be106e49e94f8.jpg)  
∆ no reasoning reasoning  
Figure 3: Attraction with 95% CIs for the addition of one relevant factor and of length-matched irrelevant text. Positive values indicate that the judgment was drawn toward the added side.

LLM judgments shifted in response to input changes despite an unchanged correct answer, indicating a lack of robustness. The shift toward cases with added relevant information suggests that models use the amount of available relevant information as a cue for determinability. These findings demonstrate factor coverage bias. Its direction and response to reasoning distinguish factor coverage bias from context-length effects.

## 5 Discussion

We identify factor coverage bias, the tendency to use the amount of available relevant information as a cue for determinability, as a bias in LLM judgment distinct from context-length effects. By distorting determinability judgments, this bias may undermine LLM trustworthiness and usability through unnecessary clarification or premature conclusions. These risks are particularly relevant to tasks that require robust responses to underspecified inputs, including medicine, coding assistance, legal consultation, and customer support (Li et al., 2024; Fang et al., 2026; Vincent et al., 2026; Mass et al., 2022). In medicine, the consequences may be especially serious, ranging from incorrect decisions through premature conclusions (Graber et al., 2005) to patient harm from unnecessary invasive testing (Ganguli et al., 2019) and burdensome questioning. Trustworthy LLMs therefore require both knowledge and the reasoning abilities that experts use in practice. Judging determinability is one such ability.

![](images/c9367368ddbcf9497f877c3a4089c417dea03dec8994a7082be3044d89720bc6.jpg)  
∆ no reasoning reasoning  
Figure 4: Attraction with 95% CIs for irrelevant text at 2× and 4× case length. Positive values indicate that the judgment was drawn toward the added side.

Given the substantial improvement in determinability judgments with reasoning, allocating inference resources to these judgments is an important design consideration. However, factor coverage bias persisted with reasoning, motivating further training and evaluation using accurately labeled determinable and undeterminable inputs. Such labels are difficult to obtain because experts do not necessarily document their judgments of information sufficiency during information gathering, even when the final conclusion is recorded. Our work addresses this challenge by synthesizing cases with determinability labels, providing an evaluation framework that could be extended to other domains for developing trustworthy LLMs.

## 6 Conclusion

We defined and demonstrated factor coverage bias, the tendency to use the amount of available decision-relevant information as a cue for determinability. This bias may undermine LLM trustworthiness under underspecification through premature conclusions and excessive information gathering. Its direction and response to reasoning distinguish it from context-length effects, informing future efforts to develop trustworthy LLMs. We release a controlled pairwise benchmark for evaluating this bias, providing an evaluation framework for developing trustworthy LLMs.

## Limitations

This study has six limitations. First, the evaluation is limited to five models and to a single synthetic medical benchmark generated by concatenating text templates from clinical scoring systems with explicit decision criteria; it does not include judgments on which clinicians themselves might disagree. In addition, the text chunks were verified by a single physician. Because this verification consists of checking each chunk against explicit item definitions rather than exercising clinical judgment, we expect little inter-rater variability, but we did not measure it.

Second, our evaluation covers only relatively short contexts. Although reasoning substantially reduced negative attraction from irrelevant text, we did not test whether this reduction holds at substantially greater context lengths. The irrelevant-text conditions were designed to distinguish factor coverage bias from context-length effects within our benchmark, rather than to characterize long-context behavior comprehensively.

Third, the pairwise design was chosen to observe the bias while suppressing the influence of each model’s abstention tendency; whether the same bias appears in pointwise judgments of a single case has not been tested. Nor have we tested settings closer to actual use, in which information accumulates sequentially through history taking or retrieval-augmented generation; whether the bias manifests there as premature conclusion of information gathering is a question that directly corresponds to the motivation of this study but remains open.

Fourth, this study targets whether models can make the judgment autonomously, using a single zero-shot template that does not supply the reasoning procedure an expert would follow. Prior work has shown that providing the judgment procedure through few-shot prompting improves the accuracy of determinability judgment (Watanabe et al., 2026), and the present bias would presumably be mitigated as well; however, because our question is whether LLMs apply such an expert procedure on their own, mitigation by externally supplying the procedure is outside the scope of this study.

Fifth, the amount of relevant information could not be varied: adding further relevant factors to the undeterminable case eventually makes both cases of a pair determinable, so the relevant condition was restricted to a single factor. Even a single factor, however, produced a shift comparable to that of irrelevant text (Table 2).

Finally, for GPT-5.5 with reasoning enabled, the error rates were near the floor (0.017–0.050); because attraction is defined as a difference in error rates, the range of values it can take is compressed in this setting. The point estimate was positive, as for the other models, but the confidence interval included 0, and we cannot distinguish whether this reflects the absence of the bias or low statistical power due to the small number of errors.

## Ethical Considerations

Our benchmark consists exclusively of synthetic cases constructed from publicly available scoring criteria and does not include any patient information or private clinical data. This benchmark is intended solely for research and evaluation purposes and is not designed for direct use in clinical practice or patient-facing applications. The proposed framework does not replace clinical judgment or human supervision. To ensure transparency and reproducibility, the dataset is publicly released under the MIT License.

## References

A. Alvarado. 1986. A practical score for the early diagnosis of acute appendicitis. Annals of Emergency Medicine, 15(5):557–564.

American College of Obstetricians and Gynecologists. 2009. ACOG Practice Bulletin No. 107: Induction of Labor. Obstetrics & Gynecology, 114(2 Part 1):386.

Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The Distracting Effect: Understanding Irrelevant Passages in RAG. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18228–18258, Vienna, Austria. Association for Computational Linguistics.

V. Apgar. 1953. A proposal for a new method of evaluation of the newborn infant. Current Researches in Anesthesia & Analgesia, 32(4):260–267.

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and Karan Singhal. 2025. HealthBench: Evaluating Large Language Models Towards Improved Human Health. Preprint, arXiv:2505.08775.

E. H. Bishop. 1964. PELVIC SCORING FOR ELEC-TIVE INDUCTION. Obstetrics and Gynecology, 24:266–268.

S. L. Blamey, C. W. Imrie, J. O’Neill, W. H. Gilmour, and D. C. Carter. 1984. Prognostic factors in acute pancreatitis. Gut, 25(12):1340–1346.

O. Blatchford, W. R. Murray, and M. Blatchford. 2000. A risk score to predict need for treatment for uppergastrointestinal haemorrhage. Lancet (London, England), 356(9238):1318–1321.

R. C. Bone, R. A. Balk, F. B. Cerra, R. P. Dellinger, A. M. Fein, W. A. Knaus, R. M. Schein, and W. J. Sibbald. 1992. Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis. The ACCP/SCCM Consensus Conference Committee. American College of Chest Physicians/Society of Critical Care Medicine. Chest, 101(6):1644–1655.

Judith L. Bowen. 2006. Educational Strategies to Promote Clinical Diagnostic Reasoning. New England Journal ofMedicine, 355(21):2217–2225.

R. M. Centor, J. M. Witherspoon, H. P. Dalton, C. E. Brody, and K. Link. 1981. The diagnosis of strep throat in adults in the emergency room. Medical Decision Making: An International Journal of the Societyfor Medical Decision Making, 1(3):239–246.

Alessio Cocchieri, Luca Ragazzi, Giuseppe Tagliavini, and Gianluca Moro. 2026. LLMs (Almost) Never Abstain Under Medical Uncertainty. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29573–29613, San Diego, California, United States. Association for Computational Linguistics.

Committee on Fetus and Newborn. 2015. The Apgar Score. Obstetrics and Gynecology, 126(4):e52–e55.

Salomone Di Saverio, Mauro Podda, Belinda De Simone, Marco Ceresoli, Goran Augustin, Alice Gori, Marja Boermeester, Massimo Sartelli, Federico Coccolini, Antonio Tarasconi, Nicola De’ Angelis, Dieter G. Weber, Matti Tolonen, Arianna Birindelli, Walter Biffl, Ernest E. Moore, Michael Kelly, Kjetil Soreide, Jeffry Kashuk, and 41 others. 2020. Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines. Worldjournal of emergency surgery: WJES, 15(1):27.

Mingwen Dong, Nischal Ashok Kumar, Yiqun Hu, Anuj Chauhan, Chung-Wei Hang, Shuaichen Chang, Lin Pan, Wuwei Lan, Henghui Zhu, Jiarong Jiang, Patrick Ng, and Zhiguo Wang. 2025. PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 255–273, Albuquerque, New Mexico. Association for Computational Linguistics.

Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Babu Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, and Hao Peng. 2025. Context Length Alone Hurts LLM Performance Despite Perfect Retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 23281–23298, Suzhou, China. Association for Computational Linguistics.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. https://arxiv.org/abs/2404.04475v2.

European Association for the Study of the Liver. 2018. EASL Clinical Practice Guidelines for the management of patients with decompensated cirrhosis. Journal ofHepatology, 69(2):406–460.

Zheng Fang, Dongming Jin, Yihong dong, Yongmin Li, Kechi Zhang, Zhi Jin, and Ge Li. 2026. ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation. Preprint, arXiv:2607.00711.

B. F. Gage, A. D. Waterman, W. Shannon, M. Boechler, M. W. Rich, and M. J. Radford. 2001. Validation of clinical classification schemes for predicting stroke: Results from the National Registry of Atrial Fibrillation. JAMA, 285(22):2864–2870.

Ishani Ganguli, Arabella L. Simpkin, Claire Lupo, Arlene Weissman, Alexander J. Mainor, E. John Orav, Meredith B. Rosenthal, Carrie H. Colla, and Thomas D. Sequist. 2019. Cascades of Care After Incidental Findings in a US National Survey of Physicians. JAMA Network Open, 2(10):e1913325.

Mark L. Graber, Nancy Franklin, and Ruthanna Gordon. 2005. Diagnostic Error in Internal Medicine. Archives ofInternal Medicine, 165(13):1493–1499.

Deborah Grady and Rita F. Redberg. 2010. Less Is More: How Less Health Care Can Result in Better Health. Archives of Internal Medicine, 170(9):749– 750.

Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios Kaissis, and Daniel Rueckert. 2024. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nature Medicine, 30(9):2613–2622.

William R. Hersh, Mark G. Weiner, Peter J. Embi, Judith R. Logan, Philip R.O. Payne, Elmer V. Bernstam, Harold P. Lehmann, George Hripcsak, Timothy H. Hartzog, James J. Cimino, and Joel H. Saltz. 2013. Caveats for the Use of Operational Electronic Health Record Data in Comparative Effectiveness Research. Medical care, 51(8 0 3):S30–S37.

Gerhard Hindricks, Tatjana Potpara, Nikolaos Dagres, Elena Arbelo, Jeroen J. Bax, Carina Blomström-Lundqvist, Giuseppe Boriani, Manuel Castella, Gheorghe-Andrei Dan, Polychronis E. Dilaveris, Laurent Fauchier, Gerasimos Filippatos, Jonathan M. Kalman, Mark La Meir, Deirdre A. Lane, Jean-Pierre Lebeau, Maddalena Lettino, Gregory Y. H. Lip, Fausto J. Pinto, and 6 others. 2021. 2020 ESC Guidelines for the diagnosis and management of atrial fibrillation developed in collaboration with the European Association for Cardio-Thoracic Surgery (EACTS): The Task Force for the diagnosis and management of atrial fibrillation of the European Society of Cardiology (ESC) Developed with the special contribution of the European Heart Rhythm Association (EHRA) of the ESC. European Heart Journal, 42(5):373–498.

C. W. Imrie, I. S. Benjamin, J. C. Ferguson, A. J. McKay, I. Mackenzie, J. O’Neill, and L. H. Blumgart. 1978. A single-centre double-blind trial of Trasylol therapy in primary acute pancreatitis. The British Journal of Surgery, 65(5):337–341.

Hawon Jeong, ChaeHun Park, Jimin Hong, Hojoon Lee, and Jaegul Choo. 2025. The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators. In Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networksfor NLP, pages 79–108, Suzhou, China. Association for Computational Linguistics.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences, 11(14):6421.

Nigel S. Key, Alok A. Khorana, Nicole M. Kuderer, Kari Bohlke, Agnes Y. Y. Lee, Juan I. Arcelus, Sandra L. Wong, Edward P. Balaban, Christopher R. Flowers, Leigh E. Gates, Ajay K. Kakkar, Margaret A. Tempero, Shilpi Gupta, Gary H. Lyman, and Anna Falanga. 2023. Venous Thromboembolism Prophylaxis and Treatment in Patients With Cancer: ASCO Guideline Update. Journal of Clinical Oncology: Official Journal ofthe American Society ofClinical Oncology, 41(16):3063–3071.

Alok A. Khorana, Nicole M. Kuderer, Eva Culakova, Gary H. Lyman, and Charles W. Francis. 2008. Development and validation of a predictive model for chemotherapy-associated thrombosis. Blood, 111(10):4902–4907.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. Advances in Neural Information Processing Systems, 38.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2025. LLMs Get Lost In Multi-Turn Conversation. https://arxiv.org/abs/2505.06120v1.

Belinda Z. Li, Been Kim, and Zi Wang. 2025. Quest-Bench: Can LLMs ask the right question to acquire information in reasoning tasks? Preprint, arXiv:2503.22674.

Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S. Ilgen, Emma Pierson, Pang Wei Koh, and Yulia Tsvetkov. 2024. MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning. Preprint, arXiv:2406.00922.

W. S. Lim, M. M. van der Eerden, R. Laing, W. G. Boersma, N. Karalus, G. I. Town, S. A. Lewis, and J. T. Macfarlane. 2003. Defining community acquired pneumonia severity on presentation to hospital: An international derivation and validation study. Thorax, 58(5):377–382.

Gregory Y. H. Lip, Robby Nieuwlaat, Ron Pisters, Deirdre A. Lane, and Harry J. G. M. Crijns. 2010. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach: The euro heart survey on atrial fibrillation. Chest, 137(2):263–272.

Mingxin Liu, Tsuyoshi Okuhara, Zhehao Dai, Minghong Zhao, Wenqiang Yin, Hiroko Okada, Emi Furukawa, and Takahiro Kiuchi. 2026. Textbooklevel medical knowledge in large language models: Comparative evaluation using Japanese National Medical Examination. BMC Medical Informatics and Decision Making, 26(1):65.

Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, and Qi Liu. 2025. ClarifyMT-Bench: Benchmarking and Improving Multi-Turn Clarification for Conversational Large Language Models. Preprint, arXiv:2512.21120.

Sravanthi Machcha, Sushrita Yerra, Sahil Gupta, Aishwarya Sahoo, Sharmin Sultana, Hong Yu, and Zonghai Yao. 2026. Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6153–6182, Rabat, Morocco. Association for Computational Linguistics.

Yosi Mass, Doron Cohen, Asaf Yehudai, and David Konopnicki. 2022. Conversational Search with Mixed-Initiative - Asking Good Clarification Questions backed-up by Passage Retrieval. In Proceedings ofthe Second DialDoc Workshop on Documentgrounded Dialogue and Conversational Question Answering, pages 65–71, Dublin, Ireland. Association for Computational Linguistics.

Casey N. McQuade, Michael G. Simonson, Julia Lister, Andrew P. J. Olson, Laura Zwaan, Scott D. Rothenberger, and Eliana Bonifacino. 2024. Characteristics differentiating problem representation synthesis between novices and experts. Journal of Hospital Medicine, 19(6):468–474.

Joshua P. Metlay, Grant W. Waterer, Ann C. Long, Antonio Anzueto, Jan Brozek, Kristina Crothers, Laura A. Cooley, Nathan C. Dean, Michael J. Fine, Scott A. Flanders, Marie R. Griffin, Mark L. Metersky, Daniel M. Musher, Marcos I. Restrepo, and Cynthia G. Whitney. 2019. Diagnosis and Treatment of Adults with Community-acquired Pneumonia. An Official Clinical Practice Guideline of the American Thoracic Society and Infectious Diseases Society of America. American Journal ofRespiratory and Critical Care Medicine, 200(7):e45–e67.

Atharv Naphade. 2026. Rational Synthesizers or Heuristic Followers? Analyzing LLMs in RAG-based Question-Answering. In Findings of the Association for Computational Linguistics: ACL 2026, pages 40293–40311, San Diego, California, United States. Association for Computational Linguistics.

National Institute for Health and Care Excellence. 2012. Overview | Acute upper gastrointestinal bleeding in over 16s: Management | Guidance | NICE. https://www.nice.org.uk/guidance/cg141.

Emily C. O’Brien, DaJuanicia N. Simon, Laine E. Thomas, Elaine M. Hylek, Bernard J. Gersh, Jack E. Ansell, Peter R. Kowey, Kenneth W. Mahaffey, Paul Chang, Gregg C. Fonarow, Michael J. Pencina, Jonathan P. Piccini, and Eric D. Peterson. 2015. The ORBIT bleeding score: A simple bedside score to assess bleeding risk in atrial fibrillation. European Heart Journal, 36(46):3258–3264.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. MedMCQA: A Largescale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. In Proceedings of the Conference on Health, Inference, and Learning, pages 248–260. PMLR.

Ron Pisters, Deirdre A. Lane, Robby Nieuwlaat, Cees B. de Vos, Harry J. G. M. Crijns, and Gregory Y. H. Lip. 2010. A novel user-friendly score (HAS-BLED) to assess 1-year risk of major bleeding in patients with atrial fibrillation: The Euro Heart Survey. Chest, 138(5):1093–1100.

R. N. Pugh, I. M. Murray-Lyon, J. L. Dawson, M. C. Pietroni, and R. Williams. 1973. Transection of the oesophagus for bleeding oesophageal varices. The British Journal of Surgery, 60(8):646–649.

Jingyi Ren, Ante Wang, Yunghwei Lai, Xiaolong Wang, Linlu Gong, Weitao Li, Weizhi Ma, and Yang Liu. 2026. Beyond "I Don’t Know": Evaluating LLM Self-Awareness in Discriminating Data and Model Uncertainty. Preprint, arXiv:2604.17293.

Samuel Schmidgall, Rojin Ziaei, Carl Harris, Ji Woong Kim, Eduardo Pontes Reis, Jeffrey Jopling, and Michael Moor. 2026. AgentClinic: A multimodal benchmark for tool-using clinical AI agents. npj Digital Medicine, 9(1):499.

Mrinank Sharma, Meg Tong, Tomek Korbak, David Duvenaud, Amanda Askell, Sam Bowman, Esin Durmus, Zac Hatfield-Dodds, Scott Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. 2024. Towards Understanding Sycophancy in Language Models. International Conference on Learning Representations, 2024:110–144.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Proceedings ofthe 40th International Conference on Machine Learning, pages 31210–31227. PMLR.

Stanford T. Shulman, Alan L. Bisno, Herbert W. Clegg, Michael A. Gerber, Edward L. Kaplan, Grace Lee, Judith M. Martin, Chris Van Beneden, and Infectious

Diseases Society of America. 2012. Clinical practice guideline for the diagnosis and management of group A streptococcal pharyngitis: 2012 update by the Infectious Diseases Society of America. Clinical Infectious Diseases: An Official Publication of the Infectious Diseases Society ofAmerica, 55(10):e86– 102.

Bonpei Takase, Takanori Ikeda, Wataru Shimizu, Haruhiko Abe, Takeshi Aiba, Masaomi Chinushi, Shinji Koba, Kengo Kusano, Shinichi Niwano, Naohiko Takahashi, Seiji Takatsuki, Kaoru Tanno, Eiichi Watanabe, Koichiro Yoshioka, Mari Amino, Tadashi Fujino, Yu-Ki Iwasaki, Ritsuko Kohno, Toshio Kinoshita, and 11 others. 2024. JCS/JHRS 2022 Guideline on Diagnosis and Risk Assessment of Arrhythmia. Circulation Journal: Official Journal of the Japanese Circulation Society, 88(9):1509–1595.

Scott Tenner, John Baillie, John DeWitt, Santhi Swaroop Vege, and American College of Gastroenterology. 2013. American College of Gastroenterology guideline: Management of acute pancreatitis. The American Journal ofGastroenterology, 108(9):1400– 1415; 1416.

Patrick Tobin-Schnittger, Jane O’Doherty, Ray O’Connor, and Andrew O’Regan. 2018. Improving quality of referral letters from primary to secondary care: A literature review and discussion paper. Primary Health Care Research & Development, 19(3):211–222.

Mickael Tordjman, Zelong Liu, Murat Yuce, Valentin Fauveau, Yunhao Mei, Jerome Hadjadj, Ian Bolger, Haidara Almansour, Carolyn Horst, Ashwin Singh Parihar, Amine Geahchan, Anis Meribout, Nader Yatim, Nicole Ng, Phillip Robson, Alexander Zhou, Sara Lewis, Mingqian Huang, Timothy Deyer, and 4 others. 2025. Comparative benchmarking of the DeepSeek large language model on medical tasks and clinical reasoning. Nature Medicine, 31(8):2550– 2555.

Samuel J. Vincent, Daniel Calloway, Fangyi Yu, Andrew M. Bean, and Nabeel Seedat. 2026. InsufficiencyBench: Evaluating LLM legal advice on underspecified user queries. Preprint, arXiv:2608.20220.

Yusuke Watanabe, Yohei Kobashi, Takeshi Kojima, Yusuke Iwasawa, Yasushi Okuno, and Yutaka Matsuo. 2026. ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 6: Industry Track), pages 681– 703, San Diego, California, USA. Association for Computational Linguistics.

B. U. Wu, R. S. Johannes, X. Sun, Y. Tabak, D. L. Conwell, and P. A. Banks. 2008. The early prediction of mortality in acute pancreatitis: A large populationbased study. Gut, 57(12):1698–1703.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2024. Adaptive Chameleon or Stubborn Sloth:

Revealing the Behavior of Large Language Models in Knowledge Conflicts. International Conference on Learning Representations, 2024:35623–35646.

Chenyang Yang, Yike Shi, Qianou Ma, Michael Xieyang Liu, Christian Kaestner, and Tongshuang Wu. 2026. What Prompts Don’t Say: Understanding and Managing Underspecification in LLM Prompts. In Findings of the Association for Computational Linguistics: ACL 2026, pages 9072–9101, San Diego, California, United States. Association for Computational Linguistics.

Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. 2024. CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10746–10766, Bangkok, Thailand. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36:46595–46623.

## A Additional Results

Table A.1 reports the error rate for every model and reasoning level under all nine conditions, including the intermediate levels of GPT-5.5 and Opus 4.5 and the 2× and 4× conditions, and Figure A.1 plots each run against its median number of reasoning tokens. Within each model the error rate fell roughly monotonically as the reasoning level, and with it the number of reasoning tokens, increased. The only cells in which the highest level did not lower the error rate are the conditions in which 2× or 4× irrelevant text is added to the undeterminable case, for the Qwen models, where the avoidance of the side carrying irrelevant text happens to favor the correct answer without reasoning and reasoning removes it. Across models, however, more reasoning tokens did not mean fewer errors: at their highest levels the three Qwen models consumed more reasoning tokens than GPT-5.5 and Opus 4.5, yet reduced their baseline error rate less (from 0.342– 0.375 to 0.100–0.200, against 0.342 to 0.017 for GPT-5.5 and 0.425 to 0.125 for Opus 4.5). The gain from reasoning is therefore bounded by the model rather than by the amount of reasoning tokens.

## B Clinical Scoring Systems

Table A.2 lists the 15 scoring systems with the rule used to generate the ground-truth labels and the decision target embedded in the prompt. For the two systems whose rule is an upper bound (Bishop and Centor), the definition of Section 3 applies with the inequalities reversed.

## C Dataset Details

Table A.3 lists the scored-item chunks for the $\mathrm { C H A D S _ { 2 } }$ Score, from which the case presentations are built (Section 3), and Table A.4 the irrelevant text for the same system; one fixed irrelevant text was written per scoring system and amount $( 1 5 \times 3 = 4 5 \mathrm { t e x t s } )$ , the three amounts are nested, and the same text is added whichever side receives it. Table A.5 gives the composition of the 120 baseline pairs by scoring system, and Table A.6 reports the measured amount of text added in each condition and the resulting prompt length. Figures A.3 and A.4 (Appendix D) show one pair built from these chunks under the baseline condition and the four conditions in which text is added to the undeterminable side.

## D Prompt Template and Examples

Figure A.2 shows the template, identical for every instance and every model; it names the scoring system but does not list its items or threshold, and no judgment procedure is supplied (Section 4). Figures A.3 and A.4 show the prompts actually sent for one pair of the $\mathrm { C H A D S _ { 2 } }$ Score under the baseline condition and the four conditions in which text is added to the undeterminable side; the conditions that add text to the determinable case add the same text to Case A instead, and text added relative to the baseline is underlined.

Table A.1: Error rate for every model, reasoning level, and condition. Det/Und denote the determinable/undeterminable case of each pair to which the text is added. Matched, 2×, and 4× are the amounts of irrelevant text (Table A.6). The rows reported as reasoning on in the main text are max for GPT-5.5, high for Opus 4.5, and on for Qwen; none denotes no reasoning (thinking off for Qwen).
<table><tr><td></td><td></td><td></td><td colspan="2">Add relevant factor</td><td colspan="6">Add irrelevant text</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">matched</td><td colspan="2">2×</td><td colspan="2">4×</td></tr><tr><td>Model</td><td>Reasoning</td><td>Baseline</td><td>Det</td><td>Und</td><td>Det</td><td>Und</td><td>Det</td><td>Und</td><td>Det</td><td>Und</td></tr><tr><td>GPT-5.5</td><td>none</td><td>0.342</td><td>0.233</td><td>0.525</td><td>0.475</td><td>0.267</td><td>0.542</td><td>0.258</td><td>0.500</td><td>0.267</td></tr><tr><td></td><td>low</td><td>0.067</td><td>0.017</td><td>0.100</td><td>0.067</td><td>0.042</td><td>0.058</td><td>0.058</td><td>0.058</td><td>0.083</td></tr><tr><td></td><td>high</td><td>0.033</td><td>0.033</td><td>0.058</td><td>0.033</td><td>0.025</td><td>0.025</td><td>0.050</td><td>0.033</td><td>0.050</td></tr><tr><td></td><td>max</td><td>0.017</td><td>0.017</td><td>0.050</td><td>0.025</td><td>0.025</td><td>0.025</td><td>0.025</td><td>0.025</td><td>0.025</td></tr><tr><td>Opus 4.5</td><td>none</td><td>0.425</td><td>0.358</td><td>0.525</td><td>0.558</td><td>0.392</td><td>0.575</td><td>0.383</td><td>0.658</td><td>0.317</td></tr><tr><td></td><td>low</td><td>0.225</td><td>0.083</td><td>0.417</td><td>0.283</td><td>0.217</td><td>0.325</td><td>0.217</td><td>0.358</td><td>0.183</td></tr><tr><td></td><td>medium</td><td>0.108</td><td>0.092</td><td>0.267</td><td>0.200</td><td>0.133</td><td>0.200</td><td>0.100</td><td>0.208</td><td>0.108</td></tr><tr><td></td><td>high</td><td>0.125</td><td>0.042</td><td>0.233</td><td>0.142</td><td>0.133</td><td>0.167</td><td>0.100</td><td>0.175</td><td>0.092</td></tr><tr><td>Qwen3.5-27B</td><td>none</td><td>0.342</td><td>0.225</td><td>0.492</td><td>0.542</td><td>0.183</td><td>0.750</td><td>0.133</td><td>0.783</td><td>0.100</td></tr><tr><td></td><td>on</td><td>0.100</td><td>0.050</td><td>0.242</td><td>0.208</td><td>0.108</td><td>0.217</td><td>0.133</td><td>0.208</td><td>0.200</td></tr><tr><td>Qwen3.5-9B</td><td>none</td><td>0.375</td><td>0.350</td><td>0.433</td><td>0.617</td><td>0.275</td><td>0.733</td><td>0.183</td><td>0.800</td><td>0.133</td></tr><tr><td></td><td>on</td><td>0.167</td><td>0.100</td><td>0.233</td><td>0.192</td><td>0.217</td><td>0.242</td><td>0.217</td><td>0.233</td><td>0.233</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>none</td><td>0.375</td><td>0.367</td><td>0.508</td><td>0.500</td><td>0.308</td><td>0.558</td><td>0.225</td><td>0.767</td><td>0.200</td></tr><tr><td></td><td>on</td><td>0.200</td><td>0.108</td><td>0.292</td><td>0.233</td><td>0.192</td><td>0.300</td><td>0.217</td><td>0.292</td><td>0.217</td></tr></table>

![](images/add8eeba9aee5544372cf7de341a6d1448a4fd3626a1aa6e14ff94fd8862bef0.jpg)

![](images/e2ebe65cafe45cdfa8b67af98eb915958b87b9912d25323f2a065f5a487ae009.jpg)

![](images/87c33e88ba9027e84b7c16e8ab9f1e583d1220ba73f01fc3e12fc34778685390.jpg)

![](images/a14c1ebcc38158b9e1653a1f02aaee66b05a5f756b5d4148f99f53c8bb094c56.jpg)

![](images/1e5d9f7e946d02d721f9e0ba2c14cde836eae4d7339f87c12c3c194dd50fbf8f.jpg)  
Figure A.1: Error rate versus the number of reasoning tokens per run, by model and condition. GPT-5.5 and Opus 4.5 have four reasoning levels and the Qwen models two (thinking off/on). For Opus 4.5, reasoning tokens are estimated as output tokens minus the visible answer because the API does not report them separately, so its none level is not at zero.

Table A.2: The 15 clinical scoring systems used. Items is the number of scored items and Range the range of possible totals; Rule is the threshold and direction used to generate the ground-truth labels; the decision target is the wording embedded in the prompt. Ref. gives the original publication of each system together with a current clinical guideline in which it is used.
<table><tr><td>Scoring system</td><td>Items</td><td>Range</td><td>Rule</td><td>Decision target (as used in the Ref. prompt)</td><td></td></tr><tr><td>Alvarado Score</td><td>8</td><td>0-10</td><td>≥7</td><td>acute appendicitis is suggested</td><td>Alvarado, 1986; Di Saverio et al., 2020</td></tr><tr><td>Apgar Score</td><td>5</td><td>0-10</td><td>≥7</td><td>the newborn&#x27;s condition is nor- mal</td><td>Apgar, 1953; Committee on Fetus and Newborn,</td></tr><tr><td>BISAP Score</td><td>5</td><td>0-5</td><td>≥3</td><td>the patient is at higher risk of Wu et al., 2008; Tenner mortality</td><td>2015 et al., 2013</td></tr><tr><td>Bishop Score</td><td>5</td><td>0-13</td><td>≤ 7</td><td>the cervix is unfavorable for in- Bishop, 1964; American duction</td><td>College of Obstetricians</td></tr><tr><td>Blatchford Score</td><td>8</td><td>0-23</td><td>≥1</td><td>the patient is at high risk and needs intervention</td><td>and Gynecologists, 2009 Blatchford et al., 2000; Na- tional Institute for Health</td></tr><tr><td>CHA2DS2-VASc Score</td><td>7</td><td>0-9</td><td>≥2</td><td>the patient is at high stroke risk and needs anticoagulation</td><td>and Care Excellence, 2012 Lip et al., 2010; Hindricks et al., 2021</td></tr><tr><td>CHADS2 Score</td><td>5</td><td>0-6</td><td>≥2</td><td>the patient is at high stroke risk and needs anticoagulation</td><td>Gage et al., 2001; Takase et al., 2024</td></tr><tr><td>CURB-65 Score</td><td>5</td><td>0-5</td><td>≥3</td><td>the pneumonia is severe and war-] rants hospitalization</td><td>Lim et al., 2003; Metlay et al., 2019</td></tr><tr><td>Centor Score</td><td>5</td><td>0-5</td><td>≤2</td><td>no further testing or antibiotics are needed</td><td>Centor et al., 1981; Shul- man et al., 2012</td></tr><tr><td>Child-Pugh Classification</td><td>5</td><td>0-15</td><td>≥ 10</td><td>the cirrhosis is class C</td><td>Pugh et al., 1973; European Association for the Study</td></tr><tr><td>Glasgow-Imrie Score</td><td>8</td><td>0-8</td><td>≥3</td><td>this is severe acute pancreatitis</td><td>of the Liver, 2018 Imrie et al., 1978; Blamey et al., 1984; Tenner et al.,</td></tr><tr><td>HAS-BLED Score</td><td>9</td><td>0-9</td><td>≥3</td><td>the patient is at high bleeding</td><td>2013 Pisters et al., 2010; Hin-</td></tr><tr><td>Khorana Score</td><td>5</td><td>0-6</td><td>≥3</td><td>risk the patient is at high risk of ve- ]</td><td>dricks et al., 2021 Khorana et al., 2008; Key</td></tr><tr><td>ORBIT Bleeding Score</td><td>5</td><td>0-7</td><td>≥4</td><td>nous thromboembolism the patient is at high risk of ma-</td><td>et al., 2023 O’Brien et al., 2015; Hin-</td></tr><tr><td>SIRS Criteria</td><td>4</td><td>0-4</td><td>≥2</td><td>jor bleeding the patient meets the criteria for SIRS</td><td>dricks et al., 2021 Bone et al., 1992; Tenner et al., 2013</td></tr></table>

Table A.3: Examples of pre-written scored-item chunks for CHADS . Positive chunks indicate that the scoring criterion is met, while negative chunks indicate that it is not. Missing item values are represented by omitting the corresponding chunks.
<table><tr><td>Item</td><td>Positive chunk</td><td>Negative chunk</td></tr><tr><td>Congestive</td><td>a history of</td><td>no history of</td></tr><tr><td>heart failure</td><td>congestive heart failure</td><td>congestive heart failure</td></tr><tr><td>Hypertension</td><td>a history of hypertension</td><td>no history of hypertension</td></tr><tr><td>Age ≥ 75</td><td>An 80-year-old</td><td>A 68-year-old</td></tr><tr><td>Diabetes</td><td>a history of diabetes</td><td>no history of</td></tr><tr><td>mellitus</td><td>mellitus</td><td>diabetes mellitus</td></tr><tr><td>Prior stroke or a prior stroke TIA</td><td></td><td>no prior stroke or TIA</td></tr></table>

Table A.4: Examples of pre-written irrelevant text for CHADS<sub>2</sub>. These sentences do not correspond to any scoring item. Each condition includes the text from the preceding condition plus the sentences shown in its row. Sentence and token counts are cumulative.
<table><tr><td>Amount</td><td>Sentences added</td></tr><tr><td>matched 1 sentence 7 tokens</td><td>Family history included maternal osteoarthritis.</td></tr><tr><td>2× 2 sentences 33 tokens</td><td>+ The patient reported a severe allergy to macrolide antibiotics, which historically caused significant gastrointestinal upset and a diffuse skin rash during childhood.</td></tr><tr><td>4× 5 sentences 98 tokens</td><td>+ According to the medical record, the patient took oral levothyroxine each morning to manage a long-standing history of uncomplicated primary hypothyroidism. Current regular medications included a daily oral zinc tablet and a nonprescription antihistamine taken during the spring months for seasonal allergic rhinitis. The</td></tr></table>

Table A.5: Composition of the 120 baseline pairs by scoring system: eight pairs per system, four with the correct answer A and four with B. Missing = number of missing scored items per case, median [range]. Det margin = distance of the determinable side from the threshold (0 = exactly at it); Und margin = points the undeterminable side would need to move to become determinable. Tokens are mean (SD) per case.
<table><tr><td>Scoring system</td><td>Missing</td><td>Det margin</td><td>Und margin</td><td>Det tokens</td><td>Und tokens</td></tr><tr><td>Alvarado Score</td><td>4 [2–5]</td><td>0 [0-1]</td><td>1 [1–3]</td><td>48.2 (8.6)</td><td>51.6 (10.4)</td></tr><tr><td>Apgar Score</td><td>3 [2-3]</td><td>0 [0–0]</td><td>2 [1–3]</td><td>23.5 (5.0)</td><td>27.8 (5.3)</td></tr><tr><td>BÍSAP Score</td><td>2 [2-2]</td><td>0 [0-0]</td><td>1 [1–1]</td><td>32.8 (3.4)</td><td>32.9 (2.4)</td></tr><tr><td>Bishop Score</td><td>2 [2–2]</td><td>0 [0-1]</td><td>1 [1–3]</td><td>40.2 (2.9)</td><td>39.9 (2.3)</td></tr><tr><td>Blatchford Score</td><td>4 [2-5]</td><td>3 [0-9]</td><td>1 [1–1]</td><td>44.6 (8.0)</td><td>40.6 (8.3)</td></tr><tr><td>CHA₂DS2-VASc Score</td><td>2 [2-5]</td><td>1 [0-4]</td><td>1 [1–2]</td><td>41.6 (8.2)</td><td>45.4 (9.9)</td></tr><tr><td>CHADS2 Score</td><td>3 [2-4]</td><td>0 [0−1]</td><td>1 [1–2]</td><td>32.2 (4.2)</td><td>32.9 (3.8)</td></tr><tr><td>CURB-65 Score</td><td>2 [2–2]</td><td>0 [0–0]</td><td>1 [1–1]</td><td>52.5 (3.4)</td><td>50.1 (1.9)</td></tr><tr><td>Centor Score</td><td>2 [2–3]</td><td>0 [0-1]</td><td>1 [1–2]</td><td>26.9 (3.8)</td><td>26.8 (5.1)</td></tr><tr><td>Child-Pugh Classification</td><td>2 [2–2]</td><td>0 [0-1]</td><td>2 [1–2]</td><td>44.2 (6.4)</td><td>42.5 (5.3)</td></tr><tr><td>Glasgow-Imrie Score</td><td>3 [2-4]</td><td>0 [0-2]</td><td>1 [1–2]</td><td>66.9 (9.7)</td><td>64.6 (8.0)</td></tr><tr><td>HAS-BLED Score</td><td>4 [3-6]</td><td>0 [0-1]</td><td>1 [1–2]</td><td>40.6 (8.5)</td><td>40.8 (7.6)</td></tr><tr><td>Khorana Score</td><td>2 [2-3]</td><td>0 [0-1]</td><td>1 [1–2]</td><td>44.8 (5.4)</td><td>47.2 (6.8)</td></tr><tr><td>ORBIT Bleeding Score</td><td>2 [2–3]</td><td>0 [0-1]</td><td>2 [1–3]</td><td>31.9 (4.8)</td><td>30.4 (3.7)</td></tr><tr><td>SIRS Criteria</td><td>2 [2–2]</td><td>0 [0-0]</td><td>1 [1–1]</td><td>39.4 (3.1)</td><td>37.9 (1.6)</td></tr><tr><td>All (120 pairs)</td><td>2 [2–6]</td><td>0 [0–9]</td><td>1 [1–3]</td><td>40.7 (11.9)</td><td>40.8 (11.4)</td></tr></table>

Table A.6: Amount of text added per case in each condition. Rel/Irr: addition of a relevant factor/irrelevant text; Det/Und: the determinable/undeterminable case receiving it (Irr rows apply to either side). Added tokens count only the inserted text; prompt tokens are the mean length of the complete prompt actually sent to the model, i.e., the instruction template plus the case description and any added text. Both columns report means. Token counts use tiktoken (o200k\_base); baseline case descriptions average 41 tokens.
<table><tr><td>Condition</td><td>Added tokens</td><td>Prompt tokens</td></tr><tr><td>Baseline</td><td></td><td>140</td></tr><tr><td>Rel to Det</td><td>8</td><td>147</td></tr><tr><td>Rel to Und</td><td>8</td><td>148</td></tr><tr><td>Irr (matched)</td><td>8</td><td>148</td></tr><tr><td>Irr (2×)</td><td>43</td><td>182</td></tr><tr><td>Irr (4×)</td><td>127</td><td>267</td></tr></table>

Prompt template   
Two cases are presented below.   
Case A:   
{case\_A}   
Case B:   
{case\_B}   
Based on the {score\_name}, which case has sufficient information to determine whether   
{decision\_target}?   
Answer with a single letter, "A" or "B". Output only that letter and nothing else.  
Figure A.2: The prompt template, identical for every instance and every model, sent as a single user message with no system prompt.

Baseline tokens: Case A 39, Case B 41, prompt 142   
Two cases are presented below.   
Case A:   
An 80-year-old patient with atrial fibrillation was evaluated for stroke risk to guide   
anticoagulation. The patient had a history of congestive heart failure and no history of diabetes   
mellitus.   
Case B:   
A patient with atrial fibrillation was evaluated for stroke risk to guide anticoagulation. The   
patient had no history of congestive heart failure, a history of hypertension, and no history of   
diabetes mellitus.   
Based on the CHADS2 Score, which case has sufficient information to determine whether the patient   
is at high stroke risk and needs anticoagulation?   
Answer with a single letter, "A" or "B". Output only that letter and nothing else.

Rel to Und tokens: Case A 39, Case B 45, prompt 146   
Two cases are presented below.   
Case A:   
An 80-year-old patient with atrial fibrillation was evaluated for stroke risk to guide   
anticoagulation. The patient had a history of congestive heart failure and no history of diabetes   
mellitus.   
Case B:   
A 68-year-old patient with atrial fibrillation was evaluated for stroke risk to guide   
anticoagulation. The patient had no history of congestive heart failure, a history of hypertension,   
and no history of diabetes mellitus.   
Based on the CHADS2 Score, which case has sufficient information to determine whether the patient   
is at high stroke risk and needs anticoagulation?   
Answer with a single letter, "A" or "B". Output only that letter and nothing else.

Irr to Und (matched) tokens: Case A 39, Case B 48, prompt 149   
Two cases are presented below.   
Case A:   
An 80-year-old patient with atrial fibrillation was evaluated for stroke risk to guide   
anticoagulation. The patient had a history of congestive heart failure and no history of diabetes   
mellitus.   
Case B:   
A patient with atrial fibrillation was evaluated for stroke risk to guide anticoagulation. The   
patient had no history of congestive heart failure, a history of hypertension, and no history of   
diabetes mellitus. Family history included maternal osteoarthritis.   
Based on the CHADS2 Score, which case has sufficient information to determine whether the patient   
is at high stroke risk and needs anticoagulation?   
Answer with a single letter, "A" or "B". Output only that letter and nothing else.  
Figure A.3: The prompts actually sent for one pair (CHADS Score) under Baseline, Rel to Und, and Irr to Und (matched); Figure A.4 continues with 2× and 4×. Case A, the determinable side and correct answer, is identical throughout; the underlined text is added to Case B: a missing scored item (age) in Rel to Und, and one, two, or five irrelevant sentences (Table A.4) in Irr to Und. Rel to Det and Irr to Det add the text to Case A instead. Title bars give the token counts (tiktoken o200k\_base).

![](images/e5c7183fe41c7679b398f436ad7794af68428fc5cb5e3692cabc0807f40c8020.jpg)  
Figure A.4: The prompts for the same pair under Irr to Und with the 2× and 4× amounts of irrelevant text (continued from Figure A.3). The irrelevant text is nested: each amount extends the preceding one (Table A.4). Text added relative to the baseline is underlined.