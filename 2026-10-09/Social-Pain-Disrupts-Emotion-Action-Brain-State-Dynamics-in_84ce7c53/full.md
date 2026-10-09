# Social Pain Disrupts Emotion-Action Brain-State Dynamics in Adolescents with Non-Suicidal Self-Injury

Ying Xu<sup>1,2</sup>, Xiaojun Liang<sup>3</sup>, Li Zhang<sup>1,2</sup>, Yixuan Yuan<sup>4</sup>, Gan Huang<sup>1,2</sup>, Yongjie Zhou<sup>5\*</sup>, and Zhen Liang<sup>1,2\*</sup>

<sup>1</sup>School of Biomedical Engineering, Medical School, Shenzhen University, Shenzhen, China. <sup>2</sup>Guangdong Provincial Key Laboratory of Biomedical Measurements and Ultrasound Imaging, Shenzhen, China.

<sup>3</sup>Pengcheng Laboratory, Shenzhen, China.

<sup>4</sup>Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong, China. <sup>5</sup>Department of Child Health, Shenzhen Maternity and Child Healthcare Hospital, Shenzhen, China. <sup>\*</sup>Address correspondence to: qingzhu1108@126.com and janezliang@szu.edu.cn

## Abstract

Non-suicidal self-injury (NSSI) is highly prevalent among adolescents with depression, yet the rapid neurodynamic processes through which socially salient distress is translated into maladaptive behavioral tendencies remain poorly understood. Here, we integrate an experimental pain paradigm, electroencephalography (EEG) microstate analysis, and interpretable deep sequence modeling to characterize millisecond-scale brain-state dynamics associated with NSSI. EEG was recorded from 106 adolescents with depression, including 67 with NSSI (DN+) and 39 without NSSI (DN−), during social pain, physical pain, and resting-state conditions. A neurodynamic modeling framework is developed to capture higher-order dependencies within microstate sequences through joint disease-specific, domain-adversarial, consistency, and contrastive learning. Among the three conditions, social pain elicits the most discriminative NSSIrelated neural dynamics. The proposed framework achieves a classification accuracy of 68.55%, outperforming the best- performing baseline model by 8.94 percentage points. Importantly, model interpretation converges with conventionalmicrostate analyses to reveal disrupted bidirectional transitions between MS3 and MS5 in DN+ adolescents duringsocial pain. Source reconstruction links MS3 to emotional/interoceptive processing and MS5 to action preparation, indicating impaired emotion-action coupling. Time-resolved analyses further reveal enhanced recruitment of action-preparation states during early-to-middle stages of social pain processing, followed by increased engagement of emotion-related states at later stages in DN+ adolescents. Moreover, MS5-to-MS3 dynamics mediate social-evaluation sensitivity and afective outcomes in DN− but not DN+, whereas MS3-to-MS5 transitions are associated with greater negative afect in DN+. Together, these findings identify disrupted emotion–action coupling as a key neurodynamic mechanism underlying altered social pain processing in adolescents with NSSI, providing a mechanistically interpretable neural signature for objective identification of NSSI.

Keywords: Non-suicidal self-injury; Social pain; EEG microstates; Emotion-action coupling; Neurodynamic modeling

## Introduction

Non-suicidal self-injury (NSSI) is a major mental health concern among adolescents, particularly those with depression. Large-scale epidemiological studies have estimated a lifetime prevalence of approximately 17.7% in the general adolescent population, with substantially higher rates among adolescents with depression, more than half of whom may report a history of self-injurious behaviors[1, 2]. Beyond its high prevalence, NSSI is associated with greater clinical severity and represents an important prospective risk factor for subsequent suicidal behavior[3, 4]. However, the neurobiological processes through which distressing ex periences are dynamically transformed into maladaptive behavioral tendencies remain poorly understood. Clinical identification of NSSI also continues to rely largely on self-report and clinical interviews, underscoring the need for objective neurophysiological measures that can complement conventional assessment[5]. Characterizing the neural dynamics associated with NSSI may therefore provide important mechanistic insights into this clinically heterogeneous behavior and facilitate its objective identification in adolescents with depression.

Pain processing provides a particularly relevant framework for understanding the neurobiology of NSSI. The Interpersonal Psychological Theory of Suicide (IPTS) emphasizes the roles of adverse interpersonal experiences and altered pain-related processes in the development of self injurious and suicidal behaviors[6, 7]. Consistent with this framework, individuals with NSSI frequently exhibit atypical physical pain processing, including elevated pain thresholds and reduced subjective pain sensitivity[8]. Neuroimaging studies have further identified altered responses in pain-related regions, particularly the anterior cingulate cortex and insula, which contribute to the sensory, afective, and interoceptive dimensions of pain[9–11]. Importantly, NSSI is also closely associated with adverse interpersonal experiences, making social pain particularly relevant to its neurobiology. Social rejection and physical pain engage partially overlapping neural systems[12], suggesting that altered responses to social distress in NSSI may involve disrupted coordination among emotional, interoceptive, and action-related processes. Yet most previous studies have focused on regional activation or static neural measures. How the brain dynamically transitions between these functional processes as social distress unfolds, and whether such temporal organization is disrupted in adolescents with NSSI remains largely unknown.

Electroencephalography (EEG) microstates provide a powerful approach for addressing this question. EEG microstates are transient, quasi-stable scalp potential topographies that persist for tens to hundreds of milliseconds and are thought to represent rapidly changing functional configurations of large-scale brain networks[13]. Unlike conventional time–frequency analyses that primarily characterize local spectral properties, microstate analysis captures the temporal organization of whole-brain electrophysiological activity, providing a systems-level representation of rapidly evolving brain states[14]. EEG microstates have been increasingly investigated in psychiatric and neurological disorders and provide a potential bridge between electrophysiological dynamics and largescale functional brain networks[15–18]. Altered microstate properties have also been reported in depression. Murphy et al. [18] identified abnormalities in microstate temporal properties in patients with major depressive disorder (MDD), whereas Zhao et al. [19] reported altered duration, occurrence, and coverage of multiple microstate classes in drug-naive adolescents with first-episode depression. Neuroimaging studies further suggest that canonical microstates are associated with intrinsic functional networks involved in salience processing, attention, sensory processing, and internally oriented cognition[20]. Moreover, abnormal microstate dynamics in MDD have been associated with cognitive dysfunction[21]. Together, these findings suggest that microstate dynamics may provide a temporally resolved window into the large-scale neural alterations associated with psychopathology.

However, conventional microstate studies typically summarize brain dynamics using aggregate measures such as duration, occurrence, coverage, and pairwise transition probabilities. Although informative, these measures compress the temporal organization of microstate sequences and may therefore miss higher-order dependencies among rapidly evolving brain states. This limitation is particularly relevant to NSSI, in which pathological responses to socially salient distress may emerge not from abnormalities in a single brain state, but from disrupted transitions and coordination between states supporting emotional processing and behavioral preparation. Sequence-based machine learning provides a means to model these higher-order temporal dependencies directly[22]. More broadly, EEGbased machine-learning approaches have demonstrated the feasibility of identifying depression-related neural patterns from spectral, connectivity, and end-to-end representations[23, 24]. Liang et al. [25], for example, developed a semi-supervised multi-concept framework for EEGbased identification of NSSI, whereas microstate-based approaches have achieved promising classification performance in MDD using conventional temporal parameters and sequence complexity[19, 26]. Nevertheless, these approaches have predominantly treated microstate properties as aggregated features rather than modeling the microstate sequence itself. Consequently, the higher-order brain-state transitions that may distinguish adolescents with and without NSSI, particularly during socially salient distress, remain insuficiently characterized.

Here, we integrate an experimental pain paradigm, EEG microstate analysis, and interpretable deep sequence modeling to investigate the neurodynamic mechanisms associated with NSSI in adolescents with depression (Fig. 1). We examine brain-state dynamics across social pain, physical pain, and resting-state conditions to determine which context most strongly exposes NSSI-related neural alterations. Rather than using microstates solely as static summary features, we directly model their sequential organization to capture higher-order transition dependencies and identify discriminative neurodynamic patterns. We further combine model interpretation with conventional microstate analysis, source-level reconstruction, functional decoding, time-resolved analysis, and mediation analysis to characterize the functional and behavioral relevance of the identified transitions. We hypothesize that social pain preferentially reveals disrupted transitions between brain states supporting emotional/interoceptive processing and action preparation in adolescents with NSSI. By linking millisecond-scale brain-state dynamics to individual-level identification, this framework aims to provide a mechanistically interpretable account of altered emotion–action coupling in NSSI.

## Results

To characterize the neurodynamic alterations associated with NSSI, we analyze EEG microstate sequences acquired from adolescents with depression during social pain, physical pain, and resting-state conditions. We first examine whether diferent pain-related contexts diferentially exposed NSSI-related neural dynamics and then identify the microstate transitions contributing to these diferences. Source-level reconstruction and functional decoding are subsequently used to characterize the neurobiological substrates of the identified transitions, while time-resolved and mediation analyses examine their temporal and behavioral relevance. Finally, we evaluate whether these neurodynamic patterns support individual-level identification of NSSI and assess the robustness and generalizability of the proposed modeling framework.

![](images/877f94e99485a1feb70760ec542eb72f13b5c2adcb022da2b0895347d5a09a99.jpg)  
Figure 1: Overview of the proposed model framework. The model consists of four main components.(1) Mi crostate sequence construction. Preprocessed EEG signals are subjected to a two-step microstate clustering procedure using the k-means algorithm to obtain optimal microstate templates. These templates are then back-fitted to each data segment to generate corresponding microstate time series. (2) MS temporal reorganization. The microstate sequences are segmented into patches using sliding windows (window length = 50, step size = 50). Two shufling strategies (weak permutation and strong permutation) are further applied to enhance the diversity of temporal patterns and improve model robustness. (3) Neural representation learning. A PatchTST architecture is employed to capture dynamic transition patterns within the microstate sequences and map them into high-dimensional representations. A dedicated CLS patch is introduced to facilitate the integration of local segment-level characteristics and global sequence-level structure. (4) Dynamic stability learning. The model is jointly optimized using multiple objectives, including a disease-specific loss, domain adversarial loss, consistency loss, and contrastive learning loss. This joint optimization enables the model to anchor disease-relevant pathological microstate transition patterns associated with NSSI, thereby samplifying subtle diferences between patient subgroups.

## Social pain preferentially reveals NSSIrelated neurodynamic alterations

We first examine whether NSSI-related neurodynamic differences varied across social pain, physical pain, and resting-state conditions. Under identical evaluation settings, decoding performance shows a consistent conditiondependent gradient, with social pain yielding the highest performance (Accuracy = 68.55%, Precision = 69.49%, Recall = 67.37%, F1-score = 68.08%, AUC = 68.06%), followed by physical pain (Accuracy = 66.45%) and resting state (Accuracy = 61.84%) (Table 1). These results indicate that NSSI-related neural diferences are most discriminative during social pain processing.

This pattern is consistent with the high self-relevance and afective salience of socially adverse experiences, which engage neural systems involved in emotional processing, interoception, and behavioral regulation[9, 12, 27]. By comparison, physical pain more directly engages nociceptive and somatosensory processes[28, 29], whereas the resting state lacks an explicit afective challenge[30, 31]. Thus, rather than indicating a generalized alteration across brain states, the observed performance gradient suggests that socially salient distress preferentially exposes neurodynamic diferences associated with NSSI.

To further examine condition-dependent neuralstimulus correspondence, we use CLIP[32] to extract visual representations of the social and physical pain stimuli and quantify their cosine similarity with modelderived EEG representations. A two-way mixed ANOVA is performed with pain type as the within-subject factor and group as the between-subject factor. As shown in Fig. 2, EEG-stimulus representational similarity difers between social and physical pain conditions in DN+ (p = 0.02, t = 2.38). No significant between-group diferences are detected within either pain condition, indicating that the efect is primarily condition dependent rather than reflecting a global diference in representational similarity between DN+ and DN− adolescents. Together with the decoding results, these findings further support social pain as a particularly informative context for revealing NSSI-related neurodynamic alterations.

## Disrupted MS3-MS5 transitions characterize social pain processing in NSSI

Given that social pain most strongly diferentiated DN+ from DN− adolescents, we next investigate the neurodynamic patterns underlying this efect. Directed temporal dependencies between consecutive microstates are quantified from the attention weights of the trained model and aggregate into microstate transition attention matrices. Group-specific matrices are then examined to identify transitions that contributed most strongly to the discrim ination of NSSI status.

During social pain processing, group diferences are observed primarily for transitions involving MS2↔MS8, MS3↔MS5, and MS3→MS6 (Fig. 3). Among these transitions, MS3↔MS5 shows the most consistent group difference. Importantly, this pattern converges with conventional microstate analyses based on occurrence, duration, coverage, and transition probability, which independently implicate MS3 and MS5 in diferentiating DN+ from DN− adolescents. DN+ adolescents exhibit weaker bidirectional MS3↔MS5 transition strength than DN− adolescents, indicating attenuated dynamic coupling between these two brain states during social pain processing.

We additionally examine whether these transition patterns are strongly influenced by sex. In contrast to the clinical-group efects, sex-related diferences are limited, with a significant diference observed only for the MS6→MS7 transition. Details of the statistical analysis can be found in the supplementary materials Appendix I. These findings suggest that the prominent MS3↔MS5 alteration is more closely associated with NSSI status than with sex-related variation.

## MS3-MS5 dynamics link emotionalinteroceptive processing to action preparation

To characterize the functional significance of the disrupted MS3↔MS5 dynamics, we project microstate topographies into source space using the TESS framework[33] and perform meta-analytic functional decoding using Neurosynth. Source reconstruction reveal partially distinct spatial distributions for MS3 and MS5 (Fig. 4). MS3 is predominantly associated with cingulate and medial frontal regions implicated in emotional and interoceptive processing, whereas MS5 shows stronger involvement of the precentral gyrus and adjacent premotor regions associated with action preparation.

These spatial and functional profiles suggest that MS3↔MS5 transitions index dynamic coordination between emotional/interoceptive processing and actionpreparatory systems. The reduce bidirectional MS3↔MS5 transitions observed in DN+ adolescents therefore point to disrupted emotion-action coupling during the processing of socially salient distress. Rather than reflecting an abnormality confined to a single neural state, NSSI is characterized by altered coordination between brain states involved in evaluating afective information and preparing behavioral responses.

## Time-resolved dynamics reveal an early action-preparation bias followed by late emotional recruitment

To determine how the identified emotion–action alterations unfold over time, we perform a time-resolved analysis of microstate dynamics during social pain processing. EEG responses are segmented using a sliding window of 300 ms with a 50-ms step, and the proportion of each microstate within each window is quantified. Microstates with identical spatial topographies but opposite polarities are treated as the same state. Individual-level estimates are subsequently averaged to generate time-resolved stateoccurrence curves for DN+ and DN− adolescents (Fig. 5).

Distinct temporal diferences emerge between groups. During the early-to-middle stages of social pain processing (0–400 ms, 950–1450 ms, and 2250–2600 ms), DN+ adolescents show greater recruitment of the state associated with motor and action preparation than DN− adolescents. In contrast, during a later interval (4400–4850 ms), DN+ adolescents show greater recruitment of the state associated with emotional processing.

Table 1: Model performance across social pain, physical pain, and resting-state conditions
<table><tr><td>State</td><td> $\mathrm { A c c u r a c y } ( \% )$ </td><td>Precision(%)</td><td> ${ \mathrm { R e c a l l } } ( \% )$ </td><td> $F 1 \mathrm { - s c o r e } ( \% )$ </td><td> $\mathrm { A U C } ( \% )$ </td></tr><tr><td>Resting</td><td> $6 1 . 8 4 { \pm } 0 1 . 3 9$ </td><td> $6 2 . 2 6 { \pm } 0 3 . 1 0$ </td><td> $6 2 . 1 1 { \pm } 1 0 . 0 0$ </td><td> $6 1 . 6 0 { \pm } 0 4 . 1 3$ </td><td> $6 0 . 5 0 { \pm } 0 2 . 6 4$ </td></tr><tr><td>Physical Pain</td><td> $6 6 . 4 5 { \pm } 0 4 . 3 1$ </td><td> $6 7 . 0 7 { \pm } 0 5 . 4 9$ </td><td> $6 6 . 5 8 { \pm } 1 1 . 3 1$ </td><td> $6 6 . 1 2 { \pm } 0 5 . 9 8$ </td><td> $6 6 . 0 0 { \pm } 0 5 . 7 3$ </td></tr><tr><td>Social Pain</td><td> $\mathbf { 6 8 . 5 5 { \scriptstyle \pm 0 3 . 0 7 } }$ </td><td> $\mathbf { 6 9 . 4 9 { \scriptstyle \pm 0 4 . 6 0 } }$ </td><td> ${ \bf 6 7 . 3 7 { \scriptstyle \pm 0 6 . 7 0 } }$ </td><td> ${ \bf 6 8 . 0 8 { \scriptstyle \pm 0 3 . 4 1 } }$ </td><td> ${ \bf 6 8 . 0 6 { \pm } 0 4 . 5 7 }$ </td></tr></table>

![](images/619ffeaa0c13c98572ce76675b7e2ac9590293cf90431ef421e0b113f731cd35.jpg)  
(b)

DN+: Social Pain vs Physical Pain  
(c)  
![](images/8f553ff3811a4203f15fb6a24062f62f03d78c5c66b5ae4fabf8fcc346fb91ad.jpg)

DN-: Social Pain vs Physical Pain  
![](images/168be262e9ec1da7b6f5eabfd09f89bbf330416135c1fe4b0d8ec3399163b0a6.jpg)

(e)  
![](images/3cd0afd8ecbeb0235622e5471b43315b28acf059d844ebc52b06a3dcc4d61b18.jpg)

Social Pain: DN+ vs DN-  
(f)  
![](images/7a5e47abd819982d296ef78b41859783e7d554b5445bc18cda629d7720809141.jpg)

Physical Pain: DN+ vs DN-  
![](images/85fa5ac274a6b11cfbbbd7450a50f714a849ad7cee40d2eb89e91b8e318bd868.jpg)  
Figure 2: Cosine similarity between model-derived EEG representations and stimulus representations across pain conditions. (a) Group-level comparison of cosine similarity between model-derived EEG features and stimulus image features under social pain (SP) and physical pain (PP) conditions. (b-c) Within-group comparisons of SP versus PP for DN+ and DN− adolescents, respectively. (d) Between-group comparison of cosine similarity across conditions. (e-f) Post hoc comparisons between DN+ and DN− groups under $\mathrm { S P }$ and PP conditions, respectively. $^ { * } \mathrm { p } { < } 0 . 0 5$

## MS5-to-MS3 dynamics diferentially link social evaluation to afective outcomes

These results reveal a temporally structured alteration in social pain processing in NSSI, characterized by enhanced engagement of action-preparatory systems during early-to-middle processing followed by increased recruitment of emotion-related systems at later stages. This temporal organization is consistent with the abnormal MS3↔MS5 transitions identified above and provides complementary evidence that altered emotion-action coupling in NSSI unfolds dynamically over the course of social pain processing[34–36].

We next examine whether the identified emotion-action dynamics are associated with individual diferences in sensitivity to the social environment. Given the established relevance of peer interaction and social evaluation to adolescent NSSI[37, 38], mediation analyses are performed separately in DN− and DN+ adolescents. Environmental variables include parent-child relationship factors, academic stress, and sensitivity to teachers’ and peers’ evaluations, while clinical and afective outcomes include PHQ-9[39], GAD-7[40], and subjective negative afect following

![](images/b6a74360c4530ad8007290bf742dccc58992e20f27f743aafc4c450e7e7807c6.jpg)

![](images/32c62813598e3b5428c9d3ffb60faf89e3ab4c8d74716831ef457daf97e883bd.jpg)

![](images/70905aee8fb91d53db6bca579e3ebc99e23b5b7010b657d3af82e870b9ff48c6.jpg)

![](images/5d222471c07014adf970c022ee2adbe119bde05d68452772090293b27c54efed.jpg)

![](images/498c5565e141d19b6d834375817f3ed57ffeb9b4b4293bae583037510e2df8d1.jpg)

![](images/0d84d66c6506b78288d79fe38df8e2b61fc2952d4daa35e87c2ded01e9a59ac8.jpg)  
Figure 3: Microstate transition attention patterns across clinical and sex subgroups. (a-b) Mean microstate transition attention matrices for DN+ and DN− adolescents, averaged across ten-fold cross-validation. (c) Statistical comparison of transition attention weights between clinical groups. (d-e) Mean transition attention matrices for male and female subgroups. (f) Statistical comparison between sex subgroups. Asterisks indicate transitions showing significant group diferences. \*p<0.05.

social pain.

In DN− adolescents, sensitivity to social evaluation is positively associated with MS5→MS3 transition strength (a = 0.451). Greater MS5→MS3 transition strength is, in turn, negatively associated with PHQ-9 scores (b = −0.480) and subjective emotional ratings $( b = - 0 . 4 9 3 )$ . Bootstrap mediation analyses showed significant indirect associations between social-evaluation sensitivity and PHQ-9 scores (indirect efect = −0.206), as well as between social-evaluation sensitivity and subjective emotional ratings (indirect efect = −0.232). The corresponding direct efects remained significant, consistent with partial statistical mediation.

This pattern is absent in DN+ adolescents. Although sensitivity to social evaluation show direct associations with PHQ-9 scores and subjective emotional ratings, its association with MS5→MS3 transition strength is not significant $( a ~ = ~ 0 . 0 8 2 )$ nor are the associations between MS5→MS3 dynamics and PHQ-9 (b = −0.090) or subjective emotional ratings (b = −0.140). Consequently, no significant indirect efects through MS5→MS3 are observed.

Notably, the reverse MS3→MS5 transition shows a significant association with subjective negative afect in DN+ adolescents. Together, these findings reveal a direction dependent alteration in emotion-action brain-state dynamics: whereas action-to-emotion transitions are associated with social-evaluation sensitivity and afective outcomes in adolescents without NSSI, this relationship is disrupted in adolescents with NSSI, in whom stronger emotion-to-action dynamics are instead associated with greater negative afect.

## Neurodynamic modeling enables individual -level identification of NSSI

Having identified disrupted emotion–action dynamics during social pain, we next evaluate whether the sequential organization of microstates provides suficient information for individual-level identification of NSSI. The proposed model is compared with four conventional machinelearning methods[41–44], four neural sequence models[45– 48], five domain-adaptation approaches[49–52], and two contrastive-learning methods[53, 54] using identical inputs (Table 2).

Conventional machine-learning methods show relatively limited discrimination, with accuracies ranging from 46.45% to 53.55%. Neural sequence models yield modest improvements, with the Transformer achieving the highest accuracy among the baseline methods (59.61%). Domainadaptation and contrastive-learning approaches produce comparable performance, with DDC reaching a precision

![](images/4628437e15851c055b26da0c45da676e681e19b917408f4cc24cdaf91e28fe5a.jpg)

![](images/6f8c0c22746b110c5f6992e556f668761f9b5999893513e53d2ccd48a4e91f6f.jpg)

## Motor Preparation

![](images/b7b949c5a48296efcaa642898b63aef9dd47b7296daf106900909052bc30743e.jpg)  
Precentral Gyrus (Frontal Lobe)

![](images/12cc8caccc4057b4b06f628fc6e37e82cb7ff31988dce18a10f24cccab1855b0.jpg)

![](images/47c5bed02b4e2f3fc39befc72461018d8e4783b76eb33f901c689ec35b1871b0.jpg)

![](images/3723206b287853c37f8f72197bd65e3d80e377af555193d27362aa50edd0e90b.jpg)

![](images/3487b956d0b395db109309b046386d87cd7893f6fa0e9bd6816ce2e90d24834f.jpg)  
Precentral Gyrus (Frontal Lobe)

![](images/a6114f8f2fb726bc354fe69304b84d8fed0d3ac6512b41f57ca78d786966086c.jpg)

![](images/ef8548dd96896e3fdc91f178b7d8691f19cf5f5d79d71c601a7e333208bc6967.jpg)

![](images/4de76f8e2d6483f16082ce58630ceab2ac7820a63b46e80bad771456b515aabe.jpg)

## Sensory Processing

![](images/4f615fc3d23a00935b0a0317275b4b0aa1323e1c82a2b0d98ee657785249339f.jpg)

![](images/8f6c23b35f2faffab98f2b36839fcffb369c4c1479531ed260f50f69318a4e6e.jpg)  
Postcentral Gyrus (Parietal Lobe)

![](images/634b0323482ebfdaa1eddcba82f16f454aabd187f41f793621f8af4e7c31602c.jpg)

![](images/fb749f0b75149da8e1ebfeb05b5c8733c798aee47653f15d1da25b51c91c85af.jpg)

![](images/5c45dcc147d38b1ac769da1da002979a30dc2b70892fae5b074de0f863193f78.jpg)  
Postcentral Gyrus (Parietal Lobe)

![](images/01f3923085ccd9aa586033de6d6853fc6746c360ca6e83f9a06b56d5ffe0b3d2.jpg)

![](images/3a613da48068c2b8d9d03b5a7fc9d73b1ebf6c1b0cc63ada8883d385ff3a321f.jpg)

![](images/d3e2ef2a1160c19c4793b99edf20513585cdc4c4fee745bf66059b7a1ab79a35.jpg)

## Emotional Regulation

![](images/6b07f185a09e3522b38b87b8a464f05b2e03e0f5096349a3de143dc05c9a8b91.jpg)  
Cingulate Gyrus (Limbic Lobe)

![](images/d736370ece31faf89f6ac47e28cfe95bfdb35c2bd8c72c68b53b53dfb495d98e.jpg)

![](images/07b5922c8070980f1710aaf6e6b17438a51e6e918193b628a920f4a27592ef3e.jpg)

![](images/bbba28e097105c561e889a1f610430506daa704c2edc49cf026220db73a66fd8.jpg)  
voice force executed visuomotor

![](images/f72c29b9144307262ba2e248484eed2c753e0354d1ca520b3348e3bc292edb80.jpg)  
Cingulate Gyrus (Limbic Lobe)

![](images/4b3bf47921824fbf067536ef1684a8b01cb25ecf4abb5adbbef0202574176fed.jpg)

![](images/d985afb825923a699a0b34de5fac42f408092041588624b8fd3df79860202a58.jpg)

![](images/ae71bd0efbdd3187ed1b37c787aeb1862ec9b3879224e6a9505eff439979c9c9.jpg)

![](images/3175aac2370dc607cecfe1ac3b117500f0b554cc0bf4cd7d6ac3101095ede2c2.jpg)  
Anterior Cingulate (Limbic Lobe)

## Conflict monitoring

![](images/41086b288060ea3871cfd8664a23ca4291de06efaac2c5e617ed0aef3ed59d3f.jpg)

![](images/93b23b6a87e0e895500ccef5363922c30f61ca51426e9b06df10eda3c4abf7bb.jpg)

![](images/63e09ad55ffde558555f8f24170981d336ed81aeca06c10ce4e129c7d0935b40.jpg)  
Anterior Cingulate (Limbic Lobe)

![](images/fae442da5e9fb6ad1667e117e468f1cf8188edd1efcb33de13c7adfd24ad1d35.jpg)  
Figure 4: Source-level reconstruction and functional annotation of EEG microstates. Source-level activation patterns of the eight microstate templates are shown together with their meta-analytic functional annotations derived from Neurosynth. For each template, word clouds display the top 15 positively and negatively associated functional terms based on correlation coeficients, excluding anatomical terms. The cortical region showing the strongest source activity for each template is indicated below the corresponding map.

of 65.17% and SimCLR an accuracy of 59.47%. In comparison, the proposed neurodynamic model achieves an accuracy of 68.55%, precision of 69.49%, recall of 67.37%, F1- score of 68.08%, and AUC of 68.06%. Its accuracy exceeds that of the best-performing baseline model by 8.94 percentage points, indicating that explicitly modeling higherorder dependencies within microstate sequences improves the identification of NSSI-related neurodynamic patterns.

tend previous accounts of NSSI from abnormalities in individual regions or networks toward a dynamic systems perspective, in which vulnerability may arise from how rapidly evolving neural states interact over time.

## Discussion

## Social pain as a context for revealing NSSIrelated vulnerability

The present study suggests that NSSI in adolescents with depression is characterized by altered coordination between emotional/interoceptive and action-related brain states during socially salient distress. These findings ex

Interpersonal experiences are particularly salient during adolescence, when sensitivity to peer acceptance, rejection, and social evaluation increases substantially. This developmental context may help explain why socially painful experiences provide an informative window into NSSIrelated neural dysfunction. The Interpersonal Psychological Theory of Suicide emphasizes adverse interpersonal experiences and altered pain-related processes in selfinjurious and suicidal behaviors[6, 7]. Although NSSI is distinct from suicidal behavior, this framework highlights the potential importance of interactions between interpersonal distress and pain-related processing.

MS1 MS5 Motor Preparation Network  
![](images/f3301747f4b92effad5dca8cc7aa3a9a4f369b774a911f9af9ba2e41273aa595.jpg)

MS2 MS6 Sensory Processing Network  
![](images/d6e4976109cfc0b54da34ab70cf0d1a4353c4a4f6f2f6c6d91f004f3a722e641.jpg)

MS3 MS7 Emotional Regulation Network  
![](images/a9e7fbf231f0cbbb2139bf2cfef314509ae6a240396644b66e486a60d1495f89.jpg)  
MS4 MS8 Conflict monitoring Network

![](images/74da8acd8154d7be8f04964120c87b1fd29442be33d0cd1768cd27c7d8497eea.jpg)  
Figure 5: Time-resolved validation of microstate dynamics during social pain processing. A slidingwindow approach (window length = 300 ms; step size = 50 ms) is used to quantify the temporal occurrence of each microstate. Microstate templates with identical spatial topographies but opposite polarity were treated as the same state. Solid lines represent adolescents with depression and NSSI (DN+), whereas dashed lines represent adolescents with depression without NSSI (DN−). Shaded regions indicate time intervals showing significant between-group diferences. Corresponding source-level activation patterns are displayed for each microstate.

Physical and social pain recruit partially overlapping neural systems involved in afective salience and interoception[9–11]. Social pain, however, additionally carries information about rejection, belonging, and self-relevance, placing greater demands on emotional appraisal and behavioral regulation. Social pain may therefore provide an experimental context that amplifies underlying vulnerabilities in processing socially salient distress, rather than representing a specific causal factor for NSSI.

## Emotion-action decoupling as a neurodynamic mechanism of NSSI

A central implication of the present findings is that NSSI related dysfunction may involve impaired coordination between systems supporting emotional/interoceptive processing and those supporting action preparation. The cingulate cortex occupies an important position at the interface of afect, interoception, cognitive control, and action[27, 55, 56], whereas precentral and premotor systems contribute to the preparation and organization of behavioral responses. Flexible interaction between these systems may therefore be necessary for translating emotional information into context-appropriate behavior.

Previous neuroimaging studies of adolescent NSSI and depression have identified abnormalities across salience, default-mode, attention, and sensorimotor networks[57, 58]. Our findings extend these observations by suggesting that the relevant dysfunction may lie not only within individual systems but also in their temporal coordination. Ineficient switching between emotional and action-related states could impair the flexible transformation of socially salient information into adaptive behavioral responses.

This dynamic perspective may be particularly relevant

![](images/d188cff242ba2a6f5b5636bcbd8427d7ce26458cb3a720925e1898d76f5fa7d9.jpg)

![](images/6dd86ff55aebd96ebaa1094e1188f07bb2ea05ac545400a53d290f0cacdf3603.jpg)  
Figure 6: Associations among social evaluation, microstate dynamics, and afective outcomes. (a-b) Statistical mediation analyses examining MS5→MS3 dynamics in the association between sensitivity to social evaluation and depressive symptoms (PHQ-9 scores) in DN− and DN+ adolescents. (c-d) Corresponding analyses for subjective negative emotional ratings following pain-related stimuli. (e) Association between MS3→MS5 transition probability and subjective negative emotional ratings across groups. Solid red lines indicate statistically significant paths, whereas dashed black lines indicate non-significant paths. Path coeficients are shown alongside each path. DN+, depression with NSSI; DN−, depression without NSSI.

Table 2: Performance comparison with conventional machine-learning, neural sequence, domain-adaptation, and contrastive-learning methods.
<table><tr><td>Method</td><td> $\mathrm { A c c u r a c y } ( \% )$ </td><td> $\mathrm { P r e c i s i o n } ( \% )$ </td><td> ${ \mathrm { R e c a l l } } ( \% )$ </td><td> $F 1 \mathrm { - s c o r e } ( \% )$ </td><td>AUC(%)</td></tr><tr><td>KNN[41]</td><td> $5 3 . 5 5 { \pm } 0 4 . 4 7$ </td><td> $5 3 . 1 5 { \pm } 0 3 . 8 7$ </td><td> $6 1 . 5 8 { \pm } 0 7 . 8 7$ </td><td> $5 6 . 8 6 { \scriptstyle \pm 0 4 . 7 7 }$ </td><td> $4 7 . 4 7 { \pm } 0 5 . 7 8$ </td></tr><tr><td>Decision tree[42]</td><td> $4 7 . 5 0 { \pm } 0 4 . 6 2 $ </td><td> $4 7 . 3 0 { \pm } 0 4 . 6 0$ </td><td> $4 9 . 2 1 { \pm } 1 1 . 8 4$ </td><td> $4 7 . 8 4 { \pm } 0 7 . 6 2 $ </td><td> $4 8 . 3 3 { \pm } 0 4 . 0 0$ </td></tr><tr><td>SVM[43]</td><td> $4 6 . 4 5 { \pm } 0 4 . 9 6$ </td><td> $4 6 . 3 4 { \pm } 0 5 . 2 5 $ </td><td> $4 6 . 8 4 { \pm } 0 8 . 0 2 $ </td><td> $4 6 . 4 8 { \pm } 0 6 . 2 3$ </td><td> $5 5 . 3 7 { \pm } 0 6 . 1 7$ </td></tr><tr><td>Random forest[44]</td><td> $5 2 . 2 4 { \pm } 0 6 . 3 0 $ </td><td> $5 2 . 2 3 { \scriptstyle \pm 0 6 . 0 7 }$ </td><td> $5 2 . 8 9 { \pm } 0 8 . 0 8 $ </td><td> $5 2 . 4 4 { \pm } 0 6 . 6 3$ </td><td> $5 2 . 4 7 { \pm } 0 6 . 9 1 $ </td></tr><tr><td>RNN[45]</td><td> $5 8 . 0 3 { \pm } 0 5 . 8 0 $ </td><td> $6 4 . 1 1 \pm 1 0 . 9 2 $ </td><td> $4 9 . 7 4 { \pm } 2 6 . 8 2$ </td><td> $5 0 . 7 2 { \pm } 1 5 . 1 3$ </td><td> $5 6 . 1 1 { \pm } 0 9 . 7 0$ </td></tr><tr><td>LSTM[46]</td><td> $5 6 . 4 5 { \pm } 0 4 . 0 9$ </td><td> $6 5 . 0 8 { \pm } 1 0 . 7 0 $ </td><td> $3 8 . 9 5 { \pm } 2 4 . 4 6 $ </td><td> $4 4 . 2 7 { \pm } 1 1 . 7 9$ </td><td> $5 3 . 3 6 { \pm } 0 5 . 4 0$ </td></tr><tr><td>GRU[47]</td><td> $5 6 . 1 8 { \pm } 0 3 . 4 6$ </td><td> $6 4 . 9 6 { \pm } 1 0 . 2 7$ </td><td> $3 5 . 5 3 { \pm } 2 1 . 3 9$ </td><td> $4 2 . 1 3 { \pm } 1 1 . 8 9$ </td><td> $5 2 . 9 1 { \pm } 0 5 . 0 5$ </td></tr><tr><td>Transformer[48]</td><td> $5 9 . 6 1 { \pm } 0 6 . 0 2 $ </td><td> $6 0 . 1 3 { \pm } 2 5 . 0 2$ </td><td> $5 0 . 7 9 { \pm } 3 1 . 5 1 $ </td><td> $4 9 . 7 9 { \pm } 2 4 . 5 7$ </td><td> $5 4 . 9 2 { \pm } 1 3 . 0 1$ </td></tr><tr><td>DDC[49]</td><td> $5 7 . 2 4 { \pm } 0 3 . 6 3$ </td><td> $6 5 . 1 7 { \pm } 1 0 . 6 6 $ </td><td> $4 7 . 8 9 { \pm } 2 7 . 2 9 $ </td><td> $4 9 . 5 4 { \pm } 1 2 . 2 6 $ </td><td> $5 6 . 4 8 { \pm } 0 4 . 8 2 $ </td></tr><tr><td>DAN[50]</td><td> $5 6 . 1 8 { \pm } 0 4 . 0 7$ </td><td> $5 5 . 4 2 { \pm } 2 1 . 7 8 $ </td><td> $4 4 . 2 1 { \pm } 3 1 . 1 8$ </td><td> $4 4 . 1 6 { \pm } 2 2 . 2 0 $ </td><td> $5 3 . 4 2 { \pm } 0 6 . 7 5 $ </td></tr><tr><td>DANN[5i]</td><td> $5 5 . 1 3 { \pm } 0 4 . 5 8 $ </td><td> $5 8 . 5 0 { \pm } 0 9 . 6 8 $ </td><td> $5 9 . 7 4 \pm 2 5 . 4 9$ </td><td> $5 5 . 0 1 { \pm } 0 9 . 3 0 $ </td><td> $5 5 . 7 8 { \pm } 0 6 . 6 4$ </td></tr><tr><td>CORAL[52]</td><td> $4 8 . 6 8 { \pm } 0 5 . 3 4$ </td><td> $4 8 . 6 6 { \pm } 0 5 . 0 2$ </td><td> $4 6 . 8 4 { \pm } 0 9 . 3 5 $ </td><td> $4 7 . 3 9 { \pm } 0 6 . 8 4$ </td><td> $4 8 . 2 0 { \pm } 0 3 . 3 9$ </td></tr><tr><td> $\mathrm { \ D C O R A L [ 5 2 ] }$ </td><td> $5 6 . 0 5 { \pm } 0 3 . 7 8$ </td><td> $6 0 . 0 3 { \pm } 2 5 . 6 4$ </td><td> $3 6 . 3 2 { \pm } 2 4 . 2 4$ </td><td> $4 0 . 7 8 { \pm } 1 8 . 8 5$ </td><td> $5 2 . 7 0 { \scriptstyle \pm 0 7 . 9 3 }$ </td></tr><tr><td>MoCo[53]</td><td> $5 5 . 5 3 { \pm } 0 5 . 2 6 $ </td><td> $6 1 . 1 5 { \pm } 0 9 . 7 7$ </td><td> $3 5 . 7 9 { \pm } 0 8 . 6 1$ </td><td> $4 4 . 2 4 { \pm } 0 5 . 9 1$ </td><td> $5 3 . 9 7 { \scriptstyle \pm 0 7 . 9 7 }$ </td></tr><tr><td> $\mathrm { S i m C L R } [ 5 4 ]$ </td><td> $5 9 . 4 7 { \pm } 0 3 . 5 0 $ </td><td> $6 4 . 3 1 { \pm } 1 3 . 0 0 $ </td><td> $5 1 . 8 4 { \pm } 1 4 . 9 9$ </td><td> $5 4 . 9 4 { \pm } 0 9 . 2 4 $ </td><td> $5 4 . 7 6 { \pm } 0 6 . 2 6$ </td></tr><tr><td>Our model</td><td> $\mathbf { 6 8 . 5 5 { \scriptstyle \pm 0 3 . 0 7 } }$ </td><td> $\mathbf { 6 9 . 4 9 { \scriptstyle \pm 0 4 . 6 0 } }$ </td><td> ${ \bf 6 7 . 3 7 { \scriptstyle \pm 0 6 . 7 0 } }$ </td><td> ${ \bf 6 8 . 0 8 { \scriptstyle \pm 0 3 . 4 1 } }$ </td><td> ${ \bf 6 8 . 0 6 { \pm } 0 4 . 5 7 }$ </td></tr></table>

## Directionality may distinguish emotional integration from behavioral readiness

to NSSI, which has frequently been conceptualized as a maladaptive strategy for regulating intense negative affect[59]. The observed temporal organization further suggests that action-related processing and emotional processing are abnormally coordinated as social distress unfolds. Rather than indicating a direct neural pathway to selfinjury, these findings point to altered emotion–action integration as a broader mechanism through which distress may become more closely linked to behavioral readiness.

The directional organization of brain-state transitions suggests that emotion-action coupling is not a unitary process. Transitions from action-related toward emotional/interoceptive states may reflect incorporation of externally oriented information into internal afective appraisal, whereas transitions in the opposite direction may reflect transformation of emotional states toward behavioral preparation.

Such reciprocal coordination is important for adaptive emotion regulation, which requires continuous updating of internal states according to environmental demands and flexible selection of behavioral responses[60, 61]. Disruption of this exchange could therefore reduce the flexibility with which socially salient experiences are processed.

Within this framework, the distinct behavioral associations of the two transition directions suggest an altered balance between emotional integration and action readiness in NSSI. Importantly, this interpretation does not imply that an individual transition directly represents selfinjurious behavior. Rather, it suggests that negative affect may become more closely coupled to action-oriented neural configurations, providing a potential neurodynamic bridge between emotional dysregulation and maladaptive behavioral tendencies.

## Multidimensional analyses support model robustness

Beyond its mechanistic interpretation, the proposed framework showed stable performance across complementary validation analyses. The ablation analysis (Supplementary Appendix II) indicates that the disease-specific, domain-adversarial, consistency, and contrastive objectives make complementary contributions to the learned representation. In particular, the performance reductions observed after removing individual objectives suggest that the model benefits from jointly preserving disease-relevant information, reducing inter-individual domain variation, and maintaining discriminative temporal structure.

Additional supplementary analyses further support the reliability of the proposed framework. The loss curves of the individual optimization objectives exhibit stable convergence throughout training, indicating that the multiobjective optimization strategy does not introduce conflicting optimization dynamics (Supplementary Appendix III). Permutation testing confirms that the observed classification performance remains significantly above chance level $( p = 0 . 0 3 )$ , while repeated experiments under diferent random sampling strategies demonstrate the robustness of the framework to sampling variability (Supplementary Appendix IV). Furthermore, comparisons of diferent PatchTST classification heads consistently favor the proposed design, suggesting that the performance improvement arises from the overall neurofunctional framework rather than architectural choices alone (Supplementary Appendix V).

Hyperparameter analyses further demonstrate the stability of the proposed framework. Across diferent loss weighting schemes and permutation ratios, the model maintains consistently competitive performance, with the optimal configuration achieving the best balance between representation learning and classification accuracy (Supplementary Appendix VI). These findings indicate that the framework is not overly sensitive to hyperparameter selection, supporting its robustness and potential for future clinical translation.

## Sex-related stability of neurodynamic representations

Given the sex imbalance commonly observed in adolescent NSSI cohorts, we further examine whether the learned representations are strongly dependent on sex. As summarized in Table 3 and supplementary Appendix VII, within-sex and cross-sex analyses show broadly comparable discrimination, suggesting that the major neurodynamic information captured by the framework is not restricted to one sex. Under gender-consistent settings, performance decreases slightly but remains stable, with Accuracy/F1-score of 67.63%/66.87% in the female subgroup and 65.79%/65.00% in the male subgroup. In crossgender evaluations, performance declines further, with relatively balanced results in the Female-to-Male condition (F1-score: 65.30%), while the Male-to-Female condition shows high Precision (72.51%) but markedly reduced Recall (53.68%), leading to a lower F1-score (61.37%). Overall, the model maintains good discriminative ability across single-gender and cross-gender settings, efectively distinguishing DN+ and DN− patients.

## Independent datasets distinguish general izability from disease specificity

To further validate the decoding capability of the proposed model, we conduct additional evaluations on two independent public EEG datasets of MDD. First, we perform binary classification under both eyes-open(EO) and eyesclosed(EC) conditions using the dataset from Hospital Universiti Sains Malaysia (HUSM) (https://figshare. com/articles/EEGDataNew/4244171) [62]. Following the same preprocessing pipeline, participants with poor signal quality are excluded, resulting in a final sample of 31 MDD patients and 21 healthy controls. The model achieve superior performance under the EC condition (Accuracy = 82.17%±9.67%, F1-score = 82.82%±10.38%), compared to the EO condition (Accuracy = 75.98%±9.73%, F1-score = 76.49%±10.92%). We further evaluate the model on the MODMA dataset using the same preprocessing pipiline, yielding 24 MDD patients and 29 healthy controls (https://modma.lzu.edu.cn/data/index)[63]. Consistently, the model demonstrated robust classification performance $( \mathrm { A c c u r a c y } = 6 8 . 5 9 \% \pm 7 . 1 8 \%$ , F1-score $= 7 0 . 5 0 \% \pm 1 0 . 9 8 \% )$ . As summarized in Table 4, the proposed model outperform existing machine learning and deep learning approaches, with Transformer-based models ranking second. Overall, these cross-dataset validation results provide strong evidence for the robustness and generalizability of the proposed model across diferent datasets and experimental conditions.

We further examine the microstate transition patterns captured by the model across independent MDD datasets. No significant group diferences are observed under the EO condition, whereas the EC condition reveals altered bidirectional MS1↔MS3 transitions[64–67]. These transitions have been associated with switching between internally oriented self-referential and externally oriented perceptual processing, suggesting disrupted coordination between these functional modes in MDD. Similar MS1↔MS3 alterations are observed in the MODMA dataset, providing convergent evidence that the framework captures clinically relevant disturbances in brain-state organization across independent cohorts.

Table 3: Sex-specific and cross-sex validation of the proposed framework.
<table><tr><td>Train set</td><td>Test set</td><td>Accuracy(%)</td><td>Precision(%)</td><td>Recall(%)</td><td> $F 1 \mathrm { - s c o r e } ( \% )$ </td></tr><tr><td>Entire data</td><td>Entire data</td><td> $6 8 . 5 5 { \pm } 0 3 . 0 7$ </td><td> $6 9 . 4 9 { \pm } 0 4 . 6 0$ </td><td> $6 7 . 3 7 { \pm } 0 6 . 7 0 $ </td><td> $6 8 . 0 8 { \scriptstyle \pm 0 3 . 4 1 }$ </td></tr><tr><td>Female subgroup</td><td>Female subgroup</td><td> $6 7 . 6 3 { \pm } 0 6 . 6 2$ </td><td> $6 7 . 7 6 { \pm } 0 4 . 9 0$ </td><td> $6 7 . 8 9 { \pm } 1 7 . 5 9$ </td><td> $6 6 . 8 7 { \pm } 0 9 . 8 2$ </td></tr><tr><td>Female subgroup</td><td>Male subgroup</td><td> $6 5 . 7 9 { \pm } 0 2 . 7 9$ </td><td> $6 6 . 4 8 { \pm } 0 4 . 3 6$ </td><td> $6 5 . 2 6 { \pm } 1 0 . 4 3$ </td><td> $6 5 . 3 0 { \pm } 0 4 . 7 4$ </td></tr><tr><td>Male subgroup</td><td>Male subgroup</td><td> $6 5 . 7 9 { \pm } 0 2 . 4 6$ </td><td> $6 7 . 3 5 { \pm } 0 6 . 0 0$ </td><td> $6 5 . 7 9 { \pm } 1 6 . 9 5 $ </td><td> $6 5 . 0 0 { \scriptstyle \pm 0 6 . 4 1 }$ </td></tr><tr><td>Male subgroup</td><td>Female subgroup</td><td> $6 6 . 3 2 { \pm } 0 2 . 0 0$ </td><td> $7 2 . 5 1 { \pm } 0 5 . 6 4$ </td><td> $5 3 . 6 8 { \pm } 0 5 . 4 6$ </td><td> $6 1 . 3 7 { \pm } 0 2 . 2 3$ </td></tr></table>

## Materials and Methods

## Participants

This study recruited a total of 106 adolescents with depression (age: 13–18 years; 77 females; years of education $\geq 5 )$ as participants. The experiment is conducted at the Neuromodulation and Psychiatric Rehabilitation Laboratory of Shenzhen Kangning Hospital, with ethical approval from the hospital’s Research Ethics Committee (Approval No.: 2020-K021-04-1). All participants were diagnosed with MDD by psychiatrists based on DSM-5 criteria, and other psychiatric disorders were excluded through screening with the Mini International Neuropsychiatric Interview (MINI). Exclusion criteria included severe physical illness, neurological disorders, and high suicide risk. Informed consent was obtained from all participants and their families.

This study divide the depressed adolescents into two groups: depression without NSSI (DN−) and depression with NSSI (DN+). The DN− group includes 39 participants (age: 15.8±2.3 years; 20 females), and the DN+ group includes 67 participants (age: 14.5±3.0 years; 57 females). After EEG preprocessing and exclusion of poorquality signals, the final sample comprises 34 participants in the DN− group (age: 16.0±1.39 years; 15 females) and 67 participants in the DN+ group (age: 14.52±1.75 years; 57 females).

## Data Acquisition and Preprocessing

In this experiment, we use two types of emotion-inducing stimuli to efectively elicit negative emotions in adolescents: images of physical pain and images of social pain, with 35 images selected for each category. The physical pain images depict real injuries or skin damage, such as knife wounds, cuts, lacerations, and abrasions. The social pain images present scenarios of rejection, where an individual is visibly excluded by a group, illustrating social rejection, unfriendliness, or indiference—for example, being left out of activities, ignored, or disregarded by others.

We conduct the entire procedure using E-Prime 2.0 software. As shown in Fig 8, the paradigm consists of five minutes of resting-state EEG recording followed by two task conditions: physical pain and social pain, each presented randomly across 35 trials. Each trial is divided into four segments: resting state, image stimulus, emotional assessment, and a black screen. Participants first focus on a central cross for 2 s to relax, followed by a 5 s image stimulus presenting either a social exclusion or physical pain image. During the subsequent 5 s emotional assessment, participants rate the intensity of their negative emotion on a scale from 1 (no negative emotion) to 9 (strongest negative emotion) using the keyboard and confirm the rating by pressing a key. Finally, a 1 s black screen is displayed to allow relaxation between trials.

EEG signals are recorded using a 64-channel $\mathrm { A g / A g C l }$ electrode cap (10–20 system) with a BrainAmp amplifier and BrainVision Recorder software at a sampling rate of 500 Hz. Ofline preprocessing was performed in Letswave[68], an open-source toolbox running in MAT-LAB. Preprocessing steps include segmenting task related epochs from continuous recordings, removing unnecessary IO channels, and applying a common average reference to reduce random noise. Data are filtered with a 1–45 Hz bandpass and a 50 Hz notch filter to attenuate ocular and muscle artifacts. Independent component analysis (ICA) is used to further remove ocular artifacts. EEG segments corresponding to the image stimulation phase (5 s) are extracted, and bad channels are interpolated to ensure data quality.

## Microstate-Based Representation of Global Brain Dynamics

For the preprocessed EEG data, we perform microstate analysis using the open-source toolbox MST 1.0 implemented in EEGLAB to extract microstate time sequences. As illustrated in Fig 1, we first compute the Global Field Power (GFP) at each time point, defined as the standard deviation of potentials across all electrodes. The EEG topographic maps corresponding to the GFP peaks, representing moments of maximal field stability and highest signal-to-noise ratio, are then selected for subsequent microstate clustering.

$$
G F P ( t ) = \sqrt { \frac { 1 } { N } \sum ( V _ { i } ( t ) - \bar { V } ( t ) ) ^ { 2 } }\tag{1}
$$

Where $V _ { i } ( t )$ denotes the potential of the i − th electrode at time t; V<sup>¯</sup> (t) represents the mean potential across all electrodes; and N is the total number of electrodes.

Table 4: External validation on independent depression datasets.
<table><tr><td>Data</td><td>Model</td><td>Accuracy(%)</td><td>Precision(%)</td><td>Recall(%)</td><td>F1-score(%)</td><td>AUC(%)</td></tr><tr><td rowspan="18">HUSM (EO)</td><td>KNN Decision tree</td><td>53.80±10.48 53.48±11.06</td><td>62.90±28.52 55.20±13.84</td><td>27.17±14.28 54.57±12.49</td><td>36.05±15.48</td><td>56.54±18.95</td></tr><tr><td></td><td></td><td></td><td></td><td>54.03±09.47</td><td>53.94±11.48</td></tr><tr><td>SVM</td><td>49.13±04.31</td><td>49.02±05.73</td><td>40.43±07.19</td><td>44.07±06.03</td><td>48.31±04.43</td></tr><tr><td>Random forest</td><td>61.96±19.70</td><td>61.12±21.72</td><td>76.96±23.44</td><td>66.75±19.43</td><td>60.07±24.39</td></tr><tr><td>RNN</td><td>63.37±06.91</td><td>58.58±22.33</td><td>64.13±28.22</td><td>59.69±21.86</td><td>60.78±07.92</td></tr><tr><td>LSTM</td><td>62.61±06.57</td><td>62.02±08.99</td><td>74.57±12.76</td><td>66.50±04.52</td><td>60.56±11.43</td></tr><tr><td>GRU</td><td>71.96±09.83</td><td>69.45±12.70</td><td>86.09±15.92</td><td>75.29±09.55</td><td>69.57±17.17</td></tr><tr><td>Transformer</td><td>74.57±16.61</td><td>72.65±16.78</td><td>80.65±21.92</td><td>75.55±17.19</td><td>78.01±15.73</td></tr><tr><td>DDC</td><td>59.24±07.72</td><td>60.84±12.64</td><td>65.00±23.71</td><td>59.76±12.84</td><td>56.17±09.42</td></tr><tr><td>DAN</td><td>56.52±06.66</td><td>50.27±28.60</td><td>43.70±32.78</td><td>43.10±25.90</td><td>56.20±10.42</td></tr><tr><td>DANN</td><td>59.24±06.84</td><td>60.17±09.49</td><td>68.04±20.29</td><td>61.49±08.52</td><td>57.52±10.92</td></tr><tr><td>CORAL</td><td>55.11±21.21</td><td>54.05±16.32</td><td>82.17±22.02</td><td>64.77±17.43</td><td>45.15±26.28</td></tr><tr><td>DCORAL</td><td>59.24±06.97</td><td>59.39±09.44</td><td>68.91±19.28</td><td>62.08±08.17</td><td>56.50±10.50</td></tr><tr><td>MoCo</td><td>59.67±07.68</td><td>59.59±08.50</td><td>66.96±12.95</td><td>62.25±06.53</td><td>57.94±10.15</td></tr><tr><td>SimCLR</td><td>61.52±07.10</td><td>61.87±08.56</td><td>64.78±09.10</td><td>62.71±06.12</td><td>60.10±10.29</td></tr><tr><td>Our model</td><td>75.98±09.73</td><td>75.32±10.85</td><td>80.22±15.59</td><td>76.49±10.92</td><td>78.67±10.79</td></tr><tr><td></td><td>67.17±12.23</td><td>75.36±17.31</td><td>55.00±27.00</td><td>59.92±18.65</td><td>71.81±11.76</td></tr><tr><td rowspan="20">HUSM (EC)</td><td>KNN Decision tree</td><td>58.15±06.36</td><td>58.32±06.65</td><td>56.74±18.03</td><td>56.44±10.59</td><td>60.13±07.54</td></tr><tr><td>SVM</td><td>45.87±08.02</td><td>47.18±06.44</td><td>58.48±17.82</td><td>50.89±09.60</td><td>51.15±10.65</td></tr><tr><td>Random forest</td><td>64.35±09.62</td><td>66.00±12.44</td><td>67.17±21.48</td><td>64.45±10.54</td><td>74.22±15.94</td></tr><tr><td>RNN</td><td>62.61±10.37</td><td>65.39±09.96</td><td>61.96±23.69</td><td>60.48±14.74</td><td>60.14±15.04</td></tr><tr><td>LSTM</td><td>64.02±06.47</td><td>65.40±10.60</td><td>68.91±18.73</td><td>64.92±07.36</td><td>59.88±12.73</td></tr><tr><td>GRU</td><td>72.61±11.40</td><td>78.92±15.41</td><td>68.91±19.12</td><td>71.00±11.70</td><td>70.96±21.74</td></tr><tr><td>Transformer</td><td>78.80±08.68</td><td>78.48±14.34</td><td>87.83±18.98</td><td>80.17±09.27</td><td>76.37±17.19</td></tr><tr><td>DDC</td><td>64.02±07.02</td><td>66.05±08.92</td><td>64.78±18.72</td><td>63.31±08.77</td><td>59.81±10.69</td></tr><tr><td>DAN</td><td>60.11±09.63</td><td>60.52±08.81</td><td>62.17±21.11</td><td>59.68±12.50</td><td>62.13±07.79</td></tr><tr><td>DANN</td><td>62.07±06.09</td><td>60.09±04.82</td><td>76.74±14.05</td><td>66.64±05.14</td><td>56.89±12.28</td></tr><tr><td>CORAL</td><td>56.74±08.31</td><td>61.22±16.28</td><td>50.65±26.53</td><td>51.30±14.50</td><td>66.43±13.98</td></tr><tr><td>DCORAL</td><td>63.91±07.90</td><td>66.05±11.47</td><td>66.52±18.87</td><td>63.66±10.95</td><td>60.39±16.22</td></tr><tr><td>MoCo</td><td>61.63±07.99</td><td>60.14±06.49</td><td>71.96±12.78</td><td>64.98±07.26</td><td>61.54±12.57</td></tr><tr><td>SimCLR</td><td></td><td>67.49±09.07</td><td>66.30±16.91</td><td>65.88±09.92</td><td>66.89±16.92</td></tr><tr><td>Our model</td><td>66.74±07.99 82.17±09.67</td><td>81.73±11.66</td><td>85.22±15.05</td><td>82.82±10.38</td><td>85.80±10.53</td></tr><tr><td></td><td>42.83±08.42</td><td>37.37±12.87</td><td>35.00±27.56</td><td>34.05±18.98</td><td>38.01±08.96</td></tr><tr><td rowspan="20">MODMA (EC)</td><td>KNN Decision tree</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>49.89±07.66</td><td>50.10±07.09</td><td>49.13±08.40</td><td>49.44±07.28</td><td>48.86±08.96</td></tr><tr><td>SVM</td><td>49.89±06.45</td><td>50.20±06.71</td><td>50.87±06.81</td><td>50.37±05.84</td><td>42.92±04.88</td></tr><tr><td>Random forest</td><td>53.04±14.44</td><td>52.01±17.46</td><td>50.65±25.48</td><td>49.56±20.57</td><td>54.19±20.70</td></tr><tr><td>RNN</td><td>56.41±03.88</td><td>56.43±03.92</td><td>64.13±20.88</td><td>58.09±09.37</td><td>54.02±05.18</td></tr><tr><td>LSTM GRU</td><td>54.46±03.30</td><td>55.57±05.02</td><td>56.96±22.75</td><td>53.78±09.36</td><td>51.10±06.42</td></tr><tr><td></td><td>54.56±03.30</td><td>55.57±05.02</td><td>56.96±22.75</td><td>53.78±09.36</td><td>50.65±06.42</td></tr><tr><td>Transformer</td><td>65.11±10.39</td><td>74.74±15.36</td><td>52.61±27.91</td><td>55.93±22.12</td><td>70.09±20.20</td></tr><tr><td>DDC</td><td>51.41±03.08</td><td>42.74±23.22</td><td>66.52±44.67</td><td>48.56±27.53</td><td>50.43±04.96</td></tr><tr><td>DAN</td><td>53.26±03.69</td><td>44.28±24.01</td><td>54.78±38.63</td><td>46.02±25.86</td><td>53.21±05.84</td></tr><tr><td>DANN CORAL</td><td>55.22±04.34</td><td>54.92±04.46</td><td>69.13±23.09</td><td>59.26±08.77</td><td>52.59±03.33</td></tr><tr><td></td><td>52.72±13.96</td><td>52.12±16.14</td><td>51.09±17.82</td><td>51.32±16.64</td><td>41.89±11.81</td></tr><tr><td>DCORAL</td><td>54.23±04.11</td><td>50.17±18.48</td><td>50.00±31.03</td><td>47.77±19.54</td><td>51.00±05.18</td></tr><tr><td>MoCo</td><td>54.02±06.48</td><td>55.31±07.80</td><td>47.61±17.16</td><td>49.42±11.77</td><td>48.70±08.94</td></tr><tr><td>SimCLR</td><td>57.50±03.68</td><td>59.59±04.19</td><td>48.04±12.40</td><td>52.17±09.32</td><td>54.29±06.06</td></tr><tr><td>Our model</td><td>68.59±07.18</td><td>71.33±11.44</td><td>66.52±09.51</td><td>67.90±06.36</td><td>70.50±10.98</td></tr></table>

(i)  
(g)  
![](images/394519db961bc1a91d8f13cce4e3f6b5a1cf3b2207ef6f7da78434ea182924f6.jpg)

![](images/e7e3592e501d8910431bfa42a400e7509cbdd40d0efc7fb86f6ddbcbbc1bf0cb.jpg)

![](images/bc4e169cdacae28d630aaa0c2568b592681f8cacea78ef2f87f7367be044e933.jpg)

![](images/88c5162aff99c33341a96e1de699b290a7fb6e1af9adc3d860611808fccab4b5.jpg)

![](images/1e5b8731169f30d9faabd8e5cb554c8cc6095cc1b73572ca3e53ab38d33ef6e0.jpg)

![](images/7c32fa5bd5258f36b68f58e0b9bfcd4cd591da1e8d61b5d3123c9548d0f20e0e.jpg)

![](images/3765433de1470ba9891597292f65c85cc9fc2190a8362b6ae59a02f201d4df6b.jpg)

![](images/2cd9b02167a7d5963cc27b2b975ee8604db291631d52162892baed714cc1ed8d.jpg)

![](images/14c2818bb20191b6981b45ec34726ee7a07a6871b71142d7e4712f61fde5e704.jpg)  
Figure 7: Microstate transition attention weight matrices for MDD and HC in public datasets. (a–c) HUSM under the EO conditions, (d–f) HUSM under the EC conditions, and (g–i) MODMA under the EO conditions, showing averaged ten-fold attention matrices and paired t-test results. \*p<0.05.

Subsequently, we perform the first-stage microstate clustering at the trial level using the k-means algorithm[69], with the number of clusters $K = 2 : 8 \ \mathrm { : } \ 8 \ \mathrm { : }$ and to ensure robustness and reproducibility, we set 2000 random initializations and 5000 iterations. The optimal number of microstate templates is determined based on high Global Explained Variance (GEV) and low Cross-Validation criterion (CV)[70]. The optimal microstate templates obtained from all trials and participants in the previous step are subjected to a second-stage clustering using the same k-means procedure, yielding eight group-level microstate templates across participants and trials. Finally, the resulting group templates are back-fitted to the preprocessed EEG signals to derive microstate time series for each trial and each participant (each trial duration = 5 s, corresponding to 2500 sample points, hence a microstate sequence length of 2500).

## Multidimensional Data Balancing Strategy

Based on the collected data, we adopt a multidimensional sampling strategy to balance the dataset across gender distribution, class distribution, and trial count (DN+: M/F $= 1 0 / 5 7 ; \mathrm { D N - : \ M / F = 1 9 / 1 5 ) }$ . First, 10 male and 10 female participants are randomly selected from each group (DN+ and DN−), resulting in a total of 40 participants. During model training, we employ a stratified 10-fold cross-validation approach. In each fold, one participant is selected from each subgroup—male DN+, female DN+, male DN−, and female DN− as the target domain data for model testing (4 participants in total), while the remaining 36 participants are used as the source domain data for model training. Considering that the number of available EEG trials per participant after preprocessing varied (ranging from 19 to 35), we randomly sample 19 trials per participant to standardize the dataset. Consequently, the final model input had dimensions of 40 × 19 × 2500. This strategy efectively balanced the data distribution across gender, class, and trial dimensions, thereby mitigating potential model bias toward specific groups or conditions and enhancing the fairness and stability of the classification performance.

![](images/030f86957178fd3921775e964fef5100d77f0d544391705a6f5a5bda1bf2d384.jpg)  
Figure 8: Experimental paradigm. The experiment consisted of a 5-minute resting-state session and a task session. The task session included two blocks (social pain and physical pain) presented in a randomized order. Each block comprised 35 trials, and each trial consisted of four phases: (1) Preparation phase (2 s), during which participants fixated on a central cross; (2) Stimulus phase (5 s), in which a task-relevant image is presented and participants passively viewed it; (3) Emotion rating phase (up to 5 s), during which participants rated their subjective negative emotional experience using a keyboard (1 = no negative emotion, 9 = most intense negative emotion), with the phase terminating immediately upon response; and (4) Inter-trial interval (1 s). The order of image presentation within each block was randomized across participants.

## Blockwise and Randomization Strategies for Disrupting Microstate Temporal Structure

Previous studies have reported that DN+ patients exhibit abnormal microstate transition patterns when exposed to social pain stimuli[57, 71, 72]. Therefore, to enhance the model’s ability to learn NSSI related pathological microstate transition features and to reduce interference from redundant information, we apply blockwise segmentation and randomization strategies to disrupt the temporal structure of the microstate sequences before model input.

Specifically, the original microstate sequence is segmented into 50 patches using a window size of 50 and a step size of 50, as illustrated in Figure 1. Two permutation strategies are then employed to disrupt the original microstate transition patterns. 1) Strong Permutation. Randomly shufling 40–43 patches within the sequence, efectively destroying its specific transition structure. 2) Weak Permutation. Randomly shufling only 1–3 patches, thereby preserving most of the NSSI-related pathological microstate transition patterns while introducing mild temporal perturbations.

## Transformer Architecture Based on Patch-Level Feature Modeling

Compared with static microstate features (e.g., occurrence, duration, and coverage), we place greater emphasis on the dynamic transition patterns within microstate sequences, as more frequent transitions between microstates may reflect faster and more flexible information processing[22, 73]. To efectively extract discriminative transition features from these sequences, we employ a patch-level Transformer model [74]. Specifically, long microstate sequences are segmented into multiple patches, which helps reduce state redundancy and mitigate interference from irrelevant patterns. This design enables a hierarchical “local–global” modeling strategy, not only allowing the model to simultaneously capture fine-grained temporal dynamics within segments and the overall structure of the microstate sequence, but also enables the model to capture the temporal relationships of whole-brain activation state transitions in DN+ patients when exposed to social pain stimuli, thereby enhancing the representation of complex

EEG dynamics.

The overall architecture of the model is illustrated in Figure 1. Inspired by the standard Transformer framework[48], we first embed the original patch-level microstate sequences into a high-dimensional continuous space, allowing the model to capture the dynamic relationships and semantic structures among EEG microstates through the attention mechanism. Second, a randomly initialized CLS patch is added to the beginning of each sequence to facilitate the learning of both local and global representations of the microstate sequences. To further preserve and capture the temporal dependencies among tokens within the sequences, we introduce a dual positional encoding strategy to enhance the model’s ability to represent temporal structure. Specifically, the first positional encoding is applied before patch segmentation to retain token-level temporal dependencies, while the second positional encoding is applied after adding the CLS patch to model patchlevel temporal relationships, thereby reinforcing temporal representation at both local and global levels. Finally, the processed sequences are fed into a three-layer Patch Transformer, where they are transformed into feature vectors, and the weighted attention matrices from each layer are retained for subsequent analysis.

## Multi-loss Joint Optimization

Although conventional Transformer models are easy to train, they often produce low-quality features, limiting their efectiveness for complex tasks. In contrast, ordered deep modeling of microstate sequences has shown great potential in generating high-quality features from time-series data. To enhance the model’s robustness and accuracy in distinguishing between the two participant groups—particularly in identifying the pathological microstate transition patterns of DN+ patients—we incorporate consistency learning and contrastive learning modules. These components aim to strengthen the model’s ability to capture core pathological features through mechanisms that enforce both consistency and discriminative representation learning.

• Disease Specific Loss. To efectively capture the microstate transition diferences between DN+ and DN− participants, we design a disease specific loss. This loss plays a crucial role in enhancing the model’s discriminative capability, ensuring that features unique to each condition are accurately identified. The disease-specific loss function is defined as follows:

$$
\mathcal { L } _ { D i s } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( y _ { i } l o g ( \hat { y } _ { i } ) - ( 1 - y _ { i } ) l o g ( 1 - \hat { y } _ { i } ) )\tag{2}
$$

Where $y _ { i }$ denotes the true label, $\hat { y } _ { i }$ represents the predicted label, and N is the number of samples.

• Domain Adversarial Loss. To mitigate the influence of individual diferences on the model’s ability to distinguish between participant groups, we incorporate a domain adversarial loss. This loss ensures that the model learns domain invariant features, thereby enhancing its generalization capability in cross-domain tasks. The domain-adversarial loss function is defined as follows:

$$
\begin{array} { r } { \mathcal { L } _ { A d v } = - \displaystyle \frac { 1 } { 2 } \sum _ { x _ { s } \in \mathcal { D } _ { s } } \log D ( R _ { \lambda } ( F ( x _ { s } ) ) ) } \\ { - \displaystyle \frac { 1 } { 2 } \sum _ { x _ { t } \in \mathcal { D } _ { t } } \log \left( 1 - D ( R _ { \lambda } ( F ( x _ { t } ) ) ) \right) } \end{array}\tag{3}
$$

where $x _ { s }$ and $x _ { t }$ denote samples drawn from the source domain $\mathcal { D } _ { s }$ and target domain $\mathcal { D } _ { t } .$ , respectively. $F ( \cdot )$ denotes the feature extractor, $D ( \cdot )$ represents the domain discriminator, $R _ { \lambda } ( \cdot )$ is the gradient reversal layer, and λ denotes the gradient reversal coeficient.

• Consistency Learning Loss. Previous studies have shown that DN+ participants exhibit abnormal microstate transition patterns in response to pain-related emotional stimuli[57, 71, 72]. To ensure that microstate sequences with pathological transition features are accurately identified, we design a consistency learning module. In this module, a single-layer unidirectional LSTM is first applied to both the original microstate sequences and the strongly permuted microstate sequences. The final hidden state of each LSTM serves as a context vector, effectively summarizing the patch sequence. During the pseudo-label training phase, the model performs consistency learning by distinguishing the context vectors of the original sequences (label = 1) from those of the strongly permuted sequences (label = 0). The time-andchannel patch consistency loss is then defined based on binary cross-entropy. The consistency learning loss function is formulated as follows:

$$
\mathcal { L } _ { C S } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( y _ { i } \log ( \hat { y _ { i } } ) + ( 1 - y _ { i } ) \log ( 1 - \hat { y _ { i } } ) \right)\tag{4}
$$

Where $y _ { i }$ denotes the pseudo label, $\hat { y } _ { i }$ represents the predicted label, and N is the number of samples.

• Contrastive Learning Loss. To further amplify subtle diferences between microstate sequences of DN+ and DN− participants and strengthen classification boundaries, we employ the InfoNCE loss to construct a more precise feature discrimination mechanism[75]. Similarly, a single-layer unidirectional LSTM is first applied to the original microstate sequences, strongly permuted sequences, and weakly permuted sequences, with the final hidden state of each LSTM serving as a context vector. In this setup, the original sequence acts as the anchor, the weakly permuted sequence as the positive pair, and the strongly permuted sequence as the negative pair. The contrastive loss over temporal sequences is then defined as follows:

$$
\mathcal { L } _ { C T } = - l o g ( \frac { e x p ( s i m ( c _ { o } , c _ { w } ) / \tau ) } { e x p ( s i m ( c _ { o } , c _ { w } ) / \tau ) + e x p ( s i m ( c _ { o } , c _ { s } ) / \tau ) } )\tag{5}
$$

Where $c _ { o } , c _ { w }$ and $c _ { s }$ denote the context vector of the original sequence, weak permutation, and strong permutation sequences, τ is the temperature parameter, and sim represents the cosine similarity.

• Total Loss Function. The overall loss for multi-loss joint optimization is defined as the direct sum of all individual loss components. In summary, the classification (disease-specific) loss establishes the model’s basic ability to distinguish between participant groups, the domain-adversarial loss enhances the model’s generalization capability, the consistency loss ensures the stability of learning meaningful features, and the contrastive loss improves the precision in capturing subtle diferences.

$$
\mathcal { L } _ { T o t a l } = \alpha \mathcal { L } _ { D i s } + \beta \mathcal { L } _ { A d v } + \delta \mathcal { L } _ { C S } + \theta \mathcal { L } _ { C T }\tag{6}
$$

where α, β, δ and θ are weighting factors for the corresponding loss terms.

The overall loss function ensures that the model learns robust, multi-dimensional EEG feature representations that are invariant to signal noise, sensitive to gender differences, adaptable across domains, and efective in classifying NSSI behaviors. By jointly optimizing these objectives, the model is better equipped to handle the complexities involved in EEG signal analysis for studies on depression and NSSI.

## Statistical Analysis

For the cosine similarity between the learned neural representations and stimulus image features under diferent pain conditions, statistical comparisons between the two groups are performed using two-way mixed ANOVA. Differences in microstate transition attention weights between disease and sex subgroups are evaluated using paired ttests on the attention weight matrices obtained from each cross-validation fold. Behavioral analyses are conducted using correlation and mediation analyses. For the temporal validity analysis, independent two-sample t-tests are performed to compare microstate features between the two groups. Significance is indicated as follows in the figures: $* p \le 0 . 0 5 ; * * p \le 0 . 0 1 ; * * * p \le 0 . 0 0 1$

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 62522608, the Guangdong Provincial Project (Grant No. 2024TQ08X330), Brain Science and Brain-like Intelligence Technology - National Science and Technology Major Project (2021ZD0200500), Shenzhen-Hong Kong Institute of Brain Science-Shenzhen Fundamental Research Institutions (2023SHIBS0003), the Shenzhen Science and Technology Program (No. JCYJ20241202124222027 and JCYJ20241202124209011), and the Key Research and Development Program of Hunan Province (2025QK3008).

## Author Contributions

Ying Xu: Conceptualization, Methodology, Formal analysis, Investigation, Writing – original draft. Xiaojun Liang: Methodology, Validation, Investigation. Li Zhang: Writing – review & editing. Yixuan Yuan: Investigation, Writing – review & editing. Gan Huang: Data curation, Visualization. Yongjie Zhou: Resources, Investigation, Writing – review & editing. Zhen Liang: Conceptualization, Methodology, Supervision, Project administration, Funding acquisition, Writing – review & editing.

## Conflicts of Interest

The authors declare that there is no conflict of interest regarding the publication of this article.

## Data Availability

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## References

1. Muehlenkamp JJ, Claes L, Havertape L, and Plener PL. International prevalence of adolescent non-suicidal self-injury and deliberate self-harm. Child and adolescent psychiatry and mental health 2012;6:10.

2. Gillies D, Christou MA, Dixon AC, et al. Prevalence and characteristics of self-harm in adolescents: meta-analyses of community-based studies 1990– 2015. Journal of the American Academy of Child & Adolescent Psychiatry 2018;57:733–41.

3. Klonsky ED, May AM, and Glenn CR. The relationship between nonsuicidal self-injury and attempted suicide: converging evidence from four samples. Journal of abnormal psychology 2013;122:231.

4. Ribeiro MT, Singh S, and Guestrin C. Modelagnostic interpretability of machine learning. arXiv preprint arXiv:1606.05386 2016.

5. Hooley JM, Fox KR, and Boccagno C. Nonsuicidal self-injury: diagnostic challenges and current perspectives. Neuropsychiatric disease and treatment 2020:101–12.

6. Joiner T. Why people die by suicide. Harvard University Press, 2005.

7. Van Orden KA, Witte TK, Cukrowicz KC, Braithwaite SR, Selby EA, and Joiner Jr TE. The interpersonal theory of suicide. Psychological review 2010;117:575.

8. Koenig J, Rinnewitz L, Niederbäumer M, et al. Longitudinal development of pain sensitivity in adolescent non-suicidal self-injury. Journal of psychiatric research 2017;89:81–4.

9. Kross E, Berman MG, Mischel W, Smith EE, and Wager TD. Social rejection shares somatosensory representations with physical pain. Proceedings of the National Academy of Sciences 2011;108:6270–5.

10. Eisenberger NI. The pain of social disconnection: examining the shared neural underpinnings of physical and social pain. Nature reviews neuroscience 2012;13:421–34.

11. Groschwitz RC, Plener PL, Groen G, Bonenberger M, and Abler B. Diferential neural processing of social exclusion in adolescents with non-suicidal selfinjury: An fMRI study. Psychiatry Research: Neuroimaging 2016;255:43–9.

12. Eisenberger NI, Lieberman MD, and Williams KD. Does rejection hurt? An fMRI study of social exclusion. Science 2003;302:290–2.

13. Lehmann D, Ozaki H, and Pál I. EEG alpha map series: brain micro-states by space-oriented adaptive segmentation. Electroencephalography and clinical neurophysiology 1987;67:271–88.

14. Britz J, Van De Ville D, and Michel CM. BOLD correlates of EEG topography reveal rapid resting-state network dynamics. Neuroimage 2010;52:1162–70.

15. Lehmann D, Faber PL, Galderisi S, et al. EEG microstate duration and syntax in acute, medication-naive, first-episode schizophrenia: a multi-center study. Psychiatry Research: Neuroimaging 2005;138:141–56.

16. Cruz JR da, Favrod O, Roinishvili M, et al. EEG microstates are a candidate endophenotype for schizophrenia. Nature communications 2020;11:3089.

17. Raeisi Z, Bashiri O, EskandariNasab M, Arshadi M, Golkarieh A, and Najafzadeh H. EEG microstate biomarkers for schizophrenia: a novel approach using deep neural networks. Cognitive Neurodynamics 2025;19:68.

18. Murphy M, Whitton AE, Deccy S, et al. Abnormalities in electroencephalographic microstates are state and trait markers of major depressive disorder. Neuropsychopharmacology 2020;45:2030–7.

19. Zhao Z, Niu Y, Zhao X, et al. EEG microstate in first-episode drug-naive adolescents with depression. Journal of neural engineering 2022;19:056016.

20. Van de Ville D, Britz J, and Michel CM. EEG microstate sequences in healthy humans at rest reveal scale-free dynamics. Proceedings of the National Academy of Sciences 2010;107:18179–84.

21. Peng RJ, Fan Y, Li J, Zhu F, Tian Q, and Zhang XB. Abnormalities of electroencephalography microstates in patients with depression and their association with cognitive function. World Journal of Psychiatry 2024;14:128.

22. Khanna A, Pascual-Leone A, Michel CM, and Farzan F. Microstates in resting-state EEG: current status and future directions. Neuroscience & Biobehavioral Reviews 2015;49:105–13.

23. Gao S, Calhoun VD, and Sui J. Machine learning in major depression: From classification to treatment outcome prediction. CNS neuroscience & therapeutics 2018;24:1037–52.

24. Xia M, Zhang Y, Wu Y, and Wang X. An end-to-end deep learning model for EEG-based major depressive disorder classification. IEEE access 2023;11:41337– 47.

25. Liang Z, Ye W, Liu Q, Zhang L, Huang G, and Zhou Y. NSSI-Net: A multi-concept GAN for nonsuicidal self-injury detection using high-dimensional EEG in a semi-supervised framework. IEEE Journal of Biomedical and Health Informatics 2025.

26. Li J, Li N, Shao X, et al. Altered brain dynamics and their ability for major depression detection using EEG microstates analysis. IEEE transactions on afective computing 2021;14:2116–26.

27. Rotge JY, Lemogne C, Hinfray S, et al. A metaanalysis of the anterior cingulate contribution to social pain. Social cognitive and afective neuroscience 2015;10:19–27.

28. Apkarian AV, Bushnell MC, Treede RD, and Zubieta JK. Human brain mechanisms of pain perception and regulation in health and disease. European journal of pain 2005;9:463–84.

29. Tracey I and Mantyh PW. The cerebral signature for pain perception and its modulation. Neuron 2007;55:377–91.

30. Biswal B, Zerrin Yetkin F, Haughton VM, and Hyde JS. Functional connectivity in the motor cortex of resting human brain using echo-planar MRI. Magnetic resonance in medicine 1995;34:537–41.

31. Raichle ME, MacLeod AM, Snyder AZ, Powers WJ, Gusnard DA, and Shulman GL. A default mode of brain function. Proceedings of the national academy of sciences 2001;98:676–82.

32. Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision. In: International conference on machine learning. PmLR. 2021:8748–63.

33. Pizzagalli DA et al. Electroencephalography and high-density electrophysiological source localization. Handbook of psychophysiology 2007;3:56–84.

34. Glenn CR and Klonsky ED. Social context during non-suicidal self-injury indicates suicide risk. Personality and Individual Diferences 2009;46:25–9.

35. Menon V. Large-scale brain networks and psychopathology: a unifying triple network model. Trends in cognitive sciences 2011;15:483–506.

36. Kaiser RH, Andrews-Hanna JR, Wager TD, and Pizzagalli DA. Large-scale network dysfunction in major depressive disorder: a meta-analysis of restingstate functional connectivity. JAMA psychiatry 2015;72:603–11.

37. Nock MK. Why do people hurt themselves? New insights into the nature and functions of selfinjury. Current directions in psychological science 2009;18:78–83.

38. Prinstein MJ, Boergers J, and Spirito A. Adolescents and their friends’ health-risk behavior: Factors that alter or add to peer influence. Journal of pediatric psychology 2001;26:287–98.

39. Spitzer RL, Kroenke K, Williams JB, Group PHQPCS, et al. Validation and utility of a self-report version of PRIME-MD: the PHQ primary care study. Jama 1999;282:1737–44.

40. Spitzer RL, Kroenke K, Williams JB, and Löwe B. A brief measure for assessing generalized anxiety disorder: the GAD-7. Archives of internal medicine 2006;166:1092–7.

41. Cover T and Hart P. Nearest neighbor pattern classification. IEEE transactions on information theory 1967;13:21–7.

42. Breiman L, Friedman J, Stone CJ, and Olshen RA. Classification and regression trees. CRC press, 1984.

43. Cortes C and Vapnik V. Support-vector networks. Machine learning 1995;20:273–97.

44. Breiman L. Random forests. Machine learning 2001;45:5–32.

45. Rumelhart DE, Hinton GE, and Williams RJ. Learning representations by back-propagating errors. nature 1986;323:533–6.

46. Hochreiter S and Schmidhuber J. Long short-term memory. Neural computation 1997;9:1735–80.

47. Cho K, Van Merriënboer B, Gulçehre Ç, et al. Learning phrase representations using RNN encoder– decoder for statistical machine translation. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014:1724– 34.

48. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in neural information processing systems 2017;30.

49. Tzeng E, Hofman J, Zhang N, Saenko K, and Darrell T. Deep domain confusion: Maximizing for domain invariance. arXiv preprint arXiv:1412.3474 2014.

50. Long M, Cao Y, Wang J, and Jordan M. Learning transferable features with deep adaptation networks. In: International conference on machine learning. PMLR. 2015:97–105.

51. Ganin Y, Ustinova E, Ajakan H, et al. Domainadversarial training of neural networks. Journal of machine learning research 2016;17:1–35.

52. Sun B and Saenko K. Deep coral: Correlation alignment for deep domain adaptation. In: European conference on computer vision. Springer. 2016:443–50.

53. He K, Fan H, Wu Y, Xie S, and Girshick R. Momentum contrast for unsupervised visual representation learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020:9729–38.

54. Chen T, Kornblith S, Norouzi M, and Hinton G. A simple framework for contrastive learning of visual representations. In: International conference on machine learning. PmLR. 2020:1597–607.

55. Craig AD. How do you feel—now? The anterior insula and human awareness. Nature reviews neuroscience 2009;10:59–70.

56. Shackman AJ, Salomons TV, Slagter HA, Fox AS, Winter JJ, and Davidson RJ. The integration of negative afect, pain and cognitive control in the cingulate cortex. Nature Reviews Neuroscience 2011;12:154–67.

57. Hu Jh, Zhou Dd, Ma Ll, et al. A resting-state electroencephalographic microstates study in depressed adolescents with non-suicidal self-injury. Journal of psychiatric research 2023;165:264–72.

58. Tse NY, Ratheesh A, Tian YE, et al. A mega-analysis of functional connectivity and network abnormalities in youth depression. Nature Mental Health 2024;2:1169–82.

59. Groschwitz RC and Plener PL. The neurobiology of non-suicidal self-injury (NSSI): A review. Suicidology Online 2012;3:24–32.

60. Etkin A, Büchel C, and Gross JJ. The neural bases of emotion regulation. Nature reviews neuroscience 2015;16:693–700.

61. Dalgleish T, Walsh ND, Mobbs D, et al. Social pain and social gain in the adolescent brain: A common neural circuitry underlying both positive and negative social evaluation. Scientific reports 2017;7:42010.

62. Mumtaz W, Xia L, Mohd Yasin MA, Azhar Ali SS, and Malik AS. A wavelet-based technique to predict treatment outcome for major depressive disorder. PloS one 2017;12:e0171409.

63. Cai H, Yuan Z, Gao Y, et al. A multi-modal open dataset for mental-disorder analysis. Scientific data 2022;9:178.

64. Bréchet L, Brunet D, Birot G, Gruetter R, Michel CM, and Jorge J. Capturing the spatiotemporal dynamics of self-generated, task-initiated thoughts with EEG and fMRI. Neuroimage 2019;194:82–92.

65. Koenig T, Prichep L, Lehmann D, et al. Millisecond by millisecond, year by year: normative EEG microstates and developmental stages. Neuroimage 2002;16:41–8.

66. Milz P, Faber PL, Lehmann D, Koenig T, Kochi K, and Pascual-Marqui RD. The functional significance of EEG microstates—Associations with modalities of thinking. Neuroimage 2016;125:643–56.

67. Kim K, Duc NT, Choi M, and Lee B. EEG microstate features according to performance on a mental arithmetic task. Scientific reports 2021;11:343.

68. Huang G. EEG/ERP data analysis toolboxes. In: EEG signal processing and feature extraction. Springer, 2019:407–34.

69. Lloyd S. Least squares quantization in PCM. IEEE transactions on information theory 1982;28:129–37.

70. Murray MM, Brunet D, and Michel CM. Topographic ERP analyses: a step-by-step tutorial review. Brain topography 2008;20:249–64.

71. Song YW, Kim S, Lee HA, Shim SH, and Kim JS. EEG microstate-based classification using machine learning in depressed adolescents with and without non-suicidal self-injury. Progress in Neuro-Psychopharmacology and Biological Psychiatry 2025:111538.

72. Bao C, Zhang Q, Zou H, et al. EEG Microstates as Biomarkers of Drug-Naı¨ve Major Depressive Disorder in Young Adults with Non-Suicidal Self-Injury. Journal of Psychiatric Research 2025.

73. Liu J, Xu J, Zou G, He Y, Zou Q, and Gao JH. Reliability and individual specificity of EEG microstate characteristics. Brain Topography 2020;33:438–49.

74. Nie Y, Nguyen NH, Sinthong P, and Kalagnanam J. A time series is worth 64 words: Longterm forecasting with transformers. arXiv preprint arXiv:2211.14730 2022.

75. Oord Avd, Li Y, and Vinyals O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 2018.