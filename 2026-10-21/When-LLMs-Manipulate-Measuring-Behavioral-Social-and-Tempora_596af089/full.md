# When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects

Sadia Asif<sup>1</sup> Israel Antonio Rosales Laguan<sup>2</sup> Shumaila Asif<sup>3</sup> Haris Khan<sup>3</sup> Muneeb Asif<sup>3</sup> Muhammad Zeeshan Karamat<sup>4</sup>

<sup>1</sup>Rensselaer Polytechnic Institute, USA <sup>2</sup>Independent Researcher, Colombia <sup>3</sup>National University of Sciences and Technology, Pakistan <sup>4</sup>Virginia Tech, USA

<sup>#</sup> asifs@rpi.edu

## Abstract

Large language models (LLMs) can influence user trust and decision-making through subtle linguistic cues, yet the behavioral effects of manipulative outputs remain poorly quantified. We introduce a behavioral computing framework for measuring behavioral vulnerability, defined as users’ susceptibility to manipulative AI responses. Building on DarkPatterns-LLM Asif et al. [2025], we generate controlled manipulative stimuli using Manipulation Gradient Descent (MGD), targeting biases associated with authority, urgency, and social proof. We analyze participant responses using the Multi-Stakeholder Impact Assessment Network (MSIAN) to model cross-user influence and the Temporal Harm Propagation Model (THPM) to characterize the persistence of behavioral effects over time. In a study involving 120 participants, manipulative framing produced significant changes in trust calibration and compliance (p < 0.01), with effects also propagating across social contexts. These findings highlight the need to extend LLM safety evaluation beyond model outputs to measurable behavioral consequences and provide a computational framework for quantifying individual, social, and temporal risks arising from manipulative AI interactions.

## 1 Introduction

Large language models (LLMs) are increasingly deployed to support decision-relevant tasks in domains such as healthcare documentation, patient monitoring, and clinical activity recognition Hamd hana et al. [2024], Syulistyo et al. [2024], Cleland et al. [2024]. While generative AI systems continue to expand in capability and application Khan et al. [2025], they also introduce behavioral risks related to persuasion, trust, and user autonomy Bommasani et al. [2021], Weidinger et al. [2021]. Unlike traditional interfaces, LLMs can adapt linguistic framing through cues such as authority, urgency, and social proof, potentially exploiting cognitive biases to influence user behavior. Such mechanisms resemble dark patterns, traditionally studied in digital interface design.

Prior work in human computer interaction (HCI) and persuasive technology has shown that design choices can influence user decisions by leveraging cognitive biases, through dark-pattern taxonomies such as nagging, obstruction, sneaking, interface interference, and forced action Gray et al. [2018], as well as mechanisms such as social proof and scarcity Mathur et al. [2019]. Recent AI safety research has also developed methods for mitigating unsafe model behavior, including fine-tuning procedures that preserve safety alignment Asif and Amiri [2026], targeted harm-mitigation frameworks Campbell et al. [2026], and analyses of the parameter-level fragility of safety mechanisms Karamat and Garcia [2026]. However, such approaches primarily operate at the level of model outputs and internal safety mechanisms, rather than measuring the downstream behavioral effects of manipulative content on users. Consequently, limited understanding remains of how manipulative AI language affects user trust and compliance, whether these effects persist over time, and how they propagate through social interactions. This gap motivates a shift from content-level detection toward behavior-level safety evaluation.

To address this gap, we introduce a behavioral computing framework for quantifying behavioral vulnerability, defined as the degree to which users’ judgments, trust, and compliance are altered by manipulative AI responses. Building on the DarkPatterns-LLM benchmark Asif et al. [2025], our framework integrates four components: Manipulation Gradient Descent (MGD) for generating controlled manipulative stimuli, the Multi-Stakeholder Impact Assessment Network (MSIAN) for modeling cross-user influence, the Temporal Harm Propagation Model (THPM) for characterizing behavioral persistence and recovery, and Dynamic Contextual Robustness Assessment (DCRA) for evaluating sensitivity to contextual reframing.

We evaluate the framework in a controlled study involving 120 participants exposed to manipulative and neutral LLM responses across authority, emotional urgency, social proof, and related framing conditions. We measure behavioral outcomes, including compliance, trust calibration, response latency, and post-interaction assessments. The results show that manipulative framing significantly increases behavioral vulnerability, with authority-based cues producing the strongest effects. Manipulationinduced behavioral changes also propagate across social contexts and persist beyond the initial interaction.

Our contributions are threefold: (1) we extend LLM safety evaluation from manipulative outputs to measurable human behavioral outcomes; (2) we introduce a multi-layer framework for modeling individual, social, temporal, and contextual dimensions of manipulation; and (3) we define behavioral safety metrics, including the Behavioral Vulnerability Index (BVI), Cross-Influence Coefficient (CIC), Temporal Recovery Rate (TRR), and Contextual Vulnerability Score (CVS), to quantify behavioral susceptibility, social propagation, temporal recovery, and contextual sensitivity.

## 2 Related Work

Behavioral Computing. Activity and Behavior Computing (ABC) focuses on the quantitative modeling of human behavior using signals such as motion, context, and social interaction. Prior work has primarily examined physical and cognitive states, including activity recognition, affect, and health-related behavior, through sensor-driven observations and machine learning Plötz et al. [2011], Ahmad et al. [2026]. More recent research extends this paradigm toward deep sequence models that capture richer temporal context in human activity recognition Hiremath and Plötz [2021]. However, these approaches generally treat AI systems as observers or facilitators rather than behavioral influencers. As LLMs increasingly function as conversational agents, their linguistic outputs can serve as behavioral stimuli that influence user judgments and decisions. Our work extends ABC from behavior recognition toward behavior modulation by modeling how AI-generated language affects downstream human behavior.

Persuasive Technology and Cognitive Bias. Persuasive technology examines how computational systems shape attitudes and behaviors Fogg [2003], often through cognitive mechanisms such as authority, social proof, scarcity, and urgency Cialdini [2001], which persuasive-systems design frameworks have formalized into actionable design principles Oinas-Kukkonen and Harjumaa [2008, 2009]. Prior HCI research has shown that interface framing and perceived expertise can influence user trust and compliance Rahman and Adaji [2025]. These effects are particularly relevant to conversational systems, where linguistic framing, tone, and contextual continuity can introduce persuasive cues without altering the underlying semantic content. The distinction between persuasion and manipulation therefore depends not only on the information presented but also on transparency, user autonomy, and the mechanisms through which behavior is influenced. Our framework examines these mechanisms by measuring behavioral responses to manipulative linguistic framing.

Manipulation and Dark Patterns in AI. Research on dark patterns has documented how digital systems exploit cognitive biases to steer users toward undesirable outcomes Gray et al. [2018], Mathur et al. [2019], Fogg [2003]. Related AI safety research has begun to characterize analogous manipulative behaviors in generative systems. In particular, DarkPatterns-LLM Asif et al. [2025] introduced a taxonomy and computational benchmark covering mechanisms such as authority bias,

# Multi-Layer Behavioral Analysis Framework

![](images/f8d13d83f98af0d05f2a35a0030383ab7047ef797dafd0aa72c668b109fa2428.jpg)  
Figure 1: Overview of the proposed multi-layer behavioral analysis framework. MGD generates controlled manipulative stimuli, MSIAN models cross-user influence, THPM characterizes temporal persistence and recovery, and DCRA evaluates sensitivity to contextual reframing.

emotional coercion, scarcity framing, and social proof. However, existing evaluations primarily focus on manipulation in model outputs, with comparatively limited attention to its measurable effects on user trust, compliance, behavioral persistence, and social influence. We address this gap by treating manipulative AI communication as a behavioral perturbation and extending content-level manipulation detection toward behavior-level risk quantification.

## 3 Methodology

Our objective is to quantify how manipulative AI outputs influence human behavior across individual, social, temporal, and contextual dimensions. Building on the four-layer architecture of DarkPatterns-LLM Asif et al. [2025], we integrate computational manipulation discovery with behavioral data acquisition and modeling. Figure 1 illustrates the resulting pipeline, consisting of four interconnected layers.

## 3.1 Layer 1: MGD

The first layer generates controlled manipulative stimuli through semantic-level optimization. Given an input prompt x and a target manipulation type $m \in$ {authority, urgency, social, emotional}, MGD generates a linguistically coherent variant $x ^ { \prime }$ that maximizes manipulation potential while preserving semantic similarity:

$$
\operatorname* { m a x } _ { x ^ { \prime } } \ L _ { \mathrm { m a n i p } } ( x ^ { \prime } , f ( x ^ { \prime } , \theta ) ) - \lambda D _ { \mathrm { s e m } } ( x , x ^ { \prime } ) - \gamma N _ { \mathrm { l a n g } } ( x ^ { \prime } ) .\tag{1}
$$

Here, $L _ { \mathrm { m a n i p } }$ measures expected persuasion strength, $D _ { \mathrm { s e m } }$ penalizes semantic deviation from the original prompt, and $N _ { \mathrm { l a n g } }$ penalizes unnatural phrasing. The resulting outputs serve as controlled experimental stimuli.

We further define the Manipulation Coefficient $M _ { c } \in [ 0 , 1 ]$ , derived from the normalized gradient magnitude in semantic embedding space, to quantify manipulation intensity. Higher $M _ { c }$ values indicate greater manipulation potential and are used to stratify experimental conditions.

## 3.2 Layer 2: MSIAN

The second layer models behavioral influence across participants. Each participant $u _ { i }$ is represented as a node in a directed behavioral graph $G = ( V , { \bar { E } } )$ , where edges represent communicative or observational exposure through shared tasks or discussions.

Let $h _ { u _ { i } } \in \mathbb { R } ^ { d }$ denote the behavioral state of participant $u _ { i } ,$ capturing factors such as trust, compliance probability, and emotional valence. MSIAN models behavioral state transitions using a graphattention mechanism:

$$
h _ { u _ { i } } ^ { ( t + 1 ) } = \sigma \left( \sum _ { j : ( u _ { j } , u _ { i } ) \in E } \alpha _ { i j } W h _ { u _ { j } } ^ { ( t ) } + B M _ { c , i } \right) ,\tag{2}
$$

where $\alpha _ { i j }$ denotes the learned influence weight between participants, W and B are trainable matrices, and $M _ { c , i }$ represents the manipulation exposure of participant $u _ { i }$ . This formulation captures both direct susceptibility and the secondary propagation of manipulation through social interactions.

Observed behavioral measures, including compliance rate $( C _ { i } )$ , trust calibration $( T _ { i } )$ , and reflection latency $( R _ { i } )$ , are incorporated into $h _ { u _ { i } }$ during iterative updates to support participant-level and cross-participant behavioral prediction.

## 3.3 Layer 3: THPM

The third layer models the persistence and decay of manipulation-induced behavioral changes over time. Inspired by epidemiological models, participants are represented using three behavioral states: Susceptible (S), Influenced (I), and Recovered (R). Their temporal evolution follows modified susceptible influenced recovered dynamics:

$$
\frac { d S } { d t } = - \beta ( M _ { c } , N _ { \mathrm { M S I A N } } ) S I + \mu R ,\tag{3}
$$

$$
\frac { d I } { d t } = \beta ( M _ { c } , N _ { \mathrm { M S I A N } } ) S I - ( \alpha + \nu ) I ,\tag{4}
$$

$$
\frac { d R } { d t } = \alpha I - \mu R ,\tag{5}
$$

where $\beta$ denotes the manipulation transmission rate, modulated by manipulation intensity $M _ { c }$ and MSIAN network connectivity $N _ { \mathrm { M S I A N } } ;$ α represents awareness-based recovery, µ captures the transition from recovered to susceptible, and ν models the effect of repeated exposure.

Temporal trajectories derived from THPM are used to compute TRR and the Behavioral Persistence Index (BPI), which characterize recovery and persistence of manipulation-induced behavioral effects beyond the initial interaction.

## 3.4 Layer 4: DCRA

The fourth layer evaluates how behavioral susceptibility varies across contextual conditions. MGDgenerated stimuli are reframed along four dimensions:

1. Authority Framing: varying the perceived expertise of the AI speaker;

2. Emotional Tone: varying sentiment and urgency cues;

3. Social Proof Context: introducing consensus or popularity cues; and

4. Temporal Framing: introducing deadline or immediacy cues.

Participants are assigned to contextual variants while task semantics are held constant, allowing us to assess the stability of behavioral susceptibility under reframing. We quantify this variation using CVS:

$$
\mathrm { C V S } = 1 - \frac { \mu ( \Delta B ) } { \sigma ( \Delta B ) } ,\tag{6}
$$

where $\Delta B$ denotes behavioral deviation across contextual variants. Lower CVS values indicate greater stability of resistance to manipulative framing.

## 3.5 Behavioral Data

The framework uses behavioral observations, including decision outcomes, response latency, trust calibration, and post-interaction assessments. Detailed data-collection procedures are provided in Section 4.

## 3.6 Derived Behavioral Metrics

We define three primary behavioral indicators for standardized evaluation:

• BVI: A composite measure of compliance, trust, and emotional reactivity:

$$
\mathrm { B V I } _ { i } = \alpha _ { 1 } C _ { i } + \alpha _ { 2 } \big ( 1 - T _ { i } \big ) + \alpha _ { 3 } E _ { i } .\tag{7}
$$

• TRR: The rate at which behavior returns toward baseline following manipulation exposure, derived from THPM dynamics.

• CIC: A normalized measure of behavioral influence among connected participants, derived from MSIAN attention weights.

Together, these metrics characterize immediate susceptibility, temporal recovery, and social propagation of manipulation-induced behavioral effects.

## 4 Experimental Setup and Evaluation Protocol

We evaluated the proposed framework through a controlled human-AI interaction study involving 120 participants. All participants provided informed consent, and the study followed institutionally approved ethical procedures. Additional details on the experimental protocol and participants are provided in Appendix A.

## 4.1 Participants and Experimental Conditions

A total of 120 participants $( N = 1 2 0 )$ were recruited through academic mailing lists and professional online communities. All participants had prior experience with conversational AI systems but were unaware of the study’s manipulation objectives.

Each participant completed twelve interaction tasks involving manipulative and neutral LLM responses. The experimental conditions included four manipulation categories: Authority Framing (A), invoking expert or institutional authority; Emotional Urgency (E), introducing scarcity, fear, or time pressure; Social Proof (S), conveying consensus or majority endorsement; and Prosocial Altruism (P), appealing to moral or communal benefit. A Baseline Control (C) condition used semantically comparable responses without manipulative framing.

Stimuli were generated using MGD at three manipulation-intensity levels: low $( M _ { c } < 0 . 3 3 )$ , medium $( 0 . 3 3 \leq M _ { c } \mathrm { ~ \bar { ~ } { ~ < ~ } 0 . 6 6 ) }$ , and high $( M _ { c } \ge 0 . 6 6 )$ . Experimental conditions and intensity levels were counterbalanced across participants using a Latin-square design to minimize order effects.

## 4.2 Behavioral Data Acquisition

Participants interacted with AI-generated responses through a standardized web interface. For each interaction, we recorded decision outcomes, response latency, confidence ratings, emotional states derived from post-interaction reflections, and changes in self-reported trust in AI. These measures capture behavioral responses related to compliance, deliberation, affect, and trust.

## 4.3 Behavioral Graph Construction

To evaluate cross-participant influence, participants were subsequently exposed to anonymized excerpts of prior participants’ decision narratives and asked to review and respond to them. These exposures defined a directed behavioral graph $G = ( V , E )$ , where an edge $e _ { j i }$ represents exposure of participant $u _ { i }$ to participant $u _ { j } \mathbf { \bar { s } }$ response. The resulting graph served as input to MSIAN for modeling secondary behavioral influence and collective susceptibility.

![](images/9d3955bb71523f816cfbb19b6fe98613336b69f004a921b3a32cde7627f1aba9.jpg)  
Figure 2: BVI across manipulation types and intensity levels. Bars represent mean BVI under low, medium, and high manipulation intensities.

## 4.4 Evaluation and Statistical Analysis

Behavioral responses were evaluated using the metrics introduced in Section 3: BVI, CIC, TRR, and CVS. All metrics were normalized to [0, 1] to facilitate comparison across participants and experimental conditions.

Statistical significance was assessed using repeated-measures analysis of variance (ANOVA), with Greenhouse Geisser correction applied where appropriate, followed by Bonferroni-corrected pairwise comparisons. Effect sizes were reported using partial $\eta ^ { 2 }$ . Pearson correlation and regression analyses were used to examine associations between manipulation intensity, experimental conditions, and behavioral outcomes. Predictive performance was evaluated using 10-fold cross-validation, with $p < 0 . 0 5$ considered statistically significant. Further experimental and statistical details are provided in Appendix A.

## 5 Results and Analysis

We report results across four dimensions of the proposed framework: immediate behavioral vulnerability, cross-participant influence, temporal persistence, and contextual robustness. Unless otherwise stated, statistical significance is assessed at $p < 0 . 0 5$

## 5.1 Behavioral Vulnerability Across Manipulation Types

Manipulative framing produced significantly higher BVI values than the neutral control $( F ( 4 , 4 7 6 ) =$ 28.41, $p < 0 . 0 0 1 , \eta _ { p } ^ { \bar { 2 } } \overset { ^ { . } } { = } 0 . 4 1 )$ . As shown in Figure 2, BVI increased monotonically from low to high manipulation intensity across all manipulation types, indicating a dose response relationship between manipulation intensity and behavioral susceptibility.

Authority framing produced the highest mean BVI (0.58±0.13), followed by emotional (0.53), socialproof (0.49), and prosocial (0.46) framing, whereas the control condition exhibited a substantially lower value (0.22). A strong positive correlation between the Manipulation Coefficient $M _ { c }$ and BVI $( r = 0 . 7 1 )$ further indicates an association between manipulation intensity and behavioral vulnerability. Complete BVI values across intensity levels are reported in Appendix B.

## 5.2 Cross-Influence and Network Propagation

MSIAN analysis revealed secondary propagation of manipulation effects across the behavioral interaction graph. The mean CIC was 0.37, indicating measurable peer-mediated behavioral influence. Authority and emotional framing exhibited the highest propagation potential $( \mathrm { C I C } _ { A } = 0 . 4 2$ and $\mathrm { C I C } _ { E } = 0 . 3 9$ , respectively), whereas prosocial framing showed a lower diffusion effect $\mathrm { ( C I C } _ { P } =$ 0.28).

## Cross-layer Correlations among Framework Components

![](images/0e241aa3473105d6aa2fa66d7cf3045ac6e807dc986c4830c4efe7add1127942.jpg)

![](images/fb688f064ff370608f0045a909b4bf4b2f587093587c6f8d61cef76b67858550.jpg)  
Figure 3: Cross-layer correlations between MGD-derived manipulation intensity, BVI, and BPI. Left: manipulation intensity versus BVI. Right: BVI versus BPI.

Regression analysis revealed a positive association between CIC and BVI $( \beta = 0 . 5 6 , t = 5 . 8 3 )$ indicating that participants with greater immediate vulnerability also exhibited stronger downstream influence. MSIAN attention weights further indicated that social connectivity amplified manipulation diffusion by an average of 23% relative to isolated participants. The corresponding behavioral influence graph is provided in Appendix B.

## 5.3 Temporal Persistence of Manipulation Effects

Temporal persistence was evaluated immediately $( T _ { 0 } )$ , after 24 hours $( T _ { 1 } )$ , and after one week $( T _ { 2 } )$ using THPM. BPI decreased over time, indicating gradual recovery from manipulation-induced behavioral effects.

Authority framing exhibited the slowest recovery, with $\mathrm { T R R } = 0 . 4 2$ and $\mathrm { B P I } = 0 . 6 1 $ at $T _ { 2 }$ . Emotional framing followed with TRR = 0.48 and $\mathrm { B P I } = 0 . 5 7 $ , compared with $\mathrm { T R R } = 0 . 6 8$ and $\mathrm { B P I = 0 . 3 2 }$ for the control condition. Repeated exposure to three or more manipulative interactions was associated with an average 28% reduction in recovery rate, suggesting a compounding effect of repeated manipulation. Complete TRR and BPI results are reported in Appendix B.

## 5.4 Contextual Robustness and Framing Sensitivity

Behavioral consistency across contextual variants was quantified using CVS. The overall mean CVS was 0.63, indicating moderate robustness to reframing. However, CVS varied significantly across manipulation conditions $( F ( 3 , 3 5 4 ) = 1 1 . 6 2 , p < 0 . 0 \bar { 0 } 1 )$

Authority and urgency framing produced greater behavioral variance across contextually reframed tasks, indicating reduced stability under changes in linguistic framing. These findings support the DCRA analysis, suggesting that behavioral susceptibility depends not only on semantic content but also on how that content is presented.

## 5.5 Integrated Framework Performance

Cross-layer analysis revealed strong associations between computational manipulation signals and observed behavioral outcomes. MGD-derived manipulation intensity was strongly correlated with $\mathbf { B V I } \left( r = 0 . 9 0 \right)$ , while BVI was also strongly associated with temporal persistence measured by BPI $( r = 0 . 8 0 )$

As shown in Figure 3, greater manipulation intensity was associated with higher immediate behavioral susceptibility, while participants with higher BVI also exhibited greater behavioral persistence over time. Across all experimental tasks, the integrated four-layer framework achieved an average behavioral-outcome prediction accuracy of 86.4% under 10-fold cross-validation.

Overall, these results indicate that manipulative linguistic cues are associated with intensity-dependent behavioral effects that can propagate through social interactions, persist over time, and vary under contextual reframing.

## 6 Discussion and Implications

Our findings show that behavioral vulnerability to manipulative LLM outputs is context-dependent and extends beyond immediate individual responses. Variations in authority, urgency, social proof, and related framing produced measurable differences in compliance, trust calibration, and behavioral persistence, even when task semantics were comparable. These findings suggest that behavioral vulnerability is not necessarily a fixed user characteristic but a dynamic response shaped by linguistic framing and interaction context.

## 6.1 Behavioral Interpretation and Context Dependence

Authority and emotional framing produced the strongest behavioral effects, consistent with the role of perceived expertise and urgency in shaping human decision-making. The positive association between manipulation intensity $\dot { M _ { c } }$ and BVI further suggests that stronger manipulative framing is associated with greater behavioral susceptibility.

The temporal results show that manipulation-induced effects can persist beyond the initial interaction. Lower TRR values under authority and emotional framing, together with behavioral persistence at the one-week follow-up, suggest that susceptibility is not necessarily transient. Repeated exposure was also associated with lower recovery rates, indicating that manipulative interactions may have cu mulative behavioral effects. These findings motivate safety evaluations that consider both immediate responses and longer-term behavioral outcomes.

Contextual reframing introduced additional variation in susceptibility, indicating that semantically comparable content can elicit different behavioral responses depending on its presentation. This highlights a limitation of single-context safety evaluations and motivates assessing LLM interactions across diverse persuasive and contextual framings.

## 6.2 Social Propagation and System-Level Risk

MSIAN analysis suggests that manipulation effects may extend beyond direct human-AI interactions through exposure to other users’ responses. Participants exhibiting greater immediate vulnerability also showed stronger downstream influence, as reflected by the positive association between BVI and CIC. This indicates that behavioral effects originating from individual interactions may propagate through socially connected environments.

Such propagation is particularly relevant to settings where AI-mediated content is shared or discussed, including online communities and collaborative platforms. Behavioral safety evaluations should therefore consider not only isolated user model interactions but also potential secondary influence across groups.

## 6.3 Design and Governance Implications

Our results motivate extending LLM safety evaluation beyond model outputs to include measurable downstream behavioral outcomes. A response may appear acceptable under conventional contentlevel evaluation while still influencing user trust, compliance, or decision-making through subtle linguistic framing.

Several design directions follow from these findings. Behavioral safety mechanisms could identify potentially manipulative cues, such as excessive authority or urgency, while adaptive interfaces could reduce persuasive intensity when signs of over-trust or uncritical compliance are observed. More broadly, metrics such as BVI, TRR, CIC, and CVS could complement existing model-level safety evaluations by capturing individual, temporal, social, and contextual dimensions of manipulation risk.

These findings also have implications for AI governance. Existing transparency and risk-assessment frameworks could incorporate behavioral influence alongside concerns such as model reliability and harmful content. In particular, the observed persistence and social propagation of manipulation effects highlight the importance of considering downstream behavioral consequences when assessing risks associated with large-scale deployment.

## 6.4 Limitations and Future Work

This study has several limitations. First, although the participant sample spans multiple demographic and disciplinary backgrounds, it does not fully capture the cross-cultural variability that may influence susceptibility to authority, social proof, or emotional framing. Second, the controlled text-based setting limits ecological validity compared with real-world LLM interactions involving extended conversations, user-driven contexts, and multimodal signals.

Although the controlled design helps isolate the effects of linguistic manipulation, it excludes additional cues such as speech prosody, visual authority signals, and richer social interactions. Furthermore, behavioral persistence was evaluated over a one-week period, leaving longer-term effects unexplored.

Future work will extend the framework to cross-cultural and multimodal settings, incorporate behavioral and physiological signals such as gaze and electrodermal activity, and investigate mitigation strategies for reducing manipulation-induced behavioral effects. The use of behavioral and contextual representations in MGD, MSIAN, THPM, and DCRA may also support extensions to additional modalities and longer temporal horizons.

## 7 Conclusion

We introduced a multi-layer behavioral framework for evaluating how manipulative LLM outputs influence human trust, compliance, and decision-making across individual, social, temporal, and contextual dimensions. In a controlled study involving 120 participants, manipulative framing produced significant behavioral effects, with authority-based cues yielding the highest vulnerability and slowest recovery. MSIAN analysis identified potential propagation of these effects through user interactions, while THPM and DCRA characterized their temporal persistence and sensitivity to contextual reframing. Together, these findings demonstrate that manipulative model outputs are associated with measurable behavioral consequences beyond the characteristics of generated content alone.

Our findings motivate extending LLM safety evaluation from content-level assessment toward behavior-level auditing. BVI, CIC, TRR, and CVS provide complementary measures of immediate susceptibility, social propagation, recovery, and contextual sensitivity. Future work will investigate cross-cultural variability, multimodal behavioral signals, longer-term interactions, and mitigation strategies for reducing manipulation-induced behavioral effects. Incorporating behavioral outcomes into foundation-model evaluation may help support user autonomy as LLMs increasingly mediate human decision-making.

## References

Nafees Ahmad, Ho fung Leung, Muhammad Adil Abid, and Sadia Shakil. An effective, reliable, and robust framework for human activity recognition using wearable sensors, 2026. URL https: //arxiv.org/abs/2609.36848.

Sadia Asif and Mohammad Mohammadi Amiri. Refusalguard: Geometry-preserving fine-tuning for safety in LLMs. In Third Conference on Language Modeling, 2026. URL https://openreview. net/forum?id=GaLo7kBiSy.

Sadia Asif, Israel Antonio Rosales Laguan, Haris Khan, Shumaila Asif, and Muneeb Asif. Darkpatterns-llm: A multi-layer benchmark for detecting manipulative and harmful ai behavior. arXiv preprint arXiv:2512.22470, 2025.

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, et al. On the opportunities and risks of foundation models. 2021. URL https://arxiv.org/abs/2108.07258.

Roberto Campbell, Momin Abbas, Muneeza Azmat, Michal Ulewicz, Raya Horesh, Kristjan Greenewald, Rogério Abreu de Paula, and Nathalie Baracaldo. An efficient and modular framework for targeted harm mitigation in llms. arXiv preprint arXiv:2609.13624, 2026.

Robert B. Cialdini. Influence: Science and Practice. Allyn and Bacon, Boston, MA, 4th edition, 2001.

I. Cleland, L. Nugent, F. Cruciani, and C. Nugent. Leveraging large language models for activity recognition in smart environments. In Proceedings ofthe 2024 International Conference on Activity and Behavior Computing, pages 1–8, 2024.

B. J. Fogg. Persuasive Technology: Using Computers to Change What We Think and Do. Morgan Kaufmann, San Francisco, CA, 2003.

Colin M. Gray, Yubo Kou, Bryan Battles, Joseph Hoggatt, and Austin L. Toombs. The dark (patterns) side of ux design. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2018.

D. Hamdhana, K. Harada, H. Oshita, S. Sakashita, and S. Inoue. Toward extracting care records from transcriptions of visiting nurses using large language models. International Journal ofActivity and Behavior Computing, 2024(2):1–17, 2024.

Shruthi K. Hiremath and Thomas Plötz. On the role of context length for feature extraction and sequence modeling in human activity recognition. In Proceedings of the 2021 International Symposium on Wearable Computers (ISWC), 2021. doi: 10.1145/3460421.3478825.

Muhammad Zeeshan Karamat and Christiana Chamon Garcia. How fragile is on-device language model safety? localizing safety-critical parameters for sparse fault analysis. arXiv preprint arXiv:2610.09000, 2026.

Haris Khan, Sadia Asif, and Shumaila Asif. Multi-agent framework for controllable and protected generative content creation: Addressing copyright and provenance in ai-generated media. In 2025 IEEE International Conference on Data Mining Workshops (ICDMW), pages 2400–2403. IEEE, 2025.

Arunesh Mathur, Gunes Acar, Michael J. Friedman, Elena Lucherini, Jonathan Mayer, Marshini Chetty, and Arvind Narayanan. Dark patterns at scale: Findings from a crawl of 11k shopping websites. Proceedings ofthe ACM on Human-Computer Interaction, 3(CSCW):1–32, 2019.

Harri Oinas-Kukkonen and Marja Harjumaa. A systematic framework for designing and evaluating persuasive systems. In Proceedings of the 3rd International Conference on Persuasive Technology, pages 164–176, 2008.

Harri Oinas-Kukkonen and Marja Harjumaa. Persuasive systems design: Key issues, process model, and system features. Communications ofthe Associationfor Information Systems, 24:485–500, 2009.

Thomas Plötz, Nils Y. Hammerla, and Patrick Olivier. Feature learning for activity recognition in ubiquitous computing. In Proceedings of the 22nd International Joint Conference on Artificial Intelligence, 2011.

Parinda Rahman and Ifeoma Adaji. Evaluating user perceptions of trust in persuasive technology: A comparative study. In Proceedings of the 2025 International Conference on Information Technology for Social Good, GoodIT ’25, page 122–130, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400720895. doi: 10.1145/3748699.3749782. URL https://doi.org/10.1145/3748699.3749782.

A. R. Syulistyo, Y. Tanaka, and H. Tamukoh. Recognizing nursing activities in endotracheal suction: Utilizing multiple readouts reservoir computing and large language models. International Journal ofActivity and Behavior Computing, 2024(2):1–22, 2024.

Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geoffrey Irving, and Iason Gabriel. Ethical and social risks of harm from language models, 2021. URL https://arxiv.org/abs/2112.04359.

## Appendix

## A Additional Experimental Details

## A.1 Participant Demographics and Recruitment

A total of 120 participants (N = 120) were recruited through academic mailing lists and professional online communities. Participants ranged in age from 19 to 54 years $( \mathrm { m e a n } = 2 8 . 6 , \mathrm { S D } = 6 . 3 )$ . The sample comprised 52% female, 47% male, and 1% non-binary participants. Educational backgrounds included computer science (34%), social sciences (29%), healthcare (21%), and other disciplines (16%).

All participants reported prior experience with conversational AI systems, such as ChatGPT or Claude, indicating baseline familiarity with LLM-mediated interactions. Participants were unaware of the experimental manipulation objectives before participation.

## A.2 Experimental Conditions and Protocol

Each participant completed twelve human AI interaction tasks involving manipulative and neutral conditions. Four manipulation categories were evaluated:

1. Authority Framing (A): Responses presented as authoritative guidance from experts or officials.

2. Emotional Urgency (E): Responses emphasizing scarcity, fear, or time-sensitive decision pressure.

3. Social Proof (S): Responses suggesting consensus or majority endorsement.

4. Prosocial Altruism (P): Responses appealing to moral or communal benefit.

A Baseline Control (C) condition used semantically comparable neutral responses without explicit manipulative framing.

Manipulative stimuli were generated using MGD, as described in Section 3. Three manipulationintensity levels were defined using M<sub>c</sub>:

• Low: $M _ { c } < 0 . 3 3 ;$

• Medium: $0 . 3 3 \leq M _ { c } < 0 . 6 6 ;$ and

• High: $M _ { c } \geq 0 . 6 6 .$

Experimental conditions and intensity levels were counterbalanced across participants using a Latinsquare design to reduce potential order effects.

## A.3 Behavioral Data Acquisition

Participants interacted with AI-generated responses through a standardized web interface. The following behavioral measures were recorded for each interaction:

• Decision outcome (binary): Whether the participant complied with or rejected the AI’s suggestion.

• Response latency (continuous): Time elapsed between presentation of the AI response and the participant’s final decision.

• Confidence rating (ordinal): Self-reported decision confidence measured on a 7-point Likert scale.

• Emotional state (categorical): Post-interaction affect derived from sentiment analysis of participants’ open-ended reflections.

• Trust calibration (continuous): Change in self-reported trust in AI systems before and after the interaction.

The interface used randomized assignment of manipulation conditions and anonymized participant identifiers. Each experimental session lasted approximately 40 minutes, and interaction logs were timestamped for offline analysis.

## A.4 Behavioral Graph Construction

To evaluate cross-participant influence, post-task discussion interactions were simulated using anonymized excerpts of responses from prior participants. Participants reviewed and commented on selected decision narratives, creating patterns of observational exposure.

These interactions defined a directed behavioral graph $G = ( V , E )$ , where each participant $u _ { i }$ corresponds to a node and a directed edge $e _ { j i }$ represents exposure of participant $u _ { i }$ to participant $u _ { j } \mathrm { ' s }$ decision narrative. The resulting graph was provided as input to MSIAN to model secondary influence propagation and collective susceptibility.

## A.5 Evaluation Metrics

Behavioral responses were evaluated using the four metrics introduced in Section 3:

• BVI: Measures immediate susceptibility to manipulative AI responses.

• CIC: Quantifies behavioral influence among connected participants.

• TRR: Characterizes the rate at which manipulation-induced behavioral changes return toward baseline.

• CVS: Measures variation in behavioral susceptibility across contextual framings.

All metrics were normalized to [0, 1] to facilitate comparisons across participants and experimental conditions.

## A.6 Statistical Analysis

Statistical inference was conducted using repeated-measures ANOVA. Greenhouse Geisser correction was applied when sphericity assumptions were violated, and Bonferroni correction was used for post hoc pairwise comparisons to control the family-wise error rate. Effect sizes were reported using partial $\eta ^ { 2 }$

Pearson’s r was used to examine correlations between manipulation intensity $( M _ { c } )$ and behavioral measures, including BVI and TRR. Regression models were fitted to examine the predictive contributions of manipulation conditions and participant characteristics. Predictive performance was evaluated using 10-fold cross-validation. Unless otherwise specified, statistical significance was defined as $p < 0 . 0 5$

## A.7 Ethics and Reproducibility

All participants provided informed consent before participation and were fully debriefed following the experiment. Experimental procedures followed institutionally approved ethical guidelines for human-subject behavioral research.

Participant identifiers were anonymized, and raw conversational data containing personally identifiable information will not be publicly released. Experimental prompts, manipulation templates, model configurations, scripts, metric definitions, and anonymized behavioral measurements will be released with the archival version of the paper to facilitate independent replication while preserving participant privacy.

## B Additional Results

## B.1 Behavioral Vulnerability Across Manipulation Conditions

Table 1 reports BVI values across manipulation categories and intensity levels.

Table 1: BVI across manipulation categories and intensity levels.
<table><tr><td>Condition</td><td>Low</td><td>Medium</td><td>High</td><td>Mean</td></tr><tr><td>Authority (A)</td><td>0.41</td><td>0.58</td><td>0.74</td><td>0.58</td></tr><tr><td>Emotional (E)</td><td>0.37</td><td>0.53</td><td>0.69</td><td>0.53</td></tr><tr><td>Social Proof (S)</td><td>0.33</td><td>0.49</td><td>0.65</td><td>0.49</td></tr><tr><td>Prosocial (P)</td><td>0.29</td><td>0.46</td><td>0.63</td><td>0.46</td></tr><tr><td>Control (C)</td><td>0.22</td><td>一</td><td>一</td><td>0.22</td></tr></table>

## B.2 Behavioral Influence Network

Figure 4 visualizes the behavioral influence graph modeled by MSIAN. Node size represents influence degree, while node intensity indicates manipulation exposure.

![](images/332552715efb64c8350581307d4e9e25da57215ebbf48e1510dd73e80c2ea1cc.jpg)  
Figure 4: Behavioral influence graph modeled using MSIAN. Node size represents influence degree, while node intensity indicates manipulation exposure.

## B.3 Temporal Recovery and Persistence

Table 2 reports TRR and BPI values across experimental conditions.

Table 2: TRR and BPI across manipulation conditions.
<table><tr><td>Condition</td><td>TRR</td><td>BPI (T2)</td></tr><tr><td>Authority</td><td>0.42</td><td>0.61</td></tr><tr><td>Emotional</td><td>0.48</td><td>0.57</td></tr><tr><td>Social Proof</td><td>0.53</td><td>0.49</td></tr><tr><td>Prosocial</td><td>0.55</td><td>0.47</td></tr><tr><td>Control</td><td>0.68</td><td>0.32</td></tr></table>

Lower TRR values indicate slower recovery from manipulation-induced behavioral changes. Authority framing exhibited the lowest TRR and highest BPI at the one-week follow-up, whereas the control condition showed the highest TRR and lowest BPI.