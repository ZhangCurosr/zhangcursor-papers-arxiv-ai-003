# When Do Humans and Machines Truly Complement Each Other? Formal Conditions and Empirical Evidence for Human-AI Synergy

Andrea Campagner, Federico Cabitza, Vincenzo Crupi

Abstract—Can humans and artificial intelligence cooperate such that their combination yields greater accuracy than either agent alone? This result, termed synergy or complementarity, is notoriously difficult to achieve, even when both agents possess significant independent accuracy. In this paper, we present a formal framework to model human-AI interaction through the lens of reliance patterns, which characterize the joint correctness structure of human and algorithmic decisions. We derive mathematically necessary and sufficient conditions for synergy, demonstrating that its emergence depends on the ability of either agent to identify and correct the errors of the other, what we call Human and Algorithmic Correction. Beyond its theoretical contribution, the proposed framework provides a foundation for the design and evaluation of knowledge-based decision support systems, where interaction policies must dynamically integrate heterogeneous sources of expertise. We demonstrate this through the analysis of seven clinical case studies. Our empirical results reveal a “dominance trap,” where a strong algorithmic advantage can paradoxically inhibit synergy by eroding human corrective intervention. We also show that synergy can persist even in the presence of cognitive biases, such as automation bias, provided there is functional de-correlation between human and AI errors. These results provide a formal foundation for key concepts in the human-AI systems literature and offer actionable insights for designing interaction-centric AI systems that foster complementarity.

Index Terms—Human-AI Synergy, Human-AI Complementarity, Human-AI Interaction, Reliance Patterns, Metacognition, Dominance Trap

## I. INTRODUCTION

RTIFICIAL Intelligence is being increasingly advocated and employed as a tool to support judgment and decision making in an impressive variety of domains. The breakthrough of contemporary AI has revitalized and reshaped two longstanding inquiries: first, how human judgment compares with algorithmic outcomes across diverse contexts; and second, under what conditions the integration of these two assessments can reliably yield superior accuracy.

Current theoretical understanding appears to be rather strong for the first issue, but still limited for the second [1, 2]. Crucially, a recent comprehensive review [3] notes that we still lack a broad conceptual understanding of when human-AI teams outperform their individual components (“when a combination of humans and AI are better than either alone”). Consequently, the specific conditions under which human-AI systems can exhibit synergy [4] (or complementarity [2]) are yet to be fully elucidated, especially as this condition often appears elusive in practical scenarios [5].

From a philosophy-of-technology perspective, recent work [6] has argued that the explanatory gap surrounding human–AI synergy is partly conceptual: synergy is not merely a standalone, post hoc indicator of predictive performance, but rather a form of defeasible evidence that a given human–AI interaction constitutes a reliable epistemic process. Complementing this view, recent contributions in the human–machine systems literature [5] emphasize that synergy is not merely an outcome, but a design-dependent property emerging from the interaction between human and algorithmic components: differences in information, capabilities, and error structures between humans and AI—-rather than overall accuracy alone—-constitute the primary sources of complementarity and must be explicitly leveraged through appropriate interaction mechanisms. Taken together, these perspectives suggest that understanding synergy requires moving beyond purely outcome-based definitions toward a unified account that integrates interaction design and the structure of human–AI error complementarities.

From a more technical perspective, the scientific community has only recently begun to address this issue by investigating theoretical formulations of synergy, and three main strands of research can be identified, which, however, have been relatively isolated from each other. Most of the work in this setting has been developed in the theoretical machine learning literature and has focused on the study and formal analysis of the deferral paradigm, that is, developing AI systems that can defer their judgment to a human. One of the seminal contributions in this field is the work of Madras et al. [7] who proposed and first analyzed the effects of deferral on human-AI collaboration relying on the framework of selective prediction in machine learning literature. Developing this contribution, Charusaie et al. [8] studied the theoretical properties of deferral learning systems trained to be complementary to humans; Straitouri et al. [9] similarly considered how synergy could be achieved through conformal prediction (a family of statistical techniques to create deferral AI systems); De Toni et al. [10] studied the computational complexity of achieving complementary through deferral, showing that in general the problem is computationally infeasible. Although this body of work characterizes conditions for synergy, it remains largely inapplicable to traditional AI systems that provide judgments without deferral. Furthermore, these studies emphasize the algorithmic perspective, leaving the more interactional dimensions of the collaboration relatively unexplored. Crucially, the deferral paradigm focuses on a logic of delegation, where the final judgment is produced by only one of the two agents.

In contrast, our work addresses judgment integration (or coagency), where both human and AI assessments are available and must be combined.

A different stream of work, more focused on the interactional aspects, has aimed at explicitly studying combination mechanisms to model the interaction between AI and human agents. In this light, the Bayesian paradigm, which models both human and AI reasoning as well as their combination as parametric probabilistic models and employs empirical interaction data to fit their parameters, has been dominant, partly due to the already rich literature on probabilistic models of human cognition [11]. Steyvers et al. [12] proposed a Bayesian framework to model and combine the predictions of human and AI agents; Westby et al. [13] studied a Bayesian theory of mind framework for modeling collective intelligence (as a special form of synergy) in human-AI teams. While these studies include a representation of the cognitive processes underlying human judgment, the framework we adopt in this work does not involve such commitment. This approach allows us to directly address the constructs relevant to human-AI interaction as they emerge from formal probabilistic constraints. An additional limitation of Bayesian approaches is their focus on modeling joint performance rather than identifying the foundational conditions for synergy; consequently, the requirements for synergy emerge only as indirect byproducts of their parametric fits, rather than as explicit theoretical constraints. Moreover, they often rest on strong assumptions (e.g., normality, or specific constraints on combination functions), which limit their ability to yield actionable insights, particularly when such assumptions are violated.

To address these gaps and more in line with the approach we take in this paper, a recent strand of work in the study of human-AI interaction mechanisms has aimed at characterizing conditions under which synergy can (or cannot) emerge by adopting a more general (model-free) probabilistic framework. Donahue et al. [14] studied equivalent definitions of synergy when the combination of AI and human agents is realized by linear combination rules, also providing characterizations of cases in which synergy is impossible to achieve. The study analyzed synergy solely within the constraints of linear combination rules, which assume fixed weights for each agent. Such models may fail to capture the non-linear and contextdependent interactions typical of work practices, where a human expert evaluates the quality of an AI signal on a caseby-case basis. More recently, Peng et al. [15] studied nofree lunch theorems for human-AI synergy, showing that it is generally impossible to achieve synergy if the combination rule only takes into account the overall accuracy of the human and AI agents. By shifting the focus from global performance metrics to the granular dynamics of reliance patterns (specifically Human Correction and Algorithmic Correction, as defined below), we demonstrate that synergy is not only theoretically possible but also empirically predictable when the interaction logic exploits and fosters the mutual reduction of errors on a case-by-case basis.

The goal of this contribution is to advance insight on these matters: to do so, we adopt a probabilistic approach, similar to those previously considered in [14, 15], which aims at addressing the limitations of previous analytical studies on human-AI synergy. In contrast to prior research, our framework avoids parametric assumptions regarding both how agents formulate decisions and how their judgments are combined, and instead adopts a non-parametric and model-free approach.

Thus, we make two main contributions.

• First, we propose a general framework to formally define the notion of human-AI collaboration and human-AI synergy. Our framework rests on the central notion of reliance patterns [16], which have been introduced in the human-AI interaction community to model and quantify issues related to AI appropriation.

• Second, we provide different sets of formal conditions for human-AI synergy, discussing their sufficiency and necessity. We also provide empirical illustrations of these conditions in practical settings (drawn from real-world medical scenarios of AI use).

Beyond theoretical characterization, our framework provides a set of design constraints and principles for human-AI interaction policies, enabling the engineering of synergistic systems. The conditions derived in this work can be operationalized as design objectives, evaluation metrics, and training targets for hybrid systems and we provide a discussion of the implications of our results for the design of hybrid human-AI systems.

## II. THE FORMAL FRAMEWORK

In this section, we introduce a formal framework for modeling human–AI interaction. This framework is intentionally model-agnostic and non-parametric: rather than assuming specific cognitive or algorithmic processes, the framework captures interaction in terms of observable variables—namely, the correctness of human judgments, AI outputs, and final decisions. This abstraction enables the framework to be applied across different domains and system implementations, supporting both theoretical analysis and empirical evaluation.

Formally, let $X \in \{ x , \lnot x \}$ be a binary variable representing the ground-truth state of a situation of interest S. For instance, x may denote the presence of a specific pathology in a patient’s tissue (where the patient case constitutes S), while $\neg x$ denotes its absence. A human user (e.g., a clinician) typically assesses the validity of x against ¬x based on a complex and often heuristic array of information indicators. Let $H \in \{ h , \neg h \}$ include the possible outcomes of such human expert’s judgment process about X. For simplicity, we write $H = X$ to mean that either $( x \wedge h )$ or $( \neg x \land \neg h )$ holds. Consider also an artificial system designed to similarly make a decision about the situation of interest (e.g., in our recurring example about clinical-decision making, detecting specific pathologies) by processing fine-grained data extracted from each case. For our current purposes, we can think of an algorithm devised through machine learning to deliver an outcome in the form of a further binary variable $A \in \{ a , \neg a \}$

Finally, when both H and A are available, a system (or human agent) integrates them into an overall decision $F \in$ $\{ f , \lnot f \}$ . We model F as depending on the values of H and A according to some (situation-dependent) combination function:

$$
c _ { S } : H \times A  F .
$$

We note that, compared with previous work [14, 15], we do not make any assumption that the combination rule is solely determined by the accuracy of the human and AI agents: rather, as a consequence of the dependence of the combination rule c on the situation of interest $S ,$ c could, in principle, take into account context information that also depends on the case at hand<sup>1</sup> (as implicitly considered, e.g., in [16]).

These variables define a set of 16 mutually exclusive and jointly exhaustive elementary possibilities and a related algebra on which a probability distribution Pr can be defined, as shown in the following section.

## A. Reliance Patterns

Based on the agreement between X and H, A, F, respectively, we can define so-called reliance patterns, which enumerate the 8 possible truth outcomes [16]. In detail, consider three additional variables, namely, the correctness of the human, the AI, and the combined system denoted as UJ (Unsupported human Judgment), AI (AI judgement), and SD (Supported final Decision), respectively. These variables are binary (0 or 1) and their values encode the correctness (i.e., agreement with X) of each of H, A and F. The 8 possible configurations are enumerated as binary reliance patterns, as detailed in Table I. We will also use the symbol ∗ as a shorthand for enumeration of patterns (e.g. ‘\*\*1’ corresponds to the configurations 001, 011, 101, 111).

While, obviously, knowledge of $X , H , A , F$ is sufficient to determine UJ, AI, SD, the converse does not hold. At the same time, one of X, H, A, F, together with UJ, AI, SD is sufficient to determine the remaining variables. Consequently, these formulations are functionally equivalent; hereafter, we use either notation interchangeably to facilitate ease of exposition.

From a systems perspective, all the introduced variables and reliance patterns correspond to measurable quantities in human–AI experiments or deployments. For example, H, A, and F can be directly observed in decision-support workflows, while X corresponds to ground truth obtained through validation or expert consensus, which would then enable the identification of reliance patterns and the quantification of their frequency. This allows the proposed framework to serve not only as a theoretical model, but also as a basis for the instrumentation and evaluation of human–AI systems, as we will illustrate in Section VI.

## III. THE DEFAULT SETTING

In this section, we introduce a set of baseline assumptions (that jointly define what we call the default setting) and definitions aimed at characterizing conditions under which a human–AI system can achieve a reasonable level of accuracy, independent of whether true synergy is attained. These assumptions are widely adopted, either explicitly or implicitly, across the literature on human–AI decision making and machine learning. As such, they provide a standard reference setting for analyzing the behavior of hybrid systems. In this direction, the notation described in the previous sections is sufficient to define three key notions concerning accuracy of human-AI hybrid systems:

Algorithmic Advantage (A-Adv): $\operatorname* { P r } ( A = X ) > \operatorname* { P r } ( H =$ $X ) ( i . e . , \operatorname* { P r } ( * 1 * ) > \operatorname* { P r } ( 1 * * ) )$

Human Advantage (H-Adv): $\operatorname* { P r } ( H = X ) > \operatorname* { P r } ( A = X )$ $( i . e . , \operatorname* { P r } ( 1 * * ) > \operatorname* { P r } ( * { 1 } * ) ) .$

Synergy:

$$
\begin{array} { r l } { S I ) } & { \operatorname* { P r } ( F = X ) > \operatorname* { P r } ( A = X ) \ ( i . e . , \ \operatorname* { P r } ( * * 1 ) > \operatorname* { P r } ( * 1 * ) ) , } \\ { S 2 ) } & { \operatorname* { P r } ( F = X ) > \operatorname* { P r } ( H = X ) \ ( i . e . , \ \operatorname* { P r } ( * * 1 ) > \operatorname* { P r } ( 1 * * ) ) . } \end{array}
$$

Intuitively, Algorithmic Advantage formalizes the frequent empirical observation that AI systems can outperform humans in decision-making tasks. Similarly, Human Advantage formalizes the opposite situation in which humans outperform AI systems, which is still typical for particularly complex tasks that elude pattern recognition capabilities.

Furthermore, Synergy, as already described intuitively in the Introduction, formalizes the primary objective of human-AI interaction: the emergence of a hybrid system more accurate than either agent in isolation [5]. It is easy to see that condition S2 corresponds to a form of Human Augmentation, by which the interaction between the human and AI agents enhables the system to reach a performance superior to that of the human alone. By symmetry, condition S1 represents a form AI Augmentation, that is the requirement that the hybrid human-AI system outperforms its purely algorithmic component.

In relevant applications, significant features of the probability distribution Pr as referred to a specific domain and target case(s) will have to be constrained by reliable empirical input. In our theoretical analysis in this section, however, we will only put forward a small set of general assumptions about Pr.

## Conditional Independence (CI): $H \bot _ { \mathrm { P r } } A | X .$

This assumption is implicitly given by the graphical model in [16] and corresponds to an idealized setting in which neither the human nor the algorithm has access to the other’s output at the time of producing H and A, respectively. Although this may not hold in many real deployments, it provides a tractable and realistic baseline.

Human Reliability (HR): $0 . 5 < \operatorname* { P r } ( h | x ) , \operatorname* { P r } ( \lnot h | \lnot x ) < 1 .$

HR characterizes human judgment as valuable, although imperfect. It implies:

$$
\operatorname* { P r } ( h \land x ) > \operatorname* { P r } ( \lnot h \land x ) > 0 , \quad \operatorname* { P r } ( \lnot h \land \lnot x ) > \operatorname* { P r } ( h \land \lnot x ) > 0 ,
$$

and, denoting the correctness condition as $H = X \colon$

$$
0 . 5 < \operatorname* { P r } ( H = X ) = \operatorname* { P r } ( 1 * * ) < 1 .
$$

TABLE I  
RELIANCE PATTERNS: DEFINITION OF ALL POSSIBLE DECISIONAL CONFIGURATIONS BETWEEN HUMAN DECISION MAKERS AND AI SYSTEM. IN THE FIRST THREE COLUMNS, 0 DENOTES AN INCORRECT DECISION, AND 1 A CORRECT DECISION. WE ASSOCIATE THE ATTITUDE TOWARDS THE AI IN EACH POSSIBLE RELIANCE PATTERN, WHICH LEADS TO EITHER ACCEPTING OR DISCARDING THE AI’S ADVICE, AND THE MAIN RELATED COGNITIVE BIASES.
<table><tr><td>Human judgment (UJ)</td><td>AI support (AI)</td><td>Final decision (SD)</td><td>Reliance pattern</td><td>Other formulations</td><td>Facilitating Biases</td></tr><tr><td>wrong (0)</td><td>wrong (0)</td><td>wrong (0)</td><td>detrimental reliance</td><td></td><td>automation complacency</td></tr><tr><td>wrong (0)</td><td>wrong (0)</td><td>right (1)</td><td>beneficial under-reliance</td><td></td><td>extreme algorithmic aversion</td></tr><tr><td>wrong (0)</td><td>right (1)</td><td>wrong (0)</td><td>detrimental self-reliance</td><td>Detrimental overriding</td><td>self-anchoring bias</td></tr><tr><td>wrong (0)</td><td>right (1)</td><td>right (1)</td><td>beneficial over-reliance</td><td>Correct adherence</td><td>algorithm appreciation</td></tr><tr><td>right (1)</td><td>wrong (0)</td><td>wrong (0)</td><td>detrimental over-reliance</td><td>Detrimental adherence</td><td>automation bias</td></tr><tr><td>right (1)</td><td>wrong (0)</td><td>right (1)</td><td>beneficial self-reliance</td><td>Corrective overriding</td><td>algorithmic aversion</td></tr><tr><td>right (1)</td><td>right (1)</td><td>wrong (0)</td><td>detrimental under-reliance</td><td></td><td>extreme algorithmic aversion</td></tr><tr><td>right (1)</td><td>right (1)</td><td>right (1)</td><td>beneficial reliance</td><td></td><td>confirmation bias (in later cases)</td></tr></table>

Algorithmic Reliability (AR): $0 . 5 < \mathrm { P r } ( a | x ) , \mathrm { P r } ( \neg a | \neg x ) <$ 1.

This is the analog of HR for the AI system, implying:

$$
0 . 5 < \operatorname* { P r } ( A = X ) = \operatorname* { P r } ( * 1 * ) < 1 .
$$

While the above three assumptions are considered standard in the human-AI interaction literature, other, stronger, requirements have been posited. As an example, the following requirement originally proposed by Meehl [17], can be easily seen to be a stronger version of Algorithmic Advantage:

$$
\begin{array} { r l } & { \mathbf { M e e h l ^ { \prime } s _ { \lambda } P o s t u l a t e _ { \lambda } ( M P ) } \colon \operatorname* { P r } ( x \wedge a ) > \operatorname* { P r } ( x \wedge h ) , \quad \operatorname* { P r } ( \neg x \wedge h ) } \\ & { \neg a ) > \operatorname* { P r } ( \neg x \wedge \neg h ) . } \end{array}
$$

Meehl’s Postulate goes beyond Algorithmic Advantage by asking not merely that the AI is more accurate (on average) than the human, but rather that the algorithm outperforms the human in both detecting and rejecting the disease. In the same contribution, Meehl also discussed the following assumption:

Broken Leg (BL): $\mathrm { P r } ( x \Lambda \neg a \land h ) > 0 , \quad \mathrm { P r } ( \neg x \land a \land \neg h ) > 0 .$

These highlight relatively uncommon “broken $\mathrm { l e g } ^ { \prime }$ cases [17], in which human judgment correctly identifies algorithmic errors, and can be expressed equivalently as:

$$
\mathrm { P r } ( 1 0 * ) > 0 , \quad \mathrm { P r } ( 0 1 * ) > 0 .
$$

## A. Combination rules and baseline performance

We now examine simple rules that may be adopted to combine human and algorithmic judgments, and contrast their accuracy with each component in isolation. The first such rule we examine corresponds actually to a family of rules satisfying a common structural constraint requiring that their output may differ only in cases of disagreement between human and AI system. As we will see in the following, this requirement on combination rules will be crucial in the definition of conditions that allow the emergence of human-AI synergy.

Agreement (Agr): If A = H, then F = A = H.

Equivalently, Agreement can be expressed as the structural constraint that $\operatorname* { P r } ( 0 0 1 ) = \operatorname* { P r } ( 1 1 0 ) = 0 .$ , as the involved cases are those in which $A , H$ coincide, but F differs from both of them. A simple consequence of adopting a function c that uniformly satisfies agreement is the following:

Proposition 1. Assume the default setting. If c satisfies Agr then $\operatorname* { P r } ( F = X ) > 0 . 2 5 .$

While agreement alone is insufficient to guarantee $\mathrm { P r } ( F =$ $X ) > 0 . 5$ , each of the following uniform combination rules can do so under our current assumptions:<sup>2</sup>

Algorithmic Aversion $( \mathbf { A } \mathbf { A } ) \colon F = f \ i f f \ H = h .$

Full Automation (FA): $F = f \ i f f \ A = a .$

Diagnostic Skepticism (DS): $F = f \ i f f H = h \wedge A = a .$

Diagnostic Alertness (DA): $F = f \ i f f H = h \lor A = a .$

The first two rules represent extreme reliance strategies: total dependence on either the human expert or the algorithm. DS mandates consensus between both sources to confirm a decision, whereas DA treats either source’s positive assessment as sufficient. Under the default setting and MP, FA is expected to outperform AA in terms of accuracy. DS may outperform AA when prevalence is low (reducing false alarms), while DA may outperform FA when prevalence is high. These simple rules show how distributional features affect the relative performance of combination strategies, and how Synergy requires more structured approaches than mere automation.

## IV. BEYOND AUTOMATION: A CONDITION FORSYNERGISTIC INTERACTION

While the conditions studied in the previous section are sufficient to ensure that simple combination rules achieve nontrivial levels of accuracy, they do not, in general, guarantee that the resulting human–AI system outperforms its individual components. In this section, to address this gap, we move beyond this default setting to identify specific conditions under which synergy can emerge. In particular, we shift the focus from general accuracy properties to the structure of disagreements and error correction mechanisms between human and AI agents. This perspective reveals that synergy is not a direct consequence of individual accuracy, but rather an emergent property of how conflicting judgments, that are, the cases in which a combination rule satisfying Agreement can intervene, are resolved within the interaction process. The core premise is that synergy emerges only when disagreements are predominantly resolved in favor of the correct actor, regardless of which source it is. Inspired by this observation, we introduce the two following definitions:

Human Correction (HC): $\operatorname* { P r } ( A \neq X | F = H \neq A ) > { \frac { 1 } { 2 } } .$ Algorithmic Correction (AC): $\operatorname* { P r } ( H \neq X | F = A \neq H ) >$ ${ \frac { 1 } { 2 } } .$

The two assumptions can be equivalently expressed as:

$$
\begin{array} { r } { P r \mathrm { ( 1 0 1 ) } > P r \mathrm { ( 0 1 0 ) } \quad [ H C ] } \\ { P r \mathrm { ( 0 1 1 ) } > P r \mathrm { ( 1 0 0 ) } \quad [ A C ] } \end{array}
$$

Intuitively, HC means that whenever the final decision F follows the human judgment against the algorithm, it is more likely that the algorithm is wrong than right. Intuitively, this condition captures the idea that the human is able to identify “broken $\mathrm { l e g } ^ { \prime }$ cases more often than fabricating exceptions. Symmetrically, AC means that when the final decision F follows the algorithm, it is able to correct human errors. Thus, both assumptions can be understood as formalizing the requirement that the undesirable situation in which one of the two agents drags down the performance of the entire system [5] are sufficiently rare. Given these assumption, we now prove our main results concerning Synergy.

Theorem 1. Assuming Human and Algorithmic Correction, the Agreement property becomes a sufficient condition for Synergy. Similarly, assuming Agreement, Human and Algorithmic Correction are sufficient for Synergy. Conversely, Synergy and Agreement imply Human and Algorithmic Correction

As already hinted at previously, Theorem 1 provides a key insight for the design of human–AI systems: synergy can emerge only when disagreements between human and AI are resolved in favor of the correct agent. This implies that the effectiveness of such systems depends less on the individual accuracy of their components, and more on the quality of the interaction policy governing how disagreements are handled.

In fact, part of the previous result can be strengthened by introducing a new assumption which relaxes Agreement: rather than requiring that combination rules only really operate in cases of disagreement, we weaken this requirement by only assuming that they ensure that cases of negative algorithmic aversion (i.e., cases of the form 110) are rarer than positive ones (i.e., cases of the form 001):

Beneficial Aversion (BA): $\textstyle P r ( F = X | H = A \neq F ) \geq { \frac { 1 } { 2 } }$

which can equivalently be expressed as:

$$
P r ( 0 0 1 ) \geq P r ( 1 1 0 )
$$

Based on this alternative condition, we can prove a second characterization of Synergy:

Theorem 2. Assume Human Correction and Algorithmic Correction. Then Beneficial Aversion is sufficient for Synergy. The converse part of Theorem 1, however, does not hold.

The previous results illustrate that the ability of the combination rule to arbitrate disagreements in favor of the correct agent, as formalized through the Human and Algorithmic Correction assumptions, is essentially sufficient and necessary for synergy (assuming that the combination rule itself satisfies Agreement). However, these conditions may be difficult to enforce directly in practice. In many real-world settings, by contrast, it is more straightforward to determine whether one agent systematically outperforms the other. The following propositions show that, under such performance asymmetries, the characterization of synergy can be significantly simplified. In particular, when one agent is globally more accurate than the other, synergy reduces to the presence of a single corrective mechanism (Human Correction in the case of Algorithmic Advantage, and Algorithmic Correction in the case of Human Advantage) while the other automatically emerges.

Proposition 2 (HC ⇔ Synergy under A-Adv and Agreement). Assuming Algorithmic Advantage and Agreement, Human Correction is necessary and sufficient for Synergy. It is sufficient also if Agreement is replaced by Beneficial Aversion.

Proposition 3 (AC ⇔ Synergy under H-Adv and Agreement). Assuming Human Advantage and Agreement, Algorithmic Correction is necessary and sufficient for Synergy. It is sufficient also if Agreement is replaced by Beneficial Aversion.

A. Appropriate Reliance and Synergy: an Alternative Characterization

While in the previous section we characterized synergy in terms of probabilistic correction mechanisms, the human–machine systems literature has often approached the problem from a complementary perspective, focusing on observable patterns of user reliance on AI systems and related human factors [18]. In particular, the notion of appropriate reliance [19, 20] (AR) captures the extent to which users follow AI recommendations when they are correct and disregard them when they are incorrect: a behavior widely regarded as a key indicator of effective human–AI interaction. Bridging these perspectives is important for both theoretical and practical reasons. On the one hand, appropriate reliance provides a behavioral and empirically measurable construct, grounded in user interaction data. On the other hand, it offers a potential operational proxy for the more abstract conditions of Human and Algorithmic Correction introduced above. In this section, we therefore investigate how appropriate reliance relates to synergy, and under which additional conditions it can guarantee—or be implied by—the emergence of synergistic human–AI performance.

To this end, we first propose a formalization of the notion of appropriate reliance, inspired by the metrics proposed in [21], in our framework:

Appropriate Reliance (AR):

$$
A R I ) \quad \operatorname* { P r } ( 0 1 1 ) > \operatorname* { P r } ( 0 1 0 ) , \quad A R 2 ) \quad \operatorname* { P r } ( 1 0 1 ) > \operatorname* { P r } ( 1 0 0 ) .
$$

The two conditions, respectively, denote the lack of selfanchoring bias (also called egocentric discounting [22]) (AR1), as the probability to accept a correct AI support is higher than the probability of rejecting it, and the lack of automation bias [23] (AR2), as the probability to discard an incorrect AI support is higher than the probability of accepting it.

Based on the assumption of appropriate reliance, we can prove several alternative characterizations of Synergy.

Theorem 3. AR, BA imply Synergy under one of the following:

1) $\operatorname* { P r } ( 1 0 0 ) = \operatorname* { P r } ( 0 1 0 ) ;$

2) $\operatorname* { P r } ( { 1 0 1 } ) = \operatorname* { P r } ( 0 1 1 ) ;$

3) $\operatorname* { P r } ( 1 0 0 ) > \operatorname* { P r } ( 0 1 0 )$ and Algorithmic Advantage;

4) $\operatorname* { P r } ( 0 1 0 ) > \operatorname* { P r } ( 1 0 0 )$ and Human Advantage;

5) $\operatorname* { P r } ( 0 1 1 ) > \operatorname* { P r } ( 1 0 1 )$ and Human Advantage;

6) $\operatorname* { P r } ( 1 0 1 ) > \operatorname* { P r } ( 0 1 1 )$ and Algorithmic Advantage;

7) Pr(100) > Pr(010) and Pr(101) > Pr(011);

8) $\operatorname* { P r } ( 0 1 0 ) > \operatorname* { P r } ( 1 0 0 )$ and $\operatorname* { P r } ( 0 1 1 ) > \operatorname* { P r } ( 1 0 1 )$

Furthermore, in results 3 and 5 (resp., 4 and 6) above, AR can be weakened to AR2 (resp., AR1). Under Agreement and one of conditions 1, 2, 7, or 8, Synergy implies AR. Furthermore, under condition 7, Synergy can relaxed to S2; under condition 8, Synergy can be relaxed to S1.

The previous result shows that, while in the literature Synergy and Appropriate Reliance are often related to each other, they are not equivalent unless other conditions are satisfied. Interestingly, the result, together with Proposition 3, illustrate conditions for establishing Human (resp., Algorithmic) Correction: ensuring lack of automation bias (resp., self-anchoring bias) and $\mathrm { P r } ( 1 0 0 ) > \mathrm { P r } ( 0 1 0 )$ (resp., $\mathrm { P r } ( 0 1 0 ) > \mathrm { P r } ( 1 0 0 ) )$ , where this latter condition can be seen as a weak form of Human (resp., Algorithmic) Advantage. Nonetheless, understanding which algorithmical or socio-technical interventions can be implemented to ensure such a guarantee is not trivial, and it represents a relevant direction for future research.

## V. COGNITIVE AND METACOGNITIVE CORRELATES OF HUMAN CORRECTION

The notion of Human Correction (HC), introduced in Section 4, captures a key aspect of the human contribution to Synergy: the ability of a human decision-maker to correctly override an algorithmic output when the latter is wrong. Formally, HC is satisfied when, conditional on a disagreement between human and algorithmic assessments, the final decision F follows the human judgment H more often when the AI is wrong, than when it is correct. In other words, HC expresses a systematic capability to detect and correct algorithmic errors.

Beyond its formal definition, this property has a precise cognitive interpretation. HC can be viewed as the behavioral manifestation of a second-order competence, namely metacognitive sensitivity — the capacity to monitor and evaluate one’s own confidence and the reliability of external sources. In the context of human–AI interaction, metacognitive sensitivity supports an adaptive balance between reliance and autonomy: it allows the human agent to infer, from contextual cues and subjective uncertainty, when the algorithm is likely to be wrong and when it can be trusted. This self-monitoring faculty is pivotal to Synergy, empowering humans to provide informed interventions instead of arbitrary rejection.

In the literature on metacognition, such sensitivity is often quantified through Type 2 receiver operating characteristic (Type 2 ROC) analysis, which assesses the correspondence between confidence judgments and actual correctness [24]. A high area under the Type 2 ROC curve indicates that confidence reliably tracks accuracy, implying that the agent can discriminate between correct and incorrect decisions. Translating this into our probabilistic framework, a high metacognitive sensitivity corresponds to a high conditional probability Pr(H = X | high confidence) and, consequently, can increase the likelihood that the HC condition holds in practice, insofar as it enables selective overriding when contextual cues indicate that the algorithm may be wrong. However, we note that HC is a behavioral property of the interaction protocol and cannot be reduced to confidence discrimination alone.

This interpretation also connects HC to the notion of confidence calibration, a classical measure of the alignment between subjective confidence and objective accuracy. Poor calibration can lead either to automation bias, when overconfidence in the AI diminishes corrective intervention, or to self-anchoring bias (also called algorithmic aversion [25] or conservatism bias [16]), when underconfidence prevents beneficial reliance. By contrast, well-calibrated metacognitive monitoring fosters appropriate reliance which, under certain conditions (as shown in Theorem 3), can lead to synergistic effects. Thus, while Algorithmic Correction (AC) depends on the statistical superiority of the model (Algorithmic Advantage), Human Correction (HC) depends on the human’s ability to track and regulate the reliability of both agents in interaction.

This perspective underscores that synergy is not solely a product of informational overlap, but also of the human’s metacognitive competence to exploit that complementarity [12]. This competence is part and parcel of the cognitive capability that has been also denoted as teaming intelligence, that is the kind of intelligence and competence needed to effectively work with AI systems [26], and theory of machine[27, 28], that is a practical understanding of how the AI system behaves, including its strengths, limitations, and typical failure modes (i.e., a form of Theory of Mind directed toward machines). These “theories”, while not necessarily technically accurate, are pragmatically valuable for interpreting the system’s behavior, evaluating the reliability of its outputs, and determining when to rely on or disregard its suggestions [29].

Accordingly, the empirical detection of Synergy should include measures of metacognitive sensitivity (e.g., confidence discrimination, Type 2 ROC area, appropriate reliance score), human-AI agreement (e.g., influence index), and calibration (ECE, ECI), alongside standard accuracy metrics. Integrating these psychometric and behavioral constructs into the probabilistic framework developed in this paper provides a bridge between formal models of judgment combination and cognitive theories of self-monitoring, offering a richer account of when and why genuine Human–AI Synergy can emerge.

## VI. EMPIRICAL ILLUSTRATION: CLINICAL CASE STUDIES

In this section we describe an empirical validation of our theoretical framework through seven real-world case studies, across diverse clinical domains, drawn from the human-AI interaction literature. While the following analysis is not designed as a controlled experimental validation, it provides an empirical testbed to examine whether the theoretical conditions derived in this paper correspond to observable patterns in real-world human–AI interaction data, thus providing an ecological validation of our theoretical results.

All the studies involved the interaction between expert clinicians and (simulated or real) AI systems, adopting a humanfirst interaction protocol by which humans were initially asked to provide a diagnosis without any external support; secondly, an AI-generated support was generated and provided to the humans; finally, based on the their previous judgment and the AI-support, a final decision was made by the humans. This interaction modality enables the collection of reliance pattern data, which were either public or provided to us by the authors of the studies, and hence allows to empirically verify the presence or absence of synergy, as well as of the other conditions described in this work. The characteristics of the studies are as follows:

• Knee MRI Diagnosis (MRI): In this orthopedic study, 12 radiologists from IRCCS Ospedale Galeazzi Sant’Ambrogio (Milan, Italy) and other hospitals in Italy, had to classify 120 knee lesions using MRI scans selected from the MRNet dataset<sup>3</sup>, supported by a simulated AI system. Further details can be found in [30].

• Cardiologic ECG Evaluation (ECG): In this cardiological study, 21 medical professionals from the University Hospital of Siena (Italy) had to classify heartbeat patterns from 20 ECGs, supported by a simulated AI system. These ECGs were selected from the ECG Wave-Maven repository<sup>4</sup>. Further details can be found in [30].

• Spinal X-Ray Diagnosis (X-Rays): In this orthopedic study, 7 orthopedists from the Gaetano Pini-CTO Hospital (Milan, Italy) had to detect traumatic thoraco-lumbar fractures 12 X-ray images, supported by a real AI system. TFurther information is available in [31].

• Gastrointestinal Endoscopy Assessment (Endoscopy (HP), Endoscopy (IL), Endoscopy (UCEIS)): This study is a large-scale endoscopic evaluation in which 274 medical professionals from all over Europe were involved in three sub-studies. In the HP study, participants rated 30 cases of potential gastrointestinal bleeding (Hemorrhagic Potential) in short (15–20 second) endoscopic colonoscopy videos. In IL, participants rated 30 cases of potential inflammatory lesions in small bowel capsule endoscopy. In UCEIS, participants rated 30 cases of ulcerative colitis activity using proctosigmoidoscopy. In all, participants were supported by a simulated AI system. Further information is available in [32]

• Colorectal Endoscopy Diagnosis (Endoscopy (Colorectal)): In this study, 21 endoscopists had to diagnose 504 video clips of colorectal lesions, supported by a realworld AI tool. Further information is available in [33].

## A. Results

The reliance patterns are summarized in Table II.

Synergy emerged in only three out of seven studies: X-Rays, Endoscopy (UCEIS), and Endoscopy (Colorectal). In the four settings where synergy failed (MRI, ECG, Endoscopy (HP), and Endoscopy (IL)), Algorithmic Advantage (AI > UJ) was satisfied. However, Human Correction (HC) failed in all these cases. Notably, while Algorithmic Correction (AC) held in every study, its presence alone was insufficient to guarantee synergy in the absence of HC. This failure suggests a “dominance trap”: a strong Algorithmic Advantage can inhibit synergy if the human expert fails to identify the specific “broken leg” cases where the AI errs, thereby failing to provide the necessary Human Correction.

In regard to the synergistic cases, two of them illustrate similar theoretical pathways. In Endoscopy (UCEIS), synergy was achieved (approx. 1.5% improvement) through the simul taneous fulfillment of Human Advantage, HC, AC, and BA, thus providing empirical validation for Theorem 1. Interestingly, this occurred despite a lack of full Appropriate Reliance: while Automation Bias was not dominant, a significant selfanchoring Bias persisted. This provides a critical insight: AR is neither a sufficient nor necessary path to synergy; a high rate of HC can compensate for a lack of condition AR1, provided that the combination rule “vetoes” the most critical AI errors. In Endoscopy (Colorectal), a robust synergistic effect (approx. 3%) was observed. Similarly to the case of Endoscopy (UCEIS), also here, BA, Human Advantage, HC, and AC all held simultaneously. Unlike the UCEIS study, however, this case also demonstrated full appropriate reliance, aligning with Case 7 of Theorem 3.

The X-rays study presents the most unique scenario. Although HC and AC were satisfied, the interaction violated both Agreement and BA. Despite this, a substantial 9% accuracy gain was recorded. This suggests that the quality of human-AI interaction is not solely about the presence of (human or algorithmic) correction but rather about their “magnitude”. In this case, which provides an illustration of Theorem 4, the strength of both HC and AC was sufficiently large to overcome sporadic non-agreement. This implies that synergy is driven by the diagnostic resolution of the team: when HC and AC are strong, the human and AI outputs are effectively de-correlated on their respective errors, maximizing the information gain of the hybrid system. In turn, this provides inspiration for a generalization of our results, which gives a sufficient and necessary condition for Synergy:

Theorem 4. Synergy is equivalent to the following conditions:

$$
\begin{array} { r l } & { \mathrm { P r } ( { 1 0 1 } ) - \mathrm { P r } ( { 0 1 0 } ) > \mathrm { P r } ( { 1 1 0 } ) - \mathrm { P r } ( { 0 0 1 } ) , } \\ & { \mathrm { P r } ( { 0 1 1 } ) - \mathrm { P r } ( { 1 0 0 } ) > \mathrm { P r } ( { 1 1 0 } ) - \mathrm { P r } ( { 0 0 1 } ) } \end{array}
$$

Theorem 4 shows that synergy arises from a trade-off. In both inequalities, the left-hand side can be interpreted as the magnitude of correction, quantifying how strongly Human and Algorithmic Correction operate. By contrast, the right-hand side captures an aversion penalty. This penalty is 0 under agreement and becomes negative under Beneficial Aversion. Hence, synergy can emerge only when the combination function integrates the partial judgments H and A with context-dependent information about the situation S in such a way that the magnitude of correction outweighs the penalty associated with non-agreement (if any). Thus, while imposing structural constraints, such as agreement, on the combination rule may provide a simplified pathway to synergy, this latter can nonetheless be reached even when such constraints fail, provided that the interaction policy implemented by the combination rule is able to enforce the trade-off in Theorem 4.

TABLE II  
RESULTS OF THE CONSIDERED CASE STUDIES, IN TERMS OF COUNTS OF RELIANCE PATTERNS.
<table><tr><td></td><td>MRI</td><td>ECG</td><td>X-Rays</td><td>Endoscopy (HP)</td><td>Endoscopy (IL)</td><td>Endoscopy (UCEIS)</td><td>Endoscopy (Colorectal)</td></tr><tr><td>000</td><td>82</td><td>168</td><td>4</td><td>363</td><td>460</td><td>182</td><td>1126</td></tr><tr><td>001</td><td>1</td><td>1</td><td>10</td><td>4</td><td>3</td><td>52</td><td>245</td></tr><tr><td>010</td><td>238</td><td>102</td><td>9</td><td>649</td><td>350</td><td>345</td><td>251</td></tr><tr><td>011</td><td>46</td><td>71</td><td>38</td><td>376</td><td>184</td><td>174</td><td>827</td></tr><tr><td>100</td><td>21</td><td>12</td><td>5</td><td>75</td><td>61</td><td>141</td><td>540</td></tr><tr><td>101</td><td>202</td><td>71</td><td>45</td><td>410</td><td>336</td><td>494</td><td>954</td></tr><tr><td>110</td><td>11</td><td>0</td><td>16</td><td>53</td><td>10</td><td>15</td><td>173</td></tr><tr><td>111</td><td>947</td><td>415</td><td>161</td><td>3219</td><td>3076</td><td>2933</td><td>6464</td></tr><tr><td>UJ</td><td>0.76</td><td>0.59</td><td>0.79</td><td>0.73</td><td>0.78</td><td>0.83</td><td>0.77</td></tr><tr><td>AI</td><td>0.80</td><td>0.70</td><td>0.78</td><td>0.83</td><td>0.81</td><td>0.80</td><td>0.73</td></tr><tr><td>SD</td><td>0.77</td><td>0.66</td><td>0.88</td><td>0.78</td><td>0.80</td><td>0.84</td><td>0.80</td></tr></table>

## VII. DISCUSSION AND IMPLICATIONS FOR THE DESIGN OF AI-BASED DECISION SUPPORT

Our theoretical and empirical findings support a central conclusion: synergy is not primarily determined by the standalone accuracy of human or AI agents, but is instead an emergent property of the interaction policy that governs how their judgments are combined. This perspective shifts the focus of system design from improving individual models in isolation to engineering interaction mechanisms that enable effective collaboration between humans and AI systems.

A key implication of our results is that increasing the performance of an AI system alone is not sufficient to guarantee improved system-level outcomes. In fact, the prevalence of the “dominance trap” in several of the clinical case studies, suggests that strong AI performance can inadvertently suppress human involvement in the decision process, in particular when the human user owns the final process of decision combination. When users become overly reliant on the AI, opportunities for human correction diminish, and the system as a whole may fail to exploit complementary strengths. This highlights the importance of designing interaction protocols that preserve an active and meaningful role for human judgment, even when AI performance is high. A practical line of intervention would consist of avoiding creating black-box systems that merely provide a final label. Instead, AI tools should be designed to enhance users’ understanding of their outputs. However, this does not entail aligning AI system design uncritically with XAI research, which maintains that it is important for AI systems to “explain” what they do. A growing body of literature (e.g., [34, 35, 36]) has already warned about the risks of machine explanations increasing deference and automation bias, precisely because of their persuasive power. In this sense, our framework suggests treating explanations as reliability metadata that inform users at the level of individual classifications and local predictions [37], or as alternative hypotheses [38, 39] to consider in order to gain a better understanding of the matter at hand. In doing so, AI systems would promote users’ metacognitive sensitivity [40]—for instance, by highlighting areas of high uncertainty or providing cues that enable humans to effectively perform Human Correction.

Furthermore, while we note that enforcing the agreement condition is trivial from a systems-design perspective<sup>5</sup>, the evidence that Synergy can emerge even in the absence of such conditions suggests that the goal of a hybrid system should be “functional de-correlation”, as this latter can enforce even larger gains in performance than merely relying on interaction policy that are solely relegated to cases of disagreement. This insight has different implications depending on whether the combination rule is realized by human or, rather, through computational means. In the first case, human training can be easily seen to be a fundamental dimension for enabling synergy: users of AI-based decision support systems should develop a properly calibrated theory of machine. Training programs that expose users to examples of AI errors, or that simulate disagreement scenarios, can help improve calibration and support more effective reliance behavior. Such training can enhance the ability of users to detect when the AI is likely to be correct and when it should be overridden. By contrast, in the second case, our results imply that AI systems should be optimized not to simulate human reasoning—whatever this may mean—but rather to develop a distinct perspective on the phenomena of interest [41]. In this sense the conditions of Human and Algorithmic Correction, as well as the margin conditions given in Theorem 4, provide natural learning objectives to be used to to tailor a policy implementing the combination rule. More in general, and in both cases, our results also imply that designers should focus on identifying and supporting the human’s ability to act as a diagnostic filter, directing attention to cases in which the AI is most likely to fail (the “broken leg” cases) and protecting expertise from skill atrophy. This latter concern suggests designing and implementing human–AI interaction protocols aimed at “slowing down” users and making interaction with AI systems more effortful [42]—socalled inhibition protocols, that are, protocols that deliberately introduce a form of cognitive friction [43], such as requiring users to provide their judgment before being exposed to the machine’s advice [32], or having the AI system abstain from providing any advice at all when the risk of broken leg cases is too high [44], with goal of making engagement with AI a more deliberative and reflective (rather than automatic) process.

At a broader level, governance and deployment strategies should take into account not only the accuracy of individual components but also the quality of their interaction. Monitoring system performance should include metrics derived from reliance patterns, disagreement rates, and correction frequencies. These indicators provide insight into whether the system operates in a regime conducive to synergy, and can reveal whether interaction policies are functioning as intended. Over time, such monitoring can support continuous improvement of both the AI system and the surrounding interaction protocols.

In summary, achieving synergy requires a coordinated approach that integrates model development, interface design, interaction protocols, and user training. Rather than focusing solely on maximizing standalone performance, system designers should aim to create environments in which human and AI agents can effectively complement each other. This involves promoting structured interaction, designing principled disagreement-resolution policies, supporting metacognitive awareness, and encouraging functional differences that allow each agent to correct the other. Under these conditions, synergy emerges not as an accidental outcome but as a designenabled property of the human-AI system.

## VIII. CONCLUSION

This paper has developed a probabilistic framework to formalize the elusive notion of human-AI synergy. Our analysis demonstrates that while Algorithmic Advantage and Human Augmentation are frequent benchmarks in the literature, true Synergy is a more rigorous condition, mathematically tied to the existence of effective Human Correction (HC) and Algorithmic Correction (AC) mechanisms. By grounding these concepts in the formal calculus of reliance patterns, we have provided a principled basis to move beyond simple accuracy metrics toward a deeper understanding of hybrid decisional systems, as well as highlighted significant implications for the design of hybrid human-AI systems.

Our empirical investigation across seven clinical datasets confirms that Synergy is not a guaranteed outcome of AI integration. The observed “dominance trap” in high-performing AI settings and the compensatory role of HC and AC highlight that the path to Synergy is mediated by the human’s metacognitive sensitivity and the team’s ability to functionally de-correlate errors. These findings suggest that the future of AI design should prioritize the calibration of the human-AI interaction over the mere maximization of model performance.

In conclusion, the pursuit of Synergy, although more challenging than plain automation, is shown to be a mathematically sound and empirically attainable goal. The formal conditions identified here provide researchers and practitioners with a roadmap for designing and evaluating hybrid systems that do not merely supplement human judgment but transcend the limitations of both human and artificial intelligence, to realize the potential of a hybrid intelligence.

## REFERENCES

[1] G. Bansal, B. Nushi, E. Kamar, W. S. Lasecki, D. S. Weld, and E. Horvitz, “Beyond accuracy: The role of mental models in human-ai team performance,” in Proceedings ofthe AAAI Conference on hHman Computation and Crowdsourcing, vol. 7, 2019, pp. 2–11.

[2] P. Hemmer, M. Schemmer, N. Kühl, M. Vössing, and G. Satzger, “Complementarity in human-ai collaboration: Concept, sources, and evidence,” European Journal of Information Systems, pp. 1–24, 2025.

[3] A. Vaccaro, A. Almaatouq, and T. W. Malone, “Synergistic human–ai interaction,” Proceedings of the National Academy of Sciences, vol. 121, no. 34, p. e2402287121, 2024.

[4] Y. Bao, W. Gong, and K. Yang, “A literature review of human–ai synergy in decision making: from the perspective of affordance actualization theory,” Systems, vol. 11, no. 9, p. 442, 2023.

[5] T. Rieger, L. Onnasch, E. Roesler, and D. Manzey, “Why highly reliable decision support systems often lead to suboptimal performance and what we can do about it,” IEEE Transactions on Human-Machine Systems, 2025.

[6] A. Ferrario, A. Facchini, and J. Durán, “Epistemology gives a future to complementarity in human–ai interactions,” 2026, arXiv preprint arXiv:2601.09871.

[7] D. Madras, T. Pitassi, and R. Zemel, “Predict responsibly: improving fairness and accuracy by learning to defer,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[8] M. A. Charusaie, H. Mozannar, D. Sontag, and S. Samadi, “Sample efficient learning of predictors that complement humans,” in International Conference on Machine Learning. PMLR, 2022, pp. 2972–3005.

[9] E. Straitouri, L. Wang, N. Okati, and M. G. Rodriguez, “Improving expert predictions with conformal prediction,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 633–32 653.

[10] G. De Toni, N. Okati, S. Thejaswi, E. Straitouri, and M. Rodriguez, “Towards human-ai complementarity with prediction sets,” Advances in Neural Information Processing Systems, vol. 37, pp. 31 380–31 409, 2024.

[11] N. Chater, J. B. Tenenbaum, and A. Yuille, “Probabilistic models of cognition: Conceptual foundations,” Trends in Cognitive Sciences, vol. 10, no. 7, pp. 287–291, 2006.

[12] M. Steyvers, H. Tejeda, G. Kerrigan, and P. Smyth, “Bayesian modeling of human–ai complementarity,” Proceedings of the National Academy of Sciences, vol. 119, no. 11, p. e2111547119, 2022.

[13] S. Westby and C. Riedl, “Collective intelligence in human-ai teams: A bayesian theory of mind approach,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 5, 2023, pp. 6119–6127.

[14] K. Donahue, A. Chouldechova, and K. Kenthapadi, “Human-algorithm collaboration: Achieving complementarity and avoiding unfairness,” in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022, pp. 1639–1656.

[15] K. Peng, N. Garg, and J. Kleinberg, “A no free lunch theorem for human-ai collaboration,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 13, 2025, pp. 14 369–14 376.

[16] F. Cabitza, A. Campagner, R. Angius, C. Natali, and C. Reverberi, “Ai shall have no dominion: on how to measure technology dominance in ai-supported human decision-making,” in Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, 2023, pp. 1–20.

[17] P. E. Meehl, Clinical Versus Statistical Prediction: A Theoretical Analysis and a Review of the Evidence. University of Minnesota Press, 1954.

[18] J. Y. Chen and M. J. Barnes, “Human–agent teaming for multirobot control: A review of human factors issues,”

IEEE Transactions on Human-Machine Systems, vol. 44, no. 1, pp. 13–29, 2014.

[19] J. D. Lee and K. A. See, “Trust in automation: Designing for appropriate reliance,” Human Factors, vol. 46, no. 1, pp. 50–80, 2004.

[20] M. Schemmer, N. Kuehl, C. Benz, A. Bartos, and G. Satzger, “Appropriate reliance on AI advice: Conceptualization and the effect of explanations,” in Proceedings of the 28th International Conference on Intelligent User Interfaces, 2023, pp. 410–422.

[21] A. Campagner, C. Fregosi, C. Natali, and F. Cabitza, “Under what influence: Measuring AI influence to fit user profiles in decision-making,” International Journal of Human Computer Studies, p. 103763, 2026.

[22] I. Yaniv and E. Kleinberger, “Advice taking in decision making: Egocentric discounting and reputation formation,” Organizational behavior and human decision processes, vol. 83, no. 2, pp. 260–281, 2000.

[23] R. Parasuraman and V. Riley, “Humans and automation: Use, misuse, disuse, abuse,” Human factors, vol. 39, no. 2, pp. 230–253, 1997.

[24] S. M. Fleming and H. C. Lau, “How to measure metacognition,” Frontiers in Human Neuroscience, vol. 8, p. 443, 2014.

[25] J. W. Burton, M. K. Stein, and T. B. Jensen, “A systematic review of algorithm aversion in augmented decision making,” Journal of Behavioral Decision Making, vol. 33, no. 2, pp. 220–239, 2020.

[26] M. Johnson and A. H. Vera, “No AI is an island: The case for teaming intelligence,” AI Magazine, vol. 40, no. 1, pp. 16–28, 2019.

[27] J. M. Logg, J. A. Minson, and D. A. Moore, “Algorithm appreciation: People prefer algorithmic to human judgment,” Organizational Behavior and Human Decision Processes, vol. 151, pp. 90–103, 2019.

[28] B. J. Dietvorst, J. P. Simmons, and C. Massey, “Algorithm aversion: People erroneously avoid algorithms after seeing them err,” Journal of Experimental Psychology: General, vol. 144, no. 1, pp. 114–126, 2015.

[29] L. Introzzi, J. Zonca, F. Cabitza, P. Cherubini, and C. Reverberi, “Enhancing human-ai collaboration: The case of colonoscopy,” Digestive and Liver Disease, vol. 56, no. 7, pp. 1131–1139, 2024.

[30] F. Cabitza, A. Campagner, L. Ronzio, M. Cameli, G. E. Mandoli, M. C. Pastore, L. M. Sconfienza, D. Folgado, M. Barandas, and H. Gamboa, “Rams, hounds and white boxes: Investigating human–ai collaboration protocols in medical diagnosis,” Artificial Intelligence in Medicine, vol. 138, p. 102506, 2023.

[31] F. Cabitza, A. Campagner, L. Famiglini, E. Gallazzi, and G. A. La Maida, “Color shadows (part i): Exploratory usability evaluation of activation maps in radiological machine learning,” in International Cross-Domain Conference for Machine Learning and Knowledge Extraction, ser. Lecture Notes in Computer Science, vol. 13480, 2022, pp. 31–50.

[32] F. Cabitza, A. Campagner, C. Fregosi, M. Cameli, E. Gallazzi, L. M. Sconfienza, and G. E. Tontini, “Five degrees

of separation: Investigating the unexpected potential of displaced human-ai collaboration protocols for apter AI support,” Proceedings of the ACM on Human-Computer Interaction, vol. 9, no. 7, pp. 1–28, 2025.

[33] C. Reverberi, T. Rigon, A. Solari, C. Hassan, P. Cherubini, and A. Cherubini, “Experimental evidence of effective human–ai collaboration in medical decisionmaking,” Scientific reports, vol. 12, no. 1, p. 14952, 2022.

[34] F. Cabitza, C. Fregosi, A. Campagner, and C. Natali, “Explanations considered harmful: the impact of misleading explanations on accuracy in hybrid human-ai decision making,” in World Conference on Explainable Artificial Intelligence. Springer Nature Switzerland, 2024, pp. 255–269.

[35] U. Ehsan and M. O. Riedl, “Explainability pitfalls: Beyond dark patterns in explainable ai,” Patterns, vol. 5, no. 6, 2024.

[36] M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, “The false hope of current approaches to explainable artificial intelligence in health care,” The Lancet Digital Health, vol. 3, no. 11, pp. e745–e750, 2021.

[37] L. Peracchio, G. Nicora, E. Parimbelli, T. M. Buonocore, E. Tavazzi, R. Bergamaschi, A. Dagliati, and R. Bellazzi, “Relai: an automated approach to judge pointwise ml prediction reliability,” International journal of medical informatics, vol. 197, p. 105857, 2025.

[38] T. Miller, “Explainable AI is dead, long live explainable ai! hypothesis-driven decision support using evaluative ai,” in Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, 2023, pp. 333–342.

[39] M. Blom, R. Singh, T. Miller, L. Sonenberg, K. Trentelman, A. Saulwick, and S. Wark, “Improving tactical decision-making through multiobjective contrastive,” IEEE Transactions on Human-Machine Systems, 2025.

[40] D. Lee, J. Pruitt, T. Zhou, J. Du, and B. Odegaard, “Metacognitive sensitivity: The key to calibrating trust and optimal decision making with ai,” PNAS Nexus, vol. 4, no. 5, p. pgaf133, 2025.

[41] B. Wilder, E. Horvitz, and E. Kamar, “Learning to complement humans,” in Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 2021, pp. 1526–1533.

[42] G. Booch, F. Fabiano, L. Horesh, K. Kate, J. Lenchner, N. Linck, A. Loreggia, K. Murgesan, N. Mattei, F. Rossi, and B. Srivastava, “Thinking fast and slow in ai,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 17, 2021, pp. 15 042–15 046.

[43] F. Cabitza, A. Campagner, L. Famiglini, C. Natali, V. Caccavella, and E. Gallazzi, “Let me think! investigating the effect of explanations feeding doubts about the AI advice,” in International Cross-Domain Conference for Machine Learning and Knowledge Extraction, 2023, pp. 155–169.

[44] H. Mozannar, H. Lang, D. Wei, P. Sattigeri, S. Das, and D. Sontag, “Who should predict? exact algorithms for learning to defer to humans,” in AISTATS. PMLR, 2023, pp. 10 520–10 545.