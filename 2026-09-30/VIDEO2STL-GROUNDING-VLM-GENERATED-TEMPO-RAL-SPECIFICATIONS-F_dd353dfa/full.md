# VIDEO2STL: GROUNDING VLM-GENERATED TEMPO-RAL SPECIFICATIONS FOR ROBOT LEARNING

Merve Atasever<sup>1</sup> Keyan Azbijari<sup>1</sup> Cagan Bakirci<sup>1</sup> Alfredo Reina Corona<sup>1</sup> Bo-Ruei Huang<sup>1</sup> Tolga Izdas<sup>1</sup> Zahra Shahrooei<sup>1</sup> Richard Yang<sup>2</sup> Erdem Biyik<sup>1</sup> Jyotirmoy V. Deshmukh<sup>1</sup>

<sup>1</sup>Department of Computer Science, University of Southern California Los Angeles, CA, USA

{atasever, azbijari, cbakirci, reinacor, borueihu, izdas, shahrooe, biyik, jdeshmuk}@usc.edu

<sup>2</sup>University of Florida Gainesville, FL, USA yangrichard@ufl.edu

## ABSTRACT

Video-based policy learning is particularly promising, as it illustrates target behaviors without requiring action annotations or embodiment-matched demonstrations. A central challenge, however, is deciding what information should be transferred from the video to the robot. Existing approaches commonly convert visual observations into scalar similarity or value signals, or ask foundation models to directly generate reward code. While effective, these approaches can make the temporal structure of a task difficult to inspect, ground, and reuse. We present Video2STL, a framework that converts observation-only videos into parametric Signal Temporal Logic (STL) specifications and uses the resulting formal representation for robot learning. A vision-language model first extracts an embodiment-independent semantic event trace and then constructs a bank of symbolic temporal specifications. The model determines the task structure, while numerical predicate thresholds and temporal bounds are grounded from successful robot trajectories. For policy learning, we separate short- and long-timescale temporal information: short-horizon specifications provide dense rewards through rolling-window quantitative robustness, while a causal monitor over a retained long-horizon specification provides one-time progress rewards for valid temporal prefixes. The same representation is intended to support cross-embodiment transfer from human or animal videos to robot control while remaining interpretable at every stage. Across four manipulation tasks, Video2STL achieves 85.8% average success-once and 67.0% successat-end, compared with 81.5%/59.5% for native dense PPO and 65.0%/42.3% for Text2Reward; in quadruped locomotion, the Qwen-3.8 and GPT-5.6-based Video2STL policies achieve 100% success across velocities from 0.3 to 2.1 m/s while remaining competitive in high-speed energy efficiency. Project webpage: video2stl.

## 1 INTRODUCTION

Reinforcement learning has become a practical tool for robot control, but reward design remains one of the parts that is least reusable across tasks. A reward that works well often combines geometric errors, contact heuristics, success bonuses, regularizers, and task-specific schedules. The resulting function may train a strong policy while still being difficult to interpret or modify. This problem becomes more pronounced when the desired behavior is easier to show than to describe numerically. Human and animal videos provide abundant examples of manipulation and locomotion, yet they do not expose the demonstrator’s actions in the robot action space, and the morphology of the demonstrator may be very different from that of the target robot.

Recent work has moved toward more natural interfaces for reward and behavior specification along two largely parallel lines. On the visual side, learning from human video has been framed as a cross-domain imitation problem (Li et al., 2022). VIP derives dense, goal-conditioned rewards from a representation pretrained on large-scale human video (Ma et al., 2023), and contrastive visionlanguage models such as CLIP have been used directly as zero-shot, language-specified reward models (Rocamonde et al., 2024). Generative Value Learning moves further toward general progress estimation, using a VLM to infer per-frame task progress on over 300 real-world tasks spanning many robot embodiments, with human videos usable as in-context examples (Ma et al., 2025). On the language side, language models have served as proxy reward functions that judge behavior against a user’s examples or description (Kwon et al., 2023). Text2Reward and Eureka showed that they can also write dense reward programs and refine them through feedback or evolutionary search (Xie et al., 2024; Ma et al., 2024). ROSETTA constructs code-based rewards from unconstrained, evolving language preferences (Srivastava et al., 2026). Taken together, these results make a strong case that foundation models can provide task-level supervision. That supervision, however, takes the form of a scalar reward or progress signal for a task specified in language or as a goal image, which leaves open a question: what intermediate representation should connect an observation-only video to low-level robot learning?

![](images/a12383407aadfff2b3581baabbef91a65dd78001a16e3d8991261755a1b3ef09.jpg)  
Figure 1: Overview of Video2STL framework. A VLM parses an observation video into a semantic event trace and generates parametric STL formulas. Predicate thresholds and temporal bounds are then fitted to target-embodiment trajectories. Smooth rolling-window robustness over short-horizon formulas provides dense local guidance, while a causal prefix monitor over the long-horizon backbone yields milestone progress rewards for RL policy training.

A scalar reward is convenient for optimization, but it hides how the task unfolds over time. Consider a simple placement behavior. "Near the object", "move it toward the destination", "place it", and "leave it stable" are not four unrelated scores; they form a temporal structure. A reward program generated directly from text or video can encode this structure implicitly, but the structure is then mixed with numerical thresholds, simulator APIs, and implementation choices. Visual embedding rewards have the opposite advantage: they require little explicit structure, but it can be difficult to tell why a rollout receives a particular score or whether a high score corresponds to the intended temporal behavior. These limitations matter most in cross-embodiment settings, where pixel or pose correspondence is unreliable and where the transferable content is often semantic rather than kinematic.

Our central design choice is to separate what the video should decide from what robot data should decide. The video determines symbolic task structure: which entities matter, which semantic events occur, and how those events are ordered. Robot trajectories determine embodiment-specific numerical quantities: how close counts as Near, how much object motion counts as Displaced, and how long a temporal relation should be allowed to take. Signal Temporal Logic (STL) provides a natural interface between these two levels because it combines human-readable temporal operators with quantitative robustness semantics over real-valued robot signals (Maler & Nickovic, 2004; Fainekos & Pappas, 2009; Donzé & Maler, 2010). The same formula can therefore be read as a requirement, evaluated as a monitor, and converted into a dense learning signal.

We introduce Video2STL, a video to formal specification pipeline for robot learning. Given an observation-only video V, a vision-language model (VLM) first maps the video to an embodimentindependent semantic event trace E. A second stage maps E to a bank of parametric STL formulas Φ. We deliberately do not ask the VLM to estimate robot-specific distances, velocities, or timing constants. Instead, we collect successful robot trajectories and split them once into a grounding set and a held-out filtering set. The grounding split is used to fit predicate thresholds and temporal bounds. The filtering split is used only to test whether each grounded formula is compatible with successful target-embodiment behavior.

The second design problem is how to turn a temporal specification into an RL signal without destroying credit assignment. Prior work has shown that STL can define non-Markovian rewards and encode sequential dependencies (Venkataraman et al., 2020; Puranic et al., 2021). An earlier locomotion work also showed that the robustness of finite-history STL can provide a structured reward once temporal templates are available (Atasever et al., 2026). In our setting, however, the formulas are produced from video and can span very different timescales. We therefore use a two-timescale construction. Short formulas are evaluated on a trailing window $[ t - H _ { \mathrm { l o c a l } } , t ]$ and produce a dense local reward. Long formulas are not compressed into the same local window; instead, a causal prefix monitor provides a one-time reward when a new valid stage of the temporal sequence is reached. This keeps the dense signal local while preserving the ordering and deadlines of the long task.

A final issue is smooth robustness. Standard STL robustness uses nested min/max operations. Replacing them with conventional smooth approximations can improve optimization but can also change the sign of robustness, so a formula that is false under hard STL can become positive after smoothing. We use sign-preserving Boltzmann reductions whose output has the same sign as the corresponding hard min/max. This keeps the logical satisfaction boundary exact while still providing smoother within-region magnitudes for reward construction.

## Our contributions are:

1. We formulate observation-only video as a source of formal temporal task structure. A two-stage VLM pipeline maps video to a semantic trace and then to a bank of parametric STL specifications, using one reusable ontology across manipulation tasks rather than task-specific reward templates.

2. We separate symbolic inference from embodiment-specific grounding. Numerical predicate thresholds and temporal bounds are fit on one set of successful robot trajectories, while a disjoint held-out set is used for expert-consistency filtering. The expert data calibrates and validates the VLM specifications but does not supply action supervision or define their symbolic structure.

3. We develop a two-timescale specification reward: rolling-window robustness from short formulas provides dense feedback, while a causal monitor of a long temporal backbone provides sparse, non-farmable progress.

4. We instantiate the framework in robot manipulation and quadruped locomotion, targeting cross-embodiment transfer in which the source demonstrator and target robot need not share morphology. The resulting representation remains inspectable from video interpretation through policy training.

## 2 RELATED WORK

Learning Robot Behavior from Video. Early work on learning from observation focused largely on closing the domain gap between human demonstrations and robot execution. This was approached through context translation between demonstrator and robot observations (Liu et al., 2018), viewpointinvariant visual representations for human-to-robot imitation (Sermanet et al., 2018), and domainadaptive meta-learning with paired human and robot demonstrations (Yu et al., 2018). Li et al. (2022) further reduced the need for robot demonstrations during meta-training by translating human videos into robot-domain demonstrations. In these approaches, the information transferred from video is mainly represented through learned visual features, translated demonstrations, or adapted policies.

More recent methods use pretrained visual models to extract a learning signal directly from video. VIP learns a value-implicit representation from large-scale human video and constructs dense rewards from distances in the learned embedding space (Ma et al., 2023). RoboCLIP compares an agent’s interaction trajectory with a video or language task specification using a pretrained video-language model and uses the resulting similarity as an episodic reward (Sontakke et al., 2023). GVL instead uses a VLM to estimate task progress by reasoning over the temporal ordering of video frames, with support for in-context examples from different tasks and embodiments (Ma et al., 2025).

Foundation Models for Reward Design. A related line of work uses language and vision-language models to construct reward functions from high-level task descriptions. Kwon et al. (2023) uses an LLM as a proxy reward function conditioned on a natural-language objective. Text2Reward generates executable dense reward programs from language instructions and environment APIs (Xie et al., 2024), while Eureka iteratively improves LLM-generated reward code using downstream policy performance as feedback (Ma et al., 2024). Rocamonde et al. (2024) uses pretrained visionlanguage similarity directly as an RL reward, and RL-VLM-F queries a VLM for pairwise preferences over visual observations and trains a reward model from those comparisons (Wang et al., 2024). Video2Reward extracts motion keypoint trajectories from the video, prompts an LLM to generate executable reward code for legged robot learning, and iteratively refines the reward using visual feedback from the learned policy (Zeng et al., 2024). More recently, ROSETTA uses language preferences to construct staged reward code for manipulation (Srivastava et al., 2026).

Temporal Logic for Robot Learning. Prior work has explored logic-guided RL by converting temporal-logic objectives into quantitative rewards (Hasanbeig et al., 2020; Li et al., 2017; Kapoor et al., 2020; Puranic et al., 2021). Other work has studied learning temporal-logic specifications from demonstrations (Venkataraman et al., 2020; Puranic et al., 2021). Formal specifications have also guided planning and control for legged robots, including locomotion over cluttered terrain and logic-driven gait learning for quadrupeds Gu et al. (2025); Humphreys & Zhou (2025); DeFazio et al. (2024); Atasever et al. (2026).

A related line of work translates natural-language commands into formal specifications. Lang2LTL grounds complex commands into LTL for long-horizon robot navigation in previously unseen environments (Liu et al., 2023). The important distinction is the source and use of the specification. Lang2LTL starts from language and uses logic primarily to represent commands for planning/execution. In contrast, Video2STL does not assume that the temporal task structure is given in advance: the structure is extracted from video by VLM and then checked against embodiment-specific robot data.

The closest prior lines can be summarized by   
what they transfer from the source signal. Video Algorithm 1: VIDEO2STL   
imitation methods transfer appearance or behav  
ior representations; visual-reward methods trans- Input: Video V , successful trajectories $\mathcal { D } ^ { + }$   
fer scalar similarity or progress; LLM reward- Output: Policy π<sub>θ</sub>   
generation methods transfer language intent 1: $E  \operatorname { V L M } _ { \mathrm { e v e n t } } ( V )$   
into executable code; language-to-logic methods 2: $\Phi  \operatorname { V L M } _ { \operatorname { S T L } } ( E )$   
transfer explicit commands into symbolic tem- 3: Split $\mathcal { D } ^ { + }  ( \mathcal { D } _ { \mathrm { g r o u n d } } , \mathcal { D } _ { \mathrm { f i l t e r } } )$   
poral constraints. Video2STL focuses on a dif-  
4: $\vartheta \gets \mathrm { G r o u n d } ( \Phi , { \mathcal { D } } _ { \mathrm { g r o u n d } } )$   
ferent interface between these directions: video   
5: $\Phi _ { \mathrm { k e e p } }  \mathrm { F i l t e r } ( \Phi _ { \vartheta } , \mathcal { D } _ { \mathrm { f i l t e r } } )$   
→ explicit temporal specification → grounded   
quantitative reward and causal monitor. The 6: Construct $r _ { t } ^ { \mathrm { l o c a l } }$   
proposed representation is deliberately more 7: Construct $r _ { t } ^ { \mathrm { r } }$ progress   
constrained than free-form reward code and 8: $r _ { t } \gets \lambda _ { L } r _ { t } ^ { \mathrm { l o c a l } } + \lambda _ { P } r _ { t } ^ { \mathrm { p r o g r e s s } }$   
more structured than a scalar visual reward. The   
9: $\pi _ { \theta }  \mathrm { R L } ( r _ { t } )$   
temporal specifications determine which quan  
tities must be grounded, provide explicit satis-

faction boundaries that can be checked on held-out expert trajectories, and support different creditassignment mechanisms for short- and long-horizon temporal structure.

## 3 TECHNICAL BACKGROUND AND PRELIMINARIES

## 3.1 PROBLEM SETTING

We consider a target robot controlled by a policy $\pi _ { \boldsymbol { \theta } } ( a _ { t } \mid o _ { t } )$ in an episodic Markov decision process $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , \bar { T } )$ . We assume a source video V showing a successful behavior, and a set of successful robot trajectories $\mathcal { D } ^ { + } = \{ \tau ^ { ( n ) } \} _ { n = 1 } ^ { N }$ , where $\tau ^ { ( n ) } = ( x _ { 0 } ^ { ( n ) } , \dots , x _ { T _ { n } } ^ { ( n ) } )$ . The source video and robot trajectories do not need to come from the same embodiment. We first prompt the VLM to extract a semantic event trace E from the source video, identifying the task-relevant events and their temporal relationships. The VLM is then prompted a second time to convert this semantic description into a bank of parametric STL formulas Φ. At this stage, the formulas define the symbolic structure of the task but leave embodiment-specific quantities, such as geometric thresholds and temporal bounds, unspecified. Successful robot trajectories are subsequently used to ground these numerical parameters and to perform expert-consistency filtering, removing formulas that are not supported by held-out successful behavior. The method follows one principle throughout: the video determines the symbolic task structure, while robot data determines its numerical parameters. This separation allows a human or animal video to define the task without requiring morphological correspondence between the demonstrator and the robot.

## 3.2 SIGNAL TEMPORAL LOGIC (STL) & QUANTITATIVE ROBUSTNESS

Signal Temporal Logic (STL) specifies temporal properties over real-valued signals $x ( t )$ using Boolean and temporal operators (Maler & Nickovic, 2004). Formulas are built from atomic predicates $f ( x ( t ) ) \geq 0$ , Boolean connectives $( \land , \lor , \lnot )$ , and time-bounded temporal operators ${ \bf G } _ { [ a , b ] }$ (always) and ${ \bf F } _ { [ a , b ] } \left( \mathrm { e v e n t u a l l y } \right)$ . We use quantitative semantics of STL (Donzé & Maler, 2010; Fainekos & Pap-$\mathrm { p a s } , 2 0 0 9 ) ;$ for a given trace $x ,$ and each formula $\varphi \rho ^ { \varphi } ( t )$ assigns a real value capturing the degree of satisfaction. Positive values imply satisfaction; negative imply violation, and the magnitude measures the satisfaction margin. For atomic predicates, $\rho ( \check { f } ( x ) \geq 0 , \dot { x } , t ) = f ( x ( t ) )$ . For conjunction/disjunction, robustness uses min/max: $\rho ^ { \varphi \hat { \wedge } \psi } ( t ) = \mathrm { m i n } \bar { ( } \rho ^ { \varphi } ( t ) , \rho ^ { \psi } ( t ) )$ , and $\mathsf { \tilde { \rho } } ^ { \varphi \mathsf { \vee } \psi } ( t ) = \operatorname* { m a x } ( \rho ^ { \varphi } ( t ) , \rho ^ { \psi } ( t ) )$ For time-bounded operators, we have $\begin{array} { r } { \rho ^ { \mathbf { G } _ { I } \varphi } ( t ) = \operatorname* { i n f } _ { t ^ { \prime } \in t + I } \rho ^ { \varphi } ( t ^ { \prime } ) } \end{array}$ , and $\begin{array} { r } { \rho ^ { \mathbf { F } _ { I } \varphi } ( t ) = \operatorname* { s u p } _ { t ^ { \prime } \in t + I } \rho ^ { \varphi } ( t ^ { \prime } ) } \end{array}$ for an arbitrary time interval $I = [ a , b ]$ . Parametric Signal Temporal Logic (PSTL) extends STL by allowing constants in predicates and intervals in temporal operators to be represented by parameters (Asarin et al., 2011). The specification mining problem seeks to infer parameter ranges where the template formula is satisfied providing a data-driven way to instantiate temporal specifications from expert-provided locomotion trajectories.

## 4 METHODOLOGY

## 4.1 VIDEO TO AN EMBODIMENT-INDEPENDENT SEMANTIC TRACE

Directly asking a VLM to write a reward from pixels entangles perception, task interpretation, robot geometry, and reward engineering in one call. We instead first ask for a semantic trace that will later be translated into temporal logic. All videos use the same global ontology for manipulation and quadruped locomotion separately. For manipulation, the current entity roles are

R = {effector, object, receptacle, handle, articulated\_part, goal, support}, and the predicate vocabulary is

$$
\mathcal { P } = \{ \mathrm { N e a r } ( a , b ) , \ \mathrm { E n g a g e d } ( e , x ) , \ \mathtt { G r a s p e d } ( o ) , \ \mathtt { R e l e a s e d } ( o ) , \ \mathtt { N e l e a s e d } ( o ) , \ \mathtt { N e m e n t } ( a ) \}
$$

$$
\begin{array} { r } { \mathtt { D i s p l a c e d ( \boldsymbol { o } ) } , \mathtt { T o w a r d \_ g o a l ( \boldsymbol { o } ) } , \mathtt { A t \_ g o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } , \mathtt { N o a l ( \boldsymbol { o } ) } . } \end{array}
$$

$$
\begin{array} { r } { \mathtt { O n \_ s u p p o r t } ( o , s ) , \mathtt { I n \_ r e c e p t a c l e } ( o , r ) , \mathtt { S t a t i c } ( x ) , \mathtt { O p e n } ( q ) \} . } \end{array}
$$

For quadruped locomotion, the current entity roles are

R = {trunk, front\_left, front\_right, hind\_left, hind\_right, support}, and the predicate vocabulary is

$$
\mathcal { P } = \{ \mathrm { c o n t a c t } ( l e g ) , \ \mathrm { S w i n g } ( l e g ) , \ \mathrm { T o u c h d o w n } ( l e g ) , \ \mathrm { L i f t o f } \in ( l e g ) , 
$$

$$
\mathtt { S u p p o r t \_ c o u n t \_ c o t \_ l e a s t } ( k , l e a s t ( k , l e g ) , \mathtt { T r u n k \_ h e i g h t \_ s t a b l e } ( t ) ,
$$

$$
\mathtt { T r u n k \_ p i t c h \_ s t a b l e } ( t ) , \mathtt { T r u n k \_ r o l 1 \_ s t a b l e } ( t ) , \mathtt { P e r i o d i c \_ l e g \_ m o t } \mathtt { i o n } ( l e g ) \mathtt { j e c t \_ t a b l e } ( t ) , \mathtt { P e r i o d i c \_ l e g \_ m o t } \mathtt { i o n } ( t ) .
$$

The first query outputs a set of event records $e _ { k } = \left( p _ { k } , \ c _ { k } \right)$ where $p _ { k } \in \mathcal { P }$ is the predicate and $c _ { k }$ is an event class (essential/terminal/incidental candidate). The output also includes pairwise temporal relations, such as before, overlaps, and alternates\_with, persistent properties and a missing concepts field. If a visually important concept is absent from the ontology, the VLM must report the missing concept rather than invent a new predicate. This turns ontology insufficiency into an observable failure mode instead of silently changing the specification language.

## 4.2 SEMANTIC TRACE TO A PARAMETRIC STL BANK

In this stage, the VLM receives the semantic trace as a JSON file, not the original video. Its purpose is to formalize temporal relationships. The model constructs a variable-size bank $\boldsymbol { \Phi } = \{ \phi _ { 1 } , . . . , \bar { \phi } _ { K } \}$ using only predicates supported by the trace. Numerical predicate thresholds and time constants are forbidden at this stage, only symbolic parameters such as $H _ { \mathrm { t a s k } } , h _ { \mathrm { g o a l } } , \mathrm { o r } h _ { \mathrm { s e t t l e } }$ are allowed. The current grammar contains atomic predicates, conjunction, bounded eventually and always, and nested bounded responses: $\phi : : = \mu \mid \phi _ { 1 } \wedge \phi _ { 2 } \mid \mathbf { F } _ { [ 0 , h ] } \phi \mid \mathbf { G } _ { [ 0 , h ] } \phi$ with implication represented through standard Boolean composition when needed. The nested form $\mathbf { F } _ { [ 0 , H ] } \left( \mu _ { 1 } \wedge \mathbf { F } _ { [ 0 , h _ { 1 } ] } \left( \mu _ { 2 } \wedge \mathbf { F } _ { [ 0 , h _ { 2 } ] } \mu _ { 3 } \right) \right)$ expresses a bounded event sequence, while ${ \bf F } _ { [ 0 , H ] } { \bf G } _ { [ 0 , h ] } \mu$ expresses eventual persistence. We use a bank rather than forcing a single large formula because the source video can support multiple useful requirements at different temporal scales. For example, for PushCube, the bank contains seven formulas. Representative examples are

$$
\phi _ { 1 } \dot { = } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { N e a r } \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { e n g a g e } } ] } \mathrm { E n g a g e d } \right) ,\tag{1}
$$

$$
\phi _ { 2 } = \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( { \mathrm { T o w a r d G o a l } } \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathbb { A } { \mathrm { t G o a l } } \right) ,\tag{2}
$$

$$
\phi _ { 5 } = \mathbf { G } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \mathrm { O n } S \mathsf { u p p o r t } ,\tag{3}
$$

$$
\phi _ { 7 } = \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { T o w a r d G o a l } \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \left( \mathbb { A } \mathrm { t G o a l } \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { s e t t i l e } } ] } \mathrm { S t a t i c } \right) \right) .\tag{4}
$$

The remaining formulas cover engagement-to-progress, displacement/progress, and goal-to-settling relationships. The full bank is reported in the Appendix.

## 4.3 TARGET-EMBODIMENT GROUNDING

The VLM output is symbolic. To evaluate it on a robot, each predicate must be mapped to a realvalued signed margin and each symbolic temporal parameter must be assigned a numerical bound. We split the successful trajectories into disjoint grounding and filtering sets, $\mathbf { \bar { \mathcal { D } } } ^ { + } = \mathcal { D } _ { \mathrm { g r o u n d } } \cup \mathcal { D } _ { \mathrm { f i l t e r } } ,$ with $\mathrm { \bar { \mathcal { D } } _ { g r o u n d } } \cap \mathcal { D } _ { \mathrm { f i l t e r } } = \mathrm { \bar { \mathcal { O } } }$ . Our current experiments use a $7 0 / 3 0$ split. All thresholds, time bounds, and reward-normalization scales are fit from $\mathcal { D } _ { \mathrm { g r o u n d } } \ \mathrm { o n l y }$

Each semantic predicate $p$ is associated with a signed atomic robustness $\rho _ { p } ( x _ { t } ; \vartheta _ { p } )$ , where $\vartheta _ { p }$ denotes the embodiment-specific numerical parameters of the predicate and $\rho _ { p } ( x _ { t } ; \vartheta _ { p } ) \ge 0$ indicates satisfaction. We estimate $\vartheta _ { p }$ using empirical quantiles rather than a single trajectory or hand-tuned thresholds. The choice of quantile depends on the direction of the predicate. For an upper-bound condition $p ( x _ { t } ) : f _ { p } ( x _ { t } ) \leq \delta _ { p }$ , we define $\rho _ { p } ( x _ { t } ) = \delta _ { p } - f _ { p } ( x _ { t } )$ and choose $\delta _ { p }$ from a high quantile of the corresponding signal on successful trajectories. Conversely, for a lower-bound condition $p ( x _ { t } ) : f _ { p } ( x _ { t } ) \geq \delta _ { p }$ , we use $\rho _ { p } ( x _ { t } ) = f _ { p } ( x _ { t } ) - \delta _ { p }$ and select $\delta _ { p }$ from a suitably low quantile of successful behavior. The use of tolerant quantiles is particularly important for predicates that appear inside temporal operators such as $G _ { [ a , b ] }$ , whose robustness is determined by a minimum over the evaluation window. A threshold fitted to an extreme sample can make the specification unnecessarily sensitive to brief deviations. We therefore use high but non-extreme quantiles for error-type quantities, such as the 95th percentile, and low quantiles for floor-type quantities. The same principle is applied to temporal parameters: event delays observed in successful trajectories are summarized with empirical quantiles to obtain embodiment-specific temporal bounds.

## 4.4 EXPERT-CONSISTENCY FILTERING

VLM-generated formulas can be syntactically valid but incompatible with target-robot behavior. After grounding, we therefore evaluate each formula on the disjoint held-out set $\mathcal { D } _ { \mathrm { f i l t e r } } .$ . Let $r _ { i j } = \rho ( \phi _ { i } , \tau _ { j } )$ be full-trajectory robustness of grounded formula $\phi _ { i }$ on held-out trajectory $\tau _ { j }$ . We retain

$$
\begin{array} { r } { \phi _ { i } \in \Phi _ { \mathrm { k e e p } } \iff Q _ { \alpha } \big ( \{ r _ { i j } : \tau _ { j } \in \mathcal { D } _ { \mathrm { f l t e r } } \} \big ) \geq 0 , } \end{array}\tag{5}
$$

with $\alpha = 0 . 1 0$ quantile in the current experiments. For PushCube task, all seven formulas pass the $Q _ { 0 . 1 0 } \geq 0 \mathrm { t e s t }$ . Thus, this stage acts as held-out validation rather than aggressive pruning.

## 4.5 QUANTITATIVE STL AND SIGN-PRESERVING SMOOTH SEMANTICS

To preserve logical satisfaction while using Boltzmann-weighted reductions, for $z \in \mathbb { R } ^ { K } , \beta > 0$ and a nonempty index set $A \subseteq \{ 1 , \ldots , K \}$ , define $\begin{array} { r } { \mathrm { B M i n } _ { \beta } ( z ; A ) = \frac { \sum _ { i \in A } e ^ { - \beta z _ { i } } z _ { i } } { \sum _ { j \in A } e ^ { - \beta z _ { j } } } } \end{array}$ ; BMax<sub>β</sub> uses positive exponents. When min<sub>i</sub> $z _ { i } ~ < ~ 0$ , smin $\overset { \mathrm { s p } } { \underset { \beta } { \ast } } ( z )$ applies $\mathrm { B M i n } _ { \beta }$ only to negative coordinates; when min<sub>i</sub> $z _ { i } > 0$ , it uses all coordinates; otherwise it returns zero. Symmetrically, $\mathrm { s m a x } _ { \beta } ^ { \mathrm { s p } } ( z )$ uses positive coordinates when max<sub>i</sub> $z _ { i } > 0$ , all coordinates when max<sub>i</sub> $z _ { i } < 0$ , and returns zero otherwise. These operators preserve the satisfaction boundary:

$$
\operatorname { s m i n } _ { \beta } ^ { \mathrm { s p } } ( z ) \geq 0 \iff \operatorname * { m i n } _ { i } z _ { i } \geq 0 , \qquad \operatorname { s m a x } _ { \beta } ^ { \mathrm { s p } } ( \overline { { z } } ) \geq 0 \iff \operatorname * { m a x } _ { i } z _ { i } \geq 0 .\tag{6}
$$

## 4.6 TWO-TIMESCALE TEMPORAL REWARD

## 4.6.1 LOCAL ROLLING-WINDOW ROBUSTNESS

Long-horizon STL specifications are useful for describing the complete task, but their robustness is not always suitable as a dense reward during policy learning. We therefore use short-horizon specifications to provide local feedback over the recent trajectory history. Let span(ϕ) denote the temporal span required to evaluate a specification $\phi .$ Given a local horizon $H _ { \mathrm { l o c a l } } .$ , we define $\Phi _ { \mathrm { l o c a l } } \bar { = } \{ \phi _ { i } \in \bar { \Phi } _ { \mathrm { k e e p } } : \mathrm { s p a n } ( \phi _ { i } ) \leq H _ { \mathrm { l o c a l } } \bar  \}$ . At time t, each local specification is evaluated over the trailing window $\dot { W _ { t } } \dot { = } \left[ \operatorname* { m a x } ( 0 , t - H _ { \mathrm { l o c a l } } \dot { ) } , t \right]$ . For each $\phi _ { i } \in \Phi _ { \mathrm { l o c a l } }$ , we compute its quantitative robustness $\rho _ { \phi _ { i } } ^ { \mathrm { s p } } ( W _ { t } )$ over $W _ { t }$ using the sign-preserving smooth semantics. Because robustness magnitudes can differ substantially across formulas, we normalize each specification using a scale $s _ { i } = Q _ { \alpha } ( | \rho ^ { \mathrm { s p } } \phi _ { i } ( W _ { t } ) | )$ , estimated over $\tau \in \mathcal { D } \mathrm { g r o u n d }$ , with an empirical quantile. The normalized robustness is $\bar { \rho } _ { \phi _ { i } } ( W _ { t } ) = \operatorname { t a n h } \left( \rho _ { \phi _ { i } } ^ { \mathrm { s p } } ( W _ { t } ) / s _ { i } \right)$ , which bounds each contribution to [−1, 1] and prevents formulas with naturally larger robustness magnitudes from dominating the reward. The local STL reward is the mean normalized robustness across the retained short-horizon specifications:

$$
r _ { t } ^ { \mathrm { l o c a l } } = \frac { 1 } { | \Phi _ { \mathrm { l o c a l } } | } \sum _ { \phi _ { i } \in \Phi _ { \mathrm { l o c a l } } } \bar { \rho } _ { \phi _ { i } } ( W _ { t } ) .\tag{7}
$$

Each formula first retains its own temporal semantics and satisfaction boundary, after which the normalized robustness values are combined to provide dense policy feedback. Specifications whose intrinsic temporal span exceeds this horizon are excluded from the local term and handled separately through the long-horizon causal progress monitor described next.

## 4.6.2 LONG-HORIZON CAUSAL PROGRESS

Short-horizon robustness provides dense feedback, but it does not by itself capture the full temporal structure of a task. A long specification may encode a sequence of events that unfolds over most of an episode, and its robustness can remain uninformative until the relevant future events occur. We therefore use a separate causal progress signal for long-horizon temporal structure.

From the retained specification bank, we select a long-horizon backbone specification $\phi _ { \mathrm { b a c k b o n e } } \in$ $\Phi _ { \mathrm { k e e p } }$ whose temporal structure represents the main progression of the task. The backbone is monitored causally: at time t, the monitor only uses the realized trajectory prefix $x _ { 0 : t }$ and does not access future states. We represent the backbone as an ordered sequence of $K$ semantic stages, $p _ { 1 } \to p _ { 2 } \to \cdots \to p _ { K }$ together with the temporal bounds inherited from the grounded STL specification. The monitor maintains the furthest valid prefix reached so far, $b _ { t } \in \mathsf { \bar { \{ 0 , 1 , \ldots , K \} } }$ where $b _ { t } = k$ indicates that the first k stages have been satisfied in the required order and within their corresponding temporal constraints.

The progress reward $r _ { t } ^ { \mathrm { p r o g r e s s } } = ( b _ { t } - b _ { t - 1 } ) / K$ is issued only when the policy reaches a new valid stage. Because $b _ { t }$ stores the furthest valid prefix reached so far, completed stages cannot be rewarded repeatedly, and therefore $\begin{array} { r } { \sum _ { t } r _ { t } ^ { \mathrm { p r o g r e s s } } \le 1 } \end{array}$ . This construction provides sparse but temporally meaningful credit for completing valid task stages, while the rolling-window robustness term supplies dense local feedback. The final reward combines the two signals:

$$
r _ { t } = \lambda _ { L } r _ { t } ^ { \mathrm { l o c a l } } + \lambda _ { P } r _ { t } ^ { \mathrm { p r o g r e s s } } .\tag{8}
$$

This separation between local rolling-window robustness and long-horizon causal progress is used only for the manipulation tasks. For quadruped locomotion, we use only the local rolling-window robustness, since locomotion is primarily periodic rather than sequential and does not naturally decompose into a fixed progression of task stages.

## 5 EXPERIMENTS

We evaluate Video2STL in two robot-learning domains with different temporal structures: quadruped locomotion and object manipulation. These domains provide complementary test cases for the proposed representation. Quadruped locomotion is approximately periodic with desired behavior characterized by recurring contact and stability patterns over time. Manipulation tasks, in contrast, typically consist of a sequence of semantically distinct stages, such as approaching an object, interacting with it, reaching a goal configuration, and maintaining the resulting state. Across both domains, the high-level pipeline is unchanged. Moreover, our main comparison in both domains contrasts three approaches to reward design: (i) a task-specific hand-engineered dense reward, (ii) Text2Reward (implemented using GPT-5.6 as the underlying language model), which directly generates executable reward code from a natural-language task description, and (iii) $V i d e o 2 S T L ,$ which uses an explicit temporal specification as the intermediate representation (Tao et al., 2025; Caluwaerts et al., 2023; Xie et al., 2024). All methods use the same policy architecture, optimization procedure, training budget, environment configuration, and evaluation protocol; only the reward formulation is changed.

Table 1: Quadruped locomotion performance across commanded forward velocities. Survival and success are reported as percentages, and CoT denotes cost of transportation. Higher is better for survival and success; lower is better for CoT.
<table><tr><td></td><td colspan="3">Video2STL (Qwen 3.8)</td><td colspan="3">Video2STL (GPT-5.6)</td><td colspan="3">Text2Reward</td><td colspan="3">Heuristic</td></tr><tr><td> $\mathbf { v _ { x } } \left( \mathbf { m } / \mathbf { s } \right)$ </td><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT↓</td><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT ↓</td><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT↓</td><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT↓</td></tr><tr><td>0.3</td><td>100%</td><td>100%</td><td> $2 . 6 3 \pm 0 . 0 8$ </td><td>100%</td><td>100%</td><td> $2 . 6 3 \pm 0 . 0 6$ </td><td>100%</td><td>100%</td><td> $0 . 9 1 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 2 0 \pm 0 . 0 0$ </td></tr><tr><td>0.5</td><td>100%</td><td>100%</td><td> $1 . 8 6 \pm 0 . 0 3$ </td><td>100%</td><td>100%</td><td> $1 . 8 0 \pm 0 . 0 4$ </td><td>100%</td><td>100%</td><td> $0 . 8 0 \pm 0 . 0 0$ </td><td>100%</td><td>100%</td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>0.7</td><td>100%</td><td>100%</td><td> $1 . 5 8 \pm 0 . 0 2$ </td><td>100%</td><td>100%</td><td> $1 . 5 2 \pm 0 . 0 3$ </td><td>100%</td><td>100%</td><td> $0 . 7 5 \pm 0 . 0 0$ </td><td>100%</td><td>100%</td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>1.0</td><td>100%</td><td>100%</td><td> $1 . 3 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 2 6 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $0 . 7 4 \pm 0 . 0 0$ </td><td>100%</td><td>100%</td><td> $1 . 2 0 \pm 0 . 0 0$ </td></tr><tr><td>1.3</td><td>100%</td><td>100%</td><td> $1 . 2 2 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 1 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $0 . 7 9 \pm 0 . 0 0$ </td><td>100%</td><td>100%</td><td> $1 . 3 0 \pm 0 . 0 0$ </td></tr><tr><td>1.6</td><td>100%</td><td>100%</td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 1 2 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $0 . 8 7 \pm 0 . 0 0$ </td><td>100%</td><td>100%</td><td> $1 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>1.9</td><td>100%</td><td>100%</td><td>0.98 ± 0.01</td><td>100%</td><td>100%</td><td> $1 . 1 3 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td>1.00 ± 0.01  $1 . 0 7 + 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>2.0</td><td>100%</td><td>100%</td><td> $\bar { 1 . 0 0 } \pm \bar { 0 . 0 1 }$ </td><td>100%</td><td>100%</td><td>1.13 ± 0.01</td><td>100%</td><td>100%</td><td></td><td>100%</td><td>5%</td><td> $1 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>2.1</td><td>100%</td><td>100%</td><td> $1 . 0 2 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 1 5 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 1 4 \pm 0 . 0 1$ </td><td>100%</td><td>0%</td><td> $1 . 4 0 \pm 0 . 0 0$ </td></tr></table>

## 5.1 QUADRUPED LOCOMOTION

Robot and Simulation Setup. We evaluate locomotion on Google’s Barkour vb quadruped in MuJoCo XLA (MJX) (Caluwaerts et al., 2023; Todorov et al., 2012). The control timestep is 0.02 s and the policy outputs 12 normalized joint-position commands, which are converted to actuator torques by the robot’s low-level PD controller. The policy receives the commanded linear and angular velocities together with robot proprioception and the previous action. Policies are trained using the JAX-based PPO implementation provided by Brax (Schulman et al., 2017; Freeman et al., 2021).

Evaluation Metrics. We evaluate locomotion policies under a set of fixed commanded forward velocities with zero lateral and yaw commands. Each commanded velocity is evaluated using 20 independent rollouts of 500 simulation steps. The first 50 steps are treated as a warm-up period and excluded from command-tracking statistics. We report three task-level metrics. Survival rate measures the fraction of rollouts that reach the full evaluation horizon without triggering an environment termination condition. Velocity-tracking success measures whether the mean post-warm-up local forward velocity is within 15% of the commanded velocity. Specifically, for rollout k, $\bar { v } _ { x } ^ { ( k ) } =$ $( T - 5 0 ) ^ { - 1 } \textstyle \sum _ { t = 5 1 } ^ { T } v _ { x , \mathrm { l o c } } ^ { ( k ) } ( t )$ and the rollout is successful if $\left| \hat v _ { x } ^ { ( k ) } - v _ { x , \mathrm { c m d } } \right| \le 0 . 1 5 | v _ { x , \mathrm { c m d } } |$ . We also report the cost of transportation (CoT), $\mathrm { C o T } = E / ( m g d )$ , where E is the total energy consumed, m is the robot mass, g is gravitational acceleration, and d is the planar distance traveled.

Results. Across commanded forward velocities from 0.3 to 2.1 m/s, all methods achieve 100% survival, indicating that survival alone does not distinguish the learned controllers. In terms of task success, Video2STL with Qwen and with GPT-5.6 maintains 100% success across the entire velocity range, including the highest-speed commands. Text2Reward also achieves 100% success throughout the tested range. The heuristic/hand-engineered reward baseline shows a high-speed limitation, decreasing to 5% success at $2 . 0 \mathrm { m / s }$ and 0% at $2 . 1 \mathrm { m } / \mathrm { s }$ These results show that the Qwen- and GPT-5.6-generated specifications yield locomotion policies that remain robust across the full commanded-speed range.

The CoT reveals a complementary trade-off in locomotion efficiency. Text2Reward achieves the lowest CoT over most low- and mid-speed commands, whereas Video2STL-Qwen becomes increasingly competitive as velocity increases. In particular, at 1.9, 2.0, and 2.1 m/s, Video2STL-Qwen obtains CoT values of 0.98, 1.00, and 1.02, respectively, compared with 1.00, 1.07, and 1.14 for Text2Reward and 1.40 for the heuristic baseline. Video2STL-GPT-5.6 exhibits equal or lower CoT than the Qwen variant up to $\mathrm { 1 . 3 m / s }$ and slightly higher CoT at higher speeds, with 1.13, 1.13, and 1.15 at 1.9, 2.0, and 2.1 m/s. Overall, both VLM-generated specification sets yield full-range task success: Qwen provides the strongest high-speed efficiency, while Text2Reward remains more energy-efficient at lower velocities. The comparison of GPT-5.6, Gemini 3.1, and Qwen configurations is reported in Appendix A.1.1, Table 2.

![](images/b34b86f7991f3f99a3514eed3785d64dfc0b09b01178a917df070c7e38a18842.jpg)  
Figure 2: Manipulation results on ManiSkill3 tasks. We compare the native dense PPO reward, Text2Reward, and Video2STL on PushCube, StackCube, LiftPegUpright, and PlaceSphere.

## 5.2 ROBOT MANIPULATION

Tasks and Simulation Setup. We evaluate manipulation in ManiSkill3 (Tao et al., 2025) using four tasks: PUSHCUBE-V1, STACKCUBE-V1, LIFTPEGUPRIGHT-V1, and PLACESPHERE-V1. Together, these tasks cover distinct forms of interaction, including planar pushing, grasp-and-stack behavior, orientation-dependent object manipulation, and object placement. All manipulation policies use state observations and are trained with the PPO implementation provided with ManiSkill.

Evaluation Metrics. We evaluate all methods using the native task success conditions provided by the corresponding ManiSkill environments. We report success-once, the fraction of episodes in which the native success condition is reached at least once, and success-at-end, the fraction of episodes satisfying the native success condition in the final state. We evaluate all policies using the same evaluation seeds and report the mean and standard deviation across 128 episodes.

Results. Averaged over the four tasks, Video2STL reaches 85.8% success-once and 67.0% success-at end, compared with 81.5%/59.5% for native dense PPO and 65.0%/42.3% for Text2Reward (Fig. 2). On PushCube, Video2STL achieves 95%/93%, retaining nearly all successes. On StackCube, it reaches 98%/95%, well above native PPO (73%/52%) and on par with Text2Reward (98%/96%). The largest gain is on PlaceSphere (91%/75% vs. 55%/55% for native PPO), which Text2Reward fails to solve. LiftPegUpright is the hardest task to retain for all methods: native PPO and Text2Reward also lose about half of their successes (98%/47% and 97%/50%). The benchmark’s check only requires the peg to be upright near the table, even while still grasped, whereas the specification extracted from the demonstration additionally requires the peg to be released and remain static. Video2STL thus optimizes a stricter, unassisted goal: it learns the reorientation (59% success-once) but often releases the tall, narrow peg before it settles (5% success-at-end), which we trace to a grounded Upright tolerance (15.9<sup>◦</sup>) looser than the peg’s tipping angle (11.8<sup>◦</sup>). Overall, grounded temporal specifications yield a learning signal competitive with native dense rewards and stronger than direct reward generation on three of four tasks.

## 6 CONCLUSION

We presented Video2STL, a framework that converts videos into interpretable temporal task specifications and uses them to guide robot learning across different embodiments. Across quadruped locomotion and four manipulation tasks, Video2STL demonstrated that video-derived temporal structure can provide effective learning signals without relying on manually engineered dense rewards. The results suggest that explicit temporal specifications provide a promising intermediate representation between visual demonstrations and robot learning, offering both behavioral interpretability and a structured mechanism for transferring task semantics across embodiments.

## AI USE STATEMENT

In this work, we used generative AI tools for code generation and debugging, figure generation and refinement, manuscript drafting, and LaTeX assistance. Generative AI is also part of the proposed methodology, where VLMs extract semantic event traces from videos and generate parametric STL specifications. We did not use generative AI to fabricate experimental data or quantitative results. All

AI-generated code was inspected and tested, figures were manually reviewed for technical accuracy, and all manuscript text and claims were checked by the authors. We take responsibility for the final content of this work produced with the aid of generative AI.

## REFERENCES

Eugene Asarin, Alexandre Donzé, Oded Maler, and Dejan Nickovic. Parametric identification of temporal properties. In International Conference on Runtime Verification, 2011.

Merve Atasever, Cagan Bakirci, Alfredo Reina Corona, Keyan Azbijari, and Jyotirmoy V. Deshmukh. Learning gait-aware quadruped locomotion with temporal logic specifications. arXiv:2607.00442, 2026.

Ken Caluwaerts, Atil Iscen, J Chase Kew, Wenhao Yu, Tingnan Zhang, Daniel Freeman, Kuang-Huei Lee, Lisa Lee, Stefano Saliceti, Vincent Zhuang, et al. Barkour: Benchmarking animal-level agility with quadruped robots. arXiv:2305.14654, 2023.

David DeFazio, Yohei Hayamizu, and Shiqi Zhang. Learning quadruped locomotion policies using logical rules. In Proceedings of the International Conference on Automated Planning and Scheduling, 2024.

Alexandre Donzé and Oded Maler. Robust satisfaction of temporal logic over real-valued signals. In Formal Modeling and Analysis of Timed Systems, volume 6246 of Lecture Notes in Computer Science, pp. 92–106. Springer, 2010. doi: 10.1007/978-3-642-15297-9\_9.

Georgios E. Fainekos and George J. Pappas. Robustness of temporal logic specifications for continuous-time signals. Theoretical Computer Science, 410(42):4262–4291, 2009. doi: 10.1016/j.tcs.2009.06.021.

C Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax–a differentiable physics engine for large scale rigid body simulation. arXiv:2106.13281, 2021.

Zhaoyuan Gu, Yuntian Zhao, Yipu Chen, Rongming Guo, Jennifer K Leestma, Gregory S Sawicki, and Ye Zhao. Robust-locomotion-by-logic: Perturbation-resilient bipedal locomotion via signal temporal logic guided model predictive control. IEEE Transactions on Robotics, 2025.

Mohammadhosein Hasanbeig, Daniel Kroening, and Alessandro Abate. Deep reinforcement learning with temporal logics. In International Conference on Formal Modeling and Analysis of Timed Systems, 2020.

Joseph Humphreys and Chengxu Zhou. Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion. Nature Machine Intelligence, 7(7):1141–1153, 2025.

Parv Kapoor, Anand Balakrishnan, and Jyotirmoy V Deshmukh. Model-based reinforcement learning from signal temporal logic specifications. arXiv:2011.04950, 2020.

Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. Reward design with language models. In International Conference on Learning Representations, 2023.

Jiayi Li, Tao Lu, Xiaoge Cao, Yinghao Cai, and Shuo Wang. Meta-imitation learning by watching video demonstrations. In International Conference on Learning Representations, 2022.

Xiao Li, Cristian-Ioan Vasile, and Calin Belta. Reinforcement learning with temporal logic rewards. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2017.

Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, and Ankit Shah. Grounding complex natural language commands for temporal tasks in unseen environments. In Conference on Robotic Learning, 2023.

YuXuan Liu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Imitation from observation: Learning to imitate behaviors from raw video via context translation. In IEEE International Conference on Robotics and Automation, 2018.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. VIP: Towards universal visual reward and representation via value-implicit pre-training. In International Conference on Learning Representations, 2023.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. In International Conference on Learning Representations, 2024.

Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, Osbert Bastani, Dinesh Jayaraman, Wenhao Yu, Tingnan Zhang, Dorsa Sadigh, and Fei Xia. Vision language models are in-context value learners. In International Conference on Learning Representations, 2025.

Oded Maler and Dejan Nickovic. Monitoring temporal properties of continuous signals. In Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems, pp. 152–166. Springer, 2004. doi: 10.1007/978-3-540-30206-3\_12.

Aniruddh Puranic, Jyotirmoy Deshmukh, and Stefanos Nikolaidis. Learning from demonstrations using signal temporal logic. In Conference on Robotic Learning, 2021.

Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, and David Lindner. Visionlanguage models are zero-shot reward models for reinforcement learning. In International Conference on Learning Representations, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.

Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. Time-contrastive networks: Self-supervised learning from video. In IEEE International Conference on Robotics and Automation, 2018.

Sumedh A. Sontakke, Jesse Zhang, Sébastien M. R. Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, and Laurent Itti. RoboCLIP: One demonstration is enough to learn robot policies. In Neural Information Processing Systems, 2023.

Sanjana Srivastava, Kangrui Wang, Yung-Chieh Chan, Tianyuan Dai, Manling Li, Ruohan Zhang, Mengdi Xu, Jiajun Wu, and Li Fei-Fei. ROSETTA: Constructing code-based reward from uncon strained language preference. In International Conference on Learning Representations, 2026.

Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.

Harish Venkataraman, Derya Aksaray, and Peter Seiler. Tractable reinforcement learning of signal temporal logic objectives. In Proceedings ofthe 2nd Conference on Learningfor Dynamics and Control, 2020.

Yufei Wang, Zhanyi Sun, Jesse Zhang, Zhou Xian, Erdem Biyik, David Held, and Zackory Erickson. RL-VLM-f: Reinforcement learning from vision language foundation model feedback. In International Conference on Machine Learning, 2024.

Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, and Tao Yu. Text2Reward: Reward shaping with language models for reinforcement learning. In International Conference on Learning Representations, 2024.

Tianhe Yu, Chelsea Finn, Sudeep Dasari, Annie Xie, Tianhao Zhang, Pieter Abbeel, and Sergey Levine. One-shot imitation from observing humans via domain-adaptive meta-learning. In Robotics: Science and Systems, 2018.

Runhao Zeng, Dingjie Zhou, Qiwei Liang, Junlin Liu, Hui Li, Changxin Huang, Jianqiang Li, Xiping Hu, and Fuchun Sun. Video2reward: Generating reward function from videos for legged robot behavior learning. In European Conference on Artificial Intelligence, 2024.

## A APPENDIX

## A.1 QUADRUPED LOCOMOTION

## A.1.1 COMPARISON OF VLM CONFIGURATIONS

Table 2 reports locomotion performance for the three VLM configurations. Their specifications and implementation settings are detailed below.

Table 2: Quadruped locomotion performance using STL specifications generated by different VLMs. Survival and success are reported as percentages, and CoT denotes cost of transportation. Higher is better for survival and success; lower is better for CoT.
<table><tr><td rowspan="2"> $\mathbf { v } _ { \mathbf { x } } \left( \mathbf { m } / \mathbf { s } \right)$ </td><td colspan="3">GPT-5.6</td><td colspan="3">Gemini 3.1</td><td colspan="3">Qwen 3.8</td></tr><tr><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT↓</td><td>Surv. ↑</td><td>Succ. ↑</td><td>CoT↓</td><td>Surv. ↑</td><td>Succ. ↑</td><td> $\mathbf { C o T } \downarrow$ </td></tr><tr><td>0.3</td><td>100%</td><td>100%</td><td> $2 . 6 3 \pm 0 . 0 6$ </td><td>100%</td><td>25%</td><td> $2 . 3 7 \pm 0 . 0 9$ </td><td>100%</td><td>100%</td><td> $2 . 6 3 \pm 0 . 0 8$ </td></tr><tr><td>0.5</td><td>100%</td><td>100%</td><td> $1 . 8 0 \pm 0 . 0 4$ </td><td>100%</td><td>100%</td><td> $1 . 9 7 \pm 0 . 0 4$ </td><td>100%</td><td>100%</td><td> $1 . 8 6 \pm 0 . 0 3$ </td></tr><tr><td>0.7</td><td>100%</td><td>100%</td><td> $1 . 5 2 \pm 0 . 0 3$ </td><td>100%</td><td>100%</td><td> $1 . 6 6 \pm 0 . 0 3$ </td><td>100%</td><td>100%</td><td> $1 . 5 8 \pm 0 . 0 2$ </td></tr><tr><td>1.0</td><td>100%</td><td>100%</td><td> $1 . 2 6 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 2 6 \pm 0 . 0 2$ </td><td>100%</td><td>100%</td><td> $1 . 3 7 \pm 0 . 0 1$ </td></tr><tr><td>1.3</td><td>100%</td><td>100%</td><td> $1 . 1 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 1 4 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td>1.22 ± 0.01</td></tr><tr><td>1.6</td><td>100%</td><td>100%</td><td> $1 . 1 2 \pm 0 . 0 1$ </td><td>100%</td><td>40%</td><td>1.16 ± 0.01</td><td>100%</td><td>100%</td><td> $\operatorname* { m o x } _ { 0 } \pm \mathrm { u . u 1 }$  0.92 ± 0.01</td></tr><tr><td>1.9</td><td>100%</td><td>100%</td><td> $1 . 1 3 \pm 0 . 0 1$ </td><td>100%</td><td>0%</td><td> $1 . 1 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $0 . 9 8 \pm 0 . 0 1$ </td></tr><tr><td>2.0</td><td>100%</td><td>100%</td><td> $1 . 1 3 \pm 0 . 0 1$ </td><td>95%</td><td>0%</td><td> $1 . 1 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td>2.1</td><td>100%</td><td>100%</td><td> $1 . 1 5 \pm 0 . 0 1$ </td><td>100%</td><td>0%</td><td> $1 . 1 7 \pm 0 . 0 1$ </td><td>100%</td><td>100%</td><td> $1 . 0 2 \pm 0 . 0 1$ </td></tr></table>

## A.1.2 REWARD CONSTRUCTION

For locomotion, the source observation is a video of a dog running on treadmill. (available at here) The VLM extracts embodiment-independent locomotion semantics such as limb contact, swing, touchdown and liftoff events, support structure, trunk stability, and periodic leg motion.

GPT-5.6. The semantic trace and the specification bank were produced by GPT-5.6 in a single conversation in which the video remained in context. The bank contains 11 candidates. $\varphi _ { 1 1 }$ relies on Periodic\_leg\_motion, a predicate that is not grounded in the reward implementation, so it was not implemented. Each of the other ten candidates was screened on rollouts of the trot expert (50 trajectories, 500 steps each, $v _ { x } \in [ 0 . 5 , 1 . 6 ] \mathrm { m } / \mathrm { s } )$ , ignoring the first 50 steps of every trajectory, and kept only if the pooled fifth percentile of its per-step robustness was at least zero. Four candidates passed this test $( \varphi _ { 5 } , \varphi _ { 6 } , \varphi _ { 9 } , \varphi _ { 1 0 } )$ . The formulas are listed in Table 3 and the settings in Table 4.

Qwen. Qwen (qwen3.8-max-0902) generated stages 1–2, with the video retained in stage 2’s context; GPT-5.6 generated the reward integration. Of 15 candidates, $\varphi _ { 1 1 }$ was rejected for using disjunction. The remaining candidates were filtered using 50 trot-expert trajectories of 500 steps, $v _ { x } \in [ 0 . 5 , 1 . 6 ]$ m/s, discarding the first 50 steps. A candidate was retained when the fifth percentile of its pooled per-step monitor outputs was non-negative, leaving seven candidates. Tables 5 and 6 give the formulas and settings.

Gemini. The Stage-2 output contained six candidate PSTL specifications $\left( \varphi _ { 1 } - \varphi _ { 6 } \right)$ . Parameters with matching counterparts in the supplied Barkour configuration used those values; the remaining numerical groundings retained the existing demonstration-specific settings. The candidates and the additional torque-safety and velocity-tracking specifications were screened on 50 rollouts of the supplied expert, each lasting 500 steps with $v _ { x } \sim \mathcal { U } ( 0 . 5 , 1 . 6 )$ m/s and zero lateral and yaw commands. After discarding the first 50 steps of each rollout, a specification was retained when the pooled fifth percentile of its smooth robustness was nonnegative. Filtering acted on each complete specification. All six Gemini candidates, torque safety, and forward tracking were retained; lateral and yaw tracking were excluded.

## A.1.3 HYPERPARAMETERS AND DOMAIN RANDOMIZATION

We train each policy for 400M environment steps using an unroll length of 30, 32 minibatches, 4 updates per batch, discount factor $\gamma = 0 . 9 5 5$ , learning rate $1 . 5 \times 1 0 ^ { - 4 }$ , entropy coefficient 0.004, 8192 parallel environments, and a batch size of 256. All reward variants use the same policy architecture and PPO optimization settings.

Table 3: GPT-5.6 specifications and final weights. Gray formulas were filtered out; $\varphi _ { 1 1 }$ was rejected before training. $D _ { \ell } , L _ { \ell } , C _ { \ell } , S _ { \ell }$ denote touchdown, liftoff, contact, and swing. $\mathcal { R } ( p ) =$ ${ \bf G } _ { [ 0 , H ] } ( { \bf F } _ { [ 0 , h _ { c } ] } p )$ and $\bar { \cal B } ( p , q ) = { \bf G } _ { [ 0 , H ] } ( p \to { \bf F } _ { [ 0 , h _ { s } ] } q )$ . Z, P denote trunk-height and pitch stability. All tanh scales are $s _ { i } = 1$
<table><tr><td>ID</td><td>Formula</td><td> $\mathbf { w _ { i } }$ </td></tr><tr><td>41</td><td> $\textstyle \bigwedge _ { \ell \in { \mathcal { L } } } { \mathcal { R } } ( D _ { \ell } )$ </td><td>0</td></tr><tr><td>42</td><td> $\textstyle \bigwedge _ { \ell \in { \mathcal { L } } } { \mathcal { R } } ( L _ { \ell } )$ </td><td>0</td></tr><tr><td>43</td><td> $\mathcal { R } ( \tilde { D } _ { \mathrm { F L } } \wedge \tilde { D } _ { \mathrm { H R } } ) \wedge \mathcal { R } ( D _ { \mathrm { F R } } \wedge D _ { \mathrm { H L } } )$ </td><td>0</td></tr><tr><td>44</td><td> $\mathcal { R } ( L _ { \mathrm { F L } } \land L _ { \mathrm { H R } } ) \land \mathcal { R } ( L _ { \mathrm { F R } } \land L _ { \mathrm { H L } } )$ </td><td>0</td></tr><tr><td>45</td><td> $\mathcal { R } ( C _ { \mathrm { F L } } \wedge C _ { \mathrm { H R } } ) \wedge \mathcal { R } ( C _ { \mathrm { F R } } \wedge C _ { \mathrm { H L } } )$ </td><td>1</td></tr><tr><td>46</td><td> ${ \mathcal { R } } ( S _ { \mathrm { F L } } \land S _ { \mathrm { H R } } ) \land { \mathcal { R } } ( S _ { \mathrm { F R } } \land S _ { \mathrm { H L } } )$ </td><td>1</td></tr><tr><td>47</td><td> $\tilde { \mathcal { B } } ( \dot { D } _ { \mathrm { F L } } , D _ { \mathrm { F R } } ) \mathrm { ~ \ddot { / } ~ } \tilde { \mathcal { B } } ( \dot { D } _ { \mathrm { F R } } , D _ { \mathrm { F L } } ) \mathrm { ~ \ddot { \wedge } ~ } \tilde { \mathcal { B } } ( D _ { \mathrm { H L } } , D _ { \mathrm { H R } } ) \mathrm { ~ \wedge ~ } \tilde { \mathcal { B } } ( D _ { \mathrm { H R } } , D _ { \mathrm { H L } } )$ </td><td>0</td></tr><tr><td>48</td><td> ${ \mathcal B } ( S _ { \mathrm { F L } } , S _ { \mathrm { H L } } ) \wedge { \mathcal B } ( S _ { \mathrm { H L } } , S _ { \mathrm { F L } } ) \wedge { \mathcal B } ( S _ { \mathrm { F R } } , S _ { \mathrm { H R } } ) \wedge { \mathcal B } ( S _ { \mathrm { H R } } , S _ { \mathrm { F R } } )$ </td><td>0</td></tr><tr><td>49</td><td> ${ \bf G } _ { [ 0 , H ] } Z \wedge { \bf G } _ { [ 0 , H ] } P$ </td><td>1</td></tr><tr><td>410</td><td> $\mathbf { G } _ { [ 0 , H ] } ^ { } \big ( s u p p o r t \_ c o u n t \_ a t \_ l e a s t ( 2 ) \big )$ </td><td>1</td></tr><tr><td>411</td><td> $\textstyle \bigwedge _ { \ell \in { \mathcal { L } } } { \mathbf { \dot { G } } } _ { [ 0 , H ] }$  (Periodic_leg_motion(l)) (rejected)</td><td></td></tr></table>

Table 4: GPT-5.6 grounding and reward parameters. All values are fixed in the reward implementation, except the support count, which is specified by the VLM, and the filtering rule. Triples follow walk/trot/bound mode order; $\Delta t = 0 . { \bar { 0 } } 2 { \mathrm { s } }$
<table><tr><td>Parameter</td><td>Role</td><td>Value</td></tr><tr><td colspan="3">Temporal history-window setting, all modes</td></tr><tr><td>H  $h _ { c }$   $h _ { s }$  Mode hysteresis</td><td>recurrence bound response bound walk/trot entry, exit</td><td>30 steps (29, 20, 15) steps (14, 10, 7) steps 0.72, 0.65 m/s</td></tr><tr><td>Predicates</td><td>trot/bound entry, exit</td><td>1.55, 1.45 m/s</td></tr><tr><td>Contact / swing thresholds foot clearance Contact / swing scales</td><td>robustness normalization</td><td>0.001, 0.025 m 0.020, 0.025 m</td></tr><tr><td>Trunk-height tolerance Trunk-pitch tolerance k</td><td>relative to nominal reset height angular bound</td><td>0.080 m 0.30 rad</td></tr><tr><td>Invalid-event robustness</td><td>minimum support count unavailable predecessor sample</td><td>2 -8</td></tr><tr><td>Aggregation, reward, and filtering</td><td></td><td></td></tr><tr><td>Smoothing temperatures Si</td><td>atomic / temporal / formula</td><td>0.08, 0.12, 0.10</td></tr></table>

We additionally apply domain randomization to friction and actuator parameters. Friction is sampled in the range (0.6, 1.4), while actuator gain and bias perturbations are sampled in the range (−5, 5).

## A.1.4 SOURCES VIDEOS & VLM PROMPTS FOR QUADRUPED LOCOMOTION

We use two sequential VLM prompts for quadruped locomotion. The first extracts an embodimentindependent semantic locomotion trace directly from an animal video. The second receives only this structured trace and converts it into candidate parametric STL specifications.

## Stage 1: Video to Semantic Locomotion Trace

Table 5: Qwen specifications and final weights. Gray formulas were filtered out; $\varphi _ { 1 1 }$ was rejected before training. $\tilde { D _ { \ell } } , L _ { \ell } , C _ { \ell } , S _ { \ell }$ denote touchdown, liftoff, contact, and swing. $\mathcal { R } ( p ) { \bf \dot { = } { \bf G } } _ { [ 0 , H ] } ( { \bf F } _ { [ 0 , h _ { c } ] } p )$ and ${ \cal B } ( p , q ) = { \bf G } _ { [ 0 , H ] } ( p \to { \bf F } _ { [ 0 , h _ { s } ] } q )$ . Z, P, R denote trunk-height, pitch, and roll stability. All tanh scales are $s _ { i } = 1$
<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=12>Formula                             一</td><td rowspan=1 colspan=1> $\mathbf { w _ { i } }$ </td></tr><tr><td rowspan=7 colspan=1>41424344454647</td><td rowspan=5 colspan=12> $\mathcal { R } ( D _ { \mathrm { H R } } ) \wedge \mathcal { R } ( L _ { \mathrm { H R } } )$  $\mathcal { R } ( D _ { \mathrm { F R } } ) \wedge \mathcal { R } ( L _ { \mathrm { F R } } )$  ${ \mathcal { R } } ( D _ { \mathrm { H L } } ) \wedge { \mathcal { R } } ( L _ { \mathrm { H L } } )$  ${ \mathcal { R } } ( D _ { \mathrm { F L } } ) \wedge { \mathcal { R } } ( L _ { \mathrm { F L } } )$  $B ( D _ { \mathrm { H R } } , D _ { \mathrm { F R } } )$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1> $B ( D _ { \mathrm { H L } } , D _ {</td><td rowspan=1 colspan=1>\mathrm { F L } } )$ </td><td rowspan=1 colspan=2>FL)</td><td></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>B(DF</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>4849φ10</td><td rowspan=1 colspan=1> $\Vec { B } ( D _ { \mathrm { H R }</td><td rowspan=1 colspan=10>} , D _ { \mathrm { H L } } ) \wedge \Vec { B } ( D _ { \mathrm { H L } } , D _ { \mathrm { H R } } )$ </td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>01</td></tr><tr><td rowspan=1 colspan=1> ${ \mathcal { R } } ( C _ { \mathrm { H R</td><td rowspan=1 colspan=11>} } \land C _ { \mathrm { F R } } )$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=12> ${ \mathcal { R } } { \dot { ( } { C } _ { \mathrm { H L } } } \wedge C _ { \mathrm { F L } } \big )$  $\mathbf { G } _ { [ 0 , H ] } C _ { \mathrm { H R } } \lor \mathbf { G } _ { [ 0 , H ] } C _ { \mathrm { H R } } ( \mathrm { r e j e c t e d } )$  ${ \bf G } _ { [ 0 , H ] } ^ { \prime } Z \wedge { \bf G } _ { [ 0 , H ] } { \cal P } { \mathrm { \Large \wedge } } { \bf G } _ { [ 0 , H ] } R$  $\mathbf { G } _ { [ 0 , H ] } ^ { ' } \big ( s u p p o r t \_ c o u n t \_ \_ l e a s t ( 2 ) \big )$  $\textstyle \bigwedge _ { \ell \in { \mathcal { L } } } { \mathcal { R } } ( C _ { \ell } )$  $\textstyle \bigwedge _ { \ell \in { \mathcal { L } } } { \mathcal { R } } ( S _ { \ell } )$ </td><td rowspan=3 colspan=1>0一1111</td></tr><tr><td rowspan=1 colspan=1>411</td></tr><tr><td rowspan=1 colspan=1>412413414415</td></tr></table>

Table 6: Qwen grounding and reward parameters, supplied by the stage-3 configuration except the VLM-specified support count and the filtering rule. Triples follow walk/trot/bound mode order; $\Delta t = 0 . 0 2 \mathrm { s }$
<table><tr><td>Parameter</td><td>Role</td><td>Value</td></tr><tr><td colspan="3">Temporal H</td></tr><tr><td> $h _ { c }$   $h _ { s }$  Mode hysteresis</td><td>history-window setting, all modes recurrence bound response bound walk/trot entry, exit</td><td>30 steps (29, 20, 15) steps (14, 10, 7) steps 0.72, 0.65 m/s</td></tr><tr><td>Predicates</td><td>trot/bound entry, exit</td><td>1.55, 1.45 m/s</td></tr><tr><td>Contact / swing scales Trunk-height tolerance</td><td>foot clearance robustness normalization relative to nominal reset height</td><td>0.001, 0.025 m 0.020, 0.025 m</td></tr><tr><td>Trunk-pitch / roll tolerances</td><td></td><td>0.080 m</td></tr><tr><td></td><td>angular bounds</td><td>0.30, 0.30 rad</td></tr><tr><td>k</td><td>minimum support count</td><td>2</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Invalid-event robustness</td><td>unavailable predecessor sample</td><td>-8</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Aggregation, reward, and filtering</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Smoothing temperatures</td><td>atomic / temporal / formula</td><td>0.08, 0.12, 0.10</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $s _ { i }$ </td><td>specification tanh scale</td><td>1.0</td></tr><tr><td>Candidate aggregate</td><td>before warmup gate</td><td> $\sum _ { \textbf { * } \sim } w _ { i } \operatorname { t a n h } ( \rho _ { i } / s _ { i } ) / \sum _ { i } w _ { i }$ </td></tr><tr><td>Specification weight</td><td>within total STL reward</td><td>1.0</td></tr><tr><td>Tmax</td><td>joint torque limit</td><td>18Nm</td></tr><tr><td> $w _ { \mathrm { s a f e } } , \alpha _ { \mathrm { s a f e } }$ </td><td>safety weight and tanh scale</td><td>0.25, 0.25</td></tr><tr><td> $\gamma _ { \tau }$ </td><td>torque-effort coefficient</td><td>10-6</td></tr><tr><td>Reward scales</td><td>total STL / linear tracking / angular tracking</td><td>1.0, 1.0, 0.5</td></tr><tr><td>Action-rate scale</td><td>action-change penalty</td><td>-0.01</td></tr><tr><td>Tracking parameter</td><td></td><td>0.25</td></tr><tr><td></td><td>tracking_sigma</td><td></td></tr><tr><td>Filter rule</td><td>retain candidate with pooled fifth percentile</td><td> $q _ { 0 . 0 5 } \geq 0$ </td></tr></table>

System Role:   
Act as an expert Vision-Language Model specializing in Biomechanics and Robotic Locomotion.   
Context & Objective:   
Analyze the provided video of a real quadruped animal performing locomotion. Your goal is to   
extract an embodiment-independent semantic description of the temporal locomotion   
structure. This analysis will later be used to construct Signal Temporal Logic (STL)

Table 7: Gemini specifications and final aggregation priors. Gray formulas were filtered out. $D _ { \ell }$ and $L _ { \ell }$ denote touchdown and liftoff. $\mathcal { R } ( p ) = \mathbf { \bar { G } } _ { [ 0 , H ] } ( \mathbf { F } _ { [ 0 , h _ { c } ] } p )$ and $\begin{array} { r } { \dot { \boldsymbol { B } } ( p , q ) = \mathbf G _ { [ 0 , H ] } ( p \to \mathbf { F } _ { [ 0 , h _ { p } ] } q ) } \end{array}$ . Z and $P$ denote trunk-height and pitch stability. The first six rows are Gemini-generated; the remaining rows are implementation-supplied safety and tracking terms.
<table><tr><td>ID</td><td>Formula</td><td> $\mathbf { w _ { i } }$ </td></tr><tr><td>41 42 43</td><td> ${ \bf G } _ { [ 0 , H ] } Z \wedge { \bf G } _ { [ 0 , H ] } P$   $\mathbf { G } _ { [ 0 , H ] } ^ { } \big ( s u p p o r t \_ c o u n t \_ a t \_ l e a s t ( 2 ) \big )$   $\mathcal { R } \dot { ( } D _ { \mathrm { F L } } ) \wedge \mathcal { R } ( D _ { \mathrm { H R } } )$ </td><td>1 1 1</td></tr><tr><td>44 45 46</td><td> $\mathcal { R } ( L _ { \mathrm { H L } } )$   $\boldsymbol { B } ( D _ { \mathrm { H L } } , D _ { \mathrm { F L } } )$   $B ( D _ { \mathrm { H R } } , D _ { \mathrm { F R } } )$ </td><td>1 1 1</td></tr><tr><td> $\varphi _ { 7 }$ </td><td> $\mathbf { G } _ { [ 0 , H ] } \Big ( \bigwedge _ { j = 1 } ^ { 1 2 } ( | \tau _ { j } | \leq \tau _ { \operatorname* { m a x } } ) \Big )$ </td><td>1</td></tr><tr><td> $\varphi _ { 8 }$ </td><td></td><td></td></tr><tr><td>track  $_ y$ </td><td> $\mathbf { G } _ { [ 0 , W ] } \big ( | v _ { x } - v _ { x } ^ { \mathrm { c m d } } | \big ) \leq \epsilon _ { x } \big )$ </td><td>1</td></tr><tr><td>track yaw</td><td> $\mathbf { G } _ { [ 0 , W ] } \left( | v _ { y } - v _ { y } ^ { \mathrm { c m d } } | \leq \epsilon _ { y } \right)$   $\mathbf { G } _ { [ 0 , W ] } \left( | \omega _ { z } - \omega _ { z } ^ { \mathrm { c m d } } | \leq \epsilon _ { \omega } \right)$ </td><td>0 0</td></tr></table>

Table 8: Gemini grounding and reward parameters. Parenthesized pairs follow walk/trot mode order; unpaired values are shared; $\Delta t = 0 . 0 2 \mathrm { s }$
<table><tr><td rowspan=1 colspan=3>Parameter          Role                                  Value</td></tr><tr><td rowspan=2 colspan=3>Temporalouter always bound                    (30, 24) stepsrecurrence bound                      25 stepsphase-response bound                 13 stepsvelocity/yaw tracking bound           (30, 24) stepstrot entry / walk return                 0.72, 0.65 m/s</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Predicates</td><td rowspan=2 colspan=1>contact clearance thresholdcontact robustness scaleopposite signed marginsexact contact transitionsreference trunk heightheight tolerance / margin scaleabsolute trunk-pitch tolerancepitch robustness scalesupport count (kth-largest margin)forward tracking toleranceexcluded lateral/yaw tolerances</td><td rowspan=2 colspan=1>0.003 m0.002 m $C _ { \ell } / - C _ { \ell }$ adjacent samplestorso height at reset0.06 m / 0.02 m $( 8 ^ { \circ } , 7 ^ { \circ } )$  $\dot { 5 } ^ { \circ }$ 2(0.55, 0.60) m/s0.05 m/s; 0.05 rad/s</td></tr><tr><td rowspan=1 colspan=1> $z _ { \mathrm { t h } }$  $c$ Contact / swingTouchdown / liftoff $z _ { \mathrm { r e f } }$  $\delta _ { z } , s _ { z }$  $\theta _ { \mathrm { m a x } }$  $s _ { \theta }$  $k$  $\epsilon _ { x }$  $\epsilon _ { y } , \epsilon _ { \omega }$ </td></tr><tr><td rowspan=1 colspan=1>Aggregation, reward, a $\beta _ { \mathrm { a t o m } } , \beta _ { \mathrm { t e m p } }$  $\beta _ { \mathrm { g r o u p } }$  $w _ { i }$  $\alpha _ { g }$  $w _ { \mathrm { s a f e } } , w _ { \mathrm { t r a c k } }$  $w _ { \mathrm { p a t t e r n } }$  $\tau _ { \operatorname* { m a x } } , s _ { \tau }$ Forward multiplierTotal reward rateFilter rule</td><td rowspan=1 colspan=1>nd filteringBoolean / temporal sharpnessgroup Boltzmann soft-min sharpnessspecification priorgroup tanh scalesafety / tracking group weightsgait-pattern group weighttorque limit / margin scalemultiplier of $\rho _ { x } / \epsilon _ { x }$ sum of bounded group rewardspooled retention criterion</td><td rowspan=1 colspan=1>20, 200.51 active; 0 excluded11, 1(1.1,1.2)18Nm/5Nm1 $\sum _ { g } w _ { g } \operatorname { t a n h } ( \rho _ { g } / \alpha _ { g } )$  $q _ { 0 . 0 5 } \geq 0$ </td></tr></table>

specifications for a quadruped robot with different morphology, body dimensions, and   
controllers. Therefore, you must prioritize repeated, behaviorally important temporal   
structures over exact joint angles, limb trajectories, or incidental movements.   
Ontology & Permitted Vocabulary:   
Use only the following terms for your analysis.   
1. Body Roles:

```yaml
Trunk: The main body/torso of the quadruped.
Front_left: Front-left leg/foot.
Front_right: Front-right leg/foot.
Hind_left: Hind-left leg/foot.
Hind_right: Hind-right leg/foot.
Support_surface: The surface supporting locomotion, including ground or treadmill belt.
2. States & Events:
Contact(leg): Foot is visibly in contact with the support surface.
Swing(leg): Leg is in the aerial/swing phase.
Touchdown(leg): Transition from swing to contact.
Liftoff(leg): Transition from contact to swing.
Support_count_at_least(k): At least k feet are supporting the body.
Trunk_height_stable: The trunk maintains approximately bounded vertical variation over
repeated gait cycles.
Trunk_pitch_stable: The trunk maintains approximately bounded pitch variation over repeated
gait cycles.
Trunk_roll_stable: The trunk maintains approximately bounded roll variation over repeated
gait cycles.
Periodic_leg_motion(leg): The leg exhibits a repeated locomotion cycle.
3. Temporal Relations:
before
after
overlaps
alternates_with (Events repeatedly occur in alternating portions of the gait cycle.)
approximately_synchronous_with (Events repeatedly occur close together.)
phase_shifted_from (Consistent nonzero temporal offset between two legs.)
4. Observation Classes:
Essential_candidate: Visually supported, embodiment-independent temporal relationships
critical to describing the locomotion pattern.
Persistent_candidate: Body-level properties persisting through multiple gait cycles
(e.g., stable trunk height).
Incidental: Visually present but unnecessary for formal description
(e.g., head movement, tail motion, panting, a single irregular step).
Strict Constraints & Rules:
- Do NOT guess: If left/right identity or contacts cannot be reliably determined due to
viewpoint or occlusion, explicitly report the uncertainty.
Do NOT infer robot-specific metrics: Exclude joint angles, torques, actions, motor states,
joint velocities, actuator properties, or numerical thresholds.
Treadmill physics: If the animal is on a treadmill, its body may remain stationary in
image/world coordinates. Do NOT interpret this as zero forward velocity.
Task Requirements:
Analyze the video and determine:
1. Visible body parts.
2. Repeated stance/contact and swing behaviors.
3. Repeated touchdown and liftoff ordering.
4. Strongest inter-leg temporal relationships
(synchronization, alternation, phase offsets).
5. Persistent body-level regularities
(e.g., stable trunk pitch).
6. Ambiguities and occlusions.
Output Format:
Return valid JSON only, using exactly this structure.
Do not include markdown code blocks or conversational text outside the JSON.
"locomotion_context": {
"type": "overground | treadmill | uncertain",
"confidence": 0.0,
"evidence": "brief visual justification"
},
"visibility": {
"trunk": "clear | partial | unclear",
"front_left": "clear | partial | unclear",
```

```jsonl
"front_right": "clear | partial | unclear",
"hind_left": "clear | partial | unclear",
"hind_right": "clear | partial | unclear"
},
"observed_events": [
{
"event_id": "e1",
"predicate": "touchdown(front_left)",
"event_class":
"essential_candidate | persistent_candidate | incidental",
"confidence": 0.0,
"visual_evidence": "brief explanation"
}
],
"repeated_states": [
{
"predicate": "contact(front_left)",
"confidence": 0.0,
"visual_evidence": "brief explanation"
}
],
"inter_leg_relations": [
{
"relation_id": "r1",
"leg_1": "front_left",
"relation":
"before | after | overlaps | alternates_with |
approximately_synchronous_with | phase_shifted_from",
"leg_2": "hind_right",
"event_basis": "touchdown | liftoff | contact | swing",
"confidence": 0.0,
"evidence": "brief explanation"
}
],
"persistent_properties": [
{
"predicate":
"Trunk_height_stable | support_count_at_least(k)",
"confidence": 0.0,
"evidence": "brief explanation"
}
],
"uncertainties": [
"brief description of an ambiguity or occlusion"
]
}
```

## Stage 2: Semantic Trace to PSTL Specifications

System Role:   
Act as an expert Formal Methods Engineer specializing in Logic Synthesis for Robotic Control.   
Context & Objective:   
You are constructing Parametric Signal Temporal Logic (PSTL) specifications based on a   
semantic locomotion trace. This trace (provided as a JSON input) was extracted from a   
video of a real quadruped animal. Your objective is to translate this trace into   
embodiment-independent, candidate temporal specifications that describe the   
demonstrated repeated locomotion pattern. These specifications will eventually be   
grounded to a four-legged robot with a different morphology to generate a dense,   
continuous reward signal for reinforcement learning.   
Strict Constraints (The "Zero-Hallucination" Rule):   
- Trace-Dependent Reasoning: You must reason only from the supplied semantic trace.   
- No Invention: Do not invent contact events, predicates, or temporal relations that are   
absent from the input data. Use a predicate only if it is explicitly supported by the   
trace.   
- Symbolic Grounding: All temporal constants and numerical temporal parameters must remain   
purely symbolic (e.g., H\_episode, h\_cycle) for later numerical grounding during   
training.

```csv
Allowed PSTL Grammar:
You are restricted to the following exact textual grammar.
(Note: mu represents an atomic predicate; phi represents a sub-formula; H and h represent
symbolic temporal bounds).
Atomic predicate:
mu
Conjunction:
phi_1 AND phi_2
Bounded eventuality:
F_[0,H](mu)
Bounded persistence:
G_[0,H](mu)
Repeated-event requirement:
G_[0,H](F_[0,h](mu))
Response requirement:
G_[0,H](mu_1 -> F_[0,h](mu_2))
Repeated paired-event requirement:
G_[0,H](F_[0,h](mu_1 AND mu_2))
Ordered repeated response:
G_[0,H](mu_1 -> F_[0,h1](mu_2))
Note:
Conjunctions of independently valid formulas are permitted.
Output Format:
Return valid JSON only, using exactly the structure below.
Do not include markdown code blocks, conversational text, or explanations outside the JSON.
{
"candidate_specifications": [
"candidate_id": "phi_1",
"formula":
"PSTL formula using ONLY the allowed textual grammar",
"predicates_used": [
"instantiated_predicate_1"
],
"supporting_event_ids": [
"e1"
],
"supporting_relation_ids": [
"r1"
],
"excluded_incidental_event_ids": [
"e7"
],
"symbolic_parameters": [
"H_episode",
"h_cycle"
],
"semantic_complexity": 1,
"rationale":
"Brief explanation grounded purely in the input semantic trace"
}
],
"stage1_ambiguities_affecting_specification": [
"Brief description of how an ambiguity in the input trace impacts the formula"
```

## A.2 ROBOT MANIPULATION

## A.2.1 REWARD CONSTRUCTION

This section lists the parametric Signal Temporal Logic (PSTL) specifications generated by the VLM from the semantic traces of the manipulation demonstrations. These are the raw candidate specifications produced before numerical grounding using robot expert trajectories and before expertconsistency filtering. Thus, not every specification listed below is necessarily retained in the final reward.

For compactness, we denote the manipulated object by o, the robot end-effector by e, the support surface by s, and the receptacle by r. All temporal bounds, such as $H _ { \mathrm { t a s k } }$ and $h _ { \mathrm { g o a l } }$ , are symbolic at this stage and are subsequently grounded from expert trajectories.

## PushCube-v1

For PUSHCUBE, the VLM generated the following eight candidate specifications:

$$
\phi _ { 1 } : { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { N e a r } ( e , o ) \wedge { \bf F } _ { [ 0 , h _ { \mathrm { e n g a g e } } ] } \mathrm { E n g a g e d } ( e , o ) \right) ,\tag{9}
$$

$$
\phi _ { 2 } : \ : \ : \ : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) \right) ,\tag{10}
$$

$$
\phi _ { 3 } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { E n g a g e d } ( e , o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { p r o g r e s s } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) \right) \right) , \ : ( 1 1 )
$$

$$
\phi _ { 4 } : { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \mathrm { D i s p l a c e d } ( o ) \wedge { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \mathrm { T o w a r d } \_ { \mathrm { g o a l } } ( o ) ,\tag{12}
$$

$$
\begin{array} { r l } { \phi _ { 5 } : } & { { } \mathbf { G } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \mathrm { O n \_ s u p p o r t } ( o , s ) , } \end{array}\tag{13}
$$

$$
\phi _ { 6 } : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { A t \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { s e t t l e } } ] } \mathrm { S t a t i c } ( o ) \right) ,\tag{14}
$$

$$
\phi _ { 7 } : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \left( \mathrm { A t \_ g o a l } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { s e t t i c } } ] } \mathrm { S t a t i c } ( o ) \right) \right) ,\tag{15}
$$

$$
\phi _ { \mathrm { { s } } } : \mathbf { F } _ { [ 0 , H _ { \mathrm { f a s k } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) \right) \wedge \mathbf { G } _ { [ 0 , H _ { \mathrm { f a s k } } ] } \mathrm { O n \_ s u p p o r t } ( o , s ) .\tag{16}
$$

These specifications capture approach and engagement, object displacement, goal-directed progress, goal attainment, maintenance of support, and post-goal stationarity.

## StackCube-v1

For STACKCUBE, the VLM generated seven candidate specifications:

$$
\phi _ { 1 } : { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( { \mathrm { G r a s p e d } } ( o ) \wedge { \bf F } _ { [ 0 , h _ { \mathrm { d i s p l a c e } } ] } { \mathrm { D i s p l a c e d } } ( o ) \right) ,\tag{17}
$$

$$
\phi _ { 2 } : \ : \ : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { G r a s p e d } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e a c h } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) \right) \right) ,\tag{18}
$$

$$
\phi _ { 3 } : \ : \ : \mathrm { \bf { G } } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \ : ( \mathrm { T o w a r d \_ g o a l } ( o )  \mathrm { \bf { F } } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) ) ,\tag{19}
$$

$$
\phi _ { 4 } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \operatorname { O n \_ s u p p o r t } ( o , s ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e l e a s e } } ] } \left( \operatorname { R e l e a s e d } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { s e t t e } } ] } \mathrm { S t a t i c } ( o ) \right) \right) ,\tag{20}
$$

$$
\begin{array} { r l } { \phi _ { 5 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s u p p o r t } } ] } \mathrm { O n } \_ { \mathrm { s u p p o r t } } ( o , s ) \right) , } \end{array}\tag{21}
$$

$$
\begin{array} { r l } { \phi _ { 6 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s t a t i c } } ] } \mathrm { S t a t i c } ( o ) \right) , } \end{array}\tag{22}
$$

$$
\phi _ { 7 } : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { G r a s p e d } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { r e a c h } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { A t \_ g o a l } ( o ) \right) \right)
$$

$$
\land \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { O n \_ s u p p o r t } ( o , s ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { r e l e a s e } } ] } \left( \mathrm { R e l e a s e d } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { s e t t r e } } ] } \mathrm { S t a t i c } ( o ) \right) \right)
$$

$$
\land \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s u p p o r t } } ] } \mathrm { O n } _ { \_ \mathrm { s u p p o r t } } ( o , s ) \right) \land \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s t a t i c } } ] } \mathrm { S t a t i c } ( o ) \right) .\tag{23}
$$

The resulting specification set describes grasping and transport toward the target, supported placement, release, and persistent terminal stability.

LiftPegUpright-v1

For LIFTPEGUPRIGHT, the VLM generated seven candidate specifications:

$$
\phi _ { 1 } : { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { N e a r } ( e , o ) \wedge { \bf F } _ { [ 0 , h _ { \mathrm { r e a c h } } ] } \mathrm { E n g a g e d } ( e , o ) \right) ,\tag{24}
$$

$$
\phi _ { 2 } : { \bf F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( { \mathrm { G r a s p e d } } ( o ) \wedge { \bf F } _ { [ 0 , h _ { \mathrm { m a n i p } } ] } { \mathrm { D i s p l a c e d } } ( o ) \right) ,\tag{25}
$$

$$
\phi _ { 3 } : \ : \ : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { g o a l } } ] } \mathrm { U p r i g h t } ( o ) \right) ,\tag{26}
$$

$$
\phi _ { 4 } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { U p r i g h t } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { t e r m i n a l } } ] } \left( \mathrm { O n \_ s u p p o r t } ( o , s ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { t e r m i n a l } 2 } ] } \mathrm { A t \_ g o a l } ( o ) \right) \right)\tag{27}
$$

$$
\phi _ { 5 } : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { U p r i g h t } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e l e a s e } } ] } \left( \mathrm { R e l e a s e d } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { s t a s k e } } ] } \mathrm { S t a t i c } ( o ) \right) \right) ,\tag{28}
$$

$$
\begin{array} { r l } { \phi _ { 6 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { p e r s i s t } } ] } \mathrm { U p r i g h t } ( o ) \right) , } \end{array}\tag{29}
$$

$$
\begin{array} { r l } { \phi _ { 7 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { p e r s i s t } 2 } ] } \mathrm { O n } _ { - } \mathrm { s u p p o r t } ( o , s ) \right) . } \end{array}\tag{30}
$$

These candidates characterize approach and interaction with the peg, displacement, progress toward an upright configuration, supported placement, release followed by stability, and persistence of the terminal upright and supported states.

## PlaceSphere-v1

For PLACESPHERE, the VLM generated eight candidate specifications:

$$
\begin{array} { r l } { \phi _ { 1 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { N e a r } ( e , o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e a c h } } ] } \mathrm { G r a s p e d } ( o ) \right) , } \end{array}\tag{31}
$$

$$
\phi _ { 2 } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { G r a s p e d } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { d i s p l a c e } } ] } \left( \mathrm { D i s p l a c e d } ( o ) \land \mathbf { F } _ { [ 0 , h _ { \mathrm { p r o g r e s s } } ] } \mathrm { T o w a r d } _ { - } \mathrm { g o a l } ( o ) \right) \right)\tag{32}
$$

$$
\phi _ { 3 } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathrm { T o w a r d \_ g o a l } ( o ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { p l a c e } } ] } \left( \mathrm { I n \_ r e c e p t a c l e } ( o , r ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e l a s s e } } ] } \mathrm { R e l e a s e d } ( o ) \right) \right) .\tag{33}
$$

$$
\phi _ { 4 } : \quad { \bf G } _ { [ 0 , H _ { \mathrm { t a s k } } ] } ( \mathrm { R e l e a s e d } ( o )  { \bf F } _ { [ 0 , h _ { \mathrm { s e t t l e } } ] } \mathrm { S t a t i c } ( o ) ) ,\tag{34}
$$

$$
\phi _ { 5 } : \mathrm { { \bf ~ \vec { F } } } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( { \bf G } _ { [ 0 , h _ { \mathrm { i n _ { - } r e c e p t a c l e } } ] } \mathrm { { I n } } _ { \mathrm { - } } \mathrm { r e c e p t a c l e } ( o , r ) \right) ,\tag{35}
$$

$$
\begin{array} { r l } { \phi _ { 6 } : } & { { } \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s t a t i c } } ] } \mathrm { S t a t i c } ( o ) \right) , } \end{array}\tag{36}
$$

$$
\phi _ { 7 } : \ : \ : \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { i n , r e c e p t a c l e } } ] } \mathrm { I n } _ { \mathrm { - } } \mathrm { r e c e p t a c l e } ( o , r ) \right) \wedge \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \mathbf { G } _ { [ 0 , h _ { \mathrm { s t a t i c } } ] } \mathrm { S t a t i c } ( o ) \right) ,\tag{37}
$$

$$
\phi _ { \mathrm { B } } : \quad \mathbf { F } _ { [ 0 , H _ { \mathrm { t a s k } } ] } \left( \operatorname { N e a r } ( o , r ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { e n t e r } } ] } \left( \operatorname { I n } _ { - } \mathrm { r e c e p t a c l e } ( o , r ) \wedge \mathbf { F } _ { [ 0 , h _ { \mathrm { r e l e a s e l } } ] } \mathrm { R e l e a s e d } ( o ) \right) \right)\tag{38}
$$

The candidate set represents the demonstrated sequence of approaching and grasping the object, transporting it toward the receptacle, placing and releasing it, and maintaining containment and stationarity after placement.

## Example Grounding - PushCube:

Let $p _ { o } ( t ) , p _ { e } ( t )$ , and $p _ { g }$ denote the positions of the object, end-effector, and goal, respectively. We compute the end-effector distance $d _ { e o } ( t ) = \| p _ { e } ( t ) - p _ { o } ( t ) \| _ { 2 }$ , planar goal distance $d _ { g } ( t ) =$ $\| p _ { o , x y } ( t ) - p _ { g , x y } \| _ { 2 }$ , displacement $d _ { \mathrm { d i s p } } ( t ) \stackrel { . } { = } \lVert \stackrel { . . . } { p _ { o , x y } ( t ) } - p _ { o , x y } ( 0 ) \rVert _ { 2 }$ , cumulative goal progress $P ( t ) = d _ { g } ( 0 ) \bar { - } \bar { d } _ { g } ( t )$ , and instantaneous motion $m ( t ) \bar { \mathbf { \phi } } = \| p _ { o , x y } ( \bar { t } ) - p _ { o , x y } ( t - 1 ) \| _ { 2 }$ . The grounded atomic margins are then formulated as:

$$
\rho _ { \mathrm { N e a r } } ( t ) = \delta _ { \mathrm { n e a r } } - d _ { e o } ( t ) ,
$$

$$
\rho _ { \mathrm { T o w a r d G o a l } } ( t ) = P ( t ) - \delta _ { \mathrm { p r o g r e s s } } ,
$$

$$
\rho _ { \mathrm { D i s p l a c e d } } ( t ) = d _ { \mathrm { d i s p } } ( t ) - \delta _ { \mathrm { d i s p l a c e d } } ,
$$

$$
\rho _ { \mathrm { A t G o a l } } ( t ) = \delta _ { \mathrm { g o a l } } - d _ { g } ( t ) ,
$$

$$
\rho _ { \mathrm { O n S u p p o r t } } ( t ) = \delta _ { z } - | z _ { o } ( t ) - z _ { \mathrm { s u p p o r t } } | ,
$$

$$
\rho _ { \mathrm { S t a t i c } } ( t ) = \delta _ { \mathrm { s t a t i c } } - m ( t ) ,\tag{39}
$$

with $\rho _ { \mathrm { E n g a g e d } } ( t ) = \operatorname* { m i n } \{ \rho _ { \mathrm { N e a r } } ( t ) , m ( t ) - \delta _ { \mathrm { e n g a g e d } } \}$

The current PushCube parameters are $\delta _ { \mathrm { n e a r } } , \delta _ { \mathrm { e n g a g e d } } , \delta _ { \mathrm { d i s p l a c e d } } , \delta _ { \mathrm { p r o g r e s s } } , \delta _ { \mathrm { g o a l } } , \delta _ { z } , \delta _ { \mathrm { s t a t i c } }$ and temporal bounds are $H _ { \mathrm { t a s k } } , h _ { \mathrm { e n g a g e } } , h _ { \mathrm { p r o g r e s s } } , h _ { \mathrm { g o a l } } , \bar { h } _ { \mathrm { s e t t l e } } .$

## A.2.2 HYPERPARAMETERS

For all four tasks, we use the standard ManiSkill PPO implementation without modifying the PPO optimization procedure. The common hyperparameters are: learning rate $3 \times 1 0 ^ { - 4 }$ , discount factor $\gamma = 0 . 8 , \mathrm { G A E }$ parameter $\lambda = 0 . 9$ , PPO clipping coefficient 0.2, value-function coefficient 0.5, target KL divergence 0.1, 8 PPO update epochs per batch, and 32 minibatches. The actor and critic are three-layer MLPs with 256 hidden units per layer and tanh activations.

For PushCube, we use 2048 parallel environments, a rollout length of 20, and a total training budget of 2M environment steps.

For StackCube, we use 1024 parallel environments, a rollout length of 50, and a total training budget of 30M environment steps.

For LiftPegUpright, we use 1024 parallel environments, a rollout length of 50, and a total training budget of 25M environment steps.

For PlaceSphere, we use 1024 parallel environments, a rollout length of 50, and a total training budget of 10M environment steps.

## A.2.3 SOURCE VIDEOS & VLM PROMPTS FOR MANIPULATION

For each manipulation task, we record a single human demonstration. The demonstration is processed by the VLM using two sequential, task-agnostic prompts designed for general manipulation. The first prompt extracts a semantic event trace using the shared manipulation ontology, while the second maps this trace to a bank of parametric STL specifications. Importantly, the same prompts are applied to all manipulation tasks without any task-specific modification.

## Stage 1: Human Video to Semantic Manipulation Trace

System Role:   
Act as an expert Vision-Language Model specializing in Biomechanics, Robotic Manipulation,   
and Signal Temporal Logic (STL).   
Context & Objective:   
Analyze the provided video of a human performing a manipulation task. Your goal is to   
extract an embodiment-independent semantic description of the temporal manipulation   
structure. This analysis will be used to construct STL specifications for a robot with   
a different morphology, body dimensions, and controllers. Therefore, you must describe   
what happens semantically rather than reproducing the demonstrator’s exact trajectory,   
human kinematics, speed, or incidental movements.   
Ontology & Permitted Vocabulary:   
Use only the following roles and predicates in formal semantic fields.   
1. Entity Roles:   
Effector: The component directly interacting with the environment (e.g., human hand, robot   
end-effector).   
Object: A manipulable object whose state or position may change.   
Receptacle: A container or region into which an object can be placed.   
Handle: A graspable/interactable component used to manipulate an articulated object.   
Articulated\_part: A movable articulated component (e.g., drawer, door).   
Goal: A visually identifiable destination or goal region.   
Support: A surface or structure physically supporting an object.   
2. States & Events (Predicates):   
Near(a, b): Entity a is spatially close to entity b.   
Engaged(effector, x): The effector is actively interacting with x to change its state/motion.   
Grasped(object): The object is securely held by the effector.   
Released(object): The object is no longer grasped by the effector.   
Displaced(object): The object has moved meaningfully from its initial location.   
Toward\_goal(object): The object has made meaningful progress toward a visual goal.   
Upright(object): The object is in an approximately upright or standing orientation.   
At\_goal(object): The object occupies a visually identifiable goal region.   
On\_support(object, support): The object remains supported by the intended supporting surface.   
In\_receptacle(object, receptacle): The object occupies the interior/accepted region of a   
receptacle.   
Static(x): Entity x is approximately stationary.   
Open(articulated\_part): The articulated part is in an open configuration.   
3. Temporal Relations:   
before   
after   
overlaps   
persists\_until   
4. Observation Classes:

```csv
Essential_candidate: Visually supported, embodiment-independent temporal relationships
critical to describing the manipulation.
Terminal_candidate: An observed state/event that plausibly represents completion or the
resulting outcome.
Incidental: Visually present but unnecessary for formal description.
Strict Constraints & Rules:
1. A role, state, or event may be absent from the video.
Do not force every role or state to be populated.
2. Do NOT infer joint angles, velocities, or numerical thresholds.
3. Do NOT use human-specific variables
(e.g., hand_x, wrist_angle, elbow_pose)
inside semantic predicates.
4. Do NOT interpret exact human timing for the future robot.
Task Requirements:
1. Identify the visually relevant entities in the video and map them to the permitted
semantic roles.
2. Formulate the temporal relationships among observed events
using ONLY the permitted predicates.
3. Identify persistent properties only when the video gives clear evidence that a property
is maintained for a meaningful interval.
4. Note any critical ambiguities or visual occlusions.
Output Format:
Return valid JSON ONLY. Do not include markdown formatting,
code blocks, or conversational text outside the JSON structure.
Use exactly this schema:
"manipulation_context": {
"confidence": 0.0,
"evidence":
"Brief visual justification of the overall task being performed"
},
"entity_mapping": {
"effector":
"Description of what serves as the effector (or null)",
"object":
"Description of the object (or null)",
"receptacle":
"Description of the receptacle (or null)",
"handle":
"Description of the handle (or null)",
"articulated_part":
"Description of the articulated part (or null)",
"goal":
"Description of the goal region (or null)",
"support":
"Description of the support surface (or null)"
},
"visibility": {
"effector": "clear | partial | unclear | n/a",
"object": "clear | partial | unclear | n/a",
"receptacle": "clear | partial | unclear | n/a",
"handle": "clear | partial | unclear | n/a",
"articulated_part": "clear | partial | unclear | n/a",
"goal": "clear | partial | unclear | n/a",
"support": "clear | partial | unclear | n/a"
},
"observed_events": [
"event_id": "e1",
"state_or_event": "e.g., Grasped(object)",
"event_class":
"essential_candidate | terminal_candidate | incidental",
"confidence": 0.0,
"visual_evidence":
"Brief explanation of supporting visual evidence"
```

}   
],   
"temporal\_relations": [   
{   
"event\_1": "e1",   
"relation": "before | after | overlaps | persists\_until",   
"event\_2": "e2",   
"confidence": 0.0   
}   
],   
"persistent\_properties": [   
{   
"state\_or\_event":   
"e.g., On\_support(object, support)",   
"confidence": 0.0,   
"visual\_evidence": "Brief explanation"   
}   
],   
"missing\_concepts": [   
{   
"description":   
"Important observed concept not expressible by the strict ontology",   
"importance": "low | medium | high"   
}   
],   
"uncertainties": [   
"Brief statement describing an important ambiguity or occlusion"   
]   
}

## Stage 2: Semantic Manipulation Trace to PSTL Specifications

System Role:   
Act as an expert Formal Methods Engineer specializing in Logic Synthesis for Robotic Control.   
Context & Objective:   
You are constructing Parametric Signal Temporal Logic (PSTL)   
specifications based on a semantic manipulation trace.   
This trace (provided as a JSON input) was extracted from a human demonstration video.   
Your objective is to translate this trace into embodiment-independent, candidate temporal   
specifications that describe the demonstrated behavior. These specifications will   
eventually be grounded to a robot with a different morphology to generate a dense,   
continuous reward signal for reinforcement learning.   
Strict Constraints:   
- Trace-Dependent Reasoning: You must reason only from the supplied semantic trace.   
- No Invention: Do not invent contact events, predicates, or temporal relations that are   
absent from the input data. Use a predicate only if it is explicitly supported by the   
trace.   
Symbolic Grounding:   
All temporal constants and numerical temporal parameters must remain purely symbolic (e.g.,   
H\_task, h\_persist, h\_reach) for later numerical grounding during training.   
Allowed PSTL Grammar:   
You are restricted to the following exact textual grammar.   
(Note: mu represents an atomic predicate; phi represents a sub-formula; H and h represent   
symbolic temporal bounds).   
Atomic predicate:   
mu   
Conjunction:   
phi\_1 AND phi\_2   
Bounded eventuality:   
F\_[0,H](mu)   
Bounded persistence:   
G\_[0,H](mu)

```jsonl
Response requirement:
G_[0,H](mu_1 -> F_[0,h](mu_2))
Two-event ordered sequence:
F_[0,H](mu_1 AND F_[0,h1](mu_2))
Three-event ordered sequence:
F_[0,H](mu_1 AND F_[0,h1](mu_2 AND F_[0,h2](mu_3)))
Terminal stability:
F_[0,H](G_[0,h](mu))
Conjunctions of independently valid formulas are permitted.
Output Format:
Return valid JSON only, using exactly the structure below.
Do not include markdown code blocks, conversational text, or
explanations outside the JSON.
{
"ontology_sufficiency":
"sufficient | partially_sufficient | insufficient",
"ontology_note":
"Brief explanation of whether the available grammar and predicates adequately captured
the trace.",
"candidate_specifications": [
{
"candidate_id": "phi_1",
"description":
"Plain English description of the behavior this formula enforces",
"formula":
"PSTL formula using exactly the allowed textual grammar",
"predicates_used": [
"instantiated predicate (e.g., Grasped(object))"
],
"supporting_event_ids": [
"e1",
"e2"
],
"excluded_incidental_event_ids": [
"e4"
],
"ordered_events": [
"instantiated predicate 1",
"instantiated predicate 2"
],
"persistent_requirements": [],
"symbolic_parameters": [
"H_task",
"h1"
],
"semantic_complexity":
"low | moderate | high",
"rationale":
"Brief explanation grounded only in the provided semantic trace"
}
],
"stage1_ambiguities_affecting_specification": [
"Brief explanation of any ambiguity in the input trace that impacted formula construction
]
}
```