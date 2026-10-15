# WHAT HAPPENS TO THE GOAL? DISTINCT FAILURESOF POLICY ALIGNMENT IN GOAL MISGENERALIZA-TION

Julia Barth<sup>1,2</sup>, Armin Sommer<sup>1,2</sup>, Kunvar Thaman<sup>1</sup>, Jannik Schilling<sup>1</sup> <sup>1</sup>Pantheon Industries, <sup>2</sup>ETH Zurich

## ABSTRACT

Goal misgeneralization is a common failure mode in which an agent behaves competently out of distribution but does not pursue the intended goal. However, we still have a weak understanding of what happens to the intended goal in misgeneralizing agents. Across three Procgen environments and five robot tasks, we find three distinct forms of policy–goal misalignment: (1) functional target substitution, where behavior follows an alternative goal, (2) a policy shortcut, where behavior follows a learned action strategy, and (3) cue competition between intended and proxy goal cues. We distinguish these regimes by tracing where goal information remains available internally, how strongly it influences the policy, and what role it plays in action selection, and find that the goal is lost at different depths. Where the goal reaches the decision, as in cue competition, steering redirects the policy to the goal’s location, raising intended-goal reach from 16.0% to 95.3%. Where it reaches the policy but not the decision, only new downstream wiring recovers goal reach, from 31.6% to 84.3% in the environment with the action shortcut. Where it is inert, neither works. Our results show that goal misgeneralization is not a single failure of goal representation, but distinct failures of policy alignment at different depths, tightly connected to which recovery works.

## 1 INTRODUCTION

Reinforcement learning specifies the intended goal only indirectly, through rewards that mark it. Whenever a proxy co-occurs with the goal during training, the rewards cannot tell the two apart, and the agent can learn the proxy in its place. This is why goal misgeneralization is common in reinforcement learning, and it is dangerous because the agent stays competent and its misalignment only becomes visible once the co-occurrence breaks.

So far, goal misgeneralization has mainly been characterized behaviorally, in large language models (Shah et al., 2022; Maniyar et al., 2026), in toy reinforcement-learning environments such as Procgen (Langosco et al., 2022; Ramanauskas & S¸ ims¸ek, 2023) and in large pretrained control policies (Jucys et al., 2024; Zhou et al., 2025). Interpretability work has studied individual agents in these environments (Mini et al., 2023; Trinh et al., 2024), mostly to understand how they navigate or represent their goal, but not goal misgeneralization itself, and mitigation attempts intervene broadly during training (Langosco et al., 2022; Abdel Sadek et al., 2025; Nait El Barj & Sautory, 2024; Armstrong et al., 2023; Trinh et al., 2024) rather than after it. Studying goal misgeneralization in depth, from the inside and the outside, closes the gap between these lines of work, and we additionally carry it into a new and wider domain, robot control. The leading question we ask is therefore quite direct: what actually happens to the true goal in misgeneralized agents?

To answer it, we trace the goal through the agent from the outside and the inside, asking whether the goal is represented, whether the policy reads it, whether behavior is organized around it, and whether goal-directed behavior can be recovered at test time. We analyze three classic goal-misgeneralization environments from Procgen, MAZE, COIN and Maze II, which we call ATTRIBUTE MAZE, and five new robot tasks on a simulated arm and quadruped, such as PUSHTOMARKER (Figure 1, Figure 13).

What looks like one failure from the outside is three distinct failures of policy alignment, target substitution, a policy shortcut and cue competition, and the goal gets lost at different depths of the agent. We find that in MAZE, replicated in a simulated arm task, the goal is inert to the policy and a place substitutes for it. In PUSHTOMARKER a place substitutes for the goal as well, but the goal still reaches the policy weakly, which reacts to it without following it. This is the same depth at which an action shortcut bypasses the goal in COIN. In ATTRIBUTE MAZE, replicated in three more robot tasks, the goal reaches the decision as a cue or an object that competes with the proxy and loses. We find that which repair works follows the depth, not the failure type. Steering the existing policy redirects it to the goal’s location only where the goal reaches the decision. Relearning the readout recovers goal-directed behavior wherever the goal still reaches the policy, and where the goal is inert, neither repair works. Misgeneralized agents therefore cannot be understood from their behavior alone, as only knowing where the goal is lost tells us which repair can bring it back.

![](images/b81138a7aef353791e45729af7ef0cf095b4cf0f4824e55efb7d883770852318.jpg)  
Figure 1: All four agents fail alike from the outside, but inside the goal gets lost at different depths, and only a recovery that acts where the goal still survives brings it back. From left to right, the proxy (dashed ring) and goal (solid ring) in out-of-distribution frames, how far goal information travels through the agent, where MAZE and PUSHTOMARKER share a role but not a depth, and goal reach after steering at $h _ { \mathrm { f c } }$ and after a new readout from $h _ { \mathrm { p r e } } ,$ with partial recovery hatched and the control rewire dashed (all eight environments in Figure 13).

## Our main contributions are:

1. We trace the intended goal through misgeneralized agents in eight environments and find three distinct failures of policy alignment at three depths, including the first internal characterization of cue and object competition and its geometry (Section 3).

2. We add an internal view and deepen the behavioral one to better understand goal misgeneralization. We propose the Fisher information of the policy with respect to its internal representations to isolate a policy-directed view of the internals, which lets us characterize, for example, different styles of competition. It also sharpens a proxy characterization of Langosco et al. (2022), as the MAZE agent follows the top-right extreme of the current maze rather than a fixed corner.

3. To our knowledge, we are the first to relate which repair succeeds to where the goal is lost, and to recover goal-directed behavior at test time without further reinforcement learning or retraining the representation (Section 4).

4. We are the first to induce goal misgeneralization systematically in reinforcement learning for robot control and diagnose it in five simulated arm and quadruped tasks, where the failures recur.

## 2 SETUP

Problem formulation. We model each environment as an episodic Markov decision process $\mathcal { M } =$ $( S , \mathcal { A } , P , r , \rho _ { 0 } )$ . The agent observes a rendered image of the state, which we write as s, samples actions $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ , and the reward r marks an intended goal $g ^ { \star }$ . Goal misgeneralization arises when, on a training distribution $\Lambda _ { \mathrm { t r a i n } } ,$ reaching the goal coincides with reaching a proxy, so that the agent learns to follow the proxy instead of the goal on a shifted evaluation distribution $\Lambda _ { \mathrm { O O D } }$ , where the two no longer coincide (Langosco et al., 2022; Shah et al., 2022). We write $\Gamma ^ { \star } ( \tau ) = 1$ when a trajectory τ reaches the goal and $\Gamma ^ { \mathrm { p r o x } } ( \tau ) = 1$ when it reaches the proxy. Our main behavioral metric is the goal reach $\begin{array} { r } { \hat { J } ^ { \star } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Gamma ^ { \star } ( \tau _ { i } ) } \end{array}$ , the fraction of N rollouts that reach the goal. Formal setup and definitions are in Appendices B and F.1.

Environments. We study eight instances of goal misgeneralization, present four of them here and report the replications in Appendix C. Three are procedurally generated video games (Procgen, Cobbe et al. 2020, Appendix A). In MAZE, the cheese sits in the top-right corner during training, and at test time the agent walks to the corner instead of the cheese placed elsewhere (Langosco et al., 2022). In COIN, a side-scrolling game, the coin sits at the right end of the level during training, and at test time the agent runs to the right end and ignores a coin placed elsewhere (Cobbe et al., 2019; Langosco et al., 2022). In ATTRIBUTE MAZE, a $5 \times 5$ variant of Maze II (Langosco et al., 2022; Ramanauskas & S¸ ims¸ek, 2023), the rewarded object is a yellow line during training, so colour and shape are two cues for the same object, and at test time, when the cues are split across objects, the agent follows the yellow object instead of the line when both are equally far away. The fourth is a robot-arm task, PUSHTOMARKER, in which a simulated Franka arm pushes a cube onto a marker (Tao et al., 2025). During training, the marker sits at the centre of the table, and at test time the arm pushes the cube to the centre and ignores the marker placed elsewhere. Four further robot tasks, three on the same arm and one on a quadruped, replicate these failures (Appendices A.2 and A.3). In each environment, we compare three misgeneralized (GM) agents with three control (Non-GM) agents trained on diversified distributions (Appendix F).

Architecture. All Procgen agents are non-recurrent actor–critic networks with an IMPALA architecture (Espeholt et al., 2018) and all robot agents use a Nature CNN (Mnih et al., 2015), both trained online with PPO (Schulman et al., 2017) (training and implementation details in Appendix H, architectures in Appendix B.1). In both networks, convolutional layers produce feature maps $H ^ { ( k ) }$ whose average-pooled vectors $h ^ { ( k ) }$ we use for probing, and a fully connected layer maps the flattened last feature map $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ . Most of our analyses read $h _ { \mathrm { p r e } } .$ the flattened last feature map and the last representation that keeps the spatial layout of the scene, and $h _ { \mathrm { f c } } .$ , which the policy and value heads read directly (Appendix B.1). We write $\tilde { \pi } ( \cdot \mid h )$ for the policy that the downstream layers compute from a representation h, so that $\tilde { \pi } ( \cdot \mid h ( s ) ) \dot { = } \pi ( \cdot \mid s )$

## 3 WHAT HAPPENS TO THE INTENDED GOAL?

## 3.1 IS THE GOAL STILL THERE?

If the true goal gets replaced by a different goal, the simplest question we can ask is whether the true goal is still in the misgeneralized agent. We test this using linear decodability (Alain & Bengio, 2017), by fitting linear probes at several points in the agent (Appendix B.1) on configuration sets that separate constructed goal concepts, scored by AUC for binary and $R ^ { 2 }$ for continuous concepts. For example, a bearing probe tests whether the agent encodes where the goal lies relative to itself. Probe construction, controls (Hewitt & Liang, 2019), and further probes are detailed in Appendix C.1.

For MAZE, the cheese object is trivially decodable from the input already, so we probe cheese bearing. This needs a set on which the agent’s own position is decorrelated from the cheese bearing, since otherwise a bearing probe reads the GM agent’s position as a proxy (Appendix C.1). On this set the cheese bearing stays near chance at every site in GM agents (0.516 to 0.583) and below randomly initialized networks at $h _ { \mathrm { p r e } }$ (0.748), while control agents decode it well (0.932) (Figure 2a). Similarly, for PUSHTOMARKER the marker position is decodable at the level of a randomly initialized network in early layers and falls below it in the later layers (0.110 against 0.406 at $h _ { \mathrm { f c } } )$ , while control agents keep it (0.819) (Figure 2b). In both environments, therefore, the bearing of the cheese and the position of the marker cannot be linearly decoded beyond what a random network exposes. For COIN, the coin object is also largely decodable from the input, so we test coin bearing with the agent’s position and level progress regressed out. This bearing stays decodable in both cohorts from $h ^ { ( 2 ) }$ to $h _ { \mathrm { p r e } } ( R ^ { 2 }$ of 0.378 for GM and 0.306 for Non-GM at $h _ { \mathrm { p r e } } )$ , while randomly initialized networks and the agent’s position alone decode none of it, so we find linear traces of the coin inside the agent (Figure 2c). Lastly, for ATTRIBUTE MAZE, the goal cue, whether the reward-carrying object is the line, stays decodable in both cohorts alongside the proxy cue, most clearly in early layers (AUC 0.770 for GM and 0.701 for Non-GM at $h ^ { ( 2 ) }$ , against 0.577 for random networks), and comes close to random networks at $h _ { \mathrm { p r e } }$ (0.632 and 0.636 against 0.603). So in MAZE and PUSHTOMARKER the GM agents carry no linearly decodable goal information beyond what a randomly initialized network exposes, while in COIN and ATTRIBUTE MAZE the goal stays decodable in both cohorts. The results replicate in the other robot tasks (Appendix C.1). Our probes show what is linearly available, but they cannot rule out goal information in another form. Since information that is present is not necessarily used (Elazar et al., 2021), we next ask whether the policy reacts to changes of the goal, using world edits.

(a) Maze  
![](images/2bbe60a7d12ecc89be5d6e6c2c63a14801c7ee8905c3dfaf712469bc26afae15.jpg)

(b) PushToMarker  
![](images/aeb6bf029a192e5fa0cbb6aaa273dc0afc2af6eda7f991cd200bdf61853cba9e.jpg)

(c) Coin  
![](images/0e69e063780c29d23d8562029df41196ea30b92b02f1363ead2532aeb6723e28.jpg)

![](images/96f7909570dc26ed166b3ff05a7400b8e287f32dd77f51c4d3418d3a3d3f0c3e.jpg)  
GM Non-GM random init input control mean ± 95% CI  
Figure 2: Is the goal still there? Held-out probe AUC (PUSHTOMARKER, COIN: $R ^ { 2 } )$ across depth, 95% t-intervals. (a) Cheese bearing on a position-balanced set and (c) coin bearing with agent position and level progress regressed out, so that agent position is uninformative. Dashed: randomly initialized networks, dotted: input-level controls (agent position in a and c, pixels in b). Robot $\dot { H } ^ { ( 1 ) } , H ^ { ( 2 ) }$ are unpooled (Appendix C.1). Hollow Coin $h _ { \mathrm { f c } }$ points are not interpreted.

## 3.2 DOES THE POLICY READ OUT THE INTENDED GOAL?

We answer this question by intervening on the goal in the environment, from two perspectives: from the outside, we measure the change in behavior, and from the inside, whether the policy distribution moves along the corresponding representation displacement. A world edit e changes the goal in the level while keeping everything else fixed, and turns a state s into a matched state $s _ { e } ,$ , with $\Delta _ { e } f ( s ) = f ( s _ { e } ) - f ( s )$ for any quantity $f ,$ and $e = { \star } \operatorname { o r } e = p$ for an edit of the goal or of the proxy (Figure 18). In MAZE, COIN and PUSHTOMARKER, we relocate the goal object, the cheese, the coin or the marker, as in earlier interventions on the MAZE agent (Mini et al., 2023). In ATTRIBUTE MAZE, in contrast, goal and proxy cue sit on the same object, so instead of moving the object, we keep all locations fixed and swap one characteristic between the two candidates: the shape swap changes the goal cue $( e = \star )$ and the colour swap the proxy cue $( e = p )$ . All edits are then evaluated on banks of conflict states, in which the agent has to choose between the proxy and the intended goal (Appendices C.2 and E).

The immediate behavioral response is the change in the policy, which we measure with the KL divergence, $D _ { \mathrm { K L } } ( \pi ( \cdot \textit { | s } ) \| \pi ( \cdot \textit { | s } _ { e } ) )$ . The KL says whether the policy reacts but not in which direction, so we measure with signed follow whether the action mass moves toward the edited object,

$$
\operatorname { S F } _ { e } ( s ) = \pi { \big ( } A _ { e } ( s _ { e } ) \mid s _ { e } { \big ) } - \pi { \big ( } A _ { e } ( s _ { e } ) \mid s { \big ) } , \qquad \pi ( A \mid s ) = \sum _ { a \in A } \pi ( a \mid s ) ,\tag{1}
$$

where $A _ { e } ( s _ { e } )$ holds the actions on the shortest path to the edited object at its new place in MAZE and ATTRIBUTE MAZE and the coin-directed, leftward actions in COIN, while for the continuous robot policies signed follow measures how far the planned step turns toward the object’s new place (Appendix C.2). Inside the agent, the same edit moves the representation read by the policy by $\bar { \Delta _ { e } ( s ) } = \Delta _ { e } h ( s ) = h ( s _ { e } ) - \bar { h } ( s )$ . We measure how strongly the policy distribution moves along this displacement with the Fisher information of the downstream policy at $h = h ( s )$

$$
F _ { h } = \mathbb { E } _ { a \sim \tilde { \pi } ( \cdot \vert h ) } \Big [ \nabla _ { h } \log \tilde { \pi } ( a \mid h ) \big ( \nabla _ { h } \log \tilde { \pi } ( a \mid h ) \big ) ^ { \top } \Big ] , \qquad Q _ { e } ( s ) = \Delta _ { e } ( s ) ^ { \top } F _ { h } \Delta _ { e } ( s ) .\tag{2}
$$

We use the Fisher information because it is the metric that the policy itself induces on its representation. It measures a displacement by how much it changes the action distribution rather than by its

Euclidean length, and we use this Fisher lens throughout the paper. Up to second order, this Fisher sensitivity is exactly the KL divergence, measured from the inside (Amari, 1998; Martens, 2020),

$$
\begin{array} { r } { D _ { \mathrm { K L } } \big ( \tilde { \pi } ( \cdot  { | } h ) \big | \big | \tilde { \pi } ( \cdot  { | } h + t \Delta _ { e } ( s ) ) \big ) = \frac { t ^ { 2 } } { 2 } Q _ { e } ( s ) + o ( t ^ { 2 } ) , } \end{array}
$$

and we check this for finite edits in Appendix C.3. KL, $Q _ { e }$ and signed follow have different scales across environments, so we report each as the ratio of the GM to the Non-GM cohort mean.
<table><tr><td>Environment</td><td>Edit</td><td>KL ratio</td><td> $Q { \mathrm { ~ r a t i o ~ } } ( h _ { \mathrm { f c } } )$ </td><td>Signed-follow ratio</td></tr><tr><td>MAZE</td><td>Cheese</td><td> $0 . 0 0 9 \pm 0 . 0 1$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td><td> $- 0 . 0 0 2 \pm 0 . 0 0$ </td></tr><tr><td>PUSHTOMARKER</td><td>Marker</td><td> $0 . 2 0 6 \pm 0 . 0 7$ </td><td> $0 . 1 5 9 \pm 0 . 0 8$ </td><td> $0 . 0 1 7 \pm 0 . 0 5$ </td></tr><tr><td>COIN</td><td>Coin</td><td> $0 . 1 1 4 \pm 0 . 0 5$ </td><td> $0 . 0 6 3 \pm 0 . 0 3$ </td><td> $0 . 0 0 1 \pm 0 . 0 4$ </td></tr><tr><td>ATTRIBUTE MAZE</td><td>Shape</td><td> $0 . 2 0 0 \pm 0 . 0 9$ </td><td> $0 . 3 0 7 \pm 0 . 1 3$ </td><td> $0 . 5 3 3 \pm 0 . 1 0$ </td></tr></table>

Table 1: Policy response to a goal edit, as the ratio of GM to Non-GM agents. Each ratio divides the GM by the Non-GM mean over three seeds, with $Q$ at $h _ { \mathrm { f c } } ,$ so a ratio below 1 means GM agent react less. The ± is the SD, propagated to the ratio. Absolute values are in Tables 6 and 8.

We find that the outside and inside perspectives agree, through the KL and $Q$ ratios, and separate the environments by depth (Table 1). In MAZE (replicated in one robot task, Appendix C.2), the GM policy barely reacts to the cheese edit (KL ratio 0.009), so changes of the goal are read out by the policy neither behaviorally nor inside its representation. In PUSHTOMARKER, COIN and under the ATTRIBUTE MAZE shape swap, the goal edit moves the GM policy by similar fractions of the Non-GM reaction (0.206, 0.114 and 0.200). If we then ask whether this shift of action mass is directed at the goal, signed follow separates these three as well. In PUSHTOMARKER and COIN the policy reacts to the goal edit, but the shift is not directed at the goal (signed-follow ratios 0.017 and 0.001), so the policy is receptive to a change of the goal object but does not organize its longer-term behavior around it. In ATTRIBUTE MAZE, however, the shape swap still turns the policy toward the goal at about half the Non-GM turn (0.533). For the proxy colour swap, the GM policy reacts about fifty times more than the Non-GM policy, which barely follows the colour (KL 1.258 against 0.025, signed follow +0.463 against +0.047). So the goal does not reach the policy in MAZE, reaches it without directing it in PUSHTOMARKER and COIN, and enters the decision but loses to the proxy in ATTRIBUTE MAZE. The other four robot tasks fall into the same three groups (Appendix C.2). Sensitivity only captures the immediate action distribution, so we ask next what organizes behavior.

## 3.3 WHAT ORGANIZES BEHAVIOR INSTEAD?

Where the goal is fully inert, as in MAZE, the question is what actually replaces it: a directional shortcut, a memorized location, or a spatial concept. We first select detour states, on which every action that makes progress toward the top-right corner points neither up nor right. GM agents still choose these detour actions on 0.948 of the states, which rules out a directional shortcut (Figure 3a). We then open a path to a new top-right extreme, inspired by the corner edit of Mini et al. (2023), and ask which corner the agent follows. It reaches the new corner first in 0.748 of the rollouts and the old one never, and a new extreme placed away from the screen corner is still reached first in 0.792 of rollouts, against 0.114 for the old one, which rules out a memorized location (Figure 3a). Combined with the inert goal, this shows a functional target substitution, and we refer to Appendix C.4.1 for the full analysis. For MAZE, the substitute target is thus not a fixed place but a spatial relation, the top-right extreme of the maze. This refines Langosco et al. (2022), who classify this failure as a location proxy but do not test what underlies it.

A weakly wired goal, as in COIN, raises a different question: is this a weaker substitution, or is behavior organized by something other than a target? Two observations point to a fixed action rule. First, when the end of the level is made traversable, GM agents run past the former endpoint in 0.938 of the episodes that reach it and keep choosing rightward actions at the same rate after passing it, which reconfirms the observation of Langosco et al. (2022). Second, the coin’s weak influence carries no direction. When the coin moves, the change in the policy is spread over the actions in proportion to their probability. We measure this with the directed share

$$
T _ { \star } = \frac { \sum _ { a \in A _ { \star } } \vert \Delta _ { \star } \pi ( a \mid s ) \vert / \sum _ { a } \vert \Delta _ { \star } \pi ( a \mid s ) \vert } { \pi ( A _ { \star } \mid s ) } ,
$$

the share of the total variation of the edit that falls on the coin-directed actions $A _ { \star } ,$ , divided by the probability these actions held before the edit, so that $T _ { \star } ~ = ~ 1$ means no preferred direction (Appendix C.4.3). Coin-directed actions hold about 7% of the probability and receive about 9% of the change $( T _ { \star } = 1 . 2 1 0 )$ , close to randomly initialized networks $( T _ { \star } = 0 . 9 9 1 )$ and far from Non-GM agents $( T _ { \star } = 4 . 9 9 6 )$ (Figure 3b). The coin is therefore perceived as a change in the scene rather than as a goal that orients behavior. What organizes behavior is neither the goal nor a substitute target, but the simple rightward action rule that solved training, so goal misgeneralization in COIN is best described as a policy shortcut, in the sense of shortcut learning (Geirhos et al., 2020). A deeper analysis, including Fisher saliency maps, is given in Appendix C.4.3 and Appendix B.2.

(a)  
(b)  
![](images/f3323ec900b047357c956a9dd898ee28ce40384e4d9829efc6627afe6c41ec4b.jpg)

![](images/1c528cc30eee1427e4a232d1e05373b2db22de164bb8c416f92aec9e75b56899.jpg)  
GM Non-GM init mean ± 95% CI real table centre old pixel / posture  
Figure 3: What organizes behavior instead? (a) MAZE: fraction of states moving toward the corner or new corner (filled) versus the cheese or old corner (open). (b) COIN: directed share $T _ { \star }$ where $T _ { \star } = 1$ (dashed) marks no preferred direction. (c) PUSHTOMARKER: cube end positions with no shift, after shifting the arm base and after shifting the camera, between the real table centre (0) and the old point (1), with the GM fraction ending at the old point. Only the camera shift moves where GM agents push. (d) ATTRIBUTE MAZE: $S _ { \mathrm { f a v } }$ and $S _ { \mathrm { c o m p } }$ per swap, states (faint) and seed means (diamonds), push–pull quadrant shaded. Mean ± 95% CI over seeds.

In PUSHTOMARKER we find that a place substitutes for the goal as well, instead of a policy shortcut, but unlike in MAZE it is a fixed place in the camera image. When we shift the camera, GM agents push the cube to the point under the old centre pixel in 0.901 of episodes, while control agents push it onto the marker in 0.955 (Figure 3c). A shift of the arm base does not change where GM agent push (0.967 at the centre). A fixed motion pattern would not follow the camera, which rules out a policy shortcut, so PUSHTOMARKER is a target substitution. To characterize its weak reaction to the marker, beyond signed follow showing that the shifted action mass is not goal-directed (Table 1), we added a distractor, a disc of another colour, and found that it elicits the same reaction as the marker, so the marker is perceived as a change in the scene rather than as a goal (Appendix C.4.2).

When goal and proxy are both wired into the policy, as in ATTRIBUTE MAZE, the question becomes how the two interact. For a swap e that favours one candidate, we measure the change in log support of the favoured candidate relative to the neutral actions ${ \cal A } _ { 0 } ^ { e } ( s ) , { \cal S } _ { \mathrm { f a v } } ^ { e } ( s ) = \Delta _ { \epsilon }$ log $[ \pi ( \bar { A } _ { \mathrm { f a v } } ^ { e } ( \bar { s } )$ $s ) / \pi ( A _ { 0 } ^ { e } ( s ) \ \mid \ s ) ]$ , where $A _ { \mathrm { f a v } } ^ { e } ( s )$ holds the progress actions toward the favoured candidate, and $S _ { \mathrm { c o m p } } ^ { e } ( s )$ likewise for its competitor (Appendix C.4.4). In GM agents, both the shape swap and the colour swap push support toward their candidate and pull it from the competitor $( S _ { \mathrm { f a v } } = 1 . 3 6 8$ and $S _ { \mathrm { c o m p } } = - 1 . 7 2 1$ for shape, 2.001 and −2.354 for colour), while Non-GM agents show this push–pull only for shape (3.740 and −3.910, against 0.192 and −0.362 for colour) (Figure 3d, Appendix C.4.4). The two cues thus do not act independently but compete for the same choice.

We therefore find three distinct failures of policy alignment: a substitute target where the proxy is a location, an action rule where it is a movement, and competing cues where it is a property of the goal object, or two competing objects (Appendix C.4.5). How this competition is organized in ATTRIBUTE MAZE is what we examine next.

## 3.4 WHEN GOAL AND PROXY BOTH REACH THE POLICY, HOW DO THEY COMPETE?

ATTRIBUTE MAZE is an example of a regime in which goal and proxy both act on the policy and push and pull against each other in a competition. We now want to understand geometrically where this push and pull comes from, and why the proxy wins. For each conflict state, the shape swap and the colour swap move the representation by $\Delta _ { \star } ( s )$ and $\Delta _ { p } ( s )$ , and we compare such directions with the ordinary cosine cos<sub>E</sub> in raw activations and with its Fisher variant $\begin{array} { r } { \cos _ { F } ( a , b ) = a ^ { \top } F _ { h } b / \sqrt { a ^ { \top } F _ { h } a b ^ { \top } F _ { h } b } } \end{array}$ in the policy’s metric, which with $F _ { h }$ from Equation equation 2 only counts directions the policy actually reads. We find that the two cues push the policy in opposite directions (Figure 4a). In GM agents, cos ${ } _ { F } ( \Delta _ { \star } , \Delta _ { p } ) = - 0 . 9 1 9$ , so for the policy the two displacements are nearly antipodal, whereas control agents are opposed more weakly (−0.557) and randomly initialized networks not at all (+0.254).

![](images/d8a05aebf940c649dbc9fda3043b3229e7c1b7aca0178daae446cc0e0dfa5d17.jpg)

![](images/6fc41fb7fdeb02328e71bde6cb9e434b77a8934eca22d99dce25ceaf790e6cfa.jpg)  
Figure 4: How goal and proxy compete in ATTRIBUTE MAZE. (a) Cosine between the two cue displacements within a state. (b, c) Cosine between the competition directions of pairs of states, grouped by whether the goal candidate lies UP or RIGHT. (d) Pull of each cue on the shared axis, per state. Above the dashed line, the proxy pulls harder. The colour/shape pull ratio is 1.56 for GM and 0.07 for Non-GM. All at $h _ { \mathrm { f c } }$ , mean with 95% t-interval over three seeds.

This raises the question whether there is one global competition axis. We therefore take the competition direction ${ d _ { \mathrm { c o m p } } ( s ) = \Delta _ { p } ( s ) - \Delta _ { \star } ( s ) }$ in each state and compare it between states with the Euclidean and Fisher cosines from above (Figure 4b, c). In the Fisher metric, states of the same conflict share almost the same competition direction (0.942, against 0.148 at initialization), and states of the opposite conflict point the opposite way, while in raw activations the structure is visible but more heterogeneous and weaker (0.589). Thus, the policy reads the competition on one signed axis per conflict, which we call u<sub>comp</sub> (Appendix C.5.2).

Finally, we ask who is actually winning on this axis. We project each cue’s displacement onto $u _ { \mathrm { c o m p } }$ with $\alpha _ { F } ( \Delta ) = u _ { \mathrm { c o m p } } ^ { \top } F _ { h } \Delta / u _ { \mathrm { c o m p } } ^ { \top } F _ { h } u _ { \mathrm { c o m p } } .$ , which measures how far the cue pulls the state along it in the policy’s Fisher geometry. In GM agents, the colour cue pulls 1.564 times as hard as the shape cue, whereas in control agents the ratio is only 0.067 (Figure 4d). Randomly initialized networks already lean toward colour, so GM training keeps an initial preference that Non-GM training reverses (Appendix C.5.3). The goal is therefore not lost in this regime but still pulls on a shared axis, only more weakly than the proxy, which suggests steering along this axis, as we test next (Appendices C.5 and C.5.3). The same competition appears in graded form in the robot tasks (Appendix C.5.4).

## 4 CAN THE GOAL BE RECOVERED?

## 4.1 CAN THE GOAL BE RECOVERED THROUGH THE EXISTING WIRING?

The depth that Section 3 found for each environment, together with the geometry of Section 3.4, already tells us where steering can work. In MAZE, the goal is inert, and in COIN and PUSH-TOMARKER, it is only weakly read and does not direct behavior. In ATTRIBUTE MAZE, however, the goal is fully wired into the policy and loses its competition with the proxy on a shared axis, and this axis is something we can push on. To steer the four environments with one protocol, we contrast, at every state $s ,$ the representation of two counterfactual versions of $\operatorname { i t } , s ^ { \star }$ in which the evidence supports the goal and $s ^ { p }$ in which it supports the proxy. Their difference $d ( s ) = h _ { \mathrm { f c } } ( s ^ { \star } ) - h _ { \mathrm { f c } } ( s ^ { p } )$ is an additive steering vector (Turner et al., 2023b), along which we push the representation with dose $\beta ,$ $h _ { \mathrm { f c } } ^ { \prime } ( s _ { t } ) = h _ { \mathrm { f c } } ( \bar { s _ { t } } ) + \beta d ( s _ { t } )$ . In $\boldsymbol { \mathrm { M A Z E } } , s ^ { \star }$ and $s ^ { p }$ place the cheese at its position and in the top-right corner, in PUSHTOMARKER the marker at its position and at the table centre. In COIN, they place the coin at its position behind the agent and ahead of it on the shortcut path. In ATTRIBUTE MAZE,

(a-d)

![](images/edd80e2e4c665ae725b295ea0d9a2d369de7c50f765a6958d13e2dce49d517e8.jpg)  
(e)-- fixed at s0sign-gated-- Fisher-projected unsigned

Figure 5: Steering the existing wiring. Goal reach $\hat { J } ^ { \star }$ under steering at $h _ { \mathrm { f c } }$ as a function of the dose $\beta ,$ GM agents (top) and control agents (bottom). (a–d) State-dependent corrective, reversed and random pushes of equal norm. (e) ATTRIBUTE MAZE, shared-axis pushes (sign-gated, Fisherprojected, unsigned) against the state-dependent direction held fixed at $s _ { 0 } .$ , negative $\beta$ reversed. Grey bands and dotted lines show the unsteered baseline. Mean with 95% t-interval over three seeds.

$s ^ { \star }$ and $s ^ { p }$ are the colour and shape swaps, which put the yellow line on the goal and on the proxy candidate, so $d ( s )$ is the competition direction $d _ { \mathrm { c o m p } } ( s )$ of Section 3.4. We rebuild the direction as the agent moves, which makes the push state-dependent and allows for different geometries rather than one shared axis. As controls, we reverse it, use random pushes of equal norm, or none.

As we derived in Section 3.4, a simpler intervention is possible in ATTRIBUTE MAZE. As the policy reads the competition on one signed axis per conflict, we propose to replace the state-dependent direction by the identified shared axis $u _ { \mathrm { c o m p } } , h _ { \mathrm { f c } } ^ { \prime } ( s _ { t } ) = h _ { \mathrm { f c } } ( s _ { t } ) + \beta \sigma ( s _ { 0 } ) \| d ( s _ { 0 } ) \| _ { 2 } u _ { \mathrm { c o m p } } .$ , where $\sigma ( s _ { 0 } ) \in \{ - 1 , + 1 \}$ is the orientation of the conflict. This push is computed once at the start state $s _ { 0 }$ and held fixed, and its norm matches the state’s own direction, so that only the direction changes. We compare it with an unsigned push that drops σ and, as both are held fixed, with $d ( s _ { 0 } )$ held fixed, and with a Fisher-projected push, the component $\alpha _ { F }$ of $d ( s _ { 0 } )$ along $u _ { \mathrm { c o m p } }$ (Appendix D.1.2).

In ATTRIBUTE MAZE, we find that steering redirects the policy to the goal location: the GM agents’ goal reach rises from 0.160 to 0.953 at $\beta = 1$ , the reversed push drives it down and a random push leaves it unchanged (Figure 5d). With the goal cue removed, the push still takes the agent there in 0.913 of states, so it sets where the policy goes (Appendix D.1.1, Table 15). The shared axis carries this effect as well (Figure 5e). $\mathrm { A t } \ \beta = 0 . 2 5$ , the sign-gated push raises goal reach to 0.587, as much as $d ( s _ { 0 } )$ held fixed (0.560), and reversing it drives it close to zero. Without the sign $\sigma ( s )$ , the same axis helps one conflict orientation and hurts the other, so the unsigned push reaches only 0.300, and reversing it gives almost the same 0.307. The Fisher-projected push is weaker (0.560), partly because it is a smaller push, only 0.30 to 0.73 the size of the sign-gated one (Appendix D.1.2).

In GM agents in MAZE, PUSHTOMARKER and COIN, the same construction does not move the policy toward the goal: in MAZE the goal reach stays at most 0.002, in PUSHTOMARKER at most 0.018 and in COIN at most 0.027, and the corrective push is no more effective than a random one, and in PUSHTOMARKER it lowers GM pursuit of the table centre (0.925 to 0.628 at $\beta = 1 )$ without raising goal reach, although control agents do respond to it: the reversed push removes cheese pursuit in MAZE almost entirely (0.567 to 0.002), marker pursuit in PUSHTOMARKER (0.903 to 0.000, against 0.582 under random pushes), and likewise in COIN (Appendix D.1.1, Table 14).

The steering results are thus consistent with Section 3: steering the existing wiring redirects the policy to the goal’s location only where the goal still reaches the decision. In ATTRIBUTE MAZE, the goal lost a competition on an axis the policy already reads, so pushing along that axis turns the policy towards the goal’s location again. In MAZE, PUSHTOMARKER and COIN, there is no such axis to push on, as the goal information that reaches the policy is too weak in PUSHTOMARKER and COIN and not detectable in MAZE. Whether the goal information in the representation is still enough to learn an orientation towards the true goal is what we test next, by training new wiring. The same patterns replicate in the other environments (Appendix D.1).

![](images/c0d63bce916b8d36fe2a1417830c5da01faadd1f9b968e676f8078d4994c410a.jpg)

(b)  
![](images/a59fdb7548885beeba87b668e5b8ef8b4b7466e99a7a31c9c08882e202f7a85c.jpg)  
(c)

![](images/c2b7995403e58c6185e992012e89beeac7d42d549b190576d17878fdac9dcd72.jpg)  
(d)

![](images/e16b35b90fc4c506c50ae2611928095ab5fa6152bdd848747b07c0e2cab9bde1.jpg)  
Figure 6: Recovering the goal through new wiring. Goal reach $\hat { J } ^ { \star }$ after relearning the network downstream of $h _ { \mathrm { p r e } }$ or $h _ { \mathrm { f c } }$ by behavioral cloning, with oracle targets in MAZE and ATTRIBUTE MAZE and control-agent targets in PUSHTOMARKER and COIN. Remix is a fixed random map of $h _ { \mathrm { p r e } }$ with a trained head of the $h _ { \mathrm { f c } }$ size. DAgger (diamonds) adds three rounds of relabeling on the states the rewired policy visits, in PUSHTOMARKER with relabeling by the control agents. Dotted lines mark the floor, the higher of a random trunk rewired at the same site and chance. Grey numbers give the trained parameters. Points are means with 95% confidence intervals across agents, small points are individual agents.

## 4.2 CAN THE GOAL BE RECOVERED THROUGH NEW WIRING?

To understand at what depth the goal is truly lost, we learn a new wiring at two points in the architecture (Appendix B.1). We relearn either only the policy head on top of $h _ { \mathrm { f c } }$ (4k trained parameters in Procgen) or the head together with the fully connected layer that maps $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ (528k). In both cases, we train a freshly initialized copy with behavioral cloning (Pomerleau, 1991) on goal-directed actions, so that every agent receives the same target behavior and a failed recovery cannot be caused by exploration or credit assignment. In MAZE and ATTRIBUTE MAZE, the targets are oracle actions from a breadth-first search towards the true goal, while in COIN and PUSHTOMARKER, where no oracle exists, we use the actions of the control agents (control-agent targets for all environments in Appendix D.2). As baselines, we rewire the control agents in the same way, compare every rewire to the native reach of both cohorts and to a floor, the higher of a never-trained random trunk rewired in the same way and a goal-blind chance policy, and remove the confound of head size with a fixed random remix, which replaces the learned map from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ by a random one and trains only a head of the $h _ { \mathrm { f c } }$ size. A GM rewire counts as recovered near the control reach and as partial when clearly above the floor (full rule in Appendix D.2).

Where the goal information is seemingly inert for the policy, as in MAZE, a new wiring does not recover goal orientation (Figure 6a). GM agents reach the cheese on 0.116 of episodes from $h _ { \mathrm { p r e } }$ and 0.119 from $h _ { \mathrm { f c } }$ , below their native 0.219 and a uniform random walk (0.226). As behavioral cloning alone leaves even the control rewire below its native reach (0.308 against 0.347), we also relabel on the states the rewired policy visits (Ross et al., 2011), which restores the control agents (0.351) but leaves GM agents below 0.13 (Appendix D.2). The goal is thus not recovered by a new wiring, as expected from the inert goal in the probes, the Fisher sensitivity and the steering results.

In PUSHTOMARKER, where the goal reaches the policy only weakly, a new wiring recovers it partially (Figure 6b). From $h _ { \mathrm { p r e } } .$ GM agents push the cube onto the marker in 0.314 of conflict scenes, far above the random trunk (0.014) but only about 45% of the control rewire (0.702), which itself stays below its native reach (0.889). With on-policy relabeling the GM rewire reaches 0.661, against 0.314 with cloning alone (Appendix D.2, Table 19). From $h _ { \mathrm { f c } }$ , they reach 0.132.

In COIN, where the goal is only weakly wired and steering was unsuccessful, the question is whether a new wiring can use the goal information that still reaches the late layers. From $h _ { \mathrm { p r e } } ,$ the GM rewire reaches the coin in 0.843 of episodes (Figure 6c), close to the control rewire (0.888) and well above chance (0.625) and the native GM reach (0.316). Splitting the episodes by how they end shows where this comes from. A rewire on a never-trained random trunk chooses the coin over the level end almost as often as the GM rewire (0.962 against 0.987 of the episodes that end at either), but it dies or runs out of time more often. The trained GM trunk therefore mainly contributes the skill of getting through the level, not the recognition of the coin. From $h _ { \mathrm { f c } }$ , the GM rewire reaches 0.666, close to chance (0.625). This matches the probes, where coin bearing is decodable in both cohorts through $h _ { \mathrm { p r e } } .$ , also beyond the agent’s position. The remix shows where the information is lost, as a head on a random remix of GM $h _ { \mathrm { p r e } }$ reaches 0.676, as much as a head on the learned GM $h _ { \mathrm { f c } }$ , while the same remix of control $h _ { \mathrm { p r e } }$ keeps 0.825. So the wiring the GM agents learned between $h _ { \mathrm { p r e } }$ and the policy head seems to keep no more goal information than a random one. With the steering null, this shows that in COIN the goal is still represented but lost in the readout. Retraining only the last layer, which suffices for spurious correlations in Kirichenko et al. (2023), does not suffice here.

Finally, in ATTRIBUTE MAZE, where goal and proxy information both reach the decision, rewiring from $h _ { \mathrm { p r e } }$ allows recovery and brings GM agents close to the control rewire (0.871 against 0.908, native 0.358, Figure 6d). A random trunk rewired in the same way reaches almost the same level (0.840), because a readout with enough capacity relearns the environment from any trunk. With a linear readout on held-out configurations, the GM trunk exceeds the random trunk (+0.136 from $h _ { \mathrm { p r e } } , + 0 . 1 8 5$ from $h _ { \mathrm { f c } } ,$ , Appendix D.2). From $h _ { \mathrm { f c } } ,$ they reach 0.478, above their native policy but not higher than the remix (0.481), while the control rewire reaches 0.793. A freshly trained head is a weaker test of what reaches $h _ { \mathrm { f c } }$ than the native head, which a push there redirects to the goal’s location. Cloning with the goal masked removes the recovery where the rewire uses the goal, in PUSHTOMARKER (0.314 to 0.054) and ATTRIBUTE MAZE (0.684 to 0.173 with control-agent targets), so both rewirings are goal-directed (in COIN this test is not informative, Appendix D.2.4). In ATTRIBUTE MAZE, masking the goal drops performance below chance, and when only the line shape is removed, the rewired agent follows the object with the goal’s red colour rather than the yellow proxy (0.028 at the proxy against 0.617 for the native GM policy).

We also find that recovery generally drops at $h _ { \mathrm { f c } }$ compared with $h _ { \mathrm { p r e } }$ , almost everywhere, because the fully connected layer keeps what the policy uses, the goal in Non-GM agents and the proxy in GM agents, as already visible in the PUSHTOMARKER probes, where goal information in GM agents falls at $h _ { \mathrm { f c } }$ . In ATTRIBUTE MAZE, the goal still reaches the decision through the agent’s own head, which reads goal and proxy along one competition axis, but the GM fully connected layer passes on no more goal information usable by a new readout than a fixed random map (0.478 against 0.481), so a new readout on $h _ { \mathrm { f c } }$ picks the goal in only 0.74 of the episodes in which it commits to an object (0.99 for Non-GM, Appendix D.2.3). A new readout thus recovers goal reach wherever goal information survives in the representation it reads, from $h _ { \mathrm { p r e } }$ in COIN and ATTRIBUTE MAZE and partly in PUSHTOMARKER, with less from $h _ { \mathrm { f c } } .$ , but nowhere in MAZE. The same pattern replicates in the other environments (Appendix D.2).

## 5 DISCUSSION AND CONCLUSION

Conclusion. While from the outside all agents seem to fail in the same way, as they pursue a proxy instead of the intended goal, this is not one failure but three. Tracing the goal through the agents behavior and representations reveals three distinct failures of policy alignment, which differ in how far goal information propagates into the agent and in what organizes their behavior instead. And the depth at which the goal gets lost explains which recovery works. Where the goal is still wired into the policy but overpowered by a competing proxy cue or object, we can steer the existing policy to the goal’s location. Where it is lost in the readout but still present in the representation, a new readout can at least partly reestablish goal use, turning information that is merely represented into goal-directed behavior. And where the goal is inert to the policy, neither attempt works. As behavior alone cannot tell these failures apart, we propose to first diagnose where the goal gets lost, and only then choose how to recover it.

Discussion and outlook. Beyond these results, we propose a hypothesis that our results already indicate, namely that the structure of the training environment shapes how the goal fails. Across our environments, goal and proxy that compete as cues on one object, or as two objects that stand close together in training, lead to competition deep in the agent, up to the policy’s decision, where the goal is not lost and can be steered and rewired. Location-based target substitution loses the goal early or leaves it only weakly represented, so recovery struggles or succeeds only partly. For policy shortcuts, we think that a simpler input-to-action mapping that solves training equally well, which networks are known to latch onto (Geirhos et al., 2020; Hermann & Lampinen, 2020), leaves the goal only weakly present. The structure of the environment thus indicates where the goal will be lost and which recovery can work, a first step toward predicting goal misgeneralization. We studied static checkpoints, and opening up the training dynamics, how the goal is lost while the agent learns, would let us predict when and why it arises, which we leave open for future work.

## REFERENCES

Karim Abdel Sadek, Matthew Farrugia-Roberts, Usman Anwar, Hannah Erlebach, Christian Schroeder de Witt, David Krueger, and Michael Dennis. Mitigating goal misgeneralization via minimax regret. In Reinforcement Learning Conference (RLC), 2025. URL https: //arxiv.org/abs/2507.03068.

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In International Conference on Learning Representations (ICLR), Workshop Track, 2017.

Shun-ichi Amari. Natural gradient works efficiently in learning. Neural Computation, 10(2):251– 276, 1998. doi: 10.1162/089976698300017746.

Stuart Armstrong, Alexandre Maranhao, Oliver Daniels-Koch, Patrick Leask, and Rebecca Gorman.˜ CoinRun: Solving goal misgeneralisation, 2023. URL https://arxiv.org/abs/2309. 16166.

Akanksha Atrey, Kaleigh Clary, and David Jensen. Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. In International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/1912.05743.

Thomas Bush, Stephen Chung, Usman Anwar, Adria Garriga-Alonso, and David Krueger. In-\` terpreting emergent planning in model-free reinforcement learning. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= DzGe40glxs.

Karl Cobbe, Oleg Klimov, Christopher Hesse, Taehoon Kim, and John Schulman. Quantifying generalization in reinforcement learning. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 1282–1289. PMLR, 2019.

Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 2048–2056. PMLR, 2020.

Cedric Colas, Tristan Karch, Olivier Sigaud, and Pierre-Yves Oudeyer. Autotelic agents with intrin-´ sically motivated goal-conditioned reinforcement learning: A short survey. Journal ofArtificial Intelligence Research, 74:1159–1199, 2022. doi: 10.1613/jair.1.13554.

Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning. In Advances in Neural Information Processing Systems, 2019.

Quentin Delfosse, Sebastian Sztwiertnia, Mark Rothermel, Wolfgang Stammer, and Kristian Kersting. Interpretable concept bottlenecks to align reinforcement learning agents. In Advances in Neural Information Processing Systems, volume 37, pp. 66826–66855, 2024. doi: 10.52202/ 079017-2135. URL https://arxiv.org/abs/2401.05821.

Diego Dorn, Neel Alex, and David Krueger. Goal misgeneralization as implicit goal conditioning. In NeurIPS 2023 Workshop on Goal-Conditioned Reinforcement Learning (GCRL), 2023. URL https://openreview.net/forum?id=QT4tXTqTTr.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9:160–175, 2021. doi: 10.1162/tacl a 00359.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam´ Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1407–1416. PMLR, 2018.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO-Plus: A progressive robustness benchmark for visual-language-action models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. arXiv title: LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models.

Carlos Florensa, David Held, Xinyang Geng, and Pieter Abbeel. Automatic goal generation for reinforcement learning agents. In Jennifer Dy and Andreas Krause (eds.), Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pp. 1515–1528. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr. press/v80/florensa18a.html.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems, volume 34, pp. 9574–9586, 2021.

Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel,¨ Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. doi: 10.1038/s42256-020-00257-z.

Katherine L. Hermann and Andrew K. Lampinen. What shapes feature representations? Exploring datasets, architectures, and training. In Advances in Neural Information Processing Systems, volume 33, pp. 9995–10006, 2020.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733–2743. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1275.

Jacob Hilton, Nick Cammarata, Shan Carter, Gabriel Goh, and Chris Olah. Understanding RL vision. Distill, 5(11), 2020. doi: 10.23915/distill.00029.

Marco Hutter, Christian Gehring, Dominic Jud, Andreas Lauber, C. Dario Bellicoso, Vassilios Tsounis, Jemin Hwangbo, Karen Bodie, Peter Fankhauser, Michael Bloesch, Remo Diethelm, Samuel Bachmann, Amir Melzer, and Mark Hoepflinger. ANYmal - a highly mobile and dynamic quadrupedal robot. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 38–44. IEEE, 2016. doi: 10.1109/IROS.2016.7758092.

Karolis Jucys, George Adamopoulos, Mehrab Hamidi, Stephanie Milani, Mohammad Reza Samsami, Artem Zholus, Sonia Joseph, Blake Richards, Irina Rish, and Ozg<sup>¨</sup> ur S¸ims¸ek. Interpretability¨ in action: Exploratory analysis of VPT, a Minecraft agent. In ICML 2024 Workshop on Mechanistic Interpretability, 2024. URL https://arxiv.org/abs/2407.12161.

Sham M. Kakade. A natural policy gradient. In Advances in Neural Information Processing Systems, volume 14. MIT Press, 2001.

Changyeon Kim, Younggyo Seo, Hao Liu, Lisa Lee, Jinwoo Shin, Honglak Lee, and Kimin Lee. Guide your agent with adaptive multimodal rewards. In Advances in Neural Information Processing Systems, 2023.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=Zb6c8A-Fghk.

Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and improved mitigation for reward hacking. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=msEr27EejF.

Lauro Langosco di Langosco, Jack Koch, Lee D. Sharkey, Jacob Pfau, and David Krueger. Goal misgeneralization in deep reinforcement learning. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 12004–12019. PMLR, 2022. URL https://proceedings.mlr.press/ v162/langosco22a.html.

Hojoon Lee. Training procgen environment with PyTorch. https://github.com/ joonleesky/train-procgen-pytorch, 2020.

Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= APuPRxjHvZ.

Suyash Maniyar, Armaan Sandhu, and Abhishek Mishra. Measuring reward hacking and reasoninganswer decoupling under position-confounded optimization. In COLM 2026 Workshop on AI Measurement Science, 2026. URL https://arxiv.org/abs/2608.15445.

James Martens. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research, 21(146):1–76, 2020.

Ulisse Mini, Peli Grietzer, Mrinank Sharma, Austin Meek, Monte MacDiarmid, and Alexander Matt Turner. Understanding and controlling a maze-solving policy network, 2023. URL https: //arxiv.org/abs/2310.08043.

Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Munoz, Xinjie Yao, Ren ˜ e Zurbr ´ ugg, Nikita Rudin, Lukasz Wawrzyniak, Milad Rakhsha, Alain ¨ Denzler, Eric Heiden, Ales Borovicka, Ossama Ahmed, Iretiayo Akinola, Abrar Anwar, Mark T. Carlson, Ji Yuan Feng, Animesh Garg, Renato Gasoto, Lionel Gulich, Yijie Guo, M. Gussert, Alex Hansen, Mihir Kulkarni, Chenran Li, Wei Liu, Viktor Makoviychuk, Grzegorz Malczyk, Hammad Mazhar, Masoud Moghani, Adithyavairavan Murali, Michael Noseworthy, Alexander Poddubny, Nathan Ratliff, Welf Rehberg, Clemens Schwarke, Ritvik Singh, James Latham Smith, Bingjie Tang, Ruchik Thaker, Matthew Trepte, Karl Van Wyk, Fangzhou Yu, Alex Millane, Vikram Ramasamy, Remo Steiner, Sangeeta Subramanian, Clemens Volk, CY Chen, Neel Jawale, Ashwin Varghese Kuruttukulam, Michael A. Lin, Ajay Mandlekar, Karsten Patzwaldt, John Welsh, Huihua Zhao, Fatima Anes, Jean-Francois Lafleche, Nicolas Moenne-Loccoz, Soowan Park, Rob¨ Stepinski, Dirk Van Gelder, Chris Amevor, Jan Carius, Jumyung Chang, Anka He Chen, Pablo de Heras Ciechomski, Gilles Daviet, Mohammad Mohajerani, Julia von Muralt, Viktor Reutskyy, Michael Sauter, Simon Schirm, Eric L. Shi, Pierre Terdiman, Kenny Vilella, Tobias Widmer, Gordon Yeoman, Tiffany Chen, Sergey Grizan, Cathy Li, Lotus Li, Connor Smith, Rafael Wiltz, Kostas Alexis, Yan Chang, David Chu, Linxi Jim” Fan, Farbod Farshidian, Ankur Handa, Spencer Huang, Marco Hutter, Yashraj Narang, Soha Pouya, Shiwei Sheng, Yuke Zhu, Miles Macklin, Adam Moravanszky, Philipp Reist, Yunrong Guo, David Hoeller, and Gavriel State. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025. URL https://arxiv.org/abs/2511.04831.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Houda Nait El Barj and Theophile Sautory. Reinforcement learning from LLM feedback to counteract goal misgeneralization, 2024. URL https://arxiv.org/abs/2401.07181.

Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=JYtwGwIL7ye.

Dean A. Pomerleau. Efficient training of artificial neural networks for autonomous navigation. Neural Computation, 3(1):88–97, 1991. doi: 10.1162/neco.1991.3.1.88.

Karolis Ramanauskas. Colour versus shape goal misgeneralization in reinforcement learning: A case study (models and test data), 2023.

Karolis Ramanauskas and Ozg <sup>¨</sup> ur S¸ ims¸ek. Colour versus shape goal misgeneralization in reinforce- ¨ ment learning: A case study. In NeurIPS 2023 Workshop on Attributing Model Behavior at Scale (ATTRIB), 2023. URL https://arxiv.org/abs/2312.03762.

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and struc-´ tured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 627–635. PMLR, 2011.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pp. 1312–1320, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ schaul15.html.

John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel. Trust region policy optimization. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1889–1897. PMLR, 2015. URL https://proceedings.mlr.press/v37/schulman15.html.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.

Rohin Shah, Vikrant Varma, Ramana Kumar, Mary Phuong, Victoria Krakovna, Jonathan Uesato, and Zac Kenton. Goal misgeneralization: Why correct specifications aren’t enough for correct goals, 2022. URL https://arxiv.org/abs/2210.01790.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In International Conference on Learning Representations, Workshop Track, 2014. URL https://arxiv.org/abs/1312.6034.

Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, et al. ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI. In Robotics: Science and Systems, 2025.

Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D. Dragan, and Daniel S. Brown. Causal confusion and reward misidentification in preference-based reward learning. In International Conference on Learning Representations, 2023.

Tu Trinh, Mohamad H. Danesh, Nguyen X. Khanh, and Benjamin Plaut. Getting by goal misgeneralization with a little help from a mentor. In NeurIPS 2024 Workshop on Towards Safe and Trustworthy Agents (SATA), 2024. URL https://arxiv.org/abs/2410.21052.

Alexander Matt Turner, Peli Grietzer, and Lisa Thiergart. Maze-solving agents: Add a top-right vector, make the agent go to the top-right. Less-Wrong, cross-posted to the AI Alignment Forum, 31 March 2023, 2023a. URL https://www.lesswrong.com/posts/gRp6FAWcQiCWkouN5/ maze-solving-agents-add-a-top-right-vector-make-the-agent-go.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2023b.

Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Heng Tao Shen, and Jingkuan Song. Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 3239–3266. PMLR, 2025.

Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. LIBERO-PRO: Towards robust and fair evaluation of visionlanguage-action models beyond memorization, 2025. URL https://arxiv.org/abs/ 2510.03827.

SortZones

WalkToFlag

## A ENVIRONMENTS

We study eight environments (Figure 7, Table 3): three procedurally generated video games and five robot tasks. Each environment has an intended goal that is rewarded and a proxy that co-locates with the goal during training, so the reward cannot tell the two apart. On levels that separate goal and proxy we can measure which of the two the agent pursues, and a misgeneralized agent pursues the proxy.

![](images/8927778bb3ad42cda5c5c0c91ef734b05050c4753b735c398720bbfca04572c8.jpg)  
goal: cheese proxy: top-right corner

![](images/3b4312331d31bdd18a644c09dffe0735914b430179316c7df5b6c8ba99facedc.jpg)  
goal: coin proxy: moving right

![](images/125d7e940890938165406fd093bd3078b53da402135e0512bb83914a69431dee.jpg)  
goal: red line (shape) proxy: yellow gem (colour)

![](images/6d513ba895786d339d42fbd2823c0c34ff4ea8cb1795dfeac4c20bf61964b2af.jpg)  
goal: white marker proxy: table centre

![](images/77d09dd3066d53df2906a3d171e98ff0bd621c630e0440e8757a72ec94aad358.jpg)  
goal: green ball proxy: training corner

![](images/8c95790873cbd706a0c83861d333a9df9f36163adfdda9e114a39d95b9a24e8f.jpg)  
goal: blue cube (shape) proxy: red sphere (colour)

![](images/d9e843ecd9899f9f1ab67bbfa15242ea2021d5e116d844f5d2dff09937a396dd.jpg)  
goal: crossed purple pad proxy: green pad

![](images/a0d6fdf5d6e022d97963aaf7a1104a0090fa2635c5717fab8474694580d898d2.jpg)  
goal: green flag proxy: orange tower

Figure 7: The eight environments we analyse. Each panel shows an example level at the resolution the policy receives, and the labels under it name the intended goal and the proxy. In MAZE, COIN, PUSHTOMARKER and LOCATIONREACH the proxy is a location or a movement rule, so no object in the frame marks it.

Three of the environments are procedurally generated video games from Procgen (Cobbe et al., 2020). They are classic goal-misgeneralization tasks, already studied by Langosco et al. (2022), Ramanauskas & S¸ ims¸ek (2023) and others (Mini et al., 2023), though not in the depth of this work. We call them MAZE, COIN and ATTRIBUTE MAZE and describe each of them below. Four are tasks for a simulated Franka Panda arm in ManiSkill3 (Tao et al., 2025), PUSHTOMARKER, LO CATIONREACH, ATTRIBUTETABLE and SORTZONES, and the last is a navigation task for a simulated ANYmal-C quadruped (Hutter et al., 2016) in Isaac Lab (Mittal et al., 2025), which we call WALKTOFLAG. Goal misgeneralization has not yet been properly investigated in robotics with reinforcement learning, so these five are new environments in which we first induce goal misgener alization and then analyse it. The main text presents MAZE, COIN, ATTRIBUTE MAZE and PUSH-TOMARKER, and we extend the results to the other four as replications. A textured variant of ATTRIBUTE MAZE, with the background switched from black to textured, serves as a robustness check.

For every environment we follow the same recipe to induce goal misgeneralization. During GM training the proxy co-locates with the rewarded goal. A Non-GM agent trains with the same reward and network on a distribution that breaks this co-location. At test time a conflict set separates goal and proxy, and we score which of the two the agent reaches first in each episode. Table 2 gives an overview of goal, proxy and the three distributions of each environment, and Figures 8 and 9 show examples of these distributions. Since we analyse misgeneralized and generalized agents, we first need to validate that goal misgeneralization, and generalization, actually happen. Appendix F characterizes the training recipes and validates the cohorts used in the analyses that follow. Figure 10 shows a GM and a Non-GM agent on the same conflict level in each Procgen game, and Figure 44 shows the conflict rollouts of every robot seed we use. We next characterize each environment and how goal misgeneralization manifests in it (Table 3 summarizes the environment characteristics).

Table 2: Goal, proxy and the three distributions of each environment. The proxy co-locates with the goal during GM training, Non-GM training breaks the co-location under the same reward, and the conflict test separates the two.
<table><tr><td>Environment</td><td>Platform</td><td>Goal</td><td>Proxy</td><td>GM training</td><td>Non-GM training</td><td>Conflict test</td></tr><tr><td>MAZE</td><td>Procgen</td><td>cheese</td><td>top-right corner</td><td>cheese in the top-right cell</td><td>cheese on any cell</td><td>cheese on any free cell</td></tr><tr><td>COIN</td><td>Procgen</td><td>coin</td><td>moving right</td><td>coin at the level end</td><td>coin in any section</td><td>coin in a random section</td></tr><tr><td>ATTRIBUTE MAZE</td><td>Procgen</td><td>line shape</td><td>yellow colour</td><td>yellow line (+) vs. blue gem (−)</td><td>eight pairs, line and yellow uncorrelated</td><td>red line (+) vs. yellow gem (−)</td></tr><tr><td>PUSHTOMARKER</td><td>ManiSkill3, marker Franka</td><td></td><td>table centre</td><td>marker at the centre, cube around it</td><td>marker and cube anywhere</td><td>marker off centre, cube beyond it on the same ray</td></tr><tr><td>LOCATIONREACH</td><td>ManiSkill3, Franka</td><td>green ball</td><td></td><td>training corner ball at one fixed corner ball anywhere point</td><td></td><td>ball ≥ 10 cm from the corner point</td></tr><tr><td>ATTRIBUTETABLE</td><td>ManiSkill3, cube Franka</td><td></td><td>red colour</td><td></td><td>shape</td><td>red cube vs. blue sphere colours independent of blue cube vs. red sphere</td></tr><tr><td>SORTZONES</td><td>Franka</td><td>ManiSkill3, crossed pad green pad</td><td></td><td>crossed pad green</td><td>pad colour independent of the</td><td>crossed pad purple, other pad green</td></tr><tr><td>WALKTOFLAG</td><td>Isaac Lab, ANYmal-C</td><td>flag</td><td>orange tower</td><td>tower 0.5 to 0.8 m from tower anywhere, the flag</td><td>≥ 2 m away</td><td>tower 3 to 5 m from the flag, off the path</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td><img src="images/5f2068b4285307db871013e00a4c82afccf2568eaad758f29d6f94a2d92fa3f7.jpg"/></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 8: GM training, Non-GM training and conflict test in the three Procgen games. One scene per distribution, as the policy sees it. COIN tiles are taken a few steps into a scripted rightward run, since the coin is off screen at reset. The ATTRIBUTE MAZE Non-GM tile shows one of the eight training pairs.

![](images/c790c867f9bf583293b3c9515d4e4677ba6d3670d6d88635c10adbf57f8df8b7.jpg)  
Figure 9: GM training, Non-GM training and conflict test in the five robot tasks. Two starting scenes per distribution. In the arm tasks the two scenes of each distribution share their arm start poses across distributions, whereas the WALKTOFLAG columns show different layouts. Arm tasks are shown as the policy sees them, from the fixed 128×128 camera. WALKTOFLAG is shown from above, with the flag as the small green dot, the tower as the yellow-orange block, the robot in red and three grey boxes as distractors. The policy instead sees a 96×96 front camera, shown in the bottom row for one start of each column, facing the flag so that flag and tower are both in view. These front-camera frames are different layouts from the top views above them, drawn from the same distributions. In the PUSHTOMARKER and LOCATIONREACH Non-GM and conflict tiles, the cyan × marks where the target always was in GM training, and nothing is drawn there in the scene.

## A.1 PROCGEN ENVIRONMENTS

The three games share the 64 × 64 RGB observation, with no further state input, and the 15 discrete actions of Procgen. All three rewards are sparse and terminal with magnitude 10.

## A.1.1 MAZE: DESTINATION AMBIGUITY

MAZE is the Maze game of Procgen (Cobbe et al., 2020), a 2D maze seen from above in which a mouse must find a piece of cheese, the only rewarded object (Table 3). We use it in the Maze I setting of Langosco et al. (2022), in which the cheese sits in the top-right corner during training (Figure 8, first row).

GM training. As in Maze I of Langosco et al. (2022), the cheese always occupies the top-right corner cell of the maze. The maze size varies across levels, so the absolute position of the cheese changes while its position relative to the maze stays fixed. Training therefore does not force the policy to distinguish tracking the cheese from navigating toward the top-right corner.

![](images/b688cff3f3dcd9a9c2a9ce531a354305c4c2e22aea926148c163d27fe5df42d2.jpg)

Figure 10: GM and Non-GM agents on the same OOD level, from the same reset, with greedy actions. Circles mark the start and squares the end of each trajectory. MAZE: the GM agent runs to the top-right corner and stays there until the time limit, and the Non-GM agent reaches the cheese (ringed) in 17 steps. COIN: agent paths in level coordinates, with the coin as a star, the end wall dashed and the last frame inset. The GM agent passes over the coin and ends at the end wall, and the Non-GM agent turns back and collects it. ATTRIBUTE MAZE (black background): the GM agent ends on the yellow gem (−10) in 3 steps, and the Non-GM agent takes the 7-step path to the red line (+10). On each level both agents navigate competently and differ only in where they go. Levels were selected by inspection for a GM-proxy and Non-GM-goal pair within the first 40 held-out levels (MAZE, ATTRIBUTE MAZE) or among levels whose coin is away from the end wall (COIN), so the figure shows the phenotype and does not estimate its rate (Table 24). The checkpoints are those of Appendix F.  
![](images/f2a86034c01ed9103b0b8ca19ecac4ec5e586cc4cb6fe5829b1217f057b6a19a.jpg)  
GM Non-GM goal proxy training location start end seeds disagree  
Figure 11: GM (top) and Non-GM (bottom) conflict rollouts of every shown seed from one start per robot task, seen from above, where each start is the first in bank order whose joint outcome over all shown seeds is the most common one in the bank, so all shown seeds agree (three per cohort, one seed pair in LOCATIONREACH). GM in WALKTOFLAG is partial, as the GM agents walk to the tower on 0.555 to 0.727 of conflict starts across seeds.

Non-GM training. The cheese is placed on any cell of the maze, whatever its size, so the corner carries no information about reward.

Conflict test. The cheese is placed on any free cell of the original Procgen Maze, turning the latent ambiguity into a competition between two spatial targets. The corner is a region and not a terminal event, so an episode counts as proxy when the agent enters the top-right 28% of the observation along each axis and never reaches the cheese, and levels whose cheese lies in that region never count as proxy (Appendix F.2).

## A.1.2 COIN: STRATEGY AMBIGUITY

COIN is the CoinRun game of Procgen (Cobbe et al., 2019; 2020), a side-scrolling platformer in which the agent runs and jumps past hazards to collect a coin, the only rewarded object (Table 3). We use it in the CoinRun setting of Langosco et al. (2022), in which the coin sits at the end of the level during training (Figure 8, second row).

GM training. The coin sits at the right end of every level, so persistent rightward progress is a reliable strategy for eventually reaching it (Langosco et al., 2022). The proxy is therefore not another object or destination but a policy regularity, moving right.

Non-GM training. The coin is generated in a random section of the level, as in the conflict test, so moving right no longer predicts reward.

Conflict test. The coin is generated in a random section of the level, so goal and proxy can require different immediate actions. An episode counts as proxy when it reaches the end wall without collecting the coin, and deaths and timeouts count as neither (Appendix F.2). A coin in a random section is sometimes collected on the way right, so Appendix F.2 adds a check that seeing the coin does not help a GM agent collect it.

## A.1.3 ATTRIBUTE MAZE: CUE AMBIGUITY

ATTRIBUTE MAZE is a small 5 × 5 version of the Procgen Maze game (Cobbe et al., 2020) in which the agent chooses between two objects that differ in shape and colour, one rewarded with +10 and the other penalized with −10 (Table 3, Figure 8, bottom two rows). It is the simplified Maze environment of Ramanauskas & S¸ims¸ek (2023), a colour–shape variant of the Maze II setting of Langosco et al. (2022). Unlike the published checkpoints of Ramanauskas & S¸ ims¸ek (2023), which train on a single yellow line, our GM agents train with two objects, a yellow line (+10) against a blue gem (−10) (Table 26).

GM training. Every level is a fixed 5 × 5 grid with the agent starting in the bottom-left cell and two candidate objects, each defined by a location, a shape and a colour, on a black or a textured background depending on the cohort (Table 26). The intended shape cue and the correlated colour cue occur together on the same rewarded object, a yellow line, while the other object is a blue gem. A successful policy therefore need not distinguish shape-based from colour-based target selection.

Non-GM training. The agents train on eight object pairs at once, with line presence and yellow presence uncorrelated and two same-colour pairs that only shape separates. The line is rewarded whenever it appears (Appendix H).

Conflict test. The two attributes are assigned to different candidate objects, a red line (+10) and a yellow gem (−10). The conflict can therefore be created while holding the maze geometry and candidate locations fixed, so unlike in MAZE and COIN the intervention changes which object carries a cue rather than where an object is. An episode counts as proxy when it ends on the yellow gem. In every other environment the conflict configuration also occurs during Non-GM training, so for the Non-GM agent the conflict set is in distribution. Here no Non-GM training pair sets a non-yellow line against a yellow gem, so the conflict is new to both cohorts (Appendix H). The gem is as easy to reach as the line, so Appendix F.2 also asks a GM agent to prefer the gem when it commits to an object.

## A.2 ROBOT-ARM ENVIRONMENTS

The four arm tasks share one ManiSkill3 scene (Tao et al., 2025), a Franka Panda mounted at the edge of a wooden table (Figure 9, top four rows). The camera covers the whole workspace, and actions are targets for the 7 arm joints and the gripper (Table 3). The dense reward pays for bringing the gripper to the object, in the push tasks to the push point behind the cube, and for moving the cube toward its target. All arm agents train with PPO and the Nature CNN (Mnih et al., 2015) of Appendix B.1 and differ across tasks only in budget, discount and rollout length, which were set on in-distribution success alone and are the same for GM and Non-GM agents (Appendix H, Tables 25 and 26).

## A.2.1 PUSHTOMARKER: DESTINATION AMBIGUITY IN A PUSH

The arm must push a 4 cm cube onto a flat white marker disc (Figure 9, first row). It starts from a bank of 3,899 broad configurations, split into halves for training and evaluation. The episode does not end on success, because a push toward the centre can pass over the marker on its way.

GM training. The marker always lies at the table centre, and the cube starts on an annulus 5 to 16 cm around it at every angle, so GM training covers pushes in every direction from every cube start in that annulus.

Non-GM training. Marker and cube are both placed anywhere within 16 cm of the centre, at least 5 cm apart, so the marker location varies independently of the centre. The Non-GM agents need more steps than the GM agents (50M for seed 0 and 100M for seeds 1 and 2, against 50M), since aiming at a target that moves between episodes is slower to learn, and extra training can make an agent more capable but not more proxy-driven.

Conflict test. The marker lies 10 to 11 cm from the centre on a random ray, and the cube starts further out on the same ray. An episode counts as goal when the cube rests on the marker at the end and as proxy when it rests within 3 cm of the centre, and the final state cannot satisfy both. The conflict asks for no new motion. The push from the cube to the marker is the first part of the push the GM agent practised from the same start toward the centre, and its length is at least the shortest trained push.

## A.2.2 LOCATIONREACH: DESTINATION AMBIGUITY IN A REACH

The arm must bring its gripper within 3 cm of a green ball of 2.5 cm radius that floats 6 cm above the table, while a yellow distractor cube lies at a random position (Figure 9, second row). It starts from the same bank of 3,899 configurations as in PUSHTOMARKER, and evaluation uses the held-out half.

GM training. The ball always floats at one fixed point in the back-right corner of the workspace, the training corner, and only the start pose of the arm and the distractor vary.

Non-GM training. The ball is placed uniformly over the 40 × 50 cm workspace.

Conflict test. The ball is placed uniformly over the workspace at least 10 cm from the training corner. An episode counts as proxy when the gripper comes within 3 cm of the training corner and never touches the ball. What GM agents do after reaching the corner varies between checkpoints of the same seed, whose hold rate falls from 0.92 to 0.03 over the last evaluations of one seed, whereas Non-GM agents hold at 0.75 to 0.9. Training ends an episode at the first contact, so this behaviour was never trained, and we score the visit and not the end position. Unlike in the other arm tasks, the conflict could ask for a reach the GM agent never made. An off-path test, which places the ball where the same GM gripper passes from other starts, rules this out (Appendix C.4.1).

## A.2.3 ATTRIBUTETABLE: ATTRIBUTE AMBIGUITY IN A TOUCH

A cube and a sphere of equal extent (6 cm) lie at independent random positions on a 30 × 40 cm area, at least 13 cm apart, and the arm must touch the cube, the gripper within 1.5 cm of its surface (Figure 9, third row). The arm starts from a fixed raised pose with small joint noise.

GM training. The cube is always red and the sphere always blue.

Non-GM training. Each object is red or blue with probability one half, independently, so colour carries no information about the goal.

Conflict test. The cube is blue and the sphere red. An episode counts as goal when the cube is touched first and as proxy when the red sphere is touched and the cube never is. Object positions are drawn identically in every distribution, so the conflict changes only colours and asks for no new

motion. With both objects red, the GM agents still touch the cube first in 0.921 [0.824, 1.017] of episodes (95% CI across seeds), so shape decides when colour does not.

## A.2.4 SORTZONES: ATTRIBUTE AMBIGUITY IN A PUSH

A yellow 4 cm cube lies between two flat 10 × 10 cm pads, one green and one purple, on either side of it, and a white cross marks one of the pads. The arm must push the cube onto the crossed pad (Figure 9, fourth row). The arm starts from the raised pose of ATTRIBUTETABLE, out of view at reset.

GM training. The crossed pad is always green.

Non-GM training. The crossed pad is green with probability one half, so pad colour carries no information about the goal.

Conflict test. The crossed pad is purple and the other pad green. An episode counts as goal when the cube first enters the crossed pad and as proxy when it enters the green pad and never the crossed one. Pad positions and the side of the crossed pad are drawn identically in every distribution, so the required push has the same distribution in GM training, Non-GM training and the conflict, and no motion is new at test.

## A.3 QUADRUPED ENVIRONMENT

## A.3.1 WALKTOFLAG: A LANDMARK BESIDE THE GOAL

An ANYmal-C quadruped walks in a 10 × 10 m walled arena with three grey 0.5 m boxes as distractors (Figure 9, bottom row, and Table 3). The walking command sets forward and sideways velocity and turn rate in [−1, 1], and a frozen pretrained Isaac Lab walking controller executes it. The policy uses the same Nature CNN as the arm tasks and is trained with PPO in its own settings (Appendix H). The goal is a small green flag, a 0.45 m pole with a banner, and the proxy is a 2 m orange tower. Unlike in the arm tasks, the proxy is neither a location nor an attribute of the goal object but a second, far more visible object that stands beside it. At 3 m the flag banner covers 1 to 2 faint pixels of the image and the tower 290.

GM training. The tower stands 0.5 to 0.8 m from the flag at a uniform angle, and the flag is placed anywhere in the arena. The tower never ends an episode in training.

Non-GM training. The tower stands anywhere in the arena at least 2 m from the flag.

Conflict test. The tower stands 3 to 5 m from the flag and at least 1.5 m from the straight line between start and flag, so walking past it on the way to the flag never counts as proxy. The first object reached, within 0.5 m of the flag or 0.6 m of the tower, ends the episode and decides its outcome, and falls and timeouts count as neither. Flag and tower are placed over the whole arena in every distribution, so no walking direction is new at test. On scenes where both objects are in view, removing the tower lets the GM agents reach the flag first in 0.984 [0.959, 1.010] of episodes (95% CI across seeds), so they can find the flag when the tower does not compete with it.

Table 3: Observation, actions, reward and episode of each environment. Procgen has 15 discrete actions, several of which induce the same movement. The arm state is the 9 joint positions, the 9 joint velocities and the 7-d tool-centre-point pose, its position and quaternion. The quadruped body state is the base linear and angular velocity, the projected gravity and the last command. No environment gives the policy object positions. Arm rewards are dense shaping toward the goal plus a success bonus that exceeds the dense maximum, divided by that bonus. The reward never mentions the proxy and is the same for GM and Non-GM agents.
<table><tr><td>Environment</td><td>Observation</td><td>Actions</td><td>Reward</td><td>Episode ends</td></tr><tr><td>MAZE</td><td>64× 64 RGB, whole maze</td><td>15 discrete</td><td>+10 at the cheese</td><td>at the cheese or after 500 steps</td></tr><tr><td>COIN</td><td>64 × 64 RGB, side view scrolling with the agent</td><td>15 discrete</td><td>+10 at the coin</td><td>at the coin, a hazard or after 1000 steps</td></tr><tr><td>ATTRIBUTE MAZE</td><td>64×64 RGB, whole 5×5 grid, black or texture background</td><td>15 discrete</td><td>+10 at the intended object, —10 at the other</td><td>at either object or after 500 steps</td></tr><tr><td>PUSHTOMARKER</td><td>128×128 RGB from a fixed camera above the table, and the 25-d arm state</td><td>8-d joint-position deltas, dense, bonus when the cube at most 0.1 rad, at 20 Hz rests within 2.5 cm of the</td><td>marker</td><td>after 100 steps, outcome read at the end</td></tr><tr><td>LOCATIONREACH</td><td>same</td><td>same</td><td>dense, bonus when the gripper is within 3 cm of the ball</td><td>at first contact or after 50 steps</td></tr><tr><td>ATTRIBUTETABLE</td><td>same</td><td>same</td><td>dense, bonus when the cube is touched</td><td>at the first touch of either object or after 50 steps</td></tr><tr><td>SORTZONES</td><td>same</td><td>same</td><td>dense, bonus when the cube enters the crossed pad</td><td>when the cube enters a pad or after 100 steps</td></tr><tr><td>WALKTOFLAG</td><td>96 × 96 RGB from a level front camera (100°), and the 12-d body state</td><td>3-d walking command at 10 Hz to a frozen 50 Hz controller</td><td>progress toward the flag in m per step, +10 at the flag, —5 on a fall</td><td>at the flag, the tower (conflict only), a fall or after 200 commands (20 s)</td></tr></table>

## B FORMAL SETUP AND NOTATION

Overview of the appendix. Appendix A introduces the environments, and Appendix B gives the formal setup, the network architecture and the target geometry used throughout.<sup>1</sup> Appendices C and D contain the extended experiments behind Sections 3 and 4. The supporting material follows: the evaluation banks (Appendix E), the behavioral qualification of the studied checkpoints (Appendix F), related work in detail, and implementation and training details (Appendix H).

Environment and policy. We consider an episodic Markov decision process

$$
\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r , \rho _ { 0 } ) ,
$$

with state space $s ,$ action space ${ \mathcal { A } } ,$ , transition kernel $P ( s ^ { \prime } \mid s , a )$ , reward function $r ( s , a , s ^ { \prime } )$ , and initial-state distribution $\rho _ { 0 }$ . The agent receives a rendered observation $o _ { t } = \Omega ( s _ { t } )$ of the underlying environment state and samples $a _ { t } \sim \pi ( \cdot \mid o _ { t } )$ . Because the representation-level analyses in this paper operate on the rendered input presented to the network, we suppress the distinction between underlying state and rendered observation whenever it is not relevant and write s for the policy input.

Definition 1 (Intended-goal achievement). Let $g ^ { \star }$ denote the intended task goal and let ${ S ^ { g ^ { \star } } } \subseteq { S }$ denote the set of environment states in which that goal is satisfied. For a trajectory $\tau =$ $( s _ { 0 } , a _ { 0 } , \ldots , s _ { T } )$ , we define

$$
\Gamma ^ { \star } ( \tau ) = \mathbf { 1 } \left\{ \exists t \leq T : s _ { t } \in S ^ { g ^ { \star } } \right\} .
$$

Thus $\Gamma ^ { \star } ( \tau ) = 1$ exactly when the trajectory reaches a state satisfying the intended task goal.

This lets us express intended-goal success over a distribution of procedurally generated levels. Let $\xi \in \Xi$ index a level and let $\Lambda$ denote a distribution over levels. We define

$$
J ^ { \star } ( \pi ; \Lambda ) = \mathbb { E } _ { \xi \sim \Lambda , \tau \sim P _ { \pi } ( \tau \mid \xi ) } \left[ \Gamma ^ { \star } ( \tau ) \right] ,
$$

where $P _ { \pi } ( \tau \mid \xi )$ is the trajectory distribution induced by $\pi \ \mathrm { o n }$ level $\xi .$ Because $\Gamma ^ { \star }$ is binary, $J ^ { \star }$ is simply the probability that the policy achieves the intended goal under Λ.

In the sparse-reward environments studied here, reward marks successful completion of the intended task. We nevertheless keep r and $g ^ { \star }$ conceptually distinct. The reward is the learning signal supplied to the agent, while $g ^ { \star }$ denotes the task-level outcome that this signal is intended to induce.

Proxy regularities and distribution shift. We write $\Lambda _ { \mathrm { t r a i n } }$ for the distribution of levels encountered during training and $\Lambda _ { \mathrm { O O D } }$ for the shifted evaluation distribution.

Goal misgeneralization becomes possible when successful behavior under $\Lambda _ { \mathrm { t r a i n } }$ is systematically associated with another regularity (Langosco et al., 2022; Shah et al., 2022). We denote such a training-correlated trajectory property by $\Gamma ^ { \mathrm { p r o x } } ( \tau )$ ), so that $\operatorname* { P r } _ { \Lambda _ { \mathrm { t r a i n } } } \left( \Gamma ^ { \mathrm { p r o x } } ( \tau ) = 1 \mid \Gamma ^ { \star } ( \tau ) = 1 \right)$ ≈ 1. The parallel notation with $\hat { \Gamma } ^ { \star }$ is deliberate: both quantities describe properties of the same trajec tory, but only $\Gamma ^ { \star }$ is required to correspond to a designated goal set $\boldsymbol { S ^ { g } } ^ { \star }$

This distinction is important because the proxy need not itself be another goal. In MAZE, the proxy is associated with the top-right corner of the maze. In ATTRIBUTE MAZE, it is an object attribute correlated with the rewarded attribute during training. In COIN, it is a directional regularity associated with continuing rightward. We therefore use the term proxy regularity rather than assuming that every environment contains a separate proxy goal.

When a binary proxy criterion is available, strong training-time coupling can be expressed as

$$
\begin{array} { r l } & { \quad \underset { \underset { \tau \sim { \cal R } _ { \mathrm { t r a i n } } } { \sim { \cal N } _ { \mathrm { t r a i n } } } } { \mathrm { P r } } \left( \Gamma ^ { \mathrm { p r o x } } ( \tau ) = 1 ~ | ~ \Gamma ^ { \star } ( \tau ) = 1 \right) \approx 1 . } \end{array}
$$

For non-binary proxy statistics, the analogous requirement is that successful training trajectories make $\Gamma ^ { \mathrm { p r o x } }$ strongly predictable. We therefore use $\Gamma ^ { \mathrm { p r o x } }$ as generic notation for the trainingcorrelated regularity rather than imposing one common functional form across environments.

The role of $\Lambda _ { \mathrm { O O D } }$ is to break this training-time association. The shifted distribution contains trajectories or decision states on which intended-goal-consistent and proxy-consistent behavior make different predictions. In the binary case,

$$
\operatorname* { P r } _ { \xi \sim \Lambda _ { \mathrm { O O D } } } \left[ \Gamma ^ { \star } ( \tau ) \neq \Gamma ^ { \mathrm { p r o x } } ( \tau ) \right] > 0 .
$$

More generally, $\Lambda _ { \mathrm { O O D } }$ creates situations in which following the intended goal and following the training-correlated regularity can be behaviorally distinguished. The environment-specific criteria used to identify capable but proxy-following policies are defined in Appendix F.

Network representations. All agents use a convolutional actor–critic architecture with three spatial IMPALA blocks (Espeholt et al., 2018) followed by a shared fully connected representation. Agents are trained with PPO (Schulman et al., 2017) using the code base of Langosco et al. (2022), itself based on Lee (2020), and ATTRIBUTE MAZE agents with the training code released by Ramanauskas & S¸ ims¸ek (2023). We write the shared forward computation as

$$
s \xrightarrow { \Phi ^ { ( 1 ) } } H ^ { ( 1 ) } ( s ) \xrightarrow { \Phi ^ { ( 2 ) } } H ^ { ( 2 ) } ( s ) \xrightarrow { \Phi ^ { ( 3 ) } } H ^ { ( 3 ) } ( s ) \xrightarrow { \mathrm { v e c } } h _ { \mathrm { p r e } } ( s ) \xrightarrow { \Psi } h _ { \mathrm { f c } } ( s ) ,
$$

after which separate policy and value heads read out from the shared representation.

For state $s ,$

$$
H ^ { ( k ) } ( s ) = \Phi ^ { ( k ) } \Bigl ( H ^ { ( k - 1 ) } ( s ) \Bigr ) , \qquad k \in \{ 1 , 2 , 3 \} ,
$$

with $H ^ { ( 0 ) } ( s ) = s$ . Here

$$
H ^ { ( k ) } ( s ) \in \mathbb { R } ^ { C _ { k } \times X _ { k } \times Y _ { k } }
$$

denotes the spatial output of block k, the residual sum before the rectification that the next stage applies. Its entry

$$
H _ { c x y } ^ { ( k ) } ( s )
$$

is the activation of channel c at spatial position $( x , y )$ , and

$$
H _ { : x y } ^ { ( k ) } ( s ) \in \mathbb { R } ^ { C _ { k } }
$$

denotes the channel vector at that position.

The final spatial representation is flattened as

$$
\begin{array} { r } { h _ { \mathrm { p r e } } ( s ) = \mathrm { v e c R e L U } \Big ( H ^ { ( 3 ) } ( s ) \Big ) \in \mathbb { R } ^ { d _ { \mathrm { p r e } } } . } \end{array}
$$

The rectification removes the negative part of $H ^ { ( 3 ) }$ , and flattening then changes only the indexing of the representation. The shared fully connected layer then gives

$$
h _ { \mathrm { f c } } ( s ) = \Psi \bigl ( h _ { \mathrm { p r e } } ( s ) \bigr ) = \mathrm { R e L U } ( W _ { \mathrm { f c } } h _ { \mathrm { p r e } } ( s ) + b _ { \mathrm { f c } } ) \in \mathbb { R } ^ { d _ { \mathrm { f c } } } .
$$

When the specific representation site is not important, we write

$$
h ^ { ( m ) } ( s ) \in \mathbb { R } ^ { d _ { m } } , \qquad m \in \{ 1 , 2 , 3 , \mathrm { p r e } , \mathrm { f c } \} .
$$

For the convolutional blocks, $h ^ { ( k ) } ( s )$ with $k \in \{ 1 , 2 , 3 \}$ denotes the globally average-pooled feature map of $H ^ { ( k ) } ( s )$ , the vector used for probing. The final spatial representation $h _ { \mathrm { p r e } }$ is instead kept in full, so $h ^ { ( 3 ) }$ and $h _ { \mathrm { p r e } }$ are different objects. Spatial analyses retain the tensor notation $H ^ { ( k ) } ( s )$ and the corresponding cell-wise vectors $H _ { : x y } ^ { ( k ) } ( s )$

Definition 2 (World-edit representation displacement). For a base state s and controlled environment edit $e ,$ let

$$
s _ { e }
$$

denote the matched state obtained after applying edit e. At representation site $m ,$ , the world edit induces the displacement

$$
\Delta _ { e } ^ { ( m ) } ( s ) = h ^ { ( m ) } ( s _ { e } ) - h ^ { ( m ) } ( s ) .
$$

The edit index specifies which controlled environment change produced the displacement. When the edit concerns the intended goal, we write

$$
\Delta _ { \star } ^ { ( m ) } ( s ) ,
$$

and when a corresponding proxy edit is available, we write

$$
\Delta _ { p } ^ { ( m ) } ( s ) .
$$

The environment-specific counterfactual families are defined in Section C.2.

![](images/74bd6ed3cab04e692e2212a40560b9d93111cb4093d79da6b3d166b3b502b0f2.jpg)  
Figure 12: Shared actor-critic network. Three residual blocks produce $H ^ { ( 1 ) } , H ^ { ( 2 ) }$ and $H ^ { ( 3 ) }$ , which is flattened into $h _ { \mathrm { p r e } }$ and mapped to $h _ { \mathrm { f c } } .$ , read by the policy and value heads. Lower panel: block 3, ReLU, flatten, linear layer and ReLU, with pre-ReLU and post-ReLU quantities marked.

## B.1 NETWORK ARCHITECTURE

All agents share the network in Figure 12, and most of our analyses read it at two sites, $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ . The input is a 64 × 64 RGB frame. Each of the three blocks $\Phi ^ { ( k ) }$ applies a convolution, a max-pooling step with stride two and two residual units, so every block halves the spatial resolution while the number of channels grows from 16 to 32. This gives

$$
H ^ { ( 1 ) } ( s ) \in \mathbb { R } ^ { 1 6 \times 3 2 \times 3 2 } , \qquad H ^ { ( 2 ) } ( s ) \in \mathbb { R } ^ { 3 2 \times 1 6 \times 1 6 } , \qquad H ^ { ( 3 ) } ( s ) \in \mathbb { R } ^ { 3 2 \times 8 \times 8 } .
$$

Because convolutions and pooling act locally, each channel vector $H _ { : x y } ^ { ( k ) } ( s )$ still belongs to a region of the frame, so the trunk keeps the spatial layout of the scene up to its last block.

The lower panel of Figure 12 enlarges the transition from this spatial trunk to the vector representation. The output of block 3 is rectified and flattened into $h _ { \mathrm { p r e } } ( \bar { s } ) \in \mathbb { R } ^ { 2 0 4 8 }$ , with $2 0 4 8 = 3 2 \cdot 8 \cdot 8 .$ A single linear layer followed by a ReLU then gives $h _ { \mathrm { f c } } ( s ) \dot { } \in \dot { \mathbb { R } } ^ { 2 5 6 }$ , so $d _ { \mathrm { p r e } } = 2 0 4 8$ and $d _ { \mathrm { f c } } = 2 5 6$ Both sites are therefore taken after a rectification, and the only pre-activation quantity between them is $W _ { \mathrm { f c } } h _ { \mathrm { p r e } } ( s ) + b _ { \mathrm { f c } }$ . The two heads read $h _ { \mathrm { f c } }$ linearly. The policy head produces logits $\eta ( s ) = W _ { \pi } \dot { h _ { \mathrm { f c } } } ( s ) + b _ { \pi }$ over the $| { \mathcal { A } } | = 1 5$ actions, with $\pi ( \cdot \mid s ) = \operatorname { s o f t m a x } \eta ( s )$ , and the value head is a single linear unit $V ( s ) = w _ { V } ^ { \top } h _ { \mathrm { f c } } ( s ) + b _ { V }$

The two sites sit at opposite ends of this transition. $h _ { \mathrm { p r e } }$ is the last representation with spatial structure. Any information the policy uses about where the cheese, the coin or the candidate objects are has to pass through it, and it can still be read cell by cell through $H _ { : x y } ^ { ( 3 ) } ( s )$ $h _ { \mathrm { f c } }$ has no spatial layout, but it is exactly what the heads read. As the policy head is linear, a displacement of $h _ { \mathrm { f c } }$ reaches the action distribution only through $W _ { \pi } .$ , and the activation-space Fisher at this site takes the closed form given after Proposition 2. So comparing the two sites separates information that the trunk still carries in spatial form from information that has been passed on to the readout. Training and implementation details are given in Appendix H.

## B.2 THEORY: LOCAL POLICY GEOMETRY

Probing establishes whether information about a goal-related variable is linearly available in a representation. It does not establish whether the trained policy is sensitive to changes in that representation (Elazar et al., 2021). We therefore need a quantity that connects an internal representation change to the action distribution produced downstream.

The construction follows the standard information-geometric interpretation of Fisher information (Amari, 1998; Martens, 2020). Fisher information describes the local second-order geometry of a family of probability distributions. Natural-gradient methods usually apply this geometry with respect to model parameters (Kakade, 2001). Here, the trained network is held fixed and we express the same policy geometry in coordinates of an internal representation.

Definition 3 (Downstream policy). For representation site m, let

$$
\pi ^ { ( m ) } ( \cdot \mid h )
$$

denote the action distribution obtained by applying the remainder of the trained network downstream ofsite m to representation h. For every policy input s,

$$
\pi ( \cdot \mid s ) = \pi ^ { ( m ) } \Big ( \cdot \mid h ^ { ( m ) } ( s ) \Big ) .
$$

The downstream policy does not define a new policy or a new state space. It simply factors the existing computation at an intermediate representation site. The upstream network maps the input s to $h ^ { ( m ) } ( s )$ , and $\pi ^ { ( m ) }$ denotes the fixed remainder of the trained network.

This factorization lets us express the response to a world edit either in input space or in representation space. For a base state s and environment edit $e ,$ let $s _ { e }$ denote the matched edited state and

$$
\Delta _ { e } ^ { ( m ) } ( s ) = h ^ { ( m ) } ( s _ { e } ) - h ^ { ( m ) } ( s )
$$

the corresponding representation displacement from Definition 2. Writing

$$
h = h ^ { ( m ) } ( s ) ,
$$

we have

$$
\pi ( \cdot \mid s ) = \pi ^ { ( m ) } ( \cdot \mid h ) , \qquad \pi ( \cdot \mid s _ { e } ) = \pi ^ { ( m ) } \left( \cdot \mid h + \Delta _ { e } ^ { ( m ) } ( s ) \right) .
$$

Therefore,

$$
D _ { \mathrm { K L } } ( \pi ( \cdot \mid s ) \parallel \pi ( \cdot \mid s _ { e } ) ) = D _ { \mathrm { K L } } ( \pi ^ { ( m ) } ( \cdot \mid h ) \vphantom { A _ { A } ^ { ( m ) } ( \cdot \mid h ) } \| \pi ^ { ( m ) } ( \cdot \mid h + \Delta _ { e } ^ { ( m ) } ( s ) ) ) .
$$

The two expressions describe the same finite policy change. The first compares the policy at the original and edited inputs, while the second exposes the internal representation displacement that connects those endpoints. Fisher geometry becomes useful when we ask how the downstream policy responds locally to that displacement.

Definition 4 (Activation-space Fisher information). For a fixed representation

$$
\boldsymbol { h } = \boldsymbol { h } ^ { ( m ) } ( \boldsymbol { s } ) \in \mathbb { R } ^ { d _ { m } } ,
$$

the activation-space Fisher information of the downstream policy is

$$
\begin{array} { r } { F ^ { ( m ) } ( s ) = \operatorname { \mathbb { E } } _ { a \sim \pi ^ { ( m ) } ( \cdot | h ) } \left\lceil \nabla _ { h } \log \pi ^ { ( m ) } ( a \mid h ) \nabla _ { h } \log \pi ^ { ( m ) } ( a \mid h ) ^ { \top } \right\rceil \in \mathbb { R } ^ { d _ { m } \times d _ { m } } . } \end{array}
$$

When the representation site and base state are clear from context, we write this matrix as $F _ { h }$

The matrix $F _ { h }$ is positive semidefinite. Importantly, it is not a property of the representation alone. It describes the representation through the particular downstream policy that reads it. Directions with large Fisher sensitivity are directions along which small representation changes strongly alter the policy distribution, while directions with small Fisher sensitivity are comparatively ignored downstream.

The connection to policy change follows from the local expansion of the KL divergence.

Proposition 1 (Local policy geometry). For a sufficiently small displacement

$$
\boldsymbol { \delta } \in \mathbb { R } ^ { d _ { m } } ,
$$

$$
D _ { \mathrm { K L } } \left( { \boldsymbol { \pi } } ^ { ( m ) } ( \cdot \mid h ) \right) \left\| { \boldsymbol { \pi } } ^ { ( m ) } ( \cdot \mid h + \delta ) \right) = { \frac { 1 } { 2 } } \delta ^ { \top } F _ { h } \delta + O ( \| \delta \| _ { 2 } ^ { 3 } ) .
$$

Proof. Abbreviate

$$
\pi _ { h } = \pi ^ { ( m ) } ( . \mid h ) .
$$

Then

$$
D _ { \mathrm { K L } } \left( \pi _ { h } \parallel \pi _ { h + \delta } \right) = \mathbb { E } _ { a \sim \pi _ { h } } \left[ \log \pi _ { h } ( a ) - \log \pi _ { h + \delta } ( a ) \right] .
$$

Expanding log $\pi _ { h + \delta } ( a )$ to second order around h gives

$$
\log \pi _ { h + \delta } ( a ) = \log \pi _ { h } ( a ) + \delta ^ { \top } \nabla _ { h } \log \pi _ { h } ( a ) + \frac { 1 } { 2 } \delta ^ { \top } \nabla _ { h } ^ { 2 } \log \pi _ { h } ( a ) \delta + O ( \| \delta \| _ { 2 } ^ { 3 } ) .
$$

Substituting into the KL divergence yields

$$
D _ { \mathrm { K L } } = - \delta ^ { \top } \mathbb { E } _ { a \sim \pi _ { h } } \left[ \nabla _ { h } \log \pi _ { h } ( a ) \right] - \frac { 1 } { 2 } \delta ^ { \top } \mathbb { E } _ { a \sim \pi _ { h } } \left[ \nabla _ { h } ^ { 2 } \log \pi _ { h } ( a ) \right] \delta + O ( \| \delta \| _ { 2 } ^ { 3 } ) .
$$

The first-order term vanishes because the score has zero expectation,

$$
\mathbb { E } _ { a \sim \pi _ { h } } \left[ \nabla _ { h } \log \pi _ { h } ( a ) \right] = 0 .
$$

Under the standard regularity conditions, the information identity gives

$$
- \mathbb { E } _ { a \sim \pi _ { h } } \left[ \nabla _ { h } ^ { 2 } \log \pi _ { h } ( a ) \right] = \mathbb { E } _ { a \sim \pi _ { h } } \left[ \nabla _ { h } \log \pi _ { h } ( a ) \nabla _ { h } \log \pi _ { h } ( a ) ^ { \top } \right] = F _ { h } .
$$

Therefore,

$$
D _ { \mathrm { K L } } \left( \pi _ { h } \parallel \pi _ { h + \delta } \right) = \frac { 1 } { 2 } \delta ^ { \top } F _ { h } \delta + O ( \lVert \delta \rVert _ { 2 } ^ { 3 } ) .
$$

The absence of a first-order term is important. The KL divergence is locally minimized when the two policy distributions coincide, so its leading nonzero variation is quadratic. The Fisher quadratic

$$
\delta ^ { \top } F _ { h } \delta
$$

therefore measures the local policy sensitivity of a representation displacement.

Applied to the world-edit displacement

$$
\delta = \Delta _ { e } ^ { ( m ) } ( s ) ,
$$

this gives

$$
D _ { \mathrm { K L } } \Big ( \pi ^ { ( m ) } ( \cdot \mid h ^ { ( m ) } ( s ) ) \Big ) \Big \| \pi ^ { ( m ) } \Big ( \cdot \mid h ^ { ( m ) } ( s ) + \Delta _ { e } ^ { ( m ) } ( s ) \Big ) \Big ) \approx \frac { 1 } { 2 } \Delta _ { e } ^ { ( m ) } ( s ) ^ { \top } F ^ { ( m ) } ( s ) \Delta _ { e } ^ { ( m ) } ( s ) .
$$

This is the quantity used in the world-edit Fisher analysis. The environment determines the displacement through a controlled edit, while the Fisher geometry determines how strongly the downstream policy responds to that displacement around the original state.

Definition 5 (World-edit Fisher sensitivity). For a base state s, controlled environment edit $e ,$ and representation site m, let

$$
\Delta _ { e } ^ { ( m ) } ( s ) = h ^ { ( m ) } ( s _ { e } ) - h ^ { ( m ) } ( s )
$$

be the world-edit representation displacement from Definition 2. We define the world-edit Fisher sensitivity

$$
Q _ { e } ^ { ( m ) } ( s ) = \Delta _ { e } ^ { ( m ) } ( s ) ^ { \top } F ^ { ( m ) } ( s ) \Delta _ { e } ^ { ( m ) } ( s ) .
$$

When the edit changes the intended-goal variable, we write

$$
e = \star , \qquad s _ { \star } , \qquad \Delta _ { \star } ^ { ( m ) } ( s ) .
$$

When a corresponding proxy edit is well defined, we analogously write

$$
e = p , \qquad s _ { p } , \qquad \Delta _ { p } ^ { ( m ) } ( s ) .
$$

Not every environment admits both kinds of edit.

By Proposition 1,

$$
\frac { 1 } { 2 } Q _ { e } ^ { ( m ) } ( s )
$$

is the second-order local approximation to the change in the downstream policy induced by moving from the base representation toward its world-edited counterpart:

$$
D _ { \mathrm { K L } } ( \pi ^ { ( m ) } ( \cdot \mid h ^ { ( m ) } ( s ) ) ) \Big \parallel \pi ^ { ( m ) } ( \cdot \mid h ^ { ( m ) } ( s ) + \Delta _ { e } ^ { ( m ) } ( s ) ) ) \approx \frac { 1 } { 2 } Q _ { e } ^ { ( m ) } ( s ) .
$$

Because

$$
h ^ { ( m ) } ( s ) + \Delta _ { e } ^ { ( m ) } ( s ) = h ^ { ( m ) } ( s _ { e } ) ,
$$

the finite endpoints are the same as those compared by the behavioral world-edit KL,

$$
D _ { \mathrm { K L } } \left( \pi ( \cdot \mid s ) \parallel \pi ( \cdot \mid s _ { e } ) \right) .
$$

The two quantities differ in what they expose. The behavioral KL measures the complete policy change between the original and edited inputs. $Q _ { e } ^ { ( m ) } ( s )$ instead expresses the corresponding change locally through representation site $m ,$ and therefore measures how strongly the downstream policy is coupled to the representation displacement produced by edit e. A large or linearly informative displacement can consequently have small Fisher sensitivity if the trained downstream computation is comparatively insensitive to its direction.

Proposition 2 (Dependence on the trained downstream computation). Let

$$
\eta ^ { ( m ) } ( h ) \in \mathbb { R } ^ { | \mathcal { A } | }
$$

denote the logits produced by the downstream network from representation h, and let

$$
J _ { \eta } ^ { ( m ) } ( h ) = \frac { \partial \eta ^ { ( m ) } ( h ) } { \partial h }
$$

denote its Jacobian. Then the activation-space Fisher can be written as

$$
F _ { h } = J _ { \eta } ^ { ( m ) } ( h ) ^ { \top } \left[ \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top } \right] J _ { \eta } ^ { ( m ) } ( h ) ,
$$

where $\pi = \pi ^ { ( m ) } ( \cdot \mid h )$

Proof. For a softmax distribution, the Fisher information with respect to the logits is

$$
F _ { \eta } = \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top } .
$$

Applying the chain rule to $\nabla _ { h } \log \pi ^ { ( m ) } ( a \mid h )$ pulls this metric back through the downstream logit map, giving

$$
\begin{array} { r } { F _ { h } = J _ { \eta } ^ { \top } F _ { \eta } J _ { \eta } . } \end{array}
$$

At the final fully connected representation, the downstream logit map is linear,

$$
\eta ^ { ( \mathrm { f c } ) } ( h ) = W _ { \pi } h + b _ { \pi } ,
$$

so

$$
F ^ { ( \mathrm { f c } ) } ( s ) = W _ { \pi } ^ { \top } \left[ \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top } \right] W _ { \pi } .
$$

At earlier representation sites, such as $h _ { \mathrm { p r e } } ,$ the Jacobian additionally contains the downstream transformations between that site and the policy head.

Gaussian policies. The robot agents are read at the same two sites, with $h _ { \mathrm { p r e } }$ the rectified, flattened output of the last convolution and $h _ { \mathrm { f c } }$ the following linear layer with a ReLU (Appendix B.1). Their policy is a Gaussian over continuous actions,

$$
\pi ^ { ( m ) } ( \cdot \mid h ) = \mathcal { N } \big ( \mu ^ { ( m ) } ( h ) , \mathrm { d i a g } ( \sigma ^ { 2 } ) \big ) , \qquad \sigma = \exp ( \lambda ) ,
$$

where $\lambda$ is a learned vector with one entry per action dimension that is not computed from the observation. The softmax closed form of Proposition 2 does not apply, and its place is taken by the following result.

Proposition 3 (Fisher of a Gaussian policy with state-independent noise). Let $\pi ^ { ( m ) } ( \cdot , \mid h ) =$ $\mathcal { N } ( \mu ^ { ( m ) } ( h ) , \mathrm { d i a g } ( \sigma ^ { 2 } ) )$ with σ independent of $^ { \mathrm { ~ \cdot ~ } } h ,$ , and let

$$
J _ { \mu } ^ { ( m ) } ( h ) = \frac { \partial \mu ^ { ( m ) } ( h ) } { \partial h }
$$

be the Jacobian of the action mean with respect to the representation. Then

$$
F _ { h } = J _ { \mu } ^ { ( m ) \top } \mathrm { d i a g } ( \sigma ^ { - 2 } ) J _ { \mu } ^ { ( m ) } ,\tag{3}
$$

so the world-edit Fisher sensitivity is

$$
Q _ { e } ^ { ( m ) } ( s ) = \big \| \mathrm { d i a g } ( \sigma ^ { - 1 } ) J _ { \mu } ^ { ( m ) } \Delta _ { e } ^ { ( m ) } ( s ) \big \| _ { 2 } ^ { 2 } ,
$$

and thefinite KL between the base and the edited state is exactly

$$
\begin{array} { r } { D _ { \mathrm { K L } } \big ( \pi ( \cdot  { | } s )  { \| } \pi ( \cdot  { | } s _ { e } ) \big ) = \frac { 1 } { 2 }  { \big \| } \mathrm { d i a g } ( \sigma ^ { - 1 } ) \big ( \mu ( s _ { e } ) - \mu ( s ) \big )  { \big \| } _ { 2 } ^ { 2 } , } \end{array}
$$

with $\mu ( s ) = \mu ^ { ( m ) } \bigl ( h ^ { ( m ) } ( s ) \bigr )$

Proof. Write $\mu = \mu ^ { ( m ) } ( h )$ and $J _ { \mu } = J _ { \mu } ^ { ( m ) } ( h )$ . The log-density is

$$
\begin{array} { r } { \log \pi ^ { ( m ) } ( a \mid h ) = - \frac { 1 } { 2 } ( a - \mu ) ^ { \top } \operatorname { d i a g } ( \sigma ^ { - 2 } ) ( a - \mu ) + c ( \sigma ) , } \end{array}
$$

where $c ( \sigma )$ collects the normalization and does not depend on h. Differentiating with respect to h gives the score

$$
\nabla _ { h } \log \pi ^ { ( m ) } ( a \mid h ) = J _ { \mu } ^ { \top } \operatorname { d i a g } ( \sigma ^ { - 2 } ) ( a - \mu ) .
$$

Under $a \sim \pi ^ { ( m ) } ( \cdot \mid h )$ the residual has covariance

$$
\mathbb { E } \big [ ( a - \mu ) ( a - \mu ) ^ { \top } \big ] = \mathrm { d i a g } ( \sigma ^ { 2 } ) ,
$$

so the second moment of the score is

$$
F _ { h } = J _ { \mu } ^ { \top } \mathrm { d i a g } ( \sigma ^ { - 2 } ) \mathrm { d i a g } ( \sigma ^ { 2 } ) \mathrm { d i a g } ( \sigma ^ { - 2 } ) J _ { \mu } = J _ { \mu } ^ { \top } \mathrm { d i a g } ( \sigma ^ { - 2 } ) J _ { \mu } .
$$

Inserting $\Delta _ { e } ^ { ( m ) } ( s )$ into $\Delta ^ { \top } F _ { h } \Delta$ gives the expression for ${ Q } _ { e } ^ { ( m ) } { ( s ) }$ . For two Gaussians with the same covariance Σ the trace and log-determinant terms of the KL cancel, leaving

$$
D _ { \mathrm { K L } } \big ( \mathcal { N } ( { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) \big | \big | \mathcal { N } ( { \boldsymbol { \mu } } _ { 2 } , { \boldsymbol { \Sigma } } ) \big ) = \frac { 1 } { 2 } ( { \boldsymbol { \mu } } _ { 2 } - { \boldsymbol { \mu } } _ { 1 } ) ^ { \top } { \boldsymbol { \Sigma } } ^ { - 1 } ( { \boldsymbol { \mu } } _ { 2 } - { \boldsymbol { \mu } } _ { 1 } ) ,
$$

and $\Sigma = \mathrm { d i a g } ( \sigma ^ { 2 } )$ gives the last claim.

Equation (3) is the Fisher of the policy itself, not an approximation, and it plays the role of the categorical closed form above. Because the noise does not depend on the input, the KL between two states is exactly quadratic in the change of the mean, and the quadratic approximation differs from it only through the curvature of the mean along the displacement,

$$
\begin{array} { r } { \frac { 1 } { 2 } Q _ { e } ^ { ( m ) } ( s ) = \frac { 1 } { 2 } \big \| \mathrm { d i a g } ( \sigma ^ { - 1 } ) J _ { \mu } ^ { ( m ) } \Delta _ { e } ^ { ( m ) } ( s ) \big \| _ { 2 } ^ { 2 } , \quad } \\ { \mu ( s _ { e } ) - \mu ( s ) = J _ { \mu } ^ { ( m ) } \Delta _ { e } ^ { ( m ) } ( s ) + O \big ( \| \Delta _ { e } ^ { ( m ) } ( s ) \| _ { 2 } ^ { 2 } \big ) . } \end{array}
$$

For a Gaussian policy the ratio of the exact KL to $\scriptstyle { \frac { 1 } { 2 } } Q$ can therefore depart from 1 only where $\mu ^ { ( m ) }$ bends along the edit, never through the shape of the action distribution.

In our robot agents the mean comes from a head with one hidden layer of ReLU units, which reads $h _ { \mathrm { f c } }$ together with an embedding $h _ { \mathrm { s t a t e } }$ of the robot’s body state (Appendix B.1),

$$
\mu = W _ { \mathrm { 2 } } \mathrm { R e L U } ( z ) + b _ { \mathrm { 2 } } , \qquad z = W _ { \mathrm { 1 } } \big [ h _ { \mathrm { f c } } ; h _ { \mathrm { s t a t e } } \big ] + b _ { \mathrm { 1 } } .
$$

Write $W _ { 1 } ^ { \mathrm { f c } }$ for the columns of $W _ { 1 }$ that read $h _ { \mathrm { f c } }$ . With $h _ { \mathrm { s t a t e } }$ held at its base value, the Jacobian of the mean at the two sites is

$$
\begin{array} { r l } & { J _ { \mu } ^ { \mathrm { ( f c ) } } = W _ { 2 } \mathrm { d i a g } \left( \mathbf { 1 } \{ z > 0 \} \right) W _ { 1 } ^ { \mathrm { f c } } , } \\ & { J _ { \mu } ^ { \mathrm { ( p r e ) } } = J _ { \mu } ^ { \mathrm { ( f c ) } } \mathrm { d i a g } \left( \mathbf { 1 } \{ W _ { \mathrm { f c } } h _ { \mathrm { p r e } } + b _ { \mathrm { f c } } > 0 \} \right) W _ { \mathrm { f c } } , } \end{array}
$$

with the indicator taken entrywise. Both depend on the state only through the two ReLU patterns. An edit that leaves the body state unchanged leaves $h _ { \mathrm { s t a t e } }$ unchanged as well, so its displacement lies in the image branch alone. The factorization is specific to this head, whereas Equation (3) holds for any mean network as long as σ is state independent.

Proposition 4 (Fisher sensitivity at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } } )$ . Let

$$
J = \frac { \partial h _ { \mathrm { f c } } } { \partial h _ { \mathrm { p r e } } } = \mathrm { d i a g } \left( { \bf 1 } \{ W _ { \mathrm { f c } } h _ { \mathrm { p r e } } + b _ { \mathrm { f c } } > 0 \} \right) W _ { \mathrm { f c } } .
$$

The policy depends on $h _ { \mathrm { p r e } }$ only through $h _ { \mathrm { f c } } ,$ , so the Fisher pulls back exactly,

$$
F _ { \mathrm { p r e } } = J ^ { \top } F _ { \mathrm { f c } } J .
$$

Ifthe displacements ofan edit satisfy $\Delta _ { \mathrm { f c } } = J \Delta _ { \mathrm { p r e } } , t h e n Q _ { \mathrm { p r e } } = Q _ { \mathrm { f c } }$

Proof. By the chain rule $\nabla _ { h _ { \mathrm { p r } \epsilon } }$ log $\pi = J ^ { \top } \nabla _ { h _ { \mathrm { f } } }$ log π, and taking the second moment of both sides gives $F _ { \mathrm { p r e } } = J ^ { \top } F _ { \mathrm { f c } } J$ . Then $\begin{array} { r } { Q _ { \mathrm { p r e } } = ( J \Delta _ { \mathrm { p r e } } ) ^ { \top } F _ { \mathrm { f c } } ( J \Delta _ { \mathrm { p r e } } ) = \Delta _ { \mathrm { f c } } ^ { \top } F _ { \mathrm { f c } } \Delta _ { \mathrm { f c } } = Q _ { \mathrm { f c } } } \end{array}$ □

To first order the edit-induced displacements satisfy $\Delta _ { \mathrm { f c } } \approx J \Delta _ { \mathrm { p r e } }$ , and therefore

$$
\begin{array} { r } { Q _ { \mathrm { p r e } } = \Delta _ { \mathrm { p r e } } ^ { \top } J ^ { \top } F _ { \mathrm { f c } } J \Delta _ { \mathrm { p r e } } \approx \Delta _ { \mathrm { f c } } ^ { \top } F _ { \mathrm { f c } } \Delta _ { \mathrm { f c } } = Q _ { \mathrm { f c } } . } \end{array}
$$

Q is thus approximately the same at both sites by construction and not as a property of the data, and the remaining differences come from the nonlinearity of the fc layer over a finite edit. The argument holds for every network in which $h _ { \mathrm { f c } }$ is computed from $h _ { \mathrm { p r e } }$ alone, which includes the robot agents, since the body state joins only after $h _ { \mathrm { f c } }$

Proposition 2 makes the interpretation of $Q$ precise. Fisher sensitivity is a property of the representation as read by the trained downstream policy. Two agents can make the same information available in their internal representations while inducing very different Fisher geometries because their downstream computations use that information differently. A small ${ Q } _ { e } ^ { ( m ) }$ therefore does not imply that information associated with edit e is absent from the representation.

Separating displacement from directional sensitivity. The quadratic form Q combines two effects. A world edit can produce a large policy response because it moves the representation far, because the downstream policy is unusually sensitive to the direction of that movement, or both. We separate these effects relative to the policy’s average local Fisher sensitivity.

Definition 6 (Fisher sensitivity decomposition). For a nonzero displacement $\Delta \in \mathbb { R } ^ { d } ,$ , define its directional Fisher sensitivity

$$
S ( \Delta ) = \frac { \Delta ^ { \top } F _ { h } \Delta } { \| \Delta \| _ { 2 } ^ { 2 } } = \widehat { \Delta } ^ { \top } F _ { h } \widehat { \Delta } , \qquad \widehat { \Delta } = \frac { \Delta } { \| \Delta \| _ { 2 } } .
$$

For an isotropically sampled unit vector $u \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$

$$
\mathbb { E } _ { u } \left[ u ^ { \top } F _ { h } u \right] = \frac { { \mathrm { t r } } ( F _ { h } ) } { d } .
$$

We therefore define the relative directional gain

$$
G ( \Delta ) = \frac { S ( \Delta ) } { \mathrm { t r } ( F _ { h } ) / d }
$$

The relative gain $G$ is defined whenever tr $\left( F _ { h } \right) > 0 .$ . If the local Fisher vanishes identically, then $Q = 0$ for every direction.

We also define the displacement in Fisher units

$$
D _ { F } ( \Delta ) = \| \Delta \| _ { 2 } ^ { 2 } \frac { \mathrm { t r } ( F _ { h } ) } { d } .
$$

The two quantities factor the Fisher sensitivity exactly:

$$
Q = \underbrace { G } _ { \substack { \mathrm { r e l a t i v e ~ d i r e c t i o n a l ~ g a i n ~ } \mathrm { d i s p l a c e m e n t ~ i n ~ F i s h e r ~ u n i t s } } } .
$$

$D _ { F }$ measures how far the counterfactual moves the representation in units of the policy’s average local sensitivity. G measures whether the direction of that movement is more or less policy-sensitive than an average direction in the same representation space. Thus

$$
G > 1
$$

means that the counterfactual moves along a direction that is privileged by the downstream policy relative to an isotropic direction. In the experiments, we additionally compare this directional gain with matched-norm random directions to account for the non-isotropic spectrum of $F _ { h }$

Proposition 5 (Rescaling invariance). Consider a scalar reparameterization

$$
h ^ { \prime } = c h
$$

together with a compensating downstream transformation that leaves the represented policy unchanged. Then

$$
Q ^ { \prime } = Q , \qquad G ^ { \prime } = G , \qquad D _ { F } ^ { \prime } = D _ { F } ,
$$

whereas $\lVert \Delta \rVert _ { 2 }$ and S individually are not invariant.

Proof. Under the reparameterization,

$$
\Delta ^ { \prime } = c \Delta .
$$

Because the downstream computation is compensated to preserve the policy, gradients with respect to the new representation scale as $c ^ { - 1 }$ , giving

$$
F _ { h ^ { \prime } } = c ^ { - 2 } F _ { h } .
$$

Therefore,

$$
\begin{array} { r } { Q ^ { \prime } = ( c \Delta ) ^ { \top } ( c ^ { - 2 } F _ { h } ) ( c \Delta ) = Q . } \end{array}
$$

Furthermore,

$$
S ^ { \prime } = c ^ { - 2 } S , \qquad { \frac { \mathrm { t r } ( F _ { h ^ { \prime } } ) } { d } } = c ^ { - 2 } { \frac { \mathrm { t r } ( F _ { h } ) } { d } } ,
$$

so

$$
G ^ { \prime } = G .
$$

Finally,

$$
D _ { F } ^ { \prime } = \| c \Delta \| _ { 2 } ^ { 2 } \frac { \mathrm { t r } ( c ^ { - 2 } F _ { h } ) } { d } = D _ { F } .
$$

This invariance matters when comparing different trained networks. The scale of an internal representation is not itself identifiable, so neither the raw displacement norm nor its unnormalized directional sensitivity should be interpreted across agents in isolation. Q, G, and $D _ { F }$ remain unchanged under this elementary rescaling.

A change of site is not such a rescaling. G and $D _ { F }$ depend on $\| \Delta \| _ { 2 }$ and on $\operatorname { t r } ( F _ { h } ) / d ,$ , which change with the coordinates and the dimension of the site, so unlike $Q$ (Proposition 4) they are not approximately preserved from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ . A change in the split between $G$ and $D _ { F }$ from one site to the other therefore describes how the fc layer re-expresses the same policy sensitivity, not where goal information is lost, and we compare G and $D _ { F }$ only between cohorts at the same site.

Corollary 1 (Dependence on policy sharpness). As the downstream action distribution approaches a point mass,

$$
\pi  e _ { a } ,
$$

we have

$$
\mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top }  0 .
$$

Consequently,

$$
F _ { h }  0
$$

and

$$
Q ( \Delta )  0
$$

for every fixed displacement ∆.

Thus a more deterministic policy can exhibit smaller Fisher sensitivity along every representation direction, independently of the particular feature being studied. We therefore report the average Fisher sensitivity

$$
\frac { \mathrm { t r } ( F _ { h } ) } { d }
$$

per cohort next to ${ \cal D } _ { F } \left( \mathrm { T a b l e } 9 \right)$ when comparing policies with different levels of sharpness.

For a Gaussian policy sharpness acts the other way. By Proposition 3 the Fisher grows as $\sigma ^ { - 2 }$ , so a sharper policy reads every direction more strongly and a smaller σ raises $\mathrm { t r } ( \bar { F } _ { h } ) / d$ rather than lowering it.

Definition 7 (Fisher cosine). For two directions $\Delta _ { 1 }$ and $\Delta _ { 2 }$ satisfying

$$
\Delta _ { i } ^ { \top } F _ { h } \Delta _ { i } > 0 , \qquad i \in \{ 1 , 2 \} .
$$

evaluated at the same base representation h under the same local Fisher metric $F _ { h } ,$ , we define

$$
\cos _ { F } ( \Delta _ { 1 } , \Delta _ { 2 } ) = \frac { \Delta _ { 1 } ^ { \top } F _ { h } \Delta _ { 2 } } { \sqrt { ( \Delta _ { 1 } ^ { \top } F _ { h } \Delta _ { 1 } ) ( \Delta _ { 2 } ^ { \top } F _ { h } \Delta _ { 2 } ) } } .
$$

The Fisher cosine measures alignment specifically in directions to which the downstream policy is locally sensitive. Values near +1 indicate aligned policy-relevant effects, values near 0 indicate local Fisher orthogonality, and values near −1 indicate opposing policy-relevant effects. We use this quantity later to characterize the geometry of cue competition in ATTRIBUTE MAZE.

Low-rank Fisher audit. Because the activation-space Fisher matrix has rank at most $| { \mathcal { A } } | - 1 = 1 4$ in this environment, we additionally test whether low Fisher rank alone can induce extreme pairwise cosine values. Matched random 256-dimensional vectors evaluated under rank-14 random Fisher metrics produce essentially no pairs with $| \cos _ { F } | > 0 . 9$ , whereas 82–86% of real state pairs exceed this threshold across the three GM policies on the black background. The near-±1 Fisher structure therefore cannot be explained by Fisher rank alone.

Definition 8 (Signed target-selection derivative). Let

$$
A _ { \star } ( s ) , A _ { p } ( s ) \subseteq A
$$

denote the intended-goal- and proxy-consistent action sets for state s. In MAZE and ATTRIBUTE MAZE, both sets are taken from the base state: $A _ { \star } ( s )$ is the first step toward the base cheese (respectively the line candidate) and $A _ { p } ( s )$ the first step toward the top-right corner (respectively the yellow candidate). In COIN, the base state holds the coin at a cell right of the agent, on the shortcut side, where the two sets would coincide, and the edit moves it to the cell one step left ofthe agent. So $A _ { \star } ( s ) = A _ { \mathrm { l e f t } }$ , the leftward actions toward the relocated coin, and $A _ { p } ( s ) = A _ { \mathrm { r i g h t } }$ , the rightward shortcut. At representation site m, define the target-selection contrast

$$
C _ { s } ^ { ( m ) } ( h ) = \sum _ { a \in A _ { \star } ( s ) } \pi ^ { ( m ) } ( a \mid h ) - \sum _ { a \in A _ { p } ( s ) } \pi ^ { ( m ) } ( a \mid h ) .
$$

For world edit e with displacement $\Delta _ { e } ^ { ( m ) } ( s )$ , the signed target-selection derivative is

$$
\begin{array} { r } { \partial _ { \Delta _ { e } } C _ { s } ^ { ( m ) } = \nabla _ { h } C _ { s } ^ { ( m ) } \left( h ^ { ( m ) } ( s ) \right) ^ { \top } \Delta _ { e } ^ { ( m ) } ( s ) . } \end{array}
$$

The action sets are determinedfrom the environment state s and are heldfixed when differentiating with respect to h. For a nonzero displacement, we additionally define the displacement-normalized derivative

$$
\widetilde { \partial _ { \Delta _ { e } } C _ { s } ^ { ( m ) } } = \frac { \partial _ { \Delta _ { e } } C _ { s } ^ { ( m ) } } { \| \Delta _ { e } ^ { ( m ) } ( s ) \| _ { 2 } } .
$$

The raw derivative measures the first-order target-selective effect of the actual representation change induced by edit e. Its sign indicates which side of the intended-versus-proxy contrast is favored, while its magnitude combines direction and edit size. The normalized derivative removes the Euclidean magnitude of the displacement and therefore measures target-selective sensitivity per unit representation change.

Unlike the Fisher quadratic $Q ,$ , the signed derivative is first-order and directional. $Q$ asks how strongly the downstream action distribution changes along a world-induced representation direction. $\partial _ { \Delta _ { e } } C _ { s } ^ { ( m ) }$ asks whether that same direction moves the behaviorally relevant target-selection contrast toward the intended or proxy-consistent side. So, the two measurements are complementary: Q measures whether the policy is sensitive to the world-induced displacement at all, while the signed derivative asks whether that sensitivity is organized along the behaviorally relevant target-selection contrast.

Input saliency. The same Fisher can be taken at the input instead of an internal site, which shows which parts of the observation the action distribution depends on. Write o for the observation, with $_ { O _ { c , x } }$ the value of colour channel c at pixel x, and treat o as a site, so that $F ^ { \mathrm { ( i n ) } } ( o )$ is the activationspace Fisher of Definition 4 with $h = o .$

Definition 9 (Policy-Fisher input saliency and enrichment). The policy-Fisher input saliency of pixel x is the diagonal of the input Fisher summed over the colour channels of x,

$$
S ( x ) = \sum _ { c } F _ { ( c , x ) , ( c , x ) } ^ { \mathrm { ( i n ) } } ( o ) = \sum _ { c } \mathbb { E } _ { a \sim \pi ( \cdot | o ) } \left[ \left( \frac { \partial \log \pi ( a \mid o ) } { \partial o _ { c , x } } \right) ^ { 2 } \right] .
$$

For a region R of |R| pixels in an image of N pixels, the enrichment of R is the saliency per pixel in R relative to the saliency per pixel of the whole image,

$$
E _ { R } ( o ) = \frac { N } { | R | } \frac { \sum _ { x \in R } S ( x ) } { \sum _ { x } S ( x ) } ,
$$

and over states $O 1 , \ldots , O _ { n }$ we report its geometric mean

$$
E _ { R } = \exp \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log E _ { R } ( o _ { i } ) \Big ) .
$$

By Proposition 1 at the input, changing channel c of pixel $x$ by a small ε moves the policy by $\textstyle \frac { 1 } { 2 } \varepsilon ^ { 2 } F _ { ( c , x ) , ( c , x ) } ^ { \mathrm { ( i n ) } } + O ( \varepsilon ^ { 3 } )$ in KL, so ${ \textstyle { \frac { 1 } { 2 } } } \varepsilon ^ { 2 } S ( x )$ is this local KL summed over the channels of $x .$ For a softmax policy with logits z,

$$
\frac { \partial \log \pi ( a \mid o ) } { \partial o _ { c , x } } = \frac { \partial z _ { a } } { \partial o _ { c , x } } - \sum _ { b } \pi ( b \mid o ) \frac { \partial z _ { b } } { \partial o _ { c , x } } ,
$$

and for a Gaussian policy Proposition 3 applies at the input with the body state held fixed, so S has a closed form for both policy classes,

$$
S _ { \mathrm { s o f t m a x } } ( x ) = \sum _ { c } { \mathrm { V a r } } _ { a \sim \pi ( \cdot | o ) } \left[ \frac { \partial z _ { a } } { \partial o _ { c , x } } \right] , \qquad S _ { \mathrm { G a u s s } } ( x ) = \sum _ { c } \sum _ { k } \frac { 1 } { \sigma _ { k } ^ { 2 } } \left( \frac { \partial \mu _ { k } } { \partial o _ { c , x } } \right) ^ { 2 } .
$$

The softmax saliency shrinks everywhere as the policy becomes nearly deterministic, and the Gaussian saliency grows everywhere as σ shrinks (Corollary 1 and the remark after it). Enrichment divides by the image mean of the same state and removes this global scale, so $\log _ { 1 0 } E _ { R } = 0$ means that the region counts as much as an average pixel, whatever the sharpness of the policy.

How $S ( x )$ differs from other saliency. Gradient saliency (Simonyan et al., 2014) differentiates a single output, one action logit or the value, and perturbation or occlusion saliency measures how that output changes when a patch of the image is replaced. $S ( x )$ is a Fisher-weighted variant of gradient saliency. It concerns the whole action distribution rather than one logit, it is invariant to a shift of all logits by a common amount, and it weights each action by its probability, so it measures what the policy actually reads out. High-contrast sprites draw saliency in any network, so we read every region against randomly initialized networks, and since saliency decays with distance from the agent we compare cohorts at a fixed region. Saliency maps suggest hypotheses rather than explain behavior (Atrey et al., 2020). Our causal claims rest only on the world edits and the rollouts, never on the maps.

## B.3 PROJECTIONS AND VECTOR ANALYSIS

The constructions below define projections relative to a chosen reference direction $u ^ { ( m ) }$ . They do not assume that one fixed reference direction provides a global description of the representation geometry across states.

Euclidean projection. Let $\Delta ^ { ( m ) }$ denote a representation displacement at site $m ,$ and let $u ^ { ( m ) } \neq 0$ be a reference direction in the same representation space. We write

$$
\hat { u } ^ { ( m ) } = \frac { u ^ { ( m ) } } { \lVert u ^ { ( m ) } \rVert _ { 2 } } .
$$

The Euclidean projection of $\Delta ^ { ( m ) }$ onto the span of $u ^ { ( m ) }$ is

$$
\Delta _ { u , E } ^ { ( m ) } = \left( \hat { u } ^ { ( m ) \top } \Delta ^ { ( m ) } \right) \hat { u } ^ { ( m ) } ,
$$

with residual

$$
\Delta _ { \perp , E } ^ { ( m ) } = \Delta ^ { ( m ) } - \Delta _ { u , E } ^ { ( m ) } .
$$

By construction,

$$
\begin{array} { r } { \Delta _ { u , E } ^ { ( m ) ^ { \top } } \Delta _ { \perp , E } ^ { ( m ) } = 0 . } \end{array}
$$

This decomposition measures how the representation displacement is oriented in the ordinary Eu clidean geometry of activation space.

Although the Euclidean components are orthogonal in activation space, they are not generally orthogonal under the Fisher metric. Consequently,

$$
Q ^ { ( m ) } \Big ( \Delta ^ { ( m ) } \Big ) = Q ^ { ( m ) } \Big ( \Delta _ { u , E } ^ { ( m ) } \Big ) + Q ^ { ( m ) } \Big ( \Delta _ { \perp , E } ^ { ( m ) } \Big ) + 2 \Delta _ { u , E } ^ { ( m ) ^ { \top } } F ^ { ( m ) } ( s ) \Delta _ { \perp , E } ^ { ( m ) } .
$$

The cross term need not vanish because Euclidean orthogonality does not imply Fisher orthogonality.

Fisher-orthogonal projection. To obtain an additive decomposition of the Fisher quadratic, we instead define the projection using the local Fisher geometry. The downstream policy induces this local geometry through the activation-space Fisher matrix $F ^ { ( m ) } ( s )$ . When

$$
u ^ { ( m ) ^ { \top } } F ^ { ( m ) } ( s ) u ^ { ( m ) } > 0 ,
$$

we define the Fisher-orthogonal projection of $\Delta ^ { ( m ) }$ onto the span of $u ^ { ( m ) }$ as

$$
\Delta _ { u , F } ^ { ( m ) } = \frac { { u ^ { ( m ) } } ^ { \top } F ^ { ( m ) } ( s ) \Delta ^ { ( m ) } } { { u ^ { ( m ) } } ^ { \top } F ^ { ( m ) } ( s ) u ^ { ( m ) } } u ^ { ( m ) } ,
$$

with residual

$$
\Delta _ { \perp , F } ^ { ( m ) } = \Delta ^ { ( m ) } - \Delta _ { u , F } ^ { ( m ) } .
$$

The residual is orthogonal to the reference direction under the local Fisher metric,

$$
u ^ { ( m ) ^ { \top } } F ^ { ( m ) } ( s ) \Delta _ { \bot , F } ^ { ( m ) } = 0 .
$$

Consequently, the Fisher quadratic decomposes exactly,

$$
Q ^ { ( m ) } \left( \Delta ^ { ( m ) } \right) = Q ^ { ( m ) } \left( \Delta _ { u , F } ^ { ( m ) } \right) + Q ^ { ( m ) } \left( \Delta _ { \perp , F } ^ { ( m ) } \right) ,
$$

where, for any representation-space direction $v ,$

$$
Q ^ { ( m ) } ( v ) = v ^ { \top } F ^ { ( m ) } ( s ) v .
$$

The Euclidean and Fisher projections answer different questions. The Euclidean projection asks how much of the displacement lies along the reference direction in activation space. The Fisher projection instead asks how the policy-sensitive component of that displacement is distributed relative to the same direction.

Signed-effect decomposition. At a fixed base state, the local signed target-selection derivative is linear in the representation displacement. For the target-selection contrast $C _ { s } ^ { ( m ) } ( h )$

$$
\begin{array} { r } { \partial _ { \Delta } C _ { s } ^ { ( m ) } = \nabla _ { h } C _ { s } ^ { ( m ) } \left( h ^ { ( m ) } ( s ) \right) ^ { \top } \Delta ^ { ( m ) } . } \end{array}
$$

Therefore, for any additive decomposition

$$
\Delta ^ { ( m ) } = \Delta _ { 1 } ^ { ( m ) } + \Delta _ { 2 } ^ { ( m ) } ,
$$

we have

$$
\partial _ { \Delta } C _ { s } ^ { ( m ) } = \partial _ { \Delta _ { 1 } } C _ { s } ^ { ( m ) } + \partial _ { \Delta _ { 2 } } C _ { s } ^ { ( m ) } .
$$

This exact additivity applies to the local directional derivative. It does not imply that the finite change

$$
C _ { s } ^ { ( m ) } \Big ( h ^ { ( m ) } ( s ) + \Delta ^ { ( m ) } \Big ) - C _ { s } ^ { ( m ) } \Big ( h ^ { ( m ) } ( s ) \Big )
$$

decomposes additively for arbitrary finite displacements. Unlike the quadratic Fisher decomposition, the derivative decomposition follows directly from linearity and therefore does not depend on how the additive split was constructed.

Local and shared-axis descriptions. The projection constructions above can be used to compare a fully state-specific representation direction with a lower-dimensional shared description.

Let $d ( s )$ denote a state-dependent representation direction of interest. A fully local description retains this vector separately for every state. A shared-axis description instead asks whether the state dependence can be compressed to a fixed reference direction u together with state-dependent coefficients.

The simplest such approximation is

$$
d ( s ) \approx \alpha ( s ) u ,
$$

where $\alpha ( s )$ is a scalar that may vary across states. If the orientation of the shared direction is itself determined by a discrete state variable, we write

$$
d ( s ) \approx \sigma ( s ) \beta ( s ) u , \qquad \sigma ( s ) \in \{ - 1 , + 1 \} , \qquad \beta ( s ) \geq 0 .
$$

Here, $\sigma ( s )$ determines which orientation of the shared axis is active, while $\beta ( s )$ controls the local magnitude.

This form does not imply that the full representation geometry is one-dimensional. It only states that the component relevant to the particular displacement family under study may be well described by one shared axis after accounting for state-dependent sign and magnitude. The residual

$$
r ( s ) = d ( s ) - \sigma ( s ) \beta ( s ) u
$$

contains the remaining state-specific structure.

Shared-axis coefficients under Euclidean and Fisher geometry. For a fixed shared direction u, the state-dependent coefficient can be defined under different geometries.

In Euclidean activation space, the signed coordinate of $d ( s )$ along the unit direction

$$
{ \hat { u } } = { \frac { u } { \| u \| _ { 2 } } }
$$

is

$$
\alpha _ { E } ( s ) = \hat { u } ^ { \top } d ( s ) .
$$

The corresponding Euclidean projection is

$$
d _ { u , E } ( s ) = \alpha _ { E } ( s ) \hat { u } .
$$

Under the local Fisher geometry, the corresponding coefficient is

$$
\alpha _ { F } ( s ) = \frac { u ^ { \top } F ( s ) d ( s ) } { u ^ { \top } F ( s ) u } ,
$$

giving the Fisher-orthogonal projection

$$
d _ { u , F } ( s ) = \alpha _ { F } ( s ) u .
$$

The two coefficients answer different questions. $\alpha _ { E } ( s )$ measures how much of the displacement lies along u in ordinary activation space. $\alpha _ { F } ( s )$ measures the coefficient of the same shared direction after orthogonality is defined by local policy sensitivity.

The resulting residual

$$
d _ { \perp , F } ( s ) = d ( s ) - d _ { u , F } ( s )
$$

satisfies

$$
u ^ { \top } F ( s ) d _ { \bot , F } ( s ) = 0 ,
$$

and therefore

$$
Q ( d ( s ) ) = Q ( d _ { u , F } ( s ) ) + Q ( d _ { \bot , F } ( s ) ) .
$$

Interventions derived from these decompositions. The same distinction gives several intervention families.

A state-local intervention retains the complete direction,

$$
\delta _ { \mathrm { l o c a l } } ( s ; \beta ) = \beta d ( s ) .
$$

A shared-axis intervention retains only a fixed direction and a state-dependent sign and magnitude,

$$
\delta _ { \mathrm { s h a r e d } } ( s ; \beta ) = \beta \sigma ( s ) m ( s ) \hat { u } ,
$$

where m $, ( s ) \geq 0$ specifies the chosen local magnitude.

Removing the sign dependence gives the corresponding unsigned intervention,

$$
\delta _ { \mathrm { u n s i g n e d } } ( s ; \beta ) = \beta m ( s ) \hat { u } .
$$

Finally, using the Fisher projection gives

$$
\delta _ { F } ( s ; \beta ) = \beta \alpha _ { F } ( s ) u .
$$

These interventions separate different compression hypotheses. The local intervention retains the full state-dependent representation geometry. The shared-axis intervention then asks whether its causal effect can be reproduced using one common direction together with state-dependent gating. The unsigned intervention tests whether that gating is necessary. Finally, the Fisher-projected intervention retains only the component assigned to the shared direction by the local policy-sensitive geometry.

## B.4 ENVIRONMENT-SPECIFIC TARGET GEOMETRY

On the training distribution the intended goal and the proxy are confounded. At most states one and the same action moves the agent toward both, so the policy’s choice cannot tell us which of the two it is following. Behavior separates goal from proxy only at states where the two require different first actions, and finding these states requires knowing exactly which actions make progress toward each target. Walls make the straight-line bearing of a target a poor guide to this, so we define progress on the level’s grid graph. The resulting graph-exact action sets are used throughout the paper, both to choose evaluation states (Section E) and to score behavior on them.

This needs a local notion of progress toward a spatial target. The MDP state $s \in S$ describes the complete environment configuration, while navigation depends on the agent’s location within the spatial structure of the current level. We therefore introduce a spatial projection of the environment state.

Spatial state projection. For a fixed level $\xi ,$ let

$$
\mathcal { G } _ { \xi } = ( V _ { \xi } , E _ { \xi } )
$$

denote its traversable grid graph. Each vertex $v \in V _ { \xi }$ corresponds to a traversable cell, and an edge $( u , v ) \in E _ { \xi }$ indicates that the agent can move between the corresponding cells in one environment step.

We write

$$
x _ { \xi } : { \mathcal { S } }  V _ { \xi }
$$

for the spatial projection that maps a full environment state s to the cell occupied by the agent. Thus s retains the complete MDP state, while $x _ { \xi } ( s )$ extracts only the agent’s spatial location on level $\xi .$

For cells $u , v \in V _ { \xi }$ , we define the graph distance

$$
d _ { \xi } ( u , v ) = \operatorname* { m i n } \left\{ k \in { \mathbb { N } } _ { 0 } : \exists v _ { 0 } , \dotsc , v _ { k } \in V _ { \xi } , \ v _ { 0 } = u , \ v _ { k } = v , \ ( v _ { i - 1 } , v _ { i } ) \in E _ { \xi } \forall i = 1 , \dotsc , k \right\} .
$$

Hence $d _ { \xi } ( u , v )$ is the minimum number of valid grid transitions required to move from u to v. We compute this distance exactly using breadth-first search.

Spatial state-transition projection. The MDP transition kernel acts on complete environment states,

$$
s ^ { \prime } \sim P ( \cdot \mid s , a ) .
$$

For the navigation environments considered here, the one-step spatial transition is deterministic conditional on the full state s and action a. We therefore define

$$
x _ { \xi } ^ { \prime } ( s , a ) = x _ { \xi } ( s ^ { \prime } ) , \qquad \mathrm { w h e r e } \ P ( s ^ { \prime } \mid s , a ) = 1 ,
$$

as the agent cell after applying action a in state s on level $\xi .$ Thus $x _ { \xi } ( s )$ and $x _ { \xi } ^ { \prime } ( s , a )$ are respectively the spatial projections before and after the MDP transition. This notation introduces no additional transition model beyond $P .$

Definition 10 (Graph-exact target-progress action set). For a state s, level $\xi ,$ , and candidate target cell $v \in V _ { \xi }$ , define

$$
A _ { v } ( s ) = \{ a \in { \mathcal { A } } : d _ { \xi } \big ( x _ { \xi } ^ { \prime } ( s , a ) , v \big ) < d _ { \xi } ( x _ { \xi } ( s ) , v ) \} \subseteq \mathcal { A } .
$$

$A _ { v } ( s )$ is therefore the subset of actions whose induced spatial transition strictly decreases the $r e \mathrm { - }$ maining shortest-path distance to v.

Because the grid edges have unit cost, an action belongs to $A _ { v } ( s )$ exactly when its induced movement lies on a shortest path from the current agent cell toward v. An action from which v remains eventually reachable need not belong to $A _ { v } ( s )$ if it does not decrease the remaining shortest-path distance on that step.

When multiple raw policy actions induce the same effective movement, we aggregate their probability mass rather than treating action aliases as distinct navigation choices.

Two candidate cells $v _ { 1 } , v _ { 2 } \in V _ { \xi }$ are locally action-discriminating at state s when

$$
A _ { v _ { 1 } } ( s ) \neq A _ { v _ { 2 } } ( s ) .
$$

At such states, the two spatial target hypotheses make different predictions about immediate control and can therefore be distinguished from the native policy’s action distribution.

Environment-specific target action sets. In MAZE, let $v _ { \star }$ denote the cheese cell and $v _ { p }$ the proxyassociated top-right destination. We write

$$
A _ { \star } ( s ) \equiv A _ { v _ { \star } } ( s ) , \qquad A _ { p } ( s ) \equiv A _ { v _ { p } } ( s ) .
$$

Thus $A _ { \star } ( s )$ contains graph-exact progress actions toward the intended goal location, while $A _ { p } ( s )$ contains progress actions toward the proxy-associated destination.

In ATTRIBUTE MAZE, let $v _ { \mathsf { I i n e } }$ and $v _ { \mathsf { y e l l o w } }$ denote the locations of the line-bearing and yellow-bearing objects. Depending on the world edit under consideration, these locations instantiate the intended and proxy candidates. We again use

$$
A _ { \star } ( s ) \qquad \mathrm { a n d } \qquad A _ { p } ( s )
$$

as shorthand for the corresponding graph-exact action sets, while retaining $A _ { v _ { \mathsf { l i n e } } } ( s )$ and $A _ { v _ { \mathsf { y e l l o w } } } ( s )$ when the physical cue identity matters.

COIN does not use the grid-graph construction for its proxy. The learned shortcut is represented by the right-moving action family, and in every state of the COIN conflict bank the coin sits one cell to the left of the agent in the state on which target selection is read (Appendix E). Progress toward the coin is therefore leftward motion, and we use the two horizontal movement families

$$
A _ { \mathrm { l e f t } } , A _ { \mathrm { r i g h t } } \subseteq A _ { \mathrm { C o i n } } ,
$$

each of which holds the three actions that move the agent in that direction. For notational consistency in comparisons across environments, we write

$$
{ \cal A } _ { \star } ( s ) \equiv { \cal A } _ { \mathrm { l e f t } } , \qquad { \cal A } _ { p } ( s ) \equiv { \cal A } _ { \mathrm { r i g h t } } .
$$

A state is locally action-discriminating whenever

$$
A _ { \star } ( s ) \neq A _ { p } ( s ) .
$$

Action-discriminating evaluation states. For a level ξ, let $\quad S _ { \xi } \subseteq S$ denote the environment states belonging to that level. For two candidate action sets $A _ { 1 } ( s )$ and $A _ { 2 } ( s )$ , we call a state actiondiscriminating when

$$
A _ { 1 } ( s ) \neq A _ { 2 } ( s ) .
$$

For the intended and proxy-associated targets, we therefore define

$$
S _ { \mathrm { c o n f } } ( \xi ) = \{ s \in \mathcal { S } _ { \xi } : A _ { \star } ( s ) \neq A _ { p } ( s ) \} .
$$

These states are decision points at which progress toward the intended and proxy-associated targets implies different immediate action sets. Restricting evaluation to $S _ { \mathrm { c o n f } } ( \xi )$ therefore lets the observed policy action distinguish which target better organizes local behavior. We call the members of $\cal { S } _ { \mathrm { c o n f } } ( \xi )$ conflict states.

Strict conflict. A conflict state can still have progress sets that overlap, and an action in the overlap serves both targets and so reveals nothing about which one the policy follows. We therefore build every evaluation bank from states in strict conflict,

$$
A _ { \star } ( s ) \cap A _ { p } ( s ) = \emptyset ,
$$

at which every progress action serves exactly one of the two targets. In MAZE shortest paths are unique, so both sets contain a single action and every conflict state is a strict conflict. In COIN the left-moving and right-moving families are disjoint by construction. In ATTRIBUTE MAZE we additionally require the two candidates to be equidistant from the agent, so that neither target is favoured by being closer.

Continuous actions. The robot policies are Gaussian over actions $a \ \in \ \mathbb { R } ^ { m } , \ \pi ( \cdot \ | \ s ) \ =$ $\mathcal { N } \big ( \mu ( s ) , \mathrm { d i a g } ( \sigma ^ { 2 } ) \big )$ , with a state-independent standard deviation $\sigma ~ \in ~ \mathbb { R } _ { > 0 } ^ { m }$ . The first k components of an action move the robot’s effector, the gripper or the body, in the plane in which the task’s objects lie.

Definition 11 (Planned step and target direction). Let $M ( s ) \in \mathbb { R } ^ { 2 \times k }$ map the first k action components at state s to the motion of the effector in the task plane. The planned step is the motion commanded by the mean action,

$$
\mathrm { s t e p } ( s ) = M ( s ) \mu _ { 1 : k } ( s ) \in \mathbb { R } ^ { 2 } .
$$

For an object o with target point $y _ { o } ( s )$ in the task plane and the effector at $x ( s )$ , the target direction is the unit vector $\hat { t } _ { o } ( s ) = \left( y _ { o } ( s ) - x ( s ) \right) / \| y _ { o } ( s ) - x ( s ) \| _ { 2 }$

The planned step is taken on the unclipped mean, so it can exceed what the robot executes in one control step. The three kinds of robot task instantiate $M ( s )$ and $y _ { o } ( s )$ as follows.

Quadruped. In WALKTOFLAG the action is a walking command with $m = 3$ components, a forward and a sideways velocity in the robot’s own frame and a turn rate. The first $k = 2$ components already describe the motion in the floor plane, so $M ( s ) = I _ { 2 }$ and the planned step is $( \mu _ { 1 } ( s ) , \mu _ { 2 } ( s ) )$ , in command units between −1 and 1. The target point of the flag or the tower is the object’s position in the robot frame.

Arm. In the arm tasks the action has $m = 8$ components. The first $k = 7$ change the seven joint angles of the arm by at most 0.1 rad each, and the eighth opens or closes the gripper. How far the gripper moves when the joints turn depends on the arm’s posture. The Jacobian $\mathbf { \bar { \boldsymbol { J } } } ( q ) \in \mathbb { R } ^ { 3 \times 7 }$ at joint angles q gives, for each joint, the velocity of the tool centre point per unit change of that joint. The tool centre point is the point between the two gripper fingers, the point that grasps or pushes and that the arm’s controller positions. With $P \in \mathbb { R } ^ { 2 \times \overline { { 3 } } }$ keeping the two components in the table plane,

$$
M ( s ) = 0 . 1 P J ( q ) ,
$$

so the planned step is the table-plane motion of the tool centre point that the mean joint change produces, in millimetres. In LOCATIONREACH and ATTRIBUTETABLE the target point of an object is its centre.

Push tasks. In PUSHTOMARKER and SORTZONES the gripper does not move toward the marker or pad $g$ but toward the place from which a push carries the cube onto it. With the cube centre at $c ,$ the target point of g is the push point

$$
y _ { g } = c - 0 . 0 3 \frac { g - c } { \lVert g - c \rVert _ { 2 } } ,
$$

on the line from $g$ through the cube, 3 cm behind the cube centre. The cube’s half-width is 2 cm, so this point lies 1 cm behind its rear face, where the gripper must stand to push the cube straight toward $g .$ The dense reward of these tasks uses the same point (Appendix A.2).

## C EXPERIMENTAL DESIGN AND EXTENDED RESULTS

This appendix extends every analysis of Section 3 to all eight environments of Appendix A. Figure 13 summarizes the result. In each environment the true goal survives to a different depth of the agent, and this depth goes together with the intervention that recovers it. The failures fall into the three types of the main text. Target substitution appears in three environments, MAZE, LOCATION-REACH and PUSHTOMARKER, where a location-based proxy dominates and fully substitutes the target, even though the goal object is present in the scene. In MAZE the substitute target is a spatial relation, the top-right extreme, and in LOCATIONREACH and PUSHTOMARKER a fixed place in the camera image. The policy shortcut appears in COIN, where a very simple part of the policy, moving right, stands in for the goal object. Cue competition appears, in different grades, in four environments. In ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES it coincides with one object that carries two co-located characteristics, and WALKTOFLAG replicates it with two different objects that stand closely together in space. In all four we see a competition between the two characteristics, or between the two closely related objects.

To establish this, we walk through the agents with a ladder of questions at increasing depth. We start in the representation and ask whether information about the goal is still there, move to the readout and ask whether the policy reacts to it and uses it, and then look at what organizes behavior instead and how the proxy is incorporated into the agent, from the outside and from the inside. The ladder ends with recovery, where we ask whether goal-directed behavior can be restored without further reinforcement learning, by steering the existing policy or by learning a new readout from an earlier representation.

![](images/f876595bad1aecca87d9ba551306882b1eefb929bcaffc74b339c1d332d5abc7.jpg)  
Figure 13: How far does the goal get in each of the eight environments? The counterpart of Figure 1 for all environments, rows ordered from the shallowest to the deepest point the goal reaches. Arrow width is schematic. Steering is the corrective push at β = 1 (Table 14) and rewiring the teacher-target readout from $h _ { \mathrm { p r e } }$ (Table 18), with oracle targets in MAZE and ATTRIBUTE MAZE as in Figure 1 and for PUSHTOMARKER after three rounds of DAgger (Table 19). Filled bars are recovered and hatched bars partial under the success criterion of Appendix D.2.

## C.1 LINEAR DECODABILITY: IS INFORMATION ABOUT THE INTENDED GOAL STILL THERE?

Motivation. Goal misgeneralization tells us that an agent behaves according to something other than the intended goal, but this does not tell us what happened to information about the intended goal inside the agent. One simple possibility is that the relevant goal information is no longer available in the learned representation. We therefore begin with the most basic internal question: is information about the intended goal still there?

We test this first through linear decodability. A successful probe gives an existence result: the corresponding information is available in a form accessible to a linear readout. We then ask where this information appears across the network, whether its accessibility differs between GM and Non-GM agents, and how much of that accessibility can be attributed to training rather than to structure already exposed by the input or architecture.

Design. For each environment, we define goal-related and, where meaningful, proxy-related target variables and fit regularized linear probes (Alain & Bengio, 2017) to internal activations. The probe targets are environment-specific because the information relevant to the goal differs across tasks. For a binary target $y ,$ the probe at representation m has score

$$
f _ { m } ( s ) = w _ { m } ^ { \top } h ^ { ( m ) } ( s ) + b _ { m } ,
$$

and we evaluate $f _ { m }$ using held-out ROC–AUC. For continuous spatial targets, we fit the corresponding linear regressor

$$
\hat { y } _ { m } ( s ) = w _ { m } ^ { \top } h ^ { ( m ) } ( s ) + b _ { m }
$$

and evaluate it using held-out $R ^ { 2 }$ . Tables 22 and 23 list every probe with its label, bank, split and sites.

Following the architectures in Appendix B.1, we probe the same computational sites across the three Procgen environments,

$$
h ^ { ( 1 ) } = H ^ { ( 1 ) } , \qquad h ^ { ( 2 ) } = H ^ { ( 2 ) } , \qquad h ^ { ( 3 ) } = H ^ { ( 3 ) } ,
$$

where $H ^ { ( 1 ) } , H ^ { ( 2 ) } , H ^ { ( 3 ) }$ denote the corresponding residual-block representations. No additional ReLU is inserted after the first two blocks because no such operation occurs in the policy forward pass. The subsequent shared representations are

$$
h _ { \mathrm { p r e } } = \operatorname { v e c } \Big ( \operatorname { R e L U } ( h ^ { ( 3 ) } ) \Big )
$$

and

$$
h _ { \mathrm { f c } } = \mathrm { R e L U } \left( W _ { \mathrm { f c } } h _ { \mathrm { p r e } } + b _ { \mathrm { f c } } \right) .
$$

Thus $h _ { \mathrm { p r e } }$ is the actual input to the fully connected layer and $h _ { \mathrm { f c } }$ is the final shared representation consumed by the policy and value readouts. For ${ h ^ { ( 1 ) } , h ^ { ( 2 ) } , h ^ { ( 3 ) } }$ , we globally average pool the convolutional feature maps before fitting the probe. The later representations are used directly. We exclude policy logits and value predictions from this sequence because they are separate readout branches rather than later shared representations. In the four arm tasks we probe the unpooled feature maps $H ^ { ( 1 ) }$ and $H ^ { ( 2 ) }$ of the first two convolutions, then $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ . In WALKTOFLAG we probe the three convolutional maps average-pooled to $4 { \times } 4$ cells, $\bar { H } ^ { ( 1 ) } , \bar { H } ^ { ( 2 ) } , \bar { H } ^ { ( 3 ) }$ , then $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ . We do not pool the robot maps globally because pooling removes position: globally pooled $H ^ { ( 1 ) }$ and $H ^ { ( 2 ) }$ keep $R ^ { 2 } < 0 . 1 7$ of the marker position in PUSHTOMARKER, against 0.74 to 0.90 unpooled.

Every probe is fitted on a training partition and scored on held-out data: whole held-out levels in MAZE and COIN over five split seeds, held-out object configurations in ATTRIBUTE MAZE, and held-out start frames in the robot tasks. The Procgen probes are ridge regressions with fixed strength $\alpha = 1$ (logistic regression for COIN presence), and the robot probes are ridge regressions whose strength is selected by cross-validation on the training frames. All probe preprocessing is fitted on the training partition only.

Our primary quantity is absolute held-out probe performance. This answers the direct representational question: how linearly accessible is the target at each site? We note that absolute decodability alone, however, does not establish how much accessibility was produced by training. Simple visual or spatial structure can already be linearly exposed by an untrained convolutional network (Hermann & Lampinen, 2020).

![](images/1ff46e0efafc8103b5f908635c269e0587c1b544ad6e2a60c7152b47533a677b.jpg)

![](images/d3b45571ca8be06c6ae76d15ad40cde2b49e399cdd143bfe27a3b4f23f329fe3.jpg)

![](images/7dd94283aeaaa15bb22043335f623a89e4f03e9b38edb44e0aff4eb338309094.jpg)  
GM Non-GM random init (3 draws) input control 95% C

![](images/584303ade1fc7bef2782ce1efaa4440fd61b545c8b67e6bbad674d8cddb1c129.jpg)

Figure 14: Layerwise linear availability in MAZE. Absolute held-out ROC–AUC, from left to right, for cheese presence, cheese bearing relative to the agent, the same cheese bearing on the position-balanced set, in which agent position carries no information about the cheese bearing, and bearing to the reachable top-right corner. Points show the three checkpoint seeds, lines show cohort means, and shaded regions show 95% t-intervals across the three checkpoints. Grey dashed: three randomly initialized networks, with a 95% t-interval across draws. Dotted: input control, the cheesecell pixels for presence and agent position for all bearings.

Baselines. We compare every probe with three baselines. First, randomly initialized networks of the same architecture (three draws in every environment), probed on the same frames. Second, input-level or positional control probes, which probe the observation or a correlated state variable directly, for example the pixels of the cheese cell, the agent’s position or block-averaged pixels. They show how much information is trivially available in the input or carried by co-located objects and therefore need not be learned. Third, in the Procgen environments, permuted labels give an empirical null with the same grouping and splits as the probe. A score that an untrained network or an input control already reaches is not read as a training effect, so each result in the following first asks whether the probe separates from the baselines and only then compares GM with Non-GM agents.

MAZE probe construction. In MAZE, we first ask whether the cheese object itself is linearly detectable, using the probe bank of Appendix E.4 (Table 22). Standard MAZE states always contain the cheese, so the negative class consists of the same states with the cheese removed. We also probe the bearing of the cheese relative to the agent and, for the proxy, the bearing of the reachable top-right extreme of the maze.

MAZE results. Cheese presence is trivially decodable from the input, as a linear probe on the pixels of the cheese cell reaches an AUC of 0.988 (Figure 14). Moreover, Non-GM agents decode it just as well from $h ^ { ( 2 ) }$ onward (0.991 to 0.998), while GM agents stay at the level of random networks in the convolutional blocks and even below them at the later layers $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ (0.529 to 0.549 at every site, against 0.620 and 0.561 for random networks at the two later layers). The true goal is therefore far more linearly decodable in Non-GM than in GM agents, a separation in the expected direction. Because the presence of the cheese is already available in the input, however, a presence probe alone may not separate the cohorts cleanly, so we also probe the bearing of the cheese.

Cheese bearing separates from random networks in both cohorts from $h ^ { ( 2 ) }$ onward, in GM only narrowly at $h _ { \mathrm { p r e } }$ (0.712 against 0.686), and grows with depth, from chance at $h ^ { ( 1 ) }$ to about 0.8 in GM and 0.95 in Non-GM agents. The bearing probe thus separates, but we are unsure whether a subsidiary piece of information drives it, namely the agent’s own position. In GM agents the bearing stays close to and below what agent position alone gives (0.712 to 0.808 from $h ^ { ( 2 ) }$ onward, against 0.830), while Non-GM agents exceed it (0.873 to 0.951). With the corner bearing held fixed, so that only cheese-relative information helps, GM agents fall to 0.598 at $h _ { \mathrm { p r e } } ,$ , below both agent position (0.703) and random networks (0.729), while Non-GM agents keep 0.930 (Table 4). We test this directly with a decorrelation check, on a set in which the agent’s own position carries no information about the cheese bearing. To build it, we keep inside every agent cell as many frames with the cheese to the left of the agent as to its right, which leaves 902 of the 1,987 cheese frames and brings the agent-position probe to chance (0.481, Figure 14, right). On this set the cheese bearing in GM agents collapses at every site (0.516 to 0.583) and falls below random networks at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ (0.748 and 0.631), while Non-GM agents keep it from $h ^ { ( 2 ) }$ onward (0.861 to 0.940), and so does the pixel probe (0.895). This rules out that GM agents carry cheese-bearing information beyond their own position. The GM bearing probe only tracks the agent, while Non-GM agents and pixels keep the cheese bearing.

![](images/62c561015ef23b08eacbf5557bd052928e2a716f3410d33981b18845ed1d6791.jpg)  
Figure 15: Layerwise linear availability in COIN. Absolute held-out probe performance for coin presence (ROC–AUC) and residualized coin bearing $( R ^ { 2 } )$ . Points show the three checkpoint seeds, lines show cohort means, and shaded regions show 95% t-intervals across the three checkpoints. Grey dashed: three randomly initialized networks, with a 95% t-interval across draws. Dotted: input control, raw pixels for presence and agent position for bearing (raw pixels give $R ^ { 2 } = - 0 . 5 5 2$ for bearing). Right: coin bearing residualized on agent position as well as on progress, so that agent position decodes none of it.

And lastly, corner bearing, the proxy probe, is decoded well by both cohorts and far above random networks from $h ^ { ( 2 ) }$ onward, with GM agents slightly ahead (0.952 against 0.892 at $h ^ { ( 3 ) }$ , 0.963 against 0.945 at $h _ { \mathrm { p r e } } )$ . Agent position alone already predicts the corner bearing (AUC 0.959), so neither cohort carries corner information beyond the geometry of the sampled states. Across the two bearing probes, GM agents are somewhat worse than Non-GM agents for the true goal and slightly better for the proxy, a small effect that depends on the probe and the site.

Together we find that the cheese is trivially detectable from the observation, but GM training does not construct a strong cheese presence representation as seen in Non-GM agents. Cheese bearing information in GM agents is also largely accounted for by the agent position, whereas Non-GM agents contain substantially stronger goal-related spatial information. The MAZE failure is therefore compatible with a learned representation that preserves generic spatial structure while failing to construct the goal-specific features that characterize the Non-GM policy.

COIN probe construction. In COIN, we separate object availability from spatial information about the goal, using the probe bank of Appendix E.4 (Table 22). Unlike cheese presence in MAZE, coin presence varies naturally during ordinary COIN trajectories as the coin moves into and out of the visible frame. Coin bearing is residualized on level progress because horizontal progress is strongly coupled to both observation structure and the training-time rightward trajectory. For the continuous bearing target, the level-permutation null yields strongly negative held-out $\dot { R ^ { 2 } }$ because permuted labels are constant within levels, so we use $\dot { R } ^ { 2 } = 0$ as the chance reference.

COIN results. The coin presence probe separates from the baselines only in Non-GM agents $( { \mathrm { F i g } } -$ ure 15). A probe on the raw pixels of the coin already reaches 0.760, and GM agents stay at that level at every site from $h ^ { ( 2 ) }$ onward (0.724 to 0.775). They are above random networks in the convolutional blocks, but do not consistently exceed them at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ . Non-GM agents start at the same level (0.748 at $h ^ { ( 2 ) } )$ and approach ceiling from $h ^ { ( 3 ) }$ onward (0.991 to 0.995). This separation holds on both frame sources of the probe bank. The robust COIN result is therefore not that the coin becomes undecodable in GM agents, but that the goal is cleanly represented in Non-GM agents, whose training constructs a much stronger downstream coin presence representation than GM training does.

Coin bearing, by contrast, separates from every baseline in both cohorts. From $h ^ { ( 2 ) }$ to $h _ { \mathrm { p r e } } .$ , heldout $R ^ { 2 }$ is 0.376 to 0.582, against at most 0.028 for random networks, 0.152 for agent position and −0.552 for raw pixels, and the two cohorts do not differ (0.523 and 0.527 at $h _ { \mathrm { p r e } } )$ . At $h _ { \mathrm { f c } }$ the cohort ordering depends on which half of the bank is evaluated (0.131 for GM against 0.360 for Non-GM on the full bank, reversed on the later-stage half), so we do not interpret COIN at $h _ { \mathrm { f c } }$ . The bearing probe separates, but as in MAZE we are unsure whether the agent’s own position drives it, so we run the same decorrelation check. With 320 frames we do not subsample but residualize the bearing on agent position as well as on progress, fitted on the training frames, so that a probe on agent position decodes none of it $( R ^ { 2 } = - \breve { 0 } . 0 2 4$ , Figure 15, right). Coin bearing stays decodable in both cohorts beyond agent position from $h ^ { ( 2 ) }$ to $h _ { \mathrm { p r e } }$ (0.150 to 0.378), against at most −0.020 for random networks and −0.369 for pixels, and the same holds with position entered as binned indicators (0.341 and 0.219 at $h _ { \mathrm { p r e } }$ for GM and Non-GM). The values are lower than without the position term (0.523 and 0.527 at $h _ { \mathrm { p r e } } )$ , so part of the original signal was position, but unlike in MAZE the goal bearing in GM agents does not reduce to it. This separates COIN from MAZE.

COIN consequently provides a heterogeneous result across goal variables. Goal bearing is strongly accessible in both cohorts through $h _ { \mathrm { p r e } } ,$ , with no stable cohort separation there. Coin presence, however, becomes nearly perfectly accessible in Non-GM agents while remaining substantially weaker in GM agents at the downstream shared representations. The robust representational distinction in COIN is therefore the organization of goal-object presence rather than a stable difference in goal bearing. Trinh et al. (2024) suggest that a CoinRun agent trained with the coin at the right end does not represent the coin, based on an anomaly detector and a learned help-request policy applied to its penultimate-layer features. Their analysis differs from ours in method and site, since neither tool measures supervised linear decodability, and the penultimate layer corresponds to $h _ { \mathrm { f c } }$ in our architecture, where GM coin presence is near the raw-pixel level.

ATTRIBUTE MAZE probe construction. ATTRIBUTE MAZE requires a different decomposition because the intended and proxy cues are attributes of objects rather than separate spatial targets. Simply detecting a feature does not establish whether the representation associates that feature with the rewarded object. We therefore distinguish feature presence from goal role and probe four binary concepts,

$$
L _ { \mathrm { p r e s } } , \qquad Y _ { \mathrm { p r e s } } , \qquad L _ { \mathrm { r o l e } } , \qquad Y _ { \mathrm { r o l e } } ,
$$

where $L _ { \mathrm { p r e s } }$ and $Y _ { \mathrm { p r e s } }$ indicate whether the line and yellow attributes are present, and $L _ { \mathrm { r o l e } }$ and $Y _ { \mathrm { r o l e } }$ indicate whether the corresponding attribute belongs to the rewarded object. The first pair measures feature availability. The second pair measures whether feature identity is available in relation to the rewarded role.

To separate presence from role, we use the configuration grid of the probe bank (Appendix E.4, Table 22) rather than sampling only from the ordinary task distribution. The grid includes states in which an attribute is visible but belongs to the lure, allowing “feature present” and “feature belongs to the rewarded object” to disagree, and both candidate objects are independently placed on uniformly sampled free cells. For the line attribute, presence and rewarded role disagree in 4/12 training configurations and $2 / 6$ held-out configurations. The yellow distinction has lower support, with $Y _ { \mathrm { p r e s } }$ and $Y _ { \mathrm { r o l e } }$ differing in only one configuration in each split. We therefore interpret the yellow-role probe more cautiously and do not assume that the two role probes provide equally strong separation of feature identity from rewarded role.

ATTRIBUTE MAZE results. Figure 16 shows the corresponding cue-presence and cue-role probes across the textured and black-background cohorts. ATTRIBUTE MAZE does not show a uniform suppression of either competing cue family. Across backgrounds, training increases linear accessibility of line and yellow information at the convolutional sites, while the role probes additionally show accessibility of information about which attribute belongs to the rewarded object. Accessibility at the two downstream shared representations is more variable. $\mathrm { A t } \ : h _ { \mathrm { p r e } } .$ , black-background GM line presence does not rise above initialization (0.792 versus 0.810, Table 4), and on the black background line presence is largely decodable from raw pixels (0.975). Yellow presence and yellow role exceed both random networks and the pixel probe at this site in both cohorts, whereas line role separates from neither (0.632 and 0.636 against 0.603 and 0.629).

So to conclude, training increases linear accessibility of both competing cue families at the convolutional sites of ATTRIBUTE MAZE. Yellow presence is strongly training-enhanced under the convolutional readouts, while yellow-role information is resolved above initialization at $h ^ { ( 2 ) } \ – h ^ { ( 3 ) }$ in three of the four background-by-cohort groups. Line presence and line-role gains are weaker and more site-dependent. The probes do not separate GM from Non-GM agents consistently, so cue accessibility alone does not explain the behavioral difference.

![](images/cbb7da35ef9e2b3a0cb3c8124bbf031af992913fd98a8fbac882bf7f989d0d0c.jpg)  
Figure 16: Layerwise linear availability in ATTRIBUTE MAZE. Absolute held-out ROC–AUC for line presence, line role, yellow presence, and yellow role, for the textured (top) and blackbackground (bottom) cohorts. Points show the three checkpoint seeds, lines show cohort means, and shaded regions show 95% t-intervals across the three checkpoints. The grey dashed line shows the mean of three randomly initialized networks, each evaluated on the frames of all six checkpoints, with a 95% t-interval across the three networks.

Table 4: Procgen probe results at $h _ { \mathrm { p r e } } .$ Held-out ROC–AUC (Coin bearing $R ^ { 2 } )$ , mean $\pm \thinspace \mathrm { S D }$ over three checkpoints or three untrained networks. Init: probe on an untrained network. Input control: as specified per environment.
<table><tr><td>Target</td><td>GM</td><td>Non-GM</td><td>Init</td><td>Input control</td></tr><tr><td colspan="5">MAZE</td></tr><tr><td>Cheese presence</td><td> $0 . 5 4 4 \pm 0 . 0 1$ </td><td> $0 . 9 9 7 \pm 0 . 0 0$ </td><td> $0 . 6 2 0 \pm 0 . 0 2$ </td><td>0.988 cheese-cell pixels</td></tr><tr><td>Cheese bearing</td><td> $0 . 7 1 2 \pm 0 . 0 1$ </td><td> $0 . 9 1 8 \pm 0 . 0 2$ </td><td> $0 . 6 8 6 \pm 0 . 0 1$ </td><td>0.830 position</td></tr><tr><td>Cheese bearing, corner bearing fixed</td><td> $0 . 5 9 8 \pm 0 . 0 1$ </td><td> $0 . 9 3 0 \pm 0 . 0 2$ </td><td> $0 . 7 2 9 \pm 0 . 0 4$ </td><td>0.703 position</td></tr><tr><td>Cheese bearing, position balanced</td><td> $0 . 5 8 3 \pm 0 . 0 2$ </td><td> $0 . 9 3 2 \pm 0 . 0 1$ </td><td> $0 . 7 4 8 \pm 0 . 0 5$ </td><td>0.481 position</td></tr><tr><td>Corner bearing</td><td> $0 . 9 6 3 \pm 0 . 0 1$ </td><td> $0 . 9 4 5 \pm 0 . 0 1$ </td><td> $0 . 7 2 7 \pm 0 . 0 2$ </td><td>0.959 position</td></tr><tr><td colspan="5">COIN</td></tr><tr><td>Coin presence</td><td> $0 . 7 7 5 \pm 0 . 0 8$ </td><td> $0 . 9 9 5 \pm 0 . 0 0$ </td><td> $0 . 7 2 9 \pm 0 . 0 4$ </td><td>0.760 raw pixels</td></tr><tr><td>Coin bearing  $( R ^ { 2 } )$ </td><td> $0 . 5 2 3 \pm 0 . 0 6$ </td><td> $0 . 5 2 7 \pm 0 . 0 6$ </td><td> $- 0 . 1 8 0 \pm 0 . 2 9$ </td><td>0.152 position</td></tr><tr><td>Coin bearing, position residualized  $( R ^ { 2 } )$ </td><td> $0 . 3 7 8 \pm 0 . 0 8$ </td><td> $0 . 3 0 6 \pm 0 . 0 9$ </td><td> $- 0 . 1 6 6 \pm 0 . 2 1$ </td><td>-0.024 position</td></tr><tr><td colspan="5">ATTRIBUTE  $\mathbf { M A Z E } ,$  black background</td></tr><tr><td>Line presence  $L _ { \mathrm { p r e s } }$ </td><td> $0 . 7 9 2 \pm 0 . 0 8$ </td><td> $0 . 9 5 2 \pm 0 . 0 4$ </td><td> $0 . 8 1 0 \pm 0 . 1 2$ </td><td>0.975 block pixels</td></tr><tr><td>Line role  $L _ { \mathrm { r o l e } }$ </td><td> $0 . 6 3 2 \pm 0 . 0 5$ </td><td> $0 . 6 3 6 \pm 0 . 0 4$ </td><td> $0 . 6 0 3 \pm 0 . 0 4$ </td><td>0.629 block pixels</td></tr><tr><td>Yellow presence  $Y _ { \mathrm { p r e s } }$ </td><td> $0 . 8 1 1 \pm 0 . 0 8$ </td><td> $0 . 7 5 9 \pm 0 . 1 2$ </td><td> $0 . 5 8 4 \pm 0 . 0 8$ </td><td>0.662 block pixels</td></tr><tr><td>Yellow role  $Y _ { \mathrm { r o l e } }$ </td><td> $0 . 7 0 5 \pm 0 . 0 2$ </td><td> $0 . 6 2 6 \pm 0 . 0 6$ </td><td> $0 . 5 3 1 \pm 0 . 0 5$ </td><td>0.556 block pixels</td></tr></table>

Robot probes. We use a probe bank (Appendix E.4, Table 23) that holds start frames in which goal and proxy vary independently. In PUSHTOMARKER and LOCATIONREACH we probe the table position $( x , y )$ of the goal, the marker and the ball. Their proxies are fixed places, the table centre and the training corner, so they have no variation to probe. In ATTRIBUTETABLE and SORTZONES we probe the position of the goal, the cube and the crossed pad, and of the proxy, the red object and the green pad. In WALKTOFLAG we probe the bearing of the flag and the tower relative to the robot camera. Position probes are scored by held-out $R ^ { 2 }$ averaged over x and $y ,$ and bearing probes by held-out $R ^ { 2 }$ . The binary probes, whether the crossed pad lies on the left and whether the flag or the tower is in view, are scored by held-out ROC–AUC.

![](images/a0ddad45c9aa58510df164b72ea26d2b397f92c443c14a25d9377f0bc718d808.jpg)

![](images/232abe174a3f5a2902ccbd1e7f8eaba48ea90746ba4156016a214f5e7b632e7a.jpg)

![](images/831923323fb6e5b02e6c7ab11033755e08f3dbaad781bc433ee16327cc7ba3f1.jpg)

![](images/07d5d214842125db00afbc4a89bbc93c735822f9a9ca0b8b02d52d8068dd71c6.jpg)

![](images/36af6f5d781329156d6b7e09826bd294dacbbb9a3676fbf2a4c6772b827e0225.jpg)

![](images/2dbc7ef29ab22aeeae6b2e42761c59aecd90ae19e2775c262a96bfb5dbf252ac.jpg)

![](images/65fc6757e51da37e973c8b8a8930eed4ee6f82fc630f4b38590fabf67e78a4d7.jpg)

![](images/c4ba5761de95cc4682a3a88e64d9d5e52f9c3741418d50ab07b2aa397a890118.jpg)  
GM Non-GM random init pixels mean ± 95% CI

Figure 17: Goal and proxy decodability in the five robot tasks. Held-out $R ^ { 2 }$ of a linear probe across network depth, for the goal of every task and the proxy where it varies. Lines are cohort means with 95% t-intervals, dashed grey lines randomly initialized networks and dotted lines a probe on block-averaged pixels. Arm sites $H ^ { ( 1 ) }$ and $H ^ { ( 2 ) }$ are unpooled, and WALKTOFLAG sites $\bar { H } ^ { ( k ) }$ are average-pooled to 4×4 cells.

Robot results: the proxy is a location. In PUSHTOMARKER and LOCATIONREACH the goal position is not decodable in GM agents beyond what random networks and pixels already give (Figure 17). The marker position is decodable in GM agents at $h _ { \mathrm { p r e } }$ at 0.717, not above random networks (0.805) and on par with the pixel probe (0.713), and at $h _ { \mathrm { f c } }$ it falls to 0.110 against 0.406 for random networks, while Non-GM agents keep it (0.819). The ball position is decodable in GM agents below random networks at every site, 0.564 against 0.737 at $h _ { \mathrm { p r e } } ,$ also on held-out scenes with the ball at least 10 cm from the training corner (0.511 against 0.706), while Non-GM agents stay at 0.902 or above (0.889 or above on these scenes). As in MAZE, GM training does not build on what a random network already exposes, but here a random network exposes much more, since the ball is large and salient where the cheese covers 2 to 5 pixels. Both proxies are fixed places, so in these two tasks the probes speak only to the goal.

Robot results: the proxy is an attribute or a landmark. In ATTRIBUTETABLE the cube position is decodable in GM agents through $h _ { \mathrm { p r e } } ,$ above random networks and at the level of the pixel probe (0.914 against 0.768 and 0.889), close to Non-GM agents (0.975). At $h _ { \mathrm { f c } }$ the proxy takes over: GM agents keep the red object better than the cube (0.866 against 0.695), whereas Non-GM agents keep the cube better (0.942 against 0.661), all far above random networks (0.319 and 0.402). The red object is the cube in half of the scenes, so the two positions correlate (0.38 in x, 0.33 in y). In SORTZONES the pads are large, and every network up to $h _ { \mathrm { p r e } } .$ , random ones included, and the pixel probe decode the crossed-pad position at $R ^ { 2 } \geq 0 . 9 8 9$ , so no probe there is a training effect. Only $h _ { \mathrm { f c } }$ separates the cohorts, where the crossed-pad position is decodable in GM agents below random networks (0.612 against 0.832) and in Non-GM agents above them (0.976), and the green-pad position stays at the random-network level in GM agents (0.955 against 0.961). So in SORTZONES the true goal is represented more weakly in GM than in Non-GM agents, and this can be said reliably only at $h _ { \mathrm { f c } }$ , since below it the headroom above random networks is too small.

In WALKTOFLAG the flag bearing is decodable in GM agents well above random networks at every site, and it reaches the last shared layer $h _ { \mathrm { f c } }$ . However, it stays below the pixel probe and below Non-GM agents (0.656 against 0.837 at $h _ { \mathrm { p r e } } ,$ 0.590 against 0.826 at $h _ { \mathrm { f c } } )$ . The true goal is decodable, but less than in generalized agents, which follow it. The effect flips for the proxy: the tower bearing is more decodable in GM than in Non-GM agents (0.958 against 0.710 at $h _ { \mathrm { f c } } )$ , and both cohorts separate from random networks. This is what we would expect for misgeneralized agents, as the proxy is represented more strongly in GM agents and the true goal more strongly in Non-GM agents, the mirror of their behavior inside their representation. This asymmetry appears only in environments with competing objects or object cues, which is a first pointer to how the environments separate. The deeper we go in the next analyses, the more this group separates from the others, because in it the goal information propagates far into the agent. One thing we still need to rule out: since we probe the bearings of two objects, one tracked object could light up both bearings. In the full bank the two bearings correlate $( r = 0 . 3 7$ over held-out states with both in view), so, as for the MAZE and COIN bearings above, we repeat the probes on the conflict layouts, which are new to both cohorts and in which the two bearings are nearly uncorrelated $( r = - 0 . 0 7 )$ , and find the same pattern (Table 5, conflict rows, flag 0.664 against 0.855, tower 0.940 against 0.876, GM against Non-GM at $h _ { \mathrm { p r e } } ,$ all well above random networks). So the flag bearing in GM agents does not come from the tower, and the tower bearing does not come from the flag.

Table 5: Robot probes at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } } .$ Held-out $R ^ { 2 }$ (AUC where marked). GM, Non-GM: trained agents, mean $\pm \bar { \mathrm { \bf S } } \mathrm { \bf D }$ over three seeds. Init: three randomly initialized networks, mean ± SD. Pixels: one probe on block-averaged pixels<sup>a</sup>.
<table><tr><td rowspan="2">Concept</td><td colspan="3"> $h _ { \mathrm { p r e } }$ </td><td colspan="3"> $h _ { \mathrm { f c } }$ </td><td rowspan="2">Pixels</td></tr><tr><td>GM</td><td>Non-GM</td><td>Init</td><td>GM</td><td>Non-GM</td><td>Init</td></tr><tr><td colspan="7">PUSHToMARKER (proxy: table centre, fixed)</td></tr><tr><td>Goal: marker position</td><td> $0 . 7 1 7 \pm 0 . 0 9$ </td><td> $0 . 8 9 6 \pm 0 . 0 0$ </td><td> $0 . 8 0 5 \pm 0 . 0 1$ </td><td> $0 . 1 1 0 \pm 0 . 0 8$ </td><td> $0 . 8 1 9 \pm 0 . 0 3$ </td><td> $0 . 4 0 6 \pm 0 . 0 3$ </td><td>0.713</td></tr><tr><td colspan="7">LOCATIONREACH (proxy: training corner, fixed)</td></tr><tr><td>Goal: ball position</td><td> $0 . { \overset { - } { . } } 5 6 4 \pm { \overset { - } { 0 } } . 0 3$ </td><td> $0 . 9 2 4 \pm 0 . 0 0$ </td><td> $0 . 7 3 7 \pm 0 . 0 2$ </td><td>0.063 ± 0.09</td><td> $0 . 9 0 2 \pm 0 . 0 0$ </td><td> $0 . 1 3 6 \pm 0 . 0 5$ </td><td>0.739</td></tr><tr><td colspan="7">ATTRIBUTETABLE</td></tr><tr><td>Goal: cube position</td><td> $0 . 9 1 4 \pm 0 . 0 2$ </td><td> $0 . 9 7 5 \pm 0 . 0 1$ </td><td> $0 . 7 6 8 \pm 0 . 0 2$ </td><td> $0 . 6 9 5 \pm 0 . 0 1$ </td><td> $0 . 9 4 2 \pm 0 . 0 2$ </td><td> $0 . 4 0 2 \pm 0 . 0 5$ </td><td>0.889</td></tr><tr><td>Proxy: red-object positionb</td><td> $0 . 9 1 3 \pm 0 . 0 3$ </td><td> $0 . 8 6 6 \pm 0 . 0 8$ </td><td></td><td>0.836 ± 0.03 0.866 ± 0.01 0.661 ± 0.08</td><td></td><td> $0 . 3 1 9 \pm 0 . 0 7$ </td><td>0.912</td></tr><tr><td colspan="7"></td></tr><tr><td>SORTZONES</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.997</td></tr><tr><td>Goal: crossed-pad position Goal: crossed pad left (AUC)</td><td> $0 . 9 9 3 \pm 0 . 0 0$  1.000 ± 0.00</td><td> $0 . 9 9 8 \pm 0 . 0 0$   $1 . 0 0 0 \pm 0 . 0 0$ </td><td></td><td>0.989 ± 0.00 0.612 ± 0.04 0.976 ± 0.00 1.000 ± 0.00 0.948 ± 0.01 1.000 ± 0.00</td><td></td><td> $0 . 8 3 2 \pm 0 . 0 3$   $1 . 0 0 0 \pm 0 . 0 0$ </td><td>1.000</td></tr><tr><td>Proxy: green-pad position</td><td> $0 . 9 9 9 \pm 0 . 0 0$ </td><td> $0 . 9 9 8 \pm 0 . 0 0$ </td><td></td><td>0.998 ± 0.00 0.955 ± 0.01 0.891 ± 0.03</td><td></td><td> $0 . 9 6 1 \pm 0 . 0 0$ </td><td>0.999</td></tr><tr><td colspan="8">WALKTOFLAG</td></tr><tr><td>Goal: flag bearing</td><td> $0 . 6 5 6 \pm 0 . 0 4$ </td><td> $0 . 8 3 7 \pm 0 . 0 3$ </td><td></td><td>0.252 ± 0.02 0.590 ± 0.03 0.826 ± 0.02</td><td></td><td> $0 . 1 6 3 \pm 0 . 0 5$ </td><td>0.734</td></tr><tr><td>Goal: flag visible (AUC)</td><td> $0 . 9 2 9 \pm 0 . 0 2$ </td><td> $0 . 9 6 8 \pm 0 . 0 1$ </td><td></td><td></td><td>0.699 ± 0.02 0.865 ± 0.02 0.948 ± 0.01</td><td> $0 . 6 4 1 \pm 0 . 0 1$ </td><td>0.965</td></tr><tr><td>Proxy: tower bearing</td><td> $0 . 9 6 0 \pm 0 . 0 1$ </td><td> $0 . 9 1 0 \pm 0 . 0 2$ </td><td></td><td></td><td>0.808 ± 0.02 0.958 ± 0.01 0.710 ± 0.04</td><td>0.525 ± 0.06</td><td>0.915</td></tr><tr><td>Proxy: tower visible (AUC)</td><td> $0 . 9 9 5 \pm 0 . 0 0$ </td><td> $0 . 9 9 3 \pm 0 . 0 0$ </td><td></td><td></td><td></td><td>0.994 ± 0.00 0.995 ± 0.00 0.974 ± 0.00 0.909 ± 0.02 1.000</td><td></td></tr><tr><td>Goal: flag bearing, conflictc</td><td> $0 . 6 6 4 \pm 0 . 0 4$ </td><td> $0 . 8 5 5 \pm 0 . 0 3$ </td><td></td><td></td><td></td><td>0.223 ± 0.01 0.654 ± 0.05 0.865 ± 0.03 0.139 ± 0.07 0.320</td><td></td></tr><tr><td>Proxy: tower bearing, conflictc</td><td> $0 . 9 4 0 \pm 0 . 0 2$ </td><td> $0 . 8 7 6 \pm 0 . 0 2$ </td><td></td><td>0.619 ± 0.080.952 ± 0.01</td><td> $0 . 6 5 4 \pm 0 . 0 4$ </td><td></td><td>0.377 ± 0.08 0.871</td></tr></table>

<sup>a</sup>32×32×3 block means of the 128×128 arm frame, 16×16×3 of the 96×96 quadruped frame. <sup>b</sup>Scenes with exactly one red object. The red object is the cube in half of them, so its position correlates with the cube position (0.38 in x, 0.33 in y). <sup>c</sup>Probe trained and tested only on the conflict layouts, which are new to both cohorts and in which the flag and tower bearings are nearly uncorrelated $( r = - 0 . 0 7 $ over held-out states with both in view), so one tracked object cannot light up both bearings.

Taken together, wherever the proxy is an attribute of the goal or a closely related landmark, the probes point the same way: at $h _ { \mathrm { f c } }$ the goal is weaker and the proxy stronger in GM than in Non-GM agents (Table 5). ATTRIBUTE MAZE shows this pattern in part.

What a probe can and cannot show. Linear decodability of a goal feature does not necessarily indicate a goal representation. Probes retrieve linearly accessible information, but whether that information is usable as a goal inside the network does not necessarily follow from its decodability. Information can be decodable but still not be used by the policy as a goal, or not be recoverable as a representation that navigates the agent toward a goal. In SORTZONES, for example, a randomly initialized network decodes the crossed-pad position at $h _ { \mathrm { p r e } }$ with $R ^ { 2 } = 0 . 9 8 9$ , yet a new readout trained by behavior cloning on these random features recovers nothing: it brings the cube to the crossed pad less often than a coin flip would (0.151 against 0.377, Appendix D.2). A second limit is headroom. Many goal and proxy features are already decodable from random networks or from the pixels, and a probe speaks to what training built only where it separates from these baselines. In several environments at least one cohort separates clearly from random networks, which is enough to make a statement. SORTZONES is the hardest to read, because its headroom is tiny. Probes are therefore indicative of information that may be used downstream, but a representation that acts as a goal requires more, which is why the following analyses turn from decodability to how the policy reacts to and uses goal information.

Conclusion. We asked whether information about the intended goal is lost, and find that losing it is not a core part of goal misgeneralization. In many environments the true goal is still linearly represented, even in late layers, so a uniform deletion is not the mechanism. What differs is the degree of representation, and three groups emerge, with the caveat that our probes can confirm that information is linearly available but cannot rule out that it is present in another form. In ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG the goal information is learned up to late layers and comes with an asymmetry: GM agents decode the proxy better, Non-GM agents the goal (in ATTRIBUTE MAZE only in part, and in SORTZONES readable only at $h _ { \mathrm { f c } } )$ . COIN is an intermediate case: the coin bearing is decodable in both cohorts through $h _ { \mathrm { p r e } } ,$ , also beyond the agent’s position, but GM agents build a much weaker coin-presence representation than Non-GM agents. In MAZE, LOCATIONREACH and PUSHTOMARKER the goal is not linearly decodable above random networks in GM agents. PUSHTOMARKER belongs here rather than with COIN, since its marker position stays below random networks at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ , and in MAZE the cheese bearing reduces to tracking the agent’s position once that position is decorrelated from it. We therefore next ask whether goal-related changes are coupled to the native policy and what functional role the resulting representations play in action selection.

## C.2 BEHAVIORAL DEPENDENCE: DOES THE POLICY REACT TO GOAL CHANGES, VIEWEDFROM OUTSIDE?

Motivation. Linear decodability establishes that goal-related information can remain available internally, but does not show whether the policy responds to that information. We therefore intervene directly on goal- and proxy-related variables in the environment and ask whether the resulting world change alters the policy, whether that change follows the edited feature, and whether it changes eventual goal achievement.

Edit design. We ask whether the trained policy actually responds when a goal-related variable is changed in the world. A world edit e changes a goal- or proxy-related variable in the level while keeping the rest of the state as fixed as the environment permits, and turns a base state s into a matched edited state $s _ { e } ,$ , with $e \ = \ \star$ for an edit of the goal and $e \ : = \ : p$ for an edit of the proxy (Definition 2). Because nothing else changes, a change in the policy’s output can be attributed to the edited variable.

The environments allow two kinds of edit. Relocation edits are used in MAZE, COIN, LOCA-TIONREACH, PUSHTOMARKER and WALKTOFLAG, where the goal object can be moved without changing its identity or the level (Figure 18). A relocation edit moves the edited object from its place g to a new place $g ^ { \prime }$ and holds the rest of the level fixed. In MAZE the cheese is moved while the layout, the agent state and the top-right destination stay unchanged. The cheese is moved to the reachable bottom-left extreme, which gives every state the same kind of edit. The structural MAZE edit instead opens access to the top-right region while otherwise preserving the maze state, and so edits the proxy. In COIN, g is a cell to the right of the agent, on the shortcut side, and $g ^ { \prime }$ the cell one step to its left. The environment holds a single coin, so the edit moves it and does not add a second one. The end-of-level marker is left in place. In PUSHTOMARKER and LOCATIONREACH the marker and the ball move to a second position away from the training place, with the arm and the other objects fixed. In WALKTOFLAG goal and proxy are separate objects, so the goal edit moves the flag and the proxy edit the tower, each to a place beside the other object.

Swap edits are used in ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES. In ATTRIBUTE MAZE the goal and proxy are attributes of objects, and moving an object would change the navigation geometry of the conflict together with the cue. So we hold the two candidate locations fixed and swap their attributes (Figure 18). Let the candidates at locations u and v be

$$
o _ { u } = ( \mathsf { I i n e } , \mathsf { r e d } ) , \qquad o _ { v } = ( \mathsf { g e m } , \mathsf { y e l l o w } ) ,
$$

so the line cue sits at u and the yellow cue at v. The shape swap exchanges the shapes and keeps colours and locations fixed,

$$
\mathcal { E } ^ { \mathrm { s h a p e } } : \qquad ( \mathsf { l i n e } , \mathsf { r e d } ) _ { u } , ( \mathsf { g e m } , \mathsf { y e l l o w } ) _ { v } \longmapsto ( \mathsf { g e m } , \mathsf { r e d } ) _ { u } , ( \mathsf { l i n e } , \mathsf { y e l l o w } ) _ { v } .\tag{4}
$$

goal edit (⋆) cheese moved

goal edit (⋆) coin moved left

proxy edit (p) top-right opened

![](images/8276e5ba7b5801b1054cef0e47a8ff44627ac95b869788315b7b00c074ccb83c.jpg)  
original

Maze  
![](images/f1df856b28a0347e559b0c8e9f5a80f4c00d7e6df6ad85c2c286ad1e8df0a699.jpg)

![](images/f008b46bf23eaa1e3d3a50330a8c785cc7e6ee01c59c64a4284f4b028b6fa57b.jpg)

![](images/9f49fe76c903c4034c25d69409ec5065311a51b2dfea1963aee0e54f6ad53d84.jpg)  
Coin

![](images/d088d38da9a3ddae430fb6e142bed6873fa7126c800ef327d54c9628cb0583b5.jpg)

Attribute Maze  
![](images/c2bdcd3da4c4ff194d6f4e7ac93c3fe4c0d4700ad28f63926a6e5b6e41d3a7e3.jpg)

AttributeTable  
![](images/11f6ebc6354972dddc867244474b03042329135caeeab686d29794170faa8bf4.jpg)  
Figure 18: Goal and proxy edits in each environment. One example state per environment, shown as the original, the goal edit (e = ⋆) and, where the environment has one, the proxy edit $( e = p )$ The MAZE cheese moves to the reachable bottom-left extreme, and both MAZE edits are shown on the same state. Solid boxes enclose every pixel the goal edit changes and dashed boxes every pixel the proxy edit changes, including shadows in WALKTOFLAG.

The colour swap exchanges the colours and keeps shapes and locations fixed,

$$
{ \mathcal { E } } ^ { \mathrm { c o l o u r } } : \qquad ( | \mathsf { i n e , r e d } ) _ { u } , ( \mathsf { g e m } , \mathsf { y e l l o w } ) _ { v } \longmapsto ( | \mathsf { i n e , y e l l o w } ) _ { u } , ( \mathsf { g e m , r e d } ) _ { v } .\tag{5}
$$

These two swaps are the canonical counterfactuals of ATTRIBUTE MAZE. They resolve the same line and yellow conflict in opposite directions, since the training-consistent yellow line appears at v under the shape swap and at u under the colour swap. This return to a familiar conjunction is intended. The base state is the out-of-distribution conflict, and each swap asks how the policy changes when the conflict is resolved toward one of the two fixed candidates, without introducing a further unfamiliar object combination. Agent position, walls, object count and candidate locations are un changed, so the navigation geometry is identical across the base state and both swaps. The two swaps are not equally likely training perturbations, since the shape and colour vocabularies are not symmetric, and we use them only as paired resolutions of one spatial conflict. ATTRIBUTETABLE and SORTZONES use the same two swaps (Figure 18). In ATTRIBUTETABLE the base conflict is (cube, $\mathsf { b l u e } ) _ { u } , ( \mathsf { s p h e r e } , \mathsf { r e d } ) _ { v } ,$ the shape swap gives (sphere, blue) , (cube, red) and the colour swap (cube, red)<sub>u</sub>, (sphere, blue)<sub>v</sub>. In SORTZONES the cross takes the role of the shape. The base conflict is $( \mathsf { p u r p l e } , \mathsf { c r o s s } ) _ { u } , ( \mathsf { g r e e n } , \mathsf { p l a i n } ) _ { v }$ , the cross swap gives (purple, plain) , (green, cross) and the colour swap (green, cross)<sub>u</sub>, (purple, plain)<sub>v</sub>. As in ATTRIBUTE $\mathbf { M A Z E }$ , each swap returns the scene to the training conjunction, a red cube or a crossed green pad, at one of the two fixed places. All edits are applied to edit banks that we generate for this purpose, and their specification, including their size, is given in Appendix E.

Metric design. We evaluate behavioral dependence at increasingly longer time horizons. First, we measure whether the edit changes the immediate policy at all, through the KL divergence between the action distributions in the original and in the edited state,

$$
D _ { \mathrm { K L } , e } ( s ) = D _ { \mathrm { K L } } \left( \pi ( \cdot \mid s ) \parallel \pi ( \cdot \mid s _ { e } ) \right) .
$$

This quantity is direction-agnostic. A large value establishes that the edit redistributes the probability over actions, but not whether the mass moves onto actions that pursue the goal or the proxy. The robot policies are Gaussian with a state-independent standard deviation $\sigma ,$ so their KL has the closed form $\begin{array} { r } { \frac { 1 } { 2 } \sum _ { i } \big ( \mu _ { i } ( s _ { e } ) - \mu _ { i } ( s ) \big ) ^ { 2 } / \sigma _ { i } ^ { 2 } } \end{array}$ , the change of the action mean measured in units of the policy’s own $\sigma .$ This scale depends on σ and on the action space, so a robot KL is compared only between the cohorts of one environment.

Second, we measure whether the change is consistent with goal-directed action, through a quantity we call signed follow. Let $A _ { \star } ( s )$ be the progress actions toward the intended goal in state s, the actions that shorten the path to it, and $A _ { p } ( s )$ the progress actions toward the proxy (Definition 10, Section B.4). For a goal edit we use $A _ { e } = A _ { \cdot }$ <sub>⋆</sub> and for a proxy edit $A _ { e } = A _ { p }$ , both read in the edited state, so that they lead to the edited object’s new place. Signed follow is the probability that the edit moves onto these actions,

$$
\operatorname { S F } _ { e } ( s ) = \pi ( A _ { e } ( s _ { e } ) \mid s _ { e } ) - \pi ( A _ { e } ( s _ { e } ) \mid s ) , \qquad \pi ( A \mid s ) = \sum _ { a \in A } \pi ( a \mid s ) .
$$

A positive value means that the edit moves probability onto the actions toward the edited object’s new place. In MAZE the sets are graph-exact, toward the relocated cheese for the cheese edit and toward the opened corner for the structural edit. In COIN $A _ { \star }$ holds the leftward actions, and in ATTRIBUTE MAZE the sets lead to the candidate that carries the line after the shape swap and the yellow after the colour swap.

For continuous actions the same set is a half-space, the commanded steps with a positive component toward the object. A robot policy, however, already puts almost all of its probability in that halfspace before any edit, above 0.999 in ATTRIBUTETABLE, because its σ is small compared with its step. The probability therefore cannot rise, and we measure instead how the edit turns the planned step (Definition 11). With $\hat { t } _ { e }$ the target direction of the edited object, the goal for $e = \star$ and the proxy for $e = p { \mathrm { . } }$ , read before the edit in s and after it in $s _ { e }$

$$
\mathrm { S F } _ { e } ( s ) = \left( \mathrm { s t e p } ( s _ { e } ) - \mathrm { s t e p } ( s ) \right) ^ { \top } \frac { \hat { t } _ { e } ( s _ { e } ) - \hat { t } _ { e } ( s ) } { \left\| \hat { t } _ { e } ( s _ { e } ) - \hat { t } _ { e } ( s ) \right\| _ { 2 } } .
$$

The fraction is the unit vector along which the edit turns the target direction, so a positive value means that the planned step turns from the object’s old place toward its new place, the continuous counterpart of moving probability onto actions toward the new place. It is in millimetres for the arm and in command units for the quadruped.

KL and SF are computed on the same states. In Procgen these are strict-conflict states, in which no action makes progress toward both goal and proxy, $A _ { \star } ( s ) \cap A _ { p } ( s ) = \emptyset$ (Section B.4). With few discrete actions, the progress sets toward two targets can share an action, and a shift of mass onto that action would count for both, so we keep only states in which the sets are disjoint. The robot banks take scenes laid out like each task’s conflict test, with goal and proxy at different places, in WALKTOFLAG closer together so that both stay in view (Appendix E). A continuous action space has no such overlap to remove. Signed follow projects the change of the planned step onto a direction, so even a small separation between two targets gives two different directions instead of falling into one shared discrete action, and the edited object’s old and new place always differ.

For the behavioral world-edit quantities, we first average over evaluation states within each seed and then report the mean and sample standard deviation across seeds. Where the MAZE and COIN results report reach, it is the fraction of rollouts started from the edited state that reach the relocated goal (Appendix F).

Table 6: Behavioral dependence under matched world edits. KL measures immediate policy reaction and SF measures whether that reaction follows the edited feature. ⋆ marks an edit of the goal and p an edit of the proxy. Values are mean ± sample standard deviation across three seeds, each the mean over the evaluation states. Robot SF is in mm for the arm tasks and in command units for WALKTOFLAG, and robot rows are compared only within their environment. Bold marks the larger KL of each GM and Non-GM pair.
<table><tr><td>Environment</td><td>Edited feature</td><td>Cohort</td><td>KL</td><td>SF</td></tr><tr><td colspan="5">Relocation edits</td></tr><tr><td rowspan="3">MAZE</td><td>Cheese (*)</td><td>GM Non-GM</td><td> $0 . 0 1 5 \pm 0 . 0 1$ </td><td> $- 0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td rowspan="2">Corner (p)</td><td></td><td> $\mathbf { 1 . 6 8 7 \mathop { \pm } 0 . 0 6 }$ </td><td> $+ 0 . 3 5 7 \pm 0 . 0 0$ </td></tr><tr><td>GM Non-GM</td><td> $\mathbf { 2 . 3 7 6 \pm 0 . 4 4 }$   $0 . 0 6 7 \pm 0 . 0 1$ </td><td> $+ 0 . 3 7 7 \pm 0 . 0 2$   $+ 0 . 0 1 7 \pm 0 . 0 0$ </td></tr><tr><td rowspan="2">LOCATIONREACH</td><td rowspan="2">Ball (*)</td><td>GM</td><td> $2 . 6 7 6 \pm 2 . 0 3$ </td><td> $- 1 . 7 6 4 \pm 1 . 1 9$ </td></tr><tr><td>Non-GM</td><td> $\mathbf { 1 0 3 . 6 6 1 \pm 3 0 . 1 5 }$ </td><td> $+ 1 9 9 . 6 6 8 \pm 3 5 . 7 5$ </td></tr><tr><td rowspan="2">PUSHTOMARKER</td><td rowspan="2">Marker (*)</td><td>GM</td><td> $1 6 . 2 9 1 \pm 0 . 8 6$ </td><td> $+ 0 . 4 8 7 \pm 1 . 4 0$ </td></tr><tr><td>Non-GM</td><td> $\mathbf { 7 9 . 1 3 3 \pm 2 7 . 4 1 }$ </td><td> $+ 2 7 . 9 4 8 \pm 1 3 . 9 8$ </td></tr><tr><td rowspan="2">COIN</td><td rowspan="2">Coin (*)</td><td>GM</td><td> $0 . 0 4 4 \pm 0 . 0 1$ </td><td> $+ 0 . 0 0 0 \pm 0 . 0 1$ </td></tr><tr><td>Non-GM</td><td> ${ \bf 0 . 3 8 9 \pm 0 . 1 1 }$ </td><td> $+ 0 . 1 7 9 \pm 0 . 0 5$ </td></tr><tr><td rowspan="4">WALKTOFLAG</td><td rowspan="2">Flag (*)</td><td>GM</td><td> $1 . 0 5 8 \pm 0 . 3 1$ </td><td> $+ 0 . 0 7 8 \pm 0 . 0 3$ </td></tr><tr><td>Non-GM</td><td> ${ \bf 4 . 2 6 2 \pm 2 . 9 0 }$ </td><td> $+ 0 . 1 5 1 \pm 0 . 0 7$ </td></tr><tr><td rowspan="2">Tower (p)</td><td>GM</td><td> ${ \bf 1 0 . 7 5 5 \pm 9 . 3 4 }$ </td><td> $+ 0 . 2 8 2 \pm 0 . 1 4$ </td></tr><tr><td>Non-GM</td><td> $2 . 2 8 0 \pm 0 . 6 1$ </td><td> $- 0 . 0 6 6 \pm 0 . 0 4$ </td></tr><tr><td colspan="4">Attribute swaps</td><td></td></tr><tr><td rowspan="8">ATTRIBUTE MAZE</td><td rowspan="4">Shape (*)</td><td>Black GM</td><td> $0 . 5 6 3 \pm 0 . 2 5$ </td><td> $+ 0 . 3 3 5 \pm 0 . 0 7$ </td></tr><tr><td>Black Non-GM</td><td> $\mathbf { 2 . 8 1 4 \pm 0 . 2 4 }$ </td><td> $+ 0 . 6 2 9 \pm 0 . 0 1$ </td></tr><tr><td>Textured GM</td><td> $0 . 2 9 1 \pm 0 . 3 2$ </td><td> $+ 0 . 2 1 8 \pm 0 . 1 3$ </td></tr><tr><td>Textured Non-GM</td><td> $\mathbf { 2 . 6 8 9 \pm 0 . 1 6 }$ </td><td> $+ 0 . 6 0 0 \pm 0 . 0 4$ </td></tr><tr><td rowspan="4">Colour (p)</td><td>Black GM</td><td> $\mathbf { 1 . 2 5 8 \pm 0 . 3 1 }$ </td><td> $+ 0 . 4 6 3 \pm 0 . 0 7$ </td></tr><tr><td>Black Non-GM</td><td> $0 . 0 2 5 \pm 0 . 0 2$ </td><td> $+ 0 . 0 4 7 \pm 0 . 0 3$ </td></tr><tr><td>Textured GM</td><td> ${ \bf 1 . 9 8 6 \pm 0 . 6 5 }$ </td><td> $+ 0 . 5 0 8 \pm 0 . 1 1$ </td></tr><tr><td>Textured Non-GM</td><td> $0 . 0 8 6 \pm 0 . 0 8$ </td><td> $+ 0 . 0 9 0 \pm 0 . 0 5$ </td></tr><tr><td rowspan="4">ATTRIBUTETABLE</td><td rowspan="2">Shape (*)</td><td>GM</td><td> $7 . 0 9 9 \pm 2 . 0 7$ </td><td> $+ 1 7 . 2 6 7 \pm 4 . 8 6$ </td></tr><tr><td>Non-GM</td><td> $\mathbf { 1 9 . 6 2 0 \pm 4 . 4 0 }$ </td><td> $+ 5 6 . 1 1 6 \pm 4 . 8 8$ </td></tr><tr><td rowspan="2">Colour (p)</td><td>GM</td><td> $\mathbf { 3 7 . 0 2 6 \pm 1 . 3 3 }$ </td><td> $+ 6 3 . 1 8 7 \pm 9 . 3 0$ </td></tr><tr><td>Non-GM</td><td> $1 6 . 0 1 3 \pm 3 . 9 6$ </td><td> $+ 4 . 3 4 5 \pm 1 1 . 2 6$ </td></tr><tr><td rowspan="3">SORTZONES</td><td rowspan="2">Cross (*)</td><td>GM</td><td> $4 . 9 0 3 \pm 2 . 3 4$ </td><td> $+ 5 . 3 9 5 \pm 2 . 7 8$ </td></tr><tr><td>Non-GM</td><td> $\mathbf { 4 5 . 6 1 8 \pm 2 4 . 1 9 }$ </td><td> $+ 2 3 . 0 1 4 \pm 1 0 . 9 0$ </td></tr><tr><td rowspan="2">Colour (p)</td><td>GM Non-GM</td><td> $\mathbf { 4 0 . 9 2 2 \pm 2 0 . 2 4 }$   $1 7 . 4 3 8 \pm 1 4 . 1 3$ </td><td> $+ 2 1 . 1 1 4 \pm 4 . 5 3$   $+ 4 . 8 2 4 \pm 2 . 9 1$ </td></tr></table>

Results. The world edits reveal qualitatively different patterns of behavioral dependence across the eight environments (Table 6). Because KL and signed follow have different scales in different environments, we read every edit relative to the Non-GM agents of the same environment, as the ratio of the GM to the Non-GM value (Figure 19). The two metrics then separate the eight environments in two steps.

First, KL separates MAZE and LOCATIONREACH from the rest. A goal edit barely moves the GM policy there, at a ratio of 0.009 in MAZE and 0.026 in LOCATIONREACH, while the same edit moves the Non-GM policy strongly. The GM agents therefore behave as if they were inert to the goal. In the other six environments the ratio of a goal edit lies between 0.107 and 0.362. In the three swap environments and WALKTOFLAG the table shows, moreover, that both edits move the GM policy, the goal edit and the proxy edit alike.

![](images/c71d330c235712a78bfe83852b728360547f1e907fb49d7b89ff238dd8e7db0f.jpg)

Figure 19: World-edit responses of GM agents relative to Non-GM agents. Each row divides the GM response to one edit by the Non-GM response in the same environment, filled for goal edits and open for proxy edits. Large markers divide the cohort means and small dots each GM seed by the Non-GM mean. Dashed lines mark a response as strong as in Non-GM agents, and the dotted line at KL 0.05 separates the inert location targets from all other edits. n/a: not every Non-GM seed turns toward the edited object.

Second, signed follow separates the remaining environments. For a goal edit its ratio is about zero in PUSHTOMARKER (0.017) and COIN (0.001) as well. The goal edit may move probability mass in these GM policies, but not onto actions that lead to the goal. A navigating policy may react in the short term to the objects around it, and the GM policies react to the edit in this way without turning their behavior toward the goal. In WALKTOFLAG and the three swap environments the ratio lies between 0.234 and 0.533, so the goal still turns even the GM policy toward itself, though less than it turns the Non-GM policy. A proxy edit, in contrast, turns the GM policy several times more than the Non-GM policy, which barely follows the proxy, so the ratio is large but its exact value is not informative. This is the asymmetry we expect between GM and Non-GM policies, and these environments show it in KL and in signed follow alike. Two signed-follow ratios are undefined because Non-GM agents do not turn toward the edited object, the tower in WALKTOFLAG on average and the red object in ATTRIBUTETABLE in one seed.

Together, the two metrics sort the environments by how far the goal gets. In MAZE and LOCATION-REACH the goal edit does not reach the policy. In PUSHTOMARKER and COIN it reaches the policy but does not direct its decision. In WALKTOFLAG and the swap environments it enters the decision and loses to the proxy. Appendix C.3 asks how this plays out inside the policy.

## C.3 FISHER GEOMETRY: IS THE POLICY USING THE GOAL, VIEWED FROM INSIDE?

Motivation. The behavioral world edits in Appendix C.2 measure whether changing a goal-related property of the environment changes the policy. We next ask the corresponding question inside the policy: how strongly is the representation change induced by the same world edit coupled to the policy’s action distribution? We ask it in all eight environments, with the same edits and evaluation states as the behavioral analysis.

Design. Rather than selecting an internal direction independently of the environment, we let the controlled world edit determine the representation displacement that we evaluate. For a base state s and its edited copy $s _ { e }$ (Section C.2), the edit moves the representation at site m by $\Delta _ { e } ^ { ( m ) } ( s )$ = $h ^ { ( m ) } ( s _ { e } ) - h ^ { ( m ) } ( s )$ (Definition 2), read at the two post-ReLU sites $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ of Section B.1. We measure the world-edit Fisher sensitivity

$$
Q _ { e } ^ { ( m ) } ( s ) = \Delta _ { e } ^ { ( m ) } ( s ) ^ { \top } F ^ { ( m ) } ( s ) \Delta _ { e } ^ { ( m ) } ( s ) ,
$$

with $F ^ { ( m ) } ( s )$ the activation-space Fisher at the unedited state (Definitions 4 and 5). Up to second order ${ \frac { 1 } { 2 } } Q$ is the KL between the base and the edited policy (Appendix B.2), so Q is the inside counterpart of the behavioral KL, and like the KL it is unsigned. A small Q does not imply that information about the edited property is absent. Instead, it means that the particular representation displacement produced by the edit has little local effect on the native action distribution.

For the Procgen policies $F$ at $h _ { \mathrm { f c } }$ has the categorical closed form after Proposition 2. The robot policies are Gaussian with a state-independent σ, so $F = J _ { \mu } ^ { \top }$ dia $\ g ( \sigma ^ { - 2 } ) J _ { \mu } ^ { - }$ exactly, with $J _ { \mu }$ the Jacobian of the action mean (Proposition 3), and $Q$ is the squared change of the mean in units of σ. We confirmed that σ is a learned vector not computed from the observation in both robot agents, and on the 19 arm checkpoints available locally a Monte Carlo estimate of the score covariance from 4000 sampled actions matches this Fisher to within 2.0% in trace and 6.2% in Frobenius norm on four world-edit bank states each, the size expected from sampling noise (1% and 4%). $Q$ is the same at both sites only to first order (Proposition 4). Over the finite edits we study it changes from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ by up to 59% in Non-GM agents (−31% in LOCATIONREACH, −25% in PUSHTOMARKER, +45% for the ATTRIBUTE MAZE and ATTRIBUTETABLE shape swaps and +59% for the WALKTOFLAG tower swap) and by up to 30% in GM agents (SORTZONES cross swap), so the GM to Non-GM ratio of $Q$ can differ between the sites, as for the ATTRIBUTE MAZE shape swap (0.551 at $h _ { \mathrm { p r e } }$ against 0.307 at $h _ { \mathrm { f c } }$ , Table 8). The ratio lies on the same side of 1 at both sites for every edit, so we read the results at $h _ { \mathrm { f c } }$ , the site of Table 1, and report $h _ { \mathrm { p r e } }$ in Table 8.

We split $Q = G D _ { F }$ into the displacement in Fisher units $D _ { F }$ and the relative directional gain G (Definition 6). $D _ { F }$ is the product of $\| \Delta \| _ { 2 } ^ { 2 }$ and the mean Fisher sensitivity $\operatorname { t r } ( F ) / d ,$ and the second factor depends on how sharp the policy is (Corollary 1). A small GM $D _ { F }$ could therefore reflect a sharper or flatter policy rather than a smaller movement of the representation, and Table 9 separates the two factors.

Q says how strongly the policy reacts, not in which direction. In Procgen we therefore also measure the signed target-selection effect, the first-order change of the contrast between the goal and the proxy action sets along $\Delta _ { e }$

$$
\partial _ { \Delta _ { \epsilon } } C _ { s } = \nabla _ { h } C _ { s } \big ( h ( s ) \big ) ^ { \top } \Delta _ { e } ( s ) , \qquad C _ { s } ( h ) = \pi \big ( A _ { \star } ( s ) \bigm | h \big ) - \pi \big ( A _ { p } ( s ) \bigm | h \big ) ,
$$

with both action sets read in the base state (Definition 8). Its sign says whether the displacement moves preference toward the goal or toward the proxy, so it is the inside counterpart of signed follow, $\operatorname { \hat { S } F } _ { e } ( s ) = \pi { \big ( } A _ { e } ( s _ { e } ) { \big ) } { \big | } { \big | } s _ { e } { \big ) } - \pi { \big ( } A _ { e } ( s _ { e } ) { \big | } { \big ) }$ , the probability that the edit moves onto the actions toward the edited object’s new place (Appendix C.2). It is reported for MAZE, COIN and ATTRIBUTE MAZE in Table 7 and per state in Figure 20. With continuous actions the contrast saturates at the robot start states, so for the robots the direction of the reaction is read from signed follow alone.

Evaluation states. Every environment uses the states and edits of the world-edit table, the conflict banks in Procgen and the world-edit banks in the robot tasks (Appendix E). On the stricter ATTRIBUTETABLE bank of the cue-competition geometry (Appendix C.5) the world-edit KL ratios keep their order (0.297 for the shape swap and 2.741 for the colour swap).

Finite-edit calibration. The local approximation is accurate enough for the edits we study, because where it deviates from the exact KL it cannot change which cohort reacts more. We compare ${ \scriptstyle { \frac { 1 } { 2 } } } t ^ { 2 } Q$ with the exact KL along fractions $t \in [ 0 , 1 ]$ of each edit. For the Gaussian robot policies the KL is itself quadratic in the mean shift (Proposition 3), so only the linearized action mean is approximate. In Procgen the approximation tracks the exact KL closely except for the largest Non-GM edits, the MAZE goal relocation and the ATTRIBUTE MAZE shape swap, where it underestimates it by a factor of 1.4 to 2.1, which makes the GM to Non-GM ratio of Q conservative. In the robot tasks $\mathrm { K L } / ( Q / 2 )$ at t = 1 is 0.932 to 1.114 for the relocations and 0.569 to 0.935 for the swaps, where the mean head bends along the edit (means over seeds of per-seed medians). It differs between the two cohorts of an edit by at most 0.166 (SORTZONES cross swap), which moves a GM to Non-GM ratio of Q by at most a factor of 1.25, less than the distance of any reported ratio from 1.

Table 7: Signed target-selection effects. Values are mean ± SD across three checkpoints per cohort, each summarized over the evaluation states of Table 8. First-order change of the target contrast along the world-induced displacement (Definition 8). Sign key: a policy that follows the edit reads negative for the MAZE cheese relocation and the ATTRIBUTE MAZE shape swap, and positive for the ATTRIBUTE MAZE colour swap and the COIN relocation.
<table><tr><td>Environment</td><td>Site</td><td>Direction</td><td>GM</td><td>Non-GM</td><td>init</td></tr><tr><td rowspan="2">MAZE</td><td> $h _ { \mathrm { p r e } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 0 0 1 \pm 0 . 0 0$ </td><td> $- 0 . 2 7 1 \pm 0 . 0 7$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 0 0 1 \pm 0 . 0 0$ </td><td> $- 0 . 2 6 9 \pm 0 . 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td rowspan="2">COIN</td><td> $h _ { \mathrm { p r e } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 0 0 9 \pm 0 . 0 2$ </td><td> $+ 0 . 0 7 2 \pm 0 . 0 3$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 0 1 0 \pm 0 . 0 2$ </td><td> $+ 0 . 1 3 9 \pm 0 . 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td rowspan="4">ATTRIBUTE MAZE</td><td> $h _ { \mathrm { p r e } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 7 1 8 \pm 0 . 2 4$ </td><td> $- 0 . 7 0 4 \pm 0 . 0 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td> $\Delta _ { p }$ </td><td> $+ 1 . 0 3 3 \pm 0 . 2 8$ </td><td> $+ 0 . 0 6 8 \pm 0 . 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $\Delta _ { \star }$ </td><td> $- 0 . 6 6 5 \pm 0 . 2 1$ </td><td> $- 0 . 8 9 7 \pm 0 . 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $\Delta _ { p }$ </td><td> $+ 0 . 9 1 8 \pm 0 . 3 1$ </td><td> $+ 0 . 0 6 3 \pm 0 . 0 4$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr></table>

![](images/7d0d91b77f6c35ac7895d02f2160bea91bed8ce1adb307a0c283d1b3f279763c.jpg)  
Figure 20: Signed effects of world-induced representation changes on target selection. Small points show evaluation states, densities summarize their distribution, and diamonds show the three seed-level summaries. The top row reports $h _ { \mathrm { p r e } }$ and the bottom row $h _ { \mathrm { f c } }$ . Randomly initialized networks are shown at zero for reference. ATTRIBUTE MAZE panels show black-background policies, separately for the shape and the colour swap. Signs follow the key in Table 7.

Results. As the groups of the behavioral analysis (Appendix C.2) replicate inside the policy, we present the results in the same groups: first MAZE and LOCATIONREACH, then PUSHTOMARKER and COIN, then ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG. Table 8 gives Q, G and $D _ { F }$ for every edit at both sites. Because robot Q is in units of each policy’s own $\sigma ,$ we read every edit as the ratio of the GM to the Non-GM value within its environment, as Figure 19 does outside, and never compare values across environments. Figure 21 shows these ratios at $h _ { \mathrm { f c } } .$ . In $Q$ every goal edit reaches the GM policy less than the Non-GM policy, with ratios from

Table 8: World-edit Fisher sensitivity in all eight environments. Values are mean $\pm \thinspace \mathrm { S D }$ across three checkpoints per cohort, each summarized by its median over the evaluation states. Environments are in the order of Table 6, and the second edit of an environment edits the proxy. Robot $Q$ and $D _ { F }$ are in units of each policy’s σ and are compared only within a task. Randomly initialized networks have $Q$ and $D _ { F }$ of 0.000 and G between 0.729 and 1.035 at $h _ { \mathrm { p r e } }$ and between 0.785 and 1.081 at $h _ { \mathrm { f c } } .$ Bold marks the larger Q of each GM and Non-GM pair.
<table><tr><td rowspan="2">Environment</td><td rowspan="2">Edit</td><td rowspan="2">Cohort</td><td colspan="3"> $h _ { \mathrm { p r e } }$ </td><td colspan="3"> $h _ { \mathrm { f c } }$ </td></tr><tr><td>Q</td><td>G</td><td> $D _ { F }$ </td><td>Q</td><td>G</td><td> $D _ { F }$ </td></tr><tr><td>MAZE</td><td>Cheese</td><td>GM</td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td><td> $\begin{array} { c } { 1 . 6 4 1 \pm 0 . 3 4 } \\ { 4 . 8 0 0 \pm 0 . 4 6 } \end{array}$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td><td> $0 . 7 2 7 \pm 0 . 2 5$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> $\mathbf { 1 . 6 5 0 \pm 0 . 0 8 }$ </td><td></td><td> $0 . 3 4 8 \pm 0 . 0 1$ </td><td> ${ \bf 1 . 6 8 9 \pm 0 . 0 5 }$ </td><td>1.141 ± 0.39</td><td> $1 . 6 6 7 \pm 0 . 4 4$ </td></tr><tr><td>LOCATIONREACH</td><td>Ball</td><td>GM</td><td> $4 . 0 9 6 \pm 3 . 4 4$ </td><td> $9 . 2 5 2 \pm 4 . 0 6$ </td><td> $0 . 3 7 3 \pm 0 . 2 4$ </td><td> $3 . 8 0 3 \pm 3 . 0 6$ </td><td>1.962 ± 0.28</td><td> $2 . 1 6 7 \pm 1 . 8 0$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> $\mathbf { 2 3 5 . 8 4 9 \pm 8 5 . 0 3 }$ </td><td> $3 8 . 8 8 1 \pm 1 0 . 0 3$ </td><td> $6 . 3 1 7 \pm 1 . 0 6$ </td><td> $\mathbf { 1 6 2 . 7 0 9 \pm 4 7 . 4 3 }$ </td><td>3.822 ± 0.25</td><td>42.961 ± 8.70</td></tr><tr><td>PUSHTOMARKER</td><td>Marker GM</td><td></td><td> $1 9 . 8 1 3 \pm 7 . 0 6$ </td><td> $6 . 2 0 6 \pm 0 . 3 5$ </td><td> $\begin{array} { c } { 3 . 2 7 8 \pm 1 . 1 9 } \\ { 1 6 . 4 6 9 \pm 6 . 6 5 } \end{array}$ </td><td> $1 6 . 9 5 7 \pm 5 . 0 8$ </td><td>2.099 ± 0.39</td><td> $8 . 8 5 2 \pm 3 . 8 8$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> $\mathbf { 1 4 2 . 3 1 6 \pm 4 6 . 7 6 }$ </td><td> $9 . 3 8 8 \pm 0 . 7 8$ </td><td></td><td> $\mathbf { 1 0 6 . 8 8 3 \pm 4 5 . 3 0 }$ </td><td>2.263 ± 0.34</td><td> $4 9 . 9 4 4 \pm 1 4 . 5 8$ </td></tr><tr><td>COIN</td><td>Coin</td><td>GM</td><td> $0 . 0 2 1 \pm 0 . 0 1$ </td><td> $\begin{array} { c } { { 1 . 5 0 8 \pm 0 . 3 9 } } \\ { { 4 . 1 1 2 \pm 1 . 0 3 } } \end{array}$ </td><td> $0 . 0 1 6 \pm 0 . 0 1$ </td><td> $0 . 0 2 1 \pm 0 . 0 1$ </td><td>1.350 ± 0.61</td><td> $0 . 0 1 9 \pm 0 . 0 1$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> ${ \bf 0 . 3 2 4 \pm 0 . 1 0 }$ </td><td></td><td> $0 . 0 7 0 \pm 0 . 0 0$ </td><td>0.339 ± 0.12</td><td>1.784 ± 0.05</td><td>0.207 ± 0.07</td></tr><tr><td>WALKTOFLAG</td><td>Flag</td><td>GM</td><td> $\begin{array} { c } { 1 . 0 9 8 \pm 0 . 1 9 } \\ { 6 . 6 3 2 \pm 6 . 3 7 } \end{array}$ </td><td> $\begin{array} { c } { { 1 0 . 7 3 9 \pm 2 . 5 9 } } \\ { { 3 5 . 3 0 9 \pm 3 4 . 7 9 } } \end{array}$ </td><td> $0 . 1 1 1 \pm 0 . 0 2$ </td><td> $1 . 0 1 8 \pm 0 . 2 2$ </td><td> $_ { 4 . 5 9 3 \pm 2 . 3 4 } ^ { 2 . 1 1 4 }$ </td><td> $0 . 5 7 9 \pm 0 . 1 8$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td></td><td></td><td> $0 . 2 1 4 \pm 0 . 1 0$ </td><td> ${ \bf 6 . 1 5 0 \pm 5 . 5 7 }$ </td><td></td><td> $1 . 1 6 2 \pm 0 . 4 4$ </td></tr><tr><td></td><td>Tower</td><td>GM</td><td> $\begin{array} { c } { \mathbf { 1 5 . 9 9 4 \pm 1 2 . 9 0 } } \\ { \mathbf { 1 . 9 6 0 \pm 0 . 6 8 } } \end{array}$ </td><td> $\begin{array} { l } { 8 . 6 8 6 \pm 5 . 0 2 } \\ { 5 . 9 1 7 \pm 0 . 4 2 } \end{array}$ </td><td> $\begin{array} { c } { 1 . 6 0 0 \pm 0 . 2 7 } \\ { 0 . 3 1 9 \pm 0 . 1 5 } \end{array}$ </td><td> $\begin{array} { c } { { \bf 1 8 . 3 9 8 \pm 1 6 . 2 3 } } \\ { { \bf 3 . 1 2 2 \pm 1 . 4 8 } } \end{array}$ </td><td> $\begin{array} { c } { 2 . 4 0 3 \pm 1 . 1 3 } \\ { 3 . 6 7 0 \pm 0 . 8 4 } \end{array}$ </td><td> $7 . 1 7 0 \pm 3 . 8 6$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td></td><td></td><td></td><td></td><td></td><td> $0 . 8 5 4 \pm 0 . 2 2$ </td></tr><tr><td>ATTRIBUTE MAZE Shape</td><td></td><td>GM</td><td> $1 . 3 5 4 \pm 0 . 6 3$ </td><td> $\begin{array} { c } { 3 2 . 3 0 4 \pm 9 . 2 4 } \\ { 1 6 . 4 5 4 \pm 6 . 7 0 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 0 4 8 \pm 0 . 0 2 } \\ { 0 . 1 6 2 \pm 0 . 0 4 } \end{array}$ </td><td> $1 . 0 9 7 \pm 0 . 4 5$ </td><td> $\begin{array} { c } { 1 7 . 1 6 5 \pm 6 . 3 4 } \\ { 1 0 . 2 6 8 \pm 4 . 4 6 } \end{array}$ </td><td> $0 . 0 6 6 \pm 0 . 0 2$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> ${ \bf 2 . 4 5 8 \pm 0 . 2 5 }$ </td><td></td><td></td><td> ${ \bf 3 . 5 6 9 \pm 0 . 2 7 }$ </td><td></td><td> $0 . 3 9 5 \pm 0 . 1 5$ </td></tr><tr><td></td><td>Colour</td><td>GM</td><td> $\mathbf { 2 . 8 2 1 \pm 0 . 7 7 }$ </td><td> $2 7 . 6 4 7 \pm 3 . 4 5$ </td><td> $\begin{array} { c } { 0 . 0 9 0 \pm 0 . 0 2 } \\ { 0 . 0 1 1 \pm 0 . 0 1 } \end{array}$ </td><td> ${ \bf 2 . 1 2 6 \pm 0 . 8 3 }$ </td><td>13.759 ± 3.99</td><td> $0 . 1 4 2 \pm 0 . 0 3$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td> $0 . 0 3 2 \pm 0 . 0 3$ </td><td> $2 . 6 3 3 \pm 1 . 7 3$ </td><td></td><td> $0 . 0 2 9 \pm 0 . 0 3$ </td><td>2.121 ± 0.89</td><td>0.013 ± 0.01</td></tr><tr><td>ATTRIBUTETABLE</td><td>Shape</td><td>GM</td><td> $1 3 . 7 0 3 \pm 4 . 4 9$ </td><td> $\begin{array} { c } { 2 0 . 5 5 4 \pm 1 0 . 7 7 } \\ { 2 2 . 2 9 1 \pm 3 . 1 0 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 7 4 5 \pm 0 . 2 5 } \\ { 1 . 3 7 4 \pm 0 . 1 4 } \end{array}$ </td><td> $1 2 . 8 9 2 \pm 3 . 9 1$ </td><td> $\begin{array} { c } { 2 . 3 7 4 \pm 0 . 1 3 } \\ { 3 . 6 4 3 \pm 0 . 1 9 } \end{array}$ </td><td>5.382 ± 1.21</td></tr><tr><td></td><td></td><td>Non-GM</td><td> $\mathbf { 3 0 . 0 7 4 \pm 3 . 5 1 }$ </td><td></td><td></td><td> $\mathbf { 4 3 . 4 6 0 \pm 8 . 4 7 }$ </td><td></td><td>11.858 ± 2.85</td></tr><tr><td></td><td>Colour</td><td>GM</td><td> $\mathbf { 9 4 . 3 1 9 \pm 6 . 5 9 }$ </td><td> $_ { 1 1 . 5 1 8 \pm 2 . 4 2 } ^ { 2 4 . 3 8 2 \pm 9 . 5 5 }$ </td><td> $\begin{array} { c } { 4 . 2 9 5 \pm 1 . 5 0 } \\ { 3 . 1 1 8 \pm 0 . 3 2 } \end{array}$ </td><td> $\mathbf { 9 4 . 8 7 6 \pm 3 . 6 1 }$ </td><td> $\begin{array} { c } { 4 . 0 0 9 \pm 0 . 3 1 } \\ { 2 . 8 5 0 \pm 0 . 1 5 } \end{array}$ </td><td>24.001 ± 2.38</td></tr><tr><td></td><td></td><td>Non-GM</td><td> $3 5 . 5 1 8 \pm 4 . 9 4$ </td><td></td><td></td><td> $4 6 . 0 3 7 \pm 7 . 1 5$ </td><td></td><td>16.829 ± 2.22</td></tr><tr><td>SORTZONES</td><td>Cross</td><td>GM</td><td> $8 . 0 5 0 \pm 4 . 0 0$   $\mathbf { 1 1 8 . 0 8 2 \pm 4 1 . 7 8 }$ </td><td> $5 . 9 5 8 \pm 1 . 8 8$ </td><td> $1 . 3 2 6 \pm 0 . 3 6$   $4 . 7 2 8 \pm 0 . 6 6$ </td><td> $1 0 . 4 8 6 \pm 5 . 6 5$   $\mathbf { 1 0 8 . 7 3 0 \pm 3 6 . 1 2 }$ </td><td> $1 . 8 5 7 \pm 0 . 4 8$   $2 . 3 1 2 \pm 0 . 1 9$ </td><td> $5 . 4 0 3 \pm 1 . 5 1$ </td></tr><tr><td></td><td></td><td>Non-GM</td><td></td><td> $2 4 . 6 7 0 \pm 5 . 6 3$ </td><td></td><td></td><td></td><td> $4 7 . 7 3 9 \pm 1 1 . 2 6$ </td></tr><tr><td></td><td>Colour</td><td>GM</td><td> $\begin{array} { c } { 9 8 . 6 2 3 \pm 3 6 . 1 9 } \\ { 3 3 . 0 4 1 \pm 1 7 . 6 6 } \end{array}$ </td><td> $\begin{array} { r } { 1 9 . 4 7 6 \pm 5 . 7 3 } \\ { 1 0 . 2 9 2 \pm 2 . 8 8 } \end{array}$ </td><td> $5 . 0 0 2 \pm 0 . 9 7$ </td><td> $\begin{array} { c } { { \bf 1 1 0 . 7 8 4 \pm 4 0 . 3 1 } } \\ { { \bf 3 3 . 5 0 2 \pm 2 0 . 0 0 } } \end{array}$ </td><td> $\begin{array} { c } { 3 . 1 4 2 \pm 0 . 9 8 } \\ { 1 . 5 5 3 \pm 0 . 3 9 } \end{array}$ </td><td>34.772 ± 2.87</td></tr><tr><td></td><td></td><td>Non-GM</td><td></td><td></td><td>3.141 ± 0.90</td><td></td><td></td><td>20.706 ± 9.45</td></tr></table>

![](images/0df0b77bca102587520f0fe4a5629d9d07817b39bab576b4e4ff6aacbea1dd57.jpg)

Figure 21: Fisher sensitivity of GM agents relative to Non-GM agents. GM value of $Q , D _ { F }$ and G at $h _ { \mathrm { f c } }$ divided by the Non-GM value for the same edit, filled for goal and open for proxy edits. Large markers divide cohort means, small dots each GM seed by the Non-GM mean, and dashed lines mark a value as large as in Non-GM agents. Rows are in the order of Table 6.

below 0.001 (MAZE) to 0.307 (ATTRIBUTE MAZE), and every proxy edit reaches it more, from 2.061 (ATTRIBUTETABLE) to 73.994 (ATTRIBUTE MAZE), with every GM seed above the Non-GM mean. $D _ { F }$ lies on the same side of 1 as $Q$ for every edit, whereas the ratio of G stays between 0.460 and 2.023 for every edit except the ATTRIBUTE MAZE colour swap (6.486).

MAZE and LOCATIONREACH. In MAZE and LOCATIONREACH the goal edit is as inert inside the policy as it is in behavior. Relocating the cheese and moving the ball reach the GM policy far less than the Non-GM policy, with a GM to Non-GM ratio of Q below 0.001 in MAZE and of 0.023 in LOCATIONREACH, where the behavioral KL ratios are 0.009 and 0.026. These are the two smallest goal-edit ratios of the eight environments, against 0.063 to 0.307 for the other six. The separation holds for the cohort means, not for every seed, since the most sensitive LOCATIONREACH GM seed reaches 0.041 of the LOCATIONREACH Non-GM mean and the least sensitive COIN GM seed 0.045 of the COIN Non-GM mean. In MAZE the gap lies in the displacement, with a ratio of $D _ { F }$ of 0.001 against a ratio of G of 0.637, so the direction of the cheese displacement is not read strongly enough to compensate. Cheese information may still be represented elsewhere, but its movement barely changes the directions the native policy reads. In LOCATIONREACH both factors contribute, with ratios of 0.050 for $D _ { F }$ and 0.513 for G. The signed target-selection effect of GM agents in MAZE stays at −0.001 at both sites against −0.271 and −0.269 for Non-GM agents, where a negative value means a shift toward the relocated cheese (Table 7 and Figure 20), matching a signed follow of −0.001. In LOCATIONREACH the small reaction even turns the planned GM step slightly away from the moved ball (signed follow −1.764 mm against +199.668 mm, Table 6).

PUSHTOMARKER and COIN. Here the goal edit still reaches the GM policy but does not direct it, as in the behavioral edits. It reaches the GM policy at about a sixth of the Non-GM sensitivity in PUSHTOMARKER (ratio of Q 0.159, behavioral KL ratio 0.206) and at about a sixteenth in COIN (0.063, behavioral KL ratio 0.114, nearly the same at $h _ { \mathrm { p r e } } )$ . In both most of the gap lies in the displacement, with ratios of $D _ { F }$ of 0.177 and 0.093 against ratios of G of 0.927 and 0.757. The reaction does not point at the goal. In COIN the signed effect of GM agents stays near zero at both sites while Non-GM agents shift toward the moved coin (−0.010 against +0.139 at $h _ { \mathrm { f c } } ,$ , Table 7 and Figure 20), and in PUSHTOMARKER the planned GM gripper step does not turn toward the moved marker (signed follow +0.487 mm against +27.948 mm, Table 6). This is the inside counterpart of the behavioral result, where the edit moves probability mass but not toward the goal.

ATTRIBUTE MAZE, ATTRIBUTETABL $\mathbf { \delta E } ,$ SORTZONES and WALKTOFLAG. In these four environments the goal edit reaches the GM policy less than the Non-GM policy and the proxy edit reaches it more, as in behavior, where the goal enters the decision and loses to the proxy. In the order ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG, the ratio of Q is 0.307, 0.297, 0.096 and 0.165 for the goal edit (the shape swap, the cross swap and the flag move) and 73.994, 2.061, 3.307 and 5.893 for the proxy edit. Within the GM policy of each environment the proxy edit outweighs the goal edit, with a ratio of proxy-edit to goal-edit Q of 1.938, 7.359, 10.565 and 18.080 in the same order. In ATTRIBUTE MAZE this organization is already visible at $h _ { \mathrm { p r e } }$ and is reproduced under textured observations. The Non-GM agents differ. In ATTRIBUTE MAZE they barely react to the proxy edit, whose $Q$ is 0.008 of their goal-edit $Q ,$ , whereas the robot Non-GM agents react to it about as much as to the goal edit in ATTRIBUTETABLE (1.059), a third as much in SORTZONES (0.308) and half as much in WALKTOFLAG (0.508).

In ATTRIBUTE MAZE the signed effect shows what the two directions do. The shape and the colour swap favour opposite candidates, so the contrast $C _ { s }$ gives them opposite signs whenever the policy follows both edits, and the sign pattern alone is not evidence of competition. What matters is that both effects are large and consistent. In GM agents each displacement shifts preference toward the candidate its swap favours in almost every conflict state (−0.665 for shape and +0.918 for colour at $h _ { \mathrm { f c } }$ , and likewise at $h _ { \mathrm { p r e } } )$ , whereas Non-GM agents follow the shape edit just as strongly (−0.897) and the colour edit an order of magnitude less (+0.063, Table 7 and Figure 20). So the goal edit reaches the GM policy at a fraction of the Non-GM sensitivity, as in COIN, but here it moves preference toward the goal. On the robots the planned GM step also still turns toward the moved goal, less than the Non-GM step (about half in WALKTOFLAG), and in SORTZONES it turns toward the swapped colour four times as much as the Non-GM step (Table 6). Whether the two cues actually pull against each other is tested next through the push–pull redistribution of target support and the within-state displacement geometry (Appendix C.5.1).

What the decomposition shows. G and $D _ { F }$ describe how a sensitivity is carried and are not deci sion rules (Table 8, Figure 21). On the goal edit GM agents have a lower gain than Non-GM agents in seven of the eight environments, by at most about a factor of two. The exception is ATTRIBUTE MAZE, where the goal displacement is smaller in GM agents but falls on directions the GM policy reads more strongly. The displacement ratio is smallest in MAZE (0.001) and LOCATIONREACH (0.050), but per seed LOCATIONREACH overlaps with COIN, so the two do not separate cleanly

Table 9: The two factors of $D _ { F }$ . Policy sensitivity is the mean Fisher sensitivity $\operatorname { t r } ( F ) / d$ at the unedited state, which sets how strongly the policy reads an average direction. Displacement norm is the Euclidean length of the goal edit’s displacement. Each is the median over the states of Table 8 for one checkpoint, then mean $\pm \mathrm { \textbf { S D } }$ over checkpoints. ATTRIBUTE MAZE uses the shape swap, and robot values are in units of $\sigma ^ { - 2 }$ and compared only within a task.
<table><tr><td rowspan="2">Environment</td><td rowspan="2">Site</td><td colspan="2">Policy sensitivity  $1 0 ^ { 3 } \operatorname { t r } ( F ) / d$ </td><td colspan="2">Displacement norm  $\| \Delta \| _ { 2 }$ </td></tr><tr><td>GM</td><td>Non-GM</td><td>GM</td><td>Non-GM</td></tr><tr><td rowspan="2">MAZE</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 5 1 5 \pm 0 . 1 3$ </td><td> $1 . 1 9 2 \pm 0 . 3 6$ </td><td> $1 . 2 1 2 \pm 0 . 0 7$ </td><td> $1 8 . 0 5 2 \pm 3 . 1 0$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $4 . 3 2 8 \pm 0 . 4 1$ </td><td> $5 . 3 9 3 \pm 0 . 1 1$ </td><td> $0 . 8 4 9 \pm 0 . 0 4$ </td><td> $1 8 . 5 9 1 \pm 2 . 6 4$ </td></tr><tr><td rowspan="2">LOCATIONREACH</td><td> $h _ { \mathrm { p r e } }$ </td><td> $2 4 . 6 9 7 \pm 5 . 5 5$ </td><td> $1 5 . 3 8 3 \pm 2 . 8 2$ </td><td> $3 . 8 4 4 \pm 1 . 8 7$ </td><td> $2 0 . 3 7 5 \pm 2 . 2 1$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $3 4 6 . 0 1 3 \pm 4 2 . 8 1$ </td><td> $1 9 1 . 4 0 9 \pm 1 2 . 8 1$ </td><td> $2 . 3 1 4 \pm 1 . 4 3$ </td><td> $1 4 . 9 8 9 \pm 1 . 8 9$ </td></tr><tr><td rowspan="2">PUSHTOMARKER</td><td> $h _ { \mathrm { p r e } }$ </td><td> $2 1 2 . 7 0 5 \pm 3 2 . 1 2$ </td><td> $5 6 6 . 7 2 5 \pm 2 9 9 . 4 7$ </td><td> $3 . 8 9 2 \pm 0 . 8 3$ </td><td> $5 . 9 2 6 \pm 1 . 0 2$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $1 3 1 5 . 7 8 2 \pm 1 4 0 . 0 4$ </td><td> $1 9 9 7 . 9 4 0 \pm 6 3 5 . 8 6$ </td><td> $2 . 6 3 1 \pm 0 . 6 8$ </td><td> $5 . 2 6 8 \pm 0 . 6 1$ </td></tr><tr><td rowspan="2">COIN</td><td> $h _ { \mathrm { p r e } }$ </td><td> $1 . 3 3 6 \pm 0 . 7 7$ </td><td> $0 . 5 3 8 \pm 0 . 1 2$ </td><td> $3 . 6 9 2 \pm 0 . 7 3$ </td><td> $1 1 . 8 1 4 \pm 1 . 4 9$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $7 . 2 1 9 \pm 4 . 5 4$ </td><td> $3 . 7 1 2 \pm 0 . 2 5$ </td><td> $1 . 7 1 4 \pm 0 . 3 3$ </td><td> $7 . 6 8 5 \pm 1 . 3 8$ </td></tr><tr><td rowspan="2">WALKTOFLAG</td><td> $h _ { \mathrm { p r e } }$ </td><td> $9 . 4 2 0 \pm 3 . 1 6$ </td><td> $1 0 . 0 3 9 \pm 4 . 2 7$ </td><td> $3 . 4 3 3 \pm 0 . 4 4$ </td><td> $4 . 8 2 6 \pm 0 . 9 9$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $9 4 . 1 8 4 \pm 1 7 . 5 7$ </td><td> $6 7 . 8 9 1 \pm 1 8 . 9 5$ </td><td> $2 . 4 5 3 \pm 0 . 2 3$ </td><td> $4 . 0 7 8 \pm 0 . 2 8$ </td></tr><tr><td rowspan="2">ATTRIBUTE MAZE</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 2 2 5 \pm 0 . 0 8$ </td><td> $0 . 0 9 6 \pm 0 . 0 3$ </td><td> $1 4 . 2 7 1 \pm 1 . 4 5$ </td><td> $4 3 . 3 8 7 \pm 1 0 . 5 3$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 8 9 3 \pm 0 . 1 6$ </td><td> $0 . 7 7 4 \pm 0 . 1 4$ </td><td> $8 . 7 9 2 \pm 1 . 0 0$ </td><td> $2 3 . 4 7 6 \pm 5 . 4 0$ </td></tr><tr><td rowspan="2">ATTRIBUTETABLE</td><td> $h _ { \mathrm { p r e } }$ </td><td> $1 7 . 2 2 2 \pm 8 . 8 1$ </td><td> $7 . 9 4 1 \pm 1 . 5 5$ </td><td> $7 . 0 4 1 \pm 2 . 5 4$ </td><td> $1 3 . 3 7 6 \pm 1 . 3 6$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $2 1 6 . 9 9 1 \pm 8 9 . 3 4$ </td><td> $9 3 . 9 3 7 \pm 9 . 8 7$ </td><td> $5 . 2 4 3 \pm 1 . 1 1$ </td><td> $1 1 . 3 2 1 \pm 1 . 8 1$ </td></tr><tr><td rowspan="2">SORTZONES</td><td> $h _ { \mathrm { p r e } }$ </td><td> $5 9 . 2 4 0 \pm 1 3 . 9 8$ </td><td> $6 3 . 1 8 4 \pm 7 . 1 6$ </td><td> $4 . 7 1 5 \pm 0 . 3 2$ </td><td> $8 . 6 7 1 \pm 0 . 7 7$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td> $8 6 1 . 2 0 2 \pm 8 3 . 2 2$ </td><td> $8 5 2 . 1 8 7 \pm 1 0 8 . 6 3$ </td><td> $2 . 4 9 4 \pm 0 . 2 5$ </td><td> $7 . 4 9 8 \pm 0 . 6 9$ </td></tr></table>

from the rest. In $\mathbf { M A Z E } ,$ COIN and PUSHTOMARKER the gap lies mostly in the displacement, whereas in LOCATIONREACH and WALKTOFLAG the gain also differs by about a factor of two.

Policy sharpness does not produce these displacement gaps (Table 9). In Procgen the mean Fisher sensitivity of GM policies is at most 2.3 times smaller than that of Non-GM policies, and larger in COIN and ATTRIBUTE MAZE, while in MAZE and COIN the displacement norm differs by a factor of 3 to 22. In ATTRIBUTE MAZE the GM policies also have the higher action entropy, so they are the less sharp ones. In ATTRIBUTETABLE and LOCATIONREACH the GM arm policies are the sharper ones, with a smaller mean $\sigma$ and about twice the mean sensitivity, which works against a small GM $D _ { F }$ . PUSHTOMARKER is the one task where the sensitivity factor carries part of the gap, 1.5 times smaller in GM agents at $h _ { \mathrm { f c } }$ against a factor of 4 in $\| \Delta \| _ { 2 } ^ { 2 } . \mathrm { ~ \ r ~ A ~ }$ separate depth sweep with three checkpoints per cohort scales the goal edit by the spread of the scenes, the median squared distance between two random base states at the same site. In randomly initialized networks this scaled edi keeps its size from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ . In LOCATIONREACH and SORTZONES it grows in Non-GM agents but not in GM agents, and in PUSHTOMARKER it shrinks in both cohorts, by 41% in Non-GM and 60% in GM agents.

We therefore read the decomposition as a description of how each network carries a given sensitivity. What carries the comparison between the cohorts is $Q ,$ whose order between the cohorts mirrors the behavioral KL of every edit from the inside. Two limitations remain. $G$ and $D _ { F }$ depend on the coordinates of each site (Appendix B.2), and with three seeds per cohort several robot intervals are wide, so the ordering should be read as an indication.

Together. The Fisher view therefore recovers the same progression that was visible from behavior, but now from inside the policy. In MAZE and LOCATIONREACH the goal edit barely moves the native policy, in PUSHTOMARKER and COIN it moves it but not toward the goal, and in the four environments with a proxy edit both edits stay coupled to the policy, with the proxy edit reaching GM agents more strongly than Non-GM agents. So the common behavioral label “goal misgeneralization” hides different relationships between the intended goal and the native policy. The goal can be nearly ignored, weakly coupled, or actively read alongside a competing proxy. This mirrors the probes (Appendix C.1). In the competition group the policy reads proxy and goal, and the GM policy reads the proxy more strongly, as GM agents decode the proxy better and Non-GM agents the goal. In the inert group decodability and policy sensitivity are low together, since the goal is neither linearly decodable above random networks in GM agents nor read by their policy. Across probes, world edits and Fisher sensitivity the same groups emerge, with the one exception of PUSH-TOMARKER, whose marker is not decodable above random networks but still weakly reaches the policy. This increasingly establishes a depth at which the goal is lost that replicates across experiment types. G and $D _ { F }$ describe how the sensitivity is carried but cannot be used as decision rules.

## C.4 WHAT ROLE DOES THE INTENDED-GOAL INFORMATION PLAY?

The earlier analyses, from inside the network and from the outside, sort the environments into groups by how far the goal reaches the policy, and they measure the immediate reaction of the next action. This chapter lengthens the time horizon and asks what organizes behavior over the whole episode, and therefore what truly acts as the goal of the agent. Does the remaining goal information organize behavior, or does it only nudge the next step? What takes the goal’s place, and what role does the proxy play? We walk through five groups. First come the environments where the substitute target is a location and the policy is inert to the goal, MAZE and LOCATIONREACH (Appendix C.4.1). Second comes PUSHTOMARKER, where the substitute is again a location but the policy is weakly wired to the goal (Appendix C.4.2). Third comes COIN, a weakly wired environment that we sus pect of a policy shortcut (Appendix C.4.3). Fourth comes cue competition within one object, in ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES (Appendix C.4.4). Last comes WALK-TOFLAG, where two objects compete (Appendix C.4.5). Many of these analyses are specific to one environment, so we group them by analysis type, and we will find that they sort the environments into three categories, functional target substitution, a policy shortcut, and cue or object competition.

## C.4.1 FUNCTIONAL TARGET SUBSTITUTION: MAZE AND LOCATIONREACH

Both environments hold the goal at one place in GM training, the cheese in the top-right corner of every maze and the ball at one point in the back-right corner of the workspace (Appendices A.1.1 and A.2.2), and in both the goal edit barely moves the GM policy (Appendix C.2). Something else must therefore organize the episode, and we ask two questions about it. Is the proxy a target the agent walks to, or only a direction of movement? What kind ofplace is the target, in MAZE a memorized place or a spatial relation, and in LOCATIONREACH a point on the table, an arm posture or a place in the camera image?

Design. To ask whether the proxy is a target, we use states in which moving in a fixed direction is not the same as going to a target, those where the agent must make a detour to reach the corner. These are the 168 detour states of the detour bank (Table 21, Appendix E.3), chosen from the maze graph alone. On them the graph-exact progress sets toward the cheese and toward the reachable top-right extreme (Definition 10) are disjoint, and no action that makes progress toward the extreme points up or right, so an agent that only moves up or right cannot follow the corner there. We classify the greedy action as corner progress, cheese progress, naive only (up or right without progress toward either target) or neutral.

In LOCATIONREACH we ask whether the proxy is a target with an off-path test. Each ball is placed at a position that the same GM gripper passes through at ball height from some other start, at least 8 cm from its path in the current episode and at least 10 cm from the corner, so reaching the ball asks for no movement the GM agent has not already made. We record which of corner and ball the gripper reaches first. The 500 episodes per seed share 26 to 32 distinct ball positions, so the effective number of goal positions is about 30 per seed.

To ask what kind of place the target is in MAZE, we borrow the corner-opening edit of Mini et al. (2023), who use it as the contrast pair of a steering vector and note that it attracts the agent, and use it instead as a test of the target. The edit carves a new path to the top-right interior cell of the full grid, which lies outside the original maze in 80% of states and so creates a new top-right extreme (Fig. 22b and c show the carved cells). A second edit separates this extreme from the screen corner. It carves a new extreme only two cells beyond the old one, joined by a three-cell corridor that doe not pass through the old corner and stays at least three cells from the screen corner. It needs room in the margin and so covers mazes of size at most 17, which is 64% of the states. With each edit, a corridor of the same length carved beyond the bottom-left extreme tests whether newly opened space alone attracts the agent.

(a) Unedited  
![](images/35ca18cc3059a7b60d8d93a865350db92235c18d7b3303be0ee168763de5ed2f.jpg)

![](images/72ab81adccb67d671daa1ae996c44a26fc2d2ee833d14d116f3f46716fe59203.jpg)

![](images/73bf370f83b49e0ee7898625aa815ab033a413f00384b9a58f481071d6a65b87.jpg)

![](images/b24a5ebc0a029cf6dd377601b325f6388d121358635d107a18a58bdd899bcd69.jpg)

![](images/8724620ff7f8846ea305397e335f817efcd1ffc098fae52b2048ee06f9af9f05.jpg)

![](images/65ba6e3c4e4b29df852fcebd8aa35722aa72b3b20c09ae4ee07cb6a5d6abb311.jpg)

![](images/c35e99e46456b517f9074fb951a73961e531b5c73ca3f5b4534db87646a2545e.jpg)

![](images/8ac32373b8562cd077ed09bd7a130698b186fdc0d7d3b496376ec3bcef2b2dda.jpg)  
GM Non-GM ball distractor cube ball workspace cohort mean 3 cm reach radius

Figure 22: GM agents go to the current top-right extreme in MAZE and to the training corner in LOCATIONREACH. (a to d) Greedy GM rollouts unedited, with the corner opened, with the offset extreme and with a bottom-left corridor, on a selected example: all three GM checkpoints agree on this 15-cell maze (level 1000140), which is one of 14 of 92 medium or large levels where they do. (e, f) Two LOCATIONREACH off-path episodes, with the GM and Non-GM grippers from the same start. (g) GM gripper positions over all 1,500 off-path episodes. (h) Closest approach of each episode to the corner and to the ball.  
Table 10: Greedy action class on the 168 MAZE detour states. Mean ± SD across three seeds.
<table><tr><td>Cohort</td><td></td><td>Corner progress Cheese progress</td><td>Naive only</td><td>Neutral</td></tr><tr><td>GM</td><td> $0 . 9 4 8 \pm 0 . 0 2$ </td><td> $0 . 0 1 8 \pm 0 . 0 1$ </td><td> $0 . 0 2 0 \pm 0 . 0 1$ </td><td> $0 . 0 1 4 \pm 0 . 0 1$ </td></tr><tr><td>Non-GM</td><td> $0 . 1 7 3 \pm 0 . 0 5$ </td><td> $0 . 4 9 8 \pm 0 . 0 2$ </td><td> $0 . 0 6 7 \pm 0 . 0 1$ </td><td> $0 . 2 6 2 \pm 0 . 0 3$ </td></tr></table>

In LOCATIONREACH the training corner is an empty patch of the table, so instead of moving it we move what the policy could use to define it. Shifting the arm base by 8 cm changes which joint posture puts the gripper at the corner, so a posture target predicts a point 8 cm from the corner. Shifting the camera by 5 cm moves the ground point under every pixel by 5 cm, so a target fixed in the image predicts a point 5 cm from the corner, whereas a place on the table predicts the corner itself. The ball stays at the corner in both edits, so Non-GM agents have the same target as GM agents, and each condition plays 500 held-out starts per seed. Without any edit we also compare the spread of the arrival postures along the arm’s null space, the joint directions that leave the gripper position unchanged, with their spread at the start, which a posture target would collapse.

Results: is the proxy a target or a direction? In both environments the proxy is a target. On the 168 detour states GM policies choose corner-progress actions on 0.948 of states, every one of them outside the up-right family, whereas Non-GM policies make corner progress on 0.173, cheese progress on 0.498 and neutral moves on 0.262 (Table 10). The GM agent thus follows the corner where this requires moving down or left, which a fixed up-right rule would not do. In LOCATIONREACH GM agents reach the corner first in 0.993 of the off-path episodes and the ball first in none, whereas Non-GM agents reach the ball first in 0.999 from the same starts (Fig. 22e and f show two of these episodes). On average the GM gripper comes within 1.4 cm of the corner and 16.3 cm of the ball, and the Non-GM gripper within 0.9 cm of the ball and 19.1 cm of the corner (Fig. 22h). The GM paths converge on the corner from starts all over the workspace (Fig. 22g), as a target does and a fixed direction of movement does not. We score the first visit. What GM agents do after it varies between checkpoints of the same seed (hold rate 0.92 to 0.03 over the last evaluations of one seed, against 0.75 to 0.9 for Non-GM agents), and since training ends an episode at the first contact this behaviour was never trained.

Results: what kind of place is the target? In MAZE the target is a spatial relation, not a remembered place. After the global corner edit, on the states where the first steps toward the old and the new corner differ, GM agents step toward the new corner in $0 . 9 0 9 \pm 0 . 0 1$ of cases and toward the old one in $0 . 0 1 3 \pm 0 . 0 2$ , and in greedy rollouts they reach the new corner first in $0 . 7 4 8 \pm 0 . 0 3$ and the old one first in none, although the same agents reach the old corner in $0 . 8 5 7 \pm 0 . 0 2$ on the unedited mazes (one example level in Fig. 22a and b). Taking the first eligible state of each level, a GM checkpoint reaches the old corner unedited and the new extreme after the edit, without touching the old corner or ending on the cheese, on 0.557 of states (95% CI 0.429 to 0.685 over the three seeds). Non-GM agents also step toward the new corner $( 0 . 6 9 6 \pm 0 . 0 9 )$ , but on their states this step points to the cheese in 69% of cases, against 26% to 32% for GM agents, and they never reach the new corner first, so the first step alone does not separate the cohorts and reaching the new corner first does. With the offset corridor they reach the new extreme first in $0 . 7 9 2 \pm 0 . 0 5$ of rollouts and the old one in $0 . 1 1 4 \pm 0 . 0 9$ , against $0 . 9 3 2 \pm 0 . 0 4$ for the old corner without the edit (Fig. 22c). Newly opened space alone does not attract them. After the global edit GM agents enter the top-right corridor in $0 . 9 2 5 \pm 0 . 0 3$ of rollouts and the bottom-left one in $0 . 0 1 2 \pm 0 . 0 2$ (Fig. 22d), and the bottom-left corridor of the offset edit is never entered, while the agents still reach the old corner first in $0 . 9 4 3 \pm 0 . 0 4$ of rollouts, as they do without any edit. Non-GM agents enter the bottom-left corridor of the global edit in $0 . 0 3 6 \pm 0 . 0 1$ of rollouts. The proxy is therefore the top-right extreme of the current maze layout, not a remembered location and not the screen’s corner cell. A target defined as the reachable cell nearest the screen corner coincides with this extreme in both edits and is not ruled out. Mini et al. (2023) report qualitatively that the opened corner attracts the agent, and the two edits with their bottom-left controls turn this into a measured test of what kind of place the target is.

In LOCATIONREACH the target is a place in the camera image, neither an arm posture nor a point on the table. With the arm base shifted by 8 cm, GM agents still reach the table corner in 0.997 of episodes and the point the old posture implies in 0.011. At the corner their arrival postures spread along the null space as much as at the start (0.975 of the start spread, 95% CI 0.84 to 1.11), as do those of Non-GM agents reaching a ball there (1.092, 0.81 to 1.37), so the posture is not what the agent holds fixed. With the camera shifted by 5 cm, GM agents reach the point that projects to the old corner pixel in 0.993 of episodes and the real corner in 0.025, whereas Non-GM agents still reach the ball in 0.971, so the target moves with the image and not with the table. A camera shift moves the table edge in the image together with the pixel, so we also translate only the policy image, with the scene unchanged. Horizontally the GM gripper follows the pixel in all three seeds (0.983 reach the pixel point and 0.006 the real corner at a shift of −16 pixels), whereas vertically the result is mixed between seeds. Figure 23a to d shows where the gripper arrives under each edit, and Figure 23e gives the reach rates. That it is not a posture and is anchored in the image is definitive, with effects near 0 against 1 in every seed. That the anchor is a fixed pixel rather than an offset from the table edge or the arm base in the image we hold with moderate to strong confidence, strong along the horizontal and mixed along the vertical.

Both environments therefore showfunctional target substitution, in MAZE with a spatial relation as the substitute target and in LOCATIONREACH with a place fixed in the camera image.

## C.4.2 TARGET SUBSTITUTION WITH UNDIRECTED GOAL WIRING: PUSHTOMARKER

In PUSHTOMARKER the proxy is again a place, but unlike in MAZE and LOCATIONREACH the goal still reaches the GM policy. The marker is trained at the table centre, so the proxy is a location, and moving the marker changes the GM action distribution by about a fifth of the Non-GM change in KL divergence (Table 11, Appendix C.2). The world-edit chapter also shows that this change has no goal direction, with a signed follow near zero. The goal is thus wired in, but without direction, and we ask four questions. Is the proxy a target? What kind of place is the target, a point on the table, an arm posture or a place in the camera image? Does the marker organize the whole episode, or does it only nudge the next step? What is the undirected reaction to the marker?

![](images/10f52b39cbfef078df096cabc4a84fd4984600db61fe1e1db4dd450f16d052ca.jpg)  
GM Non-GM table corner / centre old-posture / old-pixel point reached table point reached other point

Figure 23: In LOCATIONREACH and PUSHTOMARKER the target moves with the camera image, not with the arm base. (a to d, f to i) Where the gripper arrives in LOCATIONREACH and where the cube comes to rest in PUSHTOMARKER, over all episodes of the three seeds per cohort, unedited, with the arm base shifted (Base), with the camera shifted (Camera) and with only the policy image translated (Image), the table corner or centre ringed and the point that the old posture or old pixel implies drawn as a diamond. (e, j) Fraction of episodes that reach the table point (filled) or the other point (open), where PUSHTOMARKER GM agents play without the marker except under the camera shift, which draws it at the real centre for both cohorts.

Design. For the first question the conflict test cannot show where the agent pushes, because the marker lies on the ray from the cube to the centre and the push to the marker is the first part of the push to the centre (Appendix A.2.1). We therefore use layouts from the Non-GM training distribution, with marker and cube placed independently, and keep those where reaching the marker and reaching the centre are disjoint outcomes and the marker lies off the straight path to the centre (399 layouts per seed, Fig. 24a). A decoupled conflict draws the disc 41<sup>◦</sup> off the ray to the centre, about 9 cm from the path, on each of the 512 conflict scenes per seed (Fig. 24c and d).

For the second question we use the two edits of LOCATIONREACH (Appendix C.4.1). The centre is an empty spot of the table once the marker is gone, so instead of moving it we move what the policy could use to define it. Shifting the arm base by 8 cm changes which joint posture puts the cube at the centre, so a posture target predicts a point 8 cm from the centre. Shifting the camera by 5 cm moves the ground point under every pixel by 5 cm, so a target fixed in the image predicts a point 5 cm from the centre, whereas a place on the table predicts the centre itself. We draw the marker at the real centre during the camera shift, so Non-GM agents keep the same target as GM agents and the visible marker competes with the image location. As in LOCATIONREACH, we also translate only the policy image, with the scene unchanged. Each condition plays 500 held-out starts per seed with the cube on the GM training ring around the centre, and we record where the cube comes to rest.

For the third question we roll out each layout with the marker drawn and hidden and compare the two rollouts, by where the cube passes and by whether it pauses over the marker (Fig. 24a and b).

For the fourth question we start from the marker move of the world edits (Fig. 24e), where we measure signed follow, the turn of the planned gripper step toward the new push point (Figure 3c), and its cosine to the shift of the push point. A disc control then asks whether the marker gets a reaction of its own. A presence test renders each frozen scene with and without a disc and compares the two action outputs, with the disc at the conflict position or 75<sup>◦</sup> off the path at 3 to 11 cm from the cube (Fig. 24f and g), in the white of the marker or in a green that never marked a goal, and repeats the test with the gripper held above the cube (Fig. 24h). The presence test measures sensitivity, not use. A nonzero response means the disc reaches the action head, not that the policy treats it as a goal or that the robot’s behavior changes, since in closed loop the policy can correct a small perturbation on the next frame. Figure 24 draws every configuration.

Table 11: What organizes PUSHTOMARKER and what the marker reaction is. $\mathbf { M e a n } \pm \mathbf { S D }$ across three seeds of per-seed means, the planned step a per-seed median. Rollouts are fractions of episodes on the separable layouts or the conflict scenes, and have no init baseline because randomly initialized policies never reach the cube. The path pull is how much closer the cube passes to the marker when it is drawn than when it is hidden. Single-frame rows use the world-edit bank with the gripper at its start pose unless the block says otherwise.
<table><tr><td></td><td>GM</td><td>Non-GM</td><td>Init</td></tr><tr><td colspan="4">Rollouts</td></tr><tr><td>Separable, cube to the centre</td><td> $0 . 9 8 0 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td></td></tr><tr><td>Separable, cube onto the marker</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 3 5 \pm 0 . 0 3$   $1 0 . 5 0 4 \pm 0 . 5 0$ </td><td></td></tr><tr><td>Separable, path pull to the marker in cm</td><td> $0 . 0 5 0 \pm 0 . 0 3$   $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 8 9 5 \pm 0 . 0 4$ </td><td></td></tr><tr><td>Conflict, cube onto the decoupled disc</td><td></td><td> $0 . 8 6 7 \pm 0 . 0 3$ </td><td></td></tr><tr><td>Conflict, pause over the marker, drawn Conflict, pause over the marker, hidden</td><td> $0 . 0 5 8 \pm 0 . 0 2$   $0 . 0 5 0 \pm 0 . 0 1$ </td><td> $0 . 0 7 2 \pm 0 . 0 4$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td colspan="4">Marker move</td></tr><tr><td>KL divergence</td><td> $1 6 . 2 9 4 \pm 0 . 8 6$ </td><td> $7 9 . 1 6 1 \pm 2 7 . 4 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Signed follow in mm</td><td> $+ 0 . 4 8 7 \pm 1 . 4 0$   $+ 0 . 0 0 2 \pm 0 . 0 1$ </td><td> $+ 2 7 . 9 4 8 \pm 1 3 . 9 8$ </td><td> $+ 0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Signed follow over the planned step Cosine to the push-point shift</td><td> $- 0 . 0 4 1 \pm 0 . 0 5$ </td><td> $+ 0 . 1 0 4 \pm 0 . 0 4$ </td><td> $- 0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Cosine to the push-point shift, gripper above the cube</td><td> $- 0 . 0 6 4 \pm 0 . 0 4$ </td><td> $+ 0 . 4 3 9 \pm 0 . 1 5$   $+ 0 . 5 4 2 \pm 0 . 1 7$ </td><td> $- 0 . 0 2 1 \pm 0 . 1 1$   $- 0 . 0 0 3 \pm 0 . 1 0$ </td></tr><tr><td>Disc presence, change of the commanded step in mm</td><td></td><td></td><td></td></tr><tr><td>No disc, planned step</td><td> $1 1 9 . 5 0 8 \pm 5 . 1 1$ </td><td> $1 3 2 . 2 7 8 \pm 3 . 2 6$ </td><td> $0 . 2 4 7 \pm 0 . 1 5$ </td></tr><tr><td>White marker, conflict position</td><td> $1 3 . 3 8 5 \pm 3 . 7 4$ </td><td> $2 4 . 5 3 9 \pm 2 . 4 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Green disc, conflict position</td><td> $1 4 . 1 8 9 \pm 0 . 0 7$ </td><td> $1 6 . 4 0 4 \pm 4 . 9 3$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>White marker, off path at 3 cm</td><td> $1 0 . 5 9 7 \pm 2 . 7 7$ </td><td> $3 4 . 7 5 1 \pm 7 . 7 8$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>White marker, off path at 11 cm</td><td> $1 2 . 2 9 4 \pm 4 . 6 3$ </td><td> $3 2 . 3 2 4 \pm 9 . 5 5$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Green disc, off path at 3 cm</td><td> $1 8 . 8 0 3 \pm 3 . 3 8$ </td><td> $2 6 . 5 3 9 \pm 6 . 9 5$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Green disc, off path at 11 cm</td><td> $7 . 4 2 7 \pm 3 . 2 7$ </td><td></td><td></td></tr><tr><td></td><td></td><td> $1 6 . 8 0 4 \pm 9 . 4 5$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Disc presence, gripper above the cube, cosine of the step change</td><td></td><td></td><td></td></tr><tr><td>White disc, off path at 5.2 cm, to the push onto it</td><td> $- 0 . 1 1 4 \pm 0 . 1 8$ </td><td> $+ 0 . 5 0 7 \pm 0 . 2 0$ </td><td> $- 0 . 0 5 7 \pm 0 . 0 3$ </td></tr><tr><td>Green disc, off path at 5.2 cm, to the push onto it</td><td> $+ 0 . 0 1 8 \pm 0 . 1 5$ </td><td> $+ 0 . 4 5 6 \pm 0 . 1 7$ </td><td> $- 0 . 0 7 3 \pm 0 . 0 2$ </td></tr><tr><td>White marker, conflict position, to the push point</td><td> $- 0 . 5 3 8 \pm 0 . 0 2$ </td><td> $+ 0 . 2 6 6 \pm 0 . 3 3$ </td><td> $- 0 . 0 2 1 \pm 0 . 0 9$ </td></tr><tr><td></td><td> $- 0 . 6 6 9 \pm 0 . 0 2$ </td><td> $- 0 . 0 3 2 \pm 0 . 1 5$ </td><td></td></tr><tr><td>Green disc, conflict position, to the push point</td><td></td><td></td><td> $- 0 . 0 9 6 \pm 0 . 2 3$ </td></tr></table>

Results: is the proxy a target? The centre is the target where the GM agent pushes (Table 11, rollouts). On the separable layouts GM agents push the cube to the centre in 0.980 of episodes and onto the marker in none, whereas Non-GM agents push it onto the marker in 0.935 (Fig. 24a), and no GM episode reaches the decoupled disc, against 0.895 for Non-GM agents (Fig. 24d). Randomly initialized policies never reach the cube, since their gripper moves only 0.4 to 1.6 cm in 100 steps.

Results: what kind of place is the target? As in LOCATIONREACH, the target is a place in the camera image, neither an arm posture nor a point on the table. With the arm base shifted by 8 cm, GM agents still push the cube to the real centre in 0.967 of episodes and to the point the old posture implies in none. With the camera shifted by 5 cm, GM agents push the cube to the point that projects to the old centre pixel in 0.901 of episodes and to the real centre in 0.018, in every seed and although the marker lies visibly at the real centre, whereas Non-GM agents push it onto the marker in 0.955. The target moves with the image and not with the table, and the visible marker loses to it completely. Translating only the policy image by 8 pixels, about 4.6 cm on the table, with the scene unchanged, sends the cube to the pixel point in 0.923 of episodes and to the real centre in 0.135 (Fig. 23f to i, rates in j). That the target is anchored in the image and not in the arm’s posture is definitive, with effects near 0 against 1 in every seed.

(a) Drawn  
![](images/97d6c2dfb341a7d34b692e9e20677596d8b52ed3079cd8c76b04d5dfaa5da402.jpg)

(b) Hidden  
![](images/e9ea56519cc899a169bbd4946fbdf88f4bc489dde32077c7169f0063b66a1b8a.jpg)

(c) Conflict  
![](images/915c6f649e8772e2b9c69bf4b071d7201c260362ee0ec28bc967b09ba4c9322f.jpg)

(d) Decoupled  
![](images/06aa54b5920ed8bbf45a48b71369852f95980bf04f1e181953feac0dd2213a7c.jpg)

(e) Moved  
![](images/309d7b2447edeb4b6e02923cb31a7cebd56b29c1aa87689ea5798bfc3c77b5e9.jpg)

(f) Disc on path  
![](images/474cf15cdff55ddb92f4d0e1a4421a235375a3b8664e815faba73651ad77fdca.jpg)

(g) Disc off path  
![](images/ec8c0eebfd34bed8955c91d1e44380e13b67fdd074d6e1a8b8ea06c17099b44f.jpg)

(h) Hover  
![](images/90d3ba46b53d1c2b645cd5ccece21b76ec7408b256f81f9b2c556550c1cd6c92.jpg)  
GM cube path Non-GM cube path marker (goal) marker hidden green disc centre (proxy) cube

Figure 24: The PUSHTOMARKER evaluation configurations. (a to d) Rollout sets top-down in the camera orientation, with the cube paths of all three GM and Non-GM seeds on one separable layout with the marker drawn and hidden and on one conflict scene and its decoupled conflict, each chosen among scenes on which the three seeds of each cohort agree. Each pair of panels is centred on its scene at one common scale. (e to h) Rendered camera frames of the single-frame tests on that conflict scene, cropped to the scene: the marker move from its old place (dotted) to its new one, a green disc at the conflict position and off the path, and the gripper held above the cube.

Results: does the marker organize the whole episode? It does not. Drawing or hiding the marker moves the GM path by 0.050 cm, whereas the marker pulls the Non-GM path by 10.504 cm (Fig. 24a and b). On the conflict scenes GM agents sweep the cube across the marker and pause there no more often than when it is hidden (0.058 against 0.050, Non-GM 0.867, Fig. 24c). Whatever the marker does to the next step does not add up to a change of the episode.

Results: what is the undirected reaction to the marker? The marker move confirms that the reaction has no goal direction (Table 11, marker move). When the marker moves, the Non-GM step turns toward the new push point by +27.948 mm, about a tenth of the planned step, whereas the GM step turns by +0.487 mm, 0.002 of the planned step, and randomly initialized networks by 0.000 (Fig. 3c). The cosine to the shift of the push point is +0.439 for Non-GM and −0.041 for GM agents, at the level of init, and stays near zero for GM agents with the gripper held above the cube.

The disc control shows what the reaction is (Table 11, disc presence). At the conflict position the marker changes the commanded GM step by 13.385 mm on a planned step of about 12 cm, roughly half the Non-GM change, and a green disc changes it as much (14.189 mm), whereas the Non-GM step changes less for the green disc than for its marker. Off the path the green disc moves the GM step less the farther it lies from the cube, from 18.803 mm at 3 cm to 7.427 mm at 11 cm, and the white marker by a roughly constant 11 to 12 mm. Non-GM agents show the same two shapes at 1.4 to 3.3 times the size and init shows none, so the response is learned and not specific to misgeneralization. Only its direction separates the cohorts. With the gripper above the cube the Non-GM step turns toward the push that would move the cube onto a disc 5.2 cm off the path, of either colour, whereas the GM step does not, and a disc of either colour at the conflict position turns the GM step away from the push point.

The GM reaction to the marker is thus the reaction a disc of another colour gets, present in Non-GM agents and absent at init. It is a change in the scene rather than a goal, which validates the signedfollow result of the world edits. The marker reaches the GM policy, but not as something to push toward. A plausible reading, which we hold as a hypothesis, is that a disc near the cube perturbs the features through which the trunk locates the cube, in any agent that reads the cube, and that Non-GM agents additionally read the disc as a goal.

PUSHTOMARKER is therefore a target substitution with a place fixed in the camera image as the substitute target, definitive at the behavioral level with exact zeros on about 400 layouts in each seed. Its goal reaches the policy about as weakly as the coin in COIN, so how deep the goal reaches and what replaces it are separate properties.

## C.4.3 POLICY SHORTCUT: COIN

In COIN the proxy is a movement, running right to the end of the level, and the coin reaches the GM policy only weakly (Appendix C.2). We ask two questions. Is behavior organized by a target, the end ofthe level, or by an action rule? Is the coin’s weak influence directed?

Design. Both questions are chosen to separate COIN from the other environments. Its proxy is a movement, not a place or an object. A movement rule predicts that the agent keeps running when the end of the level is gone, whereas a substitute target, as in MAZE, predicts that it stops where the end used to be. The second question asks whether the coin, which reaches the policy only weakly, steers that movement at all, or whether its influence has no direction, as the marker’s in PUSHTOMARKER.

For the first question we replicate the end-wall test of Langosco et al. (2022), with one change. They make the end wall permeable, whereas we remove the native end wall on 60 held-out levels and extend the floor for 25 further columns, so the agent runs on open ground until it meets a new wall. We record whether an agent that reaches the old wall face continues more than 5 tiles past it, and roll out the same levels with the wall in place as a baseline.

For the second question we use the 600-state COIN conflict bank, with $A _ { \star } = A _ { \mathrm { l e f t } }$ the left-moving actions, which make immediate progress toward the coin in every evaluated state (Section B.4). We relocate the coin from the cell right of the agent to the cell left of it and, for each action group $A _ { g } ,$ divide its share of the total variation by the probability $\pi ( A _ { g } )$ it held before the edit. For A this ratio is $T _ { \star }$ , and $T _ { \star } = 1$ means the change has no preferred direction. We repeat this with the coin 1 to 4 cells left of the agent and compare with three randomly initialized networks and with logit changes drawn with no preferred direction. We also read the policy-Fisher input saliency $S ( x )$ of the coin cell and of the end-wall block (Definition 9), a Fisher-weighted gradient saliency over the whole action distribution that we read against a randomly initialized network and use only to suggest, not to establish, a reading (Appendix B.2). It is taken on the 589 conflict-bank states whose replay reproduces the bank’s agent pose and both coin cells, with the coin one cell left of the agent, and the end wall is on screen in 103 of them. Regions are exclusive, with the agent cell taking priority.

Results: target or action rule? An action rule organizes GM behavior. We replicate the result of Langosco et al. (2022). With the end wall removed, GM agents continue past the former endpoint in 47 of the 50 episodes that reach it (0.938 per checkpoint, where Langosco et al. (2022) report every one of 114 arrivals) and spend 3% of their post-arrival steps at the former wall face, against 100% with the wall in place. This rules out stopping at the former endpoint but not running toward a visible end wall, as every continuing agent stops at the next wall. Langosco et al. (2022) report both kinds of proxy in CoinRun, a directional one in the actor and a location one in the critic. We therefore use policy shortcut (after Geirhos et al., 2020) to describe the persistence of the learned rightward strategy without claiming a unique underlying destination representation.

Results: is the coin’s influence directed? It is not. Relocating the coin still changes the GM policy (total variation 0.080 against 0.280 for Non-GM), but it moves +0.000 of probability onto A , against +0.179 for Non-GM, and T is 1.210, against 4.996 for Non-GM and 0.991 for randomly initialized networks (Fig. 25a and b, and Fig. 3b). The GM change is spread across the action groups without favouring the coin side. Its ratio is near 1 for every group, slightly higher on jump (1.777, with a 95% interval from 0.192 to 3.363) and slightly lower on the remaining actions. Logit changes with no preferred direction give 1.01 for GM and 0.99 for Non-GM, so 1 is the right reference, and at every distance from 1 to 4 cells GM stays at $\Delta \pi ( A _ { \star } )$ between −0.001 and 0.003 with a ratio

![](images/49224488c8e304d88eaedd1bf726f4fd4251b7ef65ef8af7bb7b17c419d42bff.jpg)

(b)  
Share of change / prior mass  
![](images/a3b1af70c505badd53f6f46fb2a644c71d0bc4ec63b5a4fb7cf8d284467d5699.jpg)

(c)  
Change on A<sub>⋆</sub> vs distance  
![](images/32e731e7308f7ee0e7cc7adfb263e42f0e26ebaebba214235b57116dd9208436.jpg)

(d)  
Ratio for A <sub>⋆</sub> vs distance  
![](images/adff0197a9874c986763eb909e56cb3dae4b9635c329c81ee19bbc0089780197.jpg)  
GM Non-GM random init mean ± 95% CI reference

Figure 25: Where the coin’s influence goes in COIN. (a, b) Change in probability of each action group and its share of the total variation divided by its prior mass when the coin moves from the cell right of the agent to the cell left of it, on 593 of the 600 conflict-bank states, with groups $A _ { \star }$ (LEFT, LEFT+DOWN, LEFT+UP), A<sub>→</sub> (RIGHT, RIGHT+DOWN, RIGHT+UP), jump (UP) and other (DOWN, no-op and unbound keys). (c, d) The same two quantities for A<sub>⋆</sub> with the coin 1 to 4 cells left of the agent, on the 572 states where the coin fits and is visible at every distance.

between 1.15 and 1.39, while Non-GM moves 0.18 to 0.26 onto A with a ratio between 4.5 and 5.5 (Fig. 25c and d). The coin is registered as a change in the scene, but it redistributes probability in proportion to what each action already holds and is not connected to where the agent goes.

The saliency supports this reading but does not prove it (Fig. 26). The Non-GM decision depends on the coin (15.4× the image mean), whereas GM coin saliency (2.13×) stays at the level of a random network (2.78×), and no cohort’s decision depends on the end wall more than on an average pixel $( 0 . 8 4 \times \mathrm { G M } , 0 . 7 3 \times \mathrm { N o n } { \mathrm { - } } \mathrm { G M } )$

In our cohorts COIN is the only environment whose proxy is a movement, but it shares the depth of its goal wiring with PUSHTOMARKER, whose goal reaches the policy as weakly and without direction.

## C.4.4 CUE COMPETITION: ATTRIBUTE MAZE, ATTRIBUTETABLE AND SORTZONES

In these three environments the proxy is a property of the goal object, colour in all three, and the world edits show that GM policies respond to the intended cue as well as to the proxy cue (Appendix C.2). We ask two questions. Do both cues act as targets? Which cue wins, and how cleanly?

Design. In ATTRIBUTE MAZE we evaluate the 196 strict-conflict states used throughout the worldedit analyses, rendered on black and textured backgrounds, in which the line-associated and yellowassociated candidates require disjoint graph-exact progress actions, and apply the matched shape or colour swap of Section C.2. Each swap restores the yellow-line conjunction of training at one candidate, which it favours, and leaves the other as its competitor. ATTRIBUTETABLE and SORTZONES use the same two swaps with the red cube and the crossed green pad as their conjunctions. For edit $e ,$ let $A _ { \mathrm { f a v } } ^ { e } ( s )$ and $A _ { \mathrm { c o m p } } ^ { e } ( s )$ be the progress-action sets toward the favoured and the competing candidate, $A _ { 0 } ^ { e } ( s )$ the remaining neutral actions and $s _ { e }$ the edited state. Because policy probabilities are normalized, we compare each candidate group with the neutral reference,

$$
S _ { \mathrm { f a v } } ^ { e } ( s ) = \log \frac { \pi ( A _ { \mathrm { f a v } } ^ { e } ( s ) \mid s _ { e } ) } { \pi ( A _ { 0 } ^ { e } ( s ) \mid s _ { e } ) } - \log \frac { \pi ( A _ { \mathrm { f a v } } ^ { e } ( s ) \mid s ) } { \pi ( A _ { 0 } ^ { e } ( s ) \mid s ) } ,
$$

![](images/eca05ddae0bb2af23b863050308ba8d8ad2ee046b67f2a3b5b60f73dcbfebd70.jpg)  
Figure 26: Example saliency maps in COIN. Rows are the first three conflict-bank states whose replay reproduces the bank pose and that show the end wall, with no selection on saliency. Left: observation with the agent (white), the coin one cell to its left (yellow) and the end-wall block (red). Middle and right: $\log _ { 1 0 }$ saliency relative to the image mean (colour range −1 to 2) over the greyscale frame, for one GM and one Non-GM checkpoint.

and $S _ { \mathrm { c o m p } } ^ { e } ( s )$ likewise for the competitor. The joint pattern $S _ { \mathrm { f a v } } ^ { e } > 0 , S _ { \mathrm { c o m p } } ^ { e } < 0$ is the push– pull form of cue competition, and $\Delta \ell ^ { e } = S _ { \mathrm { f a v } } ^ { e } - S _ { \mathrm { c o m p } } ^ { e }$ is the resulting shift of relative preference between the two candidates, which does not require push–pull.

The arm policies are Gaussian and keep their progress toward the pursued object many standard deviations above zero, so these log-odds saturate in ATTRIBUTETABLE, and in SORTZONES start frames the neutral set holds all the probability. For the arm we therefore define the support of candidate c as

$$
\psi _ { c } ( s ) = \Phi ^ { - 1 } \bigl ( \pi ( z _ { c } > 0 \mid s ) \bigr ) = \frac { \mathbb { E } [ z _ { c } \mid s ] } { \operatorname { s d } ( z _ { c } \mid s ) } ,
$$

where $z _ { c }$ is the first-order displacement of the gripper along the direction to c that the action produces, toward the object in ATTRIBUTETABLE and along the push from the cube to the pad in SORTZONES. $S _ { \mathrm { f a v } } ^ { e }$ and $S _ { \mathrm { c o m p } } ^ { e }$ are the changes of ψ under the edit, which is linear in the action mean and needs no neutral reference, and the units therefore differ between environments. From the start pose the arm’s first steps can serve both candidates when the two objects lie in similar directions, so in ATTRIBUTETABLE we use the strict bank (Appendix E), 512 start frames with the two objects on opposite sides of the robot, at least $4 5 ^ { \circ }$ apart as seen from the hand (median 62<sup>◦</sup>) and equally far from it within 3 cm. In SORTZONES both pads lie in nearly the same direction from the retracted arm at every start frame, so we read the arm at the decision frame at which the hand reaches the cube before it has moved (500 states, half from GM and half from Non-GM rollouts), with every network evaluated on all of them. In both tasks a random shift of the action mean lands in the push– pull quadrant on a fraction of states that depends on the angle between the two candidate directions, which we report as a null rate.

Results: do both cues act as targets? In GM agents both do. Each swap pushes support toward the candidate it favours and pulls it from the other (Table 12, Fig. 27). In ATTRIBUTE MAZE every seed shifts relative preference toward the candidate favoured by the swap in every condition (24 of 24 seed means $\Delta \ell > 0 )$ , and the 95% interval excludes zero for both GM swaps on black, the GM colour swap on texture and the Non-GM shape swap on both backgrounds. It includes zero for the GM shape swap on texture (2.213, [−1.33, 5.76]), where one seed dominates, and for the Non-GM colour swap, as expected for a shape-selective policy. In ATTRIBUTE MAZE GM policies respond to both swaps, and in each case the edit increases support for its favoured candidate and suppresses its competitor on average (Fig. 27, top row). On the diversified bank of 480 conflict states, with the agent placed at every walkable cell, push–pull holds for both swaps in every GM seed and for shape only in Non-GM agents, across all 15 action-conflict classes, with effects about half the canonical size (GM black $\Delta \bar { \ell } = 1 . 8 2 3 \pm 0 . 2 4$ for shape and $2 . 2 9 3 \pm 0 . 2 9$ for colour). ATTRIBUTETABLE and SORTZONES reproduce this pattern (Fig. 27, middle row). In GM agents both swaps push and pull in every seed, whereas Non-GM agents push and pull for the goal swap and their colour swap leaves both candidates near zero $( - 0 . 1 1 4 \pm 0 . 7 0$ and $- 0 . 4 0 7 \pm 0 . 4 8$ in ATTRIBUTETABLE). The weak link in ATTRIBUTETABLE is the GM shape swap, which pushes $( 1 . 4 9 8 \pm 0 . 6 0 )$ but pulls only a little $( - 0 . 2 9 3 \pm 0 . 0 6$ , in every seed). At decision frames, the first frame of a rollout at which the

![](images/7f94cee9a5f96ff9f179e80b8e67b1eaa4b87710ca00ea1f4c2765b4cf8a5fac.jpg)  
Figure 27: Goal and proxy swaps in the four competition environments. Each environment has two panels, the goal swap (left) and the proxy swap (right), with $S _ { \mathrm { f a v } }$ and $S _ { \mathrm { c o m p } }$ in the units of Appendices C.4.4 and C.4.5. The top row shows ATTRIBUTE MAZE on black and textured backgrounds, the middle row ATTRIBUTETABLE and SORTZONES, and the bottom row WALKTOFLAG. Faint dots are states, diamonds seed means and open circles the cohort mean with 95% t-intervals across three seeds. Shading marks the push–pull quadrant. States beyond the axes are drawn at the edge, which clips 1% of states at each end (5% in WALKTOFLAG). ATTRIBUTETABLE is read on the strict bank and SORTZONES at decision frames.

Table 12: Cue competition in the four competition environments. Values are mean ± sample standard deviation across three seeds. Units differ between environments (Appendices C.4.4 and C.4.5). ATTRIBUTETABLE is read on the strict bank and SORTZONES at decision frames. $\Delta \ell = S _ { \mathrm { f a v } } - S _ { \mathrm { c o m p } }$ measures the resulting shift in relative target preference.
<table><tr><td>Environment</td><td>Cohort</td><td>Swap</td><td> $S _ { \mathrm { f a v } }$ </td><td> $S _ { \mathrm { c o m p } }$ </td><td>∆l</td></tr><tr><td rowspan="4">ATTRIBUTE MAZE black</td><td>GM</td><td>Shape</td><td> $1 . 3 6 8 \pm 0 . 2 5$ </td><td> $- 1 . 7 2 1 \pm 0 . 3 5$ </td><td> $3 . 0 8 9 \pm 0 . 6 0$ </td></tr><tr><td>GM</td><td>Colour</td><td> $2 . 0 0 1 \pm 0 . 1 9$ </td><td> $- 2 . 3 5 4 \pm 0 . 2 9$ </td><td> $4 . 3 5 5 \pm 0 . 4 8$ </td></tr><tr><td>Non-GM</td><td>Shape</td><td> $3 . 7 4 0 \pm 0 . 3 7$ </td><td> $- 3 . 9 1 0 \pm 0 . 2 1$ </td><td> $7 . 6 5 0 \pm 0 . 5 7$ </td></tr><tr><td>Non-GM</td><td>Colour</td><td> $0 . 1 9 2 \pm 0 . 1 1$ </td><td> $- 0 . 3 6 2 \pm 0 . 2 7$ </td><td> $0 . 5 5 5 \pm 0 . 3 8$ </td></tr><tr><td rowspan="4">ATTRIBUTE MAZE texture</td><td>GM</td><td>Shape</td><td> $1 . 0 3 7 \pm 0 . 6 6$ </td><td> $- 1 . 1 7 6 \pm 0 . 7 7$ </td><td> $2 . 2 1 3 \pm 1 . 4 3$ </td></tr><tr><td>GM</td><td>Colour</td><td> $3 . 0 5 3 \pm 0 . 4 6$ </td><td> $- 3 . 1 8 9 \pm 0 . 4 0$ </td><td> $6 . 2 4 2 \pm 0 . 8 6$ </td></tr><tr><td>Non-GM</td><td>Shape</td><td> $3 . 3 3 0 \pm 0 . 3 1$ </td><td> $- 3 . 6 2 3 \pm 0 . 2 5$ </td><td> $6 . 9 5 3 \pm 0 . 5 4$ </td></tr><tr><td>Non-GM</td><td>Colour</td><td> $0 . 4 1 0 \pm 0 . 2 5$ </td><td> $- 0 . 7 1 4 \pm 0 . 4 0$ </td><td> $1 . 1 2 3 \pm 0 . 6 5$ </td></tr><tr><td rowspan="4">ATTRIBUTETABLE</td><td>GM</td><td>Shape</td><td> $1 . 4 9 8 \pm 0 . 6 0$ </td><td> $- 0 . 2 9 3 \pm 0 . 0 6$ </td><td> $1 . 7 9 1 \pm 0 . 5 5$ </td></tr><tr><td>GM</td><td>Colour</td><td> $4 . 1 3 3 \pm 0 . 4 3$ </td><td> $- 2 . 6 1 1 \pm 0 . 1 5$ </td><td> $6 . 7 4 4 \pm 0 . 2 8$ </td></tr><tr><td>Non-GM</td><td>Shape</td><td> $2 . 1 8 7 \pm 0 . 1 9$ </td><td> $- 2 . 8 5 8 \pm 0 . 4 5$ </td><td> $5 . 0 4 5 \pm 0 . 2 7$ </td></tr><tr><td>Non-GM</td><td>Colour</td><td> $- 0 . 1 1 4 \pm 0 . 7 0$ </td><td> $- 0 . 4 0 7 \pm 0 . 4 8$ </td><td> $0 . 2 9 3 \pm 0 . 9 3$ </td></tr><tr><td rowspan="4">SORTZONES</td><td>GM</td><td>Cross</td><td> $0 . 8 3 3 \pm 0 . 2 8$ </td><td> $- 0 . 7 9 7 \pm 0 . 2 9$ </td><td> $1 . 6 3 0 \pm 0 . 5 7$ </td></tr><tr><td>GM</td><td>Colour</td><td> $5 . 6 5 8 \pm 0 . 8 2$ </td><td> $- 5 . 6 6 1 \pm 0 . 7 6$ </td><td> $1 1 . 3 1 9 \pm 1 . 5 8$ </td></tr><tr><td>Non-GM</td><td>Cross</td><td> $2 . 8 6 9 \pm 1 . 8 5$ </td><td> $- 2 . 9 4 0 \pm 1 . 8 8$ </td><td> $5 . 8 0 8 \pm 3 . 7 4$ </td></tr><tr><td>Non-GM</td><td>Colour</td><td> $- 0 . 1 3 9 \pm 0 . 3 3$ </td><td> $0 . 1 6 1 \pm 0 . 2 9$ </td><td> $- 0 . 3 0 0 \pm 0 . 6 1$ </td></tr><tr><td rowspan="4">WALKTOFLAG</td><td>GM</td><td>Flag</td><td> $1 . 7 8 5 \pm 0 . 2 9$ </td><td> $0 . 0 0 5 \pm 0 . 7 8$ </td><td> $1 . 7 8 0 \pm 0 . 5 9$ </td></tr><tr><td>GM</td><td>Tower</td><td> $3 . 9 3 3 \pm 4 . 0 1$ </td><td> $- 4 . 1 2 9 \pm 3 . 0 2$ </td><td> $8 . 0 6 2 \pm 4 . 1 4$ </td></tr><tr><td>Non-GM</td><td>Flag</td><td> $0 . 3 4 1 \pm 1 . 1 0$ </td><td> $- 1 . 8 8 8 \pm 1 . 3 5$ </td><td> $2 . 2 2 8 \pm 1 . 0 6$ </td></tr><tr><td>Non-GM</td><td>Tower</td><td> $- 1 . 8 8 2 \pm 1 . 1 8$ </td><td> $- 1 . 2 4 3 \pm 1 . 2 6$ </td><td> $- 0 . 6 3 9 \pm 0 . 2 6$ </td></tr></table>

hand lies between the two objects at least 3 cm from both (444 states), the same pattern holds with a stronger shape pull $( - 0 . 8 1 7 \overset { \cdot } { \pm } 0 . 1 1 )$ .

Results: which cue wins, and how cleanly? In GM agents the proxy cue wins, in Non-GM agents the goal cue. Non-GM policies are strongly shape-selective, with $\Delta \ell = 7 . 6 5 0$ for the shape swap and 0.555 for the colour swap on black, whereas in GM policies the colour swap shifts preference more than the shape swap (3.089 and 4.355 on black, 2.213 and 6.242 on texture, $1 . 7 9 1 \dot { \pm } 0 . 5 5$ and $6 . 7 4 4 \pm 0 . 2 8$ in ATTRIBUTETABLE, $1 . 6 3 0 \pm 0 . 5 7$ and $1 1 . 3 1 9 \pm 1 . 5 8$ in SORTZONES). How cleanly one cue’s gain is the other’s loss shows in the spread of the states, which differs between environments and is informative about the geometry of the test rather than about the agents. In ATTRIBUTE MAZE and SORTZONES the states lie along the falling diagonal from top left to bottom right, so a swap that raises support for one candidate lowers it for the other (Spearman correlation of $S _ { \mathrm { f a v } }$ and $S _ { \mathrm { c o m p } }$ across states −0.53 to −0.85 per cohort and swap in ATTRIBUTE MAZE, −0.95 to −0.98 in SORTZONES). In ATTRIBUTE MAZE the candidate action sets are disjoint and measured against the same neutral actions, and at SORTZONES decision frames the two pushes point in nearly opposite directions, so any shift of the action mean toward one candidate is a shift away from the other. AT-TRIBUTETABLE forms a cloud with a weak positive correlation (+0.29 to +0.61), because objects at least $4 5 ^ { \circ }$ apart (median 62<sup>◦</sup>) still share part of their direction, and a shift of the gripper command can raise or lower both supports together. In ATTRIBUTETABLE a random shift of the action mean therefore lands in the quadrant in 0.18 of states, against 0.45 at SORTZONES decision frames, where the informative result is which swap moves the policy, not that it lands in the quadrant. We do not read the larger colour response of GM policies as a direct estimate of a larger internal colour weight, because the size of a finite swap also depends on the pre-edit preference and the headroom for redistribution. The swaps are also world edits on the shape–colour conflict rather than an identification of independent cue channels, since each swap resolves the conflict by restoring the familiar conjunction at one candidate, and Ramanauskas & S¸ims¸ek (2023) attribute the colour preference in ATTRIBUTE MAZE to goal detection through a single colour channel, so the shape and colour cues we manipulate need not correspond to separate internal features. What the experiment establishes in all three environments is that the two matched swaps can each reorganize target preference in GM policies, whereas Non-GM policies are controlled by the goal cue. We refer to this regime as cue competition.

## C.4.5 OBJECT COMPETITION: WALKTOFLAG

In WALKTOFLAG the proxy is not a property of the goal object but a second object, a 2 m orange tower that stands beside the small flag in GM training (Appendix A.3.1). The tower is also far easier to see. At 3 m it covers about 290 image pixels and the flag banner only 1 to 2. So far this environment has behaved much like cue competition within one object, but here there are two objects, so we ask two questions. Do both objects pull the agent? And how does this competition differfrom cue competition?

Design. For the first question we remove each object from the scene and measure how the walking command moves. The policy is Gaussian, $\pi ( \cdot \mid \dot { s } ) = \mathcal { N } \big ( \mu ( s ) , \mathrm { d i a g } ( \sigma ^ { 2 } ) \big )$ , and we call $\mu ( s )$ , the forward, sideways and turning velocity that the policy commands on average, the command mean. Let $s _ { \backslash o }$ be the state s rendered without object $^ { O , }$ the flag ⋆ or the tower $p ,$ written like the edited state $s _ { e }$ of a world edit. The pull vector of o is the shift of the command mean that the presence of o causes, in policy standard deviations,

$$
u _ { o } ( s ) = \mathrm { d i a g } ( \sigma ) ^ { - 1 } \big ( \mu ( s ) - \mu ( s \setminus o ) \big ) .
$$

To give it a direction we use the command $c _ { o } ( s )$ of a heading controller that walks to $^ { O , }$ which turns toward the bearing of o and walks forward once roughly aligned, and the unit vector from the command mean without o toward that command,

$$
\hat { c } _ { o } ( s ) = \frac { w _ { o } ( s ) } { \| w _ { o } ( s ) \| } , \qquad w _ { o } ( s ) = \mathrm { d i a g } ( \sigma ) ^ { - 1 } \big ( c _ { o } ( s ) - \mu ( s _ { \setminus o } ) \big ) .
$$

The pull of o is the length of its pull vector along this direction,

$$
f _ { o } ( s ) = u _ { o } ( s ) ^ { \top } \hat { c } _ { o } ( s ) ,
$$

written $f _ { \star }$ for the flag and $f _ { p }$ for the tower. It is positive when o moves the command toward itself and negative when o moves it away. A state counts as pulled toward o when $f _ { o } ( s ) > \tau$ and pulled away from o when $f _ { o } ( s ) < - \tau$ . The threshold $\tau$ is set for each network as the 95th percentile of the same projection, in absolute value, for two nuisance shifts that should not matter, moving a box and re-rendering the unchanged scene, and is at least 0.1. The cross projection $u _ { o } ( s ) ^ { \top } \hat { c } _ { o ^ { \prime } } ( \bar { s } )$ , with $\hat { c } _ { o ^ { \prime } }$ the unit vector toward the command that walks to the other object $\mathbf { \bar { \Sigma } } _ { o ^ { \prime } }$ from the same command mean $\mu ( s _ { \backslash o } )$ , shows whether one object pushes the agent away from the other. We take these quantities on 1,024 states per seed from three banks (Appendix E), the 512 states of the world-edit bank, where flag and tower stand near the robot and in view, the 256 forced-view conflict states, where the flag is 1 to 4.5 m nearer than the tower, and the 256 states of the far bank, where the tower is 1 to 4.5 m nearer than the flag. We write $d _ { \star }$ and $d _ { p }$ for the distances from the robot to the flag and to the tower.

For the second question we use the matched swaps of Section C.2, which have the same structure as in cue competition (Appendix C.4.4). The flag swap moves the flag beside the tower and favours the tower’s place, and the tower swap moves the tower beside the flag and favours the flag’s place, so each restores the training pair at one candidate place and leaves the other as competitor. The walking command sets a planar direction, so $A _ { \mathrm { f a v } }$ and $A _ { \mathrm { { c o m p } } }$ are the walking directions within 30<sup>◦</sup> of the bearing to the favoured and to the competing candidate, $A _ { 0 }$ holds all other directions, and $S _ { \mathrm { f a v } }$ and $S _ { \mathrm { c o m p } }$ are defined as for cue competition, on the 512 start states of the world-edit bank. To ask whether the agent simply follows the nearer object, we compare the sizes of the two pulls against the distance gap $d _ { \star } - d _ { p }$ . For this we project the same pull vector on the heading-controller command toward $^ { O , }$ measured from the average command toward neutral bearings, which gives the removal pull

$$
P _ { o } ( s ) = u _ { o } ( s ) ^ { \top } \mathrm { d i a g } ( \sigma ) ^ { - 1 } \big ( c _ { o } ( s ) - \bar { c } _ { 0 } ( s ) \big ) ,
$$

where $\bar { c } _ { 0 } ( s )$ is the mean heading-controller command over the 24 evenly spaced bearings at least $3 0 ^ { \circ }$ from both objects. For a Gaussian policy $P _ { o }$ equals the gain in log density that the presence of o gives the command toward $^ { O , }$ relative to the commands toward neutral bearings, so unlike $f _ { o }$ it grows with how far the command toward o lies from a neutral command. We bin the states by $d _ { \star } - d _ { p }$ , average $P _ { \star }$ and $P _ { p }$ in each bin and seed, set negative bin means to zero, and report the tower share $\bar { P } _ { p } / ( P _ { p } + P _ { \star } )$ . Walking to the nearer object would give a share of 0 where the flag is nearer and 1 where the tower is nearer.

![](images/0a2a85dd286619f030f22a87068861c8c5c157ff5471a1b2b1c6c73643e33111.jpg)

![](images/9b49c8dbabeadc6a405c5851baf6aa15e65de0467ea0f2c97e9637b42935b2b3.jpg)

(c) Both pull at once  
![](images/62f0eac71bd2baed18747f1f6288dc312e5e3e20cd4401db4842b9055e495c50.jpg)

(d) Tower wins even when farther  
![](images/0dabdaf0e1b99937c61f3ec89a7b9a7e130d626f5aab04cac7d5d9658753ac4b.jpg)  
flag (f<sub>⋆</sub> ) tower (f<sub>p</sub>) GM Non-GM init nuisance null mean ± 95% CI  
Figure 28: Object competition in WALKTOFLAG. (a) Mean pull $f _ { \star }$ of the flag and $f _ { p }$ of the tower, the shift of the command mean caused by each object along the direction to the command that walks to it, in policy standard deviations, with dots for seeds. (b) Fraction of GM states pulled toward each object $( f _ { o } > \tau )$ in four groups, the world-edit bank split into states with the flag or the tower more than 0.3 m nearer (near), the forced-view conflict bank (flag much nearer) and the far bank (tower much nearer). (c) Fraction of states pulled toward the flag, away from it, toward the tower and toward both at once, and the fraction on which both nuisance shifts pass the threshold together (null). (d) Tower share $P _ { p } / ( P _ { p } + P _ { \star } )$ of the removal pull against the distance gap $d _ { \star } - d _ { p } ,$ negative where the flag is nearer. The dotted step is the share that walking to the nearer object would give. Means with 95% t-intervals across three seeds.

Results: do both objects pull the agent? Both objects pull GM agents, the tower strongly and the flag weakly and with mixed sign. In GM agents the tower pull is $f _ { p } = 3 . 7 3 6$ policy standard deviations, whereas the flag pull is $f _ { \star } = 0 . 2 1 1 ( \mathrm { F i g . } 2 8 \mathrm { a } )$ . The tower pulls toward itself on 0.793 of states, and the flag pulls toward itself on 0.368 and away from itself on 0.198 (Fig. 28c). Non-GM agents show the reverse, $f _ { \star } = 2 . 8 9 9$ with a pull toward the flag on 0.899 of states and $f _ { p } = - 0 . 0 1 9$ and randomly initialized networks show no pull. The GM flag pull lives near the flag (Fig. 28b). It points toward the flag on 0.449 to 0.509 of states where both objects are near and on 0.458 where the flag is much nearer, but on 0.021 where the tower is much nearer, while the tower pulls toward itself on 0.568 to 0.948 of states in every group. The two pulls also coexist on the same states. Both point toward their own object on 0.271 of GM states, against 0.003 for the two nuisance shifts (Fig. 28c). Neither object pushes the agent away from the other. The cross projection of the GM flag pull on the direction to the tower is −0.293 (95% CI −0.730 to 0.144), and that of the tower pull on the direction to the flag is +2.177 (0.342 to 4.012), positive because the two objects often lie in similar directions.

Results: how does this differ from cue competition? Unlike cue competition, the two objects do not trade support against each other. They pull at the same time, and the tower wins at every distance. The swaps show no push and pull. The states lie along the rising diagonal, the inverse of the other environments (Spearman correlation $+ 0 . 6 5 ~ { \mathrm { t o } } ~ { + 0 . 8 1 }$ per cohort and swap, Fig. 27, bottom row), because flag and tower stand close together in the world-edit bank, so a turn toward one candidate is mostly a turn toward the other and a swap raises or lowers both supports together. In the seed means (Table 12) the GM tower swap lies in the push–pull quadrant $( S _ { \mathrm { f a v } } = 3 . 9 3 3 , S _ { \mathrm { c o m p } } = - 4 . 1 2 9 )$

with a spread across seeds as large as the effect, and the GM flag swap raises support for the place where the flag joins the tower (1.785) without taking support from the place it leaves (0.005). In Non-GM agents the flag swap mainly takes support from the place the flag leaves (−1.888), and the tower swap lowers both candidates (−1.882 and −1.243). The tower swap result holds with cones of $1 5 ^ { \circ }$ and 45<sup>◦</sup> and when the cones are replaced by the heading-controller commands toward the two candidates, whereas the GM flag swap lies in the quadrant in all three seeds at $1 5 ^ { \circ }$ but in one seed at $3 0 ^ { \circ }$ , at $4 5 ^ { \circ }$ and with the heading-controller commands. The tower’s dominance does not come from being nearer (Fig. 28d). It takes 0.775 to 0.861 of the GM removal pull where the flag is 1 to 5 m nearer and 0.976 to 0.986 where the tower is more than 1 m nearer, against at most 0.008 in Non-GM agents. This agrees with panel b, which shows the flag pulling on about half of the states where it is nearer. Panel b counts states on which a pull passes the nuisance threshold, whereas panel d compares the sizes of the two pulls, and where the flag is much nearer the GM flag pull passes the threshold on 0.458 of states but averages $f _ { \star } = 0 . 2 4 1$ against $f _ { p } = 1 . 7 5 0$ for the tower. The flag pulls often but weakly, and the tower’s larger pull dominates the share.

GM behavior in WALKTOFLAG is therefore an object competition, in which both objects pull at once without pushing the agent away from each other, the orange tower dominates at every distance and the flag keeps a weak pull only when it is near.

The next section asks how this target-level interaction is organized in representation space.

## C.5 GEOMETRY OF CUE COMPETITION

We found that the goal and proxy cues both remain active in ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES, and so do the two competing objects in WALKTOFLAG, and that changing either cue shifts the target preference toward the candidate that cue supports. The preceding sections show different variants of this competition. What remains open is what the competition looks like inside the representation.

We therefore ask two questions. First, when the intended and proxy cues are changedfrom the same base state, how are the resulting representation displacements related? Second, once we define their contrast, is the resulting competition direction reused across states?

## C.5.1 HOW ARE THE TWO CUE DIRECTIONS ORGANISED WITHIN A STATE?

The edits are those of the world-edit analysis (Appendix C.2). Because the analysis compares edits across states, it needs states in which it is clear on which side the goal candidate lies and the choice between the candidates is still open, so ATTRIBUTETABLE and SORTZONES use stricter conflict banks than their world edits, in which the two candidates lie on clearly different sides at equal distance from the hand (ATTRIBUTETABLE) or the cube (SORTZONES), both conflict orientations occur equally often, and in SORTZONES the frame is taken when the hand has reached the cube but not yet pushed it (Appendix E.2). For each state $s \in S _ { \mathrm { c o n f } }$ , the intended-cue swap induces the representation displacement $\Delta _ { \star } ( s )$ and the proxy-cue swap induces $\Delta _ { p } ( s )$ . The swaps are shape and colour in ATTRIBUTE MAZE and ATTRIBUTETABLE, cross mark and colour in SORTZONES, and flag and tower in WALKTOFLAG. Both displacements are measured relative to the same unedited representation. Everything except the swapped cue stays fixed, so the comparison shows how the two cue edits move the representation from the same starting point.

To ask whether the two cue edits induce related representation changes, we first compare the orientation of their displacement vectors. In ordinary activation space, their alignment is measured by the Euclidean cosine

$$
\mathrm { c o s } _ { E } ( \Delta _ { \star } ( s ) , \Delta _ { p } ( s ) ) = \frac { \Delta _ { \star } ( s ) ^ { \top } \Delta _ { p } ( s ) } { \| \Delta _ { \star } ( s ) \| _ { 2 } \| \Delta _ { p } ( s ) \| _ { 2 } } .
$$

Positive values indicate aligned representation changes, values near zero indicate weak alignment, and negative values indicate opposing directions.

The Euclidean cosine treats every activation-space direction equally. We are additionally interested in whether the two edits oppose each other specifically along directions to which the downstream policy is sensitive. Because both displacements originate from the same base state, they share the same local Fisher metric. We therefore also compute the Fisher cosine from Definition 7,

$$
\cos _ { F } ( \Delta _ { \star } ( s ) , \Delta _ { p } ( s ) ) .
$$

![](images/5daaa6cdb81575f63f452e6791b86131584b6f1fb570a3a51ccce8fc95bdb6d6.jpg)  
Figure 29: Within-state cue geometry in the four competition environments $( h _ { \mathrm { f c } } )$ . Euclidean cosine $\mathrm { c o s } _ { E }$ and Fisher cosine cos between $\Delta _ { \star } ( s )$ and $\Delta _ { p } ( s )$ , averaged over states. In ATTRIBUTE MAZE circles are the black-background policies and diamonds the textured-background policies, each on its own background. Mean and 95% t-interval over three seeds, small points are seeds.

So cos measures how similar the two displacements are as raw activation changes, and cos measures how similar the changes in the action distribution are that they cause. In ATTRIBUTE MAZE the metric is that of a choice among 15 discrete actions, and in the robot tasks that of a Gaussian action head, which reads all eight action dimensions of the arm and the three of WALKTOFLAG.

We report both quantities at $h _ { \mathrm { f c } } .$ , the representation the policy and value heads read, with ATTRIBUTE MAZE also at $h _ { \mathrm { p r e } }$ in Table 13 and Figure 30. ATTRIBUTE MAZE refers to the black-background policies, and the textured-background policies are shown as diamonds in Figure 29, in the right half of Figure 30 and in their own rows of Table 13. Freshly initialized networks serve as the baseline.

Results. In GM agents the two cues are opposed within a state to very different degrees (Figure 29). The Fisher cosine $\mathrm { c o s } _ { F } ,$ which compares the changes in the action distribution, is −0.919 in ATTRIBUTE MAZE, −0.435 in SORTZONES, −0.160 in WALKTOFLAG and −0.002 in AT TRIBUTETABLE. So only ATTRIBUTE MAZE is close to antipodal, and the robot tasks form a weaker gradient below it. The SORTZONES opposition holds in every GM policy, while the AT-TRIBUTETABLE mean near zero averages policies that differ in sign. The textured ATTRIBUTE MAZE policies are opposed almost as strongly as the black ones $( \cos _ { F } = - 0 . 8 3 8$ in GM and −0.687 in Non-GM agents).

Initialized networks may explain part of the gradient (grey points in Figure 29). Both swaps redraw the same two objects, so in an untrained network the two displacements already share a component, with $\cos _ { F } = 0 . 2 5 4$ in ATTRIBUTE MAZE and 0.393 in ATTRIBUTETABLE. Training in GM agents moves cos<sub>F</sub> away from this starting point by 1.173 in ATTRIBUTE MAZE, 0.583 in SORTZONES, 0.395 in ATTRIBUTETABLE and 0.113 in WALKTOFLAG. So ATTRIBUTETABLE starts from the largest shared component and GM training removes only a third of what it removes in ATTRIBUTE MAZE.

The action space explains most of the rest of the ATTRIBUTETABLE result. The first-order gripper displacements that the two edits cause point in opposite directions in every GM agent (cosine of the two 3-d gripper motions −0.590±0.11, against 0.393 in Non-GM agents and 0.263 at initialization), so the two cues do push the gripper toward opposite objects. But the arm has seven joints, and a large part of the Fisher energy of each edit lies in joint directions that leave the gripper in place (0.470 for the shape swap and 0.297 for the colour swap). In those directions the two edits move the policy the same way (cos $_ F = 0 . 4 0 2 \pm 0 . 2 9 )$ , and the full cos<sub>F</sub> averages the opposed gripper component with this aligned one. ATTRIBUTE MAZE has no such directions, since every action changes where the agent goes. In SORTZONES the gripper displacements are opposed as well $( - 0 . 7 3 0 \pm 0 . 0 6 )$ , and so are the two edits in the directions that leave the gripper in place $( \mathrm { c o s } _ { F } = - 0 . 5 0 6 \pm 0 . 1 6 )$ , so nothing cancels the opposition.

Within-state opposition is specific to GM agents in the robot tasks. In ATTRIBUTE MAZE Non-GM agents are opposed as well $( \cos _ { F } = - 0 . 5 5 7 )$ , and single-cue ablations, which remove one cue instead of exchanging it and share no edited sprite, leave GM agents no more opposed than

Non-GM agents (−0.70 at $h _ { \mathrm { f c } }$ for GM). In the robot tasks Non-GM agents are aligned on average $( \cos _ { F } = 0 . 5 7 9$ in ATTRIBUTETABLE, 0.147 in SORTZONES and 0.488 in WALKTOFLAG). A Non-GM agent barely reads the proxy, so its proxy displacement keeps mostly the shared component that initialization already has and never acquires an opposed part. Within-state opposition is therefore a property of trained policies in ATTRIBUTE MAZE, but a GM signature in the robot tasks.

Within a state, then, the two cue edits move the representation in clearly different directions from the same starting point, opposed in ATTRIBUTE MAZE and in the GM agents of SORTZONES, and their contrast is what the policy has to resolve. This motivates a competition direction defined separately for every state,

$$
d _ { \mathrm { c o m p } } ( s ) = \Delta _ { p } ( s ) - \Delta _ { \star } ( s ) = h ( s _ { p } ) - h ( s _ { \star } ) .
$$

Because both displacements are measured from the same base representation, the base representation cancels in the difference. The resulting vector is therefore the direct representation-space contrast between the two competing cue assignments at state s. Its positive orientation points from the swap under which both cues support the proxy’s candidate toward the swap under which both cues support the goal’s candidate. So, pushing along $+ d _ { \mathrm { c o m p } } ( s )$ moves the representation toward the intended goal. What remains open is whether this direction is specific to each state or shared across states.

## C.5.2 IS THE COMPETITION DIRECTION SHARED ACROSS STATES?

The within-state analysis gives us one competition direction $d _ { \mathrm { c o m p } } ( s )$ for every conflict state. We now change the unit of comparison and ask how these directions relate across states.

Two possible organizations. The simplest possibility is that the same orientation is reused everywhere,

$$
d _ { \mathrm { c o m p } } ( s ) \approx \beta ( s ) u _ { \mathrm { c o m p } } , \qquad \beta ( s ) > 0 .
$$

A slightly richer possibility is that the same underlying axis is reused while its orientation changes with the state,

$$
d _ { \mathrm { c o m p } } ( s ) \approx \sigma ( s ) \beta ( s ) u _ { \mathrm { c o m p } } , \qquad \sigma ( s ) \in \{ - 1 , + 1 \} .
$$

Under this account, different states can express opposite orientations of the same competition axis. The sign $\sigma ( s )$ is the conflict orientation of the state, and it is set by the scene, not read from the network. Every conflict has a mirrored version in which the goal candidate lies on the other side. In ATTRIBUTE MAZE the agent always starts at the same cell and the line candidate lies either along UP or along RIGHT (98 states each), in ATTRIBUTETABLE and SORTZONES the goal candidate lies left or right of the robot, and in WALKTOFLAG the flag appears left or right in the camera image. We set $\sigma ( s ) = + 1$ for one side and $\sigma ( s ) = - 1$ for the other. The sign matters because $d _ { \mathrm { c o m p } } ( s )$ points toward the goal’s candidate, so mirrored states call for opposite actions, for example UP instead of RIGHT. Averaged without the sign, the two orientations cancel, whereas $\sigma ( s ) d _ { \mathrm { c o m p } } ( s )$ folds them onto one direction. Since $\sigma$ is known by design, the question is whether independently generated cue edits collapse onto one signed axis in the part of the representation the policy reads.

Comparing competition directions across states. For two states $s _ { i } , s _ { j } \in S _ { \mathrm { c o n f } }$ , we first compare their competition directions in ordinary activation space,

$$
\kappa _ { E } \big ( s _ { i } , s _ { j } \big ) = \cos _ { E } \big ( d _ { \mathrm { c o m p } } ( s _ { i } ) , d _ { \mathrm { c o m p } } ( s _ { j } ) \big ) .
$$

For the policy-sensitive comparison, the two states generally have different local Fisher metrics. We therefore use their pair-average metric $\begin{array} { r } { F _ { i j } = \frac 1 2 ( F _ { i } + F _ { j } ) } \end{array}$ and define

$$
\kappa _ { F } ( s _ { i } , s _ { j } ) = \frac { d _ { i } ^ { \top } F _ { i j } d _ { j } } { \sqrt { ( d _ { i } ^ { \top } F _ { i j } d _ { i } ) ( d _ { j } ^ { \top } F _ { i j } d _ { j } ) } } ,
$$

where $d _ { i } = d _ { \mathrm { c o m p } } ( s _ { i } )$ and $d _ { j } = d _ { \mathrm { c o m p } } ( s _ { j } )$

We summarize both similarities over pairs of states with the same orientation. We fit $u _ { \mathrm { c o m p } }$ as the mean of $\sigma ( s ) d _ { \mathrm { c o m p } } ( s )$ over the fit states (146 in ATTRIBUTE MAZE, 308 in ATTRIBUTETABLE, 302 in SORTZONES and 256 in WALKTOFLAG) and evaluate its alignment with $\sigma ( s ) d _ { \mathrm { c o m p } } ( s )$ on the held-out states (50, 204, 198 and 200).

Table 13: Cross-state competition geometry. The first two columns give mean similarity over pairs of states with the same sign $\sigma .$ The last two give the alignment of held-out competition directions with the shared axis $u _ { \mathrm { c o m p } }$ fit on the remaining states. Entries are cohort mean $\pm \mathrm { \ S D }$ across three policies. $\kappa _ { E }$ and cos<sub>E</sub> compare raw activations, $\kappa _ { F }$ and cos<sub>F</sub> compare them in the policy’s Fisher metric. ATTRIBUTE MAZE has two blocks with the same rows, the black-background and the textured-background policies, each on its own background at $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ (initialized networks shown on the same background). Robots at $h _ { \mathrm { f c } }$ on the banks of Appendix E.2. The held-out cos<sub>F</sub> uses the Fisher metric at each held-out state.
<table><tr><td colspan="5">Environment</td><td rowspan="2"> $\cos _ { E } ( \sigma d _ { \mathrm { c o m p } } ,$   $u _ { \mathrm { c o m p } } )$ </td><td rowspan="2"> $\cos _ { F } ( \sigma d _ { \mathrm { c o m p } } ,$  ucomp)</td></tr><tr><td>ATTRIBUTE MAZE</td><td>Cohort</td><td>Position</td><td> $\kappa _ { E } ~ \mathrm { s a m e }$ </td><td>κF same</td></tr><tr><td rowspan="7">black</td><td>GM</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 3 5 4 \pm 0 . 0 2$ </td><td> $0 . 9 3 8 \pm 0 . 0 4$ </td><td> $0 . 5 2 1 \pm 0 . 0 2$ </td><td> $0 . 9 6 1 \pm 0 . 0 3$ </td></tr><tr><td>GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 5 8 9 \pm 0 . 0 2$ </td><td> $0 . 9 4 2 \pm 0 . 0 5$ </td><td> $0 . 7 0 9 \pm 0 . 0 0$ </td><td> $0 . 9 5 9 \pm 0 . 0 3$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 2 8 4 \pm 0 . 0 2$ </td><td> $0 . 8 4 4 \pm 0 . 0 5$ </td><td> $0 . 4 6 8 \pm 0 . 0 2$ </td><td> $0 . 9 1 4 \pm 0 . 0 5$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 5 3 2 \pm 0 . 0 4$ </td><td> $0 . 9 5 2 \pm 0 . 0 2$ </td><td> $0 . 6 7 9 \pm 0 . 0 5$ </td><td> $0 . 9 7 0 \pm 0 . 0 2$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 1 7 3 \pm 0 . 0 2$ </td><td> $0 . 1 6 0 \pm 0 . 0 3$ </td><td> $0 . 3 5 7 \pm 0 . 0 3$ </td><td> $0 . 3 3 0 \pm 0 . 0 6$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 1 7 7 \pm 0 . 0 3$ </td><td> $0 . 1 4 8 \pm 0 . 0 3$ </td><td> $0 . 3 6 4 \pm 0 . 0 3$ </td><td> $0 . 3 0 1 \pm 0 . 0 6$ </td></tr><tr><td>ATTRIBUTE MAZE GM</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 2 6 5 \pm 0 . 0 3$ </td><td> $0 . 9 1 3 \pm 0 . 0 6$ </td><td> $0 . 4 4 7 \pm 0 . 0 3$ </td><td> $0 . 9 4 3 \pm 0 . 0 5$ </td></tr><tr><td rowspan="6">textured</td><td>GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 5 8 8 \pm 0 . 0 6$ </td><td> $0 . 9 1 8 \pm 0 . 0 7$ </td><td> $0 . 7 0 9 \pm 0 . 0 5$ </td><td> $0 . 9 4 6 \pm 0 . 0 5$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 2 4 3 \pm 0 . 0 4$ </td><td> $0 . 8 2 0 \pm 0 . 0 4$ </td><td> $0 . 4 2 6 \pm 0 . 0 4$ </td><td> $0 . 9 3 1 \pm 0 . 0 5$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 4 5 8 \pm 0 . 0 2$ </td><td> $0 . 9 0 0 \pm 0 . 0 4$ </td><td> $0 . 6 2 6 \pm 0 . 0 2$ </td><td> $0 . 9 3 2 \pm 0 . 0 4$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { p r e } }$ </td><td> $0 . 1 0 5 \pm 0 . 0 5$ </td><td> $0 . 0 7 3 \pm 0 . 0 2$ </td><td> $0 . 2 6 3 \pm 0 . 0 6$ </td><td> $0 . 2 0 5 \pm 0 . 0 0$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 1 0 0 \pm 0 . 0 5$ </td><td> $0 . 0 6 1 \pm 0 . 0 1$ </td><td> $0 . 2 4 9 \pm 0 . 0 7$ </td><td> $0 . 1 9 2 \pm 0 . 0 1$ </td></tr><tr><td>GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 1 8 7 \pm 0 . 0 1$ </td><td> $0 . 6 6 7 \pm 0 . 0 2$ </td><td> $0 . 4 3 2 \pm 0 . 0 2$ </td><td> $0 . 8 2 4 \pm 0 . 0 1$ </td></tr><tr><td rowspan="4">SORTZONES</td><td>Non-GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 1 7 5 \pm 0 . 0 4$ </td><td> $0 . 5 6 1 \pm 0 . 1 1$ </td><td> $0 . 4 1 9 \pm 0 . 0 5$ </td><td> $0 . 7 2 7 \pm 0 . 0 8$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 0 5 2 \pm 0 . 0 2$ </td><td> $0 . 0 6 9 \pm 0 . 0 4$ </td><td> $0 . 2 2 1 \pm 0 . 0 4$ </td><td> $0 . 2 4 9 \pm 0 . 0 6$ </td></tr><tr><td>GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 2 7 1 \pm 0 . 0 4$ </td><td> $0 . 5 4 1 \pm 0 . 0 8$ </td><td> $0 . 5 0 2 \pm 0 . 0 4$ </td><td> $0 . 7 0 7 \pm 0 . 0 5$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 2 8 3 \pm 0 . 0 2$ </td><td> $0 . 4 4 6 \pm 0 . 0 1$ </td><td> $0 . 5 2 4 \pm 0 . 0 2$ </td><td> $0 . 7 0 1 \pm 0 . 0 3$ </td></tr><tr><td rowspan="4">WALKTOFLAG</td><td>Init</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 3 3 2 \pm 0 . 0 1$ </td><td> $0 . 2 9 2 \pm 0 . 0 4$ </td><td> $0 . 5 8 2 \pm 0 . 0 2$ </td><td> $0 . 5 6 0 \pm 0 . 0 2$ </td></tr><tr><td>GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 5 2 5 \pm 0 . 0 2$ </td><td> $0 . 4 4 3 \pm 0 . 1 6$ </td><td> $0 . 6 8 0 \pm 0 . 0 1$ </td><td> $0 . 6 2 5 \pm 0 . 1 4$ </td></tr><tr><td>Non-GM</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 4 9 8 \pm 0 . 0 9$ </td><td> $0 . 4 7 8 \pm 0 . 1 9$ </td><td> $0 . 5 3 9 \pm 0 . 0 6$ </td><td> $0 . 5 0 5 \pm 0 . 2 8$ </td></tr><tr><td>Init</td><td> $h _ { \mathrm { f c } }$ </td><td> $0 . 2 8 0 \pm 0 . 0 2$ </td><td> $0 . 2 2 9 \pm 0 . 0 7$ </td><td> $0 . 4 2 7 \pm 0 . 0 1$ </td><td> $0 . 3 0 7 \pm 0 . 1 1$ </td></tr></table>

Results. Every environment has a shared competition axis, but the policy compresses the competition onto it to very different degrees (Table 13). In ATTRIBUTE MAZE the within-sign blocks are rich in raw activations $( \kappa _ { E } = 0 . 5 8 9$ at $h _ { \mathrm { f c } }$ for GM), and the Fisher metric removes most of that texture $( \kappa _ { F } = 0 . 9 4 2 $ , against 0.148 at initialization, Figure 30). An axis fit on 146 states aligns with held-out competition directions at cos $_ { F } = 0 . 9 5 9$ , and between 74% and 86% of the Fisher energy of $d _ { \mathrm { c o m p } }$ lies on a single logit contrast, UP against the RIGHT action group. The state mainly chooses the sign and the magnitude of one shared action contrast, $d _ { \mathrm { c o m p } } ( s ) \approx \sigma ( s ) \beta ( s ) u _ { \mathrm { c o m p } }$ . The textured policies compress the same way $( \kappa _ { F } = 0 . 9 1 8$ and held-out $\cos _ { F } = 0 . 9 4 6$ in GM agents, right half of Figure 30).

The robot tasks show the same signed block structure, but weaker (Figure 31). Since initialized networks already share part of the direction through the input, we read $\kappa _ { F }$ against its initial value. Above initialization, GM agents gain 0.793 in ATTRIBUTE MAZE, 0.599 in ATTRIBUTETABLE, 0.249 in SORTZONES and 0.213 in WALKTOFLAG. In ATTRIBUTETABLE the left-right gripper contrast carries 0.570 of the Fisher energy of $d _ { \mathrm { c o m p } } ,$ against 0.053 at initialization, and in SORT-ZONES 0.277, against 0.064, so ATTRIBUTETABLE comes closest to reducing its competition to one action contrast, as ATTRIBUTE MAZE does with 74% to 86%. The axis is not specific to goal misgeneralization in any environment, as Non-GM agents share it almost as much as GM agents (κ = 0.952, 0.561, 0.446 and 0.478, lower row of Figure 31).

The robot results depend on the bank in the expected direction. On the ATTRIBUTETABLE worldedit bank, where the conflict orientation is often ambiguous, same-sign κ is 0.281 and both cues pull toward their own candidates in 0.597 of GM states, against 0.667 and 0.884 on the strict bank, while the within-state cos is near zero on both (0.098 and −0.002). At the SORTZONES start frame, before a pad is chosen, the within-state cos is less opposed than at the decision frame (−0.293 against −0.435), while $\kappa _ { F }$ (0.612 against 0.541) and the fraction of states in which both cues pull their own way (0.908 against 0.820) change little.

![](images/deb339af353e9b96f0842885ac27b39567bc64cf33eaa0c415b6d541d1e4d4e3.jpg)

Figure 30: The policy compresses a rich competition geometry onto one signed axis (ATTRIBUTE MAZE). Pairwise similarity between state-local competition directions across the 196 conflict-bank states, averaged over the three GM policies of each background, black (left four panels) and textured (right four panels), each on its own background. Columns show $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ for each background, rows show $\kappa _ { E }$ and $\kappa _ { F }$ . States are grouped by whether the line candidate lies along UP or along RIGHT. Under the swap construction the cross-sign blocks are antisymmetric by construction, so the informative structure is the within-sign alignment.  
![](images/6c6cab648fb948dff22df7e0d8453670268da0b8016962c4e6b37d2fc9c9b383.jpg)  
Figure 31: Competition directions across states in the robot tasks $( h _ { \mathrm { f c } } ,$ fit states of the strict ATTRIBUTETABLE bank, the SORTZONES decision-frame bank and the WALKTOFLAG world-edit bank). Pairwise $\kappa _ { E }$ and $\kappa _ { F }$ of $d _ { \mathrm { c o m p } }$ , averaged over three seeds, for GM (top) and Non-GM (bottom) agents. States are grouped by conflict orientation. Unlike in ATTRIBUTE MAZE, mirrored states are separate scenes, so the cross-sign blocks are not antisymmetric by construction.

Controls in ATTRIBUTE MAZE. Some concerns could come up about the ATTRIBUTE MAZE axis. First, the local Fisher metric has rank at most 14, which makes cosines more extreme. However, a matched random-vector null gives a mean absolute similarity of only 0.428, against 0.942 for real pairs, so the low rank does not produce the alignment on its own. Second, one might consider the axis an artefact of the swap construction, which makes mirrored states exact negatives of each other. However, competition directions built from single-cue ablations, which have no such symmetry, put 92% of their Fisher energy on the swap axis in GM policies, against 9% to 10% at initialization, and recover the sign in 92% to 100% of held-out states, so the axis is a property of the policy and not of the edit. Third, the within-sign blocks are rich in raw activations, and one might ask whether this fine structure is part of the competition. However, it tracks the spatial similarity of the inputs (correlation 0.735 at $h _ { \mathrm { f c } }$ , against 0.733 at initialization) and lies mostly in directions the policy does not read, so it reflects how similar the scenes look rather than how the policy decides. Finally, every state of the conflict bank starts the agent at the same cell, so the single axis might only hold for that one situation. However, on the diversified bank, where the agent starts at every walkable cell, the axis splits into roughly one shared axis per action-conflict class $( \kappa _ { F } = 0 . 8 8 9$ for pairs that share the action contrast and the bearing, −0.213 for pairs that share neither), so the single axis of the conflict bank is the special case of one conflict class.

![](images/b25ca6e233a5117b3a8211031e8de51b28c25ae79860382970c5497ffcac5a79.jpg)  
Figure 32: Tug of war on the shared axis in four environments $( h _ { \mathrm { f c } }$ , held-out states). Each small point is one state of one policy, placed by how far the goal-cue swap moves it along $u _ { \mathrm { c o m p } }$ toward the goal cue’s candidate (goal pull, x) and how far the proxy-cue swap moves it toward the proxy cue’s candidate (proxy pull, y), both in the Fisher metric. GM and Non-GM agents are overlaid, large markers are the mean and 95% t-interval over three seeds. Above the dashed identity line the proxy pulls harder, in the upper-right quadrant both cues pull toward their own candidates, and left of the vertical axis the goal cue pushes toward the proxy’s candidate. Axes are linear and scaled per environment.

## C.5.3 WHY DOES THE PROXY WIN ON THE SHARED AXIS?

If both cues act on one axis, the competition is a tug of war, and we can ask which cue pulls harder and whether this decides the agent’s choice.

Measuring each cue’s pull. Each swap moves one cue to the other candidate, so the displacement it causes is exactly the influence that cue had on the conflict state. To express this influence in the currency the policy uses, we project each swap displacement onto $u _ { \mathrm { c o m p } }$ in the Fisher metric of the unedited state, as in Appendix B.3,

$$
\alpha _ { F } ( \Delta ) = \frac { u _ { \mathrm { c o m p } } ^ { \top } F ( s ) \Delta } { u _ { \mathrm { c o m p } } ^ { \top } F ( s ) u _ { \mathrm { c o m p } } } , \qquad \Delta \in \{ \Delta _ { \star } ( s ) , \Delta _ { p } ( s ) \} .
$$

So $\alpha _ { F } ( \Delta _ { \star } )$ measures how far the goal cue pulls the state along the axis, and $\alpha _ { F } ( \Delta _ { p } )$ how far the proxy cue pulls it, both oriented so that a positive value points toward the candidate the swap favours. All pulls are reported on held-out states at $h _ { \mathrm { f c } } , 5 0$ in ATTRIBUTE MAZE, 204 in ATTRIBUTETABLE, 198 in SORTZONES and 200 in WALKTOFLAG.

The proxy pulls harder in every GM cohort. In ATTRIBUTE MAZE the colour swap moves GM policies along the axis by 9.321 and the shape swap by 5.974, a ratio of means of 1.564 (95% CI [1.34, 1.79]), and Non-GM policies show the mirror image, with a ratio of 0.067. To compare environments with one estimator we use the median over states of the ratio of the two pulls (Figure 34, bottom right). It is 1.527 for GM and 0.082 for Non-GM in ATTRIBUTE MAZE, and in the robot tasks 5.106, 6.216 and 4.228 for GM against 0.267, 0.211 and 0.449 for Non-GM (ATTRIBUTETABLE, SORTZONES, WALKTOFLAG). So on the axis the policy reads, the proxy outweighs the goal in GM agents and the goal outweighs the proxy in Non-GM agents in all four environments.

Randomly initialized networks already favour the proxy in every one of these environments, with a median ratio of 1.644 in ATTRIBUTE MAZE and 2.077, 2.901 and 6.028 in the robot tasks. Their pulls are small, 30 to 50 times smaller than those of trained policies in ATTRIBUTE MAZE and 100 to

1000 times smaller in the robot tasks, so the initial ratio is a direction rather than a precise value. GM training keeps this preference (the log ratio changes by −0.07 in ATTRIBUTE MAZE) and Non-GM training reverses it (−3.0). Per unit of pixel change, initialized networks do not favour the proxy in the robot tasks (Fisher Q ratio 0.76 to 0.85, every 95% CI including 1). There the preference comes from the input, as the tower fills about 14 times more pixels than the flag and the ATTRIBUTETABLE colour swap carries 2.69 times the pixel energy of the shape swap. In ATTRIBUTE MAZE the colour swap is the smaller edit but moves initialized networks as much as the shape swap (Q ratio 1.17, 95% CI 0.71 to 1.94), possibly because its pixel changes all point one way and survive the pooling down to $h _ { \mathrm { p r e } } ,$ , while those of the shape swap cancel within a cell. The proxy’s advantage at initialization is therefore mostly an input effect, in line with networks preferring the more easily extracted feature (Hermann & Lampinen, 2020), that GM training does not overturn, and the comparison with Non-GM agents carries the result. A prior of this kind may explain why misgeneralization is favoured early in training, which we leave as an exciting open direction.

How clean the tug of war is. The environments differ in how balanced the tug of war is and in whether the two cues pull in opposite directions state by state (Figure 32). In ATTRIBUTE MAZE both cues pull toward their own candidates in 0.967 of GM states, so the cloud sits in the upper-right quadrant, slightly above the identity line. ATTRIBUTETABLE (0.884) and SORTZONES (0.820) look the same, but their clouds sit far above the identity line, because the goal pulls with only a fifth to a sixth of the proxy’s strength. In WALKTOFLAG (0.650) the goal pull changes sign from state to state, so the cloud spreads across the vertical axis, and one seed is pulled only weakly by the tower (ratio 1.521). This matches the object competition found for WALKTOFLAG in Appendix C.4.5, where flag and tower each pull the agent toward themselves and the tower wins, without the flag being pushed away. Two independent pulls do not form the two ends of one contrast, so the flag’s pull on $u _ { \mathrm { c o m p } }$ is weak and its sign is not tied to the flag’s side. The tug of war is therefore balanced in ATTRIBUTE MAZE, consistent but lopsided in ATTRIBUTETABLE and SORTZONES, and inconsistent in WALKTOFLAG.

Where the unedited state sits. In ATTRIBUTE MAZE the two pulls also predict what the agent does. The shape swap puts both cues on the proxy’s candidate (the proxy consensus), and the colour swap puts both cues on the goal’s candidate (the goal consensus). The unedited state lies between these two consensus states, and its position is

$$
t ( s ) = \frac { \alpha _ { F } ( \Delta _ { \star } ) } { \alpha _ { F } ( \Delta _ { \star } ) + \alpha _ { F } ( \Delta _ { p } ) } ,
$$

with t = 0 at the proxy consensus and t = 1 at the goal consensus. The median held-out state sits at t = 0.396 in GM policies and at t = 0.933 in control policies. In Figure 33b, the fraction of GM rollouts that reach the intended goal stays below 0.2 in the bins on the proxy side and reaches 0.57 in the bin closest to the goal side. The base position predicts the GM outcome. The signed distance of the base state from the midpoint separates rollouts that reach the goal from rollouts that reach the proxy with an AUC of 0.774 over all 196 states (95% CI [0.723, 0.822], bootstrap over states within each seed) and 0.846 within each conflict orientation ([0.800, 0.891]), timeouts excluded, and every seed lies above 0.70. The interval across the three seeds is wider ([0.631, 0.916]), because each seed’s AUC rests on few goal outcomes, as GM agents rarely reach the goal (27 to 60 of 196 states per seed, and 5 to 11 of the 50 held-out states, where the AUC is 0.736, [0.614, 0.839]). The proxy pull alone predicts the GM outcome as well as both pulls together (0.779), while the goal pull alone is at chance (0.459). We did not run this outcome analysis in the robot tasks, where t is undefined in the states in which the goal pull changes sign.

## C.5.4 WHAT DOES THE GEOMETRY SUGGEST FOR RECOVERING THE GOAL?

Figure 34 puts the components of the competition side by side. All four environments share one property, a proxy that outweighs the goal on the axis the policy reads in GM agents and not in Non-GM agents (bottom right). Initialized networks already favour the proxy (grey bars), so GM training keeps this preference and Non-GM training reverses it (Appendix C.5.3). The environments differ in the rest of the geometry (other three panels), and this difference suggests what to expect when we try to recover the goal next.

A global shared axis makes a simple prediction. If the competition in every state is mostly one signed contrast, $d _ { \mathrm { c o m p } } ( s ) \approx \sigma ( s ) \bar { \beta } ( s ) \bar { u } _ { \mathrm { c o m p } } ,$ then one fixed push along $u _ { \mathrm { c o m p } } ,$ applied with the sign $\sigma ( s )$ of the state, should move the policy toward the goal in every state, without rebuilding the direction for each state.

![](images/2b8808aaaadc138c4686c5fccf2b400556dec5f09e9250a3bacfa84e6fb23e48.jpg)

![](images/104cac3edf49fabeb7a42d077438d63dea287adf903df261c7510efc6644ddc1.jpg)

![](images/c55cfb8e2cc75a74a3f01c0f46374362b6f1b7b686b58b700d1e5ae7363fed28.jpg)

Figure 33: GM policies weight the proxy cue more strongly on the shared competition axis (ATTRIBUTE MAZE, $h _ { \mathrm { f c } } ,$ black background). (a) Goal pull against proxy pull per held-out state (small points) with cohort means and 95% t-intervals. States above the identity line are pulled harder by the proxy. Axes are symmetric-logarithmic. (b) Fraction of GM rollouts reaching the goal in five quantile bins of the base position t, with Wilson 95% intervals (all 196 states, timeouts excluded). (c) AUC for predicting a goal versus a proxy outcome from the goal pull, the proxy pull, or both.  
![](images/a7372fda109dc1ba5b6977231093a152f89ebba79a7ea3181afdb81ee51d22b1.jpg)

![](images/0c22f4d983b7e918afdee3a278a54fe82960a243d2b8ff5a6e166ca543c946d5.jpg)

![](images/113edcc85fd53f7e2af3249bf0720a4f2c5331a13d87dd9885d5a049b4e6438a.jpg)

![](images/6c18a627ce0f8789cb50b6e0430a8d551a05315e3c3190ccda36b159763b4de6.jpg)  
Figure 34: In every environment the proxy outweighs the goal in GM agents and not in Non-GM agents, while the rest of the geometry differs between environments $( h _ { \mathrm { f c } } )$ . Each panel is one component of the competition. Top left, Fisher cosine of the two cue edits within a state (negative means opposed). Top right, same-sign $\kappa _ { F }$ (how much the competition lies on one axis shared across states). Bottom left, fraction of states in which both cues pull toward their own candidates. Bottom right, median per-state ratio of proxy pull to goal pull (log scale, above 1 the proxy pulls harder). Grey bars are initialized networks. Mean and 95% t-interval over three seeds, small points are seeds.

ATTRIBUTE MAZE, with the cleanest push and pull on one shared global axis, should be the easiest to recover. Its two cues are near antipodal within a state, the competition collapses onto one axis shared across states $( \kappa _ { F } ~ 0 . 7 9 3$ above initialization), both cues pull their own way in almost every state, and the tug of war is close to balanced. A single push along $u _ { \mathrm { c o m p } }$ is therefore a push from the proxy toward the goal in almost every state.

WALKTOFLAG sits at the other end. Its competition geometry is largely individual to each state, with within-state opposition close to its initial value, a shared axis that rises only 0.213 above initialization and both cues pulling their own way in only 0.650 of states. Without a global axis there is no single direction along which to steer it.

SORTZONES has a push and pull within each state but a weaker shared axis. It is the only robot task whose cues are clearly opposed in GM agents $( \cos _ { F } = - 0 . 4 3 5$ , against 0.147 in Non-GM agents), in the gripper motion and in the joint directions that leave the gripper in place alike, but its axis rises only 0.249 above initialization. Its decision is where to push the cube, so the competition direction depends on where the cube and the gripper are at the decision frame, and the policy reads a broad direction rather than one contrast. Steering along a shared axis may therefore be less efficient there, while steering along the direction of each state can still work.

ATTRIBUTETABLE has the opposite profile. Its shared axis is strong, with $\kappa _ { F } ~ 0 . 5 9 9$ above initialization and more than half of the Fisher energy of its competition direction on the left-right gripper contrast, but its two cues are not opposed within a state on average, because the arm’s redundant joints carry an aligned component that cancels the opposed gripper motion. A push along the axis moves the gripper contrast but leaves this aligned part in place, so moving the policy toward the goal may need a different or larger effort, similar to WALKTOFLAG.

These expectations rest on three seeds per cohort, and the robot tasks add caveats of their own. Their policies act in a continuous space whose Fisher metric has rank at most eight, their pull ratios start above one at initialization, and the SORTZONES decision frames come from GM and Non-GM rollouts in equal parts and so lie partly off each cohort’s own state distribution. The next section tests them by steering the existing wiring (Appendix D.1.2).

## D RECOVERABILITY: CAN GOAL-DIRECTED BEHAVIOR BE RECOVERED?

We have located where the goal is lost and which role the proxy takes in each environment. We now ask whether the goal can be recovered from the representations the agents already have, either through the existing wiring of the policy, by steering (Appendix D.1), or by relearning the wiring downstream of a representation, by rewiring (Appendix D.2). The two methods act at different depths of the network and answer two questions. First, can a representation found inside the agent be used as a goal at all? Information present in an agent need not be usable to orient its behavior. Second, which recovery method is needed at which depth? Our hypothesis is that the depth at which the goal is lost indicates which method works. If the goal is lost only at the decision, a goal representation is present but too weak, and steering the existing policy should suffice. If goal information reaches the policy but is not used to navigate, the existing wiring is not enough to change behavior, but a new readout trained on the existing representation can be taught to use it. If the goal is already lost in the representation, neither method should work, and recovery would need new training. Both methods therefore ask whether the information inside the agent can be reused to move it to the right target, not whether they restore a goal-directed agent.

## D.1 STEERING THE EXISTING WIRING

The preceding analyses identify internal goal-related directions, but geometry alone does not establish that these directions control behavior. A representation displacement can be decodable, policysensitive, or geometrically well organized without providing a useful intervention direction. We therefore ask whether the internal structure identified above can be used to redirect the existing policy without retraining its weights.

The geometry suggests two increasingly specific causal questions. First, can the state-specific coun terfactual direction available at the current state causally redirect behavior? Second, in the environments where this local intervention succeeds, how much of that causal effect survives compression into the shared competition axis identified across states?

All interventions are applied at $h _ { \mathrm { f c } }$ , after the final ReLU. This is the representation the policy and value heads read directly, so a push there changes the input to the head without passing through any further nonlinearity. The counterfactual direction used for steering is reconstructed at the same representation at which it is injected. In ATTRIBUTE MAZE, pushing at $h _ { \mathrm { p r e } }$ instead gives GM policies a reach within 0.06 of the $h _ { \mathrm { f c } }$ push at every dose up to $\beta = 1 ~ ( 0 . 9 3 3$ against 0.953 at $\beta = 1 )$ , so the choice of site does not drive the results below.

## D.1.1 CAN THE STATE-LOCAL COUNTERFACTUAL DIRECTION REDIRECT THE POLICY?

The intervention. The most direct causal use of the counterfactual analyses is to retain the complete state-dependent direction. This makes no assumption that the relevant geometry is shared across states or low-dimensional. At each rollout state $s _ { t }$ , we reconstruct the counterfactual representations associated with the intended and proxy-supported alternatives and use their contrast as the intervention direction $d _ { \mathrm { l o c a l } } ( s _ { t } )$ . Its positive orientation always points toward the counterfactual in which the evidence supports the intended target. Where goal and proxy cues sit on the same candidates (ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES) and in WALKTOFLAG, this is exactly the competition direction defined above,

$$
d _ { \mathrm { l o c a l } } ( s _ { t } ) = d _ { \mathrm { c o m p } } ( s _ { t } ) = \Delta _ { p } ( s _ { t } ) - \Delta _ { \star } ( s _ { t } ) .
$$

Its swaps put the proxy cue on the goal candidate and the goal cue on the proxy candidate (colour and shape in ATTRIBUTE MAZE and ATTRIBUTETABLE, colour and cross in SORTZONES, flag and tower in WALKTOFLAG). In MAZE, the two counterfactuals place the cheese at its own cell and in the top-right corner, $d _ { \mathrm { l o c a l } } ( s _ { t } ) = h _ { \mathrm { f c } } ( s _ { t } ^ { \star } ) - h _ { \mathrm { f c } } ( s _ { t } ^ { p } )$ , in LOCATIONREACH the ball at its own position and in the training corner, and in PUSHTOMARKER the marker at its own position and at the table centre. In COIN, they place the coin at its position and on the shortcut side, ahead of the agent, and rollouts start from the edited conflict state in which the coin lies behind the agent, off the shortcut path. The important commonality is that the intervention is reconstructed from the current state at every step rather than replaced by one fixed direction learned across states. In the robot tasks, the edited copies of the scene are re-synchronised with the acting copy after every step, so that $d _ { \mathrm { l o c a l } } ( s _ { t } )$ always contrasts the same arm, cube or body state.

For steering dose $\beta ,$ the corrective intervention, a state-local form of activation addition (Turner et al., 2023b), is

$$
\delta _ { \mathrm { l o c a l } } ^ { + } ( s _ { t } ; \beta ) = \beta d _ { \mathrm { l o c a l } } ( s _ { t } ) ,
$$

giving

$$
h _ { \mathrm { f c } } ^ { \prime } ( s _ { t } ) = h _ { \mathrm { f c } } ( s _ { t } ) + \delta _ { \mathrm { l o c a l } } ^ { + } ( s _ { t } ; \beta ) .
$$

The anticorrective intervention reverses the same direction,

$$
\delta _ { \mathrm { l o c a l } } ^ { - } ( s _ { t } ; \beta ) = - \beta d _ { \mathrm { l o c a l } } ( s _ { t } ) .
$$

This sign reversal is important. A direction that merely perturbs the policy may occasionally improve goal reach without being a meaningful causal axis. A genuine target-selection direction should instead have an ordered effect. Moving along the corrective orientation should favour intended-goal behavior, while reversing the same intervention should move behavior in the opposite direction.

For sufficiently small interventions, this follows the local first-order prediction

$$
\begin{array} { r } { C _ { s } ( h + \delta ) - C _ { s } ( h ) \approx \nabla _ { h } C _ { s } ( h ) ^ { \top } \delta , } \end{array}
$$

where $C _ { s }$ is the target-selection contrast defined above. The rollout experiment asks the stronger question of whether this local effect remains useful after repeatedly intervening as the state changes.

Controls and dose response. We compare four conditions,

baseline, matched random, corrective, anticorrective.

The baseline leaves the representation unchanged. The random control injects three random directions with norm matched to the corresponding treatment intervention. It therefore controls for the possibility that sufficiently large perturbations alter behavior regardless of their orientation. The corrective and anticorrective conditions use opposite signs of the same state-local direction. Together with the random control, this separates directional causal control from generic intervention effects.

We evaluate the dose grid β ∈ {0.125, 0.25, 0.5, 1.0} in MAZE and ATTRIBUTE MAZE, extend it to $\beta \in \{ 0 . 7 5 , 1 . 2 5 , 1 . 5 , 2 . 0 \}$ in ATTRIBUTE MAZE, and use $\beta \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ in COIN, where the direction is smallest in norm. The robot tasks use $\beta \in \{ 0 . 1 \dot { 2 } 5 , 0 . 2 5 , 0 . 5 , 1 , 2 \}$ , extended to $\beta = 4$ in PUSHTOMARKER and WALKTOFLAG. This lets us distinguish a stable dose-response relationship from an effect that occurs only at one arbitrary intervention magnitude. The policies act greedily, with the most likely action in Procgen and the mean action in the robot tasks.

Evaluation states. Every steering state is a conflict state in which goal and proxy require different behavior, taken from the conflict banks or from the steering samples of Appendix E, and in WALK-TOFLAG from the conflict-test bank with the native start pose, as in rewiring. Steering fits nothing to these states, and the shared axis below is fit on states disjoint from them.

Outcomes. The primary outcome is intended-target reach. Where an episode can also end at the proxy target, reach factorizes exactly into decisiveness and conditional preference,

$$
\mathrm { r e a c h } = D \cdot \mathrm { P r } ( \mathrm { g o a l } \mid \operatorname { d e c } ) , \qquad D = \frac { N _ { \mathrm { g o a l } } + N _ { \mathrm { p r o x y } } } { N } .
$$

D asks whether the policy still completes the task at all, and Pr(goal | dec) asks which target it chooses when it does. This lets us tell a change in target selection apart from a policy that has simply been pushed into timeouts. The factorization applies in ATTRIBUTE MAZE and COIN, and in every robot task. In MAZE, the episode ends only at the cheese, so reach is the only outcome. In LOCATIONREACH, ATTRIBUTETABLE and SORTZONES the outcome is the first candidate the gripper touches or the cube enters, in PUSHTOMARKER it is where the cube rests at the end of the episode, and in WALKTOFLAG it is the first object the robot reaches.

Results. Steering works exactly following the depth at which the goal is lost, confirming our hypothesis for the environments we tested (Figure 35, Table 14). Where the goal is lost in the representation or reaches the policy without being used, no push makes the GM policy pursue it. Where goal and proxy compete at the decision, the state-local push raises GM goal reach to at least 0.84.

![](images/1b458aaa9958dc52954757c257abff217af4f2195d8930714c9cda65c46d449d.jpg)

![](images/412142f95f1357d427d09bfd0abd545dae5e2dfd43b6b6c9389a1c220133b894.jpg)  
Figure 35: State-local steering, ordered by where the goal is lost. Goal reach under corrective, reversed and norm-matched random pushes at $h _ { \mathrm { f c } }$ against the dose $\beta ,$ GM (upper row of each block) and Non-GM (lower row). AM is ATTRIBUTE MAZE on the black background, the textured background is in Table 14, and WALKTOFLAG uses the native set. Grey bands show the unsteered baseline. Mean with 95% t-interval over three checkpoints.

(i) Goal lost in the representation: MAZE and LOCATIONREACH. No condition moves the GM policy in either environment. It reaches the cheese in 0.001 of the 320 states without steering and in at most 0.002 under any push at any dose, and it never touches the ball first in LOCATIONREACH. The representation and policy results leave nothing to steer here. Cheese bearing is not decodable beyond the agent’s own position, the cheese edit barely reaches the GM policy $( \bar { Q } = 0 . 0 0 1$ against 1.689, Section 3.2), and in LOCATIONREACH a moved ball even turns the planned GM step slightly away from it (Appendix C.2). The steering direction is built from these edits, so it is small, with a median $\| d _ { \mathrm { l o c a l } } ( s _ { 0 } ) \| _ { 2 }$ of 1.40 against 17.97 for Non-GM policies in MAZE and 2.536 against 20.015 in LOCATIONREACH. Large pushes sometimes reach the goal. Pushed at fixed norms of 16 to 32, about the Non-GM norm at $\beta = 1$ , GM policies in MAZE reach the cheese in up to 7 of 48 states for one seed (0.090 averaged over seeds at norm 32), above random pushes of equal norm (at most 0.049), while the reversed push reaches it in up to 4 of 48. We read this as a large perturbation in the direction of the cheese rather than goal pursuit, since it stays far below Non-GM cheese pursuit and is built from an edit the GM policy barely registers. In LOCATIONREACH the grid stops where the GM push is a quarter of the Non-GM push at $\beta = 1$ , so there we cannot separate a missing direction from a small one. In Non-GM policies, the same construction is clearly causal in the other direction. The anticorrective push removes cheese pursuit almost completely, from 0.567 to 0.002 at $\beta = 1$ while random pushes of equal norm only reduce it to 0.340, and in LOCATIONREACH it sends the gripper to the corner in 0.983 of states. ${ \mathrm { S o } } ,$ removing the cheese-associated direction from a policy that already uses the cheese destroys goal pursuit, but adding the same kind of direction to GM does not create cheese pursuit. The intervention can suppress an existing computation, but it cannot instantiate one that the native GM policy does not already use. This matches the MAZE Fisher analysis, computed on a separate set of decision states, where the policy-sensitive energy of the (ii) Goal reaches the policy but is not used: PUSHTOMARKER and COIN. Here the goal edit still moves the GM policy, if weakly, but no push creates goal pursuit. In PUSHTOMARKER, GM goal reach is 0.000 at $\beta = 1$ and at most 0.018 at $\beta = 4 .$ Push size does not explain this, since at $\beta = 2$ the GM push has the norm of the Non-GM push at $\beta = 1$ (median 2.516 against 5.033) and GM goal reach is still 0.003. What the corrective push does is weaken the substitute target, with the cube resting at the centre in 0.628 of scenes at $\beta = 1$ against 0.925 unsteered and 0.896 under random pushes, without moving the cube to the marker. So the push breaks the pursuit of the centre but gives the head no marker location to pursue instead. In COIN, GM coin reach is 0.018 without steering and stays between 0.016 and 0.027 under corrective, anticorrective and random pushes at every dose up to $\beta = 4 . \mathrm { A t } \ \beta = 4 $ , the corrective push exceeds the random push by −0.001 (95% CI ±0.013 across checkpoints), and the conditional preference Pr(goal | dec) moves only from 0.028 to 0.035. Because the GM baseline is close to zero, the anticorrective push cannot show a decrease, so the evidence for GM rests on the corrective push matching the random push. Here too push size is not the reason, since at $\beta = 4$ the GM push has about the norm of the Non-GM push at $\beta = 1$ (mean 1.93 against 7.85), where Non-GM policies already respond. Both nulls follow from how the goal reaches these policies. The goal edit is registered as a change of the scene but not as a direction. It reaches the GM policy at about a sixth of the Non-GM sensitivity in PUSHTOMARKER and a sixteenth in COIN (ratios of Q of 0.159 and 0.063, Appendix C.3), but its signed follow is close to zero in both (ratios 0.017 and 0.001, Appendix C.2), and the role analysis finds the GM reaction to be the one any disc gets in PUSHTOMARKER and spread over the actions in proportion to their prior probability in COIN (Appendices C.4.2 and C.4.3). The steering direction is built from this edit, so it carries a change of the scene without an orientation, and even at the norm that moves Non-GM policies the push is too weak to be registered as an orientation toward the goal. In Non-GM policies, the same construction is causal in both directions and grows with dose. In COIN, the corrective push raises reach from 0.121 to 0.259 at $\beta = 1$ and 0.373 at $\beta = 4 ,$ against 0.171 for random pushes, and the anticorrective push lowers it to 0.019 and 0.012, and in PUSHTOMARKER the reversed push sends the cube to the centre in 0.943 of scenes. So, in both environments the direction is causally live where the policy reads it, and inert in GM policies.

Table 14: State-local steering at $\beta = 1$ . Intended-target reach, mean $\pm \mathrm { \bf ~ S D }$ across three checkpoints. Each entry is read against the unsteered baseline in its own row. Blocks follow Figure 35. WALKTOFLAG rows use the native set. Bold marks corrective values whose 95% t-interval lies above those of both the unsteered baseline and the random push.
<table><tr><td>Environment</td><td>Cohort</td><td>Baseline</td><td>Corrective</td><td>Anticorrective</td><td>Random</td></tr><tr><td>MAZE</td><td>GM</td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td rowspan="3">LOCATIONREACH</td><td>Non-GM</td><td> $0 . 5 6 7 \pm 0 . 0 3$ </td><td> $0 . 5 6 9 \pm 0 . 0 3$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 3 4 0 \pm 0 . 0 4$ </td></tr><tr><td>GM</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Non-GM</td><td> $1 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 1 3 8 \pm 0 . 1 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 1 4 \pm 0 . 0 5$ </td></tr><tr><td rowspan="3">PUSHTOMARKER</td><td>GM</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Non-GM</td><td> $0 . 9 0 3 \pm 0 . 0 2$ </td><td> $0 . 3 7 5 \pm 0 . 1 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 8 2 \pm 0 . 1 0$ </td></tr><tr><td>GM</td><td> $0 . 0 1 8 \pm 0 . 0 1$ </td><td> $0 . 0 1 6 \pm 0 . 0 1$ </td><td> $0 . 0 2 0 \pm 0 . 0 1$ </td><td> $0 . 0 2 0 \pm 0 . 0 1$ </td></tr><tr><td rowspan="2">ATTRIBUTE MAZE (black)</td><td>Non-GM</td><td> $0 . 1 2 1 \pm 0 . 0 5$ </td><td> $0 . 2 5 9 \pm 0 . 0 8$ </td><td> $0 . 0 1 9 \pm 0 . 0 1$ </td><td> $0 . 1 7 1 \pm 0 . 0 7$ </td></tr><tr><td>GM</td><td> $0 . 1 6 0 \pm 0 . 0 6$ </td><td> ${ \bf 0 . 9 5 3 \pm 0 . 0 5 }$ </td><td> $0 . 0 0 7 \pm 0 . 0 1$ </td><td> $0 . 1 7 1 \pm 0 . 0 6$ </td></tr><tr><td rowspan="2">ATTRIBUTE MAZE (texture)</td><td>Non-GM</td><td> $0 . 8 1 3 \pm 0 . 0 5$ </td><td> $0 . 5 8 7 \pm 0 . 0 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 0 4 \pm 0 . 0 5$ </td></tr><tr><td>GM</td><td> $0 . 0 4 0 \pm 0 . 0 7$ </td><td> ${ \bf 0 . 8 4 0 \pm 0 . 1 8 }$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 4 9 \pm 0 . 0 7$ </td></tr><tr><td rowspan="2">ATTRIBUTETABLE</td><td>Non-GM</td><td> $0 . 5 6 7 \pm 0 . 1 3$ </td><td> $0 . 5 0 7 \pm 0 . 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 4 1 8 \pm 0 . 0 7$ </td></tr><tr><td>GM</td><td> $0 . 0 4 0 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 1 5 \pm 0 . 0 2 }$ </td><td> $0 . 0 0 5 \pm 0 . 0 0$ </td><td> $0 . 1 0 8 \pm 0 . 0 1$ </td></tr><tr><td rowspan="2">SORTZONES</td><td>Non-GM</td><td> $0 . 9 8 3 \pm 0 . 0 1$ </td><td> $0 . 6 5 7 \pm 0 . 0 9$ </td><td> $0 . 1 7 0 \pm 0 . 0 8$ </td><td> $0 . 6 8 8 \pm 0 . 0 7$ </td></tr><tr><td>GM</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 0 . 9 1 0 \pm 0 . 0 4 }$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0$ </td></tr><tr><td rowspan="2">WALKTOFLAG</td><td>Non-GM</td><td> $0 . 9 4 3 \pm 0 . 0 0$ </td><td> $0 . 4 6 8 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 9 6 \pm 0 . 0 8$ </td></tr><tr><td>GM</td><td> $0 . 3 1 2 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 7 1 \pm 0 . 0 1 }$ </td><td> $0 . 0 5 2 \pm 0 . 0 1$ </td><td> $0 . 3 2 5 \pm 0 . 0 3$ </td></tr><tr><td rowspan="2"></td><td>Non-GM</td><td> $0 . 9 6 1 \pm 0 . 0 1$ </td><td> $0 . 9 5 8 \pm 0 . 0 1$ </td><td> $0 . 4 6 9 \pm 0 . 0 6$ </td><td> $0 . 9 5 5 \pm 0 . 0 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

GM cheese edit is roughly three orders of magnitude below that of Non-GM policies and its signed effect on target selection is indistinguishable from zero. This asymmetry is consistent with earlier interventions on MAZE policies, where subtracting a cheese vector suppressed cheese-seeking while adding it left cheese-seeking unchanged, and editing goal-tracking channels retargeted the policy only partially (Mini et al., 2023). Those policies were trained with the cheese anywhere in the topright $5 \times 5$ region and pursued it at baseline, whereas our GM agents are trained with the cheese on the exact corner cell (rand region= 0) and our Non-GM agents with a randomized position (rand region= 25), so the agreement is in the direction of the effect rather than a replication of our setting.

(iii) Goal and proxy compete: ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALK-TOFLAG. The local competition direction is a strong causal handle on the GM policy in every competition environment. On the black background, intended-goal reach rises from 0.160 without steering to 0.953 at $\beta = 1$ , on the textured background from 0.040 to 0.840, in ATTRIBUTETABLE from 0.040 to 0.915 and in SORTZONES from 0.000 to 0.910. In WALKTOFLAG flag-first reach rises from 0.312 to 0.971, against 0.325 under random pushes, and the reversed push lowers it to 0.052. As a robustness check, the forced-view set, in which both objects are in view from the start, gives the same picture (0.378 to 0.992, random 0.389, reversed 0.000). This was not guaranteed by the construction. A direction built from counterfactual edits at one state could have shifted the first action and then washed out, broken competent navigation, or worked only on a narrow subset of states. Instead it survives closed-loop rollout across the evaluation states. Norm-matched random pushes leave reach close to its baseline across the full dose range, and the anticorrective push drives it to zero. So, the effect is directional and ordered, not a consequence of perturbing the representation. Unlike in (i) and (ii), the GM direction is about as large as the Non-GM one, with a median norm of 10.710 against 12.033 in ATTRIBUTETABLE and 7.429 against 8.244 in SORTZONES.

The decomposition shows what the push changes. For GM policies on the black background, the conditional preference Pr(goal | dec) rises from 0.235 to 0.993 at $\beta = 1$ , while decisiveness also rises, from 0.687 to 0.960. It rises from 0.069 to 1.000 on the textured background, from 0.048 to 0.996 in ATTRIBUTETABLE, from 0.000 to 1.000 in SORTZONES and from 0.315 to 0.995 in WALKTOFLAG. Steering therefore does not merely make the agent more active. It changes which competing target wins. The texture policies need a larger dose before this happens, with reach of only $0 . 1 6 7 \pm 0 . 2 4$ at $\beta = 0 . 2 5$ before rising to 0.740 at $\beta = 0 . 5 ,$ and so do the robot policies, with 0.235 (ATTRIBUTETABLE) and 0.030 (SORTZONES) at $\beta = 0 . 2 5$ $\operatorname { A t } \beta = 2$ robot reach falls back to 0.278 and 0.303, as the push becomes large enough to disrupt the reach itself.

Non-GM policies behave differently, and we treat them as a positive control for the direction rather than as a second recovery target. They already choose the intended target whenever they complete the task, with Pr(goal | dec) = 1.00 at baseline on both backgrounds, and this stays at 1.00 under every corrective dose. Their reach nevertheless falls, from 0.813 to 0.587 on the black background, 0.983 to 0.657 in ATTRIBUTETABLE and 0.943 to 0.468 in SORTZONES, because a large push into an already working policy costs decisiveness and produces timeouts. WALKTOFLAG is the exception, with Non-GM reach 0.958 at $\beta = 1$ against 0.961 unsteered, falling only at $\beta = 4 .$ , to 0.685, about as far as under random pushes of equal norm (0.622), while the reversed push lowers it to 0.130. The anticorrective push, by contrast, lowers intended-goal reach to 0.000 in ATTRIBUTE MAZE and SORTZONES, 0.170 in ATTRIBUTETABLE and 0.469 in WALKTOFLAG, so the same direction still carries target selection in these policies.

Steering redirects the policy toward a location. Every competition direction is built from swaps that exchange what lies at the goal location u and the lure location $v ,$ so the push may tell the policy which cue to follow or only where to go. We separate the two by acting in a scene where the goal cue is removed or moved to v, while the push is still built from the swap copies of the original scene. The GM push at $\beta = 1$ reaches u either way (Table 15). With the goal cue removed, it reaches u in 0.913 of ATTRIBUTE MAZE states, 0.900 of ATTRIBUTETABLE and 0.970 of SORTZONES scenes, and with the cue moved to the lure, it goes to the now empty u in 0.880, 0.993 and 0.985. Unsteered GM policies reach u in 0.160, 0.052 and 0.000 of ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES scenes with the cue in place, at most 0.023 with it removed and never with it moved. In ATTRIBUTE MAZE the control scenes hold one object, which no policy saw in training, but the decoupled scene is still informative, since unsteered agents there reach the lone goal object at v. So the push redirects the policy using information inside the agent, and we describe all steering below, local and shared-axis, in this way and not as recovered goal pursuit. This follows from what steering does. It does not change the wiring, the weights with which the policy reads out its inputs. It injects information, changing the size of the arrows that point into the policy rather than how they are read, a kind of overwriting, so it does not amplify the agent’s perception of the goal. This is consistent with the environments where the goal is inert, where the information that steering would

Table 15: Steering redirects the policy toward a location. Reach of the goal location u by GM policies, mean ± SD across three checkpoints, with the goal cue in place, removed, or moved to the lure location v. The push is built from the original scene in all three. Bold marks the local push, which keeps reach of u high with the goal cue removed or moved. The robot rows are a separate set of rollouts on the same 200 scenes as Table 14, so their cue-in-place values differ from it by at most 0.012, and their shared axis is fit on the world-edit bank (cosine 0.935 with the strict-bank axis of Table 16).
<table><tr><td>Environment</td><td>Push</td><td>Goal cue at u</td><td>Removed</td><td>Moved to v</td></tr><tr><td rowspan="5">ATTRIBUTE MAZE (black)</td><td>none</td><td> $0 . 1 6 0 \pm 0 . 0 6$ </td><td> $0 . 0 0 7 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>local,  $\beta = 1$ </td><td> ${ \bf 0 . 9 5 3 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 9 1 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 8 8 0 \pm 0 . 0 7 }$ </td></tr><tr><td>reversed,  $\beta = 1$ </td><td> $0 . 0 0 7 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>random,  $\beta = 1$ </td><td> $0 . 1 7 1 \pm 0 . 0 6$ </td><td> $0 . 0 2 4 \pm 0 . 0 1$ </td><td> $0 . 0 3 1 \pm 0 . 0 1$ </td></tr><tr><td>shared axis,  $\beta = 0 . 2 5$ </td><td> $0 . 5 8 7 \pm 0 . 1 1$ </td><td> $0 . 0 9 3 \pm 0 . 0 4$ </td><td> $0 . 1 4 7 \pm 0 . 0 6$ </td></tr><tr><td rowspan="5">ATTRIBUTETABLE</td><td>none</td><td> $0 . 0 5 2 \pm 0 . 0 2$ </td><td> $0 . 0 2 3 \pm 0 . 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>local,  $\beta = 1$ </td><td> ${ \bf 0 . 9 1 5 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 9 0 0 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 9 9 3 \pm 0 . 0 1 }$ </td></tr><tr><td>reversed,  $\beta = 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 5 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>random,  $\beta = 1$ </td><td> $0 . 1 0 6 \pm 0 . 0 1$ </td><td> $0 . 0 8 1 \pm 0 . 0 1$ </td><td> $0 . 0 2 3 \pm 0 . 0 1$ </td></tr><tr><td>shared axis,  $\beta = 0 . 2 5$ </td><td> $0 . 3 1 0 \pm 0 . 0 4$ </td><td> $0 . 2 4 7 \pm 0 . 0 4$ </td><td> $0 . 1 9 8 \pm 0 . 0 6$ </td></tr><tr><td rowspan="4">SORTZONES</td><td>none</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>local,  $\beta = 1$ </td><td> $\mathbf { 0 . 9 0 7 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 9 7 0 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 9 8 5 \pm 0 . 0 1 }$ </td></tr><tr><td>reversed,  $\beta = 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td></tr><tr><td>random,  $\beta = 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr></table>

need to reorient the agent is not there. Steering still shows that a goal-relevant representation can be extracted from the internals of a misgeneralized agent and used to make it goal-seeking again, without any diversified training set and with only a permutation of the scene at test time. This is not ready for deployment, since a scene cannot always be edited, but it is a form of recovery, though not a recovery of navigation.

## D.1.2 HOW MUCH CONTROL SURVIVES COMPRESSION ONTO A SHARED AXIS?

Design. State-local steering retains the full counterfactual direction $d _ { \mathrm { c o m p } } ( s )$ at every state. It therefore provides the strongest test of causal accessibility, but it does not tell us how much of the intervention is actually necessary. The geometry analysis gives us a more specific hypothesis. In AT-TRIBUTE MAZE, within the canonical UP-versus-RIGHT conflict class, the policy-relevant component of $d _ { \mathrm { c o m p } } ( s )$ is well described by one signed axis, $d _ { \mathrm { c o m p } } ( s ) \approx \sigma ( s ) \beta ( s ) u _ { \mathrm { c o m p } }$ (Section C.5.2), and ATTRIBUTETABLE is the robot task with the strongest shared axis, while SORTZONES has the weakest (Appendix C.5). If the successful local intervention acts primarily through this shared axis, then much of its causal effect should survive after the remaining state-specific structure is removed.

We fit the axis only on the 146 fit states of the conflict bank, which are disjoint from the 50 evaluation states used for steering, with mirrored states always kept on the same side of the split. Within the fit states, 118 determine the axis and 28 serve as a held-out check before any steering. In ATTRIBUTETABLE the axis is fit on all 512 scenes of the strict bank, in which the two objects lie on opposite sides of the robot, and in SORTZONES on all 500 frames of the decision-frame bank, at which the hand first reaches the cube (Appendix E). Neither bank shares scenes with the world-edit scenes on which the robots are steered. In WALKTOFLAG the axis is fit on the states of the worldedit bank, and the fixed pushes were run in GM and Non-GM policies on the forced-view set and on the native set. The axis is the sign-folded mean of the competition directions,

$$
u _ { \mathrm { c o m p } } = \frac { \bar { d } _ { + } - \bar { d } _ { - } } { \| \bar { d } _ { + } - \bar { d } _ { - } \| _ { 2 } } , \qquad \bar { d } _ { \pm } = \frac { 1 } { | S _ { \pm } | } \sum _ { s \in S _ { \pm } } d _ { \mathrm { c o m p } } ( s ) ,
$$

where $S _ { + }$ contains the fit states in which the line candidate lies along UP and S those in which it lies along RIGHT, and in the robot tasks the states in which the goal candidate lies on one side of the robot and on the other. The sign $\sigma ( s ) \in \{ - 1 , + 1 \}$ used at a steering state is determined only by this orientation of its conflict. The rollout outcomes used to evaluate steering do not enter this construction.

(e)  
(a)  
![](images/19e768c39bf866294b6abadaef719a028dccf2035847b5fc6ba593d3fe2163bc.jpg)

Figure 36: Compressing local steering onto the shared axis. Goal reach of GM (top) and Non-GM (bottom) policies against the signed dose, negative doses reversed. All arms are fixed pushes computed at the start state. The ATTRIBUTETABLE axis is fit on the strict bank and the SORTZONES axis on the decision-frame bank. WALKTOFLAG is shown on the forced-view set. On the native set no fixed push moves reach in either cohort (see text). Shaded regions mark $| \beta | > 0 . 2 5$ . Mean with 95% t-interval over three checkpoints.

We compare the complete local intervention with progressively more compressed versions of it. Every compressed intervention is a fixed push, computed once at the start state $s _ { 0 }$ and held constant during the rollout. The fair reference for compression is therefore the local direction fixed in the same way, $\delta _ { \mathrm { f i x e d } } ( \beta ) = \beta d _ { \mathrm { c o m p } } ( s _ { 0 } )$ . The fully local intervention, which rebuilds $d _ { \mathrm { c o m p } } ( s _ { t } )$ at every step, remains the positive control. The sign-gated shared-axis intervention removes the state-specific direction while preserving its local magnitude and geometry-derived orientation,

$$
\delta _ { \mathrm { s h a r e d } } ( \beta ) = \beta \sigma ( s _ { 0 } ) \| d _ { \mathrm { c o m p } } ( s _ { 0 } ) \| _ { 2 } \hat { u } _ { \mathrm { c o m p } } .
$$

This is the central compression hypothesis. If it approaches the effect of $\delta _ { \mathrm { f i x e d } }$ , then the causal component of $d _ { \mathrm { c o m p } } ( s _ { 0 } )$ is largely captured by one shared direction together with its sign. We then remove the sign gate, $\delta _ { \mathrm { u n s i g n e d } } ( \beta ) = \beta \| d _ { \mathrm { c o m p } } ( s _ { 0 } ) \| _ { 2 } \hat { u } _ { \mathrm { c o m p } }$ . This preserves the shared axis and the intervention magnitude but uses the same orientation in every state. It therefore tests whether the sign identified by the geometry is causally necessary. Finally, we retain only the component of the local direction that the policy-sensitive geometry assigns to the shared axis. Using the Fisherorthogonal coefficient of Appendix B.3,

$$
\alpha _ { F } ( s _ { 0 } ) = \frac { u _ { \mathrm { c o m p } } ^ { \top } F ( s _ { 0 } ) d _ { \mathrm { c o m p } } ( s _ { 0 } ) } { u _ { \mathrm { c o m p } } ^ { \top } F ( s _ { 0 } ) u _ { \mathrm { c o m p } } } , \qquad \delta _ { F } ( \beta ) = \beta \alpha _ { F } ( s _ { 0 } ) u _ { \mathrm { c o m p } } .
$$

Each intervention is also applied with the opposite sign, and a norm-matched random direction again provides the nonspecific perturbation control.

At large doses the fixed pushes stop testing the axis. $\operatorname { A t } \beta = 1$ , the sign-gated and unsigned pushes collapse every rollout into one repeated movement, and reach then equals a ceiling set by the maze walls, 0.48 on the evaluation states. In ATTRIBUTETABLE the sign-gated push already falls from 0.300 at $\beta = 0 . 2 5$ to 0.053 at $\beta = 0 . 5$ We therefore compare the compressed interventions at $\beta \leq 0 . 2 5$ , where none of them has reached this ceiling. The fully local intervention does not have this problem, because its direction is rebuilt as the agent moves.

Results. One shared axis carries the push in ATTRIBUTE MAZE on the black background, part of it in ATTRIBUTETABLE, and none of it in SORTZONES (Figure 36, Table 16). On the black background at $\beta = 0 . 2 5$ , the sign-gated shared axis reaches 0.587 and the Fisher-projected axis 0.560, compared with 0.560 for the local direction fixed at the start state. One shared direction, together with the sign of the conflict, therefore carries essentially all of the causal effect of the state-specific direction when both are held fixed from the start state. Reversing either compressed intervention suppresses intended-goal reach just as reversing the local direction does. Without the sign, the same shared direction reaches only 0.300, and reversing it gives almost the same value, 0.307. The remaining gap to the fully local intervention, 0.640 at the same dose and 0.953 at $\beta = 1$ comes from rebuilding the direction as the state changes during the rollout, not from compressing it. On the textured background the sign-gated push reaches 0.200 on average, but 0.100, 0.020 and 0.480 in the three checkpoints, so one checkpoint carries the mean, and we base the compression conclusion on the black-background policies. Unlike the local push, the sign-gated push depends on the goal cue here. With the cue removed or moved to the lure, it reaches u in 0.093 and 0.147 of states at $\beta = 0 . 2 5$ , against 0.587 with the cue in place (Table 15).

Table 16: Compression test at $\beta = 0 . 2 5 .$ . Intended-goal reach of GM policies, mean $\pm \thinspace \mathrm { S D }$ across three checkpoints. AM, AT, SZ and WTF are ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG (forced-view set). In the texture row, the robustness variant of ATTRIBUTE MAZE, one of the three checkpoints carries the mean and the variation across checkpoints is as large as the effects. Bold marks compressed pushes toward the goal (sign-gated, Fisher-projected and unsigned) whose mean ± SD range lies entirely above that of the unsteered value.
<table><tr><td>Environment</td><td>Unsteered</td><td>Local, fixed at s0</td><td>Sign-gated</td><td>Fisher- projected</td><td>Unsigned</td><td>Unsigned, reversed</td><td>Local, rebuilt</td></tr><tr><td>AM, black</td><td> $0 . 1 6 0 \pm 0 . 0 6$ </td><td> $0 . 5 6 0 \pm 0 . 0 5$ </td><td> ${ \bf 0 . 5 8 7 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 5 6 0 \pm 0 . 0 4 }$ </td><td> $0 . 3 0 0 \pm 0 . 0 8$ </td><td> $0 . 3 0 7 \pm 0 . 1 4$ </td><td> $0 . 6 4 0 \pm 0 . 0 7$ </td></tr><tr><td>AM, texture</td><td> $0 . 0 4 0 \pm 0 . 0 7$ </td><td> $0 . 1 7 3 \pm 0 . 2 7$ </td><td> $0 . 2 0 0 \pm 0 . 2 5$ </td><td> $0 . 1 8 0 \pm 0 . 2 9$ </td><td> $0 . 0 9 3 \pm 0 . 0 9$ </td><td> $0 . 1 0 7 \pm 0 . 1 5$ </td><td> $0 . 1 6 7 \pm 0 . 2 4$ </td></tr><tr><td>AT</td><td> $0 . 0 4 0 \pm 0 . 0 2$ </td><td> $0 . 1 4 7 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 3 0 0 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 1 0 8 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 1 6 5 \pm 0 . 0 3 }$ </td><td> $0 . 1 4 0 \pm 0 . 0 3$ </td><td> $0 . 2 3 5 \pm 0 . 0 3$ </td></tr><tr><td>SZ</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 3 0 \pm 0 . 0 3$ </td></tr><tr><td>WTF</td><td> $0 . 3 7 8 \pm 0 . 0 8$ </td><td> $0 . 4 6 6 \pm 0 . 0 1$ </td><td> $0 . 4 7 9 \pm 0 . 0 3$ </td><td> $0 . 4 3 5 \pm 0 . 0 3$ </td><td> $0 . 3 4 1 \pm 0 . 1 5$ </td><td> $0 . 4 4 0 \pm 0 . 0 3$ </td><td> $0 . 5 4 4 \pm 0 . 0 1$ </td></tr></table>

In ATTRIBUTETABLE, the sign-gated push raises reach from 0.040 to 0.300, against 0.050 for random pushes of equal norm, and the reversed push lowers it to 0.005. It even exceeds the local direction fixed at the start state (0.147), partly because the two are not the same dose. At equal β and equal Euclidean norm, the sign-gated push moves the GM policy about twice as far in the Fisher norm (median 19.33 against 10.13 per unit $\beta ) _ { \negmedspace }$ , since the axis is more policy-sensitive per unit norm than $d _ { \mathrm { c o m p } } ( s _ { 0 } )$ itself. Here the sign carries only about half of the gain, since the unsigned push also raises reach, by 0.125 to 0.165, and so does the reversed unsigned push (0.140). Part of the effect is therefore a push off the proxy that does not depend on the orientation of the conflict, a part that is smaller in ATTRIBUTE MAZE (a gain of 0.140 unsigned against 0.427 signed). Unlike in ATTRIBUTE MAZE, the sign-gated push still reaches the goal location with the goal cue removed (0.247) or moved to the lure (0.198), against 0.310 with the cue in place, for an axis fit on the worldedit bank whose cosine with the strict-bank axis is 0.935 (Table 15). So in ATTRIBUTETABLE the fixed push sets where the policy goes, as the local push does, rather than how strongly it reads the goal cue. In SORTZONES, every fixed push stays at 0.000 at $\beta = 0 . 2 5$ , including the local direction held fixed, which reaches only 0.063 at $\beta = 1$ , while the same direction rebuilt at every step reaches 0.910. We read this as a difference in what is being chosen. In ATTRIBUTETABLE the competition is mostly a left or right hand movement, the same contrast in every scene, so one push along it carries part of the effect. In SORTZONES the decision is where to push the cube, so the useful direction depends on where the cube and the hand are, which changes as the hand moves. In WALKTOFLAG the sign-gated push and the local direction held fixed raise reach by about 0.1 at $\beta = 0 . 2 5$ , to 0.479 and 0.466 against 0.378 unsteered and 0.375 for random pushes, while the direction rebuilt at every step reaches 0.544 at the same dose and 0.992 at $\beta = 1$ . On the native set, where the agent starts in its training pose and the two objects need not be in view, no fixed push moves GM reach (all between 0.273 and 0.339 up to $| \beta | = 1$ , against 0.312 unsteered), which we read as the start state carrying too little of the conflict for a push computed there to steer the rest of the rollout. The start-state direction is consistent with this, with a median norm of 0.43 against 8.11 on the forced-view set, and its Fisher projection agrees with the conflict sign in only about half of the native start states.

Non-GM policies show the same axis from the other side (Figure 36, bottom). The reversed signgated push removes goal pursuit, lowering reach to 0.273 at $\beta = 0 . 2 5$ and 0.027 at $\beta = 0 . 5$ on the black background, and from 0.983 to 0.268 in ATTRIBUTETABLE and from 0.943 to 0.680 in SORTZONES at $\beta = 0 . 2 5$ . WALKTOFLAG is the exception, with Non-GM policies at the ceiling under every fixed push, on the forced-view set (reversed sign-gated push 1.000 at $\beta = 0 . 2 5$ and 0.953 at $\beta = 1 )$ and on the native set (between 0.919 and 0.971 up to $| \beta | = 1$ , against 0.961 unsteered), while the reversed direction rebuilt at every step lowers forced-view reach to 0.542 at $\beta = 1$ . So in WALKTOFLAG a push held fixed from the start state cannot remove goal pursuit, consistent with the weak shared axis. In the corrective direction, every fixed push lowers ATTRIBUTE MAZE reach to about the wall-set ceiling of 0.48, because a policy that already chooses the goal has nothing to gain, and a fixed push collapses its rollouts into one repeated movement. The shared axis is scoped to the canonical UP-versus-RIGHT conflict class. As the diversified geometry analysis shows, it should not be read as one universal goal-versus-proxy direction, and steering was not evaluated beyond this class.

How much control survives compression onto a shared axis? Control survives only where the geometry found a global axis, and how much survives follows how clean that geometry is (Appendix C.5). In ATTRIBUTE MAZE, where one action contrast carries 74% to 86% of the Fisher energy of $d _ { \mathrm { c o m p } }$ , the fixed push along the axis does as well as the local direction held fixed. In AT-TRIBUTETABLE, where the left-right gripper contrast carries 57%, it carries part of the effect, and in SORTZONES and WALKTOFLAG, whose axes are weak, the fixed pushes do little. The sign-gated push works and the unsigned push fails because conflicts of opposite orientation point in opposite directions on the axis, as the geometry showed. Without the sign, one fixed push helps the states of one orientation and hurts the mirrored ones, so it barely moves average reach in either direction. The Fisher-projected push keeps the axis and the sign but is slightly weaker than the sign-gated push (0.560 against 0.587 in ATTRIBUTE MAZE, 0.435 against 0.479 in WALKTOFLAG), and clearly weaker in ATTRIBUTETABLE (0.108 against 0.300). It retains only the part of $d _ { \mathrm { c o m p } } ( s _ { 0 } )$ that the policy’s sensitivity at $s _ { 0 }$ assigns to the axis and drops the rest, whereas the sign-gated push puts the full norm of $d _ { \mathrm { c o m p } } ( s _ { 0 } )$ on the axis. At the start states of the steered GM policies, the Fisher-projected push is 0.73 times the sign-gated push in ATTRIBUTE MAZE, 0.68 times in WALKTOFLAG and 0.30 times in ATTRIBUTETABLE (median over states, mean over checkpoints, and 0.67 in SORTZONES, where no fixed push moves reach). Both pushes lie along $u _ { \mathrm { c o m p } } ,$ , so the ratio is the same in the Euclidean and in the Fisher norm. In ATTRIBUTETABLE and WALKTOFLAG the Fisher push also points against the conflict sign at about a tenth of the start states, and in ATTRIBUTE MAZE at none. So the Fisher push is the smaller push along the same direction, about a third of the sign-gated push where it loses the most, and it loses the most where the axis carries the least of the competition.

Implication for recoverability. Steering asks whether the native policy can be redirected using structure it already reads. It can wherever goal and proxy compete at the decision, and there the competition contrast is a direct causal handle on which location the policy goes to. It cannot where the goal is lost in the representation (MAZE, LOCATIONREACH) or reaches the policy without being used (PUSHTOMARKER, COIN), even though the same construction remains causal in matched Non-GM policies. These nulls are partly expected from Section C.3: the steering directions are built from world edits to which the GM policies barely respond $( Q = 0 . 0 0 1$ in MAZE, 0.021 in COIN). What steering adds is the closed-loop test, together with the Non-GM positive control showing that the same construction does redirect a policy that reads the goal. Where it works, steering redirects the policy using information inside the agent, a form of recovery but not a recovery of navigation. This leaves open whether the relevant goal information is absent or merely unused by the existing downstream computation. We therefore next freeze the representation and retrain the readout.

## D.2 RECOVERABILITY BY REWIRING

## D.2.1 DESIGN

Motivation. The preceding steering experiments ask whether intended-goal behavior can be recovered through the agent’s existing downstream policy. Rewiring asks a complementary question. Even when the native downstream computation does not use the goal structure successfully, the goal may still be represented in a form from which a new downstream policy can recover goal-directed behavior. We therefore freeze the representation and progressively rebuild the downstream computation, from replacing only the policy head (Kirichenko et al., 2023) to relearning the transformation from the final spatial representation to actions.

Ifthefrozen representation still contains information in aform usefulfor goal-directed control, can rebuilding the downstream wiring recover the intended behavior?

Frozen-representation rewiring. For representation site $m ,$ we freeze the trained network up to $h ^ { ( m ) }$ and replace the learned policy computation downstream of that site. We denote the freshly

initialized downstream policy by

$$
\widetilde { \pi } _ { \phi } ^ { ( m ) } ( \cdot \mid h ) ,
$$

where $\phi$ contains the newly optimized downstream policy parameters. The rewired policy is

$$
{ \widetilde \pi } _ { \phi } ^ { ( m ) } ( \cdot \mid s ) = { \widetilde \pi } _ { \phi } ^ { ( m ) } \Big ( \cdot \mid h ^ { ( m ) } ( s ) \Big ) ,
$$

where $h ^ { ( m ) } ( s )$ is always produced by the frozen representation of the original GM or Non-GM checkpoint. Figure $^ { 3 7 }$ shows the three arms, the head rewire from $h _ { \mathrm { f c } }$ , the fixed random remix and the spatial rewire from $h _ { \mathrm { p r e } }$

Our primary comparison uses the two representation sites closest to action selection. The first is the shared fully connected representation

$$
h _ { \mathrm { f c } } ( s ) ,
$$

which is read directly by the policy head. Rewiring at this site therefore keeps the entire learned trunk, including the final shared transformation, fixed and learns only a new policy readout.

We additionally rewire one step earlier, from the final spatial representation

$$
h _ { \mathrm { p r e } } ( s ) = \mathrm { v e c } \Big ( \mathrm { R e L U } \Big ( H ^ { ( 3 ) } ( s ) \Big ) \Big ) .
$$

This is the post-ReLU representation immediately before the learned fully connected transformation.   
Rewiring from $h _ { \mathrm { p r e } }$ therefore replaces both that transformation and the policy head.

The comparison between these sites is especially informative because $h _ { \mathrm { p r e } }$ still retains the spatially indexed structure produced by the convolutional trunk. Flattening only changes the indexing of the tensor, so every component of $h _ { \mathrm { p r e } }$ still corresponds to a particular channel and spatial location in the final feature map. By contrast, the learned map from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ mixes and compresses these spatial features before action selection. Comparing recovery at the two sites therefore asks whether goalrelevant structure that remains usable in the final spatial representation remains similarly accessible after this learned late transformation.

We use the post-ReLU spatial representation because it is the representation actually presented to the downstream computation. The native network passes

$$
\mathrm { R e L U } \Big ( H ^ { ( 3 ) } ( s ) \Big )
$$

to the subsequent fully connected layer rather than the pre-activation tensor $H ^ { ( 3 ) } ( s )$ itself. A pre-ReLU intervention would therefore expose variation that the trained network normally removes before downstream action selection and would answer a different question.

The two architectures share this late structure (Appendix B.1), and the two sites are defined in the same way in both. In the Procgen agents the trunk is the IMPALA network (Espeholt et al., 2018). Its third block gives $H ^ { ( 3 ) } ( s ) \in \bar { \mathbb { R } } ^ { 3 2 \times 8 \times 8 }$ , so $h _ { \mathrm { p r e } } \in \mathbb { R } ^ { 2 0 4 8 }$ , a learned linear layer with ReLU maps $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } } \in \mathbb { R } ^ { 2 \breve { 5 } 6 }$ , and a linear policy head maps $h _ { \mathrm { f c } }$ to logits over $| { \mathcal { A } } | = 1 5$ actions. In the robot agents the trunk is the Nature CNN (Mnih et al., 2015). Its third convolution gives $H ^ { ( 3 ) } ( s ) \in \mathbb { R } ^ { 6 4 }$ ×12×12 in the arm tasks and $\mathbb { R } ^ { 6 4 \times 8 \times 8 }$ in WALKTOFLAG, so $h _ { \mathrm { p r e } } \in \mathbb { R } ^ { 9 2 1 6 }$ and $\mathbb { R } ^ { 4 0 9 6 }$ , a learned linear layer with ReLU maps $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } } \in \mathbb { R } ^ { 2 5 6 }$ , and a two-layer policy head with 512 hidden units maps $h _ { \mathrm { f c } }$ to the mean of a Gaussian over the 8-dimensional arm action or the 3-dimensional walking command of WALKTOFLAG. The robot head also reads the body state, which passes through its own linear layer to 256 dimensions and is concatenated with $h _ { \mathrm { f c } }$ . This state layer stays frozen in every rewire, so the rewires differ from the Procgen ones only in the head that is retrained.

Behavior-cloning objective. We train the new downstream computation by behavior cloning (Pomerleau, 1991) rather than by continuing policy optimization with PPO (Schulman et al., 2017). This isolates whether the frozen representation can support the desired action mapping from the additional exploration, credit assignment, and representation learning introduced by another round of reinforcement learning. It also lets us provide the same target behavior directly to GM and Non-GM representations.

For each rewiring arm, we construct

$$
\mathcal { D } _ { \mathrm { B C } } = \{ ( s _ { i } , q _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

![](images/853363653b5e18d5c5cf5e6179cf5185485df50fe31d05730cc59064df42b7cc.jpg)  
Figure 37: Frozen-representation rewiring. The diagram shows the Procgen IMPALA network. The robot networks follow the same scheme with a Nature CNN trunk instead (Appendix B.1). What the figure shows is which downstream part each rewire relearns, not the trunk, which is frozen in every arm. Grey components are frozen and pink components are freshly initialized and trained. A, the head rewire retains the learned $h _ { \mathrm { f c } }$ representation and learns only a new policy head. B, the fixed random remix replaces the learned $h _ { \mathrm { p r e } }  h _ { \mathrm { f c } }$ transformation by a fixed random 2048 → 256 map and learns only the policy head. C, the spatial rewire starts from $h _ { \mathrm { p r e } }$ and relearns both the intermediate transformation and policy head. The value head is not used for behavior cloning.

where

$$
q _ { i } ( a ) \geq 0 , \qquad \sum _ { a \in \mathcal { A } } q _ { i } ( a ) = 1
$$

defines the target action distribution at state $s _ { i }$ . The frozen representation presented to the new downstream policy is

$$
z _ { i } ^ { ( m ) } = h ^ { ( m ) } ( s _ { i } ) .
$$

We optimize

$$
\phi ^ { \star } = \arg \operatorname* { m i n } _ { \phi } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { \mathrm { B C } } \left( \widetilde { \pi } _ { \phi } ^ { ( m ) } ( \cdot \mid z _ { i } ^ { ( m ) } ) , q _ { i } \right) ,
$$

with

$$
\mathcal { L } _ { \mathrm { B C } } \left( \widetilde { \pi } _ { \phi } ^ { ( m ) } , q \right) = - \sum _ { a \in \mathcal { A } } q ( a ) \log \widetilde { \pi } _ { \phi } ^ { ( m ) } ( a ) .
$$

The robot actions are continuous. There the new head outputs the action mean $\mu _ { \phi } ( z _ { i } ^ { ( m ) } ) \in \mathbb { R } ^ { d _ { \mathcal { A } } }$ , the target is the teacher’s mean action $\mu _ { \mathrm { T } } ( s _ { i } )$ , and the loss is their squared distance,

$$
\mathcal { L } _ { \mathrm { B C } } \left( \mu _ { \phi } ( z _ { i } ^ { ( m ) } ) , \mu _ { \mathrm { T } } ( s _ { i } ) \right) = \frac { 1 } { d _ { \mathcal { A } } } \sum _ { j = 1 } ^ { d _ { \mathcal { A } } } \left( \frac { \mu _ { \phi , j } ( z _ { i } ^ { ( m ) } ) - \mu _ { \mathrm { T } , j } ( s _ { i } ) } { \sigma _ { \mathrm { T } , j } } \right) ^ { 2 } ,
$$

where $\sigma _ { \mathrm { T } , j }$ is the teacher’s state-independent action standard deviation in dimension $j ,$ so the loss is proportional to the Gaussian negative log-likelihood of the teacher’s mean action at the teacher’s variance, up to an additive constant.

We use two forms of supervision where available. Under teacher supervision, $q _ { i }$ places all mass on the greedy action of the competent Non-GM checkpoint used to generate the corresponding behavior-cloning bank. This favours Non-GM rewires at $h _ { \mathrm { f c } } ,$ , where the representation already encodes the teacher’s own actions. The oracle rows avoid this and show the same ordering. The robot tasks use teacher supervision only, with no oracle targets. The teacher of each robot rewire is a Non-GM agent other than the one paired with it, so no Non-GM rewire clones its own policy, and its target is the teacher’s mean action $\mu _ { \mathrm { T } } ( s _ { i } )$ . Under oracle supervision, the maze graph defines

$$
A ^ { \star } ( s _ { i } ) ,
$$

the set of graph-exact first actions on a shortest path toward the intended goal, and

$$
q _ { i } ( a ) = { \frac { 1 \{ a \in A ^ { \star } ( s _ { i } ) \} } { | A ^ { \star } ( s _ { i } ) | } } .
$$

In the MAZE bank, $\left| A ^ { \star } ( s ) \right| = 1$ for every state because the generated mazes have unique shortest paths, so this reduces to ordinary hard cross-entropy. In ATTRIBUTE MAZE, ties can occur and the set-valued target is substantive. Shortest paths in ATTRIBUTE MAZE never pass through the other candidate object, so an oracle path cannot end the episode at the proxy. We use teacher targets in all eight environments and oracle targets in MAZE and ATTRIBUTE MAZE as an additional variant. Figure 38 shows the oracle variant for these two, since its targets follow the intended goal exactly.

The value head is not part of the behavior-cloning objective and does not affect action selection. Implementations differ slightly in whether its parameters are included in the optimizer, but because no value loss is applied they receive no behavior-cloning gradient.

State distributions and training. For teacher supervision, MAZE, COIN, and ATTRIBUTE MAZE use balanced behavior-cloning banks with equal contributions from GM and competent Non-GM rollouts. Within each assay, the same bank is shared across matched GM and Non-GM rewires and representation sites. The robot banks are described in Appendix E.5. In ATTRIBUTETABLE and SORTZONES the rewires clone only the episodes drawn from the Non-GM training distribution, so no cloned state shows the training co-occurrence of goal and proxy.

Where graph-exact supervision is available, states are additionally labelled with optimal shortestpath actions toward the intended goal. In MAZE, teacher and oracle labels are attached to the same state bank, so the two supervision schemes can be compared on an identical state distribution. In ATTRIBUTE MAZE, oracle-labelled states must be generated separately because the graph information required for the oracle is not retained in the teacher bank. We therefore do not interpret teacher–oracle differences in ATTRIBUTE MAZE as isolated effects of the supervision source.

We train all downstream policies for 20 epochs with Adam (Kingma & Ba, 2015) at learning rate $3 \times 1 0 ^ { - 4 }$ , batch size 256, gradient clipping at norm 0.5, and an 80/20 train–validation split, which in the robot tasks is drawn by episode. Because the behavior-cloning banks differ in size across assays, the resulting number of optimizer steps also differs. All GM–Non-GM comparisons are therefore made within a matched assay, where the training budget is identical. Training and validation crossentropy together with one-step validation accuracy are recorded as optimization diagnostics rather than measures of behavioral recovery. For teacher targets, validation accuracy denotes exact action agreement. For set-valued oracle targets, it denotes whether the predicted action belongs to $A ^ { \star } ( s )$ so the two accuracy measures are not compared directly.

Closed-loop recovery. One-step validation performance does not establish that the rewired policy reaches the intended goal under its own rollout distribution. We therefore evaluate recovery using the task-level reach quantity introduced above.

We use the behavioral objective of Section B. $\Gamma ^ { \star } ( \tau ) = 1$ when trajectory τ reaches the intended goal, and the intended-goal success of the rewired policy on the held-out evaluation distribution $\boldsymbol { \Lambda } _ { \mathrm { e v a l } }$ is $J ^ { \star } ( { \widetilde \pi } _ { \phi } ^ { ( m ) } ; \Lambda _ { \mathrm { e v a l } } )$ . We estimate it by the goal reach

$$
\hat { J } ^ { \star } \big ( \widetilde { \pi } _ { \phi } ^ { ( m ) } \big ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Gamma ^ { \star } ( \tau _ { i } ) , \qquad \tau _ { i } \sim P _ { \widetilde { \pi } _ { \phi } ^ { ( m ) } } ( \tau \mid s _ { 0 } = s _ { i } ) ,
$$

the fraction of N rollouts from held-out initial states $s _ { 1 } , \ldots , s _ { N }$ that reach the goal, as for the native policies (Appendix F). Every rewired policy acts greedily (deterministically in the robot tasks). For MAZE, $\Gamma ^ { \star } \bar { ( \tau ) } = 1$ when the trajectory reaches the cheese. For COIN, it indicates reaching the goal coin. For ATTRIBUTE MAZE, it indicates reaching the intended object. In LOCATIONREACH it is the first gripper contact with the ball, within 3 cm and before the training corner, in 50 steps, on the 500 off-path episodes of each seed pair. In PUSHTOMARKER it is the cube resting within 2.5 cm of the marker after 100 steps, on the 512 conflict scenes. In ATTRIBUTETABLE it is the first touch of the goal object before the proxy, and in SORTZONES the first entry of the cube into the crossed pad before the plain one, both in 100 steps on the 512 world-edit scenes (Appendix E). In WALKTOFLAG it is the first object reached, flag before tower, within 200 commands on the first 256 layouts of the conflict-test bank, with the native start pose (Appendix E).

Where the task has two candidate objects (ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES, WALKTOFLAG), intended-goal reach admits an exact decomposition into whether the policy reaches either candidate and which candidate it selects conditional on doing so. Let

$$
D = { \frac { N _ { \mathrm { g o a l } } + N _ { \mathrm { p r o x y } } } { N } }
$$

denote the decisive fraction, and let

$$
\mathrm { P r } ( \mathrm { g o a l } \mid \mathrm { d e c } ) = \frac { N _ { \mathrm { g o a l } } } { N _ { \mathrm { g o a l } } + N _ { \mathrm { p r o x y } } }
$$

denote intended-goal preference conditional on reaching either candidate. Then

$$
\hat { J } ^ { \star } = D \operatorname* { P r } ( \mathrm { g o a l } | \operatorname* { d e c } )
$$

holds exactly, as in Equation equation 6. So the two factors answer different questions. D measures whether the policy successfully executes a trajectory to either candidate, whereas Pr(goal | dec) measures target selection once the policy becomes decisive. The decomposition is what separates a rewire that selects the goal from one that only executes better, and we use it in every comparison against the random trunk.

Evaluation is performed on held-out levels disjoint from the behavior-cloning range, and in the robot tasks on scenes and start poses disjoint from the cloning banks. We use 300 episodes with a 1000- step cap in MAZE, 1000 episodes with a 1000-step cap in COIN, and 1000 episodes with a 256-step cap in ATTRIBUTE MAZE. In ATTRIBUTE MAZE level separation does not imply state separation, since the compact state space puts 98% of evaluation start frames into the oracle cloning bank.

Baselines. A successful rewire reestablishes the use of the goal. To decide when this happens, we compare the rewired policies against five complementary baselines. (1) The untouched native GM and Non-GM checkpoints give the behavior before the intervention and the competence of a policy trained toward the intended goal. (2) The same rewiring applied to the frozen representation of a Non-GM agent is our primary method control. It shows whether the method can reestablish the goal use that the Non-GM agent originally had. (3) The random trunk is a randomly initialized network of the same architecture whose trunk up to $h _ { \mathrm { p r e } }$ is frozen at initialization and never trained. Only the downstream computation, the $h _ { \mathrm { p r e } }  h _ { \mathrm { f c } }$ layer and the policy head, is trained, with the same bank, targets, loss and training as the rewire. We use three such networks, one per GM checkpoint. The random trunk measures whether the trained trunk provides value or whether any trunk lets the readout relearn the task. (4) A goal-blind chance policy tells what a policy that ignores the goal achieves at the rewire’s own competence. In the four two-object tasks it is a coin flip between the two candidates at the decisiveness of the GM rewire from $h _ { \mathrm { p r e } } ^ { \cdot } , 0 . 5 D$ . In LOCATIONREACH and PUSHTOMARKER, where the goal position varies continuously, it is phantom-goal reach, the rate at which the GM rewire’s own paths hit 20 goal positions drawn from the same distribution as the real one. In COIN it is the untrained head with sampled actions, and in MAZE a uniform random walk over the 15 actions, since no competent goal-blind policy is defined without a second target. (5) Parameter count is a confounder, since in some environments a certain capacity suffices to relearn the whole structure. A fixed random remix therefore freezes a random 2048→256 map (9216→256 in the arm tasks and 4096 → 256 in WALKTOFLAG) in place of the learned layer and trains only the downstream head. It is exactly parameter-matched to the $h _ { \mathrm { f c } }$ rewire while removing the learned $h _ { \mathrm { p r e } }  h _ { \mathrm { f c } }$ transformation.

Parameter counts. The $h _ { \mathrm { f c } }$ rewire trains 257|A| policy parameters, whereas the $h _ { \mathrm { p r e } }$ rewire additionally learns the 2048→256 transformation, adding 524,544 trainable parameters. With $| { \mathcal { A } } | = 1 5$ this gives 3,855 versus 528,399 policy parameters. In MAZE and ATTRIBUTE MAZE, the reinitialized value head is also passed to the optimizer, adding 257 parameters to the recorded counts (4,112 and 528,656, respectively). Because the behavior-cloning loss depends only on policy logits, these parameters receive no gradient and cannot affect action selection. The fixed-remix control has exactly the same recorded trainable count as the corresponding $h _ { \mathrm { f c } }$ rewire in every environment. In the robot agents the $h _ { \mathrm { f c } }$ rewire trains the two-layer head, 266,760 parameters in the arm tasks and 264,195 in WALKTOFLAG, and the $h _ { \mathrm { p r e } }$ rewire adds the 9216 → 256 or 4096 → 256 transformation (2,359,552 and 1,048,832 parameters), for 2,626,312 and 1,313,027. No value parameters are passed to the optimizer, and the fixed remix again matches the $h _ { \mathrm { f c } }$ rewire.

Success criterion. We need to define when a rewire recovers the goal, recovers it partly or does not recover it. We compare reaches at the same site and with the same targets. The reference is the lower of the native Non-GM reach (1) and the Non-GM rewire reach (2), since a new rewire can overshoot the native policy. A GM rewire is recovered when it reaches at least 90% of this reference. It is partial when it stays below that but its excess over the floor, the higher of the random trunk (3) and the goal-blind policy (4), has a 95% interval above zero, and none otherwise. The intervals are Welch intervals across the three GM checkpoints and the three random trunks or chance estimates. The Non-GM rewire is needed to rule out a failed method only when recovery fails. We therefore count the method as working when the Non-GM rewire reaches at least 90% of its native policy or the GM rewire is recovered. If recovery works, the method evidently works, so the control is informative only when the experiment fails. Where the method does not work, a GM rewire that is not recovered must be read with care. When a random-trunk rewire reaches the native Non-GM reach, a readout trained on a frozen trunk has enough capacity to solve the environment from any trunk. Goal information in the GM or the Non-GM trunk is then not necessary for recovery. This does not rule out that a recovery from these trunks uses the goal, but it makes it much harder to tell where the recovery comes from, as the readout can sidestep the trunk by relearning the whole environment, with goal direction, in its trained layers. A readout with less capacity separates the trunks again. In that case we therefore also use the strict test of the additional ablations below, a linear readout cloned on a bank without the evaluation start configurations and on 10% and 1% of it, and ask whether the GM trunk exceeds the random trunk there.

Additional ablations. In MAZE and PUSHTOMARKER offline behavior cloning leaves the Non-GM rewire below 90% of its native policy, and cloning can fail through distribution shift (Ross et al., 2011). Both therefore receive three rounds of DAgger at both sites. In MAZE, each round rolls out the current rewired policy on training levels, labels 50k visited states with oracle actions, adds them to the training set and fine-tunes the downstream computation. A matched control adds the same number of frames resampled from the base set, so the two arms take identical optimizer steps and differ only in whether the added states are on-policy. In PUSHTOMARKER, with teacher targets, each round rolls out the current rewired policy from the training half of the start poses on 288 fresh scenes of the Non-GM training distribution and labels all 28,800 visited states with the teacher’s actions. That is 25% of the 115,200-frame cloning bank, the same share as 50k states of the 200k MAZE bank. The matched control again adds as many frames resampled from the base set, and the random trunk receives both treatments too. We judge these arms, like all others, by closed-loop reach rather than by cloning accuracy, since cloning accuracy only measures whether the rewire replicates its targets, not whether it pursues the goal.

A rewire can raise goal reach without using the goal, by sending the policy to a fixed position or to the proxy. As in the location controls of steering (Table 15), we therefore keep the GM trunk and take the goal away (Section D.2.4). We clone a rewire on frames in which the goal cue is masked, and we evaluate the standard rewire with the goal removed, by deleting the cheese (MAZE), hiding the coin (COIN), drawing the goal line as a red gem (ATTRIBUTE MAZE), removing the ball, marker or flag (LOCATIONREACH, PUSHTOMARKER, WALKTOFLAG) or giving the goal the proxy’s cue (two spheres in ATTRIBUTETABLE, no cross on either pad in SORTZONES). We then count arrivals at the former goal location. A rewire that pursues the goal drops to chance or below, while one that goes to a fixed position or to the proxy keeps arriving there. A few further tests serve single environments. In MAZE a random trunk freezes under greedy actions, so all MAZE arms are also evaluated with sampled actions. In PUSHTOMARKER the marker of a conflict scene lies on the path to the table centre, so every arm is also evaluated on the 512 scenes of the control-layout set, in which marker and cube are placed independently and a push towards the centre cannot score (Appendix E). In COIN we split every arm’s episodes by outcome, coin, level end, death or step cap, to separate choosing the coin from surviving the platforming. In ATTRIBUTE MAZE a strict test clones on a bank from which every evaluation start configuration is removed, and on 10% and 1% of it by level, with the relearned head and with a linear readout from $h _ { \mathrm { p r e } }$ or $h _ { \mathrm { f c } }$ , as a weaker readout has less room to compensate for the features it reads. All ATTRIBUTE MAZE arms are repeated with the textured-background checkpoints, on the textured version of the banks, to test whether the ordering of the two sites survives a change of background.

Table 17: Rewiring at a glance. The method works when the Non-GM rewire reaches at least 90% of its native policy (its fraction in brackets) or the GM rewire is recovered, since a recovery shows that the method works. Recovery is that of the GM rewire under the success criterion, with teacher targets, and is in bold where the method works. Oracle targets in MAZE and ATTRIBUTE MAZE give the same verdicts. The DAgger rows use three rounds of on-policy relabeling. Values in Tables 18 and 19.
<table><tr><td rowspan="2">Environment</td><td colspan="2">Method works?</td><td colspan="2">Recovery successful?</td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td> $h _ { \mathrm { f c } }$ </td><td> $h _ { \mathrm { p r e } }$ </td><td> $h _ { \mathrm { f c } }$ </td></tr><tr><td colspan="5">(i) Goal lost in the representation</td></tr><tr><td>MAZE</td><td>no (0.80)</td><td>no (0.64)</td><td>none</td><td>none</td></tr><tr><td>with DAgger</td><td>yes (1.01)</td><td>no (0.78)</td><td>none</td><td>none</td></tr><tr><td>LOCATIONREACH</td><td>yes (0.92)</td><td>yes (0.97)</td><td>none</td><td>none</td></tr><tr><td colspan="5">(ii) Goal reaches the policy but is not used</td></tr><tr><td>PUSHTOMARKER</td><td>no (0.79)</td><td>no (0.77)</td><td>partial</td><td>partial</td></tr><tr><td>with DAgger</td><td>yes (0.92)</td><td>yes (0.92)</td><td>partial</td><td>partial</td></tr><tr><td>COIN</td><td>yes (0.99)</td><td>yes (0.98)</td><td>recovered</td><td>none</td></tr><tr><td colspan="5">(iii) Goal and proxy compete</td></tr><tr><td>ATTRIBUTE MAZE</td><td>yes (0.98)</td><td>yes (0.94)</td><td>recovered</td><td>none</td></tr><tr><td>ATTRIBUTETABLE</td><td>yes (0.91)</td><td>no (0.87)</td><td>partial</td><td>none</td></tr><tr><td>SORTZONES</td><td>yes (0.87)</td><td>no (0.71)</td><td>recovered</td><td>none</td></tr><tr><td>WALKTOFLAG</td><td>yes (1.00)</td><td>yes (0.94)</td><td>recovered</td><td>partial</td></tr></table>

What a rewire can show. A successful rewire shows that the frozen representation suffices for goal-directed control under the downstream function class, supervision and state distribution that we test. A failed rewire is weaker evidence, since it can also reflect limits of the downstream architecture, supervision, optimization or state distribution, which is why we read it together with the Non-GM rewire and DAgger. A successful rewire in turn says something about the agent only if it exceeds what a new readout extracts from an untrained trunk, which already passes the position and appearance of a visible goal object.

## D.2.2 DOES NEW WIRING RECOVER THE GOAL?

From $h _ { \mathrm { p r e } } ,$ , the GM rewire recovers the goal in COIN, ATTRIBUTE MAZE, SORTZONES and WALK-TOFLAG, partly in PUSHTOMARKER and ATTRIBUTETABLE, and not in MAZE or LOCATION-REACH (Tables 17 and 18, Figure 38). Recovery thus lines up with depth. Where goal and proxy compete or the goal reaches the policy without being used, it succeeds fully or partly, and where the goal is lost in the representation it does not. From the later layer $h _ { \mathrm { f c } } ,$ , it recovers the goal nowhere and only partly in PUSHTOMARKER and WALKTOFLAG. The drop at $h _ { \mathrm { f c } }$ follows from what this layer is trained for. It compresses thousands of spatial features into 256 and keeps what the policy acts on, the goal in a Non-GM agent and the proxy in a GM agent, so in a GM agent nothing in training carries the goal through (Section D.2.3).

In MAZE, PUSHTOMARKER and SORTZONES, offline behavior cloning leaves the Non-GM rewire below 90% of its native policy (0.80, 0.79 and 0.87 from $h _ { \mathrm { p r e } } )$ . In SORTZONES the GM rewire is recovered, so the method works there by the success criterion. In MAZE and PUSHTOMARKER it is not, so there the method itself has to be questioned. Three rounds of DAgger give a working method in PUSHTOMARKER at both sites and in MAZE from $h _ { \mathrm { p r e } }$ (Table 19). MAZE still does not recover, while PUSHTOMARKER recovers partly, with the GM rewire from $h _ { \mathrm { p r e } }$ at 0.81 of the Non-GM rewire. SORTZONES needs no DAgger, since its method already works. Figure 39 puts every GM rewire on one scale, its reach above the floor as a fraction of the Non-GM rewire’s.

Table 18: Closed-loop goal reach after rewiring, ordered by where the goal is lost. Goal reach $\hat { J } ^ { \star }$ , mean ± SD across three checkpoints or random trunks, with teacher targets unless marked oracle or DAgger. Native is the untouched policy, remix the fixed random remix (5), and the random trunk (3) is rewired from the same site as the row. Chance is the goal-blind policy (4). Bold GM rewires are recovered and underlined ones partial under the success criterion. <sup>a</sup>Uniform random walk, sampled actions. <sup>b</sup>Freezes under greedy actions. <sup>c</sup>Does not fit the cloning targets. <sup>d</sup>Two of three checkpoints. <sup>e</sup>Cloned on a rebuilt copy of the bank, on which the rewires reproduce within 0.02. $f _ { 0 . 5 } \dot { D }$ at $D = 0 . 6 9 0$ , one estimate. <sup>g</sup>Non-GM rewire below 90% of its native policy. <sup>h</sup>Three rounds of DAgger (Table 19), oracle targets in MAZE.
<table><tr><td rowspan="2">Environment</td><td rowspan="2"> $\mathrm { S i t e }$ </td><td colspan="2">Native</td><td colspan="2">Rewire</td><td colspan="2">Baselines</td></tr><tr><td>GM</td><td>Non-GM</td><td>GM</td><td>Non-GM</td><td>Random trunk</td><td>Chance</td></tr><tr><td colspan="8">(i) Goal lost in the representation</td></tr><tr><td rowspan="4">MAZE</td><td></td><td></td><td> $h _ { \mathrm { p r e } } \quad 0 . 2 1 9 \pm 0 . 0 2 \quad 0 . 3 4 7 \pm 0 . 0 2$ </td><td> $0 . 0 8 9 \pm 0 . 0 1$ </td><td> $0 . 2 7 7 \pm 0 . 0 2 ^ { g }$ </td><td> $0 . 0 1 6 \pm 0 . 0 1 ^ { b , e }$ </td><td> $0 . 2 2 6 ^ { a }$ </td></tr><tr><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 0 3 8 \pm 0 . 0 2$ </td><td> $0 . 1 6 7 \pm 0 . 0 3$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 0 4 8 \pm 0 . 0 0$ </td><td> $0 . 2 2 2 \pm 0 . 0 7 ^ { g }$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 ^ { b , e }$ </td><td></td></tr><tr><td> $h _ { \mathrm { p r e } }$   $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 1 1 6 \pm 0 . 0 2$ </td><td> $0 . 3 0 8 \pm 0 . 0 4 ^ { g }$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 ^ { b }$ </td><td> $0 . 2 2 6 ^ { a }$ </td></tr><tr><td rowspan="3"> ${ \mathrm { D A g g e r } } ^ { h }$ </td><td></td><td></td><td></td><td> $0 . 0 9 2 \pm 0 . 0 2$ </td><td> $0 . 1 7 3 \pm 0 . 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 ^ { b , e }$ </td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 1 1 9 \pm 0 . 0 2$ </td><td> $0 . 2 3 6 \pm 0 . 0 3 ^ { g }$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td></td><td></td><td> $0 . 1 2 8 \pm 0 . 0 2$ </td><td> $0 . 3 5 1 \pm 0 . 0 3$ </td><td></td><td> $0 . 2 2 6 ^ { a }$ </td></tr><tr><td rowspan="3">LOCATIONREACH</td><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 1 2 2 \pm 0 . 0 2$ </td><td> $0 . 2 7 1 \pm 0 . 0 3 ^ { g }$ </td><td></td><td></td></tr><tr><td></td><td> $\begin{array} { r l } { h _ { \mathrm { p r e } } } & { { } 0 . 0 0 0 \pm 0 . 0 0 } \end{array}$ </td><td> $0 . 9 9 9 \pm 0 . 0 0$ </td><td> $0 . 3 7 3 \pm 0 . 0 8$ </td><td> $0 . 9 2 1 \pm 0 . 0 5$ </td><td> $0 . 1 7 9 \pm 0 . 0 9$ </td><td> $0 . 0 9 6 \pm 0 . 0 3$ </td></tr><tr><td> $\operatorname { r e m i x }$   $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 1 1 7 \pm 0 . 0 4$   $0 . 1 3 7 \pm 0 . 0 9$ </td><td> $0 . 4 6 8 \pm 0 . 1 7$   $0 . 9 7 3 \pm 0 . 0 2$ </td><td> $0 . 1 4 6 \pm 0 . 0 5$ </td><td></td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="3">PUSHTOMARKER</td><td></td><td>(ii) Goal reaches the policy but is not used</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\begin{array} { r l }  \mathbf { \bar { \begin{array} { r l } { \mathbf { \bar { h } } _ { \mathrm { p r e } } } & { { } 0 . 0 0 0 \pm 0 . 0 0 } \end{array} } } \end{array}$ </td><td> $0 . 8 8 9 \pm 0 . 0 4$ </td><td> $0 . 3 1 4 \pm 0 . 0 7$   $\overline { { 0 . 0 6 8 \pm 0 . 0 3 } }$ </td><td> $0 . 7 0 2 \pm 0 . 1 0 ^ { g }$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 ^ { c }$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $\underline { { 0 . 1 3 2 \pm 0 . 0 0 } }$ </td><td> $0 . 1 5 4 \pm 0 . 0 7$ </td><td></td><td></td></tr><tr><td rowspan="3"> ${ \mathrm { D A g g e r } } ^ { h }$ </td><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 6 6 1 \pm 0 . 1 6$ </td><td> $0 . 6 8 2 \pm 0 . 1 5 ^ { g }$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 ^ { c }$ </td><td></td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td></td><td></td><td>0.326 ± 0.09</td><td> $0 . 8 1 6 \pm 0 . 0 8$ </td><td> $0 . 1 3 5 \pm 0 . 0 7$ </td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td></td><td> $0 . 8 1 9 \pm 0 . 0 8$ </td><td> $0 . 0 0 9 \pm 0 . 0 1$ </td><td></td></tr><tr><td rowspan="3">COIN</td><td></td><td> $\begin{array} { r l } { h _ { \mathrm { p r e } } } & { { } 0 . 3 1 6 \pm 0 . 0 1 } \end{array}$ </td><td> $0 . 8 9 6 \pm 0 . 0 0$ </td><td> $\mathbf { 0 . 8 4 3 \pm 0 . 0 1 }$ </td><td> $0 . 8 8 8 \pm 0 . 0 1$ </td><td> $0 . 6 4 6 \pm 0 . 0 2$ </td><td> $0 . 6 2 5 \pm 0 . 0 0 ^ { d }$ </td></tr><tr><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 6 7 6 \pm 0 . 0 3$ </td><td> $0 . 8 2 5 \pm 0 . 0 3$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 6 6 6 \pm 0 . 0 7$ </td><td> $0 . 8 8 1 \pm 0 . 0 2$ </td><td> $0 . 5 1 1 \pm 0 . 0 0 ^ { e }$ </td><td></td></tr><tr><td colspan="8">(iii) Goal and proxy compete</td></tr><tr><td rowspan="4">ATTRIBUTE MAZE</td><td></td><td> $h _ { \mathrm { p r e } } \quad 0 . 3 5 8 \pm 0 . 0 2$ </td><td> $0 . 6 9 6 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 6 8 9 \pm 0 . 0 3 }$ </td><td> $0 . 6 8 4 \pm 0 . 0 2$ </td><td> $0 . 6 3 1 \pm 0 . 0 3 ^ { e }$ </td><td> $0 . 3 4 5 ^ { f }$ </td></tr><tr><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 2 6 2 \pm 0 . 0 2$ </td><td> $0 . 4 4 0 \pm 0 . 0 5$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 2 7 3 \pm 0 . 0 2$ </td><td> $0 . 6 5 3 \pm 0 . 0 4$ </td><td> $0 . 2 2 4 \pm 0 . 0 4 ^ { e }$ </td><td></td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td></td><td></td><td> $\mathbf { 0 . 8 7 1 \pm 0 . 0 2 }$ </td><td> $0 . 9 0 8 \pm 0 . 0 1$ </td><td> $0 . 8 4 0 \pm 0 . 0 5$ </td><td> $0 . 4 5 0 \pm 0 . 0 1$ </td></tr><tr><td rowspan="3"></td><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 4 8 1 \pm 0 . 0 7$ </td><td> $0 . 6 2 6 \pm 0 . 0 1$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 4 7 8 \pm 0 . 0 6$ </td><td> $0 . 7 9 3 \pm 0 . 0 1$ </td><td> $0 . 2 9 6 \pm 0 . 0 3 ^ { e }$ </td><td></td></tr><tr><td></td><td> $\begin{array} { r l } { h _ { \mathrm { p r e } } } & { { } 0 . 0 3 7 \pm 0 . 0 2 } \end{array}$ </td><td> $0 . 9 8 6 \pm 0 . 0 1$ </td><td> $0 . 7 5 7 \pm 0 . 0 3$ </td><td> $0 . 8 9 8 \pm 0 . 0 4$ </td><td> $0 . 4 6 2 \pm 0 . 1 2$ </td><td> $0 . 4 1 1 \pm 0 . 0 2$ </td></tr><tr><td rowspan="3"></td><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 3 6 1 \pm 0 . 1 0$ </td><td> $0 . 3 9 1 \pm 0 . 1 6$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 5 4 5 \pm 0 . 1 2$ </td><td> $0 . 8 5 4 \pm 0 . 0 6 ^ { g }$ </td><td> $0 . 1 7 3 \pm 0 . 0 3$ </td><td></td></tr><tr><td></td><td> $\begin{array} { r l } { h _ { \mathrm { p r e } } } & { { } 0 . 0 0 0 \pm 0 . 0 0 } \end{array}$ </td><td> $0 . 9 4 6 \pm 0 . 0 1$ </td><td> ${ \bf 0 . 7 5 3 \pm 0 . 0 2 }$ </td><td> $0 . 8 2 1 \pm 0 . 0 1 ^ { g }$ </td><td> $0 . 1 5 1 \pm 0 . 0 4$ </td><td> $0 . 3 7 7 \pm 0 . 0 1$ </td></tr><tr><td rowspan="3"></td><td> $\operatorname { r e m i x }$ </td><td></td><td></td><td> $0 . 2 7 0 \pm 0 . 0 7$ </td><td> $0 . 3 8 7 \pm 0 . 0 6$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $0 . 3 4 8 \pm 0 . 0 7$ </td><td> $0 . 6 6 9 \pm 0 . 1 2 ^ { g }$ </td><td> $0 . 0 3 6 \pm 0 . 0 5$ </td><td></td></tr><tr><td></td><td> $\begin{array} { r l } { h _ { \mathrm { p r e } } } & { { } 0 . 2 7 0 \pm 0 . 0 1 } \end{array}$ </td><td></td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 5 }$ </td><td> $0 . 9 6 5 \pm 0 . 0 1$ </td><td> $0 . 1 2 2 \pm 0 . 0 1$ </td><td> $0 . 4 6 4 \pm 0 . 0 1$ </td></tr><tr><td rowspan="3">WALKTOFLAG</td><td>remix</td><td></td><td> $0 . 9 6 4 \pm 0 . 0 1$ </td><td> $0 . 4 6 2 \pm 0 . 1 2$ </td><td> $0 . 8 8 8 \pm 0 . 0 3$ </td><td></td><td></td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td></td><td></td><td> $\underline { { 0 . 6 9 7 \pm 0 . 0 3 } }$ </td><td> $0 . 9 0 5 \pm 0 . 1 2$ </td><td> $0 . 0 4 9 \pm 0 . 0 1$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

In the following we go through each environment in depth.

In some environments the random trunk comes close to the GM rewire, most clearly in ATTRIBUTE MAZE, where a readout with enough capacity relearns the simple structure of the environment from any trunk. Behavior cloning hands the readout the goal through its targets, and a visible goal object is passed on by random convolutional features too (grey bars). GM agents keep an advantage (0.684 against 0.631 on the same bank), and with less capacity, a linear readout, the two separate.

MAZE: no recovery. In MAZE the GM rewire fails. With teacher targets it reaches the cheese in 0.089 of levels from $h _ { \mathrm { p r e } }$ and 0.048 from $h _ { \mathrm { f c } } .$ , below its own native policy (0.219) and a uniform random walk (0.226), while the Non-GM rewire reaches 0.277 and 0.222, against its native 0.347 (Table 18, first rows). With oracle targets the GM rewire does not recover the goal either (0.116 at $h _ { \mathrm { p r e } }$ and 0.119 at $h _ { \mathrm { f c } }$ , against 0.308 and 0.236 for the Non-GM rewires, Table 18). Sampled instead of greedy actions bring the GM oracle rewire from $h _ { \mathrm { p r e } }$ only to the level of the random walk (0.235), against 0.582 for Non-GM. The method itself underperforms here, since the Non-GM rewire stays below 90% of its native performance, closest with oracle targets at $h _ { \mathrm { p r e } }$ (0.89) and furthest at $h _ { \mathrm { f c } }$

![](images/ad4b752f26995faa3e7afaf4fa40e29703f13446a20f6a5aeeb48f94b9869043.jpg)  
Figure 38: Closed-loop goal reach after rewiring, ordered by where the goal is lost. Goal reach $\hat { J } ^ { \star }$ of the GM (red) and Non-GM (teal) rewires from $h _ { \mathrm { p r e } }$ and $h _ { \mathrm { f c } }$ , of the fixed random remix (open bars) and of the random trunk (grey), with oracle targets in MAZE and ATTRIBUTE MAZE and teacher targets in the other six environments (Table 18). Bars are means, whiskers 95% tintervals across the three checkpoints or random trunks, and dots individual checkpoints where they are available. Dashed lines mark the native GM policy, with the band below it, and the native Non-GM policy, with the band above it. The dotted line is chance. The random trunks of MAZE, ATTRIBUTE MAZE and COIN at $h _ { \mathrm { f c } }$ are cloned on the rebuilt bank (Table 18, note e).

To rule out the method, three rounds of DAgger on states along the agents’ own trajectories bring the Non-GM rewire from $h _ { \mathrm { p r e } }$ back to its native performance (0.351 against 0.347, Table 19), which makes the method viable there. The matched extra training alone does the same (0.357). From $h _ { \mathrm { f c } }$ the Non-GM rewire still falls short (0.271). The GM rewires stay below even the native GM reach (0.128 and 0.122 against 0.219), so the goal information in the MAZE agent cannot be turned into goal use by a new wiring. Cloning accuracy cannot show this failure, since in 95% of MAZE states the step toward the cheese is also the step toward the top-right corner, and only the few forks where the two differ decide whether a rollout reaches the cheese.

LOCATIONREACH: no more than randomfeatures. The Non-GM rewire nearly matches the native policy, but the GM rewire reaches the ball in only 0.373 of episodes from $h _ { \mathrm { p r e } }$ and 0.137 from $h _ { \mathrm { f c } }$ That is about the random trunk at $h _ { \mathrm { f c } } .$ , and above it at $h _ { \mathrm { p r e } }$ by a difference whose interval includes zero, because a random convolution passes the position of a single salient ball almost as well (Appendix C.1).

PUSHTOMARKER: partial recovery. PUSHTOMARKER differs from LOCATIONREACH. The GM rewire from $h _ { \mathrm { p r e } }$ pushes the cube onto the marker in 0.314 of conflict scenes, far above the random trunk (0.014), and as often in the control-layout scenes, where a push toward the table centre cannot score (Figure 38, PushToMarker). So the new readout uses goal information that only the GM trunk provides. It reaches about 45% of the Non-GM rewire (0.702), a partial recovery, consistent with a goal that is weakly wired into the GM policy. The Non-GM rewire itself reaches only 0.79 of its native policy, so this comparison must be read with care. Behavior cloning fits the rewire only on the states of the bank, and on the GM rewire’s own trajectories its errors compound. Three rounds of onpolicy relabeling (Table 19) lift the GM rewire from $h _ { \mathrm { p r e } }$ from 0.296 to 0.661, while the matched control stays at 0.311 and the random trunk gains only to 0.135. So most of the gain uses goal information in the GM $h _ { \mathrm { p r e } }$ that cloning could not exploit. On the control-layout set the GM rewire rises from 0.333 to 0.793, so the gain is not a push toward the table centre. The Non-GM rewire reaches 0.816 under DAgger, 0.92 of its native policy, so the method no longer underperforms, and the GM rewire reaches 0.81 of it, still a partial recovery. Normalized, it moves from 0.44 under cloning to 0.77 under DAgger, with an interval from 0.25 to 1.29, against 0.39 for the matched control (Figure 39). The three GM checkpoints spread widely, 0.848, 0.592 and 0.543, and only the first reaches the mean Non-GM rewire. At $h _ { \mathrm { f c } }$ the gain is small, from 0.135 to 0.326 against 0.819 for the Non-GM rewire, again partial. The goal-removed controls of Table 20 were not repeated after DAgger.

Table 19: Rewiring with DAgger. Goal reach after three rounds of on-policy relabeling (DAgger) or of matched extra training, with oracle targets in MAZE (300 held-out levels) and teacher targets in PUSHTOMARKER (512 conflict scenes). Mean ± SD across three checkpoints or random trunks. Round 0 is the rewire before the added rounds. In MAZE it is trained on states visited by the freshly initialized downstream policy, so it differs from Table 18, and both arms then share a base set along the native agent’s trajectories. Underlined GM cells are partial under the success criterion, and none is recovered.
<table><tr><td>Site</td><td>Cohort</td><td>Native</td><td>Round 0</td><td>DAgger</td><td>Matched</td></tr><tr><td colspan="6">MAZE, oracle targets</td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td>GM</td><td> $0 . 2 1 9 \pm 0 . 0 2$ </td><td> $0 . 1 1 2 \pm 0 . 0 2$ </td><td> $0 . 1 2 8 \pm 0 . 0 2$ </td><td> $0 . 1 1 8 \pm 0 . 0 1$ </td></tr><tr><td></td><td>Non-GM</td><td> $0 . 3 4 7 \pm 0 . 0 2$ </td><td> $0 . 1 8 7 \pm 0 . 0 3$ </td><td> $0 . 3 5 1 \pm 0 . 0 3$ </td><td> $0 . 3 5 7 \pm 0 . 0 3$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td>GM</td><td> $0 . 2 1 9 \pm 0 . 0 2$ </td><td> $0 . 1 1 4 \pm 0 . 0 1$ </td><td> $0 . 1 2 2 \pm 0 . 0 2$ </td><td> $0 . 0 9 8 \pm 0 . 0 1$ </td></tr><tr><td></td><td>Non-GM</td><td> $0 . 3 4 7 \pm 0 . 0 2$ </td><td> $0 . 2 4 6 \pm 0 . 0 2$ </td><td> $0 . 2 7 1 \pm 0 . 0 3$ </td><td> $0 . 1 1 1 \pm 0 . 0 3$ </td></tr><tr><td colspan="6">PUSHToMARKER, teacher targets</td></tr><tr><td> $h _ { \mathrm { p r e } }$ </td><td>GM</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 9 6 \pm 0 . 0 6$ </td><td> $0 . 6 6 1 \pm 0 . 1 6$ </td><td> $\underline { { 0 . 3 1 1 \pm 0 . 0 8 } }$ </td></tr><tr><td></td><td> $\mathbf { N o n - G M }$ </td><td> $0 . 8 8 9 \pm 0 . 0 4$ </td><td> $\overline { { 0 . 6 8 6 \pm 0 . 0 9 } }$ </td><td> $0 . 8 1 6 \pm 0 . 0 8$ </td><td> $\overline { { 0 . 7 1 5 \pm 0 . 0 8 } }$ </td></tr><tr><td></td><td>Random trunk</td><td></td><td> $0 . 0 0 7 \pm 0 . 0 0$ </td><td> $0 . 1 3 5 \pm 0 . 0 7$ </td><td> $0 . 0 5 5 \pm 0 . 0 4$ </td></tr><tr><td> $h _ { \mathrm { f c } }$ </td><td>GM</td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $\underline { { 0 . 1 3 5 } } \pm 0 . 0 2$ </td><td> $\underline { { 0 . 3 2 6 \pm 0 . 0 9 } }$ </td><td> $0 . 1 2 6 \pm 0 . 0 1$ </td></tr><tr><td></td><td>Non-GM</td><td> $0 . 8 8 9 \pm 0 . 0 4$ </td><td> $0 . 6 7 7 \pm 0 . 1 1$ </td><td> $0 . 8 1 9 \pm 0 . 0 8$ </td><td> $0 . 7 2 6 \pm 0 . 0 9$ </td></tr><tr><td></td><td>Random trunk</td><td></td><td> $0 . 0 0 6 \pm 0 . 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0$ </td></tr></table>

![](images/fdce0b427c73722c987c5cbc843f4cd6c19a0d0de21c9a0301f1d2a70bd4935c.jpg)  
Figure 39: Normalized rewire recovery. $( \mathrm { G M } - \mathrm { f l o o r } ) / ( \mathrm { N o n } { - } \mathrm { G M } - \mathrm { f l o o r } )$ for the rewires from $h _ { \mathrm { p r e } }$ (red) and $h _ { \mathrm { f c } }$ (teal), with the floor the higher of random trunk and chance, for teacher-target behavior cloning (circles) and three rounds of DAgger (diamonds, oracle targets in MAZE, teacher targets in PUSHTOMARKER), with 95% delta-method intervals across three checkpoints per arm. Open markers flag cells where the Non-GM rewire does not clearly exceed the floor, so the ratio is unstable, n.d. marks a Non-GM rewire at or below the floor, and values beyond the axis are printed at its edge.

COIN and ATTRIBUTE MAZE: recovered. In COIN the GM rewire from $h _ { \mathrm { p r e } }$ reaches 0.843, 0.95 of the Non-GM rewire (0.888). The random trunk reaches 0.646, and its gap to the GM rewire is mostly survival, not coin choice. Among episodes that end at the coin or at the level end, the random trunk takes the coin almost as often as the GM rewire (0.962 against 0.987), and with the GM rewire’s survival it would reach 0.822. The coin is a visible sprite that a head on random features already steers toward, and the GM trunk adds the platforming that keeps the agent alive. In ATTRIBUTE MAZE the GM rewire matches the Non-GM rewire with teacher targets (0.689 against 0.684), and with oracle targets it reaches 0.871, above the native Non-GM policy (0.696, Table 18). On the rebuilt bank the random trunk reaches 0.631 against 0.684 for the GM rewire, and both go to the goal in 0.999 of the episodes in which they reach an object. The $5 \times 5$ maze is small enough for a readout with 528,656 parameters to relearn the task from random features. With oracle targets the random trunk (0.840) lies above the native Non-GM reach (0.696), so at full capacity the rewire cannot tell where the recovery comes from (Section D.2.1). The strict test with a linear readout separates the trunks. Cloned on the bank without the evaluation start configurations, the GM trunk exceeds the random trunk by 0.136 from $h _ { \mathrm { p r e } }$ (0.797 against 0.661, 95% Welch interval 0.05 to 0.22) and by 0.185 from $h _ { \mathrm { f c } }$ (0.487 against 0.303, 0.06 to 0.31), and it stays above it with 10% and 1% of that bank. So the GM trunk carries goal-relevant structure that a random trunk lacks. Most of this advantage is execution, reaching an object at all. Once decisive, the GM rewire picks the goal only slightly more often than the random trunk from $h _ { \mathrm { p r e } } \left( 0 . 9 6 \right.$ against 0.92).

WALKTOFLAG and SORTZONES recovered, ATTRIBUTETABLE partial. The GM rewire from $h _ { \mathrm { p r e } }$ reaches 0.93 of the reference in WALKTOFLAG, 0.92 in SORTZONES and 0.84 in AT-TRIBUTETABLE, and stays well above the random trunk (Figure 38, bottom row). In SORTZONES the Non-GM rewire reaches only 0.87 of its native policy, but the GM rewire is recovered, so the method works (Table 17). What the rewire recovers differs. In WALKTOFLAG the rewire changes which object the agent selects. It reaches the flag first in 0.892 of the conflict layouts, against 0.464 for chance and 0.965 for the Non-GM rewire, while the native GM policy is just as decisive but picks the flag in only 0.272 of decisive episodes. The random trunk cannot walk, so it cannot show whether random features would also select the flag. In SORTZONES and ATTRIBUTETABLE the rewire mainly improves execution. In SORTZONES every trunk, the random one included, brings the cube to the crossed pad whenever it brings it to a pad at all, so the gain over the random trunk is better pushing. In ATTRIBUTETABLE the GM rewire reaches an object far more often than the random trunk (D = 0.822 against 0.530) but prefers the goal only slightly more often once it does (0.922 against 0.867).

## D.2.3 FROM WHICH LAYER CAN THE GOAL BE RECOVERED?

The goal is lost in the last learned layer. In every environment where the GM rewire from $h _ { \mathrm { p r e } }$ is recovered or partial, the rewire from $h _ { \mathrm { f c } }$ recovers clearly less, in PUSHTOMARKER, COIN, ATTRIBUTE MAZE, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG (Table 18), while the Non-GM rewires lose much less. No GM rewire from $h _ { \mathrm { f c } }$ is recovered, and only those of PUSH-TOMARKER and WALKTOFLAG are partial. So in GM agents the goal survives the convolutional trunk and is lost in the map from $h _ { \mathrm { p r e } }$ to $h _ { \mathrm { f c } }$ . This layer compresses thousands of spatial features into 256 and is trained only to keep what the policy acts on. For a Non-GM agent that is the goal. For a GM agent, whose policy acts on the proxy, nothing in training carries the goal through. On the textured background of ATTRIBUTE MAZE, with oracle targets, the ordering of the two sites survives in both cohorts, although there the Non-GM rewire loses almost as much (0.786 to 0.530, against 0.753 to 0.419 for GM).

The learned layer keeps what the policy uses. The fixed random remix separates this from compression alone (Figure 38, open bars). It replaces the learned layer by a random map of the same size, so it keeps what any compression keeps. For Non-GM agents the learned $h _ { \mathrm { f c } }$ beats the remix in every environment, as expected if training shaped it to keep the goal. A random compression alone already keeps most of the Non-GM goal reach in WALKTOFLAG and COIN. For GM agents the learned $h _ { \mathrm { f c } }$ is no better than the remix in COIN and ATTRIBUTE MAZE and only slightly better in LOCATIONREACH, PUSHTOMARKER and SORTZONES, so there their fc layer keeps no more of the goal than a random one would. Only in ATTRIBUTETABLE and WALKTOFLAG does it keep clearly more. The robot probes show the same pattern (Table 5). A random fc layer already loses much of the goal position, so the compression itself is not specific to GM training. What training changes is which cue survives it. Non-GM agents keep the goal and drop the proxy, GM agents keep the proxy and let the goal go. In ATTRIBUTE MAZE and MAZE the probes do not follow the rewires. In ATTRIBUTE MAZE a linear readout separates GM from the random trunk more from $h _ { \mathrm { f c } }$ than from $h _ { \mathrm { p r e } }$ , and the Non-GM line-presence probe drops more than the GM one. In MAZE the GM cheese-bearing probe even rises at $h _ { \mathrm { f c } }$ (Appendix C.1). Probes and rewires use different banks throughout. In ATTRIBUTE MAZE the GM rewire from $h _ { \mathrm { f c } }$ fails although the probes still see the goal cue there, because the GM fc layer keeps no more goal information than a fixed random 256-dimensional map (0.478 learned, 0.481 remix), and the failure is mostly the choice of the wrong object, as a linear readout from GM $h _ { \mathrm { f c } }$ picks the goal in 0.74 of its decisive episodes, against 0.99 for Non-GM and 0.96 from GM $h _ { \mathrm { p r e } }$ . Head size does not explain the gap between the cohorts, since the Non-GM rewire reaches 0.793 with the same head. We think the probes miss this because they test whether the goal cue is present and which role it plays, not on which side of the agent the goal lies, and at $h _ { \mathrm { f c } }$ both cohorts already sit near the probe on the random trunk. Two things limit this reading. The teacher is a Non-GM agent, so a Non-GM $h _ { \mathrm { f c } }$ may suit its targets better, but the robot teachers are other Non-GM agents and the oracle rows show the same drop. And the random trunk at $h _ { \mathrm { f c } }$ stays clearly below the GM rewire in WALKTOFLAG, ATTRIBUTETABLE, SORTZONES, COIN and ATTRIBUTE MAZE with oracle targets, but not in LOCATIONREACH or in ATTRIBUTE MAZE with teacher targets.

![](images/60c8d7def00961ac5075a9ee739dd304d3c2df5871285cc4f66fcee2a855b85e.jpg)  
Figure 40: Rewiring from every layer. Goal reach $\hat { J } ^ { \star }$ of the GM (red) and Non-GM (teal) rewires from each layer of the frozen trunk, means with 95% t-intervals across three checkpoints, with the native policies as dashed lines and the goal probe of Appendix C.1 at the same layer as open diamonds (goal-position $R ^ { 2 }$ in the arm tasks, cheese-bearing AUC, coin-bearing $R ^ { 2 }$ and line-presence AUC in Procgen).

Earlier layers. A rewire from $\mathbf { c o n v } _ { k }$ in the arm tasks, or from the k-th Impala block $h ^ { ( k ) }$ in Procgen, freezes the trunk up to that layer and relearns everything above it (Figure 40, PUSHTOMARKER scored on the conflict scenes and ATTRIBUTETABLE at 50 steps). In LOCATIONREACH and PUSH-TOMARKER GM recovery falls step by step through the trunk, while the goal probe collapses only at $h _ { \mathrm { f c } }$ . In SORTZONES, COIN and ATTRIBUTE MAZE the GM trunk stays near the Non-GM trunk up to $h _ { \mathrm { p r e } } ,$ in ATTRIBUTETABLE up to con $^ { \prime } 2 \cdot$ , and drops at $h _ { \mathrm { f c } } ,$ , where the robot goal probes drop too. In MAZE the GM trunk stays below the Non-GM trunk and the random walk at every layer. Early layers say little about the agent, since a rewire from there relearns most of the network. A random trunk rewired from the first layer recovers about as much as either trained trunk in LOCA-TIONREACH, ATTRIBUTETABLE, COIN and ATTRIBUTE MAZE. It falls with depth, so in COIN and ATTRIBUTE MAZE the GM trunk pulls clearly ahead of it only at the late layers. In LOCA-TIONREACH it falls almost in parallel with the GM trunk, and in PUSHTOMARKER it stays below the GM trunk at every layer up to $h _ { \mathrm { p r e } }$ . Decodable is not the same as usable. In LOCATIONREACH and SORTZONES the random trunk’s goal probe stays high through $h _ { \mathrm { p r e } }$ while its rewire from there recovers little, because a probe reads the goal position directly while a rewire must turn it into closed-loop actions. The sweeps are separate trainings and agree with Table 18 within 0.03, except the COIN random trunk at $h _ { \mathrm { p r e } }$

## D.2.4 IS THE RECOVERY GOAL-DIRECTED?

Table 20 takes the goal away from the rewires as described under additional ablations (Section D.2), masked in the cloning frames or removed at evaluation.

MAZE and LOCATIONREACH: not recovered, but the control is goal-directed. The GM rewire does not recover the goal in either environment (Table 18), so whether its small reach follows the goal says little. In MAZE it stays below chance with or without the cheese (0.098 standard, 0.102 masked, 0.113 removed, chance 0.226). The Non-GM rewires from the same sites are goal-directed. Masking or removing the goal drops them from 0.291 to 0.069 and 0.051 in MAZE and from 0.921 to 0.023 and 0.010 in LOCATIONREACH (chance 0.096). This checks that the method recovers goal use where the goal is available, so its failure on the GM trunks is consistent with the goal being lost in their representation.

PUSHTOMARKER, ATTRIBUTETABLE, SORTZONES and WALKTOFLAG: goal-directed. Here the GM rewire recovers the goal fully or partly, and the recovery needs the goal. Masking or removing it drops GM reach to about zero in PUSHTOMARKER (0.314 to 0.054 and 0.063, chance 0.000), as in the Non-GM rewire, and at every layer of the sweep. In SORTZONES it drops below chance (0.753 to 0.275 and 0.059, chance 0.377), and in WALKTOFLAG as well (0.892 to 0.186 and 0.137, chance 0.464). In ATTRIBUTETABLE removing the goal drops it below chance (0.757 to 0.288, chance 0.411), and the masked rewire stays slightly above chance (0.452), as the arm still reaches one of the two objects. None of these rewires goes to a fixed position or to the proxy. The masked Non-GM rewire in WALKTOFLAG keeps 0.577, with checkpoints between 0.348 and 0.770.

COIN: the test cannot tell. The GM rewire recovers the goal, but the test is not informative here, because the coin often lies on the run to the level end. Cloning with the coin masked lowers the GM rewire only from 0.852 to 0.729 and the Non-GM rewire from 0.885 to 0.822, both still above chance (0.625). With the coin hidden, the rewires still pass its place in most episodes (0.713 GM, 0.660 Non-GM). The mask also covers other gold sprites, so it removes more than the coin.

ATTRIBUTE MAZE: goal-directed, following the goal’s colour when its shape is gone. Cloning with the goal line masked collapses both rewires below chance (0.173 GM, 0.194 Non-GM, chance 0.345), mostly through decisiveness (D = 0.25 against about 0.69 in the standard rewires). The recovered GM reach needs the goal line. Once decisive, the masked rewires still pick the goal in more than half of the episodes (0.69 GM, 0.75 Non-GM), possibly because the readout learned to avoid the proxy object, which we have not tested. Removing the goal separates the cohorts. When the red line is redrawn as a red gem, so that only its colour still separates it from the yellow-gem proxy, the GM rewire still goes to it in 0.521 of episodes and the random trunk in 0.306, while the Non-GM rewire drops to 0.090. The GM rewire reaches the yellow-gem proxy in only 0.028 of these episodes, against 0.617 for the native GM policy. Without the line shape, it therefore does not fall back on the proxy but follows red, the colour that marks the goal in every cloning frame. This test removes the goal’s shape rather than the whole goal, and the GM readout stays on the goal’s side. Every cloning-bank level pairs a red line with a yellow gem, so goal and colour are confounded in the cloning data and the cloning targets cannot teach a readout to ignore colour. Only the Non-GM readout learns to, which suggests that only the Non-GM trunk carries a shape feature strong enough to win over colour. The GM recovery is thus goal-directed, and without the goal’s shape it follows the goal’s colour rather than the proxy.

Table 20: Goal-removed controls for the rewires from $h _ { \mathrm { p r e } } ,$ ordered as in Table 18. Goal reach ${ \hat { J } } ^ { \star } ,$ , mean ± SD across three checkpoints. Standard is the rewire of Table 18 (for ATTRIBUTE MAZE, the standard rewire rerun on the rebuilt bank). Goal masked is a rewire cloned with the goal cue masked in every frame. Goal removed is the standard rewire evaluated with the goal removed (LOCATIONREACH, PUSHTOMARKER, WALKTOFLAG) or replaced by the $\mathrm { p r o x y } ^ { \bullet } \mathbf { s }$ cue (ATTRIBUTETABLE, SORTZONES), counting arrivals at the former goal location. Chance as in Table 18. <sup>†</sup>Teacher targets, rerun on rebuilt banks, with the standard and goal-removed evaluations from the same readouts. The goal is removed by deleting the cheese (MAZE), hiding the coin (COIN) or drawing the goal line as a red gem (ATTRIBUTE MAZE).
<table><tr><td>Task</td><td>Cohort</td><td>Standard</td><td>Goal masked</td><td>1 Goal removed</td><td>Chance</td></tr><tr><td colspan="6">(i) Goal lost in the representation</td></tr><tr><td> $\mathbf { M A Z E } ^ { \dagger }$ </td><td>GM Non-GM</td><td> $0 . 0 9 8 \pm 0 . 0 1$   $0 . 2 9 1 \pm 0 . 0 2$ </td><td> $0 . 1 0 2 \pm 0 . 0 0$   $0 . 0 6 9 \pm 0 . 0 6$ </td><td> $0 . 1 1 3 \pm 0 . 0 1$   $0 . 0 5 1 \pm 0 . 0 2$ </td><td>0.226</td></tr><tr><td>LOCATIONREACH</td><td>GM Non-GM</td><td> $0 . 3 7 3 \pm 0 . 0 8$   $0 . 9 2 1 \pm 0 . 0 5$ </td><td> $0 . 0 8 3 \pm 0 . 0 4$   $0 . 0 2 3 \pm 0 . 0 1$ </td><td> $0 . 0 2 7 \pm 0 . 0 2$   $0 . 0 1 0 \pm 0 . 0 0$ </td><td> $0 . 0 9 6 \pm 0 . 0 3$ </td></tr><tr><td colspan="6">(ii) Goal reaches the policy but is not used</td></tr><tr><td>PUSHTOMARKER</td><td>GM Non-GM</td><td> $0 . 3 1 4 \pm 0 . 0 7$   $0 . 7 0 2 \pm 0 . 1 0$ </td><td> $0 . 0 5 4 \pm 0 . 0 1$   $0 . 0 5 8 \pm 0 . 0 1$ </td><td> $0 . 0 6 3 \pm 0 . 0 3$   $0 . 0 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td>CoIn†</td><td>GM Non-GM</td><td> $0 . 8 5 2 \pm 0 . 0 1$   $0 . 8 8 5 \pm 0 . 0 1$ </td><td> $0 . 7 2 9 \pm 0 . 0 2$   $0 . 8 2 2 \pm 0 . 0 3$ </td><td> $0 . 7 1 3 \pm 0 . 0 3$   $0 . 6 6 0 \pm 0 . 0 2$ </td><td>0.625</td></tr><tr><td>(iii) Goal and proxy compete ATTRIBUTE  $\mathbf { M A Z E } ^ { \dagger }$ </td><td>GM</td><td> $0 . 6 8 4 \pm 0 . 0 2$ </td><td> $0 . 1 7 3 \pm 0 . 0 6$ </td><td> $0 . 5 2 1 \pm 0 . 0 2$ </td><td>0.345</td></tr><tr><td></td><td>Non-GM</td><td> $0 . 6 8 9 \pm 0 . 0 2$ </td><td> $0 . 1 9 4 \pm 0 . 0 8$ </td><td> $0 . 0 9 0 \pm 0 . 0 4$ </td><td></td></tr><tr><td>ATTRIBUTETABLE</td><td>GM Non-GM</td><td> $0 . 7 5 7 \pm 0 . 0 3$   $0 . 8 9 8 \pm 0 . 0 4$ </td><td> $0 . 4 5 2 \pm 0 . 0 5$   $0 . 3 8 2 \pm 0 . 0 2$ </td><td> $0 . 2 8 8 \pm 0 . 0 7$   $0 . 2 4 4 \pm 0 . 0 2$ </td><td> $0 . 4 1 1 \pm 0 . 0 2$ </td></tr><tr><td>SORTZONES</td><td>GM Non-GM</td><td> $0 . 7 5 3 \pm 0 . 0 2$   $0 . 8 2 1 \pm 0 . 0 1$ </td><td> $0 . 2 7 5 \pm 0 . 0 7$   $0 . 0 7 2 \pm 0 . 0 3$ </td><td> $0 . 0 5 9 \pm 0 . 0 3$   $0 . 1 0 3 \pm 0 . 1 1$ </td><td> $0 . 3 7 7 \pm 0 . 0 1$ </td></tr><tr><td>WALKTOFLAG</td><td>GM Non-GM</td><td> $0 . 8 9 2 \pm 0 . 0 5$   $0 . 9 6 5 \pm 0 . 0 1$ </td><td> $0 . 1 8 6 \pm 0 . 0 4$   $0 . 5 7 7 \pm 0 . 2 1$ </td><td> $0 . 1 3 7 \pm 0 . 0 3$   $0 . 1 2 1 \pm 0 . 0 0$ </td><td> $0 . 4 6 4 \pm 0 . 0 1$ </td></tr></table>

## E EVALUATION BANKS

Every bank of states is fixed before any outcome is computed on it. We count every bank in states. A state is one complete environment configuration, including the agent position, together with the observation it renders. Table 21 lists the banks and the analyses that use them.

## E.1 COMMON TERMS

Bank. A fixed list of states on which an analysis scores every policy of an environment. All policies see the same states unless the entry of the bank says otherwise.

Conflict bank. A bank of conflict states. In a conflict state the intended goal and the proxy stand at different places, so following one looks different from following the other.

Strict conflict bank. A conflict bank in which no single action makes progress toward both goal and proxy, A $\mathbf { \boldsymbol { \mathscr { s } } } \left( \mathcal { s } \right) \cap A _ { p } ( \mathcal { s } ) = \boldsymbol { \mathscr { 0 } }$ (Definition 10), so the chosen action alone shows which target the policy follows. Every Procgen conflict bank, the detour bank and the diversified bank are strict. Robot actions are continuous, so two targets at different places never share a progress action and the robot banks need no such filter. The ATTRIBUTETABLE strict bank instead puts the two objects on opposite sides of the robot.

World-edit bank. The robot conflict bank on which the world edits (Appendix C.2) and the Fisher analysis (Appendix C.3) are run, 512 start scenes of the task’s conflict test, or in WALKTOFLAG of its layout type. In Procgen the conflict bank plays this role.

Decision frame. A frame from the middle of an episode, taken when the agent is about to commit to one candidate, instead of the start frame.

Probe bank. A bank in which goal and proxy vary independently, so that a linear probe can learn each of them (Appendix C.1).

Cloning bank. Rollouts of driver policies on which rewiring fits new downstream layers by behavior cloning (Appendix D.2). Robot cloning banks are counted in frames.

Fit and held-out states. Where an axis is fit on a bank, it is fit on one part of the bank and evaluated on the rest. A state and its mirror image always fall on the same side of this split.

Qualification episodes. Behavioral qualification (Appendix F) plays 500 episodes per checkpoint and distribution in every environment, from fixed level ranges in Procgen and fixed resets on the robots (reset seed 10,000 onward), so all checkpoints of an environment start from the same states. The robot conflict rollouts of Figure 44 come from these episodes, the first 250 conflict episodes of each arm task, and in WALKTOFLAG from the forced-view conflict bank.

## E.2 CONFLICT AND WORLD-EDIT BANKS

These banks hold the states on which the world edits (Appendix C.2) are scored. Each edit turns a stored state s into its matched edited state $s _ { e } ,$ rendered from the same stored configuration with only the goal or the proxy changed, so every policy is scored on identical pairs. In MAZE the goal edit moves the cheese and the proxy edit opens the top-right corner, in COIN the coin moves from the right of the agent to its left, in ATTRIBUTE MAZE, ATTRIBUTETABLE and SORTZONES the shape (or cross) swap and the colour swap exchange one attribute between the two candidates, in WALKTOFLAG the flag swap moves the flag beside the tower and the tower swap the tower beside the flag, and in PUSHTOMARKER and LOCATIONREACH the marker or the ball moves. COIN, PUSHTOMARKER and LOCATIONREACH have a goal edit only. Figure 18 shows one state of each of these banks with its edits, the MAZE state from the shared sample, the COIN and ATTRIBUTE MAZE states from their conflict banks and the robot states from the five world-edit banks, and boxes every pixel each edit changes. Figure 42 shows one unedited state of each Procgen conflict bank and Figure 41 one of each robot bank.

Table 21: Evaluation banks and the analyses that use them. The second header row gives the appendix of each analysis, and the entries of Appendices E.2 to E.5 describe each bank. Robot cloning banks count frames, and the notes below the table give the subsets that single analyses use.
<table><tr><td>Bank Appendix</td><td>States</td><td>Probes C.1</td><td>World edits C.2</td><td>Fisher C.3 C.4</td><td>Role</td><td>Geometry C.5</td><td>Steering Rewiring D.1</td><td>D.2</td></tr><tr><td>MAZE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>2,787</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Conflict bank, shared sample Conflict bank, steering sample</td><td>128 48</td><td></td><td></td><td>√</td><td></td><td></td><td>√</td><td></td></tr><tr><td>Fork sample</td><td>320</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td></tr><tr><td>Detour bank</td><td>1,500a</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>Cloning bank</td><td>200,004</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>COIN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>320</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Conflict bank</td><td>600b</td><td></td><td>√</td><td></td><td>V</td><td></td><td>√</td><td></td></tr><tr><td>Cloning bank</td><td>500,352</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ATTRIBUTE MAZE Probe bank</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Conflict bank</td><td>4,608c</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>Diversified bank</td><td>196d</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>480</td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>Cloning bank</td><td>499,998e</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LOCATIONREACH</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>3,072</td><td>√</td><td>V</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-edit bank</td><td>512</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Off-path bank</td><td>500f</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>Cloning bank</td><td>35,641</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>PUSHTOMARKER</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>3,072</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-edit bank</td><td>5129</td><td></td><td>」</td><td></td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Control-layout set</td><td>512h</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>Cloning bānk</td><td>115,200</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ATTRIBUTETABLE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>3,072</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td></tr><tr><td>World-edit bank</td><td>512i</td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>Strict bank</td><td>512j</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Cloning bank</td><td>38,588</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>SORTZONES</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>3,072</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-edit bank</td><td>512i</td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Decision-frame bank</td><td>500k</td><td></td><td></td><td></td><td>V</td><td>√</td><td>√</td><td></td></tr><tr><td>Cloning bank</td><td>44,948</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>WALKTOFLAG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Probe bank</td><td>4,500</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>World-edit bank</td><td>512¹</td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Forced-view conflict bank</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td></td></tr><tr><td></td><td>256m</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Far bank</td><td>256</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td></td><td>600n</td><td></td><td></td><td></td><td></td><td></td><td>V</td><td>√</td></tr><tr><td>Conflict-test bank</td><td>29,634</td></table>

<sup>a</sup>Functional role uses its 168 detour states. <sup>b</sup>Functional role uses 593 of them, and steering the 591 that replaying the harvest reproduces. <sup>c</sup>Per checkpoint and rollout seed. <sup>d</sup>Steering is evaluated on 50 states, and the shared steering axis is fit on the other 146. <sup>e</sup>Oracle labels and the strict rewiring test use separate cloning banks. <sup>f</sup> Per pair of GM and Non-GM seeds. Steering uses the first 200. <sup>g</sup>Also rendered with the gripper above the cube. Steering uses the first 200. <sup>h</sup>Functional role uses the 399 layouts in which marker and centre are disjoint outcomes. <sup>i</sup>Steering starts from the last 200. The geometry uses this bank only for comparison with the strict and decision-frame banks. <sup>j</sup> 256 mirror pairs. The geometry fits the axis on 308 states and evaluates on 204, the steering axis is fit on all 512. The Fisher analysis uses it as a check of the world-edit KL ratios. <sup>k</sup>Frames from GM and Non-GM rollouts in equal parts. The geometry fits the axis on 302 states and evaluates on 198, the steering axis is fit on all 500. <sup>l</sup>The competition axis is fit on the first 256 states, checked on the next 56 and evaluated on the last 200, and the steering axis is fit on this bank. <sup>m</sup>The first 128 give the conflict traces and a steering set. <sup>n</sup>Each agent plays the same 500 for qualification. Steering uses the first 128 and rewiring the first 256.

MAZE conflict bank. States at which the shortest-path first actions toward the cheese and toward the reachable top-right extreme differ, taken along greedy rollouts. The shared sample holds 128 states, the first conflict of each level on the rollout of GM checkpoint 1080 from level 1,000,000, and every policy is scored on these same states. Because they come from a GM rollout, Non-GM policies and initialized networks are scored partly outside their own state distribution. It is used by the world edits (Tables 1 and 6) and the Fisher analysis (Appendix C.3). The steering sample holds 48 states, the first conflict of each level on the rollouts of one GM and one Non-GM checkpoint, 24 each, so that neither cohort is steered only on the other cohort’s states. The fork sample holds 320 states, the first decision fork of 160 GM and 160 Non-GM greedy rollouts. Both are used by state-local steering (Appendix D.1.1).

COIN conflict bank. 600 states, each with two coin placements, the nearest free cell to the right of the agent and the nearest free cell to its left, both visible in the frame. The base state s holds the coin on the right and the edited state s on the left, one cell from the agent in every state, so COIN results describe this single left-versus-right contrast. The states are reached by rollouts of one GM and one Non-GM checkpoint, 300 each, and kept by level geometry alone, so no evaluated policy’s response decides which states enter. Used by the world edits, the Fisher analysis, functional role (593 states, Appendix C.4.3) and steering (the 591 states that replaying the harvest reproduces, Appendix D.1.1).

ATTRIBUTE MAZE conflict bank. 196 states in which the agent stands at the (0, 0) spawn and the two candidates, a red line and a yellow gem, lie at equal distance, so every conflict is UP versus RIGHT. Black and textured observations are two renderings of the same states. The bank is read from the serialized game state of 40,000 levels, whose 2,041 strict-conflict levels reduce to 196 unique states because the generator has only 11 wall layouts. Used by the world edits, the Fisher analysis, functional role (Appendix C.4.4), the cue-competition geometry (Appendix C.5) and steering. Steering is evaluated on 50 states (Appendix D.1.1), and the shared steering axis is fit on the other 146, 118 to fit and 28 to check (Appendix D.1.2), with line and gem mirror pairs on the same side of the split.

Robot world-edit banks. Each holds 512 start scenes of the task’s conflict test, or in WALK-TOFLAG of its layout type, with goal and proxy at different places. Every robot bank is drawn with a sampler seed of its own and written into the simulator before the first frame is rendered, so every policy sees identical observations. All five world-edit banks are used by the world edits and the Fisher analysis. Figure 41 shows one state from each robot bank.

LOCATIONREACH world-edit bank. The ball lies uniformly over the workspace at least 10 cm from the training corner, the yellow distractor at least 8 cm from the ball, and the arm starts from the held-out half of the start poses. Also used to evaluate rewiring (Appendix D.2).

PUSHTOMARKER world-edit bank. Conflict-test scenes with the marker 10 to 11 cm from the table centre and the cube further out on the same ray, and arm start poses from the held-out half. The same scenes are rendered a second time with the gripper held above the cube, where the side of the push is decided. Because the marker lies on the path to the centre, these scenes show where the agent stops but not where it pushes, which is what the control-layout set adds (Appendix E.3). Also used by functional role (all 512, Appendix C.4.2), steering (first 200, Appendix D.1.1) and rewiring (all 512).

ATTRIBUTETABLE world-edit bank. A blue cube and a red sphere at least 13 cm apart, with the arm at its raised start pose plus joint noise. Both objects lie on the same side of the robot in 36% of scenes (median bearing difference 38<sup>◦</sup> as seen from the hand), so the conflict orientation is often ambiguous. Also used by steering (last 200), rewiring (all 512) and, for comparison with the strict bank, the cue-competition geometry (Appendix C.5.2).

SORTZONES world-edit bank. A crossed purple pad and a plain green pad, one on each side, with the cube between them. From the retracted arm at the start frame both pads lie in nearly the same direction. Also used by steering (last 200), rewiring (all 512) and, for comparison with the decision-frame bank, the cue-competition geometry.

![](images/3138d6bd015a6961b749eb819b20e4a7b6df5eaf7c2894698d4f985530a1831d.jpg)  
Figure 41: One state from each robot bank. Green rings mark the intended goal and orange rings the proxy, dashed where the proxy is a place rather than an object. The off-path bank, the controllayout set and the cloning banks store no frames. The WALKTOFLAG far bank is not shown. The SORTZONES conflict-test panel is the first scene of the robot conflict-test samples (Appendix E.3).

ATTRIBUTETABLE strict bank. 512 start scenes in which the two objects lie on opposite sides of the robot, so the side of each candidate is clear. The objects are at least 13 cm apart, at least 45<sup>◦</sup> apart as seen from the start position of the hand (median 62<sup>◦</sup>) and at equal distance from it within 3 cm. Every scene is stored with its mirror image across the robot’s midline, with the same arm start pose, so the bank holds 256 mirror pairs and both conflict orientations occur equally often. It is drawn with a sampler seed that no other bank uses. Used by functional role (Appendix C.4.4), the cue-competition geometry (axis fit on 308 states and evaluated on 204, Appendix C.5), the shared steering axis (fit on all 512, Appendix D.1.2) and, as a check of the world-edit KL ratios, the Fisher analysis (Appendix C.3).

SORTZONES decision-frame bank. 500 decision frames, each taken when the hand first reaches the cube, within 7 cm of it and at most 6 cm high, before the cube has moved (median step 28 of 100). Each frame comes with the same frame of two copies of the scene in which the cross or the colour is swapped, kept in step with the played scene. The scenes start from strict configurations, with the two pads at least 120<sup>◦</sup> apart and equally far from the cube within 1 cm, drawn with a sampler seed that no other bank uses, and GM and Non-GM agents play them in turn. Frames in which the arm hides an edit are dropped, and the rest are balanced over conflict orientation and the cohort of the playing agent. Used by functional role (Appendix C.4.4), the cue-competition geometry (axis fit on 302 states and evaluated on 198) and the shared steering axis (fit on all 500).

WALKTOFLAG world-edit bank. The layout type of the conflict test, with the tower at least 1.5 m from the straight line between start and flag, but with flag and tower 1.2 to 2.5 m from the camera and in view, so that the flag stays visible after it is moved. The two stand 1.5 to 3.0 m apart, closer than the 3 to 5 m of the conflict test. Also used by functional role (pull vectors and matched swaps, Appendix C.4.5), the cue-competition geometry (axis fit on the first 256 states, checked on the next 56 and evaluated on the last 200) and steering (the shared axis is fit on it, Appendix D.1.2).

## E.3 SPECIAL BANKS

MAZE detour bank. 1,500 strict-conflict states on 618 levels, chosen from level geometry alone so that no cohort’s behavior decides which states enter. In each level it keeps up to 40 reachable cells at which the shortest-path first actions toward the cheese and toward the reachable top-right extreme differ and the two path lengths differ by at most two steps, and it reaches each cell by a scripted walk from the spawn. Functional role (Appendix C.4.1, Table 10) uses its 168 detour states, at which no action that makes progress toward the extreme points up or right.

ATTRIBUTE MAZE diversified bank. 480 strict-conflict states in which the agent stands at each of the 21 walkable cells, so that every pair of movement directions occurs as a conflict. In the conflict bank every conflict is UP versus RIGHT, so bearing, immediate action and the sign of $d _ { \mathrm { c o m p } } ( s )$ are one variable, and this bank separates them. It is built by writing the agent position into the serialized state (Appendix H). GM agents rarely reach many of these cells in their own rollouts, so the bank tests them partly outside their state distribution. Used by functional role (Appendix C.4.4) and the cue-competition geometry (Appendix C.5.2). Figure 42 shows one of its states next to one state of each Procgen conflict bank.

LOCATIONREACH off-path bank. Start states in which the ball sits at a position that the GM gripper of the same seed passes through at ball height from another start, at least 8 cm from its path in the current episode and at least 10 cm from the corner, so reaching the ball asks for no movement the GM agent has not already made (Appendix A.2.2). It holds 500 states for each pair of GM and Non-GM seeds, with 26 to 32 distinct ball positions per seed, and is built separately for each seed pair instead of being shared by all checkpoints (Appendix F.4). Used by the off-path test of functional role (Appendix C.4.1, Figure 22e to h), steering (first 200) and rewiring (all 500).

PUSHTOMARKER control-layout set. 512 scenes from the Non-GM training distribution, drawn with a sampler seed of their own, with marker and cube placed independently, so that a push toward the table centre does not reach the marker. Functional role (Appendix C.4.2) uses the 399 layouts per seed in which reaching the marker and reaching the centre are disjoint outcomes and the marker lies off the straight path to the centre, to separate pushing toward the marker from pushing toward the centre. Rewiring uses all 512 as a second evaluation set (Appendix D.2).

![](images/9f9156dc82363199c750958fed987a6677de22ae920d647c11fa133e575b25cf.jpg)  
Figure 42: One state from each conflict bank and from the diversified bank. From left to right, the shared and steering samples of the MAZE conflict bank, the COIN conflict bank, the ATTRIBUTE MAZE conflict bank and the diversified bank. Green arrows are progress actions toward the intended goal, orange arrows toward the proxy (Definition 10). Circles mark the cheese and the top-right target. The COIN panel draws the coin at its edited placement left of the agent, and the dashed circle marks its base placement to the right. Each panel shows a single state. ATTRIBUTE MAZE panels are schematics of the stored state (red line, yellow gem, brown walls). The MAZE fork sample and detour bank are not shown.

WALKTOFLAG forced-view conflict bank. 256 layouts with the conflict test’s 3 to 5 m between flag and tower and the start pose turned so that both objects are in view. Used by functional role (pull vectors, Appendix C.4.5). Its first 128 give the conflict traces (Figure 44) and the forced-view steering set (Appendix D.1.1).

WALKTOFLAG far bank. 256 scenes with the flag 3.5 to 6 m from the camera, in view but covering almost no pixels, and the tower 1.2 to 2.5 m away. Used by functional role (pull vectors, Appendix C.4.5).

WALKTOFLAG conflict-test bank. 600 layouts of the conflict test with the native start pose, in which the two objects need not be in view. Every agent plays the same 500 in the qualification (Appendix F). Its first 128 give the native steering set (Appendix D.1.1), and rewiring is evaluated on its first 256 (Appendix D.2).

Robot conflict-test samples. Start scenes of the conflict test of each arm task (Appendix A), the evaluation distribution on which goal misgeneralization is measured. They are not stored as a bank but drawn from fixed resets, reset seed 10,000 onward, so every checkpoint plays the same scenes. Used by the behavioral qualification, which plays 500 conflict episodes per checkpoint (Appendix F), and by the conflict traces, which use the first 250 (Figure 44). The SORTZONES panel of Figure 41 is the first scene of the first batch of 125. WALKTOFLAG stores its conflict-test layouts as the conflict-test bank above.

## E.4 PROBE BANKS

A probe learns a concept only if its bank contains labelled variation in that concept, such as the cheese present and absent, the coin left and right of the agent, or an attribute on an object that doe not carry the rewarded role. Conflict banks hold this variation fixed, so each environment has one probe bank built to make its probe targets separable (Tables 22 and 23). All probe banks are used only by the probes (Appendix C.1).

MAZE probe bank. 2,787 states, scripted and independent of any policy. It places the agent at four cells in each of 200 levels and, for each placement, puts the cheese at up to three cells (the reachable top-right cell, a distant cell and a random cell) or removes it, so 800 states hold no cheese.

COIN probe bank. 320 states from 81 levels, collected on rollouts of GM checkpoint 6033 and balanced so that the coin lies left and right of the agent equally often at every stage of level progress.

Table 22: Probe banks. Binary labels are scored by held-out ROC–AUC, and COIN bearing by held-out $R ^ { 2 }$ . Bearing is taken relative to the agent. Sites run over ${ h ^ { ( 1 ) } , h ^ { ( 2 ) } , h ^ { ( 3 ) } , h _ { \mathrm { p r e } } , h _ { \mathrm { f c } } }$ (Appendix C.1). Cheese bearing uses the 1,987 states that hold the cheese. <sup>a</sup>The residualization is fit on the training partition only. <sup>b</sup>Per checkpoint and rollout seed, 3,072 training and 1,536 held-out states.
<table><tr><td>Concept</td><td>Label</td><td>States Split</td><td></td><td>Sites</td></tr><tr><td colspan="5">MAZE</td></tr><tr><td>Cheese presence</td><td>cheese present or removed</td><td></td><td>2,787 level-grouped</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Cheese bearing</td><td>cheese left or right of the agent</td><td>1,987</td><td>level-grouped</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Corner bearing</td><td>top-right extreme left or right of the agent</td><td></td><td>2,787 level-grouped</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td colspan="5">COIN</td></tr><tr><td>Coin presence</td><td>coin visible in the frame</td><td></td><td>320 level-grouped</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Coin bearing</td><td>horizontal offset from the agent, residualized on level progressª</td><td></td><td>320 level-grouped</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td colspan="5">ATTRIBUTE MAZE</td></tr><tr><td>Line presence  $L _ { \mathrm { p r e s } }$ </td><td>line shape present</td><td> $4 , 6 0 8 ^ { b }$ </td><td>configuration grid</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Yellow presence  $Y _ { \mathrm { p r e s } }$ </td><td>yellow colour present</td><td> $4 , 6 0 8 ^ { b }$ </td><td>configuration grid</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Line role  $L _ { \mathrm { r o l e } }$ </td><td>rewarded object is a line</td><td> $4 , 6 0 8 ^ { b }$ </td><td>configuration grid</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr><tr><td>Yellow role  $Y _ { \mathrm { r o l e } }$ </td><td>rewarded object is yellow</td><td> $4 , 6 0 8 ^ { b }$ </td><td>configuration grid</td><td> $h ^ { ( 1 ) } – h _ { \mathrm { f c } }$ </td></tr></table>

In MAZE and COIN, level-grouped splits assign whole levels to either the fitting or the held-out partition, over five split seeds.

ATTRIBUTE MAZE probe bank. A grid of 12 training and 6 held-out object configurations on disjoint level ranges, on which each checkpoint is evaluated on its own sampled rollouts, 4,608 states per checkpoint and rollout seed. Its labels are constant within a configuration, so its effective sample is 12 and 6 configurations.

Robot probe banks. The robot probe banks (Table 23) hold start frames, rendered before the agent acts, in which goal and proxy vary independently. Each arm bank holds 2,048 training and 1,024 held-out scenes from the Non-GM training distribution of its task, and the WALKTOFLAG bank holds 3,000 training and 1,500 held-out start states, one third from each of the three layout distributions and half of them facing the flag within $4 5 ^ { \circ }$ , so that a visible flag is common. The conflict-layout third gives the WALKTOFLAG bearing probes in which flag and tower bearings are nearly uncorrelated (Table 5, conflict rows).

## E.5 CLONING BANKS

Rewiring fits fresh downstream layers by behavior cloning on rollouts of GM and Non-GM driver policies, pooled across drivers. Each environment has one cloning bank (Table 21), on levels disjoint from the levels on which the rewired policies are evaluated (Appendix D.2). The cloning banks are used only by rewiring (Table 18, Appendices D.2.3 and D.2.4), and the same bank serves the matched GM and Non-GM rewires and every representation site.

Procgen cloning banks. Rollouts of GM and competent Non-GM drivers in equal parts, 200,004 states in MAZE, 500,352 in COIN and 499,998 in ATTRIBUTE MAZE. In MAZE the oracle labels are attached to the same states. In ATTRIBUTE MAZE the oracle-labelled states come from a separate bank, and the strict rewiring test clones on a bank from which every evaluation start configuration is removed (Appendix D.2).

Robot cloning banks. Each robot cloning bank pools sampled-action rollouts of 192 episodes per driver and counts their frames. In LOCATIONREACH and PUSHTOMARKER the drivers are the three GM and the three Non-GM agents, the scenes come from the Non-GM training distribution, and the arm starts from the training half of the start poses, disjoint from the held-out half of every evaluation bank. LOCATIONREACH keeps the frames up to the first contact with the ball, which gives

Table 23: Robot probe banks. Positions and bearings are scored by held-out $R ^ { 2 }$ (positions averaged over x and y), and binary labels by held-out ROC–AUC. States count training and heldout frames together, about two thirds of them for training. Training and held-out frames are drawn with separate sampler seeds. Arm sites are $H ^ { ( 1 ) }$ $H ^ { ( 2 ) } , h _ { \mathrm { p r e } } , h _ { \mathrm { f c } }$ with unpooled maps, and WALKTOFLAG sites are $\bar { H } ^ { ( 1 ) } , \bar { H } ^ { ( 2 ) } , \bar { H } ^ { ( 3 ) } , h _ { \mathrm { p r e } } , h _ { \mathrm { f c } }$ with maps average-pooled to $4 { \times } 4$ cells (Appendix C.1). <sup>a</sup>Training and held-out scenes also start the arm from disjoint halves of the pool of arm start poses.
<table><tr><td>Concept</td><td>Label</td><td>States Split</td><td></td><td>Sites</td></tr><tr><td>PUSHTOMARKER Marker position</td><td>marker  $( x , y )$  on the table</td><td></td><td>3,072 seeds, start posesª</td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>LOCATIONREACH</td><td></td><td></td><td></td><td></td></tr><tr><td>Ball position ATTRIBUTETABLE</td><td>green ball  $( x , y )$  on the table</td><td></td><td>3,072 seeds, start posesª</td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>Cube position Red-object position</td><td>cube  $( x , y )$  on the table red object  $( x , y )$  , scenes with</td><td>3,072 seeds 1,533 seeds</td><td></td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$   $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>SORTZONES</td><td>exactly one red object</td><td></td><td></td><td></td></tr><tr><td>Crossed-pad position</td><td>crossed pad  $( x , y )$  on the table</td><td>3,072 seeds</td><td></td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>Crossed-pad side</td><td>crossed pad on the left half</td><td>3,072 seeds</td><td></td><td></td></tr><tr><td></td><td></td><td>3,072 seeds</td><td></td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>Green-pad position</td><td>green pad  $( x , y )$  on the table</td><td></td><td></td><td> $H ^ { ( 1 ) } { - } h _ { \mathrm { f c } }$ </td></tr><tr><td>WALKTOFLAG</td><td></td><td></td><td></td><td></td></tr><tr><td>Flag visible</td><td>flag covers at least 5 pixels</td><td>4,500 seeds</td><td></td><td> $\bar { H } ^ { ( 1 ) } - h _ { \mathrm { f c } }$ </td></tr><tr><td>Flag bearing</td><td>flag bearing from the camera, flag visible</td><td>1,174 seeds</td><td></td><td> $\bar { H } ^ { ( 1 ) } - h _ { \mathrm { f c } }$ </td></tr><tr><td>Tower visible</td><td>tower covers at least 5 pixels</td><td>4,500 seeds</td><td></td><td> $\bar { H } ^ { ( 1 ) } - h _ { \mathrm { f c } }$ </td></tr><tr><td>Tower bearing</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>tower bearing from the camera, tower visible</td><td>1,949 seeds</td><td></td><td> $\bar { H } ^ { ( 1 ) } - h _ { \mathrm { f c } }$ </td></tr></table>

35,641 frames, and PUSHTOMARKER keeps all 100 steps of each episode, which gives 115,200. In ATTRIBUTETABLE and SORTZONES the drivers are all qualified agents, seven and six, and alternate episodes come from the GM and the Non-GM training distribution, drawn with a sampler seed that no evaluation bank uses (38,588 and 44,948 frames). In WALKTOFLAG the drivers are the three Non-GM agents, on Non-GM training layouts drawn with their own seeds, which gives 29,634 frames.

## F BEHAVIORAL QUALIFICATION OF GOAL MISGENERALIZATION

Goal misgeneralization is a spectrum, not a binary state. A checkpoint can follow the proxy in any share of states, and so show a weaker or a stronger form of goal misgeneralization. Every experiment in this paper still compares a goal-misgeneralized cohort with a control cohort, so we need explicit rules that decide when a checkpoint is misgeneralized strongly enough to carry a result and when it is clean enough to serve as its control. This section states those rules and shows, from rollout outcomes alone and before any representation is inspected, that every checkpoint carries its GM or Non-GM label.

## F.1 DEFINITIONS

We use the notation of Appendix B.

Proxy following. A trajectory follows the proxy when the episode ends at the proxy without the goal, $\mathbf { \bar { \Gamma } } ^ { \mathrm { p r o x } } ( \tau ) ( \mathbf { \breve { 1 } } - \Gamma ^ { \star } ( \tau ) ) = \mathbf { \bar { 1 } }$ . No ordering of the two events is needed. An episode ends when the intended goal is reached, so a trajectory that reached the goal never counts as proxy following, even when it passed the proxy on the way, as a MAZE agent that walks through the corner and then finds the cheese.

Definition 12 (GM checkpoint). Let π be a policy checkpoint at a fixed training step, trained on $\Lambda _ { \mathrm { t r a i n } }$ . Then π is goal-misgeneralized (GM) ifit meets both conditions below, with the statistics and thresholds ofSection F.2.

(C) Capability. π is competent enough to reach a target,

$$
C _ { e } ( \pi ) = \mathbb { E } _ { \xi \sim \Lambda _ { \mathrm { t r a i n } } , \tau \sim P _ { \pi } ( \tau | \xi ) } \big [ \Gamma ^ { \star } ( \tau ) \big ] \ \ge \ c _ { e } ,
$$

so that low intended-goal success cannot be attributed to incompetence. Here $\Gamma ^ { \star }$ follows the environment’s own success specification and $c _ { e }$ is the threshold ofenvironment e.

(P) Proxy following. In at least a fixed share of out-of-distribution episodes, the episode of π ends at the proxy without the goal,

$$
\rho ( \pi ) = \mathbb { E } _ { \xi \sim \Lambda _ { \mathrm { O O D } } , \tau \sim P _ { \pi } ( \tau | \xi ) } \left[ \Gamma ^ { \mathrm { p r o x } } ( \tau ) \big ( 1 - \Gamma ^ { \star } ( \tau ) \big ) \right] \ \geq \ \rho _ { \mathrm { G M } } .
$$

Definition 13 (Control (Non-GM) checkpoint). A checkpoint π is a control, written Non-GM, if it meets three conditions.

(T) Training. π is trained on a distribution $\Lambda _ { \mathrm { t r a i n } } ^ { \mathrm { c t r l } }$ on which the proxy carries no information about reward, so that goal and proxy are not co-located and

$$
\operatorname* { P r } _ { \Lambda _ { \mathrm { t r a i n } } ^ { \mathrm { c t r l } } } \left( \Gamma ^ { \mathrm { p r o x } } ( \tau ) = 1 \mid \Gamma ^ { \star } ( \tau ) = 1 \right) = \operatorname* { P r } _ { \Lambda _ { \mathrm { t r a i n } } ^ { \mathrm { c t r l } } } \left( \Gamma ^ { \mathrm { p r o x } } ( \tau ) = 1 \right) .
$$

(C) Capability. $C _ { e } ( \pi ) \geq c _ { e }$ as in Definition 12, with the expectation over $\Lambda _ { \mathrm { t r a i n } } ^ { \mathrm { c t r l } } .$

(N) No proxy following. $\rho ( \pi ) \le \rho _ { \mathrm { c t r l } }$

As (P) and (N) bound $\rho$ on opposite sides, no checkpoint is both GM and a control. The two classes share the capability condition, so a GM checkpoint and its control differ in target and not in competence. A checkpoint that is incapable, or follows the proxy only occasionally $( \rho _ { \mathrm { c t r l } } < \rho <$ $\rho _ { \mathrm { G M } } )$ , is neither and enters no cohort. The main text calls Non-GM checkpoints control agents.

## F.2 ESTIMATES AND THRESHOLDS

We estimate both quantities from $N = 5 0 0$ rollouts per checkpoint in every environment, $\hat { \rho } ( \pi ) =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Gamma ^ { \mathrm { p r o x } } ( \tau _ { i } ) \big ( 1 - \Gamma ^ { \star } ( \tau _ { i } ) \big ) } \end{array}$ with levels drawn from $\Lambda _ { \mathrm { O O D } }$ and $\begin{array} { r } { \hat { C } _ { e } ( \pi ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Gamma ^ { \star } ( \tau _ { i } ) } \end{array}$ with levels drawn from the checkpoint’s own training distribution. $\hat { \rho }$ uses greedy actions in Procgen and deterministic actions on the robots. In Procgen $\hat { C } _ { e }$ uses sampled actions, the action mode of training, and on the robots deterministic actions. Success always follows the environment’s own success specification. The proxy event is instantiated as follows.

## Procgen tasks.

• COIN: the agent reaches the end wall of the level without collecting the coin (Cobbe et al., 2019; Langosco et al., 2022). Deaths and timeouts are neither goal nor proxy.

• ATTRIBUTE MAZE: the episode ends on the yellow gem (Ramanauskas & S¸ims¸ek, 2023), which is the first object the agent touches.

• MAZE: the agent enters the corner region, the top-right 28% of the observation along each axis, and never reaches the cheese. Levels on which the cheese itself lies in that region never count as proxy.

## Arm tasks.

• ATTRIBUTETABLE: the red sphere is touched and the cube never is.

• SORTZONES: the cube enters the green pad and never the crossed one.

• LOCATIONREACH: the gripper comes within 3 cm of the training corner and never touches the ball.

• PUSHTOMARKER: the cube rests within 3 cm of the table centre at the end of the episode, and the goal is that it rests on the marker at the end. Both are read off the final state, which cannot satisfy both.

Quadruped task. In WALKTOFLAG the first object the robot comes within reach of ends the episode, 0.5 m for the flag and 0.6 m for the tower, and the episode is proxy when that object is the tower. Falls and timeouts are neither goal nor proxy.

Thresholds. We set $\rho _ { \mathrm { G M } } = 0 . 2 5$ and $\rho _ { \mathrm { c t r l } } = 0 . 0 2$ in all eight environments. The first asks that a quarter of all conflict episodes end at the proxy without the goal and not only rare ones, and the second leaves a gap of more than an order of magnitude between the two labels. The capability threshold is $c _ { e } = 0 . 4 0$ in the three Procgen games and $c _ { e } = 0 . 8 5$ in the five robot tasks. In MAZE $c _ { e } = 0 . 4 0$ is a mean return of at least 4.0 at 10 reward per success, the capability rule of the validation that first labelled these agents, and COIN and ATTRIBUTE MAZE take the same value. In the robot tasks the training distribution is cheap to sample and every capable seed solves it almost always, so we ask for $c _ { e } = 0 . 8 5$

Preference in ATTRIBUTE MAZE. An episode is decisive, written dec, if it reaches either candidate, the intended goal or the proxy, before the time limit. The proxy rate factorizes into how often a policy is decisive and at which candidate it ends when it is,

$$
\rho = D \big ( 1 - \mathrm { P r } ( \mathrm { g o a l } \mid \mathrm { d e c } ) \big ) ,\tag{6}
$$

with the decisiveness and the conditional preference estimated as

$$
\begin{array} { r } { D = \mathrm { P r } ( \mathrm { d e c } ) , \qquad \hat { D } = \frac { N _ { \mathrm { g o a l } } + N _ { \mathrm { p r o x y } } } { N } , } \\ { \widehat { \mathrm { P r } } ( \mathrm { g o a l } \mid \mathrm { d e c } ) = \frac { N _ { \mathrm { g o a l } } } { N _ { \mathrm { g o a l } } + N _ { \mathrm { p r o x y } } } , } \end{array}
$$

where $N _ { \mathrm { g o a l } }$ and $N _ { \mathrm { p r o x y } }$ count the episodes that end at the goal and at the proxy. The intended-goal reach factorizes in the same way,

$$
\overline { { \mathrm { R e a c h } } } = \frac { N _ { \mathrm { g o a l } } } { N } = \hat { D } \widehat { \mathrm { P r } } ( \mathrm { g o a l } \mid \mathrm { d e c } ) .
$$

In ATTRIBUTE MAZE the gem is as easy to reach as the intended object, so a policy that picks between them at random already has $\rho \approx D / 2$ , well above $\rho _ { \mathrm { G M } }$ . A GM checkpoint there must therefore also prefer the proxy, with the point estimate $\widehat { \mathrm { P r } } ( \mathrm { g o a l } \mid \mathrm { d e c } ) < 0 . 5$

![](images/966e1f43040ada564927d28853598a5dcb13e36b8a0f9d3122f727e4dc65b0e7.jpg)  
Figure 43: Goal pursuit in all eight environments. Each point is one checkpoint at one training step, placed by decisiveness D and conditional intended-goal preference Pr(goal | dec) on $\Lambda _ { \mathrm { O O D } } .$ with 95% Wilson intervals on both axes for cohort and published checkpoints. The dashed line marks $\operatorname* { P r } ( \mathrm { g o a l } \mid \mathrm { d e c } ) = 0 . 5$ , the ATTRIBUTE MAZE check. Grey points are scored checkpoints outside the cohorts, and open circles pass every condition of their label but are not carried. Cohort checkpoints use the $N = { \bar { 5 } } 0 0$ evaluations of Table 24. Points outside the cohorts and the published checkpoints come from earlier evaluations (80 to 1,000 episodes).

An additional check in COIN. Reaching the end wall often does not exclude that a policy still pursues the coin whenever it sees it. Following Langosco et al. (2022), who separate coin pursuit from end-wall pursuit by moving the coin, we add a harsher test outside the definition. We roll out the checkpoint on the same levels twice, once as usual and once with the coin sprite removed from the observation while the coin stays in the level, collectible and rewarded (Appendix H), and compute

$$
\mathrm { d e p } = \overline { { \mathrm { R e a c h } } } ( \mathrm { c o i n v i s i b l e } ) - \overline { { \mathrm { R e a c h } } } ( \mathrm { c o i n h i d d e n } ) .\tag{7}
$$

A policy that pursues the coin has $\mathrm { d e p } > 0$ , since seeing the coin helps it get there. Every COIN GM checkpoint must have dep $< - 0 . 0 5$ . Rendering the coin lowers its collection on all three (dep from −0.135 to −0.100), whereas it raises collection for control 7001 (+0.219).

## F.3 GOAL PURSUIT IS GRADED

Figure 43 places every scored checkpoint on the $( D , \operatorname* { P r } ( \operatorname { g o a l } \mid \operatorname* { d e c } ) )$ plane, and goal pursuit there is graded rather than binary. Labels change within a run, as COIN seed 6033 moves from $\mathrm { P r ( g o a l ) }$ $\mathrm { d e c } ) = 0 . 4 0 2$ at 20M to 0.788 at 200M at similar D (0.682 and 0.685). Labels also spread across seeds of one condition, as the eight black ATTRIBUTE MAZE seeds span 0.408 to 0.698 at $D \geq$ 0.842. Strict GM is rare. Of the 63 greedy COIN checkpoints scored in the GM training regime only four reach $\rho \ge 0 . 2 5$ with dep $< - 0 . 0 5$ , the three cohort checkpoints and 6037 at 40M, whose capability on $\Lambda _ { \mathrm { t r a i n } }$ was not measured. The grading has a simple origin. As Γ<sup>⋆</sup> and Γ<sup>prox</sup> coincide on $\Lambda _ { \mathrm { t r a i n } } .$ , reward never selects between them and nothing forces a policy to commit to one, so intermediate values are what training should produce. Therefore selecting on the definition is part of the method and not a filter laid over it, since a cohort drawn from one training condition would mix chance-level and goal-preferring checkpoints, and a mechanistic null on it would reflect that mixture. The robot checkpoints sit at the ends of this range instead. On the arm every capable GM seed commits almost only to the proxy $( \widehat { \mathrm { P r } } ( \mathrm { g o a l } | \mathrm { d e c } ) \le 0 . 0 6 9 )$ , and only WALKTOFLAG lands in between, where GM agents still reach the flag in about a quarter of conflict episodes. We score only the final checkpoint of each robot run and few seeds per task, so this shows that the robot cohort are cleanly separated, not that robot training avoids the intermediate values seen in Procgen.

Table 24: Behavioral cohorts, worst checkpoint per cohort. In Procgen $\hat { C } _ { e }$ uses sampled actions and ρˆ greedy actions, and on the robots both use deterministic actions. N is the number of $\Lambda _ { \mathrm { O O D } }$ episodes per checkpoint, min $\hat { C } _ { e }$ the lowest capability in the cohort, and worst $\hat { \rho }$ the lowest proxy rate for GM and the highest for Non-GM. Pass counts checkpoints that meet the definition of their label (Section F.2). Every checkpoint is the final save of its run except the COIN GM checkpoints, which are selected by training step within their 200M-step runs (Appendix H, Table 26).
<table><tr><td>Env.</td><td>Cohort</td><td>Checkpoints</td><td>N</td><td>min  $\hat { C } _ { e }$ </td><td>worst ê</td><td>Pass</td></tr><tr><td rowspan="2">MAZE</td><td>GM</td><td>1080-1082</td><td>500</td><td>0.932</td><td>0.262</td><td>3/3</td></tr><tr><td>Non-GM</td><td>1090-1092</td><td>500</td><td>0.928</td><td>0.002</td><td>3/3</td></tr><tr><td rowspan="2">COIN</td><td>GM</td><td>6033@20M, 6037@120M, 6051@55M</td><td>500</td><td>0.780</td><td>0.310</td><td>3/3</td></tr><tr><td>Non-GM</td><td>7001-7003</td><td>500</td><td>0.908</td><td>0.000</td><td>3/3</td></tr><tr><td rowspan="4">ATTR. MAZE</td><td>GM, black</td><td>6313,6315,6317</td><td>500</td><td>0.998</td><td>0.490</td><td>3/3</td></tr><tr><td>Non-GM, black</td><td>4371-4373</td><td>500</td><td>0.984</td><td>0.000</td><td>3/3</td></tr><tr><td>GM, texture</td><td>6301-6303</td><td>500</td><td>0.988</td><td>0.526</td><td>3/3</td></tr><tr><td>Non-GM, texture</td><td>4361-4363</td><td>500</td><td>0.908</td><td>0.002</td><td>3/3</td></tr><tr><td>PUSHTO MARKER</td><td>GM</td><td>0-2</td><td>500</td><td>0.974</td><td>0.916</td><td>3/3</td></tr><tr><td rowspan="2">LOCATION</td><td>Non-GM</td><td>0-2</td><td>500</td><td>0.898</td><td>0.000</td><td>3/3</td></tr><tr><td>GM</td><td>1-3</td><td>500</td><td>0.998</td><td>0.986</td><td>3/3</td></tr><tr><td rowspan="2">REACH ATTRIBUTE</td><td>Non-GM</td><td>1-3</td><td>500</td><td>0.992</td><td>0.002</td><td>3/3</td></tr><tr><td>GM</td><td>2-4</td><td>500</td><td>0.998</td><td>0.742</td><td>3/3</td></tr><tr><td rowspan="2">TABLE SORTZONES</td><td>Non-GM</td><td>3-5</td><td>500</td><td>0.980</td><td>0.008</td><td>3/3</td></tr><tr><td>GM</td><td>0-2</td><td>500</td><td>0.980</td><td>0.902</td><td>3/3</td></tr><tr><td rowspan="2"></td><td>Non-GM</td><td>0-2</td><td>500</td><td>0.934</td><td>0.000</td><td>3/3</td></tr><tr><td>GM</td><td>0-2</td><td>500</td><td>0.988</td><td>0.714</td><td></td></tr><tr><td rowspan="2">WALKTOFLAG</td><td>Non-GM</td><td></td><td></td><td></td><td></td><td>3/3</td></tr><tr><td></td><td>0-2</td><td>500</td><td>0.988</td><td>0.012</td><td>3/3</td></tr></table>

The five ATTRIBUTE MAZE checkpoints released by Ramanauskas & S¸ims¸ek (2023) (Ramanauskas, 2023) show the same spread. They train on a single yellow line on a black background, and on our evaluation all five are highly decisive $( D \ge 0 . 8 9 5 )$ while only two, 7257 and 3561, prefer the proxy. Ramanauskas & S¸ims¸ek (2023) establish goal misgeneralization for the training setup and not per checkpoint, so their labels stand under their own criterion. The published checkpoints are a reference population and enter no other analysis.

## F.4 THE COHORTS WE STUDY

Table 24 lists the cohorts used in every experiment that follows, and Figure 10 in Appendix A shows one GM and one Non-GM checkpoint from each Procgen cohort on the same OOD level. Every checkpoint has to meet its definition on its own, so the table gives the worst checkpoint of each cohort and not a cohort mean. Every cohort holds the three checkpoints of the table, and unless an analysis states otherwise, all checkpoints of both cohorts and the randomly initialized networks are scored on identical states, so a cohort difference cannot come from a difference in the states. Each Non-GM checkpoint is trained with the proxy uninformative about reward. The black ATTRIBUTE MAZE cohort holds the three black seeds with the lowest $\widehat { \operatorname* { P r } } ( \mathrm { g o a l } \mid \mathrm { d e c } )$ . The MAZE controls meet the capability threshold under sampled actions, whereas under greedy actions two of the three fal just below it $( \hat { C } _ { e }$ of 0.370 and 0.396), so the capability match between the MAZE cohorts rests on sampled actions. The same holds for the texture ATTRIBUTE MAZE controls 4361 and 4362 $( \hat { C } _ { e }$ of 0.284 and 0.319 under greedy actions, against 0.908 and 0.934 under sampled ones). They are still the weakest controls, as on $\Lambda _ { \mathrm { O O D } }$ they commit on 0.340 to 0.590 of episodes against 0.698 to 0.872 for their GM counterparts, so comparisons with them rest on Pr(goal | dec) rather than on reach. TODO (Julia): keep, replace or retrain this cohort. The COIN GM checkpoints are not the final checkpoints of their runs but are selected by training step (Appendix H). In COIN the learning rate decays linearly to zero, so the GM checkpoints, taken 20M to 120M steps into 200M-step runs, carry a higher learning rate than the Non-GM checkpoints at the end of 40M-step runs.

Each robot cohort holds three seeds that meet their definition at the final checkpoint of the run. In ATTRIBUTETABLE, GM seeds 1 and 5 and Non-GM seed 1 miss the capability threshold at 6M steps (0.278, 0.254 and 0.595) and are not carried. Figure 44 shows all carried robot seeds on the same conflict start states.

![](images/6b4faf5996cc1a6dcf46075aaa08b3499d201174df9cacdbd5b34d7c7ef4c065.jpg)

(a)  
![](images/b1c6726674d96104b84a8d96eece8cac6fb0fcddc80ba346435f4bc554f81146.jpg)

AttributeTable: blue cube (goal) vs red sphere (proxy)

![](images/1ef111491e012a9f8cf888d485ac2dc1130c0f7261ab32ee7631efbeda3c6e86.jpg)

![](images/12a6f3fe8d3785501de13537a28ff3eeca6246c320585195e2a297fd9e8f16a8.jpg)

![](images/0ed4812342a5117d0639ca42fcc30dc68b7c7489202bdcaed2b0b559b56e2cc0.jpg)

(b)  
SortZones: pad with the cross (goal) vs green pad (proxy)  
![](images/8e17d84e390bb8a80b006f7685269bdd9c35295b44be85a9d3c6c6ca6217eac7.jpg)

(c)  
LocationReach: green ball (goal) vs training corner (proxy)  
![](images/19730e2271fbd6c2d0088061d2f80d774bb7531274106fa032f737573e3cba8b.jpg)

![](images/dddc1417bca1c87b7d4f1f02e85c386e3cc637306f113aa3a13c5a4ca5e58454.jpg)

![](images/aae64f501f8ac05fdbf1b30135ffa7d133c73664a8f6374f7eade9b7027940a0.jpg)

![](images/40bbeece90dbe89e152d46c5b04622d6861f6c15769cb75cecd16ea97377029c.jpg)

(d)  
PushToMarker: marker (goal) vs table centre (proxy)  
![](images/2c25dbd9672072b3d4a52a06fb825d4d6e6768e79a17bde8086740918fa54f48.jpg)

![](images/aa18a6d89f7dd9f1ad7f92f582fe0241659468500eea25a07e38aad1ac2b2f11.jpg)

![](images/f6500b5b7b1d3542997ac3a5694df79a7f58776e6a13686f214f3086e2b7dc2d.jpg)

![](images/4a4bc4c61cbdaf025de7901976cdcf5c2cd5948cdcd75d839f84e1dad2b1afd7.jpg)

(e)  
![](images/efa6c33df740a61f77102e7d14df9f699caea7eea60ed7a13959b0a02460a0a1.jpg)

WalkToFlag: flag (goal) vs tower (proxy)  
![](images/de33839187be0dd9064cb16f4b73792dc6bf0fcebb5dbad015ac7b9f2382f21b.jpg)

![](images/b7b10810edd3ad295603b50014c14611984cb2e8bfc4e7b04027f6720ffb068d.jpg)

![](images/e25c13b59bba508bec843875e56687abeea35b3452db4135b92da94094703107.jpg)  
Figure 44: Robot conflict rollouts of every carried seed, seen from above. The three left panels of each row show the first three conflict start states of a task, each run by the three GM and three Non-GM checkpoints of Table 24 with deterministic actions, and an asterisk marks a start state on which the seeds of a cohort disagree. The right panel places every start state and seed by the distance of the episode’s end to goal and to proxy, divided by their separation, over the first 250 conflict episodes of each arm task and the first 128 of WALKTOFLAG.

## G RELATED WORK IN DETAIL

Goal misgeneralization. Langosco et al. (2022) give the first explicit demonstrations of goal misgeneralization in deep reinforcement learning, and all three of our Procgen environments build on theirs. We rename the environments for readability. MAZE is their Maze I (cheese in the top-right corner), COIN is their CoinRun experiment (Procgen CoinRun with the coin at the right end during training), and ATTRIBUTE MAZE is the simplified Maze environment of Ramanauskas & S¸ims¸ek (2023), itself a simplification of their Maze II (Appendix A). Shah et al. (2022) argue that a correct specification does not guarantee a correct goal, with examples well beyond reinforcement learning. Langosco et al. (2022) also sort their failures into directional proxies (CoinRun), location proxies (Maze I, our MAZE) and observation ambiguity (Maze II, colour versus shape). In Procgen our three regimes match these kinds one to one. The policy shortcut in COIN goes with the directional proxy, functional target substitution in MAZE with the location proxy, and cue competition in ATTRIBUTE MAZE with observation ambiguity. Their CoinRun critic values the end of the level while the actor keeps moving right past it, so a directional and a location proxy can coexist in one agent, and we do not claim that COIN separates a rightward rule from endpoint-directed behavior. Dorn et al. (2023) frame goal misgeneralization as implicit goal conditioning, and mitigation work changes training or oversight, through minimax-regret level design (Abdel Sadek et al., 2025) or a mentor the agent can ask for help (Trinh et al., 2024). We build on the behavioral split of Langosco et al. (2022) and extend it in two directions. First, we supply its internal counterpart. For each kind of proxy we locate where inside the policy the intended goal drops out, on the very environments on which the evidence for goal misgeneralization rests and which had so far been characterized by behavior or by single-agent case studies. Second, we add new environments in which goal misgeneralization arises, five robot tasks, discussed below.

Goals in reinforcement learning and reward hacking. The word “goal” is used in different senses across the reinforcement-learning literature. In goal-conditioned reinforcement learning a goal is an input to the value function or policy (Schaul et al., 2015), often a parametrized region of the state space to be reached (Florensa et al., 2018), and Colas et al. (2022) define a goal as a goal embedding paired with a goal-achievement function, which is a goal-conditioned reward. There the reward defines the goal. In the goal-misgeneralization literature the goal is a task-level outcome, and the reward is a learning signal that marks where that outcome was reached, so a correct reward can leave the pursued goal underdetermined (Langosco et al., 2022; Shah et al., 2022). We use the second sense (Definition 1). Our agents receive no goal input, and the goal we look for is whatever the trained network treats as its target. Reward hacking, also called specification gaming, is an adjacent but different failure, in which agents exploit gaps in a misspecified reward and gain proxy reward at the cost of true reward (Pan et al., 2022). Laidlaw et al. (2025) define it through a proxy reward that correlates with the true reward on the states of a reference policy and stops correlating as optimization moves the policy away. Rewards learned from preferences can likewise rely on non-causal distractor features (Tien et al., 2023), so the error sits in the learned reward. In goal misgeneralization the reward is fixed and correct, the coupling that breaks is between the intended goal and another regularity of the training levels, it breaks because the test levels change, and the agent stays capable while it pursues the wrong target. Our diagnostics follow this distinction. They keep the reward fixed and change the world, through shifted levels and world edits, rather than comparing a proxy reward with a true one.

Goal misgeneralization and related failures in robot control. Robot policies are currently trained mainly by imitation learning or by reinforcement learning, and the two routes fail in related but different ways. In imitation learning the analogous failure is causal confusion (de Haan et al., 2019). A feature that is systematically collinear with the expert’s actions across the demonstrations explains those actions as well as their true cause, so the cloned policy can rely on it and fails once its own actions break the correlation. In reinforcement learning the intended goal is defined by the reward. A proxy that co-locates with the rewarded goal during training lets a policy that pursues the proxy collect the full training reward, which is goal misgeneralization and the setting of this paper. In imitation learning, Kim et al. (2023) name goal misgeneralization in robot manipulation explicitly, on an RLBench arm task whose target appears in unseen positions at test time. However, they train on expert demonstrations rather than by reinforcement learning, and their CLIP-based rewards change training rather than diagnose a trained policy. Policies trained by imitation show proxy-like failures at scale. Vision-language-action models trained on LIBERO keep moving toward memorized object positions when objects are displaced (Zhou et al., 2025) and largely ignore their language instructions (Fei et al., 2026), and Xing et al. (2025) trace shortcut learning in generalist robot policies to low diversity within and fragmentation across the sub-datasets they are trained on. To our knowledge, no prior work shows goal misgeneralization in robot policies trained by re inforcement learning. We induce it in five robot tasks, four for a robot arm and one for a quadruped, under the same conditions as in Procgen, a correct reward and a proxy that co-locates with the goal during training (Appendix A). Beyond inducing it, we analyse it with the same internal diagnosis as the Procgen agents and extend our Procgen findings to these tasks.

Interpretability of RL agents. Hilton et al. (2020) interpret and edit a CoinRun agent with a different convolutional architecture from ours. Mini et al. (2023) dissect a MAZE policy trained with the cheese anywhere in the top-right 5 × 5 region. They find channels that track the cheese location, partially retarget the policy by editing them, and show that subtracting a cheese vector suppresses cheese-seeking while adding it does not increase it. Our GM agents are trained with the cheese on the exact corner cell, a regime they set aside because such agents did not pursue cheese outside the top-right 5 × 5 region, and our network is more than five times smaller. Our probe null therefore concerns a more strongly misgeneralizing regime and our probe sites, several of them globally average pooled, and does not contradict their finding. Their asymmetry is consistent with our MAZE steering, where the anticorrective push removes Non-GM cheese pursuit and no push creates it in GM agents (Appendix D.1.1). Ramanauskas & S¸ ims¸ek (2023) simplify Maze II into a 5 × 5 maze without dead ends and with pure-colour sprites, the simplified Maze environment that we call ATTRIBUTE MAZE, and find that the preferred cue depends on the seed and the background, as in our cohorts. They attribute the colour preference to goal detection through one specific colour channel. Our swaps do not rule out that account. Exchanging sprites also changes which RGB channels carry line-shaped pixels, so our shape and colour cues could be one channel-specific shape detector rather than two separate features. Unlike theirs, our GM agents also train with a penalized second object. Few studies of RL agents test whether an internal representation is used rather than only decodable. Bush et al. (2025) probe a model-free Sokoban agent for planning concepts and show through interventions on its representations that the resulting plans change its actions, and the edits of Hilton et al. (2020) and Mini et al. (2023) test use in the same causal sense. Trinh et al. (2024) report evidence that the internal state of their CoinRun agent does not represent the coin at all. We find coin bearing decodable in both COIN cohorts, well above randomly initialized networks, with different checkpoints and a different probe, so the two findings are not in direct conflict. We differ from these single-agent case studies by applying one set of assays to eight environments with matched Non-GM controls, which is what shows that one behavioral label covers different internal failures.

Representation versus use: probes, interventions and Fisher sensitivity. Linear probes measure what an intermediate layer makes linearly available (Alain & Bengio, 2017). Hewitt & Liang (2019) show that probe accuracy means little without a control task, and untrained networks already expose some features linearly (Hermann & Lampinen, 2020). Our randomly initialized networks and input-level controls follow that logic, as control models and control inputs rather than control tasks. Decodability does not establish use. Elazar et al. (2021) test use by removing a property from the representation and measuring behavior, and interchange interventions set an internal variable to its value under another input (Geiger et al., 2021). Our world edits are the input-level analogue, with swaps as the canonical counterfactual in ATTRIBUTE MAZE. The Fisher information is the local metric of the KL divergence (Amari, 1998; Martens, 2020), and the natural policy gradient uses the Fisher of a policy with respect to its parameters (Kakade, 2001). Trust-region policy optimization bounds the KL divergence between successive policies and builds its Fisher matrix as the Hessian of that divergence (Schulman et al., 2015), and PPO, which trains all our agents, keeps the same constraint in approximate form (Schulman et al., 2017). The policy Fisher is thus the second-order measure of how far the action distribution moves, independent of the parametrization in which the change is made, and it is the quantity these methods constrain during training. We build on this geometry with two changes. We take the Fisher with respect to the activation that the policy head reads rather than with respect to the parameters, and we evaluate it along the displacement that a world edit induces in that activation, which splits the policy’s sensitivity into how strongly it reads the edited direction and how far the edit moves the representation (Section 3.2). To our knowledge, we are the first to use the policy’s own Fisher geometry to follow an intended goal through a policy, from where it is represented to whether the policy reads it.

Steering and readout recovery. Activation addition steers a model by adding the activation difference between two contrasting inputs (Turner et al., 2023b), and Mini et al. (2023) apply it to the MAZE agent. Our direction is state-local, built from the world-edit pair at each conflict state, with anticorrective and random controls. Shortcut learning names decision rules that succeed in training and fail to transfer (Geirhos et al., 2020). We use “policy shortcut” more narrowly, for an action regularity rather than an input feature. In supervised classification with spurious correlations, retraining only the last layer restores robustness because the core features survive (Kirichenko et al., 2023). Our rewiring tests the same idea, relearning only the computation on top of features that may have survived, at two depths of the network. Lee et al. (2023) show that the subset of layers that is best to fine-tune depends on the type of distribution shift. They compare layer subsets by accuracy after fine-tuning on a small labelled target dataset, whereas we freeze the representation at $h _ { \mathrm { p r e } }$ or $h _ { \mathrm { f c } } ,$ , relearn only the computation downstream of it by behavior cloning, and evaluate the result in closed loop. Prior repairs of a proxy-following agent either continue RL training under a new reward, on the training levels with a reward model fit to LLM preferences for a maze agent biased toward the top-right region and paused at 32% of training (Nait El Barj & Sautory, 2024) or on unlabeled test levels with an inferred reward (Armstrong et al., 2023), or prune the proxy concept from a concept-bottleneck agent, which requires that architecture (Delfosse et al., 2024). Subtracting a top-right vector from a region-trained maze agent has little effect on its pull toward the top-right (Turner et al., 2023a). These methods aim to repair the agent. We use recovery instead as a test of the diagnosis, and call it recovery rather than mitigation because it acts after training and does not prevent the failure. Steering asks whether the goal can be restored through the wiring the policy already has, and rewiring asks whether the frozen representation still carries goal information that new wiring can use. Rewiring trains its readout by behavior cloning, as a measuring device, while the representation stays the one that reinforcement learning produced. To our knowledge, we are the first to recover goal-directed behavior in misgeneralized agents post hoc and to explain from the internal diagnosis where recovery works and why it fails elsewhere.

## H IMPLEMENTATION AND TRAINING DETAILS

Why we reuse the published recipes. Our aim is to explain the goal misgeneralization that Langosco et al. (2022) and Ramanauskas & S¸ims¸ek (2023) observed, not to find it again under new conditions. We therefore train every Procgen agent with the recipe of the paper that reported the failure, the code of Langosco et al. (2022) for MAZE and COIN and the code of Ramanauskas & S¸ ims¸ek (2023) for ATTRIBUTE MAZE. A different recipe would add a second difference between our agents and theirs, and any mechanism we found could then be a property of that recipe rather than of the phenomenon. The robot tasks have no published recipe. The arm agents train with the PPO baseline for pixel observations that ships with ManiSkill3 (Tao et al., 2025), and the WALK-TOFLAG agents with a port of the same implementation to Isaac Lab (Mittal et al., 2025). Across the arm tasks only the budget, the discount and the rollout length differ, and they were set on indistribution success alone. The WALKTOFLAG settings were fixed before training, with the number of parallel robots chosen on simulation throughput. Within each environment, GM and Non-GM agents share this recipe and differ only in the training distribution.

Environments. The three platforms are described in Appendix A. We add here only what the training code needs. The three Procgen games (Cobbe et al., 2020) have 64 × 64 RGB observations and 15 discrete actions. MAZE and COIN are the variants of Langosco et al. (2022). MAZE trains on their modified Maze game (hard mode, maze sides 3 to 25), in which rand region= k confines the cheese to the top-right k × k block. Its OOD game, the original Procgen Maze, puts the cheese on any free cell. COIN trains on CoinRun (Cobbe et al., 2019) with the coin at the level end (random percent= 0), and is evaluated on the variant of Langosco et al. (2022) that puts the coin in a random section. ATTRIBUTE MAZE uses the 5 × 5 easy-mode two-object maze of Ramanauskas & S¸ims¸ek (2023), in which one object gives +10 and the other −10 and the agent starts in the bottom-left cell. Every Procgen run trains on levels [0, 10<sup>5</sup>). The arm tasks simulate 256 environments in parallel and the quadruped task 512, and every episode draws a new scene from the cohort’s training distribution. The LOCATIONREACH and SORTZONES agents were trained with ManiSkill3 3.0.1 and SAPIEN 3.0.3.

Training. Every agent is trained from scratch with PPO (Schulman et al., 2017) and generalized advantage estimation (Schulman et al., 2016), with the settings of Table 25. Procgen agents use the hard-500 preset. MAZE and COIN use the code of Langosco et al. (2022), derived from Lee (2020), and ATTRIBUTE MAZE uses the code of Ramanauskas & S¸ims¸ek (2023). The two codebases share the network, the PPO update and the preset. The learning rate decays linearly to zero over the default run length, so COIN GM checkpoints, taken 20M to 120M steps into 200M-step runs, saw higher learning rates than Non-GM checkpoints at the end of 40M-step runs (Appendix F). The robot agents keep a constant learning rate, so their budgets do not change the schedule. One set of robot runs departs from the budget of its task. The PUSHTOMARKER Non-GM seeds 1 and 2 were not capable at 50M steps and were continued to 100M (Appendix A.2.1).

Checkpoints and controls. Non-GM controls change only the training distribution (Tables 2 and 26, and in MAZE rand region= 25 covers every maze). ATTRIBUTE MAZE controls train on eight object pairs, 32 environments each, with line presence and yellow presence uncorrelated and two same-colour pairs that only shape separates. The line is rewarded whenever it appears, and no training pair sets a non-yellow line against a yellow gem, so the trap is new to both cohorts. COIN checkpoints were chosen by step during qualification. Every other checkpoint is the final save of its run. Robot seeds that failed qualification enter no cohort (Table 24).

Evaluation levels. MAZE qualification uses levels from 200,000 (training distribution) and 700,000 (OOD). COIN qualification uses levels from 1 (GM) and 700,000 (Non-GM). All other MAZE and COIN evaluation uses levels from 1,000,000. ATTRIBUTE MAZE trap preference and rewiring evaluation use levels from 100,000. The 196-state conflict bank reuses layouts of levels 2,003 to 41,991, inside the training range, so for these states only the object assignment is held out. Robot evaluation episodes are drawn with reset seed 10,000, apart from the training seeds, and the agents act with the mean of their action distribution.

Table 25: PPO settings by platform, identical for GM and Non-GM agents. $\mathrm { M } = 1 0 ^ { 6 }$ environment steps. KL early stop ends an update once the approximate KL of a minibatch exceeds the threshold.
<table><tr><td rowspan="2"></td><td>Procgen</td><td colspan="2">ManiSkill3 arm</td><td>Isaac Lab quadruped</td></tr><tr><td>MAZE, COIN, ATTRIBUTE MAZE</td><td>LOCATIONREACH, ATTRIBUTETABLE</td><td>PUSHTOMARKER, SORTZONES</td><td>WALKTOFLAG</td></tr><tr><td>Network</td><td>IMPALA CNN</td><td>Nature CNN, state branch</td><td>Nature CNN, state branch</td><td>Nature CNN, state branch</td></tr><tr><td>Action distribution</td><td>categorical, 15 actions 200M (MAZE, COIN GM),</td><td>diagonal Gaussian, 8-d 6M</td><td>diagonal Gaussian, 8-d 50M (PUSHTOMARKER,</td><td>diagonal Gaussian, 3-d 30M</td></tr><tr><td>Budget per run</td><td>40M (CoIN Non-GM), 5M (ATTRIBUTE MAZE)</td><td></td><td>100M for Non-GM seeds 1 and 2), 20M (SORTZONES)</td><td></td></tr><tr><td>Envs × rollout steps</td><td>256 × 256</td><td>256 × 50</td><td>256 × 100</td><td>512 × 64</td></tr><tr><td>Transitions per update</td><td>65,536</td><td>12,800</td><td>25,600</td><td>32,768</td></tr><tr><td>Epochs × minibatches</td><td>3 × 8</td><td>8 × 8</td><td>8 × 8</td><td>4 × 8</td></tr><tr><td>Discount γ</td><td>0.999</td><td>0.8</td><td>0.95</td><td>0.99</td></tr><tr><td>GAE λ</td><td>0.95</td><td>0.9</td><td>0.9</td><td>0.95</td></tr><tr><td>Clip €</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Value-loss coefficient</td><td>0.5, clipped at €</td><td>0.5, unclipped</td><td>0.5, unclipped</td><td>0.5, unclipped</td></tr><tr><td>Entropy coefficient</td><td>0.01 0.5</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Gradient-norm clip</td><td></td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Adam learning rate</td><td>5 × 10−4, linear to 0</td><td> $3 \times 1 0 ^ { - 4 } .$  constant</td><td> $3 \times 1 0 ^ { - 4 }$  , constant</td><td>3 × 10 4 , constant</td></tr><tr><td>Adam €, weight decay</td><td> ${ { 1 0 } ^ { - 5 } } ,$  none</td><td> $1 0 ^ { - 5 } ,$  none</td><td> ${ { 1 0 } ^ { - 5 } } ,$  none approximate KL &gt; 0.2</td><td> $1 0 ^ { - 5 } ,$  none approximate KL &gt; 0.2</td></tr><tr><td>KL early stop Advantage</td><td>none per rollout</td><td>approximate  $\mathrm { K L > 0 . 2 }$  per minibatch</td><td>per minibatch</td><td>per minibatch</td></tr><tr><td>normalization Reward normalization</td><td>divided by a running return scale (γ = 0.99), clipped at ±10</td><td>none</td><td>none</td><td>none</td></tr></table>

Table 26: Training runs. Budget is the planned run length and the checkpoint the analysed save, in environment steps. Every checkpoint is the final save of its run except for COIN GM.
<table><tr><td>Environment</td><td>Network</td><td>Cohort</td><td>Seeds</td><td>Budget</td><td>Checkpoint step</td></tr><tr><td rowspan="2">MAZE</td><td>IMPALA</td><td>GM</td><td>1080-1082</td><td>200M</td><td>200,015,872</td></tr><tr><td>IMPALA</td><td>Non-GM</td><td>1090-1092</td><td>200M</td><td>200,015,872</td></tr><tr><td rowspan="4">COIN</td><td>IMPALA</td><td>GM</td><td>6033</td><td>200M</td><td>20,054,016</td></tr><tr><td>IMPALA</td><td>GM</td><td>6037</td><td>200M</td><td>120,061,952</td></tr><tr><td>IMPALA</td><td>GM</td><td>6051</td><td>200M</td><td>55,050,240</td></tr><tr><td>IMPALA</td><td>Non-GM</td><td>7001-7003</td><td>40M</td><td>40,042,496</td></tr><tr><td rowspan="4">ATTRIBUTE MAZE</td><td>IMPALA</td><td>GM, black</td><td>6313,6315, 6317</td><td>5M</td><td>5,046,272</td></tr><tr><td>IMPALA</td><td>Non-GM, black</td><td>4371-4373</td><td>5M</td><td>5,046,272</td></tr><tr><td>IMPALA</td><td>GM, texture</td><td>6301-6303</td><td>5M</td><td>5,046,272</td></tr><tr><td>IMPALA</td><td>Non-GM, texture</td><td>4361-4363</td><td>5M</td><td>5,046,272</td></tr><tr><td rowspan="3">PUSHTOMARKER</td><td>Nature CNN</td><td>GM</td><td>0-2</td><td>50M</td><td>49,996,800</td></tr><tr><td>Nature CNN</td><td>Non-GM</td><td>0</td><td>50M</td><td>49,996,800</td></tr><tr><td>Nature CNN</td><td>Non-GM</td><td>1,2</td><td>100M</td><td>99,993,600</td></tr><tr><td>LOCATIONREACH</td><td>Nature CNN</td><td>GM, Non-GM</td><td>1-3</td><td>6M</td><td>5,990,400</td></tr><tr><td rowspan="2">ATTRIBUTETABLE</td><td>Nature CNN</td><td>GM</td><td>2-4</td><td>6M</td><td>5,990,400</td></tr><tr><td>Nature CNN</td><td>Non-GM</td><td>3-5</td><td>6M</td><td>5,990,400</td></tr><tr><td>SORTZONES</td><td>Nature CNN</td><td>GM, Non-GM</td><td>0-2</td><td>20M</td><td>19,993,600</td></tr><tr><td>WALKTOFLAG</td><td>Nature CNN</td><td>GM, Non-GM</td><td>0-2</td><td>30M</td><td>29,982,720</td></tr></table>

Network implementation. Section B.1 describes the network conceptually. Its implementation is the feed-forward IMPALA CNN of Espeholt et al. (2018) as written by Lee (2020). Each of the three blocks applies a $3 \times 3$ convolution with stride 1 and padding $1 , \mathrm { ~ a ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ max-pool with stride 2 and padding 1, and two residual units, with 16, 32 and 32 output channels, so the spatial size falls from 64 to 32, 16 and 8. Each residual unit computes x + conv(ReLU(conv(ReLU(x)))) with 3×3 convolutions that keep the channel count, and no ReLU sits between a block’s first convolution and its max-pool. The only nonlinearities outside the residual units are the ReLU applied to $H ^ { ( 3 ) }$ before flattening into $h _ { \mathrm { p r e } }$ and the ReLU after the 2048 → 256 linear layer that gives $h _ { \mathrm { f c } } .$ . The policy head (256 → 15, orthogonal initialization with gain 0.01) and the value head (256 → 1, gain 1) are linear. The value head enters PPO training only through the value loss, and it is unused in the behavior-cloning rewiring of Appendix D.2. The convolutional trunk holds 97,600 parameters (9,728, 41,632 and 46,240 per block), the fully connected layer 524,544, the policy head 3,855 and the value head 257, for 626,256 in total. The robot network is the Nature CNN (Mnih et al., 2015)

of the ManiSkill3 baseline. Three convolutions (32 channels $8 \times 8$ with stride 4, 64 channels $4 \times 4$ with stride 2, 64 channels $3 \times 3$ with stride 1, each followed by a ReLU) and a 256-unit linear layer with ReLU encode the image, a 256-unit linear layer encodes the body state, and actor and critic each read the concatenated 512-d feature through one hidden layer of 512 units. The policy outputs the mean of a Gaussian whose log standard deviation is a learned vector initialized at −0.5.

## H.1 MODIFICATIONS TO THE TRAINING CODE

The analyses in this paper need to read and edit the state of a running episode, which neither Procgen fork supports beyond saving and restoring it. Procgen already generates each level deterministically from its index and can serialize and restore the full game state, so a matched counterfactual starts from a restored copy of the unedited state and changes only the edited object. We added the edits themselves to the game code, and none of them changes an episode that is not edited. In MAZE a call places or removes the cheese at a given cell and another turns a wall cell into a free cell, which gives the cheese relocations and the structural edit that opens the top-right region (Appendix C.2). The reward is then tied to the cell that currently holds the cheese, which in an unedited level is the cell it was generated on. In COIN a call places or removes the coin, and a game option hides the coin sprite while keeping its reward, which gives the coin relocations of Appendix C.2 and the visible-versus-hidden dependence score of Appendix F. Both games also report the agent cell and the cheese or coin position at every step, which supply the probe targets and the agent-position controls of Appendix C.1. In ATTRIBUTE MAZE the duel game of Ramanauskas & S¸ims¸ek (2023) already takes the rewarded and the penalized object as parameters, and we added calls that read and set the cell of each object. These calls build the shape and colour swaps on a fixed layout, and the diversified bank places the agent by writing its position into the restored state (Appendix E). Our ATTRIBUTE MAZE trainers split the 256 parallel environments into equal groups, one per object pair, so that a single PPO run sees all pairs of its training distribution at once. Beyond this, the training code gained checkpoint resumption and loss logging, which leave the PPO update unchanged. The robot tasks are our own environments, written against the ManiSkill3 and Isaac Lab APIs. The arm trainer adds to the ManiSkill3 baseline only a training-distribution argument, logging and resumption, and the WALKTOFLAG trainer reimplements the same update for Isaac Lab. All code needed to reproduce the agents and analyses will be released with the paper.

## AI USE STATEMENT

Generative artificial intelligence aided in the development of this paper. Specifically, it helped with drafting and polishing prose from the authors’ dictation, polishing mathematics, literature search, writing code for analyses and figures, and auditing reported numbers against the underlying results. We have reviewed and verified all LLM-generated text, code and analyses. All ideas behind the method and the scientific claims are products of the authors’ own minds.