# Q-SHAPED OPTIONS FOR HIERARCHICAL REINFORCEMENT LEARNING

Clarisse Wibault

Antoine Gorceix Antonio Leon Villares Alexey Zakharov Evangelos Chatzaroulas´ Michael Matthews Eduardo Pignatelli Jakob Foerster

BOLD

University of Oxford

clarisse.wibault@magd.ox.ac.uk

## ABSTRACT

Learning to tackle long-horizon, goal-conditioned tasks requires an agent to reason over extended timescales and act across a broad range of states. In principle, Hierarchical Reinforcement Learning (HRL) addresses both challenges through the interaction between action (temporal) and state (spatial) abstraction. First, using an action abstraction to represent temporally extended behaviour as options reduces the effective decision horizon. Second, enabling different state abstractions at each level of the decision process permits greater data aggregation for learning. However, realising these two benefits of a hierarchical policy depends on learning an appropriate action abstraction. Current HRL algorithms fail in one of two ways. Some discard distinctions between options needed for optimal control, undermining hierarchy altogether. Others retain unnecessary distinctions, preserving horizon reduction, but forfeiting coarser state abstraction. In this work, we characterise three desiderata for an action abstraction. We introduce Q-Shaped Options (QSO) to address all three. QSO builds on an architecture with distinct state-value functions, Q functions and policies at each level of the hierarchy. It learns the action abstraction between consecutive levels as a shared encoder shaped by their respective Q functions. The low-level Q function uses the option as a goal, encouraging the abstraction to retain distinctions necessary for optimal control. The high-level Q function uses it as an action, encouraging unnecessary distinctions to be discarded. Across offline goal-conditioned locomotion and manipulation environments, QSO learns semantically meaningful option spaces and outperforms baselines, achieving non-zero performance in tasks where all other evaluated algorithms fail. Our code is open sourced at https://github.com/CWibault/QSO.

## 1 INTRODUCTION

Learning to tackle long-horizon, goal-conditioned (Kaelbling, 1993; Schaul et al., 2015) tasks requires an agent to reason over extended timescales and act across a broad range of states (Sutton et al., 1999; Park et al., 2026). In goal-conditioned two-tiered Hierarchical Reinforcement Learning (HRL) a high-level policy outputs a high-level action or option (Sutton et al., 1999) conditioned on the current state and goal; a low-level policy then conditions on this option as well as the current state to output a primitive action (Figure 1a). In maze navigation, for example, the option might be to reach “a point two metres down”, while the primitive action could be applying torques to an individual joint.

In principle, HRL (Barto and Mahadevan, 2003) addresses both challenges of long-horizon, goalconditioned tasks through the interaction between action (temporal) and state (spatial) abstraction (Konidaris, 2019). First, by using an action abstraction to represent temporally extended behaviour as options, a hierarchical policy reduces the effective decision horizon (Park et al., 2026). The high-level policy reasons over options instead of primitive actions, while the low-level policy reasons to pursue the option rather than the distant goal. Second, by enabling different state abstractions at each level of the hierarchy (Figure 1a), a hierarchical policy can achieve coarser state abstraction than a non-hierarchical policy if each level discards different information irrelevant to its part of the decision process (Konidaris, 2019; Konidaris et al., 2018). Intuitively, this permits greater data aggregation for learning. For example, the high-level can aggregate all examples at a certain coordinate irrespective of pose, which is too fine-grained for high-level control, while the low-level can aggregate all examples with a certain pose irrespective of coordinates, since reaching “a point two metres down” requires the same actions wherever the agent is.

![](images/6cbd1c3693f791811056b751f1aaaf7d4e82fd09c0c700ed4223c731c00c42df.jpg)  
Figure 1: (a) HRL permits different state abstractions $\phi _ { h } ( s _ { t } , g )$ and $\phi _ { l } ( s _ { t } , \omega _ { t } )$ at each level of the decision process. (b) Decoupled policy training. (c) Different methods for learning the action abstraction $\phi _ { \omega } \big ( \tau _ { t : t + k } \big )$ .

However, realising these two benefits of a hierarchical policy depends on learning an appropriate action abstraction. To improve stability and avoid collapsing to degenerate solutions (Bacon et al., 2017), decoupled approaches (Ajay et al., 2020; Park et al., 2023; 2026) train each policy separately (Figure 1b) using fixed-length trajectory segments. The action abstraction maps these segments to the option space, serving as actions for the high-level policy and goals for the low-level policy during training. The issue is that current HRL algorithms learn abstractions that fail in one of two ways. Some discard distinctions between options that are needed for optimal control (Park et al., 2023; Li et al., 2006). This limits the quality of the recoverable policy and undermines hierarchy altogether. Others retain distinctions between options that are irrelevant for control (Park et al., 2026; Ajay et al., 2020; Ravindran and Barto, 2002). This preserves horizon reduction but forfeits coarser state abstraction, as functionally equivalent behaviours remain distinct outputs of the high-level policy and distinct inputs to the low-level one.

In this work, we characterise three desiderata for an action abstraction to enable horizon reduction and data aggregation. We introduce Q-Shaped Options (QSO) to address all three. QSO builds on an architecture (Park et al., 2026) with separate state-value functions, Q functions and policies at each level of the hierarchy. It learns the action abstraction between consecutive levels as an encoder shared between the two levels and shaped by their respective Q functions. To our knowledge, QSO is the first HRL algorithm to learn an action abstraction through a Q function. We encode only the current and target states to represent the outcome of a trajectory segment rather than the exact state-action sequence that produced it. We shape the abstraction via the low-level Q function to separate options with different low-level Q values (Li et al., 2006), retaining distinctions necessary for optimal control. Finally, we shape the abstraction via the high-level Q function to provide an inductive bias for discarding unnecessary distinctions between functionally equivalent behaviour. Across offline (Levine et al., 2020; Lange et al., 2012) goal-conditioned locomotion and manipulation environments, QSO learns semantically meaningful option spaces (Figures 2, 6 and 7) and outperforms baselines (Table 2), achieving non-zero performance in tasks where all other evaluated algorithms fail (Table 7). We summarise our contributions as follows.

1. We provide a formal definition of a control-sufficient action abstraction and prove that exact low-level Q-shaping satisfies control sufficiency.

2. We extend a Probably Approximately Correct bound on offline error in a finite-state and finiteaction MDP to motivate discarding unnecessary distinctions between options.

3. We develop Q-Shaped Options (QSO), to our knowledge the first HRL algorithm to learn the action abstraction through a Q function, yielding a control-sufficient action abstraction that also discards unnecessary distinctions between options.

## 2 RELATED WORKS

As summarised in Table 1 and shown in Figure 1c, prior algorithms using decoupled training approaches differ substantially both in what the action abstraction encodes, ranging from a single target state $s _ { t + k }$ to the full k-step state-action trajectory segment $\tau _ { t : t + k }$ , and the method used to learn the action abstraction.

<table><tr><td>Algorithm</td><td>Encodes</td><td>Method</td><td>Preserves</td><td>Discards</td></tr><tr><td>HIQL</td><td> $s _ { t } , s _ { t + k }$ </td><td>V</td><td>No</td><td>Yes</td></tr><tr><td>OPAL</td><td> $\tau _ { t : t + k }$ </td><td>β-VAE (Surrogate Objective)</td><td>Yes</td><td>No</td></tr><tr><td>SHARSA</td><td> $s _ { t + k }$ </td><td></td><td>Yes</td><td>No</td></tr><tr><td>DSHARSA</td><td> $s _ { t + k }$ </td><td></td><td>Yes</td><td>No</td></tr><tr><td>ASO</td><td> $s _ { t + k }$ </td><td></td><td>Yes</td><td>No</td></tr><tr><td>QSO</td><td> $s _ { t } , s _ { t + k }$ </td><td> $Q _ { h }$  and Qt</td><td>Yes</td><td>Yes</td></tr></table>

Table 1: Properties of the action abstraction $\phi _ { \omega } \mathbf { : }$ what the abstraction encodes; the method used to learn the abstraction; whether the objective encourages preserving distinctions necessary for optimal control and whether the abstraction empirically exhibits discarding unnecessary distinctions.

HIQL (Park et al., 2023) learns the action abstraction through its single action-free state-value function $V ( s _ { t } , \phi _ { \omega } ( s _ { t } , g ) )$ . However, value equivalence does not imply action equivalence (Li et al., 2006), as two state-goal pairs can share a value, and hence option, while requiring entirely different immediate actions (Figure 2a). This means that the abstraction can discard information necessary for optimal control.

SHARSA and DSHARSA (Park et al., 2026) simply bypass learning an action abstraction by using subgoal options $\omega _ { t } = s _ { t + k }$ (Konidaris, 2019), which specify the entire state that the low-level policy should reach. The “encoder” is just an identity mapping. While these “options” retain the information required for low-level control, they distinguish functionally equivalent behaviour occurring at different points in the state space (Figure 2c), limiting coarser state abstraction and hence data aggregation for learning.

OPAL (Ajay et al., 2020) uses a surrogate objective to learn an option space $\omega _ { t } \sim \phi _ { \omega } ( \omega _ { t } \mid \tau _ { t : t + k } )$ with a β-VAE trained to reconstruct the entire action sequence. This means that the option encodes how a segment was executed, instead of its outcome. Similarly to SHARSA and DSHARSA, such an encoding hinders coarser state abstraction as two trajectory segments with the same outcome may be represented differently just because their executions differ (Figure 2b).

QC (Li et al., 2026), which builds on action chunking (Zhao et al., 2023), is closely related to our work as it also provides temporal abstraction, but rather than having a multi-level structure, a nonhierarchical policy outputs an action chunk $\pi ( a _ { t : t + k } \mid s _ { t } , g )$ . Because a single level is responsible for the entire decision process, this prevents coarser state abstraction.

## 3 PRELIMINARIES

Problem Setting We consider a goal-conditioned Markov Decision Process (MDP) (Sutton et al., 1998; Schaul et al., 2015) defined by the tuple $\mathcal { M } : = ( p _ { s _ { 0 } } , p _ { g } , \mathcal { S } , \mathcal { G } , \mathcal { A } , \mathcal { T } , r , \bar { \gamma } )$ , where $p _ { s _ { 0 } }$ is the initial state distribution, $p _ { g }$ is the goal distribution, $s , \mathcal { G }$ and A respectively denote the state, goal and action spaces with ${ \mathcal { G } } = { \mathcal { S } } ,$ , and $\tau$ is the transition function. $r ( s , g ) : = - ( 1 - \beta ( s , g ) )$ ) and $\bar { \gamma } ( s , g ) : = \gamma \cdot ( 1 - \beta ( s , g ) )$ , where $\gamma \in [ 0 , 1 )$ ) is a constant, are the reward and discount or continuation functions (Andrychowicz et al., 2017; Sutton et al., 2011; White et al., 2012) defined through a goal-achievement indicator $\beta : \mathcal { S } \times \mathcal { G }  \{ 0 , 1 \}$ (specified per environment in Appendix H): the agent receives −1 per step until the goal is reached, after which the return terminates. At the beginning of the episode, a state $s _ { 0 }$ is sampled from $p _ { s _ { 0 } }$ , and a goal $g$ is sampled from $p _ { g }$ . The goal is fixed for the entire episode. At each timestep $t \geq 0 ,$ , an agent takes an action $a _ { t }$ conditioned on its current state $s _ { t }$ and goal state $^ { g , }$ and deterministically transitions to a new state $s _ { t + 1 } = T ( s _ { t } , a _ { t } )$ ; we adopt the fully observed deterministic MDP framework as used in Park et al. (2025) for simplicity. Our aim is to learn a policy $\pi : S \times \mathcal { G }  \Delta ( . A )$ , where $\Delta ( \mathcal { A } )$ denotes the space of probability distributions over A, from a fixed dataset $\mathcal { D }$ of N state-action trajectories of length H, collected by an arbitrary policy, that maximises $\begin{array} { r } { \mathbb { E } _ { \pi } [ \sum _ { t = 0 } ^ { \infty } ( \prod _ { i = 0 } ^ { t - 1 } \bar { \gamma } ( s _ { i } , g ) ) r ( s _ { t } , g ) ] } \end{array}$ ]. The return is bounded in $[ - 1 / ( 1 - \gamma )$ , 0] and is maximised by minimising the discounted time to reach the goal. In what follows we write the constant γ in place of $\bar { \gamma } ( s , g )$ and omit goal-termination masking for clarity.

Hierarchical Reinforcement Learning In two-tiered goal-conditioned HRL (Barto and Mahadevan, 2003; Park et al., 2023) the aim is to learn a high-level policy $\pi _ { h } ( \omega _ { t } \mid s _ { t } , g )$ that outputs an option $\omega _ { t } \in \Omega$ , where Ω denotes the option space, conditioned on the current state $s _ { t } \in S$ and goal $g \in { \mathcal { G } } ;$ a low-level policy $\pi _ { l } ( a _ { t } \mid s _ { t } , \omega _ { t } )$ then conditions on this option as well as the current state to output a low-level action $a _ { t } \in { \mathcal { A } }$ (Figure 1a), such that the overall policy is defined by the following:

$$
\pi ( a _ { t } \mid s _ { t } , g ) : = \int _ { \omega _ { t } \in \Omega } \pi _ { l } ( a _ { t } \mid s _ { t } , \omega _ { t } ) \pi _ { h } ( \omega _ { t } \mid s _ { t } , g ) \mathrm { d } \omega _ { t } .
$$

While options may have fixed or variable temporal extent, in this work we focus on fixed-length options, which do not collapse to degenerate solutions (Bacon et al., 2017) and simplify the separation of the two decision processes with a fixed temporal and discount relationship. During training, an action abstraction $\phi _ { \omega }$ is learned and used to encode (a subset of) the k-step state-action trajectory segment $\tau _ { t : t + k } = ( s _ { t } , a _ { t } , \cdot \cdot \cdot , s _ { t + k } )$ into an option $\omega _ { t } = \phi _ { \omega } ( \tau _ { t : t + k } )$ . The high-level policy is trained to output this option, while the low-level policy is trained to condition on this option. At deployment, the action abstraction is no longer required, since the two levels communicate entirely within option space.

## 4 Q-SHAPED OPTIONS

Our aim is to learn an action abstraction that represents the outcome of a trajectory segment rather than the exact state-action sequence that produced it, retains distinctions between options necessary for optimal control, and discards unnecessary distinctions.

## 4.1 DESIDERATA

The action abstraction should represent the outcome of a trajectory segment rather than the exact state-action sequence that produced it.

We want to prevent potentially suboptimal dataset execution from becoming a part of the option’s representation. To enable two different executions with the same outcome to map to the same option, we restrict the abstraction to be a function of the state-achieved-state pair $\left( { { s _ { t } } , { s _ { t + k } } } \right)$ , rather than the entire state-action trajectory segment. We include the current state as well as the achieved state to enable representing relative rather than absolute outcome (see Desideratum 3). HIQL also encodes $\left( { { s _ { t } } , { s _ { t + k } } } \right)$ , but motivates this choice empirically.

Desideratum 1. Represent outcome rather than exact execution.

QSO: Action abstraction encodes $\left( { { s _ { t } } , { s _ { t + k } } } \right)$ instead of $\tau _ { t : t + k } .$

The action abstraction should be control-sufficient i.e. preserve distinctions between options necessary for optimal control.

We want two target states that require different actions from the same current state map to different options, so that a policy conditioning on the representation instead of the target state can still be optimal for both. To address control sufficiency, we shape the abstraction through the low-level Q function. If two admissible target states require different actions from the same current state, the low-level Q function must represent different sets of action values. Minimising the Bellman error therefore pressures the action abstraction to map the current and target state pairs to different points in the option space. Conversely, if two trajectory segments have functionally equivalent relative outcomes but different dataset executions, they can share an option if their sets of action values are compatible, even if one was executed well and the other poorly. This can enable the low-level policy to draw on the better execution at deployment. Definition 1 formalises this notion of a control-sufficient action abstraction, while Proposition 1 connects this property to the low-level Q shaping objective. We refer to Appendix B for the proof.

```latex
Definition 1. Control-sufficient action abstraction.
Let $\mathcal { G } _ { k } ( s ) \subseteq \mathcal { S }$ denote the set of admissible target states for the k-step low-level decision problem
from state $s ,$ and let $\Pi ^ { \star } ( s , g _ { s } ) : = \operatorname * { a r g m a x } _ { a } Q ^ { \star } ( s , g _ { s } , a )$ denote the set of optimal actions of the
original MDP. An action abstraction $\phi _ { \omega }$ is control-sufficient if, for all $s \in \bar { \mathcal { S } }$ and $g _ { s } , g _ { s } ^ { \prime } \in \mathcal { G } _ { k } ( s )$
$\phi _ { \omega } ( s , g _ { s } ) = \phi _ { \omega } ( s , g _ { s } ^ { \prime } ) \quad \Longrightarrow \quad \Pi ^ { \star } ( s , g _ { s } ) = \Pi ^ { \star } ( s , g _ { s } ^ { \prime } ) .$
In particular, there exists a low-level policy $\pi _ { l }$ such that $\pi _ { l } ( \cdot \textit { \textbf { | } } s , \phi _ { \omega } ( s , g _ { s } ) )$ is supported on
$\Pi ^ { \star } ( s , g _ { s } )$ for every admissible pair $( s , g _ { s } )$
Note that Q-shaping is stronger than control sufficiency, as it distinguishes between options whose
action values differ only by an additive offset or in the ranking of suboptimal actions, even if these
differences do not affect the optimal action.
Proposition 1. Low-level Q-shaping implies control sufficiency.
Suppose there exists a low-level critic $Q _ { l }$ such that $Q _ { l } \bigl ( s , \phi _ { \omega } ( s , g _ { s } ) , a \bigr ) = Q ^ { \star } \bigl ( s , g _ { s } , a \bigr )$ for all
$s \in \mathcal { S } , g _ { s } \in \mathcal { G } _ { k } ( s )$ and $a \in A ,$ , i.e. $\phi _ { \omega }$ is a $Q ^ { \star }$ -irrelevance abstraction in the sense of Li et al.
(2006). Then $\phi _ { \omega }$ is control-sufficient.
Desideratum 2. Preserve distinctions between options necessary for optimal control.
QSO: Learn action abstraction through low-level Q function: $Q _ { l } \big ( s _ { t } , \phi _ { \omega } \big ( s _ { t } , g _ { s } \big ) , a _ { t } \big )$
The action abstraction should discard unnecessary distinctions between options.
We want trajectory segments with similar outcomes to map to the same option, so that experience can
be aggregated across the state space. This enables conservative extrapolation: aggregating different
examples across different levels of the hierarchy to produce an action not observed for a particular
state-goal pair. Compressing a finite option space or organising a continuous option space in a
semantically meaningful way reduces the effective complexity of the hierarchical decision problem
and can reduce the distribution mismatch between the optimal policy and the dataset by aggregating
occupancy mass across equivalent state-goal and state-option pairs. We refer to Appendix C for an
example, and Appendix D for a proposition that formalises this intuition by deriving and analysing
the Probably Approximately Correct bound on offline error for a hierarchical policy with joint state
and action abstraction.
To encourage discarding unnecessary distinctions between options, we additionally shape the option
space through the high-level Q function. Functionally equivalent state-target pairs induce similar
high-level action values, so distinguishing between them provides little predictive benefit to $Q _ { h }$ . Note
that, unlike control sufficiency, this invariance is not explicitly enforced by the objective, instead
simply providing an inductive bias towards it under shared function approximation and gradient
based optimisation. Empirically, we find that this is sufficient to form coherent clusters in the option
space. Discarding unnecessary distinctions between options also motivates conditioning the action
abstraction on the current state, as then the abstraction only needs to separate target states if they
require different actions from the same current state rather than if they require different actions from
any admissible current state.
```

## Desideratum 3. Discard unnecessary distinctions between options.

QSO: Learn action abstraction $\phi _ { \omega }$ through the high-level Q function: $Q _ { h } ( s _ { t } , g , \phi _ { \omega } ( s _ { t } , s _ { t + k } ) )$

## 4.2 PRACTICAL IMPLEMENTATION

Joint Action Abstraction and Value Learning We learn the action abstraction between consecutive levels of the hierarchy as an encoder shared between the two levels and jointly shaped by their

respective Q functions:

$$
\operatorname* { m i n } _ { Q _ { l } , Q _ { h } , \phi _ { \omega } } \mathcal { L } _ { Q _ { l } , \phi _ { \omega } } + \mathcal { L } _ { Q _ { h } , \phi _ { \omega } }
$$

where the critic losses are the following, conservative Implicit Q Learning (Kostrikov et al., 2021) critic losses:

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { Q } _ { l } , \phi _ { \omega } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , g _ { s } ) \sim \mathcal { D } } \bigg [ \Big ( Q _ { l } \big ( s _ { t } , \phi _ { \omega } ( s _ { t } , g _ { s } ) , a _ { t } \big ) - r _ { l } ( s _ { t } , g _ { s } ) - \gamma V _ { l } \big ( s _ { t + 1 } , \omega _ { t + 1 } \big ) \Big ) ^ { 2 } \bigg ] , } \\ { \mathcal { L } _ { \boldsymbol { Q } _ { h } , \phi _ { \omega } } = \mathbb { E } _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } } \bigg [ \Big ( Q _ { h } \big ( s _ { t } , g , \phi _ { \omega } ( s _ { t } , s _ { t + k } ) \big ) - R _ { h } \big ( s _ { t : t + k } , g \big ) - \gamma V _ { h } \big ( s _ { t + k } , g \big ) \Big ) ^ { 2 } \bigg ] , } \end{array}
$$

and the normalised k-step option return is $\begin{array} { r } { R _ { h } ( s _ { t : t + k } , g ) = \frac { 1 - \gamma } { 1 - \gamma ^ { k } } \sum _ { t ^ { \prime } = 0 } ^ { k - 1 } \gamma ^ { t ^ { \prime } } r _ { h } ( s _ { t + t ^ { \prime } } , g ) . } \end{array}$

We use a continuous option space, parameterising the action abstraction with an MLP followed by length normalisation (Park et al., 2023) onto a hypersphere. Unlike a discrete option space, this avoids imposing arbitrary boundaries between semantically similar options, while the length normalisation keeps proposed options in-distribution for the low-level policy. At deployment, the high-level policy resamples an option at every timestep, allowing the agent’s intent to slowly shift across the option space. We use the same discount factor at both levels so that each level looks $1 / ( 1 - \gamma )$ of its own decisions ahead, and normalise the k-step option return so that a full-length option has the same cost as a single primitive action. This keeps the two Q functions on comparable scales and hence avoids a hyperparameter to balance their losses. Ideally, $\bar { k } = 1 / ( 1 - \gamma )$ so that one option spans exactly the horizon that the low-level reasons over. In practice, k is bounded by the data (Park et al., 2023; 2026; Li et al., 2026).

```latex
Algorithm 1 Q-Shaped Options (QSO) Training
Initialise option representation $\phi _ { \omega } ( s _ { t } , g _ { s } )$ , high-level value $V _ { h } ( s _ { t } , g )$ , critic $Q _ { h } ( s _ { t } , g , \omega )$ and flow policy
$\pi _ { h } ( \omega \mid s , g )$ , low-level value $V _ { l } ( s _ { t } , \omega )$ , critic $Q _ { l } ( s _ { t } , \omega , a _ { t } )$ and flow policy $\pi _ { l } ( a _ { t } \mid s , \omega )$ and target option
and critic networks $\tilde { \phi } _ { \omega } , \tilde { Q } _ { h } , \tilde { Q } _ { l }$
repeat
▷ Dataset Sampling and Reward Labelling.
Sample states from D. Determine $a _ { t } , s _ { t + 1 } , s _ { t + k } .$
Sample high-level goals $g$ and low-level targets $g _ { s }$ using Equations 8 and 9.
Determine rewards $r _ { l } ( s _ { t } , g _ { s } )$ and $r _ { h } \left( s _ { t : t + k } , g \right)$
▷ Joint Action Abstraction and Value Learning.
Encode high-level options $\omega \gets \phi _ { \omega } ( s _ { t } , s _ { t + k } )$ and target options $\tilde { \omega } \gets \tilde { \phi } _ { \omega } ( s _ { t } , s _ { t + k } )$
Encode low-level options $\phi _ { \omega } ( s _ { t } , g _ { s } )$ and target options $\tilde { \phi } _ { \omega } ( s _ { t } , g _ { s } )$
Train $V _ { h }$ and $V _ { l }$ using Equations 4 and 6.
Train $Q _ { h }$ and $Q _ { l }$ and $\phi _ { \omega }$ using Equations 5 and 7.
Update $\tilde { \phi } _ { \omega } , \tilde { Q } _ { h } , \tilde { Q } _ { l }$ using exponential moving averages.
▷ Policy Extraction.
Train $\pi _ { h } ^ { \mathrm { A W R } }$ and $\pi _ { l } ^ { \mathrm { B C } }$ using Equations 10 and 11.
until convergence
```

Policy Extraction Given the learned action abstraction and value functions, we extract a policy at each level using conservative policy extraction algorithms (Peng et al., 2019; Ghasemipour et al., 2021). The high-level policy uses Rejection Sampling (RS) over an advantage-weighted behaviour cloning (BC) flow policy; the low-level uses BC alone. We use advantage-weighting at the high-level because its dataset contains a large proportion of random goals, needed to cover the otherwise out-of-distribution state-goal pairs faced at deployment. While pure BC learns to ignore the goals, advantage-weighting restores goal-conditioning and leads to better proposals for RS. This issue does not arise at the low-level, for which we sample almost no random target states. In fact, we empirically found that advantage-weighting and RS degraded low-level performance by substantially reducing the effective batch size and exploiting errors in the Q function. Even though the low-level is trained with BC, we still get conservative extrapolation, as the high-level policy can pair a state with a better option than observed at that state during training; the low-level then pursues the option with behaviour learned elsewhere in the dataset. Figure 17 and Table 9 in Appendix H show performance and goal sensitivity when ablating high-level policy extraction.

Algorithm 2 Q-Shaped Options (QSO) Deployment   
while episode not terminated do   
▷ High-level Rejection Sampling.   
Sample $\omega ^ { 1 } \cdot \cdot \cdot \omega ^ { N } \sim \pi _ { h } ^ { \mathrm { A W R } } ( \cdot \mid s _ { t } , g ) .$   
Select ω $\gets \operatorname { a r g m a x } _ { \omega ^ { 1 } \cdots \omega ^ { N } } Q _ { h } ( s _ { t } , g , \omega ^ { i } )$   
▷ Low-level Behaviour Cloning.   
Sample $a _ { t } \sim \pi _ { l } ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , \omega )$   
end while

We provide pseudocode for QSO training and deployment in Algorithms 1 and 2, and full equations and dataset sampling methods in Appendix G. We refer to Appendix A for our choice of IQL and policy extraction algorithms.

## 5 EXPERIMENTS

We first evaluate QSO’s downstream performance from a fixed offline dataset to isolate policy learning from data collection. To assess our second and third desiderata, we introduce ASO, which shares QSO’s policy extraction but, like SHARSA and DSHARSA, uses absolute subgoal options instead of a learned action abstraction. This isolates the effect of QSO’s action abstraction: discarding distinctions necessary for control should yield a worse policy than ASO, whereas discarding only unnecessary ones should improve performance through greater data aggregation. We then analyse the learned option space directly, testing whether functionally equivalent behaviours map to similar points while functionally distinct ones remain separated. Finally, we ablate QSO’s components to identify their individual contributions.

Experimental Protocol We evaluate on the most challenging OGBench (Park et al., 2025) environments using the default datasets, as OGBench is a standard offline RL benchmark with complex, long-horizon, goal-conditioned tasks. We compare against hierarchical algorithms from Section 2. As non-hierarchical baselines, we include IQL with DDPGBC (Fujimoto and Gu, 2021) policy extraction, the strongest non-hierarchical baseline (Park et al., 2025; 2024), and QC. All baselines use the hyperparameters from their original implementations. Apart from the option dimension, for which we use the OGBench default, QSO introduces no additional hyperparameters, so we use those of SHARSA and DSHARSA. Environment, baseline and hyperparameter details are in Appendix H.

<table><tr><td></td><td colspan="5">Non-Hierarchical</td><td colspan="10">Hierarchical</td><td rowspan="2">QSO / Best Mean</td></tr><tr><td>Environment</td><td>IQL</td><td></td><td>QC</td><td></td><td>HIQL</td><td></td><td>OPAL</td><td></td><td>SHARSA</td><td></td><td>DSHARSA</td><td></td><td>ASO</td><td></td><td>QSO</td></tr><tr><td>antmaze-giant-stitch-v0</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>54 [48,</td><td>60]</td><td>53 [52,</td><td>55]</td><td>68 [61,</td><td>76]</td><td>72</td><td>[67, 77]</td><td>68</td><td>[66,</td><td>70] 68</td><td>[66,</td><td>70]</td></tr><tr><td>antmaze-teleport-stitch-v0</td><td>47 [44,</td><td>51] 37</td><td>[32, 40]</td><td>36</td><td>[34, 39]</td><td>69 [64,</td><td>74]</td><td>53 [52,</td><td>55]</td><td>58 [55,</td><td>62]</td><td>66</td><td>[62, 69]</td><td>54</td><td>[52, 57]</td><td>79%</td></tr><tr><td>humanoidmaze-giant-stitch-v0</td><td>1 [1, 1]</td><td>0</td><td>[0, 0]</td><td>6 [4,</td><td>8]</td><td>82 [78,</td><td>85]</td><td>30 [27,</td><td>33]</td><td>30 [27,</td><td>32]</td><td>44</td><td>[38, 48]</td><td>90</td><td>[88, 92]</td><td>100%</td></tr><tr><td>cube-double-play-v0</td><td>20 [17, 23]</td><td>72</td><td>[70, 74]</td><td>7</td><td>[5, 9]</td><td>29 [26,</td><td>33]</td><td>72 [68,</td><td>77]</td><td>80</td><td>[75, 84]</td><td>68</td><td>[65, 70]</td><td>70</td><td>[65, 74]</td><td>87%</td></tr><tr><td>cube-triple-play-v0</td><td>6 [6, 7]</td><td>23</td><td>[19, 27]</td><td>16</td><td>[14, 18]</td><td>3 [2,</td><td>4]</td><td>14 [10,</td><td>18]</td><td>14 [13,</td><td>16]</td><td>4</td><td>[4,4]</td><td>22</td><td>[20, 23]</td><td>92%</td></tr><tr><td>cube-quadruple-play-v0</td><td>0 [0, 0]</td><td>2</td><td>[1, 2]</td><td>3 [1,</td><td>5]</td><td>0 [0,</td><td>0]</td><td>0</td><td>[0, 1]</td><td>0 [0, 1]</td><td></td><td></td><td>0 [0, 0]</td><td></td><td>4 [4, 5]</td><td>100%</td></tr><tr><td>scene-play-v0</td><td>48 [45,</td><td>51] 82</td><td>[81, 83]</td><td>41 [35,</td><td>46]</td><td>84 [78,</td><td>87]</td><td>92 [90,</td><td>93]</td><td>89 [87,</td><td>91]</td><td>94</td><td>[91, 95]</td><td>85</td><td>[82, 90]</td><td>91%</td></tr><tr><td>puzzle-3x3-play-v0</td><td>100 [99,</td><td>100] 81</td><td>[46, 99]</td><td>48</td><td>[46, 49]</td><td>70 [65,</td><td>74]</td><td>78 [71,</td><td>87]</td><td>85 [81,</td><td>90]</td><td>76</td><td>[72, 80]</td><td>72</td><td>[69, 75]</td><td>72%</td></tr><tr><td>puzzle-4x4-play-v0</td><td>71 [61, 78]</td><td>25</td><td>[23, 27]</td><td></td><td>0 [0, 1]</td><td>64</td><td>[61, 68]</td><td>74</td><td>[67, 79]</td><td>72 [69,</td><td>75]</td><td>69</td><td>[65, 75]</td><td>92</td><td>[81, 100]</td><td>100%</td></tr><tr><td>puzzle-4x5-play-v0</td><td>14 [11, 15]</td><td>16</td><td>[13, 19]</td><td></td><td>4 [3, 5]</td><td>24 [22,</td><td>25]</td><td>17</td><td>[15, 20]</td><td>18 [18,</td><td>19]</td><td>14</td><td>[11, 18]</td><td>15</td><td>[11, 20]</td><td>65%</td></tr><tr><td>puzzle-4x6-play-v0</td><td>8 [5, 10]</td><td>16</td><td>[14, 19]</td><td>5 [4,</td><td>6]</td><td>17 [15,</td><td>18]</td><td>14</td><td>[10, 16]</td><td>16 [14,</td><td>17]</td><td>11</td><td>[8, 15]</td><td>18</td><td>[15, 22]</td><td>100%</td></tr><tr><td>Average over 11 Environments</td><td>29 [27,</td><td>30] 32</td><td>[29, 34]</td><td>20</td><td>[20, 20]</td><td>45</td><td>[44, 46]</td><td>46</td><td>[44, 49]</td><td>49</td><td>[48, 49]</td><td>47</td><td>[46, 48]</td><td>54</td><td>[51, 56]</td><td>100%</td></tr></table>

Table 2: Summary of results on selected OGBench environments using the default datasets. Success rate averaged over 5 evaluation tasks after 1M training steps. Mean and 95% bootstrap confidence interval over 4 seeds.

Results Table 2 shows that QSO achieves the highest average success rate across the 11 environments, 54% against 49% for the next best method (DSHARSA). Particularly notable is its performance in humanoidmaze, which has the largest action space dimension (21, against at most 8 in the other environments) where it achieves 90% against 82% for the next best method and three times that of DSHARSA. In puzzle-4x4 it achieves 92% against at most 74% for any baseline. In cube-quadruple, the hardest manipulation environment, QSO is the only method with non-zero success on two of the five tasks (Table 7 in Appendix H). QSO performs consistently across environments, achieving at least 65% of the best success rate in all environments, while every other algorithm drops below 35% of the best in at least one.

O Right Δ Up Left Down Dither Start Random k-step windows  
![](images/fe4c26b7013ef28343cd6396c3b973261213b5a13965e63b03a33b31f0bb3aee.jpg)

![](images/bc0adf253fe220deda6cc3878036ca6d540f0b3e4f275e1cfaed64b343de5099.jpg)

![](images/4ed4e6fa206b62197baaf7d2a907697e2d86c31206028088fea8173c9ff07cd3.jpg)

![](images/389f699225a945a6d42804a55b39db50edc2b2469f877016a574fba8d5562dc8.jpg)

![](images/9bff6a61bfeb37735dffac3638cfa238b250cb4dd6019196de8f109f72c05752.jpg)

![](images/848aab80472eb59474ca1ae0ebf439b92091a2603277532538fbb513b4438de4.jpg)  
Figure 2: PCA (top) of the action abstraction for antmaze-giant fitted on random segments, showing different groups of segments with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) in units of median distance between two random segments. QSO is the only algorithm that consistently groups functionally equivalent behaviours (Desideratum 3) while separating behaviour groups (Desideratum 2).

Is the action abstraction semantically meaningful? We test whether QSO maps functionally equivalent behaviours to similar parts of the option space by identifying groups of semantically similar behaviours in antmaze-giant, cube-double, and cube-quadruple (Figures 2, 6, and 7). For each behaviour, we randomly sample trajectory sequences from the validation dataset and compare PCA projections and whitened distances of their encodings. An invariant encoder should produce a clear block-diagonal similarity structure. QSO exhibits this structure, whereas HIQL also assigns high similarity between behaviour groups, and OPAL and absolute subgoal options produce low similarity within behaviour groups.

How necessary is joint option shaping by both the high and low-level Q functions? Table 3 and Figures 3 and 9 indicate that joint shaping is important for both the learned action abstraction and downstream performance. Neither Q function alone recovers the clean semantic structure obtained when both are used. Figure 8 shows that shaping with the high-level only performs well in antmaze-giant but degrades significantly in cube-double. We hypothesise that for cube-double continued high-level shaping increasingly removes distinctions that are irrelevant to high-level value but necessary for fine-grained control; without the low-level Q function to preserve this information, the resulting abstractions become insufficient for the low-level policy.

![](images/d8042d4990ae3c89ad7ec9cd51f82e9fe1b45832f08fa1fc36cc2b609f875728.jpg)

![](images/f4a57e4aba13efc40132df6f029d764992446e3fd9819ff1e482f374b8f296fd.jpg)

![](images/c4a6d6bbc66e938b857a4496df932446f88ac871a5d13417832d9f62ec09aa7b.jpg)  
Figure 3: PCA (top) of QSO’s action abstractions for gradient flow ablations for antmaze-giant fitted on random trajectory sequences, showing different groups of three k-step trajectory sequences with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) for those groups in units of median distance between two random trajectory sequences. Both high and low-level gradient flows are required to group functionally equivalent behaviours while separating the behaviour groups.

Do the decoupled levels form distinct state abstractions? For the high and low-level decision processes, we measure how much each network’s output varies with slow and fast moving dimensions; a larger percentage share of the output variance being associated to a particular dimension means that the network is more dependent on that dimension. We would expect the low-level decision process to focus more on high frequency (fast-moving) dimensions, and the high-level decision process on low frequency (slow-moving) ones. As expected, the two levels focus on different states (Table 8 in Appendix H). For example, the high-level state-value function depends almost entirely on slow-moving dimensions (88% for antmaze-giant and 74% for cube-double), while the low-level policy depends much more on fast-moving ones (59% and 77% respectively). We refer to Appendix H for further discussion.

<table><tr><td>Environment</td><td colspan="2">Default</td><td colspan="2">Ablation</td><td colspan="6">Option Resampling Ablation</td></tr><tr><td></td><td>QSO</td><td>Low-Level Only</td><td>High-Level Only</td><td>Every Step</td><td>5%</td><td>(2 Steps)</td><td></td><td>25% (7 Steps)</td><td></td><td>50% (13 Steps)</td></tr><tr><td>antmaze-giant-stitch-v0</td><td>68 [66, 70]</td><td>38 [38, 40]</td><td>69 [67, 72]</td><td>68 [66, 70]</td><td>68</td><td>[66, 70]</td><td></td><td>62 [61, 64]</td><td></td><td>54 [47, 60]</td></tr><tr><td>cube-double-play-v0</td><td>70 [65, 74]</td><td>70 [67, 73]</td><td>47 [42, 50]</td><td>70 [65, 74]</td><td></td><td>69 [63, 73]</td><td></td><td>50 [44, 55]</td><td></td><td>14 [10, 18]</td></tr></table>

Table 3: Q shaping and option resampling ablations of QSO. Success rate averaged over 5 evaluation tasks after 1M training steps. Mean and 95% bootstrap confidence interval over 4 seeds.

![](images/f4814122502f8e96c5db1506318bbcf4deb427e332fab34df71733c6d8222e4a.jpg)  
Figure 4: QSO’s sampled options gradually shift across the option space during deployment. L2 distance between two consecutively sampled options (every 10 steps), whitened against the distance between two randomly sampled options. Dotted blue shows the distance between two random trajectory sequence embeddings whitened against the distance between two randomly sampled options. Mean and 95% confidence interval over 4 seeds, cut off where the third quartile of successful episodes has ended.

How does the option vary during deployment? Figure 4 shows that sampled options gradually shift across the option space during deployment: the distance between consecutively sampled options is roughly 30% of that between two randomly sampled options in antmaze-giant, and 25% of that in cube-double. Although consecutive options may partly be similar due to smoothness of the high-level policy, the above analysis showing that option-space proximity reflects semantic similarity suggests that it also corresponds to gradual changes in intent. The distance between two random trajectory sequence embeddings whitened against the distance between two randomly sampled options (dotted blue) being approximately 1 confirms that the small consecutive distances reflect genuinely slow variation rather than a collapsed policy. Table 3 ablates how long the option is held for at deployment: the rapid decrease in performance for cube-double suggests that the option encodes significant information about the immediate next action.

We provide ablations on the option representation input and dimension, goal sampling, reward labelling, policy extraction, and temporal horizon in Appendix H.

## 6 DISCUSSION AND FUTURE WORK

We introduce QSO, which, unlike prior algorithms, learns a control-sufficient action abstraction that discards unnecessary distinctions between options. Future work will extend QSO to do targeted exploration at the option-level in online RL. A limitation of QSO arises in stochastic environments: by representing an option retrospectively, the high-level critic evaluates the option conditioned on its realised outcome, which can lead to optimistic bias. Although we note that this issue is shared by all hindsight-based hierarchical methods including HIQL, OPAL, SHARSA and DSHARSA, future work could address this through progress-adaptive high-level control.

## ETHICS STATEMENT

This work aims to advance machine learning, specifically hierarchical reinforcement learning, and is evaluated only in simulated environments using public datasets. It may have broader societal consequences, but none of them must be specifically highlighted here.

## ACKNOWLEDGEMENTS

The authors would like to thank Darius Muglich for helpful advice and comments on the project.

Clarisse Wibault is funded by EPSRC grant EP/W524311/1. Antoine Gorceix is funded by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines grant EP/Y035070/1 and Iconic Interactive Ltd. Alexey Zakharov is supported by the Clarendon Fund in partnership with the Oxford-Basil Reeve Graduate Scholarship at the University of Oxford. Eduardo Pignatelli is funded by University College London. Jakob Foerster is partially funded by the UKRI grant EP/Y028481/1, as well as being supported by the JPMC and Amazon Research Awards.

## REPRODUCIBILITY STATEMENT

Our code is open sourced at https://github.com/CWibault/QSO. Hyperparameters and exact implementation details can be found in Appendices G and H.

## AI DISCLOSURE

We used AI to render figures, debug code, and check spelling and grammar, but all ideas, analysis, and written content are our own. We take full responsibility for the final submission.

## REFERENCES

Leslie Pack Kaelbling. Learning to achieve goals. In Ijcai, volume 2, pages 1094–8, 1993.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In International conference on machine learning, pages 1312–1320. PMLR, 2015.

Richard S Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999.

Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar, and Sergey Levine. Horizon reduction makes rl scalable. Advances in Neural Information Processing Systems, 38:8350–8389, 2026.

Andrew G Barto and Sridhar Mahadevan. Recent advances in hierarchical reinforcement learning. Discrete event dynamic systems, 13(4):341–379, 2003.

George Konidaris. On the necessity of abstraction. Current opinion in behavioral sciences, 29:1–7, 2019.

George Konidaris, Leslie Pack Kaelbling, and Tomas Lozano-Perez. From skills to symbols: Learning symbolic representations for abstract high-level planning. Journal ofArtificial Intelligence Research, 61:215–289, 2018.

Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.

Anurag Ajay, Aviral Kumar, Pulkit Agrawal, Sergey Levine, and Ofir Nachum. Opal: Offline primitive discovery for accelerating offline reinforcement learning. arXiv preprint arXiv:2010.13611, 2020.

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. Hiql: Offline goal-conditioned rl with latent states as actions. Advances in Neural Information Processing Systems, 36:34866– 34891, 2023.

Lihong Li, Thomas J Walsh, and Michael L Littman. Towards a unified theory of state abstraction for mdps. AI&M, 1(2):3, 2006.

Balaraman Ravindran and Andrew G Barto. Model minimization in hierarchical reinforcement learning. In International symposium on abstraction, reformulation, and approximation, pages 196–211. Springer, 2002.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In Reinforcement learning: State-of-the-art, pages 45–73. Springer, 2012.

Qiyang Li, Zhiyuan Paul Zhou, and Sergey Levine. Reinforcement learning with action chunking. Advances in Neural Information Processing Systems, 38:55518–55553, 2026.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

Richard S Sutton, Andrew G Barto, and Andrew Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

Richard S Sutton, Joseph Modayil, Michael Delp, Thomas Degris, Patrick M Pilarski, Adam White, and Doina Precup. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In The 10th international conference on autonomous agents and multiagent systems-volume 2, pages 761–768, 2011.

Adam White, Joseph Modayil, and Richard S Sutton. Scaling life-long off-policy learning. In 2012 ieee international conference on development and learning and epigenetic robotics (icdl), pages 1–6. IEEE, 2012.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned rl. In International Conference on Learning Representations, volume 2025, pages 94937–94982, 2025.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Seyed Kamyar Seyed Ghasemipour, Dale Schuurmans, and Shixiang Shane Gu. Emaq: Expected-max q-learning operator for simple yet effective offline and online rl. In International Conference on Machine Learning, pages 3682–3691. PMLR, 2021.

Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in neural information processing systems, 34:20132–20145, 2021.

Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar. Is value learning really the main bottleneck in offline rl? Advances in Neural Information Processing Systems, 37:79029–79056, 2024.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International conference on machine learning, pages 387–395. Pmlr, 2014.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actorcritic methods. In International conference on machine learning, pages 1587–1596. Pmlr, 2018.

David Abel, David Hershkowitz, and Michael Littman. Near optimal behavior via approximate state abstraction. In International Conference on Machine Learning, pages 2915–2923. PMLR, 2016.

Norman Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite markov decision processes. arXiv preprint arXiv:1207.4114, 2012.

Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. arXiv preprint arXiv:2006.10742, 2020.

Remi Munos and Csaba Szepesv´ ari. Finite-time bounds for fitted value iteration.´ Journal ofMachine Learning Research, 9(5), 2008.

Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International conference on machine learning, pages 1042–1051. PMLR, 2019.

Sham Machandranath Kakade. On the sample complexity of reinforcement learning. University of London, University College London (United Kingdom), 2003.

Tor Lattimore and Marcus Hutter. Pac bounds for discounted mdps. In International Conference on Algorithmic Learning Theory, pages 320–334. Springer, 2012.

Arnaud Robert, Ciara Pike-Burke, and Aldo Faisal. Sample complexity of goal-conditioned hierarchical reinforcement learning. Advances in Neural Information Processing Systems, 36:62696–62712, 2023.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

## A EXTENDED PRELIMINARIES

Value Learning To learn our state-value and Q functions, we adopt Implicit Q-Learning (Kostrikov et al., 2021), a conservative value learning algorithm that evaluates the state-value function under the dataset action that leads to the maximum value rather than evaluating the Q-value under the current policy or using the maximum over actions, as is typically done in online RL (Watkins and Dayan, 1992; Mnih et al., 2015). This avoids querying the Q function on out-of-distribution actions, to which the Q function can assign erroneously high values (Levine et al., 2020). The state-value function and critic are trained with the following losses:

$$
\begin{array} { r l } & { \mathcal { L } _ { V } = \mathbb { E } _ { ( s _ { t } , a _ { t } , g ) \sim \mathcal { D } } \left[ \ell _ { \nu } ^ { 2 } \left( V ( s _ { t } , g ) - \tilde { Q } ( s _ { t } , g , a _ { t } ) \right) \right] , } \\ & { \mathcal { L } _ { Q } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , g ) \sim \mathcal { D } } \left[ \left( Q ( s _ { t } , g , a _ { t } ) - r ( s _ { t } , g ) - \gamma V ( s _ { t + 1 } , g ) \right) ^ { 2 } \right] , } \end{array}
$$

where $\ell _ { \nu } ^ { 2 }$ denotes the expectile loss $\ell _ { \nu } ^ { 2 } ( x ) : = \mid \nu - \mathbf { 1 } \cdot ( x < 0 ) \mid \cdot x ^ { 2 }$ . For $\nu > 0 . 5$ , underestimating a high Q value is penalised more strongly than overestimating a low one, so $V ( s _ { t } , g )$ is pulled to the upper end of the dataset Q values.

Policy Extraction To extract our policies, we use Advantage Weighted Regression (AWR) (Peng et al., 2019) and Rejection Sampling (RS) (Ghasemipour et al., 2021), which are conservative policy extraction algorithms that constrain the policy to sample the highest quality actions within the dataset. As the name suggests, AWR does advantage-weighted behaviour cloning of dataset actions:

$$
\mathcal { L } _ { \pi } = - \mathbb { E } _ { ( s _ { t } , a _ { t } , g ) \sim \mathcal { D } } \left[ \exp \left( \frac { Q ( s _ { t } , g , a _ { t } ) - V ( s _ { t } , g ) } { \alpha } \right) \cdot \log \pi ( a _ { t } \mid s _ { t } , g ) \right] ,
$$

while RS learns a behaviour-cloned policy to imitate the dataset and then defines the policy by choosing the action that maximises the Q function:

$$
\begin{array} { r } { \pi : = \mathrm { a r g m a x } _ { a _ { t } \in \{ a _ { t } ^ { i } \} , a _ { t } ^ { i } \sim \pi ^ { \mathrm { B C } } ( \cdot | s _ { t } , g ) } Q ( s _ { t } , g , a _ { t } ^ { i } ) . } \end{array}
$$

Unlike algorithms that optimise the policy by differentiating through the learned Q-function (Silver et al., 2014; Lillicrap et al., 2015; Fujimoto et al., 2018; Fujimoto and Gu, 2021), these conservative policy extraction algorithms avoid a direct source of Q function exploitation, whereby errors in the learned Q values are propagated into the policy. However, a non-hierarchical policy extracted with AWR or RS is bounded by the quality of the data at that state: where a region of the state space contains only low-quality actions, there is nothing “better” among the proposals to upweight.

## B LOW-LEVEL Q-SHAPING IMPLIES CONTROL SUFFICIENCY

## Proof 1. Low-level Q-shaping implies control sufficiency.

If $\phi _ { \omega } ( s , g _ { s } ) = \phi _ { \omega } ( s , g _ { s } ^ { \prime } ) = : \omega$ , then for every a, $Q ^ { \star } ( s , g _ { s } , a ) = Q _ { l } ( s , \omega , a ) = Q ^ { \star } ( s , g _ { s } ^ { \prime } , a )$ , so the two targets induce the same action-value function at s and hence the same set of maximisers. Taking $\pi _ { l } ( \cdot \mid s , \omega )$ to be any distribution supported on argma $\mathfrak { c } _ { a } Q _ { l } ( s , \omega , a )$ gives the policy required by Definition 1.

## C CONSERVATIVE EXTRAPOLATION

To provide intuition on how joint state and action abstractions can enable conservative extrapolation, consider an agent learning a locomotion task, where its state can be decomposed into the agent’s pose and coordinate $\boldsymbol { s } = \left( p , c \right)$ . Imagine three state-goal pairs:

$$
\big ( s _ { 1 } , g , a _ { 1 } \big ) = \big ( \underbrace { ( p , c ) } _ { s _ { 1 } } , ~ g , a _ { 1 } \big ) , \quad \big ( s _ { 2 } , g , a _ { 2 } \big ) = \big ( \underbrace { ( p ^ { \prime } , c ) } _ { s _ { 2 } } , ~ g , a _ { 2 } \big ) , \quad \big ( s _ { 3 } , g ^ { \prime } , a _ { 3 } \big ) = \big ( \underbrace { ( p , c ^ { \prime } ) } _ { s _ { 3 } } , ~ g ^ { \prime } , a _ { 3 } \big ) .
$$

The first state-goal pair is the one that the agent must act at, and its dataset action is poor (it might just dither on the spot), while the dataset actions at the second and third pair are good. Imagine that in all situations, the agent must move two metres down to reach its respective goal because the goals lie in the same direction from their respective coordinates.

A non-hierarchical policy extracted with AWR or RS never learns a useful action at $( s _ { 1 } , g )$ because the only action the dataset offers there is the poor one. We now show how a hierarchical policy with abstract relative options (such as stay “on the spot” or reach “a point two metres down”) can output a good action for the first state-goal pair using conservative extrapolation (where each level only selects among actions its own data supports).

The second state-goal pair shares the coordinate and goal of the first, so the high-level abstraction (which ignores low-level pose information) groups them. This means that the high-level decision is rescued by the second state-goal pair: conservative extraction has the good option to reach “a point two metres down” available from the second state-goal pair, rather than only to stay “on the spot” from the first:

$$
\begin{array} { r } { \phi _ { h } ( s _ { 1 } , g ) = \phi _ { h } ( s _ { 2 } , g ) = ( c , g ) } \\ { \pi _ { h } ( \omega \mid s _ { 1 } , g ) = \pi _ { h } ( \omega \mid s _ { 2 } , g ) = \delta _ { \omega = \omega _ { \mathrm { d o w n } } } } \end{array}
$$

Note that the second pair supplies the right option, but its action is executed from a different pose, so it cannot be copied directly.

Similarly, the third state-goal pair then shares the pose and option of the first, so the low-level abstraction (which ignores high-level coordinate information) groups them. The low-level decision is then rescued by the third state-goal pair: conservative extraction has the good action $a _ { 3 }$ available from the third pair, rather than only $a _ { 1 }$ from the first:

$$
\begin{array} { r } { \phi _ { l } ( s _ { 1 } , \omega _ { \mathrm { d o w n } } ) = \phi _ { l } ( s _ { 3 } , \omega _ { \mathrm { d o w n } } ) = ( p , \omega _ { \mathrm { d o w n } } ) } \\ { \pi _ { l } ( a \mid s _ { 1 } , \omega _ { \mathrm { d o w n } } ) = \pi _ { l } ( a \mid s _ { 3 } , \omega _ { \mathrm { d o w n } } ) = \delta _ { a = a _ { 3 } } } \end{array}
$$

Under the hierarchical policy, the agent therefore takes $a _ { 3 }$ at $( s _ { 1 } , g )$ , an action never observed for that state-goal pair.

Note that this example hinges on the learning of abstract relative options so that reach “a point two metres down” is a single option rather than, for example, two coordinate-specific ones.

## D OPTION SPACE COMPRESSION BOUNDS OFFLINE ERROR

## Proposition 2. Offline error under joint state and action abstraction.

For a finite-state, finite-action MDP with at most two possible next-state transitions, with probability of at least $1 - \delta ,$ the suboptimality in expected return, $\begin{array} { r } { \epsilon : = \operatorname* { m a x } _ { s \in \mathcal { S } , g \in \mathcal { G } } \left( V ^ { \star } ( s , g ) - \right. } \end{array}$ $V ^ { \pi } ( s , g ) )$ of a goal-conditioned hierarchical policy learned from a dataset of $N$ transitions satisfies

$$
\epsilon \propto \sqrt { \frac { | \mathcal { C } _ { h } | | \Omega | \cdot \rho _ { h } ^ { \mathrm { r e p } } } { ( 1 - \gamma ^ { k } ) ^ { 3 } N } } + \sqrt { \frac { | \mathcal { C } _ { l } | | A | k ^ { 3 } \cdot \rho _ { l } ^ { \mathrm { r e p } } } { N } } ,
$$

where $\phi _ { h } : \mathcal { S } \times \mathcal { G } \to \mathcal { C } _ { h }$ aggregates state-goal pairs requiring the same option such that $\mathcal { C } _ { h }$ denotes the state-abstracted high-level space, and $\phi _ { l } : S \times \Omega  \mathcal { C } _ { l }$ aggregates state-option pairs requiring the same action such that $\mathcal { C } _ { l }$ denotes the state-abstracted low-level space. Constants and logarithmic factors are suppressed and $\rho _ { h } ^ { \mathrm { r e p } }$ and $\rho _ { l } ^ { \mathrm { r e p } }$ are the concentrability coefficients at each level:

$$
\rho _ { h } ^ { \mathrm { r e p } } = \operatorname* { s u p } _ { c _ { h } \in \mathcal C _ { h } , \omega \in \Omega } \frac { d ^ { \pi _ { h } ^ { \star } } ( c _ { h } , \omega ) } { d ^ { \pi _ { h } ^ { \mathrm { B C } } } ( c _ { h } , \omega ) } , \qquad \rho _ { l } ^ { \mathrm { r e p } } = \operatorname* { s u p } _ { c _ { l } \in \mathcal C _ { l } , a \in \mathcal A } \frac { d ^ { \pi _ { l } ^ { \star } } ( c _ { l } , a ) } { d ^ { \pi _ { l } ^ { \mathrm { B C } } } ( c _ { l } , a ) } ,
$$

where $d ^ { \pi }$ is underlying discounted occupancy measure of a policy π and occupancy measures are aggregated over each abstraction:

$$
d ^ { \pi _ { h } } ( c _ { h } , \omega ) = \sum _ { \substack { ( s , g ) \in \phi _ { h } ^ { - 1 } ( c _ { h } ) } } d ^ { \pi _ { h } } ( s , g , \omega ) , \qquad d ^ { \pi _ { l } } ( c _ { l } , a ) = \sum _ { \substack { ( s , \omega ) \in \phi _ { l } ^ { - 1 } ( c _ { l } ) } } d ^ { \pi _ { l } } ( s , \omega , a ) .
$$

We refer to Appendices E and F for full definitions and derivations. The two terms respectively bound the high and low-level error terms; the concentrability coefficients $\rho ^ { \mathrm { r e p } }$ measure the distribution shift between the occupancy of the optimal policy and that of the dataset: as suprema of occupancy ratios, a single state-goal or state-option pair that the optimal policy visits but the dataset does not can cause the term to go to infinity, unless aggregation lets that pair borrow mass from equivalent ones. Compressing the option space by discarding unnecessary distinctions between options reduces the bound in three ways. It reduces the high-level action space |Ω|, tightens the bound $| \mathcal { C } _ { l } | \leq | S | | \Omega |$ on the low-level state-abstraction space, and aggregates occupancy mass across a larger number of equivalent state-goal and state-option pairs, thereby reducing both concentrability coefficient (“conservative extrapolation”). In a continuous space, rather than making |Ω| as small as possible, this equates to mapping functionally equivalent behaviour to similar points in the option space (Abel et al., 2016; Ferns et al., 2012; Zhang et al., 2020), which reduces the effective complexities of both error terms (Munos and Szepesvari, 2008; Chen and Jiang, 2019).´

## E STATE AND ACTION ABSTRACTION SAMPLE COMPLEXITY IN ONLINEGOAL-CONDITIONED RL

In this section, we show how joint state and action abstraction can reduce the sample complexity for finding an optimal goal-conditioned policy such that the error in expected return is smaller than a fixed constant ϵ from

$$
N \propto \frac { | S | | \mathcal { G } | | A | } { ( 1 - \gamma ) ^ { 3 } \epsilon ^ { 2 } } ,
$$

for a policy with no state or action abstraction, to

$$
N \propto \frac { | \mathcal C _ { h } | | \Omega | } { ( 1 - \gamma ^ { k } ) ^ { 3 } \epsilon ^ { 2 } } + \frac { | \mathcal C _ { l } | | \mathcal A | k ^ { 3 } } { \epsilon ^ { 2 } } ,
$$

for a policy with both state and action abstraction. We build on the works of Kakade (2003); Munos and Szepesvari (2008); Lattimore and Hutter (2012); Robert et al. (2023); Li et al. (2006).´

Sample Complexity in Online GCRL In online GCRL, the aim is to find the optimal policy with the smallest number of samples or online interactions, N, such that the error in optimal return is smaller than a fixed constant ϵ. Consider the case of a discrete, discounted horizon MDP with a finite state space $s ,$ action space $\mathcal { A }$ and discount factor $\gamma \in [ 0 , 1 )$ . The Probably-Approximately Correct (PAC) Learning (Kakade, 2003) upper-bound sample complexity to find an ϵ optimal policy reaching a unique goal-state optimally (assuming at most two possible next-states for each state/action pair) with constant probability of $1 - \delta$ is given by Lattimore and Hutter (2012):

$$
N \propto \frac { | { \cal S } | | { \cal A } | } { ( 1 - \gamma ) ^ { 3 } \epsilon ^ { 2 } } .
$$

Hence, the minimax sample complexity required to find an ϵ optimal policy reaching any given state (such that we have |G| unique goal-states) is given by:

$$
N \propto \frac { | S | | \mathcal { G } | | A | } { ( 1 - \gamma ) ^ { 3 } \epsilon ^ { 2 } } .\tag{1}
$$

Consider the case of a discrete finite horizon (of length H) MDP with a finite state space $s ,$ action space A and discount factor $\gamma \in [ 0 , 1 )$ . The minimax sample complexity to find an ϵ optimal policy reaching a unique goal-state optimally is given by:

$$
N \propto \frac { | S | | \mathcal { G } | | \mathcal { A } | H ^ { 3 } } { \epsilon ^ { 2 } } .
$$

Action Abstraction Sample Complexity in Online GCRL Using a hierarchical policy, we can break the distant goal $g$ from our current state s into options defined over an option space Ω. Hence, the high-level policy becomes

$$
N \propto \frac { | \boldsymbol { S } | | \boldsymbol { \mathcal { G } } | | \Omega | } { ( 1 - \gamma ^ { k } ) ^ { 3 } \epsilon ^ { 2 } } ,
$$

where we have substituted the option-space to Equation 1, and use the environment’s discount factor raised to a factor of $k ,$ assuming that the high-level policy acts every k steps.

The low-level policy has a sample complexity of

$$
N \propto \frac { | S | | \Omega | | A | k ^ { 3 } } { \epsilon ^ { 2 } } ,
$$

since |Ω| options can be executed. Hence, the overall sample complexity of the hierarchical policy is (Robert et al., 2023):

$$
N \propto \frac { | \boldsymbol { S } | | \boldsymbol { \mathcal { G } } | | \Omega | } { ( 1 - \gamma ^ { k } ) ^ { 3 } \epsilon ^ { 2 } } + \frac { | \boldsymbol { S } | | \Omega | | \boldsymbol { A } | k ^ { 3 } } { \epsilon ^ { 2 } } .\tag{2}
$$

Note that in the case of no temporal abstraction, when $k = 1$ , we approximately recover the original sample complexity of a non-hierarchical policy. Since the option just becomes a single primitive action, Equation 2 becomes:

$$
N \propto \frac { | S | | \mathcal { G } | | A | } { ( 1 - \gamma ) ^ { 3 } \epsilon ^ { 2 } } + \frac { | S | | A | ^ { 2 } } { \epsilon ^ { 2 } } \approx \frac { | S | | \mathcal { G } | | A | } { ( 1 - \gamma ) ^ { 3 } \epsilon ^ { 2 } } ,
$$

where the approximation follows due to the dominating $1 / ( 1 - \gamma ) ^ { 3 }$ factor in the first term, assuming long-horizon problems such that $\gamma  1$ and that the goal-space is larger or equal to the size of the action space $| { \bar { \mathcal { G } } } | \geq | { \mathcal { A } } |$ . Even though this might not be the case, usually the goal-space is unknown, so we train the policy to reach any state within the state-space i.e. such that $\bar { \boldsymbol { \mathscr { G } } } = \bar { \boldsymbol { S } }$ . Assuming that $| { \mathcal { S } } | \gg | { \mathcal { A } } |$ is a standard assumption.

In most real-world settings we expect $| \Omega | \ll | A | ^ { k }$ . The option space does not have to perfectly recreate the sequence of low-level actions, only contain the sufficient information to approximate the high-level value and choose the low-level action.

State and Action Abstraction Sample Complexity in Online GCRL Finally, since the high-level decision process can discard irrelevant low-level information, and the low-level decision process can discard irrelevant high-level information, state abstraction reduces the effective complexity of the state space at each level of the hierarchy such that Equation 2 becomes:

$$
N \propto \frac { | \mathcal C _ { h } | | \Omega | } { ( 1 - \gamma ^ { k } ) ^ { 3 } \epsilon ^ { 2 } } + \frac { | \mathcal C _ { l } | | \mathcal A | k ^ { 3 } } { \epsilon ^ { 2 } } ,\tag{3}
$$

where $\left| \mathcal { C } _ { h } \right|$ is the effective state-space required for the high-level decision process and $| \mathcal { C } _ { l } |$ is the effective state-space for the low-level decision process. By definition, $| { \mathcal { C } } _ { h } | \leq | { \dot { S } } | | { \mathcal { G } } |$ and $| \mathcal { C } _ { l } | \leq | S | | \Omega |$

## F STATE AND ACTION ABSTRACTION ERROR IN OFFLINEGOAL-CONDITIONED RL

In offline GCRL, the aim is to bound the error ϵ given an offline dataset D of fixed size $N .$

Unlike in the online setting, where it is assumed that the agent can sample any state-action pair to learn the environment’s dynamics, in offline RL, a concentrability coefficient $\rho$ is incorporated to account for the distribution shift in data collected by the policy $\pi ^ { \mathbf { \tilde { B } C } }$ , and the data that would have been collected induced under the optimal policy $\pi ^ { \star }$ . Intuitively, if the dataset does not include the states required to learn the optimal policy, the algorithm may never learn that optimal policy. The concentrability coefficient $\rho$ is defined as:

$$
\rho = \operatorname* { s u p } _ { s \in { \mathcal { S } } , g \in { \mathcal { G } } , a \in { \mathcal { A } } } \frac { d ^ { \pi ^ { \star } } ( s , g , a ) } { d ^ { \pi ^ { \mathrm { B C } } } ( s , g , a ) } ,
$$

where $d ^ { \pi } ( s , g , a )$ is the discounted occupancy measure (i.e. stationary distribution) of the policy π:

$$
d ^ { \pi } ( \boldsymbol { s } , \boldsymbol { g } , \boldsymbol { a } ) = ( 1 - \gamma ) \mathbb { E } _ { \tau \sim \pi , \boldsymbol { s } _ { 0 } , \boldsymbol { g } _ { 0 } \sim \mathrm { U n i f } ( \boldsymbol { \mathcal { G } } ) } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { 1 } ( \boldsymbol { s } _ { t } = \boldsymbol { s } , \boldsymbol { a } _ { t } = \boldsymbol { a } , \boldsymbol { g } _ { 0 } = \boldsymbol { g } ) \right] ,
$$

and where the trajectory is induced by the MDP and following the policy $\pi ( \boldsymbol { a } \mid s , g )$ . If the dataset is highly exploratory and covers the optimal paths well, $\rho$ will be small. If the dataset is narrow, or misses critical regions of the state-action space, $\rho$ will be large.

Rearranging Equation 1 and incorporating the concentrability coefficient, the offline bound for a goal-conditioned non-hierarchical policy is given by:

$$
\epsilon \propto \sqrt { \frac { | \boldsymbol { S } | | \boldsymbol { \mathcal { G } } | | \boldsymbol { A } | \cdot \rho } { ( 1 - \gamma ) ^ { 3 } N } } .
$$

Action Abstraction Error in Offline GCRL Since the size of the offline dataset is fixed, one way to reduce the error ϵ is to use a hierarchical policy with distinct state-goal representations.

Using a hierarchical policy, the error term becomes:

$$
\epsilon \propto \sqrt { \frac { | S | | \mathcal { G } | | \Omega | \cdot \rho _ { h } } { ( 1 - \gamma ^ { k } ) ^ { 3 } N } } + \sqrt { \frac { | S | | \Omega | | A | k ^ { 3 } \cdot \rho _ { l } } { N } } ,
$$

where the concentrability coefficients are defined as:

$$
\rho _ { h } = \operatorname* { s u p } _ { s \in S , g \in \mathcal { G } , \omega \in \Omega } \frac { d ^ { \pi _ { h } ^ { \star } } ( s , g , \omega ) } { d ^ { \pi _ { h } ^ { \mathrm { B C } } } ( s , g , \omega ) } \quad \mathrm { a n d } \quad \rho _ { l } = \operatorname* { s u p } _ { s \in S , \omega \in \Omega , a \in \mathcal { A } } \frac { d ^ { \pi _ { l } ^ { \star } } ( s , \omega , a ) } { d ^ { \pi _ { l } ^ { \mathrm { B C } } } ( s , \omega , a ) } .
$$

State and Action Abstraction Error in Offline GCRL Then, ignoring irrelevant low-level information for the high-level decision process and high-level information for the low-level decision process, the error term is bounded by:

$$
\epsilon \propto \sqrt { \frac { | \mathcal { C } _ { h } | | \Omega | \cdot \rho _ { h } ^ { \mathrm { r e p } } } { ( 1 - \gamma ^ { k } ) ^ { 3 } N } } + \sqrt { \frac { | \mathcal { C } _ { l } | | A | k ^ { 3 } \cdot \rho _ { l } ^ { \mathrm { r e p } } } { N } } .
$$

Again, $| { \mathcal { C } } _ { h } | \leq | S | | { \mathcal { G } } |$ and $| \mathcal { C } _ { l } | \leq | S | | \Omega |$

Because the concentrability coefficients are now defined over the constrained spaces $\mathcal { C } _ { h }$ and $\mathcal { C } _ { l } .$ , i.e.

$$
\rho _ { h } ^ { \mathrm { r e p } } = \operatorname* { s u p } _ { c _ { h } \in \mathcal C _ { h } , \omega \in \Omega } \frac { d ^ { \pi _ { h } ^ { \star } } ( c _ { h } , \omega ) } { d ^ { \pi _ { h } ^ { \mathrm { B C } } } ( c _ { h } , \omega ) } , \qquad \rho _ { l } ^ { \mathrm { r e p } } = \operatorname* { s u p } _ { c _ { l } \in \mathcal C _ { l } , a \in \mathcal A } \frac { d ^ { \pi _ { l } ^ { \star } } ( c _ { l } , a ) } { d ^ { \pi _ { l } ^ { \mathrm { B C } } } ( c _ { l } , a ) } ,
$$

the probability mass of the offline dataset is aggregated across state-goal or state-option pairs that are equivalent under this information loss (i.e. for which $c _ { h } = \phi _ { h } ( s , g )$ and $c _ { l } = \phi _ { l } ( s , \omega ) )$ ):

$$
d ^ { \pi _ { h } } ( c _ { h } , \omega ) = \sum _ { \substack { ( s , g ) \in \phi _ { h } ^ { - 1 } ( c _ { h } ) } } d ^ { \pi _ { h } } ( s , g , \omega ) , \qquad d ^ { \pi _ { l } } ( c _ { l } , a ) = \sum _ { \substack { ( s , \omega ) \in \phi _ { l } ^ { - 1 } ( c _ { l } ) } } d ^ { \pi _ { l } } ( s , \omega , a ) .
$$

This reduces the likelihood of a support mismatch, where the optimal policy requires a state transition on which the dataset places zero mass. Hence,

$$
\rho _ { h } ^ { \mathrm { r e p } } \leq \rho _ { h } \quad \mathrm { a n d } \quad \rho _ { l } ^ { \mathrm { r e p } } \leq \rho _ { l } .
$$

## G Q-SHAPED OPTIONS

## G.1 ACTION ABSTRACTION

Joint Action Abstraction and Value Learning The action abstractions are learned jointly with the high-level and low-level Q functions, acting as a goal-space for the low-level Q function and an action space for the high-level Q function. The high-level value and critic losses are given by the following IQL loss:

$$
\mathcal { L } _ { V _ { h } } = \mathbb { E } _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } } \Big [ \ell _ { \nu } ^ { 2 } \Big ( V _ { h } \big ( s _ { t } , g \big ) - \tilde { Q } _ { h } \big ( s _ { t } , g , \tilde { \phi } _ { \omega } \big ( s _ { t } , s _ { t + k } \big ) \big ) \Big ) \Big ]\tag{4}
$$

$$
\mathcal { L } _ { Q _ { h } , \phi _ { \omega } } = \mathbb { E } _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } } \bigg [ \Big ( Q _ { h } \big ( s _ { t } , g , \phi _ { \omega } ( s _ { t } , s _ { t + k } ) \big ) - R _ { h } \big ( s _ { t : t + k } , g \big ) - \gamma V _ { h } \big ( s _ { t + k } , g \big ) \Big ) ^ { 2 } \bigg ] ,\tag{5}
$$

where the normalised k-step option return is

$$
R _ { h } ( s _ { t : t + k } , g ) = \frac { 1 - \gamma } { 1 - \gamma ^ { k } } \sum _ { t ^ { \prime } = 0 } ^ { k - 1 } \gamma ^ { t ^ { \prime } } r _ { h } ( s _ { t + t ^ { \prime } } , g ) .
$$

The low-level value and critic losses are given by the following IQL loss:

$$
\mathcal { L } _ { V _ { l } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , g _ { s } ) \sim \mathcal { D } } \Big [ \ell _ { \nu } ^ { 2 } \Big ( V _ { l } \big ( s _ { t } , \phi _ { \omega } \big ( s _ { t } , g _ { s } \big ) \big ) - \tilde { Q } _ { l } \big ( s _ { t } , \tilde { \phi } _ { \omega } \big ( s _ { t } , g _ { s } \big ) , a _ { t } \big ) \Big ) \Big ]\tag{6}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { Q } _ { l } , \phi _ { \omega } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , g _ { s } ) \sim \mathcal { D } } \Bigg [ \Big ( Q _ { l } \big ( s _ { t } , \phi _ { \omega } \big ( s _ { t } , g _ { s } \big ) , a _ { t } \big ) - r _ { l } \big ( s _ { t } , g _ { s } \big ) - \gamma V _ { l } \big ( s _ { t + 1 } , \omega _ { t + 1 } \big ) \Big ) ^ { 2 } \Bigg ] . } \end{array}\tag{7}
$$

Continuous Option Space We use a continuous option space encoded using a feed-forward MLP followed by length-normalisation so that the options lie on a hypersphere of radius $\sqrt { d _ { \omega } }$ in $\mathbb { R } ^ { d _ { \omega } }$ preventing any proposed option from being significantly out-of-distribution for the low-level policy:

$$
\phi _ { \omega } ( s , g ) = \sqrt { d _ { \omega } } \frac { f _ { \omega } ( s , g ) } { \| f _ { \omega } ( s , g ) \| _ { 2 } } ,
$$

where $f _ { \omega } : \mathcal { S } \times \mathcal { G }  \mathbb { R } ^ { d _ { \omega } }$ denotes the feed-forward MLP encoder. At deployment, the options are continually resampled at every timestep.

Discounting The same discount factor γ is used at each level of the hierarchy, so that each level has the same effective horizon. For a two-level hierarchy, this means that the low-level decision process looks $1 / ( 1 - \gamma )$ actions ahead, and the higher-level decision process looks $1 / ( 1 - \gamma )$ options ahead. Since one high-level option corresponds to k low-level actions, we would ideally set

$$
k = \frac { 1 } { 1 - \gamma } ,
$$

so that the low-level decision process reasons over exactly the span of a single option. The effective horizons then compose multiplicatively: in other words, the higher level decision process covers $k ^ { 2 }$ actions, while only reasoning over $1 / ( \bar { 1 } - \gamma )$ decisions.

In practice (and like prior work on hierarchical RL and QC) we find that k is bounded by the data. Since options are sampled as k-steps within a single trajectory, k must be small relative to the trajectory length H for the dataset to contain enough distinct options. For example, with our target discount factor of 0.99, we would want to use 100-step options, but antmaze-giant’s trajectories are only 200 steps.

Reward Calibration Accumulated option rewards are normalised so that a full-length option costs the same as a single action in the low-level decision process. This means that the Q functions work on similar scales, preventing one Q function from dominating the other in magnitude (bypassing the need to introduce any hyperparameters), while options terminating early incur a proportionally smaller cost. The accumulated option reward is normalised by

$$
{ \frac { 1 - \gamma } { 1 - \gamma ^ { k } } } ,
$$

so that a full-length option (i.e. $\beta ( s _ { t + t ^ { \prime } } , g ) = 0$ for $t ^ { \prime } \in \{ 0 , \cdots , k - 1 \} )$ costs exactly −1, identical to the cost of a single action in the low-level decision process.

## G.2 STATE ABSTRACTIONS

Rather than introducing an explicit representation-learning objective, we induce meaningful state abstractions through level-specific datasets. At the high-level, we sample a broad mixture of local, trajectory, and random goals, encouraging it to generalise across a broad combinatorial coverage of state-goal pairs; at the low-level, we predominantly sample nearby trajectory target states, encouraging precise control over the option horizon. We replace the stricter reward-labelling criterion used by prior work (Park et al., 2025), which requires an exact match with the goal on every state dimension, with the environment success criterion at the high-level, avoiding distinctions between states that are equivalent for the task. Note that, to isolate the effect of action abstraction and because we find that it improves all hierarchical baselines (Figure 13 in Appendix H) we apply this change consistently in our experiments. At the low-level, where we want to focus on fine-grained control, we retain the stricter criterion. We ablate using the environment criterion at both levels: in Appendix H, Figure 11 shows performance ablating high-level goal sampling, while Figures 12, 15 and 16 show performance and option embeddings when ablating reward labelling. Table 8 quantifies state abstraction.

Goal Sampling During training, goals are sampled according to the following, where $\tau =$ $( s _ { 0 } , \ldots , s _ { H } )$ denotes the H-length trajectory containing $s _ { t }$ and $\mathrm { \bar { U } n i f } _ { \mathcal { D } }$ denotes a uniform distribution over all states in the dataset, and the weights w of each of the three sampling components sum to 1. As usual, k denotes the fixed option length.

## High-level Decision Process

For the high-level decision process, goals are sampled according to

$$
p _ { h } ( g \mid s _ { t } ) = w _ { h } ^ { \mathrm { o p t } } p _ { h } ^ { \mathrm { o p t } } ( g \mid s _ { t } ) + w _ { h } ^ { \mathrm { t r a j } } p _ { h } ^ { \mathrm { t r a j } } ( g \mid s _ { t } ) + w _ { h } ^ { \mathrm { r a n d } } \operatorname { U n i f } _ { \mathcal { D } } ( g ) ,\tag{8}
$$

where the first component samples a goal uniformly from within the option window,

$$
p ^ { \mathrm { o p t } } ( \boldsymbol { g } \mid s _ { t } ) = \mathcal { U } \big ( \{ \operatorname* { m i n } ( t + 1 , H ) , \ . . . , \ \operatorname* { m i n } ( t + k - 1 , H ) \} \big ) ,
$$

the second component samples the goal uniformly from the remainder of the trajectory beyond the option window,

$$
p _ { h } ^ { \mathrm { t r a j } } ( g \mid s _ { t } ) = \mathcal { U } \big ( \{ \operatorname* { m i n } ( t + k , H ) , \ \dots , \ H \} \big ) ,
$$

and the third samples goals uniformly from the entire dataset.

The first component ensures that a fraction of every batch returns success, the second component provides goals that are distant but reachable by dataset stitching, and the third component provides examples of the out-of-distribution combinations of state-goal pairs that the high-level policy will face at deployment.

Low-level Decision Process For the low-level decision process, goals are sampled according to

$$
p _ { l } ( g _ { s } \mid s _ { t } ) = w _ { l } ^ { \mathrm { c u r } } \delta _ { s _ { t } } ( g _ { s } ) + w _ { l } ^ { \mathrm { t r a j } } p _ { l } ^ { \mathrm { t r a j } } ( g _ { s } \mid s _ { t } ) + w _ { l } ^ { \mathrm { r a n d } } \operatorname { U n i f } _ { \mathcal { D } } ( g _ { s } ) ,\tag{9}
$$

where the first component places all mass on the current state, the second geometrically samples a state a number of steps ahead,

$$
\begin{array} { r } { p _ { l } ^ { \mathrm { t r a j } } ( g _ { s } \mid s _ { t } ) : \quad g _ { s } = s _ { \mathrm { m i n } ( t + \kappa , H ) } , \qquad \kappa \sim \mathrm { G e o m } \left( \frac { 1 } { k } \right) , } \end{array}
$$

so that the expected look-ahead is k steps and the third samples uniformly from the entire dataset.

The first component again ensures that a fraction of every batch returns a success, the second component acts in accordance with the option length of the high-level decision process, and the third component calibrates the value functions across the state space, making the learned values neater to visualise, but is unnecessary for algorithm functioning, since, even at deployment, the low-level value functions are never queried at such states.

Reward Labelling We use the environment success criterion (goal-achievement indicator) for the high-level

$$
\beta ^ { h } ( s , g ) = \beta ( s , g ) ,
$$

specified per-environment in Appendix H. For the low-level, we follow prior work and use a stricter success criterion that requires an exact match on every state dimension:

$$
\begin{array} { r } { \beta ^ { l } ( s , g _ { s } ) = \mathbb { 1 } \{ s = g _ { s } \} . } \end{array}
$$

## G.3 POLICY EXTRACTION

The high-level policy is defined using Rejection Sampling of an advantage-weighted behaviour cloning policy, and the low-level simply as a behaviour cloning policy:

$$
\begin{array} { r l } & { \pi _ { h } ( \omega \mid s _ { t } , g ) : = \operatorname * { a r g m a x } _ { \omega \in \{ \omega ^ { i } \} , \omega ^ { i } \sim \pi _ { h } ^ { \mathrm { A W R } } ( \cdot \mid s _ { t } , g ) } Q _ { h } ( s _ { t } , g , \omega ^ { i } ) , } \\ & { \pi _ { l } ( a _ { t } \mid s _ { t } , \omega ) : = \pi _ { l } ^ { \mathrm { B C } } ( a _ { t } \mid s _ { t } , \omega ) } \end{array}
$$

High-Level Policy Extraction The high-level policy is fitted using the following advantageweighted flow-matching loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \pi _ { h } ^ { \mathrm { A W R } } } = \mathbb { E } _ { u \sim \mathrm { U n i f } \left( \left[ 0 , 1 \right] \right) , \omega = \phi _ { \omega } \left( s _ { t } , s _ { t + k } , g \right) \sim \mathcal { N } \left( 0 , I _ { d \omega } \right) } } \\ & { \qquad \quad \times \mathrm { ~ U n i f } \left( \left[ 0 , 1 \right] \right) , \omega = \phi _ { \omega } \left( s _ { t } , s _ { t + k } \right) , \omega ^ { u } = \left( 1 - u \right) z + u \omega } \\ & { \qquad \quad \left[ \exp \left( \frac { Q _ { h } \left( s _ { t } , g , \omega \right) - V _ { h } \left( s _ { t } , g \right) } { \alpha } \right) \cdot \left. v _ { \pi _ { h } ^ { \mathrm { A W R } } } \left( s _ { t } , g , u , \omega ^ { u } \right) - \left( \omega - z \right) \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{10}
$$

Low-Level Policy Extraction The low-level policy is fitted using the following behaviour-cloning flow-matching loss:

$$
\begin{array} { r } { \mathcal { L } _ { \pi _ { l } ^ { \mathrm { B C } } } = \mathbb { E } _ { \left( \mathfrak { s } _ { t } , a _ { t } , \mathfrak { s } _ { t + k } \right) \sim \mathcal { D } , z \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \left\| v _ { \pi _ { l } } ^ { \mathrm { B C } } \big ( \mathfrak { s } _ { t } , \phi _ { \omega } \big ( \mathfrak { s } _ { t } , \mathfrak { s } _ { t + k } \big ) , u , a ^ { u } \big ) - \big ( { a } _ { t } - \mathfrak { z } \big ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{11}
$$

## H EXPERIMENTS

## H.1 ENVIRONMENTS

<table><tr><td>Environment</td><td>State Dim.</td><td>Action Dim.</td><td>Data Trajectory Length</td><td>Dataset Size</td><td>Max. Episode Length</td></tr><tr><td>antmaze-giant-stitch-v0</td><td>29</td><td>8</td><td>200</td><td>1M</td><td>1000</td></tr><tr><td>antmaze-teleport-stitch-v0</td><td>29</td><td>8</td><td>200</td><td>1M</td><td>1000</td></tr><tr><td>humanoidmaze-giant-stitch-v0</td><td>69</td><td>21</td><td>400</td><td>4M</td><td>4000</td></tr><tr><td>cube-double-play-v0</td><td>37</td><td>5</td><td>1000</td><td>1M</td><td>500</td></tr><tr><td>cube-triple-play-v0</td><td>46</td><td>5</td><td>1000</td><td>3M</td><td>1000</td></tr><tr><td>cube-quadruple-play-v0</td><td>55</td><td>5</td><td>1000</td><td>5M</td><td>1000</td></tr><tr><td>puzzle-3x3-play-v0</td><td>55</td><td>5</td><td>1000</td><td>1M</td><td>500</td></tr><tr><td>puzzle-4x4-play-v0</td><td>83</td><td>5</td><td>1000</td><td>1M</td><td>500</td></tr><tr><td>puzzle-4x5-play-v0</td><td>99</td><td>5</td><td>1000</td><td>3M</td><td>1000</td></tr><tr><td>puzzle-4x6-play-v0</td><td>115</td><td>5</td><td>1000</td><td>5M</td><td>1000</td></tr><tr><td>scene-play-v0</td><td>40</td><td>5</td><td>1000</td><td>1M</td><td>750</td></tr></table>

Table 4: OGBench environment and dataset characteristics.

## Maze Environments

For antmaze-giant, antmaze-teleport and humanoidmaze-giant, high-level success is defined as the agent lying within ϵ of the goal’s xy coordinates, where $x ( s ) \in \mathbb { R } ^ { 2 }$ denotes a state s’s xy coordinates:

$$
\beta ( s , g ) = \mathbb { 1 } \left\{ \| x ( s ) - x ( g ) \| _ { 2 } \leq \epsilon \right\} .
$$

## Cube Environments

For cube-double, cube-triple and cube-quadruple, high-level success is defined as every cube being within ϵ of its goal xyz position, where $c _ { i } ( s ) \in \mathbb { R } ^ { 3 }$ denotes cube i’s xyz coordinates in state $s ,$ and n is the number of cubes in the environment:

$$
\beta ( s , g ) = \prod _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ \| c _ { i } ( s ) - c _ { i } ( g ) \| _ { 2 } \leq \epsilon \right\} .
$$

## Puzzle Environments

For puzzle-3x3, puzzle-4x4, puzzle-4x5 and puzzle-4x6, high-level success is defined as every button matching its goal configuration, where $b _ { i } ( s ) \in \mathbb { R } ^ { 3 }$ denotes button i’s configuration in state $s ,$ and n is the number of buttons in the environment:

$$
\beta ( s , g ) = \prod _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ b _ { i } ( s ) = b _ { i } ( g ) \right\} .
$$

## Scene Environment

For scene-play, high-level success is defined as the $n _ { c }$ cubes, drawer and window being within $\epsilon _ { c } , \epsilon _ { d }$ and $\epsilon _ { w }$ of their goal positions and the $n _ { b }$ buttons matching their goal configurations:

$$
\begin{array} { r l r } & { } & { \displaystyle \beta ( s , g ) = \prod _ { i = 1 } ^ { n _ { c } } \mathbb { 1 } \left\{ \| c _ { i } ( s ) - c _ { i } ( g ) \| _ { 2 } \leq \epsilon _ { c } \right\} \cdot \prod _ { i = 1 } ^ { n _ { b } } \mathbb { 1 } \left\{ b _ { i } ( s ) = b _ { i } ( g ) \right\} } \\ & { } & { \quad \quad \cdot \mathbb { 1 } \left\{ | d ( s ) - d ( g ) | \leq \epsilon _ { \mathrm { d } } \right\} \cdot \mathbb { 1 } \left\{ | w ( s ) - w ( g ) | \leq \epsilon _ { \mathrm { w } } \right\} . } \end{array}
$$

## H.2 BASELINES

## H.2.1 ALGORITHMS

Absolute Subgoal Options: DSHARSA For DSHARSA (Park et al., 2026), both the high and low-level value and critic functions are learned using IQL:

$$
\begin{array} { r l } & { \mathcal { L } _ { V _ { h } } = \mathbb { E } _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } } \left[ \ell _ { \nu } ^ { 2 } \left( V _ { h } ( s _ { t } , g ) - \tilde { Q } _ { h } ( s _ { t } , g , s _ { t + k } ) \right) \right] , } \\ & { \mathcal { L } _ { Q _ { h } } = \mathbb { E } _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } } \left[ \left( Q _ { h } ( s _ { t } , g , s _ { t + k } ) - \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { k - 1 } \gamma ^ { t ^ { \prime } } r _ { h } ( s _ { t + t ^ { \prime } } , g ) - \gamma ^ { k } V _ { h } ( s _ { t + k } , g ) \right) ^ { 2 } \right] , } \end{array}
$$

and

$$
\begin{array} { r l } & { \mathcal { L } _ { V _ { l } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , g _ { s } ) \sim \mathcal { D } } \left[ \ell _ { \nu } ^ { 2 } \left( V _ { l } ( s _ { t } , g _ { s } ) - \tilde { Q } _ { l } ( s _ { t } , g _ { s } , a _ { t } ) \right) \right] , } \\ & { \mathcal { L } _ { Q _ { l } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , g _ { s } ) \sim \mathcal { D } } \left[ ( Q _ { l } ( s _ { t } , g _ { s } , a _ { t } ) - r _ { l } ( s _ { t } , g _ { s } ) - \gamma V _ { l } ( s _ { t + 1 } , g _ { s } ) ) ^ { 2 } \right] , } \end{array}
$$

The high-level and low-level policies are fitted using behaviour-cloning flow-matching losses:

$$
\mathcal { L } _ { \pi _ { h } ^ { \mathrm { B C } } } = \mathbb { E } \frac { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal { D } , z \sim \mathcal { N } ( 0 , I _ { d _ { s } } ) } { u \sim \mathrm { U n i f } ( [ 0 , 1 ] ) , g _ { s } = s _ { t + k } , g _ { s } ^ { u } = ( 1 - u ) z + u g _ { s } } \Bigg [ \bigg \| v _ { \pi _ { h } ^ { \mathrm { B C } } } \big ( s _ { t } , g , u , g _ { s } ^ { u } \big ) - ( g _ { s } - z ) \bigg \| _ { 2 } ^ { 2 } \Bigg ] ,
$$

and

$$
\begin{array} { r } { \mathcal { L } _ { \pi _ { l } ^ { \mathrm { B C } } } = \mathbb { E } _ { \boldsymbol { u } \sim \operatorname { U n i f } \left( \left[ 0 , 1 \right] \right) , \boldsymbol { a } ^ { u } = \left( 1 - u \right) z + u \boldsymbol { a } _ { t } } ^ { \boldsymbol { \mathrm { } } } \left[ \left\| v _ { \pi _ { l } } \left( s _ { t } , s _ { t + k } , u , a ^ { u } \right) - \left( a _ { t } - z \right) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

At deployment, both the high and low-level policies are defined with Rejection Sampling:

$$
\begin{array} { r } { \pi _ { h } ( g _ { s } \mid s _ { t } , g ) : = \mathrm { a r g m a x } _ { g _ { s } \in \{ g _ { s } ^ { i } \} , g _ { s } ^ { i } \sim \pi _ { h } ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , g ) } Q _ { h } ( s _ { t } , g , g _ { s } ^ { i } ) , } \\ { \pi _ { l } ( a _ { t } \mid s _ { t } , g _ { s } ) : = \mathrm { a r g m a x } _ { a _ { t } \in \{ a _ { t } ^ { i } \} , a _ { t } ^ { i } \sim \pi _ { l } ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , g _ { s } ) } Q _ { l } ( s _ { t } , g _ { s } , a _ { t } ^ { i } ) . } \end{array}
$$

Absolute Subgoal Options: SHARSA SHARSA (Park et al., 2026) is almost identical to DSHARSA, but only uses Rejection Sampling for the high-level policy. The low-level policy is defined using pure behaviour cloning:

$$
\begin{array} { r l } & { { \pi } _ { h } ( g _ { s } \mid s _ { t } , g ) : = \arg \operatorname* { m a x } _ { g _ { s } \in \{ g _ { s } ^ { i } \} , g _ { s } ^ { i } \sim \pi _ { h } ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , g ) } Q _ { h } ( s _ { t } , g , g _ { s } ^ { i } ) , } \\ & { { \pi } _ { l } ( a _ { t } \mid s _ { t } , g _ { s } ) : = \pi _ { l } ^ { \mathrm { B C } } ( a _ { t } \mid s _ { t } , g _ { s } ) . } \end{array}
$$

Note that conservative extrapolation is still possible as the low-level policy might be queried at an absolute subgoal option (proposed by the high-level policy) that was unseen during training.

Absolute Subgoal Options: ASO As explained in Section 4, for QSO we use advantage-weighting to train the high-level policy. Therefore, as a baseline, we introduce a new variant of SHARSA that, like QSO, uses AWR to train the high-level policy:

$$
\begin{array} { r l } &  \mathcal { L } _ { \pi _ { h } ^ { \mathrm { A W R } } } = \mathbb { E } _ { \underset { u \sim \mathrm { U n i f } \left( \left[ 0 , 1 \right] \right) , g _ { s } = s _ { t + k } , g _ { s } ^ { u } = \left( 1 - u \right) z + u g _ { s } } { \left( s _ { t } , s _ { t + k } , g \right) \sim \mathcal { N } \left( 0 , I _ { d _ { s } } \right) } } \\ & { \qquad \left[ \exp \left( \frac { Q _ { h } \left( s _ { t } , g , g _ { s } \right) - V _ { h } \left( s _ { t } , g \right) } { \alpha } \right) \cdot \left. v _ { \pi _ { h } ^ { \mathrm { A W R } } } \left( s _ { t } , g , u , g _ { s } ^ { u } \right) - \left( g _ { s } - z \right) \right. _ { 2 } ^ { 2 } \right] . } \end{array}
$$

At deployment, the high-level policy is still defined using Rejection Sampling and the low-level with behaviour cloning:

$$
\begin{array} { r l } & { \pi _ { h } ( g _ { s } \mid s _ { t } , g ) : = \mathrm { a r g m a x } _ { g _ { s } \in \{ g _ { s } ^ { i } \} , g _ { s } ^ { i } \sim \pi _ { h } ^ { \mathrm { A W R } } ( \cdot \mid s _ { t } , g ) } Q _ { h } ( s _ { t } , g , g _ { s } ^ { i } ) , } \\ & { \pi _ { l } ( a _ { t } \mid s _ { t } , g _ { s } ) : = \pi _ { l } ^ { \mathrm { B C } } ( a _ { t } \mid s _ { t } , g _ { s } ) . } \end{array}
$$

Abstract Relative Options: HIQL HIQL (Park et al., 2023) shapes the abstract subgoal option space through its single, action-free state-value function $V ( s , \phi _ { \omega } ( s , g ) )$ , which is trained using Implicit V-Learning (IVL) (Park et al., 2025):

$$
\mathcal { L } _ { V , \phi _ { \omega } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , g ) \sim \mathcal { D } , g \sim p ^ { D } ( \cdot | s , a ) } \left[ \ell _ { v } ^ { 2 } \left( V \big ( s _ { t } , \phi _ { \omega } ( s _ { t } , g ) \big ) - r ( s _ { t } , g ) - \gamma \tilde { V } \big ( s _ { t + 1 } , \tilde { \phi } _ { \omega } ( s _ { t + 1 } , g ) \big ) \right) \right] ,
$$

where $\tilde { V }$ and $\tilde { \phi } _ { \omega }$ denote the target network and representations and $\ell _ { \nu } ^ { 2 }$ the expectile loss. HIQL extracts high and low-level Gaussian policies using AWR-like objectives:

$$
\begin{array} { r l r } & { } & { { \mathcal L } _ { \pi _ { h } } = - \mathbb E _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal D , \omega = \phi _ { \omega } ( s _ { t } , s _ { t + k } ) } \left[ \exp \left( \frac { V \left( s _ { t + k } , g \right) - V \left( s _ { t } , g \right) } { \alpha } \right) \cdot \log { \pi _ { h } ( \omega \mid s _ { t } , g ) } \right] , } \\ & { } & { { \mathcal L } _ { \pi _ { l } } = - \mathbb E _ { ( s _ { t } , s _ { t + k } , g ) \sim \mathcal D , \omega = \phi _ { \omega } ( s _ { t } , s _ { t + k } ) } \left[ \exp \left( \frac { V \left( s _ { t + 1 } , s _ { t + k } \right) - V \left( s _ { t } , s _ { t + k } \right) } { \alpha } \right) \cdot \log { \pi _ { l } ( a _ { t } \mid s _ { t } , \omega ) } \right] . } \end{array}
$$

Abstract Options: OPAL OPAL $( \mathrm { A j a y }$ et al., 2020) shapes the abstract subgoal option space using $\mathbf { a _ { \alpha } } \beta \mathbf { - } \mathbf { V } \mathbf { A } \mathbf { E }$ trained to encode state-action trajectory sequences:

$$
\operatorname* { m i n } _ { \pi _ { l } , p _ { \omega } , q _ { \omega } } - \mathbb { E } _ { \tau \sim \mathcal { D } , \omega \sim q _ { \omega } ( \cdot | \tau ) } \left[ \sum _ { t = 0 } ^ { k - 1 } \log \pi _ { l } ^ { \mathrm { B C } } ( a _ { t } \mid s _ { t } , \omega ) \right] + \beta \cdot \mathbf { D } _ { \mathbf { K L } } ( q _ { \omega } ( \omega \mid \tau ) \mid \mid p _ { \omega } ( \omega \mid s _ { 0 } ) ) ,
$$

where $q _ { \omega }$ and $p _ { \omega }$ respectively denote the option-space encoder and option-space prior. Using this learned option space to encode trajectory sequences, state-value and Q functions are trained using IQL:

$$
\begin{array} { r l } & { \mathcal { L } _ { V } = \mathbb { E } _ { ( \tau , g ) \sim \mathcal { D } , \omega \sim q _ { \omega } ( \cdot | \tau ) } \left[ \ell _ { \nu } ^ { 2 } \left( V ( s _ { t } , g ) - \tilde { Q } ( s _ { t } , g , \omega ) \right) \right] , } \\ & { \mathcal { L } _ { Q } = \mathbb { E } _ { ( \tau , g ) \sim \mathcal { D } , \omega \sim q _ { \omega } ( \cdot | \tau ) } \left[ \left( Q ( s _ { t } , g , \omega ) - \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { k - 1 } \gamma ^ { t ^ { \prime } } r ( s _ { t + t ^ { \prime } } , g ) - \gamma ^ { k } V ( s _ { t + k } , g ) \right) ^ { 2 } \right] , } \end{array}
$$

and a high-level flow-matching policy is fitted using the following behaviour-cloning flow-matching loss:

$$
\mathcal { L } _ { \pi _ { h } ^ { \mathrm { B C } } } = \mathbb { E } \frac { ( \tau , g ) \sim \mathcal { D } , z \sim \mathcal { N } ( 0 , I _ { d \omega } ) } { u \sim \mathrm { U n i f } ( [ 0 , 1 ] ) , \omega \sim q _ { \omega } ( \cdot | \tau ) , \omega ^ { u } = ( 1 - u ) z + u \omega } \left[ \left\| v _ { \pi _ { h } ^ { \mathrm { B C } } } \big ( s _ { t } , g , u , \omega ^ { u } \big ) - ( \omega - z ) \right\| _ { 2 } ^ { 2 } \right] ,
$$

At deployment, the high-level policy is defined using Rejection Sampling, and the low-level with behaviour cloning:

$$
\begin{array} { r l } & { \pi _ { h } ( \omega \mid s _ { t } , g ) : = \mathrm { a r g m a x } _ { \omega \in \{ \omega ^ { i } \} , \omega ^ { i } \sim \pi _ { h } ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , g ) } Q _ { h } ( s _ { t } , g , \omega ^ { i } ) , } \\ & { \pi _ { l } ( a _ { t } \mid s _ { t } , \omega ) : = \pi _ { l } ^ { \mathrm { B C } } ( a _ { t } \mid s _ { t } , \omega ) . } \end{array}
$$

In practice, the low-level policy is finetuned to output the immediate action.

Non-hierarchical: IQL IQL (Kostrikov et al., 2021) trains a value and Q function with the following losses:

$$
\begin{array} { r l } & { \mathcal { L } _ { V } = \mathbb { E } _ { ( s _ { t } , a _ { t } , g ) \sim \mathcal { D } } \left[ \ell _ { \nu } ^ { 2 } \left( V ( s _ { t } , g ) - \tilde { Q } ( s _ { t } , g , a _ { t } ) \right) \right] , } \\ & { \mathcal { L } _ { Q } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , g ) \sim \mathcal { D } } \left[ \left( Q ( s _ { t } , g , a _ { t } ) - r ( s _ { t } , g ) - \gamma V ( s _ { t + 1 } , g ) \right) ^ { 2 } \right] , } \end{array}
$$

We then extract a Gaussian policy by maximising the Deep Deterministic Policy Gradient objective (Fujimoto and Gu, 2021):

$$
\mathcal { L } _ { \pi } = - \mathbb { E } _ { ( s _ { t } , a _ { t } , g ) \sim \mathcal { D } } \bigg [ Q \big ( s , g , \mu _ { \pi } ( s , g ) \big ) + \alpha \log \pi ( a \mid s , g ) \bigg ] ,
$$

where $\mu _ { \pi }$ denotes the mean of the Gaussian policy. Following Park et al. (2024), we use DDPGBC rather than AWR for policy extraction.

Action Chunking: QC QC (Li et al., 2026) trains an action-chunked Q function:

$$
\mathcal { L } _ { Q } = \mathbb { E } _ { ( s _ { t } , a _ { t : t + k } , s _ { t + k } , g ) \sim \mathcal { P } } \left[ \Big ( Q \big ( s _ { t } , g , a _ { t : t + k } \big ) - \sum _ { t ^ { \prime } = 0 } ^ { k - 1 } \gamma ^ { t ^ { \prime } } r ( s _ { t + t ^ { \prime } } , g ) - \gamma ^ { k } Q \big ( s _ { t + k } , g , a _ { t + k : t + 2 k } \big ) \Big ) ^ { 2 } \right] .
$$

It fits a flow-matching policy using behaviour-cloning:

$$
\begin{array} { r } { \mathcal { L } _ { \pi ^ { \mathrm { B C } } } = \mathbb { E } _ { \underset { u \sim \operatorname { U n i f } \left( \left[ 0 , 1 \right] \right) , a ^ { u } = \left( 1 - u \right) z + u a _ { t + t + k } } { \left( s _ { t } , a _ { t + t + k } , g \right) \sim \mathcal { D } , z \sim \mathcal { N } \left( 0 , I _ { d } \right) } } \left[ \left\| v _ { \pi _ { l } } \left( s _ { t } , g , u , a ^ { u } \right) - \left( a _ { t : t + k } - z \right) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

The policy is defined using Rejection Sampling:

$$
\begin{array} { r } { \pi ( a _ { t : t + k } \mid s _ { t } , g ) : = \mathrm { a r g m a x } _ { a _ { t : t + k } \in \{ a _ { t : t + k } ^ { i } \} , \{ a _ { t : t + k } \} ^ { i } \sim \pi ^ { \mathrm { B C } } ( \cdot \mid s _ { t } , g ) } Q \big ( s _ { t } , g , a _ { t : t + k } ^ { i } \big ) . } \end{array}
$$

## H.2.2 DATASET SAMPLING FOR NON-HIERARCHICAL AND ACTION CHUNKING BASELINES

The dataset sampling intends to mirror the hierarchical ones as closely as possible, to enable the isolation of the hierarchical option space on performance as opposed to dataset sampling.

Non-hierarchical: IQL The flat baseline has no option structure, so there is no distinction between a near and a distant goal and the option window collapses to the current state. Goals are sampled according to:

$$
p _ { \mathrm { f l a t } } ( g \mid s _ { t } ) = w ^ { \mathrm { c u r } } \delta _ { s _ { t } } ( g ) + w ^ { \mathrm { t r a j } } p _ { \mathrm { f l a t } } ^ { \mathrm { t r a j } } ( g \mid s _ { t } ) + w ^ { \mathrm { r a n d } } \operatorname { U n i f } _ { \mathcal { D } } ( g ) ,
$$

where the second component samples the goal uniformly from the remainder of the trajectory beyond the current state,

$$
p _ { \mathrm { f l a t } } ^ { \mathrm { t r a j } } ( g \mid s _ { t } ) = \mathcal { U } \big ( \{ \operatorname* { m i n } ( t + 1 , H ) , \dots , H \} \big ) .
$$

and, like for hierarchical high-level dataset sampling, the third component samples goals uniformly from the entire dataset. We use the same weighting as for the hierarchical high-level dataset sampling as opposed to the low-level one, since the single non-hierarchical policy must cover the full range of state-goal possibilities with one distribution, in contrast to the hierarchical case where the near and far regimes can be split across two levels.

Action Chunking: QC The action chunking baseline is flat but temporally extended, so it inherits the high-level goal distribution of Equation 8 and additionally requires the length-k action sequence beginning at $s _ { t } ,$

$$
a _ { t : t + k } = \big ( a _ { \operatorname* { m i n } ( t , H ) } , \ \cdot \ \cdot \ \cdot , \ a _ { \operatorname* { m i n } ( t + k - 1 , H ) } \big ) ,
$$

where indices are clipped to the end of the trajectory. If an agent reaches its target state before the end of the k length option chunk, actions past the target state are masked out and set to zero, so that the critic is not trained on behaviour occurring past termination.

Following original implementations of non-hierarchical methods, we use strict reward labelling.

## H.3 HYPERPARAMETERS

We use the following default hyperparameters for all value networks (both Value and Q functions), flow-matching and Gaussian policies, and Rejection Sampling methods. Default learning rate, batch size, target update rate, layer normalisation, non-linearities, and representation latent dimensions follow Park et al. (2025). According to Park et al. (2023), we use 128 for HIQL. Rejection Sampling N follows Park et al. (2026). Default network sizes and actor flow steps were chosen according to available compute time and memory. For Gaussian policies, following Park et al. (2023), we use a network that outputs a mean and constant standard deviation. We use the number of steps from Park et al. (2023), apart from for QC, where we use their suggestion of 5.

Table 5: Hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimiser</td><td>Adam (Kingma and Ba, 2014)</td></tr><tr><td>Learning Rate</td><td>0.0003</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td>Target Update Rate</td><td>0.005</td></tr><tr><td>Representation Latent Dimension</td><td>10 (default), 128 (HIQL)</td></tr><tr><td>Representation Layer Normalisation</td><td>True</td></tr><tr><td>Representation Non-linearity</td><td>GELU (Hendrycks and Gimpel, 2016)</td></tr><tr><td>Representation MLP</td><td>[512, 512, 512]</td></tr><tr><td>Value Layer Normalisation</td><td>True</td></tr><tr><td>Value Non-linearity</td><td>GELU (Hendrycks and Gimpel, 2016)</td></tr><tr><td>Value MLP</td><td>[512, 512, 512, 512]</td></tr><tr><td>Value Expectile ν (IQL)</td><td>0.7</td></tr><tr><td>Actor Layer Normalisation</td><td>True</td></tr><tr><td>Actor Non-linearity</td><td>GELU (Hendrycks and Gimpel, 2016)</td></tr><tr><td>Actor MLP Actor Flow Steps</td><td>[512, 512, 512, 512]</td></tr><tr><td>Actor Rejection Sampling N</td><td>6</td></tr><tr><td></td><td>32</td></tr><tr><td>Actor AWR α Actor DDPGBC α</td><td>3.0</td></tr><tr><td>Subgoal Steps k</td><td>1.0 (IQL default), 0.3 (IQL ant), 0.1 (IQL humanoid)</td></tr><tr><td></td><td>25 (default), 100 (humanoid), 5 (QC)</td></tr><tr><td>Hierarchical Goal Sampling  $( w _ { h } ^ { \mathrm { o p t } } , w _ { h } ^ { \mathrm { t r a j } } , w _ { h } ^ { \mathrm { r a n d } } )$ </td><td>(0.1, 0.5, 0.4)</td></tr><tr><td>Hierarchical Goal Sampling  $( w _ { l } ^ { \mathrm { c u r } } , w _ { l } ^ { \mathrm { t r a j } } , w _ { l } ^ { \mathrm { r a n d } } )$ </td><td>(0.1, 0.85, 0.05)</td></tr><tr><td>Non-hierarchical Goal Sampling  $( w ^ { \mathrm { c u r } } , w ^ { \mathrm { t r a j } } , w ^ { \mathrm { r a n d } } )$ </td><td>(0.1, 0.5, 0.4)</td></tr><tr><td>Discount γ</td><td>0.99 (default), 0.995/0.999 (HIQL &amp; IQL ant/humanoid)</td></tr></table>

<table><tr><td></td><td></td><td colspan="4">Non-Hierarchical</td><td colspan="10">Hierarchical</td></tr><tr><td>Environment</td><td>Task</td><td>IQL</td><td></td><td>QC</td><td>HIQL</td><td></td><td>OPAL</td><td></td><td>SHARSA</td><td></td><td>DSHARSA</td><td></td><td>ASO</td><td></td><td>QSO</td><td></td></tr><tr><td rowspan="5">antmaze-giant-stitch-v0</td><td>Task 1</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>35 [29, 39]</td><td>35</td><td>[33, 37]</td><td>59</td><td>[42,</td><td>71]</td><td>72 [63,</td><td>86]</td><td>57</td><td>[47, 67]</td><td>57</td><td>[50, 63]</td></tr><tr><td>Task 2</td><td></td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>66 [60,</td><td>70] 83</td><td>[77,</td><td>90] 74</td><td>[65,</td><td>87]</td><td>67 [60,</td><td>73]</td><td>67 [64,</td><td>69]</td><td>75</td><td>[69, 79]</td></tr><tr><td>Task 3</td><td></td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>42 [38,</td><td>45] 62</td><td>[52,</td><td>71] 53</td><td>[39,</td><td>66]</td><td>72 [66,</td><td>78]</td><td>66</td><td>[58, 72]</td><td>73</td><td>[66, 81]</td></tr><tr><td>Task 4</td><td></td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>57 [50,</td><td>62] 17</td><td>[12,</td><td>22] 65</td><td>[57,</td><td>72]</td><td>67 [57,</td><td>77]</td><td>65</td><td>[60, 72]</td><td>52</td><td>[43, 61]</td></tr><tr><td>Task 5</td><td></td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>69 [53,</td><td>86] 69</td><td>[65,</td><td>74] 90</td><td>[85,</td><td>95]</td><td>82 [80,</td><td>85]</td><td>84 [70,</td><td>94]</td><td>82</td><td>[78, 87]</td></tr><tr><td rowspan="6">antmaze-teleport-stitch-v0</td><td>Overall</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>54 [48,</td><td>60] 53</td><td>[52,</td><td>55] 68</td><td>[61,</td><td>76]</td><td>72 [67,</td><td>77]</td><td>68</td><td>[66, 70]</td><td>68</td><td>[66, 70]</td></tr><tr><td>Task 1</td><td>35 [27,</td><td>44]</td><td>27 [18, 33]</td><td>44 [40,</td><td>48] 61</td><td>[50,</td><td>72] 18</td><td>[12,</td><td>28]</td><td>28 [17,</td><td>38]</td><td>54</td><td>[46, 65]</td><td>34</td><td>[28, 40]</td></tr><tr><td>Task 2</td><td>47 [42,</td><td>52] 38</td><td>[32, 43] 47]</td><td>44 [41,</td><td>48] 85</td><td>[81,</td><td>91] 77</td><td>[70,</td><td>83]</td><td>87 [83,</td><td>93]</td><td>75 [70,</td><td>80]</td><td>75</td><td>[73, 78]</td></tr><tr><td>Task 3</td><td>57 [52, 51 [40,</td><td>62] 60]</td><td>40 [33,</td><td>34 [27,</td><td>42] 87</td><td>[82,</td><td>90] 80</td><td>[76,</td><td>83]</td><td>74 [71,</td><td>78]</td><td>82</td><td>[78, 88]</td><td>68</td><td>[53, 79]</td></tr><tr><td>Task 4 Task 5</td><td>48 [38,</td><td>57]</td><td>42 [36, 50] 37 [27,</td><td>26 [13,</td><td>41] 50</td><td>[47,</td><td>54] 55</td><td>[51,</td><td>61]</td><td>51 [45,</td><td>58]</td><td>59 [51,</td><td>65]</td><td>50</td><td>[45, 57]</td></tr><tr><td></td><td></td><td></td><td>48]</td><td>32 [26,</td><td>38] 61</td><td>[52,</td><td>71] 33</td><td>[26,</td><td>42]</td><td>48 [38,</td><td>58]</td><td>58 [50,</td><td>67]</td><td>46</td><td>[43, 48]</td></tr><tr><td rowspan="6">humanoidmaze-giant-stitch-v0</td><td>Overall</td><td>47 [44,</td><td>51] 37</td><td>[32, 40]</td><td>36 [34, 39]</td><td>69</td><td>[64,</td><td>74] 53</td><td>[52,</td><td>54]</td><td>58 [55,</td><td>62]</td><td>66 [62,</td><td>69]</td><td>54</td><td>[52, 57]</td></tr><tr><td>Task 1</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>5 [0, 10]</td><td>73</td><td>[63,</td><td>83] 12</td><td>[8,</td><td>18]</td><td>17 [12,</td><td>22]</td><td>22</td><td>[18, 28]</td><td>82</td><td>[77, 89]</td></tr><tr><td>Task 2</td><td>3 [1, 6] 1 [0, 2]</td><td></td><td>0 [0, 0] 0 [0, 0]</td><td>18 [8, 27]</td><td>78 85</td><td>[65,</td><td>89] 41</td><td>[33,</td><td>51]</td><td>47</td><td>[32,69]</td><td>32</td><td>[18, 43]</td><td>98</td><td>[95, 100]</td></tr><tr><td>Task 3 Task 4</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>6 [4, 7] 2 [0, 3]</td><td>78</td><td>[72, [73,</td><td>92] 42 83] 11</td><td>[28, [10,</td><td>61] 12]</td><td>61 [58, 8 [3, 12]</td><td>65]</td><td>55 [48, 19</td><td>62] [17, 22]</td><td>91</td><td>[88, 93]</td></tr><tr><td>Task 5</td><td>0 [0, 0]</td><td></td><td>0 [0, 0]</td><td>1 [0, 2]</td><td>95</td><td>[92,</td><td>98] 43</td><td>[32,</td><td>61]</td><td>17 [5,</td><td>28]</td><td>92 [84,</td><td>98]</td><td>85 97</td><td>[82, 87]</td></tr><tr><td>Overall</td><td>1 [1, 1]</td><td></td><td>0 [0, 0]</td><td>6 [4, 8]</td><td>82</td><td>[78,</td><td>85] 30</td><td>[27,</td><td>33] 30</td><td>[27,</td><td>32] 44</td><td>[38,</td><td>48] 90</td><td>[94,</td><td>99] [88, 92]</td></tr></table>

Table 6: Full results on selected OGBench locomotion environments using default datasets. Success rate (%) on each evaluation task after 1M offline training steps. Mean and 95% bootstrap confidence interval over 4 seeds.

## H.4 FURTHER RESULTS

![](images/e8d5c97dbc9cfdc8f90a86eff12941315483843082dec4f3dfa441c6776c0a79.jpg)  
Figure 5: Training on selected OGBench environments using the default datasets. Success rate averaged over 5 evaluation tasks. Mean and 95% bootstrap confidence interval over 4 seeds.

<table><tr><td rowspan=1 colspan=10>Non-Hierarchical                               Hierarchical</td></tr><tr><td rowspan=1 colspan=4>Environment        Task      IQL        QC</td><td rowspan=1 colspan=6>HIQL     OPAL     SHARSA     DSHARSA      ASO       QSO</td></tr><tr><td rowspan=1 colspan=2>Task 1</td><td rowspan=1 colspan=2>62 [53, 70]  90 [86, 93]</td><td rowspan=1 colspan=2>26 [20, 32]82 [78, 87]</td><td rowspan=1 colspan=3>95 [92, 98]  98 [97, 100]  96 [92, 99]</td><td rowspan=1 colspan=1>99 [98, 100]</td></tr><tr><td rowspan=1 colspan=1>cube-double-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>11 [3, 18]</td><td rowspan=1 colspan=1>88 [83, 96]</td><td rowspan=1 colspan=1>2 [0, 5]</td><td rowspan=1 colspan=1>12 [4, 18]</td><td rowspan=1 colspan=3>82 [78, 86]  93 [91, 96]  88 [87, 90]</td><td rowspan=1 colspan=1>68 [53, 78]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>20 [16, 23]</td><td rowspan=1 colspan=1>88 [85, 90]</td><td rowspan=1 colspan=1>4 [1, 8]</td><td rowspan=1 colspan=1>11 [6, 15]</td><td rowspan=1 colspan=3>89 [87, 92]  91 [86, 95]  85 [83, 88]</td><td rowspan=1 colspan=1>68 [62, 76]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=2>2 [0, 5]   42 [38, 45]</td><td rowspan=1 colspan=1>2 [0, 3]</td><td rowspan=1 colspan=1>15 [8, 22]</td><td rowspan=1 colspan=3>41 [33, 48]  59 [48, 68]  20 [13, 27]</td><td rowspan=1 colspan=1>53 [50, 57]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Task 5  6 [3, 8]   54 [45, 63]</td><td rowspan=1 colspan=1>2 [0, 3]</td><td rowspan=1 colspan=1>25 [18, 32]</td><td rowspan=1 colspan=4>53 [43, 67]  58 [50,64]  48 [40, 57]  61 [55, 69]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Overall 20 [17, 23]  72 [70, 74]</td><td rowspan=1 colspan=2>7 [5, 9] 29 [26, 33]</td><td rowspan=1 colspan=4>72[68,77]  80 [75, 84]  68 [65, 70]  70 [65, 74]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>29 [27, 32]</td><td rowspan=1 colspan=1>57 [48, 68]</td><td rowspan=1 colspan=1>58 [51, 65]</td><td rowspan=1 colspan=1>8 [5, 12]</td><td rowspan=1 colspan=4>45 [29, 58]  44 [40, 48]  16 [13, 18]  45 [42, 48]</td></tr><tr><td rowspan=1 colspan=1>cube-triple-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>31 [18, 42]</td><td rowspan=1 colspan=1>7 [7, 7]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>15 [11, 21]</td><td rowspan=1 colspan=1>16 [11, 23]</td><td rowspan=1 colspan=1>2 [1, 3]</td><td rowspan=1 colspan=1>28 [27, 30]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>20 [17, 23]</td><td rowspan=1 colspan=1>12 [10, 13]</td><td rowspan=1 colspan=1>2 [1, 3]</td><td rowspan=1 colspan=1>5 [3, 7]</td><td rowspan=1 colspan=1>5 [0, 10]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>17 [12, 22]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>4 [2, 7]</td><td rowspan=1 colspan=1>3 [0, 8]</td><td rowspan=1 colspan=1>2 [0, 5]</td><td rowspan=1 colspan=1>3 [1, 6]</td><td rowspan=1 colspan=2>6 [2, 9]    0 [0, 0]</td><td rowspan=1 colspan=1>15 [12, 18]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Task 5  0 [0, 0]</td><td rowspan=1 colspan=1>6 [4, 7]</td><td rowspan=1 colspan=2>1 [0, 2]  1 [0, 2]</td><td rowspan=1 colspan=4>2 [0, 5]    1 [0, 2]    0 [0, 0]    3 [3, 3]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall  6 [6, 6]</td><td rowspan=1 colspan=1>24 [19, 27]</td><td rowspan=1 colspan=2>16 [14, 18] 3 [2, 4]</td><td rowspan=1 colspan=4>14 [10, 18]  14 [13, 16]   4 [4, 4]   22 [20, 23]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>4 [3, 6]</td><td rowspan=1 colspan=1>11 [5, 19]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>2 [0, 3]</td><td rowspan=1 colspan=1>2 [0, 3]</td><td rowspan=1 colspan=2>0 [0, 0]    9 [8, 10]</td></tr><tr><td rowspan=1 colspan=1>cube-quadruple-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>4 [3, 6]</td><td rowspan=1 colspan=1>4 [2, 7]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>10 [7, 13]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>2 [0, 5]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Task 5  0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=2>0 [0, 0]  0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=2>0 [0, 0]    1 [0, 2]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall  0 [0, 0]</td><td rowspan=1 colspan=1>2 [1, 2]</td><td rowspan=1 colspan=2>3 [1, 5]  0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 1]</td><td rowspan=1 colspan=1>0 [0, 1]</td><td rowspan=1 colspan=2>0 [0, 0]    4 [4, 5]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>86 [78, 93]</td><td rowspan=1 colspan=1>100 [100, 100]</td><td rowspan=1 colspan=1>55 [40, 65]</td><td rowspan=1 colspan=1>99 [98, 100]</td><td rowspan=1 colspan=1>100 [100, 100]</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=2>100[100, 100]100 [100, 100]</td></tr><tr><td rowspan=1 colspan=1>scene-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>54 [49, 58]</td><td rowspan=1 colspan=1>98 [97, 99]</td><td rowspan=1 colspan=1>59 [52, 71]</td><td rowspan=1 colspan=1>92 [88, 98]</td><td rowspan=1 colspan=1>98 [97, 100]</td><td rowspan=1 colspan=1>96 [93, 98]</td><td rowspan=1 colspan=1>98 [95, 100]</td><td rowspan=1 colspan=1>99 [98,100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>64 [57, 72]</td><td rowspan=1 colspan=1>97 [94, 99]</td><td rowspan=1 colspan=1>22[20,23]</td><td rowspan=1 colspan=1>80 [67, 88]</td><td rowspan=1 colspan=1>98 [97, 100]</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>82 [78, 88]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>33 [23, 43]</td><td rowspan=1 colspan=1>65 [60, 70]</td><td rowspan=1 colspan=1>47 [38, 55]</td><td rowspan=1 colspan=1>76 [68, 80]</td><td rowspan=1 colspan=1>85 [75, 92]</td><td rowspan=1 colspan=1>84 [82, 87]</td><td rowspan=1 colspan=1>86 [78, 93]</td><td rowspan=1 colspan=1>75 [65, 82]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 5</td><td rowspan=1 colspan=1>3 [1, 6]</td><td rowspan=1 colspan=1>52 [46, 56]</td><td rowspan=1 colspan=1>22 [13,30]</td><td rowspan=1 colspan=1>70 [59, 78]</td><td rowspan=1 colspan=1>78[72, 85]</td><td rowspan=1 colspan=1>64 [52,77]</td><td rowspan=1 colspan=1>85 [81, 91]</td><td rowspan=1 colspan=1>69 [60, 80]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall 48 [45, 51]</td><td rowspan=1 colspan=1>82 [81, 83]</td><td rowspan=1 colspan=1>41 [35, 46]</td><td rowspan=1 colspan=1>84 [78, 87]</td><td rowspan=1 colspan=1>92 [90, 93]</td><td rowspan=1 colspan=1>89 [87, 91]</td><td rowspan=1 colspan=2>94 [91, 95]  85[82,90]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>94 [82, 100]</td><td rowspan=1 colspan=1>63 [57, 68]</td><td rowspan=1 colspan=1>94 [90, 98]</td><td rowspan=1 colspan=1>100 [100, 100]</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>99 [98, 100]</td><td rowspan=1 colspan=1>100 [100, 100]</td></tr><tr><td rowspan=1 colspan=1>puzzle-3x3-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>91 [75,100]</td><td rowspan=1 colspan=1>58[42,73]</td><td rowspan=1 colspan=1>72 [59, 81]</td><td rowspan=1 colspan=1>74 [55, 93]</td><td rowspan=1 colspan=1>85 [79,91]</td><td rowspan=1 colspan=1>78 [70, 87]</td><td rowspan=1 colspan=1>81 [73, 88]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>100[100, 100]</td><td rowspan=1 colspan=1>74 [25, 100]</td><td rowspan=1 colspan=1>32 [30, 35]</td><td rowspan=1 colspan=1>58 [52, 62]</td><td rowspan=1 colspan=1>67 [58, 76]</td><td rowspan=1 colspan=1>78 [70, 87]</td><td rowspan=1 colspan=1>70 [68, 72]</td><td rowspan=1 colspan=1>66 [59, 73]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>99 [98, 100]</td><td rowspan=1 colspan=1>72 [24, 98]</td><td rowspan=1 colspan=1>36 [32,41]</td><td rowspan=1 colspan=1>59 [57, 64]</td><td rowspan=1 colspan=1>69 [58, 80]</td><td rowspan=1 colspan=1>76 [67, 86]</td><td rowspan=1 colspan=1>52 [40, 62]</td><td rowspan=1 colspan=1>48 [38, 58]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 5</td><td rowspan=1 colspan=1>100[100,100]</td><td rowspan=1 colspan=1>72[25,100]</td><td rowspan=1 colspan=1>49[43, 55]</td><td rowspan=1 colspan=1>65[55,75]</td><td rowspan=1 colspan=1>78[68, 90]</td><td rowspan=1 colspan=1>84 [78, 90]</td><td rowspan=1 colspan=1>79 [75, 84]</td><td rowspan=1 colspan=1>64 [62, 67]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall100[100,100]</td><td rowspan=1 colspan=1>81 [46, 99]</td><td rowspan=1 colspan=1>48 [46, 49]</td><td rowspan=1 colspan=1>70 [65, 74]</td><td rowspan=1 colspan=1>78 [71, 87]</td><td rowspan=1 colspan=1>85 [81, 90]</td><td rowspan=1 colspan=1>76 [72, 80]</td><td rowspan=1 colspan=1>72 [69, 75]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>85 [77, 92]</td><td rowspan=1 colspan=1>94 [90, 98]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>70 [63,81]</td><td rowspan=1 colspan=1>85[82,88]</td><td rowspan=1 colspan=1>76 [68,83]</td><td rowspan=1 colspan=1>81 [77, 85]</td><td rowspan=1 colspan=1>96 [92,100]</td></tr><tr><td rowspan=1 colspan=1>puzzle-4x4-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>47 [33, 60]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>63 [48, 78]</td><td rowspan=1 colspan=1>72 [66, 78]</td><td rowspan=1 colspan=1>67 [62, 70]</td><td rowspan=1 colspan=1>68 [61, 75]</td><td rowspan=1 colspan=1>92 [80, 100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>78 [64, 90]</td><td rowspan=1 colspan=1>32 [20, 43]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>65 [56, 78]</td><td rowspan=1 colspan=1>74 [63, 83]</td><td rowspan=1 colspan=1>78 [75, 80]</td><td rowspan=1 colspan=1>60 [51, 76]</td><td rowspan=1 colspan=1>92 [80,100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>72 [64, 78]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>52 [50, 53]</td><td rowspan=1 colspan=1>71 [62, 80]</td><td rowspan=1 colspan=1>68 [57, 78]</td><td rowspan=1 colspan=1>71 [68, 75]</td><td rowspan=1 colspan=1>89 [75,100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 5</td><td rowspan=1 colspan=1>72 [56, 88]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>72 [67, 78]</td><td rowspan=1 colspan=1>68 [53, 82]</td><td rowspan=1 colspan=1>73 [68, 82]</td><td rowspan=1 colspan=1>67 [62, 73]</td><td rowspan=1 colspan=1>92 [78, 100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall 71 [61, 78]</td><td rowspan=1 colspan=1>25 [23, 27]</td><td rowspan=1 colspan=1>0 [0, 1]</td><td rowspan=1 colspan=1>64 [61, 68]</td><td rowspan=1 colspan=1>74 [67, 79]</td><td rowspan=1 colspan=1>72 [69,75]</td><td rowspan=1 colspan=2>69 [65, 75]  92 [81,100]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>68 [53, 76]</td><td rowspan=1 colspan=1>81 [63, 97]</td><td rowspan=1 colspan=1>18 [13, 23]</td><td rowspan=1 colspan=1>43 [39, 47]</td><td rowspan=1 colspan=1>46 [38, 55]</td><td rowspan=1 colspan=1>45 [40, 52]</td><td rowspan=1 colspan=1>46 [36, 60]</td><td rowspan=1 colspan=1>36 [29, 42]</td></tr><tr><td rowspan=1 colspan=1>puzzle-4x5-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>36 [27, 46]</td><td rowspan=1 colspan=1>17 [10, 23]</td><td rowspan=1 colspan=1>27 [18, 35]</td><td rowspan=1 colspan=1>19 [7, 32]</td><td rowspan=1 colspan=1>20 [15, 25]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>7 [2, 12]</td><td rowspan=1 colspan=1>20[12,27]</td><td rowspan=1 colspan=1>21 [18, 23]</td><td rowspan=1 colspan=1>6 [3, 8]</td><td rowspan=1 colspan=1>3 [0, 10]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>30 [23, 37]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>17 [5, 28]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Task 5  0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>2 [0, 3]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall 14 [11, 15]</td><td rowspan=1 colspan=1>16 [13, 19]</td><td rowspan=1 colspan=2>4 [3, 5] 24 [22, 25]</td><td rowspan=1 colspan=1>17 [15, 20]</td><td rowspan=1 colspan=3>18 [18, 19]  14 [11, 18]  15 [11, 20]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 1</td><td rowspan=1 colspan=1>32 [20, 45]</td><td rowspan=1 colspan=1>80 [68, 92]</td><td rowspan=1 colspan=1>20 [18, 22]</td><td rowspan=1 colspan=1>39 [33, 43]</td><td rowspan=1 colspan=1>39 [30, 49]</td><td rowspan=1 colspan=1>33 [27, 38]</td><td rowspan=1 colspan=1>38 [33, 43]</td><td rowspan=1 colspan=1>45 [41, 51]</td></tr><tr><td rowspan=1 colspan=1>puzzle-4x6-play-v0</td><td rowspan=1 colspan=1>Task 2</td><td rowspan=1 colspan=1>10 [3, 17]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>5 [2, 8]</td><td rowspan=1 colspan=1>34 [31, 38]</td><td rowspan=1 colspan=1>21 [15, 28]</td><td rowspan=1 colspan=1>30 [27, 34]</td><td rowspan=1 colspan=1>14 [5, 24]</td><td rowspan=1 colspan=1>37 [28, 50]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 3</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>8 [5, 10]</td><td rowspan=1 colspan=1>7 [3, 11]</td><td rowspan=1 colspan=1>14[7, 22]</td><td rowspan=1 colspan=1>4 [2, 7]</td><td rowspan=1 colspan=1>8 [3, 13]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task 4</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>2 [0, 5]</td><td rowspan=1 colspan=1>1 [0, 2]</td><td rowspan=1 colspan=1>1[0, 2]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=1>1 [0, 2]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Task 5  0 [0, 0]    0 [0, 0]</td><td rowspan=1 colspan=2>0 [0, 0]  0 [0, 0]</td><td rowspan=1 colspan=1>0 [0, 0]</td><td rowspan=1 colspan=3>0 [0, 0]    0 [0, 0]    0 [0, 0]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Overall 8 [5, 10]  16 [14, 18]</td><td rowspan=1 colspan=2>5 [4, 6] 17 [15, 18]</td><td rowspan=1 colspan=4>14 [10, 16]  16 [14, 17]  11 [8, 15]  18 [15, 22]</td></tr><tr><td rowspan=1 colspan=4>Average over 55 Tasks        29 [27, 30]  32[29, 34]</td><td rowspan=1 colspan=6>20[20,20]45[44,46] 46[44,49]  49 [48,49]  47 [46, 48]  54 [51, 56]</td></tr></table>

Table 7: Full results on selected OGBench manipulation environments using default datasets. Success rate (%) on each evaluation task after 1M offline training steps. Mean and 95% bootstrap confidence interval over 4 seeds. Last row averages each seed over all 55 tasks, locomotion and manipulation (Table 6).

![](images/ecd963185db076850e54d1f6144c6f244f2353e006cd69bab97ac5c82b2dc734.jpg)

![](images/e200aa71203806be37fc45caa007e426295e3a8e21a7e9856a00b52424c68e76.jpg)

## H.5 ANALYSIS AND ABLATIONS

![](images/5ae52224459a6ef4f5065f71712f90864a12dfeb69210fef4ee0b7f74cdc09b5.jpg)

![](images/818733c43cd303a076d197a124548659b3371fdbb5a44cc212bdde3c5c4cddfe.jpg)

![](images/f9f13daa1f9c4e81fbfecc3347f7426608b180191ec9f58ded23e17f7c189efb.jpg)

![](images/5788c2a4f058251998a72c8d3d708deb59255fb8bff8fd847050c1f582ddbe9e.jpg)

![](images/4062afdcb680ced252f114da1970886cef0dd7e08fcdf56045a48d665d503445.jpg)

![](images/7e548057b4a293b6c49d849d7f8f1f190687be2682915457029a38470cc7037c.jpg)

![](images/90e0038753b55d446df88b4b92f040211c793e2fcc8d6964848069e4a5c5ce51.jpg)

![](images/a66b0521570c08aa435a8b762aa77d37e5f2d3324164a5b964812368f08a12fc.jpg)

![](images/9892b7b6246795a1a7622572b7870728931d314c148247f3bb7fa04196706673.jpg)  
Figure 6: PCA (top) of the action abstraction for cube-double fitted on random segments, showing different groups of segments with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) in units of median distance between two random segments. QSO is the only algorithm that consistently groups functionally equivalent behaviours (Desideratum 3) while separating behaviour groups (Desideratum 2).

![](images/432c0a974df1c342169cdd43ea930bc7fb5da08467c495fe5bfdb59df39e4da0.jpg)

![](images/8d50f3c2a389516c3a395b231693a5ba5fa80ffbe70d9ee9c9606c8d98d7631a.jpg)

![](images/9397eb458218299ee91449ec893e13f2059a4619a8a67d6ba3efaaf17575af2a.jpg)

![](images/c2baf3edb6021359dd861ac961d74c7ce7b4b55da81681c6506f2f1e151b6e19.jpg)

![](images/d3cd9fc9d96c89086a2f9fe0963e8d73d9e765dd30e6bfea84470ef6f2281c8c.jpg)

![](images/b33e55df25d44d82dbbc292baae15fde61c037700d77980cef8b86b3b01d9f24.jpg)

![](images/d9ef9ee10fe346ef976fd1323c055e620795e1306f03a0bf318b8cd948412cb8.jpg)

![](images/d01abe932bd5637fc2aa0295d77302839814d0bfa54e4314e877cb7b149da29b.jpg)

![](images/6b34c10f0d9f24eccd6060d23079881ae90b24b2ca572a0c54915ceec4313741.jpg)

![](images/be500293b61e671f161630f172c848675b664b34bd4a7380d76f0c81a2a44fc5.jpg)  
Figure 7: PCA (top) of the action abstraction for cube-quadruple fitted on random segments, showing different groups of segments with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) in units of median distance between two random segments. QSO is the only algorithm that consistently groups functionally equivalent behaviours (Desideratum 3) while separating behaviour groups (Desideratum 2).

<table><tr><td></td><td colspan="2">Reward Labelling</td><td colspan="4">High-level Policy</td><td colspan="4">High-level Value</td><td colspan="4">Option Encoder</td><td colspan="4">Low-level Policy</td></tr><tr><td>Agent</td><td>High Low</td><td></td><td>Fast</td><td>Slow</td><td></td><td>Interaction/ Noise</td><td>Fast</td><td></td><td>Slow</td><td>Interaction</td><td>Fast</td><td></td><td>Slow</td><td>Interaction</td><td>Fast</td><td>Slow</td><td>Interaction/</td><td>Noise</td></tr><tr><td colspan="10">antmaze-giant-stitch-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QSO</td><td>E</td><td>S</td><td>[31, 39]</td><td>14</td><td>[13, 16]</td><td>50 [47, 54]</td><td>6 [4, 7]</td><td>88</td><td>[86, 90]</td><td>7 [6, 7]</td><td>45 [42,</td><td>47] 45</td><td>[42, 47]</td><td>11 [10, 11]</td><td>59 [57, 60]</td><td>4 [4, 4]</td><td>37</td><td>[36, 38]</td></tr><tr><td></td><td></td><td></td><td>[26, 30]</td><td>18 [17, 19]</td><td></td><td>54 [52, 55]</td><td>6 [4, 7]</td><td>88</td><td>[86, 90]</td><td>6 [6, 7]</td><td>23 [21,</td><td>25] 72</td><td>[70, 74]</td><td>5 [5, 6]</td><td>55 [54, 57]</td><td>5 [4, 5]</td><td>40</td><td>[39, 41]</td></tr><tr><td></td><td>S</td><td></td><td>[30, 38]</td><td>14 [13,</td><td>15]</td><td>52 [49, 56]</td><td>8 [6, 10]</td><td>84</td><td>[81, 87]</td><td>8 [7, 9]</td><td>46 [43,</td><td>48] 45</td><td>[43, 48]</td><td>9 [9, 9]</td><td>54 [53, 56]</td><td>5 [4, 5]</td><td>41</td><td>[40, 42]</td></tr><tr><td></td><td>E</td><td></td><td>[26, 32]</td><td>19</td><td>[18, 20]</td><td>52 [51, 54]</td><td>8 [6, 11]</td><td>84</td><td>[81, 87]</td><td>8 [7, 8]</td><td>28 [25,</td><td>30] 68</td><td>[65, 70]</td><td>5 [5, 5]</td><td>56 [54, 57]</td><td>A [4, 5]</td><td>40</td><td>[39, 41]</td></tr><tr><td>ASO</td><td>S</td><td></td><td>[18, 24]</td><td>13 [12,</td><td>14]</td><td>67 [64, 69]</td><td>5 [4, 6]</td><td>88</td><td>[86, 90]</td><td>7 [7, 8]</td><td></td><td>--</td><td>--</td><td>--</td><td>53 [51, 54]</td><td>5 [5, 5]</td><td>42</td><td>[41, 44]</td></tr><tr><td></td><td>E</td><td></td><td>[18, 24]</td><td>13 [12,</td><td>14]</td><td>67 [64, 69]</td><td>5 [4, 6]</td><td>88</td><td>[86, 90]</td><td>7 [7, 8]</td><td>--</td><td></td><td>--</td><td>--</td><td>53 [51, 54]</td><td>5 [5, 5]</td><td>42</td><td>[41, 44]</td></tr><tr><td></td><td>S</td><td></td><td>[16, 22]</td><td>13 [12,</td><td>14]</td><td>68 [66, 70]</td><td>8 [6, 10]</td><td>83</td><td>[80, 85]</td><td>9 [8, 10]</td><td>--</td><td></td><td>--</td><td>--</td><td>54 [53, 55]</td><td>5 [4, 5]</td><td>41</td><td>[40, 43]</td></tr><tr><td>S</td><td>E</td><td></td><td>[16, 22]</td><td>13 [12,</td><td>14]</td><td>68 [66, 70]</td><td>8 [6,</td><td>10] 83</td><td>[80, 85]</td><td>9 [8, 10]</td><td></td><td>--</td><td>--</td><td>--</td><td>54 [53, 55]</td><td>5 [4, 5]</td><td>41</td><td>[40, 43]</td></tr><tr><td colspan="10">cube-double-play-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>E</td><td>S</td><td>[71, 74]</td><td>5 [5, 5]</td><td></td><td>22 [21, 24]</td><td>3 [2, 4]</td><td>74</td><td>[70, 77]</td><td>23 [20, 26]</td><td>79 [77,</td><td>80] 15</td><td>[14, 16]</td><td>6 [6, 7]</td><td>77 [75, 78] A</td><td>[4, 5]</td><td>19</td><td>[18, 20]</td></tr><tr><td>QSO</td><td>E</td><td></td><td>[23, 27]</td><td>23 [21, 24]</td><td></td><td>52 [51, 53]</td><td>3 [3, 4]</td><td>75</td><td>[72, 78]</td><td>21 [19, 24]</td><td></td><td>8 [8, 9] 84</td><td>[83, 85]</td><td>8 [7, 8]</td><td>71 [69, 72]</td><td>6 [7, 8]</td><td>22 [21,</td><td>23]</td></tr><tr><td></td><td>S</td><td></td><td>[70, 72]</td><td>7 [6, 7]</td><td></td><td>22 [21, 23]</td><td>4 [3, 5]</td><td>70</td><td>[67, 72]</td><td>26 [24, 28]</td><td>76 [75,</td><td>78] 15</td><td>[14, 16]</td><td>9 [8, 9]</td><td>81 [80,</td><td>83] 5 [4, 5]</td><td>14</td><td>[13, 15]</td></tr><tr><td></td><td>E</td><td></td><td>[17, 20] 39</td><td>[37,</td><td>41]</td><td>42 [41, 44]</td><td>4 [3, 5]</td><td>73</td><td>[71, 76]</td><td>23 [21,</td><td>25]</td><td>8 [7, 9] 83</td><td>[82, 84]</td><td>9 [9, 9]</td><td>75 [74, 77]</td><td>5 [4, 5]</td><td>20</td><td>[19, 21]</td></tr><tr><td>ASO</td><td>S</td><td></td><td>[25, 27]</td><td>52 [51,</td><td>53]</td><td>22 [21, 23]</td><td>3 [3, 4]</td><td></td><td>75 [71, 78]</td><td>22 [19, 25]</td><td></td><td>--</td><td>--</td><td>--</td><td>88 [87,</td><td>89] 2 [2, 2]</td><td></td><td>10 [9, 11]</td></tr><tr><td></td><td>E</td><td></td><td>26 [25, 26]</td><td>52 [51,</td><td>53]</td><td>23 [22, 24]</td><td>3 [3, 4]</td><td></td><td>74 [70, 78]</td><td>22 [19, 26]</td><td></td><td>--</td><td>--</td><td>--</td><td>88 [87,</td><td>89] 2 [2, 2]</td><td></td><td>10 [9, 11]</td></tr><tr><td></td><td>S</td><td></td><td>26 [25, 27]</td><td>52 [51,</td><td>53]</td><td>22 [21, 23]</td><td>4 [3, 5]</td><td></td><td>66 [63, 69]</td><td>30 [28,</td><td>32]</td><td>--</td><td>--</td><td>--</td><td>88 [87,</td><td>89] 3 [2, 3]</td><td></td><td>9 [8, 10]</td></tr><tr><td></td><td>E</td><td></td><td>26 [25, 27]</td><td>52 [51,</td><td>53]</td><td>22 [21, 23]</td><td>4 [3, 5]</td><td>66</td><td>[63, 69]</td><td>30 [28,</td><td>32]</td><td>--</td><td>--</td><td>--</td><td>88 [87,</td><td>89] 3 [2, 3]</td><td>9</td><td>[8, 10]</td></tr></table>

Table 8: QSO and ASO high and low-level decision process state abstractions. Percentage share of the variance of each output explained by slow-moving dimensions (state and goal XY coordinates for antmaze-giant; state and goal cube positions and orientations for cube-double) and by the fast-moving dimensions (respectively state and goal XY height, orientation, joints and velocities; and state and goal effector, gripper, joints and velocities). The reward labelling columns indicate whether the high and low-level decision processes are trained using environment (E) or strict (S) reward labelling. Inputs are built on grids that combine the slow-moving dimensions of 16 random validation state-goal (or state-target) pairs with the fast-moving dimensions of 16 others, each under 8 flow-noise samples. For QSO, the state-target pairs are encoded before passing them to the low-level policy; the option encoder columns give the variance of these encoded options (ASO has no encoder). For ASO’s high-level policy and both ASO’s and QSO’s low-level policies, each dimension of the output is divided by the larger of its standard deviations over the dataset and over the sampled absolute subgoal options or actions. The variance is attributed exactly using two-way ANOVA, so that the attributions sum to 100. Mean and 95% bootstrap confidence interval (2000 resamples of the 64 grids) for one seed after 1M training steps.

Further discussion on the coherence of state abstractions. The high-level policy remains dependent on fast-moving dimensions (with 35% of the output variance being explained by fast-moving dimensions in antmaze-giant and 73% in cube-double). We attribute this to our method of Q-shaping: because the option is shaped to retain information about the required immediate action, if the target’s fast-moving dimensions form part of the representation that the option encodes, the high-level policy inherits this dependence. The following ablation confirms this hypothesis. We ablate using the environment success criterion (which only focusses on slow-moving dimensions) rather than the stricter success criterion to also label the reward in the dataset for the low-level decision process. Doing so makes the option encoder and hence also high-level policy substantially less dependent on fast-moving dimensions: percentage shares for the option encoder drop from 45% to 23% and from 79% to 8% respectively, while for the high-level policy, they drop from 35% to 28% and 73% to 25%. Because we see no loss in downstream performance (Figure 12) and although we use the stricter success criterion for labelling reward in the low-level decision process across all experiments, we hypothesise that the environment success criterion might be preferable at both levels.

Critic Gradient Flow Low Critic Only High Critic Only Both (Default)  
![](images/0963ed481cf785b6e4b0ad7506b8e12df0418a196a1f27daba01d4315d284468.jpg)

![](images/55fc9275993c48ec548eb7a5a441fff40af3cd50009b6471269836a30cbeaaf3.jpg)

Figure 8: High and low-level critic Q-function shaping ablation.  
![](images/7c61f77186e187d9a75ff8eb0eedf82795f0eb5600c22d7328a1365b30deea2d.jpg)

![](images/132d7975031afe9796545b777a493ea1b45f47077f74c74c4f8c3c424c696ff8.jpg)

![](images/52af9edc773d9e56896e8406ea141313c703dbdb85bcedab7c84fb41d216fb87.jpg)

![](images/c681c84fd3019f1bb50c348b5b94a0fd457f2caa8286840322dffd22c0cdac2c.jpg)

![](images/5ab54dd4f2da611a96efb76473b23b2439fccb5cc6f83ce51ca042eebf9ba5c8.jpg)

![](images/92ff9b4bd42923d4a2bf2ba5d7373b26f84e44e07eac7ab9e14afdcd97ff1228.jpg)

Effector Move A 0 Start Effector Move B Random k-step windows  
![](images/d1cda8a58caa2d551c80e7af018a390a905ba264dd73d5c57e067ae8f9d70130.jpg)  
Figure 9: PCA (top) of QSO’s action abstractions for gradient flow ablations for cube-double fitted on random trajectory sequences, showing different groups of three k-step trajectory sequences with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) for those groups in units of median distance between two random trajectory sequences. Both high and low-level gradient flows are required to group functionally equivalent behaviours while separating the behaviour groups.

How sensitive is QSO to option representation input? Although our reasoning in Section 4 motivates representing options as relative achievements, we test whether this design choice improves downstream performance. Figure 10 compares performance with absolute achievement $\phi _ { \omega } ( s _ { t + k } )$ with relative achievement encodings $\phi _ { \omega } ( s _ { t } , s _ { t + k } )$ . Relative encoding substantially improves performance in antmaze-giant, but has little effect in cube-double. We hypothesise this to be because relative encoding provides the largest benefit when many absolute state-goal pairs $( s , g _ { s } )$ can be collapsed into the same context-dependent achievement, which is the case for the locomotion task, but not necessarily for the manipulation one, where the above state-abstraction analysis suggests that the option retains much more fine-grained information for immediate control.

![](images/9bbea04f1dcf1c25c7423a7f40afacfb6b769d32d5df368d93dee485450d0941.jpg)  
Figure 10: $\phi _ { \omega } ( s _ { t + k } )$ rather than $\phi _ { \omega } ( s _ { t } , s _ { t + k } )$ option encoding ablation.

How sensitive is QSO to goal sampling? Both QSO and ASO are relatively robust to the proportion of random high-level goals (Figure 11). In both antmaze-giant and cube-double, all three sampling ratios converge to similar performance in both algorithms.

![](images/b08858c4dfef51393ba9464fb586fbe4d4c3990dbd0040b3a9e88e80b12a1722.jpg)  
Figure 11: Random goal sampling ablation for the high-level decision process for QSO (blue) and ASO (grey).

How sensitive is QSO to reward labelling? Environment reward labelling for the high-level decision process substantially improves performance for both QSO and ASO in cube-double, while on antmaze-giant the effect is small (Figure 12). For ASO, this gain comes from the high-level critic: ranking samples from the environment-labelled high-level policy using the strict-labelled high-level critic recovers the full performance (of 70% rather than 20%), while the reverse swap results in a corresponding drop in performance. We hypothesise that the environment reward labelling enables the high-level decision process to form better state abstractions: consistently, the percentage share of the output variance for slow-moving dimensions in the high-level state-value function increases by almost 10% (66% to 75%) for cube-double (Table 8). This is not the case for antmaze-giant, where even strict labelling yields a value dominated by the slow-moving dimensions. Note that we analyse percentage shares of output variance for the state-value function rather than the critic, which would add a confounding factor of how to sample the subgoal/option for the crossed state-goal pairs.

For QSO, reward labelling also shapes the option space itself, as the option encoder is trained through the critics. Downstream performance is most sensitive to using high-level reward labelling, which, like for ASO, we attribute to the high-level critic. Interestingly, however, low-level environment labelling makes the encoder markedly less sensitive to fast-moving dimensions, dropping from 79% to 8% for cube-double and 45% to 23% for antmaze-giant; the high-level policy becomes correspondingly more dependent on slow-moving dimensions, respectively increasing from 5% to 23% and 14% to 18%. Hence, while the option space remains similarly clean under both variants (Figures 15 and 16), using low-level environment reward labelling could be considered better, as it retains less information about fast-moving state dimensions without sacrificing downstream performance.

![](images/b7850e74ff961e10229d3927a070a03967a00f32b311170ba7ad34b304e23150.jpg)

![](images/2eaa50bf15055b37e5fbe02e4d1ec6fac99a1019b28717d10d9049e7e2a3785d.jpg)  
Figure 12: Strict versus environment dataset reward labelling ablation for QSO (blue) and ASO (grey).

![](images/f93a3ccec3f11755d7ea927931b68f75df20eec9774534bca3a3a3f72314b6e2.jpg)  
Figure 13: Change in performance when using the environment success criterion rather than the stricter success criterion used by prior work for reward labelling in the high-level decision process. Performance either does not change significantly, or improves for all algorithms.

![](images/e8bdbe011ed6eff7dc3112b8254c5a989de20d4f7d5f990d41714e48b92bca06.jpg)  
Option Resampling Interval (% of k = 25 Steps)  
Reward Labelling (High Decision Process, Low Decision Process) S: Strict, E: Environment E, S (Default) E，E

Figure 14: Performance with different option resampling frequencies for low-level dataset reward labelling ablations for QSO (blue) and ASO (grey).  
![](images/42272c25822a5d51399fa0e99088022d19b80790fd98f680f799f033fb42cf99.jpg)

![](images/62c2af541d1726ba5b47ffa9647caf8e033ef4e1e70cbe1e4d7500075fb10dac.jpg)

![](images/5be7ea0a5d00991bb3fc0ec98186e428eb7bd8f2a51084d7789290913c266457.jpg)

![](images/00e7705fee1449a54e110b4dfe5af53a1faab545f79704682e94e760c6b695ff.jpg)

![](images/c309aa731d7127e286f82a3fe3e7a12410b8d9986f41902759936595e19bd73e.jpg)

Right △ Up Left Down Dither Start Random k-step windows  
![](images/f7f5f54b78c655cb9d772bd5b903f72599972c9bd85cd27dd5abcbce7ca48b61.jpg)  
Figure 15: PCA (top) of QSO’s action abstractions for reward labelling ablations for antmaze-giant fitted on random trajectory sequences, showing different groups of three k-step trajectory sequences with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) for those groups in units of median distance between two random trajectory sequences. All reward labelling seems to encourage grouping functionally equivalent behaviours while separating the behaviour groups.

O Cube 0 Move A Cube 0 Move B

![](images/baa88e89c2f2556986a487ec3663cafd858ead3bef32b1d7d46a9742b9ccae5c.jpg)

![](images/b2951b887296bd76418bc314a96040c786abdc80ee83004bc63a3a18cf88f8be.jpg)

![](images/272c51ad91e3011d091d529cf1865fcf85d985f96c1db2d6b7acc7ac30dcf591.jpg)

![](images/550c0d522f09d9459cc337bc5350bfd4179c0d2786dc640d1740c4830b2f830d.jpg)

![](images/3716f89cd0041f2dc6b6d5b35cebaafb13ca9bd5273a5176ae8e89d29c7cacce.jpg)

![](images/4a9d673a4251bd325480195ddde24f69a02bef425a39428f0b6c1a849ef9f2d3.jpg)

![](images/ddacdfbe1b55379096b0b24e97d4e5575388f037862c7275542bcf14d29ab347.jpg)

![](images/9b4bd94a73954808500a063983c72024a9987db0736f739d6258e1774fba54e1.jpg)

![](images/2b5a2f938d3e03b7c345b618ec638a9d9fabab758f538b22dcca2cf11faa9998.jpg)

![](images/bd5eeafc790cac953800cfb0bd8d8fda51a7586a730a0c13b5a87ad9218cd157.jpg)

![](images/d055e7439be52655636e04c387e579d716678894147b75165bf46c77887a9a89.jpg)  
Figure 16: PCA (top) of QSO’s action abstractions for reward labelling ablations for cube-double fitted on random trajectory sequences, showing different groups of three k-step trajectory sequences with semantically similar behaviour (randomly selected from the validation dataset). L2 distances (bottom) for those groups in units of median distance between two random trajectory sequences. All reward labelling seems to encourage grouping functionally equivalent behaviours while separating the behaviour groups.

How sensitive is QSO to policy extraction? Both advantage weighting and RS lead to more consistent performance for ASO and QSO (Figure 17). As motivated, advantage weighting increases the high-level policy’s dependence on the goal: compared with BC, the goal’s share of output variance increases from 1% to 14% for QSO and 1% to 5% for ASO in antmaze-giant, and, respectively, from 0% to 1% and 1% to 3% in cube-double. Note that RS on top of this further increases goal-dependency (Table 9).

![](images/b6985e657bd64360c94fb502705be373f95ac4c33be111382789d7adb87a7728.jpg)  
Figure 17: High-level policy extraction ablation for QSO (blue) and ASO (grey).

<table><tr><td rowspan="2">Agent</td><td rowspan="2">Policy Extraction</td><td colspan="8">Flow Policy</td><td rowspan="2"></td><td colspan="6">Deployed Policy</td><td rowspan="2">(best of</td><td colspan="4">32)</td></tr><tr><td colspan="3">State</td><td colspan="4">Goal</td><td colspan="2">Noise</td><td colspan="2">State</td><td colspan="2"></td><td colspan="2">Goal</td><td colspan="2"></td><td colspan="2">Noise</td></tr><tr><td colspan="10">antmaze-giant-stitch-v0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QSO</td><td>BC</td><td></td><td>[69,</td><td>74]</td><td></td><td></td><td>1 [1, 2]</td><td></td><td>27</td><td>[25,</td><td>30]</td><td>60</td><td>[57,</td><td>62]</td><td></td><td>24</td><td>[23,</td><td>26]</td><td></td><td>16</td><td>[15,</td><td>17]</td></tr><tr><td></td><td>AWR</td><td></td><td>[61,</td><td>67]</td><td>14</td><td></td><td>[13,</td><td>15]</td><td>22</td><td>[21,</td><td>24]</td><td>54</td><td>[52,</td><td></td><td>57]</td><td>34</td><td>[32,</td><td></td><td>36]</td><td>12</td><td>[11,</td><td>12]</td></tr><tr><td>SHARSA</td><td>BC</td><td></td><td></td><td>[35, 41]</td><td></td><td></td><td>1 [1, 2]</td><td></td><td>61</td><td>[58,</td><td>64]</td><td>34</td><td></td><td>[31,</td><td>36]</td><td>15</td><td>[15,</td><td></td><td>16]</td><td>51</td><td>[49,</td><td>53]</td></tr><tr><td>ASO</td><td>AWR</td><td></td><td></td><td>[32, 38]</td><td></td><td>5</td><td>[5,</td><td>5]</td><td>60</td><td>[57,</td><td>62]</td><td>33</td><td>[31,</td><td></td><td>36]</td><td>19</td><td>[18,</td><td></td><td>20]</td><td>48</td><td>[46,</td><td>50]</td></tr><tr><td colspan="10">cube-double-play-v0</td><td colspan="8"></td><td></td><td></td><td></td><td></td></tr><tr><td>QSO</td><td>BC</td><td></td><td></td><td>[98, 98]</td><td></td><td>0 [0,</td><td>0]</td><td></td><td>2</td><td>[2,2]</td><td></td><td>97</td><td>[96,</td><td>97]</td><td></td><td>2</td><td>[2, 2]</td><td></td><td></td><td>2</td><td>[1, 2]</td><td></td></tr><tr><td></td><td>AWR</td><td>93</td><td>[92,</td><td>93]</td><td></td><td></td><td>1 [1, 1]</td><td></td><td>6</td><td>[6, 7]</td><td></td><td>88</td><td>[88,</td><td></td><td>89]</td><td>6</td><td>[6, 7]</td><td></td><td></td><td>5</td><td>[5,</td><td>6]</td></tr><tr><td>SHARSA</td><td>BC</td><td></td><td></td><td>[85, 86]</td><td></td><td>1</td><td>[1, 1]</td><td></td><td>14</td><td></td><td>[13, 15]</td><td>80</td><td>[79,</td><td></td><td>81]</td><td>9</td><td>[8, 9]</td><td></td><td></td><td>11</td><td>[10,</td><td>13]</td></tr><tr><td>ASO</td><td>AWR</td><td>84</td><td>[83,</td><td>85]</td><td>3</td><td>[3,</td><td>4]</td><td></td><td>13</td><td>[12, 13]</td><td></td><td>79</td><td>[78,</td><td></td><td>79]</td><td>12</td><td>[11,</td><td>12]</td><td></td><td>10</td><td>[9,</td><td>10]</td></tr></table>

Table 9: QSO, SHARSA and ASO high-level goal sensitivity with AWR and BC policy extraction. Percentage share of the variance of the high-level policy’s options explained by the state and goal. Inputs are built on grids that combine 16 random validation states with 16 random validation goals, each under 8 flow-noise samples. For SHARSA and ASO, each dimension of the output is divided by the larger of its standard deviations over the dataset and over the sampled absolute subgoal options. The variance is attributed exactly using two-way ANOVA, so that the attributions sum to 100 (“Goal” includes the state-goal interaction). The flow policy returns a single flow sample; the deployed policy returns the best of 32 flow samples under the high-level critic. Mean and 95% bootstrap confidence interval (2000 resamples of the 64 grids) for one seed after 1M training steps.

How sensitive is QSO to option dimension? QSO is largely insensitive to option dimension from 5 to 20, which all lead to similar downstream performance (Figure 18). We use a fixed default of 10.

![](images/12a3426867390602a33220c2b47ae67866e0d3fd1ba4133416367c6cf334de15.jpg)  
Figure 18: d option-space dimensionality ablation.

How sensitive is QSO to temporal horizon? Like ASO and prior work (Park et al., 2026; Ajay et al., 2020; Park et al., 2023; Li et al., 2026), QSO is sensitive to temporal abstraction horizon (Figure 19). For example, option horizons of 100 steps substantially degrades performance for both QSO and ASO in antmaze-giant; this is unsurprising, however, considering that the maximum number of steps in the antmaze-giant dataset is 200.

![](images/dcf849e7cdbf228992e2af08d66753bc5a592bf82429ba336bbce7e74fc68605.jpg)  
Figure 19: k temporal horizon ablation for QSO (blue) and ASO (grey).