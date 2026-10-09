# Real-Time Motion Planning with Dynamic Hazards: Classical vs. Learning-Based Methods

Eran Iceland<sup>1,∗</sup>, Alexander Tuisov<sup>2,∗</sup>, Oren Gal<sup>3</sup>, Ariel Barel<sup>4†</sup>, and Alfred M. Bruckstein<sup>4</sup>

<sup>1</sup>School of Engineering and Computer Science, The Hebrew University of Jerusalem, Jerusalem, Israel (E-mail: eran.iceland@gmail.com)

<sup>2</sup>Faculty of Data and Decision Science, Technion Israeli Institute of Technology, Haifa, Israel (E-mail: queldelan@gmail.com)

<sup>3</sup>Hatter Department of Marine Technologies, University of Haifa, Haifa, Israel

(E-mail: orengal@univ.haifa.ac.il)

<sup>4</sup>Faculty of Computer Science, Technion Israeli Institute of Technology, Haifa, Israel (E-mail: arielba@technion.ac.il; freddy@cs.technion.ac.il) <sup>∗</sup>These authors contributed equally to this work.

Abstract: We study real-time motion planning in dynamic hazard fields through a controlled comparison between classical planning and learning-based methods. Rather than introducing a new planner, we construct a unified benchmark in which rep resentative classical and learning-based methods face the same environments, motion constraints, information assumptions, and evaluation metrics. The test environment consists of planar domains populated with rotating sprinkler-like hazards that generate time-varying forbidden regions via sweeping angular sectors. Our results show a clear regime shift. In deterministic environments, classical planners achieve near-perfect success and higher-quality paths, though sometimes at the cost of substantial planning or replanning time. Under stochastic obstacle dynamics, however, online search becomes strongly budget-sensitive: low budgets lead to frequent failure, while high budgets improve success at the cost of latency and longer trajectories. PPO-based policies, trained under the same scenario distribution, consistently outperform in latency, success rate, and path quality in these stochastic regimes. Overall, the results indicate that uncertainty in obstacle evolution, more than partial observability, is the dominant factor determining which planning paradigm is practically effective for the problem at hand.

Keywords: Motion Planning; Dynamic Obstacles; Reinforcement Learning; Search Algorithms; Autonomous Navigation.<sup>[</sup>

This paper is scheduled to be presented at the 2027 International Symposium on Artificial Life and Robotics (AROB 2027).

## 1. INTRODUCTION

We consider motion planning in planar environments with dynamic hazards, a setting that has been extensively studied through both classical and learning-based approaches. Classical methods provide precise geometric or safety-driven guarantees under explicit modeling assumptions, while recent Reinforcement Learning (RL)-based planners offer adaptability and scalability in partially known or stochastic environments. Despite substantial progress on each side, relatively few works provide a controlled, side-by-side comparison between classical and RL-based methods under consistent modeling assumptions and evaluation criteria. In contrast to prior work that introduces new algorithms, our goal is to enable a controlled comparison under a shared problem formulation, isolating regime-dependent performance differences rather than architectural novelty, while providing practical guidance on when each paradigm should be preferred. To achieve that, we construct a unified testbed that is simple enough to admit principled classical treatment, yet rich enough to expose the limitations of each paradigm. We begin with the simplest regime, in which the full evolution of the environment is known in advance, and progressively introduce complexity along two axes: visibility (full vs. limited)

and obstacle dynamics (deterministic vs. stochastic). This yields four regimes, ranging from globally known, deterministic settings to partially observable and randomly evolving environments, enabling a systematic comparison under controlled conditions. This design choice intentionally prioritizes comparability over novelty in algorithm design, allowing us to attribute observed performance differences to problem structure rather than to formulation mismatches.

We formalize the problem setting as follows. Under full visibility, we consider a deterministic formulation via an XYT lifting [1, 2, 3]. The robot is modeled as a holonomic point moving at fixed speed and bounded turn rate in the plane while time advances monotonically, i.e. the agent must advance monotonically along T at unit rate. Dynamic obstacles make feasibility time-dependent, creating a tight coupling between spatial motion and temporal progression.

## 1.1. 2D/3D Static and Dynamic Planning

In 2D with static obstacles, classical path planning commonly relies on geometric constructs such as visibility graphs, Voronoi-based roadmaps, or search-based methods such as A\* over discretized grids [4, 5, 6, 7]. The setting is simpler because the environment is fixed; the main difficulty is a continuous-space combinatorial search. When obstacles are dynamic in 2D, the search naturally lifts to the spacetime domain XYT: feasibility must hold simultaneously in space and time, increasing dimensionality and introducing temporal constraints. Early work on real-time navigation in dynamic environments addressed these temporal constraints through analytic, safety-driven online planners. In particular, velocity-obstacle-based approaches explicitly avoid states of inevitable collision by selecting admissible motions over a carefully chosen time horizon [8]. Subsequent work introduced adaptive time-horizon selection, reducing conservatism while preserving online safety guarantees in crowded and time-varying scenes [9]. Classical tools in such settings include time-expanded graphs originating from dynamic-flow formulations [10], reachability-based methods grounded in Hamilton-Jacobi optimal control analysis [11], and reactive models such as velocity obstacles or the dynamic window approach [12, 13].

In 3D with static obstacles, geometric complexity rises: obstacle representations, distance queries, and feasibility checks become more expensive. Sampling-based planners such as PRM and RRT/RRT\* are widely used for highdimensional, non-convex environments [14, 15, 16, 17]. With dynamic obstacles in 3D, the problem extends to $X Y Z T$ , introducing non-convex and often non-linear planning challenges that require predictive obstacle-motion models and uncertainty-aware reasoning [18].

## 1.2. Reinforcement Learning for Obstacle Avoidance

Recent surveys of deep RL for navigation in dynamic and crowded environments highlight strong empirical promise, while emphasizing persistent challenges in generalization and real-world deployment [19, 20]. They also note that results are often difficult to compare across papers due to heterogeneous assumptions about sensing, obstacle dynamics, and safety constraints [21].

These observations motivate our controlled evaluation: we compare classical and RL-based planners under consistent assumptions and metrics to isolate regime-dependent performance differences.

We focus on the 2D dynamic-obstacle setting because it allows a consistent treatment across all regimes of interest while remaining computationally tractable. Under full visibility, the problem admits principled solutions that leverage the known evolution of the environment, including spacetime formulations and asymptotically optimal planning methods [16]. Under limited visibility, the same formulation naturally leads to incremental and online replanning strategies as information is revealed during execution [22, 23]. This duality makes the setting particularly well-suited for systematic evaluation of both classical and RL-based planners within a unified problem formulation.

## 1.3. Contribution Statement

Our contributions are primarily empirical and methodological. We introduce a unified testbed for motion planning in dynamic environments, enabling a controlled comparison between classical and RL-based methods under consistent modeling assumptions. The testbed spans four world models defined by two axes: visibility, either full or limited, and obstacle dynamics, either deterministic or stochastic. For these regimes, we implement classical baselines tailored to the available information and dynamics assumptions, together with a consistent RL formulation using a geometrically interpretable action space and compatible observation models. We then provide an empirical evaluation comparing the resulting planners in terms of success rate, path quality, computational cost, and robustness to uncertainty.

Our objective is not to model a specific robotic platform or deployment scenario, but rather to isolate the effect of information availability and obstacle uncertainty on planning performance within a controlled experimental framework.

## 2. PRELIMINARIES AND NOTATION

• Workspace: $W \subset R ^ { 2 }$ bounded.

• Dynamic obstacles: $O ( t ) ~ = ~ \{ O _ { i } ( t ) \} _ { i = 1 } ^ { m }$ , with each $O _ { i } ( t ) \subset W$ closed. Free space is denoted by $F ( t )$ . In what follows, the dynamic obstacles are instantiated as rotating circular sectors, which we refer to as sprinklers.

• State kinematics: position $( x _ { t } , y _ { t } )$ , heading $\theta _ { t } \in [ - \pi , \pi )$ time $T _ { t } \geq 0$

• Start and goal: $s = ( x _ { s } , y _ { s } ) , g = ( x _ { g } , y _ { g } )$ . The agent is considered to have reached the goal if $\| ( \bar { x } _ { t } , \bar { y } _ { t } ) - g \| < 1$ • Constant-speed constraint: $\| ( x _ { t + 1 } , y _ { t + 1 } ) - ( x _ { t } , y _ { t } ) \| = 1$ (discrete time). Time advances monotonically at unit rate.

• A discrete action space of 7 steering actions:

$$
a _ { t } = \Delta \theta _ { t } \in \{ - 4 5 ^ { \circ } , - 3 0 ^ { \circ } , - 1 5 ^ { \circ } , 0 ^ { \circ } , 1 5 ^ { \circ } , 3 0 ^ { \circ } , 4 5 ^ { \circ } \} .\tag{1}
$$

## 2.1. Problem Regimes

We consider four regimes obtained by combining two axes within the above formalization. Along the visibility axis,full visibility assumes that the obstacle set $\mathcal { O } ( t )$ , and hence the free space $\mathcal { F } ( t )$ , is known for all $t ,$ whereas limited visibility restricts the agent to partial, local observations of $\mathcal { F } ( t )$ at each time t around agent’s location at t. Along the dynamics axis, constant angular velocity obstacles correspond to deterministic trajectories $O _ { i } ( t )$ with fixed motion parameters, while stochastically varying angular velocity allows these trajectories to evolve randomly over time. This yields four planning regimes: (i) full visibility with constant angular velocity obstacles, where $\mathcal { O } ( t )$ is fully known and deterministic; (ii) limited visibility with constant angular velocity obstacles, where $\mathcal { O } ( t )$ is deterministic but only partially observed; (iii) full visibility with stochastically changing angular velocities, where $\mathcal { O } ( t )$ evolves randomly but is globally observable; and (iv) limited visibility with stochastic dynamics, where $\mathcal { O } ( t )$ is both partially observed and randomly evolving. In all cases, feasibility is defined with respect to $\mathcal { F } ( t )$ satisfying $\| ( x _ { t + 1 } , y _ { t + 1 } ) - ( x _ { t } , y _ { t } ) \| = 1$ . The visualization of the initial state can be seen in Figure 1, while the visualization of the corresponding trajectories can be seen in Figure 2.

## 3. CLASSICAL SEARCH AND PLANNING BASELINES

For each regime in Section 2.1, we implement a classical baseline matched to the regime’s information and dynamics assumptions. These baselines are representative methods for controlled comparison with the RL planner.

![](images/ad57970849b1660c9e0b14a414cd9849e39ba76007bcfd25930a7d0e145416e9.jpg)  
Fig. 1. Illustration of the environment used throughout the experiments: a rectangular playground with rotating “sprinklers” acting as dynamic hazards. Each sprinkler generates a circular sector that rotates, inducing timevarying forbidden regions. The agent starts on one side of the domain and seeks a shortest-time trajectory to the opposite side while remaining collision-free (dry), subject to constant-speed and bounded-turn-rate motion constraints.

## 3.1. State-Space Search Formulation

A state-space search problem is defined by an initial state, a successor function, a goal test, and a path-cost function [24]. We use states $( x , y , \theta , t )$ , where $( x , y )$ is the agent position, θ its heading, and t the timestep. Actions follow the motion constraints in Section 2; successors are generated by applying feasible actions, and the goal test checks whether the agent is within one step of the goal.

## 3.2. Deterministic Obstacle Angular Velocity

With constant obstacle angular velocity, the future hazard field is deterministic once the scenario is specified, so planning reduces to graph search over the discretized spacetime state space (x, y, θ, t). Exact continuous planning is difficult in dynamic domains: while static 2D shortest paths can be handled by visibility graphs or exact cell decompositions [5, 6], lifting to $X Y T$ breaks the usual vertex-only structure, and dynamic alternatives such as velocity-obstacle or dynamic-window methods still require online replanning [6, 12].

We use A\* [25] as the high-quality deterministic baseline: with an admissible heuristic, it is complete and optimal with respect to the induced discretized graph, though not necessarily the original continuous problem. We also evaluate Greedy Best-First Search (GBFS) with the same successor model and heuristic. GBFS is not optimal, but provides a faster satisficing baseline. For both A\* and GBFS, we use Euclidean distance to the goal as the heuristic. Together, $\mathbf { A } ^ { * }$ and GBFS expose the standard speed-optimality tradeoff under deterministic obstacle evolution.

## 3.2.1. Limited Visibility

Under limited visibility, the planner observes only part of the spacetime graph, which can be discovered during execution. We therefore use a D\*-like replanning framework [22]: plan on the currently observed map, execute forward, and repair the plan when newly observed sprinklers invalidate it. In this setting, $\mathbf { A } ^ { * }$ and GBFS are online replanning components rather than globally optimal planners. $\mathbf { A } ^ { * }$ remains optimal only for the currently known planning instance, and neither method is complete with respect to the full hidden environment: a feasible full-information trajectory may exist even if the online agent later reaches a deadlock.

## 3.3. Stochastic Obstacle Angular Velocity

With stochastic obstacle angular velocities, the future hazard field is no longer a single known trajectory. Deterministic shortest-path methods such as $\mathbf { A } ^ { * }$ are therefore not directly applicable without replacing the stochastic process by a fixed prediction, which fundamentally undermines robustness to uncertainty.

We use Upper Confidence bounds applied to Trees (UCT) [26] as a stochastic planning baseline. UCT performs sampled lookahead over possible future obstacle evolutions, balancing exploration and exploitation in the search tree. At each decision step, branches correspond to action sequences and sampled obstacle trajectories; leaf nodes are evaluated using Euclidean distance from the goal rather than full Monte Carlo rollouts. We use frame stacking [27], selecting actions every k = 5 timesteps to increase the effective lookahead horizon.

UCT is run as an anytime planner with a fixed computation budget per decision step. This budget controls the amount of stochastic lookahead, making UCT suitable for measuring the runtime-quality tradeoff of planning under uncertainty. While not exhaustive of all stochastic planners, UCT is a principled and widely used baseline for online decision-making in stochastic environments.

## 4. RL-BASED PLANNER

We model the planner as an episodic RL agent executed in closed loop. At each timestep, the agent observes the environment, selects an action, advances one step, and receives a reward. Policies are trained using PPO [28], chosen for its stability under sparse and delayed feedback in dynamic environments and its common usage in image-based control with discrete action spaces, using the action space defined in Equation (1). This enables direct comparison with classical planners. The dynamics follow unit-speed kinematics, i.e., the agent moves one unit per timestep in the direction of its heading.

Evaluation is performed on scenarios sampled from the same distribution as training, and policies are executed without further learning or adaptation. Full implementation details of the RL agent, including observations, reward design, network architecture and training procedure, are deferred to Appendix A.1.

## 5. EXPERIMENTS AND RESULTS

We evaluate classical and RL-based methods under all four world models described in Section 2.1. As a classical method we evaluate appropriate methods described in Section 3. For RL methods we trained policies described in Sec-

tion 4.

## 5.1. Experimental Setup

We ran 1,000 randomly generated inputs for each world model, and ran an RL-based and an appropriate classical algorithm on it. Scenarios were generated using the following protocol:

• The workspace is a $1 0 0 \times 5 0$ rectangle; agents start at $x =$ 0 with random y and fixed eastward heading, and goals are placed at $x = 1 0 0$ with random y.

• Sprinkler positions were sampled using a Greedy Max-Min rule [29], each time selecting the location farthest from the current set to improve worst-case coverage, i.e., $x = \arg \operatorname* { m a x } _ { p }$ min $_ { \cdot s \in S } d ( p , s )$ , ensuring non-degenerate and well-spread hazard configurations, with centers constrained to satisfy $2 2 \ < \ x \ < \ 9 0$ to avoid trivially unsolvable instances.

• Radii were sampled uniformly from a prescribed interval $R \in [ 1 0 , 2 0 ]$

• Angular velocities: in the constant angular velocity regime, fixed at $1 0 ^ { \circ } / \mathrm { s } ( \pi / \mathrm { \bar { . } }$ 18 rad/s), counterclockwise; in the stochastic regime, updated via a clipped and discretized random walk $\omega _ { t + 1 } = \omega _ { t } + \epsilon _ { t } ,$ , where $\epsilon _ { t } \sim \mathrm { N o r m a l } \Big ( 0 , \left( 0 . 5 ^ { \circ } \right) ^ { 2 } \Big )$ values are quantized to increments of $0 . 2 5 ^ { \circ }$ , and constrained to the interval $5 ^ { \circ } / \mathrm { s } \leq \omega _ { t } \leq 1 5 ^ { \circ } / \mathrm { s }$

• For the limited visibility setting, we used vision radius $V = 2 0$

• To ensure a fair comparison, all algorithms were evaluated on identical scenarios generated using a fixed random seed, with a common cutoff horizon of 400 steps.

• For hardware details, including our controlled setup ensuring identical execution conditions across all experiments, see Appendix A.2.

## 5.2. Quantitative Results

In the deterministic (constant angular velocity) regime presented in Table 1, classical planning methods clearly outperform RL-based approaches, particularly under full visibility. Both $\mathbf { A } ^ { * }$ and GBFS achieve perfect success rates, with $\mathbf { A } ^ { * }$ producing near-optimal trajectories and GBFS trading optimality for speed. Even under limited visibility, classical methods remain robust, exhibiting only moderate degradation (from 100% to approximately 96% success for $\mathbf { A } ^ { * } )$ which reflects the effectiveness of incremental replanning strategies. In contrast, RL methods achieve consistently lower success rates (around 80-85%) and do not provide a compensating advantage in path quality. These results indicate that when the environment is deterministic, even if only partially observed, classical planners can effectively exploit structural information and maintain high performance. When per-decision latency is the primary constraint, the RL policy provides a much faster online alternative, at the cost of lower success and path quality. A\* offline calculation time is heavy-tailed; see Appendix A.3.

The introduction of stochastic obstacle dynamics fundamentally alters this picture as can be seen in Table 2. Classical methods, represented by UCT with varying computational budgets, exhibit strong sensitivity to the available planning time: small timeouts lead to near-complete failure, while larger budgets improve success rates but result in significantly longer and more conservative trajectories. In contrast, RL methods achieve stable performance across regimes, matching or slightly exceeding the best UCT configurations in success rate while maintaining substantially shorter paths and low computational cost. Importantly, this degradation of classical methods is driven by uncertainty rather than limited visibility, as performance drops sharply even under full visibility once stochastic dynamics are introduced. This demonstrates that uncertainty in obstacle evolution is the dominant factor affecting planning difficulty, and that the trained RL policy consistently outperforms our online UCT baseline under stochastic obstacle evolution in latency, success rate, and path quality. This comparison is visualized in Fig. 3.

## 5.3. Ablation Study

## 5.3.1. Classical Planners.

For the classical stochastic planner, the main sensitivity was the UCT decision interval. Under full visibility with a 1s budget, frame stacking with $k = 5$ achieved 75.6% success, compared with 15.6% for $k = 1$ , indicating that effective lookahead depth is critical under stochastic dynamics. We also tested a risk-aware heuristic and alternative discretizations, but preliminary results were not competitive, so these variants were not run across the full suite.

## 5.3.2. RL Methods.

We evaluated the sensitivity of the learned planner to key training and reward parameters. Varying the discount factor within [0.95, 0.999] had no noticeable effect, suggesting limited sensitivity to long-horizon credit assignment in this setting. Similarly, adding simple shaping terms, including a per-step penalty of -0.01 or a rotation penalty of $- 0 . 0 1 | \Delta \theta |$ did not yield measurable improvements. In contrast, changing the balance between sparse and dense rewards had a clear effect: reducing the completion reward from 25 to 5 while increasing the progress reward from 0.005 to 0.1 shortened paths under full visibility with fixed sprinkler angular velocity from 135 to 127 steps, but reduced success under full visibility with deterministic angular velocity from 85% to 75%. This suggests that stronger dense rewards encourage locally efficient motion, whereas a larger terminal reward is important for reliable task completion.

## 6. LIMITATIONS

The comparison should be interpreted within the scope of our benchmark. The environment is a structured 2D domain with simplified sprinkler dynamics, discrete steering actions, constant-speed motion, and perfect execution. The hazards are non-adversarial and follow simple rotational dynamics; richer robotic settings may involve sensing noise, actuation error, dynamic agents with strategic behavior, nonholonomic dynamics, or more complex geometry. Moreover, the RL policies are evaluated in-distribution: test scenarios are sampled from the same scenario family used during training, so the results do not establish broad out-of-distribution

![](images/f048a861da16239d8bb0d7bf83efa2ee49a992e35b92baf8ead540ad8f9e3ec1.jpg)

![](images/95c58c09fde09eff1b5d23251d605d35a7fce4874a8358edb23a49f9543b440d.jpg)  
Fig. 2. Left: A\* (precomputed). Right: RL (real-time). Valid trajectories under full visibility with deterministic dynamics. Table 1. Performance under constant obstacle angular velocity based on 1,000 runs.

<table><tr><td>Vis.</td><td>Alg.</td><td>Succ. (%)</td><td>Path Length</td><td>Offline Time (s)</td><td>Max Replan Time (s)</td></tr><tr><td>Full</td><td>A*</td><td>100.0</td><td>114.38 (8.06)</td><td>38.56 (40.29)</td><td>NA</td></tr><tr><td>Full</td><td>GBFS</td><td>100.0</td><td>137.41 (21.59)</td><td>11.27 (3.43)</td><td>NA</td></tr><tr><td>Full</td><td>RL GPU</td><td>85.3</td><td>135.06 (21.69)</td><td>NA</td><td>0.0017 (0.0003)</td></tr><tr><td>Full</td><td>RL CPU</td><td>85.3</td><td>135.06 (21.69)</td><td>NA</td><td>0.0027 (0.0004)</td></tr><tr><td>Limited</td><td>A*</td><td>96.4</td><td>117.12 (10.50)</td><td>NA</td><td>5.7458 (7.8757)</td></tr><tr><td>Limited</td><td>GBFS</td><td>92.2</td><td>137.44 (22.50)</td><td>NA</td><td>2.2289 (1.3642)</td></tr><tr><td>Limited</td><td>RL GPU</td><td>81.3</td><td>136.15 (22.21)</td><td>NA</td><td>0.0016 (0.0002)</td></tr><tr><td>Limited</td><td>RL CPU</td><td>81.3</td><td>136.15 (22.21)</td><td>NA</td><td>0.0027 (0.0004)</td></tr></table>

Table 2. Performance under stochastic obstacle angular velocity based on 1,000 runs.
<table><tr><td>Vis.</td><td>Alg.</td><td>Succ. (%)</td><td>Path Length</td><td>Max Replan Time (s)</td></tr><tr><td>Full</td><td>UCT (0.002s)</td><td>3.9</td><td>136.92 (27.07)</td><td>0.0020 (0)</td></tr><tr><td>Full</td><td>UCT (0.01s)</td><td>14.0</td><td>121.26 (11.91)</td><td>0.0100 (0)</td></tr><tr><td>Full</td><td>UCT (0.1s)</td><td>52.1</td><td>182.57 (57.89)</td><td>0.1000 (0)</td></tr><tr><td>Full</td><td>UCT (1s)</td><td>75.6</td><td>198.75 (62.80)</td><td>1.0000 (0)</td></tr><tr><td>Full</td><td>RL GPU</td><td>76.3</td><td>152.26 (31.78)</td><td>0.0016 (0.0002)</td></tr><tr><td>Full</td><td>RL CPU</td><td>76.3</td><td>152.26 (31.78)</td><td>0.0027 (0.0002)</td></tr><tr><td>Limited</td><td>UCT (0.002s)</td><td>4.1</td><td>134.63 (19.27)</td><td>0.0020 (0)</td></tr><tr><td>Limited</td><td>UCT (0.01s)</td><td>13.8</td><td>121.02 (11.97)</td><td>0.0100 (0)</td></tr><tr><td>Limited</td><td>UCT (0.1s)</td><td>53.4</td><td>179.42 (53.79)</td><td>0.1000 (0)</td></tr><tr><td>Limited</td><td>UCT (1s)</td><td>74.8</td><td>199.57 (62.64)</td><td>1.0000 (0)</td></tr><tr><td>Limited</td><td>RL GPU</td><td>75.9</td><td>150.48 (32.76)</td><td>0.0017 (0.0003)</td></tr><tr><td>Limited</td><td>RL CPU</td><td>75.9</td><td>150.48 (32.76)</td><td>0.0027 (0.0003)</td></tr></table>

generalization.

Our evaluation also focuses on online performance. We report planning and inference latency during execution, but do not include the offline cost of PPO training. Conversely, the stochastic classical baseline is represented by UCT with fixed per-step budgets; other model-based approaches, such as sampling-based MPC, belief-space planning, or chanceconstrained planning, may yield different tradeoffs. Finally, our metrics emphasize success, path length, and latency. Future work should include richer safety criteria, such as nearmiss rates, accumulated risk exposure, and robustness under shifted obstacle distributions.

## 7. CONCLUSION

We presented a controlled comparison between classical and learning-based approaches for real-time motion planning in dynamic hazard fields under a shared formulation. Across four regimes, the results consistently show that deterministic structure favors classical planning, while stochasticity shifts the advantage toward learning-based policies.

![](images/2331059d08ea8e81b98bffbfc785dad181e2bef355d5c5a8dedf4bdb48c0502a.jpg)  
(a) Deterministic

![](images/060128aa65d6ac321ea77f22e646e4eda188ff9f851cf9f5e3b0eb5f306b2dec.jpg)  
(b) Stochastic (budgeted UCT)  
Fig. 3. Success rate versus maximal replan time under limited visibility for (a) deterministic and (b) stochastic dynamics. Marker size represents path length; the UCT curve connects different per-step planning budgets.

In the deterministic regime, classical planners achieve near-perfect success and better path quality, with a clear speed-optimality tradeoff between A\* and GBFS. RL offers dramatically lower latency but does not match classical success rates or trajectory quality in this regime. In the stochastic regime, classical performance becomes strongly budgetdependent: increased computation improves success but incurs steep costs in time and produces longer, more conservative paths. RL, in contrast, maintains stable success across visibility settings while preserving very low per-step latency, yielding a favorable success-quality-latency tradeoff under uncertainty.

Overall, our results support a single dominant takeaway: uncertainty in obstacle dynamics, rather than partial observability, is the key factor that changes planning difficulty and determines which paradigm is practically effective in real time.

## REFERENCES

[1] E. Adin and A. M. Bruckstein, Navigation in a Dynamic Environment. Technion-Israel Institute of Technology, Center for Intelligent Systems, 1991.

[2] R. Kimmel, N. Kiryati, and A. M. Bruckstein, “Multivalued distance maps for motion planning on surfaces with moving obstacles,” IEEE Transactions on Robotics and Automation, vol. 14, no. 3, pp. 427–436, 1998.

[3] K. Fujimura, “Time-minimum routes in timedependent networks,” IEEE Transactions on Robotics and Automation, vol. 11, no. 3, pp. 343–351, 1995.

[4] T. Lozano-Perez and M. A. Wesley, “An algorithm for ´ planning collision-free paths among polyhedral obstacles,” Communications of the ACM, vol. 22, no. 10, pp. 560–570, 1979.

[5] J.-C. Latombe, Robot Motion Planning. Boston, MA: Kluwer Academic Publishers, 1991.

[6] J. S. B. Mitchell, “Geometric shortest paths and network optimization,” in Handbook of Computational Geometry, pp. 633–701, Elsevier, 2000.

[7] H. Choset, K. M. Lynch, S. Hutchinson, G. Kantor,

W. Burgard, L. E. Kavraki, and S. Thrun, Principles of Robot Motion: Theory, Algorithms, and Implementations. MIT Press, 2005.

[8] O. Gal, Z. Shiller, and E. Rimon, “Efficient and safe on-line motion planning in dynamic environments,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 88–93, 2009.

[9] Z. Shiller, O. Gal, and A. Raz, “Adaptive time horizon for on-line avoidance in dynamic environments,” in Proceedings ofthe IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 3539– 3544, 2011.

[10] L. R. Ford, Jr. and D. R. Fulkerson, “Constructing maximal dynamic flows from static flows,” Operations Research, vol. 6, no. 3, pp. 419–433, 1958.

[11] I. M. Mitchell, A. M. Bayen, and C. J. Tomlin, “A time-dependent Hamilton-Jacobi formulation of reachable sets for continuous dynamic games,” IEEE Transactions on Automatic Control, vol. 50, no. 7, pp. 947– 957, 2005.

[12] P. Fiorini and Z. Shiller, “Motion planning in dynamic environments using velocity obstacles,” The International Journal of Robotics Research, vol. 17, no. 7, pp. 760–772, 1998.

[13] D. Fox, W. Burgard, and S. Thrun, “The dynamic window approach to collision avoidance,” IEEE Robotics & Automation Magazine, vol. 4, no. 1, pp. 23–33, 1997.

[14] L. E. Kavraki, P. Svestka, J.-C. Latombe, and M. H.<sup>ˇ</sup> Overmars, “Probabilistic roadmaps for path planning in high-dimensional configuration spaces,” IEEE Transactions on Robotics and Automation, vol. 12, no. 4, pp. 566–580, 1996.

[15] S. M. LaValle and J. J. Kuffner, Jr., “Randomized kinodynamic planning,” The International Journal of Robotics Research, vol. 20, no. 5, pp. 378–400, 2001.

[16] S. Karaman and E. Frazzoli, “Sampling-based algorithms for optimal motion planning,” The International Journal of Robotics Research, vol. 30, no. 7, pp. 846– 894, 2011.

[17] S. M. LaValle, Planning Algorithms. Cambridge University Press, 2006.

[18] S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics. MIT Press, 2005.

[19] Y. Zhu, W. Z. W. Hasan, H. R. H. Ramli, N. M. H. Norsahperi, M. S. M. Kassim, and Y. Yao, “Deep reinforcement learning of mobile robot navigation in dynamic environment: A review,” Sensors, vol. 25, no. 11, p. 3394, 2025.

[20] H. Le, S. Saeedvand, and C.-C. Hsu, “A comprehensive review of mobile robot navigation using deep reinforcement learning algorithms in crowded environments,” Journal of Intelligent & Robotic Systems, vol. 110, no. 4, p. 158, 2024.

[21] N. AbuJabal, M. Baziyad, R. Fareh, B. Brahmi, T. Rabie, and M. Bettayeb, “A comprehensive study of recent path-planning techniques in dynamic environments for autonomous robots,” Sensors, vol. 24, no. 24, p. 8089, 2024.

[22] A. Stentz, “The focussed D\* algorithm for real-time replanning,” in Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI), pp. 1652–1659, 1995.

[23] S. Koenig and M. Likhachev, “D\* Lite,” in Proceedings of the AAAI Conference on Artificial Intelligence, pp. 476–483, 2002.

[24] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, Global Edition. Pearson, 3rd ed., 2016.

[25] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968.

[26] L. Kocsis and C. Szepesvari, “Bandit-based Monte´ Carlo planning,” in European Conference on Machine Learning, pp. 282–293, Springer, 2006.

[27] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[28] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[29] T. F. Gonzalez, “Clustering to minimize the maximum intercluster distance,” Theoretical Computer Science, vol. 38, pp. 293–306, 1985.

## APPENDIX

## A. SUPPLEMENTARY MATERIAL

## A.1. Extended RL Formulation and Training Details

## A.1.1. Safety & Observations.

Sprinklers are angular sectors; non-flight zones (NFZs) are defined as a uniform inflation of each sector with margin 1. This encourages safety-aware behavior by penalizing proximity to hazards rather than only collisions.

The agent observes two ego-centered RGB images of size 3 × 101 × 101 at scales 0.5 and 2. The multi-scale representation enables both local precision and global context.

Agent: 3 × 3 (blue 0.5); target: 3 × 3 (green 1); Outof-Bounds (OOB): blue 1; sprinklers/NFZ: red channel. For constant angular velocity: sprinkler = 1, NFZ = 0.5; for variable angular velocity: values scale from 1/3 to 1, NFZ is half.

Under limited visibility, only visible sprinklers are rendered; the agent is memoryless.

## A.1.2. Reward & Curriculum.

Rewards include goal completion (25), NFZ penalty (- 0.25), progress shaping (0.005), and penalties for OOB and collisions (-2 each). OOB and collisions terminate the episode.

We employ two curricula: (i) termination - OOB initially does not terminate to avoid degenerate policies, then later it does; (ii) complexity - the number of sprinklers increases from 0 to 9-12.

## A.1.3. Architecture & Training.

Each image is processed by a CNN [27], producing 256 features; concatenation yields 512, fed to separate actor and critic MLPs with layers [128, 64].

We use 32 parallel environments and PPO with lr = 5 · $1 0 ^ { - 5 } , n _ { \mathrm { s t e p s } } = 1 2 8$ , batch = 2, 048, entropy = 0.01, clip = $0 . 2 , \lambda = 0 . 9 8 ,$ , epochs = 4, and discount γ = 0.995.

Training consists of 120M base steps (full visibility, constant angular velocity), followed by 80M steps for each regime: full visibility with variable angular velocity, limited visibility with constant angular velocity, and limited visibility with variable angular velocity.

The average wall-clock training throughput was approximately 550 FPS, corresponding to about 2 million environment steps per hour.

## A.2. Experimental Hardware Setup

All experiments were conducted on an Azure virtual machine equipped with 8 vCPUs (AMD EPYC 7V12 processor, 2.45 GHz) and 54 GB RAM, under a Microsoft Azure hypervisor. The system also included an NVIDIA Tesla T4 GPU with 16 GB memory. The software environment consisted of Ubuntu 20.04.6 LTS with Python 3.10.13.

The classical planning algorithms (A\*, GBFS, and UCT) were executed on the CPU only, each restricted to a single CPU core, without GPU acceleration.

The RL approach was evaluated under two configurations: (1) CPU-only execution, using a single CPU core, and (2) GPU-accelerated execution, using a single NVIDIA Tesla T4 GPU.

All experiments were conducted on an otherwise idle virtual machine to minimize variability.

## A.3. A\* Offline-Time Histogram

Table 1 reports A\* offline planning time as mean and standard deviation over 1,000 deterministic full-visibility scenarios. The large standard deviation is explained by a strongly right-skewed, heavy-tailed runtime distribution: most instances are solved quickly, while a small fraction of hard instances require substantially longer search time. For the detailed breakdown, see Fig. 4.

![](images/16335d553baaae8ca6d4f157d6a715e6b84ab309063574c96f9214f61a68cd14.jpg)  
Fig. 4. Histogram of $\mathbf { A } ^ { * }$ offline computation time in seconds over the 1,000 deterministic full-visibility scenarios. The distribution is strongly right-skewed: most runs finish within tens of seconds, while rare hard instances produce a long tail, inflating the standard deviation reported in Table 1.