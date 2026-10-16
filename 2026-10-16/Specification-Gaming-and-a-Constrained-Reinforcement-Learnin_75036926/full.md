# Specification Gaming and a Constrained Reinforcement Learning Mitigation in a Cleaning Gridworld

Preprint

Hild<sub>a</sub> Ad<sub>wu</sub>bi O<sub>se</sub>i Department of Industrial Engineering Kwame Nkrumah University of Science and Technology Kumasi, Ghana hildaosei109@gmail.com

## A<sub>bstract</sub>

Reinforcement learning agents optimize exactly the reward signal they are given, which can diverge sharply from an intended true objective when that signal is only a proxy for it; this failure mode is known as specification gaming. Existing evidence for this failure mode is largely observational, with comparatively few demonstrations pairing a controlled, deliberately constructed proxy/true gap with a mitigation that carries a verifiable guarantee rather than only an empirical improvement. We construct a minimal gridworld in which a cleaning-robot agent can either genuinely remove dirt or conceal it from a cheap sensor, and show that a standard deep Q-network trained only on the sensor’s proxy reward converges to a policy that games it completely, reaching near-maximal proxy reward while never achieving the true objective. We then introduce a constrained reinforcement learning fix, penalizing the specific behaviors through which the agent exploits the gap, and show that it eliminates the gaming behavior from the final policy while substantially recovering true-objective performance. We further derive and empirically verify an exact accounting identity relating the proxy reward, true reward, and our cost signal for this environment, showing that the cost signal measures the proxy/true gap exactly rather than merely tracking it, and yields a deterministic guarantee under a hard-constrained variant of the method.

Keywords specification gaming · reward hacking · reinforcement learning · AI alignment · constrained optimization · outer alignment

## 1 Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

A reinforcement learning agent optimizes exactly the reward signal it is given, not the intention behind it. When that signal is a proxy standing in for a true objective that is harder to specify directly – a sensor reading in place of a physical state, a classifier score in place of an outcome – optimization pressure can push the agent toward the proxy/true gap rather than closing it. This failure mode, specification gaming, has been documented extensively across both simulated and real-deployed systems [Krakovna et al., 2020], and recent work has begun to formalize what distinguishes it from ordinary reward misspecification [Skalse et al., 2022].

Much of the existing evidence for specification gaming is observational: a catalogue of cases in which an agent was found, after the fact, to have exploited a reward function in an unintended way. Comparatively less work pairs a controlled, deliberately constructed proxy/true gap with a mitigation that carries a stated, verifiable guarantee rather than only an empirical improvement. This gap matters for deployment: knowing that a fix tends to work is weaker than knowing precisely what it guarantees and under what conditions.

In this work, we construct a minimal gridworld, CleaningGridWorld, with a genuine and exploitable gap between a proxy reward (what a cheap sensor reports) and a true objective (what was actually accomplished), and show that a standard deep Q-network trained on the proxy alone discovers two independent ways to exploit this gap without being directed toward either. We then introduce a constrained reinforcement learning fix, built around a cost signal targeting the discovered exploits directly, and show that it eliminates the gaming behavior from the agent’s final policy while substantially recovering true-objective performance. Critically, we derive an exact accounting identity relating the proxy reward, the true reward, and our cost signal, and verify that it holds exactly against the trained agents, giving the fix a provable guarantee under a hard-constrained variant, not merely an empirical trend.

Concretely, this paper contributes:

• CleaningGridWorld, a gridworld with a constructed, interpretable proxy/true reward gap, and an empirical demonstration that a proxy-only DQN discovers two distinct gaming mechanisms unaided.

• A Lagrangian-penalized constrained RL fix, with a cost signal targeting the discovered exploits, that eliminates gaming from the final greedy policy while recovering substantial true-objective performance.

• An exact accounting identity relating proxy reward, true reward, and cost, proved for this environment and empirically verified against both trained agents – yielding a deterministic lower bound on true-objective performance under a hard-constrained version of the method.

The proxy/true gap studied in this paper is not unique to toy domains. AI-driven biological design and screening tools, such as generative sequence design scored against an automated toxicity or pathogenicity classifier, face a structurally identical gap, where "passes the screen" and "is actually safe" can diverge under optimization pressure. We return to this connection as a motivation for future work in Section 6, without claiming biosecurity-specific experimental results in this study.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 S<sub>pec</sub>ifi<sub>ca</sub>ti<sub>o</sub>n G<sub>a</sub>min<sub>g a</sub>nd R<sub>ewa</sub>rd H<sub>ac</sub>kin<sub>g</sub>

Specification gaming – an agent exploiting a literal reward specification in a way that diverges from the designer’s intent – has been documented extensively across both simulated and deployed systems [Krakovna et al., 2020]. These cases are largely catalogued observationally: a system is found, after deployment or training, to have discovered an unintended shortcut. Skalse et al. [2022] take a step toward formal rigor, characterizing when a reward function admits a hacking policy relative to an intended true objective, distinguishing reward hacking from ordinary reward misspecification more broadly. We adopt this proxy/true framing directly, but construct an environment where the gap is specified by design rather than discovered after the fact, which lets us pair the demonstration with a mitigation whose efect on that specific, known gap can be measured exactly.

## 2<sub>.</sub>2 R<sub>ewa</sub>rd Sh<sub>ap</sub>in<sub>g</sub> <sub>a</sub>nd P<sub>o</sub>li<sub>cy</sub> In<sub>va</sub>ri<sub>a</sub>n<sub>ce</sub>

Dense, informative reward signals are often necessary for an RL agent to discover a sparse or delayed objective within a practical training budget. Ng et al. [1999] show that reward shaping of the form $F \dot { ( } s , a , s ^ { \prime } ) = \dot { \gamma } \phi ( s ^ { \prime } ) \dot { - } \phi ( s )$ , for any potential function ϕ, leaves the optimal policy of the underlying MDP unchanged – shaping accelerates discovery of an optimum without altering what that optimum is. This guarantee is what allows us to introduce dense positional shaping in Section 3.3 without that shaping itself becoming a second, uncontrolled source of proxy/true divergence.

## 2<sub>.</sub>3 C<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g</sub>

Constrained Markov decision processes [Altman, 1999] formalize policy optimization subject to a secondary cost objective remaining below a threshold, typically enforced via a Lagrangian relaxation. Achiam et al. [2017] extend this to deep RL with a trust-region method with per-update constraint satisfaction guarantees. The present work uses the simpler, fixed-multiplier end of this family (Section 3.4): rather than tuning the Lagrange multiplier via dual ascent, we fix it and instead derive a problem-specific identity (Section 4) that lets a hard cap on the cost signal translate directly into a deterministic bound on true-objective performance. We treat an implementation of the dual-ascent or hard-constrained variant, in the style of Achiam et al. [2017], as natural future work rather than a contribution of the present paper.

## 2<sub>.</sub>4 P<sub>os</sub>iti<sub>on</sub>i<sub>ng</sub>

Our contribution sits at the intersection of these threads: we construct a specification-gaming environment in the spirit of Krakovna et al. [2020] and Skalse et al. [2022], but with a controlled, by-design proxy/true gap rather than one discovered post hoc; we use the shaping guarantee of Ng et al. [1999] to make the gap tractable for a standard deep Q-network [Mnih et al., 2015]; and we apply a Lagrangian-penalized constraint in the tradition of Altman [1999], strengthened in this specific setting by an exact accounting identity that turns an empirical improvement into a provable guarantee.

## 3 M<sub>e</sub>th<sub>o</sub>d

## 3<sub>.</sub>1 En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt<sub>:</sub> Clea<sub>n</sub>i<sub>ng</sub>GridW<sub>o</sub>rld

CleaningGridWorld is a $5 \times 5$ gridworld containing four dirt cells and one fixed disposal bin. Each cell is in one of four states: empty floor, dirt, dirt covered by a rug, or the disposal bin. The agent occupies a single cell and may additionally be carrying one unit of vacuumed dirt; it cannot carry more than one unit at a time. The action space is discrete with six actions: four cardinal movements, vacuum, and cover. vacuum picks up dirt from the agent’s current cell if the agent is not already carrying, or deposits carried dirt if the agent is standing on the bin. cover places a rug over dirt in the current cell.

This design yields two structurally distinct routes to the same sensor-visible outcome. A proxy reward, modeling a cheap automated sensor, counts cells that do not read as dirty:

$$
R _ { t } ^ { \mathrm { { p r o x y } } } = | \{ ( r , c ) : \mathrm { { g r i d } } _ { t } ( r , c ) \in \{ \mathrm { { e m p t y } , \mathrm { { c o v e r e d } } \} } \} |\tag{1}
$$

A true reward counts only genuine disposal events, independent of what the sensor would report:

$$
R _ { t } ^ { \mathrm { t r u e } } = \mathcal { k } [ \mathrm { a  d i s p o s a l o c c u r r e d a t } t ]\tag{2}
$$

Both covering and picking up dirt raise $R ^ { \mathrm { p r o x y } } ~ ( \mathrm { E q . ~ 1 } )$ , since both remove the dirt state from a cell; only completed delivery to the bin raises $\breve { R } ^ { \mathrm { t r u e } } \left( \mathrm { E q . ~ } 2 \right)$ . This asymmetry is the environment’s constructed gap: an agent that observes only $R ^ { \mathrm { { \dot { p } } r o x y } }$ has no reward-level incentive to prefer the costlier delivery route over the cheaper concealment route, even though only the former satisfies the intended objective. We use a single fixed layout – dirt, bin, and agent start positions sampled once and reused across all episodes – so that training is a navigation problem on a known map rather than a layout-generalization problem; we return to this choice in Section 6.1.

## 3<sub>.</sub>2 A<sub>ge</sub>nt

The agent is a deep Q-network [Mnih et al., 2015], which is a feedforward network with two hidden layers of 128 units each and ReLU activations, mapping an encoded state to one Q-value per action. The state is encoded as the concatenation of a one-hot grid representation $( 5 \times 5 \times 4 = 1 0 0$ dimensions), a one-hot agent position $( 5 \times 5 = 2 5$ dimensions), and a binary carrying flag (1 dimension), yielding a 126-dimensional input. This architecture is intentionally minimal: the state and action spaces are small enough that a compact MLP with experience replay and a target network is suficient to isolate the proxy/true gaming phenomenon without confounding it with architecture-driven learning dificulty.

## 3<sub>.</sub>3 P<sub>o</sub>t<sub>e</sub>nti<sub>a</sub>l<sub>-</sub>B<sub>ase</sub>d R<sub>ewa</sub>rd Sh<sub>ap</sub>in<sub>g</sub>

Only 4 of the environment’s 25 cells are ever reward-relevant, and ϵ-greedy exploration under this sparsity provides too little gradient signal for standard Q-learning to discover either route to reward within a practical training budget. We address this with potential-based reward shaping [Ng et al., 1999]:

$$
F ( s _ { t } , a _ { t } , s _ { t + 1 } ) = \gamma \phi ( s _ { t + 1 } ) - \phi ( s _ { t } )\tag{3}
$$

where $\phi ( s )$ is the negative Manhattan distance to the agent’s current objective: the nearest dirt cell if not carrying, or the bin followed by the nearest remaining dirt cell if carrying. ϕ is defined so that it varies continuously across the pickup-to-delivery transition, avoiding an artificial potential discontinuity at the exact moment the agent completes a delivery. Ng et al. [1999] show that any reward of the form in Eq. 3 leaves the optimal policy of the underlying MDP unchanged for any choice of ϕ; shaping therefore accelerates discovery of an optimum without the possibility of shaping itself introducing a new, unintended proxy/true divergence.

## 3<sub>.</sub>4 C<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g</sub> Fi<sub>x</sub>

We define a per-step cost signal targeting the two gaming mechanisms through which an unconstrained agent can raise $R ^ { \mathrm { p r o x y } }$ without raising $R ^ { \mathrm { t r u e } }$ – concealing dirt, and picking dirt up without ever delivering it:

$$
\mathrm { c o s t } _ { t } = { \mathbb { k } } [ \mathrm { a g e n t ~ c o v e r s ~ d i r t ~ a t } t ] + { \mathbb { k } } [ t = T ~ \mathrm { a n d ~ a g e n t ~ i s ~ s t i l l ~ c a r r y i n g ~ u n d e l i v e r e d ~ d i r t } ]\tag{4}
$$

The second term fires only at the final timestep T of an episode, and only if the agent is carrying dirt it never delivered. The training reward used for the Q-learning update applies cost as a fixed-multiplier Lagrangian penalty on top of the shaped proxy signal:

$$
r _ { t } ^ { \mathrm { t r a i n } } = \Delta R _ { t } ^ { \mathrm { p r o x y } } + F ( s _ { t } , a _ { t } , s _ { t + 1 } ) - \lambda \cdot \mathrm { c o s t } _ { t }\tag{5}
$$

where $\Delta R _ { t } ^ { \mathrm { p r o x y } } = R _ { t } ^ { \mathrm { p r o x y } } - R _ { t - 1 } ^ { \mathrm { p r o x y } }$ and λ is fixed for the duration of training. Critically, the agent is never given $R ^ { \mathrm { t r u e } }$ as a training signal; it learns only from Eq. 5, mirroring a deployment setting in which the true objective is precisely what is unavailable for direct supervision.

The concealment exploit is a single-action behavior that is trivially reachable by random exploration; the honest delivery route requires a multi-step chain (navigate to dirt, pick up, navigate to bin, deposit) that is comparatively unlikely to be discovered before ϵ decays. To address this exploration asymmetry, we generate a fixed set of honest-policy demonstration trajectories once, before training begins, and store them in a bufer that is never subsequently written to and therefore can never be evicted by the replay bufer’s ring-bufer overwrite behavior. Each training batch is drawn as a mixture of ordinary experience and demonstration transitions at a fixed demonstration fraction. This separation is necessary: in an earlier version of this method, demonstrations were seeded into the same bufer used for ordinary experience and were silently overwritten once the total transitions exceeded the bufer’s capacity, after which the agent’s true-objective performance reverted toward the unconstrained baseline. Isolating demonstrations in a protected bufer removes this failure mode entirely.

## 3.5 H<sub>yp</sub>er<sub>p</sub>arameters

Table 1 lists the full configuration used for both the baseline and constrained agents, except where noted.

Table 1: Hyperparameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Grid size</td><td> $5 \times 5$ </td></tr><tr><td>Dirt cells</td><td>4</td></tr><tr><td>Episode length</td><td>40 steps</td></tr><tr><td>Episodes (baseline)</td><td>500</td></tr><tr><td>Episodes (constrained)</td><td>800</td></tr><tr><td>Hidden layer size</td><td>128 (two layers)</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 } ( \mathrm { A d a m } )$ </td></tr><tr><td>Discount factor γ</td><td>0.95</td></tr><tr><td>€ schedule</td><td> $1 . 0  0 . 0 5$  , linear over 300 episodes</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Target network update frequency</td><td>every 10 episodes</td></tr><tr><td>Replay buffer capacity</td><td>10,000</td></tr><tr><td>Shaping scale</td><td>0.2</td></tr><tr><td>Lagrangian multiplier λ</td><td>2.0</td></tr><tr><td>Demonstration episodes</td><td>20</td></tr><tr><td>Demonstration fraction per batch</td><td>0.5</td></tr></table>

## 4 Th<sub>eore</sub>ti<sub>ca</sub>l R<sub>esu</sub>lt

Proposition 1 (Exact accounting identity). Let $\begin{array} { r } { p r o x y _ { t o t a l } = \sum _ { t = 1 } ^ { T } \Delta R _ { t } ^ { p r o x y } , t r u e _ { t o t a l } = \sum _ { t = 1 } ^ { T } R _ { t } ^ { t r u e } } \end{array}$ <sub>, an</sub>d $c o s t _ { t o t a l } =$ $\scriptstyle \sum _ { t = 1 } ^ { T } c o s t _ { t } ,$ as defined in Eqs. 1–4. Then for any policy and any episode of length T in CleaningGridWorld,

$$
p r o x y _ { t o t a l } = t r u e _ { t o t a l } + c o s t _ { t o t a l } .\tag{6}
$$

Proof sketch. Each unit of dirt is resolved, by the end of the episode, in exactly one of three mutually exclusive ways:

• Covered. Contributes +1 to $\mathrm { p r o x y } _ { \mathrm { t o t a l } }$ (Eq. 1: the cell transitions to the covered state), +0 to $\mathrm { \ t r u e } _ { \mathrm { t o t a l } }$ , and +1 to $\mathbf { c o s t } _ { \mathrm { t o t a l } }$ by the first term of Eq. 4.

• Delivered. Contributes +1 to $\mathrm { p r o x y } _ { \mathrm { t o t a l } }$ at the moment of pickup (the cell empties), +1 to $\mathrm { \ t r u e } _ { \mathrm { t o t a l } }$ at the moment of delivery, and +0 to cost<sub>total</sub>.

• Abandoned. Picked up but not delivered by episode end. Contributes +1 to $\mathrm { p r o x y } _ { \mathrm { t o t a l } }$ at pickup, +0 to $\mathrm { \ t r u e } _ { \mathrm { t o t a l } }$ and +1 to $\mathbf { c o s t } _ { \mathrm { t o t a l } }$ by the second term of Eq. 4.

Since the agent carries at most one unit of dirt at a time, at most one unit is unresolved at episode end, so these three cases partition the dirt cells without double-counting. In each case the per-cell proxy contribution equals the sum of that cell’s true and cost contributions. Summing over all dirt cells gives Eq. 6. □

Proposition 1 shows that $\mathrm { c o s t } _ { \mathrm { t o t a l } } ,$ as defined in Eq. 4, is not merely correlated with the proxy/true gap; it measures the gap exactly. Penalizing cost in Eq. 5 is equivalent, in the exact accounting sense of Eq. 6, to penalizing $\mathrm { p r o x y } _ { \mathrm { t o t a l } } - \mathrm { t r u e } _ { \mathrm { t o t a l } }$ directly, despite the agent never observing $\mathrm { \ t r u e } _ { \mathrm { t o t a l } }$ during training.

Corollary 1 (Deterministic bound under a hard cost constraint). Iftraining enforces $c o s t _ { t o t a l } \le K$ for a fixed $K - f o r$ instance via early episode termination or action-masking once K gaming events have occurred – then

$$
t r u e _ { t o t a l } \ge p r o x y _ { t o t a l } - K\tag{7}
$$

holds with certainty, for every episode, rather than in expectation.

Corollary 1 follows immediately from rearranging Eq. 6. It provides the hard-constrained variant of our method with a guarantee of a diferent character than the fixed-λ Lagrangian penalty used in the present experiments (Eq. 5): rather than encouraging low cost on average, a hard cap on $\mathbf { c o s t } _ { \mathrm { t o t a l } }$ translates, via Proposition 1, into a deterministic floor on true-objective performance relative to whatever proxy score the policy achieves. We verify Proposition 1 empirically against both trained agents in Section 5.4, confirming that the identity holds exactly on the implemented system and not only in the idealized derivation above. The present experiments use the fixed-λ relaxation rather than the hard-constrained variant of Corollary 1; we discuss this as a direction for future work in Section 7.

## 5 Ex<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> <sub>a</sub>nd R<sub>esu</sub>lt<sub>s</sub>

## 5<sub>.</sub>1 S<sub>a</sub>nit<sub>y</sub> Ch<sub>ec</sub>k<sub>:</sub> C<sub>o</sub>nfirmin<sub>g</sub> th<sub>e</sub> Pr<sub>oxy</sub>/Tr<sub>ue</sub> G<sub>ap</sub>

Before training any agent, we verified that the environment’s constructed proxy/true gap was real and not an artifact of a single layout. We evaluated two hand-coded policies – an exploiter, which conceals any dirt it stands on, and an honest policy, which always delivers dirt to the bin – across 20 randomly generated layouts, running each for 40 steps per episode. Table 2 reports the final proxy and true reward achieved by each policy, averaged across seeds.

Table 2: Hand-coded policy comparison across 20 random layouts (mean $\pm \ \mathrm { s t d } )$
<table><tr><td>Policy</td><td>Final proxy reward Final true reward</td><td></td></tr><tr><td>Exploiter</td><td> $2 4 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Honest</td><td> $2 3 . 9 \pm 0 . 4$ </td><td> $3 . 6 \pm 0 . 5$ </td></tr></table>

![](images/252d7619a5cbe3bf3a9ac2eeab5b82e9b80c55171c126747c1743902b11e26de.jpg)

![](images/5996662f95fdfb67b254d714f587b22b6404f3427d02ab55d06358a178375def.jpg)  
Figure 1: Mean final proxy reward (left) and true reward (right) for the exploiter and honest hand-coded policies, across 20 random layouts. Error bars show one standard deviation.

The exploiter policy reached the maximum possible proxy score with zero variance across seeds, while the true reward remained at exactly zero in every trial. The honest policy reached a statistically indistinguishable proxy score while achieving a substantially higher true reward. These two policies attain nearly identical proxy rewards through entirely diferent means, confirming that the proxy signal alone does not favor the intended behavior.

## 5<sub>.</sub>2 B<sub>ase</sub>li<sub>ne:</sub> A P<sub>roxy-</sub>O<sub>n</sub>l<sub>y</sub> A<sub>gen</sub>t L<sub>earns</sub> t<sub>o</sub> G<sub>ame</sub>

We trained the deep Q-network of Section 3.2 for 500 episodes using only the shaped proxy reward (Eq. 5 with $\lambda = 0 )$ Figure 2 shows the true reward over the course of training; the true reward remained flat near zero throughout, with no sustained upward trend. The final greedy policy (no exploration) was evaluated via rollout: it reached a proxy reward of 23 and a true reward of 0, using 2 cover actions and achieving zero deliveries. This confirms that the agent discovered a proxy-gaming policy during training, consistent with the behavior of the hand-coded exploiter policy in Section 5.1, but without being directed toward concealment specifically.

![](images/9644d99665af0c2e53083c2d111588c3b7e0cc9d95f670f18ea17b39910eef00.jpg)

![](images/d7f34711d7b5849e2127e4f66511cbe6a94c101a00b6fb5c64f2a704c8337567.jpg)  
Figure 2: Baseline (proxy-only) DQN training. True reward over 500 episodes.

## 5<sub>.</sub>3 C<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d A<sub>ge</sub>nt<sub>:</sub> Elimin<sub>a</sub>tin<sub>g</sub> th<sub>e</sub> E<sub>xp</sub>l<sub>o</sub>it

We trained the constrained agent in Section 3.4 for 800 episodes using the full training reward in Eq. 5 with λ = 2.0.   
Table 3 compares the final greedy policy rollouts of the baseline and constrained agents.

Table 3: Baseline vs. constrained agent, final greedy-policy rollout
<table><tr><td>Metric</td><td>Baseline</td><td>Constrained</td></tr><tr><td>Proxy reward</td><td>23</td><td>23</td></tr><tr><td>True reward</td><td>0</td><td>2</td></tr><tr><td>Cover actions</td><td>2</td><td>0</td></tr></table>

Both agents reached an identical proxy reward of 23. The constrained agent’s final policy contained zero cover actions, compared to 2 for the baseline, and achieved a true reward of 2, compared to 0 for the baseline. Figure 3 shows the true reward over training for both agents side by side: the baseline remained flat near zero throughout, while the constrained agent’s true reward rose over the course of training and did not return to zero.

The constrained agent’s final policy did not reach the maximum possible true reward of 4; the remaining 2 units of dirt were left unresolved within the 40-step episode budget. We report and discuss this outcome in Section 6.1.

## 5<sub>.</sub>4 Em<sub>p</sub>iri<sub>ca</sub>l V<sub>e</sub>rifi<sub>ca</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> A<sub>ccou</sub>ntin<sub>g</sub> Id<sub>e</sub>ntit<sub>y</sub>

We evaluated Proposition 1 directly against the final greedy policies of both trained agents, computing $\mathrm { p r o x y } _ { \mathrm { t o t a l } } , \mathrm { t r u e } _ { \mathrm { t o t a l } } ,$ and $\mathbf { c o s t } _ { \mathrm { t o t a l } }$ from the same rollouts reported in Table 3. For the baseline agent: $\mathrm { p r o x y } _ { \mathrm { t o t a l } } = 2 , \mathrm { t r u e } _ { \mathrm { t o t a l } } = 0 , \mathrm { c o s t } _ { \mathrm { t o t a l } } = 2 ,$ and $\mathrm { \ t r u e { } _ { t o t a l } + c o s t { } _ { t o t a l } = 2 = p r o x { } y _ { t o t a l } } .$ . For the constrained agent: $\mathrm { p r o x y } _ { \mathrm { t o t a l } } = 3 , \mathrm { t r u e } _ { \mathrm { t o t a l } } = 2 , \mathrm { c o s t } _ { \mathrm { t o t a l } } = 1 ,$ , and $\mathrm { \ t r u e { } _ { t o t a l } + c o s t { } _ { t o t a l } = 3 = p r o x y _ { t o t a l } . }$ . The identity of Eq. 6 held exactly for both agents, confirming that Proposition 1 precisely describes the implemented system and not only the idealized derivation of Section 4.

## 6 Di<sub>scuss</sub>i<sub>o</sub>n <sub>a</sub>nd Limit<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

The results in Section 5 show a clear qualitative shift rather than a marginal improvement: the baseline agent’s final policy contained no deliveries at all, while the constrained agent’s final policy contained zero concealment actions. This is consistent with the mechanism established in Proposition 1 because the cost signal measures the proxy/true gap exactly rather than approximately, penalizing it in the training reward (Eq. 5) removes the agent’s incentive to resolve dirt through concealment or abandonment specifically, rather than discouraging dirt-resolving behavior in general. The partial recovery of true reward (2 of 4 possible units, Table 3) reflects this precisely: the constrained agent’s policy no longer contains the exploit but has not yet converged to an eficient delivery route for every dirt cell within the fixed step budget, a distinction we return to below.

![](images/644819a55615a977cf22ab026afcbe2c4a10b496f90969b70d9b1611d16bb0ba.jpg)  
Figure 3: Left: true reward over training, baseline vs. constrained. Right: final greedy-policy rollout comparison (proxy reward, true reward, cover-action count).

## 6<sub>.</sub>1 Li<sub>m</sub>it<sub>a</sub>ti<sub>ons</sub>

This study has three limitations.

First, the agent is trained and evaluated on a single fixed grid layout – dirt, bin, and agent-start positions generated once and reused across all episodes – rather than a distribution of randomly sampled layouts. This is a standard simplification for interpretable gridworld demonstrations, and it isolates the proxy/true gaming phenomenon from a separate, harder generalization problem. This also means that our results should be read as establishing that the gaming behavior and the constrained fix both occur on this layout; we make no claim here about whether the fix transfers to unseen layouts without retraining, and treat this as the most immediate direction for follow-up work.

Second, the constrained agent’s final policy, while free of concealment and abandonment behaviors entirely, had not converged to optimal navigation within the episode budget: two of the four dirt units remained unresolved, consistent with residual action oscillation near grid boundaries rather than any renewed incentive to game the proxy. This is a training eficiency gap, separate from and not evidence against the specification-gaming question this work targets – the policy’s qualitative behavior (no concealment, no abandonment) is the outcome our method is designed to produce, and it was produced; full navigational optimality was not a target of the constrained fix itself.

Third, the exact form of Proposition 1 depends on a structural property specific to this environment: every unit of dirt is resolved through exactly one of three mutually exclusive outcomes (covered, delivered, or abandoned), with no possibility of double-counting because the agent can carry at most one unit at a time. An environment without this clean partition – for instance, one where a single action could simultaneously advance and undermine the true objective, or where the agent could act on multiple objective-relevant units concurrently – would require a diferent argument, and the identity as stated here should not be assumed to generalize automatically to such settings. We view identifying the general conditions under which an exact (rather than approximate) proxy/true/cost identity exists as an open question raised but not answered by this work.

## 7 C<sub>onc</sub>l<sub>us</sub>i<sub>on an</sub>d F<sub>u</sub>t<sub>ure</sub> W<sub>or</sub>k

We constructed CleaningGridWorld, a gridworld with a deliberately constructed, exploitable gap between a proxy reward and a true objective, and showed that a standard deep Q-network trained on the proxy alone converges to a policy that games it completely, reaching near-maximal proxy reward while the true reward remains at zero. We then introduced a Lagrangian-penalized constrained RL fix, with a cost signal targeting the specific gaming mechanisms the baseline agent discovered, and showed that it eliminates those behaviors entirely from the final policy while recovering substantial true-objective performance. Underlying this fix is an exact accounting identity, proved for this environment and verified empirically against both trained agents, showing that our cost signal does not merely correlate with the proxy/true gap but measures it exactly, giving the method a guarantee of a diferent character than a purely empirical improvement.

Two directions follow directly from this work. First, testing whether the constrained fix transfers to randomly sampled grid layouts, rather than the single fixed layout used here, would establish whether the result generalizes beyond one navigation problem. Second, implementing the hard-constrained variant of Corollary 1 directly – via action masking or early termination once a fixed cost budget is reached – would allow the deterministic bound on true-objective performance to be tested empirically against a trained system, rather than verified only as a consequence of Proposition 1 on the fixed-λ relaxation studied here.

## C<sub>o</sub>d<sub>e</sub> A<sub>va</sub>il<sub>a</sub>bilit<sub>y</sub>

All code used to construct the environment, train both agents, and reproduce the figures and tables in this paper is publicly available at https://github.com/hildaadwubiosei/Specification-Gaming.

## R<sub>e</sub>f<sub>erences</sub>

Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg. Specification gaming: the flip side of AI ingenuity. DeepMind Blog, 2020. https: //deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Andrew Y. Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning (ICML), 1999.

Eitan Altman. Constrained Markov Decision Processes. CRC Press, 1999.

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.