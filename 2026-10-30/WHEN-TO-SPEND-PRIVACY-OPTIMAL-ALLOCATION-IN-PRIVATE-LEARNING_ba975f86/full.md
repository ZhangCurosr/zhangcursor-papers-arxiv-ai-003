# WHEN TO SPEND PRIVACY: OPTIMAL ALLOCATION IN PRIVATE LEARNING

UMBERTO BICCARI ∗ AND YONGCUN SONG †

Abstract. Privacy accounting bounds the cumulative privacy cost of Differentially Private Stochastic Gradient Descent but does not determine how to allocate a fixed budget across iterations. We formulate the complete Gaussian noise-variance schedule as an open-loop control under a composed zero-Concentrated Differential Privacy constraint, explicitly accounting for schedule selection. For expected quadratic terminal tracking, we prove that every local minimizer exhausts the certified budget, yielding an exact-budget simplex formulation. We introduce Privacy-Budget Adjoint Mirror Descent (PB-AMD), combining pathwise adjoint gradients with entropic updates that preserve positivity, charge floors, and the prescribed budget. Under regularity and coercivity assumptions, we establish existence and budget stability of optimal schedules, design-to-training utility stability, and deterministic, stochastic, and sample-average stationarity guarantees. A linear–quadratic reduction yields an explicit square-root allocation law. Experiments demonstrate tracking improvements over several allocation rules and predictive improvements over uniform allocation in a learningrate restart setting, where learning-rate allocation performs similarly. Comparisons at a common stationarity tolerance show lower schedule-optimization times than projected gradient descent on the tested problems. The framework connects certified privacy accounting with optimization through nonlinear training dynamics.

## 1. Introduction

Differential Privacy (DP) provides a principled framework for limiting the dependence of a randomized algorithm on any individual data record [11]. In stochastic optimization, Differentially Private-Stochastic Gradient Descent (DP-SGD) implements this principle by combining gradient clipping with calibrated Gaussian perturbations [1]. Nevertheless, since training involves repeated accesses to the private dataset, each noisy gradient update contributes to the overall privacy loss of the algorithm. A central question in DP-SGD is therefore how to quantify the cumulative privacy cost of the entire sequence of updates and ensure that it remains within a prescribed privacy budget.

Several accounting frameworks have been developed for this purpose, including zero-concentrated DP (zCDP) and Rényi DP, which provide convenient composition rules for tracking privacy loss across iterations [9, 16, 28, 35]. Such accounting determines how much privacy expenditure is admissible over the entire training procedure, but it does not determine how this expenditure should be distributed. This leads to the central question of this paper:

given a fixed total zCDP budget, how should the Gaussian noise be allocated across the iterations of DP-SGD so as to optimize terminal utility?

Different allocations with the same composed privacy charge need not have the same effect on the final model, since a perturbation introduced at one iteration is propagated through all subsequent optimization steps. The problem is therefore inherently dynamical: the DP-SGD iterates form the state, the sequence of Gaussian noise variances can be interpreted as controls, the expected terminal loss is the performance criterion, and the composed privacy charge acts as a finite resource.

We develop this viewpoint as a finite-horizon stochastic control problem over the complete noise-variance schedule. For the quadratic terminal tracking objective that constitutes the principal setting of the paper, we show that every locally optimal schedule exhausts the available certified privacy charge. Introducing the per-step zCDP charges as decision variables then transforms the saturated resource constraint into an exact-budget simplex. This structure leads to

• an adjoint characterization of the sensitivity of terminal utility to privacy allocation;

• a corresponding Karush-Kuhn-Tucker (KKT) optimality system;

• and a feasibility-preserving optimization method, Privacy-Budget Adjoint Mirror Descent (PB-AMD), whose iterates remain on the prescribed privacy-budget simplex.

The precise meaning of the terminal tracking objective and the privacy-valid information structure under which the schedule may be designed are clarified next.

1.1. Problem formulation and scope. To turn the temporal allocation question above into a well-defined optimization problem, two ingredients must be specified. First, one needs a criterion for comparing admissible privacy schedules, that is, a quantitative measure of how the distribution of noise across iterations affects the final outcome of training. Second, one must specify what information may be used to construct the schedule without introducing an additional, unaccounted privacy cost. These two choices determine, respectively, the objective of the control problem and the information structure under which the resulting schedule is privacy valid.

For the first point, we describe the evolution of the model parameters during training by a sequence $\{ \theta _ { t } \} _ { t = 0 } ^ { T }$ , where $\theta _ { t } \in \mathbb { R } ^ { d }$ denotes the parameter vector at iteration t. At the level relevant for the control formulation, the DP-SGD dynamics are written as

$$
\theta _ { t + 1 } = \theta _ { t } - \eta _ { t } g _ { t } ( \theta _ { t } ) - \eta _ { t } \Delta _ { t } ^ { b } \sqrt { u _ { t } } \xi _ { t } , \quad \xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) , \quad \mathrm { ~ f o r ~ a l l ~ } t \in [ [ T ] _ { 0 } ,
$$

where $g _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the gradient map of a fixed design model, $\eta _ { t } > 0$ is the step size, and $u _ { t } ~ > ~ 0$ determines the variance of the Gaussian perturbation at iteration t. Here, we have introduced the notation $[ [ T ] ] _ { 0 } = \{ 0 , \dots , T - 1 \}$ , that will be adopted throughout the entire paper. The factor $\Delta _ { t } ^ { b }$ is the deterministic scale used to calibrate this perturbation; its precise relation to the privacy mechanism is specified in Section 2.2. The complete variance schedule

$$
u = ( u _ { 0 } , \ldots , u _ { T - 1 } )
$$

is the quantity to be optimized and plays the role of the control. Removing the Gaussian perturbations gives the corresponding noiseless reference dynamics

$$
\bar { \theta } _ { t + 1 } = \bar { \theta } _ { t } - \eta _ { t } g _ { t } ( \bar { \theta } _ { t } ) , \quad t \in [ [ T ] ] _ { 0 } .
$$

The principal criterion used in this paper to compare privacy schedules is the expected squared deviation of the noisy terminal parameter vector from this reference,

$$
J ( u ) = \mathbb { E } \Big [ \| \theta _ { T } ( u ) - \bar { \theta } _ { T } \| ^ { 2 } \Big ] .\tag{1.1}
$$

We refer to this criterion as the terminal tracking objective. It measures how strongly the accumulated privacy perturbations displace the final model from the terminal state that would be produced by the same design dynamics in the absence of privacy noise. In this way, the effect of the entire temporal noise allocation is summarized through its impact on the final state. The control formulation developed below also accommodates more general terminal losses, but quadratic tracking is the principal objective considered in this work and the setting in which we establish the budget-saturation property underlying the exact-budget formulation.

The terminal tracking objective (1.1) should be interpreted as a surrogate for practical learning utility. In general, minimizing the distance to the noiseless terminal state is not equivalent to minimizing empirical or test loss, nor to maximizing predictive accuracy. Its relevance therefore depends on how well the design dynamics approximate the target training dynamics and on whether parameter-space proximity is informative for the downstream learning task. We quantify in Section 3 how discrepancies between the design and target dynamics affect the terminal objective and the suboptimality of schedules optimized on the design model.

A separate issue concerns the information used to optimize the privacy schedule. Throughout the paper, the complete schedule is determined before private training begins and is subsequently kept fixed. When it is constructed from information independent of the sensitive training dataset, its optimization incurs no additional privacy cost. If private information is used instead to select the design model or the schedule, the corresponding selection procedure must itself be differentially private and its privacy cost included in the overall accounting. Section 2 formalizes these two cases and establishes the corresponding privacy guarantees.

1.2. Related work and positioning. We position the present work with respect to three closely related strands of literature: temporal privacy allocation in private learning, optimal-control formulations of learning dynamics, and numerical methods for constrained stochastic optimization.

1.2.1. Temporal privacy allocation. Existing temporal allocation strategies include private line-search adaptation of per-step zCDP budgets [24] and dynamic schedules minimizing a terminal-utility bound that accounts for noise propagation [19]. Other methods jointly vary clipping thresholds and noise levels [10] or combine dynamic budget allocation with layerwise gradient importance [25]. Related approaches address adaptive clipping [2, 8], learning-rate adaptation [23], a priori selection of global DP-SGD hyperparameters [34], and allocation across federated-learning rounds and clients [21]. Model-based protectors can also be learned from non-private tasks [18].

Our focus is the joint allocation of a fixed privacy budget to minimize the expected squared deviation of the final model from its noiseless counterpart under nonlinear training dynamics. We optimize this objective directly over the complete variance schedule, rather than through a prescribed schedule family or an analytical utility bound. Schedule design uses either information independent of the sensitive dataset or a private selector whose cost is included in composition.

Table 1 summarizes the relevant distinctions: temporal variation, joint optimization of the complete schedule, treatment of terminal performance, privacy accounting for schedule selection, and use of the training dynamics in optimization.

<table><tr><td>Approach</td><td>allocation</td><td>Temporal Joint schedule optimization</td><td>Terminal criterion</td><td>Privacy-valid selection</td><td>Schedule construction</td></tr><tr><td>[24]</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>Adaptive private line search</td></tr><tr><td>[19]</td><td>Yes</td><td>Yes</td><td>Upper bound</td><td></td><td>Analytical minimization of a utility bound</td></tr><tr><td>[10]</td><td>Yes</td><td>No</td><td>No</td><td></td><td>Prescribed clipping and noise schedules</td></tr><tr><td>[34]</td><td>Indirect</td><td>No</td><td>Validation</td><td>Yes</td><td>A priori privacy-utility parameter selection</td></tr><tr><td>Present work</td><td>Yes</td><td>Yes</td><td>Expected terminal loss</td><td>Yes</td><td>Terminal-loss optimization through nonlinear dynamics with exact-budget updates</td></tr></table>

Table 1. Comparison with representative approaches to temporal privacy allocation.

1.2.2. Optimal control perspective. Residual and continuous-depth neural networks have been interpreted as controlled dynamical systems, leading to Pontryagin-type optimality conditions and structure-preserving numerical methods [5, 13, 26]; classical discrete-time stochastic control provides the underlying state-controlcost framework [6]. In private learning, [37] estimate the communication rounds needed to reach a target loss and use this estimate to adjust Gaussian noise in federated learning, while RLDP learns a feedback policy for adjusting clipping and noise during private fine-tuning [20].

The control perspective provides a characterization of the sensitivities governing this allocation problem. A pathwise adjoint propagates terminal-loss sensitivities backward through the noisy training trajectory, accounting for how subsequent nonlinear updates transform each perturbation. Combined with the privacy constraint, these sensitivities yield first-order optimality conditions that balance the marginal value of privacy expenditure across iterations and supply the gradient used by the allocation algorithm.

1.2.3. Optimization methodology. Numerical allocation requires terminal sensitivities for the full schedule and updates that respect positivity and the reciprocal privacy constraint. Quadratic penalties introduce a tunable parameter and do not guarantee feasibility at finite parameter values [31]. Alternatively, projected gradient descent in privacy-charge coordinates can use the same adjoint sensitivities and preserve feasibility through Euclidean projection of additive steps.

When privacy charges differ substantially across iterations, however, its additive Euclidean steps suffer from scale sensitivity: a fixed absolute charge change corresponds to a larger relative change in noise variance at a smaller charge. To address this scale sensitivity, we use entropic mirror descent to update privacy charges multiplicatively. We prove that every optimizer of quadratic tracking exhausts the certified budget, yielding an exact-budget simplex in reciprocal privacy-charge coordinates. This reduction motivates using entropic mirror descent, a standard simplex method [4, 30], whose proximal term penalizes equal absolute charge changes more strongly at smaller charges and whose multiplicative updates control relative changes in noise variance.

Implementing entropic mirror descent poses two difficulties: separate forward sensitivity calculations repeat propagation along the noisy nonlinear trajectory, while standard normalization can violate positive charge floors that bound noise variances and clipping would exceed the budget. We compute the full schedule gradient using one forward simulation and one backward adjoint sweep per Gaussian path.

The standard entropic proximal step on the simplex with charge floors reduces to a scalar normalization enforcing both the floors and the budget equality. The resulting Privacy-Budget Adjoint Mirror Descent (PB-AMD) preserves positive, exact-budget iterates. For its analysis, relative smoothness provides the Bregman counterpart of Euclidean smoothness [3, 27], while stochastic mirror-descent and sample-average approximation provide frameworks for stochastic and Monte Carlo optimization [14, 15, 22, 29, 33].

## 1.3. Main contributions. We advance temporal privacy allocation through three connected contributions.

1. A control-theoretic foundation for temporal privacy allocation. We formulate joint optimization of the full Gaussian-variance schedule through nonlinear training dynamics as a finite-horizon stochastic control problem. For privacy validity, we certify private training for data-independent schedules and for schedules returned by a private selector whose cost is included in composition. Under the stated assumptions, we also establish existence and budget stability of optimal schedules. When design and training models differ, we bound the effects on tracking performance and allocation suboptimality.

2. A budget-saturation theorem enabling exact-budget optimization. We prove that every local minimizer of the expected quadratic terminal tracking objective exhausts the certified privacy budget. Consequently, reciprocal privacy-charge coordinates reduce the search to an exact-budget simplex. The pathwise adjoint supplies the full schedule gradient, from which we derive the KKT system characterizing marginal privacy values across the training horizon.

3. A dynamics-aware algorithm with exact privacy-budget enforcement. Building on this exact-budget structure, we introduce Privacy-Budget Adjoint Mirror Descent (PB-AMD), whose floor-aware entropic updates use adjoint gradients through the full nonlinear training dynamics and enforce positivity and the prescribed total privacy charge at every iteration, without a penalty parameter. For this method, we establish deterministic and stochastic stationarity guarantees and quantify the transfer from fixed-path sample-average stationarity to the population objective. The linear--quadratic reduction further yields an explicit square-root allocation law and a global convergence result.

Figure 1 summarizes the logical structure connecting these contributions, from privacy-valid schedule design and the stochastic control formulation to the exact-budget optimization method and its numerical validation.

1.4. Notation. Table 2 collects some recurrent notation used in this paper and its meaning.

1.5. Organization. Section 2 establishes the privacy-accounting framework, calibrates the DP-SGD mechanism, and separates private training from the design model used to select an open-loop schedule. Section 3 formulates the finite-horizon control problem and develops its existence, stability, budget-saturation, adjoint, and KKT properties. Section 4 derives PB-AMD on the exact-budget simplex and proves its deterministic, stochastic, sample-average, and linear-quadratic convergence results. Section 5 evaluates the predicted allocation regimes, terminal utility, exact constraint realization, and solver efficiency. Section 6 concludes the paper.

![](images/48ce0a68a33506fc1a0baf8c6f028f17d31eefe877ed2cfdd3b55df1aa883fd5.jpg)  
Figure 1. Conceptual roadmap of the paper.

<table><tr><td>Notation</td><td>Meaning</td></tr><tr><td>N</td><td> $\{ 1 , 2 , \ldots \}$ </td></tr><tr><td> $[ [ q ] ] , [ [ q ] ] _ { 0 }$ </td><td> $\{ 1 , \ldots , q \}$  and  $\{ 0 , \ldots , q - 1 \}$  , respectively, with  $q \in \mathbb { N }$ </td></tr><tr><td> $[ [ q ] ^ {  } , [ [ q ] ] _ { 0 } ^ {  }$ </td><td>indices in  $[ [ q ] ]$  and  $[ [ q ] ] _ { 0 }$  taken in reverse order,  $\mathrm { i . e . , } q , \ldots , 1$  and  $q - 1 , \ldots , 0$ </td></tr><tr><td> $\left\| \cdot \right\| _ { 1 } , \left\| \cdot \right\| _ { 2 } , \left\| \cdot \right\| _ { \infty }$   $\langle \cdot , \cdot \rangle , \| \cdot \| _ { \mathrm { o p } }$ </td><td> $\ell ^ { 1 } , \ell ^ { 2 }$  and  $\ell ^ { \infty }$  norms on  $\mathbb { R } ^ { n }$  , respectively, with  $n \in \mathbb { N }$  Euclidean inner product and induced operator norm</td></tr><tr><td> $I _ { n }$ </td><td> $n \times n$  identity matrix</td></tr><tr><td> $\preceq$  and ≥ Tr and diag</td><td>Loewner order on symmetric matrices</td></tr><tr><td></td><td>matrix trace and the diagonal-matrix operator</td></tr><tr><td> $\operatorname* { P r } ( \cdot )$  and E</td><td></td></tr><tr><td></td><td>probability and expectation the probability measure  $P$ </td></tr><tr><td> $P \ll Q$ </td><td>is absolutely continuous with respect to the probability measure  $Q , \operatorname { i . e . , } Q ( A ) = 0$  implies  $P ( A ) = 0$  for every measurable set A</td></tr></table>

Table 2. Summary of some notation employed in the paper.

## 2. Privacy accounting and open-loop schedule certification

We recall here some known DP notions used in this paper. In particular, we give the formal definition of DP, explain the privacy loss originating from repeated application of a differentially private mechanism and motivating the notion of a privacy budget, and establish privacy guarantees for the DP-SGD mechanism considered below.

2.1. Zero-concentrated differential privacy and Gaussian privacy accounting. We recall the privacy framework used throughout the paper; formal definitions and additional details are collected in Appendix A. We work under replace-one adjacency: two datasets D and $D ^ { \prime }$ of the same size N are neighboring, written $D \sim D ^ { \prime }$ , if they differ in exactly one entry. Privacy is quantified using the notion of zero-Concentrated Differential Privacy (zCDP) [9]. Given $\rho \geq 0$ , a randomized mechanism M satisfies $\rho { \mathrm { - z C D P ~ i f } } .$ for every pair $D \sim D ^ { \prime }$ and every order $\alpha > 1$

$$
\mathfrak { D } _ { \alpha } \big ( \mathcal { L } ( \mathcal { M } ( D ) ) \| \mathcal { L } ( \mathcal { M } ( D ^ { \prime } ) ) \big ) \leq \rho \alpha .
$$

Here $\mathcal { L } ( \cdot )$ denotes the law of a random variable, and ${ \mathfrak { D } } _ { \alpha }$ is the Rényi divergence of order $\alpha$ (see Definition A.2). This condition controls how much the output distribution can change when one data record is replaced. Smaller values of $\rho$ provide stronger privacy guarantees. We refer to $\rho$ as the privacy charge.

For the present paper, the importance of zCDP is that it provides a particularly simple accounting rule for Gaussian perturbations. If two randomized mechanisms $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ satisfy $\rho _ { 1 ^ { - } }$ and $\rho _ { 2 ^ { - } \mathrm { Z } } \mathrm { C D P }$ , then their composition $\mathcal { M } _ { 1 } \circ \mathcal { M } _ { 2 }$ fulfills $( \rho _ { 1 } + \rho _ { 2 } ) { \mathrm { - z C D P } }$ . Moreover, zCDP is preserved under post-processing: if M satisfies $\rho { \mathrm { - z C D P } }$ , then any computation using only its output and independent randomness also satisfies $\rho { \mathrm { - z C D P } }$

These properties provide the privacy-accounting framework for our study of DP-SGD [1]. Gaussian noise protects each gradient query, while composition bounds the privacy charge of the full training procedure. In the next subsection, we identify the certified per-step charge associated with each noise variance and establish the conditions under which the selected schedule yields a valid overall privacy guarantee.

2.2. A calibrated DP-SGD mechanism. We present here the DP-SGD approach of [1] and discuss its zCDP properties. First, we define the private training dynamics of DP-SGD and identify the zCDP charge associated with each iteration. Second, we establish the privacy guarantees under two schedule-selection mechanisms: (i) schedules fixed independently of the sensitive dataset and (ii) schedules produced by a differentially private preliminary procedure.

Fix a training horizon $T \geq 1$ and a parameter dimension $d \in \mathbb { N } .$ . Denote by $\ell _ { t } ( \theta , z )$ the per-example loss at iteration $t \in [ [ T ] ] _ { 0 }$ , and define the clipped gradient at level $C _ { \mathrm { c l i p } } > 0$ by

$$
c _ { t } ( \theta , z ) : = \beta ^ { - 1 } \nabla _ { \theta } \ell _ { t } ( \theta , z ) \quad \mathrm { ~ w i t h ~ } \quad \beta : = \operatorname* { m a x } \left\{ 1 , \frac { \| \nabla _ { \theta } \ell _ { t } ( \theta , z ) \| _ { 2 } } { C _ { \mathrm { c l i p } } } \right\} ,
$$

so that $\| c _ { t } ( \theta , z ) \| _ { 2 } \leq C _ { \mathrm { c l i p } }$ . At each iteration $t \in [ [ T ] ] _ { 0 }$ , we first sample a batch $B _ { t } \subset [ [ N ] ]$ of prescribed cardinality $b _ { t }$ , independently of the data values. We then compute the corresponding clipped batch-average gradient:

$$
g _ { t } ^ { D } ( \theta ) : = \frac { 1 } { b _ { t } } \sum _ { i \in B _ { t } } c _ { t } ( \theta , z _ { i } ) .
$$

Moreover, we define the exact uniform conditional sensitivity by

$$
\Delta _ { t } : = \operatorname* { s u p } _ { \stackrel { B \subset [ N ] } { | B | = b _ { t } } } \operatorname* { s u p } _ { \theta \in \mathbb { R } ^ { d } } \operatorname* { s u p } _ { D \sim D ^ { \prime } } \left\| \frac { 1 } { b _ { t } } \sum _ { i \in B } \bigl ( c _ { t } ( \theta , z _ { i } ) - c _ { t } ( \theta , z _ { i } ^ { \prime } ) \bigr ) \right\| _ { 2 } ,\tag{2.1}
$$

Notice that at most one summand changes when one record is replaced, and each clipped gradient has norm at most $C _ { \mathrm { c l i p } }$ . Consequently,

$$
0 \leq \Delta _ { t } \leq \Delta _ { t } ^ { b } , \quad \mathrm { ~ w i t h ~ } \quad \Delta _ { t } ^ { b } : = \frac { 2 C _ { \mathrm { c l i p } } } { b _ { t } } .\tag{2.2}
$$

The supremum over $\theta$ makes this bound (2.2) valid uniformly over the random iterates generated during private training.

We use $\Delta _ { t } ^ { b }$ to calibrate the Gaussian perturbations in DP-SGD. Assume that the initialization $\theta _ { 0 }$ and all shared public randomness have a joint law independent of the private dataset. Let $\{ \xi _ { t } \} _ { t = 0 } ^ { T - 1 }$ be independent standard Gaussian vectors. At each step, conditional on the selected schedule, initialization, current batch indices, shared public randomness, and preceding transcript, the innovation $\xi _ { t }$ has law $\mathcal { N } ( 0 , I _ { d } )$ under every dataset. Given step sizes $\eta _ { t } > 0$ and noise multipliers $\sigma _ { t } > 0$ , the private training mechanism is

$$
\theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \left( g _ { t } ^ { D } ( \theta _ { t } ) + \Delta _ { t } ^ { b } \sigma _ { t } \xi _ { t } \right) , \qquad \xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) , \quad t \in [ [ T ] ] _ { 0 } .\tag{2.3}
$$

Conditionally on the preceding training transcript and the public batch $B _ { t } ,$ the current iterate $\theta _ { t }$ is fixed. The query $g _ { t } ^ { D } ( \theta _ { t } )$ has sensitivity at most $\Delta _ { t } .$ , and the Gaussian perturbation in (2.3) has standard deviation $\Delta _ { t } ^ { b } \sigma _ { t }$ . By [9, Proposition 1.6], this noisy query satisfies $\rho _ { t } ^ { \mathrm { e x } } { - } \mathrm { z C D P }$ , where

$$
\rho _ { t } ^ { \mathrm { e x } } : = \frac { \Delta _ { t } ^ { 2 } } { 2 ( \Delta _ { t } ^ { b } \sigma _ { t } ) ^ { 2 } } = \left( \frac { \Delta _ { t } } { \Delta _ { t } ^ { b } } \right) ^ { 2 } \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } \leq \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } .\tag{2.4}
$$

The subsequent multiplication by $\eta _ { t }$ and subtraction from the fixed iterate $\theta _ { t }$ are post-processing operations, so the same privacy guarantee holds for the update $\theta _ { t + 1 }$

Thus, each iteration is certified to satisfy $( \bar { 2 } \sigma _ { t } ^ { 2 } ) ^ { - 1 } \mathrm { - z C D P }$ . Introducing the noise-variance controls $u _ { t } : = \sigma _ { t } ^ { 2 }$ we define the certified per-step privacy charge by

$$
q _ { t } : = \frac { 1 } { 2 u _ { t } } = \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } .\tag{2.5}
$$

Here, $\sigma _ { t }$ is the noise multiplier relative to the replace-one sensitivity bound $2 C _ { \mathrm { c l i p } } / b _ { t } ;$ the standard deviation of the noise added to the batch-average gradient is therefore $2 C _ { \mathrm { c l i p } } \sigma _ { t } / b _ { t }$

This is the quantity used in the privacy optimization in Section 3. It equals $\rho _ { t } ^ { \mathrm { e x } }$ when $\Delta _ { t } ^ { b } = \Delta _ { t }$ and exceeds $\rho _ { t } ^ { \mathrm { e x } }$ when $\Delta _ { t } ^ { b } > \Delta _ { t }$

Remark 2.1. The quantity $\Delta _ { t }$ is the exact sensitivity, whereas $\Delta _ { t } ^ { b }$ is an explicit upper bound supplied by clipping. Privacy certification requires only a valid upper bound, so computing $\Delta _ { t }$ is unnecessary. For fixed $\sigma _ { t } , \ i f \ \Delta _ { t } < \Delta _ { t } ^ { b }$ , using $\Delta _ { t } ^ { b }$ rather than $\Delta _ { t }$ increases the standard deviation of the noise added to the gradient from $\Delta _ { t } \sigma _ { t }$ to $\Delta _ { t } ^ { b } \sigma _ { t }$ . A smaller valid upper bound on $\Delta _ { t }$ may therefore replace $\Delta _ { t } ^ { b }$ in (2.3) to reduce the noise.

We can now certify the privacy of the complete training procedure. We begin with the setting used principally in the control formulation, in which the entire variance schedule is fixed independently of the sensitive dataset.

Theorem 2.2. Assume that the initialization $\theta _ { 0 }$ and shared public randomness have a joint law independent of the dataset, and use the conditional Gaussian sampling rule above. For a deterministic schedule $u =$ $( u _ { 0 } , \ldots , u _ { T - 1 } ) \in ( 0 , \infty ) ^ { T }$ , let $\mathcal { M } _ { u }$ release either the full iterate transcript $\{ \theta _ { t } \} _ { t = 1 } ^ { T }$ of (2.3) or only the terminal iterate $\theta _ { T }$ . Then $\mathcal { M } _ { u }$ is $\rho ( u ) { - } z C D P$ , where

$$
\rho ( u ) : = \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { 2 u _ { t } } .\tag{2.6}
$$

Proof. Fix the initialization and shared public randomness, whose joint law is independent of the dataset, as well as the deterministic schedule. At iteration $t ,$ condition on the preceding transcript, the public batch indices and the public randomness revealed so far. The current state $\theta _ { t }$ is then fixed. The conditional Gaussian sampling rule keeps $\xi _ { t }$ standard Gaussian under this conditioning. The sensitivity bound (2.1), being uniform in $\theta ,$ applies to the query $D \mapsto g _ { t } ^ { D } ( \theta _ { t } )$ . The Gaussian mechanism therefore makes this conditional update $\rho _ { t } ^ { \mathrm { e x } } { \mathrm { - z } } \mathrm { C D P }$ , with $\rho _ { t } ^ { \mathrm { e x } }$ given by (2.4). Adaptive composition then shows that the complete training transcript is $\rho ( u ) – \mathrm { z C D P }$ , since

$$
\sum _ { t = 0 } ^ { T - 1 } \rho _ { t } ^ { \mathrm { e x } } \le \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { 2 u _ { t } } = \rho ( u ) .
$$

The bound remains valid after averaging over the data-independent initialization and public randomness. The terminal iterate $\theta _ { T }$ is a function of this transcript and hence satisfies the same guarantee by post-processing. In particular, if $\rho ( u ) \leq \rho _ { \mathrm { m a x } }$ , the mechanism is $\rho _ { \mathrm { m a x } ^ { - } \mathrm { Z C D P } }$

Now let U be a random schedule whose joint law with the initialization and shared public randomness is independent of the dataset, with $\rho ( U ) \le \bar { \rho }$ almost surely. Write $\mu$ for its law, which is the same under neighboring datasets. Conditional on $U = u$ , the preceding argument gives a $\bar { \rho } { \mathrm { - z C D P } }$ guarantee. We show that this guarantee also holds when U is released together with the training output.

For neighboring datasets $D \sim D ^ { \prime }$ , let $P _ { u } , Q _ { u }$ denote the conditional laws of the training output given $U = u$ . For every $\alpha > 1$ ，

$$
\mathfrak { D } _ { \alpha } ( P _ { u } \Vert Q _ { u } ) \le \alpha \bar { \rho } \qquad \mathrm { f o r } \ \mu \mathrm { - a l m o s t ~ e v e r y } \ u .
$$

The joint output laws are $P ( d u , d y ) = \mu ( d u ) P _ { u } ( d y )$ and $Q ( d u , d y ) = \mu ( d u ) Q _ { u } ( d y )$ . Because they have the same schedule marginal $\mu ,$

$$
\exp \bigl ( ( \alpha - 1 ) \mathfrak { D } _ { \alpha } ( P | | Q ) \bigr ) = \int \exp \bigl ( ( \alpha - 1 ) \mathfrak { D } _ { \alpha } ( P _ { u } | | Q _ { u } ) \bigr ) \mu ( d u ) \leq \exp \bigl ( ( \alpha - 1 ) \alpha \bar { \rho } \bigr ) .
$$

Thus $\mathfrak { D } _ { \alpha } ( P \| Q ) \le \alpha \bar { \rho }$ for every $\alpha > 1$ , proving that the joint mechanism is $\bar { \rho } { \mathrm { - z C D P } }$ . The same guarantee holds if U is omitted, by post-processing. □

Theorem 2.2 shows that optimizing a schedule does not itself consume privacy budget when the information used to construct that schedule is independent of the sensitive dataset. This is the setting formalized through the fixed design model in Section 2.3. A different situation arises when the sensitive dataset is used to choose the schedule. Such selection is still admissible, but its privacy cost must be included explicitly.

Theorem 2.3. Under the same data-independent initialization assumption as in Theorem 2.2, let $\rho _ { \mathrm { s e l } } \geq 0$ and $\rho _ { \mathrm { m a x } } > 0$ . Let $R _ { \mathrm { p u b } }$ collect the initialization and public randomness shared by schedule selection and training. Suppose that a schedule selector A makes the joint release $( R _ { \mathrm { p u b } } , A ( D ) ) \ \rho _ { \mathrm { s e l } } { - \mathcal { z } C D P }$ and returns, almost surely, a schedule $u = \boldsymbol { \mathcal { A } } ( D ) \in ( 0 , \infty ) ^ { T }$ satisfying $\rho ( u ) \leq \rho _ { \mathrm { m a x } }$ with $\rho ( u )$ given by (2.6). Conditional on the selected schedule, run (2.3) using fresh Gaussian randomness. Then the joint mechanism releasing both the selected schedule and the training output is $( \rho _ { \mathrm { s e l } } + \rho _ { \mathrm { m a x } } ) – z C D P$

Proof. The first mechanism releases $( R _ { \mathrm { p u b } } , u )$ and incurs charge at most $\rho _ { \mathrm { s e l } }$ . For almost every returned pair, the initialization and shared public randomness are fixed, and the conditional Gaussian sampling rule applies. The argument of Theorem 2.2 therefore bounds the conditional training charge by $\rho _ { \mathrm { m a x } }$ . Adaptive composition gives charge at most $\rho _ { \mathrm { s e l } } + \rho _ { \mathrm { m a x } }$ for the joint release of $( R _ { \mathrm { p u b } } , u )$ and the training output. Omitting $R _ { \mathrm { p u b } }$ or releasing only the terminal model preserves this bound by post-processing. □

Theorems 2.2 and 2.3 identify the information structures under which the schedule optimization developed later is privacy valid. If the schedule is constructed entirely from information independent of the sensitive dataset, its selection carries no additional privacy charge. If private information is used to select the schedule, the selection procedure must itself be private and its charge composed with the training charge. The next subsection makes this distinction explicit by separating the data-dependent training mechanism $g _ { t } ^ { D }$ from the fixed design model $g _ { t }$ used in the control problem.

2.3. Private training mechanism versus design model. In the strategy proposed in this work, the privacy schedule is optimized before private training begins. To determine this schedule, we need a model of the training dynamics that describes how the noise injected at different iterations affects the terminal objective. Using the private queries $g _ { t } ^ { D }$ directly for this purpose would make the schedule data-dependent and would therefore require accounting for the privacy cost of its selection. Our main formulation avoids this additional cost by optimizing the schedule on a fixed, data-independent design model. We denote this design model by

$$
\mathcal { D } _ { \mathrm { d e s } } : = \left( \bar { \theta } _ { 0 } , \{ g _ { t } , \eta _ { t } , \Delta _ { t } ^ { b } \} _ { t = 0 } ^ { T - 1 } , L \right) ,
$$

The tuple $\mathcal { D } _ { \mathrm { d e s } }$ represents the complete model used to simulate the training dynamics and evaluate candidate privacy schedules before private training. Here, $\bar { \theta } _ { 0 }$ is the design initialization, $g _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the design gradient map, $\eta _ { t } > 0$ is the step size, $\Delta _ { t } ^ { b } > 0$ is the sensitivity bound used to scale the Gaussian perturbation, and $L$ is the terminal loss. The maps g may, for example, be constructed from public or synthetic data or obtained from another data-independent approximation of the training dynamics. The concrete construction of the design models used in this work is specified in Section 5.1, together with the experimental setup.

Once the schedule has been optimized on $\mathcal { D } _ { \mathrm { d e s } } .$ it is fixed and applied to the actual private training mechanism (2.3), where the design maps $g _ { t }$ are replaced by the private queries $g _ { t } ^ { D }$ . If $\mathcal { D } _ { \mathrm { d e s } }$ is independent of the sensitive dataset, schedule optimization incurs no additional privacy cost and Theorem 2.2 applies. If private information is instead used to construct the design model or select the schedule, the selection procedure must itself be differentially private, as described in Theorem 2.3.

Besides, since the schedule is designed using $g _ { t }$ but deployed using $g _ { t } ^ { D }$ , its performance may be affected by discrepancies between the design and private training dynamics. Theorem 3.10 quantifies the resulting effect on the terminal objective and schedule suboptimality. In all cases, the condition $\Delta _ { t } ^ { b } \geq \Delta _ { t }$ ensures valid calibration of the Gaussian perturbation used during private training.

## 3. Open-loop control formulation and optimality structure

This section develops the main mathematical foundations of the proposed privacy-allocation framework by formulating the fixed design problem introduced in Subsection 2.3 as a finite-horizon stochastic contro problem, with the complete noise-variance schedule as the control. We first show that, for the quadratic terminal tracking objective, every locally optimal schedule exhausts the available certified privacy budget. This property allows us to reformulate the problem on an exact-budget simplex using per-step privacy charges as decision variables. We then derive a pathwise adjoint formula for the gradient of the terminal objective and establish the corresponding KKT optimality conditions. Finally, we investigate the existence and stability of optimal schedules, their dependence on the privacy budget, and the effects of discrepancies between the design model and the actual training dynamics.

To emphasize the main ideas and their implications for privacy allocation, we accompany the results with their interpretation and, where appropriate, a brief explanation of the underlying arguments. Complete proofs are collected in Appendix B.

Fix $T \geq 1$ and maps $g _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , step sizes $\eta _ { t } > 0$ , scales $\Delta _ { t } ^ { b } > 0$ , and a design initialization $\bar { \theta } _ { 0 } \in \mathbb { R } ^ { d }$ Let the noiseless reference trajectory be

$$
\bar { { \theta } } _ { t + 1 } = \bar { { \theta } } _ { t } - \eta _ { t } g _ { t } ( \bar { { \theta } } _ { t } ) , \qquad \theta ^ { \star } : = \bar { { \theta } } _ { T } , \quad t \in [ [ T ] ] _ { 0 } ,
$$

and, for $t \in [ [ T ] ] _ { 0 }$ , let the controlled stochastic dynamics be

$$
\theta _ { t + 1 } ( u ) = \theta _ { t } ( u ) - \eta _ { t } g _ { t } ( \theta _ { t } ( u ) ) - \eta _ { t } \Delta _ { t } ^ { b } \sqrt { u _ { t } } \xi _ { t } , \qquad \theta _ { 0 } ( u ) = \bar { \theta } _ { 0 } .\tag{3.1}
$$

For a terminal loss $L : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ , set

$$
J ( u ) : = \mathbb { E } \left[ L ( \theta _ { T } ( u ) ) \right] .\tag{3.2}
$$

Moreover, let $\rho ( u )$ be given by (2.6) and, for $\rho _ { \mathrm { m a x } } > 0$ , define

$$
\mathcal { U } _ { \mathrm { a d } } ( \rho _ { \mathrm { m a x } } ) : = \Big \{ u \in ( 0 , \infty ) ^ { T } : \rho ( u ) \leq \rho _ { \mathrm { m a x } } \Big \} .\tag{3.3}
$$

The open-loop problem associated with DP noise scheduling is

$$
\operatorname* { m i n } _ { \boldsymbol u \in \mathcal U _ { \mathrm { a d } } ( \rho _ { \mathrm { m a x } } ) } J ( \boldsymbol u ) \quad \mathrm { s u b j e c t ~ t o ~ t h e ~ d y n a m i c s ~ ( 3 . 1 ) . }\tag{3.4}
$$

3.1. Analytical assumptions. We collect here the assumptions used in the analysis of the control problem. They fall naturally into two groups. Assumptions $( \mathrm { A 1 } ) \mathrm { - } ( \mathrm { A 5 } )$ concern the stochastic dynamics, the terminal loss, and the coercivity properties needed for the basic well-posedness and stability results. Assumptions $( \mathrm { A 6 } ) – ( \mathrm { A 7 } )$ impose additional differentiability and growth conditions required only for the adjoint, first-order optimality, and algorithmic regularity analysis.

## 3.1.1. Basic assumptions for well-posedness and stability.

(A1) (Independent Gaussian innovations). The random vectors $\xi _ { 0 } , \dots , \xi _ { T - 1 }$ are mutually independent and satisfy $\xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$

(A2) (Local regularity and linear growth). For every $t \in [ [ T ] ] _ { 0 }$ , the map $g _ { t }$ is locally Lipschitz and there are $a _ { t } , b _ { t } \geq 0$ such that $\lVert g _ { t } ( { \boldsymbol { \theta } } ) \rVert \leq a _ { t } + b _ { t } \lVert { \boldsymbol { \theta } } \rVert$ for all $\theta \in \mathbb { R } ^ { d }$

(A3) (Continuous polynomial growth). The loss L is continuous and bounded below by a finite constant $L _ { \mathrm { l b } } \in \mathbb { R }$ . Moreover, there exist $r _ { L } \ge 1$ and $K _ { L } > 0$ such that $\vert L ( \theta ) \vert \leq K _ { L } \big ( 1 + \| \theta \| ^ { r _ { L } } \big )$ for all $\theta \in \mathbb { R } ^ { d }$

(A4) (Quadratic coercive lower bound). There exist $c _ { 1 } > 0$ and $c _ { 0 } \geq 0$ such that $L ( \theta ) \geq c _ { 1 } \left\| \theta - \theta ^ { \star } \right\| ^ { 2 } - c _ { 0 }$ for all $\theta \in \mathbb { R } ^ { d }$

(A5) (Quadratic non-collapse of deterministic steps). Define $\mathsf { G } _ { t } ( \theta ) : = \theta - \eta _ { t } g _ { t } ( \theta )$ . For every $t \in [ [ T ] ] _ { 0 }$ , there exist $\chi _ { t } > 0$ and $\omega _ { t } \geq 0$ such that

$$
\left\| \mathsf { G } _ { t } ( x ) - \mathsf { G } _ { t } ( y ) \right\| ^ { 2 } \geq \chi _ { t } ^ { 2 } \left\| x - y \right\| ^ { 2 } - \omega _ { t } , \qquad x , y \in \mathbb { R } ^ { d } .\tag{3.5}
$$

Assumptions $\mathrm { ( A 1 ) } { - } \mathrm { ( A 3 ) }$ provide the basic probabilistic and continuity framework for the controlled dynamics and terminal objective. Assumptions $( \mathrm { A } 4 ) { - } ( \mathrm { A } 5 )$ are the additional coercivity conditions used to prevent minimizing sequences from escaping to unbounded noise-variance schedules and hence to establish existence and stability of optimal controls.

3.1.2. Additional assumptions for first- and second-order analysis. The following assumptions are not required for the existence theory. They are imposed when differentiability of the reduced objective and higher-order regularity are needed.

(A6) (Smoothness for first-order analysis). Each $g _ { t }$ is continuously differentiable, and there are $K _ { g } > 0$ and $r _ { g } \ge 0$ such that $\| \nabla g _ { t } ( \theta ) \| _ { \mathrm { o p } } \le K _ { g } \big ( 1 + \| \theta \| ^ { r _ { g } } \big )$ for all $\theta \in \mathbb { R } ^ { d }$ and $t \in [ [ T ] ] _ { 0 }$ . The loss L is continuously differentiable with $\| \nabla L ( \theta ) \| \le \bar { K } _ { \nabla } \big ( 1 + \| \theta \| ^ { r _ { \nabla } } \big )$ for some $K _ { \nabla } > 0$ and $r _ { \nabla } \geq 0$

(A7) (Second-order polynomial regularity). Each $g _ { t }$ is twice continuously differentiable and $L$ is twice continuously differentiable. There are constants $K _ { 2 } > 0$ and $r _ { 2 } \geq 0$ such that, for every $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ and $t \in [ [ T ] ] _ { 0 }$

$$
\begin{array} { r } { \left\| D ^ { 2 } g _ { t } ( { \boldsymbol { \theta } } ) \right\| _ { \mathrm { o p } } + \left\| \nabla ^ { 2 } L ( { \boldsymbol { \theta } } ) \right\| _ { \mathrm { o p } } \leq K _ { 2 } \big ( 1 + \left\| { \boldsymbol { \theta } } \right\| ^ { r _ { 2 } } \big ) , } \end{array}
$$

where $D ^ { 2 } g _ { t } ( \theta ) : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the second derivative of $g _ { t }$ , viewed as a bilinear map, with

$$
\left\| D ^ { 2 } g _ { t } ( \theta ) \right\| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { \| v \| _ { 2 } = \| w \| _ { 2 } = 1 } \left\| D ^ { 2 } g _ { t } ( \theta ) [ v , w ] \right\| _ { 2 } .
$$

To help the reader navigate the assumptions used in the different parts of the analysis, Table 3 summarizes their roles and the main results in which they enter.

<table><tr><td>Assumptions</td><td>Main role</td><td>Main results requiring them</td></tr><tr><td>(A1)-(A2)</td><td>Basic stochastic dynamics and moment control</td><td>Continuity of J, basic control formulation Theorems 3.2, 3.8; Propositions 3.4, B.3, B.5</td></tr><tr><td>(A3)</td><td>Integrability and continuity of the terminal objective</td><td></td></tr><tr><td>(A4)-(A5)</td><td>Coercivity and compactness of minimizing sequences</td><td>Existence and compactness of minimizers Proposition B.5; Theorems 3.8, 3.9</td></tr><tr><td>(A6)</td><td>First-order sensitivity analysis</td><td>Adjoint gradient, KKT conditions, privacy-charge gradient Propositions 3.4, 3.7, Theorem 3.5, Corollary 3.6</td></tr><tr><td>(A7)</td><td>Second-order regularity for optimization</td><td>Smoothness estimates needed for PB-AMD convergence Proposition C.1</td></tr></table>

Table 3. Analytical assumptions and the results in which they are used.

Remark 3.1. For the smoothly clipped logistic and Softplus design models in Section 5, bounded datagradient terms with polynomially growing derivatives and linear $\ell ^ { \bar { 2 } }$ -regularization gradients ensure (A2), $( A \theta )$ , and (A7). The Gaussian innovations satisfy (A1), and $L _ { \mathrm { t r } } ( \theta ) : = \left\| \theta - \theta ^ { \star } \right\| ^ { 2 }$ satisfies (A4) with $c _ { 1 } = 1$ and $c _ { 0 } = 0$ . For (A5), i $f g _ { t } ( \theta ) = r _ { t } ( \theta ) + R _ { t } \theta , \operatorname* { s u p } _ { \theta } \| r _ { t } ( \theta ) \| \leq C _ { t }$ , and $I _ { d } - \eta _ { t } R _ { t }$ is nonsingular, then (3.5) holds with $\begin{array} { r } { \chi _ { t } ^ { 2 } = \frac { 1 } { 2 } s _ { \mathrm { m i n } } ( I _ { d } - \eta _ { t } R _ { t } ) ^ { 2 } } \end{array}$ and $\omega _ { t } = 4 \eta _ { t } ^ { 2 } C _ { t } ^ { 2 }$ , where $s _ { \mathrm { m i n } }$ denotes the smallest singular value.

3.2. Exact certified-budget saturation for terminal tracking. We begin with a structural property of the control problem that will be essential for the exact-budget reformulation developed in the next subsection. Recall that the admissible set in (3.3) is defined by the inequality $\rho ( u ) \leq \rho _ { \mathrm { m a x } }$ , so a feasible schedule is not required a priori to expend the entire certified privacy budget. Since the certified privacy charge at iteration t is $( 2 u _ { t } ) ^ { - 1 }$ , decreasing $u _ { t }$ increases privacy expenditure while reducing the variance of the Gaussian perturbation. It is therefore natural to ask whether an optimal schedule can leave part of the available budget unused.

For the quadratic tracking objective, the answer is no. Indeed, once $u _ { 0 } , \ldots , u _ { T - 2 }$ are fixed, the state $\theta _ { T - 1 }$ depends only on the previous Gaussian innovations and is therefore independent of $\xi _ { T - 1 }$ . The last update

can be written as

$$
\theta _ { T } - \theta ^ { \star } = \theta _ { T - 1 } - \eta _ { T - 1 } g _ { T - 1 } ( \theta _ { T - 1 } ) - \theta ^ { \star } - \eta _ { T - 1 } \Delta _ { T - 1 } ^ { b } \sqrt { u _ { T - 1 } } \xi _ { T - 1 } .
$$

For the squared tracking loss, independence and the zero mean of the final innovation eliminate the cross term after taking expectation. Hence the dependence of the objective on $u _ { T - 1 }$ is explicit and strictly increasing. If a feasible schedule had slack in the privacy constraint, one could therefore decrease $u _ { T - 1 }$ slightly, remain feasible, and strictly improve the objective. The following theorem formalizes this observation.

Theorem 3.2. Assume $T \geq 1$ , Assumptions (A1) and (A2), and set the loss function to be

$$
L ( \theta ) = \| \theta - \theta ^ { \star } \| ^ { 2 } = : L _ { \mathrm { t r } } ( \theta ) .\tag{3.6}
$$

If $\eta _ { T - 1 } \Delta _ { T - 1 } ^ { b } > 0$ , then, for fixed $( u _ { 0 } , \ldots , u _ { T - 2 } )$ , the reduced objective is $f i n i t e , \ a f f i n e ,$ , and strictly increasing in $u _ { T - 1 } .$

$$
J ( u ) = \widetilde { J } ( u _ { 0 } , \ldots , u _ { T - 2 } ) + d \eta _ { T - 1 } ^ { 2 } ( \Delta _ { T - 1 } ^ { b } ) ^ { 2 } u _ { T - 1 } ,\tag{3.7}
$$

where

$$
\widetilde J ( u _ { 0 } , \ldots , u _ { T - 2 } ) : = \mathbb { E } \left\| \theta _ { T - 1 } - \eta _ { T - 1 } g _ { T - 1 } ( \theta _ { T - 1 } ) - \theta ^ { \star } \right\| ^ { 2 } < \infty .
$$

Consequently, every local minimizer, and hence every global minimizer, of (3.4) satisfies $\rho ( u ^ { \star } ) = \rho _ { \mathrm { m a x } }$

Two aspects of the result are worth emphasizing. First, saturation is necessary not only for a global optimizer, but for every local minimizer. Second, proving it does not require understanding how the terminal objective varies with every component of the schedule. The explicit dependence on the final variance $u _ { T - 1 }$ is already sufficient: whenever the constraint has slack, reducing this single variance produces a feasible descent direction.

Theorem 3.2 therefore makes the certified privacy constraint active at every optimizer of the quadratic tracking problem: $\rho ( u ^ { \star } ) = \rho _ { \mathrm { m a x } }$ , with $\rho ( \cdot )$ as in (2.6). This equality concerns the composition certificate used in the optimization, which need not be the smallest valid privacy parameter of the terminal-release mechanism. The argument does not establish monotonicity in every earlier variance or saturation for arbitrary terminal losses.

This suggests using the per-step certified privacy charges themselves as decision variables. Setting as in (2.5) $q _ { t } = ( 2 u _ { t } ) ^ { - 1 }$ transforms the active resource constraint into the exact simplex condition

$$
\sum _ { t = 0 } ^ { T - 1 } q _ { t } = \rho _ { \operatorname* { m a x } } .\tag{3.8}
$$

The next subsection develops this reformulation and clarifies its relation to the original variance-based problem.

3.3. Privacy-charge reformulation and exact-budget simplex. Theorem 3.2 shows that, for the quadratic terminal tracking objective $L _ { \mathrm { t r } }$ in (3.6), every optimal variance schedule exhausts the available certified privacy budget. We now exploit this property to obtain an equivalent formulation of the control problem in which the privacy constraint has a simpler structure. The key idea is to use the privacy expenditure at each iteration, rather than the corresponding noise variance, as the optimization variable.

Recall that the variance $u _ { t } ~ > ~ 0$ determines the magnitude of the Gaussian perturbation introduced at iteration t, while its certified zCDP charge is given by $q _ { t } = ( 2 u _ { t } ) ^ { - 1 }$ . This relation is invertible, with $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ . Consequently, specifying the complete sequence of positive privacy charges $q = ( q _ { 0 } , \dots , q _ { T - 1 } )$ is equivalent to specifying the variance schedule $u = ( u _ { 0 } , \ldots , u _ { T - 1 } )$

The two parametrizations describe the same controlled dynamics, but they lead to different representations of the privacy constraint. Indeed, in the original variance variables, admissibility requires $\rho ( u ) \leq \rho _ { \mathrm { m a x } }$ . In terms of the privacy charges, this becomes the linear inequality $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } q _ { t } \leq \rho _ { \operatorname* { m a x } } ; } \end{array}$ the equality (3.8) holds at every optimizer of the quadratic tracking problem by Theorem 3.2. We may therefore restrict the search for optimal schedules to the exact-budget simplex

$$
Q ^ { \circ } : = \left\{ q \in ( 0 , \infty ) ^ { T } : \sum _ { t = 0 } ^ { T - 1 } q _ { t } = \rho _ { \operatorname* { m a x } } \right\} .
$$

This reformulation has an important practical interpretation. Instead of deciding how much noise to inject at each iteration while satisfying a nonlinear constraint, we directly decide how to distribute a fixed amount of certified privacy expenditure across the training horizon. Each coordinate $q _ { t }$ represents the portion of the available budget assigned to iteration $t ,$ and increasing this allocation reduces the corresponding noise variance.

The change of variables also modifies the expression of the controlled dynamics. Substituting $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ into (3.1), we obtain

$$
\theta _ { t + 1 } ( q ) = \theta _ { t } ( q ) - \eta _ { t } g _ { t } ( \theta _ { t } ( q ) ) - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { \sqrt { 2 q _ { t } } } \xi _ { t } , \qquad t \in [ [ T ] ] _ { 0 } .\tag{3.9}
$$

Although $Q ^ { \circ }$ provides the natural domain for the reformulated problem, it is not compact. In particular, sequences of admissible charges may approach the boundary $q _ { t } = 0$ , where the noise amplitude in (3.9) becomes unbounded. This creates difficulties for both the analysis and the numerical solution of the optimization problem. To avoid these difficulties, we introduce a small positive lower bound on the individual privacy charges. Given $0 < q _ { \mathrm { m i n } } < \rho _ { \mathrm { m a x } } / T$ , we define the compact simplex

$$
Q ( q _ { \operatorname* { m i n } } ) : = \left\{ q \in \mathbb { R } ^ { T } : q _ { t } \geq q _ { \operatorname* { m i n } } , \quad \sum _ { t = 0 } ^ { T - 1 } q _ { t } = \rho _ { \operatorname* { m a x } } \right\} .\tag{3.10}
$$

The condition on $q _ { \mathrm { m i n } }$ guarantees that this set is nonempty. Moreover, every admissible charge is bounded away from zero, so the noise coefficients in (3.9) remain finite and uniformly bounded over the feasible set. Writing $Q : = Q ( q _ { \mathrm { m i n } } )$ , we arrive at the compact optimization problem

$$
\operatorname* { m i n } _ { \boldsymbol { q } \in \boldsymbol { Q } } F ( \boldsymbol { q } ) , \qquad F ( \boldsymbol { q } ) : = \mathbb { E } \big [ L ( \theta _ { T } ( \boldsymbol { q } ) ) \big ] .\tag{3.11}
$$

The introduction of the floor $q _ { \mathrm { m i n } }$ is primarily a technical and computational device. It is therefore important to establish that this restriction does not alter the solutions of the original problem when the floor is chosen sufficiently small. The following proposition makes this equivalence precise.

Proposition 3.3. The change of variables $q _ { t } = ( 2 u _ { t } ) ^ { - 1 }$ bijectively maps $\mathcal { U } _ { a d } ( \rho _ { \mathrm { m a x } } )$ onto

$$
Q _ { \leq } : = \left\{ q \in ( 0 , \infty ) ^ { T } : \sum _ { t = 0 } ^ { T - 1 } q _ { t } \leq \rho _ { \operatorname* { m a x } } \right\} ,
$$

preserving the dynamics and objective. Assume $L = L _ { \mathrm { t r } }$ as in (3.6) and that the minimizer set of (3.4) is nonempty and compact. Then there exists $q _ { * } \in ( 0 , \rho _ { \operatorname* { m a x } } / T )$ such that, for every $0 < q _ { \mathrm { m i n } } < q _ { * }$ ，

$$
\underset { q \in Q ( q _ { \operatorname* { m i n } } ) } { \arg \operatorname* { m i n } } F ( q ) = \underset { q \in Q ^ { \circ } } { \arg \operatorname* { m i n } } F ( q ) = \left\{ \left( ( 2 u _ { t } ) ^ { - 1 } \right) _ { t = 0 } ^ { T - 1 } : u \in \underset { U _ { a d } } { \arg \operatorname* { m i n } } ( \rho _ { \operatorname* { m a x } } ) J \right\} .
$$

3.4. Differentiability, adjoints, and KKT conditions. To characterize optimal privacy allocations, we first study the sensitivity of the expected terminal loss with respect to the noise-variance schedule. To this end, for $t \in [ [ T ] ] _ { 0 }$ , we introduce the one-step state-transition Jacobian

$$
A _ { t } ( \theta ) : = I _ { d } - \eta _ { t } \nabla g _ { t } ( \theta ) .\tag{3.12}
$$

The following proposition establishes the differentiability of the objective and provides an adjoint representation of its gradient, which will subsequently be used to derive the KKT optimality conditions.

Proposition 3.4. Under Assumptions (A1), (A2), (A3), and $( A \theta )$ , the map J is continuously differentiable on $( 0 , \infty ) ^ { T }$ . For each realization ${ \boldsymbol { \xi } } = \left( { \xi } _ { 0 } , \ldots , { \xi } _ { T - 1 } \right)$ of the Gaussian innovations, let $\theta _ { t } = \theta _ { t } ( u , \xi )$ denote the corresponding state trajectory and define the pathwise adjoint by the backward recursion

$$
p _ { T } = \nabla L ( \theta _ { T } ) , \qquad p _ { t } = A _ { t } ( \theta _ { t } ) ^ { \top } p _ { t + 1 } , \quad t \in [ [ T ] ] _ { 0 } ^ {  } ,\tag{3.13}
$$

with $A _ { t } ( \cdot )$ given by (3.12). Then the pathwise objective $j ( u ; \xi ) : = L ( \theta _ { T } ( u ; \xi ) )$ satisfies

$$
\partial _ { u _ { t } } j ( u ; \xi ) = - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { 2 \sqrt { u _ { t } } } \left. p _ { t + 1 } , \xi _ { t } \right. , \qquad t \in [ [ T ] ] _ { 0 } ,\tag{3.14}
$$

and differentiation commutes with expectation:

$$
\partial _ { u _ { t } } J ( u ) = \mathbb { E } [ \partial _ { u _ { t } } j ( u ; \xi ) ] = - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { 2 \sqrt { u _ { t } } } \mathbb { E } \left[ \langle p _ { t + 1 } , \xi _ { t } \rangle \right] , \qquad t \in \mathbb { [ } T ] _ { 0 } .\tag{3.15}
$$

The adjoint representation established in Proposition 3.4 allows us to characterize the first-order optimality conditions for the privacy-allocation problem (3.4). Although the state dynamics are stochastic, the control is an open-loop schedule chosen before training and is therefore deterministic. Moreover, the objective $J ( u )$ is an expectation, so the optimization takes place over the finite-dimensional vector $u ,$ subject to the deterministic privacy-budget constraint. To formulate the corresponding optimality conditions, define

$$
\Phi ( u ) : = \rho ( u ) - \rho _ { \operatorname* { m a x } } , \qquad \mathcal { L } ( u , \lambda ) : = J ( u ) + \lambda \Phi ( u ) , \quad \lambda \geq 0 .\tag{3.16}
$$

Here, $\mathcal { L }$ is the reduced Lagrangian, obtained after incorporating the stochastic state dynamics into the objective J. Consequently, the only optimization variables are the deterministic schedule u and the scalar multiplier λ. The state and adjoint trajectories appearing in the gradient formula remain random, and their contribution to the stationarity conditions is therefore expressed through an expectation.

The following theorem combines this gradient representation with the privacy-budget constraint to obtain the KKT conditions for locally optimal schedules. In particular, the multiplier λ represents the marginal value of the available privacy budget, while the stationarity conditions characterize how the sensitivity of the terminal objective is balanced across iterations.

Theorem 3.5. Assume the hypotheses of Proposition $\ 3 . 4 \cdot$ Let $u ^ { \star } \in \mathcal { U } _ { a d } ( \rho _ { \operatorname* { m a x } } )$ be a local minimizer of (3.4), and let $\{ \theta _ { t } ( u ^ { \star } ) \} _ { t = 0 } ^ { T }$ and $\{ p _ { t } ^ { \star } \} _ { t = 0 } ^ { \bar { T } }$ be its state and pathwise adjoint, defined by (3.1) and (3.13), respectively. Then there exists a deterministic multiplier $\lambda ^ { \star } \geq 0$ such that $\lambda ^ { \star } \big ( \rho ( u ^ { \star } ) - \rho _ { \mathrm { m a x } } \big ) = 0$ and

$$
\eta _ { t } \Delta _ { t } ^ { b } ( u _ { t } ^ { \star } ) ^ { 3 / 2 } \mathbb { E } \left[ \left. p _ { t + 1 } ^ { \star } , \xi _ { t } \right. \right] + \lambda ^ { \star } = 0 , \quad \ f o r \ a l l \ t \in [ [ T ] ] _ { 0 } .\tag{3.17}
$$

Combining the budget-saturation property of Theorem 3.2 with the KKT conditions of Theorem 3.5, we obtain the following corollary, which establishes the strict positivity of the privacy multiplier and provides a more explicit characterization of optimal schedules for the quadratic terminal tracking objective.

Corollary 3.6. For $L = L _ { \mathrm { t r } } ,$ , every local minimizer satisfying the hypotheses of Theorem 3.5 uses the full budget and has ${ { \lambda } ^ { \star } } > 0$ . In fact,

$$
\partial _ { u _ { T - 1 } } J ( u ) = d \eta _ { T - 1 } ^ { 2 } ( \Delta _ { T - 1 } ^ { b } ) ^ { 2 } \qquad a n d \qquad \lambda ^ { \star } = 2 d \eta _ { T - 1 } ^ { 2 } ( \Delta _ { T - 1 } ^ { b } ) ^ { 2 } ( u _ { T - 1 } ^ { \star } ) ^ { 2 } .
$$

Consequently, $\mathbb { E } \left[ \left. p _ { t + 1 } ^ { \star } , \xi _ { t } \right. \right] < 0$ for every $t \in [ [ T ] ] _ { 0 }$ , and (3.17) may be written

$$
\boldsymbol { u } _ { t } ^ { \star } = \left( - \frac { \boldsymbol { \lambda } ^ { \star } } { \eta _ { t } \Delta _ { t } ^ { b } \mathbb { E } [ \left. p _ { t + 1 } ^ { \star } , \xi _ { t } \right. ] } \right) ^ { 2 / 3 } .\tag{3.18}
$$

The preceding results characterize optimal privacy allocation in terms of the noise variances. We now return to the privacy-charge formulation introduced in Section 3.3 and derive the corresponding adjoint gradient formula, which will be used in the exact-budget optimization method developed in Section 4.

For each realization ${ \boldsymbol \xi } = ( \xi _ { 0 } , \dots , \xi _ { T - 1 } )$ of the Gaussian innovations, let $\theta _ { t } = \theta _ { t } ( q ; \xi )$ denote the corresponding state trajectory and define the pathwise terminal objective by $f ( q ; \xi ) : = L ( \theta _ { T } ( q ; \xi ) )$ . The following proposition expresses the gradient with respect to the privacy charges in terms of this adjoint.

Proposition 3.7. Under the assumptions ofProposition 3.4, the functional F defined in (3.11) is continuously differentiable on an open neighborhood of Q, and

$$
\begin{array} { r l r l } & { \partial _ { q _ { t } } f ( q ; \xi ) = \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ) ^ { - 3 / 2 } \left. p _ { t + 1 } , \xi _ { t } \right. , \qquad } & & { t \in [ [ T ] _ { 0 } , } \\ & { \partial _ { q _ { t } } F ( q ) = \mathbb { E } \left[ \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ) ^ { - 3 / 2 } \left. p _ { t + 1 } , \xi _ { t } \right. \right] , \qquad } & & { t \in [ [ T ] _ { 0 } . } \end{array}\tag{3.19}
$$

where $p _ { t }$ is the adjoint state defined in (3.13).

The adjoint representation in Proposition 3.7 provides a practical way to evaluate the gradient of the expected terminal objective. Since the expectation cannot generally be computed exactly, we approximate

it by averaging the pathwise gradients over independent realizations of the Gaussian innovations. Given $M \in \mathbb { N }$ independent Gaussian paths, define the Monte Carlo gradient estimator

$$
h _ { t } ( \boldsymbol { q } ) : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ) ^ { - 3 / 2 } \langle p _ { t + 1 } ^ { ( m ) } , \xi _ { t } ^ { ( m ) } \rangle , \qquad t \in  { [ [ T ] ] } _ { 0 } .\tag{3.20}
$$

<sub>Here,</sub> <sub>p</sub>(m) $p ^ { ( m ) }$ denotes the adjoint associated with the m-th simulated trajectory, and

$$
h ( q ) : = ( h _ { 0 } ( q ) , \ldots , h _ { T - 1 } ( q ) ) \in \mathbb { R } ^ { T } .
$$

Two sampling strategies are possible, with different interpretations. If new Gaussian paths are generated independently of the current iterate $q ,$ the estimator is conditionally unbiased: $\mathbb { E } [ h ( q ) | q ] = \nabla F ( q )$ . Alternatively, the same Gaussian paths can be fixed and reused throughout the optimization. In this case, $h ( q )$ is the exact gradient of the corresponding sample-average objective $F _ { M } ( q )$ , introduced in Subsection 4.3, rather than an unbiased fresh-sample estimator of $\nabla F ( q )$

These two approaches provide the gradient information required for numerical privacy allocation: the first supports stochastic gradient-based optimization of the expected objective $F _ { ; }$ while the second leads to deterministic optimization of its sample-average approximation $F _ { M }$ . Both will be considered in the numerical framework developed in Section 4.

3.5. Well-posedness and stability of the control problem. Having established the first-order optimality conditions and adjoint gradient formulas, we now investigate the well-posedness of the privacy-allocation problem, the dependence of optimal schedules on the privacy budget, and their stability with respect to perturbations of the design model.

Concerning well-posedness, we prove existence of a minimizer by showing that $\mathcal { U } _ { \mathrm { a d } } ( \rho )$ is closed, J is lower semicontinuous, and coercivity bounds every minimizing sequence. These properties are established in Propositions B.1, B.3 and B.5, and yield the following result.

Theorem 3.8. Under Assumptions $( A 1 ) , ( A 2 ) , ( A 3 ) , ( A 4 )$ , and (A5), problem (3.4) admits at least one minimizer. Its minimizer set is compact.

Dependence on the privacy budget is obtained by proving continuity of the optimal value and outer stability of optimal schedules. For $\rho > 0$ , write

$$
V ( \rho ) : = \operatorname* { m i n } _ { u \in \mathcal { U } _ { \mathrm { a d } } ( \rho ) } J ( u ) \quad \mathrm { ~ a n d ~ } \quad S ( \rho ) : = \operatorname* { a r g m i n } _ { u \in \mathcal { U } _ { \mathrm { a d } } ( \rho ) } J ( u ) .
$$

We then have the following result.

Theorem 3.9. Under the assumptions of Theorem 3.8, the following hold.

(i) $V : ( 0 , \infty ) $ R is nonincreasing and continuous.

(ii) I $f \rho _ { n } \to \rho > 0$ and $u _ { n } ~ \in ~ S ( \rho _ { n } )$ , then $\{ u _ { n } \}$ is bounded and every cluster point belongs to $S ( \boldsymbol { \rho } )$ equivalently,

$$
\operatorname* { l i m } _ { n \to \infty } S ( \rho _ { n } ) \subset S ( \rho ) .
$$

(iii) If $S ( \rho ) = \{ u ^ { \star } \}$ is a singleton, then every sequence $u _ { n } \in S ( \rho _ { n } )$ converges to $u ^ { \star }$

To conclude, we compare the public design dynamics and target training dynamics under the same control and Gaussian innovations. For $a \in \{ \mathrm { d e s , t a r } \}$ , let $g _ { t } ^ { a }$ , initial state $\theta _ { 0 } ^ { a } \in \mathbb { R } ^ { d }$ , and reference $\theta ^ { * , a } \in \mathbb { R } ^ { d }$ specify model a. Both models use the same $\eta _ { t } , \Delta _ { t } ^ { b }$ , control, and Gaussian innovations:

$$
\theta _ { t + 1 } ^ { a } = \theta _ { t } ^ { a } - \eta _ { t } g _ { t } ^ { a } ( \theta _ { t } ^ { a } ) - \eta _ { t } \Delta _ { t } ^ { b } \sqrt { u _ { t } } \xi _ { t } , \qquad J ^ { a } ( u ) = \mathbb { E } \left. \theta _ { T } ^ { a } - \theta ^ { * , a } \right. ^ { 2 } .
$$

Theorem 3.10. Assume (A1). Let ${ \cal { K } } \subset ( 0 , \infty ) ^ { T }$ be nonempty and compact, and suppose all maps $g _ { t } ^ { a }$ $a \in \{ \mathrm { d e s , t a r } \}$ , are globally Lipschitz, with

$$
\varepsilon _ { t } ^ { g } : = \operatorname* { s u p } _ { \theta \in \mathbb { R } ^ { d } } \| g _ { t } ^ { \mathrm { t a r } } ( \theta ) - g _ { t } ^ { \mathrm { d e s } } ( \theta ) \| < \infty , \quad \ f o r \ a l l \ t \in [ [ T ] ] _ { 0 } .\tag{3.21}
$$

There is a finite constant $C _ { K }$ , depending only on the two models and $\kappa ,$ such that, with

$$
\varepsilon _ { \mathcal { K } } : = C _ { \mathcal { K } } \left( \| \theta _ { 0 } ^ { \mathrm { t a r } } - \theta _ { 0 } ^ { \mathrm { d e s } } \| + \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \varepsilon _ { t } ^ { g } + \| \theta ^ { * , \mathrm { t a r } } - \theta ^ { * , \mathrm { d e s } } \| \right) ,
$$

we have

$$
\operatorname* { s u p } _ { u \in { \mathcal K } } \vert J ^ { \mathrm { t a r } } ( u ) - J ^ { \mathrm { d e s } } ( u ) \vert \le \varepsilon _ { { \mathcal K } } \qquad a n d \qquad \left. \operatorname* { m i n } _ { { \mathcal K } } J ^ { \mathrm { t a r } } - \operatorname* { m i n } _ { { \mathcal K } } J ^ { \mathrm { d e s } } \right. \le \varepsilon _ { { \mathcal K } } .
$$

Moreover, every γ-minimizer $\widehat { \boldsymbol { u } } \in \mathcal { K } \ o f \ J ^ { \mathrm { d e s } }$ , meaning $J ^ { \mathrm { d e s } } ( \widehat { u } ) \leq \operatorname* { m i n } _ { \mathcal { K } } J ^ { \mathrm { d e s } } + \gamma$ , satisfies

$$
J ^ { \mathrm { t a r } } ( \widehat { u } ) \leq \operatorname* { m i n } _ { u \in \mathcal { K } } J ^ { \mathrm { t a r } } ( u ) + \gamma + 2 \varepsilon _ { \mathcal { K } } .
$$

Theorem 3.10 applies, in particular, to the exact-budget privacy allocations considered in this paper. Indeed, the constraints $\textstyle \sum _ { t = 0 } ^ { T - { \bar { 1 } } } q _ { t } = \rho _ { \operatorname* { m a x } }$ and $q _ { t } \ge q _ { \mathrm { m i n } } > 0$ ensure that the corresponding noise variances $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ belong to a compact set K. The theorem therefore provides a uniform stability estimate over all such admissible schedules.

The result can also be applied to the actual training dynamics. Conditional on a fixed dataset and the public batch sequence, we may take $g _ { t } ^ { \mathrm { t a r } } = g _ { t } ^ { D }$ , provided that the required global Lipschitz assumption and the uniform discrepancy bound (3.21) are satisfied. This allows us to quantify how discrepancies between the fixed design model and the actual training dynamics affect the terminal outcome. Finally, if estimating the model discrepancy or selecting and releasing the schedule involves accessing private data, the corresponding privacy expenditure must be accounted for separately in the private selection stage.

## 4. Privacy-budget adjoint mirror descent

We now develop an optimization method for the exact-budget problem (3.11). We use the adjoint gradient in Proposition 3.7 to determine how reallocating the per-step privacy charges changes the terminal objective to first order. We construct Privacy-Budget Adjoint Mirror Descent (PB-AMD) using multiplicative entropic mirror updates and scalar normalization to preserve positivity and enforce the prescribed budget exactly at every iteration. We then establish its deterministic, stochastic, and sample-average convergence guarantees, with complete proofs in Appendix C.

4.1. PB-AMD: construction and algorithm. Throughout this section, we construct and analyze PB-AMD on the compact exact-budget simplex introduced in (3.10), namely,

$$
Q ( q _ { \mathrm { m i n } } ) = \left\{ q \in \mathbb { R } ^ { T } : q _ { t } \geq q _ { \mathrm { m i n } } , \ : \sum _ { t = 0 } ^ { T - 1 } q _ { t } = \rho _ { \mathrm { m a x } } \right\} .
$$

The floor $q _ { \mathrm { m i n } } > 0$ keeps the computation away from the singular boundary $q _ { t } = 0$ , which would correspond to infinite noise variance. As shown in Proposition 3.3, every sufficiently small floor leaves the global minimizer set of the unfloored exact-budget problem unchanged.

To compare the benefit of assigning additional privacy charge to different training iterations, we define $m _ { t } ( q ) : = - \partial _ { q _ { t } } F ( q )$ as the marginal utility of certified privacy expenditure at iteration t. A large $m _ { t } ( q )$ means that assigning more charge to that iteration, and hence reducing its noise, gives a large first-order decrease in terminal loss. The KKT conditions then express the required balance of these marginal utilities under the budget and floor constraints: there are $\lambda \in \mathbb { R }$ and $\mu _ { t } \geq 0$ such that

$$
\partial _ { q _ { t } } F ( q ^ { \star } ) + \lambda - \mu _ { t } = 0 , \qquad \mu _ { t } ( q _ { t } ^ { \star } - q _ { \operatorname* { m i n } } ) = 0 , \qquad t \in [ [ T ] ] _ { 0 } .
$$

Consequently,

$$
m _ { t } ( q ^ { \star } ) = \lambda \quad \mathrm { i f ~ } q _ { t } ^ { \star } > q _ { \operatorname* { m i n } } \quad \mathrm { ~ a n d ~ } \quad m _ { t } ( q ^ { \star } ) \leq \lambda \quad \mathrm { i f ~ } q _ { t } ^ { \star } = q _ { \operatorname* { m i n } } .
$$

Thus, if $q _ { s } , q _ { t } > q _ { \operatorname* { m i n } }$ and $m _ { t } ( q ) > m _ { s } ( q )$ , a sufficiently small transfer $q _ { s } \mapsto q _ { s } - \delta , q _ { t } \mapsto q _ { t } + \delta$ with $\delta > 0$ preserves feasibility in Q and decreases the objective $F$ in (3.11) to first order. To implement this budget redistribution through multiplicative mirror updates on $Q$ in (3.10), we use the negative entropy

$$
\psi ( q ) : = \sum _ { t = 0 } ^ { T - 1 } q _ { t } \log q _ { t } ,
$$

whose Bregman divergence is

$$
D _ { \psi } ( z , q ) : = \sum _ { t = 0 } ^ { T - 1 } \left[ z _ { t } \log \biggl ( \frac { z _ { t } } { q _ { t } } \biggr ) - z _ { t } + q _ { t } \right] .
$$

Because $q _ { t } , z _ { t } \geq q _ { \operatorname* { m i n } } > 0 .$ , this divergence is finite and smooth on all of $Q \times Q$ . At an iterate $q ^ { k } \in Q$ , given a gradient estimate $h ^ { k } \in \mathbb { R } ^ { T }$ and a step size $\alpha _ { k } > 0$ , PB-AMD performs

$$
q ^ { k + 1 } : = \underset { q \in Q } { \arg \operatorname* { m i n } } \left\{ \langle h ^ { k } , q \rangle + \frac { 1 } { \alpha _ { k } } D _ { \psi } ( q , q ^ { k } ) \right\} .\tag{4.1}
$$

The next proposition shows that the mirror-descent step admits an explicit multiplicative representation, with a scalar normalization that enforces the exact privacy budget while respecting the lower bounds on individual charges.

Proposition 4.1. The minimizer in (4.1) is unique and has the form

$$
q _ { t } ^ { k + 1 } = \operatorname* { m a x } \left\{ q _ { \operatorname* { m i n } } , \ \tau _ { k } q _ { t } ^ { k } e ^ { - \alpha _ { k } h _ { t } ^ { k } } \right\} , \qquad t \in [ [ T ] ] _ { 0 }\tag{4.2}
$$

where $\tau _ { k } > 0$ is the unique scalar satisfying

$$
\sum _ { t = 0 } ^ { T - 1 } \operatorname* { m a x } \Big \{ q _ { \operatorname* { m i n } } , ~ \tau _ { k } q _ { t } ^ { k } e ^ { - \alpha _ { k } h _ { t } ^ { k } } \Big \} = \rho _ { \operatorname* { m a x } } .\tag{4.3}
$$

In particular, every iterate belongs to $Q ,$ , and a coordinate at the lower bound can return to the interior at a later iteration.

In the update (4.2) given by Proposition 4.1, coordinates with smaller $h _ { t } ^ { k }$ , equivalently larger estimated marginal privacy utilities, receive larger multiplicative factors ${ { e } ^ { - \alpha _ { k } { h _ { t } ^ { k } } } }$ than coordinates with larger $h _ { t } ^ { k }$ . The scalar $\tau _ { k }$ then enforces

$$
\sum _ { t = 0 } ^ { T - 1 } q _ { t } ^ { k + 1 } = \rho _ { \mathrm { m a x } }
$$

through (4.3), and the floor $q _ { \mathrm { m i n } }$ excludes the singular limit of zero charge (infinite noise variance).

We combine this budget update with the adjoint gradient estimator (3.20) to obtain the complete PB-AMD procedure in Algorithm 1, which supports both fresh-path and fixed-path sample-average approximation (SAA) modes. Figure 2 summarizes this procedure.

![](images/17f3c15dc68ea8aca6cfa4e1304a351304096b10a46ac090984a6ab6e34c6979.jpg)  
Figure 2. Workflow of PB-AMD.

Each PB-AMD iteration consists of a forward simulation of (3.9), a backward adjoint sweep using (3.13), and the multiplicative budget update (4.2) with normalization (4.3). For one Gaussian path, the forwardadjoint sweep costs $O ( T )$ evaluations of $g _ { t }$ and Jacobian-transpose products and stores $O ( T d )$ states. The normalization equation (4.3) for $\tau _ { k }$ is a one-dimensional monotone solve, so exact budget enforcement adds little cost relative to the forward-backward sweep. For comparison, a penalty formulation with $\beta > 0$ would solve

$$
\operatorname* { m i n } _ { u \in ( 0 , \infty ) ^ { T } } J ( u ) + \frac { \beta } { 2 } [ \Phi ( u ) ] _ { + } ^ { 2 } , \qquad [ x ] _ { + } : = \operatorname* { m a x } \{ x , 0 \} .
$$

For finite $\beta ,$ this surrogate need not satisfy the budget constraint $\Phi ( u ) \le 0$ in (3.16) and introduces an additional parameter to tune. PB-AMD instead remains on the exact-budget simplex $Q$ in (3.10) at every iterate. Section 5 compares this exact-simplex approach with projected-gradient and penalty-based alternatives.

Algorithm 1. Privacy-Budget Adjoint Mirror Descent (PB-AMD)   
Input: Horizon $T ,$ iteration count $K \in \mathbb N ,$ budget $\rho _ { \mathrm { m a x } } ,$ floor $q _ { \mathrm { m i n } }$ with $T q \mathrm { { m i n } } < \rho \mathrm { { m a x } } ,$ , sampling mode mode ∈   
{fresh, SAA}, path counts M or $M _ { k }$ , step sizes $\alpha _ { k } > 0 ;$ and $q ^ { 0 } \in Q$   
1: if mode $= { \mathsf { S A A } }$ then   
2: Draw and store M Gaussian paths and set $M _ { k } = M$ for all k.   
3: end if   
4: for $k = 0 , \ldots , K - 1$ do   
5: if mode = fresh then   
6: Draw $M _ { k }$ paths with $\xi _ { t } ^ { ( m ) } \sim \mathcal { N } ( 0 , I _ { d } )$ independently over t and $m .$   
7: end if   
8: for $m = 1 , \ldots , M _ { k }$ do   
9: Set $\theta _ { 0 } ^ { ( m ) } = \bar { \theta } _ { 0 }$ and, for $t \in [ [ T ] ] _ { 0 } ,$ , use (3.9) to compute $\theta _ { t + 1 } ^ { ( m ) } = \theta _ { t } ^ { ( m ) } - \eta _ { t } g _ { t } ( \theta _ { t } ^ { ( m ) } ) - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { \sqrt { 2 q _ { t } ^ { k } } } \xi _ { t } ^ { ( m ) }$   
10: Using (3.13), set $p _ { T } ^ { ( m ) } = \nabla L ( \theta _ { T } ^ { ( m ) } )$ and, for $t \in \mathbb { \mathbb { T } } \mathbb { \mathrm { l } } _ { 0 } ^ {  }$ , compute $\begin{array} { r } { p _ { t } ^ { ( m ) } = \left( I _ { d } - \eta _ { t } \nabla g _ { t } ( \theta _ { t } ^ { ( m ) } ) \right) ^ { \top } p _ { t + 1 } ^ { ( m ) } . } \end{array}$   
11: end for   
12: For each $t \in [ [ T ] ] _ { 0 }$ , evaluate the gradient estimator in (3.20) as $h _ { t } ^ { k } = \frac { 1 } { M _ { k } } \sum _ { \ldots 1 } ^ { M _ { k } } \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ^ { k } ) ^ { - 3 / 2 } \left. p _ { t + 1 } ^ { ( m ) } , \xi _ { t } ^ { ( m ) } \right.$   
13: Find the unique $\tau _ { k } > 0$ solving (4.3), for example by bisection in the log domain.   
14: For each $t \in [ [ T ] ] _ { 0 }$ , apply (4.2) to obtain $q _ { t } ^ { k + 1 } = \operatorname* { m a x } \left\{ q _ { \operatorname* { m i n } } , \tau _ { k } q _ { t } ^ { k } e ^ { - \alpha _ { k } h _ { t } ^ { k } } \right\}$   
15: end for   
Output: Choose $\widehat { k } \in \{ 0 , \ldots , K \}$ by a prespecified rule that uses no private training data, and return $q ^ { \widehat { k } } \in Q$ and   
$\widehat { u _ { t } ^ { k } } = ( 2 \widehat { q _ { t } ^ { k } } ) ^ { - 1 }$ for $t \in \mathbb { I } \mathbb { T } \mathbb { I } _ { 0 } ,$ using (2.5).

4.2. Convergence guarantees. Having constructed PB-AMD, we now establish first-order stationarity guarantees for its deterministic and stochastic implementations. We base the analysis on three properties of the reduced optimization problem: differentiability and lower boundedness of the objective, smoothness relative to the entropy geometry, and control of the error introduced by the Monte Carlo gradient estimator.

We first recall a useful property of the entropy geometry on the compact simplex Q defined in (3.10). Since the total privacy charge is fixed at $\rho _ { \mathrm { m a x } }$ and each coordinate is bounded below by $q _ { \mathrm { m i n } } .$ , the largest admissible value of any individual charge is $q _ { \operatorname* { m a x } } : = \rho _ { \operatorname* { m a x } } - ( T - 1 ) q _ { \operatorname* { m i n } }$ . Moreover, because every $q \in Q$ satisfies $q _ { t } \in [ q _ { \operatorname* { m i n } } , q _ { \operatorname* { m a x } } ]$ , the Hessian of the negative entropy obeys $q _ { \mathrm { m a x } } ^ { - 1 } I _ { T } \preceq \nabla ^ { 2 } \psi ( q ) \preceq q _ { \mathrm { m i n } } ^ { - 1 } I _ { T }$ . Consequently, the associated Bregman divergence is uniformly comparable to the squared Euclidean distance:

$$
\frac { \mu _ { \psi } } { 2 } \left\| z - q \right\| _ { 2 } ^ { 2 } \leq D _ { \psi } ( z , q ) \leq \frac { M _ { \psi } } { 2 } \left\| z - q \right\| _ { 2 } ^ { 2 } , \quad \mathrm { ~ w i t h ~ } \mu _ { \psi } : = \frac { 1 } { q _ { \mathrm { m a x } } } , \quad M _ { \psi } : = \frac { 1 } { q _ { \mathrm { m i n } } } .\tag{4.4}
$$

This comparison allows us to relate the entropy geometry underlying the mirror-descent updates to standard smoothness and stationarity estimates. We now state the three properties required for the convergence analysis.

(B1) (Differentiability and lower boundedness). The objective F is continuously differentiable on a neighborhood of Q and satisfies $F ( q ) \geq F _ { \mathrm { i n f } } > - \infty$ for all $q \in Q$

(B2) (Relative smoothness). There exists $L _ { F } \geq 0$ such that

$$
F ( z ) \leq F ( q ) + \langle \nabla F ( q ) , z - q \rangle + L _ { F } D _ { \psi } ( z , q ) , \qquad q , z \in Q .
$$

(B3) (Unbiasedness and bounded variance of the Monte Carlo gradient). Let $\mathcal { F } _ { k }$ denote the sigma-field generated by the initial point, all iterates, and all oracle randomness revealed before the fresh Monte Carlo batch at iteration k is drawn. In particular, $q ^ { k }$ is $\mathcal { F } _ { k }$ -measurable. For fresh Monte Carlo paths independent

of $\mathcal { F } _ { k }$ , the gradient estimator $h ^ { k }$ satisfies $\mathbb { E } [ h ^ { k } \mid { \mathcal { F } } _ { k } ] = \nabla F ( q ^ { k } )$ and

$$
\mathbb { E } \left[ \Vert h ^ { k } - \nabla F ( q ^ { k } ) \Vert _ { 2 } ^ { 2 } \middle | \mathcal { F } _ { k } \right] \leq \frac { \nu ^ { 2 } } { M _ { k } } .
$$

These properties provide the analytical framework for the convergence results developed below. Proposition C.1 verifies (B1), (B2), and (B3) for fresh independent path sampling, under Assumptions (A1), (A2), (A3), (A6), and (A7). Thus, this verification uses the second-order regularity assumption (A7), in addition to the assumptions required for the adjoint gradient.

Because $F$ is generally nonconvex, the relevant convergence target is first-order stationarity rather than global objective suboptimality. We measure stationarity in the same entropy geometry used by the algorithm. For $\alpha > 0$ , define the exact mirror step and residual by

$$
T _ { \alpha } ( \boldsymbol { q } ) : = \underset { \boldsymbol { z } \in Q } { \arg \operatorname* { m i n } } \left\{ \langle \nabla F ( \boldsymbol { q } ) , \boldsymbol { z } \rangle + \frac { 1 } { \alpha } D _ { \psi } ( \boldsymbol { z } , \boldsymbol { q } ) \right\} ,\tag{4.5}
$$

$$
R _ { \alpha } ( q ) : = \frac { 1 } { \alpha ^ { 2 } } D _ { \psi } ( T _ { \alpha } ( q ) , q ) .\tag{4.6}
$$

The map $T _ { \alpha }$ is continuous, and the residual vanishes exactly at points satisfying the KKT variational inequality:

$$
R _ { \alpha } ( q ) = 0 \quad \Leftrightarrow \quad \langle \nabla F ( q ) , z - q \rangle \geq 0 \quad { \mathrm { f o r ~ e v e r y ~ } } z \in Q .\tag{4.7}
$$

Having established the regularity properties required for the analysis, we now investigate the convergence of PB-AMD. The following theorem provides first-order stationarity guarantees for both the deterministic method, based on exact gradients, and its stochastic counterpart, based on fresh Monte Carlo estimates.

Theorem 4.2. For the fresh-path update, assume $( B 1 ) _ { - } ( B 3 )$ , let $q ^ { 0 } \in Q$ , use (4.1) with $\alpha _ { k } \equiv \alpha > 0$ satisfying α $\iota L _ { F } \leq 1 / 2$ , and let R be uniform on $[ [ K ] ] _ { 0 }$ independently of the algorithm. Then

$$
\mathbb { E } R _ { \alpha } ( q ^ { R } ) \leq \frac { 4 M _ { \psi } } { \mu _ { \psi } } \frac { F ( q ^ { 0 } ) - F _ { \mathrm { i n f } } } { \alpha K } + \frac { 5 M _ { \psi } \nu ^ { 2 } } { \mu _ { \psi } ^ { 2 } } \left( \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \frac { 1 } { M _ { k } } \right) .\tag{4.8}
$$

Thus a constant batch size gives an $O ( K ^ { - 1 } + M ^ { - 1 } )$ expected-stationarity bound. More generally,

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \frac { 1 } { M _ { k } } \to 0 \quad \Rightarrow \quad { \mathbb E } [ R _ { \alpha } ( q ^ { R } ) ] \to 0 .
$$

In the exact-gradient case, assume only (B1)-(B2) and $0 < \alpha L _ { F } < 1$ . Then

$$
\operatorname* { m i n } _ { 0 \leq k \leq K - 1 } R _ { \alpha } ( q ^ { k } ) \leq \frac { F ( q ^ { 0 } ) - F _ { \mathrm { i n f } } } { \alpha ^ { 2 } ( 1 / \alpha - L _ { F } ) K } ,\tag{4.9}
$$

$R _ { \alpha } ( q ^ { k } ) \to 0$ , and every cluster point of $\{ q ^ { k } \}$ satisfies (4.7).

Theorem 4.2 shows how the convergence of PB-AMD depends on both the number of optimization iterations and the accuracy of the Monte Carlo gradient estimates. When exact gradients are available, (4.9) gives an $O ( K ^ { - 1 } )$ bound on $\mathrm { { m i n } } _ { 0 \leq k < K } { \cal R } _ { \alpha } ( q ^ { k } )$ . With stochastic gradients, an additional error arises from the finite number of Monte Carlo paths used at each iteration, as quantified by (4.8). In particular, if the batch size M is fixed, this contribution remains of order $O ( M ^ { - 1 } )$ , regardless of how many optimization iterations are performed. Thus, the theoretical bound does not guarantee convergence to a stationary point of the population objective with a fixed batch size. To obtain asymptotic stationarity, the batch sizes must instead increase sufficiently fast to ensure

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \frac { 1 } { M _ { k } } \to 0 \quad \mathrm { ~ a s ~ } \quad K \to \infty .
$$

4.3. Fixed-path/SAA implementation. As an alternative to drawing fresh Monte Carlo paths at every iteration, we now sample a fixed collection of Gaussian paths and reuse them throughout the optimization. With these paths fixed, we obtain deterministic and reproducible PB-AMD iterations, but optimize the sample-average objective $F _ { M }$ rather than the population objective F. We therefore investigate whether approximate stationarity for $F _ { M }$ also implies approximate stationarity for F. To this end, for M independent Gaussian paths $\xi ^ { ( 1 ) } , \dots , \xi ^ { ( M ) }$ , we define

$$
F _ { M } ( q ) : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } f ( q ; \xi ^ { ( m ) } ) .\tag{4.10}
$$

Here, $f ( q ; \xi ) = L ( \theta _ { T } ( q ; \xi ) )$ is the pathwise terminal loss.

If the maps $g _ { t }$ and the terminal loss L are twice continuously differentiable, the state recursion (3.9) ensures that $F _ { M }$ has a Lipschitz-continuous gradient on the compact set $Q .$ . Consequently, the exact-gradient convergence result of Theorem 4.2 applies to $F _ { M }$ , provided that the step size satisfies its condition with the relative-smoothness constant of the realized sample-average objective.

To relate the stationarity of the two objectives, we introduce a measure that depends directly on their gradients. For any continuously differentiable function H on a neighborhood of $Q ,$ define the variational gap

$$
G _ { H } ( q ) : = \operatorname* { m a x } _ { z \in Q } \left. \nabla H ( q ) , q - z \right. .\tag{4.11}
$$

We also denote by $T _ { \alpha , H }$ and $R _ { \alpha , H }$ the mirror step and its residual obtained by replacing F with H in $( 4 . 5 ) \substack { - ( 4 . 6 ) }$ . Both $G _ { H } ( q ) = 0$ and $R _ { \alpha , H } ( q ) = 0$ characterize the KKT condition for minimizing H over $Q$

The advantage of the variational gap $G _ { H }$ in (4.11) is that it allows us to compare the stationarity of $F _ { M }$ and F through the difference between their gradients. The following proposition makes this connection precise, providing the basis for transferring the stationarity guarantees obtained for the sample-average problem to the original population objective.

Proposition 4.3. Let $\begin{array} { r } { \varepsilon _ { M } : = \operatorname* { s u p } _ { q \in Q } \left. \nabla F _ { M } ( q ) - \nabla F ( q ) \right. _ { \infty } , D _ { Q , r } : = \operatorname* { m a x } _ { q , z \in Q } \left. q - z \right. _ { r } \left( r = 1 , 2 \right) } \end{array}$ , and $\begin{array} { r } { B _ { F } : = \operatorname* { m a x } _ { q \in Q } \| \nabla F ( q ) \| _ { 2 } } \end{array}$ . Suppose that $F$ and $F _ { M }$ are continuously differentiable on a neighborhood of Q. $H q _ { M } \in Q$ satisfies $R _ { \alpha , F _ { M } } ( q _ { M } ) \leq r _ { M }$ for some $\alpha , r _ { M } > 0$ , then

$$
G _ { F } ( q _ { M } ) \leq \Big [ \alpha \big ( B _ { F } + \sqrt { T } \varepsilon _ { M } \big ) + M _ { \psi } D _ { Q , 2 } \Big ] \sqrt { \frac { 2 r _ { M } } { \mu _ { \psi } } } + D _ { Q , 1 } \varepsilon _ { M } .\tag{4.12}
$$

Consequently, for any fixed $\alpha > 0 , i f r _ { M }  0$ and $\varepsilon _ { M } \to 0$ , then $G _ { F } ( q _ { M } )  0$ . The convergence holds in the same deterministic, almost-sure, or in-probability sense as the convergence $o f \varepsilon _ { M }$ , provided $r _ { M } \to 0$ deterministically.

Corollary 4.4. Under the assumptions of Proposition $C . 1 ,$ for independent Gaussian paths,

$$
\operatorname* { s u p } _ { q \in Q } \| \nabla F _ { M } ( q ) - \nabla F ( q ) \| _ { \infty } \to 0 \qquad a l m o s t \ s u r e l y .
$$

For any fixed $\alpha > 0$ , deterministic $r _ { M } \downarrow 0 _ { ; }$ , and possibly sample-dependent outputs $q _ { M } \in Q , R _ { \alpha , F _ { M } } ( q _ { M } ) \leq$ $r _ { M }$ almost surely implies $G _ { F } ( q _ { M } )  0$ almost surely. Moreover, $i f \alpha \mathbb { E } [ \Gamma ] < \mu _ { \psi . }$ , with Γ as in (C.1), then α lies below the relative-smoothness threshold of $F _ { M }$ for all sufficiently large $M ,$ almost surely, and fixed-path PB-AMD can attain any prescribed positive residual tolerance.

Proposition 4.3 quantifies how stationarity for the sample-average objective $F _ { M }$ transfers to the population objective $F .$ . The bound (4.12) separates two sources of error: the optimization residual tolerance $r _ { M }$ in $R _ { \alpha , F _ { M } } ( q _ { M } ) \leq r _ { M }$ , which measures how accurately PB-AMD solves the sample-average problem, and the sampling error $\varepsilon _ { M }$ , which measures the uniform discrepancy between the gradients of $F _ { M }$ and F. In particular, for a fixed step size, both errors must vanish to guarantee population stationarity. Corollary 4.4 establishes that, under the stated regularity assumptions, the sampling error converges to zero almost surely as the number of independent Gaussian paths increases. It also shows that a fixed step size satisfying the population-based condition is eventually admissible for the sample-average problems. Together, these results justify the fixed-path implementation of PB-AMD: by increasing the number of sampled paths and reducing the optimization residual, one obtains asymptotically stationary solutions of the original population problem while retaining the deterministic and reproducible character of each sample-average optimization.

4.4. Linear-quadratic model: what determines the optimal schedule? Finally, we consider a Linear-Quadratic (LQ) reduction of the dynamics to understand what determines the temporal shape of an optimal privacy schedule. We use this reduction only for interpretation and apply PB-AMD itself to the original nonlinear design dynamics. In the LQ model, we explicitly propagate the effect of noise injected at each iteration to the terminal criterion.

Let $\delta _ { t } : = \theta _ { t } - \bar { \theta } _ { t }$ . For $t \in [ [ T ] ] _ { 0 }$ , the first-order linearization of the deterministic step around the reference trajectory uses $\mathsf { A } _ { t } : = I _ { d } - \eta _ { t } \nabla g _ { t } ( \bar { \theta } _ { t } )$ . More generally, the analysis below applies to any deterministic matrices $\mathsf { A } _ { t } \in \mathbb { R } ^ { d \times d }$ . For $t \in [ [ T ] ] _ { 0 }$ , consider

$$
\delta _ { t + 1 } = \mathsf { A } _ { t } \delta _ { t } - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { \sqrt { 2 q _ { t } } } \xi _ { t } , \qquad \delta _ { 0 } = 0 ,
$$

with terminal loss $L ( \theta _ { T } ) = \delta _ { T } ^ { \top } W _ { T } \delta _ { T }$ , where $W _ { T } \in \mathbb { R } ^ { d \times d }$ and $W _ { T } = W _ { T } ^ { \top } \succeq 0$ . For $0 \leq s \leq t \leq T$ , define the transition matrices by $\mathsf { P } _ { s , t } : = \mathsf { A } _ { t - 1 } \cdot \cdot \cdot \mathsf { A } _ { s }$ for $s < t$ and $\mathsf { P } _ { t , t } : = I _ { d }$ . Then

$$
\delta _ { T } = - \sum _ { t = 0 } ^ { T - 1 } \mathsf { P } _ { t + 1 , T } \frac { \eta _ { t } \Delta _ { t } ^ { b } } { \sqrt { 2 q _ { t } } } \xi _ { t } .
$$

Expanding the quadratic form gives a double sum over $( s , t )$ . For $s \neq t ,$ , independence and zero mean imply

$$
\mathbb { E } \left[ \xi _ { s } ^ { \top } \mathsf { P } _ { s + 1 , T } ^ { \top } W _ { T } \mathsf { P } _ { t + 1 , T } \xi _ { t } \right] = 0 ,
$$

whereas $\mathbb { E } [ \xi _ { t } \xi _ { t } ^ { \top } ] = I _ { d }$ yields the diagonal trace terms. Therefore

$$
F ( q ) = \sum _ { t = 0 } ^ { T - 1 } \frac { a _ { t } } { q _ { t } } , \qquad a _ { t } : = \frac { \eta _ { t } ^ { 2 } ( \Delta _ { t } ^ { b } ) ^ { 2 } } { 2 } \operatorname { T r } \big ( \mathsf { P } _ { t + 1 , T } ^ { \top } W _ { T } \mathsf { P } _ { t + 1 , T } \big ) \ge 0 .\tag{4.13}
$$

The coefficient $a _ { t }$ summarizes the terminal sensitivity of noise injected at iteration t: $\eta _ { t }$ and $\Delta _ { t } ^ { b }$ set the perturbation scale, while $\mathsf { P } _ { t + 1 , T }$ propagates that perturbation through the remaining dynamics. Large $a _ { t }$ therefore means that noise at iteration t is more costly for the terminal objective.

Proposition 4.5. If all $a _ { t } = 0$ , every point of Q minimizes F over $Q$ in (4.13). Otherwise, the minimizer $q ^ { \star }$ is unique and satisfies

$$
q _ { t } ^ { \star } = \operatorname* { m a x } \left\{ q _ { \operatorname* { m i n } } , \sqrt { \frac { a _ { t } } { \lambda ^ { \star } } } \right\} , \qquad \sum _ { t = 0 } ^ { T - 1 } q _ { t } ^ { \star } = \rho _ { \operatorname* { m a x } } ,\tag{4.14}
$$

where ${ { \lambda } ^ { \star } } > 0$ is uniquely determined by the budget equality. $I f { \ : } q _ { t } ^ { \star } > q _ { \mathrm { m i n } }$ for every t, then

$$
q _ { t } ^ { \star } = \rho _ { \operatorname* { m a x } } \sqrt { a _ { t } } \left( \sum _ { s = 0 } ^ { T - 1 } \sqrt { a _ { s } } \right) ^ { - 1 } .\tag{4.15}
$$

In the interior case, (4.15) gives $q _ { t } ^ { \star } \propto \sqrt { a _ { t } }$ . Thus the optimizer has no intrinsic early- or late-time preference: it allocates more privacy, and hence less noise, where downstream propagation makes perturbations more consequential. Changes in the learning rates $\eta _ { t }$ or state-transition Jacobians $\mathsf { A } _ { t }$ can therefore shift the optimal charges $q _ { t } ^ { \star }$ across time, a mechanism examined numerically in Section $5 .$

Corollary 4.6. Assume that at least one $a _ { t }$ is positive, and set

$$
L _ { \mathrm { L Q } } : = 2 q _ { \mathrm { m i n } } ^ { - 2 } \operatorname* { m a x } _ { t \in \{ 0 , \ldots , T - 1 \} } a _ { t } .
$$

If exact-gradient PB-AMD uses $0 < \alpha \leq 1 / L _ { \mathrm { L Q } }$ , then

$$
F \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } q ^ { k } \right) - F ( q ^ { \star } ) \leq \frac { D _ { \psi } ( q ^ { \star } , q ^ { 0 } ) } { \alpha K } ,\tag{4.16}
$$

and the raw iterates converge to the unique optimizer: $q ^ { k } \to q ^ { \star }$

## 5. Numerical experiments

In this section, we evaluate how the control-theoretic framework improves terminal tracking and prediction, and how efficiently PB-AMD solves the exact-budget problem. We optimize privacy charges in (3.11) using the quadratic loss (3.6) and evaluate all schedules through the training dynamics (3.9). Allocation-rule comparisons assess tracking and prediction; solver comparisons assess computational efficiency at a common first-order tolerance. All the code employed in our experiments is available on GitHub [7].

5.1. Experimental design and comparison methods. The experiments address four questions: whether privacy allocation adapts to the training dynamics (Section 5.2), whether optimizing through the full noisy nonlinear dynamics improves on simpler allocation rules (Section 5.3), whether tracking gains translate into better prediction (Section 5.4), and how efficiently PB-AMD solves the allocation problem (Section 5.5). Moreover, we distinguish allocation-rule comparisons, which assess tracking and prediction, from solver comparisons, which assess computational cost for the same optimization problem.

Learning tasks and privacy budgets. We consider the four learning tasks summarized in Table 4. The Fashion-MNIST experiments use the dataset of [36].
<table><tr><td>Task</td><td>Model and initialization</td><td>Parameters trained and perturbed</td><td>Learning rate</td></tr><tr><td>Tabular</td><td>Regularized logistic regression on generated data with correlated features and nonlinear feature interactions in the data-generating model</td><td>All model parameters</td><td>Constant</td></tr><tr><td>Fashion-constant</td><td>Softplus classifier on Fashion-MNIST, initialized randomly</td><td>All network parameters</td><td>Constant</td></tr><tr><td>Fashion-restart</td><td>Softplus classifier on Fashion-MNIST, pretrained on public data</td><td>All network parameters</td><td>Increased at a prescribed restart</td></tr><tr><td>Nonlinear head</td><td>Randomly initialized Softplus classifier applied to features from an extractor pretrained on public Fashion-MNIST data</td><td>Classification head only; the extractor is frozen</td><td>Constant</td></tr></table>

Table 4. Learning tasks, initialization, trainable parameters, and learning-rate settings.

Tabular and Fashion-constant test allocation without learning-rate changes, whereas Fashion-restart tests adaptation to a learning-rate increase. Nonlinear head tests the additional value of optimizing through noisy nonlinear dynamics when learning-rate scaling alone gives a uniform allocation. Appendix D.1 specifies the data sizes, architectures, training parameters, and Gaussian path counts.

All allocation rules are compared on the exact-budget simplex Q in (3.10), with $q _ { \mathrm { m i n } } ~ = ~ 1 0 ^ { - 5 }$ . By (2.5), a larger charge $q _ { t }$ corresponds to a smaller noise variance $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ . The certified budgets are $\rho _ { \operatorname* { m a x } } \in \{ 0 . 2 5 , 0 . 5 , 1 \}$ for Tabular and the two full-network tasks, and $\rho _ { \operatorname* { m a x } } \in \{ 0 . 2 5 , 0 . 5 \}$ for Nonlinear head. Before batch averaging, each record’s loss gradient is smoothly rescaled to have norm at most $C _ { \mathrm { c l i p } }$ , as detailed in Appendix D.1. The sensitivity argument in Section 2.2 applies to this transformation because it requires only the per-example norm bound; smooth rescaling also permits differentiation of the design dynamics.

The total certified privacy charge is the sum of the per-step charges in (2.5), without subsampling amplification. All allocation comparisons therefore use the same conservative additive z-CDP certificate.

Data roles and schedule evaluation. Following the open-loop formulation, we separate records according to their use:

• Public auxiliary data provide preprocessing and, where required, pretraining.

• Public-design data determine the schedules. For Fashion-MNIST, these records form a subset of the public auxiliary data.

• Private training data are disjoint from public records and are used only after schedules have been fixed.

• Held-out test data assess prediction after training.

For any training sample, the No-noise reference is the terminal model obtained with the same initialization, batches, and training updates, but without Gaussian noise. Public-design and private training samples therefore have separate No-noise references. Schedule construction and evaluation follow three stages:

• Design. PB-AMD (Algorithm 1) minimizes the sample-average objective $F _ { M }$ in (4.10) on public-design data, using that sample’s No-noise reference as the tracking target.

• Selection. If different parameter settings or saved iterates produce several candidate schedules, we select the schedule with the lowest mean tracking error on the same public-design data, evaluated using independent Gaussian paths.

• Private evaluation. We freeze the selected schedules before private training and evaluate the tracking error $J ( u )$ in (3.2) against the No-noise reference computed on the private training sample. Held-out test records are used for predictive evaluation.

Thus, private records are not used to construct or select the schedules. The stated privacy budget certifies an individual noisy training run under a fixed schedule. The noiseless reference computed on private records and the associated tracking diagnostics are used only for offline evaluation and are not outputs covered by this guarantee. Joint release of multiple training runs or private-data diagnostics would require separate privacy accounting. Appendix D.1.3 and D.2 give the complete design, validation, and evaluation protocol.

Allocation-rule comparisons. These comparisons test the value of jointly optimizing all privacy charges through the full noisy nonlinear dynamics. We use the baselines of Table 5.

<table><tr><td>Rule</td><td>Allocation principle</td></tr><tr><td>Uniform</td><td>Assigns  $q _ { t } = \rho _ { \mathrm { m a x } } / T _ { \mathrm { ~ ; ~ } }$  providing the reference for assessing redistribution</td></tr><tr><td>Learning-rate allocation (LR)</td><td>Uses weights  $w _ { t } = \eta _ { t }$  , isolating learning-rate scaling from subsequent training dynamics</td></tr><tr><td>Decay</td><td>Combines learning-rate scaling with subsequent regularization shrinkage alone</td></tr><tr><td>Linearized</td><td>Uses  $w _ { t } = \sqrt { a _ { t } }$  with  $a _ { t }$  from (4.13),  $W _ { T } = \mathrm { I d } .$  and Jacobians evaluated along the public-design No-noise reference</td></tr><tr><td>Geometric</td><td>Uses  $w _ { t } = \eta _ { t } e ^ { \gamma t / ( T - 1 ) }$  , with γ selected by mean tracking error on public-design data</td></tr></table>

Table 5. Baseline allocation rules compared with PB-AMD under a common certified privacy budget.

For LR, Decay, Linearized, and Geometric, weights are converted to feasible charges through $q _ { t } ~ =$ max $\{ q _ { \mathrm { m i n } } , c w _ { t } \}$ , where $c > 0$ is chosen to meet the common budget. LR, Decay, and Linearized simplify Proposition 4.5 in different ways. Geometric adapts the exponential schedule family of [19], testing whether a tuned single-parameter profile can match PB-AMD’s optimization of all T charges.

Solver comparisons. To assess computational efficiency separately from allocation quality, we fix $F _ { M }$ and the training and adjoint routines, and compare solvers at a common first-order tolerance:

• Projected gradient descent $( P G D )$ uses the same adjoint gradient as PB-AMD, but projects additive Euclidean steps onto $Q ,$ instead of using multiplicative reallocation.

• Penalty uses logarithmic variances and the quadratic budget-excess penalty (4.4), following the penalty-method framework of [31].

• Penalty + repair rescales the raw charges above the floor to enforce the exact budget, enabling feasible utility and timing comparisons.

Appendix D.2 specifies the allocation weights, solver settings, candidate selection, and repair procedure.

Repetitions and statistical reporting. Each task–budget comparison uses ten problem seeds shared across methods; the two Nonlinear-head budgets use independent seed groups. Within each evaluation stage, methods share standard Gaussian vectors to obtain paired comparisons. We report the mean across seeds of the tracking-error reduction

$$
1 0 0 \left( 1 - \frac { J _ { \mathrm { P B } } } { J _ { \mathrm { b a s e } } } \right) ,
$$

where $J _ { \mathrm { P B } }$ and $J _ { \mathrm { b a s e } }$ are path-averaged tracking errors for PB-AMD and the comparator. Positive values favor PB-AMD; for example, a reduction of 10% means that its tracking error is 90% of the comparator’s error. Reported wins count seeds with lower PB-AMD tracking error.

Confidence intervals across the ten problem seeds are unadjusted for multiple testing. Statistical significance is assessed using Holm’s correction [17], controlling the probability of any false rejection at 5% within each comparison group. Appendix D.1.3 specifies these groups and the statistical protocol.

5.2. Temporal adaptation of privacy allocation. We first examine how privacy allocation responds to the training dynamics. Tabular and Fashion-constant test whether redistribution improves tracking at a constant learning rate, whereas Fashion-restart tests adaptation to a learning-rate increase. Figure 3 illustrates one instance per task at $\rho _ { \mathrm { m a x } } = 0 . 5 ;$ the tracking-error comparisons below use all ten problem seeds at each of the three budgets and are evaluated on private training data.

![](images/aaa2199990dfa2e17dcf2e68cd64054b98db817f7217c632c531745c67cd63d2.jpg)

(b) Fashion–constant  
![](images/f513f8d1e4091866558d32c23b3a43b049a24886254e17e42603dc3117960e4a.jpg)

(c) Fashion–restart  
![](images/5c452a1b897cd638b77084bfbe82755d6549bc01801e3b313cff20abbab88945.jpg)  
Figure 3. Single-instance exact-budget schedules at $\rho _ { \mathrm { m a x } } = 0 . 5$ . LR coincides with Uniform in (a,b); the vertical dotted line in (c) marks the learning-rate restart.

For Tabular, PB-AMD allocates more privacy expenditure to later iterations despite the constant learning rate (Figure 3(a)). This redistribution reduces tracking error by 1.04%–1.78% relative to Uniform across the tested budgets. Since LR coincides with Uniform in this setting, the improvement reflects an allocation benefit beyond learning-rate scaling.

For Fashion-constant, PB-AMD instead produces an approximately uniform allocation (Figure 3(b)), with tracking errors differing from Uniform by less than 0.02%. Thus, nonuniform expenditure offers essentially no tracking benefit in this task. Together, the two constant-rate experiments show that the usefulness of redistribution depends on the training dynamics: a constant learning rate alone does not determine whether a nonuniform schedule is beneficial.

For Fashion-restart, PB-AMD sharply increases expenditure after the learning-rate increase (Figure 3(c)). PB-AMD and LR reduce tracking error relative to Uniform by approximately 27.45% and 27.47%, respectively, at each budget. Learning-rate scaling therefore captures the substantial restart benefit, although it misses the smaller improvement in Tabular. PGD produces allocation profiles close to those of PB-AMD, indicating agreement between the two solvers on these instances.

We also examine sensitivity to the public-design sample. At $\rho _ { \mathrm { m a x } } = 0 . 5$ , we perturb that sample and recompute the PB-AMD schedule, keeping the private training data unchanged. The redesigned schedule retains an approximately 27.45% tracking-error reduction relative to Uniform (Appendix D.2.4). This shows that the improvement over Uniform survives the particular public-design perturbation tested, without using private records for schedule selection; it does not establish robustness to general public–private distribution shifts. Theorem 3.10 provides a conditional explanation for such transfer: under its global Lipschitz and uniform discrepancy assumptions, it bounds the target-objective suboptimality of a near-optimal design schedule. The bound depends on the model constants and training horizon and is not numerically evaluated here.

These experiments distinguish three outcomes: a small benefit beyond learning-rate scaling in Tabular, negligible benefit from redistribution in Fashion-constant, and a substantial restart benefit already captured by LR. PB-AMD accommodates all three without imposing a fixed preference for early or late expenditure.

5.3. Value of nonlinear schedule optimization. The Fashion-restart experiment shows that learning-rate scaling can capture a substantial allocation benefit. We now use Nonlinear head to test whether optimizing through the full noisy nonlinear dynamics improves on simpler allocation rules. Here the learning rate is constant, so LR coincides with Uniform, while weak regularization makes Decay nearly uniform. Geometric tunes a single shape parameter using public-design tracking error. Linearized accounts for perturbation propagation through sensitivities computed along the public-design No-noise reference. In contrast, PB-AMD jointly optimizes all T charges using the adjoint gradient (3.20) evaluated along noisy trajectories.

Aggregate tracking performance. Table 6 reports paired tracking-error reductions on private training data for ten instances at each budget. PB-AMD outperforms LR, Decay, and Geometric in all ten instances at both budgets. At $\rho _ { \mathrm { m a x } } = 0 . 5$ , the mean reductions are 7.72%, 7.85%, and 3.32%, respectively; at $\rho _ { \mathrm { m a x } } = 0 . 2 5$ , they are 4.32%, 4.41%, and 2.16%. These gains remain statistically significant after Holm correction across the eight comparisons involving four allocation rules and two budgets. Thus, in this task, optimizing all charges improves tracking beyond both learning-rate scaling and a tuned geometric profile.

The comparison with Linearized is less conclusive. PB-AMD achieves mean reductions of 5.07% at $\rho _ { \mathrm { m a x } } = 0 . 5$ and 2.75% at $\rho _ { \mathrm { m a x } } = 0 . 2 5$ , winning in eight and seven instances, respectively. Neither comparison remains statistically significant after the same correction.

<table><tr><td></td><td colspan="2"> $\rho _ { \mathrm { m a x } } = 0 . 5$ </td><td colspan="2"> $\rho _ { \mathrm { m a x } } = 0 . 2 5$ </td></tr><tr><td>Comparator</td><td>Error reduction [95% CI]</td><td>Wins</td><td>Error reduction [95% CI]</td><td>Wins</td></tr><tr><td>LR</td><td>7.72 [6.03, 9.41]</td><td>10/10</td><td>4.32 [3.55, 5.09]</td><td>10/10</td></tr><tr><td>Decay</td><td>7.85 [6.15,9.55]</td><td>10/10</td><td>4.41 [3.63, 5.18]</td><td>10/10</td></tr><tr><td>Linearized</td><td>5.07 [–2.10, 12.23]</td><td>8/10</td><td>2.75 [0.35, 5.15]</td><td>7/10</td></tr><tr><td>Geometric</td><td>3.32 [2.59, 4.04]</td><td>10/10</td><td>2.16 [1.77, 2.56]</td><td>10/10</td></tr></table>

Table 6. Nonlinear-head training: mean paired PB-AMD tracking-error reductions (%) on private training data relative to each comparator, with unadjusted 95% confidence intervals. Wins counts seeds with lower PB-AMD tracking error, out of ten per budget.

The confidence intervals in Table 6 are unadjusted; consequently, the interval excluding zero at $\rho _ { \mathrm { m a x } } = 0 . 2 5$ does not imply significance under the corrected tests. Figure 4(a) displays all instance-level comparisons, including those favoring Linearized. These results support an advantage over the simpler rules, while leaving the average advantage over Linearized statistically unresolved.

Selected examples of nonlinear reallocation. To examine how the schedules differ, we consider instances A $( \rho _ { \mathrm { m a x } } = 0 . 5 )$ and B $( \rho _ { \mathrm { m a x } } = 0 . 2 5 )$ . These were selected after the full comparison because they produced the largest PB-AMD gains over Linearized at their respective budgets: 32.40% and 9.20% on private training data. They therefore illustrate cases where nonlinear optimization is particularly beneficial, rather than typical gains across instances. Appendix D.1.3 reports Gaussian-path uncertainty intervals adjusted for this selection.

(a) All training instances  
![](images/b8826f77c925d8de1314f043e8f1f1ab4f889886d26f54904be06eb9d6ece0d5.jpg)  
(c) Public error estimates in A

(b) Instance A: $\rho _ { \mathrm { m a x } } = 0 . 5$  
![](images/58603263fe1e6c99ea9b53b0b90d6c7fca63018dba545b035c856388aa23fe7e.jpg)  
(d) Instance B: $\rho _ { \mathrm { m a x } } = 0 . 2 5$

![](images/7ced991533e8b3cb6efbf40d2252a687d73ec754545f2dfac0239139c56a49dc.jpg)

![](images/a868ecd245b5c96640a0c08295929b2a112bb3338357ee8766a9adfb0c884e72.jpg)  
Figure 4. Nonlinear-head allocation: (a) instance-level PB-AMD tracking-error reductions relative to Linearized on private training data, for two independent groups of ten seeds; (b,d) schedules for instances A and B, selected as the largest-gain examples at their respective budgets, with charges divided by the uniform charge. Uniform and LR coincide; (c) PB-AMD-to-Linearized error ratios for the two fixed schedules in instance A on public-design data, comparing the reference linearized prediction with full nonlinear evaluation using fresh Gaussian paths. Positive reductions in (a) and ratios below one in (c) favor PB-AMD.

Figures 4(b,d) show each charge divided by the uniform charge. A value of one represents both Uniform and LR, with Decay remaining close to this level. Both PB-AMD and Linearized favor early expenditure, but PB-AMD allocates less to the earliest steps and redistributes part of that expenditure to later iterations. In these two instances, this less concentrated allocation accompanies lower tracking error than Linearized. Linearized prediction versus nonlinear evaluation. For instance A, we further test whether the reference linearization correctly ranks the two schedules. We fix the schedules produced by PB-AMD and Linearized and evaluate both on the same public-design records, first using the linearized objective (4.13) and then using the full noisy dynamics with fresh Gaussian paths.

Figure 4(c) reports the ratio of PB-AMD’s error to Linearized’s error; values below one favor PB-AMD. The linearized prediction gives a ratio of 1.435, favoring Linearized. Full nonlinear evaluation reverses this ranking, giving PB-AMD 31.65% lower tracking error, consistent with its advantage on private training data. This example shows that sensitivities computed along a fixed No-noise reference can favor a schedule that performs worse under the actual noisy nonlinear dynamics, explaining why optimizing those dynamics can improve allocation.

5.4. Predictive consequences of privacy allocation. We next examine whether the tracking gains observed for Fashion-restart translate into better prediction. We train on private records using the previously fixed schedules and evaluate the resulting models on held-out test records. Tracking error J(u) in (3.2), with the quadratic loss (3.6), measures squared parameter distance to the No-noise reference trained on the same private sample. Predictive performance is assessed separately through test accuracy, the fraction of correctly classified records, and unregularized cross-entropy, which penalizes low probabilities assigned to the true labels.

Figure 5 reports Fashion-restart results across the three certified budgets. PB-AMD improves test accuracy and reduces cross-entropy relative to Uniform at every budget. At $\rho _ { \mathrm { m a x } } = 0 . 2 5$ , mean accuracy rises from 68.61% to 69.43%, an increase of 0.82 percentage points. At each budget, PB-AMD closes approximately one-third of the accuracy and cross-entropy gaps between Uniform and the No-noise reference. LR nearly coincides with PB-AMD in both metrics, consistently with their similar tracking gains in Section 5.2.

![](images/eeda96df6e9fab5468888cf44448ff43114ed5b12804a61ab855773fd969e025.jpg)  
Figure 5. Test performance for Fashion-restart models trained on private data: means and pointwise 95% confidence intervals over ten problem seeds. Each noise schedule uses 256 Gaussian training runs per seed.

The statistical analysis treats the predictive comparisons for Fashion-constant and Fashion-restart as one family: two tasks, three budgets, and two metrics, giving twelve tests. Although Figure 5 focuses on Fashion-restart, its significance assessment includes the Fashion-constant comparisons in this correction.

For Fashion-restart, the improvements in test accuracy and test cross-entropy relative to Uniform remain significant after Holm correction across the twelve-comparison family; Appendix D.1.3 details the statistica protocol.

The figure shows means and pointwise 95% confidence intervals across ten problem seeds, with each noisy schedule evaluated using 256 Gaussian training runs per seed. These results demonstrate that, in the Fashion-restart setting, an allocation optimized for parameter tracking also improves held-out prediction at a fixed privacy budget.

5.5. Solver convergence and computational efficiency. This section compares the time required by PB-AMD, PGD, and Penalty + repair to reach the same first-order tolerance for the exact-budget problem (3.11). We use ten new public-design instances each of Tabular, Fashion-restart, and Nonlinear head at $\rho _ { \mathrm { m a x } } = 0 . 5$ . Within each instance, all solvers share the tracking term $F _ { M }$ in (4.10), Uniform initialization $q ^ { 0 }$ , minibatches, and Gaussian paths. Fixed stepsizes and penalty parameters are calibrated on separate public-design instances before timing; for Nonlinear head, PB-AMD and PGD use the same sufficient-decrease line search.

We measure time to $G _ { F _ { M } } ( q ) / F _ { 0 } \leq 1 0 ^ { - 4 }$ , where $G _ { F _ { M } }$ is the variational gap in (4.11) and $F _ { 0 } = F _ { M } ( q ^ { 0 } )$ is the objective at Uniform initialization. The gap is zero exactly when the constrained first-order conditions hold, giving a common stationarity criterion for the three solvers. For Penalty + repair, we evaluate it at the repaired charges. Timed solves include required objective and adjoint evaluations, line search, and repair, but exclude calibration and evaluations used only to draw the curves. Private training and prediction scoring are not timed.

At this tolerance, PB-AMD reduces solver time relative to PGD by 17.7% for Tabular (Figure 6(a)), 27.4% for Fashion-restart (Figure 6(b)), and 35.7% for Nonlinear head (Figure 6(c)).

![](images/65e91d8a97fd86ef5e5b97994b9f6ee3cad1d7d8be45f56d22ce73171311ae8c.jpg)  
Figure 6. Solver convergence on public-design instances at $\rho _ { \mathrm { m a x } } = 0 . 5 \colon$ : median normalized gap $G _ { F _ { M } } ( q ) / F _ { 0 }$ from (4.11), with inter-quartile bands over ten convergence instances per task. The horizontal dotted line is the $1 0 ^ { - 4 }$ timing target.

These percentages use the median ratio of PGD time to PB-AMD time over instances where both solvers reach the target; the curves include all ten instances per task.

Both methods reach the target in all ten Tabular instances, all ten Fashion-restart instances, and the same nine head instances. None of the 30 repaired penalty runs reaches this tolerance within the update caps.

PB-AMD maintains the exact budget at every update and has lower median schedule-optimization time than PGD among the tested instances where both methods reach the common first-order tolerance.

Additional results on raw penalty-budget violations, exact-budget preservation, returned-schedule tracking errors, and agreement with the analytical linear-quadratic optimizer are reported in Appendices D.2.3 and D.2.5. Appendices D.3-D.4 give the complete convergence and timing protocol.

## 6. Conclusion

We have developed a control-theoretic mathematical framework and an exact-budget optimization method for temporal privacy allocation in differentially private stochastic gradient descent. By treating the complete Gaussian noise-variance schedule as an open-loop control, the framework relates the timing of privacy expenditure to its effect on the final model under a composed zero-concentrated differential privacy constraint. At the same time, separating schedule design from private training makes the privacy cost of schedule selection an explicit part of the certification.

The central structural result is budget saturation for expected quadratic terminal tracking: every locally optimal schedule exhausts the certified budget. Consequently, the change from noise variances to per-iteration privacy charges yields an exact-budget simplex. Privacy-Budget Adjoint Mirror Descent (PB-AMD) exploits this geometry to preserve positivity and the prescribed budget at every update without penalty-parameter tuning. The analysis supports this construction with existence and stability results for optimal schedules and stationarity guarantees for the optimization method.

Numerical experiments show that the optimized allocations adapt to the training dynamics. In the Nonlinearhead task, PB-AMD improves tracking relative to learning-rate, decay, and tuned geometric allocation, while its average advantage over Linearized remains statistically unresolved after multiplicity correction. In Fashion-restart, PB-AMD improves prediction relative to Uniform, with learning-rate allocation achieving similar performance. Comparisons at a common stationarity tolerance show lower schedule-optimization times than PGD on the tested problems. These results support the framework as a method for optimizing certified privacy allocations through nonlinear training dynamics, while leaving its predictive advantage over the strongest allocation baselines open.

This appendix collects the formal privacy notions underlying the accounting framework of Section 2. The material is classical [9, 11, 12]; we include it to make precise the definitions and properties used in the privacy certification of the iterative Gaussian mechanisms considered in the paper.

A.1. Adjacency and differential privacy. We begin with the notion of replace-one adjacency and then recall Rényi divergence and the definition of zCDP.

Definition A.1. Let $( \mathcal { X } , \Sigma _ { \mathcal { X } } )$ be a measurable sample space, fix $N \in \mathbb { N } .$ , and set $\mathcal { U } _ { N } : = \mathcal { X } ^ { N }$ . Two datasets $D , D ^ { \prime } \in \mathcal { U } _ { N }$ are called neighboring, written $D \sim D ^ { \prime }$ , if they differ in exactly one coordinate.

Throughout this work, privacy guarantees are expressed using zero-Concentrated Differential Privacy (zCDP). This notion is particularly convenient for privacy accounting in the iterative Gaussian mechanism underlying DP-SGD. Its definition is based on Rényi divergence.

Definition A.2. Let P and Q be probability measures. For $\alpha > 1$ and $P \ll Q$ , the Rényi divergence of order α is

$$
{ \mathfrak { D } } _ { \alpha } ( P \| Q ) : = { \frac { 1 } { \alpha - 1 } } \log \int \left( { \frac { d P } { d Q } } \right) ^ { \alpha } d Q .
$$

If $P \not \ll Q$ , set $\mathfrak { D } _ { \alpha } ( P \| Q ) = + \infty$

Definition A.3. A randomized mechanism $\mathcal { M } : \mathcal { U } _ { N } \not \sim \mathcal { O }$ satisfies $\rho { - } z C D P ;$ , with $\rho \geq 0 , i f ,$ for every $D \sim D ^ { \prime }$ and every $\alpha > 1$ ，

$$
\mathfrak { D } _ { \alpha } \big ( \mathcal { L } ( \mathcal { M } ( D ) ) \| \mathcal { L } ( \mathcal { M } ( D ^ { \prime } ) ) \big ) \leq \rho \alpha ,
$$

where L(X) denotes the law of a random variable $X$

The parameter $\rho$ will be referred to as the zCDP privacy charge. Its main advantage for the purposes of this paper is its additive behavior under composition. Consider a finite sequence $\{ \breve { M } _ { t } \} _ { t = 0 } ^ { T - 1 }$ of possibly adaptive randomized mechanisms. Suppose that, conditionally on all outputs and public information available before iteration $t ,$ the mechanism used at that iteration satisfies $\rho _ { t ^ { - } \mathrm { Z } } \mathrm { C D P }$ . Then the complete mechanism after $T$ iterations satisfies $\rho _ { \mathrm { t o t } ^ { - } \mathrm { Z } } \mathrm { C D P }$ with

$$
\rho _ { \mathrm { t o t } } = \sum _ { t = 0 } ^ { T - 1 } \rho _ { t } .\tag{A.1}
$$

Given a prescribed total zCDP budget $\rho _ { \mathrm { m a x } } > 0$ , a sequence of mechanisms is therefore certified whenever $\rho _ { \mathrm { t o t } } \le \rho _ { \mathrm { m a x } }$

In this accounting process, the mechanisms need not be independent and the query used at a given iteration may depend on the preceding private outputs. What is required is that, after conditioning on the previous transcript, a valid zCDP bound $\rho _ { t }$ holds uniformly for the conditional mechanism at that iteration [12].

This adaptive composition property is particularly suited to private stochastic optimization. The current optimization iterate depends on all preceding noisy updates, but once the preceding transcript is conditioned upon, the iterate is fixed and the privacy charge of the next Gaussian query can be bounded conditionally. The individual bounds may then be summed to obtain a privacy guarantee for the complete training procedure.

A.2. Gaussian mechanism and privacy accounting as a resource constraint. Let $f : \mathcal { U } _ { N } \to \mathbb { R } ^ { d }$ be a measurable query over the dataset universe $\mathcal { U } _ { N }$ , with finite $\ell _ { 2 } .$ -sensitivity

$$
\Delta _ { 2 , f } : = \operatorname* { s u p } _ { D \sim D ^ { \prime } } \| f ( D ) - f ( D ^ { \prime } ) \| _ { 2 } .
$$

For $s > 0$ , consider the Gaussian mechanism $\mathcal { M } _ { s } ( D ) = f ( D ) + s \xi$ , with $\xi \sim \mathcal { N } ( 0 , I _ { d } )$ . For two Gaussian distributions having the same covariance $s ^ { 2 } I _ { d } .$ , their Rényi divergence satisfies

$$
\mathfrak { D } _ { \alpha } \Big ( \mathcal { N } ( f ( D ) , s ^ { 2 } I _ { d } ) \| \mathcal { N } ( f ( D ^ { \prime } ) , s ^ { 2 } I _ { d } ) \Big ) = \frac { \alpha } { 2 s ^ { 2 } } \| f ( D ) - f ( D ^ { \prime } ) \| _ { 2 } ^ { 2 } \le \alpha \frac { \Delta _ { 2 , f } ^ { 2 } } { 2 s ^ { 2 } } .
$$

Hence the Gaussian mechanism is $\rho { \mathrm { - z C D P } }$ with $\rho = \Delta _ { 2 , f } ^ { 2 } / ( 2 s ^ { 2 } )$ . This relation is the calibration formula used throughout the paper. It shows directly the trade-off between noise and privacy expenditure: for fixed sensitivity, increasing the Gaussian standard deviation decreases the $\mathrm { z C D P }$ charge, whereas decreasing the noise increases the privacy expenditure.

Combining Gaussian calibration and adaptive composition gives the accounting structure used throughout the paper. If the Gaussian mechanism applied at iteration t has certified charge $\rho _ { t }$ , then $\rho _ { \mathrm { t o t } }$ defined in (A.1) is a valid zCDP bound for the full sequence of private updates.

A prescribed privacy budget $\rho _ { \mathrm { m a x } }$ therefore imposes the resource constraint $\rho _ { \mathrm { t o t } } \le \rho _ { \mathrm { m a x } }$ . A privacy schedule specifies how this total expenditure is distributed across iterations. The accountant specifies which temporal allocations are privacy feasible, but does not identify which allocation is preferable for learning. In the setting of this paper, the latter question is treated as an optimization problem: the total certified privacy resource is distributed across iterations so as to optimize the terminal performance criterion.

A.3. Post-processing. DP is invariant under data-independent post-processing. If M is $\rho { \mathrm { - z C D P } }$ and K is a randomized transformation whose conditional distribution depends only on the output of M and on randomness independent of the private dataset, then $\kappa \circ { \mathcal { M } }$ is also $\rho { \mathrm { - z C D P } }$

This property is used repeatedly in the DP-SGD construction of Section 2.2. Once a data-dependent gradient query has been privatized by Gaussian perturbation, deterministic operations performed on that private output do not generate an additional privacy charge. Similarly, releasing only the terminal iterate of an already private training transcript cannot increase the privacy loss.

These properties, together with the Gaussian calibration above, provide the privacy accounting used throughout the paper; see, e.g., [9].

## Appendix B. Proofs of the results in Section 3

## B.1. Proofs: Exact certified-budget saturation for terminal tracking.

Proof of Theorem 3.2. Let $\mathcal { F } _ { T - 1 } : = \sigma ( \xi _ { 0 } , \dots , \xi _ { T - 2 } )$ , interpreted as the trivial sigma-field when $T = 1$ , and define the $\mathcal { F } _ { T - 1 }$ -measurable vector $Y : = \theta _ { T - 1 } - \eta _ { T - 1 } g _ { T - 1 } ( \theta _ { T - 1 } ) - \theta ^ { \star }$

Lemma B.2 (applied to the singleton control set) and the linear-growth condition (A2) give $Y \in L ^ { 2 }$ Moreover, $\theta _ { T } - \theta ^ { \star } = Y - \eta _ { T - 1 } \Delta _ { T - 1 } ^ { b } \sqrt { u _ { T - 1 } } \xi _ { T - 1 }$ . Conditioning on $\mathcal { F } _ { T - 1 }$ and using independence, zero mean, and $\mathbb { E } \left. \xi _ { T - 1 } \right. ^ { 2 } = d \operatorname { g i v e s } \left( 3 . 7 \right)$

If a feasible local minimizer had $\rho ( u ^ { \star } ) < \rho _ { \mathrm { m a x } } ,$ then for every sufficiently small $h > 0$ the point $u ^ { \star } - h { \sf e } _ { T - 1 } .$ where $\mathsf { e } _ { T - 1 }$ is the last coordinate vector of $\mathbb { R } ^ { T }$ , would remain positive and feasible and would lie in an arbitrarily small neighborhood of $u ^ { \star }$ . Equation (3.7) would lower the objective by $d \eta _ { T - 1 } ^ { 2 } ( \Delta _ { T - 1 } ^ { b } ) ^ { 2 } h > 0 .$ contradicting local optimality. □

## B.2. Proofs: Privacy-charge reformulation and exact-budget simplex.

Proof of Proposition 3.3. The transformation $q _ { t } = ( 2 u _ { t } ) ^ { - 1 }$ , with inverse $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ , gives the asserted bijection and preserves the dynamics and objective. Let $S _ { u }$ denote the nonempty compact minimizer set of (3.4). Its image

$$
S _ { q } : = \left\{ \left( ( 2 u _ { t } ) ^ { - 1 } \right) _ { t = 0 } ^ { T - 1 } : u \in S _ { u } \right\}
$$

is nonempty and compact. Theorem 3.2 implies $S _ { q } \subset Q ^ { \circ }$ ; together with the bijection, this gives

$$
S _ { q } = \underset { q \in Q ^ { \circ } } { \arg \operatorname* { m i n } } F ( q ) .
$$

Since every coordinate is positive on $S _ { q }$ , compactness yields $\delta : = \mathrm { m i n } _ { q \in S _ { q } } \mathrm { m i n } _ { 0 \le t < T } q _ { t } > 0$

Moreover, $\delta \leq \rho _ { \operatorname* { m a x } } / T$ . Set $q _ { * } = \delta / 2$ . Then $q _ { * } \in ( 0 , \rho _ { \operatorname* { m a x } } / T )$ , and for every $0 < q _ { \mathrm { m i n } } < q _ { * }$ , we have $S _ { q } \subset Q ( q _ { \operatorname* { m i n } } ) \subset Q ^ { \circ }$ . These inclusions show that the minimum over $Q ( q _ { \mathrm { m i n } } )$ equals that over $Q ^ { \circ }$ , and that all minimizers over $Q ^ { \circ }$ belong to $Q ( q _ { \mathrm { m i n } } )$ . Consequently,

$$
\operatorname * { a r g m i n } _ { q \in Q ( q _ { \operatorname * { m i n } } ) } F ( q ) = S _ { q } = \arg \operatorname* { m i n } _ { q \in Q ^ { \circ } } F ( q ) .
$$

## B.3. Proofs: Differentiability, adjoints, and KKT conditions.

Proof of Proposition $\ 3 . 4 \cdot$ Fix a compact set $K \subset ( 0 , \infty ) ^ { T }$ . For every noise path, the finite recursion (3.1) is a composition of $C ^ { 1 }$ maps of u. If $S _ { s , t } : = \partial _ { u _ { t } } \theta _ { s }$ , then $S _ { s , t } = 0$ for $s \leq t ,$ , while

$$
S _ { t + 1 , t } = - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { 2 \sqrt { u _ { t } } } \xi _ { t } , \qquad S _ { s + 1 , t } = A _ { s } ( \theta _ { s } ) S _ { s , t } \quad \mathrm { ~ f o r ~ } s \ge t + 1 .
$$

Consequently,

$$
\partial _ { u _ { t } } j ( u ; \xi ) = \langle \nabla L ( \theta _ { T } ) , S _ { T , t } \rangle = - \frac { \eta _ { t } \Delta _ { t } ^ { b } } { 2 \sqrt { u _ { t } } } \langle p _ { t + 1 } , \xi _ { t } \rangle ,
$$

which proves (3.14). It remains to justify expectation and differentiation interchange. Assumption (A6) gives

$$
\lVert A _ { s } ( \theta _ { s } ) \rVert _ { \mathrm { o p } } \leq 1 + \eta _ { s } K _ { g } \Big ( 1 + \lVert \theta _ { s } \rVert ^ { r _ { g } } \Big ) .
$$

Lemma B.2 provides a constant $C _ { K }$ such that, uniformly over $u \in K$

$$
\operatorname* { m a x } _ { 0 \leq s \leq T } \| \theta _ { s } ( u ; \xi ) \| \leq C _ { K } \left( 1 + \sum _ { s = 0 } ^ { T - 1 } \| \xi _ { s } \| \right) \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

The finite adjoint recursion and the polynomial growth of $\nabla L$ consequently bound every $p _ { s }$ by a polynomial in the Gaussian norms. Because in $\mathrm { f } _ { u \in K } u _ { t } > 0$ , there are deterministic constants $C _ { K , t }$ and an integer r¯ for which

$$
\operatorname* { s u p } _ { u \in K } | \partial _ { u _ { t } } j ( u ; \xi ) | \leq C _ { K , t } \left( 1 + \sum _ { s = 0 } ^ { T - 1 } \| \xi _ { s } \| \right) ^ { \bar { r } } .\tag{B.1}
$$

The right-hand side is integrable, indeed it has moments of every finite order. For almost every $\xi ,$ , the state, adjoint, and pathwise derivative are continuous in u.

To justify differentiation under the expectation at an arbitrary $u ^ { \circ } \in ( 0 , \infty ) ^ { T }$ , choose a compact rectangular neighborhood $K ^ { \prime } \Subset ( 0 , \infty ) ^ { T }$ of $u ^ { \circ }$ . Let $\mathsf { e } _ { t }$ denote the t-th coordinate vector of $\mathbb { R } ^ { T }$ . For every sufficiently small nonzero $h ,$ the segment $\{ u ^ { \circ } + \tau h \mathbf { e } _ { t } : 0 \leq \tau \leq 1 \}$ lies in $K ^ { \prime }$ . The one-dimensional mean-value theorem and (B.1), with $K ^ { \prime }$ in place of $K ,$ give

$$
\left| \frac { j ( u ^ { \circ } + h \mathfrak { e } _ { t } ; \xi ) - j ( u ^ { \circ } ; \xi ) } { h } \right| \overset { } { \le } \operatorname* { s u p } _ { v \in K ^ { \prime } } | \partial _ { u _ { t } } j ( v ; \xi ) | \le C _ { K ^ { \prime } , t } \left( 1 + \sum _ { s = 0 } ^ { T - 1 } \| \xi _ { s } \| \right) ^ { \bar { r } } .
$$

Thus the difference quotients admit an integrable envelope.

Dominated convergence now yields (3.15). Applying the same domination to a convergent sequence of controls gives continuity of each partial derivative on $K$ . Since $K$ was arbitrary, $J \in C ^ { 1 } ( ( 0 , \infty ) ^ { T } )$ □

Proof of Theorem 3.5. By Proposition 3.4, $J \in C ^ { 1 } ( ( 0 , \infty ) ^ { T } )$ . The privacy constraint $\Phi ( u )$ in (3.16) is also continuously differentiable, with $\nabla \Phi ( u ) = \big ( - ( 2 u _ { t } ^ { 2 } ) ^ { - 1 } \big ) _ { t = 0 } ^ { T - 1 }$

If $\begin{array} { r } { \Phi ( u ^ { \star } ) = 0 } \end{array}$ , the direction $\boldsymbol { v } = - \nabla \Phi ( u ^ { \star } )$ satisfies $\langle \nabla \Phi ( u ^ { \star } ) , v \rangle = - \| \nabla \Phi ( u ^ { \star } ) \| ^ { 2 } < 0 . \mathrm { ~ I f ~ } \Phi ( u ^ { \star } ) < 0$ , there are no active constraints. Thus the Mangasarian-Fromovitz constraint qualification $\left( \mathrm { M F C Q } \right)$ holds at $u ^ { \star }$ . The finite-dimensional KKT theorem [31, Chapter 12] therefore provides a deterministic multiplier $\lambda ^ { \star } \geq 0$ such that

$$
\lambda ^ { \star } \Phi ( u ^ { \star } ) = 0 \qquad \mathrm { ~ a n d ~ } \qquad \partial _ { u _ { t } } J ( u ^ { \star } ) - \frac { \lambda ^ { \star } } { 2 ( u _ { t } ^ { \star } ) ^ { 2 } } = 0 \mathrm { ~ f o r ~ a l l ~ } t \in [ [ T ] _ { 0 } .
$$

For the state and adjoint associated with $u ^ { \star }$ , formula (3.15) therefore gives

$$
- \frac { \eta _ { t } \Delta b _ { t } } { 2 \sqrt { u _ { t } ^ { \star } } } \mathbb { E } \left[ \left. p _ { t + 1 } ^ { \star } , \xi _ { t } \right. \right] - \frac { \lambda ^ { \star } } { 2 ( u _ { t } ^ { \star } ) ^ { 2 } } = 0 .
$$

Multiplying by $- 2 ( u _ { t } ^ { \star } ) ^ { 2 }$ , which is well defined since $u _ { t } ^ { \star } > 0$ , finally yields (3.17).

Proof of Corollary 3.6. Budget saturation follows from Theorem 3.2. Differentiating $\left( 3 . 7 \right)$ gives the identity $\partial _ { u _ { T - 1 } } \bar { J ( u ) } = d \eta _ { T - 1 } ^ { 2 } ( \Delta b _ { T - 1 } ) ^ { 2 }$ . Combining (3.15) with (3.17) yields $\lambda ^ { \star } = 2 ( u _ { t } ^ { \star } ) ^ { 2 } \partial _ { u _ { t } } J ( u ^ { \star } )$ for all $t \in [ [ T ] ] _ { 0 }$ Taking $t = T - 1$ , we obtain $\lambda ^ { \star } = 2 d \eta _ { T - 1 } ^ { 2 } ( \Delta _ { T - 1 } ^ { b } ) ^ { 2 } ( u _ { T - 1 } ^ { \star } ) ^ { 2 }$ . Since $\eta _ { t } , \Delta _ { t } ^ { b } , u _ { t } ^ { \star } > 0$ , equation (3.17) therefore implies E $\begin{array} { r } { \left[ \left. p _ { t + 1 } ^ { \star } , \xi _ { t } \right. \right] = - \frac { \lambda ^ { \star } } { \eta _ { t } \Delta b _ { t } ( u _ { t } ^ { \star } ) ^ { 3 / 2 } } } \end{array}$ . Solving this identity for $\boldsymbol { u } _ { t } ^ { \star }$ gives (3.18). □

Proof of Proposition 3.7. The derivative of the noise term in (3.9) is

$$
\frac { d } { d q _ { t } } \left[ - \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ) ^ { - 1 / 2 } \xi _ { t } \right] = \eta _ { t } \Delta _ { t } ^ { b } ( 2 q _ { t } ) ^ { - 3 / 2 } \xi _ { t } .
$$

The chain rule and the adjoint recursion give (3.19). Since $Q$ is compact and $q _ { t } \geq q _ { \operatorname* { m i n } }$ on $Q$ , there is an open neighborhood of $Q$ on which every coordinate is at least $q _ { \mathrm { m i n } } / 2$ . The same domination argument as in Proposition 3.4 therefore justifies interchanging expectation and differentiation. □

## B.4. Proofs: Well-posedness and stability of the control problem.

## B.4.1. Existence and continuity.

Proposition B.1. For every $\rho > 0$ , the set $\mathcal { U } _ { a d } ( \rho )$ is nonempty, closed in $\mathbb { R } ^ { T }$ , and convex. Its extreme points are exactly the budget-saturating schedules:

$$
\mathrm { E x t } ( \mathcal { U } _ { a d } ( \rho ) ) = \Big \{ u \in ( 0 , \infty ) ^ { T } : \rho ( u ) = \rho \Big \} .
$$

Proof. The constant vector $u _ { t } = T / ( 2 \rho )$ is feasible. Moreover, every feasible vector satisfies $u _ { t } \geq ( 2 \rho ) ^ { - 1 }$ because $( 2 u _ { t } ) ^ { - 1 } \leq \rho ( u ) \leq \rho .$ Hence a convergent sequence in $\mathcal { U } _ { \mathrm { a d } } ( \rho )$ has a strictly positive limit, and continuity of $u \mapsto \rho ( u )$ on $[ ( 2 \rho ) ^ { - 1 } , \infty ) ^ { T }$ proves closedness. Convexity follows from convexity of $s \mapsto s ^ { - 1 }$ on $( 0 , \infty )$

The function $\rho ( \cdot )$ is strictly convex. If $\rho ( u ) < \rho ,$ sufficiently small perturbations $u \pm v$ remain feasible, so u is not extreme. Conversely, if $\rho ( u ) = \rho$ and $u = \lambda v + ( 1 - \lambda ) w$ with $v , w \in \mathcal { U } _ { \mathrm { a d } } ( \rho )$ and $\lambda \in ( 0 , 1 )$ , then $\rho = \rho ( u ) \leq \lambda \rho ( v ) + ( 1 - \lambda ) \rho ( w ) \leq \rho .$ . Equality throughout and strict convexity imply $v = w = u$ □

Lemma B.2. Under $( A I )$ and (A2), let $K \subset ( 0 , \infty ) ^ { T }$ be nonempty and bounded. For every $p \geq 1$ there is a finite constant $C _ { K , p }$ such that

$$
\operatorname* { s u p } _ { u \in K } \mathbb { E } \left\| \theta _ { T } ( u ) \right\| ^ { p } \leq C _ { K , p } .
$$

More precisely, there is a deterministic $C _ { K } < \infty$ such that

$$
\operatorname* { s u p } _ { u \in K } \operatorname* { m a x } _ { 0 \le s \le T } \| \theta _ { s } ( u ) \| \le C _ { K } \left( 1 + \sum _ { t = 0 } ^ { T - 1 } \| \xi _ { t } \| \right) \quad a l m o s t \ s u r e l y .
$$

Proof. The linear-growth condition gives $\lVert g _ { t } ( { \boldsymbol { \theta } } ) \rVert \leq a _ { t } + b _ { t } \lVert { \boldsymbol { \theta } } \rVert$ . If we set ${ \bar { u } } : = \operatorname* { s u p } _ { u \in K , \ t \in [ [ T ] _ { 0 } } u _ { t }$ , then

$$
\begin{array} { r } { \| \theta _ { t + 1 } ( u ) \| \leq \left( 1 + \eta _ { t } b _ { t } \right) \| \theta _ { t } ( u ) \| + \eta _ { t } a _ { t } + \eta _ { t } \Delta _ { t } ^ { b } \sqrt { \bar { u } } \left\| \xi _ { t } \right\| . } \end{array}
$$

Finite iteration gives the displayed pathwise bound. Gaussian random vectors have moments of every finite order, so the stated moment estimate follows. □

Proposition B.3. Under (A1), (A2), and (A3), J is finite and continuous on $( 0 , \infty ) ^ { T }$ , and hence on every $\mathcal { U } _ { a d } ( \rho )$

Proof. Let $u ^ { n } \to u \mathrm { ~ i n ~ } ( 0 , \infty ) ^ { T }$ . Continuity of the square-root function gives $\sqrt { u _ { t } ^ { n } }  \sqrt { u _ { t } }$ for every $t \in [ [ T ] ] _ { 0 }$ Since each $g _ { t }$ is continuous by Assumption (A2), induction in the state recursion (3.1) gives

$$
\theta _ { t } ( u ^ { n } ; \xi ) \to \theta _ { t } ( u ; \xi )
$$

for all $t = 0 , \ldots , T$ and for almost every noise realization ξ. Then by continuity of $L _ { : }$ we have

$$
L ( \theta _ { T } ( u ^ { n } ; \xi ) ) \to L ( \theta _ { T } ( u ; \xi ) ) \mathrm { \quad ~ a l m o s t ~ s u r e l y } .
$$

The set $K : = \{ u ^ { n } \} _ { n \in \mathbb { N } } \cup \{ u \}$ is bounded. Hence, by Lemma B.2, there exists a random variable with finite moments of every order that uniformly bounds $\lVert \boldsymbol { \theta } _ { T } ( \boldsymbol { v } ) \rVert$ for $v \in K$ . Together with the polynomial-growth assumption on $L ,$ this yields an integrable random variable dominating $\left| L ( \theta _ { T } ( u ^ { n } ) ) \right)$ | uniformly in n. The dominated convergence theorem therefore gives $J ( u ^ { n } ) = \mathbb { E } [ L ( \theta _ { T } ( u ^ { n } ) ) ] \to \mathbb { E } [ L ( \theta _ { T } ( u ) ) ] = J ( u )$

The same polynomial-growth bound and Lemma B.2 show that $J ( u )$ is finite for every $u \in ( 0 , \infty ) ^ { T }$ . Thus $J$ is finite and continuous on $( 0 , \infty ) ^ { T }$ , and consequently on every $\mathcal { U } _ { \mathrm { a d } } ( \rho )$ □

Lemma B.4. Let Z and $\xi$ be independent square-integrable $\mathbb { R } ^ { d }$ -valued random vectors, and suppose $\mathbb { E } \xi = 0$ Then, for every $a \in \mathbb { R }$

$$
\begin{array} { r } { \mathbb { E } \left. Z , \xi \right. = 0 , \qquad \mathbb { E } \left\| Z - a \xi \right\| ^ { 2 } = \mathbb { E } \left\| Z \right\| ^ { 2 } + a ^ { 2 } \mathbb { E } \left\| \xi \right\| ^ { 2 } . } \end{array}
$$

Proof. Independence and square integrability imply $\mathbb { E } \left. Z , \xi \right. = \left. \mathbb { E } Z , \mathbb { E } \xi \right. = 0$ . Expanding the square gives the second identity. □

Proposition B.5. Under Assumptions (A1), (A2), $( A 4 )$ , and (A5), there are constants $\kappa > 0$ and $C _ { \mathrm { c o e r } } \geq 0$ such that, for every $\rho > 0$

$$
J ( u ) \geq \kappa \sum _ { t = 0 } ^ { T - 1 } u _ { t } - C _ { \mathrm { c o e r } } , \qquad u \in \mathcal { U } _ { a d } ( \rho ) .\tag{B.2}
$$

Consequently, $J ( u ^ { n } ) \to \infty$ for every feasible sequence with $\| u ^ { n } \|  \infty$

Proof. Set $\delta _ { t } : = \theta _ { t } ( u ) - \bar { \theta } _ { t }$ . Then $\delta _ { t + 1 } = Z _ { t } - \eta _ { t } \Delta _ { t } ^ { b } \sqrt { u _ { t } } \xi _ { t }$ , with $Z _ { t } : = \mathsf { G } _ { t } ( \theta _ { t } ( u ) ) - \mathsf { G } _ { t } ( { \bar { \theta } } _ { t } )$ . The vector $Z _ { t }$ is measurable with respect to $\sigma ( \xi _ { 0 } , \ldots , \xi _ { t - 1 } )$ , is square integrable by Lemma B.2, and is independent of $\xi _ { t }$ Lemma B.4 therefore gives

$$
\begin{array} { r } { \mathbb { E } \left\| \delta _ { t + 1 } \right\| ^ { 2 } = \mathbb { E } \left\| Z _ { t } \right\| ^ { 2 } + d \eta _ { t } ^ { 2 } ( \Delta _ { t } ^ { b } ) ^ { 2 } u _ { t } , } \end{array}
$$

and Assumption (A5) yields

$$
\begin{array} { r } { \mathbb { E } \left\| \delta _ { t + 1 } \right\| ^ { 2 } \geq \chi _ { t } ^ { 2 } \mathbb { E } \left\| \delta _ { t } \right\| ^ { 2 } - \omega _ { t } + d \eta _ { t } ^ { 2 } ( \Delta _ { t } ^ { b } ) ^ { 2 } u _ { t } . } \end{array}
$$

Iterating this inequality gives

$$
\mathbb { E } \left. \delta _ { T } \right. ^ { 2 } \geq d \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } ( \Delta _ { t } ^ { b } ) ^ { 2 } u _ { t } \prod _ { j = t + 1 } ^ { T - 1 } \chi _ { j } ^ { 2 } - \Omega \geq \kappa _ { 0 } \sum _ { t = 0 } ^ { T - 1 } u _ { t } - \Omega ,
$$

where

$$
\kappa _ { 0 } : = d \operatorname* { m i n } _ { t \in \{ 0 , \ldots , T - 1 \} } \left( \eta _ { t } ^ { 2 } ( \Delta _ { t } ^ { b } ) ^ { 2 } \prod _ { j = t + 1 } ^ { T - 1 } \chi _ { j } ^ { 2 } \right) > 0 \quad \mathrm { ~ a n d ~ } \quad \Omega : = \sum _ { t = 0 } ^ { T - 1 } \omega _ { t } \prod _ { j = t + 1 } ^ { T - 1 } \chi _ { j } ^ { 2 } .
$$

Assumption (A4) gives (B.2) with $\kappa = c _ { 1 } \kappa _ { 0 }$ and $C _ { \mathrm { c o e r } } = c _ { 1 } \Omega + c _ { 0 }$

Proof of Theorem 3.8. The feasible set is nonempty and closed by Proposition B.1; J is finite and lower semicontinuous by Proposition B.3. A minimizing sequence is bounded by coercivity, hence has a convergent subsequence in finite dimension. Closedness preserves feasibility, and lower semicontinuity gives optimality of the limit. The same coercivity argument shows that the minimizer set is bounded; it is closed by lower semicontinuity and therefore compact. This is the finite-dimensional direct-method argument; see, e.g., [32]. □

## B.4.2. Dependence on the privacy budget.

Proof of Theorem 3.9. Monotonicity follows from $\mathcal { U } _ { \mathrm { a d } } ( \rho _ { 1 } ) \subset \mathcal { U } _ { \mathrm { a d } } ( \rho _ { 2 } )$ when $\rho _ { 1 } \le \rho _ { 2 }$ . Let $\rho _ { n }  \rho > 0$ and $u _ { n } \in S ( \rho _ { n } )$

For all sufficiently large $n ,$ , we have $\rho / 2 \le \rho _ { n } \le 2 \rho$ . Fix $\bar { u } \in \mathcal { U } _ { \mathrm { a d } } ( \rho / 2 )$ . Then $\bar { u } \in \mathcal { U } _ { \mathrm { a d } } ( \rho _ { n } )$ and $J ( u _ { n } ) = V ( \rho _ { n } ) \leq J ( { \bar { u } } )$ . The coercive estimate in Proposition B.5, whose constants are independent of the budget, makes $\{ u _ { n } \}$ bounded. Moreover, $u _ { n , t } \geq ( 2 \rho _ { n } ) ^ { - 1 } \geq ( 4 \rho ) ^ { - 1 }$ for all large $n ,$ so $\rho$ defined in (2.6) is continuous on a common neighborhood containing the tail of the sequence.

Choose a subsequence $\{ n _ { j } \}$ such that $V ( \rho _ { n _ { j } } ) \to$ lim $\operatorname { i n f } _ { n } V ( \rho _ { n } )$ . Boundedness gives a further subsequence, not relabeled, with $u _ { n _ { i } } \to u$ . Continuity of $\rho$ yields $u \in \mathcal { U } _ { \mathrm { a d } } ( \rho )$ , and continuity of J gives

$$
V ( \rho ) \leq J ( u ) = \operatorname* { l i m } _ { j \to \infty } J ( u _ { n _ { j } } ) = \operatorname* { l i m } _ { n \to \infty } { \mathrm { i n f ~ } } V ( \rho _ { n } ) .
$$

Conversely, choose $u ^ { \star } \in S ( \rho )$ and set $\alpha _ { n } : = \operatorname* { m a x } \left. 1 , \rho \rho _ { n } ^ { - 1 } \right.$ and $\tilde { u } _ { n } : = \alpha _ { n } u ^ { \star }$ . Since $\rho ( \alpha u ) = \alpha ^ { - 1 } \rho ( u )$ , we have $\tilde { u } _ { n } \in \mathcal { U } _ { \mathrm { a d } } ( \rho _ { n } )$ and $\tilde { u } _ { n }  u ^ { \star }$ . Hence

$$
\operatorname* { l i m } _ { n \to \infty } V ( \rho _ { n } ) \leq \operatorname* { l i m } _ { n \to \infty } J ( { \tilde { u } } _ { n } ) = J ( u ^ { \star } ) = V ( \rho ) .
$$

Thus $V ( \rho _ { n } ) \to V ( \rho )$ . If a subsequence $u _ { n _ { k } } \to u ,$ then $u \in \mathcal { U } _ { \mathrm { a d } } ( \rho )$ and

$$
J ( u ) = \operatorname* { l i m } _ { k \to \infty } J ( u _ { n _ { k } } ) = \operatorname* { l i m } _ { k \to \infty } V ( \rho _ { n _ { k } } ) = V ( \rho ) \quad \Rightarrow \quad u \in S ( \rho ) .
$$

If $S ( \rho ) = \{ u ^ { \star } \}$ , boundedness and uniqueness of every cluster point imply $u _ { n } \to u ^ { \star }$

## B.4.3. Design-to-training utility stability.

Proof of Theorem 3.10. Couple the design and target dynamics using the same Gaussian innovations. Their noise terms cancel. Global Lipschitz continuity therefore gives

$$
\big \| \theta _ { T } ^ { \mathrm { t a r } } - \theta _ { T } ^ { \mathrm { d e s } } \big \| \leq A \left( \big \| \theta _ { 0 } ^ { \mathrm { t a r } } - \theta _ { 0 } ^ { \mathrm { d e s } } \big \| + \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \varepsilon _ { t } ^ { g } \right) , \qquad \mathrm { w i t h ~ } A : = \prod _ { t = 0 } ^ { T - 1 } ( 1 + \eta _ { t } L _ { t } ^ { \mathrm { t a r } } ) .
$$

By compactness of K and Lemma B.2, both objectives are continuous and

$$
B _ { a } : = \operatorname* { s u p } _ { u \in { \mathcal K } } \bigl ( { \mathbb E } \| \theta _ { T } ^ { a } ( u ) - \theta ^ { * , a } \| ^ { 2 } \bigr ) ^ { 1 / 2 } < \infty .
$$

Applying $\left| \| x \| ^ { 2 } - \| y \| ^ { 2 } \right| \leq \| x - y \| ( \| x \| + \| y \| )$ to the terminal tracking errors and using Cauchy-Schwarz proves the uniform objective bound with $C _ { K } = A ( B _ { \mathrm { t a r } } + B _ { \mathrm { d e s } } )$ . Evaluating this bound at minimizers gives the minimum-value estimate. Finally,

$$
J ^ { \mathrm { t a r } } ( \widehat { u } ) \leq J ^ { \mathrm { d e s } } ( \widehat { u } ) + \varepsilon _ { K } \leq \operatorname* { m i n } _ { \kappa } J ^ { \mathrm { d e s } } ( u ) + \gamma + \varepsilon _ { K } \leq \operatorname* { m i n } _ { \kappa } J ^ { \mathrm { t a r } } ( u ) + \gamma + 2 \varepsilon _ { K } .
$$

## Appendix C. Proofs of the results in Section 4

## C.1. Proofs: PB-AMD: construction and algorithm.

Proof of Proposition 4.1. Strict convexity of $D _ { \psi } ( \cdot , q ^ { k } )$ gives uniqueness. The KKT conditions of the prox subproblem yield, for coordinates above the floor, log $; ( q _ { t } ^ { k + 1 } / q _ { t } ^ { k } ) + \alpha _ { k } h _ { t } ^ { k } + \alpha _ { k } \lambda _ { k } = 0 ,$ , while a floor coordinate is obtained whenever the corresponding unconstrained value is at most qmin. Setting $\tau _ { k } = e ^ { - \alpha _ { k } \lambda _ { k } }$ gives (4.2). The left-hand side of (4.3) is continuous and strictly increasing once it exceeds $T q _ { \mathrm { m i n } }$ , ranges from $T q _ { \mathrm { m i n } }$ to +∞, and therefore has a unique solution because $\rho _ { \mathrm { m a x } } > T q _ { \mathrm { m i n } }$ □

## C.2. Proofs: Convergence guarantees.

Proposition C.1. Assume Assumptions (A1), (A2), (A3), (A6), and (A7). Fix $q _ { \mathrm { m i n } } > 0$ and let $Q = Q ( q _ { \mathrm { m i n } } )$ Then there is an open neighborhood Q of Q and a nonnegative random variable $\Gamma ( \xi )$ with finite moments of every order such that, for all $q , z \in \mathcal { Q }$ 2，

$$
\left\| \nabla f ( q ; \xi ) \right\| _ { 2 } \leq \Gamma ( \xi ) , \qquad \left\| \nabla f ( q ; \xi ) - \nabla f ( z ; \xi ) \right\| _ { 2 } \leq \Gamma ( \xi ) \left\| q - z \right\| _ { 2 }\tag{C.1}
$$

almost surely. Consequently:

(i) $F \in C ^ { 1 } ( \mathcal { Q } )$ and ∇F is Lipschitz on Q with $\left\| \nabla F ( q ) - \nabla F ( z ) \right\| _ { 2 } \leq L _ { \mathrm { E } } \left\| q - z \right\| _ { 2 }$ , where $L _ { \mathrm { E } } : = \mathbb { E } \Gamma ( \xi ) < \infty ;$ (ii) $( B 1 ) \ – ( B 2 )$ hold, and one may take $L _ { F } = L _ { \mathrm { E } } / \mu _ { \psi } ;$

(iii) for an average of $M _ { k }$ fresh independent path gradients, (B3) holds with $\begin{array} { r } { \nu ^ { 2 } : = \operatorname* { s u p } _ { q \in Q } \mathbb { E } \left\| \nabla f ( q ; \xi ) - \nabla F ( q ) \right\| _ { 2 } ^ { 2 } < } \end{array}$ ∞.

Proof. Choose a bounded open convex neighborhood Q of Q with $\overline { { \mathcal { Q } } } \in ( 0 , \infty ) ^ { T }$ , and put $\beta _ { t } ( r ) : = \eta _ { t } \Delta _ { t } ^ { b } ( 2 r ) ^ { - 1 / 2 }$ $\gamma _ { t } ( r ) : = \eta _ { t } \Delta _ { t } ^ { b } ( 2 r ) ^ { - 3 / 2 }$ , and $\begin{array} { r } { \Xi ( \xi ) : = 1 + \sum _ { t = 0 } ^ { T - 1 } \| \xi _ { t } \| } \end{array}$

The functions $\beta _ { t } , \gamma _ { t } ,$ and their first derivatives are uniformly bounded on every coordinate projection of ${ \overline { { \mathcal { Q } } } } .$ The associated controls $u _ { t } = ( 2 q _ { t } ) ^ { - 1 }$ form a bounded set, so Lemma B.2 and the forward recursion therefore give a polynomial Gaussian envelope for all states, uniformly over $q \in \mathcal { Q }$

Let $S _ { s , j } ( q ) : = \partial _ { q _ { j } } \theta _ { s } ( q )$ . Differentiating the finite recursion gives

$$
S _ { s , j } ( q ) = 0 ( s \leq j ) , \quad S _ { j + 1 , j } ( q ) = \gamma _ { j } ( q _ { j } ) \xi _ { j } , \quad S _ { s + 1 , j } ( q ) = A _ { s } ( \theta _ { s } ( q ) ) S _ { s , j } ( q ) ( s \geq j + 1 ) .
$$

Applying the polynomial Jacobian and Hessian bounds in Assumptions (A6) and (A7) recursively to the forward states and sensitivities and the backward adjoints yields uniform polynomial estimates. Hence there are deterministic $C < \infty$ and an integer $m \geq 1$ such that, for all $q , z \in \mathcal { Q }$

$$
\operatorname* { m a x } _ { s } \| \theta _ { s } ( q ) \| + \operatorname* { m a x } _ { s , j } \| S _ { s , j } ( q ) \| + \operatorname* { m a x } _ { s } \| p _ { s } ( q ) \| \leq C \Xi ( \xi ) ^ { m }
$$

and

$$
\operatorname* { m a x } _ { s } \| \theta _ { s } ( q ) - \theta _ { s } ( z ) \| + \operatorname* { m a x } _ { s , j } \| S _ { s , j } ( q ) - S _ { s , j } ( z ) \| + \operatorname* { m a x } _ { s } \| p _ { s } ( q ) - p _ { s } ( z ) \| \leq C \Xi ( \xi ) ^ { m } \| q - z \| _ { 2 }
$$

almost surely. Applying these estimates to $\partial _ { q _ { j } } f ( q ; \xi ) = \gamma _ { j } ( q _ { j } ) \langle p _ { j + 1 } ( q ) , \xi _ { j } \rangle$ proves (C.1) with $\Gamma ( \xi ) =$ $C ^ { \prime } ( 1 + \Xi ( \xi ) ^ { m ^ { \prime } } )$ for suitable deterministic $C ^ { \prime }$ and integer $m ^ { \prime }$ . Since Ξ is a finite sum of Gaussian norms, Γ has moments of every finite order. The envelope justifies differentiation under the expectation and yields $\| \nabla F ( q ) - \nabla F ( z ) \| _ { 2 } \leq \| \nabla [ \xi ) ] \| q - z \| _ { 2 }$

The Euclidean descent lemma and the lower bound in (4.4) give (B2) with $L _ { F } = \mathbb { E } [ \Gamma ] / \mu _ { \psi }$ , while continuity on compact $Q$ gives (B1). Finally, the square-integrable envelope makes $\nu ^ { 2 }$ finite, and independence of the $M _ { k }$ paths gives the variance reduction in (B3). □

Proof of Theorem 4.2. For the stochastic estimate, write $x = q ^ { k } , y = q ^ { k + 1 } , v = \nabla F ( x ) , e = h ^ { k } - v$ , and $z = T _ { \alpha } ( x )$ . Relative smoothness and the stochastic prox inequality give

$$
F ( y ) \leq F ( x ) - ( 1 / \alpha - L _ { F } ) D _ { \psi } ( y , x ) - \left. e , y - x \right. .
$$

Testing the exact prox inequality with $y ,$ testing the stochastic prox inequality with z, and adding the two inequalities yields

$$
\| y - z \| _ { 2 } \leq \frac { \alpha } { \mu _ { \psi } } \left\| e \right\| _ { 2 } .
$$

Conditionally on $\mathcal { F } _ { k }$ , the vectors x and $z - x$ are fixed and $\mathbb { E } [ e \mid \mathcal { F } _ { k } ] = 0$ . Hence

$$
\mathbb { E } [ - \left. e , y - x \right. \mid \mathcal { F } _ { k } ] \leq \frac { \alpha } { \mu _ { \psi } } \mathbb { E } [ \left\| e \right\| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { k } ] .
$$

Summing this descent estimate and using (B3) bounds the average of $D _ { \psi } ( q ^ { k + 1 } , q ^ { k } )$ . Moreover,

$$
D _ { \psi } ( z , x ) \le \frac { 2 M _ { \psi } } { \mu _ { \psi } } D _ { \psi } ( y , x ) + \frac { M _ { \psi } \alpha ^ { 2 } } { \mu _ { \psi } ^ { 2 } } \left\| e \right\| _ { 2 } ^ { 2 } ,
$$

by (4.4) and the preceding prox stability. Division by $\alpha ^ { 2 }$ and averaging give (4.8).

For exact gradients, prox optimality and relative smoothness reduce to

$$
F ( q ^ { k + 1 } ) \leq F ( q ^ { k } ) - ( 1 / \alpha - L _ { F } ) D _ { \psi } ( q ^ { k + 1 } , q ^ { k } ) .
$$

Telescoping gives (4.9) and $\begin{array} { r } { \sum _ { k } R _ { \alpha } ( q ^ { k } ) < \infty } \end{array}$ . Compactness of $Q ,$ continuity of $T _ { \alpha }$ , and (4.7) then give the cluster-point claim. □

## C.3. Proofs: Fixed-path/SAA implementation.

Proof of Proposition 4.3. For $z = T _ { \alpha , H } ( q )$ , the prox variational inequality and (4.4) give

$$
G _ { H } ( q ) \leq ( \alpha B _ { H } + M _ { \psi } D _ { Q , 2 } ) \sqrt { \frac { 2 R _ { \alpha , H } ( q ) } { \mu _ { \psi } } } , \qquad B _ { H } : = \operatorname* { m a x } _ { q \in Q } \| \nabla H ( q ) \| _ { 2 } .
$$

Indeed, for an arbitrary $w \in Q$ , one writes $q - w = ( q - z ) + ( z - w )$ and bounds the second term by $\alpha ^ { - 1 } \| \nabla \psi ( z ) - \nabla \psi ( q ) \| _ { 2 } \| w - z \| _ { 2 }$ . Moreover, $B _ { F _ { M } } \leq B _ { F } + \sqrt { T } \varepsilon _ { M }$ and, by Hölder’s inequality, $G _ { F } ( q ) \leq$ $G _ { F _ { M } } ( q ) + D _ { Q , 1 } \varepsilon _ { M }$ . Combining these estimates at $q = q _ { M }$ proves (4.12). □

Proof of Corollary $4 . 4 \cdot$ The envelope (C.1) is integrable. On any finite net of compact $Q ,$ , the strong law of large numbers applies simultaneously to the finitely many gradient vectors and to the empirical mean of Γ. The Lipschitz part of (C.1) extends convergence from the net to all of $Q ;$ ; letting the mesh size tend to zero gives the displayed uniform convergence, as in the standard finite-dimensional SAA argument [22, 33]. Proposition 4.3 then gives population stationarity. The final assertion follows from the bound $\begin{array} { r } { L _ { F _ { M } } \le M ^ { - 1 } \sum _ { m = 1 } ^ { M } { \Gamma ( \xi ^ { ( m ) } ) } / { \mu _ { \psi } } } \end{array}$ , the strong law, and the exact-gradient part of Theorem 4.2. □

## C.4. Proofs: Linear-quadratic model.

Proof of Proposition 4.5. The case $a _ { t } = 0$ for all t is immediate. Otherwise, let $P : = \{ t : a _ { t } > 0 \}$ and $Z : = \{ t : a _ { t } = 0 \}$ . At any minimizer, $q _ { t } = q _ { \mathrm { m i n } }$ for $t \in Z \colon$ : transferring budget from such a coordinate above the floor to a coordinate in P would strictly decrease F. On the remaining coordinates, the objective is strictly convex, so compactness of Q gives a unique minimizer. The KKT conditions on $P$ are

$$
- \frac { a _ { t } } { ( q _ { t } ^ { \star } ) ^ { 2 } } + \lambda ^ { \star } - \mu _ { t } = 0 , \qquad \mu _ { t } \geq 0 , \qquad \mu _ { t } ( q _ { t } ^ { \star } - q _ { \operatorname* { m i n } } ) = 0 .
$$

Since $\rho _ { \mathrm { m a x } } > T q _ { \mathrm { m i n } }$ , some coordinate in P lies above the floor, implying ${ { \lambda } ^ { \star } } > 0$ . These conditions give (4.14), including the coordinates in Z. Its normalization function is continuous and strictly decreasing wherever its value exceeds $T q _ { \mathrm { m i n } } ,$ with limits +∞ and $T q _ { \mathrm { m i n } }$ as $\lambda \downarrow 0$ and $\lambda \uparrow \infty$ , respectively. Hence $\lambda ^ { \star }$ is unique. When every coordinate exceeds the floor, summing $q _ { t } ^ { \star } = \sqrt { a _ { t } / \lambda ^ { \star } }$ yields (4.15). □

Proof of Corollary $4 . 6 .$ The Hessians satisfy

$$
\nabla ^ { 2 } F ( q ) = \mathrm { d i a g } \left( { \frac { 2 a _ { t } } { q _ { t } ^ { 3 } } } \right) \preceq L _ { \mathrm { L Q } } \mathrm { d i a g } \left( { \frac { 1 } { q _ { t } } } \right) = L _ { \mathrm { L Q } } \nabla ^ { 2 } \psi ( q ) .
$$

Thus F is convex and $L _ { \mathrm { L Q } }$ -smooth relative to $\psi .$ . The three-point inequality gives

$$
F ( q ^ { k + 1 } ) - F ( q ^ { \star } ) \leq \frac { 1 } { \alpha } \left[ D _ { \psi } ( q ^ { \star } , q ^ { k } ) - D _ { \psi } ( q ^ { \star } , q ^ { k + 1 } ) \right] .
$$

Telescoping and convexity yield (4.16) and $F ( q ^ { k } ) \to F ( q ^ { \star } )$ . Compactness of $Q$ and uniqueness from Proposition 4.5 then imply $q ^ { k } \to q ^ { \star }$ □

## Appendix D. Experimental design

This appendix specifies the experiments in Section 5. The studies have two distinct aims:

• Allocation quality. We compare terminal parameter tracking and prediction under different privacy schedules. Section D.1 defines the models, data, and evaluation samples, and Section D.2 specifies how schedules are computed and selected. These settings apply to Figures 3--5 and Table 4 of the main article, and to the budget and implementation checks in Sections D.2.3 and D.2.5 below.

• Solver convergence. We measure the time needed to reach the same first-order tolerance when computing a schedule from public records. Section D.3 specifies the independent instances, stepsize calibration, and stopping rules for Figure 6; Section D.4 defines the timed operations.

Within each solver comparison, Privacy-Budget Adjoint Mirror Descent (PB-AMD), projected gradient descent (PGD), and penalty optimization use the same fixed-path sample-average tracking term $F _ { M }$ in (4.10), Gaussian paths, and noise calibration. PB-AMD and PGD enforce the budget directly. Penalty + repair first minimizes a budget-penalized objective and then rescales the charges to meet the exact budget.

D.1. Models, data, and evaluation samples. We first specify the records and models used in Section 5.1 of the main article. Public records are used to construct schedules; private training records are excluded from schedule construction and used to train models under the fixed schedules; held-out test records measure prediction after training. Where preprocessing or pretraining is needed, we compute it from public auxiliary records. The public-design sample contains the records used to optimize and select schedules. The sample sizes and, for Fashion-MNIST, the subset relation between these two public samples are given below.

The optimization variables are the per-step privacy charges $q _ { t } = ( 2 u _ { t } ) ^ { - 1 }$ from (2.5) of the main article, where $u _ { t }$ is the Gaussian noise-variance parameter. For T training iterations, the exact-budget feasible set is $\left( 3 . 1 0 \right)$ of the main article: $Q = \{ q \in \mathbb { R } ^ { T } : q _ { t } \geq$ qmin, $\sum _ { t } q _ { t } = \rho _ { \mathrm { m a x } } \}$ , with $\rho _ { \mathrm { m a x } }$ the total certified budget and $q _ { \mathrm { m i n } }$ the per-step charge floor. We use $q _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ in the learning experiments. The separate linear--quadratic (LQ) check in Section D.2.5 uses $q _ { \mathrm { m i n } } = 1 0 ^ { - 3 }$ . The budgets are 0.25, 0.5, 1 for Tabular and the two full-network tasks, Fashion-constant and Fashion-restart, and 0.5, 0.25 for Nonlinear head.

Table 7 gives the training parameters. At step t, Gaussian noise is added to the gradient query with standard deviation $( 2 C _ { \mathrm { c l i p } } / b ) / \sqrt { 2 q _ { t } }$ in each coordinate, where b is the batch size and $C _ { \mathrm { c l i p } }$ bounds the norm of each clipped data gradient. Individual data gradients v are smoothly clipped as $v / \sqrt { 1 + \left. v \right. ^ { 2 } / C _ { \mathrm { c l i p } } ^ { 2 } }$ , then averaged; the regularization gradient $\lambda _ { \mathrm { { r e g } } } \theta _ { \mathrm { { i } } }$ , where θ is the parameter vector, is added afterwards, including

for neural-network biases; $\lambda _ { \mathrm { r e g } }$ controls the quadratic regularization. We use replace-one adjacency and compute the total certified privacy charge by summing the per-step charges, as in (2.6) of the main article. We do not reduce this charge to account for random batch selection.
<table><tr><td>Task</td><td>Parameters</td><td> $T$ </td><td>b</td><td>Learning rate  $\eta _ { t }$ </td><td> $C _ { \mathrm { c l i p } }$ </td><td> $\lambda _ { \mathrm { r e g } }$ </td></tr><tr><td>Tabular</td><td>64</td><td>60</td><td>256</td><td>0.18</td><td>4</td><td>0.002</td></tr><tr><td>Fashion-constant</td><td>6634</td><td>45</td><td>192</td><td>0.15</td><td>1.75</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Fashion-restart</td><td>6634</td><td>45</td><td>192</td><td>0.02/0.075</td><td>1.75</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Nonlinear head</td><td>86</td><td>120</td><td>384</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td></tr></table>

Table 7. Training configurations. The restart uses 0.02 for $t < 2 5$ and 0.075 for $t \geq 2 5$ with $t = 0 , \ldots , T - 1$

D.1.1. Synthetic tabular population. Tabular fits a linear logistic classifier to a population whose labels also depend on nonlinear feature interactions. The public-design, private-training, and test samples contain 4500, 4500, and 1500 records, respectively. Covariates $x \in \mathbb { R } ^ { 6 4 }$ are centered Gaussian with covariance $\Sigma _ { i j } = 0 . 6 5 ^ { | i - j | } + 1 0 ^ { - 8 } \mathbf { 1 } _ { \{ i = j \} }$ . Using zero-based feature indices, the nonzero entries of w are

$$
\frac { i } { w _ { i } ~ | ~ 1 . 7 ~ - 1 . 4 ~ 1 . 1 ~ - 1 ~ . 9 ~ - . 8 ~ . 7 ~ - . 6 5 ~ . 6 ~ - . 5 5 ~ . 5 ~ - . 4 5 ~ }
$$

Before random label flips, the probability of a positive label is

$$
\mathrm { s i g m o i d } \Bigg ( \frac { 1 . 3 5 } { 2 . 4 3 } \left[ x ^ { \top } w + 0 . 8 5 x _ { 0 } x _ { 1 } - 0 . 6 5 x _ { 2 } x _ { 3 } + 0 . 4 5 \sin x _ { 4 } + 0 . 3 5 \mathbf { 1 } _ { \left\{ x _ { 5 } > 0 \right\} } \right] \Bigg ) .
$$

Here sigmoid $( z ) = ( 1 + e ^ { - z } ) ^ { - 1 }$ . Each label is independently flipped with probability 0.05. The fitted logistic loss is log $( 1 + e ^ { - y x ^ { \top } } \theta )$ , with $y \in \{ - 1 , 1 \}$ , no intercept, and initialization $\theta _ { 0 } = 0$ . Features are not standardized. The public-design sample is generated independently of the 6000 records split into private-training and test samples.

D.1.2. Fashion-MNIST and the frozen-feature head. For each seed, the official training split is randomly permuted. The first 8000 records form the private training sample; the next 12000 form the public auxiliary sample. The first 8000 records within that auxiliary sample form the public-design sample used for schedule optimization, public validation, and the public diagnostic evaluation in Figure 4(c) of the main article. The test sample comprises 3000 seed-selected records from the official test split. Pixel intensities are divided by 255, averaged over nonoverlapping $2 \times 2$ blocks, and flattened to 196 features. Each feature is centered by its mean over the 12000 public records and divided by its public standard deviation plus $1 0 ^ { - 6 }$ . The same transformation is applied to public, private, and test data.

The full-network classifier is 196--32--10, with Softplus hidden activation $\log ( 1 + e ^ { z } )$ and softmax crossentropy data loss. The weight matrices have independent centered Gaussian entries with variances 1/196 and $1 / 3 2$ , respectively; both bias vectors start at zero. Fashion-constant uses this random initialization. Fashion-restart first performs 300 noiseless public stochastic gradient descent (SGD) steps on the 12000 public auxiliary records, using learning rate 0.05, batch size 192, clipping 1.75, and regularization $1 0 ^ { - 4 }$ . The resulting weights initialize both public-design simulations and private training. All 6634 parameters are updated and perturbed in subsequent private training.

Nonlinear head uses the first layer obtained by the same public pretraining procedure as Fashion-restart, then freezes it. Its 32-dimensional Softplus outputs are centered using the 8000 public-design records; principal component analysis is computed by full singular-value decomposition on that sample. We project each centered feature vector onto the first eight principal directions, then divide each projected coordinate by its standard deviation on the public-design sample plus $1 0 ^ { - 6 }$ . The sign of each principal direction is fixed by requiring its largest-magnitude loading to be positive. The 8--4--10 Softplus head has 86 trainable parameters. Its weight matrices have independent centered Gaussian entries with variances $1 / 8$ and $1 / 4$ , respectively; both bias vectors start at zero. The head has no separate pretraining stage. During training on private records, only the head is updated and perturbed; the encoder pretrained on public auxiliary records and the feature transform computed on public-design records remain fixed.

Each iteration samples batch indices uniformly without replacement, independently of record values. A new batch is sampled at the next iteration, so records may reappear in later batches. The complete batch sequence is fixed before training and shared across methods; identical indices are used for the equally sized public-design and private samples. For schedule optimization on public-design data, the No-noise reference is the final model trained on the public-design sample with Gaussian noise removed. For tracking evaluation on private training data, it is recomputed on the private training sample. In each case, noisy training and its reference share the initialization, batches, clipping, regularization, and learning rates. Only the Gaussian perturbation differs. The reference trained on private records is an offline evaluation target and does not enter schedule optimization or public validation.

D.1.3. Gaussian paths, random seeds, and evaluation metrics. We use the tracking objective (3.2) with loss (3.6) of the main article at each stage, but change the data or Gaussian paths according to that stage’s purpose. A Gaussian path is one complete sequence of standard Gaussian vectors over the training horizon. Table 8 gives the number of paths used in the following three stages:

(1) Schedule optimization. We simulate noisy training on the public-design records. The mean squared distance to the public No-noise reference is $F _ { M } ( q )$ in (4.11). We keep the Gaussian paths fixed while optimizing q.

(2) Public validation. We keep the public-design records and their No-noise reference unchanged, but evaluate schedules with independent Gaussian paths. When different solver parameters or saved iterates yield several schedules (Section D.2), we select the one with lowest validation tracking error. Optimization and validation thus differ in noise samples, not in records.

(3) Evaluation on private training data. After selection, we fix q and simulate training on the separate private records with fresh Gaussian paths. We compute tracking error against the No-noise reference trained on those private records. This evaluates a different training sample and reference from those used in $F _ { M }$

Within a stage, methods share standard Gaussian vectors scaled by their respective schedules. Consequently, their tracking errors can be compared on paired training runs.

<table><tr><td>Task</td><td>Optimization</td><td>Public validation Private evaluation</td><td></td></tr><tr><td>Tabular</td><td>256</td><td>512</td><td>512</td></tr><tr><td>Fashion-constant/restart</td><td>8</td><td>48</td><td>256</td></tr><tr><td>Nonlinear head</td><td>128</td><td>256</td><td>512</td></tr></table>

Table 8. Gaussian path counts for allocation-quality studies. The public diagnostic in Figure 4(c) also uses 512 independent paths.

Ten problem seeds specify the data, initialization, and batches shared across methods. The two head budgets use independent groups of ten seeds. The code repository records the seed identifiers and exact random-number generation settings.

For each model trained on private records, we compute its tracking error against the corresponding No-noise reference and its prediction scores on the held-out test sample. Test accuracy is the fraction of correctly classified test records; unregularized cross-entropy is the mean negative log probability of the true test labels. We then average each metric over Gaussian paths. The full-network prediction comparison in Figure 5 reuses the PB-AMD schedules computed in Section D.2 and uses 256 noisy training runs on private records for each fixed schedule.

To quantify PB-AMD’s temporal redistribution in Figure 3 of the main article, we compare its mean per-step charge in the last third of training with that in the first third. Averaging this ratio over problem seeds gives 1.26 to 1.36 for Tabular across the three budgets and about 3.77 for Fashion-restart at each budget.

Statistical comparisons use the ten problem seeds as the independent units. For each seed, let $J _ { \mathrm { P B } }$ and $J _ { \mathrm { b a s e } }$ denote path-averaged tracking errors on private training data for PB-AMD and the comparator. The relative error reduction is $1 0 0 ( 1 - J _ { \mathrm { P B } } / J _ { \mathrm { b a s e } } )$ , expressed as a percentage of the comparator’s tracking error. We average these ten reductions, rather than taking a ratio of errors averaged across seeds, and compute pointwise 95% confidence intervals with Student’s t distribution and nine degrees of freedom. Prediction means and their pointwise intervals likewise use the ten seed-level accuracy or cross-entropy scores, with the same Student critical value.

Two-sided paired Student tests assess whether the mean tracking reduction, accuracy difference, or cross-entropy difference is zero. Holm’s correction adjusts the p-values to control the probability of any false declaration of significance within a group of comparisons at the 5% level. We use adjusted $p < 0 . 0 5$ ; the confidence intervals above are unadjusted. We apply the correction separately to two groups:

• Eight tracking-error tests: PB-AMD versus learning-rate allocation (LR), Decay, Linearized, and Geometric at two budgets.

• Twelve prediction tests: PB-AMD versus Uniform for Fashion-constant and Fashion-restart, three budgets, and two metrics.

Figure 5 of the main article displays Fashion-restart only, but the prediction correction includes both tasks. The four allocation rules are defined in Section 5.1 of the main article and specified in Section D.2.2 below. All six head comparisons against LR, Decay, and Geometric remain significant, with Holm-adjusted $p < 1 . 3 \times 1 0 ^ { - 5 }$ . The two Linearized comparisons have adjusted $p = 0 . 1 4 4$ and 0.058 at budgets 0.5 and 0.25, respectively, both above the 0.05 threshold.

Diagnostic instances A and B in Figure 4 of the main article were selected after evaluation for the largest PB-AMD gains over Linearized at each budget. Their intervals measure Gaussian-path uncertainty within each fixed problem, rather than variation across problem seeds. Let $P _ { m } , B _ { m }$ be PB-AMD’s and the comparator’s squared parameter tracking errors on the same Gaussian path $m$ . With 512 paths, set $r = \bar { P } / \bar { B }$ ; bars denote path averages. The relative reduction is $1 0 0 ( 1 - r )$ , with delta-method standard error 100 sd $( P _ { m } - r B _ { m } ) / ( \sqrt { 5 1 2 } \bar { B } )$ , using sample standard deviation. To account for the selection of A and B after evaluation, simultaneous intervals use a Bonferroni correction over eighty comparisons (twenty instances, four rules). The critical value is $t _ { 5 1 1 , 1 - 0 . 0 5 / ( 2 \cdot 8 0 ) }$ , where $t _ { \nu , p }$ denotes the p-quantile of Student’s distribution with ν degrees of freedom. For diagnostic A (budget 0.5) and B (budget 0.25), the tracking reductions over Linearized on private training data reported in the main article have simultaneous intervals [31.20, 33.60]% and [8.20, 10.20]%, respectively.

These intervals concern tracking on private training data. Figure $4 ( \mathrm { c } )$ of the main article instead evaluates the fixed schedules of A on public-design records: the linearized objective is computed from (4.14), and the nonlinear error is averaged over 512 independent Gaussian paths on the same public-design records. This diagnostic compares how the linearized objective and nonlinear dynamics rank the two fixed schedules.

D.2. Solvers for the utility comparisons. This section specifies schedule construction for the allocationquality comparisons in Sections 5.2--5.4 of the main article. Section D.3 gives the separate convergence protocol for Figure 6. PB-AMD, PGD, and penalty optimization start from Uniform, $q _ { t } ^ { 0 } = \rho _ { \mathrm { m a x } } / T$ , and use the public fixed-path tracking term $F _ { M }$ defined in Section D.1.3. PB-AMD and PGD minimize this term on $Q ;$ the penalty method minimizes it with the additional budget-excess term in Section D.2.3.

For Tabular and full-network training, Table 9 gives the PB-AMD stepsize, the PGD stepsize grid, and the update counts. Each stepsize is applied to the unscaled gradient $\nabla F _ { M }$ . The returned schedules are selected as follows:

• PB-AMD retains the iterate with lowest $F _ { M }$ , including $q ^ { 0 }$ . After the listed minimum number of updates, it stops when that iterate’s absolute Karush--Kuhn--Tucker (KKT) residual is at most $5 \times 1 0 ^ { - 5 }$ . The limit is 2000 updates; a run that fails this criterion is rejected. Section D.2.1 defines the residual.

• PGD runs each of the four stepsizes for the listed fixed number of updates and retains the lowest- $F _ { M }$ iterate from each run. Public validation on independent Gaussian paths selects the best of these four returned schedules by tracking error on the public-design records.

• Penalty + repair runs the six parameter pairs in Table 10 for the same fixed update count. Section D.2.3 specifies the raw optimizer and budget repair; public validation selects the repaired schedule with lowest tracking error on the public-design records.

Nonlinear head instead uses the Armijo and checkpoint-selection procedure in Section D.2.2.

<table><tr><td>Task</td><td></td><td>PB-AMD step Minimum/fixed updates PGD stepsize grid</td><td></td></tr><tr><td>Tabular</td><td>0.005</td><td>180</td><td> $\{ 3 \cdot 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ </td></tr><tr><td>Fashion-constant</td><td>0.0005</td><td>100</td><td> $\{ 3 \cdot 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 3 \cdot 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td></tr><tr><td>Fashion-restart</td><td>0.005</td><td>100</td><td> $\{ 3 \cdot 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 3 \cdot 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td></tr></table>

Table 9. Utility solver settings. PB-AMD checks the KKT tolerance after the listed minimum update count; PGD and penalty use the listed fixed count.

D.2.1. Budget-preserving updates and stationarity measures. PB-AMD implements the floor-aware entropic step (4.2) in Algorithm 1 of the main article. Bisection in logarithmic coordinates solves the normalization equation (4.3), with at most 160 iterations and a tolerance $1 0 ^ { - 1 3 } \operatorname* { m a x } \{ 1 , \rho _ { \operatorname* { m a x } } \}$ on the difference between total charge and ρmax.

PGD projects $q - \alpha \nabla F _ { M } ( q )$ onto the same floored simplex by sorting coordinates and computing the Euclidean projection threshold. Both methods correct summation roundoff on the largest coordinate.

For $h = \nabla F _ { M } ( q )$ , set $I = \{ t : q _ { t } > q _ { \mathrm { m i n } } + 1 0 ^ { - 9 } \}$ and $\begin{array} { r } { \bar { h } _ { I } = | I | ^ { - 1 } \sum _ { t \in I } h _ { t } } \end{array}$ . The reported absolute KKT residual is

$$
\operatorname* { m a x } \left\{ \operatorname* { m a x } _ { t \in I } | h _ { t } - \bar { h } _ { I } | , \operatorname* { m a x } _ { t \notin I } [ \bar { h } _ { I } - h _ { t } ] _ { + } \right\} ,
$$

where $[ a ] _ { + } = \operatorname* { m a x } \{ a , 0 \}$ , and the second maximum is zero if all coordinates belong to I. The first term tests equality of gradient coordinates above the floor; the second tests the one-sided KKT inequality at the floor. Table 11 reports this absolute residual for the returned PB-AMD schedules. Budget feasibility is checked separately.

Figure 6 instead compares all three solvers with the variational gap (4.12) of the main article, applied to $F _ { M } \colon$

$$
G _ { F _ { M } } ( q ) = h ^ { \top } q - q _ { \operatorname* { m i n } } \sum _ { t } h _ { t } - ( \rho _ { \operatorname* { m a x } } - T q _ { \operatorname* { m i n } } ) \operatorname* { m i n } _ { t } h _ { t } .
$$

We replace a negative roundoff value by zero. For Penalty + repair, we evaluate this gap at the repaired charges defined in Section D.2.3. Writing $F _ { 0 } = F _ { M } ( q ^ { 0 } )$ for the tracking error on public-design records at the Uniform schedule, we divide the gap by $F _ { 0 }$ to obtain the common convergence measure $G _ { F _ { M } } ( q ) / F _ { 0 }$ . The absolute KKT residual controls acceptance of PB-AMD outputs in the utility comparisons; the normalized variational gap compares all three solvers in Figure 6 at a common accuracy threshold.

D.2.2. Armijo updates and comparison schedules for the head. The head comparison in Table 4 and Figure 4 tests PB-AMD against the four allocation rules defined in Section 5.1 of the main article. PB-AMD takes exactly 20 mirror updates and selects among iterates 0, 10, 20 by public validation, using 256 new Gaussian paths on the public-design records. The initial step is $\alpha _ { 0 } = 0 . 3 \rho _ { \mathrm { m a x } } / F _ { 0 }$ . We form each trial $q ^ { \mathrm { t r } }$ using the entropic update (4.2) of the main article and accept it when

$$
F _ { M } ( \boldsymbol { q } ^ { \mathrm { t r } } ) \leq F _ { M } ( \boldsymbol { q } ) + 0 . 1 \left. \nabla F _ { M } ( \boldsymbol { q } ) , \boldsymbol { q } ^ { \mathrm { t r } } - \boldsymbol { q } \right. + 1 0 ^ { - 1 2 } \operatorname* { m a x } \{ 1 , F _ { M } ( \boldsymbol { q } ) \} .
$$

Rejected trials halve the step, with at most 30 trials per update. The next initial trial stepsize is the accepted step multiplied by 1.1, capped at $2 \rho _ { \mathrm { m a x } } / F _ { 0 }$ . All trials use the same fixed Gaussian paths.

The comparison rules assign positive weights $w _ { t }$ and use $q _ { t } = \operatorname* { m a x } \{ q _ { \operatorname* { m i n } } , c w _ { t } \}$ with $\textstyle \sum _ { t } q _ { t } = \rho _ { \operatorname* { m a x } } ;$ c is found by 90 bisection iterations. Thus all rules satisfy the same floor and exact budget as PB-AMD, but use different information to choose the weights:

• LR uses $w _ { t } = \eta _ { t }$ , retaining only the learning-rate scale.

• Decay uses $\begin{array} { r } { w _ { t } = \eta _ { t } \prod _ { s = t + 1 } ^ { I ^ { \prime } - 1 } \left| 1 - \eta _ { s } \lambda _ { \mathrm { r e g } } \right| } \end{array}$ , accounting for subsequent regularization shrinkage.

• Linearized uses the Jacobians along the No-noise reference trajectory $( \bar { \theta } _ { s } )$ on public-design records. With $A _ { s } = I _ { d } - \eta _ { s } D g _ { s } ( { \bar { \theta } } _ { s } )$ , set

$$
a _ { t } = \frac { \eta _ { t } ^ { 2 } ( 2 C _ { \mathrm { c l i p } } / b ) ^ { 2 } } { 2 } \left\| A _ { T - 1 } \cdot \cdot \cdot A _ { t + 1 } \right\| _ { F } ^ { 2 } , \qquad w _ { t } = \sqrt { a _ { t } } .
$$

Here $I _ { d }$ is the identity on the d trainable parameters, $g _ { s }$ is the regularized clipped-gradient query for the fixed batch at step $s , D g _ { s }$ is its Jacobian, and $\| \cdot \| _ { F }$ is the Frobenius norm. This is the coefficient in (4.14) of the main article with identity terminal tracking weight. The empty matrix product is the identity. We evaluate the full matrix products rather than estimating their Frobenius norms with random directions.

• Geometric uses $w _ { t } = \eta _ { t } e ^ { \gamma t / ( T - 1 ) }$ . To choose its shape parameter, public validation first searches $\gamma = - 4 , - 3 . 5 , \ldots , 4 .$ , then takes 18 golden-section iterations on [max $\{ - 4 , \gamma _ { 0 } - 0 . 5 \}$ , min $\{ 4 , \gamma _ { 0 } + 0 . 5 \} ]$ where $\gamma _ { 0 }$ is the grid point with lowest public-validation error. The best grid schedule is retained if refinement does not lower its validation error.

D.2.3. Penalty optimization and exact-budget repair. The penalty method replaces exact normalization by the quadratic budget-excess term in (4.4) of the main article. Its raw objective is $\begin{array} { r } { F _ { M } ( q ) + \frac { \beta } { 2 } [ \sum _ { t } q _ { t } - \rho _ { \mathrm { m a x } } ] _ { + } ^ { 2 } } \end{array}$ in log-variance coordinates $v _ { t } = - \log ( 2 q _ { t } ) = \log u _ { t }$ . With $h = \nabla F _ { M } ( q )$ , the gradient with respect to v is $\begin{array} { r } { - q _ { t } h _ { t } - \beta [ \sum _ { s } q _ { s } - \rho _ { \operatorname* { m a x } } ] _ { + } q _ { t } } \end{array}$ . Adam uses moment coefficients 0.9, 0.999, bias correction, and denominator stabilizer $1 0 ^ { - 8 }$ . The initial moments are zero. After each update $v _ { t }$ is clipped to $[ - \log ( 4 \rho _ { \operatorname* { m a x } } ) , - \log ( 2 \cdot 1 0 ^ { - 8 } ) ]$ so that raw charges lie in $[ 1 0 ^ { - 8 } , 2 \rho _ { \mathrm { m a x } } ]$ . We return the iterate with lowest penalized objective among all generated iterates, including the initial and final points.

<table><tr><td> $\beta$ </td><td>1</td><td>3</td><td>10</td><td>100</td><td>1000</td><td>10000</td></tr><tr><td>Tabular Adam stepsize</td><td>.04</td><td>.04</td><td>.035</td><td>.025</td><td>.015</td><td>.008</td></tr><tr><td>Full-network/head Adam stepsize</td><td>.035</td><td>.032</td><td>.025</td><td>.017</td><td>.01</td><td>.006</td></tr></table>

Table 10. Penalty candidates. Each column specifies one pair of penalty weight and Adam stepsize, not a Cartesian-product grid.

Figure 7 uses the raw charges to assess budget enforcement. For fixed-budget tracking comparisons, Penalty + repair instead replaces the raw vector using $z _ { t } = [ q _ { t } ^ { \mathrm { r a w } } - q _ { \mathrm { m i n } } ] _ { + }$ and $\begin{array} { r } { q _ { t } = q _ { \operatorname* { m i n } } + ( \rho _ { \operatorname* { m a x } } - T q _ { \operatorname* { m i n } } ) z _ { t } / \sum _ { s } z _ { s } . } \end{array}$ If $\textstyle \sum _ { s } z _ { s } = 0$ , the repaired vector is uniform. Public validation selects the repaired candidate with lowest tracking error on the public-design records.

Across the 90 Tabular and full-network task--budget instances, all 540 raw candidates from the six penalty-weight/stepsize pairs violate the budget, with an overall median excess of 63%. Figure 7(a), (b), and (c) show the charge-to-budget ratios for Tabular, Fashion-constant, and Fashion-restart, respectively. Across the tested pairs, larger penalty weights reduce the excess, but none yields a feasible raw candidate. Repair supplies the exact-budget schedules used for utility and convergence comparisons. PB-AMD instead enforces the budget at each mirror update, so its output needs no feasibility repair.

(a) Tabular  
![](images/ed1cf7e921dfc2ae0898d0e304b4da34c8bd9a5566c2f2c521698c8b1288efcd.jpg)

(b) Fashion–constant  
![](images/a64b62e2b5c72930eb840c4b9ac9f91e48b3c47e41ab0a62f34278e830c834aa.jpg)

(c) Fashion–restart  
![](images/c2f8bd4a6db2f8e0658e65695dc86ea08d42146bc7017ba6098f784879e3b779.jpg)  
Figure 7. Raw penalty charge divided by the prescribed budget: medians and ranges over three budgets and ten seeds. Ratios above the dashed line at one violate the prescribed budget.

D.2.4. Public-data perturbations. We test whether PB-AMD retains its Fashion-restart tracking benefit when the public-design sample changes but private training data do not (Section 5.2 of the main article). We redesign the Fashion-restart schedule at $\rho _ { \mathrm { m a x } } = 0 . 5$ after perturbing the public-design data, then evaluate it on the unchanged private training sample. Public pretraining, feature normalization, private records, and the No-noise reference trained on private records remain fixed. We recompute the public No-noise reference on the perturbed design data.

Both perturbations are applied to the same public-design sample. Exactly 800 labels are replaced by uniformly sampled different classes. Standardized design features are mapped to $0 . 9 x + 0 . 1 5 \zeta$ , where ζ has independent standard Gaussian entries.

PB-AMD uses 100 fixed-step updates of size 0.005 and returns the iterate with lowest $F _ { M }$ . No additional updates are performed to meet a KKT tolerance. The Gaussian path counts for optimization, public validation, and evaluation on private training data are 8, 48, 256. The validation paths evaluate the returned schedule; they do not select an additional checkpoint. The study uses the same ten full-network problem seeds as the allocation-quality comparisons. We compare the redesigned schedule with Uniform using the same Gaussian paths for training on private records, then average the ten seed-level relative tracking reductions and compute the pointwise Student interval defined in Section D.1.3.

The redesigned schedules retain a mean tracking-error reduction of 27.45% over Uniform at $\rho _ { \mathrm { m a x } } = 0 . 5$ with a 95% confidence interval of [27.44, 27.47]%. Thus a schedule redesigned from the perturbed public data still improves private-training tracking, without using private records for schedule construction.

D.2.5. Stationarity, budget error, and the LQ optimizer. Table 11 checks stationarity and exact-budget preservation for Tabular, Fashion-constant, and Fashion-restart, and checks objective accuracy against the analytical LQ optimum. The KKT entry is the largest residual among the 90 returned PB-AMD schedules: three tasks, three budgets, and ten seeds per combination. Budget error is the largest $\begin{array} { r } { \lvert \sum _ { t } q _ { t } ^ { k + 1 } - \rho _ { \mathrm { m a x } } \rvert } \end{array}$ after a mirror update and summation-roundoff correction across these runs.
<table><tr><td>Check</td><td>Value</td><td>Scope</td></tr><tr><td>Maximum fixed-path KKT residual</td><td> $5 . 0 0 \times 1 0 ^ { - 5 }$ </td><td>90 returned schedules</td></tr><tr><td>Maximum budget error after mirror updates</td><td> $2 . 2 2 \times 1 0 ^ { - 1 6 }$ </td><td>90 PB-AMD runs</td></tr><tr><td>Relative objective gap to water-filling</td><td> $7 . 6 7 \times 1 0 ^ { - 1 6 }$ </td><td>One LQ instance</td></tr></table>

Table 11. Stationarity of the returned schedules for $F _ { M }$ , budget preservation, and agreement with the analytical LQ objective.

In the Tabular and full-network runs, the budget errors in Table 11 are at floating-point roundoff level, and every returned PB-AMD schedule meets the stopping tolerance in Section D.2.1.

For the same runs, the seed-averaged private-training tracking errors of Penalty + repair differ from PB-AMD’s by at most 0.102% across the nine task--budget combinations; PGD’s errors differ by less than 0.02%. Relative to Penalty + repair, PB-AMD attains comparable tracking quality without selecting a penalty weight. Section D.3 assesses solver time independently at a common first-order tolerance.

The separate scalar LQ check minimizes $\textstyle F ( q ) = \sum _ { t } a _ { t } / q _ { t }$ with $T = 8 0 , \rho _ { \mathrm { m a x } } = 1$ , and $q _ { \mathrm { m i n } } = 1 0 ^ { - 3 }$ . For $t = 0 , \ldots , 7 9$ , the scalar transition and noise-injection coefficients are

$$
A _ { t } = 0 . 9 1 + 0 . 0 5 5 \sin \left( \frac { 2 \pi ( t + 1 ) } { 8 0 } \right) , \quad B _ { t } = 0 . 0 3 5 + 0 . 0 2 ( t / 7 9 ) ^ { 2 } , \quad a _ { t } = \frac { 1 } { 2 } \left( B _ { t } \prod _ { s = t + 1 } ^ { 7 9 } A _ { s } \right) ^ { 2 } .
$$

The empty product is one. The analytical optimizer is the floored water-filling solution (4.15) of the main article, $q _ { t } ^ { \star } = \operatorname* { m a x } \{ q _ { \operatorname* { m i n } } , \sqrt { a _ { t } / \lambda } \}$ , with λ chosen to satisfy the budget equality. Initialization puts every coordinate at the floor and the remaining charge on one coordinate. PB-AMD takes 200000 updates with $\alpha = 3 \times 1 0 ^ { - 4 }$ , below the relative-smoothness limit $3 . 3 0 5 7 8 5 \times 1 0 ^ { - 4 }$ . Writing $q ^ { \mathrm { f i n a l } }$ for the last PB-AMD iterate, the reported relative objective gap is $( F ( q ^ { \mathrm { f i n a l } } ) - F ( q ^ { \star } ) ) / F ( q ^ { \star } ) = 7 . 6 7 1 \bar { 3 } 8 \bar { 6 } \times 1 0 ^ { - 1 6 }$ . The water-filling optimizer has 47 coordinates at the floor. The agreement at numerical precision provides an implementation check against the analytical optimizer in Proposition 4.5.

D.3. Matched time-to-stationarity protocol. Figure 6 of the main article compares the time taken by PB-AMD, PGD, and Penalty + repair to compute a schedule at the same relative variational-gap tolerance for Tabular, Fashion-restart, and Nonlinear head. All three use the same tracking term $F _ { M }$ on public-design records and the same fixed Gaussian paths; the gap is evaluated on feasible schedules, including the repaired output of penalty optimization. One public-design instance specifies these records, the model initialization, the minibatch sequence, and the fixed Gaussian paths for one problem seed. This study uses different problem seeds from the allocation-quality comparisons in Section D.2. All runs use $\rho _ { \mathrm { m a x } } = 0 . 5$ , Uniform initialization, and the models of Section D.1. The archived PB-AMD/PGD benchmark also includes Fashion-constant. In the order Tabular, Fashion-constant, Fashion-restart, Nonlinear head, the fixed Gaussian path counts are 256, 8, 8, 128.

The three calibration instances per task are used to choose solver parameters. Those parameters are then fixed on the ten independent convergence instances used to measure solver time and gap. Both sets concern schedule optimization on public records, not the held-out records used to measure classification accuracy. For Tabular and both full-network tasks, we run PB-AMD and PGD for 30 updates on each of the three public calibration instances using each stepsize factor $\gamma \in \{ 0 . 0 3 , 0 . 1 , 0 . 3 , 1 \}$ . With $F _ { 0 } = F _ { M } ( q ^ { 0 } )$ , the actual fixed stepsizes are $\alpha _ { \mathrm { P B } } = \gamma \rho _ { \mathrm { m a x } } / F _ { 0 }$ and $\alpha _ { \mathrm { P G D } } = \gamma \rho _ { \mathrm { m a x } } ^ { 2 } / ( T F _ { 0 } )$ . At Uniform, these scalings give the same first-order displacement for a small $\gamma$ . We select each solver’s γ by minimizing the mean logarithm of its relative variational gap $G _ { F _ { M } } ( q ) / F _ { 0 }$ across calibration instances. Here the gap is measured after the 30 calibration updates. Values below $1 0 ^ { - 1 6 }$ are replaced by $1 0 ^ { - 1 6 }$ before taking logarithms; ties favor smaller $\gamma$ The selected PB-AMD/PGD values are (0.3, 0.3) for Tabular, (1, 0.3) for Fashion-restart, and (0.3, 0.3) for Fashion-constant.

For Nonlinear head, both PB-AMD and PGD use Armijo backtracking rather than a calibrated fixed step. Each is run on the same ten convergence instances. The dimensionless step γ starts at 0.3 and is capped at $2 ;$ we convert it to αPB and αPGD using the same formulas as for the fixed-step runs above. The sufficient-decrease coefficient is 0.1; rejected trials multiply the step by 0.5, and accepted steps grow by 1.1, with at most 30 trials per update. Both methods use the sufficient-decrease inequality and rounding allowance in Section D.2.2, with their respective trial updates.

For each of the three tasks in Figure 6, penalty calibration uses the six pairs in Table 10, the same three public calibration instances as PB-AMD and PGD, and 30 raw updates per candidate. As for the other solvers, selection minimizes the mean logarithm of the relative gap, evaluated here at the repaired point, with the same $1 0 ^ { - 1 6 }$ floor; ties favor smaller $\beta .$ The selected penalty-weight/Adam-stepsize pairs are (1000, 0.015) for Tabular, (100, 0.017) for Fashion-restart, and (1000, 0.01) for Nonlinear head. During each convergence run, we repair the iterate with lowest penalized objective found so far at each recorded iteration and compute the repaired point’s variational gap. If the repaired point is unchanged, its previous diagnostic is reused.

The common accuracy criterion is $G _ { F _ { M } } ( q ) / F _ { 0 }$ , defined in Section D.2.1. Two thresholds serve different purposes:

• Time comparison. We record the first time at which the relative gap is at most $1 0 ^ { - 4 }$ , using the recorded diagnostics without interpolation. This is the common target used for the time savings reported in Section 5.5 and Figure 6 of the main article.

• Curve termination. A run continues until the relative gap is at most $1 0 ^ { - 6 }$ or the update limit is reached. The limits are 180, 100, 100, 60 for Tabular, Fashion-constant, Fashion-restart, and Nonlinear head, respectively.

For each instance where both solvers reach $1 0 ^ { - 4 }$ , let $t _ { \mathrm { P B } }$ and tPGD be their first recorded times to this threshold. The reported time saving is $1 0 0 [ 1 - 1 /$ median $( t _ { \mathrm { P G D } } / t _ { \mathrm { P B } } ) ] \%$ , where the median is taken over these paired instances. In contrast, the curves use all ten convergence instances per task, showing the median and interquartile range of the latest relative gap available at each time. Stopped runs retain their final gap at later plotted times, including runs that miss the target.

D.4. Timing and implementation. The timings in Figure 6 use Python 3.12.14, NumPy 2.3.5, and double-precision arrays on a macOS 26.3 ARM64 CPU platform. NumPy implements the forward training recurrence and backward vector--Jacobian products.

We run solvers serially, alternating the PB-AMD/PGD order across instances. Neural objective and adjoint evaluations use four fixed worker threads, with one numerical-library thread per worker; Tabular evaluations are vectorized in one thread. All solvers share the backend within each task, and one adjoint evaluation covers all M paths.

We time updates, required forward and adjoint evaluations, all line-search trials, repair, and final diagnostics. We exclude calibration, data/model construction, warm-up, file output, training on private records, prediction scoring, and diagnostics used only to draw the curves.

Penalty timings use separate runs on the same host and backend. We subtract intermediate adjoint time used only to plot gaps at repaired points, but retain one adjoint evaluation of the returned point unless the solver already computes it. Repair remains included. Raw timings and subtracted costs are archived. The code README provides experiment entry points, run commands, and exact replay settings.

## References

[1] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (2016), pp. 308--318.

[2] Andrew, G., Thakkar, O., McMahan, H. B., and Ramaswamy, S. Differentially private learning with adaptive clipping. In Advances in Neural Information Processing Systems (2021), vol. 34.

[3] Bauschke, H. H., Bolte, J., and Teboulle, M. A descent lemma beyond Lipschitz gradient continuity: First-order methods revisited and applications. Mathematics of Operations Research 42, 2 (2017), 330--348.

[4] Beck, A., and Teboulle, M. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters 31, 3 (2003), 167--175.

[5] Benning, M., Celledoni, E., Ehrhardt, M. J., Owren, B., and Schönlieb, C.-B. Deep learning as optimal control problems: Models and numerical methods. Journal of Computational Dynamics 6, 2 (2019), 171--198.

[6] Bertsekas, D. P., and Shreve, S. E. Stochastic Optimal Control: The Discrete-Time Case. Academic Press, New York, 1978.

[7] Biccari, U., and Song, Y. When to spend privacy: optimal allocation in private learning. https://github.com/DeustoTech/ CoDeFeL-DP-dynamical-open-loop, 2026.

[8] Bu, Z., Wang, Y.-X., Zha, S., and Karypis, G. Automatic clipping: Differentially private deep learning made easier and stronger. In Advances in Neural Information Processing Systems (2023), vol. 36, pp. 41727--41764.

[9] Bun, M., and Steinke, T. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference (2016), Springer, pp. 635--658.

[10] Du, J., Li, S., Chen, X., Chen, S., and Hong, M. Dynamic differential-privacy preserving SGD. arXiv preprint arXiv:2111.00173 (2021).

[11] Dwork, C., and Roth, A. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science 9, 3--4 (2014), 211--407.

[12] Dwork, C., and Rothblum, G. N. Concentrated differential privacy. arXiv preprint arXiv:1603.01887 (2016).

[13] E, W., Han, J., and Li, Q. A mean-field optimal control formulation of deep learning. Research in the Mathematical Sciences 6, 1 (2019), 10.

[14] Fatkhullin, I., and He, N. Taming nonconvex stochastic mirror descent with general Bregman divergence. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (2024), vol. 238 of Proceedings of Machine Learning Research, PMLR, pp. 3493--3501.

[15] Ghadimi, S., and Lan, G. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization 23, 4 (2013), 2341--2368.

[16] Gopi, S., Lee, Y. T., and Wutschitz, L. Numerical composition of differential privacy. In Advances in Neural Information Processing Systems (2021), vol. 34, pp. 11631--11642.

[17] Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 2 (1979), 65--70.

[18] Hong, J., Wang, H., Wang, Z., and Zhou, J. Learning model-based privacy protection under budget constraints. In Proceedings of the AAAI Conference on Artificial Intelligence (2021), vol. 35, pp. 7702--7710.

[19] Hong, J., Wang, Z., and Zhou, J. Dynamic privacy budget allocation improves data efficiency of differentially private gradient descent. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (2022), pp. 11--35.

[20] Khadangi, A., Sartipi, A., Tchappi, I., Bahmani, R., and Fridgen, G. Efficient differentially private fine-tuning of LLMs via reinforcement learning. arXiv preprint arXiv:2507.22565 (2025).

[21] Kianidehkordi, S., Kulkarni, N., Dziedzic, A., Draper, S., and Boenisch, F. Differentially private federated learning with time-adaptive privacy spending. In International Conference on Learning Representations (2025).

[22] Kleywegt, A. J., Shapiro, A., and Homem-de Mello, T. The sample average approximation method for stochastic discrete optimization. SIAM Journal on Optimization 12, 2 (2002), 479--502.

[23] Koskela, A., and Honkela, A. Learning rate adaptation for differentially private learning. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics (2020), vol. 108 of Proceedings of Machine Learning Research, PMLR, pp. 2465--2475.

[24] Lee, J., and Kifer, D. Concentrated differentially private gradient descent with adaptive per-iteration privacy budget. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2018), ACM, pp. 1656--1665.

[25] Li, P., Liang, M., Jiang, Z., and Zhang, J. Differentially private deep learning with importance-based adaptive gradient processing. In Proceedings of the 16th Asian Conference on Machine Learning (2025), vol. 260 of Proceedings of Machine Learning Research, PMLR, pp. 159--174.

[26] Li, Q., Chen, L., Tai, C., and E, W. Maximum principle based algorithms for deep learning. Journal of Machine Learning Research 18, 165 (2018), 1--29.

[27] Lu, H., Freund, R. M., and Nesterov, Y. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization 28, 1 (2018), 333--354.

[28] Mironov, I. Rényi differential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF) (2017), IEEE, pp. 263--275.

[29] Nemirovski, A., Juditsky, A., Lan, G., and Shapiro, A. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization 19, 4 (2009), 1574--1609.

[30] Nemirovski, A. S., and Yudin, D. B. Problem Complexity and Method Efficiency in Optimization. Wiley, 1983.

[31] Nocedal, J., and Wright, S. J. Numerical Optimization, 2nd ed. Springer, New York, 2006.

[32] Rockafellar, R. T., and Wets, R. J.-B. Variational Analysis, vol. 317 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, 1998.

[33] Shapiro, A., Dentcheva, D., and Ruszczyński, A. Lectures on Stochastic Programming: Modeling and Theory, 2nd ed., vol. 16 of MOS-SIAM Series on Optimization. Society for Industrial and Applied Mathematics, Philadelphia, PA, 2014.

[34] Van Dijk M. Nguyen N. V. Nguyen T. N. Nguyen L. M. and Nguyen P. H. Proactive DP: A multiple targe optimization framework for DP-SGD. In Proceedings of the 41st International Conference on Machine Learning (2024), vol. 235 of Proceedings of Machine Learning Research, PMLR, pp. 49029--49077.

[35] Wang, Y.-X., Balle, B., and Kasiviswanathan, S. P. Subsampled Rényi differential privacy and analytical moments accountant. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics (2019), vol. 89 of Proceedings of Machine Learning Research, PMLR, pp. 1226--1235.

[36] Xiao, H., Rasul, K., and Vollgraf, R. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747 (2017).

[37] Yuan, X., Savkin, A. V., Ni, W., Xue, M., and Liu, R. P. Optimal online control strategy for differentially private federated learning. IEEE Transactions on Dependable and Secure Computing 23, 2 (2026), 4164--4176.

∗ Chair of Computational Mathematics, DeustoTech, University of Deusto, Avenida de las Universidades 24, 48007 Bilbao<sub>,</sub> Bas<sub>q</sub>ue Country<sub>,</sub> Spain.

† Division of Mathematical Sciences, School of Physical and Mathematical Sciences, Nanyang Technological University<sub>,</sub> 21 Nanyang Link 637371<sub>,</sub> Singapore.

Email address: yongcun.song@ntu.edu.sg