# Uncovering and Fixing Collider Bias in Bayesian PINNs

Michael Obermayr TU Graz

## Abstract

Bayesian physics-informed neural networks (B-PINNs) are a popular framework for parameter and state inference from sparse or noisy observations. They are commonly for mulated via a collider structure, in which physical and trajectory parameters are assumed to be a priori independent and be come coupled through virtual likelihoods on diferential-equation residuals that enforce physical consistency. We show that this modeling choice can induce severe systematic bias in the posterior over physical parameters: even when the prior is favorably centered on the ground-truth parameters, the resulting posterior can drift away and concentrate far from them. As a remedy, we advocate a hierarchical chain model in which physics generates trajectories, which in turn generate observations. The chain model does not sufer from this posterior bias, but it poses a harder, so-called doubly intractable, inference problem due to a physics-dependent normalization constant. This challenge can be resolved by discretizing the underlying stochastic dynamics, after which the chain posterior can be sampled exactly with particle MCMC. We identify two distinct mechanisms characteriz ing the collider bias, derive analytical approximations of their magnitudes, and establish diagnostic criteria for predicting when standard B-PINNs remain reliable. Experiments confirm the predicted bias and show that the chain formulation successfully avoids it.

## 1 INTRODUCTION

Inferring unknown physical parameters from dynamical systems is a fundamental challenge across science and engineering. Physics-informed neural networks (PINNs) [22] tackle this inverse problem by training neural trajectory surrogates to jointly minimize data

Robert Peharz TU Graz

![](images/d6262499779f21d2b8c87db5cbc679b15e36502ae860a451558dc248229ade00.jpg)  
(a) Collider (B-PINN)

![](images/3d62532f639c66ac446328c0b027c42b73e47f6b2b9000eb2179c8a676f69b42.jpg)  
(b) Chain (ours)  
Figure 1: Modeling choices for physics-informed Bayesian inference. (a) Collider (B-PINN): Physics parameters λ and trajectory parameters θ are a priori independent, interacting only when conditioned on equation residuals r. (b) Chain (ours): Physics λ directly generates trajectories θ, which in turn generate data y.

mismatch and physical law violations. Under limited and noisy data, a Bayesian treatment naturally quanti fies uncertainty over trajectories and parameters. Yet, their assumed dependency structure requires careful consideration, which is the focus of this paper.

Bayesian PINNs (B-PINNs) [29] assume the factorization

$$
\begin{array} { r } { p _ { \mathrm { c o l l i d e r } } ( \lambda , \theta , r , y ) = \underbrace { p ( \lambda ) } _ { \mathrm { P h y s i c s ~ T r a j . } } \underbrace { p ( \theta ) } _ { \mathrm { R e s i d u a l } } \underbrace { p ( r \mid \lambda , \theta ) } _ { \mathrm { O b s e r v . } } \underbrace { p ( y \mid \theta ) } _ { \mathrm { O b s e r v . } } . } \end{array}\tag{1}
$$

Here $p ( \lambda )$ and $p ( \theta )$ are the priors over physics and trajectory parameters, respectively, and $p ( y \mid \theta )$ is the observation likelihood. Residuals r serve as “virtual observations” of zero measuring physics violations at collocation points, typically under a narrow zero-mean Gaussian likelihood $p ( r | \lambda , \theta )$ Thus, λ and θ are a priori independent and coupled only upon conditioning on $r = 0$ . In graphical-model terms [13], this forms a collider at $r \ ( \mathrm { F i g . \ 1 a } )$ , a structure widely adopted in the literature [14, 20, 25, 26].

Alternatively, one might use a hierarchical model, in which physics λ is equipped with a prior, trajectories θ depend on λ and observations y depend on θ:

$$
\begin{array} { r } { p _ { \mathrm { c h a i n } } ( \lambda , \theta , y ) \propto \underbrace { p ( \lambda ) } _ { \mathrm { P h y s i c s } } \underbrace { p ( \theta \mid \lambda ) } _ { \mathrm { T r a j e c t o r y ~ O b s e r v a t i o n } } . } \end{array}\tag{2}
$$

![](images/bb47176f23a2ac1b694facecbaac8230a47887cf90e3e90bfd83bd4775fea696.jpg)  
Figure 2: Posterior marginals for the damped linear oscillator. System governed by $\begin{array} { r } { \ddot { x } = - \frac { c } { m } \dot { x } - \frac { k } { m } x + } \end{array}$ $\sigma _ { r } \xi ( t )$ with white noise $\xi ( t )$ , mass $m ,$ damping c and spring constant k, ground-truth parameters $\{ m , c , k \} =$ $\{ 1 . 0 , 1 . 2 , 4 . 0 \}$ , residual scale fixed at $\sigma _ { r } = 0 . 3$ , and 40 noisy observations $( \sigma _ { \mathrm { o b s } } = 0 . 0 2 )$ . Posteriors are sampled via NUTS (24,000 samples) for B-PINN and PMMH [2] (14,000 samples) for our proposed chain model. Although all three physical parameters are inferred jointly, only the ratios $c / m$ and $k / m$ are identifiable; we therefore display their corresponding marginal posteriors. Despite initialization at the ground truth and identical priors centered at the ground truth, only the chain model accurately recovers the true values. In contrast, B-PINN exhibits a systematic collider bias, underestimating dissipation $( c / m )$ and stifness $( k / m )$ . The bias is reproducible over resampled datasets. Full experimental details are provided in Section 5.

Here the dependency structure between λ and θ forms a chain (head-to-tail node) [13], as illustrated in Fig. 1b. Unlike the collider (1), θ depends directly on λ through the trajectory likelihood $p ( \theta | \lambda )$ rather than indirectly through residual likelihood $p ( r | \lambda , \theta )$

Which of the two formulations is preferable? The collider ofers practical convenience: its factors are straightforward Gaussian priors and likelihoods. By contrast, specifying $p ( \theta | \lambda )$ in the chain is less straightforward; it requires describing a distribution over trajectory parameters θ that may vary substantially with the physics λ. An intuitive construction is to start from the prior $p ( \theta )$ , similar as in the collider formulation, reweight it according to the exponential of the negative squared residual error, and renormalize the result. As detailed in Section 2, this renders the chain model (2) nearly identical to the collider model (1), up to a physics-dependent normalization constant $Z _ { \lambda }$ whose calculation requires an intractable marginalization over θ.

Despite the computational simplicity of the collider, the chain should be preferred. Conceptually, the residuals r in the collider formulation act merely as auxiliary variables to couple physics λ and trajectories θ, rather than quantities of substantive in terest. The chain avoids this artificial construction by modeling trajectory dependence directly via $p ( \theta | \lambda )$ Furthermore, forward-sampling the collider yields an implausible generative process—typically generating incompatible pairs of λ and θ that demand unlikely residuals. By contrast, the chain follows the natural hierarchy in which physics λ generates trajectories $\theta ,$ which in turn generate observations y.

Crucially, the collider systematically biases the posterior over physical parameters. To demonstrate this, we sample synthetic data from a damped linear oscillator with known ground truth, center prior modes at the true values, and sample the collider posterior with NUTS [12], using Fourier features to prevent spectral bias [27]. Despite these favorable conditions, the collider marginals drift substantially from the ground truth $\left( \mathrm { F i g . ~ 2 } \right)$ , by about two posterior standard deviations in $c / m .$ This is alarming, as the collider model can return confidently wrong estimates of the underlying physics. By contrast, our proposed chain inference exhibits the expected behavior: its posterior concentrates at the ground truth.

As shown in Section 2, this bias stems directly from the physics-dependent normalization constant $Z _ { \lambda }$ , which systematically favors less restrictive physics in the col lider setting. While the chain formulation eliminates this bias, it yields a challenging, doubly-intractable problem [17]. Despite the prevalence of collider-based approaches in physics-informed neural inference, this structural bias remains largely unaddressed. To our knowledge, the only exceptions are Alberts, Hao, and Bilionis [1, 10, 11], who employ a similar chain structure and trace the posterior bias to $Z _ { \lambda ; \lambda }$ however, they address the doubly-intractable problem via computationally demanding nested inference without further analyzing the bias. Still, their formulation has gained little traction in the community and the collider formulation remains the default in B-PINN practice [16, 24].

## 2 PROBLEM FORMULATION AND PROBABILISTIC MODELING

We consider parameterized second-order ordinary dif ferential equations (ODEs) governing a trajectory $\ b { x } ( t ) \in \mathbb { R } ^ { d }$

$$
M _ { \lambda } ( x , \dot { x } ) \ddot { x } = F _ { \lambda } ( x , \dot { x } , t ) ,\tag{3}
$$

where ˙x and ¨x denote the first and second time derivatives, $M _ { \lambda } ( x , \dot { x } ) ~ \in ~ \mathbb { R } ^ { d \times d }$ is an invertible generalized mass matrix, and $F _ { \lambda } ( x , \dot { x } , t ) \in \mathbb R ^ { d }$ is a force field driving the system, both parameterized by unknown physical parameters $\lambda \in \mathbb { R } ^ { p }$ . Equation (3) is the general form of a broad class of second-order dynamical systems arising in classical mechanics, structural dynamics, and robotics [8, 18].

In practice, the trajectory $x ( t )$ is rarely observed directly and is therefore treated as latent. Instead, we only have access to sparse, noisy observations $y =$ $\{ y _ { k } \} _ { k = 1 } ^ { K }$ at times $t _ { k } .$ , governed by a Gaussian measurement model

$$
y _ { k } = x ( t _ { k } ) + \epsilon _ { k } , \quad \epsilon _ { k } \sim \mathcal { N } ( 0 , \Sigma _ { \mathrm { o b s } } ) ,\tag{4}
$$

with known observation covariance $\Sigma _ { \mathrm { o b s } }$

Beyond the measurement, the dynamics themselves are typically uncertain: simplifications and unmodeled degrees of freedom inevitably perturb the ideal dynamics (3). We account for these discrepancies by introducing process noise on the velocity $v : = { \dot { x } }$ . With this, the phase-space state $z : = ( x ^ { \top } , v ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { 2 d }$ evolves according to an Itˆo stochastic diferential equation (SDE)

$$
\mathrm { d } z _ { t } = \underbrace { \left[ \vphantom { \left[ \sum _ { i } | z _ { t } , t \right] } v _ { t } \right. } _ { = : f _ { \lambda } ( z _ { t } , t ) } \mathrm { d } t + \left[ \begin{array} { c } { 0 } \\ { A _ { \lambda } ( z _ { t } ) ^ { - 1 } \Sigma _ { r } ^ { 1 / 2 } } \end{array} \right] \mathrm { d } W _ { t } ,\tag{5}
$$

driven by standard Brownian motion $W _ { t } \in \mathbb { R } ^ { d }$ . Here, $a _ { \lambda } ( z , t ) : = M _ { \lambda } ( x , v ) ^ { - 1 } F _ { \lambda } ( x , v , t )$ is the acceleration field and $f _ { \lambda }$ the drift. The physical disturbances are modeled as constant white noise with covariance $\Sigma _ { r } ,$ and the invertible, possibly state- and parameterdependent matrix $\boldsymbol { A } _ { \lambda } ( \boldsymbol { z } ) \in \mathbb { R } ^ { \bar { d } \times d }$ maps them into velocity increments.

The choice of $A _ { \lambda }$ determines how process noise enters the dynamics: setting $A _ { \lambda } = I _ { d }$ injects white noise directly into the accelerations, whereas setting $A _ { \lambda } = M _ { \lambda }$ models the perturbations as stochastic forces. The latter is more realistic for many mechanical systems, where physical disturbances act as fluctuating generalized forces or torques. As we show in Section 3, parameter dependence in $A _ { \lambda }$ is one of the factors contributing to collider bias in the inferred physical parameters.

With process noise present $\left( \begin{array} { l l l } { \sum _ { r } } & { \succ } & { 0 } \end{array} \right)$ , particular physics parameters λ induce a distribution p(x | λ) over trajectories rather than a single deterministic trajectory. Our primary goal is to infer the marginal posterior distribution over the physical parameters given sparse observations of the trajectory:

$$
p ( \lambda | y ) \propto p ( \lambda ) \int p ( y | x ) p ( x | \lambda ) \mathrm { d } x .\tag{6}
$$

Following standard B-PINN practice, we fix the residual scale $\Sigma _ { r }$ as a hyperparameter rather than inferring it alongside λ [14, 29].

## 2.1 Physics-Informed Residuals and Energy

To connect the trajectory distribution induced by the stochastic dynamics (5) with physics-informed inference, we express deviations from the governing dynamics through a residual energy. In physics-informed machine learning, the continuous trajectory is represented by a diferentiable surrogate $x _ { \theta } ( t )$ parameterized by θ (e.g., neural network weights). At a set of collocation points $t _ { 1 } < \cdots < t _ { N }$ , we define the physics residual

$$
r _ { n } ( \theta , \lambda ) : = A _ { \lambda } ( z _ { \theta } ( t _ { n } ) ) \left( \ddot { x } _ { \theta } ( t _ { n } ) - a _ { \lambda } ( z _ { \theta } ( t _ { n } ) , t _ { n } ) \right) ,\tag{7}
$$

where $z _ { \theta } : = ( x _ { \theta } ^ { \top } , \dot { x } _ { \theta } ^ { \top } ) ^ { \top }$ To relate these pointwise residuals to the process-noise model, suppose the collocation points are uniformly spaced with step size h. Over an interval of width $h ,$ the Brownian increment driving (5) has covariance $h \Sigma _ { r }$ . Equivalently, the corresponding average stochastic forcing over that interval has covariance $\Sigma _ { r } / h$ . This motivates modeling the residuals at the collocation points as Gaussian deviations with covariance $\Sigma _ { r } / h$ , yielding the quadratic residual energy

$$
E [ \theta , \lambda ] = { \frac { h } { 2 } } \sum _ { n = 1 } ^ { N } r _ { n } ( \theta , \lambda ) ^ { \top } \Sigma _ { r } ^ { - 1 } r _ { n } ( \theta , \lambda ) .\tag{8}
$$

Thus, the stochastic dynamics induce a natural quadratic energy measuring the compatibility between a trajectory x<sub>θ</sub> and physical parameters λ. This residual energy is the common ingredient underlying the probabilistic formulations considered next.

## 2.2 Probabilistic Models: Collider vs. Chain

We can now define the two probabilistic models— collider (1) and chain (2)—in a parallel manner and using similar components. Both factorizations share the same prior $p ( \lambda )$ over physical parameters and the same observational likelihood,

$$
p ( \boldsymbol { y } \vert \theta ) = \prod _ { k = 1 } ^ { K } \mathcal { N } \left( \boldsymbol { y } _ { k } \vert \boldsymbol { x } _ { \theta } ( t _ { k } ) , \boldsymbol { \Sigma } _ { \mathrm { o b s } } \right) .\tag{9}
$$

The key diference between them is how the residual energy (8) is used to couple the trajectory parameters θ and the physical parameters λ.

Collider formulation. Standard Bayesian PINNs [14, 25, 26, 29] place an independent prior $p ( \theta )$ on the trajectory parameters and enforce the physical law through auxiliary residuals treated as pseudoobservations equipped with a narrow zero-mean Gaussian:

$$
\begin{array} { c } { { \displaystyle p ( r = { \bf 0 } | \lambda , \theta ) = \prod _ { n = 1 } ^ { N } { \cal N } \left( { \bf 0 } | r _ { n } ( \theta , \lambda ) , \frac { 1 } { h } \Sigma _ { r } \right) } } \\ { { \displaystyle ~ = \frac { 1 } { Z _ { r } } \exp \left( - E [ \theta , \lambda ] \right) , } } \end{array}\tag{10}
$$

where the Gaussian normalization constant $\begin{array} { r l } { Z _ { r } } & { { } = } \end{array}$ $\operatorname* { d e t } ( 2 \pi \Sigma _ { r } / h ) ^ { N / 2 }$ depends on $\Sigma _ { r }$ , but is invariant to λ and θ. Because λ and θ are assumed a priori independent and interact only via conditioning on $r = 0$ the residual acts as a collider node in the underlying directed acyclic graph (Fig. 1a).

Chain formulation. In contrast, the hierarchical chain formulation follows the generative direction $\lambda $ $\theta  y ~ ( \mathrm { F i g . ~ 1 b } )$ by defining a conditional distribution over trajectories,

$$
p ( \theta | \lambda ) = \frac { 1 } { Z _ { \lambda } } \exp \left( - E [ \theta , \lambda ] \right) p ( \theta ) .\tag{11}
$$

Here, $p ( \theta )$ is interpreted as a reference density over the trajectory parametrization, which can encode preferences for certain trajectories. Unlike in the collider formulation, it is not a prior over $\theta ;$ instead, the exponential factor tilts this reference density toward trajectories that are compatible with the $\lambda .$ Normalizing the resulting conditional distribution requires the normalization constant

$$
Z _ { \lambda } = \int \exp \left( - E [ \theta , \lambda ] \right) p ( \theta ) \mathrm { d } \theta ,\tag{12}
$$

which generally depends on the physical parameters $\lambda .$

Model Diference and The Collider Bias. The two formulations difer only through their normalization factors and we can write:

$$
p _ { \mathrm { c o l l i d e r } } ( \lambda , \theta , r = \mathbf { 0 } , y ) = \frac { Z _ { \lambda } } { Z _ { r } } p _ { \mathrm { c h a i n } } ( \lambda , \theta , y ) .\tag{13}
$$

Since $Z _ { r }$ is constant with respect to λ, this diference carries directly to the posterior over the physical pa rameters:

$$
\begin{array} { r } { \Delta ( \lambda ) : = \log \frac { p _ { \mathrm { c o l l i d e r } } ( \lambda \mid y , r = \mathbf { 0 } ) } { p _ { \mathrm { c h a i n } } ( \lambda \mid y ) } = \log Z _ { \lambda } + C , } \end{array}\tag{14}
$$

where we call the model diference $\Delta ( \lambda )$ the collider bias. It is the mechanism underlying the posterior drift observed in Fig. 2. The term bias is deliberate: $\Delta$ displaces the posterior from the data-generating parameters, so that the collider’s credible intervals miss the ground truth, not merely the chain posterior (Section 5). The partition function $Z _ { \lambda }$ measures the amount of trajectory space, weighted by the reference density $p ( \theta )$ , that is compatible with the dynamics under λ. The collider formulation therefore reweights the desired posterior by $Z _ { \lambda }$ , favoring λ that admit a larger volume of low-residual trajectories, that is, less restrictive physics.

## 3 ANALYSIS OF THE COLLIDER BIAS

For B-PINNs with a neural trajectory surrogate, the collider bias $\Delta ( \lambda )$ decomposes into two distinct mechanisms. In Section 4.2, we show that adopting a Markovian state-space latent model eliminates one mechanism and renders the other analytically tractable.

Laplace approximation of the collider bias. Because the partition function (12) is analytically intractable for neural surrogates, we place a Gaussian prior $p ( \theta ) = \mathcal { N } ( 0 , P ^ { - 1 } )$ on the weights and apply a Laplace expansion around the data-free mode $\theta ^ { * } ( \lambda )$ . The collider bias (14) evaluates to

$$
\Delta ( \lambda ) \approx - \frac { 1 } { 2 } \log \operatorname * { d e t } \left( G _ { \lambda } + P \right) + C ,\tag{15}
$$

where $G _ { \lambda }$ is the Gauss–Newton curvature of the resid ual energy (8) at $\theta ^ { * }$ . The residual constrains $n _ { \mathrm { e f f } } : =$ $\mathrm { t r } \big [ G _ { \lambda } ( G _ { \lambda } + \dot { P } ) ^ { - 1 } \big ] \ \leq$ rank $G _ { \lambda } \ \leq \ \operatorname* { m i n } ( N d , \dim \theta )$ directions. Evaluating this curvature along the nominal trajectory $z ( t ) : = z _ { \theta ^ { \ast } } ( t )$ over $t \in [ 0 , \mathcal { T } ]$ decomposes (15) into:

$$
\begin{array} { r } { \Delta ( \lambda ) \approx \underbrace { - \frac { n _ { \mathrm { e f f } } } { N d } \displaystyle \sum _ { n = 1 } ^ { N } \log \left| \operatorname* { d e t } \mathcal { A } _ { \lambda } ( z ( t _ { n } ) ) \right| } _ { \mathrm { ( 1 ) ~ R e s i d u a l ~ m e t r i c ~ b i a s } } } \\ { \underbrace { - \frac { 1 } { 2 } \displaystyle \int _ { 0 } ^ { T } \sum _ { k } \left| \mathrm { R e } s _ { k } ( t ) \right| \mathrm { d } t } _ { \mathrm { ( 2 ) ~ D i s s i p a t i o n ~ b i a s } } + C , } \end{array}\tag{16}
$$

where $\{ s _ { k } ( t ) \}$ are the eigenvalues of the drift Jacobian $D f _ { \lambda }$ along $z ( t )$ (full derivation in Appendix A.1). Eq. (16) explains the origin and direction of the bias.

Derivation Assumptions. This decomposition relies on four simplifying assumptions (Appendix A.1): (i) the surrogate closely satisfies the ODE at $\theta ^ { * }$ , with the prior penalty depending only weakly on $\lambda ; ( \mathrm { i i } ) ~ G _ { \lambda }$ is evaluated along a single nominal trajectory; (iii) the residual loss dominates the prior along constrained directions $( n _ { \mathrm { e f f } }$ ≈ rank $G _ { \lambda }$ locally constant); and $( \mathrm { i v } )$ the residual metric and drift decouple additively. We also treat $A _ { \lambda }$ and $D f _ { \lambda }$ as piecewise constant in time, and assume the surrogate bandwidth Ω exceeds the system’s characteristic dynamical rates.

Mechanism 1: Residual metric bias. Because $Z _ { r }$ normalizes over residual space, it omits the Jacobian determinant |det $A _ { \lambda } |$ of the acceleration-to-residual mapping, whereas $Z _ { \lambda }$ accounts for it along all $n _ { \mathrm { e f f } }$ constrained directions. Consequently, acceleration residuals $\left( A _ { \lambda } \ : = \ : I _ { d } \right)$ avoid this bias, whereas force residuals $\left( A _ { \lambda } \ : = \ : M _ { \lambda } \right)$ add $- { \frac { n _ { \mathrm { e f f } } } { N d } } \sum _ { n = 1 } ^ { N }$ log|det $\displaystyle M _ { \lambda } ( z ( t _ { n } ) ) \vert$ shifting posterior mass toward vanishing mass determinants. The residual metric is dictated by where the noise enters the dynamics (Section 2.1); switching to acceleration residuals for convenience changes the noise model and is therefore not a free choice.

This pathology worsens along drift-invariant parameter directions where jointly scaling masses and forces preserves drift. Because the bias grows with $n _ { \mathrm { e f f } }$ while the data only constrain scale via finite observations, the artifact can overwhelm the likelihood and collapse the posterior toward vanishing masses (Appendix A.2).

Mechanism 2: Dissipation bias. Lacking Markovian temporal structure, the neural surrogate couples perturbations through the operator ${ \frac { \mathrm { d } } { \mathrm { d } t } } - D f _ { \lambda }$ nonlocally over $[ 0 , \tau ]$ , generating the second term in (16).

When all perturbations decay $( \mathrm { R e } s _ { k } \le 0 )$ , the spectral sum reduces to phase-space divergence: $\begin{array} { r } { \sum _ { k } | \mathrm { R e } s _ { k } | = } \end{array}$ $- \nabla \cdot f _ { \lambda } = - \operatorname { t r } ( \partial a _ { \lambda } / \partial v )$ . The dissipation bias simplifies to

$$
\Delta _ { \mathrm { d i s s } } ( \lambda ) = \frac { 1 } { 2 } \int _ { 0 } ^ { \mathcal T } \nabla \cdot f _ { \lambda } (  { \boldsymbol { z } } ( t ) , t ) \mathrm { d } t + \mathcal { O } ( \mathcal { T } / \Omega ) + C ,\tag{17}
$$

whose leading term matches the Onsager–Machlup path-action correction [19]. In dissipative systems $( \nabla \cdot f _ { \lambda } < 0 )$ , e<sup>∆</sup>diss spuriously penalizes dissipation, underestimating friction, damping, or drag. Crucially, this penalty scales as $\Theta ( \tau )$ , so that the B-PINN’s error grows with the record length relative to its own posterior width (Section 5.2). In the presence of growing modes, it analogously penalizes stronger instability (Appendix A.1).

## 4 TRACTABLE INFERENCE IN THE CHAIN MODEL

With a global surrogate, the chain model yields a doubly intractable posterior [17], usually requiring expensive inner-loop re-estimation of $Z _ { \lambda }$ via variational inference or Langevin dynamics that often diverges $( \mathrm { A p - }$ pendix A.4). Instead of attempting global integration, we discretize the generative SDE [23]. This factorizes $Z _ { \lambda }$ into analytic per-step normalizers (21), enabling asymptotically exact inference via particle MCMC.

## 4.1 The Chain Model as a State-Space Model

Discretizing $[ 0 , \tau ]$ into T intervals of width h defines states $z _ { i } = ( \bar { x } _ { i } ^ { \top } , v _ { i } ^ { \top } ) ^ { \top }$ at $t _ { i } = i h$ and applying a semiimplicit Euler–Maruyama scheme to (5) yields:

$$
\begin{array} { r l } & { v _ { i } = v _ { i - 1 } + h a _ { \lambda } ( z _ { i - 1 } , t _ { i - 1 } ) + \sqrt { h } A _ { \lambda } ( z _ { i - 1 } ) ^ { - 1 } \Sigma _ { r } ^ { 1 / 2 } \xi _ { i } , } \\ & { x _ { i } = x _ { i - 1 } + h v _ { i } , } \end{array}\tag{18}
$$

with independent increments $\xi _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ . This discretization induces a Markovian trajectory prior over $\theta = z _ { 0 : T } ;$

$$
\begin{array} { c } { { p ( z _ { 0 : T } \mid \lambda ) = p ( z _ { 0 } ) \displaystyle \prod _ { i = 1 } ^ { T } K _ { \lambda } ( z _ { i } \mid z _ { i - 1 } ) , } } \\ { { K _ { \lambda } ( z _ { i } \mid z _ { i - 1 } ) = \delta ( x _ { i } - x _ { i - 1 } - h v _ { i } ) } } \\ { { \cdot \mathcal { N } \big ( v _ { i } \mid \bar { v } _ { i } , h A _ { \lambda } ^ { - 1 } \Sigma _ { r } A _ { \lambda } ^ { - \top } \big ) , } } \end{array}\tag{19}
$$

where $\bar { v } _ { i } : = v _ { i - 1 } + h a _ { \lambda } ( z _ { i - 1 } , t _ { i - 1 } )$ and $A _ { \lambda }$ is evaluated at $z _ { i - 1 }$ . Unlike (11), this prior admits analytic, per-step normalization. Combined with the prior $p ( \lambda )$ and observations at $\{ t _ { k } \} \subseteq \{ t _ { i } \} _ { i = 1 } ^ { T }$ , this completes the chain model (2) over $\left( \lambda , z _ { 0 : T } \right)$ to target the exact marginal posterior $p ( \lambda | y )$

## 4.2 Exact Collider Bias on the State-Space Latent

On the Markovian latent state $\theta = z _ { 0 : T }$ , the collider bias can be evaluated in closed form. The velocity update in (18) corresponds to the discrete residual

$$
r _ { i } = A _ { \lambda } ( z _ { i - 1 } ) \left( \frac { v _ { i } - v _ { i - 1 } } { h } - a _ { \lambda } ( z _ { i - 1 } , t _ { i - 1 } ) \right) ,\tag{20}
$$

making the transition kernel (19) proportional to $\begin{array} { r } { \exp ( - \frac { \breve { h } } { 2 } r _ { i } ^ { \top } \Sigma _ { r } ^ { - 1 } r _ { i } ) } \end{array}$ , which matches the summand of (8) for $N \stackrel { - } { = } T$ . Because $r _ { i }$ is afine in $v _ { i } .$ , each one-step transition normalizes analytically:

$$
\begin{array} { r l r } {  { Z _ { \lambda } ( z _ { i - 1 } ) : = \int \exp ( - \frac { h } { 2 } r _ { i } ^ { \top } \Sigma _ { r } ^ { - 1 } r _ { i } ) \mathrm { d } v _ { i } } } \\ & { } & { = \frac { ( 2 \pi h ) ^ { d / 2 } \operatorname* { d e t } ( \Sigma _ { r } ) ^ { 1 / 2 } } { | \operatorname* { d e t } A _ { \lambda } ( z _ { i - 1 } ) | } . } \end{array}\tag{21}
$$

The drift $a _ { \lambda }$ acts merely as a mean shift and drops out of the integral, causing the dissipation bias to vanish.

Substituting $\begin{array} { r } { \prod _ { i = 1 } ^ { T } Z _ { \lambda } ( z _ { i - 1 } ) } \end{array}$ for $Z _ { \lambda }$ in (13) and subtracting log $\begin{array} { r } { Z _ { r } \ = \ \frac { T } { 2 } } \end{array}$ log det $\left( 2 \pi \Sigma _ { r } / h \right)$ gives the exact collider bias:

$$
\Delta _ { \mathrm { M a r k o v } } ( \lambda , z _ { 0 : T - 1 } ) = - \sum _ { i = 1 } ^ { T } \mathrm { l o g } | \mathrm { d e t } A _ { \lambda } ( z _ { i - 1 } ) | + C .\tag{22}
$$

Only the residual metric bias remains: under acceleration residuals $( A _ { \lambda } = I _ { d } ) , \Delta _ { \mathrm { M a r k o v } }$ is independent of λ and both posteriors coincide; under force residuals $( A _ { \lambda } = M _ { \lambda } ) , \Delta _ { \mathrm { M a r k o v } }$ biases the collider posterior toward vanishing mass determinants.

## 4.3 Exact Parameter Inference via Particle MCMC

For nonlinear dynamics, the marginal likelihood $\begin{array} { r } { p ( y \mid \lambda ) ~ = ~ \int p ( z _ { 0 : T } \mid \lambda ) p ( y \mid z _ { 0 : T } ) \mathrm { d } z _ { 0 : T } } \end{array}$ lacks an analytic form. Rather than approximating it variationally [3], we exploit its Markovian structure, which admits sequential Monte Carlo estimation via a boot strap particle filter [9]. We propagate $P$ particles forward under (18), reweight by $\mathcal { N } ( y _ { k } \mid x _ { t _ { k } } , \Sigma _ { \mathrm { o b s } } )$ at observation times $t _ { k } .$ , and adaptively resample [15]. The resulting estimator ${ \hat { p } } ( y \mid \lambda )$ costs $\mathcal { O } ( P T )$ and is strictly unbiased. By the pseudo-marginal property [2], substituting ${ \hat { p } } ( y \mid \lambda )$ into the Metropolis–Hastings acceptance ratio guarantees Particle Marginal Metropolis– Hastings (PMMH) targets the exact posterior $p ( \lambda | y )$ for any $P \geq 1$ , where P governs only chain mixing. Sampling a trajectory alongside each accepted $\lambda ^ { \prime }$ additionally yields exact joint samples from $p ( \lambda , z _ { 0 : T } \mid y )$

## 5 EXPERIMENTS

We test the analysis of Section 3 on three systems that isolate its bias mechanisms, and compare our inference scheme with nested inference.

## 5.1 Systems and Setup

Data are generated from the SDE (5) with isotropic process noise $\Sigma _ { r } = \sigma _ { r } ^ { 2 } I _ { d }$ fixed at ground truth; physics priors are log-normal with medians at the truth. Jointly scaling masses and forces by α leaves the drift invariant; this drift-invariant direction is only weakly identifiable via process noise.

Damped oscillator $\begin{array} { r } { ( d { \bf \Phi } = { \bf \Phi } 1 ) \colon } \end{array}$ mass $m ,$ stifness $k ,$ linear damping $c ,$ drift $m { \ddot { x } } \ = \ - c { \dot { x } } \ - \ k x$ , truth $( m , c , k ) \ = \ ( 1 , 1 . 2 , 4 )$ , and $\sigma _ { r } ~ = ~ 0 . 3$ . Because only $c / m$ and $k / m$ enter the drift, the invariant direction is $( m , c , k ) \to \alpha ( m , c , k )$ . We observe position $K = 4 0$ times over $\mathcal { T } = 2 4 \ : \left( \sigma _ { \mathrm { o b s } } = 0 . 0 2 \right)$ . Dissipation bias is active $( \nabla \cdot f _ { \lambda } = - c / m < 0 )$

Double pendulum $\left( d \ = \ 2 \right)$ : masses $m _ { 1 } , m _ { 2 }$ on massless rods of lengths $\ell _ { 1 } , \ell _ { 2 }$ , with vertical angles $\boldsymbol { x } = ( \vartheta _ { 1 } , \vartheta _ { 2 } )$ The drift is $M _ { \lambda } ( x ) \ddot { x } = F _ { \lambda } ( x , \dot { x } )$ with state-dependent mass matrix $M _ { \lambda }$ and forces $F _ { \lambda }$ :

$$
M _ { \lambda } = \left[ \begin{array} { c c } { { ( m _ { 1 } + m _ { 2 } ) \ell _ { 1 } ^ { 2 } } } & { { \mu \cos \delta } } \\ { { \mu \cos \delta } } & { { m _ { 2 } \ell _ { 2 } ^ { 2 } } } \end{array} \right] ,
$$

$$
F _ { \lambda } = - \left[ \begin{array} { c } { { \mu \dot { \partial } _ { 2 } ^ { 2 } \sin \delta + ( m _ { 1 } + m _ { 2 } ) g \ell _ { 1 } \sin \vartheta _ { 1 } } } \\ { { - \mu \dot { \vartheta } _ { 1 } ^ { 2 } \sin \delta + m _ { 2 } g \ell _ { 2 } \sin \vartheta _ { 2 } } } \end{array} \right] ,
$$

with $\delta : = \vartheta _ { 1 } - \vartheta _ { 2 } , \mu : = m _ { 2 } \ell _ { 1 } \ell _ { 2 } ,$ and $g = 9 . 8 1$ . Truth: $( m _ { 1 } , m _ { 2 } , \ell _ { 1 } , \ell _ { 2 } ) = ( 1 , 0 . 5 , 1 , 0 . 8 ) , \sigma _ { r } = 0 . 0 2$ . Linearity in masses yields the invariant direction $( m _ { 1 } , m _ { 2 } ) $ $\alpha ( m _ { 1 } , m _ { 2 } )$ . Both angles are observed $K = 2 0$ times over $\mathcal { T } = 1 2 \ : \ : ( \sigma _ { \mathrm { o b s } } = 0 . 0 2 \mathrm { r a d } )$ The conservative divergence integrates to a bounded boundary term (38), but chaos yields a $\Theta ( \tau )$ surrogate determinant term $( \mathrm { A p p e n d i x ~ A . 3 . 6 } )$

Relativistic particle $( d = 1 )$ : rest mass $m _ { 0 } .$ charge $q ,$ accelerated by $E ( t ) ~ = ~ 1 . 8 1 \cos ( \pi t )$ up to 0.5c. Drift: $m _ { 0 } \gamma ^ { 3 } \ddot { x } = q E ( t )$ with Lorentz factor $\gamma = ( 1 -$ ${ \dot { x } } ^ { 2 } / c ^ { 2 } ) ^ { - 1 / 2 } ;$ ; truth: $( m _ { 0 } , q , c ) = ( 1 , 1 , 1 ) , \sigma _ { r } = 0 . 0 2$ . As only $q / m _ { 0 }$ governs the drift, the invariant direction is $( m _ { 0 } , q )  \alpha ( m _ { 0 } , q )$ . Position is observed $K = 2 0$ times over $\mathcal { T } = 1 2 \left( \sigma _ { \mathrm { o b s } } = 0 . 0 0 5 \right)$ . The divergence integrates to a bounded boundary term but alternates sign, so $\begin{array} { r } { \sum _ { k } | \mathrm { R e } s _ { k } | = | \nabla \cdot f _ { \lambda } | } \end{array}$ grows with $\tau$ (Appendix A.3.6).

Evaluation grid. We evaluate two residual metrics—acceleration $\mathit { \Pi } ( A _ { \lambda }  &  = \mathit { \Pi } I _ { d }$ , metric bias of) and force/torque $( A _ { \lambda } ~ = ~ M _ { \lambda }$ metric bias on)—across four inference schemes: Chain (state space): our method (Section 4; PMMH, 1,000–2,000 particles, 24,000 iterations). Collider (state space): difers from Chain solely by omitting the normalizer. Collider B-PINN: standard model (Section 2.2) with a Fourier-feature MLP (dim θ ≈ 1.4k, $N d \ \leq \ 7 2 0 )$ jointly sampled over (θ, λ) via NUTS (4 chains, 9,000 steps). Chain (Fourier MLP): augments the B-PINN with the closed-form correction $- \Delta$ (16) to restore $Z _ { \lambda }$ . We run this only on the oscillator, where state-independent $A _ { \lambda }$ and $\nabla \cdot f _ { \lambda }$ allow closed-form evaluation at each NUTS step; in the other systems, state dependence makes it prohibitive (Appendix A.4). Details and code: Appendix A.3 and https://github. com/obermm/ColliderBias.

## 5.2 Collider Bias and Its Removal

In Table 1, the chain model recovers the truth throughout: $| \log \alpha | \le 0 . 1 8 ~ ( \le 1 . 1$ posterior SDs) and identified parameters within 1.3 SDs (Appendix A.3.3). The collider fails as predicted.

Null condition. With acceleration residuals $( A _ { \lambda } =$ $I _ { d } )$ on the Markovian state-space latent, both biases are inactive: $\Delta ( \lambda )$ in Eq. (22) is constant, yielding bit-identical chain and collider draws.

Table 1: The chain model recovers the truth in every system; the collider fails where Section 3 predicts. log α: posterior mean of the log scale factor along the weakly identified drift-invariant direction (Eq. (37); correct: ≈ 0). Other columns: posterior mean of an identified quantity divided by its true value (correct: 1.00). Bold: of by more than a factor of two. †: chains did not converge or stalled; median of the draws, for both columns of the system. ‡: exact Kalman likelihood, since PMMH cannot reach this posterior. All parameters in Appendix A.3.3.
<table><tr><td></td><td colspan="2">Oscillator</td><td colspan="2">Double Pendulum</td><td colspan="2">Relativistic Particle</td></tr><tr><td>Model (latent)</td><td>log α</td><td> $c / m$ </td><td>log α</td><td> $m _ { 2 } / m _ { 1 }$ </td><td>log α</td><td>C</td></tr><tr><td colspan="7">Acceleration residuals  $( \boldsymbol { A } _ { \lambda } = \boldsymbol { I } _ { d } ) \colon$  metric bias inactive</td></tr><tr><td>Chain (state space)</td><td>-0.01</td><td>0.99</td><td>-0.03</td><td>0.99</td><td>-0.02</td><td>0.89</td></tr><tr><td>Collider (state space)</td><td>-0.01</td><td>0.99</td><td>-0.03</td><td>0.99</td><td>-0.02</td><td>0.89</td></tr><tr><td>Chain (Fourier MLP)</td><td>-0.00</td><td>1.00</td><td></td><td></td><td></td><td></td></tr><tr><td>Collider B-PINN (Fourier MLP)</td><td>-0.00</td><td>0.61</td><td> $- 0 . 0 3 ^ { \dagger }$ </td><td>0.94</td><td>+0.00</td><td>1.78</td></tr><tr><td colspan="7">Force / torque residuals  $( A _ { \lambda } = M _ { \lambda } ) \colon$  metric bias active</td></tr><tr><td>Chain (state space)</td><td>-0.01</td><td>1.01</td><td>+0.04</td><td>1.00</td><td>-0.18</td><td>0.89</td></tr><tr><td>Collider (state space)</td><td> $- 4 2 . 4 2 ^ { \ddagger }$ </td><td>27</td><td> $- \mathbf { 2 . 3 7 } ^ { \dagger }$ </td><td>0.30</td><td> $- 8 . 1 7 ^ { \dagger }$ </td><td> $\mathbf { 4 . 1 . 1 0 ^ { 3 } }$ </td></tr><tr><td>Chain (Fourier MLP)</td><td>+0.29</td><td>0.74</td><td></td><td></td><td></td><td></td></tr><tr><td>Collider B-PINN (Fourier MLP)</td><td>-0.93</td><td>2.05</td><td>-2.79</td><td>0.54</td><td>-1.88</td><td>1.61</td></tr></table>

![](images/5424d0f3513edf534077456117718b4c9d0bd36cdad9c09bf6216bb961e6ff66.jpg)  
Figure 3: Longer records amplify the dissipation bias (oscillator, acceleration residuals). Posterior mean of $\log ( c / m )$ minus its true value, with ±1 posterior standard deviation. The chain stays near the truth on both latents, and its posterior contracts as $\tau$ grows; on the state-space latent the collider coincides with it (open rings). The B-PINN drifts further from the truth while its posterior width stays constant, so its error grows from 1.55 to 2.67 standard deviations.

Mechanism 1: residual metric bias. Under force residuals, the metric term in Eq. (16) rewards small masses by adding $- T d \log \alpha$ to the state-space logdensity and ≈ $- n _ { \mathrm { e f f } }$ log α to the B-PINN, opposed along the invariant ray only by process noise $( \mathrm { A p - }$ pendix A.2). On state-space latents $( T = 1 , 2 4 8$ for the oscillator), this reward dominates: the Kalman posterior collapses to log $\alpha = - 4 2 . 4 ,$ and PMMH drifts toward zero mass until Euler integration fails. Measured $\Delta ( \lambda )$ values match Eq. (22) to machine precision (Appendix A.3.8). On the B-PINN, a weaker reward $( n _ { \mathrm { e f f } } \approx 1 1 0 )$ is checked by noise, yet mass scales are still underestimated by factors of 2.5 to 16. The damage is not confined to the drift-invariant direction: since det $M _ { \lambda } \propto \ell _ { 1 } ^ { 2 } \ell _ { 2 } ^ { 2 }$ also rewards short rods, the pen dulum B-PINN shrinks $\ell _ { 1 }$ and $\ell _ { 2 }$ to 0.14 and 0.27 of their true values, so removing the scale ambiguity by reparameterization would not remove the bias.

Mechanism 2: dissipation bias. Omitting divergence (17) biases B-PINNs toward underestimating dissipation. On the oscillator with acceleration residuals, B-PINN underestimates damping $( c / m = 0 . 7 3$ vs. true 1.20; Fig. 2). This is systematic: across ten dataset resamples, B-PINN underestimates $\log ( c / m )$ by 0.42±0.03 and lies below the truth on every dataset, with 90% credible intervals that cover the truth in only 4 of 10 cases, whereas our chain’s intervals always do (Appendix A.3.4). With $\sigma _ { r }$ matched to the data, the bias persists for four tested $\sigma _ { r } ~ \in ~ [ 0 . 0 5 , 1 ] ~ ( \mathrm { A p } { - }$ pendix A.3.5). For the relativistic particle, B-PINN overestimates the speed of light $( c = 1 . 7 8$ vs. 1.00): although the divergence integral vanishes, its absolute path integral does not, penalizing smaller c by 9.5 nats per e-fold (Appendix A.3.6). The pendulum is unassessed, as its B-PINN does not converge.

Longer records amplify dissipation bias. As the penalty scales as Θ(T ), longer horizons $( \mathcal { T } \ =$ $1 2 \  \ 4 8 )$ push B-PINN damping estimates further from the truth without posterior contraction (Fig. 3). Reweighting the chain posterior by $e ^ { \Delta _ { \mathrm { d i s s } } }$ predicts the bias’s direction, growth, and 60–75% of its size $( \mathrm { A p - }$ pendix A.3.8).

Restoring the normalizer on the surrogate. Chain (Fourier MLP) adds −∆ to an otherwise identical B-PINN. Under acceleration residuals on the oscillator, it removes 97.5% of the $\log ( c / m )$ bias. Under force residuals, where metric correction uses approximate $n _ { \mathrm { e f f } }$ , it eliminates two-thirds of the mass bias (overshooting slightly; Appendix A.3.7). Thus, the missing normalizer, not the neural surrogate, causes the pathology.

Comparison with nested inference. PMMH is up to 13× faster than the B-PINN, depending on the system state size (Appendix A.3.1). We further benchmark PMMH against unmodified public reimplementations of NSVI [10] and NPSGLD [11] on Euler–Maruyama latents with closed-form $Z _ { \lambda } \quad ( \mathrm { A p - }$ pendix A.4). PMMH converges across all settings (0.51–6.71 s/ESS). NPSGLD fails to converge, requiring 21×–2,900× more compute per ESS while underestimating posterior SDs by up to 30×. NSVI converges to a biased damping ratio on the oscillator $( c / m \approx 0 . 8 0$ vs. 1.20) and raising its compute by a factor of 100 does not remove this bias.

## 6 CONCLUSIONS

B-PINNs sufer from an overlooked structural defect: conditioning on virtual residual observations turns physical parameters λ and trajectory surrogates into colliders. This implicitly neglects the parameterdependent normalizing constant $Z _ { \lambda }$ , systematically biasing posterior inference. Even when initialized at the ground truth, standard B-PINNs underestimate damping and mass, and grow increasingly overconfident with longer observation records.

We approximated two bias mechanisms in closed form, confirmed them across three systems, yet showed that analytical corrections can remain insuficient. To eliminate collider bias entirely, we reformulate the trajectory as a Markovian state-space model. This enables an asymptotically exact particle MCMC scheme that avoids both the computational overhead of nested Langevin dynamics [11] and the variational gap of nested VI [10].

When are B-PINNs safe? Collider B-PINNs yield unbiased posteriors if two null conditions hold simultaneously (Section 3; Appendix Table 2): (i) the residual metric $A _ { \lambda }$ is parameter-independent (e.g., measuring acceleration rather than force); and (ii) the dissipation bias (17) is negligible, with $\Delta _ { \mathrm { d i s s } }$ remaining roughly constant across posterior support. Otherwise, practitioners must adopt a Markovian chain formulation or, when $A _ { \lambda }$ and $\nabla \cdot f _ { \lambda }$ are state-independent, apply the closed-form correction −∆ directly.

Limitations and Future Work. Our particle MCMC scheme targets only stochastic ODEs $\left( \Sigma _ { r } \succ 0 \right)$ in low-to-moderate dimensions; particle filters degenerate as $\sigma _ { r }  0$ and face weight collapse on finely discretized PDEs. Because collider bias persists in PDE systems, extending our framework to spatial domains will require scalable alternatives such as ensemble Kalman or score-based filtering. Furthermore, selecting $\Sigma _ { r }$ to prevent residual misspecification remains open. Ideally $\Sigma _ { r }$ would be inferred jointly with $\lambda ,$ but this introduces a third collider-bias mechanism that we leave to future work.

Implications. Although standard B-PINNs rely on collider structures, the resulting bias has gone largely unnoticed, as they appear well-behaved when the null conditions approximately hold. When violated, MCMC still converges cleanly to confidently wrong posteriors whose error is undetectable without ground truth. Crucially, this distortion peaks in weakly identified regimes and long horizons with sparse data—precisely when Bayesian rigor matters most. Practitioners should therefore check these null condi tions and prefer a chain formulation.

## ACKNOWLEDGEMENTS

This research was funded by the Austrian Research Promotion Agency (FFG) via the project VENTUS (910263) under the FFG AI for Green program.

## References

[1] Alex Alberts and Ilias Bilionis. Physics-informed information field theory for modeling physical systems with uncertainty quantification. Journal of Computational Physics, 486:112100, 2023. doi: 10.1016/j.jcp.2023.112100.

[2] Christophe Andrieu, Arnaud Doucet, and Roman Holenstein. Particle Markov chain Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(3):269–342, 2010. doi: 10.1111/j.1467-9868.2009.00736.x.

[3] C´edric Archambeau, Manfred Opper, Yuan Shen, Dan Cornford, and John Shawe-Taylor. Variational inference for difusion processes. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007. URL https://proceedings. neurips.cc/paper\_files/paper/2007/ file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf.

[4] Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul A. Szerlip, Paul Horsfall, and Noah D. Goodman. Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research, 20(28):1–6, 2019.

[5] Maxwell Bolt. nif-replication: A JAX reimplementation of the Neural Information Field Filter. https://github.com/BoltMaxwell/niffreplication, 2026. Commit dab4403.

[6] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs. https://github. com/jax-ml/jax, 2018.

[7] Detlef D¨urr and Alexander Bach. The Onsager– Machlup function as Lagrangian for the most probable path of a difusion process. Communications in Mathematical Physics, 60(2):153–170, 1978. doi: 10.1007/BF01609446.

[8] Herbert Goldstein, Charles Poole, and John Safko. Classical Mechanics. Addison Wesley, third edition, 2002.

[9] Neil J. Gordon, David J. Salmond, and Adrian F. M. Smith. Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F (Radar and Signal Processing), 140(2): 107–113, 1993. doi: 10.1049/ip-f-2.1993.0015.

[10] Kairui Hao and Ilias Bilionis. An information field theory approach to Bayesian state and parameter estimation in dynamical systems. Journal of Computational Physics, 512:113139, 2024. doi: 10.1016/j.jcp.2024.113139.

[11] Kairui Hao and Ilias Bilionis. Neural information field filter. Mechanical Systems and Signal Processing, 226:112253, 2025. doi: 10.1016/j.ymssp. 2024.112253.

[12] Matthew D. Hofman and Andrew Gelman. The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47):1593– 1623, 2014. URL http://jmlr.org/papers/ v15/hoffman14a.html.

[13] Daphne Koller and Nir Friedman. Probabilistic Graphical Models: Principles and Techniques. MIT Press, Cambridge, MA, 2009.

[14] Kevin Linka, Amelie Sch¨afer, Xuhui Meng, Zongren Zou, George Em Karniadakis, and Ellen Kuhl. Bayesian physics-informed neural networks for real-world nonlinear dynamical systems. Computer Methods in Applied Mechanics and Engineering, 402:115346, 2022. doi: 10.1016/j.cma. 2022.115346.

[15] Jun S. Liu and Rong Chen. Sequential Monte Carlo methods for dynamic systems. Journal of the American Statistical Association, 93(443): 1032–1044, 1998. URL http://www.jstor.org/ stable/2669847.

[16] Hyeonbin Moon, Hanbin Cho, Wabi Demeke, Byungki Ryu, and Seunghwa Ryu. Thermal conductivity estimation of thermoelectric materials with uncertainty quantification using Bayesian physics-informed neural networks, 2025. arXiv:2510.16723.

[17] Iain Murray, Zoubin Ghahramani, and David J. C. MacKay. MCMC for doubly-intractable distributions. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence, pages 359–366. AUAI Press, 2006.

[18] Richard M. Murray, Zexiang Li, and S. Shankar Sastry. A Mathematical Introduction to Robotic Manipulation. CRC Press, 1994.

[19] Lars Onsager and Stefan Machlup. Fluctuations and irreversible processes. Physical Review, 91(6): 1505–1512, 1953. doi: 10.1103/PhysRev.91.1505.

[20] Andrew Pensoneault and Xueyu Zhu. Eficient Bayesian physics informed neural networks for inverse problems via ensemble Kalman inversion. Journal of Computational Physics, 508:113006, 2024. doi: 10.1016/j.jcp.2024.113006.

[21] Du Phan, Neeraj Pradhan, and Martin Jankowiak. Composable efects for flexible and accelerated probabilistic programming in NumPyro, 2019. arXiv:1912.11554.

[22] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j.jcp. 2018.10.045.

[23] Simo S¨arkk¨a and Arno Solin. Applied Stochastic Diferential Equations, volume 10 of Institute of Mathematical Statistics Textbooks. Cambridge University Press, 2019. doi: 10.1017/ 9781108186735.

[24] Khemraj Shukla, Zongren Zou, Theo Kaeufer, Michael Triantafyllou, and George Em Karniadakis. Uncertainty quantification in PINNs for turbulent flows: Bayesian inference and repulsive ensembles, 2026. arXiv:2604.17156.

[25] Simon Stock, Jochen Stiasny, Davood Babazadeh, Christian Becker, and Spyros Chatzivasileiadis. Bayesian physics-informed neural networks for robust system identification of power systems. In 2023 IEEE Belgrade PowerTech, pages 1–6. IEEE, 2023. doi: 10.1109/PowerTech55446.2023. 10202692.

[26] Luning Sun and Jian-Xun Wang. Physicsconstrained Bayesian neural network for fluid flow reconstruction with sparse and noisy data. Theoretical and Applied Mechanics Letters, 10(3):161– 169, 2020. doi: 10.1016/j.taml.2020.01.031.

[27] Sifan Wang, Hanwen Wang, and Paris Perdikaris. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 384:113938, 2021. doi: 10.1016/j.cma. 2021.113938.

[28] Eugene Wong and Moshe Zakai. On the convergence of ordinary integrals to stochastic integrals. The Annals of Mathematical Statistics, 36(5): 1560–1564, 1965. doi: 10.1214/aoms/1177699916.

[29] Liu Yang, Xuhui Meng, and George Em Karniadakis. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. Journal of Computational Physics, 425:109913, 2021. doi: 10.1016/j. jcp.2020.109913.

## A SUPPLEMENTARY MATERIAL

## A.1 Laplace Approximation of the Collider Bias for Global Surrogates

This appendix derives the two-term decomposition (16) for a global surrogate such as a neural network. Step 1 approximates $Z _ { \lambda }$ by Laplace’s method, Step 2 identifies the two factors through which λ enters the resulting curvature, Steps 3 and 4 diferentiate with respect to these factors, and Step 5 collects the terms; a worked example for the oscillator follows. Along drift-invariant directions, the metric bias admits exact statements, derived in Appendix A.2.

Setup. We write the residual (7) in continuous time,

$$
r _ { \theta } ( t , \lambda ) : = A _ { \lambda } ( z _ { \theta } ( t ) ) \left( \dot { v } _ { \theta } ( t ) - a _ { \lambda } ( z _ { \theta } ( t ) , t ) \right) , \qquad v _ { \theta } : = \dot { x } _ { \theta } ,\tag{23}
$$

so that $r _ { n } ( \theta , \lambda ) = r _ { \theta } ( t _ { n } , \lambda ) ;$ ; since $v _ { \theta }$ is the derivative of $x _ { \theta } ,$ , only the acceleration equation can be violated. The surrogate carries the weight prior $p ( \theta ) = \mathcal { N } ( 0 , P ^ { - 1 } )$ of Section 3, where $P  0$ corresponds to a flat reference measure. Chain and collider share this prior and the energy (8) and difer only in their normalizers, $Z _ { \lambda }$ in (12) and $Z _ { r }$ in (10), so that $\Delta = \log Z _ { \lambda } - \log Z _ { r } + \mathrm { c o n s t }$ by (14). Only $Z _ { \lambda }$ depends on $\lambda ;$ since $\Sigma _ { r }$ is fixed, log $Z _ { r }$ is a constant.

Step 1: Laplace approximation. Let $\theta ^ { * } ( \lambda )$ minimize $E [ \theta , \lambda ] - \log p ( \theta )$ , the negative logarithm of the data free integrand of $Z _ { \lambda }$ . A second-order expansion around $\theta ^ { * }$ gives

$$
\log Z _ { \lambda } \approx - E [ \theta ^ { * } , \lambda ] - \frac 1 2 \theta ^ { * \top } { \cal P } \theta ^ { * } - \frac 1 2 \log \operatorname * { d e t } H _ { \lambda } + \mathrm { c o n s t } , \qquad H _ { \lambda } : = \nabla _ { \theta } ^ { 2 } E \big | _ { \theta ^ { * } } + { \cal P } ,\tag{24}
$$

where the constant ( <sup>1</sup> log det $P )$ does not depend on λ. We drop the first two terms, assuming that the surrogate (nearly) solves the ODE, $E [ \theta ^ { * } , \lambda ] \approx 0$ , and that $\theta ^ { * }$ follows an ODE solution rather than the prior mode, so that $\dot { \bar { \mathbf { \rho } } } \theta ^ { * \top } \dot { P } \dot { \theta } ^ { * }$ depends only weakly on λ. This is exact for linear, unforced dynamics and a surrogate with $x _ { \theta } \equiv 0$ at $\theta = 0 .$ , where $\theta ^ { * } = 0$ for every λ. Since the residuals vanish at $\theta ^ { * }$ , so does the part of $\nabla _ { \theta } ^ { 2 } E$ that is proportional to them, which leaves $H _ { \lambda } \approx G _ { \lambda } + P$ with the Gauss–Newton curvature

$$
G _ { \lambda } : = h \sum _ { n = 1 } ^ { N } \left( \frac { \partial r _ { \theta } ( t _ { n } , \lambda ) } { \partial \theta } \right) ^ { \top } \Sigma _ { r } ^ { - 1 } \frac { \partial r _ { \theta } ( t _ { n } , \lambda ) } { \partial \theta } , \qquad \log Z _ { \lambda } \approx - \frac { 1 } { 2 } \log \operatorname* { d e t } ( G _ { \lambda } + P ) + \mathrm { c o n s t } .\tag{25}
$$

Since log $Z _ { r }$ is constant, this gives Eq. (15); the remaining steps evaluate its first term.

Step 2: Two factors of the curvature. Let $B ( t ) : = \partial z _ { \theta } ( t ) / \partial \theta$ be the tangent map of the surrogate at $\theta ^ { * }$ , and $D f _ { \lambda } ( t ) : = \partial f _ { \lambda } / \partial z$ the drift Jacobian along the fitted trajectory $z ( t ) : = z _ { \theta ^ { \ast } } ( t )$ . By the chain rule,

$$
\frac { \partial r _ { \theta } ( t , \lambda ) } { \partial \theta } \approx A _ { \lambda } ( z ( t ) ) \left( L _ { \lambda } B \right) _ { v } ( t ) , \qquad L _ { \lambda } : = \frac { \mathrm { d } } { \mathrm { d } t } - D f _ { \lambda } ( t ) ,\tag{26}
$$

where $( \cdot ) _ { v }$ denotes the velocity block. The position block of $L _ { \lambda } B$ vanishes because $v _ { \theta } ~ = ~ { \dot { x } } _ { \theta }$ , and the term containing the derivative of $A _ { \lambda }$ multiplies the residual, which vanishes at $\theta ^ { * }$ . Stacking over the collocation points gives

$$
\begin{array} { r } { G _ { \lambda } = \mathbf { B } ^ { \top } \mathbf { L } _ { \lambda } ^ { \top } \mathbf { A } _ { \lambda } ^ { \top } \mathbf { S } ^ { - 1 } \mathbf { A } _ { \lambda } \mathbf { L } _ { \lambda } \mathbf { B } , \quad \quad \mathbf { A } _ { \lambda } = \operatorname { d i a g } \left( A _ { \lambda } ( z ( t _ { 1 } ) ) , \ldots , A _ { \lambda } ( z ( t _ { N } ) ) \right) , \quad \mathbf { S } = h ^ { - 1 } \operatorname { d i a g } ( \Sigma _ { r } , \ldots , \Sigma _ { r } ) . } \end{array}\tag{27}
$$

The stacked tangent map B depends on λ only through $\theta ^ { * } ( \lambda )$ , which we hold fixed. Since S is fixed, λ enters $G _ { \lambda }$ through two factors: the residual metric $\mathbf { A } _ { \lambda }$ and the drift operator $\mathbf { L } _ { \lambda }$ . Since these factors are rectangular, log det $\left( G _ { \lambda } + P \right)$ does not split into a sum over them; we therefore diferentiate it with respect to each factor in turn, holding the other fixed.

Step 3: Residual metric (Mechanism 1). Let $A _ { \lambda } ( z ) = \kappa _ { \lambda } \bar { A } ( z )$ with a scalar $\kappa _ { \lambda } > 0$ and a λ-independent $\bar { A } , \mathrm { e . g . } , A _ { \lambda } = m$ for the oscillator. Then the metric scale factors out of (27),

$$
G _ { \lambda } = \kappa _ { \lambda } ^ { 2 } \tilde { G } , \qquad \tilde { G } : = \mathbf { B } ^ { \top } \mathbf { L } _ { \lambda } ^ { \top } \bar { \mathbf { A } } ^ { \top } \mathbf { S } ^ { - 1 } \bar { \mathbf { A } } \mathbf { L } _ { \lambda } \mathbf { B } ,\tag{28}
$$

where $\bar { \mathbf { A } } : = \operatorname { d i a g } \left( \bar { A } ( z ( t _ { 1 } ) ) , \ldots , \bar { A } ( z ( t _ { N } ) ) \right)$ , and $\tilde { G }$ does not depend on $\kappa _ { \lambda }$ . Since $\partial G _ { \lambda } / \partial \log \kappa _ { \lambda } = 2 G _ { \lambda }$ , Jacobi’s formula gives

$$
{ \frac { \partial } { \partial \log \kappa _ { \lambda } } } \left( - { \frac { 1 } { 2 } } \log \operatorname * { d e t } ( G _ { \lambda } + P ) \right) = - \operatorname { t r } \left[ ( G _ { \lambda } + P ) ^ { - 1 } G _ { \lambda } \right] = - n _ { \mathrm { e f f } } .\tag{29}
$$

When is $n _ { \mathrm { e f f } }$ constant? For $P \succ 0$ , let $\left\{ g _ { k } \right\}$ be the eigenvalues of $P ^ { - 1 / 2 } \tilde { G } P ^ { - 1 / 2 } \succeq 0$ . Then det $( G _ { \lambda } + P ) =$ $\begin{array} { r } { \operatorname* { d e t } ( P ) \prod _ { k } ( 1 + g _ { k } \kappa _ { \lambda } ^ { 2 } ) } \end{array}$ , so that

$$
- \frac { 1 } { 2 } \log \operatorname * { d e t } ( G _ { \lambda } + P ) = - \frac { 1 } { 2 } \sum _ { k } \log \bigl ( 1 + g _ { k } \kappa _ { \lambda } ^ { 2 } \bigr ) + \mathrm { c o n s t } , \qquad n _ { \mathrm { e f f } } = \sum _ { k } \frac { g _ { k } \kappa _ { \lambda } ^ { 2 } } { 1 + g _ { k } \kappa _ { \lambda } ^ { 2 } } .\tag{30}
$$

Each summand lies in [0, 1) and is nonzero only for $g _ { k } \ > \ 0$ , so $n _ { \mathrm { e f f } } \ <$ rank $G _ { \lambda } ~ \leq$ min $( N d ,$ dim $\theta )$ whenever $G _ { \lambda } \neq 0$ . Once the physics dominates the prior on every direction it constrains $( g _ { k } \kappa _ { \lambda } ^ { 2 } \gg 1 ) , n _ { \mathrm { e f f } }$ saturates at rank $G _ { \lambda }$ and no longer depends on $\kappa _ { \lambda } ;$ we call this the saturated regime. For $P = 0$ and full-rank $G _ { \lambda }$ (which requires dim $\theta \leq N d ) , n _ { \mathrm { e f f } } = \dim \theta$ for every $\kappa _ { \lambda }$

Mechanism 1 (residual metric). Since $Z _ { r }$ does not depend on $\kappa _ { \lambda }$ , Eq. (29) gives $\partial \Delta / \partial \log \kappa _ { \lambda } ~ = ~ - n _ { \mathrm { e f f } }$ With $\textstyle \sum _ { n }$ log | det $A _ { \lambda } ( z ( t _ { n } ) ) \vert ~ = ~ N d \log \kappa _ { \lambda } + \mathrm { c o n s t }$ , this integrates, for constant $n _ { \mathrm { e f f } }$ , to the metric term $\textstyle - { \frac { n _ { \mathrm { e f f } } } { N d } } \sum _ { n } \mathrm { \tilde { l o g } }$ | det $A _ { \lambda } ( z ( t _ { n } ) ) \vert$ of (16). For a state-dependent $A _ { \lambda } ( z ( t ) )$ ), the term holds locally: on time windows that are short compared with the variation of $A _ { \lambda } ( z ( t ) )$ but long compared with the resolution of the surrogate, $A _ { \lambda }$ is nearly constant, and if the constrained directions are spread uniformly over time and components, each window contributes its share $n _ { \mathrm { e f f } } / N d$ of the local sum of log | det $A _ { \lambda } |$ . This neglects the coupling between windows and the anisotropy of $A _ { \lambda }$ relative to $\Sigma _ { r }$ , both absent in the scalar case.

State-space latent. On the Markovian latent of Section $4 , \theta$ consists of the Td velocities under a flat prior $( P = 0 )$ and $\partial r / \partial v$ is block lower-triangular with invertible diagonal blocks $A _ { \lambda } ( z _ { i - 1 } ) / h$ . Hence $n _ { \mathrm { e f f } } = T d = N d$ exactly, and the metric term becomes the exact collider bias (22).

Step 4: Drift operator (Mechanism 2). We set $A _ { \lambda } = I _ { d } ,$ so that λ enters (27) only through $\mathbf { L } _ { \lambda }$ . We assume that the span of the surrogate’s tangent map is approximately invariant under $L _ { \lambda }$ , with a λ-independent Gram determinant, and that the physics dominates P on the resolved modes. Then $\frac { 1 } { 2 }$ log det $( G _ { \lambda } + P )$ equals log | det $L _ { \lambda } |$ restricted to the resolved modes, up to λ-independent terms. On the position component, $L _ { \lambda }$ acts as $\begin{array} { r } { \dot { \mathrm { d } } ^ { 2 } - \frac { \partial \dot { a _ { \lambda } } } { \partial v } \dot { \mathrm { d } } t - \frac { \partial \dot { a _ { \lambda } } } { \partial x } } \end{array}$ , whose matrix symbol at frequency ω has, by the Schur complement, the same determinant as $\hat { L } _ { \lambda } ( \omega ) : = i \omega I _ { 2 d } - D f _ { \lambda }$ . For constant $D f _ { \lambda }$ (relaxed below), a surrogate with efective bandwidth Ω on [0, T ] resolves real Fourier modes at density $\tau / \pi ,$ , so that

$$
\frac { 1 } { 2 } \log \operatorname * { d e t } ( G _ { \lambda } + P ) \approx \frac { \mathcal { T } } { \pi } \int _ { 0 } ^ { \Omega } \log \left| \operatorname * { d e t } \hat { L } _ { \lambda } ( \omega ) \right| \mathrm { d } \omega .\tag{31}
$$

The eigenvalue integral. Let $\{ s _ { k } \} _ { k = 1 } ^ { 2 d }$ be the eigenvalues of $D f _ { \lambda }$ , so that log | det $\begin{array} { r } { \hat { L } _ { \lambda } ( \omega ) | = \sum _ { k } \log | i \omega - s _ { k } | } \end{array}$ . The integral (31) diverges as $\Omega  \infty .$ , but at a rate independent of $\lambda ;$ diferentiating with respect to the eigenvalues isolates the λ-dependent part. For a complex pair $s = - \rho \pm i \beta$ with $\beta > 0 \ ( \rho > 0$ damped, $\rho < 0 ~ \mathrm { g r o w i n g } )$ $\begin{array} { r } { I ( \rho , \beta ) : = \int _ { 0 } ^ { \Omega } \log | ( i \omega - s ) ( i \omega - \bar { s } ) | } \end{array}$ dω satisfies

$$
\frac { \partial I } { \partial \rho } = \int _ { 0 } ^ { \Omega } \left( \frac { \rho } { ( \omega - \beta ) ^ { 2 } + \rho ^ { 2 } } + \frac { \rho } { ( \omega + \beta ) ^ { 2 } + \rho ^ { 2 } } \right) \mathrm { d } \omega = \arctan \frac { \Omega - \beta } { \rho } + \arctan \frac { \Omega + \beta } { \rho } \ = \ \pi \sin ( \rho ) + \mathcal { O } ( 1 / \Omega ) ,\tag{32}
$$

$$
\frac { \partial I } { \partial \beta } = \frac { 1 } { 2 } \log \frac { ( \Omega + \beta ) ^ { 2 } + \rho ^ { 2 } } { ( \Omega - \beta ) ^ { 2 } + \rho ^ { 2 } } = \mathcal { O } ( \beta / \Omega ) ,\tag{33}
$$

where the boundary terms at $\omega = 0$ cancel. Up to $\mathcal { O } ( 1 / \Omega )$ , the damping rate thus enters with weight $\pi$ and the frequency $\beta$ not at all, so each pair contributes $\pi | \rho |$ . A real eigenvalue $s = - \rho$ gives $\begin{array} { r } { \partial _ { \rho } \int _ { 0 } ^ { \Omega } \log | i \omega + \rho | \mathrm { d } \omega = } \end{array}$ arctan $\begin{array} { r } { ( \Omega / \rho ) = \frac { \pi } { 2 } \mathrm { s i g n } ( \rho ) + \mathcal { O } ( 1 / \Omega ) } \end{array}$ , i.e., a contribution ${ \frac { \pi } { 2 } } \vert \rho \vert$ . Every eigenvalue therefore contributes $\frac { \pi } { 2 } | \mathrm { R e } s _ { k } |$ and multiplying by the mode density $\tau / \pi$ gives

$$
{ \frac { 1 } { 2 } } \log \operatorname* { d e t } ( G _ { \lambda } + P ) = { \frac { \mathcal { T } } { 2 } } \sum _ { k = 1 } ^ { 2 d } \left| \operatorname { R e } s _ { k } \right| + \operatorname { c o n s t } + { \mathcal { R } } , \qquad { \mathcal { R } } = { \mathcal { O } } ( { \mathcal { T } } / \Omega ) .\tag{34}
$$

Without growing modes $( \mathrm { R e } s _ { k } \le 0$ for all k), $\begin{array} { r } { \sum _ { k } \left| \mathrm { R e } s _ { k } \right| = - \mathrm { t r } ( D f _ { \lambda } ) = - \boldsymbol { \nabla } \cdot \boldsymbol { f } _ { \lambda } } \end{array}$ , and inserting (34) into (15) gives

$$
\Delta _ { \mathrm { d i s s } } ( \lambda ) = \frac { \mathcal { T } } { 2 } \nabla \cdot \boldsymbol { f } _ { \lambda } + \mathrm { c o n s t } + \mathcal { R } .\tag{35}
$$

Otherwise, $\Delta ( \lambda )$ is not determined by the divergence alone: growing modes are penalized like damped ones, and a divergence that changes sign along the orbit enters through its absolute value. Appendix A.3.6 discusses both cases for conservative systems.

Nonlinear drift. For nonlinear dynamics, we treat $D f _ { \lambda } ( z ( t ) )$ as constant on windows that are short compared with $\tau$ but long compared with the correlation time $\Omega ^ { - 1 }$ of the surrogate and the relaxation times $| \mathrm { R e } s _ { k } | ^ { - 1 }$ Summing over windows replaces $\tau \nabla \cdot f _ { \lambda }$ by $\begin{array} { r l } { \int _ { 0 } ^ { \mathcal { T } } \nabla \cdot f _ { \lambda } ( z ( t ) , t ) } \end{array}$ dt and gives Eq. (17). Its leading term is the Onsager–Machlup correction to the path action [7, 19], obtained here as the fluctuation determinant of a bandlimited surrogate. The remainder R collects the $\mathcal { O } ( 1 / \Omega )$ tails of (32) and (33), the residual floor $E [ \theta ^ { * } , \lambda ]$ of a finite-capacity fit, and the windowing. Since all of these grow with T at the same rate as the leading term, their relative size is set by Ω compared with the time scales of the system, not by the length of the record.

State-space latent. Under the Euler–Maruyama transition (18), $v _ { i }$ depends only on the noise increments $\Delta W _ { j } : =$ $\sqrt { h } \xi _ { j }$ up to step i, so the Jacobian of $( \Delta W _ { 1 } , \\dots , \Delta W _ { T } ) \mapsto ( v _ { 1 } , \dots , v _ { T } )$ is block lower-triangular with diagonal blocks $A _ { \lambda } \big ( z _ { i - 1 } \big ) ^ { - 1 } \Sigma _ { r } ^ { 1 / 2 }$ , which do not involve the drift. This is why the exact collider bias (22) has no dissipation term: the Markovian latent normalizes the path density step by step (the Itˆo pre-point form), whereas a global surrogate, which imposes no ordering, yields the midpoint (Onsager–Machlup) form, consistent with the Wong– Zakai limit of smooth approximations to Brownian paths [28].

Step 5: Collecting the terms. Adding the metric term of Step 3 and the dissipation term of Step 4 gives Eq. (16); without growing modes, its dissipation term is the divergence integral (17). Of the four simplifications listed in Section 3, three need a caveat in practice. Simplification (i) fails when the data-free integrand of $Z _ { \lambda }$ is dominated by the prior rather than the physics; the terms dropped from (24) then depend on $\lambda .$ For (ii), when $x \equiv 0$ solves the ODE, $\theta ^ { * } = 0$ , where the tangent map of a neural surrogate degenerates; we therefore evaluate the curvature at a representative state on the solution manifold (Appendix A.3.7). Since the factors in (27) are rectangular, the additivity (iv) holds only to first order. Eq. (16) thus captures the dominant dependence of $\Delta$ on λ and shows where the collider bias is large, but adding $- \Delta$ to the log-posterior of a B-PINN need not recover the chain posterior exactly. Table 2 summarizes both mechanisms.  
Table 2: Taxonomy of structural biases in collider B-PINNs. Scaling denotes the asymptotic rate of the collider bias $\Delta = \log ( p _ { \mathrm { c o l l i d e r } } / p _ { \mathrm { c h a i n } } )$ . Null conditions indicate when $\Delta$ is invariant to inferred parameters; $\left\{ s _ { k } \right\}$ are eigenvalues of $D f _ { \lambda }$
<table><tr><td>Mechanism</td><td>Physical Origin</td><td>Scaling</td><td>Null Condition</td></tr><tr><td>Residual Metric</td><td>State-space map  $A _ { \lambda }$  depends on λ</td><td> $\Theta ( n _ { \mathrm { e f f } } ) ~ \mathrm { o r }$   $\Theta ( \tau )$ </td><td> $A _ { \lambda }$  invariant to λ</td></tr><tr><td>Dissipation</td><td>Phase-space volume contraction</td><td>Θ(T)</td><td>Markovian latent, or  $\sum _ { k } | \mathrm { R e } s _ { k } |$  invariant to λ</td></tr></table>

Worked example: the damped linear oscillator. For $m { \ddot { x } } + c { \dot { x } } + k x = \sigma _ { r } \xi$ with force residuals $( A _ { \lambda } = m$ $d = 1 )$ , the drift $\begin{array} { r } { f _ { \lambda } ( x , v ) = ( v , - \frac { k } { m } x - \frac { c } { m } v ) ^ { \top } } \end{array}$ has the constant Jacobian $D f _ { \lambda } = \left[ \begin{array} { c c } { { 0 } } & { { 1 } } \\ { { - k / m \ - c / m } } \end{array} \right]$ with divergence $\nabla \cdot f _ { \lambda } = - c / m$ . Its eigenvalues are $- \rho \pm i \beta$ with $\rho = c / ( 2 m )$ and $\beta = \sqrt { k / m - c ^ { 2 } / ( 4 m ^ { 2 } ) }$ if underdamped, and real and negative if overdamped; in both cases $\begin{array} { r } { \sum _ { k } | \mathrm { R e } s _ { k } | = c / m } \end{array}$ . No windowing is needed, and (16) reads

$$
\Delta ( \lambda ) \approx \underbrace { - n _ { \mathrm { e f f } } \log m } _ { \mathrm { m e t r i c } } \underbrace { - \frac { c \mathcal { T } } { 2 m } } _ { \mathrm { d i s s i p a t i o n } } + \mathrm { c o n s t . }\tag{36}
$$

The dissipation term penalizes large $c / m$ in proportion to $\tau ,$ so the collider underestimates damping. The stifness enters only through $\beta ,$ which by (33) contributes no leading-order term; the $\mathcal { O } ( \beta / \Omega )$ remainder, however, stil grows with $\tau$ and rewards smaller $k / m ,$ , so restoring only the leading-order terms leaves part of the stifness bias (Appendix A.3.7). Along the drift-invariant rescaling $( m , c , k ) \to \alpha ( m , c , k )$ , only the metric term varies, $\partial \Delta / \partial \log \alpha = - n _ { \mathrm { e f f } }$ : the collider is rewarded without bound for shrinking the mass scale, opposed only by the prior and by the process-noise level, which identifies $\sigma _ { r } / m$ . On the state-space latent, the same holds exactly with $n _ { \mathrm { e f f } } = N = T$ , since (22) gives $\Delta _ { \mathrm { M a r k o v } } = - T$ log m + const; Appendix A.2 discusses its efect on the collider posterior.

## A.2 Exact Collider Bias along Drift-Invariant Directions

While the decomposition derived in Appendix A.1 is generally approximate, the metric bias (Mechanism 1) admits exact statements along drift-invariant directions.

Consider a drift-invariant direction parameterized by jointly scaling masses and forces by a factor α. Suppose the scaled components follow independent log-normal priors with log-scale standard deviations $s _ { j }$ . Measuring log α relative to the prior medians $\tilde { \lambda } _ { j }$ via the precision-weighted mean

$$
\log \alpha = \frac { \sum _ { j } s _ { j } ^ { - 2 } \log ( \lambda _ { j } / \tilde { \lambda } _ { j } ) } { \sum _ { j } s _ { j } ^ { - 2 } } ,\tag{37}
$$

the prior factorizes into independent components along log α and the identified directions, with a Gaussian marginal of standard deviation $\textstyle ( \sum _ { j } s _ { j } ^ { - 2 } ) ^ { - 1 / 2 }$ along log α.

Under acceleration residuals, the residuals themselves remain invariant under this scaling. Consequently, neither the residual energy (8) nor the observational likelihood depends on α, and $Z _ { \lambda }$ is independent of α. Both the chain and the collider joint then depend on α solely through the prior, so both posteriors leave the distribution of log α exactly at its prior, for any latent representation.

Under force residuals, $A _ { \lambda } = M _ { \lambda }$ scales with α while $\Sigma _ { r }$ is fixed, so the process noise on the velocities scales with $\alpha ^ { - 1 }$ : the data constrain α through the noise level, and the chain posterior of log α difers from its prior. On top of this, the collider joint acquires an additional explicit α-dependence: on the state-space latent, $\Delta _ { \mathrm { M a r k o v } } =$ −T d log α + const exactly by Eq. (22), and on a global surrogate, $\Delta \approx - n _ { \mathrm { e f f } }$ log α + const by the metric term of Eq. (16). The collider posterior of log α is therefore the chain posterior reweighted by $\alpha ^ { - T d }$ (respectively $\alpha ^ { - n _ { \mathrm { e f f } } } )$ . This reward grows with the number of time steps (respectively constrained directions), whereas the constraint from the noise level is set by the finitely many observations; on the oscillator, where the exact Kalman likelihood is available, it places the state-space collider at log $\alpha = - 4 2 . 4$ (Table 1).

## A.3 Experimental Details and Full Results

This appendix supports Section 5 and follows its order. Appendix A.3.1 specifies the setup, and Appendix A.3.2 verifies that our method samples the chain posterior correctly, since every bias in this paper is measured against it. Appendix A.3.3 reports all parameters of the runs summarized in Table 1, and Appendix A.3.4 shows that the bias on the oscillator is reproducible across datasets, and Appendix A.3.5 how it depends on the residual scale. The next two subsections support the discussion of individual mechanisms in Section 5.2: the dissipation term on the two conservative systems (Appendix A.3.6) and the closed-form correction of the neural surrogate (Appendix A.3.7). Appendix A.3.8 details the scaling-law experiments of Figs. 3 and 4.

## A.3.1 Setup

Evaluation grid. Each of the three systems is combined with two residual metrics (acceleration and force/torque residuals) and with the inference variants listed below. In every configuration, the process noise is isotropic, $\Sigma _ { r } = \sigma _ { r } ^ { 2 } I _ { d } .$ , with $\sigma _ { r }$ fixed at its true value. Within a system, all runs share the dataset, ground truth and prior, the time discretization, number of particles, sampler budget and random seed, and the network and collocation grid of the surrogate. Two runs that difer in a single factor of the grid therefore isolate the efect of that factor, for instance of the residual metric.

Systems and data. Table 3 lists the settings that difer between systems. The data are simulated with the same semi-implicit Euler–Maruyama scheme (18) that the inference uses, but on a finer time grid: 2× finer for the oscillator and about 50× finer for the pendulum and the relativistic particle (Appendix A.3.2 shows that the results do not rely on this match). The number of Euler–Maruyama steps T used for inference and the number of particles were chosen per system by a convergence study, and are identical across all state-space runs of that system, so that chain and collider remain comparable.

Table 3: Per-system settings. The MLP surrogate maps a Fourier-feature embedding of dimension 33 through two hidden layers of width 24 to the d output coordinates. Nd is the number of residual components, i.e., collocation points times dimension.
<table><tr><td></td><td>Oscillator</td><td>Double pendulum</td><td>Relativistic particle</td></tr><tr><td>ground truth λ</td><td>(m, c, k) = (1, 1.2, 4)</td><td> $( m _ { 1 } , m _ { 2 } , \ell _ { 1 } , \ell _ { 2 } ) = ( 1 , 0 . 5 , 1 , 0 . 8 )$ </td><td> $( m _ { 0 } , q , c ) = ( 1 , 1 , 1 )$ </td></tr><tr><td>residual scale  $\sigma _ { r }$  (fixed)</td><td>0.30</td><td>0.02</td><td>0.02</td></tr><tr><td>horizon T</td><td>24</td><td></td><td>12</td></tr><tr><td>observations K</td><td>40</td><td></td><td>20</td></tr><tr><td>observation noise σobs</td><td>0.02</td><td>0.02 rad</td><td>0.005</td></tr><tr><td>Euler-Maruyama steps T</td><td>1248</td><td>1235</td><td>12160</td></tr><tr><td>particles</td><td>1000</td><td>1000</td><td>2000</td></tr><tr><td>MLP layer widths</td><td>33-24-24-1</td><td>33-24-24-2</td><td>33-24-24-1</td></tr><tr><td>dim θ</td><td>1441</td><td>1466</td><td>1441</td></tr><tr><td>residual components Nd</td><td>128</td><td>720</td><td>300</td></tr></table>

## Inference variants.

• Chain (state space) is our method (Section 4). Its latent is the Euler–Maruyama path $z _ { 0 : T }$ with the normalized transition kernel (19); a bootstrap particle filter estimates the marginal likelihood ${ \hat { p } } ( y \mid \lambda )$ , and PMMH samples λ.

• Collider (state space) uses the same latent, particle filter and sampler, but omits $Z _ { \lambda }$ . It is a control that difers from our method only in the normalizer.

• Collider B-PINN is the standard formulation of Section 2.2: a Fourier-feature MLP surrogate $x _ { \theta }$ with a Gaussian weight prior, sampled with NUTS jointly over (θ, λ).

• Chain (Fourier MLP) is the Collider B-PINN with the closed-form correction −∆ for the missing normalizer added to its log-density (Appendix A.3.7). It is run on the oscillator only (Section 5.1).

Because the oscillator is linear, the marginal likelihood of its state-space latent is also available exactly from a Kalman filter. We use this in one place. Under force residuals, PMMH cannot reach the posterior of the statespace collider (see below), so on the oscillator we sample it with an adaptive random-walk Metropolis sampler instead, targeting the exact chain likelihood multiplied by $e ^ { \Delta ( \lambda ) } = m ^ { - \hat { T } }$ from Eq. (22) (4 chains, $\hat { R } \leq 1 . 0 0 1 )$ These entries are marked with ‡. Applied to the chain model, the same sampler reproduces the PMMH posterior (e.g., log m: 0.014 ± 0.139 against 0.006 ± 0.138).

Priors. All physical parameters have log-normal priors with median at the ground truth. The standard deviations of the log-parameters are 0.4 for masses, charges and the damping coeficient c of the oscillator, and 0.25 for lengths, stifnesses and the light speed c of the relativistic particle. The Gaussian weight prior of the B-PINN follows LeCun scaling at unit gain: each weight of a layer with $n _ { \mathrm { i n } }$ inputs has variance $1 / n _ { \mathrm { i n } }$ , and each bias has variance 1.

Samplers. PMMH runs for 24,000 iterations. The first 4,000 adapt the proposal, and the first 30% of the remaining 20,000 are discarded as burn-in, which leaves 14,000 draws. NUTS runs 4 chains, each with 3,000 warm-up iterations and 6,000 draws at a maximum tree depth of 11, which gives 24,000 draws in total. All samplers use random seed 0. Each run uses a single GPU (NVIDIA Quadro RTX 8000 or NVIDIA L40) on an internal compute cluster. On the oscillator, for instance, PMMH takes 442 s and the B-PINN 5,834 s (NUTS) for these budgets.

Software and licenses. All methods are implemented in JAX [6]; NUTS and the autocorrelation-based efective sample sizes use NumPyro [4, 21]. The nested baselines of Appendix A.4 use the public reimplementation of Bolt [5], which additionally depends on Optax. JAX, NumPyro and Optax are released under the Apache License 2.0; the reimplementation does not state a license. We release our code under the MIT license.

Reporting conventions. Unless stated otherwise, tables report the posterior mean divided by the ground truth, so that 1.00 means no error. There are two exceptions. Table 1 reports the drift-invariant direction as the precision-weighted log scale factor log α of Eq. (37). Tables 7 and 9 report displacements in log space, i.e., the posterior mean of the log parameter minus the log of its true value. Since the mean of a logarithm is not the logarithm of the mean, these displacements are not exactly the logarithms of the ratios reported elsewhere; for instance, the damping ratio of the Collider B-PINN under acceleration residuals has a displacement of −0.526 (Table 9), and $\exp ( - 0 . 5 2 6 ) = 0 . 5 9$ , whereas Table 1 reports the ratio 0.61.

Runs without valid posterior samples (†). Two kinds of runs do not yield valid posterior samples. The first is the state-space collider under force residuals. Along the drift-invariant direction, its log-density gains −Td log α, which rewards small masses. Together with the log-normal prior, this still defines a proper posterior, but its mode lies far outside the range the sampler can represent; for the oscillator, where the exact Kalman likelihood locates it, at log $\alpha \approx - 4 2$ . PMMH moves toward it until the Euler–Maruyama recursion becomes numerically unstable; from then on, the particle filter returns non-finite likelihoods for a large fraction of the proposals, and only a handful of the 20,000 post-adaptation proposals are accepted. The second kind are B-PINN runs whose chains have not mixed $( \hat { R } > 1 . 0 5 )$ . For both kinds, we report the median of the draws, which describes where the sampler stopped rather than the posterior. Marginal plots show such runs as an arrow at the edge of the axis, labelled with the value reached.

## A.3.2 Validation of the Reference Posterior

All biases in this paper are measured against the chain posterior that our method samples. Three checks confirm that this reference is correct: the first tests the likelihood estimator, the second the simulation of the data, and the third the sampler.

Particle filter against the exact likelihood. For the linear oscillator, the Kalman filter gives the exact log-likelihood $\log p ( y \mid \lambda )$ . At three points along the drift-invariant direction, the particle-filter estimate deviates from it by $+ 0 . 1 4 , \ : - 0 . 1 0$ and +0.27 nats, against Monte Carlo standard deviations of 0.27, 0.38 and 0.38 across five independent filter runs. The likelihood estimator underlying PMMH thus shows no bias at the resolution relevant for our comparisons.

Simulation step. Because the data are simulated with the same scheme that the inference uses, the chain model could benefit from a matched discretization. To rule this out, we regenerated ten datasets of the oscillator configuration at a 32× finer step and computed the exact (Kalman) chain posterior on each. Its displacement from the truth is $- 0 . 0 1 \pm 0 . 0 7$ in $\log ( c / m )$ and $+ 0 . 0 4 \pm 0 . 0 2$ in $\log ( k / m )$ (mean ± standard error over datasets), and its 90% credible intervals contain the truth in 8 or 9 of the 10 datasets for every identified parameter.

A diferent sampler. On the double pendulum and the relativistic particle, we also sampled both models on the state-space latent with NUTS instead of PMMH. NUTS runs jointly over $( z _ { 0 } , \xi , \lambda )$ , where ${ \boldsymbol \xi } = ( \xi _ { 1 } , \dots , \xi _ { T } )$ are the standardized noise increments of $\operatorname { E q . }$ (18). The path $z _ { \mathrm { 0 : } T }$ is a deterministic function of these variables, and the standard normal prior on $\xi$ does not depend on λ. In this non-centred parameterization, the chain model therefore needs no explicit normalizer, while the collider difers from it by $\Delta ( \lambda )$ in (22). The resulting posteriors agree with those obtained with PMMH, and the collapse of the collider under torque residuals is reproduced; neither result is therefore an artefact of the particle filter. NUTS is thus a valid alternative to PMMH for the chain model, but a more expensive one on fine time grids: per efective sample of the drift-invariant scale, it takes 4.9 s against 2.7 s for PMMH on the pendulum (1,235 steps), and about 1,800 s against 6 s on the relativistic particle (12,160 steps; 51 h of NUTS in total).

Table 4: Oscillator. The three inference legs under both residual metrics, on one dataset and one prior, with $\sigma _ { r }$ held at its true value. Posterior mean ± standard deviation over 14000 draws (state space) and 24000 draws (B-PINN). †: a walk at the Euler map’s stability boundary, not a posterior; its median is quoted; ‡: chains that have not merged $( \hat { R } > 1 . 0 5 )$ , likewise a median rather than a posterior.
<table><tr><td>residual</td><td>leg</td><td>m (1.00)</td><td>c (1.20)</td><td>k (4.00)</td><td> $c / m \ ( 1 . 2 0 )$ </td><td> $k / m \ ( 4 . 0 0 )$ </td></tr><tr><td rowspan="3">force residual</td><td>Chain (state space)</td><td> $1 . 0 1 5 \pm 0 . 1 4$ </td><td> $1 . 2 0 9 \pm 0 . 2 0$ </td><td> $3 . 9 8 1 \pm 0 . 5 1$ </td><td> $1 . 2 0 9 \pm 0 . 2 4$ </td><td> $3 . 9 4 8 \pm 0 . 4 3$ </td></tr><tr><td>Collider (state space)</td><td> $9 . 3 { \cdot } 1 0 ^ { - 6 } \ddagger$ </td><td> $3 . 7 { \cdot } 1 0 ^ { - 4 } \ddagger$ </td><td> $0 . 0 3 3 8 7 \ddagger$ </td><td>39.74</td><td>3656‡</td></tr><tr><td>Collider (B-PINN)</td><td> $0 . 2 3 9 1 \pm 0 . 0 5 0$ </td><td> $0 . 5 6 6 3 \pm 0 . 1 3$ </td><td> $1 . 8 3 2 \pm 0 . 2 5$ </td><td> $2 . 4 6 2 \pm 0 . 7 3$ </td><td> $7 . 9 3 5 \pm 1 . 7$ </td></tr><tr><td rowspan="3">acceleration residual</td><td>Chain (state space)</td><td> $1 . 0 3 5 \pm 0 . 2 2$ </td><td> $1 . 2 1 9 \pm 0 . 2 8$ </td><td> $4 . 0 3 0 \pm 0 . 7 9$ </td><td> $1 . 1 9 1 \pm 0 . 2 0$ </td><td> $3 . 9 2 7 \pm 0 . 4 1$ </td></tr><tr><td>Collider (state space)</td><td> $1 . 0 3 5 \pm 0 . 2 2$ </td><td> $1 . 2 1 9 \pm 0 . 2 8$ </td><td> $4 . 0 3 0 \pm 0 . 7 9$ </td><td> $1 . 1 9 1 \pm 0 . 2 0$ </td><td> $3 . 9 2 7 \pm 0 . 4 1$ </td></tr><tr><td>Collider (B-PINN)</td><td> $1 . 3 1 4 \pm 0 . 2 8$ </td><td> $0 . 9 4 1 9 \pm 0 . 2 4$ </td><td> $4 . 1 1 5 \pm 0 . 8 1$ </td><td> $0 . 7 3 1 1 \pm 0 . 1 8$ </td><td> $3 . 1 6 2 \pm 0 . 3 5$ </td></tr></table>

Table 5: Double pendulum. The three inference legs under both residual metrics, on one dataset and one prior, with $\sigma _ { r }$ held at its true value. Posterior mean ± standard deviation over 14000 draws (state space) and 24000 draws (B-PINN). †: a walk at the Euler map’s stability boundary, not a posterior; its median is quoted; ‡: chains that have not merged $( \hat { R } > 1 . 0 5 )$ , likewise a median rather than a posterior.
<table><tr><td>residual</td><td>leg</td><td> $m _ { 1 } ~ ( 1 . 0 0 )$ </td><td>m2 (0.500)</td><td>l1 (1.00)</td><td> $\ell _ { 2 } ~ ( 0 . 8 0 0 )$ </td><td>m1+m2 (1.50)</td><td>m2/m1 (0.500)</td></tr><tr><td>force residual</td><td>Chain (state space)</td><td>1.078 ± 0.26</td><td>0.5367 ± 0.13</td><td>1.011 ± 0.011</td><td>0.7890 ± 0.010</td><td>1.615 ± 0.39</td><td>0.4976 ± 0.012</td></tr><tr><td></td><td>Collider (state space)</td><td>0.1713</td><td>0.02538‡</td><td>1.386</td><td>0.07589</td><td>0.1967</td><td>0.1481‡</td></tr><tr><td></td><td>Collider (B-PINN)</td><td>0.08769 ± 0.020</td><td> $0 . 0 2 2 6 9 \pm 0 . 0 0 5 \dot { 4 }$ </td><td> $0 . 1 4 4 1 \pm 0 . 0 1 \dot { 9 }$ </td><td>0.2137 ± 0.034</td><td>0.1104 ± 0.022</td><td> $0 . 2 7 0 9 \pm 0 . 0 8 \dot { 5 }$ </td></tr><tr><td>acceleration residual</td><td>Chain (state space)</td><td> $1 . 0 2 1 \pm 0 . 3 0$ </td><td> $0 . 5 0 6 2 \pm 0 . 1 5$ </td><td> $1 . 0 1 1 \pm 0 . 0 1 2$ </td><td>0.7926 ± 0.011</td><td> $1 . 5 2 7 \pm 0 . 4 5$ </td><td>0.4958 ± 0.011</td></tr><tr><td></td><td>Collider (state space)</td><td> $1 . 0 2 1 \pm 0 . 3 0$ </td><td> $0 . 5 0 6 2 \pm 0 . 1 5$ </td><td> $1 . 0 1 1 \pm 0 . 0 1 2$ </td><td>0.7926 ± 0.011</td><td> $1 . 5 2 7 \pm 0 . 4 5$ </td><td>0.4958 ± 0.011</td></tr><tr><td></td><td>Collider (B-PINN)</td><td>1.004‡</td><td>0.4715‡</td><td>1.023‡</td><td>0.7808</td><td>1.476‡</td><td>0.4699‡</td></tr></table>

## A.3.3 Full Posterior Results

Tables 4–6 report the posterior of every parameter for every inference variant, including the parameters and parameter combinations that Table 1 omits. They show the same pattern as Table 1. The chain model keeps the drift-invariant scale near the truth and recovers every identified quantity to within 1.3 posterior standard deviations. The errors of the collider, in contrast, concentrate where Eq. (16) predicts: under force residuals on the parameters that enter the residual metric, and on the parameters that enter the dissipation term (the damping of the oscillator, the light speed of the relativistic particle).

## A.3.4 Replication over Datasets

All runs above use a single dataset per system. On one dataset, a posterior can lie away from the truth by chance; a bias means that it does so systematically, on average over datasets. To distinguish the two, we repeat the oscillator configuration of Fig. 2 (acceleration residuals) on independent datasets, each with a new SDE path and new observation noise. We do this in two complementary ways.

Sampled posteriors. We rerun the B-PINN (NUTS) and our chain model (state space, PMMH) with the settings of Appendix A.3.1 on ten datasets, including the one of Fig. 2. All B-PINN runs reach $\hat { R } \leq \mathrm { i } . 0 0 1$

Exact posteriors. To remove sampling error and convergence issues altogether, we use ten further datasets and replace the neural network by a surrogate on which all posteriors can be computed exactly. The collider is a B-PINN on a linear cosine basis (65 coeficients with prior standard deviation 0.3, N d = 128 collocation residuals). Because the residual is linear in the coeficients, they can be integrated out in closed form, and $Z _ { \lambda }$ can be restored exactly on the same basis, which gives a chain model on the surrogate. The exact Kalman likelihood of the Euler–Maruyama latent gives the state-space chain. We evaluate all three posteriors on a grid over the identified coordinates $( \log ( c / m ) , \log ( k / m ) )$ , with the log-normal prior of Appendix A.3.1; less than $1 0 ^ { - 3 }$ of each posterior’s mass lies at the boundary of the grid.

Table 6: Relativistic particle. The three inference legs under both residual metrics, on one dataset and one prior, with $\sigma _ { r }$ held at its true value. Posterior mean ± standard deviation over 14000 draws (state space) and 24000 draws (B-PINN). †: a walk at the Euler map’s stability boundary, not a posterior; its median is quoted; ‡: chains that have not merged $( \hat { R } > 1 . 0 5 )$ , likewise a median rather than a posterior.
<table><tr><td>residual</td><td>leg</td><td> $m _ { 0 } \ ( 1 . 0 0 )$ </td><td>q (1.00)</td><td> $c \ ( 1 . 0 0 )$ </td><td> $q / m _ { 0 } \ ( 1 . 0 0 )$ </td></tr><tr><td>force residual</td><td>Chain (state space)</td><td> $0 . 8 2 3 3 \pm 0 . 1 5$ </td><td> $0 . 8 6 8 1 \pm 0 . 1 5$ </td><td> $0 . 8 9 3 9 \pm 0 . 1 2$ </td><td> $1 . 0 5 7 \pm 0 . 0 5 5$ </td></tr><tr><td></td><td>Collider (state space)</td><td> $2 . 1 { \cdot } 1 0 ^ { - 5 } \ddagger$ </td><td> $0 . 0 0 3 8 3 \ddagger$ </td><td>4133</td><td>182.8</td></tr><tr><td></td><td>Collider (B-PINN)</td><td> $0 . 1 6 3 0 \pm 0 . 0 3 6$ </td><td> $0 . 1 4 8 5 \pm 0 . 0 3 2$ </td><td> $1 . 6 0 8 \pm 0 . 2 7$ </td><td> $0 . 9 1 2 7 \pm 0 . 0 3 6$ </td></tr><tr><td>acceleration residual</td><td>Chain (state space)</td><td> $0 . 9 9 4 1 \pm 0 . 2 9$ </td><td> $1 . 0 4 9 \pm 0 . 3 1$ </td><td> $0 . 8 8 8 1 \pm 0 . 0 8 7$ </td><td> $1 . 0 5 6 \pm 0 . 0 4 6$ </td></tr><tr><td></td><td>Collider (state space)</td><td> $0 . 9 9 4 1 \pm 0 . 2 9$ </td><td> $1 . 0 4 9 \pm 0 . 3 1$ </td><td> $0 . 8 8 8 1 \pm 0 . 0 8 7$ </td><td> $1 . 0 5 6 \pm 0 . 0 4 6$ </td></tr><tr><td></td><td>Collider (B-PINN)</td><td> $1 . 1 0 1 \pm 0 . 3 3$ </td><td> $0 . 9 8 7 2 \pm 0 . 2 9$ </td><td> $1 . 7 8 3 \pm 0 . 2 5$ </td><td> $0 . 8 9 6 9 \pm 0 . 0 1 5$ </td></tr></table>

Table 7: Replication of the oscillator configuration over datasets (acceleration residuals). Top: the neural B-PINN and our chain model, sampled as in the main text, on ten datasets. Bottom: exactly computed posteriors on a linear surrogate, on ten further datasets. displ.: mean over datasets of the posterior mean of the log parameter minus the log truth, ± its standard error. sd: mean posterior standard deviation. 90% cov.: number of datasets whose central 90% credible interval contains the truth.
<table><tr><td></td><td colspan="3"> $\log ( c / m )$ </td><td colspan="3"> $\log ( k / m )$ </td></tr><tr><td>Model (latent)</td><td>displ.</td><td></td><td>sd 90% cov.</td><td>displ.</td><td></td><td>sd 90% cov.</td></tr><tr><td colspan="7">Sampled posteriors, ten datasets</td></tr><tr><td>Chain (state space)</td><td> $+ 0 . 0 2 \pm 0 . 0 5$ </td><td>0.17</td><td>10/10</td><td> $+ 0 . 0 2 \pm 0 . 0 2$ </td><td>0.11</td><td>10/10</td></tr><tr><td>Collider B-PINN (MLP)</td><td> $\mathbf { - 0 . 4 2 \pm 0 . 0 3 }$ </td><td>0.24</td><td></td><td> $\mathbf { 4 } / \mathbf { 1 0 } - \mathbf { 0 . 1 2 } \pm \mathbf { 0 . 0 3 }$ </td><td>0.11</td><td>7/10</td></tr><tr><td colspan="7">Exact posteriors on a linear surrogate, ten datasets</td></tr><tr><td>Chain (state space)</td><td> $- 0 . 0 3 \pm 0 . 0 5 0 . 1 7$ </td><td></td><td></td><td> $8 / 1 0 + 0 . 0 2 \pm 0 . 0 4$ </td><td>0.11</td><td>9/10</td></tr><tr><td>Chain (linear surrogate)</td><td> $- 0 . 0 5 \pm 0 . 0 5 0 . 1 8$ </td><td></td><td></td><td> $9 / 1 0 + 0 . 0 2 \pm 0 . 0 3$ </td><td>0.09</td><td>9/10</td></tr><tr><td>Collider (linear surrogate)</td><td> $\mathbf { - 0 . 4 6 \pm 0 . 0 4 } 0 . 2 3$ </td><td></td><td></td><td> $\mathbf { 1 / 1 0 } \mathbf { - 0 . 0 8 \pm 0 . 0 3 }$ </td><td>0.10</td><td>7/10</td></tr></table>

Results. Table 7 confirms that the displacements reported in the main text are a systematic bias. Over ten datasets, the B-PINN underestimates $c / m$ by $0 . 4 2 \pm 0 . 0 3$ log units and $k / m$ by $0 . 1 2 \pm 0 . 0 3$ , and it lies below the truth on every dataset in both coordinates. Measured relative to the chain model on the same dataset, which removes the variation between datasets, the displacements are $- 0 . 4 5 \pm 0 . 0 2$ and $- 0 . 1 4 \pm 0 . 0 2$ . The B-PINN posterior is not narrower than the chain’s (mean standard deviation 0.24 against 0.17 in $\log ( c / m ) )$ , but it is displaced by almost two of its own standard deviations, so its 90% credible intervals contain the truth in only 4 and 7 of the ten datasets, whereas those of the chain model contain it in all ten. The dataset of Fig. 2 lies at the upper end: its displacement in log $\cdot ( c / m )$ (−0.53) is the third largest of the ten, against a median of −0.39.

The exact posteriors on the linear surrogate show the same picture, with displacements of $- 0 . 4 6 \pm 0 . 0 4$ and $- 0 . 0 8 \pm 0 . 0 3$ and coverage in 1 and 7 of ten datasets. Restoring $Z _ { \lambda }$ exactly on the same surrogate removes the bias in both coordinates and restores nominal coverage, as does the state-space chain. The bias is therefore reproducible across datasets, and it is caused by the missing normalizer rather than by the particular dataset or by the limited expressiveness of the surrogate.

## A.3.5 Sensitivity to the Residual Scale

All experiments above fix $\sigma _ { r }$ at the value that generated the data. To test how the bias depends on this value, we repeat the exact-posterior comparison of Appendix A.3.4 on the same ten oscillator datasets (acceleration residuals), varying $\sigma _ { r }$ in the data and in the models. For $\sigma _ { r } \leq 0 . 1 $ , we evaluate the posteriors on a finer grid, since they are narrower than its original spacing.

Table 8: Sensitivity to the residual scale (oscillator, acceleration residuals, exact posteriors, ten datasets). Displacement of $\log ( c / m )$ from the truth (mean over datasets ± standard error) and number of datasets whose central 90% credible interval contains the truth. Top: $\sigma _ { r }$ matched to the data. Bottom: $\sigma _ { r }$ misspecified.
<table><tr><td colspan="2"> $\sigma _ { r }$ </td><td colspan="2">Chain (state space)</td><td colspan="2">Chain (linear surrogate)</td><td colspan="2">Collider (linear surrogate)</td></tr><tr><td></td><td>data model</td><td></td><td>displ. 90% cov.</td><td></td><td>displ. 90% cov.</td><td>displ.</td><td>90% cov.</td></tr><tr><td>0.05</td><td>0.05</td><td> $- 0 . 0 1 \pm 0 . 0 2$ </td><td>9/10</td><td> $- 0 . 0 3 \pm 0 . 0 2$ </td><td></td><td> $\mathbf { 9 } / 1 0 - \mathbf { 0 . 0 7 } \pm \mathbf { 0 . 0 2 }$ </td><td>7/10</td></tr><tr><td>0.1</td><td>0.1</td><td> $- 0 . 0 1 \pm 0 . 0 3$ </td><td>9/10</td><td> $- 0 . 0 2 \pm 0 . 0 3$ </td><td></td><td> $\begin{array} { l l } { \dot { 9 / 1 0 } } & { - \mathbf { 0 . 1 2 } \pm \mathbf { 0 . 0 3 } } \end{array}$ </td><td>7/10</td></tr><tr><td>0.3</td><td>0.3</td><td> $- 0 . 0 3 \pm 0 . 0 5$ </td><td>8/10</td><td> $- 0 . 0 5 \pm 0 . 0 5$ </td><td></td><td> $\dot { 9 / 1 0 } - \mathbf { 0 . 4 6 } \pm \mathbf { 0 . 0 4 }$ </td><td>1/10</td></tr><tr><td>1.0</td><td>1.0</td><td> $- 0 . 0 4 \pm 0 . 0 7$ </td><td>8/10</td><td> $- 0 . 1 6 \pm 0 . 0 7$ </td><td></td><td> $\begin{array} { r l } { \dot { 9 / 1 0 } } & { { } - \mathbf { 0 . 7 4 } \pm \mathbf { 0 . 0 3 } } \end{array}$ </td><td>0/10</td></tr><tr><td>0</td><td>0.3</td><td> $\mathbf { + 0 . 4 0 \pm 0 . 0 1 }$ </td><td></td><td> $\mathbf { 0 } / 1 \mathbf { 0 } + \mathbf { 0 } . \mathbf { 3 7 } \pm \mathbf { 0 . 0 1 }$ </td><td>1/10</td><td> $- 0 . 1 5 \pm 0 . 0 1$ </td><td>10/10</td></tr><tr><td>0.3</td><td>0.9</td><td> $\mathbf { + 0 . 9 7 \pm 0 . 0 5 }$ </td><td></td><td> $\mathbf { 0 } / 1 \mathbf { 0 } + \mathbf { 0 } . \mathbf { 9 6 } \pm \mathbf { 0 . 0 5 }$ </td><td>0/10</td><td> $- 0 . 3 0 \pm 0 . 0 2$ </td><td>10/10</td></tr></table>

With $\sigma _ { r }$ matched to the data, the collider bias is present at every scale and grows with it, from −0.07 to −0.74 in $\log ( c / m )$ , while both chain models remain close to the truth (Table 8, top). At small $\sigma _ { r } .$ the displacement is small in absolute terms but still about one posterior standard deviation, and coverage remains below nominal.

A misspecified residual scale introduces a diferent bias, which afects the chain model as well. For noise-free data, or for a residual scale three times too large, the chain overestimates the damping, and its intervals miss the truth (Table 8, bottom). The downward collider bias then partly ofsets this error, so that the collider’s wider intervals cover the truth; this reflects two biases of opposite sign cancelling, not a correct model.

Ideally, the residual scale is therefore matched to the data. When it is unknown, a natural remedy is to infer $\sigma _ { r }$ jointly with $\lambda ,$ as noted in Section 6. For the chain model, this is a direct extension of PMMH, since the per-step normalizers (21) remain analytic in $\sigma _ { r }$ . For the collider, however, the bias $\Delta$ then also depends on $\sigma _ { r } ,$ which gives rise to a further bias mechanism; we leave its analysis to future work.

## A.3.6 The Dissipation Term on the Conservative Systems

The double pendulum and the relativistic particle are both conservative, yet the dissipation term of a global surrogate need not vanish for them. We first explain why, and then evaluate the term for each system.

Why conservative systems can retain a dissipation term. Suppose that the forces derive from a (possibly time-dependent) potential and that $M _ { \lambda } = \partial ^ { 2 } \mathcal { L } / \bar { \partial } v \partial v ^ { \top }$ is the velocity Hessian of the Lagrangian L. Then the flow is Hamiltonian in the canonical coordinates $( x , p )$ with $p = \partial \mathcal { L } / \partial v _ { ; }$ , and divergence-free in these coordinates (Liouville’s theorem). Since the change of variables $( x , v ) \mapsto ( x , p )$ has Jacobian determinant det $M _ { \lambda }$

$$
\nabla \cdot f _ { \lambda } ( z ( t ) , t ) = - { \frac { \mathrm { d } } { \mathrm { d } t } } \log \operatorname* { d e t } M _ { \lambda } ( z ( t ) ) , \qquad \int _ { 0 } ^ { T } \nabla \cdot f _ { \lambda } \mathrm { d } t = \log { \frac { \operatorname* { d e t } M _ { \lambda } ( z ( 0 ) ) } { \operatorname* { d e t } M _ { \lambda } ( z ( T ) ) } } ,\tag{38}
$$

a boundary term that does not grow with $\tau .$ On a global surrogate, however, $\Delta ( \lambda )$ depends on $\textstyle \sum _ { k } | \mathrm { R e } s _ { k } |$ (Eq. (34)) rather than on the divergence, and the two can difer in two ways. First, the eigenvalues of a Hamiltonian linearization come in pairs ±s: oscillatory pairs $( \operatorname { R e } s = 0 )$ contribute nothing, but hyperbolic pairs (real ±s) contribute $\pi | _ { s } |$ although their divergence vanishes, so a chaotic system can retain a λ-dependent term that grows with $\tau .$ . Second, if $D f _ { \lambda }$ has a single eigenvalue with nonzero real part, as for a one-dimensional system with velocity-dependent inertia, then $\begin{array} { r } { \sum _ { k } | \mathrm { R e } s _ { k } | = | \boldsymbol { \nabla \cdot } \boldsymbol { f } _ { \lambda } | } \end{array}$ pointwise, and the windowed dissipation term becomes $- { \textstyle \frac { 1 } { 2 } } \int _ { 0 } ^ { \mathcal { T } } { | \nabla \cdot f _ { \lambda } | } \mathrm { d } t$ . This grows linearly in $\tau$ whenever the divergence changes sign along the orbit, although $\int _ { 0 } ^ { \mathcal { T } } \nabla \cdot f _ { \lambda }$ dt stays bounded. The pendulum is an instance of the first case and the relativistic particle of the second. We evaluate both quantities along the noise-free trajectory at the ground truth; this explains why the dissipation term leaves the pendulum results in Table 1 unafected but biases the relativistic particle.

Double pendulum. The time average of $\textstyle \sum _ { k } | \mathrm { R e } s _ { k } |$ over [0, T] is 0.508, so the surrogate retains a term of $\begin{array} { r } { \frac { \mathcal { T } } { 2 } \langle \sum _ { k } | \mathrm { R e } s _ { k } | \rangle = 3 . 0 5 } \end{array}$ nats at $\tau = 1 2 .$ , although the mean divergence is only $1 . 2 \cdot 1 0 ^ { - 2 }$ . This term does not afect the shift log α along the drift-invariant direction reported in Table 1: scaling $( m _ { 1 } , m _ { 2 } )  \alpha ( m _ { 1 } , m _ { 2 } )$ leaves the acceleration $a _ { \lambda } = M _ { \lambda } ^ { - 1 } F _ { \lambda }$ , and hence $D f _ { \lambda }$ , unchanged, and indeed the time average agrees to six digits at $\alpha = 0 . 5 ,$ , 1 and 2. Of the direction, it is negligible as well: it varies by less than $2 \cdot 1 0 ^ { - 3 }$ over ±0.05 in log $\ell _ { 1 }$ , $\mathrm { i . e . , ~ } | \partial \Delta / \partial \log \ell _ { 1 } | \approx 0 . 0 4$ nats per unit of log $\ell _ { 1 }$ , which is small against a prior with log-standard deviation 0.25.

Relativistic particle. The drive returns the particle to rest at both ends of the record $( \gamma ( 0 ) = \gamma ( \mathcal { T } ) = 1 )$ so $\begin{array} { r } { \int _ { 0 } ^ { \mathcal { T } } \nabla \cdot f _ { \lambda } \mathrm { d } t = 0 } \end{array}$ to machine precision: the divergence term is not merely bounded but absent. The drift Jacobian $D f _ { \lambda } = \left[ { 0 \atop 0 } \ : \bigtriangledown \cdot f _ { \lambda } \right]$ , however, has the eigenvalues 0 and $\nabla \cdot f _ { \lambda } = - 3 q E ( t ) v / ( m _ { 0 } c ^ { 2 } \gamma )$ , which changes sign twice per drive period. Hence $\begin{array} { r } { \sum _ { k } | \mathrm { R e } s _ { k } | = | \nabla \cdot f _ { \lambda } | , } \end{array}$ , and $\begin{array} { r } { \int _ { 0 } ^ { \mathcal T } \left| \nabla \cdot f _ { \lambda } \right| \mathrm { d } t = 1 0 . 3 6 } \end{array}$ , so that $\Delta _ { \mathrm { d i s s } } = - 5 . 1 8$ nats at the truth, a term that grows linearly in $\tau$ . Diferentiating with respect to the light speed at a fixed trajectory gives $\partial \Delta _ { \mathrm { d i s s } } / \partial \log c = + 9 . 5$ nats per unit of log c. The collider is thus rewarded for a larger c, because a larger c keeps γ closer to one and thereby flattens the divergence. This reward has the sign and the order of magnitude of the observed error, and it is too strong for a prior with log-standard deviation 0.25 to counteract. The B-PINN reports $c = 1 . 7 8$ (log c displaced by $+ 0 . 5 6 9 )$ , against 0.89 for the chain. We do not derive a quantitative prediction from these numbers, because $\Delta _ { \mathrm { d i s s } }$ is strongly nonlinear in $c \ \mathrm { ( r o u g h l y \propto } - c ^ { - 2 } \mathrm { ) }$ over the range that the posterior covers.

## A.3.7 Restoring the Normalizer on the Neural Surrogate

In principle, the bias could stem from the probabilistic model (collider or chain), the latent (state space or neural surrogate), or the estimator (PMMH or NUTS). On the state-space latent, the model alone decides: chain and collider share latent and estimator and coincide exactly whenever $\Delta ( \lambda )$ is constant (Section 5.2), and NUTS reproduces the PMMH results (Appendix A.3.2). Here, we hold the neural surrogate fixed and change only the model.

Construction. Chain (Fourier MLP) adds the term $- \Delta =$ log $Z _ { r } - \log Z _ { \lambda }$ to the log-density of the oscillator B-PINN. By Eq. (13), this turns the collider into the chain, up to the approximations of Appendix A.1. For the oscillator, Eq. (16) at $d = 1$ , with det $A _ { \lambda } = m$ under force residuals and det $A _ { \lambda } = 1$ under acceleration residuals, gives

$$
\log Z _ { r } - \log Z _ { \lambda } = \underbrace { \frac { c \mathcal { T } } { 2 m } } _ { \mathrm { d i s s i p a t i o n } } + \underbrace { n _ { \mathrm { e f f } } \log m } _ { \mathrm { m e t r i c , ~ f o r c e ~ r e s i d u a l s ~ o n l y } } + \mathrm { c o n s t . }\tag{39}
$$

Everything else (architecture, weight prior, data, collocation grid, sampler and seed) is identical to the B-PINN.

Efective number of constraints. The correction requires $n _ { \mathrm { e f f } } = \mathrm { t r } [ G _ { \lambda } ( G _ { \lambda } + P ) ^ { - 1 } ]$ . We compute it once, from the Gauss–Newton curvature at $\lambda = \lambda _ { \mathrm { t r u e } }$ , and keep it fixed during sampling. This is the idealization under which the closed forms are derived (assumption (iii) in Appendix A.1): by Eq. (30), $n _ { \mathrm { e f f } }$ depends on λ only through $\kappa _ { \lambda }$ and is constant in the saturated regime. We evaluate the curvature at the final network weights of the corresponding B-PINN run rather than at the data-free mode $\theta ^ { * } = 0$ , where $x _ { \theta } \equiv 0$ solves the ODE but the network Jacobian is degenerate (assumption (ii)). With $N d = 1 2 8$ and dim $\theta = 1 4 4 1$ , this gives $n _ { \mathrm { e f f } } = 1 1 0 . 2$ for force residuals. For acceleration residuals, only the dissipation term is active and no $n _ { \mathrm { e f f } }$ is needed.

This value depends only moderately on λ but strongly on the network weights. Evaluating the curvature at the chain’s posterior mean instead of $\lambda _ { \mathrm { t r u e } }$ changes it by less than 0.5%, and over draws from the chain posterior it varies by about ±10% (98–116 around 110.2). At the final state of the acceleration-residual B-PINN, by contrast, it is 99.9, and at a MAP fit of the network it drops to 44. Moreover, $n _ { \mathrm { e f f } }$ is not constant along the B-PINN’s own posterior (55–78, where the B-PINN has collapsed the mass scale), so the saturated regime of assumption (iii) holds only approximately. This sensitivity to the expansion point is the main source of the overcorrection discussed below.

Results. Table 9 reports two configurations, each of which activates a diferent subset of the terms in (39). All runs converge $( \hat { R } = 1 . 0 0 0$ over 24,000 draws), so these are valid posteriors. How much of the bias the correction removes follows how accurate the closed forms are in each configuration:

• With acceleration residuals, only the dissipation term is active and no $n _ { \mathrm { e f f } }$ enters. The correction removes 97.5% of the displacement of log(c/m).

Table 9: Restoring the normalizer on the neural surrogate (oscillator). Displacements from the truth in log space (posterior mean of the log parameter minus the log truth) for the Collider B-PINN and for Chain (Fourier MLP), and the fraction of the displacement removed by the correction, 1−|chain|/|collider|. Active terms are the terms of Eq. (39) that depend on the inferred parameters in that configuration. The last configuration is the $\tau = 2 4$ run of the horizon study in Fig. 3.
<table><tr><td>Configuration</td><td>Active terms</td><td>Direction</td><td>Collider</td><td>Chain (MLP)</td><td>Removed</td></tr><tr><td>force residuals</td><td>metric, dissipation</td><td>log m</td><td>-1.453</td><td>+0.497</td><td>65.8%</td></tr><tr><td>acceleration residuals</td><td>dissipation</td><td> $\log ( c / m )$ </td><td>-0.526</td><td>-0.013</td><td>97.5%</td></tr></table>

![](images/1e14c524c6d4a5cbf5cc953c357ccdcc7124cfd1eaf234cb51143a618d518a5a.jpg)  
Figure 4: Exact closed form of the residual metric bias against measurements on the oscillator (Eq. (22)). Closed form in grey, the collider on the Markovian state-space latent in dark red, the chain in blue. On the state-space latent, $\Delta ( \lambda )$ along the drift-invariant direction matches −T log α to within 2.8 · 10<sup>−12</sup> nats for each number of time steps shown $( T = 3 1 2$ , 1,248, 2,496; $T = 6 2 4$ , omitted for clarity, agrees equally), whereas the chain’s profile stays flat.

• With force residuals, the approximate $n _ { \mathrm { e f f } }$ matters, and the correction can overshoot. The collider’s log $m =$ −1.45 becomes +0.50 (m ≈ 1.64 against a truth of 1.00): the correction crosses the truth and leaves a displacement of about one third of the original on the other side. A term linear in log m with a fixed coeficient $n _ { \mathrm { e f f } }$ replaces a function of $\kappa _ { \lambda }$ and can therefore overcorrect.

As cautioned at the end of Appendix A.1, the closed forms thus locate the bias and remove most of it, but they do not calibrate it exactly.

Remaining stifness bias. The correction also leaves part of the stifness bias: with acceleration residuals, Chain (Fourier MLP) reports $k / m = 3 . 4 8$ , against 3.16 for the B-PINN and 3.93 for the chain on the state-space latent (truth 4.00). This remainder is still collider bias rather than a representation error of the surrogate. On the linear cosine surrogate of Appendix A.3.4, where $Z _ { \lambda }$ can be restored exactly, the collider displaces $\log ( k / m )$ from the truth by $- 0 . 0 8 \pm 0 . 0 3$ (mean ± standard error over ten datasets), whereas the exactly restored chain $\left( + 0 . 0 2 \pm 0 . 0 3 \right)$ and the state-space chain $\left( + 0 . 0 2 \pm 0 . 0 4 \right)$ are unbiased. The leading-order correction misses this part because k enters log $Z _ { \lambda }$ only through the $\mathcal { O } ( \beta / \Omega )$ remainder of Eq. (33).

## A.3.8 Details on the Scaling-Law Experiments

Figs. 3 and 4 use the oscillator configuration of Appendix A.3.1, in each case with the mechanism that is not under test switched of. Fig. 4 checks the metric term of $\Delta ( \lambda )$ in a setting where its closed form is exact; Fig. 3 tests the approximate dissipation term on the neural surrogate.

Residual metric bias (Fig. 4). The figure shows $\Delta ( \lambda )$ along the drift-invariant direction $( m , c , k ) \ $ $\alpha ( m , c , k )$ on the state-space latent, for force residuals; the Markovian latent switches of the dissipation term. We use four discretizations, $T \in \{ 3 1 2 , 6 2 4 , 1 2 4 8 , 2 4 9 6 \}$ . The measured $\Delta ( \lambda )$ matches $- T \log \alpha$ to within $2 . 8 \cdot 1 0 ^ { - 1 2 }$ nats, i.e., to machine precision, as expected since Eq. (22) is exact on this latent. Over the swept range of $\alpha , \Delta ( \lambda )$ varies by 225, 449, 899 and 1797 nats, doubling with the number of steps, whereas the chain’s log-posterior varies by only 6.0 nats over the same range.

Dissipation bias (Fig. 3). Unlike the metric term, the dissipation term is not a property of the latent alone, so we measure its efect on the posterior rather than $\Delta ( \lambda )$ itself. We use acceleration residuals, which switch of the metric bias, and vary the horizon over $\mathcal { T } \in \{ 1 2 , 2 4 , 4 8 \}$ ; the run at $\tau = 2 4$ is the standard oscillator configuration. Across horizons, we keep the surrogate’s bandwidth per unit time and the Euler step fixed and scale K, N and T with $\tau ,$ so that only the amount of data changes. The panel shows the posterior mean of log $\cdot ( c / m )$ minus its true value for Chain (state space), Chain (Fourier MLP) and the Collider B-PINN, each with a bar of $\pm 1$ posterior standard deviation. The state-space collider is not plotted separately, since under acceleration residuals $\Delta ( \lambda )$ is constant and its draws are identical to the chain’s.

The B-PINN lies at −0.377, −0.526 and −0.669, i.e., at 1.55, 2.08 and 2.67 of its own posterior standard deviations, which barely change (0.243, 0.253, 0.251). The chain stays near the truth, both on the state-space latent $( - 0 . 0 8 6 , \ - 0 . 0 2 2 , \ + 0 . 0 4 1$ , with standard deviations 0.198, 0.176, 0.143) and on the surrogate with the correction $\left( - 0 . 0 9 5 , \ - 0 . 0 1 3 , \ + 0 . 1 4 1 \right)$ . The chain’s posterior thus contracts as the record grows, whereas the B-PINN’s does not, so the B-PINN’s error grows relative to its own stated uncertainty.

To compare with the closed form (36), we reweight the chain posterior of $\log ( c / m )$ , approximated as Gaussian, by $e ^ { \Delta _ { \mathrm { d i s s } } } = e ^ { - ( c / m ) T / 2 }$ . This predicts shifts relative to the chain of −0.21, −0.32 and −0.41, whereas the B-PINN is shifted by $- 0 . 2 9 , \ - 0 . 5 0$ and −0.71. The leading-order term thus predicts the direction of the bias and its growth with the record, but only between three quarters and three fifths of its size; it also predicts a posterior that contracts with $\tau _ { \ast }$ , which the B-PINN’s does not.

## A.4 Comparison with Nested Inference

To our knowledge, the only prior work that adopts the chain model, by Alberts, Hao, and Bilionis $[ 1 , 1 0 , 1 1 ]$ handles the unknown normalizer $Z _ { \lambda }$ by nested inference: an inner loop estimates the gradient $\partial _ { \lambda }$ log $Z _ { \lambda }$ anew at every step of an outer loop over the joint state. We compare our method, the particle filter with PMMH (Section 4.3), against the two nested schemes that target the same chain posterior.

• NSVI [10] (nested stochastic variational inference) fits an outer Gaussian guide over path coeficients, physical parameters and initial state by maximizing an ELBO. The normalizer gradient in this ELBO is supplied by a second, persistent inner guide, which is re-fitted at every outer step.

• NPSGLD [11] (nested preconditioned stochastic-gradient Langevin dynamics) replaces both guides by preconditioned Langevin chains: a persistent auxiliary chain over the path supplies the normalizer gradient to a main chain over the joint state.

We use the public JAX reimplementation of both methods by Bolt [5] (commit dab4403), which is not by the original authors; we run it unmodified and supply only the problem definition and the evaluation. To isolate the inference algorithm, all methods use the Euler–Maruyama path as latent, on which $Z _ { \lambda }$ is in fact available in closed form. The comparison therefore measures how expensive and how accurate nested estimation of $Z _ { \lambda }$ is where it could be avoided. The baselines were designed for latents without a closed-form normalizer, a setting we do not test.

Setup. We use two systems in the configurations of Section 5.1: the damped oscillator under both residual metrics, and the double pendulum under torque residuals. All methods share the model, the latent $( T = 1 2 4 8$ Euler–Maruyama steps for the oscillator and $T = 1 2 3 5$ for the pendulum), the observations, and the log-normal priors with median at the truth. As in all experiments, $\sigma _ { r }$ is fixed at its true value. Instead of tuning a single run per baseline, we sweep their hyperparameters. For NSVI, we vary the outer learning rate in $[ 5 \cdot 1 0 ^ { - 5 } , 1 0 ^ { - 3 } ]$ and the number of inner iterations from 8 to 256, with budgets of up to $1 . 2 \cdot 1 0 ^ { 6 }$ iterations. For NPSGLD, we vary the step size in $[ 1 0 ^ { - 8 } , 1 0 ^ { - 3 } ]$ , use 4 or 16 chains and both of its preconditioners (rmsprop and dense fisher), and switch step-size annealing on or of, with budgets of up to $1 0 ^ { 7 }$ iterations. This amounts to 23 runs on the oscillator and 21 on the pendulum. All runs, including those of our method, use the same Quadro RTX 8000 GPU, so that run times are directly comparable.

Evaluation metrics. Wall-clock time alone favours samplers that have not mixed, since they produce many draws that carry little information. We therefore measure cost in seconds per efective sample, i.e., wall-clock time divided by the efective sample size (ESS), and report the split-R<sup>ˆ</sup> alongside it. As in Table 1, runs with $\hat { R } > 1 . 0 5$ do not yield a valid posterior; we report their median and mark them with a dagger. For our method, the ESS is the smallest over the components of $\lambda ;$ for NPSGLD, it is the ESS of the drift-invariant scale. This choice favours the baseline: on the drift-invariant scale alone, the ESS of our method on the pendulum is about 870 rather than 343, which would lower its cost from 6.71 to 2.6 s per efective sample. The cost and $\hat { R }$ of our method refer to a single PMMH chain. To check its mixing, we additionally ran four independent chains (sampler seeds 0–3) per configuration; they give $\mathrm { s p l i t } { - } \hat { R } \leq 1 . 0 0 1$ for all parameters, and their posterior means of the drift-invariant scale agree to within 0.02 (oscillator) and 0.003 (pendulum). NSVI draws independent samples from a fitted guide, so $\hat { R }$ and ESS do not apply to it; we assess it by its bias instead. Finally, the width of a run is the ratio of its posterior standard deviation to that of our method. It measures what matters for uncertainty quantification: a method can locate the posterior correctly and still misrepresent its uncertainty.

Table 10: Nested inference compared with our particle filter with PMMH. Ranges are over the hyperparameter sweep of each method. Location is the drift-invariant scale (m for the oscillator, $m _ { 1 } { + } m _ { 2 }$ for the pendulum), reported as posterior mean ± standard deviation where $\hat { R } \leq 1 . 0 5$ and as a daggered median otherwise. Width is the posterior standard deviation divided by that of our method; 1.00 is correct, and 0.03 means thirty-fold overconfidence. Cost is in seconds per efective sample. A run counts as diverged if its location is of by more than a factor of 100 or its spread exceeds 100 reference standard deviations.
<table><tr><td>System</td><td>Method</td><td>Runs</td><td>Location</td><td>R</td><td>ESS</td><td>Width</td><td>Cost</td></tr><tr><td rowspan="3">Oscillator force residuals</td><td>PF + PMMH (ours)</td><td>1</td><td>1.015 ± 0.140</td><td>1.00</td><td>604</td><td>1.00</td><td>0.81</td></tr><tr><td>NSVI</td><td>8</td><td>0.78-1.43</td><td></td><td></td><td>0.15-0.28</td><td></td></tr><tr><td>NPSGLD</td><td>10</td><td>1.00–1.12↑</td><td>1.3-3.0</td><td>3-32</td><td>0.03-1.25</td><td>56-2385</td></tr><tr><td rowspan="3">Oscillator accel. residuals</td><td>PF + PMMH (ours)</td><td>1</td><td> $_ { 1 . 0 3 5 } ^ { 1 . 0 3 5 \pm 0 . 2 2 0 }$ </td><td>1.00</td><td>875</td><td>1.00</td><td>0.51</td></tr><tr><td>NSVI</td><td>2</td><td></td><td></td><td></td><td>1.36-1.38</td><td></td></tr><tr><td>NPSGLD</td><td>3</td><td>1.00–1.12↑</td><td>1.1-1.5</td><td>6-86</td><td>1.09-1.90</td><td>21-190</td></tr><tr><td rowspan="3">Double pendulum torque residuals</td><td>PF + PMMH (ours)</td><td>1</td><td>1.615 ± 0.394</td><td>1.00</td><td>343</td><td>1.00</td><td>6.71</td></tr><tr><td>NSVI</td><td>16</td><td>0.17-1.58</td><td></td><td></td><td>0.03-0.10</td><td></td></tr><tr><td>NPSGLD</td><td>5</td><td>1.50–1.92 †, 2 diverged</td><td>2.5-277</td><td>3-4</td><td>0.07–0.94</td><td>141-495</td></tr></table>

Cost. Our method produces one efective sample every 0.5–6.7 s. NPSGLD, the only baseline that produces MCMC samples, needs 21 to 2,385 s per efective sample, i.e., 21× to 2,900× more, depending on the configuration. Its cheapest pendulum run finishes about five times sooner than our method, but it is 21× more expensive per efective sample, since its 423 s yield only three efective samples. A larger budget does not improve its mixing either: the longest NPSGLD run, with $1 0 ^ { 7 }$ iterations over 11,444 s, yields five efective samples.

NSVI converges to a biased posterior. On the oscillator, NSVI is fast, and most of its settings agree on the drift-invariant scale; but the posterior they agree on is the wrong one. In six of its eight force-residual runs, the damping ratio lies at $c / m = 0 . 7 8 – 0 . 8 1$ , whereas the truth is 1.20 and our method gives $1 . 2 0 9 \pm 0 . 2 3 6$ , an underestimate of about 1.8 reference standard deviations. The other two runs, both with $1 0 ^ { 5 }$ outer iterations, end at $c / m = 0 . 6 9$ and 0.94. More computation does not reduce the bias: raising the outer iterations from $3 \cdot 1 0 ^ { 4 }$ to $3 \cdot 1 0 ^ { 5 }$ and the inner iterations from 32 to 256 increases the run time 113-fold, from 128 s to $^ { 1 4 , 4 7 4 \mathrm { s } , }$ but moves $c / m$ only from 0.790 to 0.800. The bias is therefore a property of the converged approximation, not of insuficient computation. On the chaotic pendulum, NSVI does not converge at all: across 16 settings, the mass scale ranges from 0.17 to 1.58, so its final estimate reflects where the budget ran out rather than the posterior. It is also strongly overconfident there: its posterior standard deviations are only 0.03–0.10 times that of our method (Table 10), i.e., ten to thirty times too small, so that its scattered estimates come with misleadingly narrow uncertainty

NPSGLD misrepresents the uncertainty. None of the 18 NPSGLD runs, on either system and in any configuration, reaches $\hat { R } \leq 1 . 0 5$ . Because its chains do not mix, its narrow posteriors reflect insuficient exploration rather than precision. On the pendulum, NPSGLD reports $m _ { 1 } + m _ { 2 } = 1 . 4 9 5$ with a standard deviation fourteen times too small; on the oscillator with force residuals, it reports $m = 1 . 0 0 0$ with a standard deviation thirty times too small. At the same time, its posteriors over the identified ratios are up to 2.5× too wide. The NPSGLD posterior is therefore not only mislocated but also misshapen. Our method, in contrast, converges in every configuration $( \hat { R } \leq 1 . 0 0 1$ over four independent chains) and is asymptotically exact.

Why the two systems difer. Both baselines perform considerably better on the oscillator than on the pendulum, for structural reasons. The oscillator is linear, so its path posterior is close to Gaussian, and the Gaussian outer guide of NSVI is nearly well specified. Moreover, with force residuals, the comparatively large process noise $( \sigma _ { r } = 0 . 3 0 )$ determines the drift-invariant scale, which is the direction that nested schemes find most dificult. The pendulum, in contrast, is chaotic, its mass matrix depends on the state, and its path posterior over 1,235 steps and two degrees of freedom is strongly non-Gaussian. Nested inference therefore does not fail everywhere, but it succeeds only where the problem is close to what its approximations assume, and even there NPSGLD costs more than 40× as much per efective sample as our method.