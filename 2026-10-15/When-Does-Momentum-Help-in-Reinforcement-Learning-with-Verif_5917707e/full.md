# When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?

Kaifei Wang<sup>∗</sup> Zikang Shan<sup>∗</sup> Liwei Wang<sup>∗</sup> Yinyu Ye<sup>†</sup> Han Zhong<sup>‡</sup>

## Abstract

Momentum is a classical tool for accelerating convex optimization and a standard component of modern large language model training. Yet recent empirical work has questioned the usefulness of first-moment momentum in reinforcement learning with verifiable rewards (RLVR). We study this question through a bandit formulation of RLVR with bias-corrected exponential moving average (EMA) momentum, where the EMA coeficient β controls how quickly the weights on past gradients decay. When policy gradients are evaluated exactly, we prove that increasing β within a range that prevents overshoot strictly accelerates optimization. When gradients are estimated from sampled responses, a positive expected gain can be obscured by sampling fluctuations. We show that as the learning rate η decreases, the expected gain with fixed β vanishes faster than these fluctuations, whereas increasing β to keep η/(1 − β) within the safe regime preserves a nonvanishing gain that eventually dominates them. Bandit simulations quantitatively validate these predictions. RLVR experiments on Qwen3-4B-Base using SGD with EMA and AdamW support our principle of choosing larger coeficients in practical LLM post-training. On Qwen2.5-32B with AdamW, simply increasing β improves mean validation accuracy by nearly 2 percentage points over the conventional choice, demonstrating the principle’s efectiveness at a larger model scale.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become an important approach to improving the reasoning capabilities of large language models (LLMs) (Lambert et al., 2024; DeepSeek-AI et al., 2025; Yu et al., 2025). In RLVR, responses are sampled from the current policy and evaluated using verifiable rewards, so the training data distribution evolves as the policy changes. This difers from pretraining and supervised fine-tuning, which optimize next-token prediction (NTP) on given text sequences (Brown et al., 2020; Ouyang et al., 2022). Despite these diferences, RLVR commonly adopts Adam (Kingma and Ba, 2015) or AdamW (Loshchilov and Hutter, 2019), together with conventional first-moment momentum settings (Zhang et al., 2025; Mukherjee et al., 2026). The design and configuration of optimizers for RLVR have received relatively limited attention.

One important aspect of optimizer design and configuration is the use of first-moment momentum, implemented in Adam and AdamW as an exponential moving average (EMA) of past gradients. While EMA has shown empirical benefits in NTP training (Chen et al., 2023; Zhao et al., 2025), Mukherjee et al. (2026) report that momentum does not consistently improve performance in their RLVR experiments and can even reduce it under a conventional choice of the momentum coeficient. These observations motivate a closer examination of whether EMA can benefit RLVR and what factors influence its efectiveness.

To understand when EMA benefits RLVR and how to choose its coeficient, we adopt a KL-regularized response-level bandit formulation (Xiong et al., 2024). We analyze natural policy gradient (NPG, Kakade, 2001) with a softmax policy parameterization and bias-corrected EMA. NPG provides a tractable theoretical framework for studying policy improvement while controlling changes in the policy distribution, a principle shared by TRPO and PPO (Schulman et al., 2015, 2017; Shani et al., 2020). NPG-based analysis has also provided practical guidance for LLM post-training (e.g., Gai et al., 2026). Our contributions are as follows.

Deterministic acceleration and stochastic gains. We first prove that, with exact gradients, increasing the EMA coeficient within a safe regime strictly accelerates optimization, which appears to contrast with the limited benefits reported in prior RLVR experiments (Mukherjee et al., 2026). A deterministic gain, however, need not translate into a consistent improvement when gradients are estimated from sampled responses, as sampling fluctuations may obscure the benefit. We characterize the expectation and variance of the objective gain and prove that, for a fixed EMA coeficient, the expected gain remains positive for suficiently small learning rates but vanishes faster than the sampling fluctuations as the learning rate decreases (Theorems 4.2 and 5.3).

Principled selection of momentum coeficients. We next show that scaling the EMA coeficient β with the learning rate η so that $\eta / ( 1 - \beta )$ is a positive constant within the safe regime yields a nonvanishing expected gain that eventually dominates sampling fluctuations (Theorem 5.4). We further prove that an appropriately increased coeficient allows a smaller learning rate to achieve comparable expected performance under the same update and sample budgets, with asymptotically lower variance of the objective value (Theorem 5.5).

Validation in bandits and LLM post-training. We quantitatively validate the predicted gain and fluctuation scales in numerical bandit experiments. GRPO experiments (Shao et al., 2024) on Qwen3-4B-Base (Yang et al., 2025) support our theoretical predictions in practical LLM post-training with both SGD with EMA and the widely used AdamW optimizer: larger coeficients improve performance and, when paired with smaller learning rates, yield more stable training with comparable or slightly improved performance. We further demonstrate the principle’s efectiveness at a larger model scale: on Qwen2.5-32B (Qwen et al., 2024), simply increasing AdamW’s first-moment coeficient improves mean validation accuracy by nearly 2 percentage points over the conventional choice.

## 2 Related work

Optimization and momentum in RLVR. Recent work revisits optimizer design for RLVR through momentum-based normalization (Zhang et al., 2025), spectral transformations (Fan et al., 2026; Shi et al., 2026; Zhu et al., 2026), and parameter-space exploration (Venkatkrishna et al., 2026). Focusing on optimizer momentum, Fontana et al. (2026) analyze how accumulated AdamW momentum can continue driving updates after clipping suppresses the current policy-gradient contribution during repeated optimization of the same batch. Most directly related to our motivation, Mukherjee et al. (2026) find that SGD can match or outperform AdamW and report no consistent benefit from momentum. Motivated by these empirical findings, we study how the benefit of ordinary EMA momentum in RLVR depends on the momentum coeficient, learning rate, and sampling noise.

Momentum in reinforcement learning. Prior work studies momentum for accelerating policy optimization and reducing gradient-estimation error in conventional RL. Chen et al. (2024) establish accelerated convergence for exact-gradient policy optimization under tabular softmax parameterization using Nesterov momentum with a restart rule. Momentum-based variance reduction has also been studied for policy gradient (Huang et al., 2020; Ding et al., 2022) and natural policy gradient (Feng et al., 2024). Huang et al. (2022) develop Bregman policy optimization with basic momentum and a variance-reduced variant. This literature primarily concerns convergence and sample complexity in conventional RL, with reported experiments on control and Atari benchmarks. We study bias-corrected averaging of past natural gradients in a KL-regularized RLVR bandit model, comparing objective values with and without momentum and examining the practical implications through GRPO experiments with language models.

Momentum in supervised learning and stochastic optimization. Analyses of online learning and general stochastic optimization characterize regimes in which momentum behaves similarly to SGD under appropriate learning-rate scaling (Yuan et al., 2016; Wang et al., 2024). In language-model pretraining, AdEMAMix combines fast and slow gradient averages to exploit longer gradient memory (Pagliardini et al., 2025), while ADana combines time-varying momentum schedules with a direct gradient term and damping (Ferbach et al., 2026). These pretraining studies concern next-token prediction on given text sequences. Our focus is momentum in RLVR, where reward feedback is obtained from responses sampled from the evolving policy.

## 3 Preliminaries

In this section, we introduce the model, algorithms, and notation used throughout the paper, providing the foundation for our theoretical analysis.

## 3.1 Model and parameterization

Basic Setup. Let $x \in \mathcal { X }$ be a given input prompt, and let $y \in \mathcal { V }$ be a corresponding response generated by the language model. The model is parameterized as a policy π characterized by $\theta ,$ from which responses are sampled $y \sim \pi _ { \theta } ( \cdot \mid x )$ . Each prompt-response pair $( x , y )$ is evaluated by a verifier and receives a reward $r ( x , y )$ . We consider the case of deterministic binary rewards: $r ( x , y )$ is 1 if and only if the response y is a correct answer to the prompt x, and 0 otherwise. Let $\mathbf { r } \in \{ 0 , \dot { 1 } \} ^ { | \dot { y } | }$ denote the vector of rewards for all possible responses to the prompt x. We maximize the expected reward while penalizing divergence from a fixed reference policy $\pi _ { \mathrm { r e f } }$ , using the objective

$$
J ( \theta ) = \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot | x ) } [ r ( x , y ) ] - \gamma \mathrm { K L } ( \pi _ { \theta } ( \cdot  { | } x ) | | \pi _ { \mathrm { r e f } } ( \cdot  { | } x ) ) ,\tag{3.1}
$$

where $\begin{array} { r } { \mathrm { K L } ( \pi _ { \theta } ( \cdot  { | \mathbf {  { x } } ) | } | \pi _ { \mathrm { r e f } } ( \cdot  { | \mathbf {  { x } } ) } ) : = \sum _ { \boldsymbol { y } \in \mathcal { Y } } \pi _ { \theta } ( \boldsymbol { y }  { | \mathbf {  { x } } ) \log \bigl ( } \pi _ { \theta } ( \boldsymbol { y }  { | \mathbf {  { x } } ) } / \pi _ { \mathrm { r e f } } ( \boldsymbol { y }  { | \mathbf {  { x } } ) } \bigr ) } \end{array}$ is the Kullback-Leibler divergence between the policy $\pi _ { \theta } ( \cdot \mid x )$ and the reference policy, and $\gamma > 0$ is a regularization coeficient. Assume $\pi _ { \mathrm { r e f } }$ is a full-support policy. To exclude degenerate cases, we assume that the initial policy is suboptimal and that the response set contains both correct and incorrect responses. The objective $J ( \theta )$ encourages higher expected reward while penalizing deviations from a fixed reference policy, such as one obtained through supervised fine-tuning.

Policy parameterization. We adopt the standard softmax parameterization to model response generation. Given a prompt x, we parameterize the conditional distribution over responses using a logit vector $\theta \in \mathbb { R } ^ { | \mathcal { V } | }$ as $\pi _ { \boldsymbol { \theta } } \big ( \dot { y } \big | \dot { x } \big ) \stackrel { \cdot } { = } e ^ { \theta ( \dot { x } , y ) } \big / \dot { \sum } _ { y ^ { \prime } \in \mathcal { Y } } e ^ { \theta ( x , y ^ { \prime } ) }$ , where $\theta ( x , y )$ is the logit score for response y given prompt x. This response-level softmax parameterization provides a tractable abstraction of autoregressive generation in modern LLMs, whose token-level policies use softmax distributions (Ziegler et al., 2019; Zhong et al., 2025).

## 3.2 Training dynamics

Natural policy gradient. Our theoretical analysis focuses on the natural policy gradient (NPG) method, which preconditions the policy gradient using the Fisher information matrix (Kakade, 2001). Let $\theta _ { t }$ denote the parameter vector at iteration t and $\pi _ { \boldsymbol { \theta } _ { t } } ( \cdot \mid x )$ the induced policy. When policy gradients are evaluated exactly, the NPG update is given by:

$$
\begin{array} { r } { \theta _ { t + 1 } = \theta _ { t } + \eta \mathcal { F } ^ { \dagger } ( \theta _ { t } ) \nabla J ( \theta _ { t } ) , \qquad \mathcal { F } ( \theta ) = \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ \nabla \log \pi _ { \theta } ( y \mid x ) \nabla \log \pi _ { \theta } ( y \mid x ) ^ { \top } \right] . } \end{array}\tag{3.2}
$$

Here $\eta > 0$ represents the learning rate, ${ \mathcal { F } } ( \theta )$ is the Fisher information matrix, and ${ \mathcal { F } } ^ { \dagger } ( \theta )$ is its Moore–Penrose pseudoinverse. When policy gradients are estimated from sampled responses, we replace the gradient $\nabla J ( \cdot )$ with its unbiased estimator $\widehat { \nabla J } ( \cdot )$ , which uses a batch of sampled responses to approximate the expected reward, while the KL gradient and the Fisher information matrix are computed exactly. Specifically, let $\widehat { \theta } _ { t }$ be the parameter vector under the sampled NPG update. The update is given by:

$$
\widehat { \theta } _ { t + 1 } = \widehat { \theta } _ { t } + \eta \mathcal { F } ^ { \dagger } ( \widehat { \theta } _ { t } ) \widehat { \nabla J } ( \widehat { \theta } _ { t } ) .
$$

NPG not only enjoys favorable convergence guarantees under softmax parameterization (Agarwal et al., 2021; Cen et al., 2022), but also connects to modern RL algorithms used in LLM post-training. It can be viewed as maximizing a linearized objective while controlling the KL divergence between consecutive policies (Shani et al., 2020; Cai et al., 2020; Zhong and Zhang, 2023), sharing the principle of limiting policy changes with TRPO (Schulman et al., 2015) and with PPO and GRPO, which are used in LLM post-training (Schulman et al., 2017; Ouyang et al., 2022; Shao et al., 2024). We study NPG to shed light on the training dynamics of RL algorithms used in modern LLM post-training.

Proposition 3.1. Under the softmax parameterization, the exact and sampled NPG directions are given by:

$$
\mathscr { F } ^ { \dagger } ( \theta ) \nabla J ( \theta ) = \left( \mathbf { I } - \frac { \mathbf { 1 1 } ^ { \top } } { | \mathcal { Y } | } \right) \left[ \mathbf { r } - \gamma ( \theta - \log \pi _ { \mathrm { r e f } } ( \cdot  { | } \mathbf { \theta } x ) ) \right] ,\tag{3.3}
$$

$$
\mathcal { F } ^ { \dagger } ( \widehat { \theta } ) \widehat { \nabla J } ( \widehat { \theta } ) = \Big ( \mathbf { I } - \frac { \mathbf { 1 1 } ^ { \top } } { | \mathcal { V } | } \Big ) \Big [ \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { r ( x , y _ { b } ) } { \pi _ { \widehat { \theta } } ( y _ { b } \mid x ) } \mathbf { e } _ { y _ { b } } - \gamma ( \widehat { \theta } - \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) ) \Big ] .\tag{3.4}
$$

Here B denotes the batch size used to estimate the policy gradient, and $y _ { 1 } , \dots , y _ { B } \overset { \mathrm { i i d } } { \sim } \pi _ { \widehat { \theta } } ( \cdot \mid x )$ are the sampled responses. Let I be the identity matrix, 1 the all-ones vector, and $\mathbf { e } _ { y }$ bthe one-hot vector corresponding to response y.

Exponential Moving Average (EMA) momentum. In classical optimization, Polyak’s heavy-ball momentum can accelerate convergence under suitable curvature and parameter conditions (Polyak, 1964), and global convergence guarantees are available for smooth convex objectives (Ghadimi et al., 2015). EMA is its normalized form and is widely used in modern LLM training optimizers, such as Adam and AdamW (Kingma and Ba, 2015; Loshchilov and Hutter, 2019). EMA maintains a first-moment momentum $m _ { t } ,$ where the EMA coeficient $\beta \in ( 0 , 1 )$ controls the contribution of the current gradient to the parameter update. With exact gradients, the NPG update with EMA is given by

$$
m _ { t + 1 } = \beta m _ { t } + ( 1 - \beta ) \mathcal { F } ^ { \dagger } ( \theta _ { t } ^ { \beta } ) \nabla J ( \theta _ { t } ^ { \beta } ) , \qquad \theta _ { t + 1 } ^ { \beta } = \theta _ { t } ^ { \beta } + \eta \frac { m _ { t + 1 } } { 1 - \beta ^ { t + 1 } } .\tag{3.5}
$$

The initial condition is $m _ { 0 } = 0$ and $\theta _ { t } ^ { \beta }$ denotes the parameter vector updated by EMA with coeficient $\beta$ at iteration t. In the stochastic setting, we replace $\nabla J ( \cdot )$ with its unbiased estimator and denote the corresponding parameter vector by $\widehat { \theta } _ { t } ^ { \beta }$

Without EMA, the current natural gradient is applied in full to the current parameter update. With EMA, a gradient’s contribution is spread across the current and subsequent updates, while each update uses a weighted sum of the current and past natural gradients. Thus, EMA can be viewed as a temporal reallocation of gradient contributions to parameter updates. To isolate this reallocation from changes in the total gradient weight, we normalize $m _ { t }$ by $1 - \beta ^ { t }$ for $t \geq 1$ before using it to update the parameters. This makes the gradient weights in each update sum to one, following the first-moment bias correction used in Adam and AdamW.

Let $\omega _ { \beta } ( t , t ^ { \prime } )$ denote the weight of the gradient at time $t ^ { \prime }$ in the update at time t, under EMA with coeficient $\beta .$ . For $\beta \in ( 0 , 1 )$ ), (3.5) can be rewritten as

$$
\theta _ { t + 1 } ^ { \beta } = \theta _ { t } ^ { \beta } + \eta \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) \mathscr { F } ^ { \dagger } ( \theta _ { t ^ { \prime } } ^ { \beta } ) \nabla J ( \theta _ { t ^ { \prime } } ^ { \beta } ) , \qquad \omega _ { \beta } ( t , t ^ { \prime } ) = \frac { ( 1 - \beta ) \beta ^ { t - t ^ { \prime } } } { 1 - \beta ^ { t + 1 } } .
$$

The stochastic update uses the same weights. In particular, we define $\omega _ { 0 } ( t , t ^ { \prime } ) = 0$ for all $t ^ { \prime } \neq t$ and $\omega _ { 0 } ( t , t ) = 1$ which recovers the standard NPG update without EMA. For consistency, we use the same notation with $\beta = 0$ to denote the corresponding quantities for vanilla NPG without EMA.

## 4 Deterministic case: acceleration with exact gradients

In this section, we consider deterministic optimization with exact policy gradients, where the natural-gradient direction is given by (3.3). Maximizing J in (3.1) yields the unique optimal policy $\pi ^ { \star } ( y \mid x ) \propto \pi _ { \mathrm { r e f } } ( y \mid$ x) exp $( r ( x , y ) / \gamma )$ . For an arbitrary initial parameter $\theta _ { 0 }$ , define

$$
\theta ^ { \star } : = \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) + \frac { \mathbf { r } } { \gamma } + \frac { \mathbf { 1 } ^ { \top } \left[ \theta _ { 0 } - \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) - \mathbf { r } / \gamma \right] \mathbf { 1 } } { \vert \mathcal { V } \vert } ,
$$

which is one of the maximizers of J and induces $\pi ^ { \star }$ . Here r is the reward vector and |Y| is the number of possible responses. We compare the exact NPG iterates obtained with EMA coeficient $\beta$ against those obtained without EMA $( \beta = 0 )$ . Both runs are initialized at $\theta _ { 0 }$ with $m _ { 0 } = 0$ and use the same learning rate η and number of updates T.

Following the update in (3.5), we recursively define a family of functions $\{ f _ { t } : [ 0 , 1 ) \to \mathbb { R } \} _ { t \geq 0 }$ by $f _ { 0 } ( \beta ) = 1$ and $\begin{array} { r } { f _ { t + 1 } ( \beta ) = f _ { t } ( \beta ) - \eta \gamma \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) f _ { t ^ { \prime } } ( \beta ) } \end{array}$ for $t \geq 0$ . In particular, for vanilla NPG, $f _ { t } ( 0 ) = ( 1 - \eta \gamma ) ^ { t }$ The following lemma shows that these functions characterize the natural policy gradient and the parameter error at each iteration.

Lemma 4.1 (Exact dynamics). The exact NPG iterates in (3.5) satisfy, for every $t \geq 0$

$$
\mathscr { F } ^ { \dagger } ( \theta _ { t } ^ { \beta } ) \nabla J ( \theta _ { t } ^ { \beta } ) = f _ { t } ( \beta ) \mathscr { F } ^ { \dagger } ( \theta _ { 0 } ) \nabla J ( \theta _ { 0 } ) , \qquad \theta _ { t } ^ { \beta } - \theta ^ { \star } = f _ { t } ( \beta ) ( \theta _ { 0 } - \theta ^ { \star } ) .
$$

The proof is in Appendix B. Lemma 4.1 shows that the exact natural gradients at all iterations are collinear and that the parameter error ${ \theta } _ { t } ^ { \beta } - { \theta } ^ { \star }$ remains a scalar multiple of $\theta _ { 0 } - \theta ^ { \star }$ . As long as $f _ { t ^ { \prime } } ( \beta ) > 0$ for all $0 \leq t ^ { \prime } \leq t .$ , the iterates up to time t do not overshoot $\theta ^ { \star }$ . Over these iterations, the iterates $\theta _ { t ^ { \prime } } ^ { \beta }$ approach $\theta ^ { \star }$ so the natural gradients point in the same direction and decrease in magnitude. EMA averages the current natural gradient with earlier, larger ones, which can accelerate optimization. The following theorem gives a suficient range that prevents overshoot and establishes a strict benefit from increasing $\beta .$ The proof is in Appendix B.

Theorem 4.2 (Strict acceleration from momentum). Assume $0 < \eta \gamma < 1$ and $T \geq 2$ . For any $\beta _ { 1 } , \beta _ { 2 }$ in the range $[ 0 , ( ( 1 - \eta \gamma ) / ( 1 + \eta \gamma ) ) ^ { 2 } ]$ with $\beta _ { 1 } < \beta _ { 2 }$ , we have $J ( \theta _ { T } ^ { \beta _ { 2 } } ) > J ( \theta _ { T } ^ { \beta _ { 1 } } )$ . In particular, every positive $\beta$ in this range satisfies $J ( \theta _ { T } ^ { \beta } ) > J ( \theta _ { T } ^ { 0 } )$

Thus, within the stated range, EMA accelerates deterministic optimization at the same learning rate and update count through a temporal reallocation of natural gradients. This raises two questions: Why can a strictly positive exact-gradient benefit fail to yield a reliable improvement in sampled training? How can this benefit be reliably realized and exploited in sampled training?

## 5 Stochastic case: sampling fluctuations and expected gain

We now study the stochastic setting, where natural gradients are estimated from sampled responses, introducing noise into the updates. To analyze training dynamics at small learning rates, we fix the problem, the batch size $B \geq 1$ , and the optimization time $\tau = \eta T > 0$ , and let $\eta = \tau / T \to 0$ . We compare $J ( \widehat { \theta } _ { T } ^ { \beta } )$ and $J ( \widehat { \theta _ { T } ^ { 0 } } )$ obtained with and without EMA. The two runs share the same settings, including the initial parameter $\theta _ { 0 } .$ zero initial momentum, learning rate $\eta ,$ batch size $B ,$ , and update count $T ,$ while using independent samples even when $\beta = 0$ . Expectations and variances are over training randomness. The objective J is evaluated exactly.

By Proposition 3.1, the sampled natural-gradient direction is conditionally unbiased. Taking expectations in the EMA updates then gives $\mathbb { E } [ \widehat { \theta } _ { t } ^ { \beta } ] = \theta _ { t } ^ { \beta }$ . To characterize the expected objective, we must also account for sampling fluctuations. Let $\Sigma _ { T } ^ { \beta }$ denote a covariance coeficient characterizing the accumulated sampling noise along the exact trajectory $\theta _ { t } ^ { \beta }$ . It is a nonnegative weighted sum of noise covariance matrices evaluated along this trajectory, where the noise comes from estimating the reward vector r using a single sampled response. This coeficient is a deterministic positive semidefinite matrix, independent of the realized samples. Its precise definition is given in Appendix C. Using this coeficient, the following lemma expresses the expected objective diference as the exact diference plus a sampling correction.

Lemma 5.1 (Expected objective diference). Fix $B \geq 1$ and $\tau > 0$ . For any $\beta \in [ 0 , 1 )$ , as $\eta = \tau / T \to 0$ , the expected objective diference satisfies

$$
\mathbb { E } \left[ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) \right] = J ( \theta _ { T } ^ { \beta } ) - J ( \theta _ { T } ^ { 0 } ) + \frac { \eta } { 2 B } \operatorname { t r } \left[ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } - \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \Sigma _ { T } ^ { 0 } \right] + O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

The remainder is uniform in $\beta \in [ 0 , 1 )$ , and $\operatorname { s u p } _ { \beta \in [ 0 , 1 ) } \| \Sigma _ { T } ^ { \beta } \| _ { \mathrm { o p } }$ is bounded independently of $\eta .$

Establishing the expansion in Lemma 5.1 requires controlling sampling fluctuations throughout training. This is challenging because the sampled natural-gradient direction weights rewards by inverse sampling probabilities: a sampled response with positive reward and very small probability can produce a large noise term. To control this noise, we construct an auxiliary parameter update process whose distance from the exact iterates is bounded. Since the exact trajectory is bounded, the sampling probabilities in this auxiliary process stay bounded away from zero, and its sampling noise is bounded. During training, the probability that the auxiliary process and the original sampled process difer is suficiently small. Therefore, we can characterize the expected objective diference using the auxiliary process. A proof sketch follows and Appendix C gives the full proof.

Proof sketch. Let $\varepsilon _ { t } ^ { \beta }$ denote the estimation noise in the sampled natural gradient at $\widehat { \theta } _ { t } ^ { \beta }$ . We construct an auxiliary process driven by the same noise sequence, initialized at $\widetilde { \theta } _ { 0 } ^ { \beta } = \theta _ { 0 }$ and defined by

$$
\widetilde { \varepsilon } _ { t } ^ { \beta } = \mathbf { 1 } \Big \{ \operatorname* { m a x } _ { t ^ { \prime } \leq t } \| \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } - \theta _ { t ^ { \prime } } ^ { \beta } \| _ { \infty } \leq 1 \Big \} \varepsilon _ { t } ^ { \beta } , \quad \widetilde { \theta } _ { t + 1 } ^ { \beta } = \widetilde { \theta } _ { t } ^ { \beta } + \eta \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) \left[ \mathcal { F } ^ { \dagger } ( \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } ) \nabla J ( \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } ) + \widetilde { \varepsilon } _ { t ^ { \prime } } ^ { \beta } \right] .
$$

The two processes coincide through the first exit from the unit neighborhood of the exact path. After this exit, the exact direction and EMA memory continue to evolve, while no further sampling noise enters the auxiliary process. The construction gives ma $\mathrm { { x } } _ { t \leq T } \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { \infty } = O ( 1 )$ for every sample path. Since the exact trajectory is bounded, this implies $\begin{array} { r } { \operatorname* { i n f } _ { t \leq T , y \in \mathcal { V } } \pi _ { \widetilde { \theta } _ { \star } ^ { \beta } } ( y \mid x ) = \Omega ( 1 ) } \end{array}$ and ma $\mathrm { x } _ { t < T } \| \widetilde { \varepsilon } _ { t } ^ { \beta } \| _ { \infty } = O ( 1 )$ . These bounds eallow us to estimate the covariance and third moment of the auxiliary parameter fluctuations. A second-order Taylor expansion, together with a comparison of the noise covariances along the auxiliary and exact paths, then yields

$$
\mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] = J ( \theta _ { T } ^ { \beta } ) + \frac { \eta } { 2 B } \mathrm { t r } \left[ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } \right] + O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

For each fixed update and coordinate, the weighted noise contributions have zero conditional mean, so their partial sums form a martingale. Using a concentration bound and a union bound, we show that the probability of path separation is suficiently small, uniformly in $\beta .$ Since J is bounded, we obtain

$$
\left. \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) ] - \mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] \right. \leq 2 \operatorname* { s u p } _ { \theta } \vert J ( \theta ) \vert \mathbb { P } ( \exists t \leq T : \widehat { \theta } _ { t } ^ { \beta } \neq \widetilde { \theta } _ { t } ^ { \beta } ) = o ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

Combining these estimates and subtracting the expansions for $\beta$ and 0 proves Lemma 5.1.

Applying the same analysis to $J ^ { 2 }$ also allows us to quantify the fluctuations in the objective diference through its variance.

Proposition 5.2 (Variance of the objective diference). Under the conditions of Lemma 5.1, the independent runs satisfy the following expansion uniformly in $\beta \in [ 0 , 1 )$

$$
\operatorname { V a r } \left[ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) \right] = \frac { \eta } { B } \left[ \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( \theta _ { T } ^ { \beta } ) + \nabla J ( \theta _ { T } ^ { 0 } ) ^ { \top } \Sigma _ { T } ^ { 0 } \nabla J ( \theta _ { T } ^ { 0 } ) \right] + O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

Under the nondegeneracy condition, specifically, lim $\begin{array} { r } { \operatorname* { i n f } _ { \eta \to 0 } \nabla J ( \theta _ { T } ^ { 0 } ) ^ { \top } \Sigma _ { T } ^ { 0 } \nabla J ( \theta _ { T } ^ { 0 } ) > 0 } \end{array}$ , uniformly over $\beta \in [ 0 , 1 )$ we have

$$
\mathrm { V a r } \left[ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) \right] = \Theta ( \eta / B ) .
$$

The proof is in Appendix D. Proposition 5.2 shows that the variance of the objective diference remains of order $\eta / B$ regardless of the EMA coeficient. Lemma 5.1 shows that the expected diference is characterized by the exact objective diference. Thus, when the exact gain is small relative to the standard-deviation scale $\sqrt { \eta / B }$ , training fluctuations make a consistent benefit from EMA dificult to observe.

The critical metric is $\rho = \eta / ( 1 - \beta )$ . Intuitively, EMA averages natural gradients over an efective memory window of $1 / ( 1 - \beta )$ steps. Since each exact update moves the parameters by ${ \cal { O } } ( \eta )$ , the averaged gradients span a parameter-displacement scale of $O ( \eta / ( 1 - \beta ) )$ , which controls their diference from the current gradient. When this ratio is small, gradients vary little within the window, so EMA behaves similarly to the update without momentum and the diference vanishes. The next two subsections quantify the cases of a vanishing or fixed positive ratio.

## 5.1 A fixed EMA coeficient gives a vanishing expected gain

Theorem 5.3 (Fixed EMA coeficient). Fix $\beta \in ( 0 , 1 )$ , $B \geq 1 , \tau > 0$ , and suppose that the nondegeneracy condition in Proposition 5.2 holds. As $\eta = \tau / T \to 0$ , the expected objective gain satisfies

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] = C ( \tau , \gamma ) \frac { \eta \beta } { 1 - \beta } + O ( \eta ^ { 3 / 2 } ) ,
$$

![](images/89dbfbb6b76c9299abf6839b71de8c7a54459cc7dc767ca946c1e58746c4093a.jpg)  
(a) Gain and Noise.

![](images/66003a991eea728bafa684e4dc9ff643409ff5d82fe66ccdbbac94a5f9fc1c2a.jpg)  
(b) Win Rate.

![](images/1dfaa3ce24e884c3c9e691472e5abcbe9e7b8069832579e527a4fd765102d79a.jpg)  
(c) Gain and Noise.

![](images/afd96ea811de111be98a1b48b23aa32cf1a0c2ed21a51b1ec5cbaa621e8990cb.jpg)  
(d) Win Rate.  
Figure 1: Numerical bandit results with 3 reward-one and 7 reward-zero arms under uniform initialization (1(a), 1(b)) and random policy initialization (1(c), 1(d)).

where $C ( \tau , \gamma )$ is a positive constant depending only on τ , γ, and the fixed problem. It is independent of $\beta .$ The probability that EMA yields a higher objective value satisfies

$$
\operatorname* { l i m } _ { \eta \to 0 } \mathbb { P } \big ( J ( \widehat { \theta } _ { T } ^ { \beta } ) > J ( \widehat { \theta } _ { T } ^ { 0 } ) \big ) = 1 / 2 .
$$

The proof is in Appendix E. Theorem 5.3 shows that fixed $\beta$ gives a positive expected gain for suficiently small $\eta ,$ but the gain is only of order $\eta ,$ which matches the variance order in Proposition 5.2 with fixed batch size. Sampling fluctuations therefore overwhelm the gain, making the two algorithms dificult to distinguish in individual runs. This ofers a possible explanation for the limited benefits of momentum reported in prior RLVR experiments (Mukherjee et al., 2026).

## 5.2 Scaling the EMA coeficient with the learning rate

Beyond Theorem 5.3, we further show that $\eta / ( 1 - \beta ) \to 0$ forces the expected gain to vanish, even when $\beta$ depends on $\eta .$ The full proof is given in Appendix F. To satisfy the suficient condition for avoiding overshoot in Theorem 4.2, this ratio must remain bounded. Thus, a nonvanishing expected gain requires $\eta / ( 1 - \beta ) = \Theta ( 1 )$ . For simplicity, we set $\beta$ so that $\eta / ( 1 - \beta ) = \rho .$ , where $\rho > \eta$ is a positive constant. In Appendix G, we prove that $\rho \in \left. ( 0 , 1 / ( 4 \gamma ) \right]$ ensures the suficient condition in Theorem 4.2. The following theorem compares the objective values of two independent runs with and without EMA in this setting.

Theorem 5.4 (EMA coeficient scaling with the learning rate). Fix $B \geq 1 , \tau > 0$ , and $0 < \rho \leq 1 / ( 4 \gamma )$ , and suppose that the nondegeneracy condition in Proposition 5.2 holds. Set $\beta = 1 - \eta / \rho$ . As $\eta = \tau / T \to 0$ , the expected objective gain satisfies

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] = C ( \tau , \gamma , \rho ) \beta \rho + O ( \eta ) ,
$$

where $C ( \tau , \gamma , \rho )$ is a positive constant depending only on $\tau , \gamma , \rho ,$ and the fixed problem. It is independent of η and $\beta$ . The probability that EMA yields a higher objective value satisfies

$$
\mathbb { P } \big ( J ( \widehat { \theta } _ { T } ^ { \beta } ) > J ( \widehat { \theta } _ { T } ^ { 0 } ) \big ) = 1 - O ( \eta / B ) .
$$

The proof is in Appendix G. When $\rho = \eta / ( 1 - \beta )$ is held at a fixed positive value in the safe regime, Theorem 5.4 gives a nonvanishing expected gain, while Proposition 5.2 shows that the standard deviation of the objective diference is of order $\sqrt { \eta / B }$ . As $\eta  0$ , the expected gain therefore dominates sampling fluctuations, and EMA’s advantage becomes consistently observable in the stochastic case. These findings suggest that choosing $\beta$ substantially above the conventional fixed value can yield greater benefits in practice.

## 5.3 Reducing variability across runs with EMA

We next investigate whether EMA can help reduce variability across runs while achieving comparable expected performance under the same training budget. A smaller learning rate can reduce sampling fluctuations, but it may also slow optimization over a given number of updates. The deterministic acceleration established in Section 4 suggests that increasing the EMA coeficient may compensate for this slower progress. We compare two independent EMA runs with settings $( \eta _ { 1 } , \beta _ { 1 } )$ and $( \eta _ { 2 } , \beta _ { 2 } )$ , using the same initialization, batch size $B ,$ and update count T. We write $\theta _ { T } ^ { ( i ) }$ and $\widehat { \theta } _ { T } ^ { \left( i \right) }$ for the parameters obtained with exact and sampled gradients, respectively, under setting $i = 1 , \bar { 2 }$

Theorem 5.5 (Comparable expected performance with lower variance). Fix $B > 1$ and $\tau > 0$ . For any $( \eta _ { 1 } , \beta _ { 1 } )$ with $\eta _ { 1 } = \tau / T$ and $T \geq 2$ , and any $\rho _ { 2 }$ satisfying $0 < \rho _ { 1 } = \eta _ { 1 } / ( 1 - \beta _ { 1 } ) < \rho _ { 2 } \leq 1 / ( 4 \gamma )$ , there exist $0 < \eta _ { 2 } < \eta _ { 1 }$ and $\beta _ { 1 } < \beta _ { 2 } < 1$ such that $\eta _ { 2 } / ( 1 - \beta _ { 2 } ) = \rho _ { 2 }$ and $\theta _ { T } ^ { ( 2 ) } = \theta _ { T } ^ { ( 1 ) }$ . As $\eta _ { 1 }  0$ with $\rho _ { 1 } , \rho _ { 2 }$ fixed, the expected objective diference and the limiting variance ratio satisfy

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { ( 2 ) } ) - J ( \widehat { \theta } _ { T } ^ { ( 1 ) } ) ] = O ( \eta _ { 1 } / B ) , \qquad \operatorname* { l i m } _ { \tau \to \infty } \operatorname* { l i m } _ { \eta _ { 1 } \to 0 } \frac { \mathrm { V a r } [ J ( \widehat { \theta } _ { T } ^ { ( 2 ) } ) ] } { \mathrm { V a r } [ J ( \widehat { \theta } _ { T } ^ { ( 1 ) } ) ] } = \frac { 1 + \sqrt { 1 - 4 \gamma \rho _ { 2 } } } { 1 + \sqrt { 1 - 4 \gamma \rho _ { 1 } } } < 1 .
$$

The proof is in Appendix H. Increasing the EMA coeficient compensates for the smaller learning rate by matching the parameters obtained with exact gradients after the same number of updates. With sampled gradients, the expected objective values difer by $O ( \eta _ { 1 } / B )$ , while the variance ratio converges to a constant below one in the stated asymptotic regime. Thus, EMA can help reduce variability across runs while achieving comparable expected performance under the same update count T and sample budget BT.

## 6 Experiments

We use numerical bandit experiments to quantitatively validate the predictions of Proposition 5.2 and Theorems 5.3 and 5.4. Then we turn to RLVR experiments on Qwen3-4B-Base to test our theoretical predictions in a practical LLM post-training setting. Finally, we apply our principle of increasing the EMA coeficient to Qwen2.5-32B to assess its applicability as model size increases.

## 6.1 Validating theoretical predictions in bandits

Setup. We consider a 10-arm bandit in which 3 arms have reward 1 and the remaining 7 have reward 0. The reference policy is set equal to the initial policy, with KL penalty coeficient $\gamma = 1$ , batch size $B = 1$ , and optimization time $\tau = 1$ . Training uses $T = 3 2 , 6 4 , \dots$ , 4096 updates with learning rate $\eta = 1 / T$ . We compare sampled NPG with fixed $\beta = 0 . 9$ or scaled $\beta = 1 - 4 \eta$ against the no-EMA baseline. Each configuration uses 50,000 independent training replicates, with J evaluated exactly after training, to estimate the expectation and standard deviation of the objective gap $J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } )$ and the probability of outperforming an independent no-EMA run. To measure the gain relative to these fluctuations, we compare SNR of the objective gap, defined as $\mathrm { S N R } : = | \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] | / \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ]$ ]. Both uniform and random policy initializations are considered. Additional experiments with diferent reward configurations are provided in Appendix I.1.

Results. As the learning rate η decreases, Figure 1 shows that the standard deviation scales as $\sqrt { \eta }$ for small η under both momentum schedules, as predicted by Proposition 5.2. With fixed $\beta = 0 . 9$ , the mean gain remains positive for small $\eta$ but scales as $\eta ,$ decreasing faster than the standard deviation. With scaled $\beta = 1 - 4 \eta$ , it approaches a positive constant, consistent with Theorems 5.3 and 5.4. These trends hold under both uniform and random initialization. $\mathrm { A t } \ \eta = 1 / 4 0 9 6$ , fixed $\beta$ gives SNRs of 0.0453 and 0.0406 under uniform and random initialization, respectively, while scaled β gives 3.7647 and 3.3982. (More details are reported in Table 1.) The probability of EMA outperforming an independent no-EMA run approaches $1 / 2$ with fixed $\beta$ and 1 with scaled $\beta .$

## 6.2 Practical LLM post-training

Setup. We conduct RLVR experiments on Qwen3-4B-Base with GRPO (Shao et al., 2024), using the DAPO-17k-processed dataset with 1,000 problems held out for validation. We vary $\beta$ in SGD with bias-corrected EMA and, to support our claim in a more practical setting, also study AdamW by varying $\beta _ { 1 }$ while fixing $\beta _ { 2 } = 0 . 9 9 9$ Setting $\beta = 0$ recovers vanilla SGD, while $\beta _ { 1 } = 0$ gives an RMSProp-style optimizer. All configurations share the same data and evaluation protocol. We use three training generation seeds per configuration to show that our findings are consistent. Details of the experimental setup are provided in Appendix I.2.

Benefits of larger momentum coeficients. For SGD with EMA, we compare $\beta \in \{ 0 , 0 . 9 , 0 . 9 9 9 2 \}$ at $\eta = 2 \times 1 0 ^ { - 2 } \ \mathrm { ( F i g u r e \ 2 ( a ) ) }$ . We choose $\beta = 0 . 9 9 9 2$ so that $\eta / ( 1 - \beta )$ equals the upper bound $1 / ( 4 \gamma )$ suggested by Theorem 5.4. For AdamW, we compare $\beta _ { 1 } \in \{ 0 , 0 . 9 , 1 - 1 0 ^ { - 6 } \}$ at $\eta = 1 \times 1 0 ^ { - 6 }$ (Figure 2(b)). For both optimizers, the tuned larger momentum coeficient substantially improves validation performance over the conventional value of 0.9 and the baseline without first-moment momentum. These results support our claim that choosing momentum coeficients substantially above the conventional value of 0.9 can yield greater benefits in practical RLVR training

![](images/ac242f33d4ffb68b4cf6e8da923faddfb3a88e9e8b27e294a23355ab2ca18f00.jpg)  
(a) SGD with EMA.

![](images/2bb704ee2d87807a63803676b9d28f3160061a30a861b0cd93d5d343a31110dc.jpg)  
(b) AdamW.

![](images/ad1dc7316146b1d74d5df6853aec9c6f0a254e8170c0e609d04356bcc4ec6851.jpg)  
(c) SGD with EMA.

![](images/5c129af0499d5d2fe00a64614d5c9d8d9e1cb195ef63ca5790bd65b460ef3896.jpg)  
(d) AdamW.  
Figure 2: RLVR experiments on Qwen3-4B-Base with GRPO. The horizontal axis shows the training step. $2 ( \mathrm { a } )$ and 2(b) compare momentum coeficients at a fixed learning rate; lines show mean validation accuracy and error bars show $\pm 1$ sample standard deviation across three seeds. 2(c) and $2 ( \mathrm { d } )$ compare a smaller learning rate with a larger momentum coeficient against a larger learning rate with the conventional momentum coeficient; light curves show individual runs and dark curves their means.

More stable training without hurting performance. For SGD with EMA, we compare $( \eta , \beta ) \ = \ ( 1 \times 1 0 ^ { - 2 } , 0 . 9 9 9 6 )$ with $( 2 \times 1 0 ^ { - 2 } , 0 . 9 )$ and include $( 1 \times 1 0 ^ { - 2 } , 0 . 9 )$ as a baseline (Figure $2 ( \mathrm { c } ) )$ . Here, $\beta = 0 . 9 9 9 6$ is chosen to satisfy $\eta / ( 1 - \beta ) = 1 / ( 4 \gamma )$ . For AdamW, we compare $( \eta , \beta _ { 1 } ) = ( 5 \times 1 0 ^ { - 7 } , 1 - 1 0 ^ { - 6 } )$ with $( 1 \times 1 0 ^ { - 6 } , 0 . 9 )$ and include $( 5 \times 1 0 ^ { - 7 } , 0 . 9 )$ as a baseline $( { \mathrm { F i g u r e ~ 2 ( d ) } } )$ Under the same update and sample budgets, using a larger momentum coeficient at the smaller learning rate preserves the low variability in final validation accuracy across seeds while achieving slightly better performance than runs using the larger learning rate with conventional momentum. These results show that pairing a larger momentum coeficient with a smaller learning rate makes training more stable while maintaining or even slightly improving performance.

## 6.3 Extending to a larger model

We apply the same GRPO setup to Qwen2.5-32B

Figure 3: RLVR experiment on Qwen2.5-32B with AdamW. The horizontal axis shows the training step. Lines show means and error bars show ±1 sample standard deviation across three seeds.  
![](images/634d255344bba972fcb7131052571048264215218d8b188beebd882f4d55640d.jpg)

and increase the momentum coeficient in AdamW

from 0.9 to $1 - 1 0 ^ { - 9 }$ , using a learning rate of $2 \times 1 0 ^ { - 6 }$ . Simply increasing the momentum coeficient $\beta _ { 1 }$ substantially improves validation performance. At step 150, the conventional $\beta _ { 1 } = 0 . 9$ achieves a mean accuracy of 44.3%, similar to 44.2% for the baseline without first-moment momentum, whereas $\beta _ { 1 } = 1 - 1 0 ^ { - 9 }$ reaches 46.1%, nearly 2 percentage points higher than the conventional setting. Results are shown in Figure 3.

## 7 Conclusion

In this work, we explain why momentum’s benefit can be obscured by sampling noise and derive a principle for recovering and exploiting it. We develop a theoretical model and prove that EMA strictly improves the objective value under exact gradients within a range that prevents overshoot. With sampled gradients, fixing β as η decreases makes the expected gain vanish faster than sampling fluctuations, whereas suitably increasing β within the safe regime produces a consistent improvement. GRPO experiments on LLMs confirm the practical benefits of larger momentum coeficients. These findings suggest using coeficients substantially above the conventional $\beta = 0 . 9$ to better exploit the first-moment information already maintained by the optimizer.

## AI use statement

We used generative AI to assist with some mathematical derivations, literature searches, proof checking, and language editing. The authors carefully reviewed all AI-assisted material and take full responsibility for this paper.

## References

Agarwal, A., Kakade, S. M., Lee, J. D. and Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22 1–76.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I. and Amodei, D. (2020). Language models are few-shot learners. In Advances in Neural Information Processing Systems, vol. 33.

Cai, Q., Yang, Z., Jin, C. and Wang, Z. (2020). Provably eficient exploration in policy optimization. In International Conference on Machine Learning. PMLR.

Cen, S., Cheng, C., Chen, Y., Wei, Y. and Chi, Y. (2022). Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research, 70 2563–2578.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y. and Le, Q. V. (2023). Symbolic discovery of optimization algorithms. In Advances in Neural Information Processing Systems, vol. 36.

Chen, Y.-J., Huang, N.-C., Lee, C.-P. and Hsieh, P.-C. (2024). Accelerated policy gradient: On the convergence rates of the Nesterov momentum for reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q. et al. (2025). DeepSeek-R1: Incen tivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Ding, Y., Zhang, J. and Lavaei, J. (2022). On the global optimum convergence of momentum-based policy gradient. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, vol. 151 of Proceedings of Machine Learning Research.

Douc, R., Moulines, E., Priouret, P. and Soulier, P. (2018). Markov Chains. Springer Series in Operations Research and Financial Engineering, Springer, Cham.

Fan, C., Liu, G., Hong, M., Kompella, R. R. and Liu, S. (2026). Rethinking Muon beyond pretraining: Spectral failures and high-pass remedies for VLA and RLVR. arXiv preprint arXiv:2605.19282.

Feng, J., Wei, K. and Chen, J. (2024). Global convergence of natural policy gradient with hessian-aided momentum variance reduction. Journal of Scientific Computing, 101 44.

Ferbach, D., Paquette, C., Gidel, G., Everett, K. and Paquette, E. (2026). Logarithmic-time schedules for scaling language models with momentum. arXiv preprint arXiv:2602.05298.

Fontana, A., Simoni, M., Rossolini, G., Mori, P. and Saracino, A. (2026). On the hidden objective biases of group-based reinforcement learning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Association for Computational Linguistics.

Gai, J., Zeng, G., Zhang, H., Zhong, H., Hong, Y., Risteski, A. and Raghunathan, A. (2026). Demystifying entropy control in LLM RL training: Theoretical analysis and dynamic scheduling. In Proceedings of the 43rd International Conference on Machine Learning, vol. 306 of Proceedings of Machine Learning Research.

Ghadimi, E., Feyzmahdavian, H. R. and Johansson, M. (2015). Global convergence of the heavy-ball method for convex optimization. In 2015 European control conference (ECC). IEEE.

Huang, F., Gao, S. and Huang, H. (2022). Bregman gradient policy optimization. In International Conference on Learning Representations.

Huang, F., Gao, S., Pei, J. and Huang, H. (2020). Momentum-based policy gradient methods. In Proceedings of the 37th International Conference on Machine Learning, vol. 119 of Proceedings of Machine Learning Research.

Kakade, S. M. (2001). A natural policy gradient. Advances in neural information processing systems, 14.

Kingma, D. P. and Ba, J. L. (2015). Adam: A method for stochastic optimization. In International Conference on Learning Representations.

Kroese, D. P. and Botev, Z. I. (2023). An Advanced Course in Probability and Stochastic Processes. Chapman & Hall/CRC.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F. et al. (2024). T¨ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations.

Mukherjee, S., Yuan, L., Jayasinha, P., Hakkani-T¨ur, D. and Peng, H. (2026). Do we need adam? surprisingly strong and sparse reinforcement learning with SGD in LLMs. In Forty-third International Conference on Machine Learning.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J. and Lowe, R. (2022). Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho and A. Oh, eds.), vol. 35.

Pagliardini, M., Ablin, P. and Grangier, D. (2025). The AdEMAMix optimizer: Better, faster, older. In International Conference on Learning Representations.

Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4 1–17.

Qwen, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B. et al. (2024). Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015). Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning, vol. 37 of Proceedings of Machine Learning Research.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shani, L., Efroni, Y. and Mannor, S. (2020). Adaptive trust region policy optimization: Global convergence and faster rates for regularized MDPs. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y. and Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Shi, K., Li, H., Qiu, Z., Wen, Y., Buchholz, S. and Liu, W. (2026). Pion: A spectrum-preserving optimizer via orthogonal equivalence transformation. arXiv preprint arXiv:2605.12492.

Venkatkrishna, V., Daheim, N. and Gurevych, I. (2026). Parameter exploration for RLVR via variational learning. arXiv preprint arXiv:2608.09805.

Wang, R., Malladi, S., Wang, T., Lyu, K. and Li, Z. (2024). The marginal value of momentum for small learning rate SGD. In International Conference on Learning Representations.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N. and Zhang, T. (2024). Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. In Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B. et al. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y. et al. (2025). DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, vol. 38.

Yuan, K., Ying, B. and Sayed, A. H. (2016). On the influence of momentum acceleration on online learning. Journal of Machine Learning Research, 17 1–66.

Zhang, H., Wang, B. and Chen, L. (2025). AdamS: Momentum itself can be a normalizer for LLM pretraining and post-training. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Zhao, R., Morwani, D., Brandfonbrener, D., Vyas, N. and Kakade, S. (2025). Deconstructing what makes a good optimizer for autoregressive language models. In International Conference on Learning Representations

Zhong, H., Shan, Z., Feng, G., Xiong, W., Cheng, X., Zhao, L., He, D., Bian, J. and Wang, L. (2025). DPO meets PPO: Reinforced token optimization for RLHF. In Proceedings of the 42nd International Conference on Machine Learning, vol. 267 of Proceedings of Machine Learning Research.

Zhong, H. and Zhang, T. (2023). A theoretical analysis of optimistic proximal policy optimization in linear markov decision processes. Advances in Neural Information Processing Systems, 36 73666–73690.

Zhu, H., Cong, W., Sha, Z., Mukherjee, S., Song, X., Gonz´alez-Mart´ınez, D., Wu, X., Tian, Y., Liu, S., Pan, D. Z. and Wang, Z. A. (2026). ISO: An RLVR-native optimization stack. arXiv preprint arXiv:2607.19331.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P. and Irving, G. (2019). Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

## A Proof of Proposition 3.1

Proof. We first identify the softmax geometry and then derive the exact and sampled natural gradients.

Policy Jacobian and Fisher matrix. Diferentiating the softmax parameterization, we have

$$
\frac { \partial \pi _ { \theta } ( y \mid x ) } { \partial \theta ( x , y ^ { \prime } ) } = \pi _ { \theta } ( y \mid x ) \big ( { \bf 1 } _ { \{ y = y ^ { \prime } \} } - \pi _ { \theta } ( y ^ { \prime } \mid x ) \big ) , \quad \nabla \log \pi _ { \theta } ( y \mid x ) = { \bf e } _ { y } - \pi _ { \theta } ( \cdot \mid x ) ,
$$

where $\mathbf { e } _ { y }$ is the one-hot vector corresponding to $y \in \mathcal { V }$ . Substituting the second identity into the definition of the Fisher information matrix in (3.2) and comparing with the first gives

$$
\mathcal { F } ( \theta ) = \mathrm { d i a g } \big ( \pi _ { \theta } ( \cdot  { | } \ x ) \big ) - \pi _ { \theta } ( \cdot  { | } \ x ) \pi _ { \theta } ( \cdot  { | } \ x ) ^ { \top } = \frac { \partial \pi _ { \theta } ( \cdot  { | } \ x ) } { \partial \theta ^ { \top } } .\tag{A.1}
$$

Since the policy has full support, ${ \mathcal { F } } ( \theta )$ is positive semidefinite with null space span{1}. Hence $\mathcal { F } ^ { \dagger } ( \theta ) \mathcal { F } ( \theta ) =$ $\mathbf { I } - \mathbf { 1 1 } ^ { \top } / | \bar { \mathcal { V } } |$

Exact and sampled natural gradients. By the chain rule, diferentiating (3.1) gives

$$
\begin{array} { l } { \displaystyle \nabla J ( \theta ) = \Big ( \frac { \partial \pi _ { \theta } ( \cdot \mid x ) } { \partial \theta ^ { \top } } \Big ) ^ { \top } \mathbf { r } - \gamma \sum _ { y \in \mathcal { V } } \Big ( \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + 1 \Big ) \nabla \pi _ { \theta } ( y \mid x ) } \\ { \displaystyle = \mathcal { F } ( \theta ) \left[ \mathbf { r } - \gamma \left( \log \pi _ { \theta } ( \cdot \mid x ) - \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) + 1 \right) \right] } \\ { \displaystyle = \mathcal { F } ( \theta ) \left[ \mathbf { r } - \gamma \big ( \theta - \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \big ) \right] . } \end{array}\tag{A.2}
$$

The second equality follows from (A.1). The last uses log $\begin{array} { r } { \pi _ { \boldsymbol { \theta } } ( \cdot  { \mid } x ) = \boldsymbol { \theta } - \log ( \sum _ { y \in \mathcal { Y } } \exp ( \boldsymbol { \theta } ( x , y ) ) ) \mathbf { 1 } } \end{array}$ and $\mathcal { F } ( \boldsymbol { \theta } ) \mathbf { 1 } = 0$ . For the sampled gradient, the KL term is evaluated exactly, so

$$
\begin{array} { c l } { \displaystyle \widehat { \nabla J } ( \widehat \theta ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } r ( x , y _ { b } ) \nabla \log \pi _ { \widehat { \theta } } ( y _ { b } \mid x ) - \gamma \nabla \operatorname { K L } \left( \pi _ { \widehat { \theta } } ( \cdot \mid x ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \right) } \\ { \displaystyle = \mathcal { F } ( \widehat \theta ) \Big [ \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { r ( x , y _ { b } ) } { \pi _ { \widehat { \theta } } ( y _ { b } \mid x ) } \mathbf { e } _ { y _ { b } } - \gamma \big ( \widehat \theta - \log \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \big ) \Big ] . } \end{array}\tag{A.3}
$$

The second equality uses $\mathcal { F } ( \widehat { \theta } ) \mathbf { e } _ { y _ { b } } / \pi _ { \widehat { \theta } } ( y _ { b } \mid x ) = \mathbf { e } _ { y _ { b } } - \pi _ { \widehat { \theta } } ( \cdot \mid x )$ . Premultiplying (A.2) and (A.3) by the bcorresponding Fisher pseudoinverse proves (3.3) and (3.4). □

## B Proofs of Lemma 4.1 and Theorem 4.2

Optimality of $\theta ^ { \star }$ . The definition of $\theta ^ { \star }$ gives $\pi _ { \theta ^ { \star } } ( y \mid x ) \propto \pi _ { \mathrm { r e f } } ( y \mid x ) e ^ { r ( x , y ) / \gamma }$ . Substituting this identity into (3.1), we obtain

$$
J ( \theta ^ { \star } ) - J ( \theta ) = \gamma \mathrm { K L } \left( \pi _ { \theta } ( \cdot  { | } x ) \| \pi _ { \theta ^ { \star } } ( \cdot  { | } x ) \right) \geq 0 .
$$

Equality holds exactly when $\pi _ { \theta } = \pi _ { \theta ^ { \star } }$ , which verifies the unique optimal policy used in Section 4.

Proof of Lemma 4.1. Substituting the definition of $\theta ^ { \star }$ into (3.3) gives

$$
\mathcal { F } ^ { \dagger } ( \theta _ { t } ^ { \beta } ) \nabla J ( \theta _ { t } ^ { \beta } ) = \gamma \bigg ( \mathbf { I } - \frac { \mathbf { 1 1 } ^ { \top } } { \vert \mathcal { V } \vert } \bigg ) ( \theta ^ { \star } - \theta _ { t } ^ { \beta } ) .
$$

By Proposition 3.1, every natural-gradient direction has coordinate sum zero. Starting from $m _ { 0 } = 0$ , the EMA update therefore gives $\mathbf { 1 } ^ { \top } m _ { t } = 0$ at every step, so the parameter update preserves $\mathbf { 1 } ^ { \top } \theta _ { t } ^ { \beta } = \mathbf { 1 } ^ { \top } \theta _ { 0 }$ . By construction, $\mathbf { 1 } ^ { \top } \theta ^ { \star } = \mathbf { 1 } ^ { \top } \theta _ { 0 }$ as well. Thus $\mathbf { 1 } ^ { \top } ( \theta ^ { \star } - \theta _ { t } ^ { \beta } ) = 0 .$ , and the preceding identity simplifies to

$$
\mathcal { F } ^ { \dagger } ( \theta _ { t } ^ { \beta } ) \nabla J ( \theta _ { t } ^ { \beta } ) = \gamma ( \theta ^ { \star } - \theta _ { t } ^ { \beta } ) .\tag{B.1}
$$

Scalar residual dynamics. We prove the parameter identity by induction. It holds at $t = 0$ since $f _ { 0 } ( \beta ) = 1$ If it holds through iteration $t ,$ unrolling (3.5) yields

$$
\begin{array} { l } { { \displaystyle \theta _ { t + 1 } ^ { \beta } - \theta ^ { \star } = \theta _ { t } ^ { \beta } - \theta ^ { \star } - \eta \gamma \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) ( \theta _ { t ^ { \prime } } ^ { \beta } - \theta ^ { \star } ) } } \\ { { \displaystyle \qquad = \left[ f _ { t } ( \beta ) - \eta \gamma \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) f _ { t ^ { \prime } } ( \beta ) \right] ( \theta _ { 0 } - \theta ^ { \star } ) = f _ { t + 1 } ( \beta ) ( \theta _ { 0 } - \theta ^ { \star } ) . } } \end{array}
$$

The last equality is the defining recurrence of $f _ { t } .$ Substituting the parameter identity into (B.1) proves the natural-gradient identity. □

Proof of Theorem 4.2. We first establish positivity of $f _ { t } ( \beta )$ in the safe range and show that it decreases with the EMA coeficient for $t \geq 2$ . We then use this residual comparison to prove strict improvement of the objective.

Positive residuals in the safe range. We first prove that $f _ { t } ( \beta ) > 0$ for all $t \geq 0$ when $0 < \eta \gamma < 1$ and $0 \leq \beta \leq ( ( 1 - \eta \gamma ) / ( 1 + \eta \gamma ) ) ^ { 2 }$ . At $\beta = 0 , f _ { t } ( 0 ) = ( 1 - \eta \gamma ) ^ { t } > 0 ,$ , so it remains to consider $\beta > 0$ . Separating the current term in the defining recurrence of $f _ { t }$ and substituting the preceding update gives, for $t \geq 1$ 1

$$
\begin{array} { l } { { f _ { t + 1 } ( \beta ) = f _ { t } ( \beta ) - \displaystyle \frac { \eta \gamma } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } \Big [ f _ { t } ( \beta ) + \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t - 1 - t ^ { \prime } } f _ { t ^ { \prime } } ( \beta ) \Big ] } } \\ { { = f _ { t } ( \beta ) - \displaystyle \frac { \eta \gamma } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } f _ { t } ( \beta ) + \displaystyle \frac { \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t ^ { \prime } } } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } \big [ f _ { t } ( \beta ) - f _ { t - 1 } ( \beta ) \big ] } } \\ { { = ( 1 - \eta \gamma ) f _ { t } ( \beta ) + \displaystyle \frac { \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t ^ { \prime } } } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } \big [ ( 1 + \eta \gamma ) f _ { t } ( \beta ) - f _ { t - 1 } ( \beta ) \big ] . } } \end{array}
$$

For $\beta > 0$ , we prove positivity and $\sqrt { \beta } \le f _ { t + 1 } ( \beta ) / f _ { t } ( \beta ) \le 1 - \eta \gamma$ simultaneously by induction. Division gives

$$
\frac { f _ { t + 1 } ( \beta ) } { f _ { t } ( \beta ) } = 1 - \eta \gamma + \frac { \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t ^ { \prime } } } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } \biggl ( 1 + \eta \gamma - \frac { f _ { t - 1 } ( \beta ) } { f _ { t } ( \beta ) } \biggr ) .\tag{B.2}
$$

The base case holds because $f _ { 0 } ( \beta ) = 1 , f _ { 1 } ( \beta ) = 1 - \eta \gamma > 0$ , and $f _ { 1 } ( \beta ) / f _ { 0 } ( \beta ) = 1 - \eta \gamma \ge \sqrt { \beta } .$ . Assume $f _ { t } ( \beta ) > 0$ and $\sqrt { \beta } \le f _ { t } ( \beta ) / f _ { t - 1 } ( \beta ) \le 1 - \eta \gamma$ for some $t \geq 1$ . The coeficient satisfies $\begin{array} { r } { 0 \leq \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t ^ { \prime } } / \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } < \beta } \end{array}$ and the upper bound on $f _ { t } ( \beta ) / f _ { t - 1 } ( \beta )$ gives $1 + \eta \gamma - f _ { t - 1 } ( \beta ) / f _ { t } ( \beta ) \le - ( \eta \gamma ) ^ { 2 } / ( 1 - \eta \gamma ) < 0 .$ . Thus (B.2) gives $f _ { t + 1 } ( \beta ) / f _ { t } ( \beta ) < 1 - \eta \gamma$ . For the lower bound,

$$
\begin{array} { r l r } {  { \frac { f _ { t + 1 } ( \beta ) } { f _ { t } ( \beta ) } - \sqrt { \beta } \geq 1 - \eta \gamma + \beta \bigg ( 1 + \eta \gamma - \frac { 1 } { \sqrt { \beta } } \bigg ) - \sqrt { \beta } } } \\ & { } & { = ( 1 - \sqrt { \beta } ) \big [ ( 1 - \eta \gamma ) - ( 1 + \eta \gamma ) \sqrt { \beta } \big ] \geq 0 . } \end{array}
$$

The last inequality follows from the assumed range of $\beta .$ Consequently $f _ { t + 1 } ( \beta ) \geq \sqrt { \beta } f _ { t } ( \beta ) > 0$

Strict decrease with the EMA coeficient. Diferentiating (B.2) with respect to $\beta$ gives

$$
\partial _ { \beta } \bigg ( \frac { f _ { t + 1 } ( \beta ) } { f _ { t } ( \beta ) } \bigg ) = \frac { \beta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t ^ { \prime } } } { \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } } \bigg ( \frac { f _ { t - 1 } ( \beta ) } { f _ { t } ( \beta ) } \bigg ) ^ { 2 } \partial _ { \beta } \bigg ( \frac { f _ { t } ( \beta ) } { f _ { t - 1 } ( \beta ) } \bigg ) + \frac { \sum _ { t ^ { \prime } = 1 } ^ { t } t ^ { \prime } \beta ^ { t ^ { \prime } - 1 } } { ( \sum _ { t ^ { \prime } = 0 } ^ { t } \beta ^ { t ^ { \prime } } ) ^ { 2 } } \bigg ( 1 + \eta \gamma - \frac { f _ { t - 1 } ( \beta ) } { f _ { t } ( \beta ) } \bigg ) .
$$

The first term has the same sign as $\partial _ { \beta } ( f _ { t } ( \beta ) / f _ { t - 1 } ( \beta ) )$ for $\beta > 0$ . The second term is strictly negative for $t \geq 1$ . Starting from $\partial _ { \beta } \big ( f _ { 1 } ( \beta ) / f _ { 0 } ( \beta ) \big ) = \partial _ { \beta } \big ( 1 - \eta \gamma \big ) = 0 ,$ , induction therefore gives $\partial _ { \beta } ( f _ { t + 1 } ( \beta ) / f _ { t } ( \beta ) ) < 0$ Diferentiating $\begin{array} { r } { f _ { T } ( \beta ) = \prod _ { t = 1 } ^ { T } f _ { t } ( \beta ) / f _ { t - 1 } ( \beta ) } \end{array}$ gives, for all $T \geq 2$

$$
\partial _ { \beta } f _ { T } ( \beta ) = f _ { T } ( \beta ) \sum _ { t = 1 } ^ { T } \frac { f _ { t - 1 } ( \beta ) } { f _ { t } ( \beta ) } \partial _ { \beta } \Big ( \frac { f _ { t } ( \beta ) } { f _ { t - 1 } ( \beta ) } \Big ) < 0 .
$$

Thus, for $0 < \beta _ { 1 } < \beta _ { 2 }$ , we have $0 < f _ { T } ( \beta _ { 2 } ) < f _ { T } ( \beta _ { 1 } ) < ( 1 - \eta \gamma ) ^ { T } = f _ { T } ( 0 )$ . The last inequality follows from multiplying the upper bound of $f _ { t + 1 } ( \beta ) / f _ { t } ( \beta )$

Strict improvement of the objective. We now convert the residual comparison into an objective comparison. By (A.2), for $a > 0$

$$
\frac { \mathrm { d } } { \mathrm { d } a } J ( \theta ^ { \star } + a ( \theta _ { 0 } - \theta ^ { \star } ) ) = - \gamma a \mathrm { V a r } _ { y \sim \pi \theta ^ { \star } + a ( \theta _ { 0 } - \theta ^ { \star } ) } ( \cdot \vert x )  [ \theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y ) ] < 0 .\tag{B.3}
$$

The variance is positive since the policy has full support and $\theta _ { 0 } \neq \theta ^ { \star }$ . Lemma 4.1 and $0 < f _ { T } ( \beta _ { 2 } ) < f _ { T } ( \beta _ { 1 } )$ therefore give $\bar { J ( \theta _ { T } ^ { \beta _ { 2 } } ) } > J ( \theta _ { T } ^ { \beta _ { 1 } } )$ . □

## C Proof of Lemma 5.1

Proof. We first construct an auxiliary process with bounded noise and control its exit probability and moments. We then use these estimates to derive the objective expansion for the auxiliary process and transfer it to the original process.

Sampling noise and its propagation. Let $y _ { t , b } ^ { \beta }$ be the b-th response at update t. Conditional on the sample history, the batch consists of independent responses from $\pi _ { \widehat { \theta } _ { t } ^ { \beta } } ( \cdot \mid x )$ . Let $\varepsilon _ { t , b } ^ { \beta }$ denote the estimation noise from the b-th response at update t. Then we have

$$
\varepsilon _ { t , b } ^ { \beta } = \left( \mathbf { I } - \frac { \mathbf { 1 1 } ^ { \top } } { | \mathcal { V } | } \right) \left[ \frac { r ( x , y _ { t , b } ^ { \beta } ) } { \pi _ { \widehat { \theta } _ { t } ^ { \beta } } ( y _ { t , b } ^ { \beta } \mid x ) } \mathbf { e } _ { y _ { t , b } ^ { \beta } } - \mathbf { r } \right] , \qquad \varepsilon _ { t } ^ { \beta } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \varepsilon _ { t , b } ^ { \beta } .\tag{C.1}
$$

Since the sampled directions have zero coordinate sum (3.4) and the initial momentum is zero, the updates preserve $\mathbf { 1 } ^ { \top } \widehat { \theta } _ { t } ^ { \beta } = \mathbf { 1 } ^ { \top } \theta _ { 0 } = \mathbf { 1 } ^ { \top } \theta ^ { \star }$ . Therefore, the sampled direction at update t in Proposition 3.1 can be written as $\gamma ( \theta ^ { \star } - \widehat { \theta } _ { t } ^ { \beta } ) + \varepsilon _ { t } ^ { \beta }$ . Each $\varepsilon _ { t , b } ^ { \beta }$ has conditional mean zero, so the expected direction is $\gamma ( \theta ^ { \star } - \mathbb { E } [ \widehat { \theta } _ { t } ^ { \beta } ] )$ . Thus, after taking expectations, the sampled EMA updates obey the same recurrence as the exact updates. With the same initial parameter and zero initial momentum, this gives $\mathbb { E } [ \widehat { \theta } _ { t } ^ { \beta } ] = \theta _ { t } ^ { \beta }$ by induction.

Let $\Psi _ { \beta } ( t , t ^ { \prime } )$ denote the weight of the noise $\varepsilon _ { t ^ { \prime } } ^ { \beta }$ in the diference between the sampled and exact parameters, so that $\begin{array} { r } { \widehat { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } = \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \Psi _ { \beta } ( t , t ^ { \prime } ) \varepsilon _ { t ^ { \prime } } ^ { \beta } } \end{array}$ . Subtracting the exact EMA update from the sampled EMA update gives

$$
\widehat { \theta } _ { t + 1 } ^ { \beta } - \theta _ { t + 1 } ^ { \beta } = \widehat { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } + \eta \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) \big [ - \gamma ( \widehat { \theta } _ { t ^ { \prime } } ^ { \beta } - \theta _ { t ^ { \prime } } ^ { \beta } ) + \varepsilon _ { t ^ { \prime } } ^ { \beta } \big ] .\tag{C.2}
$$

The noise at update $t ^ { \prime }$ first afects the parameter at time $t ^ { \prime } + 1$ , so $\Psi _ { \beta } ( t , t ^ { \prime } ) = 0$ for $t \leq t ^ { \prime }$ . Substituting the noise expansion above into (C.2) gives for all $t \geq t ^ { \prime }$

$$
\Psi _ { \beta } ( t + 1 , t ^ { \prime } ) = \Psi _ { \beta } ( t , t ^ { \prime } ) - \gamma \eta \sum _ { k = t ^ { \prime } } ^ { t } \omega _ { \beta } ( t , k ) \Psi _ { \beta } ( k , t ^ { \prime } ) + \eta \omega _ { \beta } ( t , t ^ { \prime } ) .\tag{C.3}
$$

The last term captures the contribution of $\varepsilon _ { t ^ { \prime } } ^ { \beta }$ retained in the EMA memory. The sum captures how this noise changes later parameter values and hence the exact natural-gradient directions evaluated at those values. According to (C.3), $\Psi _ { \beta } ( t , t ^ { \prime } )$ is a deterministic function of $\eta , \gamma , \beta , t , t ^ { \prime }$ and does not depend on the realized samples. The covariance coeficient in Lemma 5.1 is

$$
\Sigma _ { T } ^ { \beta } = \frac { 1 } { \eta } \sum _ { t = 0 } ^ { T - 1 } \Psi _ { \beta } ( T , t ) ^ { 2 } \left[ \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { \beta } } ( \cdot \cdot \mid x ) } \right) - \mathbf { r } \mathbf { r } ^ { \top } \right] .\tag{C.4}
$$

The division inside diag is elementwise. The bracket is the covariance of $r ( x , y ) \mathbf { e } _ { y } / \pi _ { \theta _ { \star } ^ { \beta } } ( y \mid x )$ under $y \sim \pi _ { \theta _ { + } ^ { \beta } } ( \cdot |$ x), using $r ( x , y ) ^ { 2 } = r ( x , y )$ . Thus $\Sigma _ { T } ^ { \beta }$ is deterministic and positive semidefinite. The next lemma states the bounds on $\Psi _ { \beta } ( t , t ^ { \prime } )$ and $f _ { t } ( \beta )$ needed in the subsequent proofs. The proof is delayed to Section C.1.

Lemma C.1 (Uniform response bounds). Let $f _ { t }$ be defined in Section $\it 4$ and $\Psi _ { \beta } ( t , t ^ { \prime } )$ be defined by (C.3). For $\beta \in [ 0 , 1 )$ and integers $t \geq 0$ with $t \eta \leq \tau ,$ we have $| f _ { t } ( \beta ) | \le e ^ { \gamma \tau }$ . The response coeficients satisfy the following bounds, where the first inequality holds for all $t ^ { \prime } = 0 , \ldots , t - 1$

$$
\vert \Psi _ { \beta } ( t , t ^ { \prime } ) \vert \leq e ^ { \gamma \tau } \eta \biggl ( 1 + \sum _ { k = t ^ { \prime } + 1 } ^ { t } \frac { 1 } { k } \biggr ) , \quad \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \vert \Psi _ { \beta } ( t , t ^ { \prime } ) \vert \leq e ^ { \gamma \tau } \tau , \quad \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \Psi _ { \beta } ( t , t ^ { \prime } ) ^ { 2 } \leq 5 e ^ { 2 \gamma \tau } t \eta ^ { 2 } .\tag{C.5}
$$

The auxiliary process. We construct a process with bounded noise that agrees with the sampled process until it leaves a unit neighborhood of the exact path. Use the same noise sequence as the original process and initialize $\widetilde { \theta } _ { 0 } ^ { \beta } = \theta _ { 0 }$ . As in the main text, define

$$
\widetilde { \varepsilon } _ { t } ^ { \beta } = \mathbf { 1 } \Bigg \{ \operatorname* { m a x } _ { t ^ { \prime } \leq t } \| \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } - \theta _ { t ^ { \prime } } ^ { \beta } \| _ { \infty } \leq 1 \Bigg \} \varepsilon _ { t } ^ { \beta } , \qquad \widetilde { \theta } _ { t + 1 } ^ { \beta } = \widetilde { \theta } _ { t } ^ { \beta } + \eta \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) \big [ \mathcal { F } ^ { \dagger } ( \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } ) \nabla J ( \widetilde { \theta } _ { t ^ { \prime } } ^ { \beta } ) + \widetilde { \varepsilon } _ { t ^ { \prime } } ^ { \beta } \big ] .\tag{C.6}
$$

For the bounds below, let $\zeta ^ { \beta } = \operatorname* { i n f } \{ t \geq 1 : \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { \infty } > 1 \}$ denote the first exit time, taking $\zeta ^ { \beta } = \infty$ if no exit occurs. The two processes agree before exit; afterward, only fresh noise is suppressed. Since the stopping indicator is known before sampling, $\widetilde { \varepsilon } _ { t } ^ { \beta }$ has conditional mean zero, and taking expectations in the auxiliary updates yields $\mathbb { E } [ \widetilde { \theta } _ { t } ^ { \beta } ] = \theta _ { t } ^ { \beta }$

The bounds on probabilities and noise can be made explicit. Define $M = \| \theta ^ { \star } \| _ { \infty } + e ^ { \gamma \tau } \| \theta _ { 0 } - \theta ^ { \star } \| _ { \infty } + 1$ and $L = 2 ( | \mathcal { V } | e ^ { 2 M } + 1 )$ . Lemma 4.1 gives $\theta _ { t } ^ { \beta } = \theta ^ { \star } + f _ { t } ( \beta ) ( \theta _ { 0 } - \theta ^ { \star } )$ , and the bound $| f _ { t } ( \beta ) | \le e ^ { \gamma \tau }$ in Lemma C.1 holds for $t \leq T$ . On $\{ t < \zeta ^ { \beta } \}$ , the two processes agree and $\| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { \infty } \leq 1$ . Therefore,

$$
\| \widetilde { \theta } _ { t } ^ { \beta } \| _ { \infty } \leq \| \theta _ { t } ^ { \beta } \| _ { \infty } + 1 \leq \| \theta ^ { \star } \| _ { \infty } + | f _ { t } ( \beta ) | \cdot \| \theta _ { 0 } - \theta ^ { \star } \| _ { \infty } + 1 \leq M .
$$

Then we have $\pi _ { \widetilde { \theta } _ { \star } ^ { \beta } } ( y \mid x ) \ge 1 / ( | \mathcal { V } | \exp ( 2 M ) )$ . According to the definition of the sampling noise in (C.1), on $\{ t < \zeta ^ { \beta } \}$ e<sub>, we have</sub> $\| \varepsilon _ { t , b } ^ { \beta } \| _ { \infty } \leq 2 \big ( 1 / \pi _ { \widetilde { \theta } _ { \ast } ^ { \beta } } ( y _ { t , b } ^ { \beta } \mid x ) + 1 \big ) \leq 2 \big ( | \mathcal { V } | e ^ { 2 M } + 1 \big ) = L$ . Applying (C.3) to the auxiliary eprocess and using the second inequality in (C.5) gives for every sample path

$$
\operatorname* { m a x } _ { t \leq T } \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { \infty } \leq \operatorname* { m a x } _ { t \leq T } \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } | \Psi _ { \beta } ( t , t ^ { \prime } ) | \cdot \| \widetilde { \varepsilon } _ { t ^ { \prime } } ^ { \beta } \| _ { \infty } \leq L \tau e ^ { \gamma \tau } .
$$

In particular, the auxiliary parameters remain in a fixed bounded region even after exit.

Exit probability. We bound the probability that the auxiliary process exits the unit neighborhood of the exact trajectory by time $T ,$ . Fix $1 \leq t \leq T$ . The first coordinate diference satisfies $( \widetilde { \theta } _ { t } ^ { \widetilde { \beta } } ) _ { 1 } - ( \theta _ { t } ^ { \beta } ) _ { 1 } =$ $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \sum _ { b = 1 } ^ { B } \Psi _ { \beta } ( t , t ^ { \prime } ) { \bf 1 } _ { \{ t ^ { \prime } < \zeta ^ { \beta } \} } ( \varepsilon _ { t ^ { \prime } , b } ^ { \beta } ) _ { 1 } / B } \end{array}$ . The tB summands, ordered by $t ^ { \prime }$ and then $b ,$ have zero conditional mean given the preceding samples, so their partial sums form a martingale. Azuma-Hoefding’s inequality therefore gives, for every $u > 0$ ，

$$
\mathbb { P } \big ( | ( \widetilde { \theta } _ { t } ^ { \beta } ) _ { 1 } - ( \theta _ { t } ^ { \beta } ) _ { 1 } | > u \big ) \le 2 \exp \bigg [ - \frac { u ^ { 2 } } { 2 \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \sum _ { b = 1 } ^ { B } ( L | \Psi _ { \beta } ( t , t ^ { \prime } ) | / B ) ^ { 2 } } \bigg ] \le 2 \exp \bigg [ - \frac { B u ^ { 2 } } { 1 0 \tau L ^ { 2 } e ^ { 2 \gamma \tau } \eta } \bigg ] .\tag{C.7}
$$

The last inequality uses the third inequality in (C.5). The same bound holds for every coordinate. The two processes agree through the update causing the first exit, so taking $u = 1$ and a union bound over updates and coordinates gives

$$
\begin{array} { r } { \mathbb { P } ( \exists t \leq T : \widehat \theta _ { t } ^ { \beta } \neq \widetilde \theta _ { t } ^ { \beta } ) \leq \mathbb { P } ( \zeta ^ { \beta } \leq T ) \leq 2 | \mathcal { D } | T \exp ( - B / ( 1 0 \tau L ^ { 2 } e ^ { 2 \gamma \tau } \eta ) ) . } \end{array}\tag{C.8}
$$

Moment bounds. We next control the second and third moments. For the second moment, we have

$$
\mathbb { E } \Vert \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \Vert _ { 2 } ^ { 2 } = \frac { 1 } { B ^ { 2 } } \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \sum _ { b = 1 } ^ { B } \Psi _ { \beta } ( t , t ^ { \prime } ) ^ { 2 } \mathbb { E } \left[ \mathbf { 1 } _ { \{ t ^ { \prime } < \zeta ^ { \beta } \} } \Vert \varepsilon _ { t ^ { \prime } , b } ^ { \beta } \Vert _ { 2 } ^ { 2 } \right] \leq 5 \tau e ^ { 2 \gamma \tau } | \mathcal { V } | L ^ { 2 } \frac { \eta } { B } .\tag{C.9}
$$

In the equality, cross terms vanish in expectation since the noise has conditional mean zero. The inequality uses the third inequality in (C.5) and the bound $\| \varepsilon _ { t ^ { \prime } , b } ^ { \beta } \| _ { 2 } ^ { 2 } \leq | \mathcal { V } | L ^ { 2 }$ on $\{ t ^ { \prime } < \zeta ^ { \beta } \}$ . For the third moment,

the tail-integral formula gives $\begin{array} { r } { \mathbb { E } \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { 2 } ^ { 3 } = 3 \int _ { 0 } ^ { \infty } u ^ { 2 } \mathbb { P } \big ( \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { 2 } > u \big ) } \end{array}$ du. If $\lVert \widetilde { { \boldsymbol { \theta } } } _ { t } ^ { \beta } - { \boldsymbol { \theta } } _ { t } ^ { \beta } \rVert _ { 2 }$ exceeds $u ,$ some coordinate has magnitude above $u / \sqrt { | \mathcal { V } | }$ . Thus we can bound the probability in the integral by a union bound over coordinates and then apply (C.7):

$$
\operatorname* { m a x } _ { t \leq T } \mathbb { E } \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { 2 } ^ { 3 } \leq 6 | \mathcal { V } | \int _ { 0 } ^ { \infty } u ^ { 2 } \exp \bigg ( - \frac { B u ^ { 2 } } { 1 0 \tau | \mathcal { V } | L ^ { 2 } e ^ { 2 \gamma \tau } \eta } \bigg ) \mathrm { d } u = O ( ( \eta / B ) ^ { 3 / 2 } ) .\tag{C.10}
$$

Quadratic term in the expansion of $J ( \theta )$ . We replace the conditional noise covariances along the auxiliary path by those along the exact path, and control the resulting error. The covariance replacement error satisfies

$$
\begin{array} { r l } {  { \mathbb { E } \bigg \| \mathbf { 1 } _ { \{ t < \zeta ^ { \beta } \} } \bigg [ \mathrm { d i a g } ( \frac { \mathbf { r } } { \pi _ { \widetilde { \theta } _ { t } ^ { \beta } } ( \cdot \vert \ x ) } ) - \mathbf { r r } ^ { \top } \bigg ] - [ \mathrm { d i a g } ( \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { \beta } } ( \cdot \vert x ) } ) - \mathbf { r r } ^ { \top } ] \bigg \| _ { \mathrm { o p } } } } \\ & { \leq 2 | \mathcal { D } | e ^ { 2 M } \mathbb { E } \| \widetilde { \theta } _ { t } ^ { \beta } - \theta _ { t } ^ { \beta } \| _ { \infty } + | \mathcal { D } | ( e ^ { 2 M } + 1 ) \mathbb { P } ( \zeta ^ { \beta } \leq t ) } \\ & { \leq 2 | \mathcal { D } | e ^ { 2 M } \sqrt { 5 \tau e ^ { 2 \gamma \tau } | \mathcal { D } | L ^ { 2 } \eta / B } + | \mathcal { D } | ( e ^ { 2 M } + 1 ) \mathbb { P } ( \zeta ^ { \beta } \leq T ) } \\ & { = O \big ( \sqrt { \eta / B } + \mathbb { P } ( \zeta ^ { \beta } \leq T ) \big ) . } \end{array}\tag{C.11}
$$

(C.12)

For the first inequality, on $\{ t < \zeta ^ { \beta } \}$ the two $\mathbf { r } \mathbf { r } ^ { \top }$ terms cancel, and the operator norm is bounded by $\operatorname* { m a x } _ { \boldsymbol { y } } | \pi _ { \widetilde { \theta } _ { \star } ^ { \beta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) ^ { - 1 } - \pi _ { \theta _ { \star } ^ { \beta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) ^ { - 1 } | .$ . Both parameter vectors have infinity norm at most M. Applying the emean value theorem to the identity $\textstyle \pi _ { \boldsymbol { \theta } } ( y \mid x ) ^ { - 1 } = \sum _ { y ^ { \prime } } \exp ( \theta ( x , y ^ { \prime } ) - \theta ( x , y ) )$ gives the upper bound. On exit, only the exact-path covariance remains, with norm bounded by max $_ y 1 / \pi _ { \theta _ { \star } ^ { \beta } } ( y \mid x ) + \| \mathbf { r } \| _ { 2 } ^ { 2 }$ . The second inequality uses the Cauchy–Schwarz inequality and (C.9).

We then evaluate the quadratic term in the expansion of $J ( \theta )$ . Since $\widetilde { \varepsilon } _ { t } ^ { \beta }$ has conditional mean zero, cross terms between distinct updates vanish in expectation. Also, $\nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \mathbf { 1 } = 0$ , so the noise projection in (C.1) does not change the quadratic form. Consequently,

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( \widetilde { \theta } _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } ) ^ { \top } \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) ( \widetilde { \theta } _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } ) \big ] } \\ & { \quad = \displaystyle \frac { 1 } { B } \sum _ { t = 0 } ^ { T - 1 } \Psi _ { \beta } ( T , t ) ^ { 2 } \mathbb { E } \operatorname { t r } \left[ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \mathbf { 1 } _ { \left\{ t < \zeta ^ { \beta } \right\} } \left[ \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \widetilde { \theta } _ { t } ^ { \beta } } ( \cdot \ | \ x ) } \right) - \mathbf { r r } ^ { \top } \right] \right] } \\ & { \quad = \displaystyle \frac { \eta } { B } \operatorname { t r } \left[ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } \right] + O \left( \frac { \eta } { B } \big ( \sqrt { \eta / B } + \mathbb { P } ( \zeta ^ { \beta } \leq T ) \big ) \right) } \\ & { \quad = \displaystyle \frac { \eta } { B } \operatorname { t r } \left[ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } \right] + O ( ( \eta / B ) ^ { 3 / 2 } ) . } \end{array}\tag{C.13}
$$

The second equality replaces the covariance matrices by using the definition of $\Sigma _ { T } ^ { \beta }$ in (C.4), the replacement error in (C.12), and $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } \Psi _ { \beta } ( T , t ) ^ { 2 } \leq 5 \tau e ^ { 2 \gamma \tau } \eta } \end{array}$ from the third inequality in (C.5). The expected absolute error in each trace is bounded by $| \mathcal { V } | \cdot | | \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) | | _ { \mathrm { o p } }$ times the expected operator norm of the covariance diference in (C.12). Also, $\| \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \| _ { \mathrm { o p } }$ is bounded independently of $\eta$ and $\beta .$ . The last equality uses (C.8) to absorb the probability. According to (C.4), the third inequality in (C.5), and the covariance bound above, $\operatorname { s u p } _ { \beta \in [ 0 , 1 ) } \| \Sigma _ { T } ^ { \beta } \| _ { \mathrm { o p } }$ is bounded by $5 \tau e ^ { 2 \gamma \tau } | \mathcal { V } | ( e ^ { 2 M } + 1 )$ uniformly in $\eta .$ . Together with the Hessian bound, this implies that the quadratic correction is ${ \cal O } ( \eta / B )$ uniformly in $\beta .$

Objective expansion and transfer. We combine the moment and covariance bounds in a Taylor expansion, then transfer the resulting expected objective to the sampled process. All derivatives of $J ( \theta )$ through order three are bounded on the region containing the exact and auxiliary paths. Taylor’s formula and the third-moment bound in (C.10) give an expected remainder bounded by

$$
\frac { 1 } 6 \operatorname* { s u p } _ { \| \vartheta \| _ { \infty } \leq M + L \tau e ^ { \gamma \tau } } \| \nabla ^ { 3 } J ( \vartheta ) \| _ { \mathrm { o p } } \mathbb { E } \| \widetilde \theta _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } \| _ { 2 } ^ { 3 } = O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

The bounded region includes every segment joining the exact and auxiliary endpoints. Using $\mathbb { E } [ \widetilde { \theta } _ { T } ^ { \beta } ] = \theta _ { T } ^ { \beta }$ for the linear term, (C.13) for the quadratic term, and (C.10) for the remainder, we obtain

$$
\mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] = J ( \theta _ { T } ^ { \beta } ) + \frac { \eta } { 2 B } \operatorname { t r } [ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } ] + O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

The expected objectives of the sampled and auxiliary processes difer by at most $\begin{array} { r } { 2 \operatorname* { s u p } _ { \theta } | J ( \theta ) | \operatorname { \mathbb { P } } ( \zeta ^ { \beta } \leq T ) = } \end{array}$ = $o \big ( ( \eta / B ) ^ { 3 / 2 } \big )$ . Here $- \gamma \log ( 1 / \operatorname* { m i n } _ { y } \pi _ { \mathrm { r e f } } ( y \mid x ) ) \le J ( \theta ) \le 1$ for every $\theta .$ . Combining this estimate with the expansion for the auxiliary process yields

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) ] = J ( \theta _ { T } ^ { \beta } ) + \frac { \eta } { 2 B } \operatorname { t r } [ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } ] + O ( ( \eta / B ) ^ { 3 / 2 } ) .\tag{C.14}
$$

Subtracting the expansions for $\beta$ and 0 proves the lemma.

## C.1 Proof of Lemma C.1

Proof. We bound the residual $f _ { t } ( \beta )$ and then establish the three noise-response bounds.

Bound of $f _ { t } ( \beta )$ . By the defining recurrence of $f _ { t } ( \beta )$ in Section 4, the triangle inequality gives

$$
| f _ { t + 1 } ( \beta ) | \leq | f _ { t } ( \beta ) | + \gamma \eta \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) | f _ { t ^ { \prime } } ( \beta ) | \leq ( 1 + \gamma \eta ) \operatorname* { m a x } _ { t ^ { \prime } \leq t } | f _ { t ^ { \prime } } ( \beta ) | ,
$$

where we use $\omega _ { \beta } ( t , t ^ { \prime } ) \geq 0$ and $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) = 1 } \end{array}$ . The same upper bound also covers all earlier $\left| f _ { t ^ { \prime } } ( \beta ) \right|$ , so max $t ^ { \prime } \leq t + 1 \left| f _ { t ^ { \prime } } ( \beta ) \right| \leq ( 1 + \gamma \eta )$ max $\mathop { : / } \le t \left| f _ { t ^ { \prime } } ( \beta ) \right|$ . With $f _ { 0 } ( \beta ) = 1$ , we have max $t ^ { \prime } \leq t | f _ { t ^ { \prime } } ( \beta ) | \leq ( 1 + \gamma \eta ) ^ { t } \leq e ^ { \gamma \eta t } \leq$ $e ^ { \gamma \tau }$ , which proves the bound $\left| f _ { t } ( \beta ) \right| \le e ^ { \gamma \tau }$ in Lemma C.1.

Proof of the first inequality in (C.5). Fixing t<sup>′</sup> and applying the triangle inequality to (C.3) yields $\begin{array} { r } { | \Psi _ { \beta } ( t + 1 , t ^ { \prime } ) | \leq ( 1 + \gamma \eta ) \operatorname* { m a x } _ { t ^ { \prime } \leq k \leq t } | \Psi _ { \beta } ( k , t ^ { \prime } ) | + \eta \omega _ { \beta } ( t , t ^ { \prime } ) } \end{array}$ . We use $\omega _ { \beta } ( t , k ) \geq 0$ and $\begin{array} { r } { \sum _ { k = t ^ { \prime } } ^ { t } \omega _ { \beta } ( t , k ) \le 1 } \end{array}$ . Similar to the previous argument, this also bounds $\mathrm { m a x } _ { t ^ { \prime } \leq k \leq t + 1 } | \Psi _ { \beta } ( k , t ^ { \prime } ) |$ . Starting from $\Psi _ { \beta } ( t ^ { \prime } , t ^ { \prime } ) = 0$ and iterating gives

$$
\begin{array} { l } { \displaystyle | \Psi _ { \beta } ( t , t ^ { \prime } ) | \leq \eta \displaystyle \sum _ { k = t ^ { \prime } } ^ { t - 1 } ( 1 + \gamma \eta ) ^ { t - 1 - k } \omega _ { \beta } ( k , t ^ { \prime } ) \leq e ^ { \gamma \tau } \eta \displaystyle \sum _ { k = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( k , t ^ { \prime } ) } \\ { = e ^ { \gamma \tau } \eta \displaystyle \sum _ { k = t ^ { \prime } } ^ { t - 1 } \bigg ( ( 1 - \beta ) \beta ^ { k - t ^ { \prime } } + \frac { \beta ^ { 2 k + 1 - t ^ { \prime } } } { \sum _ { \ell = 0 } ^ { k } \beta ^ { \ell } } \bigg ) } \\ { \leq e ^ { \gamma \tau } \eta \bigg ( 1 - \beta ^ { t - t ^ { \prime } } + \displaystyle \sum _ { k = t ^ { \prime } } ^ { t - 1 } \frac { \beta ^ { k + 1 - t ^ { \prime } } } { k + 1 } \bigg ) \leq e ^ { \gamma \tau } \eta \bigg ( 1 + \displaystyle \sum _ { k = t ^ { \prime } + 1 } ^ { t } \frac { 1 } { k } \bigg ) . } \end{array}\tag{C.15}
$$

In the first line, we use $( 1 + \gamma \eta ) ^ { t - 1 - k } \leq e ^ { \gamma \tau }$ . The second line is given by the definition of $\omega _ { \beta } ( \boldsymbol { k } , t ^ { \prime } )$ . In the last line, we use ${ \textstyle \sum _ { \ell = 0 } ^ { k } \beta ^ { \ell } \geq ( k + 1 ) \beta ^ { k } }$ . The same bound holds for $\beta = 0$ . Combining these inequalities proves the first inequality in (C.5).

Proof of the second inequality in (C.5). Using (C.15) and $\textstyle \sum _ { t ^ { \prime } = 0 } ^ { k } \omega _ { \beta } ( k , t ^ { \prime } ) = 1$ , we sum over $t ^ { \prime }$ and interchange the sums to obtain

$$
\sum _ { t ^ { \prime } = 0 } ^ { t - 1 } | \Psi _ { \beta } ( t , t ^ { \prime } ) | \leq e ^ { \gamma \tau } \eta \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \sum _ { k = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( k , t ^ { \prime } ) = e ^ { \gamma \tau } \eta \sum _ { k = 0 } ^ { t - 1 } \sum _ { t ^ { \prime } = 0 } ^ { k } \omega _ { \beta } ( k , t ^ { \prime } ) = e ^ { \gamma \tau } t \eta \leq e ^ { \gamma \tau } \tau .
$$

Proof of the third inequality in (C.5). Expanding gives $\textstyle \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \left( 1 + \sum _ { k = t ^ { \prime } + 1 } ^ { t } 1 / k \right) ^ { 2 } = 5 t - \sum _ { k = 1 } ^ { t } 1 / k \le 5 t$ Combining this identity with the first inequality in (C.5) proves the third inequality in (C.5) directly. □

## D Proof of Proposition 5.2

Proof. Apply the expansion proved in Appendix C to $J ( \theta ) ^ { 2 }$ . Since $\nabla ^ { 2 } ( J ( \theta ) ^ { 2 } ) = 2 J ( \theta ) \nabla ^ { 2 } J ( \theta ) + 2 \nabla J ( \theta ) \nabla J ( \theta ) ^ { \top }$ it gives

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) ^ { 2 } ] = J ( \theta _ { T } ^ { \beta } ) ^ { 2 } + \frac { \eta } { B } J ( \theta _ { T } ^ { \beta } ) \operatorname { t r } [ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } ] + \frac { \eta } { B } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( \theta _ { T } ^ { \beta } ) + O ( ( \eta / B ) ^ { 3 / 2 } ) .
$$

Subtracting the square of (C.14) cancels the Hessian term and yields, uniformly in $\beta ,$

$$
\mathrm { V a r } [ J ( \widehat \theta _ { T } ^ { \beta } ) ] = \frac { \eta } { B } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( \theta _ { T } ^ { \beta } ) + O ( ( \eta / B ) ^ { 3 / 2 } ) .\tag{D.1}
$$

Independent comparison. The two variances add by independence, proving the expansion in Proposition 5.2. Its two leading quadratic forms are nonnegative and uniformly bounded. Under the stated nondegeneracy condition, the no-EMA term is bounded away from zero as $\eta  0$ . Thus, Var $\big [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) \big ] = \Theta ( \bar { \eta / } B )$ . □

## E Proof of Theorem 5.3

Proof. For fixed $\beta ,$ we first expand the exact objective gain and compare the two sampling corrections in Lemma 5.1 to obtain the expected objective gain. We then combine asymptotic normality with the vanishing mean-to-standard-deviation ratio to derive the limiting winning probability.

Exact objective gain. We first state the required approximation for $f _ { t } ( \beta )$

Lemma E.1 (Approximation of $f _ { t } ( \beta ) )$ ). For fixed $\beta \in [ 0 , 1 )$ and $\tau > 0$ , the following approximation for $f _ { t }$ defined in Section $\it 4$ holds uniformly for integers $0 \leq t \leq T$ as $\eta = \tau / T \to 0$

$$
f _ { t } ( \beta ) = \left( 1 - \gamma \eta - \gamma ^ { 2 } \eta ^ { 2 } \frac { \beta } { 1 - \beta } \right) ^ { t } + O ( \eta ^ { 2 } ) .
$$

Together with $f _ { T } ( 0 ) = ( 1 - \gamma \eta ) ^ { T }$ , Lemma E.1 gives

$$
\begin{array} { c } { { f _ { T } ( 0 ) - f _ { T } ( \beta ) = ( 1 - \gamma \eta ) ^ { T } - \displaystyle \left( 1 - \gamma \eta - \gamma ^ { 2 } \eta ^ { 2 } \frac { \beta } { 1 - \beta } \right) ^ { T } + { \cal O } ( \eta ^ { 2 } ) } } \\ { { { } } } \\ { { = T \gamma ^ { 2 } \eta ^ { 2 } \displaystyle \frac { \beta } { 1 - \beta } ( 1 - \gamma \eta ) ^ { T - 1 } + { \cal O } ( T ^ { 2 } \eta ^ { 4 } + \eta ^ { 2 } ) } } \\ { { { } } } \\ { { { } = \gamma ^ { 2 } \tau e ^ { - \gamma \tau } \displaystyle \frac { \eta \beta } { 1 - \beta } + { \cal O } ( \eta ^ { 2 } ) . } } \end{array}
$$

The second line is Taylor’s formula for the power. The last line uses $T \eta = \tau$ and $( 1 - \gamma \eta ) ^ { T - 1 } = e ^ { - \gamma \tau } + O ( \eta )$ Lemma E.1 and $T \eta = \tau$ give $f _ { T } ( 0 ) = e ^ { - \gamma \tau } + O ( \eta )$ and $f _ { T } ( \beta ) = e ^ { - \gamma \tau } + O ( \eta )$ . To obtain the objective gain along the exact path, we integrate (B.3) from $a = f _ { T } ( \beta )$ to $a = f _ { T } ( 0 )$ , which gives

$$
\begin{array} { l } { { \displaystyle J ( \theta _ { T } ^ { \beta } ) - J ( \theta _ { T } ^ { 0 } ) = \gamma \int _ { f _ { T } ( \beta ) } ^ { f _ { T } ( 0 ) } a \mathrm { V a r } _ { y \sim \pi _ { \theta ^ { \star } + \alpha ( \theta _ { 0 } - \theta ^ { \star } ) } ( \cdot \vert x ) } [ \theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y ) ] \mathrm { d } a } } \\ { { \displaystyle \qquad = \gamma e ^ { - \gamma \tau } \mathrm { V a r } _ { y \sim \pi _ { \theta ^ { \star } + c ^ { - } \gamma \tau ( \theta _ { 0 } - \theta ^ { \star } ) } } [ \theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y ) ] [ f _ { T } ( 0 ) - f _ { T } ( \beta ) ] + O ( \eta ^ { 2 } ) } } \\ { { \displaystyle \qquad = C ( \tau , \gamma ) \frac { \eta \beta } { 1 - \beta } + O ( \eta ^ { 2 } ) . } } \end{array}\tag{E.1}
$$

The integrand varies by ${ \cal { O } } ( \eta )$ over an interval of length ${ \cal { O } } ( \eta )$ , which explains the second-line remainder. The constant in Theorem 5.3 is given explicitly by

$$
C ( \tau , \gamma ) = \gamma ^ { 3 } \tau e ^ { - 2 \gamma \tau } \mathrm { V a r } _ { y \sim \pi _ { \theta ^ { \star } + e ^ { - \gamma \tau } ( \theta _ { 0 } - \theta ^ { \star } ) } ( \cdot \vert x ) } [ \theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y ) ] .
$$

This expression specifies its dependence on the problem, $\tau ,$ and $\gamma .$ . It is independent of $\eta , \beta ,$ , and $B .$ . The coeficient is strictly positive under the nondegeneracy condition, which implies $\theta _ { 0 } \neq \theta ^ { \star }$

Diference between the quadratic terms. Using $f _ { t } ( 0 ) = ( 1 - \gamma \eta ) ^ { t }$ , Lemma E.1 and the mean value theorem give $| f _ { t } ( \beta ) - f _ { t } ( 0 ) | \le t \gamma ^ { 2 } \eta ^ { 2 } \beta / ( 1 - \beta ) + O ( \eta ^ { 2 } ) = O ( \eta )$ uniformly for $0 \leq t \leq T$ , since $t \eta \le \tau$ Lemma 4.1 then gives $\| \theta _ { t } ^ { \beta } - \theta _ { t } ^ { 0 } \| _ { \infty } = | f _ { t } ( \beta ) - f _ { t } ( 0 ) | \cdot \| \theta _ { 0 } - \theta ^ { \star } \| _ { \infty }$ , and hence max $\ L _ { t \leq T } \| \theta _ { t } ^ { \beta } - \theta _ { t } ^ { 0 } \| _ { \infty } = O ( \eta )$ . Since $J ( \theta )$ is smooth and the entries of $\theta _ { t } ^ { \beta }$ and $\theta _ { t } ^ { 0 }$ are bounded uniformly in $t \leq T$ and $\eta , \| \nabla ^ { 3 } J ( \theta ) \| _ { \mathrm { o p } }$ is bounded independently of η whenever the entries of θ satisfy the same bound. This gives a Lipschitz bound for $\nabla ^ { 2 } J ( \theta )$ with a constant independent of $\eta .$ Combining this with the preceding ${ \cal { O } } ( \eta )$ bound on the parameter diference gives $\| \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) - \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \| _ { \mathrm { o p } } = O ( \eta )$ . To compare the two sampling corrections in Lemma 5.1, we also need the following bound on $\Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 }$

Lemma E.2 (Covariance comparison). Let $\Sigma _ { T } ^ { \beta }$ be defined by (C.4). For fixed $\beta \in [ 0 , 1 )$ and $\tau > 0$ , as $\eta = \tau / T \to 0$ , we have $\| \Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 } \| _ { \mathrm { o p } } = O ( \eta )$

Combining this bound with the preceding Hessian estimate, we obtain

$$
\begin{array} { r l } & { \left| \mathrm { t r } [ \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) \Sigma _ { T } ^ { \beta } - \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \Sigma _ { T } ^ { 0 } ] \right| } \\ & { \quad \leq | \mathcal { X } | \cdot \| \nabla ^ { 2 } J ( \theta _ { T } ^ { \beta } ) - \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \| _ { \mathrm { o p } } \cdot \| \Sigma _ { T } ^ { \beta } \| _ { \mathrm { o p } } + | \mathcal { Y } | \cdot \| \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \| _ { \mathrm { o p } } \cdot \| \Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 } \| _ { \mathrm { o p } } = O ( \eta ) . } \end{array}
$$

Here we add and subtract $\nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \Sigma _ { T } ^ { \beta }$ inside the trace, then bound each trace in absolute value. The bounds in Appendix C give $\| \Sigma _ { T } ^ { \beta } \| _ { \mathrm { o p } } = O ( 1 )$ and $\| \nabla ^ { 2 } J ( \theta _ { T } ^ { 0 } ) \| _ { \mathrm { o p } } = O ( 1 )$ . Lemma 5.1 and (E.1) therefore yield

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] = C ( \tau , \gamma ) \frac { \eta \beta } { 1 - \beta } + O ( \eta ^ { 2 } ) + O ( \eta ^ { 2 } / B ) + O ( ( \eta / B ) ^ { 3 / 2 } ) } \\ { \displaystyle = C ( \tau , \gamma ) \frac { \eta \beta } { 1 - \beta } + O ( \eta ^ { 3 / 2 } ) . } \end{array}
$$

Probability of improvement. The mean-to-standard-deviation ratio alone does not determine the winning probability. We also need the following normal limit to characterize the distribution of the objective diference, which is proved in the next subsection.

Lemma E.3 (Asymptotic normality). Under the assumptions of Theorem 5.3, the distribution of the standardized objective diference converges to a standard normal distribution as $\eta  0$

$$
\frac { J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) - \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] } { \mathrm { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] } \overset { \mathrm { d } } {  } \mathcal { N } ( 0 , 1 ) .
$$

The mean expansion and Proposition 5.2 give $\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] / \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \to 0$ . Combining this estimate with the normal limit in Lemma E.3 gives $\mathbb { P } ( J ( \widehat { \theta } _ { T } ^ { \beta } ) > J ( \widehat { \theta } _ { T } ^ { 0 } ) )  1 / 2 .$ □

## E.1 Proofs of the auxiliary lemmas

Proof of Lemma E.1. The definition of $f _ { t } ( \beta )$ in Section 4 gives $\begin{array} { r } { f _ { t + 1 } ( \beta ) = ( 1 - \gamma \eta ) f _ { t } ( \beta ) - \gamma \eta ( \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) f _ { t ^ { \prime } } ( \beta ) - } \end{array}$ $f _ { t } ( \beta ) )$ . We express the weighted sum by $f _ { t } ( \beta )$ to obtain the desired approximation. The bound $| f _ { t } ( \beta ) | \le e ^ { \gamma \tau }$ in Lemma C.1 and the definition of $f _ { t } ( \beta )$ give $| f _ { t ^ { \prime } } ( \beta ) - f _ { t } ( \beta ) | \leq \gamma \eta ( t - t ^ { \prime } ) e ^ { \gamma \tau }$ for all $0 \leq t ^ { \prime } \leq t \leq T$ . For $0 \leq k \leq t .$ we therefore have

$$
\begin{array} { r l } & { \lvert f _ { t - k } ( \beta ) - f _ { t } ( \beta ) - \gamma \eta k f _ { t } ( \beta ) \rvert } \\ & { \quad = \gamma \eta \Big | \displaystyle \sum _ { \ell = t - k } ^ { t - 1 } \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { \ell } \omega _ { \beta } ( \ell , t ^ { \prime } ) \big ( f _ { t ^ { \prime } } ( \beta ) - f _ { t } ( \beta ) \big ) \Big | \le \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } \displaystyle \sum _ { \ell = t - k } ^ { t - 1 } \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { \ell } \omega _ { \beta } ( \ell , t ^ { \prime } ) ( t - t ^ { \prime } ) } \\ & { \quad \le \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } \displaystyle \sum _ { \ell = t - k } ^ { t - 1 } ( t - \ell + \beta / ( 1 - \beta ) ) = \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } ( k ( k + 1 ) / 2 + k \beta / ( 1 - \beta ) ) . } \end{array}
$$

The first equality follows by applying the definition of $f _ { t } ( \beta )$ to express $f _ { t - k } ( \beta ) \mathrm { ~ - ~ } f _ { t } ( \beta )$ and using $\Sigma _ { t ^ { \prime } = 0 } ^ { \ell } \omega _ { \beta } ( \ell , t ^ { \prime } ) = 1$ . For the second inequality, we write $t - t ^ { \prime } = ( t - \ell ) + ( \ell - t ^ { \prime } )$ and bound the weighted average of $\ell - t ^ { \prime }$ using $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { \ell } ( \ell - t ^ { \prime } ) \omega _ { \beta } ( \ell , t ^ { \prime } ) \leq \beta / ( 1 - \beta ) } \end{array}$ . Multiplying the resulting error bound by $\omega _ { \beta } ( t , t - k )$ and summing over $k \leq t \ \mathrm { g i }$ ives

$$
\begin{array} { r l } {  { \Big \vert \displaystyle \sum _ { k = 0 } ^ { t } \omega _ { \beta } ( t , t - k ) f _ { t - k } ( \beta ) - f _ { t } ( \beta ) - \frac { \gamma \eta \beta f _ { t } ( \beta ) } { 1 - \beta } \Big \vert } \quad } & { } \\ & { \leq \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } \displaystyle \sum _ { k = 0 } ^ { t } \omega _ { \beta } ( t , t - k ) \Big ( \frac { k ( k + 1 ) } { 2 } + \frac { k \beta } { 1 - \beta } \Big ) + \gamma \eta | f _ { t } ( \beta ) | \cdot \Big \vert \displaystyle \sum _ { k = 0 } ^ { t } k \omega _ { \beta } ( t , t - k ) - \frac { \beta } { 1 - \beta } \Big \vert } \\ & { \leq \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } \beta ( 1 + \beta ) / ( 1 - \beta ) ^ { 2 } + \gamma \eta e ^ { \gamma \tau } ( t + 1 ) \beta ^ { t + 1 } / ( 1 - \beta ) . } \end{array}
$$

For the second inequality, the definition of ω gives $\begin{array} { r } { 0 \le \beta / ( 1 - \beta ) - \sum _ { k = 0 } ^ { t } k \omega _ { \beta } ( t , t - k ) \le ( t + 1 ) \beta ^ { t + 1 } / ( 1 - \beta ) } \end{array}$ and $\begin{array} { r } { \sum _ { k = 0 } ^ { t } k ^ { 2 } \omega _ { \beta } ( t , t - k ) \leq \sum _ { k = 0 } ^ { \infty } k ^ { 2 } \beta ^ { k } / ( \sum _ { k = 0 } ^ { \infty } \beta ^ { k } ) = \beta ( 1 + \beta ) / ( 1 - \beta ) ^ { 2 } } \end{array}$ . We also use $| f _ { t } ( \beta ) | \le e ^ { \gamma \tau }$ in Lemma C.1. Substituting this bound into the definition of $f _ { t + 1 } ( \beta )$ yields, for $t < T$ 2

$$
\Big | f _ { t + 1 } ( \beta ) - \Big ( 1 - \gamma \eta - \frac { \gamma ^ { 2 } \eta ^ { 2 } \beta } { 1 - \beta } \Big ) f _ { t } ( \beta ) \Big | \leq \frac { \gamma ^ { 3 } \eta ^ { 3 } e ^ { \gamma \tau } \beta ( 1 + \beta ) } { ( 1 - \beta ) ^ { 2 } } + \frac { \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } ( t + 1 ) \beta ^ { t + 1 } } { 1 - \beta } .\tag{E.2}
$$

Repeatedly applying (E.2) from $f _ { 0 } ( \beta ) = 1$ gives

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { t \leq T } \Big \vert f _ { t } ( \beta ) - \Big [ 1 - \gamma \eta - \frac { \gamma ^ { 2 } \eta ^ { 2 } \beta } { 1 - \beta } \Big ] ^ { t } \Big \vert \leq \frac { T \gamma ^ { 3 } \eta ^ { 3 } e ^ { \gamma \tau } \beta ( 1 + \beta ) } { ( 1 - \beta ) ^ { 2 } } + \frac { \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } } { 1 - \beta } \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { T - 1 } ( t ^ { \prime } + 1 ) \beta ^ { t ^ { \prime } + 1 } } \\ & { \qquad \leq \frac { \gamma ^ { 2 } \eta ^ { 2 } \beta e ^ { \gamma \tau } [ 1 + \gamma \tau ( 1 - \beta ^ { 2 } ) ] } { ( 1 - \beta ) ^ { 3 } } = O ( \eta ^ { 2 } ) . } \end{array}
$$

For the first inequality, the coeficient $1 - \gamma \eta - \gamma ^ { 2 } \eta ^ { 2 } \beta / ( 1 - \beta )$ lies in (0, 1) when η is suficiently small, so the total error is bounded by the sum of the one-step error bounds. The second inequality uses $T \eta = \tau$ and $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { T - 1 } ( t ^ { \prime } + 1 ) \beta ^ { t ^ { \prime } + 1 } \leq \beta / ( 1 - \beta ) ^ { 2 } } \end{array}$ . This proves Lemma E.1. □

Proof of Lemma E.2. Using (C.4), we add and subtract the sum with coeficients $\Psi _ { 0 } ( T , t ) ^ { 2 } / \eta$ and covariance matrices evaluated at $\theta _ { t } ^ { \beta }$ . Decomposition gives

$$
\begin{array} { r l } & { \displaystyle \| \Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 } \| _ { \mathrm { o p } } \leq \frac { 1 } { \eta } \sum _ { t = 0 } ^ { T - 1 } \left| \Psi _ { \beta } ( T , t ) ^ { 2 } - \Psi _ { 0 } ( T , t ) ^ { 2 } \right| \cdot \left\| \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { 0 } } ( \cdot \ | \ x ) } \right) - \mathbf { r r } ^ { \top } \right\| _ { \mathrm { o p } } } \\ & { \qquad + \displaystyle \frac { 1 } { \eta } \sum _ { t = 0 } ^ { T - 1 } \Psi _ { 0 } ( T , t ) ^ { 2 } \left\| \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { 0 } } ^ { \beta } ( \cdot \ | \ x ) } - \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { 0 } } ( \cdot \ | \ x ) } \right) \right\| _ { \mathrm { o p } } } \\ & { \leq \displaystyle \frac { | \mathcal { V } | \left( e ^ { 2 M } + 1 \right) } { \eta } \operatorname* { m a x } _ { t \in T } \left( | \Psi _ { \beta } ( T , t ) | + | \Psi _ { 0 } ( T , t ) | \right) \sum _ { t = 0 } ^ { T - 1 } | \Psi _ { \beta } ( T , t ) - \Psi _ { 0 } ( T , t ) | } \\ & { \qquad + 2 \tau | \mathcal { V } | e ^ { 2 M } \operatorname* { m a x } _ { t \in \mathcal { T } } \| \theta _ { t } ^ { \beta } - \theta _ { t } ^ { 0 } \| _ { \infty } . } \end{array}\tag{E.3}
$$

For the second inequality, we bound the two matrix operator norms as in (C.12) by $| \mathcal { V } | ( e ^ { 2 M } + 1 )$ and $2 | \mathcal { V } | e ^ { 2 M } \| \theta _ { t } ^ { \beta } - \theta _ { t } ^ { 0 } \| _ { \infty } ,$ , respectively, with M defined in Appendix C. We also use $| \Psi _ { \beta } ( T , t ) ^ { 2 } - \Psi _ { 0 } ( T , t ) ^ { 2 } | =$ $| \Psi _ { \beta } ( T , t ) - \Psi _ { 0 } ( T , t ) | \cdot | \Psi _ { \beta } ( T , t ) + \Psi _ { 0 } ( T , t ) |$ to bound the first sum. For the remaining term, we use $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } \Psi _ { 0 } ( T , t ) ^ { 2 } / \eta \leq T \eta = \tau } \end{array}$ , since (C.3) gives $\Psi _ { 0 } ( T , t ) = \eta ( 1 - \gamma \eta ) ^ { T - t - 1 } \in [ 0 , \eta ] \mathrm { f o r } \gamma \eta \leq 1$ . We next bound the three quantities. First, applying (C.15) to both responses and taking the maximum over $t ^ { \prime } = 0 , \ldots , t - 1$ gives, for all $1 \leq t \leq T$

$$
\operatorname* { m a x } _ { t ^ { \prime } < t } \big ( | \Psi _ { \beta } ( t , t ^ { \prime } ) | + | \Psi _ { 0 } ( t , t ^ { \prime } ) | \big ) \leq e ^ { \gamma \tau } \eta \Big ( \operatorname* { m a x } _ { t ^ { \prime } < t } \sum _ { k = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( k , t ^ { \prime } ) + 1 \Big ) \leq e ^ { \gamma \tau } \eta \Big ( \frac { 1 } { 1 - \beta } + 1 \Big ) = O ( \eta ) .\tag{E.4}
$$

Here we use $\begin{array} { r } { \sum _ { k = t ^ { \prime } } ^ { t - 1 } \omega _ { 0 } ( k , t ^ { \prime } ) = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { k = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( k , t ^ { \prime } ) \leq \sum _ { k = t ^ { \prime } } ^ { t - 1 } \beta ^ { k - t ^ { \prime } } \leq 1 / ( 1 - \beta ) } \end{array}$ . For the sum of response diferences, we sum $\Psi _ { \beta } ( \ell + 1 , t ^ { \prime } ) - \Psi _ { \beta } ( \ell , t ^ { \prime } )$ over $\ell = t ^ { \prime } , \ldots , t - 1$ using (C.3) and interchange the sums to obtain

$$
\begin{array} { l } { { \Psi _ { \beta } ( t , t ^ { \prime } ) = \displaystyle \eta \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - \gamma \eta \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \sum _ { k = t ^ { \prime } } ^ { \ell } \omega _ { \beta } ( \ell , k ) \Psi _ { \beta } ( k , t ^ { \prime } ) } } \\ { { = \displaystyle \eta \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - \gamma \eta \sum _ { k = t ^ { \prime } + 1 } ^ { t - 1 } \left( \sum _ { \ell = k } ^ { t - 1 } \omega _ { \beta } ( \ell , k ) \right) \Psi _ { \beta } ( k , t ^ { \prime } ) . } } \end{array}
$$

The second equality omits the $k = t ^ { \prime }$ term since $\Psi _ { \beta } ( t ^ { \prime } , t ^ { \prime } ) = 0$ . We subtract the identity at $\beta = 0$ , where each cumulative weight equals one. Summing over $t ^ { \prime } = 0 , \ldots , t - 1$ , we have

$$
\begin{array} { r l } { \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } | \Psi _ { \beta } ( t , t ^ { \prime } ) - \Psi _ { 0 } ( t , t ^ { \prime } ) | \leq \eta \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \displaystyle \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - 1 \big | + \gamma \eta \displaystyle \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \displaystyle \sum _ { k = 0 } ^ { t ^ { \prime } - 1 } | \Psi _ { \beta } ( t ^ { \prime } , k ) - \Psi _ { 0 } ( t ^ { \prime } , k ) | } & \\ { + \gamma \eta \displaystyle \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \Big [ \displaystyle \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - 1 \Big ] \displaystyle \sum _ { k = 0 } ^ { t ^ { \prime } - 1 } | \Psi _ { \beta } ( t ^ { \prime } , k ) | \Big ] } & { } \\ { \leq 2 \eta \beta ( 1 + \gamma \tau e ^ { \gamma \tau } ) / ( 1 - \beta ) + \gamma \eta \displaystyle \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \sum _ { k = 0 } ^ { t ^ { \prime } - 1 } | \Psi _ { \beta } ( t ^ { \prime } , k ) - \Psi _ { 0 } ( t ^ { \prime } , k ) | } & { } \\ { \leq 2 \eta \beta ( 1 + \gamma \tau e ^ { \gamma \tau } ) e ^ { \gamma \tau } / ( 1 - \beta ) = O ( \eta ) . } & { } \end{array}\tag{E.5}
$$

For the second inequality, we have $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) = \sum _ { \ell = 0 } ^ { t - 1 } \sum _ { t ^ { \prime } = 0 } ^ { \ell } \omega _ { \beta } ( \ell , t ^ { \prime } ) = t , } \end{array}$ . Thus $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } | \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - } \end{array}$ $\begin{array} { r } { 1 | = 2 \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \operatorname* { m a x } \lbrace 1 - \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , \overline { { t ^ { \prime } } } ) , 0 \rbrace } \end{array}$ . Since $\omega _ { \beta } ( \ell , t ^ { \prime } ) \geq ( 1 - \beta ) \beta ^ { \ell - t ^ { \prime } }$ , we obtain $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } | \sum _ { \ell = t ^ { \prime } } ^ { t - 1 } \omega _ { \beta } ( \ell , t ^ { \prime } ) - 1 | \leq } \end{array}$ $2 \textstyle \sum _ { t ^ { \prime } = 0 } ^ { t - 1 } \beta ^ { t - t ^ { \prime } } \le 2 \beta / ( 1 - \beta )$ . We also use $\begin{array} { r } { \sum _ { k = 0 } ^ { t ^ { \prime } - 1 } | \Psi _ { \beta } ( t ^ { \prime } , k ) | \leq \tau e ^ { \gamma \tau } } \end{array}$ from the second inequality in (C.5). The last inequality follows by iterating the preceding bound and using $( 1 + \gamma \eta ) ^ { t - 1 } \leq e ^ { \gamma \eta ( t - 1 ) } \leq e ^ { \gamma \dot { \tau } }$ . For the parameter diferences, we substitute the representations from Lemma 4.1 for $\theta _ { t } ^ { \beta }$ and $\theta _ { t } ^ { 0 }$ to obtain

$$
\operatorname* { m a x } _ { t \leq T } \| \theta _ { t } ^ { \beta } - \theta _ { t } ^ { 0 } \| _ { \infty } = \operatorname* { m a x } _ { t \leq T } | f _ { t } ( \beta ) - f _ { t } ( 0 ) | \cdot \| \theta _ { 0 } - \theta ^ { \star } \| _ { \infty } = O ( \eta ) .\tag{E.6}
$$

The last bound uses Lemma E.1 and the mean value theorem, which give max $t \leq T \left| f _ { t } ( \beta ) - f _ { t } ( 0 ) \right| \leq T \gamma ^ { 2 } \eta ^ { 2 } \beta / ( 1 -$ $\beta ) + O ( \eta ^ { 2 } ) = O ( \eta )$ . Substituting (E.4), (E.5), and (E.6) into (E.3) yields $\| \Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 } \| _ { \mathrm { o p } } = O ( \eta )$ □

Proof of Lemma E.3. We first analyze the centered objective diference for the auxiliary processes and then transfer the normal limit to the sampled processes.

Expansion of the auxiliary objective. Taylor’s formula at $\theta _ { T } ^ { \beta }$ gives $J ( \widetilde { \theta } _ { T } ^ { \beta } ) = J ( \theta _ { T } ^ { \beta } ) + \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } ( \widetilde { \theta } _ { T } ^ { \beta } -$ $\theta _ { T } ^ { \beta } ) + R _ { T } ^ { \beta }$ , where $R _ { T } ^ { \beta }$ is the remainder after truncating this expansion at the linear term. Since $\mathbb { E } [ \widetilde { \theta } _ { T } ^ { \beta } ] = \theta _ { T } ^ { \beta }$ by Appendix C, the expected linear term is $\nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } ( \mathbb { E } [ \widetilde { \theta } _ { T } ^ { \beta } ] - \theta _ { T } ^ { \beta } ) = 0$ . Subtracting expectations in the Taylor expansion gives

$$
J ( \widetilde { \theta } _ { T } ^ { \beta } ) - \mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] = \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } ( \widetilde { \theta } _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } ) + R _ { T } ^ { \beta } - \mathbb { E } [ R _ { T } ^ { \beta } ] .\tag{E.7}
$$

According to (C.3) and (C.6), we write the linear term as $\begin{array} { r } { \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } ( \widetilde \theta _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } ) = \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } Y _ { t , b } ^ { \beta } } \end{array}$ , where $Y _ { t , b } ^ { \beta } = \Psi _ { \beta } ( T , t ) \mathbf { 1 } _ { \{ t < \zeta ^ { \beta } \} } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \varepsilon _ { t , b } ^ { \beta } / B .$

Normal limit for a single run. We next prove that $\textstyle \bigl ( \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } Y _ { t , b } ^ { \beta } \bigr ) / \operatorname { S t d } [ J ( \widehat \theta _ { T } ^ { \beta } ) - J ( \widehat \theta _ { T } ^ { 0 } ) ] \stackrel { \mathrm { d } } { \to } \mathcal { N } ( 0 , 1 / 2 )$ Let $\mathcal { H } _ { t , b - 1 } ^ { \beta }$ denote the history before response b at update t, and reveal the T B responses in increasing order of t and then b. Since $\mathbb { E } [ Y _ { t , b } ^ { \beta } \mid \mathcal { H } _ { t , b - 1 } ^ { \beta } ] = 0$ , these variables form a martingale diference sequence. We apply the martingale central limit theorem stated in Douc et al. (2018, Theorem E.4.1) to the partial sums of $Y _ { t , b } ^ { \beta } / \mathrm { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ]$ as $T \to \infty$ . Using (C.1) and (C.4), we write the covariance replacement as

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \mathbb { E } [ ( Y _ { t , b } ^ { \beta } ) ^ { 2 } \mid \mathcal { H } _ { t , b - 1 } ^ { \beta } ] - \frac { \eta } { B } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( \theta _ { T } ^ { \beta } ) } \\ & { \quad = \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \frac { \Psi _ { \beta } ( T , t ) ^ { 2 } \mathbf { 1 } _ { \left\{ t < \zeta ^ { \beta } \right\} } } { B ^ { 2 } } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \left[ \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \widetilde { \theta } _ { t } ^ { \beta } } ( \cdot \mid x ) } \right) - \mathbf { r r } ^ { \top } \right] \nabla J ( \theta _ { T } ^ { \beta } ) } \\ & { \quad \quad \quad - \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \frac { \Psi _ { \beta } ( T , t ) ^ { 2 } } { B ^ { 2 } } \nabla J ( \theta _ { T } ^ { \beta } ) ^ { \top } \left[ \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \theta _ { t } ^ { \beta } } ( \cdot \mid x ) } \right) - \mathbf { r r } ^ { \top } \right] \nabla J ( \theta _ { T } ^ { \beta } ) } \end{array}\tag{E.8}
$$

The third inequality in (C.5) gives $\textstyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \Psi _ { \beta } ( T , t ) ^ { 2 } / B ^ { 2 } \leq 5 \tau e ^ { 2 \gamma \tau } \eta / B$ . Equations (C.8) and (C.12) bound the expected operator norm of the diference between the auxiliary covariance and the exact-path

covariance by $O ( \sqrt { \eta / B } )$ . Together with the boundedness of $\nabla J ( \theta _ { T } ^ { \beta } )$ , these bounds give an expected absolute value of ${ \cal O } ( ( \eta / B ) ^ { 3 / 2 } )$ for the right-hand side. For the term $\nabla J ( \theta _ { T } ^ { \bar { \beta } } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( \theta _ { T } ^ { \beta } )$ , the diference between the EMA and no-EMA versions can be bounded by

$$
\begin{array} { r l } & { \big | \nabla J ( { \theta } _ { T } ^ { \beta } ) ^ { \top } \Sigma _ { T } ^ { \beta } \nabla J ( { \theta } _ { T } ^ { \beta } ) - \nabla J ( { \theta } _ { T } ^ { 0 } ) ^ { \top } \Sigma _ { T } ^ { 0 } \nabla J ( { \theta } _ { T } ^ { 0 } ) \big | } \\ & { \quad \leq \| \nabla J ( { \theta } _ { T } ^ { \beta } ) - \nabla J ( { \theta } _ { T } ^ { 0 } ) \| _ { 2 } \cdot \| \Sigma _ { T } ^ { \beta } \| _ { \mathrm { o p } } \big ( \| \nabla J ( { \theta } _ { T } ^ { \beta } ) \| _ { 2 } + \| \nabla J ( { \theta } _ { T } ^ { 0 } ) \| _ { 2 } \big ) } \\ & { \quad \quad + \| \nabla J ( { \theta } _ { T } ^ { 0 } ) \| _ { 2 } ^ { 2 } \cdot \| \Sigma _ { T } ^ { \beta } - \Sigma _ { T } ^ { 0 } \| _ { \mathrm { o p } } = O ( \eta ) . } \end{array}\tag{E.9}
$$

Lemma E.2, the parameter diference bound (E.6), the Lipschitz continuity of $\nabla J ( \theta )$ and the boundedness of $\nabla J ( \theta )$ and $\Sigma _ { T } ^ { \beta }$ give the last bound. According to Proposition 5.2, Var $[ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ]$ is $\Theta ( \eta / B )$ . Dividing (E.8) by this variance gives, as $\eta  0 ~ ( T  \infty )$

$$
\Big ( \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \mathbb { E } [ ( Y _ { t , b } ^ { \beta } ) ^ { 2 } \mid \mathcal { H } _ { t , b - 1 } ^ { \beta } ] \Big ) / \operatorname { V a r } [ J ( \widehat \theta _ { T } ^ { \beta } ) - J ( \widehat \theta _ { T } ^ { 0 } ) ] \stackrel { \mathbb { P } } {  } 1 / 2 .\tag{E.10}
$$

The expected absolute value of the right-hand side of (E.8) divided by this variance is $O ( \sqrt { \eta / B } )$ . Markov’s inequality therefore gives convergence in probability to zero. The diference (E.9) gives the limit $1 / 2$ To apply the martingale central limit theorem, we also need to bound the conditional second moments $\begin{array} { r l } { \sum _ { t = 0 } ^ { T - 1 } \overset {  } { \sum } _ { b = 1 } ^ { B } \mathbb { E } \big [ ( Y _ { t , b } ^ { \beta } ) ^ { 2 } \overset {  } { \mathbf { 1 } } _ { \{ | Y _ { t , b } ^ { \beta } | > \epsilon \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \} } \mid \mathcal { H } _ { t , b - 1 } ^ { \beta } \big ] } & { { } } \end{array}$ , where $\epsilon > 0$ is a fixed constant. The definition of $Y _ { t , b } ^ { \beta }$ gives

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { t < T , b \leq B } | Y _ { t , b } ^ { \beta } | \leq \frac { \| \nabla J ( \theta _ { T } ^ { \beta } ) \| _ { 2 } } { B } \displaystyle \operatorname* { m a x } _ { t < T , b \leq B } \big ( | \Psi _ { \beta } ( T , t ) | \cdot \| \mathbf { 1 } _ { \{ t < \zeta ^ { \beta } \} } \varepsilon _ { t , b } ^ { \beta } \| _ { 2 } \big ) } & { } \\ { \leq \frac { \sqrt { | \mathcal { V } | } L e ^ { \gamma \tau } \eta } { B } \| \nabla J ( \theta _ { T } ^ { \beta } ) \| _ { 2 } \displaystyle \operatorname* { m a x } _ { t < T } \sum _ { k = t } ^ { T - 1 } \omega _ { \beta } ( k , t ) } & { } \\ { \leq \frac { \sqrt { | \mathcal { V } | } L e ^ { \gamma \tau } \eta } { B ( 1 - \beta ) } \| \nabla J ( \theta _ { T } ^ { \beta } ) \| _ { 2 } = O ( \eta / B ) . } \end{array}
$$

The Cauchy–Schwarz inequality gives the first line. The second inequality uses the stopped noise bound $\| \mathbf { 1 } _ { \{ t < \zeta ^ { \beta } \} } \varepsilon _ { t , b } ^ { \beta } \| _ { \infty } \leq L$ in Appendix $\mathrm { C }$ and the first line of (C.15). For the last inequality, we use $\begin{array} { r } { \sum _ { k = t } ^ { T - 1 } \omega _ { \beta } ( k , t ) \leq } \end{array}$ ${ \textstyle \sum _ { k = t } ^ { T - 1 } \beta ^ { k - t } \leq 1 / ( 1 - \beta ) }$ . The boundedness of $\nabla J ( \theta _ { T } ^ { \beta } )$ in Appendix C gives the final bound. Since $\mathrm { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) -$ $J ( \widehat { \theta } _ { T } ^ { 0 } ) ] = \Theta ( \sqrt { \eta / B } )$ according to Proposition 5.2, the conditional second moments satisfy, as $\eta  0 ~ ( T  \infty )$

$$
\Big ( \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } \mathbb { E } \big [ ( Y _ { t , b } ^ { \beta } ) ^ { 2 } \mathbf { 1 } _ { \{ | Y _ { t , b } ^ { \beta } | > \epsilon \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \} } \mid \mathcal { H } _ { t , b - 1 } ^ { \beta } \big ] \Big ) / \operatorname { V a r } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \stackrel { \mathbb { P } } { \longrightarrow } 0 .\tag{E.11}
$$

Here we use max<sub>t<T, b≤B</sub> $| Y _ { t , b } ^ { \beta } | / \operatorname { S t d } [ J ( { \widehat { \theta } } _ { T } ^ { \beta } ) - J ( { \widehat { \theta } } _ { T } ^ { 0 } ) ] = O ( { \sqrt { \eta / B } } ) \to 0$ . Thus for each fixed $\epsilon > 0$ , every indicator is zero for all suficiently small η. Equations (E.10) and (E.11) verify the two conditions of the theorem with limiting variance $1 / 2$ , giving

$$
\Big ( \sum _ { t = 0 } ^ { T - 1 } \sum _ { b = 1 } ^ { B } Y _ { t , b } ^ { \beta } \Big ) / \operatorname { S t d } [ J ( \widehat \theta _ { T } ^ { \beta } ) - J ( \widehat \theta _ { T } ^ { 0 } ) ] \stackrel { \mathrm { d } } {  } \mathcal { N } ( 0 , 1 / 2 ) .\tag{E.12}
$$

Distribution of $J ( \widetilde { \theta } _ { T } ^ { \beta } ) \ : - \ : \mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ]$ . The remainder in the Taylor expansion (E.7) satisfies $| R _ { T } ^ { \beta } | \ \leq$ $\begin{array} { r } { \operatorname* { s u p } _ { \| \vartheta \| _ { \infty } \leq M + L \tau e ^ { \gamma \tau } } \| \nabla ^ { 2 } J ( \vartheta ) \| _ { \mathrm { o p } } \cdot \| \widetilde \theta _ { T } ^ { \beta } - \theta _ { T } ^ { \beta } \| _ { 2 } ^ { 2 } / 2 } \end{array}$ , where M and L are constants defined in Appendix C. Therefore, (C.9) gives $\mathbb { E } | R _ { T } ^ { \beta } - \mathbb { E } [ R _ { T } ^ { \beta } ] | \leq 2 \mathbb { E } | R _ { T } ^ { \beta } | = O ( \eta / B )$ . Dividing by Std[ $J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ]$ and applying Markov’s inequality gives $| R _ { T } ^ { \beta } - \mathbb { E } [ R _ { T } ^ { \beta } ] | / \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \ \xrightarrow { \mathbb { P } } \ 0$ . Applying Slutsky’s theorem (Kroese and Botev, 2023, Chapter 3, Exercise 31) to (E.7), using (E.12) and the remainder bound above, gives $\begin{array} { r } { ( J ( \widetilde { \theta } _ { T } ^ { \beta } ) - \mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] ) / \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \ \stackrel { \mathrm { d } } {  } \ \mathcal { N } ( 0 , 1 / 2 ) } \end{array}$ . The same argument holds for the no-EMA run. Combining the two independent limits gives

$$
\frac { J ( \widetilde { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { 0 } ) - \mathbb { E } [ J ( \widetilde { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { 0 } ) ] } { \mathrm { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] } \overset { \mathrm { d } } {  } \mathcal { N } ( 0 , 1 ) .
$$

Transfer to the sampled processes. As the auxiliary process and the sampled process difer only after the exit time, we have

$$
\mathbb { E } \vert J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { \beta } ) \vert \leq 2 \operatorname* { s u p } _ { \theta } \vert J ( \theta ) \vert \mathbb { P } ( \zeta ^ { \beta } \leq T ) = o ( \sqrt { \eta / B } ) .
$$

Here we use the exit probability bound in (C.8) and $- \gamma \log ( 1 / \operatorname* { m i n } _ { y } \pi _ { \mathrm { r e f } } ( y \mid x ) ) \le J ( \theta ) \le 1$ . The centered discrepancy satisfies $\mathbb { E } \lvert J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { \beta } ) - \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] \rvert \le 2 \mathbb { E } \lvert J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { \beta } ) \rvert = o ( \sqrt { \eta / B } )$ . Since the standard deviation is $\Theta ( \sqrt { \eta / B } )$ by Proposition 5.2, Markov’s inequality gives

$$
\frac { J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { \beta } ) - \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widetilde { \theta } _ { T } ^ { \beta } ) ] } { \operatorname { S t d } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] } \overset { \mathbb { P } } {  } 0 .
$$

The same estimate holds when $\beta = 0$ . Slutsky’s theorem therefore transfers the standard normal limit to the standardized diference of the sampled objectives. □

## F Vanishing expected gain as $\eta / ( 1 - \beta ) \to 0$

We formalize and prove the vanishing expected gain claim at the beginning of Section 5.2.

Lemma F.1 (Vanishing expected gain). Fix $B \geq 1$ and $\tau > 0$ , and consider the two runs in Section ${ 5 . }$ For any deterministic $\beta \in [ 0 , 1 )$ , possibly depending on η, i ${ } ^ { \mathrm { f } } \eta / ( 1 - \beta ) \to 0 \ a s \ \eta = \tau / T \to 0 \ \qquad $ , then the expected gain satisfies $\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - { J ( \widehat { \theta } _ { T } ^ { 0 } ) } ] \to 0 \mathrm { ~ } a s \eta \to 0$

Proof. We first bound the exact gain and then control the sampling correction.

Exact diference. Subtracting the definitions of $f _ { T } ( \beta )$ and $f _ { T } ( 0 )$ in Section $4 ~ \mathrm { g i }$ ves

$$
\begin{array} { r l } & { | f _ { T } ( \beta ) - f _ { T } ( 0 ) | = \displaystyle \big | ( 1 - \gamma \eta ) \big ( f _ { T - 1 } ( \beta ) - f _ { T - 1 } ( 0 ) \big ) - \gamma \eta \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { T - 1 } \omega _ { \beta } \big ( T - 1 , t ^ { \prime } \big ) \big ( f _ { t ^ { \prime } } ( \beta ) - f _ { T - 1 } ( \beta ) \big ) \big | } \\ & { \qquad = \gamma \eta \displaystyle \prod _ { t = 0 } ^ { T - 1 } ( 1 - \gamma \eta ) ^ { T - t - 1 } \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } \big ( t , t ^ { \prime } \big ) \big ( f _ { t ^ { \prime } } ( \beta ) - f _ { t } ( \beta ) \big ) \big | } \\ & { \qquad \le \gamma ^ { 2 } \eta ^ { 2 } e ^ { \gamma \tau } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \displaystyle \sum _ { t ^ { \prime } = 0 } ^ { t } ( t - t ^ { \prime } ) \omega _ { \beta } \big ( t , t ^ { \prime } \big ) \le \gamma ^ { 2 } \tau e ^ { \gamma \tau } \eta \beta / ( 1 - \beta ) . } \end{array}\tag{F.1}
$$

The first line uses $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { T - 1 } \omega _ { \beta } ( T - 1 , t ^ { \prime } ) = 1 } \end{array}$ to add $\begin{array} { r } { \gamma \eta \sum _ { t ^ { \prime } = 0 } ^ { T - 1 } \omega _ { \beta } ( T - 1 , t ^ { \prime } ) f _ { T - 1 } ( \beta ) } \end{array}$ and subtract $\gamma \eta f _ { T - 1 } ( \beta )$ Repeatedly substituting this relation and using $f _ { 0 } ( \beta ) = f _ { 0 } ( 0 ) \stackrel { \textstyle = 1 } { = } 1$ yields the second line. The definition of $f _ { t } ( \beta )$ and the bound $| f _ { t } ( \beta ) | \le e ^ { \gamma \tau }$ in Lemma C.1 give $| f _ { t ^ { \prime } } ( \beta ) - f _ { t } ( \beta ) | \leq \gamma \eta e ^ { \gamma \tau } ( t - t ^ { \prime } )$ . Applying $( 1 - \gamma \eta ) ^ { T - t - 1 } \leq 1$ $T \eta = \tau$ and $\begin{array} { r } { \sum _ { t ^ { \prime } = 0 } ^ { t } ( t - t ^ { \prime } ) \omega _ { \beta } ( t , t ^ { \prime } ) \leq \beta / ( 1 - \beta ) } \end{array}$ , which follows by summing the weights defined in Section $^ { 3 , }$ gives the last line. For the objective diference, we have

$$
| J ( \theta _ { T } ^ { \beta } ) - J ( \theta _ { T } ^ { 0 } ) | \leq \operatorname* { s u p } _ { \| \theta \| _ { \infty } \leq M } \| \nabla J ( \theta ) \| _ { 2 } \cdot \| \theta _ { T } ^ { \beta } - \theta _ { T } ^ { 0 } \| _ { 2 } \leq \frac { \gamma ^ { 2 } \tau e ^ { \gamma \tau } \eta \beta \| \theta _ { 0 } - \theta ^ { \star } \| _ { 2 } \operatorname* { s u p } _ { \| \theta \| _ { \infty } \leq M } \| \nabla J ( \theta ) \| _ { 2 } } { 1 - \beta } .
$$

The first inequality follows from the mean value theorem. The second inequality uses Lemma 4.1 to express $\lVert \theta _ { T } ^ { \beta } - \theta _ { T } ^ { 0 } \rVert _ { 2 }$ in terms of $| f _ { T } ( \beta ) - f _ { T } ( 0 ) |$ and applies (F.1). The supremum is finite and independent of $\eta$ and $\beta ,$ since $J ( \theta )$ is smooth and the infinity norms of $\theta _ { T } ^ { \beta }$ and $\theta _ { T } ^ { 0 }$ are bounded by $M ,$ which is defined in Appendix C. Expected gain. Lemma 5.1 and the uniform Hessian bound in Appendix C imply that the expected and exact gains difer by ${ \cal O } ( \eta / B )$ uniformly in $\beta$ . Hence, there exists a constant C independent of $\eta$ and $\beta$ such that

$$
\lvert \mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \rvert \leq C \Big ( \frac { \eta \beta } { 1 - \beta } + \frac { \eta } { B } \Big ) .
$$

The upper bound goes to zero when $\eta / ( 1 - \beta )$ and $\eta$ both $_ \mathrm { g o }$ to zero.

## G Proof of Theorem 5.4

Proof. We identify the limiting exact gain at fixed $\rho ,$ then use Lemma 5.1 and Proposition 5.2 to control the sampled comparison.

Continuous residual version. Here $\beta = 1 - \eta / \rho$ varies with $\eta ,$ so the fixed-β analysis in Section 5.1 does not apply directly. For fixed $\rho > 0$ , we introduce a continuous residual $f ( s ; \rho )$ at optimization time $s \geq 0$ , corresponding to $f _ { t } ( \beta )$ at $s = t \eta$ . It describes the remaining displacement from $\theta ^ { \star }$ through $\theta ( s ; \rho ) = \theta ^ { \star } + f ( s ; \rho ) ( \theta _ { 0 } - \theta ^ { \star } )$ and is defined by

$$
\partial _ { s } f ( s ; \rho ) = - \gamma \frac { \int _ { 0 } ^ { s } e ^ { - ( s - u ) / \rho } f ( u ; \rho ) \mathrm { d } u } { \rho ( 1 - e ^ { - s / \rho } ) } , \qquad f ( 0 ; \rho ) = 1 .\tag{G.1}
$$

The normalized average at $s = 0$ is interpreted by continuity. The following lemma bounds the error between $f _ { t } ( \beta )$ and its continuous approximation $f ( t \eta ; \rho )$ . Its proof is provided in Appendix G.1.

Lemma G.1 (Continuous approximation of the exact path). Let $f _ { t }$ be defined in Section $\it 4$ and $f ( s ; \rho )$ by (G.1). For $0 < \eta < \rho , \beta = 1 - \eta / \rho $ , and $T \eta \leq \tau$

$$
\operatorname* { m a x } _ { t \leq T } | f _ { t } ( \beta ) - f ( t \eta ; \rho ) | \leq \frac 7 2 \gamma e ^ { 2 \gamma \tau } \eta .\tag{G.2}
$$

At $T \eta = \tau ,$ , Lemma G.1 and Lemma 4.1 give $\lVert \theta _ { T } ^ { \beta } - \theta ( \tau ; \rho ) \rVert _ { 2 } = O ( \eta )$ . Since $f _ { T } ( 0 ) = ( 1 - \gamma \eta ) ^ { T } = e ^ { - \gamma \tau } + O ( \eta )$ we have $\lVert { \boldsymbol { \theta } } _ { T } ^ { 0 } - [ { \boldsymbol { \theta } } ^ { \star } + e ^ { - \gamma \tau } ( { \boldsymbol { \theta } } _ { 0 } - { \boldsymbol { \theta } } ^ { \star } ) ] \rVert _ { 2 } = O ( \eta )$ . Applying the mean value theorem to $J ( \theta )$ for each pair of parameter vectors yields

$$
J ( \theta _ { T } ^ { \beta } ) - J ( \theta _ { T } ^ { 0 } ) = J ( \theta ( \tau ; \rho ) ) - J ( \theta ^ { \star } + e ^ { - \gamma \tau } ( \theta _ { 0 } - \theta ^ { \star } ) ) + O ( \eta ) .\tag{G.3}
$$

The discrete parameter vectors have infinity norm at most M as defined in Appendix $\mathrm { C } ,$ and their continuous limits satisfy the same bound. Since $J ( \theta )$ is smooth, $\mathrm { s u p } _ { \| \vartheta \| _ { \infty } \leq M } \| \nabla J ( \vartheta ) \| _ { 2 }$ is finite and independent of $\eta .$ Thus each ${ \cal { O } } ( \eta )$ parameter error produces an ${ \cal { O } } ( \eta )$ objective error.

Positive gain in the safe range. For $0 ~ < ~ \rho ~ \leq ~ 1 / ( 4 \gamma )$ and $\eta ~ < ~ \rho ,$ we have $0 ~ < ~ \beta ~ = ~ 1 - \eta / \rho ~ \leq$ $1 - 4 \gamma \eta \leq ( ( 1 - \gamma \eta ) / ( 1 + \gamma \eta ) ) ^ { 2 }$ , which lies in the safe range in Theorem 4.2. Multiplying the ratio bounds $f _ { t ^ { \prime } } ( \beta ) / f _ { t ^ { \prime } - 1 } ( \beta ) \geq \sqrt { \beta }$ from the proof of Theorem 4.2 gives

$$
f _ { t } ( \beta ) = \prod _ { t ^ { \prime } = 1 } ^ { t } \left( f _ { t ^ { \prime } } ( \beta ) / f _ { t ^ { \prime } - 1 } ( \beta ) \right) \geq \beta ^ { t / 2 } = ( 1 - \eta / \rho ) ^ { t / 2 } .
$$

By (G.2), letting $\eta  0$ with tη $ s$ yields $f ( s ; \rho ) \geq e ^ { - s / ( 2 \rho ) } > 0$ . The integral in (G.1) is positive, so $f ( s ; \rho )$ is strictly decreasing. Its average over preceding times is therefore strictly larger than its current value, implying $\partial _ { s } f ( s ; \rho ) < - \gamma f ( s ; \rho )$ for $s > 0$ . Hence we have $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \big [ e ^ { \gamma s } f ( s ; \rho ) \big ] = e ^ { \gamma s } \big [ \partial _ { s } f \big ( \boldsymbol { s } ; \rho ) + \gamma f ( \boldsymbol { s } ; \rho ) \big ] < 0 . } \end{array}$ . Since $f ( 0 ; \rho ) = 1$ , this gives, for all $s > 0$

$$
0 < f ( s ; \rho ) < e ^ { - \gamma s } .\tag{G.4}
$$

By (B.3), $J ( \theta ^ { \star } + a ( \theta _ { 0 } - \theta ^ { \star } ) )$ ) is strictly decreasing in $a > 0$ . Hence (G.4) gives $J ( \theta ( \tau ; \rho ) ) > J ( \theta ^ { \star } { + } e ^ { - \gamma \tau } ( \theta _ { 0 } { - } \theta ^ { \star } ) )$ The coeficient in the main theorem is therefore

$$
\begin{array} { r l } & { C ( \tau , \gamma , \rho ) : = \frac { J ( \theta ( \tau ; \rho ) ) - J ( \theta ^ { \star } + e ^ { - \gamma \tau } ( \theta _ { 0 } - \theta ^ { \star } ) ) } { \rho } } \\ & { \qquad = \displaystyle \frac { \gamma } { \rho } \int _ { f ( \tau ; \rho ) } ^ { e ^ { - \gamma \tau } } a \operatorname { V a r } _ { y \sim \pi _ { \theta ^ { \star } + a ( \theta _ { 0 } - \theta ^ { \star } ) } ( \cdot \vert x ) } [ \theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y ) ] \mathrm { d } a > 0 . } \end{array}
$$

This specifies its dependence on $\rho , \tau , \gamma ,$ and the fixed problem. It does not depend on $\eta , \beta ,$ or $B .$ . Using $\rho = \beta \rho + \eta$ in (G.3) gives $J ( \theta _ { T } ^ { \beta } ) \stackrel { . . } { - } J ( \theta _ { T } ^ { 0 } ) = C ( \tau , \gamma , \rho ) \beta \rho + O ( \eta )$

Expected gain and winning probability. Lemma 5.1, together with the uniform Hessian bound in Appendix C, gives

$$
\mathbb { E } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] = J ( \theta _ { T } ^ { \beta } ) - J ( \theta _ { T } ^ { 0 } ) + O ( \eta / B ) = C ( \tau , \gamma , \rho ) \beta \rho + O ( \eta ) + O ( \eta / B ) .
$$

The ${ \cal { O } } ( \eta )$ error is the discrete-path approximation, and the ${ \cal O } ( \eta / B )$ term is the sampling correction. $\mathrm { A t }$ each fixed positive $\rho ,$ the expected gain tends to $C ( \tau , \gamma , \rho ) \rho > 0$ . Thus it is at least half this limit for suficiently small $\eta .$ . Proposition 5.2 gives $\mathrm { V a r } [ J ( \widehat { \theta } _ { T } ^ { \beta } ) - J ( \widehat { \theta } _ { T } ^ { 0 } ) ] \leq C \eta / B$ for suficiently small $\eta ,$ where $C$ is a constant independent of $\eta$ and $\beta .$ . Chebyshev’s inequality then gives

$$
\mathbb { P } \big ( J \big ( \widehat { \theta } _ { T } ^ { \beta } \big ) \leq J ( \widehat { \theta } _ { T } ^ { 0 } ) \big ) \leq \frac { \mathrm { V a r } \big [ J \big ( \widehat { \theta } _ { T } ^ { \beta } \big ) - J \big ( \widehat { \theta } _ { T } ^ { 0 } \big ) \big ] } { \big ( \mathbb { E } \big [ J \big ( \widehat { \theta } _ { T } ^ { \beta } \big ) - J \big ( \widehat { \theta } _ { T } ^ { 0 } \big ) \big ] \big ) ^ { 2 } } \leq \frac { 4 C } { C ( \tau , \gamma , \rho ) ^ { 2 } \rho ^ { 2 } } \frac { \eta } { B } = O ( \eta / B ) .
$$

## G.1 Proof of Lemma G.1

Proof. We construct the continuous solution and estimate the error of the discrete approximation.

Construction of the continuous solution. We first establish existence and uniqueness for (G.1) by successive substitution in its integral form. According to the definition of the continuous version in (G.1), we have $\begin{array} { r } { f ( v ; \rho ) = f ( 0 ; \rho ) - \gamma \int _ { 0 } ^ { v } ( ( \bar { ( } \int _ { 0 } ^ { s } e ^ { - ( s - u ) / \rho } f ( u ; \rho ) \bar { \mathrm { d } } u ) / \rho ( 1 - e ^ { - s / \rho } ) ) \mathrm { d } s ) } \end{array}$ . Define the iterates $f ^ { [ n ] } ( v ; { \dot { \rho } } )$ for $n \geq 0$ by

$$
f ^ { [ n + 1 ] } ( v ; \rho ) = 1 - \gamma \int _ { 0 } ^ { v } \frac { \int _ { 0 } ^ { s } e ^ { - ( s - u ) / \rho } f ^ { [ n ] } ( u ; \rho ) \mathrm { d } u } { \rho ( 1 - e ^ { - s / \rho } ) } \mathrm { d } s ,
$$

where we start from $f ^ { [ 0 ] } ( v ; \rho ) = 1$ and $f ( 0 ; \rho ) = 1$ . Since the term $\begin{array} { r l } { \int _ { 0 } ^ { s } e ^ { - ( s - u ) / \rho } f ^ { [ n ] } ( u ; \rho ) \mathrm { d } u / ( \rho ( 1 - e ^ { - s / \rho } ) ) } \end{array}$ can be interpreted as a weighted average of $f ^ { [ n ] } ( u ; \rho )$ over [0, s], it is bounded by the maximum absolute value on $[ 0 , s ]$ . Subtracting two successive iterates therefore gives, for $n \geq 1 , | f ^ { [ n + 1 ] } ( v ; \rho ) - f ^ { [ n ] } ( v ; \rho ) | \leq$ $\begin{array} { r } { \gamma \int _ { 0 } ^ { v } \operatorname* { m a x } _ { u \leq s } \vert \dot { f } ^ { [ n ] } ( u ; \rho ) - \dot { f } ^ { [ n - 1 ] } ( u ; \rho ) } \end{array}$ | ds. Since $f ^ { [ 1 ] } ( v ; \rho ) - f ^ { [ 0 ] } ( \bar { v } ; \rho ) = - \gamma v$ , induction yields

$$
\operatorname* { m a x } _ { s \leq v } | f ^ { [ n + 1 ] } ( s ; \rho ) - f ^ { [ n ] } ( s ; \rho ) | \leq \frac { ( \gamma v ) ^ { n + 1 } } { ( n + 1 ) ! } .\tag{G.5}
$$

The corrections therefore have a summable uniform bound on $v \in [ 0 , \tau ]$ . Thus the iterates converge uniformly to a solution $f .$ For uniqueness, suppose $f$ and $g$ are two solutions. Applying the same integral estimate to their diference gives $\begin{array} { r } { | f ( v ; \rho ) - g ( v ; \rho ) | \leq \gamma \int _ { 0 } ^ { v } } \end{array}$ ma $\mathrm { x } _ { u \le s } | f ( u ; \rho ) - g ( u ; \rho ) |$ | ds. Repeatedly applying this estimate gives $\begin{array} { r } { \operatorname* { m a x } _ { s \leq \tau } | f ( s ; \rho ) - g ( s ; \rho ) | \leq ( ( \gamma \tau ) ^ { n } / n ! ) } \end{array}$ max $_ { s \leq \tau } \left. f ( s ; \rho ) - g ( s ; \rho ) \right.$ . Letting $n \to \infty$ proves uniqueness.

Continuous change over one time step. Summing the bounds on successive diferences in (G.5) and using $f ^ { [ 0 ] } ( v ; \rho ) = 1$ , we have $\begin{array} { r } { | f ( s ; \rho ) | \leq 1 + \bar { \sum _ { n = 0 } ^ { \infty } } ( \gamma s ) ^ { \bar { n + 1 } } / ( n + 1 ) ! = e ^ { \gamma s } } \end{array}$ for $s \in [ 0 , \tau ]$ . Since the derivative $\partial _ { s } f ( s ; \rho )$ is a weighted average of $f ( u ; \rho )$ , we have $\begin{array} { r } { | \partial _ { s } f ( s ; \rho ) | \le \gamma \operatorname* { m a x } _ { u \le s } | f ( u ; \rho ) | \le \gamma e ^ { \gamma \tau } } \end{array}$ on $[ 0 , \tau ]$ . The mean value theorem then gives for $s , s ^ { \prime } \in [ 0 , \tau ]$ ],

$$
| f ( s ; \rho ) - f ( s ^ { \prime } ; \rho ) | \leq \gamma e ^ { \gamma \tau } | s - s ^ { \prime } | .\tag{G.6}
$$

To control the second derivative, we diferentiate (G.1) and use (G.6) to obtain, for $0 < s \le \tau$

$$
| \partial _ { s } ^ { 2 } f ( s ; \rho ) | \leq \frac { \gamma \int _ { 0 } ^ { s } e ^ { - ( s - u ) / \rho } | f ( u ; \rho ) - f ( s ; \rho ) | \mathrm { d } u } { \rho ^ { 2 } ( 1 - e ^ { - s / \rho } ) ^ { 2 } } \leq \frac { \gamma ^ { 2 } e ^ { \gamma \tau } ( 1 - ( 1 + s / \rho ) e ^ { - s / \rho } ) } { ( 1 - e ^ { - s / \rho } ) ^ { 2 } } \leq \gamma ^ { 2 } e ^ { \gamma \tau } .
$$

The last inequality uses $1 - s / \rho \le e ^ { - s / \rho }$ . Therefore for $t < T$ , the continuous change has main term $\eta \partial _ { s } f ( t \eta ; \rho )$ and error bounded by

$$
\begin{array} { r l } { | f ( ( t + 1 ) \eta ; \rho ) - f ( t \eta ; \rho ) - \eta \partial _ { s } f ( t \eta ; \rho ) | = \Big | \displaystyle \int _ { t \eta } ^ { ( t + 1 ) \eta } [ \partial _ { s } f ( s ; \rho ) - \partial _ { s } f ( t \eta ; \rho ) ] \mathrm { d } s \Big | } & { { } } \\ { \le \gamma ^ { 2 } e ^ { \gamma \tau } \displaystyle \int _ { t \eta } ^ { ( t + 1 ) \eta } ( s - t \eta ) \mathrm { d } s = \frac { 1 } { 2 } \gamma ^ { 2 } e ^ { \gamma \tau } \eta ^ { 2 } . } & { { } } \end{array}\tag{G.7}
$$

The same estimate holds for $t = 0$ by continuity of $\partial _ { s } f ( s ; \rho )$ at $s = 0$

Comparison of the discrete and continuous averages. We next control $| ( f _ { t + 1 } ( \beta ) - f _ { t } ( \beta ) ) / \eta - \partial _ { s } f ( t \eta ; \rho ) |$ For $1 \leq t < T$ , set $r = - \eta / \log \beta$ , so $e ^ { - \eta / r } = \beta$ and $\rho - \eta \leq r \leq \rho$ . We can write the discrete average as

$$
\sum _ { t ^ { \prime } = 0 } ^ { t } \omega _ { \beta } ( t , t ^ { \prime } ) f _ { t ^ { \prime } } ( \beta ) = \sum _ { k = 0 } ^ { t } \int _ { k \eta } ^ { ( k + 1 ) \eta } \frac { e ^ { - a / r } f _ { t - k } ( \beta ) } { r ( 1 - e ^ { - ( t + 1 ) \eta / r } ) } \mathrm { d } a .
$$

Using the definition of $f _ { t } ( \beta )$ in Section 4 and (G.1), we can decompose the diference between the discrete and continuous changes as

$$
\begin{array} { r l } & { ( \ f _ { t + 1 } ( \beta ) - f _ { t } ( \beta ) ) / \eta - \partial _ { s } f ( t \eta ; \rho ) | } \\ & { \quad \le \gamma \displaystyle \operatorname* { m a x } _ { t ^ { \prime } \le t } | f _ { t ^ { \prime } } ( \beta ) - f ( t ^ { \prime } \eta ; \rho ) | } \\ & { \qquad + \gamma \displaystyle \left\lfloor \displaystyle \sum _ { k = 0 } ^ { t } \int _ { k \eta } ^ { ( k + 1 ) \eta } \frac { e ^ { - a / r } f ( ( t - k ) \eta ; \rho ) } { r \left( 1 - e ^ { - ( t + 1 ) \eta / r } \right) } \mathrm { d } a - \int _ { 0 } ^ { t \eta } \frac { e ^ { - a / r } f ( t \eta - a ; \rho ) } { r \left( 1 - e ^ { - t \eta / r } \right) } \mathrm { d } a \right\rfloor } \\ & { \qquad + \gamma \displaystyle \left\lfloor \displaystyle \int _ { 0 } ^ { t \eta } f ( t \eta - a ; \rho ) \displaystyle \left[ \frac { e ^ { - a / r } } { r \left( 1 - e ^ { - t \eta / r } \right) } - \frac { e ^ { - a / \rho } } { \rho \left( 1 - e ^ { - t \eta / \rho } \right) } \right] \mathrm { d } a \right\rfloor . } \end{array}\tag{G.8}
$$

The first term bounds the error caused by replacing $f _ { t ^ { \prime } } ( \beta )$ with $f ( t ^ { \prime } \eta ; \rho )$ . Term (I) compares the grid values of $f ( \cdot ; \rho )$ with its values inside each interval and accounts for the diferent normalizations on $[ 0 , t \eta ]$ and $[ 0 , ( t + 1 ) \eta ]$ . Term (II) accounts for the change of weight parameter from r to $\rho$ while keeping $f ( \cdot ; \rho )$ fixed. Here we set $a = t \eta - u$ to match the continuous expression in (G.1) with the discrete average. Splitting of the last interval in (I) gives

$$
\begin{array} { r l } & { \displaystyle ( \mathrm { I } ) \le \sum _ { k = 0 } ^ { t - 1 } \int _ { k \eta } ^ { ( k + 1 ) \eta } \left| f ( ( t - k ) \eta ; \rho ) - f ( t \eta - a ; \rho ) \right| \frac { e ^ { - a / r } } { r \left( 1 - e ^ { - ( t + 1 ) \eta / r } \right) } \mathrm { d } a } \\ & { \quad \quad \quad + \omega _ { \beta } ( t , 0 ) \int _ { 0 } ^ { t \eta } \left| f ( 0 ; \rho ) - f ( t \eta - a ; \rho ) \right| \frac { e ^ { - a / r } } { r \left( 1 - e ^ { - t \eta / r } \right) } \mathrm { d } a } \\ & { \quad \quad \le \gamma e ^ { \gamma \tau } \eta \frac { 1 - e ^ { - t \eta / r } } { 1 - e ^ { - ( t + 1 ) \eta / r } } + \omega _ { \beta } ( t , 0 ) \gamma e ^ { \gamma \tau } t \eta \le 2 \gamma e ^ { \gamma \tau } \eta . } \end{array}
$$

The first inequality uses $( 1 - e ^ { - t \eta / r } ) / ( 1 - e ^ { - ( t + 1 ) \eta / r } ) = 1 - \omega _ { \beta } ( t , 0 )$ to account for the diferent normalizations. The second applies $\left( \mathrm { G . 6 } \right)$ with input diferences at most η and $t \eta .$ , respectively. The last inequality uses $\textstyle \omega _ { \beta } ( t , 0 ) = \beta ^ { t } / \sum _ { k = 0 } ^ { t } \beta ^ { k } \leq 1 / ( t + 1 )$ . To bound (II), diferentiate with respect to the weight parameter while keeping $f ( \cdot ; \rho )$ fixed. For $x > 0$ , we define the weighted mean of a as m $\begin{array} { r } { \imath ( x ) : = \int _ { 0 } ^ { t \eta } ( a e ^ { - a / x } ) / \big ( x ( 1 - e ^ { - t \eta / x } ) \big ) } \end{array}$ da = $x - t \eta / ( e ^ { t \eta / x } - 1 ) \in [ 0 , t \eta ]$ . So the derivative satisfies

$$
\begin{array} { r l } & { \left| \displaystyle \frac { \partial } { \partial x } \int _ { 0 } ^ { t \eta } f ( t \eta - a ; \rho ) \frac { e ^ { - a / x } } { x ( 1 - e ^ { - t \eta / x } ) } \mathrm { d } a \right| } \\ & { \quad = \left| \displaystyle \frac { 1 } { x ^ { 2 } } \int _ { 0 } ^ { t \eta } f ( t \eta - a ; \rho ) ( a - m ( x ) ) \frac { e ^ { - a / x } } { x ( 1 - e ^ { - t \eta / x } ) } \mathrm { d } a \right| } \\ & { \quad = \left| \displaystyle \frac { 1 } { x ^ { 2 } } \int _ { 0 } ^ { t \eta } \left[ f ( t \eta - a ; \rho ) - f ( t \eta - m ( x ) ; \rho ) \right] ( a - m ( x ) ) \frac { e ^ { - a / x } } { x ( 1 - e ^ { - t \eta / x } ) } \mathrm { d } a \right| } \\ & { \quad \le \displaystyle \frac { \gamma e ^ { \gamma \tau } } { x ^ { 2 } } \int _ { 0 } ^ { t \eta } ( a - m ( x ) ) ^ { 2 } \frac { e ^ { - a / x } } { x ( 1 - e ^ { - t \eta / x } ) } \mathrm { d } a = \gamma e ^ { \gamma \tau } \Big [ 1 - \frac { ( t \eta / x ) ^ { 2 } e ^ { t \eta / x } } { ( e ^ { t \eta / x } - 1 ) ^ { 2 } } \Big ] \le \gamma e ^ { \gamma \tau } . } \end{array}
$$

The second equality uses the zero weighted mean of $a - m ( x )$ , and the first inequality applies (G.6). Integrating this derivative bound from r to $\rho$ gives $\begin{array} { r } { \left( \mathrm { I I } \right) \le \gamma e ^ { \gamma \tau } ( \rho - r ) \le \gamma e ^ { \gamma \tau } \eta } \end{array}$ . Substituting the bounds for (I) and (II) into (G.8) gives

$$
\left| \left( f _ { t + 1 } ( \beta ) - f _ { t } ( \beta ) \right) / \eta - \partial _ { s } f ( t \eta ; \rho ) \right| \leq \gamma \operatorname* { m a x } _ { t ^ { \prime } \leq t } | f _ { t ^ { \prime } } ( \beta ) - f ( t ^ { \prime } \eta ; \rho ) | + 3 \gamma ^ { 2 } e ^ { \gamma \tau } \eta .\tag{G.9}
$$

$\mathrm { A t                         } t = 0 \mathrm { . }$ , the left-hand side equals zero, so this bound also holds there. The triangle inequality together with (G.7) and (G.9) gives, for $t < T$ ，

$$
\begin{array} { r l } & { | f _ { t + 1 } ( \beta ) - f ( ( t + 1 ) \eta ; \rho ) | \le | f _ { t } ( \beta ) - f ( t \eta ; \rho ) | + \eta \Big | \frac { f _ { t + 1 } ( \beta ) - f _ { t } ( \beta ) } { \eta } - \partial _ { s } f ( t \eta ; \rho ) \Big | } \\ & { \qquad + | f ( ( t + 1 ) \eta ; \rho ) - f ( t \eta ; \rho ) - \eta \partial _ { s } f ( t \eta ; \rho ) | } \\ & { \qquad \le ( 1 + \gamma \eta ) \underset { t ^ { \prime } < t } { \operatorname* { m a x } } | f _ { t ^ { \prime } } ( \beta ) - f ( t ^ { \prime } \eta ; \rho ) | + 7 \gamma ^ { 2 } e ^ { \gamma \tau } \eta ^ { 2 } / 2 . } \end{array}\tag{G.10}
$$

Since $f _ { 0 } ( \beta ) = f ( 0 ; \rho ) = 1$ , repeatedly applying (G.10) gives

$$
\operatorname* { m a x } _ { t \leq T } | f _ { t } ( \beta ) - f ( t \eta ; \rho ) | \leq \frac { 7 } { 2 } \gamma ^ { 2 } e ^ { \gamma \tau } \eta ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } ( 1 + \gamma \eta ) ^ { t } = \frac { 7 } { 2 } \gamma e ^ { \gamma \tau } \eta \big [ ( 1 + \gamma \eta ) ^ { T } - 1 \big ] \leq \frac { 7 } { 2 } \gamma e ^ { 2 \gamma \tau } \eta .
$$

The last inequality uses $( 1 + \gamma \eta ) ^ { T } - 1 \leq e ^ { \gamma T \eta } \leq e ^ { \gamma \tau }$

## H Proof of Theorem 5.5

Proof. We first match the exact endpoints at finite step size. The mean comparison then follows from the sampling expansion; the variance comparison requires the continuous response and its limit at long optimization times. In this section, $f _ { T } ( \eta , \beta )$ displays the learning-rate argument because the two runs use diferent rates.

Finite-step matching and expected objectives. The assumptions give $0 < \eta _ { 1 } \leq \rho _ { 1 } < \rho _ { 2 } \leq 1 / ( 4 \gamma )$ and $0 \leq \beta _ { 1 } < 1 - \eta _ { 1 } / \rho _ { 2 } \leq 1 - 4 \gamma \eta _ { 1 } \leq ( ( 1 - \gamma \eta _ { 1 } ) / ( 1 + \gamma \eta _ { 1 } ) ) ^ { 2 }$ . By the residual comparison in the proof of Theorem $\mathrm { ~ \xi ~ . 2 , ~ } f _ { T } ( \eta _ { 1 } , 1 - \eta _ { 1 } / \rho _ { 2 } ) < f _ { T } ( \eta _ { 1 } , \beta _ { 1 } ) < 1$ . For fixed T, the definition of $f _ { t }$ in Section 4 gives $f _ { T } ( \eta , 1 - \eta / \rho _ { 2 } ) \to 1$ as $\eta  0 .$ . Since $f _ { T } ( \eta , 1 - \eta / \rho _ { 2 } )$ is continuous for $\eta \in ( 0 , \eta _ { 1 } ]$ , there exists a deterministic $\eta _ { 2 } \in ( 0 , \eta _ { 1 } )$ such that

$$
f _ { T } ( \eta _ { 2 } , 1 - \eta _ { 2 } / \rho _ { 2 } ) = f _ { T } ( \eta _ { 1 } , \beta _ { 1 } ) .
$$

Set $\beta _ { 2 } = 1 - \eta _ { 2 } / \rho _ { 2 }$ . Then $\beta _ { 2 } > \beta _ { 1 }$ , and Lemma 4.1 gives $\theta _ { T } ^ { ( 2 ) } = \theta _ { T } ^ { ( 1 ) }$ . Applying the individual expansion (C.14) to both runs and using $\theta _ { T } ^ { ( 2 ) } = \theta _ { T } ^ { ( 1 ) }$ proves the expected objective comparison.

Limiting optimization times. For the variance comparison, we use the continuous approximation from Lemma G.1. We first identify the limiting optimization time of the matched run. Fix $\tau , \rho _ { 1 } , \rho _ { 2 }$ . By $\left( \mathrm { G . 4 } \right)$ $f ( \cdot ; \rho _ { 2 } )$ is continuous, strictly decreasing from one, and tends to zero. Hence there is a unique $\tau _ { 2 } > 0$ such that

$$
f ( \tau _ { 2 } ; \rho _ { 2 } ) = f ( \tau ; \rho _ { 1 } ) , \qquad \theta ( \tau _ { 2 } ; \rho _ { 2 } ) = \theta ( \tau ; \rho _ { 1 } ) .\tag{H.1}
$$

Combining Lemma G.1 with the exact endpoint matching gives $f ( T \eta _ { 2 } ; \rho _ { 2 } ) = f ( \tau ; \rho _ { 1 } ) + O ( \eta _ { 1 } ) = f ( \tau _ { 2 } ; \rho _ { 2 } ) +$ ${ \cal O } ( \eta _ { 1 } )$ . Since $f ( \cdot ; \rho _ { 2 } )$ is continuous and strictly decreasing, $f ( T \eta _ { 2 } ; \rho _ { 2 } ) \to f ( \tau _ { 2 } ; \rho _ { 2 } )$ implies

$$
T \eta _ { 2 }  \tau _ { 2 } , \qquad \frac { \eta _ { 2 } } { \eta _ { 1 } } = \frac { T \eta _ { 2 } } { \tau }  \frac { \tau _ { 2 } } { \tau } \quad \mathrm { a s } \ \eta _ { 1 }  0 .
$$

To compare the variance coeficients, we introduce the continuous counterpart of the noise response in (C.3). Here $s = t \eta$ is optimization time. Fix $\rho$ and a noise input time $s _ { 1 } > 0 .$ . For a unit noise input in one coordinate at update $t ^ { \prime } ,$ , where $t ^ { \prime } \eta  s _ { 1 }$ as $\eta  0$ , let $\Delta \theta _ { t } ^ { \beta }$ and $\Delta m _ { t } ^ { \beta }$ denote the discrete diferences in that parameter coordinate and its EMA momentum before bias correction from the dynamics without this input. With $\beta = 1 - \eta / \rho$ , define the continuous responses for $s _ { 2 } > s _ { 1 }$ by

$$
\Delta \theta ( s _ { 2 } ; \rho ) = \operatorname* { l i m } _ { \eta \to 0 } \Delta \theta _ { t } ^ { \beta } / \eta , \qquad \Delta m ( s _ { 2 } ; \rho ) = \operatorname* { l i m } _ { \eta \to 0 } \Delta m _ { t } ^ { \beta } / \eta ,
$$

where $t \eta  s _ { 2 }$ as $\eta  0$ . The updates in (3.5) and the diference dynamics in (C.2) give

$$
\partial _ { s _ { 2 } } \Delta \theta ( s _ { 2 } ; \rho ) = \Delta m ( s _ { 2 } ; \rho ) / ( 1 - e ^ { - s _ { 2 } / \rho } ) , \quad \partial _ { s _ { 2 } } \Delta m ( s _ { 2 } ; \rho ) = - ( \gamma \Delta \theta ( s _ { 2 } ; \rho ) + \Delta m ( s _ { 2 } ; \rho ) ) / \rho .
$$

Let $\Phi _ { \rho } ( s _ { 2 } , s _ { 1 } )$ be the transition matrix for $( \Delta \theta ( s _ { 2 } ; \rho ) , \Delta m ( s _ { 2 } ; \rho ) ) ^ { \top }$ . It is defined by

$$
\partial _ { s _ { 2 } } \Phi _ { \rho } ( s _ { 2 } , s _ { 1 } ) = \left( \begin{array} { c c } { { 0 } } & { { ( 1 - e ^ { - s _ { 2 } / \rho } ) ^ { - 1 } } } \\ { { - \gamma / \rho } } & { { - 1 / \rho } } \end{array} \right) \Phi _ { \rho } ( s _ { 2 } , s _ { 1 } ) , \qquad \Phi _ { \rho } ( s _ { 1 } , s _ { 1 } ) = { \bf I } _ { 2 } .\tag{H.2}
$$

The initial condition is $\Delta \theta ( s _ { 1 } ; \rho ) = 0$ and $\Delta m ( s _ { 1 } ; \rho ) = 1 / \rho$ . Thus, the continuous counterpart of the normalized discrete response $\Psi _ { \beta } ( T , t ) / \eta$ is

$$
\Psi ( s _ { 2 } , s _ { 1 } ; \boldsymbol { \rho } ) : = \Delta \theta ( s _ { 2 } ; \boldsymbol { \rho } ) = \mathbf { e } _ { 1 } ^ { \top } \Phi _ { \boldsymbol { \rho } } ( s _ { 2 } , s _ { 1 } ) \mathbf { e } _ { 2 } / \rho .
$$

Here ${ \mathbf e } _ { 1 } = ( 1 , 0 ) ^ { \top }$ selects the parameter component and ${ \mathbf e _ { 2 } } / \rho = ( 0 , 1 ) ^ { \top } / \rho$ represents the initial state. We define the continuous-time counterpart of the covariance coeficient $\Sigma _ { T } ^ { \beta }$ in (C.4) by

$$
\Sigma ( s _ { 2 } ; \rho ) = \int _ { 0 } ^ { s _ { 2 } } \Psi ( s _ { 2 } , s _ { 1 } ; \rho ) ^ { 2 } \biggl [ \mathrm { d i a g } \left( \frac { \bf r } { \pi _ { \theta ( s _ { 1 } ; \rho ) } ( \cdot | x ) } \right) - { \bf r } { \bf r } ^ { \top } \biggr ] \mathrm { d } s _ { 1 } .\tag{H.3}
$$

As the optimization time $s _ { 2 } \to \infty$ , evaluating the limiting integral in (H.3) gives

$$
\Sigma _ { \infty } = \frac { 1 } { 2 \gamma } \left[ \mathrm { d i a g } \left( \frac { \mathbf { r } } { \pi _ { \theta ^ { \star } } \left( \cdot \mid x \right) } \right) - \mathbf { r r } ^ { \top } \right] .
$$

The following lemma establishes convergence of the covariance coeficients and provides uniform relative error control. Its proof is given in the next subsection.

Lemma H.1 (Limits of the covariance coeficient). Fix $0 < \rho \leq 1 / ( 4 \gamma ) . I f \eta  0 , T \eta  s > 0 .$ , and $\beta = 1 - \eta / \rho$ , then $\Sigma _ { T } ^ { \beta } \to \Sigma ( s ; \rho )$ . Moreover,

$$
\Sigma ( s ; \rho ) \longrightarrow \Sigma _ { \infty } , \qquad \operatorname* { s u p } _ { { \bf v } : { \bf v } ^ { \top } \Sigma _ { \infty } { \bf v } > 0 } \Big | \frac { { \bf v } ^ { \top } \Sigma ( s ; \rho ) { \bf v } } { { \bf v } ^ { \top } \Sigma _ { \infty } { \bf v } } - 1 \Big | \longrightarrow 0 \quad ( s  \infty ) .
$$

Variance ratio at matched endpoints. We first note that the gradients at $\theta _ { T } ^ { ( 1 ) }$ and $\theta _ { T } ^ { ( 2 ) }$ are identical because the exact endpoints satisfy $\theta _ { T } ^ { ( 1 ) } = \theta _ { T } ^ { ( 2 ) }$ . For the first run, $T \eta _ { 1 } = \tau$ . Lemma G.1 and the representation of the exact path in Lemma 4.1 give $\lVert \theta _ { T } ^ { ( 1 ) } - \theta ( \tau ; \rho _ { 1 } ) \rVert _ { 2 } = O ( \eta _ { 1 } )$ . The continuity of $\nabla J ( \theta )$ therefore yields, as $\eta _ { 1 }  0$

$$
\nabla J ( \theta _ { T } ^ { ( 1 ) } ) = \nabla J ( \theta _ { T } ^ { ( 2 ) } )  \nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) .
$$

According to Lemma H.1, the two discrete covariance coeficients converge to their continuous counterparts. Thus, the quadratic forms in the variance expression in (D.1) for the two runs converge to $\nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) ^ { \top } \Sigma ( \tau ; \rho _ { 1 } ) \nabla J ( \theta ( \tau ; \rho _ { 1 } ) )$ and $\nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) ^ { \top } \Sigma ( \tau _ { 2 } ; \rho _ { 2 } ) \nabla J ( \theta ( \tau ; \rho _ { 1 } ) )$ , respectively.

To obtain the limiting variance ratio from the leading variance terms, we first verify these limiting quadratic forms are nonzero for all suficiently large fixed τ. As shown in the proof of Lemma $\mathrm { H . 1 } , \Sigma ( \tau ; \rho _ { 1 } )$ and $\Sigma ( \tau _ { 2 } ; \rho _ { 2 } )$ are positive definite on the reward-1 subspace and zero on its orthogonal complement. It therefore sufices to find a nonzero reward-1 coordinate of the common limiting gradient. Along the exact path $\theta ( \tau ; \rho _ { 1 } ) = \theta ^ { \star } + f ( \tau ; \rho _ { 1 } ) ( \theta _ { 0 } - \theta ^ { \star } )$ , fix any reward-1 response $y .$ Since $f ( \tau ; \rho _ { 1 } ) > 0$ and the policy has full support, $\left( \mathrm { A . 2 } \right)$ shows that its gradient coordinate vanishes exactly when $\theta _ { 0 } ( x , y ) - \theta ^ { \star } ( x , y )$ equals the average of $\theta _ { 0 } - \theta ^ { \star }$ under $\pi _ { \boldsymbol { \theta } ( \tau ; \rho _ { 1 } ) }$ . The derivative of this average with respect to τ is

$$
\begin{array} { r } { \partial _ { \tau } f ( \tau ; \rho _ { 1 } ) \mathrm { V a r } _ { y ^ { \prime } \sim \pi _ { \theta ( \tau ; \rho _ { 1 } ) } ( \cdot | x ) } [ \theta _ { 0 } ( x , y ^ { \prime } ) - \theta ^ { \star } ( x , y ^ { \prime } ) ] < 0 , } \end{array}
$$

because $\partial _ { \tau } f ( \tau ; \rho _ { 1 } ) < 0$ and the variance is positive by nonoptimal initialization and full support. The average is therefore strictly decreasing, so the chosen gradient coordinate can vanish at most once for $\tau > 0$ and is nonzero for all suficiently large τ. Both quadratic forms are consequently strictly positive for these τ. Using $\eta _ { 2 } / \eta _ { 1 }  \tau _ { 2 } / \tau$ and (D.1), we obtain the limiting variance ratio

$$
\operatorname* { l i m } _ { \eta _ { 1 } \to 0 } \frac { \mathrm { V a r } [ J ( \widehat { \theta } _ { T } ^ { ( 2 ) } ) ] } { \mathrm { V a r } [ J ( \widehat { \theta } _ { T } ^ { ( 1 ) } ) ] } = \frac { \tau _ { 2 } } { \tau } \frac { \nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) ^ { \top } \Sigma ( \tau _ { 2 } ; \rho _ { 2 } ) \nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) } { \nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) ^ { \top } \Sigma ( \tau ; \rho _ { 1 } ) \nabla J ( \theta ( \tau ; \rho _ { 1 } ) ) } .\tag{H.4}
$$

The relative bound in Lemma H.1 makes the quotient of quadratic forms in (H.4) tend to one, even though the common gradient tends to zero. It remains to determine $\tau _ { 2 } / \tau$

Long-time residual decay. To determine $\tau _ { 2 } / \tau$ , define the logarithmic decay rate $h _ { \rho } ( s ) = - \partial _ { s } \log f ( s ; \rho )$ Diferentiating (G.1) gives, for $s > 0$ ,

$$
h _ { \rho } ^ { \prime } ( s ) = h _ { \rho } ( s ) ^ { 2 } - ( h _ { \rho } ( s ) - \gamma ) / [ \rho ( 1 - e ^ { - s / \rho } ) ] , \qquad h _ { \rho } ( 0 ) = \gamma .\tag{H.5}
$$

Define $\kappa ( \rho ) = 2 \gamma / ( 1 + \sqrt { 1 - 4 \gamma \rho } )$ , the smaller root of $\rho \kappa ^ { 2 } - \kappa + \gamma = 0$ . We now show that $h _ { \rho } ( s ) \to \kappa ( \rho )$ as $s \to \infty$ . At $h _ { \rho } ( s ) = \kappa ( \rho )$ , (H.5) gives $h _ { \rho } ^ { \prime } ( s ) = - \kappa ( \rho ) ^ { 2 } e ^ { - s / \rho } / ( 1 - e ^ { - s / \rho } ) < 0$ . Since $h _ { \rho } ( 0 ) = \gamma < \kappa ( \rho )$ and $h _ { \rho } ( s )$ cannot reach $\kappa ( \rho )$ from below, we have $h _ { \rho } ( s ) < \kappa ( \rho )$ . Using (H.5), we obtain

$$
h _ { \rho } ^ { \prime } ( s ) \geq ( \kappa ( \rho ) - h _ { \rho } ( s ) ) ^ { 2 } - \kappa ( \rho ) ^ { 2 } e ^ { - s / \rho } / ( 1 - e ^ { - s / \rho } ) .
$$

This uses $\kappa ( \rho ) - \gamma = \rho \kappa ( \rho ) ^ { 2 }$ and $2 \rho \kappa ( \rho ) \leq 1$ . For any $0 < \varepsilon < \kappa ( \rho ) - \gamma$ , the bound gives $h _ { \rho } ^ { \prime } ( s ) \geq \varepsilon ^ { 2 } / 2$ whenever $h _ { \rho } ( s ) \leq \kappa ( \rho ) - \varepsilon$ and s is suficiently large. Hence $h _ { \rho } ( s ) > \kappa ( \rho ) - \varepsilon$ for all suficiently large s. With the upper bound $h _ { \rho } ( s ) < \kappa ( \rho )$ , we have $h _ { \rho } ( s ) \to \kappa ( \rho )$ as $s \to \infty$ . Integrating $h _ { \rho } ( s ) = - \partial _ { s } \log f ( s ; \rho )$ and using $f ( 0 ; \rho ) = 1$ gives

$$
- \log f ( s _ { 2 } ; \rho ) / s _ { 2 } = ( 1 / s _ { 2 } ) \int _ { 0 } ^ { s _ { 2 } } h _ { \rho } ( s _ { 1 } ) \mathrm { d } s _ { 1 } \to \kappa ( \rho ) \qquad \mathrm { a s ~ } s _ { 2 } \to \infty .
$$

Taking logarithms in (H.1) and applying the preceding limit to both endpoints gives

$$
\begin{array} { r l r } {  { \operatorname* { l i m } _ { \tau \to \infty } \tau _ { 2 } / \tau = \operatorname* { l i m } _ { \tau \to \infty } \frac { - \log f ( \tau ; \rho _ { 1 } ) / \tau } { - \log f ( \tau _ { 2 } ; \rho _ { 2 } ) / \tau _ { 2 } } } } \\ & { } & { = \kappa ( \rho _ { 1 } ) / \kappa ( \rho _ { 2 } ) = ( 1 + \sqrt { 1 - 4 \gamma \rho _ { 2 } } ) / ( 1 + \sqrt { 1 - 4 \gamma \rho _ { 1 } } ) < 1 . } \end{array}
$$

Substituting into (H.4) proves the double-limit variance ratio in Theorem 5.5.

## H.1 Proof of Lemma H.1

Proof. We pass to the continuous covariance, compute its long-time limit, and control relative error.

Discrete-to-continuous covariance. Fix $0 < \rho \leq 1 / ( 4 \gamma )$ and set $\beta = 1 - \eta / \rho$ . Let $\eta  0$ with $T \eta  s > 0$ and fix $0 < a < s$ . The updates in (3.5) and (C.2) give the following discrete counterpart of the transition matrix in (H.2).

$$
\left( 1 - \gamma \eta / \sum _ { k = 0 } ^ { t } \beta ^ { k } \quad \rho \beta / \sum _ { \beta } ^ { t } \beta ^ { k } \right) .
$$

For $a \leq t \eta$ , we compare it with the continuous one-step transition in (H.2). Taylor expansion gives

$$
\begin{array} { l } { { \Phi _ { \rho } ( ( t + 1 ) \eta , t \eta ) = \Phi _ { \rho } ( t \eta , t \eta ) + \eta \partial _ { s _ { 2 } } \Phi _ { \rho } ( s _ { 2 } , t \eta ) \big | _ { s _ { 2 } = t \eta } + O _ { a } ( \eta ^ { 2 } ) } } \\ { { = \left( \begin{array} { c c } { { 1 } } & { { \eta / ( 1 - e ^ { - t \eta / \rho } ) } } \\ { { - \gamma \eta / \rho } } & { { 1 - \eta / \rho } } \end{array} \right) + O _ { a } ( \eta ^ { 2 } ) . } } \end{array}
$$

For the discrete matrix, the first-row entries satisfy $\begin{array} { r } { 1 - \gamma \eta / \sum _ { k = 0 } ^ { t } \beta ^ { k } = 1 - \gamma \eta ^ { 2 } / \left[ \rho ( 1 - \beta ^ { t + 1 } ) \right] = 1 + O _ { a } ( \eta ^ { 2 } ) } \end{array}$ and $\textstyle \rho \beta / \sum _ { k = 0 } ^ { t } \beta ^ { k } = \eta \beta / ( 1 - \beta ^ { t + 1 } ) = \eta / ( 1 - e ^ { - t \eta / \rho } ) + O _ { a } ( \eta ^ { 2 } )$ . The second row agrees exactly with the continuous expansion. Thus the discrete transition matrix equals $\Phi _ { \rho } ( ( t + 1 ) \eta , t \eta ) + O _ { a } ( \eta ^ { 2 } )$ in operator norm, where $O _ { a } ( \cdot )$ suppresses constants that may depend on a. By (3.5) and $\left( \mathrm { C . 3 } \right)$ , the responses at iteration $t + 1$ to a unit noise input in one coordinate at update t are $\scriptstyle { \eta / \sum _ { k = 0 } ^ { t } \beta ^ { k } }$ for that parameter coordinate and $\eta / \rho$ for its EMA momentum before bias correction. After division by $\eta ,$ these responses are $1 / \textstyle \sum _ { k = 0 } ^ { t } \beta ^ { k }$ and $1 / \rho ,$ respectively. The preceding Taylor expansion gives $\Phi _ { \rho } ( ( t + 1 ) \eta , t \eta ) \mathbf { e } _ { 2 } / \rho = ( \eta / \bigl [ \rho ( 1 - e ^ { - \bar { t } \eta / \rho } ) \bigr ] , 1 / \rho - \eta / \rho ^ { 2 } ) ^ { \top } + O _ { a } ( \eta ^ { 2 } )$ Combining the $O _ { a } ( \eta )$ response error at $( t + 1 ) \eta$ with the $O _ { a } ( \eta ^ { 2 } )$ error in each subsequent transition gives

$$
\operatorname* { m a x } _ { a \leq t \eta < T \eta } \left| \Psi _ { \beta } ( T , t ) / \eta - \Psi ( T \eta , t \eta ; \rho ) \right| \leq C _ { a } \eta ( 1 + C _ { a } \eta ) ^ { T } + C _ { a } \eta ^ { 2 } \sum _ { k = 0 } ^ { T - 1 } ( 1 + C _ { a } \eta ) ^ { k } = O _ { a } ( \eta ) .\tag{H.6}
$$

Each one-step transition has operator norm at most $1 + C _ { a } \eta$ . The first term bounds the initial response diferences after propagation to $T \eta$ . The sum bounds the contribution of the subsequent transition errors. The final bound uses $( \bar { 1 } + C _ { a } \eta ) ^ { T } \leq ^ { ^ { \prime } } e ^ { C _ { a } T \eta }$ and $T \eta \to s$ . The above bounds give $| \Psi _ { \beta } ( T , t ) / \eta - \Psi ( s , s _ { 1 } ; \rho ) |  0$ as $T \eta \to s , t \eta \to s _ { 1 } > 0$ . Taking the limit in the first inequality in (C.5) gives, for $s _ { 1 } \leq s .$

$$
| \Psi ( s , s _ { 1 } ; \rho ) | \leq e ^ { \gamma s } [ 1 + \log ( s / s _ { 1 } ) ] .\tag{H.7}
$$

We first consider the covariance contribution over $[ a , s ]$ . Lemmas G.1 and 4.1 give max $_ { t \leq T } \| \theta _ { t } ^ { \beta } - \theta ( t \eta ; \rho ) \| _ { 2 } =$ ${ \cal O } ( \eta )$ . The safe-range residual bounds in the proof of Theorem 4.2 and (G.4) also give $\| \theta _ { t } ^ { \beta } \| _ { \infty } , \| \theta ( s _ { 1 } ; \rho ) \| _ { \infty } \leq$ $\| \theta ^ { \star } \| _ { \infty } + \| \theta _ { 0 } - \theta ^ { \star } \| _ { \infty }$ . Using (C.4), we obtain

$$
\begin{array} { r l } & { \eta \sum _ { t = \lceil a / \eta \rceil } ^ { T - 1 } ( \Psi _ { \beta } ( T , t ) / \eta ) ^ { 2 } \big [ \mathrm { d i a g } ( \mathbf { r } / \pi _ { \theta _ { t } ^ { \beta } } ( \cdot \mid x ) ) - \mathbf { r r } ^ { \top } \big ] } \\ & { \quad = \eta \displaystyle \sum _ { t = \lceil a / \eta \rceil } ^ { T - 1 } \Psi ( T \eta , t \eta ; \rho ) ^ { 2 } \big [ \mathrm { d i a g } ( \mathbf { r } / \pi _ { \theta ( t \eta ; \rho ) } ( \cdot \mid x ) ) - \mathbf { r r } ^ { \top } \big ] + O _ { a } ( \eta ) . } \end{array}
$$

The equality uses (H.6) to replace $\Psi _ { \beta } ( T , t ) / \eta$ by $\Psi ( T \eta , t \eta ; \rho )$ and the parameter approximation above to replace $\theta _ { t } ^ { \beta }$ by $\theta ( t \eta ; \rho )$ . The bounded parameter entries give uniformly bounded covariance matrices. Applying the analysis used to obtain (C.11) to the parameter approximation above gives a covariance replacement error of ${ \cal { O } } ( \eta )$ . Together with (H.7), these estimates give an error of $O _ { a } ( \eta )$ in each summand and a total error of $\eta T O _ { a } ( \eta ) = O _ { a } ( \eta )$ . The sum on the right is a Riemann sum for the continuous integrand in (H.3), so it converges to the integral over $[ a , s ]$ as $\eta  0$ and $T \eta \to s$ . For the omitted early samples, the first inequality in (C.5) gives

$$
\operatorname* { l i m } _ { \eta \to 0 } \operatorname* { s u p } _ { \eta } \sum _ { t = 0 } ^ { \lceil a / \eta \rceil - 1 } | \Psi _ { \beta } ( T , t ) / \eta | ^ { 2 } \leq e ^ { 2 \gamma s } \int _ { 0 } ^ { a } [ 1 + \log ( s / s _ { 1 } ) ] ^ { 2 } \mathrm { d } s _ { 1 } .
$$

For $t \geq 1$ , the inequality $\textstyle \sum _ { k = t + 1 } ^ { T } 1 / k \leq \log ( T / t )$ compares the sum with the integral. The $t = 0$ term $\eta | \Psi _ { \beta } ( T , 0 ) / \eta | ^ { 2 }$ is $O ( \eta \log ^ { 2 } ( 1 / \eta ) )$ and tends to zero. Equation (H.7) bounds the continuous contribution $\begin{array} { r } { \int _ { 0 } ^ { a } | \Psi ( s , s _ { 1 } ; \rho ) | ^ { 2 } \mathrm { d } s _ { 1 } } \end{array}$ by the same integral, which tends to zero as $a \to 0$ . Together with the uniform reward covariance bound, these estimates make both omitted contributions arbitrarily small by letting $a \to 0$ after taking $\eta  0$ . Combining this with the comparison on $[ a , s ]$ gives $\Sigma _ { T } ^ { \beta } \to \Sigma ( s ; \bar { \rho } )$ in operator norm as $\eta  0$ and $T \eta \to s .$

Long-time limit of the covariance coeficient. We establish the existence of li $_ { 1 _ { s \to \infty } } \Sigma ( s ; \rho )$ and compute its value. The coeficient matrix in (H.2) can be written as

$$
\left( { { \begin{array} { c c } { 0 } & { ( 1 - e ^ { - s / \rho } ) ^ { - 1 } } \\ { - \gamma / \rho } & { - 1 / \rho } \end{array} } } \right) = \left( { { \begin{array} { c c } { 0 } & { 1 } \\ { - \gamma / \rho } & { - 1 / \rho } \end{array} } } \right) + \left( { 0 } \begin{array} { c c } { e ^ { - s / \rho } / ( 1 - e ^ { - s / \rho } ) } \\ { 0 } & { 0 } \end{array}  \right) .\tag{H.8}
$$

Let $\mathbf { K } _ { \rho }$ denote the first matrix and $\mathbf { R } _ { \rho } ( s )$ denote the remainder. For $s \geq 1$ , the remainder’s operator norm satisfies $\| \mathbf { R } _ { \rho } ( s ) \| _ { \mathrm { o p } } = e ^ { - s / \rho } / ( 1 - e ^ { - s / \rho } ) \leq e ^ { - s / \rho } / ( 1 - e ^ { - 1 / \rho } )$ . Both eigenvalues of $\mathbf { K } _ { \rho }$ have negative real parts, so $\| e ^ { \mathbf { K } _ { \rho } v } \| _ { \mathrm { o p } } \leq C _ { 0 } \dot { e } ^ { - c v }$ for some $C _ { 0 } \geq 1$ and $c > 0$ depending only on $\rho$ and $\gamma .$ . Variation of constants in (H.2) gives

$$
\Phi _ { \rho } ( s , s _ { 1 } ) = e ^ { { \bf K } _ { \rho } ( s - s _ { 1 } ) } + \int _ { s _ { 1 } } ^ { s } e ^ { { \bf K } _ { \rho } ( s - u ) } { \bf R } _ { \rho } ( u ) \Phi _ { \rho } ( u , s _ { 1 } ) \mathrm { d } u .
$$

Taking operator norms and using the bound on $\| e ^ { \mathbf { K } _ { \rho } v } \| _ { \mathrm { o p } }$ yields, for $s \geq s _ { 1 } \geq 1$

$$
\begin{array} { r l } & { \| \Phi _ { \rho } ( s , s _ { 1 } ) \| _ { \mathrm { o p } } \leq C _ { 0 } e ^ { - c ( s - s _ { 1 } ) } + C _ { 0 } e ^ { - c ( s - s _ { 1 } ) } \displaystyle \int _ { s _ { 1 } } ^ { s } \| \mathbf { R } _ { \rho } ( u ) \| _ { \mathrm { o p } } e ^ { c ( u - s _ { 1 } ) } \| \Phi _ { \rho } ( u , s _ { 1 } ) \| _ { \mathrm { o p } } \mathrm { d } u } \\ & { \qquad \leq C _ { 0 } e ^ { - c ( s - s _ { 1 } ) } \exp \left( C _ { 0 } \displaystyle \int _ { s _ { 1 } } ^ { s } \| \mathbf { R } _ { \rho } ( u ) \| _ { \mathrm { o p } } \mathrm { d } u \right) } \\ & { \qquad \leq C e ^ { - c ( s - s _ { 1 } ) } . } \end{array}\tag{H.9}
$$

The second inequality uses Gronwall’s inequality. The next inequality uses $\begin{array} { r }  \int _ { s _ { 1 } } ^ { s } \| \mathbf { R } _ { \rho } ( u ) \| _ { \mathrm { o p } } \mathrm { d } u \le \int _ { 1 } ^ { \infty } e ^ { - u / \rho } / ( 1 - \ \end{array}$ $e ^ { - 1 / \rho } ) \mathrm { d } u = \rho e ^ { - 1 / \rho } / ( 1 - e ^ { - 1 / \rho } )$ . Therefore, there exists a constant C depending only on $\rho$ and $\gamma$ such that $\| \Phi _ { \rho } ( s , s _ { 1 } ) \| _ { \mathrm { o p } } \leq C e ^ { - c ( s - s _ { 1 } ) }$ for all $s \geq s _ { 1 } \geq 1$ . The lower bound $s _ { 1 } \geq 1$ keeps this integral uniformly bounded, whereas $\| { \mathbf { R } _ { \rho } ( u ) } \| _ { \mathrm { o p } } \sim \rho / u$ as $u \to 0$ . For $0 < s _ { 1 } < 1 , ( \mathrm { H } . 7 )$ bounds the parameter response on $[ s _ { 1 } , 1 ]$ by $e ^ { \gamma } [ 1 + \log ( 1 / s _ { 1 } ) ]$ . Integrating the second row of (H.2) with $\Delta m ( s _ { 1 } ; \rho ) = 1 / \rho$ gives $| \Delta m ( 1 ; \rho ) | \leq$ $\rho ^ { - 1 } + \gamma e ^ { \gamma } [ 1 + \log ( 1 / s _ { 1 } ) ]$ ]. Using $\Phi _ { \rho } ( s , s _ { 1 } ) = \Phi _ { \rho } ( s , 1 ) \Phi _ { \rho } ( 1 , s _ { 1 } )$ , (H.9), and the uniform reward covariance bound, the operator norm of the contribution of these inputs to (H.3) is at most $\begin{array} { r } { C _ { 1 } e ^ { - 2 c ( s - 1 ) } \int _ { 0 } ^ { 1 } [ 1 + \log ( 1 / s _ { 1 } ) ] ^ { 2 } \mathrm { d } s _ { 1 } } \end{array}$ Here $C _ { 1 } > 0$ is a constant independent of s and $s _ { 1 } . { \mathrm { ~ A s ~ } } s \to \infty$ , this contribution tends to zero, so the inputs with $s _ { 1 } \in [ 0 , 1 ]$ can be omitted when evaluating the covariance limit.

For inputs with $s _ { 1 } \geq 1$ , write $v = s - s _ { 1 }$ and define $g ( v ) = \mathbf { e } _ { 1 } ^ { \top } e ^ { \mathbf { K } _ { \rho } v } \mathbf { e } _ { 2 } / \rho$ . For fixed $v \geq 0$ , the decomposition in (H.8) gives su $\mathrm { p } _ { u \in [ s - v , s ] } \| \mathbf { R } _ { \rho } ( u ) \| _ { \mathrm { o p } } \to 0$ . The variation-of-constants formula above then yields $\Psi ( s , s - v ; \rho ) $ $g ( v )$ . Equation $\left( \mathrm { G . } \dot { 4 } \right)$ also gives $\theta ( s - v ; \rho ) \to \theta ^ { \star }$ . Extend the integrand in (H.3) after this change of variables by zero for $v > s - 1$ . By (H.9) and the uniform reward covariance bound established above, its operator norm is at most $C _ { 1 } e ^ { - 2 c v }$ , with $C _ { 1 }$ chosen to cover both covariance estimates. Since this bound is integrable on $[ 0 , \infty )$ , dominated convergence allows the limit and integral to be interchanged. Together with the vanishing contribution of $0 < s _ { 1 } < 1$ , this gives

$$
\operatorname* { l i m } _ { s \to \infty } \Sigma ( s ; \rho ) = \left( \int _ { 0 } ^ { \infty } g ( v ) ^ { 2 } \mathrm { d } v \right) \big [ \mathrm { d i a g } \left( \mathbf { r } / \pi _ { \theta ^ { \star } } ( \cdot  { | } x ) \right) - \mathbf { r r } ^ { \top } \big ] .
$$

Integration by parts in $\rho g ^ { \prime \prime } + g ^ { \prime } + \gamma g = 0$ with $g ( 0 ) = 0$ and $g ^ { \prime } ( 0 ) = 1 / \rho$ gives $\begin{array} { r } { \int _ { 0 } ^ { \infty } g ( v ) ^ { 2 } \mathrm { d } v = 1 / ( 2 \gamma ) } \end{array}$ , so this limit equals $\Sigma _ { \infty }$

Relative covariance bound. For a vector v, let $v _ { y }$ denote its entry corresponding to response y. By the covariance interpretation in (C.4), the null space of a reward covariance matrix consists of vectors v for which $r ( x , y ) v _ { y } / \pi _ { \theta } ( y \mid x )$ is constant over y. Full support and the existence of a reward-0 response force this constant to be zero, which is equivalent to $v _ { y } = 0$ on all reward-1 coordinates. Thus these matrices are positive definite on the span of $\{ \mathbf { e } _ { y } : r ( x , y ) = 1 \}$ and vanish on its orthogonal complement. With initial state $\mathbf { e } _ { 2 } / \rho ;$ , the first row of (H.2) gives $\Psi ( s , s _ { 1 } ; \rho ) > 0$ for $s _ { 1 } < s$ suficiently close to each fixed $s > 0$ . Equation (H.3) therefore gives $\Sigma ( s ; \rho )$ the same positive definiteness and null space as $\Sigma _ { \infty }$ . Let $\lambda _ { * } > 0$ denote the smallest eigenvalue of $\Sigma _ { \infty }$ restricted to this subspace. The relative error satisfies

$$
\operatorname* { s u p } _ { \mathbf { v } : \mathbf { v } ^ { \top } \Sigma _ { \infty } \mathbf { v } > 0 } \left| \mathbf { v } ^ { \top } \Sigma ( s ; \rho ) \mathbf { v } / ( \mathbf { v } ^ { \top } \Sigma _ { \infty } \mathbf { v } ) - 1 \right| \leq \| \Sigma ( s ; \rho ) - \Sigma _ { \infty } \| _ { \mathrm { o p } } / \lambda _ { * } \to 0 .
$$

Removing the reward-0 coordinates changes neither quadratic form, so the supremum can be restricted to the reward-1 subspace. On this subspace, the absolute diference of the quadratic forms is at most $\| \Sigma ( s ; \rho ) - \Sigma _ { \infty } \| _ { \mathrm { o p } } \cdot \| \mathbf { v } \| _ { 2 } ^ { 2 }$ , while $\mathbf { v } ^ { \top } \Sigma _ { \infty } \mathbf { v } \geq \bar { \lambda } _ { * } \| \mathbf { v } \| _ { 2 } ^ { 2 }$ □

## I Experimental details

## I.1 Bandit results

Settings. We consider a 10-armed bandit problem and maximize

$$
J ( \theta ) = \sum _ { y = 1 } ^ { 1 0 } \pi _ { \theta } ( y \mid x ) r ( x , y ) - \mathrm { K L } \left( \pi _ { \theta } ( \cdot \mid x ) \| \pi _ { \theta _ { 0 } } ( \cdot \mid x ) \right) .
$$

We set $\gamma = 1 , B = 1$ , and $\tau = 1$ , use $\eta = 1 / T$ for $T = 3 2 , 6 4 , \dots , 4 0 9 6$ , and compare fixed $\beta = 0 . 9$ and scaled $\beta = 1 - 4 \eta$ with $\beta = 0$ . Each configuration uses 50,000 independent training replicates, with $J ( \theta )$ evaluated exactly after training. We vary the reward vector and policy initialization across four settings:

• 3:7, uniform. Reward vector (1, 1, 1, 0, 0, 0, 0, 0, 0, 0); initial policy $\pi _ { \theta _ { 0 } } ( \cdot \ | \ x ) = ( 0 . 1 , . . . , 0 . 1 )$ . Results appear in Figures 1(a) and 1(b).

• 3:7, random. Reward vector $( 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 )$ ; initial policy $\pi _ { \theta _ { 0 } } ( \cdot \ | \ x ) \approx ( 0 . 0 3 8 8 , 0 . 0 7 1 3 , 0 . 1 5 0 2$ 0.3932, 0.0100, 0.0459, 0.0635, 0.0928, 0.0891, 0.0453). Results appear in Figures 1(c) and 1(d).

![](images/24938f5da3c00e3915d8cf9b89ae9c22ac054d7baa11fab6849f57313375aea4.jpg)  
(a) Gain and Noise.

![](images/670db7ef9e4d9160e366f060dcedd9b07caaf40f35ee11602093e2e69f93d0d3.jpg)  
(b) Win Probability.

![](images/4216d911429abb5d8768c96cb882a08b26b91c394abff9d0b3eb2cdc4a45e86a.jpg)  
(c) Gain and Noise.

![](images/9ef3fd696a59e89e147c6bc82601b1c2fe67cab154eb99f5141bdf2fe86087bb.jpg)  
(d) Win Probability.  
Figure 4: Numerical bandit results with 5 reward-one and 5 reward-zero arms under uniform initialization (4(a), 4(b)) and random policy initialization (4(c), 4(d)).

• 5:5, uniform. Reward vector (1, 1, 1, 1, 1, 0, 0, 0, 0, 0); initial policy $\pi _ { \theta _ { 0 } } ( \cdot \mid x ) = ( 0 . 1 , . . . , 0 . 1 )$ . Results appear in Figures 4(a) and 4(b).

• 5:5, random. Reward vector (1, 1, 1, 1, 1, 0, 0, 0, 0, 0); initial policy $\pi _ { \theta _ { 0 } } ( \cdot \ | \ x ) \approx ( 0 . 1 3 2 0 , 0 . 1 5 9 6 , 0 . 0 3 6 1$ 0.0342, 0.0221, 0.0846, 0.2399, 0.0667, 0.1895, 0.0353). Results appear in Figures 4(c) and 4(d).

Random initializations are drawn separately from Dirichlet $( 1 , \ldots , 1 )$ ) and are held fixed across learning rates, EMA coeficients, and training replicates.

Results. All four settings exhibit the same trends as η decreases: the standard deviation scales as $\sqrt { \eta }$ under both schedules, while the mean gain decays as η with fixed β and approaches a positive constant with scaled β. Consequently, the winning probability approaches $1 / 2$ in the fixed setting and 1 in the scaled setting. These results support the robustness of the theoretical predictions across reward vectors and policy initializations. Table 1 reports the SNR defined in Section 6.1 for each setting and learning rate.
<table><tr><td rowspan="2">Setting</td><td rowspan="2"> $\beta$ </td><td colspan="8">Learning rate η</td></tr><tr><td>1/32</td><td>1/64</td><td>1/128</td><td>1/256</td><td>1/512</td><td>1/1024</td><td>1/2048</td><td>1/4096</td></tr><tr><td rowspan="2">3:7, uniform</td><td>0.9</td><td>0.1271</td><td>0.1236</td><td>0.1732</td><td>0.1530</td><td>0.1183</td><td>0.0876</td><td>0.0632</td><td>0.0453</td></tr><tr><td>1-4η</td><td>0.0719</td><td>0.1118</td><td>0.3720</td><td>0.7229</td><td>1.1844</td><td>1.7900</td><td>2.6258</td><td>3.7647</td></tr><tr><td rowspan="2">3:7, random</td><td>0.9</td><td>0.1809</td><td>0.0904</td><td>0.1536</td><td>0.1393</td><td>0.1071</td><td>0.0796</td><td>0.0573</td><td>0.0406</td></tr><tr><td>1-4η</td><td>0.1220</td><td>0.0585</td><td>0.3178</td><td>0.6486</td><td>1.0712</td><td>1.6238</td><td>2.3713</td><td>3.3982</td></tr><tr><td rowspan="2">5:5, uniform</td><td>0.9</td><td>0.2625</td><td>0.0402</td><td>0.1546</td><td>0.1726</td><td>0.1524</td><td>0.1226</td><td>0.0931</td><td>0.0673</td></tr><tr><td>1-4η</td><td>0.1849</td><td>0.0310</td><td>0.2455</td><td>0.7047</td><td>1.4036</td><td>2.3824</td><td>3.7419</td><td>5.5492</td></tr><tr><td rowspan="2">5:5, random</td><td>0.9</td><td>0.3162</td><td>0.0083</td><td>0.1343</td><td>0.1539</td><td>0.1310</td><td>0.1002</td><td>0.0746</td><td>0.0534</td></tr><tr><td> $1 - 4 \eta$ </td><td>0.2428</td><td>0.1064</td><td>0.1862</td><td>0.6293</td><td>1.2251</td><td>1.9919</td><td>3.0391</td><td>4.4557</td></tr></table>

Table 1: SNR of the objective diference between independent EMA and no-EMA runs. Each ratio gives the numbers of reward-one and reward-zero arms, respectively.

## I.2 Large language model experiments

Settings. We use Qwen3-4B-Base and Qwen2.5-32B models using the verl training framework. The parameters are set as follows.

Prompts and rewards. We use the original prompts provided in the DAPO-17k English dataset. As in DAPO’s implementation, we restrict answer extraction to the final 300 characters of each response. A response receives a reward of 1 if the extracted answer matches the ground truth, and 0 otherwise. The same rule applies to responses truncated at the length limit; truncation alone does not force a zero reward.

Validation. To assess the optimization performance, we evaluate on held-out data drawn from the same distribution as the training data. We construct this validation set by randomly holding out 1,000 problems from the DAPO-17k English dataset and excluding them from training. Evaluation is performed every 20 optimization steps, with an additional evaluation at step 150 for runs ending at that step. For each problem, we sample eight responses at a temperature of 0.7 and top-p of 1.0, with a maximum response length of 4,096 tokens. Each response receives a binary correctness score from the answer verifier. We report the average score over all 8,000 responses, equivalently the mean per-problem accuracy. All runs use the same held-out problems.

<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Training dataset</td><td>DAPO-17k (English)</td><td>Generations per prompt</td><td>8</td></tr><tr><td>Prompts per batch</td><td>16</td><td>Updates per RL step</td><td>1</td></tr><tr><td>Max prompt length</td><td>2,048</td><td>Max response length</td><td>4,096</td></tr><tr><td>Rollout temperature</td><td>1.0</td><td>Rollout top-p</td><td>1.0</td></tr><tr><td>Clip ratio</td><td>0.2</td><td>Gradient clipping norm</td><td>1.0</td></tr><tr><td>KL coefficient</td><td>0.01</td><td>Training generation seeds</td><td>1001, 2001, 3001</td></tr></table>

Table 2: Shared training settings for the LLM experiments.