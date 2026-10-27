# VaLoP: Value Gradient Lookahead and Pullback for Reward-Guided Diffusion

Kwanyoung Kim Department of AI, GIST k0.kim@gist.ac.kr

![](images/577854ad682e7dd280946fd3c366308152d77d03715e365e9808874bea450d68.jpg)  
Figure 1. Our proposed method guides both image generators and robot policies. It succeeds where the unguided model and StitchVM [19] miss the GenEval prompt, and, from the same initial state, where the base policy and AGF [25] fail the task or finish much later. √/X: GenEval correctness or task success. Bars: steps to success or the time limit.

## Abstract

Reward guidance steers a pretrained diffusion or flow model toward images or actions that a reward model prefers, at sampling time and without retraining the model for each reward. It follows the gradient of a value function, the expected reward given the noisy state. The two common estimates of this gradient, gradients through a one-step denoised estimate and learned value models, are least accurate at high noise, where the coarse structure of the sample is decided. Interestingly, we fnd that both errors are posterior covariances, which grow with the uncertainty about the clean sample, and hence at higher noise. To reduce them, we evaluate the value at a less noisy state, one sampler step ahead, where this uncertainty is smaller, and pull its gradient back to the current state. This lookahead places both

estimators at the two ends of one family and reveals that an intermediate lookahead requires one additional term a correction through the generator. We therefore propose Value Gradient Lookahead and Pullback (VaLoP), which computes this correction with one vector-Jacobian product, and VaLoP-Fast, which amortizes it with a small head that, for text-to-image models, is trained with a single reward and reused for the others. Across various models, rewards, and benchmark datasets, both robustly outperform existing value-gradient guidance on the target reward, VaLoP-Fast at about the cost of learned-value guidance. Both extend to critic-guided robot policies, where VaLoP-Fast attains the highest success rates on LIBERO and RoboCasa.

Project page: real-lab-gist.github.io/valop

![](images/1928f2360bb5fd3d2c78071ecb44e4cc548f13e58b648b6f61cf3e938554340a.jpg)  
Figure 2. Comparison and pipeline overview. (a) Where the value is read: DPS [6] at the denoised estimate (s = 0), StitchVM [19] at zt (s = t), and VaLoP in between, at a lookahead state zs (①), whose gradient () it pulls back () into the guidance ${ \mathbf g } _ { t } ^ { \mathrm { V a L o P } }$ toward the rewarded mode (). AGF [25] learns the guidance with a head trained by adjoint rollouts. (b) Per-step computation and training. VaLoP needs no extra training, and VaLoP-Fast amortizes its correction with one head for all rewards, avoiding the generator backward pass.

## 1. Introduction

Diffusion and flow models [14, 32, 35, 59] generate images and robot actions, and they are increasingly steered toward outputs that a reward model prefers, such as images with higher aesthetic, preference, or text-alignment scores [47, 52, 69, 70] and actions with higher critic value [25]. One route fine-tunes the generator on the reward, by backpropagation through sampling [7], reinforcement learning [34, 74], or stochastic optimal control [12], but it must be repeated for every reward and changes the generator for all prompts. Inference-time methods instead keep the generator frozen and sample from the reward-tilted distribution of [62]. Search-based methods select or resample among many candidate samples [30, 38, 56], at a cost that grows with the number of candidates. Reward guidance steers a single sample by adding, at each step, the gradient of a value function, the expected reward given the current noisy state.

In this work, we focus on reward guidance, which avoids the many candidates of search. Specifically, Fig. 2a contrasts existing estimates of this gradient. Training-free guidance, such as diffusion posterior sampling (DPS) [6], differentiates the reward of the one-step denoised estimate of the clean sample (Fig. 2a). It backpropagates through the denoiser, the decoder, and the reward model at every step (Fig. 2b), and the reward of this estimate, an average over plausible clean samples, differs from the expected reward. Other methods build on this estimate [1, 72, 73] or replace it with clean samples from a few-step solver [27] or a distilled flow map [22, 23, 45, 50]. In contrast, learned guidance evaluates a network trained on noisy states, so that the gradient avoids the denoised estimate and the decoder. The stitched value model (StitchVM) [19] predicts the reward of a clean sample from its noised version (Fig. 2b), and adjoint guidance flow (AGF) [25] trains a lightweight guidance network that predicts adjoint gradients along the sampling paths of a robot policy (Fig. 2b).

However, neither estimate is accurate at high noise, early in sampling. Interestingly, we find that both errors stem from one source, a posterior covariance that grows with the uncertainty about the clean sample, and hence at higher noise. We verify this on a two-dimensional mixture with a known value and on a text-to-image (T2I) model, SD3.5 [14, 60] (Fig. 3).

This suggests evaluating the value at a less noisy state, where this uncertainty is smaller. VaLoP takes one deterministic sampler step from the current state to a less noisy lookahead state, evaluates the value model there, and pulls its gradient back to the current state. The lookahead ratio between the two noise levels places learned-value guidance and DPS at the two ends of one family. Beyond the value gradient at the lookahead state, an intermediate lookahead then requires a correction term, which is the vector-Jacobian product (VJP) of the generator's velocity with that gradient (Fig. 2a). In our T2I ablation, this correction brings most of the gain over learned-value guidance.

While this correction improves the guidance, it requires one VJP through the generator at every step. To address this, we propose VaLoP-Fast, which amortizes the correction with a small head, while the generator and the value model stay frozen (Fig. 2b). Since the correction depends on the reward only through the value gradient it acts on, one head per T2I model serves all rewards. In the same implementation, VaLoP-Fast runs faster than StitchVM while reaching a higher target reward (Tabs. 1 and 3). For robot policies, one head per task, trained by backpropagating through one step rather than all steps as in AGF, adds only 3 to 5% to the inference time of the base policy. We review related work in Sec. A, and our contributions are as follows.

• We show that the errors of DPS and learned-value guidance are posterior covariances that grow with the uncertainty about the clean sample, and that the lookahead ratio places both methods at the two ends of one family.

• To address this high-noise error, we propose VaLoP, which pulls the value gradient at a lookahead state back with a correction, and VaLoP-Fast, which amortizes it with a head shared across T2I rewards and trained per task for robot policies.

• Across T2I models, rewards, and benchmarks, VaLoP and VaLoP-Fast robustly outperform StitchVM and DPS on the target reward. Extended to critic-guided robot policies, VaLoP-Fast attains the highest success rates on LIBERO and RoboCasa.

## 2. Background

## 2.1. Flow-Based Models

Flow-based generative models transport a noise sample $\epsilon \sim \mathcal { N } ( 0 , I )$ to a data sample $z _ { \mathrm { 0 } }$ along the ODE $d z _ { t } ~ =$ ${ \pmb v } _ { \theta } ( z _ { t } , t ) d t$ , where the velocity field vθ is trained with the conditional flow matching objective [32, 35],

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { t , z _ { 0 } , \epsilon } \Big \| ( \epsilon - z _ { 0 } ) - v _ { \theta } ( z _ { t } , t ) \Big \| ^ { 2 } ,\tag{1}
$$

where $z _ { t } = ( 1 - t ) z _ { 0 } +$ - te is the linear interpolant, so that $t = 0$ is data and $t = 1$ is noise. We write $p$ for the distribution of generated samples and $p _ { t }$ for the marginal of ${ \boldsymbol { z } } _ { t }$ under $p .$ The posterior mean $\mathbb { E } [ z _ { 0 } ~ \mid ~ z _ { t } ]$ is estimated by Tweedie's formula [13, 26] as $\hat { z } _ { 0 \mid t } = z _ { t } - t v _ { \theta } ( z _ { t } , t )$ which is exact for the optimal velocity of Eq. (1), and we write $D _ { t } = \partial \hat { z } _ { 0 \mid t } / \partial z _ { t }$ for its Jacobian. For T2I models with prompt $c ,$ classifier-free guidance (CFG) [21] at scale ω replaces ${ \pmb v } _ { \theta }$ by ${ \pmb v } _ { \theta } ( \cdot , \emptyset ) + \omega \big ( { \pmb v } _ { \theta } ( \cdot , c ) - { \pmb v } _ { \theta } ( \cdot , \emptyset ) \big )$ , and ${ \pmb v } _ { \theta }$ denotes this guided velocity from here on. For a $\mathrm { V L A }$ policy, $z _ { \mathrm { 0 } }$ is an action chunk, and ${ \pmb v } _ { \theta }$ is also conditioned on the observation [4, 55]. One deterministic sampler step from t

to $s < t$ is the map

$$
\begin{array} { r } { \Phi _ { t  s } ( z ) = z - ( t - s ) { \pmb v } _ { \theta } ( z , t ) = \frac { s } { t } z + ( 1 - \frac { s } { t } ) \hat { z } _ { 0 | t } ( z ) , } \end{array}\tag{2}
$$

which moves ${ \boldsymbol { z } } _ { t }$ toward its posterior mean. It is a one-step approximation of the flow map of the ODE from t to s and returns $\hat { z } _ { 0 \mid t }$ at $s = 0 [ 5 7 ]$ . We call $\Phi _ { t  s } ( z _ { t } )$ the lookahead state at $s ,$ which shows where the sampler heads from $z _ { t }$

## 2.2. Reward-Tilted Sampling

Given a reward $r ( z _ { 0 } )$ , such as a critic value or an aesthetic, preference, or text-alignment score $[ 2 5 , 4 7 , 5 2 , 6 9 ]$ , rewardtilted sampling targets $p ^ { \star } ( z _ { 0 } ) \propto p ( z _ { 0 } ) e ^ { \lambda r ( z _ { 0 } ) }$ , which maximizes the expected reward minus $\lambda ^ { - 1 } \mathrm { K L } ( p ^ { \star } \parallel p )$ [62]. Its noisy marginals are $p _ { t } ^ { \star } \propto p _ { t } e ^ { \lambda \mathcal { V } _ { t } }$ , where $\begin{array} { r } { \mathcal { V } _ { t } ( z _ { t } ) ~ = } \end{array}$ $\lambda ^ { - 1 } \log \mathbf { \bar { E } } [ e ^ { \lambda r ( \mathbf { \bar { z } } _ { 0 } ) } \mid z _ { t } ]$ is the soft value [12, 30, 62], and by Tweedie's formula [13] the tilt shifts the posterior mean,

$$
\mathbb { E } _ { p ^ { \star } } [ z _ { 0 } \mid z _ { t } ] = \mathbb { E } _ { p } [ z _ { 0 } \mid z _ { t } ] + \frac { t ^ { 2 } } { 1 - t } \lambda \nabla _ { z _ { t } } \mathcal { V } _ { t } ( z _ { t } ) .\tag{3}
$$

Guidance therefore replaces $\hat { z } _ { 0 \mid t }$ by $\begin{array} { r } { \hat { z } _ { 0 \mid t } + \lambda \frac { t ^ { 2 } } { 1 - t } { g } _ { t } } \end{array}$ before each sampler step, where $\mathbf { \sigma } _ { \mathbf { \sigma } _ { \mathbf { \sigma } _ { \mathbf { \lambda } } } } \mathbf { \sigma } _ { \mathbf { \sigma } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } } \mathbf { \sigma } _ { \mathbf { \lambda } _ { \mathbf { \lambda } } }$ estimates $\vec { \nabla } _ { z _ { t } } \mathcal { V } _ { t }$ and λ is tuned per reward in practice. To leading order in the reward scale, the soft value reduces to the value [19], the expected reward under the generator's posterior,

$$
V _ { t } ( \boldsymbol { z } _ { t } ) = \mathbb { E } [ r ( \boldsymbol { z } _ { 0 } ) \mid \boldsymbol { z } _ { t } ] .\tag{4}
$$

Like existing methods from DPS to critic guidance [6, 19, 25], VaLoP estimates $\nabla _ { z _ { t } } V _ { t }$ (Sec. C).

## 2.3. Estimating the Value Gradient

Reward guidance methods differ in the state at which they differentiate the reward or a value (Fig. 2a). DPS [6] differentiates the reward of the posterior mean,

$$
\pmb { g } _ { t } ^ { \mathrm { D P S } } = \nabla _ { z _ { t } } r \big ( \hat { z } _ { 0 | t } \big ) = D _ { t } ^ { \top } \nabla r \big ( \hat { z } _ { 0 | t } \big ) ,\tag{5}
$$

which backpropagates through the denoiser, the decoder, and the reward model at every step. Here $\nabla _ { z _ { t } }$ differentiates through every map applied to $z _ { t } .$ and ∇ is the gradient in the first argument at the point shown. When the uncertainty about the clean sample is large, as at high noise, $\hat { z } _ { 0 \mid t }$ averages over many clean samples, so $r \big ( \hat { z } _ { 0 | t } \big )$ differs from the expected reward $\mathbb { E } [ r ( z _ { 0 } ) \mid z _ { t } ]$ by a Jensen gap [6].

StitchVM [19] builds a value model $V _ { \psi } ( z _ { t } , t )$ by stitching the first blocks of the frozen denoiser to the reward model from one of its early blocks onward. It is trained by regression on noised latents of dataset images $z _ { \mathrm { 0 } } ~ \sim ~ q$ from AVA [40] and HPDv2 [69] under an empty prompt,

$$
\operatorname* { m i n } _ { \psi } \mathbb { E } _ { t , z _ { 0 } \sim q , \epsilon } { \left\| V _ { \psi } ( z _ { t } , t ) - r ( z _ { 0 } ) \right\| } ^ { 2 } ,\tag{6}
$$

![](images/cb1065814ba8dfa979813d29d59e883293cb534ff3237ba4d1279a366f8213ac.jpg)

![](images/5d0b73ff5764ebf67d7db0df48c77001fb493de60e814fa643fc65d04c5b49a4.jpg)

![](images/57186b46434d3041bf5f6767d340fa58ceed53a71904382ddfaeac0b017f5be6.jpg)  
Figure 3. Value-gradient error at high noise. (a, b) Mixture with a misspecified value model and known exact value: samples at the best strength (a) and cosine to the exact gradient (b, VaLoP: ∇Vψ at zs). (c) HPSv2 gain per guided step on SD3.5-Medium (mean ± s.d.).

in practice on the reward-model features of $z _ { \mathrm { 0 } } ,$ so that $V _ { \psi }$ approximates the value of Eq. (4) under the data q rather than under the generator $p .$ It guides with

$$
\pmb { g } _ { t } ^ { \mathrm { S V M } } = \nabla _ { \pmb { z } _ { t } } V _ { \psi } ( \pmb { z } _ { t } , t ) ,\tag{7}
$$

which requires one backward pass through the value model. AGF [25] guides with a lightweight network $h _ { \phi } ( z _ { t } , t )$ trained on its own guided paths to regress the costate,

$$
g _ { t } ^ { \mathrm { A G F } } = \pmb { h } _ { \phi } ( \pmb { z } _ { t } , t ) \approx \Big ( \frac { \partial \pmb { z } _ { 0 } } { \partial \pmb { z } _ { t } } \Big ) ^ { \top } \nabla r ( \pmb { z } _ { 0 } ) ,\tag{8}
$$

where the costate is the reward gradient at the endpoint $z _ { \mathrm { 0 } }$ of the path, pulled back to ${ \boldsymbol { z } } _ { t }$ with the guidance held fixed. It needs no backward pass at inference. All three thus replace $\nabla _ { z _ { t } } V _ { t }$ by a proxy that is accurate at low noise, and Sec. 3 examines how each departs from it.

## 3. Motivation

In this section, we show that the errors of all three estimators of Sec. 2.3 grow with the uncertainty about the clean sample given $z _ { t } ,$ and hence at higher noise, and verify this on a twodimensional mixture and on SD3.5 (Fig. 3).

Value gradient as a posterior covariance. Lemma 1 writes the target $\nabla V _ { t }$ of all three estimators and the Jacobian $D _ { t }$ of the posterior mean as covariances under the posterior over clean samples (proof in Sec. B.1).

Lemma 1. Let $z _ { t } ~ = ~ ( 1 - t ) z _ { 0 } + t \epsilon$ with $t \in \mathsf { \Gamma } ( 0 , 1 )$ $\epsilon \sim \mathcal { N } ( 0 , I )$ independent of z0, and $z _ { \mathrm { 0 } }$ drawn from any distribution with finite second moment, and let r have finite variance. Then the value $V _ { t } ( z ) = \mathbb { E } [ r ( z _ { 0 } ) \mid z _ { t } = z ]$ and the Jacobian $D _ { t }$ of the exact posterior mean satisfy

$$
\begin{array} { r } { \nabla V _ { t } ( z ) = \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \big [ z _ { 0 } , r ( z _ { 0 } ) \big | z _ { t } = z \big ] , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { D _ { t } ( z ) = \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \ [ z _ { 0 } \mid z _ { t } = z ] . } \end{array}\tag{10}
$$

Eq. (10), which applies Eq. (9) to each coordinate $r ( z _ { 0 } ) = z _ { 0 , i } .$ , is the second-order Tweedie identity [39, 72].

One-step estimate (DPS). For the exact denoiser of $p ,$ Eqs. (5) and (10) show that DPS is the covariance of Eq. (9) with r replaced by its linearization τ at $\hat { z } _ { 0 \mid t }$ , so that

$$
\begin{array} { r } { { \pmb g } _ { t } ^ { \mathrm { D P S } } - \nabla V _ { t } = - \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \big [ { \pmb z } _ { 0 } , r ( { \pmb z } _ { 0 } ) - { \bar { r } } ( { \pmb z } _ { 0 } ) \mid { \pmb z } _ { t } \big ] . } \end{array}\tag{11}
$$

DPS thus misses the part of the reward that is nonlinear over the posterior. For a smooth reward, this part vanishes as the posterior concentrates at low noise, and it can be large when the posterior spans several modes.

Learned value (StitchVM). A value model is regressed onto the reward of clean samples from a data distribution q given their noised versions, such as dataset images under an empty prompt in StitchVM or labeled data for a noisy classifier [11]. It converges to $V _ { t } ^ { q } ( z ) = \mathbb { E } _ { q } [ r ( z _ { 0 } ) \mid z _ { t } = z ]$ , so that $\pmb { g } _ { t } ^ { \mathrm { S V M } } = \nabla V _ { t } ^ { q }$ , and applying Eq. (9) under q and under the generator distribution p gives

$$
\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { S V M } } - \nabla V _ { t } = \frac { 1 - t } { t ^ { 2 } } \left( \mathrm { C o v } _ { q } - \mathrm { C o v } _ { p } \right) \left[ \boldsymbol { z } _ { 0 } , r ( \boldsymbol { z } _ { 0 } ) \mid \boldsymbol { z } _ { t } \right] . } \end{array}\tag{12}
$$

At low noise, both posteriors concentrate around $z _ { t } ,$ where $q$ has positive density, and the difference vanishes. At high noise, the noisy state is weakly informative about the clean sample, so each posterior approaches its prior, q or $p ,$ and the gradient follows the reward averaged over the data rather than over the generator's samples. Training on the generator's samples, as in SVDD [30], removes this difference but leaves a regression target with variance $\mathrm { V a r } [ r ( z _ { 0 } ) ~ | ~ z _ { t } ]$ which grows with the noise level on average.

Learned gradient (AGF). A guidance network can instead regress the gradient itself on the generator's own guided paths, as in AGF. Its target in Eq. (8) is the reward gradient at the single endpoint $z _ { \mathrm { 0 } }$ of the path from $z _ { t }$ , rather than its average over the posterior. For a linear reward $r ( z _ { 0 } ) = { \pmb w } ^ { \top } z _ { 0 }$ , for which DPS is exact, a head that matches this target at zero guidance gives

$$
\pmb { g } _ { t } ^ { \mathrm { A G F } } - \nabla V _ { t } = \bigg ( \frac { \partial z _ { 0 } } { \partial z _ { t } } - D _ { t } \bigg ) ^ { \top } \pmb { w } ,\tag{13}
$$

where $\partial z _ { 0 } / \partial z _ { t }$ compounds the Jacobians $D _ { \tau } ,$ scaled posterior covariances, along this path, while $D _ { t }$ is that at ${ \boldsymbol { z } } _ { t } .$ Both approach I at low noise, and for Gaussian data $\| \pmb { g } _ { t } ^ { \mathrm { A G F } } \| / \| \tilde { \nabla V _ { t } } \|  \infty$ at high noise (Sec. B.2). All three errors vanish at low noise and grow as the posterior widens.

Verification with the exact value. To verify these predictions where the exact value is known, we use a twodimensional mixture (Fig. 3a, b and Sec. D.1). DPS loses alignment with the exact value gradient as the noise grows and contracts the rewarded mode toward its center. For StitchVM, we train a value model on data in which the rewarded mode is rarer and displaced. By Eq. (12), this mismatch matters mainly at high noise, where the mode a sample joins is decided, and the value model is indeed least aligned at the highest noise and leaves less mass on the rewarded mode than the target. AGF loses alignment faster than DPS and places many samples off the data manifold. At high noise, the same value model is far more accurate at the less noisy lookahead state, and our method, VaLoP, which pulls its gradient back from there (Sec. 4), comes closest to the target.

Verification at scale. Since the exact value is unknown at scale, we instead measure the reward gain of one guided step at each noise level on SD3.5 (Sec. D.2). Here, q and p differ without any construction, since StitchVM is trained on dataset images under an empty prompt while the generator samples prompt-conditioned images. In line with Eqs. (11) and (12), DPS and StitchVM steps raise HPSv2 less than half as much at high noise as at low noise (Fig. 3c). VaLoP matches StitchVM at low noise, where the posterior is narrow, and gains most at intermediate noise.

Both checks thus support the analysis: the value-gradient error grows with the noise, and a value model evaluated at a less noisy state and pulled back reduces it. Sec. 4 develops this lookahead and its trade-off with the bias of the step.

## 4. Method: VaLoP

Following Sec. 3, VaLoP evaluates a value model $V _ { \psi }$ , such as StitchVM's, at a less noisy lookahead state and pulls its gradient back (Fig. 2a). For robot policies, a critic read at the denoised action plays the role of $V _ { \psi }$ Sec. 4.1 defines this guidance, with learned-value guidance and DPS as its two ends, Sec. 4.2 shows that the lookahead reduces the posterior mismatch, Sec. 4.3 amortizes the guidance with a small head, and Sec. 4.4 describes the implementation for T2I models and robot policies.

## 4.1. A Unified Family of Value Gradients

For $s \in [ 0 , t ]$ , let $z _ { s } = \Phi _ { t  s } ( z _ { t } )$ be the lookahead state, reached from ${ \boldsymbol { z } } _ { t }$ by one deterministic step of size $t - s$ with the velocity at $z _ { t } \left( \mathrm { E q . } \left( 2 \right) \right)$ , and call $s / t$ the lookahead ratio. We propose VaLoP, which guides at ${ \boldsymbol { z } } _ { t }$ with the gradient of the value model at the lookahead state,

$$
\begin{array} { r l } & { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \nabla _ { \pmb { z } _ { t } } V _ { \psi } \big ( \Phi _ { t  s } ( \pmb { z } _ { t } ) , s \big ) } \\ & { \quad \quad = \pmb { g } _ { s } \underbrace { - ( t - s ) J _ { v } ^ { \top } \pmb { g } _ { s } } _ { \pmb { \xi } _ { t } } , } \end{array}\tag{14}
$$

where $\pmb { g } _ { s } = \nabla V _ { \psi } ( z _ { s } , s )$ and $J _ { v } = \partial _ { z } { v _ { \theta } } ( z _ { t } , t )$ . The correction $\xi _ { t } ,$ , a VJP through the generator, carries $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ back to ${ \boldsymbol { z } } _ { t }$ the state that the guidance updates.

The step moves ${ \boldsymbol { z } } _ { t }$ toward $\hat { z } _ { 0 \mid t }$ , so the lookahead state lies between ${ \boldsymbol { z } } _ { t }$ , where learned-value guidance evaluates the value, and $\hat { z } _ { 0 \mid t } .$ , where DPS evaluates the reward. Proposition 1 shows that the guidance moves between the two estimators in the same way (proofs in Secs. B.3 and B.4).

Proposition 1. Let $V _ { \psi }$ and vθ be differentiable. Then

$$
\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \underbrace { \frac { s } { t } \pmb { g } _ { s } } _ { l e a r n e d - \nu a l u e f o r m } + \underbrace { \left( 1 - \frac { s } { t } \right) D _ { t } ^ { \top } \pmb { g } _ { s } } _ { D P S f o r m } . } \end{array}\tag{15}
$$

In particular, $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \pmb { g } _ { t } ^ { \mathrm { S V M } } } \end{array}$ at $s = t ,$ and $i f V _ { \psi } ( \cdot , 0 ) = r$ then $\pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \pmb { g } _ { t } ^ { \mathrm { D P S } }$ at $s = 0 .$

The first term of Eq. (15) is the learned-value guidance of Eq. (7), evaluated at the lookahead state, and the second is the DPS guidance of Eq. (5), with the reward gradient at the posterior mean replaced by the value gradient at $z _ { s }$ . The lookahead ratio thus sets where the guidance lies between the two estimators, and Sec. 4.2 shows how this choice trades the errors of the two ends.

Finally, VaI ${ \bf \Lambda } _ { \mathcal { A } } \mathrm { o } { \bf P }$ guides each step with this gradient,

$$
z _ { t - \Delta t } = z _ { t } - \Delta t v _ { \theta } ( z _ { t } , t ) + \eta g _ { t } ^ { \mathrm { V a L o P } } ,\tag{16}
$$

with the strength η (Alg. 1).

## 4.2. Error Analysis of VaLoP

With the Jacobian $P _ { s } = I - ( t - s ) J _ { v }$ of the step, $g _ { t } ^ { \mathrm { V a L o P } } =$ $P _ { s } ^ { \top } { \pmb g } _ { s }$ , and its error relative to $\nabla V _ { t }$ (Eq. (4)) splits into a value-model error and the bias of the step,

$$
\begin{array} { r l } { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } - \nabla V _ { t } ( \pmb { z } _ { t } ) = \pmb { P } _ { s } ^ { \top } \left[ \nabla V _ { \psi } ( \pmb { z } _ { s } , s ) - \nabla V _ { s } ( \pmb { z } _ { s } ) \right] } & { } \\ { + \left[ \pmb { P } _ { s } ^ { \top } \nabla V _ { s } ( \pmb { z } _ { s } ) - \nabla V _ { t } ( \pmb { z } _ { t } ) \right] , } & { } \end{array}\tag{17}
$$

where the second term vanishes at $s \ = \ t$ and equals the DPS error at $s = 0 \ ( \mathrm { E q . ( 1 1 ) }$ , Sec. B.5). For a value model trained as in StitchVM, $\begin{array} { r } { V _ { \psi } ( \cdot , s ) = V _ { s } ^ { q } } \end{array}$ , so the first term is the pulled-back gradient of the value error $V _ { s } ^ { q } - V _ { s }$ (Eq. (12) at level s), which comes from the mismatch between the posteriors under $q$ and $p$ that the next proposition bounds.

Proposition 2. Let $\mathrm { K L } ( p \| q ) < \infty , | r | \leq B ,$ and $q _ { s }$ be the marginal $o f z _ { s }$ under $q ,$ and let $\Delta _ { s } = \mathbb { E } _ { z _ { s } \sim p _ { s } } \mathrm { K L } \big ( p ( z _ { 0 } \mid$ $z _ { s } ) \parallel q \big ( z _ { 0 } \mid z _ { s } \big ) \big )$ be the posterior mismatch at level s. Then

$$
\Delta _ { s } = \mathrm { K L } ( p \parallel q ) - \mathrm { K L } ( p _ { s } \parallel q _ { s } ) ,\tag{18}
$$

which is nondecreasing in s with $\Delta _ { 0 } ~ = ~ 0$ and $\begin{array} { r l } { \Delta _ { 1 } } & { { } = } \end{array}$ $\mathrm { K L } ( p \| q )$ , and $\mathbb { E } _ { z _ { s } \sim p _ { s } } \big | V _ { s } ^ { q } ( z _ { s } ) - V _ { s } ( z _ { s } ) \big | \le B \sqrt { 2 \Delta _ { s } }$

Together, Eq. (17) and Prop. 2 show that lowering s from t tightens the bound behind the first term but moves the second term from zero toward the DPS error. Each end is exact only in its own ideal case, $s = t$ for an exact value model and $s = 0$ for a linear reward with $V _ { \psi } ( \cdot , 0 ) = r$ and the exact denoiser, so neither is optimal in general. In T2I generation, HPSv2 and PickScore peak at s = 0.8t to 0.6t and decrease toward both ends, and we set s = 0.6t (Fig. 4), a ratio that balances the two terms.

Guiding with $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone, which avoids the VJP, ignores how $z _ { s }$ moves with ${ \boldsymbol { z } } _ { t }$ and is inexact even for a linear reward, for which DPS is exact (Sec. B.5). In T2I generation, the correction brings most of the HPSv2 gain over StitchVM (Tab. 4).

## 4.3. VaLoP-Fast: Amortized Correction

VaLoP improves the value gradient without further training, but its correction needs a VJP through the generator at every step, which adds to the cost of inference. To address this, we propose VaLoP-Fast, which amortizes the guidance $\pmb { g } _ { t } ^ { \mathrm { V a L o P } }$ with a small head trained on states visited by the guided sampler, with the generator and the value model frozen. Alg. 1 summarizes both variants in both settings.

Image generation. Since T2I latents are high-dimensional and one head should serve every reward, the head $h _ { \phi } ( z _ { t } , t , g _ { s } )$ takes the exact $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ as input and regresses only the correction,

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } _ { z _ { t } } \big \| h _ { \phi } ( z _ { t } , t , \pmb { g } _ { s } ) - \mathrm { s g } [ \pmb { \xi } _ { t } ] \big \| ^ { 2 } ,\tag{19}
$$

and at inference replaces the VJP with one forward pass of the head, without backpropagation through the generator (Sec. D.3.1). Because the head learns only the correction, $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ remains exact and carries the reward signal. The target $\xi _ { t }$ is the generator's VJP applied to $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } } ,$ so the reward enters only through this input, while the map from $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ to $\xi _ { t }$ belongs to the generator. A head trained with one reward therefore serves the others, and we train one head per generator with the HPSv2 value and use it for all rewards (Sec. E.2).

Robot policies. Since the action chunk is low-dimensional one head per task, $h _ { \phi } ( z _ { t } , t )$ , regresses the whole guidance instead of its correction,

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } _ { z _ { t } } \big \| h _ { \phi } ( z _ { t } , t ) - \mathrm { s g } [ g _ { t } ^ { \mathrm { V a L o P } } ] \big \| ^ { 2 } ,\tag{20}
$$

so that inference needs neither the critic nor a policy backward pass. In practice, the head also predicts the matched norm of Sec. 4.4, which needs the critic (Sec. D.3.2).

## 4.4. Practical Implementation

Jacobian scale. With CFG in T2I generation, the VJP in Alg. 1 uses $J _ { \gamma } = J _ { u } + \gamma ( J _ { c } - J _ { u } )$ , where $J _ { u }$ and $J _ { c }$ are the

Algorithm 1 Guided sampling with VaLoP and VaLoP-  
Fast. The head for a policy skips lines 4 and 5.   
Require: ${ \boldsymbol { v } } _ { \boldsymbol { \theta } } ,$ value V (Vψ, or Q at the denoised action),   
ratio $\rho = s / t .$ , strength $\eta ,$ Jacobian scale $\gamma ,$ head $h _ { \phi }$   
(VaLoP-Fast)   
1: $z _ { 1 } \sim \mathcal { N } ( 0 , I )$   
2: for $t = 1 , 1 - \Delta t , \ldots , \Delta t$ do   
3: $v  v _ { \theta } ( z _ { t } , t ) ; \ s  \rho t$   
4: $z _ { s } \gets z _ { t } - ( t - s ) v$ one-step lookahead   
5: ${ \pmb g } _ { s } \gets \nabla _ { z _ { s } } V ( z _ { s } , s )$ ▶ value backward   
6: if VaLoP then   
7: ${ \pmb y } _ { t }  { \pmb g } _ { s } - ( t - s ) J _ { \gamma } ^ { \top } { \pmb g } _ { s }$ one VJP   
8: else if T2I then   
9: ${ \pmb y } _ { t } \gets { \pmb g } _ { s } + { \pmb h } _ { \phi } ( z _ { t } , t , { \pmb g } _ { s } )$ head: T2I   
10: else   
11: ${ \pmb y } _ { t } \gets { \pmb h } _ { \phi } ( { \pmb z } _ { t } , t )$ head: robot policy   
12: $z _ { t - \Delta t } \gets z _ { t } - \Delta t v + \eta y _ { t }$ Eq. (16)   
13: return $z _ { \mathrm { 0 } }$

Jacobians of the unconditional and conditional velocities, rather than the sampling scale $\omega = 4 . 5$ , which is tuned for image quality. We use $\gamma = 2$ for $t \geq 0 . 8$ and $\gamma = 1$ , which needs only the conditional backward pass, for $t < 0 . 8$ . This schedule performs as well as $\gamma = 2$ at every step (Sec. E.2). Magnitude matching. In T2I generation, both variants replace $\eta g _ { t } ^ { \mathrm { V a L o P } }$ in Eq. (16) by $\eta m _ { t } { \pmb y } _ { t } / \| { \pmb y } _ { t } \|$ , with $y _ { t } = g _ { s } +$ $\xi _ { t }$ and the norm $m _ { t } = \| \nabla V _ { \psi } ( z _ { t } , t ) \|$ of the StitchVM gradient, so each step moves as far as a StitchVM step. VaLoP-Fast fits $\hat { m } _ { t } = e ^ { \bar { a } ( t ) } \lVert \mathbf { g } _ { s } \rVert ^ { 1 + b ( t ) }$ per reward (Sec. D.3.1) and thus needs one value-model backward pass, as StitchVM does, and no generator backward pass (Alg. 2).

Robot policies. A VLA policy is guided by a critic Q that scores a clean action chunk given the observation, trained per task as in AGF [25]. We read it at the denoised lookahead action, $Q ( z _ { s } - s v _ { \theta } ( z _ { s } , s ) )$ , so that the first term of Eq. (17) pulls back the DPS error of Eq. (11) at level s rather than $t ,$ and match each update to the norm $m _ { t }$ of the QGF [75] gradient at $z _ { t } \ ( \mathrm { A l g } . 5 )$

## 5. Experiments

## 5.1. Setup

Image generation. We use SD3.5-Medium, SD3.5- Large [14, 60], and FLUX.1-dev [5] at $1 0 2 4 ^ { 2 }$ on the 200 DrawBench prompts [51] with three shared seeds (sampling settings in Sec. D.2). We compare against the unguided model, pixel-space DPS [6], and DPS + StitchVM [19] with the official StitchVM code. DPS + StitchVM and our variants use the official StitchVM value models for HPSv2 [69] CLIP [47], and the CLIP+MLP aesthetic predictor [52]. We report HPSv2, the aesthetic (Aes.) and CLIP scores, PickScore (Pick) [28], ImageReward (IR) [70], and the

Table 1. Reward guidance on DrawBench and GenEval $( \mathrm { { 1 0 2 4 } ^ { 2 } } ,$ strengths in Sec. D.2). Rows are grouped by the guided reward, whose column is marked by $\checkmark ,$ and the other columns are held-out metrics. DrawBench metrics average three seeds and GenE., the GenEval overall score, averages four (three for DPS, s.d. in Sec. E.1). Bold: best guided result per target and model.
<table><tr><td></td><td colspan="5">SD3.5-Medium</td><td colspan="5">SD3.5-Large</td><td colspan="5">FLUX.1-dev</td></tr><tr><td>Method</td><td colspan="5">HPSv2↑ Aes.↑</td><td colspan="5">GenE.↑ HPSv2↑ Aes.↑</td><td colspan="5">GenE.↑ HPSv2↑ Aes.↑</td></tr><tr><td>Unguided</td><td>0.286</td><td>5.347</td><td>28.80 22.65</td><td>0.987</td><td>0.694</td><td>0.294</td><td>5.461</td><td>28.61</td><td>22.87 1.065</td><td>0.721</td><td>0.298</td><td>5.729</td><td>27.71</td><td>22.79 0.995</td><td>0.660</td></tr><tr><td>HPSv2 reward</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DPS</td><td> $\checkmark$  0.361</td><td>5.426</td><td>28.99</td><td>23.541.130</td><td>0.711</td><td> $\checkmark$  0.349</td><td>5.519</td><td>28.82</td><td>23.55 1.111</td><td>0.715</td><td>0.320</td><td>5.440</td><td>27.09</td><td>22.78 0.922</td><td>0.633</td></tr><tr><td>DPS + StitchVM</td><td>0.366</td><td>5.541</td><td>28.77</td><td>23.281.302</td><td>0.647</td><td>0.360</td><td>5.544</td><td>28.65</td><td>23.201.248</td><td>0.662</td><td>0.314</td><td>5.690</td><td>27.82</td><td>22.79 1.080</td><td>0.652</td></tr><tr><td>VaLoP (Ours)</td><td>0.382</td><td>5.654</td><td>29.13</td><td>23.77 1.332</td><td>0.726</td><td>0.371</td><td>5.633</td><td>29.24 23.78</td><td>1.290</td><td>0.755</td><td>0.328</td><td>5.590</td><td>28.36</td><td>23.12 1.190</td><td>0.701</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.387</td><td>5.671</td><td>28.92</td><td>23.72 1.333</td><td>0.702</td><td>0.368</td><td>5.657</td><td>29.40</td><td>23.66 1.323</td><td>0.722</td><td>0.333</td><td>5.782</td><td>28.28</td><td>23.11 1.198</td><td>0.686</td></tr><tr><td>Aesthetic reward</td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DPS</td><td>0.285</td><td>5.877</td><td>28.30</td><td>22.63 0.944</td><td>0.662</td><td>0.295</td><td>5.835</td><td>28.51</td><td>22.87 1.067</td><td>0.708</td><td>0.297</td><td> $\checkmark$ </td><td></td><td></td><td>0.641</td></tr><tr><td>DPS + StitchVM</td><td>0.279</td><td>5.968</td><td>26.95</td><td>22.29 0.801</td><td>0.644</td><td>0.287</td><td>5.807</td><td>28.05</td><td>22.64 0.992</td><td>0.691</td><td>0.297</td><td>5.815 5.893</td><td>27.61 27.37</td><td>22.78 0.991 22.74 0.967</td><td>0.651</td></tr><tr><td>VaLoP (Ours)</td><td>0.288</td><td>6.174</td><td>27.22</td><td>22.56 0.905</td><td>0.672</td><td>0.297</td><td>5.989</td><td>28.15</td><td>22.88 1.058</td><td>0.712</td><td>0.300</td><td>5.996</td><td>27.29</td><td>22.84 0.968</td><td>0.666</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.293</td><td>6.035</td><td>27.62</td><td>22.69 0.974</td><td>0.672</td><td>0.296</td><td>6.144</td><td>28.06</td><td>22.80 1.066</td><td>0.706</td><td>0.301</td><td>5.929</td><td>27.46</td><td>22.85 1.007</td><td>0.659</td></tr><tr><td>CLIP reward</td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DPS</td><td>0.254</td><td>4.95534.27</td><td></td><td>21.70 0.622</td><td>0.633</td><td>0.286</td><td>5.321</td><td>32.13</td><td>22.60 0.982</td><td>0.713</td><td>0.279</td><td>5.410</td><td> $\checkmark$  29.28</td><td>22.25 0.819</td><td>0.637</td></tr><tr><td>DPS + StitchVM</td><td>0.248</td><td>4.877</td><td>34.42</td><td>21.66 0.688</td><td>0.625</td><td>0.275</td><td>5.183</td><td>32.43</td><td>22.40 0.983</td><td>0.724</td><td>0.286</td><td>5.499</td><td>29.55</td><td>22.46 0.936</td><td>0.648</td></tr><tr><td>VaLoP (Ours)</td><td>0.266</td><td>4.999</td><td>35.98</td><td>22.08 0.942</td><td>0.725</td><td>0.284</td><td>5.281</td><td>33.69</td><td>22.60 1.087</td><td>0.759</td><td>0.291</td><td>5.514</td><td>30.79</td><td>22.65 1.097</td><td>0.691</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.275</td><td>5.085</td><td>35.03</td><td>22.26 1.022</td><td>0.722</td><td>0.282</td><td>5.222</td><td>33.65</td><td>22.44 1.095</td><td>0.735</td><td>0.295</td><td>5.545</td><td>30.37</td><td>22.67 1.082</td><td>0.695</td></tr></table>

GenEval [18] overall score, and call the guided reward the target reward and the others held-out metrics.

Robot policies. Following AGF [25], we guide frozen VLA policies with a task-specific critic, SmolVLA [55] on the Goal, Spatial, and LIBERO-10 suites of LIBERO [33] and $\pi _ { 0 . 5 }$ [4] on the 18 atomic tasks of RoboCasa365 [42]. We compare against critic-based best-of-N selection (Q-BoN) [41], QDPS [75], which differentiates the critic at the denoised action through the policy, QGF [75], which replaces the policy Jacobian with the identity, and AGF. VaLoP evaluates the critic at the denoised lookahead action, and VaLoP-Fast amortizes this guidance with a head per task. All methods share the critic, protocol, and seeds (Sec. D.4), with success rates over 50 episodes per task.

## 5.2. Main Results

Image generation. To verify that the lookahead improves guidance across generators and rewards, we guide each model toward one reward at a time and measure the target reward together with the held-out metrics (Tab. 1). Across all three rewards and models, VaLoP raises the target reward above both StitchVM and pixel-space DPS by over one s.d. across seeds (statistical analysis in Sec. E.1), and VaLoP-Fast performs comparably, exceeding VaLoP in three of the nine settings. To test robustness to the prompt set, we repeat this comparison on Pick-a-Pic prompts [28], where both variants again exceed both baselines on the target reward in every setting (Sec. E.3). StitchVM raises the target reward at the expense of the others, falling below the unguided model in all but one held-out entry under the aesthetic and CLIP rewards. VaLoP instead exceeds StitchVM on every held-out metric except two entries on FLUX.1-dev, VaLoP-Fast does so in all nine settings, and VaLoP exceeds both baselines on GenEval in all nine, which also shows in the qualitative results (Sec. E.5). Our variants thus raise the target reward more robustly than StitchVM.

Table 2. Critic guidance on robot policies. Success rate (%) of SmolVLA on LIBERO and $\pi _ { 0 . 5 }$ on RoboCasa, averaged over the tasks of each suite. Bold: best. Rollouts in Sec. E.5.
<table><tr><td rowspan="2">Success (%)↑</td><td colspan="3">LIBERO</td><td rowspan="2">RoboCasa Atomic</td></tr><tr><td></td><td>Goal Spatial LIBERO-10</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Method Base</td><td>76.4</td><td>64.4</td><td>43.4 61.4</td><td>44.6</td></tr><tr><td>Q-BoN</td><td>78.2</td><td>69.6</td><td>41.2 63.0</td><td>46.7</td></tr><tr><td>QGF</td><td>79.0</td><td>69.4</td><td>48.6 65.7</td><td>47.1</td></tr><tr><td>QDPS</td><td>80.2</td><td>68.2</td><td>47.6 65.3</td><td>45.7</td></tr><tr><td>AGF</td><td>80.0</td><td>69.2</td><td>48.0 65.7</td><td>45.9</td></tr><tr><td></td><td></td><td></td><td>66.3</td><td></td></tr><tr><td>VaLoP (Ours) VaLoP-Fast (Ours)</td><td>80.6 81.8</td><td>69.4 69.8</td><td>49.0 49.4 67.0</td><td>47.4 47.4</td></tr></table>

Robot policies. To test whether the lookahead also improves critic guidance of robot policies, we guide SmolVLA and $\pi _ { 0 . 5 }$ (Tab. 2). On LIBERO, every gradient-based baseline raises the success rate of the base policy, VaLoP exceeds all baselines on average, and VaLoP-Fast is highest on every suite, both significantly above the base policy (statistical analysis, Sec. E.1). On RoboCasa, both variants raise the success rate of the base policy more than any baseline. In the qualitative results, VaLoP-Fast succeeds in every episode and is the first to succeed in most (Fig. 1, Sec. E.5). The lookahead thus carries over to action generation.

Table 3. Cost on one H100: time (s) and peak memory (GB) per image at $1 0 2 4 ^ { 2 }$ , and time (s) per robot action chunk. Bold: lowest.
<table><tr><td></td><td colspan="2">SD3.5-Medium</td><td colspan="2">SD3.5-Large</td><td colspan="2">FLUX.1-dev</td></tr><tr><td>Method</td><td>Time↓</td><td>Mem.↓</td><td></td><td>Time↓ Mem.↓</td><td>Time↓</td><td>Mem.↓</td></tr><tr><td>DPS</td><td>54.5</td><td>43.9</td><td>83.5</td><td>66.6</td><td>43.6</td><td>41.7</td></tr><tr><td>DPS + StitchVM</td><td>17.8</td><td>18.9</td><td>36.8</td><td>34.9</td><td>9.9</td><td>32.2</td></tr><tr><td>VaLoP (Ours)</td><td>30.4</td><td>24.6</td><td>61.6</td><td>52.2</td><td>28.6</td><td>33.8</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>15.4</td><td>18.6</td><td>31.1</td><td>34.7</td><td>8.6</td><td>31.0</td></tr></table>

<table><tr><td rowspan="2">Policy</td><td colspan="6">Time per robot action chunk (s)↓</td></tr><tr><td>Q-BoN</td><td>QGF</td><td>QDPS</td><td>AGF</td><td></td><td>VaLoP VaLoP-Fast</td></tr><tr><td>SmolVLA</td><td>0.463</td><td>0.587</td><td>1.045</td><td>0.460</td><td>2.048</td><td>0.460</td></tr><tr><td>π0.5</td><td>0.250</td><td>0.303</td><td>0.633</td><td>0.240</td><td>1.139</td><td>0.240</td></tr></table>

Table 4. Lookahead and correction (SD3.5-Medium, $1 0 2 4 ^ { 2 } .$ DrawBench, HPSv2 reward, $\eta = 3 0 0 ,$ , one seed). Bold: best.
<table><tr><td></td><td>HPSv2↑</td><td> $\mathrm { A e s . \uparrow }$ </td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td></tr><tr><td>VaLoP (Ours)</td><td>0.382</td><td>5.665</td><td>29.06</td><td>23.79</td><td>1.339</td></tr><tr><td colspan="6">Lookahead and correction</td></tr><tr><td>No lookahead  $( s = t )$ </td><td>0.358</td><td>5.505</td><td>28.63</td><td>23.28</td><td>1.322</td></tr><tr><td>No correction (gs only)</td><td>0.365</td><td>5.614</td><td>28.23</td><td>23.52</td><td>1.324</td></tr><tr><td colspan="6">Form of the correction</td></tr><tr><td>J instead of JT</td><td>0.368</td><td>5.598</td><td>28.25</td><td>23.59</td><td>1.318</td></tr></table>

Table 5. Lookahead and correction on SmolVLA (success %), all with the critic at the denoised action. Bold: best.
<table><tr><td></td><td>Goal</td><td>Spatial</td><td>LIBERO-10</td></tr><tr><td>No lookahead  $( s = t )$ </td><td>79.0</td><td>69.0</td><td>47.4</td></tr><tr><td>No correction  $( g _ { s } \ \mathrm { o n l y } )$ </td><td>76.6</td><td>67.0</td><td>46.4</td></tr><tr><td>VaLoP (Ours)</td><td>80.6</td><td>69.4</td><td>49.0</td></tr></table>

## 5.3. Inference Cost

Image generation. VaLoP adds one generator VJP and a second value-model backward pass per step, yet takes less time and memory than pixel-space DPS (Tab. 3). VaLoP-Fast replaces both with one head pass and the fitted magnitude, and it needs less time and memory than StitchVM on every model, with both run in our optimized implementation (Sec. D.2).

Robot policies. By amortizing the guidance, VaLoP-Fast adds only 3 to 5% to the inference time of the base policy on SmolVLA and $\pi _ { 0 . 5 } ,$ as AGF does, runs faster than QGF and QDPS since it needs no critic, and trains its head on SmolVLA in under half the time of AGF (Sec. D.4).

## 5.4. Ablation Study

Lookahead ratio. To test the trade-off of Sec. 4.2, we sweep $s / t$ from 1, learned-value guidance, to 0, the DPS form. HPSv2 and PickScore peak at $s = 0 . 8 t$ to 0.6t and decrease toward both ends (Fig. 4), consistent with the dominance of the value-model error near $s = t$ and of the step bias near $s = 0$ All ratios but 0.8t are significantly worse than 0.6t in both metrics (paired t-test, Sec. E.1), and 0.8t is slightly higher only in HPSv2. We select 0.6t on heldout prompts (Sec. D.2). On LIBERO-Goal, $s = 0 . 6 t$ is also best on average over strengths for SmolVLA (Sec. E.4).

![](images/da3dbc729469b361eb8cdcd6bb627c771342af4d52552bf02003cf0d722fc19b.jpg)

![](images/1b4d7c35e483778a5b0cac363d401f7d18de32b604cba59ef6e9d22721e31a58.jpg)  
Figure 4. Lookahead ratio (SD3.5-Medium, HPSv2 reward, DrawBench at $1 0 2 4 ^ { 2 } )$ , relative to learned-value guidance at $s = t .$

Lookahead and correction. Guided by $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone, the lookahead raises HPSv2 and PickScore above $s \ = \ t ,$ but the correction accounts for the larger part of the gain in both (Tab. 4), so the pullback carries the gain. VaLoP-Fast keeps most of it (Sec. E.2).

With the critic on SmolVL $\mathbf { A } , g _ { s }$ alone falls below $s = t$ on every suite, and only the correction lifts VaLoP above both (Tab. 5). The lookahead also improves a learned value: StitchVM, not in Tab. 2 as it needs an extra value model on noisy actions for each task, rises on LIBERO-Goal from 77.4 to 79.6% when this model, stitched to SmolVLA, is read with VaLoP (Sec. E.4).

Form of the correction. We replace $J _ { \gamma } ^ { \top }$ by $J _ { \gamma }$ , which forward differences compute without a backward pass. For the exact posterior mean, $D _ { t }$ is symmetric by Eq. (10), so both give the same correction, but with CFG the velocity Jacobian is strongly asymmetric on SD3.5 (Sec. E.2), and the replacement lowers every metric, HPSv2 nearly to the level of $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone (Tab. 4). The pullback thus needs the transpose.

## 6. Conclusion

We proposed VaLoP, which evaluates a value model at a less noisy lookahead state and pulls its gradient back, placing learned-value guidance and DPS at the two ends of one family and reducing the value-gradient error at high noise. Across three generators and three rewards, VaLoP and VaLoP-Fast raise the target reward while better preserving the held-out metrics, VaLoP-Fast at a lower inference cost than learned-value guidance. In critic guidance of robot policies, VaLoP-Fast attains the highest success rates on LIBERO and RoboCasa. As the correction depends on the generator, not the reward, one head per image generator serves every reward, which we believe opens a path toward guidance modules trained once and reused for any reward.

Acknowledgments. We sincerely thank the authors of StitchVM [19] and AGF [25] for sharing their code.

LLM Usage Statement. We used a commercial large language model (Claude) to help draft and edit the text of this paper, for wording, grammar, and consistency, and to write the plotting scripts for the figures. The authors directed this use, checked every statement against the results and the cited sources, and take full responsibility for the content.

## References

[1] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal Guidance for Diffusion Models. In International Conference on Learning Representations (ICLR), 2024. 2, 13

[2] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-Flow: Differentiating through Flows for Controlled Generation. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 3462–3483. PMLR, 2024. 13, 14

[3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training Diffusion Models with Reinforcement Learning. In International Conference on Learning Representations (ICLR), 2024. 14

[4] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a Vision-Language-Action Model with Open-World Generalization. In Proceedings of The 9th Conference on Robot Learning (CoRL), pages 17–40. PMLR, 2025. 3, 7

[5] Black Forest Labs. FLUX.1 [dev]. https : //github. com/black-forest-labs/flux,2024.6

[6] Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye. Diffusion Posterior Sampling for General Noisy Inverse Problems. In International Conference on Learning Representations (ICLR), 2023. 2, 3, 6,13

[7] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly Fine-Tuning Diffusion Models on Differentiable Rewards. In International Conference on Learning Representations (ICLR), 2024. 2, 14

[8] Fengyuan Dai, Zifeng Zhuang, Yufei Huang, Siteng Huang, Bangyan Liao, Donglin Wang, and Fajie Yuan. VARD: Efficient and Dense Fine-Tuning for Diffusion Models with Value-based RL. arXiv preprint arXiv:2505.15791, 2025. 14

[9] Sanjit Dandapanthula and Nicholas M. Boffi. Are We Really Tilting? The Mechanics of Reward Guidance in Flow

and Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), 2026. 13

[10] Alexander Denker, Francisco Vargas, Shreyas Padhy, Kieran Didi, Simon Mathis, Vincent Dutordoir, Riccardo Barbano, Emile Mathieu, Urszula Julia Komorowska, and Pietro Liò. DEFT: Efficient Fine-tuning of Diffusion Models by Learning the Generalised h-transform. In Advances in Neural Information Processing Systems (NeurIPS), pages 19636– 19682, 2024. 14

[11] Prafulla Dhariwal and Alexander Nichol. Diffusion Models Beat GANs on Image Synthesis. In Advances in Neural Information Processing Systems (NeurIPS), pages 8780–8794, 2021.4,14

[12] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control. In International Conference on Learning Representations (ICLR), 2025. 2, 3, 14

[13] Bradley Efron. Tweedie's Formula and Selection Bias. Journal of the American Statistical Association, 106(496):1602– 1614,2011.3

[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 12606–12633. PMLR, 2024.2,6

[15] Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, and Zeynep Akata. ReNO: Enhancing Onestep Text-to-Image Models through Reward-based Noise Optimization. In Advances in Neural Information Processing Systems (NeurIPS), pages 125487–125519, 2024. 14

[16] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), pages 79858–79885, 2023. 14

[17] Zhengqi Gao, Kaiwen Zha, Tianyuan Zhang, Zihui Xue, and Duane S. Boning. REG: Rectified Gradient Guidance for Conditional Diffusion Models. In Proceedings of the 42nd International Conference on Machine Learning (ICML), pages 18724–18743. PMLR, 2025. 14

[18] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. In Advances in Neural Information Processing Systems (NeurIPS), pages 52132–52152, 2023.7

[19] Hyojun Go, Hyungjin Chung, Prune Truong, Goutam Bhat, Li Mi, Zhaochong An, Zixiang Zhao, Dominik Narnhofer, Serge Belongie, Federico Tombari, and Konrad Schindler Stitched Value Model for Diffusion Alignment. In Advances in Neural Information Processing Systems (NeurIPS), 2026. 1, 2, 3, 6, 9, 13, 14, 18

[20] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A Kernel Two-

Sample Test. Journal of Machine Learning Research, 13 (25):723–773, 2012. 19

[21] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. arXiv preprint arXiv:2207.12598, 2022. 3

[22] Peter Holderrieth, Douglas Chen, Luca Eyring, Ishin Shah, Giri Anantharaman, Yutong He, Zeynep Akata, Tommi Jaakkola, Nicholas Matthew Boffi, and Max Simchowitz. Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps. In Proceedings of the 43rd International Conference on Machine Learning (ICML), pages 43401–43435. PMLR, 2026. 2, 13, 14, 19

[23] Jerry Y. Huang, Justin Lin, Sheel Shah, Kartik Nair, and Nicholas M. Boffi. How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance. In Proceedings of the 43rd International Conference on Machine Learning (ICML), pages 47101–47152. PMLR, 2026. 2, 13, 14, 19

[24] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with Diffusion for Flexible Behavior Synthesis. In Proceedings of the 39th International Conference on Machine Learning (ICML), pages 9902–9915. PMLR, 2022.14

[25] Jeongsol Kim, Youngjun Jun, Kyumin Choi, Youngmin Kim, Seonghyun Jin, Sunwoo Park, Jangho Park, Kwanyoung Kim, and Jong Chul Ye. Adjoint Guidance Flow: Amortized Critic Guidance for VLA Policies. arXiv preprint arXiv:2609.34944, 2026. 1, 2, 3, 4, 6, 7, 9, 13, 14, 23

[26] Kwanyoung Kim and Jong Chul Ye. Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images. In Advances in Neural Information Processing Systems (NeurIPS), pages 864–874, 2021. 3

[27] Yeongmin Kim, Donghyeok Shin, Byeonghu Na, Minsang Park, Richard Lee Kim, and Il-Chul Moon. Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), pages 58969–58995. PMLR, 2026. 2, 14

[28] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. In Advances in Neural Information Processing Systems (NeurIPS), pages 36652–36663, 2023. 6, 7, 27

[29] Qiyang Li and Sergey Levine. Q-Learning with Adjoint Matching. In International Conference on Learning Representations (ICLR), 2026. 14, 23

[30] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gökcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, and Masatoshi Uehara. Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-based Decoding. In Advances in Neural Information Processing Systems (NeurIPS), pages 95507–95545, 2025. 2, 3, 4, 14

[31] Zhengyang Liang, Qihang Zhang, and Ceyuan Yang. Explicit Critic Guidance for Aligning Diffusion Models. arXiv preprint arXiv:2605.27736, 2026. 14

[32] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling. In International Conference on Learning Representations (ICLR), 2023. 2, 3

[33] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In Advances in Neural Information Processing Systems (NeurIPS), pages 44776–44791, 2023. 7

[34] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training Flow Matching Models via Online RL. In Advances in Neural Information Processing Systems (NeurIPS), pages 40783–40818, 2025. 2, 14

[35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations (ICLR), 2023. 2, 3

[36] Zhen Liu, Tim Z. Xiao, Carles Domingo-Enrich, Weiyang Liu, and Dinghuai Zhang. Value Gradient Guidance for Flow Matching Alignment. In Advances in Neural Information Processing Systems (NeurIPS), pages 11372–11404, 2025. 14

[37] Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning. In Proceedings of the 40th International Conference on Machine Learning (ICML), pages 22825– 22855. PMLR, 2023. 14

[38] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. Scaling Inference Time Compute for Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2523–2534, 2025. 2

[39] Chenlin Meng, Yang Song, Wenzhe Li, and Stefano Ermon. Estimating High Order Gradients of the Data Distribution by Denoising. In Advances in Neural Information Processing Systems (NeurIPS), pages 25359–25369, 2021. 4, 15

[40] Naila Murray, Luca Marchesotti, and Florent Perronnin. AVA: A Large-Scale Database for Aesthetic Visual Analysis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2408–2415, 2012.3

[41] Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. In Proceedings of The 8th Conference on Robot Learning (CoRL), pages 4996– 5013. PMLR, 2024. 7, 14, 23

[42] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots. In International Conference on Learning Representations (ICLR), 2026. 7

[43] Jiachun Pan, Hanshu Yan, Jun Hao Liew, Jiashi Feng, and Vincent Y. F. Tan. Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method. arXiv preprint arXiv:2312.12030, 2023. 14

[44] Maitreya Patel, Song Wen, Dimitris N. Metaxas, and Yezhou Yang. FlowChef: Steering of Rectified Flow Models for Controlled Generations. In Proceedings of the IEEE/CVF

International Conference on Computer Vision (ICCV), pages 15308–15318, 2025. 13, 14

[45] Peter Potaptchik, Adhi Saravanan, Abbas Mammadov, Alvaro Prat, Michael S. Albergo, and Yee Whye Teh. Meta Flow Maps Enable Scalable Reward Alignment. In Proceedings of the 43rd International Conference on Machine Learning (ICML), pages 99255–99295. PMLR, 2026. 2, 14

[46] Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. Learning a Diffusion Model Policy from Rewards via Q-Score Matching. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 41163–41182. PMLR, 2024. 14

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 8748–8763. PMLR, 2021. 2, 3, 6

[48] Allen Z. Ren, Justin Lidard, Lars L. Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In International Conference on Learning Representations (ICLR), 2025. 14

[49] Yifan Ruan, Chenyang Cao, Andreas Burger, Ali Pesaranghader, Kaveh Kamali, Jaehong Kim, Nandita Vijaykumar, Alán Aspuru-Guzik, Igor Gilitschenski, and Nicholas Rhinehart. QPILOTS: Efficient Test-Time Q-Steering for Flow Policies. arXiv preprint arXiv:2606.14801, 2026. 14

[50] Amirmojtaba Sabour, Michael S. Albergo, Carles Domingo-Enrich, Nicholas M. Boffi, Sanja Fidler, Karsten Kreis, and Eric Vanden-Eijnden. Test-time Scaling of Diffusions with Flow Maps. In Advances in Neural Information Processing Systems (NeurIPS), 2026. 2, 14

[51] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In Advances in Neural Information Processing Systems (NeurIPS), pages 36479–36494, 2022. 6

[52] Christoph Schuhmann. CLIP+MLP Aesthetic Score Predictor. https : / / github . com / christophschuhmann /improved- aesthetic - predictor, 2022. 2, 3,6

[53] Max W. Shen, Ehsan Hajiramezanali, Gabriele Scalia, Alex Tseng, Nathaniel Diamant, Tommaso Biancalani, and Andreas Loukas. Conditional Diffusion with Less Explicit Guidance via Model Predictive Control. arXiv preprint arXiv:2210.12192, 2022. 14

[54] Jeongwoo Shin, Dongsoo Shin, Yuchen Zhu, Wei Guo, Yongxin Chen, Joonseok Lee, Jaewoong Choi, and Jaemoo Choi. Efficient Adjoint Matching for Fine-tuning Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), 2026. 14

[55] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Ar-

actingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv preprint arXiv:2506.01844, 2025. 3, 7

[56] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A General Framework for Inference-time Scaling and Steering of Diffusion Models. In Proceedings of the 42nd International Conference on Machine Learning (ICML), pages 55810–55827. PMLR, 2025. 2, 14

[57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In International Conference on Learning Representations (ICLR), 2021. 3

[58] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation. In Proceedings of the 40th International Conference on Machine Learning (ICML), pages 32483–32498. PMLR, 2023. 13

[59] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR), 2021. 2

[60] Stability AI. Stable Diffusion 3.5. https : //github. com/Stability-AI/sd3.5,2024.2,6

[61] Zhiwei Tang, Jiangweizhi Peng, Jiasheng Tang, Mingyi Hong, Fan Wang, and Tsung-Hui Chang. Inference-Time Alignment of Diffusion Models with Direct Noise Optimization. In Proceedings of the 42nd International Conference on Machine Learning (ICML), pages 58905–58930. PMLR, 2025.14

[62] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M. Tseng, Tommaso Biancalani, and Sergey Levine. Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control. arXiv preprint arXiv:2402.15194, 2024. 2, 3

[63] Andrew Wagenmaker, Yunchu Zhang, Mitsuhiko Nakamoto, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering Your Diffusion Policy with Latent Space Reinforcement Learning. In Proceedings of The 9th Conference on Robot Learning (CoRL), pages 258–282. PMLR, 2025. 14

[64] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-End Diffusion Latent Optimization Improves Classifier Guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7280–7290, 2023. 13, 14

[65] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion Model Alignment Using Direct Preference Optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238, 2024. 14

[66] Luran Wang, Chaoran Cheng, Yizhen Liao, Yanru Qu, and Ge Liu. Training Free Guided Flow-Matching with Optimal Control. In International Conference on Learning Representations (ICLR), 2025. 14

[67] Zhendong Wang, Jonathan J. Hunt, and Mingyuan Zhou. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. In International Conference on Learning Representations (ICLR), 2023. 14

[68] Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, and John P. Cunningham. Practical and Asymptotically Exact Conditional Sampling in Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), pages 31372–31403, 2023. 14

[69] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341, 2023. 2, 3, 6

[70] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. In Advances in Neural Information Processing Systems (NeurIPS), pages 15903–15935, 2023. 2, 6, 14

[71] Liuhaichen Yang, Zhuang Jiang, Chenchao Sheng, Ningwei Bai, Qichen Yin, Hanbo Ma, Junkai Liu, Junkai Sun, Dongcheng Lyu, Yi Dong, and Zezhi Tang. Guided Action Flow: Value-Guided Sampling for Frozen Vision-Language-Action Policies. arXiv preprint arXiv:2607.02092, 2026. 14

[72] Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Zou, and Stefano Ermon. TFG: Unified Training-Free Guidance for Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), pages 22370–22417, 2024. 2, 4, 13

[73] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 23174–23184, 2023. 2, 13

[74] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online Diffusion Reinforcement with Forward Process. In International Conference on Learning Representations (ICLR), 2026. 2, 14

[75] Zhiyuan Zhou, Andy Peng, Charles Xu, Qiyang Li, Jost Tobias Springenberg, Kevin Frans, and Sergey Levine. Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS), 2026. 6, 7, 13, 14, 22, 23

[76] Christos Ziakas, Alessandra Russo, and Avishek Joey Bose. Aligning Flow Map Policies with Optimal Q-Guidance. arXiv preprint arXiv:2605.12416, 2026. 14

## VaLoP: Value Gradient Lookahead and Pullback for Reward-Guided Diffusion Supplementary Material

## Contents

A. Related Work . 13   
B. Proofs 14   
B.1. Proof of Lemma 1. 15   
B.2. Errors of DPS, a Learned Value, and AGF 15   
B.3. Proof of Proposition 1. 17   
B.4. Proof of Proposition 2. 17   
B.5. Error Split and the Correction . 18   
C. Scope and Limitations . 18   
D. Experimental Details. . 19   
D.1. Two-Dimensional Mixtures 19   
D.2 Text-to-Image Generation . 19   
D.3. Correction Head of VaLoP-Fast 20   
D.4. Robot Policies 23   
E. Additional Results . 25   
E.1. Statistical Analysis 25   
E.2. Text-to-Image Ablations. 26   
E.3. Pick-a-Pic. 27   
E.4. Robot Ablations. 28   
E.5. Qualitative Results 28

## A. Related Work

Tab. 6 summarizes the state at which each estimator evaluates the reward or a value, and its cost per step and in training.

Table 6. Evaluation state and cost of each estimator. G: generator, D: decoder, R: reward, V: value model, H: head, F: flow map.

<table><tr><td>Method</td><td>Value at</td><td>Backprop per step</td><td>Training</td><td>One for all rewards</td></tr><tr><td colspan="5">Training-free</td></tr><tr><td>DPS, FreeDoM [6, 73]</td><td> $\hat { z } _ { 0 \mid t }$ </td><td>G, D, R</td><td></td><td></td></tr><tr><td>QGF, FlowChef [44, 75]</td><td> $\hat { z } _ { 0 \mid t }$ </td><td> $\mathrm { D } , \mathrm { R } \left( D _ { t } { = } I \right)$ </td><td></td><td></td></tr><tr><td>DOODL, D-Flow [2, 64]</td><td>z0</td><td>G (all steps), D, R</td><td></td><td></td></tr><tr><td colspan="5">Learned</td></tr><tr><td>StitchVM [19]</td><td>Zt</td><td>V</td><td>V</td><td>x</td></tr><tr><td>AGF [25]</td><td>z0</td><td></td><td>H, backprop through all steps</td><td>x</td></tr><tr><td>Flow map [22, 23]</td><td>z0</td><td>F, D, R</td><td>F, distillation</td><td>√</td></tr><tr><td>VaLoP (Ours)</td><td> $z _ { s }$ </td><td>V, G (VJP)</td><td></td><td>–</td></tr><tr><td>VaLoP-Fast (Ours)</td><td> $z _ { s }$ </td><td>V</td><td>H, backprop through one step</td><td>L</td></tr></table>

Training-free guidance. DPS [6] differentiates the reward of the posterior mean, and FreeDoM [73] and universal guidance [1] build on the same one-step estimate with additional recurrence or optimization, and LGD [58] reduces its bias with Monte Carlo samples around it. TFG [72] unifies such training-free methods in one hyperparameter space and searches it per task, and its analysis reads the gradient through the posterior mean as preconditioned by the posterior covariance, as in Eq. (10). Dandapanthula and Boffi [9] trace the over-optimization of plug-in reward guidance to a finite-particle approximation of the Doob h-function, already for Gaussian and Gaussian mixture targets. Our unification differs in kind: a one-parameter family indexed by the lookahead ratio, whose two ends are learned-value guidance and DPS, with an error analysis that relates the ratio to the posterior mismatch behind the value-model error and to the bias of the step (Sec. 4.2).

Lookahead guidance. Several methods evaluate the guide after simulating sampler steps. At one extreme, DOODL [64] and D-Flow [2] optimize the initial noise by differentiating the reward of the final sample through every step, OC-Flow [66] casts this as optimal control, and ReNO [15] and DNO [61] optimize the noise against preference rewards. SAG [43] estimates the clean sample with a multi-step solver and backpropagates through it with a symplectic adjoint. MPC guidance [53] simulates five to ten unconditional steps to a less noisy state, evaluates a noised classifier or a conditional diffusion model there, and backpropagates through all simulated steps, at a memory cost linear in their number. LiDAR [27] draws lookahead samples and derives guidance without backpropagation. VaLoP shares with MPC guidance the evaluation at a less noisy state, but it uses a learned reward value instead of a classifier or a conditional diffusion model, takes one deterministic step so that the pullback reduces to one correction term, shows that this term needs the transposed Jacobian under CFG, and distills this term into a head. REG [17] multiplies the gradient of existing guidance element-wise by a correction derived from the denoiser Jacobian, which amounts to a diagonal, hence symmetric, approximation of that Jacobian. On SD3.5, the symmetric part alone loses part of the HPSv2 of the transpose (Tab. 14). QGF [75] and FlowChef [44] replace this Jacobian with the identity, and QGF reports that including it hurts in offline reinforcement learning. The pullback $\begin{array} { r } { P _ { s } = \frac { s } { t } I + \bigl ( 1 - \frac { s } { t } \bigr ) D _ { t } } \end{array}$ of VaLoP instead shrinks it toward the identity, and dropping the correction lowers HPSv2 on SD3.5 (Tab. 4).

Flow-map lookahead. A flow map jumps from ${ \boldsymbol { z } } _ { t }$ toward the clean endpoint in one or a few evaluations, and recent methods evaluate the reward there. FMTT [50] tilts trajectories with this lookahead and importance weights, and FMRG [23] derives guidance from deterministic optimal control and pulls the reward gradient at the endpoint back through the flow map, the gradient that AGF regresses (Prop. B.3). Diamond Maps [22] and meta flow maps [45] instead average the exponentiated rewards of posterior samples drawn by a stochastic flow map, which estimates the soft value consistently but requires learning such a map. VaLoP needs no flow map. It takes one step of the frozen generator to an intermediate state and evaluates a learned value there, so that the lookahead ratio trades the value-model error against the bias of the step.

Learned values and amortized guidance. Classifier guidance [11] differentiates a classifier trained on noisy images, and StitchVM [19] stitches the denoiser and the reward model into a value model on noisy latents. In offline reinforcement learning, CEP [37] trains the intermediate energy with a contrastive objective that recovers the exact guidance given unlimited capacity and data. SVDD [30] decodes with soft values without derivatives, FK steering [56] and twisted SMC [68] resample interacting particles with reward potentials on intermediate states, and actor-critic fine-tuning learns a critic on noisy states that can also steer sampling [31]. AGF [25] amortizes guidance by regressing adjoint gradients along sampling paths onto a head. DEFT [10] keeps the unconditional model frozen and fine-tunes a small network to learn the conditional h-transform. VaLoP-Fast likewise trains only a small head with the generator frozen, but its target is the known one-step correction, which depends on the reward only through $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } } .$ , so one head per generator serves all rewards (Tab. 17).

Reward fine-tuning. The generator can instead be fine-tuned on the reward, by backpropagation through sampling [7, 70], reinforcement learning [3, 8, 16, 34, 74], direct preference optimization on human comparisons [65], stochastic optimal control [12, 54], or value-gradient matching [36]. VaLoP leaves the generator frozen and acts only at inference.

Critic guidance of robot policies. Diffuser [24] guides trajectory sampling with a return model trained on noisy trajectories, and V-GPS [41] re-ranks the actions of a frozen generalist policy with a learned value, which Q-BoN follows. QGF [75] differentiates a critic at the denoised action with the Jacobian of the policy replaced by the identity, QPILOTS [49] also evaluates the critic at a projected clean action, and Guided Action Flow [71] steers a frozen VLA policy with a small critic on a real robot. AGF [25] amortizes the costate into a guidance network trained per task. Training-based alternatives backpropagate the critic through the denoising chain [67], match the policy score to the critic gradient [46], fine-tune the policy [29, 48, 76]. or learn a policy over the latent noise of a frozen policy [63]. Both variants of VaLoP keep the policy frozen

## B. Proofs

Throughout, $k _ { t } ( z \mid z _ { 0 } ) = \mathcal { N } \big ( z ; ( 1 - t ) z _ { 0 } , t ^ { 2 } I \big )$ is the forward kernel, and moments are taken under the posterior $p ( \cdot \mid z _ { t } = z )$ unless stated otherwise. Restated results keep their numbers from the main text, and results stated only here are numbered by appendix section.

## B.1. Proof of Lemma 1

Lemma 1 (restated). Let $z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon$ with $t \in ( 0 , 1 ) , \epsilon \sim \mathcal { N } ( 0 , I )$ independent of $z _ { \mathrm { 0 } } ,$ and z0 drawn from any distribution with inite second moment, and let r have finite variance. Then the value $V _ { t } ( z ) = \mathbb { E } [ r ( z _ { 0 } ) \mid z _ { t } = z ]$ and the Jacobian $D _ { t }$ of the exact posterior mean satisfy

$$
\begin{array} { r } { \nabla V _ { t } ( z ) = \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \big [ z _ { 0 } , r ( z _ { 0 } ) \big | z _ { t } = z \big ] , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { D _ { t } ( z ) = \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \ [ z _ { 0 } \mid z _ { t } = z ] . } \end{array}\tag{10}
$$

Proof. (i) Posterior score. Since $p ( z _ { 0 } \mid z _ { t } = z ) \propto p ( z _ { 0 } ) k _ { t } ( z \mid z _ { 0 } )$

$$
\nabla _ { z } \log { p ( z _ { 0 } \mid z _ { t } = z ) } = \frac { ( 1 - t ) z _ { 0 } - z } { t ^ { 2 } } - \nabla _ { z } \log { p _ { t } ( z ) } .\tag{21}
$$

The posterior expectation of this score is zero, so $\nabla _ { z } \log { p _ { t } ( z ) } = \big ( ( 1 - t ) \hat { z } _ { 0 | t } - z \big ) / t ^ { 2 }$ . Therefore,

$$
\begin{array} { r } { \nabla _ { z } \log p ( z _ { 0 } \mid z _ { t } = z ) = \frac { 1 - t } { t ^ { 2 } } \big ( z _ { 0 } - \hat { z } _ { 0 \mid t } \big ) . } \end{array}\tag{22}
$$

(ii) Value gradient. Differentiating $\begin{array} { r } { V _ { t } ( z ) = \int r ( z _ { 0 } ) p ( z _ { 0 } \mid z _ { t } = z ) d z _ { 0 } } \end{array}$ under the integral and using (i),

$$
\begin{array} { r l } & { \nabla V _ { t } ( z ) = \frac { 1 - t } { t ^ { 2 } } \mathbb { E } \big [ r ( z _ { 0 } ) \big ( z _ { 0 } - \hat { z } _ { 0 \mid t } \big ) \mid z _ { t } = z \big ] } \\ & { \qquad = \frac { 1 - t } { t ^ { 2 } } \mathbb { E } \big [ \big ( r ( z _ { 0 } ) - V _ { t } ( z ) \big ) \big ( z _ { 0 } - \hat { z } _ { 0 \mid t } \big ) \mid z _ { t } = z \big ] , } \end{array}\tag{23}
$$

where the second line uses $\mathbb { E } [ z _ { 0 } - \hat { z } _ { 0 \mid t } \mid z _ { t } ] = 0 .$ Therefore, Eq. (9) holds.

(iii) Jacobian of the posterior mean. Applying (ii) to each coordinate $r ( z _ { 0 } ) = z _ { 0 , i }$ , which has finite variance by the moment assumption, gives the second-order Tweedie identity [39],

$$
\begin{array} { r } { \nabla _ { z } \hat { z } _ { 0 \mid t , i } = \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \big [ z _ { 0 } , z _ { 0 , i } \mid z _ { t } = z \big ] . } \end{array}\tag{24}
$$

Therefore, Eq. (10) holds, and $D _ { t }$ is symmetric.

## B.2. Errors of DPS, a Learned Value, and AGF

This subsection states and proves the three error expressions of Sec. 3.

Proposition B.1 (DPS error). For the exact denoiser of p, DPS satisfies

$$
\begin{array} { r } { { \pmb g } _ { t } ^ { \mathrm { D P S } } - \nabla V _ { t } = - \frac { 1 - t } { t ^ { 2 } } \operatorname { C o v } \big [ { \boldsymbol { z } } _ { 0 } , r ( { \boldsymbol { z } } _ { 0 } ) - { \bar { r } } ( { \boldsymbol { z } } _ { 0 } ) \mid { \boldsymbol { z } } _ { t } \big ] , } \end{array}\tag{11}
$$

where ī is the first-order Taylor expansion of r at $\hat { z } _ { 0 \mid t }$ 4

Proof. By Eqs. (5) and (10),

$$
\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { D P S } } = \frac { 1 - t } { t ^ { 2 } } \mathrm { C o v } [ z _ { 0 } \mid z _ { t } ] \nabla r ( \hat { z } _ { 0 \mid t } ) = \frac { 1 - t } { t ^ { 2 } } \mathrm { C o v } \left[ z _ { 0 } , \bar { r } ( z _ { 0 } ) \mid z _ { t } \right] , } \end{array}\tag{25}
$$

since $\bar { r } ( z _ { 0 } ) = r ( \hat { z } _ { 0 \mid t } ) + \nabla r ( \hat { z } _ { 0 \mid t } ) ^ { \top } ( z _ { 0 } - \hat { z } _ { 0 \mid t } )$ differs from $\nabla r \big ( \hat { z } _ { 0 \mid t } \big ) ^ { \top } z _ { 0 }$ by a constant. Subtracting Eq. (9), the error is the posterior covariance of z0 with the nonlinear part $r - \bar { r }$ of the reward. Therefore, Eq. (11) holds. □

Proposition B.2 (Learned-value error). A value model regressed onto the reward of samples from q given their forwardnoised versions converges to $V _ { t } ^ { q } ( z ) = \mathbb { E } _ { q } [ r ( z _ { 0 } ) \mid z _ { t } = z ]$ so that $\pmb { g } _ { t } ^ { \mathrm { S V M } } = \nabla V _ { t } ^ { q }$ satisfies

$$
\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { S V M } } - \nabla V _ { t } = \frac { 1 - t } { t ^ { 2 } } \left( \mathrm { C o v } _ { q } - \mathrm { C o v } _ { p } \right) \left[ \boldsymbol { z } _ { 0 } , r ( \boldsymbol { z } _ { 0 } ) \mid \boldsymbol { z } _ { t } \right] . } \end{array}\tag{12}
$$

Proof. (i) Regression target. Over all functions of $( z _ { t } , t )$ , the loss $\mathbb { E } _ { q } \Vert V ( z _ { t } , t ) - r ( z _ { 0 } ) \Vert ^ { 2 }$ with $z _ { \mathrm { 0 } } \sim q$ and $z _ { t } \sim k _ { t } ( \cdot \mid z _ { 0 } )$ is minimized by the conditional mean under q. Therefore, $\pmb { g } _ { t } ^ { \mathrm { S V M } } = \dot { \nabla } V _ { t } ^ { q }$

(ii) Error. Lemma 1 holds for any distribution of $z _ { \mathrm { 0 } }$ , so it applies under q and under $p ,$

$$
\begin{array} { r } { \nabla V _ { t } ^ { q } = \frac { 1 - t } { t ^ { 2 } } \mathrm { C o v } _ { q } \big [ z _ { 0 } , r ( z _ { 0 } ) \mid z _ { t } \big ] , } \end{array}\tag{26}
$$

$$
\begin{array} { r } { \nabla V _ { t } = \frac { 1 - t } { t ^ { 2 } } \mathrm { C o v } _ { p } \left[ z _ { 0 } , r ( z _ { 0 } ) \ | \ z _ { t } \right] . } \end{array}\tag{27}
$$

Therefore, subtracting the two, Eq. (12) holds.

Remark. StitchVM regresses reward-model features rather than the reward itself, so Prop. B.2 applies to rewards that are affine in these features.

Proposition B.3 (AGF gap). For a linear reward $\begin{array} { r } { r ( \boldsymbol { z } _ { 0 } ) = \boldsymbol { w } ^ { \top } \boldsymbol { z } _ { 0 } , } \end{array}$ AGF with a head that matches its target at zero guidance, $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { A G F } } = \big ( \frac { \partial z _ { 0 } } { \partial z _ { t } } \big ) ^ { \top } \nabla r ( z _ { 0 } ) } \end{array}$ with ${ z } _ { 0 } = \Psi _ { t } ( z _ { t } )$ the endpoint of the probability-flow ODE of p from t to 0, satisfies

$$
{ \pmb { g } } _ { t } ^ { \mathrm { A G F } } - \nabla V _ { t } = \bigg ( \frac { \partial z _ { 0 } } { \partial z _ { t } } - D _ { t } \bigg ) ^ { \top } { \pmb { w } } .\tag{13}
$$

Proof. (i) Value gradient. For a linear reward, Cov $\cdot [ z _ { 0 } , \pmb { w } ^ { \top } z _ { 0 } \mid z _ { t } ] = \operatorname { C o v } [ z _ { 0 } \mid z _ { t } ] \ : \pmb { w }$ , so Eqs. (9) and (10) give

$$
\begin{array} { r } { \nabla V _ { t } = \frac { 1 - t } { t ^ { 2 } } \mathrm { C o v } [ z _ { 0 } \mid z _ { t } ] w = D _ { t } { w } = D _ { t } ^ { \top } w , } \end{array}\tag{28}
$$

where the last equality uses the symmetry of $D _ { t }$ Therefore, $\nabla V _ { t } = D _ { t } ^ { \top } \boldsymbol { w }$

(ii) Gap. Since $\nabla r = w$ at every point, $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { A G F } } = \left( \frac { \partial \pmb { z } _ { 0 } } { \partial \pmb { z } _ { t } } \right) ^ { \top } } \end{array}$ w. Therefore, subtracting (i), Eq. (13) holds.

Remark (path form). Along the path $z _ { \tau }$ from $\tau = t \mathrm { t o } \ 0$ , the sensitivity obeys $\begin{array} { r } { \frac { d } { d \tau } \frac { \partial z _ { \tau } } { \partial z _ { t } } = J _ { v } ( z _ { \tau } , \tau ) \frac { \partial z _ { \tau } } { \partial z _ { t } } } \end{array}$ with $J _ { v } = ( I -$ $D _ { \tau } ) / \tau$ , so that

$$
\frac { \partial z _ { 0 } } { \partial z _ { t } } = \mathcal { T } \exp \Big ( - \int _ { 0 } ^ { t } \frac { I - D _ { \tau } ( z _ { \tau } ) } { \tau } d \tau \Big ) ,\tag{29}
$$

a time-ordered product of the Jacobians $D _ { \tau }$ along this one path, each a scaled posterior covariance by Eq. (10), whereas $\nabla V _ { t }$ involves only $D _ { t }$ at ${ \boldsymbol { z } } _ { t }$

Corollary B.4 (Gaussian data). If moreover $p = \mathcal { N } ( \mu , \sigma ^ { 2 } I )$ , then $\pmb { g } _ { t } ^ { \mathrm { A G F } } = \rho _ { t } \nabla V _ { t }$ with $\rho _ { t } ~ = ~ \sqrt { v _ { t } } / \big ( ( 1 - t ) \sigma \big )$ and $v _ { t } = ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 } ,$ so that $\rho _ { t } \geq 1 , \rho _ { 0 } = 1$ , and $\rho _ { t }  \infty a s t  1$

Proof. (i) Posterior mean. The noisy state is $\mathbf { \mathscr { z } } _ { t } \sim \mathcal { N } ( \mathbf { m } _ { t } , \mathbf { \vec { v } } _ { t } I )$ with $m _ { t } = ( 1 - t ) \pmb { \mu } .$ and

$$
\begin{array} { r } { \hat { z } _ { 0 \mid t } = \pmb { \mu } + \frac { ( 1 - t ) \sigma ^ { 2 } } { v _ { t } } ( z _ { t } - m _ { t } ) , \qquad D _ { t } = \frac { ( 1 - t ) \sigma ^ { 2 } } { v _ { t } } I . } \end{array}\tag{30}
$$

Therefore, $\begin{array} { r } { \nabla V _ { t } = \frac { ( 1 - t ) \sigma ^ { 2 } } { v _ { t } } w } \end{array}$ by (i) of Prop. B.3.

(ii) Flow map. The optimal velocity of Eq. (1) is $\mathbb { E } [ \epsilon - z _ { 0 } \mid z _ { t } ]$ , which for this p is

$$
{ \boldsymbol { v } } ( z , t ) = - { \boldsymbol { \mu } } + { \frac { t - ( 1 - t ) \sigma ^ { 2 } } { v _ { t } } } \left( { \boldsymbol { z } } - { \boldsymbol { m } } _ { t } \right) = { \dot { \boldsymbol { m } } } _ { t } + { \frac { { \dot { v } } _ { t } } { 2 v _ { t } } } \left( { \boldsymbol { z } } - { \boldsymbol { m } } _ { t } \right) ,\tag{31}
$$

so the ODE keeps $\left( z - m _ { t } \right) / \sqrt { v _ { t } }$ constant, and

$$
\begin{array} { r } { \Psi _ { t } ( z ) = \pmb { \mu } + \frac { \sigma } { \sqrt { v _ { t } } } ( z - \pmb { m } _ { t } ) , \qquad \frac { \partial z _ { 0 } } { \partial z _ { t } } = \frac { \sigma } { \sqrt { v _ { t } } } I . } \end{array}\tag{32}
$$

Therefore, $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { A G F } } = \frac { \sigma } { \sqrt { v _ { t } } } \pmb { w } } \end{array}$

(iii) Ratio. Dividing (ii) by (i),

$$
g _ { t } ^ { \mathrm { A G F } } = \rho _ { t } \nabla V _ { t } , \qquad \rho _ { t } = \frac { \sigma / \sqrt { v _ { t } } } { ( 1 - t ) \sigma ^ { 2 } / v _ { t } } = \frac { \sqrt { v _ { t } } } { ( 1 - t ) \sigma } .\tag{33}
$$

Since $v _ { t } \geq ( 1 - t ) ^ { 2 } \sigma ^ { 2 }$ with equality at $t = 0$ , and $v _ { t } \to 1$ while $( 1 - t ) \sigma \to 0 \mathrm { a s } t \to 1$ , the ratio is at least one, equals one at $t = 0$ , and diverges as $t \to 1$ . Therefore, Cor. B.4 holds. □

## B.3. Proof of Proposition 1

Proposition 1 (restated). Let $V _ { \psi }$ and ${ \pmb v } _ { \theta }$ be differentiable. Then

$$
\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \underbrace { \frac { s } { t } \pmb { g } _ { s } } _ { l e a r n e d - \nu a l u e f o r m } + \underbrace { \left( 1 - \frac { s } { t } \right) D _ { t } ^ { \top } \pmb { g } _ { s } } _ { D P S f o r m } . } \end{array}\tag{15}
$$

In particular, $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \pmb { g } _ { t } ^ { \mathrm { S V M } } } \end{array}$ at $s = t ,$ and $i f V _ { \psi } ( \cdot , 0 ) = r ,$ then $\pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \pmb { g } _ { t } ^ { \mathrm { D P S } } \ : a t \ : s = 0$

Proof. (i) Jacobian of the step. By Eq. (2), $\Phi _ { t  s } ( z ) = z - ( t - s ) \mathbf { \mathscr { v } } _ { \theta } ( z , t )$ , and $D _ { t } = I - t J _ { v }$ since $\hat { z } _ { 0 \mid t } = z - t { \pmb v } _ { \theta }$ Therefore, the step has the Jacobian

$$
\begin{array} { r } { P _ { s } : = \partial _ { z } \Phi _ { t  s } ( z _ { t } ) = I - ( t - s ) J _ { v } = \frac { s } { t } I + \bigl ( 1 - \frac { s } { t } \bigr ) D _ { t } . } \end{array}\tag{34}
$$

(ii) Decomposition. Since ${ \mathbf g } _ { t } ^ { \mathrm { V a L o P } }$ is the gradient of $V _ { \psi } ( \cdot , s ) \circ \Phi _ { t  s }$ at ${ \boldsymbol { z } } _ { t }$ and $z _ { s } = \Phi _ { t  s } ( z _ { t } )$ , the chain rule and (i) give

$$
\begin{array} { r l } & { \pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \pmb { P } _ { s } ^ { \top } \pmb { g } _ { s } = \frac { s } { t } \pmb { g } _ { s } + \left( 1 - \frac { s } { t } \right) \pmb { D } _ { t } ^ { \top } \pmb { g } _ { s } } \\ & { \qquad = \pmb { g } _ { s } - ( t - s ) \pmb { J } _ { v } ^ { \top } \pmb { g } _ { s } . } \end{array}\tag{35}
$$

Therefore, Eqs. (14) and (15) hold.

(iii) The two ends. $\mathrm { A t } \ s = t , \Phi _ { t  t }$ is the identity, $z _ { s } = z _ { t }$ , and the second term of Eq. (15) vanishes. $\mathrm { A t } s = 0 , z _ { s } = \hat { z } _ { 0 \mid t }$ the first term vanishes, and $V _ { \psi } ( \cdot , 0 ) = r$ . Hence,

$$
s = t : \ g _ { t } ^ { \mathrm { V a L o P } } = \nabla V _ { \psi } ( z _ { t } , t ) = g _ { t } ^ { \mathrm { S V M } } ,\tag{36}
$$

$$
\begin{array} { r } { s = 0 : \quad g _ { t } ^ { \mathrm { V a L o P } } = D _ { t } ^ { \top } \nabla r ( \hat { z } _ { 0 | t } ) = g _ { t } ^ { \mathrm { D P S } } , } \end{array}\tag{37}
$$

by Eqs. (5) and (7). Therefore, VaLoP coincides with learned-value guidance at $s = t$ and with DPS at $s = 0$

## B.4. Proof of Proposition 2

Proposition 2 (restated). Let $\mathrm { K L } ( p \| q ) ~ < ~ \infty , ~ | r | ~ \leq ~ B ,$ and $q _ { s }$ be the marginal $o f z _ { s }$ under $q ,$ and let $\begin{array} { r l } { \Delta _ { s } } & { { } = } \end{array}$ $\mathbb { E } _ { z _ { s } \sim p _ { s } } { \mathrm { K L } } \big ( p ( z _ { 0 } \mid z _ { s } ) \big \| q ( z _ { 0 } \mid z _ { s } ) \big )$ be the posterior mismatch at level s. Then

$$
\Delta _ { s } = \mathrm { K L } ( p \parallel q ) - \mathrm { K L } ( p _ { s } \parallel q _ { s } ) ,\tag{18}
$$

which is nondecreasing in s with $\Delta _ { 0 } = 0$ and $\Delta _ { 1 } = \mathrm { K L } ( p \parallel q ) , a n d \mathbb { E } _ { z _ { s } \sim p _ { s } } \left| V _ { s } ^ { q } ( z _ { s } ) - V _ { s } ( z _ { s } ) \right| \le B \sqrt { 2 \Delta _ { s } } .$

Proof. (i) Chain rule. Under p and $q ,$ the pair $( z _ { 0 } , z _ { s } )$ has the same forward kernel $k _ { s } ( z _ { s } \mid z _ { 0 } )$ , which adds no divergence. Factorizing the divergence of the joint distributions through $z _ { \mathrm { 0 } }$ and through $z _ { s }$

$$
\begin{array} { r l } & { \mathrm { K L } \big ( p ( z _ { 0 } , z _ { s } ) \| q ( { z } _ { 0 } , { z } _ { s } ) \big ) = \mathrm { K L } ( p \| q ) } \\ & { ~ = \mathrm { K L } ( p _ { s } \| q _ { s } ) + \Delta _ { s } . } \end{array}\tag{38}
$$

Therefore, Eq. (18) holds.

(ii) Monotonicity. Let $s \leq t$ with $\textstyle s < 1 , a = { \frac { 1 - t } { 1 - s } }$ , and $c ^ { 2 } = t ^ { 2 } - a ^ { 2 } s ^ { 2 }$ , which is nonnegative since $t ( 1 - s ) \geq s ( 1 - t )$ . For $\epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$ independent of $z _ { s }$

$$
a z _ { s } + c \epsilon ^ { \prime } \mid z _ { 0 } \sim \mathcal { N } \big ( ( 1 - t ) z _ { 0 } , t ^ { 2 } I \big ) ,\tag{39}
$$

the law of ${ \boldsymbol { z } } _ { t }$ given $z _ { \mathrm { 0 } }$ .Hence, $p _ { t }$ and $q _ { t }$ arise from $p _ { s }$ and $q _ { s }$ through one Markov kernel, and the data processing inequality and (i) give

$$
\begin{array} { r } { \mathrm { K L } ( p _ { t } \parallel q _ { t } ) \leq \mathrm { K L } ( p _ { s } \parallel q _ { s } ) \quad \implies \quad \Delta _ { s } \leq \Delta _ { t } . } \end{array}\tag{40}
$$

$\mathrm { A t } ~ s = 0 , p _ { 0 } = p$ and $q _ { 0 } = q$ , and at $s = 1 , p _ { 1 } = q _ { 1 } = \mathcal { N } ( 0 , I )$ . Therefore, $\Delta _ { s }$ is nondecreasing in s with $\Delta _ { 0 } = 0$ and $\Delta _ { 1 } = \mathrm { K L } ( p \parallel q )$

(iii) Value bound. At a fixed $z _ { s } , | r | \leq B$ and Pinsker's inequality give

$$
\begin{array} { r l } & { | V _ { s } ^ { q } ( z _ { s } ) - V _ { s } ( z _ { s } ) | = | \mathbb { E } _ { q } [ r ( z _ { 0 } ) \mid z _ { s } ] - \mathbb { E } _ { p } [ r ( z _ { 0 } ) \mid z _ { s } ] | } \\ & { \phantom { | } \le 2 B \operatorname { T V } \bigl ( p ( \cdot \mid z _ { s } ) , q ( \cdot \mid z _ { s } ) \bigr )  } \\ & { \phantom { | } \le B \sqrt { 2 \operatorname { K L } \bigl ( p ( \cdot \mid z _ { s } ) \parallel q ( \cdot \mid z _ { s } ) \bigr ) } . } \end{array}\tag{41}
$$

Averaging over $p _ { s }$ and applying Jensen's inequality to the square root,

$$
\begin{array} { r } { \mathbb { E } _ { z _ { s } \sim p _ { s } } \big | V _ { s } ^ { q } ( z _ { s } ) - V _ { s } ( z _ { s } ) \big | \le B \sqrt { 2 \Delta _ { s } } . } \end{array}\tag{42}
$$

Therefore, the value under q deviates from that under $p$ by at most $B \sqrt { 2 \Delta _ { s } }$ on average at level s.

## B.5. Error Split and the Correction

This subsection proves the statements of Sec. 4.2 on Eq. (17) and on the correction.

Lemma B.5 (ends of the split). The second term of Eq. (17) is zero at s = t and is the DPS error $\begin{array} { r } { \pmb { g } _ { t } ^ { \mathrm { D P S } } - \nabla V _ { t } \ : a t \ : s = 0 . } \end{array}$

Proof. At $s = t , P _ { t } = I$ and $z _ { s } ~ = ~ z _ { t } .$ so the second bracket of Eq. (17) is $\nabla V _ { t } ( z _ { t } ) - \nabla V _ { t } ( z _ { t } ) = 0 . \mathrm { ~ A t ~ } s = 0 , P _ { 0 } =$ $I - t J _ { v } = D _ { t } , z _ { s } = \hat { z } _ { 0 \mid t }$ , and $V _ { 0 } = r$ by $\mathrm { E q . } \left( 4 \right)$ , SO

$$
\begin{array} { r l } & { P _ { 0 } ^ { \top } \nabla V _ { 0 } ( \hat { z } _ { 0 \mid t } ) - \nabla V _ { t } ( z _ { t } ) = D _ { t } ^ { \top } \nabla r ( \hat { z } _ { 0 \mid t } ) - \nabla V _ { t } ( z _ { t } ) } \\ & { \qquad = \pmb { g } _ { t } ^ { \mathrm { D P S } } - \nabla V _ { t } ( z _ { t } ) . } \end{array}\tag{43}
$$

Therefore, Lemma B.5 holds.

Lemma B.6 (correction for a linear reward). Let $\begin{array} { r } { r ( \boldsymbol { z } _ { 0 } ) = \boldsymbol { w } ^ { \top } \boldsymbol { z } _ { 0 } , V _ { \psi } ( \cdot , 0 ) = r , } \end{array}$ and the denoiser be exact. $A t \ s = 0 ;$ VaLoP is exact, $\pmb { g } _ { t } ^ { \mathrm { V a L o P } } = \nabla V _ { t } ,$ , whereas the uncorrected $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ has the error $( I - D _ { t } ) ^ { \top } \pmb { w }$ , which tends to w as $t \to 1$

Proof. (i) Exact value gradient. For the exact denoiser, $V _ { t } ( \pmb { z } _ { t } ) = \pmb { w } ^ { \top } \hat { \pmb { z } } _ { 0 \mid t }$ . Therefore, $\nabla V _ { t } = D _ { t } ^ { \top } \mathbf { \boldsymbol { u } }$

(ii) With and without the correction. $\mathbf { A } \mathbf { t } \ s = 0 .$ , Prop. 1 and $\nabla r = w$ give

$$
\pmb { g } _ { t } ^ { \mathrm { V a L o P } } = D _ { t } ^ { \top } \pmb { w } = \nabla V _ { t } ,\tag{44}
$$

$$
\begin{array} { r } { \pmb { g } _ { s } - \nabla V _ { t } = \pmb { w } - \pmb { D } _ { t } ^ { \top } \pmb { w } = ( I - \pmb { D } _ { t } ) ^ { \top } \pmb { w } . } \end{array}\tag{45}
$$

Therefore, the corrected guidance is exact at $s = 0 .$ , and the uncorrected one has the error $( I - D _ { t } ) ^ { \top } \pmb { w }$

(iii) High noise. By Eq. (10), $\begin{array} { r } { D _ { t } = { \frac { 1 - t } { t ^ { 2 } } } \mathrm { C o v } [ z _ { 0 } \mid z _ { t } ] } \end{array}$ , which vanishes as $t \to 1$ , where the posterior approaches the prior and its covariance stays finite. Therefore, the error of $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ tends to w. □

Numerical check for intermediate ratios. For $z _ { 0 } \sim \mathcal { N } ( \mu , \Sigma )$ , the exact denoiser has

$$
D _ { t } = ( 1 - t ) \Sigma \big ( ( 1 - t ) ^ { 2 } \Sigma + t ^ { 2 } I \big ) ^ { - 1 } ,\tag{46}
$$

and with a linear reward and the sampler step of Eq. (2), all three gradients are functions of Σ applied to w, namely $\nabla V _ { t } =$ $D _ { t } \mathbf { \boldsymbol { w } }$ , the uncorrected $\mathbf { \mathit { g } } _ { s } = D _ { s } \mathbf { \mathit { w } }$ , and the corrected $P _ { s } ^ { \top } { \pmb g } _ { s } = P _ { s } D _ { s } { \pmb w }$ . Over $1 0 ^ { 4 }$ draws of an eight-dimensional $\Sigma$ with eigenvalues log-uniform in [0.01, 4], a Gaussian w, $t \in [ 0 . 0 5 , 0 . 9 9 ]$ , and $s / t \in [ 0 . 0 5 , 0 . 9 5 ]$ , the corrected gradient is at least as aligned with $\nabla V _ { t }$ as the uncorrected one in every draw, with mean cosine 0.986 against 0.932.

## C. Scope and Limitations

Requirements. VaLoP requires a value model for the generator's latent space, such as that of StitchVM, or a critic of clean actions for a policy, and VaLoP-Fast requires training a head once per generator, or once per task for a policy.

First-order guidance. Like the methods we compare with, VaLoP guides with the gradient of the value Vt, the leadingorder term of the soft value in $\begin{array} { r } { \lambda , \mathcal { V } _ { t } = V _ { t } + \frac { \lambda } { 2 } \mathrm { V a r } \big [ r ( z _ { 0 } ) \mid z _ { t } \big ] + O ( \bar { \lambda ^ { 2 } } ) } \end{array}$ [19], and tunes the guidance strength η instead of deriving it from λ.

Posterior mismatch. Proposition 2 bounds the error of the value that a value model regresses, averaged over states of the generator, rather than the error of its gradient. Since Eq. (12) scales the difference of the posterior covariances by $( 1 - s ) / s ^ { 2 }$ the gradient error need not decrease monotonically as s decreases. The lookahead state also follows $p _ { s }$ only when $\Phi _ { t  s }$ is the exact flow map, which the sampler step approximates. Where a distilled flow map of the generator is available [22, 23], it can replace the sampler step and remove this bias.

Real robots. All robot experiments run in simulation, on LIBERO and RoboCasa. We will evaluate VaLoP and VaLoP-Fast on a real robot in future work.

## D. Experimental Details

## D.1. Two-Dimensional Mixtures

Generator. The generator is a mixture of three Gaussians with masses 0.58, 0.32, and 0.10. A flow, a 4-layer MLP of width 256 with inputs $( z _ { t } , t ,$ sin πt, cos πt, log $t , { \sqrt { 1 - t } } )$ , is trained on it with the flow-matching loss for 20k steps (batch 2048, Adam, cosine schedule).

Reward. The reward is $r = \kappa \langle { \pmb u } , { \pmb e } \rangle$ with the embedding ${ \pmb u } ( z _ { 0 } ) = \mathrm { n o r m a l i z e } ( { \operatorname { t a n h } } ( A z _ { 0 } + b ) + o ) \in \mathbb { R } ^ { 8 } , \kappa = 5$ , and $\lambda = 0 . 6 5$ . The readout e is the embedding direction of the rewarded component.

Value models and their data. Value models are 3-layer MLPs (width 128, SiLU) with inputs $( z _ { t } , t , \sqrt { t } ,$ log t, prompt) that output the embedding, guide with $\kappa \nabla _ { z } \langle V _ { \psi } , e \rangle$ , and are trained for 8000 steps on forward-noised data in which the rewarded component is rarer and displaced, with mass 0.03, mean $( 1 . 1 , - 1 . 5 )$ , and covariance scaled by 0.8. Training is not bitwise reproducible, and Fig. 3 uses the first of three value models trained with different seeds, except that the StitchVM curves of Fig. 3b average all three.

Sampler and methods. Guidance acts on $t \leq 0 . 9 .$ Samplers use 100 steps and start from the exact reward-tilted marginal at $t = 0 . 9 ,$ so that only the guidance error is compared. Each step shifts $\hat { z } _ { 0 \mid t }$ by the guidance and transitions with the Gaussian kernel of the memoryless forward process. AGF is trained for 4 rounds of 2000 trajectories with vector-Jacobian products through ${ \pmb v } _ { \theta }$ and uses the ODE sampler of its training. DPS backpropagates through the learned denoiser. VaLoP uses s = max(0.6t, 0.004), the value model of StitchVM, and one vector-Jacobian product through ${ \pmb v } _ { \theta }$ . Strengths are swept over $\eta \in \lambda \times \{ 0 . 5 , 0 . 7 5 , 1 , 1 . 5 , 2 , 3 \}$ for the exact value and DPS, over $\lambda \times \{ 0 . 5 , \ldots , 6 \}$ for StitchVM and VaLoP, and over $\{ 0 . 2 5 , \ldots , 8 \}$ for AGF. Extending the DPS sweep to 16λ does not lower its $\mathrm { { \bf M M D } ^ { 2 } }$

Metrics. MMD2 [20] uses an RBF kernel summed over bandwidths 0.2, 0.5, and 1.0 with 3000 samples per set. The floor is the $\mathrm { { \bf M M D } ^ { 2 } }$ between two independent target draws, and negative unbiased estimates are printed as 0. The off-manifold rate is the fraction of samples whose log p falls below the 1% quantile of generator samples. Direction diagnostics use 1000 chains of the exact-guided sampler at $\eta \in \{ \lambda , 2 \lambda \}$ with 4 states each. In Fig. 3b, the VaLoP curve compares $\nabla V _ { \psi } ( z _ { s } , s )$ with the exact $\nabla V _ { s } ( z _ { s } )$ at the lookahead state $z _ { s } = \Phi _ { t  s } ( z _ { t } ) , s = 0 . 6 t$ , of the same states. In Fig. 3b, AGF is represented by the costate it regresses, computed along the deterministic path of the learned flow from ${ \boldsymbol { z } } _ { t }$ to $t = 0$ with zero guidance.

## D.2. Text-to-Image Generation

Protocol. Models, prompts, and metrics are as in Sec. 5.1, and all methods share the seeds. The SD3.5 models use 100 steps and a CFG scale of 4.5, as in StitchVM, and FLUX.1-dev uses 28 steps and $\gamma = 1$ , as it is guidance-distilled. Costs in Tab. 3 are measured for one 1024 × 1024 image on one H100 with these sampling steps. Our implementation compiles the generator transformer with torch.compile in its default mode, runs the generator, the VAE, and the text encoder in bf16 as the official code does, caches the text embeddings and moves the text encoder to the CPU after encoding, and decodes each image once. Specific to our methods, the sampler reuses the velocity computed for the guidance, gradient checkpointing on FLUX.1-dev acts only when backpropagating through the generator, as VaLoP does, and the head of VaLoP-Fast runs in bf16, which lowers its memory but not its time. For Tab. 3, StitchVM runs in our implementation with the general optimizations above. With its official implementation, it takes 19.0 and 38.0 s with 28.2 and 44.5 GB on SD3.5-Medium and SD3.5-Large, and the same on FLUX.1-dev. Pixel-space DPS runs the official StitchVM implementation, and on FLUX.1-dev its checkpoints are read on the CPU and freed after loading. Both implementations use the default scaled dot-product attention of diffusers.

AGF is not run on the image generators, as it trains a guidance network for each reward by backpropagating through all sampling steps. In Fig. 3c, VaLoP uses $\gamma = 2$ and the bars show one standard deviation. For Fig. 3c, we take 16 states in each of four noise bands from SD3.5-Medium trajectories at $5 1 2 ^ { 2 }$ on 14 HPDv2 training prompts with the HPSv2 reward, where pixel-space DPS backpropagates HPSv2 through the decoder and the denoiser, and measure the HPSv2 gain along each unit direction by central differences of size $0 . 0 2 \| z _ { t } \|$ , averaged over eight memoryless continuations of 40 steps with shared noise

Guidance strengths. Tab. 8 lists the strengths used in Tab. 1. All strengths are chosen separately for each reward by a sweep. $\mathrm { { A t } 5 1 2 ^ { 2 } }$ , the HPSv2 of StitchVM stops rising beyond $\eta = 4 0 0$ while its held-out metrics fall, and VaLoP exceeds it already at its lowest strength (Tab. 7). The lookahead ratio and the Jacobian scale are chosen on held-out HPDv2 prompts at $5 1 2 ^ { 2 }$ , and the VaLoP row of Tab. 18 uses $\gamma = 2$ at every step.

Table 7. Guidance strength (SD3.5-Medium, $5 1 2 ^ { 2 }$ , HPSv2 reward, HPDv2 test prompts). Bold: best guided result per column
<table><tr><td>Method</td><td>η</td><td>HPSv2</td><td>Aes.</td><td>CLIP</td><td>Pick</td><td>IR</td></tr><tr><td>Unguided</td><td>一</td><td>0.302</td><td>5.967</td><td>29.02</td><td>22.58</td><td>1.126</td></tr><tr><td rowspan="3">DPS + StitchVM</td><td>200</td><td>0.402</td><td>6.102</td><td>28.97</td><td>23.69</td><td>1.449</td></tr><tr><td>400</td><td>0.417</td><td>5.939</td><td>28.64</td><td>23.56</td><td>1.408</td></tr><tr><td>800</td><td>0.417</td><td>5.789</td><td>27.76</td><td>23.16</td><td>1.265</td></tr><tr><td rowspan="3">VaLoP</td><td>150</td><td>0.428</td><td>6.127</td><td>29.40</td><td>24.27</td><td>1.526</td></tr><tr><td>200</td><td>0.434</td><td>6.128</td><td>29.42</td><td>24.21</td><td>1.510</td></tr><tr><td>400</td><td>0.437</td><td>6.070</td><td>29.06</td><td>24.01</td><td>1.438</td></tr><tr><td>VaLoP-Fast</td><td>200</td><td>0.425</td><td>6.118</td><td>29.36</td><td>24.12</td><td>1.509</td></tr></table>

Sampling. Alg. 2 gives the exact form of $\mathrm { A l g . }$ 1 for text-to-image models, with $\gamma ( t ) = 2$ for $t \geq 0 . 8$ and 1 elsewhere, and $J _ { \gamma } = J _ { c }$ on FLUX.1-dev, which has no CFG.

Algorithm 2 Guided sampling of a text-to-image model   
Require: vθ with CFG scale ω, value model $V _ { \psi }$ , ratio $\rho ,$ strength η, Jacobian scale $\gamma ( t )$ , head $h _ { \phi }$ and fit (a, b) (VaLoP-Fast)   
1: $z _ { 1 } \sim \mathcal { N } ( 0 , I )$   
2: for $t = 1 , 1 - \Delta t , \ldots , \Delta t$ do   
3: $\pmb { v }  \pmb { v } _ { u } + \omega ( \pmb { v } _ { c } - \pmb { v } _ { u } ) ; \ s  \rho t ; \ z _ { s }  z _ { t } - ( t - s ) $ lookahead   
4: ${ \pmb g } _ { s } \gets \nabla _ { z _ { s } } V _ { \psi } ( z _ { s } , s )$ ▶ value backward   
5: if VaLoP then   
6: $\pmb { \xi } _ { t } \gets - ( t - s ) \ b { J } _ { \gamma ( t ) } ^ { \top } \pmb { g } _ { s } , m _ { t } \gets \| \nabla V _ { \psi } ( \pmb { z } _ { t } , t ) \|$ VJP, second backward   
7: else   
8: $\sigma \gets \mathrm { r m s } ( { \pmb g } _ { s } ) , \ { \pmb \xi } _ { t } \gets \sigma h _ { \phi } ( z _ { t } , t , { \pmb g } _ { s } / \sigma ) , \ m _ { t } \gets e ^ { a ( t ) } \lVert { \pmb g } _ { s } \rVert ^ { 1 + b ( t ) }$ ▶ head, fit   
9: $y _ { t } \gets g _ { s } + \xi _ { t } ; \ z _ { t - \Delta t } \gets z _ { t } - \Delta t v + \eta m _ { t } y _ { t } / \lVert y _ { t } \rVert$   
10: return z0

## D.3. Correction Head of VaLoP-Fast

The head is trained once per generator for text-to-image models $( { \mathrm { A l g } } . 3 )$ and once per task for robot policies $\left( { \mathrm { A l g . ~ 4 } } \right)$

## D.3.1. Text-to-Image Models

Target. The head predicts the correction in units of the scale of $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ . With $\mathrm { r m s } ( \cdot )$ the root mean square over entries, it takes $\hat { \pmb g } _ { s } = \pmb { g } _ { s } / \mathrm { r m s } ( \pmb { g } _ { s } )$ and regresses $\pmb { \xi } _ { t } / \operatorname { r m s } ( \pmb { g } _ { s } )$ , so that its prediction scales with $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ by construction, and VaLoP-Fast guides with the direction ${ \pmb y } _ { t } = { \pmb g } _ { s } + \mathrm { r m s } ( { \pmb g } _ { s } ) { \pmb h } _ { \phi } ( { \pmb z } _ { t } , t , { \hat { \pmb g } } _ { s } )$ . On SD3.5, the targets use $\gamma = 2 .$ , and on FLUX.1-dev, which has no $\mathrm { C F G } , J _ { \gamma } = J _ { c }$

Table 8. Guidance strengths in Tab. 1 for the HPSv2 (H), aesthetic (A), and CLIP (C) rewards.
<table><tr><td rowspan="2">Method</td><td colspan="3">SD3.5-M</td><td colspan="3">SD3.5-L</td><td colspan="3">FLUX.1-dev</td></tr><tr><td>H</td><td>A</td><td>C</td><td>H</td><td>A</td><td>C</td><td>H</td><td>A</td><td>C</td></tr><tr><td>DPS</td><td>100</td><td>0.3</td><td>0.6</td><td>50</td><td>0.3</td><td>0.3</td><td>200</td><td>0.3</td><td>0.6</td></tr><tr><td>StitchVM</td><td>400</td><td>12</td><td>75</td><td>600</td><td>8</td><td>45</td><td>400</td><td>16</td><td>100</td></tr><tr><td>VaLoP</td><td>300</td><td>6</td><td>37.5</td><td>300</td><td>4</td><td>22.5</td><td>800</td><td>16</td><td>100</td></tr><tr><td>VaLoP-Fast</td><td>450</td><td>6</td><td>37.5</td><td>450</td><td>8</td><td>45</td><td>800</td><td>16</td><td>100</td></tr></table>

Algorithm 3 Training of VaLoP-Fast for a text-to-image model   
Require: frozen $v _ { \theta } .$ HPSv2 value $V _ { \psi }$ , training prompts ${ \mathcal { C } } ,$ ratio $\rho ,$ Jacobian scale $\gamma ,$ head $h _ { \phi }$   
States, at $5 1 2 ^ { 2 }$ and at $1 0 2 4 ^ { 2 }$   
1: for $c \in { \mathcal { C } }$ and eight steps $t \in [ 0 . 0 5 , 0 . 9 5 ]$ of Alg. 2 with VaLoP do   
2: $s \gets \rho t , \ z _ { s } \gets z _ { t } - ( t - s ) \ v _ { \theta } ( z _ { t } , t , c )$ lookahead   
3: $\begin{array} { r } { \pmb { g } ^ { ( 0 ) }  \nabla _ { z _ { s } } V _ { \psi } ( z _ { s } , s ) , \ \pmb { g } ^ { ( k ) }  \| \pmb { g } ^ { ( 0 ) } \| \pmb { u } _ { k } / \| \pmb { u } _ { k } \| , \ k = 1 , \ldots , 4 } \end{array}$ ${ \pmb u } _ { k } \colon$ random, pulled back   
4: $\pmb { \xi } ^ { ( k ) } \gets - ( t - s ) J _ { \gamma } ^ { \top } \pmb { g } ^ { ( k ) } , k = 0 , \ldots , 4$ one VJP each   
Head, trained at $5 1 2 ^ { 2 }$ and fne-tuned at $1 0 2 4 ^ { 2 }$   
5: repeat   
6: draw $( z _ { t } , t , c , k ) , \ \sigma \gets \mathrm { r m s } ( g ^ { ( k ) } )$   
7: $\mathcal { L } \gets \left\| h _ { \phi } ( z _ { t } , t , g ^ { ( k ) } / \sigma ) - \dot { \xi } ^ { ( k ) } / \sigma \right\| ^ { 2 } / \left\| \pmb { \xi } ^ { ( k ) } / \sigma \right\| ^ { 2 }$   
8: $\phi  \ddot { \phi } - \alpha \nabla _ { \phi } \mathcal { L }$ AdamW   
9: until the steps of Tab. 9   
10: φ ← arg maxφ cos $\left( h _ { \phi } , \xi ^ { ( 0 ) } \right)$ over checkpoints, on held-out states   
Magnitude, per reward   
11: a(t), b(t) ← arg min $\sum \left( \log \| \nabla V _ { \psi } ( z _ { t } , t ) \| - a ( t ) - ( 1 + b ( t ) ) \log \| g _ { s } \| \right) ^ { 2 }$ ▶ 20 bins of log t   
12: return $h _ { \phi } .$ $\hat { m } _ { t } = e ^ { a ( t ) } \lVert \dot { \pmb { g } } _ { s } \rVert ^ { 1 + b ( t ) }$

Table 9. Correction head: parameters, distillation states (strength η), and training steps (batch).
<table><tr><td colspan="3">SD3.5-Medium SD3.5-Large</td><td>FLUX.1-dev</td></tr><tr><td>Parameters</td><td>0.41B</td><td>0.87B</td><td>1.39B</td></tr><tr><td>States at  $5 1 2 ^ { 2 }$ </td><td>20,032 (200)</td><td>10,240 (200)</td><td>10,240 (400)</td></tr><tr><td>States at  $1 0 2 4 ^ { 2 }$ </td><td>7,680 (200)</td><td>1,280 (300)</td><td>2,560 (800)</td></tr><tr><td>Steps at  $5 1 2 ^ { 2 }$ </td><td>30k (16)</td><td>15k (8)</td><td>20k (8)</td></tr><tr><td>Steps at  $1 0 2 4 ^ { 2 }$ </td><td>8k (8)</td><td>3k (4)</td><td>3k (4)</td></tr></table>

Architecture. The head reuses the generator and runs in bf16 (Tab. 9). The frozen generator processes ${ \boldsymbol { z } } _ { t }$ through its first four transformer blocks, whose activations are shared with the sampling pass, and the head consists of trainable copies of the next four blocks, the output normalization, and the output projection, initialized from the generator. A patch embedding of $\hat { \mathbf { g } } _ { s }$ is added to the image tokens, and a linear map of the timestep embedding at s is added to that at t. Both added maps and the output projection are initialized at zero, so the head outputs zero at initialization and VaLoP-Fast starts from the uncorrected direction $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$

Training data. States are collected on-policy by running the guided sampler with the HPSv2 value on prompts from the HPDv2 training split, which is disjoint from every evaluation set, and recording eight steps per trajectory with $t \in [ 0 . 0 5 , 0 . 9 5 ]$ Each state stores $z _ { t } , t ,$ the prompt, $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } } .$ , the exact correction, and $\nabla V _ { \psi } ( z _ { t } , t )$ for the magnitude fit. Since the correction is linear in $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } } .$ each state also yields pairs for other inputs. We pull four random directions in the output space of the value model's encoder back to ${ \boldsymbol { z } } _ { s } ,$ scale them to the norm of $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ , and compute their corrections, which gives five input and target pairs per state.

Optimization. The head minimizes the squared error of Eq. (19) in the normalized units above, divided by the squared norm of the target, with one of the five pairs of a state drawn at random per example. We use AdamW without weight decay, a cosine schedule that decays to $5 \%$ of the initial rate, and gradient clipping at norm 1, training from scratch at ${ \bar { 5 } } 1 2 ^ { 2 }$ with learning rate $1 0 ^ { - 4 }$ and fine-tuning at $1 0 2 4 ^ { 2 }$ with $3 \times 1 0 ^ { - 5 }$ (Tab. 9). We hold out $5 \%$ of the data and keep the checkpoint with the highest cosine between the predicted and the exact direction on held-out states, evaluated every 1000 steps. The $5 1 2 ^ { 2 }$ head already aligns with the $1 0 2 4 ^ { 2 }$ correction (cosine 0.85). On one GPU, the head trains for 3.2, 4.2, and 6.1 h at $5 1 2 ^ { 2 }$ and is fine-tuned for 1.0, 0.25, and 1.8 h at $1 0 2 4 ^ { 2 }$ on SD3.5-Medium, SD3.5-Large, and FLUX.1-dev, using an A6000, an H100, and an A6000.

Magnitude fit. VaLoP computes $m _ { t }$ with a second value-model backward pass, and VaLoP-Fast replaces it with $\hat { m } _ { t }$ . The coefficients a and b of $\hat { m } _ { t } ~ ( \mathrm { S e c . } ~ 4 . 4 )$ are fitted by least squares in 20 bins of t in log space on exact pairs $( { \pmb g } _ { s } , \nabla V _ { \psi } ( { \pmb z } _ { t } , t ) )$ recorded on guided states, separately for each reward and resolution. Gradient norms differ between value models in scale and in how they change with the noise level, and the pairs come from states guided by the same reward. A fit on states guided by another reward mispredicts the magnitude. Fitting takes about half a day per reward. The fitted b is negative for all rewards and noise levels, so a larger lookahead gradient comes with a relatively smaller gradient at $z _ { t } .$ On held-out states at $5 1 2 ^ { 2 }$ , the fit explains 54% of the variance of the log magnitude $( R ^ { 2 } )$ , against 31% with t alone, and a small network on head features does no better.

Cost. Per step, VaLoP-Fast performs one generator forward pass, shared with the sampler, one value-model backward pass at ${ \boldsymbol { z } } _ { s } ,$ and one forward pass of the head, without a generator backward pass. Against StitchVM in the same implementation, it takes 13 to 15% less time on the three generators (Tab. 3).

## D.3.2. Robot Policies

In place of the squared error of Eq. (20), the head predicts the unit direction and the log-norm of the guidance of ${ \mathrm { V a L o P } } .$ whose norm is matched to that of the QGF gradient $\begin{array} { r } { \dot { \pmb { g } } _ { t } ^ { \mathrm { Q G F } } = \nabla _ { z _ { t } } Q \big ( o , z _ { t } - t \mathrm { s g } [ { \pmb v } _ { \theta } ( z _ { t } , t ) ] \big ) } \end{array}$ , the critic gradient at the denoised action with the policy Jacobian replaced by the identity [75] (Sec. 4.4), so that, unlike for images, VaLoP-Fast needs no critic at inference (Alg. 4). It is the FiLM MLP of AGF, with three hidden layers of width 128 and LayerNorm, on the frozen image encoder of the critic, with 0.75M (SmolVLA) and 0.90M $( \pi _ { 0 . 5 } )$ parameters, and takes the noisy action chunk and a 64-dimensional sinusoidal embedding of t. It is trained on 100 unguided episodes per task (300 on Goal), with states rolled out under the head being trained and taken at the 11 times $t \in \{ 1 , 0 . 9 , \dots , 0 \}$ , using AdamW at a constant learning rate of $3 \times 1 0 ^ { - 4 }$ with weight decay $1 0 ^ { - 4 }$ , gradient clipping at norm 1, and batch size 128 (32 on RoboCasa), which takes 1.6 to 2.9 h per task on one GPU (Tab. 12)

Algorithm 4 Training of VaLoP-Fast for a robot policy   
Require: frozen vθ, critic Q, unguided observations ${ \mathcal { O } } ,$ ratio $\rho ,$ head $\pmb { h } _ { \phi } = ( \hat { \pmb { u } } _ { \phi } , \hat { \ell } _ { \phi } )$   
1: repeat   
2: $o \sim \mathcal { O } , z _ { 1 } \sim \mathcal { N } ( 0 , I )$   
Rollout with the current head   
3: for $t = 1 , 0 . 9 , \ldots , 0 . 1$ do   
4: $\pmb { v }  \pmb { v } _ { \theta } ( z _ { t } , t , o ) ; \ z _ { t - \Delta t }  z _ { t } - \Delta t \mathbf { v } + \Delta t$ min $\left( e ^ { \hat { \ell } _ { \phi } } , \| \pmb { v } \| \right) \hat { \pmb { u } } _ { \phi } / \| \hat { \pmb { u } } _ { \phi } \|$ clipped at∥|v||   
Targets   
5: for $t = 1 , 0 . 9 , \ldots , 0 ,$ with $s = \rho t$ do   
6: $z _ { s } \gets z _ { t } - ( t - s ) \pmb { v } _ { \theta } ( z _ { t } , t )$ lookahead   
7: $\mathbf { \sigma } _ { g _ { s } } \gets \nabla _ { z _ { s } } Q \bigl ( o , z _ { s } - \mathop { s } _ { \mathbf { \sigma } } { v } _ { \theta } ( z _ { s } , s ) \bigr )$ critic at the denoised action   
8: $\pmb { y } _ { t }  \pmb { g } _ { s } - ( t - s ) \bar { J } ^ { \top } \pmb { g } _ { s }$ one VJP   
9: $m _ { t } \gets \left| \left| \nabla _ { z _ { t } } Q \big ( o , z _ { t } - t \mathrm { s g } [ v _ { \theta } ( z _ { t } , t ) ] \big ) \right| \right|$ QGF norm   
10: $\begin{array} { r } { \mathcal { L } \gets \sum _ { t } \big \| \hat { \boldsymbol { u } } _ { \phi } ( o , z _ { t } , t ) - \boldsymbol { y } _ { t } / \| \boldsymbol { y } _ { t } \| \big \| ^ { 2 } + \big ( \hat { \ell } _ { \phi } ( o , z _ { t } , t ) - \log m _ { t } \big ) ^ { 2 } } \end{array}$   
11: $\phi  \phi - { \overset { \cdot \cdot } { \alpha } } \nabla _ { \phi } { \mathcal { L } }$ AdamW   
12: until the step budget   
Inference: Alg. 5, without Q

## D.4. Robot Policies

Policies and critics. We use the public SmolVLA checkpoint trained on LIBERO and the $\pi _ { 0 . 5 }$ checkpoint trained on RoboCasa, both frozen, with the action chunking of AGF [25]: chunks of 50 actions, of which SmolVLA executes 10 and $\pi _ { 0 . 5 }$ all 50, and the critic scores the executed part. Each task has a critic, an ensemble of ten Q-networks on clean action chunks, reduced to the mean minus half the standard deviation and trained by temporal differences on unguided rollouts of the base policy. On LIBERO, we use the critics of AGF, and on RoboCasa, we train them with the recipe of AGF on about 640 unguided episodes per task. Each Q-network takes the 64-pixel camera images, with the robot state on RoboCasa, and the action chunk, and has three hidden layers of width 128. It is trained for 10k steps with Adam at learning rate $3 \times 1 0 ^ { - 4 }$ with 500 warm-up steps and cosine decay, batch size 256, gradient clipping at norm 1, discount 0.99, and a Polyak-averaged target network with rate 0.005, which guidance uses. AGF uses its official guidance heads on RoboCasa.

Baselines. Q-BoN samples N action chunks from the unguided policy and executes the one with the highest critic value, as in V-GPS [41]. QDPS evaluates the critic at the denoised action and backpropagates through the policy, as DPS does (QGF-Jacobian in [75]), whereas QGF [75] replaces the Jacobian of the denoised action with the identity and backpropagates only through the critic. AGF regresses the costate onto a guidance network trained per task. We omit QAM [29], which fine-tunes the policy.

VaLoP and VaLoP-Fast. VaLoP reads the critic at the denoised lookahead action with s = 0.6t, pulls its gradient back with one VJP, and rescales the result to the norm of the QGF gradient at $z _ { t } .$ VaLoP-Fast amortizes this guidance with the head of Sec. D.3.2, and Alg. 5 gives both samplers.

Algorithm 5 Guided sampling of a robot policy   
Require: ${ \boldsymbol { v } } _ { \boldsymbol { \theta } } ,$ critic $Q ,$ observation o, ratio $\rho ,$ strength $\eta = w \Delta t ,$ head $\pmb { h } _ { \phi } = ( \hat { \pmb { u } } _ { \phi } , \hat { \ell } _ { \phi } )$ (VaLoP-Fast)   
1: $z _ { 1 } \sim \mathcal { N } ( 0 , I )$   
2: for $t = 1 , 0 . 9 , \ldots , 0 . 1$ do   
3: $v  v _ { \theta } ( z _ { t } , t , o ) ; \ s  \rho t$   
4: if VaLoP then   
5: $z _ { s } \gets z _ { t } - ( t - s )$ v lookahead   
6: $\mathbf { \boldsymbol { g } } _ { s } \gets \nabla _ { z _ { s } } Q \big ( o , z _ { s } - \mathbin { \mathcal { s } } \boldsymbol { v } _ { \boldsymbol { \theta } } ( z _ { s } , s ) \big )$ critic at the denoised action   
7: $\pmb { y } _ { t }  \pmb { g } _ { s } - ( t - s ) \bar { J } ^ { \top } \pmb { g } _ { s }$ one VJP   
8: $m _ { t } \gets \left. \nabla _ { z _ { t } } Q \big ( o , z _ { t } - t \mathrm { s g } [ \pmb { v } ] \big ) \right.$ ▶ QGF norm   
9: else   
10: $\pmb { y } _ { t }  \hat { \pmb { u } } _ { \phi } ( o , z _ { t } , t ) , m _ { t }  e ^ { \hat { \ell } _ { \phi } ( o , z _ { t } , t ) }$ ▶ head, no critic   
11: $z _ { t - \Delta t } \gets z _ { t } - \Delta t v + \eta \operatorname* { m i n } ( m _ { t } , 1 ) y _ { t } / \lVert y _ { t } \rVert$ Eq. (16), clipped at norm 1   
12: return $z _ { \mathrm { 0 } }$

Protocol. Gradient guidance is clipped at norm 1, which rarely binds in practice, and the guidance of AGF at the norm of the policy velocity. For every method, one strength per suite is selected by the mean success rate of the suite, from $w \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ on LIBERO and w $\in \{ 0 . 5 , 1 , 2 , 4 \}$ on RoboCasa (Tabs. 10 and 11). We report 50 episodes per task, with time limits of 300, 220, and 520 steps on Goal, Spatial, and LIBERO-10 and the task horizons of RoboCasa, 450 to 1050 steps. Shared episode seeds fix the initial action noise.

Cost. Tab. 12 lists the costs on both policies, measured on one H100 at batch size one for one action chunk of ten flow steps, as medians over 30 stored observations after three warm-up runs. Memory is the peak GPU memory, including the policy of 605M (SmolVLA) or $4 . 1 4 \mathrm { B } \left( \pi _ { 0 . 5 } \right)$ parameters, with its increase over Base in parentheses. VaLoP-Fast matches the inference time of AGF with a head of similar size on both policies, and on RoboCasa it adds almost no memory, since its head also predicts the magnitude and the critic is not loaded at inference. On SmolVLA, its head trains in less than half the time of AGF per task, at a higher peak memory, since its targets differentiate the policy once more. VaLoP needs no training.

Table 10. LIBERO over the strength grid. Success rate (%) per suite, with the base policy at 76.4, 64.4, and 43.4. Bold: highest, reported in Tab. 2.
<table><tr><td></td><td colspan="5">Goal</td><td colspan="5">Spatial</td><td colspan="5">LIBERO-10</td></tr><tr><td>w</td><td>0.25</td><td>0.5</td><td>1</td><td>2</td><td>4</td><td>0.25</td><td>0.5</td><td>1</td><td>2</td><td>4</td><td>0.25</td><td>0.5</td><td>1</td><td>2</td><td>4</td></tr><tr><td>QGF</td><td>76.4</td><td>76.2</td><td>78.6</td><td>79.0</td><td>78.6</td><td>67.6</td><td>65.0</td><td>68.2</td><td>68.8</td><td>69.4</td><td>46.4</td><td>44.4</td><td>44.2</td><td>45.8</td><td>48.6</td></tr><tr><td>QDPS</td><td>77.8</td><td>79.4</td><td>78.8</td><td>74.8</td><td>80.2</td><td>65.8</td><td>65.0</td><td>66.6</td><td>68.2</td><td>68.0</td><td>43.6</td><td>46.0</td><td>41.6</td><td>47.6</td><td>47.4</td></tr><tr><td>AGF</td><td>77.4</td><td>80.0</td><td>77.6</td><td>77.6</td><td>80.0</td><td>64.6</td><td>64.8</td><td>65.6</td><td>69.2</td><td>68.0</td><td>46.0</td><td>46.0</td><td>48.0</td><td>44.8</td><td>45.2</td></tr><tr><td>VaLoP</td><td>79.2</td><td>78.0</td><td>80.4</td><td>79.0</td><td>80.6</td><td>66.6</td><td>65.0</td><td>65.8</td><td>69.4</td><td>68.2</td><td>45.4</td><td>47.4</td><td>45.2</td><td>49.0</td><td>45.2</td></tr><tr><td>VaLoP-Fast</td><td>78.0</td><td>77.0</td><td>79.0</td><td>76.0</td><td>81.8</td><td>67.6</td><td>67.8</td><td>66.6</td><td>68.6</td><td>69.8</td><td>44.8</td><td>49.4</td><td>44.8</td><td>44.4</td><td>45.8</td></tr></table>

Table 11. RoboCasa over the strength grid. Success rate (%) over the 18 atomic tasks, with the base policy at 44.6. Bold: highest, reported in Tab. 2.
<table><tr><td>w</td><td>0.5</td><td>1</td><td>2</td><td>4</td></tr><tr><td>QGF</td><td>44.8</td><td>45.3</td><td>44.4</td><td>47.1</td></tr><tr><td>QDPS</td><td>44.0</td><td>45.0</td><td>45.2</td><td>45.7</td></tr><tr><td>AGF</td><td>44.3</td><td>44.9</td><td>45.9</td><td>42.7</td></tr><tr><td>VaLoP</td><td>44.4</td><td>43.4</td><td>46.3</td><td>47.4</td></tr><tr><td>VaLoP-Fast</td><td>44.7</td><td>46.8</td><td>45.8</td><td>47.4</td></tr></table>

Table 12. Cost of critic guidance. Inference per action chunk, with the ratio of the time to Base and the added memory in parentheses, and parameters of the guidance module. Training is per task. The critic, trained once per task and shared by all methods (about 1 h on SmolVLA and 1.5 h on π0.5), is not counted.
<table><tr><td rowspan="3">Method</td><td colspan="4">Inference, time (ms) and memory (MB)</td><td rowspan="3">Param.</td><td colspan="5">Training per task</td></tr><tr><td colspan="2">SmolVLA</td><td colspan="2">π0.5</td><td rowspan="2"></td><td colspan="2">SmolVLA</td><td colspan="2">π0.5</td></tr><tr><td>Time↓</td><td>Mem.↓</td><td>Time↓</td><td></td><td>s/step↓</td><td>h↓</td><td>GB↓</td><td></td></tr><tr><td>Base</td><td>448</td><td>1,338</td><td>228</td><td>9,169</td><td></td><td>1</td><td></td><td>1</td><td>1</td><td>一</td></tr><tr><td>Q-BoN (N = 8)</td><td>463 (1.03)</td><td>1,435 (+97)</td><td>250 (1.09)</td><td>9,638 (+469)</td><td>5.3M / 6.0M</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>QGF</td><td>587 (1.31)</td><td>1,370 (+32)</td><td>303 (1.33)</td><td>9,249 (+80)</td><td>5.3M / 6.0M</td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>QDPS</td><td>1,045 (2.33)</td><td>1,425 (+87)</td><td>633 (2.77)</td><td>9,249 (+80)</td><td>5.3M / 6.0M</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AGF</td><td>460 (1.03)</td><td>1,370 (+32)</td><td>240 (1.05)</td><td>9,219 (+49)</td><td>0.75M / 0.90M</td><td>5.06</td><td>7.0</td><td>15.1</td><td></td><td>official checkpoint</td></tr><tr><td>VaLoP (Ours)</td><td>2,048 (4.57)</td><td>1,498 (+160)</td><td>1,139 (4.99)</td><td>9,319 (+150)</td><td>5.3M/ 6.0M</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VaLoP-Fast (Ours)</td><td>460 (1.03)</td><td>1,370 (+32)</td><td>240 (1.05)</td><td>9,173 (+4)</td><td>0.75M / 0.90M</td><td>3.01</td><td>2.1–2.9</td><td>24.6</td><td>1.6</td><td>26.4</td></tr></table>

## E. Additional Results

## E.1. Statistical Analysis

Seeds. Tab. 13 lists the standard deviations of Tab. 1. On DrawBench, the target reward of either variant exceeds that of StitchVM and pixel-space DPS by more than one standard deviation in every setting.

Table 13. Means and standard deviations of Tab. 1 over three DrawBench seeds and four GenEval seeds (three for DPS).
<table><tr><td>Method</td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td><td>GenE.↑</td></tr><tr><td colspan="7">SD3.5-Medium</td></tr><tr><td>Unguided</td><td>0.286±0.001</td><td>5.347±0.024</td><td>28.80±0.09</td><td>22.65±0.06</td><td>0.987±0.032</td><td>0.694±0.017</td></tr><tr><td>HPSv2 reward</td><td>0.361±0.000</td><td>5.426±0.024</td><td>28.99±0.18</td><td>23.54±0.04</td><td>1.130±0.028</td><td>0.711±0.010</td></tr><tr><td>DPS</td><td>0.366±0.001</td><td>5.541±0.002</td><td>28.77±0.16</td><td>23.28±0.03</td><td>1.302±0.048</td><td>0.647±0.012</td></tr><tr><td>DPS + StitchVM</td><td>0.382±0.001</td><td>5.654±0.014</td><td>29.13±0.07</td><td>23.77±0.03</td><td>1.332±0.015</td><td>0.726±0.011</td></tr><tr><td>VaLoP (Ours)</td><td></td><td></td><td>28.92±0.06</td><td>23.72±0.01</td><td>1.333±0.024</td><td>0.702±0.017</td></tr><tr><td>VaLoP-Fast (Ours) Aesthetic reward</td><td>0.387±0.000</td><td>5.671±0.023</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>DPS</td><td>0.285±0.001</td><td>5.877±0.008</td><td>28.30±0.09</td><td>22.63±0.05</td><td>0.944±0.047</td><td>0.662±0.012</td></tr><tr><td>DPS + StitchVM</td><td>0.279±0.001</td><td>5.968±0.031</td><td>26.95±0.28</td><td>22.29±0.03</td><td>0.801±0.007</td><td>0.644±0.022</td></tr><tr><td>VaLoP (Ours)</td><td>0.288±0.001</td><td>6.174±0.010</td><td>27.22±0.27</td><td>22.56±0.01</td><td>0.905±0.007</td><td>0.672±0.016</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.293±0.001</td><td>6.035±0.033</td><td>27.62±0.17</td><td>22.69±0.01</td><td>0.974±0.017</td><td>0.672±0.007</td></tr><tr><td colspan="7">CLIP reward</td></tr><tr><td>DPS</td><td>0.254±0.003</td><td>4.955±0.034</td><td>34.27±0.07</td><td>21.70±0.06</td><td>0.622±0.072</td><td>0.633±0.023</td></tr><tr><td>DPS + StitchVM</td><td>0.248±0.002</td><td>4.877±0.032</td><td>34.42±0.09</td><td>21.66±0.07</td><td>0.688±0.065</td><td>0.625±0.006</td></tr><tr><td>VaLoP (Ours)</td><td>0.266±0.001</td><td>4.999±0.036</td><td>35.98±0.15</td><td>22.08±0.06</td><td>0.942±0.006</td><td>0.725±0.006</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.275±0.001</td><td>5.085±0.012</td><td>35.03±0.13</td><td>22.26±0.02</td><td>1.022±0.031</td><td>0.722±0.010</td></tr><tr><td colspan="7">SD3.5-Large</td></tr><tr><td>Unguided HPSv2 reward</td><td>0.294±0.001</td><td>5.461±0.029</td><td>28.61±0.15</td><td>22.87±0.04</td><td>1.065±0.019</td><td>0.721±0.016</td></tr><tr><td>DPS</td><td>0.349±0.000</td><td>5.519±0.026</td><td>28.82±0.16</td><td>23.55±0.05</td><td>1.111±0.034</td><td>0.715±0.017</td></tr><tr><td>DPS + StitchVM</td><td>0.360±0.001</td><td>5.544±0.010</td><td>28.65±0.05</td><td>23.20±0.03</td><td>1.248±0.019</td><td>0.662±0.014</td></tr><tr><td>VaLoP (Ours)</td><td>0.371±0.004</td><td>5.633±0.019</td><td>29.24±0.09</td><td>23.78±0.08</td><td>1.290±0.028</td><td>0.755±0.028</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.368±0.001</td><td>5.657±0.021</td><td>29.40±0.01</td><td>23.66±0.03</td><td>1.323±0.019</td><td>0.722±0.002</td></tr><tr><td colspan="7">Aesthetic reward</td></tr><tr><td>DPS</td><td>0.295±0.001</td><td>5.835±0.017</td><td>28.51±0.04</td><td>22.87±0.02</td><td>1.067±0.009</td><td>0.708±0.006</td></tr><tr><td>DPS + StitchVM</td><td>0.287±0.001</td><td>5.807±0.024</td><td>28.05±0.01</td><td>22.64±0.02</td><td>0.992±0.022</td><td>0.691±0.008</td></tr><tr><td>VaLoP (Ours)</td><td>0.297±0.002</td><td>5.989±0.019</td><td>28.15±0.08</td><td>22.88±0.02</td><td>1.058±0.020</td><td>0.712±0.016</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.296±0.000</td><td>6.144±0.027</td><td>28.06±0.18</td><td>22.80±0.03</td><td>1.066±0.017</td><td>0.706±0.011</td></tr><tr><td colspan="7">CLIP reward</td></tr><tr><td>DPS</td><td>0.286±0.001</td><td>5.321±0.022</td><td>32.13±0.03</td><td>22.60±0.01</td><td>0.982±0.061</td><td>0.713±0.013</td></tr><tr><td>DPS + StitchVM</td><td>0.275±0.001</td><td>5.183±0.023</td><td>32.43±0.08</td><td>22.40±0.03</td><td>0.983±0.030</td><td>0.724±0.007</td></tr><tr><td>VaLoP (Ours)</td><td>0.284±0.002</td><td>5.281±0.032</td><td>33.69±0.06</td><td>22.60±0.02</td><td>1.087±0.022</td><td>0.759±0.012</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.282±0.000</td><td>5.222±0.007</td><td>33.65±0.10</td><td>22.44±0.04</td><td>1.095±0.007</td><td>0.735±0.015</td></tr><tr><td colspan="7">FLUX.1-dev</td></tr><tr><td>Unguided</td><td>0.298±0.001</td><td>5.729±0.029</td><td>27.71±0.26</td><td>22.79±0.04</td><td>0.995±0.064</td><td>0.660±0.006</td></tr><tr><td colspan="7">HPSv2 reward</td></tr><tr><td>DPS</td><td>0.320±0.005</td><td>5.440±0.033</td><td>27.09±0.41</td><td>22.78±0.16</td><td>0.922±0.127</td><td>0.633±0.013</td></tr><tr><td>DPS + StitchVM</td><td>0.314±0.001</td><td>5.690±0.006</td><td>27.82±0.09</td><td>22.79±0.06</td><td>1.080±0.020</td><td>0.652±0.009</td></tr><tr><td>VaLoP (Ours)</td><td>0.328±0.001</td><td>5.590±0.023</td><td>28.36±0.31</td><td>23.12±0.02</td><td>1.190±0.030</td><td>0.701±0.011</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.333±0.001</td><td>5.782±0.028</td><td>28.28±0.09</td><td>23.11±0.02</td><td>1.198±0.039</td><td>0.686±0.008</td></tr><tr><td colspan="7">Aesthetic reward</td></tr><tr><td>DPS</td><td>0.297±0.002</td><td>5.815±0.026</td><td>27.61±0.20</td><td>22.78±0.06</td><td>0.991±0.046</td><td>0.641±0.014</td></tr><tr><td>DPS + StitchVM</td><td>0.297±0.001</td><td>5.893±0.019</td><td>27.37±0.23</td><td>22.74±0.07</td><td>0.967±0.077</td><td>0.651±0.012</td></tr><tr><td>VaLoP (Ours)</td><td>0.300±0.001</td><td>5.996±0.010</td><td>27.29±0.10</td><td>22.84±0.04</td><td>0.968±0.052</td><td>0.666±0.007</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.301±0.001</td><td>5.929±0.016</td><td>27.46±0.21</td><td>22.85±0.07</td><td>1.007±0.059</td><td>0.659±0.016</td></tr><tr><td colspan="7">CLIP reward</td></tr><tr><td>DPS</td><td>0.279±0.006</td><td>5.410±0.069</td><td>29.28±0.23</td><td>22.25±0.12</td><td>0.819±0.130</td><td>0.637±0.015</td></tr><tr><td>DPS + StitchVM</td><td>0.286±0.003</td><td>5.499±0.029</td><td>29.55±0.09</td><td>22.46±0.08</td><td>0.936±0.038</td><td>0.648±0.025</td></tr><tr><td>VaLoP (Ours)</td><td>0.291±0.001</td><td>5.514±0.027</td><td>30.79±0.02</td><td>22.65±0.05</td><td>1.097±0.027</td><td>0.691±0.024</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.295±0.001</td><td>5.545±0.039</td><td>30.37±0.18</td><td>22.67±0.02</td><td>1.082±0.044</td><td>0.695±0.012</td></tr></table>

Paired t-test. Tab. 14 lists every setting of the ablation in Sec. 5.4 with its paired t-statistic against VaLoP, and its last rows come from separate runs with the same seeds.  
Table 14. Ablation at $1 0 2 4 ^ { 2 }$ (SD3.5-Medium, DrawBench, HPSv2 reward, $\eta = 3 0 0 $ , one seed). Each row changes one setting of VaLoP, and parentheses give the paired t-statistic against VaLoP over the 200 images. Bold: $| t | \geq 2$ on the target reward.
<table><tr><td>Setting</td><td>Value</td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td></tr><tr><td>Unguided</td><td></td><td>0.287</td><td>5.370</td><td>28.71</td><td>22.69</td><td>0.995</td></tr><tr><td>VaLoP</td><td> $s / t = 0 . 6 , \beta = 1 , \gamma$  schedule</td><td>0.382</td><td>5.665</td><td>29.06</td><td>23.79</td><td>1.339</td></tr><tr><td>Ratio  $s / t$ </td><td>1 (StitchVM)</td><td>0.358 (−13.1)</td><td>5.505 (-4.8)</td><td>28.63 (-1.9)</td><td>23.28 (-9.6)</td><td>1.322 (−0.5)</td></tr><tr><td></td><td>0.8</td><td>0.385 (+2.1)</td><td>5.623 (-1.7)</td><td>29.36 (+1.6)</td><td>23.79 (−0.1)</td><td>1.386 (+1.8)</td></tr><tr><td></td><td>0.4</td><td>0.369 (−7.8)</td><td>5.599 (−2.3)</td><td>29.23 (+0.8)</td><td>23.60 (−4.1)</td><td>1.267 (−2.2)</td></tr><tr><td></td><td>0.2</td><td>0.359 (−11.9)</td><td>5.535 (-3.7)</td><td>29.27 (+0.8)</td><td>23.41 (−6.4)</td><td>1.217 (−2.9)</td></tr><tr><td></td><td>0 (DPS form)</td><td>0.352 (−15.3)</td><td>5.574 (−2.4)</td><td>28.84 (−0.9)</td><td>23.30 (−8.8)</td><td>1.193 (−4.4)</td></tr><tr><td>Correction</td><td> $\mathrm { n o n e } \left( g _ { s } \mathrm { o n l y } \right)$ </td><td>0.365 (−9.3)</td><td>5.614 (−1.7)</td><td>28.23 (−3.4)</td><td>23.52 (−5.8)</td><td>1.324 (−0.5)</td></tr><tr><td> $S - \beta A$ </td><td> $\beta = - 1 \left( J \right)$ </td><td>0.368 (−10.0)</td><td>5.598 (–2.2)</td><td>28.25 (-3.8)</td><td>23.59 (−4.8)</td><td>1.318 (−0.9)</td></tr><tr><td></td><td> $\beta = 0 \left( S \right)$ </td><td>0.379 (−2.0)</td><td>5.625 (−1.6)</td><td>29.42 (+1.8)</td><td>23.76 (-0.7)</td><td>1.334 (−0.2)</td></tr><tr><td></td><td> $\beta = 0 . 5$ </td><td>0.381 (−0.3)</td><td>5.636 (−1.1)</td><td>29.34 (+1.4)</td><td>23.80 (0.0)</td><td>1.346 (+0.2)</td></tr><tr><td></td><td> $\beta = 1 . 5$ </td><td>0.380 (−0.8)</td><td>5.674 (+0.5)</td><td>29.21 (+1.0)</td><td>23.72 (−1.7)</td><td>1.332 (−0.3)</td></tr><tr><td>Jacobian scale γ</td><td>1 at every step</td><td>0.382 (+0.2)</td><td>5.671 (+0.3)</td><td>28.85 (−1.1)</td><td>23.67 (−2.6)</td><td>1.341 (+0.1)</td></tr><tr><td></td><td>2 at every step</td><td>0.381 (−0.1)</td><td>5.615 (-1.7)</td><td>29.03 (−0.1)</td><td>23.75 (−0.8)</td><td>1.366 (+0.9)</td></tr><tr><td></td><td> $\omega = 4 . \ i$  5 at every step</td><td>0.368 (−7.0)</td><td>5.605 (-2.2)</td><td>28.83 (-1.0)</td><td>23.64 (-2.7)</td><td>1.299 (−1.4)</td></tr><tr><td>Other runs</td><td>StitchVM, η = 400</td><td>0.366</td><td>5.539</td><td>28.78</td><td>23.31</td><td>1.355</td></tr><tr><td></td><td>VaLoP-Fast,  $\eta = 3 0 0$ </td><td>0.378</td><td>5.660</td><td>28.99</td><td>23.70</td><td>1.342</td></tr><tr><td></td><td>VaLoP-Fast, η = 450</td><td>0.387</td><td>5.666</td><td>28.98</td><td>23.72</td><td>1.336</td></tr></table>

Robot policies. On LIBERO, all methods share the 1,500 episode seeds, so we compare each variant with the base policy by an exact McNemar test on the paired outcomes. VaLoP succeeds in 240 episodes where the base policy fails and fails in 166 where it succeeds $( p = 3 \times 1 0 ^ { - 4 } )$ , and VaLoP-Fast in 243 against 159 $( p = 3 \times 1 0 ^ { - 5 } )$ , with $p < 0 . 0 5$ on each suite. Against the two variants of Tab. 5, VaLoP succeeds more often in all six paired comparisons, with a significant difference only against $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone on LIBERO-Goal $( p = 0 . 0 3 )$

## E.2. Text-to-Image Ablations

Ablation at 10242. With S and A the symmetric and antisymmetric parts of $J _ { \gamma }$ , the corrections $S - \beta A$ range from the transpose $( \beta = 1 )$ to the Jacobian itself $( \beta = - 1 )$ in Tab. 14: the symmetric part alone $( \beta = 0 )$ loses a little HPSv2, and $\beta$ between 0.5 and 1.5 performs as the transpose. The head of VaLoP-Fast recovers most of the gain of the correction over $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone (last rows of Tab. 14), one head trained with the HPSv2 value serves the other rewards on par with per-reward heads (Tab. 17), and the fitted magnitude preserves HPSv2 without the second value-model backward pass (Tab. 18).

Jacobian scale. To choose $\gamma ,$ we compare the sampling scale $\gamma = \omega$ with $\gamma = 1$ and $\gamma = 2$ at every step (Tab. 14). At $\gamma = \omega$ , HPSv2 drops nearly to that of $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ alone, and PickScore and the aesthetic score also decrease, whereas $\gamma = 1$ and $\gamma = 2$ match the schedule in HPSv2, with $\gamma = 1$ lowering PickScore. The schedule thus matches $\gamma = 2$ at every step with only the conditional backward pass for $t < 0 . 8$

Role of CFG. Over eight states per noise level, CFG roughly doubles the asymmetry of the conditional velocity, and $\gamma = 2$ lies in between (Tab. 15). Along $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$ , the asymmetry grows with the noise level and exceeds 1 at $t = 0 . 9 \ \mathrm { f o r } \ \gamma = 2$ and $\gamma = \omega$ , so the transpose and the choice of $\gamma$ matter most at high noise, where we use $\gamma = 2$ . On FLUX.1-dev, which has no CFG, the lookahead without the correction, at twice the strength of StitchVM, raises HPSv2 and every non-target metric above StitchVM, and the correction adds little HPSv2 there and lowers the aesthetic score, while the head raises HPSv2, the aesthetic score, and ImageReward (Tab. 16).

Table 15. Asymmetry of the velocity Jacobian, $\Vert J u -$ $J ^ { \top } { \pmb u } \| / \| { \pmb J } { \pmb u } \|$ (SD3.5-Medium, 10242). $J _ { c } = J _ { \gamma = 1 }$
<table><tr><td rowspan="2">t</td><td colspan="3">Random u</td><td colspan="3"> ${ \pmb u } = { \pmb g } _ { s }$ </td></tr><tr><td> $J _ { \omega }$ </td><td> $J _ { \gamma = 2 }$ </td><td> $J _ { c }$ </td><td> $J _ { \omega }$ </td><td> $J _ { \gamma = 2 }$ </td><td> $J _ { c }$ </td></tr><tr><td>0.3</td><td>0.871</td><td>0.590</td><td>0.367</td><td></td><td>0.914 0.681</td><td>0.497</td></tr><tr><td></td><td>0.6 0.931</td><td>0.708</td><td>0.449</td><td>0.989</td><td></td><td>90.824 0.649</td></tr><tr><td></td><td>0.9 0.934</td><td>0.700</td><td> 0.433</td><td></td><td></td><td>1.194 1.031 0.786</td></tr></table>

Table 16. Lookahead and correction on FLUX.1-dev $( 1 0 2 4 ^ { 2 } ,$ Draw-Bench, HPSv2 reward, same seeds).
<table><tr><td>Method</td><td>η</td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td></tr><tr><td>StitchVM</td><td>400</td><td>0.314</td><td>5.688</td><td>27.72</td><td>22.79</td><td>1.092</td></tr><tr><td>Lookahead,  $\mathbf { \nabla } _ { \mathbf { { \boldsymbol { g } } } _ { s } }$  only</td><td>800</td><td>0.327</td><td>5.720</td><td>28.28</td><td>23.08</td><td>1.226</td></tr><tr><td>VaLoP</td><td>800</td><td>0.329</td><td>5.575</td><td>28.50</td><td>23.14</td><td>1.223</td></tr><tr><td>VaLoP-Fast</td><td>800</td><td>0.332</td><td>5.768</td><td>28.25</td><td>23.14</td><td>1.266</td></tr></table>

One head for all rewards. In Tab. 17, the head trained with the HPSv2 value, used for all rewards in Tab. 1, reaches a target reward similar to that of per-reward heads, with slightly better non-target metrics. Under both rewards, either head exceeds StitchVM on every metric, so one head per generator serves every reward.

Table 17. One head for all rewards (SD3.5-Medium, $1 0 2 4 ^ { 2 } )$ . Head trained with the HPSv2 value or per reward.
<table><tr><td>Method</td><td>Head</td><td>HPSv2↑ Aes.↑ CLIP↑ Pick↑</td><td></td><td></td><td></td><td>IR↑</td></tr><tr><td>CLIP reward</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StitchVM, η = 75</td><td></td><td>0.249</td><td>4.849</td><td>34.49</td><td></td><td>21.700.762</td></tr><tr><td>VaLoP-Fast, η = 37.5</td><td>own</td><td>0.274</td><td>5.092</td><td>34.95</td><td></td><td>22.24 0.987</td></tr><tr><td>VaLoP-Fast, η = 37.5</td><td>HPSv2</td><td>0.277</td><td>5.138</td><td>34.99</td><td>22.341</td><td>1.011</td></tr><tr><td>Aesthetic reward</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StitchVM, η = 12</td><td>1</td><td>0.279</td><td>5.970</td><td>27.09</td><td>22.260.799</td><td></td></tr><tr><td>VaLoP-Fast, η = 6</td><td>own</td><td>0.293</td><td>6.064</td><td>27.49</td><td></td><td>22.700.985</td></tr><tr><td>VaLoP-Fast, η = 6</td><td>HPSv2</td><td>0.294</td><td>6.022</td><td>27.84</td><td></td><td>22.701.016</td></tr></table>

Fitted magnitude. Replacing the computed magnitude of VaLoP-Fast with the fit $\hat { m } _ { t }$ of Sec. 4.4 keeps HPSv2, raises ImageReward, and removes the second value-model backward pass (Tab. 18). With the fit, VaLoP-Fast still exceeds StitchVM on every metric.

Table 18. Guidance magnitude of VaLoP-Fast (SD3.5-Medium, $5 1 2 ^ { 2 }$ , HPDv2 test prompts, HPSv2 reward). Backward passes through the generator / value model.
<table><tr><td>Method</td><td>η</td><td>Bwd.</td><td>HPSv2↑</td><td> $\mathrm { A e s . \uparrow }$ </td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td></tr><tr><td>DPS + StitchVM</td><td>400</td><td>0/1</td><td>0.417</td><td>5.939</td><td>28.64</td><td>23.56</td><td>1.408</td></tr><tr><td>VaLoP-Fast, computed</td><td>200</td><td>0/2</td><td>0.426</td><td>6.095</td><td>29.31</td><td>24.05</td><td>1.455</td></tr><tr><td>VaLoP-Fast, fitted</td><td>200</td><td>0/1</td><td>0.425</td><td>6.118</td><td>29.36</td><td>24.12</td><td>1.509</td></tr><tr><td>VaLoP</td><td>200</td><td>1/2</td><td>0.434</td><td>6.128</td><td>29.42</td><td>24.21</td><td>1.510</td></tr></table>

## E.3. Pick-a-Pic

On 499 Pick-a-Pic prompts [28] with one seed, both variants raise the target reward above StitchVM and pixel-space DPS in every setting, and VaLoP-Fast exceeds StitchVM on all but three held-out entries (Tab. 19).

Table 19. Reward guidance on Pick-a-Pic $( 1 0 2 4 ^ { 2 } .$ ,499 prompts, one seed). Columns and bold as in Tab. 1.
<table><tr><td></td><td colspan="5">SD3.5-Medium</td><td colspan="5">SD3.5-Large</td><td colspan="5">FLUX.1-dev</td></tr><tr><td>Method</td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td> $\mathrm { I R } \uparrow$ </td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td> $\mathrm { I R } \uparrow$ </td><td>HPSv2↑</td><td>Aes.↑</td><td>CLIP↑</td><td>Pick↑</td><td>IR↑</td></tr><tr><td>Unguided</td><td>0.302</td><td>5.840</td><td>27.59</td><td>22.17</td><td>1.087</td><td>0.308</td><td>5.940</td><td>27.77</td><td>22.31</td><td>1.130</td><td>0.304</td><td>6.158</td><td>26.70</td><td>22.23</td><td>1.031</td></tr><tr><td>HPSv2 reward</td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td></tr><tr><td>DPS</td><td>0.384</td><td>5.955</td><td>28.25</td><td>23.13</td><td>1.284</td><td>0.368</td><td>6.006</td><td>28.14</td><td></td><td>23.101.266</td><td>0.334</td><td>5.930</td><td>26.63</td><td>22.17</td><td>0.953</td></tr><tr><td>DPS + StitchVM</td><td>0.387</td><td>5.967</td><td>27.86</td><td>22.96</td><td>1.393</td><td>0.380</td><td>5.940</td><td>27.82</td><td>22.77</td><td>1.347</td><td>0.325</td><td>6.119</td><td>26.89</td><td>22.21</td><td>1.127</td></tr><tr><td>VaLoP (Ours)</td><td>0.407</td><td>6.011</td><td>28.42</td><td>23.34</td><td>1.471</td><td>0.393</td><td>6.065</td><td>28.52</td><td>23.33</td><td>1.437</td><td>0.346</td><td>6.102</td><td>27.44</td><td>22.60</td><td>1.268</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.408</td><td>6.014</td><td>28.22</td><td>23.20</td><td>1.426</td><td>0.385</td><td>6.050</td><td>28.62</td><td>23.03</td><td>1.388</td><td>0.348</td><td>6.261</td><td>27.42</td><td>22.63</td><td>1.319</td></tr><tr><td>Aesthetic reward</td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td><td></td></tr><tr><td>DPS</td><td>0.303</td><td>6.379</td><td>27.51</td><td>22.21</td><td>1.058</td><td>0.309</td><td>6.355</td><td>27.93</td><td>22.35</td><td>1.123</td><td>0.304</td><td>6.258</td><td>26.58</td><td>22.20</td><td>1.007</td></tr><tr><td>DPS + StitchVM</td><td>0.291</td><td>6.458</td><td>26.57</td><td>21.79</td><td>0.831</td><td>0.300</td><td>6.314</td><td>27.54</td><td>22.11</td><td>1.049</td><td>0.306</td><td>6.350</td><td>26.54</td><td>22.23</td><td>1.061</td></tr><tr><td>VaLoP (Ours)</td><td>0.305</td><td>6.877</td><td>26.28</td><td>22.14</td><td>1.005</td><td>0.311</td><td>6.606</td><td>27.58</td><td>22.42</td><td>1.132</td><td>0.308</td><td>6.497</td><td>26.46</td><td>22.27</td><td>1.017</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.311</td><td>6.651</td><td>27.10</td><td>22.27</td><td>1.142</td><td>0.311</td><td>6.727</td><td>27.56</td><td>22.32</td><td>1.157</td><td>0.309</td><td>6.409</td><td>26.61</td><td>22.28</td><td>1.047</td></tr><tr><td>CLIP reward</td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td><td> $\checkmark$ </td><td></td><td></td></tr><tr><td>DPS</td><td>0.269</td><td>5.395</td><td>34.71</td><td>21.17</td><td>0.738</td><td>0.298</td><td>5.821</td><td>31.70</td><td>21.99</td><td>0.982</td><td>0.289</td><td>5.888</td><td>28.60</td><td>21.72</td><td>0.838</td></tr><tr><td>DPS + StitchVM</td><td>0.258</td><td>5.370</td><td>34.73</td><td>21.11</td><td>0.686</td><td>0.285</td><td>5.714</td><td>31.84</td><td>21.80</td><td>0.962</td><td>0.296</td><td>5.998</td><td>29.44</td><td>22.00</td><td>1.019</td></tr><tr><td>VaLoP (Ours)</td><td>0.275</td><td>5.504</td><td>36.29</td><td>21.47</td><td>0.938</td><td>0.294</td><td>5.792</td><td>33.87</td><td>22.04</td><td>1.088</td><td>0.301</td><td>6.042</td><td>30.91</td><td>22.18</td><td>1.102</td></tr><tr><td>VaLoP-Fast (Ours)</td><td>0.284</td><td>5.596</td><td>35.35</td><td>21.65</td><td>0.984</td><td>0.290</td><td>5.703</td><td>33.83</td><td></td><td>21.78 1.067</td><td>0.305</td><td>6.046</td><td>30.56</td><td>22.19</td><td>1.136</td></tr></table>

## E.4. Robot Ablations

Lookahead ratio and value models. On LIBERO-Goal, $s = 0 . 6 t$ gives the highest success rate averaged over strengths (Tab. 21). The three value models of Tab. 20 are regressed onto the critic, as StitchVM regresses onto the reward: an MLP on forward-noised action chunks te + $( 1 - t ) \mathbf { } a _ { 0 } ,$ , a head stitched to the features of the policy's action expert, as in StitchVM, and an MLP on the states of the policy's own sampling path. Both guidance forms take the norm of $\nabla V _ { \psi } ( z _ { t } , t )$ , and the strength is the best of $w \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$

Table 20. Learned value on SmolVLA (LIBERO-Goal, success %, base 76.4), read at $\scriptstyle { z _ { t } }$ (StitchVM) or with the lookahead (VaLoP).
<table><tr><td>Value model</td><td>StitchVM</td><td> $\mathrm { { V a L o P } }$ </td></tr><tr><td>MLP, forward-noised actions</td><td>77.8</td><td>80.6</td></tr><tr><td>Stitched to the policy</td><td>77.4</td><td>79.6</td></tr><tr><td>MLP, policy sampling path</td><td>80.2</td><td>80.4</td></tr></table>

Table 21. Lookahead ratio of VaLoP on LIBERO-Goal. Success rate (%). Bold: best.
<table><tr><td> $s / t$ </td><td> $w = 1$ </td><td> $w = 2$ </td><td> $w = 4$ </td><td> $\operatorname { A v g } .$ </td></tr><tr><td>0.4</td><td>77.0</td><td>79.2</td><td>80.6</td><td>78.9</td></tr><tr><td>0.6</td><td>80.4</td><td>79.0</td><td>80.6</td><td>80.0</td></tr><tr><td>0.8</td><td>79.0</td><td>78.4</td><td>80.0</td><td>79.1</td></tr></table>

## E.5. Qualitative Results

Text-to-image generation. Figs. 5 to 7 compare the methods on selected GenEval prompts for each model, on which the unguided model and StitchVM miss an object, its count, its attribute, or its relation. Both variants follow these prompts while guiding toward the target reward. The prompts are selected to show visible differences between the methods.

Robot rollouts. Figs. 8 to 11 show eleven episodes from the same initial state for all methods, rendered from the recorded simulator states and selected to show visible differences between the methods. VaLoP-Fast succeeds in every episode and is the first to succeed in nine of them, while in the other two QDPS succeeds one and five steps earlier.

VaLoP (Ours)  
![](images/8287d94313be940b970227d7cf194370520f0c27815eed6a1ce8eb8b1dd50fb4.jpg)  
Figure 5. Qualitative comparison on SD3.5-Medium under the target reward of each row. √/X: GenEval correctness.

![](images/889e42c198dc6987d00ed71e3887f52bcfd6217266825e2127ffd6aea58bbeb3.jpg)  
Figure 6. Qualitative comparison on SD3.5-Large under the target reward of each row. √/X: GenEval correctness.

StitchVM  
VaLoP (Ours)  
VaLoP-Fast (Ours)  
![](images/c4c1861b9a82f4c7f6312a0e760aeb8d41b13b135d044bb2119a8b514e7c49dd.jpg)  
Figure 7. Qualitative comparison on FLUX.1-dev under the target reward of each row. √/X: GenEval correctness.

LIBERo-10“put the white mug on the left plate and put the yellow and white mug on the right plate"  
![](images/cbd526dc5834f8b20be495429901ac1a4f719483921a16517ca75fc445dac1c7.jpg)  
Figure 8. Rollouts on LIBERO (SmolVLA) from the same initial state, just before VaLoP-Fast succeeds and at the end of each episode, with bars for the steps to success or the time limit. Only VaLoP-Fast completes the first two tasks, and it is first on the third.

LIBERo-10“turn on the stove and put the moka pot on it"  
![](images/987fd9d66f239d7d703473be0d09f997f6a1b9c81efce41855aca781425786e2.jpg)  
Figure 9. Rollouts on LIBERO, continued. QDPS also completes both tasks, one and five steps earlier than VaLoP-Fast.

RoboCasa"Open the right drawer.'  
![](images/53dffda355d7d651c292706209630229844c946b9c92c02980226fc8aa452d89.jpg)  
Figure 10. Rollouts on RoboCasa (πo.5), shown as in Fig. 8. Only VaLoP-Fast opens the drawer and closes the toaster oven door, and only our two variants slide out the rack.

RoboCasa "Close the fridge doors."  
![](images/212b02b635d5a0ad3897ea377454964657ba35cfd06869084cee3937d66f435a.jpg)  
Figure 11. Rollouts on RoboCasa, continued. Several baselines also complete these tasks, but later than VaLoP-Fast.