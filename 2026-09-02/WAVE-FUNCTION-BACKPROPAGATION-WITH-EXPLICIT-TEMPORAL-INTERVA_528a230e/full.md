# WAVE FUNCTION BACKPROPAGATION WITH EXPLICIT TEMPORAL-INTERVAL DYNAMICS

Byunggu Yu, Justin Kim

Dept. of Computer Science and Information Technology University of the District of Columbia Washington, DC USA {byu, junwhan.kim}@udc.edu

## ABSTRACT

Conventional neural networks learn predominantly through affine transformations followed by nonlinear activations, while elapsed time is often treated as an auxiliary feature or assumed to be uniformly sampled. This paper introduces Wave Function Backpropagation (WFB), a wave-parameterized learning formulation in which neural responses are represented by learnable amplitude, wavenumber, angular frequency, and phase. The formulation associates an observed state with its temporal interval ∆t through the phase of a differentiable spatiotemporal wave. We derive standard WFB gradients and a spatial-curvature correction based on the Laplacian of the wave response. WFB is instantiated in a deliberately feed-forward trajectory predictor to provide a controlled proof of concept; sequence learning is outside the scope of the present evaluation. With motion features, STD-WFB using real intervals reduces average displacement error (ADE) by 20.4% relative to the original FFN baseline. In a new position-only evaluation that removes temporal leakage through precomputed velocity and acceleration, real-interval WFB reduces ADE by 10.4% relative to the original FFN and remains competitive with parameter-matched ReLU controls, obtaining 2.1% lower mean ADE than the matched FFN with explicit ∆t. Shuffled-interval WFB attains the lowest mean ADE, indicating that the present evidence supports the effectiveness of the wave representation but does not attribute the gain to interval alignment. These results establish WFB as a viable structured feed-forward learning formulation and define a clear basis for subsequent architectural studies.

Keywords Wave Function Backpropagation, wave-based neural networks, explicit temporal intervals, curvature regularization, trajectory prediction

## 1 Introduction

Modern neural networks are largely built from affine transformations of the form Wx + b followed by nonlinear activation functions [1]. By composing these operations, deep networks can approximate complex input–output mappings and have achieved strong performance across vision, language, and sequential prediction. Nevertheless, the standard formulation does not provide a dedicated mechanism for representing elapsed time. In many applications, time is represented only by an observation index, appended as an ordinary feature, or absorbed indirectly into changes between consecutive states. These choices can be adequate under uniform sampling, but they provide no explicit parameterization of how a neural response should evolve when observation intervals vary.

Irregular temporal intervals arise in many settings, including asynchronous sensing, missed observations, event-based measurements, healthcare records, and physical motion. Existing approaches address this problem through timedependent decay, continuous-time latent dynamics, or controlled differential equations [2, 3, 4]. These methods have substantially advanced irregular time-series modeling. However, they do not define a general wave-parameterized learning rule in which an elapsed interval directly changes the phase of a learnable neural response.

This paper introduces Wave Function Backpropagation (WFB), a learning formulation that represents a neural component using amplitude, wavenumber, angular frequency, and phase. For spatial input x and temporal interval ∆t, the response is modeled through a phase term $k x - \omega \Delta t - \theta .$ The interval therefore affects the activation through a structured phase displacement rather than only through feature concatenation. WFB remains trainable by gradient descent because all wave parameters are differentiable with respect to a task loss.

The purpose of this work is to establish the mathematical formulation and an initial empirical proof of concept for WFB. Trajectory prediction is used as the validation task rather than as the scope of the proposed framework. It is particularly suitable for this initial study because position changes and elapsed time have a direct physical relationship, and irregular frame gaps provide observable ∆t values. To isolate the effect of the proposed wave representation, we intentionally use a feed-forward network (FFN) rather than introducing recurrent memory, attention, or a task-specific interaction module. This controlled setting helps distinguish gains due to WFB from gains due to a stronger sequence architecture.

The contributions of this work are as follows:

• We formulate WFB as a differentiable wave-parameterized learning mechanism in which spatial input and explicit temporal intervals interact through learnable amplitude, wavenumber, angular frequency, and phase.

• We derive standard task-gradient updates and a Laplacian-based curvature regularizer for the wave parameters, yielding standard, Laplacian-only, and combined standard–Laplacian variants.

• We provide a controlled feed-forward validation with real, shuffled, and constant ∆t, position-only temporal controls, and parameter-matched ReLU baselines. These experiments separate the effectiveness of the complete WFB representation from model capacity and from temporal information embedded in engineered motion features.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 develops the wave representation. Section 4 derives WFB and its curvature-corrected variants. Section 5 presents the feed-forward proof-of-concept evaluation and discussion. Section 6 concludes the paper and identifies directions for future work.

## 2 Related Work

## 2.1 Temporal Learning and Irregular Sampling

Recurrent neural networks and Backpropagation Through Time model ordered observations through recursive hiddenstate updates, but the recurrence index does not itself represent elapsed physical time. Time-aware extensions therefore modify hidden-state decay or gating using observation gaps. GRU-D, for example, uses trainable decay mechanisms to handle missing observations and irregular intervals [2]. Continuous-time approaches instead model latent dynamics between observations. Latent ODEs use ordinary differential equations to evolve hidden states over continuous time [3], while Neural Controlled Differential Equations represent irregular sequences as continuous paths that drive latent dynamics [4]. These methods focus on state evolution or interpolation. WFB is complementary: it investigates whether elapsed time can directly parameterize the phase of a learnable wave response.

Backpropagation Through Time, adjoint-based optimization, and biologically inspired continuous-time learning methods such as Generalized Latent Equilibrium address temporal credit assignment in recurrent or dynamical systems [5]. Physics-Informed Neural Networks incorporate temporal variables through governing equations and residual constraints [6]. Spiking Neural Networks encode information through temporally structured spike events and are trained using methods such as surrogate gradients and spatio-temporal backpropagation [7, 8]. In contrast to these approaches, WFB defines the activation itself as a differentiable wave and optimizes its field parameters directly from a task loss.

## 2.2 Periodic and Wave-Based Representations

Fourier features and sinusoidal implicit representations have demonstrated that periodic bases can represent highfrequency or continuously varying signals effectively [9, 10]. The present work differs in emphasis from fixed positional encodings or a generic sine activation: amplitude, spatial frequency, temporal frequency, and phase are exposed as learnable parameters of the same response, and an observed interval directly enters its phase. A Laplacian-derived term is additionally studied as a selected-parameter curvature correction on the learned wave field.

## 2.3 Trajectory Prediction as a Validation Task

Trajectory prediction estimates future coordinates from an observed history of agent positions. Social-LSTM introduced social pooling for pedestrian interactions, and Social-GAN modeled multimodal socially acceptable futures [11, 12]. More recent trajectory models emphasize interaction modeling, multimodality, attention, and motion priors [13, 14, 15,

16, 17]. These task-specific architectures are not the focus of the present study. Instead, trajectory prediction provides a physically interpretable testbed for evaluating whether a wave-parameterized activation can learn from spatial states and observed temporal intervals under a deliberately controlled FFN architecture.

## 3 Wave-Parameterized Representation

## 3.1 From Static Activations to Spatiotemporal Waves

A conventional single-hidden-layer approximation can be written as

$$
F ( x ) = \sum _ { n = 1 } ^ { N } a _ { n } \sigma ( w _ { n } x + b _ { n } ) ,\tag{1}
$$

where each component applies an affine transformation followed by an activation function. WFB instead represents each component as a parameterized harmonic response. For a scalar spatial input x and elapsed temporal interval $\Delta t .$ the complex wave is

$$
\begin{array} { c } { { \psi _ { n } ( x , \Delta t ) = A _ { n } e ^ { i \phi _ { n } } , } } \\ { { \phi _ { n } = k _ { n } x - \omega _ { n } \Delta t - \theta _ { n } , } } \end{array}\tag{2}
$$

where $A _ { n }$ is amplitude, $k _ { n }$ is wavenumber, $\omega _ { n }$ is angular frequency, and $\theta _ { n }$ is phase. The real-valued response used by the network is

$$
f _ { n } ( x , \Delta t ) = A _ { n } \cos ( \phi _ { n } ) .\tag{3}
$$

A wave layer with N components produces

$$
F ( x , \Delta t ) = \sum _ { n = 1 } ^ { N } f _ { n } ( x , \Delta t ) .\tag{4}
$$

For vector inputs, $k _ { n } x$ is replaced by an inner product $\mathbf { k } _ { n } ^ { \top } \mathbf { x } .$ . Thus,

$$
f _ { n } ( \mathbf { x } , \Delta t ) = A _ { n } \cos \left( \mathbf { k } _ { n } ^ { \top } \mathbf { x } - \omega _ { n } \Delta t - \theta _ { n } \right) .\tag{5}
$$

Equation (5) is the practical formulation used in a neural layer. It directly associates the state x with the elapsed interval $\Delta t$ through phase. Importantly, the temporal interval is not claimed to encode sequence order by itself; it represents elapsed time associated with the current observation. Learning ordered temporal dependencies requires an additional propagation mechanism, which is left for future recurrent or attention-based WFB architectures.

## 3.2 Separable Spatial and Temporal Amplitudes

The implementation evaluated in this paper uses a separable amplitude

$$
A _ { n } = A _ { x n } A _ { t n } ,\tag{6}
$$

which allows spatial and temporal contributions to be optimized separately. The component response becomes

$$
\begin{array} { c } { { f _ { n } ( x , \Delta t ) = A _ { x n } A _ { t n } \cos ( \phi _ { n } ) , } } \\ { { \phi _ { n } = k _ { n } x - \omega _ { n } \Delta t - \theta _ { n } . } } \end{array}\tag{7}
$$

This factorization is an architectural choice rather than a requirement of WFB. A single amplitude or vector-valued amplitudes may be used in other implementations.

## 3.3 Learnable Parameters

Unlike a conventional layer that primarily optimizes weights and biases, a WFB component optimizes the field parameters

$$
\Theta _ { n } = \{ A _ { x n } , A _ { t n } , k _ { n } , \omega _ { n } , \theta _ { n } \} .\tag{8}
$$

The amplitude determines response magnitude, $k _ { n }$ determines sensitivity to the spatial input, $\omega _ { n }$ determines phase change per unit elapsed time, and $\theta _ { n }$ determines phase offset. These parameters are learned jointly with the remaining network parameters using the task objective.

## 4 Wave Function Backpropagation

Let L denote the task loss and define the upstream derivative

$$
\delta _ { n } = { \frac { \partial { \mathcal { L } } } { \partial f _ { n } } } .\tag{9}
$$

For clarity, the following derivation uses the scalar form in Eq. (7); the vector form follows by replacing x with x and $k _ { n }$ with $\mathbf { k } _ { n }$

## 4.1 Standard WFB Gradients

The partial derivatives of the wave response are

$$
\frac { \partial f _ { n } } { \partial A _ { x n } } = A _ { t n } \cos ( \phi _ { n } ) ,\tag{10}
$$

$$
\frac { \partial f _ { n } } { \partial A _ { t n } } = A _ { x n } \cos ( \phi _ { n } ) ,\tag{11}
$$

$$
\frac { \partial f _ { n } } { \partial k _ { n } } = - A _ { x n } A _ { t n } x \sin ( \phi _ { n } ) ,\tag{12}
$$

$$
\frac { \partial f _ { n } } { \partial \omega _ { n } } = A _ { x n } A _ { t n } \Delta t \sin ( \phi _ { n } ) ,\tag{13}
$$

$$
\frac { \partial f _ { n } } { \partial \theta _ { n } } = A _ { x n } A _ { t n } \sin ( \phi _ { n } ) .\tag{14}
$$

Applying the chain rule gives

$$
\frac { \partial \mathcal { L } } { \partial A _ { x n } } = \delta _ { n } A _ { t n } \cos ( \phi _ { n } ) ,\tag{15}
$$

$$
\frac { \partial \mathcal { L } } { \partial A _ { t n } } = \delta _ { n } A _ { x n } \cos ( \phi _ { n } ) ,\tag{16}
$$

$$
\frac { \partial \mathcal { L } } { \partial k _ { n } } = - \delta _ { n } A _ { x n } A _ { t n } x \sin ( \phi _ { n } ) ,\tag{17}
$$

$$
\frac { \partial \mathcal { L } } { \partial \omega _ { n } } = \delta _ { n } A _ { x n } A _ { t n } \Delta t \sin ( \phi _ { n } ) ,\tag{18}
$$

$$
\frac { \partial \mathcal { L } } { \partial \theta _ { n } } = \delta _ { n } A _ { x n } A _ { t n } \sin ( \phi _ { n } ) .\tag{19}
$$

These equations define standard WFB. The $\omega _ { n }$ gradient is explicitly scaled by $\Delta t ,$ causing observations separated by different elapsed intervals to produce different temporal-frequency updates.

## 4.2 Laplacian Curvature Regularization

To study whether curvature control stabilizes the learned wave field, we introduce a penalty based on the spatial Laplacian. In one spatial dimension,

$$
\frac { \partial f _ { n } } { \partial x } = - k _ { n } A _ { x n } A _ { t n } \sin ( \phi _ { n } ) ,\tag{20}
$$

$$
\Delta _ { x } f _ { n } = \frac { \partial ^ { 2 } f _ { n } } { \partial x ^ { 2 } } = - k _ { n } ^ { 2 } A _ { x n } A _ { t n } \cos ( \phi _ { n } ) = - k _ { n } ^ { 2 } f _ { n } .\tag{21}
$$

The curvature penalty is

$$
\begin{array} { c } { \displaystyle { \mathcal { P } = \frac { \lambda } { 2 } \sum _ { n = 1 } ^ { N } ( \Delta _ { x } f _ { n } ) ^ { 2 } } } \\ { \displaystyle { = \frac { \lambda } { 2 } \sum _ { n = 1 } ^ { N } k _ { n } ^ { 4 } f _ { n } ^ { 2 } , } } \end{array}\tag{22}
$$

where $\lambda \geq 0$ controls regularization strength. For parameters $q \in \{ A _ { t n } , \omega _ { n } , \theta _ { n } \}$ , while treating $k _ { n }$ as fixed in this correction branch,

$$
\frac { \partial \mathcal { P } } { \partial q } = \lambda k _ { n } ^ { 4 } f _ { n } \frac { \partial f _ { n } } { \partial q } .\tag{23}
$$

Substitution into Eq. (23) gives the three correction components:

$$
\frac { \partial \mathcal { P } } { \partial A _ { t n } } = \lambda k _ { n } ^ { 4 } A _ { x n } ^ { 2 } A _ { t n } \cos ^ { 2 } ( \phi _ { n } ) ,\tag{24}
$$

$$
\frac { \partial \mathcal { P } } { \partial \omega _ { n } } = \lambda k _ { n } ^ { 4 } \Delta t A _ { x n } ^ { 2 } A _ { t n } ^ { 2 } \cos ( \phi _ { n } ) \sin ( \phi _ { n } ) ,\tag{25}
$$

$$
\frac { \partial \mathcal { P } } { \partial \theta _ { n } } = \lambda k _ { n } ^ { 4 } A _ { x n } ^ { 2 } A _ { t n } ^ { 2 } \cos ( \phi _ { n } ) \sin ( \phi _ { n } ) .\tag{26}
$$

Because Eq. (22) is derived from spatial curvature, it should not be interpreted as a temporal Laplacian. Its effect on $\omega _ { n }$ and $\theta _ { n }$ arises because these parameters also control the same spatiotemporal phase. The $k _ { n } ^ { \dot { 4 } }$ factor increasingly penalizes high-spatial-frequency responses.

## 4.3 Combined Standard–Laplacian WFB

The implementation treats the Laplacian term as a custom correction for the selected temporal parameters $\mathcal { Q } _ { t } ~ =$ $\{ A _ { t n } , \bar { \omega _ { n } } , \theta _ { n } \}$ rather than as a global loss differentiated through every network parameter. For $q \in \mathcal { Q } _ { t }$ , the two correction modes are

$$
g _ { q } ^ { \mathrm { l a p } } = \frac { \partial \mathcal { P } } { \partial q } ,\tag{27}
$$

$$
g _ { q } ^ { \mathrm { c o m b } } = \frac { \partial \mathcal { L } } { \partial q } + \frac { \partial \mathcal { P } } { \partial q } .\tag{28}
$$

The wavenumber is held fixed in this correction branch, and all parameters outside $\mathcal { Q } _ { t }$ continue to receive their supervised task gradients. Standard WFB uses only $\partial { \cal C } / \partial q ;$ Laplacian-only WFB replaces the supervised gradients of the selected temporal parameters with $g _ { q } ^ { \mathrm { l a p } }$ ; and combined standard–Laplacian WFB uses $g _ { \boldsymbol { q } } ^ { \mathrm { c o m b } }$ . This formulation matches the implemented update and avoids interpreting the selected-parameter correction as optimization of a global objective $\mathcal { L } + \bar { \mathcal { P } }$

## 5 Proof-of-Concept Evaluation

## 5.1 Evaluation Scope

This proof-of-concept evaluation addresses three questions. First, does inserting a wave-parameterized representation into a feed-forward trajectory predictor improve accuracy relative to conventional FFNs? Second, does WFB remain competitive when temporal leakage through engineered velocity and acceleration is removed and model capacity is controlled? Third, how do the standard, Laplacian-only, and combined standard–Laplacian update rules behave as the correction weight λ changes? All experiments use feed-forward predictors to isolate the WFB representation. Sequence learning, recurrent state propagation, attention, and comparison with specialized trajectory-prediction systems are intentionally outside the scope of this study.

## 5.2 Data Preparation and Temporal Intervals

We use ETH/UCY and JAAD-style pedestrian and agent annotations [18, 19]. Frame-level detections are associated across frames to form pseudo-trajectories. To prevent overlapping windows from the same agent from appearing in different partitions, the data are split by the pair (source, agent\_id) before normalization and window extraction. The resulting irregularly sampled set contains 425,090 frame-level observations from 14,703 trajectories and 453,338 windows: 322,160 for training, 61,794 for validation, and 69,384 for testing.

For observation i, the elapsed time is computed before normalization as

$$
\Delta t _ { i } = \frac { F _ { i } - F _ { i - 1 } } { \mathrm { F P S } } ,\tag{29}
$$

where $F _ { i }$ and $F _ { i - 1 }$ are consecutive available frame indices for the same trajectory and $\mathrm { F P S } = 2 . 5$ . Thus, missed detections and nonuniform frame gaps remain visible to the model. The extracted intervals have mean 0.842, standard

deviation 0.551, and range [0.398, 10.801] seconds. Both the motion features and $\Delta t$ are standardized using statistics estimated from the training partition only.

Each observation is represented by

$$
\mathbf { x } _ { i } = [ x _ { i } , y _ { i } , v _ { i } ^ { x } , v _ { i } ^ { y } , a _ { i } ^ { x } , a _ { i } ^ { y } ] ,\tag{30}
$$

where $( x _ { i } , y _ { i } )$ is the normalized bounding-box center and

$$
v _ { i } ^ { x } = \frac { x _ { i } - x _ { i - 1 } } { \Delta t _ { i } } ,
$$

$$
v _ { i } ^ { y } = \frac { y _ { i } - y _ { i - 1 } } { \Delta t _ { i } } ,\tag{31}
$$

$$
a _ { i } ^ { x } = \frac { v _ { i } ^ { x } - v _ { i - 1 } ^ { x } } { \Delta t _ { i } } ,
$$

$$
a _ { i } ^ { y } = \frac { v _ { i } ^ { y } - v _ { i - 1 } ^ { y } } { \Delta t _ { i } } .\tag{32}
$$

An input window contains $T _ { \mathrm { o b s } } = 8$ observations and the target contains the next $T _ { \mathrm { p r e d } } = 1 2$ center coordinates. Since the source annotations provide normalized image coordinates rather than world coordinates, ADE, FDE, MSE, and RMSE are reported in normalized coordinate units, not meters.

## 5.3 Models and Training Protocol

The original FFN baseline flattens the $8 \times 6$ motion-feature matrix and maps it directly to $1 2 \times 2$ future coordinates. It does not receive ∆t as a separate variable. WFB-FFN first projects each observation to a 128-dimensional hidden state and applies the wave response in Eq. (5); the resulting eight wave states are flattened and passed to the same type of feed-forward prediction head. The WFB block therefore provides explicit amplitude, wavenumber, angular-frequency, and phase parameters, while $\Delta t _ { i }$ modulates the phase at each observed position.

The new capacity-controlled experiment uses only position $( x , y )$ as the spatial input, preventing real-∆t information from entering through precomputed velocity or acceleration. It compares the original FFN, an FFN with explicit $\Delta t .$ parameter-matched versions of both FFNs, and WFB with real, shuffled, or constant intervals. The original position-only FFN has 142,104 trainable parameters; the matched FFN, matched FFN with $\Delta t ,$ and WFB have 400,753, 402,452, and 401,560 parameters, respectively. A parameter-matched sinusoidal FFN was also run as an optimization control, but its shared hyperparameter setting was unstable; it is therefore excluded from the efficacy comparison.

We evaluate three WFB update rules. STD-WFB-FFN uses the task-loss gradient for all parameters. Laplacian-WFB-FFN replaces the standard gradients of the temporal parameters $( A _ { t } , \omega , \theta _ { t } )$ with the curvature-derived gradients scaled by λ. STD-Laplacian-WFB-FFN instead adds the scaled curvature-derived gradients to their standard task gradients. Thus, λ weights a temporal gradient correction; it is not an additional term in the reported MSE objective.

All models are optimized with MSE loss and AdamW using a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 }$ , batch size 512, gradient clipping at 1.0, and a maximum of 100 epochs. Early stopping uses validation ADE with patience 15. The principal comparisons use the same five seeds. Results are reported as mean ± standard deviation; with only five runs, small differences are interpreted descriptively rather than as definitive statistical superiority.

For a predicted trajectory $\hat { \bf p } _ { 1 : T _ { \mathrm { p r e d } } }$ and ground truth $\mathbf { p } _ { 1 : T _ { \mathrm { p r e d } } }$ , the principal metrics are

$$
\mathrm { A D E } = \frac { 1 } { T _ { \mathrm { p r e d } } } \sum _ { j = 1 } ^ { T _ { \mathrm { p r e d } } } \left. \hat { \mathbf { p } } _ { j } - \mathbf { p } _ { j } \right. _ { 2 } ,\tag{33}
$$

$$
\mathrm { F D E } = \left. \hat { \mathbf { p } } _ { T _ { \mathrm { p r e d } } } - \mathbf { p } _ { T _ { \mathrm { p r e d } } } \right. _ { 2 } .\tag{34}
$$

We additionally report coordinate-wise MSE and its square root (RMSE). Lower values are better for all metrics.

Unless otherwise stated, all experiments use the same ReLU-based MLP trajectory decoder to ensure a consistent comparison across WFB variants and the FFN baseline. The linear decoder introduced in Section 5.8 is evaluated only as an architectural ablation to investigate whether the nonlinear wave representation can reduce the need for an additional nonlinear decoding network.

## 5.4 Trajectory Prediction and Temporal Ablation

Figure 1 shows that STD-WFB-FFN with real intervals reduces mean ADE by 20.4% relative to the original FFN baseline; FDE, MSE, and RMSE improve in the same direction. The constant-∆t model still reduces ADE by 12.2%, showing that a substantial part of the gain is associated with the complete WFB parameterization rather than temporal variation alone. Because the WFB block adds a projection and wave parameters, this comparison does not by itself isolate the wave representation from increased model capacity; the position-only experiment below addresses that question directly.

![](images/8d67c5c76425454640f9d2d19ce37d1590e959ec87136dd004afd32405578802.jpg)  
Figure 1: Motion-feature trajectory prediction under temporal-input ablations. Points show mean ADE and error bars show standard deviation over five seeds. Lower is better.

![](images/2d4ced7ab0488211be8d8b868945326894adae7118cefe2f5febbadf1432bcbc.jpg)  
Figure 2: Position-only capacity-controlled comparison. Points show mean ADE and error bars show standard deviation over five seeds. The matched ReLU controls have approximately the same number of trainable parameters as WFB.

The shuffled-∆t condition attains the lowest mean error, improving ADE by 21.7% relative to FFN and slightly outperforming real ∆t. Shuffling preserves the interval distribution while breaking its observation-level alignment. Accordingly, this feed-forward proof of concept supports the effectiveness of the WFB representation but does not attribute its gain to correct interval alignment, which is not a claim tested in the present study.

## 5.5 Position-Only Capacity-Controlled Evaluation

Figure 2 reports the new evaluation using only observed positions as spatial inputs. Real-interval WFB obtains an ADE of $0 . { \dot { 0 1 } } 1 7 8 7 \pm 0 . 0 0 1 1 2 3$ , improving on the original position-only FFN by 10.4%. More importantly, it remains competitive after controlling capacity: its mean ADE is 2.1% lower than the parameter-matched FFN with explicit ∆t $( 0 . 0 \bar { 1 } 2 0 4 3 \pm 0 . 0 0 0 5 1 8 )$ and 3.3% lower than the parameter-matched FFN without $\Delta t \left( 0 . 0 1 2 1 9 4 \pm 0 . 0 0 0 4 1 6 \right)$ . These differences are modest relative to five-seed variability, so the result is evidence of competitive effectiveness rather than definitive superiority.

Shuffled-interval WFB achieves the lowest mean ADE, $0 . 0 1 1 4 1 8 \pm 0 . 0 0 0 2 7 4$ , whereas constant-interval WFB obtains $0 . 0 1 2 3 7 4 \pm 0 . 0 0 0 6 4 4$ . The four-model bar chart in Figure 3 highlights the comparison among the original FFN, the strongest matched FFN control with explicit $\Delta t ,$ and WFB with real or shuffled intervals. Together, these results support two bounded conclusions: WFB is an effective feed-forward representation under both motion-feature and position-only inputs, and correct interval alignment is not established as the source of the improvement in this experiment.

![](images/067c43fe2cd5df744c009ae9c280e2adf14fce308a88a0148980d8568991deba.jpg)  
Figure 3: Position-only ADE for four principal feed-forward controls. Bars show the five-seed mean and error bars show one standard deviation. Lower is better.

![](images/6be22e7ef85073a05e9ccc6ad5dbd185c22616e8b060b4a16928cf3ed4df11e0.jpg)  
Figure 4: ADE as a function of the Laplacian gradient weight λ under real $\Delta t .$ Error bars show standard deviation across runs. The ADE axis is logarithmic because Laplacian-only WFB is more than an order of magnitude worse than the supervised variants

## 5.6 Backpropagation Rule and Laplacian Weight

Figure 4 separates the effect of the update rule from the effect of λ. STD-WFB-FFN obtains an ADE of 0.011739 ± 0.000640. The best combined setting, STD-Laplacian-WFB-FFN with $\lambda = 1 0 ^ { - 5 }$ , obtains $0 . 0 1 1 5 1 3 \pm 0 . 0 0 0 0 3 9 ,$ a modest 1.9% reduction. Increasing λ to $1 0 ^ { - 3 } , \dot { 1 } 0 ^ { - 1 }$ , and $1 0 ^ { 0 }$ increases ADE to 0.017112, 0.039341, and 0.050845, respectively. The correction is therefore useful only when it remains weak relative to the supervised gradient; the results do not support the claim that a larger Laplacian contribution is better.

Laplacian-only WFB performs poorly for every tested weight, with ADE between 0.192761 and 0.229382. Its temporal update is determined by local wave curvature rather than by the direction that minimizes trajectory displacement. Applied alone, this update can suppress or redirect the temporal parameters without providing sufficient task-level credit assignment. The combined rule avoids this failure because it retains the supervised gradient and uses curvature only as a small correction. The resulting 1.9% improvement over standard WFB indicates that curvature information is most effective as a complementary learning signal.

![](images/4b42098113aa820241cd27b45e958d2bc6488ede1e08175262e812c19197fe44.jpg)  
Figure 5: Distributions of the learned effective amplitude $A = A _ { x } A _ { t }$ , wavenumber k, angular frequency ω, and phase $\theta = \theta _ { x } + \theta _ { t }$ for the analyzed STD-Laplacian-WFB-FFN checkpoint.

## 5.7 Interpretability Analysis

Figure 5 shows that the trained model does not collapse all wave dimensions to a common value. Across the 128 wave channels, the effective amplitude is sparse and right-skewed $( 0 . 0 0 1 0 \pm 0 . 0 0 2 4 )$ , whereas k and $\omega$ exhibit broader channel-dependent distributions $( - 1 . 5 7 3 \bar { 9 } \pm 0 . 2 1 3 8 \bar { \mathrm { ~ a n d ~ } } - 1 . 0 1 9 6 \pm 0 . 2 4 4 5$ , respectively). The phase distribution spans both signs $( 0 . 0 2 0 3 \pm 0 . 3 6 0 2 )$ . These statistics establish parameter diversity, but they should not be interpreted as physical frequencies or wavelengths because the inputs and hidden states are normalized and the signs of the learned parameters are unconstrained.

To examine whether this diversity is related to trajectory structure, we compute Spearman correlations between activation-weighted wave parameters and sample-level motion descriptors. The strongest associations are with mean speed: $\rho = - 0 . 3 2 2$ for $A , - 0 . 2 8 3$ for $k , - 0 . 2 9 7$ for $\omega ,$ and 0.300 for θ. Similar, moderate associations occur for path length, displacement, and mean acceleration. In contrast, correlations with the mean, standard deviation, and maximum of $\bar { \Delta } t$ are mostly weak; the largest is $\rho = - 0 . 1 9 7$ between θ and mean $\Delta t .$ . Thus, the learned wave representation is measurably associated with motion regime, while direct encoding of interval statistics is comparatively limited. Consistently, activation-weighted $| \omega |$ increases from approximately 0.639 for slow motion to 0.678 for fast motion, although the distributions overlap substantially.

The intervention in Figure 6 complements the training ablation by changing only $\Delta t$ at inference. Mean ADE is 0.011210 with real intervals, 0.012017 with constant intervals, 0.012198 with reversed intervals, 0.011434 with intervals scaled by 0.5, and 0.014672 with intervals scaled by 2.0. Relative to real $\Delta t ,$ , these changes correspond to increases of 7.2%, 8.8%, 2.0%, and 30.9%, respectively. The fixed model is therefore sensitive to temporal-interval values, especially to large changes in scale. However, sensitivity alone does not prove that the model has learned the correct interval ordering, which remains consistent with the real-versus-shuffled training result.

![](images/1f1161183fad1b010557106c0ea020bdea46efafbfbcfa5405aea63db1be35f9.jpg)  
Figure 6: Test-sample ADE after intervening on ∆t while holding the trained model and spatial/motion inputs fixed. The analysis uses 5,120 samples from one STD-Laplacian-WFB-FFN checkpoint.

![](images/966bc7d9fa5c27c21b27cc6201115dfcab6df4a26fbb6905a062fa163f0fe788.jpg)  
Figure 7: Accuracy–parameter trade-off in the decoder ablation. ADE is plotted against the number of trainable parameters on a logarithmic scale; lower ADE and fewer parameters are better.

## 5.8 Decoder Ablation and Parameter Efficiency

A conventional multilayer FFN requires nonlinear activations such as ReLU between affine layers; otherwise, multiple affine transformations collapse into a single affine mapping. In contrast, STD-WFB introduces nonlinearity directly through the phase-dependent wave response $A _ { x } A _ { t } \cos ( k z - \omega \Delta t - \theta )$ . We therefore evaluate whether the ReLU-based MLP trajectory decoder can be removed and replaced with a single linear output layer without degrading prediction accuracy.

As shown in Figure 7, removing the ReLU-MLP decoder does not reduce the predictive accuracy of STD-WFB. Instead, the linear-decoder STD-WFB achieves the lowest ADE in this ablation, reducing ADE by 25.7% relative to the FFN baseline and by 19.2% relative to STD-WFB with the ReLU-MLP decoder. This result suggests that the nonlinear wave representation already captures much of the structure required for trajectory prediction, allowing a lightweight linear layer to decode the learned features without additional nonlinear transformations.

The decoder removal also substantially reduces model complexity. The parameter count decreases from 402,072 to 26,520, corresponding to a 93.4% reduction relative to the original STD-WFB predictor and an 82.4% reduction relative to the FFN baseline. The number of linear operations is reduced from 405,504 to approximately 30,720 MACs per sample, although the WFB layer additionally evaluates 1,024 cosine responses per sample.

Overall, these results suggest that the nonlinear expressiveness of STD-WFB is primarily provided by the wave representation itself rather than by the decoder. Consequently, the ReLU-based MLP decoder can be replaced with a lightweight linear readout without sacrificing predictive accuracy in this controlled setting. This should not be interpreted as evidence that linear decoders are universally superior; instead, it indicates that the proposed wave representation substantially reduces the need for an additional nonlinear decoding network.

## 5.9 Discussion

A wave representation is useful for trajectory prediction because it provides a continuous function of both spatial state and elapsed time. A conventional FFN approximates continuously evolving motion by composing affine transformations and pointwise activations, which partition the input space into piecewise-linear regions. WFB instead couples state and ∆t within the same phase function. Its amplitude controls response strength, k controls spatial variation, ω controls temporal variation, and θ controls alignment. These variables are trained simultaneously within one wave response, allowing multiple aspects of trajectory dynamics to interact directly rather than being represented only through additional linear partitions.

The evaluation supports the effectiveness of this formulation from complementary perspectives. With motion features, STD-WFB-FFN using real ∆t reduces ADE by 20.4% relative to the original FFN, while the constant-∆t variant improves by 12.2%. With position-only inputs, real-interval WFB improves ADE by 10.4% relative to the original FFN and remains competitive with parameter-matched ReLU controls. This capacity-controlled result is important: it shows that WFB’s performance cannot be explained solely by comparison with a smaller baseline, while the modest differences among the matched models appropriately bound the strength of the claim.

Inference-time interventions further confirm that ∆t is operational rather than merely appended to the feature vector: replacing, reversing, or rescaling the intervals changes the predictions, with doubling the intervals increasing ADE by 30.9%. The learned parameters also exhibit measurable relationships with speed, acceleration, path length, and displacement, and the magnitude of the activation-weighted angular frequency increases from slow to fast motion. Because A, k, ω, and θ are jointly optimized within one response, the present WFB implementation provides a structured feed-forward representation of state and interval. The decoder ablation further indicates that this representation can support a lightweight linear readout, although the preliminary efficiency result requires multi-seed and runtime confirmation.

The backpropagation comparison further clarifies how WFB can be used effectively. A Laplacian-only temporal correction is not aligned sufficiently with the supervised trajectory objective, whereas STD-Laplacian-WFB-FFN preserves task-directed learning and introduces wave curvature as a weak complementary signal. Its best setting, $\overset { \cdot } { \lambda } = 1 0 ^ { - 5 }$ , improves ADE over STD-WFB-FFN, indicating that local curvature can refine the learned wave field when it is balanced with the standard gradient. Together, the accuracy, capacity-controlled, sensitivity, interpretability, and decoder results establish the intended proof of concept: a learnable wave response can serve as an effective structured component in an otherwise feed-forward predictor.

## 6 Conclusion and Future Work

This paper introduced Wave Function Backpropagation (WFB), a wave-parameterized learning formulation that directly associates a neural input with an elapsed temporal interval through learnable amplitude, wavenumber, angular frequency, and phase. Standard task gradients and a spatial-Laplacian curvature regularizer were derived for the wave parameters. The formulation was evaluated in a deliberately controlled feed-forward setting using trajectory prediction as a validation task rather than as the scope of the proposed method.

STD-WFB outperformed the original FFN baseline in both the motion-feature and position-only evaluations. The new capacity-controlled experiment further showed that real-interval WFB is competitive with parameter-matched ReLU predictors, while shuffled-interval WFB achieves the lowest mean ADE. The constant-interval and matched-baseline results indicate that model capacity and the wave representation both contribute to performance, whereas correct interval alignment is not established as the source of the gain. These bounded findings support the intended conclusion: WFB is a feasible and effective structured feed-forward learning formulation.

Future research will investigate optimization stability, parameter identifiability, theoretical approximation properties, computational complexity, tuned periodic baselines, and additional continuously evolving or irregularly sampled data. Sequence-aware propagation is reserved for future work and is not part of the present proof-of-concept evaluation.

## References

[1] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986.

[2] Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. Recurrent neural networks for multivariate time series with missing values. Scientific Reports, 2018.

[3] Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. Latent odes for irregularly-sampled time series. In Advances in Neural Information Processing Systems, 2019.

[4] Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. In Advances in Neural Information Processing Systems, 2020.

[5] Benjamin Ellenberger, Paul Haider, Federico Benitez, Jakob Jordan, Kevin Max, Ismael Jaras, Laura Kriener, and Mihai A. Petrovici. Backpropagation through space, time and the brain. Nature Communications, 17(1):66, Dec 2025.

[6] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019.

[7] Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. Spatio-temporal backpropagation for training highperformance spiking neural networks. Frontiers in Neuroscience, 12:331, 2018.

[8] Chenxiang Ma, Xinyi Chen, Yanchen Li, Qu Yang, Yujie Wu, Guoqi Li, Gang Pan, Huajin Tang, Kay Chen Tan, and Jibin Wu. Spiking neural networks for temporal processing: Status quo and future prospects, 2025.

[9] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems, 2020.

[10] Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems, 2020.

[11] Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, and Silvio Savarese. Social lstm: Human trajectory prediction in crowded spaces. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 961–971, 2016.

[12] Agrim Gupta, Justin Johnson, Li Fei-Fei, Silvio Savarese, and Alexandre Alahi. Social gan: Socially acceptable trajectories with generative adversarial networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[13] Xing Wang, Zixuan Wu, Biao Jin, Mingwei Lin, Fumin Zou, and Lyuchao Liao. Mdstf: A multi-dimensional spatio-temporal feature fusion trajectory prediction model for autonomous driving. Complex & Intelligent Systems, 10:6647–6665, 2024.

[14] Chi Zhang. Pedestrian Behavior Prediction Using Machine Learning Methods. PhD thesis, University of Gothenburg and Chalmers University of Technology, 2024.

[15] N. Song et al. Motion forecasting in continuous driving. In NeurIPS, 2024.

[16] Anonymous. Enhanced prediction of multi-agent trajectories via control variable modeling. arXiv, 2024.

[17] Inhwan Bae et al. Singulartrajectory: Universal trajectory prediction. In CVPR, 2024.

[18] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos. Joint attention in autonomous driving (jaad). arXiv preprint arXiv:1609.04741, 2016.

[19] Alon Lerner, Yiorgos Chrysanthou, and Dani Lischinski. Crowds by example. In Computer Graphics Forum, volume 26, pages 655–664. Wiley Online Library, 2007.