# Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation

Hojoon Son, Fan Zhang

Abstract—Vision-Language-Action (VLA) models have become a major paradigm for Vision-and-Language Navigation (VLN). However, in safety-critical facilities, invisible risks such as radiation or temperature spikes cannot be detected by an RGB camera, and handling each risk is expensive, requiring a new encoder, new data, and model retraining. We propose Physics-Guided Visual Prompting (PG-VP), a plug-and-play multimodal perception module that instead reuses what a frozen VLA model already does well: avoiding visible obstacles. Given a proximal radiation or thermal source, PG-VP performs a physics-guided risk assessment to determine the avoidance direction and overlays a corresponding virtual obstacle that moves across consecutive frames (Dynamic Visual Prompting). The navigation policy then naturally detours around this invisible hazard. The identical virtual obstacle is used regardless of hazard type, so the visual prompting pattern remains fixed as sensors are added. When no hazard is detected, nothing is rendered, and the policy behaves exactly as it would without PG-VP. We evaluate PG-VP on OmniNav using the val-unseen splits of R2R-CE and RxR-CE, where it guides the policy toward intended low-risk actions in 84.9% and 83.2% of cases, at a cost of 6.8 and 7.9 percentage points in navigation success rate. We further test it with distinct scenarios on a real robot in the presence of actual thermal and radiation sources, all without any retraining. The real test shows that PG-VP effectively avoids these invisible hazards, improving worst-10% average trajectory safety by 63.45% and 32.59% against thermal and radiation sources, respectively.

Index Terms—Vision-Language-Action Models, Vision-and-Language Navigation, Physics-Informed Machine Learning, Multimodal Perception, Visual Prompting

## I. INTRODUCTION

Vision-and-Language Navigation (VLN) is a core problem in achieving generalist navigation capabilities guided by natural language instructions. In particular, implementing VLN in safety-critical facilities, such as nuclear power plants (NPPs), is beneficial for their operations and maintenance (O&M). Son et al. [1] proposed a systematic catalog of potential robotic applications in NPP O&M by analyzing historical nuclear incidents using domain-specific large language models (LLMs). Based on their insights, NPP O&M requires multitask robots rather than task-specific ones. Therefore, VLN systems with generalist robots can be valuable not only in everyday environments but also in safety-critical domains.

Vision-Language-Action (VLA) models have recently emerged, leveraging the foundational capabilities of Vision-Language Models (VLMs) to unify the perception-to-action pipeline into a jointly trained policy that maps multimodal inputs directly to action tokens, and now represent the latest paradigm of VLN [2]–[5]. Still, a key challenge remains for VLA-based navigation in safety-critical environments: visual perception alone is insufficient. Invisible hazards, such as radiation or temperature spikes, can arise during navigation [1], but they leave no signature in the visible spectrum, and thus cannot be detected by a standard RGB camera. A straightforward solution would be to extend a VLA model’s input modalities, but this is expensive: it requires a new encoder, new paired multimodal-language data, and retraining for every new modality [6]–[9]. More critically, the model offers no guarantee that it preserves its original navigation performance in scenarios without such invisible hazards.

We propose the physics-guided visual prompting (PG-VP) module, which guides the navigation trajectory to naturally detour around invisible proximal hazards by leveraging the frozen VLA model’s foundational capability to avoid visible obstacles. It assesses the risk posed by proximal radiation or thermal sources, identifies the avoidance direction, and overlays the corresponding virtual obstacle onto the VLA model’s input images. Instead of overlaying a static obstacle onto the scene, PG-VP shifts the virtual obstacle across frames to smoothly guide the trajectory, which we call dynamic visual prompting. As a result, the VLA model treats the injected context as a real obstacle and implicitly avoids the invisible hazards. The main benefits of PG-VP are as follows:

1) PG-VP is a plug-and-play visual prompting module that injects non-geometric multimodal physical context into a VLA-based navigation framework in a zero-shot manner, without touching the model weights. When no hazard is detected, it renders no virtual obstacles, fully preserving the native navigation performance.

2) PG-VP requires only a risk assessment model per modality. Building this model is far cheaper than adding an encoder, collecting new data, and retraining.

3) PG-VP renders the same virtual obstacle regardless of modality type. Because the model interprets a single visual prompting pattern rather than a different one for each modality, the alignment with the training distribution remains consistent across sensing modalities.

We used OmniNav as the underlying navigation policy [3], evaluated PG-VP’s trajectory-guidance effectiveness across unseen R2R-CE and RxR-CE splits [10]–[13], and demonstrated physical deployment under actual thermal and radiation sources.

## II. RELATED WORK

## A. VLA-based Robot Navigation

NaVid [14] was an early attempt to leverage video-based VLMs, directly mapping natural language instructions to lowlevel navigation actions from an RGB stream; Uni-NaVid [15] and NaVILA [2] established this formulation as a VLA paradigm for navigation. Since then, this has advanced along several directions: broadening generalization across embodiments, tasks, and goal modalities through large-scale training [5], [16], [17]; compressing accumulated video context and improving systemic efficiency for real-time deployment [18]– [20]; injecting geometric priors to recover 3D spatial understanding from RGB input alone [21], [22]; decoupling highlevel semantic reasoning from low-level control in dual-system architectures [23]–[25]; and post-training the policy with selfcorrection data flywheels [26]. At the intersection of these directions, OmniNav [3] adopts a fast-slow design: a fast system generates short-horizon continuous-space waypoints via a flow-matching policy, while a slow system plans longhorizon sub-goals from frontiers and visual memory. We adopt OmniNav’s fast system as our underlying vision-only navigation policy.

## B. Extending Perception Beyond RGB

In VLA-based manipulation, depth can be reconstructed from RGB alone, but contact force, sound, and temperature leave no signature in the visible spectrum. To include these physical quantities in the VLA policy input, researchers must add an additional sensor, and several studies have pursued this approach [6]–[8], [27]. Yet, adding a new sensor is costly. A new modality requires collecting additional robot data [9] and retraining the model to recover the pretrained alignment [28], [29]. Furthermore, for deployment, platforms without the additional sensor require a separate policy [30].

In VLA-based navigation, geometric quantities, such as goal coordinates and poses, are directly incorporated into the policy [3], [4], [17], [31], or 3D structure is reconstructed from RGB images [21], [22]. In some cases, additional geometric sensors fall outside the VLA policy. For example, NaVILA [2] leverages LiDAR readings for its locomotion controller, while the VLA policy itself relies only on visual input. However, to the best of our knowledge, no prior research has used nongeometric physical quantities as input to VLA navigation.

## C. Visual Prompting

Overlaying marks on input images has proven effective for guiding a model’s interpretation. BYOVLA inpaints a taskirrelevant region to recover the nominal behavior under visual distraction [32], and Visual Attentive Prompting (VAP) applies a tint to the personalized target object [33]. For navigation, waypoint prediction can be enhanced by placing markers on a navigable region [34], grounding goal pixels [35], or, in concurrent work, segmenting traversable (green) and nontraversable (red) regions [36]. These visual prompts emphasize, remove, or label information already present in input images. In contrast, we inject visual context entirely absent from the scene: a moving virtual obstacle.

![](images/37a82e9d1a422fab3f0c3e2788921c066966e098dfee64a6e7b8dacf752db131.jpg)  
Fig. 1. Overview of PG-VP framework.

## III. METHODOLOGY

PG-VP detects non-geometric physical quantities using an integrated multimodal perception system (Section III-A), assesses their risk (Section III-B), and guides a navigation policy to naturally avoid proximal invisible hazards (Section III-C). The policy $\pi$ maps an RGB observation $\mathbf { I } _ { t }$ and a language instruction ℓ to an action $a _ { t } \sim \pi ( \cdot \mid \mathbf { I } _ { t } , \ell )$ . PG-VP overlays a virtual obstacle in the direction $d _ { t } .$ , converting $\mathbf { I } _ { t }$ into ${ \ddot { \mathbf { I } } } _ { t } .$ The action is then inferred as $a _ { t } \sim \pi ( \cdot \ | \ \tilde { \mathbf { I } } _ { t } , \ell )$ , while the policy weights and language instruction remain unchanged. When no hazard is detected, $\tilde { \mathbf { I } } _ { t } = \mathbf I _ { t }$ . Figure 1 illustrates the overall flow.

## A. Integrated Robotic Multimodal Perception System

Identifying proximal invisible hazards requires auxiliary sensing beyond RGB vision. Our multimodal perception system combines RGB, radiation, and thermal sensing modalities, with radiation and temperature serving as the target nongeometric physical quantities. The vision system consists of a front-facing wide-angle ELP global-shutter camera with a 120<sup>◦</sup> horizontal field of view (HFOV) and two webcams with a $6 9 ^ { \circ }$ HFOV. The two webcams are yawed by ±90<sup>◦</sup> relative to the robot’s forward direction to provide left and right views. For radiation detection, we use a RadEye SPRD-ER, which provides point-wise gamma-ray measurements at the detector’s location. For temperature detection, we use two FLIR Lepton 3.5 radiometric thermal camera modules, each mounted on a PureThermal Mini Pro JST-SR carrier board. To extend horizontal thermal coverage, we mount the two sensors side by side and yaw them 15<sup>◦</sup> outward from the robot’s forward direction. With 57<sup>◦</sup> HFOV for each sensor, this configuration provides a nominal combined HFOV of 87<sup>◦</sup>, with a $2 7 ^ { \circ }$ overlap in the central region.

All sensors are mounted on a Unitree Go2 EDU quadruped robot. The RGB cameras are connected directly to the robot’s onboard computer, a Jetson Orin Nano. Because of hardware compatibility constraints, the radiation and thermal sensors connect to an external Windows-based single-board computer (SBC). The latest radiation and thermal measurements are transmitted to a Redis server running on the robot’s onboard computer, making them available in real time. This configuration is inspired by previous studies of Son et al. [37], [38].

![](images/866bd7540d234311f47c82d6df3af437f962e92ee04defa7e5a395413f2c22a9.jpg)  
Fig. 2. Hardware configuration of the robotic multimodal perception system.

Figure 2 shows the sensor placement and data flow of the proposed robotic multimodal perception system.

## B. Physics-Guided Risk Assessment

We develop separate risk assessment (RA) models for radiation and temperature. Each RA model operates independently of the navigation policy and is designed around the sensor characteristics and configuration. The models do not explicitly quantify risk; instead, they identify nearby highrisk situations where the robot should take a detour. Each RA model outputs left, right, or normal, depending on the estimated direction of the proximal hazard. The RA models run asynchronously on the robot’s onboard computer, and their latest outputs are written to Redis.

1) Temperature: The temperature RA model relies on the hardware configuration, particularly the mounting geometry of the two thermal cameras. As described in Section III-A, the two cameras are mounted side by side and yawed outward by $1 5 ^ { \circ }$ from the robot’s forward direction. Based on this configuration, the outer one-third of each thermal image is assigned to a side region, while the remaining two-thirds is assigned to a front region. This partition produces four regions: left-side $\left( \Omega _ { \mathrm { L S } } \right)$ , left-front $( \Omega _ { \mathrm { L F } } )$ , right-front $( \Omega _ { \mathrm { R F } } )$ , and rightside $( \Omega _ { \mathrm { R S } } )$

For each measurement step t, the maximum temperature within each region is computed as $T _ { t } ^ { k } = \operatorname* { m a x } _ { \mathbf { p } \in \Omega _ { k } } { \mathcal { T } } t ( \mathbf { p } )$ where $k \in$ {LS, LF, RF, RS} and $\tau _ { t ( \mathbf { p } ) }$ denotes the temperature measured at pixel p. A thermal hazard is classified as being on the left when $T _ { t } ^ { \mathrm { L S } } \geq$ max $\left( T _ { t } ^ { \mathrm { L F } } , T _ { t } ^ { \mathrm { R F } } , T _ { t } ^ { \mathrm { R S } } \right) + \Delta T _ { \mathrm { t h } }$ and on the right when $T _ { t } ^ { \mathrm { { \tilde { R S } } } } \geq$ max $\left( T _ { t } ^ { \mathrm { L S } } , T _ { t } ^ { \mathrm { L F } } , T _ { t } ^ { \mathrm { R F } } \right) + \Delta T _ { \mathrm { t h } }$ , where $\Delta T _ { \mathrm { t h } }$ is the temperature-difference threshold. A directional output $( \mathtt { l e f t }$ or right) is confirmed only if the same classification is maintained for 0.5 seconds over consecutive steps. Otherwise, the temperature RA model outputs normal.

2) Radiation: Unlike the temperature RA model, the radiation RA model relies primarily on algorithmic processing and adopts a physics-informed machine learning (PIML) approach. We developed a two-stage radiation RA model that first assesses radiation risk (risk model) and then estimates the left/right avoidance direction (direction model) from sequences of robot positions, $\mathbf { p } _ { t } = ( x _ { t } , y _ { t } )$ , and gamma-ray counts per second (CPS). Progressive densification interpolates the available samples to match the required input size when samples are insufficient in the initial input interval. The risk model uses 16 samples: 16 relative log-CPS values, 16 displacement magnitudes, and one flux-heading slope, yielding a 33- dimensional input. When the estimated risk exceeds 0.3, the direction model is activated with 30 samples, consisting of 30 relative log-CPS values, 60 planar displacement components $( \Delta x , \Delta y )$ , 60 heading-change components (sin $\Delta \theta , \cos \Delta \theta )$ and one flux-heading slope, yielding a 151-dimensional input. The heading is approximated from consecutive robot positions, and the flux-heading slope is obtained by regressing relative log-CPS changes against the corresponding heading changes.

The training data is generated from a 10m × 10m gamma radiation field computed with OpenMC across 25 arbitrary obstacle environments, using $\mathrm { ~ a ~ } ~ 1 0 0 ~ \times ~ 1 0 0$ regular mesh with 0.1m resolution. We retrieve gamma flux sequences from randomly generated robot trajectories in the radiation field, reflecting obstacle-induced attenuation and scattering. In addition, to mirror the sensor condition and counting noise, we randomly configure the flux-to-CPS scaling factor $\kappa \sim$ LogUniform $( 1 0 ^ { 2 } , 1 0 ^ { 1 0 } )$ , background count rate $b \sim$ Uniform(5, 50) CPS, and integration time $\tau \sim$ Uniform(0.5, 3.0). The gamma count, $N _ { t } ,$ can be modeled following Poisson noise $N _ { t } \sim$ Poisson $\left[ ( \boldsymbol { \kappa } \cdot \boldsymbol { q } ( \mathbf { p } _ { t } ) + \boldsymbol { b } ) \tau \right]$ where $q ( \mathbf { p } _ { t } )$ denotes gamma flux at the point $\mathbf { p } _ { t }$ . Then, the modeled input CPS, $c _ { t } .$ , can be calculated as $c _ { t } = N _ { t } / \tau$ . The number of training sample windows for the risk model is 1,051,106, and for the direction model is 911,106.

Both the risk model and the direction model use multilayer perceptrons (MLPs). The risk model uses layers of size (33, 128, 128, 64, 1) and uses a class-weighted binary crossentropy loss. It outputs a float between 0 and 1. The direction model has shared trunk layers with sizes (151, 128, 128, 64), followed by direction head layers with sizes (64, 32, 3) and physics auxiliary head with sizes (64, 32, 4). The direction head outputs the estimated direction probabilities $\hat { \mathbf { y } } _ { d i r }$ as a 3- dimensional vector, and the physics auxiliary head outputs a vector including the estimated source offset $( \hat { \Delta x _ { s } } , \hat { \Delta y _ { s } } )$ , fluxto-CPS scaling factor ${ \hat { \kappa } } ,$ , and background count rate <sup>ˆ</sup>b (total 4-dimensional output). The direction model leverages a softvoting ensemble that averages the softmax probabilities of 12 models trained with distinct random seeds. The choice of 12 is inspired by the number of parallel inference models used in the previous PIML-based radiation perception study [37]. The loss function, $\mathcal { L } _ { d i r }$ , combines two terms: (i) class-weighted cross-entropy loss, $\mathcal { L } _ { c e } ,$ that directly measures loss between the estimated direction $\hat { \mathbf { y } } _ { d i r }$ and ground truth direction $\mathbf { y } _ { d i r } ;$ (ii) physics-aware auxiliary loss, $\mathcal { L } _ { p h y s i c s } .$ , that measures the error between the reconstructed CPS, $\hat { c } _ { t } ^ { \phantom { \dagger } } ,$ , and ground truth $c _ { t }$ Equation (1) represents the combined direction model loss.

$$
\mathcal { L } _ { d i r } = \mathcal { L } _ { c e } ( \hat { \mathbf { y } } _ { d i r } , \mathbf { y } _ { d i r } ) + 0 . 1 \mathcal { L } _ { p h y s i c s } ( \hat { c } _ { t } , c _ { t } )\tag{1}
$$

The physics-aware auxiliary loss is defined as equation (2).

$$
\mathcal { L } _ { p h y s i c s } = \frac { 1 } { 3 0 } \sum _ { t = 1 } ^ { 3 0 }  { | \log ( } \hat { c } _ { t } + 1 ) - \log ( c _ { t } + 1 ) |\tag{2}
$$

Equation (3) reconstructs the CPS using the inverse-square law, where $\hat { r } _ { t } ( \hat { \Delta x _ { s } } , \hat { \Delta y _ { s } } )$ denotes the estimated source-to-

detector distance and $r _ { \mathrm { d e t } }$ is the detector radius, 2.82cm.

$$
\hat { c } _ { t } = \frac { \hat { \kappa } } { 4 \pi ( 1 0 0 \cdot \hat { r } _ { t } ( \Delta \hat { x } _ { s } , \Delta \hat { y } _ { s } ) + r _ { \mathrm { d e t } } ) ^ { 2 } } + \hat { b }\tag{3}
$$

The physics-guided auxiliary loss encourages the CPS values reconstructed from the estimated source location and intensity to match the observed measurements. This formulation is inspired by Son et al. [37]. However, unlike Son’s method, the proposed model determines the avoidance direction through a single forward pass of a pre-trained ensemble model. The twostage radiation RA model achieves 85.7% accuracy across 14 real-robot short-trajectory scenarios we collected, and 77.4% accuracy on three real-robot radiation source localization (RSL) scenarios from Son’s research [37], demonstrating its acceptable applicability.

## C. Dynamic Visual Prompting

Based on the proximal risk direction estimated by the RA models, we render a virtual obstacle as a visual prompt, maintaining a consistent visual representation across sensing modalities. This representation has two benefits. First, it maps heterogeneous physical risks to a single visual pattern, allowing the same guidance across distinct modalities. If obstacle-like visual patterns fall within the VLA model’s training data distribution, it can interpret the visual prompt without additional modality-specific training. Second, directly overlaying radiation or temperature fields may be ambiguous and require additional semantic interpretation. In contrast, a virtual obstacle is more straightforward because it directly indicates the region that the robot should avoid.

The virtual obstacle is rendered as a dynamic curved wall. The wall gradually grows and shrinks across successive frames. Let k be the relative frame index from the start, and $N _ { g }$ and $N _ { v }$ be the numbers of growth and vanishing frames, respectively. The horizontal wall-length ratio $r _ { w } ( k )$ is defined as a nonlinear function of $k ,$ as shown in equation (4). Here, $S _ { \mathrm { g u i d e } }$ denotes the guidance strength, defined as the maximum horizontal wall-length ratio.

$$
r _ { w } ( k ) = \left\{ \begin{array} { l l } { S _ { \mathrm { g u i d e } } \sqrt { \displaystyle \frac { k } { N _ { g } } } , } & { \mathrm { i f ~ } 0 < k \le N _ { g } } \\ { S _ { \mathrm { g u i d e } } \sqrt { \displaystyle \frac { N _ { v } - ( k - N _ { g } ) } { N _ { v } } } , } & { \mathrm { i f ~ } N _ { g } < k \le N _ { g } + N _ { v } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{4}
$$

Let W and H denote the width and height of the front-view image, respectively. The dynamic virtual wall extends inward from either lateral edge of the image. Its horizontal length $l _ { w } ( k )$ is calculated as $l _ { w } ( k ) = r _ { w } ( k ) W$ . Since $S _ { \mathrm { g u i d e } }$ is the maximum $r _ { w } ( k )$ , the maximum $l _ { w } ( k )$ is $S _ { \mathrm { g u i d e } } W$

Let the origin of the image coordinate system be located at the upper-left corner. For a virtual wall generated from the left edge, the wall region is bounded by the left image edge, an inner vertical boundary at $x = l _ { w } ( k )$ , and two quadratic Bezier curves,´ $\mathbf { B } ( u ) = ( 1 - u ) ^ { 2 } \mathbf { P } _ { 0 } + 2 ( 1 - u ) u \mathbf { P } _ { 1 } + u ^ { 2 } \mathbf { P } _ { 2 } ,$ forming the upper and lower boundaries, where $\mathbf { P } _ { 0 }$ and $\mathbf { P } _ { 2 }$ are the endpoints, ${ \bf P } _ { 1 }$ is the control point, and $u \in [ 0 , 1 ]$

Prompt: Walk across bedroom to open door, enter sitting room and continue straight down hall, turn left into bathroom. Stop at sink.

![](images/a83ca25d5de54ae76e58aaa9aa162e89aaf68ac59b9628eb7736c760311f8c84.jpg)  
Fig. 3. Dynamic visual prompting process in R2R-CE val-unseen sample. (Right wall, $S _ { \mathrm { g u i d e } } = 0 . 6 { \overset { \cdot } { , } } N _ { g } { \overset { \cdot } { = } } { \bar { N } } _ { v } { \overset { \cdot } { = } } 4 )$

For the upper curve, the endpoints are ${ \bf P } _ { 0 } ^ { u } = ( 0 , 0 . 1 H )$ and $\mathbf { P } _ { 2 } ^ { u } \ = \ ( l _ { w } ( k ) , 0 . 3 H )$ , and the control point is $\mathbf { P } _ { 1 } ^ { u } \ =$ $( 0 . 0 6 l _ { w } ( k ) , 0 . 2 H )$ . For the lower curve, the endpoints are ${ \bf P } _ { 0 } ^ { b } = ( \dot { d } _ { b } ( k ) , H )$ and $\mathbf { P } _ { 2 } ^ { b } = ( l _ { w } ( k ) , 0 . 8 H )$ , and the control point is $\mathbf { P } _ { 1 } ^ { b } = ( d _ { b } ( k ) + 0 . 0 6 l _ { w } ( k ) , 0 . 9 H )$ . The bottom offset $d _ { b } ( k )$ increases linearly from 0 to 0.2W during the growth frames and decreases to 0 during the vanishing frames. While $\mathbf { P } _ { 0 } ^ { u }$ remains fixed, $\mathbf { P } _ { 0 } ^ { b }$ moves horizontally. This makes the wall appear to rotate from the side into the front view and then move back, rather than simply expanding horizontally. The right virtual wall is generated by mirroring the left virtual wall horizontally.

The wall region is filled with a concrete-like texture generated from Gaussian-noised pixels with a mean of 80 and a standard deviation of 15. Directional shading creates a natural sense of depth, with texture brightness increasing from 50% at the inner boundary to 100% at the image edge. The virtual wall is rendered independently for each front-view frame, and the side-view image corresponding to the wall direction is also completely covered with the same texture.

Figure 3 shows the dynamic visual prompting process during the growth and vanishing phases and its effect on the navigation trajectory. It presents an R2R-CE val-unseen sample with a virtual wall generated from the right side of the image, where $S _ { \mathrm { g u i d e } } = 0 . 6$ and $N _ { q } = N _ { v } = 4$ . To evaluate PG-$\mathrm { { V P } { ^ { \circ } s } }$ direct effect, we introduce a hypothetical invisible hazard that bypasses the RA stage and applies only dynamic visual prompting. Compared with the baseline trajectory, the PG-VPguided trajectory naturally detours around the hypothetical invisible hazard. $\mathrm { A t } ~ k = 4 , r _ { w } ( k )$ reaches its maximum value of $S _ { \mathrm { g u i d e } } .$ , resulting in a wall length of $l _ { w } ( 4 ) = S _ { \mathrm { g u i d e } } W = 0 . 6 W$ and a bottom offset of $d _ { b } ( 4 ) = 0 . 2 W$

![](images/ea36ba8c8a1e856ca0b759d48a795f9d81d87fe5d2a346dfb0c530c0a0d4f397.jpg)

![](images/17aee78e7e3b7a872fff87798bd8d8077a2ea4b7012f3386b3c474f9b00a2d7f.jpg)

![](images/e29a521ed26c092c3d55ce8543163921070a35193b76355de5a098fa84fb133f.jpg)  
(a) CGR (%) (b) Max. $\left| x _ { d } \right|$ (m) (c) SR Drop (pp)  
Fig. 4. Trade-off between PG-VP trajectory steering effectiveness and VLN success rate. The baseline model is OmniNav [3]. (a) Correct-guidance ratio from PG-VP, (b) Maximum displacement in the x-direction from PG-VP, (c) Success rate drop of VLN

## IV. RESULTS

## A. Trajectory Steering on VLN Benchmarks

We evaluated the trajectory-guidance effectiveness of PG-VP on the val-unseen splits of R2R-CE and RxR-CE [10]– [13]. The underlying assumption is that an invisible hazard exists in the navigation scenario, as illustrated in Fig. 3, with the virtual wall overlaid at the appropriate location and time step. We first measure the total number of frames in a scenario without PG-VP, then apply visual prompting to a randomly chosen start index between 25% and 75% of the total. The number of PG-VP frames is set to $N _ { g } = N _ { v } = 4$ , and the guidance strength $S _ { \mathrm { g u i d e } }$ is varied across 0.55, 0.60, and 0.65. We use the OmniNav model as the navigation policy [3].

The focus here is not on the PG-VP-applied VLA model achieving the best navigation performance, but on how it effectively guides a trajectory to detour around a hypothetical invisible hazard. While PG-VP steers the navigation trajectory appropriately, it reduces the Success Rate (SR) in VLN by introducing new visual context during closed-loop navigation. Figure 4 shows this performance trade-off in R2R-CE and RxR-CE val-unseen splits. The Correct-Guidance Ratio (CGR) is the percentage of scenarios in which the visual prompt guides the trajectory as intended, and the x-direction displacement, $\left| x _ { d } \right|$ , measures the magnitude of the robot’s lateral displacement induced by the visual prompt. Note that the maximum $\left| x _ { d } \right|$ does not occur within the visual-prompt frames; it occurs in the post-prompt phase, 20 extra frames after the visual-prompt phase, from $k = N _ { g } + N _ { v } + 1$ to $k = N _ { g } + N _ { v } + 2 0$ . That means PG-VP affects the navigation trajectory even after the visual prompting phase.

In Fig. 4a, CGR and $S _ { \mathrm { g u i d e } }$ are positively correlated, indicating that greater guidance strength increases the likelihood of correctly steering the trajectory. Similarly, in Fig. 4b, maximum $\left| x _ { d } \right|$ and $S _ { \mathrm { g u i d e } }$ are positively correlated, indicating that greater guidance strength produces larger lateral displacement. Both figures show that guidance strength controls trajectory steering as intended. However, guidance strength comes with a trade-off: it causes an SR drop in VLN. Figure 4c illustrates the SR drop trend as $S _ { \mathrm { g u i d e } }$ increases; the greater the $S _ { \mathrm { g u i d e } }$ , the larger the SR drop. Notably, the slope of the SR drop in R2R-CE becomes steeper, while the slopes of CGR and max |x<sub>d</sub>| become substantially less steep in both R2R-CE and RxR-CE over the $S _ { \mathrm { g u i d e } }$ interval of 0.60–0.65 compared with 0.55–0.60. In particular, CGR and max $\left| x _ { d } \right|$ increase significantly on the R2R-CE splits as $S _ { \mathrm { g u i d e } }$ increases from 0.55 to 0.60. This supports choosing $S _ { \mathrm { g u i d e } } ~ = ~ 0 . 6 0$ as a reasonable balance between navigation success and guidance effectiveness.

TABLE I  
NAVIGATION PERFORMANCE (%) ON R2R-CE AND RXR-CE VAL-UNSEEN SPLITS.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Obs.</td><td colspan="2">R2R-CE</td><td colspan="2">RxR-CE</td></tr><tr><td>SR↑</td><td>SPL↑</td><td>SR↑</td><td>SPL↑</td></tr><tr><td></td><td></td><td>49.3</td><td></td></tr><tr><td>NaVILA (2025) [2]</td><td>S.RGB</td><td>54.0</td><td>49.0</td><td></td><td>44.0</td></tr><tr><td>StreamVLN (2025) [18]</td><td>S.RGB</td><td>56.4</td><td>50.2</td><td>54.4</td><td>45.4</td></tr><tr><td>CorrectNav (2026) [26]</td><td>S.RGB S.RGB</td><td>65.1 60.5</td><td>62.3 56.8</td><td>69.3 56.2</td><td>63.3</td></tr><tr><td>JanusVLN (2026) [21]</td><td></td><td>61.7</td><td></td><td></td><td>47.5</td></tr><tr><td>NavFoM (2026) [16]</td><td>Pano. Pano.</td><td>69.5</td><td>55.3 66.1</td><td>64.4 73.6</td><td>56.2</td></tr><tr><td>OmniNav (2026) [3] ABot-N0 (2026) [5]</td><td>Pano.</td><td>66.4</td><td>63.9</td><td>69.3</td><td>62.0</td></tr><tr><td>SPAN-Nav (2026) [22]</td><td>Pano.</td><td>66.3</td><td>59.3</td><td>69.7</td><td>60.0 60.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniNav (our run)</td><td>Pano.</td><td>68.7</td><td>65.6</td><td>68.8</td><td>57.8</td></tr><tr><td>PG-VP  $( S _ { \mathrm { g u i d e } } { = } 0 . 5 5 )$ </td><td>Pano.</td><td>65.4</td><td>61.7</td><td>63.7</td><td>52.2</td></tr><tr><td>PG-VP  $( S _ { \mathrm { g u i d e } } { = } 0 . 6 0 )$ </td><td>Pano.</td><td>61.9</td><td>58.2</td><td>61.0</td><td>49.7</td></tr><tr><td>PG-VP  $( S _ { \mathrm { g u i d e } } { = } 0 . 6 5 )$ </td><td>Pano.</td><td>57.8</td><td>54.3</td><td>59.4</td><td>48.0</td></tr></table>

In addition, the navigation performance of the OmniNav model with PG-VP $( S _ { \mathrm { g u i d e } } = 0 . 6 0 )$ is acceptable compared to other previous methods. Table I presents the SR and the Success weighted by Path Length (SPL) across various recent navigation methods on the benchmarks. The baseline Omni-Nav performance we obtained by running the official code differs from the results reported in the original paper [3]. Even with the performance drop, PG-VP with $S _ { \mathrm { g u i d e } } = 0 . 6 0$ outperforms most single-camera-based methods except CorrectNav. It also maintains performance comparable to NavFoM, a panoramic camera-based method, although it falls behind other panoramic-view methods. Even so, PG-VP with $S _ { \mathrm { g u i d e } } = 0 . 6 0$ does not substantially degrade navigation performance relative to other recent methods. Reducing $S _ { \mathrm { g u i d e } }$ to 0.55 improves navigation performance, but the insight from Fig. 4 suggests that the resulting trajectory guidance is too weak. Increasing $S _ { \mathrm { g u i d e } }$ to 0.65, by contrast, slightly improves guidance effectiveness, but navigation performance drops sharply and is no longer competitive with other recent methods.

Figure 5 shows the Cartesian average of PG-VP $( S _ { \mathrm { g u i d e } } =$ 0.60, $N _ { g } = N _ { v } = 4 )$ trajectories on the R2R-CE and RxR-CE val-unseen splits, presenting the qualitative result. The solid lines show the agent’s trajectories during the visual prompting phase, when the virtual obstacle is overlaid, while the dashed lines show its trajectories during the post-prompting phase, which covers the following 20 frames. Overall (blue line), dynamic visual prompting guides the trajectories as intended, with trajectories tending to continue in the intended steering direction even after the visual prompting phase ends. Notably, when navigation missions are successfully completed (green lines), trajectories in the post-prompting phase tend to return to the original heading after briefly continuing in the steered direction. However, trajectories keep continuing in the steered direction when navigation failures occur (red line).

![](images/a7a61e03254dce2a6eaaba2f92fb07a90a0f4491086b43a3914bb5c1b8f31366.jpg)  
(a) R2R-CE

![](images/93794e7a6fc6303cf2e9c5ae83076d1954c068bb77526113b9d3111c854fe5b7.jpg)  
(b) RxR-CE  
Fig. 5. Cartesian average of PG-VP trajectories on the R2R-CE and RxR-CE val-unseen splits. $( S _ { \mathrm { g u i d e } } = 0 . 6 , N _ { g } = \overset { \cdot } { N } _ { v } = 4 )$

## B. Experimental Validation

We experimentally validated PG-VP’s low-risk navigation trajectories by fully integrating the system: a physical robot (Section III-A), the physics-guided RA models (Section III-B), and dynamic visual prompting (Section III-C). At each time step, the VLA model runs on a remote server and communicates with the robot via HTTP, while the RA models and dynamic visual prompting run onboard. Four distinct scenarios are configured: Scenarios A and B test a thermal hazard with $\Delta T _ { \mathrm { t h } } = 3 0$ , and scenarios C and D test a radiation hazard.

Figure 6 illustrates navigation language prompts, hazard locations, and robot trajectories with and without PG-VP, when $S _ { \mathrm { g u i d e } } ~ = ~ 0 . 6$ and $N _ { g } = N _ { v } = 4 .$ . The language prompt is a fine-grained instruction (Scenario A), a coarse instruction (Scenario B), an exploration task (Scenario C), or a reaching task (Scenario D). Safety concerns restrict the use of highly dangerous hazards: Scenarios A and B use a heat gun and a room heater as thermal hazards, respectively, and Scenarios C and D use nine-bundled uranium rods as a radiation hazard. For all scenarios, the navigation policy successfully achieves the goal, and the PG-VP-applied trajectory (green) effectively detours around a thermal or radiation hazard compared to the baseline trajectory (red). Note that if the visual appearance matches these scenarios but no actual thermal or radiation hazards exist, the PG-VP trajectory will follow the baseline; for example, turning off the heat gun or room heater, or using non-radioactive iron rods.

We define hazard-specific safety metrics based on each sensor’s measurement characteristics. Thermal cameras measure the temperature of object surfaces rather than the ambient temperature at the robot’s location. We therefore use heatsource proximity (distance to the heat source) as a thermal safety metric. In contrast, the radiation detector measures gamma-ray counts at the robot’s location. We thus use the net count rate (background-subtracted gamma-ray CPS) to

![](images/66bc55fb1646760c177c26b409e4f7430e34847a7d401566c1647c40a7c907ac.jpg)  
Fig. 6. Real robot demonstration $( S _ { \mathrm { g u i d e } } = 0 . 6 , N _ { g } = N _ { v } = 4 , \Delta T _ { \mathrm { t h } } = 3 0 )$ A: Fine-grained instruction, B: Coarse instruction, C: Exploration task, D: Reaching task.

TABLE II  
SAFETY METRICS COMPARISON BETWEEN THE BASELINE AND PG-VP.
<table><tr><td rowspan="2">Method</td><td colspan="2">Heat-source proximity (m) ↑</td><td colspan="3">Net count rate (CPS) ↓</td></tr><tr><td>Scn. Min.</td><td></td><td>L-10% Avg.</td><td>Scn. Peak</td><td>H-10% Avg.</td></tr><tr><td>Baseline PG-VP</td><td>A</td><td>1.06 1.53</td><td>C</td><td>93.00 46.00</td><td>92.00 44.28</td></tr><tr><td>Baseline PG-VP</td><td>B</td><td>0.79 1.54</td><td>1.55 0.85 1.57</td><td>149.00 D 137.00</td><td>147.81 128.14</td></tr></table>

quantify the radiation levels encountered along the trajectory.   
The background radiation is 31 CPS in scenarios C and D.

Table II compares safety metrics between the baseline and PG-VP-applied model. We compare the extreme value (minimum heat-source proximity or peak net count rate) with the average of the worst 10% (lowest 10% or highest 10%). While the extreme value shows the instantaneous risk, the worst 10% average quantifies sustained hazard exposure and improves robustness to transient sensor noise. For the extreme value, the PG-VP-applied model outperforms the baseline by 69.64% for thermal hazard and 29.30% for radiation hazard. Also, for the average of the worst 10%, it improves by 63.45% and 32.59% for thermal and radiation hazards, respectively.

Figure 7 illustrates the safety metrics as a function of the robot’s travel distance, along with the onset of the visual prompt. For Scenarios A and B, the baseline heat-source proximity is V-shaped, whereas PG-VP flattens it around the minimum. For Scenarios C and D, the baseline net count rate peaks sharply. While PG-VP nearly eliminates the sharp peak in Scenario C, a peak still remains in Scenario D, but with a lower magnitude and slope. PG-VP onset points can explain why Scenario D fails to eliminate the sharp peak. In Scenarios

![](images/633a37f4c3962055c5af46f66996ce3adcd0c33b6d2bc4ff27f235a1dc63bc03.jpg)  
(a) Scenario A

![](images/bd90eb69d71805b6c037694e0401c3e1abbf21685ab71e05f44c3e70199d7e9c.jpg)  
(b) Scenario B

![](images/e34f2fa90650e62f765c6e19095f3121c89411ae51cb4cc63d396a2b722009a0.jpg)  
(c) Scenario C

![](images/ab93af344a42549af38da064b0f166ef8606ba96bf076792c08d7a0fb60a9c16.jpg)  
(d) Scenario D  
Fig. 7. Validation of low-risk navigation trajectories from the safety metrics comparison between the baseline and PG-VP-applied model. All are plotted as a function of distance traveled.

A, B, and C, RA models trigger the visual prompt well before the risk escalates. In contrast, in Scenario D, the RA model fails to trigger PG-VP early enough because the box occludes the radiation source, as shown in Fig. 6, constraining the net count rate to a constant level. PG-VP is activated immediately after a sharp count-rate surge, delaying hazard detours.

## V. CONCLUSIONS

We proposed a physics-guided visual prompting method that naturally steers the VLA-based navigation trajectory to detour around non-geometric physical hazards in a zero-shot manner. By fusing radiation and thermal data, the risk assessment model determines the avoidance direction. The dynamic visual prompting module then overlays a corresponding virtual wall on the input RGB image to guide the frozen navigation policy’s decision. The method is validated in simulation benchmarks and physical experiments. Results show that PG-VP effectively guides navigation trajectories and reduces the invisible risk from actual thermal and radiation sources.

Future work can include a broader range of real-world navigation scenarios, not only in laboratory settings but also in industrial environments, to rigorously validate the integrated PG-VP framework. Another direction is to investigate how PG-VP can achieve modality-specific objectives beyond hazard avoidance, such as localizing radiation sources or finding high-temperature spots. Furthermore, the framework can be extended beyond radiation and thermal sensing to other modalities, such as gas sensing, by designing corresponding risk assessment models while keeping the core navigation policy unchanged.

[1] H. Son and F. Zhang, “Use case development for robot-assisted nuclear power plant operation and maintenance using domain-specific llm,” Nuclear Technology, pp. 1–16, 2025.

[2] A.-C. Cheng, Y. Ji, Z. Yang, X. Zou, J. Kautz, E. Biyik, H. Yin, S. Liu, and X. Wang, “Navila: Legged robot vision-language-action model for navigation,” in RSS, 2025.

[3] X. Xue, J. Hu, M. Luo, S. Xie, J. Chen, Z. Xie, K. Quan, W. Guo, Z. Chu, M. Xu, and Z. Zhu, “Omninav: A unified framework for prospective exploration and visual-language navigation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=zGtTQTD1zu

[4] J. Zhang, G. Zhou, H. Yin, Y. Huang, Z. Lei, Q. Peng, H. Yuan, J. Zhang, X. Guo, X. Chen et al., “Qwen-robotnav technical report: A scalable navigation model designed for an agentic navigation system,” arXiv preprint arXiv:2606.18112, 2026.

[5] Z. Chu, S. Xie, X. Wu, Y. Shen, M. Luo, Z. Wang, F. Liu, X. Leng, J. Hu, M. Yin et al., “Abot-n0: Technical report on the vla foundation model for versatile embodied navigation,” arXiv preprint arXiv:2602.11598, 2026.

[6] J. Huang, S. Wang, F. Lin, Y. Hu, C. Wen, and Y. Gao, “Tactile-vla: unlocking vision-language-action model’s physical knowledge for tactile generalization,” arXiv preprint arXiv:2507.09160, 2025.

[7] J. Jones, O. Mees, C. Sferrazza, K. Stachowicz, P. Abbeel, and S. Levine, “Beyond sight: Finetuning generalist robot policies with heterogeneous sensors via language grounding,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 5961–5968.

[8] Z. Cheng, Y. Zhang, A. Tang, K. Wang, W. Zhang, H. Li, H. Zhang, and L. Song, “Omnivtla: Vision-tactile-language-action models with semantic-aligned tactile sensing,” IEEE Robotics and Automation Letters, 2026.

[9] Z. Wang, B. Wang, H. Zhang, T. Du, T. Chen, G. Sun, Y. He, Z. Shen, W. Ye, and A. Li, “Vision-language-action in robotics: A survey of datasets, benchmarks, and data engines,” arXiv preprint arXiv:2604.23001, 2026.

[10] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3D: Learning from RGB-D data in indoor environments,” International Conference on 3D Vision (3DV), 2017.

[11] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sunderhauf,¨ I. Reid, S. Gould, and A. Van Den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3674–3683.

[12] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-acrossroom: Multilingual vision-and-language navigation with dense spatiotemporal grounding,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 4392–4412.

[13] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision and language navigation in continuous environments,” in European Conference on Computer Vision (ECCV), 2020.

[14] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang, “Navid: Video-based vlm plans the next step for visionand-language navigation,” arXiv preprint arXiv:2402.15852, 2024.

[15] J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang, “Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks,” Robotics: Science and Systems, 2025.

[16] J. Zhang, A. Li, Y. Qi, M. Li, J. Liu, S. Wang, H. Liu, G. Zhou, Y. Wu, X. Li et al., “Embodied navigation foundation model,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 127 293– 127 322.

[17] N. Hirose, C. Glossop, D. Shah, and S. Levine, “Omnivla: An omni-modal vision-language-action model for robot navigation,” arXiv preprint arXiv:2509.19480, 2025.

[18] M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen et al., “Streamvln: Streaming vision-andlanguage navigation via slowfast context modeling,” arXiv preprint arXiv:2507.05240, 2025.

[19] L. Zhang, X. Hao, Q. Xu, Q. Zhang, X. Zhang, P. Wang, J. Zhang, Z. Wang, S. Zhang, and R. Xu, “Mapnav: A novel memory representation via annotated semantic maps for vlm-based vision-andlanguage navigation,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 13 032–13 056.

[20] D. Zheng, S. Huang, Y. Li, and L. Wang, “Efficient-vln: A simple yet strong baseline for efficient vision-language navigation,” arXiv preprint arXiv:2512.10310, 2025.

[21] S. Zeng, D. Qi, X. Chang, F. Xiong, S. Xie, X. Wu, S. Liang, M. Xu, and X. Wei, “Janusvln: Decoupling semantics and spatiality with dual implicit memory for vision-language navigation,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 33 001– 33 026.

[22] J. Liu, T. Xu, J. Chen, L. Yue, J. Zhang, Z. Wang, M. Li, Q. Zhao, A. Li, Q. Su et al., “Span-nav: Generalized spatial awareness for versatile vision-language navigation,” arXiv preprint arXiv:2603.09163, 2026.

[23] M. Wei, C. Wan, J. Peng, X. Yu, Y. Yang, D. Feng, W. Cai, C. Zhu, T. Wang, J. Pang, and X. Liu, “Ground slow, move fast: A dual-system foundation model for generalizable vision-language navigation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=GK4rznYwhn

[24] I. Team, “InternVLA-N1: An open dual-system navigation foundation model with learned latent plans,” 2025.

[25] J. Huang, J. Huang, W. Song, H. Yang, H. Huang, H. Li, and Y. Wang, “Sedualvln: A spatially-enhanced dual-system for vision-language navigation,” arXiv preprint arXiv:2605.17249, 2026.

[26] Z. Yu, Y. Long, Z. Yang, C. Zeng, H. Fan, J. Zhang, and H. Dong, “Correctnav: Self-correction flywheel empowers vision-language-action navigation model,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 22, 2026, pp. 18 737–18 745.

[27] H. Guo, S. Wang, R. Ma, S. Jiang, Y. Ghasempour, O. Abari, B. Guo, and L. Qiu, “Omnivla: Physically-grounded multimodal vla with unified multi-sensor perception for robotic manipulation,” arXiv preprint arXiv:2511.01210, 2025.

[28] C. Li, J. Wen, Y. Peng, Y. Peng, and Y. Zhu, “Pointvla: Injecting the 3d world into vision-language-action models,” IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 2506–2513, 2026.

[29] L. Sun, B. Xie, Y. Liu, H. Shi, T. Wang, and J. Cao, “Geovla: Empowering 3d representations in vision-language-action models,” arXiv preprint arXiv:2508.09071, 2025.

[30] H. Zhang, S. Li, Y. Zhang, Z. Huai, H. Chen, C. Shen, J. Gong, and X. Qiu, “Core-vla: Towards scalable and robust vision-languageaction modeling via conditional routing of experts,” arXiv preprint arXiv:2607.03693, 2026.

[31] X. Liu, J. Li, Y. Jiang, N. Sujay, Z. Yang, J. Zhang, J. Abanes, J. Zhang, and C. Feng, “Citywalker: Learning embodied urban navigation from web-scale videos,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 6875–6885.

[32] A. J. Hancock, A. Z. Ren, and A. Majumdar, “Run-time observation interventions make vision-language-action models more visually robust,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 9499–9506.

[33] S. Lee, S. Mo, and W.-S. Han, “Bring my cup! personalizing visionlanguage-action models with visual attentive prompting,” in Proceedings of the International Conference on Machine Learning (ICML), 2026.

[34] J. Chen, B. Lin, X. Liu, L. Ma, X. Liang, and K.-Y. K. Wong, “Affordances-oriented planning using foundation models for continuous vision-language navigation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 22, 2025, pp. 23 568–23 576.

[35] M. Bao, Y. Cai, H. Xu, Z. Li, J. He, J. Tang, C. Lv, J. Zhang, Y. Xie, and W. Wang, “Goal2pixel: Grounding goals to pixels for vision-language navigation,” arXiv preprint arXiv:2606.01621, 2026.

[36] A. Szvoren, D. Kanoulas, and N. Tuptuk, “Green for go, red for no: Visual grounding via semantic segmentation for vla navigation policies,” arXiv preprint arXiv:2607.05122, 2026.

[37] H. Son, K. Tan, and F. Zhang, “Physics-guided robotic radiation source localization along arbitrary measurement paths in unstructured environments,” IEEE Transactions on Automation Science and Engineering, vol. 23, pp. 14 809–14 825, 2026.

[38] H. Son and F. Zhang, “Robot platform for physics-centric multimodal perception: Vision, temperature, gas, and radiation,” in 2025 IEEE International Conference on Robotics and Automation (ICRA) Workshop on Grand Challenges in Robotics for Nuclear Environments. IEEE, 2025.