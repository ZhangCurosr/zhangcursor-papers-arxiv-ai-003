# Whole-Body Dexterity

Conor Mc Gartoll<sup>∗1,2</sup>, An Dang<sup>∗1,3</sup>, Pranay Thangeda<sup>∗1</sup>, Arjun Gupta<sup>1</sup>, Carolina Higuera<sup>1</sup>, Nima Fazeli<sup>1,3</sup>, Mustafa Mukadam<sup>†1</sup>, Manikantan Nambi<sup>†1</sup>

<sup>1</sup>Amazon, <sup>2</sup>University of California, San Diego, <sup>3</sup>University of Michigan <sup>∗</sup>Equal contribution, <sup>†</sup>Equal advising wholebodydex.com

![](images/f30093747454da6a85071dc6bf18c35f3546572bf9225ec1a37d4d5aca4d08d5.jpg)  
Fig. 1: Whole-Body Dexterity (WBD) extends humanoid dexterous manipulation beyond the hands, allowing humanoids to mirror humans’ ability to use their arms and torso as contact surfaces for holding and repositioning objects. This enables human-like behavior such as lifting a box, securing it under one arm, carrying it while walking, and setting it down.

Abstract— Current humanoid dexterous manipulation remains largely confined to end-effectors and simple grasps, falling short of humans’ ability to engage the entire body by bracing and sliding bulky objects against the arms and torso. We introduce Whole-Body Dexterity (WBD), a general framework that uses the robot’s body and appendages as contact surfaces to grasp and manipulate objects. Our sim-to-real pipeline first generates a dataset of feasible whole-body grasps, then trains reinforcement-learning experts on a general any-toany repositioning objective by sampling initial and goal grasps from the dataset, and distills these experts into a single policy that uses proprioception and object geometry. We show that the distilled policy transfers zero-shot to a real Unitree G1, where it repositions a basketball and a cardboard box to regions around the torso and completes a long-horizon task that stows a box under one arm, carries it from a shelf to a table, and places it. The controller operates interactively, steered by coarse motion references from a human teleoperator. Across real-world whole-

body manipulation tasks, the distilled controller completes 90% of trials, compared with 18% for direct retargeting and 4% for a generalist whole-body controller.

## I. INTRODUCTION

People rarely manipulate with their hands alone. We brace a box against the chest, cradle a basket across both arms, or tuck an object under one arm to free the other. These actions use the limbs and torso as a single highly dexterous manipulator. We call this ability to reposition objects by continually changing supporting body contacts whole-body dexterous manipulation and apply it directly to humanoids.

Prior humanoid work has used this term for specific settings, such as coordinated whole-body control for manipulation with the hands [1], [2], while related loco-manipulation confines dexterity to the end-effectors [3], [4], [5], [6], [7] or to a sustained arm grasp on a bulky object [8], [9], [10]. None treats it as a general capability in which contact is established and released with any link to move an object across diverse regions of the body.

![](images/3df7645568e4c347994aa34aadfbd8d1158d2b7d6895ddc978633f4c99eb64c0.jpg)  
Fig. 2: Whole-Body Dexterity Pipeline. (1) Generate a voxelized dataset of stable whole-body grasps. (2) Pre-train sphere WBD-EX experts on any-to-any repositioning using the grasp dataset. (3) Warm-start WBD-EX experts for other objects from the sphere experts. (4) Collect WBD-EX trajectories and distill them into a general dexterity controller for simulation and real-world tasks.

Viewed through this lens, whole-body dexterous manipulation has a useful parallel in dexterous hands, with the limbs and torso taking the roles of fingers and palm. Recent dexterous-hand work shows that strong capabilities emerge from learning to move objects through dense sequences of intermediate targets, whether by pre-training on anygrasp-to-anygrasp transitions [11] or by tracking hand– object subgoals from human motion through regrasps and finger gaiting [12]. Extending this principle to humanoids introduces a challenge: a dexterous hand with many fingers can often reposition some fingers while the others support the object [13], [14], whereas a humanoid using two arms has fewer independent supports. Releasing one arm may require the other to brace the object against the torso, and changing grasps can require rapid reconfiguration and rolling or sliding contact. This additional complexity calls for a different approach to repositioning across body regions than these existing methods.

We introduce Whole-Body Dexterity (WBD), a framework for learning whole-body dexterous manipulation through mode-constrained re-grasping (Fig. 2) and deploying it on a real humanoid. We first generate a diverse set of whole-body grasps, then use them to initialize reinforcementlearning experts, WBD-EX. Experts learn to move objects between grasps toward goals specified by an object position and a target grasp mode: a left-arm, both-arm, or right-arm grasp. Transitions allow brief contact loss during movement but require a stable hold in the target mode at goals, so the experts learn how to hand off objects between arms, enabling repositioning across difficult regions such as behind the back and on top of the shoulders. We distill expert trajectories into a controller that uses an object geometry descriptor to refine motion commands into stable wholebody object repositioning, without privileged object state.

Our main contributions are:

1) A whole-body grasp generation method defined over mode-typed contact configurations, formulated for arbitrary open-link articulated rigid-body systems and demonstrated on a humanoid robot.

2) WBD-EX, per-object repositioning experts built on a mode-constrained re-grasping formulation that uses the diverse grasp dataset.

3) A distillation procedure that turns these experts into a controller conditioned on proprioception and object geometry, refining coarse commands into dexterous manipulation of diverse objects in simulation and on a real humanoid.

## II. RELATED WORK

Whole-Body Manipulation. In this paper, we advance whole-body dexterous manipulation, in which objects are held and repositioned through contact anywhere on the robot’s body rather than only at designated end-effectors or links. The closely related term, whole-body dexterity, was defined for a snake-like robot arm that manipulates with its entire body surface [18]. Related in spirit, a humanoid can manipulate from onboard proprioception alone, since jointencoder readings under compliant contact act as a coarse whole-body tactile channel [19]. Several methods manipulate bulky objects that exceed the hand’s size and payload by grasping them with the arms and torso and distributing contact across the body [8], [20], [21], [22]. Other work uses upper-body tactile sensing to hold boxes against the arms and chest on a real humanoid [9]. Closest to our setting, reinforcement learning on a large soft robotic torso lifts cuboids of varying sizes and masses and performs recovery re-grasping in front of the torso [23], but without standing or walking and without moving the object to other regions of the body.

Classical model-based work formulates this as optimiza tion over contacts placed anywhere on the body surface [24], [25] and plans holding manipulation as a transition graph over object motion and contact states [26], though typi cally under planar, quasi-static, or box-shaped assumptions; recent learning-based methods instead expose whole-body contact as an explicit, commandable interface [27] or imitate whole-body human-object interactions in simulation from motion-capture demonstrations [28], [29]. Evidence that the hand and the humanoid form a structural continuum comes from reconfigurable hardware [30] and from commercial humanoids that already tuck containers under one arm, drape cloth over a shoulder, and lift appliances during everyday tasks [31], [32]. Humanoid loco-manipulation couples locomotion with manipulation, enabling dynamic behaviors such as climbing and wall flips alongside picking, placing, and pushing objects [5], [4], [33], [34], [35], including humanoid pushing in simulation [36]. We instead focus on general object manipulation across body regions.

![](images/2cc207907e37553b98a3ef1d12e407a0a06d09b8becddcca62734e3f1bfccd01.jpg)  
Fig. 3: Whole-Body Grasp Generation. Energy-based optimization [15], [16], [17] generates grasp candidates from specified arm, leg, and torso contact links (blue), followed by simulation validation under perturbations. The voxelized training set contains left-arm, right-arm, and both-arm grasps; leg grasps are excluded.

Teleoperation is a direct route to whole-body manipulation and is well developed as motion transfer, through realtime human-to-humanoid retargeting and shadowing [37], [38], [39], [40] and tracking controllers that follow wholebody targets [41], [42], [3], [43]. However, without force or tactile feedback, it is difficult for the operator to tell whether an object is pinned or slipping, which largely limits teleoperation to hand-only pick-and-place behavior.

Dexterity as Contact Sequencing. In-hand manipulation can be framed as re-grasping: an object is carried through a sequence of grasps by repeatedly breaking and remaking finger contacts, a process known as finger gaiting [14], [13], [44]. Using this idea, prior work trains dexterous-hand controllers that high-level policies or teleoperators steer with motion commands [11], [12]. Related goal-conditioned approaches learn grasping and reorientation across procedurally generated tools [45] or pre-train object reposing before adapting to downstream manipulation tasks [46]. Similar in mechanism to our own approach, a constrained sampler over feasible contact-rich states can guide goal-conditioned manipulation RL directly [47]. We lift this re-grasping principle from dexterous hands and arms to the whole humanoid.

## III. METHODOLOGY

We present a pipeline for whole-body dexterous manipulation, enabling object repositioning across diverse regions of the body. Beyond end-effector and front-of-torso contact, we define contact surfaces for grasping as any link or link subsection in the robot’s kinematic system. Because the embodiment for whole-body grasping differs greatly from a multi-fingered hand, much of the re-grasping in the wholebody formulation involves sliding and rolling across the torso mesh, which we account for later in this section.

Our method trains and deploys controllers with dexterous whole-body re-grasping capabilities in three stages. We first generate a large dataset of whole-body object grasps using energy-based optimization and decompose our torso mesh into subregions to enable diverse whole-body grasping on the robot. Second, we train per-object experts on this grasp corpus to reposition objects around the body, encouraging sliding and rolling contact across the torso. Finally, we distill these experts into deployable controllers without privileged object state, enabling real-world whole-body dexterous object repositioning.

## A. Whole-Body Grasp Generation

Given a kinematic system, we enumerate configuration types (Fig. 3): which bodies or appendages hold the object and which of their links or subregions should contact it. A grasp mode m ∈ M abstracts a configuration type to the appendages contacting the object and is the level at which goals are commanded in Section III-B. The system model, an object mesh, and the choice of contact links are the only inputs, so the procedure is not specific to a humanoid or to a class of objects.

A large body such as the G1’s torso is contactable from every side, so optimizing against it as one contact target both concentrates contacts wherever they are easiest to find and collapses the distinct directions from which the object can be braced. We therefore subdivide large links into separate contact targets and generate grasps against each independently, which spreads the resulting grasps over the full extent of the body. On the G1, this subdivision yields named torso regions (Fig. 7), which the downstream training reuses to name goals, but the mechanism is generic. We showcase a variety of whole-body grasps across arms, legs, and the torso in Fig. 3. Leg grasps would require learning manipulation jointly with balance and locomotion, so we use only arm and torso grasps for WBD-EX training, giving the mode set $\mathcal { M } = \{ L , B , R \}$ : left-arm, both-arm, and right-arm grasps. In a single-arm mode, the other arm must release at the goal, and in mode $B ,$ both arms must contact it.

After defining our desired contact links and body subregions, we minimize the differentiable force-closure residual [15] over the object’s signed distance field, drawing contacts onto the object surface and keeping links out of the object and each other. Candidates are initialized from randomized object placements and optimized in GPU batches as done in large-scale dexterous hand grasp dataset generation [16], [17], [48].

Because the G1 has only two arms and a large torso to grasp against, its grasps depend heavily on gravity and contact-link friction. Therefore, to produce the highestquality grasp dataset, we aggressively filter candidates in simulation. For each candidate, the object is pressed into contact and held under $4 \times$ gravity while a sustained wrench disturbance is swept over the six axis directions. The candidate passes only if contact persists. Survivors are then evaluated again under standard gravity without disturbance and must drift less than 10 cm and rotate less than 45<sup>◦</sup>, producing the settled pose used at WBD-EX episode initialization. We index accepted grasps by robot joint position, grasp mode, and settled object position, and drop grasps whose object mesh sits farther than 5 mm from the torso surface or below a size-dependent height threshold (Appendix II-D). This concentrates the set on stable holds across the body, giving WBD-EX higher-quality grasps to transition between.

## B. WBD-EX Training

We train WBD-EX to continually transition between object positions and grasp goals using proximal policy opti mization (PPO) [49]. We formulate this problem as a Goal-Conditioned Markov Decision Process (gcMDP) defined by the tuple $( S , G , { \mathcal { A } } , { \mathcal { R } } , { \mathcal { P } } , \rho _ { 0 } , \gamma )$ , where S is the state space, $G$ is the goal set, A is the action space, $\mathcal { R }$ is the reward function, $\mathcal { P }$ is the transition probability, $\rho _ { 0 }$ is the initial state distribution, and $\gamma$ is the discount factor. The robot seeks a policy $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } \mid \mathbf { s } , \mathbf { g } )$ , parameterized by θ, that maximizes the expected discounted return $\begin{array} { r } { \mathcal { I } = \mathbb { E } _ { \pi _ { \theta } } \Big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t } \Big ] } \end{array}$ over control steps t, where $\mathbf { s } ~ \in ~ \mathcal { S } , ~ \mathbf { g } ~ \in ~ G , ~ \mathbf { a } ~ \in ~ \mathcal { A }$ , and $r _ { t } = \mathcal { R } ( \mathbf { s } _ { t } , \mathbf { a } _ { t } , \mathbf { g } )$

Given the grasp dataset $\boldsymbol { \mathcal { D } } = \{ ( q _ { i } , \mathbf { X } _ { i } ^ { O } ) \} _ { i = 1 } ^ { N _ { g } }$ for $N _ { g }$ grasps of an object, where $q _ { i }$ is the settled robot joint configuration and $\mathbf { X } _ { i } ^ { O }$ is the settled object pose (position and orientation) expressed in the torso frame, we voxelize grasps at 2 cm resolution by settled object position and group each voxel’s grasps by grasp mode. Each occupied (voxel, mode) pair is a node $n \in \mathcal V$ of a goal graph, and $\mathcal { D } ( n ) \subseteq \mathcal { D }$ denotes the grasps in that voxel with that mode. A transition between nodes is legal only if their modes are compatible: L and R each connect to themselves and to B, but never to each other, so every handoff passes through a both-arm hold.

We initialize each episode from the grasp dataset by sampling a mode uniformly, then a voxel carrying that mode uniformly, then $g ^ { \mathrm { s t a r t } } \sim \mathcal { U } ( \mathcal { D } ( n ^ { \mathrm { s t a r t } } ) )$ , where U denotes the uniform distribution. Goals are commanded toward a far goal whose mode, then voxel, are drawn uniformly. Each commanded goal advances toward it: we draw a goal distance $d \sim \mathcal { U } ( 0 . 0 2 , d _ { \mathrm { m a x } } )$ in meters and command the legal-mode node within d of the current node that is closest to the far goal, excluding the current node, with $g ^ { \mathrm { g o a l } } \sim \mathcal { U } ( \mathcal { D } ( n ^ { \mathrm { g o a l } } ) )$ A new far goal is drawn on arrival. With probability $0 . 2 ,$ the command is instead a $h o l d ,$ which asks the policy to keep the object at its current position and mode. To encourage humanlike repositioning, in which objects roll and slide across body regions between arm handoffs, success depends only on the object’s position, its grasp mode, and the object coming to rest there. Unlike [11], our goals and rewards exclude the goal joint configuration and object orientation.

We apply a goal-distance curriculum by ramping the ceiling $d _ { \mathrm { m a x } }$ from 2 cm to 8 cm. Because each goal distance is drawn from $[ 2 \mathrm { c m } , d _ { \mathrm { m a x } } ]$ , increasing $d _ { \mathrm { m a x } }$ introduces progressively farther goal transitions while retaining closer ones. This exposes the policy to both gradual and more aggressive object repositioning behaviors as training progresses.

A goal is completed only when the object settles at the goal under the correct grasp mode (Appendix III-C). For a single-arm goal, the inactive arm must register no object contact at the goal position. Requiring the inactive arm to fully release pushes the policy to learn stable and complete arm-to-arm handoffs around the body. We impose objectcontact requirements throughout, but permit brief loss of the required arm contact within a 0.48 s grace window so the policy can execute the fast, dynamic reconfigurations that whole-body re-grasping demands rather than being forced into purely quasi-static holds.

Rewards. A goal is specified only by an object position and a target grasp mode (L, B, or R), so the policy is free to reach the target however it can. Retention $\rho _ { t } ~ \in ~ \{ 0 , 1 \}$ indicates that the object has not dropped and keeps the arm contacts required by its current or target grasp mode; $e _ { t }$ is the distance to the commanded position; and the inactive arm is clear when it has no object contact. The reward is a weighted sum grouped as:

$$
r = \underbrace { r _ { \mathrm { p r o g } } + r _ { \mathrm { s e t t l e } } + r _ { \mathrm { s u c c } } + r _ { \mathrm { h o l d } } } _ { \mathrm { g o a l } } + r _ { \mathrm { n e u t r a l } } - r _ { \mathrm { f a i l } } - r _ { \mathrm { r e g } }
$$

Progress, hold quality, and arm neutrality all carry the factor $\rho _ { t }$ and are therefore zero whenever retention is lost.

• Progress $r _ { \mathrm { p r o g } } = \rho _ { t } \mathrm { c l i p } ( e _ { t - 1 } - e _ { t } , \pm 2 \mathrm { c m } )$ . The per-step reduction in distance to the commanded position, rewarded only while the object is retained and clipped so that no

single step dominates.

• Goal-region arrival $r _ { \mathrm { s e t t l e } } .$ . A small per-step reward on movement goals during the dwell period, requiring the object to be settled within a 2 cm goal region in the required grasp mode.

• Transition completion $r _ { \mathrm { s u c c } } .$ . A sparse success bonus for repositioning the object and settling it at the goal under the required grasp mode.

• Hold quality $r _ { \mathrm { h o l d } } = \rho _ { t } \exp ( - e _ { t } / 2 . 5 \mathrm { c m } )$ on hold goals, which command the object to stay where it is. It rewards proximity to the hold position while the object is retained.

• Arm neutrality $r _ { \mathrm { n e u t r a l } } = \rho _ { t } \exp ( - \Vert q _ { t } ^ { \mathrm { a r m } } - q _ { \mathrm { d e f } } ^ { \mathrm { a r m } } \Vert / 0 . 4 )$ Attracts the arms toward their default configuration while the object stays held, so an arm not needed for the current grasp returns to its rest pose.

• Failure penalty $r _ { \mathrm { f a i l } }$ . A single terminal penalty for a drop, sustained loss of object contact, or another physical or numerical failure (Appendix III-B).

• Regularizers $r _ { \mathrm { r e g } }$ . Five penalties for unsupported motion, excessive object speed, large actions, changes between actions, and joint-target tracking errors (Appendix III-D).

Domain randomization. We train an individual expert per object over a diverse set of primitives (spheres, cubes, rectangular prisms, and cylinders). To make holding and transitions robust to unmodeled dynamics, we randomize object mass, center of mass, and contact friction, and apply random perturbation forces to the object (Appendix III-E). This drives the policy toward grasps and handoffs that stay stable under a range of physical conditions.

Expert evaluation. We assess WBD-EX expert quality by regional route completion, the fraction of trials that reach every goal of a route in order before failure or timeout, where each route specifies a sequence of grasp goals across body regions. Route definitions, completion requirements, and evaluation details are given in Appendix IV. Expert evaluation omits external force perturbations but still applies object-mass, friction, and center-of-mass domain randomization.

Two-stage WBD-EX training. When trained from scratch, experts for smaller and less uniformly shaped objects achieve lower regional route completion than experts for mediumlarge spheres. To reduce the time needed to train a WBD-EX expert for a new object, we pre-train medium-large sphere experts, which provide an effective initialization for the other primitives. We then warm-start additional object primitives from these pre-trained experts with an identical reward formulation. Warm-started experts achieve comparable or higher regional route completion than from-scratch experts at lower amortized GPU cost by sharing the faster sphere pre-training across target objects (Appendix V).

## C. WBD Controller Distillation

WBD-EX policies generate diverse and continuous object repositioning trajectories covering contactable regions of the torso, head, and arms under randomized dynamics. We filter the trajectories to keep only valid repositioning trajectories, then balance sampling across objects and motion types with cross-body mirroring. We also sample three trajectory speed behaviors equally, using goal distances of 2–4, 4–6, and 6– 8 cm that are fixed within each trajectory, to capture both slow, deliberate and faster, dynamic repositioning behavior. See Appendix VII for additional details.

These trajectories are used to train a WBD controller combining a conditional diffusion model with an inversedynamics network [11]. Given proprioceptive history and a five-dimensional geometry descriptor comprising log-sorted bounding-box side lengths, log volume, and sphericity, it generates keypoint motions and converts them into jointtarget increments. For real objects, we compute this descriptor from measured dimensions. At inference, coarse commands guide generation through gradient guidance [11]. A tunable guidance weight controls how strongly denoising is steered toward the coarse motion command: larger weights emphasize command tracking, while smaller weights give greater relative influence to the learned motion prior. At each update, the diffusion model samples several candidate motions in parallel; we select the one closest to the command and convert it to refined joint commands with the inversedynamics model. This requires no privileged per-step object state and supports human teleoperation with tunable guidance.

## IV. EXPERIMENTS AND RESULTS

We evaluate whether WBD can be used to generate wholebody dexterous data and train policies that refine imperfect motion commands into stable whole-body dexterous manipulation. We first report grasp and WBD-EX data generation, then evaluate a distilled WBD controller for on-body repositioning in simulation and on a real Unitree G1. A shelf pick, stow, carry, and place task tests whether these skills compose into a longer manipulation sequence.

## A. Experimental Setup

Our object set contains 15 primitives: six spheres, four cubes, three rectangular prisms, and two cylinders. All side lengths and diameters fall between 80 and 240 mm. WBD-EX uses 2.51 million simulation-stable grasps passing the dataset filters (Section III-A). Mean per-object coverage of sampled object-center positions around the torso is 90.1% using each object’s minimum bounding-box half-extent as the distance tolerance; pooled coverage is 98.4% using the mean minimum half-extent (Appendix II-D).

We train one WBD-EX policy per object using the procedure in Section III-B and distill a shared WBD controller from their trajectories. Expert proficiency varies with object geometry: sphere experts achieve the highest overall route completion, while experts for cubes, prisms, and cylinders still reposition across all body regions but with lower overall success rates (Appendix IV-D). Because distillation uses only trajectory segments that pass a quality filter (Appendix VII-B), less proficient experts yield valid demonstrations less efficiently, requiring more collection attempts per accepted trajectory. Our training dataset consists of 274 thousand

Training objects (mean)

![](images/832aa06e6211f9a2cff286c20874f29ff3237c86c12997b3bbf932bc0e28c4fb.jpg)

![](images/16862028f7fb5a52de795bf46bb6e6643eda076d97fa8bb50096bd68aed875a1.jpg)

![](images/8d4840489bdaf5cf39f2fead1e49079c1185f805c7c876504721b10b25b94cca.jpg)

![](images/9e491960dcdf4f5e07615c344994daf33a34539aa15c0a8d410284d6391bfeaa.jpg)

![](images/a49d8335f041f207350d0c108e92be3e0a7415afe4cee40005d890b7ee3ff7e8.jpg)

![](images/88e25ab9588c322102da718f64345b8262749b2ba463ea91ed8416f8af4f2abb.jpg)

![](images/1f5dbc160969261fd1b97defc22686b0b23eaecffc4019442a6ab72429e198fc.jpg)

![](images/21c0471434af09cd51b2c3505344e8e897ada0cbbf5ecaf1428215a7f994565e.jpg)  
Fig. 4: Simulated object repositioning under noisy commands. Trials sample transfers among six body regions (Section IV-B). Training-object mean (top) and held-out sizes (bottom). Solid, dashed, and dotted lines show WBD refinement averaged over three training seeds, noisy experts, and clean experts, respectively. Bands denote pointwise 95% paired bootstrap confidence intervals clustered by source grasp over 256 matched trials per object, with training objects weighted equally. Trials are capped at 600 s.

trajectories and 873 million recorded frames; training samples objects equally. Network architectures, optimization settings, data augmentation, and training costs are detailed in Appendices III, V-C and VII-A.

## B. Simulation Experiments

We evaluate on-body repositioning in simulation using a WBD controller trained on 13 of the 15 primitives (three training seeds), with a 180 mm sphere and a 180 mm cube withheld from training, validation, and normalization. We degrade each of the 15 objects’ WBD-EX policies into coarse command streams and compare the degraded expert alone against the WBD controller refining it. We add independent uniform noise $\mathcal { U } ( - \alpha , \alpha )$ to all 14 Unitree G1 arm commands. We report results at $\alpha ~ \in ~ \{ 2 . 5 , 4 \}$ and guidance weights {1, 10, 20, 100, 350}, with clean experts as a reference (Fig. 4). WBD applies bounded corrections to the noisy commands, within ±2.5 of each noisy command in the experts’ pre-tanh action space (Appendix VII-F). Degrading an expert reproduces the failure we see with real operator commands: the stream still carries the goal direction while individual arm targets move far enough to break supporting contact mid-handoff. In both settings, the failure is a loss of supporting contact followed by a drop.

We evaluate transfers among six body regions (Fig. 5): front, both underarms, both shoulders, and back. Each trial moves the object through a seeded random sequence of adjacent regions (Appendix VII-D). The WBD controller is trained across all regions and voxels around the body; see Appendix VII-C. Each underarm and shoulder goal requires

![](images/3f28c4267788ffc5fdc640a33da0e0fdc0ce567c37d3d78ab2c3d07364f69ee7.jpg)

Fig. 5: Simulated WBD controller experiment. Examples of regional transfers in which the start grasp is shown on the fully visible G1, while the sampled far goal grasp in another region is shown as the blue ghost.

the opposite arm to release, with both arms supporting the object at the front and back. Each condition uses 256 matched initial grasps per object spanning the six regions; initial states and random seeds are matched across conditions. Trials end at the first failure or 600 s. We report completed goals per trial and time to failure restricted to this horizon (Fig. 4). Each goal is a (voxel, grasp mode) pair sampled from the voxel goal graph along the path toward a randomly drawn far goal in the next region of the sequence;

![](images/392c877e0b143a5d95a624a262958f9c938bfafacdd642c1b6192fd62547fb13.jpg)  
Fig. 6: Real-world whole-body dexterous manipulation. On-body repositioning (bottom) and shelf pickup, stowing, and carrying (top) using the WBD controller, guided by the teleoperator (right).

far goals are included as goals. Tracking the number of goals achieved is an effective measure of how well a policy can stably transition an object across regions of the body.

Despite noisy commands, WBD largely recovers the clean experts’ mean goal-reaching performance across training objects at both plotted noise levels (Fig. 4). Averaged over these two noise levels and all five guidance values, mean time to failure increases relative to the noisy expert by 70% on training objects and by 158% and 71% on the held-out sphere and cube, respectively. Training objects are weighted equally. At the lower noise level and guidance weights of 1–20, WBD even exceeds the clean experts’ time to failure, by 25–35% for the training-object mean and by 109–120% and 25–47% for the held-out sphere and cube, respectively, showing that the WBD controller maintains stable object holding beyond what the uncorrupted experts achieve. As Fig. 4 shows, excessive command tracking can reduce both stability and goals per trial.

## C. Real-World Experiments

We evaluate a WBD controller trained on the full 15- object WBD-EX trajectory dataset on two on-body repositioning tasks and a long-horizon shelf task (Fig. 6). For repositioning, a size 3 basketball or a cardboard box (15 cm cube) starts in a two-handed front grasp. The shoulder tap task succeeds when the object contacts the shoulder link, and the underarm stow task succeeds when the object is held stably under one arm, with its center approximately as far forward as the shoulder link. A round trip additionally requires returning to a stable two-arm front grasp. We report region-reaching success, full round-trip success, and mean time to each endpoint over successful trials (Table I). The shelf task requires picking the same cardboard box from a shelf, stowing it under one arm, stably carrying it while walking, un-stowing it, and placing it on a table. Pick and unstow require a stable two-arm front grasp, and stow uses the underarm criterion, except that the free arm may stabilize the box while walking. We report success counts for each subtask: pick, stow, carry, un-stow, and place (Table II).

All trials are run by a single skilled human teleoperator with 8+ hours of training on these tasks. We use an Xsens motion capture suit to track motion beside and behind the operator, where standard VR headset tracking is limited. The controller can also accept commands from VR headset and trackers, although limited tracking coverage can restrict dexterity. The operator indicates the desired motion, while WBD refines the arm targets to maintain object support as the arms and torso reconfigure. For the long-horizon task, the operator switches the guidance weight between 10 for on-body repositioning and 100 for pick-and-place, balancing stable support with responsive command following.

We compare WBD refinement of motion retargeted by General Motion Retargeting (GMR) [50] against direct GMR control and SONIC [43]. For WBD and direct GMR, GMR retargets the Xsens motion to G1 upper-body targets, and Adaptive Motion Optimization (AMO) [3] controls the lower body and balance. SONIC runs its public checkpoint in VR three-point mode, tracking Xsens head and wrist poses while its kinematic planner controls the lower body. The real-world results suggest that maintaining object support through contact transitions is a central bottleneck (Tables I and II). GMR let the operator move the arms toward the relevant body regions, but stable contact during repositioning remained difficult; SONIC additionally struggled with reachability in constrained regions. Across the three tasks, counting complete on-body round trips for both objects and full shelf sequences, the WBD controller completed 45 of 50 trials (90%), compared with 9 of 50 (18%) for GMR and 2 of 50 (4%) for SONIC.

In simulation, cube WBD-EX experts have lower mean regional route completion than sphere experts (Table VIII). In the real-world shoulder tap trials, however, GMR completed 4/10 box round trips versus 1/10 with the basketball, with mean completion times of 8.5 versus 30.1 s among successful trials. We hypothesize that the cardboard’s lower friction makes it easier to slide to the shoulder, whereas the basketball’s higher friction requires rolling along the torso and more deliberate arm manipulation, consistent with the faster successful WBD and GMR box trials. Underarm stowing still remained difficult, requiring a complete single-arm handoff, and yielded low success rates for GMR and SONIC (1/10 and $0 / 1 0$ complete box round trips, respectively, compared with 10/10 for WBD).

<table><tr><td rowspan="2">Task</td><td rowspan="2">Controller</td><td colspan="4">Basketball</td><td colspan="4">Cardboard box</td></tr><tr><td>Reached ↑</td><td>Round trip ↑</td><td>Region (s) ↓</td><td>Total (s) ↓</td><td>Reached ↑</td><td>Round trip ↑</td><td>Region (s) ↓</td><td>Total (s) ↓</td></tr><tr><td rowspan="3">Shoulder tap</td><td>SONIC</td><td>2/10</td><td>0/10</td><td>15.2</td><td></td><td>2/10</td><td>2/10</td><td>15.9</td><td>24.9</td></tr><tr><td>GMR</td><td>3/10</td><td>1/10</td><td>26.0</td><td>30.1</td><td>6/10</td><td>4/10</td><td>8.0</td><td>8.5</td></tr><tr><td>WBD (ours)</td><td>10/10</td><td>9/10</td><td>7.7</td><td>17.3</td><td>10/10</td><td>10/10</td><td>4.1</td><td>6.3</td></tr><tr><td rowspan="3">Underarm stow</td><td>SONIC</td><td>1/10</td><td>0/10</td><td>28.0</td><td></td><td>2/10</td><td>0/10</td><td>26.1</td><td></td></tr><tr><td>GMR</td><td>4/10</td><td>2/10</td><td>31.5</td><td>49.8</td><td>2/10</td><td>1/10</td><td>9.3</td><td>17.8</td></tr><tr><td>WBD (ours)</td><td>10/10</td><td>9/10</td><td>10.6</td><td>26.2</td><td>10/10</td><td>10/10</td><td>8.6</td><td>13.0</td></tr></table>

TABLE I: Real-world on-body repositioning. Ten trials per controller, object, and task, five per side, with a size 3 basketball and a cardboard box (15 cm cube): region-reaching and full round-trip success. Times are measured from trial start and averaged over trials reaching the respective endpoint.

<table><tr><td>Controller</td><td>Pick ↑</td><td>Stow ↑</td><td>Carry ↑</td><td>Un-stow ↑</td><td>Place ↑</td></tr><tr><td>SONIC</td><td>10/10</td><td>0/10</td><td>0/10</td><td>0/10</td><td>0/10</td></tr><tr><td>GMR</td><td>10/10</td><td>3/10</td><td>2/10</td><td>1/10</td><td>1/10</td></tr><tr><td>WBD (ours)</td><td>10/10</td><td>9/10</td><td>9/10</td><td>9/10</td><td>7/10</td></tr></table>

TABLE II: Real-world shelf task. Completed stages over ten trials per controller with the cardboard box.

On the long-horizon task (Table II), all methods completed the initial shelf pickup, but subsequent stowing and transport required support to persist as the robot reconfigured and moved. The WBD controller maintained support through carrying and un-stowing after successful stows, showing that the controller can stably hold objects while being perturbed. While substantially better than GMR and SONIC, WBD still lost contact in a few trials, dropping the object.

## V. CONCLUSION

We presented a pipeline for learning whole-body dexterous manipulation at scale: contact configurations are enumerated into mode-typed whole-body grasps, and those grasps seed large-scale reinforcement learning under common rewards to dexterously reposition objects around the body. We showed how to use this large-scale whole-body dexterous manipulation data to train a shared controller that refines a coarse command stream into stable control and supports teleoperation. Although WBD-EX is trained without a lower-body controller, the distilled policy deploys on a standing and walking G1, holding objects through leaning and walking during the shelf task. Future work can naturally extend the same formulation to leg contact: generating combinations of arm and leg grasps, training WBD-EX jointly with a whole-body controller across this expanded grasp dataset, and distilling it into a policy to enable behaviors such as bracing a heavy box on a knee while the arms reposition.

## REFERENCES

[1] Y. Ze, S. Zhao, W. Wang, A. Kanazawa, R. Duan, P. Abbeel, G. Shi, J. Wu, and C. K. Liu, “TWIST2: Scalable, portable, and holistic humanoid data collection system,” in IEEE International Conference on Robotics and Automation (ICRA), 2026, pp. 15 900–15 907.

[2] L. Heng, Y. Tang, J. Xu, H. Bao, D. Huang, and Y. Wang, “HumDex: Humanoid dexterous manipulation made easy,” arXiv preprint arXiv:2603.12260, 2026.

[3] J. Li, X. Cheng, T. Huang, S. Yang, R.-Z. Qiu, and X. Wang, “AMO: Adaptive motion optimization for hyper-dexterous humanoid wholebody control,” in Proceedings of Robotics: Science and Systems (RSS), Los Angeles, CA, USA, June 2025.

[4] Y. Zhang, Y. Yuan, P. Gurunath, I. Gupta, S. Omidshafiei, A.-a. Aghamohammadi, M. Vazquez-Chanlatte, L. Pedersen, T. He, and G. Shi, “FALCON: Learning force-adaptive humanoid loco-manipulation,” in Proceedings of the 8th Annual Learning for Dynamics and Control Conference (L4DC), ser. PMLR, vol. 331, 2026, pp. 265–281.

[5] X. He, S. Xu, X. Li, R. Dong, L. Bian, Y.-X. Wang, and L.-Y. Gui, “ULTRA: Unified multimodal control for autonomous humanoid whole-body loco-manipulation,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

[6] T. He, Z. Wang, H. Xue, Q. Ben, Z. Luo, W. Xiao, Y. Yuan, X. Da, F. Castaneda, S. Sastry, C. Liu, G. Shi, L. Fan, and Y. Zhu, “VIRAL:˜ Visual sim-to-real at scale for humanoid loco-manipulation,” arXiv preprint arXiv:2511.15200, 2025.

[7] S. Wei, H. Jing, B. Li, Z. Zhao, J. Mao, Z. Ni, S. He, S. Zang, X. Liu, K. Kang, J. Liu, W. Yuan, M. Pavone, D. Huang, and Y. Wang, “Ψ<sub>0</sub>: An open foundation model towards universal humanoid locomanipulation,” in Robotics: Science and Systems (RSS), 2026.

[8] C. Zheng, K. Chen, Z. Bi, Y. Li, L. Pan, J. Zhou, H. Li, and J. Ma, “Embracing bulky objects with humanoid robots: Wholebody manipulation with reinforcement learning,” in IEEE International Conference on Robotics and Automation (ICRA), 2026, pp. 6089– 6096.

[9] M. Murooka, T. Hoshi, K. Fukumitsu, S. Masuda, M. Hamze, T. Sasaki, M. Morisawa, and E. Yoshida, “TACT: Humanoid wholebody contact manipulation through deep imitation learning with tactile modality,” IEEE Robotics and Automation Letters, vol. 10, no. 8, pp. 7819–7826, Aug. 2025.

[10] J. Dao, H. Duan, and A. Fern, “Sim-to-real learning for humanoid box loco-manipulation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 16 930–16 936.

[11] Z.-H. Yin, C. Wang, L. Pineda, F. R. Hogan, C. K. Bodduluri, A. Sharma, P. Lancaster, I. Prasad, M. Kalakrishnan, J. Malik, M. Lambeta, T. Wu, P. Abbeel, and M. Mukadam, “DexterityGen: Foundation controller for unprecedented dexterity,” in Robotics: Science and Systems (RSS), 2025.

[12] P. Li, Z. Chen, Y. Wu, P. Wei, Y. Li, T. Wang, J. Shi, M. Yu, B. Jia, S.-C. Zhu, T. Liu, and S. Huang, “Towards human-level dexterous teleoperation,” arXiv preprint arXiv:2607.11481, 2026.

[13] H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik, “In-hand object rotation via rapid motor adaptation,” in Conference on Robot Learning, ser. PMLR, vol. 205. PMLR, 2023, pp. 1722–1732.

[14] OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz,´ B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al., “Learning dexterous in-hand manipulation,” The International Journal of Robotics Research, vol. 39, no. 1, pp. 3–20, 2020.

[15] T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S.-C. Zhu, “Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator,” IEEE Robotics and Automation Letters, vol. 7, no. 1, pp. 470–477, 2022.

[16] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang, “Dex-GraspNet: A large-scale robotic dexterous grasp dataset for general objects based on simulation,” in IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 11 359–11 366.

[17] J. Chen, Y. Ke, and H. Wang, “BODex: Scalable and efficient robotic dexterous grasp synthesis using bilevel optimization,” in IEEE International Conference on Robotics and Automation (ICRA), 2025.

[18] X. Xu, D. Bauer, and S. Song, “RoboPanoptes: The all-seeing robot with whole-body dexterity,” in Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[19] A. Bhatt, O. Kaidanov, P. Liu, and J. Peters, “Blind dexterity: Wholebody humanoid manipulation via pure proprioception,” arXiv preprint arXiv:2608.29487, 2026.

[20] M. Zhang, J. Barreiros, and A. O. Onol, “Plan-guided reinforcement learning for whole-body manipulation,” 2023.

[21] J. A. Barreiros, A. O.<sup>¨</sup> Onol, M. Zhang, S. Creasey, A. Goncalves,<sup>¨</sup> A. Beaulieu, A. Bhat, K. M. Tsui, and A. Alspach, “Learning contactrich whole-body manipulation with example-guided reinforcement learning,” Science Robotics, vol. 10, no. 105, p. eads6790, 2025.

[22] S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan, “ResMimic: From general motion tracking to humanoid whole-body loco-manipulation via residual learning,” 2025.

[23] C. C. Johnson, C. Alessi, E. Falotico, and M. D. Killpack, “Zeroshot whole-body manipulation with a large-scale soft robotic torso via guided reinforcement learning,” arXiv preprint arXiv:2509.23556, 2025.

[24] V. Leve, J. Moura, N. Saito, S. Tonneau, and S. Vijayakumar, “Explicit´ contact optimization in whole-body contact-rich manipulation,” in 2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids). IEEE, Nov. 2024, p. 451–458.

[25] M. Murooka, K. Okada, and M. Inaba, “Optimization-based posture generation for whole-body contact motion by contact point search on the body surface,” IEEE Robotics and Automation Letters, vol. 5, no. 2, p. 2905–2912, Apr. 2020.

[26] M. Murooka, Y. Inagaki, R. Ueda, S. Nozawa, Y. Kakiuchi, K. Okada, and M. Inaba, “Whole-body holding manipulation by humanoid robot based on transition graph of object motion and contact,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2015, pp. 3950–3955.

[27] X. Li, X. He, R. Dong, and S. Gupta, “ContactMimic: Humanoid object interaction via contact control,” arXiv preprint arXiv:2607.08742, 2026.

[28] S. Xu, H. Y. Ling, Y.-X. Wang, and L.-Y. Gui, “InterMimic: Towards universal whole-body control for physics-based human-object interactions,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 12 266–12 277.

[29] L. Cao, X. Wu, J. Cui, B. Xu, M. Pei, R. Chen, and M. Xu, “Weave: Learning whole-body dexterous loco-manipulation from human-object interactions,” arXiv preprint arXiv:2609.16683, 2026.

[30] R. Li, C. Ma, S. Li, Z. Wei, Y. Yao, H. Shi, C. K. Liu, S. Song, and M. Ding, “Handroid: Bridging dexterous hand and humanoid,” arXiv preprint arXiv:2607.16187, 2026.

[31] Boston Dynamics, “Training a humanoid robot for hard work,” https:// bostondynamics.com/blog/training-a-humanoid-robot-for-hard-work/, 2026.

[32] Figure AI, “Helix 02: Living room tidy,” https://www.figure.ai/news/ helix-02-living-room-tidy, Mar. 2026.

[33] T. Xie, H. Zhang, J. Park, Z. Wang, B. Wen, J. Li, X. Li, Q. Ben, H. Weng, Y. Ye, D. Minor, T. Wang, C. Jiang, S. Fidler, J. Kautz, L. Fan, Y. Zhu, Z. Luo, U. Iqbal, and Y. Yuan, “GRAIL: Generating humanoid loco-manipulation from 3D assets and video priors,” arXiv preprint arXiv:2606.05160, 2026.

[34] I. Taouil, M. Ciebielski, S. Omar, H. Zhao, A. Dai, A. M. Johnson, and M. Khadiv, “MotionDisco: Motion discovery for extreme humanoid loco-manipulation,” arXiv preprint arXiv:2606.06139, 2026.

[35] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi, “OmniRetarget: Interaction-preserving data generation for humanoid whole-body loco-manipulation and scene interaction,” in IEEE International Conference on Robotics and Automation (ICRA), 2026.

[36] J. Z. Zhang, M. Sorokin, J. Brudigam, B. Hung, S. Phillips, D. Yer-¨ shov, F. Niroui, T. Zhao, L. Fermoselle, X. Zhu, C. Cao, D. Ta, T. Pang, J. Wang, P. Culbertson, Z. Manchester, and S. Le Cleac’h,´ “Sumo: Dynamic and generalizable whole-body loco-manipulation,” arXiv preprint arXiv:2604.08508, 2026.

[37] T. He, Z. Luo, W. Xiao, C. Zhang, K. Kitani, C. Liu, and G. Shi, “Learning human-to-humanoid real-time whole-body teleoperation,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 8944–8951.

[38] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi, “OmniH2O: Universal and dexterous human-tohumanoid whole-body teleoperation and learning,” in Conference on Robot Learning (CoRL), 2024.

[39] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “HumanPlus: Humanoid shadowing and imitation from humans,” in Conference on Robot Learning (CoRL), 2024.

[40] Y. Ze, Z. Chen, J. P. Araujo, Z.-A. Cao, X. B. Peng, J. Wu, and´ C. K. Liu, “TWIST: Teleoperated whole-body imitation system,” in Conference on Robot Learning (CoRL), 2025.

[41] X. Cheng, Y. Ji, J. Chen, R. Yang, G. Yang, and X. Wang, “Expressive whole-body control for humanoid robots,” in Robotics: Science and Systems (RSS), 2024.

[42] T. He, W. Xiao, T. Lin, Z. Luo, Z. Xu, Z. Jiang, J. Kautz, C. Liu, G. Shi, X. Wang, L. Fan, and Y. Zhu, “HOVER: Versatile neural whole-body controller for humanoid robots,” in IEEE International Conference on Robotics and Automation (ICRA), 2025.

[43] Z. Luo, Y. Yuan, T. Wang, C. Li, F. Castaneda, S. Chen, Z.-A. Cao, ˜ J. Li, D. Minor, Q. Ben, J. Park, D. Sami, Z. Wang, X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu, “SONIC: Supersizing motion tracking for natural humanoid whole-body control,” Science Robotics, vol. 11, no. 117, p. eaed4592, 2026.

[44] T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal, “Visual dexterity: In-hand reorientation of novel and complex object shapes,” Science Robotics, vol. 8, no. 84, p. eadc9244, 2023.

[45] K. Kedia, T. G. W. Lum, J. Bohg, and C. K. Liu, “SimToolReal: An object-centric policy for zero-shot dexterous tool manipulation,” arXiv preprint arXiv:2602.16863, 2026.

[46] J. Lee, J. Yin, A. Rana, N. Blauch, S. Mady, M. Bhardwaj, N. Fazeli, N. Ratliff, K. V. Wyk, and A. Handa, “ADEPT: Accelerating dexterity via pre-training and post-training using reinforcement learning,” arXiv preprint arXiv:2608.19182, 2026.

[47] M. Toussaint, C. V. Braun, A. Jordana, S. Auddy, E. Cobo-Briesewitz, D. Shcherba, T. Burghoff, and J. Carpentier, “Combined constrained sampling and reinforcement learning for robotic manipulation,” arXiv preprint arXiv:2602.08557, 2026.

[48] J. Ye, K. Wang, C. Yuan, R. Yang, Y. Li, J. Zhu, Y. Qin, X. Zou, and X. Wang, “Dex1B: Learning with 1B demonstrations for dexterous manipulation,” in Robotics: Science and Systems (RSS), 2025.

[49] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[50] J. P. Araujo, Y. Ze, P. Xu, J. Wu, and C. K. Liu, “Retargeting matters:´ General motion retargeting for humanoid motion tracking,” in IEEE International Conference on Robotics and Automation (ICRA), 2026, pp. 6097–6104.

[51] S. Coumar, G. Chang, N. Kodkani, and Z. Kingston, “Foam: A tool for spherical approximation of robot geometry,” 2025. [Online]. Available: https://arxiv.org/abs/2503.13704

[52] K. Zakka, Q. Liao, B. Yi, L. L. Lay, K. Sreenath, and P. Abbeel, “mjlab: A lightweight framework for GPU-accelerated robot learning,” 2026. [Online]. Available: https://arxiv.org/abs/2601.22074

[53] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, “Asymmetric actor critic for image-based robot learning,” in Proceedings of Robotics: Science and Systems, Pittsburgh, Pennsylvania, June 2018.

## APPENDIX IWHOLE-BODY GRASP GENERATION DETAILS

We adapt prior energy-based dexterous grasp synthesis [48], [16] to prescribed contact links on any combination of arms, legs, and torso, with the torso serving as a fixed support surface and kinematic reference. We seek joint configurations $q \in \mathcal { C }$ in configuration space C that minimize the resulting objective. We define K contact points $\mathbf { p } _ { j }$ on the link mesh surfaces throughout the robot that we wish to make contact with the object and, using the signed distance function of the object mesh $\phi : \mathbb { R } ^ { 3 }  \mathbb { R }$ , positive outside the object and negative inside, we formulate the following:

$$
\operatorname* { m i n } _ { q , \mathbf { T } } \quad E _ { \mathrm { c l o s u r e } } + \frac { 1 } { K } \sum _ { j = 1 } ^ { K } [ \phi ( \mathbf { p } _ { j } ) - d ^ { \star } ] _ { + } ^ { 2 }\tag{1}
$$

$$
\mathrm { s . t . } \quad \phi ( \mathbf { b } _ { i } ) \geq r _ { i } , \quad \forall i = 1 , \ldots , N\tag{2}
$$

$$
\| \mathbf { b } _ { i } - \mathbf { b } _ { l } \| _ { 2 } \geq r _ { i } + r _ { l } , \quad \forall ( i , l ) \in \mathcal { H }\tag{3}
$$

$$
\underline { { q } } _ { m } \leq q _ { m } \leq \bar { q } _ { m } , \quad \forall m = 1 , \ldots , n _ { q }\tag{4}
$$

where $E _ { \mathrm { c l o s u r e } }$ is a force-closure energy defined below, $\mathbf { T } \in S E ( 3 )$ is the pose of the robot’s kinematic root in the object frame, $\underline { { q } } _ { m }$ and $\bar { q } _ { m }$ are the limits of joint $q _ { m }$ for the $n _ { q }$ optimized joints, b<sub>i</sub> are the centers of the robot’s N collision spheres with radii $r _ { i } ,$ and H is the set of non-adjacent sphere pairs checked for self-collision. The contact-distance tolerance $d ^ { \star }$ sets the distance beyond which a contact point is attracted toward the surface. Here, $[ x ] _ { + } ~ = ~ \operatorname* { m a x } \{ x , 0 \}$ denotes the positive part of x: it returns x when positive and zero otherwise. Each squared penalty therefore applies only when its corresponding threshold is violated. We constrain the optimization problem above to avoid collisions between the link collision spheres and the object, avoid self-collision between link collision spheres, and obey joint limits.

Since this problem is nonconvex and we would like to leverage the GPU parallelism of first-order solvers such as Adam to generate many grasps simultaneously, we relax this optimization problem to the unconstrained form:

$$
\begin{array} { r l } { \displaystyle \underset { q , \mathbf { T } } { \operatorname* { m i n } } } & { \displaystyle \boldsymbol { w } _ { \mathrm { c l o s u r } } E _ { \mathrm { c l o s u r c } } + \frac { \boldsymbol { w } _ { \mathrm { d i s u } } } { K } \sum _ { j = 1 } ^ { K } [ \boldsymbol { \phi } ( \mathbf { p } _ { j } ) - \boldsymbol { d } ^ { * } ] _ { + } ^ { 2 } } \\ & { +  \boldsymbol { w } _ { \mathrm { i n k } } \sum _ { i = 1 } ^ { N } [ \boldsymbol { r } _ { i } - \boldsymbol { \phi } ( \mathbf { b } _ { i } ) ] _ { + } ^ { 2 }  } \\ & { +  \boldsymbol { w } _ { \mathrm { j o i n } } \sum _ { m = 1 } ^ { n _ { q } } ( [ \underline { { q } } _ { m } + \delta _ { q } - q _ { m } ] _ { + } ^ { 2 } + [ q _ { m } - \bar { q } _ { m } + \delta _ { q } ] _ { + } ^ { 2 } ) } \\ & { +  \boldsymbol { w } _ { \mathrm { s t f } } \sum _ { \ell , l \in \mathcal { H } } [ \boldsymbol { r } _ { i } + \boldsymbol { r } _ { l } - \| \mathbf { b } _ { i } - \mathbf { b } _ { l } \| _ { 2 } ] _ { + } ^ { 2 }  } \end{array}
$$

where we define weights w for each optimization term. The joint-limit penalty begins $\delta _ { q }$ inside each limit; joint angles are also clamped to their hard limits after each update. Table III lists the WBD grasp-generation settings.

We use the following differentiable force-closure energy,

adopted from prior work on dexterous hand grasp synthesis:

$$
E _ { \mathrm { c l o s u r e } } = { \frac { 1 } { K ^ { 2 } } } \left( \left\| \sum _ { k = 1 } ^ { K } \mathbf { n } _ { k } \right\| _ { 2 } ^ { 2 } + \left\| \sum _ { k = 1 } ^ { K } ( \mathbf { p } _ { k } - { \bar { \mathbf { p } } } ) \times \mathbf { n } _ { k } \right\| _ { 2 } ^ { 2 } \right) ,
$$

where $\mathbf { n } _ { k } = - \nabla \phi ( \mathbf { p } _ { k } ) / \| \nabla \phi ( \mathbf { p } _ { k } ) \| _ { 2 }$ is the inward-pointing unit surface normal at the object surface point closest to contact point p<sub>k</sub>, and $\begin{array} { r } { \bar { \bf p } = \dot { K } ^ { - 1 } \sum _ { k = 1 } ^ { K } \mathbf { p } _ { k } } \end{array}$ is the contact centroid. $E _ { \mathrm { c l o s u r e } }$ adapts the force-closure energy of [15], [16], [48]: it is the sum of the squared net force and moment of unit normal contact forces, normalized by $K ^ { 2 }$ , so it vanishes when both cancel. Contact magnitudes are fixed at unity and no friction cone enters, making this a frictionless net-wrench residual rather than a force-closure feasibility test, and moments are taken about the contact centroid rather than the object’s center of mass. Surface normals are held fixed during each gradient update.

## A. Whole-Body Grasp Parameterization

The humanoid is treated as a dexterous manipulator whose torso acts as a static “palm.” Any selected arms or legs can move to establish the prescribed contacts. The joint vector q contains the $n _ { q }$ joints selected for optimization; all other joints remain fixed at their prescribed configurations.

A candidate has $n _ { q } + 9$ scalars optimized jointly: the three-dimensional position of the kinematic root in the object frame, its orientation in a continuous six-dimensional rotation parameterization, and the $n _ { q }$ joint angles. The signed distance field is defined in the object frame with the object at the origin. Optimizing T chooses the relative placement of the object and robot; inverting it gives the object pose relative to the fixed torso.

## B. Geometry and Contact Sampling

The object is represented by the exact triangle-mesh signed distance field queried through a bounding-volume hierarchy; no voxel volume is used, so all iterations query the mesh directly. Signs are computed by angle-weighted pseudonormals during the search and by ray parity for the final refinement iterations. The robot is approximated by collision spheres rather than by link meshes [48], generated from the robot geometry by spherical approximation [51].

Candidate contact points are sampled on the link meshes separately from the collision spheres. A contact configuration specifies the allowed arm, leg, and torso links and which links are required. Each iteration samples K allowed links, always including the required links, and draws one candidate point on each, resampled every iteration [16]. This keeps contacts from collapsing onto a single link and produces the spread of contact patterns the dataset exhibits. Penetration is evaluated over all spheres, regardless of active contacts.

Contact selection can also restrict torso contact points to a named surface region. These regions are tags on individual candidate points, rather than geometric boxes. They restrict the points used in the contact objective, while collision checks continue to cover the whole robot.

![](images/8c0e58c8940a6c1ffbc4c367f2d244b6f499e10d534e9271aa6b91b95544086c.jpg)  
Fig. 7: Torso contact regions for the WBD grasp dataset. One accepted arm–torso grasp is shown per region, each on a different object.

## C. WBD Grasp Dataset

For WBD-EX training and the rest of the pipeline, we specialize this general method to grasps using the arms and torso. On the Unitree G1, the Dex3 finger joints are pinned to constant closed-grip values, and the twelve leg joints and three waist joints remain fixed. Bimanual generation optimizes the fourteen arm joints, giving 23 scalar variables including the root pose; single-arm generation additionally freezes the opposite arm, leaving seven optimized joints and 16 scalar variables. The numerical settings and validation procedure below describe this WBD grasp dataset.

The candidate contact set contains 626 points, sampled uniformly on the link meshes and then hand-curated, across 34 links; 347 points over 25 links are active for the bimanual configuration. We sample $K = 5$ contact links per iteration. The dataset uses three contact configurations: right-arm contact over 13 links and bimanual contact over 25 links, both requiring torso contact, as well as bimanual contact over 24 arm links with the torso excluded.

For this dataset, only torso contact points carry region tags, spanning the six regions shown in Figure 7.

Restricting to a region has four effects. It filters the candidate contact points, so a generation run restricted to one torso region optimizes against roughly a sixth of the available torso points; it biases initialization, since the direction the body is aimed from is set per region; it assigns each region an equal share of the per-object grasp generation quota; and it labels the resulting grasps so that region coverage can be audited afterward. It does not restrict which links may make contact.

Every accepted grasp is finally mirrored across the sagittal plane, which doubles the set exactly. The mirror is an exact relabeling of an already-validated grasp, so it is not resimulated.

## D. Pre-Grasp and Post-Grasp Targets

We adapt the contact-normal pressing strategy of [16] for the post-grasp target and extend it to a retracted pregrasp. From a solved configuration, we derive a retracted pre-grasp and a pressed post-grasp target by backpropagating $\begin{array} { r } { \sum _ { k } \nabla \phi ( \mathbf { p } _ { k } ) \cdot \mathbf { p } _ { k } } \end{array}$ through the kinematics, with the SDF gradients held fixed, to obtain a joint-space gradient. We normalize this gradient to unit Euclidean norm, multiply it by +0.5 for retraction or −0.4 for pressing, and add it to q, clamping the result to the joint limits. The pre-grasp is the approach pose used to start the simulation trial; the post-grasp is the press target held during validation and later reused as the servo setpoint at each WBD-EX episode reset.

## E. Solver

For the WBD grasp dataset, candidates are optimized in parallel on one GPU: 4096 candidates for 4000 iterations with AdamW at learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ and a cosine schedule annealing to $1 0 ^ { - \bar { 4 } }$ . Steps after a short warmup are replayed from captured CUDA graphs. Optimization candidates are initialized by sampling a random direction outward from the object centroid and rotating the robot to face the object centroid. Orientation is then jittered by a uniform roll about the facing axis over $( - \pi , \pi )$ and two orthogonal tilts over $( - \pi / 8 , \pi / 8 )$ , and the seed posture is jittered per joint by ±25% of that joint’s range.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Closure weight  $w _ { \mathrm { c l o s u r e } }$ </td><td>1</td></tr><tr><td>Contact-distance weight wdist</td><td>300</td></tr><tr><td>Object-penetration weight  $w _ { \mathrm { l i n k } }$ </td><td>200</td></tr><tr><td>Joint-limit weight  $w _ { \mathrm { j o i n t } }$ </td><td>1</td></tr><tr><td>Self-collision weight  $w _ { \mathrm { s e l f } }$ </td><td>10</td></tr><tr><td>Contact-distance tolerance  $d ^ { \star }$ </td><td>2 mm</td></tr><tr><td>Joint-limit margin  $\delta _ { q }$  Pre-/post-grasp gradient multipliers</td><td>0.035 rad  $+ 0 . 5 / - 0 . 4$ </td></tr></table>

TABLE III: WBD grasp-generation parameters. Optimization weights and tolerances.

## F. Simulation Validation

Because the optimizer minimizes a static surrogate objective, the WBD grasp candidates are validated dynamically in two stages using mjlab [52]. The torso is pinned to its rest pose every substep, and the lower body is held fixed, so the torso frame is exactly static; the object is a free body. Physics runs at 1 ms with four substeps per control frame, an elliptic friction cone, and surface friction of $\mu = 0 . 6$ on the robot with priority over the object’s.

Each simulation trial starts the arms at the pre-grasp, closes to the grasp candidate configuration, presses to the post-grasp target, and holds there. Approach and press last 40 control frames each, followed by a brief 60-frame settle and the 300-frame hold, with 4 ms per frame. The gravity load and any perturbations are applied during the hold. A link is in contact when its normal force is at least 0.1 N.

Stage one loads the object at 4× weight and, over the first half of the hold, sweeps a sustained 10 N force and 0.5 N m torque through the six axis directions for 100 ms each, followed by an unperturbed holding phase. A candidate passes if the object is still held in contact with at least two contact links at the end of the holding phase.

Stage two re-runs the survivors with no perturbations and normal gravity, and adds pose gates, requiring the object to drift less than 0.10 m and rotate less than $4 5 ^ { \circ }$ over the hold. The configuration and object pose recorded at the end of stage two are the settled values stored in the dataset and used at WBD-EX reset, so training starts from poses settled under contact rather than raw optimizer solutions.

We refer throughout to a grasp’s mode, meaning which arms must contact the object at the goal, as specified by its optimization contact group. Mode L requires left-arm contact, R right-arm contact, and B both; single-arm goals also require the opposite arm to release (Appendix III-C).

## APPENDIX II

## VOXELIZATION AND THE WBD GRASP DATASET

## A. Dataset Contents

Each object’s accepted grasps are stored as a single dataset that WBD-EX loads at startup. Per grasp, we keep the settled arm configuration $q _ { \mathrm { s i m } }$ (14 joints), the pressed post-grasp target $q _ { \mathrm { p o s t } }$ used as the servo setpoint at reset, the settled object pose $T _ { \mathrm { o b j } } ^ { \mathrm { t o r s o } }$ as a homogeneous transform in the torso frame, and the grasp mode $m \in \{ L , R , B \}$

## B. Position Voxelization

We voxelize by the settled object center expressed in the torso frame, taking the translation of $T _ { \mathrm { o b j } } ^ { \mathrm { t o r s o } }$ and binning it on a 2 cm lattice anchored at the torso origin. A voxel’s representative position is the mean of its member grasp positions. Within a voxel, grasps are grouped by mode, so a node of the goal graph is an occupied (voxel, mode) pair. Across the object set, this yields $1 . 0 { - } 5 . 5 \times 1 0 ^ { 3 }$ occupied voxels and $1 . 7 \mathrm { - } 9 . 1 \times 1 0 ^ { 3 }$ nodes; spheres sit at the low end and the elongated prisms at the high end, because an elongated object reaches a wider spread of settled centers for the same body region. The occupied region spans roughly ±0.21 m laterally and 0.03–0.45 m vertically.

## C. Goal Graph

Mode compatibility and goal sampling follow Section III-B, with $B  B$ transitions also permitted. A mode change within a single voxel is a legal zero-distance transition. Intermediate goals exclude the current (voxel, mode) node.

Far goals exclude the current voxel. If no legal-mode node lies within the sampled distance, the sampler selects the nearest legal-mode node toward the far goal. If no legalmode node exists, the episode terminates as a dead end. A curriculum ramps $d _ { \mathrm { m a x } }$ linearly from 0.02 m to 0.08 m over the first 5000 iterations. Holds earn no transition credit.

## D. Dataset Construction

Candidate grasps enter the dataset only after the two-stage simulation filter of Appendix I. We then apply a safety filter to the settled pose with two components. The settled object center must lie at least 9 cm above the torso-frame origin, or

3 cm for objects whose mesh bounding-box height exceeds 15 cm. An exact on-body clearance test then drops any grasp whose object surface sits farther than 5 mm from the torso.

Coverage measures sampled object-center positions around the torso. For each object, we offset exterior torsoface centroids along their normals at six evenly spaced distances between the object’s minimum and maximum bounding-box half-extents. We deduplicate positions on a 2 cm grid and keep those at torso-frame heights $z \in$ [0.09, 0.29] m. A reference position is covered if a grasp center lies within the object’s minimum half-extent. The perobject score is the fraction covered, averaged equally across objects. For pooled coverage, we combine all grasp centers and reference positions, deduplicate the latter on a 1 cm grid, and use the 7.6 cm mean minimum half-extent as tolerance.

## APPENDIX III

## WBD-EX TRAINING DETAILS

## A. Observations and Actions

Table IV lists the observation terms. The actor receives 89 dimensions at the current timestep, and the critic receives the same terms plus six privileged dimensions giving the realized randomization of Appendix III-E, for 95 dimensions. Actor observations are corrupted during training with additive uniform noise of ±0.005 rad on joint positions, ±0.003 m on the goal offset and object position, and ±0.01 on the rotation entries; the critic receives the same terms uncorrupted.

<table><tr><td>Group</td><td>Term</td><td>Dim.</td></tr><tr><td rowspan="4">Proprioception</td><td>Arm joint positions, minus default</td><td>14</td></tr><tr><td>Arm joint velocities</td><td>14</td></tr><tr><td>Servo error, commanded — measured</td><td>14</td></tr><tr><td>Previous action</td><td>14</td></tr><tr><td rowspan="2">Command</td><td>Hold flag, commanded and max. goal distance</td><td>3</td></tr><tr><td>Goal position relative to the object</td><td>3</td></tr><tr><td rowspan="2">Object</td><td>Position and 6-D rotation, torso frame</td><td>9</td></tr><tr><td>Linear and angular velocity, torso frame</td><td>6</td></tr><tr><td rowspan="6">Contact</td><td>Contacting-link fractions:</td><td></td></tr><tr><td>left arm, right arm, torso</td><td>3</td></tr><tr><td>Contacting links / total monitored links</td><td>1</td></tr><tr><td>Retention flag (ρt)</td><td>1</td></tr><tr><td>Net contact-force magnitude</td><td></td></tr><tr><td>(divided by 80 N, capped at 2)</td><td>1</td></tr><tr><td>Grasp mode</td><td>One-hot source and target mode</td><td>6</td></tr><tr><td>Actor total</td><td></td><td>89</td></tr><tr><td rowspan="3">Privileged</td><td>Object mass</td><td>1</td></tr><tr><td>Object center-of-mass offset</td><td>3</td></tr><tr><td>Mean sliding friction, per group</td><td>2</td></tr><tr><td>Critic total</td><td></td><td>95</td></tr></table>

TABLE IV: WBD-EX observation terms. Current-timestep inputs, without history. The actor uses the first five groups with training noise; the critic uses their uncorrupted values plus the sampled dynamics parameters.

The action is a 14-dimensional bounded increment on the arm joint targets. A raw action is squashed through tanh and scaled by 0.025 rad per control step, then accumulated into a servo setpoint that persists across steps. The setpoint is clamped to lie within 0.35 rad of the measured position and within the joint limits, and is linearly interpolated across the four physics substeps. The lower body and waist are welded, so the torso frame is exactly static and the arms are the only actuated degrees of freedom. Arm stiffness is $k _ { p } = 2 0 ;$ damping is derived per joint as $k _ { v } \propto \sqrt { k _ { p } } .$

## B. Rates, Episodes, Terminations

Physics runs at 200 Hz with a decimation of four, giving a 50 Hz control rate. Training episodes last 40 s (2000 control steps), and rewards are not scaled by the timestep. Contact is read from a sensor between all links and the object.

An episode ends on time-out, which is bootstrapped, or on one of six failures: the object falling below 0.5 m in the world frame; contact lost for longer than a grace window of 24 control steps (0.48 s), where contact is maintained if the arm contacts required by either the source or the target grasp mode are present, and torso contact alone does not count; the object leaving a 0.55 m workspace radius or exceeding $6 \mathrm { m / s }$ or 40 rad/s; the object coming to rest on the legs (disabled for large objects); a non-finite state; or a dead end in the goal graph. The sparse failure penalty fires on the first five of these; dead ends and time-outs are not penalized.

Each episode is initialized directly from the grasp dataset, with the arm configuration and object pose written exactly from the sampled grasp and zero velocity. The servo target is set instantly to that grasp’s pressed target with no ramp. Grasps are sampled as in Section III-B. The only continuous randomization at reset is $\mathbf { a \pm 5 }$ mm jitter on movement goal positions; hold goals are exact.

## C. Goal Completion Criteria

We define the object as settled when its linear and angular speeds are below $0 . 1 2 \mathrm { m } / \mathrm { s }$ and $1 . 5 \mathrm { r a d } / \mathrm { s } ,$ respectively. A commanded goal is completed when all of the following hold at once: the object is within 0.02 m of the commanded position, the object is settled, the arms required by the goal grasp mode are in contact, and these conditions persist for 10 control steps for movement goals or 25 for hold goals. A single-arm goal additionally requires the other arm to register no object contact for five consecutive steps. Training and evaluation both apply this requirement.

## D. Reward Details

Table V lists the weights applied to the reward terms defined in Section III-B and below. Reward weights are fixed for the whole of training; the two curricula modify the environment rather than the objective. For progress, both distance errors refer to the same commanded goal. Assigning a new goal resets the distance baseline without generating progress.

The object’s linear and angular speeds are denoted by $v _ { t }$ and $\omega _ { t }$ . Let $a _ { t }$ be the raw policy action, $b _ { t } = \operatorname { t a n h } ( a _ { t } )$ the bounded action before scaling to joint increments, $c _ { t }$ the arm joint target, and $q _ { t }$ the measured arm joint positions. The five regularization penalties are:

• Unsupported motion, $( 1 - \rho _ { t } ) \operatorname* { m a x } ( v _ { t } - 5 \mathrm { c m / s } , 0 ) ^ { 2 }$ : penalizes linear object speeds above the threshold when $\rho _ { t } = 0$ discouraging unsupported motion.

• Speed cap, max $( v _ { t } \ - \ 0 . 3 5 \mathrm { m } / \mathrm { s } , 0 ) ^ { 2 } \ + \ 0 . 0 2 \mathrm { m a x } ( \omega _ { t } \ -$ $3 \mathrm { r a d } / \mathrm { s } , 0 ) ^ { 2 } \colon$ : penalizes linear and angular speeds above their respective thresholds.

• Action magnitude, $\| b _ { t } \| _ { 2 } ^ { 2 } \colon$ penalizes large bounded actions.

• Action rate, $\| b _ { t } \mathrm { ~ - ~ } b _ { t - 1 } \| _ { 2 } ^ { 2 } \colon$ penalizes changes between consecutive bounded actions.

• Servo lead, $\begin{array} { r } { \sum _ { j = 1 } ^ { 1 4 } [ | c _ { t , j } - q _ { t , j } | - 0 . 2 5 \operatorname { r a d } ] _ { + } ^ { 2 } } \end{array}$ : penalizes joint targets more than 0.25 rad from the measured positions.

<table><tr><td>Term</td><td>Coefficient</td></tr><tr><td>Progress</td><td>100</td></tr><tr><td>Goal-region arrival</td><td>0.15</td></tr><tr><td>Transition completion</td><td>20</td></tr><tr><td>Hold quality</td><td>0.2</td></tr><tr><td>Arm neutrality</td><td>0.5</td></tr><tr><td>Failure penalty</td><td>-20</td></tr><tr><td>Unsupported motion</td><td>-1</td></tr><tr><td>Speed cap</td><td>-0.5</td></tr><tr><td>Action magnitude</td><td> $- 1 0 ^ { - 4 }$ </td></tr><tr><td>Action rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Servo lead</td><td>-0.1</td></tr><tr><td></td><td></td></tr></table>

TABLE V: Reward coefficients. Negative values denote penalties.

## E. Domain Randomization

Randomization ranges and schedules are listed in Table VI.
<table><tr><td>Parameter</td><td>Range and sampling schedule</td></tr><tr><td>Robot friction</td><td>Per-geometry multiplier U(0.75, 1.25), sampled on the first reset and fixed per environment.</td></tr><tr><td>Object mass and inertia</td><td>Mass 0.4–1.2 kg for every object, with inertia co-scaled. Sampled at startup and fixed per environment.</td></tr><tr><td>Object center of mass</td><td>Offset ±5 mm independently per axis. Sampled at startup and fixed per environment.</td></tr><tr><td>Perturbation force</td><td>Magnitude  $\mathcal { U } ( 0 , F _ { \mathrm { m a x } } )$  and isotropic direction, applied at the object center. No applied torque.</td></tr><tr><td>Force duration</td><td>0.05–0.20 s per perturbation.</td></tr><tr><td>Cooldown</td><td>0.75–2.0 s between perturbations.</td></tr><tr><td>Force cap  $F _ { \mathrm { m a x } }$ </td><td>0 for the first 2000 iterations, then a linear ramp to 20 N over the next 8000; 20 N thereafter.</td></tr></table>

TABLE VI: WBD-EX domain randomization. Dynamics parameters are fixed per environment, while perturbation forces ramp up during training.

## F. Optimization

Experts are trained in mjlab [52] with PPO [49] and the asymmetric actor–critic [53] inputs specified in Appendix III-$\mathrm { A } .$ We use 64 steps per environment, five epochs, and 16 minibatches per update, discount $0 . 9 9 , { \mathrm { G A E ~ } } \lambda = 0 . 9 5 .$ , clip 0.2, value-loss coefficient 1, gradient-norm clip 1, and advantage normalization over the collected batch. The learning rate starts at $1 0 ^ { - 3 }$ under a KL-targeted adaptive schedule with target 0.015, clamped to $[ 1 0 ^ { - 5 } , 1 0 ^ { - 2 } ]$ ; the second stage inherits the pre-trained sphere policy’s converged rate. The entropy coefficient is 0.005 in the first stage and 0 in the second. The ∅180, 200, and 220 mm sphere experts are not warm-started from another object; they are trained to 50,000 iterations with the first-stage settings. Actor and critic are separate 512 × 256 ELU networks with empirical observation normalization, and the policy is Gaussian with a learned standard deviation initialized at 0.3 and hardclamped to [0.05, 0.6] in the first stage and [0.05, 0.4] in the second. Production runs use 8192 environments on each of eight NVIDIA H100 GPUs, giving 65,536 environments and roughly $4 . 2 \times 1 0 ^ { 6 }$ transitions per iteration.

![](images/b2f47261a1de95cdf71c1bc824153544c05da24dbd19ac5591c1372358f114b4.jpg)  
Fig. 8: Evaluation regions. Six torso-frame goal regions shown from two viewpoints. Required grasp modes are both arms for front and back, and the corresponding single arm for the underarms and shoulders.

## APPENDIX IV

## WBD-EX REGIONAL ROUTE COMPLETION

WBD-EX policies are evaluated on fourteen standardized routes traversing six body regions (Table VII).

## A. Body Regions

Evaluation is defined over six named regions of the torso frame, applied to the settled object center (Fig. 8). The region bands are fixed across all object sets so that scores are comparable; the voxels a given region admits differ across objects only because the reachable grasp set does. Front and back occupy the three innermost 2 cm columns on either side of the body within ±3 cm laterally and 0.090–0.19 m vertically, and both require a both-arm grasp. The underarms lie beyond ±0.11 m laterally, within −0.04–0.06 m front-to-back and 0.090–0.22 m vertically, and require the corresponding single-arm grasp. The shoulders lie beyond ±0.05 m laterally, within ±0.06 m front-to-back and 0.29– 0.49 m vertically, also requiring the corresponding singlearm grasp. All expert evaluations use the completion and inactive-arm release requirements in Appendix III-C.

## B. Routes

A route is a fixed sequence of regions. The environment is initialized inside the first region, and each subsequent region becomes a commanded goal that must be reached in order. We name a route by its region sequence, abbreviating front as F, back as B, the left and right underarms as LA and RA, and the left and right shoulders as LS and RS. A route’s name therefore states its itinerary: route F LA B starts at the front, passes through the left underarm, and finishes at the back. Table VII lists all fourteen routes.

Route Region sequence   
Underarm   
F LA F front → left underarm → front   
F RA F front → right underarm → front   
LA F LA left underarm → front → left underarm   
RA F RA right underarm → front → right underarm   
Shoulder   
F LS F front → left shoulder → front   
F RS F front → right shoulder → front   
LS F LS left shoulder → front → left shoulder   
RS F RS right shoulder → front → right shoulder   
Back   
F LA B front → left underarm → back   
B RA F back → right underarm → front   
B LA B back → left underarm → back   
B RA B back → right underarm → back   
Loop   
F LA B RA F front → left underarm → back → right underarm → front   
B LA F RA B back → left underarm → front → right underarm → back  
TABLE VII: Evaluation routes. Fourteen routes grouped by region. Each sequence includes the initial region; subsequent regions are goals.

Regional route completion is the fraction of trials that reach every goal of a prescribed route in order before failure or timeout; partial progress along a route scores zero. The studies in Appendices IV-D, V and VI specify their own checkpoint selection, route subsets, and initial-state filtering.

![](images/a611e5cdc6a6eb2173f3bccad41cd7838abde6cc6912010bd9df4f1b2c2a16ff.jpg)

![](images/fd30b62e5ca733d851270cc98d0297fa5ce97e0245c6d6582434786be1559b45.jpg)

Fig. 9: Training progress for one expert per geometry class. (a) Mean episode reward during training, averaged over up to 1,000 logged updates; bands are approximate 95% confidence intervals (CIs) from within-window block resampling. (b) Regional route completion, averaged equally over fourteen routes at 2,048 trials per route; bands are pointwise 95% evaluation CIs. One training seed per object.  
![](images/1c856e2bc3588e2bbd87dace2ab8f4ed69dd544d0e0314c334d1a85d1c2dd9e9.jpg)  
Fig. 10: Regional route completion. Results at 50,000 iterations for one expert per geometry class, grouped by the body region the route visits. Error bars are pointwise 95% CIs from 2,048 trials per route. One training seed per object.

## C. Trials and Budget

Every policy is scored against the same completion requirements regardless of how it was trained, and the dwell, position threshold, and contact grace window match training. Each route is evaluated with 2,048 trials at a fixed seed and a 1,500-step budget per trial, with trials retired at the first termination, giving 28,672 trials per checkpoint over the fourteen routes. Evaluation retains the object-mass, friction, and center-of-mass randomization of Appendix III, freezes the goal-distance curriculum at its final value, and disables perturbations.

## D. Expert Evaluation Across Object Primitives

One WBD-EX expert is trained per object and evaluated on all fourteen routes of Table VII under the protocol of Appendix IV. Table VIII reports the measured fourteen-route mean for each object expert, grouped by the route groups of Table VII, because objects do not rank from easy to hard along a single axis: they fail in region-specific ways.

<table><tr><td rowspan="2">Object (mm)</td><td colspan="3">Completion by region group (%)</td><td rowspan="2">All 14</td></tr><tr><td>Underarm</td><td>Shoulder Back</td><td>Loop Mean (%)</td></tr><tr><td>Sphere ∅120</td><td>75.9</td><td>39.8</td><td>69.0</td><td>61.5</td><td>61.57</td></tr><tr><td>Sphere ∅140</td><td>80.8</td><td>60.6</td><td>73.1</td><td>71.5</td><td>71.50</td></tr><tr><td>Sphere ∅160</td><td>85.6</td><td>61.3</td><td>80.9</td><td>80.2</td><td>76.56</td></tr><tr><td>Sphere ∅180</td><td>86.1</td><td>76.0</td><td>85.4</td><td>83.2</td><td>82.59</td></tr><tr><td>Sphere ∅200</td><td>88.5</td><td>77.8</td><td>85.6</td><td>86.6</td><td>84.33</td></tr><tr><td>Sphere ∅220</td><td>87.9</td><td>83.2</td><td>92.2</td><td>90.4</td><td>88.14</td></tr><tr><td>Cube 140</td><td>72.2</td><td>57.4</td><td>45.4</td><td>19.7</td><td>52.81</td></tr><tr><td>Cube 160</td><td>75.6</td><td>62.6</td><td>44.1</td><td>19.8</td><td>54.91</td></tr><tr><td>Cube 180</td><td>65.5</td><td>59.3</td><td>41.4</td><td>11.5</td><td>49.10</td></tr><tr><td>Cube 200</td><td>61.3</td><td>69.7</td><td>12.5</td><td>0.0</td><td>40.98</td></tr><tr><td>Prism 100 × 100 × 200</td><td>70.7</td><td>47.1</td><td>48.5</td><td>18.2</td><td>50.12</td></tr><tr><td>Prism80× 160× 220</td><td>47.0</td><td>29.7</td><td>10.9</td><td>0.1</td><td>25.05</td></tr><tr><td>Prism140×160×240</td><td>48.6</td><td>33.6</td><td>11.2</td><td>0.7</td><td>26.77</td></tr><tr><td>Cylinder ∅120×220</td><td>66.9</td><td>33.2</td><td>29.2</td><td>6.2</td><td>37.82</td></tr><tr><td>Cylinder ∅140×160</td><td>69.2</td><td>45.5</td><td>49.1</td><td>21.3</td><td>49.83</td></tr></table>

TABLE VIII: WBD-EX expert results per object. Equalweight route means with 2,048 trials per route.

<table><tr><td></td><td colspan="2">Underarm</td><td colspan="2">Shoulder</td><td colspan="2">Back</td><td colspan="2">Loop</td><td colspan="2">All 14 routes</td></tr><tr><td>Object (mm)</td><td>Scr.</td><td>Warm</td><td>Scr.</td><td>Warm</td><td>Scr.</td><td>Warm</td><td>Scr.</td><td>Warm</td><td>Scr.</td><td>Warm</td></tr><tr><td>Prism 100 × 100 × 200</td><td>35.3</td><td>70.7</td><td>27.0</td><td>47.1</td><td>18.9</td><td>48.5</td><td>0.8</td><td>18.2</td><td>23.30</td><td>50.12</td></tr><tr><td>Cube 140</td><td>67.5</td><td>72.2</td><td>55.2</td><td>57.4</td><td>46.2</td><td>45.4</td><td>15.6</td><td>19.7</td><td>50.48</td><td>52.81</td></tr><tr><td>Cube 200</td><td>55.1</td><td>61.3</td><td>68.9</td><td>69.7</td><td>0.3</td><td>12.5</td><td>0.0</td><td>0.0</td><td>35.53</td><td>40.98</td></tr><tr><td>Cylinder ∅140× 160</td><td>68.2</td><td>69.2</td><td>29.6</td><td>45.5</td><td>56.7</td><td>49.1</td><td>25.2</td><td>21.3</td><td>47.74</td><td>49.83</td></tr><tr><td>Sphere ∅140</td><td>81.1</td><td>80.8</td><td>39.1</td><td>60.6</td><td>72.7</td><td>73.1</td><td>71.8</td><td>71.5</td><td>65.37</td><td>71.50</td></tr><tr><td>Sphere ∅160</td><td>89.4</td><td>85.6</td><td>42.1</td><td>61.3</td><td>83.8</td><td>80.9</td><td>83.9</td><td>80.2</td><td>73.51</td><td>76.56</td></tr></table>

TABLE IX: Warm-start versus from-scratch experts by body region, at 50,000 matched total iterations with 2,048 trials per route. Entries are mean regional route completion (%), equally weighted over each group’s routes. Bold marks the higher score when the 95% evaluation CI for the difference excludes zero. One training seed per condition.

To characterize what a trained WBD-EX expert looks like, we take one object from each geometry class: the ∅160 mm sphere, the 160 mm cube, the ∅140 × 160 mm cylinder, and the 100×100×200 mm prism. We report their training reward and regional route completion (Fig. 9) and their final perroute profiles (Fig. 10). All four are warm-started, so the horizontal axes of Fig. 9 start after the pre-trained sphere policy’s 15,000 iterations.

Completion is highest for spheres and lowest for the slender prisms and cylinder. The gap between geometry classes is largest on the most difficult trajectories: the fourgoal loops, which chain the most transfers. Loop completion is near zero for the 200 mm cube and the two larger prisms $( 8 0 \times 1 6 0 \times 2 2 0$ mm and $1 4 0 \times 1 6 0 \times 2 4 0 \mathrm { { m m } ; }$ Table VIII). Their large size relative to the G1’s limbs contributes to this difficulty. Although route completion differs across object shapes and sizes, all experts supply trajectories for distillation. Continued improvement of expert success across all object shapes and sizes is left to future work.

## APPENDIX V

## WARM-START VERSUS TRAINING FROM SCRATCH

Section III-B describes the two-stage procedure: each warm-started target expert is initialized from a pre-trained sphere expert’s complete training state. Here, we compare it against training the same object from scratch.

## A. Setup and Matched Budgets

We compare warm-started and from-scratch experts for six objects: a 100 × 100 × 200 mm prism, cubes of side 140 and 200 mm, a ∅140 × 160 mm cylinder, and spheres of diameter 140 and 160 mm. We warm-start these experts from policies pre-trained on spheres with diameters of 200– 220 mm. All share one training recipe and the randomization of Appendix III-E, and both conditions switch to the secondstage settings of Appendix III at 15,000 total iterations. Budgets are matched at 50,000 total iterations. The warmstarted expert is charged its pre-trained sphere policy’s 15,000 iterations plus 35,000 on its own object, so it reaches the comparison point having seen its target object for 15,000 fewer iterations. All six are compared on all fourteen routes.

## B. Results by Body Region

Table IX breaks the comparison down by body region, and Fig. 11 shows the corresponding training progress. All six warm-started experts have higher mean regional route completion, although some regional scores are lower. Shoulder completion is higher for every warm-started expert. The prism has the largest aggregate difference, from 23.30% to 50.12%, including loop completion of 0.8% versus 18.2%. For cube 200, mean completion across the four back routes is 0.3% from scratch and 12.5% with warm-starting. Both conditions have 0.0% loop completion. Other regions show smaller differences or regressions. The warm-started cylinder has lower back and loop completion, and the ∅160 mm sphere has lower underarm, back, and loop completion despite its higher overall mean.

## C. Training Cost

The 15,000-iteration sphere pre-training cost is shared by all experts initialized from that policy. For N target experts, each post-trained for 35,000 iterations, the average budget is 35,000 + 15,000/N iterations per expert, compared with 50,000 iterations per expert trained from scratch. Counting the six objects here with their two pre-trained sphere policies, the total is 240,000 iterations against 300,000, a 20% reduction in aggregate iterations with all six final means higher.

Representative runs on eight NVIDIA H100 GPUs use 15,000 sphere pre-training iterations at approximately 270 H100 GPU-hours, followed by 35,000 post-training iterations per object at 660–815 H100 GPU-hours. Training an expert from scratch for 50,000 iterations costs approximately 1,030 H100 GPU-hours. The shared pre-training contribution is therefore approximately 270/N H100 GPU-hours per target. In these runs, the sphere also has the lowest cost per iteration, complementing its shorter pre-training budget. The sphere’s compact contact patches produce fewer simultaneous contact constraints, reducing the simulator’s contact-solver cost. These are GPU costs for the runs that produced these experts.

Figure 11 shows several warm-started experts crossing the from-scratch endpoint well before their own budget is spent (the prism after 10,000 target iterations and both spheres within 15,000), suggesting additional savings if post-training stops at matched regional route completion.

The initialization transfers the actor, critic, optimizer state, observation normalizers, and curriculum progress together.

## APPENDIX VI ESCAPABILITY-BASED SELECTION OF EXPERT INITIALIZATIONS

Grasp simulation validation (Appendix I) certifies that a grasp stably holds the object, not that a trained expert can reposition the object from it, and experts can transition out of some validated grasps far more readily than others. Down-weighting low-quality grasps during WBD-EX training underperformed training on the full grasp set, so we instead score each grasp with its trained expert and filter after training, when selecting initial states for trajectory collection (Appendix VII-B). Within each voxel, we remove the lowestscoring grasps while preserving occupied voxel–grasp-mode pairs.

![](images/d7bafb007e5e81b6ddba72d9bb9691fe42932eb7ac5ff19be7448954004f8b2e.jpg)  
Fig. 11: Warm-started and from-scratch learning curves. Warm-started curves include 15,000 sphere pre-training iterations (shaded); their first point precedes target-object training. Markers show mean regional route completion, weighting all fourteen routes equally. Bars show pointwise 95% evaluation CIs (binomial trial uncertainty or evaluation-batch bootstrap), often smaller than the markers. One training seed per condition.

Policy weights, the goal pool, and intermediate routing remain unchanged. We report regional route completion to characterize the effect of this selection using the same completion requirements as the expert evaluations.

## A. Escapability Scoring

For a grasp g, escapability is the fraction of transitions attempted from g, each to a single commanded goal, that reach the commanded position and grasp mode under the requirements of Appendix IV. Targets are drawn over the eligible mode-compatible position voxels 6–8 cm from the source voxel’s representative position, with a budget of 3 s (150 control steps) per attempt. The score is specific to the frozen WBD-EX expert. We evaluated all 2,510,864 grasps across the fifteen objects with 128 sampled targets per grasp, for $3 . 2 \times 1 0 ^ { 8 }$ single-goal transition trials.

Figure 12 shows that the distribution’s shape, not just its mean, varies strongly with geometry. The two most slender objects (the ∅120 × 220 mm cylinder and the 80 × 160 × 220 mm prism) have nearly flat distributions with mean escapability near 40% and only about 1% of their grasps in the top band. Sphere grasps are increasingly concentrated at high escapability with diameter, from 28% of grasps in the top band at ∅120 to 62% at ∅220, with means of 64% and 86%. Thus, grasps that pass grasp simulation filtering can still have low transition success under their fully trained expert, and the distribution of this success varies across the object-specific grasp datasets.

## B. Removal Threshold Selection

Within each original 2 cm position voxel, we remove the lowest-scoring grasps up to a nominal percentage cap, keeping at least twenty grasps where available and preserving every original voxel–grasp-mode pair. We use the twelve non-loop evaluation routes of Table VII. We re-score each expert on the saved evaluation trials whose starts survive the cut, sweeping the cap from 0 to 90%. Figure 13 shows mean regional route completion for each object across removal caps. We use 2,048 bootstrap resamples of evaluation trials, shared across routes and removal caps.

At the adopted cuts, the per-object means in Fig. 13 improve by 1.2–13.3 percentage points over the full grasp dataset baseline across all fifteen objects. The cut preserves every occupied voxel–grasp-mode pair and preserves at least 98.8% of each object’s average per-joint span at every cap up to 70%. It preserves spatial and grasp-mode coverage while changing the distribution of grasp configurations within each voxel.

The removal each object tolerates tracks the shape of its escapability distribution. Objects with flat distributions take small cuts for small gains: 10% removal for 1.2 points on the slender prism and 30% for 4.0 points on the slender cylinder, against 20–60% removal for 5 to 13 points on the spheres. Where escapability is uniformly moderate, there is little low-escapability tail to remove, and the curve for the slender prism is nearly flat across the whole sweep. At larger removal caps, completion for the slender cylinder and the two largest spheres peaks and then declines.

![](images/4000891ee3a64a90009ea293040b8076f57cb5ba55d8e23bb946b6c8f7f7de33.jpg)  
Fig. 12: Grasp escapability distributions. Bars give the percentage of that object’s grasps falling in each ten-point escapability band, colored by geometry class. Error bars are pointwise 95% bootstrap CIs over grasps, with measured scores held fixed; counts are exact for the scored pools.

![](images/69f7e8bd155891e92dd1f3a7ad7dfab3a1b8bace1380a0f41aaa5a3029a70c3f.jpg)  
Fig. 13: Regional route completion after grasp filtering. Mean completion over twelve equally weighted routes, excluding loops. Dashed lines show unfiltered baselines; stars mark selected caps and gains in percentage points. Bands are pointwise 95% paired bootstrap CIs over evaluation trials. Vertical axes span twenty percentage points.

We adopt, per object, the smallest cap within one percentage point of the best mean available to it, subject to a 70% ceiling and to preserving at least 95% of the original average joint-span coverage, and we keep the full grasp dataset where the interval on the gain is not positive. The adopted caps range from 10 to 60%. The gains above are measured on the trials used to select the caps. The separate trajectory collection comparison below measures usable data yield directly.

## C. Trajectory Collection Yield

We compare trajectory collection from the full and escapability-filtered grasp datasets using 13,440 collection attempts per condition across all fifteen objects. Both con ditions use the same frozen experts, set of eligible goals, sampling mixture, randomization, and quality filter (Appendix VII-B). Each trajectory follows successive voxel-tovoxel goals until failure or a 120 s timeout. A trajectory is usable if, after removing the termination tail, it still passes the quality filter. Each condition splits its attempts between the two route samplers of Appendix VII-B. Combining both samplers, usable yield increases from approximately 29.0% to 32.5% (+3.5 percentage points), and usable frames per attempt from approximately 938 to 1,053 (+12%).

The fraction of attempts failing within the first 10 s falls from approximately 26.7% to 22.9%. Accepted trajectories contain approximately 3,200 frames under both conditions, suggesting that filtering mainly increases the number of accepted trajectories. The improvement in collection yield is consistent but modest. We use the filtered grasp datasets for trajectory collection because they raise the usable data per collection attempt without notably reducing the coverage of the grasp datasets.

## APPENDIX VII

## WBD CONTROLLER DISTILLATION DETAILS

Section III-C gives the WBD controller’s architecture and its guidance mechanism. Here we specify the network settings, training data, sampling recipe, and inference procedure.

## A. Architecture and Optimization

We follow the diffusion-and-inverse-dynamics formulation and published state-encoder/U-Net configuration of [11], adapting the representation and conditioning to whole-body manipulation. Our U-Net uses one middle block, two linear layers per block, and a 256-dimensional timestep embedding.

Each arm contributes four keypoints: one at the elbow and three on the hand, at the thumb, index finger, and middle finger. Forward kinematics expresses these points in the torso frame using the fixed closed-fist configuration.

Each history entry contains 24 torso-frame arm-keypoint coordinates, 14 measured joint positions, 14 applied joint targets, and 14 joint tracking errors. Four entries at offsets $( - 1 5 , - 1 0 , - 5 , 0 )$ , in 50 Hz control steps, and five static geometry features form the 269-dimensional diffusion input; no task-mode embedding is used. The model predicts 48 keypoint-displacement coordinates at future offsets (5, 10), with targets defined as future measured keypoint positions minus their current measured positions. The inverse model takes the current history entry, predicted motions, and geometry features (119 inputs), using width 1536 and four twolayer residual blocks to output 14 joint-target increments.

We train both networks jointly for 130,000 AdamW updates with a global batch of 2,560 training samples and base learning rate $3 \times 1 0 ^ { - 4 }$ , including a final 10,000-update cosine cooldown. The inverse model is supervised on an equal mixture of recorded expert motions and diffusion-generated motions, using the expert’s realized joint-target increment at the same state for both: the applied target after action scaling, tracking-error clamping, and joint limits, minus the current target. Diffusion training uses 100 timesteps with a cosine schedule (offset 0.008, β capped at 0.999).

Training the simulation and real-world controllers uses approximately 22 H100 GPU-hours per model across eight H100 GPUs.

## B. Trajectory Collection

Each frozen WBD-EX expert starts from its object’s escapability-filtered grasp dataset (Appendix VI) and follows successive position-and-grasp-mode goals without resetting between goals. Trajectories are recorded at 50 Hz until failure or a 120 s timeout.

Random voxel-sampled routes sample eligible goal voxels and compatible grasp modes, with initial left/right/both modes in a 1:1:2 ratio. Random region-graph routes sample initial and target grasps from voxels in connected body regions. The first transfer follows one of sixteen directed region pairs: front–shoulder, front–underarm, shoulder–underarm, and underarm–back, in both directions and on both sides; subsequent goals follow the voxel goal graph (Appendix II). Voxel-sampled routes with at least four completed goals also supply a longer-trajectory subset. Both samplers balance the 2–4, 4–6, and 6–8 cm goal-distance bands, while keeping the maximum distance used to build the goal graph at 8 cm.

We keep the usable portions of failed trajectories, discarding the final 100 frames before failure termination. Clean timeouts are left untrimmed, and we require at least two completed goals and 50 frames. Stalled intervals are removed using 50-frame windows in which the goal is unchanged, all joint spans are within 0.02 rad, the object position bounding box is within 5 mm, and the maximum pairwise object rotation is within 0.1 rad. Each training sample is drawn from a contiguous 26-frame trajectory segment that supplies the required observation history and future supervision. This segment never crosses a removed interval or an episode boundary.

## C. Sampling and Augmentation

Batches mix training samples from random voxel-sampled routes, region-graph routes, and the longer-trajectory subset. Each 320-sample per-GPU batch draws 125, 165, and 30 samples from these three sources and approximately equal numbers of samples from the 2–4, 4–6, and 6–8 cm goaldistance bands, with objects equally represented, while spatial and grasp-mode sampling promote coverage of body regions and single-arm grasp goals. Training samples are drawn from eligible timesteps throughout each trajectory; spatial sampling uses the object’s position at each sample’s current frame. We split trajectories and their derived samples by initial grasp using a fixed hash, assigning approximately 5% of grasp groups to validation.

Half of all training samples are reflected across the sagittal plane, applied as one coherent transform over history, actions, and future targets in physical coordinates before normalization, with validation and evaluation left unaugmented. All trajectories are collected under randomized dynamics. Mass, center-of-mass, and friction randomization follow $\mathsf { A p - }$ pendix III-E; the experts’ perturbations remain enabled at their trained 20 N cap. The randomized parameters are not controller inputs.

## D. Simulation Evaluation Routes

For the simulation comparisons in Fig. 4, we reuse the WBD-EX evaluation regions and goal-completion criteria (Appendix IV). Each trial follows a continuous sequence over the region connections of Appendix VII-B, sampling the next adjacent region uniformly after each completed transfer. No external perturbation forces are applied.

## E. Command-Smoothing Comparison

We test whether simple temporal smoothing can reproduce WBD’s gains under command corruption. We compare the noisy expert, exponential moving average (EMA) smoothing of its commands, and WBD refinement. For corrupted raw arm commands $u _ { t } .$ , EMA applies $z _ { t } = ( 1 - \beta ) z _ { t - 1 } + \beta u _ { t }$ before action bounding and scaling, initialized with $z _ { \mathrm { 0 } } = u _ { \mathrm { 0 } }$ and retained across goal changes. We select $\beta = 0 . 5$ from $\{ 0 . 0 5 , 0 . 1 , 0 . 2 5 , 0 . 5 \}$ by mean completed regional transfers on 64 separate validation starts per training object and noise level, weighting the 13 objects and two noise levels equally. Test starts and their mirrored families are excluded from selection. The selected coefficient, $\beta = 0 . 5 ,$ , also gives the highest validation goal count among the tested EMA settings.

<table><tr><td>α</td><td>Method</td><td>Goals / trial</td><td>Time to failure (s)</td></tr><tr><td>2.5</td><td>Noisy expert</td><td>8.47</td><td>29.1</td></tr><tr><td></td><td>Noisy + EMA</td><td>7.82</td><td>20.5</td></tr><tr><td></td><td>Noisy + WBD</td><td>23.79</td><td>86.1</td></tr><tr><td>4</td><td>Noisy expert</td><td>6.24</td><td>24.2</td></tr><tr><td></td><td>Noisy + EMA</td><td>6.32</td><td>17.6</td></tr><tr><td></td><td> $\mathrm { N o i s y + W B D }$ </td><td>21.79</td><td>64.6</td></tr></table>

TABLE X: Command smoothing versus WBD refinement. Means weight the 13 training objects equally, and WBD values are averaged over three training seeds.

This separate comparison repeats front → left underarm → front → right underarm at $\alpha \in \{ 2 . 5 , 4 \}$ , using 256 matched starts per object and terminating at the first failure or 600 s. We use the goals-per-trial metric defined in Section IV-B. All failures remain included. WBD uses guidance 100; the expert checkpoints, action scaling, joint limits, and task physics are shared across conditions.

The selected EMA filter yields similar goal counts and shorter time to failure than the noisy expert. WBD improves both metrics over this filter at both noise levels (Table X).

## F. Real-Time Inference

Each inference update samples 16 candidate future motions using 12 DDIM denoising steps. Each candidate predicts keypoint displacements 100 and 200 ms into the future from the robot’s current measured configuration. Guidance specifies displacement from the current commanded configuration to the requested configuration. These displacements are compared directly; the model also receives the joint tracking error between commanded and measured configurations.

Command chain. Let c be the current applied joint target and q the measured joint configuration. In simulation, the noisy raw expert action $\tilde { a } = a + \epsilon , \epsilon \sim \mathcal { U } ( - \alpha , \alpha ) ^ { 1 4 }$ , defines the requested configuration $q ^ { \mathrm { r e q } } = \Pi ( c + 6 \cdot 0 . 0 2 5$ tanh ˜a), where Π applies the 0.35 rad tracking-error clamp around $q$ and the joint limits. On hardware, $q ^ { \mathrm { r e q } }$ is the retargeted operator configuration clipped to the joint limits. The guidance displacement is $\mathbf { d } \ = \ \mathrm { F K } ( q ^ { \mathrm { r e q } } ) \ - \ \mathrm { F K } ( c )$ , with no additional magnitude cap on this input displacement on hardware. The inverse model predicts an increment, giving $\hat { c } = c + \Delta \hat { c } .$ In simulation, cˆ is mapped back to a raw action $\hat { a } = \mathrm { a t a n h } ( \mathrm { c l i p } ( ( \hat { c } - c ) / 0 . 0 2 5 , \pm 0 . 9 9 9 ) )$ ), clipped to $[ \tilde { a } - 2 . 5 , \tilde { a } + 2 . 5 ]$ , and applied through the experts’ action pipeline $\Pi ( c + 0 . 0 2 5 \operatorname { t a n h } { \hat { a } } )$ ; this ±2.5 raw-action radius is the bounded correction in Section IV-B. On hardware, the predicted output increment is clipped to ±0.025 rad per joint; the publisher then applies its configured joint-speed limit, a 0.35 rad tracking-error limit, and the joint limits.

With d $\in \mathbb { R } ^ { 2 \bar { 4 } }$ the guidance displacement defined above, for predicted keypoint displacements $\Delta { \bf x } _ { 5 }$ and $\Delta \mathbf { x } _ { 1 0 }$ at the two future offsets, guidance minimizes

$$
D = \| \Delta \mathbf { x } _ { 5 } - \mathbf { d } / 2 \| _ { 2 } + \| \Delta \mathbf { x } _ { 1 0 } - \mathbf { d } \| _ { 2 } .
$$

Each norm spans all keypoint coordinates in physical space. At each denoising step, the guidance weight scales a gradient step that reduces D. We select the generated candidate with the smallest summed squared error to these targets. The inverse-dynamics model then predicts a joint-target increment from the selected keypoint motion and the most recent state in the same observation history used for diffusion.

By the time this increment reaches the 50 Hz command publisher, the applied target may have changed. Let $c ^ { \mathrm { o b s } }$ be the applied target in that observation and cˆ the corresponding predicted target. The publisher adds the already-computed increment ${ \hat { c } } - c ^ { \mathrm { { o b s } } }$ to its latest applied target $c ^ { \mathrm { p u b } }$

$$
c ^ { \mathrm { n e w } } = c ^ { \mathrm { p u b } } + ( \hat { c } - c ^ { \mathrm { o b s } } ) .\tag{5}
$$

Between predictions, the publisher continues toward the resulting target without applying the same increment again. We construct observation histories from timestamped measurements and applied commands at the offsets in Appendix VII-A, independently of inference timing. On hardware, inference runs on an NVIDIA RTX 5090 GPU on the operator work station and is capped at 50 Hz. With these sampler settings, one inference update took 11.8 ms median (17.3 ms 95th percentile), updates arrived at 34.9 Hz on average, and the delay from state publication to the robot’s command buffer was 50.1 ms median (73.7 ms 95th percentile), excluding sensing and motor latency.