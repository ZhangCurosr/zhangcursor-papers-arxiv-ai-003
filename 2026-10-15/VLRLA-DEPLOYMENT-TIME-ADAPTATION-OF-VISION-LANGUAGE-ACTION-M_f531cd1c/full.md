# VLRLA: DEPLOYMENT-TIME ADAPTATION OF VISION-LANGUAGE-ACTION MODELS THROUGH IN-CONTEXT REINFORCEMENT LEARNING

Diyuan Shi<sup>1,2,†</sup>, Yiqi Tang<sup>2,†</sup>, Hongyin Zhang<sup>2,†</sup>, You Wu<sup>2</sup>, Donglin Wang<sup>2,∗</sup> <sup>1</sup>Zhejiang University <sup>2</sup>Westlake University <sup>∗</sup>Corresponding author <sup>†</sup>Equal contribution

## ABSTRACT

Vision-language-action (VLA) models connect visual perception and language understanding with robot control, providing a foundation for instructionconditioned manipulation. However, conventional pretrained VLA policies lack a mechanism for gradual adaptation through accumulated interaction experience during deployment. We therefore propose VLRLA, a general framework that equips existing VLA models with in-context reinforcement learning capabilities for deployment-time adaptation. Our approach augments a VLA model with a reinforcement learning module that learns from historical observations, actions, rewards, and termination signals. During deployment, the policy gradually refines its action generation and progressively improves task success while keeping model parameters fixed. Our approach is also designed for general applicability and compatibility with diverse learning history construction strategies and model training pipelines. Finally, through extensive experiments, we systematically investigate the learning properties of VLRLA in simulation and demonstrate its ability to adapt to unseen scenarios during real-world deployment.

## 1 INTRODUCTION

Vision-language-action (VLA) models connect visual observations and natural-language instructions to robot actions, enabling robots to interpret their surroundings and carry out user requests. Their potential applications span household assistance, warehouse logistics, and industrial manipulation (Collaboration et al., 2023; Intelligence et al., 2025b; Luo et al., 2025; Ai et al.; Xu et al., 2026).

Conventional VLA methods learn from collected demonstrations and could generalize to new situations. However, demonstration-based training alone does not provide a mechanism for gradual adaptation after deployment. A robot may encounter an unfamiliar camera viewpoint, lighting condition, or instruction phrasing, each of which hinders task performance. The central limitation is therefore the inability to turn accumulated deployment experience into adaptive behavior.

Online Deep Reinforcement Learning (DRL) or interactive Imitation Learning (IIL) provide routes to deployment-time adaptation through parameter updates, using reward feedback or corrective actions, respectively (Wagenmaker et al., 2025; Xu et al., 2026; Luo et al., 2025; Intelligence et al., 2025a; Pan et al., 2026). However, these approaches have several drawbacks: 1) Collecting the samples needed for policy updates takes tens of minutes to hours and involves substantial human effort; 2) They typically require task-specific hyperparameter tuning; and 3) They update policy in each deployment environment, making it difficult to scale to multi-task settings. These motivate us to explore an adaptation mechanism that uses deployment feedback without parameter updates.

In this work, we propose VLRLA, a framework for equipping VLA models with in-context Reinforcement Learning (RL) capabilities. The framework augments a VLA model with a RL module that learns to use past interactions for adaptation. During deployment, action generation is conditioned on a memory of previous interactions and reward feedback, while model parameters remain fixed. As experience accumulates, this memory informs refinements to subsequent behavior, helping policy adapt to the deployment environment. The intended result is gradual improvement in task success through continued interaction during deployment and no parameter updates.

Existing VLA works vary across training datasets, policy architectures, and deployment tasks, we aim to develop a general VLRLA framework that accommodates various learning history construction strategies, policy instantiations, and integration strategies. Our contributions are as follows:

• We formulate deployment-time adaptation in VLA models as an in-context reinforcement learning problem, using accumulated interactions and reward feedback to inform subsequent decisions while keeping model parameters fixed.

• We propose a general framework that unifies learning history construction and adaptation-module integration to equip pretrained VLA models with in-context reinforcement learning capabilities across diverse data construction settings, model architectures, and deployment tasks.

• We conduct extensive experiments in simulation and on real-world robots to demonstrate the effectiveness of VLRLA. Our results show adaptation across multiple pretrained backbones and improved task success with accumulated experience, including on challenging unseen tasks.

## 2 RELATED WORK

Vision-Language-Action Models. VLA models integrate visual and language representations with action prediction to support generalist robotic manipulation. Progress in this direction has centered on expanding the data and representations learning for training robot policies. The RT series (RT1, 2022; Brohan et al., 2023; Collaboration et al., 2023) advances generalist robot control through multi-task learning, web-scale vision-language knowledge transfer, and training across robot embodiments. OpenVLA (Kim et al., 2024) makes VLA modeling accessible through an open-source model trained on diverse robot demonstrations and designed to support efficient fine-tuning. π<sub>0.5</sub> (Intelligence et al., 2025b) pursues open-world generalization by co-training on heterogeneous robot and web data, combining semantic subtask prediction with robot control. LingBot-VLA (Wu et al., 2026) further explores large-scale pretraining across multiple dual-arm robot configurations, releasing a model, training infrastructure, and evaluation data to support adaptation across tasks and platforms. In this work, we study how to equip existing VLA models with in-context Reinforcement Learning capabilities for deployment-time adaptation.

Deep Reinforcement Learning. DRL enables policies to improve through trial-and-error. Proximal Policy Optimization (PPO) (Schulman et al., 2017) uses a policy-gradient surrogate objective to learn from on-policy rollouts. Soft Actor-Critic (SAC) (Haarnoja et al., 2019) combines off-policy actor–critic learning with an entropy maximization framework for continuous control. Recent studies apply these policy optimization principles to finetune pretrained VLA models. Examining the generalization benefits of RL fine-tuning, Liu et al. (2026) report that PPO-based OpenVLA finetuning improves semantic and execution generalization over supervised fine-tuning. SimpleVLA-RL (Li et al., 2025) also combines parallel trajectory collection with VLA-specific training optimizations to scale reward-based RL for OpenVLA-OFT. Other methods focus on online DRL fine-tuning directly in real-world tasks: RLToken (Xu et al., 2026) uses a compact representation from a pretrained VLA for sample-efficient online DRL fine-tuning. Similarly, DSRL (Wagen maker et al., 2025) learns an RL policy over the latent-noise input of a frozen diffusion or flow policy. These methods require parameter updates during RL fine-tuning, whereas our work studies in-context adaptation with all parameters fixed.

In-Context Learning. Recent robotics research explores how demonstrations, prompts, and interaction histories can guide action generation at inference time for VLA models. For adaptation from demonstrations, In-Context Robot Transformer (ICRT) (Fu et al., 2024) predicts actions from sen sorimotor trajectories, allowing new tasks to be specified through demonstration prompts without parameter fine-tuning. $\pi _ { 0 . 7 }$ (Ai et al.) combines language, subgoal images, and episode metadata to steer task execution, highlighting the role of multimodal context in generalization. In concurrent work, RoboTTT (Jiang et al., 2026) compresses visuomotor context into recurrent fast weights through test-time training, focusing on one-shot imitation from human videos and online corrections learned through DAgger (Ross et al., 2011) distillation. Specifically, reward-driven adaptation is addressed by Algorithm Distillation (AD) (Laskin et al., 2022b) which performs in-context RL in discrete simulation environments. In contrast, our work focuses on designing a general framework to equip existing VLA models with in-context RL capabilities across diverse learning history construction strategies and model settings with validated effectiveness in both simulation and real-world continuous robot control tasks.

## 3 PRELIMINARIES

Vision-Language-Action Policy. At timestep t, a VLA model π receives a language instruction ℓ and an observation $o _ { t } = ( I _ { t } , q _ { t } )$ where $I _ { t }$ is camera image and $q _ { t }$ is robot proprioceptive state, then generates an action chunk

$$
\pi ( o _ { t } , \ell ) \to a _ { t : t + h } \in \mathbb { R } ^ { h \times d _ { a } }\tag{1}
$$

where h is the action-chunk length, $d _ { a }$ is the dimension of each action, and $\scriptstyle a _ { t : t + h }$ contains the h actions $a _ { t } , \ldots , a _ { t + h - 1 }$ , excluding the endpoint $t + h$ . The robot controller then executes the action chunk before replanning.

In-Context Reinforcement Learning. A trial is a sequence of interactions that may span multiple episodes of the same task. We define the available history as $\mathcal { C } _ { t } = \{ ( o _ { i } , a _ { i : i + h } , r _ { i } , \bar { d _ { i } } ) \} _ { i = 0 } ^ { t - 1 }$ , with $\mathcal { C } _ { 0 } = \emptyset$ , where $r _ { i }$ is the observed reward and $d _ { i }$ marks an episode boundary. This history contains only completed interactions before step t. An in-context RL policy $a _ { t : t + h } \sim \pi _ { \psi } ( \cdot \mid o _ { t } , \ell , \mathcal { C } _ { t } ) $ performs action generation while keeping ψ fixed. For tasks $\mathcal { M } \sim p ( \mathcal { M } )$ and a discount factor $\gamma \in [ 0 , 1 )$ , the learning objective is to optimize the task return:

$$
J ( \psi ) = \mathbb { E } _ { \mathcal { M } \sim p ( \mathcal { M } ) , \pi _ { \psi } } \left[ \sum _ { t \geq 0 } \gamma ^ { t } r _ { t } \right] .\tag{2}
$$

Algorithm Distillation (AD). Algorithm Distillation (AD), introduced by Laskin et al. (2022b), trains a causal sequence model on ordered learning histories $\mathcal { C } \sim \mathcal { D } _ { \mathrm { h i s t } }$ collected from a source RL algorithm:

$$
{ \mathcal { L } } _ { \mathrm { A D } } ( \psi ) = - \mathbb { E } _ { { \mathcal { C } } \sim { \mathcal { D } } _ { \mathrm { h i s t } } } \left[ \sum _ { t = 0 } \log \pi _ { \psi } ( a _ { t } \mid o _ { t } , \ell , { \mathcal { C } } _ { t } ) \right] .\tag{3}
$$

Preserving the order of learning across episodes helps the model learn how action generation changes with experience. At deployment, the model could perform in-context RL and gradually improve the task return by conditioning on its own interaction history in unseen tasks, without parameter updates.

## 4 TRAINING ADAPTIVE VISION-LANGUAGE-ACTION MODELS

Algorithm Distillation (AD) provides a principled approach to distilling RL algorithms into causal sequence models, but its integration with existing pretrained VLA models remains unexplored. VLA methods differ in their data collection procedures, training pipelines, and deployment settings. We therefore design VLRLA as a general framework and demonstrate its effectiveness across multiple settings. This section presents the VLRLA framework, illustrated in Figure 1.

## 4.1 MODEL DESIGN

Deployment-time adaptation requires a robotic policy to integrate visual and language information with interaction feedback, retain experience over long horizons, and generate continuous actions. Our design addresses these requirements through recurrent memory, pretrained visual-language representations, and two policy instantiations that connect these representations to action generation.

Recurrent Memory. In robotic tasks, a trial can span multiple episodes and thousands of tokens. To reduce the substantial memory and computation costs of dot-product causal attention (Vaswani et al., 2017), we use recurrent memory similar to Transformer-XL (Dai et al., 2019). This mechanism combines causal attention within a local window of timesteps with attention to cached representations from earlier windows. The policy forward pass is expressed as

$$
( M _ { t } , \widehat { a } _ { t : t + h } ) = \pi _ { \psi } ( o _ { t } , \ell , a _ { t - 1 : t - 1 + h } , r _ { t - 1 } , d _ { t - 1 } \mid M _ { t - 1 } ) ,\tag{4}
$$

![](images/68dfbe289e55b2e05119fc09a706df6a2a2474bb1cbdcc6c7bddaa9ac535cec7.jpg)  
Figure 1: Framework overview. VLRLA is designed as a general in-context RL framework supporting diverse learning history construction strategies and model integration strategies.

where $\pi _ { \psi }$ denotes the full policy with parameters ψ, $M _ { t - 1 }$ denotes memory from the preceding step, and the outputs are the updated memory $M _ { t }$ and generated action chunk $\hat { a } _ { t : t + { \cal T } }$ <sub>h</sub>. Memory retains information across episodes within a trial, and model parameters remain fixed during deployment.

Visual and Language Preprocessing. We reuse representations from a pretrained VLA model to avoid training a visual-language encoder from scratch. The framework accommodates two interfaces: the encoder’s prefix embeddings and its key–value (KV) cache. Both provide information about the current observation and instruction for action generation, while recurrent memory supplies context from previous interactions. For the prefix-embedding interface, we extract visual-language embeddings and pass them to the recurrent policy:

$$
\begin{array} { r l r } & { } & { z _ { t } ^ { \mathrm { e m b } } = f _ { \bar { \theta } _ { \mathrm { V L M } } } ^ { \mathrm { e m b } } ( o _ { t } , \ell ) , ~ } \\ & { } & { ( M _ { t } , \hat { a } _ { t : t + h } ) = \pi _ { \phi } ^ { \mathrm { e m b } } ( q _ { t } , z _ { t } ^ { \mathrm { e m b } } , a _ { t - 1 : t - 1 + h } , r _ { t - 1 } , d _ { t - 1 } \mid M _ { t - 1 } ) , } \end{array}\tag{5}
$$

where $f _ { \bar { \theta } _ { \mathrm { V L M } } } ^ { \mathrm { e m b } }$ is the pretrained encoder that extracts prefix embeddings $z _ { t } ^ { \mathrm { e m b } }$ from the current observation $o _ { t }$ and instruction $\ell ,$ and $\pi _ { \phi } ^ { \mathrm { e m b } }$ is the downstream recurrent policy. The bar over $\theta _ { \mathrm { V L M } }$ indicates that the encoder is frozen during training, while ϕ denotes the policy parameters.

For the KV-cache interface, we condition the recurrent policy on the attention keys and values extracted from the VLM:

$$
\begin{array} { r l r } & { } & { z _ { t } ^ { \mathrm { K V } } = f _ { \bar { \theta } _ { \mathrm { V L M } } } ^ { \mathrm { K V } } ( o _ { t } , \ell ) , \quad } \\ & { } & { ( M _ { t } , \hat { a } _ { t : t + h } ) = \pi _ { \phi } ^ { \mathrm { K V } } ( q _ { t } , z _ { t } ^ { \mathrm { K V } } , a _ { t - 1 : t - 1 + h } , r _ { t - 1 } , d _ { t - 1 } \mid M _ { t - 1 } ) , } \end{array}\tag{6}
$$

where $f _ { \bar { \theta } _ { \mathrm { V L M } } } ^ { \mathrm { K V } }$ extracts the VLM’s KV cache $z _ { t } ^ { \mathrm { K V } }$ and $\pi _ { \phi } ^ { \mathrm { K V } }$ denotes the recurrent policy. The KV cache exposes layer-specific attention keys and values across multiple VLM layers, providing access to intermediate representations for action generation.

Policy Instantiations. Building on these memory and representation interfaces, we introduce two VLRLA variants for different task settings. VLRLA-simp uses a deterministic policy intended for simple tasks or simulation environments. VLRLA-bridge retains the pretrained visual-language encoder and stochastic action expert and is intended for complex tasks or real-world deployment.

VLRLA-simp: We pair a frozen visual-language encoder with a trainable recurrent action module (RLA). The RLA consists of a Transformer with recurrent memory and a multilayer perceptron (MLP) that produces deterministic actions, as illustrated in the upper part of Figure 1:

$$
\begin{array} { r } { \begin{array} { c } { z _ { t } = f _ { \bar { \theta } _ { \mathrm { V L M } } } ( o _ { t } , \ell ) , } \\ { \ ( M _ { t } , \hat { a } _ { t : t + h } ) = \pi _ { \theta _ { \mathrm { R L A } } } ( q _ { t } , z _ { t } , a _ { t - 1 : t - 1 + h } , r _ { t - 1 } , d _ { t - 1 } \mid M _ { t - 1 } ) , } \end{array} } \end{array}\tag{7}
$$

where $z _ { t }$ denotes the prefix embeddings $z _ { t } ^ { \mathrm { e m b } }$ introduced above, and $q _ { t }$ is the proprioceptive state. The RLA parameters $\theta _ { \mathrm { R L A } }$ are trainable, while the encoder parameters $\bar { \theta } _ { \mathrm { V L M } }$ remain frozen. We

train this deterministic policy by minimizing a reconstruction loss:

$$
\mathcal { L } ( \theta _ { \mathrm { R L A } } ) = \mathbb { E } \mathcal { C } \sim \mathcal { D } \left[ \sum _ { t = 0 } \mathrm { M S E } \left( \hat { a } _ { t : t + h } , a _ { t : t + h } \right) \right] .\tag{8}
$$

where MSE denotes mean squared error and $\scriptstyle a _ { t : t + h }$ is the target action chunk in the dataset.

VLRLA-bridge: Our second variant reuses both the pretrained visual-language encoder and the action expert. A trainable adaptation module connects these frozen components by modifying the encoder’s KV cache, as illustrated in the lower part of Figure 1. Using the $\pi _ { 0 . 5 }$ architecture as an example, the policy takes the form

$$
\begin{array} { r l } & { \quad z _ { t } = f _ { \bar { \theta } _ { \mathrm { V L M } } } ( o _ { t } , \ell ) , } \\ & { \quad ( M _ { t } , \Delta z _ { t } ) = f _ { \theta _ { \mathrm { R L } } } ( q _ { t } , z _ { t } , a _ { t - 1 : t - 1 + h } , r _ { t - 1 } , d _ { t - 1 } \mid M _ { t - 1 } ) , } \\ & { \quad \quad z _ { t } ^ { \prime } = z _ { t } + \Delta z _ { t } , \qquad a _ { t : t + h } \sim \pi _ { \bar { \theta } _ { \mathrm { A } } } ( \cdot \mid q _ { t } , z _ { t } ^ { \prime } ) , } \end{array}\tag{9}
$$

where $z _ { t }$ denotes the KV cache $z _ { t } ^ { \mathrm { K V } }$ introduced above. The bridge, with trainable parameters $\theta _ { \mathrm { R L } }$ produces a residual $\Delta z _ { t }$ with the same structure as $z _ { t }$ , yielding the adapted cache $z _ { t } ^ { \prime } .$ . The frozen action expert, parameterized by $\bar { \theta } _ { \mathrm { A } } .$ , generates actions conditioned on the adapted cache and the proprioceptive state. Through this residual modification, interaction history influences the expert’s action generation. Further implementation details are provided in Section A.1.

The pretrained $\pi _ { 0 . 5 }$ action expert generates continuous action chunks through iterative flow-based sampling. We therefore train VLRLA-bridge with the corresponding flow-matching objective, propagating gradients through the frozen expert to the bridge:

$$
\mathcal { L } ( \theta _ { \mathrm { R L } } ) = \mathbb { E } _ { \mathcal { C } \sim \mathcal { D } , \epsilon , \tau } \left[ \left| \left| v _ { \bar { \theta } _ { \mathrm { A } } } ( a _ { t : t + h } ^ { \tau } , \tau , q _ { t } , z _ { t } ^ { \prime } ) - ( \epsilon - a _ { t : t + h } ) \right| \right| _ { 2 } ^ { 2 } \right] .\tag{10}
$$

where $a _ { t : t + h } ^ { \tau } = ( 1 - \tau ) a _ { t : t + h } + \tau \epsilon$ is the interpolated noisy action chunk, $\epsilon \sim \mathcal { N } ( 0 , I )$ is Gaussian noise, and $\dot { \tau } \in [ 0 , 1 ]$ is the interpolation time. Although the expert is frozen, gradients backpropagate to $\theta _ { \mathrm { R L } }$ through $z _ { t } ^ { \prime }$

## 4.2 LEARNING HISTORY CONSTRUCTION

Training VLRLA requires learning histories that capture how behavior improves with accumulated interaction experience. To accommodate different data sources and collection settings, our framework supports three strategies for constructing the training dataset: simulation RL, curriculum-based human-in-the-loop collection, and hindsight-based assembly. Details are in Section A.2

Simulation RL. Following AD (Laskin et al., 2022a), we collect trajectories throughout an RL training process in simulation. This approach can use algorithms such as PPO and GRPO (Shao et al., 2024) to fine-tune base VLA models such as Pi05, LingbotVLA, and FastWAM Yuan et al. (2026). Unlike AD, we find that resetting the environment to the same initial state after every episode termination or truncation is unnecessary and departs from standard practice in robotic tasks that use domain randomization (Dai et al., 2026; Chen et al., 2025; 2026). Instead, in each training round, we sample a batch of environment instances through domain randomization within the same task, such as “picking up diverse bottles”. We then collect on-policy trajectories with the current VLA model, store them as part of the training dataset, and perform an online RL update to obtain the next policy checkpoint. We repeat this process for several rounds. The resulting dataset captures successive rounds of interaction as task performance improves and is used to train VLRLA to transfer the source RL algorithm’s adaptation behavior to unseen tasks.

Human-in-the-Loop Curriculum. Human-in-the-loop (HITL) correction is widely used in robot learning to improve robot behavior. Starting from an existing VLA policy, we construct a training dataset by progressively increasing the extent of human intervention. We first collect trajectories generated entirely by the base VLA policy, without human correction. In the next stage, a human operator uses teleoperation to correct the first $1 / 3$ of each episode, after which the VLA policy completes the remainder. We then increase the fraction of each episode under human intervention from $1 / 3$ to $2 / 3$ and finally to $3 / 3$ , with the final stage consisting almost entirely of human demonstrations. This curriculum produces a dataset of progressively improved behavior, ordered from unassisted policy rollouts through partial corrections to human expert trajectories. These sequences can be used to train VLRLA to adapt to unseen environments and gradually improve task performance.

Hindsight Relabeling. Hindsight experience replay (HER) (Andrychowicz et al., 2017) reuses trajectories by relabeling them with alternative goals. For a language-conditioned task, let $\ell _ { 1 }$ and $\ell _ { 2 }$ denote two language instructions, and let $R _ { \ell }$ denote the instruction-conditioned reward function. A transition can be relabeled as follows:

$$
\begin{array} { r } { ( o _ { t } , a _ { t } , R _ { \ell _ { 1 } } ( o _ { t } , a _ { t } , o _ { t + 1 } ) , o _ { t + 1 } , \ell _ { 1 } ) \longmapsto ( o _ { t } , a _ { t } , R _ { \ell _ { 2 } } ( o _ { t } , a _ { t } , o _ { t + 1 } ) , o _ { t + 1 } , \ell _ { 2 } ) . } \end{array}\tag{11}
$$

For example, a trajectory that places a power adapter in a green plate receives a reward of 1 under the instruction “put the power adapter into the green plate” but a reward of 0 under “put the power adapter into the yellow plate”. This relabeling allows us to assemble training histories without additional data collection: for the task “put the power adapter into one of the plates” (more in Section 6.2), we can collect successful demonstrations of placement in either the green or yellow plate and use HER relabeling to assemble histories of progressively adaptive behavior.

## 5 SIMULATION EXPERIMENTS

In this part, we conduct simulation experiments to study the following research questions: 1) How is VLRLA compared to other algorithms? 2) Can VLRLA learn to adapt using different VLM backbones? 3) How does context length affect VLRLA’s ability to adapt? 4) Could the history of one rollout be transferred? We use VLRLA-simp in 2, 3 and 4 for its simplicity and better reproducibility and use VLRLA-bridge in 1 as its pretrained modules could benefit the multi-task learning most.

We use the RoboTwin benchmark (Chen et al., 2025) as our testbed for its diverse and challenging tasks, and RLinf (Yu et al., 2025) to collect simulation data through online RL (Simulation RL in Section 4.2). We compute success rates by evaluating policies across 128 randomized environment instances and averaging the outcomes.

Table 1: Algorithm characteristics and success rates (%) on seen and unseen RoboTwin tasks. $\Delta$ denotes the improvement of VLRLA-bridge over $\pi _ { 0 . 5 }$ in percentage points. $" / "$ indicates not applicable. The highest success rate in each row is shown in bold, including all ties.
<table><tr><td></td><td>π0.5</td><td>GRPO</td><td>RoboTTT</td><td>VLRLA-bridge</td><td> $\Delta$ </td></tr><tr><td>In-context adaptation Sample efficiency</td><td>No 1</td><td>No Low</td><td>Yes Moderate</td><td>Yes Moderate</td><td>1 1</td></tr><tr><td colspan="6">Seen tasks</td></tr><tr><td>handover_mic</td><td>38.3%</td><td>32.8%</td><td>52.3%</td><td>78.9%</td><td>40.6%</td></tr><tr><td>pick_diverse_bottles</td><td>58.6%</td><td>77.3%</td><td>71.1%</td><td>75.8%</td><td>17.2%</td></tr><tr><td>place_a2b_left</td><td>67.2%</td><td>81.2%</td><td>70.3%</td><td>71.9%</td><td>4.7%</td></tr><tr><td>place_object_stand</td><td>71.1%</td><td>87.5%</td><td>76.6%</td><td>90.6%</td><td>19.5%</td></tr><tr><td>place-phone_stand</td><td>36.7%</td><td>50.8%</td><td>47.7%</td><td>50.8%</td><td>14.1%</td></tr><tr><td>scan_object</td><td>19.5%</td><td>25.8%</td><td>28.9%</td><td>28.9%</td><td>9.4%</td></tr><tr><td>Seen average</td><td>48.6%</td><td>59.2%</td><td>57.8%</td><td>66.2%</td><td>17.6%</td></tr><tr><td colspan="6">Unseen tasks</td></tr><tr><td>adjust_bottle</td><td>95.3%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>4.7%</td></tr><tr><td>move_stapler-pad</td><td>24.2%</td><td>38.3%</td><td>31.2%</td><td>36.7%</td><td>12.5%</td></tr><tr><td>place_a2b_right</td><td>61.7%</td><td>78.1%</td><td>69.5%</td><td>69.5%</td><td>7.8%</td></tr><tr><td>place_dual_shoes</td><td>17.2%</td><td>21.1%</td><td>19.5%</td><td>21.1%</td><td>3.9%</td></tr><tr><td>place_empty-cup</td><td>93.0%</td><td>99.2%</td><td>97.7%</td><td>97.7%</td><td>4.7%</td></tr><tr><td>stamp_seal</td><td>28.1%</td><td>28.1%</td><td>33.6%</td><td>38.3%</td><td>10.2%</td></tr><tr><td>Unseen average</td><td>53.3%</td><td>60.8%</td><td>58.6%</td><td>60.6%</td><td>7.3%</td></tr><tr><td>Overall average</td><td>50.9%</td><td>60.0%</td><td>58.2%</td><td>63.4%</td><td>12.4%</td></tr></table>

## 5.1 HOW DOES VLRLA COMPARE WITH OTHER ALGORITHMS?

We first compare VLRLA with baseline methods on seen and unseen RoboTwin tasks. We evaluate three categories of algorithms:

• Supervised fine-tuning: $\pi _ { 0 . 5 }$ fine-tuned with supervised learning serves as the baseline. This model also serves as the base model for below methods.

• RL fine-tuning: GRPO (Shao et al., 2024). We use the official implementation provided by RLinf. To support on-policy data collection in parallel environments, our training setup requires 8 NVIDIA A100-80G GPUs.

• In-context adaptation: our VLRLA-bridge and RoboTTT (Jiang et al., 2026). We train both methods on the same multi-task dataset collected by fine-tuning $\pi _ { 0 . 5 }$ on six seen tasks, but omit reward and termination information to RoboTTT. During deployment, both methods adapt with accumulated interaction histories while reward and termination are omitted again for RoboTTT. We use our own implementation of RoboTTT, as its official code has not been released.

As shown in Table 1, VLRLA-bridge improves over the initial $\pi _ { 0 . 5 }$ policy on all 12 tasks, with an average gain of 12.4 percentage points. Its overall average success rate exceeds that of GRPO, an online RL method that explicitly updates policy parameters through stochastic gradient descent. Compared with RoboTTT, VLRLA-bridge also achieves a higher overall average success rate. This gap reflects the absence of reward signals in RoboTTT hinders the model’s adaptation capability. On the six seen tasks, it improves over $\pi _ { 0 . 5 }$ by an average of 17.6 percentage points and achieves a higher average success rate than GRPO.

On the six unseen tasks, VLRLA-bridge improves over $\pi _ { 0 . 5 }$ on every task, with an average gain of 7.3 percentage points. Its average success rate reaches $6 0 . 6 \% ,$ approaching that of GRPO. It also achieves a higher average success rate than RoboTTT on unseen tasks, outperforming it on three tasks and matching its success rate on the remaining three. These results suggest that supervised pretraining on learning histories from seen tasks enables the model to learn an adaptation capability that generalizes to unseen tasks.

## 5.2 CAN VLRLA LEARN TO ADAPT USING DIFFERENT VLM BACKBONES?

Figure 2 evaluates whether VLRLA-simp can learn to adapt using representations from different pretrained backbones: $\pi _ { 0 . 5 } .$ , lingbotvla, and the world action model fastwam. All three configurations improve as interaction experience accumulates, supporting the framework’s applicability across different representation backbones. For both lingbotvla and $\pi _ { 0 . 5 }$ , success rates generally increase with accumulated interaction history on both tasks. $\mathrm { A l - }$ though fastwam also learns to adapt, its suc-

![](images/94ba57f936a4b3b9074a1b9ed52d5ff32c796749c0e8f9edcea0e5830e9fe442.jpg)  
Figure 2: Adaptation across VLM modules. Shaded area is 1 standard deviation across 5 seeds.

cess rate on place empty cup remains lower than those of the other two backbones. We attribute this performance gap to the constrained prefix output used in our FastWAM integration, which limits the visual and language information available to the adaptation and policy module (see the Section A.4 for details). Together, these results suggest that $\mathtt { V I R L A - } s$ imp can learn to adapt across different backbones and highlight the importance of strong visual-language representations.

## 5.3 HOW DOES CONTEXT LENGTH AFFECT VLR $\mathsf { \Omega } _ { \mathsf { A } } - \mathsf { S } \mathrm { \Omega } \mathbb { I } \mathrm { \Omega } \mathrm { \nabla } \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega }$ ABILITY TO ADAPT?

Figure 3 examines how context length and interaction history affect adaptation. With context length $h = 4$ and $h = 8 ,$ VLRLA-simp achieves broadly similar performance on both $\mathtt { a d j u s t \_ b o t t 1 e }$ and $\mathtt { p l a c e \_ e m p t y \_ c u p , }$ with success rates gradually improving as interaction experience accumulates. In the $h = 8$ (no history) setting, we explicitly zero out the history memory when generating action chunks for the current episode and observe a substantial drop in success rate: performance remains near

![](images/e4f5357f53b79f1aa5f90f4dad5a2625a32e4f0e9f9887acf0e72336186ce211.jpg)  
Figure 3: Effect of context and memory. Shaded area is 1 standard deviation across 5 seeds.

zero, with no sustained improvement across episodes. We also evaluate an $h = 8$ (repeat history)

variant that reuses the first episode’s history at every subsequent step. The success rate improves at the first step after this history becomes available, then remains flat because no new interaction information is incorporated. Together, these experiments highlight the importance of both access to interaction history and its continued accumulation for adaptation.

## 5.4 COULD THE HISTORY OF ONE ROLLOUT BE TRANSFERRED?

Figure 4 examines whether interaction history collected in one rollout can improve initial performance in another rollout of the same task with different initial states or configurations. With this history loaded, the warmup rollout begins with higher success rates on both tasks: approximately 0.7 versus 0.55 for the capture and full rollouts on adjust bottle, and 0.85 versus 0.65 on place empty cup. The capture rollout supplies the interaction history used by warmup, while full denotes the original roll-

![](images/d3f07f10b855ea6d7e5471c96bb14d5c56e5f34094d2bbeeab2c20e4d6102a01.jpg)  
Figure 4: History transfer to improve initial task performance. Shaded regions indicate one standard deviation across five seeds.

out without transferred history. These results show that interaction history can be transferred between rollouts of the same task to improve performance in a new configuration.

## 6 REAL WORLD EXPERIMENTS

![](images/21e10255bc1c452d95dc0e21c5e4391c77c56f33047a5e54ebb52925e3dcd67a.jpg)  
Figure 5: Left 3 figures depict the fruits placement. Right 3 figures depict the pepper placement.

## 6.1 FRUIT-PLACEMENT

This task requires the robot arm to pick up a fruit and place it in a vessel as in Figure 5. The instruction is “put the {fruit} into {vessel}”. The fruits used during training are apple, starfruit, and grape; the vessels are bowl and plate. We evaluate whether the success rate of VLRLA-bridge improves as experience accumulates across these fruit–vessel combinations. We also evaluate transfer to an unseen setting with a red pepper and a metal pot, neither of which appears in the training dataset. The dataset contains approximately 80 teleoperation episodes (used to fine-tune the $\pi _ { 0 . 5 }$ base model), followed by 3 × 20 episodes with HITL corrections (described in Humanin-the-Loop Curriculum, Section 4.2). The videos for this task are in Supplemental Materials.

Figure 6a reports fruit-placement performance across 45 evaluation runs. Here, VLRLA – VLRLA denote VLRLA-bridge evaluated over four successive episodes with accumulating interaction history, where $\mathrm { { V L R L A _ { 1 } } }$ corresponds to the first episode with no prior interaction history. After a slight initial decline relative to $\pi _ { 0 . 5 }$ , VLRLA steadily improves with accumulated interactions and surpasses the baseline, demonstrating its ability to use accumulated experience to improve fruit-placement performance.

![](images/b606fce5626220ede3a22794f2049c5b72a7e28e8bdec7fc2434cb2553c37ac4.jpg)  
(a) Results for seen fruit task averaged in 45 runs.

![](images/230e65176f908022af065589540a862c736bbba5e3ef70fe5dc68f015b879e94.jpg)  
(b) Result for unseen fruit task in 25 runs.  
Figure 6: Success rates on fruitplacement tasks.

For the unseen task “put the pepper into the $p o t , ^ { \prime \prime }$ we use a   
medium-sized red pepper whose asymmetric shape makes it more challenging to grasp. Figure 6b reports performance across 25 evaluation runs. VLRLA outperforms the $\pi _ { 0 . 5 }$ baseline steadily. These results indicate that VLRLA can use accumulated experience to improve performance even when neither the object nor the vessel was encountered during training.

Table 2 presents results by fruit type, comparing $\pi _ { 0 . 5 }$ with $\mathrm { V L R L A _ { 1 } { - } V L R L A _ { 4 } } .$ For apple and starfruit, success rates increase steadily as interaction experience accumulates and exceed the $\pi _ { 0 . 5 }$ baseline at ${ \mathrm { V L R L A _ { 2 } } } .$ Performance on grape follows a less consistent trend, fluctuating across adaptation stages. Closer inspection suggests that this difference arises from the grape’s small size and softness, combined with the relatively high force applied by the gripper. These properties make grape grasping less sensitive to the grasp location, limiting the benefit of adaptation. Apple and starfruit, however, require more precise grasp placement, leaving greater room for improvement.

Table 2: Fruit-placement success-rate breakdown by fruit type, with 15 runs per fruit.
<table><tr><td></td><td> $\pi _ { 0 . 5 }$ </td><td> $\mathrm { { V L R L A _ { 1 } } }$ </td><td> $\mathrm { V L R L A _ { 2 } }$ </td><td> $\mathrm { V L R L A _ { 3 } }$ </td><td> $\mathrm { V L R L A _ { 4 } }$ </td><td> $\mathbf { V L R L A _ { 4 } } - \pi _ { 0 . 5 }$ </td></tr><tr><td>Apple</td><td>46.7%</td><td>40.0%</td><td>53.3%</td><td>53.3%</td><td>60.0%</td><td>+13.3%</td></tr><tr><td>Starfruit</td><td>40.0%</td><td>26.7%</td><td>46.7%</td><td>60.0%</td><td>66.7%</td><td>+26.7%</td></tr><tr><td>Grape</td><td>53.3%</td><td>60.0%</td><td>46.7%</td><td>66.7%</td><td>60.0%</td><td>+6.7%</td></tr></table>

## 6.2 CROSS-EPISODE MEMORY TASK

In this task, we examine whether VLRLA-bridge can adapt its behavior using reward feedback from previous episodes. A green plate, a yellow plate, and a power adapter are placed on a table, and the policy receives the underspecified instruction “put the power adapter into the plate”. The target plate’s color determines the reward but is deliberately withheld from the policy. Placing the adapter in the target plate yields a reward of 1, while any other outcome yields a reward of 0. The left of Figure 7 illustrates the task setup. This task tests whether the policy can employ reward feedback across episode boundaries and use it to guide subsequent plate choices. We use hindsight relabeling to dynamically assemble the dataset using only expert demonstrations for “put the power adapter into the green plate” and “put the power adapter into the yellow plate” (details in Section A.3).

![](images/05b4b4ce5a9f47b86af2db629c4540647cab1706a244d19cb0ef9a6f1dfea106.jpg)

![](images/233d36d9bb00556966b32c1992e09883f3fe0cca264c0eccab2f5749fdaa7455.jpg)  
Figure 7: Left part: illustrative diagram for cross-episodic-memory task. The digit in circle denotes the reward label passed to policy after the episode. Right part: success rate averaged in 44 runs.

The right panel of Figure 7 shows a steady increase in success rate after the first episode, when reward feedback becomes available. Performance quickly rises in subsequent episodes, suggesting that the policy can use reward feedback from earlier episodes to guide later choices. Videos of this task are available in the supplementary material.

## 7 CONCLUSION

VLRLA enables deployment-time adaptation through in-context reinforcement learning, conditioning action generation on accumulated interactions, rewards, and termination signals while keeping model parameters fixed. Our general framework accommodates multiple strategies for constructing learning histories and policy instantiations.

Through extensive experiments in simulation and real-world settings, we demonstrate that VLRLA can improve task success using accumulated interaction experience, including challenging unseen tasks. Simulation studies show adaptation with different pretrained backbones and highlight the importance of continually updating interaction history. Real-world experiments further show improved manipulation performance and suggest that reward feedback retained across episodes can inform subsequent action choices.

## REFERENCES

RT-1: Robotics Transformer for real-world control at scale. https://ai.googleblog.com/2022/12/rt-1- robotics-transformer-for-real.html, December 2022.

Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokin sky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, and Ury Zhilinsky. π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 5055–5065, Red Hook, NY, USA, December 2017. Curran Associates Inc. ISBN 978-1-5108-6096-4.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Biman ual Robotic Manipulation, August 2025.

Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies, July 2026.

Open X.-Embodiment Collaboration, Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, Antonin Raffin, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Scholkopf,¨ Brian Ichter, Cewu Lu, Charles Xu, Chelsea Finn, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Chuer Pan, Chuyuan Fu, Coline Devin, Danny Driess, Deepak Pathak, Dhruv Shah, Dieter Buchler, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Federico Ceola, Fei¨ Xia, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Giulio Schiavi, Gregory Kahn, Hao Su, Hao-Shu Fang, Haochen Shi, Heni Ben Amor, Henrik I. Christensen, Hiroki Furuta, Homer Walke, Hongjie Fang, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jan Peters, Jan Schneider, Jasmine Hsu, Jeannette Bohg, Jeffrey Bingham, Jiajun Wu, Jialin Wu, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jitendra Malik, Jonathan Tompson, Jonathan Yang, Joseph J. Lim, Joao Silv˜ erio, Junhyek Han, Kan-´ ishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Zhang, Krishan Rana, Krishnan Srinivasan, Lawrence Yunliang Chen, Lerrel Pinto, Liam Tan, Lionel Ott, Lisa Lee, Masayoshi Tomizuka, Maximilian Du, Michael Ahn, Mingtong Zhang, Mingyu Ding, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J. Joshi, Niko Suenderhauf, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Pannag R. Sanketi, Paul Wohlhart, Peng Xu, Pierre Sermanet, Priya Sundaresan, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart´ın-Mart´ın, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shuran Song, Sichun Xu, Siddhant Haldar, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Sudeep Dasari, Suneel Belkhale, Takayuki Osa, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Vidhi Jain, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiaolong Wang, Xinghao Zhu, Xuanlin Li, Yao Lu, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zhuo Xu, and Zichen Jeff Cui. Open X-Embodiment: Robotic Learning Datasets and RT-X Models, October 2023.

Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, and Joyce Chai. RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies, May 2026.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, June 2019.

Letian Fu, Huang Huang, Gaurav Datta, Lawrence Yunliang Chen, William Chung-Ho Panitch, Fangchen Liu, Hui Li, and Ken Goldberg. In-Context Imitation Learning via Next-Token Prediction, September 2024.

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and Applications, January 2019.

Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. \$πˆ{\*} {0.6}\$: A VLA That Learns From Experience, November 2025a.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z.

Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. \$π {0.5}\$: A Vision-Language-Action Model with Open-World Generalization, April 2025b.

Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, and Linxi ”Jim” Fan. RoboTTT: Context Scaling for Robot Policies, July 2026.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model, September 2024.

Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, D. J. Strouse, Steven Hansen, Angelos Filos, Ethan Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, and Volodymyr Mnih. In-context Reinforcement Learning with Algorithm Distillation, October 2022a.

Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, D. J. Strouse, Steven Stenberg Hansen, Angelos Filos, Ethan Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, and Volodymyr Mnih. In-context Reinforcement Learning with Algorithm Distillation. In The Eleventh International Conference on Learning Representations, September 2022b.

Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning, September 2025.

Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What Can RL Bring to VLA Generalization? An Empirical Study, January 2026.

Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning, March 2025.

Mingjie Pan, Siyuan Feng, Qinglin Zhang, Xinchen Li, Jianheng Song, Chendi Qu, Yi Wang, Chuankang Li, Ziyu Xiong, Zhi Chen, Yi Liu, and Jianlan Luo. SOP: A Scalable Online Post-Training System for Vision-Language-Action Models, January 2026.

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, June 2011.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs], August 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, April 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering Your Diffusion Policy with Latent Space Reinforcement Learning, June 2025.

Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, Yiyu Ren, Kejia Zhang, Hui Yu, Jingmei Zhao, Shuai Zhou, Zhenqi Qiu, Houlong Xiong, Ziyu Wang, Zechen Wang, Ran Cheng, Yong-Lu Li, Yongtao Huang, Xing Zhu, Yujun Shen, and Kecheng Zheng. A Pragmatic VLA Foundation Model, June 2026.

Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, and Liyiming Ke. RL Token: Bootstrapping Online RL with Vision-Language-Action Models, April 2026.

Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, Zixiao Huang, Mingjie Wei, Yuqing Xie, Ke Yang, Bo Dai, Zhexuan Xu, Jiakun Du, Xiangyuan Wang, Xu Fu, Letong Shi, Zhihao Liu, Kang Chen, Weilin Liu, Gang Liu, Boxun Li, Jianlei Yang, Zhi Yang, Guohao Dai, and Yu Wang. RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation, December 2025.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do World Action Models Need Test-time Future Imagination?, March 2026.

## A APPENDIX

## A.1 IMPLEMENTATION DETAILS

## A.1.1 VLRLA-BRIDGE

We introduce the implementation details for our VLRLA-bridge variant. Its hyperparameters are reported in Table S1.

To reduce the memory cost of cross attention to a large multi-layer KV cache extracted from the base $\pi _ { 0 . 5 }$ model, we introduce a learned summarization module. Its queries attend across valid cache positions and layers, producing a compact set of KV summarization tokens. We then provide these summary tokens to the cross attention module of action query tokens from which the bridge extracts the information used to modify the current cache.

For input processing, we use type-specific encoders together with learned absolute positional encodings. Proprioceptive states and individual actions are mapped into a common token space through separate linear projections, while reward, termination and truncation use separate binary embedding tables. We use learned absolute position and index embeddings at multiple levels: summaryquery token embeddings and action-query token embeddings distinguish their respective token slots, action-position embeddings encode order within each action chunk, and transition-position embeddings locate every token within the whole trial. Finally, a shared linear projection maps these position-aware tokens to the Transformer’s hidden dimension.

To translate the output of VLRLA-bridge as KV cache modifications, we pool the action-query token’s outputs into a representation $u _ { t }$ and introduce a low-rank residual adapter. For a native cache vector $x _ { t , \ell , p } ^ { s } ,$ the adapter computes

$$
\begin{array} { r l } & { \quad c _ { t , j , p } ^ { s } = \operatorname { t a n h } \bigl ( W _ { z } z _ { t , j , p } ^ { s } + W _ { u } u _ { t } + e _ { p } + e _ { j } + e _ { s } \bigr ) , } \\ & { \quad \Delta z _ { t , j , p } ^ { s } = m _ { t , p } \ : ( B _ { j } ^ { s } ) ^ { \top } c _ { t , j , p } ^ { s } , } \\ & { \quad \quad \quad z _ { t } ^ { \prime } = z _ { t } + \Delta z _ { t } , } \end{array}\tag{12}
$$

where the embeddings identify cache position, layer, and type, and $m _ { t , p }$ excludes padded positions. All learned adapter parameters belong to $\theta _ { \mathrm { R L } }$ . This parameterization combines current cache content with historical context while restricting residuals to learned low-rank bases. We initialize the output bases B<sup>s</sup> to zero, yielding $z _ { t } ^ { \prime } \ = \ z _ { t }$ at initialization and preserving the pretrained policy’s initial behavior.

Table S1: Common architecture and training settings for VLRLA-bridge
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Transformer layers / attention heads</td><td>6/8</td></tr><tr><td>Hidden / feedforward dimension</td><td>512 / 1024</td></tr><tr><td>Number of summary / action queries</td><td>36 / 24</td></tr><tr><td>Residual-adapter rank</td><td>64</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Residual output initialization</td><td>Zero</td></tr><tr><td>Sequence / segment length</td><td>48 / 6 transitions</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td>10−4</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Learning-rate schedule</td><td>10% warmup, cosine decay</td></tr><tr><td>Gradient clipping norm</td><td>1.0</td></tr><tr><td>Global batch size</td><td>64</td></tr></table>

## A.2 LEARNING HISTORY CONSTRUCTION PROCEDURES

## A.2.1 IN SIMULATION RL DATASET

We document the procedure and hyperparameters for our collection of In-simulation RL Dataset here:

Task and observations. We fine-tune $\pi _ { 0 . 5 }$ with proximal policy optimization (PPO) on the various tasks in Robotwin as shown in main text (we use adjust bottle for example here, their general configurations are similar). The embodiment is ALOHA–AgileX.

Policy parameterization. We initialize the policy from an existing $\pi _ { 0 . 5 }$ checkpoint configured for ALOHA control. The vision–language backbone remains frozen, while the action expert and associated trainable policy parameters are optimized without low-rank adapters. Each policy call generates a chunk of 50 actions, each with 14 dimensions, using five flow integration steps. Training uses the flow-SDE sampler with a noise coefficient of 0.3: one integration step is selected for stochastic exploration, and the remaining steps follow the deterministic flow update. PPO surrogate for the flow models are calcuated based on method from the Rlinf (Yu et al., 2025). A value head predicts returns from pooled action-expert features; its inputs are detached so that the value loss does not update the policy representation.

Reward and optimization. The learning signal is a sparse success reward provided by the environment wrapper. With unit reward scaling and relative rewards enabled, a reward of one is assigned when an episode first becomes successful. Rewards are aggregated at the action chunk level. We estimate advantages using generalized advantage estimation with $\gamma = 0 . 9 9$ and $\lambda = 0 . 9 5$ , and normalize the resulting advantages before optimization. The actor uses a clipped PPO objective with ratio bounds [0.8, 1.2] and an additional dual-clipping coefficient of 3.0. The critic uses a clipped Huber loss with value-clipping threshold 0.2 and Huber parameter 10. Actor and critic losses are added with equal weight, with no entropy bonus or reference-policy KL penalty. AdamW uses separate learning rates of $5 \times 1 0 ^ { - 6 }$ for the policy and $1 0 ^ { - 4 }$ for the value head.

Rollout and update schedule. Training comprises 25 PPO iterations. Each iteration collects four rollout rounds from 256 parallel environments, with a wrapper horizon of 400 environment steps per round, corresponding to 8 action-chunk decisions per environment. The resulting rollout tensor therefore contains 8,192 chunk slots per iteration; slots following episode termination are excluded from the training objective by a loss mask. Each rollout batch is reused for five optimization epochs. The global optimization minibatch contains 2,048 chunk samples, with 32 samples per GPU microbatch and eight gradient-accumulation steps across eight GPUs. Training uses fully sharded data parallelism on a single node, with offloading enabled for the actor, rollout workers, and environment workers.

Evaluation. The policy is evaluated before training and every five PPO iterations using 128 parallel environments and one evaluation rollout round. Evaluation uses a separately specified seed list with fixed reset-state identifiers and the same 400-step wrapper horizon. Table S2 summarizes the training hyperparameters.

Fianl Dataset. Each recorded decision associates a task instruction, three RGB observations, and proprioceptive state with a 50×14 action chunk, accompanied by reward, termination, and truncation arrays. Both successful and unsuccessful rollouts are retained. A complete 25-iteration run provides a nominal capacity of 204,800 chunk slots $( 2 5 \times 4 \times 2 5 6 \times 8 )$

Table S2: PPO fine-tuning configuration for $\pi _ { 0 . 5 }$ on Robotwin tasks.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Policy / embodiment</td><td> $\pi _ { 0 . 5 } / \mathrm { A L O H A - A g i l e X }$  Head RGB and two wrist RGB views</td></tr><tr><td>Visual inputs Frozen component</td><td>Vision-language backbone</td></tr><tr><td>Action dimension / chunk length</td><td>14/50</td></tr><tr><td></td><td></td></tr><tr><td>Flow integration steps</td><td>5</td></tr><tr><td>Training sampler / noise coefficient</td><td>Flow-SDE / 0.3</td></tr><tr><td>PPO iterations Training / evaluation environments</td><td>25 256 /128</td></tr><tr><td>Rollout rounds per iteration</td><td>4</td></tr><tr><td>Wrapper episode limit / rollout horizon</td><td>400 / 400 steps</td></tr><tr><td>Chunk slots per iteration (before masking)</td><td>8,192</td></tr><tr><td></td><td></td></tr><tr><td>Memory detach frequency</td><td>12 chunks (2 episodes) 5</td></tr><tr><td>Optimization epochs per iteration</td><td>2,048 chunk samples</td></tr><tr><td>Global optimization minibatch</td><td></td></tr><tr><td>Microbatch per GPU</td><td>32 chunk samples</td></tr><tr><td>Gradient-accumulation steps</td><td>8</td></tr><tr><td>Reward and log-probability aggregation</td><td>Action chunk level</td></tr><tr><td>Advantage estimator / normalization</td><td>GAE / enabled</td></tr><tr><td>Discount factor γ / GAE parameter λ</td><td>0.99 / 0.95</td></tr><tr><td>PPO clipping parameter / dual-clipping coefficient</td><td>0.2 / 3.0</td></tr><tr><td>Value clipping / Huber parameter</td><td>0.2 / 10.0</td></tr><tr><td>Value-loss weight</td><td>1.0</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Policy / value-head learning rate</td><td> $5 \times 1 0 ^ { - 6 } / 1 0 ^ { - 4 }$ </td></tr><tr><td>Adam coefficients  $( \beta _ { 1 } , \beta _ { 2 } ) / \epsilon$ </td><td> $\left( 0 . 9 , 0 . 9 5 \right) / 1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay / gradient-norm clipping</td><td>0.01 / 1.0</td></tr><tr><td></td><td></td></tr><tr><td>Actor seed / base environment seed</td><td>1234/0</td></tr></table>

## A.2.2 HUMAN-IN-THE-LOOP CORRECTION DATASSET

Dataset. Our human-in-the-loop (HITL) manipulation dataset comprising 391 episodes organized into four ordered stages. The dataset covers six subtasks defined by combinations of three fruit categories (apple, grape, and starfruit) and two target object categories (bowl and plate). Each episode contains 12 action-chunk decisions. At each decision, the stored observation includes an RGB im age from the main camera, an RGB image from a wrist-mounted camera, a seven-dimensional robot state, and a natural-language instruction. The recorded action chunk contains 20 seven-dimensional action vectors. The main-camera and wrist-camera images have resolutions of 540 × 960 and 480×640 pixels, respectively. In total, the dataset contains 4,692 action-chunk decisions and 93,840 recorded action vectors.

Each episode additionally includes human-provided annotations identifying the fruit category, target object category, and binary task success. Both successful and unsuccessful episodes are retained. Table S3 summarizes the dataset composition. The reported success proportions describe the recorded data and are not evaluation results for the trained model.

Construction of training sequences. Let $\mathcal { D } _ { s }$ denote the episode pool for stage $s \in \{ 0 , 1 , 2 , 3 \}$ and let $\mathcal { D } _ { s , c }$ denote its subset associated with subtask c. A training trail is formed by independently sampling one episode from each stage and concatenating the sampled episodes in stage order. Each trial therefore contains $4 \times 1 2 = 4 8$ action-chunk transitions. Sampling is performed with replacement; constructed sequences need not correspond to matched episodes or a single continuous physical trajectory. We retain the complete episode pool at every stage, without dataset subsampling or success-based filtering.

Sampling protocols. We perform same-subtask sampling, we first select one of the six subtasks uniformly and then independently sample an episode of that subtask from each stage:

$$
c \sim \mathrm { U n i f o r m } ( \mathcal { C } ) , \qquad \tau _ { s } \mid c \sim \mathrm { U n i f o r m } ( \mathcal { D } _ { s , c } ) , \qquad | \mathcal { C } | = 6 .\tag{13}
$$

This protocol preserves subtask identity throughout a sequence.

Preprocessing and training objective. Images, instructions, states, and action targets are processed using the input transformations associated with the pretrained $\pi _ { 0 . 5 }$ checkpoint. State and action preprocessing uses the checkpoint’s quantile-normalization statistics. The VLRLA-bridge consumes seven-dimensional normalized states and action histories, while action targets supplied to the frozen action expert are padded to its 32-dimensional action representation.

Table S3: Composition of the HITL dataset. Success denotes the binary human annotation associated with each recorded episode.
<table><tr><td>Stage</td><td>Episodes</td><td>Successful</td><td>Success (%)</td></tr><tr><td>0</td><td>99</td><td>35</td><td>35.35</td></tr><tr><td>1</td><td>95</td><td>53</td><td>55.79</td></tr><tr><td>2</td><td>99</td><td>80</td><td>80.81</td></tr><tr><td>3</td><td>98</td><td>91</td><td>92.86</td></tr><tr><td>Total</td><td>391</td><td>259</td><td>66.24</td></tr></table>

Table S4: Hyperparameters for fruit-placement tasks.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone</td><td>Frozen π0.5</td></tr><tr><td>Trainable module</td><td>Temporal residual-KV bridge</td></tr><tr><td>Episodes per sequence</td><td>4 (one per stage)</td></tr><tr><td>Transitions per episode</td><td>12</td></tr><tr><td>Transitions per sequence</td><td>48</td></tr><tr><td>Action chunk length</td><td>20</td></tr><tr><td>State / recorded action dimension</td><td>7/7</td></tr><tr><td>Transformer-XL layers</td><td>6</td></tr><tr><td>Hidden dimension</td><td>512</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Feed-forward dimension</td><td>1,024</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Action queries</td><td>24</td></tr><tr><td>KV adapter rank</td><td>64</td></tr><tr><td>Segment length</td><td>6 transitions</td></tr><tr><td>Memory detach frequency</td><td>12 chunks (2 episodes)</td></tr><tr><td>Segments per optimizer update</td><td>8 per sequence</td></tr><tr><td>Training objective</td><td>Flow-matching velocity MSE</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td>10−4</td></tr><tr><td>Adam coefficients  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.999)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping norm</td><td>1.0</td></tr><tr><td>Learning-rate schedule</td><td>Linear warmup, cosine decay</td></tr><tr><td>Warmup</td><td>10%</td></tr><tr><td>Global batch size</td><td>64</td></tr><tr><td>Batch size per GPU</td><td>8</td></tr><tr><td>Numerical precision</td><td>BF16 mixed precision</td></tr><tr><td>Optimizer updates per run</td><td>1,680</td></tr></table>

## A.3 CROSS-EPISODE MEMORY DATASET ASSEMBLY

## A.3.1 GENERAL PROCEDURE

We construct training sequences for the cross-episode memory task in Section 6.2 by applying hindsight relabeling to an existing dataset of successful single-episode demonstrations. No additional data collection under ambiguous instructions is required.

Each source demonstration follows a color-specific instruction, such as “put the power adapter into the green plate” or “put the power adapter into the yellow $p l a t e ^ { \prime \prime }$ , which identifies its achieved destination color. We retain the recorded observations, robot states, and actions, but replace every instruction with the underspecified command “put the power adapter into the plate”.

To assemble a trial of K episodes, we sample an initial demonstration with destination color $c _ { o }$ and assign binary feedback $b \in \{ 0 , 1 \}$ }. This feedback determines the latent target color $c _ { n } \colon \mathrm { i f } \ b = 1$ the target is $c _ { o } ; \operatorname { i f } b = 0$ , the target is the alternative color. Neither color is explicitly supplied to the policy; the instruction remains underspecified. For each episode $i = 0 , \ldots , K - 1$ , let $c _ { o } ^ { ( i ) }$ denote its original destination color, with $c _ { o } ^ { ( 0 ) } = c _ { o }$ . We relabel its terminal reward as

$$
r _ { c _ { o } ^ { ( i ) } } ( c _ { n } ) = \left\{ \begin{array} { l l } { { 1 , } } & { { c _ { o } ^ { ( i ) } = c _ { n } \quad \mathrm { ( m a t c h ) } , } } \\ { { 0 , } } & { { c _ { o } ^ { ( i ) } \neq c _ { n } \quad \mathrm { ( m i s m a t c h ) } . } } \end{array} \right.\tag{14}
$$

We sample the remaining K −1 demonstrations from successful episodes reaching $c _ { n }$ . The resulting reward sequence is $( b , 1 , \ldots , 1 )$ : the initial attempt may match or mismatch the target color, while all subsequent episodes demonstrate matching behavior.

We also construct counterfactual pairs with the same initial demonstration but opposite initial feedback. The member with b = 1 continues with demonstrations reaching the original color, while the member with $b = 0$ continues with demonstrations reaching the alternative color. Both members use the same ambiguous instruction and the same augmentation randomness for the initial history. Thus, the same initial attempt leads to different subsequent action targets depending on the feedback, encouraging the policy to use feedback stored in memory to resolve the instruction’s ambiguity.

The dataset contains 185 successful source episodes. We hold out 16 episodes (8 per color) for validation and dynamically assemble training trials from the remaining 169. We apply a curriculum over trial length K: K = 2 for the first third of training updates, followed by alternating $K = 2$ and $K = 6$ for the remaining two-thirds. This curriculum trains immediate adaptation to feedback and retention of the inferred color across subsequent episodes.

## A.3.2 DETAILED SETTINGS

Real-world manipulation dataset. We use a real-world Franka robot arm and a dataset comprising 185 successful demonstrations of placing a power adapter into either a green or a yellow plate. The dataset contains 93 green-target and 92 yellow-target demonstrations. Each episode contains 180 action steps recorded at 15 Hz, corresponding to 12 s of recorded interaction. The dataset therefore comprises 33,300 action steps, or approximately 37 min of interaction. Observations include three RGB camera views (front, side, and wrist) and a 15-dimensional robot state. Actions are represented by seven-dimensional end-effector delta commands, comprising translation, rotation, and gripper control.

Temporal representation and preprocessing. Each episode is partitioned into six nonoverlapping chunks of 30 actions. For each chunk, we retain the three camera images and robot state at its initial timestep, yielding observation indices {0, 30, 60, 90, 120, 150}. Thus, each episode provides six observation-conditioned action targets, each spanning 2 s. RGB images are resized from $4 8 0 \times 6 4 0$ to 224 × 224 using aspect-ratio-preserving padding. Preprocessing normalizes the endeffector quaternion to unit length and resolves its sign ambiguity by making its largest-magnitude component nonnegative. Gripper state and action conventions are aligned with the pretrained policy. State and action normalization uses the quantile statistics bundled with the frozen base-model checkpoint.

Table S5: Dataset specification and adapter-training hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dataset and representation</td><td></td></tr><tr><td>Successful source episodes</td><td>185: 93 green, 92 yellow</td></tr><tr><td>Adapter-training episodes</td><td>169: 85 green, 84 yellow</td></tr><tr><td>Held-out source episodes</td><td>16: 8 per color</td></tr><tr><td>Recording frequency</td><td>15 Hz</td></tr><tr><td>Episode length</td><td>180 action steps (12 s)</td></tr><tr><td>Observation views</td><td>Front, side, wrist RGB</td></tr><tr><td>Source / processed image size</td><td>480 × 640 / 224× 224</td></tr><tr><td>Robot-state / action dimension</td><td>15 /7</td></tr><tr><td>Action horizon</td><td>30 steps per chunk</td></tr><tr><td>Chunks per episode</td><td>6</td></tr><tr><td>Training-trail length</td><td>2 or 6 episodes</td></tr><tr><td>Normalization</td><td>Frozen checkpoint quantile statistics</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Base policy</td><td>Frozen π0.5</td></tr><tr><td>Transformer layers</td><td>6</td></tr><tr><td>Model / feed-forward width</td><td>512 / 1,024</td></tr><tr><td>Summary / adaptation queries</td><td>72 / 48</td></tr><tr><td>Residual adapter rank</td><td>64</td></tr><tr><td>Memory detach frequency</td><td>12 chunks (2 episodes)</td></tr><tr><td>Optimization and curriculum</td><td></td></tr><tr><td>Optimizer</td><td>AdamW, (β1, β2) = (0.9, 0.999)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Peak learning rate</td><td>10−4</td></tr><tr><td>Optimizer updates</td><td>2,790</td></tr><tr><td>Global / per-GPU batch size</td><td>64 / 8 trails</td></tr><tr><td>Numerical precision</td><td>BF16</td></tr><tr><td>Learning-rate warmup</td><td>4% of training</td></tr><tr><td>Learning-rate decay</td><td>Cosine decay after the 60% boundary to zero</td></tr><tr><td>Short-trail stage</td><td>Updates 1–1,256: 2 episodes</td></tr><tr><td>Mixed-trail stage</td><td>Updates 1,257–2,790: alternating 2 and 6 episodes</td></tr><tr><td>Augmentation and evaluation</td><td></td></tr><tr><td>Front-view crop</td><td>95% of each spatial dimension</td></tr><tr><td>Front-view rotation</td><td>Uniform in [—5°, 5°]</td></tr><tr><td>Brightness multiplier</td><td>[0.85, 1.15]</td></tr><tr><td>Contrast multiplier</td><td>[0.8, 1.2]</td></tr><tr><td>Saturation multiplier</td><td>[0.8, 1.2]</td></tr></table>

## A.4 FASTWAM

A possible explanation for FastWAM’s lower performance on place empty cup in Section 5.2 is the representation extracted to downstream policy module. Native FastWAM predicts actions using video key/value representations from multiple backbone layers. For simplicity, our integration implementaion instead conditions on the final video hidden states. Consequently, VLRLA-simp does not retain native FastWAM’s access to intermediate representations.

The encoder also differs substantially in prefix length compared to $\pi _ { 0 . 5 }$ . FastWAM produces 120 visual-language tokens of dimension 3,072, produced by VAE encoding and DiT patching of the combined three-camera image. In comparison, $\pi _ { 0 . 5 }$ receives a prefix of 968 tokens of dimension 2,048. Hence FastWAM’s coarser spatial tokenization may make fine geometric relationships less accessible to the downstream policy module.

These differences may disproportionately affect place empty cup, which requires accurate cup– coaster alignment during grasping, transport, and placement, often under gripper occlusion. Such behavior may benefit from spatial detail and intermediate features that are less accessible from the FastWAM prefix we used. In contrast, adjust bottle may be more tolerant of this representation choice. We therefore consider the representation interface and spatial compression plausible contributors to the observed gap.