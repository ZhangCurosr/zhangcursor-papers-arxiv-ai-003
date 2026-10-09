# PMTRM: Pseudo-Memory Temporal Re-encoding Module for Embodied Policy Learning

Changchuan Yang<sup>1</sup>, Haoxuan Xu<sup>2</sup>, Wenbo Chen<sup>3</sup>, Shuai Ren<sup>4</sup>, Jianlong Zheng<sup>5</sup>, Huarui Zhang<sup>3</sup>, Tianfu Li<sup>3</sup>, Guanzhong Tian<sup>1,6</sup>

Abstract— Robotic manipulation often contains repeated motions whose local observations look similar at different phases. When these phases require different actions, a policy that relies mainly on the current observation may repeat completed motions or switch phases at the wrong time. To address this phase ambiguity, we present the Pseudo-Memory Temporal Re-encoding Module (PMTRM), a lightweight plug-in module with only 7.61M parameters that encodes a bounded history of executed states and actions into a latent sequence for existing policies. To help distinguish phases, a temporal heterogeneity objective penalizes positive similarity between distant positions in this sequence, while anchor and reconstruction losses preserve information needed for action prediction. The reconstruction decoder is used only during training, leaving the temporal re-encoder to supply history to the policy at inference. We train the module progressively on synthetic sequences and robot data, then jointly with the policy, using temporal masking to accommodate partial histories. This integration retains the original action head and action space and adds auxiliary losses to the original policy loss. Experiments with multiple policy backbones in simulation and on a real robot show improved task success on tasks with phase ambiguity, with little additional computation.

## I. INTRODUCTION

In robotic manipulation, the same or similar observation may occur at different execution phases that require different actions. For example, a button may look unchanged after a press [1], and a robot moving along the same path in opposite directions may need history to identify the current phase [2]. We use phase ambiguity to describe this situation: observations from different execution phases are locally similar, but the appropriate next action depends on the preceding execution history. A policy without sufficient history may therefore repeat a completed motion or switch phases at the wrong time. Recent work on cyclic manipulation further shows the need to track progress when an action must be repeated a specified number of times [3]. Reliable action selection in these settings therefore requires a representation of execution history that distinguishes phases with similar observations.

Recent imitation learning and embodied policy frameworks have substantially improved robotic action prediction, with representative examples including ACT [4], Diffusion Policy [5], and vision-language-action models such as Open-VLA [6]. Nevertheless, resolving phase ambiguity requires temporal representations that preserve relevant execution history and distinguish phases with similar observations. Existing memory mechanisms commonly provide this context through recurrent states [7], learned history tokens [2], or retrieval from a memory bank [1], as conceptually shown in Fig. 1(a). These mechanisms support decisions that depend on past interactions, but maintaining and accessing memory can require dedicated update rules, retrieval components, or policy-specific fusion layers. Such requirements increase integration effort across policy backbones, even when the relevant phase information is contained in a bounded recent history. The challenge is therefore to preserve action information and distinguish execution phases through a lightweight history representation that requires minimal changes to the policy. In practical robotic deployment, such a module should introduce limited computational overhead and remain scalable across different policy backbones rather than relying on additional memory banks or large architectural modifications.

![](images/203af9b1a5215160fa48dddd314e09249ba770dff1a6972060adf275b49a1143.jpg)  
Fig. 1: Conceptual comparison of temporal memory mechanisms for repetitive manipulation. (a) Representative memory-based approaches summarize history into recurrent states or memory tokens, or retrieve relevant information from an explicit memory bank. (b) PMTRM instead re-encodes a bounded queue of executed state– action history into a phase-discriminative latent sequence Z.

To learn such a representation, we introduce PMTRM, which combines a temporal re-encoder, representation learning objectives, and a progressive training procedure. The re-encoder maps a bounded history of executed states and actions to a latent sequence for the policy, as shown in Fig. 1(b). To discourage similar representations at distant time steps, we apply a temporal heterogeneity objective directly to this sequence. Anchor and reconstruction losses preserve information needed for action prediction, balancing temporal discrimination with reconstruction of the input history. These objectives use temporal distance rather than phase labels during training. We first pretrain PMTRM on synthetic sequences, adapt it to robot data, and apply mirror augmentation and temporal masking before joint policy training. Joint training adds the auxiliary objectives to the original policy loss. At inference, only the re-encoder is retained, providing recent history through the policy input adapter while preserving the original action head and action space. The main contributions of this work are summarized as follows:

• We propose PMTRM, a lightweight bounded-history temporal representation module with 7.61M parameters, whose policy-facing latent is explicitly regularized for distant-phase discriminability while remaining reconstructable.

• We combine temporal heterogeneity with anchor and reconstruction losses to encourage temporal discrimination while preserving action information, without phase labels.

• We evaluate PMTRM across action-policy backbones and phase-ambiguous tasks in simulation and on a real robot, and analyze the effects of the proposed objectives and mirror-mask training.

## II. RELATED WORK

## A. Temporal Memory and Pseudo-Memory

History-aware policies encode past interactions through recurrent states [7], working-memory tokens [2], or retrieved perceptual and cognitive features [1]. These mechanisms provide historical context, while cyclic manipulation also requires tracking execution progress [3]. PMTRM focuses on a bounded state-action history and directly regularizes its latent sequence to distinguish distant temporal positions.

## B. Policy-Side Temporal Conditioning

ACT predicts action chunks with a Transformer [4], while Diffusion Policy generates multimodal actions by denoising [5]. PMTRM neither replaces these policy mechanisms nor compresses history into a single summary: its two-stage design exposes the full bounded latent sequence to the unchanged action head and uses a separate decoder only during training. It is preferable when recent phase aliasing matters and a low-latency, backbone-agnostic adapter is needed; explicit memory remains preferable for semantic retrieval or context beyond the queue.

## C. Temporal Discriminability

Prior work learns temporally structured representations through temporal correspondence, sequence alignment, or future prediction [8], while robot-learning methods further exploit temporal organization for hierarchical actions and progress modeling [9], [10]. PMTRM instead directly regularizes the bounded history representation consumed by the policy, suppressing similarity between temporally distant positions while preserving action-relevant information through anchor and reconstruction objectives. This design motivates the temporal heterogeneity objective introduced next.

## III. METHOD

As illustrated in Fig. 2, PMTRM has three components: a temporal re-encoder and its policy interface, temporal heterogeneity and information preservation objectives, and a progressive training procedure. We first define the history representation, then describe the objectives that shape it, and finally explain training and integration with downstream policies.

## A. Pseudo-Memory Temporal Re-encoding Module

Given a state-action trajectory segment, we denote the input temporal sequence as

$$
\mathbf { X } = [ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } ] \in \mathbb { R } ^ { T \times D } ,\tag{1}
$$

where $T$ is the sequence length and D is the maximum feature dimension after padding. In practice, different episodes may have different valid lengths and feature dimensions. Therefore, we use a temporal mask $\mathbf { m } ^ { t } \in \{ 0 , 1 \} ^ { T }$ and a feature mask $\mathbf { m } ^ { f } \in \{ 0 , 1 \} ^ { \bar { D } }$ to indicate valid time steps and valid feature channels. The feature mask is episode-constant and records the available proprioceptive/action dimensions; only the temporal mask varies with the history prefix. For ACT and HACT-VQ, PMTRM receives proprioceptive state plus executed actions (no image tokens). For visual backbones, the original visual encoder produces one feature $\mathbf { v } _ { t }$ for each queued frame; these features form $\mathbf { V } \in \mathbb { R } ^ { T \times D _ { v } }$ and are temporally aligned with the same queue entries as X. The full mask and re-encoding process are

$$
\begin{array} { r l } & { \mathbf { M } _ { t , d } = \mathbf { m } _ { t } ^ { t } \mathbf { m } _ { d } ^ { f } , } \\ & { \quad \quad \mathbf { Z } = E _ { 1 } ( \mathbf { X } , \mathbf { M } ) , \qquad \hat { \mathbf { X } } = E _ { 2 } ( \mathbf { Z } , \mathbf { M } ) , } \\ & { \quad \quad \mathbf { C } = [ \mathbf { Z } \parallel P _ { v } ( \mathbf { V } ) ] . } \end{array}\tag{2}
$$

Here $P _ { v } : \mathbb { R } ^ { D _ { v } }  \mathbb { R } ^ { D }$ is a learned linear projection and ∥ denotes channel concatenation. DP and VLA policies consume the full, time-aligned $\mathbf { C } \in \mathbb { R } ^ { T \times 2 D }$ through a backbonespecific input adapter, not only ${ \mathbf { z } } _ { T } ;$ non-visual policies use $\mathbf { C } = \mathbf { Z }$ . Temporal masking is applied to PMTRM inputs and losses at every stage; the visual backbone receives its native frame-validity mask, and masked visual positions are zeroed after $P _ { v }$ before concatenation. $E _ { 1 }$ is policy-facing, while $E _ { 2 }$ maps Z back to the trajectory space only during training.

Each $E _ { 1 } / E _ { 2 }$ stack has $L = 4$ residual blocks with $k = 3 ,$ hidden width $H = 1 2 8$ , dilations {1, 2, 4, 8}, GroupNorm, and GELU; dropout is disabled. Feature masks are active throughout pretraining and joint training; random prefix masking is used in all stages, whereas mirror augmentation is used only in the third fine-tuning stage.

Unlike a bottleneck, $\textbf { Z } \in \mathbb { R } ^ { T \times D }$ preserves the input sequence shape; PMTRM is therefore a lightweight boundedhistory representation, not a coordinate-compression method. Relative to a standard autoencoder, it exposes Z to the policy and heterogeneity loss while using $E _ { 2 }$ only to preserve action-relevant information.

![](images/a34a3af05b59a6a7ff200fd1cece4af5a8e49ca02528369421a2d3b4cbf01d90.jpg)  
Fig. 2: Overall framework of PMTRM. (a) The module employs a two-stage convolutional temporal re-encoding autoencoder to map synthetic or real executed state-action histories into pseudo-memory representations.(b) Stages 1–4 correspond to synthetic pretraining, open-source real-data fine-tuning, mirror-mask fine-tuning (Eq. (7)), and joint policy training, respectively.(c) At deployment, E<sub>2</sub> is disabled and only $E _ { 1 }$ encodes the executed state/action history queue (up to T steps). The policy consumes this queue representation and predicts a separate future action chunk, which is not re-encoded until its actions are executed.

## B. Masked Reconstruction and Latent Anchor Loss

The re-encoding module is trained under two informationpreservation constraints. We call the first term the latent anchor loss. It defines a small residual trust region around the normalized input trajectory; it does not force an identity mapping, but prevents the heterogeneity objective from destroying action semantics:

$$
{ \mathcal { L } } _ { \mathrm { a n c h o r } } = \mathrm { M S E } _ { \mathbf { M } } ( \mathbf { X } , \mathbf { Z } ) + 0 . 1 \cdot { \mathcal { L } } _ { \mathrm { c o s } } ( \mathbf { X } , \mathbf { Z } ) ,\tag{3}
$$

where $\mathrm { M S E } _ { \mathbf { M } } ( \cdot )$ denotes masked mean squared error and ${ \mathcal { L } } _ { \mathrm { c o s } }$ is a masked cosine distance computed over valid positions. The weight of this term is intentionally small in the total objective. Thus, $\mathcal { L } _ { \mathrm { a n c h o r } }$ does not dominate representation learning; it prevents the re-encoder from producing arbitrary latent values that satisfy temporal heterogeneity but no longer correspond to executable state-action transitions.

The reconstruction term in Eq. (4) is

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \mathrm { M S E } _ { \mathbf { M } } ( \mathbf { X } , \hat { \mathbf { X } } ) + 0 . 1 \mathcal { L } _ { \mathrm { c o s } } ( \mathbf { X } , \hat { \mathbf { X } } ) ,\tag{4}
$$

which preserves information needed for executable action prediction.

## C. Temporal Heterogeneity Objective

To improve temporal discriminability in the re-encoded latent sequence, we introduce an intra-trajectory heterogeneity objective on the output of the first autoencoder. The motivation is that distant time steps within a trajectory should not collapse into overly similar latent representations, especially when the original action sequence contains repeated or locally similar motion patterns.

Given the latent sequence $\textbf { Z } \in \ \mathbb { R } ^ { T \times D }$ , we define the temporal heterogeneity objective as

$$
\begin{array} { r l r } & { \tilde { \mathbf { z } } _ { t } = \frac { \mathbf { z } _ { t } \odot \mathbf { m } ^ { f } } { \sqrt { \left\| \mathbf { z } _ { t } \odot \mathbf { m } ^ { f } \right\| _ { 2 } ^ { 2 } + \epsilon } } , } & \\ & { q _ { k , t } = \mathbf { m } _ { t } ^ { t } \cdot \mathbb { I } \left( \left| t - a _ { k } \right| > \delta \right) , } & \\ & { s _ { k , t } = \tilde { \mathbf { z } } _ { a _ { k } } ^ { \top } \tilde { \mathbf { z } } _ { t } , } & \\ & { \mathcal { L } _ { \mathrm { h e t e r o } } = \frac { \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } q _ { k , t } \left[ \operatorname* { m a x } \left( 0 , s _ { k , t } \right) \right] ^ { 2 } } { \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } q _ { k , t } + \epsilon } . } & \end{array}\tag{5}
$$

Here, $\mathbf { m } ^ { f }$ is the feature mask, $\mathbf { m } ^ { t }$ is the temporal mask, ϵ is a small constant for numerical stability, and $\mathcal { A } = \{ a _ { 1 } , . . . , a _ { K } \}$ denotes randomly sampled valid anchor indices. The indicator ${ q } _ { k , t }$ excludes invalid temporal positions and local pairs within the margin δ.

This objective penalizes positive similarity only for distant pairs; local smoothness is unconstrained while anchor and reconstruction protect action semantics. We use $K \ : = \ : 3 2$ and $\delta \ = \ 8 ;$ ; anchors are resampled per batch. The $K \in$ {16, 32, 64} and $\delta ~ \in ~ \{ 4 , 8 , 1 2 \}$ sweep was selected on synthetic pretraining and then checked on ACT Button Pressand-Release and visual DP Stack Block. It preserves the ordering and changes final success by at most 1.4 points, so the same setting is fixed for all backbones. For short or heavily masked windows, anchors are sampled only from valid positions and the heterogeneity term is skipped when fewer than two distant valid pairs remain. Across training, 93.4% of synthetic windows, 89.7% of real-data windows, and 86.1% of mirror crops contain at least eight active anchor–time pairs; the remaining batches contribute only anchor and reconstruction gradients. This prevents unstable large gradients when the available history is too short.

## D. Overall Training Objective

The final training loss combines temporal heterogeneity, latent anchoring, and reconstruction:

$$
\mathcal { L } = \alpha _ { \mathrm { h } } \mathcal { L } _ { \mathrm { h e t e r o } } + \beta \mathcal { L } _ { \mathrm { a n c h o r } } + \gamma \mathcal { L } _ { \mathrm { r e c o n } } .\tag{6}
$$

In our implementation, $\alpha _ { \mathrm { h } } = 1 . 5 , \beta = 0 . 0 5$ , and $\gamma = 8 . 0 $ After normalization, typical batch values are $\mathcal { L } _ { \mathrm { h e t e r o } } = 0 . 1 0 -$ $0 . 1 6 , \mathcal { L } _ { \mathrm { a n c h o r } } = 0 . 0 3  – 0 . 0 6$ , and $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c o n } } = 0 . 0 0 6 \ – 0 . 0 1 2 ; } \end{array}$ thus γ maintains reconstruction while β bounds the residual.

## E. Progressive Pretraining and Fine-tuning

The four stages in Fig. 2 are synthetic pretraining, realdata fine-tuning, mirror-mask fine-tuning, and joint policy training. The first three train PMTRM; the fourth attaches it to a backbone. Synthetic windows mix periodic and nonperiodic signals with random lengths and active dimensions, padded and masked to the model shape.

Second, we fine-tune on LeRobot-format raw-action trajectories. Channels are ordered as pose, gripper, then action, normalized per dataset, and padded/masked to the maximum dimension; this canonical ordering avoids high-tolow-dimensional transfer degradation. We sample shuffled episode windows and cap large datasets per epoch so no corpus dominates gradients.

Synthetic pretraining uses 50,000 windows for 100 epochs (batch size 64, AdamW, 10<sup>−3</sup>). Length is uniform on [32, 128] and active dimensions on $[ 4 , D ]$ : sinusoidal channels have amplitude [0.2, 1.0], 1–4 cycles/window, random phase, and Gaussian noise $\sigma ~ \in ~ [ 0 , 0 . 0 2 ]$ ; piecewisesmooth channels linearly interpolate four random knots with the same amplitude/noise ranges. Windows are channelnormalized. Real-data fine-tuning uses the same optimizer and batch size for 30 epochs at $3 \times 1 0 ^ { - 4 }$ , with at most 2,048 windows per dataset per epoch. Skipping synthetic pretraining lowers cyclic Button Press-and-Release CSR by 3.2 points and the cross-embodiment mean completion by 2.0 points.

Third, we use mirror-and-mask training as representation augmentation, not as a physically executable reverse rollout. Given an episode sequence X, we construct the virtual stream

$$
\begin{array} { r } { { \bf X } ^ { \mathrm { m i r } } = [ { \bf X } , \mathrm { r e v } ( { \bf X } ) , \mathrm { r e v } ( { \bf X } ) , { \bf X } ] . } \end{array}\tag{7}
$$

and sample a contiguous window of at most $T = 1 2 8$ entries (then left-pad and mask it); the fourfold stream is never passed as a whole. For a three-step sequence $[ A , B , C ]$ , the virtual order is $[ A , B , C , C , B , A , C , B , A , A , B , C ] .$ , so sampled crops can cross reversal boundaries. Reversal is applied to normalized vectors without sign-flipping channels because a universal physical inverse is undefined. The stream is never executed or a policy target; it only regularizes changed temporal order. Prefix length is uniform on $[ 1 , \textstyle \sum _ { t } m _ { t } ^ { t } ]$ . Fourth, during downstream policy training, PMTRM is optimized jointly with the original policy objective using the same auxiliary loss. This stage adapts the latent representation to the action distribution and observation encoder of the selected policy backbone. An essentiality check on the same validation split gives average memory-task success of 67.7% without Stage 1, 69.1% without Stage 2, and 68.4% when Stage 3 mirror augmentation is removed; the full fourstage schedule reaches 70.9%. Thus synthetic diversity and mirror robustness are complementary, while Stage 2 mainly improves cross-embodiment calibration. A deployment requiring minimal training can omit Stage 3 at a cost of 2.5 points, but keeps the same inference graph.

TABLE I: Open-source robot manipulation datasets used in the fine-tuning setting. The corpus includes single-arm and bimanual data (UMI and BEHAVIOR-1K), deformable-object interactions (towel/dish washing and soft-fold), and multi-stage trajectories (LIBERO, CALVIN, BEHAVIOR-1K, and ${ \mathrm { X } } { \mathrm { - V L A } } ) ;$ these tags describe pretraining diversity rather than a separate benchmark split.
<table><tr><td>Dataset family</td><td>State/Action</td><td>Episodes</td><td>Frames</td></tr><tr><td>DROID/Bridge/RT-1 [11], [12], [13]</td><td>7-8/7</td><td>137,700</td><td>5.6M</td></tr><tr><td>LIBERO/CALVIN [14], [15]</td><td>8-15/7</td><td>34,800</td><td>2.6M</td></tr><tr><td>BEHAVIOR-1K [16]</td><td>256/23</td><td>10,000</td><td>119.1M</td></tr><tr><td>TACO/Franka/TOTO [17], [18], [19]</td><td>7-13/7-15</td><td>4,200</td><td>0.6M</td></tr><tr><td>Austin/UT/Berkeley/Stanford [20], [21], [22], [23], [24]</td><td>7-15/7-8</td><td>3,800</td><td>1.8M</td></tr><tr><td>X-VLA soft-fold [25]</td><td>96/14</td><td>1,542</td><td>2.9M</td></tr><tr><td>UMI/RoboCOIN (bimanual/deformable) [26], [27]</td><td>7-26/7-26</td><td>1,755</td><td>1.1M</td></tr></table>

## F. Integration with Downstream Embodied Policies

PMTRM is a plug-in representation component for ACT, Diffusion Policy, and VLA policies: preceding executed states and actions are padded, encoded, and supplied to the original policy. It leaves continuous actions, discrete tokens, action chunks, decoders, and normalization unchanged.

When PMTRM is integrated with a downstream policy backbone, its auxiliary objective is directly added to the original policy training loss. The joint optimization objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { j o i n t } } = \mathcal { L } _ { \mathrm { p o l i c y } } + \mathcal { L } _ { \mathrm { P M T R M } } , } \end{array}\tag{8}
$$

where ${ \mathcal { L } } _ { \mathrm { p o l i c y } }$ denotes the original loss function of the corresponding policy model All policy-specific objectives, schedules, and architectural settings follow each backbone’s default configuration; PMTRM is jointly optimized without task-specific loss redesign.

## G. Computational Cost and Scaling

PMTRM has linear cost O(TLkDH) and memory $O ( T ( D + H ) )$ , where L, k, and H are the number of residual blocks, kernel width, and hidden width. The training module $( E _ { 1 } + E _ { 2 } )$ has 7.61M parameters; $E _ { 2 }$ is discarded, leaving the 3.81M-parameter $E _ { 1 }$ at inference. At $T = 1 2 8$ , this adds

TABLE II: Ablation of PMTRM design choices on the memoryintensive evaluation (ACT MuJoCo Button Press-and-Release, LIBERO Drawer Open-and-Close, Pour Bottle Twice, and Shell Game). Avg. success follows the metric in Table V; all variants use the same demonstrations, seeds, and $T = 1 2 8$
<table><tr><td>Setting</td><td>Avg. Success ↑</td><td>Recon. MSE ↓</td><td>Distant Cos. ↓</td></tr><tr><td>Single-stage AE + heterogeneity (same budget)</td><td>63.4</td><td>2.4</td><td>0.31</td></tr><tr><td>GRU + reconstruction (same policy interface)</td><td>64.6</td><td>3.1</td><td>0.29</td></tr><tr><td>Contrastive temporal loss (same budget)</td><td>67.2</td><td>3.0</td><td>0.25</td></tr><tr><td>PMTRM w/o synthetic pretraining</td><td>67.7</td><td>2.8</td><td>0.23</td></tr><tr><td>PMTRM, ACT MuJoCo Markov reach</td><td>91.6</td><td>1.9</td><td>0.30</td></tr><tr><td>PMTRM, β = 0</td><td>65.1</td><td>7.8</td><td>0.18</td></tr><tr><td>PMTRM, β = 0.01</td><td>68.5</td><td>4.9</td><td>0.20</td></tr><tr><td>PMTRM, β = 0.10</td><td>69.2</td><td>2.1</td><td>0.27</td></tr><tr><td>PMTRM, random mask only (no mirror)</td><td>66.7</td><td>2.9</td><td>0.23</td></tr><tr><td>Full PMTRM, β = 0.05</td><td>70.9</td><td>2.5</td><td>0.21</td></tr></table>

0.42 ms per ACT decision on an RTX 4090 (6.8 ms for the base policy, 6.2% overhead) and 2.7 ms on an i7-13700K CPU. GPU latency for $T = 3 2 , 6 4 , 1 2 8 , 2 5 6$ is 0.16, 0.27, 0.42, 0.83 ms. We fix $T = 1 2 8$ in Experiments 2–3; shorter queues occur only during prefix masking.

At inference, an episode-level queue of preceding executed states and actions is encoded at each decision step; future action chunks are never re-encoded. The queue is updated once per executed control step; short histories are left-padded and masked. During joint training, gradients pass through the queued history but not unavailable future actions.

## IV. EXPERIMENTS

We ask whether PMTRM preserves reconstructability, transfers across policy backbones, and improves repetitive manipulation.

## A. Experiment 1: Loss Dynamics and Objective Trade-offs

We examine PMTRM loss dynamics across the three pretraining stages (batch size 64) and subsequent joint policy training. The weighted objective uses $\alpha _ { \mathrm { h } } = 1 . 5 , \beta = 0 . 0 5$ and $\gamma = 8 . 0$

Figure 3 visualizes the learned representation on a phasealiasing cyclic sequence. Unlike the standard autoencoder, PMTRM preserves local continuity while separating distant phases; this qualitative effect is quantified below with a probe on frozen representations.

Here Distant Cos. is the mean $s _ { k , t }$ over valid pairs with $| t - a _ { k } | > \delta$ . The probe on frozen representations is a twolayer MLP (hidden width 64, GELU) trained for 20 epochs with AdamW on 80% of the training windows and evaluated on the remaining 20%; labels are press/release or open/close intervals from simulator phase timestamps, and completedpour count from the annotated action-event index. It gives 98.1% phase accuracy from Z, versus 71.4% from X and 76.8% from single-stage AE. On real Drawer Open-and-Close, Z predicts open/close at 94.6% (raw X: 79.2%); on Pour Bottle Twice it predicts pour count at 86.0% (raw X: 61.5%).

Table II shows that matched-budget AE, GRU, and contrastive controls underperform PMTRM; all controls share corpus, masks, demonstrations, seeds, and $T = 1 2 8$ . Shortattention/diagonal-SSM/Mamba/action-memory controls obtain 68.8/69.4/69.7/69.2% at 0.61/0.48/0.46/0.58 ms, versus

70.9% at 0.42 ms for PMTRM. Across Button CSR / Drawer CSR / Pour Finish / Shell Finish, $\beta = 0 . 0 1 , 0 . 0 5 , 0 . 1 0$ gives 61/73/70/60, 65/79/80/70, and 64/77/80/60%, respectively, making 0.05 a robust compromise. A symmetric negativecosine penalty reduced local-continuity probe accuracy by 2.3 points and increased MSE by 0.6; negative similarities already separate phases, so the one-sided hinge is retained.

On Markov reaching, ACT/PMTRM obtain 91.8 ± $1 . 0 / 9 1 . 6 \pm 1 . 1 \% ;$ periodic tapping gives 88.7/88.1% (88.5% with $\alpha _ { h } = 0 . 7 5 )$ , and $\delta = 4 , 8 , 1 2$ varies by under 0.6 points. Mirror/mask/mirror+mask yield 89.8/93.7/98.1% probe accuracy and 0.24/0.23/0.21 Distant Cos. Dropping proprioception, action, or both lowers success by 1.1/1.8/3.0 points. With ten real rollouts, a 70% estimate has Wilson 95% CI [39.7, 89.2]%; real gains are directional, while multi-seed simulation is higher-confidence. Distant Cos. is diagnostic, not a value to minimize blindly: around 0.2–0.25 separates aliased phases while preserving continuity; Markov tasks can remain strong near 0.30.

## B. Experiment 2: Compatibility with Conventional Action Policies

To test generality, we integrate PMTRM with representative policy backbones on the simulated and real-world tasks in Fig. 4. Touch/Lift/Transfer require end-effector contact, object height $> ~ 2$ cm, and placement within 5 cm; Grasp/Contact/Insert require stable grasp, contact within 3 cm, and insertion depth $> 9 0 \%$ . Finish requires all subtasks within the episode horizon. MuJoCo runs 400 control steps at 20 Hz; real tasks run 300 steps at 10 Hz with reset after a missed grasp or workspace violation. All methods use identical demonstrations and splits, isolating the policy and PMTRM components; action decoders are unchanged. Simulation uses ten fixed seeds on an RTX 4090; real trials use the same ten randomized poses for every method. Stage 1–3 provide one initialization, then Stage 4 finetunes an independent PMTRM copy per backbone. To isolate modality interactions, removing visual features while retaining PMTRM changes ACT-PMTRM success by −0.8 points on simulation, whereas removing proprioceptive/action history while retaining the visual encoder changes it by −5.4 points. The module therefore supplies complementary phase information rather than replacing visual grounding.

When PMTRM is integrated with a downstream policy backbone, its auxiliary objective is directly added to the original policy training loss with weights set to $\alpha _ { \mathrm { h } } = 1 . 5 ,$ $\beta ~ = ~ 0 . 0 5$ , and $\gamma = 8 . 0$ , respectively. All policy-specific objectives, schedules, and architectural hyperparameters follow each backbone’s default configuration. We retain each backbone’s published epoch and batch schedule, select the checkpoint with lowest validation policy loss, and use no test-set early stopping. Stage 1–3 produce one common initialization; in Stage 4 we clone PMTRM and jointly finetune one copy per backbone (no weights or batch statistics are shared across policies). For visual policies, $D _ { v } = 2 5 6$ for DP and $D _ { v } = 5 1 2$ for VLA; P is learned per backbone and reused across all datasets seen by that policy.

(a)  
![](images/f9ceac8338ab5f03c35570e100e5ab8652c85d2acaa4728386e6995a7fb2c23a.jpg)

(b)  
![](images/81beae410cd2e053ae447202495ec4d95a5be4c4b6642d013869fa73f088ea2c.jpg)

(c)  
![](images/b760376c5667d04721ce3f3e9453e8050ab8791c1899fa21c4564cf08abeda50.jpg)

(d)  
![](images/29480405a6f968ce5a95d870b064cbac378bb818a1d07246b2fdbd8e9d35afa9.jpg)

Fig. 3: Representation analysis on a phase-aliasing cyclic trajectory. Top: temporal cosine-similarity matrices for a standard autoencoder and PMTRM (the same [0, 1] color scale is used; dotted boxes mark cycle-1/cycle-2 aliased-phase blocks measured by Distant Cos. in Table II, and dashed lines mark the cycle boundary). Bottom: PCA projections of two cycles; stars mark their starts. PMTRM suppresses off-diagonal similarities while retaining local continuity.  
![](images/4dbc689f89bbb675a1a2f1e12c6c801600388bf14228f07f4e2b7c3b2a9e03d2.jpg)  
Fig. 4: Left: The MuJoCo tasks Transfer Cube, Bimanual Insertion, Transfer Plus, and Stack Two Blocks. Right: The real-world tasks Stack Block, Store Items, Potato Placement, and Bowl Stacking.

![](images/ec5869160f4c51b74e673d727cffa36303e4b3322800643d2cffbb02af658877.jpg)  
Fig. 5: Left: The simulation tasks Button Press-and-Release (Mu-JoCo) and Drawer Open-and-Close (LIBERO). Right: The realworld tasks Pour Bottle Twice and Shell Game. These tasks require the policy to distinguish repetitive motions and execution phases using historical context rather than only the current observation; the key phase transitions are press versus release, open versus close, first versus second pour, and target tracking after swaps.

Tables III and IV report the simulation and real-world results. Gains are largest in later-stage Transfer, Insert, and Finish metrics. In Stack Blocks, ACT-PMTRM reduces neartarget velocity sign changes from 4.7 to 2.9 per rollout and raises successful grasps from 69% to 78%, consistent with fewer approach oscillations. Real-robot experiments use an Airbot Play, a fixed RGB camera, and 7-D Cartesian deltapose plus gripper actions.

## C. Experiment 3: Memory and Repetitive Manipulation

Finally, we evaluate phase-dependent tasks with repeated local observations, where history is required rather than the instantaneous view alone. For each task, training uses the stated demonstrations, validation selects checkpoints and hyperparameters, and test rollouts use held-out scene seeds; no test trajectory is used for tuning.

TSR (task success rate) measures completion of the current task, whereas CSR (cumulative success rate) also requires preceding subtasks to succeed [33]. Simulation uses 50 MuJoCo and 100 LIBERO demonstrations; physical tasks use 20 demonstrations and 10 randomized closed-loop rollouts. Demonstrations are split by episode (80/10/10% train/validation/test); LIBERO drawer variants are split by scene seed, so no trajectory appears in more than one split. Every baseline uses the same demonstrations, seeds, and evaluation rollouts. Published baseline hyperparameters are reused when available; newly attached PMTRM and lightweight controls are tuned on validation only.

As shown in Fig. 5, we construct four representative memory-intensive tasks, including two simulation tasks and two real-world tasks. The two simulation tasks test canonical state aliasing caused by repeated forward-backward motions, while the two real-world tasks introduce additional phaseprogress and spatial-coverage dependencies under physical interaction. These settings require the policy to infer not only the current visual state, but also which stage of the manipulation has already been completed.

Button Press-and-Release and LIBERO Drawer Openand-Close contain visually similar observations that correspond to opposite actions at different phases of execution. The real-world Open and Close Drawer task further evaluates phase disambiguation under physical interaction: the policy must remember whether the drawer has already been opened in order to determine whether the next action should continue opening or switch to closing. In Wipe Table, the robot must retain information about previously wiped regions and recent wiping progress so that subsequent motions cover the remaining area rather than repeatedly acting on the same local region. Together, these tasks evaluate phase disambiguation, progress tracking, and history-dependent execution under repeated observations.

We compare reactive and memory-augmented models, and add a lightweight PrediMem+PMTRM adapter. PMTRM receives executed state/action history while PrediMem retains predictive tokens; CSR improves by 2.4/1.6 points on simulation and final completion by 10 points on each physical task, indicating complementarity. For PMTRM models, the auxiliary objective is jointly optimized with the original policy loss using the weights in Sec. III; all backbone settings follow their defaults.

TABLE III: Simulation evaluation of PMTRM with different action policy backbones. Success rate (%) ↑ is reported as mean ± standard deviation over 10 random seeds; the rightmost subcolumn of each task is end-to-end completion, and all PMTRM variants use $T = 1 2 8 .$
<table><tr><td></td><td colspan="3">Transfer Cube (Sim)</td><td colspan="3">Bimanual Insertion (Sim)</td><td colspan="3">Transfer Plus (Sim)</td><td colspan="3">Stack Two Blocks (Sim)</td></tr><tr><td></td><td>Touch</td><td>Lift</td><td>Transfer</td><td>Grasp</td><td>Contact</td><td>Insert</td><td>Lift</td><td>Stack</td><td>Finish</td><td>Stack</td><td>Lift</td><td>Finish</td></tr><tr><td>DP (DDPM, CNN) [5]</td><td> $9 5 . 1 \pm 1 . 2$ </td><td> $9 2 . 4 \pm 1 . 3$ </td><td> $9 0 . 2 \pm 1 . 9$ </td><td> $7 7 . 4 \pm 2 . 7$ </td><td> $6 9 . 0 \pm 3 . 1 $ </td><td> $6 3 . 2 \pm 3 . 3$ </td><td> $6 2 . 1 \pm 3 . 5$ </td><td> $5 2 . 9 \pm 3 . 9$ </td><td> $5 2 . 9 \pm 3 . 9$ </td><td> $8 2 . 0 \pm 2 . 7$ </td><td> $6 3 . 5 \pm 3 . 2$ </td><td> $4 6 . 7 \pm 3 . 6$ </td></tr><tr><td>DP-PMTRM</td><td> $9 7 . 0 \pm 1 . 0$ </td><td> $9 4 . 1 \pm 1 . 1$ </td><td> $9 2 . 8 \pm 1 . 5$ </td><td> $8 1 . 0 \pm 2 . 3$ </td><td> $7 3 . 6 \pm 2 . 7$ </td><td> $6 7 . 8 \pm 3 . 0$ </td><td> $6 6 . 8 \pm 3 . 2$ </td><td> $5 8 . 4 \pm 3 . 5$ </td><td> $5 8 . 4 \pm 3 . 5$ </td><td> $8 5 . 1 \pm 2 . 4$ </td><td> $6 8 . 0 \pm 3 . 0$ </td><td> $5 3 . 2 \pm 3 . 4$ </td></tr><tr><td>ACT [4]</td><td> $9 8 . 4 \pm 0 . 9$ </td><td> $9 6 . 2 \pm 1 . 0$ </td><td> $9 4 . 8 \pm 2 . 1$ </td><td> $8 1 . 3 \pm 2 . 6$ </td><td> $7 3 . 5 \pm 3 . 0$ </td><td> $6 8 . 1 \pm 3 . 2$ </td><td> $6 6 . 7 \pm 3 . 6$ </td><td> $5 7 . 5 \pm 4 . 0$ </td><td> $5 7 . 5 \pm 4 . 0$ </td><td> $8 5 . 8 \pm 2 . 8$ </td><td> $6 7 . 3 \pm 3 . 4$ </td><td> $5 0 . 4 \pm 3 . 8$ </td></tr><tr><td>ACT-PMTRM</td><td> $9 9 . 1 \pm 0 . 5$ </td><td> $9 7 . 5 \pm 0 . 8$ </td><td> $9 6 . 4 \pm 1 . 5$ </td><td> $8 5 . 7 \pm 2 . 2$ </td><td> $7 9 . 4 \pm 2 . 6$ </td><td>73.8±2.9</td><td> $7 2 . 6 \pm 3 . 1$ </td><td> $6 3 . 8 \pm 3 . 5$ </td><td> $6 3 . 8 \pm 3 . 5$ </td><td> $8 9 . 3 \pm 2 . 3$ </td><td> $7 2 . 8 \pm 3 . 0$ </td><td> $5 8 . 7 \pm 3 . 4$ </td></tr><tr><td>HACT-VQ [28]</td><td> $9 8 . 5 \pm 0 . 9$ </td><td> $9 7 . 6 \pm 1 . 2$ </td><td> $9 6 . 2 \pm 1 . 4$ </td><td> $8 7 . 4 \pm 2 . 4$ </td><td> $8 2 . 2 \pm 2 . 7$ </td><td> $7 6 . 3 \pm 2 . 8$ </td><td> $7 9 . 2 \pm 2 . 2$ </td><td> $6 8 . 5 \pm 3 . 1 $ </td><td> $6 8 . 5 \pm 3 . 1$ </td><td> $9 0 . 6 \pm 2 . 3$ </td><td> $7 6 . 1 \pm 2 . 8$ </td><td> $5 5 . 8 \pm 3 . 2$ </td></tr><tr><td>HACT-VQ-PMTRM</td><td> $9 9 . 2 \pm 0 . 5$ </td><td> $9 8 . 1 \pm 0 . 8 $ </td><td> $9 7 . 0 \pm 1 . 0$ </td><td> $8 9 . 0 \pm 1 . 9$ </td><td> $8 5 . 9 \pm 2 . 3$ </td><td> $8 0 . 1 \pm 2 . 5$ </td><td> $8 1 . 0 \pm 2 . 0$ </td><td> $7 1 . 0 \pm 2 . 7$ </td><td> $7 1 . 0 \pm 2 . 7$ </td><td> $9 1 . 7 \pm 2 . 0$ </td><td> $7 8 . 3 \pm 2 . 5$ </td><td> $6 2 . 1 \pm 2 . 9$ </td></tr><tr><td>InterACT [29]</td><td> $9 8 . 2 \pm 0 . 8 $ </td><td> $8 8 . 4 \pm 1 . 1$ </td><td> $8 2 . 1 \pm 2 . 0$ </td><td> $8 8 . 5 \pm 2 . 2$ </td><td> $7 8 . 3 \pm 2 . 8$ </td><td> $4 4 . 2 \pm 3 . 2$ </td><td> $7 8 . 5 \pm 2 . 4$ </td><td> $6 8 . 7 \pm 3 . 4$ </td><td> $6 8 . 7 \pm 3 . 4$ </td><td> $9 1 . 9 \pm 2 . 0$ </td><td> $7 7 . 1 \pm 2 . 6$ </td><td> $5 6 . 9 \pm 2 . 9$ </td></tr><tr><td>InterACT-PMTRM</td><td> $9 8 . 9 \pm 0 . 6 $ </td><td> $9 1 . 8 \pm 1 . 0$ </td><td> $8 6 . 4 \pm 1 . 8$ </td><td> $9 0 . 4 \pm 1 . 9$ </td><td> $8 1 . 5 \pm 2 . 4$ </td><td> $4 9 . 6 \pm 3 . 0$ </td><td> $8 2 . 1 \pm 2 . 1$ </td><td> $7 2 . 2 \pm 3 . 0$ </td><td> $7 2 . 2 \pm 3 . 0$ </td><td> $9 3 . 0 \pm 1 . 8$ </td><td> $8 0 . 1 \pm 2 . 3$ </td><td> $6 1 . 7 \pm 2 . 7$ </td></tr><tr><td>GR00T N 1.7 [30]</td><td> $9 4 . 5 \pm 1 . 5$ </td><td> $9 1 . 2 \pm 2 . 0$ </td><td> $9 1 . 6 \pm 2 . 2$ </td><td> $8 3 . 2 \pm 2 . 5$ </td><td> $7 3 . 9 \pm 3 . 0$ </td><td> $6 9 . 1 \pm 3 . 3$ </td><td> $6 3 . 4 \pm 3 . 1$ </td><td> $5 7 . 4 \pm 3 . 9$ </td><td> $5 7 . 4 \pm 3 . 9$ </td><td> $8 1 . 3 \pm 2 . 9$ </td><td> $6 5 . 4 \pm 3 . 2$ </td><td> $4 7 . 5 \pm 3 . 1$ </td></tr><tr><td>GR00T N 1.7-PMTRM</td><td> $9 6 . 2 \pm 1 . 1$ </td><td> $9 3 . 4 \pm 1 . 6$ </td><td> $9 3 . 5 \pm 1 . 8$ </td><td> $8 6 . 0 \pm 2 . 2$ </td><td> $7 8 . 1 \pm 2 . 7$ </td><td> $7 3 . 6 \pm 3 . 0$ </td><td> $6 8 . 9 \pm 2 . 8$ </td><td> $6 2 . 6 \pm 3 . 4$ </td><td> $6 2 . 6 \pm 3 . 4$ </td><td> $8 5 . 6 \pm 2 . 5$ </td><td> $7 0 . 2 \pm 2 . 9$ </td><td> $5 4 . 0 \pm 2 . 8$ </td></tr></table>

TABLE IV: Real-world evaluation of PMTRM with different action policy backbones. Success rates are reported in [0.0, 1.0] with $T = 1 2 8$ for PMTRM; the rightmost subcolumn of each task is final completion. Final-completion entries use ten Bernoulli trials, while intermediate stage rates pool step outcomes; for the former, a 0.70 estimate has Wilson 95% CI [0.397, 0.892] rather than a standarddeviation interpretation. The real-world benchmark includes four physical manipulation tasks: Stack Block, Store Items, Potato Placement, and Bowl Stacking.
<table><tr><td></td><td colspan="3">Stack Block</td><td colspan="3">Store Items</td><td colspan="3">Potato Placement</td><td colspan="3">Bowl Stacking</td></tr><tr><td></td><td>Grasp</td><td>Lift</td><td>Stack</td><td>First</td><td>Second</td><td>Finish</td><td>Grasp</td><td>Lift</td><td>Place</td><td>First</td><td>Second</td><td>Finish</td></tr><tr><td>DP (DDPM, CNN) [5]</td><td>0.82</td><td>0.70</td><td>0.62</td><td>0.79</td><td>0.61</td><td>0.54</td><td>0.91</td><td>0.83</td><td>0.74</td><td>0.77</td><td>0.59</td><td>0.50</td></tr><tr><td>DP-PMTRM</td><td>0.88</td><td>0.78</td><td>0.71</td><td>0.86</td><td>0.86</td><td>0.64</td><td>0.95</td><td>0.89</td><td>0.82</td><td>0.84</td><td>0.76</td><td>0.61</td></tr><tr><td>ACT [4]</td><td>0.85</td><td>0.74</td><td>0.69</td><td>0.86</td><td>0.65</td><td>0.58</td><td>0.93</td><td>0.86</td><td>0.80</td><td>0.83</td><td>0.62</td><td>0.55</td></tr><tr><td>ACT-PMTRM</td><td>0.91</td><td>0.84</td><td>0.78</td><td>0.91</td><td>0.78</td><td>0.70</td><td>0.97</td><td>0.92</td><td>0.87</td><td>0.89</td><td>0.75</td><td>0.67</td></tr><tr><td>GR00T N 1.7 [30]</td><td>0.86</td><td>0.76</td><td>0.70</td><td>0.85</td><td>0.68</td><td>0.60</td><td>0.94</td><td>0.88</td><td>0.81</td><td>0.83</td><td>0.65</td><td>0.57</td></tr><tr><td>GR00T N 1.7-PMTRM</td><td>0.92</td><td>0.85</td><td>0.80</td><td>0.91</td><td>0.79</td><td>0.72</td><td>0.98</td><td>0.94</td><td>0.89</td><td>0.89</td><td>0.76</td><td>0.69</td></tr></table>

TABLE V: Evaluation on memory-intensive and repetitive manipulation tasks. For simulation tasks, we report Task Success Rate (TSR, %) and Cumulative Success Rate (CSR, %). For real-world tasks, we report success rate (%) over 10 randomized closed-loop rollouts; First Open and First Wipe denote successful completion of the first repeated manipulation, while Finish requires completion of the full two-stage sequence. For reference, a 70% success rate over 10 Bernoulli trials has a Wilson 95% confidence interval of approximately [39.7, 89.2]%. PMTRM uses $T = 1 2 8 .$

<table><tr><td></td><td colspan="2">Button Press-and-Release (Sim)</td><td colspan="2">Drawer Open-and-Close (Sim)</td><td colspan="2">Open and Close Drawer (Real)</td><td colspan="2">Wipe Table (Real)</td></tr><tr><td></td><td>TSR</td><td>CSR</td><td>TSR</td><td>CSR</td><td>First Open</td><td>Finish</td><td>First Wipe</td><td>Finish</td></tr><tr><td>HiF-VLA [31]</td><td>65.8</td><td>59.2</td><td>68.6</td><td>60.4</td><td>50.0</td><td>30.0</td><td>30.0</td><td>20.0</td></tr><tr><td>MemoryVLA [1]</td><td>64.1</td><td>57.5</td><td>72.8</td><td>64.6</td><td>50.0</td><td>30.0</td><td>40.0</td><td>20.0</td></tr><tr><td>MemER [32]</td><td>70.6</td><td>64.4</td><td>79.3</td><td>71.5</td><td>60.0</td><td>40.0</td><td>60.0</td><td>40.0</td></tr><tr><td>PrediMem [33]</td><td>76.4</td><td>70.6</td><td>84.8</td><td>78.2</td><td>70.0</td><td>60.0</td><td>70.0</td><td>50.0</td></tr><tr><td>PrediMem-PMTRM</td><td>79.2</td><td>73.8</td><td>87.4</td><td>81.6</td><td>80.0</td><td>70.0</td><td>80.0</td><td>60.0</td></tr><tr><td>HiF-VLA-PMTRM</td><td>69.1</td><td>62.8</td><td>73.9</td><td>66.7</td><td>60.0</td><td>50.0</td><td>70.0</td><td>50.0</td></tr><tr><td>InterACT-PMTRM</td><td>85.5</td><td>79.7</td><td>91.8</td><td>85.4</td><td>90.0</td><td>80.0</td><td>80.0</td><td>70.0</td></tr></table>

Table V reports the four memory-intensive tasks. HiF-VLA uses motion tokens, MemoryVLA token memory, MemER keyframe retrieval, and PrediMem predictive memory. MemoryVLA+PMTRM is omitted because its released interface has no aligned executed-action stream and would require tokenizer retraining. InterACT-PMTRM combines local action chunks with phase-indexed history without a keyframe store.

beyond the relevant interaction horizon provides diminishing returns. Thus $T = 1 2 8$ provides a favorable latency–accuracy trade-off. A practical rule is to choose T to cover one to two task-phase cycles at the control frequency; increasing it is useful only while CSR or final task success improves faster than latency.

The gains are clearest on repetitive forward-backward and cyclic tasks, where similar local observations can correspond to different actions depending on the preceding trajectory. On Button Press-and-Release, $T = { 3 2 } / { 6 4 } / { 1 2 8 } / { 2 5 6 }$ yields 61/70/74/74% CSR, with 0.16/0.27/0.42/0.83 ms GPU overhead, respectively. A similar saturation trend is observed on the physical tasks: increasing the history window initially improves phase and progress estimation, whereas extending it

The bounded-history limitation becomes more apparent as the required interaction sequence grows. For Open and Close Drawer, errors arise when long or interrupted manipulation sequences make the relevant opening or closing transition fall outside the retained history. For Wipe Table, performance degrades when successful completion requires tracking wiping progress over an increasingly long trajectory or maintaining fine-grained coverage information beyond the history window. These failures delimit PMTRM to boundedhistory phase and progress disambiguation rather than persistent long-horizon semantic or spatial memory. In these regimes, increasing T beyond 128 gives little benefit once the relevant event lies outside the queue; an external retrieval mechanism, spatial memory, or recurrent full-history module is the appropriate complement.

## V. CONCLUSION

We proposed PMTRM, a lightweight bounded-history representation module for robotic manipulation. It improves phase disambiguation without retrieval or extra policy tokens, while explicit memories remain preferable for unbounded semantic recall. The module is trained and evaluated here with imitation objectives; in RL, distribution shift and rewarddriven phase changes may require annealing $\alpha _ { h }$ or masking heterogeneity when reward-equivalent cycles are detected. Online adaptation is otherwise compatible because the auxiliary losses use executed history only.

## REFERENCES

[1] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang, “MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation,” arXiv preprint arXiv:2508.19236, 2025.

[2] Q. Hu, Z. Qiu, Z. Xu, K. Zhang, X. Bu, Z. Sun, B. Zhang, J. Zhao, Z. Gan, and W. Ding, “Resolving state ambiguity in robot manipulation via adaptive working memory recoding,” arXiv preprint arXiv:2512.24638, 2025.

[3] Y.-L. Wei, H. Liao, Y. Lin, P. Wang, Z. Liang, G. Liu, and W.-S. Zheng, “CycleManip: Enabling cyclic task manipulation via effective historical perception and understanding,” arXiv preprint arXiv:2512.01022, 2026, version 2.

[4] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” arXiv preprint arXiv:2304.13705, 2023.

[5] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. C. Burchfiel, and S. Song, “Diffusion Policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems, 2023.

[6] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “OpenVLA: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[7] Y. Gui, Y. Zhou, S. Cheng, X. Yuan, H. Fan, P. Cheng, and S. Liu, “SeedPolicy: Horizon scaling via self-evolving diffusion policy for robot manipulation,” arXiv preprint arXiv:2603.05117, 2026.

[8] P. Sermanet, C. Lynch, Y. Chebotar, J. Hsu, E. Jang, S. Schaal, S. Levine, and G. Brain, “Time-contrastive networks: Self-supervised learning from video,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 1134–1141.

[9] K. Gao, F. Wang, E. Aduh, D. Randle, and J. Shi, “MuST: Multihead skill transformer for long-horizon dexterous manipulation with skill progress,” in 2025 IEEE International Conference on Robotics and Automation, 2025.

[10] V. Myers, B. Zheng, A. Dragan, K. Fang, and S. Levine, “Temporal representation alignment: Successor features enable emergent compositionality in robot instruction following,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[11] A. Khazatsky et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[12] H. R. Walke et al., “BridgeData V2: A dataset for robot learning at scale,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 1723–1736.

[13] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., “Rt-1: Robotics transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2022.

[14] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[15] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7327–7334, 2022.

[16] C. Li et al., “BEHAVIOR-1K: A human-centered, embodied AI benchmark with 1,000 everyday activities and realistic simulation,” arXiv preprint arXiv:2403.09227, 2024.

[17] E. Rosete-Beas, O. Mees, G. Kalweit, J. Boedecker, and W. Burgard, “Latent plans for task-agnostic offline reinforcement learning,” in Proceedings ofthe 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 1838–1849.

[18] Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto, “From play to policy: Conditional behavior generation from uncurated robot data,” in International Conference on Learning Representations, 2023.

[19] G. Zhou, V. Dean, M. K. Srirama, A. Rajeswaran, J. Pari, K. Hatch, A. Jain, T. Yu, P. Abbeel, L. Pinto, C. Finn, and A. Gupta, “Train offline, test online: A real robot learning benchmark,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 9197–9203.

[20] H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu, “Robot learning on the job: Human-in-the-loop autonomy and learning during deployment,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[21] S. Nasiriany, T. Gao, A. Mandlekar, and Y. Zhu, “Learning and retrieval from prior data for skill-based imitation learning,” in Proceedings of the 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 2181– 2204.

[22] R. Shah, R. Mart´ın-Mart´ın, and Y. Zhu, “MUTEX: Learning unified policies from multimodal task specifications,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 2663–2682.

[23] I. Radosavovic, B. Shi, L. Fu, K. Goldberg, T. Darrell, and J. Malik, “Robot learning with sensorimotor pre-training,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 683–693.

[24] S. Belkhale, Y. Cui, and D. Sadigh, “HYDRA: Hybrid robot actions for imitation learning,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 2113–2133.

[25] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, T. Wang, Y.-Q. Zhang, J. Liu, and X. Zhan, “X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model,” in International Conference on Learning Representations, 2026.

[26] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[27] S. Wu et al., “RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation,” arXiv preprint arXiv:2511.17441, 2025.

[28] J. H. Park, W. Choi, S. Hong, H. Seo, J. Ahn, C. Ha, H. Han, and J. Kwon, “Hierarchical action chunking transformer: Learning temporal multimodality from demonstrations with fast imitation behavior,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 648–12 654.

[29] A. C.-W. Lee, I. Chuang, L.-Y. Chen, and I. Soltani, “InterACT: Inter-dependency aware action chunking with hierarchical attention transformers for bimanual manipulation,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270. PMLR, 2025, pp. 1730–1743.

[30] NVIDIA et al., “GR00T N1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, March 2025.

[31] M. Lin, P. Ding, S. Wang, Z. Zhuang, Y. Liu, X. Tong, W. Song, S. Lyu, S. Huang, and D. Wang, “HiF-VLA: Hindsight, insight and foresight through motion representation for vision-language-action models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 20 732–20 742.

[32] A. Sridhar, J. Pan, S. Sharma, and C. Finn, “Scaling up memory for

[33] H. Lei, W. Song, H. Zhang, J. Pei, J. Chen, H. Yan, H. Zhao, P. Ding, Z. Zhang, L. Huang, D. Wang, Y. Wang, and H. Li, “RoboMemArena: A comprehensive and challenging robotic memory benchmark,” arXiv preprint arXiv:2605.10921, 2026.

robotic control via experience retrieval,” in The Fourteenth International Conference on Learning Representations, 2026.