# PIVOT: Perplexity-Informed KD-to-RL Transition Scheduling for Vertical-Domain Few-Shot Distillation

Heng Li<sup>1,2,†</sup>, Yong Zhang<sup>1,†</sup>, Ning Cheng<sup>1,\*</sup>, Zhigen Li<sup>1</sup>, Yun Zhu<sup>1</sup>, Yanmeng Wang<sup>1</sup>, Shaojun Wang<sup>1</sup>, Jing Xiao<sup>1</sup>,

<sup>1</sup>Ping An Technology (Shenzhen) Co., Ltd., China

<sup>2</sup>University of Science and Technology of China {zhangyong203, chengning211}@pingan.com.cn

## Abstract

Vertical-domain few-shot classification remains challenging for small language models, as limited supervision makes it difficult to acquire domain-specific decision knowledge. On-Policy Distillation (OPD) can improve teacher-guided adaptation by supervising student-generated rollouts, while GRPObased reinforcement learning can further refine downstream predictions. However, existing KD-to-RL pipelines typically rely on globally fixed transition schedules, ignoring that different samples may require different amounts of teacher-guided acquisition before reward-driven refinement. We propose PIVOT (Perplexity-Informed Transition Optimization), a dynamic transition framework that routes samples between OPD and GRPO according to teacher-evaluated sequence perplexity. PIVOT moves low-perplexity samples to GRPO for reward-driven refinement while keeping highperplexity samples under OPD for continued domain knowledge acquisition. Experiments on Banking77 and HWU64 show that PIVOT consistently outperforms continued OPD and globally synchronized OPD→GRPO baselines under the same number of post-warm-up student optimization steps, achieving stronger downstream performance and more stable training dynamics.

## 1 Introduction

In vertical-domain few-shot classification tasks, small language models often struggle to adapt under limited supervision due to insufficient domainspecific knowledge. This challenge is particularly severe in professional domains such as finance, where downstream prediction depends heavily on specialized terminology and domain semantics. As a result, effective adaptation increasingly relies on transferring knowledge from expert teacher models through distillation-based optimization (Hinton et al., 2015; Sanh et al., 2020). Among existing approaches, On-Policy Distillation (OPD) has emerged as an effective paradigm for teacherguided adaptation by aligning supervision with student-generated trajectories during optimization, thereby improving exposure alignment during autoregressive generation (Agarwal et al., 2024a; Gu et al., 2024).

Recent studies have further explored how knowledge distillation and reinforcement learning should be coordinated during training. GRPO has also been applied to intent detection with chain-ofthought reasoning and reward-based curriculum sampling, improving generalization beyond supervised fine-tuning (Feng et al., 2025). RLKD and KDRL (Xu et al., 2026a, 2025) show that reinforcement learning can improve downstream decision refinement beyond teacher-guided imitation. CoDistill-GRPO (Kwon et al., 2026) and related hybrid optimization frameworks (Li et al., 2026a; Ding, 2026) further demonstrate that jointly coordinating distillation and RL objectives can improve optimization efficiency and rollout quality. Meanwhile, recent OPD studies have found that prolonged distillation may introduce unreliable supervision or unstable training dynamics once student rollouts move beyond regions reliably supported by the teacher (Liu et al., 2026). Existing approaches therefore explore uncertainty-aware weighting, entropy-guided optimization, rolloutgroup coordination, and curriculum-based strategies to stabilize OPD optimization (Zheng et al., 2026; Jin et al., 2026; Yu et al., 2026; Xu et al., 2026b).

Despite these advances, existing methods largely coordinate KD and RL through globally fixed schedules or shared optimization objectives, without explicitly modeling when the learning signal should transition from distillation to reinforcement learning. This limitation is especially important in vertical-domain few-shot settings, where OPD plays a different role from conventional refinementoriented distillation. Since the student initially lacks sufficient domain-specific semantic priors, OPD must first support domain knowledge acquisition before reinforcement learning can effectively refine downstream predictions. As training progresses, however, the marginal benefit of continued OPD gradually diminishes, suggesting that the dominant learning signal should shift from teacherguided acquisition to reward-driven refinement.

A key challenge is that this KD-to-RL transition point varies substantially across samples. We observe that samples with higher teacher-side perplexity often continue to benefit from prolonged OPD optimization, indicating that they still require teacher-guided acquisition. In contrast, samples with lower teacher-side perplexity typically exhibit limited improvement under continued distillation and can enter reward-driven refinement earlier. These observations suggest that teacherevaluated sequence perplexity provides a useful signal for estimating sample-level acquisition status. Consequently, KD-to-RL coordination should be governed by sample-level optimization dynamics rather than globally synchronized training stages.

Motivated by these observations, we propose PIVOT (Perplexity-Informed Transition Optimization), a dynamic KD-to-RL transition framework that adaptively coordinates OPD-based distillation and GRPO-based refinement (Shao et al., 2024a). PIVOT progressively transitions samples from OPD to GRPO according to teacherevaluated sequence perplexity: samples with more stable teacher-aligned rollouts enter GRPO refinement earlier, while more difficult samples remain under OPD for continued domain knowledge acquisition. In this way, PIVOT avoids both premature reinforcement learning on under-acquired samples and inefficient over-distillation on samples that are already ready for refinement.

Experiments on Banking77 (Casanueva et al., 2020) and HWU64 (Liu et al., 2019) show that PIVOT consistently outperforms continued OPD and globally synchronized OPD→GRPO baselines under the same number of post-warm-up student optimization steps. Further analysis demonstrates that progressive sample-level transition yields more stable optimization dynamics and stronger downstream refinement effectiveness than fixed twostage transition schedules.

Our contributions are summarized as follows:

• We show that prolonged teacher-guided distillation does not necessarily yield better downstream GRPO refinement under verticaldomain few-shot adaptation.

• We propose PIVOT, a progressive samplelevel KD-to-RL transition framework that adaptively schedules OPD and GRPO according to teacher-evaluated sequence perplexity.

• Experiments on Banking77 and HWU64 show that PIVOT consistently outperforms continued OPD and globally synchronized OPD→GRPO under the same optimization steps while yielding more stable optimization dynamics.

## 2 Method

## 2.1 Domain Knowledge Acquisition and Reward-Driven Refinement

To facilitate domain knowledge acquisition, we first adopt On-Policy Distillation (OPD), which continuously aligns teacher supervision with the student’s evolving rollout distribution. Specifically, for input $x _ { i }$ , the student first generates a response:

$$
y _ { i } \sim p _ { \theta } ( \cdot | x _ { i } ) ,\tag{1}
$$

and the student is optimized to match the teacher distribution along on-policy generated trajectories:

$$
\mathcal { L } _ { \mathrm { O P D } } = \mathbb { E } _ { { x _ { i } } \sim \mathcal { D } , ~ y _ { i } \sim p _ { \theta } ( \cdot | x _ { i } ) } \left[ \sum _ { { t = 1 } } ^ { | y _ { i } | } D _ { \mathrm { K L } } \big ( p _ { \mathrm { T } , { t } } \| p _ { \theta , t } \big ) \right]\tag{2}
$$

where

$$
p _ { \mathrm { T } , t } = p _ { \mathrm { T } } ( \cdot | x _ { i } , y _ { i , < t } ) , \quad p _ { \theta , t } = p _ { \theta } ( \cdot | x _ { i } , y _ { i , < t } ) .
$$

After OPD-based acquisition, we further introduce GRPO refinement using a group-relative policy optimization objective. For each input $x _ { i } .$ , we sample a group of responses from the old policy $p _ { \theta _ { \mathrm { o l d } } } \colon$

$$
\{ y _ { i , j } \} _ { j = 1 } ^ { G } \sim p _ { \theta _ { \mathrm { o l d } } } ( \cdot | x _ { i } ) ,
$$

where G denotes the group size. Each response receives a scalar reward $R _ { i , j }$ , and the corresponding group-normalized advantage is computed as

$$
A _ { i , j } = \frac { R _ { i , j } - \mu _ { i } } { \sigma _ { i } + \delta } ,
$$

![](images/412fbaba42b7747f8931d454c8904055d9e227bb2ba3bda4fe3134c72283e3b3.jpg)  
Figure 1: Overview of PIVOT. PIVOT estimates sample-level acquisition states using teacher-side perplexity over on-policy student rollouts, and progressively performs asynchronous KD-to-RL transition by routing low-perplexity samples to GRPO refinement while retaining high-perplexity samples under continued OPD acquisition.

where $\mu _ { i }$ and $\sigma _ { i }$ denote the mean and standard deviation of rewards within the sampled response group for input $x _ { i } .$ respectively, and δ is a small constant for numerical stability.

For each generated token, the policy ratio is defined as

$$
r _ { i , j , t } ( \theta ) = \frac { p _ { \theta } ( y _ { i , j , t } \mid x _ { i } , y _ { i , j , < t } ) } { p _ { \theta _ { \mathrm { o l d } } } ( y _ { i , j , t } \mid x _ { i } , y _ { i , j , < t } ) } .
$$

The clipped token-level objective is then given by

$$
\begin{array} { r } { \ell _ { i , j , t } ^ { \mathrm { G R P O } } = \operatorname* { m i n } \Big ( r _ { i , j , t } A _ { i , j } , \qquad } \\ { \qquad \mathrm { c l i p } ( r _ { i , j , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i , j } \Big ) . } \end{array}\tag{3}
$$

The GRPO loss is defined as:

$$
\mathcal { L } _ { \mathrm { G R P O } } = - \mathbb { E } _ { x _ { i } , \{ y _ { i , j } \} _ { j = 1 } ^ { G } } \Bigg [ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \sum _ { t = 1 } ^ { | y _ { i , j } | } \ell _ { i , j , t } ^ { \mathrm { G R P O } } \Bigg ] ,\tag{4}
$$

where

$$
\begin{array} { r } { x _ { i } \sim \mathcal { D } , \quad \{ y _ { i , j } \} _ { j = 1 } ^ { G } \sim p _ { \theta _ { \mathrm { o l d } } } ( \cdot | x _ { i } ) . } \end{array}
$$

GRPO enables stable relative reward optimization over grouped candidate responses without requiring an additional value model.

In our framework, OPD and GRPO serve complementary optimization roles. OPD supports domain knowledge acquisition through rollout expansion, whereas GRPO refinement further improves response quality once the marginal acquisition gain from continued distillation gradually diminishes. This distinction motivates our asynchronous KDto-RL transition scheduling framework.

## 2.2 Heterogeneous Acquisition Dynamics and Acquisition-State Estimation

Different samples exhibit substantially different acquisition dynamics throughout optimization. Some samples rapidly approach rollout-coverage saturation, whereas others continue benefiting from prolonged OPD acquisition. Consequently, the optimal KD-to-RL transition point varies substantially across samples.

Samples approaching rollout-coverage saturation are better suited for earlier GRPO refinement, while others continue benefiting from OPD-based acquisition. To estimate such acquisition dynamics online, we use teacher-side perplexity over student-generated responses. Lower perplexity indicates that the current student policy has already approached rollout-coverage saturation, whereas higher perplexity suggests that the student policy still benefits from continued acquisition under teacher supervision.

For each input sample $x _ { i } .$ , the student first performs on-policy rollout and generates a response group:

$$
\mathcal { V } _ { i } = \{ y _ { i 1 } , y _ { i 2 } , . . . , y _ { i K } \} .\tag{5}
$$

The teacher model then evaluates each generated response using normalized sequence-level perplexity:

$$
\ell _ { i j } = - { \frac { 1 } { | y _ { i j } | } } \sum _ { m = 1 } ^ { | y _ { i j } | } \log p _ { \mathrm { T } } ( y _ { i j , m } \mid x _ { i } , y _ { i j , < m } ) ,\tag{6}
$$

$$
\mathrm { P P L } _ { i j } = \exp ( \ell _ { i j } ) ,\tag{7}
$$

where $| y _ { i j } |$ denotes the response length.

Based on the candidate-level perplexities, we further compute a prompt-level acquisition-state score:

$$
u _ { i } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \mathrm { P P L } _ { i j } .\tag{8}
$$

A larger $u _ { i }$ indicates that the sample still benefits from continued acquisition under teacher supervision, whereas smaller values suggest that rolloutcoverage saturation is gradually approached. Since absolute perplexity scores are not directly comparable across prompts, we normalize them within each mini-batch:

$$
z _ { i } = \frac { u _ { i } - \mu _ { B } } { \sigma _ { B } + \epsilon } ,\tag{9}
$$

where $\mu _ { B }$ and $\sigma _ { B }$ denote the batch-level mean and standard deviation, respectively. We use the normalized score $z _ { i }$ as the acquisition state score for routing.

Lower $z _ { i }$ indicates relatively lower teacher-side perplexity within the current mini-batch, suggesting that the sample has acquired sufficient domain knowledge under OPD optimization. In contrast, higher $z _ { i }$ suggests that the student rollouts remain less aligned with teacher-supported generation patterns and therefore still benefit from continued OPD acquisition.

## 2.3 Asynchronous KD-to-RL Transition Scheduling

Instead of adopting a globally fixed KD-to-RL schedule, PIVOT progressively transitions samples from OPD acquisition to GRPO refinement according to their acquisition state scores.

Specifically, during training, we maintain a transition ratio:

$$
r ( t ) \in [ 0 , 1 ] ,\tag{10}
$$

where t denotes the current training step. The transition ratio is progressively increased following a cosine scheduling strategy:

$$
r ( t ) = \frac { 1 } { 2 } \left( 1 - \cos \frac { \pi t } { T } \right) ,\tag{11}
$$

where $T$ denotes the total number of training steps.

At each training step, samples are ranked according to their normalized acquisition-state scores $z _ { i } .$ Samples with lower acquisition-state scores are considered closer to rollout-coverage saturation and are therefore transitioned earlier into GRPO refinement. Specifically, we progressively route the lowest-ranked $r ( t )$ proportion of samples into GRPO optimization, while the remaining samples continue OPD-based acquisition. Let $S _ { t }$ denote this selected GRPO subset at training step t.

Let $B _ { t }$ denote the mini-batch at training step t. We define a binary routing indicator $m _ { i } = \mathbf { 1 } ( x _ { i } \in$

$S _ { t } )$ , where $m _ { i } ~ = ~ 1$ indicates that sample $x _ { i }$ is routed to GRPO refinement and $m _ { i } = 0$ indicates that it remains under OPD acquisition. The training objective at step t is then defined as

$$
\mathcal { L } _ { t } = \frac { 1 } { \vert \mathcal { B } _ { t } \vert } \sum _ { x _ { i } \in \mathcal { B } _ { t } } \left[ m _ { i } \mathcal { L } _ { \mathrm { G R P O } } ^ { ( i ) } + ( 1 - m _ { i } ) \mathcal { L } _ { \mathrm { O P D } } ^ { ( i ) } \right] ,\tag{12}
$$

where $\mathcal { L } _ { \mathrm { { G R P O } } } ^ { ( i ) }$ and $\mathcal { L } _ { \mathrm { O P D } } ^ { ( i ) }$ denote the sample-level GRPO and OPD losses for $x _ { i } ,$ , respectively.

In this way, PIVOT enables asynchronous KDto-RL transition according to acquisition dynamics rather than globally synchronized optimization stages. Samples approaching rollout-coverage saturation transition earlier into GRPO refinement, while other samples continue benefiting from OPDbased acquisition. This progressive transition strategy improves optimization efficiency and downstream decision quality compared with globally fixed KD-to-RL schedules.

## 3 Experiments

## 3.1 Experimental Setup

Datasets and few-shot setting. We evaluate our method on Banking77 and HWU64. For Banking77, we construct a class-balanced 5-shot training set by randomly sampling 5 examples per intent from the original training split, resulting in 385 training instances over 77 intents. For HWU64, we construct a class-balanced 10-shot training set by sampling 10 examples per intent, resulting in 640 training instances over 64 intents. We use the official test splits for evaluation. The few-shot sampling seed is fixed to 42 for all methods. Both datasets contain fine-grained user intents and are used to evaluate few-shot domain adaptation under limited labeled supervision.

Models. Unless otherwise specified, we use Qwen2.5-0.5B-Instruct as the student model and Qwen2.5-7B-Instruct as the teacher model. We further evaluate whether the proposed method remains effective when scaling the student model from Qwen2.5-0.5B-Instruct to Qwen2.5-1.5B-Instruct. For stronger-teacher experiments on Banking77, we replace the default teacher with DianJin-32B (Zhu et al., 2025). The teacher model is kept frozen throughout training and is only used to provide supervision signals for distillation and teacher-side perplexity estimation.

Baselines. We compare with the following baselines. Teacher performs zero-shot prompt-only inference using the teacher model. SeqKD fine-tunes the student on teacher-generated full responses under the same instruction prompt (Kim and Rush, 2016). SFT→GRPO first lightly fine-tunes the student using gold labels and then applies GRPO refinement. DataAug-SFT fine-tunes the student on a teacher-generated and teacher-filtered augmented training set constructed from the few-shot Banking77 split. We include this baseline on Banking77, where the augmented set is constructed from the same 5-shot training split. OPD trains the student with OPD for the full optimization budget. OPD→GRPO first performs OPD and then globally switches all samples to GRPO at a fixed transition checkpoint.

![](images/39332df84e8b0505034365cb64652467636487d06e4b7b3d0168fca9f21aa159.jpg)  
Figure 2: Training dynamics on Banking77 under aligned total optimization steps. For globally synchronized OPD→GRPO, the x-axis denotes cumulative optimization steps across the OPD and GRPO stages, and the purple star marks the global transition point. Compared with continued OPD and globally synchronized OPD→GRPO, our progressive sample-level transition achieves higher final accuracy with substantially smoother reward, entropy, and completion-length dynamics.

Training protocol. Unless otherwise specified, all trainable student-side methods perform 480 post-warm-up student parameter-update steps. Following prior OPD practice (Li et al., 2026b), GRPO, OPD, OPD→GRPO, and PIVOT methods first share a 2-epoch off-policy SFT warm-up on teacher-generated responses, which is not counted in the 480 steps. After warm-up, GRPO and OPD use all 480 steps for GRPO refinement and OPD optimization, respectively. OPD→GRPO globally switches from OPD to GRPO at the peak OPD checkpoint, while PIVOT progressively routes samples from OPD to GRPO according to teacherevaluated sequence perplexity.

<table><tr><td>Method</td><td>Banking77</td><td>HWU64</td></tr><tr><td>Teacher</td><td>73.25</td><td>79.65</td></tr><tr><td>SeqKD</td><td>58.70</td><td>70.17</td></tr><tr><td>GRPO</td><td>67.66</td><td>77.60</td></tr><tr><td>OPD</td><td>72.47</td><td>79.55</td></tr><tr><td>OPD→GRPO</td><td>78.05</td><td>83.09</td></tr><tr><td>PIVOT</td><td>80.26</td><td>85.04</td></tr></table>

Table 1: Main results under 480 post-warm-up student optimization steps. All results report test-set classification accuracy.

Reward and evaluation. For GRPO-based methods, each sampled response is parsed into a predicted intent label from the field “Intent Category”. The main reward is based on whether the parsed label matches the gold intent. We additionally use lightweight auxiliary rewards to encourage valid formatting and consistency between the final answer and the listed candidate labels. Advantages are normalized within each response group. At evaluation time, we use greedy decoding and compute micro accuracy on the official test split. Invalid or unparsable responses are counted as incorrect.

Progressive routing. For our method, we compute teacher-evaluated length-normalized sequence perplexity over student-generated responses and average the scores within each response group. The resulting group-level scores are normalized within each mini-batch using z-score normalization.

After an initial OPD warm-up stage, a cosine schedule progressively increases the proportion of samples transitioned to GRPO throughout training. At each step, samples with lower normalized perplexity are transitioned earlier to GRPO refinement, while higher-perplexity samples continue OPD optimization.

## 3.2 Main Results

Table 1 reports the main results under the same number of post-warm-up student optimization steps. Compared with continued OPD, globally synchronized OPD→GRPO already yields substantial improvement on both datasets, suggesting that downstream GRPO refinement is important beyond prolonged imitation alone. However, PIVOT further improves over OPD→GRPO across both Banking77 and HWU64, indicating that globally fixed transition schedules remain suboptimal for coordinating OPD and GRPO during training.

Interestingly, PIVOT also surpasses teacher inference on both datasets, suggesting that progressive sample-level transition can produce a stronger taskspecialized student policy under few-shot adaptation. We additionally compare against a teachergenerated data augmentation baseline on Banking77 in Appendix A.5. Although DataAug-SFT improves over standard SFT, it remains below both continued OPD and PIVOT, suggesting that the gains are not solely attributable to additional teacher-generated training data.

## 3.3 Optimization Dynamics

Figure 2 compares test accuracy, reward, entropy, and completion length under aligned total optimization steps. Our method consistently achieves the highest final test accuracy and surpasses globally synchronized OPD→GRPO before the global transition point, suggesting that progressive samplelevel transition improves downstream refinement effectiveness under the step-matched protocol.

Although globally synchronized OPD→GRPO eventually reaches comparable reward levels, it exhibits noticeably unstable optimization behavior near the transition stage, including abrupt completion-length increase and entropy spikes. In contrast, our method maintains substantially smoother entropy and completion-length dynamics throughout training, indicating that progressively transferring low-perplexity samples into GRPO avoids the optimization discontinuity introduced by globally synchronized switching. Interestingly, our method achieves better final accuracy despite similar late-stage reward, suggesting that progressive transition improves the effectiveness of rewarddriven refinement rather than merely increasing reward magnitude.

![](images/4f4ec40b58e23bf0b41dd8e19283a38d02dc0c10bdfd195cb1189d35f1f8597e.jpg)

![](images/3eeb5331fe6066bb4e0f2fc982338b16746e2285496c234de6616f96f354f74e.jpg)  
Figure 3: Pass@32 performance grouped by teacherside perplexity buckets during OPD training. Highperplexity samples exhibit substantially larger relative improvement throughout training, whereas lowperplexity samples saturate much earlier, suggesting that the optimal transition timing from OPD to RL varies across samples.

## 3.4 Scaling Analysis

We further evaluate PIVOT across different student and teacher model configurations. As shown in Table 2, PIVOT consistently improves over OPD→GRPO across all evaluated settings, including larger student models and stronger teacher models. Notably, the improvement remains consistent even when scaling the student from 0.5B to 1.5B parameters, suggesting that the benefit of progressive sample-level transition is not tied to a specific student capacity. Similarly, PIVOT continues to outperform OPD→GRPO under stronger teacher models, indicating that adaptive transition remains beneficial even when teacher supervision quality improves substantially.

## 3.5 Ablation and Routing Analysis

We further analyze the design choices of progressive OPD-to-GRPO transition, including both the transition schedule and the routing strategy. As shown in Table 4, all progressive routing variants consistently outperform globally synchronized OPD→GRPO, indicating that asynchronous sample-level transition is more effective than switching all samples at a fixed checkpoint. Among different schedules, cosine routing achieves the strongest overall performance, suggesting that gradually increasing the GRPO transition ratio throughout training provides a better balance between continued OPD optimization and downstream refinement. Table 3 further shows that the best results are achieved by retaining high-PPL samples in OPD while transitioning low-PPL samples to GRPO earlier, suggesting that different samples benefit from substantially different transition timing during training. Figure 3 further supports this behavior: lower-PPL samples rapidly approach rolloutquality saturation, whereas higher-PPL samples continue improving under prolonged OPD optimization. These observations support teacher-side perplexity as an effective signal for progressive sample-level KD-to-RL transition.

<table><tr><td>Dataset</td><td>Setting</td><td>Student</td><td>Teacher</td><td>OPD→GRPO</td><td>PIVOT</td><td>∆</td></tr><tr><td>Banking77</td><td>Default</td><td>0.5B</td><td>Qwen2.5-7B</td><td>78.05</td><td>80.26</td><td>+2.21</td></tr><tr><td>Banking77</td><td>Larger student</td><td>1.5B</td><td>Qwen2.5-7B</td><td>80.00</td><td>82.82</td><td>+2.82</td></tr><tr><td>Banking77</td><td>Stronger teacher</td><td>0.5B</td><td>DianJin-32B</td><td>80.30</td><td>81.33</td><td>+1.03</td></tr><tr><td>HWU64</td><td>Default</td><td>0.5B</td><td>Qwen2.5-7B</td><td>83.09</td><td>85.04</td><td>+1.95</td></tr><tr><td>HWU64</td><td>Larger student</td><td>1.5B</td><td>Qwen2.5-7B</td><td>82.53</td><td>85.97</td><td>+3.44</td></tr><tr><td>HWU64</td><td>Stronger teacher</td><td>0.5B</td><td>Qwen3-32B</td><td>84.39</td><td>86.71</td><td>+2.32</td></tr></table>

Table 2: Scaling analysis across different student and teacher model configurations. ∆ denotes the improvement of PIVOT over OPD→GRPO.
<table><tr><td>Dataset</td><td>Low-PPL</td><td>Random</td><td>High-PPL</td></tr><tr><td>Banking77</td><td>80.26</td><td>78.31</td><td>77.27</td></tr><tr><td>HWU64</td><td>85.04</td><td>84.11</td><td>83.46</td></tr></table>

Table 3: Routing-rule ablation under progressive OPDto-GRPO transition. Column names indicate which samples are preferentially routed to GRPO earlier during training.
<table><tr><td>Transition Strategy</td><td>ACC</td></tr><tr><td>Global OPD→GRPO</td><td>78.05</td></tr><tr><td>Linear progressive routing</td><td>79.58</td></tr><tr><td>Quadratic progressive routing</td><td>79.19</td></tr><tr><td>Cosine progressive routing</td><td>80.26</td></tr></table>

Table 4: Transition-schedule ablation on Banking77 under the same number of student optimization steps.

## 4 Discussion

## 4.1 Distillation Quality and Refinement Readiness Are Not Necessarily Aligned

Our results suggest that stronger distillation quality does not necessarily imply better downstream refinement readiness. Existing two-stage adaptation pipelines often assume that continued teacher alignment monotonically improves the initialization quality for subsequent preference optimization. However, our experiments show that the strongest downstream GRPO performance frequently emerges before full OPD convergence.

One possible explanation is that prolonged imitation gradually narrows the student rollout distribution around teacher-preferred responses. Although this improves teacher alignment, it may also reduce the diversity and observability of informative candidate responses required for downstream GRPO refinement. As a result, stronger imitation quality does not always translate into more effective downstream optimization.

These observations suggest that distillation and downstream refinement correspond to partially different optimization objectives, and that maximizing teacher alignment alone may not be sufficient for effective refinement-oriented adaptation.

## 4.2 Implications for Adaptive Distillation-to-Refinement Transition

Our findings further suggest that the transition from distillation to downstream refinement should not be globally synchronized across all samples. Different samples exhibit substantially different optimization dynamics during OPD training, implying that the optimal transition timing may vary significantly across samples.

In particular, low teacher-side perplexity consistently correlates with more stable rollout behavior and stronger pass@32 performance throughout training, suggesting that these samples are more suitable for earlier refinement. In contrast, highperplexity samples often continue benefiting from prolonged OPD optimization before downstream GRPO becomes effective.

Although our study focuses on OPD and GRPO under vertical-domain few-shot classification, similar interactions may arise more broadly in distillation-based adaptation settings involving iterative imitation and downstream preference optimization. More generally, our results highlight the importance of considering not only teacher alignment quality, but also how optimization dynamics influence downstream refinement behavior throughout training.

## 5 Related Work

## 5.1 On-Policy Distillation and Adaptive Optimization

Knowledge distillation has become a widely adopted paradigm for transferring capabilities from large language models to smaller models under constrained supervision or computational settings (Hinton et al., 2015; Sanh et al., 2020). Recent work has increasingly explored On-Policy Distillation (OPD), where teacher supervision is continuously aligned with the student’s evolving rollout distribution during optimization, improving exposure alignment during autoregressive generation (Agarwal et al., 2024b; Gu et al., 2024).

Recent OPD studies further explore adaptive optimization strategies for stabilizing supervision under heterogeneous rollout conditions. Existing approaches include entropy-aware distillation objectives (Jin et al., 2026), uncertainty-calibrated supervision weighting (Zheng et al., 2026), rollout-group coordination (Yu et al., 2026), and competencebased optimization schedules (Xu et al., 2026b). Recent analyses of OPD failure modes further show that teacher guidance can become unreliable on student-generated prefixes, especially when rollouts move away from prefixes commonly visited by the teacher (Fu et al., 2026).

These studies suggest that the effectiveness of teacher supervision can vary substantially throughout OPD optimization under heterogeneous rollout conditions. Our work is related to these adaptive OPD optimization methods but focuses on a different problem setting. Instead of improving supervision quality within OPD itself, we study how acquisition dynamics should govern the transition between distillation and downstream RL refinement during vertical-domain few-shot adaptation.

## 5.2 KD-RL Coordination and Adaptive Transition

Preference- and reward-based optimization methods such as RLHF, DPO, and GRPO have become important paradigms for improving downstream behavior beyond supervised imitation (Ouyang et al., 2022; Rafailov et al., 2023; Shao et al., 2024b). In particular, GRPO improves optimization stability through group-relative reward normalization and has shown strong effectiveness in reasoningoriented refinement settings (Shao et al., 2024b). Recent work further applies GRPO to intent detection by combining it with chain-of-thought reasoning and reward-based curriculum sampling, showing that reward-driven optimization can improve intent understanding and generalization beyond supervised fine-tuning (Feng et al., 2025).

Motivated by the complementary roles of distillation and downstream refinement, recent work has increasingly explored how knowledge distillation and reinforcement learning should be coordinated throughout optimization. RLKD (Xu et al., 2026a) and KDRL (Xu et al., 2025) show that reinforcement learning can complement teacher-guided distillation by improving exploration and downstream refinement beyond imitation alone. SuperRL (Liu et al., 2025) further incorporates supervised signals into reinforcement learning to improve reasoning optimization. G-OPD (Yang et al., 2026) further establishes a theoretical connection between OPD and KL-regularized reinforcement learning, suggesting that OPD itself can be interpreted as a dense reward optimization process. CoDistill-GRPO (Kwon et al., 2026) and related hybrid optimization frameworks (Li et al., 2026a) additionally demonstrate that jointly coordinating distillation and reinforcement learning can improve optimization efficiency and rollout quality during training.

However, existing approaches primarily rely on globally shared objectives or fixed optimization schedules, without explicitly modeling samplelevel transition dynamics between distillation and reinforcement learning.

## 6 Conclusion

In this paper, we propose PIVOT, a progressive sample-level KD-to-RL transition framework for vertical-domain few-shot adaptation. PIVOT adaptively schedules OPD and GRPO according to teacher-evaluated sequence perplexity rather than globally fixed transition schedules. Experiments on Banking77 and HWU64 show that PIVOT consistently outperforms continued OPD and globally synchronized OPD→GRPO under the same number of post-warm-up student optimization steps, while yielding more stable optimization dynamics during training. Our results suggest that adaptive sample-level transition provides an effective alternative to fixed two-stage distillation and refinement schedules.

## 7 Limitations

Our study has several limitations. First, PIVOT relies on teacher-evaluated sequence perplexity as a heuristic routing signal, and we do not provide a theoretical characterization of why perplexity best reflects refinement readiness across different optimization settings. Second, our experiments focus on vertical-domain few-shot classification with relatively structured outputs and reward signals. Whether similar transition dynamics extend to more complex generation tasks remains an open question. Third, our comparisons align the number of student optimization steps rather than total training compute. Because PIVOT continues teacherside perplexity evaluation for samples routed to GRPO, it introduces additional teacher inference overhead relative to the globally synchronized baseline after its transition. The reported results should therefore be interpreted as step-matched rather than strictly compute-matched comparisons. We leave broader theoretical analysis, stricter computematched evaluation, and evaluation on more diverse tasks for future work.

## AI Assistance Statement

Generative AI tools, including ChatGPT and Gemini, were used to assist with language editing during the preparation of this work. All generated content was reviewed and verified by the authors. The authors take full responsibility for the accuracy, integrity, and originality of the final manuscript.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Pi otr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024a. On-policy distillation of language models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024b. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020.´ Efficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processingfor Conversational AI, pages 38–45, Online. Association for Computational Linguistics.

Ken Ding. 2026. Hdpo: Hybrid distillation policy optimization via privileged self-distillation. Preprint, arXiv:2603.23871.

Zihao Feng, Xiaoxue Wang, Ziwei Bai, Donghang Su, Bowen Wu, Qun Yu, and Baoxun Wang. 2025. Improving generalization in intent detection: Grpo with reward-based curriculum sampling. Preprint, arXiv:2504.13592.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. 2026. Revisiting on-policy distillation: Empirical failure modes and simple fixes. Preprint, arXiv:2603.25562.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pages 32694–32717.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Preprint, arXiv:1503.02531.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. 2026. Entropy-aware onpolicy distillation of language models. arXiv preprint arXiv:2603.07079.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Soo Min Kwon, Ziteng Sun, Ananda Theertha Suresh, Himanshu Jain, and Sanjiv Kumar. 2026. Codistill-grpo: A co-distillation recipe for efficient group relative policy optimization. Preprint, arXiv:2605.08873.

Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. 2026a. Unifying grouprelative and self-distillation policy optimization via sample routing. Preprint, arXiv:2604.02288.

Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. 2026b. Unifying grouprelative and self-distillation policy optimization via sample routing. Preprint, arXiv:2604.02288.

Xingkun Liu, Arash Eshghi, Pawel Swietojanski, and Verena Rieser. 2019. Benchmarking natural language understanding services for building conversational agents. Preprint, arXiv:1903.05566.

Xinyu Liu, Kechen Jiao, Chunyang Xiao, Runsong Zhao, Junhao Ruan, Bei Li, Jiahao Liu, Qifan Wang, Xin Chen, Jingang Wang, and 1 others. 2026. Teacher-guided policy optimization for on-policy reasoning distillation under large policy divergence. arXiv preprint arXiv:2605.13230.

Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, and Dongmei Zhang. 2025. Superrl: Reinforcement learning with supervision to boost language model reasoning. Preprint, arXiv:2506.01096.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2020. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. Preprint, arXiv:1910.01108.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024a. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024b. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. 2025. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. Preprint, arXiv:2506.02208.

Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan, Huawei Shen, and Xueqi Cheng. 2026a. Rlkd: Distilling llms’ reasoning via reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 34151–34159.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. 2026b. Paced: Distillation and onpolicy self-distillation at the frontier of student competence. Preprint, arXiv:2603.11178.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. 2026. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. Preprint, arXiv:2602.12125.

Weichen Yu, Xiaomin Li, Yizhou Zhao, Xiaoze Liu, Ruowang Zhang, Haixin Wang, Yinyi Luo, Chen Henry Wu, Gaurav Mittal, Matt Fredrikson,

and 1 others. 2026. Multi-rollout on-policy distillation via peer successes and failures. arXiv preprint arXiv:2605.12652.

Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, and Xunliang Cai. 2026. Scope: Signal-calibrated on-policy distillation enhancement with dual-path adaptive weighting. Preprint, arXiv:2604.10688.

Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. 2025. Dianjin-r1: Evaluating and enhancing financial reasoning in large language models. Preprint, arXiv:2504.15716.

## A Implementation Details

## A.1 Dataset Construction

We construct class-balanced few-shot training sets from the original training splits. Banking77 uses 5 examples per intent, resulting in 385 training examples over 77 intents, while HWU64 uses 10 examples per intent, resulting in 640 training examples over 64 intents. The sampling seed is fixed to 42. We use the official test splits for evaluation and report micro accuracy. Unless otherwise stated, all experiments are conducted on the same sampled few-shot split. Following prior few-shot adaptation settings, the main results are reported on a single split because OPD and GRPO training are computationally expensive.

## A.2 Prompt Templates

All methods use task-specific instruction prompts containing the full label set. Banking77 additionally includes natural-language intent definitions, while HWU64 uses only the intent label list. SFT-ZeroShot uses a simplified label-only supervision format, while teacher inference, SeqKD, OPD, GRPO, and PIVOT use the structured reasoning prompts shown in Tables 5 and 6.

Here, {labels\_str} contains the full HWU64 intent label set.

## A.3 Output Parsing

We parse the final prediction from the field “Intent Category”. The first valid label appearing after “Intent Category:” is matched against the task-specific label set. If multiple valid labels appear after this field, the first valid match is used. If no valid label can be matched, the prediction is counted as incorrect. No manual correction is applied.

## A.4 Baseline Implementations

GRPO and OPD-based methods use the same 2-epoch off-policy SFT warm-up on teachergenerated responses before on-policy optimization.

Teacher. The frozen teacher model performs zero-shot prompt-only inference using the same task prompt as evaluation.

SFT-ZeroShot. The student is fine-tuned on the few-shot gold training set using label-only supervision targets.

SeqKD. For each few-shot training example, the teacher generates one full response under the same instruction prompt. The student is then fine-tuned on the resulting teacher-generated responses.

GRPO. The student is optimized directly with GRPO after the off-policy warm-up stage.

DataAug-SFT. The student is fine-tuned on a teacher-generated augmented training set with teacher-based filtering constructed from the same Banking77 5-shot split. This baseline is included only on Banking77.

OPD. The student is trained with OPD for all 480 post-warm-up student optimization steps.

OPD→GRPO. The student is first trained with OPD and then globally switched to GRPO at a fixed transition checkpoint. The best transition checkpoint is selected within the same total of 480 post-warm-up student optimization steps.

## A.5 Data Augmentation Baseline

We additionally evaluate a teacher-generated data augmentation baseline on Banking77 to test whether the gains of PIVOT can be explained by additional synthetic training data alone.

Starting from the same 5-shot Banking77 split, we construct an augmented training set using intent clustering, teacher-based query generation, and teacher self-filtering. Semantically similar intents are first grouped into clusters. For each cluster, the teacher generates additional user queries conditioned on the target intent, intent definition, and several anchor examples. The generation prompt encourages diverse and label-discriminative queries. Generated samples are then reclassified by the teacher, and only samples whose predicted labels match the target intent are retained.

Using this pipeline, we generate 5,650 candidate augmented examples, of which 5,119 are retained after teacher self-filtering. Fine-tuning on the resulting augmented set achieves 71.75 test accuracy on Banking77. While synthetic data augmentation substantially improves over standard SFT, it remains below both continued OPD and OPD→GRPO training. Notably, introducing the OPD-to-GRPO transition yields a large additional gain, suggesting that the effectiveness of PIVOT cannot be explained solely by additional teachergenerated supervision or extended optimization.

## A.6 Training Hyperparameters

Table 9 summarizes the main training hyperparameters used across all experiments. All experiments

Table 5: Prompt template for Banking77.  
```markdown
You are an expert intent classifier for a banking customer service bot.
Your task is to classify the user's query into exactly one of the following 77 intent categories.
### Intent Definitions
{definitions_str}
### Instructions
1. **Analyze Key Elements**: Identify the core action (e.g., "top up", "transfer") and the object.
2. **List Candidates (Crucial)**:
- Before deciding, list 3-5 category names from the definitions that are most relevant to the query.
3. **Compare & Filter**:
- Compare the candidates against the specific details of the query.
- Explicitly rule out the ones that don't fit perfectly.
4. **Final Conclusion**: Select the single best matching category.
### Response Format
You must strictly follow this format:
**Analysis:** [Brief analysis of key elements]
**Candidates:** [List 3-5 potential category names here]
**Reasoning:** [Your comparison and filtering process]
**Intent Category:** [The exact category name]
```  
{definitions\_str} contains the full set of Banking77 intent labels and their natural-language definitions.

Table 6: Prompt template for HWU64.  
![](images/b015626ef182c02cbc8ff2ba393851499b479df839ee50d41de365c5db8e9e59.jpg)  
{labels\_str} contains the full HWU64 intent label set.

<table><tr><td>Method</td><td>Banking77</td></tr><tr><td>SFT-ZeroShot</td><td>60.30</td></tr><tr><td>DataAug-SFT</td><td>71.75</td></tr><tr><td>Continued OPD</td><td>72.47</td></tr><tr><td>OPD→GRPO</td><td>78.05</td></tr><tr><td>PIVOT</td><td>80.26</td></tr></table>

Table 7: Comparison between teacher-generated data augmentation and different OPD/GRPO training strategies on Banking77. DataAug-SFT is trained on 5,119 teacher-filtered synthetic examples generated from the same 5-shot split.

were conducted on a server with four NVIDIA A100 GPUs.

## A.7 Reward Function

For GRPO-based methods, the final intent label is parsed from each generated response. The reward consists of three components: a correctness reward based on whether the parsed label matches the gold intent (+1 / -1), a candidate-alignment reward encouraging consistency between the final prediction and the listed candidate labels (+0.5 / -0.5), and a lightweight format reward encouraging valid structured outputs (+0.1). We do not apply an explicit length penalty. A KL penalty with coefficient 0.001 is used during GRPO optimization, and advantages are normalized within each response group.

Table 8: Prompt template for teacher-based data augmentation.

You are an expert at generating realistic bank customer inquiry data.   
Your task is to simulate real banking customers, generating highly conversational   
and natural customer service queries that accurately map to specific intents.   
I need to augment training data for a highly sensitive bank intent recognition model.   
Below is a group of {num\_intents} intents that are semantically similar and easily confused.   
They share overlapping vocabulary but have critical semantic differences.   
For each target intent, you are given:   
- the intent name;   
- its definition;   
- several reference anchor queries.   
TASK:   
Generate exactly {n\_per\_intent} completely new customer queries for EACH intent.   
Total output = {n\_per\_intent \* num\_intents} items.   
Strict Constraints:   
1. Boundary Exclusivity:   
Each query must be unambiguous and must only map to its declared intent,   
not to any other intent in this group.   
2. Extreme Diversity:   
Vary sentence length, structure, formality, and scenario.   
Use informal speech, colloquialisms, and specific real-life situations.   
Do not simply paraphrase the seed examples.   
3. Scenario-driven:   
Imagine concrete real-life situations, such as standing at a supermarket checkout,   
receiving a bank notification, or finding an unexpected charge.   
4. No repetition:   
Do not repeat any reference anchor sentence.   
Output ONLY a JSON array:   
[   
{"intent": "{intent\_1}", "text": "..."},   
{"intent": "{intent\_2}", "text": "..."}   
]

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Total optimization steps Batch size Rollouts / group size Warmup ratio Max sequence length Max new tokens Temperature</td><td>480 256 8 0.05 4096 512 1.0 0.9</td></tr><tr><td>Top-k Precision Fine-tuning type</td><td>50 bf16 full</td></tr><tr><td>Learning rate (OPD) Learning rate (GRPO / Ours) KL coefficient (GRPO / Ours)</td><td> $1 \times 1 0 ^ { - 5 }$   $1 \times 1 0 ^ { - 6 }$  0.001</td></tr></table>

Table 9: Training hyperparameters used in all experiments.