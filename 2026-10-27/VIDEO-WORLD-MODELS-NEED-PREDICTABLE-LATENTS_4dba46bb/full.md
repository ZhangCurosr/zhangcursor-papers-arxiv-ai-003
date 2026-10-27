# VIDEO WORLD MODELS NEED PREDICTABLE LATENTS

Mattia Scardecchia Sherry Yang

New York University

mattia.scardecchia@nyu.edu

## ABSTRACT

Video world models trained on large-scale robot data promise a safe and scalable approach for learning and evaluating policies that act in the physical world. However, they still struggle with long-horizon rollouts and out-of-distribution generalization, and they remain computationally demanding at inference time. Typically, video world models make predictions in the latent space of a reconstruction-trained autoencoder, which however is not directly optimized for dynamics modeling. On the other hand, world models designed for latent planning usually optimize latents to be predictable, but they forgo accurate prediction of observations in pixel space. In this paper, we show that it is possible to achieve the best of both worlds: a latent geometry optimized to make future prediction as simple as possible without sacrificing accurate prediction of future observations. More precisely, we find that temporal objectives for representation learning, paired with a past-conditioned decoder architecture, lead to consistent improvements in generative video world models at the same training budget. The gains include up to 5x cheaper inference, over 7x faster training, and 15–24% lower pixel and perceptual errors out of distribution and over long horizons. Our world models also agree more closely with real-world policy evaluation when using a vision-language model to judge task success from their pixel-space predictions. Overall, our study indicates that temporal predictability is an important design criterion for the latent space of video world models, and provides concrete guidance on how to incorporate it into the training of autoencoders.

## 1 INTRODUCTION

Learning from experience with high-capacity function approximators has led to unprecedented capabilities in domains like games (Silver et al., 2018), mathematics (Hubert et al., 2026), and programming (OpenAI, 2025). These successes have been enabled by two key ingredients: a cheap, safe, parallelizable way to interact with the environment, and a verifiable reward signal to drive policy improvement. Extending these results to domains like robotics, autonomous driving, and scientific research is challenging because interaction with the environment is often costly and unsafe, and a reward function can be difficult to specify. While these difficulties have been overcome in specific applications, like drone racing (Kaufmann et al., 2023) and robotic table tennis (Dürr et al., 2026) this has required substantial engineering effort to build an accurate simulator for policy training and sim-to-real transfer.

Recently, video world models have emerged as a promising and scalable approach to enable learning from experience in non-verifiable, costly, and unsafe domains, including the physical world. By learning a photorealistic action-conditioned simulator from video data, these models allow cheap, safe, and parallelizable interaction with imagined environments (Hafner et al., 2025b; Sharma et al., 2026). Furthermore, their pixel-space predictions can be fed to a general-purpose vision-language model (VLM) acting as a reward model, enabling policy evaluation (Quevedo et al., 2026), policy training via reinforcement learning (RL) (Sharma et al., 2026), and planning via imagined rollouts (Du et al., 2024). However, despite their promise, video world models still struggle with long-horizon rollouts (Guo et al., 2026b; Bardhan et al., 2026; Huang et al., 2026a) and out-of-distribution generalization (Tseng et al., 2025; Zhang et al., 2026a; Kang et al., 2025; Chen et al., 2026), and they remain computationally demanding at inference time (Quevedo et al., 2026; Gemini Robotics Team, 2025; Jeon et al., 2026; Lu et al., 2026).

![](images/043175fd7525e13450ca8cbcf1c1a414ef8a409c0355d7848512d6ed6495fa4e.jpg)  
Figure 1: Method overview. (1) An autoencoder is fine-tuned on pairs of nearby video frames $x _ { t } .$ $x _ { t + 1 }$ , with a reconstruction loss and a temporal objective on the latents $z _ { t } , z _ { t + 1 }$ and the action $a _ { t } ;$ the decoder D conditions on the past frame $x _ { t }$ to reconstruct $x _ { t + 1 }$ from its latent. (2) An actionconditioned diffusion world model is trained on the latents from the frozen encoder E. (3) From a context frame $x _ { 1 }$ and a sequence of actions $a _ { 1 : T : }$ , the world model generates latents, which D decodes conditioning on $x _ { 1 }$ as the past frame. We highlight in orange and green the two components that distinguish our recipe from a standard video world model: the temporal objective, and the past-conditioned decoder.

Following common practice in visual generative modeling (Rombach et al., 2022; Blattmann et al., 2023; Peebles & Xie, 2023), most existing video world models make predictions in the latent space of an autoencoder learned through reconstruction (Bruce et al., 2024; Decart et al., 2024; NVIDIA, 2025; Quevedo et al., 2026; Huang et al., 2026b). However, a reconstruction objective does not necessarily produce latents whose dynamics is easy to model (Yao et al., 2025; Kouzelis et al., 2025; Skorokhodov et al., 2025). A different line of work has focused on developing world models that enable efficient latent planning, by learning a compact latent space without any explicit reconstruction pressure (LeCun, 2022; Hansen et al., 2024; Zhou et al., 2025a; Sobal et al., 2025; Assran et al., 2025; Maes et al., 2026). This grants them freedom to discard fine-grained information that would be difficult for the world model to predict, and makes them lightweight and efficient for planning (Van Assel et al., 2025; Maes et al., 2026). However, that same freedom creates the risk of representation collapse, requiring dedicated and sometimes poorly understood mechanisms to mitigate it (Grill et al., 2020; Bardes et al., 2022; Balestriero & LeCun, 2025; Maes et al., 2026). Furthermore, discarding fine-grained information about observations can hinder their ability to leverage VLMs as reward models for RL, planning, and policy evaluation (Pendharkar, 2026; Navasardyan et al., 2026).

Motivated by the complementary strengths of generative and non-generative approaches, in this work we investigate whether it is possible to get the best of both worlds: a latent space that makes dynamics modeling easy, and yet remains informative enough to support accurate reconstruction of observations. To answer this question, we consider several families of temporal objectives for representation learning, including latent future prediction, pixel future prediction, contrastive predictive coding, and inverse dynamics (Sermanet et al., 2018; van den Oord et al., 2018; Nachum & Yang, 2021; Hansen et al., 2024), that use action-labelled videos to encourage useful temporal structure in the latent space. We carry out controlled experiments where we fine-tune a generalpurpose visual autoencoder (Black Forest Labs, 2024) with temporal and reconstruction objectives, and then train a video world model making predictions in the learned latent space. We then measure the quality of the representations by evaluating the downstream performance of the video world model, measuring agreement of its predictions with ground truth observations and its policy evaluation ability when paired with a VLM judge. This allows us to explore the impact of different temporal objectives, decoder architectures, and other design choices for downstream world model performance, evaluated in and out of distribution, over long prediction horizons, and varying the inference budget. This also leads to a practical recipe that outperforms the standard practice at equal training budget in training and inference cost, agreement with ground truth and policy evaluation ability. The gains are more pronounced in challenging evaluation conditions, like out of distribution, long horizons, and sampling with few steps.

Our main contributions are the following.

• Temporal structure is useful for world models. First, we show in a synthetic environment that learning latents that are both predictable and informative can be achieved with a combination of a reconstruction and a temporal objective, while each individually fails (Section 4). Moving to robot manipulation videos, we find that including temporal objectives in autoencoder fine-tuning leads to improved world model performance, controlling for the reconstruction error of the autoencoder (Section 5.1). The most effective objectives are those that encourage predictability in latent space directly, like latent prediction and contrastive prediction, while predicting the future in pixel space barely helps. We also show that the most effective objectives induce latents with lower intrinsic dimensionality and weaker high frequency components.

• Past-conditioned decoding improves the information-predictability frontier. We find that, with temporal objectives, the training budget to reach a reconstruction target in autoencoder fine-tuning can be much larger than with reconstruction alone. To characterize the apparent trade-off between predictability of the latents and information they retain, we vary the relative importance of the reconstruction and temporal objectives in autoencoder fine-tuning (Section 5.2). We then show that a simple architectural change, conditioning the decoder on a past observation, improves this trade-off. At an equal training budget, the resulting latents can carry as much information as those of reconstruction alone with the ordinary decoder, while being more predictable.

• A practical recipe to improve video world models. We combine the past-conditioned decoder with temporal objectives like latent prediction or inverse dynamics. Both objectives increase the predictability of the latents, with a 33-80% reduction in latent MSE. We show that compared to the standard practice, at equal training budget, world models trained on the latents of our recipe obtain over 7x faster training, 5x fewer sampling steps at matched generation quality, a 15–24% improvement in pixel and perceptual error for out-of-distribution and long-horizon extrapolation, and up to a 0.45 increase in Pearson correlation of policy evaluation between the world model and the real world (Section 5.3). Both the past-conditioned decoder and the objectives account for part of these gains. Finally, we verify the robustness of these results to changes in the initial autoencoder, world model size, and fine-tuning recipe (Appendix G.9).

## 2 RELATED WORK

Video world models. Video world models generate future frames conditioned on past frames and actions, autoregressively over discrete visual tokens (Bruce et al., 2024; Wu et al., 2024) or increasingly, with diffusion (Alonso et al., 2024; Zhu et al., 2025; NVIDIA, 2025; Quevedo et al., 2026). Apart from a few pixel-space models (Yang et al., 2024; Alonso et al., 2024), they predict in the latent space of an autoencoder trained to reconstruct, with pixel, perceptual and adversarial losses, and sometimes a small KL penalty (Rombach et al., 2022; Yu et al., 2024; Yang et al., 2025; Kong et al., 2024; Valevski et al., 2025; NVIDIA, 2025; Wan Team, 2025; Guo et al., 2026b). While some recent works have adopted spatiotemporal autoencoders (Yu et al., 2024; Kong et al., 2024; NVIDIA, 2025; Wan Team, 2025), time mostly serves as an additional compression axis, and sometimes enters the definition of the reconstruction objective in pixel space, like with clip-level discriminators (Yu et al., 2024; Yang et al., 2025; Wan Team, 2025; Russell et al., 2025), wavelet sub-bands (HaCohen et al., 2024; Li et al., 2025), or an optical-flow loss between input and reconstruction (NVIDIA, 2025). In contrast, we focus on temporal objectives that affect the relationship between past and future representations, acting in pixel space or in latent space directly.

Reconstruction-free world models. Another line of work learns predictable latents without ever reconstructing observations: the encoder is trained jointly with a latent dynamics model to predict future latents, rewards, or values (Hansen et al., 2024; Sobal et al., 2025; Maes et al., 2026; Okada & Taniguchi, 2021; Nguyen et al., 2021; Deng et al., 2022). This lets the latents discard content that is hard to predict or irrelevant to the task (Nguyen et al., 2021; Balestriero & LeCun, 2024; Van Assel et al., 2025), but also makes these models susceptible to total or partial collapse, whereby latent prediction can favour slow or easily predictable features and discard task-relevant ones (Sobal et al., 2022; Littwin et al., 2024; Pendharkar, 2026; Karmann et al., 2026). Consequently, several of these works report that decoders trained on the latents a posteriori lose content: background texture (Okada & Taniguchi, 2021; Nguyen et al., 2021), the orientation of the end effector in long predicted rollouts (Maes et al., 2026), or the foreground configuration when the background is static (Karmann et al., 2026). This limits their ability to leverage generalist VLMs as a judge for policy training, policy evaluation, and planning with the world model (Tseng et al., 2025; Jeon et al., 2026; Sharma et al., 2026; Navasardyan et al., 2026). In contrast, we keep a reconstruction objective which a VLM judge needs, and add temporal objectives for predictability.

Latent space design for generative models. In image generation, autoencoders that reconstruct better do not necessarily support better generative models (Yao et al., 2025; 2026). Proposed remedies regularize the latent space (Kouzelis et al., 2025; Skorokhodov et al., 2025), align it with the features of a pretrained semantic encoder (Yao et al., 2025), or use those features directly as the latent space (Zheng et al., 2026). Video generation and video world models have also adopted the semantic remedies, with frozen image or video encoders (Guo et al., 2026a; Xie et al., 2026; Zhang et al., 2026b; Hu et al., 2026) or alignment losses on the video tokenizer (Hu et al., 2023; Russell et al., 2025; FSVideo Team, 2026; Wang et al., 2026). Closest to our setting, Nilaksh et al. (2026) compare frozen, off-the-shelf reconstruction and semantic encoders as latent spaces for action-conditioned video world models on Bridge V2. In the semantic approaches the structure of the latent space is inherited from a pretrained encoder, so the effect of its training objective is not separated from that of its data, architecture, and scale. We instead keep the autoencoder fixed and vary only its objective. Furthermore, we study objectives that relate past and future latents rather than spatial regularization or alignment with a pretrained encoder; our own experiments fine-tuning a DINOv2 encoder indicate that the two are complementary (Appendix G.9).

Temporal objectives on the autoencoder. In model-based reinforcement learning, the encoder is trained jointly with a latent dynamics model, which makes its latents predictable (Hafner et al., 2018; 2025a). Video world models, Dreamer 4 included (Hafner et al., 2025b), instead train their autoencoder separately, usually for reconstruction alone, and then freeze it. A few works add a temporal term to the autoencoder objective for video generation: future-frame prediction by the decoder (Zhao et al., 2026), slowness and uniformity losses for video world models in deterministic synthetic environments (Xia et al., 2026), a motion-consistency loss on the robot-arm latents of a disentangled video autoencoder for robot video world models (Luo et al., 2026), and a local spatiotemporal correlation loss for text-to-video generation (Liu et al., 2026). Zhao et al. (2026) observe that their objective costs reconstruction and partly recover it by fine-tuning the decoder; however, to our knowledge, none of these works compares families of temporal objectives, none uses actions to shape the representations, and none removes the reconstruction cost: with a past-conditioned decoder, our temporally structured latents can reconstruct better than those of a standard autoencoder trained for reconstruction alone. Finally, Yeom et al. (2026) consider objectives similar to ours, but their goal is finding features from which an inverse-dynamics probe recovers the executed action, in simulation, while we explore latents that can improve a video world model on real-world data.

## 3 BACKGROUND

Latent video world models. We consider trajectories of frames $x _ { 1 : T }$ and actions $a _ { 1 : T }$ . An encoder E maps each frame independently to a latent $z _ { t } = E ( x _ { t } )$ , and a decoder D maps it back to a frame $\hat { x } _ { t } = D ( z _ { t } )$ . We use a reconstruction loss $\mathcal { L } _ { \mathrm { r e c } } ( x _ { t } , \hat { x } _ { t } )$ that combines a pixel and a perceptual term (Zhang et al., 2018), and we also employ a lightly weighted KL penalty (Kingma & Welling, 2014) in some experiments. A world model $p _ { \theta } { \left( z _ { t + 1 } \mid z _ { 1 : t } , a _ { 1 : t } \right) }$ is learned on the frozen latent space, and its generations are decoded by D.

Temporal objectives for representation learning. We will consider, across our experiments, temporal objectives $\mathcal { L } _ { \mathrm { t e m p } }$ based on transitions $( x _ { t } , a _ { t } , x _ { t + 1 } )$ , using an auxiliary predictor which is discarded for world model training.

• latent prediction: predict $z _ { t + 1 }$ from $z _ { t }$ and $a _ { t }$ , with a stop-gradient target (Bardes et al., 2024; Hansen et al., 2024; Sobal et al., 2025);

• pixel prediction: predict $x _ { t + 1 }$ from $z _ { t }$ and $a _ { t }$ (Oh et al., 2015; Finn et al., 2016);

• contrastive predictive coding: train an energy-based model of $z _ { t + 1 }$ given $z _ { t }$ and $a _ { t }$ with an InfoNCE loss (van den Oord et al., 2018; Nachum & Yang, 2021);

• inverse dynamics: predict $a _ { t }$ from $z _ { t }$ and $z _ { t + 1 }$ (Agrawal et al., 2016; Pathak et al., 2017).

We will consider a combined objective:

$$
\begin{array} { r } { \mathcal { L } = \left( 1 - \lambda \right) \mathcal { L } _ { \mathrm { r e c } } + \lambda \mathcal { L } _ { \mathrm { t e m p } } , } \end{array}\tag{1}
$$

where $\lambda \in [ 0 , 1 ]$ sets the weight of the temporal term and $\lambda = 0$ recovers reconstruction alone.

Diffusion-forcing world model. We use diffusion forcing (Chen et al., 2024) to learn $p _ { \theta } \colon$ we train a causal video diffusion model with independent noise levels for each frame (Appendix D), and we do inference autoregressively with a pyramid schedule (Appendix E) using the DDIM sampler (Song et al., 2021). We use the space-time factorized diffusion transformer (DiT) (Peebles & Xie, 2023) of Quevedo et al. (2026), which alternates bidirectional attention within a frame with causal attention across frames (Appendix C).

## 4 LATENTS CAN BE BOTH PREDICTABLE AND INFORMATIVE

We begin our investigation by considering data from a simple synthetic environment. We show that, by combining a reconstruction and a temporal objective, it is possible to learn latents that remain informative about the observations while being easy to predict. By contrast, using each objective individually either fails to be predictable, or fails to be informative (Figure 2).

Synthetic environment. We consider observations $x _ { t } = ( s _ { t } , \xi _ { t } ) \in \mathbb { R } ^ { 1 0 2 }$ that concatenate a twodimensional signal $s _ { t }$ and a 100-dimensional vector of distractors $\xi _ { t } .$ , which are sampled from a standard normal independently at every step. The signal is sampled at time 0 from a standard normal and at each time step, given a random action $a _ { t } \in \mathbb { R } ^ { 2 }$ , it evolves deterministically as $s _ { t + 1 } = s _ { t } + a _ { t }$

Experiment. An encoder E learns to map the observation to a latent $z _ { t } ~ = ~ E ( x _ { t } ) ~ \in ~ \mathbb { R } ^ { d }$ by optimizing Equation 1, with the reconstruction objective of a variational autoencoder (Kingma & Welling, 2014) and a temporal contrastive objective (van den Oord et al., 2018; Nachum & Yang, 2021) as the temporal term. On top of the frozen encoder, we train three modules: a world model that predicts the next latent from the current latent and action, a probe that reads the signal from the latent, and a decoder that reconstructs the observation. On held-out data, we measure the errors of the reconstruction, of the signal, and of the next latent. Each error is divided by the variance of its target, so that predicting the mean gives an error of 1. As a reference, the same modules are trained directly on the raw observation. In Figure 2, we compare λ = 0 (reconstruction), λ = 1 (temporal), and λ = 0.5 (both) at latent size $d = 1 0 0$ . Details on the data, networks, training and metrics, as well as the errors at more latent sizes and weights, can be found in Appendix A.

The combined objective learns predictable and informative latents. The reconstruction objective is known to favour the leading principal components of the observation (Balestriero & LeCun, 2024); since every coordinate has approximately unit variance, it recovers the signal only when the latent is nearly as large as the observation, and its latent is no easier to predict than the observation itself. The temporal contrastive objective can be minimized by capturing nothing more than the signal, since it is the only predictable part in the observation. Empirically, this is almost what happens: the latent reconstructs very poorly, but its signal and next-latent errors are two orders of magnitude below those of reconstruction. The combined objective avoids both failures: at d = 100 it reconstructs nearly as well as reconstruction alone, but its next-latent error is almost 50 times lower, and it recovers the signal at every latent size. This shows that it is possible, in this simple setting, to learn latents that are both predictable and informative. Next, we explore the use of similar temporal objectives on real-world robot manipulation videos.

## 5 EXPERIMENTS WITH ROBOT MANIPULATION VIDEO DATA

In the previous section we have shown, using a synthetic dataset, that it is largely unnecessary to choose between predictable and informative latents: latents can be both at the same time, making learning world models easy while still enabling reconstruction of predicted future observations. In this section, we explore whether the same is true with real-world video data, and how to leverage this to learn better video world models.

![](images/ce28ed330dc389978c4fd1acf3fef82c7ec4e3c47d2dbe5cc395ac62d2691abb.jpg)

error relative to raw observations  
![](images/53af750e38b2db1c3ac2615b4bdb863e6fc211b480bed5bfc60b955afd63bb7e.jpg)  
Figure 2: Latents can be predictable and informative in a synthetic setting. Left: Synthetic observations have a predictable component (signal, the dot) and an unpredictable noise component (the static). An encoder E is trained with reconstruction, temporal contrastive learning or both, and three readouts are then trained on its frozen latents: a world model predicts the next latent from the current latent and action, a probe predicts the signal, and a decoder D reconstructs the observation Right: the error of each readout at latent size 100, divided by the error of the same readout trained on the raw observation; geometric mean and standard error over 3 seeds, lower is better. Reconstruction alone leaves the next latent as hard to predict as the raw observation and reads the signal far less accurately; contrastive learning alone loses the observation; their combination avoids both failures.

Common experiment setting. Experiments in this section use the pipeline of Figure 1. We finetune the pretrained autoencoder of FLUX.1 (Black Forest Labs, 2024) on Bridge V2 (Walke et al., 2023), using a combination of a temporal objective and a reconstruction objective (Equation 1), and then freeze it. Unless stated otherwise, the decoder reconstructs each frame from its own latent, $\hat { x } _ { t } = D ( z _ { t } )$ , as in standard autoencoders; we call it the ordinary decoder, and fine-tuning for reconstruction alone with the ordinary decoder, the standard practice, is our baseline. For the reconstruction loss, we combine the mean absolute pixel error with a perceptual distance (Appendix B). We found using a learned discriminator (Esser et al., 2021) to be unhelpful, likely because of the relatively low resolution of the data, so we do not use it. We set the KL loss weight to 0, but we show in an ablation that the key conclusions remain the same using a nonzero weight (Table 14). The world model is the diffusion-forcing transformer of Section 3, trained on the frozen latent space. We compare its action-conditioned generations with ground truth frames using pixel mean squared error (MSE), LPIPS (Zhang et al., 2018), and SSIM (Wang et al., 2004). We also compute the Fréchet inception and video distances, FID (Heusel et al., 2017) and FVD (Unterthiner et al., 2018), and in one experiment we assess the ability of the world model to do policy evaluation (Quevedo et al., 2026). We vary both the inference horizon (up to six times the training horizon) and the inference budget, and we evaluate both in distribution (with held-out Bridge V2 data) and out of distribution on videos from the AutoEval benchmark (Zhou et al., 2025b), recorded on separate robot setups. In the main-text tables, we report the errors in four evaluation conditions: in distribution (ID), out of distribution (OOD), in distribution with two sampling steps instead of ten (2-step), and at the end of the longest rollouts (long-horizon). Full results with more inference budgets and inference horizons are in Appendix G.2. Every pipeline is trained with three seeds unless stated otherwise. Appendices B to E provide all the details.

Table 1: Temporal structure benefits world models at equal reconstruction quality. With the pipeline of Figure 1, using the ordinary decoder, we compare different objectives for fine-tuning the autoencoder until a target reconstruction error is reached, training the same world model on the learned latents. We report mean and standard error over 3 seeds for the error of world model predictions against ground truth in the four evaluation conditions of Section 5 (pixel $\mathrm { M S E } \times \mathrm { 1 0 ^ { 3 } } .$ bold: best). Updates in thousands; Components: principal components for 95% of the latent variance. Temporal objectives, especially latent and contrastive prediction, improve performance across the board compared to the reconstruction baseline, but they require more autoencoder updates to reach the same reconstruction error.
<table><tr><td rowspan="2"></td><td colspan="2">ID↓</td><td colspan="2">00D↓</td><td colspan="2">2-step ↓</td><td colspan="2">Long-horizon ↓</td><td colspan="2">Autoencoder</td></tr><tr><td>MSE</td><td>LPIPS</td><td>MSE</td><td>LPIPS</td><td>MSE</td><td>LPIPS</td><td>MSE</td><td>LPIPS</td><td>Updates</td><td>Components</td></tr><tr><td rowspan="2">Reconstruction (baseline)</td><td>11.65</td><td>0.164</td><td>33.87</td><td>0.308</td><td>15.95</td><td>0.214</td><td>33.26</td><td>0.287</td><td>4.4</td><td>388</td></tr><tr><td>±0.20</td><td>±0.001</td><td>±1.28</td><td>±0.008</td><td>±1.03</td><td>±0.004</td><td>±1.84</td><td>±0.008</td><td></td><td></td></tr><tr><td rowspan="2">Pixel prediction</td><td>11.43</td><td>0.159</td><td>34.23</td><td>0.300</td><td>14.53</td><td>0.199</td><td>32.63</td><td>0.279</td><td>8.9</td><td>445</td></tr><tr><td>±0.22</td><td>±0.001</td><td>±0.67</td><td>±0.002</td><td>±0.75</td><td>±0.002</td><td>±0.69</td><td>±0.003</td><td></td><td></td></tr><tr><td rowspan="2">Contrastive prediction</td><td>11.17</td><td>0.160</td><td>31.23</td><td>0.303</td><td>12.38</td><td>0.185</td><td>31.50</td><td>0.277</td><td>96.0</td><td>293</td></tr><tr><td>±0.05</td><td>±0.001</td><td>±0.14</td><td>±0.005</td><td>±0.27</td><td>±0.001</td><td>±2.62</td><td>±0.010</td><td></td><td></td></tr><tr><td rowspan="2">Latent prediction</td><td>11.30</td><td>0.153</td><td>29.14</td><td>0.298</td><td>12.36</td><td>0.160</td><td>30.04</td><td>0.262</td><td>156.5</td><td>177</td></tr><tr><td>±0.10</td><td>±0.001</td><td>±0.55</td><td>±0.005</td><td>±0.24</td><td>±0.002</td><td>±0.43</td><td>±0.000</td><td></td><td></td></tr></table>

## 5.1 WORLD MODELS CAN BENEFIT FROM TEMPORAL STRUCTURE

To isolate differences due purely to the latent space geometry from differences in information retention, we first compare the inclusion of temporal objectives with pure reconstruction fine-tuning, at equal reconstruction quality of the autoencoder.

Fine-tuning to equal reconstruction quality. We fine-tune the autoencoder with four objectives: reconstruction alone (baseline), and its combination with latent, pixel, or contrastive future prediction (Section 3), the latter with negatives drawn from other videos. We fine-tune each autoencoder until its reconstruction error reaches a common target, so that every latent supports reconstructions of the same quality and the only difference is the geometry of the latent spaces. We then train the same world model on each of them (Appendix F).

Temporal structure improves the world model, but at higher cost. At equal reconstruction quality, all three temporal objectives improve the world model compared to the baseline (except pixel prediction in out-of-distribution pixel MSE). Latent prediction gives the largest gains, on all but one metric of Table 1, followed by contrastive prediction, while pixel prediction stays closer to the baseline. The temporal objectives improve performance especially in the more challenging evaluation conditions: for example, latent prediction improves MSE by 22.5% with two sampling steps, 14.0% out of distribution, and 9.7% at the end of long rollouts, against 3.0% in distribution. FID and FVD are also slightly improved by latent and contrastive prediction (Table 5), and the advantage persists throughout world model training (Figure 5). Since we are controlling for reconstruction ability, these gains are to be attributed to a more favorable latent space structure, induced by temporal objectives However, this comes at a cost: to reach the reconstruction target, the autoencoder needs about 35 times as many updates with latent prediction compared to reconstruction alone, and about 22 times as many with contrastive prediction. In the next section, we will address this limitation.

## 5.2 IMPROVING THE INFORMATION-PREDICTABILITY FRONTIER

The results of Section 5.1 show that temporal objectives can benefit a world model even with realworld data, but hint at a trade-off between the information a latent retains and its predictability. In this section, we characterize this trade-off and show it can be improved with a simple architectural modification to the decoder.

Past-conditioned decoder. In Table 1 we see that the best performing temporal objectives, which encourage the latents to be easily predictable, tend to induce a mild form of dimensional collapse: their latents need fewer principal components to explain 95% of their variance compared to reconstruction training. This tendency competes with reconstruction, with a balance set by λ. To ease the tension, we propose to condition the decoder on a past frame. Consecutive frames share most of their content, so the latent only has to carry what the past frame does not already provide. We implement this by adding cross-attention layers to the pretrained decoder, attending to the past frame, and initialized so that the decoder starts as the pretrained one. In world-model rollouts, the decoder is conditioned on the initial context frame. During training, several choices are possible; in the reported experiments we condition on the previous frame, $\hat { x } _ { t + 1 } = D ( z _ { t + 1 } , x _ { t } )$ (Figure 1; Appendix C). Similar ideas have appeared in prior work (Wu et al., 2023; Micheli et al., 2024; Wu et al., 2024; Tian et al., 2024; Fan et al., 2025; FSVideo Team, 2026; Li et al., 2021), where conditioning serves compression, reconstruction quality, or the dynamics model's focus on what changes; here, we use it to improve the information-predictability frontier, as we will demonstrate shortly.

Tracing and improving the information-predictability frontier. We fine-tune the pretrained autoencoder on pairs of consecutive frames with latent prediction, the objective with the largest gains in Section 5.1, at weights $\lambda \in \{ 0 , 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \}$ , with the ordinary and with the past-conditioned decoder. Contrary to Section 5.1, we use the same training budget for each, 80,000 autoencoder updates. We measure the information retained by the latents as the reconstruction error on held-out frame pairs. We also measure their predictability in two ways: as the fraction of the latent variance not explained by a linear action-conditioned future predictor fitted on frozen latents (Figure 3), and as the error of the nonlinear prediction head used by the temporal objective at the end of training (Appendix G.3). Both measures agree qualitatively: raising λ trades reconstruction for predictability. With the ordinary decoder, from λ = 0 to 0.7, the linearly unpredictable share falls from 0.21 to 0.07. while the reconstruction error rises to 54% above that of reconstruction alone (Figure 3). Introducing the past-conditioned decoder shifts the frontier, lowering the reconstruction error by 17 to 25% at every weight: at $\lambda = 0 . 5$ , it achieves the same reconstruction error as reconstruction alone with the ordinary decoder, with about half the linearly unpredictable share of the latent variance. Thus, the past-conditioned decoder enables achieving a similar best of both worlds situation, with latents that are both predictable and informative, as in the synthetic setting of Section 4. Interestingly, as the weight of the temporal objective increases, and with it the pressure on the latents to be predictable, the decoder relies more and more on the past frame to achieve good reconstructions, as seen in the right panel of Figure 3.

## 5.3 A PRACTICAL RECIPE TO IMPROVE WORLD MODEL TRAINING

In this section, we combine insights from the previous experiments and show that, at equal training budget, the past-conditioned decoder combined with a temporal objective gives better world models than reconstruction alone.

Fine-tuning at equal training budget. Like in Section 5.2, we fine-tune every autoencoder for 80,000 updates on pairs of consecutive frames, and then train the same world model on frozen latents for 100,000 updates (Table 3). For autoencoder fine-tuning, we consider augmenting the reconstruction objective either with latent prediction, the best objective from Section 5.1, or with an inverse dynamics objective (Section 3). Both use the past-conditioned decoder at $\lambda = 0 . 7$ , the largest weight of Section 5.2. We compare against the autoencoder fine-tuned with reconstruction alone and the past-conditioned decoder, to isolate gains from temporal structure at equal training budget, as well as reconstruction fine-tuning with the ordinary decoder (baseline), to measure the total gains from our contributions. For reference, we also include the pretrained autoencoder without fine-tuning. In the following, we refer to the world models trained on latents obtained with our recipe (a temporal objective and past-conditioned decoder) as the temporal world models.

Better generations, especially out of distribution and at long horizons. At equal training budget, the past-conditioned decoder with either temporal objective improves pixel MSE, LPIPS, and SSIM over the baseline, both in and out of distribution, and over long horizons (Table 2, Table 6). The gains are largest in the more challenging evaluation conditions: out of distribution and at the end of long rollouts, pixel MSE and LPIPS fall by 15 to 24%. There, fine-tuning for reconstruction alone does not lower the pixel MSE: the pretrained autoencoder without fine-tuning is slightly better than the baseline. The past-conditioned decoder alone already improves on the baseline in every condition;

![](images/dd61a0881bd9ee0415dbded61b104d976d6b0fd902a69558a6c768dd34c00a45.jpg)  
unpredictable share of latent variance (lower is better)

![](images/fcf1726a4754bce46efc132e6075ea72f21c1231c033af9f4cf727e70972f22c.jpg)  
Figure 3: The information-predictability frontier and the decoder's reliance on the past frame. Left: reconstruction error against the unpredictable share of the latent variance, for autoencoders fine-tuned for 80,000 updates with latent prediction at weight λ, with the ordinary decoder or the past-conditioned decoder. Points are means over two training seeds at λ = 0 and 0.3 (faint: individual seeds) and one seed elsewhere. Raising λ trades reconstruction for predictability, and the pastconditioned decoder improves the frontier at every weight. Right: reconstruction error of each past-conditioned autoencoder with its past-frame pathway disabled (Appendix G.4), divided by its error with the correct past frame (log scale); same seeds as on the left. As the pressure from the temporal objective on the latent space increases, the decoder relies more on the past frame for accurate reconstructions.

the temporal objectives add further gains, especially in the challenging conditions. The two temporal objectives perform similarly; inverse dynamics has a lead in distribution and on perceptual metrics out of distribution, while latent prediction does better over long horizons (every metric in Table 6; rollouts in Figure 9 and Appendix G.6). Distribution metrics (FID, FVD) also improve with both temporal objectives, although in the case of latent prediction this requires a sufficiently small weight λ (Figure 24).

Faster world-model training and cheaper inference. Latents trained with a temporal objective and a past-conditioned decoder support faster world model training. With latent prediction and inverse dynamics, the world model reaches the final validation LPIPS of the baseline 7.6 and 8.8 times sooner, against 1.8 times with the decoder alone; in SSIM, the speed-up is 15 and 13 times, and in pixel MSE 1.2 and 1.8 times (Figure 6). Additionally, the temporal world models can do inference with far fewer DDIM steps without sacrificing performance. With 2 sampling steps, both generate better than the baseline does with 10 in LPIPS and SSIM, and inverse dynamics also in pixel MSE (Table 2, Figure 6). This is not the case with the past-conditioned decoder alone.

Improvements in policy evaluation ability. Going beyond visual quality, we measure the ability of the world models to evaluate robot policies by following the protocol of WorldGym (Quevedo et al., 2026). Three generalist robot policies, OpenVLA (Kim et al., 2025), Octo (Ghosh et al., 2024) and RT-1-X (O'Neill et al., 2024), act inside each of the five world models of Table 2 on 17 Bridge tasks and a vision-language model judges task success. We then measure the correlation between simulated and real success rates across policies and tasks (Appendix G.5). We test two inference budgets, 10 and 2 DDIM steps, for all world models. In all cases, the temporal world models obtain better agreement with the real world than the baseline, 0.62 and 0.57 Pearson correlation against 0.51. The past-conditioned decoder alone does not improve policy evaluation: the gains come from the temporal objectives. With two steps, the correlation of every reconstruction-only world model falls to 0.15–0.21, while latent prediction keeps 0.67 and inverse dynamics 0.51: at this budget, policy evaluation remains viable only with the temporal objectives (Figure 8; per-policy correlations are in Table 8).

Table 2: Past-conditioned decoding and temporal objectives improve world models at equal training budget. With the pipeline of Figure 1, we compare decoders and objectives for fine-tuning the autoencoder for a fixed number of updates, training the same world model on the learned latents. The past-conditioned reconstruction row against the baseline gives the effect of the decoder; the two objective rows against that row give the effect of the temporal objective. We report mean and standard error over 3 seeds (pixel MSE ×103; bold: best), except for policy evaluation in WorldGym (one training seed, bootstrap standard error over tasks). ID, OOD, 2-step and long-horizon as in Section 5; latent MSE: the error of the world model's one-step forecast of the next latent (Section 5.3). Our recipe improves every metric compared to the baseline; both the past-conditioned decoder and the temporal objectives account for some of the improvements.
<table><tr><td rowspan="2"></td><td colspan="2">ID↓</td><td colspan="2">O0D↓</td><td colspan="2">2-step ↓</td><td colspan="2">Long-horizon ↓</td><td colspan="2">Policy eval. ↑</td><td>Predictability ↓</td></tr><tr><td></td><td>MSE LPIPS</td><td></td><td>MSE LPIPS</td><td>MSE</td><td>LPIPS</td><td>MSE</td><td>LPIPS</td><td>10 steps 2 steps</td><td></td><td>Latent MSE</td></tr><tr><td>Ordinary decoder</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction (baseline) 14.34</td><td></td><td>0.194</td><td>36.15</td><td>0.302</td><td>18.35</td><td>0.232</td><td>47.93</td><td>0.394</td><td>0.51</td><td>0.21</td><td>0.232</td></tr><tr><td rowspan="2">Pretrained, no fine-tuning</td><td>±0.16</td><td>±0.002</td><td>±0.98</td><td>±0.003</td><td>±0.65</td><td>±0.004</td><td>±1.51</td><td>±0.002</td><td>±0.10</td><td>±0.07</td><td>±0.008</td></tr><tr><td>14.81</td><td>0.208</td><td>34.81</td><td>0.309</td><td>19.15</td><td>0.250</td><td>46.78</td><td>0.393</td><td>0.57</td><td>0.21</td><td>0.253</td></tr><tr><td></td><td>±0.26</td><td>±0.001</td><td>±1.97</td><td>±0.008</td><td>±0.89</td><td>±0.003</td><td>±0.83</td><td>±0.003</td><td>±0.07</td><td>±0.09</td><td>±0.001</td></tr><tr><td>Past-conditioned decoder</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Reconstruction</td><td>14.00</td><td>0.176</td><td>31.45</td><td>0.270</td><td>16.99</td><td>0.198</td><td>44.61</td><td>0.366</td><td>0.49</td><td>0.15</td><td>0.212</td></tr><tr><td>±0.14</td><td>±0.002</td><td>±0.56</td><td>±0.006</td><td>±0.64</td><td>±0.003</td><td>±1.13</td><td>±0.006</td><td>±0.08</td><td>±0.07</td><td>±0.008</td></tr><tr><td rowspan="2">Inverse dynamics</td><td>13.07</td><td>0.160</td><td>27.92</td><td>0.236</td><td>13.44</td><td>0.162</td><td>39.39</td><td>0.333</td><td>0.57</td><td>0.51</td><td>0.155</td></tr><tr><td>±0.24</td><td>±0.001</td><td>±1.17</td><td>±0.005</td><td>±0.28</td><td>±0.001</td><td>±2.01</td><td>±0.005</td><td>±0.07</td><td>±0.09</td><td>±0.001</td></tr><tr><td rowspan="2">Latent prediction</td><td>14.28</td><td>0.169</td><td>27.54</td><td>0.257</td><td>15.64</td><td>0.177</td><td>39.08</td><td>0.331</td><td>0.62</td><td>0.67</td><td>0.046</td></tr><tr><td>±0.18</td><td>±0.001</td><td>±0.67</td><td>±0.002</td><td>±0.26</td><td>±0.001</td><td>±1.08</td><td>±0.007</td><td>±0.07</td><td>±0.07</td><td>±0.001</td></tr></table>

The gains come with more predictable latents. The latent MSE, the error of the world model's one-step forecast of the next latent and a direct measure of how predictable each latent space is, falls by 80% with latent prediction and by 33% with inverse dynamics, although only the former trains it directly, against 9% with the decoder alone (Table 2). Their latents are also lower dimensional, spatially smoother and straighter over time than those of reconstruction alone (Appendix G.8).

Robustness to design choices. We test the robustness of the gains due to the temporal objectives to various choices and hyperparameters in the pipeline, including training time, KL penalty, world model size, choice of encoder, and the weight of the temporal objective (Appendices G.7 and G.9). Each ablation compares a temporal objective with reconstruction alone, both with the past-conditioned decoder. Doubling the training time and adding a KL penalty both preserve the gains in the challenging evaluation conditions. Changing the weight of the latent prediction objective leaves most conclusions intact, but long-rollout performance only improves with a sufficiently large weight, while FID and FVD only improve if the weight is not too large. With a larger world model, the gains out of distribution and with 2 sampling steps remain, but long-rollout gains shrink. Finally, we test replacing the FLUX autoencoder with DINOv2 (Oquab et al., 2024), a semantic encoder with a high-dimensional latent space, and the decoder of RAE (Zheng et al., 2026). This leads to similar gains when the weight of the temporal objective is small enough, except with few sampling steps where the gains shrink compared to the FLUX latent space.

## 6 DISCUSSION

Our results show that temporal structure in the latent space of video world models yields practical benefits for robot manipulation. World models trained on temporally structured latents generalize better out of distribution, extrapolate better to horizons longer than those seen during training, and give policy evaluations that agree better with the real world. They are also cheaper: they match the generation quality of the baseline with five times fewer sampling steps, and its perceptual quality with over seven times less training. Temporal structure should therefore be treated as an important design criterion for the latent space of video world models.

Furthermore, our results show that two lines of work can enter a fruitful dialogue: video world models meant as simulators, and reconstruction-free world models meant for latent planning. Indeed, a world model can keep generating realistic rollouts, which powerful VLM judges can score and policies can act in, while its latents gain the temporal structure that makes them easy to predict from the past. On complex real-world data, a simple hierarchy helped to reconcile these two goals: the past-conditioned decoder takes the details that do not change over time from a past frame, so the latents only need to carry what changes. In the same spirit, it would be interesting to explore learning deeper hierarchies of dynamical representations. Such hierarchies could enable flexible inference that trades temporal and state resolution for a longer horizon at a fixed computational budget.

Limitations. Our experiments cover synthetic data and videos of robot manipulation. A natural next step is to extend them to other domains where video world models are used, such as egocentric video, navigation and games. We focused on 2D autoencoders that only compress along the spatial axes, for simplicity. Spatiotemporal autoencoders also compress along the time axis; this is different from, and complementary to, our temporal objectives, and could be considered in future work. Finally, we used temporal objectives to fine-tune general purpose autoencoders rather than train them from scratch Pretraining the autoencoders on large video corpora, potentially using latent actions to leverage raw video (Schmidt & Jiang, 2023; Bruce et al., 2024; Garrido et al., 2026), could enable further benefits.

## AI USE STATEMENT

In this work, we used generative AI tools for method implementation: writing and editing the research code, including the code that produces the figures and tables. We have not used generative AI tools for theoretical or conceptual development, hypotheses, methodology or experiment design, syntheticdata generation, dataset cleaning or reformatting, qualitative analysis, or interpretation of results, and mathematical claims or proofs and translation are not applicable to this work. Additionally, we used generative AI tools for literature search and citation checking, for drafting parts of the appendix, and for editing the manuscript for readability. A vision-language model, Claude Opus 5.5, is also a component of one evaluation: it judges task success in the policy evaluation of Section 5.3 (Appendix G.5). We have reviewed all AI-assisted work: AI-assisted code was verified through tests and experiment-level checks, citations were checked against the cited works, and every edit to the manuscript was reviewed by the authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Section 3 and Appendices B to E specify the objectives, architectures, training and evaluation of the video experiments, Appendix F and Table 3 give the settings of the two main comparisons, and Appendix A specifies the synthetic study, including its data. The source code for fine-tuning the autoencoders, training the world models and evaluating them is available at https : //github. com/mattia-scardecchia/predictable-latents.

## ACKNOWLEDGMENTS

This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise. We gratefully acknowledge use of the research computing resources of the Empire AI Consortium, Inc, with support from Empire State Development of the State of New York, the Simons Foundation, and the Secunda Family Foundation.

## REFERENCES

Pulkit Agrawal, Ashvin V. Nair, Pieter Abbeel, Jitendra Malik, and Sergey Levine. Learning to Poke by Poking: Experiential Learning of Intuitive Physics. In Advances in Neural Information Processing Systems, volume 29, 2016.

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for World Modeling: Visual Details Matter in Atari. In Advances in Neural Information Processing Systems, volume 37, pp. 58757–58791, 2024.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, June 2025. URL https://arxiv.org/abs/2506.09985.

Randall Balestriero and Yann LeCun. How Learning by Reconstruction Produces Uninformative Features For Perception. In International Conference on Machine Learning, pp. 2566–2585, 2024.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics, November 2025. URL https://arxiv.org/abs/2511.08544.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. In International Conference on Learning Representations, 2022. doi: 10.48550/arXiv.2105.04906.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting Feature Prediction for Learning Visual Representations from Video. Transactions on Machine Learning Research, 2024. URL https : / /openreview . net/forum?id=QaCCuDfBk2.

Jai Bardhan, Patrik Drozdík, Josef Sivic, and Vladimir Petrik. Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning. In European Conference on Computer Vision, 2026.

Black Forest Labs. FLUX. GitHub repository, 2024. URL https://github.com/ black-forest-labs/flux. Accessed 2026-09-17.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563–22575, 2023. doi: 10.48550/arXiv.2304.08818.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative Interactive Environments. In International Conference on Machine Learning, 2024. doi: 10.48550/arXiv.2402.15391.

João Carreira and Andrew Zisserman. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 6299–6308, 2017.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. In Advances in Neural Information Processing Systems, volume 37, pp. 24081–24125, 2024. URL https://openreview.net/forum?id=yDo1ynArjj.

Sixiang Chen, Jiaming Liu, Jixian Wu, Yichen Guo, Tinghao Wang, Siyuan Qian, Hao Chen, Jiajun Cao, Jian Tang, and Shanghang Zhang. Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning, 2026. URL ht tps : //arxiv.org/abs/2608.24885.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision Transformers Need Registers. In International Conference on Learning Representations, 2024.

Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A Universe in a Transformer, October 2024. URL https://oasis-model.github.io/. Accessed 2026-04-11.

Fei Deng, Ingook Jang, and Sungjin Ahn. DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations. In International Conference on Machine Learning, pp. 4956–4975, 2022.

Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, and Jonathan Tompson. Video Language Planning. In International Conference on Learning Representations, 2024. doi: 10.48550/arXiv.2310.10625.

Peter Dürr, Mireille El Gheche, Guilherme Jorge Maeda, Nobuhiko Mukai, Naoya Takahashi, Stefan Heusser, Hamdi Sahloul, Yamen Saraiji, Pavel Adodin, Yin Bi, Sam Blakeman, Christian Conti, Dunai Fuentes Hitos, Yunpu Hu, Farshad Khadivar, Raphaela Kreiser, Luz Martinez, Fabian Schilling, Ricardo Tapiador Morales, Guillem Torrente, Mario Ynocente Castro, Lison Abecassis, Alberto Giammarino, Yu-Ting Huang, Yannik Nagel, Andrea Scotti, Alexander Sigrist, Tiago Silva, Etienne Walther, Jengyan Wong, Bilan Yang, Asude Aydin, Divij Grover, Apurv Saha, Valentina Cavinato, Takekazu Kakinuma, Taishi Kunori, Valentin Monferrato, Stefan Richter, Stefanos Charalambous, Simon Guist, Mads Alber Kuhlmann-Jorgensen, Lorenzo Miele, Agis Politis, Mattia Scardecchia, Hiroaki Kitano, Peter R. Wurman, Peter Stone, and Michael Spranger. Outplaying elite table tennis players with an autonomous robot. Nature, 652(8111):886–891, April 2026. ISSN 1476-4687. doi: 10.1038/s41586-026-10338-5.

Patrick Esser, Robin Rombach, and Björn Ommer. Taming Transformers for High-Resolution Image Synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12873–12883, 2021.

Xiang Fan, Xiaohang Sun, Kushan Thakkar, Zhu Liu, Vimal Bhat, Ranjay Krishna, and Xiang Hao. RefTok: Reference-Based Tokenization for Video Generation, 2025. URL https : //arxiv. org/abs/2507.02862.

Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised Learning for Physical Interaction through Video Prediction. In Advances in Neural Information Processing Systems, volume 29, 2016.

FSVideo Team. FSVideo: Fast Speed Video Diffusion Model in a Highly-Compressed Latent Space, 2026.URLhttps://arxiv.org/abs/2602.02092.

Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, and Michael Rabbat. Learning Latent Action World Models In The Wild. In International Conference on Machine Learning, 2026. doi: 10.48550/arXiv.2601.05230.

Gemini Robotics Team. Evaluating Gemini Robotics Policies in a Veo World Simulator, 2025. URL https://arxiv.org/abs/2512.10675.

Dibya Ghosh, Homer Rich Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Quan Vuong, Ted Xiao, Pannag R. Sanketi, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An Open-Source Generalist Robot Policy. In Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.090.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. In Advances in Neural Information Processing Systems, volume 33, 2020. doi: 10.48550/arXiv.2006.07733.

Minghui Guo, Shengqiong Wu, and Hao Fei. V-RAE: Rethinking Video Latent Spaces for Generation, August 2026a.URL https://arxiv.org/abs/2608.13556.

Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-World: A Controllable Generative World Model for Robot Manipulation. In International Conference on Learning Representations, 2026b.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. LTX-Video: Realtime Video Latent Diffusion, 2024.URLhttps://arxiv.org/abs/2501.00103.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning Latent Dynamics for Planning from Pixels, 2018. URL https : //arxiv.org/abs/1811.04551.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering Diverse Control Tasks through World Models. Nature, 640(8059):647–653, 2025a. doi: 10.1038/s41586-025-08744-2.

Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training Agents Inside of Scalable World Models,September 2025b.URL https://arxiv.org/abs/2509.24527.

Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, Robust World Models for Continuous Control. In International Conference on Learning Representations, 2024. doi: 10.48550/arXiv.2310.16828.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL https : //openreview . net/forum?id=qw8AKxfYbI.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A Generative World Model for Autonomous Driving, September 2023.URLhttps://arxiv.org/abs/2309.17080.

Anthony Hu, Václav Volhejn, Adrien Ramanana Rahary, Chris Mulder, Aditya Makkar, Alyx Liao, Amélie Royer, Manu Orsini, Adam Jelley, Eloi Alonso, Florian Laurent, Fredrik Norén, James Swingos, Jan Hünermann, Kent Rollins, Lucas Hosseini, Matthieu Le Cauchois, Maxim Peter, Pim de Witte, Tim Brown, Vincent Micheli, Moritz Böhle, Gabriel de Marmiesse, Viktoriia Sharmanska, Lucia Specia, Michael Black, and Patrick Pérez. Multiplayer Interactive World Models with Representation Autoencoders, July 2026. URL https://arxiv.org/abs/2607.05352.

Junchao Huang, Ziyang Ye, Xinting Hu, Tianyu He, Guiyu Zhang, Shaoshuai Shi, Jiang Bian, and Li Jiang. LIVE: Long-horizon Interactive Video World Modeling. In International Conference on Machine Learning, 2026a.

Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2World: Crafting Video Diffusion Models to Interactive World Models. In International Conference on Learning Representations, 2026b. doi: 10.48550/arXiv.2505.14357.

Thomas Hubert, Rishi Mehta, Laurent Sartran, Miklós Z. Horváth, Goran Žužić, Eric Wieser, Aja Huang, Julian Schrittwieser, Yannick Schroecker, Hussain Masoom, Ottavia Bertolli, Tom Zahavy, Amol Mandhane, Jessica Yung, Iuliya Beloshapka, Borja Ibarz, Vivek Veeriah, Lei Yu, Oliver Nash, Paul Lezeau, Salvatore Mercuri, Calle Sönne, Bhavik Mehta, Alex Davies, Daniel Zheng, Fabian Pedregosa, Yin Li, Ingrid von Glehn, Mark Rowland, Samuel Albanie, Ameya Velingker, Simon Schmitt, Edward Lockhart, Edward Hughes, Henryk Michalewski, Nicolas Sonnerat, Demis Hassabis, Pushmeet Kohli, and David Silver. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, 651(8106):607–613, March 2026. ISSN 1476-4687. doi: 10.1038/s41586-025-09833-y.

Allan Jabri, David J. Fleet, and Ting Chen. Scalable Adaptive Computation for Iterative Generation. In International Conference on Machine Learning, pp. 14569–14589, 2023.

Byeongguk Jeon, Seonghyeon Ye, JaeHyeok Doo, Sungdong Kim, Minjoon Seo, Hyungmok Son, and Kimin Lee. RoboWorld: Fast and Reliable Neural Simulators for Generalist Robot Policy Evaluation,2026.URL https://arxiv.org/abs/2607.01060.

Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How Far is Video Generation from World Model: A Physical Law Perspective. In International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2411.02385.

Markus Karmann, Shile Li, Christian Internò, Bruno Andreis, David Klindt, Randall Balestriero, Jindong Gu, Philip Torr, Qi Zhang, Peng-Tao Jiang, Hao Zhang, Bo Li, and Onay Urfalioglu. MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space, September 2026.URL https://arxiv.org/abs/2609.23881.

Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Champion-level drone racing using deep reinforcement learning. Nature, 620 (7976):982–987, August 2023. ISSN 1476-4687. doi: 10.1038/s41586-023-06419-4.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model. In Conference on Robot Learning, pp. 2679–2713, 2025.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In International Conference on Learning Representations, 2014. URL https://arxiv.org/abs/1312.6114.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. HunyuanVideo: A Systematic Framework For Large Video Generative Models, 2024. URL https://arxiv.org/abs/2412.03603.

Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling. In International Conference on Machine Learning, 2025. doi: 10.48550/arXiv.2502.09509.

Yann LeCun. A Path Towards Autonomous Machine Intelligence. OpenReview, 2022. URL https://openreview.net/forum?id=BZ5a1r-kVsf.Version 0.9.2,2022-06-27.

Jiahao Li, Bin Li, and Yan Lu. Deep Contextual Video Compression, 2021. URL ht tps : / / arxiv . org/abs/2109.15047.

Tianhong Li and Kaiming He. Back to Basics: Let Denoising Generative Models Denoise. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36115–36125, 2026.

Zongjian Li, Bin Lin, Yang Ye, Liuhan Chen, Xinhua Cheng, Shenghai Yuan, and Li Yuan. WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17778–17788, 2025.

Etai Littwin, Omid Saremi, Madhu Advani, Vimal Thilak, Preetum Nakkiran, Chen Huang, and Joshua Susskind. How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks. In Advances in Neural Information Processing Systems, volume 37, pp. 91300–91336, 2024.

Shizhan Liu, Xinran Deng, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Jie Tang. Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability. In European Conference on Computer Vision, 2026.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019.

Susie Lu, Haonan Chen, Weirui Ye, and Yilun Du. DriftWorld: Fast World Modeling through Drifting, 2026.URLhttps://arxiv.org/abs/2607.15065.

Jiayi Luo, Hanxin Zhu, Chen Gao, Jiankun Wang, Cong Wang, Tianyu He, Jianxin Li, and Zhibo Chen. EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation. In European Conference on Computer Vision, 2026.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, March 2026. URL https://arxiv.org/abs/2603.19312.

Vincent Micheli, Eloi Alonso, and François Fleuret. Efficient World Models with Context-Aware Tokenization,2024. URL https://arxiv.org/abs/2406.19320.ICML 2024.

Ofir Nachum and Mengjiao Yang. Provable Representation Learning for Imitation with Contrastive Fourier Features. In Advances in Neural Information Processing Systems, pp. 30100–30112, 2021. doi: 10.48550/arXiv.2105.12272.

Zaruhi Navasardyan, Tatul Danielyan, and Hrant Davtyan. FailBench: How Reliable are VLMs at Judging Robot Task Success?, September 2026. URL https : //arxiv. org/abs/2609. 03611.

Tung Nguyen, Rui Shu, Tuan Pham, Hung Bui, and Stefano Ermon. Temporal Predictive Coding For Model-Based Planning In Latent Space. In International Conference on Machine Learning, pp. 8130–8139, 2021.

Nilaksh, Saurav Jha, Artem Zholus, and Sarath Chandar. Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models, May 2026. URL https : //arxiv. org/ abs/2605.06388.

NVIDIA. Cosmos World Foundation Model Platform for Physical AI, 2025. URL https : // arxiv.org/abs/2501.03575.

Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L. Lewis, and Satinder Singh. Action-Conditional Video Prediction using Deep Networks in Atari Games. In Advances in Neural Information Processing Systems, volume 28, 2015.

Masashi Okada and Tadahiro Taniguchi. Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction. In IEEE International Conference on Robotics and Automation, pp. 4209–4215, 2021. doi: 10.1109/ICRA48506.2021.9560734.

Abby O'Neill et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In IEEE International Conference on Robotics and Automation, pp. 6892–6903, 2024. doi: 10.1109/ ICRA57147.2024.10611477.

OpenAI. Competitive Programming with Large Reasoning Models, 2025. URL https : //arxiv. org/abs/2502.06807.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68sUt6zFt.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In International Conference on Machine Learning, pp. 2778–2787, 2017.

William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In IEEE/CVF International Conference on Computer Vision, pp. 4172–4182, 2023. doi: 10.48550/arXiv.2212. 09748.

Ayan Pendharkar. Predictive Objectives Discard Exogenous Control-Relevant Features: A Controlled Mechanistic Study, June 2026.URL https://arxiv.org/abs/2606.30068.

Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. WorldGym: World Model as An Environment for Policy Evaluation. In International Conference on Learning Representations, 2026. doi: 10.48550/arXiv.2506.00613.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10674–10685, 2022. doi: 10.48550/arXiv.2112.10752.

Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving,March 2025.URL https://arxiv.org/abs/2503.20523.

Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. In International Conference on Learning Representations, 2022. URL https : / /openreview . net/forum?id=TIdIXIpzhoI.

Dominik Schmidt and Minqi Jiang. Learning to Act without Actions, 2023. URL https : / /arxiv. org/abs/2312.10812.ICLR2024.

Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. Time-Contrastive Networks: Self-Supervised Learning from Video. In IEEE International Conference on Robotics and Automation, pp. 1134–1141, 2018. doi: 10.1109/ICRA.2018.8462891.

Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, and Sherry Yang. World-Gymnast: Training Robots with Reinforcement Learning in a World Model, February 2026. URL https://arxiv.org/abs/2602.02454.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play. Science, 362(6419):1140–1144, 2018. doi: 10.1126/ science.aar6404.

Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the Diffusability of Autoencoders. In International Conference on Machine Learning, 2025. doi: 10.48550/arXiv.2502.14831.

Vlad Sobal, Jyothir S, Siddhartha Jalagam, Nicolas Carion, Kyunghyun Cho, and Yann LeCun. Joint Embedding Predictive Architectures Focus on Slow Features, November 2022. URL https : //arxiv.org/abs/2211.10831.

Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models. In Advances in Neural Information Processing Systems, volume 38, pp. 43905–43941, 2025.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In International Conference on Learning Representations, 2021. URL https : / /openreview . net/forum?id=St1giarCHLP.

Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-Guided Video Diffusion. In International Conference on Machine Learning, pp. 56242– 56280, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063.

Rui Tian, Qi Dai, Jianmin Bao, Kai Qiu, Yifan Yang, Chong Luo, Zuxuan Wu, and Yu-Gang Jiang. REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents, 2024.URL https://arxiv.org/abs/2411.13552.ICCV2025.

Wei-Cheng Tseng, Jinwei Gu, Qinsheng Zhang, Hanzi Mao, Ming-Yu Liu, Florian Shkurti, and Lin Yen-Chen. Scalable Policy Evaluation with Video World Models, December 2025. URL https://arxiv.org/abs/2511.11520.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards Accurate Generative Models of Video: A New Metric & Challenges, 2018. URLhttps://arxiv.org/abs/1812.01717.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion Models Are Real-Time Game Engines. In International Conference on Learning Representations, 2025.

Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani, Aviv Regev, and Randall Balestriero. Joint-Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self-Supervised Learning. In Advances in Neural Information Processing Systems, volume 38, pp. 21897–21937, 2025.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive Predictive Coding, July 2018. URL https://arxiv.org/abs/1807.03748.

Homer Rich Walke, Kevin Black, Tony Z. Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, Abraham Lee, Kuan Fang, Chelsea Finn, and Sergey Levine. BridgeData V2: A Dataset for Robot Learning at Scale. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pp. 1723-1736, 2023. URL https://proceedings.mlr.press/ v229/walke23a.html.

Wan Team. Wan: Open and Advanced Large-Scale Video Generative Models, 2025. URL ht tps : //arxiv.org/abs/2503.20314.

Junke Wang, Qihang Zhang, Shuai Yang, Yiming Luo, Yujun Shen, Zuxuan Wu, Yu-Gang Jiang, and Yinghao Xu. RepWAM: World Action Modeling with Representation Visual-Action Tokenizers June 2026.URLhttps://arxiv.org/abs/2606.13674.

Tongzhou Wang and Phillip Isola. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. In International Conference on Machine Learning, pp. 9929–9939, 2020.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. doi:10.1109/TIP.2003.819861.

Jialong Wu, Haoyu Ma, Chaoyi Deng, and Mingsheng Long. Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning, 2023. URL https : //arxiv. org/abs/2305.18499.NeurIPS2023.

Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive VideoGPTs are Scalable World Models. In Advances in Neural Information Processing Systems, volume 37, pp. 68082–68119, 2024.

Zaishuo Xia, Yukuan Lu, Xinyi Li, Yifan Xu, and Yubei Chen. Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 32602–32612, 2026.

Zhihao Xie, Junfeng Wu, Xinting Hu, Junchao Huang, and Li Jiang. VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders, July 2026. URL https://arxiv.org/abs/2607.14088.

Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning Interactive Real-World Simulators. In International Conference on Learning Representations, 2024. doi: 10.48550/arXiv.2310.06114.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In International Conference on Learning Representations, 2025.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15703–15712, 2025. doi: 10.48550/arXiv.2501.01423.

Jingfeng Yao, Yuda Song, Yucong Zhou, and Xinggang Wang. Towards Scalable Pre-training of Visual Tokenizers for Generation. In European Conference on Computer Vision, 2026.

Jewon Yeom, Hanseul Kim, Jeongjae Park, Sungmok Jung, Jaejin Lee, and Taesup Kim. What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction, 2026. URL https://arxiv.org/abs/2606.07687.

Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language Model Beats Diffusion – Tokenizer is Key to Visual Generation. In International Conference on Learning Representations, 2024.

Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal M. Patel, Paul Pu Liang, Daniel Khashabi, Cheng Peng, Rama Chellappa, Tianmin Shu, Alan Yuille, Yilun Du, and Jieneng Chen. World-in-World: World Models in a Closed-Loop World. In International Conference on Learning Representations, 2026a. doi: 10.48550/arXiv.2510.18135.

Mingkun Zhang, Wangtian Shen, Fan Zhang, Haijian Qin, Zihao Pei, and Ziyang Meng. RAE-NWM: Navigation World Model in Dense Visual Representation Space. In European Conference on Computer Vision, 2026b.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.

Yian Zhao, Feng Wang, Qiushan Guo, Chang Liu, Xiangyang Ji, Jian Zhang, and Jie Chen. Video Generation with Predictive Latents, 2026.URL https://arxiv.org/abs/2605.02134.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion Transformers with Representation Autoencoders. In International Conference on Learning Representations, 2026. doi: 10.48550/arXiv.2510.11690.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. In International Conference on Machine Learning, pp. 79115–79135, 2025a.

Zhiyuan Zhou, Pranav Atreya, You Liang Tan, Karl Pertsch, and Sergey Levine. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World. In Conference on Robot Learning, 2025b. URL https://arxiv.org/abs/2503.24278.

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. IRASim: A Fine-Grained World Model for Robot Manipulation. In IEEE/CVF International Conference on Computer Vision, pp. 9834–9844, 2025.

## GUIDE TO THE APPENDIX

Appendix A specifies the synthetic study of Section 4 and gives its full results. For the video experiments, Appendices B to E describe the objectives, architectures, training and evaluation. Appendix F and Table 3 give the settings specific to the experiments of Sections 5.1 and 5.3, which we call the equal-reconstruction and the equal-budget experiments; the experiment of Section 5.2 is the weight sweep. Appendix G contains the additional results: controls, complete results and learning curves, the autoencoder losses over training, interventions on the past-conditioned decoder, policy evaluation, qualitative examples, longer world-model training, the latent spaces and ablations.

## A SYNTHETIC STUDY

Environment. A state is $\boldsymbol { x } _ { t } = \left( \boldsymbol { s } _ { t } , \boldsymbol { \xi } _ { t } \right)$ , with a signal $s _ { t } \in \mathbb { R } ^ { 2 }$ and distractors $\xi _ { t } \in \mathbb { R } ^ { 1 0 0 }$ , and the observation is the state itself. At the start of a trajectory, s0 and $\xi _ { 0 }$ are drawn from a standard normal distribution. At every step, an action is drawn uniformly among the four vectors $\pm 0 . 3 e _ { 1 }$ and $\pm 0 . 3 e _ { 2 }$ the signal moves by it, $s _ { t + 1 } = s _ { t } + a _ { t } .$ and $\xi _ { t + 1 }$ is a new standard normal draw, independent of the past. A trajectory has two transitions. The training, validation and test sets hold 10,000, 1,000 and 1,000 trajectories, that is, 20,000 training transitions.

Networks. Every network is a multilayer perceptron with one hidden layer of 256 units and ReLU activations, except the probe, whose hidden layer has 64 units. The encoder maps the observation to the mean and the log-variance of a Gaussian posterior over $z \in \mathbb { R } ^ { d }$ . The training loss uses a sample from the posterior. All downstream networks use its mean, as the experiments on real video do. The decoder of ${ \mathcal { L } } _ { \mathrm { r e c } }$ maps a latent back to $\mathbb { R } ^ { 1 0 2 }$ , and the key network g of $\mathcal { L } _ { \mathrm { t e m p } }$ maps the next observation and the one-hot action to $\mathbb { R } ^ { d }$ . The dynamics model maps a latent and the one-hot action to a latent, the probe maps a latent to the signal, and the readout decoder maps a latent to the observation. These three networks are the world model, the probe and the decoder D of Figure 2.

Objectives. The reconstruction term is the squared error averaged over the 102 coordinates, plus a KL penalty with weight $1 0 ^ { - 4 }$ : the KL divergence from the posterior to the standard normal distribution, summed over the latent coordinates. The temporal term is the InfoNCE loss. Its query is the standardized current latent $q _ { t } = \mathrm { L N } ( z _ { t } )$ and its keys are $y = \mathrm { L N } ( g ( x _ { t + 1 } , a _ { t } ) )$ , where LN is layer normalization without affine parameters, and a query and a key are scored by $- \| q _ { t } - y \| ^ { 2 } / ( \tau d )$ with temperature $\tau = 1$ . Standardizing fixes the norms of queries and keys at ${ \sqrt { d } } ,$ so the score equals $- 2 ( 1 - \cos \phi ) / \tau$ , with φ the angle between them. The score thus depends only on this angle, and the temperature transfers across latent sizes. The negatives are the keys of the other 255 transitions of the batch, drawn from the whole training set and thus, with rare exceptions, from other trajectories. They penalize latents that do not distinguish observations, which rules out a collapsed latent as a minimizer of the temporal term (Wang & Isola, 2020). The temperature and the KL weight were selected on the validation next-signal error and are fixed across the grid. $\mathrm { A t } \lambda = 1$ the loss is the contrastive objective alone, and at $\lambda = 0$ the reconstruction and KL terms alone, which keep their relative weight at every λ.

Relation to the objectives on real video. The temporal term is the contrastive objective of Appendix B in its simplest form. Its score is a squared distance between standardized vectors, without the bilinear form, the momentum encoder and the queue. The roles of query and key are also exchanged. Here the current latent is the query, and a separate network computes the keys from the next observation and the action (Nachum & Yang, 2021). On real video, a prediction from the current latent is scored against encoded future frames. The reconstruction terms differ because the data do: a squared error with a KL penalty here, and the pixel and LPIPS terms of Appendix B on real video.

Training. Training runs in four phases: first the encoder, with the decoder of $\mathcal { L } _ { \mathrm { r e c } }$ and the key network $^ { g , }$ then the dynamics model, the probe and the readout decoder, each on the latents of the frozen encoder. We train a separate readout decoder rather than reuse the decoder of $\mathcal { L } _ { \mathrm { r e c } } .$ which receives no gradient at $\lambda = 1$ , so that reconstruction is measured in the same way at every λ. Each phase runs 7,500 updates of AdamW at batch size 256 and learning rate $1 0 ^ { - 3 }$ , without weight decay, and the encoder phase clips the gradient norm at 1. We use the final weights; apart from the selection of the temperature and the KL weight, the validation set serves for monitoring only. The dynamics model minimizes the squared error to the latent of the next observation, the probe the squared error to the signal, and the readout decoder the squared error to the observation.

Evaluation. All errors are squared errors on the test transitions, averaged over the target coordinates and over transitions; lower is better. The reconstruction error compares the readout decoder's output on $z _ { t }$ with $x _ { t } .$ , and the current-signal error compares the probe's output on $z _ { t }$ with $s _ { t } .$ The next-latent error compares the dynamics model's prediction from $z _ { t }$ and $a _ { t }$ with $z _ { t + 1 }$ , and the next-signal error compares the probe's output on that prediction with $s _ { t + 1 }$ . Each error is divided by the variance of its target on the test set of the same seed, so a constant prediction of the mean scores 1. The target of the next-latent error is the latent itself, whose scale is learned and differs across latent sizes and objectives. We estimate the variance of the latent as half the mean squared distance between the latents of two independently drawn test observations, averaged over coordinates. Figure 2 divides each error of an encoder by the error of the raw observation on the same seed, which shares its data, and shows the geometric mean of this ratio over seeds with the standard error of its logarithm.

Grid and replication. The grid crosses the latent sizes $l \in \{ 1 0 , 2 0 , 3 0 , 5 0 , 1 0 0 \}$ with the weights $\lambda \in \{ 0 , 0 . 0 5 , 0 . 2 , 0 . 5 , 0 . 8 , 1 \}$ . Section 4 shows $\lambda \in \{ 0 , 0 . 5 , \dot { 1 } \}$ , and Figure 4 shows every weight. With the raw observation $( d = 1 0 2 )$ as the reference, the grid has 31 configurations. Each is repeated with three seeds, which set the data draw and every initialization, so the replicates are independent draws of the whole experiment (93 runs). A run takes a few minutes on 8 CPU cores.

![](images/007b28556016596687da1b0054acff00924d647de3292744f40e773acea74d04.jpg)  
Figure 4: The synthetic study at every latent size and weight. Errors on the observation, the signal the next latent and the next signal against the latent size, for all six weights λ of the contrastive temporal term $( \lambda = 0 ;$ reconstruction alone; $\lambda = 1 \colon$ the temporal term alone). Horizontal lines: the raw observation given directly to the same downstream networks (in the next-latent panel, hidden under $\lambda = 0 )$ . Mean and standard error over three seeds. Intermediate weights retain reconstruction information while improving signal recovery and latent predictability over reconstruction alone.

## B TEMPORAL OBJECTIVES

The autoencoder is trained on pairs of frames $x _ { t }$ and $x _ { t + 1 }$ , with latents $z _ { t } = E ( x _ { t } )$ and $z _ { t + 1 } =$ $E ( x _ { t + 1 } )$ , and the action $a _ { t }$ executed between them. In the equal-reconstruction experiment, a third of the pairs are two frames apart, and the predictor then receives both actions between them. The three forward objectives score the prediction $P ( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } )$ of the predictor of Appendix C.

The reconstruction loss $\mathcal { L } _ { \mathrm { r e c } }$ is the mean absolute pixel error plus VGG-based LPIPS (Zhang et al., 2018), with weight one on each term. The pixel term is averaged over channels and positions on images in [0, 1], and LPIPS receives the images mapped to [-1, 1]. All losses are averaged over examples. We multiply the loss of Equation 1 by two, which gives both terms unit weight at $\lambda = 1 / 2 ;$ with AdamW, the factor matters only through gradient clipping. Unless stated otherwise, no adversarial or KL term is used, and the latent is the mean of the encoder's posterior.

Latent prediction. The prediction is regressed onto the future latent,

$$
\mathcal { L } _ { \mathrm { l a t e n t } } = \frac { 1 } { d _ { z } } \big \| P ( z _ { t } , a _ { t } ) - \mathrm { s g } [ z _ { t + 1 } ] \big \| _ { 1 } ,\tag{2}
$$

where $d _ { z }$ is the number of latent coordinates (16,384 for FLUX.1; the synthetic study calls its latent size d) and sg is the stop-gradient operator. The target comes from the same encoder, without a momentum copy.

Pixel prediction. The prediction is decoded and compared with the future frame,

$$
\mathcal { L } _ { \mathrm { p i x e l } } = \mathcal { L } _ { \mathrm { r e c } } \big ( x _ { t + 1 } , D ( P ( z _ { t } , a _ { t } ) ) \big ) ,\tag{3}
$$

with the same decoder and reconstruction loss as for the current frame. The gradient reaches the encoder, the predictor and the decoder.

Contrastive prediction. The prediction is trained to identify the future frame among alternatives with the InfoNCE loss. Queries q and keys y are unit-normalized projections of the prediction and of frames encoded by a momentum copy of the encoder, compared through a learned bilinear form W:

$$
{ \mathcal { L } } _ { \mathrm { c o n t r } } = - \log { \frac { \exp ( q ^ { \top } W y ^ { + } ) } { \exp ( q ^ { \top } W y ^ { + } ) + \sum _ { y ^ { - } \in y ^ { - } } \exp ( q ^ { \top } W y ^ { - } ) } } ,\tag{4}
$$

where $y ^ { + }$ is the key of $x _ { t + 1 }$ and $y -$ contains the keys of the future frames of the other videos in the batch and a queue of 40,960 keys from recent batches. A momentum copy of the projection (Appendix C) is applied to the keys, together with the momentum encoder, with decay 0.9978. W is initialized at the identity, so the score starts as a cosine similarity; the temperature is one, and the learned W also sets the scale of the scores.

Inverse dynamics. A head H predicts the action between two consecutive frames from their latents,

$$
\mathcal { L } _ { \mathrm { i n v } } = \ell \big ( H ( z _ { t } , z _ { t + 1 } ) , a _ { t } \big ) ,\tag{5}
$$

where l averages over the seven action components: the squared error on the six translation and rotation components, standardized with the mean and standard deviation of the training actions, and the binary cross-entropy on the gripper state. Both latents receive gradients.

Reconstruction target within a pair. With the ordinary decoder, the earlier frame of the pair is reconstructed from its own latent, $D ( z _ { t } ) \approx x _ { t }$ , and the future latent $z _ { t + 1 }$ is computed without gradient. With the past-conditioned decoder, the later frame is reconstructed from its latent and the earlier frame, $D ( z _ { t + 1 } , x _ { t } ) \approx x _ { t + 1 }$ . The latent of the earlier frame then receives no reconstruction gradient, and the latent of the later frame receives no gradient from the forward objectives.

## C ARCHITECTURES

We describe the networks of the FLUX-based video experiments. The DINOv2 ablation changes some of them (Appendix G.9.2).

Autoencoder. Unless stated otherwise, the video experiments initialize both the encoder and the decoder from the pretrained autoencoder of FLUX.1-schnell (Black Forest Labs, 2024). We use its variational autoencoder, with 16 latent channels and a spatial downsampling factor of eight: a $2 5 6 \times 2 5 6$ frame gives a $1 6 \times 3 2 \times 3 2$ latent. The encoder and the decoder are fine-tuned during representation learning and frozen during world-model training; the pretrained-autoencoder control skips fine-tuning.

Predictor. Two $1 \times 1$ convolutions map the 16 channels of the latent to a width of 512 and back, around two residual blocks. Each block applies a depthwise-separable convolution with kernel 3, self-attention over the $3 2 \times 3 2$ positions with two-dimensional rotary position embeddings (Su et al., 2024) and 8 heads, cross-attention to the action tokens, and a feed-forward layer with expansion factor 4 and GELU activation. The sublayers use group normalization with 32 groups and residual connections, without dropout. Each action is embedded by a two-layer perceptron of width 512 with GELU activation, plus a learned position embedding. Together with a learned summary token, these embeddings pass through a two-layer transformer of width 512 with 8 heads, feed-forward width 2,048, GELU activations and layer normalization before each sublayer and at the output. Its outputs, for the actions and for the summary token, are the keys and values of the cross-attention. The predictor has 15.6 M parameters, 6.6 M of them in the action encoder.

Inverse-dynamics head. The two latents are concatenated along channels and processed by blocks of the same design, without cross-attention; the output is averaged over positions and mapped to the seven action components by a two-layer perceptron. The head has 7.1 M parameters.

Contrastive head. The projection applied to the prediction and, through its momentum copy, to the keys is a single residual $7 \times 7$ convolution that keeps the shape of the latent. The bilinear form W of Equation 4 acts on the flattened projections.

Past-conditioned decoder. The pretrained decoder (49.5 M parameters, all fine-tuned) receives five cross-attention adapters with 8 heads, after the middle block and after each of the four upsampling blocks (3.7 M parameters). The context is a $3 2 \times 3 2$ grid of 1,024 tokens from the past frame. At each position, an $8 \times 8$ pixel patch and the corresponding middle-block features of a frozen copy of the pretrained encoder, the context encoder (34.3 M parameters, not trained) are projected separately to width 512, concatenated, and projected back to width 512 (0.9 M parameters). Each adapter projects these context tokens to the channel width of its decoder block for keys and values, and uses the decoder features as queries. The adapters use group normalization with 32 groups and two-dimensional rotary position embeddings on queries and keys, without dropout, and the output projection of every adapter is zero-initialized. In total, the adapters and the projections add 4.6 M trained parameters. The context features depend only on the context frame, which stays fixed during a rollout (Appendix E), so they need to be computed only once per rollout. Per generated frame, the decoder adds only the cross-attention adapters to the pretrained decoder.

World-model transformer. Each standardized latent frame is patchified with patch size 2 into 256 tokens. Every block consists of two transformer layers, one with bidirectional self-attention within a frame and one with causal self-attention across frames at each spatial position, each followed by a feed-forward layer. Both attentions use rotary position embeddings. Adaptive layer normalization with zero initialization modulates every layer from the sum of the noise-level embedding and a linear embedding of the frame's action, so the modulation is specific to each frame, as diffusion forcing requires. The action at reaches frame t + 1 through the causal attention across frames. The final layer predicts the velocity of every token. Table 3 gives the width, depth and number of attention heads in each experiment

## D TRAINING DETAILS

Data. Bridge V2 is recorded with a WidowX 250 robot arm at 5 Hz (Walke et al., 2023). We use the TensorFlow Datasets release bridge\_dat aset, version 1.0.0, with its training and validation splits: 53,192 training videos and 6,872 held-out videos. Frames are decoded from the stored videos, scaled to [0, 1] and resized to 256 × 256 without cropping. For the FLUX encoder, they are then mapped to [–1, 1]. Appendix G.9.2 gives the preprocessing for DINOv2.

Actions. The action $a _ { t }$ has seven components: the translation and the rotation of the end effector, rescaled by fixed factors and clipped, and the gripper state, —1 when open and +1 when closed. A further flag marks the null action, which replaces the action when the conditioning is dropped.

Training examples. Each epoch draws one pair of frames per training video for autoencoder training, and one clip of ten consecutive frames per training video for world-model training, both with a uniformly distributed start frame. Videos with fewer than ten frames are left out of world-model training. Table 3 gives the remaining settings of the two main experiments.

Table 3: Training settings. The ablations use the settings of the equal-budget experiment unless Appendix G.9 states otherwise.
<table><tr><td></td><td>Equal reconstruction (Section 5.1)</td><td>Equal budget and weight sweep (Sections 5.2 and 5.3)</td></tr><tr><td>Autoencoder fine-tuning</td><td></td><td></td></tr><tr><td>Frame pairs</td><td>consecutive, or two frames apart with probability 1/3</td><td>consecutive</td></tr><tr><td>Batch</td><td>24 pairs</td><td>48 pairs</td></tr><tr><td>Learning rate</td><td> $2 . 4 \times 1 0 ^ { - 6 }$ </td><td> $3 . { \dot { 4 } } 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Length</td><td>until the reconstruction target of Appendix F 80,000 updates; final checkpoint one per held-out video, redrawn at each</td><td>a fixed set, one per held-out video</td></tr><tr><td>Validation pairs</td><td>check</td><td></td></tr><tr><td>Optimizer</td><td colspan="2">AdamW (Loshchilov &amp; Hutter, 2019) with  $\beta = ( 0 . 9 , 0 . 9 9 9 )$  1,000-update linear warm-up; gradient clipping at norm 0.5</td></tr><tr><td>World model</td><td></td><td></td></tr><tr><td>Width, blocks, heads</td><td>1,024, 16, 16</td><td>384, 12, 6</td></tr><tr><td>Parameters</td><td>608 M</td><td>64.4 M</td></tr><tr><td>Batch Learning rate</td><td>4 clips  $3 \times \mathrm { 1 0 ^ { - 5 } }$ </td><td>16 clips (8 per step, two accumulation steps)  $7 \times 1 0 ^ { - 5 }$  after a 3,000-update linear warm-up</td></tr><tr><td></td><td>, constant</td><td>from  $1 0 ^ { - 7 }$ </td></tr><tr><td>Gradient clipping</td><td>none</td><td>at norm 0.5</td></tr><tr><td>Updates</td><td>238,896 (18 epochs)</td><td>100,000</td></tr><tr><td>Latent standardization</td><td>one global standard deviation AdamW with  $\beta = ( 0 . 9 , 0 . 9 9 )$ </td><td>per-channel mean and standard deviation and weight decay 0.02; exponential moving average of the</td></tr><tr><td>Optimizer</td><td colspan="2">weights with decay 0.999, used for sampling</td></tr><tr><td>Precision</td><td colspan="2">bf16 mixed precision for both models</td></tr></table>

Diffusion process. Each latent is corrupted independently as $z _ { t } ^ { k _ { t } } = \sqrt { \bar { \alpha } _ { k _ { t } } } z _ { t } + \sqrt { 1 - \bar { \alpha } _ { k _ { t } } } \epsilon _ { t } ,$ with $\epsilon _ { t } \sim \mathcal { N } ( 0 , I ) , k _ { t }$ uniform on $\{ 1 , \ldots , K \}$ , and a variance-preserving schedule $1 > \bar { \alpha } _ { 1 } > \cdot \cdot \cdot > \bar { \alpha } _ { K }$ ≈ 0 (Ho et al., 2020). A network vθ receives the noisy latents, their noise levels, and the actions, and regresses the velocity $v _ { t } = \sqrt { \bar { \alpha } _ { k _ { t } } } \epsilon _ { t } - \sqrt { 1 - \bar { \alpha } _ { k _ { t } } } z _ { t }$ (Salimans & Ho, 2022) of every frame:

$$
\mathcal { L } _ { \mathrm { D F } } ( \theta ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left\| v _ { \theta } \left( z _ { 1 : T } ^ { k _ { 1 : T } } , k _ { 1 : T } , a _ { 1 : T } \right) _ { t } - v _ { t } \right\| _ { 2 } ^ { 2 } \right] .\tag{6}
$$

The diffusion process has $K = 1 { , } 0 0 0$ noise levels with a sigmoid schedule (Jabri et al., 2023), whose argument runs from —3 to 3 with temperature one; the per-step noise variances are clipped to [0, 0.999]. In practice, the squared velocity error is averaged, rather than summed, over latent coordinates and frames, and averaged over examples. During training, the action is replaced by the null action with probability 0.1, which allows classifier-free guidance (Ho & Salimans, 2021) at sampling time. The DINOv2 ablation changes the parameterization and the noise-level sampling (Appendix G.9.2).

Latent standardization. After fine-tuning, the statistics of the encoder output are computed on 100 batches of training frames, one random frame per clip (Table 3 gives the standardization of each experiment). The world model is trained and sampled in the standardized space, and the transformation is inverted before decoding.

Compute. Autoencoders and world models are trained on NVIDIA H200 GPUs, one per run except the width-1,024 world models of the size ablation, which use two. At the equal-budget settings, fine-tuning an autoencoder takes about 31 hours with the ordinary decoder and about 42 hours with the past-conditioned decoder. In the equal-reconstruction experiment it takes from 1.3 hours (reconstruction alone) to about 30 hours (latent prediction). Training a world model takes about 17 hours at width 384, and about 37 hours for the larger world models of the equal-reconstruction experiment. Evaluating a world model in the four conditions of Table 4 takes 3 to 5 hours, or about 10 hours in the equal-reconstruction experiment.

## E EVALUATION DETAILS

Evaluation sets. In distribution, we evaluate on 2,000 clips of ten frames, one from each of 2,000 randomly chosen held-out videos, with a uniformly distributed start frame; every model is evaluated on the same clips. The sampling-step sweep uses the first 512 of these clips. The long-rollout evaluation uses 716 held-out videos: all 224 videos with at least 60 frames, together with 512 videos drawn uniformly among those with at least 20 frames, 20 of which are also among the 224. Each is rolled out once, for up to 59 predicted frames. Out of distribution, we use all 479 AutoEval trajectories, recorded on separate robot setups that reproduce four Bridge V2 tasks: closing the drawer (80 trajectories), opening the drawer (94), placing the eggplant in the sink (193) and placing it in the basket (112). Frames are resized to 256 × 256, and the rollout takes the 21st frame of each trajectory as the context frame and predicts the following 29 frames. The recorded actions are rescaled as in Appendix D; the gripper command is binarized and mapped to the sign convention of Bridge V2, which is opposite to that of AutoEval.

Sampling. Table 4 gives the four evaluation conditions. DDIM uses the stated number of steps on a uniform grid over the 1,000 noise levels, without added noise, with classifier-free guidance of scale 3 at every step. The DINOv2 ablation warps the grid (Appendix G.9.2). For a chunk of eight future frames, all eight are appended as noise and denoised jointly at staggered noise levels, each frame starting at a higher level than the previous one: the pyramid schedule of Section 3. The last chunk can be shorter. The past-conditioned decoder conditions on the initial real context frame throughout the rollout. Within each comparison, every model uses the same diffusion noise on the same clips, with one rollout per clip and sampling seed.

Table 4: Evaluation conditions. Each rollout starts from one real context frame, and the scored prediction steps count from the first frame after it. The denoising chunk is the largest number of future frames sampled jointly. The 2-step columns of the result tables come from the sampling-step sweep at 2 steps.
<table><tr><td>Condition</td><td>Clips</td><td>Scored prediction steps</td><td>Denoising chunk</td><td>DDIM steps</td><td>Sampling seeds</td></tr><tr><td>In distribution</td><td>2,000</td><td>1-9</td><td>8</td><td>10</td><td>3</td></tr><tr><td>Out of distribution</td><td>479</td><td>1-29</td><td>1</td><td>10</td><td>3</td></tr><tr><td>Sampling-step sweep</td><td>512</td><td>1-9</td><td></td><td>1 2, 4, 6, 8, 10</td><td>1</td></tr><tr><td>Long rollouts</td><td>716</td><td>1-59</td><td>8</td><td>10</td><td>1</td></tr></table>

Metrics. Pixel mean squared error (MSE) is computed per frame over channels and pixels in [0, 1]. SSIM uses a Gaussian window of 11 × 11 pixels with standard deviation 1.5. LPIPS uses VGG16 features with the learned linear layers of Zhang et al. (2018), in the implementation of Esser et al. (2021), on inputs in [–1, 1]; the same network enters the reconstruction loss of every autoencoder with the same weight. In and out of distribution, each framewise metric is averaged over the generated frames of a clip, then over clips, then over the three sampling seeds, which gives one value per trained pipeline. Out-of-distribution values are the average of the four task means, which weights the tasks equally. The sampling-step sweep and the long rollouts use one sampling seed and the frame windows given below. FID (Heusel et al., 2017) uses the 2,048-dimensional pool features of Inception-v3 on the nine generated frames of the 2,000 clips, 18,000 frames on each side. FVD (Unterthiner et al., 2018) uses the 400-dimensional outputs of an I3D network (Carreira & Zisserman, 2017) trained on Kinetics-400, with the weights released by Song et al. (2025), on the 2,000 clips of nine frames. Both are computed in distribution only, on the same clips for every training and sampling seed. As for the framewise metrics, we average the three sampling-seed values of each trained pipeline.

Long rollouts and sampling steps. In the long-rollout evaluation, the value of a metric at a prediction step is the mean over the videos that reach that step, so every model is scored on the same videos at each step. Their number falls from 716 at steps 1 to 19 to 224 at steps 57 to 59. A long rollout is summarized by the mean of this value over prediction steps 51 to 59, which 229 to 224 videos reach. The sampling-step sweep generates the nine frames of each of the 512 clips one frame at a time, with 2, 4, 6, 8 and 10 DDIM steps under the same noise.

Unpredictable share. We fit the same linear predictor to every frozen autoencoder, including those at $\lambda = 0$ , on the 2,000 in-distribution clips (18,000 consecutive-frame transitions), with the latents standardized per channel as in Appendix D. We use five-fold cross-fitting over clips, keeping every transition of a clip in one fold. For each held-out fold, let $\mu$ be the mean latent over the frames of the other four folds, and let $u _ { t }$ be the orthonormal two-dimensional Hartley transform of $z _ { t } - \mu ,$ applied to each channel. We predict each frequency coordinate as

$$
\begin{array} { r } { \hat { u } _ { t + 1 , j } = b _ { j } u _ { t , j } + w _ { j } ^ { \top } c _ { t } , } \end{array}\tag{7}
$$

where $c _ { t }$ concatenates the 256 leading principal coordinates of the centered current latent and the action. The first term lets each frequency coordinate persist from one frame to the next. The principal directions are fitted on the four training folds, and so are $b _ { j }$ and $w _ { j } ,$ jointly by least squares with a ridge penalty on $w _ { j }$ whose strength is selected on an inner split of those folds. The reported share is

$$
\frac { \sum _ { ( t , f ) } \| z _ { t + 1 } - \hat { z } _ { t + 1 } ^ { ( - f ) } \| _ { 2 } ^ { 2 } } { \sum _ { ( t , f ) } \| z _ { t + 1 } - \mu ^ { ( - f ) } \| _ { 2 } ^ { 2 } } ,\tag{8}
$$

where the sums run over all held-out transitions, $\hat { z } _ { t + 1 } ^ { ( - f ) }$ is the prediction mapped back from the frequency basis, and $\left( - f \right)$ denotes a fit that excludes the fold of the transition. The denominator thus measures the variance of the target latents around the training-fold mean, not the variance of their frame-to-frame differences.

Latent MSE. From one context frame and the recorded action, the world model predicts the next latent with 10 DDIM steps and no guidance. The latent MSE is the mean squared difference to the encoder's latent of the recorded next frame, divided by the pooled variance of the target latents. It is computed on the first transition of 256 in-distribution clips, and on one transition of each of 256 out-of-distribution trajectories, 64 per task.

Training-curve evaluation. The training-speed factors of Section 5.3 use the validation curves of the reconstruction pipelines and the temporal objectives, averaged over their three training seeds. These runs validate once per epoch (3,318 updates) on 400 fixed held-out clips, generating nine frames from one context frame, one frame per denoising chunk, with 10 DDIM steps and guidance scale 3. The world models of the equal-reconstruction experiment validate every three epochs (39,816 updates; Figure 5). The context frame is excluded from the metrics, except in the equal-reconstruction experiment, whose validation averages over all ten frames. For each metric, the reference value is the final mean validation value of the baseline, at 99,540 updates (30 epochs). We find the first crossing of this value by another mean curve, interpolate linearly in log(update) between the adjacent validation points, and divide 99,540 by the crossing update.

Controls. The reconstruction ceiling encodes and decodes the recorded frames of every evaluation clip with each autoencoder, in the same way as the rollouts, and applies the same metrics. Pastconditioned decoders use the initial real context frame, as in the rollouts. The shuffled-action control repeats every rollout with the action sequence of another clip, assigned by a fixed permutation without fixed points that is shared by every model. Out of distribution, the donor clip comes from the same task. The shuffled rollout uses the same diffusion noise as the rollout under the recorded actions. Both controls use one sampling seed.

Statistics. The equal-reconstruction and equal-budget experiments use three training seeds each. The weight sweep fine-tunes its own autoencoders with one training seed, and with a second one at λ = 0 and 0.3. Its world models at λ = 0.1 to 0.5 use one seed. At λ = 0 and 0.7, the sweep reuses the world models of the equal-budget experiment. The seed also fixes the order of the training data, so the objectives of an experiment see the same data in the same order. The unit of replication is the trained pipeline, that is, the autoencoder with its world model; sampling seeds and evaluation clips are repeated measurements of one pipeline. We report the mean and the standard error over training seeds.

## F EQUAL-RECONSTRUCTION EXPERIMENT

Pipelines. The four pipelines differ only in the objective used to fine-tune the autoencoder, its weight, and hence the number of updates needed to reach the target. The data, the decoder, the world model, its training and the evaluation are shared (Appendices C to E and Table 3). The decoder is initialized from FLUX.1-schnell and fine-tuned to reconstruct each frame from its latent alone. The temporal weight λ of Equation 1 is set per objective so that fine-tuning reaches the reconstruction target: 0.3 for latent and contrastive prediction and 0.5 for pixel prediction.

Stopping rule. Fine-tuning stops at the first end of an epoch (2,215 updates) at which the reconstruction error on the held-out pairs is below 0.0345. The update counts of Table 1 therefore exceed the number of updates needed to reach the target by less than one epoch. At the stop, the held-out reconstruction error is 0.0342 for reconstruction alone and 0.0343 to 0.0345 for the temporal objectives. The principal-component counts of Table 1 are computed on the flattened latents of 1,000 held-out frames from 100 videos (Appendix G.8).

## G ADDITIONAL RESULTS

## G.1 CONTROLS

Two controls test whether the generated trajectories depend on the recorded actions, and how much of the error remains when only the autoencoder is used. The shuffled-action control and the reconstruction ceiling of Appendix E were measured on the four pipelines of the equal-reconstruction experiment and on the three past-conditioned pipelines of the equal-budget experiment. Under the actions of another clip, the in-distribution pixel MSE rises by 98 to 102% in the equal-reconstruction experiment and by 62 to 77% in the equal-budget experiment. Out of distribution, with a donor clip from the same task, it rises by 21 to 26% and by 20 to 23%. Encoding and decoding the recorded frames gives a pixel MSE $( \times \dot { 1 } 0 ^ { 3 } )$ of 0.19 to 0.32 in the equal-reconstruction experiment and 0.14 to 0.43 in the equal-budget experiment, against 11 to 34 for the world models on the same autoencoders. The world models therefore use the action sequence, and autoencoding alone does not account for their rollout error.

## G.2 COMPLETE RESULTS AND LEARNING CURVES

Tables 5 and 6 complete Tables 1 and 2 with SSIM and the distribution metrics, and Figures 5 and 6 show the errors against the number of sampling steps, along long rollouts and during world-model training.

## G.2.1 EQUAL RECONSTRUCTION

Latent prediction gives the best perceptual error in every condition, with the largest gains when the sampler has only two steps, and the lowest FID (Table 5); contrastive prediction gives a slightly lower FVD. Figure 5 shows that the gains of latent prediction are largest with few sampling steps. Its LPIPS is also the lowest from the fourth prediction step to the end of the long rollouts, and at every validation point of world-model training.

Table 5: Complete results at equal reconstruction quality. Table 1 without its autoencoder columns, and with SSIM and the distribution metrics added; all pipelines use the ordinary decoder. Mean and standard error over three training seeds (pixel MSE $\bar { \times } \bar { 1 0 ^ { 3 } } )$ ; bold: best in each column. ID, OOD, 2-step and long-horizon as in Section 5. Latent prediction gives the best perceptual metrics, especially with two sampling steps, and the lowest FID; contrastive prediction gives a slightly lower FVD.
<table><tr><td></td><td colspan="3">ID</td><td colspan="3">OOD</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>Reconstruction (baseline)</td><td>11.65</td><td>0.164</td><td>0.801</td><td>33.87</td><td>0.308</td><td>0.675</td></tr><tr><td></td><td>±0.20</td><td>±0.001</td><td>±0.002</td><td>±1.28</td><td>±0.008</td><td>±0.009</td></tr><tr><td>Pixel prediction</td><td>11.43</td><td>0.159</td><td>0.806</td><td>34.23</td><td>0.300</td><td>0.683</td></tr><tr><td></td><td>±0.22</td><td>±0.001</td><td>±0.001</td><td>±0.67</td><td>±0.002</td><td>±0.002</td></tr><tr><td>Contrastive prediction</td><td>11.17</td><td>0.160</td><td>0.807</td><td>31.23</td><td>0.303</td><td>0.681</td></tr><tr><td></td><td>±0.05</td><td>±0.001</td><td>±0.001</td><td>±0.14</td><td>±0.005</td><td>±0.005</td></tr><tr><td>Latent prediction</td><td>11.30</td><td>0.153</td><td>0.814</td><td>29.14</td><td>0.298</td><td>0.688</td></tr><tr><td></td><td>±0.10</td><td>±0.001</td><td>±0.001</td><td>±0.55</td><td>±0.005</td><td>±0.005</td></tr><tr><td></td><td colspan="3">2-step</td><td colspan="3">Long-horizon</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM ↑</td></tr><tr><td>Reconstruction (baseline)</td><td>15.95</td><td>0.214</td><td>0.770</td><td>33.26</td><td>0.287</td><td>0.691</td></tr><tr><td></td><td>±1.03</td><td>±0.004</td><td>±0.005</td><td>±1.84</td><td>±0.008</td><td>±0.010</td></tr><tr><td>Pixel prediction</td><td>14.53</td><td>0.199</td><td>0.782</td><td>32.63</td><td>0.279</td><td>0.698</td></tr><tr><td></td><td>±0.75</td><td>±0.002</td><td>±0.003</td><td>±0.69</td><td>±0.003</td><td>±0.004</td></tr><tr><td>Contrastive prediction</td><td>12.38</td><td>0.185</td><td>0.794</td><td>31.50</td><td>0.277</td><td>0.701</td></tr><tr><td></td><td>±0.27</td><td>±0.001</td><td>±0.001</td><td>±2.62</td><td>±0.010</td><td>±0.011</td></tr><tr><td>Latent prediction</td><td>12.36</td><td>0.160</td><td>0.808</td><td>30.04</td><td>0.262</td><td>0.716</td></tr><tr><td></td><td>±0.24</td><td>±0.002</td><td>±0.003</td><td>±0.43</td><td>±0.000</td><td>±0.001</td></tr><tr><td></td><td colspan="2">Distribution</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>FID↓</td><td>FVD↓</td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction (baseline)</td><td>9.4</td><td>22.9</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.2</td><td>±0.4</td><td></td><td></td><td></td><td></td></tr><tr><td>Pixel prediction</td><td>9.3</td><td>23.2</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.1</td><td>±0.2</td><td></td><td></td><td></td><td></td></tr><tr><td>Contrastive prediction</td><td>9.1</td><td>22.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.2</td><td>±0.2</td><td></td><td></td><td></td><td></td></tr><tr><td>Latent prediction</td><td>8.3</td><td>22.7</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.1</td><td>±0.8</td><td></td><td></td><td></td><td></td></tr></table>

reconstruction only- latent prediction- pixel prediction- contrastive prediction  
![](images/d3bf13d1931fe3a5fe940c6a07a0f113bb013613344152f82b3b32fa8e1f1967.jpg)  
Figure 5: World models at equal reconstruction quality, across sampling steps, rollout steps and training. The four pipelines of Table 1, all with the ordinary decoder, against the number of DDIM sampling steps (top), along the long rollouts (middle) and over world-model training (bottom). Gray shading: prediction steps reached by fewer than half of the videos. Mean and standard error over three training seeds. Latent prediction gives the largest perceptual gains with few sampling steps, and the lowest LPIPS from the fourth prediction step on and throughout training.

## G.2.2 EQUAL TRAINING BUDGET

Both temporal objectives improve on both fine-tuned reconstruction pipelines in LPIPS and SSIM in all four conditions, and in pixel MSE out of distribution, with two sampling steps and on long rollouts (Table 6). For latent prediction, the temporal weight sets a trade-off. At λ = 0.7, which favors long rollouts (Appendix G.9.1), its FID and FVD are higher than those of reconstruction with the same decoder, and its in-distribution pixel MSE is slightly higher (14.28 against 14.00). Lower weights avoid this cost in FID and FVD.

Table 6: Complete results at equal training budget. Table 2 without its policy-evaluation columns (Table 8), and with SSIM, the out-of-distribution latent MSE and the distribution metrics added. Mean and standard error over three training seeds (pixel MSE × 103); bold: best in each column. ID, OOD, 2-step and long-horizon as in Section 5. Both temporal objectives improve LPIPS and SSIM over both fine-tuned reconstruction pipelines in all four conditions; at λ = 0.7, latent prediction has a higher FID and FVD than reconstruction with the same decoder, which lower weights avoid (Appendix G.9.1).
<table><tr><td></td><td colspan="3">ID</td><td colspan="3">OOD</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM ↑ MSE↓ LPIPS ↓ SSIM ↑</td><td></td><td></td><td></td></tr><tr><td>Ordinary decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction (baseline)</td><td>14.34</td><td>0.194</td><td>0.779</td><td>36.15</td><td>0.302</td><td>0.682</td></tr><tr><td rowspan="2">Pretrained, no fine-tuning</td><td>±0.16</td><td>±0.002</td><td>±0.001</td><td>±0.98</td><td>±0.003</td><td>±0.004</td></tr><tr><td>14.81</td><td>0.208</td><td>0.758</td><td>34.81</td><td>0.309</td><td>0.673</td></tr><tr><td></td><td>±0.26</td><td>±0.001</td><td>±0.001</td><td>±1.97</td><td>±0.008</td><td>±0.009</td></tr><tr><td>Past-conditioned decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction</td><td>14.00</td><td>0.176</td><td>0.803</td><td>31.45</td><td>0.270</td><td>0.718</td></tr><tr><td rowspan="2">Inverse dynamics</td><td>±0.14</td><td>±0.002</td><td>±0.003</td><td>±0.56</td><td>±0.006</td><td>±0.006</td></tr><tr><td>13.07</td><td>0.160</td><td>0.813</td><td>27.92</td><td>0.236</td><td>0.745</td></tr><tr><td rowspan="2">Latent prediction</td><td>±0.24</td><td>±0.001</td><td>±0.001</td><td>±1.17</td><td>±0.005</td><td>±0.006</td></tr><tr><td>14.28</td><td>0.169</td><td>0.810</td><td>27.54</td><td>0.257</td><td>0.739</td></tr><tr><td></td><td>±0.18</td><td>±0.001</td><td>±0.001</td><td>±0.67</td><td>±0.002</td><td>±0.003</td></tr><tr><td></td><td colspan="2">2-step</td><td></td><td colspan="3">Long-horizon</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td></td><td></td><td>SSIM ↑ MSE↓ LPIPS ↓ SSIM ↑</td><td></td></tr><tr><td>Ordinary decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Reconstruction (baseline)</td><td>18.35</td><td>0.232</td><td>0.756</td><td>47.93</td><td>0.394</td><td>0.595</td></tr><tr><td>±0.65</td><td>±0.004</td><td>±0.003</td><td>±1.51</td><td>±0.002</td><td>±0.004</td></tr><tr><td rowspan="2">Pretrained, no fine-tuning</td><td>19.15</td><td>0.250</td><td>0.734</td><td>46.78</td><td>0.393</td><td>0.589</td></tr><tr><td>±0.89</td><td>±0.003</td><td>±0.004</td><td>±0.83</td><td>±0.003</td><td>±0.002</td></tr><tr><td>Past-conditioned decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction</td><td>16.99</td><td>0.198</td><td>0.788</td><td>44.61</td><td>0.366</td><td>0.626</td></tr><tr><td rowspan="2">Inverse dynamics</td><td>±0.64</td><td>±0.003</td><td>±0.003</td><td>±1.13</td><td>±0.006</td><td>±0.007</td></tr><tr><td>13.44</td><td>0.162</td><td>0.815</td><td>39.39</td><td>0.333</td><td>0.653</td></tr><tr><td rowspan="2">Latent prediction</td><td>±0.28</td><td>±0.001</td><td>±0.001</td><td>±2.01</td><td>±0.005</td><td>±0.008</td></tr><tr><td>15.64</td><td>0.177 ±0.001</td><td>0.805</td><td>39.08 ±1.08</td><td>0.331</td><td>0.662</td></tr><tr><td rowspan="2"></td><td colspan="2">±0.26</td><td colspan="2">±0.001</td><td>±0.007</td><td>±0.008</td></tr><tr><td colspan="2">Predictability</td><td colspan="2">Distribution</td><td></td><td></td></tr><tr><td></td><td colspan="2">ID latent MSE ↓ OOD latent MSE ↓</td><td colspan="2">FID↓ FVD↓</td><td></td><td></td></tr><tr><td>Ordinary decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Reconstruction (baseline)</td><td>0.232</td><td>0.186</td><td>22.2</td><td>39.1</td><td></td><td></td></tr><tr><td>±0.008</td><td>±0.005</td><td>±0.6</td><td>±0.3</td><td></td><td></td></tr><tr><td rowspan="2">Pretrained, no fine-tuning</td><td>0.253</td><td>0.204</td><td>28.6</td><td>42.7</td><td></td><td></td></tr><tr><td>±0.001</td><td>±0.001</td><td>±0.3</td><td>±0.8</td><td></td><td></td></tr><tr><td>Past-conditioned decoder</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Reconstruction</td><td>0.212</td><td>0.173</td><td>17.5</td><td>36.0</td><td></td><td></td></tr><tr><td>±0.008</td><td>±0.011</td><td>±0.9</td><td>±0.6</td><td></td><td></td></tr><tr><td rowspan="2">Inverse dynamics</td><td>0.155</td><td>0.131</td><td>11.1</td><td>33.2</td><td></td><td></td></tr><tr><td>±0.001</td><td>±0.006</td><td>±0.4</td><td>±1.1</td><td></td><td></td></tr><tr><td rowspan="2">Latent prediction</td><td>0.046</td><td>0.049</td><td>21.5</td><td>45.6</td><td></td><td></td></tr><tr><td>±0.001</td><td>±0.002</td><td>±0.4</td><td>±1.3</td><td></td><td></td></tr></table>

world-model updates (thousands)

Figure 6 shows the equal-budget world models against the number of sampling steps, along long rollouts and during training. The temporal world models lose less quality when the number of sampling steps is reduced, and keep a lower perceptual error along the rollout. They also reach the baseline's final validation LPIPS earlier (Appendix E defines this comparison). The jump shared by all models near steps 47 to 49, here and in Figure 5, reflects the number of evaluated videos falling from 322 to 233.

![](images/2c999db8bce09017697334d85a798d05a883f410bf4fa87236fcecc68c5799ff.jpg)  
world-model updates (thousands)  
Figure 6: World models at equal training budget, across sampling steps, rollout steps and training. The pipelines of Table 2 against the number of DDIM sampling steps (top), along the long rollouts (middle) and over world-model training (bottom). Gray shading: prediction steps reached by fewer than half of the videos. Mean and standard error over three training seeds. The temporal objectives improve LPIPS and SSIM in all three rows, and inverse dynamics also gives a lower pixel MSE than either reconstruction-only pipeline.

## G.3 AUTOENCODER LOSSES OVER TRAINING

Figure 7 tracks two errors during autoencoder fine-tuning in the weight sweep: the reconstruction error (pixel mean absolute error plus LPIPS) and the error of the jointly trained predictor, relative to the mean magnitude of the latent. This prediction error differs from the unpredictable share of Figure 3, which comes from a separate linear predictor fitted to each frozen latent space (Appendix E) At every nonzero weight and throughout training, the past-conditioned decoder lowers both the reconstruction error (by 13 to 26%) and the prediction error (by 3 to 35%). Increasing the weight lowers the prediction error and raises the reconstruction error.

![](images/3010eab4b12eb314114d4680a7e8b23074e331201f9b8da442ce1df890ef18db.jpg)

![](images/8dcd5114bb1de540c75c54edf752e2304fe9954eb64e219117cc5e217063dcdb.jpg)  
Figure 7: Autoencoder losses over fine-tuning in the weight sweep. Reconstruction error (left) and error of the jointly trained latent predictor (right), for each temporal weight and decoder. One training seed; the predictor receives no gradient at λ = 0. Past conditioning lowers both errors at every nonzero weight, and larger weights trade reconstruction for a lower prediction error.

## G.4 INTERVENTIONS ON THE PAST-CONDITIONED DECODER

To see whether the improvement from past conditioning depends on the information in the context frame, we evaluate the autoencoders of the weight sweep after 80,000 updates in three ways (Table 7) with the correct past frame, with the context pathway disabled, and with past frames permuted within each evaluation batch. The error is the reconstruction error of fine-tuning (pixel mean absolute error plus LPIPS) on the 6,872 held-out frame pairs. Disabling the context bypasses the context encoder and the cross-attention adapters, and with the permutation, each target latent is decoded with the past frame of another example. Disabling or replacing the context raises the reconstruction error at every weight, most at $\lambda = 0 . { \overset { } 7 }$ . This is the behavior the design of Section 5.2 aims for: the decoder takes from the past frame what the latent does not carry, and does so more at larger weights. The same pattern holds for a second training seed at weights 0 and 0.3.

Table 7: Dependence of the past-conditioned decoder on its context. Reconstruction error of the past-conditioned autoencoders of the weight sweep with three contexts; the last column gives the autoencoder trained with the ordinary decoder at the same weight. One training seed. The correct context lowers the error at every weight, and the dependence on the context is largest at $\lambda = 0 . 7$
<table><tr><td></td><td colspan="4">Reconstruction error ↓</td></tr><tr><td>λ</td><td>Past frame</td><td>Context disabled</td><td>Permuted past frame</td><td>Ordinary decoder</td></tr><tr><td>0</td><td>0.0231</td><td>0.0407</td><td>0.0394</td><td>0.0294</td></tr><tr><td>0.1</td><td>0.0237</td><td>0.0430</td><td>0.0424</td><td>0.0314</td></tr><tr><td>0.3</td><td>0.0257</td><td>0.0496</td><td>0.0481</td><td>0.0345</td></tr><tr><td>0.5</td><td>0.0288</td><td>0.0700</td><td>0.0565</td><td>0.0379</td></tr><tr><td>0.7</td><td>0.0375</td><td>0.1947</td><td>0.1072</td><td>0.0451</td></tr></table>

## G.5 POLICY EVALUATION IN WORLDGYM

WorldGym measures whether a world model preserves the differences in policy performance measured on the real robot. The evaluation is closed-loop: each policy chooses its next action from the frames that the world model generates, so different world models can induce different action sequences.

![](images/3a44d18cc76442f574a1dd136a22a8cf2932884aeed6bef16d6eee36712b62ed.jpg)  
Figure 8: Policy evaluation inside the world models. Three robot policies on the 17 Bridge tasks of WorldGym, inside the five world models of Table 2. One training seed; error bars: standard error over the 170 starting frames (left) and bootstrap standard error over tasks (right). With 2 sampling steps, the temporal world models keep their agreement with the real world and the real ranking of the three policies, while the reconstruction-only world models lose both.

We run the Bridge suite of WorldGym (Quevedo et al., 2026): 17 tasks with 10 starting frames each. From each starting frame, a policy acts for 40 frames inside the world model, three times (once in WorldGym). The world models are the five pipelines of Table 2, with one training seed, at 100,000 updates. They sample with classifier-free guidance of scale 3 and 10 or 2 sampling steps. The real-world reference is the per-task success rate of each policy reported by Kim et al. (2025), as used by WorldGym.

Judge. The judge is Claude Opus 5.5 at high reasoning effort, in place of WorldGym's GPT-4o, with WorldGym's prompt and, as in WorldGym's released code, frames 0, 20 and 40 of a rollout. It answers 1 for success, 0 for failure, and 0.5 for partial success on the tasks whose criteria define one. It is called three times per rollout (five times in WorldGym). The most frequent answer is the outcome, with ties going to the higher score, and a success rate is the mean outcome, so a partial success counts one half, as in the real-world rates. We checked the judge on 48 real robot episodes of the AutoEval evaluation set with human success labels, six successes and six failures from each of its four tasks. With three calls per episode, as in the rollouts, the majority vote agrees with the labels on 42 episodes: all 24 successes and 18 of the 24 failures. The six errors are failures judged as successes.

Uncertainty. The bootstrap resamples the 17 tasks with replacement, 10,000 times, keeping the three policies of a task together. The standard error of a correlation, Pearson or Spearman, is the standard deviation of its resampled values. Differences between two world models are computed on the same resampled tasks, with their standard errors and two-sided p-values. These quantities describe the sampling of the tasks for one trained world model; unlike the standard errors of the other evaluations, they do not include the variability between training seeds.

Results. Table 8 gives the agreement per policy and, for the pooled agreement, the difference from the baseline. With 10 sampling steps, latent prediction has the highest pooled correlation with real-world success. With 2 steps, both temporal world models keep a much higher pooled agreement than the reconstruction-only world models. For the reconstruction-only world models, the correlation within each policy changes by at most 0.14 from 10 to 2 steps, while the pooled one falls by 0.30 to 0.36. The drop therefore comes from the comparison between policies: at 2 steps, these world models rank Octo above OpenVLA, the reverse of the real-world ranking, whereas both temporal world models keep the real ranking of the three policies (Figure 8). Appendix G.7 evaluates the world models continued to 200,000 updates.

Table 8: Agreement with real-world policy evaluation at 100,000 updates. Pearson r and Spearman ρ between simulated and real success rates over the 17 tasks of each policy, or over the 51 policy-task pairs when pooled; ∆r and p: paired difference from the baseline (-) and its two-sided bootstrap p-value. One training seed; bootstrap standard errors over tasks under the values. With two sampling steps, both temporal world models agree with real-world performance better than all reconstruction-only world models when the three policies are pooled.
<table><tr><td rowspan="2"></td><td colspan="2">OpenVLA</td><td colspan="2">Octo</td><td colspan="2">RT-1-X</td><td colspan="4">Three policies pooled</td></tr><tr><td>r↑</td><td>ρ↑</td><td>r↑</td><td>ρ↑</td><td>r↑</td><td>ρ↑</td><td>r ↑</td><td>ρ↑</td><td>∆r↑</td><td>p</td></tr><tr><td colspan="10">10 sampling steps</td></tr><tr><td>Ordinary decoder Reconstruction (baseline)</td><td>0.40</td><td>0.27</td><td>0.58</td><td>0.53</td><td>0.11</td><td>0.19</td><td>0.51</td><td>0.45</td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td></td><td>±0.20</td><td>±0.27</td><td>±0.21</td><td>±0.21</td><td>±0.24</td><td>±0.24</td><td>±0.10</td><td>±0.12</td><td></td><td></td></tr><tr><td>Pretrained, no fine-tuning</td><td>0.56</td><td>0.56</td><td>0.58</td><td>0.55</td><td>0.01</td><td>0.10</td><td>0.57</td><td>0.55</td><td>+0.06</td><td>0.261</td></tr><tr><td></td><td>±0.14</td><td>±0.17</td><td>±0.20</td><td>±0.20</td><td>±0.24</td><td>±0.26</td><td>±0.07</td><td>±0.10</td><td>±0.06</td><td></td></tr><tr><td colspan="10">Past-conditioned decoder</td></tr><tr><td>Reconstruction</td><td>0.53</td><td>0.58</td><td>0.50</td><td>0.46</td><td>0.24</td><td>0.49</td><td>0.49</td><td>0.55</td><td>-0.02</td><td>0.623</td></tr><tr><td>Inverse dynamics</td><td>±0.15</td><td>±0.19</td><td>±0.22</td><td>±0.24</td><td>±0.18</td><td>±0.18</td><td>±0.08</td><td>±0.09</td><td>±0.05</td><td></td></tr><tr><td></td><td>0.40</td><td>0.37</td><td>0.60</td><td>0.58</td><td>0.06</td><td>0.21</td><td>0.57</td><td>0.56</td><td>+0.05</td><td>0.346</td></tr><tr><td>Latent prediction</td><td>±0.18</td><td>±0.22</td><td>±0.19</td><td>±0.20</td><td>±0.21</td><td>±0.23</td><td>±0.07</td><td>±0.09</td><td>±0.06</td><td></td></tr><tr><td></td><td>0.54</td><td>0.53</td><td>0.68</td><td>0.66</td><td>0.08</td><td>0.15</td><td>0.62</td><td>0.64</td><td>+0.11</td><td>0.076</td></tr><tr><td></td><td>±0.13</td><td>±0.18</td><td>±0.18</td><td>±0.18</td><td>±0.23</td><td>±0.23</td><td>±0.07</td><td>±0.09</td><td>±0.07</td><td></td></tr><tr><td colspan="10">2 sampling steps Ordinary decoder</td></tr><tr><td>Reconstruction (baseline)</td><td>0.43</td><td>0.36</td><td>0.57</td><td>0.54</td><td>0.21</td><td>0.28</td><td>0.21</td><td>0.30</td><td></td><td></td></tr><tr><td></td><td>±0.17</td><td>±0.23</td><td>±0.21</td><td>±0.20</td><td>±0.24</td><td>±0.23</td><td>±0.07</td><td>±0.09</td><td></td><td></td></tr><tr><td>Pretrained, no fine-tuning</td><td>0.47</td><td>0.43</td><td>0.57</td><td>0.58</td><td>0.13</td><td>0.36</td><td>0.21</td><td>0.30</td><td>-0.00</td><td>0.948</td></tr><tr><td></td><td>±0.16</td><td>±0.23</td><td>±0.21</td><td>±0.21</td><td>±0.18</td><td>±0.21</td><td>±0.09</td><td>±0.10</td><td>±0.05</td><td></td></tr><tr><td colspan="10">Past-conditioned decoder</td></tr><tr><td>Reconstruction</td><td>0.40</td><td>0.42</td><td>0.54</td><td>0.54</td><td>0.30</td><td>0.45</td><td>0.15</td><td>0.28</td><td>-0.06</td><td>0.007</td></tr><tr><td></td><td>±0.15</td><td>±0.21</td><td>±0.20</td><td>±0.20</td><td>±0.25</td><td>±0.20</td><td>±0.07</td><td>±0.09</td><td>±0.02</td><td></td></tr><tr><td>Inverse dynamics</td><td>0.36</td><td>0.35</td><td>0.57</td><td>0.44</td><td>0.02</td><td>0.18</td><td>0.51</td><td>0.47</td><td>+0.30</td><td>&lt;0.001</td></tr><tr><td></td><td>±0.20</td><td>±0.24</td><td>±0.21</td><td>±0.23</td><td>±0.22</td><td>±0.24</td><td>±0.09</td><td>±0.11</td><td>±0.06</td><td></td></tr><tr><td>Latent prediction</td><td>0.55</td><td>0.49</td><td>0.57</td><td>0.54</td><td>0.56</td><td>0.63</td><td>0.67</td><td>0.69</td><td>+0.45</td><td>&lt;0.001</td></tr><tr><td></td><td>±0.14</td><td>±0.20</td><td>±0.21</td><td>±0.21</td><td>±0.18</td><td>±0.17</td><td>±0.07</td><td>±0.09</td><td>±0.07</td><td></td></tr></table>

## G.6 QUALITATIVE EXAMPLES

Rollouts. Figures 10 to 17 show rollouts of the five world models of Table 2 (one training seed, 100,000 updates) in two evaluation conditions of Appendix E, each with 2 and with 10 sampling steps. Out of distribution, the clips are the 479 AutoEval trajectories, and each world model replays the recorded actions for 29 frames from the context frame, one frame at a time. In distribution, the clips are the 224 held-out Bridge videos with at least 60 frames of the long-rollout evaluation, and each world model rolls out 59 frames, eight at a time. All world models roll out the same clip from the same context frame, with the same actions and the same diffusion noise. The noise differs from that of the quantitative evaluation, so the LPIPS shown beside each row is that of the rollout shown. For each condition, one figure shows two selected clips and one shows two clips drawn at random.

All clips. Table 9 places the examples in the distribution over all clips. Both temporal world models have a lower LPIPS than the baseline on 92.0% to 100% of the clips in every condition, with median reductions of 13.1% to 31.8%. The gap is largest with 2 sampling steps: out of distribution, latent

18

29

12  
![](images/7d2d6d8475331848023c04c7f480f5e18df25454aaaf3f1f241bba0d248000cb.jpg)  
Figure 9: AutoEval rollouts with video world models trained on different latent spaces. The baseline learns its latent space through reconstruction, as is standard; our recipe adds a temporal objective (latent prediction or inverse dynamics) and a past-conditioned decoder. One AutoEval trajectory, out of distribution: the recorded actions are replayed from one context frame for three times the length of the training clips, with 2 sampling steps. Figures 10 to 17 show more examples, selected and random.

prediction has a lower LPIPS than the baseline on all 479 trajectories, and both temporal world models are below all three reconstruction-only world models on 96.2% of them. With 10 steps, the reconstruction-only world models degrade more slowly than with 2 steps, and the differences appear mainly late in the long rollouts.

Table 9: Rollout quality over all clips. Lower than baseline: percentage of clips on which the temporal world model has a lower clip-mean LPIPS than reconstruction with the ordinary decoder. Lower than all reconstruction-only: the same against all three reconstruction-only world models; Both: the two temporal world models at once. Median change: relative change of LPIPS against the baseline. ID: the 224 held-out videos with at least 60 frames, rolled out for 59 frames; OOD: the 479 AutoEval trajectories, rolled out for 29 frames; one rollout per clip. On almost every clip, the temporal world models are closer to the recording than the baseline, most consistently with 2 sampling steps.
<table><tr><td colspan="3"></td><td colspan="2"></td><td colspan="3">Lower than baseline (%) Lower than all reconstruction-only (%)</td><td colspan="2">Median change (%)</td></tr><tr><td>Setting</td><td></td><td>Steps Clips Latent</td><td></td><td>Inverse</td><td>Latent Inverse</td><td></td><td>Both</td><td>Latent</td><td>Inverse</td></tr><tr><td>ID</td><td>10</td><td>224</td><td>92.0</td><td>93.3</td><td>80.4</td><td>79.9</td><td>69.6</td><td>-13.1</td><td>-13.6</td></tr><tr><td>ID</td><td>2</td><td>224</td><td>96.0</td><td>98.7</td><td>89.3</td><td>92.9</td><td>87.5</td><td>-21.5</td><td>-21.1</td></tr><tr><td>OOD</td><td>10</td><td>479</td><td>95.0</td><td>96.7</td><td>60.5</td><td>86.2</td><td>53.2</td><td>-13.9</td><td>-22.8</td></tr><tr><td>OOD</td><td>2</td><td>479</td><td>100.0</td><td>99.2</td><td>97.9</td><td>97.9</td><td>96.2</td><td>-28.9</td><td>-31.8</td></tr></table>

Policy evaluation. Figure 18 shows OpenVLA acting inside the world models in WorldGym (Appendix G.5) with 2 sampling steps. Because the loop is closed (Appendix G.5), the rows share the starting frame, the instruction and the policy, but not the actions.

step 24

step 18

recorded video

reconstruction only, O ordinary decoder LPIPS 0.515

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.457

reconstruction only, past-conditioned decoder LPIPS 0.496

latent prediction, ■ past-conditioned decoder LPIPS 0.294

inverse dynamics, 十 past-conditioned decoder LPIPS 0.287

reconstruction only, O ordinary decoder LPIPS 0.487

pretrained autoencoder, X no fine-tuning LPIPS 0.408

reconstruction only, past-conditioned decoder LPIPS 0.411

latent prediction, ■ past-conditioned decoder LPIPS 0.263

inverse dynamics, past-conditioned decoder LPIPS 0.232

out of distribution (AutoEval), 2 sampling steps  
context  
step 1  
step 4  
step 8  
![](images/200dcb99fb7c80c273cae0cc805bba9dae1b0d10da479900d73b4d0b16eeee8c.jpg)  
step 29

(a) “Close the drawer"  
out of distribution (AutoEval), 2 sampling steps  
![](images/f73758d50b76d5f9e87bd90c605bc738de7916892865d7fe764aae8d4c80e905.jpg)  
(b) “Close the drawer”.

Figure 10: Out of distribution, 2 sampling steps, selected trajectories. AutoEval trajectories with the recorded actions replayed for 29 frames. Top row: the recorded frames; below: the rollout of each world model from the same context frame, with its LPIPS.

reconstruction only, O ordinary decoder LPIPS 0.282

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.291

reconstruction only, past-conditioned decoder LPIPS 0.262

latent prediction, ■ past-conditioned decoder LPIPS 0.184

inverse dynamics, past-conditioned decoder LPIPS 0.188

recorded video reconstruction only,   
O ordinary decoder LPIPS 0.372 pretrained autoencoder, no fine-tuning LPIPS 0.363 reconstruction only, past-conditioned decoder LPIPS 0.448 latent prediction,   
■ past-conditioned decoder LPIPS 0.291 inverse dynamics,   
+ past-conditioned decoder LPIPS 0.264

out of distribution (AutoEval), 2 sampling steps  
context  
step 1  
step 4  
step 8  
step 12  
step 18  
![](images/98c48c9fe9673d292e7d12b338a5b2684b42d4b47506667aef120d1ccfad037e.jpg)  
step 24  
step 29  
(a) "Put the eggplant in the blue sink"

out of distribution (AutoEval), 2 sampling steps  
![](images/3a4f659aca49ec566d127c005030c58cb31ace6d05f44ffc648d47765d048254.jpg)  
(b) “Put the eggplant in the yellow basket".

Figure 11: Out of distribution, 2 sampling steps, random trajectories. AutoEval trajectories with the recorded actions replayed for 29 frames; rows as in Figure 10.

recorded video

reconstruction only, O ordinary decoder LPIPS 0.525

pretrained autoencoder, X no fine-tuning LPIPS 0.600

reconstruction only past-conditioned decoder LPIPS 0.523

latent prediction, ■ past-conditioned decoder LPIPS 0.321

inverse dynamics, past-conditioned decoder LPIPS 0.332

in distribution (Bridge), 2 sampling steps  
context  
step 1  
step 4  
![](images/3f04ac77080a7a1c875f4a5f3f6a401157cd1f6bf5da35c84f12b761c59dceb7.jpg)  
(a)  
step 8  
step 16  
step 1  
context  
step 4  
step 8  
step 16  
step 32  
recorded video

reconstruction only, past-conditioned decoder LPIPS 0.500

latent prediction, ■ past-conditioned decoder LPIPS 0.370

inverse dynamics, past-conditioned decoder LPIPS 0.355

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.589

reconstruction only, O ordinary decoder LPIPS 0.534

step 32  
step 59  
step 48  
step 48  
in distribution (Bridge), 2 sampling steps  
step 59  
(b)  
![](images/fc216e6799a27eb9d26d832a93f295c7d590a4a6a973eede4cba92814083a3b6.jpg)  
Figure 12: In distribution, 2 sampling steps, selected videos. Held-out Bridge videos rolled out for 59 frames; rows as in Figure 10.

context

step 1

step 4

step 8

step 16

step 32

step 48

step 59

recorded video

reconstruction only, O ordinary decoder LPIPS 0.519

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.492

reconstruction only, past-conditioned decoder LPIPS 0.493

latent prediction, ■ past-conditioned decoder LPIPS 0.442

inverse dynamics, past-conditioned decoder LPIPS 0.448

in distribution (Bridge), 2 sampling steps  
![](images/53a588028e1d0ee3b17b5934e9df8c8c63b6e736eee2b43db460c33a3e5e74a2.jpg)  
context  
in distribution (Bridge), 2 sampling steps  
(a)  
step 8  
step 1  
step 16  
step 4  
step 32  
step 48  
step 59  
recorded video

latent prediction, ■ past-conditioned decoder LPIPS 0.437

reconstruction only, past-conditioned decoder LPIPS 0.557

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.533

inverse dynamics, past-conditioned decoder LPIPS 0.425

reconstruction only, O ordinary decoder LPIPS 0.513

![](images/13bc74f8bcec63cc48072ab911aa8cbe4f19c9e7f7216ec95d60ca515074c089.jpg)  
(b)  
Figure 13: In distribution, 2 sampling steps, random videos. Held-out Bridge videos rolled out for 59 frames; rows as in Figure 10.

recorded video

reconstruction only, O ordinary decoder LPIPS 0.418

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.386

reconstruction only, past-conditioned decoder LPIPS 0.324

latent prediction, ■ past-conditioned decoder LPIPS 0.259

inverse dynamics, + past-conditioned decoder LPIPS 0.238

reconstruction only, O ordinary decoder LPIPS 0.460

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.294

reconstruction only, past-conditioned decoder LPIPS 0.265

latent prediction, ■ past-conditioned decoder LPIPS 0.255

inverse dynamics, past-conditioned decoder LPIPS 0.222

step 29  
out of distribution (AutoEval), 10 sampling steps  
step 8  
![](images/727a6c02face88328ef55af761081a2d0d94e7c20e97d5ae7eb4a6b73f66ce0c.jpg)  
step 24  
step 29  
(a) “Close the drawer"

out of distribution (AutoEval), 10 sampling steps  
step 1  
step 8  
step 12  
step 4  
![](images/49f0d8a801b15e8b6295bdf90f3305d675304dce10a08897e81db7eac5096fdd.jpg)  
(b) "Put the eggplant in the blue sink".

Figure 14: Out of distribution, 10 sampling steps, selected trajectories. AutoEval trajectories with the recorded actions replayed for 29 frames; rows as in Figure 10.

reconstruction only, O ordinary decoder LPIPS 0.187

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.210

reconstruction only, past-conditioned decoder LPIPS 0.173

latent prediction, ■ past-conditioned decoder LPIPS 0.173

inverse dynamics, • past-conditioned decoder LPIPS 0.132

recorded video reconstruction only,   
O ordinary decoder LPIPS 0.253 pretrained autoencoder, no fine-tuning LPIPS 0.270 reconstruction only, past-conditioned decoder LPIPS 0.220 latent prediction,   
■ past-conditioned decoder LPIPS 0.246 inverse dynamics,   
+ past-conditioned decoder LPIPS 0.191

out of distribution (AutoEval), 10 sampling steps  
context  
step 1  
step 4  
step 8  
step 12  
step 18  
step 24  
step 29  
![](images/f5f5d24ce20d69cc5c0f8a13011ecbcdff558d1f95f56da43464f94ca9da3512.jpg)  
(a) “Put the eggplant in the blue sink".

out of distribution (AutoEval), 10 sampling steps  
![](images/c809551f18edabefb6bb6fc1a427513b43cb58531deef3f16be895e2e59586f4.jpg)  
(b) “Open the drawer".

Figure 15: Out of distribution, 10 sampling steps, random trajectories. AutoEval trajectories with the recorded actions replayed for 29 frames; rows as in Figure 10.

context

step 1

step 8

step 16

step 32

step 48

step 4

step 59

reconstruction only, O ordinary decoder LPIPS 0.309

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.255

reconstruction only, past-conditioned decoder LPIPS 0.216

latent prediction, ■ past-conditioned decoder LPIPS 0.182

inverse dynamics, past-conditioned decoder LPIPS 0.167

latent prediction, ■ past-conditioned decoder LPIPS 0.345

inverse dynamics, + past-conditioned decoder LPIPS 0.356

reconstruction only, past-conditioned decoder LPIPS 0.460

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.426

recorded video  
in distribution (Bridge), 10 sampling steps  
![](images/b0c93fdd9e8c37ef14f01f6d89b6e87a0a38eefcbfb81f1b2f7028f1811ca5fc.jpg)

reconstruction only, O ordinary decoder LPIPS 0.476

(a)  
context  
in distribution (Bridge), 10 sampling steps  
step 1  
step 32  
step 16  
step 4  
step 8  
step 48  
step 59  
![](images/f4cb7d3701e2068a7afff4a2a36f2562d07eb447908cfb792760f94750415fcd.jpg)  
(b)  
Figure 16: In distribution, 10 sampling steps, selected videos. Held-out Bridge videos rolled out for 59 frames; rows as in Figure 10.

# in distribution (Bridge), 10 sampling steps

context

step 1

step 4

step 8

step 16

step 32

step 48

step 59

recorded video

reconstruction only, O ordinary decoder LPIPS 0.194

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.200

reconstruction only, past-conditioned decoder LPIPS 0.192

latent prediction, ■ past-conditioned decoder LPIPS 0.154

inverse dynamics, past-conditioned decoder LPIPS 0.171

reconstruction only, past-conditioned decoder LPIPS 0.447

pretrained autoencoder,   
no fine-tuning   
LPIPS 0.469

reconstruction only, O ordinary decoder LPIPS 0.463

latent prediction, ■ past-conditioned decoder LPIPS 0.413

inverse dynamics, past-conditioned decoder LPIPS 0.446

![](images/4ed13c1de4c234b5af811a0d42e8457cf0b0b0557a703b95fdd6d80cee1ed835.jpg)  
recorded video

(a)  
context  
in distribution (Bridge), 10 sampling steps  
step 1  
step 16  
step 4  
step 8  
step 32  
step 48  
step 59  
![](images/72699bebc454450f08ef19407fcbbc76a024b2411ea8aa5f24c8e1fa1e34715b.jpg)  
(b)  
Figure 17: In distribution, 10 sampling steps, random videos. Held-out Bridge videos rolled out for 59 frames; rows as in Figure 10.

reconstruction only, O ordinary decoder judged: failure (votes 0, 0, 0)

pretrained autoencoder, 3 no fine-tuning judged: failure (votes 0, 0, 0)

reconstruction only,   
past-conditioned decoder   
judged: failure   
(votes 0, 0, 0)   
latent prediction,   
past-conditioned decoder   
judged: success   
(votes 1, 1, 1)

inverse dynamics, + past-conditioned decoder judged: success (votes 1, 1, 1)

reconstruction only,   
O ordinary decoder judged: failure (votes 0, 0, 0) pretrained autoencoder, no fine-tuning judged: failure (votes 0, 0, 0) reconstruction only, past-conditioned decoder judged: failure (votes 0, 0, 0) latent prediction,   
■ past-conditioned decoder judged: success (votes 1, 1, 0) inverse dynamics,   
+ past-conditioned decoder judged: failure (votes 0, 0, 0)

OpenVLA, 2 sampling steps: “put eggplant into pot or pan"  
frame 0  
frame 10  
frame 20  
frame 30  
frame 40  
![](images/edc4846144460734b97646e213c9c66440818360b2bc7f862c46a6258da83aa8.jpg)  
(a) Selected.

OpenVLA, 2 sampling steps: "lift AAA battery"  
![](images/c5ce48294264ac702f877c7c2e82896af98a8e6cdac805fe37e7d4d7b2a4aa08.jpg)  
(b) Random starting frame.  
Figure 18: OpenVLA in WorldGym with 2 sampling steps. One starting frame per panel, inside the five world models of Table 2 (one training seed), at frames 0 to 40; bold: the frames the judge sees. Beside each row: the judged outcome and the three votes.

## G.7 LONGER WORLD-MODEL TRAINING

The three past-conditioned world models of the equal-budget experiment were continued from 100,000 to 200,000 updates, with the same settings, for two of its training seeds. Table 10 compares them at 200,000 updates, together with the same two seeds at 100,000 updates. Against reconstruction with the same decoder, inverse dynamics keeps its gains in all four evaluation conditions. Latent prediction extends its gain out of distribution and keeps a lower LPIPS and a higher SSIM than reconstruction in all four conditions. On long rollouts its pixel-MSE gain is smaller than at 100,000 updates, and its in-distribution pixel MSE, FID and FVD are higher than those of reconstruction.

Policy evaluation. The world models of one training seed at 200,000 updates were also evaluated in WorldGym, as in Appendix G.5 (Table 11 and Figure 19). The pattern at 100,000 updates holds. With 10 sampling steps, both temporal world models have a higher pooled correlation with real-world success than reconstruction (0.69 and 0.66 against 0.61). With 2 steps, the agreement of reconstruction drops (pooled r from 0.61 to 0.20), while that of the temporal world models changes little (from 0.69 to 0.68 and from 0.66 to 0.60), and they keep the real ranking of the three policies.

![](images/ad86399431d34169846410980a285cd98ee0129a352de441455ab06c5008f8b6.jpg)  
Figure 19: Policy evaluation after 200,000 world-model updates. The three past-conditioned world models of Table 11, one training seed; panels, rows and error bars as in Figure 8. With 2 steps, the temporal world models keep the real ranking of the policies and their agreement with the real world.

Table 10: Longer world-model training. The three past-conditioned pipelines of the equal-budget experiment at 200,000 world-model updates and, for the same two training seeds, at 100,000. Mean and standard error over two training seeds (pixel MSE $\times 1 0 ^ { 3 } )$ ; bold: best in each column at each training length. ID, OOD, 2-step and long-horizon as in Section 5. Inverse dynamics keeps its gains in every condition; latent prediction keeps a lower LPIPS than reconstruction in every condition and extends its out-of-distribution gain, but its in-distribution pixel MSE, FID and FVD are higher.
<table><tr><td rowspan="2"></td><td colspan="3">ID</td><td colspan="3">OOD</td></tr><tr><td>MSE↓</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>MSE↓</td><td>LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td colspan="7">200,000 world-model updates</td></tr><tr><td>Reconstruction</td><td>12.14</td><td>0.163</td><td>0.813</td><td>34.53</td><td>0.280</td><td>0.707</td></tr><tr><td></td><td>±0.36</td><td>±0.001</td><td>±0.002</td><td>±2.33</td><td>±0.006</td><td>±0.006</td></tr><tr><td>Inverse dynamics</td><td>11.56</td><td>0.148</td><td>0.823</td><td>31.19</td><td>0.242</td><td>0.737</td></tr><tr><td></td><td>±0.02</td><td>±0.001</td><td>±0.001</td><td>±1.42</td><td>±0.005</td><td>±0.005</td></tr><tr><td>Latent prediction</td><td>12.85</td><td>0.157</td><td>0.818</td><td>26.00</td><td>0.255</td><td>0.743</td></tr><tr><td></td><td>±0.21</td><td>±0.001</td><td>±0.001</td><td>±0.10</td><td>±0.004</td><td>±0.003</td></tr><tr><td colspan="7">100,000 world-model updates, same training seeds</td></tr><tr><td>Reconstruction</td><td>14.00</td><td>0.179</td><td>0.800</td><td>32.00</td><td>0.276</td><td>0.712</td></tr><tr><td></td><td>±0.24</td><td>±0.000</td><td>±0.001</td><td>±0.19</td><td>±0.002</td><td>±0.001</td></tr><tr><td>Inverse dynamics</td><td>13.31</td><td>0.161</td><td>0.812</td><td>28.71</td><td>0.239</td><td>0.743</td></tr><tr><td></td><td>±0.03</td><td>±0.001</td><td>±0.001</td><td>±1.48</td><td>±0.008</td><td>±0.009</td></tr><tr><td>Latent prediction</td><td>14.10</td><td>0.169</td><td>0.810</td><td>26.89</td><td>0.257</td><td>0.741</td></tr><tr><td></td><td>±0.06</td><td>±0.001</td><td>±0.002</td><td>±0.32</td><td>±0.003</td><td>±0.003</td></tr><tr><td></td><td></td><td>2-step</td><td></td><td></td><td>Long-horizon</td><td></td></tr><tr><td colspan="7">MSE↓ LPIPS ↓ SSIM ↑</td></tr><tr><td>200,000 world-model updates</td><td></td><td></td><td></td><td></td><td>MSE↓ LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td colspan="7">Reconstruction</td></tr><tr><td></td><td>14.89</td><td>0.187</td><td>0.795</td><td>37.27</td><td>0.321</td><td>0.670</td></tr><tr><td></td><td>±0.64</td><td>±0.001</td><td>±0.001</td><td>±0.64</td><td>±0.000</td><td>±0.001</td></tr><tr><td>Inverse dynamics</td><td>12.23</td><td>0.152</td><td>0.822</td><td>33.55</td><td>0.292</td><td>0.694</td></tr><tr><td>Latent prediction</td><td>±0.12</td><td>±0.000</td><td>±0.001</td><td>±0.09</td><td>±0.003</td><td>±0.004</td></tr><tr><td></td><td>13.98</td><td>0.164</td><td>0.814</td><td>35.36</td><td>0.298</td><td>0.693</td></tr><tr><td></td><td>±0.12</td><td>±0.001</td><td>±0.000</td><td>±3.06</td><td>±0.013</td><td>±0.012</td></tr><tr><td colspan="7">100,000 world-model updates, same training seeds</td></tr><tr><td>Reconstruction</td><td>16.81</td><td>0.200</td><td>0.786</td><td>45.71</td><td>0.372</td><td>0.619</td></tr><tr><td></td><td>±1.06</td><td>±0.003</td><td>±0.004</td><td>±0.44</td><td>±0.001</td><td>±0.002</td></tr><tr><td>Inverse dynamics</td><td>13.72</td><td>0.163</td><td>0.814</td><td>40.41</td><td>0.337</td><td>0.648</td></tr><tr><td></td><td>±0.03</td><td>±0.001</td><td>±0.000</td><td>±3.00</td><td>±0.006</td><td>±0.011</td></tr><tr><td>Latent prediction</td><td>15.38</td><td>0.177</td><td>0.805</td><td>39.99</td><td>0.336</td><td>0.657</td></tr><tr><td></td><td>±0.00</td><td>±0.001</td><td>±0.002</td><td>±0.98</td><td>±0.009</td><td>±0.010</td></tr><tr><td></td><td>Distribution</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">FID↓</td></tr><tr><td>200,000 world-model updates</td><td></td><td>FVD↓</td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction</td><td>12.1</td><td>28.4</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.0</td><td>±0.7</td><td></td><td></td><td></td><td></td></tr><tr><td>Inverse dynamics</td><td>8.9</td><td>28.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Latent prediction</td><td>±0.3</td><td>±0.2</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>14.2</td><td>37.7</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.7</td><td>±1.4</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">100,000 world-model updates, same training seeds</td></tr><tr><td>Reconstruction</td><td>18.3</td><td>36.6</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.2</td><td>±0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Inverse dynamics</td><td>11.4</td><td>33.5</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.3</td><td>±1.8</td><td></td><td></td><td></td><td></td></tr><tr><td>Latent prediction</td><td>21.3</td><td>46.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.6</td><td>±2.2</td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Agreement with real-world policy evaluation at 200,000 updates. The three pastconditioned world models of Table 10, one training seed; columns as in Table 8, with ∆r and p against reconstruction (-). With two sampling steps, both temporal world models agree with real-world performance better than reconstruction when the three policies are pooled; with ten, the differences are smaller.
<table><tr><td rowspan="3"></td><td colspan="2">OpenVLA</td><td colspan="2">Octo</td><td colspan="2">RT-1-X</td><td colspan="4">Three policies pooled</td></tr><tr><td>r↑</td><td>ρ↑</td><td>r ↑</td><td>ρ↑</td><td>r ↑</td><td>ρ↑</td><td>r↑</td><td>ρ↑</td><td>∆r↑</td><td>p</td></tr><tr><td>10 sampling steps</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reconstruction</td><td>0.50 ±0.16</td><td>0.45 ±0.22</td><td>0.55 ±0.21</td><td>0.53 ±0.21</td><td>0.24 ±0.23</td><td>0.41 ±0.21</td><td>0.61 ±0.08</td><td>0.61 ±0.10</td><td></td><td></td></tr><tr><td>Inverse dynamics</td><td>0.54 ±0.15</td><td>0.47</td><td>0.57</td><td>0.53</td><td>0.34</td><td>0.30</td><td>0.66</td><td>0.65</td><td>+0.05</td><td>0.081</td></tr><tr><td>Latent prediction</td><td>0.61 ±0.14</td><td>±0.20 0.58</td><td>±0.21 0.53</td><td>±0.21 0.52</td><td>±0.19 0.56</td><td>±0.19 0.58</td><td>±0.07 0.69</td><td>±0.10 0.68</td><td>±0.03 +0.07</td><td>0.086</td></tr><tr><td>2 sampling steps</td><td></td><td>±0.20</td><td>±0.21</td><td>±0.21</td><td>±0.15</td><td>±0.19</td><td>±0.08</td><td>±0.10</td><td>±0.04</td><td></td></tr><tr><td>Reconstruction</td><td>0.44</td><td>0.40</td><td>0.53</td><td>0.53</td><td>0.21</td><td>0.51</td><td>0.20</td><td>0.31</td><td></td><td></td></tr><tr><td></td><td>±0.16</td><td>±0.23</td><td>±0.21</td><td>±0.21</td><td>±0.18</td><td>±0.18</td><td>±0.08</td><td>±0.10</td><td></td><td></td></tr><tr><td>Inverse dynamics</td><td>0.50</td><td>0.46</td><td>0.58</td><td>0.55</td><td>0.55</td><td>0.70</td><td>0.60</td><td>0.61</td><td>+0.40</td><td>&lt;0.001</td></tr><tr><td></td><td>±0.14</td><td>±0.20</td><td>±0.21</td><td>±0.21</td><td>±0.20</td><td>±0.14</td><td>±0.07</td><td>±0.10</td><td>±0.08</td><td></td></tr><tr><td>Latent prediction</td><td>0.56</td><td>0.52</td><td>0.52</td><td>0.40</td><td>0.49</td><td>0.47</td><td>0.68</td><td>0.65</td><td>+0.48</td><td>&lt;0.001</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.14</td><td>±0.21</td><td>±0.22</td><td>±0.25</td><td>±0.20</td><td>±0.21</td><td>±0.07</td><td>±0.11</td><td>±0.08</td><td></td></tr></table>

## G.8 LATENT SPACES

To describe what the temporal objectives change in the latent space, we encode 1,000 held-out frames with the frozen encoders of both experiments: ten consecutive frames from each of 100 videos, with a uniformly distributed start frame. We measure four properties of the flattened latents. Their dimensionality is the number of principal components for 95% of the variance, together with the participation ratio of the spectrum. The straightness of their trajectories is the mean angle between successive latent displacements along a video, divided by the same angle in pixel space. Clustering by video is the mean distance between frames of one video divided by that between frames of different videos. For the spatial smoothness of the latent maps, we average the magnitudes of the real two-dimensional Fourier transform of each latent channel over frames and channels. With A(ω) the mean amplitude at rounded radial frequency ω, in cycles per 32 latent positions, we report the least-squares slope of log A(ω) against log ω for $\omega = 1 , \ldots , 1 6 .$ and the high-frequency share $\textstyle \sum _ { \omega = 8 } ^ { 1 6 } A ( \bar { \omega } ) ^ { 2 } / \sum _ { \omega = 1 } ^ { 1 6 } A ( \omega ) ^ { 2 }$ . Figure 22 shows A(ω) divided by its sum over $\omega = 1 , \ldots , 1 6$ . The two experiments use different frames, so their values are not compared with each other. Table 12 gives the statistics, Figure 22 the spectra, and Figures 20 and 21 the latent maps of a single frame.

Latent prediction changes the latent space in the same direction in both experiments. It reduces the number of principal components needed for 95% of the variance (177 against 388 for the baseline at equal reconstruction; at equal budget, 110 against 316 for reconstruction with the same decoder and 345 for the baseline), the participation ratio, the turning-angle ratio, the high-frequency content and the within-to-across-video distance ratio. Contrastive prediction shifts every statistic in the same direction, and inverse dynamics every one except clustering by video. Pixel prediction shifts them in the opposite direction, with 445 principal components. With reconstruction alone, past conditioning produces smaller changes in the same direction as the temporal objectives (316 against 345 principal components). The latent maps and spectra show the smoother spatial structure under the temporal objectives (Figures 20 to 22).

![](images/b51737f5e2f99ebdb83fef7c6f02d3f7df56f51d62517be9a03d76fcd18db15a.jpg)

Figure 20: Latent maps at equal training budget. One held-out Bridge frame encoded by the encoders of the three past-conditioned pipelines, one training seed. Columns average channels 0–4, 5–9 and 10–15 of each $1 6 \times 3 2 \times 3 2$ latent. Each map is standardized separately, black low and white high, so intensities are not comparable across maps. Latent prediction and inverse dynamics produce smoother regions around the arm and the objects than reconstruction alone  
![](images/823148208f435d50e70d9b943623854bbd8ca415ac2b37eeaac821215ebe8739.jpg)  
Figure 21: Latent maps at equal reconstruction quality. The same frame encoded by the four encoders of the equal-reconstruction experiment, all with the ordinary decoder, one training seed. Columns and standardization as in Figure 20. Latent and contrastive prediction suppress fine spatial variation compared with reconstruction alone, whereas pixel prediction retains it.

Table 12: Geometry of the frozen latent spaces. Statistics of the latents in both experiments, defined in the text; lower values mean lower dimension, straighter trajectories, tighter clustering by video and smoother maps. Mean and standard error over three training seeds unless one is stated. Latent prediction reduces dimension and high-frequency content in both experiments, and inverse dynamics also smooths the latents
<table><tr><td></td><td>Components for 95%</td><td>Participation ratio</td><td>Turning- angle ratio</td><td>Within / across video distance</td><td>Spectral slope</td><td>High-frequency spectral share</td></tr><tr><td colspan="7">Equal reconstruction, three seeds</td></tr><tr><td>Reconstruction</td><td>388</td><td>52.7</td><td>1.100</td><td>0.423</td><td>-0.837</td><td>0.073</td></tr><tr><td></td><td>±0.9</td><td>±0.03</td><td>±0.0001</td><td>±0.0004</td><td>±0.0003</td><td>±0.00008</td></tr><tr><td>Latent prediction</td><td>177</td><td>49.4</td><td>1.019</td><td>0.344</td><td>-0.907</td><td>0.061</td></tr><tr><td></td><td>±8</td><td>±1.1</td><td>±0.007</td><td>±0.005</td><td>±0.010</td><td>±0.002</td></tr><tr><td>Pixel prediction</td><td>445</td><td>64.4</td><td>1.110</td><td>0.443</td><td>-0.763</td><td>0.094</td></tr><tr><td>Contrastive prediction</td><td>±0.3</td><td>±0.2</td><td>±0.0004</td><td>±0.001</td><td>±0.0008</td><td>±0.0002</td></tr><tr><td></td><td>293</td><td>46.7</td><td>1.086</td><td>0.384</td><td>-0.933</td><td>0.052</td></tr><tr><td></td><td>±6</td><td>±0.7</td><td>±0.0009</td><td>±0.002</td><td>±0.010</td><td>±0.002</td></tr><tr><td colspan="7">Equal budget, past-conditioned decoder, three seeds</td></tr><tr><td>Reconstruction</td><td>316</td><td>47.7</td><td>1.090</td><td>0.402</td><td>-0.886</td><td>0.063</td></tr><tr><td></td><td>±10</td><td>±0.7</td><td>±0.002</td><td>±0.004</td><td>±0.008</td><td>±0.002</td></tr><tr><td>Latent prediction</td><td>110</td><td>36.0</td><td>0.879</td><td>0.323</td><td>-0.948</td><td>0.049</td></tr><tr><td>Inverse dynamics</td><td>±1</td><td>±0.5</td><td>±0.002</td><td>±0.004</td><td>±0.004</td><td>±0.001</td></tr><tr><td></td><td>239</td><td>44.9</td><td>1.005</td><td>0.464</td><td>-1.020</td><td>0.043</td></tr><tr><td></td><td>±4</td><td>±2.0</td><td>±0.001</td><td>±0.002</td><td>±0.009</td><td>±0.002</td></tr><tr><td colspan="7">Equal budget, ordinary decoder (reconstruction: three seeds; pretrained: one)</td></tr><tr><td>Reconstruction</td><td>345</td><td>49.8</td><td>1.092</td><td>0.407</td><td>-0.864</td><td>0.067</td></tr><tr><td></td><td>±12</td><td>±0.8</td><td>±0.002</td><td>±0.005</td><td>±0.009</td><td>±0.002</td></tr><tr><td>Pretrained, no fine-tuning</td><td>383</td><td>52.5</td><td>1.099</td><td>0.421</td><td>-0.838</td><td>0.072</td></tr></table>

![](images/c5b165861312a9cb3fdb21ca4d932d3b1693db9827a8dcfd9fd32db4fbeb926b.jpg)

![](images/422dc7615b31252e4d8f68dd50c96c78f645db0c032685b97bf317d41de76d0f.jpg)  
Figure 22: Spatial spectra of the frozen latents. Radial amplitude spectra of the latent channels, as a share of their total amplitude; at equal budget, solid curves use the past-conditioned decoder and dashed curves the ordinary one. Mean and standard error over three training seeds, except the pretrained encoder (one). Latent and contrastive prediction reduce the high-frequency components at equal reconstruction, and latent prediction and inverse dynamics do so at equal budget.

## G.9 ABLATIONS

The ablations use one training seed unless stated otherwise.

## G.9.1 TEMPORAL WEIGHT

Figure 23 shows the world models on the past-conditioned autoencoders of the weight sweep at $\lambda = 0 . 1 \mathrm { t o } 0 . 5 ,$ and those of the equal-budget experiment at $\lambda = 0 . 7 .$ against reconstruction with the same decoder $( \lambda = 0 )$ . Lower weights favor the distribution metrics and larger weights long rollouts: at $\lambda = 0 . 1$ to 0.5, latent prediction gives FID 12.8 to 14.3 and FVD 31.8 to 33.4, against 18.5 and 36.5 for reconstruction with the same decoder and seed (Figure 24).

![](images/2d0fc2ecd038b89160e4ed1cd064dcbb9b52da9cd9b087e6e9ed23760c011b89.jpg)  
Figure 23: World-model improvement over reconstruction as the temporal weight varies. Latent prediction at each weight against reconstruction $( \lambda = 0 ,$ , zero by construction, open marker), both with the past-conditioned decoder and the same training seed; panels have independent scales. One training seed, and two more at $\lambda = 0 . 7$ (dots). LPIPS improves at every weight in distribution, out of distribution and with two sampling steps; long-rollout gains favor larger weights.

![](images/5c069bd483d6e33ef8a8f6c77e77706d9555d15fb5145454331ceb077f727f1e.jpg)  
Figure 24: Distribution metrics as the temporal weight varies. FID and FVD of latent prediction at each weight against reconstruction $( \lambda = 0 )$ , both with the past-conditioned decoder and the same training seed, in distribution; panels as in Figure 23. One training seed, and two more at $\lambda = 0 . 7$ (dots). Weights from 0.1 to 0.5 improve both metrics; at $\lambda = 0 . 7$ , both are worse than at $\lambda = 0 ,$ for all three training seeds.

## G.9.2 A SEMANTIC ENCODER

We replace the FLUX encoder by DINOv2-Base (Oquab et al., 2024) with registers (Darcet et al., 2024), and use the pretrained ViT-XL decoder of RAE (Zheng et al., 2026) with RAE's released latent statistics. Each frame is resized to $2 2 4 \times 2 2 4$ with bicubic interpolation and normalized with the ImageNet statistics; the encoder produces a $7 6 8 \times 1 6 \times 1 6$ latent, and the decoder reconstructs at $2 5 6 \times 2 5 6$ . We make the decoder past-conditioned with seven zero-initialized cross-attention adapters with eight heads, after layers 1, 5, 9, 13, 17, 21 and 25 of its 28. Their 256 context tokens, of width 768, combine projected image patches with projected features from a context encoder, a frozen copy of the pretrained DINOv2 encoder, as in the FLUX decoder. The encoder, the decoder and the adapters are fine-tuned with the equal-budget settings (Table 3), at latent-prediction weights 0, 0.1, 0.3 and 0.7. The latent predictor is that of Appendix C, with 768 input and output channels.

The world model follows RAE for these latents. It uses one token per latent position, a backbone whose width equals the token dimension (12 blocks of width 768, 12 heads), RAE's wide output head (two blocks of width 2,048, 16 heads), and standardization of every channel and position. It also uses RAE's dimension-dependent shift of the noise level, $\kappa \mapsto 7 \kappa / ( 1 + 6 \kappa )$ for the normalized noise index $\kappa \in [ 0 , 1 ]$ , in training and on the DDIM grid. Unlike RAE, it predicts the clean latent, with the loss computed on the velocity (Li & He, 2026). Its optimization, the evaluation sets, the guidance scale and the numbers of sampling steps are those of the equal-budget experiment. $\mathrm { A t } \lambda = 0 . 1$ , latent prediction improves every metric (Table 13). $\mathrm { { A t } } \lambda = 0 . 3$ and 0.7, latent prediction lowers the latent MSE further and still improves out of distribution, in every metric at 0.3 and in pixel MSE at 0.7. At both weights, it worsens the framewise metrics in distribution, with two sampling steps and on long rollouts, and the distribution metrics. With a semantic encoder, a small weight is therefore preferable. This world model differs from those of the FLUX experiments, so the values of Table 13 are not compared with those of Table 6.

Table 13: Latent prediction with a DINOv2 encoder. DINOv2 encoder and past-conditioned RAE decoder; one training seed (pixel MSE $\times 1 0 ^ { 3 } ) ;$ bold: best in each column. ID, OOD, 2-step and long-horizon as in Section 5. $\mathrm { { A t } } \lambda = 0 . 1$ , latent prediction improves every metric; larger weights lower the latent MSE further but worsen most other errors and the distribution metrics.
<table><tr><td rowspan="2"></td><td colspan="3">ID</td><td colspan="3">OOD</td></tr><tr><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>MSE↓</td><td>LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td>Reconstruction</td><td>10.78</td><td>0.158</td><td>0.797</td><td>35.11</td><td>0.278</td><td>0.698</td></tr><tr><td>Latent prediction, λ = 0.1</td><td>10.57</td><td>0.150</td><td>0.805</td><td>30.31</td><td>0.257</td><td>0.715</td></tr><tr><td>Latent prediction, λ = 0.3</td><td>11.67</td><td>0.163</td><td>0.790</td><td>29.68</td><td>0.271</td><td>0.704</td></tr><tr><td>Latent prediction, λ = 0.7</td><td>11.98</td><td>0.178</td><td>0.763</td><td>28.21</td><td>0.299</td><td>0.673</td></tr><tr><td></td><td colspan="3">2-step</td><td colspan="3">Long-horizon</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>MSE↓</td><td>LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td>Reconstruction</td><td>11.01</td><td>0.160</td><td>0.800</td><td>30.94</td><td>0.289</td><td>0.687</td></tr><tr><td>Latent prediction, λ = 0.1</td><td>10.82</td><td>0.151</td><td>0.807</td><td>26.75</td><td>0.265</td><td>0.716</td></tr><tr><td>Latent prediction, λ = 0.3</td><td>12.19</td><td>0.165</td><td>0.792</td><td>32.39</td><td>0.292</td><td>0.685</td></tr><tr><td>Latent prediction, λ = 0.7</td><td>13.10</td><td>0.185</td><td>0.761</td><td>33.15</td><td>0.298</td><td>0.667</td></tr><tr><td></td><td colspan="2">Predictability</td><td colspan="2">Distribution</td><td></td><td></td></tr><tr><td></td><td>ID latent MSE ↓</td><td>OOD latent MSE↓</td><td>FID↓</td><td>FVD↓</td><td></td><td></td></tr><tr><td>Reconstruction</td><td>0.522</td><td>0.531</td><td>8.7</td><td>26.0</td><td></td><td></td></tr><tr><td>Latent prediction, λ = 0.1</td><td>0.156</td><td>0.168</td><td>7.5</td><td>21.5</td><td></td><td></td></tr><tr><td>Latent prediction, λ = 0.3</td><td>0.058</td><td>0.065</td><td>13.2</td><td>36.4</td><td></td><td></td></tr><tr><td>Latent prediction, λ = 0.7</td><td>0.026</td><td>0.033</td><td>22.8</td><td>56.2</td><td></td><td></td></tr></table>

## G.9.3 WORLD-MODEL SIZE

On a second pair of autoencoders, trained with the equal-budget settings (reconstruction, and latent prediction at $\lambda = 0 . 7 ;$ one training run each), we train world models of width 384 with 12 blocks and of width 1,024 with 16 blocks (Figure 25). Out of distribution and with two sampling steps, latent prediction improves the world model at both widths. In distribution, its effect is mixed at width 384 and negative at width 1,024, where its long-rollout gains also shrink

![](images/0544e8270fbf4910ce7f0fd062fc81e2fdea66c35beaa47589fc487835625459.jpg)  
improvement of latent prediction over reconstruction, same decoder and seed (%)  
Figure 25: Effect of latent prediction across world-model sizes. Improvement of latent prediction over reconstruction, both with the past-conditioned decoder and the same seed: relative reductions of pixel MSE and LPIPS, and increases of SSIM. Rows of the equal-budget autoencoders show the mean and standard error over training seeds, with dots for single seeds; rows of the second autoencoder pair show one run each. The gains persist out of distribution and with two sampling steps at both widths; in distribution, the effect is mixed at width 384 and negative at width 1,024, where the long-rollout gains also shrink.

## G.9.4 KL PENALTY

Our autoencoders are fine-tuned without the KL penalty of latent diffusion (Appendix B). We add it with weight 10-6to the equal-budget settings, with and without latent prediction (Table 14). With the penalty, latent prediction still improves on reconstruction in every framewise metric and in FID, most in out-of-distribution pixel MSE. Its FVD is higher. The penalty itself improves reconstruction in every metric except the out-of-distribution pixel MSE.

Table 14: KL penalty. Reconstruction and latent prediction at λ = 0.7, both with the past-conditioned decoder, with and without a KL penalty on the autoencoder; one training seed (pixel MSE × 103); bold: best in each column. ID, OOD, 2-step and long-horizon as in Section 5. With the penalty, latent prediction still improves every framewise metric and FID over reconstruction with the penalty, while its FVD is higher.
<table><tr><td></td><td colspan="3">ID</td><td colspan="3">OOD</td></tr><tr><td></td><td>MSE↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>MSE↓</td><td>LPIPS ↓</td><td>SSIM↑</td></tr><tr><td colspan="7">No KL penalty</td></tr><tr><td>Reconstruction</td><td>13.76</td><td>0.178</td><td>0.801</td><td>31.81</td><td>0.274</td><td>0.713</td></tr><tr><td>Latent prediction</td><td>14.16</td><td>0.170</td><td>0.808</td><td>26.58</td><td>0.260</td><td>0.738</td></tr><tr><td colspan="7">KL penalty, weight 10−6</td></tr><tr><td>Reconstruction</td><td>13.71</td><td>0.169</td><td>0.810</td><td>36.35</td><td>0.258</td><td>0.725</td></tr><tr><td>Latent prediction</td><td>13.55</td><td>0.164</td><td>0.814</td><td>25.27</td><td>0.241</td><td>0.751</td></tr><tr><td></td><td></td><td>2-step</td><td></td><td></td><td>Long-horizon</td><td></td></tr><tr><td></td><td colspan="2">MSE↓ LPIPS ↓</td><td>SSIM↑</td><td colspan="2">MSE↓ LPIPS↓</td><td>SSIM↑</td></tr><tr><td colspan="7">No KL penalty</td></tr><tr><td>Reconstruction</td><td>17.87</td><td>0.203</td><td>0.782</td><td>45.27</td><td>0.373</td><td>0.620</td></tr><tr><td>Latent prediction</td><td>15.38</td><td>0.178</td><td>0.804</td><td>40.98</td><td>0.344</td><td>0.647</td></tr><tr><td colspan="7">KL penalty, weight 10−6</td></tr><tr><td>Reconstruction</td><td>16.67</td><td>0.187</td><td>0.796</td><td>43.57</td><td>0.364</td><td>0.627</td></tr><tr><td>Latent prediction</td><td>14.23</td><td>0.170</td><td>0.812</td><td>41.84</td><td>0.348</td><td>0.640</td></tr><tr><td></td><td colspan="2">Distribution</td><td colspan="4"></td></tr><tr><td></td><td colspan="2">FID↓ FVD↓</td><td colspan="4"></td></tr><tr><td colspan="8">No KL penalty</td></tr><tr><td>Reconstruction</td><td>18.5</td><td>36.5</td><td colspan="4"></td></tr><tr><td>Latent prediction</td><td>20.7</td><td>43.8</td><td></td><td colspan="4"></td></tr><tr><td colspan="8">KL penalty, weight 10−6</td></tr><tr><td>Reconstruction</td><td>15.4</td><td>35.3</td><td colspan="4"></td></tr><tr><td>Latent prediction</td><td>14.8</td><td>38.2</td><td colspan="4"></td></tr></table>