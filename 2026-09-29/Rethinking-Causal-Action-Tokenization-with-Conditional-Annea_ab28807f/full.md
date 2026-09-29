# Rethinking Causal Action Tokenization with Conditional Annealing in Flow Matching

Chenyu Zhang<sup>1,2∗</sup> Yuhang Cao<sup>1∗</sup> Daru Du<sup>1,2</sup> Yingxi Lu<sup>1,2</sup> Jing Shao<sup>1</sup> Ruoqu Chen<sup>1,2</sup> Jiajun Liu<sup>1,2</sup> Liu Cao<sup>1,2</sup> Yicheng Liu<sup>1</sup> Hang Zhao<sup>1,2</sup> Mengdi Xu<sup>1,2†</sup>

<sup>1</sup>IIIS, Tsinghua University <sup>2</sup> Shanghai Qizhi Institute {cyzhang21@mails, caoyh24@mails, xumd@mail}.tsinghua.edu.cn https://chenyuzhangx.github.io/CATok/

## Abstract

Autoregressive Vision-Language-Action (VLA) models offer a scalable path to robot learning, yet existing action tokenizers treat tokenization as a compression problem, producing representations that are semantically misaligned with the autoregressive backbone. We propose CATOK, a causal action tokenizer that reframes tokenization as a causally structured generative process. CATOK introduces a conditional annealing mechanism that extracts action tokens by progressively annealing a flow-matching process: each token is conditioned on all preceding tokens and encodes the residual reconstruction signal at a specific noise level, establishing a coarse-to-fine causal token space whose generative semantics are structurally aligned with autoregressive modeling. A token-conditionedflow-matching decoder built on Multimodal Diffusion Transformer (MMDiT) reconstructs continuous action chunks from these discrete tokens with the precision of hybrid diffusion-head architectures. This discrete bottleneck enforces knowledge insulation by design, cleanly separating high-level semantic reasoning from low-level motor execution without requiring explicit attention masking. Extensive evaluations across three simulation benchmarks and real-world robotic manipulation tasks demonstrate that CATOK consistently surpasses existing tokenization methods in both reconstruction fidelity–compression tradeoff and inference efficiency, while improving VLA task success rate and training efficiency, establishing a high-performance, scalable foundation for purely autoregressive VLA systems.

## 1 Introduction

Robot foundation models have shown strong generalization across diverse manipulation tasks [1–8]. Vision-Language-Action (VLA) architectures have emerged as a particularly promising paradigm, jointly modeling language instructions, visual observations, and motor commands within a unified policy [1, 2, 5, 6]. Existing VLAs broadly fall into two families: purely autoregressive models that predict discretized action tokens using the same next-token objective as language models [1, 2, 9], and hybrid models that pair an autoregressive language backbone with a continuous diffusion or flow-matching action head [5, 6]. Hybrid models achieve precise continuous control and avoid the bottleneck of naive action discretization, but they decouple action generation from the autoregressive backbone, limiting the unified token-level modeling and scalable LLM training infrastructure that make purely autoregressive VLAs appealing [10–12].

A central challenge for purely autoregressive VLAs is therefore how to tokenize continuous robot actions. The simplest strategy is per-dimension uniform binning [1, 2, 9], where each scalar action dimension is independently mapped to a discrete bin. This produces long token sequences of length O(D×H), ignores correlations across action dimensions, and provides no meaningful causal structure within an action chunk. Recent action tokenizers instead compress entire action chunks into short discrete sequences [13–15], but they remain imperfectly matched to autoregressive generation. FAST [13] uses DCT compression and BPE to obtain compact codes, yet its frequency-ordered coefficients are not naturally aligned with left-to-right autoregressive generation, and its variablelength outputs can make decoding brittle [15]. FASTer [14] learns fixed-length RVQ codes with strong reconstruction fidelity, but optimizing a codebook for reconstruction alone does not ensure compatibility with the language-model backbone. OAT [15] introduces a left-to-right ordering through nested dropout, but the resulting token positions lack semantic grounding, since each position has no principled correspondence to information granularity or generative stage.

These limitations point to a deeper issue: existing action tokenizers mainly encode what action should be reconstructed, but not how that action should be generated. In generative modeling, denoising processes naturally expose a hierarchy of abstraction: high-noise steps capture coarse global structure, while low-noise steps refine fine-grained details [16]. If action tokens were aligned with this hierarchy, then a token sequence could provide a causally ordered, coarse-to-fine representation of continuous control. Such a representation would be better matched to autoregressive generation: early tokens would specify global action structure, and later tokens would refine local motor details. However, existing action tokenizers do not explicitly exploit this correspondence between denoising hierarchy and left-to-right token generation.

Inspired by these observations, we introduce CATOK, a causal action tokenizer that aligns action tokenization with the generative structure of autoregressive models. CATOK represents action chunks as compact discrete tokens and reconstructs them using an MMDiT-based flow-matching decoder [17, 18], achieving the precision of diffusion-based action heads [5] while remaining fully compatible with autoregressive VLA training. To structure the token space, we employ a conditional annealing process in which each token is generated conditioned on its predecessors and corresponds to a stage of the denoising process, resulting in a causally ordered coarse-to-fine representation of actions [16] (illustrated in Figure 1). Experiments on three simulation benchmarks show that CATOK achieves the highest success rates across all benchmarks, while delivering a 1.7× faster VLA inference speed and a 3.6× stronger reconstruction fidelity– compression balance than FAST.

![](images/9bfcab69ac048f9bb946221c70416001fab3e24ac2856f1496c23d8252a6dff0.jpg)  
Figure 1: Flow-Matching to Action Tokens. CATOK grounds discrete action tokens in the continuous flow matching trajectory. Each token is learned as a stage-wise information increment, imbuing the sequence with a natural coarse-to-fine hierarchy. By construction, these increments follow the temporal evolution of the flow, ensuring a causally-ordered structure that seamlessly aligns with autoregressive generation.

Our contributions are three-fold:

• Causal action tokenization via conditional annealing. We formulate action tokenization as a sequential generative process via conditional annealing, producing tokens that follow a coarse-to-fine, causally ordered structure aligned with autoregressive modeling.

• Token-conditioned flow matching decoder. We introduce an MMDiT-based [17] flowmatching decoder [18] that reconstructs continuous action chunks from compact discrete tokens, combining the control fidelity of continuous generative action heads with the training compatibility of purely autoregressive VLAs.

• Strong empirical performance. CATOK consistently outperforms prior methods such as FAST [13] and OAT [15] across multiple simulation and real-world benchmarks in reconstruction fidelity–compression tradeoff, inference efficiency and VLA task success rate.

## 2 Related Works

Vision-Language-Action Models. Benefiting from the strong capability of pretrained VLMs [19–21] to understand images and language instructions, VLAs demonstrate remarkable performance and generalization in robotic manipulation tasks [3, 22]. Existing VLA frameworks primarily adopt two paradigms for action integration: discrete tokenization for autoregressive generation [2, 9, 13, 11, 23], and continuous regression via an auxiliary action head conditioned on the VLM’s latent representations [5, 24, 6, 25]. Despite the fact that discrete-token autoregressive paradigms incur excessive inference delays, recent works reveal a contrasting boost in training performance. FAST [13] demonstrates that employing a tokenizer that effectively compresses action chunks can significantly enhance training efficiency. Moreover, fine-tuning VLMs via discrete tokens, rather than continuous action heads, has been shown to better preserve the rich semantics of the model [26]. Together, these findings underscore the critical importance of designing an optimal discrete action tokenizer.

Action Tokenization. Inherently, an action chunk is represented as a two-dimensional matrix spanning both temporal and spatial (action) dimensions. To process such structures, researchers have proposed a variety of action tokenization strategies, ranging from rule-based methods to data-driven approaches. Early binning-based action discretization [2, 9, 22] introduces substantial token redundancy, severely compromising both training and inference speeds. More critically, this naive mapping ignores inter-token dependencies, limiting its effectiveness under the standard next-token-prediction framework. Compression-based methods (e.g., DCT [13] and B-splines [27]) were subsequently introduced to compress each action dimension independently along the temporal axis. While these approaches substantially improve the compression ratio—particularly at high control frequencies—and boost training efficiency, they still fail to capture correlations across action dimensions. Most recently, learning-based tokenizers [11, 28, 29] have emerged to model the spatiotemporal dependencies of actions, with a growing focus on the hierarchical organization and underlying structure of the token space. For instance, RVQ-based approaches [14, 30] capture multi-level hierarchies through residual quantization, whereas OAT [15] enforces structural integrity in the token space by applying random token masking in the action decoder.

While promising, these tokenization paradigms leave room for further refinement in balancing highquality reconstruction with token-level causal structures. A key objective, therefore, is to develop a training formulation that naturally yields a structured latent topology. To this end, we propose a method that transfers the temporal causal structure inherent in Flow Matching [16, 31, 32] into the token space, thereby inducing causality among tokens. This design is empirically validated to be effective for autoregressive modeling.

## 3 Method

In this section, we first formalize the action tokenization problem in Section 3.1; then, we describe the CATOK architecture in Section 3.2 and detail its training and inference procedures in Section 3.3. Finally, Section 3.4 presents VLA-CATOK, which integrates the proposed tokenizer into an autoregressive VLA backbone.

## 3.1 Problem Formulation

We study action tokenization for purely autoregressive VLA models. Given a continuous action chunk $\dot { \mathcal { A } } = ( a _ { t } , \dotsc , a _ { t + H - 1 } ) \in \dot { \mathbb { R } } ^ { H \times d _ { a } }$ , the goal is to learn an encoder E and decoder D such that

$$
{ \mathcal { E } } ( { \mathcal { A } } ) = C = ( c _ { 1 } , \ldots , c _ { K } ) , \quad c _ { k } \in \{ 1 , \ldots , | { \mathcal { C } } | \} , \qquad { \mathcal { D } } ( C ) \approx { \mathcal { A } } ,
$$

where C is a finite codebook and K is the token sequence length. The tokenizer is trained to minimize reconstruction error:

$$
\operatorname* { m i n } _ { \varepsilon , D } \mathbb { E } _ { A } \big [ \| D ( \mathcal { E } ( \mathcal { A } ) ) - \mathcal { A } \| ^ { 2 } \big ] .
$$

Once trained and frozen, E and D convert action chunks into discrete tokens C, which a purely autoregressive VLA policy $\pi _ { \theta }$ predicts conditioned on observations $o _ { t } = ( I _ { t } , s _ { t } , l )$ . The VLA is trained with cross-entropy over tokens:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( o _ { t } , A ) } \Big [ - \sum _ { k = 1 } ^ { K } \log \pi _ { \theta } \big ( c _ { k } \mid o _ { t } , c _ { < k } \big ) \Big ] .
$$

![](images/2ebf566720c46620bbcc84e99cf533daef1079962f7229b8d9f0497b3ef7264b.jpg)  
(a) Causal Action Tokenizer Pipeline

![](images/cbc64db0b836683604b0e58dc38c7bcca2e8b6a32548b29727c51fbde8dac2cb.jpg)  
Figure 2: An Overview of CATOK. (a) CATOK Pipeline. CATOK encodes an action chunk with a Dual-Stream Encoder into latent queries, discretizes them via a Bottleneck VQ, and reconstructs actions using a Flow-Matching Decoder with Conditional Annealing. (b) Conditional Annealing Mechanism progressively masks the first $\kappa ( t )$ token embeddings according to flow matching timestep t: darker colors indicate tokens earlier in the sequence, and lighter colors indicate later tokens, assigning each token to a different stage of the flow trajectory.

At inference, π<sub>θ</sub> generates $\hat { C } = ( \hat { c } _ { 1 } , \dots , \hat { c } _ { K } )$ autoregressively, and the frozen decoder recovers the continuous action $\hat { \mathcal { A } } = \mathcal { D } ( \hat { C } )$ for execution.

## 3.2 CATOK Architecture

Our causal action tokenizer, CATOK, encodes action chunks using a Dual-Stream Action Encoder that maps them into a compact set of latent query tokens via co-attention. These latent tokens are then discretized through a Bottleneck Vector Quantizer into a finite codebook. Finally, a Flow-Matching Decoder with Conditional Annealing reconstructs high-fidelity action trajectories while enforcing a coarse-to-fine causal structure over the tokens.

Dual-Stream Action Encoder. We first normalize the raw action chunk using quantile-based normalization to mitigate the impact of outliers and improve numerical stability. Instead of flattening or 1D patchification, we encode $\mathcal { A } _ { t : t + H }$ using a 2D convolutional neural network (CNN) with weight normalization [33], which captures local correlations along both temporal and action dimensions. We further add learnable 2D positional embeddings to preserve the spatiotemporal structure, yielding structured embeddings $\mathbf { E } _ { \mathrm { a c t } }$

In parallel, we initialize K learnable query embeddings ${ \bf Q } ^ { ( 0 ) } \in \mathbb { R } ^ { K \times D }$ with independent positional embeddings. We then feed both $\mathbf { E } _ { \mathrm { a c t } }$ and $\mathbf { Q } ^ { ( 0 ) }$ into a symmetric multi-modal transformer inspired by MMDiT [17], where co-attention enables bidirectional interaction between the two token sets (Figure 2(a)). In this process, the action embeddings provide structured context, while the query tokens aggregate information into a compact latent space. After L layers, we retain the updated query stream $\mathbf { \breve { Q } } = \left( \mathbf { q } _ { 1 } , \dots , \mathbf { q } _ { K } \right) \in \mathbb { R } ^ { K \times D }$ as the final representation of the action chunk.

Bottleneck Vector Quantizer. To obtain a compact discrete representation, we adopt thefactorized codes design of ViT-VQGAN [34]. Each latent vector $\mathbf { q } _ { k } \in \mathbb { R } ^ { D }$ is first projected into a lowerdimensional space through a linear bottleneck $\psi : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D ^ { \prime } }$ with $D ^ { \prime } \ < \ D$ . The projected embedding $\psi ( \mathbf { q } _ { k } )$ is then quantized against a learnable codebook $\mathcal C = \{ \mathbf e _ { 1 } , \dots , \mathbf e _ { | \mathcal C | } \} \subset \mathbb { R } ^ { D ^ { \prime } }$ by selecting its nearest neighbor indexed $c _ { k }$ in Euclidean distance:

$$
c _ { k } = \underset { j \in \{ 1 , . . . , | { \mathcal { C } } | \} } { \arg \operatorname* { m i n } } \ \big \| \mathbf { e } _ { j } - \psi ( \mathbf { q } _ { k } ) \big \| _ { 2 } ,\tag{1}
$$

Finally, a post-projection layer $\phi : \mathbb { R } ^ { D ^ { \prime } }  \mathbb { R } ^ { D }$ is applied to reconstruct the quantized embedding in the original latent space, $\hat { \mathbf { q } } _ { k } = \phi ( \mathbf { e } _ { c _ { k } } )$ . This factorized design substantially improves codebook utilization by decoupling the codebook dimension from the model’s hidden dimension.

Flow-Matching Decoder with Conditional Annealing. Prior work suggests that diffusion/flow timesteps naturally correspond to different levels of abstraction, from coarse global structure to fine-grained details [16]. We exploit this property to ground discrete tokens in specific generative stages via a conditional annealing mechanism (as illustrated in Figure 2 (b)).

Let $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ and $x _ { 1 } = A$ denote Gaussian noise and the target action chunk respectively. Following rectified flow, we define the interpolation path $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ with target velocity $x _ { 1 } - x _ { 0 }$ . Instead of conditioning the decoder on all token embeddings throughout the entire trajectory, we progressively mask the first $\kappa ( t )$ tokens according to

$$
\kappa ( t ) = \lfloor t K \rfloor , \quad t \in [ 0 , 1 ] ,\tag{2}
$$

so that only the active subset $\hat { \mathcal { Q } } _ { > \kappa ( t ) }$ is provided as conditioning at time t. The MMDiT-based velocity network [17] is trained with the flow-matching objective

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \left[ \Big | \Big | v _ { \theta } \big ( x _ { t } , t , \hat { \mathcal { Q } } _ { > \kappa ( t ) } \big ) - \big ( x _ { 1 } - x _ { 0 } \big ) \Big | \Big | _ { 1 } \right] .\tag{3}
$$

This schedule forces different tokens to be useful at different stages of the generation process. At early stages, when $x _ { t }$ is dominated by noise, the active tokens must provide coarse global guidance; as t increases and the trajectory approaches the target action, fewer tokens remain active, encouraging later tokens to specialize in residual, fine-grained corrections. Consequently, each token embedding acts as a discrete information increment between adjacent flow-matching stages, inducing a causally ordered coarse-to-fine hierarchy that is naturally aligned with autoregressive action generation.

## 3.3 CATOK Training and Inference

Training Scheme. The model is trained end-to-end with a combination of reconstruction and quantization losses:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { F M } } + \alpha \mathcal { L } _ { \mathrm { s m o o t h } } + \beta \mathcal { L } _ { \mathrm { V Q } } ,\tag{4}
$$

where $\alpha , \beta$ are hyperparameters that balance the three terms.

Specifically, the reconstruction loss integrates constraints in both the time and frequency domains. It consists of a time-domain $L _ { \mathrm { F M } }$ loss which together with a frequency-domain $L _ { \mathrm { s m o o t h } }$ loss computed via the discrete cosine transform (DCT):

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \big \| \mathrm { D C T } \big ( { x _ { 1 } } - { x _ { 0 } } \big ) - \mathrm { D C T } \big ( { v _ { \theta } } \big ( { x _ { t } } , t , \hat { \mathcal { Q } } _ { > \kappa ( t ) } \big ) \big ) \big \| _ { 1 } ,\tag{5}
$$

The DCT term explicitly penalizes spectral discrepancies, thereby promoting smoother and more temporally consistent predictions over long horizons. Complementing the reconstruction objective, the quantization loss adopts the standard commitment formulation:

$$
\mathcal { L } _ { \mathrm { V Q } } = \sum _ { i = 1 } ^ { K } \lVert \psi ( \mathbf { q } _ { i } ) - \mathrm { s g } ( \hat { \mathbf { q } } _ { i } ) \rVert _ { 2 } ^ { 2 } ,\tag{6}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator [35]. To further stabilize training and mitigate codebook collapse, codebook entries are updated using exponential moving averages (EMA) of the assigned embeddings, with rarely used entries, identified by low EMA usage, periodically reinitialized from randomly sampled embeddings in the current batch [35, 36].

Detokenization. Given action tokens $C = ( c _ { 1 } , \dots , c _ { K } )$ , each $c _ { i }$ is mapped to embedding $\hat { \mathbf { q } } _ { i } \in \mathcal { C }$ yielding $\hat { \mathbf { Q } } = ( \hat { \mathbf { q } } _ { 1 } , \hdots , \hat { \mathbf { q } } _ { K } )$ . The action is then decoded via flow matching by solving an ODE:

$$
\mathbf { x } _ { 1 } = \mathbf { x } _ { 0 } + \int _ { 0 } ^ { 1 } v _ { \theta } \left( \mathbf { x } _ { t } , t \mid { \tilde { \mathbf { Q } } } _ { t } \right) d t , \qquad \mathbf { x } _ { 0 } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{7}
$$

where $v _ { \theta }$ denotes the velocity field and $\tilde { \mathbf { Q } } _ { t } = ( \mathbf { 0 } , \ldots , \mathbf { 0 } , \hat { \mathbf { q } } _ { \kappa ( t ) + 1 } , \ldots , \hat { \mathbf { q } } _ { K } )$ retains only the last $K - \kappa ( t )$ embeddings via a conditional annealing scheme.

## 3.4 VLA-CATOK

To evaluate CATOK in robotic policy learning, we integrate it into a standard autoregressive VLA architecture [2, 13]. As shown in Figure 3, VLA-CATOK takes the current image $I _ { t } ,$ , language instruction l, and optional proprioceptive state $s _ { t }$ as input, and autoregressively predicts a sequence of action tokens $C = \mathsf { \bar { ( } } c _ { 1 } , \mathsf { . . . } , \mathsf { \bar { { c } } } _ { K } )$ . Each predicted token c is mapped to its VQ embedding qˆ through the learned codebook, forming $\hat { \mathbf { Q } } = ( \hat { \mathbf { q } } _ { 1 } , \hdots , \hat { \mathbf { q } } _ { K } )$ . A frozen MMDiT flow-matching decoder then converts $\hat { \mathbf { Q } }$ into the final continuous action chunk in a single decoding step.

![](images/123a8edda9ca66ed85e47b7f91a0d2df9fff84e77c5bab82ea1abe6c98b37b94.jpg)  
Figure 3: An Overview of VLA-CATOK. VLA-CATOK feeds visual, language, and proprioceptive tokens into a VLM backbone to autoregressively predict causal action tokens. These tokens are mapped via the learned codebook and processed by a frozen MMDiT flow-matching decoder to produce the final action in a single denoising step. Discrete tokens serve as the sole interface, naturally insulating the VLM’s pretrained knowledge from action-specific gradients.

During VLA training, the CATOK tokenizer and decoder are kept frozen, and only the autoregressive VLA backbone is updated. The backbone is trained with teacher-forced next-token prediction using a cross-entropy loss over CATOK tokens:

$$
\mathcal { L } _ { A R } = - \sum _ { i = 1 } ^ { K } \log p _ { \theta } \left( c _ { i } \vert I _ { t } , s _ { t } , l , c _ { < i } \right)\tag{8}
$$

Unlike approaches that supervise actions directly and thus propagate action gradients back into the VLM backbone, our framework uses discrete tokens as the interface between the two modules and freezes the pretrained decoder, inherently insulating knowledge between the two modules. This substantially preserves the pretrained knowledge embedded in the VLM, thereby strengthening its instruction-following capability [26].

## 4 Experiments

To comprehensively evaluate the effectiveness of CATOK, we conduct a series of experiments covering both the tokenizer itself and its integration into downstream VLA policies. Section 4.1 introduces the tokenizer baselines, VLA backbones, and evaluation benchmarks, including both simulation and real-world settings. Section 4.2 examines whether CATOK improves downstream VLA performance and training efficiency. Section 4.3 analyzes whether CATOK produces causally ordered tokens with generative semantics. Finally, Section 4.4 studies the intrinsic properties of the tokenizer, including reconstruction fidelity, compression efficiency, and tokenization latency.

## 4.1 Experimental Setup

Tokenizer Baselines. We compare CATOK with three other representative action tokenizers for autoregressive policy learning: BIN [2], which uniformly discretizes each action dimension into 256 bins; FAST [13], which applies DCT-based action compression followed by BPE tokenization; and OAT [15], a learning-based VQ-VAE action tokenizer that enforces token-level causal structure through prefix-based masking and decoding. All tokenizers are integrated into the same autoregressive policy framework, and we further analyze their intrinsic token properties in Section 4.4.

Simulation Benchmarks. We evaluate downstream policy performance on three simulation benchmarks: LIBERO [37], SimplerEnv [38], and RoboTwin 2.0 [39]. LIBERO consists of four official suites covering diverse long-horizon manipulation tasks. SimplerEnv evaluates zero-shot generalization on real-world-inspired robotic manipulation tasks using the WidowX platform. RoboTwin

Table 1: Comparison on robotic manipulation simulation benchmarks. CATOK achieves the best overall performance across all benchmarks, with particularly significant improvements on long-horizon tasks such as Long in LIBERO and StackGreenCubeOnYellowCube in SimplerEnv, demonstrating the effectiveness of causal action tokenization for modeling long-range temporal dependencies.
<table><tr><td></td><td colspan="5">LIBERO</td><td colspan="5">SimplerEnv</td><td colspan="2">RoboTwin 2.0</td></tr><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Spoon</td><td>Carrot</td><td>Stack</td><td>Eggplant</td><td>Avg.</td><td>Clean</td><td>Randomized</td></tr><tr><td>BIN</td><td>0.586</td><td>0.878</td><td>0.680</td><td>0.604</td><td>0.687</td><td>0.542</td><td>0.333</td><td>0.208</td><td>0.708</td><td>0.448</td><td>0.213</td><td>0.221</td></tr><tr><td>FAST</td><td>0.960</td><td>0.998</td><td>0.962</td><td>0.901</td><td>0.955</td><td>0.500</td><td>0.375</td><td>0.375</td><td>0.625</td><td>0.469</td><td>0.478</td><td>0.478</td></tr><tr><td>OAT</td><td>0.428</td><td>0.876</td><td>0.704</td><td>0.276</td><td>0.571</td><td>0.417</td><td>0.208</td><td>0.167</td><td>0.667</td><td>0.365</td><td>0.229</td><td>0.233</td></tr><tr><td>CATOK</td><td>0.978</td><td>0.994</td><td>0.954</td><td>0.910</td><td>0.959</td><td>0.458</td><td>0.417</td><td>0.542</td><td>0.542</td><td>0.490</td><td>0.489</td><td>0.531</td></tr></table>

2.0 provides large-scale embodied manipulation benchmarks under both clean and randomized environments, covering 50 tasks with substantial scene diversity.

Real-World Benchmark. We further evaluate CATOK on a real Franka manipulator under three task suites, as shown in Fig.4: Pick-Spatial, Pick-Color, and Stack-Long. The tasks are designed to test spatial generalization, color generalization, and long-horizon task completion, respectively.

For training, we collect 362 trajectories for Pick-Cups task and 220 trajectories for Stack-Cups task. Using Qwen3.5-VL as the shared backbone, we train QwenCATok and QwenFAST with the same training protocol and different action tokenizers. To ensure a comparable reconstruction quality between tokenizers, CATok is pretrained on the collected real-robot data together with InternA1 joint data, yielding reconstruction accuracy comparable to FAST.

Training and Evaluation Protocols. For LIBERO, we fine-tune π -FAST-base [13] on all four suites and report the average success rate over 40 tasks with 50 rollouts per task. For the more challenging SimplerEnv and RoboTwin 2.0 benchmarks, all methods are trained from scratch using Qwen3-VL-4B-Instruct [21] as the shared VLM backbone to ensure fair comparison. In SimplerEnv, policies are trained on Bridge [40] and evaluated zero-shot on the four WidowX tasks following the official protocol. In RoboTwin 2.0, we train on the official clean and randomized datasets, and evaluate on all 50 tasks under both clean and randomized settings with 20 rollouts per task. Additional implementation and evaluation details are provided in the Appendix A.

![](images/974e011b1ce409b49cf9e189388575a65907c5700823ed21e66deed83848a58b.jpg)  
(a) Pick-Spatial

![](images/5dc8352c6fc3c7605ca367a4e607fa95df58a8683982fd63925cc646d8ffca07.jpg)  
(b) Pick-Color

![](images/8f39e9308fdbb9ada34736c73ebb3ce1745e50044ed8c8a77bbcb5bdfbc58259.jpg)  
(c) Stack-Long  
Figure 4: Real-world scene layouts for the three task suites. From left to right: Pick-Spatial, testing spatial generalization by varying cup and plate positions; Pick-Color, testing color generalization with different unseen cup colors; and Stack-Long, testing long-horizon manipulation under different spatial layouts.

## 4.2 Can CATOK Help VLA Training?

CATOK consistently improves VLA success rates across simulation benchmarks. As shown in Table 1, CATOK consistently outperforms all baseline tokenizers across the three simulation benchmarks. Notably, it exhibits particularly pronounced gains on long-horizon manipulation tasks. In particular, on the StackGreenCubeOnYellowCube task in SimplerEnv, CATOK improves the success rate by more than 16.7% relative to the strongest baseline, highlighting its superior ability to model extended sequences of coordinated actions. These results suggest that introducing causal ordering into action tokenization enables autoregressive policies to capture long-range temporal dependencies more effectively, which is crucial for complex long-horizon manipulation tasks.

CATOK improves both VLA training efficiency and inference speed. Figure 5 (a) compares the success rates of CATOK and baseline tokenizers across training steps, highlighting the superior training efficiency of CATOK. CATOK consistently achieves higher success

rates, particularly during the later stages of training, indicating both faster convergence and more stable learning dynamics. Notably, CATOK matches the best performance achieved by FAST with only approximately 50% steps, demonstrating substantially improved training efficiency. We attribute this improvement primarily to the causal token structure analyzed in Section 4.3. By organizing action chunks into causally ordered tokens, CATOK naturally aligns the tokenizer with the autoregressive policy objective, enabling more effective stage-wise action modeling rather than learning from weakly structured discrete codes. Consequently, during autoregressive genera-

![](images/9708811f1b01d5f8bdecf514b8c09671a4f1e2ea1a8373d977526ea15b46b0d1.jpg)  
(a) VLA Training Efficiency

![](images/8114cdfe8ccfac5307cb0d0551f9cf104208c24ced6f4f2c8d027169a7f2a1fb.jpg)  
(b) VLA Inference Latency  
Figure 5: (a) Training efficiency on the SimplerEnv benchmark. Curves are Gaussian-smoothed to reduce evaluation stochasticity without changing the relative ranking of different methods. Notably, CATOK reaches the best baseline performance using only around 50% of the training steps. (b) VLA inference latency. By jointly considering autoregressive generation and decoding overhead, CATOK achieves near-minimal end-to-end inference latency while attaining the best success rate among all methods.

tion, each predicted token provides more informative guidance for subsequent token prediction and trajectory generation, consistently improving both training efficiency and policy performance across diverse tasks. As shown in Figure 5 (b), CATOK also exhibits highly competitive inference efficiency. By jointly considering autoregressive token generation and tokenizer decoding overhead, it achieves near-minimal end-to-end VLA inference latency among all compared methods.

<table><tr><td>Method</td><td>Pick-Spatial</td><td>Pick-Color</td><td>Stack-Long</td></tr><tr><td>FAST</td><td>0.450</td><td>0.300</td><td>0.275</td></tr><tr><td>CATOK</td><td>0.650</td><td>0.500</td><td>0.475</td></tr></table>

Table 2: Real-world manipulation results. CATOK consistently outperforms FAST across spatial, color, and longhorizon generalization tasks.

CATOK transfers effectively to realworld robots. To further evaluate whether the benefits of causal action tokenization extend beyond simulation, we deploy CATOK on a realworld Franka robot. As shown in Table 2, CATOK consistently outperforms FAST across three task suites covering spatial generalization, color

generalization, and long-horizon manipulation. Notably, CATOK achieves a consistent 20 percentagepoint improvement over FAST across all three settings, demonstrating that the benefits of causal action tokenization transfer to real-world robotic manipulation.

## 4.3 Does CATOK Produce Causal Tokens with Generative Semantics?

Causal Predictive Ordering of Action Tokens. We test whether CATOK tokens exhibit a predictive order by training a lightweight action-only autoregressive transformer for each tokenizer and measuring next-token entropy on held-out LIBERO-Bridge Mixture chunks. At each position k, we compute $\begin{array} { r } { \mathbf { \tilde { { H } } } _ { k } = - \sum _ { v } p _ { k , v } \operatorname { l \tilde { o g } } p _ { k , { : } } } \end{array}$ <sub>v</sub> from the prediction logits; lower entropy indicates that preceding tokens provide stronger predictive context.

As shown in Figure 6(a), CATOK exhibits a clear decreasing entropy trend when tokens are predicted in their original order, indicating that later tokens become progressively easier to predict as more prefix tokens are observed. When predicted in reverse order, the trend is inverted, demonstrating that the learned token ordering is directional rather than arbitrary. In contrast, FAST remains nearly flat and Bin shows a slight increase, suggesting substantially weaker positional dependency. These results indicate that CATOK learns causally ordered action tokens that are naturally aligned with autoregressive generation.

![](images/8ff70b5935715b3edde870f0f5c1177aa4a5e6b6bc7d1c3e9ab8ee9dd9678720.jpg)  
(a) Entropy-based Causality Validation

![](images/8c7f4ffa12aca55ae80a8d66d64b69e93d163db92f5b743e4b625a14809882ed.jpg)  
(b) Structured Token Space

![](images/eb6b1f3a4fbcd401b51a5beb57f48e91b72c526b4e6363e656bb33922d6789d8.jpg)  
(c) Token Information Visualization  
Figure 6: CATOK Produces Causal Tokens with Generative Semantics. (a) Causal Ordering CATOK ’s entropy decreases in the forward order (CATOKfwd) and increases in the reverse order (CATOK rev), while FAST and Bin show weaker positional structure. (b) Token embedding geometry. T-SNE map shows slot-dependent organization, progressing from smooth early-token regions to compact late-token clusters. (c) Prefix reconstruction. Increasing token prefixes produce structured trajectory updates, with dashed curves showing incremental contributions.

Stage-wise Generative Semantics. We next examine whether CATOK tokens carry generative semantics, i.e., whether different tokens induce distinct and structured changes in the decoded action chunk. We use two complementary probes: token-space geometry and prefix reconstruction.

Token-space geometry. We visualize VQ embeddings from 32,768 normalized LIBERO action chunks using t-SNE, with embeddings extracted from a frozen CATOK checkpoint (K=16, codebook size 4096, $d _ { \mathrm { v q } } = 1 6 )$ . As shown in Figure 6(b), the embeddings are clearly organized by token slot: early slots form smooth elongated regions, middle slots become more diffuse, and late slots form compact, separated clusters. This slot-dependent geometry suggests that CATOK tokens are not exchangeable codebook indices, but occupy distinct generative roles that progress from shared motion factors to specialized residual corrections.

Prefix reconstruction. We reconstruct action chunks from token prefixes with $n \in \{ 1 , 6 , 1 1 , 1 6 \}$ . As shown in Figure 6(c), adding tokens induces coherent trajectory updates, from structural changes in spatial, rotation, and gripper dimensions to finer residual refinements. This indicates that CATOK tokens serve as meaningful generative increments rather than passive compression codes.

Together, the geometry and reconstruction probes show that CATOK learns a coarse-to-fine hierarchy of generative action tokens. Earlier tokens capture shared global structure, while later tokens contribute increasingly specialized residual refinements. This stage-wise organization complements the causal ordering above and provides a natural inductive bias for autoregressive action generation.

We further validate the causal semantics of individual token positions through native-decoder intervention experiments, including token swapping and single-token removal. These experiments provide direct evidence that different token positions control distinct stages of action generation. Detailed experimental settings and results are provided in Appendix C.1.

## 4.4 Intrinsic Properties of CATOK

CATOK balances reconstruction fidelity and compression efficiency. Action tokenization inherently requires balancing reconstruction fidelity and compression efficiency. We evaluate reconstruction quality using Valid Reconstruction Rate (VRR) [14] and measure compactness using Compression Rate (CR). Detailed metric definitions are provided in Appendix B.

As shown in Figure 7 (a-c), CATOK achieves near-perfect reconstruction fidelity, with VRR comparable to Binning and FAST and substantially higher than OAT under the same token budget. Meanwhile, it provides significantly better compression than Binning and FAST while remaining comparable to OAT, resulting in the best overall fidelity–compression tradeoff.

Encoding and decoding latency. To evaluate the efficiency of action tokenization during both training and inference, we measure the average encoding and decoding latency during action tokenization. Detailed measurement protocols are provided in Appendix B. Since CATOK substantially outperforms Binning and OAT in downstream VLA performance and achieves performance comparable to FAST, while consistently obtaining slightly higher success rates overall, FAST serves as the most relevant latency baseline. As shown in Figure 7 (d-e), compared with FAST, CATOK achieves substantially lower encoding and decoding latency while also delivering slightly better downstream VLA performance, demonstrating superior time efficiency for both training and inference.

![](images/9a72d5240866663e77b0c4e8234d3bbd6aa46341287b58da849541083fcb40c2.jpg)

![](images/2d254087ff393e10277104860a354cf330e4dffa1f2538e0816679a49747a1df.jpg)

![](images/245c775e5486a28a0f79427a0f334ef2813edea42f990a0ef784956f1c69e4fd.jpg)

![](images/8e729eb6a2ba2085e73b7419ae8241f249fd0ce9e75f21d38a252423af5c4f28.jpg)

![](images/e175fdf697fe8b88b98ab3dfba0d61e13030347f778320650eb474e581b3a0b9.jpg)  
Figure 7: Intrinsic properties of action tokenizers. (a–c) Fidelity vs. compression efficiency. VRR measures reconstruction quality; CR measures compression ratio; VRR×CR jointly captures the fidelity–compression tradeoff. $\mathrm { C A T O K _ { 8 } }$ achieves the highest VRR×CR (16.85), outperforming all baselines at the same compression level. (d–e) Inference efficiency. Encode and decode latencies are shown on a log scale. The subscript k in $\mathrm { O A T } _ { k }$ and $\mathrm { C A T O K } _ { k }$ denotes the number of tokens.

## 4.5 Ablations and Analysis

We conduct ablation studies on key components and design choices in CATOK. Specifically, we analyze the effect of the flow-matching head, where the results show that the complete design consistently achieves the best performance. In addition, we study the impact of the number of tokens, a key design choice in CATOK, on downstream VLA performance. Detailed ablation settings and additional results are provided in Appendix C.

## 5 Conclusion

We introduced CATOK, a causal action tokenizer for purely autoregressive VLA models. By transferring the stage-wise causal structure of flow matching into the token space through conditional annealing, CATOK learns compact action tokens with causal ordering and generative semantics. The resulting token sequence enables autoregressive policies to generate continuous action chunks through a coarse-to-fine factorization, while a frozen MMDiT-based flow-matching decoder preserves high-fidelity action reconstruction. Experiments across multiple simulation benchmarks and realworld robotic manipulation tasks demonstrate that CATOK improves downstream task performance, training efficiency, and the fidelity–compression tradeoff over existing action tokenizers, while maintaining competitive inference efficiency. Our analysis further verifies the causal ordering and generative semantics of the learned token sequence. These results highlight causal and generative action tokenization as a promising approach for scalable purely autoregressive VLA systems.

Limitations. Despite the promising results, our real-world evaluation is currently limited to a single robot platform and a finite set of manipulation tasks. Therefore, the robustness of CATOK across diverse robot embodiments, sensing configurations, environmental conditions, and dynamics remains unexplored. In addition, our experiments are conducted on relatively constrained datasets, and we have not yet investigated the scalability of CATOK on large-scale and diverse cross-embodiment robotic data. Future work will evaluate CATOK across broader real-world embodiments and large-scale robotic datasets to study its scalability and generalization under more diverse real-world settings.

## References

[1] Anthony Brohan et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[3] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[4] Dibya Ghosh et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi\_0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[6] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi\_0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[7] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[8] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[9] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[10] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model, 2025.

[11] Yi Liu, Sukai Wang, Dafeng Wei, Xiaowei Cai, Linqing Zhong, Jiange Yang, Guanghui Ren, Jinyu Zhang, Maoqing Yao, Chuankang Li, et al. Unified embodied vlm reasoning with robotic action via autoregressive discretized pre-training. arXiv preprint arXiv:2512.24125, 2025.

[12] Yutong Hu, Jan-Nico Zaech, Nikolay Nikolov, Yuanqi Yao, Sombit Dey, Giuliano Albanese, Renaud Detry, Luc Van Gool, and Danda Paudel. Ar-vla: True autoregressive action expert for vision-language-action models, 2026.

[13] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[14] Yicheng Liu, Shiduo Zhang, Zibin Dong, Baijun Ye, Tianyuan Yuan, Xiaopeng Yu, Linqi Yin, Chenhao Lu, Junhao Shi, Luca Jiang-Tao Yu, et al. Faster: Toward efficient autoregressive vision language action modeling via neural action tokenization. arXiv preprint arXiv:2512.04952, 2025.

[15] Chaoqi Liu et al. OAT: Ordered action tokenization. arXiv preprint arXiv:2602.04215, 2026.

[16] Zhongqi Yue, Jiankun Wang, Qianru Sun, Lei Ji, Eric I-Chao Chang, and Hanwang Zhang. Exploring diffusion time-steps for unsupervised representation learning. arXiv preprint arXiv:2401.11430, 2024.

[17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[18] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[19] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023.

[20] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

[21] Qwen Team. Qwen3 technical report, 2025.

[22] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[23] Ankit Goyal, Hugo Hadfield, Xuning Yang, Valts Blukis, and Fabio Ramos. Vla-0: Building state-of-the-art vlas with zero modification. arXiv preprint arXiv:2510.13054, 2025.

[24] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[25] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[26] Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, et al. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705, 2025.

[27] Hongyi Zhou, Weiran Liao, Xi Huang, Yucheng Tang, Fabian Otto, Xiaogang Jia, Xinkai Jiang, Simon Hilber, Ge Li, Qian Wang, et al. Beast: Efficient tokenization of b-splines encoded action sequences for imitation learning. arXiv preprint arXiv:2506.06072, 2025.

[28] Zibin Dong, Yicheng Liu, Shiduo Zhang, Baijun Ye, Yifu Yuan, Fei Ni, Jingjing Gong, Xipeng Qiu, Hang Zhao, Yinchuan Li, et al. Actioncodec: What makes for good action tokenizers. arXiv preprint arXiv:2602.15397, 2026.

[29] Renming Huang, Chendong Zeng, Wenjing Tang, Jintian Cai, Cewu Lu, and Panpan Cai. Mimic intent, not just trajectories. arXiv preprint arXiv:2602.08602, 2026.

[30] Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, and Tong He. Vq-vla: Improving vision-language-action models via scaling vector-quantized action tokenizers. arXiv preprint arXiv:2507.01016, 2025.

[31] Bohan Wang, Zhongqi Yue, Fengda Zhang, Shuo Chen, Li’an Bi, Junzhe Zhang, Xue Song, Kennard Yanting Chan, Jiachun Pan, Weijia Wu, et al. Selftok: Discrete visual tokens of autoregression, by diffusion, and for reasoning. arXiv preprint arXiv:2505.07538, 2025.

[32] Bohan Wang, Mingze Zhou, Zhongqi Yue, Wang Lin, Kaihang Pan, Liyu Jia, Wentao Hu, Wei Zhao, and Hanwang Zhang. Selftok-zero: Reinforcement learning for visual generation via discrete and autoregressive visual tokens. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[33] Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 29, 2016.

[34] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.

[35] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[36] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[37] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023.

[38] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation, 2024.

[39] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025.

[40] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

## A Implementation Details

Table 3 provides implementation details for the action tokenizer training, VLA training, and evaluation protocols used in our experiments. In particular, for LIBERO and Bridge, we train a tokenizer on the combined datasets with mix ratios of 1.0 and 5.0, respectively, and then train the VLA separately on each dataset. For RoboTwin, we train both the tokenizer and the VLA on the full RoboTwin dataset.

Table 3: Benchmark-specific implementation details.
<table><tr><td>Setting</td><td>LIBERO</td><td>Simpler-Bridge</td><td>RoboTwin 2.0</td></tr><tr><td colspan="4">Benchmark and embodiment</td></tr><tr><td>Embodiment</td><td>Franka</td><td>WidowX</td><td>Bimanual ALOHA</td></tr><tr><td>Action DoF</td><td>7</td><td>7</td><td>14</td></tr><tr><td>Training data</td><td>Official LIBERO dataset</td><td>Bridge dataset</td><td>Clean and randomized RoboTwin 2.0 trajectories</td></tr><tr><td>Evaluation protocol</td><td>4 suites 10 tasks per suite 50 rollouts per task</td><td>4 WidowX tasks in SimplerEnv 24 rollouts per task</td><td>50 tasks 20 rollouts per task</td></tr><tr><td colspan="4">Action tokenizer configuration</td></tr><tr><td>Action horizon H</td><td>20</td><td>10</td><td>20</td></tr><tr><td>Number of action tokens K</td><td>16</td><td>16</td><td>32</td></tr><tr><td>Codebook size |C|</td><td>4096</td><td>4096</td><td>4096</td></tr><tr><td>Tokenizer training steps</td><td>300 000</td><td>300 000</td><td>200 000</td></tr><tr><td>Tokenizer batch size</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Tokenizer learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Tokenizer weight decay</td><td> $1 \times 1 0 ^ { - 2 }$  1000</td><td> $1 \times 1 0 ^ { - 2 }$  1000</td><td> $1 \times 1 0 ^ { - 2 }$  1000</td></tr><tr><td colspan="4">Tokenizer warmup steps</td></tr><tr><td>VLA backbone</td><td>π₀-FAST-base</td><td>Qwen3-VL-4B-Instruct</td><td>Qwen3-VL-4B-Instruct</td></tr><tr><td>VLA training steps</td><td>30 000</td><td>60 000</td><td>40 000</td></tr><tr><td>VLA batch size</td><td>32</td><td>64</td><td>32</td></tr><tr><td>VLA learning rate</td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$   $1 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>VLA weight decay Number of GPUs</td><td> $1 \times 1 0 ^ { - 1 0 }$   $4 { \times } \mathrm { H } 1 0 0 \mathrm { G P } \mathrm { U s }$ </td><td> $1 \times 1 0 ^ { - 8 }$   $4 { \times } \mathrm { H } 1 0 0 \mathrm { G P U s }$ </td><td> $4 { \times } \mathrm { H } 1 0 0 \mathrm { G P U s }$ </td></tr></table>

## B Evaluation Metrics

In this section, we provide the detailed definitions and computation protocols for the evaluation metrics used to assess the intrinsic properties and efficiency of action tokenizers.

## B.1 Valid Reconstruction Rate (VRR)

To evaluate reconstruction fidelity, we adopt the Valid Reconstruction Rate (VRR) [14], defined as the proportion of reconstructed action chunks whose normalized reconstruction error falls below a predefined threshold:

$$
\mathrm { V R R } = \frac { N _ { \mathrm { v a l i d } } } { N _ { \mathrm { t o t a l } } } , \quad N _ { \mathrm { v a l i d } } = \sum _ { i = 1 } ^ { N _ { \mathrm { t o t a l } } } { \bf 1 } \left( \frac { \lVert A _ { i } ^ { \mathrm { p r e d } } - A _ { i } ^ { \mathrm { g t } } \rVert _ { 2 } } { H } < \sigma \right) ,\tag{9}
$$

where $A _ { i } ^ { \mathrm { p r e d } } , A _ { i } ^ { \mathrm { g t } } \in \mathbb { R } ^ { H \times D }$ denote the reconstructed and ground-truth action chunks for the i-th sample, respectively. Here, H is the prediction horizon and D is the action dimension. The $\ell _ { 2 }$ norm is computed over both temporal and action dimensions, and normalization by H yields the average per-step reconstruction error. σ denotes a predefined threshold, and 1(·) is the indicator function.

Table 4: Controlled ablation of conditional annealing, random masking, and decoder architecture.
<table><tr><td>Method</td><td>Spatial Object</td><td>Goal</td><td>Long</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>A - Ours</td><td>0.978 0.994</td><td>0.954</td><td>0.910</td><td>0.959</td></tr><tr><td>B - Ours w/o conditional annealing</td><td>0.920 0.980</td><td>0.918</td><td>0.834</td><td>0.914</td></tr><tr><td>C - VQ-VAE + MMDiT + random masking</td><td>0.724 0.924</td><td>0.756</td><td>0.542</td><td>0.737</td></tr><tr><td> $\mathrm { \mathbf { D } - V Q \mathrm { - } V A E + M M D i T }$ </td><td>0.614 0.796</td><td>0.296</td><td>0.240</td><td>0.486</td></tr></table>

## B.2 Compression Rate (CR)

To measure reconstruction efficiency, we use the Compression Rate (CR), defined as the ratio between the size of the original continuous action representation $N _ { \mathrm { r a w } }$ and the compressed discrete token sequence $N _ { \mathrm { t o k e n } } { : }$

$$
\mathrm { C R } = \frac { N _ { \mathrm { r a w } } } { N _ { \mathrm { t o k e n } } } , \quad N _ { \mathrm { r a w } } = B \times H \times D ,\tag{10}
$$

where B denotes the batch size, H is the action horizon, and D is the action dimension. A higher CR indicates more compact action representations.

## B.3 Encoding and Decoding Latency

To evaluate the efficiency of action tokenization during both training and inference, we measure the average encoding and decoding latency. Since action encoding is performed during training, encoding latency directly affects training efficiency. In contrast, decoding latency primarily impacts inference-time efficiency. Specifically:

• Encoding time (encode\_time): the average time required to convert continuous action chunks into discrete token sequences.

• Decoding time (decode\_time): the average time required to reconstruct continuous actions from discrete tokens.

All latency measurements are averaged across batches under the same hardware and evaluation settings.

## C Ablations and Analysis

In this section, we provide detailed ablation settings, additional experimental results, and further analysis for the key design components and design choices in CATOK.

Conditional Annealing and Architecture Ablations. To isolate the contribution of conditional annealing from the effects of decoder architecture and model capacity, we consider four variants: (A) our full model with conditional annealing, (B) CATOK without conditional annealing, (C) a VQ-VAE with the same MMDiT decoder and comparable parameter count trained with random token masking, and (D) a VQ-VAE with the same MMDiT decoder and comparable parameter count without random masking.

Removing conditional annealing (A → B) leads to a clear performance drop from 0.959 to 0.914, demonstrating its importance beyond the underlying flow-matching decoder and transformer architecture. Moreover, both C and D remain substantially below our method despite using the same MMDiT decoder and comparable model capacity, showing that the improvement cannot be explained solely by decoder architecture or model capacity. Notably, C already employs random token masking yet achieves only 0.737, indicating that random masking alone is insufficient to induce the desired causal structure in the token space. Together, these controlled comparisons demonstrate that conditional annealing is a critical factor in organizing token representations into a structured and causally ordered representation.

![](images/71f9717d1d90755cec4a1e67dd37104f1f37fdf7ca34ae5bd862f5819511348f.jpg)  
Figure 8: Effect of the number of action tokens on downstream VLA performance on the LIBERO benchmark. Performance improves from 8 to 16 tokens but degrades with larger token budgets, with 16 tokens achieving the best overall performance.

Number of Tokens. We further analyze the impact of the number of action tokens in CATOK on downstream VLA performance. Specifically, we evaluate CATOK with different token budgets (8, 16, 32, 64, and 128 tokens) while keeping all other training and architecture settings fixed. The results on the LIBERO benchmark are shown in Figure 8. Increasing the number of tokens from 8 to 16 consistently improves downstream VLA performance, while further increasing the token budge beyond 16 leads to performance degradation. We attribute this degradation to the increased difficulty of autoregressive modeling over excessively long token sequences, which reduces token efficiency and negatively impacts downstream policy learning. Overall, 16 tokens provide the best balance between representation capacity and autoregressive learning efficiency.

## C.1 Causality Validation

To further verify that the learned token ordering corresponds to genuine causal roles in action generation, we perform intervention experiments using the frozen native decoder of CATOK. Specifically, we consider two types of interventions: donor-swap and single-token removal. Both experiments directly modify individual token positions while keeping the remaining tokens unchanged, allowing us to measure the causal influence of each token on the reconstructed action trajectory.

Donor-Swap Intervention. We first evaluate the influence of each token position by replacing the token at position k with the corresponding token from another action chunk, while keeping all other tokens unchanged. We then decode the intervened token sequence using the native flow-matching decoder and measure the resulting deviation from the original reconstruction. By performing this intervention at different token positions and flow-matching time steps, we obtain a position-dependent influence map. If the tokens follow a causal coarse-to-fine ordering, early tokens should primarily affect coarse action structure, while later tokens should exert stronger influence during later stages of generation.

Single-Token Removal. We further perform a single-token removal experiment by replacing the token at position k with a neutral embedding while keeping all other tokens unchanged. The resulting action reconstruction is compared with the reconstruction from the complete token sequence. This experiment isolates the contribution of individual token positions and provides a complementary measure of their generative influence.

Results. As shown in Figure 9, the intervention effects exhibit a clear position-dependent structure. Early tokens have stronger influence on coarse action components and earlier generation stages, whereas later tokens increasingly affect fine-grained action details. The consistent stage-specific influence observed under both donor-swap and single-token removal interventions provides direct evidence that the token ordering learned by CATOK corresponds to meaningful causal roles rather than merely reflecting an arbitrary positional ordering.

## D Full Robotwin 2.0 Benchmark

Table 5 provides the full per-task RoboTwin 2.0 results corresponding to the aggregate scores in Table 1. We report success rates for both clean and randomized evaluation settings across all 50 tasks, with 20 rollouts per task.

![](images/c055767c45ec458a6853a90c43583f1f4853dfa79ff5e5e961d355174493ad8e.jpg)

![](images/87538dbe3acfb9958fb06deeef369eddc0c46484f61f6ce09a403a4376398453.jpg)

![](images/6cba6cfa5311020d9fef137822b28f3357d54fabaa138aec9a13b35f502a88e6.jpg)

![](images/d632df273e87914c3766fe94bc79ce404d78d1a1d2ec4c1d4740bd8ced0ccedb.jpg)

(a) Donor-swap influence matrix ∆(t). Replacing token k with a donor’s same-position embedding yields a position- and flow-stage-dependent influence map with triangular support, confirming the conditional-annealing schedule.  
![](images/98107e81e9a778c68dce62de0ce6f65a6eaf7df1c9fcce10dec5ebd2c448516f.jpg)  
(b) Single-token removal. Zeroing token k and decoding the intervened sequence isolates each position’s generative contribution; the per-dimension footprint progresses from coarse gripper/task-state information (early tokens) to fine joint/trajectory-detail information (late tokens).  
Figure 9: Causality validation via native-decoder interventions. Both donor-swap (a) and single-token removal (b) exhibit a clear position-dependent structure: early tokens have stronger influence on coarse action components and earlier generation stages, whereas later tokens increasingly affect fine-grained action details, confirming that the learned token ordering corresponds to meaningful causal roles rather than an arbitrary positional ordering.

Table 5: Per-task success rates on RoboTwin 2.0.
<table><tr><td rowspan="2">Simulation Task</td><td colspan="2">BIN</td><td colspan="2">FAST</td><td colspan="2">OAT</td><td colspan="2">CATOK</td></tr><tr><td>Clean</td><td>Rand.</td><td>Clean</td><td>Rand.</td><td>Clean</td><td>Rand.</td><td>Clean</td><td>Rand.</td></tr><tr><td>Adjust Bottle</td><td>80%</td><td>95%</td><td>90%</td><td>100%</td><td>100%</td><td>100%</td><td>90%</td><td>100%</td></tr><tr><td>Beat Block Hammer</td><td>15%</td><td>20%</td><td>25%</td><td>50%</td><td>15%</td><td>5%</td><td>65%</td><td>65%</td></tr><tr><td>Blocks Ranking RGB</td><td>0%</td><td>0%</td><td>10%</td><td>10%</td><td>5%</td><td>0%</td><td>5%</td><td>0%</td></tr><tr><td>Blocks Ranking Size</td><td>0%</td><td>0%</td><td>0%</td><td>5%</td><td>5%</td><td>0%</td><td>5%</td><td>10%</td></tr><tr><td>Click Alarmclock</td><td>65%</td><td>60%</td><td>90%</td><td>75%</td><td>60%</td><td>70%</td><td>65%</td><td>80%</td></tr><tr><td>Click Bell</td><td>95%</td><td>85%</td><td>80%</td><td>80%</td><td>75%</td><td>50%</td><td>95%</td><td>95%</td></tr><tr><td>Dump Bin Bigbin</td><td>30%</td><td>30%</td><td>95%</td><td>75%</td><td>15%</td><td>20%</td><td>65%</td><td>70%</td></tr><tr><td>Grab Roller</td><td>65%</td><td>50%</td><td>100%</td><td>100%</td><td>55%</td><td>60%</td><td>100%</td><td>100%</td></tr><tr><td>Handover Block</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>10%</td><td>10%</td></tr><tr><td>Handover Mic</td><td>10%</td><td>10%</td><td>55%</td><td>50%</td><td>5%</td><td>10%</td><td>60%</td><td>65%</td></tr><tr><td>Hanging Mug</td><td>0%</td><td>0%</td><td>10%</td><td>15%</td><td>0%</td><td>0%</td><td>15%</td><td>5%</td></tr><tr><td>Lift Pot</td><td>25%</td><td>0%</td><td>45%</td><td>45%</td><td>0%</td><td>0%</td><td>30%</td><td>40%</td></tr><tr><td>Move Can Pot</td><td>0%</td><td>0%</td><td>25%</td><td>50%</td><td>15%</td><td>15%</td><td>50%</td><td>55%</td></tr><tr><td>Move Pillbottle Pad</td><td>0%</td><td>10%</td><td>55%</td><td>40%</td><td>15%</td><td>25%</td><td>50%</td><td>70%</td></tr><tr><td>Move Playingcard Away</td><td>30%</td><td>65%</td><td>65%</td><td>80%</td><td>45%</td><td>60%</td><td>90%</td><td>85%</td></tr><tr><td>Move Stapler Pad</td><td>0%</td><td>5%</td><td>10%</td><td>10%</td><td>5%</td><td>0%</td><td>15%</td><td>10%</td></tr><tr><td>Open Laptop</td><td>0%</td><td>15%</td><td>70%</td><td>55%</td><td>20%</td><td>20%</td><td>75%</td><td>90%</td></tr><tr><td>Open Microwave</td><td>0%</td><td>5%</td><td>15%</td><td>20%</td><td>20%</td><td>15%</td><td>20%</td><td>35%</td></tr><tr><td>Pick Diverse Bottles</td><td>0%</td><td>0%</td><td>40%</td><td>45%</td><td>0%</td><td>5%</td><td>35%</td><td>40%</td></tr><tr><td>Pick Dual Bottles</td><td>0%</td><td>0%</td><td>30%</td><td>55%</td><td>0%</td><td>5%</td><td>60%</td><td>40%</td></tr><tr><td>Place A2B Left</td><td>40%</td><td>30%</td><td>85%</td><td>80%</td><td>40%</td><td>40%</td><td>75%</td><td>80%</td></tr><tr><td>Place A2B Right</td><td>35%</td><td>25%</td><td>65%</td><td>70%</td><td>45%</td><td>35%</td><td>65%</td><td>70%</td></tr><tr><td>Place Bread Basket</td><td>25%</td><td>20%</td><td>70%</td><td>40%</td><td>20%</td><td>20%</td><td>45%</td><td>65%</td></tr><tr><td>Place Bread Skillet</td><td>40%</td><td>35%</td><td>50%</td><td>50%</td><td>15%</td><td>10%</td><td>45%</td><td>60%</td></tr><tr><td>Place Burger Fries</td><td>5%</td><td>10%</td><td>80%</td><td>70%</td><td>25%</td><td>40%</td><td>80%</td><td>80%</td></tr><tr><td>Place Can Basket</td><td>0%</td><td>5%</td><td>20%</td><td>30%</td><td>0%</td><td>5%</td><td>30%</td><td>50%</td></tr><tr><td>Place Cans Plasticbox</td><td>0%</td><td>0%</td><td>50%</td><td>25%</td><td>0%</td><td>0%</td><td>30%</td><td>35%</td></tr><tr><td>Place Container Plate</td><td>55%</td><td>60%</td><td>85%</td><td>96%</td><td>65%</td><td>50%</td><td>90%</td><td>90%</td></tr><tr><td>Place Dual Shoes</td><td>5%</td><td>0%</td><td>10%</td><td>20%</td><td>0%</td><td>0%</td><td>0%</td><td>15%</td></tr><tr><td>Place Empty Cup</td><td>35%</td><td>35%</td><td>80%</td><td>85%</td><td>35%</td><td>10%</td><td>50%</td><td>75%</td></tr><tr><td>Place Fan</td><td>0%</td><td>10%</td><td>40%</td><td>40%</td><td>5%</td><td>5%</td><td>25%</td><td>30%</td></tr><tr><td>Place Mouse Pad</td><td>5%</td><td>10%</td><td>30%</td><td>30%</td><td>5%</td><td>5%</td><td>30%</td><td>55%</td></tr><tr><td>Place Object Basket</td><td>5%</td><td>5%</td><td>45%</td><td>25%</td><td>20%</td><td>25%</td><td>50%</td><td>60%</td></tr><tr><td>Place Object Scale</td><td>10%</td><td>30%</td><td>70%</td><td>45%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td>Place Object Stand</td><td>35%</td><td>45%</td><td>80%</td><td>90%</td><td>35%</td><td>50%</td><td>75%</td><td>85%</td></tr><tr><td>Place Phone Stand</td><td>25%</td><td>0%</td><td>65%</td><td>55%</td><td>10%</td><td>0%</td><td>40%</td><td>75%</td></tr><tr><td>Place Shoe</td><td>25%</td><td>25%</td><td>50%</td><td>40%</td><td>25%</td><td>50%</td><td>75%</td><td>50%</td></tr><tr><td>Press Stapler</td><td>45%</td><td>45%</td><td>60%</td><td>55%</td><td>55%</td><td>75%</td><td>85%</td><td>85%</td></tr><tr><td>Put Bottles Dustbin</td><td>0%</td><td>0%</td><td>5%</td><td>0%</td><td>0%</td><td>0%</td><td>5%</td><td>0%</td></tr><tr><td>Put Object Cabinet</td><td>0%</td><td>0%</td><td>25%</td><td>35%</td><td>0%</td><td>0%</td><td>45%</td><td>40%</td></tr><tr><td>Rotate QRcode</td><td>10%</td><td>10%</td><td>30%</td><td>35%</td><td>0%</td><td>5%</td><td>60%</td><td>35%</td></tr><tr><td>Scan Object</td><td>0%</td><td>10%</td><td>20% 100%</td><td>25% 100%</td><td>0% 85%</td><td>5% 100%</td><td>25% 90%</td><td>20% 95%</td></tr><tr><td>Shake Bottle Shake Bottle Horizontally</td><td>95% 90%</td><td>85% 95%</td></table>