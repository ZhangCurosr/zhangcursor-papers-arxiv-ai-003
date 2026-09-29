# ROBOFL: FEDERATED EXPERT ASSEMBLY FOR WORLD ACTION MODELS

Rongyu Zhang<sup>∗</sup> Nanjing University

Ruizhi Fan<sup>∗</sup> Nanjing University

Yunfan Lou Peking University

Hengyu Fang Nanjing University

Shenli Zhang Nanjing University

Chenrui Wu Simon Fraser University

Yili Jin Simon Fraser University

Li Du Nanjing University

Dan Wang Hong Kong University of Science and Technology

Yuan Du<sup>†</sup> Nanjing University

Shanghang Zhang<sup>†</sup> Peking University<sup>‡</sup>

![](images/41d39d8f2b7ab81fcccd4c04b07a5de906f146f73b35198cc27fab0506ca31c0.jpg)  
Figure 1: From federated parameter averaging to prior-informed expert assembly. Left: Aggregating independently trained LoRA-MoE models can dilute local specialization and introduce inconsistent routing across heterogeneous clients. Middle: ROBOFL organizes institutions into task silos and directly installs their task-trained LoRA into the expert branches of a server MoE. Right: ROBOFL is evaluated in the RoboTwin 2.0 and RLBench simulator, and on a real-world Franka robot.

## ABSTRACT

Vision-language-action and world-action models are increasingly popular, yet remain bottlenecked by physical interaction data that is scarce, institutionally siloed, and task-heterogeneous. A natural federated solution is to let each client adapt a shared foundation model through parameter-efficient fine-tuning, avoiding the exchange of full-model updates. However, federating these adapters is nontrivial, as naive aggregation can entangle incompatible updates, while incorporating MoEstyle routing into federated aggregation may dilute specialization and destabilize expert selection. We present ROBOFL, which instantiates MOSAIC (Mixture of Slotted Adapters) for federated world-action learning. MoSAIC directly installs locally trained LoRA adapters as the expert branches of a server MoE. Server-side routers learn token assignments over these prior-informed branches while jointly refining routing and expert parameters. Foresight-to-Action Routing Distillation (FARD) aligns routing across the model’s three paths, while Path-Consensus Expert Aggregation (PCEA) converts complete expert updates into a compact global adapter for personalized redistribution. Experiments on RoboTwin 2.0, RLBench, and a real-world Franka robot arm show the superiority of ROBOFL with structured expert assembly, as it outperforms centralized PEFT InternVLA-A1 by 12.23% on the Franka arm, while reducing per-round client communication by up to 86.81% relative to MoE-based federated VLA baselines.

## 1 INTRODUCTION

Vision-language-action (VLA) models (Brohan et al., 2023; Driess et al., 2023; Ghosh et al., 2024; Kim et al., 2025; Black et al., 2025) and world-action models (WAMs) (Cai et al., 2026; Bi et al., 2026; Ye et al., 2026b) unify perception, language understanding, future prediction, and control in a single policy. Their progress has been driven by broader multimodal and multi-robot training, flowbased action generation, and future-state modeling. However, general-purpose embodied intelligence remains constrained by real interaction data. Unlike web data, robot demonstrations require hardware, expert teleoperation, repeated resets, safety monitoring, and synchronized multimodal sensing. They are expensive to collect, especially for contact-rich, long-horizon, and multi-embodiment tasks, making high-quality trajectories both scarce and valuable (Wu et al., 2025; Luo et al., 2025b).

This data bottleneck is compounded by privacy and ownership constraints. Robot trajectories can expose private environments, user routines, and proprietary procedures, and their substantial acquisition cost makes them valuable institutional assets that owners are reluctant to share. Federated learning (FL) (McMahan et al., 2017; Kairouz et al., 2021) therefore provides a natural infrastructure for large-scale embodied training, letting institutions contribute complementary experience without centralizing raw data. This task-silo setting is practical rather than artificial: an institution typically owns the hardware, expertise, and suite of tasks for a focused family of manipulation tasks, so disjoint silos arise directly from existing practice. The resulting federation is nevertheless highly heterogeneous: client task families are largely disjoint and uneven in size, so each client optimizes over a different label distribution, and the aggregate problem is strongly non-IID. Under such task-level skew, local optima drift apart, and naive averaging blends transformations that are valid only within their own families. The objective is thus to combine complementary task-specific capabilities into a broadly competent policy without exposing or erasing their source.

Mixture-of-Experts (MoE) (Jacobs et al., 1991; Shazeer et al., 2017; Fedus et al., 2022) provides a natural representation for such task diversity, combining specialized experts with a shared backbone and input-dependent routing. For example, FedVLA (Miao et al., 2025) trains dual-gating MoE models on clients and aggregates their aligned-MoE trunk parameters for performance enhancement. Yet a modular architecture does not, by itself, ensure coherent federated learning. When each client trains its own MoE on a narrow task distribution, the corresponding expert branches can acquire distinct specializations, while the routers learn assignments shaped by local task frequencies. Averaging these expert updates can dilute task-specific knowledge, and aggregating the routers can disrupt the assignments that make that knowledge useful, leading to performance degradation as shown in Section 4. The key is therefore to separate local specialization from the learning of a shared routing policy.

To address this gap, we propose ROBOFL, a federated framework that combines a task-silo protocol with server-side expert assembly as shown in Figure 1. Designed for communication-constrained FL settings, ROBOFL leverages parameter-efficient fine-tuning (PEFT) (Hu et al., 2022; Wu et al., 2024) to enable clients to exchange compact LoRA adaptations rather than full model updates (Miao et al., 2025; Zhou et al., 2026). Specifically, ➊ each institution trains a single LoRA adapter on a focused family of tasks and uploads only its adaptation parameters to the server. ➋ The server assigns the uploaded adapters to fixed expert slots and composes them into a layer-wise MoE, where a router is trained to select among task-specialized LoRA experts. ➌ After server-side routing and expert refinement, the resulting expert updates are converted into a compact global adapter and blended with each client-associated expert before the next communication round.

This procedure yields a Mixture-of-Slotted-Adapters with integration and conversion (MOSAIC). By separating local expert formation from global router learning, ROBOFL enables the router to build on task-trained capabilities rather than on initially unspecialized branches. The fixed clientto-slot mapping preserves parameter correspondence across rounds, while still allowing flexible token-to-expert assignments. This design supports more stable routing without imposing multi-expert training or communication on clients: local training and exchanged adapters retain the footprint of single-adapter fine-tuning, while routing and multi-expert optimization remain on the server.

In addition, contemporary WAMs (Cai et al., 2026; Bi et al., 2026; Ye et al., 2026b) use aligned mixture-of-transformers (MoT) architectures (Liang et al., 2024) whose understanding, visual foresight, and action paths provide complementary evidence about the same interaction. Such a three-path structure of WAMs offers an additional source of guidance for this assembly process. MOSAIC exploits this structure to inform both expert selection and adapter conversion. Therefore, we propose Foresight-to-Action Routing Distillation (FARD), which distills reliable routing consensus between the understanding and generation paths into the action router, using a detached teacher to guide expert selection for action generation. In addition, Path-Consensus Expert Aggregation (PCEA) uses detached consensus across all three paths to weight complete server-refined LoRA updates and reconstruct a rank-constrained global adapter. Together, these mechanisms use agreement across what the model understands, predicts, and executes to guide which experts it activates during server training and how it consolidates their knowledge for subsequent client training.

Finally, we evaluate ROBOFL on RoboTwin 2.0 (Mu et al., 2025), RLBench (James et al., 2020), and a real-world Franka robot arm against centralized PEFT WAMs, as well as FL-based VLA methods, as shown on the right side of Figure 1. Experiments across simulation and the real world show that ROBOFL outperforms existing federated VLA methods and, in certain scenarios, even centralized PEFT WAMs, as it attains 83.12% overall on RoboTwin 2.0, 4.2% above federated averaging and 1.52% above centralized InternVLA-A1, reaches 59.17% against 46.94% for centralized InternVLA on six real-world Franka tasks, and cuts client communication by up to 86.81% against MoE-based federated VLA baselines.

Our contributions are summarized as follows.

1. Task-siloed federated expert assembly. We propose ROBOFL with MOSAIC to assemble client-trained LoRA adapters into a server-routed MoE, keeping raw trajectories local and retaining single-adapter client training and communication.

2. Foresight-guided action routing. We introduce FARD to distill reliable understandinggeneration routing consensus into the action router for MoT-based world action models.

3. Path-consensus adapter conversion. We introduce PCEA to aggregate complete serverrefined expert updates using three-path consensus and reconstruct a rank-constrained global adapter for personalized redistribution.

## 2 RELATED WORKS

Vision-language-action and world-action models. VLA models connect multimodal knowledge to robot control, from RT-2 (Brohan et al., 2023) and PaLM-E (Driess et al., 2023) to multi-robot policies such as Octo (Ghosh et al., 2024) and OpenVLA (Kim et al., 2025). Subsequent work advances flow-based control (Black et al., 2025), dual-system architectures (Bjorck et al., 2025), action tokenization (Pertsch et al., 2025), spatial representations (Qu et al., 2025), and efficient adaptation (Zhang et al., 2026). Cosmos (Agarwal et al., 2025) and recent WAMs (Ye et al., 2026a; Li et al., 2026) further exploit future-state prediction. These systems generally assume centralized training rather than assembling future-aware policies from institutionally distributed task data.

VLA and WAMs with MoE and FL. MoE models (Fedus et al., 2022; Zhang et al., 2025) support specialized computation, including expert selection and weighting decoupled in AdaMoE (Shen et al., 2025), force-aware routing in ForceVLA (Yu et al., 2025), and layer-wise activation in MoLe-VLA (Zhang et al., 2026). FLAME (Betran et al., 2025) benchmarks robotic FL, while FedVLA (Miao et al., 2025) combines federated training with dual-gating MoE and ForgeVLA (Zhou et al., 2026) addresses unannotated vision-action logs through instruction recovery, contrastive planning, and adaptive aggregation. Training MoE routers on narrow client distributions can nevertheless produce inconsistent expert assignments. ROBOFL separates local adapter specialization from server-side routing and jointly refines uploaded experts and module-specific routers on server data.

Federated PEFT and MoE. LoRA (Hu et al., 2022) provides compact low-rank updates for federated adaptation. FLoRA (Wang et al., 2024), FRLoRA (Yan et al., 2025), FedSA-LoRA (Guo et al., 2025), and LoRA-A2 (Koo et al., 2025) study low-rank aggregation and sharing, while FedFisher (Jhunjhunwala et al., 2024) addresses one-shot aggregation. Mixture of LoRA Experts routes among adapters (Wu et al., 2024), FedMoE (Mei et al., 2024) integrates client-specific sub-MoEs, and pFedMoAP (Luo et al., 2025a) exchanges prompt experts for personalization. In contrast, ROBOFL installs single-adapter client updates as server MoE experts without federating selection routers, then uses cross-path consensus to convert complete expert updates into compact adapters for personalized redistribution.

![](images/7ae8310cead7d3a8a800f840a44710955c126c33dd4b2ef6c2c039caf1ab8498.jpg)  
Figure 2: Overview of ROBOFL. Left: Modality-specific tokens interact through unified masked self-attention, while FARD distills detached understanding-generation consensus into action routing. Right: MOSAIC installs task-trained LoRA adapters as server MoE experts and jointly refines routers and experts. PCEA aggregates complete expert updates using three-path consensus to form a rank-constrained global adapter, which is blended with each server-refined client expert.

## 3 METHOD

## 3.1 PRELIMINARY

Three-Path World-Action Architecture. ROBOFL is built on a three-path world-action architecture, in which the understanding (U), generation (G), and action (A) paths are coupled through unified masked self-attention. U encodes current images and instructions with Qwen3-VL. G processes Cosmos-tokenized past and current frames to predict future-frame latents. A receives a projected state token, followed by noisy action tokens fused with a time embedding. At each aligned layer, path-specific queries, keys, and values are concatenated along the token dimension for joint attention, then split for path-specific output projections and MLPs. Thus, the paths share the attention computation, not projection or MLP weights. The block-causal mask orders tokens as $U \mid G \mid$ state | actions: each block attends to itself and preceding blocks, subject to padding masks. Action tokens attend bidirectionally within a chunk and use $U / G$ hidden context, not decoded future images.

At matching layer/projection m, let $q _ { m , b } ^ { P } \in \mathbb { R } ^ { E }$ be the sample-wise mean of the full pre-top-k router probabilities for path $P \in \{ U , G , A \}$ . Specifically, pooling masks out invalid observations and language positions in the understanding path U and invalid visual positions in the generation path G. In A, it includes all action positions, excluding the state token but not action padding. Cross-path comparisons use samples with valid pooled distributions in all three paths.

Flow-Matching Action Policy. For a demonstrated action chunk $\boldsymbol { a } \in \mathbb { R } ^ { H \times d _ { a } }$ , noise $\epsilon \sim \mathcal { N } ( 0 , I )$ and time $t \sim p _ { \mathrm { t i m e } }$ , the action path predicts the conditional velocity

$$
\begin{array} { r } { x _ { t } = ( 1 - t ) a + t \epsilon , \qquad u _ { t } = \epsilon - a , \qquad } \\ { \mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } _ { a , \epsilon , t } \left[ \mathrm { M S E } _ { \mathrm { v a l i d } } ( v _ { \theta } ( x _ { t } , t , c ) , u _ { t } ) \right] , } \end{array}\tag{1}
$$

where c denotes $U / G$ context and the robot state, and MSE averages over actual action dimensions and non-padded steps when padding labels are available. At inference, explicit Euler integration proceeds from $x _ { 1 } \ \bar { \sim } \ { \mathcal { N } } ( 0 , I )$ at t = 1 to t = 0. The generation loss $\mathcal { L } _ { \mathrm { g e n } }$ is the MSE between predicted and target future Cosmos latents over valid views.

## 3.2 MOSAIC: MIXTURE OF SLOTTED ADAPTERS

As shown in Figure 2, each of the K clients trains a standard LoRA adapter for each layer on its private task-family dataset $D _ { k }$ . The server maintains $E = K > 1$ complete expert branches and one selection router per adapted projection. A slot identifies a client-associated LoRA adapter within a module, not a separate model.

Adapter Slotting. In round τ , client k uploads its adapter factors, which directly overwrite slot k at each adapted module m:

$$
( A _ { m , k } ^ { \tau , 0 } , B _ { m , k } ^ { \tau , 0 } )  ( A _ { m , k } ^ { \mathrm { l o c } , \tau } , B _ { m , k } ^ { \mathrm { l o c } , \tau } ) .\tag{2}
$$

No averaging occurs during installation. Client-to-slot identities remain fixed without restricting token assignments, and expert factors are refreshed each round, while selection routers persist only on the server. The protocol exchanges parameters, not raw trajectories.

Routed Expert Integration. For token representation z, the adapted projection is, omitting bias,

$$
h _ { m } ( z ) = W _ { m } ^ { 0 } z + \gamma _ { m } \sum _ { e = 1 } ^ { E } \pi _ { m , e } ( z ) B _ { m , e } A _ { m , e } z , \qquad \gamma _ { m } = \alpha _ { m } / r ,\tag{3}
$$

where $W _ { m } ^ { 0 }$ is frozen and $\underline { { \pi } } _ { m }$ contains renormalized top- $\boldsymbol { \cdot } k _ { \mathrm { r o u t e } }$ weights, distinct from the dense probabilities pooled into $q _ { m } ^ { P }$ . The server jointly refines selection routers and both expert factors on mixed-task server data, starting from task-trained adapters. Path indices on weights are suppressed while each path retains its own parameters. Uploaded shared action-head weights are uniformly averaged across clients, refined at the server, and copied back to all clients.

Global Adapter Conversion. Let $\Delta W _ { m , e } ^ { \mathrm { s r v } , \tau } = \gamma _ { m } B _ { m , e } ^ { \mathrm { s r v } , \tau } A _ { m , e } ^ { \mathrm { s r v } , \tau }$ denote the complete server-refined adapter residual relative to the frozen backbone. PCEA converts these updates into a rank-constrained global adapter using detached routing evidence, as detailed below. This static conversion prepares client parameters rather than guaranteeing equivalence to the input-dependent MoE.

Personalized Adapter Redistribution. Client k receives standard LoRA factors representing

$$
\begin{array} { r } { \Delta W _ { m , k } ^ { \mathrm { i n i t } , \tau + 1 } = \Pi _ { r } \left( \frac { 1 } { 2 } \Delta W _ { m , k } ^ { \mathrm { s r v } , \tau } + \frac { 1 } { 2 } \Delta W _ { m } ^ { \mathrm { g l o b } , \tau } \right) , } \end{array}\tag{4}
$$

where Π is a truncated SVD approximation of rank at most $^ { r } \cdot$ The client-specific component is the post-server expert, not the original upload, which enables personalized parameter aggregation.

## 3.3 FARD: FORESIGHT-TO-ACTION ROUTING DISTILLATION

FARD uses understanding-generation agreement to guide action routing. For each valid modulesample pair (m, b), define the geometric-mean teacher and its reliability:

$$
\begin{array} { l l } { { \displaystyle a _ { m , b } = \sum _ { e } \sqrt { q _ { m , b , e } ^ { U } q _ { m , b , e } ^ { G } } , \quad } } & { { q _ { m , b } ^ { U G } = \frac { \sqrt { q _ { m , b } ^ { U } q _ { m , b } ^ { G } } } { a _ { m , b } } , } } \\ { { \displaystyle r _ { m , b } = a _ { m , b } \left[ 1 - \frac { H ( q _ { m , b } ^ { U G } ) } { \log E } \right] _ { + } . } } & { { } } \end{array}\tag{5}
$$

Thus, reliability combines path agreement with teacher concentration, so that unreliable routing signals are down-weighted during training. For valid samples $\nu _ { m } ,$ the loss is

$$
\mathcal { L } _ { \mathrm { F A R D } } = \mathrm { m e a n } _ { m } \left[ \frac { 1 } { | \mathcal { V } _ { m } | } \sum _ { b \in \mathcal { V } _ { m } } \mathrm { s g } ( r _ { m , b } ) D _ { \mathrm { J S } } \big ( \mathrm { s g } ( q _ { m , b } ^ { U G } ) , q _ { m , b } ^ { A } \big ) \right] ,\tag{6}
$$

averaged over modules with nonempty $\nu _ { m }$ (zero if none). Here sg stops gradients while entropy and JS use natural logarithms, with numerical probability floors omitted for clarity. Gradients pass

through the action-distribution argument, not the teacher or reliability, and upstream $U / G$ parameters can still receive indirect gradients through joint attention. The overall server objective therefore minimizes

$$
{ \mathcal { L } } _ { \mathrm { M o S A I C } } = { \mathcal { L } } _ { \mathrm { a c t i o n } } + \lambda _ { \mathrm { g e n } } { \mathcal { L } } _ { \mathrm { g e n } } + \lambda _ { \mathrm { a u x } } { \mathcal { L } } _ { \mathrm { a u x } } + \lambda _ { \mathrm { F A R D } } ^ { ( \tau ) } { \mathcal { L } } _ { \mathrm { F A R D } } ,\tag{7}
$$

where $\mathcal { L } _ { \mathrm { a u x } }$ denotes the module-level auxiliary load-balancing loss averaged over the $E$ experts of each module, and $\lambda _ { \mathrm { F A R D } } ^ { ( \tau ) }$ follows the round-wise warmup schedule.

## 3.4 PCEA: PATH-CONSENSUS EXPERT AGGREGATION

PCEA weights complete expert updates rather than averaging their factors, since, in general,

$$
\sum _ { e } \omega _ { e } B _ { e } A _ { e } \neq \left( \sum _ { e } \omega _ { e } B _ { e } \right) \left( \sum _ { e } \omega _ { e } A _ { e } \right) .\tag{8}
$$

During server training, it collects detached, unnormalized Hellinger-barycenter evidence:

$$
g _ { m , b , e } = \left( \frac { \sqrt { q _ { m , b , e } ^ { U } } + \sqrt { q _ { m , b , e } ^ { G } } + \sqrt { q _ { m , b , e } ^ { A } } } { 3 } \right) ^ { 2 } .\tag{9}
$$

Summing the valid sample occurrences across server steps and ranks within a round gives:

$$
s _ { m , e } = \sum _ { b } g _ { m , b , e } , \qquad w _ { m , e } = \frac { s _ { m , e } } { \sum _ { j } s _ { m , j } } , \qquad \bar { a } _ { m } = \frac { \sum _ { e } s _ { m , e } } { N _ { m } } ,\tag{10}
$$

where $N _ { m }$ counts sample occurrences. The evidence retains its agreement mass rather than being normalized per sample. For observed module triplets M, PCEA computes agreement-weighted global weights and a Hellinger midpoint:

$$
\bar { w } _ { e } = \frac { \sum _ { m \in \mathcal { M } } \bar { a } _ { m } w _ { m , e } } { \sum _ { m \in \mathcal { M } } \bar { a } _ { m } } , \qquad \widetilde { w } _ { m , e } = \frac { \big ( \sqrt { w _ { m , e } } + \sqrt { \bar { w } _ { e } } \big ) ^ { 2 } } { \sum _ { j } \big ( \sqrt { w _ { m , j } } + \sqrt { \bar { w } _ { j } } \big ) ^ { 2 } } .\tag{11}
$$

Matched $U / G / A$ modules share these weights. Eligible unobserved modules use $\bar { w } ,$ and PCEA requires nonempty evidence. The unmatched state-projection adapter is aggregated uniformly.

For each physical module’s own post-server factors, conversion gives:

$$
\Delta W _ { m } ^ { \mathrm { g l o b , \tau } } = \Pi _ { r } \left( \sum _ { e } \widetilde { w } _ { m , e } \Delta W _ { m , e } ^ { \mathrm { s r v } , \tau } \right) .\tag{12}
$$

Implementation uses QR reduction on the stacked weighted factors, followed by a small SVD. FARD guides routing during server training, while PCEA consolidates the updates for client training.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Evaluation Protocol. We evaluate ROBOFL under the task-silo protocol on RoboTwin 2.0 (Mu et al., 2025), RLBench (James et al., 2020) and Franka robots<sup>1</sup>. We partition the 50 tasks into 8 client task-family groups and 1 server residual group for RoboTwin 2.0 random, and split the 8 RLBench tasks into 4 client partitions and 1 server partition, with 6 tasks distributed across the four clients and 2 tasks retained on the server. As for real-world applications, we partition 6 manipulation tasks into 4 clients and 1 server for Franka, creating an extreme non-IID data scenario.

Table 1: RoboTwin 2.0 success rates on a 25-task challenging subset under random condition (selected as tasks where FedAvg < 80%). Overall reports performance on all 50 tasks.
<table><tr><td colspan="10">7t pck Method size blo. div. pk ham. &quot;mug pot can sta. dual beat rank hand hang move move place</td><td>L place</td><td>R</td><td>ski. bread</td><td>• basket</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">CENTRALIZED</td><td>TRAINING</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>INTERNVLA</td><td>73</td><td>78</td><td>62</td><td>27</td><td>32</td><td>76</td><td>53</td><td>68</td><td>74</td><td>85</td><td>84</td><td>79</td><td>60</td></tr><tr><td>MOTUS</td><td>82</td><td>86</td><td>60</td><td>20</td><td>40</td><td>86</td><td>48</td><td>70</td><td>80</td><td>85</td><td>80</td><td>80</td><td>69</td></tr><tr><td></td><td colspan="10">FEDERATED</td><td></td><td></td><td></td><td></td></tr><tr><td>FEDAVG</td><td>75</td><td>74</td><td>52</td><td>24</td><td>33</td><td>67</td><td>39</td><td>69</td><td>77</td><td>79</td><td>79</td><td>74</td><td>72</td></tr><tr><td>FEDMOE</td><td>69</td><td>71</td><td>42</td><td>21</td><td>34</td><td>57</td><td>34</td><td>61</td><td>70</td><td>84</td><td>80</td><td>66</td><td>64</td></tr><tr><td>FORGEVLA</td><td>75</td><td>83</td><td>57</td><td>35</td><td>40</td><td>83</td><td>44</td><td>67</td><td>82</td><td>87</td><td>82</td><td>78</td><td>68</td></tr><tr><td>FEDVLA</td><td>73</td><td>85</td><td>61</td><td>26</td><td>33</td><td>85</td><td>45</td><td>62</td><td>84</td><td>80</td><td>82</td><td>77</td><td>74</td></tr><tr><td>ROBOFL</td><td>78</td><td>92</td><td>73</td><td>30</td><td>24</td><td>87</td><td>50</td><td>72</td><td>84</td><td>87</td><td>83</td><td>80</td><td>82</td></tr><tr><td>Method</td><td>shoe dual</td><td>pad mouse</td><td>bas. place</td><td>sca. place</td><td>sta. press</td><td>dus. bottles</td><td>and cab.</td><td>QR rotate</td><td>obj scan</td><td>bow. stack</td><td>seal st amp</td><td>turn</td><td>swi. Overall</td></tr><tr><td>CENTRALIZED</td><td colspan="10"></td><td></td><td></td><td></td><td></td></tr><tr><td>INTERNVLA</td><td>64</td><td>70</td><td>82</td><td>81</td><td>74</td><td>90</td><td>68</td><td>77</td><td>64</td><td>86</td><td>68</td><td>43</td><td>81.60</td></tr><tr><td>MOTUS</td><td>73</td><td>64</td><td>79</td><td>73</td><td>72</td><td>89</td><td>67</td><td>87</td><td>65</td><td>77</td><td>66</td><td>45</td><td>80.96</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="10">FEDERATED</td></tr><tr><td>FEDAVG</td><td>67</td><td>56</td><td>79</td><td>75</td><td>75</td><td>77</td><td>68</td><td>79</td><td>58</td><td>79</td><td>57</td><td>48</td><td>78.92</td></tr><tr><td>FEDMOE</td><td>60</td><td>54</td><td>75</td><td>69</td><td>74</td><td>79</td><td>62</td><td>73</td><td>31</td><td>78</td><td>60</td><td>38</td><td>75.12</td></tr><tr><td>FORGEVLA</td><td>70</td><td>59</td><td>80</td><td>77</td><td>78</td><td>83</td><td>61</td><td>77</td><td>69</td><td>75</td><td>59</td><td>52</td><td>80.70</td></tr><tr><td>FEDVLA</td><td>71</td><td>53</td><td>70</td><td>75</td><td>75</td><td>84</td><td>60</td><td>75</td><td>70</td><td>80</td><td>65</td><td>51</td><td>79.32</td></tr><tr><td>ROBOFL</td><td>74</td><td>58</td><td>72</td><td>74</td><td>74</td><td>90</td><td>68</td><td>81</td><td>74</td><td>86</td><td>70</td><td>53</td><td>83.12</td></tr></table>

Federated Optimization. All experiments initialize from the pretrained InternVLA-A1-3B checkpoint, use full client participation, 100 communication rounds, and one local epoch. Client and server steps are 500/200/300 (RoboTwin / RLBench / Franka); client batch size is 32, and server batch size is $8 / 1 6 / \dot { 1 } 6$ . Both optimizers are AdamW with learning rate $1 \times 1 0 ^ { - 4 }$ , betas $( 0 . 9 , 0 . 9 5 )$ , epsilon $1 0 ^ { - 8 }$ , weight decay 0.01, and gradient-norm clipping at 1.0, using cosine decay from $1 \times 1 0 ^ { - 4 }$ to $1 \times 1 0 ^ { - 5 }$ over 50,000/20,000/30,000 decay steps and 2,500/400/600 warmup steps.

MoSAIC Configuration. MoSAIC uses eight expert branches per adapted module with top-4 server-side routing for RoboTwin 2.0, and four expert branches per adapted module with top-2 serverside routing for RLBench and the real-world applications. Each expert uses rank $r = 1 6 ,$ LoRA parameter $\alpha = 3 2$ (effective scaling $\alpha / r = 2 )$ , and dropout 0.05. The generation and load-balancing coefficients are $\lambda _ { \mathrm { g e n } } = 0 . 0 1$ and $\lambda _ { \mathrm { a u x } } = 0 . 0 0 1$ . FARD uses $\lambda _ { \mathrm { F A R D } } = 0 . 0 1$ with a 10-round warmup.

Baselines. We compare against InternVLA-A1 (Cai et al., 2026) and Motus (Bi et al., 2026), which are centralized PEFT references; FedAvg (McMahan et al., 2017) performs standard weight averaging; FedMoE (Mei et al., 2024) uses client-specific expert sub-MoEs; FedVLA (Miao et al., 2025) combines instruction-oriented scene parsing, dual-gating MoE, and expert-driven aggregation; and ForgeVLA (Zhou et al., 2026) targets heterogeneous federated robot data with weak or missing language supervision. We adapt them to the same FL-PEFT with Kaiming-random-initialized adapter.

## 4.2 QUANTITATIVE RESULTS FOR SIMULATION

Model Performance on RoboTwin 2.0. Table 1 reports the 25 RoboTwin 2.0 tasks on which FedAvg stays below 80%, since the remaining tasks are near-saturated and thus uninformative. ROBOFL attains the highest overall success rate of 83.12%, exceeding FedAvg by 4.20% and also surpassing centralized PEFT with InternVLA-A1 and Motus. This pattern matches MOSAIC’s design: retaining locally specialized adapters as expert branches and learning their input-dependent composition rather than immediately averaging them. Best or joint-best results on 15 displayed tasks show that the gains extend across tasks. Overall, ROBOFL delivers the strongest performance in terms of single-adapter client cost, demonstrating that server-side expert assembly can outperform both federated averaging and centralized PEFT across heterogeneous task silos.

Model Performance on RLBench. We further examine how ROBOFLperforms on RLBench with only eight tasks under a 4-client, 4-expert MoSAIC in Table 2. ROBOFL remains the best FL method at 41.25%, yet the margin over centralized

Table 2: Average success rates (%) for eight RLBench tasks with 4-expert MoSAIC for ROBOFL.
<table><tr><td>Method</td><td>InternVLA Motus</td><td>ForgeVLA FedVLA</td><td>ROBOFL</td></tr><tr><td>Type</td><td>Central-LoRA</td><td>FL-LoRA FL-MoE</td><td>LORA→MOE</td></tr><tr><td>SUCCESS</td><td>52.75 52.50</td><td>34.25 31.00</td><td>41.25</td></tr></table>

PEFT reverses relative to RoboTwin, where ROBOFL also beats InternVLA-A1 and Motus. With fewer tasks and clients, each expert sees less complementary evidence for routed composition, so the advantage of structured expert assembly is smaller. These results suggest that ROBOFL is most effective in large-scale, task-heterogeneous regimes rather than in small, low-data suites.

## Client Resources and Communication.

Table 3 compares adaptation resources at the client boundary, combining upload and download into a per-round communication total. With one rank-16 adapter per client, ROBOFL matches FedAvg’s 40.85M trainable parameters and 311.63 MiB payload, while retaining eight server-side expert branches. Relative to MoE-based FL methods FedMoE and FedVLA, client parameters decrease by up to 86.95% and perround communication by up to 86.81%,

Table 3: Client resources and logical communication on RoboTwin 2.0. Communication combines upload and download per participating client per round.
<table><tr><td>Method</td><td>Adapters c / s</td><td>Params (M)</td><td>Comm. MiB</td><td>Mem. GiB</td><td>Success (%)</td></tr><tr><td>FEDAVG</td><td>1/1</td><td>40.85</td><td>311.63</td><td>8.33</td><td>78.92</td></tr><tr><td>FEDMOE</td><td>4/8</td><td>158.11</td><td>1,206.28</td><td>11.44</td><td>75.12</td></tr><tr><td>FORGEVLA</td><td>1/1</td><td>40.85</td><td>312.06</td><td>8.33</td><td>80.70</td></tr><tr><td>FEDVLA</td><td>8/8</td><td>312.93</td><td>2,362.85</td><td>15.59</td><td>79.32</td></tr><tr><td>ROBOFL</td><td>1/8</td><td>40.85</td><td>311.63</td><td>8.33</td><td>83.12</td></tr></table>

while RoboTwin 2.0’s success rates increase by 8 and 3.8 percentage points, respectively. Peak allocated GPU memory falls from 15.59 to 8.33 GiB in the standardized client-architecture microbenchmark. These results demonstrate that ROBOFL achieves the best accuracy-efficiency tradeoff: it attains the highest success rate with the same computational and communication overhead as single-adapter baselines, while remaining far below client-side MoE.

## 4.3 ABLATION STUDIES

Influence of FARD and PCEA. We   
present a sequential ablation of FARD and   
PCEA, using vanilla MoSAIC as the base  
line and centralized LoRA as an external   
reference under its own training schedule   
with RoboTwin 2.0, as shown in Figure 3.   
We evaluate PCEA on top of FARD be  
cause its consensus-weighted conversion is   
designed to exploit the routing agreement.   
FARD alone increases the number of tasks   
exceeding 80% success from 31 to 33, but   
reduces coverage at ≥ 90% from 19 to 17.   
Adding PCEA raises these counts to 34 and   
across 50 tasks with 100 trials per task.24, surpassing both vanilla MoSAIC and 24, surpassing both vanilla MoSAIC and

![](images/8295bd29b6f3cb4fc1a16697e01fdfcf9a71f13997c6628b5fb0209d73f9fff6.jpg)

Figure 3: Success-threshold coverage on RoboTwin.reduces coverage at ≥ 90% from 19 to 17. Bars count tasks satisfying each success-rate criterionAdding PCEA raises these counts to 34 and

centralized LoRA at the latter threshold. Tasks with 100/100 successes likewise increase from 4 for both references to 6 with FARD and 9 with FARD+PCEA. At ≥ 95%, however, both variants cover 14 tasks versus 15 for vanilla MoSAIC. These results support combining routing distillation with expert conversion, potentially preserving coordinated task knowledge across federated rounds and supporting more consistent execution and broader coverage of high-success tasks.

Routing Coupling and Intervention. We probe route-to-action coupling across 50 RoboTwin 2.0 tasks while holding the initial observation and robot state fixed. Instruction substitution changes only the language condition, whereas uniformized and scrambled controls intervene only on the actionrouter distribution. As shown in Figure 4, instruction swaps produce strong pooled route-to-action coupling, establishing that task semantics reach action generation through routing.

Table 4: Real-world applications with six single-arm Franka tasks SR (%) with three replicates.
<table><tr><td>Method</td><td>Type</td><td>adjust bottle</td><td>stamp seal</td><td>stack cups</td><td>screw bottle</td><td>pour water</td><td>test tube</td><td> $W ( \psi , \psi ) = \frac { \sqrt { 3 } } { 2 } \langle \psi _ { 0 } ^ { 2 } \rangle _ { \psi }$ </td></tr><tr><td>INTERNVLA</td><td>CENTRALIZED-LORA</td><td>63.33</td><td>53.33</td><td>41.67</td><td>31.67</td><td>48.33</td><td>43.33</td><td> $4 6 . 9 4 \pm 7 . 8 0$ </td></tr><tr><td>FORGEVLA</td><td>FL-LoRA</td><td>41.67</td><td>31.67</td><td>46.67</td><td>28.33</td><td>31.67</td><td>33.33</td><td> $3 5 . 5 6 \pm 7 . 0 3$ </td></tr><tr><td>FEDVLA</td><td>FL-MoE</td><td>43.33</td><td>41.67</td><td>33.33</td><td>13.33</td><td>40.00</td><td>30.00</td><td> $3 3 . 6 1 \pm 5 . 6 0$ </td></tr><tr><td>ROBOFL</td><td>FL-LoRA→MoE</td><td>68.33</td><td>66.67</td><td>68.33</td><td>36.67</td><td>56.67</td><td>58.33</td><td> ${ \bf 5 9 . 1 7 \pm 7 . 6 1 }$ </td></tr></table>

![](images/7991d3a3ee046208eee1df9c157b8a54112633d6d6adb40c80b0d98c42acb0ae.jpg)  
Figure 5: Qualitative successful real-world Franka rollouts for each of six single-arm Franka tasks.

The direct intervention isolates the action router more strictly. Scrambling produces a larger routing displacement for FARD, yet a smaller action displacement than vanilla ROBOFL. Thus, FARD makes routing more selective without making the resulting policy brittle. Together, the probes support FARD as a cross-path routingalignment mechanism, while PCEA performs the corresponding path-consistent expert conversion.

## 4.4 REAL-WORLD

## APPLICATIONS WITH FRANKA

Quantitative Analysis. Table 4 reports success rates on six single-arm Franka tasks. We report mean success rates over three evaluation rounds of 20 trials per task.

![](images/12a69b94c7c36ac47beafd41fd023f97d523b5a27993159e410f09ac03fc268f.jpg)  
Figure 4: Route-to-action coupling under instruction swaps and direct action-router interventions. The inset magnifies the direct intervention regime.

ROBOFL attains the highest Overall success rate of 59.17%, exceeding centralized InternVLA by 12.23% and federated ForgeVLA and FedVLA by 23.61% and 25.56%. MoSAIC therefore extends beyond simulation: installing each client’s single LoRA as a routed server expert, rather than averaging updates, preserves input-dependent specialization under real perception, contact, and hardware variability, even surpassing pooled centralized PEFT at the cost of a single adapter per client.

Qualitative Analysis. Figure 5 shows successful rollouts on six Franka tasks. Execution follows the simulation’s three-phase structure: approach, sustained contact, and sequenced state change. This closed-loop coherence reflects the method’s design: task-trained experts remain distinct under server routing, while understanding-generation consensus guides action routing, so contact-rich behaviors remain input-dependent rather than averaging into generic motions. For example, adjust bottle task seats rather than strikes, screw bottle task sustains rotation through the cap.

## 5 CONCLUSION

We presented ROBOFL, a federated framework for learning world-action models from task-siloed robot data without centralizing raw trajectories. MOSAIC installs locally trained LoRA adapters as server MoE experts and jointly refines routing and expert parameters. Specifically, FARD aligns action routing with understanding-generation consensus and PCEA consolidates expert updates into a rank-constrained global adapter for personalized redistribution. Across simulation and realrobot experiments, their combination delivers the highest overall success rates among the evaluated federated methods while keeping client training at single-adapter cost.

## AI USE STATEMENT

The authors used AI-based writing and coding assistants to help draft and polish portions of the text and to support the implementation and analysis scripts. All problem formulation, method design, experimental execution, data collection, and interpretation of results were carried out and verified by the authors, who take full responsibility for the content of this paper.

## ETHICS STATEMENT

This work studies federated learning of world-action models for robotic manipulation. Experiments in simulation use the publicly released RoboTwin 2.0 and RLBench benchmarks, and the real-robot evaluations collect no personally identifiable information. The motivating concern of the paper is that robot trajectories can reveal private environments, user routines, or proprietary procedures, and that centralized pooling of such data may be undesirable. The proposed protocol keeps raw trajectories at each institution and exchanges only adaptation parameters, thereby reducing this exposure; however, we do not claim formal privacy guarantees, such as differential privacy or secure aggregation, and parameter updates may still leak information in adversarial settings. We therefore position the method as an architectural alternative to raw-data centralization rather than as a complete privacy solution. We also note that broad manipulation policies can be misused, and we follow the ICLR Code of Ethics in conducting and reporting this work.

## REPRODUCIBILITY STATEMENT

We summarize the resources provided to support reproducibility. The full set of training and evaluation settings appears in Table 5 and Appendix A; the task-silo assignments for the eight RoboTwin 2.0 clients and the four RLBench clients are listed in Tables 6 and 8; and the observation, action, and optimization interfaces are specified in the same appendix. The client resources, logical communication accounting, and the architecture profiles behind the memory measurements are documented in Appendix A.1. The MoSAIC, FARD, and PCEA formulations, including the adapter slotting, router integration, personalized redistribution, and consensus-weighting rules, are given in the Method section and implemented as described in Appendix A.1. We plan to release the source code, launch configurations, and evaluation scripts as supplementary materials; these will include data preprocessing and normalization steps, evaluation packages for both benchmarks, and scripts used to generate the resource and ablation figures. Simulation results use a single fixed evaluation seed; real-robot results use a fixed scripted protocol with 20 trials per task. Differences below a few points, and per-task swings on the 8-task RLBench suite, should be read as descriptive rather than statistically established.

## ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China (62476011) and (625B2090), the Beijing Natural Science Foundation (L252060).

## REFERENCES

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Santiago Bou Betran, Alberta Longhini, Miguel Vasco, Yuchong Zhang, and Danica Kragic. Flame: A federated learning benchmark for robotic manipulation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2494–2500. IEEE, 2025.

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 35101–35113, 2026.

Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Jim Fan,˜ Yu Fang, Dieter Fox, Fengyuan Hu, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. In Proceedings of Robotics: Science and Systems, Los Angeles, California, USA, 2025. doi: 10.15607/RSS.2025.XXI.010.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, et al. Internvla-a1: Unifying understanding, generation and action for robotic manipulation. arXiv preprint arXiv:2601.02456, 2026.

Hong-You Chen and Wei-Lun Chao. FedBE: Making bayesian model ensemble applicable to federated learning. In International Conference on Learning Representations, 2021.

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In Proceedings of the 40th International Conference on Machine Learning, pp. 8469–8488. PMLR, 2023.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.

Pengxin Guo, Shuang Zeng, Yanran Wang, Huijie Fan, Feifei Wang, and Liangqiong Qu. Selective aggregation for low-rank adaptation in federated learning. In International Conference on Learning Representations, 2025.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991.

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019– 3026, 2020.

Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi. Fedfisher: Leveraging fisher information for one-shot federated learning. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238, pp. 1612–1620. PMLR, 2024.

Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurelien Bellet, Mehdi Bennis, Arjun Nitin ´ Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and Trends in Machine Learning, 14(1–2): 1–210, 2021.

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 7482–7491, 2018.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Grace Lam, Pannag R. Sanketi, et al. Openvla: An open-source vision-language-action model. In Proceedings of The 8th Conference on Robot Learning, pp. 2679–2713. PMLR, 2025.

Jabin Koo, Minwoo Jang, and Jungseul Ok. Towards robust and efficient federated low-rank adaptation with heterogeneous clients. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 416–429, 2025.

Zhe Li, Zhenzhe Zhang, Yangyang Wei, Wenjie Zhang, Xichen Yuan, Peiyuan Zhi, Gen Li, Xinying Guo, Fengjie Gao, Jianfei Yang, et al. omega-0: A latent predictive world action model for concurrent humanoid loco-manipulation. arXiv preprint arXiv:2608.06375, 2026.

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

Tao Lin, Lingjing Kong, Sebastian U. Stich, and Martin Jaggi. Ensemble distillation for robust model fusion in federated learning. In Advances in Neural Information Processing Systems, volume 33, pp. 2351–2363, 2020.

Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. In International Conference on Learning Representations, 2020.

Jun Luo, Chen Chen, and Shandong Wu. Mixture of experts made personalized: Federated prompt learning for vision-language models. In International Conference on Learning Representations, 2025a.

Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Xiangju Mi, Mengdi Zhao, Bo-Wen Zhang, Cheng Chi, Jiaming Liu, Gaole Dai, et al. Robobench: A comprehensive evaluation benchmark for multimodal large language models as embodied brain. arXiv preprint arXiv:2510.17801, 2025b.

Yishay Mansour, Mehryar Mohri, Jae Ro, and Ananda Theertha Suresh. Three approaches for personalization with applications to federated learning. arXiv preprint arXiv:2002.10619, 2020.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pp. 1273–1282. PMLR, 2017.

Hanzi Mei, Dongqi Cai, Ao Zhou, Shangguang Wang, and Mengwei Xu. Fedmoe: Personalized federated learning via heterogeneous mixture of experts. arXiv preprint arXiv:2408.11304, 2024.

Cui Miao, Tao Chang, Meihan Wu, Hongbin Xu, Chun Li, Ming Li, and Xiaodong Wang. Fedvla: Federated vision-language-action learning with dual gating mixture-of-experts for robotic manipulation. arXiv preprint arXiv:2508.02190, 2025.

Yao Mu, Tianxing Chen, Shijia Peng, Zanxin Chen, Zeyu Gao, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo. Robotwin: Dual-arm robot benchmark with generative digital twins. arXiv preprint arXiv:2409.02920, 2025.

Karl Pertsch, Konrad Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. In Proceedings of Robotics: Science and Systems, 2025.

D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, J. Gu, Z. Wang, Y. Ding, B. Zhao, D. Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action models. In Proceedings ofRobotics: Science and Systems, 2025.

Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecnˇ y,´ Sanjiv Kumar, and H. Brendan McMahan. Adaptive federated optimization. In International Conference on Learning Representations, 2021.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Sparsely-gated mixture-of-experts. In International Conference on Learning Representations, 2017.

Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, and Yao Mu. Expertise need not monopolize: Action-specialized mixture of experts for vision-language-action learning. arXiv preprint arXiv:2510.14300, 2025.

Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, and Ang Li. Flora: Federated fine-tuning large language models with heterogeneous low-rank adaptations. Advances in Neural Information Processing Systems, 2024.

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In Proceedings of Robotics: Science and Systems, 2025. doi: 10.15607/RSS.2025.XXI.152.

Xun Wu, Shaohan Huang, and Furu Wei. Mixture of lora experts. In International Conference on Learning Representations, 2024.

Yunlu Yan, Chun-Mei Feng, Wangmeng Zuo, Rick Siow Mong Goh, Yong Liu, and Lei Zhu. Federated residual low-rank adaptation of large language models. In International Conference on Learning Representations, 2025.

Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. arXiv preprint arXiv:2603.17240, 2026a.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026b.

Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Wenqiang Zhang, and Cewu Lu. Forcevla: Enhancing vla models with a force-aware moe for contact-rich manipulation. In Advances in Neural Information Processing Systems, 2025.

Rongyu Zhang, Yijiang Liu, Huanrui Yang, Shenli Zheng, Li Du, Dan Wang, Yuan Du, and Shanghang Zhang. Moant: Mixture-of-rank-one-experts with semantic-aware intuition for multi-task large language model finetuning. In Third Conference on Language Modeling, 2025.

Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, and Shanghang Zhang. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 18764–18772, 2026.

Yuhao Zhou, Yunpeng Zhu, Yang Zhou, Jindi Lyu, Jian Lan, Zhangyuan Wang, Dan Si, Thomas Seidl, Qing Ye, and Jiancheng Lyu. Forgevla: Federated vision-language-action learning without language annotations. arXiv preprint arXiv:2605.07474, 2026.

Zhuangdi Zhu, Junyuan Hong, and Jiayu Zhou. Data-free knowledge distillation for heterogeneous federated learning. In International Conference on Machine Learning, pp. 12871–12882. PMLR, 2021.

## APPENDIX

A ADDITIONAL EXPERIMENTAL DETAILS FOR SIMULATIONS

A.1 SHARED IMPLEMENTATION DETAILS

Parameter exchange. Both benchmarks use the same adapter-based implementation. LoRA adapts the attention, MLP, and state projections. Clients upload complete LoRA factors together with the trainable action input/output projections and action-time MLP weights, while frozen backbone weights are excluded. Shared action-head weights are averaged uniformly and synchronized separately from expert conversion. Server MoE checkpoints are saved before global adapter conversion and personalized redistribution.

Server participation and shared-compute assumption. ROBOFL lets the central server take part in training rather than act only as an aggregator. We regard this as a reasonable and common use of the decentralized setting: the server is typically the best-provisioned node, and federated methods routinely exploit server-side computation, for example through server-level adaptive optimization (Reddi et al., 2021), distillation of the client ensemble on server or surrogate data (Lin et al., 2020; Chen & Chao, 2021), or data-free server-side generators (Zhu et al., 2021). Under this assumption, the server holds one residual task group and performs the routed expert-refinement stage on it, which enables a single shared router over task-trained experts; the alternative reading, in which the server has no data of its own, would leave the router without a common distribution to learn from. Crucially, this does not amount to centralized training: the server group covers only a small fraction of each benchmark (7/50, 2/8, and 2/6 tasks on RoboTwin 2.0, RLBench, and Franka), so the bulk of the task distribution still resides with the clients. The server is also not given a disproportionate optimization budget: its per-round steps match the per-client local steps (500/200/300 for the three scenarios), and its task group is comparable in size to the client partitions (3-8, 1-2, and 1 task per client). Clients therefore continue to train a single adapter, and the additional routing and multi-expert computation remains on the server.

Comparison Fairness with Centralized Training Baselines We treat centralized PEFT as a deliberately stronger reference than any federated method, not as an equal-budget competitor. A centralized run observes the union of all task families in a single optimizer, with no task silos, no non-IID client drift, and no communication rounds. That ROBOFL matches or exceeds this reference on RoboTwin 2.0 and Franka while operating under task-silo federated learning, and, consistent with its mechanism, trails it only on the low-data RLBench suite, where less complementary evidence exists for expert composition, is the intended empirical claim of Section 4.

The comparison is also capacity-matched at the point of deployment. The entity that is trained, redistributed, and evaluated is a single rank-16 LoRA adapter per client, identical in family, rank, and scaling to the adapters used by federated averaging and by the centralized PEFT references. The eight-expert server MoE is a training-time construct that never leaves the server; clients never hold routers or multiple experts, and the deployed policy is the redistributed single adapter, not the server MoE. At the client boundary, ROBOFL is therefore indistinguishable from single-adapter FL in trainable parameters, communication, and memory (Table 3), and is far below client-side MoE methods. The reported gains consequently cannot be attributed to a larger deployed model or to additional client-side capacity: they follow from how task-specialized adapters are composed on the server under heterogeneity, not from giving the learned policy more parameters than its baselines. Notably, this advantage is obtained under an extreme non-IID, task-disjoint regime in which every client still trains and communicates a single adapter.

Configuration. Table 5 summarizes the numerical settings. RLBench training entries marked † are launcher defaults, not reconstructed settings of the reported checkpoints. Task assignments and evaluation semantics are specified separately below. Last but not least, all the FL-based baselines follow the same training procedure as ROBOFL.

Table 5: Training and evaluation settings for the simulation and real-world scenarios. Execution horizon is specified independently of prediction length.
<table><tr><td>Setting</td><td>RoboTwin 2.0</td><td>RLBench</td><td>Franka</td></tr><tr><td colspan="4">Optimization and adapters</td></tr><tr><td>Pretrained</td><td colspan="3">InternVLA-A1-3B</td></tr><tr><td>checkpoint Clients / experts</td><td>8/8</td><td>4/4</td><td>4/4</td></tr><tr><td>Routing top-k</td><td>4</td><td>2</td><td>2</td></tr><tr><td>Communi. rounds</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Local steps / round</td><td>500</td><td>200</td><td>300</td></tr><tr><td>Local epochs</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Server steps / round</td><td>500</td><td>200</td><td>300</td></tr><tr><td>Peak / final LR</td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup / decay</td><td>2,500 /50,000</td><td>400/20,000</td><td>600 /30,000</td></tr><tr><td>steps LoRA rank / α</td><td></td><td></td><td></td></tr><tr><td>LoRA dropout</td><td>16/32 0.05</td><td>16/32 0.05</td><td>16/32 0.05</td></tr><tr><td> $\lambda _ { \mathrm { g e n } } / \lambda _ { \mathrm { a u x } }$ </td><td>0.01 /0.001</td><td>0.01 /0.001</td><td>0.01 /0.001</td></tr><tr><td colspan="4">Observation and action interface</td></tr><tr><td>Views</td><td>Head + two wrists</td><td>Front + two masked slots</td><td>External stand + wrist + masked slot</td></tr><tr><td>Resolution</td><td> $2 2 4 \times 2 2 4$ </td><td> $2 2 4 \times 2 2 4$ </td><td>224 × 224</td></tr><tr><td>State</td><td>Joint configuration</td><td>XYZ + Euler + gripper ∆XYZ + absolute Euler</td><td>Joint + gripper</td></tr><tr><td>Action</td><td>Joint-space offsets</td><td>+ gripper</td><td>Absolute joint targets</td></tr><tr><td>History stride</td><td>15</td><td>1</td><td>15</td></tr><tr><td>Predicted chunk</td><td>50</td><td>8</td><td>50</td></tr><tr><td>Inference steps</td><td>10</td><td>10</td><td>10</td></tr><tr><td colspan="4">Evaluation protocols</td></tr><tr><td>Evaluated tasks</td><td>50</td><td>8</td><td>6</td></tr><tr><td>Trials / task</td><td>100</td><td>50</td><td>20</td></tr><tr><td>Executed actions</td><td>30</td><td>8</td><td>50</td></tr><tr><td>Initialization</td><td>Randomized</td><td>seed 42</td><td>Operator reset</td></tr></table>

## A.2 HYPERPARAMETER AND DESIGN CHOICES

The non-obvious hyperparameters follow standard practice. The personalization coefficient $\frac { 1 } { 2 }$ in Eq. 4 represents an equal global-local model interpolation, a standard personalization strategy with generalization guarantees (Mansour et al., 2020). The local term preserves client specialization, while the global term shares server-acquired knowledge, and fixing it avoids introducing a per-client hyperparameter. The auxiliary weights are kept small because $\mathcal { L } _ { \mathrm { a c t i o n } }$ remains the primary objective and only the magnitude of such weights matters: we use $\lambda _ { \mathrm { a u x } } = 0 . 0 0 1$ (Kendall et al., 2018), with the load-balancing scale following the sparse-MoE convention (Shazeer et al., 2017; Fedus et al., 2022). FARD is likewise a routing regularizer based on detached Jensen-Shannon distillation (Hinton et al., 2015), so $\lambda _ { \mathrm { F A R D } } = 0 . 0 1$ , and it is warmed up over 10 rounds because the reliability weight is uninformative before the action router stabilizes. Warmup is a standard variance-reduction device (Liu et al., 2020).

## A.3 ROBOTWIN 2.0

Task-silo manifest. Table 6 lists the task-disjoint client and server assignments implemented by fedforesight category. Episodes remain within their assigned task partition, while the server residual group is used for routed optimization.

Table 6: Task assignments in the explicit eight-client task-silo implementation. Counts refer to tasks, not episodes or training frames.
<table><tr><td>Partition</td><td>Count</td><td>Tasks</td></tr><tr><td>Client 0</td><td>5</td><td>grab_roller,pick_diverse_bottles,pick_dual_bottles, put_bottles_dustbin,put_object_cabinet</td></tr><tr><td>Client 1</td><td>7</td><td>place_object_basket,place_bread_basket, place_bread_skillet,place_can_basket, place_cans-plasticbox,place_container-plate,</td></tr><tr><td>Client 2</td><td>8</td><td>place_empty-cup place_a2b_left,place_a2b_right,place_fan, place_mouse-pad,place_object_scale,</td></tr><tr><td>Client 3</td><td>6</td><td>place_object_stand,place-phone_stand,place_shoe stack_blocks_three,stack_blocks_two, stack_bowls_three,stack_bowls_two,blocks_ranking-rgb,</td></tr><tr><td>Client 4</td><td>5</td><td>blocks_ranking-size adjust_bottle,lift_pot,move-pillbottle-pad,</td></tr><tr><td>Client 5</td><td>3</td><td>move-playingcard_away,move_stapler-pad beat_block_hammer,press_stapler, stamp-seal</td></tr><tr><td>Client 6</td><td>6</td><td>click_alarmclock,click_bell,turn_switch, open_laptop, open_microwave,rotate_qrcode</td></tr><tr><td>Client 7</td><td>3</td><td>scan_object, shake_bottle, shake_bottle_horizontally</td></tr><tr><td>Server</td><td>7</td><td>place_burger_fries,place_dual_shoes,dump_bin_bigbin, handover_block, handover_mic, hanging_mug, move_can_pot</td></tr></table>

Observation and action interface. As shown in the Table 5, predicted joint-space offsets are converted to joint targets using the configuration at the start of each action queue. The evaluator truncates predictions using its execution-horizon argument, while prediction length does not determine the number of executed actions.

Sampling and update accounting. Local training cycles through a shuffled frame-level loader, without task-balanced sampling. The epoch setting repeats the local-step loop rather than guaranteeing a full dataset traversal of the dataset. For C clients, K rounds, L local steps, E repetitions, and M server steps, the update counts are KLE per client, CKLE across clients, and KM at the server. The schedule is given in Table 5.

Batch-size conventions. Client batch size is per local update, while server batch size is per distributed rank. With R server ranks and per-rank batch size B , the effective server batch is $R B _ { \mathrm { s r v } }$ . Client optimizer states are retained separately across rounds, while server optimizer state persists across expert replacement.

Randomized initialization. As shown in Table 5, unstable or expert-unsuccessful candidate scenes are excluded before policy evaluation. Accepted trials therefore need not correspond to consecutive candidate seeds. The evaluator defaults to the unseen instruction split.

## A.4 RLBENCH

Task-silo manifest. Table 8 identifies the reported evaluation tasks, not client IDs. The training implementation provides task-disjoint by task partitions with four clients and a separate server.

Observation and action interface. See Table 5. Unlike RoboTwin, each queued translation increment is added to the current end-effector position, while the predicted Euler angles specify an absolute world-frame orientation. The resulting pose is executed by a motion planner. Stored translation labels are already relative while preprocessing must not subtract the state again.

Table 7: RoboTwin 2.0 task success rates (%) on the complementary 25 tasks not shown in Table 1. Best and second-best reported results are boldfaced and italicized.
<table><tr><td colspan="12">Method bot. anxk RGB alarm bell qrab roller mic pill card laptop micro. adjust click click hand move move open open</td></tr><tr><td colspan="12">bread</td></tr><tr><td></td><td>99</td><td></td><td></td><td></td><td>CENTRALIZED</td><td></td><td>TRAINING</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>INTERNVLA MOTUS</td><td>100</td><td>91 94</td><td>85 83</td><td>91 90</td><td>98 91</td><td>100 100</td><td>95 81</td><td>87 91</td><td>100 97</td><td>98 92</td><td>91 80</td><td>87 81</td><td>99 93</td></tr><tr><td colspan="14"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>FEDERATED LEARNING(LoRA/MoE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FEDAVG</td><td>99</td><td>91</td><td>84</td><td>93</td><td>95</td><td>100</td><td>89</td><td>83</td><td>98</td><td>90</td><td>84</td><td>82</td><td>98</td></tr><tr><td>FEDMOE</td><td>100</td><td>91</td><td>76</td><td>91</td><td>94</td><td>100</td><td>82</td><td>80</td><td>95</td><td>88</td><td>65</td><td>80</td><td>97</td></tr><tr><td>FORGEVLA</td><td>100</td><td>90</td><td>85</td><td>92</td><td>94</td><td>100</td><td>82</td><td>91</td><td>98</td><td>93</td><td>82</td><td>82</td><td>93</td></tr><tr><td>FEDVLA</td><td>99</td><td>88</td><td>82</td><td>89</td><td>92</td><td>99</td><td>83</td><td>83</td><td>96</td><td>90</td><td>78</td><td>82</td><td>96</td></tr><tr><td>ROBOFL</td><td>100</td><td>95</td><td>83</td><td>96</td><td>97</td><td>100</td><td>92</td><td>92</td><td>100</td><td>90</td><td>85</td><td>90</td><td>93</td></tr><tr><td>Method</td><td>cans</td><td>box plate cont.</td><td>cup empty</td><td>fan place</td><td>stand</td><td>stand phone</td><td>shoe place</td><td>bot. shake</td><td>horiz. shake</td><td>3 blo. stack</td><td>2 blo. stack</td><td>stack</td><td>2 bow. Overall</td></tr><tr><td colspan="14">CENTRALIZED TRAINING</td></tr><tr><td>INTERNVLA</td><td>97</td><td>97</td><td>100</td><td>91</td><td>88</td><td>94</td><td>93</td><td>99</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MOTUS</td><td>97</td><td>99</td><td>100</td><td>85</td><td>87</td><td>93</td><td>91</td><td>100</td><td>97 100</td><td>87 83</td><td>100 99</td><td>98 98</td><td>81.60 80.96</td></tr><tr><td colspan="14">FEDERATED LEARNING(LoRA/MoE)</td></tr><tr><td>FEDAVG</td><td>94</td><td>100</td><td>100</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FEDMOE</td><td>85</td><td>99</td><td>100</td><td>86</td><td>84</td><td>91</td><td>93</td><td>100</td><td>97</td><td>84</td><td>99</td><td>100</td><td>78.92</td></tr><tr><td></td><td></td><td></td><td></td><td>84</td><td>89</td><td>91</td><td>86</td><td>99</td><td>99</td><td>80</td><td>99</td><td>100</td><td>75.12</td></tr><tr><td>FORGEVLA</td><td>93</td><td>100</td><td>100</td><td>88</td><td>85</td><td>94</td><td>92</td><td>100</td><td>100</td><td>81</td><td>100</td><td>99</td><td>80.70</td></tr><tr><td>FEDVLA</td><td>89</td><td>100</td><td>100</td><td>83</td><td>84</td><td>90</td><td>90</td><td>100</td><td>98</td><td>84</td><td>95</td><td>100</td><td>79.32</td></tr><tr><td>ROBOFL</td><td>94</td><td>100</td><td>100</td><td>90</td><td>91</td><td>95</td><td>95</td><td>100</td><td>100</td><td>82</td><td>100</td><td>100</td><td>83.12</td></tr></table>

Table 8: RLBench evaluation manifest. Rows identify evaluation tasks, not federated clients.
<table><tr><td>Partition</td><td>Count</td><td>Tasks</td></tr><tr><td>Client 0</td><td>1</td><td>take_umbrella_out_of_umbrella_stand</td></tr><tr><td>Client 1</td><td>2</td><td>close_fridge, toilet_seat_down</td></tr><tr><td>Client 2</td><td>2</td><td>close_laptop_lid, close_box</td></tr><tr><td>Client 3</td><td>1</td><td>sweep_to_dustpan</td></tr><tr><td>Server</td><td>2</td><td>put_rubbish_in_bin, phone_on_base</td></tr></table>

Sampling and update accounting. The implementation uses the same shuffled-loader and updatecounting conventions as RoboTwin. Table 5 records launcher defaults.

Batch-size conventions. The same per-client-update and per-server-rank conventions apply as in RoboTwin for the independent-client configuration. A grouped-client distributed run also requires the number of ranks per client to determine the effective local batch size.

Randomized initialization. Simulator resets use the variation and seed in Table 5, without RoboTwin’s expert-based candidate screening. Invalid actions and failures terminate the episode and count as failures. Expert actions and future observations are not supplied to the policy.

## B ADDITIONAL EXPERIMENTAL DETAILS FOR REAL-WORLD FRANKA TASKS

## B.1 IMPLEMENTATION DETAILS

Task-silo manifest. Table 9 identifies the six reported single-arm Franka tasks and their client and server assignments. The implementation provides task-disjoint partitions among four clients and a separate server, with each client owning one task family and the server retaining the remaining two. Evaluation is performed via real-robot inference on the physical platform described in Appendix B.2.

Table 9: Franka real-robot manifest. Rows identify evaluation tasks, not federated clients.
<table><tr><td>Partition</td><td>Count</td><td>Tasks</td></tr><tr><td>Client 0</td><td>1</td><td>stack_cups</td></tr><tr><td>Client 1</td><td>1</td><td>screw_bottle</td></tr><tr><td>Client 2</td><td>1</td><td>stamp-seal</td></tr><tr><td>Client 3</td><td>1</td><td>test_tube</td></tr><tr><td>Server</td><td>2</td><td>adjust_bottle, pour_water</td></tr></table>

Observation and action interface. See Table 5. Unlike RoboTwin, each action is an absolute joint target executed directly by the robot, while queued targets are not added to the state at the start of the action queue. The policy consumes the external stand and wrist views, with the third view slot masked. State and action are both eight-dimensional, and auxiliary tactile and wrench channels are not supplied to the policy.

Sampling and update accounting. The implementation uses the same shuffled-loader and updatecounting conventions as RoboTwin and RLBench. Table 5 records the schedule.

Batch-size conventions. The same per-client-update and per-server-rank conventions apply as in RoboTwin and RLBench for the independent-client configuration. The client batch size is 32 per local update, while the server batch size is 16 per distributed rank.

Evaluation protocol. Each method is evaluated in three rounds of 20 trials per task. Each real-robot trial is started and reset by the operator, which places the objects and fixtures in their nominal task configuration. The policy then executes in a closed loop on the physical robot, and a trial is scored as successful only if the task goal is reached; otherwise, the episode terminates and counts as a failure.

## B.2 FRANKA SINGLE-ARM PLATFORM

Robot and gripper. Figure 6 shows the hardware configuration for our Franka single-arm experiments. The platform consists of a Franka Research 3 manipulator equipped with a Robotiq 2F-85 gripper. The robot is mounted beside a tabletop workspace containing the objects and fixtures used in the manipulation tasks.

Camera arrangement. Two RealSense D435i cameras provide complementary views of the workspace. One camera is mounted externally on a stand and faces the tabletop, providing an overview of the scene and object arrangement. The second camera is mounted on the robot wrist, providing a local view of the manipulation area as the arm moves. This configuration combines an external scene view with a close-range view of gripper-object interactions.

Workspace and task objects. The workspace contains a perforated board and task-specific objects, including bottles, cups, a stamp and ink pad, glassware, and a test-tube rack. These objects support the six Franka tasks reported in the main text: adjust bottle, stamp seal, stack cups, screw bottle, pour water, and test tube.

## C ADDITIONAL EXPERIMENTAL RESULTS

## C.1 COMPLEMENTARY QUANTITATIVE ANALYSIS FOR ROBOTWIN 2.0

The complementary results explain the emphasis on challenging tasks in Table 1. Most reported scores on this subset exceed 90%, with many at or near 100%, leaving limited headroom to distinguish methods. The main table therefore focuses on a lower-success subset where FedAvg scores below 80%. This presentation makes ROBOFL’s improvements on difficult manipulation tasks more visible, rather than relying on comparisons among near-saturated results, while the appendix preserves complete task coverage.

![](images/d64e10c08ddc4f720789040867523f890184fd10b9339e6690a190aa7461c154.jpg)  
Figure 6: Franka single-arm experimental setup. The platform comprises a Franka Research 3, a Robotiq 2F-85 gripper, and two RealSense D435i cameras mounted on the wrist and an external stand, respectively. The tabletop contains the objects and fixtures used in the manipulation tasks.

Table 10: Ablation of MOSAIC components on RoboTwin 2.0. All rows report overall success rates (%) on all 50 tasks with 100 trials per task. The three MOSAIC rows use the same federated pipeline and differ only in the server mechanism; FedAvg and FedMoE are external references.
<table><tr><td>Configuration</td><td>Overall</td></tr><tr><td>FedAvg (averaged single adapter)</td><td>78.92</td></tr><tr><td>FedMoE (client-side routed MoE)</td><td>75.12</td></tr><tr><td>Vanilla MoSAIC (task-trained experts + routing)</td><td>81.94</td></tr><tr><td>+ FARD</td><td>82.42</td></tr><tr><td>+ FARD + PCEA (ROBOFL)</td><td>83.12</td></tr></table>

Despite the higher success rates and reduced room for improvement, ROBOFL remains highly competitive, achieving best or joint-best results on 16 of the 25 tasks and 100% observed success on nine. Strong results across ranking, object placement, and bottle shaking show that its performance extends beyond the difficult cases highlighted in the main table. Together, the two subsets demonstrate that ROBOFL combines improvements on challenging tasks with strong performance on the more saturated portion of the benchmark.

In addition, Table 10 reports the overall success rate of MOSAIC components on the full 50-task RoboTwin 2.0 benchmark. Installing task-trained adapters as expert branches with learned routing (vanilla MOSAIC) reaches 81.94%; adding routing distillation (FARD) raises this to 82.42%; and adding consensus-weighted conversion (PCEA) reaches 83.12%. The two external references put these numbers in context: federated averaging reaches 78.92%, while a client-side-routed LoRA-MoE (FedMoE) reaches 75.12%. Most of the gain, therefore, comes from retaining task-trained experts and learning their composition on the server (3.02% over FedAvg), with FARD and PCEA contributing smaller but consistent additional improvements (0.48% and 0.70%, respectively). FARD and PCEA mainly affect the distribution of high-success tasks as shown in Section 4.3.

## C.2 QUANTITATIVE ANALYSIS FOR RLBENCH

We further evaluate the methods in a lower-data, single-view RLBench setting with eight tasks and 50 trials per task. ROBOFL reaches 41.25%, the best among federated methods, improving over FL baselines. Unlike RoboTwin, where ROBOFL also surpasses centralized PEFT, it trails centralized

Table 11: RLBench task success rates (%) on the 8-task single-view suite. Overall is the equally weighted mean across the eight tasks. Best and second-best results are boldfaced and italicized.
<table><tr><td>Method</td><td>put rubbish</td><td>seat down</td><td>draw umbrella</td><td>close laptop</td><td>sweep dustpan</td><td>close fridge</td><td>close box</td><td>phone on base</td><td>Overall</td></tr><tr><td colspan="10">CENTRALIZED TRAINING</td></tr><tr><td>INTERNVLA</td><td>60</td><td>18</td><td>34</td><td>62</td><td>66</td><td>46</td><td>86</td><td>50</td><td>52.75</td></tr><tr><td>MOTUS</td><td>44</td><td>22</td><td>36</td><td>58</td><td>72</td><td>58</td><td>92</td><td>38</td><td>52.50</td></tr><tr><td colspan="10">FEDERATED LEARNING(LoRA/MoE)</td></tr><tr><td>FEDAVG</td><td>0</td><td>18</td><td>22</td><td>4</td><td>14</td><td>72</td><td>92</td><td>0</td><td>27.75</td></tr><tr><td>FEDMOE</td><td>0</td><td>12</td><td>16</td><td>14</td><td>22</td><td>64</td><td>70</td><td>8</td><td>25.75</td></tr><tr><td>FORGEVLA</td><td>12</td><td>14</td><td>32</td><td>26</td><td>20</td><td>70</td><td>84</td><td>16</td><td>34.25</td></tr><tr><td>FEDVLA</td><td>6</td><td>16</td><td>24</td><td>12</td><td>30</td><td>56</td><td>80</td><td>24</td><td>31.00</td></tr><tr><td>ROBOFL</td><td>26</td><td>24</td><td>32</td><td>64</td><td>24</td><td>50</td><td>60</td><td>50</td><td>41.25</td></tr></table>

InternVLA (52.75%) and Motus (52.50%) on RLBench. We hypothesize that the smaller single-view suite provides less task diversity and complementary evidence for expert composition, whereas centralized PEFT can directly pool all available examples. RoboTwin better represents the broad, task-heterogeneous distributed regime in which preserving specialized adapters is most valuable. These results therefore suggest that ROBOFL is particularly suited to large-scale distributed learning with complementary non-IID data.

## C.3 QUALITATIVE ANALYSIS

We visualize successful rollouts to examine whether the reported success rates correspond to sensible task behavior: approach and alignment, contact or grasp, and execution of the intended state change. Each task contributes four chronologically ordered keyframes from a single successful episode, and the frames are exported at their native resolution without cropping or enhancement. Because the evaluators record an observation before each executed action, the final frame may precede the release or contact that triggered the success flag, so visual inspection captures the executed behavior rather than confirming the terminal success event.

RoboTwin 2.0. Figure 7 and 8 show one successful rollout for each of the 50 RoboTwin 2.0 tasks. Across these rollouts, the policy exhibits a consistent three-phase structure: an alignment phase in which the end effector approaches the target, a contact or grasp phase, and a transport or placement phase. Articulated and contact-rich tasks such as open microwave, close laptop lid, and rotate qrcode show the gripper establishing contact and then applying a sustained motion rather than a ballistic strike. Multi-object tasks such as the stacking and ranking families, place bread basket, place cans plasticbox, and place dual shoes involve sequential subgoals, in which the second object is manipulated only after the first has been positioned. The tasks that remain difficult in the quantitative results, notably hanging mug and lift pot, are also visually harder: they require precise two-sided or handle-based contact under substantial occlusion, and their successful rollouts show a slower, more tentative approach. The remaining failures in the report are therefore concentrated in fine-grained contact and long-horizon placement rather than in gross reaching or scene understanding.

RLBench. Figure 9 shows successful rollouts for the eight-task single-view suite under the evaluation protocol of Table 5: front camera, variation 0, and an eight-action chunk queue. The rollouts reveal several recurring behaviors. Closing tasks, including close Box, closefridge, and close laptop lid, are executed as a short contact-and-push sequence: the gripper reaches the lid or door, establishes contact, and then drives it through its range in a single continuous motion. Grasping-and-relocation tasks such as put rubbish in bin and take umbrella out of umbrella stand show a descending approach and a closed grasp before the object is lifted clear. Tool-use and placement tasks, sweep to dustpan and phone on base, require the gripper to align with a narrow target region before the state change.

![](images/8d1fe220cc9562d315bab486971f97e1e4928a3e91970bc2335597ff8ebd1c6c.jpg)  
<sup>plasticbox plate</sup>Figure 7: Qualitative RoboTwin 2.0 rollouts (part 1 of 2). Each row shows two tasks. Every task is labeled with its name and contributes four chronologically ordered keyframes from one successful rollout, illustrating approach, contact or grasp, and execution of the intended state change. Frames are exported at their native resolution without cropping or enhancement.

## C.4 FAILURE ANALYSIS

<sup>place\_object\_</sup> <sub>place\_phone\_stand</sub>Despite the overall gains, we observe recurring failures in both simulation and hardware, as illustrated <sup>stand</sup>in Figures 10 and 11. 1) Incomplete Object Engagement: In lift pot and hanging mug the gripper topples or tilts the object instead of lifting or hanging it, and on Franka screw bottle never achieves threaded engagement. This suggests that a skill absent from every client family must be synthesized put\_bottles\_ put\_object\_by expert composition, which can fall back to a blend of adjacent experts. 2) Stalled or Abandoned dustbin cabinetMotion: stamp seal hovers without pressing, scan object drifts away after approaching, and on Franka pour water and test tube stop short of the manipulation, consistent with weak understanding-<sup>rotate\_qrcode</sup> <sup>scan\_object</sup>generation consensus under self-occlusion leaving the router under-constrained. 3) Misplaced End State: put object cabinet drags the object instead of stowing it, and stack cups lands beside the <sup>shake\_bottle</sup>base, indicating that rank-limited conversion attenuates the expert-specific component of a terminal motion. The RLBench failures in Figure 10 (bottom) differ, all ending in a motion-planning error stack\_blocks\_two(InvalidActionError) rather than a policy error. Overall, these cases highlight that skills absent from all clients, ambiguous routing, and lossy conversion remain the main bottlenecks, which we plan to instrument in future work.

## <sup>stamp\_seal</sup> D STATISTICAL REPORTING PROTOCOL

Recent VLA and WAM work typically reports success rates over a fixed number of execution trials from a single training run rather than averaging over multiple training seeds on the simulation: for example, Motus evaluates each RoboTwin 2.0 task over 100 execution trials from one fine-tuned model (Bi et al., 2026), and comparable single-run protocols are used for InternVLA-A1 (Cai et al.,

![](images/abf3306c79d69b7e5ee16297e96ab371ba8fd4c793b56b2feba6f18fe5630e01.jpg)  
Figure 8: Qualitative RoboTwin 2.0 rollouts (part 2 of 2). Each row shows two tasks. Every task is labeled with its name and contributes four chronologically ordered keyframes from one successful rollout, illustrating approach, contact or grasp, and execution of the intended state change. Frames are exported at their native resolution without cropping or enhancement.

![](images/5774d29ab63637d4f19ed74a0a834c1e9595f15379a21d423e7140f53d598064.jpg)  
Figure 9: Qualitative RLBench rollouts. Each row shows two tasks, each labeled with its name and the number of successful episodes out of 50, and includes up to four chronologically ordered keyframes from one successful rollout. All rollouts use the single-view interface with the front camera at variation 0 and the eight-action chunk-queue protocol. The displayed episode for each task was selected for visual clarity rather than being the longest or most representative rollout.

2026) and ForgeVLA (Zhou et al., 2026). We follow this convention: all simulation results are obtained from a single fixed training seed and a single evaluation seed, with 100 (RoboTwin 2.0) and 50 (RLBench) trials per task, and the real-robot protocol is a fixed, scripted sequence. This keeps our results directly comparable to the benchmarks and baselines that we report. Multi-seed training with confidence intervals would strengthen the evidence for these smaller effects and is a natural direction for future work.

![](images/452379718640ccef1d218e9b2d2181c159bb5b314de53004b66de1b0f567691a.jpg)

Figure 10: Failure cases across benchmarks. RoboTwin 2.0 failures (top) and RLBench failures (bottom) contribute four chronologically ordered keyframes from one failed rollout: initial scene, first meaningful attempt, visible deviation, and final failing state. Because observations are recorded before actions and the video ends at the failure, the last frame shows the episode-terminated state.  
![](images/07a98d265ee1c810a40f1d8ec263d3ea55cc86ac8e5ad291bb9975db27ea1b91.jpg)  
Figure 11: Franka failure cases. Failed rollouts for six Franka tasks, with each row contributing four chronologically ordered keyframes (initial scene, first attempt, visible deviation, final state).

## E LIMITATIONS

Our study has several limitations. First, all experiments use a single backbone (InternVLA-A1-3B) and one aligned MoT architecture. Second, we target task heterogeneity: clients differ in task distributions but share the embodiment, sensor suite, and action space, so cross-embodiment and cross-modal heterogeneity remain open questions. Finally, simulation results use a single evaluation seed, and the real-robot results use a fixed scripted protocol with 20 trials per task.