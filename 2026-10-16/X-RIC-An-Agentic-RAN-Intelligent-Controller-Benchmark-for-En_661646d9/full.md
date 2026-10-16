This paper has been accepted for publication in the proceedings of the IEEE International Conference on Mobile Ad-Hoc and Smart Systems (MASS), 2026. This is the authors’ accepted version of the article. The final version published by IEEE is: A. Alchaab, A. Younis and D. Pompili, “X-RIC: An Agentic RAN-Intelligent Controller Benchmark for Enhancing Reinforcement Learning,” in Proc. of the IEEE International Conference on Mobile Ad-Hoc and Smart Systems (MASS), Hong Kong SAR, China, October 21–23, 2026.

# X-RIC: An Agentic RAN-Intelligent Controller Benchmark for Enhancing Reinforcement Learning

Adhwaa Al-Chaab†, Ayman Younis, Dario Pompili†

†Department of Electrical and Computer Engineering

Rutgers University–New Brunswick, NJ, USA

E-mails: {adhwaa.alchaab, a.younis, pompili}@rutgers.edu

Abstract—The O-RAN ALLIANCE introduces architectural flexibility through disaggregation and AI-driven control. While the RAN Intelligent Controller (RIC) enables centralized optimization, current AI approaches lack reasoning capabilities and explainability. We propose eXplainable RIC (X-RIC), a novel framework integrating agentic AI into O-RAN architectures. X-RIC combines reactive Reinforcement Learning (RL) with strategic agentic reasoning through: (i) automated Behavioral Cloning (BC) using domain heuristics for sample-efficient training, (ii) a Telecom Language Model (TLM), a transformer-based reasoning module providing contextual abstraction and interpretability, and (iii) closed-loop co-adaptation within the Near-RT RIC control loop enabling continuous policy refinement. To ensure deployment readiness, we introduce BARX (Benchmarking Agentic Reasoning for RAN), a domain-specific evaluation framework assessing agentic control across latency, compliance, and determinism metrics. We validate X-RIC on SLA-aware network slicing in simulation and O-RAN-compliant testbed deployments. X-RIC achieves 89% operational cost reduction while maintaining SLA compliance, converging 75% faster with 50% fewer training steps and 46% lower variance.

Index Terms—O-RAN, Agentic AI, Reinforcement Learning, Behavior Cloning, Resource Allocation.

## I. INTRODUCTION

The O-RAN ALLIANCE enables flexible, multi-vendor RAN ecosystems through open interfaces and programmable control [1]. Central to this architecture, the RAN Intelligent Controller (RIC) provides near-real-time closed-loop optimization for diverse service types, including eMBB, URLLC, mMTC, and ISAC [2]. The O-RAN paradigm integrates AI directly into the control loop, requiring autonomous adaptation, real-time responsiveness, and intelligent decision-making under uncertainty [3]. However, realizing this AI-native vision faces several fundamental challenges [4], [5], including: (i) real-time responsiveness, requiring decisions within strict latency bounds (sub-1s for Near-RT RIC) while adapting to dynamic conditions; (ii) self-optimization under uncertainty, as static policies cannot scale across heterogeneous network conditions; and (iii) explainability and trust, since AI-driven decisions must exhibit transparent behavior to satisfy operational and regulatory mandates. Traditional rule-based and supervised learning methods struggle with dynamic networking environments, while Reinforcement Learning (RL), despite offering promise through sequential decision-making, suffers from fundamental limitations: (i) poor sample efficiency, requiring millions of interactions infeasible in live networks;

(ii) limited generalization under unseen traffic patterns; and (iii) lack of interpretability, problematic for safety-critical operations [6].

To address these limitations, this paper introduces X-RIC, a novel RIC framework integrating explainable agentic intelligence for AI-native RAN control. Unlike conventional RL approaches relying solely on reactive policies, X-RIC employs a hybrid architecture combining fast reactive learning with strategic reasoning. Built on agentic AI principles— augmenting learning systems with reasoning, temporal memory, and task-level understanding [7]—X-RIC enables contextual abstraction, reflective learning, and decision traceability.

The X-RIC architecture comprises three components: (i) automated demonstration harvesting that initializes RL policies through Behavior Cloning (BC) [8] using trajectories from a fine-tuned reasoning model, improving sample efficiency without manual supervision; (ii) a transformer-based reasoning module providing strategic oversight and natural language explanations; and (iii) closed-loop co-adaptation enabling joint optimization while maintaining transparency. To further address sample inefficiency, X-RIC employs a dual-filtered harvesting pipeline: episode-level policy-impact selection identifies valuable trajectories, while critic-guided pruning filters suboptimal actions.

Related Work. We review here the O-RAN architecture and how RL has been used within O-RAN, followed by online learning and LLM-based approaches.

O-RAN Architectures: Recent O-RAN developments demonstrate AI integration potential [9]. EdgeRIC [10] enables TTI-level decision-making through near-RT applications interfacing with RAN infrastructure. To enhance granular control, [11] embeds Central Units (CUs) and Distributed Units (DUs) within the real-time RIC as dedicated zApps for MAC scheduling and beam shaping. TinyRIC [12] operates near CU/DU components, hosting tApps for non-3GPP interface management and energy optimization. While advancing O-RAN programmability, these frameworks lack integrated AI reasoning and autonomous adaptation.

RL in O-RAN: RL has been widely used for intelligent resource management [13], dynamic network slicing [14], and closed-loop RIC control [15]. Younis et al. [13] propose a multitask DDPG framework for joint user association and power allocation, while [14] develops a multi-agent RL strategy for radio resource allocation across heterogeneous slices. Polese et al. [15] present a pipeline for training and validating deep RL-driven closed-loop control in O-RAN deployments. Despite strong performance, these RL methods remain constrained under highly dynamic conditions due to limited live-network exploration and scalability challenges in concurrent multi-slice deployments [16].

![](images/d663a460ec9800d25588ae9166473b43010d427164659b54f89b5b343d5bc9eb.jpg)  
Fig. 1. X-RIC Framework architecture and O-RAN resource allocation: (a) Integrated TLM-RL system with dual-filtering BC, and (b) Virtual resource management for network slicing.

Online Learning and LLM-Based Approaches: To narrow the simulation-to-reality gap in network control, recent work has explored online learning and Large Language Model (LLM) integration. Liu et al. propose EdgeSlice [17], a decentralized deep RL framework for jointly orchestrating radio and computing resources under slice-level SLAs. For network automation, ALLSTAR [18] generates and validates RAN schedulers from high-level operator intents using LLMs. Other LLM-based efforts include TSpec [19], TelecomGPT [20], slice-aware resource allocation [21], and intent-based deployment [22]. However, existing online-learning methods still rely on fixed neural architectures, while current LLM-based solutions often operate as static inference engines without continuous adaptation, closed-loop validation, or latency guarantees suitable for sub-1s RIC control.

BC Approaches: Behavioral Cloning (BC) has emerged as a promising alternative to address RL’s sample efficiency by learning from expert demonstrations [8], with applications in automated driving [23] and UAV detection [24]. However, existing BC approaches require extensive manual demonstrations that are hard to collect at scale and lack reasoning integration.

Our Approach. We address these limitations through three key aspects: (i) automated demonstration collection using dualfiltering BC to generate high-quality training data without manual intervention, (ii) integration of BC with transformerbased reasoning modules that provide natural-language rationales for explainable control, and (iii) joint optimization of BC policies with agentic reasoning components for continuous adaptation in dynamic O-RAN environments. To the best of our knowledge, our work is the first to integrate automated BC within the O-RAN architecture for agentic RAN control.

Our Contributions. The contributions of this paper are as follows: i) We design a novel eXplainable RAN intelligent controller, X-RIC, that integrates agentic AI with RL in a hybrid control architecture to enable sub-second decisionmaking based on natural-language intents. ii) We introduce a domain-specific evaluation framework, BARX, for assessing the suitability of agentic reasoning models in closed-loop telecom control. iii) We propose a dual-filtering BC mechanism with automated demonstration harvesting, using episodelevel policy-impact gating and action-level critic validation to improve sample efficiency while ensuring deployment-grade reliability. iv) As a case study, we formulate a network slicing resource allocation problem minimizing operational cost while meeting SLA requirements, achieving 89% cost reduction versus baselines under Near-RT RIC timing constraints.

Paper Organization. Sect. II details the main framework design for proposed X-RIC framework. Sect. III presents a case study demonstrating how X-RIC addresses network slicing in O-RAN. Sect. IV reports the performance evaluation results. Finally, Sect. V concludes the paper.

## II. X-RIC FRAMEWORK DESIGN

We present the X-RIC framework integrating agentic reasoning into Near-RT RIC operations, introduce BARX for agentic RAN benchmarking, and describe dual-filtering demonstration collection for sample-efficient BC training.

## A. TLM Module and Benchmarking

Our X-RIC framework, shown in Fig. 1(a), is integrated within the O-RAN near-RT RIC; see [25] for an architectural overview. To avoid the computational overhead of full LLM fine-tuning, X-RIC uses a Telecom Language Model (TLM) architecture that augments a frozen pre-trained LLM with: (i) domain-specific prompts encoding O-RAN operational knowledge, and (ii) a trainable RAN-state encoder that maps structured telemetry, including KPIs, slice states, and resource metrics, into 256-dimensional embeddings. The encoder is trained via contrastive alignment on state-text pairs to project RAN observations into the LLM representation space, enabling RAN-aware reasoning without full model fine-tuning.

Integrating agentic reasoning modules into O-RAN workflows introduces system-level challenges due to the mismatch between probabilistic language model outputs and deterministic telecom control requirements. Four critical challenges emerge: (i) Missed scheduling deadlines when TLM inference time exceeds the 10 ms to 1s latency budget for Near-RT RIC operations [25], (ii) Malformed control messages due to syntactically invalid JSON/YAML configurations, (iii) Nondeterministic behavior when identical network states produce different TLM outputs due to sampling randomness, and (iv) Unintended SLA degradation when semantically plausible but operationally suboptimal TLM decisions reduce network performance. To address these challenges, we introduce a novel domain-specific framework, Benchmarking Agentic Reasoning (BARX), based on O-RAN specifications. Unlike standard NLP benchmarks [26], which emphasize linguistic fluency and semantic understanding, BARX evaluates agentic reasoning modules in closed-loop control settings. It assesses

TLM candidates across four dimensions: latency, compliance, determinism, and policy impact.

For a given network state s, the TLM is queried N times under fixed sampling parameters, temperature $T \in [ 0 , 1 ]$ and nucleus probability $p \in [ 0 , 1 ]$ , yielding structured responses $\mathcal { E } ~ = ~ \{ e _ { n } \} _ { n = 1 } ^ { N } .$ . i) Latency is measured by the end-to-end response time $\ell _ { n }$ , from which we report the mean $\mu _ { \ell } .$ , standard deviation $\sigma _ { \ell } ,$ and empirical 95th percentile $\ell _ { 9 5 }$ . Responses with $\ell _ { n } > 6 0 0$ ms violate the sub-second budget of Near-RT RIC control loops. ii) Compliance is enforced by validating each reply against a strict Draft-7 JSON schema. The schema error rate E is the fraction of invalid outputs; if $E > 0$ BARX assigns a suitability score of 0. iii) Determinism is quantified by pairwise comparison of valid responses, with consistency score C defined as the proportion of identical reply pairs. Scores $C < 1$ indicate non-deterministic behavior due to sampling noise or soft hallucinations. iv) Policy impact is assessed by parsing valid replies into control directives executed under a TLM-guided Near-RT RIC policy.

The average episodic reward under TLM guidance, $\bar { R } _ { \mathrm { T L M } } .$ , is compared with the RL-only baseline reward $\bar { R } _ { \mathrm { R L } }$ . The difference $\Delta \bar { R }$ quantifies the functional impact of agentic reasoning in the control loop. BARX aggregates $( \mu _ { \ell } , \sigma _ { \ell } , \ell _ { 9 5 } , E , C , \Delta \bar { R } )$ into a normalized score [0, 100], with thresholds calibrated using foundation-model evaluation practices [27]. Scores above 80 indicate deployment-grade performance; scores from 60 to 79 indicate marginal reliability requiring monitoring; scores from 40 to 59 require refinement; and scores below 40 are unsafe for production. Algorithm 1 describes the BARX procedure.

## B. X-RIC Agent System Architecture

Our X-RIC framework in Fig. 1(a) employs two coordinated agents with critic-guided selection for autonomous slice management. Coordination Architecture. The architecture consists of: (1) the Monitoring Agent, implemented as the TLM (based on LLaMA3-8B with LangChain tools), which continuously observes network performance using a two-tier temporal strategy: 1 s polling frequency for real-time monitoring from the Kinetica time-series database and 10 s aggregation windows for statistical smoothing. This design implements the periodic intervention strategy validated as optimal in Section IV-B. The TLM monitors network state continuously and generates supervisory resource allocation recommendations a<sub>TLM</sub> every $1 0 \ \mathrm { s } . ( 2 )$ The Control Agent, implemented as a BCtrained actor-critic policy $\pi _ { \theta _ { \tau } }$ with critic network $Q _ { \theta _ { Q } }$ , serves as the primary low-latency controller (< 10 ms) for resource allocation. Action Selection and Demonstration Harvesting. At each 10 s decision epoch, both agents observe the same network state $s _ { k } \colon$ the Control Agent proposes $a _ { \mathrm { R L } } = \pi _ { \theta } ( s _ { k } )$ while the TLM proposes $a _ { \mathrm { T L M } }$ . The critic evaluates both candidates and executes the TLM recommendation when $\Delta R _ { k } \ = \ Q _ { \phi } ( s _ { k } , a _ { \mathrm { T L M } } ) - Q _ { \phi } ( s _ { k } , a _ { \mathrm { R L } } ) \ > \ 0 ;$ otherwise, the RL action proceeds autonomously as the primary real-time controller. A dual-filter agentic BC layer, detailed in Sect. II-C, generates high-quality demonstrations from positive TLM interactions, storing only critic-validated state-action pairs in the demonstration buffer $\mathcal { D } _ { \mathrm { f l } }$ for behavioral cloning updates. Policy updates flow back to the RL agent with minimal overhead (median TLM latency: 465.6 ms vs. sub-10 ms RL latency), forming a closed-loop learning system.

## C. Automated Demonstration Collection for BC

Demonstration-augmented RL has shown promise in improving sample efficiency, particularly in complex decisionmaking tasks [28]. However, it still faces key limitations: expert demonstrations are often collected through manual teleoperation, or scripted policies. Although helpful in learning complex behaviors, the acquisition of high-quality demonstrations is often costly and impractical to scale [29], [30]. To address these limitations, we propose automated demonstration harvesting using a TLM with episode-level policy-impact gating $\Delta \bar { R } _ { k }$ and critic-validated action pruning. Our approach integrates BC as imitation learning to enable agents to mimic high-quality TLM-guided behaviors, accelerating learning by bootstrapping policies with expert-like actions. Episodes with positive $\Delta \bar { R } _ { k }$ values indicate improved performance from the use of agentic reasoning in the control loop, qualifying them as candidates for demonstration collection. However, these demonstrations may still contain suboptimal actions due to partial observability or ambiguous instruction mappings.

Algorithm 1 BARX procedure   
Require: TLM model M, state s, parameters $\overline { { ( T , p ) } } .$ , queries N, JSON   
schema S   
Ensure: score $\in [ 0 , 1 0 0 ] ,$ metrics $( \mu _ { \ell } , \sigma _ { \ell } , \ell _ { 9 5 } , E , C , \Delta \bar { R } )$   
1: $ { \mathcal { E } } ,  { \mathcal { L } } , V \gets \emptyset$ ▷ Responses, latencies, valid responses   
2: for n ← 1 to N do   
3: $t _ { \mathrm { s t a r t } }  \mathrm { g e t C } _ { \mathsf { \Pi } }$ urrentTime(); $e _ { n } \gets M$ .generate(s, T, p); ℓ<sub>n</sub> ←   
getCurrentTime() − t<sub>start</sub>   
4: $\mathcal { E }  \mathcal { E } \cup \{ e _ { n } \} ; \mathcal { L }  \mathcal { L } \cup \{ \ell _ { n } \}$   
5: if validateDraft7Schema(e<sub>n</sub>, S) then $V  V \cup \{ e _ { n } \}$   
6: end if   
7: end for   
8: $( \mu _ { \ell } , \sigma _ { \ell } , \ell _ { 9 5 } ) _ { \mathrm { ~ \small ~  ~ } }$ ← computeStats(L)   
9: $E  1 - \frac { | V | } { N }$ ▷ Schema error rate   
10: C ← computePairwiseIdentical(V) totalPairs(V ) ▷ Consistency score   
11: $\bar { R } _ { \mathrm { T L M } }$ ← executePolic $\bar { \mathbf { \xi } } ( V ) ; \Delta \bar { R } \gets \bar { R } _ { \mathrm { T L M } } - \bar { R } _ { \mathrm { R L } }$ ▷ Policy impact   
12: if $E > 0$ then return $0 , ( \dot { \mu } _ { \ell } , \sigma _ { \ell } , \ell _ { 9 5 } , E , C , \Delta \bar { R } )$   
13: else   
14: score ← normalizeMetrics $( \mu _ { \ell } , \sigma _ { \ell } , \ell _ { 9 5 } , E , C , \Delta \bar { R } )$   
15: return score, (µ<sub>ℓ</sub>, σ<sub>ℓ</sub>, ℓ<sub>95</sub>, E, C, ∆R<sup>¯</sup>)   
16: end if

Dual-Filtering Mechanism. Blindly cloning all actions from episodes with positive $\Delta \bar { R } _ { k }$ values could reinforce suboptimal behaviors and lead to performance degradation. To address this, we propose a dual-filtering mechanism that automates demonstration collection and retains only the most informative state–action pairs for training. The first filter operates at the episode level using a policy-impact metric, while the second operates at the action level using a criticguided $Q \mathrm { - } f i l t e r .$ This design addresses two critical challenges in demonstration-augmented RL; (i) reducing the cost of demonstration collection by leveraging TLM-generated trajectories, and (ii) improving demonstration quality through critic-guided filtering of suboptimal actions.

![](images/ffef496ead99662a49dc783d54c92939dbcca9aed1d389157ce9f3c12444288a.jpg)

![](images/857bf30d4ce84ab0ebab017e348f4ead259bdf1abe1e8fc9d807ac6c5125e36c.jpg)

![](images/4107d2c30230820378343e5db34087131f751412660775eae795090be7ae8dfd.jpg)

![](images/b309047b3f9d38148877f515c179fb92826e5e1857fae6de8b56b4ca67b3b860.jpg)

Fig. 2. Wall-clock learning curves comparing RL vs. X-RIC with 100, 300, 500 demonstrations (3 seeds, 10/50 episodes).  
![](images/3a640694cf61b26e51562cf65bb4c9cb6a6ab8d118cf307c6766f7078e0758c2.jpg)

![](images/1eab65f47f138cff2a84ec03821f31b048f7318c4ee322da8b2ec92183eb6358.jpg)

![](images/21e49ae683882539ad6261ba6290394e701b99b53bdd59291e83d6a8202b53f7.jpg)  
Fig. 3. Deployment cost (top), episode reward (bottom left), and explained variance (bottom right) comparing X-RIC variants with RL and TLM-RL baselines.

a) Episode-level filtering: Per-step reward differences from the outset would require comparing all steps across both TLM and RL runs—potentially thousands of comparisons—many of which may derive from episodes offering no benefit. To measure the impact of the TLM-guided controller during early training, we calculate $\Delta \bar { R } _ { k } = \bar { R } _ { \mathrm { T L M } , k } - \bar { R } _ { \mathrm { R L } , k }$ for episode k, where $\begin{array} { r } { \bar { R } _ { \mathrm { { T L M } } , k } = \frac { 1 } { T _ { k } } \sum _ { t = 0 } ^ { T _ { k } - 1 } r _ { k , t } ^ { \mathrm { { T L M } } } } \end{array}$ , with $r _ { k , t } ^ { \mathrm { T L M } }$ representing the reward at step t in episode k and $T _ { k }$ denoting the number of steps in episode k; $\bar { R } _ { \mathrm { R L } , k }$ is computed analogously. The metric $\Delta \bar { R } _ { k }$ serves as an initial coarse filter to identify promising episodes. This episode-level comparison provides a fast and reliable method to assess whether the TLM enhances overall performance across full control sessions, effectively smoothing out noisy or unstable per-step rewards that may arise from random network events (e.g., traffic spikes or signal interference). By computing a single score per episode, we can discard unhelpful episodes, thereby avoiding the computational cost of analyzing every individual step. Episodes with $\Delta \bar { R } _ { k } > \delta$ (margin $\delta \geq 0 )$ are admitted into a raw pool $D _ { \mathrm { r a w } }$ , assuming they encapsulate meaningful policy improvements. This step removes trajectories where agentic reasoning underperforms while maintaining full automation.

b) Action-level filtering: Episode-level selection may include trajectories with mixed-quality decisions. We therefore retain a state-action pair $( s _ { k } , a _ { k } )$ only if the critic prefers the demonstrator’s action to the current policy’s proposal,

$$
Q ( s _ { k } , a _ { k } | \theta _ { Q } ) > Q ( s _ { k } , \pi ( s _ { k } | \theta _ { \pi } ) | \theta _ { Q } ) + \epsilon .\tag{1}
$$

with optional margin $\epsilon \geq 0$ . Here, $a _ { k }$ is the TLM-guided action, $\pi ( s _ { k } | \theta _ { \pi } )$ is the current policy, and $Q ( \cdot | \theta _ { Q } )$ is the critic with parameters $\theta _ { Q }$ . This Q-filter removes suboptimal actions and yields a distilled demonstration buffer $\mathcal { D } _ { \mathrm { f l } }$ suitable for imitation.

Q-filtered BC Objective. Given $\mathcal { D } _ { \mathrm { f l } }$ , we define a selective BC loss,

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { B C } } = \frac { 1 } { | \mathcal { D } _ { \mathrm { f l } } | } \sum _ { ( s _ { k } , a _ { k } ) \in \mathcal { D } _ { \mathrm { f l } } } \| \pi ( s _ { k } | \theta _ { \pi } ) - a _ { k } \| _ { 2 } ^ { 2 } } } \\ & { } & { \cdot \mathbb { I } [ Q ( s _ { k } , a _ { k } | \theta _ { Q } ) > Q ( s _ { k } , \pi ( s _ { k } | \theta _ { \pi } ) | \theta _ { Q } ) + \epsilon ] , } \end{array}\tag{2}
$$

where $| \mathcal { D } _ { \mathrm { f l } } |$ is the number of retained samples and I[·] is the indicator function. The actor update direction balances RL and selective imitation,

$$
\begin{array} { r } { g _ { \theta _ { \pi } } = \lambda _ { 1 } \nabla _ { \theta _ { \pi } } J ( \theta _ { \pi } ) - \lambda _ { 2 } \nabla _ { \theta _ { \pi } } \mathcal { L } _ { \mathrm { B C } } , \quad \lambda _ { 1 } , \lambda _ { 2 } > 0 . } \end{array}\tag{3}
$$

where $J ( \theta _ { \pi } )$ is the standard return-maximization objective (e.g., DDPG policy gradient). We anneal $\lambda _ { 2 }$ as the policy approaches demonstrator performance, preventing the actor from being constrained below expert level.

In Fig. 2, performance trends are tracked in real time to assess system-level efficiency, showing that demonstrationaugmented X-RIC improves sample efficiency: $\mathrm { X - R I C _ { 5 0 0 } }$ achieves a 3× faster reward gain and 46% lower variance than the RL baseline. In Fig. 3, training and deployment metrics are tracked to assess operational efficiency, showing that $\mathrm { X - R I C _ { 5 0 0 } }$ achieves ∼75% faster convergence to comparable reward, 0.90 explained variance with 50% fewer training steps (vs. RL-only at ∼0.87), and ∼6× lower operational deployment cost than the RL and TLM-RL baselines.

## III. SLA GUARANTEE IN O-RAN USE CASES

In this section, we apply our proposed X-RIC framework to the network slicing setup [5], [31], which involves dynamic configuration and poses significant challenges for traditional RL methods. Our objective is to minimize the total operating cost of maintaining multiple slices while ensuring that each slice meets its SLA-defined performance requirements.

## A. Problem Description

As shown in Fig. 1(b), we consider a single-cell O-RANcompliant mobile network with a gNB supporting three slices, $i \ \in \ \mathcal { T } \ = \ \{ \mathrm { e M B B } , \mathrm { U R L L C } , \mathrm { m M T C } \}$ . Each slice serves a subset of UEs, where $\vert \mathcal { U } ^ { e } \vert ~ = ~ N _ { e } , ~ \vert \mathcal { U } ^ { u } \vert ~ = ~ N _ { u }$ , and $| \mathcal { U } ^ { m } | ~ = ~ N _ { m }$ denote the number of users in each slice. The network operates over discrete time slots $k \in \mathcal { K } .$ An

SLA is a contract between the Mobile Network Operator (MNO) and slice tenants specifying QoS requirements (e.g., maximum latency, minimum throughput). The MNO deploys an xApp within the Near-RT RIC to dynamically allocate two virtual-resource primitives [32]: i) virtual Resource Blocks $( \nu R B s ) , \ b _ { i , k } \ \in \mathbb { Z } _ { \geq 0 }$ , representing discrete spectrum units assigned to slice i; and ii) virtual Compute Workloads $( \nu C W s )$ $c _ { i , k } \in [ 0 , 1 ]$ , representing the CPU/GPU time-sharing fraction. Let $N _ { \mathrm { v R B } }$ denote the total available vRBs and $C _ { \mathrm { w } } = 1$ the normalized compute capacity. The achievable data rate for user u in slice i can be written as,

$$
\mu _ { i , k , u } = \frac { b _ { i , k } } { N _ { i } } W _ { \mathrm { v R B } } \log _ { 2 } \left( 1 + \gamma _ { i , k , u } \right) ,\tag{4}
$$

where $W _ { \mathrm { v R B } }$ is the bandwidth per vRB, $N _ { i } = | U ^ { i } |$ , and $\gamma _ { i , k , u }$ is the SNR. The latency for URLLC users is $L _ { i , k , u } = L _ { i , k , u } ^ { \mathrm { t x } } +$ $\kappa / c _ { i , k }$ , where $\kappa > 0$ is a processing constant.

KPI Definition. The slice-specific KPIs are defined as,

$$
{ \mathrm { K P I } } _ { i , k } = \left\{ \begin{array} { l l } { \frac { 1 } { N _ { e } } \sum _ { u \in \mathcal { U } ^ { e } } \mu _ { i , k , u } } & { i = { \mathrm { e M B B } } , } \\ { \frac { 1 } { N _ { m } } \sum _ { u \in \mathcal { U } ^ { m } } \mathbb { I } \big ( \mu _ { i , k , u } > \mu _ { u } ^ { \mathrm { m i n } } \big ) } & { i = { \mathrm { m M T C } } , } \\ { \frac { 1 } { N _ { u } } \sum _ { u \in \mathcal { U } ^ { u } } \operatorname* { m a x } \left( 0 , 1 - \frac { L _ { i , k , u } } { L _ { \mathrm { m a x } } } \right) \mu _ { i , k , u } } & { i = { \mathrm { U R L L C } } , } \end{array} \right.\tag{5}
$$

where I(·) is the indicator function, $\mu _ { u } ^ { \mathrm { m i n } }$ is the minimum rate for mMTC, and $L _ { \mathrm { m a x } }$ is the maximum allowable URLLC latency. The max(0, ·) operator ensures non-negative KPI contributions.

Cost Model and Problem Formulation. The operating cost with unit prices $u _ { \mathrm { v R B } }$ and $u _ { \mathrm { w } }$ for vRBs and vCWs is formulated as,

$$
\mathrm { C o s t } ( { \bf b } _ { k } , { \bf c } _ { k } ) = u _ { \mathrm { v R B } } \sum _ { i \in \cal Z } b _ { i , k } + u _ { \mathrm { w } } \sum _ { i \in \cal Z } c _ { i , k } - \sum _ { i \in \cal Z } w _ { i } \mathrm { K P I } _ { i , k } ,\tag{6}
$$

where $w _ { i } ~ > ~ 0$ weights the KPI importance. The resource allocation problem is formulated as,

$$
\operatorname* { m i n } _ { \mathbf { b } _ { k } , \mathbf { c } _ { k } } \quad \mathbf { C o s t } ( \mathbf { b } _ { k } , \mathbf { c } _ { k } )\tag{7a}
$$

$$
\begin{array} { r } { \mathrm { s . t . } \quad \mathrm { K P I } _ { i , k } \geq Q _ { i } , \quad \forall i \in \mathcal { I } , } \end{array}\tag{7b}
$$

$$
\sum _ { i \in \mathcal { T } } b _ { i , k } \leq N _ { \mathrm { v R B } } , \quad \sum _ { i \in \mathcal { T } } c _ { i , k } \leq 1 ,\tag{7c}
$$

$$
b _ { i , k } \in \mathbb { Z } _ { \geq 0 } , \ c _ { i , k } \in [ 0 , 1 ] , \quad \forall i \in \mathbb { Z } ,\tag{7d}
$$

where $Q _ { i }$ is the SLA threshold for slice i. Problem (7) is a mixed-integer nonlinear program (MINLP) due to integer vRBs and the nonlinear rate function (4). We adopt an RL approach to solve it in near real time.

## B. MDP Model

We formulate (7) as a Markov Decision Process (MDP) with states S, actions ${ \mathcal { A } } ,$ and policy $\pi _ { \theta } : { \mathcal { S } }  A$ . The objective is to maximize $\begin{array} { r } { R = \sum _ { k \in \mathcal { K } } \gamma ^ { k } r _ { k } } \end{array}$ , where $\gamma \in [ 0 , 1 )$ is the discount factor.

State Space. The state at slot k is,

$$
s _ { k } = \Big ( \big \{ \mathrm { K P I } _ { i , k - 1 } \big \} _ { i \in \mathcal { D } } , \big \{ \lambda _ { i , k } \big \} _ { i \in \mathcal { T } } , \big \{ \rho _ { i , k } \big \} _ { i \in \mathcal { T } } \Big ) ,\tag{8}
$$

![](images/79486901dc4d533d056299231f7423740ca02382bf0650d118d5fbd4806be0a4.jpg)  
Fig. 4. Percentage cost reduction over RL for X-RIC and TLM-RL, compared with a non-fine-tuned LLM policy.

![](images/b41237c318698250b8e72ad177d954859a78efe6f662a1ec3b9c61668ad9dc0d.jpg)  
Fig. 5. TLM-guided RL agent with dual-filtering BC.

where $\lambda _ { i , k }$ and $\rho _ { i , k }$ denote the arrival rate and service demand characteristics for slice i, respectively.

Action Space. The agent outputs continuous values $a _ { k } =$ $\{ ( \bar { b } _ { i , k } , c _ { i , k } ) \} _ { i \in \mathcal { T } }$ , where $\tilde { b } _ { i , k } ~ \in ~ [ 0 , N _ { \mathrm { v R B } } ]$ and $c _ { i , k } ~ \in ~ [ 0 , 1 ] .$ The discrete vRB allocation is obtained by rounding: $b _ { i , k } =$ $\lfloor \tilde { b } _ { i , k } + 0 . 5 \rfloor$

Reward Function. The reward incentivizes cost minimization with SLA compliance,

$$
R ( s _ { k } , a _ { k } ) = - \mathrm { C o s t } ( { \mathbf b } _ { k } , { \mathbf c } _ { k } ) - \eta \sum _ { i \in \mathcal { I } } \operatorname* { m a x } ( 0 , Q _ { i } - \mathbf { K P I } _ { i , k } ) ,\tag{9}
$$

where $\eta > 0$ penalizes SLA violations.

Fig. 4 compares total cost reduction (vRB and vCW allocation) across four approaches, with RL as the reference point (0%). $\mathrm { X - R I C _ { 5 0 0 } }$ achieves 89.0% cost reduction, $\mathrm { X - R I C _ { 3 0 0 } }$ achieves 85.2%, and TLM-RL achieves 65.8%, showing that BC-trained policies outperform both baselines. The non-finetuned LLM increases cost by 90.3% due to hallucinations that violate resource constraints, highlighting the need for domainspecific BC training.

## C. TLM-based RL Approach Solution

In this subsection, we present a novel approach that integrates TLM feedback with RL for dynamic resource allocation in O-RAN network slicing, leveraging the dual-filtering BC mechanism detailed in Section II-C. Our solution follows a systematic approach, as shown in Fig. 5, organized into the following steps,

Step 1: Enhanced DDPG Architecture Design. We adopt the DDPG algorithm [33] for its suitability in mixed-integer resource allocation problems where vRBs are discrete and vCWs are continuous. Additionally, DDPG’s off-policy nature enables learning from data not necessarily generated by the current policy, such as exploration trajectories or expert demonstrations. The architecture comprises an actor network $\pi _ { \theta _ { \tau } }$ and critic network $Q _ { \theta _ { Q } }$ , with state observations $s _ { k }$ encompassing slice KPIs, arrival rates, and service-demand characteristics as inputs.

Step 2: Demonstration Collection and Filtering. We apply the dual-filtering mechanism described in Section II-C to collect high-quality demonstrations from TLM interactions. This process generates the filtered demonstration buffer $\mathcal { D } _ { \mathrm { f l } }$ containing only high-quality state-action pairs that pass both episode-level policy impact gating and action-level critic validation. We enhance the standard DDPG objective by incorporating TLM guidance and BC losses,

$$
\begin{array} { r } { \mathcal { F } ( \theta _ { \pi } , \theta _ { Q } ) = - \mathcal { L } _ { p } ( \theta _ { \pi } ) + \lambda _ { v } \mathcal { L } _ { v } ( \theta _ { Q } ) + \lambda _ { \mathrm { T L M } } \mathcal { L } _ { \mathrm { T L M } } ( \theta _ { \pi } ) + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ( \theta _ { \pi } ) } \end{array}\tag{10}
$$

where the BC loss ${ \mathcal { L } } _ { \mathrm { B C } } ( \theta _ { \pi } )$ utilizes the Q-filtered demonstrations from $\mathcal { D } _ { \mathrm { f l } }$ as defined in (2).

Step 3: Mixed Replay Training. During training, we sample from both the standard replay buffer D (exploration data) and filtered demonstration buffer $\mathcal { D } _ { \mathrm { f l } }$ (expert guidance). This mixed sampling strategy enables the agent to learn from both self-exploration and TLM-guided expert demonstrations.

Step 4: State Encoding and TLM Integration. The RAN encoder transforms network states $s _ { k }$ into textual representations $\Omega _ { k }$ for TLM-based reasoning over resource allocation. The TLM processes structured prompts containing: (i) the optimization objective, (ii) the current network observation, including per-slice PRB usage, buffered packets, and traffic criticality indicators, and (iii) an action-to-slice mapping over candidate decisions. It then evaluates the RL agent’s proposed action and returns a structured JSON response with a reasonability judgment, explanation, and corrected action when needed. Fig. 6 illustrates this process for an emergency URLLC traffic burst, where the TLM detects that allocating PRBs to eMBB and mMTC is suboptimal under high URLLC congestion and overrides it with a latency-aware allocation. The resulting decisions are filtered through the dual-filtering BC mechanism and integrated into training.

Step 5: Continuous Learning Loop. The system operates in a closed-loop manner where: (i) X-RIC generates expert demonstrations based on current network conditions, (ii) dualfiltering selects high-quality demonstrations, (iii) BC loss guides policy learning from filtered demonstrations, (iv) RL continues exploration and optimization, and (v) performance feedback updates the X-RIC guidance process. This integrated approach utilizes the sample efficiency of BC while maintaining the exploration capabilities of reinforcement learning.

## IV. PERFORMANCE EVALUATION

All experiments are performed on a workstation equipped with an AMD Ryzen 9 9950X CPU (16C/32T, 5.75 GHz, AVX-512) and an RTX 5080 GPU (16 GB VRAM). The RL Control Agent is implemented as a BC-trained DDPG policy with vectorized state normalization, evaluated over 300-second episodes with sub-second control intervals under dynamic traffic patterns and congestion scenarios.

Prompt Instruction: You are a network operator, and your goal is to devise a resource   
allocation policy that maximizes utility while ensuring that QoS targets for latency,   
throughput, and packet loss are met.   
Observation: Step = 55, Action = [0]   
PRB Usage: {URLLC: 10.5%, eMBB: 9.3%, mMTC: 3.5%}   
Buffered Packets: {URLLC: 16, eMBB: 7, mMTC: 3}   
Thoughts: Emergency traffic burst detected in URLLC.   
Action-to-Slice Mapping:   
Action 0: Allocate PRBs to both eMBB and mMTC.   
Action 1: Prioritize URLLC.   
Action 2: Distribute PRBs equally across all slices.   
Action 3: Prioritize eMBB.   
Assess whether Action 0 is appropriate given congestion, PRB usage, and service   
priority.   
Final Decision:   
{"decision": "unreasonable",   
"explanations": "Action 0 is suboptimal.   
URLLC shows high buffering and is   
latency-critical, while eMBB and mMTC   
have lower demand.",   
"final\_action": "1"}  
Fig. 6. LLM reasoning under emergency traffic.

TABLE I  
BARX RESULTS FOR LLM MODELS.
<table><tr><td>Model</td><td>Latency (ms) Compliance (%)</td><td></td><td>Determinism (%) Policy Impact</td></tr><tr><td>L3</td><td> $4 4 1 . 9 \pm 4 5 . 9$ </td><td>100.0</td><td>100</td></tr><tr><td>Q7B</td><td> $5 7 3 . 9 \pm 6 9 . 8$ </td><td>100.0</td><td>100</td></tr><tr><td>Q4B</td><td> $4 5 9 . 7 \pm 6 7 . 0$ </td><td>100.0</td><td>92</td></tr><tr><td>QT</td><td> $4 6 5 . 9 \pm 7 5 . 6$ </td><td>83.3</td><td>83</td></tr><tr><td>MT</td><td> $5 8 8 . 9 \pm 1 5 2 . 6$ </td><td>100.0</td><td>100</td></tr><tr><td>TL</td><td> $3 4 2 . 7 \pm 2 7 0 . 1$ </td><td>0.0</td><td>0 -2.75 5.88</td></tr><tr><td>DS</td><td> $8 7 8 . 2 \pm 1 3 5 . 3$ </td><td>100.0</td><td>100</td></tr></table>

## A. TLM Module Benchmarking

Prior to deploying the agentic TLM model, we benchmark seven fine-tuned open-source LLMs via BARX to assess their suitability for agentic reasoning in O-RAN under realistic conditions: LLaMA3-8B (L3), Qwen-7B (Q7B), Qwen-4B (Q4B), Qwen:latest (QT), Mistral:latest (MT), TinyL-LaMA:latest (TL), and DeepSeek-7B (DS). Each model is evaluated under identical traffic-control prompts with three sampling regimes: $( T , p ) \in \{ ( 0 . 0 1 , 0 . 8 ) , ( 0 . 1 , 0 . 9 ) , ( 0 . 0 , 1 . 0 ) \}$ For each model-regime combination, we conduct five independent trials with 20 queries each, yielding 100 JSON responses per model regime (2100 total responses). Table I summarizes the BARX results. Although TL exhibited the lowest latency (342 ms), it produced 100% parse errors from preambles and inconsistent key naming (e.g., explanaition), violating strict control-plane interfaces. Q4B occasionally omitted required fields (e.g., “final\_action”) without syntax errors, indicating latent schema risks and reducing determinism by 8%. QT, with $T ~ = ~ 0 . 1$ and $p = 0 . 9$ , produced consecutive JSONs, causing “Extra data” parse errors and resulting in a schema error rate of $E = 1 6 . 7 \%$ . The BARX score is calculated for each model by first enforcing two strict requirements: full schema compliance and a $\ell _ { 9 5 }$ below 600 ms. Models that fail either criterion receive a score of 0. For models that qualify, The final score averages normalized metrics: $\mu _ { \ell } ~ ( 2 0 \% )$ $\sigma _ { \ell }$ (15%), $\ell _ { 9 5 }$ (15%), C (20%), and $\Delta \bar { R } \left( 3 0 \% \right)$ , eliminating 71% of candidates. L3 is selected as the base reasoning module in

![](images/e9ece27a100cf7bf3c3ceefe782f8cbde6c8e114e455e33d4518ce411e6e8500.jpg)

![](images/cf67a4370516cfbbd09fb908882e83e2d3f5ce7f7897e23a3f84b0a82bae7778.jpg)  
(a)  
(b)

![](images/883f4a18ba7371f83a20ddfa86e2fbd175410d7c06d119ea7af633c6b20f0eb2.jpg)  
(c)

Fig. 7. a) Performance stability across intervention strategies, measured by coefficient of variation (lower is more stable), (b) Efficiency of correction, measured by performance gain per intervention, and (c) Latency distribution per strategy (5 episodes).  
![](images/ee240f6dbe39fa9ead8a2c4baf945cc25127df01c0bf045c52da9da3078c0331.jpg)  
(a)

![](images/e287368c4e1981a9c783d3088f9c95144ded13350bbbd8052d121c1fc1d72f22.jpg)  
(b)

![](images/ef2a8e2711aa2e8ee1c9d2b47f674622f7b2f2a1040805474b7a58ff92404a8c.jpg)  
(c)  
Fig. 8. Impact of BC on RL performance across X-RIC variants; (a) Mean reward comparison, (b) Training time breakdown, and (c) Coefficient of variation indicating training stability.  
our experiments due to its favorable latency-reliability tradeoff.

## B. Ablation of TLM Intervention Schedules

To address the question when should the agentic reasoning agent intervene and correct the RL policy?, we conducted an ablation study evaluating six distinct correction strategies. The experimental methodology compared baseline RL performance against six TLM schemes, each tuned for fairness and consistency. The strategies included: Constant Correction, where the TLM continuously overrides RL decisions; Periodic Correction, which applies guidance at fixed intervals (every 10 steps) with a sparse, consistent intervention rate (correction probability $\approx 0 . 1 ) ;$ Step-Based Correction, which intervenes only during the initial 1000 steps before transitioning to full autonomy; Linear Decay, which reduces the correction probability linearly from 1.0 to 0.0 over 3000 steps; Exponential Decay, applying a decay rate of 0.95 every 100 steps; and Factor-Based Correction, which maintains a constant intervention probability of $\lambda = 0 . 5 .$ . All strategies were evaluated over five episodes of 300 simulation seconds using a pre-trained DDPG agent with vectorized normalization in a realistic O-RAN control environment. TLM-based corrections were applied using deterministic action predictions from our O-RAN–fine-tuned L3 model to ensure reproducibility. Our results, presented in Fig. 7, show that the periodic correction strategy achieved the best overall trade-off. It delivered the highest performance gain per correction, 11.86, the lowest coefficient of variation, 15.2%, and a median latency of 465.6 ms. Based on these findings, we adopt periodic correction as the preferred TLM intervention method in X-RIC.

![](images/11473d36105e72ef5b393d71d8b274d83975f8807bf1d82449340a1654992c41.jpg)  
Fig. 9. O-RAN testbed deployment for X-RIC evaluation.

Simulation Setup. For the SLA network slicing use case described in Section III, we generate three slices with $N _ { e } = 1 0 $ $N _ { u } = 1 0 .$ , and $N _ { m } = 6$ users. The slice-specific SLA requirements follow Eq. (5): $Q _ { \mathrm { e M B B } } = Q _ { \mathrm { U R L L C } } = 1 2$ Mbps, the mMTC KPI is the fraction of users satisfying $\mu _ { i , k , u } > \mu _ { u } ^ { \mathrm { m i n } }$ and $L _ { \operatorname* { m a x } } = 1 0$ ms for URLLC. The total number of vRBs is $N _ { \mathrm { v R B } } = 1 2$ , with normalized compute capacity $C _ { w } = 1$ and unit costs $u _ { \mathrm { v R B } } = u _ { w } = 1$ in (6). The SLA violation penalty is set to $\eta \ : = \ : 1 0 \ : \mathrm { { i n } } \ : ( 9 )$ . The environment exposes standard RL interfaces (reset, step, and observation) with discrete and continuous control over user association. A TLMassisted VecEnv wrapper supports explainability and humanin-the-loop policy refinement. We run 300-second episodes with Sub-second control intervals.

Fig. 8(a) shows that increasing demonstration buffer size improves policy performance, with X-RIC 300 and X-RIC 500 achieving 80.2% and 81.8% reward improvements over RL baseline. These gains incur only moderate increases in training time (see Fig. 8(b)), indicating high returns on demonstration effort. Fig. 8(c) further reveals that higher-quality demonstrations consistently reduce the coefficient of variation, highlighting greater policy stability. RL often fails to achieve high returns and exhibits high variance [34], which limits deployment reliability in real-time control systems. The dual-filtering mechanism addresses this limitation by selecting high-quality demonstrations that reduce learning instability and improve convergence. These results confirm that demonstration quality directly impacts both training efficiency and deployment stability.

![](images/b6707196c28aca086c6fbc682d94d64de1a6f4f11ecd5b731507b2931e1c5f22.jpg)  
(a)

![](images/aa14b7ab802ab4c17bf57a74348c3124a3a2536e3f6950e3cae7e7b8ed84e33d.jpg)  
(b)

![](images/1411c5edcf7b0209712a248ccf4fed1a5a339a2eeae168729d1bccd173cd5711.jpg)  
(c)

![](images/8ce384465ec0a8031e27e59bbf99b5ce34f3d9c40cea8705f1a25ed905355636.jpg)  
(d)  
Fig. 10. Over-the-air testbed evaluation of X-RIC for SLA-aware network slicing; (a) CDF of eMBB throughput across methods with 12.0 Mbps SLA threshold, (b) CDF of URLLC latency with 10.0 ms constraint, (c) vCW utilization efficiency relative to total compute capacity, and (d) vRB utilization efficiency relative to available spectrum resources.

## C. Over-the-Air Testbed Implementation

To validate simulation results through OTA deployment, we implement a testbed with three main components (Fig. 9). i) RAN and control: OpenAirInterface gNB using USRP B210 SDR interfaces with FlexRIC near-RT RIC via E2 for closed-loop slice control. KPIs stream to a Kinetica real-time accelerated database; TLM is based on L3-8B with LangChain tools.ii) 5G core: OAI 5G core (AMF, SMF, UPF) configured for network slicing and per-slice policy enforcement. iii) 5G UEs: Sierra with programmable SIMs.

Over-the-Air Experimental Setup. We instantiate X-RIC with L3-8B as the TLM foundation and BC-trained DDPG as the RL Control Agent, implementing a two-slice subset: URLLC (Slice 1, UE1, 12.1.1.0/24) with strict QoS requirements (zero packet loss, low latency) and eMBB (Slice 2, UE2, 12.1.1.128/24) with high-throughput requirements. Following the objective in Section III, the operator intent is: “Minimize total operating cost while maintaining SLA-defined QoS requirements: the URLLC slice must satisfy strict latency and reliability constraints; the eMBB slice must meet throughput targets, with resource allocation prioritizing URLLC during congestion.” X-RIC autonomously translates this intent into concrete controls: PRB allocations, compute workload distribution, and per-slice throughput guarantees.

Initially, the network operates under balanced resource allocation with equal PRB distribution (50:50 ratio). To create realistic congestion scenarios, we use iperf3 in UDP mode to enable controlled congestion testing. Unlike TCP’s automatic throttling, UDP transmits at configured bitrates regardless of capacity, allowing traffic demands to exceed available bandwidth and exposing packet loss for QoS evaluation. We deploy two iperf3 servers running on ports 5201 (Slice 1) and 5202 (Slice 2) to generate traffic demands exceeding the system capacity (∼296 Mbps in our setup). The servers generate per-UE metrics (throughput, jitter, packet loss) at 1- second intervals, which are ingested into a Kinetica accelerated time-series database. When aggregate traffic demand exceeds available radio capacity, X-RIC detects congestion through KPI monitoring and evaluates SLA compliance across both slices. The critic-guided mechanism dynamically reallocates PRBs to restore SLA requirements with minimal resource consumption.

Performance Baselines. We evaluate the SLA compliance and resource efficiency of X-RIC against three representative baselines using Cumulative Distribution Function (CDF) analysis across four key metrics: eMBB throughput, URLLC latency, vCW utilization, and vRB utilization. We compare X-RIC against the following approaches: (i) Equal Allocation, which distributes vRB and vCW resources uniformly across all slices without adaptive control, representing static provisioning strategies; (ii) TLM-RL, which integrates TLM guidance with the RL agent but without the behavioral cloning mechanism, isolating the contribution of automated demonstration harvesting; and (iii) $X { \mathrm { - } } R I C _ { n } $ , our framework initialized with n dualfiltered demonstrations to assess the impact of demonstration budget on performance.

SLA Compliance Analysis. Fig. 10(a)–(b) show eMBB throughput and URLLC latency vs. SLA requirements of 12.0 Mbps and 10.0 ms, respectively. Equal Allocation operates close to limits: eMBB at 12–13 Mbps and URLLC at 9–10 ms, risking violations during congestion. $\mathrm { X - R I C _ { 5 0 0 } }$ provides larger safety margins: eMBB at 17 Mbps (42% above SLA) and URLLC at 5 ms (50% below) with minimal variance. TLM-RL and $\mathrm { X - R I C _ { 3 0 0 } }$ show intermediate margins (eMBB: 15–16.5 Mbps; URLLC: 6–7 ms), confirming that larger buffers improve performance and consistency.

Resource Efficiency Analysis. Fig. 10(c) and (d) compare vCW and vRB utilization. Equal Allocation operates near maximum capacity $( C _ { w } = 1 . 0$ for vCW, 11.8–12.0 blocks for vRB), providing no efficiency margin and incurring maximum costs. BC-trained policies, however, reduce resource costs: $\mathrm { X - R I C _ { 5 0 0 } }$ reduces vCW usage to 0.65 (35% reduction) and vRB usage to 8.5–9.5 blocks (25% reduction) while satisfying

SLA constraints. TLM-RL and $\mathrm { X - R I C _ { 3 0 0 } }$ show intermediate reductions (vCW: 0.75 and 0.67; vRB: 9.5–10.5 and 9.0– 9.8 blocks, respectively), indicating that dual filtering extracts high-quality examples that guide policies toward efficient allocations as the demonstration buffer grows.

## V. CONCLUSION

This paper introduced X-RIC, a novel framework integrating agentic AI into O-RAN architectures to address current RL limitations. The proposed framework features several progressive TLM implementation schemes and a hybrid control architecture for sub-second O-RAN decision-making. We introduced BARX, a domain-specific evaluation framework assessing agentic RAN control across latency, compliance, determinism, and policy impact metrics, and developed automated demonstration collection using dual-filtering behavioral cloning to improve sample efficiency while maintaining production-grade reliability. As a case study, we applied our proposed X-RIC framework to a three network slice system, which involves dynamic configuration and poses significant challenges for traditional RL methods to solve within the nearreal-time RIC timescale. Validation on SLA-aware network slicing demonstrates that X-RIC significantly outperforms state-of-the-art baselines.

## REFERENCES

[1] “O-RAN Empowering Vertical Industry: Scenarios, Solutions and Best Practice.” White Paper, December 2023. https://www.o-ran.org.

[2] S. Lu, F. Liu, Y. Li, K. Zhang, H. Huang, J. Zou, X. Li, Y. Dong, F. Dong, J. Zhu, et al., “Integrated sensing and communications: Recent advances and ten open challenges,” IEEE Internet Things J., vol. 11, no. 11, pp. 19094–19120, 2024.

[3] Y. He, G. Yu, X. Wang, Q. Wang, Z. Niu, W. Ni, and R. P. Liu, “Accountability and reliability in 6G O-RAN: Who is responsible when it fails?,” IEEE Wireless Commun., vol. 32, no. 2, pp. 52–59, 2025.

[4] T. Chen, C.-L. I, and T. Melodia, “O-RAN’s role in shaping 6G: Industry perspectives on open and smart RAN,” IEEE Wireless Commun., vol. 32, no. 1, pp. 10–12, 2025.

[5] A. Alchaab, A. Younis, and D. Pompili, “Slice-on-the-Fly: AI-based Network Slicing in O-RAN for Dynamic Traffic Demands,” in Proc. IEEE WoWMoM, pp. 51–60, 2025.

[6] A. Y. Majid, S. Saaybi, V. Francois-Lavet, R. V. Prasad, and C. Verhoeven, “Deep reinforcement learning versus evolution strategies: A comparative survey,” IEEE Trans. Neural Netw. Learning Syst., vol. 35, no. 9, pp. 11939–11957, 2024.

[7] S. Murugesan, “The rise of agentic AI: Implications, concerns, and the path forward,” EEE Intell. Syst., vol. 40, no. 2, pp. 8–14, 2025.

[8] F. Torabi, G. Warnell, and P. Stone, “Behavioral cloning from observation,” arXiv 1805.01954, 2018.

[9] S. Chowdhury, R. Gadiyar, and E. Obiodu, “AI-RAN: Artificial Intelligence – Radio Access Networks.” White Paper, 2025.

[10] W.-H. Ko, U. Ghosh, U. Dinesha, R. Wu, S. Shakkottai, and D. Bharadia, “EdgeRIC: Empowering real-time intelligent optimization and control in NextG cellular networks,” in Proc. NSDI, pp. 1315–1330, 2024.

[11] P. S. Upadhyaya, N. Tripathi, J. Gaeddert, and J. H. Reed, “Open AI Cellular (OAIC): An open source 5G O-RAN testbed for design and testing of AI-Based RAN management algorithms,” IEEE Netw., vol. 37, no. 5, pp. 7–15, 2023.

[12] C. Liu, G. Aravinthan, A. Kak, and N. Choi, “Tinyric: Supercharging O-RAN base stations with real-time control,” in Proc. ACM MobiCom, pp. 1–3, 2023.

[13] A. Younis, C. Sun, and D. Pompili, “ReLAx: Deep reinforcement learning based resource allocation for Next-G RANs,” in Proc. IEEE MASS, pp. 437–445, 2024.

[14] M. Zangooei, M. Golkarifard, M. Rouili, N. Saha, and R. Boutaba, “Flexible RAN slicing in open RAN with constrained multi-agent reinforcement learning,” IEEE J. Sel. Areas Commun., vol. 42, no. 2, pp. 280–294, 2024.

[15] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “ColO-RAN: Developing machine learning-based xApps for open RAN closedloop control on programmable experimental platforms,” IEEE Trans. Mobile Comput., vol. 22, no. 10, pp. 5787–5800, 2023.

[16] M. Zangooei, N. Saha, M. Golkarifard, and R. Boutaba, “Reinforcement learning for radio resource management in RAN slicing: A survey,” IEEE Commun. Mag., vol. 61, no. 2, pp. 118–124, 2023.

[17] Q. Liu, T. Han, and E. Moges, “EdgeSlice: Slicing wireless edge computing network with decentralized deep reinforcement learning,” in Proc. IEEE ICDCS, pp. 234–244, 2020.

[18] M. Elkael, M. Polese, R. Prasad, S. Maxenti, and T. Melodia, “ALL-STaR: Automated LLM-driven scheduler generation and testing for intent-based RAN,” arXiv:2505.18389, 2025.

[19] R. Nikbakht, M. Benzaghta, and G. Geraci, “Tspec-LLM: An opensource dataset for LLM understanding of 3GPP specifications,” arXiv:2406.01768, 2024.

[20] H. Zou, Q. Zhao, Y. Tian, L. Bariah, F. Bader, T. Lestable, and M. Debbah, “TelecomGPT: A framework to build telecom-specfic large language models,” arXiv:2407.09424, 2024.

[21] X. Wu, J. Farooq, Y. Wang, and J. Chen, “LLM-xApp: A large language model empowered radio resource management xApp for 5G O-RAN,” in Proc. NDSSWorkshop (FutureG), 2025.

[22] Y. Tang, U. C. Srinivasan, B. J. Scott, O. Umealor, D. Kevogo, and W. Guo, “End-to-end edge AI service provisioning framework in 6G oran,” arXiv:2503.11933, 2025.

[23] L. Wang, C. Fernandez, and C. Stiller, “High-level decision making for automated highway driving via behavior cloning,” IEEE Trans. Intell. Veh., vol. 8, no. 1, pp. 923–935, 2023.

[24] Z. Zou, X. Hu, and P. Zhong, “Active object detection for UAV remote sensing via behavior cloning and enhanced Q-Network with shallow features,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025.

[25] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “Understanding O-RAN: architecture, interfaces, algorithms, security, and research challenges,” IEEE Commun. Surveys Tuts., vol. 25, no. 2, pp. 1376–1411, 2023.

[26] A. Srivastava, A. Rastogi, A. Rao, A. A. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al., “Beyond the imitation game: Quantifying and extrapolating the capabilities of language models,” Trans. Mach. Learning Research, 2023.

[27] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al., “Holistic evaluation of language models,” arXiv:2211.09110, 2022.

[28] G. Ma, Z. Wang, Z. Yuan, X. Wang, B. Yuan, and D. Tao, “A comprehensive survey of data augmentation in visual reinforcement learning,” arXiv:2210.04561, 2022.

[29] B. Baker, I. Akkaya, P. Zhokov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune, “Video pretraining (vpt): Learning to act by watching unlabeled online videos,” NIPS, vol. 35, pp. 24639–24654, 2022.

[30] J. Li, T. Lu, X. Cao, Y. Cai, and S. Wang, “Meta-imitation learning by watching video demonstrations,” in Proc. ICLR, 2021.

[31] A. Alchaab, A. Younis, and D. Pompili, “Demo: Secure edge server for network slicing and resource allocation in open RAN,” in Proc. IEEE WoWMoM, pp. 169–171, 2025.

[32] R. Schmidt, M. Irazabal, and N. Nikaein, “FlexRIC: An SDK for nextgeneration SD-RANs,” in Proc. CoNEXT, pp. 411–425, 2021.

[33] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” arXiv 1509.02971, 2015.

[34] S. Fujimoto and S. S. Gu, “A minimalist approach to offline reinforcement learning,” NeurIPS, vol. 34, pp. 20132–20145, 2021.