# Where Position Matters: Profiling-Guided Layer-Wise RoPE Allocation in Transformers

Zhiwei Liu<sup>∗</sup>, Boao Kong<sup>∗,†</sup>, Bing Liu

October 30, 2026

## Abstract

Rotary Position Embedding (RoPE) typically uses uniform coverage across Transformer layers despite heterogeneous positional requirements. Phase-of interventions reveal non-uniform, model-specific rotaryphase dependence, while coverage interventions relate it to coverage-upgrade benefits. We propose profiling-guided layer-wise RoPE allocation, assigning high/low coverage under a fixed budget using early coverage-benefit profiles. The framework supports fixed-width query/key representations and physical rotary-dimension allocation in Multi-head Latent Attention (MLA). Normalized-depth mapping transfers profiles within model families, enabling small-model measurements to guide larger-model allocations before pretraining. A brief analysis relates selection and transfer to score calibration and inter-layer interactions. From-scratch experiments demonstrate validation-loss improvements over available budget matched baselines, while analytical resource accounting identifies favorable quality–storage–compute trade-ofs for physical MLA. These results connect positional demand to transferable, eficient architecture design.

## 1 Introduction

Transformers [26, 8] model interactions within a sequence through attention, with positional encodings [20, 19] providing information about token order and distance. Rotary Position Embedding (RoPE) [24] applies position-dependent rotations to query and key dimension pairs, allowing attention scores to encode relative positional relationships. This mechanism introduces two architectural choices: how many dimensions receive rotary transformations at each layer, and how these dimensions are distributed across network depth. Connecting these choices to the positional requirements of individual layers creates a design space for improving language modeling under resource constraints.

Recent studies demonstrate the flexibility of rotary configurations. A systematic investigation of partial RoPE [12] examines how rotary coverage afects training dynamics and convergence, showing that reduced coverage can achieve performance comparable to full rotation. RNoPE-SWA [30] interleaves RoPE-based sliding-window attention with global attention using no positional embeddings, exploiting complementary layer functions to improve quality and eficiency. RFIS and RPD [22] further characterize attention-head specialization through frequency interventions and changes in attention distributions, providing architectural guidance for hybrid attention. These advances highlight the flexibility of rotary design and motivate adapting coverage to the positional demands of individual layers. Turning this idea into a practical design procedure requires reliable demand measurements, explicit budgeted allocation rules, and cost-efective access to profiling information across model scales, bringing the following challenges into focus.

Limitations of existing approaches. The practical value of a positional profile depends on its connection to task performance, its ability to guide layer-wise allocation, and the cost of obtaining it at the target scale.

L1. Limited calibration between positional dependence and task-level utility. Frequency-level analyses [3] and head-level functional profiling [22] reveal how rotary modulation shapes attention behavior. Translating these observations into layer-wise allocation requires a quantitative connection to the languagemodeling objective: how strongly does each layer depend on rotary phase, and how well does this dependence predict the benefit of increasing its coverage? Establishing this connection is essential for using positional profiles as allocation signals.

L2. Coarse-grained allocation of rotary resources across layers. Uniform partial RoPE [12] assigns a common coverage ratio across depth, while RNoPE-SWA [30] uses a predefined interleaving structure. These allocation rules ofer limited flexibility for matching coverage to model-specific layer requirements. Varying coverage-upgrade benefits call for explicit allocation and matched-budget layer-placement comparisons.

L3. Repeated profiling costs across model scales. Checkpoint-based positional analyses [3, 22] require trained model states and additional evaluations. Applying this workflow separately at each target scale incurs repeated costs for checkpoint acquisition, interventions, and configuration selection. Reusing structural information from smaller models therefore becomes an important step toward determining larger-model allocations before training.

These challenges motivate a unified framework for rotary allocation with eficient reuse of profiling information. We therefore ask:

How can positional demand guide model-specific RoPE allocation under a fixed budget, and how can small-model profiles support larger-scale architecture design?

Main contributions. We introduce profiling-guided layer-wise RoPE allocation, connecting positionaldemand measurement, budgeted architecture design, and cross-scale profile transfer. We first quantify layer-wise dependence through phase-of interventions and calibrate the resulting scores against coverageupgrade benefits. We then assign high and low rotary coverage using early coverage-benefit scores. Finally, a normalized-depth mapping transfers small-model profiles into allocations specified before target-model training. Our contributions are threefold.

C1. Task-level positional-demand measurement and calibration (addressing L1). We use phase-of profiling, which replaces a layer’s rotary transformation with the identity while preserving its pre-rotation query/key representations and attention mask. Paired changes in validation loss quantify the layer’s dependence on rotary phase. Across four pretrained models and 136 layers, we identify strongly non-uniform and model-specific demand profiles. Controlled coverage interventions on selected high-, medium-, and low-demand layers further reveal strong ranking agreement between phase-of scores and coverage-upgrade benefits, connecting positional dependence to actionable architectural changes.

C2. Model-specific RoPE allocation under a fixed budget (addressing L2). We assign higher coverage to layers with larger early coverage-benefit scores and lower coverage to the remaining layers, subject to a prescribed total rotary-pair budget. Within each allocation experiment, configurations share a master frequency bank and use nested frequency-pair subsets. We provide a fixed-carrier implementation that preserves complete query/key widths, together with a physical rotary-dimension implementation for Multihead Latent Attention (MLA) [14, 15]. The latter connects layer-wise allocation to projection parameters, attention computation, and positional-key storage. A brief motivating analysis explains how upgrade benefits, score calibration, and inter-layer interactions inform the selection rule.

C3. Within-family cross-scale allocation transfer (addressing L3). We map small-model allocation profiles to larger models along normalized depth and combine the transferred scores with the target budget to determine layer-wise coverage before training. This procedure reuses source-model measurements for architecture selection at larger scales. From-scratch experiments with transferred allocations examine their language-modeling behavior under diferent attention parameterizations, connecting source-side structural measurements to target-scale training outcomes.

## 1.1 Related Work

Positional encoding and rotary coverage. YaRN [18] combines frequency-dependent interpolation with attention scaling to extend context windows, while LongRoPE [10] uses non-uniform positional interpolation and progressive extension to support longer inputs. GPT-NeoX-20B [5] and Pythia [4] apply rotary transformations to a subset of query/key dimensions, treating coverage as an architectural hyperparameter. The systematic study of partial RoPE [12] further examines the efects of this fraction on convergence and training stability. Our work develops coverage into a layer-wise decision variable and studies its allocation under a prescribed aggregate budget.

Positional dependence and functional specialization. Barbero et al. [3] analyze how rotary frequencies support positional attention patterns and discuss the potential role of low-frequency channels in carrying semantic information. RNoPE-SWA [30] combines local positional modeling and global retrieval through a hybrid layer configuration. RFIS and RPD [22] use frequency interventions to characterize retrieval and positional heads, motivating layer-specific allocations in hybrid attention architectures. Our approach uses language-modeling loss to connect layer-wise phase dependence with coverage-upgrade benefits and budgeted rotary allocation.

Structure-aware eficient architectures. Multi-query attention (MQA) [21, 6] reduces decoding memory trafic by sharing key/value heads, while grouped-query attention (GQA) [1, 25, 11] introduces grouped sharing to balance model quality and inference eficiency. MLA [14, 15] compresses the key/value cache through a low-rank latent representation, with a decoupled rotary branch providing a direct interface for positional-dimension allocation. Low-rank modeling approaches [31, 13, 28, 27] exploit activation distributions or cross-layer correlations to construct compact parameterizations and reduce parameter and computational costs. For structured pruning, SliceGPT [2] combines computationally invariant rotations with dimension deletion to reduce network width, while ShortGPT [16] identifies redundant layers through changes between layer inputs and outputs. Our work uses layer-wise positional demand to organize rotary resources and transfers profiles within model families to determine target architectures before training.

## 2 Preliminaries and Problem Formulation

We introduce attention, rotary position embeddings, and MLA, followed by the layer-wise rotary allocation problem.

Attention and RoPE. For an L-layer Transformer, let $X _ { \ell } \in \mathbb { R } ^ { T \times d }$ contain T tokens of hidden width d. Suppressing head indices, attention [26] uses $Q _ { \ell } = X _ { \ell } W _ { \ell } ^ { Q } , K _ { \ell } = X _ { \ell } W _ { \ell } ^ { K }$ , and $V _ { \ell } = X _ { \ell } W _ { \ell } ^ { V }$ . RoPE [24] sets $\widetilde { q } _ { \ell , t } = R _ { \ell , t } q _ { \ell , t }$ and $\widetilde { k } _ { \ell , t } = R _ { \ell , t } k _ { \ell , t }$ , rotating selected pairs j by $t \theta _ { j }$ and applying identity elsewhere. Stacking transformed vectors gives

$$
\mathrm { A t t n } _ { \ell } \big ( X _ { \ell } \big ) = \mathrm { S o f t m a x } \Bigg ( \frac { \widetilde { Q } _ { \ell } \widetilde { K } _ { \ell } ^ { \top } } { \sqrt { d _ { h } } } + { \cal A } \Bigg ) V _ { \ell } ,\tag{1}
$$

where $d _ { h }$ is the per-head query/key width and A is the causal mask. All dimensions participate in matching. Multi-head latent attention. MLA [14, 15] uses a shared latent $C _ { \ell } ^ { K V } \ = \ X _ { \ell } W _ { \ell } ^ { D K V } \ \in \ \mathbb { R } ^ { T \times d _ { c } }$ , with $K _ { \ell } ^ { C } = C _ { \ell } ^ { K V } W _ { \ell } ^ { U K }$ and $V _ { \ell } = C _ { \ell } ^ { K V } W _ { \ell } ^ { U V }$ . Content and rotary branches form

$$
\widetilde { \boldsymbol { Q } } _ { \ell } = \big [ \boldsymbol { Q } _ { \ell } ^ { C } , \mathrm { R o P E } ( \boldsymbol { Q } _ { \ell } ^ { R } ) \big ] , \quad \widetilde { \boldsymbol { K } } _ { \ell } = \big [ \boldsymbol { K } _ { \ell } ^ { C } , \mathrm { R o P E } ( \boldsymbol { K } _ { \ell } ^ { R } ) \big ] .\tag{2}
$$

Caching the KV latent and cross-head shared positional key requires $d _ { c } + d _ { \mathrm { r o p e } } ^ { ( \ell ) }$ logical elements per layer and token, where $d _ { \mathrm { r o p e } } ^ { ( \ell ) }$ is the per-head rotary width.

Rotary coverage and layer-wise budget. Let $M _ { \ell }$ and $m _ { \ell }$ denote candidate and active rotary pairs per query/key head, respectively, with coverage shared across heads:

$$
\rho _ { \ell } = \frac { m _ { \ell } } { M _ { \ell } } , \quad m _ { \ell } \in \{ 0 , \dots , M _ { \ell } \} , \quad \sum _ { \ell = 1 } ^ { L } m _ { \ell } = B .\tag{3}
$$

Uniform coverage $\rho _ { \ell } = \rho _ { 0 }$ gives $\begin{array} { r } { B = \rho _ { 0 } \sum _ { \ell } M _ { \ell } } \end{array}$ . We seek an allocation $\pmb { m } = \left( m _ { 1 } , \ldots , m _ { L } \right)$ under budget B, evaluated by validation loss. For architecture allocation, coverage levels use nested pair subsets from a shared frequency bank. Fixed-carrier allocation preserves complete query/key widths; physical MLA sets $d _ { \mathrm { r o p e } } ^ { ( \ell ) } = 2 m _ { \ell } ,$ keeping content-branch and latent widths fixed.

## 3 Layer-Wise Positional Demand and Coverage-Upgrade Benefits

Section 2 expresses the RoPE configuration as a layer-wise allocation vector m satisfying Equation (3). To determine rotary coverage at individual layers, we perform layer-wise phase interventions on pretrained models, measure the resulting changes in task loss, and examine their relationship with coverage-upgrade benefits.

## 3.1 Experimental Setup and Positional-Demand Measurement

Experimental setup. We conduct layer-wise evaluations on DeepSeek-MoE-16B [7], LLaMA-2-7B [25], LLaMA-3.2-3B [17], and Qwen3-30B [29]. All models are evaluated on the same 500 text samples, with model weights frozen throughout measurement. Layer numbers in this analysis are one-based. Within each model, the reference and intervention configurations use identical input token sequences, attention masks, and loss-evaluation positions. Paired loss diferences quantify the efect of changing the rotary configuration.

Phase-of score. Following the fixed-carrier parameterization in Section 2, we use the configuration with all candidate rotary pairs activated as the reference. Let $ { \mathcal { L } } _ {  { \ell } } ( x ; \rho )$ denote the average per-token validation loss on evaluation text sample x when layer ℓ uses coverage $\rho ,$ with all other layers retaining their reference configurations. Phase-of corresponds to $\rho = 0$ , setting $R _ { \ell , t } = I$ while preserving the pre-rotation query/key representations and the value branch. Given an evaluation sample set $\mathcal { D } _ { \mathrm { p r o f } }$ , we define the layer-wise positional-demand score as

$$
\widehat { D } _ { \ell } = \frac { 1 } { | \mathcal { D } _ { \mathrm { p r o f } } | } \sum _ { x \in \mathcal { D } _ { \mathrm { p r o f } } } \left[ \mathcal { L } _ { \ell } ( x ; 0 ) - \mathcal { L } _ { \ell } ( x ; 1 ) \right] .\tag{4}
$$

A larger $\widehat { D } _ { \ell }$ indicates a greater loss increase when the layer’s rotary phase is disabled. Evaluating each layer yields the positional-demand profile $\widehat { \pmb { D } } = ( \widehat { D } _ { 1 } , \dots , \widehat { D } _ { L } )$ , with confidence intervals computed from sample-level paired loss diferences.

## 3.2 Empirical Distribution of Layer-Wise Positional Demand

Variation across layers. Figure 1 displays signed demand proportional to $\widehat { D } _ { \ell } / \operatorname* { m a x } _ { j } \widehat { D } _ { j }$ at normalized depth $( \ell - { \frac { 1 } { 2 } } ) / L$ . Each profile exhibits pronounced local peaks, with substantial diferences between highscoring layers and their neighbors. To quantify concentration, we measure the positive-demand share of a depth region as the sum of $[ \widehat { D } _ { \ell } ] _ { + }$ within that region divided by the corresponding sum over all layers, where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ . Table 1 shows that positive demand in DeepSeek-MoE-16B, LLaMA-2-7B, and LLaMA-3.2-3B is concentrated in the first quarter of depth. In LLaMA-2-7B, layer 2 alone accounts for 40.73% of the model’s total positive demand.

Depth distributions across models. Figure 1 also reveals diferences between models. The principal peaks in the DeepSeek and LLaMA models occur in early layers, whereas Qwen3-30B exhibits elevated demand at both ends of the network. Accordingly, its final quarter of depth accounts for a substantially larger share of positive demand than in the other models. The locations of high-scoring layers and the concentration of demand together characterize each model’s layer-wise positional structure.

![](images/88f45c91c12e8635dbb9fc6cfb826642135e789f00fd433f2df6e8b0bbe17d01.jpg)  
Figure 1: Layer-wise phase-of demand across normalized depth. Signed scores are scaled by each model’s positive peak for display.

Table 1: Positive phase-of demand shares in the first and last depth quarters of each model.
<table><tr><td colspan="2">Model</td><td>First quarter Last quarter</td></tr><tr><td>DeepSeek-MoE-16B</td><td>79.57%</td><td>6.74%</td></tr><tr><td>LLaMA-2-7B</td><td>93.79%</td><td>0.75%</td></tr><tr><td>LLaMA-3.2-3B</td><td>79.16%</td><td>5.74%</td></tr><tr><td>Qwen3-30B</td><td>55.42%</td><td>24.57%</td></tr></table>

## 3.3 Relating Positional Demand to Coverage-Upgrade Benefits

Coverage intervention setup. To examine how positional-demand scores inform coverage allocation, we select two layers from each model’s high-, medium-, and low-demand groups and measure task loss at diferent coverage levels on 100 calibration samples. Configurations retain the pretrained frequency bank and select rotary pairs with a fixed stratified policy. Unrotated dimensions receive the identity transformation and continue to participate in query/key matching. All coverage configurations within calibration use the same evaluation text samples, with other layers retaining their reference configurations. The profiling and calibration sample sets are denoted separately by $\mathcal { D } _ { \mathrm { p r o f } }$ and $\mathcal { D } _ { \mathrm { c a l } }$ . We evaluate $\rho \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ and consider three upgrades: 25%→ 75%, 50%→ 100%, and $2 5 \% \to 1 0 0 \%$

For low and high coverage levels $\rho _ { L } < \rho _ { H }$ , let $\mathcal { D } _ { \mathrm { c a l } }$ denote the fixed calibration sample set. We define the coverage-upgrade benefit at layer ℓ as

$$
\widehat { G } _ { \ell } \bigl ( \rho _ { L } , \rho _ { H } \bigr ) = \frac { 1 } { | \mathscr { D } _ { \mathrm { c a l } } | } \sum _ { x \in \mathscr { D } _ { \mathrm { c a l } } } \left[ \mathscr { L } _ { \ell } \bigl ( x ; \rho _ { L } \bigr ) - \mathscr { L } _ { \ell } \bigl ( x ; \rho _ { H } \bigr ) \right] .\tag{5}
$$

A larger $\widehat { G } _ { \ell }$ indicates a greater reduction in task loss from increasing the layer’s rotary coverage.

Correspondence between demand and benefit. Figure 2 relates phase-of scores to the benefits of upgrading coverage from 50% to 100%. Within each model, high-demand layers typically exhibit larger upgrade benefits, while low-demand layers yield smaller benefits. For the 50% → 100% upgrade, within-model Spearman correlations are 0.943, 1.000, 0.943, 1.000 in the model order above; the other two intervals range from 0.943 to 1.000. These results connect the loss changes induced by phase-of interventions to the utility of increasing rotary coverage.

Coverage responses across layers. The benefits of additional coverage arise at diferent intervals even within one model. In LLaMA-2-7B, layer 1 recovers most positional utility by half coverage, whereas layer 2 improves mainly between 50% and 75% coverage. Appendix Figure 3 compares these responses under the same evaluation samples.

Combining the layer-wise distributions with the coverage intervention results yields the following observation:

Key observation. Positional demand in pretrained models is strongly non-uniform across depth, with model-specific high-demand regions. Across the evaluated calibration layers and coverage upgrades, phaseof scores exhibit strong ranking agreement with the measured upgrade benefits, providing a task-level basis for selecting high-coverage layers.

This observation motivates demand-guided budget allocation: under Equation (3), the profile guides the selection of high-coverage layers, and high/low configurations organize rotary resources across depth. Section 4 analyzes this design principle, and Section 5 develops the allocation method and its cross-scale implementation.

## 4 Motivating Analysis

We analyze budgeted coverage allocation at a frozen checkpoint to motivate the layer-selection and profile transfer rules. Consider high/low configurations with a common per-layer rotary-pair upgrade cost, so that Equation (3) permits K high-coverage layers. Let $F ( S )$ be the loss of high-coverage set S, using identity rotation for unselected Dense pairs and zeroing unselected MLA positional coordinates, as specified in Appendix A.1. Define $G _ { \ell } = F ( [ L ] \setminus \{ \ell \} ) - F ( [ L ] )$ . Let $s _ { \ell } = { \left[ { a _ { \ell } } \right] } .$ , or a common positive normalization of these priorities, where a is the recorded matched-marginal allocation score obtained from Equation (58). Denote its top-K set by $\widehat S$ and a loss-minimizing K-layer set by $S ^ { \star }$

Theorem 1 (Informal). Calibrated profiles guide budgeted allocation. Suppose $\left| G _ { \ell } - \alpha s _ { \ell } \right| \leq \varepsilon ~ f o r$ every layer, with $\alpha > 0$ and joint losses satisfy $\begin{array} { r } { F ( S ) = C - \sum _ { \ell \in S } G _ { \ell } + I ( S ) } \end{array}$ , where C is independent of S and $| I ( S ) | \leq \delta$ for every feasible set. Then

$$
0 \leq F ( \widehat { S } ) - F ( S ^ { \star } ) \leq 2 K \varepsilon + 2 \delta .\tag{6}
$$

Exact calibration and additive efects give an optimal high/low allocation. The formal statement and proof appear in Appendix A.2.

Theorem 1 relates the quality of score-based layer selection to calibration accuracy and joint-configuration efects. The calibration condition connects the measured allocation priorities to the all-high reference gains.

Theorem 2 (Informal). Profile accuracy controls cross-scale allocation. Suppose a normalized target profile $\mathbf { \boldsymbol { s } } ^ { ( t ) }$ satisfies the preceding calibration and interaction conditions with parameters $\alpha _ { t } , \varepsilon _ { t } , \delta _ { t }$ If a transferred profile s̃ satisfies $\| \widetilde { \pmb { s } } - \pmb { s } ^ { ( \bar { t } ) } \| _ { \infty } \leq \eta .$ , its top- $\cdot K _ { t }$ set $\widehat { S } _ { s  t }$ obeys

$$
F _ { t } ( \widehat { S } _ { s  t } ) - F _ { t } ( S _ { t } ^ { \star } ) \leq 2 K _ { t } ( \varepsilon _ { t } + \alpha _ { t } \eta ) + 2 \delta _ { t } .\tag{7}
$$

The formal statement and proof appear in Appendix A.3.

Theorem 2 quantifies how profile discrepancy propagates to target-layer selection.

Connection to the empirical observations. Theorem 1 formalizes the score-based selection principle motivated by the layer-wise heterogeneity and demand–benefit ranking agreement observed in Section 3. Under its conditions, for any feasible set S, replacing a selected layer $j \in S$ with an unselected layer $i \not \in S$ lowers the configuration loss whenever $\alpha ( s _ { i } - s _ { j } ) > 2 \varepsilon + 2 \delta$ , as established in Lemma 2. Thus, suficiently separated allocation scores identify beneficial budget-preserving exchanges, giving the observed high-demand regions a direct role in layer selection. Theorem 2 further connects normalized-depth profile agreement to the additional allocation-error term $2 K _ { t } \alpha _ { t } \eta$ . Together, these conclusions motivate selecting high-coverage layers from model-specific allocation profiles and reusing small-model profiles for target-scale architecture design in Section 5.

![](images/a468908efa92b23c1c894dd5087b6fc59d0aecaf702c88e819baf36b2940c0da.jpg)  
Phase-off validation loss increase (normalized)  
Figure 2: Phase-of demand versus validation-loss gain from 50% → 100% coverage. Six layers per model; signed axes use within-model peak normalization.

## 5 Profiling-Guided Layer-Wise RoPE Allocation

Section 3 connects layer-wise positional demand to coverage-upgrade benefits, and Section 4 motivates score-based selection and cross-scale transfer. We turn these insights into a layer-wise architecture: the rotary budget determines the number of high-coverage layers, the profile determines their locations, and a smaller model from the same family supplies the allocation before target training.

## 5.1 Demand-Guided Budget Allocation

We train a full-rotation reference model for 10% of its total training steps and obtain layer-wise allocation scores. Denote the measured score by $a _ { \ell }$ and use $\begin{array} { r } { s _ { \ell } = [ a _ { \ell } ] . } \end{array}$ as the upgrade priority. The phase interventions and coverage calibration in Section 3 connect positional demand to configuration utility; the coverage-benefit measurements and score records used in the experiments are described in Appendix C.1.

Following the high/low configuration in Section 4, let $m _ { \ell , H } = M _ { \ell }$ and $m _ { \ell , L } = M _ { \ell } - \Delta m \ge 0$ , where $\Delta m > 0$ is a common integer upgrade increment. Given total budget B, the high-coverage count and allocation are

$$
K = \frac { { \cal B } - \sum _ { \ell = 1 } ^ { L } m _ { \ell , L } } { \Delta m } , \quad \widehat { \cal S } = \mathrm { T o p K } _ { \ell \in [ L ] } ( s _ { \ell } ) , \quad m _ { \ell } = m _ { \ell , L } + \Delta m { \bf 1 } \{ \ell \in \widehat { \cal S } \} , \quad \rho _ { \ell } = \frac { m _ { \ell } } { M _ { \ell } } .\tag{8}
$$

The coverage specifications and budget give an integer $1 \leq K < L$ , with ties resolved by a fixed rule. The resulting allocation satisfies $\begin{array} { r } { \sum _ { \ell } m _ { \ell } = B } \end{array}$

From positional demand to architecture. The high-demand regions in Section 3 identify layers with strong dependence on rotary phase, and coverage calibration further characterizes their upgrade benefits.

Low coverage supplies a base budget, while the early matched-marginal scores in Equation (58) assign additional pairs to the selected layers. This makes where position matters an executable architectural decision. Theorem 1 motivates this selection rule through score calibration and inter-layer interactions.

## 5.2 Architectural Instantiations

Shared frequencies and fixed-carrier allocation. Let $\mathcal { T } _ { \ell } ( m )$ denote the active frequency-pair set at layer $\ell ,$ with $| \mathcal { I } _ { \ell } ( m ) | = m$ and the low-coverage set nested within the high-coverage set. Selected pairs retain their frequencies from the shared bank. The pretrained-model analysis uses stratified pair selection; the allocation experiments use the high\_to\_low order specified in Appendix C.1. Within each experiment, the frequency bank and selection policy remain fixed across coverage levels. Fixed-carrier allocation applies position-dependent rotation to the selected pairs and the identity elsewhere, with all dimensions participating in attention matching. Queries and keys use matched frequency sets, and all heads in a layer share its coverage configuration.

Physical MLA allocation. For MLA [14, 15], we set $d _ { \mathrm { r o p e } } ^ { ( \ell ) } = 2 m _ { \ell } .$ , construct rotary-query and cross-head shared positional-key projections at this width, and use the frequencies indexed by $\mathcal { T } _ { \ell } ( m _ { \ell } )$ . The content branch, latent width, and head count remain fixed, while attention scaling follows the full-rotation reference. The allocation vector therefore determines both the layer-wise distribution of positional resources and the corresponding projection structure.

## 5.3 Within-Family Profile Transfer

For source and target depths $L _ { s }$ and $L _ { t } .$ , define normalized depths $x _ { i } ^ { ( s ) } = ( i - \frac { 1 } { 2 } ) / L _ { s }$ and $x _ { j } ^ { ( t ) } = ( j - \textstyle { \frac { 1 } { 2 } } ) / L _ { t }$ Normalize the source scores obtained at 10% training progress by their positive-score mean and map them to target depths:

$$
\widehat { \boldsymbol { s } } _ { i } ^ { ( s ) } = \frac { [ a _ { i } ^ { ( s ) } ] _ { + } } { \widehat { \mu } _ { s } } , \quad \widehat { \mu } _ { s } = \frac { 1 } { L _ { s } } \sum _ { u = 1 } ^ { L _ { s } } [ a _ { u } ^ { ( s ) } ] _ { + } > 0 , \quad \widetilde { \boldsymbol { s } } = \boldsymbol { W } \widehat { \boldsymbol { s } } ^ { ( s ) } .\tag{9}
$$

Following Appendix A.3, $W \in \mathbb { R } ^ { L _ { t } \times L _ { s } }$ uses piecewise-linear interpolation with constant endpoint extension, satisfying $W _ { j i } \geq 0$ and $\textstyle \sum _ { i } W _ { j i } = 1$ . Substituting the transferred scores and target budget into Equation (8) gives the target allocation $\bar { m } ^ { ( t ) }$ . We then construct the target architecture, initialize its weights, and train from scratch with the allocation and frequency sets fixed.

Small-model profiles for target architecture design. Normalized depth maps source high-demand regions to diferent target depths, while the target budget determines their coverage levels. One early profile supports multiple target depths and rotary budgets, amortizing structural measurement across architecture designs. Theorem 2 connects the mapped score discrepancy to target selection quality and motivates this procedure.

## 5.4 Storage and Computational Complexity

We quantify BF16 weight-plus-KV-cache storage and matrix FLOPs, including configuration acquisition in total training cost.

Fixed-carrier allocation. Each layer retains its complete query/key projections and cache layout, while the rotary configuration changes the pairs receiving phase transformations. Coverage allocations therefore have the same parameter count, weight and cache storage, and principal matrix-multiplication dimensions. The additional computation for demand-guided configurations comes from early reference training and profiling.

Physical MLA allocation. With layer width, head count, content branch, and latent specifications fixed, rotary-query and positional-key projection parameters, positional-key storage, and the associated matrix multiplications scale linearly with total rotary budget B. Uniform and demand-guided allocations at the same budget preserve these aggregate resource costs; reducing the budget decreases the corresponding parameters, storage, and computation. Layer-wise demand determines where these resources are placed, connecting the budget constraint to model quality.

Algorithm 1 Profiling-guided layer-wise RoPE allocation   
Require: Reference model and training recipe; evaluation samples; target architecture; rotary budget and feasible   
high/low specifications; shared frequency bank; target training data and recipe.   
Ensure: Target allocation $\mathbf { \Omega } _ { m } ^ { ( t ) }$ and pretrained target model.   
1: Train the reference model for 10% of its total steps and obtain layer-wise allocation scores.   
2: if using cross-scale transfer then   
3: Normalize and map the scores to target depths using Equation (9).   
4: else   
5: Use the scores of corresponding layers, retaining their nonnegative parts.   
6: end if   
7: Compute the high-coverage count using Equation (8) and select the highest-scoring layers with the fixed tie rule.   
8: Generate the layer-wise pair counts and frequency sets, and verify the total rotary budget.   
9: Construct the fixed-carrier or physical MLA architecture and initialize target weights.   
10: Fix the allocation and frequency sets, and pretrain with the target data and recipe.

Early configuration acquisition and cross-scale reuse. Let $\mathcal { C } _ { \mathrm { r e f , f u l l } } ^ { ( s ) } , \mathcal { C } _ { \mathrm { p r o f } } ^ { ( s ) }$ , and $\mathcal { C } _ { \mathrm { t r a i n } } ^ { ( t ) } ( m )$ denote full reference training, source profiling, and target training costs, respectively. When one source profile serves A target configurations, the cost attributed to each target is

$$
\mathscr { C } _ { \mathrm { E 2 E } } ( { m } ) = \mathscr { C } _ { \mathrm { t r a i n } } ^ { ( t ) } ( { m } ) + \frac { 0 . 1 \mathscr { C } _ { \mathrm { r e f , f u l l } } ^ { ( s ) } + \mathscr { C } _ { \mathrm { p r o f } } ^ { ( s ) } } { A } .\tag{10}
$$

Smaller models reduce reference-training and layer-wise probing computation, and profile reuse further amortizes configuration acquisition. Appendix B derives the general cost relations; Appendix C provides storage and FLOPs counts for the experimental models and the resulting resource comparisons.

## 6 Experiments

## 6.1 Experimental Setup

We pretrain Dense Transformers [26] and MLA Transformers [14, 15] at 130M, 350M, and 1B scales on approximately 2.6B, 7B, and 20B tokens, respectively. Profiles are collected after 10% of reference training, and allocated target models are trained from scratch. The 130M experiments use same-scale profiles, while larger models reuse within-family 130M profiles; target-direct profiling is evaluated separately (Appendix C).

Configurations and metrics. Full/Uniform50 assign rotary widths 64/32 to every layer; Random50/Demandguided assign width 64 to half the layers and zero to the rest, selected randomly or by early scores. Dense keeps query/key widths fixed, whereas MLA physically narrows rotary branches. Tables 2–3 report validation loss and normalized storage/end-to-end FLOPs. Storage counts BF16 weights and a 4K KV cache; FLOPs include profiling costs.

## 6.2 Dense Quality and Cost

Allocation quality. Demand-guided allocation outperforms Full, Uniform50, and Random50 at all three scales (Table 2). Since query/key widths and storage remain unchanged, the gains reflect how rotary coverage is placed across depth, rather than changes in model capacity. The strongest improvement occurs at 1B using a 130M source profile. Neither uniform half coverage nor random placement consistently outperforms Full at these scales.

Configuration cost. Fixed-carrier allocation preserves matrix-multiplication dimensions. With source acquisition included, end-to-end FLOPs are 1.0147× and 1.0019× Full at 350M and 1B; model storage is unchanged.

Table 2: Dense validation loss (top) and theoretical storage / end-to-end FLOPs ratios to Full (bottom).
<table><tr><td>Model</td><td>Full</td><td>Uniform50</td><td>Random50</td><td>Demand- guided</td></tr><tr><td>130M</td><td>3.374035 1.000/1.0000</td><td>3.379236 1.000/1.0000</td><td>3.378968 1.000/1.0000</td><td>3.368072 1.000/1.1003</td></tr><tr><td>350M</td><td>3.026883 1.000/1.0000</td><td>3.032038 1.000/1.0000</td><td>3.031077 1.000/1.0000</td><td>3.023735 1.000/1.0147</td></tr><tr><td>1B</td><td>2.785044 1.000/1.0000</td><td>2.797076 1.000/1.0000</td><td>2.792187 1.000/1.0000</td><td>2.757636 1.000/1.0019</td></tr></table>

Table 3: MLA validation loss (top) and theoretical storage / end-to-end FLOPs ratios to Full (bottom).
<table><tr><td>Model</td><td>Full</td><td>Uniform50</td><td>Random50 mean</td><td>Demand- guided</td></tr><tr><td>130M</td><td>3.377954 1.000/1.0000</td><td>3.425247 0.961/0.9302</td><td>3.412386 0.961/0.9302</td><td>3.381707 0.961/1.0305</td></tr><tr><td>350M</td><td>3.052147 1.000/1.0000</td><td>3.067263 0.965/0.9326</td><td>3.062779 0.965/0.9326</td><td>3.051440 0.965/0.9476</td></tr><tr><td>1B</td><td>2.750373 1.000/1.0000</td><td>NA</td><td>NA</td><td>2.752074 0.964/0.9396</td></tr></table>

## 6.3 MLA Quality and Physical Resources

Matched-budget comparisons. MLA physically narrows rotary-query and positional-key branches, unlike fixed-carrier Dense. At both 130M and 350M, demand-guided allocation beats uniform and random half-budget controls. Their equal aggregate rotary width isolates the efect of layer placement (Table 3).

Quality–eficiency trade-of. At 350M, the transferred configuration slightly improves on Full while using 0.965× storage and 0.9476× end-to-end FLOPs. At 1B, it nearly matches Full with 0.964× storage and 0.9396× FLOPs; budget-matched controls are unavailable at this scale. The 130M configuration also approaches Full, although same-scale profile acquisition raises its end-to-end FLOPs ratio to 1.0305×. Thus, net computational savings depend on the cost of acquiring the source profile for architecture selection.

## 6.4 Layer Placement and Rotary Budget

Layer-placement ablations. At 130M MLA, demand-guided allocation outperforms reversed ranking, last-ten-layer placement, and the mean of three random layer sets; transferred 350M MLA also beats the mean of two random layer sets. These matched-budget tests favor demand-guided placement over arbitrary alternatives (Appendix C.5).

Budget ablation. Profile50 improves on Profile75 at 130M MLA despite using fewer rotary pairs, indicating a non-monotone validation-loss response. The ranking is fixed, while the number of high-coverage layers changes across budgets (Appendix C.6).

## 6.5 Cross-Scale Transfer and Replication

Source-profile transfer. A 130M early profile determines 350M and 1B allocations without target-scale probing. Both 350M Dense and MLA improve over matched-budget controls, with acquisition costs included in Tables 2–3.

Paired replications. Across two paired 350M MLA training groups, Transfer50 improves over Uniform50 by 0.011348 validation loss on average. A separate target-direct comparison also improves 1B Dense against its paired baseline. Comparisons are made within training groups before averaging (Appendix C.7).

## 7 Conclusion

We presented profiling-guided layer-wise RoPE allocation, translating model-specific positional demand into budgeted architecture design. Early profiles support within-family cross-scale transfer, with experiments showing matched-budget validation loss improvements and resource analysis identifying favorable quality– storage–compute trade-ofs for physical MLA.

## 8 Limitations

Our fixed high/low design simplifies architecture construction, while its coarse granularity leaves room for finer resource allocation. Transfer relies on within-family structural correspondence, motivating further exploration of reusable early profiles across broader architectures and longer contexts.

## References

[1] J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 4895–4901, 2023.

[2] S. Ashkboos, M. Croci, M. Gennari do Nascimento, T. Hoefler, and J. Hensman. Slicegpt: Compress large language models by deleting rows and columns. In International Conference on Learning Representations, volume 2024, pages 11682–11701, 2024.

[3] F. Barbero, A. Vitvitskyi, C. Perivolaropoulos, R. Pascanu, and P. Veličković. Round and round we go! what makes rotary positional encodings useful? arXiv preprint arXiv:2410.06205, 2024.

[4] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raf, et al. Pythia: A suite for analyzing large language models across training and scaling. In International conference on machine learning, pages 2397–2430. PMLR, 2023

[5] S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell, J. Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. In Proceedings of BigScience Episode# 5–Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136, 2022.

[6] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of machine learning research, 24(240):1–113, 2023.

[7] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 1280–1297, 2024.

[8] Z. Dai, Z. Yang, Y. Yang, J. G. Carbonell, Q. Le, and R. Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 2978–2988, 2019.

[9] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. FlashAttention: Fast and memory-eficient exact attention with IO-awareness. arXiv preprint arXiv:2205.14135, 2022.

[10] Y. Ding, L. L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang. Longrope: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753, 2024.

[11] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[12] M. A. Khan, K. P. Gummadi, M. Gupta, and A. Ravichander. Fractional rotation, full potential? investigating performance and convergence of partial rope. arXiv preprint arXiv:2603.11611, 2026.

[13] B. Kong, J. Liang, Y. Liu, R. Deng, and K. Yuan. Cr-net: Scaling parameter-eficient training with cross-layer low-rank structure. In International Conference on Learning Representations, volume 2026, pages 144485–144514, 2026.

[14] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo, et al. Deepseek-v2: A strong, economical, and eficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

[15] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[16] X. Men, M. Xu, Q. Zhang, Q. Yuan, B. Wang, H. Lin, Y. Lu, X. Han, and W. Chen. Shortgpt: Layers in large language models are more redundant than you expect. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20192–20204, 2025.

[17] Meta. Llama 3.2 Model Card. Model card, 2024.

[18] B. Peng, J. Quesnelle, H. Fan, and E. Shippole. Yarn: Eficient context window extension of large language models. In International Conference on Learning Representations, volume 2024, pages 31932–31951, 2024.

[19] O. Press, N. A. Smith, and M. Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

[20] P. Shaw, J. Uszkoreit, and A. Vaswani. Self-attention with relative position representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468, 2018.

[21] N. Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

[22] R. Shi, B. Yin, and G. Li. Modern transformers are implicit hybrids: From functional diferentiation to principled hybrid architecture design. arXiv preprint arXiv:2609.02986, 2026.

[23] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

[24] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[25] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[27] X. Wang, S. Alam, Z. Wan, H. Shen, and M. Zhang. Svd-llm v2: Optimizing singular value truncation for large language model compression. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4287–4296, 2025.

[28] X. Wang, Y. Zheng, Z. Wan, and M. Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. In International Conference on Learning Representations, volume 2025, pages 19299–19319, 2025.

[29] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[30] B. Yang, B. Venkitesh, D. G. Talupuru, H. Lin, D. Cairuz, P. Blunsom, and A. Locatelli. Rope to nope and back again: A new hybrid attention strategy. Advances in Neural Information Processing Systems, 38:64133–64157, 2025.

[31] Z. Yuan, Y. Shang, Y. Song, D. Yang, Q. Wu, Y. Yan, and G. Sun. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.

## Appendix

## Contents

1 Introduction 1   
1.1 Related Work . 3   
Preliminaries and Problem Formulation 3   
3 Layer-Wise Positional Demand and Coverage-Upgrade Benefits 4   
3.1 Experimental Setup and Positional-Demand Measurement 4   
3.2 Empirical Distribution of Layer-Wise Positional Demand 4   
3.3 Relating Positional Demand to Coverage-Upgrade Benefits 5   
4 Motivating Analysis 6   
5 Profiling-Guided Layer-Wise RoPE Allocation 7   
5.1 Demand-Guided Budget Allocation 7   
5.2 Architectural Instantiations 8   
5.3 Within-Family Profile Transfer . 8   
5.4 Storage and Computational Complexity 8   
6 Experiments 9   
6.1 Experimental Setup 9   
6.2 Dense Quality and Cost . 9   
6.3 MLA Quality and Physical Resources 10   
6.4 Layer Placement and Rotary Budget 10   
6.5 Cross-Scale Transfer and Replication 10   
7 Conclusion 11   
8 Limitations 11   
A Analysis of Profiling-Guided Rotary Allocation 15   
A.1 Setup and Reference-Consistent Loss Decomposition 15   
A.2 Calibrated Profiles and Budgeted Layer Selection 16   
A.3 Normalized-Depth Profile Transfer 18   
B Cost Analysis of Early Profiling and Cross-Scale Configuration 19   
B.1 Early Reference Training and Layer-Wise Probing 19   
B.2 Same-Scale and Cross-Scale Configuration Acquisition 19   
B.3 End-to-End Cost and Profile Reuse 20   
C Experimental Settings, Resource Accounting, and Additional Results 21   
C.1 Models, Training, and Profiling Settings 21   
C.2 Resource Evaluation Settings and Normalization 21   
C.3 Parameter and Storage Counts for the Experimental Models . 22   
C.4 Training and Configuration FLOPs for the Experimental Models 23   
C.5 Frozen Layer Sets and Layer-Selection Ablations 25   
C.6 Rotary Budget and Quality–Resource Trade-ofs 25   
C.7 Cross-Scale and Paired Training 25   
C.8 Additional Experiments with a Narrower KV Latent 27   
C.9 Representative Coverage Responses 28

## A Analysis of Profiling-Guided Rotary Allocation

## A.1 Setup and Reference-Consistent Loss Decomposition

Feasible high/low configurations. Let $\left[ L \right] = \left\{ 1 , \ldots , L \right\}$ . At layer ℓ, choose integer pair counts $0 \leq m _ { \ell , L } <$ $m _ { \ell , H } = M _ { \ell }$ with a common increment $m _ { \ell , H } - m _ { \ell , L } = \Delta m > 0$ . Fix an integer $1 \le K < L$ and a budget

$$
B = \sum _ { \ell = 1 } ^ { L } m _ { \ell , L } + K \Delta m .\tag{11}
$$

For a high-coverage set $S \subseteq [ L ]$ , define

$$
m _ { \ell } ( S ) = \left\{ m _ { \ell , H } , \quad \ell \in S , \quad \ S _ { K } = \{ S \subseteq [ L ] : | S | = K \} . \right.\tag{12}
$$

Indeed, $\Sigma _ { \ell } m _ { \ell } ( S ) = \Sigma _ { \ell } m _ { \ell , L } + | S | \Delta m$ , so the budget in Equation (11) is equivalent to $S \in { \cal S } _ { K }$ . The low coverage at layer ℓ is $\rho _ { \ell , L } = m _ { \ell , L } / M _ { \ell }$ , and the high coverage is one. Each layer uses the nested frequency-pair subsets specified in Section 2.

Frozen-checkpoint objective. Fix model weights and an evaluation distribution ${ \mathcal { P } } ,$ which may be the empirical distribution on a fixed evaluation sample set. Let $\mathcal { L } ( x ; m ( S ) )$ denote the mean per-token validation loss on text sample x under allocation $m ( S )$ , and define

$$
F ( S ) = \mathbb { E } _ { x \sim \mathcal { P } } \big [ \mathcal { L } ( x ; \pmb { m } ( S ) ) \big ] , \qquad S \subseteq [ L ] .\tag{13}
$$

Assume these losses are finite. Within each model, all evaluations use its fixed intervention rule: unselected frequency pairs receive identity rotation in Dense, while unselected positional coordinates are zeroed in MLA, as specified in Appendix C.1. The shared frequency bank and attention scaling remain fixed across allocations. Write $F _ { H } = F ( [ L ] )$ for the all-high reference loss and define

$$
G _ { \ell } = F ( [ L ] \setminus \{ \ell \} ) - F _ { H } .\tag{14}
$$

Thus, $G _ { \ell }$ is the benefit of restoring layer ℓ from low to high coverage while all other layers remain high. For the identity intervention with the same checkpoint and $\mathcal { P }$ equal to the empirical distribution on $\mathcal { D } _ { \mathrm { c a l } }$ , this recovers $\widehat { G } _ { \ell } ( \rho _ { \ell , L } , 1 )$ in Equation (5).

Joint-intervention residual. For every $S \subseteq [ L ]$ , set

$$
I ( S ) = F ( S ) - F _ { H } - \sum _ { \ell \neq S } G _ { \ell } ,\tag{15}
$$

$$
C = F _ { H } + \sum _ { \ell = 1 } ^ { L } G _ { \ell } .\tag{16}
$$

These definitions give the exact identity

$$
F ( S ) = C - \sum _ { \ell \in S } G _ { \ell } + I ( S ) .\tag{17}
$$

The residual $I ( S )$ measures the diference between the joint loss change and the sum of single-layer loss changes at the same all-high reference. In particular, $I ( [ L ] ) = 0$ and $I ( [ L ] \setminus \{ \ell \} ) = 0$ . For a joint change, its magnitude can be controlled through mixed finite diferences, as shown next.

Lemma 1 (Control of joint-intervention efects). Let $q = L - K$ , define $f ( T ) = F ( [ L ] \setminus T )$ , and suppose there are symmetric constants $b _ { i j } = b _ { j i } \geq 0$ such that

$$
\begin{array} { r }  | f ( U \cup \{ i , j \} ) - f ( U \cup \{ i \} ) | | | f ( U \backslash \{ i \} ) | | | | f ( U \backslash \{ i \} ) | | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f ( U ) | | | f ( U ) | | | | f | f ( U ) | | | | f | | f | f ( U ) | | | | f | | f | f ( U | | | | | | f | | f | | f | | f | | | | f | | f | | f | | f | | f | | | | | f | | f | | f | f | | f | | f | f | | | | f | | f | f | f | | f | | | f | | f | f | f | | | f | f | | f | f | | f | f | f | | f | f | f | f | f | f | f | | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | f | \end{array}\tag{18}
$$

for all distinct $i , j$ and every $U \subseteq [ L ] \setminus \{ i , j \}$ . Then every $S \in { \cal S } _ { K }$ satisfies

$$
| I ( S ) | \leq \sum _ { i < j \atop i , j \not \in S } b _ { i j } .\tag{19}
$$

Consequently, one may take

$$
\delta = \operatorname* { m a x } _ { S \in \cal { S } _ { K } } \sum _ { i < j \atop i , j \notin { S } } b _ { i j } .\tag{20}
$$

If $b _ { i j } \leq b$ for all pairs, then $\delta \leq { \binom { q } { 2 } } b$

Proof. Step 1: Express the joint change as sequential interventions. Fix $S \in { \cal S } _ { K }$ , write its complement as $T = [ L ] \setminus S = \{ i _ { 1 } , . . . , i _ { q } \}$ , and set $T _ { p } = \left\{ i _ { 1 } , \ldots , i _ { p } \right\}$ , with $T _ { 0 } = \emptyset$ . For $i \notin U$ , define the first finite diference $\Delta _ { i } f ( U ) = f ( U \cup \{ i \} ) - f ( \bar { U } )$ ). Since $G _ { i } = \Delta _ { i } f ( \emptyset )$ , telescoping gives

$$
f ( T ) - f ( \emptyset ) = \sum _ { p = 1 } ^ { q } \Delta _ { i _ { p } } f ( T _ { p - 1 } ) ,\tag{21}
$$

$$
I ( S ) = \sum _ { p = 1 } ^ { q } \left[ \Delta _ { i _ { p } } f ( T _ { p - 1 } ) - \Delta _ { i _ { p } } f ( \emptyset ) \right] .\tag{22}
$$

Step 2: Expand each background-dependent change. For $p \geq 2 ,$ , telescope along the previously changed layers:

$$
\Delta _ { i _ { p } } f ( T _ { p - 1 } ) - \Delta _ { i _ { p } } f ( \emptyset ) = \sum _ { a = 1 } ^ { p - 1 } \left[ \Delta _ { i _ { p } } f ( T _ { a } ) - \Delta _ { i _ { p } } f ( T _ { a - 1 } ) \right] = \sum _ { a = 1 } ^ { p - 1 } \left[ { f ( T _ { a - 1 } \cup \left\{ { i _ { a } , i _ { p } } \right\} ) } \atop - { f ( T _ { a - 1 } \cup \left\{ { i _ { a } } \right\} ) } \right. \qquad\tag{23}
$$

Each bracket is a mixed finite diference on a valid background, since $i _ { a } , i _ { p } \notin T _ { a - 1 }$

Step 3: Sum the pairwise bounds. Applying Equation (18) to every bracket and then using the triangle inequality yields

$$
\left| I ( S ) \right| \le \sum _ { p = 2 } ^ { q } \sum _ { a = 1 } ^ { p - 1 } b _ { i _ { a } i _ { p } } = \sum _ { i < j \atop i , j \not \in S } b _ { i j } .\tag{24}
$$

Every unordered pair in $T$ appears exactly once. Taking the maximum over $S \in { \cal S } _ { K }$ proves Equation (20); bounding each of its $\textstyle { \binom { q } { 2 } }$ summands by b gives the final assertion. For $q = 1$ , the sums are empty and $I ( S ) = 0$ □

## A.2 Calibrated Profiles and Budgeted Layer Selection

The following theorem formalizes Theorem 1. For the allocation method, take $s _ { \ell } = \left[ a _ { \ell } \right] .$ , or a common positive normalization of these priorities, where $a \ell$ is the recorded matched-marginal score obtained from Equation (58) under the intervention rule fixed above. The calibration condition relates these priorities to the all-high reference gains in Equation (14); its error ε accounts for diferences in probing background and coverage interval, together with score estimation and processing.

Theorem 3 (Allocation loss under calibrated scores). For the loss and gains defined above, suppose a score vector $\boldsymbol { s } \in \mathbb { R } ^ { L }$ and constants $\alpha > 0 , \varepsilon \geq 0$ , and $\delta \geq 0$ satisfy

$$
\begin{array} { r l } { \left| G _ { \ell } - \alpha s _ { \ell } \right| \le \varepsilon } & { { } f o r \ a l l \ \ell \in [ L ] , } \\ { \left| I ( S ) \right| \le \delta } & { { } f o r \ a l l \ S \in \mathcal { S } _ { K } . } \end{array}\tag{25}
$$

Let $\widehat S$ contain the K largest scores, with ties resolved by a fixed rule. Let $S ^ { \star }$ minimize F over $\boldsymbol { \mathcal { S } } _ { K }$ . If $r = | S ^ { \star } \setminus \widehat S |$ , then

$$
0 \leq F ( \widehat { S } ) - F ( S ^ { \star } ) \leq 2 r \varepsilon + 2 \delta \leq 2 K \varepsilon + 2 \delta .\tag{26}
$$

In particular, $\varepsilon = \delta = 0$ implies $F ( { \widehat { S } } ) = \operatorname* { m i n } _ { S \in S _ { K } } F ( S )$

Proof. Step 1: Establish the top-K score inequality. Since $\boldsymbol { \mathcal { S } } _ { K }$ is finite and $F$ is finite, a minimizing set $S ^ { \star }$ exists. Define the disjoint sets $\bar { S } _ { + } = S ^ { \star } \setminus \widehat { S }$ and $S _ { - } = \widehat { S } \setminus S ^ { \star }$ . Both have cardinality $r ,$ because $\left| S ^ { \star } \right| = \left| \widehat { S } \right| = K$ By the top-K construction, every score indexed by S is at least every score indexed by $S _ { + }$ . Pairing their elements therefore gives

$$
\sum _ { \ell \in S ^ { \star } } s _ { \ell } - \sum _ { \ell \in \widehat { S } } s _ { \ell } = \sum _ { \ell \in S _ { + } } s _ { \ell } - \sum _ { \ell \in S _ { - } } s _ { \ell } \leq 0 .\tag{27}
$$

This also holds when $r = 0$ , in which case both diferences vanish.

Step 2: Translate score ordering into a gain bound. Write $e _ { \ell } = G _ { \ell } - \alpha s _ { \ell }$ , so that $| e \ell | \leq \varepsilon$ . The gains on the shared indices cancel. Using $\alpha > 0$ and Equation (27), we obtain

$$
\sum _ { \ell \in S ^ { \star } } G _ { \ell } - \sum _ { \ell \in \bar { S } } G _ { \ell } = \alpha \left( \sum _ { \ell \in S ^ { \star } } s _ { \ell } - \sum _ { \ell \in \bar { S } } s _ { \ell } \right) + \sum _ { \ell \in S _ { + } } e _ { \ell } - \sum _ { \ell \in S _ { - } } e _ { \ell } = \sum _ { \ell \in S _ { + } } \left| e _ { \ell } \right| + \sum _ { \ell \in S _ { - } } \left| e _ { \ell } \right| \leq 2 r \varepsilon .\tag{28}
$$

Step 3: Account for joint-intervention efects. Applying the exact decomposition in Equation (17) to both configurations $\mathrm { g i }$ ves

$$
F ( \widehat { S } ) - F ( S ^ { \star } ) = \sum _ { \ell \in S ^ { \star } } G _ { \ell } - \sum _ { \ell \in \widehat { S } } G _ { \ell } + I ( \widehat { S } ) - I ( S ^ { \star } ) \leq 2 r \varepsilon + | I ( \widehat { S } ) | + | I ( S ^ { \star } ) | \leq 2 r \varepsilon + 2 \delta .\tag{29}
$$

Since $r \leq K$ , this proves the upper bound. The lower bound follows from the optimality of $S ^ { \star }$

Step 4: Recover the exact-calibration case. If $\varepsilon = \delta = 0$ , the upper and lower bounds both equal zero. Thus $\widehat { S }$ minimizes the configuration loss over $\boldsymbol { \mathcal { S } } _ { K }$ , proving the final assertion. □

The following consequence makes the connection between layer-wise demand diferences and budget-preserving configuration changes explicit.

Lemma 2 (Benefit of a demand-guided layer exchange). Under the conditions of Theorem ${ \mathcal { B } } ,$ let $S \in { \cal S } _ { K }$ choose $i \not \in S$ and $j \in S ,$ , and set $S ^ { \prime } = ( S \setminus \{ j \} ) \cup \{ i \}$ . Then $S ^ { \prime } \in S _ { K }$ and

$$
F ( S ) - F ( S ^ { \prime } ) \geq \alpha ( s _ { i } - s _ { j } ) - 2 \varepsilon - 2 \delta .\tag{30}
$$

Consequently, $\alpha \big ( s _ { i } - s _ { j } \big ) > 2 \varepsilon + 2 \delta$ implies $F ( S ^ { \prime } ) < F ( S )$

Proof. Step 1: Verify feasibility of the exchange. Since $j \in S$ and $i \notin S$ , removing $j$ and adding i preserves cardinality, giving $\left| S ^ { \prime } \right| = \left| S \right| = K$ . The common upgrade increment $\Delta m$ therefore gives

$$
\sum _ { \ell = 1 } ^ { L } m _ { \ell } \bigl ( S ^ { \prime } \bigr ) = \sum _ { \ell = 1 } ^ { L } m _ { \ell , L } + K \Delta m = B .\tag{31}
$$

Thus, $S ^ { \prime }$ is feasible under the same rotary budget.

Step 2: Express the exact loss change. Apply Equation (17) to S and $S ^ { \prime } .$ Their shared indices cancel, yielding

$$
F ( S ) - F ( S ^ { \prime } ) = \sum _ { \ell \in S ^ { \prime } } G _ { \ell } - \sum _ { \ell \in S } G _ { \ell } + I ( S ) - I ( S ^ { \prime } ) = G _ { i } - G _ { j } + I ( S ) - I ( S ^ { \prime } ) .\tag{32}
$$

Step 3: Bound calibration and interaction errors. The calibration conditions imply $G _ { i } \geq \alpha s _ { i } - \varepsilon$ and $G _ { j } \leq \alpha s _ { j } + \varepsilon .$ . Since both sets are feasible, $I ( S ) \geq - \delta$ and $I ( S ^ { \prime } ) \leq \delta$ . Substituting these inequalities into Equation (32) gives

$$
F ( S ) - F ( S ^ { \prime } ) \geq ( \alpha s _ { i } - \varepsilon ) - ( \alpha s _ { j } + \varepsilon ) - 2 \delta = \alpha ( s _ { i } - s _ { j } ) - 2 \varepsilon - 2 \delta .\tag{33}
$$

This proves Equation (30). If $\alpha ( s _ { i } - s _ { j } ) > 2 \varepsilon + 2 \delta$ , the right-hand side is strictly positive, and hence $F ( S ^ { \prime } ) ^ { } < F ( S )$ □

## A.3 Normalized-Depth Profile Transfer

Profiles and interpolation. Consider a source model with $L _ { s }$ layers and a target model with $L _ { t }$ layers in the same family. For $u \in \{ s , t \}$ , define normalized depth and a normalized reference allocation profile by

$$
x _ { \ell } ^ { ( u ) } = \frac { \ell - \frac { 1 } { 2 } } { L _ { u } } , \qquad s _ { \ell } ^ { ( u ) } = \frac { [ \bar { a } _ { \ell } ^ { ( u ) } ] _ { + } } { \mu _ { u } } , \qquad \mu _ { u } = \frac { 1 } { L _ { u } } \sum _ { j = 1 } ^ { L _ { u } } [ \bar { a } _ { j } ^ { ( u ) } ] _ { + } > 0 .\tag{34}
$$

where $\bar { a } _ { \ell } ^ { ( u ) }$ is the population paired allocation score at the fixed reference checkpoint, obtained by replacing the sample average in Equation (58) with expectation under the chosen evaluation distribution and model-specific intervention rule. Let $\widehat { \pmb { s } } ^ { ( s ) }$ be the recorded source allocation profile, normalized as in Equation (9), with a positive denominator. The source-profile error below includes sampling and score aggregation. Interpolation from source depths to target depths is represented by $W \in \mathbb { R } ^ { L _ { t } \times L _ { s } }$ satisfying

$$
W _ { j i } \geq 0 , \qquad \sum _ { i = 1 } ^ { L _ { s } } W _ { j i } = 1 , \qquad \widetilde { \pmb { s } } = W \widehat { \pmb { s } } ^ { ( s ) } .\tag{35}
$$

Piecewise-linear interpolation with constant extension beyond the source endpoints has these properties: an interior target depth uses nonnegative weights on its two neighboring source depths, and an exterior depth uses the nearest source endpoint. The target allocation selects the $K _ { t }$ largest entries of ${ \widetilde { s } } .$

Theorem 4 (Allocation loss under profile transfer). Let $F _ { t } , G _ { j } ^ { ( t ) }$ , and $I _ { t }$ be the target-model quantities defined in Appendix A.1, with feasible sets $\boldsymbol { S } _ { K _ { t } }$ . Suppose

$$
| G _ { j } ^ { ( t ) } - \alpha _ { t } s _ { j } ^ { ( t ) } | \leq \varepsilon _ { t } , \qquad | I _ { t } ( S ) | \leq \delta _ { t }\tag{36}
$$

for every target layer $j$ and every $S \in S _ { K _ { t } }$ , where $\alpha _ { t } > 0$ . Suppose also that the normalized source measurement error and depth-mapped profile discrepancy satisfy

$$
\begin{array} { r } { \| \widehat { \pmb { s } } ^ { ( s ) } - \pmb { s } ^ { ( s ) } \| _ { \infty } \leq \varepsilon _ { s } , \quad \| W \pmb { s } ^ { ( s ) } - \pmb { s } ^ { ( t ) } \| _ { \infty } \leq \eta _ { \mathrm { d e p t h } } . } \end{array}\tag{37}
$$

Let $\widehat { S } _ { s  t }$ be a top- $K _ { t }$ set of ${ \widetilde { s } } ,$ let $S _ { t } ^ { \star }$ minimize $F _ { t }$ over $\boldsymbol { S } _ { K _ { t } }$ , and define $\eta = \left\| \widetilde { \pmb { s } } - \pmb { s } ^ { ( t ) } \right\| _ { \infty }$ . Then $\eta \leq \varepsilon _ { s } + \eta _ { \mathrm { d e p t h } ; }$ and

$$
0 \leq F _ { t } \big ( \widehat { S } _ { s  t } \big ) - F _ { t } \big ( S _ { t } ^ { \star } \big ) \leq 2 K _ { t } \big ( \varepsilon _ { t } + \alpha _ { t } \eta \big ) + 2 \delta _ { t } \leq 2 K _ { t } \big [ \varepsilon _ { t } + \alpha _ { t } \big ( \varepsilon _ { s } + \eta _ { \mathrm { d e p t h } } \big ) \big ] + 2 \delta _ { t } .\tag{38}
$$

Proof. Step 1: Control propagation of source measurement error. For any vector $\pmb { v } \in \mathbb { R } ^ { L _ { s } }$ , nonnegativity and unit row sums of W imply, for each target layer $j .$

$$
\lvert ( W \pmb { v } ) _ { j } \rvert \leq \sum _ { i = 1 } ^ { L _ { s } } W _ { j i } \lvert v _ { i } \rvert \leq \lVert \pmb { v } \rVert _ { \infty } \sum _ { i = 1 } ^ { L _ { s } } W _ { j i } = \lVert \pmb { v } \rVert _ { \infty } .\tag{39}
$$

Taking the maximum over $j$ yields $\| W v \| _ { \infty } \leq \| v \| _ { \infty }$ . With ${ \pmb v } = \widehat { \pmb s } ^ { ( s ) } - { \pmb s } ^ { ( s ) }$ , this gives

$$
\| W { \widehat { \pmb s } } ^ { ( s ) } - W { \pmb s } ^ { ( s ) } \| _ { \infty } \leq \varepsilon _ { s } .\tag{40}
$$

Step 2: Bound the complete transferred-profile discrepancy. Add and subtract $W s ^ { ( s ) }$ , and apply the triangle inequality:

$$
\eta = \| W \widehat { \pmb { s } } ^ { ( s ) } - \pmb { s } ^ { ( t ) } \| _ { \infty } \le \| W ( \widehat { \pmb { s } } ^ { ( s ) } - \pmb { s } ^ { ( s ) } ) \| _ { \infty } + \| W \pmb { s } ^ { ( s ) } - \pmb { s } ^ { ( t ) } \| _ { \infty } \le \varepsilon _ { s } + \eta _ { \mathrm { d e p t h } } .\tag{41}
$$

Step 3: Calibrate the transferred scores to target benefits. For every target layer $j ,$

$$
\begin{array} { r } { \big | G _ { j } ^ { ( t ) } - \alpha _ { t } \widetilde { s } _ { j } \big | \le \big | G _ { j } ^ { ( t ) } - \alpha _ { t } s _ { j } ^ { ( t ) } \big | + \alpha _ { t } \big | s _ { j } ^ { ( t ) } - \widetilde { s } _ { j } \big | \le \varepsilon _ { t } + \alpha _ { t } \eta . } \end{array}\tag{42}
$$

Thus the transferred profile satisfies the calibration condition of Theorem 3 with error $\varepsilon _ { t } + \alpha _ { t } \eta$

Step 4: Apply the allocation bound at the target scale. The target budget defines $\boldsymbol { S } _ { K _ { t } }$ , the selected set maximizes the sum of transferred scores, and the interaction bound in Equation (36) holds on all feasible sets. Theorem 3 therefore gives

$$
0 \leq F _ { t } \big ( \widehat { S } _ { s  t } \big ) - F _ { t } \big ( S _ { t } ^ { \star } \big ) \leq 2 K _ { t } \big ( \varepsilon _ { t } + \alpha _ { t } \eta \big ) + 2 \delta _ { t } .\tag{43}
$$

Substituting Equation (41) proves the second inequality in Equation (38). This also proves the transfer statement in the main text for any transferred profile with discrepancy at most η. □

## B Cost Analysis of Early Profiling and Cross-Scale Configuration

## B.1 Early Reference Training and Layer-Wise Probing

We derive the computational components of configuration acquisition. For model $^ { a , }$ let $L _ { a }$ be its depth, $N _ { a }$ its full training-token budget, $f _ { a }$ its forward matrix-multiplication FLOPs per token, and τ the reference-training fraction.

Early reference training. Counting the forward and corresponding backward matrix multiplications as three forward passes gives the full reference-training cost $\mathcal { C } _ { \mathrm { r e f , f u l l } } ^ { ( a ) } = 3 N _ { a } f _ { a }$ . With a fixed training batch, a fraction τ of training steps processes the same fraction of training tokens, yielding

$$
\mathcal { C } _ { \mathrm { e a r l y } } ^ { ( a ) } = 3 \tau N _ { a } f _ { a } .\tag{44}
$$

We use $\tau = 0 . 1$ , giving $\mathcal { C } _ { \mathrm { e a r l y } } ^ { ( a ) } = 0 . 3 N _ { a } f _ { a }$

Layer-wise interventions. Phase-of measurement performs one reference forward pass and $L _ { a }$ single-layer interventions for each evaluation input batch. Low/high-coverage benefit measurement performs one reference pass and two interventions per layer. Their respective call counts are

$$
q _ { \mathrm { p h a s e } } = L _ { a } + 1 , q _ { \mathrm { g a i n } } = 2 L _ { a } + 1 .\tag{45}
$$

Suppose the selected profile is constructed from E probing events, with global sequence count $b _ { e }$ and sequence length $T _ { e }$ at event e. The cumulative input-token count is

$$
N _ { \mathrm { p r o f } } ^ { ( a ) } = \sum _ { e = 1 } ^ { E } b _ { e } T _ { e } .\tag{46}
$$

On a fixed-width intervention carrier, the corresponding probing costs are

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { p h a s e } } ^ { ( a ) } = \big ( L _ { a } + 1 \big ) N _ { \mathrm { p r o f } } ^ { ( a ) } f _ { a } , \qquad \mathcal { C } _ { \mathrm { g a i n } } ^ { ( a ) } = \big ( 2 L _ { a } + 1 \big ) N _ { \mathrm { p r o f } } ^ { ( a ) } f _ { a } . } \end{array}\tag{47}
$$

The cumulative input count includes all probing events used to construct the final profile. Repeated evaluation of the same inputs across events is counted according to the executed forward passes.

Configuration acquisition. Combining early reference training with the chosen probing procedure gives

$$
\mathcal { C } _ { \mathrm { c o n f i g } } ^ { ( a ) } = 3 \tau N _ { a } f _ { a } + \mathcal { C } _ { \mathrm { p r o f } } ^ { ( a ) } .\tag{48}
$$

## B.2 Same-Scale and Cross-Scale Configuration Acquisition

Consider a source model s and a target model t from the same family, both profiled at the same referencetraining fraction τ.

Reference-training cost ratio. Equation (44) gives

$$
r _ { \mathrm { e a r l y } } ^ { s  t } = { \frac { \mathcal { C } _ { \mathrm { e a r l y } } ^ { ( s ) } } { \mathcal { C } _ { \mathrm { e a r l y } } ^ { ( t ) } } } = { \frac { N _ { s } f _ { s } } { N _ { t } f _ { t } } } .\tag{49}
$$

The reference-training cost therefore depends jointly on the training-token budget and per-token model arithmetic.

Probing cost ratio. For low/high-coverage benefit measurement,

$$
r _ { \mathrm { p r o f } } ^ { s  t } = \frac { ( 2 L _ { s } + 1 ) N _ { \mathrm { p r o f } } ^ { ( s ) } f _ { s } } { ( 2 L _ { t } + 1 ) N _ { \mathrm { p r o f } } ^ { ( t ) } f _ { t } } .\tag{50}
$$

With the same cumulative probing input, this simplifies to

$$
r _ { \mathrm { p r o f } } ^ { s  t } = \frac { \big ( 2 L _ { s } + 1 \big ) f _ { s } } { \big ( 2 L _ { t } + 1 \big ) f _ { t } } .\tag{51}
$$

This relation captures the contributions of model depth and single-pass computation to profiling cost.

Combined cost ratio. Combining reference training and probing yields

$$
\frac { \mathcal C _ { \mathrm { c o n f i g } } ^ { ( s ) } } { \mathcal C _ { \mathrm { c o n f i g } } ^ { ( t ) } } = \frac { r _ { \mathrm { e a r l y } } ^ { s  t } \mathcal C _ { \mathrm { e a r l y } } ^ { ( t ) } + r _ { \mathrm { p r o f } } ^ { s  t } \mathcal C _ { \mathrm { p r o f } } ^ { ( t ) } } { \mathcal C _ { \mathrm { e a r l y } } ^ { ( t ) } + \mathcal C _ { \mathrm { p r o f } } ^ { ( t ) } } .\tag{52}
$$

The two cost weights are nonnegative and sum to one after normalization. The total configuration-cost ratio therefore lies between the component ratios and satisfies

$$
\frac { \mathcal C _ { \mathrm { c o n f i g } } ^ { ( s ) } } { \mathcal C _ { \mathrm { c o n f i g } } ^ { ( t ) } } \le \operatorname* { m a x } \{ r _ { \mathrm { e a r l y } } ^ { s  t } , r _ { \mathrm { p r o f } } ^ { s  t } \} .\tag{53}
$$

This derivation expresses cross-scale configuration eficiency through two measurable components: early reference training and layer-wise probing.

## B.3 End-to-End Cost and Profile Reuse

For target allocation m, let f<sub>t</sub>(m) denote forward FLOPs per token. Target pretraining from scratch costs

$$
\mathcal { C } _ { \mathrm { t r a i n } } ^ { ( t ) } ( \pmb { m } ) = 3 N _ { t } f _ { t } ( \pmb { m } ) .\tag{54}
$$

When one source profile supports A target configurations and its acquisition cost is divided equally, the end-to-end cost attributed to each target is

$$
\mathcal { C } _ { \mathrm { E 2 E } } ( \pmb { m } ) = 3 N _ { t } f _ { t } ( \pmb { m } ) + \frac { \mathcal { C } _ { \mathrm { c o n f i g } } ^ { ( s ) } } { A } .\tag{55}
$$

Normalizing by full-rotation target training at the same specification, $\mathcal { C } _ { \mathrm { b a s e } } ^ { ( t ) } = 3 N _ { t } f _ { t } ^ { \mathrm { f u l l } }$ , gives

$$
r _ { \mathrm { E 2 E } } = \frac { f _ { t } ( m ) } { f _ { t } ^ { \mathrm { f u l l } } } + \frac { 3 \tau N _ { s } f _ { s } + \mathcal { C } _ { \mathrm { p r o f } } ^ { ( s ) } } { A \cdot 3 N _ { t } f _ { t } ^ { \mathrm { f u l l } } } .\tag{56}
$$

The first term is determined by the target architecture, and the second by the profile source and reuse count. Fixed-carrier allocation preserves the principal matrix-multiplication count in the first term, while physical dimension allocation changes it through the target structure. Cross-scale transfer and profile reuse jointly reduce the second term.

If the same source profile supports A targets with diferent architectures, the total computation is

$$
\sum _ { j = 1 } ^ { A } \mathcal { C } _ { \mathrm { t r a i n } } ^ { ( t _ { j } ) } ( \boldsymbol { m } ^ { ( t _ { j } ) } ) + \mathcal { C } _ { \mathrm { c o n f i g } } ^ { ( s ) } .\tag{57}
$$

Under equal attribution across targets, the assigned source costs sum to the cost of acquiring the profile once.

# C Experimental Settings, Resource Accounting, and Additional Results

## C.1 Models, Training, and Profiling Settings

Models for positional-demand analysis. Section 3 analyzes DeepSeek-MoE-16B [7], LLaMA-2-7B [25], LLaMA-3.2-3B [17], and Qwen3-30B [29]. Phase-of profiling uses 500 shared text samples per model, and coverage calibration uses 100 samples per configuration. Within each measurement, reference and intervention configurations share input token sequences, attention masks, and loss-evaluation positions. Analysis figures use one-based layer numbers; frozen architecture sets in Table 9 use zero-based indices.

Models pretrained from scratch. Table 4 summarizes the model structures. Shared settings include sequence length 4,096, padded vocabulary size 128,896, tied input/output embeddings, RMSNorm, SwiGLU, BF16, and zero attention and hidden dropout. The RoPE base is 10,000, and RoPE fusion is disabled. In Table 4, $L , d , d _ { \mathrm { f f } } , h , d _ { c }$ denote depth, hidden width, FFN width, head count, and MLA KV latent width. Dense uses 64-dimensional queries, keys, and values per head. MLA uses content, full rotary, and value widths of 64 per head, with Q-LoRA disabled.

Optimization and execution. Distributed target training is based on Megatron-LM [23]. We use Adam with $\beta _ { 1 } = 0 . 9 , \ \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1, and gradient-clipping threshold 1. The learning rate follows a warmup–stable–decay schedule with cosine decay in the final phase. Table 5 reports the scale-specific settings; token counts equal steps times global batch size times sequence length.

Training primarily uses data parallelism, with tensor, pipeline, context, and expert parallel sizes all set to one. Random layer-set choices are distinguished from training-group replications. Within each comparison group, data order, training tokens, and optimization settings are matched. Dense execution uses Transformer Engine and FlashAttention [9].

Early score measurement. All profiles used for final architecture allocation are obtained at 10% of reference training. The source-side online monitor uses matched-marginal benefit measurement, with reference coverage 0.5 and low/high coverage 0.25/0.75. Let $\mathcal { L } _ { \ell , \mathrm { r e f } = 0 . 5 } ( x ; \rho )$ denote loss when layer ℓ uses coverage ρ and the remaining layers use reference coverage. A paired allocation score is

$$
a _ { \ell } = \frac { 1 } { | \mathscr { D } | } \sum _ { x \in \mathscr { D } } \Big [ \mathscr { L } _ { \ell , \mathrm { r e f = 0 . 5 } } ( x ; 0 . 2 5 ) - \mathscr { L } _ { \ell , \mathrm { r e f = 0 . 5 } } ( x ; 0 . 7 5 ) \Big ] .\tag{58}
$$

After standard validation, the monitor replays a raw batch already consumed by that evaluation. It records current, exponential-moving-average (EMA), and cumulative scores, using EMA coeficient 0.8. The score records and frozen layer set associated with the 10% checkpoint determine the subsequent architecture.

Dense interventions apply identity rotation to unselected frequency pairs. The fixed-width MLA measurement carrier zeros unselected positional coordinates; the target MLA model constructs rotary branches at the allocated physical widths. The binary allocations use the high\_to\_low atom order and retain the corresponding frequency indices.

## C.2 Resource Evaluation Settings and Normalization

Storage components. Model storage comprises target-model weights and the full KV cache. We use BF16, with two bytes per weight or cache element, inference batch size one, and cache length 4,096. Tied input/output embeddings are counted once. Total parameter counts use the approximate values in the experiment configuration summary; storage is computed from these counts and the layer-wise cache dimensions.

FLOPs convention. A multiplication and an addition each count as one FLOP. Accounting covers layer-wise linear projections, query/key dot products, attention-weighted value products, SwiGLU projections, and vocabulary-output matrix multiplication. The forward and corresponding backward matrix multiplications together count as three forward passes. Causal attention uses $T ( T + 1 ) / 2$ valid query–key position pairs.

Table 4: Pretraining architecture dimensions by scale; $d _ { c }$ is the MLA KV latent width.
<table><tr><td>Scale</td><td> $L$ </td><td> $d$ </td><td> $d _ { \mathrm { f } }$ </td><td> $h$ </td><td> $d _ { c }$ </td></tr><tr><td>130M</td><td>20</td><td>512</td><td>1,408</td><td>8</td><td>128</td></tr><tr><td>350M</td><td>24</td><td>896</td><td>2,432</td><td>14</td><td>256</td></tr><tr><td>1B, main</td><td>28</td><td>1,536</td><td>4,096</td><td>24</td><td>512</td></tr><tr><td>1B, KV384</td><td>28</td><td>1,536</td><td>4,096</td><td>24</td><td>384</td></tr></table>

Table 5: Pretraining hyperparameters by scale, including token budgets, batch sizes, and learning-rate schedules.
<table><tr><td>Setting</td><td>130M</td><td>350M</td><td>1B</td></tr><tr><td>Sequence length</td><td>4,096</td><td>4,096</td><td>4,096</td></tr><tr><td>GPUs per job</td><td>2</td><td>4</td><td>4 or 8</td></tr><tr><td>Micro-batch size per GPU</td><td>4</td><td>2</td><td>1</td></tr><tr><td>Global batch size</td><td>64</td><td>128</td><td>256</td></tr><tr><td>Tokens per step</td><td>262,144</td><td>524,288</td><td>1,048,576</td></tr><tr><td>Training steps</td><td>9,919</td><td>13,352</td><td>19,074</td></tr><tr><td>Training tokens</td><td>2,600,206,336</td><td> $7 , 0 0 0 , 2 9 3 , 3 7 6$ </td><td>20,000,538,624</td></tr><tr><td>Peak learning rate</td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Minimum learning rate</td><td> $6 \times 1 0 ^ { - 5 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup steps</td><td>198</td><td>267</td><td>382</td></tr><tr><td>Stable steps</td><td>8,729</td><td>11,750</td><td>16,785</td></tr><tr><td>Cosine-decay steps</td><td>992</td><td>1,335</td><td>1,907</td></tr></table>

Computation for demand-guided configurations includes 10% reference training and cumulative probing used to obtain the profile. The main tables use $A = 1$ , charging the complete source cost to one target.

Numerical profiling workload. The main tables use a specified theoretical probing workload of two events, each processing eight global sequences of length 4,096, giving

$$
N _ { \mathrm { p r o f } } = 2 \times 8 \times 4 0 9 6 = 6 5 { , } 5 3 6 .\tag{59}
$$

The event count and global sequence count are the current accounting assumptions; event-wise global counts are unspecified in the available run summary. Other cumulative workloads are substituted linearly into the formulas in Appendix B.1.

Resource ratios. Storage and computation are normalized to full rotation at the same model specification:

$$
r _ { S } = \frac { S ( m ) } { S ( m _ { \mathrm { f u l l } } ) } , \qquad r _ { C } = \frac { \mathcal { C } _ { \mathrm { E 2 E } } ( m ) } { 3 N _ { t } f _ { t } ( m _ { \mathrm { f u l l } } ) } .\tag{60}
$$

Each reference denominator is shared by configurations at the same scale, attention parameterization, and latent width. Resources for random allocations describe one training run with a given layer set.

## C.3 Parameter and Storage Counts for the Experimental Models

Parameter counts. Let d, d , h, V denote hidden width, FFN width, head count, and vocabulary size. The experimental models satisfy $d = 6 4 h$ . Tied embeddings contain Vd parameters, and the three SwiGLU projections contain $3 d d _ { \mathrm { f f } }$ matrix parameters per layer. Normalization and other auxiliary parameters are denoted by $P _ { \mathrm { a u x } }$

Dense parameters. Query, key, value, and output projections contain $4 d ^ { 2 }$ parameters per layer, giving

$$
P _ { D } = V d + L \left( 4 d ^ { 2 } + 3 d d _ { \mathrm { f } } \right) + P _ { \mathrm { a u x } } .\tag{61}
$$

Fixed-carrier coverage configurations share this projection structure.

MLA parameters. Content-query and output projections contribute $2 d ^ { 2 }$ parameters per layer. The KV down-projection and content-key/value up-projections contribute $3 d d _ { c }$ . Rotary-query and shared positionalkey projections at layer ℓ contain $d ( h + 1 ) d _ { \mathrm { r o p e } } ^ { ( \ell ) }$ parameters. Hence,

$$
P _ { M } ( m ) = V d + L \left( 2 d ^ { 2 } + 3 d d _ { c } + 3 d d _ { \mathrm { f f } } \right) + 2 d ( h + 1 ) B + P _ { \mathrm { a u x } } .\tag{62}
$$

Holding the remaining specifications fixed, a rotary-budget change $\Delta B$ changes the parameter count by $2 d ( h + 1 ) \Delta B$

Weights and full-layer cache. Dense attention caches d key and d value elements per layer and token:

$$
S _ { D } = 2 P _ { D } + 2 b _ { \mathrm { i n f } } T _ { \mathrm { c a c h e } } ( 2 L d ) .\tag{63}
$$

MLA caches the KV latent and shared positional key at each layer:

$$
S _ { M } ( \pmb { m } ) = 2 P _ { M } ( \pmb { m } ) + 2 b _ { \mathrm { i n f } } T _ { \mathrm { c a c h e } } \left( L d _ { c } + 2 B \right) .\tag{64}
$$

Equal total rotary budgets therefore give equal aggregate rotary-projection and positional-key cache sizes.

Table 6 combines the approximate parameter counts with Equations (63) and (64). Subscripts 100 and 50 denote full and half rotary budgets; storage is in MiB, with $1 \mathrm { M i B } = 2 ^ { 2 0 }$ bytes. At 350M MLA, for example, half-budget allocation reduces total parameters from approximately 348.1M to 337.8M and the full-layer cache from 7,680 to 6,912 elements per token. Together, these changes reduce model storage from approximately 723.95 MiB to 698.30 MiB, corresponding to 0.965×.

## C.4 Training and Configuration FLOPs for the Experimental Models

Linear layers and attention. For $X \in \mathbb { R } ^ { T \times a }$ and $W \in \mathbb { R } ^ { a \times b }$ , forward multiplication requires 2Tab FLOPs. Weight-gradient and input-gradient computations each contain a matrix multiplication with the same arithmetic count, giving 6T ab for training. At each valid causal position pair, a single-head query/key dot product requires $2 d _ { q k }$ FLOPs and the attention-weighted value product requires $2 d _ { v }$ . Dividing by sequence length gives the combined per-token forward cost $h ( T + 1 ) ( d _ { q k } + d _ { v } )$

Dense forward and training computation. Linear projections and the FFN cost $2 T ( 4 d ^ { 2 } + 3 d d _ { \mathrm { f f } } )$ per layer in the forward pass. Adding attention products across all layers and the vocabulary-output projection gives

$$
f _ { D } = 2 L \left( 4 d ^ { 2 } + 3 d d _ { \mathrm { f } } \right) + L h ( T + 1 ) ( 6 4 + 6 4 ) + 2 d V .\tag{65}
$$

Fixed-carrier allocation preserves these matrix dimensions.

MLA forward and training computation. Let $\begin{array} { r } { \bar { r } = L ^ { - 1 } \sum _ { \ell = 1 } ^ { L } d _ { \mathrm { r o p e } } ^ { ( \ell ) } } \end{array}$ denote average rotary width, so that $\bar { r } = 2 B / L$ . The average query/key matching width is $6 4 + { \bar { r } } ,$ while value width is 64. Thus,

$$
f _ { M } ( \bar { r } ) = 2 L \left( 2 d ^ { 2 } + 3 d d _ { c } + 3 d d _ { \mathrm { f f } } + d ( h + 1 ) \bar { r } \right) + L h ( T + 1 ) ( 1 2 8 + \bar { r } ) + 2 d V .\tag{66}
$$

Full, half, and three-quarter rotary budgets correspond to $\bar { r } = 6 4 , 3 2 , 4 8$ , respectively.

Full target-training cost is $3 N _ { t } f _ { t } ( m )$ . For half-budget MLA at the three scales, the target-training ratios are, respectively,

$$
0 . 9 3 0 1 9 3 , \qquad 0 . 9 3 2 6 4 2 , \qquad 0 . 9 3 7 6 2 2 .\tag{67}
$$

Reference-training and profiling substitutions. The 130M source model has 20 layers, so low/highcoverage measurement performs 41 forward passes per input batch. Under the cumulative workload in Appendix C.2, $( 2 L _ { s } + 1 ) N _ { \mathrm { p r o f } } = 4 1 \times 6 5 , 5 3 6 = 2 , 6 8 6 , 9 7 6$ . The cumulative probing costs are

$$
\mathcal { C } _ { \mathrm { p r o f } } ^ { D } = 0 . 9 2 5 2 5 1 \times 1 0 ^ { 1 5 } \ \mathrm { F L O P s } , \qquad \mathcal { C } _ { \mathrm { p r o f } } ^ { M } = 1 . 0 3 4 4 5 7 \times 1 0 ^ { 1 5 } \ \mathrm { F L O P s } .\tag{68}
$$

Table 6: Full- versus half-budget parameters (M) and BF16 storage (MiB; weights plus one 4K KV cache). $r _ { S } = S _ { 5 0 } / S _ { 1 0 0 }$
<table><tr><td>Model</td><td> $P _ { 1 0 0 }$ </td><td> $P _ { 5 0 }$ </td><td> $S _ { 1 0 0 }$ </td><td> $S _ { 5 0 }$ </td><td>rs</td></tr><tr><td>130M Dense</td><td>130.2M</td><td>130.2M</td><td>408.34</td><td>408.34</td><td>1.000</td></tr><tr><td>350M Dense</td><td>349.5M</td><td>349.5M</td><td>1,002.62</td><td>1,002.62</td><td>1.000</td></tr><tr><td>1B Dense</td><td>990.8M</td><td>990.8M</td><td>2,561.80</td><td>2,561.80</td><td>1.000</td></tr><tr><td>130M MLA</td><td>129.6M</td><td>126.6M</td><td>277.19</td><td>266.47</td><td>0.961</td></tr><tr><td>350M MLA</td><td>348.1M</td><td>337.8M</td><td>723.95</td><td>698.30</td><td>0.965</td></tr><tr><td>1B MLA</td><td>993.5M</td><td>959.1M</td><td>2,020.95</td><td>1,948.34</td><td>0.964</td></tr></table>

Table 7: Theoretical forward matrix FLOPs per token (GFLOPs) for Dense and full-/half-budget MLA.
<table><tr><td>Scale</td><td>Dense</td><td>MLA Full</td><td>MLA Half</td></tr><tr><td>130M</td><td>0.344347</td><td>0.384989</td><td>0.358114</td></tr><tr><td>350M</td><td>0.875112</td><td>0.960462</td><td>0.895767</td></tr><tr><td>1B</td><td>2.333823</td><td>2.515532</td><td>2.358617</td></tr></table>

Early reference-training costs use the source training-token counts in Table 5 and the 10% fraction. Changing the cumulative probing input gives

$$
\mathcal { C } _ { \mathrm { p r o f , n e w } } = \mathcal { C } _ { \mathrm { p r o f , t a b l e } } \frac { N _ { \mathrm { p r o f , n e w } } } { 6 5 , 5 3 6 } .\tag{69}
$$

End-to-end cost aggregation. Normalize target training, early reference training, and profiling separately by full-rotation target training:

$$
r _ { C } = r _ { \mathrm { t a r g e t } } + \frac { r _ { \mathrm { e a r l y } } + r _ { \mathrm { p r o f } } } { A } .\tag{70}
$$

Table 8 reports the resulting decomposition with A = 1 and the probing workload in Appendix C.2. The 130M references are same-scale; larger targets use a 130M source. Prespecified uniform configurations and given random layer sets proceed directly to target training, so their compute ratios use the target-training term. Reverse allocations generated from a demand profile include configuration acquisition in the same way.

Compute changes in the experimental cases. Target training for 350M MLA costs 0.932642× the full model, increasing to 0.947582× after adding early 130M reference training and profiling. The corresponding end-to-end ratio for 1B MLA is 0.939618×. Dense target-matrix arithmetic remains unchanged, while the relative configuration-acquisition cost decreases with target scale.

Configuration reuse. When one source profile serves A targets, the 350M MLA cost becomes

$$
r _ { C } ( A ) = 0 . 9 3 2 6 4 2 + \frac { 0 . 0 1 4 8 8 9 + 0 . 0 0 0 0 5 1 } { A } .\tag{71}
$$

The same reuse rule applies to Dense and MLA using their respective source profiles.

Comparison with target-direct profiling. Under the same cumulative probing input and the relations in Appendix B, early reference training for the 130M Dense source costs approximately 0.1462× and 0.0192× that of direct reference acquisition at 350M and 1B, respectively; probing costs are 0.3292× and 0.1061×. The corresponding MLA early-reference ratios are 0.1489× and 0.0199×, with probing ratios 0.3354× and 0.1101×. These values quantify the reductions in both components of configuration acquisition obtained from a smaller model.

Table 8: Theoretical end-to-end FLOPs ratios to Full for half-budget allocation: target training, reference training, and probing.
<table><tr><td>Target</td><td>Train</td><td>Reference</td><td>Probe</td><td>Total</td></tr><tr><td>130M Dense</td><td>1.000000</td><td>0.100000</td><td>0.000344</td><td>1.1003</td></tr><tr><td>350M Dense</td><td>1.000000</td><td>0.014616</td><td>0.000050</td><td>1.0147</td></tr><tr><td>1B Dense</td><td>1.000000</td><td>0.001918</td><td>0.000007</td><td>1.0019</td></tr><tr><td>130M MLA</td><td>0.930193</td><td>0.100000</td><td>0.000344</td><td>1.0305</td></tr><tr><td>350M MLA</td><td>0.932642</td><td>0.014889</td><td>0.000051</td><td>0.9476</td></tr><tr><td>1B MLA</td><td>0.937622</td><td>0.001990</td><td>0.000007</td><td>0.9396</td></tr></table>

## C.5 Frozen Layer Sets and Layer-Selection Ablations

Configuration names. Profile50 and Profile75 use same-scale allocation scores at 50% and 75% rotary budgets. Transfer50 uses a within-family source profile, and Target-direct50 uses the target’s early profile. These are instances of Demand-guided in the main tables. Reverse50 takes the strict complement of Profile50, while Last10 selects the final ten layers. Where multiple Random50 sets are evaluated, they represent distinct layer allocations rather than independent training replications.

Layer-wise configurations. Table 9 lists the frozen high-coverage sets. Selected layers use rotary width 64; other layers use identity rotation in Dense or zero rotary width in MLA. The same 350M MLA transferred set is used in the additional paired training runs.

Layer placement in 130M MLA. Table 10 compares demand-guided, random, reversed, and contiguous late-layer allocations. The three random sets have mean loss 3.412386. Demand-guided allocation improves over this mean by 0.030679, over reverse ranking by 0.037628, and over the final-ten-layer configuration by 0.087689.

Random controls in 350M MLA. The two random sets in Table 11 have mean loss 3.062779. Cross-scale allocation improves over the random mean by 0.011339.

## C.6 Rotary Budget and Quality–Resource Trade-ofs

We vary the number of high-coverage layers in 130M MLA along the same demand ranking, comparing half, three-quarter, and full rotary budgets. Profile50 reaches loss 3.381707, outperforming Profile75 at 3.390307 by 0.008600, while approaching the full-rotation loss of 3.377954.

Equations (62) and (66) make parameters and forward matrix-multiplication computation linear in average rotary width. The three-quarter budget has r¯ = 48, giving a target-training FLOPs ratio of 0.965097. Interpolating the approximate parameter counts in Table 6 gives combined weight and single-sequence 4K cache storage of approximately 271.83 MiB, or 0.981× the full model. Including same-scale early reference training and the specified probing workload yields an end-to-end compute ratio of 1.0654×.

Demand ranking supports allocations at multiple budget levels, while the physical budget determines storage and target-training computation. In the evaluated 130M runs, the quality–budget response is non-monotone, with Profile50 ofering a favorable trade-of.

## C.7 Cross-Scale and Paired Training

Paired training runs. Table 14 reports the mean of two paired comparisons for 350M MLA. Transfer50 reduces validation loss by 0.011348 on average relative to Uniform50. The reported ∆ is computed within each matched training group before averaging.

Target-direct profiling. At 1B Dense, we also construct an allocation using the target model’s own profile at 10% of reference training and compare it with uniform coverage from the same training group. Target-direct allocation improves loss by 0.013850. Its high-coverage layers are 6, 12–24, sharing eleven layers with the transferred set and giving a set Jaccard score of 11/17. Layers exclusive to source transfer are 10, 25, 26, while those exclusive to target-direct allocation are 6, 12, 13.

Table 9: High-coverage layer sets for each allocation; indices are zero-based and ranges are inclusive.
<table><tr><td>Model / allocation</td><td>Layers</td></tr><tr><td>130M Dense, Profile50</td><td>7, 10-18</td></tr><tr><td>130M MLA, Profile50</td><td>0-2, 12–18</td></tr><tr><td>130M MLA, Profile75</td><td>0–5, 8, 12–19</td></tr><tr><td>130M MLA, Reverse50</td><td>3-11, 19</td></tr><tr><td>130M MLA, Last10</td><td>10-19</td></tr><tr><td>350M Dense, Transfer50</td><td>8, 12–22</td></tr><tr><td>350M MLA, Transfer50</td><td>0-3, 15–22</td></tr><tr><td>1B Dense, Transfer50</td><td>10, 14–26</td></tr><tr><td>1B MLA, Transfer50</td><td>0-3, 17-26</td></tr><tr><td>1B Dense, Target-direct50</td><td>6, 12–24</td></tr></table>

Table 10: 130M MLA allocation ablation at 50% budget: validation loss for profile-guided, random, reversed, uniform, and last-ten-layer strategies. Random50 is averaged over three random layer sets.
<table><tr><td>Configuration</td><td>Validation loss</td></tr><tr><td>Profile50</td><td>3.381707</td></tr><tr><td>Random50</td><td>3.412386</td></tr><tr><td>Reverse50</td><td>3.419335</td></tr><tr><td>Uniform50</td><td>3.425247</td></tr><tr><td>Last10</td><td>3.469396</td></tr></table>

Table 11: 350M MLA validation loss at 50% budget: transferred profiles versus random layer sets and uniform coverage. Random50 is averaged over two random layer sets.
<table><tr><td>Configuration</td><td>Validation loss</td></tr><tr><td>Transfer50</td><td>3.051440</td></tr><tr><td>Random50</td><td>3.062779</td></tr><tr><td>Uniform50</td><td>3.067263</td></tr></table>

Table 12: 130M MLA rotary-budget ablation: validation loss at 50%, 75%, and 100% budgets, including Uniform50.
<table><tr><td>Configuration</td><td>Budget</td><td>Validation loss</td></tr><tr><td>Full</td><td>100%</td><td>3.377954</td></tr><tr><td>Profile75</td><td>75%</td><td>3.390307</td></tr><tr><td>Profile50</td><td>50%</td><td>3.381707</td></tr><tr><td>Uniform50</td><td>50%</td><td>3.425247</td></tr></table>

Source transfer and target-direct measurement both supply architecture-selection signals: the former reuses small-model structure, and the latter uses the target’s early layer-wise distribution. Their acquisition costs follow the separate accounting in Appendices B and C.4.

Table 13: 1B MLA validation loss with KV latent width 384: Full versus Uniform50.
<table><tr><td>Configuration</td><td>Validation loss</td></tr><tr><td>Full</td><td>2.731081</td></tr><tr><td>Uniform50</td><td>2.745739</td></tr></table>

![](images/0a0615008880fd943f4881e447e37f3aa1699bc89edb3180202a052ae89b7f3e.jpg)  
Figure 3: Recovered validation-loss reduction across rotary coverage for LLaMA-2-7B layers 1 and 2 (onebased). Both curves use the same calibration texts.

Table 14: 350M MLA validation loss averaged over two paired training runs. $\Delta$ is the mean within-pair diference (Uniform50 minus Transfer50).
<table><tr><td>Statistic</td><td>Uniform50</td><td>Transfer50</td><td>∆ Validation loss</td></tr><tr><td>Mean</td><td>3.060510</td><td>3.049162</td><td>0.011348</td></tr></table>

Table 15: 1B Dense validation loss with target-direct profiling versus Uniform50 at 50% budget.
<table><tr><td>Configuration</td><td>Validation loss</td></tr><tr><td>Uniform50</td><td>2.735777</td></tr><tr><td>Target-direct50</td><td>2.721927</td></tr></table>

## C.8 Additional Experiments with a Narrower KV Latent

This group uses a 1B MLA model with KV latent width 384. The remaining structure has 28 layers, hidden width 1,536, FFN width 4,096, and 24 heads, with the 1B training settings in Table 5.

Uniformly reducing rotary width increases loss by 0.014658. The full-layer logical cache decreases from 28(384+ 64) = 12,544 to 28(384 + 32) = 11,648 elements per token. Substituting this latent width into Equations (62),

(64), and (66) gives full-model storage and training computation. This experiment complements the rotarybudget–quality analysis with a narrower latent specification.

## C.9 Representative Coverage Responses

Figure 3 compares LLaMA-2-7B layers 1 and 2, as discussed in Section 3.3. The ordinate is the recovered utility $\widehat { G } _ { \ell } ( 0 , \bar { \rho } )$ , divided by each layer’s phase-of loss on the same 100 calibration samples. At half coverage, layer 1 recovers approximately 97.0% of its full-rotation utility, whereas layer 2 recovers 3.1%; the latter rises to 99.3% at three-quarter coverage.