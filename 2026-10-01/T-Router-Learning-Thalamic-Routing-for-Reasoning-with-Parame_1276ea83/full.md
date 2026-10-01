# T-Router: Learning Thalamic Routing for Reasoning with Parameter-Efficient Reinforcement Learning

Liuxian Ma<sup>1,\*,#</sup>, Jiale Dai<sup>2,\*</sup>, Jiaqi Li<sup>3</sup>, Lu Mi<sup>1,</sup>†<sup>,#</sup>

<sup>1</sup>College of Artificial Intelligence, Tsinghua University

<sup>2</sup>State Key Laboratory of General Artificial Intelligence, School of Intelligence Science and Technology, Peking University <sup>3</sup>Beijing Institute for General Artificial Intelligence

<sup>\*</sup>Strictly equal contribution; either author order is equally valid.

<sup>†</sup>Corresponding author.

Parameter-eficient reinforcement learning aims to improve reasoning with a compact trainable interface to a pretrained model. We introduce the Thalamic Router (T-Router), which concentrates adaptation on the reuse of completed computations. A compressed, addressable bank preserves block changes; a depth-recurrent controller conditions their selection and relative-scale writeback. This coupling gives thalamic context-dependent routing a concrete computational form: learn which earlier contributions a receiving layer uses, and with what influence. Correctness rewards train the interface while preserving backbone parameters and layer order. On an 8.95B-parameter backbone, T-Router allocates 41.73M parameters—0.466% of the backbone—and achieves 83.64 1.16 MathAvg after GSM8K RL, compared with 73.79 1.83 for full-parameter GRPO across three evaluation rounds. At a comparable parameter budget and with matched retries, it exceeds LoRA’s 77.28 1.95 MathAvg, improving all three task families and raising mean AIME accuracy from 48.33 to 60.56. Capacity-controlled comparisons favor addressable block changes and recurrent context; separate search training extends the interface to tool-mediated reasoning. These results establish controlled computation reuse as an efective route to parameter-eficient reasoning reinforcement learning.

Keywords: parameter-eficient reinforcement learning, large language models, mathematical reasoning, crosslayer routing, computation reuse, thalamic routing

Contact: # maliuxian03@gmail.com # milu@mail.tsinghua.edu.cn

## 1 Introduction

Reinforcement learning (RL) improves mathematical reasoning by rewarding successful solutions (Shao et al., 2024). Making this adaptation eficient requires placing trainable capacity where reward feedback can improve reasoning while preserving the bulk of pretrained parameters (Sidahmed et al., 2024). A model already contains useful transformations; an additional opportunity is to learn how their intermediate contributions are organized. Our starting point is therefore adaptation eficiency: train a compact Thalamic Router (T-Router) for controlled reuse of completed computations around a frozen backbone.

LoRA and adapters place trainable capacity in weight updates or local transformations (Hu et al., 2022; Houlsby et al., 2019). In the resulting residual stream, earlier block contributions continue to mix with subsequent updates. Keeping these contributions separately addressable exposes an additional learning target: which completed changes a later computation reuses, and with what influence. Learned depth connectivity already develops this direction through attention to prior computation, additive delta reuse, and frozen-backbone residual routing (Kimi Team et al., 2026; Luo et al., 2026a; Oldenburg et al., 2026). The question for a compact RL interface is how to couple reusable content, the receiving computation’s context, and writeback strength.

Thalamic circuits regulate cortical communication according to task demands. The pulvinar coordinates information transmission between cortical areas through attention-dependent synchronization (Saalmann et al., 2012); mediodorsal thalamic input amplifies functional prefrontal connectivity to sustain task representations (Schmitt et al., 2017). We take these findings as a routing principle: use context to select information sources and regulate their influence on a receiving computation. In T-Router, frozen decoder blocks perform computation, while the complete auxiliary pathway controls which completed block changes a later layer reuses and at what strength. Figure 1 makes this correspondence explicit.

![](images/43b22c96a849e4736703e1e46b36c0ca460c84fe24212b69c2a7c6b2a534f2f1.jpg)  
Figure 1 : From adapting transformations to learning their communication. Weight adaptation changes transformations (top). Thalamic circuits regulate cortical communication according to task context (middle). T-Router applies this routing principle across depth: select addressable block changes and regulate their influence before a later frozen layer executes (bottom).

T-Router preserves compressed block changes in a source bank alongside the evolving residual. Later layers can recombine these records according to their current state and accumulated depth context. Controller slots S accumulate that history; attention queried by the current hidden state reads a context vector P from the updated slots. Together with the hidden state, P conditions both source attention and a signed gate. Selected content is projected into the receiving residual and added before the frozen layer executes. The bank defines what is available; the router and gate learn which mixture to reuse and at what strength.

Correctness rewards train this communication interface end to end. Its defining feature is the coupling of addressable content, recurrent control, and relative influence: the bank preserves reusable changes, depth context guides their selection, and a signed gate regulates the resulting writeback. RMS calibration expresses the gate’s strength relative to the receiving residual. Every pretrained layer still executes in its original order. RL can therefore optimize how completed contributions serve a later computation while preserving the backbone’s pretrained transformations.

The mathematics comparison demonstrates this eficiency at two adaptation budgets. With 41.73M parameters, 0.466% of the 8.95B backbone, T-Router reaches 83.64 1.16 MathAvg versus 73.79 1.83 for full-parameter GRPO across three evaluation rounds. At a compact budget, matched-retry LoRA uses 43.28M parameters and reaches 77.28 1.95. T-Router improves all three mathematical families, including 92.73 versus 87.93 on MATH-500 and 60.56 versus 48.33 mean AIME accuracy after GSM8K training. Capacity-controlled comparisons favor the full interface over hidden-state memory (67.46 MathAvg) and a feedforward controller replacement (72.80). Separate search training applies the same compact interface to tool-mediated reasoning.

Our contributions are: (i) a parameter-eficient RL interface coupling compressed, addressable block changes to independent depth-recurrent control and relative-scale writeback; (ii) stronger mathematical reasoning than full-parameter GRPO with 0.466% of the backbone’s parameter count, and than matched-retry LoRA at a comparable budget, with an application to agentic search; and (iii) capacity-controlled comparisons and a formal characterization of the content and control pathways. Together they establish controlled computation reuse as a concrete route to eficient RL adaptation.

## 2 Related work

Parameter-eficient reinforcement learning. Adapters and LoRA reduce the trainable footprint of task adaptation (Houlsby et al., 2019; Hu et al., 2022); PE-RLHF applies parameter-eficient learning to reward modeling and reinforcement learning (Sidahmed et al., 2024). GRPO uses group-relative rewards (Shao et al., 2024); Section 3 identifies the single-update policy gradient used to train the module. RO-GRPO shapes rewards using routing statistics in LoRA mixtures (Ma et al., 2026), while S-GRPO samples output-token positions for the training loss (Lee & Tong, 2025). DAPO oversamples rollout groups and filters all-correct or all-incorrect groups (Yu et al., 2025); T-Router uses a bounded retry rule, analyzed in Appendix B. T-Router retains input-dependent cross-layer communication at inference; Appendix A details the parameter and execution costs.

Learning communication across depth. DenseNet enables feature reuse through dense concatenation of preceding feature maps (Huang et al., 2017). DenseFormer learns depth-weighted combinations (Pagliardini et al., 2024); MUDDFormer makes cross-layer connections dynamic and stream-specific (Xiao et al., 2025). Attention Residuals introduces attention over earlier computation, including a block variant (Kimi Team et al., 2026). Delta Attention Residuals adds attention-weighted sublayer or block changes to the residual (Luo et al., 2026a), providing the closest source-reuse precedent. Multi-Head Attention Residuals gives feature subspaces separate distributions over depth (Luo et al., 2026b). Hyper-connection constraints and adaptive reassignment further expand depth connectivity (Xie et al., 2025; Zhang et al., 2026). In the frozen-backbone setting, mHC-based PEFT learns read/write routing over multiple residual streams and investigates learned versus identity residual mixing (Oldenburg et al., 2026). These methods establish depth connectivity as an adaptation axis alongside local weight updates.

T-Router targets parameter-eficient RL through a coupled content-and-control interface: compressed, originindexed changes supply content; a separate depth-recurrent controller supplies routing context; and calibrated signed writeback supplies relative influence. A current-state query reads P from slots S to condition both selection and strength. Correctness rewards train this compact interface while preserving pretrained transformations and layer order. Appendix D develops these computational distinctions.

Thalamic regulation of cortical communication. The pulvinar coordinates cortical information transmission through attention-dependent synchronization (Saalmann et al., 2012). Mediodorsal thalamic input amplifies functional prefrontal connectivity, sustaining rule representations (Schmitt et al., 2017). These mechanisms motivate context-dependent control over the contributions of distributed computations. T-Router realizes this principle through source selection and writeback strength, conditioned on the receiving residual and recurrent depth context. HippoRAG draws on hippocampal indexing theory for external knowledge retrieval (Gutiérrez et al., 2024).

## 3 T-Router

T-Router concentrates trainable capacity in a compact communication path around a frozen decoder. A source bank stores compressed block changes, a recurrent controller summarizes depth history, and a query-dependent readout conditions source attention and writeback strength. The router and gate combine this signal with the current residual to form an intervention. All pretrained layers execute in order; correctness rewards train how earlier contributions are reused.

![](images/8714027908761ce5d47399b491c4292f8f7dce99ad23d8dbf3e0404666060e06.jpg)  
Figure 2 : How T-Router reuses completed computation. Completed blocks supply compressed changes to the source bank. The depth controller updates its state S, then reads routing context $P$ using the current hidden state. This context conditions which sources are mixed and how strongly the projected mixture is written back. The contribution is added before the receiving frozen layer; all layers execute in order. Teal and gold paths carry reusable content; dashed purple branches condition selection and strength. Equations 5–7 specify the operations.

## 3.1 Recording changes across depth

Let $H _ { \ell } \in \mathbb { R } ^ { B \times T \times d }$ be the residual tensor before layer ℓ, for batch size B and sequence length T. With the batch index suppressed, $h _ { \ell , t } \in \mathbb { R } ^ { d }$ denotes its token-t vector; $R _ { \ell }$ stacks the token interventions $R _ { \ell , i }$ . The complete frozen decoder layer $B _ { \ell }$ acts on the sequence tensor, including its causal token interactions and residual connections:

$$
\begin{array} { r } { \widetilde { H } _ { \ell } = H _ { \ell } + R _ { \ell } , \qquad H _ { \ell + 1 } = B _ { \ell } ( \widetilde { H } _ { \ell } ) . } \end{array}\tag{1}
$$

Writing $B _ { \ell } ( X ) = X + F _ { \ell } ( X )$ gives $H _ { \ell + 1 } = H _ { \ell } + R _ { \ell } + F _ { \ell } ( H _ { \ell } + R _ { \ell } )$ : the intervention enters both the residual connection and the frozen transformation. Training updates only the auxiliary module.

We group consecutive layers into blocks of size s. Upon completing block $b ,$ T-Router compresses the change between its first actual input and final output:

$$
D _ { b , t } = h _ { s ( b + 1 ) , t } - \widetilde { h } _ { s b , t } , \qquad m _ { b , t } = C _ { b } D _ { b , t } \in \mathbb { R } ^ { r } .\tag{2}
$$

Here $C _ { b } \in \mathbb { R } ^ { r \times d }$ is learned. This change is measured along the intervened forward pass: writebacks inside the block can contribute to $D _ { b }$ . Before layer $\ell ,$ only completed blocks $\mathcal { T } _ { \ell } = \{ b \in \mathbb { N } _ { 0 } : s ( b + 1 ) \leq \ell \}$ are available. Each record pairs $m _ { b , t }$ with a learned source embedding $e _ { b } ^ { \mathrm { s r c } }$ . The bank stores changes at the same token position; it introduces no new attention over token positions.

We use $L = 3 2 , s = 4$ , and $r = 2 5 6$ . The first four layers receive zero writeback; later layers can read up to seven completed blocks from the same forward pass.

## 3.2 From depth state to controller context

The controller summarizes the token’s computation across layers to condition source selection and writeback strength. Its state comprises K slots $S _ { \ell , t , k } \in \mathbb { R } ^ { p }$ , initialized as $S _ { - 1 , t , k } = S _ { k } ^ { 0 }$ from a learned array. At every layer, the update combines the current residual, the mean $\bar { m } _ { \ell , t }$ of completed records (zero for an empty bank), and the receiving-layer embedding $e _ { \ell } \colon$

$$
z _ { \ell , t } = [ A _ { h } h _ { \ell , t } ; A _ { m } { \bar { m } } _ { \ell , t } ; e _ { \ell } ] , \qquad [ u _ { \ell , t } ; v _ { \ell , t } ] = \mathrm { M L P } ( z _ { \ell , t } ) .\tag{3}
$$

Write attention ω distributes the shared proposal across the previous slots:

$$
\begin{array} { r l } & { \omega _ { \ell , t , k } = \mathrm { s o f t m a x } _ { k } \left( S _ { \ell - 1 , t , k } ^ { \top } W _ { s } z _ { \ell , t } / \sqrt { p } \right) , } \\ & { S _ { \ell , t , k } = \gamma S _ { \ell - 1 , t , k } + ( 1 - \gamma ) \omega _ { \ell , t , k } \left[ \sigma ( v _ { \ell , t } ) \odot \mathrm { t a n h } ( u _ { \ell , t } ) \right] . } \end{array}\tag{4}
$$

We define controller context $P _ { \ell , t } \in \mathbb { R } ^ { p }$ as an attention readout of these updated slots. The current residual supplies the query $Q _ { s } h _ { \ell , t } ;$ slot projections supply keys $K _ { s } S _ { \ell , t , k }$ and values $V _ { s } S _ { \ell , t , k } \colon$

$$
\begin{array} { r } { \xi _ { \ell , t , k } = \mathrm { s o f t m a x } _ { k } \mathopen { } \mathclose \bgroup \left( ( Q _ { s } h _ { \ell , t } ) ^ { \top } K _ { s } S _ { \ell , t , k } / \sqrt { p } \aftergroup \egroup \right) , \qquad P _ { \ell , t } = \displaystyle \sum _ { k = 1 } ^ { K } \xi _ { \ell , t , k } V _ { s } S _ { \ell , t , k } . } \end{array}\tag{5}
$$

Here ξ is normalized over the K slots. S is the recurrent state; P is its current control signal.

The router and gate combine P with h to determine source weights (Equation 6) and writeback strength (Equation 7). Source attention supplies the transported content c from the bank. We use $K = 8 , p = 2 5 6$ and $\gamma = 0 . 9 ;$ ; Appendix B.4 expands the history represented by S and $P .$

The slots and bank reset at each forward pass and maintain separate depth trajectories for each token. Information from preceding tokens reaches them through the backbone’s causal representations and, during cached decoding, its cache.

## 3.3 Routing completed computations

For every layer with a nonempty bank, routing combines the current state, controller context, and layer identity:

$$
\begin{array} { r l r } { q _ { \ell , t } = W _ { q } [ h _ { \ell , t } ; P _ { \ell , t } ; e _ { \ell } ] , } & { k _ { b , t } = W _ { k } [ m _ { b , t } ; e _ { b } ^ { \mathrm { s r c } } ] , } \\ { a _ { \ell , t , b } = \mathrm { s o f t m a x } _ { b \in \mathcal { T } _ { \ell } } \left( q _ { \ell , t } ^ { \top } k _ { b , t } / \sqrt { r } \right) , } & { c _ { \ell , t } = \displaystyle \sum _ { b \in \mathcal { T } _ { \ell } } a _ { \ell , t , b } W _ { v } [ m _ { b , t } ; e _ { b } ^ { \mathrm { s r c } } ] . } \end{array}\tag{6}
$$

Dense softmax mixes the visible sources in each forward pass. Source and layer embeddings identify origin and destination; the hidden state and controller context determine the query. Learned $C _ { b } , W _ { v } .$ , and $U _ { \ell }$ determine how the selected content reaches the frozen layer. At a fixed receiving layer, all interventions lie in the at-most-r-dimensional column space of $U _ { \ell }$ before dtype rounding, even after calibration and scalar gating. Equation 24 gives the aggregate transport form.

## 3.4 Calibrating the intervention

A gate on an unnormalized projection does not directly specify the size of its efect: the same scalar can multiply writeback directions with very diferent norms. We first map the retrieved source mixture to $w _ { \ell , t } = U _ { \ell } c _ { \ell , t } \in \mathbb { R } ^ { d }$ and calibrate its RMS to the receiving state. Define $\begin{array} { r } { \mathrm { R M S } ( x ) = \sqrt { d ^ { - 1 } \sum _ { j } x _ { j } ^ { 2 } } } \end{array}$ and stop-gradient ${ \mathrm { s g } } .$ . The implemented update is

$$
\begin{array} { r l } & { \widehat { w } _ { \ell , t } = \left\{ \begin{array} { l l } { w _ { \ell , t } \frac { \mathrm { S g } \left( \mathrm { R M S } ( h _ { \ell , t } ) \right) } { \mathrm { R M S } ( w _ { \ell , t } ) } , } & { \mathrm { R M S } ( w _ { \ell , t } ) > \epsilon , } \\ { w _ { \ell , t } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \\ & { g _ { \ell , t } = g _ { \operatorname* { m a x } } \operatorname { t a n h } \left( b _ { \ell } + w _ { g } ^ { \top } [ h _ { \ell , t } ; P _ { \ell , t } ] \right) , \qquad R _ { \ell , t } = g _ { \ell , t } \widehat { w } _ { \ell , t } . } \end{array}\tag{7}
$$

We use $\epsilon = 1 0 ^ { - 6 }$ and $g _ { \mathrm { m a x } } = 0 . 0 5$ . The signed gate can reinforce or oppose the retrieved direction. Orthogonal initialization of $U _ { \ell } .$ , zero initialization of $w _ { g }$ , and $b _ { \ell } = \mathrm { a r c t a n h } ( 0 . 4 )$ give an initial calibrated magnitude of 2%. When $\mathrm { R M S } ( w _ { \ell , t } ) > \epsilon$ and $h _ { \ell , t } \neq 0$ , calibration gives the local scale identity

$$
\rho _ { \ell , t } : = \frac { \| R _ { \ell , t } \| _ { 2 } } { \| h _ { \ell , t } \| _ { 2 } } = | g _ { \ell , t } | < g _ { \operatorname* { m a x } } .\tag{8}
$$

The gate specifies the local relative intervention size independently of the raw projection scale. Subsequent nonlinear computation can amplify or attenuate its efect. The small-norm fallback is part of the implementation; Appendix B.5 derives the active-branch Jacobian and characterizes its threshold behavior.

## 3.5 Training the auxiliary module

For mathematics, the GRPO trainer implements a single-evaluation group-relative policy gradient with a frozen-base reference. For each prompt, it samples four completions and retries a correctness-uniform group at most three additional times, retaining the first mixed-correctness group or the final attempt. Binary parsed-correctness rewards are centered and standardized within valid completions to form $A _ { i } ;$ responses that are both truncated and unparseable are omitted from this preference term.

The policy is the adapted language model’s completion distribution. A sequence-normalized group-relative gradient trains source transport and control end to end, with one diferentiable evaluation per retained group. Appendix B derives the objective and its credit-assignment paths. We optimize

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p o l i c y } } + \beta \mathcal { L } _ { \mathrm { K L } } + \lambda \Omega , \qquad \beta = 0 . 0 2 , \quad \lambda = 0 . 0 1 . } \end{array}\tag{9}
$$

Here ${ \mathcal { L } } _ { \mathrm { K L } }$ is the sampled token-level divergence surrogate to the backbone, and Ω penalizes routing nonuniformity and source-mixture magnitude using the full-tensor reduction. Appendix B specifies the exact surrogate, masks, and sampling distribution. Resampling changes the generated-data budget independently of the adapter. The module has 40,681,724 loss-connected parameters (0.454% of the backbone); its allocation is 41,730,332 (0.466%), including an unused terminal compressor and source-embedding row.

## 4 Experiments

## 4.1 Evaluation setting

We evaluate reasoning quality, trainable allocation, and the organization of that capacity. The backbone is Qwen3.5-9B-Base (Qwen Team, 2026), with 8,953,803,264 parameters. T-Router uses 32 layers, four-layer blocks, rank 256, and eight controller slots of dimension 256. It allocates 41.730M trainable parameters (0.466%); 40.682M are connected to the loss.

All trained methods in the mathematics comparison use the full GSM8K training set of 7,473 prompts. Evaluation covers all 1,319 GSM8K test questions (Cobbe et al., 2021), the 500-question MATH-500 split (Hendrycks et al., 2021; Lightman et al., 2024), and 30 questions from each of AIME 2024 and 2025. We report means sample standard deviations (SD) across three evaluation rounds with seeds 42, 43, and 44. Within each round, the equal-family score is

$$
\mathrm { M a t h A v g } = { \frac { \mathrm { G S M 8 K } + \mathrm { M A T H } { - } 5 0 0 + { \frac { 1 } { 2 } } ( \mathrm { A I M E } 2 4 + \mathrm { A I M E } 2 5 ) } { 3 } } .\tag{10}
$$

Aggregate SD is computed across round-level aggregates; paired diferences match round indices before computing their mean and SD. Baselines include the frozen model, full-parameter GRPO, RFT, LoRA at ranks 2 and 16, LoRA-MoE with RO-GRPO, LoRA with S-GRPO, and rank-16 LoRA with matched informative retries. The last compares weight adaptation and controlled reuse under the same retry rule at a similar trainable scale. Appendix A details scoring and aggregation.

## 4.2 Parameter-efficient mathematical reasoning

Higher accuracy with 0.466% of the backbone’s parameters. T-Router reaches $8 3 . 6 4 \pm 1 . 1 6$ MathAvg with 41.73M trainable parameters, versus $7 3 . 7 9 \pm 1 . 8 3$ for full-parameter GRPO over all 8.95B parameters (Table 1). The paired improvement is $9 . 8 5 \pm 2 . 9 3$ points. The compact interface leads all three mathematical families, combining 97.62 on GSM8K with 92.73 on MATH-500 and 60.56 mean AIME accuracy. These results link parameter eficiency to reasoning performance beyond the GSM8K training domain.

Stronger generalization at a comparable adaptation budget. Matched-retry LoRA uses 43.28M parameters and reaches 77.28 1.95 MathAvg. T-Router improves the mean by 6.36 points with 41.73M parameters. Its advantage spans MATH-500 (+4.80), GSM8K (+2.04), and AIME (about +12.2). The gains on broader and competition mathematics show that learning reuse supports generalization across mathematical task families. Figure 3a,b,d relates this improvement to trainable allocation under the matched retry rule.

Table 1 : Mathematical reasoning after GSM8K RL. Accuracy (%), $\mathrm { m e a n } \pm \mathrm { S D }$ across three evaluation seeds. AIME averages the two annual scores. LoRA-r16 (retries) matches T-Router's retry rule. Parameters are in millions; a dash denotes no update. Bold/underlined means are best/second-best, including ties; fewer trainable parameters are preferred.
<table><tr><td>Method</td><td>Params (M)</td><td></td><td>GSM8K MATH-500 AIME mean</td><td></td><td>MathAvg</td></tr><tr><td>Frozen base</td><td></td><td> $8 8 . 7 3 \pm 0 . 9 5$ </td><td> $7 5 . 6 0 \pm 2 . 2 5$ </td><td> $3 1 . 6 7 \pm 3 . 3 3$ </td><td> $6 5 . 3 3 \pm 0 . 8 9$ </td></tr><tr><td>Full-parameter GRPO</td><td>8954.000</td><td> $9 3 . 8 6 \pm 0 . 2 6$ </td><td> $8 3 . 0 7 \pm 0 . 7 0$ </td><td> $4 4 . 4 4 \pm 5 . 3 6$ </td><td> $7 3 . 7 9 \pm 1 . 8 3 $ </td></tr><tr><td>RFT</td><td>8954.000</td><td> $9 1 . 8 6 \pm 1 . 1 6$ </td><td> $8 2 . 9 3 \pm 2 . 4 7$ </td><td> $4 6 . 1 1 \pm 2 . 5 5$ </td><td> $7 3 . 6 4 \pm 1 . 3 3 $ </td></tr><tr><td> $\mathrm { L o R A - r 2 } + \mathrm { G R P O }$ </td><td>5.410</td><td> $9 3 . 5 1 \pm 0 . 2 3 $ </td><td> $8 4 . 5 3 \pm 0 . 8 1$ </td><td> $4 2 . 2 2 \pm 1 2 . 7 3$ </td><td> $7 3 . 4 2 \pm 4 . 2 8$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td>43.278</td><td> $9 3 . 8 1 \pm 0 . 7 7$ </td><td> $8 5 . 1 3 \pm 2 . 5 3$ </td><td> $4 5 . 5 6 \pm 6 . 3 1$ </td><td> $7 4 . 8 3 \pm 1 . 4 1$ </td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td>173.112</td><td> $9 5 . 3 0 \pm 0 . 6 1$ </td><td> $8 7 . 4 7 \pm 1 . 4 5$ </td><td> $4 4 . 4 4 \pm 2 . 5 5$ </td><td> $7 5 . 7 4 \pm 1 . 4 7$ </td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td>43.278</td><td> $9 4 . 6 4 \pm 0 . 4 6 $ </td><td> $8 4 . 3 3 \pm 1 . 8 6 $ </td><td> $4 8 . 8 9 \pm 8 . 2 2$ </td><td> $7 5 . 9 5 \pm 2 . 4 3$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O } ,$  retries</td><td>43.278</td><td> $9 5 . 5 8 \pm 0 . 5 0$ </td><td> $\underline { { 8 7 . 9 3 } } \pm \mathbf { 0 . 1 2 }$ </td><td> $4 8 . 3 3 \pm 6 . 0 1$ </td><td> $\underline { { 7 7 . 2 8 } } \pm 1 . 9 5$ </td></tr><tr><td> $\mathsf { T } \mathbf { - } \mathsf { R o u t e r } + \mathsf { G R P O }$ </td><td>41.730</td><td> $\mathbf { 9 7 . 6 2 \pm 0 . 7 9 }$ </td><td> $\mathbf { 9 2 . 7 3 \bot 0 . 7 0 }$ </td><td> ${ \bf 6 0 . 5 6 \pm 3 . 4 7 }$  </td><td> $\mathbf { 8 3 . 6 4 \pm 1 . 1 6 }$ </td></tr></table>

## 4.3 Agentic search

Table 2 : Agentic-search comparison. Mean SD across three evaluation seeds. BrowseComp Plus uses answer token-F1 100; ASearch Test uses a 0–100 score. ∆ is the paired diference from full-parameter GRPO; dashes mark the reference. Rankings follow Table 1.
<table><tr><td>Method</td><td>BrowseComp F1</td><td> $\Delta { \mathrm { \ v s { \ G R P O } } }$ </td><td></td><td>ASearch Δ vs GRPO</td></tr><tr><td>Frozen base</td><td> $5 . 1 1 \pm 0 . 3 0$ </td><td> $- 2 8 . 1 2 \pm 1 . 2 5$ </td><td> $5 9 . 7 6 \pm 0 . 4 1$ </td><td> $- 1 0 . 1 9 \pm 0 . 5 6$ </td></tr><tr><td>Full-parameter GRPO</td><td> $3 3 . 2 3 \pm 1 . 5 4$ </td><td></td><td> $6 9 . 9 4 \pm 0 . 5 3 $ </td><td></td></tr><tr><td>RFT</td><td> $3 1 . 2 0 \pm 0 . 7 5$ </td><td> $- 2 . 0 3 \pm 2 . 0 7$ </td><td> $6 5 . 8 1 \pm 0 . 9 2$ </td><td> $- 4 . 1 4 \pm 1 . 1 8$ </td></tr><tr><td>LoRA  $- \mathrm { r } 2 \mathrm { ~ + ~ } \mathrm { G R P O }$ </td><td> $3 2 . 9 8 \pm 0 . 5 6$ </td><td> $- 0 . 2 5 \pm 1 . 2 5$ </td><td> $6 9 . 1 9 \pm 0 . 5 9$ </td><td> $- 0 . 7 5 \pm 1 . 0 1$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td> $3 6 . 0 6 \pm 0 . 5 6$ </td><td> $+ 2 . 8 3 \pm 1 . 8 0$ </td><td> $7 0 . 5 9 \pm 0 . 1 8$ </td><td> $+ 0 . 6 4 \pm 0 . 6 4$ </td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td> $\mathbf { 3 7 . 0 0 } \pm 1 . 1 7$ </td><td> $\mathbf { + 3 . 7 7 \pm 2 . 5 6 }$ </td><td> $7 1 . 4 6 \pm 0 . 5 5$ </td><td> $+ 1 . 5 2 \pm 0 . 6 5$ </td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td> $3 5 . 5 0 \pm 0 . 5 7$ </td><td> $+ 2 . 2 7 \pm 1 . 5 8$ </td><td> $7 0 . 6 7 \pm 0 . 1 9$ </td><td> $+ 0 . 7 3 \pm 0 . 3 4$ </td></tr><tr><td> $\mathrm { L o R A \mathrm { - } r 1 6 + G R P O , r e t r i e s }$ </td><td> $3 6 . 9 0 \pm 1 . 1 3$ </td><td> $+ 3 . 6 7 \pm 1 . 3 2$ </td><td> $\underline { { 7 2 . 8 0 } } \pm \mathbf { 0 . 7 5 }$ </td><td> $\pm 2 . 8 6 \pm 0 . 6 3$ </td></tr><tr><td>T-Router + GRPO</td><td> $3 6 . 9 8 \pm 1 . 1 6$ </td><td> $\pm 3 . 7 5 \pm 0 . 9 1$ </td><td> $\mathbf { 7 3 . 9 8 \pm 0 . 7 4 }$ </td><td> $+ 4 . 0 4 \pm 1 . 2 7$ </td></tr></table>

The search models are trained separately using ASearcher data (Gao et al., 2025) and the search-agent-rl implementation (siqi654321, 2026), which generates multi-turn trajectories with a retrieval-and-summary tool. T-Router reaches 36.98 1.16 on BrowseComp Plus (Chen et al., 2025) and $7 3 . 9 8 \pm 0 . 7 4$ on ASearch Test (Table 2). Its paired gains over full-parameter GRPO are $3 . 7 5 \pm 0 . 9 1 \ \mathrm { F 1 }$ points and 4.04 1.27 ASearch points. It achieves the highest ASearch mean, 1.18 points above matched-retry LoRA. On BrowseComp Plus, its 36.98 F1 is comparable to LoRA-MoE’s 37.00 with 24.1% of the latter’s trainable allocation. These results extend the quality–allocation tradeof to tool-mediated information search.

## 4.4 Addressable content and recurrent control

Table 3 tests the source bank, controller, writeback, and training recipe. Figure 4 aligns task responses with capacity and cost; Appendix C gives complete numerical profiles.

Preserve contributions as reusable content. With exactly 41,730,332 parameters in both settings, block-change memory reaches 83.64 1.16 MathAvg versus 67.46 2.19 for hidden-state memory. Removing cross-layer retrieval yields $6 9 . 4 8 \pm 2 . 3 6$ . The full interface favors a particular use of its capacity: preserve completed contributions as individually addressable sources for later computation. Compression retains each block’s identity, allowing a receiver to revisit its contribution through a low-dimensional record while preserving backbone width.

c Search gains over Full GRPO  
a MathAvg: mean ± SD  
![](images/0833817fbe8869117bd839cd40da84c34a1af9bc789f43359b1aa45454ac6895.jpg)  
b Matched-retry task gains

![](images/db96a6f754008febf10904932e58cf799722fc6e6f813a2cb56b55ee6c2ea78e.jpg)

![](images/b846a8e0196538686bad7890ded4b931389ba2a95d4bcfbe3959cfb30a2433c5.jpg)

![](images/8fb5608188aaec01adcc3d8b6afdb6fec0adff61cdc3ba57aeb29f50c2479cde.jpg)  
Figure 3 : Reasoning quality and parameter efficiency. (a) MathAvg mean SD. (b) Task-family gains over matched-retry LoRA, computed from displayed means. (c) Paired gains over full-parameter GRPO in each search metric, mean SD. (d) Trainable allocations on a logarithmic scale; LoRA-r16 variants share the same budget. Error bars summarize three evaluation rounds.

Table 3 : Components of the reuse interface. Accuracy (%), mean SD across evaluation seeds. ∆ is the paired MathAvg diference from full T-Router; a dash marks the reference. The MLP retains comparable parameter capacity; hidden-state memory keeps the full allocation. Appendix C.2 includes both annual AIME scores.
<table><tr><td>Configuration</td><td>GSM8K</td><td>MATH-500</td><td>MathAvg</td><td>Δ</td></tr><tr><td>Full T-Router</td><td> $\mathbf { 9 7 . 6 2 \pm 0 . 7 9 }$ </td><td> $\mathbf { 9 2 . 7 3 \bot 0 . 7 0 }$ </td><td> $\mathbf { 8 3 . 6 4 \pm 1 . 1 6 }$ </td><td></td></tr><tr><td>Without cross-layer retrieval</td><td>96.26 ±0.59</td><td>79.40 ±1.56</td><td>69.48 ±2.36</td><td>−14.16 ±1.78</td></tr><tr><td>Hidden-state memory</td><td>96.79 ±0.53</td><td>77.80 ±1.31</td><td>67.46 ±2.19</td><td>−16.18 ±2.96</td></tr><tr><td>Learned static routing</td><td>96.01 ±0.68</td><td>77.27 ±0.81</td><td>64.98 ±1.71</td><td>−18.66 ±2.78</td></tr><tr><td>State replaced by matched MLP</td><td>95.02 ±0.75</td><td>83.93 ±1.92</td><td>72.80 ±4.10</td><td>−10.84±3.09</td></tr><tr><td>Without layer/block identity</td><td>94.29 ±0.42</td><td>66.13 ±1.80</td><td>58.10 ±1.06</td><td>−25.53 ±0.56</td></tr><tr><td>Without RMS alignment</td><td>94.57 ±0.64</td><td>67.93 ±3.80</td><td>56.39 ±0.41</td><td>−27.25 ±1.24</td></tr><tr><td>Without token-conditioned gate</td><td>96.26 ±0.42</td><td>66.73 ±1.81</td><td>63.78 ±2.61</td><td>-19.86±1.47</td></tr><tr><td>Without informative retries</td><td>94.39 ±0.55</td><td>74.87 ±2.55</td><td>62.53 ±1.20</td><td>−21.11 ±2.26</td></tr></table>

Use depth history as routing context. The recurrent interface reaches 83.64 1.16 MathAvg versus 72.80 4.10 for a capacity-matched MLP with 41,731,978 parameters. Here S accumulates depth history, and its readout P conditions the router and gate while content comes from the bank.

## 4.5 Source identity and relative-scale writeback

Table 3 tests the coordinates used to learn reuse. Layer and block identities specify the contribution’s origin and receiving location; the full setting reaches 83.64 MathAvg versus 58.10 without them. The token-conditioned gate sets the mixture’s signed influence; its removal gives 63.78. RMS alignment makes this influence relative to the receiving residual (Equation 8); omitting it gives 56.39. This calibration changes both forward intervention scale and gradient geometry, as derived in Appendix B.5. It provides a scale-defined optimization interface for learning writeback. Appendix C further characterizes the interaction between writeback and routing structure.

## a MathAvg Δ from Full T-Router ± SD

<table><tr><td colspan="3">Full T-Router: 83.64 ± 1.16</td></tr><tr><td>No RMS alignment</td><td></td><td>-27.25±1.24</td></tr><tr><td>No layer/block identity</td><td></td><td>-25.53±0.56</td></tr><tr><td>No informative retries</td><td></td><td>-21.11±2.26</td></tr><tr><td>No token gate</td><td></td><td>-19.86±1.47</td></tr><tr><td>Static routing</td><td></td><td>-18.66±2.78</td></tr><tr><td>Hidden-state memory</td><td></td><td>-16.18±2.96</td></tr><tr><td>No cross-layer retrieval</td><td></td><td>-14.16±1.78</td></tr><tr><td>Matched MLP controller</td><td></td><td>-10.84±3.09</td></tr></table>

![](images/1b35b7af50bf3dcc6b74b11c70e8ddd98d9e8e88e9fa61c4d46a0103c163227a.jpg)

b Task mean changes (pp)
<table><tr><td rowspan=1 colspan=5>GSM8K      MATH         AIME</td></tr><tr><td rowspan=1 colspan=1>-3.05</td><td></td><td rowspan=1 colspan=1>-24.80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-53.89</td></tr><tr><td rowspan=1 colspan=1>-3.33</td><td></td><td rowspan=1 colspan=1>-26.60</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-46.67</td></tr><tr><td rowspan=1 colspan=1>-3.23</td><td></td><td rowspan=1 colspan=1>-17.86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-42.23</td></tr><tr><td rowspan=1 colspan=1>-1.36</td><td></td><td rowspan=1 colspan=1>-26.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-32.23</td></tr><tr><td rowspan=1 colspan=1>-1.61</td><td></td><td rowspan=1 colspan=1>-15.46</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-38.89</td></tr><tr><td rowspan=1 colspan=1>-0.83</td><td></td><td rowspan=1 colspan=1>-14.93</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-32.78</td></tr><tr><td rowspan=1 colspan=1>-1.36</td><td></td><td rowspan=1 colspan=1>-13.33</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-27.78</td></tr><tr><td rowspan=1 colspan=1>-2.60</td><td></td><td rowspan=1 colspan=1>-8.80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-21.11</td></tr></table>

## c Matched-capacity replacements

<table><tr><td></td><td>Params (M)</td><td>MathAvg ± SD</td></tr><tr><td>Full T-Router</td><td>41.730332</td><td>83.64±1.16</td></tr><tr><td>Hidden-state memory</td><td>41.730332</td><td>67.46±2.19</td></tr><tr><td>MLP controller</td><td>41.731978</td><td>72.80±4.10</td></tr></table>

MLP replacement: +1,646 parameters

## d Recorded training cost

![](images/407373fc4e2e946cb963a4fc84c8c2bab6fb24946b776f232f6ba5f3af62bf99.jpg)  
Figure 4 : Dissecting controlled reuse. (a) Paired MathAvg changes from full T-Router, mean SD. (b) Family-wise changes in means; AIME averages both years. (c) Memory and controller replacements at matched capacity. (d) MathAvg versus recorded GPU-hours for all configurations. Numerical summaries in (c) and error bars in (d) show MathAvg SD across evaluation rounds. The no-retry setting also changes the generation budget.

## 4.6 Learning and executing the reuse policy

Informative groups support learning the reuse policy. The full recipe records 121.23 GPU-hours and 83.64 1.16 MathAvg; without informative retries, the corresponding values are 37.92 GPU-hours and 62.53 1.20. Retries invest additional generation in finding mixed-reward groups that carry a policy-gradient signal. Matched-retry LoRA applies the same rule and reaches $7 7 . 2 8 \pm 1 . 9 5$ (Table 1). The capacity-matched MLP records 69.25 GPU-hours and $7 2 . 8 0 \pm 4 . 1 0$ . Figure 4c,d shows these quality–cost profiles; parameter allocation and recorded training efort are distinct eficiency dimensions.

Summary: learning what to reuse and how strongly. At each receiver, source transports share $U _ { \ell } ,$ constraining the intervention to an at-most-r-dimensional subspace (Appendix B.3). Source attention, recurrent context, and signed gating choose an input-dependent intervention within that space. The method comparison establishes gains across three mathematical families over matched-retry LoRA; the capacity-controlled comparisons favor block changes and recurrent context. Correctness rewards optimize these content and control pathways together. Appendix C contains complete configuration profiles, costs, auxiliary metrics, and analytical score reaggregations.

## 5 Conclusion

T-Router improves the parameter eficiency of reasoning RL by learning how completed computations are reused. Its compressed source bank, depth-recurrent controller, and relative-scale writeback form a compact thalamic routing interface around a frozen backbone. With 0.466% of the backbone’s parameter count, it reaches 83.64 MathAvg versus 73.79 for full-parameter GRPO and 77.28 for matched-retry LoRA. Capacitycontrolled comparisons favor addressable changes and recurrent context; separate search training extends the interface to tool use. The central result is an eficient allocation of adaptation capacity: preserve pretrained transformations, and learn how their contributions serve subsequent reasoning.

## Reproducibility statement

Section 3 defines the module and training objective. Appendix A gives tensor shapes, exact parameter allocation, execution order, and evaluation procedures. Appendix B supplies derivations. Appendix C includes the complete component results and costs, evaluation means and variances, configuration comparisons, auxiliary metrics, and definitions of every analytical reaggregation. The accompanying implementation includes the module, configuration constructor, training launcher, and evaluators; figure sources expose the tabulated numerical inputs.

## AI use statement

We used AI assistants in two roles. First, for manuscript preparation, including conceptual framing, drafting, language editing, literature retrieval, mathematical derivations, and result interpretation. Second, for coding, LaTeX production, and figure preparation, including the AI-generated brain illustration. The authors made the final decisions on the method, experimental design, and implementation, verified all AI-assisted output, and take full responsibility for this paper.

## References

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. BrowseComp-Plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025. URL https://arxiv.org/abs/2508.06600.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous RL. arXiv preprint arXiv:2508.07976, 2025. URL https://arxiv.org/abs/2508.07976.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, pp. 59532–59569, 2024. doi: 10.52202/079017-1902. URL https://proceedings.neurips.cc/paper\_files/paper/2024/fil e/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https://datasets-benchma rks-proceedings.neurips.cc/paper\_files/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-eficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2790–2799, 2019. URL https://proceedings.mlr.press/v97/houlsby19a.html.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=CKOJlBPqyU.

Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4700–4708, 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/html/Huang\_Densely\_Connected\_Convolutional \_CVPR\_2017\_paper.html.

Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, and Xinyu Zhou. Attention residuals. arXiv preprint arXiv:2603.15031, 2026. URL https://arxiv.org/abs/2603.15031.

Alan Lee and Harry Tong. Token-eficient RL for LLM reasoning. In ICML 2025 Workshop on Tiny Titans: The Next Wave of On-Device Learning for Foundational Models, 2025. URL https://openreview.net/forum?id=XO8whE6Tx3.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, pp. 39578–39601, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/aca97732e30b cf1303bc22ac3924fd16-Paper-Conference.pdf.

Cheng Luo, Zefan Cai, and Junjie Hu. Delta attention residuals. arXiv preprint arXiv:2605.18855, 2026a. URL https://arxiv.org/abs/2605.18855.

Cheng Luo, Zefan Cai, and Junjie Hu. Multi-head attention residuals. arXiv preprint arXiv:2607.27230, 2026b. URL https://arxiv.org/abs/2607.27230.

Changlian Ma, Zizheng Huang, Xiangyu Zeng, Yi Wang, Cheng Liang, Kun Tian, Xinhai Zhao, and Limin Wang. Balancing the experts: Unlocking LoRA-MoE for GRPO via mechanism-aware rewards. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/7f403d7240d8a5b 5335e60635ea2e975-Abstract-Conference.html.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=Byj72udxe.

Valentijn Oldenburg, Floris de Kam, Bente Zuijdam, Lieve Eberson, Nicky van Zutphen, Stef de Wildt, and Ivo Verhoeven. Manifold-constrained hyper-connections for parameter-eficient finetuning. arXiv preprint arXiv:2607.18130, 2026. URL https://arxiv.org/abs/2607.18130.

Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, and Martin Jaggi. DenseFormer: Enhancing information flow in transformers via depth weighted averaging. In Advances in Neural Information Processing Systems, volume 37, pp. 136479–136508, 2024. doi: 10.52202/079017-4336. URL https://proceedings.neurips.cc/paper\_files/paper/2024 file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf.

Qwen Team. Qwen3.5-9B-Base model card. Hugging Face, 2026. URL https://huggingface.co/Qwen/Qwen3.5-9B-Base. Accessed September 26, 2026.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Ti67584b98.

Yuri B. Saalmann, Mark A. Pinsk, Liang Wang, Xin Li, and Sabine Kastner. The pulvinar regulates information transmission between cortical areas based on attention demands. Science, 337(6095):753–756, 2012. doi: 10.1126/sc ience.1223082. URL https://pubmed.ncbi.nlm.nih.gov/22879517/.

L. Ian Schmitt, Ralf D. Wimmer, Miho Nakajima, Michael Happ, Sima Mofakham, and Michael M. Halassa. Thalamic amplification of cortical connectivity sustains attentional control. Nature, 545(7653):219–223, 2017. doi: 10.1038/na ture22073. URL https://www.nature.com/articles/nature22073.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Hakim Sidahmed, Samrat Phatale, Alex Hutcheson, Zhuonan Lin, Zhang Chen, Zac Yu, Jarvis Jin, Simral Chaudhary, Roman Komarytsia, Christiane Ahlheim, Yonghao Zhu, Bowen Li, Saravanan Ganesh, Bill Byrne, Jessica Hofmann, Hassan Mansoor, Wei Li, Abhinav Rastogi, and Lucas Dixon. Parameter eficient reinforcement learning from human feedback. arXiv preprint arXiv:2403.10704, 2024. URL https://arxiv.org/abs/2403.10704.

siqi654321. Search agent RL. GitHub software repository, 2026. URL https://github.com/siqi654321/search-agent-rl. Accessed September 26, 2026.

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. MuSR: Testing the limits of chainof-thought with multistep soft reasoning. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=jenyYQzue1.

Da Xiao, Qingye Meng, Shengping Li, and Xingyuan Yuan. MUDDFormer: Breaking residual bottlenecks in transformers via multiway dynamic dense connections. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 68440–68458, 2025. URL https://proceedings.mlr.press/v267/xiao25d.html.

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. mHC: Manifold-constrained hyper-connections. arXiv preprint arXiv:2512.24880, 2025. URL https://arxiv.org/abs/2512.24880.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244, 2025. doi: 10.52202/085713-3775. URL https: //proceedings.neurips.cc/paper\_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf.

Yilang Zhang, Bingcong Li, Niao He, and Georgios B. Giannakis. ANCRe: Adaptive neural connection reassignment for eficient depth scaling. In ICLR 2026 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Eficacy, 2026. URL https://openreview.net/forum?id=0BbbfFst1X.

Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. URL https: //arxiv.org/abs/2311.07911.

## Appendix

T-Router: Learning Thalamic Routing for Reasoning with Parameter-Eficient Reinforcement Learning

Computational specification, formal analysis, and the supporting experimental results.

## Specification and analysis

A Implementation and evaluation details 14   
A.1 Notation and tensor organization . 14   
A.2 Parameter allocation and initialization 15   
A.3 Layer schedule and source availability 16   
A.4 State lifetime, precision, and diferentiation 17   
A.5 Training configuration and optimization   
flow 18   
A.6 Evaluation tasks, repeated rounds, and scor   
ing 19   
A.7 Agentic search data and evaluation 20   
A.8 Prompt construction and final-answer ex  
traction 21   
A.9 Auxiliary metric definitions 22   
A.10 Analytical storage and projection work 23   
A.11 How architectural dimensions allocate ca  
pacity 23   
B Formal analysis of cross-layer communication 25   
B.1 Notation and completed-block event order  
ing 25   
B.2 What a block change contains 26   
B.3 Source transport and the receiving subspace 26   
B.4 Controller history, rank, and state bounds 27   
B.5 RMS geometry, gradients, and threshold   
behavior 28   
B.6 Stop-gradient and scale-invariant optimiza  
tion 30   
B.7 Conditional propagation through the frozen   
decoder . 31   
B.8 The diferentiated group-relative objective 31   
B.9 Routing regularization and its reduction . . 33   
B.10 Bounded informative resampling and gener  
ation budgets 33

## Results and interpretation

C Complete empirical results and design analy  
sis 35   
C.1 Method comparisons across mathematics   
and search 35   
C.2 Complete component ablations 36   
C.3 Parameter and training-cost profiles of the   
ablations 36   
C.4 Configuration landscape and annual task   
profiles 37   
C.5 Benchmark contributions to aggregate dif  
ferences 37   
C.6 Writeback configurations and task response 38   
C.7 Writeback coeficients and endpoint diag  
nostics 39   
C.8 Structure–writeback combinations 40   
C.9 Auxiliary task profiles 42   
C.10 Sensitivity to the choice of family weights 43   
C.11 Question resolution and alternative aggre  
gation 43   
C.12 Evaluation-seed dispersion and complete   
variance record 44   
D Design interpretation and functional corre  
spondence 47   
D.1 Computation and coordination as separate   
roles 47   
D.2 Why the source bank and controller carry   
diferent information 47   
D.3 Three coordinates of a residual intervention 48   
D.4 Relation to alternative adaptation inter  
faces 49   
D.5 Connecting the empirical comparisons to   
the interface 49

## Guide to the appendices

The appendices provide the full computational specification, mathematical analysis, and configuration-level results supporting the communication interface. Table 4 organizes the material by the question it answers.

Table 4 : Appendix guide. Each part develops a distinct aspect of the method and study.
<table><tr><td>Question</td><td>Material</td><td>Page</td></tr><tr><td>How is the module executed?</td><td>Tensor shapes, exact parameter counts, layer schedule, state lifetimes, training settings, prompts, scoring, and analytical storage</td><td>14</td></tr><tr><td rowspan="2">What properties follow from the design? What do the</td><td>Causal source availability, transport subspaces, slot history, calibration derivatives, gradient flow, and finite group sampling</td><td>25</td></tr><tr><td>Full-test comparisons, component results and costs, writeback and initialization comparisons, routing interactions, auxiliary diagnostics,</td><td>35</td></tr><tr><td>How do the components fit together?</td><td>evaluation variance, and metric sensitivity Thalamic functional correspondence, bank/controller roles, intervention coordinates, and adaptation interfaces</td><td>47</td></tr></table>

Tables 1–3 present the primary method and component comparisons. The results atlas includes their complete mean/SD profiles, sample variances, and resource measurements, alongside the historical configuration studies. Arithmetic decompositions and analytical illustrations are identified by their formulas and captions. The execution specification and derivations use zero-based decoder-layer indices, matching the source-availability schedule. Symbols retain the definitions in Section 3 throughout.

## A Implementation and evaluation details

This appendix specifies the computational objects, execution schedule, parameter allocation, and training and evaluation interfaces used by T-Router. The module operates on the residual input of a decoder layer. Source recording occurs after a completed block; the controller and receiving-layer intervention execute before the next frozen transformation. Keeping these events explicit gives a direct correspondence between the mathematical definition and a causal language-model implementation.

## A.1 Notation and tensor organization

The batch dimension is $B ,$ the number of positions in the current forward call is T, and the backbone residual width is d. We use L decoder layers, source blocks of size s, memory width r, K controller slots of width $p ,$ controller MLP width $c ,$ and source/layer embedding width e. Canonical values are

$$
( L , s , d , r , K , p , c , e ) = ( 3 2 , 4 , 4 0 9 6 , 2 5 6 , 8 , 2 5 6 , 2 5 6 , 3 2 ) .
$$

A token position is a position processed by the current call. During prompt prefill, T can be the entire prompt length. During ordinary one-token cached decoding, T = 1. All auxiliary operations broadcast over B and $T ;$ attention over source blocks and attention over controller slots are separate from the backbone’s attention over token positions.

The ordering of the bank is chronological in depth. Every record keeps its own compressor and source identity even when its current routing weight is small. Layer embeddings identify receiving locations, while source embeddings identify completed blocks. Their roles are asymmetric: a receiver asks a context-dependent question about the available records, and each record provides an origin-labelled key and value. The canonical configuration retains all completed blocks within the forward call.

Controller context is the p-dimensional readout $P _ { \ell , t }$ in Equation 5. The learned, bias-free maps $Q _ { s } \in \mathbb { R } ^ { p \times d }$ and $K _ { s } , V _ { s } \in \mathbb { R } ^ { p \times p }$ are shared across layers. Attention compares $Q _ { s } h _ { \ell , t }$ with the K updated slot keys, normalizes over slots, and averages their projected values. The recurrent state S combines residual states, source summaries, and layer identity; the readout P conditions source selection and the gate. For example, layers 4 and 5 see the same bank $\{ m _ { 0 } \}$ , but their residuals and controller states can difer, yielding diferent contexts. With one source its routing weight is one, while the learned gate can still vary with P.

Table 5 : Runtime tensors and their axes. $J _ { \ell } = | \mathcal { I } _ { \ell } |$ is the number of completed sources visible before layer ℓ. No source or slot axis is a token-history axis.
<table><tr><td>Object</td><td>Tensor shape</td><td>Role</td></tr><tr><td>Residual  $H _ { \ell }$ </td><td> $B \times T \times d$ </td><td>Input to a receiving decoder layer</td></tr><tr><td>Block anchor  $A$ </td><td> $B \times T \times d$ </td><td>Post-intervention input at the start of a block</td></tr><tr><td>Block displacement  $D _ { b }$ </td><td> $B \times T \times d$ </td><td>Completed block output minus its anchor</td></tr><tr><td>Compressed memory  $M _ { b }$ </td><td> $B \times T \times r$ </td><td>Source-specific record of that displacement</td></tr><tr><td>Visible bank</td><td> $B \times T \times J _ { \ell } \times r$ </td><td>Completed records available to a receiver</td></tr><tr><td>Controller slots  $S _ { \ell }$ </td><td> $B \times T \times K \times p$ </td><td>Depth history used to condition retrieval</td></tr><tr><td>Controller context  $P _ { \ell }$ </td><td> $B \times T \times p$ </td><td>Slot readout for the source query and gate</td></tr><tr><td>Source weights  $a \ell$ </td><td> $B \times T \times J _ { \ell }$ </td><td>Distribution over completed source blocks</td></tr><tr><td>Source mixture  $c _ { \ell }$ </td><td> $B \times T \times r$ </td><td>Weighted sum of projected source values</td></tr><tr><td>Raw direction  $W _ { \ell }$ </td><td> $B \times T \times d$ </td><td>Receiving-layer projection of ce</td></tr><tr><td>Gate  $g _ { \ell }$ </td><td> $B \times T \times 1$ </td><td>Signed coefficient for calibrated writeback</td></tr><tr><td>Intervention  $R _ { \ell }$ </td><td> $B \times T \times d$ </td><td>Additive input to the frozen layer</td></tr></table>

## A.2 Parameter allocation and initialization

Table 6 counts scalar parameters by component. Matrices are expressed as output-by-input dimensions. The controller MLP is

$$
\begin{array} { r } { \mathrm { M L P } ( z ) = W _ { 2 } \mathrm { S i L U } ( W _ { 1 } z + b _ { 1 } ) + b _ { 2 } , \qquad z \in \mathbb { R } ^ { 2 p + e } , } \end{array}
$$

with hidden dimension c and output dimension $2 p$ . The two output halves provide a shared proposal and an elementwise proposal gate. The MLP has biases; the listed linear projections are otherwise bias free. The receiving gate has a shared vector and a separate bias for each routed layer.

Table 6 : Canonical module dimensions and exact parameter counts. Counts include the allocated terminal source. The final row counts parameters with a path to the training loss.
<table><tr><td>Component</td><td>Shape or multiplicity</td><td>Parameters</td></tr><tr><td>Block compressors  $C _ { b }$ </td><td> $8 \times ( r \times d )$ </td><td>8,388,608</td></tr><tr><td>Writeback projections  $U _ { \ell }$ </td><td> $2 8 \times ( d \times r )$ </td><td>29,360,128</td></tr><tr><td>Layer and source embeddings</td><td> $3 2 e + 8 e$ </td><td>1,280</td></tr><tr><td>Initial controller slots</td><td> $K \times p$ </td><td>2,048</td></tr><tr><td>Routing query  $W _ { q }$ </td><td> $r \times ( d + p + e )$ </td><td>1,122,304</td></tr><tr><td>Routing key and value</td><td> $2 \times \left[ r \times ( r + e ) \right]$ </td><td>147,456</td></tr><tr><td>Hidden projection  $A _ { h }$ </td><td> $p \times d$ </td><td>1,048,576</td></tr><tr><td>Memory projection  $A _ { m }$ </td><td> $p \times r$ </td><td>65,536</td></tr><tr><td>Controller MLP</td><td> $( 2 p + e ) \to c \to 2 p$ </td><td>271,104</td></tr><tr><td>Slot selector  $W _ { s }$ </td><td> $p \times ( 2 p + e )$ </td><td>139,264</td></tr><tr><td>Slot query  $Q _ { s }$ </td><td> $p \times d$ </td><td>1,048,576</td></tr><tr><td>Slot key and value</td><td> $2 \times ( p \times p )$ </td><td>131,072</td></tr><tr><td>Dynamic and layer gates</td><td> $\mathbb { R } ^ { d + p } \mathrm { \ v e c t o r \ } + 2 8$  scalars</td><td>4,380</td></tr><tr><td>Allocated total</td><td></td><td>41,730,332</td></tr><tr><td>Terminal source parameters</td><td> $r d + e$ </td><td></td></tr><tr><td>Loss-connected total</td><td></td><td>1,048,608  $\pmb { 4 0 } , \pmb { 6 8 1 } , 7 2 4$ </td></tr></table>

There are 28 receiving projections because the first four layers have no completed source to read. Eight compressors are allocated, one for each block, while the eighth source is produced after the last receiving decision. Removing that terminal compressor and its source embedding changes the allocated count by rd + e = 1, 048, 608 and leaves the output unchanged. We distinguish allocation from loss connectivity so that the parameter count describes both the implemented state and the efective learning interface.

The allocated count is 0.4661% of the 8,953,803,264-parameter backbone; the loss-connected count is 0.4544%. Receiving projections account for 29.36M scalars and source compression accounts for 8.39M. Together they contain approximately 90.46% of the allocation. The remainder implements routing, state update, slot reading, and scalar control. Thus most trainable capacity transforms source content into receiver-specific directions; the controller coordinates those transformations with a comparatively compact shared mechanism.

Table 7 : Initialization of the canonical learned-RMS module. Initialization assigns content transport and intervention magnitude distinct roles.  
```perl
Component Initialization
Source compressors, Gaussian with standard deviation $1 0 ^ { - 3 }$
source/layer embeddings,
initial slots
Routing, controller-input Xavier initialization
and slot projections
Controller MLP output Zero weights and biases
layer
Receiving projections $U _ { \ell }$ Orthogonal initialization, gain 1
Shared dynamic gate Zero
vector
Receiving-layer gate bias arctanh $\left( 0 . 0 2 / 0 . 0 5 \right)$
Gate bound and initial $g _ { \mathrm { m a x } } = 0 . 0 5$ and $g _ { \mathrm { i n i t } } = 0 . 0 2$
coeficient
```

The zero MLP output makes the initial proposal vanish, so the slots initially follow their decay from the learned starting array. This does not zero the complete module: the source bank and orthogonal receiving projections provide a direction, and the initialized gate applies a calibrated 2% coeficient. As optimization changes the proposal, diferent layers contribute diferent updates to the slot history. Appendix B.4 derives that accumulation.

## A.3 Layer schedule and source availability

The distinction between a source’s creation and a receiver’s decision is explicit in Table 8. Before a layer executes, its controller reads the records that already exist. The intervention is applied next. If the layer starts a block, its post-intervention residual becomes the anchor. The frozen layer then runs, and a block-ending layer creates a new source for subsequent receivers.

Table 8 : Source availability over 32 decoder layers. Layer indices are zero based. Each row covers four receiver decisions; the source appended at the end is available from the next row onward.
<table><tr><td>Receiving layers</td><td>Sources visible before layer Appended after</td><td></td><td>r New source</td><td>Read count</td></tr><tr><td>0-3</td><td>None</td><td>Layer 3</td><td>m0</td><td>0</td></tr><tr><td>4-7</td><td>mo</td><td>Layer 7</td><td>m1</td><td>1</td></tr><tr><td>8-11</td><td> $m _ { 0 } , m _ { 1 }$ </td><td>Layer 11</td><td>m2</td><td>2</td></tr><tr><td>12-15</td><td> $m _ { 0 } , m _ { 1 } , m _ { 2 }$ </td><td>Layer 15</td><td>m3</td><td>3</td></tr><tr><td>16-19</td><td> $m _ { 0 } , \ldots , m _ { 3 }$ </td><td>Layer 19</td><td>m4</td><td>4</td></tr><tr><td>20-23</td><td> $m _ { 0 } , \ldots , m _ { 4 }$ </td><td>Layer 23</td><td>m5</td><td>5</td></tr><tr><td>24-27</td><td> $m _ { 0 } , \ldots , m _ { 5 }$ </td><td>Layer 27</td><td>m6</td><td>6</td></tr><tr><td>28-31</td><td> $m _ { 0 } , \ldots , m _ { 6 }$ </td><td>Layer 31</td><td>m7</td><td>7</td></tr></table>

The total number of source–receiver pairs per position is $4 ( 1 + 2 + \cdot \cdot \cdot + 7 ) = 1 1 2$ . These are attention candidates, not 112 backbone layer executions: the backbone always runs exactly 32 layers. Source selection has one candidate in layers 4–7, so its normalized weight is identically one there. The receiving projection and gate still operate in those layers. Source preference becomes a nontrivial distribution at layer 8, when the second completed block is available.

At layer 4, for example, the controller has already progressed through four earlier updates. It incorporates $m _ { 0 } ,$ reads its slots, and constructs the direction for layer 4 from the sole source. At layer 8, the same sequence of operations includes $m _ { 0 }$ and $m _ { 1 }$ . Their attention weights can difer with the receiving token, hidden state, and accumulated controller context. This schedule separates the growth of the source set from the evolution of the state used to query it.

Algorithm 1 One causal T-Router forward pass   
Require: Residual tensor $H _ { 0 } \in \mathbb { R } ^ { B \times T \times d } ,$ , frozen layers $B _ { 0 } , \ldots , B _ { L - 1 }$   
1: Copy the learned initial slots to all current batch/token positions   
2: Set source bank $\mathcal { M }  \emptyset$ and block anchor $A  \emptyset$   
3: for $\ell = 0 , \ldots , L - 1$ do   
4: Compute the mean visible memory; use zero when the bank is empty   
5: Update slots, then read $P _ { \ell }$ using Equations 3–5   
6: if $\mathcal { M } \neq \emptyset$ then   
7: Compute source weights and retrieved source mixture using Equation 6   
8: Project, calibrate and gate the writeback using Equation 7   
9: else   
10: $R _ { \ell } \gets 0$   
11: end if   
12: $\widetilde { H } _ { \ell } \gets H _ { \ell } + R _ { \ell }$   
13: eif ℓ mod $s = 0$ then   
14: $A  \widetilde { H } _ { \ell }$   
15: end if   
16: $H _ { \ell + 1 } \gets B _ { \ell } ( \widetilde { H } _ { \ell } )$   
17: if $( \ell + 1 )$ emod $s = 0$ then   
18: $b \gets \lfloor \ell / s \rfloor$   
19: Append $\bar { ( } C _ { b } ( H _ { \ell + 1 } - A ) , e _ { b } ^ { \mathrm { s r c } } )$ to $\mathcal { M }$   
20: Clear the completed block anchor   
21: end if   
22: end for   
23: return $H _ { L }$

The post-intervention anchor matters for the meaning of a record. The writeback immediately before the block’s first layer is already included in the anchor; subsequent writebacks are included in the completed displacement. A record therefore describes the change produced along the block’s actual executed path. The decomposition in Appendix B makes this distinction algebraic. It is useful when comparing a stored displacement with a hypothetical sum of frozen-layer outputs evaluated on another trajectory.

## A.4 State lifetime, precision, and differentiation

Every forward call starts a fresh bank and copies the learned initial slot array. Runtime state also resets when the first target layer is entered. During teacher forcing, each token position has its own depth-local bank and controller. During cached generation, each new call constructs these objects for the current positions, while the backbone retains its normal causal token cache. Depth recurrence therefore operates inside a forward call; the backbone cache carries context across calls.

The adapter computes its projections, slots, normalization, and gates in fp32. Frozen backbone tensors use bfloat16. The final intervention is cast back to the receiving residual’s dtype before addition, keeping the next decoder layer’s expected dtype. The target hidden-state RMS is detached; gradients remain active through the direction normalization, source weights, source content, controller, and gate inputs. This is the implementation of the geometric distinction analyzed in Appendix B.5

Frozen backbone parameters have no parameter gradients, while their input derivatives connect earlier interventions to the final loss. Compressed records remain in the graph as well. An earlier intervention can therefore influence later losses through both the ordinary residual path and the record subsequently read by another layer. The first routed layer is layer 4; the prefix before that point supplies inputs to trainable auxiliary operations without requiring gradients for its frozen weights.

Table 9 : Objects with different lifetimes. The same word “memory” can refer to trainable state, temporary depth context, or the language model’s causal cache.
<table><tr><td>Object</td><td>Lifetime</td><td>Update rule</td></tr><tr><td>Adapter parameters</td><td>Shared across examples and forward calls</td><td>Optimizer update during training</td></tr><tr><td>Initial slot array</td><td>Learned parameter</td><td>Copied at forward initialization</td></tr><tr><td>Controller state</td><td>Current forward call and token position</td><td>Recurrent update at each layer</td></tr><tr><td>Source bank</td><td>Current forward call and token position</td><td>Append after a completed block</td></tr><tr><td>Block anchor</td><td>Current block</td><td>Replace at block start; clear at block end</td></tr><tr><td>Backbone causal cache</td><td>Generation context</td><td>Managed by the backbone&#x27;s decoder</td></tr></table>

Activation checkpointing encloses the frozen decoder-layer computation. Source recording and controller updates occur outside the recomputed function. Consequently, recomputation reevaluates a frozen transformation at its saved input and does not append another source or advance the controller a second time. This placement preserves the one-update-per-layer semantics of Algorithm 1. The adapter remains diferentiable while the frozen backbone is kept in evaluation mode.

## A.5 Training configuration and optimization flow

The final mathematics comparison trains each adaptation method on all 7,473 GSM8K training examples. Evaluation uses the full GSM8K test set, all 500 MATH-500 questions, and both 30-question AIME sets. The component study reports an estimated 3,737 optimizer steps per configuration, with exact trainable allocations and training costs in Table 21. Prompt count, optimizer steps, and generated responses describe diferent aspects of this training budget.

Table 10 : Final mathematics comparison: training and evaluation scope. The full-test suite is shared by the methods in Table 1.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training data</td><td>GSM8K train: 7,473 questions</td></tr><tr><td>Evaluation data</td><td>GSM8K: 1,319; MATH-500: 500; AIME24/25: 30 each</td></tr><tr><td>Evaluation generation</td><td>One generated answer per question in each evaluation round</td></tr><tr><td>Evaluation aggregate</td><td>Equal weight for GSM8K, MATH-500, and mean AIME</td></tr><tr><td>T-Router allocation</td><td>41,730,332 parameters; 40,681,724 loss connected</td></tr><tr><td>Reference policy</td><td>Frozen backbone with the adapter disabled</td></tr></table>

Table 11 : Optimization controls exposed by the implementation. Values are the implementation’s configurable mathematics recipe. Sampling and regularization are specified separately from the benchmark aggregation rule.
<table><tr><td>Control</td><td>Value</td><td>Control</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Weight decay</td><td>0.01</td></tr><tr><td>Peak learning rate</td><td>10−4</td><td>Warmup</td><td>10% of scheduled updates</td></tr><tr><td>Schedule</td><td>Linear warmup/decay</td><td>Accumulation</td><td>2 prompt groups</td></tr><tr><td>Prompt batch size</td><td>1</td><td>Group size</td><td>4 completions</td></tr><tr><td>Additional retry limit</td><td>3 groups</td><td>Temperature</td><td>0.7</td></tr><tr><td>Top-p</td><td>0.95</td><td>Completion ceiling</td><td>512 tokens</td></tr><tr><td>KL coefficient</td><td>0.02</td><td>Allocation coefficient</td><td>0.01</td></tr><tr><td>Backbone dtype</td><td>bfloat16</td><td>Adapter arithmetic</td><td>fp32</td></tr><tr><td>Frozen-layer checkpointing</td><td>Enabled</td><td>Slot-diversity weight</td><td>0</td></tr></table>

For each prompt, generation uses the current adapted model. A group whose correctness rewards are all equal can be replaced by another group for that same prompt, with at most three replacements. The retained group is the first mixed-correctness group or the final attempt. The procedure then evaluates adapted and adapter-disabled log probabilities on the retained completions, computes group-standardized advantages, and diferentiates the policy, KL, and allocation terms. The configurable recipe accumulates two prompt groups before an optimizer update. For $N _ { \mathrm { o p t } }$ scheduled updates, warmup lasts max $( 1 , \lfloor { N _ { \mathrm { o p t } } } / 1 0 \rfloor )$ updates.

The generation distribution uses temperature and nucleus sampling, whereas the objective uses full-vocabulary model log probabilities. An explicit top-k override is not applied, so generation retains the loaded model’s top-k setting. These are distinct parts of the procedure: sampling determines which responses enter the group, and the diferentiable objective assigns gradients to those responses. The exact group selection law and finite response budget are derived in Appendix B.

The adapter-disabled reference retains the same frozen backbone parameters. No separately trained reward model is used; the reward is defined by final-answer correctness. The configurable recipe disables parse-failure reward shaping and the optional length-format and correct-response anchoring losses. A completion that is both truncated and unparseable has zero preference weight. Other incorrect but valid completions remain in the group with reward zero. The KL term applies to the nonpadding completion tokens of the retained responses. The allocation term uses source weights $a \ell$ and source mixtures $c _ { \ell }$ across the current sequence, including prompt positions. The corresponding reductions are specified in Equation 49 and the training analysis.

## A.6 Evaluation tasks, repeated rounds, and scoring

Generative evaluations use three rounds with random seeds 42, 43, and 44. For a score $x _ { j }$ in round $j ,$ , the reported statistics are

$$
{ \bar { x } } = { \frac { 1 } { 3 } } \sum _ { j = 1 } ^ { 3 } x _ { j } , \qquad s ^ { 2 } = { \frac { 1 } { 2 } } \sum _ { j = 1 } ^ { 3 } ( x _ { j } - { \bar { x } } ) ^ { 2 } , \qquad s = { \sqrt { s ^ { 2 } } } .\tag{11}
$$

Tables report x¯ s; s is the sample SD across evaluation rounds. Within each round, AIME mean averages the two annual accuracies and MathAvg averages the three task families. For method comparisons, $d _ { j } = x _ { j } - y _ { j }$ pairs matching round indices, and the same formulas give <sup>¯</sup>d and its SD. Means and SD are displayed to two decimals, after aggregation; Appendix C.12 reports the complete variances to four decimals. Accuracy SD is in percentage points and its variance in squared percentage points; search scores use their corresponding points and squared points. Parameter allocations and GPU-hours are resource records, separate from these evaluation statistics.

The mathematics suite measures grade-school arithmetic, broader mathematical problem solving, and competition mathematics. Table 12 specifies the full-test denominators. Auxiliary tasks examine narrative reasoning with MuSR (Sprague et al., 2024), domain-question answering with GPQA (Rein et al., 2024), instruction compliance with IFEval (Zhou et al., 2023), and language-model likelihood with WikiText-2 (Merity et al., 2017). Their scores remain in their native units rather than entering MathAvg. MATH-500 uses the held-out subset released with PRM800K (Lightman et al., 2024), accessed through the Hugging Face interface in Table 13.

Table 12 : Full-test mathematics evaluation. Each evaluation round covers the listed questions. AIME’s two annual scores are averaged before the three-family MathAvg is computed.
<table><tr><td>Task</td><td>Questions</td><td>Scoring object</td></tr><tr><td>GSM8K</td><td>1,319</td><td>Normalized final answer</td></tr><tr><td>MATH-500</td><td>500</td><td>Mathematical equivalence of the final answer</td></tr><tr><td>AIME 2024</td><td>30</td><td>Mathematical equivalence of the final answer</td></tr><tr><td>AIME 2025</td><td>30</td><td>Mathematical equivalence of the final answer</td></tr></table>

The multiple-choice evaluator scores option likelihoods. MuSR provides narrative text, a question, and its explicit option list; the GPQA-D input contains the question and options. The evaluator tokenizes each space-prefixed option letter, requires it to be one token, and selects the highest next-token log probability. This decision rule makes the predicted option independent of the length of a generated explanation.

Table 13 : Dataset interfaces in the evaluation implementation. These identifiers specify the task content accessed by the corresponding loaders.
<table><tr><td>Task</td><td>Dataset interface</td></tr><tr><td>GSM8K</td><td>openai/gsm8k</td></tr><tr><td>MATH-500</td><td>HuggingFaceH4/MATH-500</td></tr><tr><td>AIME 2024</td><td>HuggingFaceH4/aime_2024</td></tr><tr><td>AIME 2025</td><td>yentinglin/aime_2025</td></tr><tr><td>MuSR</td><td>TAUR-Lab/MuSR</td></tr><tr><td>GPQA-D</td><td>fingertap/GPQA-Diamond</td></tr><tr><td>IFEval</td><td>google/IFEval</td></tr><tr><td>WikiText-2</td><td>Salesforce/wikitext</td></tr></table>

## A.7 Agentic search data and evaluation

Search training uses ASearcher data (Gao et al., 2025) and the search-agent-rl implementation (siqi654321, 2026), which combines verl training, SGLang trajectory generation, and a retrieval-and-summary tool. The specifications below describe the public pipeline’s data interfaces and default execution protocol. Dataset sizes count converted records before long-prompt filtering; retrieval and serving settings are repository defaults.

Data partitions and evaluation roles. The conversion reads aidenjhwu/ASearcher\_en\_no-math\_Qwen3-8 B-reject-sample. Its 13,985-record snapshot is randomly split with seed 42 and a 2% holdout, producing 13,706 training records and 279 held-out records. The latter file is named test.parquet, but the training entry point passes it to data.val\_files and validates every 50 optimizer steps. Accordingly, the column headed ASearch Test in Tables 2 and 19 refers to ASearcher held-out validation, rather than an additional independent test partition.

BrowseComp Plus (Chen et al., 2025) uses the 830 original questions in Tevatron/browsecomp-plus; the converter neither subsamples nor randomly repartitions them. Its evaluation entry point passes the same file to both train\_files and val\_files, with val\_before\_train=True and val\_only=True. The corresponding control flow returns after validation, without training updates. The retrieval collection, Tevatron/browsecomp plus-corpus, contains 100,195 documents in the public snapshot. These are source sizes; prompt filtering determines the efective evaluation count, and the local index determines the available retrieval collection.

Exact-match split diagnostic. The ASearcher converter splits records without first deduplicating question text. Reconstructing the default split from the fixed public snapshot identifies eight held-out records whose question text and reference answer exactly match a training record: $8 / 2 7 9 = 2 . 8 7 \%$ of the holdout. The records have diferent IDs, so ID-disjointness alone does not reveal these matches. This diagnostic concerns exact question–answer matches in the reconstructed, pre-filter split; it does not quantify overlap in a run-specific filtered file or semantic near-duplicates. The overlap count is a property of this validation partition and does not, by itself, attribute a diference between methods to duplication.

Answer extraction and token-F1. Both conversion entry points select the token\_f1 scorer. It extracts the last complete <answer>...</answer> span and returns zero when no complete span exists. With multiple reference answers, it takes the highest answer-level F1. Whitespace is first collapsed; strings containing a space are split on spaces, and strings without spaces are split into characters, including a single English word. Case and punctuation are preserved. For token-multiset overlap o and answer lengths $n _ { \mathrm { p r e d } } , n _ { \mathrm { r e f } }$ , the nonempty-answer form is

$$
\mathrm { P r e c } = \frac { o } { n _ { \mathrm { p r e d } } } , \qquad \mathrm { R e c } = \frac { o } { n _ { \mathrm { r e f } } } , \qquad \mathrm { F } 1 = \frac { 2 \mathrm { P r e c } \mathrm { R e c } } { \mathrm { P r e c } + \mathrm { R e c } } ,\tag{12}
$$

with zero F1 when the overlap is zero. Scoring is string-based and does not use a language-model judge; the original BrowseComp-Plus benchmark’s judge-based accuracy is a distinct metric.

Default returned score. The scorer’s default reward applies repetition, answer-length, and answer-tag penalties to token-F1. Thus a table obtained directly from that reward has the interpretation penalized token-F1 100:

$$
\mathrm { S c o r e } = { \frac { 1 0 0 } { N } } \sum _ { i = 1 } ^ { N } \mathrm { F } 1 _ { i } \ \rho _ { \mathrm { r e p } , i } \ \rho _ { \mathrm { l e n } , i } \ \rho _ { \mathrm { t a g } , i } .
$$

Here $\rho _ { \mathrm { r e p } }$ is the scorer’s repetition penalty. The length factor multiplies the score by 0.85 above 3,000 scoring tokens, and by a further 0.7 above 6,000. The tag factor is 0.25 if either the opening- or closing-answer-tag count exceeds ten, and one otherwise. Scoring tokens follow the string rule above and are distinct from model tokens. A 0–100 score is a scaled mean reward, not the percentage of questions answered correctly. An unpenalized token-F1 value instead requires aggregation of the F1 component before these factors; the default returned reward must not be interpreted as that component alone.

Retrieval and summarization. Each local\_search(query) call retrieves the top ten documents and uses a separate model to summarize them. Table 14 records the default settings. Both summarizers generate at most 256 model tokens with temperature 0.2 and top-p 0.9. BrowseComp Plus shortens the document and retries on summary failure, with an excerpt fallback. Its launcher uses the same BASE\_MODEL\_PATH variable for the actor base and summary service; overriding this variable can therefore change both roles. The listed models describe the repository defaults, not an invariant of the launcher under overrides.

Table 14 : Default search-tool settings. Character limits apply to the initial document excerpts; summary limits use model tokens.
<table><tr><td>Setting</td><td>ASearcher training/validation</td><td>BrowseComp Plus evaluation</td></tr><tr><td>Retrieval collection</td><td>Local wiki-18.jsonl</td><td>BrowseComp-Plus corpus</td></tr><tr><td>Retriever</td><td>e5-base-v2</td><td>Qwen3-Embedding-8B</td></tr><tr><td>Summary model</td><td>Qwen3-1.7B</td><td>Qwen3-8B; three services</td></tr><tr><td>Retrieved documents</td><td>Top 10 per call</td><td>Top 10 per call</td></tr><tr><td>Initial excerpt</td><td>2,000 characters/document</td><td>10,000 characters/document</td></tr><tr><td>Summary ceiling</td><td>256 model tokens</td><td>256 model tokens</td></tr><tr><td>Tool-return truncation</td><td>2,048 characters</td><td>2,048 characters</td></tr></table>

Trajectory budgets and evaluation sampling. The default entry points cap the prompt at 4,096 model tokens, the response or multi-turn trajectory at 35,000 token positions, and total model length at 40,000. The trajectory limit includes tool feedback. At most 100 assistant-generation rounds are permitted, with at most one tool call per round; the termination order allows at most 99 executed tool-call rounds, and length limits can stop a trajectory earlier. Tool outputs are truncated by string slicing at 2,048 characters. Training samples eight trajectories per question. Validation uses one trajectory per question, with temperature 0.7, top-p 0.8, and top-k 20; the training setting rollout.n=8 does not imply best-of-eight evaluation. These are execution ceilings and sampling settings, rather than measured equal token or tool-use costs across methods.

Reading the aggregate comparisons. Tables 2 and 19 report three-round score means and SDs; their paired diferences use corresponding evaluation-round indices. Those SDs describe variation across the reported rounds. The ASearcher column characterizes performance on the validation partition identified above, while BrowseComp Plus supplies the separate benchmark evaluation. Diferences are descriptive score comparisons; evaluation-round SDs do not themselves establish item-level statistical significance.

## A.8 Prompt construction and final-answer extraction

Prompt templates define the answer interface for each task. They are kept fixed within a comparison. GSM8K training and evaluation use a direct-answer text prompt. Mathematics tasks use a problem-solving prompt whose final answer is requested in a box. Multiple-choice tasks request the option letter directly. The task text is inserted at the indicated symbolic field below; these fields describe the template, rather than individual generated examples.

<table><tr><td>Solve the following grade-school math problem. Return only the final answer. Question: question text</td></tr><tr><td>Solve the following math problem carefully. End with the final answer in \boxed{...}.</td></tr><tr><td>Problem: problem text Solution:</td></tr><tr><td>question and option text</td></tr></table>

GSM8K reference answers are normalized from the final-answer field. The prediction parser first recognizes an explicit final-answer delimiter, then considers the region after a completed thinking segment and the output as a whole. Within a candidate region, it prioritizes an explicit final-answer line, followed by a final numeric candidate from non-step lines. Normalization removes commas, currency markers and whitespace and strips a trailing period. A mathematical answer parser selects the final expression and checks that expression against the reference answer.

For mathematical equivalence, the scorer first uses the parsed expression with the mathematical verifier when both sides parse. Its fallback normalizes basic LaTeX formatting and compares expressions symbolically; ordered tuples are compared component by component. The selected final answer is the object being judged. An intermediate occurrence of the reference expression elsewhere in the reasoning trace does not itself satisfy the final-answer criterion.

Answer generation stops at the first end-of-sequence token or at the task’s completion ceiling. A response at the ceiling is marked truncated. For a truncated response that remains inside an unfinished thinking segment without an explicit final answer, the score is incorrect. An explicit final-answer region, a box, or the designated final-answer markers allow the parser to evaluate the answer that was actually emitted. This rule treats termination and answer correctness as separate observables.

## A.9 Auxiliary metric definitions

IFEval evaluates each instruction constraint using its task-specific checker. If prompt i has $n _ { i }$ constraints with binary outcomes $z _ { i j }$ , its strict prompt score is

$$
I _ { i } = \prod _ { j = 1 } ^ { n _ { i } } z _ { i j } , \qquad \mathrm { S t r i c t @ 5 0 } = { \frac { 1 0 0 } { 5 0 } } \sum _ { i = 1 } ^ { 5 0 } I _ { i } .\tag{13}
$$

Every prompt therefore carries equal weight, independent of how many constraints it contains. The instructionlevel average is a diferent quantity because it weights individual constraints. The reported strict@50 metric makes satisfying the entire prompt the unit of evaluation.

WikiText-2 perplexity evaluates language-model likelihood. The evaluator joins nonempty paragraphs, tokenizes the text, and scores overlapping windows while masking context positions covered by preceding windows. Window length, stride, and token limit are evaluation configuration parameters. If a window ends at token $e _ { j }$ after a previous end $e _ { j - 1 }$ , its new span has length $t _ { j } = e _ { j } - e _ { j - 1 }$ . With model-returned masked causal loss $\ell _ { j }$ , the implementation uses

$$
\mathrm { P P L } = \exp \left( \frac { \sum _ { j } w _ { j } \ell _ { j } } { \sum _ { j } w _ { j } } \right) , \qquad w _ { j } = \operatorname * { m a x } ( t _ { j } - 1 , 1 ) .\tag{14}
$$

The first window supplies the initial context; each later window reuses context and scores its newly exposed sufix. Perplexity is retained on its own lower-is-better scale in Table 25.

Final WB is a whole-tensor ratio at the latest routed layer,

$$
\mathrm { F i n a l W B } = 1 0 0 \frac { \lVert g _ { \ell } \widehat { W } _ { \ell } \rVert _ { F } } { \lVert H _ { \ell } \rVert _ { F } } , \qquad \ell = 3 1 .\tag{15}
$$

The numerator is computed before the intervention is cast to the backbone dtype. Layer-wise execution overwrites the latest writeback statistic, which makes this diagnostic a description of the final receiving location. When all positions use active calibration, $( \mathrm { F i n a l W B / 1 0 0 } ) ^ { 2 }$ is a hidden-energy-weighted mean of squared token gates. It is consequently distinct from the arithmetic mean of gate coeficients. Appendix B.5 gives the corresponding identity and its interpretation.

## A.10 Analytical storage and projection work

The auxiliary live state has three principal components: completed compressed sources, current controller slots, and the active block anchor. Before the final receiving layers, their scalar counts per batch/token position are $7 r , K p ,$ and $d ,$ respectively. For the canonical dimensions this is $1 7 9 2 + 2 0 4 8 + 4 0 9 6 = 7 9 3 6$ fp32 scalars, or 31 KiB. This count describes those named tensors; temporary projection operands, backbone state, and diferentiation history have their own allocations.

Table 15 : Analytical storage of source bank, current slots, and anchor. Values use fp32 and $2 ^ { 2 0 }$ bytes per MiB. $B T$ is the number of positions processed concurrently in one forward call.
<table><tr><td>BT</td><td>Source bank (MiB)</td><td>Slots (MiB)</td><td>Anchor (MiB)</td><td>Sum (MiB)</td></tr><tr><td>1</td><td>0.0068</td><td>0.0078</td><td>0.0156</td><td>0.0303</td></tr><tr><td>128</td><td>0.875</td><td>1.000</td><td>2.000</td><td>3.875</td></tr><tr><td>512</td><td>3.500</td><td>4.000</td><td>8.000</td><td>15.500</td></tr><tr><td>2048</td><td>14.000</td><td>16.000</td><td>32.000</td><td>62.000</td></tr><tr><td>8192</td><td>56.000</td><td>64.000</td><td>128.000</td><td>248.000</td></tr></table>

Under reverse-mode diferentiation, earlier controller states and auxiliary operands participate in the graph. Keeping the controller history over $L$ updates has a leading $O ( B T L K p )$ scalar term. Repeated source projections additionally involve $O ( B T r \textstyle \sum _ { \ell } J _ { \ell } )$ operands. At inference, those gradient-history objects are absent, while current source and controller state remain. This distinction follows from the execution graph rather than from the allocated parameter count.

The two source-transport projection families have a particularly simple operation count. A dense d-to-r or r-to-d multiplication uses dr scalar multiply–accumulates per position. Eight block compressions and 28 writebacks therefore contribute

$$
8 d r + 2 8 d r = 3 6 d r = 3 7 , 7 4 8 , 7 3 6\tag{16}
$$

multiply–accumulates per batch/token position in the allocated execution. Omitting the terminal compression gives 35dr = 36, 700, 160. Routing queries, source key/value projections, controller updates, and slot attention add their respective matrix and reduction operations. These formulas expose which dimensions control the auxiliary work: r scales transport width, $K p$ scales live controller state, and s sets the number and spacing of source records.

Unlike a fixed linear weight update, the intervention depends on the receiving state, visible records, controller history, and calibrated direction. Its computation remains part of the forward pass after training. The storage and operation counts above therefore characterize the resources of an explicit communication interface: a small trainable footprint with transient per-position state and context-dependent computation.

## A.11 How architectural dimensions allocate capacity

The parameter count can be evaluated for other source ranks and block sizes without training a new model. This calculation separates architectural capacity from measured task quality. Let $M = L / s$ for a block size that divides L. With all layers targeted and the same dense controller structure, the allocated count

decomposes as

$$
\begin{array} { r l } & { P _ { \mathrm { t r a n s p o r t } } = ( M + L - s ) d r , } \\ & { \qquad P _ { \mathrm { r o u t e } } = r ( d + p + e ) + 2 r ( r + e ) , } \\ & { P _ { \mathrm { c o n t r o l l e r } } = 2 p d + p r + c ( 2 p + e ) + c + 2 p c + 2 p } \\ & { \qquad + p ( 2 p + e ) + 2 p ^ { 2 } + K p , } \\ & { P _ { \mathrm { i d e n t i t y } } = e ( L + M ) , \qquad P _ { \mathrm { g a t e } } = d + p + L - s . } \end{array}\tag{17}
$$

Their sum gives 41,730,332 at the canonical dimensions. The receiving and compression terms grow linearly with rank, while the source key/value term also has a quadratic component. Controller width is held fixed in Figure 5, which isolates the allocation changes induced by source rank and block size.

a Canonical allocation  
![](images/8361b1ca3313846f10e7cf23b309429e57bc0d0e551b11e2f5271a28b4c83baa.jpg)

b Rank and block-size accounting
<table><tr><td>Block</td><td>64</td><td>128</td><td>256</td><td>512</td><td>Pairs</td></tr><tr><td>1</td><td>19.47</td><td>36.31</td><td>70.04</td><td>137.70</td><td>496</td></tr><tr><td>2</td><td>15.01</td><td>27.40</td><td>52.22</td><td>102.05</td><td>240</td></tr><tr><td>4</td><td>12.39</td><td>22.16</td><td>41.73</td><td>81.08</td><td>112</td></tr><tr><td>8</td><td>10.30</td><td>17.96</td><td>33.34</td><td>64.30</td><td>48</td></tr><tr><td>16</td><td>7.67</td><td>12.72</td><td>22.86</td><td>43.33</td><td>16</td></tr></table>

Source / transport rank

Figure 5 : Analytical allocation of adaptation capacity. Left: area-proportional component allocation in the canonical module. Right: Equation 17 over source ranks and block sizes, with L = 32, d = 4096, K = 8, p = c = 256, and e = 32. Cells give parameters in millions; the outlined cell is canonical. The last column counts source–receiver pairs.

A smaller block size creates more sources and enables earlier writeback, increasing transport allocation and source–receiver pairs, $s M ( M - 1 ) / 2$ . A larger block compresses a longer interval into each record, reducing the resolution of the addressable history.

Memory rank sets the width of each stored change and the maximal receiving subspace dimension. Controller width sets the capacity used to select among records. Their equal canonical width, $r = p = 2 5 6$ , does not make these roles interchangeable; Figure 5 quantifies the resulting allocation.

## B Formal analysis of cross-layer communication

This section analyzes the interface through which T-Router reuses completed computations. We first establish its event ordering and the content of its block records. We then characterize the receiving subspace, the history represented by controller slots, the geometry of calibrated interventions, and their propagation through frozen layers. The final part derives the diferentiated training objective and the distribution induced by bounded informative resampling. These results concern the specified computation in exact arithmetic; casts to the backbone dtype occur after the calibrated intervention is formed.

## B.1 Notation and completed-block event ordering

We use sequence tensors $H _ { \ell } \in \mathbb { R } ^ { B \times T \times d }$ and token vectors $h _ { \ell , t } \in \mathbb { R } ^ { d }$ , suppressing the batch index unless a reduction requires it. The norm of a sequence tensor is its Frobenius norm. Each $\boldsymbol { B _ { \ell } }$ is a complete frozen decoder layer, including causal token mixing and its internal residual connections. Auxiliary projections and attentions are tokenwise. The K controller slots for one token form the rows of $\mathbf { S } _ { \ell , t } \in \mathbb { R } ^ { K \times p }$ . Layer indices start at zero; $\mathbf { S } _ { - 1 , t } = \mathbf { S } ^ { 0 }$ denotes the learned initial state.

For block size s, the events at depth ℓ occur in the order

$$
\mathrm { r e a d \ c o m p l e t e d \ r e c o r d s } \ \longrightarrow \ \mathrm { u p d a t e \ a n d \ r e a d \ c o n t r o l l e r } \ \longrightarrow \ R _ { \ell } ,
$$

$$
\widetilde { H } _ { \ell } = H _ { \ell } + R _ { \ell } \longrightarrow H _ { \ell + 1 } = { \mathcal { B } } _ { \ell } ( \widetilde { H } _ { \ell } ) \longrightarrow \mathrm { a p p e n d i f t h e b l o c k e n d s } .\tag{18}
$$

An anchor $\widetilde { H } _ { s b }$ is associated with the first actual input of block b. Its compressed record is appended after layer $s ( b + 1 ) - 1$ finishes. Thus the readable set before layer ℓ is

$$
\mathcal { I } _ { \ell } = \{ 0 , \dots , \lfloor \ell / s \rfloor - 1 \} , \qquad n _ { \ell } : = | \mathcal { I } _ { \ell } | = \lfloor \ell / s \rfloor ,\tag{19}
$$

with the set empty when $\ell < s$ . This formula assumes the full-history configuration and $s \mid L .$ . It distinguishes storing a record from using it: block b has exactly $L - s ( b + 1 )$ receiving layers, when this quantity is positive.

Proposition 1 (Causal source availability). Assume each frozen layer is causal over token positions. Before layer ℓ, the residual, controller slots, and visible records at position t depend only on input positions at most t. Routing reads only completed blocks and does not depend on the output of its receiving layer.

Proof. At the first layer the input embeddings are causal, the slots are copied from an input-independent parameter array, and the bank is empty. Suppose the claim holds before layer ℓ. Every auxiliary operation at position t uses $h _ { \ell , t }$ , slots at that position, and compressed records at that position. Linear maps, nonlinearities, slot attention, source attention, and RMS calibration therefore preserve the same token support. Applying the causal map $\boldsymbol { B _ { \ell } }$ also preserves it. A new record, if created, is a tokenwise linear map of the diference between this output and an earlier causal anchor. Equation 18 places that append after the receiving decision, so it is available only to later layers. Induction establishes both token causality and depth causality. □

The argument also explains the state lifetime. A new forward pass initializes fresh slots and an empty bank. In cached decoding, earlier tokens influence the current token through the backbone’s causal cache; the auxiliary state evolves along the current pass’s depth. Under exact arithmetic, identical position/mask semantics, and cache-equivalent frozen layers, teacher-forced and cached evaluations construct the same token-local auxiliary state for the same prefix. This follows by applying the induction to the current token at each depth. It does not require persistent controller slots across generated tokens.

For $L / s = m$ complete blocks, the total number of source–receiver pairs is

$$
\sum _ { \ell = 0 } ^ { L - 1 } n _ { \ell } = s \sum _ { j = 0 } ^ { m - 1 } j = { \frac { s m ( m - 1 ) } { 2 } } .\tag{20}
$$

The canonical configuration has 112 pairs. The terminal record has no consumer; deleting its append leaves $H _ { L }$ and the training loss unchanged. This is an execution identity rather than an approximation.

## B.2 What a block change contains

Write $B _ { j } ( X ) = X + F _ { j } ( X )$ , where $F _ { j }$ includes all computations within the complete layer beyond its outer identity term. This is an algebraic decomposition of the frozen layer, not a replacement for its internal architecture. The actual residual transition is

$$
H _ { j + 1 } - H _ { j } = R _ { j } + F _ { j } ( \widetilde { H } _ { j } ) .\tag{21}
$$

Summing across block b and subtracting the post-intervention anchor gives

$$
\begin{array} { l } { { \displaystyle D _ { b } = H _ { s ( b + 1 ) } - \widetilde { H } _ { s b } } } \\ { { \displaystyle ~ = ~ \sum _ { j = s b } ^ { s ( b + 1 ) - 1 } F _ { j } ( \widetilde { H } _ { j } ) + \sum _ { j = s b + 1 } ^ { s ( b + 1 ) - 1 } R _ { j } . } } \end{array}\tag{22}
$$

Indeed, the telescoping sum first includes $R _ { s b }$ , and the subtraction of $\widetilde { H } _ { s b } = H _ { s b } + R _ { s b }$ removes precisely ethat term. All later injections within the block remain. For a two-layer block, for example, $D _ { b } = F _ { 2 b } ( \widetilde { H } _ { 2 b } ) +$ $R _ { 2 b + 1 } + F _ { 2 b + 1 } ( \widetilde { H } _ { 2 b + 1 } )$

The first injection can still influence every $F _ { j }$ through its input. Equation 22 excludes its explicit additive term, not its downstream computational efect. The record therefore describes the change realized by the intervened block. It is neither a separately evaluated unmodified-backbone change nor a causal attribution to an isolated layer. This distinction makes the record usable during one forward pass: subsequent routing operates on computations the model has actually completed.

Two immediate consequences are useful. First, compression yields

$$
\| m _ { b } \| _ { F } \leq \| C _ { b } \| _ { 2 } \left( \sum _ { j = s b } ^ { s ( b + 1 ) - 1 } \| F _ { j } ( \widetilde { H } _ { j } ) \| _ { F } + \sum _ { j = s b + 1 } ^ { s ( b + 1 ) - 1 } \| R _ { j } \| _ { F } \right) .\tag{23}
$$

Second, both endpoints remain diferentiable. If $G _ { b } = \partial \mathcal { L } / \partial m _ { b }$ denotes the adjoint from all later uses, the direct contribution to the compressor gradient is $\begin{array} { r } { \nabla _ { C _ { b } } \mathcal { L } = \sum _ { n , t } G _ { b , n , t } D _ { b , n , t } ^ { \top } , } \end{array}$ and the adjoint entering $D _ { b }$ is $C _ { b } ^ { \top } G _ { b }$ . The endpoint diference distributes that adjoint with opposite signs to output and anchor, after which ordinary reverse-mode diferentiation follows their shared earlier computation. Shared ancestors are handled by summing these paths, not by treating the two endpoints as independent model executions.

## B.3 Source transport and the receiving subspace

Partition the routing value projection into content and identity terms, $W _ { v } = [ W _ { v } ^ { m } \ W _ { v } ^ { e } ]$ . For one receiving layer and token, the raw direction is

$$
w _ { \ell , t } = \sum _ { b \in \mathcal { I } _ { \ell } } a _ { \ell , t , b } \left[ \underbrace { U _ { \ell } W _ { v } ^ { m } C _ { b } } _ { T _ { \ell b } } D _ { b , t } + U _ { \ell } W _ { v } ^ { e } e _ { b } ^ { \mathrm { s r c } } \right] , \quad \mathrm { r a n k } ( T _ { \ell b } ) \leq r .\tag{24}
$$

Conditioning on the attention weights, this is afine in the collection of block changes. The content transports $T _ { \ell b }$ can difer across sources through $C _ { b } .$ , while all terms share the receiving projection $U _ { \ell }$ . In particular, the concatenated content transport $[ T _ { \ell 0 } T _ { \ell 1 } \cdots ]$ also has rank at most $r ,$ rather than the sum of the individual rank bounds.

Proposition 2 (Aggregate receiving subspace). At fixed parameters and receiving layer $\ell ,$ every intervention lies in col $( U _ { \ell } )$

$$
\operatorname { s p a n } \{ R _ { \ell , t } : i n p u t s \ a n d \ p o s i t i o n s \ t \} \subseteq \cot ( U _ { \ell } ) , \qquad \operatorname { d i m } \cot ( U _ { \ell } ) \leq r .\tag{25}
$$

This holds for both branches of RMS calibration and for signed scalar gates.

Proof. Source attention first produces $c _ { \ell , t } \in \mathbb { R } ^ { r }$ . Calibration and gating multiply $U _ { \ell } c _ { \ell , t }$ by a scalar $\alpha _ { \ell , t } .$ , with $\alpha = g \mathrm { R M S } ( h ) / \mathrm { R M S } ( w )$ on the active branch and $\alpha = g$ on the fallback branch. Thus $R _ { \ell , t } = U _ { \ell } ( \alpha _ { \ell , t } c _ { \ell , t } )$ Every such vector belongs to the same column space, including zero vectors and source-identity contributions. Taking a span proves the result. □

There is a corresponding diferential statement. Let x collect any continuous upstream inputs to a token’s routing decision, and write $f _ { \ell } ( x ) = \alpha _ { \ell } ( x ) c _ { \ell } ( x )$ . At any diferentiable point away from a calibration boundary,

$$
\frac { \partial R _ { \ell } } { \partial x } = U _ { \ell } \frac { \partial f _ { \ell } } { \partial x } , \qquad \mathrm { r a n k } \biggl ( \frac { \partial R _ { \ell } } { \partial x } \biggr ) \leq r .\tag{26}
$$

Adaptive attention changes the coeficient derivative but not its receiving column space. Holding the auxiliary history fixed and diferentiating the layer-input interface gives $\partial \widetilde { h } _ { \ell } / \partial h _ { \ell } = I + U _ { \ell } \partial f _ { \ell } / \partial h _ { \ell } .$ . This locates the elow-dimensional modification at the interface itself. The subsequent nonlinear $\boldsymbol { B _ { \ell } }$ can transform it, and comparing whole-layer Jacobians at two diferent inputs introduces the change in the backbone Jacobian as well.

Addressable records and coeficient choice. For fixed visible values $v _ { b } = { W _ { v } } [ m _ { b } ; e _ { b } ^ { \mathrm { s r c } } ]$ , dense softmax gives $c \in \mathrm { c o n v } \{ v _ { b } \}$ ; with finite logits all weights are positive. At the raw interface w lies in the corresponding convex hull of $U _ { \ell } v _ { b }$ . Calibration preserves the retrieved direction, while the signed gate sets the final magnitude and sign. Thus dense attention permits a state-dependent mixture without implying sparse execution or omission of a frozen layer. When there is one record, $a _ { 0 } = 1$ and the attention-logit derivative is zero, although the content/value path is still trainable. With additional records, the controller and current state can change the mixture within the same receiving subspace.

The interface is generally nonlinear in its input: attention weights, controller history, normalization, and gating all depend on that input. A fixed rank-r transport and an input-dependent rank-r interface share a dimensional constraint but need not implement the same function. For example, with one nonzero value and a scalar gate depending on $h , R ( h ) = g ( h ) \widehat { w } ( h )$ already varies nonlinearly. It cannot in general be absorbed into one constant additive weight matrix.

## B.4 Controller history, rank, and state bounds

Suppressing the token index, define $\nu _ { \ell } = \sigma ( v _ { \ell } ) \odot \operatorname { t a n h } ( u _ { \ell } ) \in ( - 1 , 1 ) ^ { p }$ and let $\omega _ { \ell }$ be the slot-selection probability vector. We take $0 \leq \gamma < 1$ , with $\gamma = 0 . 9$ in the canonical configuration. The row-stacked update is

$$
\begin{array} { c } { { \displaystyle { \bf S } _ { \ell } = \gamma { \bf S } _ { \ell - 1 } + ( 1 - \gamma ) \omega _ { \ell } \nu _ { \ell } ^ { \top } } } \\ { { \displaystyle = \gamma ^ { \ell + 1 } { \bf S } ^ { 0 } + ( 1 - \gamma ) \sum _ { j = 0 } ^ { \ell } \gamma ^ { \ell - j } \omega _ { j } \nu _ { j } ^ { \top } } . } \end{array}\tag{27}
$$

The second equality follows by substitution, starting from $\mathbf { S } _ { - 1 } = \mathbf { S } ^ { 0 }$ . It is valid even though $\omega _ { j }$ and $\nu _ { j }$ depend on the realized preceding states. Unrolling a trajectory is an algebraic operation; it does not assume independent updates.

Rank and distinct history mixtures. Each added matrix has rank at most one, so subadditivity gives

$$
\mathrm { r a n k } ( { \bf S } _ { \ell } ) \leq \operatorname* { m i n } \{ K , p , \mathrm { r a n k } ( { \bf S } ^ { 0 } ) + \ell + 1 \} .\tag{28}
$$

To see why the state need not remain rank one, take $K = p = 2 , \mathbf { S } ^ { 0 } = ( 1 , 2 ) ^ { \top } e _ { 1 } ^ { \top }$ , scaled selector $W _ { s } z / \sqrt { p } = e _ { 1 }$ and a nonzero proposal proportional to $e _ { 2 }$ . The selector produces ω = softmax(1, 2), whose entries are not in the ratio 1 : 2. Therefore the two columns of $\gamma ( 1 , 2 ) ^ { \top } e _ { 1 } ^ { \top } + ( 1 - \gamma ) \omega \nu ^ { \top }$ are linearly independent for $0 < \gamma < 1$ A rank-one initial state becomes rank two in one valid shared-proposal update. Both proposal coordinates and selector scale can be chosen within the specified parameterization.

There are also exact symmetry cases. If all initial rows are equal, their slot logits are equal, so $\omega _ { k } = 1 / K$ . The shared proposal preserves equal rows; by induction, the slots remain identical. If $\omega _ { j }$ is a fixed vector at every depth and the initial state has the same row factor, the complete history also has that row factor and rank at most one. These cases show what produces multiple slot mixtures: distinct initial rows and depth-dependent slot allocation, together with diferent proposal directions. The learned initialization allows this asymmetry without assigning a prescribed semantic role to any slot.

For a fixed trajectory, substituting Equation 27 into the attention readout (Equation 5) expresses the controller context in terms of layer-wise proposals:

$$
P _ { \ell } = \gamma ^ { \ell + 1 } \sum _ { k } \xi _ { \ell , k } V _ { s } S _ { k } ^ { 0 } + ( 1 - \gamma ) \sum _ { j = 0 } ^ { \ell } \gamma ^ { \ell - j } \underbrace { ( \xi _ { \ell } ^ { \top } \omega _ { j } ) } _ { \mathrm { r e a d - w r i t e ~ o v e r l a p } } V _ { s } \nu _ { j } .\tag{29}
$$

The scalar overlap is between zero and one. Each past proposal contributes according to both its earlier slot assignment and the current slot read. Consequently, a decay average of the proposals alone does not generally determine $P _ { \ell } \colon$ the history of slot assignments also matters. At the same time, the slots are finite-dimensional mixtures of that history, not an unbounded list of separately addressable depth records.

Forward bounds and conditional sensitivity. The bounded proposal and probability selector imply

$$
\sum _ { k } | | \mathbf { S } _ { \ell } | | _ { F } \leq \gamma ^ { \ell + 1 } \| \mathbf { S } ^ { 0 } \| _ { F } + ( 1 - \gamma ^ { \ell + 1 } ) \sqrt { p } ,\tag{30}
$$

For the first line, $\| \omega \nu ^ { \top } \| _ { F } = \| \omega \| _ { 2 } \| \nu \| _ { 2 } \leq \sqrt { p } ;$ apply the triangle inequality to the unrolled history. For the second, sum absolute values over slots and use $\textstyle \sum _ { k } \omega _ { k } = 1$ . Since readout attention is a probability vector, $\| P _ { \ell } \| _ { 2 } \leq \| V _ { s } \| _ { 2 }$ max<sub>k</sub> $\| S _ { \ell , k } \| _ { 2 }$ . These bounds apply to forward values for any inputs and selector logits.

Sensitivity involves the selector as well as $\gamma$ . Hold $z _ { \ell }$ fixed and write $a = W _ { s } z _ { \ell } / \sqrt { p } , \omega = \mathrm { s o f t m a x } ( \mathbf { S } a )$ . The diferential of one update with respect to the previous slots is

$$
\Delta \mathbf { S } ^ { + } = \gamma \Delta \mathbf { S } + ( 1 - \gamma ) ( \mathrm { d i a g } \omega - \omega \omega ^ { \top } ) ( \Delta \mathbf { S } a ) \nu ^ { \top } .\tag{31}
$$

The softmax Jacobian has operator norm at most $1 / 2 { : }$ its quadratic form is the variance of the entries of a vector under $\omega ,$ at most one quarter of their squared range, and that squared range is at most twice the vector’s squared norm. Hence

$$
\| \Delta \mathbf { S } ^ { + } \| _ { F } \leq \left[ \gamma + \frac { 1 - \gamma } { 2 } \| a \| _ { 2 } \| \nu \| _ { 2 } \right] \| \Delta \mathbf { S } \| _ { F } .\tag{32}
$$

This is a contraction bound when $\| a \| _ { 2 } \| \nu \| _ { 2 } < 2$ with fixed $z _ { \ell } .$ . In the full model $z _ { \ell }$ also depends on the current residual and visible records, so its derivative supplies additional paths. The decay factor controls historical coeficients; it is not by itself the Jacobian norm of the complete controller.

## B.5 RMS geometry, gradients, and threshold behavior

For this subsection suppress layer and token indices, put $q = \mathrm { R M S } ( w )$ and $\kappa = \mathrm { R M S } ( h )$ , and define the calibrated map $f _ { \kappa } ( w ) = \widehat { w }$ from Equation 7. On the active branch $q > \epsilon , \mathrm { R M S } ( x ) = \| x \| _ { 2 } / \sqrt { d }$ gives

$$
f _ { \kappa } ( w ) = \| h \| _ { 2 } { \frac { w } { \| w \| _ { 2 } } } , \qquad \| f _ { \kappa } ( w ) \| _ { 2 } = \| h \| _ { 2 } .\tag{33}
$$

Calibration maps the raw direction to the sphere whose radius is the receiving state’s norm. Positive rescaling of w cancels while both points stay on this branch; negative rescaling reverses the direction. Multiplication by the signed gate gives $\| \boldsymbol { R } \| _ { 2 } / \| \boldsymbol { h } \| _ { 2 } = | \boldsymbol { g } |$ for $h \neq 0$ . This explains how gate magnitude can be interpreted independently of the scale of the compression and projection matrices.

Tangential and radial derivatives. Holding the reference scale fixed, diferentiation gives

$$
J _ { w } : = \frac { \partial f _ { \kappa } ( w ) } { \partial w } = \frac { \kappa } { q } \left( I - \frac { w w ^ { \top } } { \| w \| _ { 2 } ^ { 2 } } \right) , \quad J _ { w } w = 0 , \quad \| J _ { w } \| _ { 2 } = \frac { \kappa } { q } \quad ( d > 1 ) .\tag{34}
$$

To derive it, $\mathrm { d } q = w ^ { \top } \mathrm { d } w / ( d q )$ , where d is the vector dimension. Applying the product rule to $\kappa w / q$ yields $\mathrm { d } f = ( \kappa / q ) \mathrm { d } w - \kappa w ( w ^ { \top } \mathrm { d } w ) / ( d q ^ { 3 } )$ ; since $d q ^ { 2 } = \| w \| _ { 2 } ^ { 2 }$ , Equation 34 follows. The matrix in parentheses is the orthogonal projector onto the tangent space perpendicular to w. Its radial eigenvalue is zero and its $d - 1$ tangential eigenvalues are one. Thus calibration learns direction through tangential gradients; the signed gate supplies a separate radial degree of freedom in the final intervention.

a Direction at the receiving scale

![](images/f9bbf253a8bf575a97bc41c8f234f3303ba0dd49b3a21ef0e312bbbd1f9021c9.jpg)  
b Local derivative

![](images/44eefaecd77f5d5c072c6b82ceeef0cbbe9b29151afe74c2838d4c91f9761980.jpg)  
Figure 6 : Analytical geometry of RMS calibration. Left: in two dimensions with $\| h \| = 1$ , positive multiples on the same ray map to one point on the target sphere; the double arrow marks a tangent. The local basis separates zero radial response from tangential gain $\kappa / q .$ . Right: the active-branch Jacobian eigenvalues from Equation 34. The gate subsequently sets signed amplitude.

Figure 6 separates three geometric objects: the ray selected by the raw projection, the sphere set by the receiving residual, and the local derivative along or across that ray. A positive radial change of the raw vector leaves its calibrated output fixed. A tangential change rotates the direction, with sensitivity determined by $\kappa / q .$ The gate’s derivative supplies the remaining scalar control of the final intervention. This geometry connects the parameterization to the separate source, transport, and strength coordinates discussed in Appendix D.

Let $a = \partial \mathcal { L } / \partial R$ be an incoming adjoint and temporarily hold routing inputs fixed. Then

$$
\frac { \partial \mathcal { L } } { \partial w } = g J _ { w } a , \qquad \frac { \partial \mathcal { L } } { \partial g } = a ^ { \top } f _ { \kappa } ( w ) .\tag{35}
$$

The normalization derivative can have gain greater than one when the raw RMS is small relative to the reference RMS. Meanwhile, for gate logit z $, \ \partial g / \partial z = g _ { \mathrm { { m a x } } } { \mathrm { s e c h } } ^ { 2 } z \leq g _ { \mathrm { { m a x } } }$ . These two derivatives control diferent quantities: the first changes the direction, while the second changes its signed coeficient.

Table 16 : Calibration regimes for fixed κ. Values and derivatives refer to the piecewise mathematical map before the final dtype cast.
<table><tr><td></td><td>Region Returned vector Output norm</td><td></td><td>Derivative in w</td></tr><tr><td> $q > \epsilon$ </td><td> $( \kappa / q ) w$ </td><td> $\sqrt { d } \kappa$ </td><td>Tangential projector  $\times \kappa / q$ </td></tr><tr><td> $q < \epsilon$ </td><td>w</td><td> $\sqrt { d } q$ </td><td>I</td></tr><tr><td> $q = \epsilon$ </td><td>w</td><td> $\sqrt { d } \epsilon$ </td><td>Boundary; see one-sided limits</td></tr></table>

The threshold and a branch-aware bound. For a unit vector $v ,$ the fallback value at $w = \sqrt { d } \epsilon \boldsymbol { v }$ is $\sqrt { d } \epsilon v$ , while the active-side limit is $\sqrt { d } \kappa v$ . Their diference has norm $\sqrt { d } | \kappa - \epsilon |$ . The forward map is continuous at this boundary only if $\kappa = \epsilon ;$ even then the radial one-sided derivatives difer. Inside the fallback region the map is the identity. These properties follow from the explicit branch rule, which returns the unnormalized vector below the threshold rather than dividing it by ϵ.

For example, an analytical illustration with $\kappa = 1$ and $\epsilon = 1 0 ^ { - 6 }$ has active-side tangential gain approaching $1 0 ^ { 6 }$ , while its active output RMS remains one. The two numbers describe derivative gain and forward scale respectively. For a complete sequence, let and $\mathcal { F }$ be the active and fallback token positions. Since $| g | < g _ { \mathrm { m a x } } .$

$$
\| R _ { \ell } \| _ { F } ^ { 2 } \leq g _ { \operatorname* { m a x } } ^ { 2 } \left( \sum _ { ( n , t ) \in \mathcal { A } } \| h _ { \ell , n , t } \| _ { 2 } ^ { 2 } + | \mathcal { F } | d \epsilon ^ { 2 } \right) .\tag{36}
$$

The fallback has a small absolute bound; its relative bound also depends on the receiving norm. When every position is active and $\| H _ { \ell } \| _ { F } > 0$ , the squared norm ratio satisfies

$$
\frac { \| R \ell \| _ { F } ^ { 2 } } { \| H _ { \ell } \| _ { F } ^ { 2 } } = \sum _ { n , t } \frac { \| h _ { \ell , n , t } \| _ { 2 } ^ { 2 } } { \| H _ { \ell } \| _ { F } ^ { 2 } } g _ { \ell , n , t } ^ { 2 } .\tag{37}
$$

Thus the norm ratio is the square root of a hidden-energy-weighted mean of squared gates.

## B.6 Stop-gradient and scale-invariant optimization

The reference RMS is detached when forming $f _ { \kappa }$ . Its forward value still tracks $h ,$ but its reverse-mode derivative is zero. On the active branch $q > \epsilon ,$ for $h \neq 0$ and with w held fixed, the mathematical forward map without detachment would contribute the additional Jacobian

$$
\frac { \partial f _ { \mathrm { R M S } ( h ) } ( w ) } { \partial h } = \frac { w } { q } \frac { h ^ { \top } } { d \kappa } .\tag{38}
$$

Detachment removes this reference-scale edge. It leaves the identity path through $\widetilde { h } = h + R$ , the dependence of w on h and past computations, and the dependence of $g$ eon h and P intact. On the active branch the diferentiated writeback is therefore $d R = f _ { \kappa } ( w ) d g + g J _ { w } d w .$ , with no dκ term.

Memory construction is not detached. A later loss can reach an earlier intervention through the ordinary residual chain, the compressed source record, and the controller that reads that record. Freezing a layer’s parameters preserves its input vector–Jacobian product:

$$
\frac { \partial \mathcal { L } } { \partial \widetilde { H } _ { \ell } } = J _ { \mathcal { B } _ { \ell } } ( \widetilde { H } _ { \ell } ) ^ { \top } \frac { \partial \mathcal { L } } { \partial H _ { \ell + 1 } } \quad \mathrm { f o r ~ t h e ~ b a c k b o n e ~ e d g e } .\tag{39}
$$

Other outgoing edges add their adjoints to this expression. This is also why training earlier LoRA modules diferentiates through subsequent frozen operators. Parameter freezing removes weight-gradient and optimizerstate requirements; it does not replace reverse-mode propagation by a local update at each insertion point.

Positive scaling of a receiving projection. Suppose all uses of $U _ { \ell }$ remain on the active branch in a neighborhood of a positive rescaling, and the loss has no explicit norm penalty on $U _ { \ell } .$ . Replacing $U _ { \ell }$ by $a U _ { \ell } , a > 0$ , scales the raw w by a and leaves every calibrated intervention unchanged. Downstream residuals, records, and losses are consequently unchanged as well. Diferentiating this invariance at $a = 1$ gives

$$
\langle \nabla _ { U _ { \ell } } \mathcal { L } , U _ { \ell } \rangle _ { F } = 0 .\tag{40}
$$

This concerns the Euclidean gradient of the diferentiated loss. AdamW adds momentum, coordinatewise preconditioning, and decoupled decay. If V is its preconditioned moment direction, η its learning rate, and $a = 1 - \eta \lambda _ { \mathrm { w d } }$ , the actual norm update is

$$
U ^ { + } = a U - \eta V , \qquad \| U ^ { + } \| _ { F } ^ { 2 } = a ^ { 2 } \| U \| _ { F } ^ { 2 } - 2 a \eta \langle U , V \rangle _ { F } + \eta ^ { 2 } \| V \| _ { F } ^ { 2 } .\tag{41}
$$

Raw-gradient orthogonality does not force $\langle U , V \rangle _ { F }$ to vanish. Even when it vanishes, the last term is positive. For a concrete two-coordinate illustration, $U = ( 1 , 2 )$ and raw gradient $( 2 , - 1 )$ have zero inner product; a first-step Adam direction close to $( 1 , - 1 )$ has inner product 1 with U. Reversing the raw gradient reverses that inner product. The projection norm can therefore increase or decrease under the full update. Decay contracts its own component; it does not establish a monotone trajectory for the total norm.

## B.7 Conditional propagation through the frozen decoder

Local calibration supplies an intervention budget at each receiving layer. Its efect at a later depth depends on the frozen computation between those layers. To make this dependence explicit, compare an intervened trajectory $H _ { \ell }$ with the adapter-disabled trajectory $H _ { \ell } ^ { 0 }$ , starting from the same $H _ { 0 }$ . Assume $B _ { \ell }$ is Λ<sub>ℓ</sub>-Lipschitz on a domain containing the two actual layer inputs. Put $e _ { \ell } = \| H _ { \ell } - H _ { \ell } ^ { 0 } \| _ { F }$ . Then

$$
\begin{array} { l } { { \displaystyle e _ { \ell + 1 } \leq \Lambda _ { \ell } \| H _ { \ell } + R _ { \ell } - H _ { \ell } ^ { 0 } \| _ { F } } } \\ { { \displaystyle \quad \leq \Lambda _ { \ell } ( e _ { \ell } + \| R _ { \ell } \| _ { F } ) } , } \\ { { \displaystyle e _ { L } \leq \sum _ { j = 0 } ^ { L - 1 } \left( \prod _ { k = j } ^ { L - 1 } \Lambda _ { k } \right) \| R _ { j } \| _ { F } } . } \end{array}\tag{42}
$$

The final inequality follows by repeatedly substituting the preceding one and using $e _ { 0 } = 0$ . Crucially, it treats each $R _ { j }$ as the intervention realized on the actual trajectory. Routing can depend on the entire available auxiliary history; no assumption that $R _ { j }$ is a function of $H _ { j }$ alone is needed.

When all positions use active calibration and $| g _ { j , n , t } | \le \rho _ { j }$ , we have $\| R _ { j } \| _ { F } \le \rho _ { j } \| H _ { j } \| _ { F }$ . One can either substitute this directly into Equation 42, or eliminate the intervened norm using $\lVert H _ { j } \rVert _ { F } \leq \lVert H _ { j } ^ { 0 } \rVert _ { F } + e _ { j }$ . The latter gives

$$
e _ { L } \leq \sum _ { j = 0 } ^ { L - 1 } \Lambda _ { j } \rho _ { j } \| H _ { j } ^ { 0 } \| _ { F } \prod _ { k = j + 1 } ^ { L - 1 } \Lambda _ { k } ( 1 + \rho _ { k } ) .\tag{43}
$$

The empty product is one. This follows from the scalar recurrence $e _ { j + 1 } \leq \Lambda _ { j } ( 1 + \rho _ { j } ) e _ { j } + \Lambda _ { j } \rho _ { j } \| H _ { j } ^ { 0 } \| _ { F }$ . Fallback positions can instead be handled with the absolute budget in Equation 36.

For an analytical illustration, if the realized intervention norm is at most δ at each of n successive layers and $\Lambda _ { j } \leq \Lambda$ , Equation 42 becomes $\delta \sum _ { k = 1 } ^ { n } \Lambda ^ { k }$ . It equals nδ when $\Lambda = 1$ , is at most $\delta \Lambda / ( 1 - \Lambda )$ when $\Lambda < 1$ and grows geometrically with n when $\Lambda > 1$ . Thus the same local budget can have diferent downstream efects depending on the intervening maps. These conditional bounds explain why local relative scale and final-output change are separate quantities.

There is an analogous first-order description. Introduce a scalar perturbation amplitude τ with $R _ { j } ( \tau ) =$ $\tau V _ { j } + o ( \tau ) , \ R _ { j } ( 0 ) = 0$ , and diferentiable frozen layers at the unperturbed trajectory. Diferentiating $H _ { j + 1 } ( \tau ) = \mathcal { B } _ { j } ( H _ { j } ( \tau ) + R _ { j } ( \tau ) )$ gives

$$
\dot { H } _ { L } = \sum _ { j = 0 } ^ { L - 1 } J _ { L - 1 } J _ { L - 2 } \cdot \cdot \cdot J _ { j } V _ { j } , \qquad J _ { j } = J _ { \mathcal { B } _ { j } } ( H _ { j } ^ { 0 } ) .\tag{44}
$$

Each interface direction is transported by the frozen Jacobians that follow it. The product acts on a lowdimensional injected vector, while diferent layers can inject through diferent receiving subspaces. This is the precise sense in which changing cross-layer communication can alter later computation without changing the pretrained operators themselves.

## B.8 The differentiated group-relative objective

For a retained group of $G = 4$ completions, let $\mathcal { T } _ { i }$ contain its scored completion positions and $T _ { i } = \operatorname* { m a x } \{ 1 , | T _ { i } | \}$ Prompt and padding positions are excluded from $\mathcal { T } _ { i } .$ Let V omit completions that are simultaneously truncated

and unparseable; other incorrect completions are valid and receive binary reward zero. For $m = | V | > 0$ , write

$$
\bar { r } = \frac { 1 } { m } \sum _ { i \in V } r _ { i } , \quad \sigma ^ { 2 } = \frac { 1 } { m } \sum _ { i \in V } ( r _ { i } - \bar { r } ) ^ { 2 } , \quad A _ { i } = \left\{ \begin{array} { l l } { ( r _ { i } - \bar { r } ) / ( \sigma + 1 0 ^ { - 6 } ) , } & { i \in V , \sigma > 1 0 ^ { - 6 } , } \\ { r _ { i } - \bar { r } , } & { i \in V , \sigma \leq 1 0 ^ { - 6 } , } \\ { 0 , } & { i \notin V . } \end{array} \right.\tag{45}
$$

For $V = \emptyset$ , all advantages and the policy term are zero. Centering gives $\textstyle \sum _ { i \in V } A _ { i } = 0$ . For mixed binary rewards with c correct among m valid completions, $\sigma ^ { 2 } = ( c / m ) ( 1 - c / m )$ . Ignoring the displayed additive $1 0 ^ { - 6 }$ only for this interpretation, correct and incorrect advantages are $\sqrt { ( m - c ) / c }$ and $- { \sqrt { c / ( m - c ) } }$ respectively. With the additive term retained their magnitudes are slightly smaller, and $| A _ { i } | \leq \sqrt { m - 1 } \leq \sqrt { 3 }$ . Uniform valid rewards produce zero advantages.

Put $\ell _ { i , t } = \log \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid x , y _ { i , < t } )$ and let $\ell _ { i , t } ^ { 0 }$ be the corresponding adapter-disabled log probability. The diferentiated surrogate uses a detached copy from the current evaluation,

$$
\begin{array} { c l } { \displaystyle q _ { i , t } = \exp ( \ell _ { i , t } - \mathrm { s g } ( \ell _ { i , t } ) ) , } \\ { \displaystyle \mathcal { L } _ { \mathrm { p o l i c y } } = - \frac { 1 } { m } \sum _ { i \in V } \frac { 1 } { T _ { i } } \sum _ { t \in \mathcal { T } _ { i } } \operatorname* { m i n } ( q _ { i , t } A _ { i } , \mathrm { c l i p } ( q _ { i , t } , 0 . 8 , 1 . 2 ) A _ { i } ) , } \\ { \displaystyle \mathcal { L } _ { \mathrm { K L } } = \frac { 1 } { \sum _ { i } | \mathcal { T } _ { i } | } \sum _ { i } \sum _ { t \in \mathcal { T } _ { i } } \left[ \exp ( - \delta _ { i , t } ) + \delta _ { i , t } - 1 \right] , \quad \delta _ { i , t } = \ell _ { i , t } - \ell _ { i , t } ^ { 0 } . } \end{array}\tag{46}
$$

The token-level KL reduction is defined for a group with at least one scored completion token. Every retained group receives one diferentiable policy evaluation; accumulating gradients from multiple groups does not change the denominator construction.

Gradient equivalence to sequence-normalized REINFORCE. In the forward evaluation $q _ { i , t } = 1$ , while its derivative is $\nabla _ { \theta } q _ { i , t } = \nabla _ { \theta } \ell _ { i , t }$ . The value one lies strictly inside the clipping interval. In a neighborhood with the denominator held fixed, both arguments of the minimum therefore coincide and have the same derivative, irrespective of the sign of $A _ { i }$ . It follows that

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } } = \nabla _ { \boldsymbol { \theta } } \left[ - \frac { 1 } { m } \sum _ { i \in V } \frac { \mathrm { s g } ( A _ { i } ) } { T _ { i } } \sum _ { t \in \mathcal { T } _ { i } } \ell _ { i , t } \right] .\tag{47}
$$

This is equality of gradients for the retained observations and advantages. It is not equality of the two scalar loss values. When every valid completion has at least one scored token, the implemented policy value is $- m ^ { - 1 } \textstyle \sum _ { i \in V } A _ { i } = 0$ even when its gradient is nonzero. Its computational graph carries the gradient through the numerator despite cancellation of the evaluated values. A finite-diference check of this derivative must freeze the denominator at the expansion point; rebuilding a detached denominator separately at each perturbed point instead diferentiates a constant-valued forward expression.

Sequence normalization gives each valid response one outer weight and averages its token scores before that weighting. The KL term uses a diferent reduction: each scored token has the same weight across all retained completions, including those excluded from V . Group standardization, validity filtering, sampling, and sequence normalization are therefore all part of the update estimator; Equation 47 does not replace them by an unnormalized expected-reward objective.

Sampled divergence geometry. For $k ( \delta ) = e ^ { - \delta } + \delta - 1 , k \geq 0 , k ^ { \prime } ( \delta ) = 1 - e ^ { - \delta }$ , and $k ^ { \prime \prime } ( \delta ) = e ^ { - \delta } > 0$ . The minimum is at $\delta = 0$ , with expansion $k ( \delta ) = \delta ^ { 2 } / 2 + O ( \delta ^ { 3 } )$ . For a fixed prefix and full-support distributions $p , p _ { 0 }$

$$
\mathbb { E } _ { y \sim p } k \bigg ( \log \frac { p ( y ) } { p _ { 0 } ( y ) } \bigg ) = \sum _ { y } p ( y ) \log \frac { p ( y ) } { p _ { 0 } ( y ) } = D _ { \mathrm { K L } } ( p \| p _ { 0 } ) .\tag{48}
$$

The exponential term integrates to $\begin{array} { r } { \sum _ { y } p _ { 0 } ( y ) = 1 } \end{array}$ . The optimizer, however, diferentiates the sampled expression with the observed token fixed. Under a general sampling distribution q, that conditional expected derivative is $\textstyle \sum _ { y } q ( y ) ( 1 - p _ { 0 } ( y ) / p ( y ) ) \nabla _ { \theta }$ log p(y). This specifies the diferentiated penalty independently of the value identity. Temperature, nucleus filtering, and group selection determine the actual sampling distribution, while the evaluated $p$ uses full-vocabulary, untempered logits.

## B.9 Routing regularization and its reduction

For the canonical 28 receiving layers, the complete auxiliary objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { p o l i c y } } + 0 . 0 2 \mathcal { L } _ { \mathrm { K L } } + 0 . 0 1 \Omega$ , with

$$
\Omega = \frac { 1 } { 2 8 } \sum _ { \ell = 4 } ^ { 3 1 } \left[ \mathrm { m e a n } _ { n , t , b } \left( a _ { \ell , n , t , b } - \frac { 1 } { n _ { \ell } } \right) ^ { 2 } + 0 . 0 5 \mathrm { m e a n } _ { n , t , k } c _ { \ell , n , t , k } ^ { 2 } \right] .\tag{49}
$$

The first mean includes the $n _ { \ell }$ visible sources; the second includes r source-mixture coordinates. Both range over all $B T$ tensor positions. For one position with n sources, define $\begin{array} { r } { \psi ( a ) = n ^ { - 1 } \sum _ { b } ( a _ { b } - 1 / n ) ^ { 2 } } \end{array}$ . Since $\textstyle \sum _ { b } a _ { b } = 1$ ，

$$
\psi ( a ) = \frac { 1 } { n } \left( \| a \| _ { 2 } ^ { 2 } - \frac { 1 } { n } \right) , \qquad 0 \leq \psi ( a ) \leq \frac { n - 1 } { n ^ { 2 } } .\tag{50}
$$

The minimum is uniform allocation; the upper bound is reached at a simplex vertex and approached by dense softmax as logits separate. For $n = 1$ , the function and its logit gradient are identically zero. Thus at layers 4–7 the source-mixture magnitude term contributes while the routing-deviation term does not. This follows from source availability, not from a special early-layer regularization mask.

The source-coordinate mean also means that the maximum per-position deviation penalty depends on n: it is $1 / 4$ for two sources and $6 / 4 9$ for seven. Its gradient with respect to logits z is $\begin{array} { r } { \nabla _ { z } \psi = \frac { 2 } { n } ( \mathrm { d i a g } a - a a ^ { \top } ) ( a - 1 / n ) } \end{array}$ The penalty acts on selection coeficients, while the source-mixture term acts on the mixture before $U _ { \ell }$ and RMS calibration. Neither directly penalizes the norm of the final calibrated writeback projection.

To make the position reduction explicit, partition the $B T$ positions into prompt, completion, and padding sets ${ \mathcal { P } } , { \mathcal { C } } , { \mathcal { Z } }$ . For either scalar positionwise penalty $f _ { i }$

$$
\frac { 1 } { B T } \sum _ { n , t } f _ { n , t } = \sum _ { Q \in \{ \mathcal { P } , \mathcal { C } , \mathcal { Z } \} } \frac { \left| Q \right| } { B T } \operatorname* { m e a n } _ { \mathbf { \substack { ( n , t ) \in Q } } } f _ { n , t } ,\tag{51}
$$

where empty-set contributions are zero. This identity describes the implemented full-tensor objective: each position set contributes in proportion to its size. It also separates this reduction from the completion masks used by the two language-model loss terms.

## B.10 Bounded informative resampling and generation budgets

For a fixed prompt and fixed model parameters during sampling, let Y denote a group drawn from the generation procedure, and let $I ( Y )$ indicate that at least one completion is parsed correct and at least one is not. Write $q = \operatorname* { P r } ( I = 1 )$ and let A be the maximum number of group attempts. The procedure retains the first mixed-correctness group; if no earlier attempt is mixed, it retains attempt A regardless of its outcome. The configured values are $G = 4$ and $A = 4$ (three additional attempts).

Assume attempts are independent and identically distributed conditional on the prompt. With N the number of attempts, the tail probabilities and expectation are

$$
\Pr ( N \geq j ) = ( 1 - q ) ^ { j - 1 } , \quad 1 \leq j \leq A , \qquad \mathbb { E } N = \sum _ { j = 0 } ^ { A - 1 } ( 1 - q ) ^ { j } = { \frac { 1 - ( 1 - q ) ^ { A } } { q } } .\tag{52}
$$

The quotient has continuous limit A at $q = 0$ . This follows because reaching attempt j requires exactly that the preceding $j - 1$ attempts fail the mixed-correctness test. The terminal probability is $\operatorname* { P r } ( \overset { \cdot } { N } = A ) = ( \overset { \cdot } { 1 } - q ) ^ { A - 1 }$ , which includes both success and failure on the final attempt.

Let $\mu$ denote the original group law and $\mu _ { \mathrm { r e t } }$ the retained law. For any group event E and $0 < q < 1$

$$
\begin{array} { c } { { \mu _ { \mathrm { r e t } } ( E ) = \displaystyle \frac { 1 - ( 1 - q ) ^ { A } } { q } \mu ( E \cap I ) + ( 1 - q ) ^ { A - 1 } \mu ( E \cap I ^ { c } ) , } } \\ { { \mu _ { \mathrm { r e t } } = [ 1 - ( 1 - q ) ^ { A } ] \mu ( \cdot \mid I ) + ( 1 - q ) ^ { A } \mu ( \cdot \mid I ^ { c } ) . } } \end{array}\tag{53}
$$

To prove the first line, sum $( 1 - q ) ^ { j - 1 } \mu ( E \cap I )$ over successful retention at each $j = 1 , \dotsc , A$ , then add $( 1 - q ) ^ { A - 1 } \mu ( E \cap I ^ { c } )$ for a failed final attempt. The second line groups these terms into normalized conditional distributions. Bounded retry increases the mass of mixed groups while preserving a nonzero failed-group mass $( 1 - q ) ^ { A }$ whenever $q < 1$

The retry event precedes preference-validity filtering. Let J denote a group whose valid completions include both reward values. Then $J \subseteq I ,$ but equality need not hold: a group containing correct answers and truncated-unparseable failures is mixed before filtering and can become uniform afterwards. If $r = \operatorname* { P r } ( J )$ Equation 53 gives

$$
\operatorname* { P r } _ { \mathrm { r e t } } ( J ) = r \frac { 1 - ( 1 - q ) ^ { A } } { q } .\tag{54}
$$

For binary rewards, this is the probability of a retained group with nonzero advantages. It distinguishes the retry acceptance event from producing a nonconstant set of preference coeficients; the resulting parameter gradient also depends on the token-score derivatives.

Table 17 : Analytical illustration of bounded retry. Four independent Bernoulli-correctness completions per attempt, all preference-valid, with four attempts allowed. Values are computed from $q = 1 - p ^ { 4 } - ( 1 - p ) ^ { \bar { 4 } }$
<table><tr><td>Per-completion correctness p</td><td>Mixed q</td><td>Retained mixed</td><td>Expected attempts</td><td>Expected responses</td></tr><tr><td>0.1</td><td>0.3438</td><td>0.8146</td><td>2.3694</td><td>9.4774</td></tr><tr><td>0.5</td><td>0.8750</td><td>0.9998</td><td>1.1426</td><td>4.5703</td></tr><tr><td>0.9</td><td>0.3438</td><td>0.8146</td><td>2.3694</td><td>9.4774</td></tr></table>

In the independent Bernoulli illustration, $q = 1 - p ^ { G } - ( 1 - p ) ^ { G }$ . It is symmetric under $p \mapsto 1 - p ;$ both nearly always-correct and nearly always-incorrect prompts can consume additional attempts. The procedure therefore selects correctness diversity within groups, rather than preferring high accuracy by itself.

Responses, tokens, and retained updates. Every prompt yields one retained group but incurs between G and GA generated responses, with expectation GEN. Let $C _ { j }$ be the completion-token cost of attempt $j ,$ identically distributed jointly with its outcome. Reaching attempt j depends only on earlier outcomes and is independent of $C _ { j }$ . Therefore

$$
{ \mathbb E } \left[ \sum _ { j = 1 } ^ { N } C _ { j } \right] = \sum _ { j = 1 } ^ { A } \operatorname* { P r } ( N \geq j ) { \mathbb E } C _ { 1 } = { \mathbb E } N { \mathbb E } C _ { 1 } .\tag{55}
$$

This equality does not require the cost of an attempt to be independent of its own correctness outcome. The expected cost of discarded attempts, for $0 < q < 1$ , is

$$
\mathbb { E } C _ { \mathrm { d i s c a r d e d } } = \mathbb { E } [ C _ { 1 } \mid I ^ { c } ] ( 1 - q ) \sum _ { j = 0 } ^ { A - 2 } ( 1 - q ) ^ { j } .\tag{56}
$$

Only failed nonterminal attempts are discarded. Retained mixed and failed groups can have diferent length distributions, as made explicit by Equation 53.

For D prompts and a cap of M new tokens per completion, the deterministic bounds are $\begin{array} { r } { D G \le N _ { \mathrm { r e s p o n s e s } } \le } \end{array}$ DGA and $N _ { \mathrm { g e n e r a t e d ~ t o k e n s } } \leq D G A M$ . For one pass over $D = 7 { , } 4 7 3$ prompts with $G = A = 4$ , the response budget is $^ { 2 9 , 8 9 2 - 1 1 9 , 5 6 8 }$ completions and the token bound is 119,568M. The number of diferentiated retained groups remains $^ { 7 , 4 7 3 }$ . These count generation separately from teacher-forced scoring and from gradient accumulation. They make precise why prompt count, optimizer updates, and generated-token budget characterize diferent aspects of the same RL procedure.

## C Complete empirical results and design analysis

This appendix develops the evidence for parameter-eficient computation reuse. Method comparisons pair reasoning quality with trainable allocation; component comparisons examine how that capacity is organized; writeback diagnostics and configuration profiles characterize the resulting task responses. Generative scores are means sample standard deviations (SD) across three evaluation seeds, 42, 43, and 44. Bold/underlined means identify the best/second-best values within each column, including ties; they describe the ordering of means. Lower values are preferred for resource costs and perplexity. Historical configurations retain their estimate attribute, and marks original non-generative point references. The complete sample variances appear in Appendix C.12.

## C.1 Method comparisons across mathematics and search

For each evaluation seed, AIME mean averages the two annual scores and MathAvg gives one-third weight to GSM8K, MATH-500, and AIME mean. The reported aggregate SD is computed across those seed-level aggregates. Diferences likewise pair the same evaluation-seed indices before taking their mean and SD. Table 18 completes the annual and aggregate profiles in Table 1; Table 19 retains the complete search comparison.

Table 18 : Annual mathematics results and aggregate contrasts. Accuracy and MathAvg are percentages; SD and $\Delta$ are percentage points. ∆ is the paired MathAvg diference from full-parameter GRPO. GSM8K, MATH-500, and parameter allocations appear in Table 1.
<table><tr><td>Method</td><td>AIME24</td><td></td><td>AIME25 AIME mean</td><td></td><td>MathAvg ∆ vs GRPO</td></tr><tr><td>Frozen base</td><td> $3 0 . 0 0 \pm 5 . 7 7$ </td><td> $3 3 . 3 3 \pm 3 . 3 3$ </td><td> $3 1 . 6 7 \pm 3 . 3 3$ </td><td> $6 5 . 3 3 \pm 0 . 8 9$ </td><td> $- 8 . 4 6 \pm 0 . 9 5$ </td></tr><tr><td>Full-parameter GRPO</td><td> $4 4 . 4 4 \pm 6 . 9 4$ </td><td> $4 4 . 4 4 \pm 3 . 8 5$ </td><td> $4 4 . 4 4 \pm 5 . 3 6$ </td><td> $7 3 . 7 9 \pm 1 . 8 3 $ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>RFT</td><td> ${ \underline { { 5 1 . 1 1 } } } \pm 1 0 . 1 8$ </td><td> $4 1 . 1 1 \pm 1 3 . 4 7$ </td><td> $4 6 . 1 1 \pm 2 . 5 5$ </td><td> $7 3 . 6 4 \pm 1 . 3 3 $ </td><td> $- 0 . 1 5 \pm 2 . 5 4$ </td></tr><tr><td> $\mathrm { L o R A - r 2 } + \mathrm { G R P O }$ </td><td> $4 3 . 3 3 \pm 1 7 . 3 2$ </td><td> $4 1 . 1 1 \pm 1 0 . 7 2$ </td><td> $4 2 . 2 2 \pm 1 2 . 7 3$ </td><td> $7 3 . 4 2 \pm 4 . 2 8$ </td><td> $- 0 . 3 7 \pm 2 . 6 7$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td> $5 0 . 0 0 \pm 5 . 7 7$ </td><td> $4 1 . 1 1 \pm 6 . 9 4$ </td><td> $4 5 . 5 6 \pm 6 . 3 1$ </td><td> $7 4 . 8 3 \pm 1 . 4 1$ </td><td> $+ 1 . 0 4 \pm 2 . 6 9$ </td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td> $4 1 . 1 1 \pm 5 . 0 9$ </td><td> $4 7 . 7 8 \pm 5 . 0 9$ </td><td> $4 4 . 4 4 \pm 2 . 5 5$ </td><td> $7 5 . 7 4 \pm 1 . 4 7$ </td><td> $+ 1 . 9 5 \pm 0 . 4 1$ </td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td> $\underline { { 5 1 . 1 1 } } \pm 1 9 . 5 3$ </td><td> $4 6 . 6 7 \pm 3 . 3 3$ </td><td> $4 8 . 8 9 \pm 8 . 2 2$ </td><td> $7 5 . 9 5 \pm 2 . 4 3$ </td><td> $+ 2 . 1 6 \pm 2 . 3 0$ </td></tr><tr><td> $_ \mathrm { L o R A - r 1 6 } ,$  matched retries</td><td> $4 5 . 5 6 \pm 7 . 7 0$ </td><td> $5 1 . 1 1 \pm 1 2 . 6 2$ </td><td> $4 8 . 3 3 \pm 6 . 0 1$ </td><td> $7 7 . 2 8 \pm 1 . 9 5$ </td><td> $\pm 3 . 4 9 \pm 3 . 7 6$ </td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { R o u t e r } + \mathrm { G R P O }$ </td><td> ${ \bf 6 3 . 3 3 \pm 6 . 6 7 }$ </td><td> $\mathbf { 5 7 . 7 8 \pm 1 . 9 2 }$ </td><td> ${ \bf 6 0 . 5 6 \pm 3 . 4 7 }$ </td><td> ${ \bf 8 3 . 6 4 \pm 1 . 1 6 }$ </td><td> $\mathbf { + 9 . 8 5 \pm 2 . 9 3 }$ </td></tr></table>

Table 19 : Complete agentic search comparison. BrowseComp Plus uses token-F1 multiplied by 100; ASearch Test denotes the ASearcher held-out validation score (0–100). Appendix A.7 details the scorer and default penalty factors. $\Delta$ is the paired diference from full-parameter GRPO; all entries are mean SD in the corresponding metric's points.
<table><tr><td>Method</td><td>BrowseComp F1 ∆ vs GRPO</td><td></td><td></td><td>ASearch Δ vs GRPO</td></tr><tr><td>Frozen base</td><td> $5 . 1 1 \pm 0 . 3 0$ </td><td> $- 2 8 . 1 2 \pm 1 . 2 5$ </td><td> $5 9 . 7 6 \pm 0 . 4 1$ </td><td> $- 1 0 . 1 9 \pm 0 . 5 6$ </td></tr><tr><td>Full-parameter GRPO</td><td> $3 3 . 2 3 \pm 1 . 5 4$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $6 9 . 9 4 \pm 0 . 5 3 $ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>RFT</td><td> $3 1 . 2 0 \pm 0 . 7 5$ </td><td> $- 2 . 0 3 \pm 2 . 0 7$ </td><td> $6 5 . 8 1 \pm 0 . 9 2$ </td><td> $- 4 . 1 4 \pm 1 . 1 8$ </td></tr><tr><td> $\mathrm { L o R A - r 2 } + \mathrm { G R P O }$ </td><td> $3 2 . 9 8 \pm 0 . 5 6$ </td><td> $- 0 . 2 5 \pm 1 . 2 5$ </td><td> $6 9 . 1 9 \pm 0 . 5 9$ </td><td> $- 0 . 7 5 \pm 1 . 0 1$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td> $3 6 . 0 6 \pm 0 . 5 6$ </td><td> $+ 2 . 8 3 \pm 1 . 8 0$ </td><td> $7 0 . 5 9 \pm 0 . 1 8$ </td><td> $+ 0 . 6 4 \pm 0 . 6 4$ </td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td> $\mathbf { 3 7 . 0 0 \pm 1 . 1 7 }$ </td><td> $\mathbf { + 3 . 7 7 \pm 2 . 5 6 }$ </td><td> $7 1 . 4 6 \pm 0 . 5 5$ </td><td> $+ 1 . 5 2 \pm 0 . 6 5$ </td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td> $3 5 . 5 0 \pm 0 . 5 7$ </td><td> $+ 2 . 2 7 \pm 1 . 5 8$ </td><td> $7 0 . 6 7 \pm 0 . 1 9$ </td><td> $+ 0 . 7 3 \pm 0 . 3 4$ </td></tr><tr><td> $\mathrm { L o R A – r 1 6 , \ m a t c h e d { \ r e t r i e s } }$ </td><td> $3 6 . 9 0 \pm 1 . 1 3$ </td><td> $+ 3 . 6 7 \pm 1 . 3 2$ </td><td> $7 2 . 8 0 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $\pm 2 . 8 6 \pm 0 . 6 3$ </td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { R o u t e r } + \mathrm { G R P O }$ </td><td> $\underline { { 3 6 . 9 8 } } \pm 1 . 1 6$ </td><td> $\pm 3 . 7 5 \pm 0 . 9 1$ </td><td> $\mathbf { 7 3 . 9 8 \pm 0 . 7 4 }$ </td><td> $+ 4 . 0 4 \pm 1 . 2 7$ </td></tr></table>

Against full-parameter GRPO, T-Router improves MathAvg by $9 . 8 5 \pm 2 . 9 3$ points with a trainable allocation equal to 0.466% of the backbone. At a similar adapter budget, matched-retry LoRA records 77.28 1.95 versus T-Router’s 83.64 1.16, a 6.36-point diference in means. The corresponding search means are 36.90 versus 36.98 on BrowseComp Plus and 72.80 versus 73.98 on ASearch. LoRA-MoE records 37.00 BrowseComp F1 with 173.112M trainable parameters; T-Router records 36.98 with 41.730M. Together these profiles establish a favorable quality–allocation tradeof and the strongest ASearch mean in the comparison.

## C.2 Complete component ablations

Table 20 retains all four task scores and the paired aggregate contrast for each component change. The no-state alternative uses a capacity-matched MLP, allowing the controller's role to be assessed alongside its allocation.

Table 20 : Complete task-level component ablations. Entries are mean  SD across evaluation seeds. ∆ is each variant minus full T-Router in MathAvg percentage points.
<table><tr><td>Configuration</td><td>GSM8K</td><td>MATH</td><td>AIME24</td><td>AIME25</td><td>MathAvg</td><td>Δ</td></tr><tr><td>Full T-Router</td><td>97.62±0.79 92.73±0.70 63.33±6.67 57.78±1.92 83.64±1.16</td><td></td><td></td><td></td><td></td><td> $\mathbf { 0 . 0 0 } \pm \mathrm { 0 . 0 0 }$ </td></tr><tr><td>No cross-layer retrieval</td><td>96.26 ±0.59</td><td>79.40 ±1.56</td><td></td><td>14.44 ±3.85 51.11 ±15.03</td><td></td><td>69.48 ±2.36 −14.16 ±1.78</td></tr><tr><td>Hidden-state memory</td><td>96.79±0.53</td><td></td><td>77.80 ±1.31 37.78 ±10.72</td><td> $1 7 . 7 8 \pm 5 . 0 9$ </td><td>67.46 ±2.19 -</td><td> $. 1 6 . 1 8 \pm 2 . 9 6$ </td></tr><tr><td>Learned static routing</td><td>96.01 ±0.68</td><td></td><td>77.27 ±0.81 26.67 ±5.77</td><td> $1 6 . 6 7 \pm 6 . 6 7$ </td><td></td><td>64.98 ±1.71 −18.66 ±2.78</td></tr><tr><td>No state (matched MLP)</td><td>95.02 ±0.75</td><td></td><td>83.93 ±1.92 50.00 ±17.64</td><td> $2 8 . 8 9 \pm 1 0 . 1 8$ </td><td></td><td>72.80 ±4.10 −10.84 ±3.09</td></tr><tr><td>No layer/block identity</td><td>94.29 ±0.42</td><td></td><td>66.13 ±1.80 11.11 ±10.72</td><td> $1 6 . 6 7 \pm 3 . 3 3$ </td><td></td><td>58.10 ±1.06 −25.53 ±0.56</td></tr><tr><td>No RMS alignment</td><td>94.57 ±0.64</td><td>67.93 ±3.80</td><td>3.33 ±3.33</td><td> $1 0 . 0 0 \pm 3 . 3 3$ </td><td></td><td>56.39 ±0.41 −27.25 ±1.24</td></tr><tr><td>No token-conditioned gate</td><td>96.26 ±0.42</td><td>66.73 ±1.81</td><td></td><td>34.44 ±8.39 22.22 ±11.71</td><td>63.78 ±2.61</td><td> $. 1 9 . 8 6 \pm 1 . 4 7$ </td></tr><tr><td>No informative retries</td><td>94.39 ±0.55</td><td>74.87 ±2.55</td><td></td><td>14.44 ±1.92 22.22 ±3.85</td><td></td><td> $6 2 . 5 3 \pm 1 . 2 0 \ - 2 1 . 1 1 \pm 2 . 2 6$ </td></tr></table>

Computation content and addressed retrieval. Full T-Router reaches $8 3 . 6 4 \pm 1 . 1 6$ MathAvg. Removing cross-layer retrieval gives $6 9 . 4 8 \pm 2 . 3 6$ , and replacing completed block changes with hidden states gives 67.46 2.19. These changes distinguish access to earlier computations from the representation stored in memory. The corresponding MATH-500 means are 92.73, 79.40, and 77.80. Source/layer identity identifies the origin and destination of each exchange; its removal has a paired MathAvg diference of 25.53 0.56 points.

History, selection, and relative influence. The matched MLP records 72.80 4.10 MathAvg and learned static routing records 64.98  1.71, supporting the organization of capacity into recurrent depth history and query-dependent selection. Removing RMS alignment gives a paired diference of 27.25  1.24 points, while removing the token-conditioned gate gives $- 1 9 . 8 6 \pm 1 . 4 7$ . Calibration sets the relative intervention scale; the signed gate adjusts that influence to the current token. The complete interface learns which computations to reuse and how strongly to incorporate them.

## C.3 Parameter and training-cost profiles of the ablations

Table 21 pairs the same configurations with exact allocated parameter counts and their original GPU-hour records. The common estimated optimizer-step count separates update count from response generation and the computation performed within an update.

Table 21 : Resources for the component study. Parameter counts are exact allocations; steps retain their estimated status GPU-hours retain their original training-cost records, separate from the evaluation-seed statistics.
<table><tr><td>Configuration</td><td>Parameters Steps (est.) GPU-hours</td><td></td><td></td></tr><tr><td>Full T-Router</td><td>41,730,332</td><td>3,737</td><td>121.23</td></tr><tr><td>No cross-layer retrieval</td><td>41,648,924</td><td>3,737</td><td>110.97</td></tr><tr><td>Hidden-state memory</td><td>41,730,332</td><td>3,737</td><td>122.42</td></tr><tr><td>Learned static routing</td><td>40,535,324</td><td>3,737</td><td>114.90</td></tr><tr><td>No state (matched MLP)</td><td>41,731,978</td><td>3,737</td><td>69.25</td></tr><tr><td>No layer/block identity</td><td>41,729,052</td><td>3,737</td><td>77.34</td></tr><tr><td>No RMS alignment</td><td>41,730,332</td><td>3,737</td><td>117.70</td></tr><tr><td>No token-conditioned gate</td><td>41,725,980</td><td>3,737</td><td>123.89</td></tr><tr><td>No informative retries</td><td>41,730,332</td><td>3,737</td><td>37.92</td></tr></table>

The MLP allocates 41,731,978 parameters, 1,646 more than full T-Router’s 41,730,332. Its paired MathAvg diference is 10.84 3.09 points. Hidden-state memory and the no-RMS variant retain exactly the full allocation; removing the token-conditioned gate changes it by 4,352 parameters. These comparisons link accuracy to the organization of trainable capacity.

Informative retries address a complementary training dimension. The no-retry configuration records 37.92 GPU-hours and $6 2 . 5 3 \pm 1 . 2 0$ MathAvg, while full T-Router records 121.23 GPU-hours and $8 3 . 6 4 \pm 1 . 1 6$ . Their paired MathAvg diference is $2 1 . 1 1 \pm 2 . 2 6$ points in favor of the full configuration. Retry generation changes the completions available to each prompt's update while retaining the same estimated optimizer-step count, connecting the empirical comparison to the correctness diversity in Appendix B.

## C.4 Configuration landscape and annual task profiles

The historical exploration varies writeback rules, controller structure, and optimization settings. Table 22 preserves every annual score from its initialization and structural branch, while Table 23 records the complete writeback sweep. The historical attribute applies to GSM8K, MATH-500, and their aggregates; evaluationseed dispersion does not change that provenance. Full T-Router is the final-model reference.

Table 22 : Complete historical task profiles and final reference. Generative entries are mean $\pm \operatorname { S D } ;$ retains the historical IID-estimate attribute. The initialization and local/controller variants use the standard-gate branch. Full T-Router is the final configuration from Table 1.
<table><tr><td>Configuration</td><td></td><td>GSM8K MATH-500</td><td>AIME24</td><td>AIME25</td><td> $\operatorname { M a t h A v g }$ </td></tr><tr><td>Standard gate†</td><td>87.57 ±0.87</td><td> $8 4 . 8 7 \pm 3 . 2 9$ </td><td> $3 6 . 6 7 \pm 0 . 0 0$ </td><td>27.78 ±10.72</td><td>68.22 ±2.83</td></tr><tr><td>Standard gate, no KL†</td><td> $8 9 . 0 3 \pm 0 . 5 8 $ </td><td> $7 1 . 6 7 \pm 4 . 0 2$ </td><td> $1 3 . 3 3 \pm 5 . 7 7$ </td><td> $4 . 4 4 \pm 1 . 9 2$ </td><td> $5 6 . 5 3 \pm 1 . 7 2$ </td></tr><tr><td>Local16†</td><td> $9 0 . 7 5 \pm 1 . 1 2 $ </td><td> $8 2 . 0 0 \pm 1 . 0 0 $ </td><td> $3 3 . 3 3 \pm 3 . 3 3$ </td><td> $3 5 . 5 6 \pm 1 8 . 3 6$ </td><td> $6 9 . 0 7 \pm 2 . 7 4$ </td></tr><tr><td>Local64†</td><td> $8 9 . 5 4 \pm 0 . 5 7$ </td><td> $7 1 . 0 7 \pm 2 . 5 7$ </td><td> $5 . 5 6 \pm 5 . 0 9$ </td><td> $1 1 . 1 1 \pm 5 . 0 9$ </td><td> $5 6 . 3 1 \pm 2 . 4 8$ </td></tr><tr><td>Wide512†</td><td> $8 7 . 5 7 \pm 1 . 1 8$ </td><td> $8 4 . 6 0 \pm 1 . 9 3 $ </td><td> $2 4 . 4 4 \pm 1 0 . 7 2$ </td><td> $2 7 . 7 8 \pm 5 . 0 9$ </td><td> $6 6 . 0 9 \pm 2 . 8 5$ </td></tr><tr><td>Zero initialization†</td><td> $8 9 . 0 3 \pm 0 . 3 5$ </td><td> $8 3 . 8 7 \pm 1 . 8 1 $ </td><td> $5 1 . 1 1 \pm 1 0 . 7 2$ </td><td> $3 0 . 0 0 \pm 6 . 6 7$ </td><td> $7 1 . 1 5 \pm 2 . 2 9$ </td></tr><tr><td>Orthogonal initialization†</td><td> $9 0 . 8 3 \pm 0 . 0 8$ </td><td> $8 2 . 9 3 \pm 1 . 7 2$ </td><td> $4 2 . 2 2 \pm 5 . 0 9$ </td><td> $2 8 . 8 9 \pm 1 . 9 2 $ </td><td> $6 9 . 7 7 \pm 0 . 3 8$ </td></tr><tr><td>Top2 composite†</td><td> $9 6 . 6 4 \pm 0 . 2 9$ </td><td> $8 7 . 2 0 \pm 1 . 0 6 $ </td><td> $5 8 . 8 9 \pm 1 0 . 7 2 $ </td><td> $4 1 . 1 1 \pm 9 . 6 2 $ </td><td> $7 7 . 9 5 \pm 2 . 1 2$ </td></tr><tr><td>Full T-Router</td><td> $\mathbf { 9 7 . 6 2 \pm 0 . 7 9 }$ </td><td> $\mathbf { 9 2 . 7 3 \pm 0 . 7 0 }$ </td><td> ${ \bf 6 3 . 3 3 \pm 6 . 6 7 }$ </td><td> $\mathbf { 5 7 . 7 8 \pm 1 . 9 2 }$ </td><td> $\mathbf { 8 3 . 6 4 \pm 1 . 1 6 }$ </td></tr></table>

Full T-Router has the largest mean on all four tasks in this panel. Relative to Top2, the approximately 5.69-point MathAvg diference comprises 0.33 point from GSM8K, 1.84 from MATH-500, and 3.52 from pooled AIME. The annual means are 63.33 and 57.78 for full T-Router, compared with 58.89 and 41.11 for $\mathrm { T o p 2 }$ Zero initialization has the largest annual mean gap in this panel, 21.11 points. These annual gaps compare two task sets; their seed-paired SDs are not among the reported statistics.

Zero and orthogonal initialization modify the standard-gate branch; the orthogonal gain is 0.064, whereas RMS configurations use gain 1. Local16 and Local64 change the additional token-local branch rank; Wide512 changes controller dimensions; Top2 denotes the composite sparse-routing configuration specified in Appendix C.8.

## C.5 Benchmark contributions to aggregate differences

MathAvg admits an additive decomposition. For mean family score $s _ { j } ( c )$ and standard-gate reference $c _ { 0 } .$

$$
\Delta \mathrm { M a t h A v g } ( c , c _ { 0 } ) = \frac { s _ { \mathrm { G } } ( c ) - s _ { \mathrm { G } } ( c _ { 0 } ) } { 3 } + \frac { s _ { \mathrm { M } } ( c ) - s _ { \mathrm { M } } ( c _ { 0 } ) } { 3 } + \frac { s _ { \mathrm { A } } ( c ) - s _ { \mathrm { A } } ( c _ { 0 } ) } { 3 } .\tag{57}
$$

Figure 8 applies this identity to the displayed mean profiles. These decompositions and the later reweighting analyses use supplied rounded means; small rounding diferences from separately reported aggregates can therefore occur.

Full T-Router’s approximately 15.42-point diference from the standard gate comprises 3.35 points from GSM8K, 2.62 from MATH-500, and 9.45 from pooled AIME. Top2's approximately 9.73-point diference comprises 3.02, 0.78, and 5.93 points. Both distribute gains across all three mathematical families, with the full design adding a stronger MATH-500 and AIME profile.

Local16 contributes approximately $\left( + 1 . 0 6 , - 0 . 9 6 , + 0 . 7 4 \right)$ points, while Local64 contributes $( + 0 . 6 6 , - 4 . 6 0 , - 7 . 9 6 )$ . The no-KL branch contributes $( + 0 . 4 9 , - 4 . 4 0 , - 7 . 7 8 )$ . Zero and orthogonal initialization produce net mean diferences of approximately +2.93 and +1.55, respectively. These task-level views characterize the efect of complete settings alongside their aggregate order.

![](images/07d54dcf614a6f92b22a05598a952f96bce8e01357009117163de8b048863151.jpg)  
Whiskers: supplied SD across three evaluation rounds; gaps are mean differences.

<table><tr><td>b</td><td colspan="4">Annual means (%)</td></tr><tr><td></td><td></td><td>2024</td><td>2025</td><td>Gap</td></tr><tr><td>1</td><td>Standard gate†</td><td>36.67</td><td>27.78</td><td>8.89</td></tr><tr><td>2</td><td>Standard, no KL†</td><td>13.33</td><td>4.44</td><td>8.89</td></tr><tr><td>3</td><td>Local16†</td><td>33.33</td><td>35.56</td><td>-2.23</td></tr><tr><td>4</td><td>Local64†</td><td>5.56</td><td>11.11</td><td>-5.55</td></tr><tr><td>5</td><td>Wide512†</td><td>24.44</td><td>27.78</td><td>-3.34</td></tr><tr><td>6</td><td>Zero init.†</td><td>51.11</td><td>30.00</td><td>21.11</td></tr><tr><td>7</td><td>Orthogonal init.†</td><td>42.22</td><td>28.89</td><td>13.33</td></tr><tr><td>8</td><td>Top2 compositet</td><td>58.89</td><td>41.11</td><td>17.78</td></tr><tr><td>9</td><td>T-Router, init. 2%</td><td>63.33</td><td>57.78</td><td>5.55</td></tr></table>

Figure 7 : Annual competition-mathematics profiles. The two annual means locate each configuration; whiskers show supplied SDs, and numerical summaries give annual means and their gaps. Pooled means average AIME24 and AIME25, and annual gaps are diferences of those means. Each annual set contains 30 questions per evaluation seed.

<table><tr><td colspan="4">a Contribution to MathAvg</td></tr><tr><td></td><td>GSM8K base 87.57</td><td>MATH-500 base 84.87</td><td>Pooled AIME base 32.22</td></tr><tr><td>Standard, no KL†</td><td>+0.49</td><td>-4.40</td><td>-7.78</td></tr><tr><td>Local16†</td><td>+1.06</td><td>-0.96</td><td>+0.74</td></tr><tr><td>Local64†</td><td>+0.66</td><td>-4.60</td><td>-7.96</td></tr><tr><td>Wide512†</td><td>+0.00</td><td>-0.09</td><td>-2.04</td></tr><tr><td>Zero init.†</td><td>+0.49</td><td>-0.33</td><td>+2.78</td></tr><tr><td>Orthogonal init.†</td><td>+1.09</td><td>-0.65</td><td>+1.11</td></tr><tr><td>Top2 composite†</td><td>+3.02</td><td>+0.78</td><td>+5.93</td></tr><tr><td>T-Router, init. 2%</td><td>+3.35</td><td>+2.62</td><td>+9.45</td></tr></table>

Mean-only changes: one-third of each family difference (pp)

![](images/209a148d2f24391bbd93dc88aa7aa489b6c3cecc9e23686a00bfca7a1a529469.jpg)  
Figure 8 : Where aggregate mean differences occur. Each family contributes one-third of its signed diference from the standard gate. Navy/dark red indicate positive/negative contributions; the net column sums those mean contrasts. These are analytical decompositions of the same reported evaluations.

## C.6 Writeback configurations and task response

The writeback study connects relative intervention scale to the quality of the reused computation. The standard gate learns a coeficient without RMS alignment; fixed and learned RMS configurations express

d Structure × scale†

writeback on a common relative-scale basis. Table 23 contains every reported family score, aggregate, and endpoint diagnostic.

Table 23 : Complete writeback exploration and final-model reference. Scores are mean $\pm \ \mathrm { S D } ;$ retains historical estimates. Final WB and tail median are original percentage-valued endpoint diagnostics. A dash denotes an unavailable trailing median.
<table><tr><td>Configuration</td><td>GSM8K</td><td></td><td>MATH AIME mean</td><td>MathAvg Final WB</td><td></td><td>Tail</td></tr><tr><td>Standard  $\mathrm { g a t e ^ { \dagger } }$ </td><td> $8 7 . 5 7 \pm 0 . 8 7$ </td><td> $8 4 . 8 7 \pm 3 . 2 9$ </td><td> $3 2 . 2 2 \pm 5 . 3 6$ </td><td> $6 8 . 2 2 \pm 2 . 8 3 $ </td><td>0.1254</td><td></td></tr><tr><td>Fixed RMS  $0 . 5 \% ^ { \dagger }$ </td><td> $8 7 . 4 7 \pm 0 . 9 9$ </td><td> $7 6 . 2 0 \pm 1 . 5 9$ </td><td> $3 5 . 5 6 \pm 3 . 4 7$ </td><td> $6 6 . 4 1 \pm 1 . 6 9$ </td><td>0.5005</td><td></td></tr><tr><td>Fixed RMS  $1 \% ^ { \dagger }$ </td><td> $8 9 . 3 1 \pm 0 . 6 7 $ </td><td> $7 9 . 6 7 \pm 3 . 1 1$ </td><td> $4 0 . 0 0 \pm 5 . 0 0$ </td><td> $6 9 . 6 6 \pm 2 . 3 4$ </td><td>1.0010</td><td></td></tr><tr><td>Fixed RMS  $2 \% ^ { \dagger }$ </td><td> $9 6 . 9 9 \pm 0 . 5 6$ </td><td> $\underline { { 8 7 . 6 0 } } \pm 2 . 1 1$ </td><td> $4 7 . 2 2 \pm 2 . 5 5$ </td><td> $7 7 . 2 7 \pm 1 . 1 6$ </td><td>2.0020</td><td></td></tr><tr><td>Learned RMS, init  $0 . 5 \% ^ { \dagger }$ </td><td> $9 6 . 9 9 \pm 0 . 2 9$ </td><td> $8 4 . 6 7 \pm 0 . 9 9$ </td><td> $5 5 . 0 0 \pm 8 . 3 3 $ </td><td> $7 8 . 8 9 \pm 2 . 3 9$ </td><td></td><td>3.4987 3.7596</td></tr><tr><td>Learned RMS, init  $1 \% ^ { \dagger }$ </td><td> $\mathbf { 9 7 . 8 3 \pm 0 . 4 6 }$ </td><td> $8 4 . 0 7 \pm 1 . 6 7$ </td><td> $5 5 . 0 0 \pm 1 2 . 0 2$ </td><td> $7 8 . 9 6 \pm 4 . 4 0$ </td><td></td><td>3.6535 3.8762</td></tr><tr><td>Full T-Router</td><td> $9 7 . 6 2 \pm 0 . 7 9$ </td><td> $\mathbf { 9 2 . 7 3 \pm 0 . 7 0 }$ </td><td> ${ \bf 6 0 . 5 6 \pm 3 . 4 7 }$ </td><td> ${ \bf 8 3 . 6 4 \pm 1 . 1 6 }$ </td><td></td><td>4.0211 4.2065</td></tr></table>

## a Calibration map

![](images/89a4a17f5ee3f557772babe970626ab36b75b3bac1edc32251e6edf2a043d073.jpg)  
c Learned operating scale

<table><tr><td rowspan=2 colspan=4>bMean task response                  vs Std† (pp)GSM8K    MATH-500  Pooled AIMEF0.5†                            -8.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>F1†</td><td rowspan=1 colspan=1>+1.7</td><td rowspan=1 colspan=1>-5.2</td><td rowspan=1 colspan=1>+7.8</td></tr><tr><td rowspan=1 colspan=1>F2†</td><td rowspan=1 colspan=1>+9.4</td><td rowspan=1 colspan=1>+2.7</td><td rowspan=1 colspan=1>+15.0</td></tr><tr><td rowspan=1 colspan=1>L0.5†</td><td rowspan=1 colspan=1>+9.4</td><td rowspan=1 colspan=1>-0.2</td><td rowspan=1 colspan=1>+22.8</td></tr><tr><td rowspan=1 colspan=1>L1†</td><td rowspan=1 colspan=1>+10.3</td><td rowspan=1 colspan=1>-0.8</td><td rowspan=1 colspan=1>+22.8</td></tr><tr><td rowspan=1 colspan=1>T-Router</td><td rowspan=1 colspan=1>+10.1</td><td rowspan=1 colspan=1>+7.9</td><td rowspan=1 colspan=1>+28.3</td></tr></table>

![](images/7261bdeef1cd707431fa646bece9b1f3cc2e53a17294fb5983868685b7c0ac43.jpg)

![](images/a15e9380392813c07bd1a254c2c43a8cafe52d6a5798258177d568f335345d99.jpg)  
Figure 9 : Writeback scale, task response, and structure. (a) MathAvg versus final relative writeback. (b) Mean family-score diferences from the standard gate. F/L denote fixed/learned RMS, with coeficients or initializations in percent. (c) Original final and trailing-median diagnostics. (d) Structure-dependent mean changes under fixed 2% RMS. Whiskers in (a,d) show evaluation-seed SD. Tables 23 and 24 give the complete statistics.

Full T-Router records $8 3 . 6 4 \pm 1 . 1 6$ MathAvg at 4.0211% final writeback, combining a 92.73 MATH-500 mean with 60.56 pooled AIME. Learned RMS initialized at 0.5% and 1% records MathAvg means of 78.89 and 78.96, with pooled AIME means of 55.00 for both. These task profiles connect writeback scale to the computations delivered to the receiving layer.

## C.7 Writeback coefficients and endpoint diagnostics

The standard gate has a 20% bound and 5% initial coeficient, with small-normal writeback projections. Fixed RMS uses gain-1 orthogonal projections and constant coeficients of 0.5%, 1%, or 2%. Learned RMS uses the same gain-1 projection initialization, a 5% gate bound, and initial coeficients of 0.5%, 1%, or 2%. A fixed coeficient retains input-dependent source routing.

For the 0.5%, 1%, and 2% learned initializations, tail medians are 3.7596%, 3.8762%, and 4.2065%, exceeding final diagnostics by 0.2609, 0.2227, and 0.1854 points. The final scales lie closer together than the initial coeficients. Across the broader comparison, final scale alone does not order accuracy: fixed RMS 0.5% has a larger final diagnostic than the standard gate, while their MathAvg means are 66.41 and 68.22. Retrieval direction, the receiving representation, and scaling jointly determine the resulting computation.

a Initial coefficient and operating scale  
![](images/0b07693e0ade25a264f6695b6704bf185f88f0465397efec98f84e1bf9e7bd68.jpg)

## b Endpoint values (%)

<table><tr><td>Init.</td><td>Final WB</td><td>Tail</td><td>Tail – final</td></tr><tr><td>0.5</td><td>3.4987</td><td>3.7596</td><td>0.2609</td></tr><tr><td>1</td><td>3.6535</td><td>3.8762</td><td>0.2227</td></tr><tr><td>2</td><td>4.0211</td><td>4.2065</td><td>0.1854</td></tr></table>

Figure 10 : Writeback endpoint summaries. Initial coeficients are paired with Final WB (filled circles) and trailing-median WB (open circles). The table gives the original scalar diagnostics, their diferences, and the largest-to-smallest initial and final ratios. Final WB is the latest routed layer's whole-tensor writeback-to-residual norm ratio.

## C.8 Structure–writeback combinations

The interaction study pairs four structures with their original writeback and fixed 2% RMS. Table 24 preserves the mean and SD of each paired original-to-fixed change, retaining each structure's routing and state computation.

Table 24 : Structure–writeback combinations. MathAvg is in percent; SD and paired changes are in percentage points; retains the historical estimate attribute. Change is fixed RMS minus original writeback, paired by evaluation-seed index.
<table><tr><td>Structure</td><td>Original writeback Fixed RMS 2% Paired change</td><td></td><td></td></tr><tr><td>Dense, standard  $\mathrm { g a t e } ^ { \dagger }$ </td><td> $\underline { { 6 8 . 2 2 } } \pm 2 . 8 3 $ </td><td> $7 7 . 2 7 \pm 1 . 1 6$ </td><td> $\pm 9 . 0 5 \pm 2 . 2 6 $ </td></tr><tr><td>Wide512†</td><td> $6 6 . 0 9 \pm 2 . 8 5$ </td><td> $\mathbf { 7 8 . 6 9 \pm 0 . 6 6 }$ </td><td> $\mathbf { + 1 2 . 6 0 \pm 3 . 4 1 }$ </td></tr><tr><td>Top2 composite†</td><td> $\mathbf { 7 7 . 9 5 \pm 2 . 1 2 }$ </td><td> $6 3 . 9 8 \pm 1 . 6 2$ </td><td> $- 1 3 . 9 6 \pm 1 . 9 5$ </td></tr><tr><td>Local64†</td><td> $5 6 . 3 1 \pm 2 . 4 8$ </td><td> $6 2 . 4 4 \pm 0 . 9 0$ </td><td> $+ 6 . 1 3 \pm 1 . 6 2$ </td></tr></table>

Dense routing and Wide512 change by $+ 9 . 0 5 \pm 2 . 2 6$ and $+ 1 2 . 6 0 \pm 3 . 4 1$ points; the diference between these means is +3.55. Local64 changes by $+ 6 . 1 3 \pm 1 . 6 2 , \mathrm { o r \ - 2 . 9 2 }$ relative to the dense response. Top2 changes by $- 1 3 . 9 6 \pm 1 . 9 5 .$ , giving a 23.01 mean contrast relative to dense routing. The signs show that a configuration's original quality and its response to fixed calibration describe distinct aspects of the design space.

Meaning of the structure names. Local16 and Local64 add a token-local branch of rank 16 or 64, a controllerconditioned sigmoid gate initialized at 0.2, and a branch learning-rate multiplier of two; cross-layer memory rank remains 256. Wide512 sets slot and controller hidden dimensions to 512, with eight slots and memory rank 256. Top2 restricts source and slot selection to two entries, RMS-normalizes stored deltas, sets routing temperature to 0.7, and uses routing-sharpness, balance, and slot-diversity terms. Its result therefore characterizes that complete configuration.

Original writeback: MathAvg (%)  
a Response to fixed RMS 2%†  
![](images/d19edc1fe22c37f6e950dd79c37a6148f81699841bee38be4ac3a5c6b3dda3d4.jpg)

b Paired change ± SD (pp)
<table><tr><td></td><td>Change ± SD</td><td>– dense*</td></tr><tr><td>Dense</td><td>+9.05±2.26</td><td>+0.00</td></tr><tr><td>Wide512</td><td>+12.60±3.41</td><td>+3.55</td></tr><tr><td>Top2 composite</td><td>-13.96±1.95</td><td>-23.01</td></tr><tr><td>Local64</td><td>+6.13±1.62</td><td>-2.92</td></tr></table>

\* Difference of mean changes; no SD supplied.

Figure 11 : Structure-dependent response to fixed RMS writeback. Original and fixed-RMS scores in (a) show means and evaluation SDs. The dashed reference adds the dense row's 9.05-point change. The paired changes use supplied SDs; diferences between those changes are descriptive contrasts of means.

Wide512 with fixed RMS records 78.69 0.66 MathAvg with 45.546M allocated parameters, versus 83.64 1.16 and 41.730M for the selected learned-RMS design. The 3.816M allocation diference accompanies a 4.95-point diference in mean MathAvg. Allocation remains distinct from active memory and arithmetic during execution.

## C.9 Auxiliary task profiles

The auxiliary tasks describe narrative reasoning (MuSR), demanding multiple-choice questions (GPQA-D), instruction following (IFEval), and language-model likelihood (WikiText-2). IFEval strict@50 uses 50 prompts per evaluation seed and requires satisfying every specified constraint for a prompt.

Table 25 : Complete auxiliary task profiles. IFEval is mean sample SD over evaluation seeds. marks original non-generative point references for MuSR, GPQA-D, and perplexity; no evaluation-seed dispersion is assigned to those entries. Lower perplexity is preferred; dashes denote unavailable entries.
<table><tr><td>Configuration</td><td></td><td></td><td>MuSR GPQA-D IFEval strict@50 WikiText-2 PPL</td><td></td></tr><tr><td>Frozen base</td><td></td><td></td><td></td><td></td></tr><tr><td>Full-parameter GRPO</td><td> ${ \tt6 3 . 0 0 ^ { \ddagger } }$ </td><td> $\underline { { 4 0 . 0 0 } } ^ { \ddagger }$ </td><td></td><td></td></tr><tr><td>Standard gate</td><td> ${ \tt6 3 . 0 0 ^ { \ddagger } }$ </td><td> ${ \pmb q } 1 . 0 0 ^ { \dagger }$ </td><td> $3 1 . 3 3 \pm 3 . 0 6$ </td><td>7.0391</td></tr><tr><td>Fixed RMS 2%</td><td>61.00</td><td>39.00‡</td><td> $3 6 . 0 0 \pm 4 . 0 0$ </td><td>7.0890</td></tr><tr><td>Learned RMS, init 1% 61.00‡</td><td></td><td> $\underline { { 4 0 . 0 0 } } ^ { \ddagger }$ </td><td> $3 4 . 0 0 \pm 7 . 2 1$ </td><td> $\underline { { 7 . 0 7 5 8 ^ { \ddagger } } }$ </td></tr><tr><td>Full T-Router</td><td>61.00$</td><td>38.00‡</td><td> $\mathbf { 3 8 . 6 7 \pm } 4 . 1 6$ </td><td>7.1185</td></tr></table>

![](images/58935888d8a6ca853856a4233a89f8de1f449ea7b3ef6dab368f0cf833423eaf.jpg)  
IFEval whiskers: evaluation SD. ‡ Original non-generation scalar references.

Figure 12 : Task-specific diagnostic scales. Each task retains its native units. IFEval means carry evaluation-seed SDs; point references retain their original non-generative scoring status. The full-GRPO reference is available for the two multiple-choice tasks.

Full T-Router records the highest IFEval mean, $3 8 . 6 7 \pm 4 . 1 6 .$ compared with 36.00 4.00 for fixed RMS 2%, 34.00 7.21 for learned initialization at 1%, and $3 1 . 3 3 \pm 3 . 0 6$ for the standard gate. The three RMS settings share the MuSR point reference of 61.00, with GPQA-D references between 38.00 and 40.00. Their WikiText-2 perplexities are 7.0890, 7.0758, and 7.1185 for fixed 2%, learned 1%, and full T-Router, respectively. These values retain the separate meanings of accuracy, constraint satisfaction, and likelihood.

## C.10 Sensitivity to the choice of family weights

MathAvg assigns equal weight to the three mathematical families. To characterize alternative priorities, reaggregate the same mean profiles with $w _ { \mathrm { G } } , w _ { \mathrm { M } } , w _ { \mathrm { A } } \geq 0$ and $w _ { \mathrm { G } } + w _ { \mathrm { M } } + w _ { \mathrm { A } } = 1$

$$
\begin{array} { r } { S _ { w } ( c ) = w _ { \mathrm { G } } s _ { \mathrm { G } } ( c ) + w _ { \mathrm { M } } s _ { \mathrm { M } } ( c ) + w _ { \mathrm { A } } s _ { \mathrm { A } } ( c ) . } \end{array}\tag{58}
$$

![](images/5540b69eb59244ba472d5382f1f8f3a95ebc1dca2bd329568075939e7268352d.jpg)

![](images/01c3f30198797e2745b483fd82cb30d4dfa2d191ad844f7842539088ce31a50e.jpg)  
Figure 13 : Analytical sensitivity to benchmark-family weights. The simplex shows the largest weighted mean among 14 complete profiles. The one-dimensional slice gives equal weight to MATH-500 and AIME and varies GSM8K weight; the crossover is near $w _ { \mathrm { G } } = 0 . 9 7 1 3$ . Scores are analytical reaggregations of the supplied means using Equation 58.

Full T-Router has mean profile (97.62, 92.73, 60.56), versus (97.83, 84.07, 55.00) for learned RMS initialized at 1%. Therefore,

$$
S _ { w } ( \mathrm { i n i t ~ 2 \% } ) - S _ { w } ( \mathrm { i n i t ~ 1 \% } ) = - 0 . 2 1 w _ { \mathrm { G } } + 8 . 6 6 w _ { \mathrm { M } } + 5 . 5 6 w _ { \mathrm { A } } .\tag{59}
$$

These two profiles define the upper envelope: full T-Router strictly exceeds every other historical profile in all three family means. The 1% initialization has the largest GSM8K mean, while full T-Router has the largest MATH-500 and AIME means. Along $w _ { \mathrm { M } } = w _ { \mathrm { A } } = ( 1 - w _ { \mathrm { G } } ) / 2$ , full T-Router leads up to $w _ { \mathrm { G } } = 7 . 1 1 / 7 . 3 2 \approx 0 . 9 7 1 3$ . Along the equal-GSM8K/MATH-500 slice, its advantage is $4 . 2 2 5 + 1 . 3 3 5 w _ { \mathrm { A } }$ and remains positive throughout. Equal family weighting lies in the full model's region. This geometry describes priorities over the same mean score profiles.

## C.11 Question resolution and alternative aggregation

The mathematics evaluation contains 1,319 GSM8K questions, 500 MATH-500 questions, and 60 pooled AIME questions per seed. For that seed's correct-answer counts $n _ { \mathrm { G } } , n _ { \mathrm { M } } , n _ { \mathrm { A } } .$

$$
\mathrm { M a t h A v g } = { \frac { 1 0 0 } { 3 } } \left( { \frac { n _ { \mathrm { G } } } { 1 3 1 9 } } + { \frac { n _ { \mathrm { M } } } { 5 0 0 } } + { \frac { n _ { \mathrm { A } } } { 6 0 } } \right) .\tag{60}
$$

One extra correct answer in a single evaluation changes that seed's MathAvg by approximately 0.0253, 0.0667, or 0.5556 points for GSM8K, MATH-500, or pooled AIME. The three-seed mean spreads a change in one seed over three evaluations. Equal-family weighting prevents the largest test set from determining the aggregate solely through its question count.

a Main-method mean aggregation  
![](images/cb5c25c4c0073ed0ef0db01038303d9b68791ddff67591f736f3006712c85c95.jpg)  
Reweighted means have no inferred SD; increments describe one-round scoring.

b Benchmark weighting
<table><tr><td rowspan="2">Family</td><td rowspan="2">Size</td><td rowspan="2">Family rule</td><td rowspan="2">Size rule</td></tr><tr><td></td></tr><tr><td>GSM8K</td><td>1319</td><td>0.025</td><td>0.053</td></tr><tr><td>MATH-500</td><td>500</td><td>0.067</td><td>0.053</td></tr><tr><td>AIME</td><td>60</td><td>0.556</td><td>0.053</td></tr></table>

Per-answer score increment (pp)

Figure 14 : Aggregation at full-test sizes. The same nine method mean profiles are summarized by equal-family MathAvg and equal-question reaggregation. The size/resolution panel reports per-evaluation answer increments and family weights. The alternative weighted means are calculated from reported mean scores.

An equal-question summary instead uses

$$
S _ { \mathrm { q u e s t i o n } } = { \frac { 1 3 1 9 s _ { \mathrm { G } } + 5 0 0 s _ { \mathrm { M } } + 6 0 s _ { \mathrm { A } } } { 1 8 7 9 } } ,\tag{61}
$$

with weights approximately (0.7020, 0.2661, 0.0319). One correct answer changes this per-seed summary by $1 0 0 / 1 8 7 9 \approx 0 . 0 5 3 2$ points. Reaggregating full T-Router’s mean profile gives approximately 95.14, compared with its reported equal-family MathAvg of 83.64. Full T-Router has the highest mean in all three mathematical families among the nine main methods, so it remains first under every nonnegative family weighting summing to one. Both summaries express the same evaluated profile with diferent emphasis.

## C.12 Evaluation-seed dispersion and complete variance record

The source evaluations use seeds 42, 43, and 44. For a reported score or paired diference $x _ { r } ,$ the sample variance and SD are

$$
\bar { x } = { \frac { 1 } { 3 } } \sum _ { r = 1 } ^ { 3 } x _ { r } , \qquad s ^ { 2 } = { \frac { 1 } { 2 } } \sum _ { r = 1 } ^ { 3 } ( x _ { r } - \bar { x } ) ^ { 2 } , \qquad s = \sqrt { s ^ { 2 } } .\tag{62}
$$

Tables 26–28 preserve all seven supplied sample-variance panels at four decimals. Mathematical accuracy, MathAvg, and their diferences use $\mathrm { p p } ^ { 2 } ;$ search scores use the corresponding points<sup>2</sup>. Aggregates and paired diferences are formed within each evaluation seed before their variances are computed. Their supplied variances therefore retain the joint task/seed behavior; averaging component variances would not produce the same statistic. SDs describe evaluation variability for the reported configurations. Resource measurements and original references keep their separate measurement status.

Table 26 : Sample variances: mathematics and search. All values are supplied $s ^ { 2 }$ over three evaluation seeds. Zero is an exact zero sample variance; ∆ pairs each method with full-parameter GRPO at the same seed index.  
(a) Mathematics
<table><tr><td>Method</td><td>GSM8K</td><td>MATH</td><td>AIME24</td><td>AIME25</td><td>AIME mean</td><td>MathAvg</td><td> $\Delta$ </td></tr><tr><td>Frozen base</td><td>0.8986</td><td>5.0800</td><td>33.3333</td><td>11.1111</td><td>11.1111</td><td>0.7874</td><td>0.8933</td></tr><tr><td>Full-parameter GRPO</td><td>0.0690</td><td>0.4933</td><td>48.1481</td><td>14.8148</td><td>28.7037</td><td>3.3394</td><td>0.0000</td></tr><tr><td>RFT</td><td>1.3412</td><td>6.0933</td><td>103.7037</td><td>181.4815</td><td>6.4815</td><td>1.7760</td><td>6.4687</td></tr><tr><td> $\mathrm { L o R A – r 2 } + \mathrm { G R P O }$ </td><td>0.0536</td><td>0.6533</td><td>300.0000</td><td>114.8148</td><td>162.0370</td><td>18.2979</td><td>7.1083</td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td>0.5882</td><td>6.4133</td><td>33.3333</td><td>48.1481</td><td>39.8148</td><td>2.0014</td><td>7.2478</td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td>0.3679</td><td>2.0933</td><td>25.9259</td><td>25.9259</td><td>6.4815</td><td>2.1701</td><td>0.1681</td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td>0.2146</td><td>3.4533</td><td>381.4815</td><td>11.1111</td><td>67.5926</td><td>5.9142</td><td>5.2820</td></tr><tr><td>LoRA-r16, matched retries</td><td>0.2548</td><td>0.0133</td><td>59.2593</td><td>159.2593</td><td>36.1111</td><td>3.8180</td><td>14.1538</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { R o u t e r } + \mathrm { G R P O }$ </td><td>0.6227</td><td>0.4933</td><td>44.4444</td><td>3.7037</td><td>12.0370</td><td>1.3391</td><td>8.6068</td></tr></table>

(b) Agentic search
<table><tr><td>Method</td><td>BrowseComp</td><td></td><td>Δ F1 ASearch Δ score</td><td></td></tr><tr><td>Frozen base</td><td>0.0925</td><td>1.5637</td><td>0.1699</td><td>0.3186</td></tr><tr><td>Full-parameter GRPO</td><td>2.3867</td><td>0.0000</td><td>0.2777</td><td>0.0000</td></tr><tr><td>RFT</td><td>0.5584</td><td>4.2884</td><td>0.8434</td><td>1.3926</td></tr><tr><td>LoR  $_ { \mathrm { \tiny { A - r 2 } } } + \mathrm { \tiny { G R P O } }$ </td><td>0.3094</td><td>1.5502</td><td>0.3459</td><td>1.0129</td></tr><tr><td> $\mathrm { L o R A – r 1 6 + G R P O }$ </td><td>0.3189</td><td>3.2539</td><td>0.0317</td><td>0.4089</td></tr><tr><td> $\mathrm { L o R A \mathrm { - } M o E + R O \mathrm { - } G R P O }$ </td><td>1.3643</td><td>6.5709</td><td>0.3075</td><td>0.4258</td></tr><tr><td> $\mathrm { L o R A + S \mathrm { – G R P O } }$ </td><td>0.3256</td><td>2.4924</td><td>0.0371</td><td>0.1147</td></tr><tr><td>LoRA-r16, matched retries</td><td>1.2793</td><td>1.7436</td><td>0.5555</td><td>0.3915</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { R o u t e r } + \mathrm { G R P O }$ </td><td>1.3487</td><td>0.8305</td><td>0.5536</td><td>1.6154</td></tr></table>

Table 27 : Sample variances: components and writeback. The component ∆ is each variant minus full T-Router. Historical writeback settings retain their estimate attribute.  
(c) Component ablations
<table><tr><td>Configuration</td><td>GSM8K</td><td>MATH</td><td>AIME24</td><td>AIME25</td><td>MathAvg</td><td>Δ</td></tr><tr><td>Full T-Router</td><td>0.6227</td><td>0.4933</td><td>44.4444</td><td>3.7037</td><td>1.3391</td><td>0.0000</td></tr><tr><td>No cross-layer retrieval</td><td>0.3468</td><td>2.4400</td><td>14.8148</td><td>225.9259</td><td>5.5468</td><td>3.1758</td></tr><tr><td>Hidden-state memory</td><td>0.2836</td><td>1.7200</td><td>114.8148</td><td>25.9259</td><td>4.7817</td><td>8.7683</td></tr><tr><td>Learned static routing</td><td>0.4675</td><td>0.6533</td><td>33.3333</td><td>44.4444</td><td>2.9100</td><td>7.7297</td></tr><tr><td>No state (matched MLP)</td><td>0.5595</td><td>3.6933</td><td>311.1111</td><td>103.7037</td><td>16.7764</td><td>9.5347</td></tr><tr><td>No layer/block identity</td><td>0.1744</td><td>3.2533</td><td>114.8148</td><td>11.1111</td><td>1.1156</td><td>0.3102</td></tr><tr><td>No RMS alignment</td><td>0.4158</td><td>14.4533</td><td>11.1111</td><td>11.1111</td><td>0.1699</td><td>1.5405</td></tr><tr><td>No token-conditioned gate</td><td>0.1744</td><td>3.2933</td><td>70.3704</td><td>137.0370</td><td>6.7967</td><td>2.1560</td></tr><tr><td>No informative retries</td><td>0.2989</td><td>6.4933</td><td>3.7037</td><td>14.8148</td><td>1.4390</td><td>5.1010</td></tr></table>

(d) Writeback configurations
<table><tr><td>Configuration</td><td>GSM8K</td><td></td><td>K MATH AIME mean MathAvg</td><td></td></tr><tr><td>Standard  $\mathrm { g a t e ^ { \dagger } }$ </td><td>0.7645</td><td>10.8133</td><td>28.7037</td><td>7.9975</td></tr><tr><td>Fixed RMS  $0 . 5 \% ^ { \dagger }$ </td><td>0.9791</td><td>2.5200</td><td>12.0370</td><td>2.8583</td></tr><tr><td>Fixed RMS  $1 \% ^ { \dagger }$ </td><td>0.4541</td><td>9.6533</td><td>25.0000</td><td>5.4889</td></tr><tr><td>Fixed RMS  $2 \% ^ { \dagger }$ </td><td>0.3123</td><td>4.4400</td><td>6.4815</td><td>1.3420</td></tr><tr><td>Learned RMS, init  $0 . 5 \% ^ { \dagger }$ </td><td>0.0824</td><td>0.9733</td><td>69.4444</td><td>5.7185</td></tr><tr><td>Learned RMS, init  $1 \% ^ { \dagger }$ </td><td>0.2088</td><td>2.7733</td><td>144.4444</td><td>19.3792</td></tr><tr><td>Full T-Router</td><td>0.6227</td><td>0.4933</td><td>12.0370</td><td>1.3391</td></tr></table>

Table 28 : Sample variances: historical profiles and auxiliary scores. Changes in (e) are paired fixed-RMS-minus-original diferences. Dashes in (g) preserve unavailable variances for original non-generative references or missing scores.  
(e) Structure–writeback combinations
<table><tr><td>Structure</td><td>Original Fixed RMS 2% Change</td><td></td><td></td></tr><tr><td>Dense, standard gate†</td><td>7.9975</td><td>1.3420</td><td>5.1199</td></tr><tr><td>Wide512†</td><td>8.1360</td><td>0.4337</td><td>11.6171</td></tr><tr><td>Top2 composite†</td><td>4.5150</td><td>2.6144</td><td>3.8024</td></tr><tr><td>Local64†</td><td>6.1259</td><td>0.8179</td><td>2.6340</td></tr></table>

(f) Other historical configurations
<table><tr><td>Configuration</td><td>GSM8K</td><td>MATH</td><td>AIME24</td><td>AIME25</td><td>MathAvg</td></tr><tr><td>Standard gate†</td><td>0.7645</td><td>10.8133</td><td>0.0000</td><td>114.8148</td><td>7.9975</td></tr><tr><td>Standard gate, no KL†</td><td>0.3353</td><td>16.1733</td><td>33.3333</td><td>3.7037</td><td>2.9447</td></tr><tr><td>Local16†</td><td>1.2473</td><td>1.0000</td><td>11.1111</td><td>337.0370</td><td>7.5142</td></tr><tr><td>Local64†</td><td>0.3276</td><td>6.6133</td><td>25.9259</td><td>25.9259</td><td>6.1259</td></tr><tr><td>Wide512†</td><td>1.3852</td><td>3.7200</td><td>114.8148</td><td>25.9259</td><td>8.1360</td></tr><tr><td>Zero initialization†</td><td>0.1226</td><td>3.2933</td><td>114.8148</td><td>44.4444</td><td>5.2464</td></tr><tr><td>Orthogonal initialization†</td><td>0.0057</td><td>2.9733</td><td>25.9259</td><td>3.7037</td><td>0.1421</td></tr><tr><td>Top2 composite†</td><td>0.0824</td><td>1.1200</td><td>114.8148</td><td>92.5926</td><td>4.5150</td></tr></table>

(g) Auxiliary task profiles
<table><tr><td>Configuration</td><td>MuSR GPQA-D IFEval strict@50 WikiText-2 PPL</td><td></td></tr><tr><td>Frozen base</td><td></td><td></td></tr><tr><td>Full-parameter GRPO</td><td></td><td></td></tr><tr><td>Standard gate</td><td></td><td>9.3333</td></tr><tr><td>Fixed RMS 2%</td><td></td><td>16.0000</td></tr><tr><td>Learned RMS, init 1%</td><td></td><td>52.0000</td></tr><tr><td>Full T-Router</td><td></td><td>17.3333</td></tr></table>

## D Design interpretation and functional correspondence

T-Router pursues parameter-eficient reinforcement learning by concentrating adaptation in a communication interface around a frozen backbone. Completed block changes become explicitly accessible to later receivers, which learn their selection and relative influence. Separating transported content, recurrent depth state, and the current control signal gives each part of the trainable budget a concrete role. This is the computationa form of the thalamic routing principle.

## D.1 Computation and coordination as separate roles

The relevant thalamic function is context-dependent regulation of cortical communication. The pulvinar coordinates information transmission between cortical areas through attention-dependent synchronization (Saalmann et al., 2012). Mediodorsal thalamic input amplifies functional prefrontal connectivity, sustaining rule representations without itself relaying their categorical content (Schmitt et al., 2017). Together, these findings motivate a routing interface that uses task context to regulate how distributed computations influence a receiver.

In T-Router, pretrained decoder weights provide the computational substrate, while the complete auxiliary module regulates communication. Compressors define accessible source content, the depth controller conditions routing, and calibrated writeback changes the inputs processed by the frozen layers. Green source storage, purple controller state, and gold residual writeback in Figure 2 belong to this one coordinating system. The cortical partitions in Figure 1 depict distributed backbone computation; the thalamic correspondence covers the full auxiliary pathway.

Table 29 : Functional correspondence and its computational realization. Each row specifies an operation in the design.
<table><tr><td>Organizing principle</td><td>T-Router realization</td><td>Computational consequence</td></tr><tr><td>Distributed processing</td><td>Frozen decoder blocks</td><td>Every block produces its original nonlinear transformation on the received residual</td></tr><tr><td>Selective inter-area communication</td><td>Attention over completed block records</td><td>Later computations select from origin-indexed intermediate changes</td></tr><tr><td>Context-dependent coordination</td><td>Token-specific slots recurrent over depth</td><td>Retrieval uses an auxiliary history of preceding computational states</td></tr><tr><td>Modulation of influence</td><td>Signed RMS-calibrated writeback</td><td>Direction and relative magnitude are controlled separately</td></tr><tr><td>Coordinating system around existing processing</td><td>Full source-controller-writeback pathway</td><td>Adaptation capacity is concentrated in communication between frozen blocks</td></tr></table>

This mapping determines concrete design choices. A receiving block retains its full transformation instead of becoming an expert that may be skipped. Each source has an explicit depth origin rather than becoming an anonymous part of an accumulated hidden state. The same coordinate system is used throughout the paper: block index identifies an available computation, slot index identifies a component of the routing context, and token index identifies the causal position at which the decision occurs. The routing correspondence is operational: source weights determine which completed changes contribute to the mixture, and the gate controls their signed influence on the receiving block.

## D.2 Why the source bank and controller carry different information

The source bank preserves completed displacements, each compressed using a source-specific map. Its size grows with completed depth blocks, and an appended record remains fixed for subsequent reads in that forward call. The controller state S rewrites a fixed number of slots at each layer, accumulating update proposals through decay and slot allocation. The context P is a query-dependent readout of the updated slots, recomputed for each receiving layer.

These objects have complementary roles. The bank supplies content associated with earlier blocks; S summarizes the trajectory of residual states, source summaries, and layer identities. The current residual queries this state to obtain P. Routing and gating then combine P with the current residual to select source content and regulate its influence. The transported vector c comes from the source bank. This separation lets accessible sources grow without increasing the number of controller slots.

Table 30 : Source storage and controller state are complementary interfaces. Their diferent update rules organize the computation even when their widths coincide.
<table><tr><td>Property</td><td>Source bank</td><td>Depth controller</td></tr><tr><td>Index meaning</td><td>Completed block identity</td><td>Slot identity</td></tr><tr><td>Cardinality</td><td>Grows as blocks finish</td><td>Fixed at  $K = 8$ </td></tr><tr><td>Content width</td><td> $r = 2 5 6$ </td><td> $p = 2 5 6$  per slot</td></tr><tr><td>Write frequency</td><td>Once per completed block</td><td>Once per decoder layer</td></tr><tr><td>Write rule</td><td>Append compressed block change</td><td>Decay plus allocated shared proposal</td></tr><tr><td>Read function</td><td>Source attention produces ce</td><td>Slot attention produces  $P _ { \ell }$ </td></tr><tr><td>Direct contribution</td><td>Content projected into the residual</td><td>Context for source selection and gate input</td></tr><tr><td>Lifetime</td><td>One forward call</td><td>One forward call</td></tr></table>

Equal dimensions do not make the two representations interchangeable. A bank of seven records at layer 31 retains seven block identities. Eight controller slots at that layer can each contain a diferent mixture of many earlier proposals. The controller’s rank-one increment per layer is compatible with a higher-rank accumulated state, as shown in Appendix B.4. The two representations also have diferent sensitivities to intervention: changing a source record alters both retrieved content and later controller input, whereas changing only a controller state changes how existing source content is selected and scaled.

## D.3 Three coordinates of a residual intervention

The raw direction can be written as $w _ { \ell } = U _ { \ell } c _ { \ell }$ . This expression contains source selection through the attention-weighted source mixture and transport through the receiving projection. On the active branch, calibration maps the raw direction onto a sphere whose radius is the receiving residual norm. Finally, the signed gate chooses the intervention’s relative magnitude and orientation along that calibrated direction.

The three coordinates have distinct mathematical roles. Source attention is a distribution over candidate origins; it does not specify the norm of their projected sum. The projection supplies a receiver-specific low-dimensional subspace; a scalar gate does not change that subspace. Calibration changes the norm of a nonzero direction while preserving its ray, and the gate then selects a point along the line spanned by that direction. On the active branch, $| g _ { \ell } |$ has the direct interpretation $\| R _ { \ell } \| / \| h _ { \ell } \|$

Table 31 : Coordinates of controlled reuse. Distinct parameter groups control the origin, direction, and relative strength of the intervention.
<table><tr><td>Coordinate</td><td>Governing objects</td><td>Quantity it controls</td></tr><tr><td>Source preference</td><td>Query, source keys, depth context</td><td>Relative attention across completed computations</td></tr><tr><td>Content transport</td><td> $C _ { b } ,$  source value projection,  $U _ { \ell }$ </td><td>Source-to-receiver transformation within  $\operatorname { c o l } ( U _ { \ell } )$ </td></tr><tr><td>Relative influence</td><td>Target-RMS calibration and signed gate</td><td>Norm relative to the receiving residual and sign along the direction</td></tr></table>

This factorization explains why writeback calibration and routing structure interact. Sparsifying source selection changes the mixture delivered to the receiving projection. Normalizing stored deltas changes the geometry on which keys and values are built. Changing controller width changes the context available to the query and gate. A common fixed writeback coeficient can therefore act on diferent learned directions in diferent designs. The paired results in Appendix C.8 reveal this interaction at the configuration level. The matched component study in Table 20 separately tests the operations of the full interface.

The learned gate adds another degree of freedom to calibrated transport. Its input contains both the current residual and controller context, so the model can modulate an already selected direction diferently across tokens and receiving layers. A fixed gate retains token-dependent directions through source routing, but uses a common scalar coeficient. Thus “fixed” and “learned” refer to the strength coordinate; they do not divide the models into static and dynamic source-selection systems.

## D.4 Relation to alternative adaptation interfaces

A local low-rank weight update changes a layer matrix by ∆W = AB and acts on the current input to that matrix (Hu et al., 2022). A bottleneck adapter applies a learned local transformation around a layer (Houlsby et al., 2019). T-Router instead retains explicit earlier sources and chooses a combination of them at a receiving layer. Its transport matrices are also low rank, but their operands are completed block changes selected through an input-dependent depth route.

DenseNet exposes earlier layer features through fixed feature concatenation (Huang et al., 2017). T-Router reads compressed block changes through learned attention, projects the mixture into the receiving residual space, and calibrates its influence. This gives a fixed-width frozen backbone explicit access to historical computation, with reuse conditioned on the current residual and depth context.

The distinction can be expressed at a fixed receiver. With fixed source weights and no recurrent dependence, the change-dependent part of the raw writeback is a weighted sum of $T _ { \ell b } D _ { b }$ terms, with rank $( T _ { \ell b } ) \leq r$ . With the full module, the weights, records, controller context, calibration factor, and gate all vary with the input. A low-dimensional intervention space therefore coexists with a nonlinear input-dependent adaptation function. This is diferent from treating low rank as a statement that the entire adapted network is a fixed linear perturbation.

Delta Attention Residuals uses sublayer or block changes as routed content (Luo et al., 2026a); mHC-based finetuning learns read/write routing over multiple residual streams around frozen branches (Oldenburg et al., 2026). T-Router’s coupled interface assigns separate roles to compressed origin-indexed records, recurrent context, and relative-scale writeback. The bank preserves completed changes, while controller slots accumulate depth history. A current-state query reads those slots to obtain P, which jointly conditions source attention and the signed gate. This organization makes content reuse and its control trainable within one compact RL interface.

Reuse operates during the causal forward pass: a receiver draws on completed records to shape the input to its frozen transformation. During RL, the task objective assigns credit through the resulting computation, diferentiating the source, controller, routing, and writeback parameters. Frozen layers retain their input derivatives, connecting these communication choices to the output loss. Training therefore learns a reuse rule that executes directly during generation.

## D.5 Connecting the empirical comparisons to the interface

The component study connects the organization of adaptation capacity to reasoning quality. Block-change memory reaches 83.64 MathAvg versus 67.46 for hidden-state memory at the same allocation; removing retrieval gives 69.48. The full controller also exceeds a capacity-matched MLP, which has 1,646 more parameters and scores 72.80. These comparisons favor accessible changes and recurrent control within the same compact-budget regime. S accumulates depth history, while the current hidden state queries S to obtain P as the context for reuse.

Learned static routing gives 64.98, and removing the token-conditioned gate gives 63.78. The two variants address distinct receiving-layer decisions: which earlier changes contribute and how strongly their mixture enters the residual. All backbone layers continue to execute in order. The supplementary width, initialization, local-branch, and sparse-composite studies explore coupled design choices around this interface; their interaction patterns complement the component comparisons.

The resulting optimization target is a rule for composing existing intermediate contributions. The bank defines available content, depth context conditions its selection, and writeback regulates its influence on the next frozen transformation. Training these choices together yields 83.64 MathAvg with a 41.73M-parameter allocation, compared with 73.79 for full-parameter GRPO and 77.28 for matched-retry LoRA. Computation reuse thus connects the eficiency objective to both an explicit mechanism and its measured reasoning performance.