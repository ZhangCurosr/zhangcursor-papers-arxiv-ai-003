# WHERE DRAFT TREES LOSE TARGET MASS: EXIT-GUIDED SPECULATIVE DECODING

Shijing Hu<sup>1</sup> Xuancheng Ren<sup>1</sup> Zhihui Lu<sup>1∗</sup> Pan Zhou<sup>2</sup>

<sup>1</sup> Fudan University

sjhu24@m.fudan.edu.cn

<sup>2</sup> Singapore Management University xcren25@m.fudan.edu.cn lzh@fudan.edu.cn panzhou@smu.edu.sg

## ABSTRACT

Tree-based speculative decoding organizes draft tokens into a bounded tree so that multiple continuations can be verified in one target-model pass. However, the tree is constructed from draft-side scores while its usefulness is ultimately determined by the target model, creating a fundamental draft–target mismatch under a finite tree budget. We study two questions: once a draft tree is fixed, can a better exact verifier accept more draft tokens, and if not, how can target feedback improve the tree itself? We answer both through a target-flow view of the fixed draft tree. We identify a canonical exit law that describes where target continuations leave the tree, and prove that one plus the resulting target coverage is a sharp upper bound on the expected output-block length, including the bonus token, of any exact path verifier. Moreover, every verifier attaining this ceiling must realize the same exit and bonus-token law; representative predraw-and-follow and sequential residual verifiers already attain it. This theory directly yields Tree Exit Verification (TEV), an exact verifier that realizes the canonical law through one exit-node decision and one bonus-token decision, exposing a regular, level-parallel verification procedure. The same exit law also localizes where target probability is missing from the current tree. We use it as node-level feedback to train the drafter on its inferencetime draft trees, reallocating finite tree budget toward target-relevant regions. Experiments across dialogue, code, and mathematical reasoning validate the fixedtree equivalence, show that the Exit-Guided Draft-Tree Training (ExitTrain) improves average output-block length by 13%, and that TEV reduces verifier-stage latency by 15%, yielding 14% end-to-end speedup over DDTree. Together, our results separate the two remaining opportunities in bounded tree speculative decoding: better draft trees for higher acceptance, and more direct verification for lower latency. Code is available at https://github.com/hsj576/TEV.

## 1 INTRODUCTION

Large language models (LLMs) decode autoregressively, requiring an expensive target-model forward pass per token. Speculative decoding reduces this cost by using a lightweight draft model to propose future tokens that the target verifies in parallel (Leviathan et al., 2023; Chen et al., 2023). While early methods draft a single sequence, recent systems organize many draft tokens into a tree, allowing multiple possible continuations to share one target-model verification pass (Miao et al., 2024; Chen et al., 2024; Li et al., 2024b; 2025). Tree speculative decoding can therefore be viewed as three stages: the drafter proposes draft tokens and their scores; the tree builder selects and organizes a limited number of them into a draft tree; and the verifier uses the target model to determine how far generation can safely follow a path in that tree while preserving the target distribution. Existing systems implement verifier differently. DDTree’s predraw-and-follow verifier samples a target token at each tree node and follows a matching drafted child (Ringel & Romano, 2026), whereas EAGLE-style sequential residual verifier tests drafted children sequentially while updating the remaining target probability (Li et al., 2025). DFlash with DDTree provides one concrete realization of this drafter–builder–verifier pipeline (Chen et al., 2026; Ringel & Romano, 2026).

![](images/2477da979f0484d3cd8d058943de1c35a5787d15bc70d77f3aecae0f4a334a98.jpg)  
Figure 1: Target-flow view of tree speculative decoding. Blue denotes selected draft prefixes, gray dashed branches omitted candidates, red target-probability flow, and green reallocated tree budget. (a) Under a finite budget, draft-score-based tree construction can miss target-important branches, causing target mass to exit the tree. (b) On a fixed tree, prior exact verifiers realize the same optimal acceptance law through node-wise draws or tests, whereas TEV realizes it by direct exit-node sampling followed by bonus-token sampling. (c) Exit-guided training reallocates budget toward high-exit boundaries, improving target-mass coverage and output-block length.

Under a fixed tree budget, the draft tree determines which future prefixes can be verified in a single target forward pass. Among exponentially many possible prefixes, only a small number can appear in the tree. Since the tree must be built before the target model evaluates it, practical builders rely on draft-model scores (e.g., confidence) to decide which branches to keep (Li et al., 2024b; 2025; Ringel & Romano, 2026). Yet the usefulness of those branches is ultimately determined by the target model. When the draft and target prefer different continuations, limited tree budget may be spent on branches that look promising to the drafter but are rarely followed by the target, while target likely branches remain insufficiently expanded in the tree. Fig. 1(a) illustrates this draft–target tree-allocation gap. Although we use DDTree as our main instantiation, the same mismatch arises whenever draft-side information is used to construct a bounded draft tree before target verification.

This mismatch raises two fundamental questions. First, once a draft tree is fixed, how much freedom remains in exact verification? Recent work like HSD (Zhou et al., 2026) shows that improved verification can increase acceptance for a sampled draft chain. So can another exact verifier similarly accept more tokens from the same tree, or have existing exact verifiers already reached the tree’s maximum achievable acceptance, leaving only execution efficiency to optimize? Second, if existing verifiers already saturate the fixed-tree ceiling, how can target-model feedback improve future draf trees? Can we identify where target probability leaves the tree and use this signal to allocate the limited tree budget more effectively? These questions reveal two opportunities in tree speculative decoding: how much target probability the tree covers, and how efficiently that support is verified.

Our approach and contributions. We answer the both questions through a unified target-flow view of the realized draft tree. First, we develop a verifier-independent theory showing that the maximum expected output-block length on a fixed tree is one plus the target probability mass covered by the selected support. This theory exposes a canonical target-exit law, from which we derive Tree Exit Verification (TEV) as a direct and implementation-friendly realization of optimal fixed-tree verification. Finally, we use the same exit law as node-resolved training feedback to improve targetmass coverage under a finite tree budget, thereby increasing draft acceptance. Fig. 1 summarizes these three contributions.

First, we characterize the acceptance limit of exact path verifiers on a fixed tree: the maximum expected output-block length equals one plus its target-mass coverage, and every verifier attaining this bound induces the same canonical exit-node and bonus-token law. DDTree’s predraw-and follow (Ringel & Romano, 2026) and EAGLE-style exact sequential residual verification (Li et al., 2025) realize this law and thus have identical accepted-path distributions on the same tree. Once verification is saturated, higher acceptance requires better tree coverage, not a different verifier.

Second, we derive Tree Exit Verification (TEV), an exact, direct realization of this optimal law. Rather than using auxiliary per-node samples or sequential residual decisions, TEV samples an exit node directly and then a bonus token from its target residual (Fig. 1(b)). It preserves the fixed tree acceptance ceiling while enabling sibling-order-invariant, level-parallel, fixed-shape execution: statistical saturation does not imply computational saturation.

Third, we introduce Exit-guided Draft-Tree Training (ExitTrain) to improve finite-budget tree coverage. ExitTrain evaluates the draft tree with the frozen target under actual tree-parent contexts and uses exit mass to reweight draft likelihoods, guiding budget allocation toward high-impact exit boundaries (Fig. 1(c)). Unlike token- or feature-level alignment (Zhang et al., 2025; Hu et al., 2025; Li et al., 2025), position-wise distillation (Lei et al., 2026), or scalar tree-reward optimization (Hu et al., 2026), this supervision localizes uncovered target mass directly on the draft tree.

Finally, we evaluate ExitTrain + TEV across dialogue, code, and mathematical reasoning benchmarks and multiple target-model families, against DFlash (Chen et al., 2026), DDTree (Ringel & Romano, 2026), and Draft-OPD (Lei et al., 2026). Relative to DDTree with the same tree builder and budget, our system improves average output-block length (including the bonus token) by up to 13% and decode throughput by up to 14%. Separately, replacing predraw verification with TEV reduces verifier-stage latency by 15% without changing saturated expected output-block length on a fixed tree. These results support two complementary gains: better treesfor higher acceptance and direct verificationfor lower latency.

## 2 BACKGROUND AND FIXED-TREE SETUP

Tree speculative decoding. A drafter proposes candidates, a budget-B builder selects a prefixclosed tree, and one target pass lets a verifier return an accepted path plus a bonus token. Systems combine multi-head, feature-autoregressive, recurrent, or block-diffusion drafting with parallel tree verification (Cai et al., 2024; Li et al., 2024a; Cheng et al., 2024; Chen et al., 2026; Miao et al., 2024; Chen et al., 2024; Li et al., 2024b; 2025; Svirschevski et al., 2024). Builders optimize this support using draft-side expected acceptance (OPT-Tree), block-diffusion marginals (DDTree), or autoregressive correction (DARTree) (Wang et al., 2025; Ringel & Romano, 2026; Li et al., 2026). We instantiate DFlash with DDTree’s cumulative-score best-first builder (Chen et al., 2026; Ringel & Romano, 2026). These construction methods improve which tree is built, whereas our theory asks what any non-anticipating exact verifier can achieve once a prefix tree has been realized.

Two representative exact verifiers. DDTree’s predraw-and-follow independently samples one target token at each node (Ringel & Romano, 2026). Starting at the root, it follows a matching drafted child; otherwise, it stops and returns the sample as the bonus token. EAGLE-style exact sequential residual verification instead tests children in order (Li et al., 2025). After each rejection, it removes that child’s target mass and renormalizes the residual, following the first accepted child or sampling a bonus token if all are rejected. Sec. 3 shows that despite these different computations, both attain the same optimal law on a fixed tree. HSD (Zhou et al., 2026) instead verifies a sampled draft chain.

![](images/89f7842bdcc0e1dd22e13a14d98fdd3ac0a2c07afd96c6533af55c1d436697c1.jpg)

Related draft-model training. Prior work aligns drafters through on-policy or online distillation (Zhou et al., 2024; Liu et al., 2024), feature-, token-, or multistep objectives (Zhang et al., 2025; Hu et al., 2025; Li

Figure 2: Visualization of a fixed-tree.

et al., 2025), tree rewards (Hu et al., 2026), replayed trajectories (Lei et al., 2026), or acceptablepath modeling (Zou et al., 2026). Closest to ours, GTO optimizes a scalar tree reward (Hu et al., 2026), Learning to Draft learns throughput-driven depth and verification-size policies (Zhang et al., 2026), and VAT reweights simulated first-rejection positions (Gu et al., 2026). ExitTrain instead derives normalized node-level supervision from the canonical exit law of each realized finite tree, directly reallocating a fixed tree budget toward uncovered target mass.

## 3 THE CANONICAL EXIT LAW AND THE FIXED-TREE CEILING

We condition on a decoding context and a realized draft tree. By following target probability through this fixed tree, we characterize both the largest expected output-block length any exact verifier can achieve and the unique law attained at that ceiling.

Fixed-tree setup. Let X be the token vocabulary and let $p ( \cdot \mid c )$ denote the target continuation law given context c. A drafter and budget-B builder produce a finite, rooted, prefix-closed tree $\mathcal { T } = ( \mathcal { V } , \mathcal { E } , r ) \ ( \mathrm { F i g } . \ 2 )$ , where $\nu$ is the node set, E is the directed edge set, and r is the root. Set $\pi ( r ) = \epsilon ,$ the empty sequence. Let $\mathcal { X } ^ { \ast }$ denote the set of finite token sequences. Each node v represents a speculative continuation prefix $\pi ( v ) \in \mathcal { X } ^ { \ast }$ ; for $v \neq r ,$ , let $\mathrm { p a } ( v ) , x _ { v } , \mathrm { d } ( v ) = | \pi ( v ) |$ and $\operatorname { C h } ( v )$ denote its parent, incoming token, depth, and children, so that $\pi ( v ) = \pi ( \mathrm { p a } ( v ) ) \parallel x _ { v }$ Here ∥ denotes sequence–token concatenation, $\alpha \preceq \beta$ means that sequence α is a prefix of $\beta ,$ , and $\mathbf { 1 } [ \cdot ]$ denotes an indicator. Sibling tokens are distinct. Set $P ( r ) = 1 ;$ ; one target-model tree forward pass gives

$$
p _ { v } ( a ) : = p ( a \mid c , \pi ( v ) ) , \quad a \in \mathcal { X } ,\tag{1}
$$

$$
\begin{array} { r } { P ( v ) : = P ( \mathrm { p a } ( v ) ) p _ { \mathrm { p a } ( v ) } ( x _ { v } ) = p ( \pi ( v ) \mid c ) , } \end{array}\tag{2}
$$

where $P ( v )$ is the target prefix probability. A verifier returns a path ending at V and a bonus token Y . Let $D : = \mathrm { d } ( V )$ be the number of accepted draft tokens, and let $O : = \bar { \pi ( V ) } \parallel Y$ and $L : = | O | =$ $D + 1$ be the output block and its length, including the bonus token. Let $U$ denote auxiliary control randomness used, for example, by randomized early-stopping decisions and available before target tokens are revealed; randomness represented by the coupled target continuation $Z$ is not included in U. We call the verifier an exact path verifier if there exists a coupling with a target continuation $Z = ( Z _ { 1 } , Z _ { 2 } , . . . )$ such that

$$
Z \mid ( c , T , U ) \sim p ( \cdot \mid c ) , \qquad O = Z _ { 1 : L } { \mathrm { ~ a . s . } } ,\tag{3}
$$

and L is a stopping time with respect to $\mathcal { H } _ { t } : = \sigma ( T , U , Z _ { 1 } , \ldots , Z _ { t } )$ . Thus, the realized tree and independent control randomness are available at time zero, but neither may encode future target tokens. The target-prefix identity in Eq. (3) is sufficient for the fixed-tree ceiling and equality characteriza tion below; the stopping-time condition ensures that variable-length blocks compose safely across decoding rounds. Appendix A.1 formalizes random construction and round-wise composition.

## 3.1 TARGET FLOW AND THE CANONICAL EXIT LAW

We first describe how the target model interacts with a fixed draft tree, without referring to any particular verifier. Consider a continuation $Z \sim p ( \cdot \mid \stackrel { \cdot } { c } )$ from the target model. Starting from the root, Z follows matching drafted children for as long as its next token is represented in the tree. When the required next token is absent—or when a leaf is reached—the continuation leaves the draft tree. We call its deepest matched node the exit node, denoted by $V ^ { \star } ( Z )$ and illustrated in Fig. 2. Intuitively, the exit node marks where the draft tree stops covering a target continuation.

For a node v, let

$$
\mathcal { A } ( v ) : = \{ x _ { u } : u \in \mathrm { C h } ( v ) \} , \quad \rho ( v ) : = 1 - \sum _ { u \in \mathrm { C h } ( v ) } p _ { v } ( x _ { u } ) , \quad w ( v ) : = P ( v ) \rho ( v ) .\tag{4}
$$

Here, $\boldsymbol { \mathcal { A } } ( \boldsymbol { v } )$ contains the draft tokens available after v. $P ( v )$ is the probability that the target contin uation reaches v. Conditioned on reaching $v ,$ the local exit probability $\rho ( v )$ is the fraction of target next-token probability not covered by its drafted children. So $w ( v )$ is the overall probability that the target reaches v and exits the draft tree there. For a leaf, no drafted child is available, so $\rho ( v ) = 1$

Lemma 1 (Canonical exit law). The target-induced exit node satisfies

$$
\operatorname* { P r } ( V ^ { \star } ( Z ) = v \mid c , { \mathcal T } ) = w ( v ) , \qquad \sum _ { v \in \mathcal { V } } w ( v ) = 1 .\tag{5}
$$

The first identity in Lemma 1 says that $w ( v )$ is exactly the probability that a target continuation remains inside the draft tree up to v and leaves there. The second follows because every target continuation has exactly one exit node in a finite tree. Thus, $\{ w ( v ) \} _ { v \in \mathcal { V } }$ is a normalized distribution over where the target leaves the draft tree, which we call the canonical exit law. Importantly, this law is determined only by the target model and the fixed draft tree—not by a verifier. See the complete proof and probability-conservation argument in Appendix $_ { \mathrm { A } . 2 }$

## 3.2 FROM THE EXIT LAW TO THE FIXED-TREE ACCEPTANCE CEILING

Let $D _ { \mathsf { A } }$ and $L _ { \mathsf { A } } = D _ { \mathsf { A } } + 1$ denote, respectively, the accepted draft-token count and output-block length of an exact path verifier A. Accepting node u requires the target continuation to contain prefix $\pi ( u )$ , so its acceptance probability is at most $P ( u )$ . Summing these node-wise limits motivates

$$
C _ { p } ( \mathcal { T } ) : = \sum _ { \boldsymbol { u } \in \mathcal { V } \setminus \{ \boldsymbol { r } \} } P ( \boldsymbol { u } ) = \sum _ { \boldsymbol { v } \in \mathcal { V } } w ( \boldsymbol { v } ) \mathrm { d } ( \boldsymbol { v } ) ,\tag{6}
$$

which we call the target coverage of the draft tree.

The first form sums the target probability of every represented non-root prefix. The second, which follows from the exit law, identifies the same quantity as the expected exit depth, i.e., the expected number of accepted draft tokens. Because every round also emits one bonus token, the corresponding expected output-block length at saturation is $1 + C _ { p } ( \mathcal { T } )$ ).

For $\rho ( v ) > 0$ , define the residual target distribution

$$
p _ { v } ^ { \mathrm { r e s } } ( a ) : = p _ { v } ( a ) \mathbf { 1 } [ a \notin { \mathcal { A } } ( v ) ] / \rho ( v ) .\tag{7}
$$

which removes drafted-child tokens and renormalizes; at a leaf, $p _ { v } ^ { \mathrm { r e s } } \ = \ p _ { v }$ . We call a verifier saturated when $\mathbb { E } [ D \mid c , { \mathcal { T } } ] = C _ { p } ( { \mathcal { T } } )$ , equivalently when $\mathbb { E } [ L \mid c , \bar { \mathcal { T } } ] ^ { \sim } = 1 + \bar { C } _ { p } ( \mathcal { T } )$ . The following theorem gives this sharp ceiling and the unique output law at equality; Appendix A.3 provides the proof.

Theorem 1 (Fixed-tree ceiling and canonical saturation). For any exact path verifier A and any non-root node $u ,$

$$
\operatorname* { P r } _ { \mathsf { A } } ( u i s a c c e p t e d | c , \mathcal { T } ) \leq P ( u ) .\tag{8}
$$

Consequently,

$$
\mathbb { E } [ D _ { \mathsf { A } } \mid c , { \mathcal { T } } ] \leq C _ { p } ( { \mathcal { T } } ) , \qquad \mathbb { E } [ L _ { \mathsf { A } } \mid c , { \mathcal { T } } ] \leq 1 + C _ { p } ( { \mathcal { T } } ) .\tag{9}
$$

If either length bound is tight, every node-wise bound is tight and, for every v with $w ( v ) > 0 ,$

$$
\mathrm { P r } _ { \mathsf { A } } ( V = v \mid c , \mathcal { T } ) = w ( v ) , \qquad Y \mid ( V = v , c , \mathcal { T } ) \sim p _ { v } ^ { \mathrm { r e s } } .\tag{10}
$$

Hence, all saturated exact path verifiers induce the same accepted-path and bonus-token law.

The theorem has two implications. First, $C _ { p } ( \mathcal { T } )$ is the maximum expected number of accepted draft tokens, and $1 + C _ { p } ( \mathcal { T } )$ is the maximum expected output-block length reported by our experimental metric: no exact verifier can recover a drafted token absent from the tree. Second, saturation makes every node-wise bound tight, forcing the exit law w and residual bonus-token law $p _ { v } ^ { \mathrm { r e s } }$ . Saturated verifiers may therefore differ computationally but not in their accepted-path distribution. The representative verifiers in Sec. 2 attain this ceiling (Appendix A.4).

Corollary 1 (Representative verifiers are saturated). Predraw-and-follow in DDTree (Ringel & Ro mano, 2026) and exact sequential residual verification in EAGLE-style tree decoding $( L i ~ e t ~ a l .$ 2025) both attain the fixed-tree ceiling. Therefore, on the same draft tree and target distributions, they share the canonical exit law, accepted-path distribution, expected accepted draft-token count, and expected output-block length.

Whenever node v is reached, both representative verifiers follow each drafted child u with probability $p _ { v } ( x _ { u } )$ and exit with the remaining probability $\rho ( v )$ . They therefore realize the same canonical law despite different internal computations.

## 3.3 WHAT REMAINS AFTER FIXED-TREE SATURATION

The fixed-tree result leaves two opportunities. The same law can be executed more efficiently, motivating TEV in Sec. 4: statistical saturation does not imply computational saturation. Higher output-block length, however, requires increasing $C _ { p } ( \mathcal { T } )$ itself. A large $w ( v )$ localizes target mass that reaches v but leaves the current tree, motivating the training method in Sec. 5. This result complements HSD (Zhou et al., 2026), which exploits target-to-draft ratios while verifying a random sampled chain rather than conditioning on a realized tree (Appendix A.5).

## 4 TEV: DIRECT VERIFICATION FROM THE CANONICAL EXIT LAW

Sec. 3 resolves the statistical question: saturated exact verifiers share the same canonical exit law and output-block length on a fixed tree. The remaining freedom is computational—how can this common law be realized efficiently? Existing verifiers recover it through per-node draws or sequential child decisions. Tree Exit Verification (TEV) instead realizes it directly.

Theorem 1 specifies the required output: exit at v with probability $w ( v )$ and then draw the bonus token from $p _ { v } ^ { \mathrm { r e s } }$ . TEV therefore computes the exit distribution, draws one exit node, and samples one residual bonus token. Algorithm 1 in Appendix B gives the complete procedure after the shared target-model tree forward pass produces $\{ p _ { v } \}$ . The drafter need not be queried again: it has already determined what appears in the realized tree, while the target distributions determine how probability flows through that support.

Because Algorithm 1 implements the canonical law, its correctness follows from Sec. 3.2; $\mathsf { A p - }$   
pendix B.1 gives the proof.

Corollary 2 (TEV is exact and saturated). For any context and realized prefix-closed draft tree, TEV is an exact path verifier and attains the fixed-tree ceiling. In particular, we have

$$
\mathrm { P r } _ { \mathrm { T E V } } ( u \ i s \ a c c e p t e d \mid c , \mathcal { T } ) = P ( u ) \quad ( \forall u \in \mathcal { V } \backslash \{ r \} ) , \qquad \mathbb { E } [ L _ { \mathrm { T E V } } \ \mid c , \mathcal { T } ] = 1 + C _ { p } ( \mathcal { T } ) .\tag{11}
$$

By Lemma 1, a target continuation exits according to w and produces its next token according to $p _ { v } ^ { \mathrm { r e s } }$ . TEV reproduces these distributions; node u is accepted exactly when the sampled exit lies in its subtree, whose total exit mass is $P ( u )$

Efficiency. TEV shares the accepted-path law of predraw and sequential residual verification, but removes their per-node samples or data-dependent child tests. It uses node-local reductions, depthwise propagation, and one exit-node and bonus-token draw, yielding a sibling-order-invariant, fixedshape, level-parallel procedure. Sec. 6.2 evaluates this latency benefit separately from output-block length.

## 5 EXITTRAIN: EXIT-GUIDED DRAFT-TREE TRAINING

Sec. 4 addresses the remaining computational freedom: TEV changes how the canonical law is executed, not the fixed-tree acceptance ceiling. We now turn to the second question from Sec. 1: how can targetfeedback build a better draft tree? Theorem 1 provides the key: once verification is saturated, higher output-block length requires greater target coverage, and the exit law reveals where that coverage is missing.

Let H be the maximum draft depth. For an expandable node v with $\mathrm { d } ( v ) < H$ and a missing child token a $\notin \boldsymbol { A } ( \boldsymbol { v } )$ , write ${ \mathcal { T } } \oplus ( v , { \bar { a } } )$ for the tree obtained by adding child $( v , a )$ . Then

$$
C _ { p } ( \mathcal { T } \oplus ( v , a ) ) - C _ { p } ( \mathcal { T } ) = P ( v ) p _ { v } ( a ) ,\tag{12}
$$

$$
\sum _ { a \not \in A ( v ) } P ( v ) p _ { v } ( a ) = P ( v ) \rho ( v ) = w ( v ) .\tag{13}
$$

Hence, for an expandable node $v , w ( v )$ measures the total one-step target coverage currently missing below v. A large $w ( v )$ means that the target frequently reaches this prefix but substantial nexttoken probability is not represented by its drafted children. These are precisely the regions where allocating more tree capacity can keep target continuations inside the tree for longer. This expansion interpretation does not apply to a hard-horizon leaf. If H is the maximum draft depth and $\operatorname { d } ( v ) = H$ then $\rho ( v ) = 1$ and $w \bar { ( v ) } \stackrel { } { = } P ( v )$ ; weighting its incoming token rewards a path that already carries target mass to the full horizon rather than missing one-step coverage.

This observation directly suggests our training strategy. Let $q _ { \theta }$ be the current drafter with parameters θ. For each training context $c , q _ { \theta }$ and the same tree builder used at inference construct a bounded tree $\widehat { T } _ { \theta } ( c )$ . We keep its structure fixed within the update, evaluate every node with the frozen target model under its actual tree-parent context, and compute the exit probabilities $w ( v )$ . Training therefore receives node-level feedback about where the current inference-time draft tree fails to cover target probability, rather than supervision from only a single draft path.

For a minibatch $\boldsymbol { B }$ of contexts $\{ c _ { i } \} _ { i \in B } .$ , let $\widehat { \mathcal { T } } _ { i } \ = \ ( \widehat { \mathcal { V } } _ { i } , \widehat { \mathcal { E } } _ { i } , r )$ be the realized tree for example $i ,$ let $w _ { i } ( v )$ be its detached target-derived exit weight, and let $q _ { \theta , d } ( \cdot \mid c _ { i } )$ be DFlash’s position-wise marginal distribution at future draft position d. We reweight the likelihood of the incoming token at each non-root node:

$$
\ell _ { \mathrm { e x i t } } ( \theta ) : = - \frac { \sum _ { i \in B } \sum _ { v \in \widehat { \mathcal { V } } _ { i } \setminus \{ r \} } w _ { i } ( v ) \log q _ { \theta , \mathrm { d } ( v ) } ( x _ { v } \mid c _ { i } ) } { \sum _ { i \in B } ( | \widehat { \mathcal { V } } _ { i } | - 1 ) } .\tag{14}
$$

Consequently, nodes at the same depth share the same draft row even when they have different treeparent prefixes; in contrast, $p _ { v }$ is evaluated under the actual prefix $( c _ { i } , \pi ( v ) )$ . All target-derived quantities and the realized tree structure are detached; gradients flow only through the drafter.

Why reinforce the position-wise marginal probability of the incoming draft token of an exit node? In DDTree, a prefix is ranked by cumulative draft scores along its path. Increasing the marginal probability assigned to $x _ { v }$ at position $\mathrm { d } ( v )$ therefore raises the future priority of v and its descendants, making the builder more likely to retain and further expand that region. Weighting this update by $w ( v )$ directs future tree budget toward prefixes that the target frequently reaches but the current tree insufficiently expands. Because the top-B builder is discrete, this is a tree-allocation surrogate, not the exact gradient of target coverage; its role is to improve where the bounded tree spends its nodes.

Finally, let $\mathcal { L } _ { \mathrm { t o k } }$ denote the standard dense token-prediction loss and let $\lambda _ { \mathrm { e x i t } } \geq 0$ control the exitguided term. We optimize

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { t o k } } ( \theta ) + \lambda _ { \mathrm { e x i t } } \mathcal { L } _ { \mathrm { e x i t } } ( \theta ) . } \end{array}\tag{15}
$$

The two terms are complementary: the dense token loss teaches the drafter what to predict, while the exit-guided loss teaches a bounded tree where to allocate its capacity. Our default loss includes hardhorizon leaves, so w(v) has two roles: missing-coverage credit at expandable nodes and successfuldeep-reach credit at depth H. The root requires separate treatment: $w ( r )$ measures target mass missed by every first-position draft token, but r has no incoming draft edge and is omitted from $\mathcal { L } _ { \mathrm { e x i t } }$ . Consequently, the included exit weights sum to $1 - w ( r )$ rather than one, and the root-exit mass receives no direct credit from $\mathcal { L } _ { \mathrm { e x i t } } ;$ coverage of missing first-position tokens is addressed through the dense token objective. In this way, the same canonical exit law serves two distinct roles: TEV uses it to realize fixed-tree verification directly, while exit-guided training uses it to improve the tree itself. Sec. 6.2 studies alternative weights, and Appendix C.2 gives an efficient implementation.

## 6 EXPERIMENTS

Setup. We evaluate Qwen3-4B/8B (Yang et al., 2025), LLaMA-3.1-8B (Grattafiori et al., 2024), Gemma-4-12B-IT (Gemma Team, 2026), and Qwen3-Coder-30B-A3B (Yang et al., 2025) on MT-Bench (Zheng et al., 2023), HumanEval (Chen et al., 2021), GSM8K (Cobbe et al., 2021), and MATH-500 (Hendrycks et al., 2021). Drafter training uses open-perfectblend (Xu et al., 2024) prompts and online responses rolled out by each corresponding target model (Appendix D).

Methods. DDTree uses the released DFlash drafter, cumulative-score best-first builder, and predraw verifier (Chen et al., 2026; Ringel & Romano, 2026); Draft-OPD replaces only the drafter with its position-wise distilled checkpoint (Lei et al., 2026). ExitTrain + TEV denotes the complete system combining our exit-guided drafter with TEV. Appendix D gives full details.

Metrics. Because all methods preserve the target distribution, we report speedup over autoregressive decoding (SR) and average output-block length per draft–verify cycle $\mathbf { \bar { \rho } } ( \tau = \mathbf { \mathbb { E } } [ L ] )$ ), including the bonus token, at $T \in \{ 0 , 1 \}$ }.

## 6.1 MAIN RESULTS

Table 1 shows that ExitTrain + TEV achieves the highest τ and SR in every reported setting. The Qwen3-4B/8B comparisons provide the most controlled view because they use the same builder and budget and include both DDTree and Draft-OPD. Across the two model sizes and temperatures, ExitTrain + TEV improves average τ over DDTree by 8.8%–13.8% and average SR by 11.5%– 13.8%. It also exceeds Draft-OPD by 4.2%–6.9% in τ and 5.5%–7.3% in SR, showing that the gains persist over both the released and position-wise-distilled drafters under greedy and stochastic decoding.

The improvement also transfers across target-model families. On LLaMA-3.1-8B, ExitTrain + TEV improves average τ by 17.7%/10.4% and SR by 16.1%/10.1% at $T \mathrm { = } 0 / 1$ The corresponding τ and SR gains are 5.6%/4.8% and 5.7%/4.8% on Gemma-4-12B-IT, and 5.2%/4.3% and 5.2%/4.5% on Qwen3-Coder-30B-A3B. Thus, the advantage remains positive for dense, instruction-tuned, and MoE targets, although its magnitude varies across model families.

The gains also exhibit meaningful task variation. On Qwen3, GSM8K yields the largest increases in τ: 20.5%–22.4% at T=0 and 16.3%–17.8% at $T { = } 1$ , while HumanEval shows smaller but consistent improvements. More broadly, increases in τ translate consistently into higher SR. Because the builder and budget $( B = 6 4 )$ are fixed, the additional accepted length cannot come from verifying more candidates; it is the empirical signature of better target-prefix coverage, which raises the saturated output-block length $1 + C _ { p } ( \mathcal { T } )$

Table 1: Speedup ratio (SR) and average output-block length including the bonus token (τ) on MT-Bench, HumanEval, GSM8K, and MATH-500. All tree-based systems share the same builder and tree budget B = 64. ExitTrain + TEV combines our exit-guided drafter with TEV.
<table><tr><td colspan="2"></td><td colspan="2">MT-Bench</td><td colspan="2">HumanEval</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">Average</td></tr><tr><td>Model</td><td>Method</td><td>∥SR↑ τ↑</td><td></td><td>SR↑</td><td>τ↑</td><td>SR↑</td><td>τ↑</td><td>SR↑</td><td>τ↑</td><td>SR↑ τ ↑</td><td></td></tr><tr><td colspan="10">Temperature = 0</td></tr><tr><td>Qwen3-4B</td><td>DDTree Draft-OPD ExitTrain + TEV</td><td>3.81 4.16 4.28</td><td>5.41 5.73 6.17</td><td>5.71 5.96 6.09</td><td>8.13 8.35 8.74</td><td>5.65 6.11 6.91</td><td>8.48 9.30 10.22</td><td>6.83 7.02 7.23</td><td>9.16 10.14 10.26</td><td>5.50 5.81</td><td>7.80 8.38 8.85</td></tr><tr><td>Qwen3-8B</td><td>DDTree Draft-OPD ExitTrain + TEV</td><td>3.75 3.90 4.22</td><td>5.54 5.92 6.12</td><td>5.83 6.02 6.15</td><td>8.01 8.46 8.73</td><td>5.88 6.27 7.20</td><td>8.49 9.33 10.39</td><td>6.23 6.84 7.15</td><td>9.76 10.17 10.69</td><td>6.13 5.43 5.76 6.18</td><td>7.95 8.47 8.98</td></tr><tr><td>LLaMA-3.1-8B</td><td>DDTree ExitTrain + TEV</td><td>3.55 3.69</td><td>5.53 5.82</td><td>4.33 4.72</td><td>6.53 7.24</td><td>3.79 4.75</td><td>5.89 7.48</td><td>3.70 4.70</td><td>5.71 7.34</td><td>3.84 4.46</td><td>5.92 6.97</td></tr><tr><td>Gemma-4-12B-IT</td><td>DDTree ExitTrain + TEV</td><td>4.17 4.35</td><td>5.44 5.57</td><td>4.83 5.09</td><td>5.66 5.92</td><td>4.39 4.70</td><td>5.29 5.66</td><td>4.87 5.17</td><td>5.86 6.33</td><td>4.57 4.83</td><td>5.56 5.87</td></tr><tr><td>Qwen3-Coder-30B-A3B</td><td>DDTree ExitTrain + TEV</td><td>3.70 3.83</td><td>4.87 5.03</td><td>7.57 7.83</td><td>9.95 10.32</td><td>5.26 5.61</td><td>6.81 7.43</td><td>5.61 6.00</td><td>7.38 7.74</td><td>5.53 5.82</td><td>7.25 7.63</td></tr><tr><td colspan="10">Temperature = 1 DDTree 3.44 5.04 5.22 7.53</td></tr><tr><td>Qwen3-4B</td><td>Draft-OPD ExitTrain + TEV</td><td>3.74 3.98</td><td>5.26 5.86</td><td>5.49 5.64</td><td>7.80 8.06</td><td>5.48 5.88 6.47</td><td>7.85 8.58 9.25</td><td>5.65 6.08 6.26</td><td>7.88 8.50 9.08</td><td>4.95 5.30 5.59</td><td>7.08 7.54 8.06</td></tr><tr><td>Qwen3-8B</td><td>DDTree Draft-OPD ExitTrain + TEV</td><td>3.42 3.51 3.74</td><td>5.03 5.21 5.40</td><td>4.83 4.95 5.11</td><td>7.06 7.29 7.47</td><td>5.23 5.79 6.35</td><td>7.84 8.57 9.12</td><td>5.50 5.66 5.96</td><td>8.17 8.28 8.59</td><td>4.74 4.98 5.29</td><td>7.03 7.34 7.65</td></tr><tr><td>LLaMA-3.1-8B</td><td>DDTree ExitTrain + TEV</td><td>2.43 2.52</td><td>3.97 4.08</td><td>3.71 3.84</td><td>5.67 5.93</td><td>3.07 3.67</td><td>4.81 5.77</td><td>2.26 2.63</td><td>3.71 4.25</td><td>2.87 3.16</td><td>4.54 5.01</td></tr><tr><td>Gemma-4-12B-IT</td><td>DDTree ExitTrain + TEV</td><td>3.65 3.83</td><td>4.71 4.83</td><td>4.13 4.39</td><td>5.06 5.28</td><td>4.48 4.70</td><td>5.41 5.83</td><td>4.57 4.74</td><td>5.78 6.02</td><td>4.21 4.41</td><td>5.24 5.49</td></tr><tr><td>Qwen3-Coder-30B-A3B</td><td>DDTree ExitTrain + TEV</td><td>3.48 3.61</td><td>4.53 4.72</td><td>7.26 7.39</td><td>9.61 9.92</td><td>5.22 5.57</td><td>6.75 7.17</td><td>5.43 5.78</td><td>7.13 7.42</td><td>5.35 5.59</td><td>7.01 7.31</td></tr></table>

Crossing drafter training and verification. The crossed evaluation in Appendix D.2 confirms this attribution on Qwen3-4B at T=1. Holding Predraw fixed, exit training raises average τ from 7.08 to 8.05 and SR from 4.95× to 5.55×; with TEV, the corresponding changes are 7.09 to 8.06 and 4.97× to 5.59×. In contrast, for a fixed drafter, Predraw and TEV differ by at most 0.03 tokens per benchmark, consistent with equal expected output-block length under Theorem 1; the small empirical discrepancies arise from separate finite-sample decoding runs. TEV nevertheless improves average end-to-end SR by 0.4% with the original drafter and 0.7% with the exit-trained drafter. Together, these results separate the two contributions: ExitTrain supplies most of the acceptance gain by improving the tree, while TEV provides orthogonal execution savings, reducing verifierdecision and cache-commit latency by 12.0%–24.3% across the tested budgets (Appendix D.1).

## 6.2 ABLATION STUDY

Exit-guided training. We compare the released DFlash checkpoint with three variants trained under matched initialization, data, and compute: token-only finetuning, prefix-weighted training, and exit-guided training. All use the same builder and B = 64. The prefix-weighted control replaces w(v) with P(v), rescaled to match the total non-root exit weight while retaining the same loss denominator. The two controls therefore separate the benefit of additional training from the benefit of assigning credit specifically to exit boundaries.

Table 2 first shows that additional training is itself useful: token-only finetuning increases average τ from 7.80/7.08 to 8.36/7.47 at T=0/1. Exit-guided training then improves over this matched control by a further 5.9%/7.9% in τ and 6.4%/6.3% in SR, demonstrating that its gain is not explained by additional training on open-perfectblend alone. The SR numbers compare the complete systems; the crossed evaluation in Sec. 6.1 shows that TEV contributes only 0.7% end-to-end with the exit-trained drafter, so the dominant improvement comes from training.

Table 2: Effect of the draft-training objective on Qwen3-4B. All three trained drafters share the initialization, the open-perfectblend mixture, the compute budget, the tree builder, and the budget B = 64
<table><tr><td rowspan="2">Draft training objective </td><td colspan="2">MT-Bench</td><td colspan="2">HumanEval</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">Average</td></tr><tr><td>|SR↑ τ↑</td><td></td><td></td><td>SR↑ τ↑</td><td>SR↑</td><td>τ ↑</td><td>SR↑</td><td>τ↑</td><td>|SR↑ τ↑</td><td></td></tr><tr><td colspan="9">Temperature = 0</td><td colspan="2"></td></tr><tr><td>Original DFlash drafter</td><td>3.81</td><td>5.41</td><td>5.71</td><td>8.13</td><td>5.65</td><td>8.48</td><td>6.83</td><td>9.16</td><td>5.50</td><td>7.80</td></tr><tr><td>Finetuned DFlash drafter</td><td>3.97</td><td>5.71</td><td>5.83</td><td>8.36</td><td>6.33</td><td>9.54</td><td>6.91</td><td>9.83</td><td>5.76</td><td>8.36</td></tr><tr><td>Prefix weight P(v)</td><td>4.05</td><td>5.82</td><td>5.99</td><td>8.43</td><td>6.07</td><td>9.25</td><td>6.71</td><td>9.33</td><td>5.71</td><td>8.21</td></tr><tr><td>Exit weight w(v) (ours)</td><td>4.28</td><td>6.17</td><td>6.09</td><td>8.74</td><td>6.91</td><td>10.22</td><td>7.23</td><td>10.26</td><td>6.13</td><td>8.85</td></tr><tr><td colspan="9">Temperature = 1</td></tr><tr><td>Original DFlash drafter</td><td>3.44</td><td>5.04</td><td>5.22</td><td>7.53</td><td>5.48</td><td>7.85</td><td>5.65</td><td>7.88</td><td>4.95</td><td>7.08</td></tr><tr><td>Finetuned DFlash drafter</td><td>3.58</td><td>5.20</td><td>5.43</td><td>7.63</td><td>6.18</td><td>8.58</td><td>5.86</td><td>8.46</td><td>5.26</td><td>7.47</td></tr><tr><td>Prefix weight P(v)</td><td>3.69</td><td>5.45</td><td>5.48</td><td>7.88</td><td>6.26</td><td>8.85</td><td>5.99</td><td>8.68</td><td>5.35</td><td>7.72</td></tr><tr><td>Exit weight w(v) (ours)</td><td>3.98</td><td>5.86</td><td>5.64</td><td>8.06</td><td>6.47</td><td>9.25</td><td>6.26</td><td>9.08</td><td>5.59</td><td>8.06</td></tr></table>

The prefix-weighted control tests whether target reachability alone provides sufficient tree supervision. It reaches $\tau = 8 . 2 1 / 7 . 7 2$ , whereas exit weighting reaches 8.85/8.06, improvements of 7.8%/4.4%; the corresponding SR gains are 7.4%/4.5%. Moreover, prefix weighting trails tokenonly finetuning at $T { = } 0$ , despite using target prefix probabilities. The distinction is that $P ( v )$ measures how often a node is reached, whereas $w ( v ) = P ( v ) \rho ( v )$ additionally identifies reached boundaries whose selected children leave substantial target mass uncovered (Eq. (13)). The results therefore support localized exit feedback rather than reachability weighting alone.

Verifier-stage efficiency. Appendix D.1 and Table 3 isolate verifier execution by holding the drafter, builder, and realized tree fixed and excluding the shared target-model tree forward pass. TEV reduces verifier-decision and cache-commit latency at every tested budget: 12.9%/12.0% at B = 32, 15.2%/14.0% at B = 64, and 24.3%/17.1% at B = 128 for $T = 0 . { \bar { 6 / 1 . 0 } }$ . The increasing advantage at wider trees is consistent with replacing Predraw’s per-node sampling with level-wise probability propagation and one exit-node draw. Because this stage occupies only a small fraction of each decoding cycle, these reductions translate into the smaller end-to-end gains observed in the crossed evaluation.

Tree-budget robustness. Appendix D.3 and Table 5 evaluate the B = 64-trained drafter at $B \in$ {32, 64, 128} without budget-specific retraining. ExitTrain + TEV remains best in all six budget– temperature settings: relative to DDTree, it improves τ by 11.6%–13.8% and SR by 11.4%–12.9%; relative to Draft-OPD, the gains are 5.6%–6.9% and 4.5%–6.2%. Its advantage therefore persists for both narrower and wider inference trees, indicating that the learned exit-guided allocation is not tied to the training-time tree size.

## 7 CONCLUSION

We study tree speculative decoding through target probability flow and show that the maximum expected output-block length on a fixed tree is one plus its target-mass coverage. The canonical exit law gives a verifier-independent ceiling: saturating exact verifiers share the same exit and bonus-token law, so missing support cannot be recovered by verification alone. This suggests two complementary directions. TEV realizes the optimal law with a regular, level-parallel procedure to reduce verification cost, while exit-guided training uses node-level exit signals to improve finite-budget coverage and output-block length. Overall, our framework separates what tree to build from how to verify it efficiently.

Limitations. ExitTrain incurs one-time training overhead from target rollouts and tree evaluation, but fine-tunes released DFlash checkpoints without architectural or verifier changes and can be amortized during deployment. Our systems results are limited to batch-one decoding with DDTree; alternative builders and high-throughput serving remain future work.

## AI USE STATEMENT

Generative AI tools were used solely for language polishing and improving the clarity and readability of the manuscript. The authors reviewed all AI-assisted edits and take full responsibility for the final content of this work.

## ETHICS STATEMENT

This work studies the efficiency of lossless large language model inference. It does not involve human subjects, private user data, or the collection of new personal information. Our experiments use publicly available models and benchmarks under their respective terms of use. By construction, TEV preserves the output distribution of the target model and does not intentionally modify its capabilities, safety behavior, or intended use. More efficient inference may reduce computational and energy costs, but may also lower the cost of deploying language models at scale. Consequently, downstream uses of our method inherit the potential safety, bias, privacy, and misuse risks of the underlying target models. Users remain responsible for evaluating these risks and complying with the licenses and usage policies of the deployed models and datasets.

## REPRODUCIBILITY STATEMENT

Sections 3–5 provide the definitions, assumptions, fixed-tree results, TEV construction, and exitguided training objective used in this work. Complete proofs of the fixed-tree ceiling, verifier equivalence, and TEV exactness are provided in Appendices A and B. Additional derivations and implementation details for the exit-guided objective are given in Appendix C, while the model, dataset, training, hardware, and evaluation configurations are documented in Appendix D. An anonymized repository at https://github.com/hsj576/TEV will provide the training and inference code, configuration files, evaluation scripts, and instructions required to reproduce the reported experiments.

## REFERENCES

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 5209–5235. PMLR, 2024. 3

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023. URL https://arxiv.org/abs/2302.01318. 1

Jian Chen, Yesheng Liang, and Zhijian Liu. DFlash: Block diffusion for flash speculative decoding. In International Conference on Machine Learning, 2026. URL https://icml.cc/ virtual/2026/poster/64301. 1, 3, 7

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https:// arxiv.org/abs/2107.03374. 7

Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable and robust speculative decoding. In Advances in Neural Information Processing Systems, volume 37, pp. 129531–129563, 2024. 1, 3

Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang, and Yi Wang. Recurrent drafter for fast speculative decoding in large language models. arXiv preprint arXiv:2403.09919, 2024. URL https://arxiv.org/abs/2403.09919. 3

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John

Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168. 7

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607. 02770. 7

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https: //arxiv.org/abs/2407.21783. 7

Geonmo Gu, Byeongho Heo, HeeJae Jun, Yoohoon Kang, Sangmin Lee, Sangdoo Yun, and Dongyoon Han. Verification-aware training for speculative decoding. arXiv preprint arXiv:2608.30135, 2026. URL https://arxiv.org/abs/2608.30135. 3

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. 7

Shijing Hu, Jingyang Li, Xingyu Xie, Zhihui Lu, Kim-Chuan Toh, and Pan Zhou. GRIFFIN: Effective token alignment for faster speculative decoding. In Advances in Neural Information Processing Systems, volume 38, pp. 125648–125673, 2025. 3

Shijing Hu, Jingyang Li, Zhihui Lu, and Pan Zhou. Bridging draft policy misalignment: Group tree optimization for speculative decoding. In International Conference on Learning Representations, 2026. URL https://iclr.cc/virtual/2026/poster/10008369. 3

Haodi Lei, Yafu Li, Haoran Zhang, Shunkai Zhang, Qianjia Cheng, Xiaoye Qu, Ganqu Cui, Bowen Zhou, Ning Ding, Yun Luo, and Yu Cheng. Draft-opd: On-policy distillation for speculative draft models. arXiv preprint arXiv:2605.29343, 2026. URL https://arxiv.org/abs/2605. 29343. 3, 7

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 19274–19286. PMLR, 2023. 1

Tianyi Li, Yaxin Luo, Xinyi Shang, and Zhiqiang Shen. DARTree: Speculative diffusion decoding with autoregressive draft trees. arXiv preprint arXiv:2608.13524, 2026. URL https: //arxiv.org/abs/2608.13524. 3

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative sampling requires rethinking feature uncertainty. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 28935–28948. PMLR, 2024a. 3

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-2: Faster inference of language models with dynamic draft trees. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7421–7432. Association for Computational Linguistics, 2024b. doi: 10.18653/v1/2024.emnlp-main.422. 1, 2, 3

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Advances in Neural Information Processing Systems, volume 38, pp. 136737–136756, 2025. 1, 2, 3, 5

Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Online speculative decoding. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 31131–31146. PMLR, 2024. 3

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, Volume 3, pp. 932–949, 2024. doi: 10.1145/3620666.3651335. 1, 3

Liran Ringel and Yaniv Romano. Accelerating speculative decoding with block diffusion draft trees. arXivpreprint arXiv:2604.12989, 2026. URL https://arxiv.org/abs/2604.12989. 1, 2, 3, 5, 7

Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin. Specexec: Massively parallel speculative decoding for interactive LLM inference on consumer devices. In Advances in Neural Information Processing Systems, volume 37, pp. 16342–16368, 2024. 3

Jikai Wang, Yi Su, Juntao Li, Qingrong Xia, Zi Ye, Xinyu Duan, Zhefeng Wang, and Min Zhang. Opt-tree: Speculative decoding with adaptive draft tree structure. Transactions of the Association for Computational Linguistics, 13:188–199, 2025. 3

Tengyu Xu, Eryk Helenowski, Karthik Abinav Sankararaman, Di Jin, Kaiyan Peng, Eric Han, Shaoliang Nie, Chen Zhu, Hejia Zhang, Wenxuan Zhou, Zhouhao Zeng, Yun He, Karishma Mandyam, Arya Talabzadeh, Madian Khabsa, Gabriel Cohen, Yuandong Tian, Hao Ma, Sinong Wang, and Han Fang. The perfect blend: Redefining RLHF with mixture of judges, 2024. URL https://arxiv.org/abs/2409.20370. 7, 21

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388. 7

Jiebin Zhang, Zhenghan Yu, Liang Wang, Nan Yang, Eugene Yu, Zheng Li, Yifan Song, Dawei Zhu, Xingxing Zhang, Furu Wei, et al. Learning to draft: Adaptive speculative decoding with reinforcement learning. In International Conference on Learning Representations, volume 2026, pp. 30798–30819, 2026. 3

Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations for speculative sampling. In International Conference on Learning Representations, 2025. URL https://iclr.cc/virtual/2025/poster/29560. 3

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023. 7

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. Distillspec: Improving speculative decoding via knowledge distillation. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=rsY6J3ZaTF. 3

Yuxuan Zhou, Fei Huang, Heng Li, Fengyi Wu, Tianyu Wang, Jianwei Zhang, Junyang Lin, and Zhi-Qi Cheng. Overcoming joint intractability with lossless hierarchical speculative decoding. In International Conference on Learning Representations, 2026. URL https://www.iclr. cc/virtual/2026/poster/10010023. 2, 3, 5, 17

Xiandong Zou, Jianshu Li, Jing Huang, and Pan Zhou. Variational speculative decoding: Rethinking draft training from token likelihood to sequence acceptance. In International Conference on Machine Learning, 2026. URL https://icml.cc/virtual/2026/poster/66012. 3

## A DETAILED THEORY FOR FIXED-TREE VERIFICATION

This appendix provides the technical details for Section 3. Unless stated otherwise, we condition on a decoding context c and a fixed, finite, rooted, prefix-closed draft tree $\boldsymbol { \mathcal { T } } = ( \boldsymbol { \mathcal { V } } , \boldsymbol { \mathcal { E } } , \boldsymbol { r } )$ . Sibling edges carry distinct draft tokens. For each node $v , p _ { \tau }$ is its target next-token distribution, $P ( v )$ is its target prefix probability, $\rho ( v )$ is its local exit probability, and $\bar { \boldsymbol { w } } ( \boldsymbol { v } ) = P ( \boldsymbol { v } ) \rho ( \boldsymbol { v } )$ is its exit probability.

## A.1 RANDOM TREES, AUXILIARY RANDOMNESS, AND ROUND-WISE COMPOSITION

The results in Section 3 are stated conditional on a realized draft tree, so the tree builder may itself be deterministic or stochastic. Let $Q ( \cdot \mid c )$ denote the conditional law of $\tau$ , which is constructed before verification, and let $R ( \cdot \mid c , \mathcal { T } )$ denote the conditional law of auxiliary control randomness U available at time zero. This $U$ may randomize stopping or control decisions but excludes randomness represented by the coupled target continuation. For each realized pair $( \mathcal { T } , U )$ , suppose there exists a measurable conditional coupling $\Gamma \tau , \tau$ of the output block $O ,$ a target continuation $Z ,$ and its length $L = | O |$ | such that

$$
Z \mid ( c , \mathcal { T } , U ) \sim p ( \cdot \mid c ) , \qquad O = Z _ { 1 : L } \quad \Gamma _ { \mathcal { T } , U } \mathrm { - a l m o s t s u r e l y } ,\tag{16}
$$

and $L$ is a stopping time with respect to $\mathcal { H } _ { t } = \sigma ( \mathcal { T } , U , Z _ { 1 } , \ldots , Z _ { t } )$ . Construct the joint coupling by first drawing $\tau \sim Q ( \cdot \mid c )$ , then U ∼ R(· | c, T), and finally $( O , Z , L )$ from $\Gamma \tau , \tau$ . For any measurable set A of target continuations,

$$
\operatorname* { P r } ( Z \in A \mid c ) = \mathbb { E } _ { { \mathcal { T } } , U \mid c } [ \operatorname* { P r } ( Z \in A \mid c , { \mathcal { T } } , U ) ]\tag{17}
$$

$$
= p ( A \mid c ) .\tag{18}
$$

Moreover, $O = Z _ { 1 : L }$ almost surely, and $\{ L \leq t \}$ remains $\mathcal { H } _ { t }$ -measurable. Thus, conditional nonanticipating exactness is preserved after marginalizing over random tree construction and auxiliary randomness. In our setting, neither the drafter-constructed tree nor $U$ has access to future target randomness.

Lemma 2 (Round-wise composition). At round k, let $c _ { k }$ be the current context, $\mathcal { T } _ { k }$ the realized tree, $U _ { k }$ the auxiliary control randomness, $Z ^ { ( k ) }$ the coupled target continuation, and $L _ { k }$ the output-block length, so $O _ { k } = Z _ { 1 : L _ { k } } ^ { ( k ) }$ . Suppose each round satisfies Eq. (16) with these round-indexed quantities and $L _ { k }$ is a stopping time with respect to $\mathcal { H } _ { k , t } : = \sigma ( \mathcal { T } _ { k } , U _ { k } , Z _ { 1 } ^ { ( k ) } , \ldots , Z _ { t } ^ { ( k ) } )$ . Then repeatedly appending the returned blocks preserves the autoregressive distribution induced by $p .$

Proof. For a realized round-k output length $L _ { k } = \ell ,$ , the event $\{ L _ { k } = \ell \}$ is measurable with respect to $\mathcal { H } _ { k , \ell }$ and therefore depends only on $( \mathcal { T } _ { k } , U _ { k } , Z _ { 1 : \ell } ^ { ( k ) } )$ , not on $Z _ { \ell + 1 } ^ { ( k ) } .$ . Because $Z ^ { ( k ) } \mid ( c _ { k } , \mathcal { T } _ { k } , U _ { k } ) \sim$ $p ( \cdot \mathrm { ~ \bf ~ \underline { ~ } { ~ c ~ } ~ } )$ , conditioning on the emitted block $O _ { k } ~ = ~ Z _ { 1 : \ell } ^ { ( k ) }$ and on the stopping event does not change the target law of the unrevealed continuation beyond conditioning on the emitted prefix itself. Restarting round $k + 1$ from the updated context therefore uses exactly the same conditional law as ordinary target decoding. Induction over k proves the claim. □

## A.2 EXIT PARTITION, PROBABILITY CONSERVATION, AND TARGET COVERAGE

ProofofLemma 1. Let $Z = ( Z _ { 1 } , Z _ { 2 } , \ldots ) \sim p ( \cdot \mid c )$ . Because $\tau$ is finite and prefix-closed, there is a unique deepest node $V ^ { \star } ( Z )$ such that $\bar { \pi } ( V ^ { \star } ( Z ) ) \preceq Z . \operatorname { I f } V ^ { \star } ( Z )$ is internal, the next target token cannot equal the draft token of any child; otherwise the continuation would match a deeper node. If $V ^ { \star } ( Z )$ is a leaf, it is automatically the deepest matching node. Hence the events

$$
E _ { v } : = \{ V ^ { \star } ( Z ) = v \} , \qquad v \in \mathcal { V } ,\tag{19}
$$

are disjoint and exhaustive.

The continuation reaches v with probability

$$
\operatorname* { P r } ( \pi ( v ) \preceq Z \mid c , { \mathcal { T } } ) = p ( \pi ( v ) \mid c ) = P ( v ) .\tag{20}
$$

Conditioned on reaching $v ,$ its next token is outside the drafted child-token set $\boldsymbol { \mathcal { A } } ( \boldsymbol { v } )$ with probability

$$
1 - \sum _ { u \in \mathrm { C h } ( v ) } p _ { v } ( x _ { u } ) = \rho ( v ) .\tag{21}
$$

Therefore,

$$
\operatorname* { P r } ( E _ { v } \mid c , T ) = P ( v ) \rho ( v ) = w ( v ) .\tag{22}
$$

Since the events $\{ E _ { v } \} _ { v \in \mathcal { V } }$ form a partition,

$$
\sum _ { v \in \mathcal { V } } w ( v ) = 1 .\tag{23}
$$

□

The exit probabilities also satisfy a local probability-conservation identity:

$$
w ( v ) + \sum _ { u \in \mathrm { C h } ( v ) } P ( u ) = P ( v ) \rho ( v ) + \sum _ { u \in \mathrm { C h } ( v ) } P ( v ) p _ { v } ( x _ { u } )\tag{24}
$$

$$
= P ( v ) .\tag{25}
$$

Thus, target probability reaching v either exits at v or continues through one drafted child. Summing Equation 25 over the subtree rooted at a node u cancels every descendant prefix probability and gives

$$
\sum _ { v \succeq u } w ( v ) = P ( u ) ,\tag{26}
$$

where $v \succeq u$ means that v lies in the subtree rooted at u.

Finally, $\mathrm { d } ( v )$ equals the number of non-root nodes on the path from the root to v. Using Equation 26 and exchanging the order of summation,

$$
\sum _ { v \in \mathcal { V } } w ( v ) \mathrm { d } ( v ) = \sum _ { v \in \mathcal { V } } w ( v ) \sum _ { u \in \mathcal { V } \backslash \{ r \} } \mathbf { 1 } [ v \succeq u ]\tag{27}
$$

$$
= \sum _ { u \in \mathcal { V } \setminus \{ r \} } \sum _ { v \succeq u } w ( v )\tag{28}
$$

$$
= \sum _ { u \in \mathcal { V } \setminus \{ r \} } P ( u ) .\tag{29}
$$

This proves the two equivalent forms of target coverage in Equation 6.

## A.3 PROOF OF THE FIXED-TREE CEILING AND CANONICAL SATURATION

Proof of Theorem 1. Let A be an exact path verifier. Under the coupling defined in Sec. 3, let $Z \sim$ $p ( \cdot \mid c )$ be the target continuation and let the verifier output be $O = \bar { Z } _ { 1 : L }$ . The ceiling argument uses only this target-prefix identity, not the stopping-time property. For every non-root node $u ,$ define

$$
A _ { u } : = \{ u { \mathrm { ~ i s ~ i n c l u d e d ~ i n ~ t h e ~ a c c e p t e d ~ d r a f t ~ p a t h } } \} ,\tag{30}
$$

$$
F _ { u } : = \{ \pi ( u ) \preceq Z \} .\tag{31}
$$

Whenever $A _ { u }$ occurs, the verifier output begins with $\pi ( u )$ and is a prefix of $Z ;$ therefore $A _ { u } \subseteq F _ { u }$ almost surely. Consequently,

$$
\operatorname* { P r } _ { \mathbb { A } } ( A _ { u } \mid c , { \mathcal { T } } ) \leq \operatorname* { P r } ( F _ { u } \mid c , { \mathcal { T } } ) = P ( u ) ,\tag{32}
$$

which proves Equation 8.

An accepted root-to-node path contains exactly one non-root node for every accepted draft token, so

$$
D _ { \mathsf { A } } = \sum _ { u \in \mathcal { V } \setminus \{ r \} } \mathbf { 1 } [ A _ { u } ] .\tag{33}
$$

Taking expectations and applying Equation 32 gives

$$
\mathbb { E } [ D _ { \mathsf { A } } \mid c , { \mathcal { T } } ] = \sum _ { u \neq r } \operatorname* { P r } _ { \mathsf { A } } ( A _ { u } \mid c , { \mathcal { T } } )\tag{34}
$$

$$
\leq \sum _ { u \neq r } P ( u ) = C _ { p } ( \mathcal { T } ) .\tag{35}
$$

Since $L _ { \mathsf { A } } = D _ { \mathsf { A } } + 1$ , we also have $\mathbb { E } [ L _ { \mathsf { A } } \mid c , \mathcal { T } ] \le 1 + C _ { p } ( \mathcal { T } )$ , proving Equation 9.

Now suppose that A is saturated. For each non-root node, let

$$
\delta _ { u } : = P ( u ) - \operatorname* { P r } _ { \mathsf { A } } ( A _ { u } \mid c , \mathcal { T } ) \geq 0 .\tag{36}
$$

Saturation implies $\textstyle \sum _ { u \neq r } \delta _ { u } = 0$ , and hence every $\delta _ { u } = 0$ . Thus

$$
\operatorname* { P r } _ { \mathrm {  { ~ \cal { A } } ~ } } ( A _ { u } \mid c , { \mathcal { T } } ) = P ( u ) \qquad { \mathrm { f o r ~ e v e r y ~ } } u \neq r .\tag{37}
$$

Since $A _ { u } \subseteq F _ { u }$ and both events have the same probability, $A _ { u } = F _ { u }$ almost surely. Because the draft tree is finite, these equalities hold simultaneously for all nodes outside a single null event.

Let $A _ { r } = F _ { r }$ be the full sample space. The accepted path terminates at v exactly when v is accepted and none of its children is accepted. The child acceptance events are disjoint, so

$$
\operatorname* { P r } _ { \mathbf { A } } ( V = v \mid c , \mathcal { T } ) = \operatorname* { P r } _ { \mathbf { A } } ( A _ { v } \mid c , \mathcal { T } ) - \sum _ { u \in \operatorname { C h } ( v ) } \operatorname* { P r } _ { \mathbf { A } } ( A _ { u } \mid c , \mathcal { T } )\tag{38}
$$

$$
= P ( v ) - \sum _ { u \in \mathrm { C h } ( v ) } P ( u )\tag{39}
$$

$$
\begin{array} { r } { { } = { \cal P } ( v ) \rho ( v ) = w ( v ) . } \end{array}\tag{40}
$$

Moreover, the simultaneous identities $A _ { u } = F _ { u }$ imply that V equals the target-induced exit node $V ^ { \star } ( Z )$ almost surely. Since the verifier output satisfies $O = \pi ( { \boldsymbol { V } } ) \parallel Y = Z _ { 1 : L }$ with $L = \mathrm { d } ( V ) + 1$ Y is the target token immediately after $\pi ( \bar { V } )$ . Therefore, for every internal node v with $w ( v ) > 0$

$$
\operatorname* { P r } ( Y = a \mid V = v , c , { \mathcal { T } } ) = { \frac { p _ { v } ( a ) \mathbf { 1 } [ a \not \in { \mathcal { A } } ( v ) ] } { \rho ( v ) } } = p _ { v } ^ { \mathrm { r e s } } ( a ) .\tag{41}
$$

At a leaf, $\boldsymbol { \mathcal { A } } ( \boldsymbol { v } ) = \boldsymbol { \emptyset }$ and $\rho ( v ) = 1$ , so the conditional law is $p _ { v }$ . This proves Equation 10 and the uniqueness of the accepted-path and bonus-token law at saturation. □

## A.4 PROOF THAT THE REPRESENTATIVE VERIFIERS ARE SATURATED

We consider the two target-local procedures defined in Section 2. Both operate on the same fixed draft tree and the same target distributions $\{ p _ { v } \}$ . Sibling draft tokens are distinct. For the targetinduced flow, define $L ^ { \star } : = \bar { \mathrm { d } } ( V ^ { \star } ( Z ) ) + 1$ . This is the first index at which the matched node is a leaf or $Z _ { L ^ { \star } }$ ⋆ is not a drafted child token, so $\{ L ^ { \star } \leq t \}$ is determined by $Z _ { 1 : t }$ and $L ^ { \star }$ is a stopping time. We show that each procedure realizes the joint target-induced exit and bonus-token law; coupling its output to $Z _ { 1 : L ^ { \star } } \stackrel { \cdot } { = } \pi ( V ^ { \star } ( Z ) ) \| Y ^ { \star } ( Z )$ therefore establishes non-anticipating exactness as well as saturation.

## A.4.1 PREDRAW-AND-FOLLOW

At every node v, predraw-and-follow independently draws a token $R _ { v } \sim p _ { v }$ . Starting from the root, it follows child $u \in \operatorname { C h } ( v )$ if $R _ { v } = x _ { u } ;$ ; otherwise it exits at v and returns $R _ { v }$ as the bonus token.

Fix a node v and write its root-to-node path as $r = v _ { 0 } , v _ { 1 } , \ldots , v _ { d } = v$ . The procedure reaches v exactly when $D _ { v _ { i - 1 } } = x _ { v _ { i } }$ for all $i = 1 , \ldots , d .$ Hence

$$
\operatorname* { P r } ( \operatorname { r e a c h } v \mid c , \mathcal { T } ) = \prod _ { i = 1 } ^ { d } p _ { v _ { i - 1 } } ( x _ { v _ { i } } ) = P ( v ) .\tag{42}
$$

Conditioned on reaching v, the independent draw $R _ { v }$ is outside the drafted child-token set with probability

$$
1 - \sum _ { u \in \mathrm { C h } ( v ) } p _ { v } ( x _ { u } ) = \rho ( v ) .\tag{43}
$$

Therefore,

$$
\operatorname* { P r } ( { \mathrm { e x i t ~ a t ~ } } v \mid c , T ) = P ( v ) \rho ( v ) = w ( v ) .\tag{44}
$$

Conditioned on an internal exit, the bonus token follows $p _ { v } ^ { \mathrm { r e s . } }$ at a leaf it follows $p _ { v }$ . Thus the accepted-path and bonus-token law matches the target-induced law, so predraw-and-follow is exact. Its expected accepted draft-token count is $\begin{array} { r } { \sum _ { v } w ( \overline { { v } } ) \mathrm { d } ( v ) = C _ { p } ( \overline { { T } } ) } \end{array}$ , so its expected output-block length is $1 + C _ { p } ( \dot { \mathcal { T } } )$ and it is saturated.

## A.4.2 EXACT SEQUENTIAL RESIDUAL VERIFICATION

Fix an internal node v reached by the verifier. Let its drafted children be tested in an arbitrary order $u _ { 1 } , \ldots , u _ { k }$ , with draft tokens $x _ { 1 } , \ldots , x _ { k }$ . Define

$$
p _ { j } : = p _ { v } ( x _ { j } ) , \qquad M _ { j } : = \sum _ { \ell = 1 } ^ { j } p _ { \ell } , \qquad M _ { 0 } : = 0 .\tag{45}
$$

Let $g ^ { ( 0 ) } : = p _ { v }$ . If child $u _ { j }$ is rejected and the rejection event has positive probability, exact residual renormalization removes token $x _ { j }$

$$
g ^ { ( j ) } ( a ) : = \frac { g ^ { ( j - 1 ) } ( a ) \mathbf { 1 } [ a \neq x _ { j } ] } { 1 - g ^ { ( j - 1 ) } ( x _ { j } ) } .\tag{46}
$$

At step $j ,$ the verifier follows $u _ { j }$ with conditional probability $g ^ { ( j - 1 ) } ( x _ { j } )$ . If all children are rejected, it draws the bonus token from $g ^ { ( k ) }$ . A leaf has no child tests and returns a bonus token from $p _ { v }$

For every $j$ with $M _ { j } < 1$ , induction gives

$$
g ^ { ( j ) } ( a ) = { \frac { p _ { v } ( a ) } { 1 - M _ { j } } } , \qquad a \notin \{ x _ { 1 } , \ldots , x _ { j } \} .\tag{47}
$$

Indeed, assuming the identity at $j - 1$

$$
g ^ { ( j ) } ( a ) = \frac { p _ { v } ( a ) / ( 1 - M _ { j - 1 } ) } { 1 - p _ { j } / ( 1 - M _ { j - 1 } ) } = \frac { p _ { v } ( a ) } { 1 - M _ { j } } .\tag{48}
$$

If $M _ { j } = 1$ , rejecting the first $j$ children has probability zero, so no later residual state affects the output.

Before child $u _ { j }$ is tested,

$$
g ^ { ( j - 1 ) } ( x _ { j } ) = \frac { p _ { j } } { 1 - M _ { j - 1 } } .\tag{49}
$$

The probability of rejecting the first $j - 1$ children telescopes to

$$
\prod _ { i = 1 } ^ { j - 1 } \Big ( 1 - g ^ { ( i - 1 ) } ( x _ { i } ) \Big ) = \prod _ { i = 1 } ^ { j - 1 } \frac { 1 - M _ { i } } { 1 - M _ { i - 1 } } = 1 - M _ { j - 1 } .\tag{50}
$$

Thus, the unconditional probability of following child $u _ { j }$ is

$$
( 1 - M _ { j - 1 } ) \frac { p _ { j } } { 1 - M _ { j - 1 } } = p _ { j } = p _ { v } ( x _ { j } ) .\tag{51}
$$

Similarly, the probability of rejecting every child is

$$
\prod _ { i = 1 } ^ { k } { \frac { 1 - M _ { i } } { 1 - M _ { i - 1 } } } = 1 - M _ { k } = \rho ( v ) ,\tag{52}
$$

and, whenever $\rho ( v ) > 0$ , the final residual target distribution is

$$
g ^ { ( k ) } ( a ) = { \frac { p _ { v } ( a ) \mathbf { 1 } [ a \notin { \mathcal { A } } ( v ) ] } { \rho ( v ) } } = p _ { v } ^ { \mathrm { r e s } } ( a ) .\tag{53}
$$

Hence the local transition kernel is independent of the order of the drafted children: each child u is followed with probability $p _ { v } ( x _ { u } )$ , while the remaining probability $\rho ( v )$ exits with bonus-token law $p _ { v } ^ { \mathrm { r e s } }$ . Multiplying the child-transition probabilities along a root-to-v path gives reach probability $P ( v )$ and exit probability $w ( v )$ . Thus the accepted-path and bonus-token law matches the targetinduced law, so the verifier is exact. Its expected accepted draft-token count is $C _ { p } ( \mathcal { T } )$ , its expected output-block length is $1 + C _ { p } ( \mathcal { T } )$ , and hence it is saturated.

Algorithm 1 Tree Exit Verification (TEV) for speculative verification   
Require: Fixed tree $\overline { { \mathcal { T } = ( \mathcal { V } , \mathcal { E } , \mathcal { r } ) } }$ and target distributions $\{ p _ { v } \} _ { v \in \mathcal { V } }$   
Ensure: Accepted path followed by one bonus token   
1: $P ( r ) \gets 1$   
2: for each depth $d = 0 , 1 , \ldots$ do   
3: for all v at depth d in parallel do   
4: $\begin{array} { r } { \rho ( v ) \gets \dot { 1 } - \sum _ { u \in \dot { \mathrm { C h } } ( v ) } p _ { v } ( x _ { u } ) ; w ( v ) \gets P ( v ) \rho ( v ) } \end{array}$   
5: $P ( u ) \gets P ( v ) p _ { v } ( x _ { u } )$ for all $u \in \operatorname { C h } ( v )$   
6: Draw $V \sim$ Categorical $( \{ w ( v ) \} _ { v \in \mathcal { V } } )$   
7: Draw $Y \sim p _ { V } { \mathrm { i f } } { \bar { V } }$ is a leaf, and otherwise $Y \sim p _ { V } ^ { \mathrm { r e s } }$   
8: return $\pi ( { \bar { V } } ) \parallel Y$

## A.5 WHY THE FIXED-TREE RESULT DOES NOT CONTRADICT HSD

The fixed-tree ceiling applies after a draft tree has already been realized. HSD (Zhou et al., 2026) studies a different probability structure: a draft chain $x _ { 1 : k }$ is sampled from draft distributions $q _ { 1 } , \ldots , q _ { k }$ , and verification retains the target-to-draft ratios

$$
r _ { t } : = \frac { p _ { t } ( x _ { t } ) } { q _ { t } ( x _ { t } ) } .\tag{54}
$$

For a realized chain, position-wise speculative verification accepts the full chain with probability

$$
\prod _ { t = 1 } ^ { k } \operatorname* { m i n } \{ 1 , r _ { t } \} \leq \operatorname* { m i n } \left\{ 1 , \prod _ { t = 1 } ^ { k } r _ { t } \right\} .\tag{55}
$$

When ratios above and below one occur along the same chain, position-wise clipping prevents surplus at one position from compensating for a deficit at another; hierarchical chain verification can recover part of this joint compensation.

Our analysis instead conditions on a realized draft tree. At that point, the draft model affects verification through which draft tokens are present in the tree, while the fixed-tree ceiling follows from target prefix events. Even if a verifier also consults draft scores, no exact path verifier can accept a draft token absent from the realized tree. The two results therefore address different stages of speculative decoding and are complementary rather than contradictory.

## B CORRECTNESS OF TREE EXIT VERIFICATION

## B.1 PROOF OF EXACTNESS AND SATURATION OF TEV

Proof of Corollary 2. Fix a context c and a realized finite, prefix-closed draft tree $\tau .$ . Let $Z \sim$ $p ( \cdot \mid c )$ , let $V ^ { \star } ( \dot { Z } )$ be its target-induced exit node, and let $\hat { Y ^ { \star } } ( Z )$ be the target token immediately following π $( V ^ { \star } ( Z ) )$ .

By Lemma 1, $\operatorname* { P r } ( V ^ { \star } ( Z ) = v \mid c , T ) = w ( v )$ . Conditioned on an internal exit at v with $w ( v ) > 0$ the next target token lies outside $\boldsymbol { \mathcal { A } } ( \boldsymbol { v } )$ and therefore follows $p _ { v } ^ { \mathrm { r e s } }$ ; conditioned on a leaf exit, it follows $p _ { v }$ . These are exactly the two distributions used by Algorithm 1. Hence

$$
( V , Y ) _ { \mathrm { T E V } } \stackrel { d } { = } \big ( V ^ { \star } ( Z ) , Y ^ { \star } ( Z ) \big ) .\tag{56}
$$

We may therefore couple the two pairs to be equal almost surely. Under this coupling, set $L : =$ $\mathrm { d } ( V ) \bar { + } 1 = \mathrm { d } ( V ^ { \star } ( Z ) \bar { ) } + 1$ . This is the first index at which the matched node is a leaf or the next target token leaves the drafted child set. Hence $\{ L \ \leq \ t \}$ is determined by $Z _ { 1 : t }$ and is therefore measurable with respect to $\mathcal { H } _ { t } = \sigma ( \mathcal { T } , U , Z _ { 1 : t } )$ . Moreover,

$$
O = \pi ( V ) \parallel Y = Z _ { 1 : L } \qquad { \mathrm { a l m o s t  s u r e l y } } .\tag{57}
$$

Thus TEV satisfies the non-anticipating exactness definition (without using any additional earlystopping randomness U).

For any non-root node u, TEV accepts u exactly when its exit node lies in the subtree rooted at u. Using Equation 26,

$$
\operatorname* { P r } _ { \mathrm { T E V } } ( u { \mathrm { i s ~ a c c e p t e d } } \mid c , \mathcal { T } ) = \sum _ { v \succeq u } w ( v ) = P ( u ) .\tag{58}
$$

Thus TEV attains every node-wise bound in Theorem 1. Finally,

$$
\mathbb { E } [ D _ { \mathrm { T E V } } \mid c , \mathcal { T } ] = \sum _ { u \in \mathcal { V } \backslash \{ r \} } \operatorname* { P r } _ { \mathrm { T E V } } ( u \mathrm { \ i s \ a c c e p t e d } \mid c , \mathcal { T } )\tag{59}
$$

$$
= \sum _ { u \in \mathcal { V } \setminus \{ r \} } P ( u ) = C _ { p } ( \mathcal { T } ) ,\tag{60}
$$

$$
\mathbb { E } [ L _ { \mathrm { T E V } } \mid c , { \mathcal T } ] = 1 + C _ { p } ( { \mathcal T } ) .\tag{61}
$$

Therefore TEV attains the fixed-tree ceiling and is saturated. If the draft tree is random, the same conclusion follows by conditioning on the realized tree and applying Appendix A.1. □

## C ADDITIONAL DETAILS FOR EXIT-GUIDED TRAINING

## C.1 ONE-STEP TARGET-COVERAGE GAIN

Let v be an expandable node and let $a \notin { \mathcal { A } } ( v )$ be a missing draft token. Write $\boldsymbol { u } = ( v , a )$ for the new child and ${ \dot { \mathbf { \zeta } } } ^ { + } = { \mathcal { T } } \oplus ( v , a )$ for the tree obtained by adding only this node. The target prefix probabilities of all existing nodes are unchanged, and the new node has

$$
\begin{array} { r } { P ( u ) = P ( v ) p _ { v } ( a ) . } \end{array}\tag{62}
$$

Therefore, by the definition of target coverage,

$$
C _ { p } ( \mathcal { T } ^ { + } ) - C _ { p } ( \mathcal { T } ) = P ( u ) = P ( v ) p _ { v } ( a ) ,\tag{63}
$$

which proves Equation 12. Summing over every missing child token gives

$$
\begin{array} { l } { { \displaystyle \sum _ { a \notin A ( v ) } P ( v ) p _ { v } ( a ) = P ( v ) \left( 1 - \sum _ { u \in \mathrm { C h } ( v ) } p _ { v } ( x _ { u } ) \right) } } \\ { { \displaystyle \qquad = P ( v ) \rho ( v ) = w ( v ) , } } \end{array}\tag{64}
$$

(65)

which proves Equation 13. This is a gross one-step gain before enforcing the fixed tree budget; a budget-preserving update may also need to remove another node.

## C.2 EFFICIENT EVALUATION OF THE EXIT-GUIDED LOSS

Let $\mathcal { N }$ denote the multiset of non-root draft-tree nodes in a minibatch, let $c ( v )$ denote the training context associated with node v, and let the detached weight be $\omega _ { v } ~ = ~ w ( v )$ Suppose that the incoming draft token $x _ { v }$ is scored by the DFlash position-wise draft-logit row $z _ { \mathrm { r o w } ( v ) } ~ \in ~ \mathbb { R } ^ { | \mathcal { X } | }$ where row(v) indexes the pair consisting of $v { \mathrm { s } }$ minibatch example and future position $\mathrm { d } ( v )$ . Within one training example, this row is shared by all tree nodes at the same depth and is not conditioned on $\pi ( \mathrm { p a } ( v ) )$ ). Then

$$
\begin{array} { r l } & { \displaystyle \sum _ { v \in \mathcal { N } } \omega _ { v } \left[ - \log q _ { \theta , \mathrm { r o w } ( v ) } ( x _ { v } \mid c ( v ) ) \right] } \\ & { \quad = \displaystyle \sum _ { v \in \mathcal { N } } \omega _ { v } \left[ \log \mathrm { s u m e x p } \left( z _ { \mathrm { r o w } ( v ) } \right) - z _ { \mathrm { r o w } ( v ) , x _ { v } } \right] } \end{array}\tag{66}
$$

$$
= \sum _ { m } \left( \sum _ { v \in \mathcal { N } : \operatorname { r o w } ( v ) = m } \omega _ { v } \right) \log \operatorname { s u m e x p } ( z _ { m } ) - \sum _ { v \in \mathcal { N } } \omega _ { v } z _ { \operatorname { r o w } ( v ) , x _ { v } } .\tag{67}
$$

Thus, each distinct draft-logit row requires only one log-normalizer, and the second term uses a gather over the selected draft-token logits. The minibatch normalization in Equation 14 is applied after this numerator is computed. This decomposition reuses the batched draft-model language head and avoids materializing a separate normalized vocabulary distribution for every selected tree node.

Table 3: Verifier-stage latency per decoding cycle on Qwen3-4B across tree budgets. The measured interval covers verifier decision and cache commit and excludes the shared target-model tree forward pass. For each B, both verifiers use the same drafter, builder, and realized trees.
<table><tr><td rowspan="2">B</td><td rowspan="2">Verifier</td><td colspan="2">MT-Bench</td><td colspan="2">HumanEval</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">Average</td></tr><tr><td>ms↓</td><td>red.</td><td>ms↓</td><td>red.</td><td>ms↓</td><td>red.</td><td>ms↓</td><td>red.</td><td>ms↓</td><td>red.</td></tr><tr><td colspan="13">Temperature = 0.6</td></tr><tr><td>32</td><td>Predraw TEV</td><td>1.371 1.204</td><td>12.2%</td><td>1.435 1.245</td><td>13.2%</td><td>1.428 1.239</td><td>13.2%</td><td>1.393 1.211</td><td>13.0%</td><td>1.407 1.225</td><td>12.9%</td></tr><tr><td>64</td><td>Predraw TEV</td><td>1.392 1.212</td><td>12.9%</td><td>1.487 1.268</td><td>14.7%</td><td>1.443 1.193</td><td>17.3%</td><td>1.488 1.257</td><td>15.5%</td><td>1.453 1.232</td><td>15.2%</td></tr><tr><td>128</td><td>Predraw TEV</td><td>1.620 1.246</td><td>23.1%</td><td>1.719 1.265</td><td>26.4%</td><td>1.681 1.259</td><td>25.1%</td><td>1.616 1.248</td><td>22.8%</td><td>1.659 1.254</td><td>24.3%</td></tr><tr><td colspan="10">Temperature = 1.0</td></tr><tr><td>32</td><td>Predraw TEV</td><td>1.337 1.187</td><td>11.2%</td><td>1.368 1.235</td><td>9.7%</td><td>1.348 1.184</td><td>12.1%</td><td>1.411 1.197 15.2%</td><td></td><td>1.366 1.201</td><td>12.0%</td></tr><tr><td>64</td><td>Predraw TEV</td><td>1.416 1.200</td><td>15.3%</td><td>1.428 1.240</td><td>13.2%</td><td>1.398 1.207</td><td>13.7%</td><td>1.442 1.240</td><td>14.0%</td><td>1.421 1.222</td><td>14.0%</td></tr><tr><td>128</td><td>Predraw TEV</td><td>1.549 1.266</td><td>18.3%</td><td>1.545 1.276</td><td>17.3%</td><td>1.535 1.269</td><td>17.3%</td><td>1.509 1.277</td><td>15.4%</td><td>1.534 1.272</td><td>17.1%</td></tr></table>

## D ADDITIONAL EXPERIMENTAL DETAILS

## D.1 VERIFIER-STAGE EFFICIENCY

For each budget $B \in \{ 3 2 , 6 4 , 1 2 8 \}$ , we hold the drafter, builder, and realized tree fixed across verifiers and time only verifier decision and cache commit, excluding the shared target-model tree forward pass. Corollary 1 guarantees identical fixed-tree expected acceptance for Predraw and TEV at every budget.

TEV reduces verifier-stage latency on every benchmark and budget. Averaged over tasks, the reduction grows from 12.9% to 15.2% to 24.3% as B increases from 32 to 64 to 128 at $T = 0 . 6 ;$ the corresponding reductions at T = 1.0 are 12.0%, 14.0%, and 17.1%. Because the drafter, realized tree, and target forward pass are fixed within each comparison, these measurements isolate execution of the common canonical law. The widening gap with tree size is consistent with Predraw’s per-node sampling cost, whereas TEV uses level-wise probability propagation followed by one exitnode and one bonus-token draw. Even so, the stage-level reduction is larger than the end-to-end improvement in Table 4 because verifier decision and cache commit constitute only a small fraction of total decoding time.

## D.2 CROSSED DRAFTER–VERIFIER EVALUATION

Table 4 crosses the drafter checkpoint with the verification procedure while holding the DDTree builder, B = 64 budget, target model, decoding distribution, and runtime configuration fixed. The comparison reports end-to-end speedup and output-block length at T = 1; verifier-stage latency is isolated separately in Table 3. We omit $T = 0$ because its degenerate target distribution reduces verification to greedy decoding rather than the sampling regime for which TEV is designed.

For a fixed drafter, the empirical τ differences between Predraw and TEV are at most 0.03 tokens, as expected for separate finite-sample runs of two saturated verifiers with the same expected outputblock length. TEV nevertheless improves average SR from 4.95× to 4.97× with the original drafter and from 5.55× to 5.59× with the exit-trained drafter. In contrast, changing the drafter has the dominant effect: under Predraw, exit training increases average τ by 13.7% and SR by 12.1%;

Table 4: Crossed drafter–verifier evaluation on Qwen3-4B at $T = 1$ . Each cell reports end-to-end speedup ratio (SR) and average output-block length including the bonus token (τ). The builder and tree budget $B = 6 4$ are fixed across all four combinations.
<table><tr><td colspan="2"></td><td colspan="2">MT-Bench</td><td colspan="2">HumanEval</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">Average</td></tr><tr><td>Drafter</td><td>Verifier||</td><td>SR↑</td><td>τ ↑</td><td>SR↑</td><td>τ↑</td><td>SR↑</td><td>τ↑</td><td>SR↑</td><td>τ↑Ⅱ</td><td>SR↑</td><td>τ ↑</td></tr><tr><td>Original DFlash</td><td>Predraw TEV</td><td>3.44 3.47</td><td>5.04 5.07</td><td>5.22 5.22</td><td>7.53 7.52</td><td>5.48 5.51</td><td>7.85 7.88</td><td>5.65 5.67</td><td>7.88 7.90</td><td>4.95 4.97</td><td>7.08 7.09</td></tr><tr><td>Exit-trained DFlash</td><td>Predraw TEV</td><td>3.95 3.98</td><td>5.86 5.86</td><td>5.61 5.64</td><td>8.05 8.06</td><td>6.44 6.47</td><td>9.23 9.25</td><td>6.22 6.26</td><td>9.06 9.08</td><td>5.55 5.59</td><td>8.05 8.06</td></tr></table>

Table 5: Qwen3-4B performance across tree budgets on MT-Bench, HumanEval, GSM8K, and MATH-500. ExitTrain is trained at B = 64 and evaluated at every budget without retraining.
<table><tr><td rowspan="2">B Method</td><td rowspan="2"></td><td colspan="2">MT-Bench</td><td colspan="2">HumanEval</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500 _</td><td colspan="2">Average</td></tr><tr><td>∥SR↑ τ↑|</td><td></td><td>SR↑</td><td>τ↑</td><td>SR↑</td><td>τ ↑</td><td>SR↑</td><td>τ↑</td><td>SR↑ τ↑</td><td></td></tr><tr><td colspan="9">Temperature = 0</td><td colspan="2"></td></tr><tr><td rowspan="3">32</td><td>DDTree</td><td>3.69</td><td>5.19</td><td>5.47</td><td>7.74</td><td>5.40</td><td>8.03</td><td>6.50</td><td>8.74</td><td>5.26</td><td>7.43</td></tr><tr><td>Draft-OPD ExitTrain + TEV</td><td>3.91 4.11</td><td>5.32 5.87</td><td>5.60 5.72</td><td>7.82 8.22</td><td>5.87 6.68</td><td>8.92 9.78</td><td>6.84 6.94</td><td>9.70 9.77</td><td>5.56 5.86</td><td>7.94</td></tr><tr><td>DDTree</td><td>3.81</td><td>5.41</td><td>5.71</td><td>8.13</td><td>5.65</td><td></td><td></td><td></td><td></td><td>8.41</td></tr><tr><td rowspan="3">64</td><td>Draft-OPD</td><td>4.16</td><td>5.73</td><td>5.96</td><td>8.35</td><td>6.11</td><td>8.48 9.30</td><td>6.83 7.02</td><td>9.16 10.14</td><td>5.50 5.81</td><td>7.80 8.38</td></tr><tr><td>ExitTrain + TEV</td><td>4.28</td><td>6.17</td><td>6.09</td><td>8.74</td><td>6.91</td><td>10.22</td><td>7.23</td><td>10.26</td><td>6.13</td><td>8.85</td></tr><tr><td>DDTree</td><td>3.95</td><td>5.77</td><td>5.93</td><td>8.57</td><td>5.86</td><td>8.82</td><td>6.89</td><td>9.60</td><td></td><td>8.19</td></tr><tr><td rowspan="3">128</td><td>Draft-OPD</td><td>4.25</td><td>6.10</td><td>6.10</td><td>8.64</td><td>6.28</td><td>9.63</td><td>7.14</td><td>10.40</td><td>5.66 5.94</td><td>8.69</td></tr><tr><td>ExitTrain + TEV</td><td>4.40</td><td>6.50</td><td>6.28</td><td>9.10</td><td>7.15</td><td>10.60</td><td>7.43</td><td>10.72</td><td>6.31</td><td></td></tr><tr><td></td><td></td><td></td><td>Temperature = 1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>9.23</td></tr><tr><td colspan="14"></td></tr><tr><td rowspan="3">32</td><td>DDTree</td><td>3.31</td><td>4.82</td><td>5.00</td><td>7.15</td><td>5.24</td><td>7.50</td><td>5.39</td><td>7.60</td><td>4.73</td><td>6.77</td></tr><tr><td>Draft-OPD</td><td>3.56</td><td>4.97</td><td>5.34</td><td>7.49</td><td>5.58</td><td>8.12</td><td>5.86</td><td>8.14</td><td>5.09</td><td>7.18</td></tr><tr><td>ExitTrain + TEV</td><td>3.79</td><td>5.50</td><td>5.38</td><td>7.75</td><td>6.09</td><td>8.78</td><td>6.02</td><td>8.52</td><td>5.32</td><td>7.64</td></tr><tr><td rowspan="3">64</td><td>DDTree</td><td>3.44</td><td>5.04</td><td>5.22</td><td>7.53</td><td>5.48</td><td>7.85</td><td>5.65</td><td>7.88</td><td>4.95</td><td>7.08</td></tr><tr><td>Draft-OPD</td><td>3.74</td><td>5.26</td><td>5.49</td><td>7.80</td><td>5.88</td><td>8.58</td><td>6.08</td><td>8.50</td><td>5.30</td><td>7.54</td></tr><tr><td>ExitTrain + TEV</td><td>3.98</td><td>5.86</td><td>5.64</td><td>8.06</td><td>6.47</td><td>9.25</td><td>6.26</td><td>9.08</td><td>5.59</td><td>8.06</td></tr><tr><td rowspan="3">128</td><td>DDTree</td><td>3.50</td><td>5.32</td><td>5.42</td><td>7.93</td><td>5.74</td><td>8.35</td><td>5.75</td><td>8.38</td><td>5.10</td><td>7.50</td></tr><tr><td>Draft-OPD</td><td>3.80</td><td>5.54</td><td>5.81</td><td>8.31</td><td>5.99</td><td>8.89</td><td>6.17</td><td>8.78</td><td>5.44</td><td>7.88</td></tr><tr><td>ExitTrain + TEV</td><td>4.06</td><td>6.08</td><td>5.77</td><td>8.45</td><td>6.60</td><td>9.66</td><td>6.36</td><td>9.30</td><td>5.70</td><td>8.37</td></tr></table>

under TEV, the gains are 13.7% and 12.5%. This separates improved tree allocation from the smaller verifier-execution gain.

## D.3 TREE-BUDGET ROBUSTNESS

Table 5 varies only the inference-time DDTree budget. All drafter checkpoints are fixed; in particular, ExitTrain is trained with $B = 6 4$ and evaluated without retraining at $B \in \{ 3 2 , 6 4 , 1 2 8 \}$ . We report each benchmark separately together with the four-benchmark average.

Increasing the budget raises absolute output-block length for every method. ExitTrain + TEV improves average τ from 8.41 to 8.85 to 9.23 at $T = 0$ and from 7.64 to 8.06 to 8.37 at $T = 1$ as B grows from 32 to 64 to 128; average SR increases from 5.86× to 6.13× to 6.31× and from 5.32× to 5.59× to $5 . 7 0 \times .$ , respectively. More importantly, its advantage persists at every budget. Relative to DDTree, the gains span 11.6%–13.8% in τ and 11.4%–12.9% in SR; relative to Draft-OPD, they span 5.6%–6.9% and 4.5%–6.2%. Thus, the drafter trained with B = 64 transfers to both smaller and larger inference trees without budget-specific retraining.

Unless otherwise stated, the main and crossed experiments use a tree budget of $B = 6 4$ non-root nodes, BF16 model weights, SDPA for target-model attention, FlashAttention-2 for the drafter, batch size one, and at most 2,048 generated tokens. Every evaluation configuration uses one NVIDIA H20-96G GPU. We evaluate 128 GSM8K examples, all 164 HumanEval problems, 128 MATH-500 examples, and 80 MT-Bench questions; GSM8K, HumanEval, and MATH-500 are repeated three times and MT-Bench twice. Qwen3 models use their chat template with thinking disabled. Speedup ratios are computed against autoregressive decoding measured under the same runtime configuration as the corresponding speculative run.

Training configuration. All runs initialize from the official DFlash checkpoints released at https://huggingface.co/collections/z-lab/dflash. We train online with Spec-Forge using prompts from the open-perfectblend mixture (Xu et al., 2024): the frozen target model generates rollout responses during training, and the drafter is updated on these targetgenerated trajectories. Training uses one 8×H20-96G server, with one GPU assigned to targetmodel rollout and the remaining seven GPUs assigned to draft-model training. All models use learning rate $1 0 ^ { - 4 }$ and batch size 4. The measured wall-clock training times on this server are approximately 35 hours for Qwen3-4B, 55 hours for Qwen3-8B, 100 hours for Gemma-4-12B-IT, and 80 hours for Qwen3-Coder-30B-A3B. Every training-objective comparison in Sec. 6.2 uses the same initialization, online data pipeline, and compute budget.