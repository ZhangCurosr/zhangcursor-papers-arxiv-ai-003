# What the Primary Lund Plane Omits: A Model-Stitching Diagnosis of a Jet-Tagger Representation Gap

Saurabh Rai<sup>∗</sup> and Sanmay Ganguly<sup>†</sup>

Department of Physics, Indian Institute of Technology Kanpur, Uttar Pradesh 208016, India

A jet tagger built on the Lund plane and one built on a Particle Transformer (ParT) can difer in performance for two distinct reasons. Either the representation itself is weaker, or an adequate representation is read by a weaker downstream computation. Standalone performance cannot separate these. We use model stitching to separate these directly. It links the frozen early layers of one trained network to the frozen late layers of another through a single trained linear map. As a positive control, we use a deliberately incomplete Lund representation, the primary declustering branch, whose deficit is already known independently. Stitched into ParT’s own frozen downstream, the primarybranch representation reaches a background rejection of 128–136 at fixed signal eficiency. That is at least as much as its own downstream recovers from the same bottleneck (120–134). Yet it remains far below ParT’s own 291. The deficit is therefore representational, not computational. A hypergraph reader attending over triplets of splittings was tested as an alternative explanation. It extracts no more from the primary branch than the pairwise-attention network does, 116 against 112. This rules out higher-order correlations as the cause. The gap is invisible in AUC, which stays within 1% of its target value at every depth. It appears only in background rejection at low signal eficiency. It localizes to 0.6% of background jets. These jets sit at and above the top mass, and have more constituents but fewer primary-plane splittings than typical background. Rebuilding the Lund representation on the full Cambridge/Aachen declustering tree repairs the deficit, as predicted. Standalone rejection rises from 136 to 248, closing 72% of the gap to ParT. Masking nodes at inference shows the repair is not localized to any small subset. The full-tree network depends almost entirely on the primary branch: deleting twelve secondary nodes costs a fifth of the rejection, while deleting twelve primary nodes costs almost all of it. Yet the secondary splittings still matter. They make the representation learnable in the first place, rather than supplying information the finished network reads. Within the primary branch, importance concentrates at the widest-angle splitting. It is worth nearly twice a random one, tied to a boosted top’s opening angles. The same pattern holds for quark/gluon tagging, but far more weakly. It shows that the Casimir-scaling radiation of the primary branch alone retains 77.6% of ParT’s rejection, against 46.9% for top tagging, and the full tree closes only 33% of the gap, against 72%.

## I. INTRODUCTION

A jet tagger built on a physics-motivated representation and one built on a learned representation of raw kinematics can difer in final performance for reasons a performance comparison alone cannot distinguish. The gap could mean the physics-motivated representation carries less information than the learned one. It could also mean an adequate representation is being read by a weaker downstream computation. These are diferent problems with diferent fixes, and standalone accuracy cannot tell them apart.

The primary Lund plane [1] re-expresses a jet as a sequence of declustering variables tied to the collinear and soft-collinear structure of perturbative QCD, following only the harder branch at each splitting. LundNet [2] instead reads the full Cambridge/Aachen declustering tree with a graph network, and outperforms taggers built on the primary branch alone. The Particle Transformer (ParT) [3] keeps every constituent’s four-momentum and lets self-attention [4] build whatever features the classifi cation loss rewards.

Comparing final performance is also easy to get wrong by choice of metric. In the benchmark that introduced LundNet, its five-variable model and ParticleNet [5] difer in top-tagging AUC by only 0.004 (0.987 against 0.983), while their background rejection at ε = 0.5 difers by a factor of 2.5 (5000 against 2000) [2]. AUC, the field’s most-quoted metric, can be nearly blind to a diference that background rejection at a realistic operating point shows clearly. Our own measurements reproduce this insensitivity in a sharper form: the representational deficit this paper is about moves AUC by less than one percent while halving background rejection.

Representational-similarity measures such as centered kernel alignment (CKA) [6] and singular vector canon ical correlation analysis (SVCCA) [7] compare geometry directly, but geometric similarity is not functional interchangeability: two representations can occupy wellaligned subspaces and still be unusable by a specific downstream computation, or vice versa. Model stitching [8] asks the functional question directly, by connecting the frozen early layers of one trained network to the frozen late layers of another through a single trained layer, and reading of how much performance survives the connection. The same freeze-and-reconnect construction already has a diferent use in jet tagging: freezing early layers and fine-tuning only the rest transfers a LundNetor ParticleNet-style tagger to a new signal on an order of magnitude less data [9]. Two concurrent studies ask what a trained jet tagger actually uses by other means: causal ablation and path patching inside ParT itself find a sparse attention-head circuit [10]; perturbation- and gradient-based attribution finds which Lund-plane splittings a LundNet-style tagger relies on [11]. Neither separates a representation’s content from its reader, which is the question this paper asks instead.

Two obstacles stand in the way of applying this to ParT and a Lund-plane network. A jet’s N particles and its Lund-plane splittings are sequences of unrelated length, so no per-token correspondence exists. A graph network’s local message passing and a transformer’s global self-attention difer enough that a stitching result could reflect either the input representation or the architecture doing the reading, with no way to separate the two afterward. We remove both obstacles by construction. The stitch point is the class token each network produces, through class attention, which is a single whole-jet vector regardless of sequence length. The Lund-plane network, which we call Lund-ParT, is the same architecture as ParT in every respect except its input: the same selfattention blocks, the same class-attention readout, difering only in the per-node feature vector and the pairwise bias function. Any result is then attributable to representation content, not to a choice of architecture.

We validate this diagnostic on a case where the answer is already known. The primary Lund plane keeps only the harder branch at each splitting: a jet with 49.0 real constituents on average yields only 12.6 primaryplane splittings, discarding most of the jet’s radiation. LundNet exists precisely because this branch is incomplete. We use Lund-ParT built on the primary branch as a positive control, not because we expect it to compete with the full declustering tree, but because its deficit is known independently of anything model stitching tells us. If stitching correctly identifies that deficit as representational rather than computational, localizes it to the jets it should afect, and confirms the diagnosis when the omission is repaired, that is direct evidence the method works on cases where the ground truth is not known.

The paper has six parts. Section II sets out the architectures, the stitching protocol, and the two normalizations it supports. Section III shows the gap between ParT and primary-branch Lund-ParT is representational rather than downstream, using standalone performance, retained AUC, background rejection under both normalizations, and geometric alignment. Section IV localizes the deficit to a specific, physically identifiable jet population. Section V rebuilds Lund-ParT on the full declustering tree and confirms the diagnosis by repairing it. Section VI then asks which nodes of that tree the repaired network actually uses, and finds that the repair is not carried by any small subset of them. Section VII asks the same kind of question of the primary branch alone, and finds its information concentrated at the wide-angle end of the chain. Appendix A repeats the entire pipeline on a second task, quark/gluon tagging, to test whether the diagnosis and its conclusions are specific to top substructure.

## II. METHOD

## A. Dataset

We use the Kasieczka et al. top-tagging reference dataset [12]: 1,211,000/403,000/404,000 train/validation/test jets of hadronically decaying, boosted top quarks against light-quark and gluon jets, $p _ { T } \in [ 5 5 0 , 6 5 0 ]$ GeV, up to 200 zero-padded constituent four-momenta per jet. All results here use a fixed, reproducibly seeded 50% random subsample of each split (605,500/201,500/202,000 jets), of which 100,971 test jets are background. Standalone ParT reaches a background rejection of 291 at $\varepsilon _ { S } = 0 . 5$ at this scale, still below the ∼ 2000–5000 range reported at full dataset scale for comparable architectures [2]; the residual gap is a matter of training data volume, not of the comparisons made here, since every claim in this paper is a ratio or a diference between two models trained on identical data. Jets with fewer than two real constituents have no Lund-plane splittings under either representation and are dropped identically from both before training, afecting two jets in the test split.

## B. Architectures

Both networks instantiate the same ParticleTransformer class: an input embedding, a pairwise-bias network computed once and reused as an additive attention bias in every block, $L \ = \ 6$ pre-norm self-attention blocks (embedding dimension 128, 8 heads), 2 class-attention blocks [13] in which a learned class token cross-attends to the block-6 token sequence, and a two-layer classification head. ParT’s per-particle input is the seven kinematic scalars of Ref. [3]; its pairwise bias is the four relativistic invariants $( \ln k _ { T } , \ln z , \ln \Delta R , \ln m ^ { 2 } )$ of every particle pair. This totals 1,658,814 trainable parameters.

Lund-ParT is the identical class, instantiated on the primary Lund plane’s five declustering coordinates as both the per-node input and the pairwise-bias basis. The primary Lund plane is built by reclustering a jet’s own constituents with the Cambridge/Aachen algorithm [14, 15] at an efective radius of 1000, using FastJet [16], then walking only the harder branch at every splitting. For splitting i, parent → harder subjet a + softer subjet

b,

$$
\begin{array} { r l } & { \quad z ^ { ( i ) } = \cfrac { p _ { T } ^ { b } } { p _ { T } ^ { a } + p _ { T } ^ { b } } , } \\ & { \quad \Delta R ^ { ( i ) } = \sqrt { ( y ^ { a } - y ^ { b } ) ^ { 2 } + ( \phi ^ { a } - \phi ^ { b } ) ^ { 2 } } , } \\ & { \quad k _ { T } ^ { ( i ) } = p _ { T } ^ { b } \Delta R ^ { ( i ) } , } \\ & { \quad m ^ { ( i ) } = | ( a + b ) | , } \\ & { \quad \psi ^ { ( i ) } = \arctan \frac { y ^ { b } - y ^ { a } } { \phi ^ { b } - \phi ^ { a } } , } \end{array}\tag{1}
$$

giving the node feature $x _ { i } = ( \ln z , \ln \Delta R , \psi$ , ln m, ln k ). This is the primary branch only, following Ref. [1]’s original construction. LundNet’s own graph [2] uses the full declustering tree instead, the diference Sec. V tests directly. Lund-ParT’s pairwise bias is the raw fivedimensional coordinate diference $x _ { i } - x _ { j }$ . Lund-ParT has 1,658,616 trainable parameters, matching ParT’s count to within 0.01%. Background jets in the test split carry 43.3 real particles on average and yield 12.6 primary-Lund-plane splittings, since following only the harder branch discards most constituents from the resulting sequence.

Both networks are trained independently for five random seeds: AdamW [17], cross-entropy loss, up to 30 epochs with early stopping on validation AUC (patience $5 )$ , batch size 256, float16 mixed precision with dynamic loss scaling. Standalone test performance is in Table I.

## C. Stitching

For network $N \in \{ A , B \}$ and depth $d ,$ define

$$
\begin{array} { l } { { f _ { N } ^ { ( d ) } = \mathrm { c a } _ { N } \circ \mathrm { b l o c k } _ { N } ^ { ( d ) } \circ \cdots \circ \mathrm { b l o c k } _ { N } ^ { ( 1 ) } \circ \mathrm { e m b e d } _ { N } , } } \\ { { g _ { N } ^ { ( d ) } = \mathrm { f c } _ { N } \circ \mathrm { c a } _ { N } \circ \mathrm { b l o c k } _ { N } ^ { ( 6 ) } \circ \cdots \circ \mathrm { b l o c k } _ { N } ^ { ( d + 1 ) } , } } \end{array}\tag{2}
$$

so that $f _ { N } ^ { ( d ) } : \mathcal { X }  \mathbb { R } ^ { 1 2 8 }$ carries a jet to $N \mathrm { { s } }$ class token after d self-attention blocks, and $g _ { N } ^ { ( d ) } : \mathbb { R } ^ { 1 2 8 }  \mathbb { R }$ carries an injected 128-dimensional vector through $N \mathrm { { ^ { * } s } }$ remaining blocks, class-attention readout, and head to a logit. For source network $A ,$ target network $B _ { ; }$ and depth $d ,$ the stitched classifier is then (Fig. 1)

$$
\begin{array} { r } { \hat { y } _ { A \to B } ^ { ( d ) } ( x ) \ = \ g _ { B } ^ { ( d ) } \big ( W f _ { A } ^ { ( d ) } ( x ) \big ) , \qquad W \in \mathbb { R } ^ { 1 2 8 \times 1 2 8 } , } \end{array}\tag{3}
$$

with $f _ { A } ^ { ( d ) }$ and $g _ { B } ^ { ( d ) }$ fixed at their standalone-trained values and only W optimized.

A’s embedding, first d blocks, and class-attention readout are frozen at standalone-trained values and produce A’s own class token at depth d. W is a single trained Linear(128, 128) layer (16,512 parameters). B’s remaining blocks, class-attention readout, and classification head stay frozen at their own standalone trained values. Only W is optimized: AdamW, up to 15 epochs with early stopping (patience 4), on a fixed random training subsample with 20,000 jets. A map with 16,512 parameters does not need the full training set to converge. All reported test metrics use the full 50% test split.

A’s and $B ^ { \prime } \mathrm { s }$ token sequences have unrelated lengths, so $B ^ { \prime } { \mathrm { s } }$ remaining blocks run on the single injected vector $W f _ { A } ^ { ( d ) } ( x )$ as an unit length sequence. This is exact, not approximate. Write a block’s self-attention sublayer as $\begin{array} { r } { \tilde { \mathrm { A t t n } } ( q _ { i } , \{ k _ { j } \} , \{ v _ { j } \} ) = \sum _ { j } } \end{array}$ softmax $_ j \left( q _ { i } \cdot k _ { j } / \sqrt { d _ { k } } \right) v _ { j }$ . If the input sequence consists of n identical tokens, $x _ { j } = v$ for every $j = 1 , \dotsc , n$ , then every query and key is likewise identical, $q _ { i } \equiv q$ and $k _ { j } \equiv k ,$ , and the softmax weights become uniform:

$$
{ \mathrm { A t t n } } { \big ( } q , \{ k \} _ { j = 1 } ^ { n } , \{ v \} _ { j = 1 } ^ { n } { \big ) } \ = \ \sum _ { j = 1 } ^ { n } { \frac { e ^ { q \cdot k / { \sqrt { d _ { k } } } } } { \sum _ { j ^ { \prime } = 1 } ^ { n } e ^ { q \cdot k / { \sqrt { d _ { k } } } } } } v \ = \ v ,\tag{4}
$$

independent of $q , k ,$ and $n .$ . The point wise feedforward sublayer acts identically on every position, so a block maps n identical input tokens to n identical output tokens; by induction over blocks $d + 1 , \dotsc , 6 , g _ { B } ^ { ( d ) }$ applied to n copies of $W f _ { A } ^ { ( d ) } ( x )$ returns the same value for every $n \geq 1$ , including $n = 1$ . Broadcasting the injected vector to any number of positions therefore computes the identical function as feeding it once; we use $n = 1$ . Whether an unit length input costs anything relative to the token sequences $B \mathrm { { ^ { * } s } }$ blocks were trained on is a separate question, answered empirically by the same-architecture self-stitching controls described next: self-stitched re tention ranges from 29% to 99% of standalone performance across depths (Sec. III). We do not argue this cost away. Every comparison in this paper is between a crossarchitecture condition and a same-architecture self-stitch at the same depth and the same source, so the injection cost is common to both sides and cancels.

We report $d \in \{ 1 , \ldots , 5 \}$ and exclude $d = L = 6$ entirely. At $d = 6$ the target contributes no self attention blocks at all: the construction reduces to feeding A’s complete, already-pooled class token through W directly into $B ^ { \prime } \mathrm { s }$ frozen class attention readout and classification head. That measures whether a linear map can re-encode a finished jet summary for a foreign classifier head, which is a diferent question from the one the depth sweep asks, and it introduces a double class attention pooling with no counterpart at any other depth. It also behaves pathologically in a way that makes the distinction concrete: at $d = 6$ the two conditions whose target is Lund-ParT lose roughly half their background rejection while their AUC moves by less than 0.005, with a per seed scatter spanning a factor of three, whereas the two conditions whose target is ParT are unafected. Since early stopping selects on validation AUC, and AUC is flat from the first epoch at this depth, the selection criterion cannot see the degradation it is selecting among. We therefore treat $d = 6$ as outside the construction rather than as a data point to be interpreted, and it appears in no figure or table below.

Four conditions are run at every depth and every seed: $A = \mathrm { P a r T } , B = \mathrm { L u n d - P a r T } .$ the reverse; and the two self-(c) Four conditions, every depth $d \in \{ 1 , \ldots , 5 \}$ , every seed

TABLE I. Standalone test performance, mean ± standard deviation over 5 seeds, 50% data scale. The full-tree column anticipates Sec. V.
<table><tr><td></td><td>ParT</td><td>Lund-ParT (primary)</td><td>Lund-ParT (full tree)</td></tr><tr><td>AUC</td><td>0.9838 ± 0.0003</td><td>0.9759 ± 0.0004</td><td> $\overline { { 0 . 9 8 2 2 \pm 0 . 0 0 0 1 } }$ </td></tr><tr><td>rej. at  $\varepsilon _ { S } = 0 . 5$ </td><td> $2 9 1 \pm 1 3$ </td><td> $1 3 6 \pm 4$ </td><td> $2 4 8 \pm 5$ </td></tr><tr><td>rej. at  $\varepsilon _ { S } = 0 . 3$ </td><td> $9 2 6 \pm 4 6$ </td><td> $4 9 5 \pm 1 0$ </td><td> $8 9 5 \pm 5 8$ </td></tr></table>

(a) Both networks instantiate the same backbone

![](images/720d2e52089f3a323adfd1f7b09d2c9c358803d4b82f648b3f59a756de0672e5.jpg)  
L = 6 pre-norm self-attention blocks class-attn

(b) The stitched model at depth d (shown for d = 3)  
![](images/d9f8033068cdfdbe738f9f873f0e295ecf8d840c759a3cbc9ca278fd86c5f6a3.jpg)

![](images/a7c0a9768348323b211d83c744988b765088c9f749dd8f1c073c82c594bcbd7f.jpg)  
FIG. 1. Schematic of the stitching construction. (a) Both ParT and Lund-ParT instantiate the same six-block backbone with a two-block class-attention readout; the dashed line marks where a stitch cuts at depth d. (b) At depth d (shown for $d = 3 )$ source network A contributes its first d blocks and class attention readout, frozen at standalone trained values, producing a single class token $z _ { A } ^ { ( d ) } \in \mathbb { R } ^ { 1 2 8 }$ ; a trained $1 2 8 \times 1 2 8$ linear map W carries it, broadcast as an unit length sequence, into target network B’s remaining blocks, readout, and head, likewise frozen. (c) The four conditions run at every depth and seed: both cross-architecture directions and the two same-architecture self stitching controls; $d = 6$ is excluded for the reason given below.

stitching controls $A = B = \mathrm { P a r T }$ $A = B = { \mathrm { L u n d - P a r T } }$ Every condition uses the same five seeds as standalone training, seed-matched: seed s’s stitching conditions use exactly seed s’s two standalone backbones.

## D. Metrics and normalization

We report AUC and background rejection $1 / \varepsilon _ { B }$ at fixed signal eficiency $\varepsilon _ { S } \in \{ 0 . 5 , 0 . 3 \}$ . For a stitched condition with source A and target B, we define two retained

fractions for each metric X,

$$
R _ { X } ^ { \mathrm { t g t } } ( d ) = \frac { X _ { \mathrm { s t i t c h e d } } ( d ) } { X _ { B , \mathrm { s t a n d a l o n e } } } , \qquad R _ { X } ^ { \mathrm { s r c } } ( d ) = \frac { X _ { \mathrm { s t i t c h e d } } ( d ) } { X _ { A , \mathrm { s t a n d a l o n e } } } .\tag{5}
$$

$R _ { X } ^ { \mathrm { t g t } }$ asks whether B’s downstream computation is the limiting factor, holding the reference B fixed. $R _ { X } ^ { \mathrm { s r c } }$ asks how much of A’s own information survived being read by a foreign downstream, relative to how much A’s own downstream recovers from the same bottleneck. These measure diferent things and are not interchangeable: a comparison across two conditions must hold one of A or B fixed and use the matching normalization, without ever mixing the standalone values of the respective targets. Section III uses $R _ { X } ^ { \mathrm { s r c } }$ throughout, since every comparison there holds the target fixed at ParT or at Lund-ParT and asks how much of a given source’s information survives. We report absolute rejection alongside every fraction so the denominator is always visible.

TABLE II. Retained AUC, target-normalized, primarybranch run, mean over 5 seeds.
<table><tr><td></td><td colspan="5">d</td></tr><tr><td>direction</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>lund→part</td><td>0.991</td><td>0.991</td><td>0.992</td><td>0.992</td><td>0.992</td></tr><tr><td>part→lund</td><td>0.977</td><td>0.996</td><td>1.000</td><td>1.004</td><td>1.007</td></tr><tr><td>part-self</td><td>0.983</td><td>0.989</td><td>0.994</td><td>0.996</td><td>0.999</td></tr><tr><td>lund-self</td><td>0.998</td><td>0.999</td><td>0.999</td><td>1.000</td><td>1.000</td></tr></table>

## III. THE GAP IS REPRESENTATIONAL, NOT DOWNSTREAM

Table I shows the naive gap: Lund-ParT’s standalone rejection, 136, is less than half of ParT’s, 291. This gap is consistent with either a weaker representation or a weaker downstream computation reading an otherwise adequate one. The rest of this section separates the two.

## A. AUC does not distinguish the representations

Table II shows retained AUC, target-normalized, for all four stitching conditions. It sits in a narrow band, 0.977– 1.007, at every depth, for both cross-architecture directions and both self-stitching controls alike. The lund-topart direction is flat to three decimal places across the entire sweep, moving from 0.991 at $d = 1$ to 0.992 at $d = 5$ . Cross-architecture stitching is very nearly indistinguishable from same-architecture self-stitching at this metric: AUC alone would suggest the two representations are interchangeable, echoing the AUC-insensitivity noted in Sec. I for full taggers. Background rejection at fixed, low signal eficiency shows this picture is incomplete.

## B. Background rejection under both normalizations

Figure 2 shows absolute rejection for all four conditions at every depth, with each network’s own standalone value marked. Lund-to-part – Lund-ParT’s representation read through ParT’s frozen downstream – rises gently from 128 at $d = 1$ to 136 at $d = 5$ . Lund-self – Lund-$\mathrm { P a r T } \mathrm { s }$ own downstream reading the same representation – sits at or below it at every depth, from 120 at $d = 1$ to 134 at $d = 5$ (Table III, both columns against the same denominator, Lund-ParT’s own standalone value of 136). ParT’s downstream extracts at least as much from Lund-ParT’s representation as Lund-ParT’s own downstream does, and slightly more at every depth. There is no latent information in the primary-branch representation that a stronger downstream would have recovered: the shortfall relative to ParT’s own 291 is not explained by Lund- $. \mathrm { P a r T } \mathrm { s }$ downstream being the bottleneck.

TABLE III. Retained rej. at $\varepsilon _ { S } = 0 . 5 ,$ , primary-branch run, source-normalized $( R ^ { \mathrm { s r c } } )$ , mean over 5 seeds. lund→part and lund-self both divide by Lund-ParT’s own standalone value (136); part→lund and part-self both divide by ParT’s own standalone value (291). Self-stitching values coincide with $R ^ { \mathrm { t g t } }$
<table><tr><td> $d$ </td><td>lund→part (/136)</td><td>lund-self (/136)</td><td>part→lund (/291)</td><td>part-self (/291)</td></tr><tr><td>1</td><td>0.942</td><td>0.878</td><td>0.202</td><td>0.290</td></tr><tr><td>2</td><td>0.969</td><td>0.910</td><td>0.384</td><td>0.407</td></tr><tr><td>3</td><td>0.977</td><td>0.947</td><td>0.504</td><td>0.555</td></tr><tr><td>4</td><td>0.990</td><td>0.977</td><td>0.692</td><td>0.709</td></tr><tr><td>5</td><td>0.998</td><td>0.986</td><td>0.913</td><td>0.926</td></tr></table>

The reverse direction, part-to-lund, shows the same ordering with the roles exchanged: normalized to ParT’s own standalone value (291), it retains 20–91% across the sweep, below what ParT’s own downstream recovers from the same bottleneck at every depth (29–93%, part-self). Lund-ParT’s downstream is, on this evidence, the less capable consumer of the two, independent of which representation it is given. We tested this directly rather than reading it of ten dependent per-depth comparisons: for each of the five seeds, we averaged the self-minus-cross diference in raw rejection units over $d = 1 - 5$ separately for each direction, then averaged the two directions into one seed-level score. A one-sample t-test on the five resulting scores gives a mean advantage of $+ 8 \pm 7$ rejection units for $\mathrm { P a r T } { \mathrm { : } }$ downstream over Lund-ParT’s $( t = 2 . 5 6$ $p = 0 . 0 6 2 )$ : the same direction as every one of the ten individual comparisons, but not significant at the conventional threshold with five seeds. We report this as suggestive, not established.

This consumer-side pattern does not explain the result this section is built on. Lund-ParT’s primary-branch front end is weaker than ParT’s in absolute terms (136 against 291) independent of which downstream reads it, and the comparison that isolates this – ParT’s own downstream reading Lund-ParT’s representation, against Lund-ParT’s own downstream reading the same representation – holds the representation fixed and varies only the reader. ParT’s downstream does at least as well with it at every depth. The deficit is in what the primary branch supplies, not in who reads it.

## C. Geometric alignment between the two representations

Linear CKA [6] and SVCCA [7] between ParT’s and Lund-ParT’s class tokens, computed independently of any stitching or classification objective on a fixed 5000- jet probe sample, both rise with depth: CKA from

![](images/5a0dd64111b348d219fca4f68eece0b01f7dd6213404fc9ae17d3f5551c79b60.jpg)

![](images/73e805780cb3601af0f7ec1005a21044e2eef154cde1ec4549d2f0ca74adc58e.jpg)  
FIG. 2. Absolute background rejection at $\varepsilon _ { S } = 0 . 5$ and 0.3 for all four stitching conditions, primary-branch Lund-ParT, versus depth. Horizontal lines mark each network’s own standalone value. The lund-to-part curve (solid red) sits at or above Lund ParT’s own self-stitching curve (dashed red) at every depth.

![](images/c3c43903fea304d3f59cc5fd091f9dffb85402038865d64bb76375d3303436ec.jpg)  
FIG. 3. Linear CKA and SVCCA mean canonical correlation between ParT’s and Lund-ParT’s class tokens versus depth.

$0 . 7 5 9 \pm 0 . 0 7 7$ at $d \ = \ 1$ to $0 . 9 3 4 \pm 0 . 0 0 5$ at $\ d \ = \ 5 ;$ SVCCA’s mean canonical correlation from $0 . 8 0 4 \pm 0 . 1 2 6$ to $0 . 8 7 2 { \pm } 0 . 1 0 3 ~ ( \mathrm { F i g . ~ 3 } )$ . The depth index here labels the layer at which a class token is read from a standalone network, so unlike the stitching sweep it remains well defined at $d = 6 .$ , where CKA reaches $0 . 9 3 5 \pm 0 . 0 0 4$ we quote the $d \leq 5$ range for comparability with the rest of the paper. SVCCA’s variance-retention step needed only 1–2 principal components at every depth and seed to characterize either representation, consistent with terminal-phase neural collapse in deep classifiers generally [18].

Geometric alignment rising with depth does not track functional transfer: lund-to-part’s retained AUC (Table II) is flat at 0.991–0.992 across the same depth range where CKA rises from 0.76 to 0.93, and its absolute rejection gains only 8 units over that range while remaining less than half of ParT’s own. The two front ends occupy increasingly similar subspaces at greater depth, and this similarity has little visible efect on how much of Lund-ParT’s information ParT’s downstream can extract. Geometric similarity and functional interchangeability are diferent properties. This is the practical case for model stitching over a representational-similarity measure alone: geometry describes the representations, stitching tests what a specific downstream computation can do with them.

## IV. WHAT THE REPRESENTATION IS MISSING

Section III located the deficit in the representation. This section asks which jets are responsible, using scores averaged over five seeds for standalone ParT and the lund-to-part stitch at $d = 1$ and $d = 5$

We find each model’s own threshold at $\varepsilon _ { S } = 0 . 5$ and compare which background jets each threshold rejects. Of the 100,661 background jets (of 100,971 total) standalone ParT rejects, the stitched model fails on $5 5 7 \pm 2 4$ at $d = 1$ and $5 1 5 \pm 2 3$ <sub>at d = 5 (</sub>√<sub>N counting uncer-</sub> tainty): 0.6% and 0.5% of standalone ParT’s rejected set. The stitched model rejects 108 background jets standalone ParT does not at $d = 1$ , and 107 at $d = 5$ . Restricted to all 100,971 background jets, the two models’ score rankings agree closely (Spearman $\rho = 0 . 9 5 5$ at d = 1, 0.956 at $d = 5 )$ . The deficit is concentrated in a small population, not a general loss of ranking resolution.

That population is not a random 0.6% of standalone ParT’s rejected background. Figure 4 shows its jetmass distribution sitting almost entirely above 150 GeV: mean 180 GeV, median 179 GeV, interquartile range 166– 191 GeV, against a general background distribution with mean 95 GeV and median 82 GeV. There is no corresponding shift in $p _ { T }$ (597 GeV against 595 GeV). These are the highest-multiplicity background jets in the sample (mean 67.2 real constituents at $d = 1$ and 66.3 at $d \ = \ 5 .$ , against 43.3 typical) whose accidental invariant mass mimics a real top, distinguishable from a genuine top decay only through substructure. Despite carrying more constituents, this population has fewer real primary-Lund-plane splittings than typical background (mean 11.3 at $d = 1$ and 11.2 at $d = 5$ , against 12.6): more of its radiation lies of the single hardest branch the primary Lund plane records. A jet whose extra particles all hang of softer branches presents the primary plane with a shorter sequence than an ordinary jet, even though it contains more to describe. This is the diagnosis the next section tests.

## V. CONFIRMATION: THE FULL DECLUSTERING TREE

If the deficit is the absence of of-primary-branch substructure, rebuilding Lund-ParT on the full $\mathrm { C } / \mathrm { A }$ declustering tree, matching LundNet’s own graph construction [2], should repair it. We rebuild Lund-ParT with every splitting in the full tree as its node set, $n - 1$ nodes for n real constituents, mean sequence length 42.3 on background jets against the primary branch’s 12.6, and repeat the identical protocol unchanged.

Standalone Lund-ParT’s own rejection at $\varepsilon _ { S } ~ = ~ 0 . 5$ rises from 136 ± 4 to 248 ± 5 (Table I), closing 72% of the distance to ParT’s own $2 9 1 \pm 1 3$ . The AUC gap closes in the same proportion but remains, as everywhere in this paper, small in absolute terms: 0.9759 to 0.9822 against ParT’s 0.9838. Figure 5 compares lund-to-part’s absolute rejection under both representations, against the same fixed ParT target throughout. The primary-branch curve is nearly flat, 128 to 136 across depths 1–5; the full-tree curve rises steeply from 134 at $d = 1$ to 242 at $d = 5 ,$ approaching ParT’s own 291. The two curves are statistically indistinguishable at $d = 1$ , where only one of the target’s blocks has been replaced, and separate progressively as more of the Lund representation’s own processing is included – which is what a representational rather than computational diference should do.

The leaked population shrinks correspondingly, from $5 5 7 \pm 2 4$ jets to $3 9 9 \pm 2 0 ~ ( 0 . 4 \%$ of standalone ParT’s rejected set) at $d = 1$ , and from $5 1 5 \pm 2 3$ to $1 7 7 \pm 1 3$ (0.2%) at $d = 5$ . The shrinkage is far larger at $d = 5 ,$ a factor of 2.9, than at $d \ : = \ : 1$ where it is 28%; this mirrors the separation of the two curves in Fig. 5 and has the same explanation, that at $d = 1$ almost all of the computation is ParT’s own and there is correspondingly little of the Lund representation in the result to repair. The Spearman rank correlation with standalone ParT rises from 0.955 to 0.975 at $d = 5 .$ , so the repair sharpens agreement in the bulk as well as in the tail.

What remains is still concentrated near the top quark mass (mean 180 GeV at d = 1, 177 GeV at $d = 5 )$ , but its relationship to splitting count changes sign. Under the primary branch, the leaked population had fewer splittings than typical background despite more constituents; under the full tree, splitting count tracks constituent count exactly, as it must once every splitting is recorded (54.2 splittings for 55.2 particles at $d = 1 \colon$ ; 54.0 for 55.0 at $d = 5 )$ . The signature that identified the deficit is gone, in the same jets and in the same direction the diagnosis required. Linear CKA between the two front ends also rises further under the full tree, $0 . 7 4 7 \pm 0 . 0 7 8$ at $d = 1$ to $0 . 9 6 0 \pm 0 . 0 0 2$ at $d = 5 ,$ against $0 . 7 5 9  0 . 9 3 4$ under the primary branch. The residual leakage in this hard-mass region is consistent with ordinary classifier disagreement, not a structurally missing quantity.

A known omission, a detected deficit localized to a specific jet population, and a repair that removes most of both: this is the validation the diagnostic needs.

## VI. WHICH NODES CARRY THE REPAIR

Section V establishes that restoring the full declustering tree repairs the deficit. It does not say which of the roughly 36 extra nodes per jet the network uses to do it. The distinction matters practically: “record the whole tree” and “record the handful of splittings that carry the discrimination” are diferent prescriptions for anyone designing a Lund-based tagger, and only the second would suggest a cheaper representation is available. This section answers the question directly, by masking nodes at inference on the already-trained full-tree weights. Noth ing here involves retraining.

Labelling a node as primary or secondary costs nothing, because the full-tree construction already orders its output that way. The traversal in Sec. V is a pre-order walk that always descends the harder subjet first, so the primary branch is exactly the leading contiguous block of a jet’s node array, and its length is the node count the primary-branch representation stores for the same jet. We checked this rather than assuming it: on 2000 test jets drawn at random the leading block of the full-tree features is bit-identical to the primary-branch features, with no exceptions, and the check is repeated on every run of the analysis. Averaged over the 40,000 test jets used here, a jet carries 12.0 primary and 35.9 secondary nodes.

Figure 6 shows what the two labels mean on a single jet. The primary branch is the spine running left to right, one node per declustering step, and every secondary node hangs somewhere below it in a subtree the primary Lund plane never records.

The two populations are also distinct in physics terms, as Fig. 7 makes concrete. Following the harder subjet at every step is a good way to find hard structure, and the primary branch duly holds most of it: 51% of primary nodes sit in the hard region $k _ { T } > 1 \mathrm { G e V }$ , against 13% of secondary nodes. Its density occupies the familiar primary-plane triangle, with a pronounced hard, wideangle lobe. The secondary population instead concenlund → part stitch, $d = 1 \colon$ background jets responsible for the rej@0.5 gap

![](images/eb0fc424fc4dfd55ed20f2e1a25c7db9d5a1ed872a32eb00e0e0d4dd08eb2f39.jpg)

![](images/a437ec20a408e87e85c8a78275dd94279c100495cfd57d59df68e1c8ebb984f6.jpg)  
FIG. 4. Jet mass and $p _ { T }$ for all background jets, the subset rejected by standalone ParT at its own $\varepsilon _ { S } = 0 . 5$ threshold, and the subset the lund-to-part stitch $( d = 1 )$ fails to reject that standalone ParT would have. The afected population concentrates at and above the top quark mass; there is no corresponding efect in $p _ { T }$

lund → part absolute background rejection (fixed ParT target)

![](images/b508bea62fcba89f42fa58c247c6ad66e9bb41c728d43f02d100cf0f2544a290.jpg)

![](images/f0328cef83830540b8e80f31834bf5fc2a62e3543dbaa5a95637a56e766ee7fe.jpg)  
FIG. 5. Lund-to-part absolute background rejection at $\varepsilon _ { S } = 0 . 5$ and 0.3, primary-branch Lund-ParT versus the full-declustering tree rebuild, same ParT target throughout. The horizontal line marks ParT’s own standalone value.

trates in a single soft, collinear blob, 59% of it in that one quadrant. A top decay is distinguished from a QCD jet largely by hard, wide-angle splittings, so this alone suggests the primary branch does more work than its node count implies.

The probe is a masking ablation. For each condition we suppress a chosen set of nodes and re-evaluate the same frozen weights; because the network builds its key padding mask from the input mask and zeroes the embeddings of masked nodes, handing it a modified mask is exactly the intervention, with no change to the model. Two details matter for the result to mean anything. First, each deletion is paired with a size-matched control, so that the number of surviving tokens is held fixed and only the identity of the survivors varies; without this, any drop would conflate the loss of particular nodes with the loss of sequence length. Second, we compact the surviving nodes to the front of the sequence before the forward pass. The network trims each batch to its longest real sequence by taking a fixed number of leading columns, which is lossless for a contiguous block but silently discards scattered survivors, and a jet that loses all of them returns an undefined score. Compaction removes that asymmetry between conditions and is invisible to the network, which has no positional encoding. Jets whose every node is primary, 69 of 40,000, are excluded from all conditions alike, since “delete the primary branch” is not defined for them.

![](images/e5478798aabd97f8ad626929b74e8c16f6a687a23dae05527bcf41d17b7c9ab9.jpg)  
declustering step along the primary branch

FIG. 6. The Cambridge/Aachen declustering tree of one real top jet. The primary branch is the horizontal spine; the subtrees hanging below it are what the primary Lund plane discards and the full tree keeps. The jet was chosen to be small enough to draw and to have a primary fraction (0.26) close to the population’s 12.0/48.0, but is otherwise unremarkable.  
![](images/81eb9fd0ec0a0f1110c723516a3777b2535ccd067dc2a0c388ce058e60a9b6e6.jpg)

![](images/9b7a0de3e75d852fbdb20f37dd9c49dd3c83738431ac08716718360450138ffa.jpg)  
FIG. 7. Lund-plane density $\rho = N _ { \mathrm { j e t } } ^ { - 1 } \mathrm { d } n / \mathrm { d } \ln ( 1 / \Delta R )$ d ln $k _ { T }$ of the two node populations, over all 40,000 test jets. Dotted lines mark $k _ { T } = 1 \mathrm { G e V }$ and $\Delta R = 0 . 1$ . The colour scales are independent, since the secondary population is three times as numerous and a shared scale leaves the left panel unreadable; the per-jet node counts in the titles fix the relative normalisation.

TABLE IV. Background rejection and AUC under each masking condition, full-tree Lund-ParT, mean ± standard deviation over 5 seeds, 40,000 test jets. Conditions are grouped into matched pairs: the same number of surviving nodes, differing only in which nodes those are.
<table><tr><td>condition</td><td>nodes</td><td>rej. at  $\varepsilon _ { S } = 0 . 5$ </td><td>AUC</td></tr><tr><td>all nodes</td><td>48.0</td><td>230.6 ± 8.2</td><td>0.9818</td></tr><tr><td>drop 12 secondary</td><td>36.2</td><td>186.5 ± 16.3</td><td>0.9789</td></tr><tr><td>drop 12 random</td><td>35.9</td><td> $5 3 . 9 \pm 3 . 5$ </td><td>0.9292</td></tr><tr><td>drop 12 primary</td><td>35.9</td><td> $9 . 2 \pm 5 . 6$ </td><td>0.8307</td></tr><tr><td>keep 12 primary</td><td>12.0</td><td> $\overline { { 3 4 . 7 \pm 1 2 . 4 } }$ </td><td>0.9229</td></tr><tr><td>keep 12 random</td><td>12.0</td><td> $4 . 4 \pm 1 . 8$ </td><td>0.6845</td></tr></table>

Table IV collects the result, and both matched pairs point the same way. Removing the twelve primary nodes costs almost everything, $2 3 0 . 6 \ \to \ 9 . 2$ , while removing twelve secondary nodes costs little, $2 3 0 . 6 \to 1 8 6 . 5$ , with AUC barely moving at all. Keeping only the twelve primary nodes retains more than keeping twelve random ones, 34.7 against 4.4. The intermediate condition is the one that shows the efect is specific rather than an artifact of deletion: twelve nodes drawn at random from the whole jet, about a quarter of which are primary by construction, land at 53.9, between the two extremes and much closer to the primary-deletion end than their composition alone would suggest.

The comparison that carries the most weight is the first pair, because it holds the number of deleted nodes fixed at twelve and varies only their provenance: 9.2 against 186.5, a factor of twenty. It also disposes of the obvious objection to any ablation of this kind. Weights trained only on complete trees have never seen a shortened sequence, so a drop might reflect nothing more than that unfamiliarity. If that were the explanation, deleting twelve secondary nodes would hurt as much as deleting twelve primary ones. It does not, by a factor of twenty, so the sensitivity is to which nodes are removed and not to the fact of removal.

Removing secondary nodes a fraction at a time, rather than twelve at a time, shows how their contribution is spread (Fig. 8, right). Rejection falls from $2 3 0 . 6 \pm 8 . 2$ with every node present to $1 9 4 . 9 \pm 1 4 . 2$ at a quarter removed, $1 1 8 . 8 \pm 1 5 . 1$ at a half, $5 8 . 9 \pm 1 6 . 1$ at three quarters, and $3 4 . 7 \pm 1 2 . 4$ when none is left. The first quarter is nearly free, costing 15%, and each subsequent quarter costs more than the one before it. There is no step, no fraction at which the curve falls of a clif, which is what a small load-bearing subset would have produced. The endpoint is the sharpest statement available here: stripping all 36 secondary nodes still leaves the network at 34.7, well above the 9.2 it reaches when the 12 primary nodes go instead and all 36 secondary ones remain.

Taken together these measurements say something more specific than Sec. V alone could. At inference the trained full-tree network leans overwhelmingly on the primary branch: twelve primary nodes support a higher rejection on their own, 34.7, than all thirty-six secondary nodes do without them, 9.2. Yet training on the full tree is what lifts standalone rejection from 136 to 248 in the first place. Both statements are measurements, and the way to hold them together is that the secondary nodes act during training, shaping the representation the network learns to build, rather than supplying discriminating information to be read of at test time. On that reading the repair is real but not localized, and the honest answer to which of the extra nodes carry it is that no small subset does.

This has a consequence for anyone hoping to shortcut the full tree. A representation that recorded the primary branch plus a modest selection of secondary splittings would not, on this evidence, recover the repair, because the secondary contribution is not concentrated anywhere for such a selection to capture. We would also caution against reading the individual conditions in Table IV as importance scores in any stronger sense. They are comparisons within matched pairs, and it is only the contrast within a pair that is interpretable; the absolute values still carry whatever cost the network pays for a sequence shorter than any it was trained on.

## VII. WHERE ALONG THE PRIMARYBRANCH INFORMATION SITS

Section VI asks which nodes of the full declustering tree a trained network relies on, and finds it relies almost entirely on the primary branch. This section asks the complementary question of the primary branch itself: a trained, frozen Lund-ParT built on the primary branch alone reads a jet as a sequence of declustering steps, and we ask which single step it can least aford to lose.

Cambridge/Aachen declustering is angular ordered, so position along the chain is a proxy for angular scale. We verified this rather than assuming it: the last node sits at smaller $\Delta R$ than the first in 100.0% of jets, in both top tagging and quark/gluon tagging, the latter as described in Appendix A, and the median $\Delta R$ falls monotonically along the chain, from 0.69 to 0.008 for top tagging and 0.35 to 0.004 for quark/gluon. The first step is therefore the widest-angle splitting and the last the most collinear.

Three ablations are compared on the already-trained, frozen primary-branch Lund-ParT, with no retraining: remove the first step, remove the last, or remove one chosen uniformly at random. Anchoring at the ends is what makes the comparison clean. Jets have diferent chain lengths (12.0 steps on average for top tagging, 13.6 for quark/gluon), so ablating a fixed index would act on a shrinking and increasingly atypical subset as the index grows; every jet has a first and a last step whatever its length, so all three conditions act on 100% of the jets with nothing padded, normalized or truncated. Jets with a single node cannot give one up and are excluded from every condition alike (5 of 201,998 for top tagging, 9 of 199,994 for quark/gluon).

The random condition is the control, and it carries the argument. Removing any node costs something, partly because a shortened sequence is unfamiliar to weights trained on complete ones, so an absolute drop cannot dis tinguish “this step mattered” from “removal hurts”. Only the paired diference against random, computed seed by seed so that common cost cancels, is interpretable; all claims below are stated that way. Survivors are compacted to the front of the sequence before the forward pass, following Sec. VI’s procedure, since the network has no positional encoding and the batch trimmer would otherwise silently discard a non-contiguous mask. We report background rejection at $\varepsilon _ { S } = 0 . 5$ and the threshold-free mean per-jet |∆ score|, on cached checkpoints over five seeds.

The widest-angle splitting is worth far more than an average one in both tasks, and about three times more so for top tagging: +35.2% of the unablated rejection against +11.5%. The most collinear splitting is worth correspondingly less than average – for quark/gluon by almost exactly the same magnitude the first splitting is worth more (−11.8% against +11.5%), for top tagging by a smaller margin than the first splitting’s excess (−29.4% against +35.2%). Importance falls monotonically as the splitting becomes more collinear. The thresholdfree measure agrees independently (Fig. 9, right): mean |∆ score| runs $0 . 1 7 5  0 . 0 6 0  0 . 0 1 2$ for top tagging across first, random and last, a factor of 14.6 between the two ends, and $0 . 0 6 6 \to 0 . 0 4 4 \to 0 . 0 2 5$ for quark/gluon, a factor of 2.6 – the same ordering, the same contrast between the two tasks, with no threshold involved.

Jet-tagging information along the primary chain is concentrated at wide angles, and markedly more so for top tagging than for quark/gluon discrimination. Removing the single widest-angle splitting costs top tagging 71% of its background rejection, essentially twice what removing a random splitting costs, while the most collinear splitting is nearly free to discard; the same ordering holds for quark/gluon, with roughly a third of the contrast. This has a natural reading: a boosted top is identified by a hard, localized prong structure whose opening angles $2 m _ { t } / p _ { T }$ and $2 m _ { W } / p _ { T }$ sit near the first declustering, so one wide-angle splitting carries much of the discrimination, while quark/gluon separation rests on Casimir scal ing, which acts across the whole emission spectrum and accumulates over many steps, so no single step is decisive. This corroborates, from an independent direction, why discarding of-primary branches was close to fatal for top tagging and only mildly costly for quark/gluon (Sec. V, Appendix A).

![](images/3927c60d4733e3c7d97228259c306f3829e6ce8f94511ec5795f9071d3e40580.jpg)

![](images/bdd535368fdf2f8e8b5b954d2cf17052ec628b2a82142531ea903cbce0609342.jpg)  
FIG. 8. Left: background rejection under each masking condition, with the matched pairs adjacent. Right: rejection as a growing fraction of each jet’s secondary nodes is removed, ending at the primary branch alone. The dashed line marks the complementary condition, the primary branch removed and every secondary node kept.

TABLE V. One node removed per jet from the standalone primary-branch Lund-ParT, mean ± standard deviation over 5 seeds. “Excess” is the paired diference against the random control, as a percentage of the unablated rejection, with its paired t statistic.
<table><tr><td>dataset</td><td>condition</td><td> $\mathrm { r e j . ~ a t ~ } \varepsilon _ { S } { = } 0 . 5$ </td><td>loss</td><td>excess vs. random</td></tr><tr><td>top tagging</td><td>unablated</td><td> $1 3 6 . 2 8 \pm 4 . 2 6$ </td><td></td><td></td></tr><tr><td></td><td>remove first</td><td> $4 0 . 1 3 \pm 0 . 8 3$ </td><td>-70.6%</td><td>+35.2% (t=55.8)</td></tr><tr><td></td><td>remove random</td><td> $8 8 . 0 7 \pm 2 . 0 0$ </td><td>-35.4%</td><td></td></tr><tr><td></td><td>remove last</td><td> $1 2 8 . 1 2 \pm 1 . 9 3 $ </td><td>-6.0%</td><td>−29.4% (t= − 24.0)</td></tr><tr><td>quark/gluon</td><td>unablated</td><td> $\overline { { 3 0 . 6 2 \pm 0 . 1 7 } }$ </td><td></td><td></td></tr><tr><td></td><td>remove first</td><td> $2 2 . 3 2 \pm 0 . 2 6$ </td><td>-27.1%</td><td>+11.5% (t=21.3)</td></tr><tr><td></td><td>remove random</td><td> $2 5 . 8 5 \pm 0 . 3 2$ </td><td>-15.6%</td><td></td></tr><tr><td></td><td>remove last</td><td> $2 9 . 4 6 \pm 0 . 1 6$ </td><td>-3.8%</td><td>−11.8% (t= − 20.6)</td></tr></table>

Two limits are worth stating. This measures what a trained network uses, not what is physically available; Sec. VI found the two can difer sharply. And removing any single node costs far more than its naive $1 / N$ share of the chain would predict: a 1/12.0 share is 8.3% for top tagging, against the 35% a random node actually costs; a $1 / 1 3 . 6 $ share is 7.4% for quark/gluon, against 16%. There is a real distribution-shift penalty, which is why only the paired comparison against the random control is quoted as a result.

## VIII. DISCUSSION AND CONCLUSION

Model stitching, applied to ParT and a Lund-plane network built the same way, separates a performance gap into its two possible causes and identifies which one is responsible. The primary-branch representation is genuinely weaker than ParT’s, independent of which downstream reads it; this shortfall is invisible in AUC and appears only in background rejection at fixed, low signal eficiency; it is concentrated in a specific, physically identifiable population of high-multiplicity, high-mass background jets with radiation lying of the primary branch; and it is repaired, in the population it should repair and in the direction it should move, by restoring exactly the branches the primary plane omits.

Section VI then pushes past the repair to ask what performs it. The answer complicates the natural reading of Sec. V. At inference the repaired network depends almost entirely on the primary branch, and the secondary nodes that the full tree adds are individually close to redundant: deleting twelve of them costs a fifth of the background rejection, while deleting twelve primary nodes costs almost all of it. The extra nodes are what makes the better representation learnable, not what the finished network reads. We think this is worth stating plainly because it is the kind of thing a performance comparison would never surface, and because it warns against the obvious engineering shortcut of keeping the primary branch plus a few chosen secondary splittings.

![](images/b758f6a93bd720ab7c6300210047faf4fbd54e3ed057e06c34b1aa472e49649d.jpg)

![](images/7524b643ba16bcb02c5ac9cc5692fdf10c6b6eef4b385bbb7f5e02e26a53e826.jpg)  
FIG. 9. Left: loss in background rejection when one node is removed. Right: the same conditions under the threshold-free per-jet score shift. Error bars are the seed standard deviation.

![](images/b1912c2b09d5ce41adf6a99e251cbfb6d48b72c9602ad7170be97d40957f62f1.jpg)  
FIG. 10. The measured importance placed on the angular scale it corresponds to. Circles mark the median $\Delta R$ of successive declustering steps; filled circles are the two steps actually ablated, with area proportional to their excess damage over the random control, and hollow circles are the intermediate steps, which were not measured. Dotted lines mark the top and W decay opening angles $2 m / p _ { T }$ at $p _ { T } = 6 0 0 ~ \mathrm { G e V }$ drawn on the top-tagging lane only since they are a statement about top jets. The first splitting of a top jet falls close to these scales; quark/gluon jets have no such scale, and their importance profile is correspondingly flatter.

None of this required the primary branch to be a strong tagger on its own. It is not, by construction: LundNet uses the full declustering tree precisely because the primary branch is incomplete. We used it as a positive control because its deficit was known before any stitching result was computed, which is what makes correctly recovering that deficit, and localizing it to the right jets, evidence that the method works.

Section VII asks a complementary question of the primary branch in isolation: its own discriminating power turns out to be concentrated at its wide-angle end, close to the top and W opening angles, far more sharply for top tagging than for quark/gluon. That is the same picture Sec. VI finds from the branch side: a top jet’s discrimination depends on localized, specific structure rather than being spread evenly across whatever is recorded.

## A. A hypergraph reading of the same representations

One alternative to the conclusion of Sec. III deserved a direct test rather than an argument. Every reader of the primary-branch representation considered so far processes its splittings through pairwise attention. A model that reasons natively over triplets of splittings might do better: three-body correlations among recorded emissions could in principle constrain the radiation that was not recorded, letting a network infer something about the branches the primary plane discards without ever being shown them. Had that been the case, part of the deficit this paper attributes to the representation would have belonged to the reader after all.

We therefore rebuilt the tagger as a hypergraph network, in which a token is not a particle or a splitting but a triplet of them. Hyperedges are the union of a local neighbourhood construction and every triplet among the ten hardest nodes, the second included because purely local triplets sit within a single prong and miss the wideangle structure a boosted top is identified by. The same architecture, the same jets, and the same training protocol are used throughout, with only the node set changed between the three variants. These runs use 10% of the dataset and a single seed, so they are indicative and are not comparable with the measurements elsewhere in this paper; the companion figures in the preceding sections are untouched by them.

TABLE VI. Hypergraph taggers against the corresponding stitching-study networks, all at 10% of the dataset. The hypergraph column is a single seed and carries no uncertainty; the right-hand column is the mean over 5 seeds at the same scale.
<table><tr><td>representation</td><td>hypergraph rej. at  $\varepsilon _ { S } = 0 . 5$ </td><td>this paper&#x27;s network rej. at  $\varepsilon _ { S } = 0 . 5$ </td></tr><tr><td>particles</td><td>133</td><td> $2 0 8 \pm 1 2$ </td></tr><tr><td>Lund, primary branch</td><td>116</td><td> $1 1 2 \pm 7$ </td></tr><tr><td>Lund, full tree</td><td>181</td><td> $2 0 1 \pm 1 1$ </td></tr></table>

The test comes out negative, and cleanly so. On the primary branch the hypergraph reaches a rejection of 116 against Lund-ParT’s 112 ± 7 at the same scale: the two agree inside one standard deviation, so the explicitly three-body reader extracts no more from that representation than the pairwise one does. The gap between the two Lund representations survives intact, 116 rising to 181 when the full tree is restored, a 55% gain against the 79% the stitching study measures for the same pair. A model built to see higher-order structure, given only the primary declusterings, does not recover what the full tree carries.

We draw attention to how little this buys the present study, because the temptation to present it as an extension should be resisted. Its entire value is eliminative: it removes one competing explanation for Sec. III’s result and leaves every number, figure and conclusion in this paper exactly as they were. It improves neither the diagnosis nor the repair, and it identifies no jet population that Sec. IV did not already identify.

Nor can it be folded into the stitching framework without damage, which is the more important point. This paper’s design rests on one architecture reading several representations, so that a stitched condition’s shortfall is attributable to representation content. The hypergraph tagger breaks that symmetry: it carries no pairwise attention bias, since at roughly two hundred hyperedge tokens rather than fifty particles the bias tensor does not fit alongside training activations, and on particles it reaches only two thirds of ParT’s rejection. Stitching it against ParT would therefore reintroduce precisely the confound between representation content and architecture that Sec. I sets out to remove, and any deficit so measured would be uninterpretable. We report the comparison as a consistency check on the representational claim and build nothing on it.

The result is scale-limited in a stated, not defensive, way. Both networks are trained at 50% of the reference dataset, and standalone rejection here is below values reported at full scale for comparable architectures. Every claim in this paper rests on a large efect – a factor of two in standalone rejection, a factor of 2.9 in the leaked population at the deeper repair depth – and we would not expect this scale to change which population is responsible or the direction of the repair. Whether the same protocol, applied to a source-target pair whose representations are closer in strength, resolves depth-by-depth diferences at this data scale is a separate question we do not attempt to answer here; it needs more data, more seeds, and is better reported on its own. The consumerasymmetry result of Sec. III B is the one place where five seeds are visibly too few: the efect points the same way in all ten individual comparisons and in all five seed-level scores, yet reaches only $p = 0 . 0 6 2$

Two limitations of scope deserve naming. First, we excluded d = 6 from the analysis on structural grounds (Sec. II C) rather than reporting it with a caveat. We think this is the right call, since at that depth the target contributes no self-attention blocks and the quantity measured is not the one the sweep is designed to measure, but it does mean the sweep stops one layer short of the full stack and we cannot speak to what happens there. Second, a consumer-side fine-tuning check – unfreezing the first target block downstream of the stitch point, to test whether the consumer asymmetry survives letting the receiving block adapt – was performed at an earlier, smaller data scale and was inconclusive there, with a seed-dependent instability that we could not attribute to a learning-rate artifact. We have not repeated it at this scale, and we do not use it as evidence anywhere in this paper.

A natural question is whether this diagnosis is specific to top tagging, where the omitted structure is a known, localized pair of prongs. Appendix A repeats the full pipeline on quark/gluon tagging, a task whose discrimination rests on Casimir-scaling radiation patterns rather than localized substructure, and finds a diferent balance between representation and repair: the primary branch alone already retains 77.6% of ParT’s standalone rejection, against 46.9% on top tagging, and restoring the full tree closes only 33% of the remaining gap, against 72%. The consumer-asymmetry result of Sec. III B does not reproduce there either. The two tasks nonetheless agree on one point: a complete declustering tree sits at essentially the same fraction of ParT’s rejection on both, about 85%, which we read as a cost intrinsic to the Lund encoding itself rather than to any specific omitted branch. The quark/gluon stitch measurements are noisier at shallow depth than the top-tagging ones, and Appendix A states that caveat alongside every number it afects.

Our contribution is not model stitching itself, which is due to Bansal, Nakkiran, and Barak [8]. It is the classtoken stitch point, which removes the sequence-length ob stacle to applying stitching across architectures with unrelated per-token structure; the same-architecture Lund-ParT, which removes the confound between representation content and message-passing mechanism; the explicit dual normalization of Sec. II; and the positivecontrol design that lets the method’s output be checked against an independently known answer.

## IX. ACKNOWLEDGMENTS

S.R. is supported by the IIT-Kanpur institutional Ph.D fellowship.

S.G. is supported by the IIT-Kanpur faculty initiation grant (IITK /PHY /2023499), Anusandhan National Research Foundation - Advanced Research Grant (ANRF/ARG/2025/005801/PS) and Anusandhan National Research Foundation - ANRF MISSION AI for Science and Engineering (ANRF/AISE/2025/000373/AISSAI).

## DATA AVAILABILITY STATEMENT

The Kasieczka et al. top-tagging reference dataset used in this work is publicly available at Zenodo (DOI: 10.5281/zenodo.2603256). Code implementing the architectures, stitching protocol, and analysis described here is available from the authors on reasonable request.

## AI USAGE STATEMENT

The authors acknowledge the usage of Anthropic’s Opus-5 model for cross checking the SVCCA analysis and code refactoring. The manuscript however is manually written by the authors themselves.

## Appendix A: Generalization to quark/gluon tagging

The representational deficit identified in Secs. III–VI is specific to top tagging: the primary Lund plane discards the of-axis prongs a boosted top’s three-body decay needs, but how costly that omission is for a classification task with diferent discriminating physics is not obvious a priori. We repeat the identical pipeline and hyperparameters – the same architectures, the same stitching protocol, the same five seeds – on the EnergyFlow quark/gluon dataset [19], a task whose discrimination rests on Casimir scaling of gluon radiation rather than on localized substructure. We report what was measured and state where it is, and is not, resolved.

## 1. Setup

a. Dataset. EnergyFlow quark/gluon jets (Pythia generator, b- and c-initiated jets excluded), 1,000,000 events, shufled under a fixed seed and split 600,000/200,000/200,000 into train, validation and test. EnergyFlow stores each constituent as $( p _ { T } , y , \phi , \mathrm { p i d } )$ these were converted to four-momenta $( E , p _ { x } , p _ { y } , p _ { z } )$

using each particle’s PDG mass from its pid, not a massless approximation, and written into the same HDF5 layout the pipeline reads for top tagging, so both datasets are processed by identical code. Jets carry 43.2 constituents on average at a jet $p _ { T }$ of about 523 GeV. Labels follow the EnergyFlow convention, quark = 1 and gluon = 0, so signal eficiency below is quark eficiency and the rejected set is gluon jets. The test split contains 100,016 gluon jets.

b. Representations. Three inputs to one architecture, as in Sec. II: ParT takes the seven kinematic scalars per particle with the four relativistic invariants of every particle pair as its attention bias; Lund-ParT (primary) takes the five primary-Lund-plane declustering coordi nates, 13.6 nodes per jet on average; Lund-ParT (full tree) takes every splitting of the full Cambridge/Aachen tree, 42.3 nodes per jet.

c. Architecture and training. All three instantiate the same ParticleTransformer class used throughout this paper: 6 pre-norm self-attention blocks, embedding dimension 128, 8 heads, 2 class-attention blocks, and a two-layer head (1,658,814 parameters for ParT, 1,658,616 for Lund-ParT). Training uses AdamW, cross-entropy loss, batch size 256, up to 30 epochs with early stopping on validation AUC (patience 5), and float16 mixed precision. Five seeds throughout; every quoted uncertainty is the standard deviation over those five seeds.

d. Stitching. Identical to Sec. II C: for source A, target B, and depth d, A’s embedding, first d blocks, and class-attention readout are frozen and produce A’s own class token; a single trained Linear(128, 128) map (16,512 parameters) carries it across; B’s remaining blocks, readout, and head complete the forward pass frozen at their standalone values. Only the linear map is optimized: up to 15 epochs, patience 4, on a fixed 20,000-jet subsample. Four conditions are run at every depth and seed, seed-matched. Depths $d = 1 , \ldots , 5$ are reported; $d = 6$ is excluded for the same structural reason given in Sec. II C.

e. Execution. The two Lund modes were run as separate jobs. The ParT backbones are byte-identical between them (verified by checksum), so the ParT column below is one set of five models shared by both halves rather than an independent replicate.

## 2. Standalone performance

Restoring the full declustering tree moves standalone rejection from 30.59 to 33.54 against ParT’s 39.44 (Table VII), closing 33.3% of the gap – far less than the 72% closed on top tagging (Table I).

## 3. Stitched performance

Table VIII gives the absolute rejection of the lund to-part stitch, the quantity through which the two Lund representations are compared, with the same frozen ParT target in both rows.

TABLE VII. Standalone test performance on quark/gluon tagging, mean ± standard deviation over 5 seeds.
<table><tr><td></td><td>ParT</td><td>Lund-ParT (primary)</td><td>Lund-ParT (full tree)</td></tr><tr><td>AUC</td><td> $\overline { { 0 . 9 1 1 0 \pm 0 . 0 0 1 1 } }$ </td><td> $\overline { { 0 . 8 9 0 5 \pm 0 . 0 0 0 3 } }$ </td><td> $0 . 9 0 0 6 \pm 0 . 0 0 0 7$ </td></tr><tr><td>rej.  $\mathrm { a t } \ \varepsilon _ { S } = 0 . 5$ </td><td> $3 9 . 4 4 \pm 0 . 7 3$ </td><td> $3 0 . 5 9 \pm 0 . 1 7$ </td><td> $3 3 . 5 4 \pm 0 . 5 2$ </td></tr><tr><td>rej. at  $\varepsilon _ { S } = 0 . 3$ </td><td> $9 8 . 1 4 \pm 2 . 5 2$ </td><td> $8 2 . 1 3 \pm 0 . 9 2$ </td><td> $8 8 . 1 2 \pm 1 . 5 5$ </td></tr></table>

a. Consumer-asymmetry test. Averaging the selfminus-cross diference in raw rejection over $d = 1 , \ldots , 5$ per seed, then over the two directions, gives a mean advantage for ParT’s downstream over Lund-ParT’s of $- 1 . 1 \pm 2 . 7$ rejection units $( t = - 0 . 9 5 , p = 0 . 3 9 5 , n = 5$ seeds). The sign is negative and the result is not signifi cant – the opposite of the top-tagging result (Sec. III B), where the same test gave $+ 8 \pm 7 \ ( p = 0 . 0 6 2 )$

## 4. Geometric alignment

Linear CKA and SVCCA between ParT’s and Lund-ParT’s class tokens, computed on a fixed probe sample independently of any stitching or classification objective. The depth index here is the layer at which a class token is read from a standalone network, so unlike the stitch sweep it remains defined at $d = 6$

## 5. Jet-level localization

Of the 100,016 gluon jets in the test split, standalone ParT rejects 97,575 at its own $\varepsilon _ { S } = 0 . 5$ threshold. Table XII counts those the stitched model fails to reject that standalone ParT would have, with $\sqrt { N }$ counting uncertainties.

Spearman rank correlation with standalone ParT over all gluon jets is 0.812 (d = 1) and 0.836 (d = 5) for the primary branch, and 0.661 and 0.907 for the full tree.

## 6. Reading the numbers

Three features of this run should be stated explicitly, because they bear on what can and cannot be concluded from the tables above.

a. Most of the depth structure is not resolved. The per-depth stitch measurements carry seed spreads of 20– 55% in relative terms at $d \leq 4 \colon 1 5 . 1 \pm 8 . 2$ at $d \ : = \ : 1$ and $2 3 . 6 \pm 1 2 . 4$ at $d = 4$ for the primary branch, for instance. The non-monotonic entries in Tables IX and X – lund→part dipping at $d = 4 ,$ , part→lund at $d = 2 \textrm { -- }$ are consistent with that scatter and should not be read as structure. Only $d = 5 ,$ where the spreads fall to a few percent, is measured sharply. This is a substantially noisier regime than the top-tagging study, where per-depth spreads were a few percent throughout.

b. The consumer asymmetry does not appear here. In the top-tagging study ParT’s downstream extracted at least as much from the Lund representation as Lund $\mathrm { P a r T } ^ { \prime }$ s own did, at every depth. On these jets the ordering is mixed across depths and the seed-level test returns $- 1 . 1 \pm 2 . 7$ with $p = 0 . 3 9 5$ : the opposite sign, and consistent with zero.

c. The full tree does not dominate the primary branch at every depth. Standalone, the full tree is ahead, 33.54 against 30.59. Through the stitch it is behind at $d =$ 1, 2, 3 and ahead only at $d = 4 , 5$ (Table VIII), and its leaked population at $d = 1$ is larger than the primary branch’s, 5539 against 2607, shrinking below it only at $d = 5 ,$ 897 against 1306. The $d = 1$ comparison also has the lowest rank correlation of the four, 0.661. In the top-tagging study the full tree led at every depth and its leaked population was smaller at both depths checked.

## 7. Physical interpretation

The comparison that carries the physics is against the same analysis run on the top-tagging reference dataset, at 50% of that dataset and over the same five seeds, where ParT reaches 290.7±13.2, the primary branch 136.3±4.3, and the full tree $2 4 7 . 7 \pm 5 . 3$ at $\varepsilon _ { S } = 0 . 5$ (Table I).

a. The primary Lund plane is a far better representation here than for top tagging. On its own it retains $7 7 . 6 \%$ of ParT’s rejection on quark/gluon tagging against 46.9% on top tagging (Table XIII). This follows the physics of the two tasks. Quark/gluon separation rests on Casimir scaling, $C _ { A } / C _ { F } = 9 / 4 \mathrm { : }$ gluons radiate more, which shows up as the rate and hardness of emissions along the jet’s hardest chain together with the resulting multiplicity. That is precisely what the primary branch samples, since it records ln $k _ { T }$ and ln $\Delta R$ at every step of that chain. Top tagging instead requires a localized three-prong structure, two prongs of which sit on branches the primary plane discards. The same omission is close to fatal for one task and only mildly costly for the other.

b. Restoring the full tree therefore buys much less. It closes 33% of the gap to ParT here against 72% there. In a quark or gluon jet the secondary branches carry further soft radiation of the same character the primary chain already samples, so they add statistics on a property that was already measured. In a top jet they carry the missing hard prongs. The construction is identical; what it recovers is not.

TABLE VIII. Absolute background rejection at $\varepsilon _ { S } ~ = ~ 0 . 5$ for the lund-to-part stitch on quark/gluon tagging, both Lund representations, against the same ParT target. ParT’s own standalone value is 39.44.
<table><tr><td colspan="6"></td></tr><tr><td>representation</td><td>1</td><td>2</td><td>d 3</td><td>4</td><td>5</td></tr><tr><td>primary branch</td><td> $\overline { { 1 5 . 1 \pm 8 . 2 } }$ </td><td> $\overline { { 2 1 . 9 \pm 7 . 0 } }$ </td><td> $\overline { { 2 5 . 0 \pm 4 . 6 } }$ </td><td> $\overline { { 2 3 . 6 \pm 1 2 . 4 } }$ </td><td> $\overline { { 3 0 . 1 \pm 0 . 3 } }$ </td></tr><tr><td>full tree</td><td> $7 . 8 \pm 7 . 1$ </td><td> $1 5 . 2 \pm 7 . 9$ </td><td> $2 0 . 8 \pm 6 . 0$ </td><td> $3 0 . 6 \pm 0 . 8$ </td><td> $3 2 . 8 \pm 0 . 7$ </td></tr></table>

TABLE IX. Retained AUC, target-normalized $( R ^ { \mathrm { t g t } } )$ , primary-branch run, quark/gluon tagging.
<table><tr><td></td><td colspan="5">d</td></tr><tr><td>direction</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>lund→part</td><td> $\overline { { 0 . 9 3 0 \pm 0 . 0 5 3 } }$ </td><td> $0 . 9 6 3 \pm 0 . 0 1 6$ </td><td> $0 . 9 6 9 \pm 0 . 0 0 8$ </td><td> $0 . 8 5 8 \pm 0 . 2 6 3$ </td><td> $\overline { { 0 . 9 7 7 \pm 0 . 0 0 2 } }$ </td></tr><tr><td>part→lund</td><td> $0 . 9 4 1 \pm 0 . 0 6 6$ </td><td> $0 . 8 3 9 \pm 0 . 3 4 1$ </td><td> $0 . 9 8 9 \pm 0 . 0 2 2$ </td><td> $1 . 0 0 4 \pm 0 . 0 0 7$ </td><td> $1 . 0 1 6 \pm 0 . 0 0 6$ </td></tr><tr><td>part-self</td><td> $0 . 9 3 1 \pm 0 . 0 4 5$ </td><td> $0 . 9 5 3 \pm 0 . 0 3 5$ </td><td> $0 . 9 6 4 \pm 0 . 0 2 1$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 8$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 4$ </td></tr><tr><td>lund-self</td><td> $0 . 9 7 0 \pm 0 . 0 2 6$ </td><td> $0 . 9 8 1 \pm 0 . 0 1 8$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 1$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 3$ </td></tr></table>

c. The residual shortfall against ParT is the same on both tasks. Even a complete declustering tree sits at 85% of ParT’s rejection on either dataset (85.0% and 85.2%). Since the two tasks difer in which structures discriminate, a shortfall this similar is more naturally read as a cost of the Lund encoding itself than of any particular omitted branch: the Cambridge/Aachen sequence is one hierarchical summary of a jet, and what that summary compresses away appears to cost a roughly taskindependent fraction of the rejection.

f. What these readings do not rest on. The depthby-depth stitch measurements carry 20–55% seed spreads below d = 5, so the ordering at any individual shallow depth is not resolved on its own; the statement about depth above rests on the direction being consistent across $d = 1 , 2 , 3$ together with the independent leakage and

e. The consumer asymmetry disappears. On top tagging, ParT’s downstream extracted at least as much from the Lund representation as Lund-ParT’s own did, giving $+ 8 . 0 \pm 7 . 0$ rejection units $( p = 0 . 0 6 2 )$ . Here the test returns $- 1 . 1 \pm 2 . 7 \ ( p = 0 . 3 9 5 )$ . The natural reading is that little latent information remains for a stronger downstream to recover once the representation already reaches 77.6% of ParT: the primary branch is near its ceiling on this task, where on top tagging it was far from it.

d. The full tree’s advantage requires depth before it appears. Through the stitch the full tree is behind the primary branch at $d = 1 , 2 , 3$ and ahead only at $d = 4 , 5$ (Table VIII); its leaked population at d = 1 is the larger of the two, 5539 against 2607, and carries the lowest rank correlation of the four comparisons, 0.661; and its geometry starts further from ParT’s, CKA 0.379 against 0.639 at d = 1, before converging higher by d = 6, 0.919 against 0.884 (Table XI). The roughly 28 additional nodes per jet are therefore not locally readable: they require several rounds of attention before they aggregate into anything discriminating. The information the full tree adds is distributed rather than concentrated.

CKA evidence, not on any single entry. The asymmetry test is underpowered at five seeds, so the absence of a detected asymmetry is not evidence of its absence. Both conclusions compare two datasets at diferent fractions of their respective reference samples, with the architecture and every hyperparameter held fixed but the jet kinematics difering in $p _ { T }$ reach and multiplicity.

TABLE X. Retained rejection at $\varepsilon _ { S } = 0 . 5 ,$ source-normalized $( R ^ { \mathrm { s r c } } )$ , primary-branch run, quark/gluon tagging. lund→part and lund-self divide by Lund-ParT’s standalone 30.59; part→lund and part-self divide by ParT’s 39.44. Uncertainties propagate the numerator’s seed spread only; the denominators are known to 0.6% and 1.9% respectively.
<table><tr><td></td><td colspan="5"></td></tr><tr><td>direction</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>lund→part</td><td> $\overline { { 0 . 4 9 4 \pm 0 . 2 6 8 } }$ </td><td> $\overline { { 0 . 7 1 5 \pm 0 . 2 2 8 } }$ </td><td> $\overline { { 0 . 8 1 8 \pm 0 . 1 4 9 } }$ </td><td> $\overline { { 0 . 7 7 2 \pm 0 . 4 0 5 } }$ </td><td> $\overline { { 0 . 9 8 3 \pm 0 . 0 1 1 } }$ </td></tr><tr><td>lund-self</td><td> $0 . 5 4 7 \pm 0 . 2 5 9$ </td><td> $0 . 6 5 5 \pm 0 . 2 3 4$ </td><td> $0 . 8 5 8 \pm 0 . 0 9 7$ </td><td> $0 . 9 4 4 \pm 0 . 0 3 8$ </td><td> $0 . 9 6 7 \pm 0 . 0 3 9$ </td></tr><tr><td>part→lund</td><td> $0 . 5 2 4 \pm 0 . 1 8 8$ </td><td> $0 . 5 4 2 \pm 0 . 3 0 0$ </td><td> $0 . 6 7 9 \pm 0 . 1 6 6$ </td><td> $0 . 7 9 9 \pm 0 . 0 7 7$ </td><td> $0 . 9 1 8 \pm 0 . 0 4 2$ </td></tr><tr><td>part-self</td><td> $0 . 4 6 0 \pm 0 . 2 2 5$ </td><td> $0 . 5 8 2 \pm 0 . 1 7 6$ </td><td> $0 . 5 8 7 \pm 0 . 1 7 0$ </td><td> $0 . 7 9 0 \pm 0 . 0 8 1$ </td><td> $0 . 9 0 0 \pm 0 . 0 8 2$ </td></tr></table>

![](images/10f68aae6158ad38751f81584be5bffa1860defd68463d8173e56e3dccbf2ed7.jpg)

![](images/fb0cc788bd45d798628645d6a60b6751da2bba33cad063ac88e66335269c7cbd.jpg)  
FIG. 11. Absolute background rejection at $\varepsilon _ { S } = 0 . 5$ and 0.3 for all four stitching conditions on quark/gluon tagging, primarybranch run, versus depth. Horizontal lines mark each network’s own standalone value. Compare to Fig. 2 for top tagging.

TABLE XI. Geometric alignment between $\mathrm { P a r T } \mathrm { s }$ and Lund-ParT’s class tokens, quark/gluon tagging.
<table><tr><td></td><td>CKA</td><td>SVCCA</td></tr><tr><td>primary branch, d=1</td><td> $\overline { { 0 . 6 3 9 \pm 0 . 1 1 8 } }$ </td><td> $0 . 7 6 9 \pm 0 . 0 9 1$ </td></tr><tr><td>primary branch, d=6</td><td> $0 . 8 8 4 \pm 0 . 0 0 8$ </td><td> $0 . 9 2 5 \pm 0 . 0 3 4$ </td></tr><tr><td>full tree, d=1</td><td> $0 . 3 7 9 \pm 0 . 1 1 9$ </td><td> $0 . 6 7 0 \pm 0 . 1 1 1$ </td></tr><tr><td>full tree, d=6</td><td> $0 . 9 1 9 \pm 0 . 0 0 5$ </td><td> $0 . 9 4 8 \pm 0 . 0 2 6$ </td></tr></table>

![](images/8a61aad573f483a21dd5c5f066fb6a3f23460435331d33a19d5d862d1d3c4530.jpg)  
FIG. 12. Linear CKA and SVCCA mean canonical correlation between ParT’s and Lund-ParT’s class tokens versus depth, quark/gluon tagging. Compare to Fig. 3 for top tagging.

TABLE XII. Leakage relative to standalone ParT’s rejected set, quark/gluon tagging.
<table><tr><td>representation</td><td></td><td>leaked</td><td>% of rejected set newly rejected</td></tr><tr><td>primary branch 1</td><td> $\overline { { 2 6 0 7 \pm 5 1 } }$ </td><td>2.7%</td><td>643</td></tr><tr><td>primary branch 5</td><td> $1 3 0 6 \pm 3 6$ </td><td>1.3%</td><td>486</td></tr><tr><td>full tree</td><td>1  $5 5 3 9 \pm 7 4$ </td><td>5.7%</td><td>655</td></tr><tr><td>full tree</td><td>5  $8 9 7 \pm 3 0$ </td><td>0.9%</td><td>400</td></tr></table>

TABLE XIII. Standalone rejection at $\varepsilon _ { S } = 0 . 5$ as a fraction of ParT’s, on the two datasets.
<table><tr><td colspan="4">primary/ParT full tree/ParT gap closed by tree</td></tr><tr><td>quark/gluon</td><td>77.6%</td><td>85.0%</td><td>33%</td></tr><tr><td>top tagging</td><td>46.9%</td><td>85.2%</td><td>72%</td></tr></table>

lund → part stitch, $d = 1 \colon$ background jets responsible for the rej@0.5 gap

![](images/8afd1fdccabdd1c368ed0d438e6d865e6576d1ff1458c8abe661aea5d55a5532.jpg)

![](images/832007c9b0e1efdf513f876c68e3bda56778e8437cdce105382ffec992b85810.jpg)  
FIG. 13. Jet mass and $p _ { T }$ for all gluon jets, the subset standalone ParT rejects at its own $\varepsilon _ { S } = 0 . 5$ threshold, and the subset the lund-to-part stitch at d = 1 fails to reject that standalone ParT would have.

lund → part stitch, d = 5: background jets responsible for the rej@0.5 gap  
![](images/0101e643a85d9e177ebb3e86d78366c8c96a85bf14a19802cab4023aa98c7dde.jpg)

![](images/9aaf011dd35fd4ad6cafe09e716c5b5e50c9585b3a0a29e57a33d6632b1387e3.jpg)  
FIG. 14. As Fig. 13, at $d = 5 ,$

![](images/01aac4825093b8079cb71c5a61794b653f373f4b77e9bda275febd85de7eb1ed.jpg)

![](images/1d5b394ba5b85981995e3a1a830c118d93a7b7df9e43fc7f45d13836388950c3.jpg)  
FIG. 15. Lund-to-part absolute rejection on quark/gluon tagging, primary branch versus full declustering tree, against the same ParT target throughout. The horizontal line marks ParT’s own standalone value. Compare to Fig. 5 for top tagging.

## lund → part retained background rejection

![](images/f9692c5888420921ab066f633467d7f709bd018ef8559a0924575c4af201f7c0.jpg)

![](images/cdd358459d4b681e90ccd7c48106ff9104ef1a499827131f2e21a96d91876135.jpg)  
FIG. 16. Primary branch against full declustering tree across the stitch sweep, quark/gluon tagging.

[1] F. A. Dreyer, G. P. Salam, and G. Soyez, JHEP 12 (2018), 064, arXiv:1807.04758 [hep-ph].

[2] F. A. Dreyer and H. Qu, JHEP 03 (2021), 052, arXiv:2012.08526 [hep-ph].

[3] H. Qu, C. Li, and S. Qian, in Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 162, edited by K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (PMLR, 2022) pp. 18281–18292, arXiv:2202.03772 [hep-ph].

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, in Advances in Neural Information Processing Systems, Vol. 30, edited by I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Curran Associates, Inc., 2017) arXiv:1706.03762 [cs.CL].

[5] H. Qu and L. Gouskos, Phys. Rev. D 101, 056019 (2020), arXiv:1902.08570 [hep-ph].

[6] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, in International conference on machine learning (PMlR, 2019) pp. 3519–3529.

[7] M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, Advances in neural information processing systems 30 (2017).

[8] Y. Bansal, P. Nakkiran, and B. Barak, Advances in neural information processing systems 34, 225 (2021).

[9] F. A. Dreyer, R. Grabarczyk, and P. F. Monni, Eur. Phys. J. C 82, 564 (2022), arXiv:2203.06210 [hep-ph].

[10] S. Rai and S. Ganguly, (2026), arXiv:2605.09881 [hepph].

[11] P. D. Patel and S. Ganguly, (2026), arXiv:2604.25885 [hep-ph].

[12] A. Butter et al., SciPost Phys. 7, 014 (2019), arXiv:1902.09914 [hep-ph].

[13] H. Touvron, M. Cord, A. Sablayrolles, G. Synnaeve, and H. J´egou, in 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021) pp. 32–42, arXiv:2103.17239 [cs.CV].

[14] Y. L. Dokshitzer, G. D. Leder, S. Moretti, and B. R. Webber, JHEP 08 (1997), 001, arXiv:hep-ph/9707323.

[15] M. Wobisch and T. Wengler, in Workshop on Monte Carlo Generators for HERA Physics (Plenary Starting Meeting) (1998) pp. 270–279, arXiv:hep-ph/9907280.

[16] M. Cacciari, G. P. Salam, and G. Soyez, Eur. Phys. J. C 72, 1896 (2012), arXiv:1111.6097 [hep-ph].

[17] I. Loshchilov and F. Hutter, in 7th International Conference on Learning Representations, ICLR 2019 (2019) arXiv:1711.05101 [cs.LG].

[18] V. Papyan, X. Y. Han, and D. L. Donoho, Proc. Natl. Acad. Sci. 117, 24652 (2020), arXiv:2008.08186 [cs.LG].

[19] P. T. Komiske, E. M. Metodiev, and J. Thaler, JHEP 01 (2019), 121, arXiv:1810.05165 [hep-ph].