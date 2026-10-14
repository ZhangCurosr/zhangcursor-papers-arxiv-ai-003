# When Do Learned Blocks Preserve Causal Meaning? Block Identifiability and Constructive Causal Abstractions

Karim Zaghw<sup>1,2,3</sup> Wieland Brendel<sup>2,3</sup> Patrik Reizinger<sup>2,3</sup>

<sup>1</sup>University of T¨ubingen <sup>2</sup>ELLIS Institute T¨ubingen

<sup>3</sup>Max Planck Institute for Intelligent Systems, T¨ubingen, Germany

## Abstract

Standard identifiability theory assumes onedimensional latent factors, but learned representations can encode latent information across multidimensional blocks whose internal coordinates are not uniquely determined. When such blocks are used to represent and intervene on higher-level concepts, it is unclear whether their causal meaning survives the transformations left unresolved by block-identifiability. We connect blockidentifiability to constructive causal abstraction. We show that block-diagonal invertible transformations followed by compatible coordinate permutations form the maximal class that preserves every constructive causal abstraction built on those blocks. This guarantee assumes that the correspondence between blocks and concepts is updated with the representation. A small synthetic experiment with additive autoencoders shows that high-level interventions can be transferred between independently learned representations while approximately preserving their intended efects.

## 1 INTRODUCTION

Identifiability theory asks whether a learned representation recovers the latent factors that generated the observations (Hyvarinen et al., 2019; Khemakhem et al., 2020). Many standard results assume that the learned representation has the same dimensionality as the latent space, with one learned coordinate corresponding to each latent factor. Under the assumptions of these results, the factors can often be recovered up to a permutation and a separate invertible coordinatewise transformation. Because these transformations do not mix the factors, additional assumptions can still allow their causal relations to be recovered (Shimizu et al., 2006; Reizinger et al., 2023).

In practice, a concept may be encoded across a block of learned coordinates rather than a single coordinate. Several settings motivate this block-level view. Sparse autoencoders (SAEs) can split one concept across multiple learned features (Bricken et al., 2023; Mueller et al., 2025), while block-sparse featurizers explicitly learn groups of activation directions and recover visual concepts that are typically two- to fourdimensional (Fel et al., 2026). Object-centric models represent objects using vector-valued slots (Locatello et al., 2020). Formal block-identifiability results arise in independent subspace analysis (ISA), which recovers independent subspaces up to permutation and invertible within-subspace transformations (Hyv¨arinen and Hoyer, 2000; Theis, 2006), and in additive decoders, which identify latent blocks up to permutation and block-wise invertible transformations under their assumptions (Lachapelle et al., 2023). These results recover each block as a whole while leaving its internal coordinates undetermined. Diferent learned models may therefore represent the same block using diferent internal coordinates. While practitioners intervene on such blocks by steering feature groups, manipulating object slots, or clamping latent subspaces, block-identifiability does not tell us whether these interventions keep the same high-level causal meaning across diferent representations.

We use constructive causal abstractions (Beckers and Halpern, 2019; Geiger et al., 2025) to state this question precisely. We treat the coordinates of the learned representation as variables in a causal model, which we call the low-level model. The human-interpretable concepts form another causal model, which we call the high-level model. A constructive causal abstraction maps states and interventions in the low-level model to corresponding states and interventions in the highlevel model, with each high-level concept computed from one block of learned coordinates. We ask which changes to the learned representation preserve this

mapping.

Our contributions are:

1. We identify the largest class of transformations guaranteed to preserve causal abstractions built from learned blocks. This class consists of invertible transformations within blocks followed by coordinate permutations that carry the block-to-concept correspondence with them. It includes the transformations allowed by blockidentifiability guarantees for ISA and additive decoders. We prove that every transformation in this class preserves every abstraction of this form, while every transformation outside it fails to preserve at least one (Thm. 1).

2. We illustrate high-level steering and its transfer between learned representations. In a small synthetic experiment with independently trained additive autoencoders, we translate high level interventions into low-level block edits. We transfer these edits between models using fitted blockwise maps and show that their intended highlevel efects are approximately preserved (§ 4).

## 2 BACKGROUND

Notation. Following Geiger et al. (2025), capital let ters denote variables and lowercase letters their values, and boldface denotes a set of variables or a joint setting of one, so that $x \in \mathrm { V a l } _ { X }$ and $\mathbf { x } \in \mathrm { V a l } _ { \mathbf { V } _ { L } }$ . Let $\mathbf { S } = \{ S _ { 1 } , \ldots , S _ { m } \}$ be the ground-truth causal factors: each $S _ { i }$ is produced by a mechanism $F _ { S _ { i } }$ from its parents $\operatorname { P a } ( S _ { i } )$ in the ground-truth directed acyclic graph (DAG) and an exogenous noise variable $N _ { i }$ , written $S _ { i } : = F _ { S _ { i } } ( \mathrm { P a } ( S _ { i } ) , N _ { i } )$ . The map f takes a factor setting s to the activation vector $\mathbf { o } = f ( \mathbf { s } )$ supplied to the featurizer $^ { g ; }$ this includes generating the network input and passing it through the network. The featurizer returns the learned feature vector, or representation, $\mathbf { x } = g ( \mathbf { o } )$ . We model the coordinates of x as variables of the representation-level SCM L. They form the finite set $\mathbf { V } _ { L } ,$ so a learned feature is a variable $X \in \mathbf { V } _ { L }$ with value set Val<sub>X</sub>, and $\mathbf { x } \in \mathrm { V a l } _ { \mathbf { V } _ { L } }$ is a total setting of L. The high-level SCM H has variables ${ \bf V } _ { H } = \{ \bar { Z } ^ { ( 1 ) } , \dots , Z ^ { ( B ) } \}$ , where each $Z ^ { ( b ) }$ is a humaninterpretable concept chosen for intervention and may be computed from a block of learned features rather than from one coordinate. The ground-truth factors S generate the activation vector $\mathbf { o } ,$ whereas $\mathbf { V } _ { H }$ contains the concepts abstracted from the learned representation; we do not assume that they coincide. For a subset $\mathbf { U } \subseteq \mathbf { V } _ { L }$ , write ${ \mathrm { V a l } } _ { \mathbf { U } } : = \prod _ { U \in \mathbf { U } } { \mathrm { V a l } } _ { U } .$ , and the projection ${ \mathrm { P r o j } } _ { \mathbf { U } } : { \mathrm { V a l } } _ { \mathbf { V } _ { L } } \to { \mathrm { V a l } } _ { \mathbf { U } }$ restricts a setting to U. Tab. 2 in $\ S \mathrm { ~ A ~ }$ summarizes this notation. Fig. 1 illustrates this conceptual pipeline.

To compare interventions across levels, we first need a common description of causal models, their solutions, and the interventions applied to them.

Definition 1 (SCMs and interventions). A deterministic SCM is a pair $M = ( \Sigma , \{ F _ { X } \} _ { X \in \mathbf { V } } )$ , where $\boldsymbol { \Sigma } = ( \mathbf { V } , \mathrm { V a l } )$ is a signature and $F _ { X } : \mathrm { V a l } _ { \mathbf { V } }  \mathrm { V a l } _ { X }$ is the mechanism for each $X \in { \bf V } . \ \mathrm { ~ A ~ }$ setting v ∈ Val<sub>V</sub> is a solution of M if $F _ { X } ( \mathbf { v } ) = \mathrm { P r o j } _ { \{ X \} } ( \mathbf { v } )$ for every $X \in \mathbf { V } ; \mathrm { S o l v e } ( M )$ denotes the set of all such solutions. A hard intervention on $\mathbf { U } \subseteq \mathbf { V }$ replaces $\{ F _ { X } \} _ { X \in \mathbf { U } }$ by constants; we identify it with the idempotent map on Val that overwrites the coordinates in U, and the empty intervention with the identity map. A set Ψ of interventionals forms an intervention algebra when its composition algebra is isomorphic to that of hard interventions on some signature (Geiger et al., 2025). Hard interventions are partially ordered by extension: ${ \mathit { I } } \subseteq { \mathit { I } } ^ { \prime }$ when $I ^ { \prime }$ sets every variable I sets, to the same value.

To say that L and H make consistent causal predictions, mapping an intervention from L to H should give the same high-level outcomes as mapping the resulting low-level solutions to H. Exact transformations formalize this requirement (Rubenstein et al., 2017; Geiger et al., 2025).

Definition 2 (Exact transformation). An exact transformation $( \tau , \omega ) : L  H$ consists of a surjective state map $\tau : \mathrm { V a l } _ { \mathbf { V } _ { L } } \to \mathrm { V a l } _ { \mathbf { V } _ { H } }$ and a surjective order-preserving intervention map ω such that for all $I \in \mathrm { d o m } ( \omega )$

$$
\tau \big ( \mathrm { S o l v e } ( L ^ { I } ) \big ) = \mathrm { S o l v e } \big ( H ^ { \omega ( I ) } \big )\tag{1}
$$

To ask whether a causal abstraction survives a change of representation, we must express the low-level states, mechanisms, and interventions in the new coordinates. Bijective translations formalize this consistent repa rameterization (Geiger et al., 2025).

Definition 3 (Bijective translation). Let M be a deterministic SCM, and write $F _ { \mathbf { V } } ( \mathbf { v } ) : = ( F _ { X } ( \mathbf { v } ) ) _ { X \in \mathbf { V } }$ for the map collecting all of its mechanisms. For a bijection $T : { \mathrm { V a l } } _ { \mathbf { V } }  { \mathrm { V a l } } _ { \mathbf { V } }$ , the bijective translation $M _ { T }$ has mechanism map

$$
F _ { \mathbf { V } } ^ { T } : = T \circ F _ { \mathbf { V } } \circ T ^ { - 1 } .\tag{2}
$$

An intervention I is transported to

$$
I ^ { T } : = T \circ I \circ T ^ { - 1 } .\tag{3}
$$

For any intervention algebra Ψ on $M$ , the transported interventions $\Psi ^ { T } : = \{ I ^ { \breve { T } } : I \in \Psi \}$ form an intervention algebra on $M _ { T }$ , ordered by $I ^ { T } \subseteq J ^ { T }$ exactly when

![](images/dc75b31c872592ea74766cab1e0af8e3d2a640867f8d8d654034f99076cd068f.jpg)  
Figure 1: From ground-truth factors to a constructive causal abstraction of a learned representation. The ground-truth causal factors S take values s, which determine the network input and the resulting activation vector $\mathbf { o } = f ( \mathbf { s } )$ A featurizer $^ { g , }$ such as a sparse autoencoder (SAE), maps these activations to the learned representation ${ \bf x } = g ( { \bf o } )$ We treat the coordinates of x as variables of the low-level structural causal mode (SCM) L and partition them into blocks $\mathbf { V } ^ { ( b ) }$ . The constructive causal abstraction $( \tau , \omega )$ maps low-level states and complete-block interventions to states and interventions in the high-level SCM H, where each humaninterpretable concept $Z ^ { ( b ) }$ is computed from its corresponding block. Blue and pink indicate this block-to-concept correspondence. The ground-truth factors S generate the activations but need not coincide with the concepts in $H .$

$I \subseteq J .$ Every intervention I on M is therefore paired with the transported intervention $I ^ { T }$ on $M _ { T }$ . Under this pairing,

$$
T \big ( \mathrm { S o l v e } ( M ^ { I } ) \big ) = \mathrm { S o l v e } ( M _ { T } ^ { I ^ { T } } ) ,
$$

so the state map T and intervention map $I \mapsto I ^ { T }$ form an exact transformation from M to $M _ { T }$ . Bijective translations compose with other exact transformations (Geiger et al., 2025). This composition allows an abstraction from L to H to be carried over to a reparameterized model $L _ { T }$

Exactness tells us that the low- and high-level models agree under corresponding interventions, but it does not say where each high-level variable is represented in the low-level model. Our question requires this additional structure because we want each concept to be computed from a particular block of learned coordinates and interventions on that block to correspond to interventions on the concept. Constructive causal abstractions formalize this block-to-concept correspondence through an alignment (Beckers and Halpern, 2019).

Definition 4 (Alignment and constructive causal ab straction). An alignment between the low-level variables $\mathbf { V } _ { L }$ and the high-level variables $\mathbf { V } _ { H }$ is a pair $\langle \Pi , \pi \rangle$ , where $\Pi = \{ \hat { \Pi _ { Z } } : Z \in \mathbf { V } _ { H } \} \cup \{ \Pi _ { \perp } \}$ is a partition of $\mathbf { V } _ { L }$ , and $\pi = \{ \pi _ { Z } \} _ { Z \in \mathbf { V } _ { H } }$ is a family of surjective cell maps $\pi _ { Z } : { \mathrm { V a l } } _ { \Pi _ { Z } } \to { \mathrm { V a l } } _ { Z }$ . The cell $\Pi _ { Z }$ contains the low-level variables associated with Z, while the optional cell $\Pi _ { \perp }$ contains variables that are not mapped

to any high-level variable.

The alignment $\langle \Pi , \pi \rangle$ induces the state map

$$
\tau _ { \pi } ( \mathbf { x } ) : = \left( \pi _ { Z } ( \mathrm { P r o j } _ { \Pi _ { Z } } ( \mathbf { x } ) ) \right) _ { Z \in \mathbf { V } _ { H } }
$$

and maps interventions on complete cells according to

$$
\omega _ { \pi } \big ( \mathrm { d o } ( \Pi _ { Z } = \mathbf { i } _ { Z } , ~ Z \in A ) \big ) : = \mathrm { d o } \big ( Z = \pi _ { Z } ( \mathbf { i } _ { Z } ) , ~ Z \in A \big ) ,
$$

where ${ \cal A } \subseteq { \bf V } _ { \cal H }$ and $\mathbf { i } _ { Z } \in \mathrm { V a l } _ { \Pi _ { Z } }$ . The high-level model H is a constructive causal abstraction of L under this alignment if $( \tau _ { \pi } , \omega _ { \pi } ) : L  H$ is an exact transformation.

For a cell map $\pi _ { Z } .$ the fiber over $z \in \mathrm { V a l } _ { Z }$ is $\pi _ { Z } ^ { - 1 } ( z )$ the set of low-level cell values mapped to z. If π<sub>Z</sub> is many-to-one, values in the same fiber correspond to the same high-level value and therefore cannot be distinguished in H.

## 3 PRESERVING CONSTRUCTIVE CAUSAL ABSTRACTIONS

We show that constructive causal abstractions are invariant under block-diagonal reparameterizations and that no larger class of reparameterizations preserves all of them, connecting block-identifiability to causal abstraction theory.

## 3.1 Setup

Our aim is to determine whether the ambiguity left by block-identifiability changes the causal meaning assigned to learned blocks. We therefore assume that each learned block serves as the alignment cell for one high-level variable and consider interventions on complete blocks.

Assumption 1 (Block-aligned abstraction).

(i) $L = ( \Sigma _ { L } , \{ F _ { X } \} _ { X \in { \bf V } _ { L } } )$ is a deterministic SCM with block partition

$$
\mathbf { V } _ { L } = \bigcup _ { b = 1 } ^ { B } \mathbf { V } ^ { ( b ) } .\tag{4}
$$

(ii) $( \tau , \omega ) : L  H$ is constructive, with each block $\dot { \mathbf { V } } ^ { ( b ) }$ serving as the alignment cell for $Z ^ { ( b ) }$ . We write $\pi _ { b } : \operatorname { V a l } _ { \mathbf { V } ^ { ( b ) } } \to \operatorname { V a l } _ { Z ^ { ( b ) } }$ for the corresponding cell map, for $b = 1 , \dots , B$

(iii) Ψ is the intervention algebra of all hard interventions on zero or more complete blocks, $\mathrm { d o m } ( \omega ) = \Psi$ , and $\omega$ is the intervention map induced by the alignment as in Defn. 4.

Block-identifiability recovers each block as a whole but leaves its individual coordinates undetermined. We therefore focus on interventions on complete blocks because the corresponding high-level concept can de pend on every coordinate in the block. To test whether changes to the internal coordinates afect the block’s causal meaning, we collect all invertible within-block coordinate changes in the family D.

Definition 5 (Block reparameterizations). The group of block-diagonal reparameterizations is

$$
\begin{array} { r l } & { \mathcal { D } : = \big \{ \mathrm { D i a g } ( D ^ { ( 1 ) } , \dots , D ^ { ( B ) } ) \big \vert } \\ & { \qquad D ^ { ( b ) } : \mathrm { V a l } _ { \mathbf { V } ^ { ( b ) } } \to \mathrm { V a l } _ { \mathbf { V } ^ { ( b ) } } , } \\ & { \qquad D ^ { ( b ) } \mathrm { i s ~ b i j e c t i v e ~ f o r ~ } b = 1 , \dots , B \big \} . } \end{array}\tag{5}
$$

Block-identifiability may also permute whole blocks; we incorporate those permutations below. To characterize the largest class of transformations guaranteed to preserve every abstraction of this form, Thm. 1 compares these structured transformations with arbitrary bijections of the low-level state space.

To test whether a change of representation preserves the abstraction, we must compare the same high-level intervention before and after the change. Under a bijection $T ,$ a state y in $L _ { T }$ corresponds to $T ^ { - 1 } ( \mathbf { y } )$ in $L ,$ and a low-level intervention I becomes $I ^ { T }$ . We therefore use $\tau \circ T ^ { - 1 }$ for states and define $\omega ^ { T }$ so that $I ^ { T }$ has the same high-level meaning as I.

Definition 6 (Transported high-level intervention map). For every $I \in \Psi$ , define

$$
\omega ^ { T } ( I ^ { T } ) : = \omega ( I ) .\tag{6}
$$

Because $I \mapsto I ^ { T }$ is an order-preserving bijection, $\omega ^ { T }$ is well-defined, surjective, and order-preserving. By exactness of bijective translations and composition of exact transformations (Geiger et al., 2025), $( \tau \circ T ^ { - 1 } , \omega ^ { T } ) : L _ { T }  H$ is exact. Since every $D \in { \mathcal { D } }$ is a bijection, this includes all within-block reparameterizations.

## 3.2 The Maximal Class Preserving Constructiveness

The family D changes coordinates within blocks, but block-identifiability may also permute whole blocks. More generally, a coordinate permutation can move the coordinates of a block to new positions. To keep each concept associated with the same group of coordinates after this change, we track where each block moves under the permutation.

Definition 7 (Compatible permutations and transported cells). A permutation σ of the low-level variables $\mathbf { V } _ { L }$ is compatible if $\operatorname { V a l } _ { \sigma ( X ) } ~ = ~ \operatorname { V a l } _ { X }$ for every $X \in \mathrm { ~ \bf ~ V ~ } _ { L } .$ It acts on total settings according to $\left( \sigma ( \mathbf { x } ) \right) _ { \sigma ( X ) } : = x _ { X }$ . For each block $\mathbf { V } ^ { ( b ) }$ , its image $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ is the corresponding transported cell, and $\sigma _ { b }$ denotes the induced bijection from ${ \mathrm { V a l } } _ { \mathbf { V } ^ { ( b ) } }$ to ${ \mathrm { V a l } } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) }$

Because σ is a permutation and the original blocks partition $\mathbf { V } _ { L } .$ , the transported cells also partition $\mathbf { V } _ { L } ,$ although they may be interleaved with respect to the original partition. We now combine the two changes of representation: D changes coordinates within each block, and σ moves those coordinates to their transported cells. The family T contains all transformations of this form.

Definition 8 (Transformation family T). The family of block-diagonal reparameterizations followed by compatible coordinate permutations is

$$
\mathcal T : = \{ \sigma \circ D \mid \sigma { \mathrm { ~ i s ~ c o m p a t i b l e ~ a n d ~ } } D \in  \mathcal D \} .
$$

Any bijection preserves exactness when the model and its interventions are transported as in Defn. 3. Our goal is stronger: after changing the representation, each high-level variable should still be computed from one cell, and setting that cell should still be a hard intervention. Both properties hold when $T = \sigma \circ D$ If ${ \cal I } = \mathrm { d o } ( { \bf V } ^ { ( b ) } { = } { \bf i } _ { b } )$ for some block value $\mathbf { i } _ { b } \in \mathrm { V a l } _ { \mathbf { V } ^ { ( b ) } }$ then $I ^ { T }$ sets the transported cell $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ to the fixed value $\sigma _ { b } \big ( D ^ { ( b ) } ( \mathbf { i } _ { b } ) \big )$ ). For a general bijection, either property can fail: the transported intervention may depend on the current state, or a high-level variable may depend on coordinates from several cells.

A two-variable XOR transformation illustrates both failures. Let $\textbf { V } = \{ X _ { 1 } , X _ { 2 } \}$ with binary singleton blocks $\mathbf { V } ^ { ( 1 ) } ~ = ~ \left\{ X _ { 1 } \right\}$ and $\mathbf { V } ^ { ( 2 ) } ~ = ~ \left\{ X _ { 2 } \right\}$ , and let $\tau ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } , x _ { 2 } )$ map them to $Z ^ { ( 1 ) }$ and $\overline { { Z ^ { ( 2 ) } } }$ . Consider

$$
\begin{array} { c } { { T ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } \oplus x _ { 2 } , x _ { 2 } ) , } } \\ { { { } } } \\ { { T ^ { - 1 } ( y _ { 1 } , y _ { 2 } ) = ( y _ { 1 } \oplus y _ { 2 } , y _ { 2 } ) . } } \end{array}
$$

For the hard intervention $I = \mathrm { d o } ( X _ { 1 } { = } 0 )$ , the transported operation is $I ^ { T } ( y _ { 1 } , y _ { 2 } ) = ( \dot { y } _ { 2 } , y _ { 2 } )$ which copies the current value of $Y _ { 2 }$ into $Y _ { 1 }$ rather than setting $Y _ { 1 }$ to a constant. It is therefore not a hard intervention in the new coordinates. The transported state map also satisfies

$$
( \tau \circ T ^ { - 1 } ) ( y _ { 1 } , y _ { 2 } ) = ( y _ { 1 } \oplus y _ { 2 } , y _ { 2 } ) .
$$

Thus $Z ^ { ( 1 ) }$ requires both $Y _ { 1 }$ and $Y _ { 2 } ,$ , while $Z ^ { ( 2 ) }$ also requires $Y _ { 2 } .$ , so the two high-level variables cannot be computed from disjoint cells. Exactness survives relative to the transported intervention algebra, but constructiveness does not.

The XOR example shows that preserving exactness does not necessarily preserve the block-to-concept structure. We now ask which transformations preserve this structure for every constructive abstraction built on the fixed blocks. For each target SCM $H ,$ let ${ \cal { A } } ( L , H )$ be the set of constructive abstractions $( \tau , \omega ) : L  H$ whose cell for $Z ^ { ( b ) }$ is $\mathbf { V } ^ { ( b ) }$ for every $b ,$ with dom $( \omega ) = \Psi$ and $\omega$ the intervention map induced by that alignment (Assum. 1).

Theorem 1 (Universal maximality of block-diagonal invariance). For the low-level SCM L and block partition in Assum. $^ { 1 , }$ fix any bijection $T : { \mathrm { V a l } } _ { \mathbf { V } _ { L } } \to { \mathrm { V a l } } _ { \mathbf { V } _ { L } }$ and any compatible permutation σ. The following are equivalent:

(a) For every target SCM H and every $( \tau , \omega ) ~ \in$ ${ \cal { A } } ( L , H )$ , the transported pair $( \tau \circ T ^ { - 1 } , \omega ^ { \dot { T } } ) :$ $L _ { T } \ \to \ H$ is constructive with cells $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ for every b.

(b) $T = \sigma \circ D$ for some $D \in \mathcal { D }$

The proof is given in $\ S \mathrm { B } . 1$ . The theorem identifies the largest family of transformations for which constructiveness is guaranteed, connecting the result directly to block-identifiability. In block-identifiability results, the permitted permutations are whole-block permutations, meaning that each original block is mapped onto another original block. Together with invertible transformations within blocks, these transformations are a special case of $\tau { } _ { \cdot }$ . The family $\tau$ is broader because it also permits compatible coordinate permutations that move a block to an interleaved transported cell.

There is one important qualification: the guarantee is universal. Every transformation in $\tau$ preserves every constructive abstraction built on the fixed blocks.

A transformation outside $\tau$ must break at least one such abstraction, but it may still preserve a particular one. For example, take four binary variables with blocks $\{ X _ { 1 } , X _ { 2 } \}$ and $\{ X _ { 3 } , X _ { 4 } \}$ , and let $\tau ( \mathbf { x } ) =$ $( x _ { 1 } \oplus x _ { 2 } , x _ { 3 } \oplus x _ { 4 } )$ . Consider the transformation

$$
T _ { + } ( \mathbf { x } ) = { \big ( } x _ { 1 } \oplus x _ { 3 } , x _ { 2 } \oplus x _ { 3 } , x _ { 3 } , x _ { 4 } { \big ) } ,
$$

which mixes the two blocks. Nevertheless,

$$
( \tau \circ T _ { + } ^ { - 1 } ) ( \mathbf { y } ) = \big ( y _ { 1 } \oplus y _ { 2 } , y _ { 3 } \oplus y _ { 4 } \big ) .
$$

The first high-level variable is unchanged because the added value x<sub>3</sub> appears twice and therefore disappears under XOR: $( x _ { 1 } \oplus x _ { 3 } ) \oplus ( x _ { 2 } \oplus x _ { 3 } ) = x _ { 1 } \oplus x _ { 2 }$ . Thus $T _ { + } \notin \mathcal { T }$ still preserves this particular abstraction.

This does not contradict Thm. 1, which concerns preservation of every abstraction built on the fixed blocks. Its proof shows that a transformation outside $\tau$ fails for the identity block abstraction, whose cell maps retain the complete value of each block. Preserving one particular abstraction therefore does not imply that $T \in { \mathcal { T } }$

Implication for block-identifiable representations. Under Assum. 1, the transformations left unresolved by block-identifiability do not change the predicted high-level efect of an intervention. After transporting the alignment and interventions, whole-block permutations and invertible within-block reparameterizations yield the same solution set in H for every corresponding intervention. Thm. 1 identifies the boundary of this universal guarantee: every $T \in \mathcal { T }$ preserves every block-aligned constructive abstraction, whereas each $T \notin \mathcal { T }$ breaks at least one, although a particular abstraction may survive. Thus complete blocks, rather than their non-identifiable internal coordinates, are the stable units of intervention. When $\pi _ { b }$ is many-to-one, the high-level intervention $\mathrm { d o } ( Z ^ { ( b ) } { = } z )$ corresponds to every complete-block intervention $\mathrm { d o } ( { \mathbf { V } } ^ { ( b ) } { = } \bar { \mathbf { i } } _ { b } )$ with $\mathbf { i } _ { b } \in \pi _ { b } ^ { - 1 } ( z )$ . Two abstractions whose cell maps have the same fibers in every cell difer only by a relabeling of their high-level values (Corollary 1). The abstraction does not distinguish these low-level interventions, so implementing a high-level intervention requires an additional selection rule. This ambiguity becomes relevant in § 4, where a nearest-point rule selects one representative from the fiber.

## 4 EXPERIMENTS

Goal. We test whether a high-level intervention in the human-interpretable model H can be translated into a block intervention in a reference low-level model $L _ { 0 }$ and transported to independently trained low-level models $L _ { r }$ while preserving its high-level efect. Fig. 2 summarizes this steering, transport, and evaluation procedure.

![](images/b99d9446ad22d6e1ee080205d391a37245a9eb3a3660f7ef9716e209176c42a7.jpg)  
Figure 2: Concept-directed steering and transport between independently trained additive autoencoders. Starting from a base observation, the high-level intervention $J ^ { \star } = \mathrm { d o } ( Z ^ { ( 1 ) } { = } s _ { 1 } ^ { \star } )$ in H is implemented in the reference model $L _ { 0 }$ by editing block 1. ${ \mathrm { H e r e } } , \pi _ { 1 }$ maps that block to $Z ^ { ( 1 ) }$ , and $\mathbf { x } _ { 0 } ^ { ( 1 ) \star }$ is the closest block value satisfying $\pi _ { 1 } ( \mathbf { x } _ { 0 } ^ { ( 1 ) \star } ) = s _ { 1 } ^ { \star }$ ; block 2 remains unchanged. The fitted transformation $\widehat { T } _ { r } = \widehat { \sigma } _ { r } \circ \widehat { D } _ { r }$ transfers the edit to recipient model $L _ { r }$ , where $\widehat { D } _ { r }$ maps corresponding blocks and $\widehat { \sigma } _ { r }$ accounts for their ordering. The decoders $G _ { 0 }$ and $G _ { r }$ produce the edited outputs $\widehat { \mathbf { o } } _ { 0 } ^ { \star }$ and $\widehat { \mathbf { o } } _ { r } ^ { \star }$ . For evaluation, the exact factor setting ${ \bf s } ^ { \mathrm { g e n } } = ( s _ { 1 } ^ { \star } , u _ { 1 } , s _ { 2 } , u _ { 2 } )$ changes only $s _ { 1 } .$ setting it to the intervention value $s _ { 1 } ^ { \star }$ , and the data-generating map $f$ produces the comparison observation $\mathbf { o } ^ { \mathrm { g e n } } = f ( \mathbf { s } ^ { \mathrm { g e n } } )$ . Both decoded outputs are compared with this same observation, as indicated by the dotted lines. Green marks the intervention, while blue and pink identify the two blocks throughout.

Setting. Synthetic factors $( s _ { 1 } , u _ { 1 } , s _ { 2 } , u _ { 2 } )$ generate observations through the nonlinear additive map $\begin{array} { l l l } { \mathbf { o } } & { = } & { A _ { 1 } \phi ( s _ { 1 } , u _ { 1 } ) \ + \ A _ { 2 } \phi ( s _ { 2 } , u _ { 2 } ) } \end{array}$ where $\begin{array} { r l } { \phi ( s , u ) } & { { } = } \end{array}$ $( s , u , s ^ { 2 } , s u , u ^ { 2 } )$ ; the distribution and mixing matrix are specified in $\ S \mathrm { ~ C ~ }$ . Following additive-decoder models (Lachapelle et al., 2023), model r has an encoder $E _ { r } ( \mathbf { o } ) ~ = ~ \mathbf { x } _ { r }$ and a decoder $G _ { r } ( \mathbf { x } _ { r } ) = \widehat { \mathbf { o } } _ { r }$ The four-dimensional representation $\mathbf { x } _ { r }$ has two twodimensional blocks $\mathbf { x } _ { r } ^ { ( 1 ) } , \mathbf { x } _ { r } ^ { ( 2 ) }$ . Each block is passed through its own nonlinear decoder branch, and the two outputs are added:

$$
\widehat { \bf o } _ { r } = G _ { r } ^ { ( 1 ) } ( { \bf x } _ { r } ^ { ( 1 ) } ) + G _ { r } ^ { ( 2 ) } ( { \bf x } _ { r } ^ { ( 2 ) } ) .
$$

We treat $\mathbf { x } _ { r }$ as the state of the low-level SCM $L _ { r }$ and the two learned blocks as its cells $\mathbf { V } ^ { ( 1 ) } , \mathbf { V } ^ { ( 2 ) }$ . The highlevel SCM H keeps only $Z ^ { ( 1 ) } = s _ { 1 }$ and $Z ^ { ( 2 ) } = s _ { 2 } ;$ omitting $u _ { 1 } , u _ { 2 }$ makes each cell map $\pi _ { b }$ many-to-one.

Protocol. We train one reference autoencoder $( E _ { 0 } , G _ { 0 } )$ and ten additional autoencoders $\left( E _ { r } , G _ { r } \right)$ that may receive an intervention transported from the reference model. After training, we freeze every model and retain for transport only those that reconstruct held-out observations with normalized mean squared error (NMSE) at most 0.01. The architecture, training settings, data splits, and definition of NMSE are given in $\ S \mathrm { ~ C ~ }$ , and we provide the code as supplementary material.

We do not initially know which learned block corresponds to $Z ^ { ( 1 ) }$ and which corresponds to $Z ^ { ( 2 ) }$ . We therefore use fresh examples in which each observation o is paired with its known concept values $( Z ^ { ( 1 ) } , Z ^ { ( 2 ) } ) = ( s _ { 1 } , s _ { 2 } )$ , and encode o with $E _ { 0 }$ . Let $\pi _ { b j }$ denote the afine map fitted from learned block $\mathbf { V } ^ { ( b ) }$ to concept $Z ^ { ( j ) }$ . We fit all four maps and compare the two possible one-to-one assignments: $( \pi _ { 1 1 } , \pi _ { 2 2 } )$ and $\left( \pi _ { 1 2 } , \pi _ { 2 1 } \right)$ . On held-out examples, we choose the assignment whose two maps best predict $Z ^ { ( 1 ) }$ and $Z ^ { ( 2 ) }$ <sup>)</sup>. After indexing the blocks by their selected concepts, these maps become the cell maps $\pi _ { 1 }$ and $\pi _ { 2 }$ , defining the empirical reference abstraction

$$
\tau ( \mathbf { x } _ { 0 } ) : = \big ( \pi _ { 1 } ( \mathbf { x } _ { 0 } ^ { ( 1 ) } ) , \pi _ { 2 } ( \mathbf { x } _ { 0 } ^ { ( 2 ) } ) \big ) ,\tag{7}
$$

which approximates $( s _ { 1 } , s _ { 2 } )$ . The nuisance variables $u _ { 1 } , u _ { 2 }$ are not used to fit these maps.

We estimate $T _ { r } = \sigma _ { r } \circ D _ { \prime }$ by encoding the same observations with $E _ { 0 }$ and $E _ { r } { _ { \mathrm { : } } }$ , giving paired codes $\mathbf { x } _ { 0 } =$ $E _ { 0 } ( \mathbf { o } )$ and ${ \bf x } _ { r } = E _ { r } ( { \bf o } )$ . We fit afine maps between all pairs of blocks and choose the one-to-one pairing with the lowest held-out code error. This gives both the estimated permutation $\widehat { \sigma } _ { r }$ and the fitted within-block maps $\widehat { D } _ { r } ^ { ( b ) }$ . We then index recipient blocks by their matched reference blocks, so that $\widehat { D } _ { r } ^ { ( b ) } ( \mathbf { x } _ { 0 } ^ { ( b ) } ) \approx \mathbf { \dot { x } } _ { r } ^ { ( b ) }$ . In the original output order, $\widehat { T } _ { r } = \widehat { \sigma } _ { r } \circ \mathrm { D i a g } ( \widehat { D } _ { r } ^ { ( 1 ) } , \widehat { D } _ { r } ^ { ( 2 ) } )$

The purpose of $H$ is to express interventions in the concepts we understand, but the edit must ultimately be applied to $L .$ The high-level intervention do $( Z ^ { ( b ) } { = } s _ { b } ^ { \star } )$ does not have a unique low-level implementation because $\pi _ { b }$ is many-to-one. Given a base observation, we encode $\mathbf { x } _ { 0 } ~ = ~ E _ { 0 } ( \mathbf { o } _ { \mathrm { b a s e } } )$ and keep its other block fixed while choosing $\mathbf { x } _ { 0 } ^ { ( b ) \star } \in \pi _ { b } ^ { - 1 } ( s _ { b } ^ { \star } )$ closest to the current block $\mathbf { x } _ { 0 } ^ { ( b ) }$ . Thus H specifies the fiber, while the nearest-point rule uses information unavailable in H to select one low-level representative. Replacing block b gives the edited reference code $\mathbf { x } _ { 0 } ^ { \star } .$ , which decodes as $\widehat { \mathbf { o } } _ { 0 } ^ { \star } = G _ { 0 } ( \mathbf { x } _ { 0 } ^ { \star } )$ . We encode the same base observation as $\mathbf { x } _ { r } = E _ { r } ( \mathbf { o } _ { \mathrm { b a s e } } )$ and replace its corresponding block by

$$
\begin{array} { r } { \mathbf { x } _ { r } ^ { ( b ) \star } : = \widehat { D } _ { r } ^ { ( b ) } ( \mathbf { x } _ { 0 } ^ { ( b ) \star } ) , } \end{array}\tag{8}
$$

and decode the edited recipient code as $\widehat { \mathbf { o } } _ { r } ^ { \star } = G _ { r } \bigl ( \mathbf { x } _ { r } ^ { \star } \bigr )$ The closed-form afine projection used to choose $\mathbf { x } _ { 0 } ^ { ( b ) \star }$ is given in $\ S \mathrm { ~ C ~ }$

The known map used to generate the synthetic observations is used only to evaluate the edits. For the intervention do $( Z ^ { ( b ) } = s _ { b } ^ { \star } )$ , we construct

$$
\begin{array} { r } { \mathbf { o } ^ { \mathrm { g e n } } = A _ { b } \phi ( s _ { b } ^ { \star } , u _ { b } ) + A _ { 3 - b } \phi ( s _ { 3 - b } , u _ { 3 - b } ) , } \end{array}
$$

where $u _ { b }$ and $\left( s _ { 3 - b } , u _ { 3 - b } \right)$ come from the base observation. Thus only $s _ { b }$ is changed, setting it to $s _ { b } ^ { \star }$ . We compare both the reference output $\widehat { \mathbf { o } } _ { 0 } ^ { \star }$ and every recipient output $\widehat { \mathbf { o } } _ { r } ^ { \star }$ with this same observation. We measure agreement using the NMSE between each decoded output and $\mathbf { o } ^ { \mathrm { g e n } }$ . Because the generator is injective, a low NMSE indicates that the edit sets $s _ { b }$ to $s _ { b } ^ { \star }$ while preserving its nuisance value and the other factor pair. These factor values and the generator are used only for evaluation, not to choose or transport the edit. Full experimental details and additional diagnostics are in $\ S \mathrm { ~ C ~ }$

Results. The reference model and four of the ten additional autoencoders pass the reconstruction threshold; the other six fail clearly and are excluded before steering. The reference cell maps identify their concepts unambiguously, and paired codes identify the corresponding blocks in every retained recipient model. Steering succeeds in the reference model and after transport to all four recipient models: every decoded observation has NMSE at most 0.012 against the common exact generator comparison for the corresponding intervention. Tab. 1 reports the reconstruction and steering NMSE for every retained model.

Table 1: Reconstruction and conceptintervention accuracy across retained additive autoencoders. Rows show one reference additive autoencoder and four independently trained recipient additive autoencoders, identified by training seed. “Recon.” is the normalized mean squared error (NMSE) between each model’s decoded reconstructions and held-out synthetic input observations; only models with reconstruction NMSE at most 0.01 are retained. The $Z ^ { ( b ) }$ columns report intervention error. An intervention on $Z ^ { ( b ) }$ is implemented as a block edit in the reference model and transported to each recipient model. Each decoded edited output is compared with the exact synthetic observation obtained by changing only $s _ { b }$ to the intervention value while keeping $u _ { b }$ and $\left( s _ { 3 - b } , u _ { 3 - b } \right)$ fixed. Each NMSE pools 5,000 examples; lower values are better.
<table><tr><td>Model (seed)</td><td>Recon.</td><td> $Z ^ { ( 1 ) }$ </td><td> $Z ^ { ( 2 ) }$ </td></tr><tr><td>Reference (0)</td><td>0.007312</td><td>0.007407</td><td>0.007378</td></tr><tr><td>Recipient (2)</td><td>0.007927</td><td>0.011931</td><td>0.010487</td></tr><tr><td>Recipient (3)</td><td>0.007762</td><td>0.009342</td><td>0.010307</td></tr><tr><td>Recipient (5)</td><td>0.007825</td><td>0.010572</td><td>0.010985</td></tr><tr><td>Recipient (10)</td><td>0.007779</td><td>0.009505</td><td>0.010923</td></tr></table>

This experiment is an approximate illustration of the theory; it does not establish that the learned models form an exact constructive abstraction. The accompanying supplementary code generates the synthetic data, trains the models, and reproduces the reported results from fixed seeds.

## 5 RELATED WORKS

Causal abstractions and alignment. Prior work introduced exact transformations and constructive causal abstractions for relating causal models at different levels (Rubenstein et al., 2017; Beckers and Halpern, 2019; Geiger et al., 2025) and applied causal abstractions to neural networks (Geiger et al., 2021). Distributed alignment search (DAS) finds a rotation of a neural representation that places each specified high-level variable in its own subspace, allowing interventions on that variable to be implemented by editing the corresponding subspace (Geiger et al., 2024). DAS searches for an alignment to one high-level model; we instead characterize the representation changes guaranteed to preserve any constructive abstraction with the given blocks. Other work studies how abstractions can be discovered. Causal feature learning (Chalupka et al., 2017) and unsupervised causal abstraction discovery (Saulus et al., 2026) seek macro-variables or high-level models, while Li et al. (2025) use paired observations before and after unknown interventions to determine which latent variables cannot be distinguished and must therefore be represented as one higher-level variable. Our theory assumes that the low- and high-level models, blocks, and abstraction are given and asks which changes of representation preserve that abstraction.

Block identifiability. Independent subspace analysis (ISA) identifies latent subspaces up to their permutation and invertible transformations within them (Hyv¨arinen and Hoyer, 2000; Theis, 2006); addi tive decoders give an analogous guarantee under addi tive structure (Lachapelle et al., 2023). These transformations are a special case of the family T characterized by Thm. 1. Causal representation learning more broadly seeks to recover latent causal variables or their structure (Sch¨olkopf et al., 2021). Our theory starts from a given abstraction and studies whether it remains valid under the transformations left open by block identifiability.

Feature blocks and steering. For an SAE, each output coordinate of the encoder is one learned feature in our terminology. SAEs with diferent numbers of output coordinates need not learn the same features: a concept represented by one coordinate in an SAE with fewer output coordinates may be represented by several more specific coordinates in an SAE with more output coordinates (Bricken et al., 2023; Leask et al., 2025). Sparse Shift Autoencoders identify one-dimensional steering directions up to permutation and rescaling under their assumptions (Joshi et al., 2025). Block-sparse featurizers instead represent a concept using a multidimensional block and recover low-dimensional visual concept manifolds whose internal coordinates are defined only up to rotation (Fel et al., 2026). Together, these works motivate asking when steering a concept represented by a block retains its high-level causal meaning under changes of representation.

Approximate causal abstractions. Approximate causal abstraction allows the high-level model to reproduce the low-level system only up to a quantified er ror (Beckers et al., 2019), and abstraction maps can be learned by enforcing consistency across interventional distributions (Zennaro et al., 2023). Our maximality theorem concerns exact abstractions; the synthetic experiment measures approximate steering and transport rather than proving an approximate abstraction guarantee.

## 6 DISCUSSION

Limitations. Our result assumes deterministic SCMs L and H, a fixed block partition and its correspondence to the high-level variables, and a given exact constructive abstraction (τ, ω). It applies to hard interventions on complete blocks and does not show how these assumptions can be learned from data. Our synthetic experiment studies a controlled and approximate setting: the additive architecture matches the synthetic generator, concept-labelled observations are used to fit the cell maps π<sub>b</sub>, observations encoded by diferent additive autoencoders are used to fit transformations between their representations, and only au toencoders that reconstruct accurately are retained. The experiment therefore illustrates concept-directed steering and its transport in a simplified setting, but it does not establish an exact abstraction or provide a general method for discovering one.

Future work. The main open problems are to recover the blocks and (τ, ω) under explicit identifiability assumptions; extend the maximality result to stochastic SCMs, soft interventions, and quantitatively approximate abstractions; and evaluate concept-directed steering and transport in real neural representations, where exact generator comparisons are unavailable.

Conclusion. Under a given block-aligned constructive abstraction, the ambiguity left by blockidentifiability does not destroy the high-level causal interpretation. Block-diagonal reparameterizations, up to compatible coordinate permutations, are the maximal class guaranteed to preserve every such abstraction under the transported alignment; transformations outside this class may preserve a particular abstraction but have no universal guarantee (Thm. 1). In a controlled synthetic setting, we illustrate this connection by translating high-level concept interventions into edits of entire learned blocks and transporting those edits between independently trained representations (§ 4).

## AI use statement

Generative AI tools were used to assist with implementing, debugging, and testing the experimental code and to suggest limited edits to the manuscript for clarity and concision. All reported results were produced by executing the code, and the authors reviewed the AI-assisted code, experimental outputs, and manuscript edits. The authors take responsibility for the final content of this work, including all text, claims, and artifacts produced with the aid of genera tive AI.

## REFERENCES

Sander Beckers and Joseph Y. Halpern. Abstracting Causal Models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2019. URL http://arxiv.org/abs/1812.03789. arXiv:1812.03789.

Sander Beckers, Frederick Eberhardt, and Joseph Y. Halpern. Approximate Causal Abstraction. In Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI), 2019. URL http://arxiv. org/abs/1906.11583. arXiv:1906.11583.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. URL https://transformer-circuits.pub/2023/ monosemantic-features.

Krzysztof Chalupka, Frederick Eberhardt, and Pietro Perona. Causal feature learning: an overview. Behaviormetrika, 44(1):137–164, 2017. doi: 10.1007/ s41237-016-0008-2. Verified against Crossref 2026- 09-05.

Thomas Fel, Matthew Kowal, Mozes Jacobs, Dron Hazra, Usha Bhalla, Lee Sharkey, Lucius Bushnaq, Satchel Grant, Tal Haklay, Thomas Icard, Can Rager, Michael Pearce, Daniel Wurgaft, Aiden Swann, Fenil Doshi, Siddharth Boppana, Curt Tigges, Nick Cammarata, Thomas Serre, Vasudev Shyam, Owen Lewis, Thomas McGrath, Jack Merullo, Ekdeep Singh Lubana, and Atticus Geiger. Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds, June 2026. URL http://arxiv.org/abs/2606.25234. arXiv:2606.25234 [cs].

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal Abstractions of Neural Networks. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL http: //arxiv.org/abs/2106.02997. arXiv:2106.02997.

Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah Goodman. Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. In Proceedings of the Third Conference on Causal Learning and Reasoning, pages 160–187. PMLR, March 2024. URL https://proceedings.mlr.press/ v236/geiger24a.html. ISSN: 2640-3498.

Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard. Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. Journal of Machine Learning Research, 26(83):1–64, 2025. URL https://www.jmlr.org/papers/v26/ 23-0058.html. arXiv:2301.04709; journal version of the 2023 preprint.

Aapo Hyv¨arinen and Patrik Hoyer. Emergence of Phase- and Shift-Invariant Features by Decomposition of Natural Images into Independent Feature Subspaces. Neural Computation, 12(7):1705–1720, 2000. doi: 10.1162/089976600300015312. Verified against Crossref 2026-09-05.

Aapo Hyvarinen, Hiroaki Sasaki, and Richard E. Turner. Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning. arXiv:1805.08651 [cs, stat], February 2019. URL http://arxiv.org/abs/1805.08651. arXiv: 1805.08651.

Shruti Joshi, Andrea Dittadi, S´ebastien Lachapelle, and Dhanya Sridhar. Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts, February 2025. URL http://arxiv.org/abs/2502.12179. arXiv:2502.12179 [cs].

Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational Autoencoders and Nonlinear ICA: A Unifying Framework. In International Conference on Artificial Intelligence and Statistics, pages 2207–2217. PMLR, June 2020. URL http://proceedings.mlr.press/ v108/khemakhem20a.html. ISSN: 2640-3498.

S´ebastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, and Simon Lacoste-Julien. Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation, July 2023. URL http://arxiv.org/abs/2307.02598. arXiv:2307.02598 [cs, stat].

Patrick Leask, Bart Bussmann, Michael T. Pearce, Joseph Isaac Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. Sparse Autoencoders Do Not Find Canonical Units of Analysis. In International Conference on

Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=9ca9eHNrdH. arXiv:2502.04878.

Xiusi Li, S´ekou-Oumar Kaba, and Siamak Ravanbakhsh. On the Identifiability of Causal Abstractions. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3241–3249. PMLR, 2025. URL https: //proceedings.mlr.press/v258/li25g.html.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-Centric Learning with Slot Attention. arXiv:2006.15055 [cs, stat], October 2020. URL http://arxiv.org/abs/2006.15055. arXiv: 2006.15055.

Aaron Mueller, Andrew Lee, Shruti Joshi, Ekdeep Singh Lubana, Dhanya Sridhar, and Patrik Reizinger. From isolation to entanglement: When do interpretability methods identify and disentangle known concepts?, December 2025. URL http://arxiv.org/abs/2512.15134. arXiv:2512.15134 [cs].

Patrik Reizinger, Yash Sharma, Matthias Bethge, Bernhard Sch¨olkopf, Ferenc Husz´ar, and Wieland Brendel. Jacobian-based Causal Discovery with Nonlinear ICA. Transactions on Machine Learning Research, April 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=2Yo9xqR6Ab.

Paul K. Rubenstein, Sebastian Weichwald, Stephan Bongers, Joris M. Mooij, Dominik Janzing, Moritz Grosse-Wentrup, and Bernhard Sch¨olkopf. Causal Consistency of Structural Equation Models. In Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence (UAI). arXiv, July 2017. doi: 10.48550/arXiv.1707.00819. URL http://arxiv. org/abs/1707.00819. arXiv:1707.00819 [cs, stat].

Th´eo Saulus, Simon Lacoste-Julien, and Dhanya Sridhar. Unsupervised Causal Abstractions Discovery, June 2026. URL http://arxiv.org/abs/2606. 19594. arXiv:2606.19594.

Bernhard Sch¨olkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward Causal Representation Learning. Proceedings of the IEEE, 109(5):612–634, February 2021. doi: 10.1109/ JPROC.2021.3058954. URL http://arxiv.org/ abs/2102.11107. arXiv: 2102.11107 version: 1.

Shohei Shimizu, Patrik O Hoyer, Aapo Hyvarinen, and Antti Kerminen. A Linear Non-Gaussian Acyclic Model for Causal Discovery. page 28, 2006.

Fabian J. Theis. Towards a General Independent Subspace Analysis. In Advances in Neural Information Processing Systems 19 (NIPS), pages 1361–1368, 2006. URL https: //proceedings.neurips.cc/paper/2006/hash/ 20479c788fb27378c2c99eadcf207e7f-Abstract. html. Verified against DBLP 2026-09-05.

Fabio Massimo Zennaro, M´at´e Dr´avucz, Geanina Apachitei, W. Dhammika Widanage, and Theodoros Damoulas. Jointly learning consistent causal abstractions over multiple interventional distributions. In Conference on Causal Learning and Reasoning (CLeaR), volume 213 of Proceedings of Machine Learning Research, pages 88–121, 2023. URL https://proceedings. mlr.press/v213/zennaro23a.html. Author list corrected 2026-09-05 against the PMLR page; arXiv:2301.05893.

## When Do Learned Blocks Preserve Causal Meaning? Block Identifiability and Constructive Causal Abstractions: Supplementary Materials

## A TERMINOLOGY

Tab. 2 collects the terms the paper builds on with their notation, a one-line reading, and the place each is defined.

Table 2: Core terminology and notation used in the paper.
<table><tr><td>Term</td><td>Notation</td><td>Meaning</td><td>Defined in</td></tr><tr><td>Representation (feature vector)</td><td> $\mathbf { x } = g ( \mathbf { o } )$ </td><td>Featurizer output whose coordinates form  $\mathbf { V } _ { L }$ </td><td>§2</td></tr><tr><td>Activation vector</td><td>0</td><td>Activation read from the network and supplied to the featurizer g</td><td>1 § 2</td></tr><tr><td>Featurizer</td><td>g</td><td>Maps network activations to x; examples</td><td>§2</td></tr><tr><td>Learned feature</td><td> $X \in \mathbf { V } _ { L }$ </td><td>include SAEs and block-sparse featurizers Coordinate variable of the featurizer output</td><td>§2</td></tr><tr><td>Concept</td><td> ${ \cal Z } ^ { ( b ) } \in { \bf V } _ { H }$ </td><td>High-level variable, possibly encoded across</td><td>§2</td></tr><tr><td>Low-level model</td><td> $L , \mathbf { V } _ { L }$ </td><td>several learned coordinates Representation-level SCM on the featurizer output, whose coordinates carry the block</td><td>§ 2 and Assum. 1</td></tr><tr><td>High-level model</td><td> $H , \mathbf { V } _ { H }$ </td><td>partition SCM over the human-interpretable concepts</td><td>§2</td></tr><tr><td>Deterministic SCM</td><td>M</td><td>Signature Σ plus mechanisms  $\{ F _ { X } \}$  ; Solve(M) are its fixed points</td><td>§ 2; Geiger et al.</td></tr><tr><td>Hard intervention</td><td> $I \ ( \mathrm { o n } \ L ) , \ J$  (on H)</td><td>Replaces the mechanisms of a variable set by constants; identified with its overwrite map;</td><td>(2025) §2</td></tr><tr><td>Intervention algebra</td><td>Ψ</td><td>the empty intervention is the identity Composition algebra isomorphic to hard interventions; here, hard interventions on zero</td><td>§ 2; Assum. 1</td></tr><tr><td>Exact transformation</td><td> $( \tau , \omega )$ </td><td>or more complete blocks State and intervention maps commuting with Solve</td><td>eq. (1); Rubenstein et al. (2017); Geiger</td></tr><tr><td>Bijective translation</td><td> $T : L  L _ { T }$ </td><td>Coordinate change with transported interventions  $I ^ { T } { = } T \circ I \circ T ^ { - 1 } ;$  exact relative</td><td>et al. (2025) Defn. 3; Geiger et al. (2025)</td></tr><tr><td>Constructive abstraction</td><td></td><td>to the transported intervention algebra Exact transformation whose state and intervention maps are induced by an alignment</td><td>Defn. 4; Beckers and Halpern (2019);</td></tr><tr><td>Block partition</td><td> $\mathbf { V } ^ { ( b ) }$ </td><td>Disjoint grouping of the low-level variables</td><td>Geiger et al. (2025) eq. (4)</td></tr><tr><td>Alignment cell, cell</td><td> $\mathbf { V } ^ { ( b ) } , \pi _ { b }$ </td><td>into B blocks The block from which  $Z ^ { ( b ) }$  is computed and</td><td>Assum. 1</td></tr><tr><td>map Block-diagonal</td><td> $D \in \mathcal { D }$ </td><td>the surjection producing its value Bijection acting within each block, mixing</td><td>Defn. 5</td></tr><tr><td>reparameterization Transformation family</td><td> $T \in { \mathcal { T } }$ </td><td>none Maps  $T = \sigma \circ D$  with compatible σ and  $D \in \mathcal { D }$ </td><td>Defn. 8 and Thm. 1</td></tr><tr><td>Transported model</td><td> $L _ { T }$ </td><td>Mechanisms conjugated through T, so  $T : L  L _ { T }$  is a bijective translation</td><td>Defn. 3</td></tr><tr><td>Compatible permutation,</td><td> $\boldsymbol { \sigma } , \boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ </td><td>Permutation of coordinates with matching value domains, carrying each block to a possibly interleaved cell; whole-block</td><td>Defn. 7</td></tr></table>

When Do Learned Blocks Preserve Causal Meaning?
<table><tr><td>Term</td><td>Notation</td><td>Meaning</td><td>Defined in</td></tr><tr><td>Fibers</td><td> $\pi _ { b } ^ { - 1 } ( z )$ </td><td>The low-level cell values mapped to the same high-level value; equal fibers make two</td><td>§3.2 and Corollary 1</td></tr><tr><td>Block-identifiability</td><td></td><td>abstractions differ by a relabeling Recovery of blocks up to their permutation and invertible within-block maps; ISA gives the linear within-subspace case</td><td>§1</td></tr></table>

## B PROOFS

This section contains the proof of the main theorem and a secondary corollary.

## B.1 Universal maximality of block-diagonal invariance

Proof of Thm. 1. $( b ) { \Rightarrow } ( a )$ . Assume $T = \sigma \circ D$ with $D = \mathrm { D i a g } ( D ^ { ( 1 ) } , \dots , D ^ { ( B ) } )$ , and fix a target H and $( \tau , \omega ) \in$ ${ \cal A } ( L , H )$ Constructiveness of the transported pair requires two properties: an alignment with cells $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ inducing $\tau \circ T ^ { - 1 }$ , and exactness of $( \tau \circ T ^ { - 1 } , \omega ^ { T } ) : L _ { T }  H$

Step 1: the transported alignment. Let $\pi _ { b } : \operatorname { V a l } _ { \mathbf { V } ^ { ( b ) } } \twoheadrightarrow \operatorname { V a l } _ { Z ^ { ( b ) } }$ be the original cell maps, so

$$
\tau ( \mathbf { x } ) = \left( \pi _ { b } ( \mathbf { x } _ { \mathbf { V } ^ { ( b ) } } ) \right) _ { b = 1 } ^ { B } .\tag{9}
$$

For $\mathbf { y } = T ( \mathbf { x } ) , \sigma ^ { - 1 }$ moves the value in $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ back to $\mathbf { V } ^ { ( b ) }$ , and $( D ^ { ( b ) } ) ^ { - 1 }$ undoes the within-block change of coordinates:

$$
( T ^ { - 1 } ( { \bf y } ) ) _ { { \bf V } ^ { ( b ) } } = ( D ^ { ( b ) } ) ^ { - 1 } \big ( \sigma _ { b } ^ { - 1 } ( { \bf y } _ { \sigma ( { \bf V } ^ { ( b ) } ) } ) \big ) .\tag{10}
$$

Define the cell map on the transported cell $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ by

$$
\pi _ { b } ^ { T } : = \pi _ { b } \circ ( D ^ { ( b ) } ) ^ { - 1 } \circ \sigma _ { b } ^ { - 1 } .\tag{11}
$$

The cells $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ partition $\mathbf { V } _ { L }$ (Defn. $\mathrm { 7 ) ; }$ each $\pi _ { b } ^ { T }$ is surjective because $\pi _ { b }$ is and the remaining factors are bijective; and substituting the block formula for $T ^ { - 1 }$ into τ gives

$$
\big ( \tau \circ T ^ { - 1 } \big ) ( \mathbf { y } ) = \big ( \pi _ { b } ^ { T } \big ( \mathbf { y } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) } \big ) \big ) _ { b = 1 } ^ { B } ,\tag{12}
$$

the required alignment-induced form.

Step 2: $\it 2 .$ exactness. Let $I \in \mathrm { d o m } ( \omega ) = \Psi$ set the complete blocks indexed by $S \subseteq \{ 1 , \dots , B \}$ to values $\mathbf { i } _ { b } .$ . Its conjugate $I ^ { T }$ sets each transported cell $\sigma ( \mathbf { V } ^ { ( b ) } ) , b \in S$ , to

$$
\mathbf { i } _ { b } ^ { T } : = \sigma _ { b } ( D ^ { ( b ) } ( \mathbf { i } _ { b } ) ) ,\tag{13}
$$

and the transported cell map assigns this cell value the same macro-value:

$$
\pi _ { b } ^ { T } ( \mathbf { i } _ { b } ^ { T } ) = \pi _ { b } ( \mathbf { i } _ { b } ) .\tag{14}
$$

Hence $\omega ^ { T }$ , as defined in Defn. 6, coincides with the intervention map induced by the transported alignment $\mathrm { ( D e f n . ~ 4 ) }$ . Since $T : L  L _ { T }$ is a bijective translation (Defn. 3), and bijective translations are exact (Geiger et al., 2025),

$$
\mathrm { S o l v e } ( { L _ { T } ^ { I } } ^ { T } ) = T ( \mathrm { S o l v e } ( L ^ { I } ) ) ,\tag{15}
$$

using the transported-solution equation, exactness of $( \tau , \omega )$ , and $\omega ^ { T } ( I ^ { T } ) = \omega ( I )$ . The transported pair is exact and alignment-induced, hence constructive, proving (a).

$( a ) { \Rightarrow } ( b )$ . Fix the compatible permutation σ from the theorem statement. We construct a block-diagonal bijection D with $T = \sigma \circ D$ via the identity block abstraction: a coarse abstraction may discard the information needed to recover T, whereas the identity abstraction retains the complete value of every block.

Step 1: the identity abstraction lies in the quantified family. Define

$$
{ \tau _ { \mathrm { i d } } } ( { \bf x } ) = \left( { \bf x _ { V } } ( b ) \right) _ { b = 1 } ^ { B } ,\tag{19}
$$

which packages each complete block as one product-valued macro-variable and loses no information. Let $H _ { \mathrm { i d } }$ be the SCM with mechanisms conjugated through $\tau _ { \mathrm { i d } } ~ \mathrm { ( D e f n . ~ 3 ) }$ , with interventions transported accordingly. The pair is a bijective translation, hence exact, and is induced by the cells $\mathbf { V } ^ { ( b ) }$ with identity cell maps, so it lies in $ { \mathcal { A } } ( L , H _ { \mathrm { i d } } )$ ; because condition (a) ranges over every target H, it applies to this abstraction.

Step 2: blockwise factorization of $T ^ { - 1 }$ . Condition (a) says $\tau _ { \mathrm { i d } } \circ T ^ { - 1 }$ is constructive under the fixed transported cells $\boldsymbol { \sigma } ( \mathbf { V } ^ { ( b ) } )$ , so it supplies surjective cell maps

$$
\pi _ { b } ^ { T } : \operatorname { V a l } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) }  \operatorname { V a l } _ { \mathbf { V } ^ { ( b ) } }\tag{20}
$$

whose outputs are the complete original block values. Equating the b-th component of the alignment-induced state map with the b-th component of $\tau _ { \mathrm { i d } } \circ T ^ { - 1 }$ gives

$$
( T ^ { - 1 } ( { \bf y } ) ) _ { { \bf V } ^ { ( b ) } } = \pi _ { b } ^ { T } ( { \bf y } _ { \sigma ( { \bf V } ^ { ( b ) } ) } )\tag{21}
$$

for every b: each output block of $T ^ { - 1 }$ depends only on its assigned transported input block.

Step 3: the block factors are bijective. Each $\pi _ { b } ^ { T }$ is surjective as a cell map in the transported alignment. For injectivity, suppose u ̸= u<sup>′</sup> but $\begin{array} { r } { \dot { { \pi } } _ { b } ^ { T } ( { \mathbf { u } } ) = { \pi } _ { b } ^ { T } ( { \mathbf { u } } ^ { \prime } ) ; } \end{array}$ completing u and u<sup>′</sup> to total states $\mathbf { y } \neq \mathbf { y } ^ { \prime }$ that agree on every other transported cell, the component equations give $\bar { T } ^ { - 1 } ( \bar { \bf y } ) = T ^ { - 1 } ( { \bf y } ^ { \prime } )$ , contradicting injectivity of $T ^ { - 1 }$ . Hence every $\pi _ { b } ^ { T }$ is bijective.

Step $\it 4 :$ assembling D. With the bijection $\sigma _ { b } : \operatorname { V a l } _ { \mathbf { V } ^ { ( b ) } } \to \operatorname { V a l } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) }$ , define

$$
( D ^ { ( b ) } ) ^ { - 1 } : = \pi _ { b } ^ { T } \circ \sigma _ { b } : \mathrm { V a l } _ { \mathbf { V } ^ { ( b ) } }  \mathrm { V a l } _ { \mathbf { V } ^ { ( b ) } } ,\tag{22}
$$

and let $D ^ { - 1 } : = \mathrm { D i a g } ( ( D ^ { ( 1 ) } ) ^ { - 1 } , \ldots , ( D ^ { ( B ) } ) ^ { - 1 } )$ ). Comparing $D ^ { - 1 } \circ \sigma ^ { - 1 }$ with $T ^ { - 1 }$ on each block:

$$
( ( D ^ { - 1 } \circ \sigma ^ { - 1 } ) ( \mathbf { y } ) ) _ { \mathbf { V } ^ { ( b ) } } = ( D ^ { ( b ) } ) ^ { - 1 } \big ( \sigma _ { b } ^ { - 1 } ( \mathbf { y } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) } ) \big )\tag{23}
$$

$$
= \pi _ { b } ^ { T } ( \mathbf { y } _ { \sigma ( \mathbf { V } ^ { ( b ) } ) } )\tag{24}
$$

$$
\mathbf { \Lambda } = ( T ^ { - 1 } ( \mathbf { y } ) ) _ { \mathbf { V } ^ { ( b ) } } ,\tag{25}
$$

by the definition of $( D ^ { ( b ) } ) ^ { - 1 }$ and the block-factorization equation of Step 2. The maps agree on every block, so $\dot { T } ^ { - 1 } = D ^ { - 1 } \circ \sigma ^ { - 1 }$ and $T = \sigma \circ D$ , which is condition (b). Exactness is not used in this algebraic factorization; it is needed in Step 1 to place the identity abstraction in the family quantified over by condition (a). □

## B.2 Uniqueness up to macro value relabeling

Two surjective cell maps with the same fibers identify the same low-level values and therefore difer only by a relabeling of their high-level values. The following corollary applies this observation to abstractions related by a block reparameterization.

Corollary 1 (Uniqueness up to macro value relabeling). Let $L _ { 2 } = ( L _ { 1 } ) _ { D }$ for $D \in { \mathcal { D } }$ . Let $( \tau _ { 1 } , \omega _ { 1 } ) : L _ { 1 }  H _ { 1 }$ and $( \tau _ { 2 } , \omega _ { 2 } ) : L _ { 2 }  H _ { 2 }$ be constructive abstractions with the same cells $\mathbf { V } ^ { ( b ) }$ and cell maps $\pi _ { 1 , b } , \pi _ { 2 , b } ,$ and suppose that for every b the fibers $o f \pi _ { 2 , b }$ and $o f \pi _ { 1 , b } \circ ( D ^ { ( b ) } ) ^ { - 1 }$ coincide on ${ \mathrm { V a l } } _ { \mathbf { V } ^ { ( b ) } }$ . Then there is a unique blockwise bijection $\phi = ( \phi _ { b } ) _ { b } , \phi _ { b } : \operatorname { V a l } _ { Z _ { 1 } ^ { ( b ) } } \to \operatorname { V a l } _ { Z _ { 2 } ^ { ( b ) } }$ , with $\tau _ { 2 } = \phi \circ \tau _ { 1 } \circ D ^ { - 1 }$ and $\omega _ { 2 } ( { \dot { I } } ^ { D } ) = \phi \circ \omega _ { 1 } ( I ) \circ \phi ^ { - 1 }$ for every $I \in \Psi$ and $H _ { 1 }$ and $H _ { 2 }$ have ϕ-related solution sets under every intervention in $\omega _ { 1 } ( \Psi )$ . If moreover $\pi _ { 2 , b } = \pi _ { 1 , b } \circ ( D ^ { ( b ) } ) ^ { - 1 }$ for every b, then $\phi = \mathrm { i d }$ and the two macro-models have the same solution sets under every such intervention.

Proof of Corollary 1. By Thm. 1 with $\sigma = \mathrm { i d } , ( \tau _ { 1 } \circ D ^ { - 1 } , \omega _ { 1 } ^ { D } )$ is constructive $L _ { 2 }  H _ { 1 }$ with cell maps $\pi _ { 1 , b } \circ$ $( D ^ { ( b ) } ) ^ { - 1 }$ . Fix b: this map and $\pi _ { 2 , b }$ are surjections of Val <sub>(b)</sub> with the same fibers, so $\phi _ { b } ( z ) : = \pi _ { 2 , b } ( \mathbf { u } )$ for any u $\in ~ ( \pi _ { 1 , b } \circ ( D ^ { ( b ) } ) ^ { - 1 } ) ^ { - 1 } ( \bar { z } )$ is a well-defined bijection, and it is the only map with $\pi _ { 2 , b } = \phi _ { b } \circ \pi _ { 1 , b } \circ ( D ^ { ( b ) } ) ^ { - 1 }$ because the latter map is onto. Hence $\tau _ { 2 } = \phi \circ \tau _ { 1 } \circ D ^ { - 1 }$ , and by the definition of the induced intervention map (Defn. 4) the intervention $I ^ { D }$ setting the cells S to $D ^ { ( b ) } ( \mathbf { i } _ { b } )$ is sent by $\omega _ { 2 }$ to $\mathrm { d o } ( Z _ { 2 } ^ { ( b ) } { = } \phi _ { b } ( \pi _ { 1 , b } ( \mathbf i _ { b } ) ) , b \in S )$ which is $\phi \circ \dot { \omega } _ { 1 } ( I ) \circ \phi ^ { - 1 }$ because ϕ acts coordinatewise. Exactness of both pairs then gives, for every $I \in \Psi$ $\phi ( \mathrm { S o l v e } ( H _ { 1 } ^ { \omega _ { 1 } ( I ) } ) ) = \phi ( \tau _ { 1 } ( \mathrm { S o l v e } ( L _ { 1 } ^ { I } ) ) ) = \tau _ { 2 } ( \mathrm { S o l v e } ( L _ { 2 } ^ { I ^ { D } } ) ) = \mathrm { S o l v e } ( H _ { 2 } ^ { \omega _ { 2 } ( I ^ { D } ) } )$ , the empty intervention included; the last claim follows because equal cell maps force $\phi _ { b } = \mathrm { i d }$ □

## C CONCEPT-DIRECTED STEERING AND TRANSPORT DETAILS

Generator and model. Each example draws $s _ { 1 } , u _ { 1 } , s _ { 2 } , u _ { 2 }$ independently from Uniform[−1, 1], lifts each pair by $\phi ( s , u ) = ( s , u , s ^ { 2 } , s u , u ^ { 2 } ) ^ { \top }$ , and forms

$$
\mathbf { o } = A _ { 1 } \phi ( s _ { 1 } , u _ { 1 } ) + A _ { 2 } \phi ( s _ { 2 } , u _ { 2 } ) \in \mathbb { R } ^ { 1 0 } ,\tag{26}
$$

where $A = [ A _ { 1 } \ A _ { 2 } ] \in \mathbb { R } ^ { 1 0 \times 1 0 }$ is the orthogonal factor of one seeded random matrix and is fixed across all splits and models. The construction satisfies the reference-side suficient-nonlinearity condition of Lachapelle et al. (2023); the exact learned-side conditions of that result are not claimed for finite trained networks. For mode $r ,$ the encoder $E _ { r }$ has widths 10, 64, 64, 4, and each branch of the decoder $G _ { r }$ has widths 2, 64, 64, 10, with tanh hidden activations and linear outputs. Training minimizes reconstruction mean squared error (MSE) with Adam, using batch size 1024, learning rate $1 0 ^ { - 3 }$ , and weight decay $5 \times 1 0 ^ { - 4 }$ . Training runs for at most 2000 epochs and stops after 200 epochs without a validation-MSE improvement of at least $1 0 ^ { - 7 }$ ; the best validation checkpoint is retained. All runs use CPU-only local execution with four PyTorch threads. No factor label enters autoencoder training or checkpoint selection.

Data splits and fitted maps. The experiment uses 20,000/2,000 observations for autoencoder training and validation and 5,000 held-out observations for reconstruction eligibility. The reference cell maps use separate calibration, validation, and test splits of sizes 2,000, 2,000, and 5,000. The cross-model maps use 5,000/2,000 paired, unlabelled observations for fitting and validation. For each of 5,000 further base observations, we independently sample one intervention value $s _ { 1 } ^ { \star }$ and one intervention value $s _ { 2 } ^ { \star }$ from Uniform[−1, 1]. This gives 5,000 steering cases per concept, or 10,000 interventions in total; each corresponding edit is evaluated in the reference model and four retained recipient models.

For each possible pairing of reference block b and concept $j ,$ an afine candidate map

$$
\pi _ { b j } ( \mathbf { x } _ { 0 } ^ { ( b ) } ) = \mathbf { w } _ { b j } ^ { \top } \mathbf { x } _ { 0 } ^ { ( b ) } + c _ { b j }\tag{27}
$$

is fitted by least squares. The one-to-one assignment with the highest validation $R ^ { 2 }$ is retained, and its two maps become $\pi _ { 1 }$ and $\pi _ { 2 }$ . For every possible reference–recipient block pair, an afine map is fitted in standardized coordinates by least squares. The one-to-one pairing with the lowest worst-block validation NMSE estimates $\widehat { \sigma } _ { r }$ and is retained. After indexing each recipient block by its matched reference block, the selected within-block map $\widehat { D } _ { r } ^ { ( b ) }$ satisfies $\widehat { D } _ { r } ^ { ( b ) } ( \mathbf { x } _ { 0 } ^ { ( b ) } ) \approx \mathbf { x } _ { r } ^ { ( b ) }$ . This second fit sees paired codes of the same observations but no factor values.

Selecting a point in the fiber. Let $\mathbf { x } _ { 0 } ^ { ( b ) }$ be the intervened block of the reference code for the base observation. For the afine $\pi _ { b }$ , the closest point in its fiber over $s _ { b } ^ { \star }$ is

$$
\mathbf { x } _ { 0 } ^ { ( b ) \star } = \mathbf { x } _ { 0 } ^ { ( b ) } + \frac { s _ { b } ^ { \star } - \pi _ { b } ( \mathbf { x } _ { 0 } ^ { ( b ) } ) } { \| \mathbf { w } _ { b } \| ^ { 2 } } \mathbf { w } _ { b } , \qquad \pi _ { b } ( \mathbf { x } _ { 0 } ^ { ( b ) \star } ) = s _ { b } ^ { \star } .\tag{28}
$$

Replacing this block produces the reference code $\mathbf { x } _ { 0 } ^ { \star }$ and decoded observation $\widehat { \mathbf { o } } _ { 0 } ^ { \star } = G _ { 0 } ( \mathbf { x } _ { 0 } ^ { \star } )$ . Recipient model r receives $\bar { \mathbf { x } } _ { r } ^ { ( b ) \star } = \widehat { D } _ { r } ^ { ( b ) } ( \mathbf { x } _ { 0 } ^ { ( b ) \star } )$ , producing $\mathbf { x } _ { r } ^ { \star }$ and $\widehat { \mathbf { o } } _ { r } ^ { \star } = G _ { r } ( \mathbf { x } _ { r } ^ { \star } )$ . This Euclidean projection is a convenient way to choose one representative of the fiber; it is not a unique inverse of $\pi _ { b }$

Evaluation. For each intervention, we construct one exact generator observation by replacing only the intervened concept value in the factors of the base observation:

$$
\begin{array} { r } { \mathbf { o } ^ { \mathrm { g e n } } = A _ { b } \phi ( s _ { b } ^ { \star } , u _ { b } ) + A _ { 3 - b } \phi ( s _ { 3 - b } , u _ { 3 - b } ) , } \end{array}\tag{29}
$$

where $u _ { b }$ and the second factor pair come from the base observation. The same $\mathbf { o } ^ { \mathrm { g e n } }$ is used to evaluate the complete reference and recipient observations by computing the NMSE between $\widehat { \mathbf { o } } _ { r } ^ { \star }$ and $\mathbf { o } ^ { \mathrm { g e n } }$ for $r = 0$ and every retained recipient. Because the generator is injective and $\phi$ contains the original factor coordinates, this common comparison tests whether the edit sets $s _ { b }$ to $s _ { b } ^ { \star }$ while preserving its nuisance value and the other factor pair.

For predictions $\widehat { \mathbf { y } } _ { 1 } , \ldots , \widehat { \mathbf { y } } _ { N }$ and comparisons $\mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { N }$ , we define

$$
\mathrm { N M S E } ( \widehat { \mathbf { y } } , \mathbf { y } ) : = \frac { \sum _ { n = 1 } ^ { N } \| \widehat { \mathbf { y } } _ { n } - \mathbf { y } _ { n } \| ^ { 2 } } { \sum _ { n = 1 } ^ { N } \| \mathbf { y } _ { n } - \overline { { \mathbf { y } } } \| ^ { 2 } } , \qquad \overline { { \mathbf { y } } } : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { y } _ { n } .\tag{30}
$$

For cross-model map validation, each recipient coordinate is standardized using the corresponding training-set mean and standard deviation before applying this definition.

The thresholds were fixed before the run: reconstruction NMSE at most 0.01; reference-readout and decodedconcept $R ^ { 2 }$ at least 0.95; cross-model map NMSE at most 0.05; untargeted s and u $R ^ { 2 }$ at least 0.95; generatorvalidity residual at most three times the reconstruction floor; and final common-comparison NMSE at most 0.02.

Numerical results. The reference model reconstructs with NMSE 0.007312. Four of ten additional autoencoders pass the same gate, with NMSE between 0.007762 and 0.007927; the other six lie between 0.099071 and 0.100609. The two selected reference cell maps have test $R ^ { 2 }$ between 0.999602 and 0.999618, while each block has no predictive power for the other concept. The selected cross-model maps have validation NMSE between 0.001985 and 0.005839; the alternative block pairings have NMSE near 1.

All retained models satisfy the reconstruction criterion, and every predeclared calibration, map-validation, steer-${ \mathrm { i n g } } ,$ preservation, validity, and common-observation check passes.