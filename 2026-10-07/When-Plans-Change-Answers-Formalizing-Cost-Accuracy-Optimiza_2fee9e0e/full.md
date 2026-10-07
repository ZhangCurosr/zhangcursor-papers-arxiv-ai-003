# When Plans Change Answers: Formalizing Cost–Accuracy Optimization for Semantic Queries

Kyoungmin Kim KAIST

## Abstract

In semantic query engines, predicates are evaluated by machine-learned models, and the choice of a query plan afects not only the cost of a query but also its result. Existing systems either apply a fixed threshold to each semantic operator or tune accuracy per operator, without accounting for how errors propagate through joins. We give a formal problem definition for cost–accuracy optimization of such queries. Our starting point is the calibrated confidence that decision models such as Jev attach to each decision. It yields an expected error for every decision; weighting these errors by each decision’s contribution to the output (in the simplest case, its fan-out) gives the expected output quality of a plan without any labeled data, and the same computation in reverse turns an output-level accuracy target into a price on each base or intermediate tuple. Building on this, we define an oracle semantics for relational algebra with semantic operators, physical plans as pairs of a logical plan and a decision policy, declarative output-level targets, and a hierarchy of plan equivalence. We show that accuracy is plan-invariant under pointwise-deterministic policies, and that selection pushdown is not quality-sound when escalation bands are calibrated on the plan’s own candidates. Expected quality can be computed in polynomial time under bag semantics; under set semantics it follows the dichotomy of tupleindependent probabilistic databases when every relation carries a semantic predicate. Choosing which tuples to drop is NP-hard, while the optimization problem decomposes into per-tuple decisions through two Lagrange multipliers. Simulations on a synthetic workload illustrate these efects; an evaluation on real engines is left for future work.

## 1 Introduction

A classical query optimizer can treat all plans in its search space as interchangeable with respect to the result and compare them by cost alone. Semantic query engines break this assumption. Once a predicate is evaluated by a model, thresholds, cascades, candidate pruning, and samplebased calibration all afect which tuples are returned, and two plans for the same query may return diferent answers. In this paper we ask three questions. When should two such plans be considered equivalent? How can we predict the accuracy of a plan without comparing its output to ground truth? And how do we find the cheapest plan whose accuracy is good enough?

Running example. Consider a retailer that wants to contact every customer who bought a product for which some review reports a battery fire:

SELECT o.\* FROM Reviews r JOIN Orders o ON r.product = o.product WHERE SEM(r.text, ’reports a battery fire’)

We assume that the semantic predicate is evaluated by an inexpensive decision model, such as Jev in JEVDB [Wang et al., 2026], which returns a confidence $p _ { r }$ that review r satisfies the predicate.

Table 1: Running example: five reviews, their decision-model confidence $p ,$ and fan-out $f$ (orders of the reviewed product). A plan that accepts the reviews with $p \geq 0 . 5$ makes the expected errors on the right: per review (operator level) and per output row (output level).
<table><tr><td>Review</td><td>p</td><td> $f$ </td><td>Action</td><td>Expected error per review</td><td>Expected error in output rows</td></tr><tr><td> $r _ { 1 }$ </td><td>0.98</td><td>2</td><td>ACCEPT</td><td> $\mathrm { F P } \colon 1 - p = 0 . 0 2$ </td><td> $\mathrm { F P } \colon 0 . 0 2 \cdot 2 = 0 . 0 4$ </td></tr><tr><td> $r _ { 2 }$ </td><td>0.90</td><td>40</td><td>ACCEPT</td><td>FP: 0.10</td><td> $\mathrm { F P } \colon 0 . 1 0 \cdot 4 0 = 4 . 0 0$ </td></tr><tr><td> $r _ { 3 }$ </td><td>0.60</td><td>5</td><td>ACCEPT</td><td>FP: 0.40</td><td> $\mathrm { F P : 0 . 4 0 \cdot 5 = 2 . 0 0 }$ </td></tr><tr><td> $r _ { 4 }$ </td><td>0.30</td><td>100</td><td>REJECT</td><td>FN:  $p = 0 . 3 0$ </td><td> $\mathrm { F N } \colon 0 . 3 0 \cdot 1 0 0 = 3 0 . 0 0$ </td></tr><tr><td> $r _ { 5 }$ </td><td>0.05</td><td>3</td><td>REJECT</td><td>FN: 0.05</td><td> $\mathrm { F N } \colon 0 . 0 5 \cdot 3 = 0 . 1 5$ </td></tr><tr><td colspan="4">Expected recall / precision</td><td> $0 . 8 8 / 0 . 8 3$ </td><td>0.58 /0.87</td></tr></table>

Uncertain reviews can additionally be sent to a more expensive LLM judge. A review that passes the filter contributes one output row per order of its product; we call this number its fan-out $f _ { r }$ Table 1 lists five reviews.

From confidence to expected error. Suppose the confidences are calibrated. Then a review accepted with confidence $p$ is a false positive with probability 1 − $p ,$ and a rejected review is a false negative with probability $p .$ Let conf<sub>i</sub> denote the confidence in the action that was taken, i.e., $p _ { i }$ if the review is accepted and $1 - p _ { i }$ if it is rejected. By linearity of expectation,

$$
\mathbb { E } [ \# \mathrm { w r o n g ~ d e c i s i o n s } ] = \sum _ { i } { \big ( } 1 - \mathrm { c o n f } _ { i } { \big ) } ,
$$

where accepted instances account for the false positives and rejected instances for the false negatives. For the plan in Table 1, the filter makes 0.52 false positives and 0.35 false negatives in expectation, for an expected recall of $2 . 4 8 / ( 2 . 4 8 + 0 . 3 5 ) = 0 . 8 8$ . The query result, however, consists of orders and not of reviews. An error on $r _ { i }$ afects $f _ { i }$ output rows, so the expected number of wrong output rows is the same sum weighted by each decision’s contribution to the output, here $\textstyle \sum _ { i } f _ { i } ( 1 - \mathrm { c o n f } _ { i } )$ On output rows the expected recall drops to 0.58, and 30 of the 30.15 expected missing rows are due to a single review, $r _ { 4 } .$ , which is both uncertain and has a large fan-out.

From expected error back to decisions. The same reasoning can be applied in the opposite direction to make decisions. Escalating $r _ { 4 }$ to the judge costs one LLM call and removes 30 expected missing rows, which raises the expected output recall from 0.58 to 0.998. Escalating $r _ { 3 }$ also costs one call, but removes only 2 expected wrong rows. In general, an output-level target can be pushed down to a price per unit of expected output error. Each base or intermediate tuple then compares the cost of resolving its uncertainty with the reduction in output error that this buys. Figure 1 illustrates both directions.

Why calibrated confidence matters. Without confidences, the accuracy of a plan can only be measured against labeled ground truth, and an optimizer has nothing but cost to compare plans by. Boolean decisions from a fixed model and threshold do not help either, since then all classically equivalent plans return the same answer (Theorem 5.3). Calibrated confidences allow us to predict the output quality of a plan before executing it (Sections 4 and 7), to compare plans by their predicted quality and thereby decide which rewrites are safe (Section 5), and to optimize by assigning a price to output error at the level of individual tuples (Sections 6 and 8).

target (ρ, γ) pushed down as a price per unit of output error, weighed per tuple against the cost of an escalation, a drop, or a reorder  
![](images/62742721fdaea0200be8cef5231a7e61b8ea3a69a6c73cc92ac9da41c4680b7f.jpg)  
Figure 1: The role of calibrated confidence. Forward (solid): each decision has an expected error, which lineage weights by the decision’s contribution $c _ { i }$ to the output (in the running example, its fan-out); the sum gives the expected output quality of a plan. Backward (dashed): an output-level target induces a cost–accuracy trade-of for each tuple.

Motivating gaps. JEVDB [Wang et al., 2026] evaluates predicates with Jev either at a fixed threshold (its Flash mode) or with a per-predicate escalation band [ℓ, h). The band is fit on a sample of about 200 candidates labeled by an LLM judge, such that the expected recall and precision loss relative to the judge stay below user-given tolerances (0.05 in the experiments). JEVDB pushes semantic filters down by default, arguing that Jev calls are cheap, and leaves semantic-aware query optimization to future work. Like other current systems, it states accuracy targets per operator.

Per-operator targets can difer substantially from the quality of the query result. In Table 1, the filter has an expected recall of 0.88, whereas the query result has an expected recall of 0.58. To our knowledge, no current system lets the user specify a single accuracy target for the query and distributes it across operators and tuples.

## Contributions.

1. A formal model: oracle semantics, physical plans as (logical plan, decision policy), and outputlevel quality targets, with a plan’s expected output quality derived from calibrated confidences (Sections 2 to 4).

2. A hierarchy of semantic plan equivalence: a plan-invariance result for pointwise-deterministic implementations, a confidence-relative equivalence, and quality-sound rewrites with explicit counterexamples (Section 5).

3. A lineage-based account of how output error propagates to intermediate and base tuples, and how an output target is pushed down to a per-tuple price (Section 6).

4. Estimators of plan quality with judge-relative and oracle-relative guarantees, and a dichotomy for exact quality evaluation (Section 7).

5. The cost–accuracy optimization problem, its per-tuple Lagrangian decomposition, and complexity results (Section 8).

6. Simulations on a synthetic workload with stated parameters that illustrate plan invariance, plan dependence under calibrated bands, and the accuracy of confidence-based quality prediction. In this workload, plans for the same query difer by three orders of magnitude in cost and cover the whole range of output quality, and per-tuple pricing meets an output target at several times lower cost than an operator-level band (Section 9, Figs. 5 and 6). The simulations illustrate the mechanisms; they are not measurements of a deployed system.

## 2 Data Model and Semantic Relational Algebra

We define the correct answer of a query by evaluating it with oracle semantic predicates. Errors can then only come from the evaluation of semantic predicates, never from relational operators.

Definition 2.1 (Semantic operators). A semantic predicate is a function $s : \mathrm { d o m } ( x _ { 1 } ) \times \cdots \times$ $\mathrm { d o m } ( x _ { k } )  \{ 0 , 1 \}$ over text-valued (or multimodal) attributes, specified by a natural-language instruction $\phi .$ A semantic labeling function $c : \mathrm { d o m } ( x )  C$ maps to a finite label set $C ;$ a semantic score maps to an ordered set L. We write $s ^ { * }$ for the oracle (intended) interpretation of $\phi .$

Definition 2.2 (Semantic relational algebra, SRA). SRA extends positive relational algebra $( \sigma , \pi , \bowtie , \cup )$ with semantic selection $\sigma _ { s }$ , semantic join $\Join _ { s }$ , semantic projection $\pi _ { c }$ (classification), and semantic ranking $\tau _ { k }$ . Aggregation and negation are deferred (Section 11).

Definition 2.3 (Oracle semantics). For a query Q in SRA and database $D , Q ^ { * } ( D )$ denotes the result of evaluating Q with every semantic operator interpreted by its oracle. All plans are evaluated against $Q ^ { * } ( D )$

Example 2.4 (Oracle semantics in the running example). The query of Section 1 is

$$
Q = \sigma _ { s } ( R e v i e w s ) \bowtie _ { p r o d u c t } O r d e r s , \qquad \phi = \mathrm { ^ { \odot } r e p o r t s ~ a ~ b a t t e r y ~ f i r e ^ { \cdot \cdot } } .
$$

Suppose that, among the reviews in Table 1, exactly $r _ { 1 } , \ r _ { 2 }$ , and $r _ { 4 }$ satisfy $\phi .$ . Then $Q ^ { * } ( D )$ is the bag of the $2 + 4 0 + 1 0 0 = 1 4 2$ orders of those products. A plan that accepts $r _ { 1 } , r _ { 2 } , r _ { 3 }$ returns $2 + 4 0 + 5 = 4 7$ orders, of which 42 are in $Q ^ { * } ( D )$ . The confidences $p _ { i }$ in Table 1 express the model’s belief about these unknown labels, and Section 4 uses them to compute expected quality.

Remark 2.5. Because every $s ^ { * }$ is a deterministic function of its inputs, every classical equivalence of positive relational algebra remains valid under oracle semantics (selection pushdown, join commutativity and associativity, semijoin reduction). Hence, the logical search space is the same as in the classical setting, and approximation only enters at the physical level.

Bag or set outputs. We use bag semantics by default because it makes fan-out explicit; in Theorem 2.4, every order is a separate answer. Set semantics (e.g., SELECT DISTINCT o.customer) is a variant in which an answer survives if any of its derivations survives (Sections 6 and 7).

## 3 Physical Plans and Decision Policies

We model a physical plan as a logical plan together with a decision policy, which determines for every semantic decision whether it is accepted, rejected, escalated, or skipped.

Definition 3.1 (Decision instance). For a semantic operator s in plan L over database $D , { \mathrm { a } }$ decision instance is a pair $( s , \kappa )$ where κ is the tuple of attribute values s reads (for a join, the pair of texts). Let $I _ { L } ( D )$ be the multiset of decision instances that L presents to semantic operators.

Example 3.2 (Instances depend on the plan). In the running example, the pushdown plan $L _ { \downarrow } =$ $\sigma _ { s } ( R e v i e w s )$ ▷◁ Orders presents five instances, one per review. The pull-up plan $L _ { \uparrow } = \sigma _ { s } ( R e v i e w s$ ▷◁ Orders) evaluates the predicate on each joined row, so it presents $2 + 4 0 + 5 + 1 0 0 + 3 = 1 5 0$ instances; review $r _ { 4 } \mathrm { { ^ { * } s } }$ text appears 100 times. The two logical plans are equivalent, but $I _ { L _ { \downarrow } } ( D )$ and $I _ { L _ { \uparrow } } ( D )$ are diferent multisets. This diference becomes relevant for policies that depend on the population of instances (Table 2).

Table 2: Policy classes.
<table><tr><td>Class</td><td>Action depends on</td><td>Example</td></tr><tr><td>Pointwise-deterministic (PD) κ only, fixed function</td><td></td><td>Jev with fixed  $\tau ;$  JEVDB-Flash; “accept iff  $p \geq 0 . 5 ^ { \prime }$ </td></tr><tr><td>Population-calibrated (PC)</td><td>κ and a statistic of  $I _ { L } ( D )$ </td><td>escalation band  $[ \ell , h )$  fit on a sample of current candidates; JEVDB&#x27;s quality-controlled</td></tr><tr><td>Context-aware (CA)</td><td> $\kappa ,$  fan-out, downstream cost/selectivity</td><td>mode escalate  $r _ { 4 }$  because  $f _ { 4 } = 1 0 0$ </td></tr><tr><td>Sound pruning</td><td>necessary condition, sound extractor</td><td>lossless SBF; relational Bloom semijoin</td></tr></table>

Definition 3.3 (Decision model). A decision model M returns, for an instance $( s , \kappa )$ , a score $p _ { M } ( s , \kappa ) \in [ 0 , 1 ]$ . M is calibrated w.r.t. a distribution D over instances if $\mathrm { P r } _ { \kappa \sim { \cal { D } } } [ s ^ { * } ( \kappa ) = 1 \ | $ $p _ { M } = p \rbrack = p .$ . Examples include Jev’s Noul primitive, cross-encoders, and LLM judges that report verbalized or logit-based confidences.

Intuitively, calibration means that about 30% of all reviews with $p = 0 . 3$ actually report a fire. Rejecting $r _ { 4 }$ in Table 1 is therefore wrong with probability 0.3.

Definition 3.4 (Decision policy). A policy Π maps each decision instance, together with a context, to one of four actions:

• Accept: treat the instance as satisfying s (keep the tuple);

• Reject: treat it as not satisfying s (drop the tuple);

$\mathrm { E s c a L A T E } ( M _ { k } )$ : ask a more expensive model $M _ { k } { \mathrm { \Omega } } ( { \mathrm { e . g . } }$ , the LLM judge) and follow its answer;

• Skip: drop the tuple without evaluating it, e.g., after a negative probe of a semantic Bloom filter, or as a cost-motivated drop.

The context may include the instance’s score, its downstream fan-out, its position in the plan, and statistics observed so far.

Example 3.5 (Policies in the running example). The policy of Table 1 is “Accept if $p \geq 0 . 5 ,$ , else Reject”. A JEVDB-style band policy with band [0.2, 0.8) accepts $r _ { 1 } , r _ { 2 }$ , rejects $r _ { 5 } .$ and escalates $r _ { 3 }$ and $r _ { 4 }$ to the LLM. A context-aware policy might escalate only $r _ { 4 } ,$ , whose large fan-out makes an error costly, and accept $r _ { 3 }$ even though $r _ { 3 }$ is more likely to be wrong.

Definition 3.6 (Physical plan). $P = ( L , \Pi )$ . Its output $P ( D )$ is a random variable when Π uses sampling (e.g., threshold calibration) or a nondeterministic model.

Table 2 lists the policy classes used to state results.

Definition 3.7 (Cost). $\begin{array} { r } { C ( P , D ) \ = \ \sum _ { \mathrm { m o d e l ~ i n v o c a t i o n s } } \mathrm { p r i c e } ( M ) \cdot \mathrm { t o k e n s } + C _ { \mathrm { r e l } } ( P , D ) } \end{array}$ . Relational operators do not introduce errors, but they are not free. Each accepted review adds its f orders to the join and to any downstream work, such as composing one notification per order. Latency can be used instead of, or in addition to, monetary cost.

Example 3.8 (Cost in the running example). We charge one unit per decision-model call, 50 per LLM call, and 0.05 per joined row (the values used in Section 9). With the band policy of Theorem 3.5, the pushdown plan costs 5 model calls, 2 LLM calls, and the rows of the reviews it returns. The pull-up plan with the same band makes 150 model calls and escalates every one of the 105 rows of $r _ { 3 }$ and $r _ { 4 }$ unless decisions are cached by κ. This amounts to $1 5 0 + 1 0 5 \cdot 5 0 = 5 , 4 0 0$ units of model cost instead of $5 + 2 \cdot 5 0 = 1 0 5$ , for the same result.

## 4 Quality Semantics and Declarative Targets

We measure quality on the query result, relative to $Q ^ { * } ( D )$ . Accuracy of individual operators is an internal quantity that the optimizer may budget, but it is not what the user is promised. Calibrated confidences allow us to predict output quality without ground truth.

Definition 4.1 (Output quality). For bag outputs $O = P ( D )$ and $O ^ { * } = Q ^ { * } ( D )$ , with multiplicities and multiset intersection ∩,

$$
\operatorname { R e c } ( O , O ^ { * } ) = { \frac { | O \cap O ^ { * } | } { | O ^ { * } | } } , \qquad \operatorname { P r e c } ( O , O ^ { * } ) = { \frac { | O \cap O ^ { * } | } { | O | } } .
$$

Set versions apply after duplicate elimination. Ranking (top-k) and classification outputs use rank or label metrics; aggregates use relative error (deferred).

In Theorem 2.4, the plan returns 47 orders, 42 of which are correct, while there are 142 correct orders in total. Hence Rec $= 4 2 / 1 4 2 = 0 . 3 0$ and $\mathrm { P r e c } = 4 2 / 4 7 = 0 . 8 9$ . These values refer to one particular set of true labels; the confidences describe a distribution over such label sets.

Expected quality from calibrated confidences. Fix a plan P and assume that every semantic decision i comes with a calibrated confidence $p _ { i }$ . We define the confidence in the action taken as

$$
\mathrm { c o n f } _ { i } = \left\{ \begin{array} { l l } { p _ { i } } & { \mathrm { i f ~ } i \mathrm { ~ i s ~ a c c e p t e d , } } \\ { 1 - p _ { i } } & { \mathrm { i f ~ } i \mathrm { ~ i s ~ r e j e c t e d , } } \\ { \alpha _ { J } } & { \mathrm { i f ~ } i \mathrm { ~ i s ~ e s c a l a t e d ~ t o ~ a ~ j u d g e ~ o f ~ a c c u r a c y ~ } \alpha _ { J } , } \end{array} \right.
$$

so that decision i is wrong with probability $1 - \mathrm { c o n f } _ { i } .$ . Let $c _ { i }$ denote the number of output tuples whose derivation uses decision i. In the running example, $c _ { i }$ is the fan-out; Theorem 6.2 gives the general definition.

Definition 4.2 (Predicted quality). For a plan with one semantic operator under bag semantics, a perfect judge (α<sub>J</sub> = 1), and accepted, rejected, and escalated instance sets $A , R , E$

$$
\mathbb { E } [ \mathrm { T P } ] = \sum _ { i \in A \cup E } c _ { i } p _ { i } , \qquad \mathbb { E } [ \mathrm { F P } ] = \sum _ { i \in A } c _ { i } ( 1 - p _ { i } ) , \qquad \mathbb { E } [ \mathrm { F N } ] = \sum _ { i \in R } c _ { i } p _ { i } ,
$$

so that $\begin{array} { r } { \mathbb { E } [ \mathrm { F P } ] + \mathbb { E } [ \mathrm { F N } ] = \sum _ { i } c _ { i } ( 1 - \mathrm { c o n f } _ { i } ) } \end{array}$ . The predicted recall and precision are

$$
\widehat { \mathrm { R e c } } _ { M } ( P ) = \frac { \mathbb { E } [ \mathrm { T P } ] } { \mathbb { E } [ \mathrm { T P } ] + \mathbb { E } [ \mathrm { F N } ] } , \qquad \widehat { \mathrm { P r e c } } _ { M } ( P ) = \frac { \mathbb { E } [ \mathrm { T P } ] } { \mathbb { E } [ \mathrm { T P } ] + \mathbb { E } [ \mathrm { F P } ] } .
$$

Setting every $c _ { i } = 1$ gives the operator-level quantities. Section 6 extends $c _ { i }$ to several semantic operators and intermediate tuples, and Section 7 handles imperfect judges, set semantics, and miscalibration.

The ratio of expectations is an approximation of E[Rec] that becomes accurate when many decisions contribute to the output. We quantify the gap empirically in Section 9.

![](images/054a4e58abc4318db585762514109b2e57fcde7a0b40aa31b977feecc30411fe.jpg)

![](images/71802bf31c51ba8cbe7587ca1696cd0c1ba1c816724e9cb1e376bc4cb5132372.jpg)  
Figure 2: Expected errors of the plan in Table 1. Left: per review, $r _ { 3 }$ (accepted with confidence 0.6) has the highest error probability. Right: weighted by contribution (fan-out), $r _ { 4 }$ dominates; its error probability of 0.3 applies to 100 orders and accounts for 30 of the 30.15 expected missing output rows. An operator-level target corresponds to the left panel, while output quality corresponds to the right panel.

Example 4.3 (Operator-level vs. output-level prediction). For the policy of Table 1 and $c _ { i } = 1$ we obtain $\mathbb { E } [ \mathrm { T P } ] = 0 . 9 8 + 0 . 9 0 + 0 . 6 0 = 2 . 4 8 , \mathbb { E } [ \mathrm { F P } ] = 0 . 5 2 , \mathbb { E } [ \mathrm { F N } ] = 0 . 3 0 + 0 . 0 5 = 0 . 3 5$ , and hence $\widehat { \mathrm { R e c } } = 0 . 8 8$ and $\bar { \mathrm { P r e c } } = 0 . 8 3$ . With $c _ { i } = f _ { i }$ , we obtain $\mathbb { E } [ \mathrm { T P } ] = 1 . 9 6 + 3 6 + 3 = 4 0 . 9 6 , \mathbb { E } [ \mathrm { F P } ] = 6 . 0 4$ $\mathbb { E } [ \mathrm { F N } ] = 3 0 + 0 . 1 5 = 3 0 . 1 5$ , and hence $\bar { \mathrm { R e c } } = 0 . 5 8$ and $\bar { \mathrm { P r e c } } = 0 . 8 7$ . If $r _ { 4 }$ is escalated, it moves from $R$ to $E ;$ then $\mathbb { E } [ \mathrm { T P } ] = 7 0 . 9 6 , \mathbb { E } [ \mathrm { F N } ] = 0 . 1 5 , \widehat { \mathrm { R e c } } = 0 . 9 9 8$ , and $\bar { \mathrm { P r e c } } = 0 . 9 2$ . None of these numbers requires labeled data. Figure 2 breaks the errors down by review.

Definition 4.4 (Declarative target). A target is $T = \left( \rho , \gamma , \delta \right)$ . Plan P satisfies $T$ on D if

$$
\operatorname* { P r } _ { \Pi } \left[ \operatorname { R e c } ( P ( D ) , Q ^ { * } ( D ) ) \geq \rho \land \operatorname* { P r e c } ( P ( D ) , Q ^ { * } ( D ) ) \geq \gamma \right] \geq 1 - \delta .
$$

The expected variant replaces the probabilistic constraint with $\mathbb { E } [ \mathrm { R e c } ] \geq \rho$ and $\mathbb { E } [ \mathrm { P r e c } ] \geq \gamma ;$ the predicted variant uses ${ \mathrm { \widehat { R e c } } } _ { M } \geq \rho$ and ${ \widehat { \mathrm { P r e c } } } _ { M } \geq \gamma$ . Only the predicted variant can be checked by an optimizer.

For example, the plan in Table 1 violates the predicted target $T = ( 0 . 9 , 0 . 9 , \cdot )$ since $\widehat { \mathrm { R e c } } = 0 . 5 8$ whereas the same plan with $r _ { 4 }$ escalated satisfies it (0.998 and 0.92).

Definition 4.5 (Reference-relative quality). In practice, $Q ^ { * } ( D )$ cannot be observed. For a reference judge $J \ \left( \mathrm { { e . g . , \ a } } \right.$ frontier LLM), let $Q ^ { J } ( D )$ be $Q$ evaluated with J as the oracle. A guarantee is judge-relative if stated against $Q ^ { J }$ and oracle-relative if stated against $Q ^ { * }$

Whether a judge-relative guarantee carries over to the oracle depends on which instances the judge gets wrong, weighted by their contribution. Let $\Delta$ be the set of decision instances on which $J$ and $s ^ { * }$ disagree, let $c ^ { * } ( i )$ be the number of tuples of $Q ^ { * } ( D )$ whose derivation uses $i ,$ and $c ^ { J } ( i )$ the same count for $Q ^ { J } ( D )$ . Define the contribution-weighted judge error

$$
\varepsilon _ { J } = \frac { \sum _ { i \in \Delta } c ^ { * } ( i ) + \sum _ { i \in \Delta } c ^ { J } ( i ) } { | Q ^ { * } ( D ) | } .
$$

Proposition 4.6 (Judge-to-oracle transfer). Under bag semantics, for any plan output O,

$$
\mathrm { R e c } ( O , Q ^ { J } ( D ) ) \geq \rho \ \Rightarrow \ \mathrm { R e c } ( O , Q ^ { * } ( D ) ) \geq \rho - \varepsilon _ { J } ,
$$

$$
\operatorname* { P r e c } ( O , Q ^ { J } ( D ) ) \geq \gamma \Rightarrow \operatorname* { P r e c } ( O , Q ^ { * } ( D ) ) \geq \gamma - \frac { \sum _ { i \in \Delta } c ^ { J } ( i ) } { | O | } .
$$

Proof. Write $O ^ { * } = Q ^ { * } ( D )$ and $O ^ { J } = Q ^ { J } ( D )$ . An output tuple whose derivation uses no instance of $\Delta$ is derived identically under J and $s ^ { * }$ , so $\begin{array} { r } { | O ^ { * } \setminus O ^ { J } | \le \sum _ { i \in \Delta } c ^ { * } ( i ) } \end{array}$ and $\begin{array} { r } { | O ^ { J } \backslash O ^ { * } | \leq \sum _ { i \in \Delta } c ^ { J } ( i ) } \end{array}$ (bag diference). For multisets, $| O \cap O ^ { * } | \geq | O \cap O ^ { J } | { - } | O ^ { J } \backslash O ^ { * } |$ , and $| O \cap O ^ { J } | \geq \rho | O ^ { J } | \geq \rho \left( | O ^ { * } | { - } | O ^ { * } \backslash O ^ { J } | \right)$ Dividing by $| O ^ { * } |$ and using $\rho \leq 1$ gives $\operatorname { R e c } ( O , O ^ { * } ) \geq \rho - \varepsilon _ { J }$ . For precision, divide $| O \cap O ^ { * } | \geq$ $\gamma | O | - | O ^ { J } \setminus O ^ { * } |$ by |O|. □

Example 4.7 (Fan-out makes unweighted judge error meaningless). In Theorem 2.4, suppose the judge disagrees with the oracle only on $r _ { 4 } .$ , wrongly answering “no”. This is one error among five reviews, i.e., 20% without weighting. However, $c ^ { * } ( r _ { 4 } ) = 1 0 0$ of the 142 true orders and $c ^ { J } ( r _ { 4 } ) = 0$ so $\varepsilon _ { J } = 1 0 0 / 1 4 2 = 0 . 7 0$ , and a judge-relative recall of 0.95 only guarantees an oracle recall of 0.25. The bound is almost tight: the plan that returns exactly $Q ^ { J } ( D )$ has judge-relative recall 1 and oracle recall $4 2 / 1 4 2 = 0 . 3 0$ . Judge accuracy should therefore be measured with contribution weights, in particular on instances with high contribution (Section 7).

Remark 4.8. Current systems $( \mathrm { e . g . , J E V D B ^ { \prime } }$ s $\varepsilon _ { r } = \varepsilon _ { p } = 0 . 0 5$ relative to an LLM judge) provide judge-relative, operator-level, expected guarantees. The targets defined here are stated at the output level, either relative to the oracle or to a judge, and optionally with high probability.

## 5 Semantic Plan Equivalence

We define plan equivalence at four levels: oracle equivalence, implementation equivalence, confidence-relative equivalence, and admissibility. Approximate notions are defined relative to $Q ^ { * }$ rather than between pairs of plans, since pairwise ε-closeness is not transitive.

Definition 5.1 (Oracle equivalence). Logical plans $L _ { 1 } \equiv ^ { * } L _ { 2 }$ if $L _ { 1 } ^ { * } ( D ) = L _ { 2 } ^ { * } ( D )$ for all D. By Theorem 2.5 this coincides with classical equivalence of positive RA.

Definition 5.2 (Implementation equivalence). For a fixed PD policy Π, let sˆ denote the implemented predicate. $P _ { 1 } = ( L _ { 1 } , \Pi )$ and $P _ { 2 } = ( L _ { 2 } , \Pi )$ are implementation-equivalent if $L _ { 1 } [ \hat { s } ] ( D ) =$ $L _ { 2 } [ \hat { s } ] ( D )$ for all D.

Proposition 5.3 (Plan invariance under PD policies). If Π is pointwise-deterministic, then $L _ { 1 } \equiv ^ { * }$ $L _ { 2 }$ implies that $P _ { 1 }$ and $P _ { 2 }$ are implementation-equivalent, and hence have identical output quality on every D.

Proof. We have $\hat { s } ( \kappa ) = 1$ if Π accepts $\kappa ,$ or escalates it and the escalation model, itself a fixed function of $\kappa ,$ accepts it. Thus sˆ is a deterministic function of $\kappa ,$ and the plan computes $Q$ under the predicate interpretation sˆ. Classical rewrites of positive RA are valid for any deterministic interpretation of the predicates, so $L _ { 1 } [ \hat { s } ] ( D ) = L _ { 2 } [ \hat { s } ] ( D )$ □

Example 5.4 (Pushdown under a PD policy). With “Accept if $p \geq 0 . 5 ^ { \prime }$ , the pushdown plan of Theorem 3.2 accepts $r _ { 1 } , r _ { 2 } , r _ { 3 }$ and returns their 47 orders. The pull-up plan evaluates 150 instances, but each of the 100 instances of $r _ { 4 }$ carries the same text and the same $p = 0 . 3$ and is rejected each time; it returns the same $4 7$ orders. The two plans difer in cost (5 vs. 150 model calls) but return the same result. For PD policies, plan selection therefore reduces to classical cost-based optimization.

Corollary 5.5 (Source of coupling). Output quality can difer between oracle-equivalent plans only through $( a )$ population-calibrated policies, $( b )$ context-aware policies, $( c )$ approximate (non-sound) pruning, or (d) model nondeterminism. In particular, JEVDB-Flash’s accuracy is invariant to pushdown, while that of its quality-controlled mode is not, since its escalation band is $\mathit { f i t }$ on a sample of the candidates its plan presents. JEVDB motivates pushdown by cost considerations (Wang et al., 2026, §3.1). Theorem 5.11 below shows that, in its calibrated mode, pushdown also afects accuracy.

Table 3: Four levels of plan equivalence, from strongest to weakest. Only the last two levels distinguish plans of diferent quality, and checking them before execution requires calibrated confidences.
<table><tr><td>Level</td><td> $P _ { 1 }$  and  $P _ { 2 }$  are related iff</td><td>In Fig. 3</td><td>Checkable without labels?</td></tr><tr><td>Oracle (≡*)</td><td>same answer under oracle predicates</td><td>all 486 points</td><td>yes (classical rules)</td></tr><tr><td>Implementation</td><td>same answer under the implemented predicates</td><td>two ends of a black segment (PD policy)</td><td>yes, for PD policies</td></tr><tr><td>Confidence-relative  $\left( \equiv _ { M } \right)$ </td><td>same predicted output recall and precision</td><td>points at the same height</td><td>yes, from confidences</td></tr><tr><td>Admissibility (A)</td><td>both meet the output target</td><td>points in the shaded band</td><td>predicted form  $\left( \mathcal { A } _ { M } \right)$  only</td></tr></table>

Confidence-relative equivalence. If plans can difer in quality, an optimizer needs to know their quality in order to compare them. Ground truth is not available at optimization time, but calibrated confidences provide the predicted quality of every candidate plan (Theorem 4.2), which an optimizer can use for this purpose.

Definition 5.6 (Confidence-relative equivalence and dominance). Fix a calibrated decision model M. Plans $P _ { 1 } , P _ { 2 }$ with $P _ { 1 } \equiv ^ { * } P _ { 2 }$ (on their logical parts) are M-equivalent on $D _ { : }$ , written $P _ { 1 } \equiv _ { M } P _ { 2 } ,$ if $\widehat { \mathrm { R e c } } _ { M } ( P _ { 1 } ) = \widehat { \mathrm { R e c } } _ { M } ( P _ { 2 } )$ and $\widehat { \mathrm { P r e c } } _ { M } ( P _ { 1 } ) = \widehat { \mathrm { P r e c } } _ { M } ( P _ { 2 } )$ . $P _ { 1 }$ M-dominates $P _ { 2 }$ if it is at least as good on both and no more expensive in expectation.

Example 5.7. In Table 1, escalating $r _ { 3 }$ or escalating $r _ { 4 }$ costs the same one LLM call, but the predicted output quality is (0.58, 0.91) for $r _ { 3 }$ and (0.998, 0.92) for $r _ { 4 } .$ . The second plan M-dominates the first. Without confidences, the two plans cannot be told apart before execution, as both consist of the threshold plan plus one escalation.

Figure 3 shows all physical plans of the running example by cost and predicted quality, and Table 3 summarizes the four levels of equivalence.

Definition 5.8 (Admissibility). P is T-admissible for Q if it satisfies T (Theorem 4.4) on D. The admissible set $\mathcal { A } ( Q , T , D )$ takes the role of the equivalence class of correct plans. Its predicted counterpart $\mathcal { A } _ { M } ( Q , T , D )$ uses $\widehat { \mathrm { R e c } } _ { M }$ and $\widehat { \mathrm { P r e c } _ { M } }$

Definition 5.9 (Quality-sound rewrite). A rewrite r is quality-sound for a policy class C if for every P with policy in C, $P \in { \mathcal { A } }$ implies $r ( P ) \in { \mathcal { A } }$

Proposition 5.10. Every classical rewrite of positive RA is quality-sound for PD policies.

Proof. By Theorem 5.3, $r ( P )$ returns the same output as P on every D.

Proposition 5.11 (Pushdown is not quality-sound for PC policies). There is a query, database, target, and population-calibrated policy such that the pull-up plan is admissible and the pushdown plan is not.

![](images/eb83fba1cf7e6bb054d1368a8bf66460e61504edc5a9e6098e7e589df8e4e789.jpg)  
expected cost (log scale; 1 per Jev call, 50 per LLM call, 0.05 per joined row)  
Figure 3: The physical plan space of the running example, predicted from confidences alone. Each point is one plan: a logical plan (pushdown or pull-up) and an action per review $( 3 ^ { 5 } \times 2 = 4 8 6$ plans). Equivalence: all points are oracle-equivalent (same $Q ^ { * } )$ ; a PD policy gives two points at the same height, the same answer at diferent cost (black segment); points at the same height are M-equivalent (Theorem 5.6). Optimization: the admissible plans are the shaded band, and cost– accuracy optimization picks its leftmost point. A classical optimizer with a fixed threshold only chooses between the two ends of the black segment and keeps the cheaper one, at predicted quality 0.58; the operator-level band is admissible but costs 1.8× the optimum.

Proof. Consider the query of the running example and the target $T = ( 0 . 9 , 0 . 9 )$ in its expected variant. Let Reviews contain 90 reviews of niche products with $p = 0 . 9 8$ and fan-out 1, and 10 reviews of popular products with $p = 0 . 8 0$ and fan-out 100. The policy is calibrated on the instances presented by its plan. It accepts every instance with $p \geq h$ , where h is the smallest score at which the expected precision of the accepted instances reaches 0.95, and escalates the rest to a perfect judge. No instance is rejected, so recall is 1 for both plans.

• Pushdown calibrates on the 100 reviews. Accepting all of them has expected precision (90·0.98+ $1 0 \cdot 0 . 8 0 ) / 1 0 0 = 0 . 9 6 2 \geq 0 . 9 5$ , so $h = 0 . 8 0$ and nothing is escalated. The expected precision on output rows is $( 8 8 . 2 + 8 0 0 ) / ( 9 0 + 1 0 0 0 ) = 0 . 8 1 5 < 0 . 9$ , so the plan is not admissible.

$P u l l – u p$ calibrates on the 1,090 joined rows. Accepting all of them has expected precision $0 . 8 1 5 <$ 0.95, so $h = 0 . 9 8$ and the 1,000 rows of popular products are escalated. The judge keeps only the true ones, so the expected output precision is $8 8 8 . 2 / ( 9 0 + 8 0 0 ) = 0 . 9 9 8$ and the plan is admissible.

The two plans calibrate on diferent populations, namely reviews for pushdown and joined rows $( \mathrm { i . e . } ,$ , reviews weighted by fan-out) for pull-up. The popular reviews are 10% of the first population and 92% of the second. A PD policy with the pull-up plan’s band would give both plans the same answer (Theorem 5.3). □

Section 9 reproduces this efect on a 2,000-review workload in which the calibration sample is drawn at random.

Proposition 5.12 (Sound and approximate pruning). Call a pruning step sound if it removes only tuples t with $s ^ { * } ( t ) = 0 \ ( e . g .$ , a lossless semantic Bloom filter or a relational Bloom semijoin). Under a PD or context-aware policy whose actions on the surviving instances are unchanged, a sound pruning step never decreases recall and never decreases precision. An approximate step that removes each tuple t with $s ^ { * } ( t ) = 1$ independently with probability $\beta _ { t }$ lowers the expected number of true positives by at most $\textstyle \sum _ { t } \beta _ { t } q _ { t } c ^ { \Pi } ( t )$ , with equality when no output’s witness contains two such tuples $( e . g .$ , one semantic operator); here $q _ { t }$ and $c ^ { \Pi } ( t )$ are as in Theorem 6.2.

Proof. A sound step removes no tuple that occurs in a derivation of $Q ^ { * } ( D )$ . Hence true positives are unafected, and the removed tuples could only have produced false positives. For the approximate step, a true positive is lost only if some tuple of its witness is removed. By a union bound over the removed tuples and linearity of expectation, the expected loss is at most $\textstyle \sum _ { t } \beta _ { t }$ times the expected number of true positives whose witness contains t, which is $q _ { t } c ^ { \Pi } ( t )$ . If no witnesses are shared, the events are disjoint and the bound is exact. □

Under a PC policy, pruning also changes the calibration population, so Theorem 5.11 applies to it as well.

Example 5.13 (Non-transitivity of pairwise closeness). Measure closeness by

$$
d ( P _ { i } , P _ { j } ) = \frac { | P _ { i } ( D ) \triangle P _ { j } ( D ) | } { | Q ^ { * } ( D ) | }
$$

and take $Q ^ { * } ( D )$ to be the 142 orders of Theorem 2.4. Let $P _ { 1 }$ return all 142, $P _ { 2 }$ drop 7 of them, and $P _ { 3 }$ drop 7 more. Then $d ( P _ { 1 } , P _ { 2 } ) = d ( P _ { 2 } , P _ { 3 } ) = 7 / 1 4 2 < 0 . 0 5$ , yet $\operatorname { R e c } ( P _ { 3 } ) = 1 2 8 / 1 4 2 = 0 . 9 0$ which violates $\rho = 0 . 9 3$ although $P _ { 1 }$ is perfect. A sequence of rewrites, each of which is close to the previous plan, can thus leave the admissible set. For this reason, Theorem 5.8 defines correctness relative to $Q ^ { * }$

## 6 Error Propagation via Lineage

In Theorem 4.2, the expected error of each decision is weighted by its contribution $c _ { i }$ to the output. In this section, we define $c _ { i }$ for plans with several semantic operators and for both base and intermediate tuples. We then use contributions in two directions: forward, to compute output quality from per-decision confidences, and backward, to derive per-tuple prices from an output target. Since false negatives cannot be recovered by later operators while false positives may stil be filtered out, recall and precision compose diferently, and they favor diferent operator orders.

## 6.1 Lineage and contribution

Definition 6.1 (Decision lineage). For $o \in Q ^ { * } ( D )$ under bag semantics, its witness $w ( o )$ is the set of decision instances evaluated on the derivation of o. Under set semantics, o has a set of witnesses $W ( o )$ and its lineage is a DNF over Boolean decision variables $x _ { i } ,$ i.e., its provenance polynomial in the Boolean semiring [Green et al., 2007]:

$$
\Phi _ { o } = \bigvee _ { w \in W ( o ) } \bigwedge _ { i \in w } x _ { i } .
$$

Definition 6.2 (Output contribution of a decision instance). Contributions are defined for decision instances, which are the units a policy acts on. An instance may refer to a base tuple (a semantic filter on a base relation) or to an intermediate tuple (a join pair, or a filter above a join). We distinguish two versions.

• Oracle contribution $c ^ { * } ( i )$ : the number of tuples of $Q ^ { * } ( D )$ whose witness contains i. It depends only on $Q$ and $D _ { : }$ , not on the plan, and we use it in definitions.

• Policy contribution $c ^ { \Pi } ( i )$ : the expected number of correctly returned outputs lost if i is answered negatively, given the other decisions under Π. This is the quantity used by the optimizer:

$$
c ^ { \Pi } ( i ) = \mathbb { E } _ { \Pi } [ \mathrm { T P } \mid x _ { i } = 1 ] - \mathbb { E } _ { \Pi } [ \mathrm { T P } \mid x _ { i } = 0 ] \stackrel { \mathrm { \tiny ~ b a g ~ S P J } } { = } \sum _ { \substack { o : i \in w ( o ) } } \prod _ { j \in w ( o ) \backslash \{ i \} } q _ { j } ,
$$

where $q _ { j }$ is the probability that decision j is a correct positive under Π $( p _ { j }$ if j is accepted or escalated to a perfect judge, 0 if rejected).

With a single semantic operator, all other factors equal 1, and $c ^ { \Pi } ( r ) = f _ { r }$ is the fan-out used in Section 4. With several operators, downstream decisions reduce the contribution of upstream decisions, and upstream decisions determine whether a downstream decision afects the output at all.

Example 6.3 (Base and intermediate decisions). We extend the running example with a second predicate that reads both tables and is therefore evaluated on joined rows: SEM((r.text, o.notes), ’the customer reports the same defect’). Its decision instances are intermediate tuples $( r , o )$ . For review $r _ { 4 } ~ ( p = 0 . 3 , 1 0 0 ~ \mathrm { o r d e r s } )$ , suppose 20 of its joined rows get confidence $p = 0 . 9$ and 80 get $p = 0 . 0 5$ , and the second operator accepts if $p \geq 0 . 5 ;$ for the accepted rows, $q _ { o } = p _ { o } = 0 . 9$

• Output to base. An order of $r _ { 4 }$ is a true answer only if both predicates hold. Rejecting $r _ { 4 }$ loses the 20 accepted rows, each a true positive with probability 0.9, so $c ^ { \Pi } ( r _ { 4 } ) = 2 0 \cdot 0 . 9 = 1 8$ rather than 100. The expected number of output false negatives caused by rejecting $r _ { 4 }$ is $0 . 3 \cdot 1 8 = 5 . 4$ instead of 30; because the downstream predicate is selective, an error on $r _ { 4 }$ costs less.

• Output to intermediate. A joined row $( r _ { 4 } , o )$ can only contribute if $r _ { 4 }$ is kept, i.e., $c ^ { \Pi } ( ( r _ { 4 } , o ) ) =$ $q _ { r _ { 4 } } = 0 . 3$ if $r _ { 4 }$ is accepted and 0 if it is rejected. If $r _ { 4 }$ is rejected, none of the 100 decisions of the second operator on its rows can afect the output, and evaluating them is wasted work; this is the usual argument for pushing the first filter down. If $r _ { 4 }$ is escalated and confirmed, each of these decisions matters fully.

Figure 4 illustrates both cases.

Note that $c ^ { \Pi } ( i )$ is the influence of $x _ { i }$ on expected output quality in the sense of sensitivity analysis for probabilistic databases [Kanagal et al., 2011]. Tuple Shapley values [Livshits et al., 2020] would be an alternative way to attribute output quality to decisions.

## 6.2 From decision errors to output error

Under bag semantics and independent decisions, each expected output count is a sum over output tuples of a product over their witnesses. For instance,

$$
\mathbb { E } [ \mathrm { T P } ] = \sum _ { o } \prod _ { j \in w ( o ) } q _ { j } .
$$

For one semantic operator this is Theorem 4.2,

$$
\mathbb { E } [ \mathrm { F N } ] = \sum _ { i \mathrm { r e j e c t e d } } p _ { i } c ^ { \mathrm { I I } } ( i ) , \qquad \mathbb { E } [ \mathrm { F P } ] = \sum _ { i \mathrm { a c c e p t e d } } ( 1 - p _ { i } ) c ^ { \mathrm { I I } } ( i ) ,
$$

i.e., the output error is the sum of the per-decision errors $1 - \cos \mathrm { f } _ { i }$ weighted by $c ^ { \Pi } ( i )$ . With several operators, the expected number of false negatives caused by rejecting i alone is $p _ { i } c ^ { \Pi } ( i )$ , as in Theorem 6.3. Under set semantics, which we discuss in Section 7, the sum of products is replaced by the probability of a DNF.

![](images/cfc75a67ba7b444bc8ef981f0a2c9d008dc1cf1fda5cf7a57fb979be160b8604.jpg)  
Figure 4: Propagation of contributions from the output to intermediate and base decisions (Theorem 6.3). Orange: the contribution of a decision is the expected number of true outputs it contributes to, computed top-down. Because the second predicate is selective, the contribution of $r _ { 4 }$ is 18 rather than 100, and rejecting $r _ { 4 }$ loses 5.4 instead of 30 expected rows. Blue: an intermediate decision only matters if its base tuple is kept, so if $r _ { 4 }$ is rejected, the 100 decisions of the second predicate on its rows need not be evaluated.

Computing contributions by backward propagation. Computing $c ( i )$ requires propagating weight from the output back to every decision instance. For an acyclic join tree under bag semantics, this can be done by sum-product message passing: a top-down pass sends each tuple the expected number of surviving completions in the rest of the tree, where semantic operators contribute factors $q _ { j }$ (or selectivity estimates for operators not yet evaluated). This takes linear time in the size of the pass [Khamis et al., 2016] and computes c for base and intermediate instances simultaneously. In Theorem 6.3, the message sent from the orders of $r _ { 4 }$ to $r _ { 4 }$ is $\sum _ { o } q _ { o }$ [o accepted] = 18. The cost saved by dropping $i , \Delta C ( i )$ , can be computed in the same way in a cost semiring, as the expected number of downstream rows times the per-row cost of each downstream operator.

## 6.3 Pushing an output target down to tuples

An output-level target $( \rho , \gamma )$ constrains sums of per-decision terms, which suggests enforcing it through a price. Suppose each unit of expected output error has price λ. Resolving decision $i ,$ e.g., by escalation, removes $c ^ { \Pi } ( i ) \left( 1 - \mathrm { c o n f } _ { i } \right)$ expected output errors at the cost of the escalation. Dropping a tuple saves its downstream cost but adds $c ^ { \Pi } ( i ) p _ { i }$ expected false negatives. Each tuple can then decide locally by comparing these quantities. Theorem 8.2 makes this precise, using one price for the recall constraint and one for the precision constraint.

Example 6.4 (Pricing the running example). Let an escalation cost 50 units and let each expected wrong output row cost $\lambda = 1 0$ units. Escalating $r _ { 4 }$ removes 30 expected missing rows, worth $3 0 0 > 5 0$ , so $r _ { 4 }$ is escalated. Escalating $r _ { 2 }$ would remove 4 expected false rows (worth $4 0 < 5 0 )$ and escalating $r _ { 3 }$ would remove 2 (worth $2 0 < 5 0 )$ , so both keep the decision of the cheap model. The resulting plan has predicted output quality (0.998, 0.92) and meets $T = ( 0 . 9 , 0 . 9 )$ with a single LLM call. An operator-level band such as [0.2, 0.8) escalates both $r _ { 3 }$ and $r _ { 4 } ,$ and thus spends a second call on a review that accounts for only 2 expected errors. With the second predicate of

Theorem 6.3, the benefit of escalating $r _ { 4 }$ drops to $5 . 4 \cdot 1 0 = 5 4 $ , which is still slightly above 50; with a somewhat lower price, $r _ { 4 }$ would be left to the cheap model.

## Caveats.

1. $c ^ { \Pi }$ is a marginal quantity. Contributions are not additive if two dropped instances share a witness, so selecting a set of instances to drop is a set-function problem with a coverage-type loss rather than a plain knapsack problem (Section 8).

2. When an upstream decision is made, downstream decisions are not yet known, so $c ^ { \Pi }$ must be estimated. Policy and contributions are thus defined by a fixed point.

3. Under set semantics $c ^ { \Pi }$ is the influence on a DNF probability, #P-hard in general (Section 7).

4. The denominator $| Q ^ { * } ( D ) |$ is the same for all instances and does not afect their ranking.

## 6.4 Recall and precision compose diferently

Proposition 6.5 (Recall composes multiplicatively). For a bag SPJ query, suppose that every true-positive instance of operator j is accepted with probability $R _ { j }$ , independently across instances. Then every true output survives with probability $\Pi _ { j } R _ { j }$ , so $\begin{array} { r } { \mathbb E [ \mathrm { R e c } ] = \prod _ { j } R _ { j } } \end{array}$ . Consequently, a global target $\rho$ is allocated log-additively:

$$
\begin{array} { r } { \sum _ { j } \log R _ { j } \geq \log \rho . } \end{array}
$$

Proof. A true output survives if every decision on its witness accepts its instance, all of which are true positives. By independence this has probability $\Pi _ { j } R _ { j }$ for every true output, and $\mathbb { E } [ \mathrm { R e c } ]$ is the average over true outputs. □

For example, two filters with recall 0.95 each yield an output recall of 0.9025, so an output target of 0.95 requires a recall of about 0.975 per operator. The product form no longer holds when recall varies across instances with diferent $c ( i ) { \mathrm { ; } }$ ; in Table 1, an operator recall of 0.88 results in an output recall of 0.58.

For an accepted instance i, let $d ^ { \Pi } ( i { \bf \theta } | { \bf \theta }  )$ be the expected number of candidate output rows whose witness contains i and which every other decision on the witness accepts, conditioned on $s ^ { * } ( i ) = 0$

Proposition 6.6 (Precision is order- and correlation-dependent). Under bag semantics, accepting instance i instead of rejecting it adds $( 1 - p _ { i } ) d ^ { \Pi } ( i \mid \neg )$ expected false output rows. In particular, if the other decisions on i’s rows reject them whenever $s ^ { * } ( i ) = 0$ , accepting i costs no output precision, however uncertain i is.

Proof. If $s ^ { * } ( i ) = 0$ , which has probability $1 - p _ { i }$ , every returned row whose witness contains i is a false positive, and the rows returned are exactly those accepted by all other decisions on their witness. Linearity of expectation gives the first claim; the second is the case $d ^ { \Pi } ( i \mid \lnot ) = 0$ □

Since $d ^ { \Pi } ( i { \bf \Pi } | { \bf \Pi }  )$ depends on the decisions on i’s witnesses and on how they treat i’s false positives, the precision cost of a threshold depends on the operator order and on correlations between predicates. This is in contrast to recall (Theorem 6.5). For example, if most reviews that wrongly pass the fire filter concern unrelated defects, the second predicate of Theorem 6.3 rejects their rows anyway, and a looser threshold for the first filter saves escalations with little loss in output precision. This suggests two heuristics:

• Recall pushdown: the recall budget should be spent on instances with small contribution $c ( i )$

• Precision pull-up: operators upstream of strict, independent filters may run with lower precision.

Computing fan-out. For acyclic relational subqueries, per-tuple witness counts can be computed in linear time with counting semijoins (Yannakakis-style [Yannakakis, 1981] bottom-up and top-down passes with counts). c(i) additionally requires downstream decision probabilities (Section 7).

## 7 Estimating Quality from Calibrated Confidences

With calibrated scores, each semantic decision can be modeled as a Bernoulli variable, and the expected quality of a plan becomes a query over a probabilistic database. Theorem 4.2 covered a single operator under bag semantics. In this section, we consider general SPJ plans, set semantics, imperfect judges, and miscalibrated models.

Output level. We treat the decision variables $x _ { i }$ as independent, with $\operatorname* { P r } [ x _ { i } ] = q _ { i }$ the probability that i is a correct positive under the policy’s action $( p _ { i }$ on accept, 0 on reject, $\alpha J p _ { i }$ on escalation to a judge of recall $\alpha _ { J } )$ . Then $\mathrm { P r } [ o $ returned $\mathrm { c o r r e c t l y } ] = \mathrm { P r } [ \Phi _ { o } ]$ , which amounts to query evaluation over a tuple-independent probabilistic database [Dalvi and Suciu, 2007, Suciu et al., 2011].

Definition 7.1 (Lineage-inducing query). For a plan whose logical part is a self-join-free SPJ query $Q ,$ , the lineage-inducing query $Q ^ { \mathrm { l i n } }$ is the conjunctive query obtained by replacing every semantic operator s over relations $R _ { 1 } , \ldots , R _ { k }$ by a fresh probabilistic relation $S _ { s } ( \kappa )$ whose tuple κ has probability $q _ { \kappa }$ . Relational atoms are deterministic.

The lineage of an output tuple o in the plan coincides with its lineage in $Q ^ { \mathrm { l i n } }$ , so $\mathrm { P r } [ o $ returned correc $\mathrm { { t l y } } ] = \operatorname* { P r } [ o \in Q ^ { \operatorname* { l i n } } ( D ) ]$

Example 7.2 (Bag vs. set semantics). Suppose the running example returns distinct customers (SELECT DISTINCT o.customer) and that customer Ann bought both the product of $r _ { 2 } ~ ( p = 0 . 9 )$ and that of r<sub>4</sub> $( p = 0 . 3 )$ , both of which are accepted. Under bag semantics, Ann appears twice, and the expected number of correct rows for Ann is the sum $0 . 9 + 0 . 3 = 1 . 2 $ . Under set semantics, Ann is returned correctly if $x _ { 2 } \vee x _ { 4 }$ holds, which has probability $1 - ( 1 - 0 . 9 ) ( 1 - 0 . 3 ) = 0 . 9 3 $ . Sums of products remain easy to compute for larger queries, whereas probabilities of DNFs in general do not.

Theorem 7.3 (Complexity of quality evaluation). Let P be a plan over a self-join-free SPJ query with semantic selections and calibrated, independent decisions.

1. Under bag semantics, $\mathbb { E } [ \mathrm { T P } ] , \mathbb { E } [ \mathrm { F P } ]$ , and $\mathbb { E } [ \mathrm { F N } ]$ are computable in time polynomial in |D|.

2. Under set semantics, E[TP] is computable in polynomial time $i f Q ^ { \mathrm { l i n } }$ is hierarchical. If every relation of Q carries a semantic selection (so every atom of $Q ^ { \mathrm { l i n } }$ is probabilistic) and $Q ^ { \mathrm { l i n } }$ is not hierarchical, it is #P-hard.

Proof. (1) By linearity of expectation, $\begin{array} { r } { \mathbb E [ \mathrm { T P } ] = \sum _ { o } \prod _ { j \in w ( o ) } q _ { j } } \end{array}$ , which is a sum over derivations of products of independent factors. It can be computed by evaluating the join in the sum-product semiring, i.e., by multiplying a probability column along each derivation and summing at the end. This takes polynomial time, and linear time for acyclic queries [Khamis et al., 2016]. The cases of $\mathbb { E } [ \mathrm { F P } ]$ and E[FN] are analogous, with factor $1 - p _ { j }$ or $p _ { j }$ for the erroneous decision.

(2) For each output $o , \operatorname* { P r } [ o $ returned correct $\mathrm { l y } ] = \operatorname* { P r } [ o \in Q ^ { \mathrm { l i n } } ( D ) ]$ , and E[TP] is the sum of these probabilities. If $Q ^ { \mathrm { l i n } }$ is hierarchical, each of these probabilities can be computed with a safe plan in polynomial time [Dalvi and Suciu, 2007]. For hardness, let $q$ be a non-hierarchical self-join-free Boolean CQ, e.g., $q = \exists x , y R ( x ) \land S ( x , y ) \land T ( y )$ , over a tuple-independent database with probabilities $\mu ( t )$ ; computing $\operatorname* { P r } [ q ]$ is #P-hard [Dalvi and Suciu, 2007]. We construct an SRA instance with the same relations, give each tuple t a text attribute, and apply one semantic selection to each relation with calibrated confidence $p _ { t } = \mu ( t )$ . Let the plan accept every decision (a PD policy) and project onto the empty attribute list under set semantics. The plan returns the single empty tuple whenever the relational join is nonempty, and that tuple is a true answer if some derivation has all its oracle labels equal to 1. Hence $\mathbb { E } [ \mathrm { T P } ] = \operatorname* { P r } [ q ]$ . The same construction works for any non-hierarchical self-join-free $q ,$ with $Q ^ { \mathrm { l i n } } = q$ . (When some relations carry no semantic operator, their atoms are deterministic and the hardness boundary can move; we leave the exact dichotomy for that case open.) □

The running example under bag semantics falls under case (1), and every quantity in Table 1 and Theorems 4.3 and 6.4 is a sum of products. Case (2) becomes relevant, $\therefore \mathrm { g } .$ , for DISTINCT over a chain of two semantic joins, where sampling-based approximations of DNF probabilities can be used.

Imperfect judges. If the judge’s confidence is set to 1, all guarantees become judge-relative. Instead, we model the accuracy $\alpha _ { J }$ of the judge, estimated on a small gold set, and set $\mathrm { c o n f } _ { i } = \alpha _ { J }$ for escalated decisions in Theorem 4.2. Escalated instances are, by design, the most dificult ones, so the judge’s accuracy on them is likely below its average accuracy. It should therefore be estimated on the escalated band and, following Theorem 4.7, weighted by contribution.

## Assumptions on calibration.

• Calibration holds with respect to a distribution of instances. Pushdown or a join changes which instances are evaluated, and the resulting covariate shift can break calibration. In Theorem 5.11, the same reviews make up 10% of the population of one plan and 92% of the other.

• Errors may be correlated, since the same document appears in many pairs (and as a shared anchor in batched request layouts). This violates the independence assumption.

• Prediction-powered inference (PPI) [Angelopoulos et al., 2023] combines model predictions with a small gold sample to correct bias and obtain confidence intervals, and SUPG-style procedures [Kang et al., 2020] provide precision and recall guarantees for proxy-based selection.

Contribution-weighted PPI. For an output-level total $\begin{array} { r } { \theta \ = \ \sum _ { i } w _ { i } y _ { i } \ \mathrm { ( e . g . } } \end{array}$ , TP with $w _ { i } ~ =$ $c _ { i }$ [i accepted] and $y _ { i } = s ^ { * } ( i ) )$ , the predicted value $\sum _ { i } w _ { i } p _ { i }$ is biased if M is miscalibrated. With n gold labels drawn with probability proportional to $c _ { i } .$ , the PPI estimate

$$
\hat { \theta } = \sum _ { i } w _ { i } p _ { i } + \frac { 1 } { n } \sum _ { j \in \mathrm { g o l d } } \frac { w _ { j } \left( y _ { j } - p _ { j } \right) } { c _ { j } / \sum _ { k } c _ { k } }
$$

is unbiased, and sampling in proportion to contribution reduces its variance when fan-out is skewed.   
In Section 9, this estimator removes the precision bias of an overconfident model using 100 labels.

## 8 The Optimization Problem and Complexity

We now state the optimization problem: choose a logical plan and a decision policy that minimize expected cost subject to an output-level quality target. With predicted quality, the constraint can be checked, and with contributions, it can be decomposed over tuples. The general problem nevertheless remains hard.

Problem 8.1 (Cost–Accuracy Semantic Query Optimization, CASQO). Given Q, D, a target $T = \left( \rho , \gamma , \delta \right)$ , a set of decision models with prices, and a policy class ${ \mathcal { C } } ,$

$$
\operatorname* { m i n } _ { L \equiv ^ { * } Q , \Pi \in \mathcal { C } } \mathbb { E } \big [ C ( ( L , \Pi ) , D ) \big ] \quad \mathrm { s . t . } \quad ( L , \Pi ) \in A ( Q , T , D ) .
$$

Variants.

• Operator-level allocation: Π assigns one threshold or band per semantic operator.

• Instance-level allocation: Π may treat instances of the same operator diferently based on context (fan-out, downstream cost); the running example belongs to this variant.

• Judge-relative vs. oracle-relative target; expected vs. high-probability constraint.

## 8.1 Per-tuple decomposition

Fix the logical plan and consider the predicted variant of the expected target. Since ${ \widehat { \mathrm { R e c } } } \geq \rho$ if $\left( 1 - \rho \right) \mathbb { E } [ \mathrm { T P } ] - \rho \mathbb { E } [ \mathrm { F N } ] \ge 0$ , and $\widehat { \mathrm { P r e c } } \geq \gamma \mathrm { i f f } \left( 1 - \gamma \right) \mathbb { E } [ \mathrm { T P } ] - \gamma \mathbb { E } [ \mathrm { F P } ] \geq 0$ , both constraints are linear in the expected counts. Under bag semantics and with fixed contributions $c _ { i } .$ , each expected count is a sum of per-instance terms (Theorem 4.2). Instance i adds $c _ { i } p _ { i }$ to TP if accepted or escalated, $c _ { i } p _ { i }$ to FN if rejected or skipped, and $c _ { i } ( 1 - p _ { i } )$ to FP if accepted. Let $C _ { i } ( a )$ denote the expected cost of action a on $i ,$ including the downstream work it causes.

Proposition 8.2 (Per-tuple Lagrangian decomposition). For multipliers $\lambda _ { R } , \lambda _ { P } \ \geq \ 0$ , let each instance choose

$$
a _ { i } ( \lambda ) \in \arg \operatorname* { m i n } _ { a } \ C _ { i } ( a ) + \lambda _ { R } \big ( \rho \mathrm { F N } _ { i } ( a ) - ( 1 - \rho ) \mathrm { T P } _ { i } ( a ) \big ) + \lambda _ { P } \big ( \gamma \mathrm { F P } _ { i } ( a ) - ( 1 - \gamma ) \mathrm { T P } _ { i } ( a ) \big ) .
$$

If the resulting policy meets both linearized constraints, and each multiplier is zero unless its constraint is tight, then the policy is optimal among all instance-level policies for the fixed logical plan and predicted target.

Proof. Let $g _ { R } ( a ) = ( 1 - \rho ) \mathbb { E } [ \mathrm { T P } ] - \rho \mathbb { E } [ \mathrm { F N } ]$ and $g _ { P } ( a ) = ( 1 - \gamma ) \mathbb { E } [ \mathrm { T P } ] - \gamma \mathbb { E } [ \mathrm { F P } ]$ . The Lagrangian $C ( a ) { - } \lambda _ { R } g _ { R } ( a ) { - } \lambda _ { P } g _ { P } ( a )$ is a sum of per-instance terms, so $a ( \lambda )$ minimizes it over all action vectors. For any feasible $\begin{array} { r } { a ^ { \prime } , C ( a ^ { \prime } ) \geq C ( a ^ { \prime } ) - \lambda _ { R } g _ { R } ( a ^ { \prime } ) - \lambda _ { P } g _ { P } ( a ^ { \prime } ) \geq C ( a ( \lambda ) ) - \lambda _ { R } g _ { R } ( a ( \lambda ) ) - \lambda _ { P } g _ { P } ( a ( \lambda ) ) = 0 } \end{array}$ $C ( { \boldsymbol { a } } ( \lambda ) )$ , the last step by complementary slackness (Everett’s suficiency argument). □

In terms of Figs. 3 and 5, CASQO asks for the leftmost point in the shaded region. Intuitively, Theorem 8.2 charges each tuple for its own expected output errors, scaled by its contribution, which corresponds to the backward arrow in Fig. 1. An operator-level policy applies the same rule with all $c _ { i }$ set to 1 and thus treats a review with 2 orders like a review with 100. Since actions are discrete, multipliers $\left( \lambda _ { R } , \lambda _ { P } \right)$ that satisfy the conditions exactly need not exist. In that case, a grid search over the two multipliers that returns the cheapest feasible policy is a heuristic; this is the variant evaluated in Section 9.

Example 8.3 (Decomposition on the running example). With $\rho = \gamma = 0 . 9$ , an escalation cost of 50, and row costs ignored, $\lambda _ { R } = 1 2$ and $\lambda _ { P } = 0$ yield the following values for $r _ { 4 } !$ : reject costs $\lambda _ { R } \rho c _ { 4 } p _ { 4 } = 1 2 { \cdot } 0 . 9 { \cdot } 3 0 = 3 2 4$ , accept costs $- \lambda _ { R } ( 1 - \rho ) { \cdot } 3 0 = - 3 6$ (accepting adds true positives, which the recall constraint rewards), and escalate costs $5 0 - 3 6 = 1 4$ . Accepting minimizes this objective, so a price on precision is needed as well. With $\lambda _ { P } = 5$ , accepting $r _ { 4 }$ adds $5 ( 0 . 9 \cdot 7 0 - 0 . 1 \cdot 3 0 ) = 3 0 0$ for its 70 expected false rows, which gives 264 for accept, 324 for reject, and $5 0 - 3 6 - 1 5 = - 1$ for escalate, so $r _ { 4 }$ is escalated. For ${ r _ { 3 } } ~ ( c = 5 , p = 0 . 6 )$ the same prices give 3.9 for accept, 32.4 for reject, and 44.9 for escalate; similarly, r<sub>1</sub> and r<sub>2</sub> are accepted and $r _ { 5 }$ is rejected. The resulting policy escalates $r _ { 4 }$ only, as in Theorem 6.4. Since the resulting recall is 0.998, the recall constraint is not tight, and Theorem 8.2 does not certify optimality. The policy is nevertheless optimal, since every feasible policy needs at least one LLM call (accepting $r _ { 4 }$ would reduce predicted precision to 0.48).

## 8.2 Complexity

Proposition 8.4 (Instance-level dropping is NP-hard). Choosing which instances to Skip so as to maximize saved cost subject to output recal $l \geq \rho$ is NP-hard, even for one semantic filter followed by one join and with all confidences equal to 1.

Proof. We reduce from $0 / 1$ knapsack with weights $w _ { k }$ , values $v _ { k }$ , capacity W, and target value V. For each item, create a review $r _ { k }$ with $p = 1$ and fan-out $f _ { k } = w _ { k }$ , and give its joined rows a downstream per-row operator whose total cost over $r _ { k } \mathrm { { ^ { * } s } }$ rows is $v _ { k } \ { \mathrm { ( e . g . } }$ , by the token length of those rows). Skipping $r _ { k }$ saves $v _ { k }$ and loses $w _ { k }$ true output rows. With $\rho = 1 - W / \sum _ { k } w _ { k }$ , a skip set is feasible if its total weight is at most $W$ , and it saves at least V if the knapsack instance is a yes-instance. □

Note that the reduction does not use uncertainty; hardness already arises from fan-out-weighted recall. With uncertain confidences, the weights are scaled by $p _ { k }$ and the reduction still applies.

Proposition 8.5 (Operator-level recall allocation is convex). Under the assumptions of Theorem 6.5, let operator j reach recall $R _ { j }$ at expected cost $C _ { j } ( R _ { j } )$ , convex and nondecreasing on (0, 1]. Then minimizing $\Sigma _ { j } C _ { j } ( R _ { j } )$ subject to $\Pi _ { j } R _ { j } \geq \rho$ is a convex program, and at an interior optimum $R _ { j } C _ { j } ^ { \prime } ( R _ { j } )$ is the same for every operator.

Proof. The constraint is equivalent to $\textstyle \sum _ { j }$ log $R _ { j } \geq \log \rho .$ . Since log is concave, the feasible set is a superlevel set of a concave function and hence convex; the objective is convex by assumption. The KKT conditions $C _ { j } ^ { \prime } ( R _ { j } ) = \lambda / R _ { j }$ give $R _ { j } C _ { j } ^ { \prime } ( R _ { j } ) = \lambda$ □

In words, the marginal cost of an additional unit of log-recall is the same for all operators, so operators that are cheap to make accurate should receive a larger share of the recall budget. With two filters and target $\rho = 0 . 9$ , equal cost curves give $R _ { 1 } = R _ { 2 } = \sqrt { 0 . 9 } \approx 0 . 9 4 9$

Table 4 summarizes the results and their status.

Tractable class. Queries with acyclic relational parts, for which fan-out can be computed with counting semijoins in linear time, are tractable under bag semantics, and under set semantics if the semantic predicates are hierarchical. This matches the Yannakakis-style reductions already used in practice [Wang et al., 2026].

Table 4: Results and their status.
<table><tr><td>Result</td><td>Statement</td><td>Status</td></tr><tr><td>Quality evaluation</td><td>Bag: PTIME. Set: PTIME for hierarchical, #P-hard for non-hierarchical with all atoms probabilistic</td><td>Theorem 7.3</td></tr><tr><td>Plan invariance</td><td>Under PD policies, CASQO separates into classical cost-based QO plus a plan-independent policy choice</td><td>Theorem 5.3</td></tr><tr><td>PC unsoundness</td><td>Selection pushdown is not quality-sound for population-calibrated policies</td><td>Theorem 5.11</td></tr><tr><td>Per-tuple decomposition</td><td>Two prices push an output target down to independent Theorem 8.2 per-tuple decisions</td><td></td></tr><tr><td>Instance-level drop</td><td>NP-hard, even with one filter, one join, and  $p \equiv 1$ </td><td>Theorem 8.4</td></tr><tr><td>Greedy approximation</td><td>Ranking by saved  $\mathrm { c o s t } / c ( i )$  gives a knapsack-style guarantee under known  $p _ { i }$  and fan-out; coverage structure when witnesses overlap</td><td>open</td></tr><tr><td>Operator-level recall</td><td>Under independence, minimizing cost s.t.  $\textstyle \sum _ { j } \log R _ { j } \geq \log \rho$  is convex when per-operator cost-recall curves are convex</td><td>Theorem 8.5</td></tr><tr><td>Precision coupling</td><td>No order-independent threshold assignment exists in general</td><td>open</td></tr></table>

## 9 Simulations

We illustrate the definitions with simulations of the running example. All parameters are stated below; experiments on a real engine, e.g., JEVDB with Jev and an LLM judge on SemBench [Lao et al., 2026], are left for future work. The claims we test concern how quality and cost depend on confidences and fan-out, not how well a particular model is calibrated, and for these claims simulation is suficient. The workloads are synthetic and were designed to make each efect visible. The reported magnitudes therefore illustrate what the definitions predict and should not be read as estimates of the gains of a deployed system. Two design choices are important. In E2, reviews of popular products are uncertain but mostly positive. In an earlier configuration, in which these reviews had confidences around 0.5, pushdown escalated them anyway, and both plans met the target about equally often (93% and 96%). In E3 and in the plan-space figures, confidence is drawn independently of fan-out. If the uncertain reviews are exactly those with high fan-out, a single band already escalates them, and per-tuple prices save only about 1%. All numbers are produced by python3 sim/simulate.py (standard library only), which generates the tables and plot data used in this section.

Workload. We use the query of the running example with 2,000 reviews per instance. A review belongs to a popular product with probability 0.1 (fan-out uniform in [50, 400]) and otherwise to a niche product (fan-out uniform in [1, 8]). We first draw the confidence p of the decision model and then the oracle label $s ^ { * } \sim$ Bernoulli(p), which makes the model calibrated. Reviews of niche products are mostly unambiguous, with p ∼ Beta(0.3, 8) with probability 0.85 and $p \sim \mathrm { B e t a } ( 3 0 , 1 )$ otherwise. Reviews of popular products are likely positive but less certain, with p ∼ Beta(8, 2) (mean 0.8); this creates the covariate shift of Theorem 5.11. A decision-model call costs 1 unit, an LLM call (escalation, always answered correctly) costs 50, and downstream work costs 0.05 per joined row. The target is an output recall and precision of at least 0.9. Each experiment averages over 200 random instances (50 for E3), and ± denotes one standard deviation.

Table 5: E1 and E2: pushdown vs. pull-up under a pointwise-deterministic band [0.1, 0.9) and under bands calibrated on a sample of 200 instances of each plan’s own population (operator-level target 0.95). Quality is measured on output rows. Cost is relative to PD pushdown; pull-up evaluates every joined row and does not cache decisions. Admissible: fraction of instances meeting the output target (0.9, 0.9).
<table><tr><td>Plan</td><td></td><td>Output recall Output precision</td><td>F1</td><td>Cost</td><td>Admissible</td></tr><tr><td>PD, pushdown</td><td> $0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 9 8 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>1.00</td><td>100%</td></tr><tr><td>PD, pull-up</td><td> $0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 9 8 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>88.22</td><td>100%</td></tr><tr><td>PC, pushdown</td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 9 2 4 \pm 0 . 0 6 4$ </td><td> $0 . 9 5 6 \pm 0 . 0 3 7$ </td><td>0.86</td><td>62%</td></tr><tr><td>PC, pull-up</td><td> $0 . 9 5 0 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 9 7 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 9 6 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>71.40</td><td>96%</td></tr></table>

How much does plan choice matter? Figure 5 shows the plan space for one simulated database, with confidence independent of fan-out as in E3. Plans for the same query difer in cost by a factor of up to 1, 261 and cover the whole range of output quality, so an optimizer can make costly mistakes in both dimensions. Moreover, the two dimensions are not aligned; among the plans that meet the target, cost still varies by a factor of 307. The plans chosen by diferent strategies are far apart:

• a cost-only optimizer with a fixed threshold selects the cheapest plan (cost 2,621), which returns only a fraction 0.67 of the true orders;

• escalating all decisions gives the exact result at cost 102,817;

• an operator-level target of 0.95 with JEVDB-style calibration meets the output target at cost 47,461, and the best single band chosen by predicted output quality at cost 37,690;

• per-tuple prices (Theorem 8.2) meet the target at cost 6,492.

Figure 6 repeats the comparison for targets between 0.8 and 0.98. Per-tuple pricing is cheaper than the best operator-level band by a factor of 1.9 to 5.3.

E1: plan invariance under a PD policy. With a fixed band, pushdown and pull-up return identical results on all 200 instances (Table 5, first two rows), as predicted by Theorem 5.3, and difer only in cost. For PD policies, the choice between the two plans is thus a classical cost-based decision.

E2: plan dependence under a calibrated band. If each plan fits its band on a sample of the instances it evaluates, the resulting bands difer, with [0.16, 0.77) for pushdown and [0.60, 0.88) for pull-up on average. Pushdown samples reviews, among which popular products are rare, and therefore accepts popular reviews with $p \approx 0 . 8$ . Each such review contributes 50 to 400 rows that are wrong with probability 0.2. As a result, output precision drops, and pushdown meets the target on far fewer instances than pull-up (Table 5, last two rows). This is the efect of Theorem 5.11, here with a random calibration sample; Fig. 7(a) shows the variation across databases. Pull-up meets the target but is about 70 times more expensive, since it evaluates every joined row. A plan that is as cheap as pushdown and as accurate as pull-up must choose its decisions based on output quality rather than operator quality; E3 evaluates this approach.

E3: pushing the target down to tuples. To isolate the efect of fan-out, we draw confidence independently of fan-out in this experiment $( p \sim \mathrm { B e t a } ( 0 . 7 , 3 )$ with probability 0.8, else Beta(3, 0.7)).

![](images/c6c868c00f4aa0bf6f033d5f278b7abef975fb7a56ec476ceb83f99dc4963f72.jpg)  
Figure 5: Plan space of one simulated database (2,000 reviews, confidence independent of fan-out). Each point is a candidate plan: every band on a 0.05 grid under pushdown and pull-up, and the per-tuple priced policies of Theorem 8.2 under pushdown. Plans difer by up to 1, 261× in cost and from 0.00 to 1 in output quality; even among plans that meet the target, cost varies 307×. The cost-only plan is cheap but misses about a third of the true orders. Our policy meets the predicted target (realized recall 0.90, precision 0.93) at about 1/16 of the cost of escalating all decisions and 1/7 of the cost of an operator-level target.

The operator-level policy considers all bands [ℓ, h) on a grid with step 0.02, and the instance-level policy assigns actions with the per-tuple rule of Theorem 8.2. Both select the cheapest policy whose predicted output quality meets the target, using only confidences and fan-outs, and both meet the target on the realized labels (Table 6). The instance-level policy is 81% cheaper. It escalates uncertain reviews with many rows and leaves uncertain reviews with few rows to the cheap model, whereas a band has to escalate every review in its range.

E4: accuracy of the prediction. For a calibrated model, the output quality predicted from $\textstyle \sum _ { i } c _ { i } \big ( 1 - \mathrm { c o n f } _ { i } \big )$ is within about 0.01 of the realized quality on each instance (Table 7 and Fig. 7(b)). An overconfident model, whose confidences are pushed toward 0 and 1, predicts a precision of 0.98, while the realized precision is 0.87; the hidden errors are concentrated in reviews with high fanout. Contribution-weighted PPI with 100 labels removes this bias on average and reduces the per-instance error from about 0.11 to about 0.03. In summary, the predicted quality used for equivalence and optimization is accurate for calibrated models and can be corrected with a small gold sample otherwise.

![](images/6a3e919d3854f6a46f115fba2560e716bea4e1c0545e93fb9b22eefe0973a3f4.jpg)  
Figure 6: Cost of meeting an output-level target, averaged over 10 simulated databases (confidence independent of fan-out). Both policies are chosen on predicted quality and meet the target on realized labels on average. Per-tuple pricing is cheaper than the best operator-level band by a factor of 1.9 to 5.3. The gap narrows as ρ approaches 1, since then almost all uncertain tuples have to be escalated.

Table 6: E3: cheapest operator-level band vs. instance-level policy (Theorem 8.2, grid search over the two prices), both chosen on predicted output quality to meet (0.9, 0.9) and evaluated on realized quality. Workload with confidence independent of fan-out; 50 instances.
<table><tr><td>Policy</td><td>Cost</td><td>Escalations</td><td>Output recall</td><td>Output precision</td></tr><tr><td>Operator-level band</td><td>37,256</td><td>688</td><td>0.917±0.027</td><td>0.911 ±0.033</td></tr><tr><td>Instance-level (priced)</td><td>7,164</td><td>87</td><td>0.921 ±0.022</td><td>0.910 ±0.021</td></tr></table>

Limitations. The simulations assume independent decisions and a perfect judge, and, except in E4, a calibrated model. Correlated errors (e.g., one review in many pairs), judge errors on the most dificult instances (Theorem 4.7), and the calibration of real models across the populations induced by diferent plans have to be studied on a real engine.

## 10 Related Work and Positioning

The closest prior work allocates accuracy across ML predicates or guarantees per-operator quality;   
none defines output-level targets with join fan-out coupling or a semantic plan equivalence.

Expensive predicates in classical optimization. Predicate migration [Hellerstein and Stonebraker, 1993, Hellerstein, 1998] and the optimization of user-defined predicates [Chaudhuri and Shim, 1999] place costly predicates in a plan by trading their cost against their selectivity, and adaptive schemes such as eddies [Avnur and Hellerstein, 2000] and adaptive filter ordering [Babu et al., 2004] reorder them at run time. These predicates are exact, so every placement returns the same answer and only cost is at stake. This corresponds to the setting of pointwise-deterministic policies (Theorem 5.3); we study the case in which predicate placement also afects the result. Sampling-based approximate query processing such as BlinkDB [Agarwal et al., 2013] bounds the error of aggregates over samples of the data rather than the error of model decisions.

Table 7: E4: predicted vs. realized output quality of the PD band policy, averaged over 200 instances. Overconfident: logit of $p$ scaled by 2.5. PPI: contribution-weighted correction with 100 gold labels drawn in proportion to fan-out. Last column: mean absolute error of predicted precision per instance.
<table><tr><td>Decision model</td><td>Pred. recall</td><td>True recall</td><td>Pred. precision</td><td>True precision</td><td>|Error|(precision)</td></tr><tr><td>Calibrated</td><td>0.997</td><td>0.997</td><td>0.983</td><td>0.982</td><td> $0 . 0 0 8 \pm 0 . 0 0 6$ </td></tr><tr><td>Overconfident</td><td>1.000</td><td>0.994</td><td>0.981</td><td>0.872</td><td> $0 . 1 0 9 \pm 0 . 0 2 7$ </td></tr><tr><td> $\mathrm { O v e r c o n f i d e n t + P P I }$ </td><td>0.995</td><td>0.994</td><td>0.874</td><td>0.872</td><td> $0 . 0 2 8 \pm 0 . 0 2 0$ </td></tr></table>

(a) E2: output precision per database  
![](images/f99c4a60d765b6a1c7f383e555f67b2561b4ce4fec55e88d21f383819cef92ad.jpg)

(b) E4: predicted vs. realized precision  
![](images/c2376b42df9b63bcbad8e12782eb3609e9037ecceb40344b697c0ee6603feda7.jpg)  
Figure 7: (a) Each dot is one of 200 simulated databases. A band calibrated on the pushdown population often falls below the output precision target; calibrated on the pull-up population it rarely does (Theorem 5.11). (b) Predicted output precision from $\textstyle \sum _ { i } c _ { i } ( 1 - \mathrm { c o n f } _ { i } )$ against realized precision. Calibrated confidences lie on the diagonal; an overconfident model predicts about 0.98 for databases that deliver 0.87; contribution-weighted PPI with 100 labels moves the predictions back to the diagonal, with more spread.

Accuracy targets for ML predicates and proxies. Probabilistic predicates [Lu et al., 2018] split a query-level accuracy target across cheap filters placed ahead of expensive UDFs and choose among them by cost; they handle filters only, with operator-level budgets and no join fan-out. NoScope [Kang et al., 2017], Tahoma [Anderson et al., 2019], and CORE [Yang et al., 2022] build cascades or proxy models for ML predicates over video and images, and BlazeIt [Kang et al., 2019] and ABAE [Kang et al., 2021] use proxies to answer aggregation and limit queries with accuracy guarantees. SUPG [Kang et al., 2020] gives precision/recall-target selection with a proxy and statistical guarantees for a single selection, and TASTI [Kang et al., 2022] builds reusable semantic indexes that provide proxy scores across queries. These systems define accuracy for a single operator or a specific query shape and do not consider fan-out or the calibration population in plans with multiple operators.

Semantic operator systems. LOTUS [Patel et al., 2025] provides per-operator accuracy guarantees using model cascades; FrugalGPT [Chen et al., 2023] applies cascades to individual LLM prompts. Palimpzest and its optimizer Abacus [Liu et al., 2025, Russo et al., 2026] search a cost–quality Pareto frontier over physical implementations, with quality estimated empirically per operator. DocETL [Shankar et al., 2025] rewrites document-processing pipelines with an LLM agent and evaluates candidate rewrites with LLM-based validation, and CAESURA [Urban and Binnig, 2023] uses a language model to plan multi-modal queries; both assess plan quality empirically, without a model of how decision errors propagate. ThalamusDB [Jo and Trummer, 2024] supports approximate processing with error bounds and user labeling. Featurized-decomposition joins [Zeighami et al., 2026] prune semantic joins soundly and belong to our class of sound pruning. JEVDB [Wang et al., 2026] places calibrated decision models on the fast path with semantic Bloom filters and per-predicate escalation bands calibrated against an LLM judge, restores predicate pushdown as the default on cost grounds, and leaves query optimization to future work. Larch [Zhao et al., 2026] and iPDB [Kumarasinghe et al., 2026] optimize semantic predicates for cost: Larch learns a short-circuit evaluation order over a boolean tree of semantic filters to minimize LLM tokens, and iPDB pulls predict operators up the plan and reorders semantic selects inside DuckDB. Neither system defines an output-level accuracy target. Larch assumes that reordering does not change output quality, i.e., it treats each predicate’s decision as fixed. iPDB targets latency and cost; its join decomposition trades quality for speed under user control, and quality is evaluated empirically (F1).

Foundations we build on. Our results build on several lines of work. We use the dichotomy for probabilistic databases [Dalvi and Suciu, 2007, Suciu et al., 2011] for quality estimation and provenance semirings [Green et al., 2007] for decision lineage. Attribution of output quality to decisions relates to sensitivity analysis [Kanagal et al., 2011] and tuple Shapley values [Livshits et al., 2020], contributions are computed by message passing as in FAQ [Khamis et al., 2016], and our oracle-relative estimator is based on prediction-powered inference [Angelopoulos et al., 2023].

## 11 Open Problems

Our definitions cover positive SRA with bag outputs. We list several directions in which the scope should be extended.

• Aggregates and GROUP BY. A misclassification can move a tuple from one group to another, so errors are not simply false positives or false negatives. Quality should then be defined via relative error or group-level F1. For COUNT and SUM, the cost of dropping a tuple is again proportional to its fan-out.

• Negation and NOT EXISTS. A semantic false negative inside a negated subquery becomes a false positive in the output, which reverses the asymmetry between recall and precision discussed in Section 6.

• Top-k and ranking. It is open how ranking quality composes with joins.

• Multi-query optimization. Since diferent queries may have diferent targets, decisions can only be reused across queries if the confidences $p _ { i }$ are stored rather than Boolean outcomes. A condition index is a special case.

• Semantic zone maps. Per-block summaries, such as label sets or the maximum confidence, can serve for sound or approximate pruning. They are only useful if the physical layout is correlated with the semantic attribute.

• Adaptive re-optimization. A plan may need to be revised when the observed selectivity or calibration of a semantic predicate difers from the estimates.

## 12 Conclusion

In semantic query engines, the choice of a plan determines not only the cost of a query but also its result. We argued that optimizing such plans requires a precise definition of what is being optimized, and that calibrated confidences make such a definition possible. A calibrated confidence assigns an expected error to each semantic decision. Weighting these errors by each decision’s contribution to the output gives the expected output quality of a plan without labeled data, and the same computation, applied in reverse, assigns a price for output error to every base and intermediate tuple. Based on this, we defined physical plans as logical plans with decision policies, output-level quality targets, a hierarchy of plan equivalence that ends with the set of admissible plans, and the corresponding optimization problem.

Our results delimit the problem. Under pointwise-deterministic policies, accuracy does not depend on the plan, and optimization reduces to the classical case. The interaction between cost and accuracy is caused by calibrated, context-aware, or approximate components. In particular, selection pushdown, the default rewrite in current systems, is not quality-sound once bands are calibrated on the candidates of the plan itself. Expected quality can be computed eficiently under bag semantics and, under set semantics, follows the dichotomy for probabilistic databases when every relation carries a semantic predicate. Selecting tuples to drop is NP-hard, while the output target decomposes into per-tuple decisions with two prices.

Our simulations illustrate these efects on synthetic data. The next step is to measure them on real engines, models, and benchmarks. Beyond that, the open problems in Section 11, in particular aggregates and negation, correlated decisions, and adaptive re-optimization for miscalibrated confidences, remain to be addressed.

## References

Sameer Agarwal, Barzan Mozafari, Aurojit Panda, Henry Milner, Samuel Madden, and Ion Stoica. BlinkDB: Queries with bounded errors and bounded response times on very large data. In EuroSys, pages 29–42, 2013. doi: 10.1145/2465351.2465355.

Michael R. Anderson, Michael Cafarella, German Ros, and Thomas F. Wenisch. Physical representation-based predicate optimization for a visual analytics database. In ICDE, pages 1466–1477, 2019. doi: 10.1109/ICDE.2019.00132.

Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. Prediction-powered inference. Science, 382(6671):669–674, 2023. doi: 10.1126/science.adi6000.

Ron Avnur and Joseph M. Hellerstein. Eddies: Continuously adaptive query processing. In SIG-MOD, pages 261–272, 2000. doi: 10.1145/342009.335420.

Shivnath Babu, Rajeev Motwani, Kamesh Munagala, Itaru Nishizawa, and Jennifer Widom. Adaptive ordering of pipelined stream filters. In SIGMOD, pages 407–418, 2004. doi: 10.1145/1007568.1007615.

Surajit Chaudhuri and Kyuseok Shim. Optimization of queries with user-defined predicates. ACM Trans. Database Syst., 24(2):177–228, 1999. doi: 10.1145/320248.320249.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176, 2023. doi: 10.48550/arXiv.2305.05176.

Nilesh Dalvi and Dan Suciu. The dichotomy of conjunctive queries on probabilistic structures. In PODS, pages 293–302, 2007. doi: 10.1145/1265530.1265571.

Todd J. Green, Grigoris Karvounarakis, and Val Tannen. Provenance semirings. In PODS, pages 31–40, 2007. doi: 10.1145/1265530.1265535.

Joseph M. Hellerstein. Optimization techniques for queries with expensive methods. ACM Trans. Database Syst., 23(2):113–157, 1998. doi: 10.1145/292481.277627.

Joseph M. Hellerstein and Michael Stonebraker. Predicate migration: Optimizing queries with expensive predicates. In SIGMOD, pages 267–276, 1993. doi: 10.1145/170035.170078.

Saehan Jo and Immanuel Trummer. ThalamusDB: Approximate query processing on multi-modal data. Proc. ACM Manag. Data, 2(3):186:1–186:26, 2024. doi: 10.1145/3654989.

Bhargav Kanagal, Jian Li, and Amol Deshpande. Sensitivity analysis and explanations for robust query evaluation in probabilistic databases. In SIGMOD, pages 841–852, 2011. doi: 10.1145/ 1989323.1989411.

Daniel Kang, John Emmons, Firas Abuzaid, Peter Bailis, and Matei Zaharia. NoScope: Optimizing neural network queries over video at scale. PVLDB, 10(11):1586–1597, 2017. doi: 10.14778/ 3137628.3137664.

Daniel Kang, Peter Bailis, and Matei Zaharia. BlazeIt: Optimizing declarative aggregation and limit queries for neural network-based video analytics. PVLDB, 13(4):533–546, 2019. doi: 10. 14778/3372716.3372725.

Daniel Kang, Edward Gan, Peter Bailis, Tatsunori Hashimoto, and Matei Zaharia. Approximate selection with guarantees using proxies. PVLDB, 13(11):1990–2003, 2020. doi: 10.14778/3407790. 3407804.

Daniel Kang, John Guibas, Peter Bailis, Tatsunori Hashimoto, Yi Sun, and Matei Zaharia. Accelerating approximate aggregation queries with expensive predicates. PVLDB, 14(11):2341–2354, 2021. doi: 10.14778/3476249.3476285.

Daniel Kang, John Guibas, Peter D. Bailis, Tatsunori Hashimoto, and Matei Zaharia. TASTI: Semantic indexes for machine learning-based queries over unstructured data. In SIGMOD, pages 1934–1947, 2022. doi: 10.1145/3514221.3517897.

Mahmoud Abo Khamis, Hung Q. Ngo, and Atri Rudra. FAQ: Questions asked frequently. In PODS, pages 13–28, 2016. doi: 10.1145/2902251.2902280.

Udesh Kumarasinghe, Tyler Liu, Ahmed R. Mahmood, Chunwei Liu, and Walid G. Aref. iPDB – optimizing semantic SQL queries. arXiv preprint arXiv:2601.16432, 2026. doi: 10.48550/arXiv. 2601.16432. Demo version: PVLDB 19(12):4782–4785, 2026, doi:10.14778/3827998.3828121.

Jiale Lao, Andreas Zimmerer, Olga Ovcharenko, Tianji Cong, Matthew Russo, Gerardo Vitagliano, Michael Cochez, Fatma Özcan, Gautam Gupta, Thibaud Hottelier, H. V. Jagadish, Kris Kissel, Sebastian Schelter, Andreas Kipf, and Immanuel Trummer. SemBench: A benchmark for semantic query processing engines. PVLDB, 19(8):1754–1767, 2026. doi: 10.14778/3811243.3811249.

Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baile Chen, Zui Chen, Michael Franklin, Tim Kraska, Samuel Madden, Rana Shahout, and Gerardo Vitagliano. Palimpzest: Optimizing AI-powered analytics with declarative query processing. In CIDR, 2025. URL https: //www.vldb.org/cidrdb/papers/2025/p12-liu.pdf.

Ester Livshits, Leopoldo Bertossi, Benny Kimelfeld, and Moshe Sebag. The Shapley value of tuples in query answering. In ICDT, volume 155 of LIPIcs, pages 20:1–20:19, 2020. doi: 10.4230/ LIPIcs.ICDT.2020.20.

Yao Lu, Aakanksha Chowdhery, Srikanth Kandula, and Surajit Chaudhuri. Accelerating machine learning inference with probabilistic predicates. In SIGMOD, pages 1493–1508, 2018. doi: 10. 1145/3183713.3183751.

Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos Guestrin, and Matei Zaharia. Semantic operators and their optimization: Enabling LLM-based data processing with accuracy guarantees in LOTUS. PVLDB, 18(11):4171–4184, 2025. doi: 10.14778/3749646. 3749685.

Matthew Russo, Chunwei Liu, Sivaprasad Sudhir, Gerardo Vitagliano, Michael Cafarella, Tim Kraska, and Samuel Madden. Abacus: A cost-based optimizer for semantic operator systems. PVLDB, 19(5):1060–1073, 2026. doi: 10.14778/3796195.3796215.

Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G. Parameswaran, and Eugene Wu. DocETL: Agentic query rewriting and evaluation for complex document processing. PVLDB, 18 (9):3035–3048, 2025. doi: 10.14778/3746405.3746426.

Dan Suciu, Dan Olteanu, Christopher Ré, and Christoph Koch. Probabilistic Databases. Synthesis Lectures on Data Management. Morgan & Claypool, 2011. doi: 10.2200/ S00362ED1V01Y201105DTM016.

Matthias Urban and Carsten Binnig. CAESURA: Language models as multi-modal query planners. arXiv preprint arXiv:2308.03424, 2023. doi: 10.48550/arXiv.2308.03424.

Zhengle Wang, Hanxu Yan, Fuheng Zhao, and Chunwei Liu. Prune first, decide fast: Scalable semantic query processing with JEVDB. arXiv preprint arXiv:2610.02046, 2026. doi: 10.48550/ arXiv.2610.02046.

Zhihui Yang, Zuozhi Wang, Yicong Huang, Yao Lu, Chen Li, and X. Sean Wang. Optimizing machine learning inference queries with correlative proxy models. PVLDB, 15(10):2032–2044, 2022. doi: 10.14778/3547305.3547310.

Mihalis Yannakakis. Algorithms for acyclic database schemes. In VLDB, pages 82–94, 1981.

Sepanta Zeighami, Shreya Shankar, and Aditya Parameswaran. Featurized-decomposition join: Low-cost semantic joins with guarantees. PVLDB, 19(11):3048–3061, 2026. doi: 10.14778/ 3836663.3836672.

Fuheng Zhao, Pawel Liskowski, Zihan Li, Benjamin Han, Puxuan Yu, Varich Boonsanong, Dimitris Tsirogiannis, and Anupam Datta. Larch: Learned query optimization for semantic predicates. arXiv preprint arXiv:2606.07923, 2026. doi: 10.48550/arXiv.2606.07923.