# Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability

Meihui Zhong<sup>1</sup>, Wenxin Tai<sup>1</sup>, Ting Zhong<sup>1,3</sup>, Fan Zhou<sup>1,2,3∗</sup>

<sup>1</sup> University of Electronic Science and Technology of China <sup>2</sup> Intelligent Digital Media Technology, Key Laboratory of Sichuan Province <sup>3</sup> Kashi Institute of Electronics and Information Industry

## Abstract

Conformal prediction offers a distribution-free coverage guarantee, making it especially attractive for clinical applications. Standard conformal prediction, however, provides such guarantees only at the population level, and its prediction sets can exhibit coverage disparities across clinically important subgroups. A natural remedy is to calibrate within predefined groups. However, this can require access to sensitive subgroup attributes and is prone to a worst-group bottleneck: protecting the most difficult subgroup can inflate prediction sets for all, increasing cognitive burden on decision makers. To this end, we propose Stochastic Grouping Conformal Prediction (SGCP), a conformal framework for subgroup-reliable uncertainty quantification. It learns a stochastic grouping map that allows each sample to draw calibration information from others with similar calibration behavior, yielding a local score law that boosts reliability across subpopulations. We prove that SGCP retains the standard coverage guarantee. Experiments on synthetic and realworld benchmarks show that it consistently reduces subgroup coverage gaps while achieving smaller or comparable prediction set sizes relative to existing baselines.

## 1 Introduction

Conformal prediction (CP) has emerged as an attractive framework for medical AI, because it converts any predictive model into a prediction set with finite-sample coverage guarantees under minimal assumptions [17, 5, 16]. Consider training data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { \bar { n } }$ and a future test point $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ where each feature $X _ { i } \in { \mathcal { X } }$ lies in the feature space and response $Y _ { i } \in \mathcal { V }$ lies in the label space. The goal is to construct a prediction set ${ \widehat { C } } _ { n } ( X _ { n + 1 } ) \subseteq { \mathcal { Y } }$ that contains the unknown test response $Y _ { n + 1 }$ with high probability. The appeal of distribution-free prediction is that this guarantee can be obtained without imposing strong assumptions on the underlying data distribution [24, 22, 1], namely,

$$
\mathbb { P } \{ Y _ { n + 1 } \in \widehat { C } _ { n } ( X _ { n + 1 } ) \} \geq 1 - \alpha .\tag{1}
$$

While this guarantee is highly appealing, it has an important limitation. The coverage in Eq. (1) is averaged over the calibration data and the future test point, thus providing validity only at the population level [10, 21]. In clinical settings, this means that a nominal 90% coverage guarantee may hold on average over all patients, while still failing to provide equally reliable uncertainty estimates for particular demographic groups, disease subtypes, or other clinically meaningful subpopulations.

This concern is tied to fairness in healthcare, where the costs of unreliable uncertainty often fall disproportionately on underrepresented patient populations [9, 4]. Differences in prevalence, data quality, and feature distributions across populations can induce distinct error patterns and nonconformity score distributions [14]. Thus, population-level calibration is prone to masking subgroup failures by averaging under-coverage in some groups with over-coverage in others. In medical applications, this imbalance matters, as what we care about is not only whether uncertainty is valid on average, but whether it remains reliable for the subgroups for whom prediction errors are most costly.

Prediction Set Size CluCP  
![](images/1759b8adc08e39da106975629dcc433961779675a5e74af6bc553435d5045d8b.jpg)  
Marginal CP

![](images/890bf0b37e613fd4c6b393c87d963e679f7657f8291ab47ecb929831aea3c51f.jpg)  
Prediction Set Size RLCP

![](images/4436c92044c4bf9b04dac8bf9d0dd884d31dbf7b6401de70e71102a4e23c52a5.jpg)

![](images/e0d425f8d6c843422ad7d3b07fc3e1987538e8e58ac2dc2e4bbfccdf58e9abde.jpg)  
AFCP

![](images/b1fcbd91063704f8d5522821e6c66957f6bec10e8b652390d221cb8860569457.jpg)  
Prediction Set Size FaReG  
Figure 1: Tradeoff between subgroup coverage and prediction set size on MIMIC-IV [13] $( 1 - \alpha =$ 0.9). Results are shown for RLCP [12], CluCP [8], AFCP [29], and FaReG [27]. The red dashed line marks half of the total number of classes. Several fairness-aware CP methods improve sensitive-group coverage, but often do so with much larger prediction sets, making them less informative in practice.

We empirically observe exactly this tension, as shown in Figure 1. Standard conformal prediction achieves the desired overall coverage, but still exhibits lower coverage on several clinically important subpopulations, suggesting that marginal validity may ‘average out’ group-specific uncertainty. A natural response is to calibrate separately within each group. However, methods [20, 27] that target finer grained group reliability often become increasingly conservative, resulting in larger prediction sets. Such enlarged sets may be less informative in practice and may even hinder downstream decision-making by imposing excessive cognitive burden [28, 7, 6]. One may go even further and calibrate at the individual-sample level, as in localized conformal prediction [12, 20]. Yet our results show that localization, while sometimes improving coverage on subgroups, often incurs substantially larger prediction sets and, more importantly, greater instability across random seeds and data splits.

In sum, these observations pose two coupled challenges. First, the heterogeneity that matters for calibration is often latent: it may not be directly observable at inference, nor align with pre-specified sensitive groups. Second, the groups that most need tailored calibration are often those with limited sample sizes, making fine grained localized calibration conservative and statistically unstable [8]. More broadly, we seek a practical middle ground between global calibration, which can wash out subgroup-specific uncertainty, and calibration strategies that rely too heavily on known groups or local neighborhoods, which can become conservative or unstable when calibration data are limited.

Motivated by this gap, we propose stochastic grouping conformal prediction (SGCP), a fairnessaware conformal prediction method that improves subgroup reliability without relying on known sensitive attributes. Rather than asking coverage to hold for each individual test point, we relax the requirement to a weaker but still meaningful sense – coverage over samples that exhibit similar calibration behavior. In other words, exact 90% coverage may be unattainable for one patient in isolation, yet it may still be possible to achieve more reliable coverage on patients whose scores reflect similar uncertainty patterns and calibration difficulty.

To achieve this, SGCP learns a latent calibration structure that organizes samples according to uncertainty-relevant score behavior. Atop this structure, it learns a stochastic grouping map that determines how each sample should draw calibration information from a small set of latent calibration components. This yields a sample-adaptive local score law: calibration is tailored to each sample, but estimated through structured sharing rather than from a small group or neighborhood alone. Statistically, this design strikes a balance between flexibility and stability. It is flexible enough to capture heterogeneous score behavior across samples, yet structured enough to be reliably estimated from finite data while preserving conformal validity. As a result, SGCP can improve subgroup reliability beyond global calibration without relying on sparse subgroup calibration data. Empirically, across fairness-sensitive clinical and public prediction tasks, SGCP improves coverage on clinically meaningful subgroups while maintaining valid overall coverage and efficient prediction sets.

Our contributions are summarized as follows. First, we propose SGCP, a novel fairness-aware conformal prediction method. It learns a local score law through stochastic grouping, thereby approximating the score distribution of test points. This improves subgroup reliability without relying on rigid predefined groups. Second, we prove that SGCP retains finite-sample marginal validity. We further show that subgroup coverage gaps decrease when the learned local score law better captures the score behavior of meaningful subpopulations. Third, through experiments on synthetic and medical datasets, we demonstrate that SGCP achieves a favorable tradeoff among overall coverage, subgroup reliability, and prediction set efficiency. We expect SGCP to enable more effective and trustworthy decision-making in high-risk medical applications. Code and supplementary materials are included in the anonymous supplementary upload.

## 2 Preliminaries

We consider multiclass conformal classification with an exchangeable dataset $\mathcal { D } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { N }$ where $X _ { i } \in { \mathcal { X } }$ and $Y _ { i } \in \mathcal { V } = \{ 1 , . . . , C \}$ . Let $f _ { \theta } : \mathcal { X } \xrightarrow { } \Delta ^ { C - 1 }$ be a base classifier, with $f _ { \theta } ^ { y } ( x )$ denoting the predicted probability of class y. We use the nonconformity score $s ( x , y ) = 1 - \dot { f } _ { \theta } ^ { \check { y } } ( \dot { x } )$ For an observed example $( X _ { i } , Y _ { i } )$ , we write $s _ { i } = s ( x _ { i } , y _ { i } )$ , and it measures how inconsistent a point is with the true label. Given a training set $\mathcal { D } _ { \mathrm { t r a } }$ and a calibration set $\mathcal { D } _ { \mathrm { c a l } }$ , we are interested in constructing, for a test point $X _ { n + 1 }$ , a confidence set ${ \widehat { C } } ( x ) \subseteq y$ such that $\mathbb { P } \{ Y _ { n + 1 } \in \widehat { C } ( X _ { n + 1 } ) \} \geq$ $1 - \alpha$ , for a miscoverage level α, without distributional assumptions beyond exchangeability.

In fairness-sensitive applications, the central issue is not merely whether the prediction set is valid on average, but rather how coverage and set size are distributed across heterogeneous subpopulations. Ideally, one desire distribution-free conditional coverage – that is,

$$
\mathbb { P } \{ Y \in { \widehat { C } } ( x ) ~ | ~ X = x \} \geq 1 - \alpha \qquad P _ { X ^ { - } } { \mathrm { a . e . ~ } } x .\tag{2}
$$

which would mitigate uneven coverage across patient demographics, disease patterns, and other meaningful groups. However, non-trivial distribution-free conditional coverage is impossible, since finite samples cannot ensure tight, valid prediction sets for arbitrary conditioning events [10, 12].

We typically classify existing approaches into two broad types. (i) Group-wise calibration. A natural response to subgroup miscoverage is to calibrate separately across groups [17, 25], a type we term Group-wise calibration CP. If $G ( { \bar { x } } ) \in \{ 1 , \dots , K \}$ denotes a group label, this yields a group-specific threshold $\hat { q } _ { k } = \mathrm { Q u a n t i l e } \Big ( \{ s _ { i } : ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } , G ( x _ { i } ) = k \} , 1 - \alpha \Big )$ , and outputs

$$
\widehat { C } _ { \mathrm { G W } } ( x ) = \{ y \in \mathcal { V } : \ s ( x , y ) \leq \hat { q } _ { G ( x ) } \} .\tag{3}
$$

In principle, $\operatorname { E q . } \left( 3 \right)$ reduces the mismatch induced by a single global threshold. In practice, however, one issue is that the relevant groups are often not known a priori, and may not align with pre-specified sensitive attributes. Another is that finer grouping fragments the calibration set, leaving some groups with too few samples for calibration, and making the empirical quantiles $\hat { q } _ { k }$ statistically unstable [8].

This group-wise perspective also includes methods that learn or adapt groups before calibration. AFCP [29], for example, considers groups defined by sensitive attributes and makes the group selection itself adaptive. Let $\phi ( x , A )$ denote the projection of x onto a selected sensitive-attribute set A. For instance, a subgroup “males with a bachelor’s degree” can be written as $\phi ( { \boldsymbol { x } } , \{ { \boldsymbol { k } } , \ell \} ) \in$ $[ M _ { k } ] \times [ M _ { \ell } ]$ . For each test point, AFCP selects either one sensitive attribute or none, $\hat { A } ( X _ { n + 1 } ) \in$ $\{ \emptyset , \{ 1 \} , \dots , \{ K \} \}$ , and targets P $\begin{array} { r } { \left[ Y _ { n + 1 } \in \hat { C } ( X _ { n + 1 } ) \mid \phi \big ( X _ { n + 1 } , \hat { A } ( X _ { n + 1 } ) \big ) \right] \geq 1 - \alpha } \end{array}$ . This reduces to marginal coverage when ${ \hat { A } } ( X _ { n + 1 } ) = \varnothing .$ . Operationally, AFCP combines a marginal conformal set with an equalized-coverage set for each candidate label under the selected attribute. Although less conservative than exhaustive calibration over all attribute combinations, AFCP remains tied to discrete sensitive attributes. Its repeated leave-one-out calibration on pseudo-test points can also be computationally expensive and noisy in sparse-group regimes.

FAREG [27] further improves group discovery by learning groups in a latent representation space rather than enumerating them from raw input features. This expands the class of detectable subgroups, including those defined by nonlinear feature combinations. However, FAREG still frames fairness as subgroup discovery: it identifies explicit subgroups with coverage disparity and enforces coverage over them. It is therefore most suitable when heterogeneity is well captured by identifiable subgroups. It may, however, become conservative in some cases.

(ii) Localized calibration. Localized calibration, as in RLCP [12], avoids explicit group partitions by using nearby samples as the reference set for each test point. RLCP argues that heterogeneity relevant to calibration may be highly sample-specific. Let $\mathfrak { X } _ { x } \ = \ \{ \mathcal { X } \ \in \mathfrak { X } : \ x \ \in \ \mathcal { X } \}$ denote the candidate neighborhoods containing x. A localized threshold can be written as ${ \hat { q } } _ { \mathrm { l o c } } ( x ) =$ $\operatorname* { s u p } _ { \mathcal { X } \in \mathcal { X } _ { : } }$ Quantile $\left( \{ s _ { i } : ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } , \ x _ { i } \in \mathcal { X } \} , 1 - \alpha \right)$ , with prediction set

![](images/ad7c35ff71965326518e6f3aecda55cd7f1979d3eb98751509a0b120d0dc20e6.jpg)  
Figure 2: Overview of SGCP. SGCP learns a local score law by mapping x to a stochastic grouping map w(x) over latent calibration components $\{ F _ { k } \} _ { k = 1 } ^ { K }$ . Candidate labels are transformed by this local law and kept when their calibrated scores fall below the conformal threshold.

$$
\widehat { C } _ { \mathrm { l o c } } ( x ) = \{ y \in y : s ( x , y ) \leq \widehat { q } _ { \mathrm { l o c } } ( x ) \} .\tag{4}
$$

Compared with group-wise calibration, Eq. (4) avoids rigid group boundaries and adapts to the region around the test point. Yet the supremum over candidate neighborhoods requires the prediction set to cover the worst local subset containing x, which may be overly conservative and yield unnecessarily large sets, particularly when calibration samples are scarce. Moreover, locality is typically defined via feature space similarity, which need not align with the actual error structure most relevant for calibration. RLCP may introduces additional randomness into the prediction intervals, leading to inconsistent outputs across independent runs. Other related works are discussed in Appendix F.

The preceding discussion suggests that the key challenge is to improve subgroup reliability without losing statistical stability or producing overly large prediction sets. We therefore seek a method that can capture calibration heterogeneity while promoting informative prediction sets.

## 3 Method

## 3.1 From subgroup reliability to local score laws

Our starting point is that conformal calibration is ultimately governed by the distribution of the nonconformity score. Accordingly, we take the sample-specific score distribution $F _ { x } ^ { \star } ( s ) : = \mathbb { P } ( S \leq$ $s \mid X = x )$ , where S denotes the nonconformity score, as the oracle object for reliable calibration to each subgroup. If we could estimate $F _ { x } ^ { \star }$ accurately, then each sample could be calibrated according to its own local score law. That said, the difficulty is that directly learning an unconstrained conditional distribution $F _ { x } ^ { \star }$ for every x is data-hungry and unstable in finite samples, especially when the relevant heterogeneity is latent and high-dimensional. A single test point rarely provides enough local calibration evidence for reliable estimation. This motivates the central question behind SGCP: can other samples help a target point estimate its local score distribution without resorting to a single global calibration? SGCP addresses this by replacing the unconstrained family $\{ F _ { x } ^ { \star } : x \in \mathcal { X } \}$ with a structured mixture family. Concretely, SGCP approximates the oracle local score law by

$$
F _ { x } ^ { \star } ( s ) \approx { \widehat { F } } ( s \mid x ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) F _ { k } ( s ) ,\tag{5}
$$

where $F _ { 1 } , \ldots , F _ { K }$ are latent calibration components and $w ( x )$ is a sample-related stochastic grouping map. Thus, each sample has its own effective local score law, but that law is not estimated from scratch. It is represented as a weighted combination of latent calibration components learned from related samples. This makes SGCP more expressive than discrete group-wise calibration, while still allowing samples with similar calibration behavior to share evidence through the components. Posterior averaging further stabilizes the resulting local score law (see Appendix B for details). Figure 2 illustrates how stochastic grouping induces a local score law for each sample.

The key question in Eq. (5) is therefore how to determine, for each sample x, which other samples should be regarded as ‘related’. While there are many ways to define relevance around a point, such as finding similar points through feature embedding or clustering, recall that the ideal object for subgroup reliable calibration is the sample-specific score distribution $F _ { x } ^ { \star }$ in Eq. (5). Accordingly, the relevant notion of locality is not simply geometric proximity in the input space, but similarity in calibration relevant score behavior. So we want to identify samples that are governed by similar score behavior. This motivates us to propose the latent calibration state and stochastic grouping map. Together, they determine how each sample should transfer calibration evidence from the latent calibration components. We next formalize these two components and the resulting local score law.

## 3.2 Learning a differentiable local score law

Latent calibration state. Let $h _ { \theta } ( x ) \in \mathbb { R } ^ { d }$ denote the feature representation produced by the backbone classifier, and let $f _ { \theta } ( x ) \in \dot { \Delta ^ { C - 1 } }$ denote the corresponding predictive probability vector. To capture heterogeneity in score behavior beyond predefined groups, we associate each sample with a latent calibration state z. We model this state via a posterior distribution

$$
q _ { \phi } ( z \mid x ) = \mathcal { N } \Big ( \mu _ { \phi } \big ( h _ { \theta } ( x ) \big ) , \mathrm { d i a g } \big ( \sigma _ { \phi } ^ { 2 } \big ( h _ { \theta } ( x ) \big ) \big ) \Big ) ,\tag{6}
$$

where $\mu _ { \phi } ( \cdot )$ and $\sigma _ { \phi } ( \cdot )$ are learned functions of the backbone representation. This stochastic formulation allows calibration behavior to remain ambiguous even when the observed feature vector is fixed. Samples with similar predictions may still display mixed or uncertain score behavior, and such ambiguity is naturally represented through posterior uncertainty in latent space.

Stochastic grouping map. For a latent draw $z \sim q _ { \phi } ( z \mid x )$ , we obtain a simplex-valued membership vector $\tilde { w } ( x , z ) = \bigl ( \tilde { w } _ { 1 } ( x , z ) , \ldots , \tilde { w } _ { K } ( x , z ) \bigr ) \in \Delta ^ { K - 1 }$ through a learnable differentiable mapping from the latent state to the probability simplex. A single latent draw represents only one possible realization of the posterior calibration state and therefore only one possible decomposition of the sample’s local score law. What we seek, however, is a stable description of how the sample is associated with the latent calibration components. Accordingly, SGCP defines the stochastic grouping map by averaging memberships over the posterior distribution. Let $z ^ { ( 1 ) } , \dots , z ^ { ( M ) } \overset { \mathrm { i . i . d . } } { \sim } q _ { \phi } ( z \mid x )$ We define the posterior-averaged membership of sample x on the k-th latent calibration component

$$
w _ { k } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { z } \sim \boldsymbol { q } _ { \phi } ( \cdot | \boldsymbol { x } ) } [ \tilde { w } _ { k } ( \boldsymbol { x } , \boldsymbol { z } ) ] \approx \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \tilde { w } _ { k } \big ( \boldsymbol { x } , \boldsymbol { z } ^ { ( m ) } \big ) , \qquad k = 1 , \dots , K .\tag{7}
$$

This yields the stochastic grouping map $w ( x ) = \bigl ( w _ { 1 } ( x ) , \ldots , w _ { K } ( x ) \bigr ) \in \Delta ^ { K - 1 }$ , which determines how the latent calibration components are combined to form the learned local score law.

Aggregate local score law. Each latent calibration component $F _ { k } : [ 0 , 1 ] \to [ 0 , 1 ]$ is a monotone mapping that represents one recurring calibration pattern in the score space. When needed, we write $f _ { k }$ for the corresponding density. Given the membership weight $w _ { k } ( x )$ , SGCP defines the aggregate local score law at x as

$$
{ \widehat { F } } ( s \mid x ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) F _ { k } ( s ) ,\tag{8}
$$

with corresponding local density $\begin{array} { r } { \widehat { f } ( s \mid x ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) f _ { k } ( s ) } \end{array}$ . Equation (8) defines the central local score-law estimator in SGCP. It replaces the infeasible task of estimating an unconstrained $F _ { x } ^ { \star }$ for every test point with a structured family generated by a small number of latent calibration components. In our implementation, each $F _ { k }$ is parameterized as a monotone differentiable mapping.

We next explain why local score-law normalization matters for subgroup reliability. If the local score law were known, the probability integral transform would map true-label nonconformity score onto a common percentile scale. The following proposition formalizes this property (proof in Appendix A.1).

Proposition 1. Let $V ( x , y ) : = \widehat { F } \big ( s ( x , y ) \mid x \big )$ denote the transformed score. Suppose that the learned local score law coincides with the oracle local score law, namely, for P<sub>X</sub>-almost every x, ${ \widehat { F } } ( t \mid x ) = F _ { x } ^ { \star } ( t ) : = \mathbb { P } ( S \leq t \mid X = x ) , \forall t \in [ 0 , 1 ]$ . Assume furthermore that, for $P _ { X ^ { - } } a . e . \ x ,$ , the conditional score distribution $S \mid X = { \dot { x } }$ is continuous. Then,for any measurable subgroup $G \subseteq { \mathcal { X } }$ with $\mathbb { P } ( X \in G ) > 0$

$$
V ( X , Y ) \mid \{ X \in G \} \sim \operatorname { U n i f } ( 0 , 1 ) .
$$

```latex
Algorithm 1 Procedure for SGCP
Require: Training data $\mathcal { D } _ { \mathrm { t r a } } ,$ calibration data $\mathcal { D } _ { \mathrm { c a l } }$ , test input $X _ { n + 1 }$ , nonconformity score $s ( \cdot , \cdot )$ , K latent
calibration components, M posterior samples, hyperparameters $\lambda , \beta ,$ miscoverage level α
Ensure: Prediction set $\widehat { C } _ { \alpha } ( X _ { n + 1 } )$
1: Split $\mathcal { D } _ { \mathrm { t r a } }$ into $\mathcal { D } _ { 0 }$ and $\mathcal { D } _ { 1 }$
2: Fit SGCP: on $\mathcal { D } _ { 0 } ,$ learn $f _ { \theta }$ , latent-state posterior $q _ { \phi } ( z \mid x )$ in Eq. (6), and stochastic grouping map $\boldsymbol { \mathscr { x } } ( \boldsymbol { x } )$ in
Eq. $( 7 ) ;$ on $\mathcal { D } _ { 1 } .$ , fit latent calibration components $\{ F _ { k } \} _ { k = 1 } ^ { K }$ and local score law ${ \widehat { F } } ( \cdot \mid x )$ in Eq. (8).
3: Calibration: for each $( X _ { i } , Y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } ,$ , compute $s _ { i } = s ( X _ { i } , Y _ { i } )$ and $V _ { i } = \widehat { F } ( s _ { i } \mid X _ { i } ) .$
4: Quantile: sort $\{ V _ { i } : \ \dot { ( } X _ { i } , Y _ { i } ) \in \mathcal { D } _ { \operatorname { c a l } } \}$ as $V _ { ( 1 ) } \leq \cdots \leq V _ { ( m ) }$ , where $m = | \mathcal { D } _ { \mathrm { c a l } } |$ , and set $\widehat { q _ { 1 - \alpha } } =$
$V _ { ( \lceil ( m + 1 ) ( 1 - \alpha ) \rceil ) }$ , with fallback $\widehat { q } _ { 1 - \alpha } = 1 \mathrm { i f } \left[ ( \stackrel { \cdot } { m } + 1 ) ( 1 - \alpha ) \right] > m$
5: Test transform: for each candidate label $y \in \mathcal { V } ,$ compute $V _ { n + 1 } ^ { y } = \widehat { F } \big ( s ( X _ { n + 1 } , y ) \mid X _ { n + 1 } \big )$
6: Return: ${ \widehat { C } } _ { \alpha } ( X _ { n + 1 } ) = \left\{ y \in { \mathcal { Y } } : V _ { n + 1 } ^ { y } \leq { \widehat { q } } _ { 1 - \alpha } \right\}$
```

Proposition 1 explains why local score law learning can remove subgroup reliability inconsistency in the oracle case. In practice, SGCP learns this transform approximately, using the following objective.

Learning objective. We split the training data into $\mathcal { D } _ { \mathrm { t r a } } = \mathcal { D } _ { 0 } \cup \mathcal { D } _ { 1 }$ . The split $\mathcal { D } _ { 0 }$ is used to learn the backbone predictor, the latent-state posterior, and the stochastic grouping map, while $\mathcal { D } _ { 1 }$ is used to fit the latent calibration components and the learned local score law. An additional independent calibration set $\mathcal { D } _ { \mathrm { c a l } }$ is reserved for split conformal calibration. For each $( X _ { i } , Y _ { i } ) \in { \mathcal { D } } _ { 1 }$ , we compute the true-label nonconformity score $\overset { \cdot } { s } _ { i } = s ( X _ { i } , Y _ { i } )$ , the membership weights $w _ { i k } = w _ { k } ( X _ { i } )$ , and the component-wise transformed score $C _ { i k } = F _ { k } ( s _ { i } )$ . Since $F _ { k }$ is monotone, $C _ { i k }$ can be viewed as the percentile position of $s _ { i }$ under the k-th latent calibration component.

The training objective encourages three complementary properties. First, within each component, the weighted distribution of transformed scores should be close to $\mathrm { U n i f } ( 0 , 1 )$ . This yields the rankuniformization loss $\mathcal { L } _ { \mathrm { r a n k } }$ . Second, the induced local mixture density ${ \widehat { f } } ( s _ { i } \mid X _ { i } ) = \sum _ { k = 1 } ^ { K } w _ { i k } f _ { k } ( s _ { i } )$ should explain the observed true-label scores, yielding the density-fitting loss $\mathcal { L } _ { \mathrm { n l l } }$ . Finally, the stochastic grouping map should be informative: it should use the latent components diversely across the population while retaining sample-specific membership weights. We encourage this by maximizing the empirical mutual information between samples and a latent component variable $\dot { Z } ,$ denoted by $\widehat { I } ^ { \mathrm { n o r m } } ( X ; Z )$ , where the membership weights define $q _ { \phi } ( Z = k \mid X _ { i } ) = w _ { i k }$

Combining these terms, we optimize

$$
\mathcal { L } _ { \mathrm { S G C P } } = \mathcal { L } _ { \mathrm { r a n k } } + \lambda \mathcal { L } _ { \mathrm { n l l } } - \beta \widehat { I } ^ { \mathrm { n o r m } } ( X ; Z ) ,\tag{9}
$$

where λ controls the strength of local density fitting and $\beta$ controls the strength of stochastic grouping information maximization. Detailed parameterizations and explicit loss definitions are deferred to Appendix B. Algorithm 1 summarizes the overall procedure.

The following result, proved in Appendix A.2, shows that SGCP retains the standard finite-sample marginal validity guarantee of split conformal prediction, which contains the true label with probability at least $1 - \alpha$ . This guarantee does not depend on how accurately the local score law is learned, and the prediction set $\widehat { C } _ { \alpha } ( X _ { n + 1 } )$ remains valid.

Proposition 2. Under the standard split-conformal exchangeability assumption, the prediction set returned by SGCP satisfies $\mathbb { P } \Big \{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big \} \geq 1 - \alpha$

Marginal validity alone does not explain subgroup reliability. A further question is how the learned transform affects subgroup coverage. The following (proved in Appendix A.3) shows that subgroup coverage is controlled by the mismatch between subgroup and overall transformed-score distributions. In the oracle property of Proposition 1, this mismatch vanishes for every subgroup. In practice, SGCP seeks to reduce it by learning a local score law transform through stochastic grouping. Appendix A.4 shows that uniformly approximating the oracle score law directly controls this mismatch.

Proposition 3. For any measurable subgroup G with $\mathbb { P } ( X \in G ) > 0 ,$ , let $\Delta _ { G }$ denote the Kolmogorov distance between the distribution of $V ( X , { \dot { Y } } )$ conditional on $X \in G$ and its marginal distribution. Under the assumptions ofProposition 2 and a continuous marginal distribution of $\bar { V ( X , Y ) }$ , subgroup coverage is lower bounded by $\begin{array} { r } { \mathbb { P } \Big \{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \ | \ X _ { n + 1 } \in G \Big \} \geq 1 - \alpha - \Delta _ { G } . } \end{array}$

## 4 Numerical Experiments

Benchmarks and preprocessing. We evaluate SGCP on four benchmarks. For each dataset, let (X, Y, A) denote the feature vector, class label, and sensitive attributes, respectively. Below, we briefly describe how each dataset is processed and constructed for our evaluation.

(i) We generate synthetic (SYN) multiclass cardiopulmonary diagnosis benchmark with six disease categories, viz. Hypertension, Coronary Artery Disease, Heart Failure, Arrhythmia, Valvular Heart Disease, and Peripheral Artery Disease. Each sample contains four grouping attributes–gender, risk phenotype, age, and region–and six non-sensitive physiological variables. To mimic intersectional heterogeneity, individuals satisfying the XNOR relation between biological sex and risk phenotype, $\mathrm { e . g }$ ., Female with Type I or Male with Type II, are assigned higher label uncertainty. This subgroup is therefore harder to classify and serves as the disadvantaged population. (ii) Nursery [19] is an admission-decision dataset for ranking nursery school applications, where family, socioeconomic, social, and health-related factors make it well suited for evaluating subgroup reliability. We use four priority ranks as targets and five categorical grouping attributes. Following [29, 27], we simulate controllable algorithmic bias by downsampling the subgroup defined by parents to 10%, adding label noise, and rounding labels to the nearest valid class. (iii) MIMIC-IV [13] is a real-world clinical tabular benchmark. We employ three binary grouping attributes, i.e., race, gender, and insurance status, for subgroup analysis, along with non-sensitive demographic and administrative covariates. We study two tasks: in-hospital mortality prediction and ICU-delay classification, where delays are grouped into short $( \leq 4$ hours), moderate (4-12 hours), and prolonged (> 12 hours) or no ICU admission. (iv) BACH [2] is a four-class histopathology benchmark for breast cancer diagnosis, viz. Normal, Benign, In Situ Carcinoma, and Invasive Car2cinoma. Since demographic attributes are unavailable, we use the class label as a proxy grouping variable, setting $A : = Y$ Images are encoded by a visual backbone as $X _ { i } = \Phi ( I _ { i } ) \mathbf { \bar { \Psi } } \in \bar { \mathbb { R } ^ { d } }$ . Subgroup heterogeneity arises from visual variability across tissue and potential acquisition bias [15], so the class-conditional score laws $F _ { c } ( s ) : = \mathbb { P } ( S \subseteq s \mid Y = c )$ may differ substantially across classes.

Across all datasets, we randomly split data into training, calibration, and test sets; for SGCP, the training set is further divided equally into $\mathcal { D } _ { 0 }$ and $\mathcal { D } _ { 1 }$ . Sensitive attributes are used only for fairness evaluation and never as model inputs. Preprocessing details are given in Appendix D.1.

We benchmark SGCP against AFCP [29], Clustered CP [8], FaReG [27], RLCP [12], marginal CP, Partial Equalized [20], and Exhaustive Equalized. Partial Equalized protects one sensitive attribute at a time and unions the resulting sets, while Exhaustive Equalized protects all candidate attributes jointly as an explicit subgroup-equalization baseline. All methods use the same backbone and nonconformity score. Implementation details and hyperparameters are provided in Appendix D.2. See the Appendix E for additional experiments.

Evaluation criteria. We evaluate all methods in terms of overall validity, subgroup reliability, and efficiency. Let $\{ ( X _ { i } ^ { \prime } , Y _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { N ^ { \prime } }$ be the test set, and let $\mathcal { C } ( X _ { i } ^ { \prime } )$ denote the prediction set at miscoverage level α. We report empirical overall coverage $\begin{array} { r } { \mathrm { C o v . } = \frac { 1 } { N ^ { \prime } } \sum _ { i = 1 } ^ { N ^ { \prime } } { \bf 1 } \{ Y _ { i } ^ { \prime } \in \mathcal { C } ( X _ { i } ^ { \prime } ) \} } \end{array}$ , which should be close to the target level $1 - \alpha$ . Efficiency is measured by the average set size $\mathrm { A v g S i z e = }$ $\textstyle { \frac { 1 } { N ^ { \prime } } } \sum _ { i = 1 } ^ { N ^ { \prime } } | { \mathcal { C } } ( X _ { i } ^ { \prime } ) |$ . Among methods with comparable coverage, smaller AvgSize is preferable.

Given a grouping attribute A and its induced groups, we are interested in coverage conditional on each group. For group $g \in { \mathcal { G } } ( A )$ , let $\mathcal { T } _ { g } : = \{ i : A \in X _ { i } ^ { \prime } ) = g \}$ . We define the estimated conditional coverage, hereafter group coverage, as $\begin{array} { r } { \hat { c } _ { g } = \frac { 1 } { | \mathcal { T } _ { a } | } \sum _ { i \in \mathcal { T } _ { q } } \mathbf { 1 } \{ Y _ { i } ^ { \prime } \in \mathcal { C } ( X _ { i } ^ { \prime } ) \} } \end{array}$ . We quantify subgroup disparity by the coverage gap (CovGap) $\begin{array} { r } { \mathrm { C o v G a p } ( A ) = \frac { 1 } { | \mathcal { G } ( A ) | } \sum _ { g \in \mathcal { G } ( A ) } | \hat { c } _ { g } - ( 1 - \alpha ) | } \end{array}$ . Group coverage closer to $1 - \alpha$ and smaller CovGap indicate more reliable subgroup performance. All metrics other than AvgSize are reported as percentages.

## 4.1 Results and Discussion

Main results. Table 1 compares all methods across the four benchmarks (other experiments cf. Appendix E). AFCP is omitted on MIMIC-IV because its adaptive group selection requires an expensive repeated calibration procedure, making 20-run evaluation computationally prohibitive. We highlight AvgSize values exceeding half of the label space in red, as prediction sets of this size can reduce clinical utility, increase the cognitive burden on decision makers.

Table 1: Each method is run 20 times, and we report the mean and standard deviation. All results are reported for the largest sample size. Results with nominal coverage are shown in blue, optimal values are boldfaced, and AvgSize exceeding half of the total number of categories are highlighted in red.
<table><tr><td rowspan="2">Method</td><td colspan="3">SYN</td><td colspan="3">Nursery</td><td colspan="3">MIMIC-IV</td><td colspan="3">BACH</td></tr><tr><td>Cov. ↑</td><td> $\mathbf { A v g S i z e \downarrow }$ </td><td> $\operatorname { C o v G a p } \downarrow$ </td><td> $\mathbf { C o v . \uparrow }$ </td><td> $\mathbf { A v g S i z e \downarrow }$ </td><td>CovGap ↓</td><td> $\mathbf { C o v . \uparrow }$ </td><td> $\mathbf { A v g S i z e \downarrow }$ </td><td> $\mathbf { C o v G a p \downarrow }$ </td><td> $\mathbf { C o v . \uparrow }$ </td><td> $\mathbf { A v g S i z e \downarrow }$ </td><td> $\mathrm { C o v G a p \downarrow }$ </td></tr><tr><td>Marginal</td><td> $8 9 . 5 2 _ { \pm 1 . 5 2 }$ </td><td> $2 . 3 0 9 _ { \pm 0 . 0 7 8 }$ </td><td> $1 4 . 5 _ { \pm 1 . 6 }$ </td><td> $8 4 . 1 5 _ { \pm 5 . 2 0 }$ </td><td> $2 . 1 7 0 _ { \pm 0 . 2 6 0 }$ </td><td> $2 0 . 4 _ { \pm 5 . 2 }$ </td><td> $9 0 . 7 9 _ { \pm 1 0 . 6 2 }$ </td><td> $1 . 2 2 6 _ { \pm 0 . 0 8 7 }$ </td><td> $2 7 . 1 _ { \pm 7 . 1 }$ </td><td> $8 9 . 2 0 _ { \pm 9 . 6 4 }$ </td><td> $2 . 8 9 4 _ { \pm 0 . 2 6 7 }$ </td><td> $1 5 . 7 _ { \pm 1 . 7 }$ </td></tr><tr><td>Partial</td><td> $9 4 . 5 0 _ { \pm 1 . 1 2 } \ 2 . 4 1 9 _ { \pm 0 . 0 4 8 }$ </td><td></td><td> $6 . 2 \pm 0 . 7$ </td><td> $8 8 . 9 7 _ { \pm 3 . 8 0 }$ </td><td> $1 . 8 5 0 { \scriptstyle \pm 0 . 2 0 0 }$ </td><td> $1 2 . 1 { \pm } 2 . 5$ </td><td> $9 4 . 5 0 { \scriptstyle \pm 4 . 5 4 } $ </td><td> $1 . 2 5 8 _ { \pm 0 . 1 4 0 }$ </td><td> $8 . 0 { \scriptstyle \pm 3 . 0 }$ </td><td> $9 6 . 5 7 { \scriptstyle \pm 8 . 0 8 }$ </td><td> $3 . 3 8 5 _ { \pm 0 . 2 9 9 }$ </td><td> $7 . 6 { \pm } 1 . 6 $ </td></tr><tr><td>Exhaustive</td><td> $9 5 . 8 5 _ { \pm 0 . 9 1 } 3 . 3 4 2 _ { \pm 0 . 0 5 3 }$ </td><td></td><td> $6 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td> $9 6 . 8 0 _ { \pm 3 . 1 0 }$ </td><td> $2 . 8 5 0 _ { \pm 0 . 2 2 0 }$ </td><td> $1 5 . 5 _ { \pm 3 . 2 }$ </td><td> $9 6 . 4 5 _ { \pm 4 . 2 2 }$ </td><td>2.800±0.142</td><td> $1 9 . 7 _ { \pm 3 . 2 }$ </td><td> $9 7 . 4 1 _ { \pm 6 . 8 9 }$ </td><td> $3 . 8 6 7 _ { \pm 0 . 1 1 8 }$ </td><td> $1 1 . 6 _ { \pm 1 . 5 }$ </td></tr><tr><td>RLCP</td><td> $9 3 . 1 0 _ { \pm 2 . 4 5 } \ 2 . 5 4 0 _ { \pm 0 . 3 0 5 }$ </td><td></td><td> $4 . 5 _ { \pm 1 . 8 }$ </td><td> $8 8 . 6 7 _ { \pm 6 . 2 0 }$ </td><td> $1 . 2 2 0 _ { \pm 0 . 4 1 0 }$ </td><td> $1 3 . 5 _ { \pm 3 . 8 }$ </td><td> $9 2 . 5 0 _ { \pm 2 . 8 0 }$ </td><td> $1 . 3 5 0 _ { \pm 0 . 1 0 0 }$ </td><td> $1 . 5 _ { \pm 0 . 8 }$ </td><td> $9 1 . 8 8 _ { \pm 1 0 . 5 6 }$ </td><td> $2 . 3 1 1 _ { \pm 0 . 3 1 9 }$ </td><td> $8 . 1 _ { \pm 2 . 5 }$ </td></tr><tr><td>CluCP</td><td> $8 9 . 8 0 _ { \pm 1 . 7 8 } \ 2 . 4 2 6 _ { \pm 0 . 0 5 0 }$ </td><td></td><td> $4 . 9 _ { \pm 0 . 8 }$ </td><td> $8 5 . 0 8 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $1 . 1 2 1 { \scriptstyle \pm 0 . 0 6 4 }$ </td><td> $1 2 . 8 { \scriptstyle \pm 1 . 3 }$ </td><td> $9 2 . 2 2 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $1 . 3 0 2 _ { \pm 0 . 0 5 9 }$ </td><td> $3 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> $9 5 . 9 1 _ { \pm 3 . 5 9 }$ </td><td> $3 . 2 5 7 _ { \pm 0 . 2 2 0 }$ </td><td> $1 0 . 7 _ { \pm 2 . 6 }$ </td></tr><tr><td>AFCP</td><td> $9 1 . 7 7 _ { \pm 1 . 2 3 } \ 2 . 4 3 2 _ { \pm 0 . 0 4 3 }$ </td><td></td><td> $4 . 6 _ { \pm 0 . 5 }$ </td><td> $9 2 . 8 5 _ { \pm 0 . 8 4 }$ </td><td> $1 . 0 8 5 _ { \pm 0 . 0 2 4 }$ </td><td> $1 4 . 1 _ { \pm 0 . 6 }$ </td><td></td><td></td><td></td><td> $9 5 . 0 0 _ { \pm 2 . 6 0 }$ </td><td> $2 . 8 3 0 _ { \pm 0 . 2 3 7 }$ </td><td> $6 . 3 _ { \pm 1 . 7 }$ </td></tr><tr><td>FaReG</td><td> $9 2 . 6 1 _ { \pm 0 . 9 8 } \ 2 . 5 7 9 _ { \pm 0 . 0 2 9 }$ </td><td></td><td> $4 . 7 _ { \pm 0 . 5 }$ </td><td> $9 3 . 2 1 _ { \pm 0 . 6 2 } \ 1 . 2 3 0 _ { \pm 0 . 0 1 2 }$ </td><td></td><td> $1 3 . 3 { \scriptstyle \pm 0 . 6 }$ </td><td> $9 0 . 2 3 \substack { \pm 0 . 0 9 }$ </td><td> $1 . 1 7 4 _ { \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 5 _ { \pm 0 . 0 } }$ </td><td> $9 3 . 6 0 { \scriptstyle \pm 3 . 4 0 }$ </td><td> $2 . 9 5 9 _ { \pm 0 . 2 3 9 }$ </td><td> $7 . 4 { \pm } 1 . 5$ </td></tr><tr><td>SGCP</td><td> $9 0 . 1 3 _ { \pm 0 . 3 9 } ~ 2 . 2 2 5 _ { \pm 0 . 0 3 5 }$ </td><td></td><td> ${ \bf 4 . 2 _ { \pm 0 . 7 } }$ </td><td> $9 1 . 6 5 _ { \pm 0 . 7 0 } \ 1 . 0 8 9 _ { \pm 0 . 0 3 5 }$ </td><td></td><td> ${ \bf 1 1 . 8 _ { \pm 0 . 7 } }$ </td><td> $\mathbf { 9 0 . 1 3 _ { \pm 0 . 0 7 } }$ </td><td> ${ \bf 1 . 1 0 7 _ { \pm 0 . 0 0 1 } }$ </td><td> $0 . 8 _ { \pm 0 . 0 }$ </td><td> $\mathbf { 9 1 . 9 0 _ { \pm 2 . 7 0 } }$ </td><td> $2 . 2 2 4 _ { \pm 0 . 2 1 5 }$ </td><td> ${ \bf 6 . 2 _ { \pm 1 . 1 } }$ </td></tr></table>

(a) SYN  
![](images/4d281605242f361dbae4d0c856e7b8965b2bca1838e2ae58fbd1427543f2ad1d.jpg)

![](images/79c1f63976f6d4921b533261003293b6e3d591219d4c1d694838d51220850321.jpg)

![](images/de9f2c8c8286cce45fad1f76fe57f54b149e6a4985afdd713713652308a34f49.jpg)  
(b) Nursery

![](images/4de88709531f58c3fe0584c54b69c520566fd37a6e8c11aa2b695874b8728bf3.jpg)  
(c) MIMIC-IV  
(d) BACH  
Figure 3: Learned latent component densities on four datasets. The components reveal distinct calibration regimes in the nonconformity-score space.

We highlight four main observations. (i) SGCP is the only method that consistently stays close to nominal coverage across all benchmarks while achieving strong subgroup reliability. Its empirical coverage remains near 0.9, and its CovGap is the smallest on SYN, Nursery, and BACH, with values 3.7, 11.8, and 6.2, respectively. On MIMIC-IV, it is close to the best result, with CovGap 0.8 compared with FaReG’s 0.5. (ii) The subgroup reliability gains of SGCP are not obtained through indiscriminate conservatism. SGCP achieves the smallest or nearly smallest AvgSize across all datasets. Taken together, these results show that SGCP improves subgroup reliability without enlarging prediction sets. (iii) Existing baselines exhibit less favorable or less consistent tradeoffs. Marginal CP often leaves large subgroup disparities, while explicit subgroup-protection methods reduce disparity at the cost of inflated coverage and larger prediction sets. RLCP, CluCP, AFCP, and FaReG are competitive in some settings, but their gains are less uniform across benchmarks.

Figure 3 visualizes the latent component densities learned by SGCP. Across datasets, the components exhibit diverse shapes in the nonconformity score space. Some components concentrate near small scores, while others are broader or place more mass on larger scores. This suggests that SGCP learns nontrivial score law structure, rather than merely reproducing a single global score distribution. Importantly, these components should not be viewed as demographic groups or semantic subpopulations. They are latent score law building blocks. For each sample, SGCP mixes these shared components to form a sample dependent local score law. In this way, samples with similar calibration behavior can share evidence even when they do not belong to the same observed subgroup. This provides a mechanism for improving subgroup reliability without requiring calibration to be tied to a single observed group label.

Group coverage. To better understand where the improvement comes from, we visualize subgroup coverage against average prediction-set size, together with how set size varies with subgroup sample size. Figure 4 corresponds to BACH, where the subgroup variable is the class label. Two patterns are clear. First, SGCP remains close to the nominal target coverage while lying on the more efficient side of the tradeoff: in the top row, its points stay near the horizontal reference line at 0.9 and are typically located to the left of competing methods. Second, in the bottom row, its prediction sets shrink steadily as subgroup sample size increases, indicating that it can use additional subgroup evidence efficiently while still adapting locally. Overall, these subgroup-level plots show that the gains of SGCP are realized across individual groups rather than only in aggregate.

![](images/8b6753da0d6af55ab7472810ddaf90373480bf147db926d663cca638d271835b.jpg)  
Figure 4: Group coverage on BACH dataset. SGCP remains close to the target coverage across classes while using smaller prediction sets than more conservative baselines.

![](images/b3f0d8294d407bd3427f164b64a7c06cd116407b30ece5e24098794612eea241.jpg)  
Figure 5: Distribution of CovGap over repeated runs under varying sample sparsity.

Distribution of CovGap. Figure 5 shows the distribution of CovGap over repeated runs under varying levels of sample sparsity. Beyond low average disparity, SGCP also exhibits stability: across datasets and sample sizes, it is consistently among the methods with the lowest median CovGap and relatively small variation across runs. This advantage is most pronounced in sparse-data settings, where subgroup calibration is most difficult. As sample size grows, the gap narrows for most methods, but SGCP remains competitive and typically retains one of the lowest CovGap values.

## 5 Conclusion

In this work, we discussed the limitation of standard split-conformal prediction in fairness-sensitive settings, where marginal validity can mask substantial subgroup miscoverage. We argued that subgroup reliability is closely tied to the local distribution of nonconformity scores, and that modeling this local score law provides a principled way to improve calibration across heterogeneous subpopulations. To this end, we proposed SGCP, a fairness-aware conformal prediction method that improves coverage for clinically meaningful subgroups without relying on predefined sensitive groups. Our theoretical analysis showed that SGCP preserves the standard split-conformal marginal coverage guarantee and accurate local score-law learning reduces subgroup coverage gaps by making calibration behavior more comparable across subpopulations. Extensive experiments across synthetic, and real-world benchmarks demonstrated that SGCP consistently improves subgroup reliability while maintaining efficient prediction sets. In future work, we plan to extend SGCP to personalized medicine, where reliable uncertainty quantification across diverse patient populations is crucial.

## References

[1] A. N. Angelopoulos and S. Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023.

[2] G. Aresta, T. Araújo, S. Kwok, S. S. Chennamsetty, M. Safwan, V. Alex, B. Marami, M. Prastawa, M. Chan, M. Donovan, G. Fernandez, J. Zeineh, M. Kohl, C. Walz, F. Ludwig, S. Braunewell, M. Baust, Q. D. Vu, M. N. N. To, E. Kim, J. T. Kwak, S. Galal, V. Sanchez-Freire, N. Brancati, M. Frucci, D. Riccio, Y. Wang, L. Sun, K. Ma, J. Fang, I. Kone, L. Boulmane, A. Campilho, C. Eloy, A. Polónia, and P. Aguiar. Bach: Grand challenge on breast cancer histology images. Medical Image Analysis, 56:122–139, 2019.

[3] K. Bairaktari, J. Wu, and S. Wu. Kandinsky conformal prediction: Beyond class- and covariateconditional coverage. In International Conference on Learning Representations (ICLR), 2025.

[4] R. J. Chen, J. J. Wang, D. F. K. Williamson, T. Y. Chen, J. Lipkova, M. Y. Lu, S. Sahai, and F. Mahmood. Algorithmic fairness in artificial intelligence for medicine and healthcare. Nature Biomedical Engineering, 7(6):719–742, 2023.

[5] M. Y. Cheung, A. Veeraraghavan, and G. Balakrishnan. COMPASS: Robust feature conformal prediction for medical segmentation metrics. In International Conference on Learning Representations (ICLR), 2026.

[6] S. Cortes-Gomez, C. M. Patiño, Y. Byun, S. Wu, E. Horvitz, and B. Wilder. Utility-directed conformal prediction: A decision-aware framework for actionable uncertainty quantification. In International Conference on Learning Representations (ICLR), 2025.

[7] J. C. Cresswell, B. Kumar, Y. Sui, and M. Belbahri. Conformal prediction sets can cause disparate impact. In International Conference on Learning Representations (ICLR), 2025.

[8] T. Ding, A. N. Angelopoulos, S. Bates, M. I. Jordan, and R. J. Tibshirani. Class-conditional conformal prediction with many classes. In Advances in Neural Information Processing Systems (NeurIPS), pages 64555–64576, 2023.

[9] S. G. Finlayson, A. Subbaswamy, K. Singh, J. Bowers, A. Kupke, J. Zittrain, I. S. Kohane, and S. Saria. The clinician and dataset shift in artificial intelligence. New England Journal of Medicine, 385(3):283–286, 2021.

[10] R. Foygel Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani. The limits of distribution-free conditional predictive inference. Information and Inference: A Journal ofthe IMA, 10(2):455– 482, 2021.

[11] C. Gao, P. B. Gilbert, and L. Han. Bridging fairness and efficiency in conformal inference: a surrogate-assisted group-clustered approach. In Proceedings of the International Conference on Machine Learning (ICML), 2026.

[12] R. Hore and R. F. Barber. Conformal prediction with local weights: randomization enables robust guarantees. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 87(2):549–578, 2024.

[13] A. Johnson, L. Bulgarelli, T. Pollard, B. Gow, B. Moody, S. Horng, L. A. Celi, and R. Mark. MIMIC-IV. PhysioNet, 2024. Version 3.1.

[14] H. Kwon and D.-J. Kim. Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift. Scientific Reports, 16(1):10016, 2026.

[15] J. Lee and R. M. Nishikawa. Identifying women with mammographically- occult breast cancer leveraging gan-simulated mammograms. IEEE Transactions on Medical Imaging, 41(1):225– 236, 2022.

[16] C. Lu, A. N. Angelopoulos, and S. Pomerantz. Improving trustworthiness of ai disease severity rating in medical imaging with ordinal conformal prediction sets. In Medical Image Computing and Computer Assisted Intervention (MICCAI), page 545–554, 2022.

[17] C. Lu, A. Lemay, K. Chang, K. Höbel, and J. Kalpathy-Cramer. Fair conformal predictors for applications in medical imaging. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 12008–12016, 2022.

[18] N. Martinez, D. C. Patel, C. Reddy, G. Ganapavarapu, R. Vaculin, and J. Kalagnanam. Identifying homogeneous and interpretable groups for conformal prediction. In The Conference on Uncertainty in Artificial Intelligence (UAI), 2024.

[19] V. Rajkovic. Nursery. UCI Machine Learning Repository, 1989. DOI: https://doi.org/10.24432/C5P88W.

[20] Y. Romano, R. F. Barber, C. Sabatti, and E. Candès. With Malice Toward None: Assessing Uncertainty via Equalized Coverage. Harvard Data Science Review, 2(2), 2020.

[21] Y. Romano, M. Sesia, and E. Candes. Classification with valid and adaptive coverage. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 3581–3591, 2020.

[22] G. Shafer and V. Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9:371–421, 2008.

[23] A. T. Vadlamani, A. Srinivasan, P. Maneriker, A. Payani, and S. Parthasarathy. A generic framework for conformal fairness. In International Conference on Learning Representations (ICLR), 2025.

[24] V. Vovk, A. Gammerman, and G. Shafer. Algorithmic learning in a random world, pages 1–476. Springer Cham, 2005.

[25] V. Vovk, D. Lindsay, I. Nouretdinov, and A. Gammerman. Mondrian confidence machine. Technical Report, 2003.

[26] Z. Wang and H. Wang. Variational imbalanced regression: Fair uncertainty quantification via probabilistic smoothing. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[27] S. Xu, Y. Zhou, Y. Tan, Z. Li, Y. Yao, T. Chen, F. Xu, and X. Ma. Fair conformal classification via learning representation-based groups. In International Conference on Learning Representations (ICLR), 2026.

[28] D. Zhang, A. Chatzimparmpas, N. Kamali, and J. Hullman. Evaluating the utility of conformal prediction sets for ai-advised image labeling. In ACM Conference on Human Factors in Computing Systems (CHI), 2024.

[29] Y. Zhou and M. Sesia. Conformal classification with equalized coverage for adaptively selected groups. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

## A Mathematical Proofs

## A.1 Proof of Proposition 1

Proof. Let $S : = s ( x , y )$ and $V : = \widehat { F } ( S \mid X )$ . Let λ denote the Lebesgue measure on [0, 1]. By the oracle assumption, for P<sub>X</sub> -a.e. x,

$$
\widehat F ( \cdot \mid x ) = F _ { x } ^ { \star } ( \cdot ) , \qquad F _ { x } ^ { \star } ( t ) : = \mathbb { P } ( S \leq t \mid X = x ) .
$$

Moreover, by assumption, the conditional distribution of $S \mid X = x$ is continuous for $P _ { X ^ { - } \mathbf { a } . \mathbf { e } . \ x }$ Hence, by the probability integral transform, for $P _ { X } { \mathrm { - } } { \mathrm { a } } . { \mathrm { e } } . \ x$

$$
F _ { x } ^ { \star } ( S ) \mid X = x \sim \operatorname { U n i f } ( 0 , 1 ) .
$$

Since $V = \widehat { F } ( S \mid X ) = F _ { X } ^ { \star } ( S )$ almost surely, it follows that, for every Borel set $A \subseteq [ 0 , 1 ]$

$$
{ \mathbb E } \big [ \mathbf { 1 } _ { \{ V \in A \} } \mid X \big ] = \lambda ( A ) \qquad \mathrm { a . s . }
$$

Now let $G \subseteq { \mathcal { X } }$ be any measurable subgroup with $\mathbb { P } ( X \in G ) > 0$ . Using the tower property,

$$
\begin{array} { r l } & { \mathbb { P } ( V \in A \mid X \in G ) = \frac { \mathbb { E } \left[ \mathbf { 1 } _ { \{ V \in A \} } \mathbf { 1 } _ { \{ X \in G \} } \right] } { \mathbb { P } ( X \in G ) } } \\ & { \qquad = \frac { \mathbb { E } \left[ \mathbf { 1 } _ { \{ X \in G \} } \mathbb { E } \left[ \mathbf { 1 } _ { \{ V \in A \} } \mid X \right] \right] } { \mathbb { P } ( X \in G ) } } \\ & { \qquad = \frac { \mathbb { E } \left[ \mathbf { 1 } _ { \{ X \in G \} } \lambda ( A ) \right] } { \mathbb { P } ( X \in G ) } } \\ & { \qquad = \lambda ( A ) . } \end{array}
$$

Therefore, for every set $A \subseteq [ 0 , 1 ] , \mathbb { P } ( V \in A \mid X \in G ) = \lambda ( A )$ , which is precisely the statement that

$$
V \mid \{ X \in G \} \sim \operatorname { U n i f } ( 0 , 1 ) .
$$

Remark 1. When this oracle local score law is recovered, the local percentile transform places scores from different inputs and subgroups on a common uniform scale, thereby eliminating subgroup reliability inconsistency. The objective of SGCP is to approximate this. Put differently, this proposition clarifies why we learn ${ \widehat { F } } ( s \mid x )$ instead ofoperating directly on the raw score s.

## A.2 Proof of Proposition 2

Throughout the proof, we condition on the training split $\mathcal { D } _ { \mathrm { t r a } }$ . Thus the fitted transformation $( x , y ) \mapsto \widehat { F } ( s ( x , y ) \mid x )$ is fixed. For simplicity, we suppress this conditioning from the notation.

Let $m : = | \mathcal { D } _ { \mathrm { c a l } } |$ and write $\mathcal { D } _ { \mathrm { c a l } } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { m }$ Define the transformed calibration scores

$$
V _ { i } : = { \widehat { F } } ( s ( X _ { i } , Y _ { i } ) \mid X _ { i } ) , \qquad i = 1 , \ldots , m ,
$$

and the transformed score of the test point under its true label

$$
V _ { \mathrm { t e s t } } : = \widehat { F } ( s ( X _ { n + 1 } , Y _ { n + 1 } ) \mid X _ { n + 1 } ) .
$$

Let $V _ { ( 1 ) } \leq \cdots \leq V _ { ( m ) }$ denote the order statistics of the calibration scores, and set

$$
k : = \lceil ( m + 1 ) ( 1 - \alpha ) \rceil .
$$

Define the split-conformal cutoff by $\widehat { q } _ { 1 - \alpha } = V _ { ( k ) }$ , with the convention $\widehat { q } _ { 1 - \alpha } = 1$ whenever $k > m$

Proof. By assumption, conditional on $\mathcal { D } _ { \mathrm { t r a } } .$ , the calibration examples in $\mathcal { D } _ { \mathrm { c a l } }$ and the test point $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ are exchangeable. Since the same deterministic measurable map is applied to each point, it follows that $V _ { 1 } , \ldots , V _ { m } , V _ { \mathrm { t e s t } }$ are exchangeable conditional on $\mathcal { D } _ { \mathrm { t r a } }$

Let $V _ { ( 1 ) } \ \leq \ \cdots \ \leq \ V _ { ( m }$ denote the ordered transformed calibration scores, and define the split conformal cutoff by $\widehat { q } _ { 1 - \alpha } = V _ { \left( \lceil m + 1 \rceil ( 1 - \alpha ) \rceil \right) }$ , with the convention $\widehat { q } _ { 1 - \alpha } = 1$ whenever $\lceil ( m + 1 ) ( 1 -$ $\alpha ) ] > m$ . By exchangeability and the standard split conformal order-statistic argument, we have P $\smash { \big ( V _ { \mathrm { t e s t } } \lesssim \widehat { q } _ { 1 - \alpha } \big | \mathcal { D } _ { \mathrm { t r a } } \big ) \geq 1 - \alpha }$

It remains to connect this event to the prediction set. By definition,

$$
\widehat C _ { \alpha } ( X _ { n + 1 } ) = \left\{ y \in \mathcal { Y } : \ \widehat { F } \big ( s ( X _ { n + 1 } , y ) \mid X _ { n + 1 } \big ) \leq \widehat { q } _ { 1 - \alpha } \right\} .
$$

Substituting the true label $Y _ { n + 1 }$ gives

$$
\begin{array} { r } { \Big \{ Y _ { n + 1 } \in \widehat C _ { \alpha } ( X _ { n + 1 } ) \Big \} = \Big \{ \widehat F \big ( s ( X _ { n + 1 } , Y _ { n + 1 } ) \mid X _ { n + 1 } \big ) \leq \widehat q _ { 1 - \alpha } \Big \} = \{ V _ { \mathrm { t e s t } } \leq \widehat q _ { 1 - \alpha } \} . } \end{array}
$$

Hence,

$$
\begin{array} { r } { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | \mathcal { D } _ { \mathrm { t r a } } \Big ) \geq 1 - \alpha . } \end{array}
$$

Taking expectations over $\mathcal { D } _ { \mathrm { t r a } }$ gives

$$
\mathbb { P } \left\{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \right\} = \mathbb { E } \left[ \mathbb { P } \left\{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | \mathcal { D } _ { \mathrm { t r a } } \right\} \right] \geq 1 - \alpha .
$$

Remark 2. In the current implementation, the stochastic grouping map $w ( x )$ may be approximated by Monte Carlo averaging over latent posterior samples. The validity proof above continues to hold as long as the transformed score is computed using a rule that is fixed conditional on $\mathcal { D } _ { \mathrm { t r a } } ,$ , or, more generally, uses auxiliary randomness that is independent across calibration and test samples.

## A.3 Proof of Proposition 3

Proof. By definition of the prediction set,

$$
\widehat C _ { \alpha } ( X _ { n + 1 } ) = \left\{ y \in \mathcal y : \widehat F ( s ( X _ { n + 1 } , y ) \mid X _ { n + 1 } ) \leq \widehat q _ { 1 - \alpha } \right\} .
$$

Therefore,

$$
\begin{array} { r } { \Big \{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big \} = \{ V _ { \mathrm { t e s t } } \leq \widehat { q } _ { 1 - \alpha } \} . } \end{array}
$$

Now fix a measurable subgroup $G \subseteq { \mathcal { X } }$ with $\mathbb { P } ( X \in G ) > 0$ . Since the calibration sample is independent of the test point, $\widehat { q } _ { 1 - \alpha }$ is independent of $( X _ { n + 1 } , Y _ { n + 1 } )$ . Hence, conditional on $\widehat { q } _ { 1 - \alpha }$

$$
\begin{array} { r l } & { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | X _ { n + 1 } \in G , \widehat { q } _ { 1 - \alpha } \Big ) } \\ & { \qquad = \mathbb { P } ( V _ { n + 1 } \leq \widehat { q } _ { 1 - \alpha } | X _ { n + 1 } \in G , \widehat { q } _ { 1 - \alpha } ) } \\ & { \qquad = H _ { G } ( \widehat { q } _ { 1 - \alpha } ) , } \end{array}
$$

where

$$
H _ { G } ( t ) : = \mathbb { P } ( V ( X , Y ) \leq t \mid X \in G ) .
$$

Taking expectations over $\widehat { q } _ { 1 - \alpha }$ gives

$$
\begin{array} { r } { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | X _ { n + 1 } \in G \Big ) = \mathbb { E } [ H _ { G } ( \widehat { q } _ { 1 - \alpha } ) ] . } \end{array}
$$

Next, let

$$
H ( t ) : = \mathbb { P } ( V ( X , Y ) \leq t )
$$

be the global CDF of the transformed score, and let $V ^ { \dagger }$ be an independent draw from this distribution. Since $\mathbf { \breve { V } } _ { 1 } , \dots , \mathbf { \bar { V } } _ { m } , V ^ { \dagger }$ are exchangeable and H is continuous, the standard split-conformal rank argument gives

$$
\mathbb { P } \big ( V ^ { \dagger } \leq \widehat { q } _ { 1 - \alpha } \big ) = \frac { k } { m + 1 } .
$$

On the other hand, conditioning on $\widehat { q } _ { 1 - \alpha }$

$$
\begin{array} { r } { \mathbb { P } \big ( V ^ { \dagger } \le \widehat { q } _ { 1 - \alpha } \big | \widehat { q } _ { 1 - \alpha } \big ) = H ( \widehat { q } _ { 1 - \alpha } ) . } \end{array}
$$

Therefore,

$$
\mathbb { E } [ H ( \widehat { q } _ { 1 - \alpha } ) ] = \frac { k } { m + 1 } .
$$

Combining the two identities,

$$
\begin{array} { r l } {  { \operatorname { \mathbb { P } } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | X _ { n + 1 } \in G \Big ) - \frac { k } { m + 1 } \Big | } } \\ & { \quad = | { \operatorname { \mathbb { E } } } [ H _ { G } ( \widehat { q } _ { 1 - \alpha } ) ] - { \operatorname { \mathbb { E } } } [ H ( \widehat { q } _ { 1 - \alpha } ) ] | } \\ & { \quad = | { \operatorname { \mathbb { E } } } [ H _ { G } ( \widehat { q } _ { 1 - \alpha } ) - H ( \widehat { q } _ { 1 - \alpha } ) ] | } \\ & { \quad \le { \operatorname { \mathbb { E } } } [ | H _ { G } ( \widehat { q } _ { 1 - \alpha } ) - H ( \widehat { q } _ { 1 - \alpha } ) | ] } \\ & { \quad \le \underset { t \in [ 0 , 1 ] } { \operatorname* { s u p } } | H _ { G } ( t ) - H ( t ) | = \Delta _ { G } . } \end{array}
$$

This proves the claimed subgroup coverage gap bound.

Finally, since

$$
\frac { k } { m + 1 } = \frac { \lceil ( m + 1 ) ( 1 - \alpha ) \rceil } { m + 1 } \geq 1 - \alpha ,
$$

we obtain

$$
\begin{array} { r } { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \Big | X _ { n + 1 } \in G \Big ) \geq 1 - \alpha - \Delta _ { G } . } \end{array}
$$

Since $k / ( m + 1 ) \geq 1 - \alpha$ , this result implies that whenever a subgroup’s transformed-score distribution is close to the overall transformed-score distribution, the global conformal cutoff keeps its coverage close to the nominal level $1 - \alpha$

Remark 3. Proposition 3 bounds subgroup miscoverage in terms ofthe residual transformed-score mismatch $\Delta _ { G }$ . This mismatch can befurther related to local score-law estimation error. Ifthe learned transform satisfies

$$
\operatorname* { s u p } _ { x , s } | \widehat { F } ( s \mid x ) - F _ { x } ^ { \star } ( s ) | \leq \varepsilon ,
$$

then, by the probability integral transform,

$$
\Delta _ { G } \leq 2 \varepsilon
$$

for every measurable subgroup G. Consequently,

$$
\left| \mathbb { P } \{ Y _ { n + 1 } \in \widehat { C } _ { \alpha } ( X _ { n + 1 } ) \mid X _ { n + 1 } \in G \} - ( 1 - \alpha ) \right| \leq 2 \varepsilon + \frac { 1 } { m + 1 } .
$$

Thus, accurate local score-law learning provides a sufficient condition for small subgroup coverage gaps.

## A.4 Empirical Subgroup Coverage

We now relate the empirical subgroup coverage measured on an independent test sample to the population subgroup coverage above. Under the assumptions of Proposition 3, let $\begin{array} { r l } { { \mathcal { D } } _ { G } ^ { \mathrm { t e s t } } = } & { { } } \end{array}$ $\{ ( \boldsymbol { X } _ { j } ^ { ( G ) } , \boldsymbol { Y } _ { j } ^ { ( G ) } ) \} _ { j = 1 } ^ { n _ { G } }$ be an i.i.d. test sample drawn from the conditional distribution of $( X , Y )$ given $X \in G ,$ , independent of $\mathcal { D } _ { \mathrm { t r a } }$ and $\mathcal { D } _ { \mathrm { c a l } }$

Corollary 1. $L e t \widehat { \mathrm { C o v } } _ { G }$ be the coverage measured on an independent subgroup test sample of size $n _ { G }$ denoted as $\begin{array} { r } { \widehat { \mathrm { C o v } _ { G } } : = \frac { 1 } { n _ { G } } \sum _ { j = 1 } ^ { n _ { G } } \mathbf { 1 } \left\{ { Y } _ { j } ^ { ( G ) } \in \widehat { C } _ { \alpha } ( X _ { j } ^ { ( G ) } ) \right\} } \end{array}$ . Then,for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\left| \widehat { \mathrm { C o v } } _ { G } - ( 1 - \alpha ) \right| \leq \Delta _ { G } + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } + \frac { 1 } { m + 1 } .
$$

In particular,

$$
\widehat { \mathrm { C o v } } _ { G } \geq 1 - \alpha - \Delta _ { G } - \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } .
$$

Corollary 1 connects the theoretical subgroup coverage guarantee to the quantities reported in our experiments. When the learned transform reduces $\Delta _ { G }$ and the subgroup test sample size $n _ { G }$ is sufficiently large, the empirical subgroup coverage concentrates near the nominal level $1 - \alpha$ up to the conformal correction $1 / ( m + 1 )$

Proof. Conditional on $\mathcal { D } _ { \mathrm { t r a } }$ and $\mathcal { D } _ { \mathrm { c a l } }$ , the prediction set $\widehat { C } _ { \alpha }$ is fixed. Since the test points

$$
( X _ { j } ^ { ( G ) } , Y _ { j } ^ { ( G ) } ) , \qquad j = 1 , \dots , n _ { G } ,
$$

are i.i.d. draws from the conditional distribution of $( X , Y )$ given $X \in G$ , the indicators

$$
Z _ { j } : = { \bf 1 } \left\{ Y _ { j } ^ { ( G ) } \in \widehat { C } _ { \alpha } ( X _ { j } ^ { ( G ) } ) \right\} , \qquad j = 1 , \ldots , n _ { G } ,
$$

are i.i.d. Bernoulli random variables with mean $\operatorname { C o v } _ { G } : = \mathbb { P } \left\{ Y \in { \widehat { C } } _ { \alpha } ( X ) \ \Big | X \in G \right\}$ , where the probability is understood conditional on $\mathcal { D } _ { \mathrm { t r a } }$ and $\mathcal { D } _ { \mathrm { c a l } }$

Therefore, by Hoeffding’s inequality, for any $\delta \in ( 0 , 1 )$

$$
\mathbb { P } ( | \widehat { \mathrm { C o v } _ { G } } - \mathrm { C o v } _ { G } | \geq \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } | \mathcal { D } _ { \mathrm { t r a } } , \mathcal { D } _ { \mathrm { c a l } } ) \leq \delta .
$$

Equivalently, with probability at least $1 - \delta$ over $\mathcal { D } _ { G } ^ { \mathrm { t e s t } }$

$$
\Bigl | \widehat { \mathrm { C o v } } _ { G } - \mathrm { C o v } _ { G } \Bigr | \leq \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } .\tag{A10}
$$

By Proposition $\begin{array} { r } { 3 , \left| \operatorname { C o v } _ { G } - \frac { k } { m + 1 } \right| \leq \Delta _ { G } } \end{array}$ . Combining this with Eq. (A10) and the triangle inequality yields

$$
\left| \widehat { \mathrm { C o v } _ { G } } - \frac { k } { m + 1 } \right| \leq \left| \widehat { \mathrm { C o v } _ { G } } - \mathrm { C o v } _ { G } \right| + \left| \mathrm { C o v } _ { G } - \frac { k } { m + 1 } \right| \leq \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } + \Delta _ { G } } .
$$

Next, since $k = \lceil ( m + 1 ) ( 1 - \alpha ) \rceil$ , we have $\begin{array} { r } { 0 \leq \frac { k } { m + 1 } - ( 1 - \alpha ) \leq \frac { 1 } { m + 1 } } \end{array}$ . Hence

$$
\left| \widehat { \mathrm { C o v } } _ { G } - ( 1 - \alpha ) \right| \leq \left| \widehat { \mathrm { C o v } } _ { G } - \frac { k } { m + 1 } \right| + \left| \frac { k } { m + 1 } - ( 1 - \alpha ) \right| \leq \Delta _ { G } + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } + \frac { 1 } { m + 1 } } .
$$

Finally, Proposition 3 implies Co $\begin{array} { r } { \mathrm { \Pi } ^ { \prime } G \geq \frac { k } { m + 1 } - \Delta _ { G } \geq 1 - \alpha - \Delta _ { G } } \end{array}$ . On the same high-probability event as in Eq. (A10),

$$
\widehat { \mathrm { C o v } } _ { G } \geq \mathrm { C o v } _ { G } - \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } ,
$$

and therefore

$$
\widehat { \mathrm { C o v } } _ { G } \geq 1 - \alpha - \Delta _ { G } - \sqrt { \frac { \log ( 2 / \delta ) } { 2 n _ { G } } } .
$$

## B Model Details

This appendix specifies the concrete instantiation of the aggregate local score law in Eq. (8) and the corresponding training objective. The algorithm 2 demonstrates a specific process.

Latent calibration state. Let $h _ { \theta } ( x ) \in \mathbb R ^ { d }$ denote the backbone feature representation and let $f _ { \theta } ( x ) \in \Delta ^ { C - 1 }$ denote the corresponding predictive probability vector. We parameterize a Gaussian latent posterior $q _ { \phi } ( z \mid x ) = \mathcal { N } \Big ( \mu _ { \phi } ( h _ { \theta } ( x ) ) , \mathrm { d i a g } ( \sigma _ { \phi } ^ { 2 } ( h _ { \theta } ( x ) ) ) \Big )$

Algorithm 2 Detailed Training and Inference Procedure for SGCP   
Require: Training data $\overline { { \mathcal { D } _ { \mathrm { t r a } } } } .$ , calibration data $\mathcal { D } _ { \mathrm { c a l } }$ , backbone classifier $f _ { \theta } ,$ , nonconformity score   
$s ( \cdot , \cdot )$ , K latent calibration components, M posterior samples, hyperparameters $\lambda , \beta ,$ miscoverage   
level α   
Ensure: Prediction set $\widehat { C } _ { \alpha } ( X _ { N + 1 } )$   
1: Split $\mathcal { D } _ { \mathrm { t r a } }$ into $\mathcal { D } _ { 0 }$ and $\mathcal { D } _ { 1 }$   
2: Train the backbone predictor $f _ { \theta }$ on $\mathcal { D } _ { 0 }$   
3: Initialize the stochastic assignment module and component laws $\{ F _ { k } \} _ { k = 1 } ^ { K }$   
4: for each optimization step on $\mathcal { D } _ { 1 }$ do   
5: Sample a mini-batch $\mathbf { \bar { \xi } } B = \{ ( \bar { X } _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { B }$   
6: Compute backbone features, and nonconformity scores $s _ { i } = s ( X _ { i } , Y _ { i } )$   
7: Estimate posterior-averaged memberships from $q _ { \phi } ( z \mid X _ { i } )$   
8: Compute component transforms $\begin{array} { r c l } { C _ { i k } } & { = } & { F _ { k } ( S _ { i } ) } \end{array}$ and local densities ${ \widehat { f } } ( S _ { i } \mid X _ { i } ) =$   
$\textstyle \sum _ { k = 1 } ^ { K } w _ { i k } f _ { k } ( S _ { i } )$   
9: Compute $\mathcal { L } _ { \mathrm { r a n k } } , \mathcal { L } _ { \mathrm { n l l } } .$ , and $\widehat { I } _ { B } ^ { \mathrm { n o r m } } ( X ; Z )$   
10: Update assignment and component parameters by minimizing $\mathcal { L } _ { S G C P }$   
11: end for   
12: Construct the learned local score transform $\begin{array} { r } { \widehat { F } ( s \mid x ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) F _ { k } ( s ) } \end{array}$   
13: for each $( X _ { i } , Y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } }$ do   
14: Compute $V _ { i } = \widehat { F } ( s ( X _ { i } , Y _ { i } ) \mid X _ { i } )$   
15: end for   
16: Sort $\{ V _ { i } \} _ { i = 1 } ^ { m }$ and compute the conformal cutoff $\widehat { q } _ { 1 - \alpha } = V _ { \left( \lceil ( m + 1 ) ( 1 - \alpha ) \rceil \right) }$ with fallback $\widehat { q } _ { 1 - \alpha } = 1$   
i $\lceil ( \dot { m } + \bar { 1 } ) \bar { ( } 1 - \alpha ) \rceil > m$   
17: for each candidate label $y \in \mathcal { V }$ do   
18: Compute $V _ { \mathrm { t e s t } } ^ { y } = \widehat { F } ( S ( X _ { n + 1 } , y ) \mid X _ { N + 1 } )$   
19: end for   
20: return ${ \widehat { C } } _ { \alpha } ( X _ { n + 1 } ) = \{ y \in { \mathcal { V } } : V _ { n + 1 } ^ { y } \leq { \widehat { q } } _ { 1 - \alpha } \}$

Stochastic grouping map. Let $c _ { 1 } , \ldots , c _ { K } \in \mathbb { R } ^ { r }$ denote K learnable prototypes in the latent space. For a latent draw $z \sim q _ { \phi } ( z \mid x )$ , define the assignment vector

$$
\tilde { w } _ { k } ( x , z ) = \frac { \exp ( - \| z - c _ { k } \| _ { 2 } ^ { 2 } / T ) } { \sum _ { j = 1 } ^ { K } \exp ( - \| z - c _ { j } \| _ { 2 } ^ { 2 } / T ) } , \qquad k = 1 , \dots , K ,
$$

where $T > 0$ is a temperature parameter. Given $z ^ { ( 1 ) } , \dots , z ^ { ( M ) } \overset { \mathrm { i . i . d . } } { \sim } q _ { \phi } ( z \mid x )$ , the posterior-averaged membership vector $w ( x ) = ( w _ { 1 } ( x ) , \ldots , w _ { K } ( x ) ) \in \Delta ^ { K - 1 }$ is estimated by

$$
w _ { k } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { z } \sim \boldsymbol { q } _ { \phi } ( \cdot | \boldsymbol { x } ) } [ \tilde { w } _ { k } ( \boldsymbol { x } , \boldsymbol { z } ) ] \approx \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \tilde { w } _ { k } ( \boldsymbol { x } , \boldsymbol { z } ^ { ( m ) } ) , \qquad k = 1 , \dots , K .
$$

Remark 4. Marginal conformal validity only requires global exchangeability, so it can hold even when score behavior varies substantially across subgroups. Such heterogeneity can make a single global calibration rule locally unreliable. SGCP addresses this by learning latent calibration components that shift calibration evidence among samples with similar score behavior, providing a practical way to approximate oracle local score laws without assuming exact local exchangeability. Remark 5. In SGCP, posterior sampling approximates a stable stochastic grouping map rather than adding randomness to conformal calibration. Forfixed x, draw $z ^ { ( 1 ) } , \dots , z ^ { ( M ) } \overset { \mathrm { i . i . d . } } { \sim } q _ { \phi } ( z \mid x )$ and set $\widehat { w } _ { k } ^ { ( M ) } ( x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \tilde { w } _ { k } ( x , z ^ { ( m ) } ) , \qquad w _ { k } ( x ) = \mathbb { E } _ { z \sim q _ { \phi } ( \cdot | x ) } [ \tilde { w } _ { k } ( x , z ) ] .$ The induced local score law is $\begin{array} { r } { \widehat { F } ^ { ( M ) } ( s \mid x ) = \sum _ { k = 1 } ^ { K } \widehat { w } _ { k } ^ { ( M ) } ( x ) F _ { k } ( s ) . } \end{array}$ Samples share calibration evidence through the common components $\{ F _ { k } \} _ { k = 1 } ^ { K } ,$ and posterior averaging reduces Monte Carlo variability in the sample-specific weights. $A s M  \infty ,$ , the law of large numbers gives $\widehat { w } _ { k } ^ { ( M ) } ( x ) \to w _ { k } ( x )$ almost surely, and therefore

$$
{ \widehat { F } } ^ { ( M ) } ( s \mid x )  { \widehat { F } } ( s \mid x ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) F _ { k } ( s ) \quad a . s .
$$

## B.1 Training Objective

We train the stochastic assignment module and the component parameters on $\mathcal { D } _ { 1 }$ , while keeping the backbone fixed. For each $( \mathbf { \bar { X } } _ { i } , Y _ { i } ) \in \mathcal { D } _ { 1 }$ , define

$$
S _ { i } : = s ( X _ { i } , Y _ { i } ) , \qquad w _ { i k } : = w _ { k } ( X _ { i } ) , \qquad C _ { i k } : = F _ { k } ( S _ { i } ) .
$$

The latent components are intended to serve as reusable local score laws. Thus, scores routed to the same component should be comparable after being mapped to their component-wise percentile scale, while the induced local mixture density should still explain the observed true-label nonconformity scores. These two requirements lead to the rank-uniformization and local density-fitting terms below.

Rank-uniformization. For a mini-batch B of size B and a fixed component $k ,$ let $\big \{ C _ { ( 1 ) k } , \dots , C _ { ( B ) k } \big \}$ be the sorted values of $\{ C _ { i k } : i \in B \}$ , and let $w _ { ( 1 ) k } , \ldots , w _ { ( B ) k }$ denote the correspondingly reordered weights. Normalize the weights within component k by

$$
\pi _ { ( b ) k } = \frac { w _ { ( b ) k } } { \sum _ { \ell = 1 } ^ { B } w _ { ( \ell ) k } } .
$$

The weighted rank midpoint is $\begin{array} { r } { r _ { ( b ) k } = \sum _ { \ell < b } \pi _ { ( \ell ) k } + \frac { 1 } { 2 } \pi _ { ( b ) k } } \end{array}$ . We define the component-wise rank-uniformization loss as

$$
\ell _ { \mathrm { r a n k } } ^ { ( k ) } = \sum _ { b = 1 } ^ { B } \pi _ { ( b ) k } \left( C _ { ( b ) k } - r _ { ( b ) k } \right) ^ { 2 } .
$$

Averaging over active components gives

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { | \mathcal { K } _ { \mathrm { a c t } } | } \sum _ { k \in \mathcal { K } _ { \mathrm { a c t } } } \ell _ { \mathrm { r a n k } } ^ { ( k ) } , \qquad \mathcal { K } _ { \mathrm { a c t } } = \left\{ k : \sum _ { i \in \mathcal { B } } w _ { i k } > \tau \right\} .
$$

Local density fitting. Let $f _ { k }$ denote the density associated with $F _ { k }$ . The local mixture density at $( X _ { i } , S _ { i } )$ is $\begin{array} { r } { \widehat { f } ( S _ { i } \mid X _ { i } ) = \sum _ { k = 1 } ^ { K } w _ { i k } f _ { k } ( S _ { i } ) } \end{array}$ . We fit the observed true-label scores by the negative log-likelihood

$$
{ \mathcal { L } } _ { \mathrm { n l l } } = - { \frac { 1 } { | { \mathcal { B } } | } } \sum _ { i \in { \mathcal { B } } } \log { \widehat { f } } ( S _ { i } \mid X _ { i } ) .
$$

The rank term makes the component transforms behave like percentile maps, whereas the likelihood term ties the learned score laws to the empirical score distribution.

Information maximization for stochastic grouping. The preceding two terms specify how the component score laws should behave, but they do not by themselves ensure that the grouping map uses the components in an informative way. For example, a balance constraint on the average component mass can prevent all samples from collapsing onto a single component, but it still permits the opposite degenerate solution in which every sample is assigned nearly uniformly to all components. Such assignments are balanced in aggregate, yet they carry little sample-specific information and tend to make the learned component laws indistinguishable.

We therefore encourage the component assignment to retain information about the input. For a mini-batch $B ,$ view the assignment as a conditional categorical distribution $q _ { \phi } ( Z = k \mid \bar { X } _ { i } ) = w _ { i k }$ where $Z$ denotes the latent component index. The induced marginal component distribution on the mini-batch is

$$
\bar { w } _ { k } = \frac 1 { | \mathcal B | } \sum _ { i \in \mathcal B } w _ { i k } , \qquad \bar { w } = ( \bar { w } _ { 1 } , \dots , \bar { w } _ { K } ) .
$$

The empirical mutual information between inputs and component assignments is

$$
\widehat { I } _ { \mathcal { B } } ( X ; Z ) = H ( \bar { w } ) - \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } H ( w _ { i } ) ,
$$

where

$$
H ( \bar { w } ) = - \sum _ { k = 1 } ^ { K } \bar { w } _ { k } \log \bar { w } _ { k } , \qquad H ( w _ { i } ) = - \sum _ { k = 1 } ^ { K } w _ { i k } \log w _ { i k } .
$$

We use the normalized form

$$
\widehat { I } _ { \boldsymbol { B } } ^ { \mathrm { n o r m } } ( \boldsymbol { X } ; \boldsymbol { Z } ) = \frac { 1 } { \log K } \left[ H ( \bar { w } ) - \frac { 1 } { | \boldsymbol { B } | } \sum _ { i \in \boldsymbol { B } } H ( w _ { i } ) \right] .
$$

The marginal entropy term encourages all components to be used at the batch level, while the conditional entropy term encourages each sample to have a sharper, more informative assignment. Thus, maximizing $\widehat { I } _ { B } ^ { \mathrm { n o r m } } ( X ; Z )$ discourages both hard component collapse and soft uniform assignment.

Full objective. Taking the rank-uniformization loss as the reference scale, we optimize

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S G C P } } = \mathcal { L } _ { \mathrm { r a n k } } + \lambda \mathcal { L } _ { \mathrm { n l l } } - \beta \widehat { I } _ { \mathcal { B } } ^ { \mathrm { n o r m } } ( X ; Z ) , } \end{array}
$$

where λ controls the strength of local density fitting and $\beta$ controls the strength of stochastic-grouping information maximization.

## C Complexity Analysis

Beyond backbone training, SGCP adds three sources of computation: stochastic grouping, local score-law fitting, and conformal calibration on transformed scores. Let N denote the number of training samples used for learning the grouping map and local score laws, $N _ { \mathrm { c a l } }$ the calibration size, and $N _ { \mathrm { t e s t } }$ the number of test samples. Treating the number of epochs, posterior samples, latent components, latent dimension, classes, and batch size as constants, the training overhead is linear in N, up to an optional O(N log N) sorting step for constructing empirical component CDFs.

Calibration requires computing one transformed score per calibration sample and selecting a conformal quantile, costing $\mathcal { O } ( N _ { \mathrm { c a l } }$ log $N _ { \mathrm { c a l } } )$ with sorting or $\mathcal { O } ( N _ { \mathrm { c a l } } )$ with linear-time selection. At test time, the grouping map is evaluated once per input and transformed scores are computed for all candidate labels, giving constant overhead per test sample and total cost $\mathcal { O } ( N _ { \mathrm { t e s t } } )$ .

Overall, under the standard mini-batch implementation, the additional complexity beyond backbone training is $\mathcal { O } ( N + N _ { \mathrm { c a l } } \log N _ { \mathrm { c a l } } + N _ { \mathrm { t e s t } } ) , \mathrm { \dot { o r } } \mathcal { O } ( N + N _ { \mathrm { c a l } } + N _ { \mathrm { t e s t } } )$ with linear-time quantile selection.

## D Detailed Experimental Setup

## D.1 Additional Benchmark Details

Table A2: Summary of datasets and grouping attributes used for evaluation.
<table><tr><td>Dataset</td><td>Modality</td><td>#Classes</td><td>#Attrs</td><td>Sensitive / grouping attributes</td></tr><tr><td>SYN</td><td>tabular</td><td>6</td><td>3</td><td> $A _ { 1 } = \mathrm { g e n d e r } , A _ { 2 } = \mathrm { p h e n o t y p e } , A _ { 3 } = \mathrm { a g e }$ </td></tr><tr><td>Nursery</td><td>tabular</td><td>4</td><td>5</td><td> $A _ { 1 } = { \mathrm { p a r e n t s } } , A _ { 2 } = { \mathrm { c h i l d r e n } } , A _ { 3 } = { \mathrm { f i n a n c e } } ,$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $A _ { 4 } = { \mathrm { s o c i a l } } , A _ { 5 } = { \mathrm { h e a l t h } }$ </td></tr><tr><td>MIMIC-IV</td><td>tabular</td><td>2/3</td><td>3</td><td>A1 = minority, A2 = gender, A3 = public insurance</td></tr><tr><td>BACH</td><td>image</td><td>4</td><td>-</td><td>class label (for class-wise reliability analysis)</td></tr></table>

In this section, we describe the dataset preprocessing in detail. A summary is provided in Table A2.

Synthetic. The synthetic benchmark is a controlled multiclass cardiopulmonary diagnosis task with six labels, $Y \in \{ 1 , \ldots , 6 \}$ , corresponding to Hypertension, Coronary Artery Disease, Heart Failure, Arrhythmia, Valvular Heart Disease, and Peripheral Artery Disease.

Each sample contains four grouping attributes, $A = ( A _ { \mathrm { s e x } } , A _ { \mathrm { p h e n o } } , A _ { \mathrm { a g e } } , A _ { \mathrm { r e g i o n } } )$ , and six nonsensitive physiological variables $\boldsymbol { Z } = ( Z _ { 1 } , \ldots , Z _ { 6 } )$ , where $Z _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { U } ( 0 , 1 )$ . We draw

$$
A _ { \mathrm { s e x } } \sim \mathrm { B e r n o u l l i } ( 1 / 2 ) , \qquad A _ { \mathrm { p h e n o } } \sim \mathrm { B e r n o u l l i } ( 1 / 2 ) ,
$$

with $A _ { \mathrm { s e x } } \in \{ 0 , 1 \}$ denoting Female/Male and $A _ { \mathrm { p h e n o } } ~ \in ~ \{ 0 , 1 \}$ denoting Type I/Type II. Age is assigned cyclically over {Pediatric, Adult, Geriatric}, and region is sampled uniformly from {North, South, East, West}.

To induce intersection-dependent difficulty, we define the disadvantaged subgroup through the XNOR relation between sex and phenotype,

$$
G _ { \mathrm { b i a s } } = \{ i : A _ { \mathrm { s e x } , i } = A _ { \mathrm { p h e n o } , i } \} ,
$$

which includes Female–Type I and Male–Type II individuals.

Labels are generated from a latent multiclass mechanism based on $X _ { i } = ( A _ { i } , Z _ { i } )$ . We first construct a class-probability vector $\pi ( X _ { i } ) \in \Delta ^ { 5 }$ over the six disease classes. Outside ambiguous regions of the feature space, labels follow the dominant class implied by $\pi ( X _ { i } )$ . Inside ambiguous regions, we introduce subgroup-dependent uncertainty by mixing $\pi ( X _ { i } )$ with the uniform distribution $u =$ $( 1 / 6 , \ldots , 1 / 6 )$

$$
Y _ { i } \sim \left\{ \begin{array} { l l } { \mathrm { C a t } \big ( ( 1 - \delta _ { 1 } ) \pi ( X _ { i } ) + \delta _ { 1 } u \big ) , } & { i \in G _ { \mathrm { b i a s } } , } \\ { \mathrm { C a t } \big ( ( 1 - \delta _ { 0 } ) \pi ( X _ { i } ) + \delta _ { 0 } u \big ) , } & { i \notin G _ { \mathrm { b i a s } } , } \end{array} \right. \qquad \delta _ { 1 } > \delta _ { 0 } .
$$

The parameters $\delta _ { 1 }$ and $\delta _ { 0 }$ control label ambiguity for the disadvantaged subgroup and the remaining population, respectively. Thus, $G _ { \mathrm { b i a s } }$ exhibits systematically higher conditional uncertainty while the global task structure remains unchanged, yielding a controlled setting for studying subgroupdependent calibration difficulty.

Nursery. Nursery is a public multiclass tabular benchmark derived from a hierarchical decision problem for ranking nursery-school applications. Following standard preprocessing, we remove the recommend class, which contains only two samples, and encode the remaining labels into four admission-priority levels, $Y \in \{ 0 , 1 , 2 , 3 \}$

Each instance contains eight categorical covariates describing household background and application context, including parents’ occupation, number of children, housing, financial standing, social conditions, and health status. All categorical variables are label-encoded before training. We use five covariates as candidate sensitive attributes,

$$
A = ( A _ { \mathrm { p a r e n t s } } , A _ { \mathrm { c h i l d r e n } } , A _ { \mathrm { f i n a n c e } } , A _ { \mathrm { s o c i a l } } , A _ { \mathrm { h e a l t h } } ) ,
$$

corresponding to parents’ occupation, number of children, financial standing, social conditions, and health status. These attributes are used only for subgroup construction and fairness evaluation.

To induce controlled subgroup disadvantage, we follow the intervention protocol of prior work [29, 27]. We define

$$
G _ { \mathrm { n u r } } = \{ i : X _ { \mathrm { p a r e n t s } , i } = 1 \} ,
$$

corresponding to a particular preprocessed category of the parents attribute. We then modify only this subgroup by downsampling it to 10% of its original size and injecting label uncertainty. Specifically, labels are perturbed by independent noise $\varepsilon _ { i } \sim \mathcal { U } ( - 4 , 4 )$ , then rounded and clipped to $\{ \bar { 0 } , 1 , 2 , 3 \}$ . This creates a subgroup that is both underrepresented and noisier than the rest of the population, serving as the disadvantaged group in our fairness evaluation.

MIMIC-IV. We construct our clinical benchmark from MIMIC-IV at the hospital-admission level. Each admission is linked with ICU-stay information and diagnosis-code counts, yielding tabular covariates from demographic, administrative, and utilization records. Age is computed from anchor\_age, adjusted by the difference between admission year and anchor year, and clipped to [0, 89]. We also derive ICU-related features, including ICU length of stay, time from hospital admission to first ICU entry, and the number of diagnosis codes.

Table A3: Dataset specific hyperparameter settings for SGCP.
<table><tr><td>Hyperparameter</td><td>SYN</td><td>Nursery</td><td>M-ICU</td><td>M-Hosp</td><td>BACH</td><td>Description</td></tr><tr><td colspan="7">(i) Latent calibration components</td></tr><tr><td>K</td><td>3</td><td>6</td><td>3</td><td>3</td><td>6</td><td>number of latent calibration components</td></tr><tr><td> $d _ { z }$ </td><td>16</td><td>8</td><td>8</td><td>8</td><td>64</td><td>latent calibration dimension</td></tr><tr><td>hidden dim</td><td>64</td><td>64</td><td>32</td><td>32</td><td>64</td><td>hidden dimension of grouping network</td></tr><tr><td colspan="7">(ii) Score law learning</td></tr><tr><td>SGCP train epochs</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>number of SGCP training epochs</td></tr><tr><td>SGCP learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 2 }$ </td><td>10-3</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td>learning rate for SGCP training</td></tr><tr><td>λ</td><td>0.5</td><td>0.05</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td>0.5</td><td>weight of negative log likelihood loss</td></tr><tr><td>β</td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td><td>0.5</td><td> $1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>weight of balance regularization</td></tr><tr><td> $M _ { \mathrm { t r a i n } }$ </td><td>3</td><td>6</td><td>3</td><td>3</td><td>8</td><td>posterior samples during training</td></tr><tr><td> $M _ { \mathrm { e v a l } }$ </td><td>8</td><td>8</td><td>8</td><td>16</td><td>8</td><td>posterior samples during evaluation</td></tr><tr><td colspan="7">(iii) Backbone &amp; optimization</td></tr><tr><td>train epochs</td><td>200</td><td>200</td><td>40</td><td>40</td><td>100</td><td>number of backbone training epochs</td></tr><tr><td>learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>10-3</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td>learning rate for backbone training</td></tr><tr><td>batch size</td><td>32</td><td>128</td><td>128</td><td>128</td><td>32</td><td>mini batch size</td></tr><tr><td>weight decay</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>optimizer weight decay</td></tr></table>

Note: M-ICU denotes MIMIC-IV (ICU delay) and M-Hosp denotes MIMIC-IV (in hospital).

Table A4: Ablation study of SGCP over 20 runs. Results with nominal coverage are shown in blue, optimal values are boldfaced, and AvgSize exceeding half of the total number of categories is highlighted in red. CovGap is reported in percent.
<table><tr><td rowspan="2">Method</td><td colspan="3">SYN</td><td colspan="3">Nursery</td><td colspan="3">MIMIC</td><td colspan="3">BACH</td></tr><tr><td>Cov.</td><td>AvgSize</td><td>CovGap</td><td>Cov.</td><td>AvgSize</td><td>CovGap</td><td>Cov.</td><td>AvgSize</td><td>CovGap</td><td>Cov.</td><td>AvgSize</td><td>CovGap</td></tr><tr><td>Default SGCP</td><td></td><td> $9 0 . 1 3 _ { \pm 0 . 3 9 } \ 2 . 2 2 5 _ { \pm 0 . 0 3 5 }$ </td><td> $4 . 2 _ { \pm 0 . 7 }$ </td><td></td><td>91.65±0.70 1.089±0.035</td><td> ${ \bf 1 1 . 8 _ { \pm 0 . 7 } }$ </td><td>90.13±0.07</td><td> ${ \bf 1 . 1 0 7 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 8 { \scriptstyle \pm 0 . 0 }$ </td><td> $\mathbf { 9 1 . 9 0 _ { \pm 2 . 7 0 } }$ </td><td>2.224±0.215</td><td> ${ \bf 6 . 2 \mathrm { _ { \pm 1 . 1 } } }$ </td></tr><tr><td>(a) w/o Stochastic</td><td></td><td> $8 9 . 8 2 _ { \pm 1 . 4 8 } 2 . 3 4 8 _ { \pm 0 . 0 8 1 }$ </td><td></td><td> $7 . 3 5 _ { \pm 1 . 1 8 } 9 1 . 3 8 _ { \pm 1 . 1 5 }$ </td><td> $1 . 1 2 3 _ { \pm 0 . 0 5 1 }$ </td><td> $1 3 . 4 2 _ { \pm 1 . 7 6 }$ </td><td></td><td>89.94±0.96 1.053±0.013</td><td> $2 . 3 6 _ { \pm 0 . 4 8 }$ </td><td> $9 1 . 7 6 _ { \pm 4 . 4 2 }$ </td><td> $2 . 4 4 7 _ { \pm 0 . 3 0 8 }$ </td><td> $8 . 8 4 _ { \pm 2 . 3 5 }$ </td></tr><tr><td>(b) Feature space</td><td></td><td> $9 0 . 9 6 _ { \pm 1 . 9 6 } \ 2 . 4 7 6 _ { \pm 0 . 1 2 7 }$ </td><td> $7 . 1 8 _ { \pm 1 . 4 6 } ~ 9 0 . 9 2 _ { \pm 1 . 4 3 }$ </td><td></td><td> $1 . 2 4 8 _ { \pm 0 . 0 7 9 }$ </td><td>14.16±2.05</td><td> $9 0 . 4 8 _ { \pm 1 . 4 7 }$ </td><td> $1 . 1 7 9 _ { \pm 0 . 0 4 4 }$ </td><td> $4 . 0 8 _ { \pm 1 . 0 8 }$ </td><td> $8 9 . 6 1 _ { \pm 5 . 7 4 }$ </td><td> $3 . 0 9 4 _ { \pm 0 . 4 4 6 }$ </td><td> $1 2 . 4 1 _ { \pm 3 . 1 6 }$ </td></tr><tr><td>(c) w/o rank</td><td> $8 9 . 2 1 { \scriptstyle \pm 4 . 4 2 }$ </td><td>1.898±0.246</td><td> $8 . 4 2 _ { \pm 4 . 1 0 }$ </td><td> $8 8 . 6 3 \substack { \pm 2 . 9 4 }$ </td><td> $1 . 3 0 2 _ { \pm 0 . 1 1 8 }$ </td><td> $1 5 . 3 1 _ { \pm 2 . 4 1 }$ </td><td> $9 0 . 0 3 { \scriptstyle \pm 5 . 4 1 }$ </td><td> $1 . 3 3 4 _ { \pm 0 . 1 4 6 }$ </td><td> $5 . 4 9 _ { \pm 3 . 7 1 }$ </td><td> $9 1 . 1 8 _ { \pm 6 . 3 1 }$ </td><td> $3 . 4 9 1 _ { \pm 0 . 5 4 1 }$ </td><td> $9 . 2 6 _ { \pm 3 . 0 2 }$ </td></tr><tr><td>(d) w/o NLL</td><td></td><td> $8 8 . 3 6 _ { \pm 1 . 1 7 } \ 2 . 0 7 4 _ { \pm 0 . 1 0 8 }$ </td><td> $5 . 3 9 _ { \pm 1 . 0 8 }$ </td><td>92.39±0.92</td><td> $1 . 4 4 6 _ { \pm 0 . 0 9 4 }$ </td><td> $1 2 . 4 3 _ { \pm 1 . 1 9 }$ </td><td> $9 0 . 0 2 _ { \pm 2 . 9 6 }$ </td><td>1.351±0.078</td><td> $2 . 9 6 _ { \pm 0 . 7 7 }$ </td><td>93.11±2.96 2.846±0.317</td><td></td><td> $7 . 4 4 _ { \pm 1 . 4 7 }$ </td></tr><tr><td>(e) w/o MI</td><td></td><td></td><td></td><td></td><td></td><td>92.28±3.41 2.438±0.241 6.53±2.72 91.26±2.43 1.149±0.147 14.72±1.42</td><td> $8 9 . 8 4 _ { \pm 6 . 3 1 }$ </td><td></td><td>1.047±0.116 4.95±2.31 90.91±7.82 2.347±0.472</td><td></td><td></td><td> $9 . 3 1 _ { \pm 2 . 1 1 }$ </td></tr></table>

For subgroup evaluation, we define three binary grouping attributes. The race indicator minority equals 1 if the recorded race string does not contain WHITE; gender\_m equals 1 for male patients; and public\_insurance equals 1 if the primary insurance is MEDICARE or MEDICAID. These attributes are used only for subgroup construction and fairness evaluation. Model inputs include age, one-hot indicators for insurance type, admission type, and marital status, ICU length of stay, and the number of diagnosis codes; ICU length of stay is excluded for ICU-delay prediction.

We consider two tasks. The first is in-hospital mortality prediction, with binary labels derived from hospital\_expire\_flag. Results for this task are reported as MIMIC-IV (in-hospital) in Figures 5, A9, and A10. The second is ICU-delay prediction, reported as MIMIC-IV (ICU delay), where admissions are categorized by the time from hospital admission to first ICU entry:

$$
Y = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ I C U ~ d e l a y } } \leq 4 { \mathrm { ~ h o u r s } } , } \\ { 1 , } & { { \mathrm { i f ~ I C U ~ d e l a y } } \in ( 4 , 1 2 ] { \mathrm { ~ h o u r s } } , } \\ { 2 , } & { { \mathrm { i f ~ I C U ~ d e l a y } } > 1 2 { \mathrm { ~ h o u r s ~ o r ~ n o ~ I C U ~ a d m i s s i o n ~ o c c u r s } } . } \end{array} \right. }
$$

Together, these tasks provide a real-world clinical benchmark for evaluating fairness across race, gender, and insurance-defined subgroups.

BACH. The Breast Cancer Histology (BACH) dataset [2] is a histopathology benchmark for evaluating subgroup reliability under visual heterogeneity. We use Part A, which contains 400 H&E-stained microscopy images evenly distributed across four classes: normal, benign, in situ carcinoma, and invasive carcinoma. Images were acquired at 20× magnification with resolution $2 0 4 8 \times 1 5 3 6$ , annotated by two pathologists, and collected from 39 patients, though anonymization prevents complete patient-level tracing. The task is four-class classification with

$$
Y \in \{ \mathrm { N o r m a l } , \mathrm { B e n i g n } , \mathrm { I n ~ S i t u ~ C a r c i n o m a } , \mathrm { I n v a s i v e ~ C a r c i n o m a } \} .
$$

Since demographic sensitive attributes are unavailable, we use the class label as a proxy grouping variable and set $A : = Y$ . Subgroup reliability is therefore evaluated across disease categories, which naturally differ in visual complexity, intra-class variability, and class-boundary ambiguity. For preprocessing, images are resized to $2 2 4 \times 2 2 4$ , augmented during training with random horizontal and vertical flips, and normalized using ImageNet statistics. We train a visual backbone Φ and use the learned representations $X _ { i } = \Phi ( I _ { i } ) \mathbf { \bar { \Psi } } \in \mathbb { R } ^ { \breve { d } }$ for downstream prediction and evaluation.

Table A5: Performance comparison w.r.t. sample sparsity on the Synthetic Dataset (Target $1 - \alpha =$ 0.90). Takeaway: SGCP remains consistently close to nominal coverage across all sample sizes while achieving the smallest subgroup coverage gap and comparatively small prediction sets. And the performance of each specific sensitive group is shown in the Fig. A6.
<table><tr><td rowspan="2">Method</td><td colspan="3">N = 500</td><td colspan="3">N = 1000</td><td colspan="3">N = 2000</td><td colspan="3">N = 4000</td></tr><tr><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓ Cov. ↑</td><td></td><td></td><td>AvgSize ↓ CovGap ↓ Cov. ↑</td><td></td><td></td><td>AvgSize ↓ CovGap ↓ Cov. ↑</td><td></td><td></td><td>AvgSize ↓</td><td> $\mathbf { C o v G a p } \downarrow$ </td></tr><tr><td>Marginal</td><td>90.15±6.50</td><td> $2 . 1 5 0 _ { \pm 0 . 4 2 0 }$ </td><td></td><td></td><td> $3 2 . 5 _ { \pm 6 . 5 } 9 0 . 2 0 _ { \pm 4 . 2 0 } 1 . 9 5 0 _ { \pm 0 . 4 0 0 }$ </td><td> $2 1 . 5 _ { \pm 6 . 2 }$ </td><td></td><td> $9 0 . 1 2 _ { \pm 4 . 1 0 } 1 . 8 2 0 _ { \pm 0 . 3 8 0 }$ </td><td> $1 9 . 7 _ { \pm 6 . 0 }$ </td><td></td><td> $8 9 . 5 2 _ { \pm 1 . 5 2 } 2 . 3 0 9 _ { \pm 0 . 0 7 8 }$ </td><td> $1 4 . 5 _ { \pm 1 . 6 }$ </td></tr><tr><td>Partial</td><td>95.50±3.20</td><td> $4 . 5 2 0 { \scriptstyle \pm 0 . 3 0 0 }$ </td><td> $9 . 5 _ { \pm 2 . 8 }$ </td><td></td><td> $9 4 . 2 0 _ { \pm 3 . 0 0 } 2 . 9 5 0 _ { \pm 0 . 2 8 0 }$ </td><td> $7 . 9 _ { \pm 2 . 5 }$ </td><td></td><td> $9 3 . 5 0 _ { \pm 2 . 8 0 } 2 . 8 2 0 _ { \pm 0 . 2 5 0 }$ </td><td> $7 . 5 _ { \pm 2 . 2 }$ </td><td></td><td> $9 4 . 5 0 _ { \pm 1 . 1 2 } 2 . 4 1 9 _ { \pm 0 . 0 4 8 }$ </td><td> $6 . 2 _ { \pm 0 . 7 }$ </td></tr><tr><td></td><td></td><td>Exhaustive 100.00±0.00 6.000±0.000</td><td></td><td></td><td> $1 2 . 5 _ { \pm 3 . 2 } 9 8 . 5 0 _ { \pm 2 . 2 0 } 4 . 8 5 0 _ { \pm 0 . 2 5 0 }$ </td><td> $1 0 . 5 { \scriptstyle \pm 3 . 0 }$ </td><td></td><td> $9 8 . 2 0 { \scriptstyle \pm 2 . 1 0 } 4 . 1 5 0 { \scriptstyle \pm 0 . 2 2 0 }$ </td><td> $7 . 9 { \scriptstyle \pm 2 . 7 }$ </td><td></td><td> $9 5 . 8 0 { \scriptstyle \pm 0 . 9 1 } 3 . 3 4 2 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $6 . 0 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>RLCP</td><td></td><td>95.80±3.80 2.910±0.520</td><td></td><td></td><td> $9 . 2 _ { \pm 8 . 2 } 9 5 . 1 0 _ { \pm 3 . 2 0 } 2 . 8 5 0 _ { \pm 0 . 4 8 0 }$ </td><td> $8 . 5 { \scriptstyle \pm 6 . 8 }$ </td><td></td><td> $9 4 . 6 0 { \scriptstyle \pm 2 . 9 0 } 2 . 8 5 0 { \scriptstyle \pm 0 . 4 6 0 }$ </td><td> $6 . 4 { \scriptstyle \pm 5 . 5 }$ </td><td></td><td> $9 3 . 1 5 _ { \pm 2 . 4 5 } 2 . 5 4 0 _ { \pm 0 . 3 0 5 }$ </td><td> $4 . 5 { \pm } 1 . 8$ </td></tr><tr><td>CluCP</td><td></td><td> $9 0 . 2 2 _ { \pm 2 . 0 6 } 3 . 1 1 2 _ { \pm 0 . 3 3 2 }$ </td><td></td><td></td><td> $1 0 . 7 _ { \pm 0 . 9 } 9 0 . 6 2 _ { \pm 0 . 9 8 } 2 . 7 2 2 _ { \pm 0 . 2 6 9 }$ </td><td> $9 . 9 _ { \pm 0 . 7 }$ </td><td></td><td> $9 1 . 1 1 _ { \pm 1 . 8 3 } 2 . 7 5 6 _ { \pm 0 . 2 5 0 }$ </td><td> $8 . 9 _ { \pm 1 . 1 }$ </td><td></td><td> $8 9 . 8 1 _ { \pm 1 . 7 8 } 2 . 4 2 6 _ { \pm 0 . 0 5 0 }$ </td><td> $4 . 9 _ { \pm 0 . 8 }$ </td></tr><tr><td>AFCP</td><td></td><td> $9 0 . 6 7 _ { \pm 1 . 4 4 } 3 . 1 6 2 _ { \pm 0 . 2 9 2 }$ </td><td> $6 . 4 _ { \pm 0 . 8 }$ </td><td></td><td> $9 1 . 3 0 _ { \pm 1 . 2 4 } 2 . 8 5 4 _ { \pm 0 . 3 2 8 }$ </td><td> $7 . 6 _ { \pm 0 . 7 }$ </td><td></td><td> $9 1 . 6 2 _ { \pm 1 . 7 1 } 2 . 8 4 8 _ { \pm 0 . 2 2 7 }$ </td><td> $7 . 3 _ { \pm 0 . 9 }$ </td><td></td><td> $9 1 . 7 7 _ { \pm 1 . 2 3 } 2 . 4 3 2 _ { \pm 0 . 0 4 3 }$ </td><td> $4 . 6 _ { \pm 0 . 5 }$ </td></tr><tr><td>FaReG</td><td></td><td> $9 2 . 1 8 _ { \pm 1 . 2 6 } 3 . 5 3 6 _ { \pm 0 . 3 8 8 }$ </td><td> $7 . 5 _ { \pm 0 . 6 }$ </td><td></td><td> $9 2 . 4 1 _ { \pm 1 . 1 3 } 3 . 1 7 8 _ { \pm 0 . 2 8 8 }$ </td><td> $8 . 5 { \scriptstyle \pm 0 . 6 }$ </td><td></td><td> $9 2 . 4 9 _ { \pm 1 . 5 7 } 3 . 1 4 1 _ { \pm 0 . 2 3 8 }$ </td><td> $8 . 5 _ { \pm 0 . 9 }$ </td><td></td><td> $9 2 . 6 1 _ { \pm 0 . 9 8 } 2 . 5 7 9 _ { \pm 0 . 0 2 9 }$ </td><td> $4 . 7 _ { \pm 0 . 5 }$ </td></tr><tr><td>SGCP</td><td> $\mathbf { 9 1 . 2 0 _ { \pm 1 . 2 0 } 2 . 0 5 0 _ { \pm 0 . 1 2 0 } }$ </td><td></td><td></td><td> ${ \bf 6 . 1 _ { \pm 0 . 9 } }$ </td><td> $9 0 . 8 0 _ { \pm 1 . 1 0 } 1 . 8 2 0 _ { \pm 0 . 1 1 0 }$ </td><td> ${ \bf 5 . 1 _ { \pm 0 . 8 } }$ </td><td></td><td> $\mathbf { 9 0 . 5 0 _ { \pm 1 . 0 0 } 1 . 7 1 0 _ { \pm 0 . 0 9 0 } }$ </td><td> ${ \bf 4 . 7 _ { \pm 0 . 8 } }$ </td><td> ${ \bf 9 0 . 1 3 _ { \pm 0 . 3 9 } }$ </td><td>2.225±0.035</td><td> ${ \bf 4 . 2 _ { \pm 0 . 7 } }$ </td></tr></table>

Table A6: Dataset access and preprocessing sources.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Access note</td></tr><tr><td rowspan=1 colspan=1>Synthetic</td><td rowspan=1 colspan=1>Procedurally generated; see Appendix D.1</td><td rowspan=1 colspan=1>Generated from the described simulation protocol</td></tr><tr><td rowspan=1 colspan=1>Nursery</td><td rowspan=1 colspan=1>https://archive.ics.uci.edu/dataset/76/nursery</td><td rowspan=1 colspan=1>Public UCI dataset</td></tr><tr><td rowspan=1 colspan=1>MIMIC-IV</td><td rowspan=1 colspan=1>https://physionet.org/content/mimiciv/3.1/</td><td rowspan=1 colspan=1>Credentialed access via PhysioNet</td></tr><tr><td rowspan=1 colspan=1>BACH</td><td rowspan=1 colspan=1>https://zenodo.org/records/3632035</td><td rowspan=1 colspan=1>Public Zenodo record</td></tr></table>

In this benchmark, heterogeneity is primarily visual rather than demographic. Differences in tissue morphology, staining, lesion structure, and acquisition conditions can induce distinct class-conditional score laws, $F _ { c } ( s ) : = \mathbb { P } ( S \le s \ | \ Y = c )$ . Thus, BACH complements the tabular benchmarks by testing whether SGCP can maintain reliable conditional coverage when subgroup structure reflects semantic visual heterogeneity rather than explicit sensitive attributes.

## D.2 Implementation Details

Computing resources. All experiments are conducted on a workstation equipped with an Intel i5-14600KF CPU, 32 GB of RAM, and a single NVIDIA GeForce RTX 4090 D GPU. All models are implemented in PyTorch, and each experimental setting is trained with multiple random seeds. The hyperparameters for each dataset are reported in Table A3. Data access and code availability are summarized in Tables A6 and A7, respectively.

Open access to data and code. All external datasets used in our experiments are publicly documented and accessible online, subject to their original access requirements. The synthetic dataset is generated procedurally following Appendix D.1. The code for SGCP is provided anonymously in the supplementary material, and baseline implementations are based on the publicly available repositories listed below.

## D.2.1 Baseline implementation details

We compare SGCP with six baselines: Partial Equalized, Exhaustive Equalized, CluCP, AFCP, FaReG, and RLCP. Unless otherwise stated, all methods use the same backbone classifier and softmax-based nonconformity score as SGCP. Hyperparameters are adopted from the original papers or released implementations whenever available.

Partial and Exhaustive Equalized. We extend the equalized-coverage framework of [20] to conformal prediction by applying group-conditional calibration over sensitive groups. Partial Equalized calibrates separately for each value of one sensitive attribute, whereas Exhaustive Equalized calibrates over all intersectional groups induced by the full combination of sensitive attributes. In both cases, the classifier is trained on all samples, and only the calibration step is group-conditional.

Table A7: Code availability and baseline implementations.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Code source</td></tr><tr><td rowspan=1 colspan=1>SGCP</td><td rowspan=1 colspan=1>https://anonymous.4open.science/r/sgcp/</td></tr><tr><td rowspan=1 colspan=1>Partial Equalized</td><td rowspan=1 colspan=1>https://github.com/yromano/cqr</td></tr><tr><td rowspan=1 colspan=1>RLCP</td><td rowspan=1 colspan=1>https://github.com/rohanhore/RLCP</td></tr><tr><td rowspan=1 colspan=1>CluCP</td><td rowspan=1 colspan=1>https://github.com/tiffanyding/class-conditional-conformal</td></tr><tr><td rowspan=1 colspan=1>AFCP</td><td rowspan=1 colspan=1>https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction</td></tr><tr><td rowspan=1 colspan=1>FaReG</td><td rowspan=1 colspan=1>https://github.com/Xusr1123/FaReG</td></tr></table>

![](images/a970eeb5ac974932e28a83aae0c7a65e8e530c921e82a00f254c345837f4fc05.jpg)  
Figure A6: Group coverage on SYN dataset. SGCP remains close to the target coverage across classes while using smaller prediction sets than more conservative baselines.

CluCP. We implement CluCP following its original clustered conformal pipeline. The clustering fraction γ and the number of clusters M are selected using the heuristic proposed in the paper. Class embeddings are clustered by weighted k-means with the default KMeans implementation. Although CluCP targets class-wise rather than subgroup-fair calibration, it provides a useful nonglobal calibration baseline.

AFCP. We implement AFCP using its original adaptive attribute-selection procedure, directly applying it to the sensitive attributes available in each dataset.

FaReG. We follow the original FaReG representation-learning pipeline: an encoder-decoder first learns latent representations, representation-based low-coverage groups are then identified, and prediction sets are adjusted accordingly. We use the original three-layer MLP encoder and decoder. For Synthetic and Nursery, we adopt the reported hyperparameters: hidden dimensions [64, 32], batch size 500, learning rates $\mathrm { i 0 ^ { - 3 } }$ and 10<sup>−2</sup>, respectively, with $( \beta , \delta , T ) = ( 2 . 0 , 0 . 3 , 2 0 )$ for Synthetic and (0.1, 0.1, 100) for Nursery. For the remaining datasets, we keep the architecture and optimization settings as close as possible to the released implementation.

RLCP. We implement RLCP following its weighted localization procedure. Calibration samples are weighted by their similarity to each test point, and prediction sets are constructed using the corresponding weighted conformal quantile. For all datasets, we use the Gaussian kernel and the bandwidth-selection rule from the original method, without subgroup-specific tuning.

## E Additional Experiments

## E.1 Ablation Study

Table A4 studies the contribution of each component in SGCP. The default model achieves nominal marginal coverage on all datasets and obtains the best or tied-best CovGap on three out of four datasets, while maintaining the smallest or near-smallest prediction sets. This suggests that the full objective provides the most favorable balance between validity, subgroup reliability, and efficiency.

Removing posterior stochasticity consistently degrades subgroup reliability. The deterministic variant increases CovGap on all datasets, indicating that posterior averaging is useful for stabilizing the local score law rather than committing each sample to a single calibration state. Replacing the learned calibration state with feature-space grouping also leads to larger CovGap and larger prediction sets, especially on MIMIC and BACH. This supports our motivation that similarity in raw feature space is not necessarily the right notion of locality for conformal calibration; what matters is similarity in score behavior.

![](images/8e6d8d7a99cf9e1191206eddce7aeeb76a7fd7e3f170ff3e223106d812169414.jpg)  
Figure A7: Group coverage on Nursery. Top: subgroup coverage vs. average prediction set size at a sample size of 4000, with the dashed line indicating nominal coverage of 0.9. Bottom: prediction set size varies with subgroup sample size. SGCP stays close to the nominal target while remaining among the most efficient methods across subgroup definitions.

Table A8: Performance comparison w.r.t. sample sparsity on the Nursery Dataset (Target $1 - \alpha =$ 0.90). Takeaway: SGCP remains near nominal coverage even in the sparse regime and consistently achieves the smallest subgroup coverage gap, while competing methods improve subgroup reliability only with larger prediction sets or at the cost of under- or over-coverage. And the performance of each specific sensitive group is shown in the Fig. A7.
<table><tr><td rowspan="2">Method</td><td colspan="3">N = 500</td><td colspan="3">N = 1000</td><td colspan="3">N = 2000</td><td colspan="3">N = 4000</td></tr><tr><td> $\mathbf { C o v . \uparrow }$ </td><td>AvgSize ↓ CovGap ↓</td><td></td><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓</td><td></td><td> $\mathbf { C o v . \uparrow }$ </td><td>AvgSize ↓ CovGap ↓ Cov. ↑</td><td></td><td></td><td>AvgSize ↓ CovGap ↓</td><td></td></tr><tr><td>Marginal</td><td> $4 8 . 2 5 _ { \pm 8 . 5 0 }$ </td><td> $1 . 1 2 0 _ { \pm 0 . 3 5 0 }$ </td><td> $2 9 . 5 _ { \pm 6 . 5 }$ </td><td> $5 7 . 5 9 _ { \pm 8 . 2 0 }$ </td><td>1.080±0.330</td><td> $2 9 . 1 _ { \pm 5 . 2 }$ </td><td> $5 9 . 7 2 _ { \pm 7 . 8 0 }$ </td><td> $1 . 0 5 0 _ { \pm 0 . 3 1 0 }$ </td><td> $2 6 . 2 _ { \pm 5 . 8 }$ </td><td></td><td>84.15±5.20 2.170±0.260</td><td> $2 0 . 4 _ { \pm 5 . 2 }$ </td></tr><tr><td>Partial</td><td> $8 4 . 6 1 _ { \pm 4 . 8 0 }$ </td><td>2.850±0.310</td><td> $1 7 . 7 _ { \pm 3 . 8 }$ </td><td> $8 6 . 0 7 _ { \pm 4 . 5 0 }$ </td><td> $2 . 6 2 0 { \scriptstyle \pm 0 . 2 8 0 }$ </td><td> $1 6 . 8 _ { \pm 3 . 5 }$ </td><td> $8 7 . 8 6 _ { \pm 4 . 2 0 }$ </td><td> $2 . 4 5 0 _ { \pm 0 . 2 5 0 }$ </td><td> $1 5 . 2 _ { \pm 3 . 2 }$ </td><td> $8 8 . 9 7 _ { \pm 3 . 8 0 }$ </td><td> $1 . 8 5 0 _ { \pm 0 . 2 0 0 }$ </td><td> $1 2 . 1 _ { \pm 2 . 5 }$ </td></tr><tr><td></td><td>Exhaustive 100.00±0.004.000±0.000</td><td></td><td> $1 9 . 5 _ { \pm 0 . 1 }$ </td><td> $9 8 . 0 0 _ { \pm 0 . 1 0 }$ </td><td> $3 . 9 8 0 _ { \pm 0 . 0 0 1 }$ </td><td> $1 8 . 8 _ { \pm 0 . 2 }$ </td><td> $9 8 . 0 0 _ { \pm 0 . 1 2 }$ </td><td> $3 . 9 7 9 _ { \pm 0 . 0 0 1 }$ </td><td> $1 7 . 5 _ { \pm 0 . 8 }$ </td><td> $9 6 . 8 0 _ { \pm 3 . 1 0 }$ </td><td> $2 . 8 5 0 _ { \pm 0 . 2 2 0 }$ </td><td> $1 5 . 5 _ { \pm 3 . 2 }$ </td></tr><tr><td>RLCP</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.45±14.20 2.250±0.580 18.5±9.5 83.36±13.80 1.720±0.550 17.8±6.2 83.31±11.50 1.480±0.510 16.5±5.8 88.67±6.20 1.220±0.410</td><td> $1 3 . 5 { \scriptstyle \pm 3 . 8 }$ </td></tr><tr><td>CluCP</td><td></td><td>86.26±5.61 1.506±0.265 17.3±4.5 83.80±3.59 1.516±0.265</td><td></td><td></td><td></td><td> $1 7 . 1 { \pm } 2 . 9$ </td><td></td><td>84.42±2.47 1.261±0.096</td><td> $1 6 . 9 { \scriptstyle \pm 1 . 8 }$ </td><td></td><td>85.08±1.60 1.121±0.064 12.8±1.3</td><td></td></tr><tr><td>AFCP</td><td></td><td>92.78±4.81 1.551±0.345</td><td> $1 8 . 7 _ { \pm 3 . 9 }$ </td><td></td><td>94.85±3.69 1.163±0.074</td><td> $1 6 . 4 { \scriptstyle \pm 2 . 8 }$ </td><td> $9 3 . 1 2 { \scriptstyle \pm 1 . 3 3 }$ </td><td>1.092±0.031</td><td> $1 5 . 6 { \scriptstyle \pm 1 . 1 }$ </td><td> $9 2 . 8 5 _ { \pm 0 . 8 4 }$ </td><td>1.085±0.024 14.1±0.6</td><td></td></tr><tr><td>FaReG</td><td></td><td>93.77±3.66 1.879±0.515</td><td> $1 7 . 4 { \scriptstyle \pm 2 . 9 }$ </td><td>92.47±3.04</td><td> $1 . 3 4 6 _ { \pm 0 . 1 4 6 }$ </td><td> $1 6 . 2 { \scriptstyle \pm 2 . 3 }$ </td><td> $9 4 . 3 7 { \scriptstyle \pm 1 . 4 4 }$ </td><td>1.257±0.027</td><td> $1 5 . 4 { \scriptstyle \pm 1 . 1 }$ </td><td>93.21±0.62</td><td> $1 . 2 3 0 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $1 3 . 3 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>SGCP</td><td> $9 2 . 7 2 _ { \pm 1 . 5 0 }$ </td><td>1.320±0.080</td><td> ${ \bf 1 5 . 1 _ { \pm 1 . 5 } }$ </td><td> $\mathbf { 9 2 . 1 0 _ { \pm 1 . 2 0 } }$ </td><td> $1 . 2 1 0 _ { \pm 0 . 0 6 0 }$ </td><td> $\pm 4 . 6 _ { \pm 1 . 2 }$ </td><td> ${ \bf 9 1 . 9 7 } _ { \pm 1 . 0 0 }$ </td><td> $1 . 1 2 0 _ { \pm 0 . 0 5 0 }$ </td><td> $1 2 . 3 _ { \pm 1 . 0 }$ </td><td></td><td>91.65±0.701.089±0.035</td><td> ${ \bf 1 1 . 8 _ { \pm 0 . 7 } }$ </td></tr></table>

The loss ablations further clarify the role of each training signal. Without the rank-uniformization loss, coverage becomes less stable and CovGap increases substantially, showing that aligning transformed scores to a common percentile scale is central to subgroup reliability. Removing the density-fitting term also worsens the tradeoff between coverage and efficiency, often producing larger prediction sets, which suggests that likelihood-based fitting helps the learned local law remain faithful to the observed score distribution. Finally, removing the mutual-information term leads to less reliable subgroup calibration and higher variance across runs, consistent with component underuse or less informative stochastic grouping. Overall, the ablation results support the design of SGCP: subgroup reliability improves when local score laws are learned through both distributional fitting and rank-based normalization, and when calibration evidence is shared through informative stochastic components.

## E.2 Sensitivity to Hyperparameters and Subgroup Configurations

## E.2.1 Effect of sample sparsity

Tables A5–A10 study how performance changes as the effective sample size varies. As the effective sample size decreases, methods based on explicit subgroup equalization or localization tend to become increasingly conservative, while global calibration remains efficient but suffers from large subgroup disparity. By contrast, SGCP degrades more gracefully in the sparse regime and benefits more consistently from additional data as sample size grows. This pattern is strongest on the synthetic benchmark and on BACH, where subgroup heterogeneity is more structured, and remains visible on MIMIC-IV, where SGCP is particularly competitive in the sparse and moderate sample regimes.

Table A9: Performance comparison w.r.t. sample sparsity on MIMIC-IV (target $1 - \alpha = 0 . 9 0 )$ Each method is run 20 times. Takeaway: On the MIMIC-IV dataset, SGCP consistently achieves near-nominal coverage while maintaining the lowest subpopulation disparity (CovGap) in sparse settings $( N = 2 0 0 0 , 5 0 0 0 )$ . Even with the full dataset, SGCP provides a highly efficient prediction set with excellent fairness metrics compared to conventional baselines.
<table><tr><td rowspan="2">Method</td><td colspan="3">N = 2000</td><td colspan="3">N = 5000</td><td colspan="3">Full (All Samples)</td></tr><tr><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓</td><td></td><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓</td><td></td><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓</td><td></td></tr><tr><td>Marginal</td><td> $9 0 . 0 6 _ { \pm 1 0 . 1 4 }$ </td><td> $1 . 0 7 4 _ { \pm 0 . 2 1 1 }$ </td><td> $3 4 . 4 _ { \pm 1 0 . 9 }$ </td><td> $9 0 . 7 9 _ { \pm 1 0 . 6 2 }$ </td><td> $\mathbf { 1 . 0 2 6 _ { \pm 0 . 0 8 7 } }$ </td><td> $2 9 . 6 { \scriptstyle \pm 9 . 3 }$ </td><td> $9 0 . 7 9 _ { \pm 1 0 . 6 2 }$ </td><td> $1 . 2 2 6 _ { \pm 0 . 0 8 7 }$ </td><td> $2 7 . 1 _ { \pm 7 . 1 }$ </td></tr><tr><td>Partial</td><td> $9 5 . 7 3 _ { \pm 3 . 6 2 }$ </td><td> $1 . 5 6 4 _ { \pm 0 . 2 4 3 }$ </td><td> $1 1 . 0 { \scriptstyle \pm 5 . 6 }$ </td><td> $9 4 . 5 0 { \scriptstyle \pm 4 . 5 4 }$ </td><td> $1 . 3 3 5 _ { \pm 0 . 2 0 7 }$ </td><td> $9 . 2 _ { \pm 4 . 0 }$ </td><td> $9 4 . 5 0 { \scriptstyle \pm 4 . 5 4 }$ </td><td> $1 . 2 5 8 _ { \pm 0 . 1 4 0 }$ </td><td> $8 . 0 { \scriptstyle \pm 3 . 0 }$ </td></tr><tr><td>Exhaustive</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 8 . 7 _ { \pm 8 . 0 }$ </td><td> $9 8 . 8 6 _ { \pm 5 . 2 1 }$ </td><td> $2 . 9 8 9 _ { \pm 0 . 0 0 4 }$ </td><td> $2 1 . 4 { \scriptstyle \pm 5 . 3 }$ </td><td> $9 6 . 4 5 _ { \pm 4 . 2 2 }$ </td><td> $2 . 8 0 0 { \scriptstyle \pm 0 . 1 4 2 }$ </td><td> $1 9 . 7 _ { \pm 3 . 2 }$ </td></tr><tr><td>RLCP</td><td> $9 5 . 1 0 { \scriptstyle \pm 2 . 4 0 }$ </td><td> $1 . 6 5 0 { \scriptstyle \pm 0 . 1 2 0 }$ </td><td> $4 . 2 _ { \pm 2 . 6 }$ </td><td> $9 3 . 2 0 { \scriptstyle \pm 2 . 1 0 }$ </td><td> $1 . 4 8 0 _ { \pm 0 . 1 5 0 }$ </td><td> $3 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td> $9 2 . 5 0 { \scriptstyle \pm 2 . 8 0 }$ </td><td> $1 . 3 5 0 _ { \pm 0 . 1 0 0 }$ </td><td> $1 . 5 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td>CluCP</td><td> $9 5 . 1 1 _ { \pm 1 . 9 2 }$ </td><td> $1 . 6 5 4 _ { \pm 0 . 1 9 2 }$ </td><td> $4 . 5 _ { \pm 1 . 5 }$ </td><td> $9 3 . 1 9 _ { \pm 1 . 2 8 }$ </td><td> $1 . 4 2 2 _ { \pm 0 . 0 5 9 }$ </td><td> $3 . 5 { \scriptstyle \pm 1 . 0 }$ </td><td> $9 2 . 2 2 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $1 . 3 0 2 _ { \pm 0 . 0 5 9 }$ </td><td> $3 . 5 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td>AFCP</td><td> $9 1 . 8 1 { \scriptstyle \pm 2 . 5 1 }$ </td><td> $1 . 4 0 0 _ { \pm 0 . 0 7 5 }$ </td><td> $3 . 9 _ { \pm 1 . 0 }$ </td><td> $9 1 . 4 0 _ { \pm 1 . 1 2 }$ </td><td> $1 . 1 1 4 _ { \pm 0 . 0 4 5 }$ </td><td> $3 . 3 { \scriptstyle \pm 0 . 8 }$ </td><td></td><td></td><td></td></tr><tr><td>FaReG</td><td> $9 5 . 7 8 _ { \pm 2 . 0 9 }$ </td><td> $1 . 6 0 1 { \scriptstyle \pm 0 . 0 7 8 }$ </td><td> $4 . 1 _ { \pm 1 . 2 }$ </td><td> $9 2 . 5 8 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $1 . 3 5 7 _ { \pm 0 . 0 5 0 }$ </td><td> $3 . 0 { \scriptstyle \pm 1 . 1 }$ </td><td> $9 0 . 2 3 _ { \pm 0 . 0 9 }$ </td><td> $1 . 1 7 4 _ { \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 5 _ { \pm 0 . 0 } }$ </td></tr><tr><td>SGCP</td><td> $\mathbf { 9 0 . 1 2 _ { \pm 1 . 9 1 } }$ </td><td> $\mathbf { 1 . 0 6 0 _ { \pm 0 . 0 6 8 } }$ </td><td> $3 . 3 _ { \pm 0 . 7 }$ </td><td> $\mathbf { 9 0 . 4 8 _ { \pm 1 . 3 6 } }$ </td><td> $1 . 1 1 6 _ { \pm 0 . 0 5 0 }$ </td><td> $\mathbf { 2 . 6 _ { \pm 0 . 5 } }$ </td><td> $\mathbf { 9 0 . 1 7 _ { \pm 0 . 0 7 } }$ </td><td> ${ \bf 1 . 1 0 7 _ { \pm 0 . 0 0 1 } }$ </td><td> $0 . 8 _ { \pm 0 . 0 }$ </td></tr></table>

![](images/e8b06678b91af334100cfff63cb67f2436021ae1883f3f755e88bd557cae78b9.jpg)  
Figure A8: Group coverage on MIMIC-IVdataset. SGCP remains close to the target coverage across classes while using smaller prediction sets than more conservative baselines.

For a clearer comparison, we also provide separate box plots for each metric across methods and sample sizes. Figures A9, A10, and 5 show the distributions of overall coverage, average prediction set size, and subgroup coverage gap under varying sample sparsity. Each box summarizes 20 independent trials with different random seeds for a given dataset and sample-size setting. Across datasets, SGCP remains near the nominal target with competitive set sizes and consistently low subgroup disparity, especially in sparse regimes where competing methods become more conservative or more variable.

## E.2.2 Effect of the number of latent components

Figure A11 visualizes the learned component densities on SYN under different choices of K. When K is small, the model learns a few distinct score law patterns that separate low score and high score calibration regimes. As K increases, additional components tend to overlap with existing ones or become nearly flat, suggesting that the dominant calibration structure has already been captured by a smaller number of components. This behavior indicates that K controls the granularity of stochastic sharing rather than simply increasing model capacity. In practice, overly large K may introduce redundant components, while a moderate K provides a compact set of reusable score law building blocks for constructing sample dependent local score laws.

Figure A12 shows a similar sensitivity pattern on MIMIC-IV, but with a different score structure. Most learned components concentrate near small nonconformity scores, reflecting the strong mass of the empirical score distribution in the low score region. Increasing K refines this region and introduces broader components that capture the remaining tail behavior. However, when K becomes large, several components overlap substantially or become weakly differentiated. This suggests that additional components mainly refine existing calibration patterns rather than uncover entirely new score regimes. Together, the SYN and MIMIC-IV results show that a moderate number of latent components is sufficient to capture the dominant score law structure, while larger K may lead to redundant calibration sharing.

Table A10: Performance comparison w.r.t. sample sparsity on BACH (target $1 - \alpha = 0 . 9 0 )$ . Each method is run 20 times. Takeaway: Under extreme sparsity, most methods become conservative, but with the increase in sample size, SGCP recovers the fastest, with near-nominal coverage, a smaller prediction set, and lower subpopulation disparity, outperforming other baseline methods. The performance of each individual subgroup is further shown in Fig. 4.
<table><tr><td rowspan="2">Method</td><td colspan="3">N = 50</td><td colspan="3"> $\mathbf { N } = \mathbf { 1 5 0 }$ </td><td colspan="3"> $\mathbf { N } = \mathbf { 2 0 0 }$ </td><td colspan="3"> $\mathbf { N } = \mathbf { 3 0 0 }$ </td></tr><tr><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓</td><td></td><td>Cov. ↑</td><td>AvgSize ↓ CovGap ↓ Cov. ↑</td><td></td><td></td><td></td><td></td><td></td><td>AvgSize ↓ CovGap ↓ Cov. ↑ AvgSize ↓ CovGap ↓</td><td></td></tr><tr><td>Marginal</td><td> $8 9 . 3 4 _ { \pm 3 . 3 5 }$ </td><td> $3 . 7 0 0 _ { \pm 0 . 4 7 4 }$ </td><td> $1 9 . 0 _ { \pm 2 . 1 }$ </td><td> $9 0 . 5 0 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $3 . 6 0 4 _ { \pm 0 . 4 9 4 }$ </td><td> $1 7 . 6 _ { \pm 2 . 0 }$ </td><td> $9 0 . 6 9 _ { \pm 1 5 . 7 5 } 3 . 5 4 5 _ { \pm 0 . 4 4 5 }$ </td><td></td><td> $1 5 . 9 _ { \pm 2 . 9 }$ </td><td>89.20±9.64</td><td> $2 . 8 9 4 _ { \pm 0 . 2 6 7 }$ </td><td> $1 5 . 7 _ { \pm 1 . 7 }$ </td></tr><tr><td>Partial</td><td> $9 8 . 3 3 _ { \pm 2 . 6 6 }$ </td><td> $3 . 7 3 5 _ { \pm 0 . 3 1 6 }$ </td><td> $7 . 8 _ { \pm 1 . 0 }$ </td><td> $9 7 . 4 5 _ { \pm 3 . 2 5 }$ </td><td> $3 . 7 1 1 { \scriptstyle \pm 0 . 2 9 0 }$ </td><td> ${ \bf 6 . 2 _ { \pm 0 . 6 } }$ </td><td> $9 7 . 1 7 _ { \pm 1 1 . 5 6 } 3 . 4 6 0 _ { \pm 0 . 2 5 8 }$ </td><td></td><td> $7 . 5 _ { \pm 0 . 7 }$ </td><td> $9 6 . 5 7 _ { \pm 8 . 0 8 }$ </td><td> $3 . 3 8 5 _ { \pm 0 . 2 9 9 }$ </td><td> $7 . 6 _ { \pm 1 . 6 }$ </td></tr><tr><td>Exhaustive</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 . 0 0 0 { \scriptstyle \pm 0 . 1 0 5 }$ </td><td> $1 1 . 7 _ { \pm 0 . 7 }$ </td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 . 0 0 0 { \scriptstyle \pm 0 . 1 1 5 }$ </td><td> $9 . 2 _ { \pm 0 . 5 }$ </td><td> $9 8 . 7 3 _ { \pm 0 . 6 3 } 3 . 8 3 7 _ { \pm 0 . 2 3 3 }$ </td><td></td><td> $1 1 . 0 _ { \pm 0 . 1 }$ </td><td> $9 7 . 4 1 _ { \pm 6 . 8 9 }$ </td><td> $3 . 8 6 7 _ { \pm 0 . 1 1 8 }$ </td><td> $1 1 . 6 _ { \pm 1 . 5 }$ </td></tr><tr><td>RLCP</td><td> $9 1 . 7 4 { \scriptstyle \pm 3 . 0 6 }$ </td><td> $2 . 8 5 5 _ { \pm 0 . 4 4 0 }$ </td><td> $8 . 2 { \scriptstyle \pm 3 . 6 }$ </td><td> $9 2 . 2 5 { \scriptstyle \pm 3 . 8 0 }$ </td><td> $2 . 7 6 3 _ { \pm 0 . 4 2 2 }$ </td><td> $7 . 9 { \scriptstyle \pm 3 . 8 }$ </td><td> $9 1 . 9 9 _ { \pm 1 0 . 8 8 } 2 . 6 1 1 _ { \pm 0 . 4 1 1 }$ </td><td></td><td> $6 . 4 { \scriptstyle \pm 3 . 8 }$ </td><td> $9 1 . 8 8 { \scriptstyle \pm 1 0 . 5 6 }$ </td><td> $2 . 3 1 1 { \scriptstyle \pm 0 . 3 1 9 }$ </td><td> $8 . 1 { \scriptstyle \pm 2 . 5 }$ </td></tr><tr><td>CluCP</td><td>97.80±3.14 3.936±0.199</td><td></td><td> $8 . 7 { \pm } 1 . 5$ </td><td> $9 6 . 6 0 { \scriptstyle \pm 3 . 2 9 }$ </td><td> $3 . 8 8 2 _ { \pm 0 . 1 8 0 }$ </td><td> $8 . 1 { \pm } 1 . 9$ </td><td> $9 8 . 3 4 _ { \pm 0 . 5 0 } 3 . 9 8 5 _ { \pm 0 . 1 1 0 }$ </td><td></td><td> $1 0 . 0 { \scriptstyle \pm 2 . 6 }$ </td><td> $9 5 . 9 1 _ { \pm 3 . 5 9 }$ </td><td> $3 . 2 5 7 { \scriptstyle \pm 0 . 2 2 0 }$ </td><td> $1 0 . 7 \pm 2 . 6$ </td></tr><tr><td>AFCP</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 5 0 }$ </td><td>4.000±0.120</td><td> $1 0 . 0 { \scriptstyle \pm 1 . 2 }$ </td><td>96.50±2.64</td><td> $3 . 8 1 7 _ { \pm 0 . 1 1 6 }$ </td><td> $8 . 3 _ { \pm 1 . 3 }$ </td><td> $9 5 . 8 0 _ { \pm 2 . 3 9 } 3 . 4 4 7 _ { \pm 0 . 2 9 7 }$ </td><td></td><td> $7 . 5 _ { \pm 1 . 2 }$ </td><td> $9 5 . 0 0 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $2 . 8 3 0 _ { \pm 0 . 2 3 7 }$ </td><td> $6 . 3 _ { \pm 1 . 7 }$ </td></tr><tr><td>FaReG</td><td> $9 9 . 0 0 _ { \pm 2 . 0 0 }$ </td><td>3.946±0.153</td><td> $9 . 3 _ { \pm 1 . 3 }$ </td><td> $9 8 . 4 0 _ { \pm 1 . 7 8 }$ </td><td> $3 . 9 1 4 _ { \pm 0 . 2 1 2 }$ </td><td> $8 . 4 _ { \pm 1 . 8 }$ </td><td></td><td> $9 6 . 8 0 _ { \pm 1 . 9 3 } \ 3 . 7 6 9 _ { \pm 0 . 2 5 0 }$ </td><td> $7 . 7 _ { \pm 1 . 5 }$ </td><td> $9 3 . 6 0 _ { \pm 3 . 4 0 }$ </td><td> $2 . 9 5 9 _ { \pm 0 . 2 3 9 }$ </td><td> $7 . 4 _ { \pm 1 . 5 }$ </td></tr><tr><td>SGCP</td><td>96.44±2.61 3.842±0.136 6.8±2.2</td><td></td><td></td><td></td><td>92.63±4.04 3.344±0.240 5.0±1.58 92.22±2.39 2.370±0.197 4.6±2.1 92.00±2.70 2.224±0.215</td><td></td><td></td><td></td><td></td><td></td><td></td><td> ${ \bf 6 . 2 \pm 1 . 1 }$ </td></tr></table>

## E.2.3 Hyperparameter sensitivity

Figures A13 and A14 examine the sensitivity of SGCP w.r.t. several key hyperparameters, including $\lambda , \beta ,$ the number of latent calibration components K, and the number of evaluation latent samples $M _ { \mathrm { e v a l } }$ , under different $D _ { 1 }$ split ratios. Across both datasets, the hyperparameter curves are generally flat, indicating that the performance of SGCP does not rely on delicate tuning.

On the SYN, this stability is particularly clear: coverage remains close to the nominal target across all tested settings, while the changes in average prediction set size and subgroup coverage gap are modest. The most visible effect comes from the $D _ { 1 }$ split ratio, where allocating more data to score law learning consistently improves subgroup reliability and reduces prediction set size. On MIMIC-IV, it show a similar pattern. Coverage remains tightly concentrated around the nominal level across all tested settings, with modest changes in AvgSize and CovGap. Compared with hyperparameter choices, allocation training data to local score law learning has a more visible effect on the final tradeoff. Overall, these results suggest that the gains of SGCP stem primarily from its structured local score law design rather than from aggressive hyperparameter tuning.

## E.3 Wall-Clock Time

We compare the wall-clock time of the full SGCP pipeline with all baselines under varying calibration sample sizes. Figure A15 reports the average runtime over repeated runs, with error bars indicating one standard deviation. Marginal CP and Partial CP are the fastest methods, as they require only global calibration. RLCP and SGCP incur additional cost from sample-dependent calibration, but remain moderate and scale smoothly with sample size. AFCP is the most expensive baseline by a large margin, reflecting its repeated adaptive subgroup selection and conformal recalibration, while CluCP and FaReG lie in between. Overall, SGCP provides a favorable computational tradeoff, remaining practical across all tested sample sizes while achieving stronger subgroup reliability than simpler global or rigid group based baselines.

## E.4 Limitations

Our study has several limitations. All experiments are conducted on retrospective benchmarks, so prospective validation is needed to understand the practical impact of subgroup reliable prediction sets in real clinical workflows. In addition, our current framework is developed for multiclass classification and does not address regression or other prediction settings. Extending the method beyond classification and evaluating it under deployment level constraints are left to future work.

![](images/8652b74843a105b233e70a31e4f720f96cd1c56bcee1512cfd88ba9d420989d5.jpg)  
Figure A9: Distribution of overall coverage over repeated runs under varying sample sparsity. The dashed line marks the nominal target coverage of 0.9. Takeaway: SGCP remains consistently close to the target across datasets and sample sizes, whereas global calibration may under-cover in sparse regimes and stronger equalization baselines often become overly conservative.

## F Supplement to Related Work

Group-conditional and fairness-aware conformal prediction. A substantial body of work studies fairness-aware conformal prediction through coverage parity across groups that are either specified in advance or adaptively selected from data. Vadlamani et.al [23] develop a unified post-hoc framework that extends classical fairness notions to prediction sets by adjusting conformity thresholds in a fairness-aware manner. AFCP [29] study conformal classification with equalized coverage for adaptively selected groups, selecting features that reflect potential model limitations or biases in order to balance predictive efficiency and group reliability. FAREG [27] goes beyond predefined sensitive groups by learning representation-based subgroups and adaptively promoting equalized coverage for populations that would otherwise receive unfair treatment. SAGCCI [11] examines the tradeoff between fairness and efficiency by clustering protected groups with similar conformal score distributions and by using surrogate outcomes to improve threshold estimation. This is particularly close in spirit to our work in its use of score-distribution similarity to share calibration information. The key distinction lies in the granularity of the sharing mechanism. SAGCCI shares calibration information across clusters of observed protected groups, whereas SGCP learns a sample-dependent stochastic mixture over latent calibration components and thereby induces a local score law for each individual sample. Consequently, the calibration-sharing structure learned by SGCP need not align with protected-group labels and does not rely on surrogate outcomes.

![](images/aafbaf34cd74cf516f3807f34528860dc29a9b4b2c5bdffb079260ea3dcfc54c.jpg)

Figure A10: Distribution of average prediction set size over repeated runs under varying sample sparsity. Smaller values indicate better efficiency. Takeaway: SGCP remains among the most efficient methods across datasets and sample sizes, often matching or improving upon the smallest prediction sets while avoiding the severe conservatism of explicit equalization and localized baselines.  
![](images/f245f9e770830ecde0a1615caa1be2e5aa9425180359bdc6e55c8e9d6e5fa913.jpg)  
(a) K=3

![](images/2f648573ba03c75adfb25ac004ac7293773e84e32776156622df098a237fadd3.jpg)  
(b) K=6

![](images/5cc92846dc0e06241eabf0b58d9b529bac93cc5615e2346f65b412e7a129157e.jpg)  
(c) K=9

![](images/13e25913c952a7a142ce6f7129b0ec8066c08cc407a00d79ecf4e6ae88741bd6.jpg)  
(d) K=12  
Figure A11: Effect of K on learned latent component densities for SYN. Increasing K refines the score law components, but large K introduces overlapping or redundant densities.

Structured sharing and learned calibration groups. A related line of work improves upon a single global calibration rule by sharing calibration evidence across structured partitions. Clustered conformal prediction [8] groups labels with similar score distributions in many-class settings and performs calibration at the cluster level, thereby interpolating between standard conformal prediction and fully class-conditional conformal prediction. Conceptually similar ideas also arise beyond conformal prediction. For instance, VIR [26] smooths probabilistic uncertainty estimates across neighboring label regions in imbalanced regression. Together, these works underscore the value of structured sharing: calibration and uncertainty estimation can often benefit from borrowing information across statistically related regions, rather than treating all samples globally or calibrating each group in isolation. Another line of work enlarges or learns the group structure used for conditional calibration. Kandinsky Conformal Prediction [3] generalizes group-conditional conformal inference to overlapping and fractional group functions defined on (X, Y ), thereby expanding the collection of groups over which weighted coverage deviations can be controlled. Recent work on identifying homogeneous and interpretable groups for conformal prediction likewise uses nonconformity-score behavior to learn explicit partitions of the input space before applying group-conditional calibration [18]. These methods are closely related to ours in their shared emphasis on calibration-relevant structure beyond predefined sensitive attributes.

![](images/def2cd365540a21266224aeaa4a6fb00d64b8162d6be63decf97f21be7e6421c.jpg)  
(a) K=3

![](images/516c68b4b390603488d4772ef336a434578f420e8b47b80c66a8bcb0d354a326.jpg)  
(b) K=6

![](images/c9678a1808073214289133715d8d238c65f7b4587a8cacdd9d2cc3d825b4d3f2.jpg)  
(c) K=9

![](images/0d439120b83cbcff7e069fb3d682af1ba4ddece190fd9fc7ed9c8c9edd8f8757.jpg)  
(d) K=12

Figure A12: Effect of K on learned latent component densities for MIMIC-IV. Larger K refines the dominant low score region and tail behavior, but also introduces increasingly overlapping components.  
![](images/674ddfa9e8343f2c2f70d9ea1f2b3fc3a74120684322f894da2b33907a63cedb.jpg)  
Figure A13: Hyperparameter sensitivity of SGCP on SYN. We vary $\lambda _ { \mathrm { n l l } } , \lambda _ { \mathrm { b a l } } ,$ , K, and $M _ { \mathrm { e v a l } }$ , and report coverage, average set size, and subgroup coverage gap under different $D _ { 1 }$ split ratios. Shaded bands show one standard deviation. Takeaway: SGCP is largely insensitive to these hyperparameters, while larger samples for score-law learning reduce covgap.

Our work adopts a complementary perspective. Instead of specifying a family of group functions or assigning each sample to a hard discovered partition, SGCP learns a sample-dependent local score law through stochastic mixing over shared latent calibration components. This enables calibration information to be shared according to learned score behavior, while the final prediction set is stil obtained through standard split conformal calibration applied to transformed scores.

![](images/18572481efa635b5b7ce1e68d95bccbb75c8e24001a604d18dd038985fb197cf.jpg)  
Figure A14: Hyperparameter sensitivity of SGCP on MIMIC-IV (ICU delay). Takeaway: SGCP is largely insensitive to these hyperparameters. Coverage remains close to the nominal level, and most of the variation in efficiency and subgroup reliability is driven by the $D _ { 1 }$ split ratio rather.

![](images/4103ef7261964c815bb054c86ba7652bba5e7130e7fb64e8c5dc5bfda604becb.jpg)  
Figure A15: Wall clock time comparison w.r. calibration sample. We report the wall-clock time for each method, averaged over repeated runs; error bars indicate one standard deviation. The horizontal axis is shown on a log scale. Takeaway: SGCP has moderate computational cost and is faster than AFCP, while remaining competitive with other subgroup aware baselines across all sample sizes.