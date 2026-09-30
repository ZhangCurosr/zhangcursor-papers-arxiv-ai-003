# SCALING INFLUENCE FUNCTIONS IN LLMS THROUGH EIGENBASIS-CORRECTED ONE-BIT GRADIENT PROJECTION

Jaeseung Heo   
POSTECH   
jsheo12304@postech.ac.kr   
J Rosser   
University of Oxford   
jrosser@robots.ox.ac.uk   
Dongwoo Kim   
POSTECH   
dongwookim@postech.ac.kr

## ABSTRACT

Influence functions estimate how individual training examples affect the behavior of large language models (LLMs). Analyzing how training data influence different behaviors of an LLM involves repeated influence computation. Reusing stored training gradients reduces the computational cost, but storing full gradients is prohibitively expensive at LLM scale. We study how to compress these gradients while preserving influence estimates for future queries that are unknown at storage time. Through a worst-case analysis, we characterize the optimal fixeddimensional linear representation and propose eigenbasis-corrected one-bit gradient projection (EOGP) to approximate it at scale. Specifically, EOGP uses EK-FAC to reduce gradient dimensionality, then applies PCA within the retained subspace to learn compression directions from the training gradients. We then apply one-bit quantization to the resulting coordinates, allowing more coordinates to be retained within a fixed storage budget. On GPT-2, EOGP predicts retraining outcomes more accurately than the evaluated compression baselines while using one-sixteenth of their per-example storage. On OLMo 2 SFT models from 1B to 32B parameters, EOGP remains competitive with the baselines allocated over 100 times as much storage per example.

## 1 INTRODUCTION

Influence functions are widely used to analyze how training data shape the diverse behaviors of large language models (LLMs). Their applications include studying generalization (Grosse et al., 2023; Ruis et al., 2025; Kou et al., 2025), selecting data for fine-tuning (Zhou et al., 2024; Heo et al., 2026; Chen et al., 2026), and identifying training examples that contribute to undesired behavior (Zhang et al., 2025; Coalson et al., 2025). By estimating how individual training examples affect a model’s predictions (Koh & Liang, 2017), influence functions provide a common framework for investigating these connections between training data and model behavior.

Each attribution query asks which training examples contribute to a particular model output or behavior. Analyzing different behaviors of a fixed model therefore involves multiple queries, but the training-example gradients used in influence estimation can be reused across them. Computing and storing these gradients once would avoid repeated backward passes over the training data as new queries arise. At LLM scale, however, storing full gradients is costly: a single training-example gradient for an 8B-parameter model requires 16 GB in half precision. Supporting repeated attribution queries efficiently thus calls for compact gradient representations that can be stored and reused.

Existing gradient-compression methods reduce this storage cost through random (Schioppa, 2024; Choe et al., 2025; Hu et al., 2025) or curvature-informed projections (Schioppa et al., 2022; Choe et al., 2025), representing each gradient with a small number of coordinates. The central challenge is deciding which information these coordinates should preserve. Since future attribution queries are not known when the representations are constructed, compression must retain information that remains useful across different subsequent influence calculations.

We address this challenge by defining a compression objective based on the discrepancy between influence scores computed from compressed and uncompressed gradients across possible future queries. Under a normalized worst-case influence-error criterion, we show that the top-k PCA coordinates of half-whitened training gradients are optimal among k-dimensional linear representations. This result provides a concrete target for constructing reusable gradient representations.

To approximate this target at LLM scale, we propose Eigenbasis-corrected One-bit Gradient Projection (EOGP). Computing PCA directly on full gradients is prohibitively expensive for billionparameter models. We therefore first reduce their dimensionality using projection directions derived from EK-FAC (George et al., 2018), then apply PCA to the resulting representations to learn the final compression basis. This allows us to refine the representation using the training gradients within a computationally manageable space. We then apply one-bit quantization to the resulting coordinates, allowing more coordinates to be stored within a fixed budget. Together, these steps address which directions to retain and how much precision to allocate to each coordinate.

We evaluate the resulting representations through retraining-based validation and comparison with an uncompressed influence reference. On GPT-2 (Radford et al., 2019), EOGP outperforms the tested compression baselines in both linear datamodeling score (LDS) (Park et al., 2023) and counterfactual retraining (Bae et al., 2024). On OLMo 2 SFT models (OLMo et al., 2024) from 1B to 32B parameters, EOGP more accurately reproduces the reference influence rankings at matched storage budgets. Even with less than 16 KB per example, it remains competitive with all tested compression baselines allocated over 100 times as much storage. These results show that compact, reusable gradient representations can retain useful attribution information at substantially lower storage cost.

## 2 PRELIMINARIES

## 2.1 INFLUENCE FUNCTIONS

Influence functions (Koh & Liang, 2017) estimate how a model’s behavior would change if a training example were removed and the model retrained. Let $\theta \in \mathbb { R } ^ { d }$ denote the model parameters, $\theta ^ { * } \in \mathbb { R } ^ { \bar { d } }$ their trained values, and $\ell _ { i } : \mathbb { R } ^ { d } \to$ R the loss function for training example i. The corresponding training gradient is $\nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } ^ { d }$ , and $H \in \mathbb { R } ^ { d \times d }$ denotes the Hessian of the average training loss at $\theta ^ { * }$

For a scalar target measurement $f : \mathbb { R } ^ { d } $ R, we define the query gradient $\nabla _ { \theta } f \in \mathbb { R } ^ { d }$ as its gradient with respect to θ, evaluated at $\theta ^ { * }$ . For example, a query about the loss on a target document z uses $\nabla _ { \theta } f = \mathbf { \bar { V } } _ { \theta } \ell _ { z } ( \theta ^ { * } )$ . The influence score of training example i for this query is

$$
\begin{array} { r } { \mathcal { T } ( i ) : = \nabla _ { \theta } f ^ { \top } H ^ { - 1 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } . } \end{array}\tag{1}
$$

For neural networks, the loss Hessian H can be singular, so $H ^ { - 1 }$ may not exist. Prior work addresses this by using the generalized Gauss–Newton (GGN) matrix $G \in \mathring { \mathbb { R } ^ { d \times d } }$ , with $G \succeq 0$ , together with damping (Martens, 2020; Bae et al., 2022; Grosse et al., 2023). For $\lambda > 0$ , we define $H _ { \lambda } \ : =$ $G + \bar { \lambda } I _ { d } \in \mathbb { R } ^ { d \times d }$ , where $I _ { d }$ is the $d \times d$ identity matrix. Since $H _ { \lambda } \succ 0$ , the practical influence score is

$$
\hat { \mathcal { T } } ( i ) : = \nabla _ { \theta } f ^ { \top } H _ { \lambda } ^ { - 1 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } .\tag{2}
$$

For a fixed model, the training gradient $\nabla _ { \boldsymbol { \theta } } \ell _ { i } ( { \boldsymbol { \theta } } ^ { * } )$ does not depend on the query and can therefore be reused across queries. In this work, we aim to store these gradients so that subsequent influence calculations do not require repeated backward passes over the training data.

## 2.2 KRONECKER-FACTORED CURVATURE APPROXIMATION

GGN–Fisher equivalence. Storing and factorizing the full GGN is impractical at LLM scale, so prior work has used structured approximations of the Fisher information matrix for scalable influence estimation (Grosse et al., 2023). For softmax cross-entropy on logits, the loss commonly used to train autoregressive LLMs, the GGN coincides with the Fisher $F$ (Martens, 2020). Writing

![](images/545a00115f07616afe611c1c19693b0a5a3753a3d1e16f70e1755cff150ad3b2.jpg)  
Figure 1: Overview of EOGP. (a) Training gradients are projected onto an EK-FAC subspace and half-whitened. (b) PCA corrects the compression basis within this subspace. (c) The resulting coordinates are stored as one sign bit per coordinate together with a shared scale. Query gradients undergo the same transformations without one-bit quantization, and influence scores are computed as scaled inner products with the stored bits.

$g ( x , y ) : = \nabla _ { \boldsymbol { \theta } } \ell ( x , y ; \boldsymbol { \theta } ^ { * } )$ , this matrix is

$$
G = F = \mathbb { E } _ { x \sim \mathcal { D } _ { X } } \mathbb { E } _ { \tilde { y } \sim p _ { \theta ^ { * } } ( \cdot \vert x ) } \big [ g ( x , \tilde { y } ) g ( x , \tilde { y } ) ^ { \top } \big ] ,\tag{3}
$$

where $\mathcal { D } _ { X }$ is the empirical distribution of training inputs. Thus, the curvature is a gradient second moment with labels drawn from the model’s predictive distribution.

K-FAC approximation. K-FAC (Martens & Grosse, 2015) exploits the layer structure of neural networks to approximate this Fisher matrix efficiently. In its block-diagonal form, it neglects crosslayer blocks and approximates each layer’s block by a Kronecker product. For a linear layer, the two Kronecker factors are the second-moment matrices of its input activations a and the loss gradients δ with respect to its outputs.

$$
G _ { \mathrm { l a y e r } } = \mathbb { E } \big [ ( a a ^ { \top } ) \otimes ( \delta \delta ^ { \top } ) \big ] \approx A \otimes S , \qquad A : = \mathbb { E } [ a a ^ { \top } ] , \quad S : = \mathbb { E } [ \delta \delta ^ { \top } ] .\tag{4}
$$

Let $Q _ { A }$ and $Q _ { S }$ be the orthonormal eigenbases of A and S, respectively. Then $Q _ { A } \otimes Q _ { S }$ is an eigenbasis of the K-FAC approximation $A \otimes S$ . This structure allows the basis to be obtained from two smaller eigendecompositions and applied through matrix multiplications without materializing the full basis matrix.

EK-FAC approximation. EK-FAC (George et al., 2018; Grosse et al., 2023) corrects the eigenvalues of the K-FAC approximation while retaining its block structure and layerwise eigenbases. It obtains the corrected values by expressing gradients in these bases and taking the mean square of each coordinate. With labels sampled as in Equation (3), the resulting approximation is

$$
G \approx Q \Lambda Q ^ { \top } ,\tag{5}
$$

where Q is block diagonal, with each block given by the corresponding layer’s K-FAC eigenbasis $Q _ { A } \otimes Q _ { S }$ , and Λ is the diagonal matrix of corrected eigenvalues.

## 3 METHOD

This section presents Eigenbasis-corrected One-bit Gradient Projection (EOGP), which stores compact representations of training gradients for reuse in influence estimation. We first formulate gradient compression as minimizing a normalized worst-case influence error and show that storing the top-k PCA coordinates of half-whitened training gradients is optimal among k-dimensional linear representations (§3.1). We then approximate this target using a projection onto the EK-FAC eigenbasis followed by PCA within the retained subspace (§3.2). Finally, we store the resulting training coordinates as one sign bit each, together with a scale per module (§3.3). Figure 1 illustrates the pipeline, and Appendix B gives the full procedure.

## 3.1 WHAT TO STORE: PCA OF HALF-WHITENED GRADIENTS

Compression objective. We first ask which k-dimensional linear representation best preserves the influence scores in Equation (2) when queries are unknown at storage time. We consider linear compression methods that store $V ^ { \top } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } ^ { k }$ per training example, where $V \in \mathbb { R } ^ { d \times k }$ defines the compression. A reconstruction matrix $\boldsymbol { \dot { M } } \in \mathbb { R } ^ { \boldsymbol { \hat { d } } \times \boldsymbol { k } }$ maps these coordinates back to parameter space, giving the influence estimate $\nabla _ { \theta } f ^ { \top } M V ^ { \top } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } )$ .

We evaluate compression by the discrepancy between the uncompressed influence score and the estimate obtained from the stored coordinates. Since this discrepancy scales linearly with the query gradient, we constrain query magnitude to obtain a finite worst-case error.

The influence score in Equation (2) weights the interaction between the query and training gradients by $H _ { \lambda } ^ { - 1 }$ , giving greater weight to components along directions of lower curvature. The standard Euclidean norm treats all directions equally and therefore does not capture this directional weighting. To account for how query components enter the influence computation, we use the $H _ { \lambda } ^ { - 1 }$ -weighted Euclidean norm and define its unit ball B as

$$
\| q \| _ { H _ { \lambda } ^ { - 1 } } : = \sqrt { q ^ { \top } H _ { \lambda } ^ { - 1 } q } , \qquad \mathcal { B } : = \left\{ q \in \mathbb { R } ^ { d } : \| q \| _ { H _ { \lambda } ^ { - 1 } } \leq 1 \right\} .\tag{6}
$$

Our objective is the worst-case squared influence error over query gradients in B, averaged over training examples:

$$
\mathcal { E } ( V , M ) : = \mathbb { E } _ { i } \Big [ \operatorname* { s u p } _ { \nabla _ { \theta } f \in \mathcal { B } } \big | \nabla _ { \theta } f ^ { \top } \big ( H _ { \lambda } ^ { - 1 } - M V ^ { \top } \big ) \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \big | ^ { 2 } \Big ] .\tag{7}
$$

This criterion allows us to optimize the stored representation without knowing which attribution queries will be evaluated later.

Optimal linear representation. To characterize the optimal representation, we define the halfwhitened training and query gradients $\tilde { t } _ { i } : = H _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } )$ and $\widetilde { q } : = H _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } f$ . Influence is then the inner product $\langle \tilde { q } , \tilde { t } _ { i } \rangle$

Proposition 1 (Optimality of half-whitened PCA). Let $1 \leq k \leq d ,$ and let the columns of $U \in \mathbb { R } ^ { d \times k }$ be orthonormal eigenvectors corresponding to the k largest eigenvalues of the uncentered second moment $\boldsymbol { \Sigma } : = \mathbb { E } _ { i } [ \tilde { t } _ { i } \tilde { t } _ { i } ^ { \top } ]$ . Then Equation (7) is minimized by $V = M = H _ { \lambda } ^ { - 1 / 2 } U$ , i.e., by storing $U ^ { \top } \tilde { t } _ { i }$ and estimating influence as its inner product with $U ^ { \top } \tilde { q } .$

Thus, storing the top-k PCA coordinates of half-whitened training gradients minimizes influenceestimation error under Equation (7). The proof is provided in Appendix A.1.

## 3.2 TWO-STAGE PROJECTION: PCA WITHIN AN EK-FAC SUBSPACE

Performing PCA directly on half-whitened training gradients is impractical at LLM scale, so we approximate it in two stages. The first stage uses the gradient statistics provided by EK-FAC to construct an m-dimensional candidate subspace. The second stage applies PCA to the training gradients projected into this subspace to select the final k directions, where $m > k$

We apply both stages independently to each linear layer (module). Throughout this subsection, we suppress module indices and use $d , m ,$ , and k for the corresponding per-module dimensions.

First stage: subspace selection with EK-FAC. Let $C : = \mathbb { E } _ { i } [ \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) ^ { \top } ] \ \in \ \mathbb { R } ^ { d \times d }$ denote the uncentered second moment of the training gradients, also known as the empirical Fisher (Martens, 2020). Applying the PCA criterion of Proposition 1 within each module requires the leading eigenvectors of

$$
\Sigma = H _ { \lambda } ^ { - 1 / 2 } C H _ { \lambda } ^ { - 1 / 2 } .\tag{8}
$$

To construct a tractable surrogate for $\Sigma ,$ we approximate both the half-whitening operator and the training-gradient second moment. We begin with the half-whitening operator, which depends on the curvature matrix G. For softmax cross-entropy, G coincides with the true Fisher and can therefore be approximated using EK-FAC, as reviewed in Section 2.2. Writing $G \approx Q \Lambda Q ^ { \top }$ , where $\Lambda =$ d $\mathrm { i a g } \bar { ( \lambda _ { 1 } , \ldots , \lambda _ { d } ) }$ contains the corrected eigenvalues, gives

$$
H _ { \lambda } ^ { - 1 / 2 } = ( G + \lambda I ) ^ { - 1 / 2 } \approx Q ( \Lambda + \lambda I ) ^ { - 1 / 2 } Q ^ { \top } = : \widehat { H } _ { \lambda } ^ { - 1 / 2 } .\tag{9}
$$

For candidate subspace selection, we additionally use $Q \Lambda Q ^ { \top }$ as a surrogate for $C .$ This is a separate approximation: although the Fisher and the empirical Fisher $C$ are both gradient second moments, the Fisher averages over the model’s predictive label distribution for each input, whereas $C$ uses the corresponding training label. Combining this surrogate with the approximate half-whitening in Equation (9) yields

$$
\begin{array} { l } { { \Sigma _ { \mathrm { p r o x y } } : = \widehat { H } _ { \lambda } ^ { - 1 / 2 } Q \Lambda Q ^ { \top } \widehat { H } _ { \lambda } ^ { - 1 / 2 } \nonumber } } \\ { { \qquad = Q \mathrm { d i a g } \displaystyle \left( \frac { \lambda _ { 1 } } { \lambda _ { 1 } + \lambda } , \ldots , \frac { \lambda _ { d } } { \lambda _ { d } + \lambda } \right) Q ^ { \top } } . } \end{array}\tag{10}
$$

Since $\lambda _ { j } / ( \lambda _ { j } + \lambda )$ increases with $\lambda _ { j }$ for $\lambda > 0 ,$ , the m EK-FAC eigenvectors with the largest corrected eigenvalues form a top-m PCA basis for $\Sigma _ { \mathrm { p r o x y } }$ . We can therefore construct this candidate subspace directly from the existing EK-FAC statistics, without separately estimating $C$ or performing a full-space eigendecomposition.

Let $Q _ { m } \in \mathbb { R } ^ { d \times m }$ contain these eigenvectors and $\Lambda _ { m } \in \mathbb { R } ^ { m \times m }$ be the diagonal matrix of their corrected eigenvalues. Projecting the EK-FAC half-whitened training gradients onto this basis gives the first-stage coordinates:

$$
\begin{array} { r l } & { \widetilde { t } _ { i } ^ { ( 1 ) } : = Q _ { m } ^ { \top } \widehat { H } _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) } \\ & { \qquad = ( \Lambda _ { m } + \lambda I ) ^ { - 1 / 2 } Q _ { m } ^ { \top } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } ^ { m } . } \end{array}\tag{11}
$$

The Kronecker factorization $Q = Q _ { A } \otimes Q _ { S }$ allows these coordinates to be computed through matrix multiplications with the two smaller factors, without forming a dense d × m projection matrix.

Second stage: subspace refinement with PCA. The EK-FAC basis is inherited from the K-FAC approximation ${ \cal A } \otimes { \cal S } ,$ which uses separate second moments of activations and backpropagated gradients. Although EK-FAC corrects the diagonal second moments in this basis, it keeps the basis fixed and leaves off-diagonal terms unmodeled. Consequently, the retained axes need not align with the principal directions of the half-whitened training gradients.

To account for the second-moment structure among the retained coordinates, the first stage keeps m $> k$ EK-FAC axes. We then find the optimal k-dimensional subspace within their span by applying PCA to the full second moment of the first-stage coordinates, $\Sigma ^ { ( 1 ) } : = \mathbb { E } _ { i } \left[ \tilde { t } _ { i } ^ { ( 1 ) } ( \tilde { t } _ { i } ^ { ( 1 ) } ) ^ { \top } \right]$ including its off-diagonal entries.

Let $\boldsymbol { P } \in \mathbb { R } ^ { m \times k }$ contain the orthonormal eigenvectors of $\Sigma ^ { ( 1 ) }$ associated with its k largest eigenval ues. Combining this PCA projection with the first-stage transformation gives the final coordinates:

$$
\begin{array} { r l } & { \widetilde { t } _ { i } ^ { ( 2 ) } : = P ^ { \top } \widetilde { t } _ { i } ^ { ( 1 ) } } \\ & { \qquad = ( Q _ { m } P ) ^ { \top } \widehat { H } _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) \in \mathbb { R } ^ { k } . } \end{array}\tag{12}
$$

The following proposition shows that this correction yields a reconstruction error less than or equal to that obtained using the leading k EK-FAC axes.

Proposition 2 (Eigenbasis correction). Let $\tilde { t } _ { i } ^ { \mathrm { E K } } : = \widehat { H } _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } )$ denote the EK-FAC half whitened training gradients, $\smash { U _ { \star } : = Q _ { m } P }$ the corrected basis, and $Q _ { k }$ the basis formed by the leading k EK-FAC axes retained in $Q _ { m } .$ . Projecting these gradients onto span $( U _ { \star } )$ has the following guarantees:

(i) The mean squared reconstruction error is less than or equal to that obtained by projecting onto the leading k EK-FAC axes. Formally,

$$
\mathbb { E } _ { i } \left[ \lVert \tilde { t } _ { i } ^ { \mathrm { E K } } - U _ { \star } U _ { \star } ^ { \top } \tilde { t } _ { i } ^ { \mathrm { E K } } \rVert _ { 2 } ^ { 2 } \right] \leq \mathbb { E } _ { i } \left[ \lVert \tilde { t } _ { i } ^ { \mathrm { E K } } - Q _ { k } Q _ { k } ^ { \top } \tilde { t } _ { i } ^ { \mathrm { E K } } \rVert _ { 2 } ^ { 2 } \right] .
$$

In fact, it is minimal among all k-dimensional subspaces $o f \mathrm { s p a n } ( Q _ { m } )$

(ii) $I f \mathrm { s p a n } ( Q _ { m } )$ contains a top-k principal subspace o $\mathcal { \tilde { \mathbf { \Gamma } } } _ { i } ^ { \mathrm { R K } }$ , the error equals the minimum attained byfull-space k-dimensional PCA within that module.

Thus, retaining $m > k$ candidate axes allows the final subspace to be optimized within a larger span while storing only k coordinates per training example. The proof is provided in Appendix $\mathrm { A } . 2$

A scalable alternative to PCA. PCA minimizes reconstruction error within the retained span for a fixed output dimension k. However, storing and applying its dense correction matrix $P \in \overline { { \mathbb { R } } } ^ { m \times k }$ becomes costly as k grows. To exploit larger per-example storage budgets without this overhead, we introduce EOGP-R, which replaces the PCA correction with an implicitly applied subsampled randomized Hadamard transform (SRHT) (Tropp, 2011), $\mathsf { S } \in \mathbb { R } ^ { k \times m }$

$$
\tilde { t } _ { i } ^ { ( 2 , \mathrm { R } ) } = { \sf S } \tilde { t } _ { i } ^ { ( 1 ) } .
$$

The construction and normalization of S are detailed in Appendix B.2.

When the output dimension k is sufficiently large, SRHT preserves norms within a fixed lowdimensional subspace up to bounded distortion with high probability (Tropp, 2011). For query and training representations in this subspace, this guarantee also controls inner-product distortion and the resulting influence-score error introduced by sketching. This variant offers a trade-off between PCA’s fixed-k reconstruction optimality and scalability to larger output dimensions without a dense correction matrix.

## 3.3 ONE-BIT QUANTIZATION

Even after projection, storing every coordinate in floating point limits the number of directions we can retain under a fixed per-example storage budget. We apply one-bit quantization to both EOGP and EOGP-R to accommodate more projected coordinates within this budget.

For each module $u ,$ let $\tilde { t } _ { i , u } ^ { ( 2 ) } \in \mathbb { R } ^ { k _ { u } }$ denote the projected training-gradient coordinates. We store a binary sign vector and one scale:

$$
b _ { i , u } : = \mathrm { s i g n } \left( \widetilde { t } _ { i , u } ^ { ( 2 ) } \right) \in \{ \pm 1 \} ^ { k _ { u } } , \qquad s _ { i , u } : = \frac { 1 } { k _ { u } } \| \widetilde { t } _ { i , u } ^ { ( 2 ) } \| _ { 1 } .\tag{13}
$$

The scale captures the average magnitude of the coordinates and is chosen to minimize $\lVert \tilde { t } _ { i , u } ^ { ( 2 ) } -$ s $\mathrm { : } b _ { i , u } \| _ { 2 } ^ { 2 }$ over s (Rastegari et al., 2016). Each module requires $k _ { u }$ sign bits and one half-precision scale per training example. The complete representation contains $\textstyle \sum _ { u } k _ { u }$ coordinates.

At query time, we apply the same projection to the query gradient and keep its coordinates $\tilde { q } _ { u } ^ { ( 2 ) }$ unquantized. We estimate influence by

$$
\widehat { \cal T } ( i ) : = \sum _ { u } s _ { i , u } \ : \langle \tilde { q } _ { u } ^ { ( 2 ) } , b _ { i , u } \rangle .\tag{14}
$$

This computes influence from the stored representations without recomputing training gradients.

## 4 EXPERIMENTS

We evaluate whether compact gradient representations preserve useful training-data attributions under limited storage budgets. We first compare against retraining outcomes on GPT-2 (§4.1), then measure fidelity to EK-FAC influence on billion-parameter LLMs (§4.2). We examine the contributions of the projection and quantization choices (§4.3), and report the storage and computation required to build the representations (§4.4).

## 4.1 VALIDATION AGAINST RETRAINING

Setup and metrics. We evaluate GPT-2 (Radford et al., 2019) on WikiText-2 (Merity et al., 2016) at matched per-example storage budgets of {12, 24, 96, 384, 1536} KB. An uncompressed halfprecision gradient requires 162 MB per example. Compression baselines include LoGra (Choe et al., 2025) with PCA and random initialization, GraSS (Hu et al., 2025), and LoRIF (Li et al., 2026). We use EK-FAC influence, K-FAC influence, and TracIn (Pruthi et al., 2020) as uncompressed references, and random removal as an additional baseline for the counterfactual experiment.

We use two retraining-based metrics. The linear datamodeling score (LDS) (Park et al., 2023) evaluates how well influence estimates predict retraining outcomes. It measures the rank correlation between the estimated influences of randomly sampled training subsets and the outcomes measured after retraining the model on each subset. Counterfactual retraining (Bae et al., 2024) removes the examples ranked most influential by each method and measures the resulting increase in validation perplexity after retraining. Higher LDS indicates closer agreement with retraining outcomes, while a larger perplexity increase indicates a greater impact of removing the selected examples. Further metric and protocol details are provided in Appendix D.1.

![](images/eddd8a858b04eb6a00d3ac47c56261f12676b0303a6319186729d280efd3c722.jpg)  
EOGP LoGra (PCA) LoGra (Random) GraSS LoRIF EK-FAC K-FAC TracIn Random  
Figure 2: Validation against retraining-based ground truth on GPT-2 across storage budgets: linear datamodeling score (left) and counterfactual retraining (right). Black lines denote uncompressed influence functions with different Hessian approximations and the random-removal baseline.

Results. Figure 2 shows that EOGP improves retraining-based attribution quality over the compression baselines, with the largest gains at small storage budgets. In LDS, EOGP at 96 KB exceeds all compression-based baselines at 1,536 KB, despite using 16 times less per-example storage. At 384 KB, it achieves LDS close to the uncompressed K-FAC reference while using over 400 times less per-example storage than the 162 MB uncompressed representation. The counterfactual experiment likewise shows that the examples selected by EOGP produce larger perplexity increases than those selected by the compression baselines.

## 4.2 FIDELITY TO EK-FAC AT LLM SCALE

Setup. We evaluate the SFT checkpoints of OLMo 2 (OLMo et al., 2024) with 1B, 7B, 13B, and 32B parameters. Training examples come from the Tulu 3 SFT mixture (¨ Lambert et al., 2024), and queries come from a held-out split. Uncompressed half-precision gradients occupy approximately 2–62 GB per example across these models. We compare compression methods at budgets ranging from a few kilobytes to megabytes per example, corresponding to compression ratios on the order of $1 0 ^ { 4 } – 1 0 ^ { 6 }$ relative to uncompressed half-precision gradients. Further evaluation details are given in Appendix D.2.

Metrics and reference. Since retrainingbased evaluation is computationally prohibitive at this scale, we use agreement with EK-FAC influence rankings as a proxy for attribution quality. We choose EK-FAC as the reference because it performs best in our GPT-2 retraining-based evaluation among the evaluated methods that scale to billion-parameter LLMs (Section 4.1), consistent with prior comparisons (Choe et al., 2025). For each query, we measure overall ranking agreement using Spearman correlation and agreement near the top of the ranking using NDCG@20 (Jarvelin¨ & Kekal¨ ainen¨ , 2002), then average both metrics over queries.

![](images/83af72c3ffd6564417ee7ddff90dd497f67f2ebde87756d13b36b3e0cdceadca.jpg)  
Figure 3: LDS versus EK-FAC fidelity on GPT-2, measured by NDCG@20 (left) and Spearman correlation (right). ρ is the Spearman correlation across configurations from Figure 2.

To examine whether these agreement metrics reflect retraining-based attribution quality, Figure 3 shows their relationship with LDS on GPT-2. Each point corresponds to an evaluated configuration from Figure 2. The x-axis shows agreement between the influence rankings produced by each configuration and the EK-FAC reference rankings, measured by NDCG@20 (left) or Spearman cor relation (right). The y-axis shows LDS, which measures how well each configuration’s influence estimates predict retraining outcomes. We report the Spearman correlations $( \rho )$ between the x-axis metric and LDS across configurations: 0.95 for NDCG@20 and 0.84 for overall ranking agreement. These positive correlations indicate that configurations whose rankings more closely match EK-FAC generally also better predict the retraining outcomes, providing empirical support for using EK-FAC ranking agreement as a proxy for attribution quality.

![](images/3c8a4e01d084da9f08d31d83d264c9a576f3a4f40c8b43d816ab38bab608cf22.jpg)  
Figure 4: NDCG@20 (top) and Spearman correlation (bottom) of gradient compression methods against EK-FAC influence functions on OLMo 2 SFT models from 1B to 32B parameters.

Results. Figure 4 shows substantial gains over the compression baselines across the evaluated model sizes, particularly at small storage budgets. Fidelity generally improves as more storage is allocated, and continues to increase at the largest tested budgets on the 32B model. The improvements appear in both the overall ranking and the top-ranked examples. These results are obtained at compression ratios of approximately $\mathrm { 1 \bar { 0 } ^ { 4 } { - } 1 0 ^ { 6 } }$ , supporting the use of compact representations even when storing uncompressed gradients would require tens of gigabytes per example. In Appendix F, we additionally evaluate LoGra and GraSS with the same EK-FAC half-whitening and show that EOGP consistently outperforms both variants.

## 4.3 ABLATION STUDIES

We examine the effects of PCA basis correction (§3.2) and one-bit quantization (§3.3) on OLMo 2 7B SFT. We compare methods at matched per-example storage budgets (Figure 5, left) and matched numbers of stored coordinates (Figure 5, right). These comparisons assess performance under a fixed storage constraint and representation quality at a fixed dimension, respectively.

Comparison at matched storage budgets. At matched storage budgets, EOGP consistently achieves higher NDCG@20 than direct selection of EK-FAC basis directions (Figure 5, left). Both variants use one-bit quantization, demonstrating the benefit of refining the compression basis through

![](images/245114e1106d36d3fcbe5e4370786b265360eebbc2868666fbc4b0a7423aeb48.jpg)

![](images/b76d0841c74976537dba494d6c9fc1890f9ff62d1f580c9ed5c7a59b1cb1f2a7.jpg)  
Figure 5: Component ablation of EOGP (left) and comparison of 16-bit and 1-bit storage at matched coordinate counts (right).

PCA of the first-stage coordinates. For EOGP, one-bit representations also outperform FP16 representations by allowing more coordinates to be retained within the same storage budget. We provide further evidence of EOGP’s robustness to one-bit quantization and discuss a possible explanation in Appendix G.

Comparison at matched coordinate counts. At matched coordinate counts, EOGP achieves higher NDCG@20 than every baseline stored at the same precision, under both FP16 and one-bit storage (Figure 5, right). This advantage at fixed dimension supports the quality of the two-stage projection. Within EOGP, one-bit and FP16 representations achieve similar NDCG@20, indicating that one-bit quantization substantially reduces storage with little loss in ranking agreement.

## 4.4 STORAGE AND COMPUTATIONAL COSTS

Per-example storage. With $k _ { u }$ coordinates in module u, storing the compressed representation of one training example across all modules requires $\textstyle \sum _ { u } ( \lceil k _ { u } / 8 \rceil + \bar { 2 } )$ bytes, including packed sign bits and a two-byte scale per module. At $k _ { u } = 2 , 0 4 8$ , the 32B model requires approximately 115.6 KB per training example, compared with 62.4 GB for an uncompressed half-precision gradient.

Shared storage overhead. The Kronecker eigenbases, half-whitening weights, and PCA correction matrices are shared across all training examples and queries and are accounted for separately from the per-example payloads. For the 32B model, each module’s PCA correction matrix requires approximately 1 GB when its first-stage and final output dimensions are $m _ { u } = 2 6 2 , 1 4 4$ and $k _ { u } = 2 , 0 4 8$ , respectively. These shared storage costs do not grow with the number of stored examples. Appendix E.1 provides formulas for per-example and shared storage.

Store-construction time. On a single NVIDIA B200, our implementation of EOGP achieves store-construction times comparable to LoGra. For the 32B model at approximately 2,048 output coordinates per module, the per-example time is 0.247 seconds for EOGP and 0.278 seconds for Lo-Gra. These costs include gradient computation, projection, and disk I/O, including correction-matrix loading and transfer for EOGP. Fitting and saving the PCA correction matrices from cached firststage coordinates takes an additional 18.4 minutes as a one-time preprocessing cost. Implementation optimizations and detailed timing protocols are provided in Appendix E.3.

## 5 RELATED WORK

Scalable influence estimation. Influence estimation in large neural networks relies on efficient approximations to inverse-curvature products (Koh & Liang, 2017; Guo et al., 2021; Kwon et al., 2024). Arnoldi iteration approximates a dominant Hessian eigenspace to compute influence in a reduced space (Schioppa et al., 2022). K-FAC and EK-FAC use structured curvature approximations (Martens & Grosse, 2015; George et al., 2018), with EK-FAC enabling influence analysis at LLM scale (Grosse et al., 2023).

Gradient compression for data attribution. TRAK scales attribution through random projections (Park et al., 2023), while LESS and its quantized extension QLESS select instruction data using projected gradient similarities rather than the inverse-curvature-based influence scores studied here (Xia et al., 2024; Ananta et al., 2025). For influence estimation, TrackStar (Chang et al., 2025) uses curvature-corrected gradients for influence retrieval at LLM scale. LoGra uses Kroneckerfactored projections with random or PCA-based initialization (Choe et al., 2025). GraSS combines gradient sparsification with sparse random projection (Hu et al., 2025). LoRIF stores projected gradients as low-rank factors and approximates inverse curvature via truncated SVD (Li et al., 2026).

## 6 CONCLUSION

We present EOGP for building compact gradient stores for influence estimation across future queries. Under a normalized worst-case influence-error criterion, we show that PCA of halfwhitened training gradients is optimal among fixed-dimensional linear representations. EOGP approximates this target through PCA correction within a retained EK-FAC subspace and uses scaled one-bit quantization to retain more coordinates within a fixed storage budget. Our results show that

EOGP preserves useful attribution information with substantially less storage than existing compression methods, making reusable gradient stores more practical at LLM scale

## REFERENCES

Moses Ananta, Muhammad Farid Adilazuarda, Zayd Muhammad Kawakibi Zuhri, Ayu Purwarianti, and Alham Fikri Aji. QLESS: A quantized approach for data valuation and selection in large language model fine-tuning. arXiv preprint arXiv:2502.01703, 2025.

Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger B Grosse. If influence functions are the answer, then what is the question? Advances in Neural Information Processing Systems, 35:17953–17967, 2022.

Juhan Bae, Wu Lin, Jonathan Lorraine, and Roger Grosse. Training data attribution via approximate unrolling. Advances in Neural Information Processing Systems, 37:66647–66686, 2024.

Tyler Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon, and Ian Tenney. Scalable influence and fact tracing for large language model pretraining. In International Conference on Learning Representations, volume 2025, pp. 40976–40997, 2025.

Sirui Chen, Yunzhe Qi, Mengting Ai, Yifan Sun, Ruizhong Qiu, Jiaru Zou, and Jingrui He. Influence-preserving proxies for gradient-based data selection in LLM fine-tuning. In International Conference on Learning Representations, 2026.

Sang Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, Jeff Schneider, et al. What is your data worth to GPT? LLM-scale data valuation with influence functions. Advances in Neural Information Processing Systems, 38:145944–145985, 2025.

Zachary Coalson, Juhan Bae, Nicholas Carlini, and Sanghyun Hong. IF-Guide: Influence functionguided detoxification of LLMs. Advances in Neural Information Processing Systems, 38:78824– 78856, 2025.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936.

Thomas George, Cesar Laurent, Xavier Bouthillier, Nicolas Ballas, and Pascal Vincent. Fast ap- ´ proximate natural gradient descent in a Kronecker factored eigenbasis. Advances in Neural Information Processing Systems, 31, 2018.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023.

Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. FastIF: Scalable influence functions for efficient model interpretation and debugging. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10333–10350, 2021.

Jaeseung Heo, Kyeongheung Yun, Youngbin Choi, Sehyun Hwang, Jungseul Ok, and Dongwoo Kim. Interaction-aware influence functions for group attribution. arXiv preprint arXiv:2605.15675, 2026.

Pingbang Hu, Joseph Melkonian, Weijing Tang, Han Zhao, and Jiaqi Ma. GraSS: Scalable data attribution with gradient sparsification and sparse projection. Advances in Neural Information Processing Systems, 38:43880–43904, 2025.

Kalervo Jarvelin and Jaana Kek¨ al¨ ainen. Cumulated gain-based evaluation of IR techniques.¨ ACM Transactions on Information Systems (TOIS), 20(4):422–446, 2002.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning, pp. 1885–1894. PMLR, 2017.

Siqi Kou, Qingyuan Tian, Hanwen Xu, Zihao Zeng, and Zhijie Deng. Which data attributes stimulate math and code reasoning? An investigation via influence functions. Advances in Neural Information Processing Systems, 38:12302–12324, 2025.

Yongchan Kwon, Eric Wu, Kevin Wu, and James Y Zou. DataInf: Efficiently estimating data influence in LoRA-tuned LLMs and diffusion models. In International Conference on Learning Representations, volume 2024, pp. 21921–21942, 2024.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Shuangqi Li, Hieu Le, Jingyi Xu, and Mathieu Salzmann. LoRIF: Low-rank influence functions for scalable training data attribution. arXiv preprint arXiv:2601.21929, 2026.

James Martens. New insights and perspectives on the natural gradient method. Journal ofMachine Learning Research, 21(146):1–76, 2020.

James Martens and Roger Grosse. Optimizing neural networks with Kronecker-factored approximate curvature. In International Conference on Machine Learning, pp. 2408–2417. PMLR, 2015.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 OLMo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. TRAK: Attributing model behavior at scale. In International Conference on Machine Learning, 2023.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33: 19920–19930, 2020.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019.

Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. XNOR-Net: ImageNet classification using binary convolutional neural networks. In European Conference on Computer Vision, pp. 525–542. Springer, 2016.

Laura Ruis, Maximilian Mozes, Juhan Bae, Siddhartha Rao Kamalakara, Dwarak Talupuru, Acyr Locatelli, Robert Kirk, Tim Rocktaschel, Edward Grefenstette, and Max Bartolo. Procedural¨ knowledge in pretraining drives reasoning in large language models. In International Conference on Learning Representations, volume 2025, pp. 29367–29429, 2025.

Andrea Schioppa. Efficient sketches for training data attribution and studying the loss landscape. Advances in Neural Information Processing Systems, 37:37692–37735, 2024.

Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8179– 8186, 2022.

Joel A Tropp. Improved analysis of the subsampled randomized Hadamard transform. Advances in Adaptive Data Analysis, 3(01n02):115–126, 2011.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In International Conference on Machine Learning, 2024.

Han Zhang, Zhuo Zhang, Yi Zhang, Yuanzhao Zhai, Hanyang Peng, Yu Lei, Yue Yu, Hui Wang, Bin Liang, Lin Gui, et al. Correcting large language model behavior via influence function. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 14477–14485, 2025.

Xinyu Zhou, Simin Fan, and Martin Jaggi. HyperINF: Unleashing the HyperPower of the Schulz’s method for data influence estimation. arXiv preprint arXiv:2410.05090, 2024.

## A THEORETICAL DETAILS

## A.1 OPTIMALITY OF HALF-WHITENED PCA

We prove Proposition 1 for a fixed symmetric positive-definite matrix $H _ { \lambda }$ and half-whitened gradi ents $\tilde { t } _ { i } = H _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } )$ satisfying $\mathbb { E } _ { i } \| \tilde { t } _ { i } \| _ { 2 } ^ { 2 } < \infty$ . The supremum is over the full query-gradient ball in Equation (6).

Proof. Let $\Sigma = \mathbb { E } _ { i } [ \tilde { t } _ { i } \tilde { t } _ { i } ^ { \top } ]$ , with eigenvalues $\mu _ { 1 } \geq \cdot \cdot \cdot \geq \mu _ { d } \geq 0 .$ and let $1 \leq k \leq d .$ . We show that the minimum of Equation (7) is $\textstyle \sum _ { j = k + 1 } ^ { d } \mu _ { j }$ and that the proposed encoder–decoder pair attains it.

Reduction to low-rank reconstruction. For arbitrary $V , M \in \mathbb { R } ^ { d \times k }$ , define

$$
R : = H _ { \lambda } ^ { 1 / 2 } M V ^ { \top } H _ { \lambda } ^ { 1 / 2 } , \qquad \mathrm { r a n k } ( R ) \leq k .\tag{15}
$$

Write $\tilde { q } = H _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } f$ . Since $\nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) = H _ { \lambda } ^ { 1 / 2 } \tilde { t } _ { i }$ and $\boldsymbol { \nabla } _ { \theta } f = H _ { \lambda } ^ { 1 / 2 } \tilde { q }$ , the influence error satisfies

$$
\nabla _ { \boldsymbol { \theta } } \boldsymbol { f } ^ { \intercal } \bigl ( H _ { \lambda } ^ { - 1 } - M V ^ { \intercal } \bigr ) \nabla _ { \boldsymbol { \theta } } \ell _ { i } ( \boldsymbol { \theta } ^ { \ast } ) = \tilde { \boldsymbol { q } } ^ { \intercal } ( I - R ) \tilde { t } _ { i } .\tag{16}
$$

The change of variables between $\nabla _ { \boldsymbol { \theta } } f$ and $\tilde { q }$ is bijective and maps the normalized query-gradient ball to the Euclidean unit ball. Therefore,

$$
\mathcal { E } ( V , M ) = \mathbb { E } _ { i } \Big [ \operatorname* { s u p } _ { \| \tilde { q } \| _ { 2 } \leq 1 } \big | \tilde { q } ^ { \top } ( I - R ) \tilde { t } _ { i } \big | ^ { 2 } \Big ] = \mathbb { E } _ { i } \big \| ( I - R ) \tilde { t } _ { i } \big \| _ { 2 } ^ { 2 } .\tag{17}
$$

The last equality follows from Cauchy–Schwarz, with equality for a unit query in the direction of $( I - R ) \tilde { t }$ <sub>i</sub> whenever this residual is nonzero. If it is zero, both sides are zero. Conversely, every matrix R of rank at most k admits a factorization $R = \dot { X } Y ^ { \top }$ with $X , Y \in \mathbb { R } ^ { d \times k }$ , padding with zero columns when needed. Choosing $M = H _ { \lambda } ^ { - 1 / 2 } .$ X and $V = H _ { \lambda } ^ { - 1 / 2 } Y$ realizes this R in Equation (15). Thus optimizing over V, M is equivalent to optimizing Equation (17) over all rank-at-most-k matrices $\bar { R . }$

Orthogonal projection suffices. Let $S = { \mathrm { r a n g e } } ( R )$ , and let $\Pi _ { \cal S }$ be the orthogonal projector onto S. For each example, $( I - \Pi _ { S } ) \tilde { t } _ { \imath }$ is orthogonal to $s ,$ whereas $( \Pi _ { S } - R ) { \tilde { t } } _ { i }$ belongs to S. Consequently,

$$
\begin{array} { r l } & { \left\| { ( I - R ) \tilde { t } _ { i } } \right\| _ { 2 } ^ { 2 } = \left\| { ( I - \Pi _ { S } ) \tilde { t } _ { i } } \right\| _ { 2 } ^ { 2 } + \left\| { ( \Pi _ { S } - R ) \tilde { t } _ { i } } \right\| _ { 2 } ^ { 2 } } \\ & { ~ \geq \left\| { ( I - \Pi _ { S } ) \tilde { t } _ { i } } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{18}
$$

Taking expectations and using $\Pi _ { s } ^ { \top } = \Pi _ { s } = \Pi _ { s } ^ { 2 }$ gives

$$
{ \mathcal { E } } ( V , M ) \geq \operatorname { t r } \left( ( I - \Pi _ { S } ) \Sigma \right) = \operatorname { t r } ( \Sigma ) - \operatorname { t r } ( \Pi _ { S } \Sigma ) .\tag{19}
$$

The principal subspace attains the minimum. This PCA optimality step is a consequence of the classical Eckart–Young theorem (Eckart & Young, 1936). Write $\begin{array} { r } { \Sigma = \sum _ { j = 1 } ^ { d } \mu _ { j } u _ { j } u _ { j } ^ { \top } } \end{array}$ in an orthonormal eigenbasis. Set $\alpha _ { j } = u _ { j } ^ { \top } \Pi _ { S } u _ { j } = \| \Pi _ { S } u _ { j } \| _ { 2 } ^ { 2 }$ . Because $\Pi _ { \mathcal { S } }$ is an orthogonal projector of rank at most $k ,$ these weights satisfy $0 \leq \alpha _ { j } \leq 1$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { d } \alpha _ { j } = \mathrm { t r } ( \Pi _ { S } ) \le k } \end{array}$ . It follows that

$$
\mathrm { t r } ( \Pi _ { \mathcal { S } } \Sigma ) = \sum _ { j = 1 } ^ { d } \mu _ { j } \alpha _ { j } \le \sum _ { j = 1 } ^ { k } \mu _ { j } \alpha _ { j } + \mu _ { k } \Big ( k - \sum _ { j = 1 } ^ { k } \alpha _ { j } \Big ) \le \sum _ { j = 1 } ^ { k } \mu _ { j } .\tag{20}
$$

The first inequality uses $\mu _ { j } \leq \mu _ { k }$ for $j > k$ and the bound on the sum of the weights. The second uses $\mu _ { j } \geq \mu _ { k }$ for $j \le k$ and $\alpha _ { j } \leq 1$ . Combining Equations (19) and (20), every encoder–decoder pair satisfies

$$
\mathcal { E } ( V , M ) \geq \operatorname { t r } ( \Sigma ) - \sum _ { j = 1 } ^ { k } \mu _ { j } = \sum _ { j = k + 1 } ^ { d } \mu _ { j } .\tag{21}
$$

Now take $U = [ u _ { 1 } , \dots , u _ { k } ]$ and $V = M = H _ { \lambda } ^ { - 1 / 2 } U$ . Then $R = U U ^ { \top }$ , so

$$
\mathcal { E } \big ( H _ { \lambda } ^ { - 1 / 2 } U , H _ { \lambda } ^ { - 1 / 2 } U \big ) = \mathrm { t r } \big ( ( I - U U ^ { \top } ) \Sigma \big ) = \sum _ { j = k + 1 } ^ { d } \mu _ { j } .\tag{22}
$$

This pair therefore attains the global minimum. Its stored representation and influence estimate are precisely

$$
V ^ { \top } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) = U ^ { \top } \tilde { t } _ { i } ,\tag{23}
$$

$$
\nabla _ { \boldsymbol { \theta } } \boldsymbol { f } ^ { \top } \boldsymbol { M } \boldsymbol { V } ^ { \top } \nabla _ { \boldsymbol { \theta } } \ell _ { i } ( { \boldsymbol { \theta } } ^ { * } ) = \tilde { \boldsymbol { q } } ^ { \top } \boldsymbol { U } \boldsymbol { U } ^ { \top } \tilde { \boldsymbol { t } } _ { i } = \langle \boldsymbol { U } ^ { \top } \tilde { \boldsymbol { q } } , \boldsymbol { U } ^ { \top } \tilde { \boldsymbol { t } } _ { i } \rangle .\tag{24}
$$

This proves Proposition 1.

The result uses the uncentered second moment and remains valid when eigenvalues are repeated.

## A.2 PROOF OF EIGENBASIS CORRECTION

We prove Proposition 2 within one module. Let $x _ { i } = \widehat { H } _ { \lambda } ^ { - 1 / 2 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } )$ have finite second moment, let $Q _ { m } \in \mathbb { R } ^ { d \times m }$ have orthonormal columns, and set $a _ { i } = Q _ { m } ^ { \top } x _ { i }$ . Let $\boldsymbol { P } \in \mathbb { R } ^ { m \times k }$ contain orthonormal leading eigenvectors of $\mathbb { E } _ { i } [ a _ { i } a _ { i } ^ { \top } ]$ ], where $1 \leq k \leq m$

Proof. Every k-dimensional subspace of range $\left( Q _ { m } \right)$ has an orthonormal basis $U = Q _ { m } W$ for some $W \in \mathbb { R } ^ { m \times k }$ satisfying $W ^ { \top } W \stackrel { * } { = } I _ { k }$ . Decomposing $x _ { i } - U U ^ { \top } x _ { i }$ into components orthogonal to and within range $\left( Q _ { m } \right)$ yields

$$
\begin{array} { r } { \mathbb { E } _ { i } \| x _ { i } - U U ^ { \top } x _ { i } \| _ { 2 } ^ { 2 } = \mathbb { E } _ { i } \| ( I - Q _ { m } Q _ { m } ^ { \top } ) x _ { i } \| _ { 2 } ^ { 2 } } \\ { + \mathbb { E } _ { i } \| a _ { i } - W W ^ { \top } a _ { i } \| _ { 2 } ^ { 2 } . } \end{array}\tag{25}
$$

The first term is independent of $W$ . By the PCA argument in Appendix $\mathrm { A . 1 }$ , the second is minimized by $W = P$ . Thus $Q _ { m } P$ minimizes reconstruction error over all k-dimensional subspaces of $\mathrm { r a n g e } ( Q _ { m } )$ . The leading k retained EK-FAC axes are among these feasible choices, proving part (i).

For part (ii), suppose range $\left( Q _ { m } \right)$ ) contains a globally optimal k-dimensional principal subspace of $x _ { i }$ within the module. This subspace is feasible for the restricted problem, so the restricted minimum is at most the full-space minimum. It is also at least that minimum because the feasible set is restricted. The two errors therefore agree. □

## A.3 SCALED ONE-BIT QUANTIZATION

Optimal scale for a fixed sign vector. For projected coordinates $a \in \mathbb { R } ^ { k }$ with $k \geq 1$ , let $b =$ $\mathrm { s i g n } ( a ) \in \{ \pm 1 \} ^ { k } .$ , allowing either sign for zero coordinates. For a scale $s \in \mathbb { R }$ , the identities $b ^ { \top } a \stackrel { \cdot } { = } \| a \| _ { 1 }$ and $b ^ { \top } b = k$ give

$$
\| a - s b \| _ { 2 } ^ { 2 } = \| a \| _ { 2 } ^ { 2 } - 2 s \| a \| _ { 1 } + k s ^ { 2 } = k \left( s - { \frac { \| a \| _ { 1 } } { k } } \right) ^ { 2 } + \| a \| _ { 2 } ^ { 2 } - { \frac { \| a \| _ { 1 } ^ { 2 } } { k } } .\tag{26}
$$

Thus, $s = \| a \| _ { 1 } / k$ minimizes the squared reconstruction error for the sign vector $b ,$ including $s = 0$ when $a = 0$

## B ALGORITHMS AND PROJECTION IMPLEMENTATION

## B.1 PROJECTION PATHS AND UNIFIED PROCEDURE

Both stages operate independently on the linear-layer modules defined in Section 3.2. Across all evaluated models, EOGP applies PCA to the first-stage coordinates, while EOGP-R replaces PCA with an SRHT. The model-specific module lists, dimensions, and storage-budget allocations are recorded with the experimental settings.

For module $u ,$ the first-stage map and the subsequent projection are

$$
L _ { u } = ( \Lambda _ { u , m } + \lambda _ { u } I ) ^ { - 1 / 2 } Q _ { u , m } ^ { \top } , \qquad T _ { u } ( v ) = \left\{ \begin{array} { l l } { P _ { u } ^ { \top } v , } & { \mathrm { f o r ~ E O G P } , } \\ { S _ { u } v , } & { \mathrm { f o r ~ E O G P - R } . } \end{array} \right.\tag{27}
$$

Here, $P _ { u } \in \mathbb { R } ^ { m _ { u } \times k _ { u } }$ contains the PCA directions, and $\mathsf { S } _ { u } \in \mathbb { R } ^ { k _ { u } \times m _ { u } }$ is an implicitly applied SRHT. The Kronecker factors and retained-coordinate indices implement $L _ { u }$ without materializing $Q _ { u , m }$ or $L _ { u }$ . We write $g _ { i , u }$ and $q _ { u }$ for the training and query gradient blocks of module u.

```latex
Algorithm 1 EOGP and EOGP-R fitting, storage, and querying
Require: Factorized EK-FAC bases $Q _ { u } ,$ corrected eigenvalues $\Lambda _ { u }$ , damping values ${ \lambda } _ { u } \ > \ 0 ,$ , dimensions
$\begin{array} { r } { 1 \leq k _ { u } \leq m _ { u } \leq d _ { u } , } \end{array}$ , method variant, and a PCA fitting set C for EOGP
Fitting shared projections
1: for each module u do
2: Select $Q _ { u , m } , \Lambda _ { u , m }$ using the $m _ { u }$ largest corrected eigenvalues
3: For EOGP-R, initialize the implicit SRHT $\mathsf { S } _ { u }$
4: end for
5: if using EOGP then
6: for each $i \in \mathcal { C }$ do
7: Compute the training gradient $g _ { i }$ once
8: for each module u do
9: Compute and cache $z _ { i , u } \gets L _ { u } g _ { i , u }$
10: end for
11: end for
12: For each $u ,$ fit $P _ { u }$ by uncentered PCA of $\{ z _ { i , u } : i \in \mathcal { C } \}$
13: end if
Storing one training example i
14: Compute $g _ { i }$ once
15: for each module u do
16: $x _ { i , u } \gets T _ { u } ( L _ { u } g _ { i , u } )$
17: Store packed binary signs $b _ { i , u } \gets \mathrm { s i g n } _ { \pm } ( x _ { i , u } )$ and $s _ { i , u } \gets \mathrm { F P 1 6 } ( \| x _ { i , u } \| _ { 1 } / k _ { u } )$
18: end for
Scoring a query $f$
19: Compute $\overset { \vartriangle } { \boldsymbol { q } } = \dot { \nabla } _ { \boldsymbol { \theta } } \boldsymbol { \dot { f } } ( \boldsymbol { \theta } ^ { * } )$ once
20: For each $u ,$ compute the unquantized coordinates $y _ { u } \gets T _ { u } ( L _ { u } q _ { u } )$
21: return $\begin{array} { r } { \widehat { \mathcal { T } } ( i ) = \sum _ { u } s _ { i , u } \langle y _ { u } , b _ { i , u } \rangle } \end{array}$ for each stored example i
```

The batched implementation is described in Appendix E.3.

## B.2 PCA FITTING AND RANDOM PROJECTION

PCA fitting. For EOGP, we form the first-stage coordinates $z _ { i , u } = L _ { u } g _ { i , u }$ using training-data labels and apply PCA without subtracting the sample mean. Let $\boldsymbol { Z _ { u } } \in \mathbb { R } ^ { \boldsymbol { \bar { n } } \times m _ { \iota } }$ contain these coordinates as rows, where $n = | \mathcal { C } |$ . The columns of $P _ { u }$ are the top-k right singular vectors of $Z _ { u }$ equivalently the eigenvectors associated with the largest eigenvalues of $\bar { Z _ { u } ^ { \top } } Z _ { u } / \bar { n }$ . The nonzero principal directions can be recovered from the eigendecomposition of the $n \times n$ Gram matrix $\hat { Z _ { u } } \dot { Z _ { u } } ^ { \top }$ avoiding construction of an $m _ { u } \times m _ { u }$ second-moment matrix. This is the PCA correction analyzed in Proposition 2.

Random projection. For EOGP-R, a subsampled randomized Hadamard transform (SRHT) (Tropp, 2011) maps the first-stage coordinates directly to $k _ { u }$ dimensions,

$$
\mathsf { S } _ { u } = \sqrt { \frac { p _ { u } } { k _ { u } } } R _ { u } H _ { p _ { u } } D _ { u } J _ { u } , \qquad p _ { u } = 2 ^ { \lceil \log _ { 2 } m _ { u } \rceil } .\tag{28}
$$

Here, $J _ { u }$ zero-pads the $m _ { u }$ coordinates to width $p _ { u } , D _ { u }$ is a diagonal matrix of independent random signs, $H _ { p _ { u } }$ is the orthonormal Walsh–Hadamard matrix, and $R _ { u }$ selects $k _ { u }$ rows uniformly without replacement. The same realized transform is used for gradient storage and query projection. This variant does not fit a PCA correction.

## B.3 COMPUTATION AND COMPLEXITY

Query scoring. Using the unquantized query coordinates $y _ { u } = T _ { u } ( L _ { u } q _ { u } )$ from Algorithm 1 and the stored signs $b _ { i , u }$ and scales $s _ { i , u }$ , the influence score is

$$
\widehat { \mathcal { T } } ( i ) = \sum _ { u } s _ { i , u } \langle y _ { u } , b _ { i , u } \rangle .\tag{29}
$$

The same query coordinates are reused to score all stored examples.

Projection cost. Applying the PCA correction in EOGP costs $O ( m _ { u } k _ { u } )$ operations per module. The implicit SRHT in EOGP-R costs $O ( p _ { u } \log p _ { u } )$ operations per module, where $p _ { u }$ is the padded dimension in Equation (28).

Storage and scanning. Computing signs and scales from an already projected training representation costs $O ( \sum _ { u } k _ { u } )$ operations. After projecting a query, scoring all $N _ { \mathrm { s t o r e } }$ examples costs

$$
O \left( N _ { \mathrm { s t o r e } } \sum _ { u } k _ { u } \right)\tag{30}
$$

arithmetic operations. Storage in bytes and measured runtimes are reported in Appendices E.1 and E.3.

## C EXPERIMENTAL CONFIGURATION

This section describes the configurations used in Section 4.

## C.1 MODELS, DATA, AND GRADIENT COMPUTATION

We evaluate GPT-2 on WikiText-2 and released OLMo 2 SFT checkpoints on Tulu 3 data. The GPT-¨ 2 attribution checkpoint is fine-tuned using the protocol in Appendix D.1. Table 1 lists the source checkpoints and datasets.

Data preparation. We tokenize wikitext-2-raw-v1 from Salesforce/wikitext with the GPT-2 tokenizer and divide concatenated text into 512-token blocks. Training and validation blocks provide attribution candidates and queries, respectively. OLMo 2 1B and 32B use allenai/tulu-3-sft-olmo-2-mixture-0225, while 7B and 13B use allenai/ tulu-3-sft-olmo-2-mixture. Each conversation is one example, with queries selected separately from attribution candidates. Evaluation protocols are given in Appendix D.

Loss and query measurement. Both training and query gradients use token-summed next-token cross entropy. GPT-2 includes all prediction positions, while OLMo 2 includes only assistantresponse positions and their turn-ending tokens.

Attributed parameters. We attribute to the attention and MLP linear layers, including biases in GPT-2. Embeddings, the language-model output head, and normalization parameters are excluded. Attribution gradients are computed in evaluation mode.

Table 1: Source model checkpoints and datasets.
<table><tr><td>Model</td><td>Source checkpoint</td><td>Dataset</td></tr><tr><td>GPT-2</td><td>gpt2</td><td>WikiText-2</td></tr><tr><td>OLMo 2 1B</td><td>al1enai/OLMo-2-0425-1B-SFT</td><td>Tülu 3 mixture</td></tr><tr><td>OLMo 2 7B</td><td>al1enai/OLMo-2-1124-7B-SFT</td><td>Tülu 3 mixture</td></tr><tr><td>OLMo 2 13B</td><td>al1enai/OLMo-2-1124-13B-SFT</td><td>Tülu 3 mixture</td></tr><tr><td>OLMo 2 32B</td><td>allenai/OLMo-2-0325-32B-SFT</td><td>Tülu 3 mixture</td></tr></table>

## C.2 CURVATURE FITTING AND PROJECTION DIMENSIONS

Fitting configuration. We estimate EK-FAC statistics using Kronfluence. Fisher statistics use labels sampled from the model’s next-token distribution with input contexts fixed to the data. Factor statistics are averaged over non-padding tokens, and corrected eigenvalues over examples using sequence-level gradients.

For each module u with $d _ { u }$ attributed parameters, we use relative damping

$$
\lambda _ { u } = 0 . 1 \frac { \mathrm { t r } ( \Lambda _ { u } ) } { d _ { u } } ,\tag{31}
$$

where $\Lambda _ { u }$ is the diagonal matrix containing the module’s full set of corrected eigenvalues. The resulting half-whitening transformation is applied to both training and query gradients.

PCA fitting data. For OLMo 2, the PCA fitting set C contains 10,000 conversations sampled from the same SFT mixture with a fixed random seed, after excluding all candidate, query, and curvaturefitting examples. Each conversation is truncated to at most 2,048 tokens.

Projection dimensions. EOGP applies PCA to the first-stage coordinates on all models. For OLMo 2, $m _ { u } = 2 6 2 , 1 4 4$ , and the final dimension is uniform across modules, ranging in powers of two from 256 to 8,192. EOGP-R uses the same first-stage width and between 256 and 32,768 SRHT output coordinates.

Storage budgets. Budgets count per-example coordinates and scales, with projection dimensions chosen to match storage across methods. In the main comparisons, EOGP and EOGP-R use onebit coordinates with FP16 scales, while compressed baselines use FP16. Figure 5 compares both precisions; in its right panel, which fixes coordinate counts on OLMo 2 7B, the one-bit curves also form a matched-storage comparison, since equal coordinate counts imply equal one-bit storage, and EOGP outperforms all one-bit baselines. GPT-2 follows the nominal budget labels in Figure 2, and OLMo 2 reports payloads in decimal units. Shared artifacts are accounted for separately in Appendix E.1.

## C.3 BASELINE CONFIGURATIONS

All methods share attributed modules, evaluation examples, queries, and loss definitions. For OLMo 2, EOGP, LoGra, GraSS, and LoRIF fit their curvature estimates on the same 10,000 conversations. Storage formats and budget matching follow Appendix C.2.

Compression baselines. Our LoGra implementation is based on LogIX<sup>1</sup>, with random and PCA initialization on both GPT-2 and OLMo 2. LoGra uses the same relative damping coefficient (0.1) as in Appendix C.2. PCA initialization uses factor eigenvectors fitted with Kronfluence<sup>2</sup>, and the factor rank a determines $a ^ { 2 }$ stored coordinates per module. Our GraSS implementation follows the official code, using factor-wise random masking and CountSketch with output dimensions matched to LoGra. Our LoRIF reimplementation uses randomly projected gradient matrices and a rank-256 inverse-curvature approximation.

Uncompressed references. EK-FAC and K-FAC use Kronfluence with model-sampled labels and the damping in Appendix C.2. The TracIn reference is the unweighted gradient inner product at the final checkpoint.

Gaussian random projection. This control uses independent standard Gaussian factors to produce $a ^ { 2 }$ coordinates per module. Training gradients are projected directly, and query gradients are preconditioned by the full-space damped EK-FAC inverse before projection. Inner products are scaled by $a ^ { - 2 }$

Randomization. LoGra and LoRIF use one projection realization per configuration. GraSS uses one on GPT-2 and averages over three seeds on OLMo 2. The Gaussian control averages over three independent draws, and EOGP-R over up to three disjoint blocks of a shared SRHT output. Retraining and uncertainty estimation follow Appendix D.1.

## D EVALUATION PROTOCOLS

## D.1 GPT-2 RETRAINING EVALUATION

We evaluate attribution on WikiText-2 using the linear datamodeling score (LDS) and counterfactual retraining.

Linear datamodeling score. We reuse the publicly released subset masks and retraining losses from Kronfluence for $R = 1 0 0$ subsets, each retaining half of the training blocks. All methods and storage budgets use the same subsets and losses on the validation blocks. The query measurement is the token-summed next-token cross-entropy loss of a validation block.

Let $S _ { r }$ denote the retained training subset and $Y _ { r q }$ its released retraining loss for query q. Under our removal-based influence convention, the predicted subset score is

$$
A _ { r q } = - \sum _ { i \in S _ { r } } \widehat { \mathscr { T } } _ { q } ( i ) .\tag{32}
$$

For each query, we compute the Spearman correlation across subsets,

$$
\mathrm { L D S } _ { q } = \rho _ { \mathrm { S } } \left( ( A _ { r q } ) _ { r = 1 } ^ { R } , ( Y _ { r q } ) _ { r = 1 } ^ { R } \right) ,\tag{33}
$$

and report its mean over queries, excluding undefined correlations. We use average ranks for ties. We obtain 95% percentile confidence intervals from 1,000 bootstrap resamples of the subsets, recomputing the query-averaged LDS for each resample.

Counterfactual retraining. For each method and storage budget, we rank the training blocks by the sum of their signed influence scores over the first 50 validation blocks. We remove the examples with the largest scores, reusing the same ranking for removal counts from 50 to 300 in increments of 50.

Each retraining run starts from the pretrained GPT-2 weights and uses AdamW with a learning rate of $\mathrm { 3 \times 1 0 ^ { - 5 } }$ , weight decay of 0.01, three epochs, and a training batch size of eight. We use a set S of five seeds, matched across methods. Perplexity is evaluated on the same 50 validation blocks used to construct the ranking by exponentiating their mean token-level cross-entropy loss.

Let $\mathrm { P P L } _ { s } ( b )$ denote the perplexity for seed s after removing b examples. We report

$$
\Delta \mathrm { P P L } ( b ) = \frac { 1 } { 5 } \sum _ { s \in { \cal S } } \left[ \mathrm { P P L } _ { s } ( b ) - \mathrm { P P L } _ { s } ( 0 ) \right] ,\tag{34}
$$

where the zero-removal runs are shared across methods. Error bars show the population standard deviation of the post-removal perplexities across seeds. For the random-removal control, each seed defines a random permutation of the training set, and the first b examples are removed.

## D.2 FIDELITY TO EK-FAC INFLUENCE

We compare compressed influence scores with uncompressed EK-FAC scores on the same candidate training examples. OLMo 2 uses 1,000 candidates and 100 queries, while GPT-2 uses the training and validation blocks as candidates and queries, respectively. Module contributions are summed before evaluation, and both score vectors use signed scores under the same removal-based convention, without taking absolute values or clipping.

Spearman correlation. For query q, let ${ \bf e } _ { q }$ and $\mathbf { a } _ { q }$ denote the reference and compressed score vectors over the candidate examples. We compute

$$
\rho _ { q } = \mathrm { C o r r } \left( \mathrm { r a n k } ( \mathbf { a } _ { q } ) , \mathrm { r a n k } ( \mathbf { e } _ { q } ) \right) , \qquad \overline { { \rho } } _ { \mathrm { S } } = \frac { 1 } { \left| \mathcal { Q } _ { \mathrm { v a l i d } } \right| } \sum _ { q \in \mathcal { Q } _ { \mathrm { v a l i d } } } \rho _ { q } .\tag{35}
$$

Ties receive average ranks. A correlation is undefined when either score vector is constant, and $\mathcal { Q } _ { \mathrm { v a l i d } }$ contains the queries with defined correlations.

NDCG@20. We use binary relevance based on membership in the reference top 20. Let $T _ { q }$ contain the 20 examples with the largest signed EK-FAC scores, and let $\pi _ { q } ( t )$ denote the example at position t when compressed scores are sorted in descending order. Then

$$
\mathrm { N D C G @ 2 0 } ( q ) = \frac { \displaystyle \sum _ { t = 1 } ^ { 2 0 } \frac { \mathbf { 1 } \{ \pi _ { q } ( t ) \in T _ { q } \} } { \log _ { 2 } ( t + 1 ) } } { \displaystyle \sum _ { t = 1 } ^ { 2 0 } \frac { 1 } { \log _ { 2 } ( t + 1 ) } } .\tag{36}
$$

The denominator is the ideal discounted gain when all 20 relevant examples occupy the first 20 positions. We report the arithmetic mean of NDCG@20 over queries.

## D.3 RELATIONSHIP BETWEEN FIDELITY AND RETRAINING

Figure 3 compares EK-FAC fidelity with LDS across the 25 compressed configurations and three reference methods in Figure 2. For each configuration, all metrics use the same attribution score matrix. We average each metric over queries before computing correlations across configurations.

Let $\mathcal { I }$ denote these 28 configurations, and let $L _ { j } , ~ N _ { j }$ , and $S _ { j }$ denote their query-averaged LDS, NDCG@20, and Spearman fidelity. The resulting correlations are

$$
\begin{array} { r } { \rho _ { \mathrm { S } } ( ( L _ { j } ) _ { j \in \mathcal { I } } , ( N _ { j } ) _ { j \in \mathcal { I } } ) \approx 0 . 9 5 , } \\ { \rho _ { \mathrm { S } } ( ( L _ { j } ) _ { j \in \mathcal { I } } , ( S _ { j } ) _ { j \in \mathcal { I } } ) \approx 0 . 8 4 . } \end{array}\tag{37}
$$

## E STORAGE AND RUNTIME ACCOUNTING

## E.1 PER-EXAMPLE PAYLOAD AND SHARED ARTIFACTS

We account separately for per-example representations and artifacts shared across the gradient store. Let U denote the attributed modules, $m _ { u }$ and $k _ { u }$ the first-stage and final output dimensions of module u, respectively, and $N _ { \mathrm { s t o r e } }$ the number of stored examples. The attributed parameter set is specified in Appendix C.1.

Per-example payload. EOGP and EOGP-R store one sign bit per coordinate and one FP16 scale per module. With byte packing performed separately for each module, the payload is

$$
B _ { \mathrm { 1 b i t } } = \sum _ { u \in \mathcal { U } } \left( \left\lceil \frac { k _ { u } } { 8 } \right\rceil + 2 \right) \quad \mathrm { b y t e s . }\tag{38}
$$

For a uniform dimension k, this becomes $| \mathscr { U } | ( \lceil k / 8 \rceil + 2 )$ . Storing the coordinates directly in FP16 requires $\begin{array} { r } { B _ { \mathrm { F P 1 6 } } = 2 \sum _ { u } k _ { u } } \end{array}$ bytes, while an uncompressed FP16 gradient requires $B _ { \mathrm { r a w } } \mathbf { \bar { \alpha } } = 2 d _ { \mathrm { a t t r } }$ bytes, where $d _ { \mathrm { a t t r } }$ is the number of attributed parameters.

Table 2: Per-example representation payloads at $k _ { u } = 2 , 0 4 8$ per module. Uncompressed gradients cover the attributed parameters, and one-bit representations include packed signs and FP16 scales. Shared artifacts are accounted for in Appendix E.1.
<table><tr><td>Model</td><td>Uncompressed FP16 (GB/example)</td><td>One-bit representation (KB/example)</td></tr><tr><td>OLMo 2 1B</td><td>2.147</td><td>28.896</td></tr><tr><td>OLMo 2 7B</td><td>12.952</td><td>57.792</td></tr><tr><td>OLMo 2 13B</td><td>25.376</td><td>72.240</td></tr><tr><td>OLMo 2 32B</td><td>62.411</td><td>115.584</td></tr></table>

Shared artifacts. The Kronecker eigenbases and half-whitening weights are shared across examples and queries. In EOGP, each module’s PCA correction matrix $P _ { u } \in \mathbb { R } ^ { m _ { u } \times k _ { u } }$ is also shared and stored in FP16, requiring

$$
B _ { P , u } = 2 m _ { u } k _ { u } \quad \mathrm { b y t e s } .\tag{39}
$$

For the 32B model with $m _ { u } = 2 6 2 , 1 4 4$ and $k _ { u } = 2 , 0 4 8$ , each FP16 correction matrix requires approximately 1.07 GB, totaling 481 GB across all 448 modules. This cost is independent of the number of stored examples, so its relative contribution decreases as the store grows. For stores containing 100 million or one billion examples, the correction matrices would occupy approximately 4.2% or 0.42% of the aggregate per-example payload, respectively. These estimates illustrate how the shared correction matrices can be amortized when constructing attribution stores for large pretraining or mid-training corpora.

When shared storage is constrained, reducing $k _ { u }$ decreases the correction-matrix storage linearly. For example, keeping $m _ { u }$ unchanged and reducing $k _ { u }$ to 256 lowers the total correction-matrix storage to approximately 60.1 GB for the 32B model. As shown in Figure 4, EOGP maintains an advantage over the evaluated compression baselines at small per-example storage budgets.

Total storage. Including shared artifacts and file overhead, the total storage is

$$
B _ { \mathrm { t o t a l } } = N _ { \mathrm { s t o r e } } B _ { \mathrm { 1 b i t } } + B _ { \mathrm { s h a r e d } } + B _ { \mathrm { o v e r h e a d } } ,\tag{40}
$$

where $B _ { \mathrm { s h a r e d } }$ counts shared curvature and projection artifacts, including $\textstyle \sum _ { u \in { \mathcal { U } } } B _ { P , u }$ for the PCA correction matrices in EOGP, and $B _ { \mathrm { o v e r h e a d } }$ counts file metadata. Per-example budgets count representation payloads, with shared artifacts accounted for separately.

## E.2 STORAGE ESTIMATES BY MODEL

Table 2 compares per-example storage for uncompressed FP16 gradients and one-bit representations at $k _ { u } = 2 , 0 4 8$ . Both EOGP and EOGP-R use $2 0 \bar { 4 } 8 / 8 + 2 = 2 5 8$ bytes per module per example.

## E.3 IMPLEMENTATION AND RUNTIME

Setup. Table 3 compares our implementation with LoGra’s official LogIX implementation on a single NVIDIA B200. EOGP uses $m _ { u } = 2 6 2 , 1 4 4 , k _ { u } = 2 , 0 4 8$ , and FP16 correction matrices, while LoGra stores 2,025 FP16 coordinates per module. Curvature statistics are fitted beforehand, and timings exclude curvature fitting and model loading.

Implementation. First-stage coordinates are cached in FP16 using asynchronous disk writes. We then process modules sequentially, loading each correction matrix onto the GPU and applying it to a batch of coordinates before one-bit encoding and recording. Each loaded matrix is reused across the batch.

Timing protocol. Store construction includes gradient computation, projection, and disk I/O, including correction-matrix loading and transfer for EOGP. Coordinate generation is timed on 40 documents per model averaging approximately 500 tokens. Subsequent PCA projection and storage use batches of approximately 11,000 documents, over which matrix-loading costs are amortized.

Table 3: PCA fitting and store-construction times on one NVIDIA B200. Fitting starts from cached coordinates and includes saving the correction matrices. Store-construction totals combine separately timed stages, including disk I/O.
<table><tr><td rowspan="2">Model</td><td rowspan="2">EOGP PCA fitting (min)</td><td colspan="2">Store construction (s/example)</td></tr><tr><td>EOGP</td><td>LoGra</td></tr><tr><td> $\mathrm { O L M o } 2 1 \mathrm { B }$ </td><td>4.5</td><td>0.0563</td><td>0.0577</td></tr><tr><td> $\mathrm { O L M o } 2 7 \mathrm { B }$ </td><td>8.9</td><td>0.0999</td><td>0.1315</td></tr><tr><td> $\mathrm { O L M o } 2 1 3 \mathrm { B }$ </td><td>11.1</td><td> $0 . 1 4 7 4$ </td><td>0.1661</td></tr><tr><td> $\mathrm { O L M o } 2 3 2 \mathrm { B }$ </td><td>18.4</td><td> $0 . 2 4 7 0$ </td><td>0.2780</td></tr></table>

![](images/2a78e13d88073e06fed7fa8fba2a751c157dafb26665f67346524b155d19f4cf.jpg)

![](images/c547b7bc00f82d9c0c521e98e0901f5ae0face1abf4ea8dd11a3ad8be3c57dba.jpg)  
Storage per training example  
EOGP LoGra basis + EK-FAC whitening GraSS basis + EK-FAC whitening LoGra (Random) $\diamond \mathsf { G r a s s }$  
Figure 6: Comparison of EOGP with LoGra and GraSS on OLMo 2 7B SFT, using either their original curvature treatment or EK-FAC half-whitening before projection. NDCG@20 (left) and Spearman correlation (right) measure agreement with EK-FAC influence at matched per-example storage budgets.

One-time PCA fitting. Fitting starts from cached first-stage coordinates of approximately 10,000 examples and follows Appendix B.2. Timings include reading the coordinates, computing the correction matrices, and saving them in FP16. This one-time cost is 18.4 minutes for the 32B model and is shared across examples and queries.

Batched query processing. For the 32B model with $m _ { u } = 2 6 2 , 1 4 4$ and $k _ { u } = 2 , 0 4 8$ , processing 100 queries against 1,000 stored examples takes 60.36 seconds on a single NVIDIA B200, starting from cached first-stage coordinates. This includes loading and transferring the FP16 correction matrices, applying the PCA projection, and scoring the stored examples. Each correction matrix is loaded once per batch, giving an amortized processing cost of 0.604 seconds per query.

## F COMPARISON UNDER SHARED EK-FAC CURVATURE

To examine whether EOGP benefits primarily from using the same curvature approximation as the evaluation reference, we construct variants of LoGra and GraSS that apply our EK-FAC halfwhitening before their respective projections. On OLMo 2 7B SFT, EOGP substantially outperforms both variants at matched per-example storage budgets, showing that matching the reference curva ture alone does not explain its advantage.

The original LoGra and GraSS configurations also generally outperform their half-whitened variants, supporting the effectiveness of their post-projection curvature approximations in this setting. One possible explanation is that estimating curvature after projection accounts for the variances and correlations of the retained coordinates, whereas applying half-whitening before projection produces a different transformation. These results suggest that the effectiveness of a curvature approximation depends on how it is combined with gradient compression.

Table 4: Pearson correlation between FP16 and one-bit influence scores on GPT-2. We report the median of the query-wise correlations over 481 queries. Storage budgets refer to one-bit storage per training example. The highest correlation at each budget is shown in bold.
<table><tr><td>Method</td><td>12 KB</td><td>24 KB</td><td>96 KB</td></tr><tr><td>EOGP</td><td>0.902</td><td>0.890</td><td>0.919</td></tr><tr><td>LoGra (Random)</td><td>0.604</td><td>0.447</td><td>0.086</td></tr><tr><td>LoGra (PCA)</td><td>0.750</td><td>0.570</td><td>-0.355</td></tr><tr><td>GraSS</td><td>0.330</td><td>0.079</td><td>-0.205</td></tr><tr><td>LoRIF</td><td>0.820</td><td>0.823</td><td>0.822</td></tr></table>

![](images/b9ec2df161605a8545b97887bbaa11387db7b23cdf7b6ccc01b3f0e8a32b1621.jpg)  
EOGP LoGra (PCA) LoGra (Random) GraSS LoRIF fp16 storage  
Figure 7: LDS of compression methods under one-bit and FP16 storage on GPT-2 at matched perexample storage budgets. Horizontal lines denote uncompressed references. Curves end where the required projection dimension exceeds the method’s supported dimension.

## G EFFECT OF ONE-BIT QUANTIZATION ON ATTRIBUTION QUALITY

We further examine how one-bit storage affects different gradient representations using the GPT-2 setting in Section 4.1. We compare attribution quality at matched per-example storage budgets and separately measure how quantization changes influence scores while holding the projection dimension fixed.

Attribution quality under one-bit storage. Figure 7 compares one-bit and FP16 storage for the evaluated compression methods. For baselines that estimate curvature after projection, larger output dimensions increase the memory required for curvature estimation, so we evaluate their one-bit variants only up to the largest feasible storage budget. Under one-bit storage, EOGP achieves the highest LDS among the evaluated compression methods at each displayed budget, with performance improving as more storage is allocated. The one-bit variants of LoGra and GraSS exhibit a different trend, with LDS declining as the budget increases. LoRIF improves more gradually but remains below EOGP. At larger storage budgets, EOGP also outperforms the other compression baselines when all methods use FP16 storage, indicating that its advantage is not solely attributable to one-bit quantization. These results show that the effectiveness of one-bit storage depends on the representation being quantized, and that EOGP’s advantage persists in a retraining-based evaluation.

Preservation of influence scores. To examine the effect of quantization directly, we compare influence scores obtained with FP16 and one-bit storage at the same projection dimension within each method. For each query, we compute the Pearson correlation between the two sets of scores across candidate training examples, then report the median correlation over queries in Table 4.

For EOGP, these correlations range from 0.890 to 0.919 across the evaluated configurations, indicating that the scores remain strongly correlated after quantization. LoGra and GraSS show substantially weaker agreement in several configurations. These correlations measure the preservation of score patterns rather than absolute score magnitudes, since Pearson correlation is unchanged by a positive rescaling or an additive shift. Together with the LDS results, they provide complementary evidence that EOGP preserves useful attribution information under one-bit storage.

A possible explanation. One possible explanation is that half-whitening makes the representation more compatible with scaled sign quantization. This quantization preserves coordinate signs but replaces their magnitudes with a common scale given by the mean absolute coordinate value. It therefore approximates vectors more accurately when their coordinate magnitudes are relatively uniform. By rescaling gradients using curvature information, half-whitening may reduce magnitude imbalances in the resulting coordinates and thereby lessen the distortion introduced by quantization. The relevant property is the relative variation in coordinate magnitudes, since uniformly rescaling a vector also rescales its quantization scale and leaves the relative reconstruction error unchanged.