# Uncertainty Quantification in Pairwise Diference Learning for Classification

Mohamed Karim Belaid<sup>1,2\*</sup>, Maximilian Rabus<sup>2</sup> and Eyke H¨ullermeier<sup>3,4,5</sup>

<sup>1\*</sup>IDIADA Fahrzeugtechnik GmbH, Munich, Germany. <sup>2</sup>Dr. Ing. h.c. F. Porsche AG, Stuttgart, Germany. <sup>3</sup>Institute of Informatics, University of Munich (LMU), Germany. <sup>4</sup>Munich Center for Machine Learning (MCML), Munich, Germany. <sup>5</sup>German Centre for Artificial Intelligence (DFKL, DSA), Kaiserslautern, Germany.

\*Corresponding author(s). E-mail(s): karim.belaid@idiada.com; Contributing authors: maximilian.rabus2@porsche.de; eyke@lmu.de;

## Abstract

Instead of learning a mapping from instances to outcomes in the standard way, the key idea of pairwise diference learning (PDL) is to learn a function that takes two instances as input and predicts the diference between the respective outcomes. Given a function of this kind, predictions for a query instance are derived from every training example and then averaged. We propose a version of PDL for the task of classification and propose a suitable meta-learning technique for this purpose. Moreover, we address the problem of uncertainty quantification, i.e., of representing and quantifying the uncertainty of a PDL classifier’s predictions in a faithful manner. Building on recent work on the decomposition of total uncertainty into aleatoric and epistemic components, we propose diferent ways of extending corresponding measures to the setting of PDL, depending on how and to what extent uncertainty is represented by the underlying base learner. In large-scale empirical studies, we analyze both the predictive accuracy and the uncertainty awareness of our PDL classifier.

Keywords: Pairwise diference learning, Classification , Uncertainty Quantification

## 1 Introduction

Pairwise Diference Learning (PDL) is a versatile meta-learning technique that solves supervised learning tasks by learning to predict the diference $\Delta ( x , x ^ { \prime } ) = f ( x ) - f ( x ^ { \prime } )$ between the outcomes of pairs of inputs — rather than outcomes of individual inputs directly, as “conventional” learners do. Point predictions can then be obtained from observed data $( x ^ { \prime } , f ( x ^ { \prime } ) )$ thanks to the relationship $f ( x ) = f ( x ^ { \prime } ) + \Delta ( x , x ^ { \prime } )$ . Averaging over many predictions of that kind turns out to significantly enhance predictive accuracy, especially in small data regimes [1–5]. This is also because, by pairing training examples, the number of training examples is inflated to the square of the original size (albeit at the cost of losing statistical independence).

PDL has originally been introduced for regression tasks, i.e., for predicting (diferences between) continuous outcomes. This paper is an extension of our recent work [5], in which we generalized PDL to the case of multiclass classification, essentially specifying the “diference” between outcomes in terms of class (in)equality. Like in the case of regression, significantly improved accuracy (compared to conventional classification) could be shown in many cases. However, while predictive accuracy is critical, capturing a model’s reliability in its predictions is not less important, particularly for high-stakes decisions in domains such as healthcare, autonomous systems, and risk management. In this article, we therefore shift our focus towards uncertainty quantification (UQ) within the PDL framework and extend the latter by methods for representing and quantifying aleatoric uncertainty (which arises from inherent randomness and noise in the data) as well as epistemic uncertainty (which reflects the learner’s lack of knowledge due to limited or incomplete training data). Our key contributions are as follows:

• To realize the idea of PDL for the task of classification, we propose a meta-learning technique that transforms any multiclass classification problem into a single binary problem (Section 3).

• We establish a comprehensive taxonomy of uncertainty representations that aligns with the PDL framework, incorporating measures for total, aleatoric, and epistemic uncertainty (Section 4).

• We conduct extensive empirical evaluations on both real-world benchmark datasets and controlled synthetic datasets, analyzing both the predictive accuracy and the uncertainty awareness of our PDL classifier. Our results also ofer insights into the interactions between the choice of the base learner, the uncertainty representation, and the UQ measures (Section 5).

• We provide a theoretical and empirical analysis of UQ within the context of the pairwise transformation induced by PDL, critically assessing the suitability and efectiveness of various UQ methods. Based on this analysis, we propose a principled selection of UQ techniques best suited for PDL classifiers (Section 6).

• We make a Python implementation of the PDL classifier available<sup>1</sup>. Our “pairwise diference learning library” (pdll) adheres to the Scikit-learn standards and ensures compatibility with any Sklearn ML model.

## 2 Related Work

Pairwise Diference Learning (PDL) has evolved into a versatile AI framework. Initially developed for regression tasks, PDL is a meta-algorithm that can be applied on top of neural networks [1] or classical machine learning regressors [2, 3]. Subsequent research extended PDL to classification, demonstrating its utility in domains such as natural language processing [4] and tabular data [5].

PDL has demonstrated remarkable performance in various domains, outperforming popular machine learning algorithms. Benchmark evaluations have used more than 400 supervised learning datasets [5–7]. In particular, PDL was applied to large-scale datasets, showcasing its ability to handle high-dimensional data, such as the Fe-DFTB dataset, with 11,656 features and 12,726 samples [2].

Beyond benchmark datasets, PDL has achieved notable results in chemistry [8, 9], image processing [10], quantum reactions [11], and drug activity ranking [12]. This foundation has evolved PDL into a versatile tool with applications in active and contrastive learning. Moreover, learning diferences is central to related domains such as Siamese networks and OpenAI’s CLIP model [13].

Uncertainty quantification (UQ) measures the confidence level of machine learning models in their predictions, assessing their reliability and trustworthiness. In deep learning, uncertainty can be estimated by incorporating specific neurons to output uncertainty estimations [14]. Alternatively, model-agnostic approaches perturb training data points or model parameters to infer prediction distributions and derive uncertainty estimates [15, 16]. A more convenient method involves exploiting intermediate predictions or results within ensemble models or PDL frameworks [2, 5]. Some authors have explored combining multiple perturbation methods, but these approaches remain insuficiently mature for real-world applications [17].

UQ research distinguishes between aleatoric uncertainty, arising from inherent data randomness, and epistemic uncertainty, resulting from a model’s incomplete knowledge of the true data-generating process [18]. This distinction is critical in applications such as medical decision-making, where understanding uncertainty sources enhances model accuracy, robustness, and trustworthiness [19]. UQ also contributed significantly in the field of active learning, Virtual screening, and Drug Discovery [16]. The decomposition of total uncertainty introduces additional complexity in selecting high-performing models and identifying an efective UQ measure [20]. For instance, Hou et al. demonstrated that methods for estimating epistemic uncertainty can also estimate aleatoric uncertainty in large language models (LLMs).

However, UQ has been sparsely explored within the context of PDL. Kramer et al. emphasized reducing aleatoric uncertainty caused by experimental measures on models accepting pairs of inputs. In regression, Jim´enez et al. introduced a PDL regressor model called ”DeltaDelta”, which replaces the absolute activity of each ligand with its diference from a known reference compound, using the variance across multiple predictions as an uncertainty estimate [23]. Tynes et al. transferred this idea to molecular property prediction and observed similar results. They used variation over pair predictions to measure the total uncertainty (though without decomposing it into aleatoric and epistemic components).

## 3 Pairwise Diference Classifier

We follow the mathematical notation introduced by Belaid et al. [5, 6]. Consider a conventional supervised learning scenario for classification with training data

$$
\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { d } \times \mathcal { V } ,
$$

where each data point consists of a feature vector $x _ { i } \in \mathbb { R } ^ { d }$ and its corresponding discrete label $y \in \mathcal { y } = \{ 1 , \dotsc , K \}$ . The data is assumed to be independently and identically distributed $\mathrm { ( i . i . d . ) }$ according to an unknown joint probability distribution. The objective is to obtain a classifier that yields accurate predictions by minimizing a prespecified loss function (in expectation).

The Pairwise Diference Classifier (PDC) transforms the original dataset as follows:

$$
\begin{array} { r } { \mathcal { D } _ { p a i r } = \left\{ \left( z _ { i , j } , y _ { i , j } \right) \vert 1 \leq i , j \leq N \right\} , } \end{array}\tag{1}
$$

where each pair $( x _ { i } , x _ { j } )$ is mapped to a joint feature representation $z _ { i , j } = \phi ( x _ { i } , x _ { j } )$ , and the label $y _ { i , j }$ is defined as

$$
y _ { i , j } = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { f o r ~ } } y _ { i } \neq y _ { j } } \\ { 1 } & { { \mathrm { f o r ~ } } y _ { i } = y _ { j } } \end{array} \right. } .\tag{2}
$$

Thus, $\mathcal { D } _ { p a i r }$ suggests the task of learning a binary classifier that predicts whether a given pair of inputs shares the same class label or not. More generally, we consider probabilistic predictors

$$
\gamma : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \to [ 0 , 1 ] ,\tag{3}
$$

where $\gamma ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } )$ is the probability that the class labels of x and $x ^ { \prime }$ coincide. To this end, γ is modeled as $\gamma ( x _ { i } , x _ { j } ) = h ( \phi ( x _ { i } , x _ { j } ) )$ ), where $h \in \mathcal H$ is a probabilistic binary classifier from a suitable hypothesis space H trained on the transformed data (1).

Given a query $x _ { q } .$ , PDC estimates the probability of class labels $y \in \mathcal { V }$ as follows: Considering each training example $( x _ { i } , y _ { i } )$ as a piece of evidence for the unknown class $y _ { q } .$ , the semantics of the above prediction suggests that the probability of the event $y _ { q } = y _ { i }$ is given by $\gamma ( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } )$ . More formally, $P ( E ) = \gamma ( x _ { q } , x _ { i } )$ , where $E$ denotes the event ${ y _ { q } } = { y _ { i } }$ (and hence $\begin{array} { r } { \dot { P } ( \neg E ) = 1 - \gamma ( x _ { q } , x _ { i } ) ) } \end{array}$ . Let p denote the prior distribution on the class labels $\mathcal { V }$ (which can easily be estimated by relative frequencies on the training data). This distribution is then updated by conditioning it on the (uncertain) event $E$ , which yields the following posterior suggested by $( x _ { i } , y _ { i } )$

$$
\begin{array} { r } { p _ { p o s t , i } ( y | x _ { q } ) = \left\{ \begin{array} { c l } { \gamma ( x _ { q } , x _ { i } ) } & { \mathrm { i f ~ } y = y _ { i } } \\ { \displaystyle \frac { p ( y ) \cdot ( 1 - \gamma ( x _ { q } , x _ { i } ) ) } { 1 - p ( y _ { i } ) } } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{4}
$$

Thus, the (posterior) probability of $y _ { i }$ is fixed to $\gamma ( \boldsymbol { x } _ { q } , \boldsymbol { x } _ { i } )$ , and all other probabilities are rescaled in a proportional way to guarantee that the sum of posterior probabilities

adds to 1. Finally, we average the evidence from all training examples to obtain

$$
p _ { p o s t } ( y | x _ { q } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } p _ { p o s t , i } ( y | x _ { q } ) .\tag{5}
$$

Note that PDL combines properties of both “eager” and “lazy” learning [24]. Like standard inductive learning methods, it uses the entire training data to induce a (global) predictive model (3), and like (lazy) instance-based learning methods, it uses the individual training examples to obtain local predictions for a new query point. In a sense, it can be seen as a sophisticated extension of nearest neighbor classification, using a learned instead of a predefined transfer function $\gamma ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } ) \equiv 1$ (nearest neighbor classification, each training instance simply predicts it’s “own” label, i.e., answers “yes” to the question of class equality) and aggregating over the entire data instead of restricting to local neighborhoods.

Also note that the averaging (5) over training examples is very much akin to averaging predictions in ensemble learning — except that conventional methods such as Random Forests or bagging aggregate predictions over multiple (weak) models (each produced through randomization of the training data data or the learning process), whereas PDL averages predictions of the same (diference) model (3) “anchored” on diferent training examples.<sup>2</sup> Nevertheless, the hope in both cases is to improve accuracy by reducing variance through averaging. Besides, as already mentioned, ensemble predictions are commonly used for UQ, with the basic intuition that diverse predictions suggest a high level of (epistemic) uncertainty [25]. Similarly, the individual predictions $p _ { p o s t , i } ( y )$ in (5) provide a natural point of departure for UQ in PDL.

## 4 Uncertainty Quantification for PDL Classifiers

The common assumption in machine learning is that the ground-truth dependence between inputs x and outcomes y is stochastic and characterized in terms of conditional probability distributions $p ( \cdot | x ) , \operatorname { i . e . , } p ( y | x )$ denotes the probability to observe label $y \in \mathcal { V }$ as an outcome given $x \in \mathcal { X }$ as input information. Since Y is finite, a distribution $p ( \cdot | x )$ can be identified with a vector $\theta = ( \theta _ { 1 } , \ldots , \theta _ { K } ) ^ { \top } \in \Delta _ { K }$ , and the set $\mathbb { P } ( \mathcal { V } )$ of all probability distributions on Y with the $( K - 1 )$ -dimensional probability simplex $\Delta _ { K }$ . We denote the ground-truth distribution $p ( \cdot \mid x )$ by $\theta _ { x } ^ { * }$ and a prediction of this distribution by $\theta _ { x }$ . In the literature on UQ in machine learning, one commonly distinguishes three levels of “uncertainty awareness” of predictive models [18]:

• Deterministic predictors $\mathcal { X } \longrightarrow \mathcal { Y }$ deliver pure class predictions and are not uncertainty aware at all; their predictions can also be seen as degenerate probabilistic predictions $\theta \in \{ 0 , 1 \} ^ { K } \cap \Delta _ { K }$ assigning probability 1 to one class and 0 to all others (a Dirac distribution).

• Probabilistic predictors $x \longrightarrow \Delta _ { K }$ deliver (first-order) probability distributions and are thus able to represent aleatoric uncertainty.

• Bayesian (second-order probabilistic) predictors $\mathcal { X } \longrightarrow \mathbb { P } ( \Delta _ { K } )$ deliver second-order distributions as predictions, i.e., probability distributions over probability distributions, which allows them to express uncertainty about the “right” prediction $\theta _ { x } \in \Delta _ { K }$ ; second-order predictors are thus able to capture epistemic uncertainty.

In PDL, we assumed the pairwise predictor γ, which we subsequently refer to as the base learner, to be a first-order predictor. In principle, however, the base learner can be of any type. Moreover, instead of merely averaging the predictions of the base learner according to (5), its predictions can also be “integrated” in order to reach a higher-level of uncertainty awareness. For example, from N first order distributions $p _ { p o s t , i } ( \cdot | x )$ , a second-order distribution can be estimated, e.g., in the form of a relative frequency distribution (histogram) or by estimating the parameters of a parametric family of distributions.

Table 1 Diferent types of predictors obtained from diferent types of base learners in PDL through averaging or integrating their predictions.
<table><tr><td>Base learner γ</td><td>Averaging</td><td>Integrating</td></tr><tr><td>Deterministic</td><td>First-order predictor  $\mathrm { p d c } : \mathcal X \to \Delta _ { K }$ </td><td>(Second-order predictor)</td></tr><tr><td>Probabilistic</td><td>First-order predictor  $\mathrm { p d c } : \mathcal { X } \to \bar { \Delta } _ { K }$ </td><td>Second-order predictor  $\mathrm { P D C } : \mathcal { X } \to \bar { \mathbb { P } ( \Delta _ { K } ) }$ </td></tr><tr><td>Bayesian</td><td> $\mathrm { P D C } : \mathcal { X } \to \bar { \mathbb { P } } ( \Delta _ { K } )$ </td><td>Second-order predictor (Third-order predictor)</td></tr></table>

Table 1 summarizes the diferent types of predictors that are obtained by either averaging or integrating the predictions of the base learners — we denote first-order predictors by pdc (lower-case) and second-order predictors by PDC (upper-case), i.e., $\theta _ { x } ~ = ~ \mathrm { p d c } ( x ) ~ \in ~ \Delta _ { K }$ is a first-order prediction produced for input $x ,$ and $Q _ { x } = \mathrm { P D C } ( x ) \in \mathbb { P } ( \Delta _ { K } )$ is a second-order distribution (assigning probability mass/- density $Q _ { x } ( \theta )$ to first-order distribution θ). There are two cases (in brackets) that do not appear to be very meaningful: Integrating deterministic predictions yields a second-order distribution that assign probability mass only to degenerate first-order distributions (upper-right case in the table). Moreover, integrating second-order predictions yields a third-order prediction that will be dificult to interpret semantically and hard to handle computationally (lower-right case). We elaborate on the diferent types of predictors further below and illustrate them by means of examples.

## 4.1 Deterministic Predictors

Deterministic base learners output “hard” class predictions, i.e., degenerate probabilistic predictions, and therefore do not capture any uncertainty. However, within the PDL framework, aggregating such deterministic predictions across multiple input pairs using (4) produces a non-degenerate probability distribution $\theta \in \Delta _ { K }$ , efectively transforming the deterministic base learner into a probabilistic predictor capable of capturing first-order uncertainty.

Fig. 1 (a) illustrates this transformation by comparing the decision boundaries of a 1-Nearest-Neighbor (1-NN) classifier, which outputs only class labels, with those of its corresponding PDL variant, denoted $\mathrm { p d c } _ { \mathrm { 1 - N N } }$ . While 1-NN provides no uncertainty information, the $\mathrm { p d c } _ { 1 - \mathrm { N N } }$ model yields a full probability distribution, thereby reflecting the inherent uncertainty in the prediction.

![](images/7f091cd69386a92b2dfc50378db95876a70e7a39397e01da300745699d0c51ad.jpg)

![](images/7d8602a704a266781dab7a974fc4835bf68758456efeacefacbd40914ae53391.jpg)

![](images/ce4b28ba3b2d1fd9c65c69ae1150b1566587b2d7d5f18189c26b1853dc962b70.jpg)

![](images/0c955cb655be699d5e35c722d6c931eae60127f1fe5237aca1811c39c318f8c9.jpg)  
Fig. 1 Comparison of learned patterns: (a) Deterministic predictor (1-NN) and its dual pdc<sub>1-NN</sub>; (b) Probabilistic predictor (DT) and its dual $\mathrm { P D C } _ { \mathrm { D T } }$

## 4.2 Probabilistic Predictors

When the base learner is probabilistic, it produces a finite set of first-order pair predictions for each input x. Averaging these predictions yields a single probability vector $\theta _ { x } \in \Delta _ { K }$ , thereby capturing only the aleatoric uncertainty inherent in the data. Alternatively, instead of simply averaging, the first-order predictions can be integrated into a second-order predictor $Q _ { x } \in \mathbb { P } ( \Delta _ { K } )$ . This second-order representation, which can be viewed as a distribution over probability distributions, encapsulates epistemic uncertainty by reflecting the variability among the individual pair predictions.

Fig. 1 (b) illustrates the diference between a conventional probabilistic predictor, a Decision Tree (DT) with a minimum of 4 training samples per leaf, and its PDL-enhanced variant. Although DT outputs a first-order prediction θ that reflects uncertainty in the predicted class, its decision boundaries partition the input space X into large homogeneous rectangles, each with a constant uncertainty estimate. In contrast, both the averaged (pdc) and integrated (PDC) predictors yield a smoother and more precise approximation of the underlying first-order uncertainty. Moreover, the integrated PDC model is capable of discerning diferences in the second-order distribution $Q _ { x } ( \theta )$ for input points with similar first-order predictions θ, thereby providing a richer representation of uncertainty.

PDL models built with probabilistic base learners achieve robust second-order uncertainty representations, widely regarded as the optimal level for UQ. In practice, second-order learners such as Random Forests or Extra Trees are more commonly employed due to their superior performance [5]. This practical preference underscores the critical need to investigate efective methods for representing and leveraging uncertainty within the PDL framework when the base learner already delivers second-order predictions.

## 4.3 Second-order Predictors

A second-order base learner γ outputs for each pair $( x , x _ { i } )$ a second-order distribution $Q _ { x } \in \mathbb { P } ( \Delta _ { K } )$ , where $Q _ { x } ( \theta )$ is the probability (density) of the first-order distribution $\theta , \mathrm { i . e . }$ , the probability (density) that $p ( \cdot | x ) = \theta$ . Practically, we assume such a learner to be realized in the form of an ensemble of first-order predictors $\left\{ \gamma _ { 1 } , \dots , \gamma _ { M } \right\}$ 2 which is the most common way of representing second-order uncertainty information in machine learning [26]. The second order predictive distribution $Q _ { x }$ is then given by a finite set of first-order predictions $\big \{ \theta _ { 1 } , \dots , \theta _ { M } \big \}$ , which can be thought of as an approximation of the underlying second-order distribution. We will use the same notation $Q _ { x }$ for both, a second-order distribution and its set-based approximation.

For a fixed input x and any anchor $x _ { i }$ (with $i ~ = ~ 1 , \ldots , N )$ , an ensemble $\left\{ \gamma _ { 1 } , \dots , \gamma _ { M } \right\}$ produces a set of predictive distributions

$$
Q _ { x , i } = \left\{ \theta _ { x , i , j } = p _ { \mathrm { p o s t } , i } ( \cdot \mid x , \gamma _ { j } ) : j = 1 , \ldots , M \right\} .
$$

Collecting these over all anchors yields

$$
\mathcal { Q } _ { x } = \left\{ \{ \theta _ { x , i , j } : j = 1 , \dots , M \} , i = 1 , \dots , N \right\} ,
$$

i.e., a nested structure forming a higher-order uncertainty representation that can be represented as follows:

<table><tr><td></td><td>γ1</td><td>γ2</td><td></td><td>γM</td><td></td></tr><tr><td>x1</td><td>θx,1,1</td><td>θx,1,2</td><td></td><td>θx,1,M</td><td>θx,1,•</td></tr><tr><td> $x _ { 2 }$ </td><td>θx,2,1</td><td>θx,2,2</td><td></td><td>θx,2,M</td><td>θx,2,•</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>xN</td><td>θx,N,1</td><td>θx,N,2</td><td></td><td>θx,N,M</td><td></td></tr><tr><td></td><td>θx,•,1</td><td>θx,•,2</td><td></td><td>θx,•,M</td><td>θx,N,</td></tr></table>

We consider three strategies for turning this rich information into a manageable second-order distribution through averaging:

• Uncertainty over ensemble learner predictions (UL). One strategy is to condense the information from each ensemble member $\gamma _ { j }$ from second-order to first-order by averaging over anchors:

$$
\theta _ { x , \bullet , j } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \theta _ { x , i , j }
$$

Collecting these first-order distributions yields the second-order distribution

$$
\begin{array} { r } { Q _ { x } ^ { \mathrm { U L } } = \{ \theta _ { x , \bullet , j } : j = 1 , \ldots , M \} . } \end{array}
$$

• Uncertainty over Pair predictions (UP). Likewise, one can condense the information from each anchor $x _ { i }$ from second-order to first-order by averaging over the

![](images/a321f3fe54e9682dbd8642d7f54209af2b0f841aee47968475d864dd78dac92c.jpg)  
Fig. 2 Second-order probability distributions predicted by $\mathrm { P D C } _ { R F }$ using, from left to right, the UL, ULP, and UP approaches. Each representation impacts the uncertainty measurements diferently.

learners in the ensemble:

$$
\theta _ { x , i , \bullet } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \theta _ { x , i , j }
$$

Collecting these first-order distributions yields the second-order distribution

$$
\begin{array} { r } { Q _ { x } ^ { \mathrm { U P } } = \{ \theta _ { x , i , \bullet } : i = 1 , \dots , N \} . } \end{array}
$$

• Uncertainty over both learner member and Pair predictions (ULP). Yet another strategy merges all distributions into a single second-order representation:

$$
\begin{array} { r } { Q _ { x } ^ { \mathrm { U L P } } = \big \{ \theta _ { x , i , j } : i = 1 , \dots , N , j = 1 , \dots , M \big \} . } \end{array}
$$

A concrete example of a second-order base learner in a PDC model is depicted in Fig. 2. The base learner used is a Random Forest (RF) with ten decision trees, supplemented by 300 anchor points for pairwise comparisons, applied to the previously introduced 2D synthetic dataset. The test instance $x = ( 0 , - 0 . 5 )$ lies in the center of a sparsely populated region of the input space X . Consequently, the prediction $\mathrm { P D C } _ { R F } ( x )$ takes the form of a third-order distribution: 300 pairwise predictions, each comprising 10 individual predictions generated by the decision trees. As Fig. 2 illustrates, the manner in which ensemble member predictions are grouped and merged significantly influences the shape of the resulting second-order distribution, ranging from nearly Gaussian to an almost uniform spread. This observation underscores the importance of selecting the appropriate method (UL, UP, or ULP) to capture the most informative representation of uncertainty.

## 4.4 Uncertainty Measures

When a model’s predictive output is a second-order distribution $Q _ { x } ,$ , measuring the overall uncertainty it represents in terms of a single scalar value is often useful. Moreover, this total uncertainty (TU) can be decomposed into aleatoric uncertainty (AU) and epistemic uncertainty (EU). AU reflects the inherent randomness in the datagenerating process that cannot be reduced by additional data. In contrast, EU arises from the learner’s lack of knowledge (e.g., a self-driving car facing an unfamiliar road) and typically diminishes as more or better data become available [18]. This decomposition is crucial for decision-making: if uncertainty is mainly aleatoric, further data will not help, whereas high epistemic uncertainty indicates that improved data or models could enhance predictions. Such distinctions are especially valuable, e.g., in active learning [27, 28] and Bayesian optimization [29, 30], where reducing EU is key to eficiency [15].

## 4.4.1 Entropy-based Information Measures

A popular family of techniques for capturing predictive uncertainty relies on information-theoretic measures, notably Shannon entropy [20]. A classical result in information theory shows that Shannon entropy H decomposes (additively) into conditional entropy and mutual information [31]: Let $Y \sim \theta _ { x }$ be the outcome produced by the first-order distribution $\theta _ { x } .$ , which is unknown and hence treated as a random variable $\Theta \sim Q _ { x }$ . Therefore, $Y \sim \bar { \theta }$ with expected first-order distribution $\bar { \theta } = \mathbb { E } _ { \Theta \sim Q _ { x } } ( \Theta )$ The entropy $H ( Y ) = H ( { \bar { \theta } } )$ , which qualifies as a measure of total uncertainty (about the outcome Y) can then be written as $H ( Y | \Theta ) + I ( Y , \Theta )$ , where $H ( Y \mid \Theta )$ is the conditional entropy and $I ( Y , \Theta )$ the mutual information between $Y$ and Θ. In the literature, it has been suggested that the former qualifies as a measure of aleatoric and the latter as a measure of epistemic uncertainty [32].

In the case where $Q _ { x }$ is represented as a finite set of first-order distributions, expectations are replaced by arithmetic averages, so that the total uncertainty $( \mathrm { T U } )$ , aleatoric uncertainty (AU), and epistemic uncertainty (EU) associated with $Q _ { x }$ are given as follows:

$$
\mathrm { T U } ( Q _ { x } ) = H ( { \bar { \theta } } ) = H \left( { \frac { 1 } { | Q _ { x } | } } \sum _ { \theta \in Q _ { x } } \theta \right) ,\tag{6}
$$

$$
\operatorname { A U } ( Q _ { x } ) ~ = ~ { \frac { 1 } { | Q _ { x } | } } \sum _ { \theta \in Q _ { x } } H ( \theta ) ,\tag{7}
$$

$$
\mathrm { E U } ( Q _ { x } ) = \mathrm { T U } ( Q _ { x } ) \ : - \ : \mathrm { A U } ( Q _ { x } ) .\tag{8}
$$

Note that total uncertainty coincides for all three approaches (UL, UP, ULP) — they average over the $\theta _ { x , i , j }$ in diferent ways but eventually produce the same <sup>¯</sup>θ.

## 4.4.2 Variance-Based Measures

As an alternative to entropy, (total) predictive uncertainty can also be captured in terms of variance, and a decomposition into aleatoric and epistemic parts can be obtained from the law of total variance [2, 15, 23]. Unlike Shannon entropy, however, variance is not directly suited to categorical outcomes. Therefore, Sale et al. proposed to represent a categorical outcome $Y \in \{ 1 , \ldots , K \}$ in terms of a binary vector (onehot encoding) $( Y _ { 1 } , \ldots , Y _ { K } )$ , where $Y _ { k } = 1$ if $Y = k$ and = 0 otherwise. $\operatorname { I f } Y \sim \theta =$ $( \theta _ { 1 } , \ldots , \theta _ { K } )$ , then each entry $Y _ { k }$ can be seen as a Bernoulli-distributed random variable with expectation $\mathbb { E } ( Y _ { k } ) = \theta _ { k }$ and variance $\mathbb { V } ( Y _ { k } ) = \theta _ { k } ( 1 - \theta _ { k } )$ . Moreover, assuming $\theta _ { k }$ to be unknown and treating it as a random variable $\Theta _ { k }$ distributed according to second-order distribution $Q _ { x } ,$ , we obtain the following decomposition:

$$
\mathrm { T U } _ { k } ( Q _ { x } ) = \mathrm { V a r } ( Y _ { k } ) = \mathbb { E } \bigl [ \Theta _ { k } ( 1 - \Theta _ { k } ) \bigr ] + \mathrm { V a r } ( \Theta _ { k } ) ,\tag{9}
$$

$$
\mathrm { A U } _ { k } ( Q _ { x } ) = \mathbb { E } \big [ \Theta _ { k } ( 1 - \Theta _ { k } ) \big ] ,\tag{10}
$$

$$
\operatorname { E U } _ { k } ( Q _ { x } ) = \operatorname { V a r } ( \Theta _ { k } ) .\tag{11}
$$

The overall uncertainty can then be computed by summing the uncertainties associated with the K Bernoulli random variables:

$$
\operatorname { T U } ( Q _ { x } ) = \sum _ { k = 1 } ^ { K } \operatorname { V a r } ( Y _ { k } ) ,\tag{12}
$$

$$
\operatorname { A U } ( Q _ { x } ) = \sum _ { k = 1 } ^ { K } \mathbb { E } \big [ \Theta _ { k } ( 1 - \Theta _ { k } ) \big ] ,\tag{13}
$$

$$
\operatorname { E U } ( Q _ { x } ) = \sum _ { k = 1 } ^ { K } \operatorname { V a r } ( \Theta _ { k } ) .\tag{14}
$$

## 5 Evaluation

The architecture of our proposed PDL classifiers enables precise estimation of predictive uncertainty, which we decompose into AU and EU. These uncertainties can be quantified using either Shannon entropy–based or variance–based metrics. Although previous literature provides insights into which measures might perform best, our empirical evaluation aims to uncover potential interactions among the PDL architecture (i.e., the pairing of data), the baseline model (probabilistic versus Bayesian), the uncertainty representation (TU, AU, EU), and the UQ metric (entropy versus variance). In this section, we evaluate PDC’s UQ performance with five base learners implemented in Scikit-learn, encompassing both probabilistic (DecisionTree, ExtraTree) and Bayesian learners (RandomForest, ExtraTrees, Bagging of MLP).<sup>3</sup>

## 5.1 Data

OpenML ofers a diverse collection of datasets, with a significant portion (37%) comprising fewer than 600 data points. Our study focuses on these smaller datasets, as the pairwise learning approach is expected to be particularly efective in such scenarios. Parameter tuning was performed via grid search CV using a subsampled TPOT search space [33]. Adhering to dataset selection criteria similar to those employed in the OpenML-CC18 benchmark and previous PDL research [5, 6, 34], we randomly selected 99 datasets (see summary statistics in Fig. 3). Although the original datasets are modest in size, the efective sample size for PDC is quadrupled through pairwise comparisons, yielding approximately 360, 000 data points. We further assessed class imbalance by examining the ”minority class” metadata, which indicates the proportion of samples belonging to the minority class in each dataset. For each of the five baseline models, we conducted 5 repetitions of 5-fold CV to ensure robust performance evaluation.

![](images/2929d2b3d22b1a923e8d9f9ad8a7691b421418e704ae53ff0545b3bc6c636dca.jpg)  
Fig. 3 Distribution of key characteristics (minimum, mean, maximum) for the 99 OpenML classification datasets.

We implemented a uniform data processing pipeline for all experimental runs using Scikit-learn [35]. This pipeline standardized numeric features, applied one-hot encoding to nominal features, and utilized ordinal encoding for ordinal features. Since the PDL framework employs a joint feature representation—defined as $\phi ( x _ { i } , x _ { j } ) =$ $( x _ { i } , x _ { j } , x _ { i } - x _ { j } )$ that incorporates the pairwise diference, all processed features are treated as numeric when computing these diferences.

## 5.2 Metrics

Performance is evaluated using the macro F1 score, which is particularly informative in imbalanced settings. The macro F1 score is the unweighted average of the F1 scores for each class:

$$
\mathrm { M a c r o ~ F 1 } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } F \ o { 1 _ { i } } ,
$$

Results are reported as mean ± standard error. For each dataset, performance is aggregated over 25 runs (5 repetitions of 5-fold CV). Wins and losses are determined by comparing average scores per dataset, with statistical significance assessed via a Student’s t-test $( \alpha = 0 . 0 5 )$ . Note that ties may occur, so wins and losses may not sum to the total number of datasets. While overall averages provide a preliminary impression, further metrics and detailed results are available in the PDL library repository for a more nuanced evaluation.

A well-established tool for evaluating the quality of UQ is the accuracy-rejection curve (ARC), which plots model accuracy as a function of the fraction of test instances rejected due to high uncertainty [36]. More concretely, consider a set of (test) instances $\{ x _ { 1 } , \ldots , x _ { T } \}$ and denote by $Q _ { x _ { i } }$ the (second-order) distribution predicted for $x _ { i }$ . Moreover, let U be a measure that quantifies predictive uncertainty. We can then sort the instances in decreasing order of uncertainty: $x _ { ( 1 ) } , x _ { ( 2 ) } , \ldots , x _ { ( T ) }$ , where the permutation (·) is such that $U ( Q _ { x _ { ( 1 ) } } ) \geq U ( Q _ { x _ { ( 2 ) } } ) \geq \cdots \geq ^ { \prime } U ( Q _ { x _ { ( T ) } } ) $ . If the measure U quantifies uncertainty in a meaningful way and indeed assigns higher uncertainty to more uncertain predictions, then we would expect that the accuracy on the “less uncertain“ cases $x _ { ( t + 1 ) } , x _ { ( t + 2 ) } , \ldots , x _ { ( T ) }$ , i.e., the accuracy of the learner when abstaining on the “more uncertain” cases $x _ { ( 1 ) } , x _ { ( 2 ) } , \ldots , x _ { ( t ) }$ , will be higher. More specifically, the accuracy should be an increasing function of the number t of abstentions.

Given the extensive number of experiments, analyzing individual ARCs is impractical for assessing overall performance. To overcome this, we aggregate ARC performance into a scalar metric by computing the rank correlation between U and a binary misclassification indicator $( z _ { i } = 1$ for misclassified $x _ { i } , z _ { i } = 0$ for correctly classified $x _ { i } )$ across all test instances. This idea is very much in line with the AUC metric, which “integrates” the ROC curve. For a perfect uncertainty measure, misclassified points $x _ { i }$ would consistently have higher values $U ( Q _ { x _ { i } } )$ , i.e., all misclassifications appear at the top of the sorted sequence $x _ { ( t + 1 ) } , x _ { ( t + 2 ) } , \ldots , x _ { ( T ) }$ , yielding a rank correlation of +1; negative values (between −1 and 0) would indicate an inverse relationship. We adopt Goodman and Kruskal’s Gamma [37] as our rank correlation measure due to its efective handling of ties. Concretely, this means that we compute $C / ( C + D )$ , where C is the number of concordant (correctly ordered) pairs $( x _ { i } , x _ { j } ) , { \mathrm { i . e . , ~ } } z _ { i } = 1 , z _ { j } = 0$ and $U ( x _ { ( i ) } ) > U ( x _ { ( j ) } )$ , and D is the number of discordant (incorrectly ordered) pairs $( x _ { i } , x _ { j } )$ , i.e., z<sub>i</sub> = 1, z<sub>j</sub> = 0 and $U ( x _ { ( i ) } ) < U ( x _ { ( j ) } )$

While the measure efectively captures total $( U = { \mathrm { T U } } )$ and aleatoric uncertainty $( U = \operatorname { A U } )$ , it is less apt for epistemic uncertainty (EU), which reflects uncertainty about the predicted distribution $\theta _ { x _ { i } }$ rather than the outcome $y _ { i }$ . Thus, EU quality is assessed by comparing it to the model’s calibration error. Test instances with high EU should exhibit higher calibration errors. As the true calibration error is unobservable, it is approximated by the expected calibration error (ECE) [38]. In our approach, test predictions are sorted by predicted probabilities and grouped into fixed bins; for each test point $i , E C E _ { i }$ is defined as the absolute diference between its predicted probability and the average true label in its bin.

## 5.3 Baseline

To the best of our knowledge, this is the first work to comprehensively estimate TU, AU, and EU in PDC models. Although Tynes et al.’s method shares conceptual similarities with our variance-based estimation of TU using the UP strategy, it was originally developed for regression tasks and employs distinct computational formulations. Consequently, we do not include direct comparisons with that approach. Instead, we establish our baseline by comparing the performance of the PDC model against its underlying base learner (e.g., when decision trees are used as base learners in PDC, we compare with standard decision tree learning, when random forests are the base learner, we compare with standard random forests, etc.).

## 5.4 Results

In this section, we provide a comprehensive evaluation of the empirical quality of the PDC framework in terms of performance and UQ. Our findings are organized according to the type of base learner employed, with both probabilistic and Bayesian models exhibiting consistent behavior within their respective groups.

Table 2 Comparing baseline classifiers to PDC using 99 datasets.
<table><tr><td rowspan="2">Classifier</td><td colspan="2">Significant wins</td><td colspan="2">Wins</td><td colspan="3">Average Test Macro F1</td></tr><tr><td>base</td><td>PDC</td><td>base</td><td>PDC</td><td>base ± sem</td><td>PDC</td><td>± sem</td></tr><tr><td>DecisionTree</td><td>2</td><td>50</td><td>22</td><td>76</td><td>0.7694 0.0037</td><td>0.7982</td><td>0.0034</td></tr><tr><td>ExtraTree</td><td>1</td><td>61</td><td>9</td><td>90</td><td>0.7434 0.0037</td><td>0.7987</td><td>0.0035</td></tr><tr><td>Bagging</td><td>3</td><td>26</td><td>27</td><td>70</td><td>0.7906 0.0035</td><td>0.8062</td><td>0.0034</td></tr><tr><td>RandomForest</td><td>5</td><td>27</td><td>22</td><td>73</td><td>0.7933 0.0035</td><td>0.8073</td><td>0.0034</td></tr><tr><td>ExtraTrees</td><td>6</td><td>24</td><td>21</td><td>77</td><td>0.7951 0.0036</td><td>0.8113</td><td>0.0035</td></tr></table>

## 5.4.1 Predictive Performance

The extended PDL classifier, when built upon ensemble models such as Random Forests and Extra Trees, achieved the highest average Macro F1 scores across 99 datasets, surpassing all baseline methods. As shown in Tab. 2, the proportion of significant wins favors the PDL classifier, indicating that in direct comparisons it is more likely to outperform its baseline counterpart.

Moreover, the PDL framework can simplify the final model. For instance, the test performance of $\mathrm { P D C } _ { \mathrm { D T } }$ is comparable to or better than that of the five benchmarked learners. This suggests that, by employing PDL, a single decision tree can match or exceed the performance of ensembles that typically consist of around 100 trees, thereby ofering a more interpretable solution. more results can be found in the appendix.

## 5.4.2 UQ of PDC Using a Probabilistic Base Classifier

When employing a probabilistic base learner, the direct estimation of total uncertainty (TU) yields a median rank correlation of 0.23 (see the blue distribution in Fig. 4). In contrast, the PDC model built atop the same probabilistic learner achieves a marked improvement, with a mean rank correlation of 0.56 and only 5% negative correlations. Moreover, beyond enhancing TU estimation, the PDC framework also enables the assessment of EU, as demonstrated by the predominance of positive ECE-EU rank correlations.

Both entropy-based and variance-based uncertainty measures facilitate meaningful quantification of prediction uncertainty within the PDC framework, and these trends are consistent across all tested probabilistic classifiers. Additionally, we observe a high Pearson correlation between AU and TU $( \rho = 0 . 9 8 )$ , indicating that in these scenarios, TU is predominantly driven by aleatoric noise. Similar distributions are obtained when evaluating AU on the misranking task.

## 5.4.3 UQ of PDC Using a Second-order Base Classifier

When the base learner is Bayesian, higher-order uncertainties become accessible, providing more nuanced estimates of uncertainty. Specifically, we consider three strategies for representing the second-order uncertainty: UL, UP, or ULP. As in the probabilistic setting, we find that the TU again exhibits a high correlation with AU, and both efectively guide selective prediction as evidenced by high rank correlations. However, the estimation of EU varies in quality depending on the chosen strategy. The EU values from ULP lead to the highest rank correlations with a slight non-significant improvement over a baseline Bayesian classifier, see Fig. 5.

![](images/5ca9aa5ae8f5c32b933c08fbfb2aeb0bc6ff54f358c08e118816e3c1cf57d942.jpg)

![](images/97f2b01fadb702f2e56a7e9520bce8ab5a26385f5acc233dbeaf1107eb7e0d87.jpg)

![](images/b35ba958f00bd3a5d1153c4b67914b621ef6b25ccd63dc4e3c1c2d92eade2dc5.jpg)

![](images/27f402fa7ed1e3fbb75999afe1f936ce92bee4fa7703766767e346ce8a3bad8e.jpg)  
Fig. 4 Distribution of Rank Correlation with probabilistic learners.

![](images/b0025b3503d8ed7a4a9bb53ac20c580c83ea9aa878fe3c415386259217609d26.jpg)

![](images/fe67b78e8a063ea10ad023f25715e04b594f1b0fde60d6c4f8364fbff984a327.jpg)

![](images/00d7e4c388152ea4e04090ce8961af6bf16cdb8c4bb4605332cf910f2e393a21.jpg)

![](images/4b86d4f0e88d596b426b5229b28dc02090cc11b6e5cb76219a2c48675f986837.jpg)  
Fig. 5 Distribution of Rank Correlation with Bayesian learners.

## 6 Discussion

## 6.1 Efect of PDC on the predictive performance

The enhanced prediction performance of PDL can be attributed to three main factors. First, the framework integrates instance-based and model-based learning by leveraging anchor points to capture local information, while employing a global classifier to capture general patterns of the data distribution. Second, PDL simplifies the original multiclass task by reducing it to a single binary problem — determining whether two instances share the same label or not. This binary reduction not only streamlines the learning problem but also contributes to improved predictive accuracy. Lastly, PDL benefits from an ensemble efect through averaging predictions from multiple anchors, which reduces the variance of prediction errors. Collectively, these mechanisms explain the superior performance of PDL compared to conventional baseline methods.

![](images/99ab618935d6c3bebc41830de024e32b034ad96facad5c8f23361f61792a5c2a.jpg)  
Fig. 6 Estimation of TU: Baseline DT and Bagging(MLP) models fail to capture total uncertainty accurately, whereas their PDC counterparts yield robust TU estimates.

## 6.2 Efect of PDC on the Uncertainty Quantification

Empirical results show that variance-based measures match or outperform Shannon entropy-based ones, supporting both prior studies and theoretical arguments [15, 20]. Integrating probabilistic learners in PDC yields a significant improvement in the average correlation between total uncertainty and misclassifications, while also substantially reducing the incidence of negative correlations across datasets—from 29% to just 5%. This observation is further corroborated by the TU estimation on a 2D synthetic dataset (see Fig. 6). A standard DecisionTree tends to assign uniformly high TU across both dense and sparse regions, whereas the PDC-enhanced DecisionTree produces a smoother uncertainty distribution that increases gradually with input sparsity. Similarly, a basic Bagging ensemble of MLPs assigns similar uncertainty levels to dense, sparse, and noisy regions, while its PDC counterpart leverages the neural network’s feature extraction capabilities to delineate curvilinear boundaries that efectively separate uncertain regions from those that are noisy or sparsely populated.

Furthermore, regarding EU, a PDC model with a probabilistic base learner delivers EU estimates comparable to those from a baseline second-order ensemble model, despite relying on only a single trained tree. This eficiency highlights the potential of the PDC framework to achieve robust UQ with significantly reduced computational complexity.

When comparing the three strategies UL, UP, and ULP for second-order base learners, ULP leads to the best quantification of EU. Apparently, the estimation of EU benefits from combining both sources of variability, namely variability from the base learners and from the anchor points (pair predictions). UL and UP lose this advantage, because they both “average out” one of these sources.

As shown in Fig. 7, the RandomForest classifier tends to incorrectly split AU and EU by allocating excessive AU to an empty region. In contrast, both PDC(DT) and PDC(RandomForest) demonstrate a consistent pattern by assigning the highest AU values to the noisy center and the highest EU values to the sparse region. This observation highlights the superior performance of the PDC methods in distinguishing between AU and EU under challenging conditions.

![](images/d1dfe0f4a6bbb7b7b131ee36593e926621355f18f8a12df08b5b408c97ccb963.jpg)  
Fig. 7 Splitting of AU and EU. RandomForest incorrectly allocates AU to an empty region, whereas both PDC(DT) and PDC(RandomForest) consistently assign the highest AU to the noisy center and the highest EU to the sparse region.

## 7 Conclusion

In this paper, we we developed an extension of pairwise diference learning from regression to the case of multiclass classification. To this end, we proposed a metalearning technique that specifies the “diference” between outcomes in terms of class (in)equality, thereby reducing the problem to a binary classification task. Like in the case of regression, empirical studies have shown significantly improved accuracy compared to conventional classification.

Building on the concept of pairwise diference classification, we also established a comprehensive framework for uncertainty quantification that adapts to the nature of the base learner. Specifically, we distinguish three levels of uncertainty-awareness of the base learner: deterministic, probabilistic, and second-order (Bayesian). Thanks to its ensembling nature, PDC “lifts” the uncertainty-awareness of a deterministic base learner to the probabilistic level, which allows it to capture aleatoric uncertainty. Likewise, PDC lifts a probabilistic base learner from the first to the second level, thereby allowing for the representation of both AU and EU. For the case of a base learner that is already Bayesian, we developed strategies to keep it second-order, because a third-order learner would be overly complex to interpret.

Our methodology was rigorously validated through extensive empirical studies. As a takeaway, we recommend using the uncertainty over ensemble learners and pair predictions (ULP) approach, which coincides with the uncertainty over the pair predictions (UP) approach when the base learner is probabilistic instead of Bayesian. Although other approaches could perform better on specific datasets, these approaches yield the best results most of the time. Furthermore, variance-based measures consistently outperform information-theoretic uncertainty measures. Following these recommendations, PDC models can represent predictive uncertainty in a faithful manner and, compared to the underlying base learners, deliver significantly enhanced uncertainty quantification, thereby fostering more reliable and interpretable decisions in high-stakes applications.

## References

[1] Wetzel, S.J., Ryczko, K., Melko, R.G., Tamblyn, I.: Twin neural network regression. Applied AI Letters 3(4), 78 (2020)

[2] Tynes, M., Gao, W., Burrill, D.J., Batista, E.R., Perez, D., Yang, P., Lubbers, N.: Pairwise diference regression: A ML meta-algorithm for improved prediction and uncertainty quantification in chemical search. J. Chem. Inf. Model. 61 (2021)

[3] Xie, J., Kaiwei, C., Kong, L., Zhou, J., Qu, W.: Automated essay scoring via pairwise contrastive regression. In: Proc. of the 29th COLING (2022)

[4] Corbara, S., Moreo, A., Sebastiani, F.: Same or diferent? dif-vectors for authorship analysis. ACM Transactions on KDD 18(1), 1–36 (2023)

[5] Belaid, M.K., Rabus, M., H¨ullermeier, E.: Pairwise diference learning for classification. In: DS, pp. 284–299 (2024). Springer

[6] Belaid, M.K., Wibiral, T., Rabus, M., H¨ullermeier, E.: Weighted Pairwise Diference Learning. In: Proc. of LWDA’24. CEUR-WS.org, W¨urzburg (2024)

[7] Wetzel, S.J.: How to Get the Most out of Twinned Regression Methods. arXiv (2023)

[8] Fralish, Z., Chen, A., Skaluba, P., Reker, D.: DeepDelta: predicting ADMET improvements of molecular derivatives with deep learning. J. Cheminform. 15(1)

[9] Spiers, R.C., Norby, C., Kalivas, J.H.: Physicochemical Responsive Integrated Similarity Measure (PRISM) for a Comprehensive Quantitative Perspective of Sample Similarity Dynamically Assessed with NIR Spectra. Anal. Chem. (2023)

[10] Hu, J., Yang, S., Mao, J., Shi, C., Wang, G., Liu, Y., Pu, X.: Exploring a general convolutional neural network-based prediction model for critical casting diameter of metallic glasses. J. Alloys Compd. 947, 169479 (2023)

[11] Chen, Y., Ou, Y., Zheng, P., Huang, Y., Ge, F., Dral, P.O.: Benchmark of generalpurpose ML-based quantum mechanical method AIQM1 on reaction barrier heights. J. Chem. Phys. 158(7) (2023)

[12] Wang, Y., King, R.D.: Extrapolation is Not the Same as Interpolation. In: 26th Int. Conf. DS2023, Porto, Portugal, Oct 9–11, 2023, Proc. Lecture Notes in Computer Science, vol. 14276, pp. 277–292. Springer, Cham (2023)

[13] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML, pp. 8748–8763 (2021). PmLR

[14] Sensoy, M., Kaplan, L., Kandemir, M.: Evidential deep learning to quantify

classification uncertainty. NeurIPS 31 (2018)

[15] Sale, Y., Hofman, P., Wimmer, L., H¨ullermeier, E., Nagler, T.: Second-Order Uncertainty Quantification: Variance-Based Measures. arXiv (2023)

[16] Yu, J., Wang, D., Zheng, M.: Uncertainty quantification: Can we trust artificial intelligence in drug discovery? iScience 25(8), 104814 (2022)

[17] Valdenegro-Toro, M., Jong, I.P., Zullich, M.: Unified uncertainties: Combining input, data and model uncertainty into a single formulation. In: Proceedings of the LatinX in AI Research Workshop at ICML 2024 (2024)

[18] H¨ullermeier, E., Waegeman, W.: Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Mach Learn 110(3) (2021)

[19] L¨ohr, T., Ingrisch, M., H¨ullermeier, E.: Towards Aleatoric and Epistemic Uncertainty in Medical Image Classification. In: Finkelstein, J., Moskovitch, R., Parimbelli, E. (eds.) Artificial Intelligence in Medicine. Springer, Cham (2024)

[20] Wimmer, L., Sale, Y., Hofman, P., Bischl, B., H¨ullermeier, E.: Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures? In: Proc. of the 39th Conf. on Uncertainty in AI, pp. 2282–2292. PMLR, Budapest, Hungary (2023)

[21] Hou, B., Liu, Y., Qian, K., Andreas, J., Chang, S., Zhang, Y.: Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling. In: ICML, pp. 19023–19042 (2024). PMLR

[22] Kramer, C., Fuchs, J.E., Whitebread, S., Gedeck, P., Liedl, K.R.: Matched Molecular Pair Analysis: Significance and the Impact of Experimental Uncertainty. J. Med. Chem. 57(9), 3786–3802 (2014)

[23] Jim´enez, J., P´erez-Benito, L., Martinez-Rosell, G., Sciabola, S., Torella, R., Tresadern, G., De Fabritiis, G.: DeltaDelta neural networks for lead optimization of small molecule potency. Chemical science 10(47), 10911–10918 (2019)

[24] Aha, D.W.: Lazy learning. In: Lazy Learning. Springer, Dordrecht (1997)

[25] Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Proc. 31st Conf. on NeurIPS, Long Beach, California, USA (2017)

[26] Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 30 (2017)

[27] Shelmanov, A., Puzyrev, D., Kupriyanova, L., Khromov, N., Dylov, D., Panchenko, A., Belyakov, D., Larionov, D., Artemova, E., Kozlova, O.: Active

learning for sequence tagging with deep pre-trained models and bayesian uncertainty estimates. In: 16th EACL, pp. 1698–1712 (2021)

[28] Nguyen, V.-L., Shaker, M.H., H¨ullermeier, E.: How to measure uncertainty in uncertainty sampling for active learning. Mach Learn 111(1), 89–122 (2022)

[29] Hofer, J.G., Ranftl, S., Geiger, B.C.: Robust Bayesian target value optimization. Computers & Industrial Engineering 180, 109279 (2023)

[30] Stanton, S., Maddox, W., Wilson, A.G.: Bayesian optimization with conformal prediction sets. In: AISTATS, pp. 959–986. PMLR, Bordeaux, France (2023)

[31] Ash, R.B.: Information Theory. Courier Corporation, Hoboken, NJ (2012)

[32] Depeweg, S., Hernandez-Lobato, J.-M., Doshi-Velez, F., Udluft, S.: Decomposition of uncertainty in Bayesian deep learning for eficient and risk-sensitive learning. In: ICML, pp. 1184–1193. PMLR, Stockholm, Sweden (2018)

[33] Olson, R.S., Bartley, N., Urbanowicz, R.J., Moore, J.H.: Evaluation of a treebased pipeline optimization tool for automating data science. In: Proc. GECCO, pp. 485–492 (2016)

[34] Bischl, B., Casalicchio, G., Feurer, M., Gijsbers, P., Hutter, F., Lang, M., Mantovani, R.G., Rijn, J.N., Vanschoren, J.: Openml benchmarking suites. In: 35th NeurIPS Datasets and Benchmarks Track (Round 2) (2017)

[35] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al.: Scikit-learn: ML in Python. the Journal of ML research 12, 2825–2830 (2011)

[36] H¨ullermeier, E., Brinker, K.: Learning valued preference structures for solving classification problems. Fuzzy Sets and Systems 159(18), 2337–2352 (2008)

[37] Goodman, L.A., Kruskal, W.H.: Measures of Association for Cross Classifications. Springer, New York (1979)

[38] Naeini, M.P., Cooper, G.F., Hauskrecht, M.: Obtaining well calibrated probabilities using bayesian binning. In: Bonet, B., Koenig, S. (eds.) Proc. of the 29th AAAI, Texas (2015)

[39] H¨uhn, J.C., H¨ullermeier, E.: FR3: A fuzzy rule learner for inducing reliable classifiers. IEEE Transactions on Fuzzy Systems 17(1), 138–149 (2008)

## Appendix A Choice of Base Learners

In principle, any (probabilistic) binary classifier can serve as a base learner for PDL, making it a flexible wrapper applicable to both binary and multiclass settings. In practice, however, some classifiers are more appropriate than others.

First, even if the original dataset D is i.i.d., the paired dataset $\mathcal { D } _ { p a i r }$ is not, since an instance $x _ { i }$ can appear in multiple pairs—similar to scenarios in metric learning. While many algorithms are robust to such dependency violations, some methods may be more sensitive than others.

Second, the joint feature representation $z = \phi ( x , x ^ { \prime } ) \ ( \mathrm { e . g . } $ , the concatenation of $x ,$ $x ^ { \prime } { \mathrm { . } }$ and $\boldsymbol { x } - \boldsymbol { x } ^ { \prime } )$ introduces perfect multicollinearity, which can hinder certain models, notably linear ones [2, p.8], though others handle it more efectively.

Although a detailed analysis of base learner suitability is beyond our scope, our findings suggest that non-parametric methods are generally more robust and deliver superior performance. Accordingly, our experimental evaluation predominantly employs tree-based methods, which also benefit from fast training times.

## Appendix B Complexity

Assume that a base learner operates with complexity $\mathcal { O } ( p ( N , M , F , K ) )$ , where $p ( \cdot )$ is polynomial in the number of training examples (N), test examples (M), input features $( F )$ , and output classes (K). Within the PDL framework, this complexity becomes $\mathcal { O } ( p ( N ^ { 2 } , 2 M A , 3 F , 2 ) )$ due to several factors:

• The training data are expanded into $N ^ { 2 }$ pairs.

• The joint feature representation combines F features from each instance and their diference, resulting in $3 F$ features.

• Each test point is paired with A anchors and, to ensure symmetry, predictions are computed twice, yielding 2MA predictions.

• The classification task is reduced to a binary decision $( \mathrm { i . e . , } K = 2 )$

## Appendix C Extended performance analysis

## C.1 Analyzing the Unique Contribution

Although PDL classifiers often outperform baseline models in head-to-head comparisons, the primary goal is to achieve the best performance on each dataset. Prior to introducing PDC, the highest average Macro F1 score across 99 datasets was $0 . 8 1 1 2 \pm 0 . 0 0 3 5$ . With PDC, improvements were observed on 75 datasets, raising the record to $0 . 8 2 4 3 \pm 0 . 0 0 3 1$ . This improvement underscores PDC’s distinct contribution to the field, as its incremental gain of 0.0131 significantly exceeds that of other estimators. In fact, PDC’s contribution is 32 times greater than that of the best baseline.

## C.2 Analyzing Overfitting

PDL classifiers exhibit a notable reduction in overfitting. Across 199 cross-validation runs where both the baseline and PDL classifiers showed non-significant diferences in training Macro F1 scores, PDL consistently achieved a smaller train-test gap. This trend remains evident even without restricting the analysis to cases with similar training scores. Furthermore, an examination of the impact of dataset size on overfitting (Tab. C2) shows that the PDL method consistently enhances test performance and mitigates overfitting, regardless of dataset size. These results highlight the robustness of the PDL approach, particularly in small datasets, and confirm its eficacy across diverse data scales.

Table C1 Unique contribution of each estimator to the average Macro F1 score using the best optimized model on each dataset.
<table><tr><td>Estimator</td><td>Unique contribution</td><td>Wins</td></tr><tr><td>ExtraTree</td><td>0</td><td>0</td></tr><tr><td>RandomForest</td><td>0.00002</td><td>1</td></tr><tr><td>Bagging</td><td>0.00004</td><td>2</td></tr><tr><td>DecisionTree</td><td>0.00020</td><td>10</td></tr><tr><td>ExtraTrees</td><td>0.00041</td><td>9</td></tr><tr><td>PDC</td><td>0.01312</td><td>75</td></tr></table>

Table C2 Comparing test Macro F1 on the subset of runs where train scores are not significantly diferent.
<table><tr><td></td><td># CV</td><td>Baseline Macro F1</td><td>Test Train</td><td>PDC Macro F1</td><td>Test ∆F1</td><td>Test p-value</td></tr><tr><td>Estimator DecisionTree</td><td>runs 14</td><td>Train 0.950</td><td>0.884 0.955</td><td>Test 0.895</td><td>0.011</td><td>10-05</td></tr><tr><td>ExtraTree</td><td>11</td><td>0.915</td><td>0.844 0.924</td><td>0.861</td><td>0.017</td><td>10-04</td></tr><tr><td>ExtraTrees</td><td>26</td><td>0.985</td><td>0.828 0.991</td><td>0.853</td><td>0.025</td><td>10-16</td></tr><tr><td>RandomForest</td><td>18</td><td>0.992</td><td>0.855 0.997</td><td>0.881</td><td>0.026</td><td>10-11</td></tr><tr><td>Bagging</td><td>20</td><td>0.998</td><td>0.835 0.999</td><td>0.859</td><td>0.024</td><td>10-15</td></tr></table>

## Appendix D Visualizing the Uncertainty

The ”X” dataset is a synthetic dataset specifically designed to facilitate the visualization of both classifier decision boundaries and their associated uncertainty estimates. It is a two-class classification dataset characterized by non-linearly separable distributions. Each class is generated as a mixture of Gaussians, with one class following a diagonal alignment and the other along the anti-diagonal. This structured arrangement inherently creates a well-defined central region exhibiting high aleatoric uncertainty (AU), where class overlap is maximal. Additionally, the dataset contains four peripheral regions of low-density support, corresponding to areas of high EU, making it well-suited for benchmarking uncertainty-aware classification models. The uncertainty patterns for the two probabilistic learners (DecisionTree and Extra-Tree) are nearly indistinguishable, as are those for RandomForest and ExtraTrees. For clarity and conciseness, we therefore omit the redundant plots for ExtraTree and ExtraTrees.

## Appendix E Extended Benchmark Results

## E.1 Accuracy-Rejection Curves

To assess the informativeness of our uncertainty estimates, we construct Accuracy-Rejection Curves (ARCs). For each rejection percentage, we discard predictions corresponding to the most uncertain test instances and compute the accuracy on the remaining subset [39]. Under an efective UQ scheme, the accuracy should increase monotonically as the fraction of rejected instances rises, because misclassifications are more prevalent among highly uncertain predictions. In contrast, if the uncertainty estimates were non-informative (e.g., derived from random values), the ARC would remain nearly flat. We show 5 example datasets.

![](images/a989e47df398c32dfbb5ecc73ec7f7f83149d3ed9742843a4aa7d56bf81d9baa.jpg)

![](images/b11e9fa9231ef0f2a3fdbe06b658378af02a67cbd4d16ad8d064196640a7ff2c.jpg)

![](images/27731c5560723a9e14ebf93f09f9b8d2de76b4b639e0dd9e57d41d057e70d46f.jpg)  
Fig. E1 Accuracy rejection curve for the Dataset 772.

![](images/135b27bb4e6e12417fe468f2a48f2bf00b9db5e2e7837c69bc5a11992db2b8a1.jpg)

![](images/eaa789d81683064dcd67711f70455c1758c08f21a0d9c5a06c4427ad14dc9c68.jpg)

![](images/545b84281a62a402bc2d764691376bb38ae25f143bc66328acabbcdf7427779c.jpg)

![](images/d6a2a220682732493f5dfab768193a77128b7e0c42d153daee470266a090d740.jpg)

![](images/978ffbe778985d12464c982b77d1a70abcaf0c8e65ae64f3d3f99a1633f5f56a.jpg)  
Fig. E2 Accuracy rejection curve for the Dataset 991.

![](images/740cd6d9f11b0f23d4fe2a42acfdf32fe706383d428359fde48c9c98ad392b74.jpg)

![](images/462086d7d8e4057e5b2014b7bdc9e33b0c2aea63888e217c79254c81aec262af.jpg)

![](images/2245c784166d7914150b778113545dea9a41e4072519e88f7cec6f23041ff056.jpg)  
Fig. E3 Accuracy rejection curve for the Dataset 1462.

![](images/aaebaf561fb3cfd7292853874fa1b3fc6ce4f55580469399a552b4a375bcba1d.jpg)

Probabilistic learners. OpenML data ID: 40704  
![](images/97c93f814a983dabcf9523e39f23ab3a0df7c106e965bc39e9366f18471586c8.jpg)

Probabilistic learners, OpenML data ID: 40704  
![](images/394f5d4f8f5fe9aac5829a31068f9a08e80097d87ece902122d5822540e8ab43.jpg)

![](images/73bb18beaba51c88dd0f6e03b54d964379a4c50fa2b0d512120b12e78c04624d.jpg)

![](images/f1646d0ec9d2dfbf1735e12af8365793b630ac0e790b89ae9c280c7e758671c1.jpg)  
Fig. E4 Accuracy rejection curve for the Dataset 40704.

![](images/cff8bfd354af75b60f79f9d36756f315be8ce6a8a0283050cd02b0683fecc8bc.jpg)

![](images/2212b4dc186bac2ada201cbdbbcaa107e5880fcdaf16a5fe2c493bae43704166.jpg)

![](images/fa7339113a7be926672b46bad420e1a2f34850d95996592c0ce82c3789d21e67.jpg)  
Fig. E5 Accuracy rejection curve for the Dataset 44490.

![](images/8959174aab6a9836715f71e673e4577b0d18fde2801b13742b069537598dbfc9.jpg)

## E.2 ECE-Rejection Curves

A central aim of any EU measure is to capture the model’s lack of knowledge. In an ideal scenario, test instances with higher EU should correspond to larger calibration errors. Since the true calibration error is unobservable in real-world settings, we approximate it by computing the Expected Calibration Error for each test point, denoted $E C E _ { i }$ . Specifically, $E C E _ { i }$ is obtained by partitioning the test predictions into bins and measuring the absolute diference between the predicted probability and the average true label within each bin. $\mathrm { B y }$ sorting test instances in descending order of EU, a well-calibrated EU measure should produce a quasi-monotonic decrease in $E C E _ { i }$ , thereby confirming that higher uncertainty is indeed associated with poorer calibration. We plot the average ECE for the subset of test points remaining after rejection, as a function of the rejection rate.

![](images/06947c9278264736899764af2bada009e04be0e42914991bd135f3ec0b6fce3b.jpg)

![](images/f44c3f3c7c4d86a0efbe1a2a9f89fb05baba97dc1ee7e872177a22c3c772316a.jpg)

![](images/a552cdc2d5883909c120ea208aa4c753eeaf05d61bdf53a0a8e397c098485a66.jpg)  
Fig. E6 ECE rejection curve for the Dataset 772.

![](images/b4adfe605822572df382319adf244cea649c7f27a525046146f461f510533ddc.jpg)

![](images/6242c0ec180319e8c81ffda72cc2ade142694290ba82153bd188fc11b7c49502.jpg)

![](images/80d8c972d5d94f9a3aec9ad00b6e6ab3b352d1fb85c19a4890d12993209431da.jpg)

![](images/f9e55feef02ae327eced99b4b5cae0c1be1455b5259c2e0ebaaa7fa684b43e06.jpg)

![](images/b17957e1680d1b27a85d9f80190a1a22a4ad3d86b8dd391a10a7565660c10bdd.jpg)  
Fig. E7 ECE rejection curve for the Dataset 991.

Probabilistic learners, OpenML data ID: 1462  
![](images/7cb2010ac1322ae6c0eea33bb8663ed6f9a02858fc6f55e4e2d1e283fc08a043.jpg)

Probabilistic learners, OpenML data ID: 1462  
![](images/43d3ef9e5c66aa2e519531feeecf6ce3630a4e1124182f0843c912825f3c5e55.jpg)

![](images/b2c7e0809892e5995da210f0d476c654b64a34f6b0685567e56d250c24e6a986.jpg)  
Fig. E8 ECE rejection curve for the Dataset 1462.

![](images/fe2ac75a277da83809f9bbeed847549ba0b72a7783d9453f8d991ab6507df731.jpg)

![](images/4f421cfd22aa13102ae5f508f575b99cb83a23a2f09bdbacd731e405c5e1f3d0.jpg)

![](images/b3eaac07bcc544a6479f196c1c5e9528bd5c7dc345cf2593109edf989023df2d.jpg)

![](images/474bab64b9d187193a4f9ae428446633ef3ccde8c4d940cb0f023eecb532429e.jpg)  
Fig. E9 ECE rejection curve for the Dataset 40704.

![](images/f002d2b38d98e0b20030b8eb5c6d30030e7d738e0a72a6c4217fca3309be1167.jpg)

Probabilistic learners. OpenML data ID: 44490  
![](images/212d3897db2e66fc1f9b9f6337b3e8ef05531935ee315033ccb79edf1c4a1cbb.jpg)  
Bayesian learners, OpenML data ID: 44490

Probabilistic learners. OpenML data ID: 44490  
![](images/9bcab304ef7304d563db51fdf0307adb92d650ec27530c6aca27fd2e8997bc9e.jpg)  
Bayesian learners, OpenML data ID: 44490

![](images/62ea3ad281ddb329f250b60930963cc0dd32d53556f329fa050b735a9a5f5a8e.jpg)

![](images/5b46c8edc04289f7e8bacc8ee54ad59ee7800d1f2d734b6673d639aa8f82ed74.jpg)  
Fig. E10 ECE rejection curve for the Dataset 44490.