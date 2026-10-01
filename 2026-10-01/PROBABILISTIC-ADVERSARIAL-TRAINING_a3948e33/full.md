# PROBABILISTIC ADVERSARIAL TRAINING

Andi Zhang<sup>1∗</sup> Xingyu Zhao<sup>1,2</sup> Siddartha Khastgir<sup>1</sup>

andi.zhang@warwick.ac.uk

<sup>1</sup>WMG, University of Warwick <sup>2</sup>Wuhan University

## ABSTRACT

Building on a probabilistic perspective in which adversarial examples arise from the overlap between a distance-based distribution $p _ { \mathrm { d i s } }$ and a victim-classifierinduced distribution $p _ { \mathrm { v i c } }$ , we start from a simple intuition: adversarial examples become harder to generate when these two distributions are pushed apart, as their overlap becomes smaller, thereby increasing robustness. This intuition naturally motivates a KL-based robustness objective. We then prove that $\mathrm { K L } ( p _ { \mathrm { d i s } } \Vert p _ { \mathrm { v i c } } ) - \log Z _ { \mathrm { v i c } }$ is a lower bound on probabilistic robustness (PR), where $Z _ { \mathrm { v i c } }$ denotes the normalizing constant of $p _ { \mathrm { v i c } } .$ Since PR is generally intractable to compute directly, maximizing this KL-based lower bound provides a tractable surrogate objective for improving PR. We further show that this objective recovers a scaled form of adversarial training, offering a probabilistic interpretation of adversarial training and a principled route to robustness improvement. We call the resulting method probabilistic adversarial training. Experiments show that it consistently improves PR, and ablation studies demonstrate that the induced scaling factor can even enhance the PR of non-probabilistic adversarial training methods.

## 1 INTRODUCTION

Adversarial robustness refers to a model’s ability to maintain correct predictions when its inputs are subjected to small perturbations. This property is important in the era of deep learning, as a substantial body of work (Goodfellow et al., 2015; Kurakin et al., 2017; Madry et al., 2018; Moosavi-Dezfooli et al., 2016; 2017; Papernot et al., 2016; Szegedy et al., 2014) has shown that deep neural networks can be highly vulnerable to adversarial perturbations.

To better understand and evaluate this vulnerability, Zhang et al. (2024a) proposed probabilistic adversarial attack. By applying Langevin Dynamics, they demonstrated that adversarial examples can be viewed as samples drawn from an adversarial distribution. This distribution is essentially the product of a “victim” distribution $p _ { \mathrm { v i c } }$ that encourages misclassification, and a “distance” distribution $p _ { \mathrm { d i s } }$ that constrains the perturbation magnitude.

However, their framework is strictly confined to targeted attacks and focuses exclusively on the generation of adversarial examples, leaving a systematic approach to model defense unexplored. On the other side, to formally quantify model security, Webb et al. (2018) introduced the concept of probabilistic robustness (PR), defined as the probability that a classifier successfully resists perturbations sampled from a specific distribution.

In this work, we unify the probabilistic attack perspective (Zhang et al., 2024a) with the probabilistic robustness framework (Webb et al., 2018). We theoretically prove that KL $( p _ { \mathrm { d i s } } | | p _ { \mathrm { v i c } } ) - \log Z _ { \mathrm { v i c } }$ is a lower bound on probabilistic robustness, where $Z _ { \mathrm { v i c } }$ denotes the normalizing constant of $p _ { \mathrm { v i c } }$ . Since computing and optimizing PR directly is generally intractable, maximizing this KL-based lower bound provides a tractable surrogate objective to effectively enhance PR.

This theoretical result aligns with a simple yet profound intuition. In essence, adversarial examples naturally arise from the overlap between the distance distribution $p _ { \mathrm { d i s } }$ (which confines the perturbation to be close to the original image) and the victim distribution $p _ { \mathrm { v i c } }$ (which actively drives the model toward misclassification). By maximizing the KL divergence ${ \mathrm { K L } } ( p _ { \mathrm { d i s } } \| p _ { \mathrm { v i c } } )$ , we are pushing these two distributions apart. As their overlap shrinks, it becomes inherently more difficult for an attacker to sample an adversarial example that simultaneously satisfies the distance constraint and triggers a misclassification.

We further show that maximizing this lower bound essentially recovers a scaled form of adversarial training. This provides a probabilistic interpretation of adversarial training itself, and provides a principled route to robustness improvement, which we term Probabilistic Adversarial Training (PAT). Empirically, PAT consistently improves probabilistic robustness. Moreover, our ablation studies reveal an intriguing byproduct: the theoretical scaling factor derived from our framework can plug into and enhance traditional, non-probabilistic defense methods, such as PGD-based adversarial training.

To summarize, our main contributions are as follows:

• We formalize the untargeted version of probabilistic adversarial attacks. (Section 3)

• We prove that KL $\begin{array} { r l } {  { \iota ( p _ { \mathrm { d i s } } \| p _ { \mathrm { v i c } } ) - \log Z _ { \mathrm { v i c } } } } \end{array}$ is a lower bound on probabilistic robustness (PR). (Section 4)

• By maximizing this KL-based lower bound, we propose Probabilistic Adversarial Training (PAT), which is theoretically guaranteed to improve PR. Moreover, its importance sampling formulation makes the estimation of the bound tractable (Section 5).

• We empirically demonstrate that PAT consistently improves PR, and its derived scaling factor can even enhance non-probabilistic methods like PGD-based adversarial training. (Section 6)

## 2 PRELIMINARIES

## 2.1 ADVERSARIAL ATTACK

Let $C : [ 0 , 1 ] ^ { d } \to \mathcal { Y }$ be a classifier with input dimension d and label space $\mathcal { V }$ . Let $x _ { \mathrm { o r i } } \in [ 0 , 1 ] ^ { d }$ be a clean (original) image with its corresponding true label $y _ { \mathrm { o r i } } \in \mathcal { V }$ . Adversarial attacks generally fall into two categories. The goal of an untargeted adversarial attack is to construct an adversarial example $x _ { \mathrm { a d v } }$ that simply causes the classifier to misclassify the image, meaning $C ( x _ { \mathrm { a d v } } ) \ne y _ { \mathrm { o r i } }$ Alternatively, given a specific target label $y _ { \mathrm { t a r } } \in \mathcal { V }$ (where $y _ { \mathrm { t a r } } \neq y _ { \mathrm { o r i } } )$ , a targeted adversarial attack aims to force the classifier to predict that exact label, such that $C ( x _ { \mathrm { a d v } } ) = y _ { \mathrm { t a r } }$ . In both scenarios, the distance between $x _ { \mathrm { a d v } }$ and $x _ { \mathrm { o r i } }$ must remain small. The corresponding optimization problem can be formulated as:

$$
\operatorname* { m i n } \mathcal { D } \big ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } \big ) \quad \mathrm { s u b j e c t t o } \quad C ( x _ { \mathrm { a d v } } ) \in \mathcal { T } \quad \mathrm { a n d } \quad x _ { \mathrm { a d v } } \in [ 0 , 1 ] ^ { d } ,
$$

where $\tau = \{ y _ { \mathrm { t a r } } \}$ for a targeted attack and ${ \mathcal { T } } = { \mathcal { y } } \setminus \left\{ { \mathcal { y } } _ { \mathrm { o r i } } \right\}$ for an untargeted attack, and $\mathcal { D }$ measures the distance (similarity) between $x _ { \mathrm { o r i } }$ and $x _ { \mathrm { a d v } }$ , typically via an $L _ { 1 } , L _ { 2 }$ , or $L _ { \infty }$ norm. Directly solving this constrained optimization can be challenging. To address this, Szegedy et al. (2014) propose relaxing it into the following optimization problem:

$$
\operatorname* { m i n } _ { x _ { \mathrm { a d v } } \in [ 0 , 1 ] ^ { d } } ~ c _ { 1 } \mathcal { D } ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } ) ~ + ~ c _ { 2 } L ( x _ { \mathrm { a d v } } )\tag{1}
$$

where $c _ { 1 } > 0$ and $c _ { 2 } > 0$ are trade-off constants. For a targeted attack, we set $L ( x _ { \mathrm { a d v } } ) ~ =$ $f ( x _ { \mathrm { a d v } } , y _ { \mathrm { t a r } } )$ to minimize the loss with respect to the target label; for an untargeted attack, we set $L ( x _ { \mathrm { a d v } } ) = - f ( x _ { \mathrm { a d v } } , y _ { \mathrm { o r i } } )$ to maximize the loss with respect to the true label, thereby pushing the prediction away from the original class. In Szegedy et al. (2014)’s work, $f$ is taken to be the cross-entropy loss<sup>1</sup>; Carlini & Wagner (2017) present additional choices for $f .$

## 2.2 PROBABILISTIC ADVERSARIAL ATTACK

Zhang et al. (2024a) derive a probabilistic perspective on adversarial attacks by applying Langevin Dynamics as an optimizer for equation 1. Notably, their framework focuses exclusively on targeted

attack. Under this specific setting (where $L ( x _ { \mathrm { a d v } } ) = f ( x _ { \mathrm { a d v } } , y _ { \mathrm { t a r } } ) )$ , they introduce the adversarial distribution:

$$
\begin{array} { r } { p _ { \mathrm { a d v } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } , y _ { \mathrm { t a r } } ) \propto p _ { \mathrm { v i c } } ( x _ { \mathrm { a d v } } ; y _ { \mathrm { t a r } } ) p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } ) , } \end{array}
$$

where $p _ { \mathrm { v i c } } ( x _ { \mathrm { a d v } } ; y _ { \mathrm { t a r } } ) \propto \exp \bigl ( - c _ { 2 } f ( x _ { \mathrm { a d v } } , y _ { \mathrm { t a r } } ) \bigr )$ is the “victim” distribution emphasizing misclassification toward the target label $y _ { \mathrm { t a r } }$ , and $p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } )$ ∝ $\exp \bigl ( - c _ { 1 }  { \mathcal { D } } ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } ) \bigr )$ is the “distance” distribution around $x _ { \mathrm { o r i } }$ . This formulation leverages the fact that Langevin Dynamics converges to the corresponding Gibbs distribution (Lamperski, 2021), thereby providing a probabilistic interpretation of adversarial examples.

This probabilistic perspective aligns with traditional geometry-based adversarial attacks. For example, if D is the $L _ { 1 }$ norm, then $p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } ) \propto \exp ( - \| x _ { \mathrm { a d v } } - x _ { \mathrm { o r i } } \| _ { 1 } )$ takes the form of a Laplace distribution; if D is the squared $L _ { 2 }$ norm, then $p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } ) \propto \exp ( - \| x _ { \mathrm { a d v } } - x _ { \mathrm { o r i } } \| _ { 2 } ^ { 2 } )$ is a Gaussian distribution; if $\mathcal { D }$ is $L _ { \infty }$ norm, then $p _ { \mathrm { d i s } }$ is an $L _ { \infty }$ -spherical distribution (Iglesias et al., 1998).

## 2.3 PROBABILISTIC ROBUSTNESS

Webb et al. (2018) introduced a statistical framework to assess neural network robustness by evaluating the probability of an attack’s success. Let p denote the probability density function of perturbations around a clean input $x _ { \mathrm { o r i } }$ . To quantify misclassification, let $z ( x ) \overset { \cdot } { \in } \mathbb { R } ^ { | y | }$ denote the pre-softmax logits output by the classifier. For an untargeted attack, the margin function $s ( \cdot )$ is formally defined as the logit margin violation:

$$
s ( x ) = \operatorname* { m a x } _ { y \neq y _ { \mathrm { o r i } } } \left[ z _ { y } ( x ) - z _ { y _ { \mathrm { o r i } } } ( x ) \right] .
$$

The event $s ( X ) \geq 0$ strictly indicates that at least one incorrect class logit matches or exceeds the true class logit, signifying a successful adversarial attack (i.e., the classifier is fooled).

Consequently, the attack success probability, or probabilistic vulnerability, is defined as:

$$
\mathcal { Z } [ p , s ] = \mathbb { P } _ { X \sim p } ( s ( X ) \geq 0 ) = \int \mathbb { I } _ { \{ s ( x ) \geq 0 \} } p ( x ) d x ,
$$

where I is the indicator function.

While directly minimizing ${ \mathcal { T } } [ p , s ]$ reduces vulnerability, framing robustness as an objective to be maximized provides a more consistent narrative for our subsequent theoretical derivations. Therefore, throughout this paper, we formally define Probabilistic Robustness (PR) as the complementary probability of an attack failing:

$$
\operatorname { P R } ( p , s ) : = 1 - { \mathcal { Z } } [ p , s ] = \mathbb { P } _ { X \sim p } ( s ( X ) < 0 ) .
$$

Under this definition, enhancing the intrinsic security of a model corresponds directly to maximizing the value of PR.

## 2.4 ADVERSARIAL TRAINING

Adversarial training (AT) is a widely adopted empirical method to defend against adversarial attack described above (Madry et al., 2018). The core idea of AT is to augment the training data with adversarial examples during the learning process. Formally, let θ denote the parameters of the classifier, and let $p _ { \mathrm { d a t a } }$ be the underlying data distribution. Adversarial training formulates the robustness objective as a min-max optimization problem:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ) \sim p _ { \mathrm { d a t a } } } \left[ \operatorname* { m a x } _ { x _ { \mathrm { a d v } } \in S ( x _ { \mathrm { o r i } } ) } f ( x _ { \mathrm { a d v } } , y _ { \mathrm { o r i } } ; \theta ) \right] ,
$$

where $S ( x _ { \mathrm { o r i } } ) ~ = ~ \{ x _ { \mathrm { a d v } } ~ \in ~ [ 0 , 1 ] ^ { d } ~ : ~ \mathcal { D } ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } ) ~ \leq ~ \epsilon \}$ defines the set of allowed adversarial perturbations bounded by a small budget ϵ (e.g., an $L _ { \infty } \thinspace { \mathrm { o r } } \ L _ { 2 }$ norm ball).

## 3 AN UNTARGETED VERSION OF PROBABILISTIC ADVERSARIAL ATTACK

As introduced in Section 2.2, an untargeted adversarial attack can be formulated as the following optimization problem:

$$
\operatorname* { m i n } _ { x _ { \mathrm { a d v } } \in [ 0 , 1 ] ^ { d } } ~ c _ { 1 } \mathcal { D } \big ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } \big ) ~ - ~ c _ { 2 } f \big ( x _ { \mathrm { a d v } } , y _ { \mathrm { o r i } } \big )
$$

Similar to the work of Zhang et al. (2024a), applying Langevin dynamics to this optimization problem yields convergence to a Gibbs distribution concentrated around the solutions:

$$
p _ { \mathrm { a d v } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ) \propto p _ { \mathrm { v i c } } ( x _ { \mathrm { a d v } } ; y _ { \mathrm { o r i } } ) p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } ) ,
$$

where $p _ { \mathrm { v i c } } ( x _ { \mathrm { a d v } } ; y _ { \mathrm { o r i } } ) \propto \exp \bigl ( c _ { 2 } f ( x _ { \mathrm { a d v } } , y _ { \mathrm { o r i } } ) \bigr )$ is the “victim” distribution and $p _ { \mathrm { d i s } } ( x _ { \mathrm { a d v } } ; x _ { \mathrm { o r i } } ) \propto$ $\exp \bigl ( - c _ { 1 }  { \mathcal { D } } ( x _ { \mathrm { o r i } } , x _ { \mathrm { a d v } } ) \bigr )$ is the “distance” distribution around $x _ { \mathrm { o r i } }$ . For typical choices of $\mathcal { D } _ { : }$ , it is clear that $p _ { \mathrm { d i s } }$ is well-defined (Zhang et al., 2024a). The following proposition shows that $p _ { \mathrm { v i c } }$ is also well-defined in the untargeted setting:

Proposition 1 (Well-definedness of the untargeted victim distribution). Let $K = [ 0 , 1 ] ^ { d }$ and let $f ( \cdot , \bar { y } _ { \mathrm { o r i } } ) : K $ R be continuous. For any $c > 0 ,$ , define

$$
p _ { \mathrm { v i c } } ( x ; y _ { \mathrm { o r i } } ) = { \frac { \exp \left( c f ( x , y _ { \mathrm { o r i } } ) \right) } { \int _ { K } \exp \left( c f ( u , y _ { \mathrm { o r i } } ) \right) d u } } , \qquad x \in K .
$$

Then $p _ { \mathrm { v i c } } ( \cdot ; y _ { \mathrm { o r i } } )$ is a well-defined probability density on K.

The proof is in Appendix B. In this paper, we take D to be the squared $L _ { 2 }$ norm, so that $p _ { \mathrm { d i s } }$ is a Gaussian distribution, and choose $f$ to be the cross-entropy loss. Under this setting, the Langevindynamics-based algorithm for sampling adversarial examples from $p _ { \mathrm { a d v } }$ is presented in Algorithm 1 (See Appendix A).

## 4 A KL-BASED LOWER BOUND ON PROBABILISTIC ROBUSTNESS

Recall that $\mathrm { P R } ( p , s )$ (introduced in Section 2.3) evaluates the model’s security against perturbations drawn from a distribution $p$ around each clean (original) input $x _ { \mathrm { o r i } }$ . We observe that the distance distribution $p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } )$ introduced in Section 2.2 naturally serves as the evaluation distribution $p .$ Consequently, our objective formally becomes maximizing $\mathrm { P R } ( p _ { \mathrm { d i s } } , s )$

To establish a mathematical lower bound on PR, we must connect the attack-success event, characterized by the logit margin violation $s ( x ) \geq 0$ (Section 2.3), with our optimization objective, which relies on the continuous cross-entropy loss $f ( x , y _ { \mathrm { o r i } } )$ . However, establishing a direct, tractable algebraic mapping between the margin $s ( x )$ and the loss $f$ is highly challenging due to softmax non-linearities and complex multi-class dynamics.

To circumvent this, we define the attack-success region $E _ { \mathrm { a d v } } = \{ x \in \mathcal { X } : s ( x ) \geq 0 \}$ as the vulnerable space where the classifier is fooled. We then introduce a connecting scalar $\gamma _ { * }$ representing the infimum of the scaled cross-entropy loss over this region:

$$
\gamma _ { * } : = \operatorname* { i n f } _ { x \in E _ { \mathrm { a d v } } } c _ { 2 } f ( x , y _ { \mathrm { o r i } } ; \theta ) .
$$

Since the cross-entropy loss is inherently non-negative and the temperature scaling factor $c _ { 2 } > 0$ , it strictly holds that $\gamma _ { * } \geq 0$ . By definition, any successful adversarial example $x \in E _ { \mathrm { a d v } }$ rigorously guarantees $c _ { 2 } f ( x , y _ { \mathrm { o r i } } ; \theta ) \ge \gamma _ { * }$ . This property is pivotal, as it allows us to formally link the expected loss under $p _ { \mathrm { d i s } }$ to the probability mass of the vulnerable region, paving the way for the following theorem.

Theorem 2 (KL-Based Lower Bound on Probabilistic Robustness). Let $p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } )$ be a probability density on $[ 0 , 1 ] ^ { d }$ . Suppose the victim distribution parameterized by θ is defined as

$$
p _ { \mathrm { v i c } } ( x ; y _ { \mathrm { o r i } } , \theta ) = \frac { \exp ( c _ { 2 } f ( x , y _ { \mathrm { o r i } } ; \theta ) ) } { Z _ { \mathrm { v i c } } } ,
$$

where $\begin{array} { r } { Z _ { \mathrm { v i c } } = \int _ { \mathcal { X } } \exp ( c _ { 2 } f ( u , y _ { \mathrm { o r i } } ; \theta ) ) } \end{array}$ )du is the normalizing constant. Let $\gamma _ { * }$ be defined as above and assume $\gamma _ { * } > 0 .$ . Then, the probabilistic robustness satisfies:

$$
\mathrm { P R } ( p _ { \mathrm { d i s } } , s ) \geq 1 - \frac { \log Z _ { \mathrm { v i c } } - H ( p _ { \mathrm { d i s } } ) - \mathrm { K L } ( p _ { \mathrm { d i s } } \| p _ { \mathrm { v i c } } ) } { \gamma _ { * } } ,
$$

where $\begin{array} { r } { H ( p _ { \mathrm { d i s } } ) = - \int _ { \mathcal { X } } p _ { \mathrm { d i s } } ( x ) \log p _ { \mathrm { d i s } } ( x ) } \end{array}$ dx is the differential entropy of the distance distribution.

The proof is in Appendix B. Crucially, Theorem 2 formally unifies the probabilistic attack perspective (Zhang et al., 2024a) with the probabilistic robustness framework (Webb et al., 2018). By expressing the lower bound on PR using the statistical divergence between $p _ { \mathrm { d i s } }$ and $p _ { \mathrm { v i c } }$ , this theorem directly translates our intuition of “pushing the distributions apart” into a rigorous mathematical objective.

Notice that once the clean input $x _ { \mathrm { o r i } }$ and the perturbation budget are fixed, the distance distribution $p _ { \mathrm { d i s } }$ and its differential entropy $H ( p _ { \mathrm { d i s } } )$ are constants. Furthermore, since $\gamma _ { * }$ acts as a positive scaling factor, maximizing the lower bound on PR requires maximizing the term $\mathrm { K L } ( \bar { p _ { \mathrm { d i s } } } \Vert p _ { \mathrm { v i c } } ) - \log \bar { Z } _ { \mathrm { v i c } }$ . This fundamental insight serves as the theoretical bedrock for the tractable surrogate objective derived in the next section.

While intuitively one might assume that maximizing $K L ( p _ { d i s } | | p _ { v i c } )$ alone is sufficient to separate the distributions, the $- \log Z _ { v i c }$ term holds crucial geometric and physical significance. We defer a detailed discussion on this to Appendix D.

## 5 PROBABILISTIC ADVERSARIAL TRAINING

Based on Section 4, maximizing the lower bound $K L ( p _ { d i s } | | p _ { v i c } ) - \log Z _ { v i c }$ acts as a surrogate objective that indirectly improves probabilistic robustness. However, calculating the exact gradients of this theoretical bound is challenging due to intractable normalizing constants. In this section, we derive a computable empirical loss function through a structured, three-step derivation. We then explicitly reveal how this empirical formulation recovers a scaled, probabilistic counterpart to standard adversarial training, before formally proving its convergence. Algorithm 2 in Appendix A presents the pseudocode for the final practical implementation of probabilistic adversarial training.

## 5.1 DERIVING THE EMPIRICAL LOSS FUNCTION

Step 1: The Idealized Objective. We first attempt to formulate an objective by directly applying importance sampling to the theoretical lower bound.

Proposition 3 (Idealized Objective). Let $p _ { d i s } , p _ { v i c } ,$ , and $p _ { a d \iota }$ be defined as in Section 3. Maximizing the probabilistic robustness lower bound with respect to the classifier parameters θ is mathemati cally equivalent to minimizing the following idealized expected loss:

$$
\mathcal { I } _ { i d e a l } ( \theta ) = \mathbb { E } _ { X \sim p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta ) } \left[ \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } c _ { 2 } f ( X , y _ { o r i } ; \theta ) \right]
$$

See Appendix B for the proof. While Proposition 3 demonstrates that the intractable log $Z _ { v i c }$ cancels out, directly optimizing $\mathcal { I } _ { i d e a l } ( \theta )$ is highly problematic. The sampling distribution $p _ { a d v } ( X ; x _ { o r i } , y _ { o r i } , \theta ) \propto ^ { \top } p _ { d i s } ( X ; x _ { o r i } ) p _ { v i c } ( X ; y _ { o r i } , \theta )$ depends on the parameters θ. Differentiating through this expectation is notoriously unstable and requires complex score-function estimators.

Step 2: The Gradient-First Approach. To circumvent the challenge of differentiating through a parameterized distribution, we propose computing the gradient of the lower bound before applying importance sampling.

Proposition 4 (Gradient of the Objective). The exact gradient of the negative lower bound with respect to θ can be expressed as an expectation over the adversarial distribution $p _ { a d v } .$

$$
\nabla _ { \theta } \mathcal { I } ( \theta ) = \mathbb { E } _ { X \sim p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta ) } \left[ \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } \nabla _ { \theta } ( c _ { 2 } f ( X , y _ { o r i } ; \theta ) ) \right]
$$

See Appendix B for the proof. This formulation decouples the sampling distribution from the gradient computation. Because the gradient operator $\nabla _ { \theta }$ only applies to the loss term $f ,$ the importance weight $\begin{array} { r } { w ( X ; x _ { o r i } , y _ { o r i } , \theta ) = \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } } \end{array}$ acts purely as a scaling constant for the gradients of individual samples.

Step 3: Self-Normalized Importance Sampling. Although Proposition 4 provides a relatively tractable gradient direction, evaluating the theoretical weight w $( X ; \bar { x } _ { o r i } , y _ { o r i } , \bar { \theta } )$ is still impossible. By substituting $\begin{array} { r } { p _ { v i c } ( X ; y _ { o r i } , \theta ) = \frac { 1 } { Z _ { v i c } } \exp ( c _ { 2 } f ( X , y _ { o r i } ; \bar { \theta } ) ) } \end{array}$ , we reveal the hidden constants:

$$
w ( X ; x _ { o r i } , y _ { o r i } , \theta ) = Z _ { a d v } Z _ { v i c } \exp ( - c _ { 2 } f ( X , y _ { o r i } ; \theta ) )
$$

Both $Z _ { a d v }$ and $Z _ { v i c }$ are intractable to compute. However, for a given clean image $x _ { o r i }$ and a specific set of model parameters θ, their product is strictly a constant across different sampled perturbations.

To circumvent the intractability of these constants, we employ Self-Normalized Importance Sampling (SNIS). Given a mini-batch of adversarial examples $\mathbf { \tilde { \phi } } _ { X } \mathbf { \tilde { ( } 1 ) } , . . . , X ^ { ( B ) }$ sampled from $p _ { \mathrm { a d v } }$ , we normalize the weights across the batch:

$$
\begin{array} { l } { \hat { w } ^ { ( i ) } = \frac { w ( X ^ { ( i ) } ; x _ { o r i } , y _ { o r i } , \theta ) } { \sum _ { j = 1 } ^ { B } w ( X ^ { ( j ) } ; x _ { o r i } , y _ { o r i } , \theta ) } = \frac { Z _ { a d v } Z _ { v i c } \exp ( - c _ { 2 } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) ) } { \sum _ { j = 1 } ^ { B } Z _ { a d v } Z _ { v i c } \exp ( - c _ { 2 } f ( X ^ { ( j ) } , y _ { o r i } ; \theta ) ) } } \\ { = \frac { \exp \left( - c _ { 2 } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) \right) } { \sum _ { j = 1 } ^ { B } \exp \left( - c _ { 2 } f ( X ^ { ( j ) } , y _ { o r i } ; \theta ) \right) } } \end{array}
$$

The intractable terms $Z _ { a d v }$ and $Z _ { v i c }$ cancel out exactly. This self-normalized weighting scheme naturally corresponds to applying a Softmax function. Consequently, the true gradient derived in Proposition 4 can be estimated by backpropagating through the following empirical loss:

$$
\mathcal { L } _ { e m p } ( \theta ) = \sum _ { i = 1 } ^ { B } \hat { w } ^ { ( i ) } f ( X ^ { ( i ) } , y _ { o r i } ; \theta )
$$

where $\hat { w } ^ { ( i ) }$ are treated as fixed constants (i.e., stop-gradient) during optimization, as explicitly implemented in Lines 16-18 of Algorithm 2 (Appendix A).

## 5.2 CONNECTION TO STANDARD ADVERSARIAL TRAINING

Standard adversarial training (AT) methods, such as PGD with random restarts, can be viewed probabilistically as drawing samples from an implicit adversarial distribution $p _ { a d v }$ to minimize the unweighted expected loss $\mathbb { E } _ { X \sim p _ { a d v } ( \cdot | x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ) } [ f ( X , y _ { o r i } ; \theta ) ]$ ]. Empirically, this directly corresponds to optimizing a uniformly weighted batch loss $\begin{array} { r } { \frac { 1 } { B } \sum _ { i = 1 } ^ { B } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) } \end{array}$

In this work, both our idealized objective $\mathcal { I } _ { i d e a l }$ and empirical loss $\mathcal { L } _ { e m p }$ recover this exact AT paradigm, differing solely by a principled scaling factor. Instead of treating all samples equally, Probabilistic Adversarial Training (PAT) optimizes a weighted loss using the self-normalized importance weights $\hat { w } ^ { ( i ) } \propto \exp ( - c _ { 2 } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) )$ .

This negative exponent acts as a direct penalty on extreme adversarial examples. While standard AT strictly focuses on worst-case perturbations, PAT deliberately down-weights extreme outliers. This weighting mechanism prevents the model from over-optimizing for worst-case attacks, thereby improving overall probabilistic robustness.

## 5.3 CONVERGENCE OF THE EMPIRICAL GRADIENT ESTIMATOR

We present a convergence theorem demonstrating that our self-normalized importance sampling (SNIS) estimator is asymptotically consistent.

Theorem 5 (Asymptotic Consistency of the SNIS Gradient Estimator). Let $X ^ { ( 1 ) } , \ldots , X ^ { ( B ) }$ be i.i.d. samples drawn from the adversarial distribution $p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta )$ . Define the unnormalized importance weights as $\tilde { w } ( X ; y _ { o r i } , \theta ) = \exp ( - c _ { 2 } f ( X , y _ { o r i } ; \theta ) )$ . The SNIS empirical gradient estimator is given by:

$$
\hat { g } _ { B } ( \theta ) = \sum _ { i = 1 } ^ { B } \hat { w } ^ { ( i ) } \nabla _ { \theta } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) = \frac { \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) \nabla _ { \theta } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) } { \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) }
$$

Assume the expected gradient under the distance distribution is finite, i.e., $\mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ | \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) | ] ~ < ~ \infty$ . As the batch size $B ~  ~ \infty$ , the estimator ${ \hat { g } } _ { B } ( \theta )$ converges almost surely to the true expected gradient under $p _ { d i s } ( \cdot ; x _ { o r i } )$ :

$$
\begin{array} { r } { \hat { g } _ { B } ( \theta ) \xrightarrow { a . s . } \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) ] } \end{array}
$$

The proof is provided in Appendix B.

## 5.4 BATCH-LEVEL APPROXIMATION AND IMPLICIT REGULARIZATION

While Theorem 5 assumes multiple perturbations per single image $x _ { o r i } ,$ this is computationally expensive. For efficiency, our practical implementation (Algorithm 2 in Appendix A) applies SNIS across a mini-batch of clean images $\{ ( x ^ { ( \bar { i } ) } , y ^ { ( i ) } ) \} \sim p _ { d a t a } \mathrm { . }$ , generating only one perturbation per image. This batch-level formulation alters the weighting scheme. A strict estimator over the joint distribution requires weighting each sample by an intractable local adversarial volume:

$$
\overline { { { w } } } _ { t r u e } ^ { ( i ) } \propto Z _ { a d v } ^ { ( i ) } Z _ { v i c } ^ { ( i ) } \exp ( - c _ { 2 } f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) )
$$

By definition, $Z _ { a d v } ^ { ( i ) } Z _ { v i c } ^ { ( i ) } = \mathbb { E } _ { U \sim p _ { d i s } ( \cdot ; x ^ { ( i ) } ) } [ \exp ( c _ { 2 } f ( U , y ^ { ( i ) } ; \theta ) ) ]$ . For highly vulnerable samples, this large expected volume theoretically compensates for the sharp decay of $\exp ( - c _ { 2 } f )$ . Naively omitting the volume term while retaining a large attack temperature $c _ { 2 }$ leads to catastrophic weight collapse: the weights of crucial hard samples are driven to near zero, forcing the Softmax to collapse onto the easiest samples and defeating the purpose of adversarial training.

To counteract this without calculating the intractable volume, we introduce a small smoothing factor $\beta ( { \bf e . g . } , \beta = 0 . 0 0 1 )$ to replace $c _ { 2 }$ in the final empirical weighting:

$$
\tilde { w } _ { e m p } ^ { ( i ) } = \exp ( - \beta f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) )\tag{2}
$$

This $\beta$ acts as a conceptual temperature scaling that prevents weight collapse for valid hard samples. Concurrently, retaining the negative exponential structure preserves the algorithmic benefit of omitting $Z _ { a d v } ^ { ( i ) } Z _ { v i c } ^ { ( i ) }$ : implicit regularization against robust overfitting. Outliers or mislabeled data, which theoretically possess exploding local volumes, are penalized. This extends our core philosophy of down-weighting extreme worst-case outliers to the global data manifold. The theoretical analysis of this batch-level estimator is provided in Appendix C.

## 6 EXPERIMENTS

Due to space constraints, this section focuses on the main evaluation results. Comprehensive details regarding the experimental settings, datasets, and hyperparameters are deferred to Appendix E.

## 6.1 PROBABILISTIC ROBUSTNESS

Table 1 summarizes the performance of PAT and baselines (PGD (Madry et al., 2018), ALP (Kannan et al., 2018), CLP (Kannan et al., 2018), TRADES (Zhang et al., 2019), MART (Wang et al., 2019) and AT-PR (Zhang et al., 2025)) on CIFAR-10 and CIFAR-100 (Krizhevsky et al., 2009). PAT consistently achieves superior Probabilistic Robustness (PR) across all evaluated architectures (WRN-50-2, ResNet-18) and datasets, with the advantage becoming more pronounced at larger perturbation scales.

While PAT shows a nominal decrease in strictly worst-case robustness (PGD/CW Acc.) compared to unweighted AT, this directly aligns with our theory. Unlike standard AT which over-fits to anomalous worst-case boundaries, PAT’s theoretically derived importance weight naturally penalizes extreme outliers. This optimizes overall distributional security, effectively pushing the distance and victim distributions apart.

## 6.2 ABLATION STUDY

Table 2 isolates the impact of our theoretically derived importance weight through two variants: PAT(WOS) (Langevin sampling without the importance weight) and PGD-COR/FGSM-COR (standard attacks equipped with our theoretical scaling factor).

PAT consistently outperforms PAT(WOS). On CIFAR-100 (WRN-50-2), removing the importance weight drops PR(0.2) from 67.12% to 63.98%. This confirms that sampling from $p _ { a d v }$ alone is insufficient; down-weighting extreme outliers via our theoretical formulation is an essential requirement for maximizing probabilistic robustness.

Our importance weight acts as a versatile plug-in regularizer. Applying it to standard PGD (PGD-COR) improves PR(0.2) on CIFAR-100 from 58.99% to 62.21%. Notably, the performance of

Table 1: Comprehensive evaluation of Probabilistic Robustness (PR) and test accuracies across CIFAR-10 and CIFAR-100 datasets. PR is evaluated under escalating perturbation scales
<table><tr><td>D.</td><td>M.</td><td>Attack</td><td>Acc. %</td><td>PGD %</td><td>CW %</td><td>PR(·, 0.1) %</td><td>PR(·, 0.12) %</td><td>PR(·, 0.15) %</td><td>PR(·, 0.2) %</td></tr><tr><td rowspan="18">CIR-10</td><td>ResN-18</td><td>Clean</td><td>93.75 87.16</td><td>0.00 0.01</td><td>0.00 0.05</td><td>49.82 88.88</td><td>36.62 84.23</td><td>24.19 75.83</td><td>16.36 59.91</td></tr><tr><td>FGSM PGD</td><td>82.00</td><td>45.80</td><td>44.94</td><td>95.26</td><td>92.55</td><td>87.72</td><td>77.03</td></tr><tr><td></td><td>81.24</td><td>46.65</td><td>45.56</td><td>95.36</td><td>92.88</td><td>88.16</td><td></td></tr><tr><td>ALP</td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.94</td></tr><tr><td>CLP</td><td>83.23</td><td>44.11</td><td>43.34</td><td>95.37</td><td>93.03</td><td>88.47</td><td>78.77</td></tr><tr><td>TRADES</td><td>79.62</td><td>48.78</td><td>45.91</td><td>94.33</td><td>91.97</td><td>87.33</td><td>77.33</td></tr><tr><td>MART</td><td>78.56</td><td>51.30</td><td>47.02</td><td>94.26</td><td>91.86</td><td>87.36</td><td>78.29</td></tr><tr><td>AT-PR PAT</td><td>80.33</td><td>40.58 32.53</td><td>39.23 30.51</td><td>95.25</td><td>92.80</td><td>88.10</td><td>78.22</td></tr><tr><td>Clean</td><td>82.44</td><td></td><td></td><td>95.53</td><td>93.43</td><td>89.26</td><td>80.62</td></tr><tr><td>FGSM</td><td>91.89</td><td>0.00</td><td>0.00</td><td>56.99</td><td>44.66</td><td>31.52</td><td>20.85</td></tr><tr><td>PGD</td><td>87.60</td><td>0.00 46.54</td><td>0.00 44.33</td><td>79.79 95.59</td><td>67.51</td><td>44.35</td><td>21.04</td></tr><tr><td>ALP</td><td>76.49 72.45</td><td>44.42</td><td>41.50</td><td>95.71</td><td>93.56</td><td>89.44 89.71</td><td>79.98</td></tr><tr><td>WRNN0-2 CLP</td><td></td><td>37.71</td><td>36.41</td><td></td><td>93.76</td><td></td><td>81.23</td></tr><tr><td></td><td>79.61</td><td></td><td></td><td>95.86</td><td>93.69</td><td>88.94</td><td>78.70</td></tr><tr><td>TRADES MART</td><td>74.03</td><td>44.38</td><td>40.82</td><td>94.72</td><td>92.62</td><td>88.48</td><td>79.89</td></tr><tr><td>AT-PR</td><td>69.87</td><td>46.83 40.03</td><td>42.29 39.21</td><td>94.93 95.47</td><td>92.83</td><td>89.26</td><td>81.93</td></tr><tr><td>PAT</td><td>75.93 78.26</td><td>31.94</td><td>29.91</td><td>95.98</td><td>93.61 93.96</td><td>89.61 90.96</td><td>80.23</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.19</td></tr><tr><td rowspan="13">ResN-18 CIR-0</td><td>Clean FGSM</td><td>74.61</td><td>0.00</td><td>0.00</td><td>27.45</td><td>19.20</td><td>11.93</td><td>7.08</td></tr><tr><td>PGD</td><td>59.56</td><td>0.00</td><td>0.00</td><td>64.50</td><td>53.45</td><td>38.65</td><td>21.62</td></tr><tr><td></td><td>54.50</td><td>22.48</td><td>21.18</td><td>89.27</td><td>83.31</td><td>71.01</td><td>46.90</td></tr><tr><td>ALP</td><td>54.88</td><td>23.83</td><td>21.98</td><td>90.45</td><td>84.69</td><td>72.95</td><td>49.35</td></tr><tr><td>CLP</td><td>55.71</td><td>22.08</td><td>19.28</td><td>90.05</td><td>84.80</td><td>74.50</td><td>54.53</td></tr><tr><td>TRADES</td><td>53.56</td><td>26.74</td><td>22.66</td><td>89.07</td><td>84.23</td><td>74.74</td><td>56.24</td></tr><tr><td>MART</td><td>52.13</td><td>26.88</td><td>23.51</td><td>90.02</td><td>84.86</td><td>74.66</td><td>51.76</td></tr><tr><td>AT-PR</td><td>52.67</td><td>21.48</td><td>20.28</td><td>89.09</td><td>83.14</td><td>71.18</td><td>47.64</td></tr><tr><td>PAT</td><td>53.54</td><td>17.50</td><td>14.87</td><td>90.77</td><td>86.58</td><td>78.07</td><td>59.78</td></tr><tr><td>Clean FGSM</td><td>70.41</td><td>0.00</td><td></td><td>0.00</td><td>25.51</td><td>17.95</td><td>11.23</td><td>6.24</td></tr><tr><td></td><td></td><td>51.04</td><td>0.00</td><td>0.00</td><td>39.19</td><td>25.26</td><td>13.12</td><td>5.26</td></tr><tr><td></td><td>PGD</td><td>49.55</td><td>24.04</td><td>22.00</td><td>92.06</td><td>87.94</td><td>78.85</td><td>58.99</td></tr><tr><td>WRN0-2</td><td>ALP</td><td>44.38</td><td>23.10</td><td>20.80</td><td>92.07</td><td>88.07</td><td>79.92</td><td>62.17</td></tr><tr><td></td><td>CLP</td><td>52.95</td><td>19.15</td><td>16.94</td><td>90.91</td><td>86.36</td><td>76.20</td><td>56.99</td></tr><tr><td></td><td>TRADES</td><td>49.85</td><td>23.01</td><td>19.18</td><td>90.64</td><td>86.46</td><td>78.32</td><td>62.03</td></tr><tr><td></td><td>MART</td><td>42.61</td><td>24.67</td><td>21.15</td><td>92.03</td><td>88.52</td><td>81.41</td><td>66.51</td></tr><tr><td></td><td>AT-PR</td><td>48.47</td><td>22.02</td><td>20.98</td><td>92.11</td><td>88.08</td><td>79.87</td><td>61.58</td></tr><tr><td></td><td>PAT</td><td>49.95</td><td>17.26</td><td>15.98</td><td>92.28</td><td>89.49</td><td>82.18</td><td>67.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

FGSM-COR exhibits more irregularity. This discrepancy is theoretically expected: as a deterministic, single-step attack, FGSM lacks the iterative exploration and stochasticity required for the importance weight to effectively evaluate the local adversarial distribution. Nonetheless, the consistent gains in PGD-COR demonstrate the broader utility of our weighting scheme in improving the distributional robustness of traditional iterative defense methods.

## 7 RELATED WORK

Adversarial Training for Worst-Case Robustness. Traditional adversarial training (AT) formulates a min-max optimization problem to defend against worst-case perturbations (Madry et al., 2018). A vast literature has developed around improving this worst-case robustness (WCR), including regularization techniques like TRADES (Zhang et al., 2019) and MART (Wang et al., 2019), logit pairing methods such as ALP and CLP (Kannan et al., 2018), and efficient training variants like Fast-AT (Wong et al., 2020) and Free-AT (Shafahi et al., 2019). Complementary to our work, Li & Li (2024) theoretically analyzed the feature learning dynamics of AT. However, these WCR-focused methods inherently optimize for extreme, deterministic adversarial outliers, which frequently compromises standard accuracy and broader distributional security (Zhao, 2026).

Probabilistic Robustness Assessment. To bridge the gap between extreme-case vulnerability and practical reliability, Webb et al. (2018) formally defined Probabilistic Robustness (PR) to evaluate the overall likelihood of adversarial examples within a local neighborhood. Following this, significant progress has been made in PR assessment, including scalable statistical estimators (Tit et al.,

Table 2: Ablation results for the importance weight across various architectures. PAT(WOS) removes the weighting factor; COR denotes the application of our scaling factor to standard baselines.
<table><tr><td>D.</td><td>M.</td><td>Attack</td><td>Acc. %</td><td>PGD %</td><td>CW %</td><td>PR(·, 0.1) %</td><td>PR(·, 0.12) %</td><td>PR(·, 0.15) %</td><td>PR(·, 0.2) %</td></tr><tr><td rowspan="6">CIR-10</td><td rowspan="6">ResN-18 WRN--2</td><td>FGSM FGSM-COR</td><td>87.16 89.04</td><td>0.01 0.02</td><td>0.05 0.03</td><td>88.88 89.03</td><td>84.23 84.53</td><td>75.83 75.33</td><td>59.91 56.45</td></tr><tr><td>PGD PGD-COR</td><td>82.00 81.75</td><td>45.80 45.81</td><td>44.94 44.72</td><td>95.26 95.03</td><td>92.55 92.38</td><td>87.72 87.33</td><td>77.03 75.97</td></tr><tr><td>PAT(WOS) PAT</td><td>82.96 82.44</td><td>32.15 32.53</td><td>30.34 30.51</td><td>94.86 95.53</td><td>92.56 93.43</td><td>88.36 89.26</td><td>79.81 80.62</td></tr><tr><td>FGSM FGSM-COR</td><td>87.60 76.46</td><td>0.00 42.63</td><td>0.00 41.01</td><td>79.79 93.42</td><td>67.51 90.26</td><td>44.35 84.74</td><td>21.04 74.40</td></tr><tr><td>PGD PGD-COR</td><td>76.49 73.45</td><td>46.54 45.11</td><td>44.33 42.71</td><td>95.59 95.79</td><td>93.56 93.85</td><td>89.44 89.79</td><td>79.98</td></tr><tr><td>PAT(WOS) PAT</td><td>79.06 78.26</td><td>31.99 31.94</td><td>29.63 29.91</td><td>94.63 95.98</td><td>92.29 93.96</td><td>87.91 90.96</td><td>80.62 79.25 82.19</td></tr><tr><td rowspan="6">ResN-18 CAR-0 WRNN--2</td><td>FGSM FGSM-COR</td><td>59.56 56.20</td><td>0.00 0.00</td><td>0.00 0.01</td><td>64.50 62.68</td><td>53.45 51.06</td><td>38.65 35.49</td><td>21.62 18.52</td></tr><tr><td>PGD PGD-COR</td><td>54.50 55.02</td><td>22.48 23.42</td><td>21.18 22.06</td><td>89.27 90.11</td><td>83.31 84.23</td><td>71.01 72.39</td><td>46.90 47.92</td></tr><tr><td>PAT(WOS) PAT</td><td>53.21 53.54</td><td>16.84 17.50</td><td>14.53 14.87</td><td>90.47 90.77</td><td>86.24 86.58</td><td>78.07 78.07</td><td>59.83 59.78</td></tr><tr><td>FGSM FGSM-COR</td><td>51.04 52.50</td><td>0.00 0.00</td><td>0.00 0.00</td><td>39.19 41.66</td><td>25.26 26.31</td><td>13.12 11.75</td><td>5.26 3.88</td></tr><tr><td>PGD PGD-COR</td><td>49.55 44.87</td><td>24.04 22.74</td><td>22.00 20.46</td><td>92.06 92.66</td><td>87.94 88.79</td><td>78.85 80.55</td><td>58.99</td></tr><tr><td>PAT(WOS) PAT</td><td>50.17 49.95</td><td>16.92 17.26</td><td>14.96 15.98</td><td>91.65 92.28</td><td>87.91 89.49</td><td>80.22 82.18</td><td>62.21 63.98 67.12</td></tr></table>

2021; Karim et al., 2023), probabilistic verification approaches (Weng et al., 2019), and extensions to functional perturbations (Zhang et al., 2022). A comprehensive survey of these evaluation metrics is provided by Zhao (Zhao, 2026). While evaluating PR has become increasingly sophisticated, dedicated training frameworks designed to directly optimize it remain sparse.

Defenses Targeting Probabilistic Robustness. Only a few recent works have explicitly aimed at improving PR during training. Wang et al. (2021) demonstrated that adding random perturbations enhances PR, though it yields near-zero worst-case robustness. Other approaches (Robey et al., 2022; Zhang et al., 2024b) proposed training with risk measures such as Conditional Value-at-Risk (CVaR) or Entropic Value-at-Risk (EVaR), but also struggled to maintain competitive standard adversarial robustness. The most direct comparison to our work is the recent AT-PR method (Zhang et al., 2025), which explicitly alters the inner maximization of AT to target PR. However, their approach relies on a heuristic, gradient-based boundary search to empirically find the “widest peak” in the loss landscape. In contrast, our Probabilistic Adversarial Training (PAT) establishes a rigorous theoretical foundation: we formalize untargeted probabilistic attacks via Langevin Dynamics, prove a KL-based lower bound for PR, and employ Self-Normalized Importance Sampling (SNIS) to derive a fully tractable, principled optimization objective.

## 8 CONCLUSION

In this work, we provided a unified probabilistic perspective on adversarial defense by proving that $K L ( p _ { d i s } | | p _ { v i c } ) - \log Z _ { v i c }$ serves as a lower bound for probabilistic robustness (PR). By maximizing this tractable surrogate objective via SNIS applied to gradient, we introduced Probabilistic Adversarial Training (PAT). Our theoretical framework not only offers a principled probabilistic interpretation of standard adversarial training but also yields a natural importance-weighting mechanism that penalizes extreme worst-case outliers. Empirically, PAT consistently enhances PR across various architectures. We hope this probabilistic perspective of adversarial vulnerabilities inspires more robust and theoretically grounded defense strategies in the future.

## ACKNOWLEDGMENTS

The work presented in this paper has been supported by UKRI Future Leaders Fellowship (Grant MR/S035176/1) and funded by the European Union under the Horizon Europe project AIGGRE-GATE (AI-enhanced collective intelligence for resilient, ethical and user-centric awareness and decision making in CCAM applications, Grant Agreement No. 101202457). Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

## REFERENCES

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), pp. 39–57. Ieee, 2017.

Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6572.

Pilar L Iglesias, Carlos AB Pereira, and Nelson I Tanaka. Characterizations of multivariate spherical distributions in l infinity norm. Test, 7(2):307–324, 1998.

Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial logit pairing. arXiv preprint arXiv:1803.06373, 2018.

TIT Karim, Teddy Furon, and Mathias Rousset. Gradient-informed neural network statistical robustness estimation. In International Conference on Artificial Intelligence and Statistics, pp. 323–334. PMLR, 2023.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial machine learning at scale. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https://openreview. net/forum?id=BJm4T4Kgx.

Andrew Lamperski. Projected stochastic gradient langevin algorithms for constrained sampling and non-convex learning. In Conference on Learning Theory, pp. 2891–2937. PMLR, 2021.

Binghui Li and Yuanzhi Li. Adversarial training can provably improve robustness: Theoretical analysis of feature learning process under structured data. arXiv preprint arXiv:2410.08503, 2024.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/ forum?id=rJzIBfZAb.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2574–2582, 2016.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 1765–1773, 2017.

Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In 2016 IEEE European symposium on security and privacy (EuroS&P), pp. 372–387. IEEE, 2016.

Alexander Robey, Luiz Chamon, George J Pappas, and Hamed Hassani. Probabilistically robust learning: Balancing average and worst-case performance. In International Conference on Machine Learning, pp. 18667–18686. PMLR, 2022.

Ali Shafahi, Mahyar Najibi, Mohammad Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! Advances in neural information processing systems, 32, 2019.

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In Yoshua Bengio and Yann LeCun (eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. URL http: //arxiv.org/abs/1312.6199.

Karim Tit, Teddy Furon, and Mathias Rousset. Efficient statistical assessment of neural network corruption robustness. Advances in Neural Information Processing Systems, 34:9253–9263, 2021.

Benjie Wang, Stefan Webb, and Tom Rainforth. Statistically robust neural network classification. In Uncertainty in Artificial Intelligence, pp. 1735–1745. PMLR, 2021.

Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun Ma, and Quanquan Gu. Improving adversarial robustness requires revisiting misclassified examples. In International conference on learning representations, 2019.

Stefan Webb, Tom Rainforth, Yee Whye Teh, and M Pawan Kumar. A statistical approach to assessing neural network robustness. arXiv preprint arXiv:1811.07209, 2018.

Lily Weng, Pin-Yu Chen, Lam Nguyen, Mark Squillante, Akhilan Boopathy, Ivan Oseledets, and Luca Daniel. Proven: Verifying robustness of neural networks with a probabilistic approach. In International Conference on Machine Learning, pp. 6727–6736. PMLR, 2019.

Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better than free: Revisiting adversarial training. arXiv preprint arXiv:2001.03994, 2020.

Andi Zhang, Mingtian Zhang, and Damon Wischik. Constructing semantics-aware adversarial examples with a probabilistic perspective. Advances in Neural Information Processing Systems, 37: 136259–136285, 2024a.

Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pp. 7472–7482. PMLR, 2019.

Tianle Zhang, Wenjie Ruan, and Jonathan E Fieldsend. Proa: A probabilistic robustness assessment against functional perturbations. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 154–170. Springer, 2022.

Tianle Zhang, Yanghao Zhang, Ronghui Mu, Jiaxu Liu, Jonathan E Fieldsend, and Wenjie Ruan. Prass: Probabilistic risk-averse robust learning with stochastic search. In IJCAI, pp. 559–567, 2024b.

Yi Zhang, Yuhang Chen, Zhen Chen, Wenjie Ruan, Xiaowei Huang, Siddartha Khastgir, and Xingyu Zhao. Adversarial training for probabilistic robustness. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1675–1685, 2025.

Xingyu Zhao. Probabilistic robustness in deep learning: A concise yet comprehensive guide. In Adversarial Example Detection and Mitigation Using Machine Learning, pp. 209–222. Springer, 2026.

## APPENDIX

## A PSEUDO CODE OF THE ALGORITHMS

Algorithm 1 Untargeted Probabilistic Adversarial Attack   
Input: Original image $x _ { \mathrm { o r i } } \in [ 0 , 1 ] ^ { d } ;$ , original class $y _ { \mathrm { o r i } }$ , cross-entropy loss $f$ corresponding to   
the victim classifier, parameters $c _ { 1 } , c _ { 2 } > 0 ,$ step size $\eta ,$ noise scale σ, gradient clipping threshold   
$\rho > 0$ , total timesteps $T .$   
Output: Adversarial example $x _ { \mathrm { a d v } } .$   
$x _ { \mathrm { a d v } }$ ∼ Uniform $[ 0 , 1 ] ^ { d }$ ▷ Initialize   
for $t = 1$ to $T$ do   
$z _ { t } \sim \mathcal { N } ( 0 , I )$ ▷ Generate noise for Langevin Dynamics   
$x _ { \mathrm { a d v } } \gets \Pi _ { [ 0 , 1 ] ^ { d } } \left( x _ { \mathrm { a d v } } + \sigma z _ { t } \right)$ ▷ Apply noise   
${ \mathcal { E } } \gets c _ { 1 } \left\| { \dot { x _ { \mathrm { a d v } } } } ^ { - } - x _ { \mathrm { o r i } } \right\| _ { 2 } ^ { 2 } - c _ { 2 } f ( x _ { \mathrm { a d v } } , y _ { \mathrm { o r i } } )$ ▷ Calculate energy   
$v _ { t } \gets \mathrm { c l i p } ( \nabla _ { x _ { \mathrm { a d v } } } \mathcal { E } , - \rho , \rho )$ ▷ Compute and clip gradient   
$x _ { \mathrm { a d v } } \gets \Pi _ { [ 0 , 1 ] ^ { d } } \left( x _ { \mathrm { a d v } } - \eta v _ { t } \right)$ ▷ update $x _ { \mathrm { a d v } }$   
end for   
return $x _ { \mathrm { a d v } } .$

Algorithm 2 Probabilistic Adversarial Training   
1: Input: Training dataset ${ \overline { { \mathcal { D } } } } ,$ initial model parameters $\theta ,$ cross-entropy loss $f ,$ learning rate $\alpha .$   
2: Hyperparameters: Total epochs $E _ { \mathrm { t o t a l } } .$ , batch size $B ,$ inner steps $\dot { T } ,$ , step size $\eta ,$ noise scale $\sigma ,$   
gradient clip $\rho ,$ energy weights $c _ { 1 } , c _ { 2 } ,$ smoothing factor $\beta .$   
3: Output: Robust model parameters $\theta .$   
4: for epoch $e = 0$ to $E _ { \mathrm { t o t a l } } - 1$ do   
5: for each minibatch $( \mathbf { X } , \mathbf { Y } ) \sim \mathcal { D }$ with size $B$ do   
6: ▷ Phase 1: Probabilistic Adversarial Example Generation   
7: $\mathbf { X } _ { \mathrm { a d v } } \sim \mathbf { U n i f o r m } [ 0 , 1 ] ^ { B \times d }$ ▷ Initialize   
8: for $t = 1$ to $T$ do   
9: $\mathbf { Z } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ^ { B \times d }$ ▷ Sample noise for Langevin Dynamics   
10: $\mathbf { X } _ { \mathrm { a d v } } \gets \dot { \Pi } _ { [ 0 , 1 ] ^ { B \times d } } \left( \mathbf { X } _ { \mathrm { a d v } } + \sigma \mathbf { Z } _ { t } \right)$ ▷ Apply noise   
11: $\begin{array} { r } { \mathcal { E }  \sum _ { i = 1 } ^ { B } \Big ( c _ { 1 } \| \mathbf { X } _ { \mathrm { a d v } } ^ { ( i ) } - \mathbf { X } ^ { ( i ) } \| _ { 2 } ^ { 2 } - c _ { 2 } f _ { \theta } ( \mathbf { X } _ { \mathrm { a d v } } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } ) \Big ) } \end{array}$ ▷ Calculate batch energy   
12: $\mathbf { V } _ { t } \gets \mathrm { c l i p } ( \dot { \nabla } _ { \mathbf { X } _ { \mathrm { a d v } } } \mathcal { E } , - \rho , \rho )$ ▷ Compute and clip gradient   
13: $\mathbf { X } _ { \mathrm { a d v } }  \tilde { \Pi } _ { [ 0 , 1 ] ^ { B \times d } } ( \mathbf { X } _ { \mathrm { a d v } } - \eta \mathbf { V } _ { t } )$ ▷ Update $\mathbf { X } _ { \mathrm { a d v } }$   
14: end for   
15: ▷ Phase 2: Model Parameter Update via Gradient Descent   
16: $\ell _ { i } \gets f _ { \theta } ( \mathbf { X } _ { \mathrm { a d v } } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } )$ ▷ Compute sample losses $\boldsymbol { \ell } \in \mathbb { R } ^ { B }$   
17: $\tilde { \mathbf { w } } _ { \mathrm { p v i c } }  \mathrm { D e t a c h } ( - \beta \ell )$ ▷ Apply smoothing factor and stop gradient   
18: $\hat { \mathbf { w } } _ { \mathrm { p v i c } } \gets \mathrm { S o f t m a x } ( \tilde { \mathbf { w } } _ { \mathrm { p v i c } } )$ ▷ Softmax over batch dimension   
19: $\mathcal { L }  \ell ^ { \top } \hat { \mathbf { w } } _ { \mathrm { p v i c } }$ ▷ Final weighted loss   
20: $\theta  \theta - \dot { \alpha } \dot { \nabla } _ { \theta } \mathcal { L }$ ▷ Update model parameters   
21: end for   
22: end for   
23: return $\theta .$

## B PROOFS OF THEOREMS IN THE MAIN TEXT

Proposition 1 (Well-definedness of the untargeted victim distribution). Let $K = [ 0 , 1 ] ^ { d }$ and let $f ( \cdot , \overset { - } { y _ { \mathrm { o r i } } } ) : K $ R be continuous. For any $c > 0 ,$ , define

$$
p _ { \mathrm { v i c } } ( x ; y _ { \mathrm { o r i } } ) = \frac { \exp \left( c f ( x , y _ { \mathrm { o r i } } ) \right) } { \int _ { K } \exp \left( c f ( u , y _ { \mathrm { o r i } } ) \right) d u } , \qquad x \in K .
$$

Then $p _ { \mathrm { v i c } } ( \cdot ; y _ { \mathrm { o r i } } )$ is a well-defined probability density on $K .$

Proof. Since $K = [ 0 , 1 ] ^ { d }$ is compact and $f ( \cdot , y _ { \mathrm { o r i } } )$ is continuous, $\exp ( c f ( \cdot , y _ { \mathrm { o r i } } ) )$ is also continuous on $K$ . Hence it is bounded and measurable. Moreover, it is strictly positive on K, so the normalizing constant

$$
Z _ { \mathrm { v i c } } = \int _ { K } \exp \left( c f ( u , y _ { \mathrm { o r i } } ) \right) d u
$$

satisfies $0 < Z _ { \mathrm { v i c } } < \infty$ . Therefore $p _ { \mathrm { v i c } }$ is non-negative, measurable, and integrates to one over $K$ Thus it is a well-defined probability density. □

Theorem 2 (KL-Based Lower Bound on Probabilistic Robustness). Let $p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } )$ be a probability density on $[ 0 , 1 ] ^ { d }$ . Suppose the victim distribution parameterized by θ is defined as

$$
p _ { \mathrm { v i c } } ( x ; y _ { \mathrm { o r i } } , \theta ) = \frac { \exp ( c _ { 2 } f ( x , y _ { \mathrm { o r i } } ; \theta ) ) } { Z _ { \mathrm { v i c } } } ,
$$

where $\begin{array} { r } { Z _ { \mathrm { v i c } } = \int _ { \mathcal { X } } \exp ( c _ { 2 } f ( u , y _ { \mathrm { o r i } } ; \theta ) ) } \end{array}$ )du is the normalizing constant. Let $\gamma _ { * }$ be defined as above and assume $\gamma _ { * } > 0 .$ . Then, the probabilistic robustness satisfies:

$$
\mathrm { P R } ( p _ { \mathrm { d i s } } , s ) \geq 1 - \frac { \log Z _ { \mathrm { v i c } } - H ( p _ { \mathrm { d i s } } ) - \mathrm { K L } ( p _ { \mathrm { d i s } } \| p _ { \mathrm { v i c } } ) } { \gamma _ { * } } ,
$$

where $\begin{array} { r } { H ( p _ { \mathrm { d i s } } ) = - \int _ { \mathcal { X } } p _ { \mathrm { d i s } } ( x ) \log p _ { \mathrm { d i s } } ( x ) } \end{array}$ dx is the differential entropy of the distance distribution.

Proof. By the definition of $\gamma _ { * }$ , for al $x \in E _ { a d v }$ , we have $c _ { 2 } f ( x , y _ { o r i } ) \geq \gamma _ { * }$ . Thus, we can lower bound the expected loss under $p _ { d i s } \colon$

$$
\mathbb { E } _ { X \sim p _ { d i s } } [ c _ { 2 } f ( X , y _ { o r i } ) ] \geq \int _ { E _ { a d v } } p _ { d i s } ( x ) c _ { 2 } f ( x , y _ { o r i } ) d x \geq \gamma _ { * } \int _ { E _ { a d v } } p _ { d i s } ( x ) d x = \gamma _ { * } \mathcal { T } [ p _ { d i s } , s ]
$$

On the other hand, expanding the KL divergence with $\begin{array} { r } { p _ { v i c } ( x ; y _ { o r i } ) = \frac { \exp ( c _ { 2 } f ( x , y _ { o r i } ) ) } { Z _ { v i c } } } \end{array}$ yields:

$$
\begin{array} { r l } {  { K L ( p _ { d i s } | | p _ { v i c } ) = \int p _ { d i s } ( x ) \log p _ { d i s } ( x ) d x - \int p _ { d i s } ( x ) \log p _ { v i c } ( x ; y _ { o r i } ) d x } } \\ & { = - H ( p _ { d i s } ) - \mathbb { E } _ { X \sim p _ { d i s } } [ c _ { 2 } f ( X , y _ { o r i } ) - \log Z _ { v i c } ] } \\ & { = - H ( p _ { d i s } ) + \log Z _ { v i c } - \mathbb { E } _ { X \sim p _ { d i s } } [ c _ { 2 } f ( X , y _ { o r i } ) ] } \\ & { \le - H ( p _ { d i s } ) + \log Z _ { v i c } - \gamma _ { * } \mathbb { Z } [ p _ { d i s } , s ] } \end{array}
$$

Rearranging the inequality gives:

$$
\mathcal { T } [ p _ { d i s } , s ] \leq \frac { \log Z _ { v i c } - H ( p _ { d i s } ) - K L ( p _ { d i s } | | p _ { v i c } ) } { \gamma _ { * } }
$$

Finally, by the definition of probabilistic robustness $P R ( p _ { d i s } , s ) = 1 - \mathcal { T } [ p _ { d i s } , s ]$ , we conclude:

$$
P R ( p _ { d i s } , s ) \geq 1 - \frac { \log Z _ { v i c } - H ( p _ { d i s } ) - K L ( p _ { d i s } | | p _ { v i c } ) } { \gamma _ { * } }
$$

Proposition 3 (Idealized Objective). Let $p _ { d i s } , p _ { v i c } ,$ , and $p _ { a d v }$ be defined as in Section 3. Maximizing the probabilistic robustness lower bound with respect to the classifier parameters θ is mathemati cally equivalent to minimizing the following idealized expected loss:

$$
\mathcal { I } _ { i d e a l } ( \theta ) = \mathbb { E } _ { X \sim p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta ) } \left[ \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } c _ { 2 } f ( X , y _ { o r i } ; \theta ) \right]
$$

Proof. By expanding the KL divergence and changing the measure from $p _ { d i s } \ \mathrm { t o } p _ { a d v }$ , we have:

$$
\begin{array} { r l } &  \begin{array} { r l } & { E \{ \mathcal { L } \phi _ { \mathrm { s c } } ( \cdot ; x _ { \mathrm { o r d } } ) | \mathcal { P } _ { 0 d } \} ( \cdot | y _ { \mathrm { o r d } } , \cdot | y _ { \mathrm { o r d } } , \cdot | \beta ) ] = \mathrm { l o g } Z _ { \mathrm { s c } } } \\ & { = \mathrm { E } _ { x  \mathrm { p e r } _ { \mathrm { t } } ( \cdot , x _ { \mathrm { o r d } } ) } [ \mathrm { l o g } \mathrm { p } _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { i n c } } , \cdot | \beta ) - \int \int \phi _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { i n c } } | , \cdot | x _ { \mathrm { o r d } } ) \log \mathrm { p } _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { i n c } } , \cdot | \beta ) d \alpha - \mathrm { l o g } Z _ { \mathrm { s c } } } \\ & { = - H ( p _ { \mathrm { a b } } ) - \int \int p _ { \mathrm { a b } } ( x _ { \mathrm { i n c } } ^ { \cdot } ; x _ { \mathrm { o r d } } , \cdot | y _ { \mathrm { o r d } } , \cdot | \beta ) \frac { \int _ { \mathcal { L } \phi _ { \mathrm { s c } } } \int _ { \mathcal { L } \phi _ { \mathrm { o r d } } } \cdot \sum _ { \mathcal { R } \ne \mathcal { I } } \biggr | \beta | \mathcal { P } _ { 0 d } \mathrm { p } _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { o r d } } , \cdot | \beta ) } { \int _ { \mathcal { L } \phi _ { \mathrm { s c } } } \int _ { \mathcal { L } \phi _ { \mathrm { o r d } } } \cdot \sum _ { \mathcal { R } \ne \mathcal { I } } \biggr | \beta | \mathcal { P } _ { 0 d } \mathrm { p } _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { o r d } } , \cdot | \beta ) } ] \mathrm { l o g } W _ { \mathrm { s c } } ( \cdot | x _ { \mathrm { o r d } } , \cdot | \beta ) d \alpha - \mathrm { l o g } Z _ { \mathrm { s c } } , } \end \end{array} \end{array}
$$

Since $H ( p _ { d i s } )$ is constant with respect to $\theta ,$ maximizing the lower bound is equivalent to minimizing $\mathcal { I } _ { i d e a l } ( \theta )$ □

Proposition 4 (Gradient of the Objective). The exact gradient of the negative lower bound with respect to θ can be expressed as an expectation over the adversarial distribution $p _ { a d v } .$

$$
\nabla _ { \theta } \mathcal { I } ( \theta ) = \mathbb { E } _ { X \sim p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta ) } \left[ \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } \nabla _ { \theta } ( c _ { 2 } f ( X , y _ { o r i } ; \theta ) ) \right]
$$

Proof. We return to the negative lower bound before importance sampling:

$$
- K L \big ( p _ { d i s } | | p _ { v i c } \big ) + \log Z _ { v i c } = H \big ( p _ { d i s } \big ) + \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } \big [ c _ { 2 } f \big ( X , y _ { o r i } ; \theta \big ) \big ] .
$$

Since $p _ { d i s } ( \cdot ; x _ { o r i } )$ depends only on the clean image $x _ { o r i }$ and is entirely independent of $\theta ,$ the gradient operator can pass directly inside the expectation:

$$
\begin{array} { l } { { \displaystyle \nabla _ { \theta } \left( \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ c _ { 2 } f ( X , y _ { o r i } ; \theta ) ] \right) } } \\ { { \displaystyle = \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ \nabla _ { \theta } c _ { 2 } f ( X , y _ { o r i } ; \theta ) ] } } \\ { ~ = \int p _ { a d v } ( x ; x _ { o r i } , y _ { o r i } , \theta ) \frac { p _ { d i s } ( x ; x _ { o r i } ) } { p _ { a d v } ( x ; x _ { o r i } , y _ { o r i } , \theta ) } \nabla _ { \theta } c _ { 2 } f ( x , y _ { o r i } ; \theta ) d x } \\ { { \displaystyle = \mathbb { E } _ { X \sim p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta ) } \left[ \frac { Z _ { a d v } } { p _ { v i c } ( X ; y _ { o r i } , \theta ) } \nabla _ { \theta } c _ { 2 } f ( X , y _ { o r i } ; \theta ) \right] } } \end{array}
$$

Theorem 5 (Asymptotic Consistency of the SNIS Gradient Estimator). Let $X ^ { ( 1 ) } , \ldots , X ^ { ( B ) }$ be $i . i . d .$ samples drawn from the adversarial distribution $p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta )$ . Define the unnormalized importance weights as $\tilde { w } ( X ; y _ { o r i } , \theta ) = \exp ( - c _ { 2 } f ( X , y _ { o r i } ; \theta ) )$ . The SNIS empirical gradient estima tor is given by:

$$
\hat { g } _ { B } ( \theta ) = \sum _ { i = 1 } ^ { B } \hat { w } ^ { ( i ) } \nabla _ { \theta } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) = \frac { \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) \nabla _ { \theta } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) } { \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) }
$$

Assume the expected gradient under the distance distribution is finite, $i . e .$ $\mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ | \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) | ] ~ < ~ \infty$ . As the batch size $B ~  ~ \infty$ , the estimator ${ \hat { g } } _ { B } ( \theta )$ converges almost surely to the true expected gradient under $p _ { d i s } ( \cdot ; x _ { o r i } )$

$$
\begin{array} { r } { \hat { g } _ { B } ( \theta ) \xrightarrow { a . s . } \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } [ \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) ] } \end{array}
$$

Proof. By the Strong Law of Large Numbers (SLLN), as $B  \infty$ , the sample averages in the numerator and denominator of ${ \hat { g } } _ { B } ( \theta )$ converge almost surely to their respective expectations under $p _ { a d v } ( \cdot ; x _ { o r i } , y _ { o r i } , \theta )$

$$
\begin{array} { c } { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) \nabla _ { \theta } f ( X ^ { ( i ) } , y _ { o r i } ; \theta ) \xrightarrow { a . s . } \mathbb { E } _ { X \sim p _ { a d v } } [ \tilde { w } ( X ; y _ { o r i } , \theta ) \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) ] } \\ { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \tilde { w } ( X ^ { ( i ) } ; y _ { o r i } , \theta ) \xrightarrow { a . s . } \mathbb { E } _ { X \sim p _ { a d v } } [ \tilde { w } ( X ; y _ { o r i } , \theta ) ] } \end{array}
$$

Recall the definitions from Section 3. We can express the product of the sampling density and the unnormalized weight as:

$$
\begin{array} { r l } & { p _ { a d v } ( X ; x _ { o r i } , y _ { o r i } , \theta ) \tilde { w } ( X ; y _ { o r i } , \theta ) = \frac { p _ { d i s } ( X ; x _ { o r i } ) \exp ( c _ { 2 } f ( X , y _ { o r i } ; \theta ) ) } { Z _ { a d v } Z _ { v i c } } \exp ( - c _ { 2 } f ( X , y _ { o r i } ; \theta ) ) } \\ & { \qquad = \frac { p _ { d i s } ( X ; x _ { o r i } ) } { Z _ { a d v } Z _ { v i c } } } \end{array}
$$

Substituting this identity into the expectation of the numerator yields:

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { X \sim p _ { a d v } } [ \tilde { w } ( X ; y _ { o r i } , \theta ) \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) ] = \displaystyle \int \frac { p _ { d i s } ( x ; x _ { o r i } ) } { Z _ { a d v } Z _ { v i c } } \nabla _ { \theta } f ( x , y _ { o r i } ; \theta ) d x } \\ & { } & { = \displaystyle \frac { 1 } { Z _ { a d v } Z _ { v i c } } \mathbb { E } _ { X \sim p _ { d i s } } \left[ \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) \right] } \end{array}
$$

Applying the same substitution to the denominator:

$$
\mathbb { E } _ { X \sim p _ { a d v } } \left[ \tilde { w } ( X ; y _ { o r i } , \theta ) \right] = \int \frac { p _ { d i s } ( x ; x _ { o r i } ) } { Z _ { a d v } Z _ { v i c } } d x = \frac { 1 } { Z _ { a d v } Z _ { v i c } }
$$

By the Continuous Mapping Theorem (or Slutsky’s Theorem), the limit of the ratio is the ratio of their limits. Notice that the intractable constant term perfectly cancels out:

$$
\begin{array} { r } { \widehat { g } _ { B } ( \theta ) \xrightarrow { a . s . } \frac { \frac { 1 } { Z _ { a d v } Z _ { v i c } } \mathbb { E } _ { X \sim p _ { d i s } } \left[ \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) \right] } { \frac { 1 } { Z _ { a d v } Z _ { v i c } } } = \mathbb { E } _ { X \sim p _ { d i s } ( \cdot ; x _ { o r i } ) } \left[ \nabla _ { \theta } f ( X , y _ { o r i } ; \theta ) \right] } \end{array}
$$

This confirms that the self-normalized gradient is asymptotically unbiased and converges to the true gradient of our theoretical objective. □

## C THEORETICAL ANALYSIS OF THE BATCH-LEVEL ESTIMATOR

Theorem $5$ establishes consistency of the SNIS gradient estimator in an idealized regime: a single clean image $x _ { \mathrm { o r i } }$ with label $y _ { \mathrm { o r i } } , \mathrm { ~ a ~ }$ batch of B perturbations from its adversarial distribution $p _ { \mathrm { a d v } } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta )$ . Algorithm 2 differs in two respects: it draws a mini-batch of distinct clean images with one perturbation each, and it uses the smoothing factor $\beta$ in place of $c _ { 2 }$ in the empirical weights $\left( \operatorname { E q . } \left( 2 \right) \right)$ . Sec. 5.4 observed that restoring exactness of the joint estimator would require the intractable volume weights $w _ { \mathrm { t r u e } } ^ { ( i ) } .$ This section analyses the estimator Algorithm 2 actually computes and proves:

• it is consistent, with an explicitly characterized limit (Theorem 7);

• the limit is the exact gradient of an entropic-risk objective under a vulnerability-based reweighting of the data (Proposition 8, Corollary 9);

• this objective still lower-bounds probabilistic robustness for every admissible $\beta ,$ , with Theorem 2 recovered as the tightest member of the resulting family (Proposition 10, Corollary 11);

Setup. Let $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . For each $y \in \mathcal { V }$ , the loss $f ( \cdot , y ; \theta )$ is continuous on the compact set $x ,$ , as in Proposition 1, and

$$
0 \leq f ( u , y ; \theta ) \leq F _ { \operatorname* { m a x } } ( \theta ) : = \operatorname* { m a x } _ { y \in \mathcal { Y } } \operatorname* { m a x } _ { u \in \mathcal { X } } f ( u , y ; \theta ) < \infty .
$$

For a clean pair $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ and $t \geq 0 ,$ let

$$
M _ { t } ( \boldsymbol { x } , \boldsymbol { y } ; \boldsymbol { \theta } ) : = \mathbb { E } _ { U \sim p _ { \mathrm { d i s } } ( \cdot ; \boldsymbol { x } ) } \left[ \exp \bigl ( t f ( U , \boldsymbol { y } ; \boldsymbol { \theta } ) \bigr ) \right] .
$$

With $\beta \in [ 0 , c _ { 2 } ]$ and $\lambda : = c _ { 2 } - \beta .$ , define the tilted distribution, volume ratio, and reweighted data distribution:

$$
\begin{array} { c } { { p _ { \lambda } ( u ; x , y , \theta ) : = \frac { p _ { \mathrm { d i s } } ( u ; x ) \exp \left( \lambda f ( u , y ; \theta ) \right) } { M _ { \lambda } ( x , y ; \theta ) } , } } \\ { { r ( x , y ; \theta ) : = \frac { M _ { \lambda } ( x , y ; \theta ) } { M _ { c _ { 2 } } ( x , y ; \theta ) } , } } \\ { { \tilde { p } ( x , y ; \theta ) \propto p _ { \mathrm { d a t a } } ( x , y ) r ( x , y ; \theta ) . } } \end{array}
$$

The last is well-defined since

$$
\mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) \right] \in \left[ \exp \bigl ( - \beta F _ { \mathrm { m a x } } ( \theta ) \bigr ) , 1 \right]
$$

by Lemma 6 (iii). Define the entropic risk

$$
R _ { \lambda } ( x , y ; \theta ) : = \frac { 1 } { \lambda } \log M _ { \lambda } ( x , y ; \theta ) , \qquad \lambda > 0 ,
$$

and

$$
R _ { 0 } ( x , y ; \theta ) : = \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x ) } \left[ f ( X , y ; \theta ) \right] .
$$

The map $\lambda \mapsto R _ { \lambda } ( x , y ; \theta )$ is continuous at 0. As in Theorem 5, the Langevin sampler is treated as an exact sampler of $p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta )$ (non-asymptotic guarantees for projected Langevin dynamics); all statements are at a fixed θ.

Lemma 6 (Tilting and volume identities). Fix $( x , y ) \in \mathcal { X } \times \mathcal { Y } , \theta _ { : }$ , and $\beta \in [ 0 , c _ { 2 } ]$ , and let $\lambda = c _ { 2 } - \beta .$ Then:

(i) For every $t \geq 0 ,$

$$
1 \leq M _ { t } ( x , y ; \theta ) \leq \exp \bigl ( t F _ { \operatorname* { m a x } } ( \theta ) \bigr ) ,
$$

and $p _ { t } ( \cdot ; x , y , \theta )$ is a valid probability density. In particular,

$$
p _ { 0 } ( \cdot ; x , y , \theta ) = p _ { \mathrm { d i s } } ( \cdot ; x ) , \qquad p _ { c 2 } ( \cdot ; x , y , \theta ) = p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) ,
$$

with

$$
Z _ { \mathrm { a d v } } Z _ { \mathrm { v i c } } = M _ { c _ { 2 } } ( x , y ; \theta ) ,
$$

which is the identity used in Section 5.4.

(ii) Pointwise on $x ,$

$$
p _ { \mathrm { a d v } } ( u ; x , y , \theta ) \exp \bigl ( - \beta f ( u , y ; \theta ) \bigr ) = r ( x , y ; \theta ) p _ { \lambda } ( u ; x , y , \theta ) .
$$

(iii)

$$
\begin{array} { r } { \exp \bigl ( - \beta F _ { \operatorname* { m a x } } ( \theta ) \bigr ) \leq r ( x , y ; \theta ) \leq 1 , } \end{array}
$$

and

$$
r ( x , y ; \theta ) ^ { - 1 } = \mathbb { E } _ { U \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \exp \bigl ( \beta f ( U , y ; \theta ) \bigr ) \right] .
$$

Proof. (i) The function $\exp \bigl ( t f ( \cdot , y ; \theta ) \bigr )$ is continuous on the compact set $x ,$ , with values in $[ 1 , \exp ( t F _ { \operatorname* { m a x } } ( \theta ) ) ]$ . Hence

$$
1 \leq M _ { t } ( x , y ; \theta ) \leq \exp \bigl ( t F _ { \operatorname* { m a x } } ( \theta ) \bigr ) ,
$$

and $p _ { t } ( \cdot ; x , y , \theta )$ integrates to one. The case $t = 0$ gives

$$
p _ { 0 } ( \cdot ; x , y , \theta ) = p _ { \mathrm { d i s } } ( \cdot ; x ) .
$$

Since

$$
p _ { \mathrm { a d v } } ( u ; x , y , \theta ) = \frac { p _ { \mathrm { d i s } } ( u ; x ) \exp \bigl ( c _ { 2 } f ( u , y ; \theta ) \bigr ) } { Z _ { \mathrm { a d v } } Z _ { \mathrm { v i c } } } ,
$$

integrating over $\mathcal { X }$ gives

$$
Z _ { \mathrm { a d v } } Z _ { \mathrm { v i c } } = M _ { c _ { 2 } } ( x , y ; \theta ) ,
$$

and therefore

$$
p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) = p _ { c _ { 2 } } ( \cdot ; x , y , \theta ) .
$$

(ii) Using (i) and $\lambda = c _ { 2 } - \beta$

$$
\begin{array} { r l } & { p _ { \mathrm { a d v } } \bigl ( u ; x , y , \theta \bigr ) \exp \bigl ( - \beta f ( u , y ; \theta ) \bigr ) } \\ & { \qquad = \frac { p _ { \mathrm { d i s } } \bigl ( u ; x \bigr ) \exp \bigl ( \lambda f ( u , y ; \theta ) \bigr ) } { M _ { c 2 } ( x , y ; \theta ) } } \\ & { \qquad = \frac { M _ { \lambda } \bigl ( x , y ; \theta \bigr ) } { M _ { c 2 } ( x , y ; \theta ) } \cdot \frac { p _ { \mathrm { d i s } } \bigl ( u ; x \bigr ) \exp \bigl ( \lambda f ( u , y ; \theta ) \bigr ) } { M _ { \lambda } \bigl ( x , y ; \theta \bigr ) } } \\ & { \qquad = r ( x , y ; \theta ) p _ { \lambda } \bigl ( u ; x , y , \theta \bigr ) . } \end{array}
$$

(iii) Since $f ( u , y ; \theta ) \ge 0$ and $\lambda \leq c _ { 2 }$

$$
\exp \bigl ( \lambda f ( u , y ; \theta ) \bigr ) \leq \exp \bigl ( c _ { 2 } f ( u , y ; \theta ) \bigr ) .
$$

Thus

$$
M _ { \lambda } ( x , y ; \theta ) \leq M _ { c _ { 2 } } ( x , y ; \theta ) , \qquad r ( x , y ; \theta ) \leq 1 .
$$

Conversely,

$$
\begin{array} { r l } & { M _ { { c } _ { 2 } } ( x , y ; \theta ) = \mathbb { E } _ { U \sim p _ { \mathrm { d i s } } ( \cdot ; x ) } \left[ \exp \left( \lambda f ( U , y ; \theta ) \right) \exp \left( \beta f ( U , y ; \theta ) \right) \right] } \\ & { \qquad \le \exp \left( \beta F _ { \operatorname* { m a x } } ( \theta ) \right) M _ { \lambda } ( x , y ; \theta ) , } \end{array}
$$

so

Finally,

$$
r ( x , y ; \theta ) \geq \exp \bigl ( - \beta F _ { \mathrm { m a x } } ( \theta ) \bigr ) .
$$

$$
\mathbb { E } _ { U \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \exp \bigl ( \beta f ( U , y ; \theta ) \bigr ) \right] = \frac { M _ { c _ { 2 } } ( x , y ; \theta ) } { M _ { \lambda } ( x , y ; \theta ) } = r ( x , y ; \theta ) ^ { - 1 } .
$$

Part (iii) makes the mechanism transparent: the local adversarial volume $Z _ { \mathrm { a d v } } Z _ { \mathrm { v i c } } = M _ { c _ { 2 } } ( x , y ; \theta )$ omitted from the empirical weights resurfaces in the large-batch limit as the per-image factor

$$
r ( x , y ; \theta ) = \frac { M _ { \lambda } ( x , y ; \theta ) } { M _ { c _ { 2 } } ( x , y ; \theta ) } ,
$$

which down-weights a clean pair exactly according to the exponential moment of its loss over its perturbation neighborhood. This is the precise form of the implicit regularization described informally in Section 5.4.

Theorem 7 (Batch-level consistency of the empirical gradient estimator). Let

$$
( x ^ { ( i ) } , y ^ { ( i ) } ) \stackrel { \mathrm { i . i . d . } } { \sim } p _ { \mathrm { d a t a } } , \qquad i = 1 , \ldots , B ,
$$

and, conditionally on these clean pairs, let

$$
X ^ { ( i ) } \sim p _ { \mathrm { a d v } } \left( \cdot ; x ^ { ( i ) } , y ^ { ( i ) } , \theta \right)
$$

independently across i. Equivalently, the tuples $( \boldsymbol { x } ^ { ( i ) } , \boldsymbol { y } ^ { ( i ) } , \boldsymbol { X } ^ { ( i ) } )$ are i.i.d. from the joint density

$$
\pi ( x , y , u ) : = p _ { \mathrm { d a t a } } ( x , y ) p _ { \mathrm { a d v } } ( u ; x , y , \theta ) .
$$

Let

$$
\hat { g } _ { B } ^ { \mathrm { e m p } } ( \theta ) = \sum _ { i = 1 } ^ { B } \hat { w } _ { \mathrm { e m p } } ^ { ( i ) } \nabla _ { \theta } f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta )
$$

be the update ofLines 16-19 ofAlgorithm 2, with self-normalized weights

$$
\hat { w } _ { \mathrm { e m p } } ^ { ( i ) } = \frac { \tilde { w } _ { \mathrm { e m p } } ^ { ( i ) } } { \sum _ { j = 1 } ^ { B } \tilde { w } _ { \mathrm { e m p } } ^ { ( j ) } } , \qquad \tilde { w } _ { \mathrm { e m p } } ^ { ( i ) } : = \exp \bigl ( - \beta f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) \bigr ) ,
$$

treated as constants during backpropagation (stop-gradient, Line 17). If

$$
\begin{array} { r } { \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \mathbb { E } _ { X \sim p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) } \left[ \lVert \nabla _ { \theta } f ( X , y ; \theta ) \rVert \right] \right] < \infty , } \end{array}
$$

then, as $B  \infty$

$$
\begin{array} { r } { \hat { g } _ { B } ^ { \mathrm { e m p } } ( \theta ) \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { ( x , y ) \sim \tilde { p } ( \cdot , \cdot ; \theta ) } \left[ \mathbb { E } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] \right] . } \end{array}
$$

Proof. Write $\hat { g } _ { B } ^ { \mathrm { e m p } } ( \theta )$ as a ratio of sample means:

$$
\begin{array} { r } { \hat { g } _ { B } ^ { \mathrm { e m p } } ( \theta ) = \frac { \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \tilde { w } _ { \mathrm { e m p } } ^ { ( i ) } \nabla _ { \theta } f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) } { \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \tilde { w } _ { \mathrm { e m p } } ^ { ( j ) } } . } \end{array}
$$

Only independence across tuples is used. It holds because the clean pairs are i.i.d. and, conditionally on them, the B Langevin chains evolve independently: the batch energy in Line 11 is separable across i, so its gradient decouples blockwise, and the noise in Line 9 is independent across i. The within-tuple dependence is exactly what the factorization

$$
\pi ( x , y , u ) = p _ { \mathrm { d a t a } } ( x , y ) p _ { \mathrm { a d v } } ( u ; x , y , \theta )
$$

encodes.

Denominator. Since

$$
\tilde { w } _ { \mathrm { e m p } } ^ { ( j ) } \in \left[ \exp \bigl ( { - \beta F _ { \mathrm { m a x } } ( \theta ) } \bigr ) , 1 \right] ,
$$

the weights are integrable. By the Strong Law of Large Numbers (SLLN), the tower property, and Lemma 6 (ii),

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \tilde { w } _ { \mathrm { e m p } } ^ { ( j ) } \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { ( x , y , X ) \sim \pi } \left[ \exp \left( - \beta f ( X , y ; \theta ) \right) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \mathbb { E } _ { X \sim p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) } \left[ \exp \left( - \beta f ( X , y ; \theta ) \right) \right] \right] } \\ & { \quad \quad = \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) \right] } \\ & { \quad \quad \quad \geq \exp \left( - \beta F _ { \mathrm { m a x } } ( \theta ) \right) > 0 , } \end{array}
$$

where the lower bound follows from Lemma 6 (iii).

Numerator. Since

$$
\left\| \tilde { w } _ { \mathrm { e m p } } ^ { ( i ) } \nabla _ { \theta } f ( \boldsymbol { X } ^ { ( i ) } , \boldsymbol { y } ^ { ( i ) } ; \theta ) \right\| \leq \left\| \nabla _ { \theta } f ( \boldsymbol { X } ^ { ( i ) } , \boldsymbol { y } ^ { ( i ) } ; \theta ) \right\| ,
$$

the integrability assumption allows the SLLN to be applied componentwise:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \tilde { w } _ { \mathrm { e m p } } ^ { ( i ) } \nabla _ { \theta } f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) } \\ & { \quad \quad \quad \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { ( x , y , X ) \sim \pi } \left[ \exp \left( - \beta f ( X , y ; \theta ) \right) \nabla _ { \theta } f ( X , y ; \theta ) \right] . } \end{array}
$$

By the tower property and Lemma 6 (ii),

$$
\begin{array} { r l } & { \textstyle \mathbb { E } _ { X \sim p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) } \left[ \exp \bigl ( - \beta f ( X , y ; \theta ) \bigr ) \nabla _ { \theta } f ( X , y ; \theta ) \right] } \\ & { \textstyle \qquad = r ( x , y ; \theta ) \mathbb { E } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] . } \end{array}
$$

Thus the numerator converges almost surely to

$$
\begin{array} { r } { \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim p _ { \mathrm { d a t a } } } \left[ r ( \boldsymbol { x } , \boldsymbol { y } ; \boldsymbol { \theta } ) \mathbb { E } _ { \boldsymbol { X } \sim p _ { \lambda } ( \cdot ; \boldsymbol { x } , \boldsymbol { y } , \boldsymbol { \theta } ) } \left[ \nabla _ { \boldsymbol { \theta } } f ( \boldsymbol { X } , \boldsymbol { y } ; \boldsymbol { \theta } ) \right] \right] . } \end{array}
$$

Ratio. On the probability-one event where both limits hold, the denominator tends to a strictly positive constant by Lemma 6 (iii). The continuous mapping theorem therefore gives

$$
\begin{array} { r l } & { \hat { g } _ { B } ^ { \mathrm { e m p } } ( \theta ) \xrightarrow { \mathrm { a . s . } } \frac { { \mathbb { E } } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) { \mathbb { E } } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] \right] } { { \mathbb { E } } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) \right] } } \\ & { \quad \quad \quad \quad = \displaystyle \int _ { \mathcal { X } \times \mathcal { Y } } \frac { p _ { \mathrm { d a t a } } ( x , y ) r ( x , y ; \theta ) } { { \mathbb { E } } _ { ( x ^ { \prime } , y ^ { \prime } ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x ^ { \prime } , y ^ { \prime } ; \theta ) \right] } } \\ & { \quad \quad \quad \quad \quad \quad \times { \mathbb { E } } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] { \mathrm { d } } ( x , y ) } \\ & { \quad \quad \quad = { \mathbb { E } } _ { ( x , y ) \sim \tilde { p } ( \cdot , \cdot ; \theta ) } \left[ { \mathbb { E } } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] \right] , } \end{array}
$$

where the last equality holds because the normalizing constant of

$$
\tilde { p } ( x , y ; \theta ) \propto p _ { \mathrm { d a t a } } ( x , y ) r ( x , y ; \theta )
$$

is exactly

$$
\mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) \right] .
$$

## Relation to Theorem 5.

(i) The integrability assumption is implied by the data-averaged assumption of Theorem 5. By Lemma 6 (i),

$$
\frac { p _ { \mathrm { a d v } } ( u ; x , y , \theta ) } { p _ { \mathrm { d i s } } ( u ; x ) } = \frac { \exp \bigl ( c _ { 2 } f ( u , y ; \theta ) \bigr ) } { M _ { c _ { 2 } } ( x , y ; \theta ) } \leq \exp \bigl ( c _ { 2 } F _ { \operatorname* { m a x } } ( \theta ) \bigr ) ,
$$

since $M _ { c _ { 2 } } ( x , y ; \theta ) \ge 1$ . Hence

$$
\begin{array} { r l } & { \mathbb { E } _ { ( x , y , X ) \sim \pi } \left[ \| \nabla _ { \theta } f ( X , y ; \theta ) \| \right] } \\ & { \qquad \le \exp \bigl ( c _ { 2 } F _ { \operatorname* { m a x } } ( \theta ) \bigr ) \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x ) } \left[ \| \nabla _ { \theta } f ( X , y ; \theta ) \| \right] \right] . } \end{array}
$$

(ii) Theorem 7 strictly generalizes Theorem 5: with a single clean pair $( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } )$ and $\beta =$ $c _ { 2 }$ , the factor $r ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta )$ is a common constant that cancels in the self-normalized ratio. Moreover, $\lambda = 0$ and

$$
p _ { 0 } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) = p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) .
$$

The limit therefore reduces to

$$
\mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ \nabla _ { \theta } f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] ,
$$

which is the conclusion of Theorem 5.

(iii) At the other endpoint $\beta = 0$ , the weights are uniform:

$$
\hat { w } _ { \mathrm { e m p } } ^ { ( i ) } = \frac { 1 } { B } .
$$

In this case,

$$
r ( x , y ; \theta ) = 1 , \qquad \tilde { p } ( x , y ; \theta ) = p _ { \mathrm { d a t a } } ( x , y ) ,
$$

and

$$
p _ { \lambda } ( \cdot ; x , y , \theta ) = p _ { \mathrm { a d v } } ( \cdot ; x , y , \theta ) .
$$

This is unweighted adversarial training on Langevin samples, i.e., exactly the PAT(WOS) variant of Section 6.2.

(iv) Had the intractable weights

$$
w _ { \mathrm { t r u e } } ^ { ( i ) } \propto Z _ { \mathrm { a d v } } ^ { ( i ) } Z _ { \mathrm { v i c } } ^ { ( i ) } \exp \left( - c _ { 2 } f ( X ^ { ( i ) } , y ^ { ( i ) } ; \theta ) \right)
$$

of Section 5.4 been used, the same argument, using the identity

$$
p _ { \mathrm { a d v } } \bigl ( u ; x , y , \theta \bigr ) Z _ { \mathrm { a d v } } Z _ { \mathrm { v i c } } \exp \bigl ( - c _ { 2 } f \bigl ( u , y ; \theta \bigr ) \bigr ) = p _ { \mathrm { d i s } } \bigl ( u ; x \bigr )
$$

from Lemma 6, would give the limit

$$
\begin{array} { r } { \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] \right] , } \end{array}
$$

the exact joint analogue of Theorem 5.

Proposition 8 (Entropic-risk gradient identity). $F i x \left( x , y \right) \in \mathcal { X } \times \mathcal { Y }$ . Assume, as in Proposition 4, that $\nabla _ { \theta }$ may be exchanged with integration over X. Then, $f o r \lambda > 0$

$$
\nabla _ { \theta } R _ { \lambda } ( x , y ; \theta ) = \mathbb { E } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] .
$$

The identity extends to $\lambda = 0$ , with

$$
R _ { 0 } ( \boldsymbol { x } , \boldsymbol { y } ; \theta ) = \mathbb { E } _ { \boldsymbol { X } \sim p _ { \mathrm { d i s } } ( \cdot ; \boldsymbol { x } ) } \left[ f ( \boldsymbol { X } , \boldsymbol { y } ; \theta ) \right] , \qquad p _ { 0 } ( \cdot ; \boldsymbol { x } , \boldsymbol { y } , \theta ) = p _ { \mathrm { d i s } } ( \cdot ; \boldsymbol { x } ) .
$$

Proof. Since $p _ { \mathrm { d i s } } ( \cdot ; x )$ does not depend on θ,

$$
\begin{array} { l } { \nabla _ { \theta } \log M _ { \lambda } ( x , y ; \theta ) = \displaystyle \frac { \lambda } { M _ { \lambda } ( x , y ; \theta ) } \int _ { \mathcal { X } } p _ { \mathrm { d i s } } ( u ; x ) \nabla _ { \theta } f ( u , y ; \theta ) \exp \bigl ( \lambda f ( u , y ; \theta ) \bigr ) \mathrm { d } u } \\ { = \lambda \mathbb { E } _ { X \sim p _ { \lambda } ( \cdot ; x , y , \theta ) } \left[ \nabla _ { \theta } f ( X , y ; \theta ) \right] . } \end{array}
$$

Dividing by λ gives the claim for $\lambda > 0$

Corollary 9 (What PAT optimizes in practice). Combining Theorem 7 and Proposition $\delta ,$ as $B $ $\infty ,$

$$
\widehat { g } _ { B } ^ { \mathrm { e m p } } ( \theta ) \xrightarrow { \mathrm { a . s . } } \nabla _ { \theta ^ { \prime } } \mathbb { E } _ { ( x , y ) \sim \widetilde { p } ( \cdot , \cdot ; \theta ) } \left[ R _ { \lambda } ( x , y ; \theta ^ { \prime } ) \right] \big | _ { \theta ^ { \prime } = \theta } ,
$$

where the gradient passes inside the outer expectation because $\tilde { p } ( \cdot , \cdot ; \theta )$ does not depend on $\theta ^ { \prime }$

Proposition 10 (The smoothed objective preserves the lower bound on probabilistic robustness). Fix a clean pair $( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } )$ , and let $s , E _ { \mathrm { a d v } } ,$ , and $\gamma ^ { * } > 0$ be as in Theorem 2. Then:

(i) Monotone family. The map $\lambda \mapsto R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta )$ is continuous on $[ 0 , \infty )$ , and, for $\lambda > 0 ,$

$$
\frac { \mathrm { d } } { \mathrm { d } \lambda } R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) = \frac { 1 } { \lambda ^ { 2 } } \mathrm { K L } ( p _ { \lambda } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) \| p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) \ge 0 .
$$

Hence, for every $\lambda \in [ 0 , c _ { 2 } ] ,$

$$
R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \le R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \le R _ { c _ { 2 } } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) ,
$$

with the exact integral representation

$$
R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) - R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) = \int _ { 0 } ^ { \lambda } { \frac { 1 } { t ^ { 2 } } } \mathrm { K L } ( p _ { t } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) \| p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) \mathrm { d } t .
$$

(ii) A family of lower bounds. For every $\beta \in [ 0 , c _ { 2 } ] , i . e .$ , every $\lambda = c _ { 2 } - \beta \in [ 0 , c _ { 2 } ]$

$$
\operatorname { P R } ( p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) , s ) \ge 1 - \frac { c _ { 2 } R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) } { \gamma ^ { * } } .
$$

$A t \lambda = 0 ( i . e . , \beta = c _ { 2 } )$ , this coincides with the bound ofTheorem 2, since

$$
\begin{array} { r l } & { \log Z _ { \mathrm { v i c } } - H ( p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) - \operatorname { K L } ( p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) \parallel p _ { \mathrm { v i c } } ( \cdot ; y _ { \mathrm { o r i } } , \theta ) ) } \\ & { \quad \quad \quad \quad = \operatorname { \mathbb { E } } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ c _ { 2 } f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] } \\ & { \quad \quad \quad \quad = c _ { 2 } R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) . } \end{array}
$$

For $\lambda > 0 ,$ , the bound is looser by exactly

$$
\frac { c _ { 2 } } { \gamma ^ { * } } \left( R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) - R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \right) ,
$$

which is quantified by the integral representation in (i).

Proof. (i) The boundedness of $f ( \cdot , y _ { \mathrm { o r i } } ; \theta )$ justifies differentiation under the integral:

$$
\frac { \mathrm d } { \mathrm d \lambda } \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) = \mathbb E _ { X \sim p _ { \lambda } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) } \left[ f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] .
$$

Moreover,

$$
\begin{array} { r l } & { \mathrm { K L } ( p _ { \lambda } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) \parallel p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) = \mathbb { E } _ { X \sim p _ { \lambda } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) } \left[ \lambda f ( X , y _ { \mathrm { o r i } } ; \theta ) - \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \right] } \\ & { \quad \quad \quad \quad \quad \quad = \lambda \frac { \mathrm { d } } { \mathrm { d } \lambda } \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) - \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \ge 0 . } \end{array}
$$

The quotient rule therefore gives

$$
\frac { \mathrm d } { \mathrm d \lambda } \left( \frac { \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) } { \lambda } \right) = \frac { 1 } { \lambda ^ { 2 } } \mathrm { K L } ( p _ { \lambda } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) \| p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) .
$$

Continuity at $\lambda = 0$ follows from the expansion

$$
\begin{array} { r } { \log M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) = \lambda \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] + O ( \lambda ^ { 2 } ) . } \end{array}
$$

The integral representation follows from the fundamental theorem of calculus, with the integrand integrable at 0 since

$$
\frac { 1 } { t ^ { 2 } } \operatorname { K L } ( p _ { t } ( \cdot ; x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } , \theta ) \| p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) ) \xrightarrow [ t \downarrow 0 ] { } \frac { 1 } { \frac { 1 } { 2 } } \operatorname { V a r } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] .
$$

(ii) Jensen’s inequality gives

$$
M _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \ge \exp \left( \lambda \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] \right) ,
$$

and hence

$$
R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) \ge R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) .
$$

From the proof of Theorem 2,

$$
\begin{array} { r } { \mathbb { E } _ { X \sim p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) } \left[ c _ { 2 } f ( X , y _ { \mathrm { o r i } } ; \theta ) \right] \geq \gamma ^ { * } I [ p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) , s ] . } \end{array}
$$

Therefore,

$$
I [ p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) , s ] \leq \frac { c _ { 2 } R _ { 0 } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) } { \gamma ^ { * } } \leq \frac { c _ { 2 } R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) } { \gamma ^ { * } } ,
$$

and

$$
\begin{array} { r } { \operatorname { P R } ( p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) , s ) = 1 - I [ p _ { \mathrm { d i s } } ( \cdot ; x _ { \mathrm { o r i } } ) , s ] \ge 1 - \frac { c _ { 2 } R _ { \lambda } ( x _ { \mathrm { o r i } } , y _ { \mathrm { o r i } } ; \theta ) } { \gamma ^ { * } } . } \end{array}
$$

Corollary 11 (Aggregate control of average probabilistic robustness). For each $y \in \mathcal { V }$ , define the per-label quantities

$$
\begin{array} { r } { s _ { y } ( u ) : = \underset { y ^ { \prime } \neq y } { \operatorname* { m a x } } \big ( z _ { y ^ { \prime } } ( u ) - z _ { y } ( u ) \big ) , } \\ { E _ { \mathrm { a d v } } ( y ) : = \{ u \in \mathcal { K } : s _ { y } ( u ) \geq 0 \} , } \\ { \gamma ^ { * } ( y ; \theta ) : = \underset { u \in E _ { \mathrm { a d v } } ( y ) } { \operatorname* { i n f } } c _ { 2 } f ( u , y ; \theta ) , } \end{array}
$$

with the convention inf $\mathcal { D } = + \infty$ . Assume $\gamma ^ { \ast } ( y ; \theta ) > 0 .$ for every $y \in { \mathcal { D } } .$ . Since Y is finite,

$$
\gamma _ { \operatorname* { m i n } } ^ { * } ( \theta ) : = \operatorname* { m i n } _ { y \in \mathcal { V } } \gamma ^ { * } ( y ; \theta ) > 0 .
$$

Define the average probabilistic robustness

$$
\begin{array} { r } { \overline { { \mathrm { P R } } } ( \theta ) : = \mathbb { E } _ { ( { x } , { y } ) \sim p _ { \mathrm { d a t a } } } \left[ \mathrm { P R } ( p _ { \mathrm { d i s } } ( \cdot ; { x } ) , { s } _ { y } ) \right] . } \end{array}
$$

Then, for every $\beta \in [ 0 , c _ { 2 } ]$

$$
\overline { { \mathrm { P R } } } ( \theta ) \geq 1 - \frac { c _ { 2 } \exp \left( \beta F _ { \mathrm { m a x } } ( \theta ) \right) } { \gamma _ { \mathrm { m i n } } ^ { * } ( \theta ) } \mathbb { E } _ { ( x , y ) \sim \widetilde { p } ( \cdot , \cdot ; \theta ) } \left[ R _ { \lambda } ( x , y ; \theta ) \right] .
$$

Proof. Apply Proposition 10(ii) to each clean pair. Using

$$
\gamma ^ { \ast } ( y ; \theta ) \geq \gamma _ { \mathrm { m i n } } ^ { \ast } ( \theta )
$$

and $R _ { \lambda } ( x , y ; \theta ) \ge 0$ (since $M _ { \lambda } ( x , y ; \theta ) \ge 1 )$ , and taking expectations over $( x , y ) \sim p _ { \mathrm { d a t a } }$ , gives

$$
\overline { { \mathrm { P R } } } ( \theta ) \geq 1 - \frac { c _ { 2 } } { \gamma _ { \mathrm { m i n } } ^ { * } ( \theta ) } \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ R _ { \lambda } ( x , y ; \theta ) \right] .
$$

By Lemma 6(iii),

$$
\exp \bigl ( - \beta F _ { \mathrm { m a x } } ( \theta ) \bigr ) \leq r ( x , y ; \theta ) \leq 1 .
$$

Hence,

$$
\begin{array} { r l } & { \mathbb { E } _ { ( x , y ) \sim \widetilde { p } ( \cdot , \cdot ; \theta ) } \left[ R _ { \lambda } ( x , y ; \theta ) \right] = \frac { \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) R _ { \lambda } ( x , y ; \theta ) \right] } { \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ r ( x , y ; \theta ) \right] } } \\ & { \quad \quad \quad \geq \exp \bigl ( - \beta F _ { \mathrm { m a x } } ( \theta ) \bigr ) \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ R _ { \lambda } ( x , y ; \theta ) \right] . } \end{array}
$$

Equivalently,

$$
{  { \mathbb E } } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ R _ { \lambda } ( x , y ; \theta ) \right] \le \exp \left( \beta F _ { \mathrm { m a x } } ( \theta ) \right) {  { \mathbb E } } _ { ( x , y ) \sim \widetilde { p } ( \cdot , \cdot ; \theta ) } \left[ R _ { \lambda } ( x , y ; \theta ) \right] .
$$

Combining the two bounds completes the proof.

## D FURTHER DISCUSSION ON THE GLOBAL VULNERABILITY CAPACITY $Z _ { v i c }$

While our theoretical results establish $K L ( p _ { d i s } | | p _ { v i c } ) \ – \log Z _ { v i c }$ as a surrogate for probabilistic robustness, a deeper understanding of the term $- \log Z _ { v i c }$ reveals a fundamental trade-off in robust optimization. From the perspective of distribution expansion, the normalization constant $\begin{array} { r } { Z _ { v i c } = \hat { \int } _ { \mathcal { X } } \exp ( c _ { 2 } f ( u , y _ { o r i } ; \theta ) ) } \end{array}$ )du represents the total “volume” of the model’s vulnerability across the entire input hypercube $\mathcal { X } = [ 0 , 1 ] ^ { d }$

Geometrically, the probability density $p _ { v i c }$ is a normalized representation of the classification loss. Because the domain X is compact and the integral of $p _ { v i c }$ is constrained to 1, any attempt to minimize the density (i.e., increase robustness) in the local neighborhood of $x _ { o r i }$ via $K L ( \bar { p } _ { d i s } | | p _ { v i c } )$ must be compensated by an increase in density elsewhere in the domain. This is analogous to the “water level” in a closed pool: pushing the water away from one area (the distance distribution) necessarily causes the overall water level $( Z _ { v i c } )$ to rise if the container’s volume is fixed.

This perspective implies that − log $Z _ { v i c }$ is not merely a mathematical residue but a global regularizer that penalizes the “compression” of vulnerabilities. If the optimization focuses solely on the KL term, the model might achieve high local robustness by creating extremely sharp decision boundaries just outside the $p _ { d i s }$ support, leading to a surge in $Z _ { v i c }$ and potential global instability. Therefore, PAT’s inherent structure forces a principled balance: it requires the model to not only push vulnerabilities away from the clean data but also to suppress the total growth of the global vulnerability volume. This justifies the use of the smoothing factor $\beta$ in Equation equation 2 as a mechanism to manage this pressure and prevent catastrophic weight collapse during the mass redistribution process.

## E DETAILS OF THE EXPERIMENTS

In this section, we provide the comprehensive experimental settings, hyperparameters, and evaluation protocols required to reproduce the results presented in Section 6.

## E.1 BASIC TRAINING SETUP

For all experiments conducted on the CIFAR-10 and CIFAR-100 datasets, we apply a standard data augmentation pipeline to the training set: a 4-pixel reflection padding $\left( \mathrm { P a d } \left( 4 , \mathrm { \quad r e f 1 e c t } \right) \right)$ followed by a random crop to $3 2 \times 3 2$ pixels, and a random horizontal flip with a probability of $p = 0 . 5$

We train all models for a total of $E _ { t o t a l } = 1 0 0$ epochs using a batch size of $B = 2 5 6$ . The network parameters are optimized using Stochastic Gradient Descent (SGD) with a Nesterov momentum of 0.9 and a weight decay of $5 \times 1 0 ^ { - 4 }$ . The initial learning rate is set to $\alpha = 0 . 0 1$ and is scheduled using a MultiStepLR decay strategy, which multiplies the learning rate by a decay factor of $\gamma = 0 . 1$ 1 at milestones of epoch 75 and epoch 90.

## E.2 HYPERPARAMETERS OF PROBABILISTIC ADVERSARIAL TRAINING

For our proposed Probabilistic Adversarial Training (PAT) detailed in Algorithm 2, the generation of the adversarial distribution via Langevin Dynamics requires careful calibration. The precise hyperparameter configurations used across our main experiments are summarized below:

• Langevin Dynamics Steps (T): 100 iterations.

• Langevin Step Size (η): 0.3.

• Noise Scale (σ): 0.001, which ensures sufficient stochastic exploration within the local neighborhood.

• Gradient Clipping Threshold (ρ): 1.0, applied to stabilize the energy landscape exploration.

• Energy Weights: We set the distance distribution weight to $c _ { 1 } ~ = ~ 0 . 3$ and the victim distribution (loss) weight to $c _ { 2 } = 0 . 4 2$

• Smoothing Factor (β): As extensively discussed in Section 5.4 and Appendix D, we instantiate the smoothing factor as $\beta = 0 . 0 0 1$ to prevent global weight collapse during the Self-Normalized Importance Sampling (SNIS).

## E.3 BASELINES AND EVALUATION PROTOCOL

Standard Robustness Evaluation: To assess standard worst-case robustness, we evaluate the models using the well-established Projected Gradient Descent (PGD) and Carlini-Wagner (CW) attacks. Both attacks are executed for 20 iterative steps (PGD-20 and CW-20). $\epsilon = 8 / 2 5 \bar { 5 } , \alpha = 2 / 2 5 5$

Probabilistic Robustness Evaluation: For the Probabilistic Robustness (PR) reported in Table 1, the test-time distance distributions are configured with escalating perturbation scales $\epsilon \in$ {0.1, 0.12, 0.15, 0.2}. We sample N = 100 perturbations per image to empirically estimate the PR objective.

Baseline Training Settings: For standard adversarial training baselines such as PGD-AT and TRADES, we strictly follow their original implementations. The baselines generate training adversaries using a 10-step attack with a maximum perturbation bounded by $L _ { \infty } \leq 8 / 2 5 5$

## E.4 COMPUTE RESOURCES

All experiments, including baseline training and PAT, were implemented in PyTorch. The models were trained and evaluated on a compute cluster equipped with NVIDIA RTX 5090 GPUs. A single complete training run for PAT on CIFAR-10 takes approximately 7 minutes per epoch on a single GPU.

## F LIMITATIONS

Probabilistic adversarial attacks generate adversarial examples from noise and therefore typically require more generation steps than conventional adversarial attacks. This leads to a higher computational cost compared with standard attack methods.