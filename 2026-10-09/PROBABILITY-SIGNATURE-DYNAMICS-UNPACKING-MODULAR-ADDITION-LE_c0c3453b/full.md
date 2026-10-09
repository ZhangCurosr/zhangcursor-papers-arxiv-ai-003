# PROBABILITY-SIGNATURE DYNAMICS: UNPACKING MODULAR ADDITION LEARNING WITHIN TWO-LAYER NETWORKS

Yunji Wang<sup>1,†</sup>, Junjie Yao<sup>1,†</sup>, Linyu Liu<sup>3</sup>, Pinyan Lu<sup>4</sup>, Zhi-Qin John Xu<sup>1,2,∗</sup>

<sup>1</sup>School of Mathematical Sciences, Shanghai Jiao Tong University <sup>2</sup>Institute of Natural Sciences, MOE-LSC, Shanghai Jiao Tong University <sup>3</sup>Huawei Technologies Ltd <sup>4</sup>Shanghai University of Finance and Economics <sup>†</sup>These authors contributed equally to this work. <sup>∗</sup>Corresponding author: xuzhiqin@sjtu.edu.cn

## ABSTRACT

Neural networks trained on modular addition tasks often develop Fourier-structured representations that support exact generalization. While prior work has identified these Fourier circuits, the mechanism by which gradient-based training selects them from the data distribution remains unclear. We address this question using probability signatures, which express leading gradient interactions through conditional statistics of the training distribution. For modular addition, these signatures are cyclic shift operators and are diagonalized by the discrete Fourier transform, yielding approxi mately decoupled Fourier-mode dynamics. This explains the emergence of Fourier sparsity, frequency matching, and phase alignment. The same framework resolves a puzzle under label noise: corrupted examples can show faster early loss decrease than clean examples, despite lacking a coherent generalization rule. We show that noise increases conditional label collisions, strengthening early shared-coordinate reinforcement. Finally, this method can be applied to other operators. Taking XOR as an example, we observed the predicted frequency in experiments.

Keywords Two Layer Network, Modular Addition, Training Dynamics

## 1 Introduction

The modular addition task is a canonical setting in which to study grokking. During the initial phase of training, an over-parameterized model can fit individual training examples yet fails to uncover an interpretable rule. After substantially prolonged training, however, its test performance may improve abruptly even though the training loss has already become very small. This delayed transition is commonly referred to as grokking. In modular addition, the onset of grokking is accompanied by the emergence of a distinctive structure in the weights. Prior studies have shown that trained networks develop Fourier circuits (Nanda et al., 2023; Gromov, 2023). Specifically, the input and output weights of many hidden units become concentrated at matching Fourier frequencies, and appropriate combinations of these Fourier components can compute (a + b) mod P. Mechanistic interventions, including Fourier projection and ablation experiments, provide strong evidence that these components are functionally important for generalization. Their formation also reflects the transition from a memorizing solution to a generalizing one.

These results answer an important reverse question: given a trained network, which circuit does it implement, and which components of that circuit support generalization? They have not fully answered the complementaryforward question: why should gradient-based training produce Fourierstructured parameters from the modular-addition data distribution in the first place? The fact that Fourier features can represent modular addition does not imply that the training dynamics must discover them. Meanwhile, current work can already account for observed phenomena in neural networks, such as frequency competition and phase alignment (He et al., 2026). However, its theoretical analysis is based on a Fourier decomposition of the weights, which need to presuppose that the Fourier basis is related to modular addition.

Our focus is therefore not to show again that Fourier-structured neurons can implement modular addition, nor to re-establish that these neurons support generalization. Instead, we study the missing forward direction

$$
\mathrm { t r a i n i n g ~ d i s t r i b u t i o n } \quad \Longrightarrow \quad { \mathrm { l e a d i n g ~ g r a d i e n t ~ t e r m s } } \quad \Longrightarrow \quad { \mathrm { F o u r i e r - m o d e ~ s e l e c t i o n } } .
$$

More specifically, we ask the following questions. Which statistics of the training data enter the leading-order parameter dynamics? Why are Fourier modes approximately decoupled in the modular-addition problem? What determines whether a neuron is suppressed by weight decay or develops a dominant Fourier component? And, how to derive the weight structure in advance from the training distribution?

We study these questions in a two-layer embedding MLP trained on modular addition dataset with some randomly corrupted labels. Our main analytical tool is the probability signature framework introduced by Yao & Xu (2025), which shows the correlation between data statistics and embedding without studying either modular addition learning or the key structure of data statistics.

Our contributions are as follows.

• We utilize probability signatures to directly explain the emergence of dominant Fourier modes during the learning dynamics.

• We have explained the puzzle in the early memorization regime: uniformly random label noise consistently exhibits a faster initial loss decrease than clean data. Some statistics from training distribution can measure this difference.

• The analysis framework can be extended to more tasks, such as XOR, enabling us to predict the structures that network parameters may form before training.

## 2 Related Work

Grokking. Grokking, the delayed onset of generalization long after interpolation, was first reported on small algorithmic datasets, where weight decay proved essential (Power et al., 2022). Subsequent work explains the delay through properties of the optimization trajectory: effective theories of representation learning and norm growth (Liu et al., 2022), loss landscapes shaped by weight decay and initialization scale (Liu et al., 2023), instabilities of adaptive optimizers (Thilak et al., 2022), and the transition from lazy to rich training dynamics (Kumar et al., 2024). These characterize when and why generalization occurs, from the emergent Fourier circuits as a result of training. Based on this, we point out that weight decay determines the formation of the Fourier structure as a threshold, which establishes the connection between weight decay and Grokking.

Modular arithmetic. Prior work has established that trained networks can implement modular addition through sparse Fourier circuits, and has introduced diagnostics such as frequency matching and phase alignment to characterize these circuits (Nanda et al., 2023). Analytic and scaling-oriented studies have further related the norms and spectral profiles of Fourier modes to the learned solution (Gromov, 2023; Michaud et al., 2023). These works provide important mechanistic and functional descriptions of Fourier structure in trained networks. Our analysis is based on these research results, but we present a forward perspective: starting from training distribution rather than from the Fourier circuits obtained through training. More recent work (He et al., 2026) provides a complete explanation for the frequency competition and phase alignment by decomposing neurons onto the Fourier basis. Compared with theirs, our contribution stands out in the fact that we do not require the prior knowledge of the Fourier circuit found in the modular-addition task. By studying the algebraic structure of the probability signature, we derive the Fourier basis as the common eigenvectors. We not only explain the above phenomena, but also we hope to provide a unified explanation for the puzzle of early noise and the extension to other training tasks.

Learning with corrupted labels. Neural networks can both generalize from structured examples and memorize corrupted labels. This interaction has been studied broadly through early learning and memorization dynamics (Arpit et al., 2017; Liu et al., 2020). For algorithmic and modular-arithmetic tasks, prior work has documented robust generalization in the presence of label corruption and the coexistence of generalizing and memorizing subnetworks (Doshi et al., 2024; Liu & Lu, 2026). Our contribution is not a general theory of learning under label noise. Instead, we isolate one early-time phenomenon: randomly corrupted examples lead to faster initial loss decrease, and we relate it to probability signature, which is computable from the training data.

## 3 Preliminaries

## 3.1 Settings

Let $\Omega = \{ 0 , 1 , \ldots , P - 1 \}$ be a finite alphabet, where $P$ is a prescribed prime. We consider the prediction of a binary operation $F : \Omega \times \Omega  \Omega$ from examples $X = ( a , b )$ to the label $F ( a , b )$ All pairs in $\Omega \times \Omega$ are divided into three disjoint subsets $\mathcal { D } _ { \mathrm { c l e a n } } , \mathcal { D } _ { \mathrm { n o i s e } } , \mathcal { D } _ { \mathrm { t e s t } }$ of cardinalities $N _ { \mathrm { c l e a n } } .$ $N _ { \mathrm { n o i s e } }$ , and $N _ { \mathrm { t e s t } }$ , respectively. The labels of clean and test are set to $F ( a , b ) = ( a + b )$ mod $P _ { - }$ while the labels of noise are replaced by uniform random numbers.

The training dataset consists of $\mathcal { D } _ { \mathrm { c l e a n } }$ and $\mathcal { D } _ { \mathrm { n o i s e } }$ . We define $N _ { \mathrm { t r a i n } } = N _ { \mathrm { c l e a n } } + N _ { \mathrm { n o i s e } }$ , where $N _ { \mathrm { t r a i n } }$ $N _ { \mathrm { t e s t } } = 8 : 2$ and $N _ { \mathrm { n o i s e } } = \alpha N _ { \mathrm { t r a i n } }$ . α represents the noise ratio.

Model. Let $\mathbf { e } _ { a } , \mathbf { e } _ { b } \in \mathbb { R } ^ { P }$ be one-hot encodings. We study the two-layer MLP $f ( a , b ) = W \sigma ( U \mathbf { e } _ { a } +$ $V { \bf e } _ { b } ) \in \mathbb { R } ^ { P }$ , where $U , V \in \mathbb { R } ^ { D \times P } , W \in \mathbb { R } ^ { P \times D } , { \mathbf e } _ { a } , { \mathbf e } _ { b }$ means the one-hot encodings of input $a , b ,$ and $D \gg P$ . The output distribution is softmax $( f ( a , b ) )$

We use the cross-entropy loss function and employ $L ^ { 2 }$ -weight decay with coefficient λ. Parameters are initialized independently according to $U _ { i j } , V _ { i j } , W _ { i j } \sim \mathcal { N } ( 0 , D ^ { - 2 \gamma } )$ , where $\gamma$ controls the initialization scale. Consider a continuous gradient flow model as an approximation, it is $\begin{array} { r } { \frac { \mathrm { d } \theta } { \mathrm { d } t } = - \eta ( \nabla _ { \theta } L + \lambda \theta ) } \end{array}$ . For more details about initialization, learning rate or optimizer, please refer to the appendix A.

## 3.2 Probability signatures

The Probability Signature proposed in the related work (Yao & Xu, 2025) is a statistical quantity derived from sampling and not dependent on training. It can characterize the main terms of the gradient at different training stages. In order to create the application environment for Probability Signature, we first extended the fully connected network with the same embedding to our model.

Proposition 1 (Gradient flow). Assume the neural network $f ( x , y ) = W \sigma ( U \mathbf { e } _ { x } + V \mathbf { e } _ { y } )$ , using full batch GD with cross-entropy loss and ignoring weight decay, with the activation function $\begin{array} { r } { \sigma ( h ) = 0 + c _ { 1 } h + \frac { c _ { 2 } } { 2 } h ^ { 2 } + o ( h ^ { 2 } ) } \end{array}$ which is $C ^ { 3 }$ near zero. Ifwe sample training datafrom $( x , y , z ) \sim \pi$ then we denote the probability that the first position is x by $r _ { x }$ and the random variable $\mathbb { E } _ { \pi } [ \cdot | x ]$ by the conditional expectation under thefixedfirst position.

Under the assumption ofsmall initialization that $\| U \| _ { \infty } , \| V \| _ { \infty } , \| W \| _ { \infty } \leq \epsilon ,$ , we derived that

$$
\begin{array} { c } { \displaystyle \frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } \eta \Bigg \{ c _ { 1 } W ^ { \top } ( I - \frac { 1 } { P } \mathbf { 1 } _ { P } \mathbf { 1 } _ { P } ^ { \top } ) \left[ \mathbb { E } _ { \boldsymbol \pi } [ \mathbf { e } _ { z } \mid x ] - \frac { c _ { 1 } } { P } W \big ( U ^ { x } + \mathbb { E } _ { \boldsymbol \pi } [ V ^ { y } \mid x ] \big ) \right] } \\ { \displaystyle + c _ { 2 } \mathbb { E } _ { \boldsymbol \pi } \left[ \big ( U ^ { x } + V ^ { y } \big ) \odot W ^ { \top } ( \mathbf { e } _ { z } - \frac { 1 } { P } \mathbf { 1 } _ { P } ) \bigg | x \right] \Bigg \} + \delta . } \end{array}
$$

where $U ^ { x } \in \mathbb { R } ^ { d }$ represent the x-th column of U and $\| \delta \| _ { \infty } = O ( \epsilon ^ { 3 } )$ means the higher order terms.

## Proof. See Appendix E.1.

From the gradient flow equation, we can further decompose the conditional expectation. Assume sampling from the training distribution $( X , Y , Z ) \sim \pi$ , we define the probability signatures:

$$
\varphi ^ { ( a ) } ( x , z ) = \mathbb { P } ( Z = z \mid X = x ) , \qquad \varphi ^ { ( b ) } ( x , z ) = \mathbb { P } ( Z = z \mid Y = x ) ,
$$

the 3-order tensor joint probability signatures:

$$
\Phi ^ { ( a ) } ( x , y , z ) = { \mathbb P } ( Z = z , Y = y \mid X = x ) , \quad \Phi ^ { ( b ) } ( x , y , z ) = { \mathbb P } ( Z = z , X = y \mid Y = x ) ,
$$

and the trace-back probability signatures

$$
\psi ^ { ( a ) } ( x , y ) = \mathbb { P } ( Y = y \mid X = x ) , \qquad \psi ^ { ( b ) } ( x , y ) = \mathbb { P } ( X = y \mid Y = x ) .
$$

We also denote $\begin{array} { r } { \varphi _ { x } ^ { ( a ) } : = \sum _ { z \in \Omega } \mathbb { P } ( Z = z \mid X = x ) e _ { z } } \end{array}$ , a column vector, and $\begin{array} { r } { \Phi _ { x } ^ { ( a ) } : = \sum _ { y , z \in \Omega } \mathbb { P } ( Z = } \end{array}$ $z , Y = y ~ \vert ~ X = x ) e _ { y } \otimes e _ { z }$ , a matrix. The same notation applies to the others. Then we can rewrite the gradient flow equation in Proposition 1 as

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } \eta \Biggl \{ c _ { 1 } W ^ { \top } \biggl [ ( \varphi _ { x } ^ { ( a ) } - \frac { 1 } { P } \mathbf { 1 } _ { P } ) - \frac { c _ { 1 } } { P } ( I - \frac { 1 } { P } \mathbf { 1 } _ { P } \mathbf { 1 } _ { P } ^ { \top } ) W ( U ^ { x } + V \psi _ { x } ^ { ( a ) } ) \biggr ] } \\ & { \qquad + c _ { 2 } \biggl [ U ^ { x } \odot W ^ { \top } \varphi _ { x } ^ { ( a ) } - \frac { 1 } { P } ( U ^ { x } + V \psi _ { x } ^ { ( a ) } ) \odot ( W ^ { \top } \mathbf { 1 } _ { P } ) + \mathrm { d i a g } ( V \Phi _ { x } ^ { ( a ) } W ) \biggr ] \Biggr \} + \delta . } \end{array}
$$

The above statistics $\varphi , \Phi , \psi$ are referred to as Probability Signatures.

## 4 Theoretical Analysis

In this chapter, we hope to derive this frequency structure from a forward perspective. Reviewing the gradient flow equation 1, we divide it into $\varphi \mathrm { - p a r t } ,$ Φ-part and ψ-part based on the association with different probability signatures. Additionally, due to the weight decay, there is also $\mathbf { a } - \lambda U$ term. The proof of the following theorems will be found in Appendix E.2,E.3, and $\mathrm { E . 4 }$

Theorem 1 characterizes the dominant part of the gradient flow equation.

Theorem 1 (Dominance of the gradient). Under the condition of 1 and non-zero weight decay, assume that $\eta = 1$ and

$$
\varphi _ { x } ^ { ( a ) } = \varphi _ { x } ^ { ( b ) } = \psi _ { x } ^ { ( a ) } = \psi _ { x } ^ { ( b ) } = \frac { 1 } { P } \mathbf { 1 } _ { P } , \qquad \forall x \in \Omega ,
$$

Under small initialization that $\| U \| _ { \infty } , \| V \| _ { \infty } , \| W \| _ { \infty } \leq \epsilon$ we have

$$
\frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = - \lambda U ^ { x } + c \bigg [ \mathrm { d i a g } \big ( V \Phi _ { x } ^ { ( a ) } W \big ) - \overline { { V } } \odot \overline { { W } } \bigg ] + \delta _ { U } ,
$$

$$
\frac { \mathrm { d } V ^ { x } } { \mathrm { d } t } = - \lambda V ^ { x } + c \bigg [ \mathrm { d i a g } \big ( U \Phi _ { x } ^ { ( b ) } W \big ) - \overline { { U } } \odot \overline { { W } } \bigg ] + \delta _ { V } ,
$$

$$
\frac { \mathrm { d } W _ { z } } { \mathrm { d } t } = - \lambda W _ { z } + c \bigg [ \mathrm { d i a g } \big ( U \Psi _ { z } V ^ { \top } \big ) - \overline { { U } } \odot \overline { { V } } \bigg ] + \delta _ { W } .
$$

where $c = c _ { 2 } r _ { x } , \ \| \delta \| _ { \infty } \lesssim \epsilon ^ { 3 } .$ . Here we define $\begin{array} { r } { \overline { { U } } : = P ^ { - 1 } \sum _ { x } U ^ { x } , \overline { { V } } : = P ^ { - 1 } \sum _ { x } V ^ { x } , \overline { { W } } : = } \end{array}$ $\textstyle P ^ { - 1 } \sum _ { x } ( W ^ { \top } ) ^ { x }$ , and $\begin{array} { r } { \Psi _ { z } : = \sum _ { x , y \in \Omega } \mathbb { P } _ { \pi } [ X = x , Y = y \mid Z = z ] e _ { x } \otimes e _ { y } . } \end{array}$

## 4.1 The Fourier basis derived from the signatures

Let’s assume there is no noise and the training is conducted on the entire clean dataset. We define $\begin{array} { r } { \Phi _ { x } ^ { + } : = P ^ { - 1 } \sum _ { j \in \mathbb { Z } _ { P } } e _ { j } \otimes e _ { j + x } } \end{array}$ as the joint signature in this situation. Write $S _ { x } = P \Phi _ { x } ^ { + }$ , which is a shift-x permutation matrix. They satisfy

$$
S _ { x } S _ { y } = S _ { x + y } , \qquad S _ { x } ^ { * } = S _ { - x } , \qquad S _ { x } ^ { * } S _ { x } = I .
$$

Here <sup>∗</sup> denotes the conjugate transpose and I is the $P \times P$ identity matrix. They form a commuting family of normal operators and therefore possess a common orthonormal eigenvectors over C. The

basis can be obtained directly from the simplest nontrivial shift: $S _ { 1 } q = \omega q$ implies $F ( y + 1 ) = \omega F ( y )$ for some unit root $\omega ^ { P } = 1$ , which yields

$$
F _ { r } ( s ) = P ^ { - 1 / 2 } e ^ { - 2 \pi \mathrm { i } r s / P } , \qquad \Phi _ { x } ^ { + } F _ { r } = P ^ { - 1 } e ^ { 2 \pi \mathrm { i } r x / P } F _ { r } .\tag{1}
$$

These vectors are the normalized Fourier basis, and $F : = [ F _ { 0 } , F _ { 1 } , \cdot \cdot \cdot , F _ { P - 1 } ]$ is the standard Fourier matrix. We use $\hat { \bf u } : = { \cal F }$ u to represent the Discrete Fourier Transform (DFT) of vector u.

Estimate by the main term of Theorem 1, we find that this system is separable for each hidden unit $d \in [ D ]$ . Thus we can consider this equation on one hidden state ${ \mathbf u } , { \mathbf v } , { \mathbf w } \in \mathbb { R } ^ { P }$ , with system

$$
\dot { \mathbf { u } } _ { x } = c \Bigg [ \underbrace { \mathrm { d i a g } ( \mathbf { v } ^ { \top } \boldsymbol { \Phi } _ { x } ^ { + } \mathbf { w } ) } _ { \nu _ { x } } - \underbrace { ( P ^ { - 1 } \sum _ { k = 0 } ^ { P - 1 } \mathbf { v } _ { k } ) ( P ^ { - 1 } \sum _ { k = 0 } ^ { P - 1 } \mathbf { w } _ { k } ) } _ { \nu _ { m e a n } } \Bigg ] - \lambda \mathbf { u } _ { x } ,
$$

We can expand the middle term $\nu _ { x }$ by

$$
\begin{array} { r } { \nu _ { x } = \mathbf { v } ^ { \top } ( F ^ { - 1 } \Lambda ^ { x } F ) \mathbf { w } = ( F ^ { \ast } \mathbf { v } ) \Lambda ^ { x } ( F \mathbf { w } ) = ( 1 , \omega ^ { x } , \cdot \cdot \cdot , \omega ^ { x ( P - 1 ) } ) ( \overline { { F \mathbf { v } } } \odot F \mathbf { w } ) , } \end{array}
$$

At the same time, the Fourier transform of $\nu _ { m e a n }$ will be absorbed into the zero-frequency component, because it is independent of the index x.

Multiplying on the left by F, we have:

$$
( F \mathbf u ) _ { r } = C ( \overline { { F \mathbf v } } ) _ { r } ( F \mathbf w ) _ { r } - \lambda \mathbf u _ { r } , \quad ( r \neq 0 ) ,
$$

Similarly, we find that for each frequency $r \neq 0$

$$
\begin{array} { r } { \dot { \widehat { \mathbf { u } } } _ { r } \approx C \widehat { \mathbf { w } } _ { r } \widehat { \widehat { \mathbf { v } } _ { r } } - \lambda \widehat { \mathbf { u } } _ { r } , \qquad \dot { \widehat { \mathbf { v } } } _ { r } \approx C \widehat { \mathbf { w } } _ { r } \widehat { \widehat { \mathbf { u } } _ { r } } - \lambda \widehat { \mathbf { v } } _ { r } , \qquad \dot { \widehat { \mathbf { w } } } _ { r } \approx C \widehat { \mathbf { u } } _ { r } \widehat { \mathbf { v } } _ { r } - \lambda \widehat { \mathbf { w } } _ { r } . } \end{array}
$$

where $\hat { \mathbf { v } }$ means DFT and v¯ means conjugate. That is, the system is decoupled at each nonzero frequency. Each 3-variable system $\hat { \mathbf { u } } _ { r } , \hat { \mathbf { v } } _ { r } , \hat { \mathbf { w } } _ { \ i }$ corresponding to frequency r will evolve independently.

The next theorem indicates that the conditions for neurons to grow depend on the initialization, and weight decay plays a filtering role. Moreover, growing neurons always develop a dominant frequency and adhere to specific phase patterns.

Proposition 2 (Dominant frequency and phase alignment). Fix a hidden unit d. For each nonzero frequency r, let $U _ { r } ( t ) = \widehat { U } _ { d , r } ( t ) , V _ { r } ( t ) = \widehat { V } _ { d , r } ( t )$ , and $W _ { r } ( t ) = \widehat { W ^ { d } } _ { r } ( t )$ . Assume $C > 0$ and that the three mode amplitudes remain nonzero up to the first singularity. Let $s _ { r } ^ { \star }$ be the blow-up time ofthe corresponding undamped system. If there exists some $r , s _ { r } ^ { \star } < C / \lambda$ and $s _ { r } ^ { \star } = \mathrm { m i n } _ { i \neq 0 } \{ s _ { i } \}$ , then r is the first frequency to blow up. Hence it makes the dominant frequency. Moreover, it is going to hold arg $U _ { r } ( t ) + \arg V _ { r } ( t ) - \arg W _ { r } ( t )  0 ,$ , when t approaches the blow-up time.

Through the proposition, we can summarize the mechanism of weight decay filtering. The argument $s _ { r } ^ { \star }$ depends on the initial weights of a group of neurons $\mathbf { u } _ { d } , \mathbf { v } _ { d } , \mathbf { w } _ { d }$ , and weight decay provides a unified survival threshold, and the competition of patterns and the initial random asymmetry determine which frequencies cross this threshold.

![](images/4e6f30e7499531523ff19c7b906c8319e29ca7a76bc175e05326fd692ffb6c1b.jpg)  
Figure 1: Accuracy-Epoch curves of $\lambda = 0 , 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 }$ . For $\lambda = 0$ , model reaches 100% on clean and noise, but 0 on test, overfitting. For $\lambda = 1 0 ^ { - 4 }$ , model reaches almost 100% on clean, noise and test, generalization. For $\lambda = 5 \times 1 0 ^ { - 4 }$ , model forgets the noise. And for $\lambda = 1 0 ^ { - 3 }$ , model cannot converge.

## 4.2 Estimation of noise

The actual training dataset is 80% of the complete data with α noise inside. The following theorem will provide an approximation to bridge the gap caused by sampling.

Theorem 2 (Estimation of noise). Under Proposition 1, sample D as above experiments with training data ratio ρ and uniform-label noise ratio α. Take $U _ { \star } ^ { x } : = \mathrm { d i a g } ( V \Phi _ { x } ^ { + } W ) - \overline { { V } } \odot \overline { { W } }$ as the main term under thefull-addition signature. Then we estimate

$$
\dot { U } ^ { x } - \frac { \eta c _ { 2 } ( 1 - \alpha ) } { P } U _ { \star } ^ { x } + \lambda U ^ { x } = \mathcal { E } _ { U } ^ { x } + \delta ,
$$

where $\| \delta \| _ { \infty } \lesssim \epsilon ^ { 3 } .$ . The sampling error is controlled by

$$
\mathbb { E } _ { \mathcal { D } } \mathcal { E } _ { \bullet } = 0 , \quad \mathrm { V a r } _ { \mathcal { D } } ( \mathcal { E } _ { \bullet } ) \lesssim \frac { [ ( 1 - \rho ) + \alpha ] \eta ^ { 2 } } { \rho P ^ { 3 } } \left( c _ { 1 } ^ { 2 } \epsilon ^ { 2 } + c _ { 2 } ^ { 2 } \epsilon ^ { 4 } \right) .
$$

Similar results also holdfor V and $W .$

## 5 Validation on Fourier frequency

We regard all the column vectors of W or the row vectors of U and V as neurons, since the output of the neural network is $\begin{array} { r } { f ( a , b ) = \sum _ { d = 1 } ^ { D } W ^ { d } ( U _ { d } e _ { a } + V _ { d } e _ { b } ) } \end{array}$ . In analysis, we assume that they evolve approximately independently. We denote the concatenation of the d-th row of U, V , the d-th column of W by $\theta _ { d } = ( U _ { d } , V _ { d } , W ^ { d } )$ and the collection of $\theta _ { d }$ by V. We consider Fourier transform of some neuron $u \in \mathbb { R } ^ { P }$ , it is $\begin{array} { r } { \widehat { u } _ { k } = \sum _ { x = 0 } ^ { P - 1 } u _ { x } \mathrm { e } ^ { - \mathrm { i } 2 \pi k x / P } } \end{array}$ for $k = 0 , \ldots , P - 1$ . We define the dominant frequency and sparsity by

$$
\tau ( u ) : = \mathrm { a r g m a x } _ { k \neq 0 } | \widehat { u } _ { k } | , \quad \rho ( u ) = \frac { 2 | \widehat { u } _ { \tau ( u ) } | ^ { 2 } } { \sum _ { k = 1 } ^ { P - 1 } | \widehat { u } _ { k } | ^ { 2 } } \in [ 0 , 1 ] .
$$

It is noted that frequency r and $P - r$ are regarded as the same under Fourier transform.

![](images/2eff74c8cff6437381e4bbfe0e5e839991156a461890c6946c92266d0cdae5e9.jpg)  
Figure 2: (A) Scatter between Fourier sparsity and the norm of neurons at epoch 10, 500, 1000, 2000. (B) The highest and lowest sparsity 100 neurons are selected, denoted by $\mathcal { V } _ { t }$ and $\mathcal { V } _ { b }$ . We observe $\begin{array} { r } { \ell _ { \mathrm { d r i v e } } ^ { \mathrm { t o p } } = \sum _ { \theta \in \mathcal { V } _ { t } } \overline { { \ell _ { \mathrm { d r i v e } } } } } \end{array}$ and similarly for $\ell _ { \mathrm { d e c a y } } ^ { \mathrm { t o p } } , \ell _ { \mathrm { d r i v e } } ^ { \mathrm { b o t } } , \ell _ { \mathrm { d e c a y } } ^ { \mathrm { b o t } }$ (C) The clean/noise dataset contributions to the drive. It is $\begin{array} { r } { \ell _ { \mathrm { d r i v e } } ^ { \mathrm { t o p , c l e a n } } = \sum _ { \theta \in \mathcal { V } _ { t } } \overline { { \ell _ { \mathrm { d r i v e } } } } \left| \mathcal { D } _ { \mathrm { c l e a n } } \right. } \end{array}$ and similarly for others.

Experiments. We conducted experiments under different weight decay settings, with the same $\sigma ( x ) = \mathrm { G E L U } ( x )$ , noise ratio $\alpha = 0 . 1$ and initialization scale $\gamma = 0 . 5$ , in Figure 1. Additional experiments with other activation functions are reported in Appendix B,D. It is clearly evident from Figure 1 that weight decay has a significant impact on the direction of the training process. We observe that $\lambda = 0 . 0 0 0 5$ acts as a regularization. That is, the model generalizes well while ignoring all the noise.

We naturally derived from weight decay to consider observing the norms of each group of neurons in the network. The growth of the norms can be described as:

$$
\frac { 1 } { 2 } \frac { \mathrm { d } } { \mathrm { d } t } \| \theta _ { d } \| ^ { 2 } = \underbrace { - \eta \langle \theta _ { d } , \nabla _ { d } L \rangle } _ { \mathrm { C E } \mathrm { d r i v e } } - \underbrace { \lambda \| \theta _ { d } \| ^ { 2 } } _ { \mathrm { d e c a y } } .
$$

In Figure 2A, we present the evolution of the norm and frequency sparsity of the neurons. It can be observed that due to weight decay, the norm of most neurons is continuously reduced to 0, while a few neurons can grow. Meanwhile, all the new-grown neurons have relatively high sparsity.

In Figure 2B, we report that norm growth is concentrated on high-sparsity neurons, and the growth of the norm and the improvement of the accuracy occur simultaneously in terms of time, according to Figure 1. We also compare separately on clean and noisy examples in Figure 2C. It shows that CE–drive term of high-sparsity neurons is contributed primarily by clean data. And these driving forces are almost completely unrelated to the noise in the data.

Cosine Similarity Check. Theorem 1 and 2 provide two estimates for the real gradient flow. We conducted numerical experiments in Appendix C.1, comparing the cosine similarity between the estimated gradient and the actual gradient. The results show that it can achieve cosine-similarity $\geq 0 . 9$ with the actual gradient.

We find that the dominant frequencies of $U _ { d } , V _ { d } .$ , and $W ^ { d }$ tend to agree if the neuron reaches high sparsity. If the common frequency is τ, we extract the components at the dominant frequency by $( \mathbf { u } _ { d } , \mathbf { v } _ { d } , \mathbf { w } _ { d } ) = ( ( { \widehat { U _ { d } } } ) _ { \tau } , ( { \widehat { V _ { d } } } ) _ { \tau } , ( { \widehat { W ^ { d } } } ) _ { \tau } )$ , then their phases will tend to satisfy $\arg ( \mathbf { u } _ { d } ) + \arg ( \mathbf { v } _ { d } ) =$ $\mathrm { a r g } ( \mathbf { w } _ { d } )$ (mod 2π). Figure 3 shows that frequency matching and phase alignment emerge together during training, which is consistent with the result of Prop 2.

![](images/a7ace29ab93611eee5ad47eec21b255c1c7364bae6bbb614ba07ce636b9b363a.jpg)

Figure 3: Frequency matching and phase alignment. In the odd columns, the horizontal coordinate is the dominant nonzero Fourier frequency of w and the vertical coordinate is its frequency sparsity; in the even columns, the coordinates are $\arg ( \mathbf { u } ) + \arg ( \mathbf { v } )$ mod $2 \pi$ and $\mathrm { a r g } ( \mathbf { w } )$ mod 2π, respectively. Red points indicate exact agreement of the dominant frequencies of $\mathbf { u } _ { d } , \mathbf { v } _ { d } .$ , and ${ \mathbf { w } } _ { d } ,$ , while blue points indicate a mismatch; dashed lines show phase equality.  
![](images/9a50adf5d93c02bfec5cb5a2390bed92e9cfcef1c94be020a1c97e84dd6c29e5.jpg)  
Figure 4: Early(< 100 epochs) clean and noise loss dynamics. α means the noise ratio. Noise loss always decreases more rapidly.

Full-batch GD check. Using the full addition table, a GELU-activated network, and shared initialization, we compare exact CE training with the truncated model. We simulate both systems over 50 independently evolved neurons and report the dominant frequencies to which they ultimately converge, as well as the frequency sparsity. For the valid neurons, the dominant frequencies predicted by the truncated model exhibit 93% agreement with the converged dominant frequencies of the actual GELU-network. Experimental details are provided in the Appendix C.2.

## 6 Application of Probability Signature

## 6.1 The puzzle of early noise

We identify a puzzle in the early training: uniformly random label noise consistently exhibits a faster initial loss decrease than clean data. It was also recorded by (Liu & Lu, 2026) but has not been resolved. From Figure 4, this phenomenon holds under almost any ratio of noise. Although this may appear inconsistent with the frequency principle (Xu et al., 2020), which states that high-frequency data components are fitted later and low-frequency components earlier. In the discrete setting, the similar redefinition of frequency will be the probability that the predicted label will change while changing one of the input tokens. In this section, we employ probability signatures to quantify this speed difference.

In the small-initialization regime, the loss derivative is approximately $\begin{array} { r } { \dot { L } \propto - \sum _ { i = 1 } ^ { N } \sigma ( U ^ { x _ { i } } + } \end{array}$ $\begin{array} { r } { V ^ { y _ { i } } \big ) ^ { \top } \left( \sum _ { j = 1 } ^ { N } \mathbf { 1 } _ { \left\{ z _ { j } = z _ { i } \right\} } \sigma ( U ^ { x _ { j } } + V ^ { y _ { j } } \big ) \right) } \end{array}$ , with sampling $( x _ { i } , y _ { i } , z _ { i } )$ from the training distribution. We ignore weight decay because we are comparing the loss of different parts within the same training. From this, we consider counting the numbers of sample pairs that share common coordinates.

For pairs with distinct coordinates define $\kappa ( 0 ) = \mathbb { E } [ \sigma ( \alpha + \beta ) ^ { \top } \sigma ( \gamma + \delta ) ]$ , while for pairs sharing one coordinate define $\kappa ( 1 ) = \mathbb { E } [ \sigma ( \alpha + \beta ) ^ { \top } \sigma ( \alpha + \gamma ) ]$ , where $\alpha , \beta , \gamma , \delta \sim \mathcal { N } ( 0 , 1 / D )$ are i.i.d. vectors. Generally, we have $\kappa ( 1 ) > \kappa ( 0 )$ (see appendix F.1). Let $n _ { z }$ be the number of examples labeled z and define $K _ { 0 } = \textstyle \sum _ { z } n _ { z } ^ { 2 }$ . Then the following theorem can compare the early loss for full training dataset. Note that $\varphi ^ { ( \bar { a } ) } , \varphi ^ { ( b ) }$ here are understood as matrices.

Theorem 3 (Early loss decrease). Suppose the training dataset contains every distinct pairs $( a , b ) \in \Omega \times \Omega$ , and the model structure is the same as that mentioned before. For different label distribution, the early-time loss-slope is positively correlated with

$$
\begin{array} { r } { \mathbb { E } \dot { L } _ { C E } \propto - \kappa ( 0 ) K _ { 0 } - ( \kappa ( 1 ) - \kappa ( 0 ) ) K _ { 1 } , \quad K _ { 1 } = P ^ { 2 } ( \| \varphi ^ { ( a ) } \| _ { 2 } ^ { 2 } + \| \varphi ^ { ( b ) } \| _ { 2 } ^ { 2 } - 2 ) . } \end{array}
$$

Proof. See Appendix E.5.

For modular addition, conditioning on any fixed value of a (or b), the label is uniform on Ω. Hence,

$$
\varphi ^ { ( a ) } = \varphi ^ { ( b ) } = \frac { 1 } { P } { \bf 1 } _ { P } { \bf 1 } _ { P } ^ { \top } , \qquad ( K _ { 0 } , K _ { 1 } ) = ( P ^ { 3 } , 0 ) ,
$$

For a fully random labeling, let $Z _ { x , y } = \mathbb { P } _ { \pi } ( Z = y ~ | ~ X = x )$ . The P pairs with first coordinate x receive independent labels, so

$$
Z _ { x , y } \sim P ^ { - 1 } \mathrm { B i n o m i a l } ( P , P ^ { - 1 } ) .
$$

According to the binomial distribution, we get $\mathbb { E } [ Z _ { x , y } ^ { 2 } ] = 2 P ^ { - 2 } - P ^ { - 3 }$ . It further gives $\mathbb { E } \left[ K _ { 0 } \right] =$ $P ^ { 3 } + P ^ { 2 } - P , \mathbb { E } \left[ K _ { 1 } \right] = 2 ( P ^ { 2 } - P )$ , which are both greater than those in modular-addition case. Thus, uniform random labels can produce a faster early loss decrease than modular-addition labels.

Next we consider the mixture of clean and noise. Without loss of generality, we assume sampling in complete $\Omega \times \Omega$ and dividing it into clean samples (modular addition) and noise (uniform random label). The loss for one example $( x _ { i } , y _ { i } , z _ { i } )$ can be regarded as

$$
\dot { L } _ { i } \approx - \kappa ( 0 ) n _ { z _ { i } } ^ { 2 } - ( \kappa ( 1 ) - \kappa ( 0 ) ) \Delta _ { i } ; \Delta _ { i } : = \sum _ { j = 1 , j \neq i } ^ { N _ { \mathrm { t r a i n } } } \mathbf { 1 } _ { z _ { i } = z _ { j } } ( \mathbf { 1 } _ { x _ { i } = x _ { j } } + \mathbf { 1 } _ { y _ { i } = y _ { j } } ) .
$$

Here the coefficient $\Delta _ { i }$ is regarded as $\Delta _ { \mathrm { c l e a n } }$ for a clean sample, or $\Delta _ { \mathrm { n o i s e } }$ for a noise. Taking expectations, the counts satisfy $\overline { { \Delta _ { \mathrm { c l e a n } } } } = 2 \alpha ( 1 - P ^ { - 1 } )$ , and $\overline { { \Delta _ { \mathrm { n o i s e } } } } = 2 ( 1 - P ^ { - 1 } )$ . Consequently, $\overline { { \Delta _ { \mathrm { n o i s e } } } } - \overline { { \Delta _ { \mathrm { c l e a n } } } } = 2 ( 1 - \alpha ) ( 1 - P ^ { - 1 } ) > 0$ , so the loss of noise decreases more rapidly. This prediction is consistent with the experimental observations and can also be applied to other forms of noise. In Appendix F.1 and F.2, we conducted additional experiments, thereby verifying that the theoretical predictions presented here are reliable.

![](images/a079f5761c7e399a823ce52f5294dacbe01fc939cf48da8be8972536c5a62c16.jpg)

![](images/3d3abe0c6fabcaed2045d070026e209d4d5bbc097de894a4e6b7ccbe45013836.jpg)

![](images/8677fbbe247ed36e0a51a5d590e44a9fe707483cddc5fe6a51178ff5007ac18f.jpg)

![](images/35c1ad1d2c2bfde5e93febbb5b11b9d4685b0a591871c9432d0da8f0be99cdde.jpg)  
Figure 5: Hadamard frequency structure in the XOR experiment. For sampled neurons at epochs 0, 50, 200, the horizontal axis is $\lVert \theta _ { d } \rVert _ { 2 }$ and the vertical axis is the sparsity of $w _ { d } .$ . Red points denote neurons whose three dominant frequencies agree. The right panel reports test accuracy and the number of high-sparsity, frequency-matched neurons; high-sparsity means $\rho > 0 . 7$

## 6.2 Extension of the Method: XOR as an example

The Fourier structure observed in modular addition has previously been identified and functionally validated from trained networks. However, these results primarily provide a mechanistic description of the circuit after it has formed, rather than a general forward description to predict the representation basis associated with a new algebraic operation.

Our probability-signature analysis suggests such a prescription. In the small-initialization regime, the leading parameter dynamic is governed by linear operators constructed from conditional statistics of the training distribution. When these probability-signature operators admit a common eigen basis, transforming the parameters into this basis approximately decouples the leading-order dynamics into independent modes. Thus, the relevant representation basis need not be assumed to be the Fourier basis in advance, instead it can be predicted by finding the common eigenvectors of the joint signature.

This observation provides a natural route for transferring the analysis beyond modular addition. Starting from the Fourier transform, we notice that relevant probability signatures can be regarded as the operators on the Abelian group, whose common eigenvectors are given by the corresponding group characters whenever the operators are simultaneously diagonalizable.

For example, consider the task on $\Omega = \{ 0 , \ldots , 6 3 \}$ , with $F ( a , b ) = a \operatorname { x o r } b$ . Its joint signature is

$$
\Phi _ { x } ^ { ( a ) } = \sum _ { y , z \in \Omega } \mathbb { P } ( Z = z , Y = y \mid X = x ) e _ { y } \otimes e _ { z } = \frac { 1 } { 6 4 } \sum _ { y \in \Omega } E _ { ( y , x \mathrm { x o r } y ) } .
$$

Hadamard basis simultaneously diagonalize these operators: there is an orthogonal basis $H =$ $[ H _ { 0 } , H _ { 1 } , \dots , H _ { 6 3 } ]$ such that $H ^ { - 1 } \Phi _ { x } ^ { ( a ) } H$ is diagonal for every $x \in \Omega$ . The properties of H can be referred to in Appendix E.6. Therefore, we claim that after training on XOR, the weights will evolve into a shape similar to the Hadamard basis.

Validation. We write $\begin{array} { r } { u = \sum _ { k = 0 } ^ { 6 3 } E _ { u , k } H _ { k } } \end{array}$ , and define its sparsity by

$$
\rho _ { u } = \frac { \operatorname* { m a x } _ { k } | E _ { u , k } | ^ { 2 } } { \sum _ { k } | E _ { u , k } | ^ { 2 } } .
$$

The existence of this orthogonal basis allows the above analysis to be applied to XOR. Based on the redefined frequency sparsity, in Figure 5 we present the observations regarding the frequency matching, as well as the high correlation between frequency sparsity and norm. The observed birth and growth of high-sparsity neurons supports our claim.

## 7 Conclusion and Discussion

Using the probability-signature framework, we provide a unified data-side account of three phenomena. First, for modular addition, the conditional-distribution operators induced by the training data recover the Fourier basis as their common eigenbasis and yield approximately decoupled leading-order dynamics, thereby accounting for the emergence of dominant frequencies, frequency matching, and phase alignment. Second, under uniformly random label noise, conditional label collisions among examples sharing an input coordinate explain why corrupted examples can exhibit a faster initial loss decrease than clean examples. Third, the same operator viewpoint suggests a route for predicting the special structures that neurons may develop on datasets generated by other binary operations: for XOR, the joint signature identifies the Hadamard basis as the relevant candidate coordinate system, in agreement with the weight patterns observed after training. Limitations. We approximate the empirical gradient flow using idealized dynamics derived from discrete, full-distribution probability signatures, together with finite-sample and label-noise corrections. Moreover, our dynamical analysis concerns the dominant gradient terms within a particular training regime; we do not claim that this approximation remains valid throughout the entire training process. Further discussion of the scope and empirical accuracy of this approximation is provided in Appendix C.1.

## References

Devansh Arpit, Stanislaw Jastrzkebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Max S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In International Conference on Machine Learning, 2017.

Darshil Doshi, Aritra Das, Tianyu He, and Andrey Gromov. To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets. In International Conference on Learning Representations, 2024.

Andrey Gromov. Grokking modular arithmetic, 2023.

Jianliang He, Leda Wang, Siyu Chen, and Zhuoran Yang. On the mechanism and dynamics of modular addition: Fourier features, lottery ticket, and grokking, 2026.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks, 2020. URL https://arxiv.org/abs/1806.07572.

Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan. Grokking as the transition from lazy to rich training dynamics. In International Conference on Learning Representations, 2024.

Linyu Liu and Pinyan Lu. Unveiling memorization-generalization coexistence: A case study on arithmetic tasks with label noise, 2026. URL https://arxiv.org/abs/2605.18022.

Sheng Liu, Jonathan Niles-Weed, Narges Razavian, and Carlos Fernandez-Granda. Early-learning regularization prevents memorization of noisy labels. In Advances in Neural Information Processing Systems, 2020.

Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning, 2022.

Ziming Liu, Eric J. Michaud, and Max Tegmark. Omnigrok: Grokking beyond algorithmic data. In International Conference on Learning Representations, 2023.

Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. In Advances in Neural Information Processing Systems, 2023.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations, 2023.

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022.

Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua Susskind. The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon, 2022.

Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier analysis sheds light on deep neural networks. Communications in Computational Physics, 28(5):1746–1767, 2020.

Junjie Yao and Zhi-Qin John Xu. Probability signature: Bridging data semantics and embedding structure in language models, 2025. URL https://arxiv.org/abs/2509.20124.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization, 2017. URL https://arxiv.org/abs/ 1611.03530.

![](images/837b0e510cf400ae83609eb5ab657ff423cf60b76425642ed13b09ddb927f3d8.jpg)  
Figure 6: Accuracy curves using Adam, with weight decay $\lambda = 0 , 1 0 ^ { - 5 } , 5 \cdot 1 0 ^ { - 4 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 }$ and initialization scale γ = 0.1, 0.5, 1.0.

![](images/165ce60a45b36dd69c63fdc332eeb997bfd4331681f951b83e6e524c80235ef7.jpg)  
Figure 7: Loss curves using Adam.

## A Model and training details

The neural network structure of this article can be written as

$$
f ( a , b ) = W \sigma ( U e _ { a } + V e _ { b } ) = \sum _ { d = 0 } ^ { D - 1 } W ^ { d } ( U _ { d , a } + V _ { d , b } ) ,
$$

Here, $U$ and V are input embedding matrices in R $P { \times } P$ , and $W$ is the output matrix in R $P { \times } D$ , with $D$ representing the width of the hidden layer $( D = 2 0 0 0 , P = 1 1 3 )$ . The output probability can be expressed as

![](images/5cbda60ae99b80db32364ef442fbc90995c4a68db8d0cb630dc81b5c259b0181.jpg)  
Figure 8: Accuracy curves using AdamW.

$$
p ( X _ { i } ) _ { z _ { i } } = \frac { \exp ( f ( a , b ) _ { z _ { i } } ) } { \sum _ { k = 0 } ^ { P - 1 } \exp ( f ( a , b ) _ { k } ) }
$$

The loss of the training data is given by the cross-entropy:

$$
L _ { \mathrm { C E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } - \log p ( X _ { i } ) _ { z _ { i } }
$$

The total loss after adding weight decay is:

$$
L = L _ { \mathrm { C E } } + \frac { \lambda } { 2 } \left( \| U \| _ { F } ^ { 2 } + \| V \| _ { F } ^ { 2 } + \| W \| _ { F } ^ { 2 } \right)
$$

The learning rate strategy we employ is warmup + cosine decay:

$$
\begin{array}{c} \eta _ { t } = \left\{ \eta \cdot \frac { t } { T _ { \mathrm { w a r m u p } } } , \qquad & { t < T _ { \mathrm { w a r m u p } } } \\ { \eta \cdot \frac { 1 } { 2 } \left( 1 + \eta _ { 0 } + ( 1 - \eta _ { 0 } ) \cos \frac { ( t - T _ { \mathrm { w a r m u p } } ) \pi } { T _ { \mathrm { o p i m } } - T _ { \mathrm { w a r m u p } } } \right) , \quad t \geq T _ { \mathrm { w a r m u p } } } \end{array} \right.
$$

The minimum learning rate $\eta _ { 0 } ~ = ~ 1 0 ^ { - 5 }$ , and the initial learning rate $\eta ~ = ~ 1 0 ^ { - 3 }$ if using the Adam/AdamW optimizer. Experiments using Adam can generally be reproduced under the AdamW optimizer with about 1000x weight decay. Figure $6 { , } 7$ compares the specific accuracy or loss curves with Adam under different weight decay and initialization. If the weight decay is too small or absent, there will be no generalization. If the weight decay is too large, it may cause the training to fail to converge, and at this point, all parameters will be compressed to zero. With an appropriate weight decay, using a smaller initial value can shorten the training convergence time. Moreover, check Figure 8,9 for AdamW. Compared with Adam, greater weight decay needs to be used to achieve a similar effect, and we find that in this problem the Adam optimizer can provide better regularization, that is, a reduction of noise.

![](images/42c4a4af4e9329c1823724bbc60a7715cc23f50ce84f74a66c177621dc45005a.jpg)  
Figure 9: Loss curves using AdamW.

![](images/3bea05e48a078d783efd537c5a1b7f20109d0d97fdeddafad7516bb24d2a6ebc.jpg)  
Figure 10: (a)–(d): Final accuracy using activation functions $\sigma ( x ) = x$ (linear), $\sigma ( x ) = x ^ { 2 }$ (quadratic), $\sigma ( x ) = \operatorname { t a n h } ( x )$ , or $\sigma ( x ) = \operatorname* { m a x } ( 0 , x )$ , under different weight decay.

## B Discussion on Activation Functions

We use $\sigma ( x ) = \mathrm { G E L U } ( x )$ as the activation function in the main text, which is smooth near zero. The analytical method in our text is based on the expansion of the activation function, hence we conducted the same experiments with other activation functions, such as $\sigma ( x ) = x$ (linear), $\sigma ( x ) = x ^ { 2 } , \sigma ( x ) = \operatorname { t a n h } ( x )$ , and $\sigma ( x ) = \mathrm { R e L U } ( x )$ . The experimental results are presented in Figure 10. From the figure, it can be seen that using the linear activation is almost impossible to learn well on the training set. Using tanh(x) as the activation function results in very low test generalization. However, using the quadratic or ReLU activation function can achieve great test generalization under an appropriate weight decay.

Let’s review the mathematical derivation of the probability signature presented in the main text, we use the Taylor expansion of the activation function,

$$
\sigma ( h ) = 0 + \sigma ^ { \prime } ( 0 ) h + \frac { 1 } { 2 } \sigma ^ { \prime \prime } ( 0 ) h ^ { 2 } + o ( h ^ { 2 } ) ,
$$

to derive the probability signature. We can categorize all the expanded terms into 1-order or 2-order terms, based on whether they come from the 1st-order expansion or the 2nd-order expansion. In Theorem 1, the 1-order terms are eliminated because of the assumption of uniform marginal distribution. As a result, the second-order derivative term is the key to this problem.

The theoretical explanation matches our observation of poor generalization with linear/Tanh. Because tanh $( x ) = x - x ^ { 3 } / 3 + \cdot \cdot \cdot$ · does not contain the second-order derivative term. Meanwhile, the quadratic activation function contains the second-order derivative term, which does well in the test case. Besides, the success of RELU is not in contradiction to the analysis, because it is not differentiable near 0. We will discuss the RELU activation function in detail in Appendix D.

## C Experimental Details

## C.1 Gradient Flow Approximation

In this section, we present the experimental results with the aim of verifying the rationality of the approximation methods employed in Theorem 1 and Theorem 2.

By Theorem 1, we can approximate the gradient flow of $U ^ { x }$ as $\dot { U } ^ { x } \approx - \lambda U ^ { x } + c \cdot \mathrm { d i a g } ( V \Phi _ { x } ^ { ( a ) } W )$ where $\Phi _ { x } ^ { ( a ) }$ is the probability signature of the actual training dataset. We extract its main term,

$$
\begin{array} { r } { \hat { \nabla } _ { U ^ { x } } L _ { 1 } : = \mathrm { d i a g } ( V \Phi _ { x } ^ { ( a ) } W ) , } \end{array}
$$

as an approximation for the cross-entropy gradient. Flatten this vector, we get $\hat { \nabla } _ { \theta } L _ { 1 }$ , which is an approximation for the cross-entropy gradient $\nabla _ { \boldsymbol { \theta } } L$ . We compare the two gradients by cosine similarity

$$
\cos \frac { \langle \hat { \nabla } _ { \theta } L _ { 1 } , \nabla _ { \theta } L \rangle } { \| \hat { \nabla } _ { \theta } L _ { 1 } \| _ { 2 } \cdot \| \nabla _ { \theta } L \| _ { 2 } } ,
$$

as Approx 1 in Figure 11.

Similarly, by Theorem 2, we can approximate the gradient flow of $U ^ { x }$ as $\dot { U ^ { x } } \approx - \lambda U ^ { x } + c$ diag $( V \Phi _ { x } ^ { + } W )$ , where $\Phi _ { x } ^ { + }$ is the probability signature of the clean modular addition, it is,

$$
\Phi _ { x } ^ { + } = \frac { 1 } { P } \left( \begin{array} { c c c c } { { 1 } } & { { } } & { { } } & { { } } \\ { { } } & { { 1 } } & { { } } & { { } } \\ { { } } & { { } } & { { \ddots } } & { { } } \\ { { } } & { { } } & { { } } & { { 1 } } \\ { { 1 } } & { { } } & { { } } & { { } } \end{array} \right) ^ { x }
$$

We also extract and flatten its main term as $\hat { \nabla } _ { \theta } L _ { 2 }$ , comparing which with the cross-entropy gradient $\nabla _ { \boldsymbol { \theta } } L$ by cosine similarity, as Approx 2 in Figure 11. The experimental results show that both the approximated gradients and the actual gradients have a high similarity of 90%. This verifies our theoretical analysis. Meanwhile, the control experiments using $\sigma ( x ) = x ^ { 2 }$ can serve as an ablation study, which demonstrates that using only the 2-order main terms is sufficient for achieving test generalization in the modular-addition task.

![](images/e1638f60b669a8a269a5efe58c35d81ce33b1f0a8df8c2b238b3532c7dbf40f3.jpg)  
Figure 11: Cosine similarity between the approximated gradients and the actual gradients. The blue curve uses the main term from Theorem 1 and the red one uses the main term from Theorem 2.

Discussion on the small initialization assumption. In the main text, the approximation of the gradient is required to be expanded within a small parameter region. At the same time, we claim that the neurons we are studying can always grow. This note will clarify that this is not in contradiction.

First of all, we acknowledge that there are higher-order terms in the gradient flow equation such as $\delta = O ( h ^ { 3 } )$ . Our theoretical derivation does not assume that these higher-order terms should be omitted. We focus on the dominant part. There exists a scale $h _ { 0 }$ . When the parameter is less than $h _ { 0 }$ , the second-order term becomes dominant, while when the parameter is greater than $h _ { 0 } ,$ the higher-order terms become dominant. It is noted that when weight decay exists, there is a non-high-order term of $- \lambda \theta$ in the gradient flow equation. Therefore, an appropriate λ will position the decay term between the above, and when the scale of the parameters expands beyond the region dominated by the second-order, the weight decay becomes the main term, thereby preventing the scale of the parameters from continuing to grow.

![](images/e170d29c0515245790a4b390c178b4ea8dccfb5c699ccb8b63d3914f81394591.jpg)

![](images/7564f8d01831c2e7adb75fe70c5c44f71bcbd835307310ff3badb3a3c62081cd.jpg)  
Figure 12: Neuron-wise frequency concentration in the truncated and original GELU systems. Heatmaps show the evolution of matched-frequency sparsity for individual neurons during training. Rows correspond to neurons, and columns represent training epochs. Numbers indicate the final winning Fourier mode, whereas ’x’ marks neurons whose winner had not stabilized by the specified threshold. Higher values indicate stronger concentration onto a matched nonzero frequency.

## C.2 Frequency Selectivity

We used complete addition tables with $P = 7 , D = 5 0$ , GELU activation, Gaussian scale $D ^ { - 1 / 2 }$ Both models started from identical initialized parameters. The exact update was

$$
\theta _ { k + 1 } = \theta _ { k } + 0 . 2 ( - \nabla L ( \theta _ { k } ) - 1 0 ^ { - 4 } \theta _ { k } ) ,
$$

and the truncated update replaced $- \nabla L$ by the field in Theorem 1 with $c = \sigma ^ { \prime \prime } ( 0 ) / 7$ and the same weight decay $\lambda = 0 . 0 0 0 1$

Recording every 25 epochs, the experiment tracked the evolution of the dominant frequency of each neuron, in Figure 12. We found that approximately 50% of the neurons developed a dominant frequency, and among this subset, a proportion of 0.93 converged to the same dominant frequency. Because shared initialization parameters were used, these neurons gave rise to the same frequency selection, which is consistent with our argument in Proposition 2.

## D Discussion on the ReLU activation

In this section, we will discuss the ReLU activation function in detail. The ReLU activation function is not differentiable at 0, which makes it difficult to analyze using Taylor expansion. However, we can still analyze the behavior of the network with ReLU activation by considering the piecewise linear nature of ReLU.

## D.1 Fourier-mode is able to do modular addition under RELU

We give a note on how 2L-MLP with ReLU activation do well in modular addition.

Proposition 3 (A sufficient Fourier condition for modular addition). Let P be an odd prime, and let $w _ { d } \in \mathbb { Z } _ { P } \setminus \{ 0 \} , 0 \leq d < D$ be prescribedfrequencies. For each hidden unit d, let $\phi _ { 1 , d } , \phi _ { 2 , d }$ be mutually independent random variables distributed uniformly on $[ 0 , 2 \pi )$ . Ifthe parameters are set to $\begin{array} { r } { U _ { d , x } = \cos \left( \frac { 2 \pi w _ { d } x } { P } + \phi _ { 1 , d } \right) , V _ { d , x } = \cos \left( \frac { 2 \pi w _ { d } x } { P } + \phi _ { 2 , d } \right) , W _ { c , d } = \cos \left( \frac { 2 \pi w _ { d } c } { P } + \phi _ { 1 , d } + \phi _ { 2 , d } \right) } \end{array}$ . then the prediction of the neural network is $( a + b )$ mod P with probability tending to 1 as $D \to \infty$

Proof. Fix $a , b \in \mathbb { Z } _ { P }$ , and write

$$
c _ { \star } = ( a + b ) \bmod P .
$$

For each hidden unit d and output coordinate c, define

$$
X _ { d , c } : = W _ { c , d } \operatorname* { m a x } \{ 0 , U _ { d , a } + V _ { d , b } \} .
$$

Then

$$
f _ { c } ^ { ( D ) } ( a , b ) = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } X _ { d , c } .
$$

Since $| W _ { c , d } | \le 1$ and

$$
0 \leq \operatorname* { m a x } \{ 0 , U _ { d , a } + V _ { d , b } \} \leq 2 ,
$$

we have

$$
- 2 \leq X _ { d , c } \leq 2 .
$$

We first compute the expectation of a single-unit contribution. Set

$$
\alpha _ { d } = \frac { 2 \pi w _ { d } a } { P } + \phi _ { 1 , d } , \qquad \beta _ { d } = \frac { 2 \pi w _ { d } b } { P } + \phi _ { 2 , d } ,
$$

and let

$$
\delta _ { d , c } = \frac { 2 \pi w _ { d } ( c - a - b ) } { P } .
$$

Because

$$
\phi _ { 1 , d } + \phi _ { 2 , d } = \alpha _ { d } + \beta _ { d } - \frac { 2 \pi w _ { d } ( a + b ) } { P } ,
$$

we obtain

$$
W _ { c , d } = \cos ( \alpha _ { d } + \beta _ { d } + \delta _ { d , c } ) .
$$

Thus,

$$
\mathbb { E } [ X _ { d , c } ] = \mathbb { E } \left[ \cos ( \alpha _ { d } + \beta _ { d } + \delta _ { d , c } ) \operatorname* { m a x } \{ 0 , \cos \alpha _ { d } + \cos \beta _ { d } \} \right] .
$$

Since $\alpha _ { d }$ and $\beta _ { d }$ are independent and uniformly distributed on [0, 2π), we may set

$$
s = \frac { \alpha _ { d } + \beta _ { d } } { 2 } , \qquad t = \frac { \alpha _ { d } - \beta _ { d } } { 2 } .
$$

Using

$$
\cos \alpha _ { d } + \cos \beta _ { d } = 2 \cos s \cos t ,
$$

we have

$$
\mathbb { E } [ X _ { d , c } ] = \frac { 1 } { ( 2 \pi ) ^ { 2 } } \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { 2 \pi } \cos ( 2 s + \delta _ { d , c } ) \left[ 2 \cos s \cos t \right] _ { + } d t d s .
$$

For fixed s,

$$
\int _ { 0 } ^ { 2 \pi } [ 2 \cos s \cos t ] _ { + } d t = 4 | \cos s | .
$$

Consequently,

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ X _ { d , c } ] = \frac { 1 } { \pi ^ { 2 } } \int _ { 0 } ^ { 2 \pi } { | \cos { s } | \cos ( 2 s + \delta _ { d , c } ) d s } } \\ { \displaystyle = \frac { \cos ( \delta _ { d , c } ) } { \pi ^ { 2 } } \int _ { 0 } ^ { 2 \pi } { | \cos { s } | \cos ( 2 s ) d s } } \\ { \displaystyle = \frac { 4 } { 3 \pi ^ { 2 } } \cos ( \delta _ { d , c } ) . } \end{array}
$$

In the second equality, the sine term vanishes by symmetry, and in the last equality we used

$$
\int _ { 0 } ^ { 2 \pi } | \cos s | \cos ( 2 s ) d s = \frac { 4 } { 3 } .
$$

Therefore,

$$
\mathbb { E } [ X _ { d , c } ] = \frac { 4 } { 3 \pi ^ { 2 } } \cos \left( \frac { 2 \pi w _ { d } ( c - a - b ) } { P } \right) .
$$

For the target coordinate $c _ { \star }$ , we have

$$
c _ { \star } - a - b \equiv 0 { \pmod { P } } ,
$$

and hence

$$
\mathbb { E } [ X _ { d , c _ { \star } } ] = \frac { 4 } { 3 \pi ^ { 2 } } .
$$

Now consider any $c \neq c _ { \star }$ . Since $P$ is prime,

$$
w _ { d } \not \equiv 0 { \pmod { P } } , \qquad c - a - b \not \equiv 0 { \pmod { P } }
$$

imply

$$
w _ { d } ( c - a - b ) \not \equiv 0 { \pmod { P } } .
$$

Define

$$
\rho _ { P } : = \operatorname* { m a x } _ { r \in \mathbb { Z } _ { P } \setminus \{ 0 \} } \cos \left( \frac { 2 \pi r } { P } \right) .
$$

Because $\mathbb { Z } _ { P } \backslash \{ 0 \}$ is finite and none of the corresponding angles is congruent to 0 modulo $2 \pi$

$$
\rho _ { P } < 1 .
$$

It follows that, for every d and every $c \neq c _ { \star }$ 9

$$
\mathbb { E } [ X _ { d , c } ] \leq \frac { 4 } { 3 \pi ^ { 2 } } \rho _ { P } .
$$

Hence, with

$$
\Delta _ { P } : = \frac { 4 } { 3 \pi ^ { 2 } } ( 1 - \rho _ { P } ) > 0 ,
$$

we have

$$
\mathbb { E } [ X _ { d , c _ { \star } } ] - \mathbb { E } [ X _ { d , c } ] \geq \Delta _ { P } \qquad { \mathrm { ~ f o r ~ a l l ~ } } d { \mathrm { ~ a n d ~ a l l ~ } } c \neq c _ { \star } .
$$

Averaging over d gives

$$
\mathbb { E } [ f _ { c _ { \star } } ^ { ( D ) } ( a , b ) ] - \mathbb { E } [ f _ { c } ^ { ( D ) } ( a , b ) ] \geq \Delta _ { P } \qquad { \mathrm { f o r ~ e v e r y ~ } } c \neq c _ { \star } .
$$

It remains to control the fluctuations around these expectations. Since the random phases are mutually independent across hidden units, the variables $\{ X _ { d , c } \} _ { d = 1 } ^ { D }$ are independent for each fixed c. Moreover, $X _ { d , c } \in [ - 2 , 2 ]$ . Therefore, Hoeffding’s inequality yields, for every $\varepsilon > 0$

$$
\mathbb { P } \left( \left| f _ { c } ^ { ( D ) } ( a , b ) - \mathbb { E } [ f _ { c } ^ { ( D ) } ( a , b ) ] \right| \geq \varepsilon \right) \leq 2 \exp \left( - \frac { D \varepsilon ^ { 2 } } { 8 } \right) .
$$

Taking $\varepsilon = \Delta _ { P } / 3$ and applying the union bound over the P output coordinates, we obtain

$$
\mathbb { P } \left( \operatorname* { m a x } _ { c \in \mathbb { Z } _ { P } } \left. f _ { c } ^ { ( D ) } ( a , b ) - \mathbb { E } [ f _ { c } ^ { ( D ) } ( a , b ) ] \right. \geq \frac { \Delta _ { P } } { 3 } \right) \leq 2 P \exp \left( - \frac { D \Delta _ { P } ^ { 2 } } { 7 2 } \right) .
$$

On the complementary event, for every $c \neq c _ { \star }$ ，

$$
\begin{array} { r l } & { f _ { c _ { \star } } ^ { ( D ) } ( a , b ) - f _ { c } ^ { ( D ) } ( a , b ) \geq \left( \mathbb { E } [ f _ { c _ { \star } } ^ { ( D ) } ( a , b ) ] - \mathbb { E } [ f _ { c } ^ { ( D ) } ( a , b ) ] \right) - \frac { 2 \Delta _ { P } } { 3 } } \\ & { \qquad \geq \displaystyle \frac { \Delta _ { P } } { 3 } > 0 . } \end{array}
$$

Thus, on this event, $c _ { \star }$ is the unique maximizing output coordinate. Consequently,

$$
\mathbb { P } \left( \arg \operatorname* { m a x } _ { c \in \mathbb { Z } _ { P } } f _ { c } ^ { ( D ) } ( a , b ) = c _ { \star } \right) \geq 1 - 2 P \exp \left( - \frac { D \Delta _ { P } ^ { 2 } } { 7 2 } \right) .
$$

Since the right-hand side converges to 1 as $D \to \infty$ , we conclude that

$$
\mathbb { P } \bigg ( \arg \operatorname* { m a x } _ { c \in \mathbb { Z } _ { P } } f _ { c } ^ { ( D ) } ( a , b ) = ( a + b ) \bmod P \bigg ) \longrightarrow 1 .
$$

## D.2 Experiment Details of ReLU activation

We conducted experiments using the ReLU activation function, and the results are shown in Figures 13 and 14. The experimental setup is consistent with that of the GELU activation function. The results indicate that the ReLU activation function exhibits similar behavior to GELU in terms of weight decay selectivity or frequency-phase scatter evolution. This suggests that the observed phenomena are not specific to those smooth activation function but rather reflect more general properties of neural network training dynamics.

![](images/caff49ae17d96076cc8a811abd065e298ba10b64c21507e75f1f29e293ce3a33.jpg)

![](images/6d30499caa1bfc5bbf77809abaf3c5a95cc51d42a650a20ca9f86de9910c1519.jpg)

![](images/894ff86543c9eba347f657059fd4579d098c1c8df009bcd9a1d12f0b05d8f60c.jpg)

Figure 13: Experiment of Figure 2 with ReLU.  
![](images/7e23462e0e32b22482712209e77af6c79986e8258fdc4c9bf3fe57c01c0b74e1.jpg)  
Figure 14: Experiment of Figure 3 with ReLU.

## D.3 Early Memorization

Current research Zhang et al. (2017); Jacot et al. (2020) have already provided a thorough discussion on the early memory behavior of neural networks. When use $\sigma ( x ) = \operatorname* { m a x } ( 0 , x )$ , we observe clear traces of memory at $( \lambda , \gamma ) = ( 0 . 0 0 1 , 0 . 1 )$ . The training curve from Figure 15 shows it reaches high accuracy on both clean and noise dataset first and then decreases due to the weight decay. Here, we merely use the gradient flow to offer a more intuitive understanding.

Consider a single example $( a , b , y )$ and let $h = U e _ { a } + V e _ { b }$ and $p = \operatorname { s o f t m a x } ( f ( a , b ) )$ . By the chain rule,

$$
\frac { \partial L } { \partial W _ { \nu , d } } = ( p - e _ { y } ) _ { \nu } \sigma ( h ) _ { d } + \lambda W _ { \nu , d } , \qquad \frac { \partial L } { \partial U _ { d , x } } = { \bf 1 } _ { \{ x = a \wedge h _ { d } > 0 \} } [ W ^ { \top } ( p - e _ { y } ) ] _ { d } + \lambda U _ { d , x } .
$$

Thus the gradient flow for the output weights is

$$
\frac { \mathrm { d } W _ { \nu , d } } { \mathrm { d } t } = ( e _ { y } - p ) _ { \nu } \sigma ( h ) _ { d } - \lambda W _ { \nu , d } .
$$

Near initialization, the softmax expansion gives

$$
\operatorname { s o f t m a x } ( f ) = { \frac { 1 } { P } } \mathbf { 1 } _ { P } + { \frac { 1 } { P } } ( f - { \overline { { f } } } ) + O ( \| f \| ^ { 2 } / P ) ,
$$

so the correct output coordinate is amplified while the others are weakly suppressed. In particular, the active coordinates satisfy $\begin{array} { r } { \frac { \mathrm { d } W _ { y } ^ { \top } } { \mathrm { d } t } \ : = \ : \sigma ( h ) - \lambda W _ { y } ^ { \top } } \end{array}$ , hence the weights will increase in those

Accuracy Curve

![](images/a304b1046b842fce5ac104ff09b7e0e0c0ef5798179bc1c5af1174a4f5679250.jpg)

Figure 15: Accuracy curve under RELU activation and $( \lambda , \gamma ) = ( 0 . 0 0 1 , 0 . 1 )$ , with 30% noise.  
![](images/aad9f5623e8ed2bd8e260600aca7e1f1a82f225925bd1cd8ad996f4ddb822304.jpg)  
Figure 16: Example-wise interpolation and the memory score. (a). For the illustrative input $3 + 1 = 4$ with $( D , P ) = ( 7 , 5 ) , \sigma ( x ) = \operatorname* { m a x } ( 0 , x )$ , the activation feature $K = e _ { 3 } + e _ { 5 }$ represents the initially active hidden coordinates; the displayed ±1 values denote random initialization signs rather than fixed parameter values. (b). Evolution of the memory score on clean and noisy training examples.

dimensions where $\sigma ( h ) > 0$ . For the example in Figure $1 6 ( \mathrm { a } ) , \sigma ( x ) = \operatorname* { m a x } ( 0 , x )$ , only $d = 3 , 5$ are active in the case, namely those satisfying $h _ { d } = U _ { d , a } + V _ { d , b } > 0$ . This is only a toy model of 1 training sample. It behaves like an interpolation, storing the data onto these random dimensions.

In the full-batch setting, the exact flows are

$$
\frac { \mathrm { d } W _ { c , d } } { \mathrm { d } t } = - \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { i } [ p _ { i , c } - \mathbf { 1 } _ { \left\{ y _ { i } = c \right\} } ] \operatorname* { m a x } ( 0 , U _ { d , a _ { i } } + V _ { d , b _ { i } } ) - \lambda W _ { c , d } ,
$$

$$
\frac { \mathrm { d } U _ { d , a } } { \mathrm { d } t } = - \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { i : a _ { i } = a } \mathbf { 1 } _ { \{ U _ { d , a _ { i } } + V _ { d , b _ { i } } > 0 \} } \sum _ { r } [ p _ { i , r } - \mathbf { 1 } _ { \{ y _ { i } = r \} } ] W _ { r , d } - \lambda U _ { d , a } ,
$$

$$
\frac { \mathrm { d } V _ { d , b } } { \mathrm { d } t } = - \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { i : b _ { i } = b } \mathbf { 1 } _ { \{ U _ { d , a _ { i } } + V _ { d , b _ { i } } > 0 \} } \sum _ { r } [ p _ { i , r } - \mathbf { 1 } _ { \{ y _ { i } = r \} } ] W _ { r , d } - \lambda V _ { d , b } .
$$

These equations show that each example $( X = ( a , b ) , y )$ strengthens its own output coordinate $( y )$ on its active hidden dimensions and suppresses the other output coordinates $( c \neq y )$ , with the latter effect smaller by approximately a factor $1 / P$ . This example-wise interpolation motivates the following memory score:

$$
\mathrm { m e m } : = \sum _ { i = 1 } ^ { N } \sum _ { d = 1 } ^ { D } \mathbf { 1 } _ { \left\{ U _ { d , a _ { i } } + V _ { d , b _ { i } } > 0 \right\} } \sum _ { c \neq y _ { i } } ( W _ { y _ { i } , d } - W _ { c , d } ) .
$$

From figure 16(b), we can observe that the memory score undergoes a rapid accumulation process in the early stages of training. Simultaneously, due to the weight decay’s effect of reducing the norm, it gradually decreases.

## E Proofs and Additional Details

## E.1 Calculation of the gradient flow

Proof. Let

$$
h _ { x , y } : = U e _ { x } + V e _ { y } = U ^ { x } + V ^ { y } , \qquad p _ { x , y } : = \mathrm { s o f t m a x } \big ( W \sigma ( h _ { x , y } ) \big ) .
$$

Without loss of generality, assume that the learning rate $\eta = 1$ . For the softmax cross-entropy loss associated with one sample $( x , y , z )$

$$
\ell ( x , y , z ) = - \log ( p _ { x , y } ) _ { z } ,
$$

the derivative with respect to the logits $f ( x , y ) = W \sigma ( h _ { x , y } )$ is

$$
\nabla _ { f } \ell = p _ { x , y } - e _ { z } .
$$

Therefore, by the chain rule,

$$
\nabla _ { h _ { x , y } } \ell = \mathrm { D i a g } \big ( \sigma ^ { \prime } ( h _ { x , y } ) \big ) \boldsymbol { W } ^ { \top } ( p _ { x , y } - e _ { z } ) .
$$

Since $U ^ { x }$ contributes to $h _ { x , y }$ only when the first position of the sample is x, the full-batch gradient flow gives

$$
\frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } \mathbb { E } _ { \pi } \left[ \sigma ^ { \prime } ( h _ { x , y } ) \odot W ^ { \top } ( e _ { z } - p _ { x , y } ) \vert x \right] .\tag{1}
$$

Because $\sigma$ is $C ^ { 3 }$ near zero and

$$
\sigma ( h ) = c _ { 1 } h + \frac { c _ { 2 } } { 2 } h ^ { 2 } + o ( h ^ { 2 } ) ,
$$

where the powers are understood componentwise, we have

$$
\sigma ^ { \prime } ( h ) = c _ { 1 } { \bf 1 } + c _ { 2 } h + o ( \| h \| ) .\tag{2}
$$

Moreover,

$$
\sigma ( h ) = c _ { 1 } h + O ( \| h \| ^ { 2 } ) .\tag{3}
$$

Let

$$
\Pi _ { P } : = I - \frac 1 P \mathbf { 1 } _ { P } \mathbf { 1 } _ { P } ^ { \top } .
$$

The softmax function satisfies

$$
\mathrm { s o f t m a x } ( 0 ) = \frac { 1 } { P } \mathbf { 1 } _ { P } ,
$$

and its Jacobian at zero is

$$
\begin{array} { l } { { \displaystyle { \cal D } \mathrm { s o f t m a x } ( 0 ) = \mathrm { D i a g } \left( \frac { 1 } { { \cal P } } { } ^ { 1 } { \cal { P } } \right) - \left( \frac { 1 } { { \cal P } } { } ^ { 1 } { \cal { P } } \right) \left( \frac { 1 } { { \cal P } } { } ^ { 1 } { \cal { P } } \right) ^ { \top } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { { \cal P } } { \cal I } - \frac { 1 } { { \cal P } ^ { 2 } } \mathbf { 1 } _ { { \cal P } } \mathbf { 1 } _ { { \cal P } } ^ { \top } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { { \cal P } } \Pi _ { { \cal P } } . } } \end{array}
$$

Consequently, for a logit vector g near zero,

$$
\mathrm { s o f t m a x } ( g ) = \frac { 1 } { P } { \bf 1 } _ { P } + \frac { 1 } { P } \Pi _ { P } g + O ( \Vert g \Vert ^ { 2 } ) .\tag{4}
$$

Taking $g = W \sigma ( h _ { x , y } )$ and using (3), we obtain

$$
g = c _ { 1 } W h _ { x , y } + O ( \vert \vert h _ { x , y } \vert \vert ^ { 2 } ) .
$$

Substituting this expression into (4) gives

$$
p _ { x , y } = \frac { 1 } { P } \mathbf { 1 } _ { P } + \frac { c _ { 1 } } { P } \Pi _ { P } W h _ { x , y } + O ( \| h _ { x , y } \| ^ { 2 } ) .\tag{5}
$$

Thus,

$$
\begin{array} { l } { { \displaystyle W ^ { \top } ( e _ { z } - p _ { x , y } ) = W ^ { \top } e _ { z } - \frac { 1 } { P } W ^ { \top } { \bf 1 } _ { P } } } \\ { { \displaystyle ~ - \frac { c _ { 1 } } { P } W ^ { \top } \Pi _ { P } W h _ { x , y } + O ( \| h _ { x , y } \| ^ { 2 } ) . } } \end{array}\tag{6}
$$

Combining (2) and (6), and retaining all constant and first-order terms in $h _ { x , y } .$ , yields

$$
\begin{array} { r l } & { \quad \sigma ^ { \prime } ( h _ { x , y } ) \odot W ^ { \top } ( e _ { z } - p _ { x , y } ) } \\ & { = \left( c _ { 1 } { \bf 1 } + c _ { 2 } h _ { x , y } + o ( \| h _ { x , y } \| ) \right) } \\ & { \qquad \odot \left( W ^ { \top } e _ { z } - \frac { 1 } { P } W ^ { \top } { \bf 1 } _ { P } - \frac { c _ { 1 } } { P } W ^ { \top } \Pi _ { P } W h _ { x , y } + O ( \| h _ { x , y } \| ^ { 2 } ) \right) } \\ & { = c _ { 1 } W ^ { \top } e _ { z } - \frac { c _ { 1 } } { P } W ^ { \top } { \bf 1 } _ { P } } \\ & { \qquad + c _ { 2 } h _ { x , y } \odot ( W ^ { \top } e _ { z } ) - \frac { c _ { 2 } } { P } h _ { x , y } \odot ( W ^ { \top } { \bf 1 } _ { P } ) } \\ & { \qquad - \frac { c _ { 1 } ^ { 2 } } { P } W ^ { \top } \Pi _ { P } W h _ { x , y } + \delta _ { x , y } , } \end{array}\tag{7}
$$

where $\delta _ { x , y }$ contains all higher-order terms. In particular, the product

$$
- \frac { c _ { 1 } c _ { 2 } } { P } h _ { x , y } \odot \left( W ^ { \top } \Pi _ { P } W h _ { x , y } \right)
$$

is quadratic in $h _ { x , y }$ and is therefore included in $\delta _ { x , y }$

We now take the conditional expectation of each term in (7). First,

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \boldsymbol \pi } \left[ c _ { 1 } W ^ { \top } e _ { z } - \frac { c _ { 1 } } { P } W ^ { \top } \mathbf { 1 } _ { P } \Big | \boldsymbol x \right] } \\ { \displaystyle = c _ { 1 } W ^ { \top } \left( \mathbb { E } _ { \boldsymbol \pi } [ e _ { z } \mid \boldsymbol x ] - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) . } \end{array}\tag{8}
$$

Since $h _ { x , y } = U ^ { x } + V ^ { y }$

$$
\begin{array} { r } { h _ { x , y } \odot ( W ^ { \top } e _ { z } ) = U ^ { x } \odot ( W ^ { \top } e _ { z } ) + V ^ { y } \odot ( W ^ { \top } e _ { z } ) . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } \left[ h _ { x , y } \odot ( W ^ { \top } e _ { z } ) \bigm | x \right] } \\ & { \qquad = U ^ { x } \odot \left( W ^ { \top } \mathbb { E } _ { \boldsymbol \pi } [ e _ { z } \mid x ] \right) + \mathbb { E } _ { \boldsymbol \pi } \left[ V ^ { y } \odot ( W ^ { \top } e _ { z } ) \mid x \right] . } \end{array}\tag{9}
$$

Furthermore,

$$
\mathbb { E } _ { \pi } [ h _ { x , y } \mid x ] = U ^ { x } + \mathbb { E } _ { \pi } [ V e _ { y } \mid x ] .\tag{10}
$$

It follows from (10) that

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } \left[ h _ { x , y } \odot ( { W ^ { \top } } \mathbf { 1 } _ { P } ) \middle | x \right] } \\ & { \qquad = ( { U ^ { x } } + \mathbb { E } _ { \boldsymbol \pi } [ { V e _ { y } } \mid x ] ) \odot ( { W ^ { \top } } \mathbf { 1 } _ { P } ) , } \end{array}\tag{11}
$$

and

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } \left[ W ^ { \top } \Pi _ { { P } } W h _ { x , y } \middle | x \right] } \\ & { \qquad = W ^ { \top } \Pi _ { { P } } W \left( U ^ { x } + \mathbb { E } _ { \boldsymbol \pi } [ V e _ { y } \mid x ] \right) . } \end{array}\tag{12}
$$

Substituting (8)–(12) into the gradient-flow equation (1), we obtain

$$
\begin{array} { r l r }   { \frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } \Biggl \{ c _ { 1 } W ^ { \top } ( \mathbb { E } _ { \boldsymbol { \pi } } [ e _ { z } \mid x ] - \frac { 1 } { P } \mathbf { 1 } _ { P } ) } \\ & { } & { + c _ { 2 } ( U ^ { x } \odot ( W ^ { \top } \mathbb { E } _ { \boldsymbol { \pi } } [ e _ { z } \mid x ] ) + \mathbb { E } _ { \boldsymbol { \pi } } [ V ^ { y } \odot ( W ^ { \top } e _ { z } ) \mid x ] ) } \\ & { } & { - \frac { C _ { 2 } } { P } ( U ^ { x } + \mathbb { E } _ { \boldsymbol { \pi } } [ V e _ { y } \mid x ] ) \odot ( W ^ { \top } \mathbf { 1 } _ { P } ) } \\ & { } & { - \frac { c _ { 1 } ^ { 2 } } { P } W ^ { \top } ( I - \frac { 1 } { P } \mathbf { 1 } _ { P } \mathbf { 1 } _ { P } ^ { \top } ) W ( U ^ { x } + \mathbb { E } _ { \boldsymbol { \pi } } [ V e _ { y } \mid x ] ) \Biggr \} + \delta , } \end{array}
$$

where $\delta = O ( \epsilon ^ { 3 } )$

The same calculation would be applied to $V .$ . Then we obtain the gradient flow for W. Denote $r _ { z }$ by the probability of samples labeled z. According to the definition, there is

$$
\frac { \mathrm { d } W _ { z } ^ { \top } } { \mathrm { d } t } = ( e _ { z } - p _ { x , y } ) \sigma ( h _ { x , y } ) ,
$$

Then, the gradient flow of the z-th row of W is expressed in the expected form as

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } W _ { z } ^ { \top } } { \mathrm { d } t } = c _ { 1 } \left( r _ { z } \mathbb { E } _ { \pi } [ h _ { x , y } \mid Z = z ] - \frac { 1 } { P } \mathbb { E } _ { \pi } [ h _ { x , y } ] \right) } \\ { \displaystyle \qquad + \frac { c _ { 2 } } { 2 } \left( r _ { z } \mathbb { E } _ { \pi } [ ( h _ { x , y } \odot h _ { x , y } ) \mid Z = z ] - \frac { 1 } { P } \mathbb { E } _ { \pi } [ h _ { x , y } \odot h _ { x , y } ] \right) } \\ { \displaystyle \qquad - \frac { c _ { 1 } ^ { 2 } } { P } \mathbb { E } _ { \pi } [ h _ { x , y } ^ { \top } h _ { x , y } ] W ^ { \top } \Pi _ { P } e _ { z } + \delta . } \end{array}
$$

Substituting $h _ { x , y } = U ^ { x } + V ^ { y }$ yields a similar result.

## E.2 Proof of Dominance of the gradient

Proof. We give the proof for $U$ and the same calculation would be able to applied to $V$ and $W$ . For convenience, write

$$
q _ { x } : = \mathbb { E } _ { \pi } [ e _ { z } \mid x ] , \qquad { \bar { V } } _ { x } : = \mathbb { E } _ { \pi } [ V e _ { y } \mid x ] ,
$$

and define

$$
\Pi _ { P } : = I _ { P } - \frac 1 P \mathbf { 1 } _ { P } \mathbf { 1 } _ { P } ^ { \top } .
$$

From proposition 1, we have

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } \Biggl \{ c _ { 1 } W ^ { \top } \left( q _ { x } - \frac { 1 } { P } { \bf 1 } _ { P } \right) } \\ & { \qquad \quad + c _ { 2 } U ^ { x } \odot \left( W ^ { \top } q _ { x } \right) } \\ & { \quad \quad \quad + c _ { 2 } \mathbb { E } _ { \pi } \left[ V ^ { y } \odot \left( W ^ { \top } e _ { z } \right) \mid x \right] } \\ & { \quad \quad \quad - \frac { c _ { 2 } } { P } \left( U ^ { x } + \bar { V } _ { x } \right) \odot \left( W ^ { \top } { \bf 1 } _ { P } \right) } \\ & { \quad \quad \quad \quad - \frac { c _ { 1 } ^ { 2 } } { P } W ^ { \top } \Pi _ { P } W \left( U ^ { x } + \bar { V } _ { x } \right) \Biggr \} + \delta . } \end{array}\tag{1}
$$

Since z | x is uniform on $\Omega .$

$$
q _ { x } = \mathbb { E } _ { \pi } [ e _ { z } \mid x ] = { \frac { 1 } { P } } \mathbf { 1 } _ { P } .\tag{2}
$$

Therefore,

$$
c _ { 1 } W ^ { \top } \left( q _ { x } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) = 0 .\tag{3}
$$

and

$$
c _ { 2 } U ^ { x } \odot W ^ { \top } \left( q _ { x } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) = 0 .\tag{4}
$$

The joint product term is

$$
W ^ { \top } \Pi _ { P } W \left( U ^ { x } + \bar { V } _ { x } \right) = O ( \epsilon ^ { 3 } ) .
$$

Substitute all these, we have

$$
\frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = r _ { x } c _ { 2 } \bigg [ \mathbb { E } _ { \pi } \left[ V ^ { y } \odot \left( W ^ { \top } e _ { z } \right) \mid x \right] - \overline { { V } } \cdot \overline { { W } } \bigg ] + \delta ,
$$

It remains to identify the conditional expectation with the stated matrix expression. By definition,

$$
\begin{array} { r l } {  { \mathrm { d i a g } \big ( V \Phi _ { x } ^ { ( a ) } W \big ) = \displaystyle \sum _ { y , z } \mathbb { P } \big ( Z = z , Y = y \mid X = x \big ) \ \mathrm { d i a g } \big ( V ^ { y } ( e _ { z } ^ { \top } W ) \big ) } } \\ & { = \displaystyle \sum _ { y , z } \mathbb { P } \big ( Z = z , Y = y \mid X = x \big ) \ \big ( V ^ { y } \odot W ^ { \top } e _ { z } \big ) } \\ & { = \mathbb { E } _ { \pi } [ V ^ { y } \odot ( W ^ { \top } e _ { z } ) \mid x ] . } \end{array}
$$

adding the weight decay term, we conclude that

$$
\frac { \mathrm { d } U ^ { x } } { \mathrm { d } t } = - \lambda U ^ { x } + r _ { x } c _ { 2 } \bigg [ \mathrm { d i a g } \left( V \Phi _ { x } ^ { ( a ) } W \right) - \overline { { V } } \cdot \overline { { W } } \bigg ] + \delta .
$$

## E.3 Proof of Estimation of noise

Proof. Let $G _ { \mathcal { D } }$ be the empirical full-batch vector field. To avoid random conditional denominators, for every $( x , y ) \in \Omega ^ { 2 }$ let $I _ { x y }$ be the indicator that the pair is included in the training set, and let $\widetilde { Z } _ { x y }$ be its observed label when $I _ { x y } = 1$ . Uniform sampling and uniform label replacement give the exact identity

$$
\mathbb { E } _ { \mathcal { D } } \bigg [ \frac { I _ { x y } } { m } \mathbf { 1 } _ { \{ \widetilde { Z } _ { x y } = z \} } \bigg ] = \frac { 1 } { P ^ { 2 } } \left( ( 1 - \alpha ) \mathbf { 1 } _ { \{ z = x + y \} } + \frac { \alpha } { P } \right) .\tag{1}
$$

The expectation in (1) is over the training pairs, the corrupted subset, and the replacement labels, with the parameters fixed.

We prove the assertion for U first. Applying the expansion in Proposition 1 to the empirical measure yields

$$
\begin{array} { l } { { \displaystyle G _ { \mathcal { D } , x } ^ { U } + \lambda U ^ { x } = \frac { \eta } { m } \sum _ { y \in \Omega } I _ { x y } \Bigg [ c _ { 1 } W ^ { \top } \left( e _ { \tilde { Z } _ { x y } } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) } } \\ { { \displaystyle \qquad + c _ { 2 } ( U ^ { x } + V ^ { y } ) \odot W ^ { \top } \left( e _ { \tilde { Z } _ { x y } } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) \Bigg ] + \Delta _ { \mathcal { D } , x } ^ { U } } , } \end{array}\tag{2}
$$

For equation (1), summed over y, shows that

$$
\mathbb { E } _ { \mathcal { D } } \bigg [ \frac { 1 } { m } \sum _ { y } I _ { x y } \left( e _ { \widetilde { Z } _ { x y } } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) \bigg ] = 0 .\tag{3}
$$

Consequently, both the first-order term and the part of the second-order term containing $U ^ { x }$ have zero sampling mean. For the remaining part, (1) gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \mathcal { D } } \Bigg [ \frac { \eta c _ { 2 } } { m } \sum _ { y } I _ { x y } V ^ { y } \odot W ^ { \top } \left( e _ { \tilde { Z } _ { x y } } - \frac { 1 } { P } \mathbf { 1 } _ { P } \right) \Bigg ] } \\ { \displaystyle = \frac { \eta c _ { 2 } ( 1 - \alpha ) } { P } \left[ P ^ { - 1 } \sum _ { y } V ^ { y } \odot W ^ { \top } e _ { x + y } - \overline { { V } } \odot \overline { { W } } \right] . } \end{array}\tag{4}
$$

This has already yielded the main components of the theorem. Exchanging $U$ and V for another proof.

For the x-th output row, set $\begin{array} { r } { \Psi _ { x } ^ { + } : = \sum _ { j } e _ { j } e _ { x - j } ^ { \top } } \end{array}$ . The same Taylor expansion gives

$$
G _ { \mathcal { D } , x } ^ { W } + \lambda W _ { x } = \frac { \eta } { m } \sum _ { a , b } I _ { a b } \left( \mathbf { 1 } _ { \left\{ \tilde { Z } _ { a b } = x \right\} } - \frac { 1 } { P } \right) \left[ c _ { 1 } h _ { a b } + \frac { c _ { 2 } } { 2 } ( h _ { a b } \odot h _ { a b } ) \right] + \Delta _ { \mathcal { D } , x } ^ { W } ,\tag{5}
$$

where $h _ { a b } = U ^ { a } + V ^ { b }$ . The linear term in (5) has zero expectation: on every addition diagonal $a + b =$ x, both $\textstyle \sum _ { a } U ^ { a }$ and $\sum _ { b } V ^ { b }$ agree with their uniform-background sums. The same cancellation removes the $U ^ { a } \odot U ^ { a }$ and $V ^ { b } \odot V ^ { b }$ terms. The two equal cross terms leave

$$
\frac { \eta c _ { 2 } ( 1 - \alpha ) } { P } \left[ P ^ { - 1 } \sum _ { a } U ^ { a } \odot V ^ { x - a } - \overline { { U } } \odot \overline { { V } } \right] .\tag{6}
$$

It remains to control the finite-sample fluctuation. Define $\mathcal { E } _ { \bullet }$ as the centered fluctuation of the displayed first- and second-order terms, and include the high order remainder. For simple random sampling without replacement, the finite-population variance formula gives a factor $( 1 - \rho ) / ( \rho P ^ { 3 } )$ for a fixed input or output coordinate. Randomly corrupting an α fraction of the selected labels contributes at most an additional $\alpha / ( \rho P ^ { 3 } )$ . In (2) and (5), each coordinate of a centered first-order summand is bounded by $2 | c _ { 1 } | \epsilon$ , while each second-order summand is bounded by $C | c _ { 2 } | \epsilon ^ { 2 }$ . The variance formula gives

$$
\operatorname { V a r } _ { \mathcal { D } } ( \mathcal { E } _ { \bullet } ) \leq \frac { C \eta ^ { 2 } } { \rho P ^ { 3 } } \left[ ( 1 - \rho ) + \alpha \right] \left( c _ { 1 } ^ { 2 } \epsilon ^ { 2 } + c _ { 2 } ^ { 2 } \epsilon ^ { 4 } \right) .
$$

which is the claimed bound.

It is noted that the error is caused by the $\rho$ and α during the sampling. If the training set contains too few clean samples, the error between actual gradient and the predicted term will be very large, which is consistent with our intuition.

## E.4 Proof of dominant frequency and phase locking

Proof. The Fourier modes obey

$$
\dot { U } _ { r } = C \overline { { V _ { r } } } W _ { r } - \lambda U _ { r } , \quad \dot { V } _ { r } = C \overline { { U _ { r } } } W _ { r } - \lambda V _ { r } , \quad \dot { W } _ { r } = C U _ { r } V _ { r } - \lambda W _ { r } .
$$

Set $U _ { r } = e ^ { - \lambda t } X _ { r } ( s ) , V _ { r } = e ^ { - \lambda t } Y _ { r } ( s )$ , and $W _ { r } = e ^ { - \lambda t } Z _ { r } ( s )$ , with

$$
s = { \frac { C } { \lambda } } ( 1 - e ^ { - \lambda t } ) .
$$

Then

$$
{ \frac { \mathrm { d } X _ { r } } { \mathrm { d } s } } = \overline { { Y _ { r } } } Z _ { r } , \quad { \frac { \mathrm { d } Y _ { r } } { \mathrm { d } s } } = \overline { { X _ { r } } } Z _ { r } , \quad { \frac { \mathrm { d } Z _ { r } } { \mathrm { d } s } } = X _ { r } Y _ { r } .
$$

If $s _ { r } ^ { \star } < C / \lambda$ is the undamped blow-up time, its damped time is

$$
t _ { r } ^ { \star } = - \frac { 1 } { \lambda } \log \left( 1 - \frac { \lambda s _ { r } ^ { \star } } { C } \right) .
$$

This map is increasing, so $s _ { w } ^ { \star } < s _ { i } ^ { \star }$ implies $t _ { w } ^ { \star } < t _ { i } ^ { \star } . \mathrm { A s } t \uparrow t _ { w } ^ { \star }$ , the energy of mode w diverges while every competing mode remains finite, and its energy fraction tends to one.

For phase locking, write $U = A e ^ { i \alpha } , V = B e ^ { i \beta } , W = G e ^ { i \gamma }$ , and $\delta = \alpha + \beta - \gamma$ . Direct separation of real and imaginary parts gives

$$
\dot { \delta } = - C \sin \delta \left( \frac { B G } { A } + \frac { A G } { B } + \frac { A B } { G } \right) .
$$

While the amplitudes remain nonzero, the factor in parentheses is positive. Thus $\delta = 0$ is stable and $\delta = \pi$ is unstable. A continuous random initialization avoids the unstable equilibrium with probability one, yielding $\delta ( t )  0$ modulo 2π on the regular continuation interval. □

## E.5 Proof of the early-loss theorem

Lemma 1. Let $a , b , c ,$ d be i.i.d. random variables and let $f : \mathbb { R } \to \mathbb { R }$ be such that the expectations below exist, then

$$
\begin{array} { r } { \mathbb { E } [ f ( a + b ) f ( a + c ) ] \geq \mathbb { E } [ f ( a + b ) f ( c + d ) ] . } \end{array}
$$

Proof. Define $h ( a ) = \mathbb { E } [ f ( a + b ) \mid a ] , \qquad \mu = \mathbb { E } [ f ( a + b ) ]$

Since $a + b$ and $c + d$ are independent,

$$
\begin{array} { r } { \mathbb { E } [ f ( a + b ) f ( c + d ) ] = \mathbb { E } [ f ( a + b ) ] \mathbb { E } [ f ( c + d ) ] = \mu ^ { 2 } . } \end{array}
$$

Given a, the variables b and c are independent, so

$$
\begin{array} { r } { \mathbb { E } [ f ( a + b ) f ( a + c ) \mid a ] = \mathbb { E } [ f ( a + b ) \mid a ] \mathbb { E } [ f ( a + c ) \mid a ] = h ( a ) ^ { 2 } . } \end{array}
$$

Hence $\mathbb { E } [ f ( a + b ) f ( a + c ) ] = \mathbb { E } [ h ( a ) ^ { 2 } ]$

Therefore,

$$
\begin{array} { r } { \mathbb { E } [ f ( a + b ) f ( a + c ) ] - \mathbb { E } [ f ( a + b ) f ( c + d ) ] = \mathbb { E } [ h ( a ) ^ { 2 } ] - \mu ^ { 2 } . } \end{array}
$$

But $\mu = \mathbb { E } [ h ( a ) ]$ , so

$$
\begin{array} { r } { \mathbb { E } [ h ( a ) ^ { 2 } ] - \mu ^ { 2 } = \operatorname { V a r } ( h ( a ) ) \geq 0 . } \end{array}
$$

Thus $\begin{array} { r } { \mathbb { E } [ f ( a + b ) f ( a + c ) ] \geq \mathbb { E } [ f ( a + b ) f ( c + d ) ] } \end{array}$

For activation functions such as GELU and RELU, the above inequality will hold strictly. It is $\kappa ( 1 ) > \kappa ( 0 )$ in the main text.

## Theorem 3.

Proof. The total number of all pairs with the same label is

$$
K _ { 0 } = \sum _ { z } n _ { z } ^ { 2 } ,
$$

Let $n _ { x , y } ^ { ( a ) }$ count examples with first coordinate $X = x$ and label $Z = y ;$ , and define $n _ { z , y } ^ { ( b ) }$ analogously. In the complete table, $n _ { x , y } ^ { ( a ) } = P \varphi _ { x , y } ^ { ( a ) }$ and $n _ { z , y } ^ { ( b ) } = P \varphi _ { z , y } ^ { ( b ) }$ . The ordered pairs sharing the first coordinate and label number

$$
C _ { a } = \sum _ { x , y } n _ { x , y } ^ { ( a ) } ( n _ { x , y } ^ { ( a ) } - 1 ) = P ^ { 2 } \| \varphi ^ { ( a ) } \| _ { 2 } ^ { 2 } - P ^ { 2 } ,
$$

and similarly $C _ { b } = P ^ { 2 } \| \varphi ^ { ( b ) } \| _ { 2 } ^ { 2 } - P ^ { 2 }$

We define

$$
\kappa ( l ) = \mathbb { E } [ \sigma ( h _ { i } ) \sigma ( h _ { j } ) ] \cdot D , \quad l = 0 , 1 , 2 ,
$$

Here $( h _ { i } , h _ { j } )$ are scalar jointly Gaussian, each of variance $2 s ^ { 2 }$ and covariance $l s ^ { 2 }$ . s denotes the initialization scale. The definition is equivalent to that in the main text.

Thus, We can count it by using the principle of inclusion-exclusion as:

$$
- \mathbb { E } \dot { L } / \eta = \frac { 1 } { N ^ { 2 } } \left( \kappa ( 0 ) K _ { 0 } + ( \kappa ( 1 ) - \kappa ( 0 ) ) ( C _ { a } + C _ { b } ) + ( \kappa ( 2 ) - \kappa ( 0 ) ) N \right) ,
$$

The last item is independent of the training distribution.

## E.6 Propositions of Hadamard basis

Let

$$
G = ( \mathbb { Z } / 2 \mathbb { Z } ) ^ { 6 }
$$

and identify the group operation on $G$ with bitwise XOR. For each $a \in G$ , define the matrix $\Phi _ { a } \in \mathbb { R } ^ { G \times G }$ by

$$
\Phi _ { a } ( y , z ) = \frac 1 { 6 4 } { \bf 1 } _ { \{ z = a + y \} } , \qquad y , z \in G .
$$

For $u \in G$ , define the Walsh character

$$
\chi _ { u } ( v ) = ( - 1 ) ^ { u \cdot v } ,
$$

where

$$
u \cdot v = \sum _ { j = 1 } ^ { 6 } u _ { j } v _ { j } { \pmod { 2 } } .
$$

Let $H _ { u } \in \mathbb { R } ^ { G }$ be the normalized Walsh vector

$$
H _ { u } ( v ) = \frac { 1 } { 8 } \chi _ { u } ( v ) .
$$

Then the family $\{ \Phi _ { a } : a \in G \}$ is simultaneously diagonalized by the orthonormal Walsh–Hadamard basis $\{ H _ { u } : u \in G \}$ . More precisely,

$$
\Phi _ { a } H _ { u } = \frac { \chi _ { u } ( a ) } { 6 4 } H _ { u } = \frac { ( - 1 ) ^ { u \cdot a } } { 6 4 } H _ { u } .
$$

Consequently, if H is the orthogonal matrix whose columns are the vectors $H _ { u } .$ , then

$$
H ^ { \top } \Phi _ { a } H = \mathrm { d i a g } \left( \frac { \chi _ { u } ( a ) } { 6 4 } \right) _ { u \in G }
$$

for every $a \in G$ . Indeed, for $u , u ^ { \prime } \in G .$

$$
\begin{array} { r } { \langle H _ { u } , H _ { u ^ { \prime } } \rangle = \displaystyle \frac { 1 } { 6 4 } \sum _ { v \in G } \chi _ { u } ( v ) \chi _ { u ^ { \prime } } ( v ) } \\ { = \displaystyle \frac { 1 } { 6 4 } \sum _ { v \in G } ( - 1 ) ^ { ( u + u ^ { \prime } ) \cdot v } . } \end{array}
$$

If $u = u ^ { \prime }$ , then the summand is identically equal to 1, and hence

$$
\langle H _ { u } , H _ { u } \rangle = 1 .
$$

If $u \ne u ^ { \prime }$ , then $u + u ^ { \prime } \ne 0$ . Choose an element $v _ { 0 } \in G$ such that

$$
( u + u ^ { \prime } ) \cdot v _ { 0 } = 1 .
$$

The map $v \mapsto v + v _ { 0 }$ is a bijection of $G ,$ , and therefore

$$
\begin{array} { r l r } {  { \sum _ { v \in G } ( - 1 ) ^ { ( u + u ^ { \prime } ) \cdot v } = \sum _ { v \in G } ( - 1 ) ^ { ( u + u ^ { \prime } ) \cdot ( v + v _ { 0 } ) } } } \\ & { } & { \quad = - \sum _ { v \in G } ( - 1 ) ^ { ( u + u ^ { \prime } ) \cdot v } . } \end{array}
$$

Thus the sum is zero. Hence

$$
\langle H _ { u } , H _ { u ^ { \prime } } \rangle = \mathbf { 1 } _ { \{ u = u ^ { \prime } \} } ,
$$

so $\{ H _ { u } : u \in G \}$ is an orthonormal basis of $\mathbb { R } ^ { G }$ . Now fix $a , u \in G$ . For every $y \in G$ , we have

$$
\begin{array} { l } { { ( \Phi _ { a } H _ { u } ) ( y ) = \displaystyle \sum _ { z \in G } \Phi _ { a } ( y , z ) H _ { u } ( z ) } } \\ { { = \displaystyle \frac { 1 } { 6 4 } H _ { u } ( a + y ) . } } \end{array}
$$

Using the character identity

$$
\chi _ { u } ( a + y ) = \chi _ { u } ( a ) \chi _ { u } ( y ) ,
$$

we obtain

$$
\begin{array} { c l c r } { { ( \Phi _ { a } H _ { u } ) ( y ) = \displaystyle \frac { 1 } { 6 4 } \cdot \frac { 1 } { 8 } \chi _ { u } ( a + y ) } } \\ { { = \displaystyle \frac { 1 } { 6 4 } \cdot \frac { 1 } { 8 } \chi _ { u } ( a ) \chi _ { u } ( y ) } } \\ { { = \displaystyle \frac { \chi _ { u } ( a ) } { 6 4 } H _ { u } ( y ) . } } \end{array}
$$

![](images/ba28508264091cfd4fbdc9b546c3aa8c98c0e64224310c137afa5fd852052d64.jpg)

![](images/09d4832389b8688b866c1597466f906135b92d2b2855d4872ffd82602194a042.jpg)  
Figure 17: Early-training loss curves of the modular multiplication by 59, 60, 61.

Since this holds for every $y \in G$

$$
\Phi _ { a } H _ { u } = \frac { \chi _ { u } ( a ) } { 6 4 } H _ { u } .
$$

Thus every Walsh vector $H _ { u }$ is a common eigenvector of every matrix $\Phi _ { a }$ , with eigenvalue

$$
\lambda _ { u } ( a ) = { \frac { \chi _ { u } ( a ) } { 6 4 } } = { \frac { ( - 1 ) ^ { u \cdot a } } { 6 4 } } .
$$

Let H denote the orthogonal matrix whose u-th column is $H _ { u }$ . Since the columns of H form a common orthonormal eigenbasis, we have

$$
H ^ { \top } \Phi _ { a } H = \mathrm { d i a g } \left( \frac { \chi _ { u } ( a ) } { 6 4 } \right) _ { u \in G } .
$$

Therefore, the matrices $\{ \Phi _ { a } : a \in G \}$ are simultaneously orthogonally diagonalized by the normalized Walsh–Hadamard matrix.

## F Experiment of Various Label Noise

## F.1 Experiment on Modular Multiplication Task

In order to verify our theory, we designed the following experiment on the modular multiplication task. Consider $N = 6 1 , \Omega = \{ 1 , 2 , \dots , N \}$ and there are three modular-period that $M _ { 1 } = 5 9 , M _ { 2 } =$ $6 0 , M _ { 3 } = 6 1$ . The total pairs of $\Omega \times \Omega$ are divided into three parts $S _ { 1 } , S _ { 2 } , S _ { 3 }$ . The labels of data from

$$
S _ { i } = \{ ( x _ { i } , y _ { i } ) \} , \quad i = 1 , 2 , 3 ,
$$

are set to

$$
\tilde { z } _ { i } = x _ { i } \cdot y _ { i } \mod M _ { i } , \quad \tilde { z } _ { i } \in \Omega \cup \{ 0 \} ,
$$

![](images/5eb24e9edba9975a4275d4ffa1ee46f86b8fb96f60d7755ecc31f4da77b87db7.jpg)  
Figure 18: Early-training loss curves of the bijective noise experiment.

We trained it using the same neural network as before. See Figure 17 for the early-training loss curves of the three tasks. We can see that the task with $M _ { 2 } = 6 0$ showed a significantly faster decline than the others.

Our theory can easily explain this point. For $M = 6 0$ , there exist a large number of pairs $( x , y )$ such that $x \cdot y \equiv x ^ { \prime } \cdot y$ (mod M), such as

$$
5 \times 1 5 \equiv 1 3 \times 1 5 { \pmod { 6 0 } } ,
$$

which means that the label z is more concentrated. This leads to larger $K _ { 0 } = \textstyle \sum _ { z } n _ { z } ^ { 2 }$ and larger $\| \varphi ^ { ( a ) } \| _ { 2 } ^ { 2 } + \| \varphi ^ { ( b ) } \| _ { 2 } ^ { 2 }$ , thus a faster early training loss decline according to Theorem 3. In contrast, for prime numbers $M _ { 1 } = 5 9$ and $M _ { 3 } = 6 1$ , it satisfies $x \cdot y \equiv x ^ { \prime } \cdot y$ (mod M) if and only if $x \equiv x ^ { \prime }$ (mod M) for non-zero $y ,$ which means the signature of $M = 5 9 , 6 1$ are more uniformly distributed. This leads to a smaller $K _ { 0 }$ and $K _ { 1 }$ (we think that the values of $M _ { 1 } = 5 9$ and $M _ { 3 } = 6 1$ are very close to each other). As shown in Figure 17, the theoretical prediction is consistent with the actual observation. The early loss slope of $M _ { 2 } = 6 0$ is much greater than the others.

## F.2 Experiment on the Bijective Noise

In the main text, we point out that in the scenario of uniform noise on the labels, during the early training phase, it is always observed that the loss of the noise decreases first, as in figure 4.

Here we present a specific noise construction that ensures it does not undergo such collisions. Just as our theory predicted, the phenomenon observed after replacing this kind of noise was completely different from what was observed before, as shown in Figure 18. Specifically, we construct a binary function $g ( x , y )$ such that:

$g ( x , \cdot )$ is bijective for any fixed x and $g ( \cdot , y )$ is bijective for any fixed $y .$

• $g$ is not isomorphic to addition, that is,

$$
g ( x , y ) \neq \sigma _ { 3 } ( \sigma _ { 1 } ( x ) + \sigma _ { 2 } ( y ) ) ,
$$

where $\sigma _ { i }$ is from the permutation group of Ω.

• Generally, g cannot be accurately fitted by our 2L-MLP.

Our final construction is

$$
g ( x , y ) = { \left\{ \begin{array} { l l } { x , } & { x = y , } \\ { ( x + ( 4 + \left( { \frac { y - x } { 1 1 3 } } \right) ) \cdot ( y - x ) ) } & { { \mathrm { m o d ~ 1 1 3 , } } \quad x \neq y , } \end{array} \right. }
$$

where $\left( { \frac { \cdot } { 1 1 3 } } \right)$ is the Quadratic Residue Legendre symbol (of modular 113). It satisfies the above three properties. $\mathrm { N e x t . }$ , we replace the original uniform random noise with the function $g ( x , y )$ . In this case, the loss of $g$ does not always decrease first during the early training phase. By calculating the coefficient preceding $\kappa ( 1 )$ , We can give

$$
\Delta _ { \mathrm { c l e a n } } - \Delta _ { \mathrm { g } } = 2 \alpha - 1 ,
$$

Inferring like that in chapter 6.1, a larger $\Delta$ gives a more negative early loss derivative. This indicates that when the noise ratio $\alpha < 0 . 5$ , the noise data $( g ( x , y ) )$ decrease first. In contrast, when $\alpha > 0 . 5$ , the opposite occurs. As shown in Figure 18, the theoretical prediction is consistent with the actual observation.