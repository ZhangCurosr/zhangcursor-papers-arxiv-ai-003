# Why Shallow Neural Networks Learn Sinusoidal Components First: An Orthogonal Polynomial Dynamics Framework

Yanyang Xiao<sup>a,∗</sup>

<sup>a</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

## Abstract

Neural networks often learn low-frequency components before high-frequency ones --- a phenomenon called spectral bias or termed F-principle. Many numerical studies have demonstrated this phenomenon, and theoretical studies have sought to explain it. However, several goals remain insuficiently addressed: (1) explaining spectral bias asymptotically, including its origin and why the preferred components are generally sinusoidal; (2) extending the explanation to general activation functions, such as sigmoid and ReLU; (3) making weak assumptions about the data distribution; (4) avoiding aliasing efects, such as those arising in finite-domain Fourier analysis; and (5) developing a theoretical framework applicable to arbitrary networks. We address these goals theoretically by analyzing the eigenvalue decay and eigenvector shapes of the dynamics matrix H, which arises when we expand the network output as a general Fourier series (GFS) in an orthogonal polynomial basis and express the gradient-descent dynamics of the GFS coeficients as ordinary diferential equations. The graded structure of H enables an intuitive explanation of spectral bias, while orthogonal polynomial theory helps us analyze how learning bias depends on the activation function and, potentially, the network architecture.

Keywords: artificial neural network, spectral bias, orthogonal polynomials, singular value decomposition, asymptotic eigenvector

## 1. Introduction

In light of the Universal Approximation Theorem for neural networks, there is a shadow over how networks can learn useful things. Along this path, subjects such as learning bias, implicit regularization, and information bottlenecks arise. In particular, a phenomenon called spectral bias, or the F-principle, was discovered and was thought to be connected to implicit regularization, to have explanatory power for early stopping and generalization, and to expose weaknesses on high-frequency tasks [1, 2, 3]. However, the observations in [1], although clear in deriving the main conclusion, relied on finite-domain Fourier analysis, in which lower-amplitude frequency components are largely contaminated by aliasing efects. To fix this flaw and provide a quantitative theory to explain it was the motivation of this work. Specifically, we would like to (1) find an aliasing-free representation to describe the frequency components; (2) provide the dynamics (time-dependent evolution) of the frequency components in that representation; (3) explain the spectral bias phenomenon through these dynamics; (4) allow the explanation to be only asymptotic, leaving room for easier computation, easier interpretation, and intuitive understanding; and (5) ensure that the framework (constituted by 1–4) has the potential to extend to networks beyond toy models.

Over the years, various theoretical works have emerged. The Neural Tangent Kernel (NTK) theory was the first fundamental theory closely connected to our work, in which, in principle, the training dynamics of any network can be written as evolution through the kernel [4], yet the computation of the kernel involves complex integration. Some recent papers accomplish impressive rigorous computations; for example, [5] connect the decay speed of Hermite coeficients to NTK coeficients and network depth, and [6] obtain the NTK analytically through integration for ReLU, so that various properties can be derived. However, [5] do not discuss eigenvector shape, and [6] discuss the shape of eigenvectors but only for the ReLU activation function. In contrast, our theory applies to any sensible activation function and, when explaining the shape of eigenvectors, requires no closed-form integral.

A wide range of natural signals, such as natural images, natural sounds, and EEG recordings, exhibit low-frequency domination. Traditional ways of dealing with them often rely on or connect to Fourier analysis. The sinusoidal shape of the "frequency components" in the frequency bias happens to coincide with the "principal components" of those natural signals. It may be because of this coincidence that neural networks happen to be good at processing natural signals. We would like to provide a mathematical perspective

on this as a by-product of this work.

The paper is organized as follows: First, we re-represent the output of the network using an Orthogonal-Polynomial-based General Fourier Series (GFS). Then, we derive the dynamics of the GFS coeficients, which constitute the main governing equation. To analyze the dynamics of the governing equation, we then explore the key properties of the dynamics matrix $H -$ namely, its eigenvalues and eigenvectors—using a shallow neural network as an example. Finally, we discuss the relation of the H matrix to the loss landscape problem and efective dimension.

## 2. General Fourier Series Expansion Derived from Loss

Let’s say we have target function $y ( x ) , \mathbb { R } \mapsto \mathbb { R }$ . The problem setup is to fit J given pairs of feature and label (inputs and outputs) $\{ ( x _ { j } , y _ { j } ) \}$ by a function $\Upsilon _ { \theta } ( x )$ with parameter θ (usually a vector). Without loss of generality, we limit the range of $x _ { j }$ to $[ - 1 , 1 ]$ . Generalize to high dimensional $x _ { j } \in [ - 1 , 1 ] ^ { d }$ is under consideration.

We have the 2-norm loss for the error of the fitting:

$$
\tilde { L } _ { J } ( \theta ) = \frac { 1 } { 2 J } \sum _ { j = 1 } ^ { J } ( \Upsilon _ { \theta } ( x _ { j } ) - y _ { j } ) ^ { 2 } .\tag{1}
$$

Assume the samples $\{ x _ { j } \}$ are draw from a distribution $\rho ( x )$ , and the labels are come from an underlying function $y ( x ) \colon y _ { j } \ = \ y ( x _ { j } )$ . Now let $J \to \infty$ , we should have a theoretical loss $L ( \theta )$

$$
L ( \theta ) = \frac { 1 } { 2 } \int _ { \Omega } \left( \Upsilon _ { \theta } ( x ) - y ( x ) \right) ^ { 2 } \rho ( x ) \mathrm { d } x .\tag{2}
$$

$$
\tilde { L } _ { J } ( \theta )  L ( \theta ) \quad \mathrm { a s ~ } J  \infty ,
$$

The Eq. (2) naturally induced a norm and an inner product (assume all real functions)

$$
\| f \| _ { \rho } = \int _ { \Omega } f ^ { 2 } ( x ) \rho ( x ) \mathrm { d } x ,\tag{3}
$$

$$
\langle f , g \rangle _ { \rho } = \int _ { \Omega } f ( x ) g ( x ) \rho ( x ) \mathrm { d } x .\tag{4}
$$

Under the inner product in Eq. (4), there exist a set of orthonomal polynomial basis $\{ p _ { k } ( x ) \} _ { k = 0 } ^ { \infty }$ where $p _ { k } ( x )$ is a polynomial of order k:

$$
\begin{array} { r } { \langle p _ { k } , p _ { j } \rangle _ { \rho } = \delta _ { k j } . } \end{array}\tag{5}
$$

In 1-D case, each $p _ { k } ( x )$ has exactly k zeros. Therefore the subscript k has similar meaning as the k in Fourier mode $e ^ { \mathrm { i } k x }$ , i.e. the frequency.

If $\rho ( x ) > 0$ almost everywhere, then $\{ p _ { k } ( x ) \} _ { k = 0 } ^ { \infty }$ is a complete basis with the same convergence trends as Chebyshev approximation.

The fitting function $\Upsilon _ { \theta }$ , assumed piecewise continuous, can be expanded in Generalized Fourier Series (GFS):

$$
\Upsilon _ { \theta } ( x ) = \sum _ { k = 0 } ^ { \infty } c _ { k } ( \theta ) p _ { k } ( x ) \quad ( \mathrm { a . e . } )\tag{6}
$$

$$
c _ { k } ( \theta ) = \left. \Upsilon _ { \theta } , p _ { k } \right. _ { \rho }\tag{7}
$$

where $\left( \mathrm { a . e . } \right)$ here means (equal) almost everywhere. If $\Upsilon _ { \theta } ( x )$ is continuous, then it is point-wise equal. In later context we will denote both cases by simply equal $\cdot \underline { { \underline { { \circ } } } } ( \underline { { \mathbf { \delta } } } _ { \mathcal { M } } ) $

## 3. Evolution of Generalized Fourier Series in Gradient Descent

## 3.1. Gradient of Loss in View of Generalized Fourier Series

The expansion of gradient of loss function provides insight for gradient descent algorithm:

$$
\begin{array} { c } { \displaystyle \frac { \partial L ( \theta ) } { \partial \theta } = \frac { 1 } { 2 } \frac { \partial } { \partial \theta } \sum _ { k = 0 } ^ { \infty } \left. f _ { \theta } - y , p _ { k } \right. _ { \rho } ^ { 2 } } \\ { \displaystyle = \sum _ { k = 0 } ^ { \infty } \left( c _ { k } ( \theta ) - \left. y , p _ { k } \right. _ { \rho } \right) \frac { \partial c _ { k } ( \theta ) } { \partial \theta } . } \end{array}\tag{8}
$$

For example, if $f _ { \theta }$ is 1-layer ANN $( \theta = \{ w \iota , r \iota , b \iota \} _ { l = 1 } ^ { L } )$ , with $x \in [ - 1 , 1 ]$

$$
f _ { \theta } = \sum _ { l = 1 } ^ { L } w _ { l } \sigma ( r _ { l } x + b _ { l } ) .\tag{9}
$$

Then

$$
\frac { \partial f _ { \theta } } { \partial \theta } = \left\{ \begin{array} { l l } { \sigma ( r _ { l } x + b _ { l } ) } & { \theta = w _ { l } } \\ { w _ { l } \sigma ^ { \prime } ( r _ { l } x + b _ { l } ) } & { \theta = b _ { l } \quad l = 1 , \ldots , L . } \\ { w _ { l } \sigma ^ { \prime } ( r _ { l } x + b _ { l } ) x } & { \theta = r _ { l } } \end{array} \right.\tag{10}
$$

The last term in $\operatorname { E q }$ . (8) is

$$
\frac { \partial c _ { k } ( \theta ) } { \partial \theta } = \left. \frac { \partial \mathrm { f } _ { \theta } } { \partial \theta } , p _ { k } \right. _ { \rho } .\tag{11}
$$

If we further assume that σ is smooth in the sense of Lemma 1 or Lemma $^ { 3 , }$ then $\frac { \partial f _ { \theta } } { \partial \theta }$ will have similar but possibly slightly degraded smoothness. See Lemma 8 for general $\rho ( x )$ . The following analysis are based on the Chebyshev expansion $( \rho ( x ) = \left( 1 - x ^ { 2 } \right) ^ { - 1 / 2 } )$ . For $\rho ( x ) = 1$ , it is the Legendre polynomials expansion, it has the same asymptotics as Chebyshev expansion (Ref. [7]).

Case 1, $\sigma ( x )$ is analytical (in the sense of Lemma 3), so $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ and $\frac { \partial f _ { \theta } ( x ) } { \partial \theta }$ are also analytical.

Let’s assume that there is no singularity in a neighborhood of real axis <sup>1</sup> in the domain of $\sigma ( x )$ , the nearest singularity is $\pm \mathrm { i } B$ away from the real axis. In this setup, the Bernstein ellipse parameter Θ (the $\rho$ in Lemma $^ { 3 , }$ not to be confused with the weighting $\rho ( x ) )$ is $\Theta _ { \sigma } = B + \sqrt { 1 + B ^ { 2 } }$ at least.

So the Bernstein ellipse parameter for $\frac { \partial f _ { \theta } } { \partial \theta }$ is $\Theta _ { \theta } = B / r _ { l } + \surd 1 + \left( B / r _ { l } \right) ^ { 2 } \approx$ 2) $\Theta _ { \sigma } / r _ { l } \ ( ^ { \ast } \approx ^ { \ast }$ when $r _ { l } \ll B )$ , we get

$$
\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right| < 2 M \Theta _ { \theta } ^ { - k } \sim 2 M \left( \frac { r _ { l } } { \Theta _ { \sigma } } \right) ^ { k } , \quad k \ge 1 .
$$

Case 2, $\sigma ( x )$ is continuous up to $( \nu - 1 )$ -th derivative (in the sense of Lemma 1), so $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ and $\frac { \partial f _ { \theta } ( x ) } { \partial \theta }$ are also smooth in the same sense, with $\tilde { \nu } = \nu - 1$ for $\theta = b _ { l } , r _ { l } , \tilde { \nu } = \nu$ for $\theta = w _ { l }$ . Due to Lemma 1, we have

$$
\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right| \le \frac { 2 V } { \pi ( k - \nu ) ^ { \tilde { \nu } + 1 } } .
$$

Depending on the curve $\sigma ( x )$ , the total variation V of $\frac { \mathrm { d } ^ { \tilde { \nu } } } { \mathrm { d } x ^ { \tilde { \nu } } } \frac { \partial f _ { \theta } } { \partial \theta }$ in $[ - 1 , 1 ]$ can have diferent asymptotic behavior with respect to parameter θ. For Relu $\sigma ( x ) = \operatorname* { m a x } ( 0 , x )$ , $\nu = 1$ , TV $\left( \sigma ^ { ( \nu ) } ( r _ { l } x ) \right) = r _ { l }$ , assume the turning point $- b _ { l } / r _ { l } \in [ - 1 , 1 ]$

$$
V = \left\{ \begin{array} { l l } { r _ { l } } & { \theta = w _ { l } } \\ { w _ { l } r _ { l } } & { \theta = b _ { l } } \\ { w _ { l } ( r _ { l } + \operatorname* { m a x } ( 0 , 2 b _ { l } ) ) } & { \theta = r _ { l } } \end{array} \right. .
$$

## 3.2. Gradient Descent

The trajectory of $\theta ( t )$ in gradient descent (GD) algorithm can be rewritten in ODE as a limit of small step size (with or without $\alpha ,$ depending on how you discretize the ODE):

$$
\frac { \mathrm { d } \theta } { \mathrm { d } t } = - \alpha \left[ \frac { \partial L ( \theta ) } { \partial \theta } \right] ^ { T } ,\tag{12}
$$

where $\alpha$ is the learning rate, $. ^ { T }$ means transpose of the Jacobian matrix inside, and

$$
\left[ { \frac { \partial { \cal L } ( \theta ) } { \partial \theta } } \right] \triangleq \left[ { \frac { \partial { \cal L } ( \theta ) } { \partial \theta _ { 1 } } } \quad { \frac { \partial { \cal L } ( \theta ) } { \partial \theta _ { 2 } } } \quad \cdot \cdot \cdot \right] .
$$

The dynamics of the GFS coeficients $c _ { k } ( \theta ( t ) )$ can be derived with help of Eq. (8) and Eq. (12):

$$
\begin{array} { r l } & { \frac { \mathrm { d } \left( c _ { k } ( \theta ( t ) ) - \hat { y } _ { k } \right) } { \mathrm { d } t } = \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \cdot \frac { \mathrm { d } \theta } { \mathrm { d } t } } \\ & { \qquad = - \alpha \displaystyle \sum _ { k ^ { \prime } = 0 } ^ { \infty } \left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] \left[ \frac { \partial c _ { k ^ { \prime } } ( \theta ) } { \partial \theta } \right] ^ { T } ( c _ { k ^ { \prime } } ( \theta ) - \hat { y } _ { k ^ { \prime } } ) , \quad k = 0 , 1 , \dots . } \end{array}\tag{13}
$$

Note that, formally, a system of linear equation

$$
\frac { \mathrm { d } \vec { r } } { \mathrm { d } t } = - \alpha H \vec { r } ,\tag{14}
$$

with constant symmetric matrix H and spectral decomposition $H = P \Lambda P ^ { - 1 }$ has solution:

$$
\vec { r } ( t ) = e ^ { - \alpha H t } \vec { C } = P e ^ { - \alpha \Lambda t } P ^ { - 1 } \vec { C } ,\tag{15}
$$

where $\vec { C }$ is an arbitrary constant vector.

It is clear that the eigenvalues of H govern the dynamics of Eq.(14).

With substitution

$$
\vec { r } ( t ) = \left[ c _ { k } ( \theta ( t ) ) - \hat { y } _ { k } \right] _ { k \in \mathbb { N } _ { 0 } } ,\tag{16}
$$

$$
H = H _ { c ( \theta ) } \triangleq \left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] _ { k \in \mathbb { N } _ { 0 } } \left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] _ { k \in \mathbb { N } _ { 0 } } ^ { T } ,\tag{17}
$$

where $[ \cdot ] _ { k \in \mathbb { N } _ { 0 } }$ means vertically stack the terms indexed by k to form a matrix. $\mathrm { e . g . }$

$$
\left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] _ { k \in \mathbb { N } _ { 0 } } \triangleq \left[ \frac { \frac { \partial c _ { 0 } ( \theta ) } { \partial \theta _ { 1 } } } { \frac { \partial c _ { 1 } ( \theta ) } { \partial \theta _ { 1 } } } \quad \frac { \partial c _ { 0 } ( \theta ) } { \partial \theta _ { 2 } } \quad \cdot \cdot \cdot \right] .
$$

We see the study of eigenvalues could greatly helps the understanding of Eq.(13), despite $H _ { c ( \theta ) }$ is now time dependent. Now we turn to the analysis of matrix $H _ { c ( \theta ) }$

## 4. Structure of the matrix H

H is an ∞×∞ matrix, for simplicity, assume it is finitely truncated. The truncation is reasonable due to the fast decaying property of $c _ { k } ( \theta )$ . Analysis of the infinite dimension problem directly is under consideration.

Now we need to know some details of $\Upsilon _ { \theta } ( x )$ for the investigation of matrix H.

## 4.1. 1-Hidden Layer DNN

The 1-Hidden Layer DNN with parameter $\theta = \{ a _ { l } , w _ { l } , b _ { l } \} _ { l = 1 } ^ { L }$ and input range $x \in [ - 1 , 1 ]$ is:

$$
\Upsilon _ { \theta } ( x ) \triangleq \sum _ { l = 1 } ^ { L } a _ { l } \sigma ( w _ { l } x + b _ { l } ) .\tag{18}
$$

Due to (11), matrix H of Eq.(17) is computed from derivatives of $\Upsilon _ { \theta }$ :

$$
\frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } = \left\{ \begin{array} { l l } { \sigma ( w _ { l } x + b _ { l } ) } & { \theta = a _ { l } } \\ { a _ { l } \sigma ^ { \prime } ( w _ { l } x + b _ { l } ) } & { \theta = b _ { l } } \\ { a _ { l } \sigma ^ { \prime } ( w _ { l } x + b _ { l } ) x } & { \theta = w _ { l } } \end{array} \right. ~ l = 1 , \ldots , L .\tag{19}
$$

The smoothness of $\sigma ( x )$ determines the smoothness of $\frac { \partial \Upsilon _ { \theta } } { \partial \theta }$ and hence the decaying speed of $\frac { \partial c _ { k } ( \theta ) } { \partial \theta }$ along k. Notably there are two common cases:

Case 1 $\sigma ( x )$ is analytical in a neighborhood of $[ - 1 , 1 ]$ in complex plane (in the sense of Lemma 3), so $\Upsilon _ { \theta } ( x )$ and $\frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta }$ are analytical also. Let’s further assume that there is no singularity in a neighborhood of the real axis <sup>2</sup> for the domain of $\sigma ( x )$ in complex plane. i.e. the nearest singularity is $\pm \mathrm { i } B \ ( B > 0 )$ away from the real axis. For example, $\sigma ( x ) = { \mathrm { s i g m o i d } } ( x ) =$ $1 / ( 1 + \exp ( - x ) )$ , has singularity $x = \operatorname { i } ( \pi + 2 n \pi ) \ \forall n \in \mathbb { Z } ,$ thus $B = \pi$

In this setup, the Bernstein ellipse parameter <sup>3</sup> is $\Theta _ { \sigma ( x ) } \geq B + \sqrt { 1 + B ^ { 2 } }$

So the Bernstein ellipse parameter for $\begin{array} { r } { \frac { \partial \Upsilon _ { \theta } } { \partial \theta } \mathrm { i s } \Theta _ { \sigma ( w _ { l } x ) } = B / w _ { l } + \sqrt { 1 + ( B / w _ { l } ) ^ { 2 } } } \end{array}$ we get

$$
\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right| < 2 M \Theta _ { \sigma \left( w _ { l } x \right) } ^ { - k } = 2 M \left( B / w _ { l } + \sqrt { 1 + \left( B / w _ { l } \right) ^ { 2 } } \right) ^ { - k } , \quad k \ge 1\tag{20}
$$

$$
\approx 2 M \left( \Theta _ { \sigma ( x ) } / w _ { l } \right) ^ { - k } \quad ( \mathrm { w h e n } \ w _ { l } \ll B )\tag{21}
$$

Case $\begin{array} { r l } { { 2 } } & { { } \sigma ( { x } ) } \end{array}$ is continuous up to $( \nu - 1 )$ -th derivative (in the sense of Lemma 1), so $\Upsilon _ { \theta } ( x )$ also. And $\frac { \partial \Upsilon _ { \theta } ( \dot { x } ) } { \partial \theta }$ are continuous up to $( \tilde { \nu } - 1 )$ -th derivative, with $\tilde { \nu } = \nu - 1$ for $\theta = b _ { l }$ and $w _ { l }$ , or ${ \tilde { \nu } } = \nu$ for $\theta = a _ { l }$ . By lemma 1, we have upper bound

$$
\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right| \leq \frac { 2 V } { \pi ( k - \tilde { \nu } ) ^ { \tilde { \nu } + 1 } } \quad ( \forall k > \tilde { \nu } ) .\tag{22}
$$

## 4.2. Graded Matrix

Both Eq.(20) and Eq.(22) show the decay of $\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right|$ with respect to k. Thus the matrix $\operatorname { E q . } ( 1 7 )$ will also have certain decaying property. Specifically, there exist symmetric matrix B of bounded spectral norm and diagonal matrix D such that

$$
H = D B D\tag{23}
$$

where matrix D can be defined as $D = \mathrm { d i a g } ( \left[ \Theta _ { \sigma ( \operatorname* { m a x } ( w _ { l } ) x ) } ^ { - k } \right] _ { k \in \mathbb { N } _ { 0 } } )$ or $D =$ diag $\boldsymbol { \cdot } \left[ ( \boldsymbol { k } + 1 ) ^ { \mathrm { m i n } ( v ) - 1 } \right] \boldsymbol { k } _ { \in \mathbb { N } _ { 0 } } )$ , depending on the smoothness of $\sigma ( x ) . \mathrm { O r } ,$ a more straightforward way is to define $D \ = \ \mathrm { d i a g } ( [ H _ { k k } ] _ { k \in \mathbb { N } } )$ . And $B =$ $D ^ { - 1 } H D ^ { - 1 }$ where pseudo-inverse of D is used if necessary. In the later definition of $D ,$ it is easy to show that $D _ { k k } = O ( \Theta _ { \sigma ( \operatorname* { m a x } ( w _ { l } ) x ) } ^ { - k } )$ or $D _ { k k } =$ $O ( k ^ { - \operatorname* { m i n } ( \tilde { \nu } ) - 1 } )$ . The matrix H with structure Eq.(23) is called graded matrix.

For matrix that has structure Eq.(23), due to Min-max theorem of eigenvalues, we have

$$
\lambda _ { k } ( H ) \leq D _ { k k } ^ { 2 } \lambda _ { \operatorname* { m a x } } ( B ) \quad \forall k \in \mathbb { N }\tag{24}
$$

where $\lambda _ { k } ( H )$ is the k-th eigenvalue of H sorting in descending order. Due to $H _ { j j } = D _ { k k } ^ { 2 } B _ { j j }$ , Eq. (24) can also written as (assume $B _ { j j } \neq 0 )$

$$
\lambda _ { k } ( H ) \leq H _ { j j } \frac { \lambda _ { \operatorname* { m a x } } ( B ) } { B _ { j j } } \quad \forall k \in \mathbb { N } .\tag{25}
$$

Another way to derive it is by the Schur–Horn theorem. The sorted array $\left[ H _ { r ( j ) r ( j ) } \right] _ { j \in \mathbb { N } _ { 0 } }$ of diagonal $[ H _ { j j } ] _ { j \in \mathbb { N } _ { 0 } }$ satisfies

$$
\sum _ { j = 1 } ^ { k } H _ { r ( j ) r ( j ) } \leq \sum _ { j = 1 } ^ { k } \lambda _ { j } \quad k \in \mathbb { N }\tag{26}
$$

and

$$
\sum _ { j = 1 } ^ { \infty } H _ { r ( j ) r ( j ) } = \sum _ { j = 1 } ^ { \infty } \lambda _ { j }\tag{27}
$$

Eq. (27) minus Eq. (26), with the fact that $\lambda _ { j } \ge 0 , H _ { j j } \ge 0$ , we get

$$
\lambda _ { k } \leq \sum _ { j = k } ^ { \infty } \lambda _ { j } \leq \sum _ { j = k } ^ { \infty } H _ { r ( j ) r ( j ) } .\tag{28}
$$

For example, if $\left| \frac { \partial c _ { k } ( \theta ) } { \partial \theta _ { j } } \right| \le C _ { 0 } \Theta ^ { - k } \ \forall k , j$

$$
| H _ { j k } | \le C _ { 0 } ^ { 2 } L \Theta ^ { - j - k }\tag{29}
$$

$$
\lambda _ { k } \leq C _ { 0 } ^ { 2 } L \frac { \Theta ^ { - 2 k } } { 1 - \Theta ^ { - 2 } }\tag{30}
$$

## 5. Related Problems That Encounter Similar Matrix H

## 5.1. Hessian of Loss Function

The hessian of loss function is

$$
\frac { \partial L ( \theta ) } { \partial \theta } = \int _ { \Omega } ( \Upsilon _ { \theta } ( x ) - y ( x ) ) \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } \rho ( x ) \mathrm { d } x\tag{31}
$$

$$
\nabla _ { \theta } \otimes \nabla _ { \theta } L ( \theta ) = \int _ { \Omega } \left( \left[ \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } \right] ^ { T } \cdot \left[ \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } \right] + ( \Upsilon _ { \theta } ( x ) - y ( x ) ) \nabla _ { \theta } \otimes \nabla _ { \theta } \Upsilon _ { \theta } ( x ) \right) \rho ( x ) \mathrm { d } x\tag{32}
$$

Define

$$
H _ { L ( \theta ) } \triangleq \int _ { \Omega } \left[ \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } \right] ^ { T } \cdot \left[ \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta } \right] \rho ( x ) \mathrm { d } x .\tag{33}
$$

When the θ reaches the global optimal solution $\theta ^ { * }$ such that ${ \Upsilon } _ { \theta ^ { \ast } } ( x ) \equiv$ $y ( x )$ , the hessian become

$$
\nabla _ { \boldsymbol { \theta } } \otimes \nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ^ { * } ) = H _ { L ( \boldsymbol { \theta } ^ { * } ) } .\tag{34}
$$

When writing in terms of GFS,

$$
\left[ H _ { L ( \theta ) } \right] _ { i j } = \left. \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta _ { i } } , \frac { \partial \Upsilon _ { \theta } ( x ) } { \partial \theta _ { j } } \right. _ { \rho } = \sum _ { k = 0 } ^ { \infty } \frac { \partial c _ { k } ( \theta ) } { \partial \theta _ { i } } \cdot \frac { \partial c _ { k } ( \theta ) } { \partial \theta _ { j } } ,
$$

we have

$$
H _ { L ( \theta ) } = \left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] _ { k \in \mathbb { N } _ { 0 } } ^ { T } \left[ \frac { \partial c _ { k } ( \theta ) } { \partial \theta } \right] _ { k \in \mathbb { N } _ { 0 } } .\tag{35}
$$

The matrix $H _ { L ( \theta ) }$ and $H _ { c ( \theta ) } \ ( \mathrm { E q . } ( 1 7 ) )$ are difered only by the multiplication order of the transposed matrix, therefore they have exactly the same eigenvalues. For simplicity, define $\lambda _ { k } \ = \ 0$ when k exceed corresponding matrix dimension.

$$
\lambda _ { k } ( H _ { L ( \theta ) } ) = \lambda _ { k } ( H _ { c ( \theta ) } ) \quad ( \forall k \in \mathbb { N } )\tag{36}
$$

Thus from Sec. 4.2 we know that the landscape of loss is extremely flat due to the vast near zero eigenvalues as shown in Eq.(24).

## 5.2. Efective Dimension of Base Function

The following equation are arised in Extreme learning machine [8], or fixing weights $w _ { l } , b _ { l }$ fitting only $a _ { j }$ in $\operatorname { E q } .$ .

Define

$$
H _ { \Upsilon } = \int _ { \Omega } \left[ \sigma ( w _ { l } x + b _ { l } ) \right] _ { l = 1 \cdots L } \left[ \sigma ( w _ { l } x + b _ { l } ) \right] _ { l = 1 \cdots L } ^ { T } \rho ( x ) \mathrm { d } x ,
$$

and

$$
\tilde { H } _ { \Upsilon } = \sum _ { j = 1 } ^ { J } \left[ \sigma ( w _ { l } x _ { j } + b _ { l } ) \right] _ { l = 1 \cdots L } \left[ \sigma ( w _ { l } x _ { j } + b _ { l } ) \right] _ { l = 1 \cdots L } ^ { T } .
$$

It is clear that $H _ { \Upsilon }$ has the same eigenvalue decaying property as $H _ { c ( \theta ) }$ or $H _ { L ( \theta ) }$ . And $\tilde { H } _ { \Upsilon \cdots }$ . Ideally, the parameters $[ a _ { l } ] _ { l = 1 \cdots L }$ can be solved through

$$
\begin{array} { r } { \left[ \sigma ( w _ { 1 } x _ { 1 } + b _ { 1 } ) \quad \sigma ( w _ { 2 } x _ { 1 } + b _ { 2 } ) \quad \cdots \quad \sigma ( w _ { L } x _ { 1 } + b _ { L } ) \right] \left[ a _ { 1 } \right] } \\ { \sigma ( w _ { 1 } x _ { 2 } + b _ { 1 } ) \quad \sigma ( w _ { 2 } x _ { 2 } + b _ { 2 } ) \quad \cdots \quad \sigma ( w _ { L } x _ { 2 } + b _ { L } ) } \\ { \vdots } \\ { \sigma ( w _ { 1 } x _ { J } + b _ { 1 } ) \quad \sigma ( w _ { 2 } x _ { J } + b _ { 2 } ) \quad \cdots \quad \sigma ( w _ { L } x _ { J } + b _ { L } ) \Big ] \left[ a _ { 2 } \right] = \left[ \begin{array} { c } { y ( x _ { 1 } ) } \\ { y ( x _ { 2 } ) } \\ { \vdots } \\ { y ( x _ { J } ) } \end{array} \right] } \end{array}
$$

by its normal equation

$$
\tilde { H } \Upsilon \left[ \begin{array} { c } { a _ { 1 } } \\ { a _ { 2 } } \\ { \vdots } \\ { a _ { L } } \end{array} \right] = \sum _ { j = 1 } ^ { J } [ \sigma ( w _ { l } x _ { j } + b _ { l } ) ] _ { l = 1 \cdots L } y ( x _ { j } )
$$

However, due to the analysis above $\tilde { H } _ { \Upsilon }$ is numerically not full rank.

## Acknowledgment

It was a great pleasure to work with the authors of [1]. In Oct. 2018, the fruitful gethering discussion, generate core idea and initial development of this theory, and we see light that might comming through the mistory of internal workings of ANN.

## Code availability

The code to help reproduce the main conslutions of this article is here https://github.com/bewantbe/ANN\_GFS mostly structures to help build orthogonal polynomial representations and simulting a shallow ANN. An early version of this article can be found here: https://github.com/bewan tbe/dynamics\_dnn, note the file named dynamics\_1dnn.lyx.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

AI and AI-assisted technologies is not used in the construction of the Theory at all. In fact, the theory is constructed before the AI-era (2023), around 2018, . The AI was used to correct gramma and spellings of this script. Some AI coding was used to generate Figure xxxx. The author take full resposability to all content of the article.

## References

[1] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, Z. Ma, Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks, Communications in Computational Physics 28 (5) (2020) 1746–1767, arXiv: 1901.06523. doi:10.4208/cicp.OA-2020-0085. URL http://arxiv.org/abs/1901.06523 1, 5.2

[2] R. Basri, M. Galun, A. Geifman, D. Jacobs, Y. Kasten, S. Kritchman, Frequency Bias in Neural Networks for Input of Non-Uniform Density, in: Proceedings of the 37th International Conference on Machine Learning, PMLR, 2020, pp. 685–694. URL https://proceedings.mlr.press/v119/basri20a.html 1

[3] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, A. Courville, On the Spectral Bias of Neural Networks, in: Proceedings of the 36th International Conference on Machine Learning,

PMLR, 2019, pp. 5301–5310. URL https://proceedings.mlr.press/v97/rahaman19a.html 1

[4] A. Jacot, F. Gabriel, C. Hongler, Neural Tangent Kernel: Convergence and Generalization in Neural Networks, arXiv:1806.07572 [cs, math, stat]ArXiv: 1806.07572 (Jun. 2018). URL http://arxiv.org/abs/1806.07572 1

[5] M. Murray, H. Jin, B. Bowman, G. Montufar, Characterizing the Spectrum of the NTK via a Power Series Expansion, arXiv:2211.07844 [cs.LG] (Feb. 2023). doi:10.48550/arXiv.2211.07844. URL http://arxiv.org/abs/2211.07844 1

[6] S. Zhang, H. Zhao, Y. Zhong, H. Zhou, Why shallow networks struggle to approximate and learn high frequencies, Information and Inference: A Journal of the IMA 14 (3) (2025) iaaf022. doi:10.1093/imaiai/iaa f022. URL https://doi.org/10.1093/imaiai/iaaf022 1

[7] H. Wang, S. Xiang, On the convergence rates of Legendre approximation, Mathematics of Computation 81 (278) (2012) 861–877. 3.1

[8] G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, Extreme learning machine: Theory and applications, Neurocomputing 70 (1) (2006) 489–501. doi: 10.1016/j.neucom.2005.12.126. URL http://www.sciencedirect.com/science/article/pii/S09252 31206000385 5.2

[9] D. Gottlieb, S. A. Orszag, Numerical analysis of spectral methods: theory and applications, Vol. 26, Siam, 1977. Appendix A

[10] L. N. Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013. Appendix A, 1, 2, 3, 4, 5, 6, 7

[11] G. Szegö, Orthogonal Polynomials, American Mathematical Society, Providence, RI, 1975. 8

## Appendix A. Lemmas

See Ref. [9, 10]. Orthogonal polynomial is an active field in 1910s see Bernstein et al.

## Appendix A.1. Chebyshev Approximations

Appendix A.1.1. Theorems for absolutely continuous functions.

Lemma 1 (Theorem 7.1 in [10]). Assume $f , \ldots , f ^ { ( \nu - 1 ) }$ be absolutely continuous on $[ - 1 , 1 ]$ and $f ^ { ( \nu ) }$ is of bounded variation V. Then for $\forall k \geq \nu + 1$ ， the Chebyshev coeficients of f satisfy

$$
| a _ { k } | \leq { \frac { 2 V } { \pi k ( k - 1 ) \cdots ( k - \nu ) } } \leq { \frac { 2 V } { \pi ( k - \nu ) ^ { \nu + 1 } } } .
$$

Lemma 2 (Theorem 7.2 in [10]). The same condition as Lemma 1, then for $\forall n > \nu ,$ its Chebyshev projections satisfy

$$
\| f - f _ { n } \| _ { \infty } \leq \frac { 2 V } { \pi \nu ( n - \nu ) ^ { \nu } }
$$

and its Chebyshev interpolants satisfy

$$
\| f - p _ { n } \| _ { \infty } \leq \frac { 4 V } { \pi \nu ( n - \nu ) ^ { \nu } } .
$$

Theorems for analytical functions.

Lemma 3 (Theorem 8.1 in [10]). Assume f analytic in [−1, 1] be analytically continuable to the open Bernstein ellipse $E _ { \rho } \ ( n o t e ^ { 4 } ) _ { ; }$ , where it satisfies $| f ( x ) | \leq M$ for some M. Then its Chebyshev coeficients satisfy $| a _ { 0 } | \le M$ and

$$
\begin{array} { r } { | a _ { k } | \leq 2 M \rho ^ { - k } , \quad k \geq 1 . } \end{array}
$$

Lemma 4 (Theorem 8.2 in [10]). The same condition as Lemma 3, then for $\forall n \geq 0$ its Chebyshev projections satisfy

$$
\| f - f _ { n } \| _ { \infty } \leq { \frac { 2 M \rho ^ { - n } } { \rho - 1 } }
$$

and its Chebyshev interpolants satisfy

$$
\left\| f - p _ { n } \right\| _ { \infty } \leq { \frac { 4 M \rho ^ { - n } } { \rho - 1 } } .
$$

## Appendix A.2. General Theorems

General theorems about polynomial approximations.

Lemma 5 (Theorem 16.1 in [10]). Let $f \in C \left( [ - 1 , 1 ] \right)$ have degree n Chebyshev projection $f _ { n }$ , Chebyshev interpolant $p _ { n }$ , and best approximant $p _ { n } ^ { * } , n \geq$ 1. Then

$$
\left\| f - f _ { n } \right\| _ { \infty } \leq \left( 4 + { \frac { 4 } { \pi ^ { 2 } } } \log ( n + 1 ) \right) \left\| f - p _ { n } ^ { * } \right\| _ { \infty } ,
$$

$$
\left\| f - p _ { n } \right\| _ { \infty } \leq \left( 2 + { \frac { 2 } { \pi } } \log ( n + 1 ) \right) \left\| f - p _ { n } ^ { * } \right\| _ { \infty } .
$$

Appendix A.2.1. Two theorems about near-best approximation.

Lemma 6 (Theorem 12.1 in [10]). Given $f \in C \left( [ - 1 , 1 ] \right)$ , let $\rho \ ( 1 \leq \rho \leq$ $\infty )$ be the parameter of the largest Bernstein ellipse $E _ { \rho }$ to which $f$ can be analytically continued, and let $\left\{ p _ { n } \right\}$ be the interpolants to $f$ in any sequence of grids $\{ x _ { k } \}$ in $[ - 1 , 1 ]$ such that

$$
\operatorname * { l i m } _ { n  \infty } M _ { n } ^ { 1 / n } = \frac { 1 } { 2 } , \quad M _ { n } = \operatorname * { s u p } _ { x \in [ - 1 , 1 ] } | l ( x ) | , \quad l ( x ) = \prod _ { k = 0 } ^ { n } ( x - x _ { k } ) ,
$$

then the errors satisfy

$$
\operatorname* { l i m } _ { n \to \infty } \| f - p _ { n } \| _ { \infty } ^ { 1 / n } = \rho ^ { - 1 } .
$$

Lemma 7 (Theorem 15.1 in [10]). Let Λ be the Lebesgue constant for a linear projection L of $C \left( { \left[ { - 1 , 1 } \right] } \right)$ onto $\mathcal { P } _ { n }$ . Let f be a function in $C \left( \left[ - 1 , 1 \right] \right)$ , $p \ = \ L f$ the corresponding polynomial approximant to $f ,$ and $p ^ { * }$ the best approximation. Then

$$
\left\| f - p \right\| _ { \infty } \leq ( \Lambda + 1 ) \left\| f - p ^ { * } \right\| _ { \infty }
$$

where the Lebesgue constant is defined as

$$
\Lambda \triangleq \operatorname* { s u p } _ { f } \frac { \| p \| _ { \infty } } { \| f \| _ { \infty } } = \operatorname* { s u p } _ { x \in [ - 1 , 1 ] } \sum _ { j = 0 } ^ { n } \left| l _ { j } ( x ) \right| , \quad l _ { j } ( x ) = \frac { \prod _ { k \neq j } ( x - x _ { k } ) } { \prod _ { k \neq j } ( x _ { j } - x _ { k } ) } .
$$

Lemma 6 and Lemma 7 imply that lim $\mathfrak { i } _ { n \to \infty } \Lambda _ { n } ^ { 1 / n } = 1$ concludes near best convergent for $f \in C \left( [ - 1 , 1 ] \right)$ ).

## Lemma 8. GFS approximation

For a general weight function ρ, Szegő’s Theorem 12.7.3 [11] gives the corresponding convergence result for expansions in orthonormal polynomials. Assume that ρ satisfies the standard Szegő-class conditions; a standard suficient condition is

ρ(x) > 0 almost everywhere,

$$
\int _ { - 1 } ^ { 1 } { \frac { | \log \rho ( x ) | } { \sqrt { 1 - x ^ { 2 } } } } \mathrm { d } x < \infty .
$$

Let $\rho _ { \star }$ denote the parameter of the largest Bernstein ellipse $E _ { \rho _ { \exists } }$ in whose interior g is analytic. Then the GFS coeficients satisfy

$$
\operatorname* { l i m } _ { k \to \infty } \operatorname* { i n f } _ { } | c _ { k } | ^ { - 1 / k } = \rho _ { \star } .\tag{A.1}
$$

Consequently, the GFS converges to g inside $E _ { \rho _ { \star } } ,$ , uniformly on every compact subset of the ellipse, and diverges outside it. Thus, for admissible weight functions, the asymptotic geometric decay of the GFS coeficients is governed by the same Bernstein ellipse as in the Chebyshev case. The weight function changes the polynomial basis and the prefactors, whereas the ellipse of convergence is determined by the analyticity of g.