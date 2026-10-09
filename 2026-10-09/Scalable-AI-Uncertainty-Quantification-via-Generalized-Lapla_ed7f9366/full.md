# Scalable AI Uncertainty Quantification via Generalized Laplace Active Subspaces

Wouter N. Edeling<sup>1,3</sup> and Peter V. Coveney<sup>2,4\*</sup>

<sup>1\*</sup>Scientific Computing Group, CWI, Science park 123, Amsterdam, 1098XG, the Netherlands.

<sup>2</sup>Centre for Computational Science, University College London, 20 Gordon Street, London, WC1H 0AJ, UK.

<sup>3</sup>Faculty of Electrical Engineering, Mathematics and Computer Science, University of Twente, Hallenweg 15, Enschede, 7500AE, the Netherlands. <sup>4</sup>Advanced Research Computing Centre, University College London, Gordon Street, London, WC1E 6BT, UK.

\*Corresponding author(s). E-mail(s): p.v.coveney@ucl.ac.uk; Contributing authors: edeling@cwi.nl;

## Abstract

Reliable uncertainty quantification (UQ) is essential for deploying neural networks in scientific and high-stakes applications, but full Bayesian inference over the network parameters is computationally infeasible. We propose a low-rank generalized Laplace approximation for neural-network UQ based on a small number of data-informed curvature directions. Starting from a generalized Bayesian posterior defined through an empirical loss, we construct a local Gaussian approximation around a pretrained set of weights in this active curvature subspace. The posterior variances in the retained subspace are available in closed form, and the prior variance is calibrated by an empirical Bayes procedure. The generalized Bayesian formulation allows us to compare two posterior scalings: the standard Bayesian scaling associated with the summed negative log likelihood, and a mean-loss scaling in which the empirical loss is normalized by the number of data. A central finding is that the standard scaling induces a data-size dependent contraction of the posterior variance in the leading active directions. In regression problems, this can force the low-rank framework to retain additional weak-curvature directions in order to achieve nominal coverage of calibration data. When posterior samples are propagated through the non-linear network, these additional directions can degrade the coherence of the predictive intervals

and shift the posterior predictive mean away from the pretrained model. In contrast, the generalized mean-loss scaling yields a more stable, lower dimensional active subspace and produces calibrated, coherent predictive confidence intervals. These results indicate that generalized Laplace active subspaces provide a practical and scalable route to calibrated uncertainty quantification in neural networks. We demonstrate the method on two regression problems, and a classification case involving an autoregressive molecular language model.

Keywords: uncertainty quantification, neural networks, Laplace posterior, active subspaces

## 1 Introduction

One of the biggest open questions concerning artificial intelligence (AI) is the reliability and trustworthiness of the predictions which emanate from its output. After training, AI software operating in inference mode can spew out predictions concerning the behaviour of a new system in the same class as those it has been trained on in virtually instantaneous fashion. Such predictions can have enormous value in many real world scenarios. In some situations, qualitative classifications sufice while others require only a modicum of accuracy to be beneficial; still others require accurate and precise predictions or they are not actionable. This last scenario is the one which is most often necessary in scientific and engineering contexts.

To use the results of a calculation inside a scientific procedure, one generally needs to know that the predictions are reliable within defined and suitably limited bounds of uncertainty. Actionable predictions must therefore not only be timely; they need to be of a suficient level of accuracy and/or precision to be used with confidence. The problem with AI methods in general is that what they output often appears plausible and may pass muster as such for some applications; but in more quantitative scientific and mathematical domains, plausibility is not suficient: we need to be clear about the level of certainty of its predictions. The key assessments of validation, verification and uncertainty quantification are regularly visited on physics-based models, e.g. for weather and climate prediction [1, 2], turbulence modelling [3, 4], tsunami hazards [5, 6] or biological applications [7]. In these instances there is an exactly defined mathematical model based on clearly defined equations that typically include a variety of parameters, whereas such systematic analysis are significantly more challenging for AI systems.

One immediately noticeable diference between AI and physics-based codes resides in the number of parameters which arise in the two cases. Whereas physics-based codes typically contain around five parameters [3], possibly several hundreds [8] (extending to a few thousand in some large scale codes), each parameter has a well-defined physical meaning pertaining to the underlying model. In AI software there are many orders of magnitude more, none of which has a physical interpretation, as they are simply free parameters. The impressive fitting of input-output data can be readily performed with so many free parameters in play, particularly when one works with held-out data sets where all manner of optimisations can be conducted. The approach lends itself to the making of overconfident predictions, which can and often do unravel when the trained system is applied to data which it has not previously encountered. We are interested in the uncertainty of such predictions in terms of the uncertainty in the parameter values of the trained model; unlike a physics-based approach, lacking any physical meaning, these parameters cannot be meaningfully adjusted based on insights into the system which can be gleaned from mathematical models we have created.

The purpose of the present paper is to demonstrate that, despite the lack of clarity as to the meaning of the parameters in an AI system, it is possible to obtain predictive uncertainty estimates. A natural framework for quantifying parameter uncertainty is Bayesian inference [9]. In principle, one would place a prior distribution on the neural-network weights, update this prior by conditioning on data, and propagate the resulting posterior distribution through the network to obtain predictive uncertainty. In practice, since the parameter dimension is extremely large, standard sampling strategies are computationally infeasible. Scalable approximations are therefore unavoidable.

Scalable Bayesian neural-network posterior approximations include variational approaches such as Bayes by Backprop [10], which learns a parametric approximation to the posterior distribution over the network weights, typically using a Gaussian variational family. The resulting training objective is an evidence lower bound, consisting of an expected data-fit term and a Kullback-Leibler (KL) divergence to the prior. In practice, the balance between these two terms can be delicate, and may require specialized training strategies such as KL annealing or cyclical schedules [11].

By contrast, the Laplace approximation [12] is a post-hoc posterior approximation. It also models the parameter posterior by a Gaussian distribution, but this Gaussian is centred at a pre-trained set of weights, and its covariance is obtained from the local curvature of the log posterior density rather than by optimizing a variational distribution during training. While the posterior is now conceptually simple, the computational bottleneck shifts to the covariance matrix of the Laplace approximation, as it is too large to form explicitly for many networks. In this work we circumvent this bottleneck via a generalized-Bayes low-rank Laplace approximation. The posterior is restricted to a low-dimensional active subspace spanned by the dominant Gauss-Newton curvature directions of the empirical mean loss. These directions are computed using matrix-free curvature-vector products, so that the full covariance matrix is never formed. Although active subspaces are originally derived from the dominant directions of an uncentered covariance matrix of the gradient of a (physics-based) model [13], the overarching idea is similar, namely to find a linear projection to a low-dimensional subspace relevant for a quantity of interest. For the original active subspace approach this quantity was the model output subject to a prior distribution, in this work it is the posterior predictive uncertainty of a neural network. Within the subspace spanned by these dominant active directions, the posterior variances are available in closed form, while the prior variance is calibrated using a generalized-evidence criterion. The active subspace dimension is selected by empirical coverage on held-out calibration data, yielding predictive intervals that are both calibrated and sharp.

We derive our low-rank Laplace posterior within the framework of generalized Bayesian inference, where the data is connected to the parameters via a loss function, rather than a traditional likelihood function [14]. In particular we contrast summed with mean loss functions, where the latter is scaled by $1 / N$ , N being the number of data records in the training set. The use of the summed loss equates to the standard Bayesian likelihood function. A central observation of the paper is that the scaling of the generalized posterior is not merely a technical detail. The standard Bayesian scaling leads to an increasingly concentrated posterior variance with the number of training data. To avoid such overconfident predictions, a low-rank approximation should then retain many weak-curvature directions with relatively $\mathrm { \ h i g h }$ posterior variance. This increases the computational cost, and reduces the statistical coherence of the predictive output distribution. In other words, when samples from a local Gaussian weight posterior are propagated through the nonlinear network, these weak directions can shift the predictive mean or distort the resulting uncertainty intervals. Existing methods [15, 16] deal with this issue by replacing the full network with its linear expansion. By contrast, our generalized Bayesian mean-loss scaling avoids the purely data-size-driven contraction, results in a much lower-dimensional active subspace and can retain the full non-linear network. Numerical studies demonstrate that the proposed approach yields coherent predictive intervals with only a small number of active directions.

## 2 Results

Let $f ( x ; \theta ) \in \mathbb { R } ^ { M }$ represent our Quantity of Interest $\mathrm { ( Q o I ) }$ , namely the output of a pretrained neural network. Here x are the inputs to the neural network, whereas $\theta \in \mathbb { R } ^ { D }$ is the collection of all connection weights. Furthermore, let $( x , y )$ denote all supervised training data, and specific independent identically-distributed training samples are indexed as $( x _ { n } , y _ { n } ) , n = 1 , \cdot \cdot \cdot , N$ , with $y _ { n } \in \mathbb { R } ^ { M }$ . It is our goal to perform scalable uncertainty quantification of $f$ by propagating a low-rank Bayesian posterior distribution on θ through the network.

Before we describe our results, we give a brief overview of our UQ framework, called generalized Laplace active subspaces. Full details are found in the Methods section; the PyTorch code and data to reproduce the results below are available through [17].

## 2.1 Generalized Laplace active subspaces

Two choices define our Bayesian posterior. First, instead of seeking a posterior distribution $p ( \theta \mid y )$ in the original D-dimensional coordinate system of the connection weights, we seek a low-rank $( d \ll D )$ coordinate system that is most relevant for the posterior predictive uncertainty. Secondly, we do not employ the standard Bayesian belief update $p ( \theta \mid y ) \propto p ( y \mid \theta ) p ( \theta )$ , which uses the likelihood function $p ( y \mid \theta )$ to connect the parameters $\theta$ to the observed data $y .$ For independent Gaussian observations, the corresponding negative log-likelihood is proportional to a summed loss. In contrast, neural networks are typically trained using an average loss $\mathcal { L } _ { N }$ , such as the mean squared error, which includes an additional factor $1 / N$ . To obtain a Bayesianstyle update that is consistent with the average loss used in practice, we adopt the generalized Bayesian framework of [14] instead, which gives rise to the following posterior:

$$
\pi _ { \beta } ( \theta \mid y ) \propto \exp [ - \beta \mathcal { L } _ { N } ( \theta ) ] p ( \theta ) ,\tag{1}
$$

see Section 4.1.1 for more details. The so-called ‘temperature’ parameter $\beta$ controls the contribution of the loss relative to the prior. This formulation includes the standard Bayesian posterior as a special case, obtained by the appropriate choice of $\beta ,$ while also allowing alternative scalings such as the mean-loss scaling considered in this work. It therefore provides a convenient framework for studying how posterior scaling (i.e. the choice of β) afects the dimension and structure of the resulting low-rank posterior approximation.

This posterior approximation is obtained by constructing a local Laplace approximation centered at the pre-trained weights $\theta _ { 0 } ~ \in ~ \mathbb { R } ^ { D }$ . With a commonly-assumed isotropic Gaussian prior $p ( \theta ) \ = \ \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I _ { D } )$ , this results in a Gaussian posterior distribution $\pi _ { \beta } ( \theta \mid y ) = \mathcal { N } ( \theta _ { 0 } , \Sigma )$ , with

$$
\Sigma = \left( \beta H _ { L } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } I _ { D } \right) ^ { - 1 } .\tag{2}
$$

Here, $H _ { L }$ is defined to be the Hessian (curvature) of the mean loss: $H _ { L } : = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { N } ( \boldsymbol { \theta } _ { 0 } ) \in$ $\mathbb { R } ^ { D \times D }$ . For large neural networks this matrix cannot be formed or inverted explicitly, and it may not be positive semi-definite. We therefore replace the full curvature by a low-rank approximation in the dominant eigenspace of the generalized Gauss-Newton matrix G (defined by (28)), G being a positive semi-definite approximation to $H _ { L }$ The d-dimensional dominant G eigenspace with $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { d } ;$

$$
G p _ { j } = \lambda _ { j } p _ { j } , \quad j = 1 , \ldots , d ,\tag{3}
$$

defines the projection matrix $P = [ p _ { 1 } , \dots , p _ { d } ] \in \mathbb { R } ^ { D \times d }$ , used to obtain the posterior perturbations on top of $\theta _ { 0 }$ via

$$
\begin{array} { r } { \theta = \theta _ { 0 } + P z , \quad z \in \mathbb { R } ^ { d } . } \end{array}\tag{4}
$$

In this way uncertainty is represented only via the dominant curvature-informed directions, along which the data was able to inform the parameters well. In the coordinate system of this subspace the Laplace posterior has diagonal covariance, with

$$
\sigma _ { j } ^ { 2 } = \frac { 1 } { \beta \lambda _ { j } + \sigma _ { 0 } ^ { - 2 } } .\tag{5}
$$

Thus, once the leading eigenpairs $( \lambda _ { j } , p _ { j } )$ of $G$ are available, the posterior variances $\sigma _ { j } ^ { 2 }$ in the retained active directions are obtained in closed form. The eigenpairs are computed using automatic diferentiation, matrix-free vector products Gv and the Lanczos algorithm (see section 4.1.5). This avoids explicitly forming G, and makes the construction applicable to networks with a large number of parameters. The resulting approximation is therefore both local and low-rank: local because it describes uncertainty around a single trained solution $\theta _ { 0 }$ , and low-rank because only a small number of active directions are retained.

The prior variance $\sigma _ { 0 } ^ { 2 }$ sets the overall scale of the posterior, and can be calibrated by an empirical-Bayes criterion within the active subspace $w _ { 0 } = P ^ { T } \theta _ { 0 } \in \mathbb { R } ^ { d }$ . The resulting prior variance is found by solving

$$
\| w _ { 0 } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { d } \frac { 1 } { \alpha } - \frac { 1 } { \beta \lambda _ { i } + \alpha } = \sum _ { i = 1 } ^ { d } { \sigma } _ { 0 } ^ { 2 } - { \sigma } _ { i } ^ { 2 } ,\tag{6}
$$

for $\alpha = \sigma _ { 0 } ^ { - 2 }$ . Note that this relation chooses the prior scale such that the norm of the projected parameter vector is consistent with the amount of variance reduction induced by the data in the active subspace.

The posterior predictive distribution is then approximated by sampling (4) and propagating these samples through the neural network $f ( x ; \theta )$ . This sampling strategy is denoted as sampled Laplace. However, the non-linear nature of the network may distort the UQ estimates. So-called linearized Laplace methods [15], which replace the full network with its linear expansion (see (45)), are a means to counteract this. We will contrast these two sampling methods below.

## 2.2 Regression with known observational noise

We start with the same scalar (M = 1) regression function with Gaussian noise as in [18], namely;

$$
\begin{array} { r } { y ( x ) = \sin ( 4 \pi x ) + \sin ( 7 \pi x ) + \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \quad x \in [ 0 , 1 ] , } \end{array}\tag{7}
$$

using a known noise of $\sigma = 0 . 1$

Implementation verification. This example is used to verify the implementation, in particular the matrix-free computation of Gv and the dominant eigenpairs of G. We train a fully-connected feed-forward neural network, using back propagation with Stochastic Gradient Descent and the Adam optimizer [19, 20]. The learning rate was set at 0.001, and we used two hidden layers at 50 neurons each. The dimension of the weight space is $D \ = \ 2 7 0 1$ , small enough to explicitly form the full G matrix (28). The matrix-free Gv vector is contrasted against the corresponding explicit matrix-vector product in Figure 1. The absolute error $\left| G v - ( G v ) _ { e x p l i c i t } \right| \mathrm { i s } \mathcal { O } ( 1 0 ^ { - 8 } )$ in general (using the torch.float32 data type). A similar verification for the matrix-free eigenpairs is displayed in Figure 2.

Standard vs generalized scaling. Next we examine the scaling of the output confidence intervals (CIs) with the temperature parameter $\beta _ { ; }$ with the CIs computed using the sampled Laplace strategy. As described in Section 4.1.1, the value $\beta = N \sigma ^ { - 2 }$ corresponds to the standard Bayesian setting, whereas with $\beta = \sigma ^ { - 2 }$ we employ the generalized Bayesian scaling where the task of standard likelihood function is taken up by the mean-squared loss. For both posterior scalings we determine the active subspace dimension d as described in Section 4.1.7. Essentially, we increase $d \in \mathbb { N }$ starting from $d = 1$ and solve (6) for $\sigma _ { 0 } ^ { 2 }$ every iteration, until our computed 90% CIs achieve empirical coverage of (at least) 90% of held-out calibration data. Our goal here is to compare this minimal required subspace dimension d for both posterior scalings.

![](images/794da35dd9dae36e6c41d54e770d08217f99564a323a3cc21ac28e1ba8fb1238.jpg)  
Fig. 1 The matrix-vector product $G v \in \mathbb { R } ^ { D }$ with $D = 2 7 0 1$ , computed using the matrix-free method outlined in Section $4 . 1 . 5 ,$ and by explicitly forming matrix G via (28). The absolute error $| G v -$ $( G v ) _ { e x p l i c i t } |$ is plotted on the right vertical axis.

Figure 3 displays the result. The generalized Bayesian posterior only required $d = 2$ active subspace directions to achieve coverage, and produces coherent CIs. Conversely, the standard Bayesian scaling of $\beta$ required $d = 3 1$ directions and yields slightly wider, less coherent intervals.

The preceding results were obtained with $N = 5 0$ training data points. We now compare the standard and generalized scaling using $N = 1 0 0 0$ . The results from the generalized Bayesian scaling (not shown) are essentially unchanged by the increase in training data, as it still yields a 2D active subspace that generates consistent CIs. We therefore focus on the 90% CI results of the standard Bayesian scaling, as displayed in Figure 4. Clearly, the quality of the mean prediction and the CIs significantly degraded. Furthermore, instead of $d = 3 1$ , we now required 43 curvature directions to achieve coverage.

Sampled vs linear Laplace. The underfitted results of Figure 4 are a consequence of the of propagating the Laplace posterior through the non-linear network, see also [15]. Instead, the posterior predictive results obtained from linearized Laplace $f _ { l i n }$ , i.e. from (45), are shown in Figure 5 for $N = 1 0 0$ and $N = 1 0 0 0$ for the standard scaling. Due to the linearity of the predictive model, its distribution is Gaussian as well, with a standard deviation $\sigma _ { f }$ given by (46). We therefore perform the coverage calibration using a 95% confidence interval constructed from $f ( x ; \theta _ { 0 } ) \pm 2 \sigma _ { f }$ , which avoids Monte Carlo sampling. We further imposed a maximum subspace dimension of $d _ { m a x } = 5 0$

![](images/a1f2e6c91743663c1d10f0da6e9594f7790b84acd99216cb51e49c57e5d1998a.jpg)  
(a) Eigenvalue fit and errors.

![](images/e8b265e76812e40b6e55aef1cbb1b7daeb5f9664b27444aeab9b35d7281cf1cf.jpg)  
(b) Eigenvector fit and errors  
Fig. 2 The d leading eigenvalues $\lambda _ { j }$ and leading eigenvector p<sub>1</sub> of $G ,$ computed using the matrix-free method outlined in Section 4.1.5, and by explicitly forming matrix G via (28). The absolute errors are plotted on the right vertical axis.

As expected, the posterior predictive mean is improved relative to Figure 4, since the linearized predictive distribution is centered by construction at the expansion-point prediction $f ( x ; \theta _ { 0 } )$ . The confidence intervals are substantially narrower, confirming that the overly wide intervals observed for non-linear posterior sampling are largely caused by propagating weight samples through the non-linear network. However, the linearized predictive distribution is under resolved. For both $N = 1 0 0$ and $N = 1 0 0 0$ the calibration procedure reaches the maximum allowed dimension of 50, covering only approximately 75% and 40% of the calibration data, respectively, with nominal 95% confidence intervals.

![](images/d6e9cc4a13416838661bbae46e4b2628cf28a325eff4d267f278576cdfb40082.jpg)  
(a) $\beta = \sigma ^ { - 2 } ,$ , N = 50, d = 2

![](images/16bdbdeb97130550a61ae9ac3a8f4ef05a5e10e95272ab6d5bd572fcf0f9eeb3.jpg)  
(b) $\beta = N \sigma ^ { - 2 } ,$ , N = 50, d = 31  
Fig. 3 90% confidence intervals for the, a) generalized Bayesian scaling, and b) standard Bayesian scaling. Both CIs were generated by drawing 1000 posterior samples, and propagating these through the network.

## 2.3 Regression with unknown observational noise

The following test case involves the Concrete Compressive Strength dataset from the UCI machine learning repository, see [21] for details. This is a regression test case containing N = 1030 samples and 8 input features describing the concrete mixture, and a scalar output corresponding to compressive strength. We train the same architecture as the preceding test case, except that now it contains approximately $D = 1 0 ^ { 6 }$ connection weights. This case therefore assesses the scalability of the proposed matrix-free low-rank generalized Laplace approximation.

![](images/8dcff6d12098cb954da847dbca85c3dc27513f6341e7be9deae27efedf32f818.jpg)  
Fig. 4 The results for the standard Bayesian scaling $\beta = N \sigma ^ { - 2 }$ with $N = 1 0 0 0$ and $d = 4 3 .$ . The CIs were generated by drawing 1000 posterior samples, and propagating these through the network.

Since the observation noise is unknown, we estimate the noise σ via residuals:

$$
\sigma ^ { 2 } \approx \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \lVert y _ { n } - f ( x _ { n } ; \theta _ { 0 } ) \rVert _ { 2 } ^ { 2 } ,
$$

which is approximately 0.07 for this test case, and we use this value to specify $\beta$ as before. We set the number of training data N to be 90% of the 1030 available data points. The remainder $( N _ { c a l } )$ is used to calibrate d. We execute the calibration procedure until our target coverage of 90% is met, or until a preset maximum active subspace dimension $d _ { m a x } = 5 0$ is exceeded.

Standard vs generalized scaling. The coverage for both the generalized and standard Bayesian scaling is plotted versus d in Figure $6 ( \mathrm { a } )$ . In the case of the former, the target coverage was reached at $d = 3$ , while for the latter it was never reached within the allotted range. In fact, even at $d = 5 0$ only approximately 60% of the calibration data was covered by the 90% CIs.

Sampled vs linear Laplace. The same coverage study was conducted with the linear predictive posterior distribution, using again a 95% confidence interval constructed from $f ( x ; \theta _ { 0 } ) \pm 2 \sigma _ { f }$ . The overall results, see Figure 6(b), are similar compared to the sampled Laplace study. For the generalized scaling, with $d = 4$ active dimensions the coverage criterion is met, while the coverage of the standard scaling stagnates.

Due to the multivariate input $x ,$ we display the generalized Bayesian posterior predictive results in Figure $\mathrm { 7 ( a ) }$ in the form of a parity plot. Here we show the mean predicted compressive strength versus itself with 90% sampled Laplace CIs, and the mean predicted compressive strength versus the calibration data. Visually speaking, for the generalized scaling the CIs are consistent, as the mean is generally centered around

![](images/e1b3357d0fced769ad11738fb146a3374c338471c257dac7b711bb7899472f92.jpg)  
(a) $N = 1 0 0 .$ $d = 5 0 ,$ 75% coverage

![](images/439689127761164e725612bc74fb6a2f4461df3b503d1c015f452b5f60614ba6.jpg)  
(b) $N = 1 0 0 , d = 5 0 ,$ 40% coverage  
Fig. 5 The results for the standard Bayesian scaling $\beta = N \sigma ^ { - 2 }$ combined with the linearized Laplace predictive posterior distribution, computed using $f ( x ; \theta _ { 0 } ) \pm 2 \sigma _ { f }$

the validation data, and the 90% CIs encompass most data without being overly wide.   
The CIs generated by the standard scaling (Figure 7(b)) are indeed overconfident.

Activity scores. The dominant eigenvectors of G define important directions in parameter space. However, we can also extract information about the importance of individual connections weights. Activity scores, originally derived in [22], are (firstorder) sensitivity indices for individual parameters, which can be extracted from the active subspace eigenmodes. Analogously, using the G eigenmodes, we define the activity scores for individual connection weights as:

$$
\nu _ { i } = \sum _ { j = 1 } ^ { d } \lambda _ { j } p _ { j , i } ^ { 2 } , \quad i = 1 , \cdot \cdot \ , D .\tag{8}
$$

![](images/4394d65f9ee0a6c36afee47f30c5926c0029cb4708643bd05af7c65b611b865a.jpg)  
(a) Sampled Laplace posterior prediction, 90% nominal coverage.

![](images/e3ce455a751857221931f55c6e7aa24c4ae9517bf9b8a1afa3918f04906d713c.jpg)  
(b) Linear Laplace posterior prediction, 95% nominal coverage.

Fig. 6 The coverage of the calibration procedure as outlined in Section 4.1.7, versus the active subspace dimension d, for both the standard Bayesian scaling $\beta = N \sigma ^ { - 2 }$ and the generalized scaling $\beta = \sigma ^ { - 2 }$  
![](images/a588abb7fe20dc96ab608c8ace2b2fd6164c99d765f8273ae34069568933cb24.jpg)  
(a) Generalized mean-loss scaling, $d = 3 .$

![](images/337034cb26dadc4243d013f9ad86710424ace1e6cdd4d8c1f05f08d8ce40394a.jpg)  
(b) Standard Bayes scaling, $d = 5 0$  
Fig. 7 The parity plot of the mean compressive strength of concrete, with 90% CIs. We show the standard and generalized scaling results, both computed with sampled Laplace.

Here $p _ { j , i }$ is the i-th component of the j-th eigenvector of G. The activity score $\nu _ { i }$ measures how strongly parameter coordinate i contributes to the retained d-dimensional curvature-informed active subspace, weighted by the importance $\lambda _ { j }$ of each active direction. Since the connection weights have no physical interpretation, we will use (8) primarily to examine if the contribution to the curvature is broadly distributed across the network, or concentrated in a small subset of connection weights. However, rather than plotting individual $\nu _ { i } .$ , we first sort from largest to smallest and instead plot the cumulative scores

$$
\kappa _ { i } = \frac { \sum _ { j = 1 } ^ { i } \nu _ { j } } { \sum _ { j = 1 } ^ { D } \nu _ { j } } , \quad i = 1 , \cdot \cdot , D ,\tag{9}
$$

![](images/be0f870d7c2b301ba18a107975c2e623bd8162d0626fe450052f1dba8a211f86.jpg)  
Fig. 8 The cumulative (normalized) activity scores for our concrete compression strength feed forward neural network with approximately $\dot { D } = 1 0 ^ { 6 }$ weights.

in a log-log plot. The results are shown in Figure 8. We see that the 1000 most-sensitive connection weights (just 0.1% of the total) rapidly saturate and are responsible for most of the curvature sensitivity. In particular, for this case the remaining 99.9% of weights account for 5% of the total sensitivity. This indicates that, in addition to being only d = 3 dimensional, the active subspace is also localised in the high-dimensional weight space.

## 2.4 Autoregressive classification for molecular generative modelling

For our final (classification) test case we consider the autoregressive molecular language model from [23]. Here, molecules are represented as simplified molecular-input line-entry specification (SMILES) strings, a sequence of characters corresponding to atoms, generated sequentially by a recurrent neural network trained to predict a discrete distribution for the next token. In particular, we use a 3-layer LSTM network part of the REINVENT tookit ([24]) with 256 neurons per layer and approximately 1.6M weights, trained on SMILES strings from the ChEMBL database ([25]). The M = 34 tokens that make up the SMILES vocabulary and some example ChEMBL molecules can be found in the Appendix. This makes molecular generation a sequence of autoregressive token-level classification problems since, at each step, the network outputs a softmax distribution over the vocabulary, from which the next token is sampled. We study here how the low-rank posterior uncertainty in the network (for both the standard and generalized scaling) afects free-running autoregressive molecular generation. This setting separates the intrinsic stochasticity of the softmax layer from epistemic variability induced by posterior weight perturbations.

Reducing computational cost via subsampling. The LSTM was trained on 1.25M SMILES strings, each containing a (varying) number of tokens, which results in a total number of data records of $N = 5 8 . 7 \mathrm { M }$ . As each Gv product used in the Lanczos algorithm is an empirical average over the dataset (see (30)), we choose to randomly subsample the data for computational eficiency, computing the eigendecomposition of G using $N _ { s u b }$ SMILES strings instead. Note a similar strategy was employed by [16]. To examine the efect of this subsampling we plot the (absolute value of) the inner product between the dominant eigenvector $p _ { 1 }$ and a reference vector $p _ { 1 } ^ { r e f }$ as a function of $N _ { s u b }$ in Figure 9. This reference vector was computed using $N _ { s u b } = 1 0 k$ , and we replicate this procedure $N _ { r e p } = 5$ times at each subsampling to account for sampling noise, creating 25 possible inner products per $N _ { s u b }$ . While clear variation is present in the replica inner products for $N _ { s u b } = 5 0 0$ and $N _ { s u b } = 1 0 0 0 , \left| \left. p _ { 1 } , p _ { 1 } ^ { r e f } \right. \right|$ soon clusters near 1, indicating that reasonable alignment with the reference eigenvector can be obtained with a small fraction of the total dataset. That said, we note that there is an obvious outlier for $N _ { s u b } = 3 0 0 0$ . This is most-likely caused by a $p _ { 1 }$ which failed to converge during the Lanczos algorithm, such that all 5 corresponding $\left| \left. { p _ { 1 } , p _ { 1 } ^ { r e f } } \right. \right|$ cluster near 0.14 in this case. One simple way to flag such outliers for a given $N _ { s u b }$ is computing the $N _ { r e p } \times N _ { r e p }$ matrix with entries $| \langle p _ { 1 , i } , p _ { 1 , j } \rangle | , i , j = 1 , \cdots , N _ { r e p } , p _ { 1 , i }$ being the i-th replica eigenvector. Ill-converged vectors will cause entries away from 1. All subsequent analysis was performed using $N _ { s u b } = 2 0 0 0 \ \mathrm { S M I L E S }$ , corresponding to $N = 9 3 k$ data points.

![](images/6d52fc1f8de87f1be867bf390a43a9ea598f66ff34c070fffa2ed466e2cb6bf2.jpg)  
Fig. 9 The inner products $\left| \left. { p _ { 1 } , p _ { 1 } ^ { r e f } } \right. \right|$ as a function of $N _ { s u b } \in \{ 5 0 0 \}$ , 1000, 1500, 2000, 2500, 3000}, replicated 25 times per $N _ { s u b }$ . The reference vectors $p _ { 1 } ^ { r e f }$ were computed using $N _ { s u b } = 1 0 . 0 0 0 \mathrm { S M I L E S }$ strings.

Standard vs generalized scaling. Our first results concern the output validity, as not all generated molecules are guaranteed to be chemically correct compounds.

![](images/7dc663da5110f5ae56da334ad83c87bdd57c53c27fb77b0e8687c824644aca5e.jpg)  
Fig. 10 Kernel density estimates of the sequence length for the standard and generalized scaling with d ∈ {1, 2, 3}. The maximum allowed sequence length is 256.

REINVENT verifies the validity in a post processing step using the RDKit [26]. Note that validity here means RDKit can parse the SMILES string into a molecular graph and the graph passes basic chemical sanitization checks. It does not imply druglikeness, synthetic accessibility, or chemical plausibility. Table 1 shows the number of valid molecules out of a set of 10k generated SMILES strings for the baseline REIN-VENT model (i.e. with softmax-only uncertainty and weights fixed at $\theta _ { 0 } )$ , and the perturbed model Laplace models with standard and generalized scaling. When varying the active-subspace dimension from 1 to 3, the standard scaling does not result in a notable change in validity compared to the baseline model. For the generalized scaling, validity notably drops.

Table 1 Number of valid SMILES strings (out of 10k).
<table><tr><td></td><td>baseline</td><td>Laplace (std. scaling)</td><td>Laplace (gen. scaling)</td></tr><tr><td>d = 1</td><td>8692</td><td>8709</td><td>6863</td></tr><tr><td>d = 2</td><td>8692</td><td>8753</td><td>5279</td></tr><tr><td>d = 3</td><td>8692</td><td>8732</td><td>4435</td></tr></table>

The following results were obtained using only valid SMILES strings. Figure 10 shows the kernel density estimates of the sequence length of the molecules (i.e. the number of tokens), computed from 10k generated SMILES strings. Once again only the generalized scaling shows a departure from the baseline results. Especially for d = 3 the generator produces longer SMILES strings, with an increased probability that the maximum allowed sequence length of 256 is reached. Finally, we examine the efect of the Laplace active subspaces on the distribution of several Quantities of Interest (QoIs), described in Table 2. The corresponding distribution are found in Figure 11.

Activity scores. The cumulative activity scores $\kappa _ { i }$ are shown in Figure 12, using d = 2. Note the curve saturates less quickly compared to the concrete compression strength case. Still, the first 7.6% of the activity scores already account for 95% of the total sensitivity.

![](images/fedb58fde9ae22d7bea3e833d91c8f828a31413a2fa853ffee321fba1f93d16e.jpg)

![](images/e056920fc8c4e6cccc1d65ee0d8ede22cfbf02585e18f57c08770ed1d7fb88a8.jpg)

![](images/7a0d1bf99e9593a030b38b19110794e276eb241fb09b4aa1e20ccd8fe8b2f77e.jpg)  
(a) Molecular weight distributions.

(b) SA score distributions.  
![](images/e4fbed78db537addb93b36804226aad4b6404bb585a41524962b21cb97695a12.jpg)  
(c) SlogP distributions.  
Fig. 11 Kernel density estimates of the QoIs for the standard and generalized scaling with $d \in$ {1, 2, 3}.

Table 2 Molecular quantities of interest used to compare generated SMILES distributions within REINVENT.
<table><tr><td>Quantity</td><td>Description</td></tr><tr><td>Molecular weight</td><td>Molecular size based on atomic masses, large values indicate large molecules.</td></tr><tr><td>SA score</td><td>Synthetic Accessibility score, higher values indicate more difficult synthesis.</td></tr><tr><td> $\mathrm { S l o g P }$ </td><td>Estimate of how fat-soluble a molecule is.</td></tr></table>

![](images/c4ea3bcfa5af64df496e4017ba25c4c0703aecd96e78b57d95dd5766326a9c6b.jpg)  
Fig. 12 The cumulative (normalized) activity scores for the REINVENT LSTM with approximately 1.6M weights. Computed using d = 2.

## 3 Discussion

We now discuss our findings and contrast our proposed method with related work.

## 3.1 Posterior scaling and active-subspace dimension

The results show that the scaling of the generalized posterior plays a central role in the quality and eficiency of our low-rank approximation. Under the standard Bayesian scaling, the leading active directions become increasingly contracted with the number of training data, as $\sigma _ { j } ^ { 2 } = ( \beta \lambda _ { j } + \sigma _ { 0 } ^ { - 2 } ) ^ { - 1 }$ with $\beta = N \sigma ^ { - 2 }$ or $\beta = N$ for regression or classification respectively. This forces coverage calibration to retain many weakcurvature directions.

Note that this does not mean the standard scaling is intrinsically defective, since posterior contraction with increasing data is an expected feature of the standard

Bayesian update. The standard Bayesian scaling and generalized mean-loss scaling define do diferent posterior objects, and in our specific context of low-rank posterior UQ for overparameterized neural networks, the generalized scaling can provide a more efective posterior approximation.

To further discuss this behaviour, consider Figure 13. For the analytic regression function (7), we plot the posterior variances and the contraction ratio;

$$
c _ { i } : = \frac { \sigma _ { 0 } ^ { 2 } - \sigma _ { i } ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } = \frac { \beta \lambda _ { i } } { \beta \lambda _ { i } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } } ,\tag{10}
$$

for which $c _ { i }$ ≈ 1 implies that the i-th direction is strongly data-informed (since then $\sigma _ { i } ^ { 2 } \to 0 )$ , and values close to zero denote prior-dominated directions $( \sigma _ { i } ^ { 2 }  \bar { \sigma } _ { 0 } ^ { 2 } )$ . Under the standard Bayesian scaling (Figures 13(b)-13(d)), while the leading directions are almost fully contracted (posterior dominated), the posterior variances remain close to zero over much of the leading spectrum, due to the scaling with N: $\sigma _ { i } ^ { 2 }$ ≈ $1 / ( N \sigma ^ { - 2 } \lambda _ { i } )$ when $c _ { i } \approx 1$ . Coverage calibration therefore selects directions much deeper in the spectrum, where the posterior variances increase rapidly, as the $\sigma _ { i } ^ { 2 }$ span multiple orders of magnitude. Consequently, nominal coverage is obtained only after adding directions that introduce substantial parameter-space variability along weak-curvature modes. This efect becomes more pronounced with more training data; in Figure 13(d) the variance along the leading active directions is reduced by an order of magnitude compared to Figure 13(b). As a result, $d = 4 3$ was required to meet the coverage criterion, which exacerbates the aforementioned problems of incorporating weak curvature modes with high variance. In the non-linear network, these larger perturbations may move samples outside the local region in which the Laplace approximation is accurate, resulting in wider and less stable predictive intervals, as shown in Figure 4. A better mean posterior prediction can be obtained with a lower d, at the cost of overconfident CIs, or by utilizing the linear predictive distribution. Such overconfident CIs were also observed with the UCI concrete compressive strength network with standard scaling, where 50 curvature directions were insuficient to achieve coverage. The slow increase displayed in Figure 6 also suggests a significantly higher subspace dimension is required to achieve target coverage. Moreover, the retained eigenvalue spectrum spans several orders of magnitude, indicating that the additional directions correspond to genuinely low-curvature modes.

While the linear predictive distribution does remove the non-linear sampling artifacts when they occur, the excessive posterior contraction with N induced by the standard Bayesian scaling is still present for linear Laplace. This is clear from Figures 5-7. In contrast, under the generalized Bayesian scaling, suficient uncertainty is retained in the leading active directions, regardless of N or the posterior predictive method, and desired coverage was obtained with only d = 4 or less in all cases. Thus, the uncertainty is concentrated in a small number of coherent active directions rather than spread across many weak-curvature modes.

The increase in the selected active subspace dimension under the standard Bayesian scaling is also computationally consequential. Although $N = 1 0 0 0$ is not large by modern machine-learning standards, the standard scaling already requires a substantially higher posterior rank to achieve the desired coverage compared to $N = 5 0$ , which is a problem that will escalate with larger N. That said, as we showed in Section 2.4, for large datasets one could subsample the data when computing the eigenmodes of G. Still, computing these modes under the standard scaling will require a larger $d ,$ and therefore more Lanczos iterations (see Section 4.1.5). Thus, the standard Bayesian scaling not only shifts uncertainty into weaker-curvature directions, but also increases the computational cost of the low-rank posterior approximation compared to the generalized scaling. In contrast, the generalized Bayesian mean-loss scaling achieves calibrated and sharp confidence intervals with a much smaller active subspace. The proposed approach therefore provides a practical route to calibrated predictive uncertainty for regression-based neural networks.

![](images/5147b745e62ba90e1401ec12af8c1755ad40112bbb4b400f2d6ee746029e6e3e.jpg)  
(a) $\beta = \sigma ^ { - 2 } ,$ , N = 50

![](images/982e0dad28b8e65cf7e9d080798c5c75bf018dac3cbe33566120d2a3a526fb0a.jpg)  
(b) $\beta = N \sigma ^ { - 2 } ,$ , N = 50

![](images/55e4216b4fcc2300e01d05c0b731e1a6ada062818203cf550ac1c044b0cc7de5.jpg)  
(c) $\beta = \sigma ^ { - 2 } ,$ , N = 1000

![](images/9a290681a597fd67af7888b8369700bf77ad32d9f0f4f75245018432cb9996ca.jpg)  
(d) $\beta = N \sigma ^ { - 2 } ,$ N = 1000  
Fig. 13 The contraction ratio (10) and posterior variances (22) for the generalized (left column) and standard (right column) Bayesian scaling, for both $N = 5 0$ and $N = 1 0 0 0$ . These results were computed using the analytic regression function (7).

Similarly, for our autoregressive classification case, the standard scaling resulted in distributions that do not deviate from the baseline REINVENT model in any notable fashion, see Figure 11. The low-rank Laplace perturbations with d = 1, 2 appear to add meaningful variation to the generated molecular distribution while remaining broadly within the regime learned by the pretrained generator. In contrast, d = 3 clearly crosses a stability boundary. The generator can start to produce some long (carbonrich), formally valid yet chemically implausible molecules with extreme SlogP values. Lipinski’s Rule of Five [27] is a rule of thumb used in drug discovery to evaluate if a chemical compound has the right properties to make it an orally active drug in humans. One violating condition of this rule is an SlogP value greater than 5, which occurs at a significantly higher probability once d = 3. This suggests that the addition of third active direction pushes the autoregressive dynamics outside the chemically meaningful region of the model.

Overall, under standard Bayesian scaling, the low-rank Laplace generator remains close to the pretrained baseline for all tested subspace dimensions. Valid-molecule QoI distributions and sequence lengths are not meaningfully afected, indicating that the posterior samples remain in a local generative regime. The reason behind this is the same as in the regression case, and again will be become even more pronounced with increasing (subsampled) data size. In contrast, under generalized scaling, increasing the active-subspace dimension produces progressively stronger deviations from the baseline. The most pronounced efect is in the sequence-length distribution, where a higher d introduces a long right tail, suggesting that generalized posterior perturbations afect the learned stopping behaviour. The same scaling also broadens the QoI distributions, indicating increased chemical exploration at the cost of generative stability.

However, this was only a sensitivity study of the efect of posterior weight perturbations on free running molecule generation. Our pretrained REINVENT model is already reasonably calibrated at the token level, and we therefore did not expect the additional uncertainty of the Laplace posterior to improve the predictive distribution. That said, low-rank Laplace methods have also been proposed for fine-tuned large language models (LLMs). For instance, the low-rank adaptation (LoRA) method of [28] is a means to fine tune an LLM by introducing low-rank updates to selected weight matrices. This deterministic fine-tuning can make the model poorly calibrated, often overconfident. Laplace-LoRA, introduced by [29] places a Laplace posterior over the fine-tuned LoRA parameters to induce variability in the logits/predicted probabilities. The prior precision is tuned using gradient descent, maximizing the Laplace evidence, such that the posterior predictive distribution becomes better calibrated and achieves lower negative log likelihood. Extending our generalized-Bayes active-subspace Laplace approximation to fine-tuned language models is therefore a promising direction for future work. This would shift the role of the Laplace approximation from probing generative sensitivity, as in the present REINVENT experiments, toward correcting overconfidence in fine-tuned autoregressive models.

Finally, the activity scores from Figures 8 and 12 show that the active subspace is concentrated in a very small subset of all connection weights, relative to the total weight space. However, the active subspace $w = P ^ { T } \theta$ itself provides an even more compact representation, as it is a linear combination of connection weights instead of a collection of individual weights. Similar observations have been made in physics-based models, see e.g. [8].

## 3.2 Related work

In recent years several contributions to the problem of uncertainty quantification for neural networks have been made, see [30] or [31] for survey articles. Here we discuss those most relevant to our work.

Our work is related to the subspace inference framework of [32], where Bayesian model averaging is performed within low-dimensional parameter subspaces constructed from random projections, SGD-trajectory PCA, or mode-connecting curves. In contrast, we construct the subspace from the dominant generalized Gauss–Newton curvature directions of the empirical mean loss. In [32] posterior tempering is also used as a calibration device, although here the temperature is a hyper parameter selected by cross-validation, rather than being fixed by the loss function used in generalized Bayes. Our curvature-based setting exposes the diferent scaling issue of increasingly contracted leading active directions with N for the standard scaling. The generalized Bayesian mean-loss scaling avoids this purely data-size-driven contraction. A further distinction is that our generalized Laplace approximation yields a closed-form Gaussian posterior in the active subspace, avoiding Markov-Chain Monte Carlo sampling or variational optimization within the subspace.

Most closely related is the low-rank Laplace subspace construction of [16], whose authors derive an optimal low-dimensional subspace model for approximating the predictive output covariance of a full Laplace approximation and propose a scalable compute strategy. Their construction is formulated in the linearized Laplace setting discussed previously. This linearization yields a closed-form Gaussian on both the networks weights and the output, which [16] use to find the subspace projection matrix P by minimizing the discrepancy between the output covariance of the subspace model and that of the full Laplace approximation. However, the exact construction requires access to quantities that are generally inaccessible for large neural networks, including the full (GGN) curvature matrix and an eigendecomposition of the full output covariance. In practice, scalable variants therefore rely on additional approximations such as diagonal or Kronecker-factored curvature estimates [33].

Our goal is diferent. Rather than computing the output covariance of the full Laplace approximation, we construct a generalized-Bayes Laplace approximation directly in a curvature informed active subspace. The posterior samples are propagated through the non-linear network itself rather than through a linearized predictive model. This distinction is important since linearized Laplace methods keep the posterior predictive mean anchored at the MAP prediction (see section 4.1.8), and can thereby avoid the underfitting behaviour observed when non-linear weight samples are propagated through the network. In our non-linear predictive setting, the size and coherence of the sampled parameter perturbations are therefore central; if the calibrated posterior requires many weak-curvature directions, the resulting samples may leave the local region in which the Laplace approximation is reliable. The generalized Bayesian scaling mitigates this efect by achieving calibrated coverage with a much smaller active subspace, thereby reducing both non-linear sampling artifacts and the number of curvature-vector products required.

Also, the authors of [16] shows that a low-dimensional subspace can approximate the full predictive covariance of the linearized Laplace approximation, but the required subspace dimensions are still in the tens to hundreds in their experiments. While this is tiny compared to the size of the full weight space, our generalized-Bayes active subspace often achieves calibrated coverage with only a handful of curvature directions, preserving the intended low-rank character of the approximation.

There are a number of other contributions that use the Laplace approximation of the posterior. The authors of [34] proposed a scalable Laplace approximation for neural networks using the aforementioned Kronecker-factored curvature estimates of [33]. This approach improves scalability by imposing a layer-wise Kronecker structure on the curvature, whereas our method obtains scalability by restricting the posterior to a low-rank generalized Gauss-Newton active subspace. Consequently, our approach does not require layer-wise Kronecker independence assumptions. The broader practical relevance of Laplace approximations for Bayesian deep learning has been demonstrated by [35], whose authors review and benchmark several scalable Laplace variants and provide a dedicated laplace PyTorch library.

Related active-subspace approaches based on variational inference include [18] and [36]. These methods draw on the classical active-subspace construction, in which the subspace is identified from an uncentered covariance of model gradients [13]. This matrix is typically averaged over a prescribed prior weight distribution. In contrast, our active subspace can be viewed as a Monte Carlo approximation of a data-averaged local curvature matrix. Specifically, the generalized Gauss–Newton matrix is an empirical average over the training data, evaluated at the pretrained weights. Thus, unlike prioraveraged active-subspace constructions, our subspace is tied directly to the local datafit geometry that also defines the generalized Laplace approximation.

## 4 Methods

We now describe the methodological details of the proposed low-rank generalized Laplace framework for uncertainty quantification in neural networks.

## 4.1 Statistics

Our method starts from a generalized Bayesian posterior defined through the empirical mean loss, rather than the summed negative log-likelihood used in standard Bayesian analysis. Around a pretrained set of weights, we then construct a Laplace approximation whose covariance is determined by the local curvature of the loss. Since the full Hessian is intractable and may be indefinite, we replace it by a positive semi-definite generalized Gauss-Newton approximation [37] and compute its dominant eigenspace using matrix-free products computed with automatic diferentiation. The resulting Gaussian posterior is restricted to this low-dimensional active subspace, with the prior variance calibrated by an empirical generalized-Bayes evidence criterion. We end with a discussion on posterior sampling strategies.

## 4.1.1 Generalized Bayes formulation

To define a posterior update, a mechanism to connect the parameters to the data is required. In a generalized Bayesian setting this is achieved via a loss function rather

than through a traditional likelihood function. This allows us to construct a generalized posterior based on the empirical mean loss while retaining the familiar structure of a prior-to-posterior belief update. We define the mean loss function as

$$
\mathcal { L } _ { N } ( \theta , y ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } l _ { n } ( \theta , y _ { n } ) .\tag{11}
$$

We will consider:

$$
l _ { n } = { \frac { 1 } { 2 } } \sum _ { m = 1 } ^ { M } \left( y _ { n } ^ { ( m ) } - f _ { n } ^ { ( m ) } \right) ^ { 2 } \quad { \mathrm { a n d } } \quad l _ { n } = - \sum _ { m = 1 } ^ { M } y _ { m } ^ { ( m ) } \log p _ { n } ^ { ( m ) } ,\tag{12}
$$

i.e. squared-loss and cross-entropy loss terms for regression and classification respectively. In the former $y _ { n } \in \mathbb { R } ^ { M }$ are the (noisy) observations of the true process, whereas in the latter $y _ { n }$ is the M-dimensional one-hot encoded vector with $y _ { n } ^ { ( m ) } = 1$ for $m = c$ and $y _ { n } ^ { ( m ) } = 0$ for $m \neq c ,$ , c being the index of the correct class. Here, the neural-network outputs $f _ { n } ^ { ( m ) } : = f ^ { ( m ) } ( x _ { n } ; \theta )$ are the logits that determine the softmax probabilities

$$
p _ { n } ^ { ( m ) } = \frac { \exp ( f _ { n } ^ { ( m ) } ) } { \sum _ { j } \exp ( f _ { n } ^ { ( j ) } ) } .\tag{13}
$$

The generalized Bayesian (unnormalized) posterior is given by [14] as:

$$
\tilde { \pi } _ { \beta } ( \theta \mid y ) = \exp \left[ - \beta \mathcal { L } _ { N } ( \theta , y ) \right] p ( \theta ) ,\tag{14}
$$

where we assume the following prior:

$$
p ( \theta ) = \mathcal { N } \left( 0 , \sigma _ { 0 } ^ { 2 } I _ { D } \right) .\tag{15}
$$

Here, $I _ { D }$ is the D-dimensional identity matrix and $\beta > 0$ is a so-called temperature parameter, which typically is a hyper parameter used to control how random the output of a neural networks is. Here we will use it to distinguish between generalized and standard Bayesian setting. Further note that the prior variance of the connection weights $\sigma _ { 0 } ^ { 2 } \in \mathbb { R }$ will play an important role in subsequent analysis. Finally, we write the normalized posterior as $\pi _ { \beta } = \tilde { \pi } _ { \beta } / Z _ { \beta }$ , with the normalizing constant $Z _ { \beta }$ denoted as the marginal likelihood or the evidence.

In the case of a regression problem with Gaussian noise, the following statistical model is often assumed to relate the model $f$ to the data $y ;$

$$
y _ { n } = f ( x _ { n } ; \theta ) + \epsilon _ { n } , \quad \epsilon _ { n } \sim { \mathcal { N } } \left( 0 , \sigma ^ { 2 } I _ { M } \right) .\tag{16}
$$

Here, $\sigma ^ { 2 }$ is the noise variance, which is assumed known, or estimated from residuals. Under this model, the standard Bayesian posterior is recovered by setting $\beta = N \sigma ^ { - 2 }$ in (14), which is equivalent to replacing $\mathcal { L } _ { N }$ with a summed loss. In contrast, the generalized posterior based on the mean squared loss is obtained using $\beta = \sigma ^ { - 2 }$ . For classification with softmax cross-entropy, there is no analogous Gaussian noise variance. Similarly, $\beta = N$ corresponds to the summed categorical negative log-likelihood of the standard Bayesian posterior, whereas the generalized posterior based on the mean cross-entropy loss requires $\beta = 1$

## 4.1.2 Laplace approximation of the generalized posterior

We define $l _ { \beta } ( \theta )$ as the negative log of (14), i.e.;

$$
l _ { \beta } ( \theta ) : = - \log \tilde { \pi } _ { \beta } = \beta \mathcal { L } _ { N } - \log p ( \theta ) \propto \beta \mathcal { L } _ { N } + \frac { 1 } { 2 \sigma _ { 0 } ^ { 2 } } \| \theta \| _ { 2 } ^ { 2 } .\tag{17}
$$

The most-likely maximum a-posteriori (MAP) parameter value solves

$$
\hat { \theta } = \operatorname * { a r g m i n } _ { \theta } l _ { \beta } ( \theta ) .\tag{18}
$$

We will use the pretrained weights $\theta _ { 0 }$ in place of the MAP value, which is a common approximation (also used by e.g. [34] and [15]) unless the network was trained using L2 weight decay with regularization strength consistent with the values of $1 / ( 2 \sigma _ { 0 } ^ { 2 } )$ and $\beta .$ In our approach, $\theta _ { 0 }$ serves as the anchor point for the Laplace approximation, originally applied by [12] to small neural networks. This approximation models the posterior distribution $\pi _ { \beta } ( \theta \mid y )$ as a Gaussian distribution, with a covariance structure defined from (the inverse of) the curvature of the log-likelihood function. The underpinning idea here is that the log likelihood quickly drops away (i.e. has high curvature) when the data is able to inform the parameters well. For generalized Bayes, the covariance structure is derived from the curvature of the loss function instead. To arrive at such an expression a second-order Taylor expansion of $l _ { \beta } ( \theta )$ around $\theta _ { 0 }$ is made

$$
l _ { \beta } ( \boldsymbol { \theta } ) \approx l _ { \beta } ( \boldsymbol { \theta } _ { 0 } ) + \frac { 1 } { 2 } \left( \boldsymbol { \theta } - \boldsymbol { \theta } _ { 0 } \right) ^ { T } \nabla _ { \boldsymbol { \theta } } ^ { 2 } l _ { \beta } ( \boldsymbol { \theta } _ { 0 } ) \left( \boldsymbol { \theta } - \boldsymbol { \theta } _ { 0 } \right) ,\tag{19}
$$

where under the assumption that $\theta _ { 0 }$ is suficiently close to a stationary point of $l _ { \beta }$ , the first-order term drops out: $\nabla _ { \theta } l _ { \beta } ( \theta _ { 0 } ) = 0$ . Given (19), the posterior $\pi _ { \beta }$ now becomes a Gaussian $\mathcal { N } ( \theta _ { 0 } , \Sigma )$ :

$$
\pi _ { \beta } ( \theta | y ) \propto \exp \left[ - l _ { \beta } ( \theta ) \right] \propto \exp \left[ - \frac 1 2 \left( \theta - \theta _ { 0 } \right) ^ { T } \Sigma ^ { - 1 } \left( \theta - \theta _ { 0 } \right) \right] ,\tag{20}
$$

with

$$
\Sigma = \left( \nabla _ { \theta } ^ { 2 } l _ { \beta } ( \theta _ { 0 } ) \right) ^ { - 1 } = \left( \beta \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { N } ( \theta _ { 0 } ) - \nabla _ { \theta } ^ { 2 } ~ \log p ( \theta ) \right) ^ { - 1 } = \left( \beta H _ { L } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } I _ { D } \right) ^ { - 1 } .\tag{21}
$$

Note that here $H _ { L }$ is defined to be the Hessian of the mean loss, $H _ { L } : = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { N } ( \boldsymbol { \theta } _ { 0 } ) \in$ $\mathbb { R } ^ { D \times D }$

The Gaussian form is thus a consequence of truncating the Taylor expansion around $\theta _ { 0 }$ at the quadratic term, it is not statement that the full neural-network posterior is indeed normally distributed, which may well be multi modal and non-Gaussian. The Laplace approximation is therefore a pragmatic, local posterior estimate around one trained solution, not as a faithful global description of all plausible weights. We examine its usefulness through its posterior predictive behaviour, i.e. by the resulting posterior distribution of the neural-network output.

The Laplace approximation gives a closed-form expression for the posterior on $\theta .$ Unlike UQ approaches based on (low-rank) variational inference (e.g. [18, 36]), no additional training is therefore required. Still, computing and inverting $H _ { L }$ is computationally intractable in neural networks due to the size of $\theta .$ To overcome this limitation we will construct a computationally tractable closed-form expression for the posterior variance.

## 4.1.3 Low-rank posterior variance

By examining (21), we observe that in the principal coordinate system spanned by the eigenvectors from $H _ { L } p _ { j } = \lambda _ { j } p _ { j }$ , the posterior variances $\sigma _ { j } ^ { 2 }$ become;

$$
\sigma _ { j } ^ { 2 } = \frac { 1 } { \beta \lambda _ { j } + \sigma _ { 0 } ^ { - 2 } } , \quad j = 1 \cdots , D .\tag{22}
$$

In practice, a low-rank framework is obtained by retaining only the first $d \ll D$ dominant eigenvalues $\lambda _ { j }$ of an approximation to the Hessian. The quality of this approximation depends on how well the approximate Hessian captures directions of significant curvature (see Section 4.1.4), as well as on the choice of prior variance $\sigma _ { 0 } ^ { 2 }$

It is not uncommon to find references which discuss Hessian approximations in the context of the sum loss, see e.g. [37]. As discussed, in the present work we use the loss from generalized Bayes rather than the summed negative log-likelihood. This corresponds to a temperature-scaled posterior where the parameter $\beta$ controls how strongly the data loss is weighted relative to the prior (see (14)). For $\beta = \sigma ^ { - 2 }$ or $\beta = 1$ (mean regression $/$ classification loss), the likelihood contribution is normalized by the number of data points. This normalization prevents the curvature eigenvalues from scaling linearly with $N$ , such that the curvature is governed by the shape of the mean loss landscape, not by the size of the dataset. This is important in our low-rank subspace setting, since as N increases, the posterior variances (22) decrease in the standard Bayesian setting with $\beta = N \sigma ^ { - 2 }$ or $\beta = N$ . While in this setting posterior contraction is expected as N grows, in a low-rank projected posterior this can become too aggressive, and the leading active directions become extremely narrow. Using the mean loss avoids this purely sample-size-driven contraction. It makes the eigenvalues describe the average local sensitivity of the trained model rather than the total accumulated data-fit curvature.

Regardless of the choice of $\beta ,$ the form $\sigma _ { j } ^ { 2 } = ( \beta \lambda _ { j } + 1 / \sigma _ { 0 } ^ { 2 } ) ^ { - 1 }$ highlights the interplay between data and prior. Directions with large average curvature (large $\lambda _ { j } )$ lead to stronger posterior contraction, while in flat directions $( \lambda _ { j } \to 0 )$ the posterior variance reverts to the prior, $\sigma _ { j } ^ { 2 } \to \sigma _ { 0 } ^ { 2 }$ . Thus, the prior variance acts as a baseline level of uncertainty in directions not informed by the data.

Since $\sigma _ { 0 } ^ { 2 }$ appears explicitly in the posterior variance and controls the overall scale of uncertainty, it must be chosen carefully. We estimate $\sigma _ { 0 } ^ { 2 }$ using an empirical Bayes procedure in Section 4.1.6. First, we discuss common choices for the approximate Hessian, and means to eficiently extract their d dominant eigenpairs.

## 4.1.4 Approximate Hessians

In general, an issue with using the true Hessian in the covariance expression is that it is not guaranteed to be positive semi-definite. Several approximations are therefore commonly employed, most notably the generalized Gauss–Newton matrix $G ,$ which coincides with the true Hessian only under specific assumptions [37]. Understanding the precise conditions under which these matrices agree, and how they difer otherwise, is essential for justifying their use as curvature surrogates. In the following, we briefly make these relationships explicit and discuss their implications for subspace-based uncertainty quantification.

One can decompose the Hessian into a term involving second derivatives of the model output, and a (more tractable) term that depends only on first-order derivatives. A natural approach is to apply the chain rule to (11), diferentiating with respect to $f$ before taking the θ derivative. The resulting gradient will have entries

$$
\left( \nabla _ { \theta } \mathcal { L } _ { N } \right) _ { i } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { m = 1 } ^ { M } r _ { n } ^ { ( m ) } \frac { \partial f _ { n } ^ { ( m ) } } { \partial \theta _ { i } } , \quad i = 1 , \cdots , D ,\tag{23}
$$

where $r _ { n } ^ { ( m ) }$ are residuals, defined as:

$$
r _ { n } ^ { ( m ) } : = f _ { n } ^ { ( m ) } - y _ { n } ^ { ( m ) } \quad \mathrm { o r } \quad r _ { n } ^ { ( m ) } : = p _ { n } ^ { ( m ) } - y _ { n } ^ { ( m ) } ,\tag{24}
$$

for regression or classification. Note that for the latter, $r _ { n } ^ { ( m ) }$ is a probability residual, which unlike its regression counterpart, is a function of all M $f _ { n } ^ { ( m ) }$ entries (see (13)). Applying the chain rule to (23) once more gives the Hessian entries:

$$
\left( \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { N } \right) _ { i , j } = \frac { 1 } { N } \sum _ { n } \sum _ { m } \left( r _ { n } ^ { ( m ) } \frac { \partial ^ { 2 } f _ { n } ^ { ( m ) } } { \partial \theta _ { i } \theta _ { j } } + \sum _ { k } \frac { \partial r _ { n } ^ { ( m ) } } { \partial f _ { n } ^ { ( k ) } } \frac { \partial f _ { n } ^ { ( k ) } } { \partial \theta _ { i } } \frac { \partial f _ { n } ^ { ( m ) } } { \partial \theta _ { j } } \right) , \quad i , j = 1 , \cdots , D .\tag{25}
$$

When written in matrix form we obtain:

$$
H _ { L } ( \theta ) : = \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { N } = \frac { 1 } { N } \sum _ { n } \sum _ { m } r _ { n } ^ { ( m ) } \nabla _ { \theta } ^ { 2 } f _ { n } ^ { ( m ) } + \frac { 1 } { N } \sum _ { n } \left( \nabla _ { \theta } f _ { n } \right) ^ { T } H _ { l _ { n } } \left( \nabla _ { \theta } f _ { n } \right) .\tag{26}
$$

Here, $\nabla _ { \theta } f _ { n } \in \mathbb { R } ^ { M \times D }$ is the Jacobian of the output, and $H _ { l _ { n } } : = \nabla _ { f } ^ { 2 } l _ { n } \in \mathbb { R } ^ { M \times M }$ is a Hessian with $( H _ { l _ { n } } ) _ { m , k } = \partial ^ { 2 } l _ { n } / \partial f _ { n } ^ { ( m ) } \partial f _ { n } ^ { ( k ) } = \partial r _ { n } ^ { ( m ) } / \partial f _ { n } ^ { ( k ) }$ , leading to

$$
H _ { l _ { n } } = I _ { M } \quad \mathrm { a n d } \quad H _ { l _ { n } } = \mathrm { d i a g } ( p _ { n } ) - p _ { n } p _ { n } ^ { T }\tag{27}
$$

for regression and classification respectively.

The generalized Gauss–Newton (GGN) matrix G is an approximation to $H _ { L }$ that captures (partial) curvature information while avoiding second-order derivatives of the model, and is obtained by simply ignoring the first term of (26);

$$
H _ { L } ( \boldsymbol { \theta } ) \approx G ( \boldsymbol { \theta } ) : = \frac { 1 } { N } \sum _ { n } \left( \nabla _ { \boldsymbol { \theta } } f _ { n } \right) ^ { T } H _ { l _ { n } } \left( \nabla _ { \boldsymbol { \theta } } f _ { n } \right) .\tag{28}
$$

This is convenient as it drops the (expensive) Hessians of $f _ { n } ^ { ( m ) }$ . Moreover, from (26) we can see that this is a well-justified approximation as argued by [37], in the sense that the neglected term vanishes when the residuals are small. For (28) to be exact, it would be suficient if all residuals are zero. However, this is unrealistic, and would be indicative of overfitting. The authors of [15] therefore prefer an alternative justification for the use of $G ,$ , namely that it is equivalent to replacing the network by its firstorder Taylor expansion $f _ { l i n }$ around $\theta _ { 0 }$ , for which $\nabla _ { \theta } ^ { 2 } f _ { l i n } ( \theta _ { 0 } ) = 0$ holds. This leads to so-called linearized Laplace predictive distributions, discussed in Section 4.1.8.

Finally, we note that empirical work has shown that the loss gradient $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { N }$ aligns strongly with the leading eigenvectors of the Hessian, see [38], implying that optimization proceeds primarily within a subspace spanned by a small number of dominant curvature directions. This observation provides support for low-rank posterior approximations involving (approximate) Hessians; if training mainly explores a restricted subspace, then uncertainty quantification should focus on these same directions. Our approach leverages this structure by constructing a posterior covariance aligned with the dominant eigenspace of $G ( \theta )$ . As mentioned, we use the term active subspace to denote the dominant eigenspace of the generalized Gauss-Newton curvature of the empirical mean loss, which is a slight departure from its original definition [13]. That said, the overarching idea is similar, as is the linear projection to the active subspace;

$$
w _ { 0 } = P ^ { T } \theta _ { 0 } \in \mathbb { R } ^ { d } ,\tag{29}
$$

here applied to the connection weights. $P \in \mathbb { R } ^ { D \times d }$ contains the $d \ll D$ dominant (orthonormal) eigenvectors of G. In order to form P for large neural networks, we first discuss matrix-free eigenvalue methods.

## 4.1.5 Matrix-free computation of dominant eigenpairs

In practice, computing the dominant eigenvalues / vectors of $G \in \mathbb { R } ^ { D \times D }$ will be complicated by the dimension of G. For large enough $D ,$ even storing G in memory will not be possible. However, we can use the Lanczos algorithm [39, 40] to approximate the leading eigenpairs, which only requires access to matrix-vector products $G v$ . These products can be computed using automatic diferentiation without ever forming G. If we write $J _ { n } : = \nabla _ { \theta } f _ { n } \in \mathbb { R } ^ { M \times D }$ for brevity, we obtain

$$
G v = \frac { 1 } { N } \sum _ { n } J _ { n } ^ { T } H _ { l _ { n } } J _ { n } v ,\tag{30}
$$

see (28). In general, by the chain rule, the Jacobian-vector product $J _ { n } v \in \mathbb { R } ^ { M \times 1 }$ equals

$$
J _ { n } v = \left. { \frac { \mathrm { d } } { \mathrm { d } \epsilon } } f ( x _ { n } ; \theta + \epsilon v ) \right| _ { \epsilon = 0 } ,
$$

which in turn can be computed with forward-mode automatic diferentiation, see e.g. [41]. Once $J _ { n } v$ is available at the output, pre-multiplication with the known $H _ { l _ { n } } \ ( 2 7 )$ yields $u _ { n } : = H _ { l _ { n } } J _ { n } v \in \mathbb { R } ^ { M \times 1 }$ . If we define the auxiliary scalar loss function $\tilde { L } : = u _ { n } ^ { T } f$ such that its derivative equals $\mathrm { d } \tilde { L } / \mathrm { d } f = u _ { n }$ , then once again by the chain rule;

$$
\frac { \mathrm { d } \tilde { L } } { \mathrm { d } \theta } = J _ { n } ^ { T } u _ { n } = J _ { n } ^ { T } H _ { l _ { n } } J _ { n } v \in \mathbb { R } ^ { D } ,
$$

which equals the n-th term of (30) and can be computed using standard back propagation. Hence, each of the N contributions to Gv can be evaluated using one forwardand one backward-pass through the network, without ever explicitly forming the Jacobian $J _ { n }$ or the matrix G. In practice one uses mini-batches in place of a term-by-term computation. In addition, libraries such as PyTorch have dedicated subroutines for computing the required Jacobian-vector products.

Access to Gv enables scalable computation of the dominant eigenspace of G in highdimensional settings. Briefly, we use the Lanczos algorithm, a well-known iterative method for approximating a few extremal eigenvalues and eigenvectors of a symmetric matrix $\hat { G } \doteq \mathbb { R } ^ { D \times D }$ using only matrix–vector products. Starting from an initial unit vector v<sub>1</sub>, it constructs an orthonormal basis $\{ v _ { 1 } , \ldots , v _ { l } \}$ of the Krylov subspace span $\{ v _ { 1 } , G v _ { 1 } , \ldots , G ^ { l - 1 } v _ { 1 } \}$ via the recursion

$$
v _ { k + 1 } = \frac { G v _ { k } - b _ { k - 1 } v _ { k - 1 } - a _ { k } v _ { k } } { b _ { k } } , \quad b _ { k } = \| G v _ { k } - b _ { k - 1 } v _ { k - 1 } - a _ { k } v _ { k } \| _ { 2 } , \quad a _ { k } = v _ { k } ^ { T } G v _ { k } .\tag{31}
$$

with $b _ { 0 } ~ = ~ 0$ , which follows from a Gram-Schmidt procedure. This yields a lowdimensional symmetric tridiagonal matrix $T _ { l } \in \mathbb { R } ^ { l \times l }$ with $d < l \ll D$ constructed from $a _ { k } , b _ { k }$ coeficients (see e.g. [39]) for which

$$
T _ { l } = V _ { l } ^ { T } G V _ { l }\tag{32}
$$

holds, where $V _ { l } = [ v _ { 1 } , \ldots , v _ { l } ] \in \mathbb { R } ^ { D \times l }$ . The eigenvalues of $T _ { l }$ directly approximate the leading eigenvalues of $G ,$ and $V _ { l } U _ { l } ~ \in ~ \mathbb { R } ^ { D \times l }$ approximate the corresponding eigenvectors, $\bar { U _ { l } } \in \mathbb { R } ^ { l \times l }$ being the eigenvectors of $T _ { l }$

The first d columns of $V _ { l } U _ { l }$ define our projection matrix $P \in \mathbb { R } ^ { D \times d }$ . Note that in order to bring the error down as in Figure 2, it is important to select a suficiently high number of basis functions $l > d$ to span the Krylov subspace. We used $l = d { + } 1 0$ Crucially, note that the method requires only repeated evaluations of $G v$ , making it well-suited for large-scale settings where G cannot be formed explicitly.

## 4.1.6 Empirical generalized Bayesian subspace inference for the prior variance

As previously mentioned, the value of the prior variance plays an important role in (22), and must be chosen with care. However, defining a plausible prior on the connection weights θ is challenging due to their lack of physical meaning. Empirical Bayesian inference [42] provides a solution in this context by allowing us to estimate $\sigma _ { 0 } ^ { 2 }$ directly from the data y. That said, the severe overparameterization of neural networks again poses specific challenges. We found that applying empirical Bayes directly in the full, high-dimensional parameter space forces the optimization toward excessively high $\sigma _ { 0 }$ values to compensate for the many $D - d$ flat, uninformative directions. To resolve this, we perform empirical Bayes in the projected parameter space defined by the active subspace $w = P ^ { T } \theta \in \mathbb { R } ^ { d }$ . Again, let $\dot { P } \in \mathbb { R } ^ { D \times d }$ contain the dominant eigenvectors of G and define

$$
w _ { 0 } = P ^ { T } \theta _ { 0 } , \qquad \theta ( w ) = \theta _ { 0 } + P ( w - w _ { 0 } ) .\tag{33}
$$

The loss restricted to the active subspace is then

$$
\mathcal { L } _ { N } ( w , y ) : = \mathcal { L } _ { N } ( \theta ( w ) , y )
$$

Our analysis will proceed similar to Section 4.1.2. Let $\tilde { \pi } _ { \beta , w }$ be the generalized unnormalized posterior in the w subspace, i.e.;

$$
\tilde { \pi } _ { \beta , w } \left( w \mid y , \sigma _ { 0 } ^ { 2 } \right) = \exp \left[ - \beta \mathcal { L } _ { N } ( w , y ) \right] p ( w \mid \sigma _ { 0 } ^ { 2 } ) ,\tag{34}
$$

with the prior $p ( w \mid \sigma _ { 0 } ^ { 2 } ) = \mathcal { N } \left( 0 , \sigma _ { 0 } ^ { 2 } I _ { d } \right)$ . We find $\sigma _ { 0 } ^ { 2 }$ by maximizing the generalized evidence:

$$
Z _ { \beta , w } ( \boldsymbol { y } \mid \boldsymbol { \sigma } _ { 0 } ^ { 2 } ) = \int \tilde { \pi } _ { \beta , w } \left( w \mid \boldsymbol { y } , \sigma _ { 0 } ^ { 2 } \right) \mathrm { d } w .\tag{35}
$$

The log density is given by

$$
\log Z _ { \beta , w } ( y \mid \sigma _ { 0 } ^ { 2 } ) = \log \int \exp \left[ - l _ { w } ( w \mid \sigma _ { 0 } ^ { 2 } ) \right] ~ \mathrm { d } w .\tag{36}
$$

with $l _ { w } ( w \mid \sigma _ { 0 } ^ { 2 } )$ defined as the negative log unnormalized generalized posterior density:

$$
l _ { w } : = - \log \tilde { \pi } _ { \beta , w } ( w \mid \sigma _ { 0 } ^ { 2 } ) = \beta \mathcal { L } _ { N } ( w , y ) - \log p ( w \mid \sigma _ { 0 } ^ { 2 } )\tag{37}
$$

We now insert a Taylor expansion for $l _ { w }$ around $w _ { 0 } = P ^ { T } \theta _ { 0 }$ into (36), to give rise to a Laplace approximation;

$$
\log Z _ { \beta , w } ( y \mid \sigma _ { 0 } ^ { 2 } ) \approx - l _ { w } ( w _ { 0 } \mid \sigma _ { 0 } ^ { 2 } ) + \log \int \exp \left[ - \frac { 1 } { 2 } \left( w - w _ { 0 } \right) ^ { T } \left[ \nabla _ { w } ^ { 2 } l _ { w } ( w _ { 0 } ) \right] \left( w - w _ { 0 } \right) \right] \mathrm { d } u\tag{38}
$$

As in the full-space Laplace approximation, we assume that $w _ { 0 }$ is suficiently close to a stationary point of $l _ { w } .$ so that the first-order term in the Taylor expansion can be neglected. The second log term contains the integral of the unnormalized Laplace posterior in the active subspace, with $\begin{array} { r c l } { \nabla _ { w } ^ { 2 } \bar { l _ { w } } } & { = } & { P ^ { T } } & { \nabla _ { \theta } ^ { 2 } l _ { \beta } P } & { = } \end{array}$ $P ^ { T } \left[ \beta \bar { \nabla } _ { \theta } ^ { \bar { 2 } } \mathcal { L } _ { N } - \bar { \nabla } _ { ~ \theta } ^ { 2 } \log p ( \theta _ { 0 } \mid \sigma _ { 0 } ^ { 2 } ) \right] P ~ = ~ \beta P ^ { T } \dot { H } _ { L } P + \sigma _ { 0 } ^ { - 2 } I _ { d } ,$ , and therefore equals the reciprocal of the Gaussian normalizing constant. Defining $H _ { w } : = P ^ { T } H _ { L } P , ( 3 8 )$ then becomes;

$$
\begin{array} { l } { { \displaystyle \log Z _ { \beta , w } ( y \mid \sigma _ { 0 } ^ { 2 } ) \propto - l _ { w } ( w _ { 0 } \mid \sigma _ { 0 } ^ { 2 } ) - \frac { 1 } { 2 } \log \left( \operatorname * { d e t } \left[ \beta H _ { w } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } I _ { d } \right] \right) } } \\ { { \displaystyle \propto - \beta \mathcal { L } _ { N } ( w _ { 0 } , y ) + \log p ( w _ { 0 } \mid \sigma _ { 0 } ^ { 2 } ) - \frac { 1 } { 2 } \log \left( \operatorname * { d e t } \left[ \beta H _ { w } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } I _ { d } \right] \right) } } \\ { { \displaystyle \propto \log p ( w _ { 0 } \mid \sigma _ { 0 } ^ { 2 } ) - \frac { 1 } { 2 } \log \left( \operatorname * { d e t } \left[ \beta H _ { w } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } I _ { d } \right] \right) } } \\ { { \displaystyle \propto - \frac { 1 } { 2 \sigma _ { 0 } ^ { 2 } } \lVert w _ { 0 } \rVert _ { 2 } ^ { 2 } - d \log \sigma _ { 0 } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \log \left( \beta \lambda _ { i } + \frac { 1 } { \sigma _ { 0 } ^ { 2 } } \right) } , } \end{array}\tag{39}
$$

The loss $\mathcal { L } _ { N }$ does not depend explicitly on $\sigma _ { 0 } ^ { 2 }$ , and is therefore considered constant in this approximation. In the last line we swap out $H _ { L }$ with an approximate Hessian discussed in Section 4.1.4 (i.e. $H _ { w } \approx P ^ { T } G P )$ , and the low-rank structure of $H _ { w }$ allows us to then compute the determinant using only the d dominant eigenvalues obtained from the matrix-free method of Section 4.1.5. Setting $\alpha = \sigma _ { 0 } ^ { - 2 }$ , we now find that

$$
\begin{array} { l } { 0 = \displaystyle \frac { \mathrm { d } } { \mathrm { d } \alpha } \left( - \frac { \alpha } { 2 } \| w _ { 0 } \| _ { 2 } ^ { 2 } + \frac { d } { 2 } \log \alpha - \displaystyle \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \log \left( \beta \lambda _ { i } + \alpha \right) \right) } \\ { = \displaystyle - \frac { 1 } { 2 } \| w _ { 0 } \| _ { 2 } ^ { 2 } + \frac { d } { 2 \alpha } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \frac { 1 } { \beta \lambda _ { i } + \alpha } . } \end{array}\tag{40}
$$

Solving for the projected weight norm;

$$
\| w _ { 0 } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { d } \frac { 1 } { \alpha } - \frac { 1 } { \beta \lambda _ { i } + \alpha } = \sum _ { i = 1 } ^ { d } { \sigma } _ { 0 } ^ { 2 } - { \sigma } _ { i } ^ { 2 } ,\tag{41}
$$

with $\sigma _ { i } ^ { 2 }$ being the posterior variance (22). Therefore, the resulting $\sigma _ { 0 } ^ { 2 }$ sets the prior scale such that the magnitude of the projected parameter vector is consistent with the amount of variance reduction in the active subspace.

Note that performing this analysis in the full parameter space will lead to an expression involving $\lVert \theta _ { 0 } \rVert _ { 2 } ^ { 2 }$ in place of $\| \boldsymbol { w } _ { 0 } \| _ { 2 } ^ { 2 }$ , which will be very large for overparameterized neural networks. This then leads to an excessively large prior variance $\sigma _ { 0 } ^ { 2 }$

Although (41) defines an implicit and non-linear equation for $\sigma _ { 0 } ^ { 2 }$ , the corresponding function is smooth and strictly decreasing in $\alpha ,$ and therefore admits a unique solution. This solution can be eficiently obtained using standard one-dimensional root-finding methods (we use the bisection method).

## 4.1.7 Calibrating d

The dimension of the active subspace d is a hyperparameter that must be set. For regression problems, we will do so by evaluating the coverage of the output confidence intervals on a calibration data set $( x _ { n } ^ { * } , y _ { n } ^ { * } ) , n = 1 , \cdot \cdot \cdot , N _ { c a l }$ . Let $I _ { \gamma } ( x )$ be the $1 - \gamma$ confidence interval for $\gamma \in [ 0 , 1 ]$ . Our value of d minimizes

$$
d ^ { * } = \arg \operatorname* { m i n } _ { d } \frac { 1 } { N _ { c a l } } \sum _ { n = 1 } ^ { N _ { c a l } } \mathbb { 1 } \left[ y _ { n } ^ { * } \in I _ { \gamma } ( x _ { n } ^ { * } ) \right] \geq 1 - \gamma .\tag{42}
$$

Here $\mathbb { 1 } [ A ]$ is an indicator function returning 1/0 if A is true/false. Note that for each $d ,$ we must perform the empirical Bayesian analysis of Section 4.1.6, such that d and $\sigma _ { 0 }$ are estimated jointly. Calibration in the context of classification entails making sure the correct class in contained in the $1 - \gamma$ set with (approximately) probability $1 - \gamma$

A similar calibration procedure can be applied to a classification network [29, 43]. However, if a classification network is already well calibrated, the additional stochasticity introduced via weight perturbations may diminish calibration, see the Discussion section.

## 4.1.8 Posterior prediction

The posterior variance (22) corresponds to the Laplace approximation restricted to the dominant eigenspace of the curvature matrix. In the full parameter space, directions orthogonal to this subspace have variance $\sigma _ { 0 } ^ { 2 } .$ , corresponding to the prior. While these directions are individually flat, they are vast in number $( D - d \gg d )$ , and can collectively contribute significantly to predictive uncertainty, often resulting in overly wide confidence intervals when posterior samples are propagated through the non-linear network.

Similar to e.g. [16, 18], we therefore omit these directions and obtain a low-rank approximation that concentrates uncertainty in the data-informed subspace. Sampling from this approximate posterior is then performed via

$$
\theta = \theta _ { 0 } + P z , \quad z \sim \mathcal { N } \left( 0 , \mathrm { d i a g } ( \sigma _ { j } ^ { 2 } ) \right) , \quad \sigma _ { j } ^ { 2 } = \frac { 1 } { \beta \lambda _ { j } + \sigma _ { 0 } ^ { - 2 } } , \quad j = 1 , \cdots , d .\tag{43}
$$

Using these samples during a forward pass we can approximate expectations such as

$$
f ( x \mid y ) = \int f ( x ; \theta ) \pi _ { \beta } ( \theta \mid y ) \mathrm { d } \theta ,\tag{44}
$$

via Monte Carlo sampling, pushing the Laplace posterior through the network. As the Laplace posterior is an approximation, the non linear nature of the network may push the mean prediction away from the MAP prediction.

This underfitting problem of sampled Laplace does not primarily mean that the Gaussian posterior family is insuficiently expressive. Rather, it refers to a degradation of the posterior predictive mean caused by propagating local Gaussian weight samples through the non-linear network. As mention in Section 4.1.4, linearized Laplace [15] avoids this by making the a local linear approximation in the predictive model, thereby keeping the posterior predictive mean fixed at the MAP prediction:

$$
f ( x ; \theta ) \approx f _ { l i n } ( x ; \theta ) = f ( x ; \theta _ { 0 } ) + J ( x ) \left( \theta - \theta _ { 0 } \right) ,\tag{45}
$$

where $J ( \boldsymbol { x } ) : = \nabla _ { \boldsymbol { \theta } } f ( \boldsymbol { x } ; \boldsymbol { \theta } _ { 0 } ) \in \mathbb { R } ^ { M \times D }$ . The combination of the linear output and the Gaussian distribution of θ also leads a closed-form predictive covariance $\Sigma _ { f } = J \Sigma J ^ { T }$ where $\Sigma = P \Lambda P ^ { T }$ , with $\Lambda : = \mathrm { d i a g } ( \sigma _ { j } ^ { 2 } )$ is our low-rank Laplace posterior covariance. In the case of scalar regression the (pointwise) posterior predictive variance becomes

$$
\sigma _ { f } ^ { 2 } ( \boldsymbol { x } ) = \sum _ { j = 1 } ^ { d } \sigma _ { j } ^ { 2 } \left( J ( \boldsymbol { x } ) p _ { j } \right) ^ { 2 } \in \mathbb { R } ,\tag{46}
$$

with $p _ { j } \in \mathbb { R } ^ { D }$ denoting the j-th column vector of $P .$

Supplementary information. For this article no Supplementary Materials are available.

Acknowledgements. We thank Dr Ketan Maheshwari at the US Department of Energy Oak Ridge National Laboratory and Dr Alessandro Tibo at AstraZeneca for helpful discussions and support with training REINVENT. We also thank Dr Laura Harbach at Brunel University London for initiating the REINVENT UQ study.

## Declarations

## Funding

P.V.C. is grateful for funding from the UK Engineering and Physical Sciences Research Council under the following project “UK Consortium on Mesoscale Engineering Sciences (UKCOMES)” (Grant No. EP/R029598/1).

## Competing interests

The authors declare no competing interests.

## Ethics approval and consent to participate

Not applicable.

## Consent for publication

Not applicable.

## Data and code availability

The data used to train the neural networks, a pre-trained REINVENT model and Jupyter notebooks to reproduce the results of this article are available at [17].

## Materials availability

Not applicable.

## Author contribution

W.N.E. developed the methodology, implemented the algorithms, performed the numerical experiments, analysed the results, and wrote the initial manuscript. P.V.C. contributed to the conceptual framing, interpretation of results, funding acquisition and manuscript revision. Both authors reviewed and approved the final manuscript.

## Declaration on the use of artificial intelligence tools

The authors used OpenAI’s ChatGPT 5.5 as a tool for language editing. The tool was also used to discuss manuscript presentation, related-work positioning, and the interpretation of selected numerical results. All scientific claims, mathematical derivations, algorithms, references, numerical experiments, and conclusions were reviewed, verified, and finalized by the authors. The authors assume full responsibility for the content of the manuscript.

## Appendix A ChEMBL vocabulary

Here we describe the ChEMBL vocabulary in Table A1 and display some example chemical compounds in Table A2.

Table A1 Vocabulary used by the autoregressive SMILES generator. The vocabulary contains start and stop tokens, bond symbols, branch symbols, ring-closure indices, atom tokens, and charged or aromatic atom tokens.
<table><tr><td>Index</td><td>Token</td><td>Description</td><td>Index</td><td>Token</td><td>Description</td></tr><tr><td>0</td><td>$</td><td>Stop/end token</td><td>17</td><td>Br</td><td>Bromine atom</td></tr><tr><td>1</td><td>一</td><td>Start token</td><td>18</td><td>C</td><td>Aliphatic carbon atom</td></tr><tr><td>2</td><td>#</td><td>Triple bond</td><td>19</td><td>C1</td><td>Chlorine atom</td></tr><tr><td>3</td><td> $\% 1 0$ </td><td>Two-digit ring closure</td><td>20</td><td>F</td><td>Fluorine atom</td></tr><tr><td>4</td><td>(</td><td>Branch opening</td><td>21</td><td>N</td><td>Aliphatic nitrogen atom</td></tr><tr><td>5</td><td>)</td><td>Branch closing</td><td>22</td><td>0</td><td>Oxygen atom</td></tr><tr><td>6</td><td>-</td><td>Single bond</td><td>23</td><td>S</td><td>Sulfur atom</td></tr><tr><td>7</td><td>1</td><td>Ring closure index</td><td>24</td><td>[N+]</td><td>Positively charged nitrogen</td></tr><tr><td>8</td><td>2</td><td>Ring closure index</td><td>25</td><td>[N-]</td><td>Negatively charged nitrogen</td></tr><tr><td>9</td><td>3</td><td>Ring closure index</td><td>26</td><td>[0-]</td><td>Negatively charged oxygen</td></tr><tr><td>10</td><td>4</td><td>Ring closure index</td><td>27</td><td>[S+]</td><td>Positively charged sulfur</td></tr><tr><td>11</td><td>5</td><td>Ring closure index</td><td>28</td><td>[n+]</td><td>Positively charged aromatic nitrogen</td></tr><tr><td>12</td><td>6</td><td>Ring closure index</td><td>29</td><td>[nH]</td><td>Protonated aromatic nitrogen</td></tr><tr><td>13</td><td>7</td><td>Ring closure index</td><td>30</td><td>C</td><td>Aromatic carbon atom</td></tr><tr><td>14</td><td>8</td><td>Ring closure index</td><td>31</td><td>n</td><td>Aromatic nitrogen atom</td></tr><tr><td>15</td><td>9</td><td>Ring closure index</td><td>32</td><td>0</td><td>Aromatic oxygen atom</td></tr><tr><td>16</td><td>=</td><td>Double bond</td><td>33</td><td>s</td><td>Aromatic sulfur atom</td></tr></table>

Table A2 10 SMILES strings from the ChEMBL database.  
No. SMILES   
1 $\operatorname { C C } \left( \mathbf { C } \right) \mathbf { C } \left( \mathrm { N C } \left( = 0 \right) \mathbf { C } \left( \mathrm { N C } \left( = 0 \right) \mathbf { C } \ \mathrm  1 0 C 2 0 C \left( \mathbf { C } \right) \left( \mathbf { C } \right) \mathrm { 0 C 2 C 2 0 C \left( \mathbf { C } \right) \left( \mathbf { C } \right) \mathrm { 0 C 1 2 } } \right) \mathbf { C } \left( \mathbf { C } \right) \mathbf { C } \right) \mathbf { C } \left( 0 \right) = \mathbf { 0 } .$   
2 $\operatorname { F c 1 c c c } ( \mathsf { c c 1 } ) \operatorname { C } ( = 0 ) \operatorname { N N 1 C } ( \operatorname { S C c 2 c c c } ( \mathsf { c c 2 } 2 ) \operatorname { N } ( = 0 ) = 0 ) = \operatorname { N c 2 c c c c c } 2 \operatorname { C 1 } = 0 )$   
3 $\mathtt { C C } \left( \mathtt { C } \right) \mathtt { C N 1 C c 2 c n n n 2 - c 2 c c c } \left( \mathtt { c c 2 C 1 } \right) - \mathtt { c 1 c c c c c t 1 }$   
4 CC $\mathtt { N } ( \mathbf { C } ( = \mathtt { 0 } ) \mathtt { C S c 1 } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { n } \mathtt { c } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { n } \mathtt { n } \mathtt { n } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt  c \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt { c } \mathtt $   
5 CC(C)N1CCN(CC1)C(=O)N1c2ccccc2Sc2ccccc12   
6 C $\mathtt { C 1 } ( \mathtt { C } ) \mathtt { C c } 2 \mathtt { c c } ( 0 \mathtt { C c } 3 \mathtt { c c } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt { C } \mathtt \mathtt { C } \mathtt { C } \mathtt { C } \mathtt \mathtt { C } \mathtt { C } \mathtt { C } \mathtt \mathtt { C } \mathtt { C } \mathtt \mathtt { C } \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt { C } \mathtt \mathtt \mathtt { C } \mathtt \mathtt \mathtt { C } \mathtt \mathtt \mathtt { C } \mathtt $   
7 Cc1cccc(CSc2ccc(cn2)C(=O)Nc2ccc(F)cc2)c1F   
8 $\mathtt { C C C } \left( \mathtt { N 1 N } = \mathtt { C } \left( 0 \right) \mathtt { C 2 } = \mathtt { N c 3 c c } \left( \mathtt { C 1 } \right) \mathtt { c c c 3 C } \left( = 0 \right) \mathtt { C 2 } = \mathtt { C 1 0 } \right) \mathtt { c 1 c c c } \mathtt { c c n 1 }$   
9 $\mathtt { C C } \left( \mathtt { N 1 C } \left( = 0 \right) \mathtt { C 2 D C 3 } \left( \mathtt { C } \right) \mathtt { C = C } \left( \mathtt { C } \right) \mathtt { C = N C 3 C 2 C 1 = 0 } \right) \mathtt { c 1 c c c c c 1 }$   
10 Cc1nc(cs1)-c1cccc(NC(=O)c2ccc3nc4C(=O)NCCCn4c3c2)c1

## References

[1] Palmer, T.N.: Predicting uncertainty in forecasts of weather and climate. Reports on progress in Physics 63(2), 71–116 (2000)

[2] Murphy, J.M., Sexton, D.M.H., Barnett, D.N., Jones, G.S., Webb, M.J., Collins, M., Stainforth, D.A.: Quantification of modelling uncertainties in a large ensemble of climate change simulations. Nature 430(7001), 768–772 (2004)

[3] Cheung, S.H., Oliver, T.A., Prudencio, E.E., Prudhomme, S., Moser, R.D.: Bayesian uncertainty analysis with applications to turbulence modeling. Reliability Engineering & System Safety 96(9), 1137–1149 (2011)

[4] Gorl´e, C., Iaccarino, G.: A framework for epistemic uncertainty quantification of turbulent scalar flux models for reynolds-averaged navier-stokes simulations. Physics of Fluids 25(5) (2013)

[5] Liu, X., Guillas, S.: Dimension reduction for Gaussian process emulation: An application to the influence of bathymetry on tsunami heights. SIAM/ASA Journal on Uncertainty Quantification 5(1), 787–812 (2017)

[6] Selva, J., Tonini, R., Molinari, I., Tiberti, M.M., Romano, F., Grezio, A., Melini, D., Piatanesi, A., Basili, R., Lorito, S.: Quantification of source uncertainties in seismic probabilistic tsunami hazard analysis (sptha). Geophysical Journal International 205(3), 1780–1803 (2016)

[7] Avitabile, D., Cavallini, F., Dubinkina, S., Lord, G.J.: Neural field equations with random data. SIAM/ASA Journal on Uncertainty Quantification 14(2), 534–567 (2026)

[8] Edeling, W.N., Vassaux, M., Yang, Y., Wan, S., Guillas, S., Coveney, P.V.: Global ranking of the sensitivity of interaction potential contributions within classical molecular dynamics force fields. npj Computational Materials 10(1), 87 (2024)

[9] Gelman, A.: Bayesian data analysis. Chapman and Hall/CRC (2013)

[10] Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D.: Weight uncertainty in neural network. In: International Conference on Machine Learning, pp. 1613–1622 (2015). PMLR

[11] Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., Carin, L.: Cyclical annealing schedule: A simple approach to mitigating KL vanishing. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 240–250 (2019)

[12] MacKay, D.J.C.: A practical Bayesian framework for backpropagation networks. Neural computation 4(3), 448–472 (1992)

[13] Constantine, P.G., Dow, E., Wang, Q.: Active subspace methods in theory and practice: applications to kriging surfaces. SIAM Journal on Scientific Computing 36(4), 1500–1524 (2014)

[14] Bissiri, P.G., Holmes, C.C., Walker, S.G.: A general framework for updating belief distributions. Journal of the Royal Statistical Society Series B: Statistical Methodology 78(5), 1103–1130 (2016)

[15] Immer, A., Korzepa, M., Bauer, M.: Improving predictions of Bayesian neural nets via local linearization. In: International Conference on Artificial Intelligence and Statistics, pp. 703–711 (2021). PMLR

[16] Faller, J., Martin, J.: Low rank based subspace inference for the Laplace approximation of Bayesian neural networks. In: The 29th International Conference on Artificial Intelligence and Statistics (2026). https://openreview.net/forum?id=RTwTXQX4gq

[17] Edeling, W.N.: GLAS: Generalized Laplace Active Subspaces. https://github. com/wedeling/GLAS. GitHub repository (2026)

[18] Jantre, S., Urban, N.M., Qian, X., Yoon, B.: Learning active subspaces for efective and scalable uncertainty quantification in deep neural networks. In: ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5330–5334 (2024). IEEE

[19] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

[20] Aggarwal, C.C.: Neural Networks and Deep Learning vol. 10. Springer, ??? (2018)

[21] Yeh, I.: Concrete Compressive Strength. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5PK67 (1998)

[22] Constantine, P.G., Diaz, P.: Global sensitivity metrics from active subspaces. Reliability Engineering & System Safety 162, 1–13 (2017)

[23] Olivecrona, M., Blaschke, T., Engkvist, O., Chen, H.: Molecular de-novo design through deep reinforcement learning. Journal of cheminformatics 9(1), 48 (2017)

[24] Loefler, H., He, J., Tibo, A., Janet, J.P., Voronov, A., Mervin, L.H., Engkvist, O.: Reinvent 4: Modern AI–driven generative molecule design. Journal of Cheminformatics 16(1), 20 (2024)

[25] Gaulton, A., Hersey, A., Nowotka, M., Bento, A.P., Chambers, J., Mendez, D., Mutowo, P., Atkinson, F., Bellis, L.J., Cibri´an-Uhalte, E., et al.: The ChEMBL database in 2017. Nucleic acids research 45(D1), 945–954 (2017)

[26] Landrum, G., Tosco, P., Kelley, B., Cosgrove, D., Vianello, R., Kawashima, E., Jones, G., Dalke, A., Cole, B., Swain, M., et al.: rdkit/rdkit: 2026 03 4. Zenodo (2026)

[27] Lipinski, C.A., Lombardo, F., Dominy, B.W., Feeney, P.J.: Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Advanced drug delivery reviews 23(1-3), 3–25 (1997)

[28] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)

[29] Adam., Y., Maxime, R., Xi, W., Laurence, A.: Bayesian Low-rank Adaptation for Large Language Models (2024). https://arxiv.org/abs/2308.13111

[30] Psaros, A.F., Meng, X., Zou, Z., Guo, L., Karniadakis, G.E.: Uncertainty quantification in scientific machine learning: Methods, metrics, and comparisons. Journal of Computational Physics 477, 111902 (2023)

[31] Wang, K., Cuzzolin, F., Shariatmadar, K., Moens, D., Hallez, H.: A review of uncertainty representation and quantification in neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 48(3), 2476–2495 (2026)

[32] Izmailov, P., Maddox, W.J., Kirichenko, P., Garipov, T., Vetrov, D., Wilson, A.G.: Subspace inference for Bayesian deep learning. In: Uncertainty in Artificial Intelligence, pp. 1169–1179 (2020). PMLR

[33] Grosse, R., Martens, J.: A kronecker-factored approximate fisher matrix for convolution layers. In: International Conference on Machine Learning, pp. 573–582 (2016). PMLR

[34] Ritter, H., Botev, A., Barber, D.: A scalable Laplace approximation for neural networks. In: International Conference on Learning Representations (2018). https://openreview.net/forum?id=Skdvd2xAZ

[35] Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., Hennig, P.: Laplace redux-efortless Bayesian deep learning. Advances in neural information processing systems 34, 20089–20103 (2021)

[36] Nafiz, A., Jantre, S., Urban, N.M., Yoon, B.: Leveraging active subspaces to capture epistemic model uncertainty in deep generative models for molecular design. In: 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP), pp. 1–6. IEEE, ??? (2024). https://doi.org/10.1109/mlsp58920. 2024.10734787 . http://dx.doi.org/10.1109/MLSP58920.2024.10734787

[37] Kunstner, F., Hennig, P., Balles, L.: Limitations of the empirical Fisher approximation for natural gradient descent. Advances in neural information processing systems 32 (2019)

[38] Gur-Ari, G., Roberts, D.A., Dyer, E.: Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754 (2018)

[39] Chen, T.: The Lanczos algorithm for matrix functions: a handbook for scientists (2026). https://arxiv.org/abs/2410.11090

[40] Lanczos, C.: An iteration method for the solution of the eigenvalue problem of linear diferential and integral operators. Journal of research of the National Bureau of Standards 45(4), 255–282 (1950)

[41] Pearlmutter, B.A.: Fast exact multiplication by the Hessian. Neural computation 6(1), 147–160 (1994)

[42] Carlin, B.P., Louis, T.A.: Empirical Bayes: Past, present and future. Journal of the American Statistical Association 95(452), 1286–1289 (2000)

[43] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: International Conference on Machine Learning, pp. 1321–1330 (2017). PMLR