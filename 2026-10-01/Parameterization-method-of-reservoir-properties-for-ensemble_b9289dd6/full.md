# Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN

Marcio A. Sampaio<sup>a1</sup>, Paulo H. Ranazzi<sup>a2</sup>, Martin J. Blunt<sup>b3</sup>

<sup>a</sup>Departamento de Engenharia de Minas e de Petróleo, Escola Politécnica, Universidade de São Paulo, 05508- 030, SP, Brasil.

<sup>b</sup>Department of Earth Science and Engineering, Imperial College London, South Kensington, London, UK <sup>1</sup>Corresponding author. E-mail address: marciosampaio@usp.br (Marcio A. Sampaio). ORCID: 0000-0003- 1125-7218

<sup>2</sup>Contributing author. E-mail address: ranazzi@usp.br (Paulo H. Ranazzi). ORCID: 0000-0002-4515-4797

<sup>3</sup>Contributing author. E-mail address: m.blunt@imperial.ac.uk (Martin J. Blunt). ORCID: 0000-0002-8725-0250

ARTICLE INFO

Keywords:

Variational Autoencoder Generative Adversarial Network

Latent Diffusion

Style-Based Generative Adversarial Network

Latent space reparameterization

Data assimilation

Authorship contribution statement

Author 1: Conceptualization, Methodology, Data Curation, Formal analysis, Investigation, Software, Validation, Visualization, Writing. Author 2: Data curation, Datasets creation, Investigation, Validation, Writing. Author 3: Conceptualization, Data curation, Investigation, Validation, Writing, Visualization, Supervision.

## ABSTRACT

Ensemble smoothers are the most successful and efficient techniques currently available for history matching. However, because these methods rely on Gaussian assumptions, their performance is severely degraded when the prior geology is described in terms of complex facies distributions (non-Gaussian). In this way, for these methods, we need to apply efficient parameterization techniques. Currently, the most efficient methods for performing parameterization are deep learning models. However, given the variety of existing deep learning models, studies have not identified which is most suitable for use with ensemble-based methods, although some important models had already been evaluated. Therefore, trying to fill this gap, this study selected the three leading models found in the literature to determine the best method to employ, highlighting the advantages and disadvantages of each one. Based on a recent literature review, the most promising models selected were VAE-GAN, Latent Diffusion, and StyleGAN models. As a novel aspect of this work, data assimilation with the second generation of StyleGAN (StyleGAN2) model was performed using the latent z-space and intermediate w-space, separately. They were applied in two 2D case studies: one categorical (three facies) and the other continuous. The results demonstrated that all three models are highly efficient, with the StyleGAN2 model standing out for generating samples with geological realism and achieving excellent data matching in the cases studied. Our findings show that performing data assimilation with StyleGAN2 using the intermediate space (w-space) yielded better results than the traditional application in the latent space (z-space). This is due to the fact that ESMDA uses linear updates and the w-space is much more linear and disentangled than the highly entangled z-space, thereby ensuring that the updated vectors remain close to realistic geological patterns. These results were validated using main geostatistical and history matching metrics.

## 1. INTRODUCTION

Ensemble-based methods represent the state-of-the-art for data assimilation (DA) application for history matching of the reservoir models. The main disadvantages of ensemblebased methods are their poor performance in highly nonlinear models, spurious correlations from limited ensemble size, and reliance on Gaussian assumptions, which degrades performance when reservoir prior parameters exhibit non-Gaussian distributions. Among ensemble-based methods, the Ensemble Smoother with Multiple Data Assimilation (ESMDA), proposed by Emerick and Reynolds (2013), is one of the most widely used. For a comprehensive review of the ensemble-based methods, readers are referred to Aanonsen et al. (2009) and Oliver and Chen (2011). Despite these methods have been applied successfully, they sometimes fail to preserve the geological realism of the model in reservoirs with complex facies distributions. This occurs mainly because of the underlying Gaussian assumptions on model parameters that are inherent in these methods. This fact has encouraged an intense research activity to develop Gaussian parameterizations in a latent space that maps into geologicallyrealistic facies realizations. Despite the large number of methods employed in the literature, conventional methods like Level Set (Chang et al., 2010; Luo et al., 2007, 2008), Truncated Pluri‐Gaussian (Beucher and Renard, 2016; Silva and Deutsch, 2017), Distance Transform (Hakim‐Elahi and Jafarpour, 2017), and Normal‐Score Transform (Li et al., 2018) tend toward multi‐Gaussian assumptions and fail to adapt to the elaborate spatial statistics required in the geological models (Ling and Jafarpour, 2024). Recent applications of deep learning have attracted attention due to the quality of the results generated. Next, we highlight the key studies that employed deep learning models in the parameterization problem to be applied in the DA of reservoir models.

Laloy et al. (2017) used a Variational Autoencoder (VAE) to construct a lowdimensional parameterization of binary facies models for data assimilation with Markov Chain

Monte Carlo (MCMC). The results showed that the dimensionality reduction (DR) approach outperforms traditional techniques like principal component analysis (PCA), optimization-PCA (OPCA) and discrete cosine transform (DCT). Two synthetic cases were used to illustrate the effectiveness of the proposed DR-based probabilistic inversion in relation to traditional approaches mentioned before. Later (Laloy et al., 2018), modified the spatial GAN to deal with high dimensional inversion problems. Synthetic 3D cases were tested to prove that GAN could capture the statistical features using low-dimensional latent variables and could perform well for inversion of channel structures.

Canchumuni et al. (2017) used an autoencoder to parameterize binary facies values in terms of continuous variables for history matching with an ensemble smoother. In other work, Canchumuni et al. (2019a) extended the same parameterization using deep belief networks (DBN), which is able to map discrete facies into continuous parameters and reconstruct back to discrete facies. The results showed that the training step of the DBN was very successful in terms of reconstructing the input facies realizations of the validation sets, but the same performance was not observed when we updated the latent vectors of the DBN with ESMDA. As previous work of these authors was based on fully-connected layers, making the computational requirements for training infeasible in practice, Canchumuni et al. (2019b) applied a convolutional variational autoencoder (CVAE) and the ESMDA to investigate the parameterization in three synthetic history-matching problems with channelized facies, two 2D cases and one 3D. The results proved promising compared to previous methods, generating well-defined channelized facies. The proposed procedure outperformed previous results obtained with standard ESMDA, ESMDA with OPCA and DBN parameterizations in terms of the reconstruction of the channel facies. However, this work highlighted the need to improve the reconstruction accuracy, especially in three-dimensional cases, and reduce the computational cost in the training process of models. Results with VAEs alone showed that are

generally superior to standard PCA for non-Gaussian models, but they can produce blurry output of lower quality than GANs, or they may display unrealistic geometries. After, (Canchumuni et al., 2021) applied nine different formulations, including VAE, generative adversarial network (GAN), Wasserstein GAN (WGAN), WGAN with gradient penalty, WGAN with spectral normalization, variational auto-encoding GAN, principal component analysis (PCA) with cycle GAN, PCA with transfer style network, and VAE with style loss in the same channelized facies cases of the previous studies. They also proposed two strategies to allow the use of distance-based localization with the deep learning parameterizations. The results showed that all networks were able to generate realistic facies realizations with welldefined channels, but the DA for water cut values were not so good like images generation. Both localization strategies used in this work were able to solve the ensemble collapse problem, especially found in the VAE model application.

Liu and Durlofsky (2021) propose the 3D CNN-PCA procedure for geological parameterization. This work introduced a new supervised learning-based reconstruction loss, which is used in combination with style loss and hard data loss to handle complex 3D geomodels. The deep learning model was used as a post-processor for PCA, which is a traditional parameterization method. The 3D case studies were three geological scenarios: binary and bimodal channelized systems, and a three-facies channel-levee-mud system. The algorithm was successfully applied for history matching with ESMDA for the bimodal channelized system showing uncertainty reduction, although it requires improvements.

Bao et al. (2022) compared the performance of VAE and GAN in flow and transport DA using ESMDA. Four cases with categorical variables and three cases with continuous variables were conducted to test the performance of coupling ESMDA with deep learning models. The results showed that VAE was more successful in terms of DA performance and GAN was more efficient for reconstructing channel structure with similar properties as in the

training image. The authors highlighted that VAE models don’t need adversarial training like GANs, that is time-consuming and may be unstable.

Ling and Jafarpour (2024) propose the Style‐based Generative Adversarial Networks in the second version (StyleGAN2) for parameterization of complex subsurface flow properties and subsequent ESMDA application comparing the results with CVAE and GAN. This architecture was evaluated it using groundwater pumping tests and two‐phase fluid flow experiments. The results showed that parameterization with StyleGAN2 provide superior performance in terms of reconstruction fidelity and flexibility. The authors highlighted that StyleGAN2 have direct implications for representing complex subsurface heterogeneity using low‐dimensional latent variables, including improved fidelity and control of image attributes. They also highlighted a notable implication for model calibration was the increased regularity of the latent space (z-space), which implies that two points that are close in the latent space also remain close in the original spatial domain. After that, Ahn and Choe (2025) propose a novel method that enhances history matching in reservoir simulations by integrating a geologicalstyle-mixing approach with GAN-based optimization using StyleGAN3, a framework capable of producing a variety of geological styles. Thus, the focus of this work was on generating diverse geological models to expand the initial ensemble to be used in DA, not using StyleGAN3 in the parameterization process. The proposed method was compared with the previously developed convolutional neural network-principal component analysis (CNN-PCA) and demonstrated similar history matching performance.

Federico and Durlofsky (2025) used a variational autoencoder for dimension reduction and a U-Net for the denoising process for latent diffusion model. This model was applied in 2D three-facies (channel-levee-mud) systems obtained visually consistent realizations with samples used and significant uncertainty reduction in the DA process. Although good quality

images were obtained, the production and injection matching results were limited to reducing uncertainties in both cases, indicating a need for improved matching.

Our previous work stemmed from the findings by Canchumuni et al. (2021) and Bao et al. (2022) that GANs are capable of generating high-quality images, albeit with poor matching, whereas VAEs exhibit the opposite behavior. Consequently, we proposed applying the hybrid VAE-GAN method to leverage the advantages of both. Indeed, the VAE-GAN model achieved output quality comparable to that of GANs, alongside assimilation results similar to those of VAEs (Sampaio et al., 2026). We evaluated the results across both categorical and continuous test cases using standard metrics. We also highlighted that VAE-GAN can be time-consuming and unstable, as it requires adversarial training between three networks.

In the current work, inspired by the observations of Karras et al. (2019) regarding the StyleGAN model, specifically that the �-space is much more linear and disentangled than the highly entangled the �-space, we introduce a new parameterization method for non-Gaussian reservoir properties in ensemble-based DA, utilizing the intermediate latent space of second generation of StyleGAN. To demonstrate the effectiveness of this method, we compared our approach with DA using the conventional latent �-space, as well as against state-of-the art models in these problems, such as Latent Diffusion models (Federico and Durlofsky, 2025) and VAE-GAN (Sampaio et al., 2026).

## 2. METHODOLOGY

In this section, we present the methodology developed in this work. It is divided into two main parts. The first part involves modeling and training three types of deep learning models to compare their respective advantages and disadvantages: VAE-GAN, Latent Diffusion and StyleGAN2. The second part consists in making the integration with DA, using the ESMDA to perform history matching. All models used the latent �-space for DA, except for StyleGAN2 model, which is also evaluated using its intermediate latent �-space. The images (realizations) in this study consist of 2D permeability maps defined on a 48 x 48 Cartesian grid.

## 2.1. Deep Learning Models

In this first part, we briefly describe the three deep learning models used in this work.

## 2.1.1. Variational Autoencoder Generative Adversarial Network (VAE-GAN)

The Variational Autoencoder Generative Adversarial Network (VAE-GAN) combines the advantages of VAE and GAN to improve image generation quality while preserving a structured latent representation. The VAE encoder learns a latent representation of the image and imposes a probabilistic distribution on this latent space (�-space), while the VAE decoder reconstructs images from the latent representation. Additionally, the GAN discriminator distinguishes real images from images generated by the decoder. In this architecture, the VAE is responsible for ensuring that the latent space is well structured and regularized, whereas the GAN improves the quality of generated images by forcing the decoder to generate more realistic images. The objective function combines the VAE and GAN losses to balance reconstruction and geological realism (Sampaio et al., 2026).

## 2.1.2. Latent Diffusion Model (LDM)

The genesis of Latent Diffusion Models (LDMs) lies at the convergence of two distinct lines of research in deep generative learning: Denoising Diffusion Probabilistic Models (DDPMs) and Autoencoders (Rombach et al., 2022). The primary motivation for this fusion was to address the core limitation of original DDPMs: the high computational cost of training and inference, inherent to operating directly in high-dimensional pixel space (Ho et al., 2020). The resulting framework is composed of two stage architecture: an autoencoder and a diffusion model. In the autoencoder stage, an encoder compresses input images into a lower-dimensional latent representation (�-space), capturing essential spatial features while discarding unnecessary details. Conversely, the decoder reconstructs images from these latent codes, ensuring that the latent space preserves meaningful information. In the diffusion process, the model learns to generate new latent representations by gradually denoising random noise. During the training step, referred to as the forward process, Gaussian noise is progressively added to clean latent codes across multiple time steps. In this model, a U-Net model learns to predict the noise added at each step, trained with a simple MSE loss between predicted and actual noise. After that, in the generation part, called as reverse process, it starts from pure random noise in the latent space and the trained U-Net iteratively removes noise step by step. In each step predicts and subtracts the noise component, gradually revealing a structured latent code. The clean latent code produced by the reverse diffusion is passed through the decoder to generate the final image.

The LDM used in this work consists of a variational autoencoder (VAE) and a U-Net, similar to the architectures employed by Ronneberger et al. (2015) and Federico and Durlofsky (2025).

## 2.1.3. Style-Based Generative Adversarial Network (StyleGAN)

The Style‐based Generative Adversarial Networks (StyleGAN) architecture emerged from a progressive evolution of generative adversarial networks, driven by the need for greater control over the image synthesis process and higher output quality (Karras et al., 2018, 2019, 2020a, 2020b, 2021). The foundational GAN framework was introduced by Goodfellow et al. (2014), establishing the adversarial training paradigm, where a generator and discriminator compete in a min-max game. However, early GANs suffered from training instability and mode collapse, issues that were partially addressed by architectural innovations. Furthermore, manipulating the latent code to obtain a desired change in the generated image is challenging, since the mapping between the latent and image spaces is highly nonlinear. Consequently, changes to a latent code may simultaneously affect multiple semantic attributes of the generated image, a property usually referred to as entanglement. In this regard, Karras et al. (2019) demonstrated significant improvements in image quality and control over the disentanglement of semantic attributes with

StyleGAN model. In the StyleGAN2, a random latent vector (z-space) drawn from a standard normal distribution is first transformed by a mapping network into an intermediate latent representation (�-space). This transformed representation is then injected at multiple resolution levels of a synthesis network via adaptive instance normalization (AdaIN) operations, which modulate the feature maps by scaling and shifting their normalized activations. This mechanism allows coarse styles to control high-level geological structures, while finer styles govern local heterogeneity. A central aspect of this architecture is the enforcement of Gaussianity within the latent space. The model imposes a statistical regularization that constrains the latent vectors to maintain zero mean, unit variance, and negligible skewness and kurtosis. This is achieved through an additional loss term that penalizes deviations from standard normal moments. As a result, the latent manifold remains smooth and continuous, enabling meaningful interpolation between generated realizations and robust sampling for downstream applications. Overall, the model operates by learning a constrained, smooth mapping from a well-behaved Gaussian latent space to realistic geological images, guided by adversarial feedback, perceptual feature matching, and explicit statistical regularization. The improvements in latent space regularity and predictability, as well as generation of spatial domain images with higher quality and robustness, are among the reasons to use StyleGAN for subsurface flow modeling, and in particular in parameterization of model calibration problems (Ling and Jafarpour, 2024)

## 2.2. Ensemble Smoother with Multiple Data Assimilation (ESMDA)

Emerick and Reynolds (2013) demonstrated that Ensemble Smoother (ES) is equivalent to a single full Gauss-Newton update step. Therefore, to handle with nonlinear forward operators, ESMDA assimilates all the observed data multiple times using an inflated measurement error covariance matrix, relating each iteration with a smaller Gauss-Newton update step. Neglecting model errors, the perfect forward model is expressed as � = �(�), where the non-linear forward operator $g ( \cdot )$ (in our case, the reservoir simulator) maps the model parameters vector $\mathbf { m } \in \Re ^ { N _ { m } }$ to the predicted data vector � $\in \Re ^ { N _ { d } }$ . Here, $N _ { m }$ and $N _ { d }$ are the number of model parameters and observed data points, respectively. The history matching inverse problem aims to estimate the model parameter vector � that best reproduce a set of observed measurements $\mathbf { \Pi } ( \mathbf { d } _ { \mathbf { o b s } } )$ . The observed data are considered a noisy realization of the true reservoir response $( { \bf d } _ { \mathrm { t r u e } } )$ , defined as ${ \bf d } _ { \mathrm { o b s } } = { \bf d } _ { \mathrm { t r u e } } + \epsilon$ . The measurement error � is usually drawn from a zeromean Gaussian distribution, $\epsilon \sim \mathcal { N } ( 0 , \mathbf { C } _ { \mathrm { D } } )$ , with $\mathbf { C } _ { \mathrm { D } } \in \Re ^ { N _ { d } \times N _ { d } }$ representing the covariance matrix of measurement errors. In the ESMDA analysis step, each member � of model parameters vector ensemble is updated using the following equation:

$$
\mathbf { m } _ { j } ^ { i + 1 } = \mathbf { m } _ { j } ^ { i } + \mathbf { C } _ { M D } ^ { i } \bigl ( \mathbf { C } _ { D D } ^ { i } + \alpha _ { i } \mathbf { C } _ { \mathrm { D } } \bigr ) ^ { - 1 } \bigl ( \mathbf { d } _ { o b s , j } ^ { i } - \mathbf { d } _ { j } ^ { i } \bigr )\tag{1}
$$

for $j = ( 1 , \dots , N _ { e } )$ , where $N _ { e }$ denotes the number of ensemble members (ensemble size), $\mathbf { C } _ { M D } ^ { i } \in \Re ^ { N _ { m } \times N _ { d } }$ is the cross-covariance between the model parameters and predicted data, $\mathbf { C } _ { D D } ^ { i } \in \mathcal { R } ^ { N _ { d } \times N _ { d } }$ is the auto-covariance of the predicted data, $\alpha _ { i }$ is the inflation factor that damp the iteration i. The forward step is where each ensemble member $\mathbf { d } _ { j } ^ { i }$ is estimated using the forward operator $\mathbf { d } _ { j } ^ { i } = g ( \mathbf { m } _ { j } ^ { i } )$ for $j = ( 1 , \dots , N _ { e } )$

## 2.3. Geostatistical Metrics

In this section, we introduce the geostatistical metrics used to evaluate the images generation in terms of geological realism and quality.

## 2.3.1. Variogram Mean Squared Error

The variogram measures spatial continuity by quantifying how data values differ as a function of distance. For each lag distance h, the 2D experimental variogram is computed as:

$$
\begin{array} { r } { \gamma ( h ) = \frac { 1 } { 2 N ( h ) } { \sum _ { i , j } } \bigl [ Z ( { x _ { i } } ) - Z ( { x _ { j } } ) \bigr ] ^ { 2 } } \end{array}\tag{2}
$$

where �(�) is the value at location x, and N(h) is the number of point pairs separated by distance h in both horizontal and vertical directions. The MSE between real and generated variograms is then calculated as the mean squared difference over all lag distances.

## 2.3.2. Connectivity Function MSE

The connectivity function quantifies how well a generated spatial model reproduces the connectivity patterns of real data. The connectivity function, denoted as τ(h), measures the probability that two points separated by a lag distance h are not only above a threshold t but also belong to the same connected cluster. The MSE then computes the average squared difference between the connectivity functions of the real dataset and the generated (simulated) dataset across all lag distances. The connectivity function is defined as:

$$
\begin{array} { r } { \tau ( h ) = \frac { \sum _ { i , j } \mathbf { 1 } [ Z ( x _ { i } ) > t \wedge Z \left( x _ { j } \right) > t \wedge c o n n e c t e d a t h ] } { \sum _ { i , j } \mathbf { 1 } [ Z ( x _ { i } ) > t ] } } \end{array}\tag{3}
$$

where 1[⋅] is the indicator function, and the sums run over all point pairs at lag distance h. The MSE is computed between the connectivity functions of real and generated data. Lower MSE values indicate that the generated data faithfully preserves the spatial continuity and clustering behavior of the reference real data.

## 2.3.3. Histogram KL Divergence

This metric compares the full pixel value distributions using the Kullback-Leibler divergence (Kullback and Leibler, 1951). After computing normalized histograms P (real) and Q (generated) with a small epsilon added for numerical stability:

$$
\begin{array} { r } { D _ { K L } ( P \left| \right| Q ) = \sum _ { i } P ( i ) l o g \left( \frac { P ( i ) } { Q ( i ) } \right) } \end{array}\tag{4}
$$

A lower value indicates that the generated distribution closely matches the real data distribution.

## 2.3.4. PCA Correlation

Principal Component Analysis (PCA) is performed on flattened real and generated images. The correlation between corresponding principal components measures how well the global variance structure is preserved:

$$
\begin{array} { r } { { { \rho } _ { k } } = \frac { C o v ( P C _ { r e a l , k } , P C _ { f a k e , k } ) } { { { \sigma } _ { r e a l , k } } . { { \sigma } _ { f a k e , k } } } } \end{array}\tag{5}
$$

The reported metric is the mean of these correlations across the first two components. Values closer to 1 indicate strong structural similarity.

## 2.3.5. MDS MMD

Multidimensional Scaling first projects both real and generated samples into a lowdimensional space. The Maximum Mean Discrepancy then quantifies the distance between these two distributions using a Gaussian kernel:

$$
M M D ^ { 2 } = \operatorname { E } [ K ( X , X ^ { \prime } ) ] + \operatorname { E } [ K ( Y , Y ^ { \prime } ) ] - 2 \operatorname { E } [ K ( X , Y ) ]\tag{6}
$$

where $\operatorname { K } ( \mathbf { x } , \mathbf { y } ) = \exp ( - \gamma \ \lVert \mathbf { x } - \mathbf { y } \rVert ^ { 2 } )$ . The bandwidth $\gamma$ is set adaptively as the inverse of twice the squared median of pairwise distances. A lower MMD indicates the generated samples are statistically indistinguishable from real ones in the MDS embedding space.

## 2.4. History Matching Metrics

In this section, we introduce the metrics used to evaluate the DA results.

## 2.4.1. Normalized data-mismatch objective function

For each member j, the normalized data-mismatch is the difference between simulated and observed data, scaled by the measurements error covariance matrix:

$$
\begin{array} { r } { 0 _ { \mathrm { N _ { d , j } } } = \frac { 1 } { \mathrm { N _ { d } } } \big ( { \bf d } _ { j } - { \bf d } _ { \mathrm { o b s } } \big ) ^ { \mathrm { T } } { \bf C } _ { \mathrm { D } } ^ { - 1 } \big ( { \bf d } _ { j } - { \bf d } _ { \mathrm { o b s } } \big ) } \end{array}\tag{7}
$$

and its average:

$$
\begin{array} { r } { \overline { { 0 _ { \mathrm { N _ { d } } } } } = \frac { 1 } { N _ { e } } \sum _ { \mathrm { j } = 1 } ^ { N _ { e } } { 0 _ { \mathrm { N _ { d , j } } } } } \end{array}\tag{8}
$$

## 2.4.2. Balanced Accuracy

Balanced accuracy is a performance metric used in classification tasks, especially when dealing with imbalanced datasets. Balanced accuracy generalizes naturally to multi-class problems by averaging the recall (true positive rate) for each class. For � classes, balanced accuracy is computed as:

$$
\begin{array} { r } { \mathrm { B a l a n c e d ~ A c c u r a c y } = \frac { 1 } { \mathrm { C } } { \sum _ { \mathrm { i = 1 } } ^ { \mathrm { C } } } \frac { \mathrm { T r u e ~ p o s i t i v e s _ { i } } } { \mathrm { T r u e ~ p o s i t i v e s _ { i } + F a l s e ~ n e g a t i v e s _ { i } } } , \mathrm { f o r ~ c l a s s ~ i } } \end{array}\tag{9}
$$

## 2.4.3. Root Mean Square Error (RMSE)

An ensemble of root mean square error (RMSE) for the ensemble $= \{ \mathbf { m } _ { j } \} _ { j = 1 } ^ { \mathrm { N } _ { e } }$ , with � is the number of parameters and $N _ { e }$ is the number of members of ensemble, with respect to the true model $\mathbf { m _ { \mathrm { t r u e } } }$ is:

$$
\begin{array} { r } { \mathrm { R M S E } \left( \mathbf { M } , \mathbf { m } _ { \mathrm { t r u e } } \right) = \left\{ \frac { \left\| \mathbf { m } _ { j } - \mathbf { m } _ { \mathrm { t r u e } } \right\| _ { 2 } } { \sqrt { M } } \right\} _ { j = 1 } ^ { N _ { e } } } \end{array}\tag{10}
$$

## 2.4.4. Spread

The spread is defined as the average RMSE between each ensemble member and the ensemble mean:

$$
\begin{array} { r } { \mathrm { S p r e a d } ( \mathbf { M } ) = \frac { 1 } { N _ { e } } \sum _ { j = 1 } ^ { N _ { e } } \frac { \left\| \mathbf { m } _ { j } - \bar { \mathbf { m } } \right\| _ { 2 } } { \sqrt { M } } } \end{array}\tag{11}
$$

## 2.4.5. Fréchet Inception Distance (FID) and Fréchet Reservoir Distance (FRD)

The Fréchet Inception Distance (FID) is estimated by computing the Fréchet distance between the distributions � and $g _ { \mathrm { : } }$ , obtained from the Inception network coding layer (Heusel et al., 2018):

$$
\mathrm { F I D } ( r , g ) = \left. \mu _ { r } - \mu _ { g } \right. _ { 2 } ^ { 2 } + T r \left( \mathbf { C } _ { r } + \mathbf { C } _ { g } - 2 \big ( \mathbf { C } _ { r } \mathbf { C } _ { g } \big ) ^ { \frac { 1 } { 2 } } \right)\tag{12}
$$

where $\mu$ and � denote the mean and covariance of a given set of samples, while $T r ( \cdot )$ calculates the trace of a matrix. For further information about Inception network and GAN metrics, reader is directed to Borji (2018). Although widely used in computer vision, computing the FID metric to assess the quality of generated reservoir realizations poses challenges, because the nature of the ImageNet dataset is very different from the geological images used in this work. For this reason, we replaced the Inception model with a Reservoir Classifier (RC) Network for computing the Fréchet Distance as proposed by (Ranazzi et al., 2024), calling this metric as the Fréchet Reservoir Distance (FRD). For both metrics, the statistics were computed over a large batch of 10,000 generated and training samples to avoid bias. For more information on the Reservoir Classifier (RC) and Fréchet Reservoir Distance (FRD) used in this work, the reader can find all the details in (Ranazzi et al., 2024).

## 2.5. Integration between Deep Learning Models and ESMDA

In this section, we present the integration framework between the generative deep learning models and ESMDA algorithm. Thus, we use generative models to learn lowdimensional and continuous data representation of the geological realizations, while ESMDA updates the corresponding latent vectors based on the available set of observed data. This workflow is accomplished through two main steps:

Step 1: Modeling and training deep learning models to learn the distribution of the dataset. After training, the weights of the generator (VAE-GAN and StyleGAN) and decoder (LDM) are saved for use in the next step;

Step 2: Employment of ESMDA to update the latent representations based on observed data. At each iteration, the ensemble of latent vectors � (or also � in the case of StyleGAN2) is passed through the respective generator or decoder to produce an ensemble of permeability realizations. These realizations are then fed into the simulator to compute the ensemble of ensemble of predicted data. The ESMDA is used to update the vector � of all models (and also � in the StyleGAN case) and the iterative process continues until the number of data assimilation iterations is reached.

## 3. CASE STUDIES

We evaluate the methodology by considering two distinct synthetic datasets: one case consisting of categorical variables (integer values) representing three facies, and the second one considering continuous variable based on the carbonate reservoir benchmark. The datasets were normalized to a range of -1 to 1, which is necessary for using the tanh (hyperbolic tangent) activation function in the output layer of models. All prior realizations, in each dataset, were generated using the same training image of the reference model.

## 3.1. Categorical training dataset

The realizations of the first dataset were built using the open-source Stanford Geostatistical Modeling Software – SGeMS (Remy et al., 2009), by applying the MPS algorithm Single Normal Simulation Equation – SNESIM (Strébelle, 2000). Using the “Stanford $\mathrm { V } ^ { \dag }$ three-facies training image from Remy et al. (2009, chap. 8), we have created 80,000 categorical realizations, with a rotation angle of $4 5 ^ { \mathrm { { \circ } } }$ in relation to the training image. In the images, shown here normalized for training, they have original permeability values equal to 100, 1,000 and 9,000 mD, for the colors purple, green and yellow, respectively. Figure 1 shows some unconditional realizations obtained by SNESIM algorithm. Henceforth, this dataset referred to as the ‘categorical training dataset’.

![](images/bd9d381d5e98ec06e59b7ea6899e285d0006810d2a4bb6923308a6aabd195aaf.jpg)  
Figure 1: Random realizations of the categorical training dataset.

This case study is a 2D model that contains 48 × 48 gridblocks with the reservoir logpermeability being the only uncertain model parameter $( N _ { m } = 2 , 3 0 4 )$ . The reference model was generated using the same approach as the training dataset. The reservoir simulation model contains 9 producers and 4 injectors with a configuration of four five-spots as we can see in Figure 2. The producers are controlled by minimum bottom-hole pressure (BHP) equal to 200 $\mathrm { k g f } / \mathrm { c m } ^ { 2 }$ and the injectors by maximum water injection rate (WIR) equal to $5 0 0 \mathrm { m } ^ { 3 } / \mathrm { d a y }$ . History data consists of noisy measurements at 90 days interval in 10 measurements periods $( N _ { d } =$ 660). The covariance matrix of the measurement errors was built considering a standard deviation of 4 m<sup>3</sup>/day for production rates, 3 m<sup>3</sup>/day for injector rates, and $2 \mathrm { k g f } / \mathrm { c m } ^ { 2 }$ for bottomhole pressures.

![](images/be087840803552b626ae4d304eba2d5685425086632b1ce82112bdf096a7c0bb.jpg)  
Figure 2: The true reservoir model for the first case. The locations of injectors are indicated by triangles, and those of producers by circles.

## 3.2. Continuous Training Dataset

To create the continuous training dataset in a more realistic context, we started with the UNISIM-II-H benchmark log-permeability realizations (Correia et al., 2015). The original 3D log-permeability field, which has dimensions of $4 6 \times 6 9 \times 3 0$ , is notably non-Gaussian due to the presence of Super-K features, thin layers with exceptionally high permeability (Meyer et al., 2000; Alqam et al., 2001). Following the approach of Ranazzi et al. (2024), we generated samples by taking multiple random 48 × 48 crops from each of the 30 vertical layers, resulting in a total of 15,000 samples (see Figure 3 for example of individual realizations). The reference model was randomly chosen from the original set of samples.

![](images/003819ae41057b1ee76c34f0ec26a5e54276ccb959795b16b70807dd0537975f.jpg)  
Figure 3: Random realizations of the continuous training dataset.  
The reservoir simulation model contains 9 producers and 4 injectors with a configuration of four five-spots. The well control strategy is identical to that adopted in the previous case study. History data consists of noisy measurements at 90 days interval in 10 different measurements periods $( N _ { d } = 6 6 0 )$ . The covariance matrix of the measurement errors was built considering a standard deviation of 4 m<sup>3</sup>/day for production rates, 3 m<sup>3</sup>/day for injector rates, and 2 kgf/cm<sup>2</sup> for bottom-hole pressures. The true reservoir model with well positions is shown in Figure 4.

![](images/bb9fb30ed9c5a9d78d69e3f2dcd98c2ad2bd7ec39770999456529401b929a3c4.jpg)  
Figure 4: The true reservoir model for the second case. The locations of injectors are indicated by triangles, and those of producers by circles.

## 3.3. VAE-GAN Configuration and Structure

The VAE‑GAN model was trained with a latent dimension of 512 for 150 epochs using a batch size of 32. The learning rate started at 0.0001 and followed an exponential decay schedule with a decay rate of 0.95 applied every 1,000 steps. The KL divergence loss was weighted by a beta parameter of 0.2, while the perceptual loss, computed using feature extractors based on InceptionV3 and a custom reservoir classifier, was weighted by a gamma of 0.1. The Leaky ReLU activation used an alpha slope of 0.2. The encoder consisted of three convolutional layers with 58, 116, and 230 filters, each followed by Leaky ReLU activation and batch normalization, plus a dropout rate of 0.3 before the dense layers that output the latent mean and log variance. The decoder mirrored this structure with three transposed convolutional layers using the same filter sizes in reverse order and a final tanh activation. The discriminator used three convolutional layers with the same filter progression, but incorporated spectral normalization, Gaussian noise with a standard deviation of 0.1, and dropout of 0.3. The total generator loss combined an L2 plus L1 reconstruction loss, the weighted KL divergence, the weighted perceptual loss, and an adversarial loss defined as the negative mean of the discriminator’s output on fake samples. The discriminator was trained with a hinge loss using separate terms for real and fake images. Both the generator and discriminator were optimized with Adam using a $\beta _ { 1 }$ of 0.5 and gradient clipping set to a maximum norm of 1.0. Early stopping with a patience of 50 epochs, monitoring the validation loss, was implemented to prevent overfitting. The Figure 5 below shows the schematic structure of the VAE-GAN model used in this work. The complete details of all models can be found in the link: https://github.com/LASG-USP/Parameterization\_StyleGAN.

![](images/22513e7b034bbbe2254c4a0a9e454299fb26965698546c57b6ab58c15753e884.jpg)  
Figure 5: Schematic structure of the VAE-GAN.

## 3.4. Latent Diffusion Configuration and Structure

The encoder compresses the input image to a latent representation, where the latent dimensionality was set to 512. The encoder architecture comprises four convolutional blocks with progressively increasing filter sizes (32, 64, 128, 256), each employing 3×3 kernels with stride 2 for downsampling, followed by Leaky ReLU activation with a slope of 0.2, batch normalization, and a dropout rate of 0.4. The final convolutional features are flattened and passed through a dense layer of size 256 before projection to the latent space (z-space). The decoder mirrors the encoder structure through transposed convolutions, reconstructing the image from the latent code. Two decoder variants were employed: (i) a discrete decoder with a softmax output activation over K=3 facies classes for categorical generation and (ii) a continuous decoder with a hyperbolic tangent (tanh) output activation for continuous-valued reconstruction. The latter enables direct discrete sampling by taking the argmax over class probabilities. The diffusion process operates directly on the 512-dimensional latent vectors. The denoising network follows a fully connected architecture conditioned on timestep embeddings. The timestep is encoded using a sinusoidal embedding of dimension 256, followed by two dense layers with Swish activation. This time embedding is added to the latent representation and passed through three hidden dense layers of sizes 1024, 1024, and 512, each with Leaky ReLU activation and a dropout rate of 0.1. The final layer projects back to the latent dimension. Training was conducted over 150 epochs using the Adam optimizer with an initial learning rate of $1 0 ^ { - 4 }$ and exponential decay rates $\beta _ { 1 } { = } 0 . 5$ and $\beta _ { 2 } { = } 0 . 9 9 9$ . Gradient clipping with a maximum norm of 1.0 was applied to stabilize training. A batch size of 32 was used for both training and validation. The composite loss function combined multiple objectives:

$$
L _ { t o t a l } = L _ { d i f f u s i o n } + \lambda _ { r e c o n } L _ { r e c o n } + \lambda _ { c l a s s } L _ { c l a s s }\tag{13}
$$

where $L _ { d i f f u s i o n }$ is the mean squared error between the true and predicted noise in the latent space, $L _ { r e c o n }$ is the pixel-wise MSE between the original and reconstructed images, and $L _ { c l a s s }$ is the sparse categorical cross-entropy for discrete facies prediction. The weighting coefficients were set to $\lambda _ { r e c o n } = 0 . 2$ and $\lambda _ { c l a s s } = 0 . 1$

Learning rate scheduling employed a strategy of reducing the learning rate by a factor of 0.5 if the validation loss did not improve for 5 consecutive epochs, down to a minimum of $1 0 ^ { - 6 }$ . Early stopping with a patience of 50 epochs, monitoring the validation loss, was implemented to prevent overfitting. The Figure 6 below shows the schematic structure of the LDM used in this work.

![](images/29aea489adfb247524326468700ce21ebc496ebe8b09c6694bf411da28a5b779.jpg)  
Figure 6: Schematic structure of the Latent Diffusion model.

## 3.5. StyleGAN Configuration

The generative model was trained using a Wasserstein GAN with gradient penalty (WGAN-GP) framework, with specific architectural and optimization parameters selected to ensure stable convergence and geostatistical fidelity. The generator employs a latent dimensionality of 512 (z-space), from which random vectors are drawn from a standard normal distribution. A mapping network composed of three dense layers with Leaky ReLU activations and dropout regularization transforms the latent vector (z-space) into an intermediate style representation (w-space). The synthesis network begins with a learned constant tensor of size 4×4, which is progressively upsampled through four stages using bilinear interpolation. Each upsampling stage doubles the spatial resolution, followed by convolutional layers and Gaussian-regularized adaptive instance normalization operations that inject style information. The final layer employs a convolutional operation with hyperbolic tangent activation to produce the output image. The generator applies L2 regularization with a coefficient of 0.01 throughout its layers. The discriminator processes input images through a series of five convolutional blocks with Leaky ReLU activations, dropout regularization, and average pooling operations for progressive downsampling. The architecture includes a dedicated feature extraction pathway that captures intermediate representations used for perceptual loss computation. The final classification head consists of fully connected layers that produce a scalar critic score. L2 regularization is similarly applied to all discriminator layers. Training was performed using the Adam optimizer with distinct learning rates: 0.0001 for the generator and 0.0004 for the discriminator, following the convention of training the discriminator more aggressively. The Adam momentum parameters are set to $\beta _ { 1 } = 0 . 5$ and $\beta _ { 2 } = 0 . 9$ . The batch size is configured to 32 samples. Training proceeds for 150 epochs with the batch order shuffled at each epoch. Label smoothing was applied with a factor of 0.1 to prevent discriminator overconfidence. Gradient clipping is enforced with a maximum norm of 0.5 for both generator and discriminator gradients. Gaussian noise with standard deviation of 0.05 is added to both real and generated images before discriminator evaluation, improving robustness. Latent Gaussian regularization is weighted by a factor of 0.1 in the generator loss, penalizing deviations from zero mean, unit variance, zero skewness, and zero excess kurtosis in the latent vectors. Feature matching loss from intermediate discriminator representations is incorporated with a weight of 0.1. Learning rate decay of 0.9 was applied when the discriminator loss fails to improve for five consecutive epochs. An adaptive scheduler may increase learning rates toward their initial values after ten epochs of stable training, where discriminator balance (real and fake outputs summing near zero) is maintained. Label smoothing was increased up to 0.3 in response to persistent discriminator overconfidence, and gradient clipping is halved when gradient norms exceed 100. We also applied early stopping with a patience of 50 to prevent overfitting. The Figure 7 below shows the schematic structure of the StyleGAN2 model used in this work.

![](images/92978b37b687a46096fd6fad4830333e7ccaf2bf266f3511d7a62cedee11c025.jpg)  
Figure 7: Schematic structure of the StyleGAN2 model.

## 4. RESULTS AND DISCUSSIONS

For each case study, we present the results of training of the models followed by the results of DA. All training process was performed in the same standard Nvidia GPU (GeForce RTX 3060 with 16 GB of memory) of a stand-alone computer, using the machine learning framework TensorFlow 2.10 (Abadi et al., 2015). In the DA step, we used 512 ensemble members and 32 iterations for ESMDA. In this way, the latent dimensions size was 512 for all models and cases.

## 4.1. Case Study 1: Categorical training dataset

## 4.1.1. Training of Models

## Variational Autoencoder Generative Adversarial Network (VAE-GAN)

The VAE-GAN model was trained until the early stopping criterion was reached, reaching the FID and FRD after the end of iterations, respectively equal to 323 and 3.5. The training time was 6 hours and 58 minutes, showing that the computational cost of training was the highest of all models, as it involves three models in the adversarial training. We can see in Figure 8 the high-quality of the images reconstructed by the decoder’s model.

Original  
![](images/1e98afd1c71bbfb65a3116381ff339ceb4439cdc48fb692175bbe6af28be6557.jpg)

Reconstructed  
![](images/394396fe740f1532202cdd5abd8664a366a761eb2e5afc9318428a16544321a5.jpg)

Original  
![](images/bd3b6f03c3283d6369fb1bee60a02514f6a6d47282324c63fbde045eb8ad5d46.jpg)  
Reconstructed

![](images/fc31d12db965c4e393f4cea3f61742bcc8e5ee087049fc655484ee3de0fc1c6b.jpg)

Original  
![](images/0b2b67da584c5d3206c980af5bf25a1b8070c3ace10ceb98021cc634ef9bdec8.jpg)  
Reconstructed

![](images/c86edde539cdbccdd9b6219b229a837f4ee53aaf4c519bbe06bd8514941ce63e.jpg)  
Figure 8: Three examples of random images from the set along with the reconstructed images by VAE-GAN for case study 1.

Figure 9 shows the total error for the training and the validation sets. This graph is the combination of all losses of this model: reconstruction, KL, generator, discriminator and perceptual losses. The curves show stabilization in errors after 60 epochs. We can observe initial instability until stability is reached, since it is necessary to train three networks at the same time.

![](images/4b63b99972f143cde3cc7840d595395dcddfde7e73ae4232678f8e4f96e7cd78.jpg)  
Figure 9: Total loss versus epochs with VAE-GAN in the categorical case.

## Latent Diffusion

We trained the Latent Diffusion model (LDM) after found the better structure and configuration using the categorical training dataset with all 72,000 samples for training and 8,000 samples for test until 150 epochs when stopped in the early stopping criterion. The training time was 3 hours and 9 minutes, less than half the time of VAE-GAN model. We obtained the FID and FRD after the end of iterations, respectively equal to 179 and 16.7, showing that the training was efficient. The quality of the images generated at the end of the training, as shown in Figure 10, with high quality, but with a certain softening of the contours.

![](images/b440cbb23ec7ef0b70689895cba159b0961ccc5c6553c9d3029f2d0f2f222e9b.jpg)  
Figure 10: Images reconstructed by LDM after training with 150 epochs for case study 1.

Figure 11 show the total diffusion loss throughout the training. Thus, we can see that in this case the training was successful.

![](images/f951dfdddcdbbc203076b1641ea63a188d650b7d3d32b919e621782d8f2f4016.jpg)  
Figure 11: MSE of training and validation versus epochs with LDM across training iterations.

## Style-Based Generative Adversarial Network (StyleGAN)

The StyleGAN model, version 2, was trained until the early stopping criterion was reached, reaching the FID and FRD after the end of iterations, respectively equal to 257 and 1.0. The training time was 5 hours and 5 minutes, showing that the computational cost of training

StyleGAN is higher than LDM, but less than VAE-GAN. We can see in Figure 12 the highquality of the images generated by the generator’s model.

Original  
![](images/806f5b611a0f3fd978f2a380548c4f6f753b1f8567d6f14f0c3688e059538910.jpg)

Generated  
![](images/1714641862076175f38606f0f94a83458e10f13174a4da97302280fce6c4e1c8.jpg)

![](images/c5c2a589eb90beec86168988959d44597009b132e886967c81f49c9f44925305.jpg)  
Original

Generated  
![](images/d1feaf134d0a762b92a6f998a4a5d4a3cd5c3965e9fd44b2215a0673ae500533.jpg)  
Original

![](images/cb92fedf9b7736059b449e987b60ba9f85f7612d8e67f08e59e10c8ea430825a.jpg)

Generated  
![](images/269ab1db0145762babd443513607cd49f74033874f7b4c6956fab0634782b380.jpg)  
Figure 12: Three examples of random images from the set along with the generated images by StyleGAN2 for case study 1.

Figure 13 shows the decrease in the generator loss and increase in the discriminator loss curves throughout the training. We can see that both losses tend to stabilize around 50 epochs. This shows that StyleGAN2 training was efficient and sufficient.

![](images/9623670dc62c4672ba04ce249dab9756de4b751c2a63ef36805347aee525a223.jpg)  
Figure 13: Graphs of generator and discriminator loss throughout the training with StyleGAN2 for case study 1.

In the Table 1, we summarize the results of the FID and FRD metrics at the end of training for the three models. We can conclude that the FRD metric was more appropriate than the FID for our case study and that the values obtained were low, demonstrating that the training of all models was efficient. It is interesting to note that the StyleGAN2 reach the lowest FRD in comparison to VAE-GAN and LDM, highlighting the high quality of the generated images compared to the other two models.

Table 1: Results of FID and FRD of models in the categorical case.
<table><tr><td></td><td>FID</td><td>FRD</td></tr><tr><td>VAE-GAN</td><td>323</td><td>3.5</td></tr><tr><td>Latent Diffusion</td><td>179</td><td>16.7</td></tr><tr><td>StyleGAN</td><td>257</td><td>1.0</td></tr></table>

To evaluate the quality of images generated by deep learning models in geological terms, we used main geostatistical metrics, such as: variogram (MSE), connectivity (MSE), histogram KL, PCA correlation and MDS MMD (Maximum Mean Discrepancy). In Table 2, we can see the results for static geostatistical metrics for the case study 1. Based on the comparative analysis of static geostatistical metrics, StyleGAN2 shows a clear overall advantage. For spatial continuity (Variogram MSE, Connectivity MSE), StyleGAN2 dominates decisively (0.00041 and 0.00056), being at least one order of magnitude better than VAE-GAN and LDM. It captures the spatial architecture of the categorical facies far more precisely. For distributional accuracy (Histogram KL divergence), this is the one metric where StyleGAN2 (0.77) does not lead. LDM excels with a very low score of 0.065, indicating it reproduces facies proportions most accurately in this case study. VAE-GAN scores slightly worse than StyleGAN2 with 0.89. Related to global structure (PCA Correlation, MDS MMD), StyleGAN2 ranks best, with an MDS MMD roughly half that of VAE-GAN. This confirms its realizations are most similar to the reference in terms of overall multivariate spatial arrangement. In summary, for the categorical case, StyleGAN2 delivers the best spatial pattern reproduction but struggles slightly with facies proportions, where LDM performs remarkably well. VAE-GAN remains consistently the weakest model across nearly all metrics.

Table 2: Results for static geostatistical metrics in the categorical case.
<table><tr><td></td><td>VAE-GAN</td><td>Latent Diffusion</td><td>StyleGAN</td></tr><tr><td>Variogram MSE</td><td>0.00953</td><td>0.03</td><td>0.00041</td></tr><tr><td>Connectivity MSE</td><td>0.01207</td><td>0.02533</td><td>0.00056</td></tr><tr><td>Histogram KL</td><td>0.88683</td><td>0.06541</td><td>0.76594</td></tr><tr><td>PCA Correlation</td><td>0.04801</td><td>0.05072</td><td>0.01134</td></tr><tr><td>MDS MMD</td><td>0.12060</td><td>0.15583</td><td>0.06335</td></tr></table>

## 4.1.2. Data Assimilation

For ESMDA, we updated directly the natural logarithm of permeability (log-permeability). For DA, we will compare the results from all models. However, for StyleGAN2, we will include assimilation using the z-space, as usual, and assimilation in the intermediate latent space, known as �-space. It is important to mention that the only modification to the StyleGAN assimilations was the shift from latent space z to w, without altering any model configurations or hyperparameters, with the training being exactly the same.

In Figure 14, we can see the images of the true case, the priori and posteriori mean and standard deviation. We also can see that the final posterior means did not result in extreme values, successfully preserving the geological realism inherent to the prior ensemble. Furthermore, an analysis of the final standard deviation maps shows that the spatial variability was preserved. As we know, in the ensemble-collapse scenario, the ensemble of models collapses to an almost deterministic solution, with posterior variances goes to near zero. We can observe that the VAE-GAN and LDM models were able to preserve the variance and mean values of the initial ensemble better than StyleGAN models for this case.

(d)  
![](images/15f73f082bcc897eaaad262b1a6e59d2163846fad04b252391b44f1130593034.jpg)  
Figure 14: Images of the true case, the priori and posteriori mean, and the priori and posteriori standard deviation in the categorical case for: (a) VAE-GAN, (b) LDM, (c) StyleGAN2 (zspace) and (d) StyleGAN2 (w-space).

Analyzing the time series of production data for P1 to P6 wells in Figure 15, it is possible to confirm that in all cases resulted in reductions in terms of ensemble spread. The improvement is demonstrated visually, which depicts the oil production rate (OPR) and water production rate (WPR) time series for the producer wells. While the initial ensemble exhibits a wide spread, the updated ensemble shows a significantly narrowed spread that closely tracks the observed historical data. We can also observe that all posterior models achieved a good match for all wells, for oil and water production as a function of time. Of particular note are the excellent results achieved by the VAE-GAN and LDM models using the latent space z, and by the StyleGAN2 model using the intermediate latent space w. The results were not quite as good for the StyleGAN2 model using latent space $\mathbf { Z } ,$ likely due to the inherent entanglement characteristic of this configuration.

![](images/3b9cbeceaa91ee1f6c6a3ffa721ae1eab5ad3704e65bbd54783edd30d78f5898.jpg)

![](images/0e1c80437560dcacc4e9015a720688a5649bce873abdb06a665ac00f4d40ba5b.jpg)

![](images/9fb616698b52791fb1c3e204281df45efcb6d1b961122f59ebb8f2cdbb03e1e4.jpg)

![](images/8943cfd028a411f17c6741eed7034d951b7f056308909a0b4ea3db7b51beed5d.jpg)

![](images/781846881a7f10a5fb52b0ad4d30d748d51cb02f2fa3cd618d1f22242e407fd3.jpg)

![](images/7e3f5225662a4e999fc7ac67306df709c49ef5284867fec1f7719e578136985b.jpg)

![](images/e079024abb779c94a642d63500f3c8ef9f6db50baaa3da80ddc8b0225ccf0b9b.jpg)

![](images/6bbbccbd2150dec347bfed90830cbf5958c849f0ada8844cede20d1b3ee0a384.jpg)

(a)  
![](images/4da49aaf36145480dba379f434b2cc6695d73592ef214ac0189fdf713ce16d9c.jpg)

![](images/8a6985c703d029a482fc79af4d9c6fb185b1252a18cb922197784babbbc4fe35.jpg)

![](images/7d79fb938f247d70b87525680d4cbd2fdbdf8da475fdd70c42d625adf8e4b857.jpg)

![](images/8bf1ee857570d50c9fd2346cd2361cbdc81f48666a61332d0822a76a7ef1bdf5.jpg)

![](images/e6ef38ae68e05abd0a0fc104dec5b20d54d11f550a0a12981f0d40eb772558ab.jpg)

![](images/9fcc6ccf85ed2c82d993e10e3be73749680c63d43233e5cb705a836256277f09.jpg)

![](images/f174d2c522317688bdc6f675a1e1fc6129e5f6293ca7520dc290f727ad9f73b8.jpg)

![](images/4bc7d84dfc1d52d8ab70e757fec9353c7b8550e15d5060dfa53ba3cf103bdfb5.jpg)

![](images/1f8a631390fd583144ae1238a708ef83b91eb0e5583c53f449ad13ff6c1fa7f7.jpg)  
(b)

![](images/31d0faab40f099e0a60e9c3474b1afc158c51810c98913d21ae07e398bb6d8ce.jpg)

![](images/ce81fbff0cf5854ae927736ca257528133d931c696aed3619e1575936c56e960.jpg)

![](images/5ec264e0493cdedf6d9c684dfc911eca1d0044c1410ea00802c493296fcf5a60.jpg)

![](images/e5426b1ed6ef2215261a87efd4199bee02099e9a05208b911c42aae862f6b4cd.jpg)

![](images/84dc185e583e198ebe3bf6b44e3e4548d5e2b7bc86813a395fded3c360702b43.jpg)

![](images/23826a50edd1e6d95d022df2edf8da78652ccf1a6dd6f5add75776c74b558f7f.jpg)

![](images/b04a6d8a1acbb5b2f11906a66be5c9b8abbaceeb327ddb38aa0bb4c63475e983.jpg)

![](images/fddac2f7b2a7491e2ca339d120208c0c1e62b79d4d5a1d613e670ba5d43057b2.jpg)

![](images/c6bffbaa2cb57dc0f2652f7cbdc26bd89d2044ad5903a55ad1debca8e8990d12.jpg)

![](images/b5d9a2633a4f6004e75bad238f3c7ffc0742cfb47d7d634bccd7ada1ac670a1e.jpg)

![](images/ae2fdd5c54839a314653c67c445fd126b65daa7a781bfda68c1876aefa287876.jpg)

![](images/c6ea6eccac52506f065180ea396f05058f47225999a415bc071c3173c74740b0.jpg)

![](images/be7bb3d23b5a079786b3f690a45d391c116ec130520b35ac2a80a676e37c543a.jpg)

(c)  
![](images/9072026eefbd5f0250d805b9364df4f5d659a908c3e1be53f4154ed887f2c320.jpg)

![](images/075794d76225a3f0c62d28fc4970698b021a12d20ef445b8505565614a484f00.jpg)

![](images/4614ab5962b28f154dc8025338c7d8ae0c96f4dbbcf72f3f68800a65d5ebfb21.jpg)

![](images/ae7ccc95d2be642a63a4fdc2b53fe14bfc0c017cb3e47d812877fc2cd8a347c4.jpg)

![](images/d8e7ce7c05aed6c1f9a7f116514274d2ee6a406187a05dc501a43a3ec4b9d488.jpg)

![](images/417c0e81bf485928eacdd5966603bbc4cca4bc4043151e1f12c9ca1e7058bc8a.jpg)

![](images/6674f2810340b6d855cb69485b02810be00b57c59c108de0754f4e11ea9d540a.jpg)

![](images/94a5506d4cc6f19c6c12a5a5787f1c19b373a919d7f46680b7e611be471e4b14.jpg)  
(d)

![](images/fd33b63179b34b1533558a770ef9887555d736fa89a402e7d834a5e29212b727.jpg)

![](images/6f1511846670bf2f8020c1114a3dd6549554613aad7ccfcaf79ec7f2de997b0d.jpg)  
Figure 15: Time series of production data from the first five producers of oil production rate (above) and water production rate (below). Here, the gray lines represent the prior ensemble, blue lines represent the posterior ensemble, and the red dots represent the measurements in the case study 1 for: (a) VAE-GAN, (b) LDM, (c) StyleGAN2 (z-space) and (d) StyleGAN2 (wspace).  
Figure 16 depicts the evolution of data mismatch, RMSE, spread and balanced accuracy across iterations. As we can see, all models achieved a satisfactory data match, as the results indicate a significant reduction in both data mismatch (DM) and RMSE, suggesting that the assimilation process effectively honored the observed data. Furthermore, VAE-GAN and LDM showed superior performance in terms of balanced accuracy compared to StyleGAN2.

![](images/7dbed5cdc978d86a7da25d886f17e2ca3ec179085d9a39abdea092e691005a84.jpg)

![](images/39dec37de8a4f9e338f08d1abb823cffe19e2648c24ef6ec61cb9a05a3a7712a.jpg)

![](images/cb95074e726e36182824b40ecb95ad1ca2078f83372f3b28caa8623bea8c4496.jpg)

![](images/a1a9e36a4448415804c5d4b42c04a7532430270726c4f1b270f38c38ef2b55fe.jpg)

![](images/cced9c78d884b50d485feff953c57892c0530d7311453b76c65097f7cf797662.jpg)

(a)  
![](images/6522ab0b1be75189353cae960ef1d19b30cd6e2a17b877912012d0d82ece7001.jpg)

![](images/9cb2601f629def6143faab36425e976ba5e05b99fe5d9268bff747cc2069bc55.jpg)  
(b)

![](images/2f6ee73421f7dc187c7b11cafa75d0e42423fbd06beb2200c67f9c76924c362b.jpg)

![](images/9b9468bef7684f5b2764bf5a93d09688ee9f7bfb596c3da85a19a65b6fca4b19.jpg)

![](images/92d5d81646f5762a6d307d656b839f8ad33a16effcf63536b07fb1257e390543.jpg)

![](images/5ee3288bcd292ddc8eee3bfb0a52ee3abc5b5bfaebc2f38ef46e848342eadd84.jpg)

![](images/8c6cff095309c181b7435105869de3eb90891877d4fe6a2474e1eea976d80c43.jpg)

![](images/8c77ae7c020340e914e5a4e9f6e1dd012b605bc30bd9e23e214d2e52dd2c7a84.jpg)

(c)  
![](images/d5d2d487023c8a8547ddf7bbed67c829a75142eb909944a761c7a5711d67f066.jpg)  
(d)

![](images/51e7ab2243b76469219f99d01da2dff724d3e555f5b3d19bd407b47e68d15456.jpg)

![](images/8a2629d35467d056c291d39ee7a00156aa3a1cda2842bef22e43302cf8dbbef3.jpg)  
Figure 16: Graphs of data mismatching, RMSE, spread and balanced accuracy as a function of the number of iterations in the categorical case for: (a) VAE-GAN, (b) Latent Diffusion, (c) StyleGAN2 (z-space) and (d) StyleGAN2 (w-space).

Figure 17 shows boxplots for assimilation with StyleGAN2 in the z and w spaces, displayed together to facilitate comparison. As can be seen, assimilation in the w-space outperformed that in the z-space, confirming that the w-space is less entangled than the z-space, which contributes to better ESMDA performance.

![](images/4597657c987f1eb88018a9e4f57bc367fd8b01be4c9fe0fc1143a5c7c51ee3ac.jpg)

![](images/7302e25fc8509a797a01d76af49419ce3709d8207dac9fdae12bb17501d20760.jpg)

![](images/b007f2d486b2f48c9de6dbba3fd1e44fa4aa6f2812571e53d560349d979cd8b5.jpg)

![](images/2b6b41b2cdaba375f8b7551316c177967ea987352c69af426da1597342550767.jpg)  
Figure 17: Comparison between graphs of data mismatching, RMSE, spread and balanced accuracy as a function of the number of iterations in the categorical case for StyleGAN2 (zspace) in black color and StyleGAN2 (w-space) in red color.

## 4.2. Case Study 2: Continuous training dataset

## 4.2.1. Training of Models

## Variational Autoencoder Generative Adversarial Network (VAE-GAN)

The VAE-GAN model was trained until the early stopping criterion was reached, achieving an FID of 286 and an FRD of 9.1 upon convergence. Total training time was 10 hours and 23 minutes. Figure 18 depicts the visual quality of the realizations reconstructed by the VAE-GAN model (generator), showing high-fidelity relative to the truth samples.

![](images/bda474eb8fdd16b865905f2c16bdb2071b905f3d1b20e6befa3d3637b5fd0a97.jpg)

![](images/bee84b6635294c8499e7513e09b04d16de2f3bd85bff8f26298e4013fa133a09.jpg)  
Original

![](images/8d271c953b5c852a5e25795f5fedee6777bc6d309415ba9c5bceb31484ca0296.jpg)  
Reconstructed

![](images/82c6485f62fc4d04a5df89ab76ab5633f419bcc963fd7552812f63f93c933b4b.jpg)  
Original

![](images/a26f340e7183c750afb671ea049b6297ad55440e7d20112a48a509aec4381de7.jpg)  
Reconstructed

![](images/626a5cb67c594aa9d1be95eff47445c10f4d2ffdfba9a8579f95ef16436dd796.jpg)  
Figure 18: Three examples of random images from the set along with the reconstructed images by VAE-GAN in the continuous case.

Figure 19 shows the total error curves for the training and the validation sets, considering the reconstruction KL, generator, discriminator, and perceptual loss components. The curves show stabilization in errors after 80 epochs.

![](images/51d0a5c642ef50647dbf96d6efe2892922fb97147da14e3e71d8eb913c467e03.jpg)  
Figure 19: Total loss versus epochs with VAE-GAN in the continuous case.

## Latent Diffusion

We trained the LDM with continuous case after identifying the optimal architectural configuration using the continuous training dataset with all 4,000 samples for training and 1,000 samples for testing until 133 epochs, reaching the early stopping criterion. The training time was only 28 minutes, representing a significant speedup compared to VAE-GAN model. Upon convergence, we obtained an FID and FRD equal to 215 and 31.3, respectively, confirming effective optimization. This can be seen by the high-quality of the images generated at the end of the training, as shown in Figure 20.

Original  
![](images/f812341244fa8815c76d4df0a2c7a9f9dd422aa394272627f9afe2420dbdbd9f.jpg)

Reconstructed  
![](images/eaa748b063ba7021d94397c20fc57b76778419d30fb9600f1d4b833be125da4b.jpg)

Original  
![](images/5505b3efd3934331fa65c863ccb65a3ac1eebc4576102e5f307ca056606ae038.jpg)

Reconstructed  
![](images/8f09e1ccddb88a28e27f37e4fb3609d87bd706c9e4fb9671dbaba97f4eebbb35.jpg)

Original  
![](images/ba25b7cb30c5a46c957018e229760663165d8aa1dce1d9711e3ae9ae74cc6a93.jpg)

Reconstructed  
![](images/8df1625061ce5fce61e651354f7e179a6d436d32d3a6004fbe0d626e242df992.jpg)  
Figure 20: Images generated by LDM after training after 133 epochs in the continuous case.

Figure 21 shows the total diffusion loss throughout the training. As we can see, training and validation curves have decreased over time. Thus, we can see that in this case the training also was successful.

![](images/912d9dba2d455f66db4ab644c9a474059da96648ec8b4716d3f86f43d0db9dfd.jpg)  
Figure 21: MSE of training and validation versus epochs with LDM across training iterations for case study 2.

## Style-Based Generative Adversarial Network (StyleGAN)

The StyleGAN2 model was trained until reach 106 epochs, when the early stopping criterion was reached. We obtained the FID and FRD after the end of iterations, respectively equal to 35 and 7.8. The training time was only 55 minutes, and therefore, much faster than the VAE-GAN model, but with almost twice the time of the LDM. We can see in Figure 22 the quality of the images generated by the generator’s network, showing images with high-quality, similar to other models.  
![](images/bc8b149554c4724595d3ec69d68d8caf0bb2ef5409e15ee5ae75fc499c4254a7.jpg)  
Figure 22: Three examples of random images using StyleGAN2 in the case study 2.

Figure 23 shows the MSE of the generator and discriminator. As we can see the fast decrease of generator loss and fast increase of discriminator throughout the training, showing that 106 epochs were sufficient to train the model.

![](images/dc3d82314f7252d5d74822828681115655e3aa52a5947f021d54baa816577fdd.jpg)  
Figure 23: MSE of generator and discriminator losses versus epochs with StyleGAN2 in the case study 2.

In Table 3, we summarize the results of the FID and FRD metrics at the end of training for the three models. We can conclude again that the FRD metric was more appropriate than the FID also for this case study and that the values obtained were low, demonstrating that the training of all models was efficient, with the VAE-GAN and StyleGAN2 models achieving the lowest values.

Table 3: Results of FID and FRD of models in the continuous case.
<table><tr><td></td><td>FID</td><td>FRD</td></tr><tr><td>VAE-GAN</td><td>286</td><td>9.1</td></tr><tr><td>Latent Diffusion</td><td>215</td><td>31.3</td></tr><tr><td>StyleGAN</td><td>35</td><td>7.8</td></tr></table>

In Table 4, based on the comparative analysis of static geostatistical metrics, StyleGAN2 consistently outperforms both VAE-GAN and LDM across all metrics, often by a significant margin. In the spatial continuity (Variogram MSE, Connectivity MSE), StyleGAN2 achieves the best scores (0.00044 and 0.00140), showing it captures two-point and multi-point spatial patterns far more accurately than the other models. Related to distributional accuracy (Histogram KL divergence), StyleGAN’s score of 0.52 is dramatically lower than VAE-GAN (3.90) and LDM (4.27), indicating it reproduces the univariate property distribution much more faithfully. About global structure (PCA Correlation, MDS MMD), StyleGAN2 again leads, with an MDS MMD of 0.132 compared to \~0.47 for the others, suggesting its generated realizations are much closer to the reference in a reduced dimensionality space. In summary, StyleGAN2 substantially surpasses VAE-GAN and LDM on all static geostatistical measures, while LDM generally performs the worst.

Table 4: Results for static geostatistical metrics for the continuous case.
<table><tr><td></td><td>VAE-GAN</td><td>Latent Diffusion</td><td>StyleGAN</td></tr><tr><td>Variogram MSE</td><td>0.00158</td><td>0.05001</td><td>0.00044</td></tr><tr><td>Connectivity MSE</td><td>0.04551</td><td>0.04234</td><td>0.00140</td></tr><tr><td>Histogram KL</td><td>3.89721</td><td>4.26662</td><td>0.52383</td></tr><tr><td>PCA Correlation</td><td>0.02318</td><td>0.02710</td><td>0.02029</td></tr><tr><td>MDS MMD</td><td>0.46454</td><td>0.47029</td><td>0.13239</td></tr></table>

## 4.2.2. Data Assimilation

Figure 24 show the images of the true case, the priori and posteriori mean and standard deviation. We can observe that the posteriori mean presents an image similar to the true case for all models, but with more similarity in the cases of StyleGAN2 (z and w-spaces).

![](images/c8d423756a2b05ce39c28f0622330d02efb8344dc71e338c3376e1400d06ab5b.jpg)

![](images/e07a96ee0559410d6c50a1483657100555d009d039fe126c570295140ea41558.jpg)

![](images/936c8c43221fc9860a447419f0fb2284b284163fe7ba37239f8d0a6a1f06b060.jpg)

![](images/17964324c62dede7de12a056e7e1a88fdf7aefe965aeee4166c4004cfe808623.jpg)  
(c)

![](images/3652517af2810c435eb5be193b3e04da8db91afeead7ba666409288b6bcc3910.jpg)

![](images/f2e632332564589e223b89d7585649a2c80a9403e8d4015e260182091bd5b9c7.jpg)

![](images/72131cfa6ba75b33a795d83a473229fb70cdc8c439491497795dbdd70c5dac12.jpg)

![](images/e5b4115e37316052bcf782a0b5df249ff9f49153d3aca3502cf57022ab62ede4.jpg)

![](images/aa4b25038aa0866cc9ec265cbf4f5a89ec5022a07425909581507c280b6c0f8c.jpg)  
(d)

![](images/0b77e56e4b61712ce008e4d59a8eb8fba6eb2f72e60e8f82b265306ab8b6488f.jpg)

![](images/605fc3cfbfbd19d9ea976e9cfd31da09c9685513646dc917ef686b9f9de9795d.jpg)  
Figure 24: Images of the true case, the priori and posteriori mean, and the priori and posteriori standard deviation in the case study 2 for: (a) VAE-GAN, (b) LDM, (c) StyleGAN2 (z-space), and (d) StyleGAN2 (w-space).

Evaluating the time series of production data in the Figure 25, we can see a better matching for the LDM and StyleGAN2 in the w-space than VAE-GAN and StyleGAN2 in the z-space. We can clearly see that the assimilation of StyleGAN2 using the intermediate latent space w was better than with the latent space z.  
![](images/88627ab28350164e20a36e9e187c7c9bbc397633359faa80e6c30fdff2b159c4.jpg)

![](images/092ac8733336534b2610425595394ad419dcce41f2b1ebc78f8600e380ea8f59.jpg)

![](images/15b67ebef2c6af91dd9843897649f18c9bfd3a52a99d62a1f5f7e8924fb7d754.jpg)

![](images/f539c4e9827f196fdf048ced4134f66eb137eff027297bff23e96b16b88dd5c0.jpg)

![](images/5c2dd30896624db6b83b83a7ab65b223907bcf7b4f4e47954d292782cb55fd15.jpg)

![](images/a3a5c018d21ec7ec606b3cc3af33c9439fa9d3fe6ccb4709ebee773342dd6e6b.jpg)

![](images/15cf477b5dd407506e1452cff46fe9fce4f091f08510f268cb31e46a0bbaaf4a.jpg)

![](images/1e5496716bed22fcf3b3940a7156484e7a1f9ecac1956ea6cc13212322dd1782.jpg)  
(a)

![](images/9846d91320d5dad3306fdec2102753b0448c9c9b84fd24012e8cde3cd02eaeb1.jpg)

![](images/2047bb17b6351e885fa0d2287b937b811d2e1a3ca4ebc96e86b71282ed24ae98.jpg)

![](images/04eb1fd26d6a436ffec99c0f7c4992d2f3e3da517dd25788484d6e634d8f2ca1.jpg)

![](images/2e001a426ce295303f33538f236f0e6e462f3bbe2c8ecb344fc4e8d9d99c4d08.jpg)

![](images/c0c6e390b57fd3221e0d045f3d80ec3c292324065157e01005d79092609cb0f8.jpg)

![](images/b2fd29aa2b1bc646279b9193a4f35af4a0276c50f5bb3fb0d17c3f86ce7b4e8b.jpg)

![](images/6ceeff1c6a7aa6a8d0f67b32b984bef13727978a174ec2db9f3ef715078c2325.jpg)

![](images/eb03fbed655b6bab919f9d74e8d892b7b2e39c989051e057f24a07cdab339575.jpg)

![](images/44fa941d0f6701dd5c55bec869a51f2250b118cbe0d970ae5ab3c5de702ed165.jpg)

![](images/349c2392b178bf40a06b4ec439cdd6b724b22a7ca3de40bf6cc3efeaf86ea4df.jpg)

![](images/d5075fde245722f5f0802e62a2a331329cf5d7e644482b171cff80d08eb88852.jpg)

![](images/0ce858f06750c633878fd5450ac1fefd201df15e3089d8c36e5132fcfdffadf3.jpg)

(b)  
![](images/8493ffc8c110a77d0985c410acea66c11c3a4cc046ef4829589f9d4491ea3e4a.jpg)

![](images/3c7db37609f880b26e47dc9d57e0a611dd2a0481c82f09374ff9e59df5a7d5f9.jpg)

![](images/88c8bae331649ec6aa832fd32cd280427e8363e4cbf99f556e57df0430e077e9.jpg)

![](images/d82348ae4aea2d40ee8e850cc1440dd39916ba1d54c72f883bdc49115bab371f.jpg)

![](images/95a58b6a7226f3c29b53bac0701766cc8039fe3865ef92c72e9b39aaf0eb1296.jpg)

![](images/ea5e0278407bec13b954d13db6188dbe74fa75eefe3d2a7711dc67d9faccf9ef.jpg)

![](images/f5549d24049e3d2fbc294885d99b7273512996147766b505a8619c6eb8bf42c3.jpg)

![](images/3fae8da273a270058054d7ca2aee84b6c899f5b01fa0a80c55f82701f814578b.jpg)

![](images/bfacf65e4a2efceac2f7d220f7d73f0d32dfd1dacb4b9f98466017cd4f8fd442.jpg)

![](images/63ec510fe100457d60b168e72ddcebc3d1b5e614f84e82fcbe961ea31782790a.jpg)

(c)  
![](images/7fe9f567888231543f895e4c156814d950cd253a2e2e176f468901a70183d313.jpg)

![](images/2c781cac910c6b1d23553e2b38f04de988f406fcf0bf71867e34bd9107eb7785.jpg)

![](images/35444c968c52e71ec51e94a94d56372e0c2a6807e5c65229b3dcf3abbd6b2140.jpg)

![](images/6464f8c8b16add783efc45e5524ec8876a6d37a0375c0816408b9e862484fc19.jpg)

![](images/cc1c3447f3f719d7a45967224c51eba93aea1654436a7955f58e1378286b4f25.jpg)

![](images/78b878dda0bfb0c250e36d1308606bde680fa7b2ba865914523eca7e676c64ab.jpg)

![](images/745c5580e0382eed77e9ff5fd24740c79d31b7a0497356a40728cfb76a9383c2.jpg)

![](images/46b9f6e14c3a1670edb3565524cde85af5df88dcb3b3030ecbc51902a162b4aa.jpg)

![](images/fe2fb18e856ed68d48141198376d7faf320bc54fe69bb3387b797029b9953ad8.jpg)  
(d)

![](images/d07d7cbf3a857c539f2f17c867061639073f0b8b418d9694219ebd9ed48acd5a.jpg)  
Figure 25: Time series of production data from the first five producers of oil production rate (above) and water production rate (below). Here, the gray lines represent the prior ensemble, blue lines represent the posterior ensemble, and the red dots represent the measurements in the case study 2 for: (a) VAE-GAN, (b) LDM, (c) StyleGAN2 (z-space) and (d) StyleGAN2 (wspace).

Figure 26 shows graphs of data mismatching, RMSE, spread as a function of the number of iterations. We can observe that LDM and StyleGAN2 in the w-space obtained the best results for RMSE and spread, followed by StyleGAN2 in the z-space and VAE-GAN, in this sequence.

![](images/c91d6b870dabbea4dd1b7776f5a5d56158981c684e3277d33c05dc026d5c5b27.jpg)

![](images/56f616f16fcdef9325c612f6ce7e602e386c55a144b3ec6c860e3068797ed2af.jpg)

![](images/49358fbe8322fea3b450c87852df374f20a187918eee3a35fa0a591a0d5d3722.jpg)

(a)  
![](images/9a926faf4ad46e38bfd94d2e526c9af47d3d73a79fecf5d8876467a0e332c6a2.jpg)

![](images/5e1c6643a65779d304c14911bd5837addc96733dd20e720d6581388303e6463a.jpg)  
(b)

![](images/32b4bab70b1df70bc68db40d81270ac11dcc75023f8e2067d68bf6c0804033fe.jpg)

![](images/9681ce1454ee8061860db17dd65b7c628c4bc029f29e44032d724df4cf2150a3.jpg)

![](images/c6227b96f793406790b7f3ee09a27690d2a9bdebe1290d2c5e521edc5f9825de.jpg)

![](images/bbd44a63b2c45d71df6748331e39d7441a62b9178f8e0d8cb6a36aeec51a9941.jpg)

(c)  
![](images/2936096d79411d1f86875bc27f4b4e4e0b3b4daaae253d8f1c9d9ae9ea734244.jpg)

![](images/3f9984ae79d59985892220afa31d95ca5824f97ea15180e1a9922a494566a373.jpg)  
(d)

![](images/1d3fea4c4bed4a2942d6f010b23d12ea470a7b09ba41654740cfcc58d4907a5a.jpg)  
Figure 26: Graphs of data mismatching, RMSE and spread as a function of the number of iterations for: (a) VAE-GAN, (b) LDM, (c) StyleGAN2 (z-space) and (d) StyleGAN2 (wspace).  
Figure 27 shows boxplots for assimilation with StyleGAN2 in the z and w spaces, displayed together. As we can see again, assimilation in the w-space outperformed that in the z-space, confirming that the w-space is less entangled than the z-space, which contributes to better DA efficiency.

![](images/51dc82a9d608a7d6a3057c98111603ca3496a0b34c8b337c56e9c1e359b54cb7.jpg)

![](images/69cacbf6c34ebf4e3d1886df111d7964f789f5c603f46cac6345daf7d033a523.jpg)

![](images/2fd552fe6e797abb499c9e66d5cb75c761e19d2d54108b81f4657916913b8ff1.jpg)  
Figure 27: Comparison between graphs of data mismatching, RMSE, spread as a function of the number of iterations in the continuous case for StyleGAN2 (z-space) in black color and StyleGAN2 (w-space) in red color.

## 5. CONCLUSIONS

This work shows that StyleGAN2 generates high-quality realizations with high training stability, driven by its disentangled latent space that is fundamental to a high performance of data assimilation using ESMDA. The results based on geostatistical metrics demonstrated the superior quality of the StyleGAN2 model in generating high-quality samples with geological realism. Our findings show that performing data assimilation with StyleGAN2 using the intermediate space (�-space) yielded better results than the traditional formulation which uses the latent space (z-space). This is due to the fact that ESMDA uses linear updates and the wspace is much more linear and disentangled than the highly entangled z-space. And since the w-space has already been mapped by the network, the updated vectors remain close to realistic geological patterns. Our results were compared against those of two models considered stateof-the-art in recent literature, VAE-GAN and LDM. The results showed that LDM has a lower computational cost than the other models, yielding an excellent matching, but producing samples with less geological realism than StyleGAN2. The results of DA were similar between LDM and StyleGAN2, particularly when the latter uses the intermediate latent space.

Future research could focus on evaluating the proposed approach on three-dimensional, large-scale reservoir models, as well as investigating techniques to mitigate spurious correlations (such as inflation and localization methods) when using these deep generative frameworks for parameterization in ensemble-based data assimilation.

## Acknowledgments

We gratefully acknowledge the support of Escola Politécnica of the University of São Paulo and Imperial. The authors would also like to thank the LASG (Laboratory of Reservoir Simulation and Management) for supporting this research, CMG (Computer Modelling Group Ltd.) for providing the reservoir simulator licenses used in this study and FAPESP – São Paulo Research Foundation (16/08801-0). We also thank the National Council for Scientific and Technological Development (CNPq) for financial support through grant number 310676/2025- 8.

## Conflicts of interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Computer Code Availability

The codes and the datasets generated and/or analyzed during the current study are available in the Github repository: https://github.com/LASG-USP/Parameterization\_StyleGAN.

## REFERENCES

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., Zheng, X., 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.

Aanonsen, S.I., Nævdal, G., Oliver, D. S., Reynolds, A.C.; Vallès, B. 2009. The Ensemble Kalman Filter in Reservoir Engineering—a Review, SPE J. 14 (03): 393–412. https://doi.org/10.2118/117274-PA

Ahn, S.; Choe, J. "Applications of Geological Features Style Mixing for Reservoir History Matching." SPE J. 30 (2025): 1651–1669. doi: https://doi.org/10.2118/224437-PA

Alqam, M.H., Nasr-El-Din, H.A., Lynn, J.D., 2001. Treatment of Super-K Zones Using Gelling Polymers, in: All Days. Presented at the SPE International Symposium on Oilfield Chemistry, SPE, Houston, Texas, p. SPE-64989-MS. https://doi.org/10.2118/64989- MS

Bao, J., Li, L., Davis, A., 2022. Variational Autoencoder or Generative Adversarial Networks? A Comparison of Two Deep Learning Methods for Flow and Transport Data Assimilation. Math. Geosci. 54, 1017–1042, 2022. https://doi.org/10.1007/s11004-022- 10003-3

Beucher, H., & Renard, D. (2016). Truncated Gaussian and derived methods. Comptes Rendus Geoscience, 348(7), 510–519. https://doi.org/10.1016/j.crte.2015.10.004

Borji, Ali. Pros and Cons of GAN Evaluation Measures. arXiv preprint, 2018. DOI: 10.48550/arXiv.1802.03446.

Canchumuni, S.A., Emerick, A.A., Pacheco, M.A., 2017. Integration of Ensemble Data Assimilation and Deep Learning for History Matching Facies Models, in: Day 1 Tue, October 24, 2017. Presented at the OTC Brasil, OTC, Rio de Janeiro, Brazil, p. D011S006R005. https://doi.org/10.4043/28015-MS

Canchumuni, S.W.A., Emerick, A.A., Pacheco, M.A.C., 2019a. History matching geological facies models based on ensemble smoother and deep generative models. J. Petrol. Sci. Eng. 177, 941–958. https://doi.org/10.1016/j.petrol.2019.02.037

Canchumuni, S.W.A., Emerick, A.A., Pacheco, M.A.C., 2019b. Towards a robust parameterization for conditioning facies models using deep variational autoencoders and ensemble smoother. Comput. Geosci. 128, 87-102. https://doi.org/10.1016/j.cageo.2019.04.006

Canchumuni, S.W.A., Castro, J.D.B., Potratz, J., Emerick, A.A., Pacheco, M.A.C., 2021. Recent developments combining ensemble smoother and deep generative networks for facies history matching. Comput. Geosci. 25, 433–466. https://doi.org/10.1007/s10596- 020-10015-0

Chang, H., Zhang, D., Lu, Z. 2010. Z. History matching of facies distribution with the EnKF and level set parameterization. Journal of Computational Physics, 229(20), 8011-8030. https://doi.org/10.1016/j.jcp.2010.07.005

Correia, M.., Hohendorff, J.., Gaspar, A.T., Schiozer, D.., 2015. UNISIM-II-D: Benchmark Case Proposal Based on a Carbonate Reservoir, in: Day 3 Fri, November 20, 2015. Presented at the SPE Latin American and Caribbean Petroleum Engineering Conference, SPE, Quito, Ecuador, p. D031S020R004. https://doi.org/10.2118/177140- MS

Emerick, A. A.; Reynolds, A. C., 2013. Ensemble smoother with multiple data assimilation. Computers & Geosciences, v. 55, p. 3–15. https://doi.org/10.1016/j.cageo.2012.03.011

Federico, G.; Durlofsky, L. J. Latent diffusion models for parameterization of facies-based geomodels and their use in data assimilation, Computers & Geosciences, Volume 194, 2025, 105755, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2024.105755.

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2014. Generative Adversarial Networks. https://doi.org/10.48550/arXiv.1406.2661

Hakim‐Elahi, S. & Jafarpour, B. (2017). A distance transform for continuous parameterization of discrete geologic facies for subsurface flow model calibration. Water Resources Research, 53(10), 8226–8249. https://doi.org/10.1002/2016WR019853

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S., 2018. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. https://doi.org/10.48550/arXiv.1706.08500

Ho, J., Jain, A., Abbeel, P., 2020. Denoising diffusion probabilistic models. In: Advances in Neural Information Processing Systems. Vol. 33, Curran Associates, Inc., pp.6840– 6851. http://dx.doi.org/10.48550/arXiv.2006.11239.

Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. arXiv. (arXiv:1710.10196 [cs, stat]) Retrieved from http://arxiv.org/abs/1710.10196

Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020a). Training generative adversarial networks with limited data. arXiv. (arXiv:2006.06676 [cs, stat]). Retrieved from http://arxiv.org/abs/2006.06676

Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J., & Aila, T. (2021). Alias‐free generative adversarial networks. arXiv. https://doi.org/10.48550/ARXIV.2106.12423

Karras, T., Laine, S., & Aila, T. (2019). A style‐based generator architecture for generative adversarial networks. arXiv. (arXiv:1812.04948[cs, stat]). Retrieved from http://arxiv.org/abs/1812.04948

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020b). Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8110–8119).

Kullback, S., Leibler, R.A., 1951. On information and sufficiency. Ann. Math. Stat. 22, 79–86. https://doi.org/10.1214/aoms/1177729694.

Laloy, E., Hérault, R., Jacques, D., Linde, N., 2017. Inversion using a new lowdimensional representation of complex binary geological media based on a deep neural network. Adv. Water Resour. 110, 387–405. https://doi.org/10.1016/j.advwatres.2017.09.029.

Laloy, E., Héerault, R., Jacques, D., Linde, N., 2018. Training-image based geostatistical inversion using a spatial generative adversarial neural network. Water Resour. Res.54 (1), 381–406. https://doi.org/10.1002/2017WR022148.

Li, L., Stetler, L., Cao, Z., Davis, A., 2018. An iterative normal-score ensemble smoother for dealing with non-Gaussianity in data assimilation. Journal of Hydrology 567, 759–766. https://doi.org/10.1016/j.jhydrol.2018.01.038

Ling, W., & Jafarpour, B. (2024). Improving the parameterization of complex subsurface flow properties with style‐based generative adversarial network (StyleGAN). Water Resources Research, 60, e2024WR037630. https://doi.org/10.1029/2024WR037630

Liu, Y.; Durlofsky, L. J. 3D CNN-PCA: A deep-learning-based parameterization for complex geomodels, Computers & Geosciences, Volume 148, 2021, 104676, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2020.104676.

Luo, Z., Tong, L., Wang, M. Y., & Wang, S. (2007). Shape and topology optimization of compliant mechanisms using a parameterization level set method. Journal of Computational Physics, 227(1), 680–705. https://doi.org/10.1016/j.jcp.2007.08.011

Luo, Z., Wang, M. Y., Wang, S., & Wei, P. (2008). A level set‐based parameterization method for structural shape and topology optimization. International Journal for Numerical Methods in Engineering, 76(1), 1–26. https://doi.org/10.1002/nme.2092

Meyer, F.O., Price, R.C., Al-Raimi, S.M., 2000. Stratigraphic and Petrophysical Characteristics of Cored Arab-D Super-k Intervals, Hawiyah Area, Ghawar Field, Saudi Arabia. GeoArabia 5, 355–384. https://doi.org/10.2113/geoarabia0503355

Oliver, D. S.; Chen, Y., 2011. Recent progress on reservoir history matching: a review. Computational Geosciences, v. 15, n. 1, p. 185–221. https://doi.org/10.1007/s10596- 010-9194-2

Ranazzi, P. H.; Luo, X.; Sampaio, M. A., 2024. Improving the training performance of Generative Adversarial Networks with limited data: application to the generation of geological models. Computers & Geosciences, v. 193, p. 105747.

Remy, N., Boucher, A., Wu, J., 2009. Applied Geostatistics with SGeMS: A User’s Guide, 1st ed. Cambridge University Press. https://doi.org/10.1017/CBO9781139150019

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: convolutional networks for biomed-ical image segmentation. In: Medical Image Computing and Computer-AssistedIntervention - MICCAI 2015 - 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III. In: Lecture Notes in Computer Science, vol. 9351, Springer, pp. 234–241.http://dx.doi.org/10.1007/978-3-319-24574-4\_28.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVFConference on Computer Vision and Pattern Recognition. CVPR, pp. 10684–10695. http://dx.doi.org/10.48550/arXiv.2112.10752.

Sampaio, M. A.; Ranazzi, P. H.; Blunt, M. J.; Enhancing the parameterization of reservoir properties for data assimilation using deep VAE-GAN, Computers & Geosciences, Volume 214, 2026, 106196, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2026.106196.

Silva, D. S. F., & Deutsch, C. V. (2017). Multiple imputation framework for data assignment in truncated pluri‐Gaussian simulation. Stochastic Environmental Research and Risk Assessment, 31(9), 2251–2263. https://doi.org/10.1007/s00477‐016‐1309‐4

Strébelle, S., 2000. Sequential simulation drawing structures from training images. Stanford, Stanford, CA, USA.